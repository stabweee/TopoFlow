"""OpenStreetMap highway vector geometry ingestion and directional carriageway assembly.

Extracts high-fidelity road geometries for study and control corridors using Overpass API,
stitches fragmented OSM way segments into continuous directional centerlines, extracts
metadata (lanes, bridge status, speed limits), and exports to GeoJSON and Parquet.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import pyproj
import requests
from shapely.geometry import LineString

from topoflow.config import CorridorConfig, CorridorsConfig, Settings, get_settings
from topoflow.constants import CRS_METRIC, CRS_WGS84

logger = logging.getLogger(__name__)


class OSMExtractorError(Exception):
    """Base exception for OSM extraction errors."""


class OSMNetworkError(OSMExtractorError):
    """Network-level error or repeated failure connecting to Overpass API."""


class OSMQueryError(OSMExtractorError):
    """Error in Overpass query syntax or malformed response payload."""


class OSMTopologicalError(OSMExtractorError):
    """Topological error stitching disjoint or fragmented road segments."""


def build_overpass_query(corridor: CorridorConfig, timeout: int = 30) -> str:
    """Construct an Overpass QL query string filtering motorways within a corridor bounding box.

    Args:
        corridor: CorridorConfig specifying bounding box and route identifier.
        timeout: Execution timeout in seconds requested from Overpass server.

    Returns:
        Formatted Overpass QL query string.
    """
    bbox = corridor.bounding_box
    route = corridor.route

    # Match common route representations in OSM (e.g., 'I 495', 'I-495', or highway name)
    if "495" in route:
        ref_regex = "I 495|I-495|Capital Beltway"
    elif "95" in route:
        ref_regex = "I 95|I-95"
    elif "295" in route:
        ref_regex = "I 295|I-295|Anacostia Freeway"
    else:
        ref_regex = f"{route}|{route.replace('-', ' ')}"

    query = (
        f"[out:json][timeout:{timeout}];\n"
        f"(\n"
        f'  way["highway"="motorway"]["ref"~"{ref_regex}"]'
        f"({bbox.min_lat},{bbox.min_lon},{bbox.max_lat},{bbox.max_lon});\n"
        f");\n"
        f"out body;\n"
        f">;\n"
        f"out skel qt;\n"
    )
    return query


def fetch_osm_data(
    query: str,
    endpoint: str | None = None,
    timeout: int = 30,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Execute an Overpass QL query with exponential backoff on HTTP 429 and 504.

    Args:
        query: Overpass QL query string.
        endpoint: Target Overpass API interpreter URL (defaults to Settings.overpass_url).
        timeout: HTTP request timeout in seconds.
        max_retries: Number of retry attempts before raising OSMNetworkError.
        backoff_factor: Multiplier for exponential backoff sleep interval.
        session: Optional pre-configured requests.Session.

    Returns:
        Parsed JSON dictionary from Overpass response.

    Raises:
        OSMNetworkError: If retries are exhausted or a network error occurs.
        OSMQueryError: If Overpass returns an error status code or invalid JSON.
    """
    settings = get_settings()
    url = endpoint or settings.overpass_url
    client = session or requests.Session()

    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(
                "Executing Overpass query (attempt %d/%d) against %s", attempt, max_retries, url
            )
            response = client.post(url, data={"data": query}, timeout=timeout)

            if response.status_code == 200:
                try:
                    data = response.json()
                    if "elements" not in data:
                        raise OSMQueryError(
                            f"Malformed Overpass response missing 'elements': {data}"
                        )
                    return data
                except ValueError as err:
                    raise OSMQueryError(
                        f"Failed to parse Overpass response as JSON: {err}"
                    ) from err

            # Handle rate-limiting (429) or gateway timeout (504)
            if response.status_code in (429, 502, 503, 504):
                logger.warning(
                    "Overpass HTTP %d on attempt %d/%d. Waiting %.1fs before retrying...",
                    response.status_code,
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                delay *= backoff_factor
                continue

            # Unhandled HTTP error status
            raise OSMQueryError(
                f"Overpass query failed with HTTP {response.status_code}: {response.text[:200]}"
            )

        except requests.RequestException as exc:
            logger.warning(
                "Network exception during Overpass call (%s) on attempt %d/%d. Retrying...",
                exc,
                attempt,
                max_retries,
            )
            if attempt == max_retries:
                raise OSMNetworkError(
                    f"Exhausted {max_retries} retries connecting to Overpass API: {exc}"
                ) from exc
            time.sleep(delay)
            delay *= backoff_factor

    raise OSMNetworkError(f"Failed to fetch Overpass data after {max_retries} attempts.")


def parse_osm_elements(
    data: dict[str, Any],
) -> tuple[dict[int, tuple[float, float]], list[dict[str, Any]]]:
    """Parse raw Overpass JSON elements into node coordinate map and way dictionaries.

    Args:
        data: Overpass JSON response.

    Returns:
        Tuple of (node_map: {node_id: (lon, lat)}, ways: list of way dictionaries).
    """
    nodes: dict[int, tuple[float, float]] = {}
    ways: list[dict[str, Any]] = []

    for elem in data.get("elements", []):
        elem_type = elem.get("type")
        if elem_type == "node":
            nodes[elem["id"]] = (float(elem["lon"]), float(elem["lat"]))
        elif elem_type == "way":
            ways.append(elem)

    return nodes, ways


def _classify_direction(
    way: dict[str, Any],
    nodes: dict[int, tuple[float, float]],
    corridor: CorridorConfig,
) -> str | None:
    """Classify an OSM way into a corridor direction (e.g. inner_loop vs outer_loop)."""
    tags = way.get("tags", {})
    name = tags.get("name", "").lower()
    ref = tags.get("ref", "").lower()
    way_nodes = [nodes[nid] for nid in way.get("nodes", []) if nid in nodes]

    if len(way_nodes) < 2:
        return None

    # First check explicit naming in tags
    if "inner loop" in name or "inner loop" in ref:
        return "inner_loop"
    if "outer loop" in name or "outer loop" in ref:
        return "outer_loop"
    if "north" in name or "northbound" in name:
        return "northbound"
    if "south" in name or "southbound" in name:
        return "southbound"

    # Compute overall latitude displacement vector
    dlat = way_nodes[-1][1] - way_nodes[0][1]

    if "inner_loop" in corridor.directions and "outer_loop" in corridor.directions:
        # For ALMB (VA -> MD), Inner Loop heads northeast (dlat > 0), Outer Loop heads southwest (dlat < 0)
        return "inner_loop" if dlat >= 0 else "outer_loop"

    if "northbound" in corridor.directions and "southbound" in corridor.directions:
        return "northbound" if dlat >= 0 else "southbound"

    return corridor.directions[0] if corridor.directions else None


def _stitch_segments_to_linestring(
    segments: list[list[tuple[float, float]]],
    target_direction: str,
    max_gap_degrees: float = 0.005,
) -> LineString:
    """Stitch an unordered collection of coordinate chains into a single continuous LineString.

    Handles way reversal and chain connection. Validates that resulting LineString has
    zero breaks and is oriented along the intended direction of travel.

    Args:
        segments: List of coordinate lists [(lon, lat), ...].
        target_direction: Desired direction ('inner_loop', 'outer_loop', 'northbound', 'southbound').
        max_gap_degrees: Maximum allowable distance gap between connecting segment endpoints (~500m).

    Returns:
        Continuous Shapely LineString.

    Raises:
        OSMTopologicalError: If segments cannot be chained or result in zero-length geometry.
    """
    if not segments:
        raise OSMTopologicalError(
            f"No coordinate segments provided for direction '{target_direction}'"
        )

    if len(segments) == 1:
        coords = segments[0]
        if len(coords) < 2:
            raise OSMTopologicalError("Segment contains fewer than 2 coordinates.")
        line = LineString(coords)
        return _ensure_directional_bearing(line, target_direction)

    # Greedily connect segments matching endpoints
    remaining = [list(seg) for seg in segments]
    stitched: list[tuple[float, float]] = remaining.pop(0)

    while remaining:
        head = stitched[0]
        tail = stitched[-1]

        best_match_idx = -1
        best_mode: str | None = None
        min_dist = float("inf")

        for idx, candidate in enumerate(remaining):
            cand_head = candidate[0]
            cand_tail = candidate[-1]

            # Distances: tail-to-head, tail-to-tail, head-to-tail, head-to-head
            d_tail_head = (tail[0] - cand_head[0]) ** 2 + (tail[1] - cand_head[1]) ** 2
            d_tail_tail = (tail[0] - cand_tail[0]) ** 2 + (tail[1] - cand_tail[1]) ** 2
            d_head_tail = (head[0] - cand_tail[0]) ** 2 + (head[1] - cand_tail[1]) ** 2
            d_head_head = (head[0] - cand_head[0]) ** 2 + (head[1] - cand_head[1]) ** 2

            local_min = min(d_tail_head, d_tail_tail, d_head_tail, d_head_head)
            if local_min < min_dist:
                min_dist = local_min
                best_match_idx = idx
                if local_min == d_tail_head:
                    best_mode = "append_normal"
                elif local_min == d_tail_tail:
                    best_mode = "append_reverse"
                elif local_min == d_head_tail:
                    best_mode = "prepend_normal"
                else:
                    best_mode = "prepend_reverse"

        if best_match_idx < 0:
            raise OSMTopologicalError(
                f"Failed to find connecting segment in direction '{target_direction}'"
            )

        if min_dist > (max_gap_degrees**2):
            raise OSMTopologicalError(
                f"Topological gap ({min_dist**0.5:.5f} deg) exceeds maximum tolerance "
                f"({max_gap_degrees} deg) in direction '{target_direction}'"
            )

        next_seg = remaining.pop(best_match_idx)
        if best_mode == "append_normal":
            stitched.extend(next_seg[1:])
        elif best_mode == "append_reverse":
            stitched.extend(list(reversed(next_seg))[1:])
        elif best_mode == "prepend_normal":
            stitched = next_seg[:-1] + stitched
        elif best_mode == "prepend_reverse":
            stitched = list(reversed(next_seg))[:-1] + stitched

    # Remove consecutive duplicate vertices
    cleaned: list[tuple[float, float]] = []
    for pt in stitched:
        if not cleaned or pt != cleaned[-1]:
            cleaned.append(pt)

    if len(cleaned) < 2:
        raise OSMTopologicalError("Stitched road line collapsed into fewer than 2 distinct points.")

    line = LineString(cleaned)
    return _ensure_directional_bearing(line, target_direction)


def _ensure_directional_bearing(line: LineString, direction: str) -> LineString:
    """Ensure LineString coordinate order points in the physical direction of traffic flow."""
    coords = list(line.coords)
    dlat = coords[-1][1] - coords[0][1]

    # Inner loop (VA -> MD) and Northbound travel in direction of increasing latitude
    if direction in ("inner_loop", "northbound") and dlat < 0:
        return LineString(list(reversed(coords)))

    # Outer loop (MD -> VA) and Southbound travel in direction of decreasing latitude
    if direction in ("outer_loop", "southbound") and dlat > 0:
        return LineString(list(reversed(coords)))

    return line


def calculate_metric_length(line: LineString) -> float:
    """Compute the geodesic length of a WGS84 LineString in meters projected to UTM 18N (EPSG:26918)."""
    transformer = pyproj.Transformer.from_crs(CRS_WGS84, CRS_METRIC, always_xy=True)
    projected_coords = [transformer.transform(x, y) for x, y in line.coords]
    projected_line = LineString(projected_coords)
    return float(projected_line.length)


class OSMExtractor:
    """Extraction engine for highway vector geometries from OpenStreetMap."""

    def __init__(
        self,
        settings: Settings | None = None,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.settings = settings or get_settings()
        self.timeout = timeout
        self.max_retries = max_retries

    def extract_corridor(
        self,
        corridor: CorridorConfig,
        raw_data: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> gpd.GeoDataFrame:
        """Extract and assemble directional centerlines for a single highway corridor.

        Args:
            corridor: Target CorridorConfig instance.
            raw_data: Optional pre-fetched Overpass JSON data.
            use_cache: If True, check local raw cache before querying remote API.

        Returns:
            GeoDataFrame containing assembled directional centerlines with metadata.
        """
        cache_path = self.settings.raw_data_dir / "geo" / f"{corridor.corridor_id}_osm_raw.json"

        # Check local disk cache
        if raw_data is None and use_cache and cache_path.exists():
            logger.info(
                "Loading cached Overpass response for '%s' from %s",
                corridor.corridor_id,
                cache_path,
            )
            with open(cache_path, encoding="utf-8") as f:
                raw_data = json.load(f)

        # Fetch from remote Overpass API if not cached or passed
        if raw_data is None:
            query = build_overpass_query(corridor, timeout=self.timeout)
            logger.info(
                "Fetching OSM road geometry for '%s' via Overpass API...", corridor.corridor_id
            )
            raw_data = fetch_osm_data(
                query=query,
                endpoint=self.settings.overpass_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            # Write cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f)

        nodes, ways = parse_osm_elements(raw_data)
        if not ways:
            raise OSMTopologicalError(
                f"No highway ways found for corridor '{corridor.corridor_id}'"
            )

        # Group ways by direction
        directional_ways: dict[str, list[dict[str, Any]]] = {d: [] for d in corridor.directions}
        for way in ways:
            direction = _classify_direction(way, nodes, corridor)
            if direction and direction in directional_ways:
                directional_ways[direction].append(way)

        rows: list[dict[str, Any]] = []

        for direction, dir_ways in directional_ways.items():
            if not dir_ways:
                logger.warning(
                    "No ways identified for direction '%s' in corridor '%s'",
                    direction,
                    corridor.corridor_id,
                )
                continue

            segments: list[list[tuple[float, float]]] = []
            has_bridge = False
            total_lanes = []
            speed_limits = []

            for way in dir_ways:
                tags = way.get("tags", {})
                if tags.get("bridge") in ("yes", "viaduct"):
                    has_bridge = True
                lane_val = tags.get("lanes")
                if lane_val and lane_val.isdigit():
                    total_lanes.append(int(lane_val))
                maxspeed_val = tags.get("maxspeed")
                if maxspeed_val:
                    speed_limits.append(str(maxspeed_val))

                coords = [nodes[nid] for nid in way.get("nodes", []) if nid in nodes]
                if len(coords) >= 2:
                    segments.append(coords)

            stitched_line = _stitch_segments_to_linestring(segments, direction)
            metric_len = calculate_metric_length(stitched_line)
            dominant_lanes = round(sum(total_lanes) / len(total_lanes)) if total_lanes else 4
            dominant_speed = speed_limits[0] if speed_limits else "55 mph"

            rows.append(
                {
                    "corridor_id": corridor.corridor_id,
                    "direction": direction,
                    "route": corridor.route,
                    "name": corridor.name,
                    "lanes": dominant_lanes,
                    "maxspeed": dominant_speed,
                    "has_bridge": has_bridge,
                    "length_meters": round(metric_len, 2),
                    "is_continuous": True,
                    "geometry": stitched_line,
                }
            )

        if not rows:
            raise OSMTopologicalError(
                f"Failed to assemble any continuous centerlines for corridor '{corridor.corridor_id}'"
            )

        gdf = gpd.GeoDataFrame(rows, crs=CRS_WGS84)
        return gdf

    def extract_all(
        self,
        corridors_config: CorridorsConfig | None = None,
        use_cache: bool = True,
    ) -> gpd.GeoDataFrame:
        """Extract and assemble road geometries for all study and control corridors.

        Args:
            corridors_config: Optional CorridorsConfig; loads default if omitted.
            use_cache: Whether to use cached raw Overpass responses if present.

        Returns:
            Unified GeoDataFrame containing all corridor directional centerlines.
        """
        config = corridors_config or self.settings.load_corridors()
        frames: list[gpd.GeoDataFrame] = []

        for corridor in config.corridors.values():
            gdf = self.extract_corridor(corridor, use_cache=use_cache)
            frames.append(gdf)

        if not frames:
            raise OSMTopologicalError("No corridors were successfully extracted.")

        unified_gdf = gpd.GeoDataFrame(
            pd_concat(frames, ignore_index=True),
            crs=CRS_WGS84,
        )
        return unified_gdf

    def export(
        self,
        gdf: gpd.GeoDataFrame,
        output_dir: Path | None = None,
        stem: str = "osm_corridors",
    ) -> tuple[Path, Path]:
        """Export GeoDataFrame to both GeoJSON and Parquet formats in EPSG:4326.

        Args:
            gdf: GeoDataFrame to serialize.
            output_dir: Output directory (defaults to data/raw/geo).
            stem: Base file name without extension.

        Returns:
            Tuple of (geojson_path, parquet_path).
        """
        out_path = output_dir or (self.settings.raw_data_dir / "geo")
        out_path.mkdir(parents=True, exist_ok=True)

        geojson_path = out_path / f"{stem}.geojson"
        parquet_path = out_path / f"{stem}.parquet"

        # Ensure CRS is strictly WGS84 for ingestion/export contracts
        export_gdf = gdf.to_crs(CRS_WGS84) if gdf.crs != CRS_WGS84 else gdf

        # Export GeoJSON
        export_gdf.to_file(geojson_path, driver="GeoJSON")

        # Export Parquet
        export_gdf.to_parquet(parquet_path)

        logger.info("Exported %d corridors to %s and %s", len(gdf), geojson_path, parquet_path)
        return geojson_path, parquet_path

    @staticmethod
    def load_from_fixture(fixture_path: Path) -> gpd.GeoDataFrame:
        """Load pre-stitched corridor geometries from an offline GeoJSON fixture."""
        if not fixture_path.exists():
            raise FileNotFoundError(f"Fixture file not found at {fixture_path}")
        return gpd.read_file(fixture_path)


def pd_concat(dfs: list[gpd.GeoDataFrame], ignore_index: bool = True) -> gpd.GeoDataFrame:
    """Concatenate a list of GeoDataFrames into a single GeoDataFrame."""
    import pandas as pd

    concatenated = pd.concat(dfs, ignore_index=ignore_index)
    return gpd.GeoDataFrame(concatenated, crs=CRS_WGS84)
