"""Unit tests for OpenStreetMap highway vector extraction and carriageway assembly.

Validates Overpass QL query generation, element parsing, directional classification,
continuous LineString stitching, bridge tagging, network retry/backoff, and serialization.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
import requests
from pyproj import Transformer
from shapely.geometry import LineString

from topoflow.config import BoundingBox, Centroid, CorridorConfig, Settings
from topoflow.constants import CONTINUITY_TOLERANCE_METERS, CRS_METRIC, CRS_WGS84
from topoflow.geo.discretization import SpatialDiscretizer
from topoflow.geo.osm_extractor import (
    OSMExtractor,
    OSMNetworkError,
    OSMQueryError,
    OSMTopologicalError,
    _classify_direction,
    _is_deflection_acceptable,
    _is_managed_or_express_lane,
    _stitch_segments_to_linestring,
    build_overpass_query,
    calculate_metric_length,
    compute_way_intervals,
    fetch_osm_data,
    parse_osm_elements,
    stitch_segments_with_gap,
)


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to tests/fixtures directory."""
    return Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture
def alb_corridor() -> CorridorConfig:
    """Return study corridor configuration for American Legion Bridge."""
    return CorridorConfig(
        corridor_id="alb_i495",
        name="I-495 American Legion Memorial Bridge",
        corridor_type="study",
        route="I-495",
        state="VA/MD",
        bounding_box=BoundingBox(
            min_lat=38.95,
            max_lat=39.00,
            min_lon=-77.23,
            max_lon=-77.18,
        ),
        centroid=Centroid(lat=38.9697, lon=-77.2092),
        directions=["inner_loop", "outer_loop"],
    )


@pytest.fixture
def i95_corridor() -> CorridorConfig:
    """Return control corridor configuration for I-95."""
    return CorridorConfig(
        corridor_id="i95_lorton_woodbridge",
        name="I-95 Flat Tangent Control Corridor",
        corridor_type="control",
        route="I-95",
        state="VA",
        bounding_box=BoundingBox(
            min_lat=38.64,
            max_lat=38.72,
            min_lon=-77.26,
            max_lon=-77.22,
        ),
        centroid=Centroid(lat=38.6800, lon=-77.2400),
        directions=["northbound", "southbound"],
    )


@pytest.fixture
def raw_alb_json(fixtures_dir: Path) -> dict:
    """Load raw Overpass mock JSON for ALB corridor."""
    path = fixtures_dir / "osm_alb_response.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestOverpassQueryGeneration:
    """Test formulation of Overpass QL queries."""

    def test_query_generation_alb_i495(self, alb_corridor: CorridorConfig) -> None:
        query = build_overpass_query(alb_corridor, timeout=25)
        assert "[out:json][timeout:25];" in query
        assert 'way["highway"="motorway"]["ref"~"I 495|I-495|Capital Beltway"]' in query
        assert "(38.95,-77.23,39.0,-77.18)" in query
        assert "out body;" in query
        assert ">;" in query
        assert "out skel qt;" in query

    def test_query_generation_i95(self, i95_corridor: CorridorConfig) -> None:
        query = build_overpass_query(i95_corridor, timeout=30)
        assert 'way["highway"="motorway"]["ref"~"I 95|I-95"]' in query
        assert "(38.64,-77.26,38.72,-77.22)" in query


class TestOSMElementParsing:
    """Test parsing of raw Overpass JSON elements."""

    def test_parse_elements_valid(self, raw_alb_json: dict) -> None:
        nodes, ways = parse_osm_elements(raw_alb_json)
        assert len(nodes) > 100
        assert len(ways) >= 10
        first_node_id = next(iter(nodes.keys()))
        assert isinstance(nodes[first_node_id], tuple)
        assert len(nodes[first_node_id]) == 2

    def test_parse_elements_empty(self) -> None:
        nodes, ways = parse_osm_elements({})
        assert nodes == {}
        assert ways == []


class TestDirectionalClassification:
    """Test tagging and bearing-based directional classification."""

    def test_classify_by_name_tag(self, alb_corridor: CorridorConfig) -> None:
        nodes = {1: (-77.22, 38.95), 2: (-77.21, 38.96)}
        way = {
            "nodes": [1, 2],
            "tags": {"name": "Capital Beltway (Inner Loop)", "ref": "I 495"},
        }
        direction = _classify_direction(way, nodes, alb_corridor)
        assert direction == "inner_loop"

    def test_classify_by_bearing_inner_outer(self, alb_corridor: CorridorConfig) -> None:
        # Heading NE (VA -> MD): Inner loop
        nodes = {1: (-77.22, 38.95), 2: (-77.20, 38.98)}
        way = {"nodes": [1, 2], "tags": {"ref": "I 495"}}
        assert _classify_direction(way, nodes, alb_corridor) == "inner_loop"

        # Heading SW (MD -> VA): Outer loop
        nodes_rev = {1: (-77.20, 38.98), 2: (-77.22, 38.95)}
        way_rev = {"nodes": [1, 2], "tags": {"ref": "I 495"}}
        assert _classify_direction(way_rev, nodes_rev, alb_corridor) == "outer_loop"


class TestStitchingAndTopologicalAssembly:
    """Test stitching way segments into continuous directional centerlines."""

    def test_stitch_segments_normal(self) -> None:
        seg1 = [(-77.22, 38.95), (-77.21, 38.96)]
        seg2 = [(-77.21, 38.96), (-77.20, 38.97)]
        line = _stitch_segments_to_linestring([seg1, seg2], "inner_loop")
        assert isinstance(line, LineString)
        assert len(line.coords) == 3
        assert list(line.coords)[0] == (-77.22, 38.95)
        assert list(line.coords)[-1] == (-77.20, 38.97)

    def test_stitch_inverted_segments(self) -> None:
        # seg2 is given in reverse order
        seg1 = [(-77.22, 38.95), (-77.21, 38.96)]
        seg2 = [(-77.20, 38.97), (-77.21, 38.96)]
        line = _stitch_segments_to_linestring([seg1, seg2], "inner_loop")
        assert isinstance(line, LineString)
        assert len(line.coords) == 3
        assert list(line.coords)[0] == (-77.22, 38.95)
        assert list(line.coords)[-1] == (-77.20, 38.97)

    def test_calculate_metric_length(self) -> None:
        line = LineString([(-77.2250, 38.9550), (-77.1850, 38.9950)])
        length = calculate_metric_length(line)
        # Bounding box diagonal is approx 5.5 to 6.0 km
        assert 5000.0 < length < 7000.0


class TestOSMExtractorEndToEnd:
    """Test complete extraction workflow using mock data and offline fixtures."""

    def test_extract_corridor_from_mock_json(
        self,
        alb_corridor: CorridorConfig,
        raw_alb_json: dict,
    ) -> None:
        extractor = OSMExtractor()
        gdf = extractor.extract_corridor(alb_corridor, raw_data=raw_alb_json, use_cache=False)

        assert len(gdf) == 2
        assert set(gdf["direction"]) == {"inner_loop", "outer_loop"}
        assert all(gdf["is_continuous"])
        assert all(gdf["has_bridge"])  # Bridge present in both directions
        assert all(isinstance(geom, LineString) for geom in gdf.geometry)

        # Inner Loop: VA -> MD (increasing latitude)
        inner = gdf[gdf["direction"] == "inner_loop"].iloc[0]
        assert inner.geometry.coords[0][1] < inner.geometry.coords[-1][1]
        assert 5000 < inner.length_meters < 7000

        # Outer Loop: MD -> VA (decreasing latitude)
        outer = gdf[gdf["direction"] == "outer_loop"].iloc[0]
        assert outer.geometry.coords[0][1] > outer.geometry.coords[-1][1]
        assert 5000 < outer.length_meters < 7000

        # Verify maxspeed is captured
        assert all(gdf["maxspeed"] == "65 mph")

    def test_extract_corridor_cache_hit(
        self,
        alb_corridor: CorridorConfig,
        raw_alb_json: dict,
        tmp_path: Path,
    ) -> None:
        cache_dir = tmp_path / "data" / "raw" / "geo"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{alb_corridor.corridor_id}_osm_raw.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(raw_alb_json, f)

        extractor = OSMExtractor(settings=Settings(project_root=tmp_path))
        with patch("topoflow.geo.osm_extractor.fetch_osm_data") as mock_fetch:
            gdf = extractor.extract_corridor(alb_corridor, use_cache=True)
            mock_fetch.assert_not_called()
            assert len(gdf) == 2

    def test_extract_all_mocked(
        self,
        alb_corridor: CorridorConfig,
        raw_alb_json: dict,
        tmp_path: Path,
    ) -> None:
        extractor = OSMExtractor(settings=Settings(project_root=tmp_path))
        sample_gdf = extractor.extract_corridor(
            alb_corridor, raw_data=raw_alb_json, use_cache=False
        )

        with patch.object(extractor, "extract_corridor", return_value=sample_gdf):
            from topoflow.config import CorridorsConfig

            config = CorridorsConfig(corridors={"alb_i495": alb_corridor})
            unified = extractor.extract_all(corridors_config=config)
            assert len(unified) == 2
            assert "corridor_id" in unified.columns

    def test_load_from_offline_fixture(self, fixtures_dir: Path) -> None:
        fixture_path = fixtures_dir / "osm_corridors_sample.geojson"
        gdf = OSMExtractor.load_from_fixture(fixture_path)

        assert len(gdf) == 6
        assert set(gdf["corridor_id"]) == {
            "alb_i495",
            "i95_lorton_woodbridge",
            "i295_anacostia",
        }
        alb_subset = gdf[gdf["corridor_id"] == "alb_i495"]
        assert all(alb_subset["has_bridge"])

    def test_export_parquet_and_geojson(
        self,
        alb_corridor: CorridorConfig,
        raw_alb_json: dict,
        tmp_path: Path,
    ) -> None:
        extractor = OSMExtractor(settings=Settings(project_root=tmp_path))
        gdf = extractor.extract_corridor(alb_corridor, raw_data=raw_alb_json, use_cache=False)

        geojson_path, parquet_path = extractor.export(gdf, output_dir=tmp_path / "geo")
        assert geojson_path.exists()
        assert parquet_path.exists()

        # Re-read and check round-trip integrity
        reread_geojson = gpd.read_file(geojson_path)
        reread_parquet = gpd.read_parquet(parquet_path)

        assert len(reread_geojson) == len(gdf)
        assert len(reread_parquet) == len(gdf)
        assert reread_geojson.crs == CRS_WGS84
        assert reread_parquet.crs == CRS_WGS84


class TestNetworkResilienceAndRetries:
    """Test HTTP backoff, retries, and error handling for Overpass API."""

    @patch("requests.Session.post")
    def test_fetch_retry_success_after_429(self, mock_post: MagicMock) -> None:
        resp_429 = MagicMock()
        resp_429.status_code = 429

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "elements": [{"type": "node", "id": 1, "lat": 38.0, "lon": -77.0}]
        }

        mock_post.side_effect = [resp_429, resp_200]

        data = fetch_osm_data("dummy_query", timeout=5, max_retries=2, backoff_factor=0.01)
        assert "elements" in data
        assert mock_post.call_count == 2

    @patch("requests.Session.post")
    def test_fetch_exhausted_retries_raises_network_error(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = requests.RequestException("Connection timeout")

        with pytest.raises(OSMNetworkError, match="Exhausted 2 retries"):
            fetch_osm_data("dummy_query", timeout=5, max_retries=2, backoff_factor=0.01)

    @patch("requests.Session.post")
    def test_fetch_invalid_json_raises_query_error(self, mock_post: MagicMock) -> None:
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = resp_200

        with pytest.raises(OSMQueryError, match="Failed to parse Overpass response as JSON"):
            fetch_osm_data("dummy_query", timeout=5, max_retries=1)

    @patch("requests.Session.post")
    def test_fetch_missing_elements_raises_query_error(self, mock_post: MagicMock) -> None:
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"version": 0.6}  # Missing elements
        mock_post.return_value = resp_200

        with pytest.raises(OSMQueryError, match="missing 'elements'"):
            fetch_osm_data("dummy_query", timeout=5, max_retries=1)


class TestManagedAndExpressLaneFiltering:
    """Test identification and filtering of managed, express, and reversible lanes."""

    def test_filter_reversible_oneway(self) -> None:
        way = {"tags": {"oneway": "reversible", "ref": "I 95"}}
        assert _is_managed_or_express_lane(way) is True

    def test_filter_express_name(self) -> None:
        way = {"tags": {"name": "95 Express Lanes", "ref": "I 95"}}
        assert _is_managed_or_express_lane(way) is True

    def test_filter_express_tag(self) -> None:
        way = {"tags": {"express": "yes", "ref": "I 95"}}
        assert _is_managed_or_express_lane(way) is True

    def test_filter_hov_hot_tags(self) -> None:
        way1 = {"tags": {"name": "I-95 HOV Lanes"}}
        way2 = {"tags": {"name": "HOT Lanes"}}
        assert _is_managed_or_express_lane(way1) is True
        assert _is_managed_or_express_lane(way2) is True

    def test_retain_general_purpose_lanes(self) -> None:
        way = {
            "tags": {
                "name": "Henry G. Shirley Memorial Highway",
                "ref": "I 95",
                "oneway": "yes",
                "highway": "motorway",
            }
        }
        assert _is_managed_or_express_lane(way) is False


class TestDeflectionAndUTurnGuards:
    """Test geometric deflection checks preventing cross-median U-turns."""

    def test_deflection_forward_aligned(self) -> None:
        v_in = (10.0, 10.0)
        v_out = (10.0, 12.0)
        assert _is_deflection_acceptable(v_in, v_out, min_cos=0.0) is True

    def test_deflection_opposing_u_turn_rejected(self) -> None:
        v_in = (10.0, 10.0)  # Heading NE
        v_out = (-10.0, -10.0)  # Heading SW (180 deg turnaround)
        assert _is_deflection_acceptable(v_in, v_out, min_cos=0.0) is False

    def test_prepend_reverse_valid(self) -> None:
        # seg1 is northern half: (0, 1) -> (0, 2)
        seg1 = [(0.0, 1.0), (0.0, 2.0)]
        # seg2 is southern half digitized backwards: (0, 1) -> (0, 0)
        # Prepending seg2 reversed yields (0, 0) -> (0, 1) -> (0, 2)
        seg2 = [(0.0, 1.0), (0.0, 0.0)]
        line = _stitch_segments_to_linestring([seg1, seg2], "northbound")
        coords = list(line.coords)
        assert len(coords) == 3
        assert coords[0] == (0.0, 0.0)
        assert coords[-1] == (0.0, 2.0)


# =============================================================================
# Metric-gap stitching, way intervals and provenance-honest corridor attributes
# =============================================================================

_TO_WGS84 = Transformer.from_crs(CRS_METRIC, CRS_WGS84, always_xy=True)

# Base point in UTM 18N near the ALB study corridor.
_E0 = 310400.0
_N0 = 4314000.0


def _ll(easting: float, northing: float) -> tuple[float, float]:
    """Convert a UTM 18N coordinate to a (lon, lat) tuple."""
    lon, lat = _TO_WGS84.transform(easting, northing)
    return (float(lon), float(lat))


def _north_run(start_m: float, end_m: float, step_m: float = 250.0) -> list[tuple[float, float]]:
    """A due-north (lon, lat) polyline from N0+start_m to N0+end_m."""
    n = int(round((end_m - start_m) / step_m))
    return [_ll(_E0, _N0 + start_m + i * step_m) for i in range(n + 1)]


class TestStitchSegmentsWithGap:
    """Tests for metric (EPSG:26918) junction gaps and exact WGS84 coordinate preservation."""

    def test_single_and_touching_segments_zero_gap(self) -> None:
        seg1 = _north_run(0.0, 1000.0)
        seg2 = _north_run(1000.0, 2000.0)
        line, gap = stitch_segments_with_gap([seg1], "northbound")
        assert isinstance(line, LineString)
        assert gap == 0.0
        assert list(line.coords) == seg1

        line, gap = stitch_segments_with_gap([seg1, seg2], "inner_loop")
        assert gap == 0.0
        # The shared junction vertex is not duplicated.
        assert len(line.coords) == len(seg1) + len(seg2) - 1

    @pytest.mark.parametrize("join", ["forward_append", "reversed_prepend"])
    def test_coordinates_preserved_exactly(self, join: str) -> None:
        """Every input vertex survives a bridged gap, exactly (no reprojection round trip)."""
        seg1 = _north_run(0.0, 1000.0)
        seg2 = _north_run(1020.0, 2020.0)
        if join == "forward_append":
            segments = [seg1, seg2]
        else:
            # Start from the northern half digitised backwards; the southern half
            # (also backwards) has to be prepended.
            segments = [list(reversed(seg2)), list(reversed(seg1))]
        line, gap = stitch_segments_with_gap(segments, "northbound")
        coords = [tuple(c) for c in line.coords]
        assert gap == pytest.approx(20.0, abs=0.05)
        assert len(coords) == len(seg1) + len(seg2)
        assert coords == seg1 + seg2  # northbound: south-to-north order

    def test_largest_bridged_gap_reported_in_metres(self) -> None:
        seg1 = _north_run(0.0, 1000.0)
        seg2 = _north_run(1010.0, 2010.0)
        seg3 = _north_run(2040.0, 3040.0)
        line, gap = stitch_segments_with_gap([seg1, seg2, seg3], "northbound")
        assert gap == pytest.approx(30.0, abs=0.05)
        assert line.coords[0] == seg1[0]
        assert line.coords[-1] == seg3[-1]

    def test_gap_limit_in_metres(self) -> None:
        seg1 = _north_run(0.0, 1000.0)
        seg2 = _north_run(1060.0, 2060.0)
        with pytest.raises(OSMTopologicalError, match="Topological gap"):
            stitch_segments_with_gap([seg1, seg2], "northbound")  # default 50 m

        _, gap = stitch_segments_with_gap([seg1, seg2], "northbound", max_gap_m=100.0)
        assert gap == pytest.approx(60.0, abs=0.05)

        seg3 = _north_run(1020.0, 2020.0)
        with pytest.raises(OSMTopologicalError, match="Topological gap"):
            stitch_segments_with_gap([seg1, seg3], "northbound", max_gap_m=10.0)

    @pytest.mark.parametrize(
        ("direction", "reverse_input", "northward"),
        [("inner_loop", True, True), ("southbound", False, False)],
    )
    def test_orientation(self, direction: str, reverse_input: bool, northward: bool) -> None:
        """inner_loop/northbound end further north; outer_loop/southbound further south."""
        segs = [_north_run(0.0, 1000.0), _north_run(1000.0, 2000.0)]
        if reverse_input:
            segs = [list(reversed(s)) for s in segs]
        line, _ = stitch_segments_with_gap(segs, direction)
        assert (line.coords[-1][1] > line.coords[0][1]) is northward

    def test_empty_raises(self) -> None:
        with pytest.raises(OSMTopologicalError, match="No coordinate segments"):
            stitch_segments_with_gap([], "northbound")

    def test_short_segment_raises(self) -> None:
        with pytest.raises(OSMTopologicalError, match="fewer than 2 points"):
            stitch_segments_with_gap([_north_run(0.0, 1000.0), [_ll(_E0, _N0)]], "northbound")

    def test_u_turn_raises(self) -> None:
        seg1 = _north_run(0.0, 1000.0)
        # Opposite carriageway 15 m east, running back south: a >90 deg median U-turn.
        seg2 = [_ll(_E0 + 15.0, _N0 + 1000.0), _ll(_E0 + 15.0, _N0)]
        with pytest.raises(OSMTopologicalError):
            stitch_segments_with_gap([seg1, seg2], "northbound")

    def test_wrapper_matches_and_rejects_legacy_kwarg(self) -> None:
        seg1 = _north_run(0.0, 1000.0)
        seg2 = _north_run(1060.0, 2060.0)
        line, _ = stitch_segments_with_gap([seg1, seg2], "northbound", max_gap_m=100.0)
        wrapped = _stitch_segments_to_linestring([seg1, seg2], "northbound", max_gap_m=100.0)
        assert isinstance(wrapped, LineString)
        assert list(wrapped.coords) == list(line.coords)
        with pytest.raises(TypeError):
            _stitch_segments_to_linestring([seg1], "northbound", max_gap_degrees=0.001)  # type: ignore[call-arg]


@pytest.fixture
def interval_network() -> tuple[LineString, dict[int, tuple[float, float]]]:
    """A 2 km due-north line with nodes every 500 m (ids 1..5)."""
    nodes = {i + 1: _ll(_E0, _N0 + 500.0 * i) for i in range(5)}
    line = LineString([nodes[i] for i in range(1, 6)])
    return line, nodes


def _way(way_id: int, node_ids: list[int], **tags: str) -> dict:
    return {"id": way_id, "nodes": node_ids, "tags": dict(tags)}


class TestComputeWayIntervals:
    """Tests for chainage intervals of OSM ways along a stitched carriageway."""

    def test_basic_intervals_and_keys(self, interval_network) -> None:
        line, nodes = interval_network
        ways = [
            _way(10, [1, 2, 3], lanes="4", maxspeed="55 mph"),
            _way(20, [3, 4, 5], bridge="yes", lanes="5", maxspeed="65 mph"),
        ]
        intervals = compute_way_intervals(line, ways, nodes)
        assert len(intervals) == 2
        for iv in intervals:
            assert set(iv.keys()) >= {"start_m", "end_m", "bridge", "lanes", "maxspeed", "way_id"}
        first, second = intervals
        assert first["way_id"] == 10
        assert first["start_m"] == pytest.approx(0.0, abs=0.5)
        assert first["end_m"] == pytest.approx(1000.0, abs=0.5)
        assert first["bridge"] is False
        assert first["lanes"] == 4
        assert first["maxspeed"] == "55 mph"
        assert second["way_id"] == 20
        assert second["start_m"] == pytest.approx(1000.0, abs=0.5)
        assert second["end_m"] == pytest.approx(2000.0, abs=0.5)
        assert second["bridge"] is True
        assert second["lanes"] == 5

    def test_reverse_digitised_way_same_interval(self, interval_network) -> None:
        line, nodes = interval_network
        forward = compute_way_intervals(line, [_way(1, [2, 3, 4])], nodes)
        backward = compute_way_intervals(line, [_way(1, [4, 3, 2])], nodes)
        assert forward[0]["start_m"] == pytest.approx(backward[0]["start_m"], abs=1e-6)
        assert forward[0]["end_m"] == pytest.approx(backward[0]["end_m"], abs=1e-6)
        assert backward[0]["start_m"] == pytest.approx(500.0, abs=0.5)
        assert backward[0]["end_m"] == pytest.approx(1500.0, abs=0.5)

    def test_sorted_by_start(self, interval_network) -> None:
        line, nodes = interval_network
        ways = [_way(3, [4, 5]), _way(1, [1, 2]), _way(2, [2, 3, 4])]
        intervals = compute_way_intervals(line, ways, nodes)
        assert [iv["way_id"] for iv in intervals] == [1, 2, 3]

    def test_skips_degenerate_ways(self, interval_network) -> None:
        line, nodes = interval_network
        assert compute_way_intervals(line, [], nodes) == []
        ways = [
            _way(1, [1, 2]),
            _way(2, [3]),  # single node
            _way(3, [999, 4]),  # only one resolvable node
            _way(4, [3, 3]),  # zero chainage length
            _way(5, []),  # no nodes
        ]
        intervals = compute_way_intervals(line, ways, nodes)
        assert [iv["way_id"] for iv in intervals] == [1]

    def test_bridge_tag_true(self, interval_network) -> None:
        line, nodes = interval_network
        ways = [_way(1, [1, 2], bridge="yes"), _way(2, [2, 3], bridge="viaduct")]
        assert [iv["bridge"] for iv in compute_way_intervals(line, ways, nodes)] == [True, True]

    def test_bridge_tag_false(self, interval_network) -> None:
        line, nodes = interval_network
        ways = [_way(1, [1, 2], bridge="no"), _way(2, [2, 3], bridge=""), _way(3, [3, 4])]
        intervals = compute_way_intervals(line, ways, nodes)
        assert [iv["bridge"] for iv in intervals] == [False, False, False]

    def test_lanes_digits_parsed_as_int(self, interval_network) -> None:
        line, nodes = interval_network
        ways = [_way(1, [1, 2], lanes="3"), _way(2, [2, 3], lanes="12")]
        lanes = [iv["lanes"] for iv in compute_way_intervals(line, ways, nodes)]
        assert lanes == [3, 12]
        assert all(isinstance(v, int) for v in lanes)

    def test_lanes_non_numeric_or_absent_is_none(self, interval_network) -> None:
        line, nodes = interval_network
        ways = [_way(1, [1, 2], lanes="3;2"), _way(2, [2, 3], lanes="two"), _way(3, [3, 4])]
        assert [iv["lanes"] for iv in compute_way_intervals(line, ways, nodes)] == [
            None,
            None,
            None,
        ]

    def test_maxspeed_and_json_serialisable(self, interval_network) -> None:
        line, nodes = interval_network
        intervals = compute_way_intervals(
            line, [_way(1, [1, 2], maxspeed="65 mph", bridge="yes"), _way(2, [2, 3])], nodes
        )
        assert intervals[0]["maxspeed"] == "65 mph"
        assert intervals[1]["maxspeed"] is None
        assert json.loads(json.dumps(intervals))[0]["bridge"] is True


def _synthetic_raw(inner_ways: list[dict], outer_ways: list[dict], nodes: dict) -> dict:
    elements: list[dict] = [
        {"type": "node", "id": nid, "lon": lon, "lat": lat} for nid, (lon, lat) in nodes.items()
    ]
    for way in inner_ways + outer_ways:
        elements.append({"type": "way", **way})
    return {"elements": elements}


@pytest.fixture
def synthetic_alb_raw() -> dict:
    """Two untagged-lane, untagged-speed carriageways inside the ALB bounding box."""
    # Inner loop runs NE, outer loop runs SW ~30 m east.
    nodes: dict[int, tuple[float, float]] = {}
    for i in range(5):
        nodes[100 + i] = _ll(_E0 + 150.0 * i, _N0 + 500.0 * i)
        nodes[200 + i] = _ll(_E0 + 30.0 + 150.0 * (4 - i), _N0 + 500.0 * (4 - i))
    base_tags = {"highway": "motorway", "ref": "I 495", "oneway": "yes"}
    inner = [
        {
            "id": 1,
            "nodes": [100, 101, 102],
            "tags": {**base_tags, "name": "Capital Beltway (Inner Loop)"},
        },
        {
            "id": 2,
            "nodes": [102, 103, 104],
            "tags": {**base_tags, "name": "Capital Beltway (Inner Loop)"},
        },
    ]
    outer = [
        {
            "id": 3,
            "nodes": [200, 201, 202, 203, 204],
            "tags": {**base_tags, "name": "Capital Beltway (Outer Loop)"},
        },
    ]
    return _synthetic_raw(inner, outer, nodes)


class TestExtractCorridorProvenance:
    """No fabricated defaults: lanes / maxspeed only from OSM tags."""

    def test_untagged_lanes_and_maxspeed_are_none(
        self, alb_corridor: CorridorConfig, synthetic_alb_raw: dict
    ) -> None:
        gdf = OSMExtractor().extract_corridor(
            alb_corridor, raw_data=synthetic_alb_raw, use_cache=False
        )
        assert set(gdf["direction"]) == {"inner_loop", "outer_loop"}
        for _, row in gdf.iterrows():
            assert row["lanes"] is None or pd.isna(row["lanes"])
            assert row["maxspeed"] is None or pd.isna(row["maxspeed"])
            assert not bool(row["has_bridge"])
            assert isinstance(row["max_junction_gap_m"], float)
            assert row["max_junction_gap_m"] == pytest.approx(0.0, abs=1e-6)
            assert bool(row["is_continuous"]) is True
            intervals = json.loads(row["way_intervals"])
            assert isinstance(intervals, list)
            assert intervals
            assert not any(iv["bridge"] for iv in intervals)

    def test_lanes_median_and_maxspeed_mode(self, alb_corridor: CorridorConfig) -> None:
        nodes = {100 + i: _ll(_E0 + 150.0 * i, _N0 + 500.0 * i) for i in range(4)}
        tags = {"highway": "motorway", "ref": "I 495", "oneway": "yes"}
        inner = [
            {"id": 1, "nodes": [100, 101], "tags": {**tags, "lanes": "3", "maxspeed": "55 mph"}},
            {
                "id": 2,
                "nodes": [101, 102],
                "tags": {**tags, "lanes": "4", "maxspeed": "65 mph", "bridge": "yes"},
            },
            {"id": 3, "nodes": [102, 103], "tags": {**tags, "lanes": "4", "maxspeed": "65 mph"}},
        ]
        raw = _synthetic_raw(inner, [], nodes)
        gdf = OSMExtractor().extract_corridor(alb_corridor, raw_data=raw, use_cache=False)
        inner_row = gdf[gdf["direction"] == "inner_loop"].iloc[0]
        assert int(inner_row["lanes"]) == 4
        assert inner_row["maxspeed"] == "65 mph"
        assert bool(inner_row["has_bridge"]) is True
        intervals = json.loads(inner_row["way_intervals"])
        assert [iv["way_id"] for iv in intervals] == [1, 2, 3]
        assert [iv["bridge"] for iv in intervals] == [False, True, False]

    def test_junction_gap_reported(self, alb_corridor: CorridorConfig) -> None:
        nodes = {
            1: _ll(_E0, _N0),
            2: _ll(_E0 + 200.0, _N0 + 1000.0),
            3: _ll(_E0 + 200.0, _N0 + 1012.0),
            4: _ll(_E0 + 400.0, _N0 + 2000.0),
        }
        tags = {"highway": "motorway", "ref": "I 495", "oneway": "yes"}
        inner = [
            {"id": 1, "nodes": [1, 2], "tags": dict(tags)},
            {"id": 2, "nodes": [3, 4], "tags": dict(tags)},
        ]
        gdf = OSMExtractor().extract_corridor(
            alb_corridor, raw_data=_synthetic_raw(inner, [], nodes), use_cache=False
        )
        row = gdf[gdf["direction"] == "inner_loop"].iloc[0]
        assert row["max_junction_gap_m"] == pytest.approx(12.0, abs=0.1)
        # Bridged, but a 12 m gap exceeds the 1 m continuity tolerance.
        assert bool(row["is_continuous"]) is False

    def test_sub_metre_gap_is_continuous(self, alb_corridor: CorridorConfig) -> None:
        nodes = {
            1: _ll(_E0, _N0),
            2: _ll(_E0, _N0 + 1000.0),
            3: _ll(_E0, _N0 + 1000.6),
            4: _ll(_E0, _N0 + 2000.0),
        }
        tags = {"highway": "motorway", "ref": "I 495", "oneway": "yes"}
        inner = [
            {"id": 1, "nodes": [1, 2], "tags": dict(tags)},
            {"id": 2, "nodes": [3, 4], "tags": dict(tags)},
        ]
        gdf = OSMExtractor().extract_corridor(
            alb_corridor, raw_data=_synthetic_raw(inner, [], nodes), use_cache=False
        )
        row = gdf[gdf["direction"] == "inner_loop"].iloc[0]
        assert row["max_junction_gap_m"] == pytest.approx(0.6, abs=0.05)
        assert CONTINUITY_TOLERANCE_METERS == 1.0
        assert row["max_junction_gap_m"] <= CONTINUITY_TOLERANCE_METERS
        assert bool(row["is_continuous"]) is True

    @pytest.mark.parametrize(
        ("lane_tags", "expected"),
        [
            (["4", "5"], 4),
            (["4", None, "5", "6"], 5),
        ],
    )
    def test_lanes_summary_is_median_low(
        self, alb_corridor: CorridorConfig, lane_tags: list[str | None], expected: int
    ) -> None:
        """The carriageway lane summary is always an actually-tagged value."""
        n = len(lane_tags)
        nodes = {100 + i: _ll(_E0, _N0 + 500.0 * i) for i in range(n + 1)}
        tags = {"highway": "motorway", "ref": "I 495", "oneway": "yes"}
        inner = []
        for i, lanes in enumerate(lane_tags):
            way_tags = dict(tags)
            if lanes is not None:
                way_tags["lanes"] = lanes
            inner.append({"id": i + 1, "nodes": [100 + i, 101 + i], "tags": way_tags})
        gdf = OSMExtractor().extract_corridor(
            alb_corridor, raw_data=_synthetic_raw(inner, [], nodes), use_cache=False
        )
        row = gdf[gdf["direction"] == "inner_loop"].iloc[0]
        assert int(row["lanes"]) == expected

    def test_fixture_bridge_interval(
        self, alb_corridor: CorridorConfig, raw_alb_json: dict
    ) -> None:
        """The American Legion Bridge way is a few-hundred-metre interval inside each line."""
        alb_way_ids = {
            el["id"]
            for el in raw_alb_json["elements"]
            if el.get("type") == "way"
            and el.get("tags", {}).get("bridge:name") == "American Legion Bridge"
        }
        assert alb_way_ids

        gdf = OSMExtractor().extract_corridor(alb_corridor, raw_data=raw_alb_json, use_cache=False)
        assert "max_junction_gap_m" in gdf.columns
        assert "way_intervals" in gdf.columns
        for _, row in gdf.iterrows():
            assert isinstance(row["way_intervals"], str)
            intervals = json.loads(row["way_intervals"])
            starts = [iv["start_m"] for iv in intervals]
            assert starts == sorted(starts)
            assert all(iv["end_m"] > iv["start_m"] for iv in intervals)
            assert max(iv["end_m"] for iv in intervals) <= row["length_meters"] + 1.0
            assert bool(row["is_continuous"]) == (
                row["max_junction_gap_m"] <= CONTINUITY_TOLERANCE_METERS
            )

            alb = [iv for iv in intervals if iv["way_id"] in alb_way_ids]
            assert len(alb) == 1, f"{row['direction']}: expected one ALB bridge interval"
            span = alb[0]
            assert span["bridge"] is True
            assert 200.0 < span["end_m"] - span["start_m"] < 800.0
            assert span["start_m"] > 500.0
            assert span["end_m"] < row["length_meters"] - 500.0


class TestOSMToStationsRoundTrip:
    """Synthetic OSM JSON -> extract_corridor -> SpatialDiscretizer, bridge flags end to end."""

    # Due-north inner loop. Chainage of each node from the first, in metres (EPSG:26918).
    APPROACH_S = [0.0, 500.0, 1000.0, 1500.0]
    BRIDGE = [1510.0, 1730.0, 1950.0]  # ~10 m junction gap from 1500 -> 1510
    APPROACH_N = [1950.0, 2500.0, 3000.0]

    @pytest.fixture
    def raw_osm(self) -> dict:
        node_ids: dict[float, int] = {}
        nodes: dict[int, tuple[float, float]] = {}
        for i, ch in enumerate(sorted({*self.APPROACH_S, *self.BRIDGE, *self.APPROACH_N})):
            node_ids[ch] = 1000 + i
            nodes[1000 + i] = _ll(_E0, _N0 + ch)

        tags = {
            "highway": "motorway",
            "ref": "I 495",
            "oneway": "yes",
            "name": "Capital Beltway (Inner Loop)",
        }

        def reversed_ids(chainages: list[float]) -> list[int]:
            # Digitised against the direction of travel (north-to-south).
            return [node_ids[c] for c in reversed(chainages)]

        ways = [
            {"id": 11, "nodes": reversed_ids(self.APPROACH_S), "tags": {**tags, "lanes": "4"}},
            {
                "id": 12,
                "nodes": reversed_ids(self.BRIDGE),
                "tags": {**tags, "lanes": "5", "bridge": "yes"},
            },
            {"id": 13, "nodes": reversed_ids(self.APPROACH_N), "tags": {**tags, "lanes": "4"}},
        ]
        return _synthetic_raw(ways, [], nodes)

    def test_bridge_stations_inside_bridge_way_extent(
        self, alb_corridor: CorridorConfig, raw_osm: dict
    ) -> None:
        gdf = OSMExtractor().extract_corridor(alb_corridor, raw_data=raw_osm, use_cache=False)
        row = gdf[gdf["direction"] == "inner_loop"].iloc[0]

        # Line runs south-to-north despite the reversed digitisation.
        assert row.geometry.coords[-1][1] > row.geometry.coords[0][1]
        assert row["max_junction_gap_m"] == pytest.approx(10.0, abs=0.1)
        assert bool(row["is_continuous"]) is False
        assert bool(row["has_bridge"]) is True

        intervals = json.loads(row["way_intervals"])
        bridge_ivs = [iv for iv in intervals if iv["bridge"]]
        assert [iv["way_id"] for iv in bridge_ivs] == [12]
        bridge_iv = bridge_ivs[0]
        # Independent expectation: the bridge nodes' metric chainage along the line.
        assert bridge_iv["start_m"] == pytest.approx(self.BRIDGE[0], abs=0.5)
        assert bridge_iv["end_m"] == pytest.approx(self.BRIDGE[-1], abs=0.5)
        assert row["length_meters"] == pytest.approx(self.APPROACH_N[-1], abs=1.0)

        stations = SpatialDiscretizer(interval_m=100.0).discretize_corridors_dataframe(gdf)
        inner = stations[stations["direction"] == "inner_loop"]
        assert len(inner) >= 30
        is_bridge = inner["has_bridge"].fillna(False).astype(bool)
        chainage = inner["chainage_m"]
        inside = (chainage >= bridge_iv["start_m"]) & (chainage <= bridge_iv["end_m"])
        # Exactly the stations inside the bridge way's metric extent are flagged.
        assert (is_bridge == inside).all()
        assert sorted(chainage[is_bridge].round(3).tolist()) == [1600.0, 1700.0, 1800.0, 1900.0]
        lanes = dict(zip(chainage.round(3), inner["lanes"], strict=True))
        assert lanes[1700.0] == 5
        assert lanes[1000.0] == 4
        assert lanes[2500.0] == 4
