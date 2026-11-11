"""Unit tests for OpenStreetMap highway vector extraction and carriageway assembly.

Validates Overpass QL query generation, element parsing, directional classification,
continuous LineString stitching, bridge tagging, network retry/backoff, and serialization.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
import requests
from shapely.geometry import LineString

from topoflow.config import BoundingBox, Centroid, CorridorConfig, Settings
from topoflow.constants import CRS_WGS84
from topoflow.geo.osm_extractor import (
    OSMExtractor,
    OSMNetworkError,
    OSMQueryError,
    OSMTopologicalError,
    _classify_direction,
    _stitch_segments_to_linestring,
    build_overpass_query,
    calculate_metric_length,
    fetch_osm_data,
    parse_osm_elements,
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

    def test_stitch_empty_raises_error(self) -> None:
        with pytest.raises(OSMTopologicalError, match="No coordinate segments"):
            _stitch_segments_to_linestring([], "inner_loop")

    def test_stitch_gap_exceeds_tolerance_raises_error(self) -> None:
        seg1 = [(-77.2200, 38.9500), (-77.2100, 38.9600)]
        # seg2 endpoint has an unphysical gap > 0.0005 degrees (~50m, e.g. cross-median jump)
        seg2 = [(-77.2090, 38.9600), (-77.2000, 38.9700)]
        with pytest.raises(OSMTopologicalError, match="Topological gap"):
            _stitch_segments_to_linestring([seg1, seg2], "inner_loop")

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
