"""Unit tests for USGS 3DEP LiDAR point cloud ingestion module.

Validates octree bounding box calculations, bounding box intersection logic,
EPT hierarchy traversal and sub-hierarchy expansion, single-node LAZ fetching
with exponential backoff retries, spatial coordinate reprojection between
EPSG:3857 and EPSG:26918 (UTM 18N), and end-to-end American Legion Bridge
corridor LiDAR point cloud ingestion with caching and spatial filtering.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import laspy
import numpy as np
import pyproj
import pytest
import requests
from shapely.geometry import Point

from topoflow.constants import CRS_METRIC
from topoflow.ingestion.lidar_usgs import (
    CRS_EPT,
    DEFAULT_EPT_BASE_URL,
    compute_node_bounds,
    fetch_ept_node_laz,
    find_intersecting_ept_nodes,
    ingest_usgs_alb_lidar,
    node_intersects_bbox,
    reproject_points,
)

# =============================================================================
# Helper Utilities & Fixtures
# =============================================================================


def make_mock_laz_bytes(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    return_number: np.ndarray | None = None,
) -> bytes:
    """Create in-memory compressed LAZ bytes containing specified points.

    Args:
        x: X coordinate array.
        y: Y coordinate array.
        z: Z coordinate array.
        classification: Optional classification array (uint8).
        intensity: Optional intensity array (uint16).
        return_number: Optional return number array (uint8).

    Returns:
        Raw bytes representing a valid LAS/LAZ point cloud file.
    """
    n = len(x)
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = [
        float(np.floor(np.min(x))) if n > 0 else 0.0,
        float(np.floor(np.min(y))) if n > 0 else 0.0,
        float(np.floor(np.min(z))) if n > 0 else 0.0,
    ]
    header.scales = [0.01, 0.01, 0.01]

    las_data = laspy.LasData(header)
    las_data.x = x
    las_data.y = y
    las_data.z = z
    las_data.classification = (
        classification if classification is not None else np.full(n, 2, dtype=np.uint8)
    )
    las_data.intensity = intensity if intensity is not None else np.full(n, 100, dtype=np.uint16)
    las_data.return_number = (
        return_number if return_number is not None else np.ones(n, dtype=np.uint8)
    )

    buf = io.BytesIO()
    las_data.write(buf)
    return buf.getvalue()


@pytest.fixture
def sample_root_bounds() -> list[float]:
    """Sample root octree bounding box in EPSG:3857 [minx, miny, minz, maxx, maxy, maxz]."""
    return [-9000000.0, 4500000.0, 0.0, -8500000.0, 5000000.0, 1000.0]


@pytest.fixture
def sample_alb_bridge_stations() -> gpd.GeoDataFrame:
    """Sample corridor stations with valid bridge stations along ALB Potomac crossing."""
    # UTM 18N coordinates within ALB bridge northing range [4315400, 4316500]
    coords = [
        (308000.0, 4315200.0, False),  # Outside northing and not bridge
        (308100.0, 4315600.0, True),  # ALB bridge span 1
        (308200.0, 4315900.0, True),  # ALB bridge span 2 (center)
        (308300.0, 4316300.0, True),  # ALB bridge span 3
        (308400.0, 4316800.0, True),  # Beyond ALB northing threshold
    ]
    return gpd.GeoDataFrame(
        {
            "corridor_id": ["alb_i495"] * len(coords),
            "station_m": [0.0, 400.0, 700.0, 1100.0, 1600.0],
            "has_bridge": [c[2] for c in coords],
        },
        geometry=[Point(c[0], c[1]) for c in coords],
        crs=CRS_METRIC,
    )


# =============================================================================
# Test Suite: Module Constants
# =============================================================================


class TestModuleConstants:
    """Test suite for LiDAR ingestion module constants."""

    def test_constants_definitions(self) -> None:
        """Verify default EPT archive URL and projection definitions."""
        assert DEFAULT_EPT_BASE_URL.startswith("https://")
        assert "USGS_LPC_VA_Fairfax_County_2018" in DEFAULT_EPT_BASE_URL
        assert CRS_EPT == "EPSG:3857"
        assert CRS_METRIC == "EPSG:26918"


# =============================================================================
# Test Suite: compute_node_bounds
# =============================================================================


class TestComputeNodeBounds:
    """Test suite for octree node bounding box calculation."""

    def test_root_node_depth_0(self, sample_root_bounds: list[float]) -> None:
        """At depth 0 ('0-0-0-0'), node bounds must match root horizontal bounds."""
        min_x, min_y, max_x, max_y = compute_node_bounds("0-0-0-0", sample_root_bounds)
        assert min_x == pytest.approx(sample_root_bounds[0])
        assert min_y == pytest.approx(sample_root_bounds[1])
        assert max_x == pytest.approx(sample_root_bounds[3])
        assert max_y == pytest.approx(sample_root_bounds[4])

    def test_depth_1_halved_spans(self, sample_root_bounds: list[float]) -> None:
        """At depth 1, horizontal spans must be halved and partitioned by quadrant."""
        span_x = sample_root_bounds[3] - sample_root_bounds[0]
        half_span = span_x / 2.0

        # Bottom-left quadrant: 1-0-0-0
        bl_min_x, bl_min_y, bl_max_x, bl_max_y = compute_node_bounds("1-0-0-0", sample_root_bounds)
        assert bl_min_x == pytest.approx(sample_root_bounds[0])
        assert bl_max_x == pytest.approx(sample_root_bounds[0] + half_span)
        assert bl_min_y == pytest.approx(sample_root_bounds[1])
        assert bl_max_y == pytest.approx(sample_root_bounds[1] + half_span)

        # Top-right quadrant: 1-1-1-0
        tr_min_x, tr_min_y, tr_max_x, tr_max_y = compute_node_bounds("1-1-1-0", sample_root_bounds)
        assert tr_min_x == pytest.approx(sample_root_bounds[0] + half_span)
        assert tr_max_x == pytest.approx(sample_root_bounds[3])
        assert tr_min_y == pytest.approx(sample_root_bounds[1] + half_span)
        assert tr_max_y == pytest.approx(sample_root_bounds[4])

    def test_depth_2_quadrants(self, sample_root_bounds: list[float]) -> None:
        """At depth 2, horizontal spans must be divided by 4 (2^2)."""
        span = (sample_root_bounds[3] - sample_root_bounds[0]) / 4.0
        min_x, min_y, max_x, max_y = compute_node_bounds("2-2-3-0", sample_root_bounds)
        expected_min_x = sample_root_bounds[0] + 2 * span
        expected_min_y = sample_root_bounds[1] + 3 * span
        assert min_x == pytest.approx(expected_min_x)
        assert max_x == pytest.approx(expected_min_x + span)
        assert min_y == pytest.approx(expected_min_y)
        assert max_y == pytest.approx(expected_min_y + span)


# =============================================================================
# Test Suite: node_intersects_bbox
# =============================================================================


class TestNodeIntersectsBbox:
    """Test suite for bounding box intersection checks."""

    def test_overlapping_boxes(self) -> None:
        """Partially overlapping bounding boxes must return True."""
        node_bounds = (100.0, 100.0, 200.0, 200.0)
        target_bbox = (150.0, 150.0, 250.0, 250.0)
        assert node_intersects_bbox(node_bounds, target_bbox) is True

    def test_contained_boxes(self) -> None:
        """Fully contained bounding boxes in either direction must return True."""
        outer = (0.0, 0.0, 500.0, 500.0)
        inner = (100.0, 100.0, 200.0, 200.0)
        assert node_intersects_bbox(outer, inner) is True
        assert node_intersects_bbox(inner, outer) is True

    def test_disjoint_boxes(self) -> None:
        """Disjoint bounding boxes on any cardinal axis must return False."""
        base = (100.0, 100.0, 200.0, 200.0)
        # To the right
        assert node_intersects_bbox(base, (250.0, 100.0, 300.0, 200.0)) is False
        # To the left
        assert node_intersects_bbox(base, (0.0, 100.0, 50.0, 200.0)) is False
        # Above
        assert node_intersects_bbox(base, (100.0, 250.0, 200.0, 300.0)) is False
        # Below
        assert node_intersects_bbox(base, (100.0, 0.0, 200.0, 50.0)) is False

    def test_touching_edges(self) -> None:
        """Boxes sharing an exact edge must return True."""
        box1 = (100.0, 100.0, 200.0, 200.0)
        # Touches right edge
        box_right = (200.0, 100.0, 300.0, 200.0)
        assert node_intersects_bbox(box1, box_right) is True
        # Touches top edge
        box_top = (100.0, 200.0, 200.0, 300.0)
        assert node_intersects_bbox(box1, box_top) is True


# =============================================================================
# Test Suite: find_intersecting_ept_nodes
# =============================================================================


class TestFindIntersectingEptNodes:
    """Test suite for traversing EPT octree hierarchy without live network access."""

    @patch("requests.Session.get")
    def test_find_nodes_basic_and_depth_filter(self, mock_get: MagicMock) -> None:
        """Traverse root hierarchy and filter by depth and target bounding box."""
        root_bounds = [0.0, 0.0, 0.0, 1000.0, 1000.0, 100.0]
        ept_json_resp = MagicMock()
        ept_json_resp.status_code = 200
        ept_json_resp.json.return_value = {"bounds": root_bounds}

        # Root hierarchy with various depths and points
        hier_resp = MagicMock()
        hier_resp.status_code = 200
        hier_resp.json.return_value = {
            "0-0-0-0": 500,  # Depth 0: intersects target
            "1-0-0-0": 250,  # Depth 1: [0, 0, 500, 500] -> intersects
            "1-1-1-0": 300,  # Depth 1: [500, 500, 1000, 1000] -> disjoint
            "2-0-0-0": 100,  # Depth 2: within target
            "9-0-0-0": 50,  # Depth 9: exceeds max_depth of 8
            "1-0-1-0": 0,  # 0 points -> excluded
        }

        mock_get.side_effect = [ept_json_resp, hier_resp]

        target_bbox = (50.0, 50.0, 200.0, 200.0)
        nodes = find_intersecting_ept_nodes(
            ept_base_url="https://mock-ept.s3.amazonaws.com/test",
            target_bbox_3857=target_bbox,
            max_depth=8,
        )

        node_keys = [k for k, _ in nodes]
        assert "0-0-0-0" in node_keys
        assert "1-0-0-0" in node_keys
        assert "2-0-0-0" in node_keys
        assert "1-1-1-0" not in node_keys  # Disjoint
        assert "9-0-0-0" not in node_keys  # Exceeds max depth
        assert "1-0-1-0" not in node_keys  # 0 points

    @patch("requests.Session.get")
    def test_subhierarchy_expansion(self, mock_get: MagicMock) -> None:
        """Nodes with count == -1 must trigger sub-hierarchy fetch and parsing."""
        root_bounds = [0.0, 0.0, 0.0, 1000.0, 1000.0, 100.0]
        ept_json_resp = MagicMock(status_code=200)
        ept_json_resp.json.return_value = {"bounds": root_bounds}

        # Root hierarchy contains sub-hierarchy pointer (-1)
        root_hier_resp = MagicMock(status_code=200)
        root_hier_resp.json.return_value = {
            "1-0-0-0": -1,  # Sub-hierarchy marker
        }

        # Sub-hierarchy contents
        subhier_resp = MagicMock(status_code=200)
        subhier_resp.json.return_value = {
            "2-0-0-0": 120,
            "2-1-0-0": 0,  # 0 count -> skip
            "10-0-0-0": 50,  # Exceeds max depth -> skip
        }

        mock_get.side_effect = [ept_json_resp, root_hier_resp, subhier_resp]

        target_bbox = (0.0, 0.0, 300.0, 300.0)
        nodes = find_intersecting_ept_nodes(
            ept_base_url="https://mock-ept.s3.amazonaws.com/test",
            target_bbox_3857=target_bbox,
            max_depth=5,
        )

        assert len(nodes) == 1
        assert nodes[0] == ("2-0-0-0", 120)

    @patch("requests.Session.get")
    def test_subhierarchy_404_gracefully_handled(self, mock_get: MagicMock) -> None:
        """HTTP 404 on a sub-hierarchy URL must be skipped without raising an error."""
        root_bounds = [0.0, 0.0, 0.0, 1000.0, 1000.0, 100.0]
        ept_json_resp = MagicMock(status_code=200)
        ept_json_resp.json.return_value = {"bounds": root_bounds}

        root_hier_resp = MagicMock(status_code=200)
        root_hier_resp.json.return_value = {"1-0-0-0": -1}

        subhier_404 = MagicMock(status_code=404)

        mock_get.side_effect = [ept_json_resp, root_hier_resp, subhier_404]

        nodes = find_intersecting_ept_nodes(
            ept_base_url="https://mock-ept.s3.amazonaws.com/test",
            target_bbox_3857=(0.0, 0.0, 500.0, 500.0),
        )
        assert nodes == []

    @patch("requests.Session.get")
    def test_http_error_on_root_raises(self, mock_get: MagicMock) -> None:
        """HTTP 500 on ept.json must raise an HTTPError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_get.return_value = mock_resp

        with pytest.raises(requests.HTTPError):
            find_intersecting_ept_nodes(
                ept_base_url="https://mock-ept.s3.amazonaws.com/test",
                target_bbox_3857=(0.0, 0.0, 100.0, 100.0),
            )


# =============================================================================
# Test Suite: fetch_ept_node_laz
# =============================================================================


class TestFetchEptNodeLaz:
    """Test suite for downloading and parsing single LAZ octree chunks."""

    @patch("requests.Session.get")
    def test_fetch_success_returns_las_data(self, mock_get: MagicMock) -> None:
        """Successful fetch returns populated laspy.LasData object."""
        x = np.array([100.0, 105.0, 110.0])
        y = np.array([200.0, 205.0, 210.0])
        z = np.array([10.0, 12.0, 15.0])
        laz_bytes = make_mock_laz_bytes(x, y, z)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = laz_bytes
        mock_get.return_value = mock_resp

        result = fetch_ept_node_laz(
            node_key="5-10-12-0",
            ept_base_url="https://mock-ept.s3.amazonaws.com/test",
        )

        assert isinstance(result, laspy.LasData)
        assert len(result.points) == 3
        np.testing.assert_allclose(result.x, x, atol=0.01)
        np.testing.assert_allclose(result.y, y, atol=0.01)
        np.testing.assert_allclose(result.z, z, atol=0.01)

    @patch("requests.Session.get")
    def test_fetch_404_returns_none(self, mock_get: MagicMock) -> None:
        """HTTP 404 response must immediately return None without raising."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = fetch_ept_node_laz(
            node_key="nonexistent-key",
            ept_base_url="https://mock-ept.s3.amazonaws.com/test",
        )
        assert result is None
        assert mock_get.call_count == 1

    @patch("time.sleep")
    @patch("requests.Session.get")
    def test_fetch_transient_error_retry_success(
        self, mock_get: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Transient HTTP 503 error triggers retry with backoff and succeeds."""
        laz_bytes = make_mock_laz_bytes(np.array([1.0]), np.array([2.0]), np.array([3.0]))

        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_503.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.content = laz_bytes

        mock_get.side_effect = [resp_503, resp_200]

        result = fetch_ept_node_laz(
            node_key="5-1-1-0",
            ept_base_url="https://mock-ept.s3.amazonaws.com/test",
            max_retries=3,
            backoff_factor=0.1,
        )

        assert isinstance(result, laspy.LasData)
        assert len(result.points) == 1
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(0.1)

    @patch("time.sleep")
    @patch("requests.Session.get")
    def test_fetch_exhausted_retries_returns_none(
        self, mock_get: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Exhausted retries on continuous network timeouts must return None."""
        mock_get.side_effect = TimeoutError("Connection timed out")

        result = fetch_ept_node_laz(
            node_key="5-1-1-0",
            ept_base_url="https://mock-ept.s3.amazonaws.com/test",
            max_retries=3,
            backoff_factor=0.01,
        )

        assert result is None
        assert mock_get.call_count == 3


# =============================================================================
# Test Suite: reproject_points
# =============================================================================


class TestReprojectPoints:
    """Test suite for coordinate reprojection to canonical UTM 18N (EPSG:26918)."""

    def test_reproject_known_coordinates(self) -> None:
        """Verify transformation from EPSG:3857 to EPSG:26918 matches PyProj Transformer."""
        # ALB bridge approximate EPSG:3857 coordinates
        src_x = np.array([-8595680.0, -8595500.0], dtype=np.float64)
        src_y = np.array([4717650.0, 4717800.0], dtype=np.float64)

        tx, ty = reproject_points(src_x, src_y, source_crs=CRS_EPT, target_crs=CRS_METRIC)

        expected_t = pyproj.Transformer.from_crs(CRS_EPT, CRS_METRIC, always_xy=True)
        exp_x, exp_y = expected_t.transform(src_x, src_y)

        np.testing.assert_allclose(tx, exp_x, rtol=1e-7)
        np.testing.assert_allclose(ty, exp_y, rtol=1e-7)
        assert tx.dtype == np.float64
        assert ty.dtype == np.float64

    def test_reproject_empty_arrays(self) -> None:
        """Empty input arrays must return empty float64 arrays without raising."""
        empty = np.array([], dtype=np.float64)
        tx, ty = reproject_points(empty, empty)
        assert len(tx) == 0
        assert len(ty) == 0
        assert tx.dtype == np.float64
        assert ty.dtype == np.float64


# =============================================================================
# Test Suite: ingest_usgs_alb_lidar
# =============================================================================


class TestIngestUsgsAlbLidar:
    """Test suite for American Legion Bridge LiDAR ingestion pipeline."""

    def test_cached_file_bypasses_network(self, tmp_path: Path) -> None:
        """If output_path already exists and force_download is False, return immediately."""
        cached_file = tmp_path / "cached_lidar.laz"
        cached_file.write_bytes(b"dummy_cached_laz_content")

        with patch("topoflow.ingestion.lidar_usgs.find_intersecting_ept_nodes") as mock_find:
            out = ingest_usgs_alb_lidar(
                output_path=cached_file,
                force_download=False,
            )
            assert out == cached_file
            mock_find.assert_not_called()

    def test_missing_bridge_stations_raises_value_error(self, tmp_path: Path) -> None:
        """If stations GeoDataFrame contains no bridge stations in northing range, raise ValueError."""
        # Stations outside ALB bridge northing [4315400, 4316500]
        invalid_stations = gpd.GeoDataFrame(
            {
                "corridor_id": ["alb_i495", "alb_i495"],
                "has_bridge": [True, True],
            },
            geometry=[Point(308000.0, 4310000.0), Point(308000.0, 4320000.0)],
            crs=CRS_METRIC,
        )

        target_file = tmp_path / "bridge.laz"
        with pytest.raises(ValueError, match="No bridge stations found for alb_i495"):
            ingest_usgs_alb_lidar(
                output_path=target_file,
                stations_gdf=invalid_stations,
            )

    def test_missing_stations_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """When stations_gdf is None and default file is absent, raise FileNotFoundError."""
        target_file = tmp_path / "bridge.laz"
        with patch("topoflow.ingestion.lidar_usgs.get_settings") as mock_settings:
            mock_settings.return_value.interim_data_dir = tmp_path / "nonexistent_dir"
            with pytest.raises(FileNotFoundError, match="Stations dataset not found"):
                ingest_usgs_alb_lidar(
                    output_path=target_file,
                    stations_gdf=None,
                )

    @patch("topoflow.ingestion.lidar_usgs.fetch_ept_node_laz")
    @patch("topoflow.ingestion.lidar_usgs.find_intersecting_ept_nodes")
    def test_empty_lidar_returns_raises_runtime_error(
        self,
        mock_find: MagicMock,
        mock_fetch: MagicMock,
        sample_alb_bridge_stations: gpd.GeoDataFrame,
        tmp_path: Path,
    ) -> None:
        """If EPT chunks return no points, raise RuntimeError."""
        mock_find.return_value = [("7-20-30-0", 100)]
        # fetch returns None (e.g. 404 or corrupted)
        mock_fetch.return_value = None

        target_file = tmp_path / "empty.laz"
        with pytest.raises(RuntimeError, match="No LiDAR returns were retrieved"):
            ingest_usgs_alb_lidar(
                output_path=target_file,
                stations_gdf=sample_alb_bridge_stations,
            )

    @patch("topoflow.ingestion.lidar_usgs.fetch_ept_node_laz")
    @patch("topoflow.ingestion.lidar_usgs.find_intersecting_ept_nodes")
    def test_successful_ingestion_end_to_end(
        self,
        mock_find: MagicMock,
        mock_fetch: MagicMock,
        sample_alb_bridge_stations: gpd.GeoDataFrame,
        tmp_path: Path,
    ) -> None:
        """Full mock pipeline: discovers nodes, downloads LAZ chunk, reprojects, and writes LAZ."""
        mock_find.return_value = [("7-20-30-0", 4)]

        # Bridge stations in fixture center around UTM (308200, 4315900)
        # Transform (308200, 4315900) to EPSG:3857 so raw points land inside buffer
        t_to_3857 = pyproj.Transformer.from_crs(CRS_METRIC, CRS_EPT, always_xy=True)
        center_x_3857, center_y_3857 = t_to_3857.transform(308200.0, 4315900.0)

        raw_x = np.array([center_x_3857, center_x_3857 + 10.0, center_x_3857 - 10.0])
        raw_y = np.array([center_y_3857, center_y_3857 + 10.0, center_y_3857 - 10.0])
        raw_z = np.array([55.2, 56.1, 54.8])
        raw_cls = np.array([2, 17, 2], dtype=np.uint8)
        raw_intensity = np.array([80, 120, 95], dtype=np.uint16)
        raw_returns = np.array([1, 1, 1], dtype=np.uint8)

        chunk_header = laspy.LasHeader(point_format=3, version="1.4")
        chunk_header.offsets = [
            float(np.floor(np.min(raw_x))),
            float(np.floor(np.min(raw_y))),
            float(np.floor(np.min(raw_z))),
        ]
        chunk_header.scales = [0.01, 0.01, 0.01]
        chunk_las = laspy.LasData(chunk_header)
        chunk_las.x = raw_x
        chunk_las.y = raw_y
        chunk_las.z = raw_z
        chunk_las.classification = raw_cls
        chunk_las.intensity = raw_intensity
        chunk_las.return_number = raw_returns

        mock_fetch.return_value = chunk_las

        target_file = tmp_path / "alb_lidar_out.laz"
        persisted_path = ingest_usgs_alb_lidar(
            output_path=target_file,
            stations_gdf=sample_alb_bridge_stations,
            buffer_m=120.0,
            force_download=True,
        )

        assert persisted_path == target_file
        assert target_file.exists()

        # Read back and inspect written LAZ point cloud
        written_las = laspy.read(target_file)
        assert len(written_las.points) == 3
        # Check coordinates are in UTM Zone 18N (e.g. northing ~ 4.315e6)
        assert 4315000.0 < np.mean(written_las.y) < 4317000.0
        assert 307000.0 < np.mean(written_las.x) < 309000.0
        # Check Z elevation values
        np.testing.assert_allclose(written_las.z, raw_z, atol=0.01)
        # Check classifications
        np.testing.assert_array_equal(written_las.classification, raw_cls)
