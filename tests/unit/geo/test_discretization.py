"""Unit tests for strict 100-meter spatial discretization and horizontal curvature profiling.

Validates metric reprojection, equidistant chainage sampling, Three-Point Menger osculating
circle curvature calculations against analytical circular arcs, heading azimuths,
station tagging, and Parquet serialization.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from topoflow.constants import (
    CRS_METRIC,
    CRS_WGS84,
    MAX_STRAIGHT_RADIUS_METERS,
    SPATIAL_RESOLUTION_METERS,
)
from topoflow.geo.discretization import (
    DiscretizationError,
    SpatialDiscretizer,
    assign_station_way_attributes,
    compute_heading,
    generate_segment_id,
    menger_curvature,
    project_linestring,
    resample_carriageway_stations,
)


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to tests/fixtures directory."""
    return Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture
def sample_corridors_gdf(fixtures_dir: Path) -> gpd.GeoDataFrame:
    """Load sample corridors GeoDataFrame from offline fixture."""
    geojson_path = fixtures_dir / "osm_corridors_sample.geojson"
    assert geojson_path.exists(), f"Missing fixture at {geojson_path}"
    return gpd.read_file(geojson_path)


class TestProjectLineString:
    """Tests for metric coordinate transformation."""

    def test_project_linestring_valid(self) -> None:
        """Verify transformation from EPSG:4326 to EPSG:26918 yields valid metric coordinates."""
        line_wgs84 = LineString([(-77.20, 38.96), (-77.18, 38.97)])
        line_metric = project_linestring(line_wgs84, source_crs=CRS_WGS84, target_crs=CRS_METRIC)

        assert isinstance(line_metric, LineString)
        assert len(line_metric.coords) == 2

        # UTM Zone 18N coordinates for DC/VA/MD are approximately Easting ~300,000m, Northing ~4,315,000m
        x0, y0 = line_metric.coords[0]
        assert 280000.0 < x0 < 350000.0
        assert 4250000.0 < y0 < 4350000.0
        assert line_metric.length > 1000.0  # ~2 km distance

    def test_project_linestring_empty_raises(self) -> None:
        """Verify empty LineString raises DiscretizationError."""
        with pytest.raises(DiscretizationError, match="empty LineString"):
            project_linestring(LineString([]))


class TestComputeHeading:
    """Tests for forward tangent azimuth heading computation."""

    def test_segment_id_formatting(self) -> None:
        """Verify canonical segment_id formatting."""
        seg_id = generate_segment_id("alb_i495", "inner_loop", 12)
        assert seg_id == "alb_i495_inner_loop_s0012"

    def test_cardinal_directions(self) -> None:
        """Verify heading azimuths along cardinal and intercardinal axes."""
        # Due North
        assert compute_heading((0.0, 0.0), (0.0, 100.0)) == 0.0
        # Due East
        assert compute_heading((0.0, 0.0), (100.0, 0.0)) == 90.0
        # Due South
        assert compute_heading((0.0, 0.0), (0.0, -100.0)) == 180.0
        # Due West
        assert compute_heading((0.0, 0.0), (-100.0, 0.0)) == 270.0
        # Northeast (45 degrees)
        assert compute_heading((0.0, 0.0), (100.0, 100.0)) == 45.0
        # Northwest (315 degrees)
        assert compute_heading((0.0, 0.0), (-100.0, 100.0)) == 315.0


class TestMengerCurvature:
    """Tests for Three-Point Menger osculating circle curvature formulation."""

    def test_analytical_circular_arc(self) -> None:
        """Validate calculated curvature on a known circle (R=500m) matches within +/-1%."""
        target_r = 500.0
        expected_kappa = 1.0 / target_r

        # Generate three points on a circle of radius target_r
        # Arc angle theta = 10 degrees (~87 meters apart along arc)
        theta1 = math.radians(0.0)
        theta2 = math.radians(10.0)
        theta3 = math.radians(20.0)

        p1 = (target_r * math.cos(theta1), target_r * math.sin(theta1))
        p2 = (target_r * math.cos(theta2), target_r * math.sin(theta2))
        p3 = (target_r * math.cos(theta3), target_r * math.sin(theta3))

        kappa, radius = menger_curvature(p1, p2, p3)

        assert pytest.approx(kappa, rel=0.001) == expected_kappa
        assert pytest.approx(radius, rel=0.001) == target_r

    def test_straight_line_collinear(self) -> None:
        """Verify perfectly collinear points return kappa=0 and cap radius at MAX_STRAIGHT_RADIUS_METERS."""
        p1 = (0.0, 0.0)
        p2 = (100.0, 0.0)
        p3 = (200.0, 0.0)

        kappa, radius = menger_curvature(p1, p2, p3)
        assert kappa == 0.0
        assert radius == MAX_STRAIGHT_RADIUS_METERS

    def test_sub_threshold_mild_curvature_capped(self) -> None:
        """Verify curves with R > 10,000m (kappa < 1e-4) are cleanly capped at max radius."""
        # Circle with R = 20,000m
        r_large = 20000.0
        p1 = (r_large * math.cos(0.0), r_large * math.sin(0.0))
        p2 = (r_large * math.cos(0.005), r_large * math.sin(0.005))
        p3 = (r_large * math.cos(0.010), r_large * math.sin(0.010))

        kappa, radius = menger_curvature(p1, p2, p3)
        assert kappa == 0.0
        assert radius == MAX_STRAIGHT_RADIUS_METERS

    def test_degenerate_zero_distance_points(self) -> None:
        """Verify coincident points do not trigger ZeroDivisionError."""
        p1 = (100.0, 100.0)
        p2 = (100.0, 100.0)
        p3 = (200.0, 200.0)

        kappa, radius = menger_curvature(p1, p2, p3)
        assert kappa == 0.0
        assert radius == MAX_STRAIGHT_RADIUS_METERS


class TestStationResampling:
    """Tests for equidistant chainage resampling along metric LineStrings."""

    def test_equidistant_spacing_and_attributes(self) -> None:
        """Verify stations are sampled at exact 100.0m intervals with correct attributes."""
        # 1 km straight line in metric coordinates
        line_metric = LineString([(300000.0, 4310000.0), (301000.0, 4310000.0)])
        stations = resample_carriageway_stations(
            line_metric=line_metric,
            corridor_id="alb_i495",
            direction="inner_loop",
            interval_m=100.0,
        )

        # 1000m line -> 11 stations (0m, 100m, ..., 1000m)
        assert len(stations) == 11

        for i, st in enumerate(stations):
            assert st["station_index"] == i
            assert st["chainage_m"] == float(i * 100.0)
            assert st["segment_id"] == f"alb_i495_inner_loop_s{i:04d}"
            assert st["corridor_id"] == "alb_i495"
            assert st["direction"] == "inner_loop"
            assert st["radius_m"] == MAX_STRAIGHT_RADIUS_METERS
            assert st["curvature_kappa"] == 0.0
            assert st["heading_deg"] == 90.0  # Eastward heading
            assert isinstance(st["geometry"], Point)

            # Test adjacent inter-station Euclidean distance
            if i > 0:
                prev_st = stations[i - 1]
                dx = st["easting"] - prev_st["easting"]
                dy = st["northing"] - prev_st["northing"]
                dist = math.hypot(dx, dy)
                assert pytest.approx(dist, abs=0.1) == 100.0

    def test_zero_length_line_raises(self) -> None:
        """Verify zero-length line raises DiscretizationError."""
        line_zero = LineString([(100.0, 100.0), (100.0, 100.0)])
        with pytest.raises(DiscretizationError, match="non-positive length"):
            resample_carriageway_stations(
                line_metric=line_zero,
                corridor_id="test",
                direction="nb",
            )


class TestSpatialDiscretizerEndToEnd:
    """End-to-end integration tests using offline fixtures."""

    def test_discretize_corridors_dataframe(self, sample_corridors_gdf: gpd.GeoDataFrame) -> None:
        """Verify full discretization across multiple corridor carriageways."""
        discretizer = SpatialDiscretizer(interval_m=SPATIAL_RESOLUTION_METERS)
        stations_gdf = discretizer.discretize_corridors_dataframe(sample_corridors_gdf)

        assert not stations_gdf.empty
        assert len(stations_gdf) > 100  # Multiple kilometers of corridors

        # Verify required column schema
        expected_cols = {
            "segment_id",
            "corridor_id",
            "direction",
            "station_index",
            "chainage_m",
            "easting",
            "northing",
            "longitude",
            "latitude",
            "heading_deg",
            "curvature_kappa",
            "radius_m",
            "geometry",
        }
        assert expected_cols.issubset(set(stations_gdf.columns))

        # Verify segment_id uniqueness
        assert stations_gdf["segment_id"].is_unique

        # Verify corridors present
        corridors = stations_gdf["corridor_id"].unique()
        assert "alb_i495" in corridors

        # Verify ALB study corridor has expected physical characteristics
        alb_df = stations_gdf[stations_gdf["corridor_id"] == "alb_i495"]
        assert len(alb_df) > 30

        # Verify no NaN or infinite values in geometric columns
        assert not stations_gdf["curvature_kappa"].isna().any()
        assert not stations_gdf["radius_m"].isna().any()
        assert not stations_gdf["heading_deg"].isna().any()

    def test_export_and_read_parquet(
        self,
        sample_corridors_gdf: gpd.GeoDataFrame,
        tmp_path: Path,
    ) -> None:
        """Verify stations matrix exports cleanly to Parquet and reloads without loss."""
        discretizer = SpatialDiscretizer()
        stations_gdf = discretizer.discretize_corridors_dataframe(sample_corridors_gdf)

        export_path = tmp_path / "stations_test.parquet"
        out_file = discretizer.export(stations_gdf, export_path)

        assert out_file.exists()
        reloaded_gdf = gpd.read_parquet(out_file)

        assert len(reloaded_gdf) == len(stations_gdf)
        assert list(reloaded_gdf.columns) == list(stations_gdf.columns)
        assert reloaded_gdf["segment_id"].equals(stations_gdf["segment_id"])

    def test_empty_dataframe_raises(self) -> None:
        """Verify empty input GeoDataFrame raises DiscretizationError."""
        discretizer = SpatialDiscretizer()
        with pytest.raises(DiscretizationError, match="empty GeoDataFrame"):
            discretizer.discretize_corridors_dataframe(gpd.GeoDataFrame())

    def test_carriageway_columns_not_propagated(
        self, sample_corridors_gdf: gpd.GeoDataFrame
    ) -> None:
        """Carriageway-level QA columns never leak into station rows."""
        stations_gdf = SpatialDiscretizer().discretize_corridors_dataframe(sample_corridors_gdf)
        for col in ("way_intervals", "max_junction_gap_m", "length_meters", "is_continuous"):
            assert col not in stations_gdf.columns


# =============================================================================
# Per-station OSM way attributes
# =============================================================================

BRIDGE_START_M = 3650.0
BRIDGE_END_M = 4090.0


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _three_way_intervals() -> list[dict]:
    """Approach / bridge / approach intervals along a 5 km carriageway (metres)."""
    return [
        {
            "start_m": 0.0,
            "end_m": BRIDGE_START_M,
            "bridge": False,
            "lanes": 4,
            "maxspeed": "55 mph",
            "way_id": 1,
        },
        {
            "start_m": BRIDGE_START_M,
            "end_m": BRIDGE_END_M,
            "bridge": True,
            "lanes": 5,
            "maxspeed": "65 mph",
            "way_id": 2,
        },
        {
            "start_m": BRIDGE_END_M,
            "end_m": 5000.0,
            "bridge": False,
            "lanes": None,
            "maxspeed": None,
            "way_id": 3,
        },
    ]


class TestAssignStationWayAttributes:
    """Tests for interval lookup of has_bridge / lanes / maxspeed per station chainage."""

    def test_values_inherited_and_aligned(self) -> None:
        chainages = np.array([100.0, 3700.0, 4000.0, 4500.0])
        out = assign_station_way_attributes(chainages, _three_way_intervals())
        assert set(out.keys()) == {"has_bridge", "lanes", "maxspeed"}
        for values in out.values():
            assert len(values) == len(chainages)
        assert [bool(v) for v in out["has_bridge"]] == [False, True, True, False]
        assert out["lanes"] == [4, 5, 5, None]
        assert out["maxspeed"] == ["55 mph", "65 mph", "65 mph", None]

    def test_empty_inputs(self) -> None:
        out = assign_station_way_attributes(np.array([0.0, 100.0, 200.0]), [])
        assert [bool(v) for v in out["has_bridge"]] == [False, False, False]
        assert out["lanes"] == [None, None, None]
        assert out["maxspeed"] == [None, None, None]

        out = assign_station_way_attributes(np.array([], dtype=float), _three_way_intervals())
        assert all(list(out[k]) == [] for k in ("has_bridge", "lanes", "maxspeed"))

    def test_bounds_inclusive(self) -> None:
        intervals = [
            {
                "start_m": 100.0,
                "end_m": 200.0,
                "bridge": True,
                "lanes": 3,
                "maxspeed": "50 mph",
                "way_id": 9,
            }
        ]
        out = assign_station_way_attributes(np.array([99.9, 100.0, 200.0, 200.1]), intervals)
        assert [bool(v) for v in out["has_bridge"]] == [False, True, True, False]
        assert out["lanes"] == [None, 3, 3, None]
        assert out["maxspeed"] == [None, "50 mph", "50 mph", None]

    def test_uncovered_stations_default_false_none(self) -> None:
        intervals = [
            {
                "start_m": 1000.0,
                "end_m": 1500.0,
                "bridge": True,
                "lanes": 4,
                "maxspeed": "65 mph",
                "way_id": 1,
            }
        ]
        out = assign_station_way_attributes(np.array([0.0, 500.0, 2000.0]), intervals)
        assert [bool(v) for v in out["has_bridge"]] == [False, False, False]
        assert out["lanes"] == [None, None, None]
        assert out["maxspeed"] == [None, None, None]

    def test_bridge_only_from_bridge_interval(self) -> None:
        """A shared-boundary station is a bridge only if a bridge interval covers it."""
        out = assign_station_way_attributes(
            np.array([BRIDGE_START_M, BRIDGE_END_M, 3000.0]), _three_way_intervals()
        )
        assert [bool(v) for v in out["has_bridge"]] == [True, True, False]

    def test_unsorted_intervals(self) -> None:
        out = assign_station_way_attributes(
            np.array([100.0, 3800.0]), list(reversed(_three_way_intervals()))
        )
        assert [bool(v) for v in out["has_bridge"]] == [False, True]
        assert out["lanes"] == [4, 5]


class TestResampleWithWayIntervals:
    """Tests for per-station attribute precedence inside resample_carriageway_stations."""

    @pytest.fixture
    def line_5km(self) -> LineString:
        return LineString([(310000.0, 4310000.0), (310000.0, 4315000.0)])

    def test_way_intervals_take_precedence(self, line_5km: LineString) -> None:
        stations = resample_carriageway_stations(
            line_metric=line_5km,
            corridor_id="alb_i495",
            direction="inner_loop",
            extra_metadata={
                "has_bridge": True,
                "lanes": 9,
                "maxspeed": "99 mph",
                "route": "I-495",
            },
            way_intervals=_three_way_intervals(),
        )
        assert len(stations) == 51
        by_chainage = {st["chainage_m"]: st for st in stations}
        bridge_chainages = sorted(c for c, st in by_chainage.items() if st["has_bridge"])
        assert bridge_chainages == [3700.0, 3800.0, 3900.0, 4000.0]
        assert by_chainage[100.0]["lanes"] == 4
        assert by_chainage[100.0]["maxspeed"] == "55 mph"
        assert by_chainage[3800.0]["lanes"] == 5
        assert by_chainage[4500.0]["lanes"] is None
        assert by_chainage[4500.0]["maxspeed"] is None
        # Non-way metadata still propagates
        assert all(st["route"] == "I-495" for st in stations)

    def test_without_way_intervals_metadata_unchanged(self, line_5km: LineString) -> None:
        meta = {"has_bridge": True, "lanes": 5, "maxspeed": "65 mph", "route": "I-495"}
        stations = resample_carriageway_stations(
            line_metric=line_5km,
            corridor_id="alb_i495",
            direction="inner_loop",
            extra_metadata=meta,
        )
        for st in stations:
            assert st["has_bridge"] is True
            assert st["lanes"] == 5
            assert st["maxspeed"] == "65 mph"
            assert st["route"] == "I-495"

    def test_empty_way_intervals_yields_false_none(self, line_5km: LineString) -> None:
        stations = resample_carriageway_stations(
            line_metric=line_5km,
            corridor_id="alb_i495",
            direction="inner_loop",
            extra_metadata={"has_bridge": True, "lanes": 5, "maxspeed": "65 mph"},
            way_intervals=[],
        )
        assert not any(st["has_bridge"] for st in stations)
        assert all(st["lanes"] is None for st in stations)
        assert all(st["maxspeed"] is None for st in stations)


class TestDiscretizeWithWayIntervalsColumn:
    """End-to-end per-station attribution from a JSON ``way_intervals`` column."""

    @pytest.fixture
    def carriageway_gdf(self) -> gpd.GeoDataFrame:
        line_metric = LineString([(310000.0, 4310000.0), (310000.0, 4315000.0)])
        line_wgs84 = gpd.GeoSeries([line_metric], crs=CRS_METRIC).to_crs(CRS_WGS84).iloc[0]
        return gpd.GeoDataFrame(
            {
                "corridor_id": ["alb_i495"],
                "direction": ["inner_loop"],
                "route": ["I-495"],
                "lanes": [5],
                "maxspeed": ["65 mph"],
                "has_bridge": [True],
                "length_meters": [5000.0],
                "max_junction_gap_m": [0.0],
                "is_continuous": [True],
                "way_intervals": [json.dumps(_three_way_intervals())],
            },
            geometry=[line_wgs84],
            crs=CRS_WGS84,
        )

    def test_per_station_bridge_flags(self, carriageway_gdf: gpd.GeoDataFrame) -> None:
        stations = SpatialDiscretizer(interval_m=100.0).discretize_corridors_dataframe(
            carriageway_gdf
        )
        assert len(stations) >= 50
        is_bridge = stations["has_bridge"].astype(bool)
        bridge_chainages = sorted(stations.loc[is_bridge, "chainage_m"].round(3).tolist())
        assert bridge_chainages == [3700.0, 3800.0, 3900.0, 4000.0]
        assert int((~is_bridge).sum()) == len(stations) - 4

    def test_per_station_lanes_and_maxspeed(self, carriageway_gdf: gpd.GeoDataFrame) -> None:
        stations = SpatialDiscretizer().discretize_corridors_dataframe(carriageway_gdf)
        at = stations.set_index(stations["chainage_m"].round(3))
        assert at.loc[100.0, "lanes"] == 4
        assert at.loc[100.0, "maxspeed"] == "55 mph"
        assert at.loc[3800.0, "lanes"] == 5
        assert at.loc[3800.0, "maxspeed"] == "65 mph"
        assert _is_missing(at.loc[4500.0, "lanes"])
        assert _is_missing(at.loc[4500.0, "maxspeed"])

    def test_carriageway_columns_dropped(self, carriageway_gdf: gpd.GeoDataFrame) -> None:
        stations = SpatialDiscretizer().discretize_corridors_dataframe(carriageway_gdf)
        for col in ("way_intervals", "max_junction_gap_m", "length_meters", "is_continuous"):
            assert col not in stations.columns
        assert (stations["route"] == "I-495").all()


class TestDiscretizeWithoutWayIntervals:
    """Without per-way intervals, carriageway-level bridge/lane/speed must not be smeared."""

    @staticmethod
    def _carriageway(way_intervals: object = "absent") -> gpd.GeoDataFrame:
        line_metric = LineString([(310000.0, 4310000.0), (310000.0, 4312000.0)])
        line_wgs84 = gpd.GeoSeries([line_metric], crs=CRS_METRIC).to_crs(CRS_WGS84).iloc[0]
        data: dict[str, list] = {
            "corridor_id": ["alb_i495"],
            "direction": ["inner_loop"],
            "route": ["I-495"],
            "lanes": [5],
            "maxspeed": ["65 mph"],
            "has_bridge": [True],
        }
        if way_intervals != "absent":
            data["way_intervals"] = [way_intervals]
        return gpd.GeoDataFrame(data, geometry=[line_wgs84], crs=CRS_WGS84)

    @pytest.mark.parametrize("way_intervals", ["absent", None])
    def test_station_attributes_null(self, way_intervals: object) -> None:
        stations = SpatialDiscretizer().discretize_corridors_dataframe(
            self._carriageway(way_intervals)
        )
        assert len(stations) == 21
        for col in ("has_bridge", "lanes", "maxspeed"):
            assert col in stations.columns
            assert all(_is_missing(v) for v in stations[col].tolist()), col
        # Non-way metadata still propagates.
        assert (stations["route"] == "I-495").all()
        assert "way_intervals" not in stations.columns

    def test_mixed_rows_with_and_without_intervals(self) -> None:
        with_iv = self._carriageway(
            json.dumps(
                [
                    {
                        "start_m": 500.0,
                        "end_m": 700.0,
                        "bridge": True,
                        "lanes": 3,
                        "maxspeed": "50 mph",
                        "way_id": 1,
                    }
                ]
            )
        )
        without_iv = self._carriageway(None)
        without_iv["direction"] = ["outer_loop"]
        gdf = gpd.GeoDataFrame(pd.concat([with_iv, without_iv], ignore_index=True), crs=CRS_WGS84)
        stations = SpatialDiscretizer().discretize_corridors_dataframe(gdf)
        inner = stations[stations["direction"] == "inner_loop"]
        outer = stations[stations["direction"] == "outer_loop"]
        inner_bridge = sorted(
            inner.loc[inner["has_bridge"].fillna(False).astype(bool), "chainage_m"].tolist()
        )
        assert inner_bridge == [500.0, 600.0, 700.0]
        assert all(_is_missing(v) for v in outer["has_bridge"].tolist())
        assert all(_is_missing(v) for v in outer["lanes"].tolist())
