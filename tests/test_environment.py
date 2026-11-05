"""Environment smoke tests verifying all critical dependencies and path fixtures."""

from pathlib import Path


def test_tabular_and_storage_imports():
    """Verify high-performance tabular engines are importable."""
    import duckdb
    import polars as pl
    import pyarrow as pa

    assert pl.__version__ is not None
    assert duckdb.__version__ is not None
    assert pa.__version__ is not None

    # Test basic in-memory execution
    df = pl.DataFrame({"station_id": [1, 2], "speed_mph": [60.5, 45.2]})
    assert df.height == 2

    conn = duckdb.connect()
    res = conn.execute("SELECT 1 + 1 AS two").fetchone()
    assert res[0] == 2


def test_geospatial_and_lidar_imports():
    """Verify geospatial projection and LiDAR parsing libraries are importable."""
    import geopandas as gpd
    import laspy
    import pyproj
    import rasterio
    from shapely.geometry import Point

    assert gpd.__version__ is not None
    assert laspy.__version__ is not None
    assert pyproj.__version__ is not None
    assert rasterio.__version__ is not None

    # Test basic geometry creation
    pt = Point(310000.0, 4315000.0)
    assert pt.is_valid
    assert not pt.is_empty


def test_econometrics_and_stats_imports():
    """Verify statistical, econometric, and causal estimation libraries are importable."""
    import linearmodels
    import numpy as np
    import scipy
    import statsmodels

    assert linearmodels.__version__ is not None
    assert np.__version__ is not None
    assert scipy.__version__ is not None
    assert statsmodels.__version__ is not None


def test_configuration_and_utilities_imports():
    """Verify settings validation, configuration parsers, and plotting libraries."""
    import matplotlib
    import pydantic
    import seaborn
    import tqdm
    import yaml

    assert pydantic.__version__ is not None
    assert yaml.__version__ is not None
    assert matplotlib.__version__ is not None
    assert seaborn.__version__ is not None
    assert tqdm.__version__ is not None


def test_conftest_fixtures(
    project_root: Path,
    data_dir: Path,
    configs_dir: Path,
    alb_corridor_bounds: dict[str, float],
    master_metric_crs: str,
):
    """Verify session fixtures and repository path resolution."""
    assert project_root.exists()
    assert (project_root / "pyproject.toml").exists()
    assert data_dir.exists()
    assert configs_dir.exists()
    assert alb_corridor_bounds["min_lat"] < alb_corridor_bounds["max_lat"]
    assert master_metric_crs == "EPSG:26918"
