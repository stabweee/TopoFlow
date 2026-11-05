"""Pytest configuration and session-wide fixtures for TopoFlow."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute path to the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def fixtures_dir(project_root: Path) -> Path:
    """Return the path to the test fixtures directory."""
    return project_root / "tests" / "fixtures"


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Return the path to the data directory."""
    return project_root / "data"


@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    """Return the path to the configs directory."""
    return project_root / "configs"


@pytest.fixture(scope="session")
def outputs_dir(project_root: Path) -> Path:
    """Return the path to the outputs directory."""
    return project_root / "outputs"


@pytest.fixture(scope="session")
def alb_corridor_bounds() -> dict[str, float]:
    """Bounding box for the I-495 American Legion Bridge study corridor (EPSG:4326)."""
    return {
        "min_lat": 38.95,
        "max_lat": 39.00,
        "min_lon": -77.23,
        "max_lon": -77.18,
    }


@pytest.fixture(scope="session")
def master_metric_crs() -> str:
    """Master metric analytical projection spanning MD and VA."""
    return "EPSG:26918"


@pytest.fixture(scope="session")
def wgs84_crs() -> str:
    """Standard unprojected geodetic coordinate reference system."""
    return "EPSG:4326"
