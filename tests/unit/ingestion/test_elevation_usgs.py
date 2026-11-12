"""Unit tests for USGS 3DEP bare-earth elevation ingestion via EPQS API.

Validates single-point elevation fetching from the USGS Elevation Point Query Service
(EPQS), response payload parsing, retry backoff on HTTP 429/503 and network disconnects,
error handling on invalid/nodata values, station coordinate deduplication, concurrent
batch ingestion, Parquet caching, and input interoperability with Polars and GeoPandas.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import polars as pl
import pytest
import requests
from shapely.geometry import Point

from topoflow.ingestion.elevation_usgs import (
    fetch_epqs_elevation,
    ingest_usgs_bare_earth_elevations,
)

# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def sample_epqs_response() -> dict[str, object]:
    """Standard valid USGS EPQS JSON response payload."""
    return {
        "location": {"x": -77.2092, "y": 38.9697},
        "value": "43.276",
        "resolution": 1,
    }


@pytest.fixture
def sample_polars_stations() -> pl.DataFrame:
    """Sample highway corridor stations in Polars format."""
    return pl.DataFrame(
        {
            "segment_id": ["seg_001", "seg_002", "seg_003"],
            "latitude": [38.9697, 38.9710, 38.9725],
            "longitude": [-77.2092, -77.2105, -77.2120],
        }
    )


@pytest.fixture
def sample_geodataframe_stations() -> gpd.GeoDataFrame:
    """Sample corridor stations in GeoPandas WGS84 format."""
    return gpd.GeoDataFrame(
        {
            "segment_id": ["seg_001", "seg_002"],
            "latitude": [38.9697, 38.9710],
            "longitude": [-77.2092, -77.2105],
        },
        geometry=[Point(-77.2092, 38.9697), Point(-77.2105, 38.9710)],
        crs="EPSG:4326",
    )


# =============================================================================
# Test Suite: fetch_epqs_elevation
# =============================================================================


class TestFetchEpqsElevation:
    """Test suite for single-point USGS EPQS elevation queries."""

    @patch("requests.Session.get")
    def test_fetch_success_string_value(
        self, mock_get: MagicMock, sample_epqs_response: dict[str, object]
    ) -> None:
        """Verify successful parsing of elevation when returned as a numeric string."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_epqs_response
        mock_get.return_value = mock_resp

        elev = fetch_epqs_elevation(lat=38.9697, lon=-77.2092)
        assert isinstance(elev, float)
        assert elev == pytest.approx(43.276)

    @patch("requests.Session.get")
    def test_fetch_success_numeric_float_value(self, mock_get: MagicMock) -> None:
        """Verify successful parsing of elevation when returned as a raw JSON float."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "location": {"x": -77.24, "y": 38.68},
            "value": 52.84,
            "resolution": 1,
        }
        mock_get.return_value = mock_resp

        elev = fetch_epqs_elevation(lat=38.68, lon=-77.24)
        assert isinstance(elev, float)
        assert elev == pytest.approx(52.84)

    @patch("requests.Session.get")
    def test_fetch_query_parameters_and_url(self, mock_get: MagicMock) -> None:
        """Verify URL and query parameters conform to USGS EPQS endpoint spec."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"value": "25.0"}
        mock_get.return_value = mock_resp

        lat, lon = 38.95, -77.20
        fetch_epqs_elevation(lat=lat, lon=lon)

        mock_get.assert_called_once()
        call_args, call_kwargs = mock_get.call_args
        # Either URL query string contains params or params dict contains x, y, units
        url = call_args[0] if call_args else call_kwargs.get("url", "")
        params = call_kwargs.get("params", {})

        if params:
            assert float(params.get("x", params.get("lon"))) == pytest.approx(lon)
            assert float(params.get("y", params.get("lat"))) == pytest.approx(lat)
            assert params.get("units", "").lower() in ("meters", "meter")
        else:
            assert f"x={lon}" in url or f"x={float(lon)}" in url
            assert f"y={lat}" in url or f"y={float(lat)}" in url
            assert "units=Meters" in url or "units=meters" in url

    def test_fetch_with_custom_session(self) -> None:
        """Verify custom session object is utilized when provided."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"value": "100.5"}
        mock_session.get.return_value = mock_resp

        elev = fetch_epqs_elevation(lat=38.9, lon=-77.2, session=mock_session)
        assert elev == pytest.approx(100.5)
        mock_session.get.assert_called_once()

    @patch("requests.Session.get")
    def test_fetch_retry_on_503_service_unavailable(self, mock_get: MagicMock) -> None:
        """Verify backoff retry on HTTP 503 before succeeding."""
        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_503.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"value": "45.0"}

        mock_get.side_effect = [resp_503, resp_200]

        elev = fetch_epqs_elevation(lat=38.9, lon=-77.2, max_retries=3)
        assert elev == pytest.approx(45.0)
        assert mock_get.call_count == 2

    @patch("requests.Session.get")
    def test_fetch_retry_on_429_rate_limit(self, mock_get: MagicMock) -> None:
        """Verify backoff retry on HTTP 429 Too Many Requests."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.raise_for_status.side_effect = requests.HTTPError("429 Rate Limit")

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"value": "48.2"}

        mock_get.side_effect = [resp_429, resp_200]

        elev = fetch_epqs_elevation(lat=38.9, lon=-77.2, max_retries=3)
        assert elev == pytest.approx(48.2)
        assert mock_get.call_count == 2

    @patch("requests.Session.get")
    def test_fetch_retry_on_connection_error(self, mock_get: MagicMock) -> None:
        """Verify backoff retry on transient network connection drop."""
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"value": "60.0"}

        mock_get.side_effect = [requests.ConnectionError("Connection dropped"), resp_200]

        elev = fetch_epqs_elevation(lat=38.9, lon=-77.2, max_retries=3)
        assert elev == pytest.approx(60.0)
        assert mock_get.call_count == 2

    @patch("requests.Session.get")
    def test_fetch_exhausted_retries_raises_error(self, mock_get: MagicMock) -> None:
        """Verify ConnectionError or RuntimeError raised when all retries fail."""
        mock_get.side_effect = requests.ConnectionError("Host unreachable")

        with pytest.raises((ConnectionError, RuntimeError)):
            fetch_epqs_elevation(lat=38.9, lon=-77.2, max_retries=2)

    @patch("requests.Session.get")
    def test_fetch_missing_value_field_raises_value_error(self, mock_get: MagicMock) -> None:
        """Verify ValueError raised when response lacks 'value' key."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"location": {"x": -77.2, "y": 38.9}}
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="(?i)value|elevation"):
            fetch_epqs_elevation(lat=38.9, lon=-77.2)

    @patch("requests.Session.get")
    def test_fetch_null_or_nodata_value_raises_value_error(self, mock_get: MagicMock) -> None:
        """Verify ValueError raised on null, empty string, or nodata (-1000000)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        # Case 1: value is None
        mock_resp.json.return_value = {"value": None}
        mock_get.return_value = mock_resp
        with pytest.raises(ValueError):
            fetch_epqs_elevation(lat=38.9, lon=-77.2)

        # Case 2: value is nodata string "-1000000"
        mock_resp.json.return_value = {"value": "-1000000"}
        mock_get.return_value = mock_resp
        with pytest.raises(ValueError):
            fetch_epqs_elevation(lat=38.9, lon=-77.2)

        # Case 3: non-numeric string
        mock_resp.json.return_value = {"value": "invalid_elevation"}
        mock_get.return_value = mock_resp
        with pytest.raises(ValueError):
            fetch_epqs_elevation(lat=38.9, lon=-77.2)


# =============================================================================
# Test Suite: ingest_usgs_bare_earth_elevations
# =============================================================================


class TestIngestUsgsBareEarthElevations:
    """Test suite for batch USGS 3DEP elevation ingestion."""

    @patch("topoflow.ingestion.elevation_usgs.fetch_epqs_elevation")
    def test_ingest_from_polars_dataframe(
        self,
        mock_fetch: MagicMock,
        sample_polars_stations: pl.DataFrame,
    ) -> None:
        """Verify ingestion from a Polars DataFrame returns expected schema."""
        mock_fetch.side_effect = [43.0, 44.5, 46.0]

        result = ingest_usgs_bare_earth_elevations(
            stations=sample_polars_stations,
            max_workers=1,
        )

        assert isinstance(result, pl.DataFrame)
        required_cols = {
            "segment_id",
            "latitude",
            "longitude",
            "elevation_usgs_m",
            "data_source",
        }
        assert required_cols.issubset(set(result.columns))
        assert len(result) == len(sample_polars_stations)
        assert result["elevation_usgs_m"].to_list() == [43.0, 44.5, 46.0]
        assert all("USGS" in s for s in result["data_source"].to_list())

    @patch("topoflow.ingestion.elevation_usgs.fetch_epqs_elevation")
    def test_ingest_from_geodataframe(
        self,
        mock_fetch: MagicMock,
        sample_geodataframe_stations: gpd.GeoDataFrame,
    ) -> None:
        """Verify ingestion from a GeoPandas GeoDataFrame produces a Polars DataFrame."""
        mock_fetch.side_effect = [40.0, 42.0]

        result = ingest_usgs_bare_earth_elevations(
            stations=sample_geodataframe_stations,
            max_workers=1,
        )

        assert isinstance(result, pl.DataFrame)
        assert "elevation_usgs_m" in result.columns
        assert len(result) == 2
        assert result["elevation_usgs_m"].to_list() == [40.0, 42.0]

    @patch("topoflow.ingestion.elevation_usgs.fetch_epqs_elevation")
    def test_coordinate_deduplication(self, mock_fetch: MagicMock) -> None:
        """Verify duplicate lat/lon coordinates are queried only once."""
        stations_with_dups = pl.DataFrame(
            {
                "segment_id": ["seg_A", "seg_B", "seg_C", "seg_D"],
                "latitude": [38.9697, 38.9697, 38.9710, 38.9710],
                "longitude": [-77.2092, -77.2092, -77.2105, -77.2105],
            }
        )

        mock_fetch.side_effect = [43.276, 45.100]

        result = ingest_usgs_bare_earth_elevations(
            stations=stations_with_dups,
            max_workers=1,
        )

        # Exactly 2 unique coordinates -> exactly 2 API calls
        assert mock_fetch.call_count == 2
        assert len(result) == 4
        # All 4 rows must be correctly populated
        elevations = result["elevation_usgs_m"].to_list()
        assert elevations[0] == pytest.approx(43.276)
        assert elevations[1] == pytest.approx(43.276)
        assert elevations[2] == pytest.approx(45.100)
        assert elevations[3] == pytest.approx(45.100)

    @patch("topoflow.ingestion.elevation_usgs.fetch_epqs_elevation")
    def test_caching_behavior_without_force_download(
        self,
        mock_fetch: MagicMock,
        sample_polars_stations: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Verify cached Parquet file is loaded directly without invoking network."""
        cache_file = tmp_path / "cached_elevations.parquet"

        cached_df = pl.DataFrame(
            {
                "segment_id": ["seg_001", "seg_002", "seg_003"],
                "latitude": [38.9697, 38.9710, 38.9725],
                "longitude": [-77.2092, -77.2105, -77.2120],
                "elevation_usgs_m": [10.0, 20.0, 30.0],
                "data_source": ["USGS_CACHED"] * 3,
            }
        )
        cached_df.write_parquet(cache_file)

        result = ingest_usgs_bare_earth_elevations(
            stations=sample_polars_stations,
            output_path=cache_file,
            force_download=False,
        )

        assert mock_fetch.call_count == 0
        assert result["elevation_usgs_m"].to_list() == [10.0, 20.0, 30.0]

    @patch("topoflow.ingestion.elevation_usgs.fetch_epqs_elevation")
    def test_force_download_bypasses_and_overwrites_cache(
        self,
        mock_fetch: MagicMock,
        sample_polars_stations: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Verify force_download=True queries network and overwrites existing cache."""
        cache_file = tmp_path / "cached_elevations.parquet"

        old_df = pl.DataFrame(
            {
                "segment_id": ["seg_001", "seg_002", "seg_003"],
                "latitude": [38.9697, 38.9710, 38.9725],
                "longitude": [-77.2092, -77.2105, -77.2120],
                "elevation_usgs_m": [10.0, 20.0, 30.0],
                "data_source": ["OLD_CACHE"] * 3,
            }
        )
        old_df.write_parquet(cache_file)

        mock_fetch.side_effect = [50.0, 51.0, 52.0]

        result = ingest_usgs_bare_earth_elevations(
            stations=sample_polars_stations,
            output_path=cache_file,
            force_download=True,
        )

        assert mock_fetch.call_count == 3
        assert result["elevation_usgs_m"].to_list() == [50.0, 51.0, 52.0]

        # Verify disk cache is updated
        disk_df = pl.read_parquet(cache_file)
        assert disk_df["elevation_usgs_m"].to_list() == [50.0, 51.0, 52.0]

    @patch("topoflow.ingestion.elevation_usgs.fetch_epqs_elevation")
    def test_output_path_persists_parquet(
        self,
        mock_fetch: MagicMock,
        sample_polars_stations: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Verify newly fetched data is saved to output_path if provided."""
        out_file = tmp_path / "new_dir" / "elevations.parquet"
        mock_fetch.side_effect = [33.0, 34.0, 35.0]

        result = ingest_usgs_bare_earth_elevations(
            stations=sample_polars_stations,
            output_path=out_file,
        )

        assert out_file.exists()
        persisted = pl.read_parquet(out_file)
        assert persisted.equals(result)

    @patch("topoflow.ingestion.elevation_usgs.fetch_epqs_elevation")
    def test_concurrency_with_max_workers(
        self,
        mock_fetch: MagicMock,
        sample_polars_stations: pl.DataFrame,
    ) -> None:
        """Verify multithreaded execution across workers operates cleanly."""
        mock_fetch.side_effect = [41.0, 42.0, 43.0]

        result = ingest_usgs_bare_earth_elevations(
            stations=sample_polars_stations,
            max_workers=3,
        )

        assert len(result) == 3
        assert set(result["elevation_usgs_m"].to_list()) == {41.0, 42.0, 43.0}

    def test_empty_input_stations(self) -> None:
        """Verify ingestion with empty stations returns empty Polars DataFrame with schema."""
        empty_stations = pl.DataFrame(
            schema={"segment_id": pl.Utf8, "latitude": pl.Float64, "longitude": pl.Float64}
        )

        result = ingest_usgs_bare_earth_elevations(stations=empty_stations)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        required_cols = {
            "segment_id",
            "latitude",
            "longitude",
            "elevation_usgs_m",
            "data_source",
        }
        assert required_cols.issubset(set(result.columns))
