"""USGS 3DEP bare-earth elevation ingestion client via The National Map EPQS API.

Queries the official USGS Elevation Point Query Service (EPQS) to fetch authoritative
1-meter bare-earth LiDAR elevations (NAVD88 meters) for study and control corridor stations.
Provides concurrency, exponential backoff, coordinate deduplication, and Parquet caching.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import geopandas as gpd
import polars as pl
import requests

logger = logging.getLogger(__name__)

USGS_EPQS_ENDPOINT: str = "https://epqs.nationalmap.gov/v1/json"


def fetch_epqs_elevation(
    lat: float,
    lon: float,
    session: Any = None,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
) -> float:
    """Fetch bare-earth elevation in meters from the official USGS EPQS API.

    Args:
        lat: Latitude in decimal degrees (EPSG:4326).
        lon: Longitude in decimal degrees (EPSG:4326).
        session: Optional requests.Session instance for connection reuse.
        max_retries: Maximum number of retry attempts on network/server errors (default 3).
        backoff_factor: Exponential backoff delay multiplier in seconds (default 0.5s).

    Returns:
        Float elevation in meters (NAVD88).

    Raises:
        ConnectionError: If network connection fails and retries are exhausted.
        ValueError: If response is missing the 'value' field or contains nodata/invalid entries.
    """
    http = session or requests.Session()
    params = {"x": lon, "y": lat, "units": "Meters"}

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = http.get(USGS_EPQS_ENDPOINT, params=params, timeout=15)

            if resp.status_code in (429, 503, 504):
                sleep_time = backoff_factor * (2**attempt)
                logger.warning(
                    "USGS EPQS returned HTTP %d for (%f, %f). Retrying in %.2fs (attempt %d/%d)...",
                    resp.status_code,
                    lat,
                    lon,
                    sleep_time,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(sleep_time)
                continue

            resp.raise_for_status()
            payload = resp.json()

            if "value" not in payload:
                raise ValueError(
                    f"USGS EPQS response missing required 'value' field for ({lat}, {lon}): {payload}"
                )

            val_raw = payload["value"]
            if val_raw is None or val_raw == "":
                raise ValueError(f"USGS EPQS returned null/empty value for ({lat}, {lon})")

            try:
                elevation_m = float(val_raw)
            except (ValueError, TypeError) as parse_err:
                raise ValueError(
                    f"Non-numeric elevation value '{val_raw}' returned by USGS EPQS for ({lat}, {lon})"
                ) from parse_err

            # USGS nodata sentinel is typically -1000000 or -999999
            if elevation_m <= -999999 or elevation_m > 9000:
                raise ValueError(
                    f"USGS nodata sentinel elevation ({elevation_m}) returned for ({lat}, {lon})"
                )

            return elevation_m

        except (requests.RequestException, ConnectionError, TimeoutError) as err:
            last_error = err
            sleep_time = backoff_factor * (2**attempt)
            logger.warning(
                "Network error connecting to USGS EPQS for (%f, %f): %s. Retrying in %.2fs...",
                lat,
                lon,
                err,
                sleep_time,
            )
            time.sleep(sleep_time)

    raise ConnectionError(
        f"Failed to fetch USGS elevation for ({lat}, {lon}) after {max_retries} attempts: {last_error}"
    )


def ingest_usgs_bare_earth_elevations(
    stations: pl.DataFrame | gpd.GeoDataFrame,
    output_path: Path | None = None,
    max_workers: int = 4,
    force_download: bool = False,
) -> pl.DataFrame:
    """Ingest authentic 1m bare-earth USGS elevations across highway stations with caching.

    Args:
        stations: Input station DataFrame (Polars or GeoPandas GeoDataFrame).
        output_path: Optional destination Path to cache the raw Parquet mart.
        max_workers: Thread worker pool size for concurrent EPQS queries (default 4).
        force_download: If True, bypasses local cache and re-downloads fresh data (default False).

    Returns:
        Polars DataFrame with station attributes and real USGS 3DEP elevations.
    """
    if output_path is not None and Path(output_path).exists() and not force_download:
        logger.info("Loading cached authentic USGS 3DEP elevations from %s", output_path)
        return pl.read_parquet(output_path)

    # Convert GeoDataFrame to Polars DataFrame if necessary
    if isinstance(stations, gpd.GeoDataFrame):
        df_dict: dict[str, Any] = {}
        for col in stations.columns:
            if col != "geometry":
                df_dict[col] = stations[col].to_list()
        if "latitude" not in df_dict and "geometry" in stations:
            df_dict["latitude"] = stations.geometry.y.to_list()
            df_dict["longitude"] = stations.geometry.x.to_list()
        stations_pl = pl.DataFrame(df_dict)
    else:
        stations_pl = stations.clone()

    if len(stations_pl) == 0:
        empty_schema: dict[str, Any] = {
            "segment_id": pl.String,
            "latitude": pl.Float64,
            "longitude": pl.Float64,
            "elevation_usgs_m": pl.Float64,
            "data_source": pl.String,
        }
        for col in stations_pl.columns:
            if col not in empty_schema:
                empty_schema[col] = stations_pl.schema[col]
        return pl.DataFrame(schema=empty_schema)

    # Deduplicate coordinates while maintaining original occurrence order
    unique_coords = (
        stations_pl.select(["latitude", "longitude"]).unique(maintain_order=True).to_dicts()
    )

    logger.info(
        "Ingesting authentic USGS 3DEP bare-earth elevations for %d unique coordinates (%d total stations)...",
        len(unique_coords),
        len(stations_pl),
    )

    elevation_map: dict[tuple[float, float], float] = {}
    session = requests.Session()

    def _worker(coord: dict[str, float]) -> tuple[tuple[float, float], float]:
        lat = float(coord["latitude"])
        lon = float(coord["longitude"])
        elev = fetch_epqs_elevation(lat=lat, lon=lon, session=session)
        return (lat, lon), elev

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, c) for c in unique_coords]
        for future in as_completed(futures):
            coord_tuple, elev_val = future.result()
            elevation_map[coord_tuple] = elev_val

    # Map elevations back to stations
    elevations_list: list[float] = [
        elevation_map[(float(lat), float(lon))]
        for lat, lon in zip(stations_pl["latitude"], stations_pl["longitude"], strict=True)
    ]

    result_df = stations_pl.with_columns(
        [
            pl.Series("elevation_usgs_m", elevations_list, dtype=pl.Float64),
            pl.lit("USGS 3DEP 1m EPQS").alias("data_source"),
        ]
    )

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(out_p)
        logger.info(
            "Saved authentic USGS bare-earth elevations to %s (%d rows)", out_p, len(result_df)
        )

    return result_df
