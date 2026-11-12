"""Ingest authentic 1m bare-earth USGS 3DEP elevations for all corridor stations.

Loads data/interim/geo/stations_100m.parquet, queries the official USGS EPQS API
for each coordinate, and exports the authoritative raw dataset to
data/raw/elevation/usgs_epqs_bare_earth.parquet.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import polars as pl

from topoflow.config import get_settings
from topoflow.ingestion.elevation_usgs import ingest_usgs_bare_earth_elevations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    stations_path = settings.interim_data_dir / "geo" / "stations_100m.parquet"
    if not stations_path.exists():
        raise FileNotFoundError(f"Stations dataset not found at {stations_path}")

    logger.info("Loading stations from %s...", stations_path)
    stations_gdf = gpd.read_parquet(stations_path)
    logger.info(
        "Loaded %d stations across %d corridors.",
        len(stations_gdf),
        stations_gdf["corridor_id"].nunique(),
    )

    output_path = settings.raw_data_dir / "elevation" / "usgs_epqs_bare_earth.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_usgs = ingest_usgs_bare_earth_elevations(
        stations=stations_gdf,
        output_path=output_path,
        max_workers=6,
        force_download=False,
    )

    logger.info("Authentic USGS 3DEP Ingestion Summary:")
    summary = (
        df_usgs.group_by(["corridor_id", "direction"])
        .agg(
            station_count=pl.len(),
            mean_elevation=pl.col("elevation_usgs_m").mean().round(2),
            min_elevation=pl.col("elevation_usgs_m").min().round(2),
            max_elevation=pl.col("elevation_usgs_m").max().round(2),
        )
        .sort(["corridor_id", "direction"])
    )
    for row in summary.iter_rows(named=True):
        logger.info(
            "  - %s (%s): %d stations, Elev: %.2fm to %.2fm (mean: %.2fm)",
            row["corridor_id"],
            row["direction"],
            row["station_count"],
            row["min_elevation"],
            row["max_elevation"],
            row["mean_elevation"],
        )


if __name__ == "__main__":
    main()
