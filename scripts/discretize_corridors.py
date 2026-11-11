"""Discretize ingested OSM corridor centerlines into 100m chainage stations.

Loads data/raw/geo/osm_corridors.parquet, executes SpatialDiscretizer, and exports
the canonical station matrix to data/interim/geo/stations_100m.parquet.
"""

from __future__ import annotations

import logging

import geopandas as gpd

from topoflow.config import get_settings
from topoflow.geo.discretization import SpatialDiscretizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    input_path = settings.raw_data_dir / "geo" / "osm_corridors.parquet"
    if not input_path.exists():
        # Fallback to GeoJSON if parquet not present
        input_path = settings.raw_data_dir / "geo" / "osm_corridors.geojson"

    if not input_path.exists():
        raise FileNotFoundError(f"Corridors input not found at {input_path}")

    logger.info("Loading corridor centerlines from %s...", input_path)
    corridors_gdf = (
        gpd.read_parquet(input_path)
        if input_path.suffix == ".parquet"
        else gpd.read_file(input_path)
    )

    discretizer = SpatialDiscretizer()
    logger.info("Discretizing centerlines at %sm intervals...", discretizer.interval_m)
    stations_gdf = discretizer.discretize_corridors_dataframe(corridors_gdf)

    output_path = settings.interim_data_dir / "geo" / "stations_100m.parquet"
    discretizer.export(stations_gdf, output_path)

    logger.info("Discretization Summary:")
    summary = (
        stations_gdf.groupby(["corridor_id", "direction"])
        .agg(
            station_count=("segment_id", "count"),
            total_chainage_km=("chainage_m", lambda s: round(s.max() / 1000.0, 2)),
            min_radius_m=("radius_m", "min"),
            mean_radius_m=("radius_m", lambda r: round(r.mean(), 1)),
        )
        .reset_index()
    )
    for _, row in summary.iterrows():
        logger.info(
            "  - %s (%s): %d stations, %s km, min R: %sm, mean R: %sm",
            row["corridor_id"],
            row["direction"],
            row["station_count"],
            row["total_chainage_km"],
            row["min_radius_m"],
            row["mean_radius_m"],
        )


if __name__ == "__main__":
    main()
