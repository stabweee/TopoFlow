"""Ingest authentic USGS 3DEP LiDAR point cloud for the American Legion Bridge.

Queries the USGS 3DEP public archive on AWS Open Data (USGS_LPC_VA_Fairfax_County_2018),
downloads octree returns covering the Potomac River crossing, reprojects into
NAD83 / UTM Zone 18N (EPSG:26918), and saves data/raw/elevation/alb_i495_bridge_lidar.laz.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import laspy
import numpy as np

from topoflow.config import get_settings
from topoflow.ingestion.lidar_usgs import ingest_usgs_alb_lidar

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    stations_path = settings.interim_data_dir / "geo" / "stations_100m.parquet"
    if not stations_path.exists():
        raise FileNotFoundError(f"Stations dataset not found at {stations_path}")

    logger.info("Loading stations from %s...", stations_path)
    stations_gdf = gpd.read_parquet(stations_path)

    output_path = settings.raw_data_dir / "elevation" / "alb_i495_bridge_lidar.laz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Ingesting authentic USGS 3DEP LiDAR point cloud for American Legion Bridge...")
    persisted_path = ingest_usgs_alb_lidar(
        output_path=output_path,
        stations_gdf=stations_gdf,
        buffer_m=120.0,
        max_depth=7,
        max_workers=6,
        force_download=False,
    )

    # Inspect persisted LAZ file
    las = laspy.read(persisted_path)
    cls_vals, cls_counts = np.unique(las.classification, return_counts=True)
    logger.info("Persisted LiDAR Point Cloud Summary:")
    logger.info("  - File: %s (%d bytes)", persisted_path, persisted_path.stat().st_size)
    logger.info("  - Point Count: %d", len(las.points))
    logger.info("  - Easting Range: [%.2fm, %.2fm]", float(np.min(las.x)), float(np.max(las.x)))
    logger.info("  - Northing Range: [%.2fm, %.2fm]", float(np.min(las.y)), float(np.max(las.y)))
    logger.info("  - Elevation Range: [%.2fm, %.2fm]", float(np.min(las.z)), float(np.max(las.z)))
    logger.info(
        "  - Classification Breakdown: %s",
        dict(zip(cls_vals.tolist(), cls_counts.tolist(), strict=False)),
    )


if __name__ == "__main__":
    main()
