"""Extract hybrid elevation profiles, compute vertical grades, and generate visualization figures.

Loads data/interim/geo/stations_100m.parquet, samples bare-earth DEM and ASPRS Class 17 LiDAR,
validates bridge deck elevations against As-Built blueprints, filters profiles using
Savitzky-Golay polynomials, exports data/interim/geo/elevation_profile.parquet,
and generates milestone visual verification artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import polars as pl

from topoflow.config import get_settings
from topoflow.geo.elevation import ElevationExtractor
from topoflow.visualization.elevation_plot import (
    plot_corridor_elevation_profile,
    plot_savgol_vs_raw_differences,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    stations_path = settings.interim_data_dir / "geo" / "stations_100m.parquet"
    if not stations_path.exists():
        raise FileNotFoundError(f"Stations dataset not found at {stations_path}")

    logger.info("Loading 100m discretized stations from %s...", stations_path)
    stations_gdf = gpd.read_parquet(stations_path)
    logger.info(
        "Loaded %d stations across %d corridors.",
        len(stations_gdf),
        stations_gdf["corridor_id"].nunique(),
    )

    extractor = ElevationExtractor(settings)

    # Authentic USGS 3DEP inputs only: production never falls back to test fixtures.
    laz_path = settings.raw_data_dir / "elevation" / "alb_i495_bridge_lidar.laz"
    usgs_epqs_path = settings.raw_data_dir / "elevation" / "usgs_epqs_bare_earth.parquet"
    for required in (laz_path, usgs_epqs_path):
        if not required.exists():
            raise FileNotFoundError(f"Required raw elevation input missing: {required}")

    # Independent, cited reference profile (e.g. as-built plan sheets). Optional.
    as_built_csv_path = settings.configs_dir / "bridge_asbuilt_reference.csv"

    logger.info("Processing corridor stations through hybrid elevation extraction...")
    df_elevation = extractor.process_corridor_stations(
        stations_gdf=stations_gdf,
        dem_path=None,
        laz_path=laz_path,
        usgs_epqs_path=usgs_epqs_path,
    )

    # Validate measured ALB inner-loop deck elevations against an independent reference
    alb_bridge = df_elevation.filter(
        (pl.col("corridor_id") == "alb_i495")
        & (pl.col("direction") == "inner_loop")
        & pl.col("is_bridge_deck")
        & (pl.col("elevation_source") == "lidar_c17")
    ).sort("chainage_m")
    if not as_built_csv_path.exists():
        logger.warning(
            "No independent bridge reference at %s; deck elevations are NOT externally "
            "validated (LiDAR Class 17 measurements only).",
            as_built_csv_path,
        )
    elif len(alb_bridge) > 0:
        bridge_z = alb_bridge["elevation_raw_m"].to_numpy()
        bridge_chainage = alb_bridge["chainage_m"].to_numpy()
        logger.info(
            "Validating %d inner loop bridge stations against VDOT/MDOT As-Built blueprints...",
            len(alb_bridge),
        )
        extractor.validate_bridge_as_built(
            extracted_deck_z=bridge_z,
            bridge_stations_chainage_m=bridge_chainage,
            as_built_csv_path=as_built_csv_path,
            tolerance_m=0.2,
        )

    # Export canonical elevation profile mart
    output_mart_path = settings.interim_data_dir / "geo" / "elevation_profile.parquet"
    output_mart_path.parent.mkdir(parents=True, exist_ok=True)
    df_elevation.write_parquet(output_mart_path)
    logger.info(
        "Exported canonical elevation profile mart to %s (%d rows)",
        output_mart_path,
        len(df_elevation),
    )

    # Evaluate Savitzky-Golay vs Raw Discrete Differences Metrics on ALB Inner Loop
    df_alb_inner = df_elevation.filter(
        (pl.col("corridor_id") == "alb_i495") & (pl.col("direction") == "inner_loop")
    ).sort("chainage_m")

    chainage = df_alb_inner["chainage_m"].to_numpy()
    raw_z = df_alb_inner["elevation_raw_m"].to_numpy()
    raw_grade = df_alb_inner["grade_raw_pct"].to_numpy()
    sg_grade = df_alb_inner["grade_pct"].to_numpy()

    metrics = extractor.evaluate_filtering_metrics(chainage, raw_z, raw_grade, sg_grade)
    logger.info("Savitzky-Golay Filtering Performance on ALB Inner Loop:")
    logger.info("  - Total Variation Raw:    %.2f%%", metrics["total_variation_raw"])
    logger.info("  - Total Variation S-G:    %.2f%%", metrics["total_variation_savgol"])
    logger.info("  - TV Reduction:           %.1f%%", metrics["tv_reduction_pct"])
    logger.info("  - Jerk Raw Energy:        %.2e", metrics["jerk_raw"])
    logger.info("  - Jerk S-G Energy:        %.2e", metrics["jerk_savgol"])

    # Generate Visual Verification Artifacts
    figures_dir = Path("outputs/figures/phase1")
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig1_path = figures_dir / "p1_elevation_grade_profile.png"
    plot_corridor_elevation_profile(df_elevation, fig1_path)
    logger.info("Generated milestone figure: %s", fig1_path)

    fig2_path = figures_dir / "p1_savgol_vs_raw_differences.png"
    plot_savgol_vs_raw_differences(df_elevation, metrics, fig2_path)
    logger.info("Generated comparison figure: %s", fig2_path)


if __name__ == "__main__":
    main()
