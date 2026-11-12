"""Geospatial Feature Mart Integration (Phase 1 Final Integration).

Consolidates 100m station discretization geometries, horizontal Menger curvature,
and hybrid LiDAR bare-earth/bridge deck elevation profiles into an immutable,
query-ready analytical spatial feature mart in Parquet and DuckDB.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import duckdb
import polars as pl

from topoflow.config import get_settings
from topoflow.constants import (
    GRADE_FLAT_LOWER_THRESHOLD,
    GRADE_FLAT_UPPER_THRESHOLD,
    GRADE_STEEP_DOWNGRADE_THRESHOLD,
    GRADE_STEEP_UPGRADE_THRESHOLD,
    TFI_CURVATURE_NORM_INV_M,
    TFI_GRADE_NORM_PCT,
    TFI_WEIGHT_CURVATURE,
    TFI_WEIGHT_GRADE,
)

logger = logging.getLogger(__name__)


def compute_asymmetric_grades(df: pl.DataFrame) -> pl.DataFrame:
    """Compute asymmetric vertical grade decomposition.

    Splits raw vertical grade (%) into positive climb resistance (Grade+)
    and negative gravitational descent (Grade-), along with absolute grade.

    Args:
        df: Input DataFrame containing 'grade_pct' column.

    Returns:
        DataFrame augmented with 'grade_climb_pct', 'grade_descent_pct',
        and 'grade_abs_pct'.
    """
    if "grade_pct" not in df.columns:
        raise ValueError("Input DataFrame missing required column 'grade_pct'")

    return df.with_columns(
        pl.when(pl.col("grade_pct") > 0.0)
        .then(pl.col("grade_pct"))
        .otherwise(0.0)
        .alias("grade_climb_pct"),
        pl.when(pl.col("grade_pct") < 0.0)
        .then(pl.col("grade_pct"))
        .otherwise(0.0)
        .alias("grade_descent_pct"),
        pl.col("grade_pct").abs().alias("grade_abs_pct"),
    )


def classify_grade_regimes(df: pl.DataFrame) -> pl.DataFrame:
    """Classify roadway stations into AASHTO geometric grade regimes.

    Categories:
        - steep_downgrade: grade_pct <= -2.5%
        - moderate_downgrade: -2.5% < grade_pct < -0.5%
        - flat_tangent: -0.5% <= grade_pct <= +0.5%
        - moderate_upgrade: +0.5% < grade_pct < +2.5%
        - steep_upgrade: grade_pct >= +2.5%

    Args:
        df: Input DataFrame containing 'grade_pct' column.

    Returns:
        DataFrame augmented with categorical string column 'grade_class'.
    """
    if "grade_pct" not in df.columns:
        raise ValueError("Input DataFrame missing required column 'grade_pct'")

    return df.with_columns(
        pl.when(pl.col("grade_pct") <= GRADE_STEEP_DOWNGRADE_THRESHOLD)
        .then(pl.lit("steep_downgrade"))
        .when(pl.col("grade_pct") < GRADE_FLAT_LOWER_THRESHOLD)
        .then(pl.lit("moderate_downgrade"))
        .when(pl.col("grade_pct") <= GRADE_FLAT_UPPER_THRESHOLD)
        .then(pl.lit("flat_tangent"))
        .when(pl.col("grade_pct") < GRADE_STEEP_UPGRADE_THRESHOLD)
        .then(pl.lit("moderate_upgrade"))
        .otherwise(pl.lit("steep_upgrade"))
        .alias("grade_class")
    )


def compute_topographical_friction_index(
    df: pl.DataFrame,
    weight_grade: float = TFI_WEIGHT_GRADE,
    weight_curvature: float = TFI_WEIGHT_CURVATURE,
    grade_norm: float = TFI_GRADE_NORM_PCT,
    curvature_norm: float = TFI_CURVATURE_NORM_INV_M,
) -> pl.DataFrame:
    """Compute the composite Topographical Friction Index (TFI).

    Combines normalized absolute vertical grade and horizontal Menger curvature:
        TFI_i = w_g * (|Grade_i| / Grade_norm) + w_k * (kappa_i / kappa_norm)

    Also derives the slope angle parameter:
        sin_slope_angle = Grade_i / 100.0
    for downstream EPA MOVES emissions modeling (VSP).

    Args:
        df: Input DataFrame containing 'grade_pct' and 'curvature_kappa'.
        weight_grade: Relative weight for vertical grade component.
        weight_curvature: Relative weight for horizontal curvature component.
        grade_norm: Normalization divisor for vertical grade (percent).
        curvature_norm: Normalization divisor for horizontal curvature (m^-1).

    Returns:
        DataFrame augmented with 'topographical_friction_index' and 'sin_slope_angle'.
    """
    if "grade_pct" not in df.columns:
        raise ValueError("Input DataFrame missing required column 'grade_pct'")
    if "curvature_kappa" not in df.columns:
        raise ValueError("Input DataFrame missing required column 'curvature_kappa'")
    if grade_norm <= 0.0:
        raise ValueError(f"grade_norm must be strictly positive, received {grade_norm}")
    if curvature_norm <= 0.0:
        raise ValueError(f"curvature_norm must be strictly positive, received {curvature_norm}")

    grade_abs = pl.col("grade_pct").abs()
    curvature = pl.col("curvature_kappa")

    tfi_expr = (
        pl.lit(weight_grade) * (grade_abs / pl.lit(grade_norm))
        + pl.lit(weight_curvature) * (curvature / pl.lit(curvature_norm))
    ).alias("topographical_friction_index")

    sin_slope_expr = (pl.col("grade_pct") / 100.0).alias("sin_slope_angle")

    return df.with_columns([tfi_expr, sin_slope_expr])


def build_spatial_feature_mart(
    stations_path: Path | None = None,
    elevation_path: Path | None = None,
    output_path: Path | None = None,
    partition_by_corridor: bool = True,
) -> pl.DataFrame:
    """Consolidate spatial geometry and elevation profiles into the analytical mart.

    Performs an exact inner join on (corridor_id, segment_id), derives asymmetric
    grades, AASHTO grade classifications, and composite TFI, validates completeness,
    and exports to Parquet.

    Args:
        stations_path: Path to interim stations_100m.parquet (defaults to settings).
        elevation_path: Path to interim elevation_profile.parquet (defaults to settings).
        output_path: Path to target processed mart parquet (defaults to settings).
        partition_by_corridor: Whether to partition the parquet store by corridor.

    Returns:
        Integrated Polars DataFrame representing the spatial feature mart.
    """
    settings = get_settings()
    s_path = stations_path or (settings.interim_data_dir / "geo" / "stations_100m.parquet")
    e_path = elevation_path or (settings.interim_data_dir / "geo" / "elevation_profile.parquet")
    out_path = output_path or (settings.processed_data_dir / "spatial_feature_mart.parquet")

    if not s_path.exists():
        raise FileNotFoundError(f"Stations parquet file not found at {s_path}")
    if not e_path.exists():
        raise FileNotFoundError(f"Elevation parquet file not found at {e_path}")

    logger.info("Loading interim stations from %s", s_path)
    df_stations = pl.read_parquet(s_path)
    logger.info("Loading interim elevation profile from %s", e_path)
    df_elevation = pl.read_parquet(e_path)

    # Columns to take from elevation profile (avoiding duplicate keys)
    elevation_cols = [
        "segment_id",
        "elevation_raw_m",
        "elevation_smooth_m",
        "grade_pct",
        "grade_raw_pct",
        "vertical_curvature",
        "is_sag",
        "is_crest",
        "is_bridge_deck",
    ]
    # Filter elevation cols to those present
    available_e_cols = [c for c in elevation_cols if c in df_elevation.columns]

    # Perform equi-join on segment_id
    df_mart = df_stations.join(
        df_elevation.select(available_e_cols),
        on="segment_id",
        how="inner",
    )

    if df_mart.height != df_stations.height:
        logger.warning(
            "Row count mismatch after join: stations=%d, joined=%d",
            df_stations.height,
            df_mart.height,
        )

    # Ensure both corridor_id and corridor exist for ergonomic SQL access
    if "corridor_id" in df_mart.columns and "corridor" not in df_mart.columns:
        df_mart = df_mart.with_columns(pl.col("corridor_id").alias("corridor"))
    elif "corridor" in df_mart.columns and "corridor_id" not in df_mart.columns:
        df_mart = df_mart.with_columns(pl.col("corridor").alias("corridor_id"))

    # Feature Engineering
    df_mart = compute_asymmetric_grades(df_mart)
    df_mart = classify_grade_regimes(df_mart)
    df_mart = compute_topographical_friction_index(df_mart)

    # Validate completeness across critical metrics
    critical_cols = [
        "segment_id",
        "corridor_id",
        "corridor",
        "chainage_m",
        "easting",
        "northing",
        "grade_pct",
        "curvature_kappa",
        "topographical_friction_index",
    ]
    for col in critical_cols:
        if col in df_mart.columns:
            null_count = df_mart[col].null_count()
            if null_count > 0:
                raise ValueError(f"Column '{col}' contains {null_count} nulls in feature mart")

    # Export to Parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()

    if partition_by_corridor:
        logger.info("Writing partitioned spatial feature mart to %s", out_path)
        df_mart.write_parquet(out_path, partition_by="corridor", compression="zstd")
    else:
        logger.info("Writing monolithic spatial feature mart to %s", out_path)
        df_mart.write_parquet(out_path, compression="zstd")

    return df_mart


def register_spatial_feature_mart(
    con: duckdb.DuckDBPyConnection | None = None,
    mart_path: Path | None = None,
    view_name: str = "spatial_feature_mart",
) -> duckdb.DuckDBPyConnection:
    """Register spatial feature mart as a DuckDB view for zero-copy analytical queries.

    Args:
        con: Optional active DuckDB connection. If None, an in-memory connection is created.
        mart_path: Path to spatial_feature_mart.parquet (or partitioned directory).
        view_name: Name of the registered SQL view.

    Returns:
        DuckDB connection with view registered.
    """
    settings = get_settings()
    target_path = mart_path or (settings.processed_data_dir / "spatial_feature_mart.parquet")

    if not target_path.exists():
        raise FileNotFoundError(f"Spatial feature mart not found at {target_path}")

    connection = con or duckdb.connect()
    path_str = target_path.as_posix()

    query = f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{path_str}')"
    connection.execute(query)
    logger.info("Registered DuckDB view '%s' -> '%s'", view_name, path_str)

    return connection
