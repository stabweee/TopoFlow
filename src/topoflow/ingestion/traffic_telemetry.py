from pathlib import Path

import polars as pl

from topoflow.ingestion.schemas import TELEMETRY_POLARS_SCHEMA


def read_raw_telemetry(source_path: Path | str) -> pl.DataFrame:
    """
    Reads raw telemetry data from a Parquet file and returns it as a Polars DataFrame.

    Args:
        source_path (Path | str): The path to the Parquet file containing the raw telemetry data.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the raw telemetry data.
    """

    extension = Path(source_path).suffix.lower()
    if extension == ".csv":
        return pl.read_csv(source_path, try_parse_dates=True)
    elif extension == ".json":
        return pl.read_json(source_path)
    elif extension == ".parquet":
        return pl.read_parquet(source_path)
    raise ValueError(
        f"File format {extension} is not supported. Please provide a CSV, JSON, or Parquet file."
    )


def validate_telemetry(df: pl.DataFrame) -> pl.DataFrame:
    """
    Validates the telemetry data against the defined schema.

    Args:
        df (pl.DataFrame): The Polars DataFrame containing the telemetry data to be validated.

    Returns:
        pl.DataFrame: The validated Polars DataFrame.
    """

    existing_cols = set(df.schema.names())
    required_cols = set(TELEMETRY_POLARS_SCHEMA.names())
    missing_cols = required_cols - existing_cols

    if missing_cols:
        formatted_missing = ", ".join(sorted(missing_cols))
        raise ValueError(
            f"Incoming DataFrame is missing required telemetry columns: {formatted_missing}"
        )

    if df.schema["timestamp_utc"] in (pl.String, pl.Utf8):
        df = df.with_columns(pl.col("timestamp_utc").str.to_datetime(time_zone="UTC"))

    df = df.with_columns(
        [pl.col(col).cast(dtype, strict=False) for col, dtype in TELEMETRY_POLARS_SCHEMA.items()]
    )
    df = df.filter(
        pl.col("segment_id").is_not_null(),
        pl.col("corridor").is_not_null(),
        pl.col("direction").is_not_null(),
        pl.col("timestamp_utc").is_not_null(),
        pl.col("speed_mph").is_between(0.0, 120.0, closed="both"),
        pl.col("occupancy_pct").is_between(0.0, 100.0, closed="both"),
        pl.col("volume_vph") >= 0.0,
        pl.col("free_flow_speed_mph").is_between(30.0, 90.0, closed="both"),
        pl.col("confidence_score").is_between(0.0, 1.0, closed="both"),
    )

    return df


def aggregate_telemetry_5min(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates telemetry data into 5-minute intervals.

    Args:
        df (pl.DataFrame): The Polars DataFrame containing the telemetry data to be aggregated.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the aggregated telemetry data.
    """

    df = df.with_columns(pl.col("timestamp_utc").dt.truncate("5m"))
    df = (
        df.group_by(["segment_id", "corridor", "direction", "timestamp_utc"])
        .agg(
            speed_mph=pl.when(pl.sum("volume_vph") > 0)
            .then(pl.col("speed_mph").dot("volume_vph") / pl.sum("volume_vph"))
            .otherwise(pl.mean("speed_mph")),
            volume_vph=pl.mean("volume_vph"),
            occupancy_pct=pl.mean("occupancy_pct"),
            free_flow_speed_mph=pl.mean("free_flow_speed_mph"),
            confidence_score=pl.min("confidence_score"),
        )
        .sort(by=["segment_id", "corridor", "direction", "timestamp_utc"])
    )

    return df


def write_partitioned_telemetry(df: pl.DataFrame, output_dir: Path | str) -> Path:
    """Writes the telemetry data to Parquet files partitioned by corridor, year, and month.

    Args:
        df (pl.DataFrame): The Polars DataFrame containing the telemetry data to be written.
        output_dir (Path | str): The directory where the Parquet files will be saved.
    Returns:
        Path: The directory of the written Parquet files.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.with_columns(
        year=pl.col("timestamp_utc").dt.year().cast(pl.Int32),
        month=pl.col("timestamp_utc").dt.month().cast(pl.Int32),
    )

    df.write_parquet(
        output_dir,
        partition_by=["corridor", "year", "month"],
        compression="zstd",
    )

    return output_dir


def ingest_telemetry_pipeline(source_path: Path | str, output_dir: Path | str) -> Path:
    """Ingests telemetry data from a source file, validates it, aggregates it into 5-minute intervals,
    and writes the processed data to Parquet files partitioned by corridor, year, and month.

    Args:
        source_path (Path | str): The path to the source file containing the raw telemetry data.
        output_dir (Path | str): The directory where the processed Parquet files will be saved.

    Returns:
        Path: The directory of the written Parquet files.
    """

    df = read_raw_telemetry(source_path)
    df = validate_telemetry(df)
    df = aggregate_telemetry_5min(df)
    output_path = write_partitioned_telemetry(df, output_dir)

    return output_path
