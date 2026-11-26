"""5-minute traffic telemetry ingestion: schema validation, aggregation, lakehouse storage.

Validation never repairs or invents values. Rows breaking the data contract are either
rejected loudly (``strict=True``, raising ``TelemetryValidationError`` with per-rule
counts) or separated into a rejected frame with a ``rejection_reason`` column.

Storage is append-only: every write lands in a new run-specific file inside the Hive
layout ``corridor=XXX/year=YYYY/month=MM/``, and readers de-duplicate on
``(segment_id, timestamp_utc)`` keeping the most recently ingested observation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from topoflow.ingestion.schemas import (
    CONFIDENCE_SCORE_BOUNDS,
    FREE_FLOW_SPEED_MPH_BOUNDS,
    OCCUPANCY_PCT_BOUNDS,
    SPEED_MPH_BOUNDS,
    TELEMETRY_OPTIONAL_COLUMNS,
    TELEMETRY_POLARS_SCHEMA,
    TELEMETRY_REQUIRED_COLUMNS,
    VOLUME_VPH_BOUNDS,
)

logger = logging.getLogger(__name__)

PARTITION_KEYS: tuple[str, ...] = ("corridor", "year", "month")
INGESTED_AT_COLUMN = "ingested_at_utc"
SPEED_AGG_HARMONIC = "harmonic_volume_weighted"
SPEED_AGG_HARMONIC_PARTIAL = "harmonic_partial_volume"
SPEED_AGG_ARITHMETIC = "arithmetic_unweighted"


class TelemetryValidationError(ValueError):
    """Raised when telemetry rows violate the data contract under strict validation.

    Attributes:
        violations: Mapping of rule name -> number of offending rows.
        rejected: The offending rows with a ``rejection_reason`` column.
    """

    def __init__(self, violations: dict[str, int], rejected: pl.DataFrame) -> None:
        self.violations = violations
        self.rejected = rejected
        summary = ", ".join(f"{rule}={count}" for rule, count in violations.items())
        super().__init__(f"{rejected.height} telemetry row(s) failed validation: {summary}")


def read_raw_telemetry(source_path: Path | str) -> pl.DataFrame:
    """Read a raw telemetry dump (CSV, JSON or Parquet) into a Polars DataFrame.

    Args:
        source_path: Path to the raw telemetry file.

    Returns:
        The raw, unvalidated DataFrame.

    Raises:
        ValueError: If the file extension is not supported.
    """
    extension = Path(source_path).suffix.lower()
    if extension == ".csv":
        return pl.read_csv(source_path, try_parse_dates=True)
    if extension == ".json":
        return pl.read_json(source_path)
    if extension == ".parquet":
        return pl.read_parquet(source_path)
    raise ValueError(
        f"File format {extension} is not supported. Please provide a CSV, JSON, or Parquet file."
    )


def _coerce_to_schema(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cast columns to the telemetry schema, recording which values failed to parse.

    Returns:
        Tuple of (coerced frame, boolean frame of ``<col>`` -> "had a value that failed to cast").
    """
    missing_required = set(TELEMETRY_REQUIRED_COLUMNS) - set(df.columns)
    if missing_required:
        raise ValueError(
            "Incoming DataFrame is missing required telemetry columns: "
            + ", ".join(sorted(missing_required))
        )

    df = df.with_columns(
        [
            pl.lit(None, dtype=TELEMETRY_POLARS_SCHEMA[col]).alias(col)
            for col in TELEMETRY_OPTIONAL_COLUMNS
            if col not in df.columns
        ]
    )

    was_present = df.select(
        [pl.col(col).is_not_null().alias(col) for col in TELEMETRY_POLARS_SCHEMA.names()]
    )

    ts = pl.col("timestamp_utc")
    if df.schema["timestamp_utc"] in (pl.String, pl.Utf8):
        ts = ts.str.to_datetime(time_zone="UTC", strict=False, time_unit="ms")

    coerced = df.with_columns(ts.alias("timestamp_utc")).with_columns(
        [pl.col(col).cast(dtype, strict=False) for col, dtype in TELEMETRY_POLARS_SCHEMA.items()]
    )

    cast_failed = pl.DataFrame(
        {col: was_present[col] & coerced[col].is_null() for col in TELEMETRY_POLARS_SCHEMA.names()}
    )
    return coerced.select(TELEMETRY_POLARS_SCHEMA.names()), cast_failed


def _out_of_bounds(col: str, bounds: tuple[float, float]) -> pl.Expr:
    """True where a non-null value lies outside the closed interval ``bounds``."""
    return pl.col(col).is_not_null() & ~pl.col(col).is_between(bounds[0], bounds[1], "both")


def partition_telemetry(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split telemetry rows into contract-valid rows and rejected rows.

    Rules (first violated rule is reported as ``rejection_reason``):
      * ``invalid_<col>``: a supplied value could not be parsed (e.g. bad timestamp).
      * ``missing_<col>``: a required column is null.
      * ``<col>_out_of_range``: a value lies outside its physical bounds.
    Null optional measurements are valid and stay null.

    Args:
        df: Raw telemetry DataFrame.

    Returns:
        Tuple of (valid rows matching TELEMETRY_POLARS_SCHEMA, rejected rows + reason).

    Raises:
        ValueError: If a required column is absent entirely.
    """
    coerced, cast_failed = _coerce_to_schema(df)

    rules: list[tuple[str, pl.Expr]] = []
    rules += [
        (f"invalid_{col}", pl.col(f"__fail_{col}")) for col in TELEMETRY_POLARS_SCHEMA.names()
    ]
    rules += [(f"missing_{col}", pl.col(col).is_null()) for col in TELEMETRY_REQUIRED_COLUMNS]
    rules += [
        ("speed_mph_out_of_range", _out_of_bounds("speed_mph", SPEED_MPH_BOUNDS)),
        (
            "free_flow_speed_mph_out_of_range",
            _out_of_bounds("free_flow_speed_mph", FREE_FLOW_SPEED_MPH_BOUNDS),
        ),
        ("volume_vph_out_of_range", _out_of_bounds("volume_vph", VOLUME_VPH_BOUNDS)),
        ("occupancy_pct_out_of_range", _out_of_bounds("occupancy_pct", OCCUPANCY_PCT_BOUNDS)),
        (
            "confidence_score_out_of_range",
            _out_of_bounds("confidence_score", CONFIDENCE_SCORE_BOUNDS),
        ),
    ]

    reason_expr: pl.Expr = pl.lit(None, dtype=pl.Utf8)
    for name, violated in reversed(rules):
        reason_expr = pl.when(violated).then(pl.lit(name)).otherwise(reason_expr)

    labelled = pl.concat(
        [coerced, cast_failed.rename({c: f"__fail_{c}" for c in cast_failed.columns})],
        how="horizontal",
    ).with_columns(reason_expr.alias("rejection_reason"))

    fail_cols = [f"__fail_{c}" for c in cast_failed.columns]
    valid = labelled.filter(pl.col("rejection_reason").is_null()).select(
        TELEMETRY_POLARS_SCHEMA.names()
    )
    rejected = labelled.filter(pl.col("rejection_reason").is_not_null()).drop(fail_cols)
    return valid, rejected


def validate_telemetry(df: pl.DataFrame, *, strict: bool = True) -> pl.DataFrame:
    """Validate telemetry against the data contract.

    Args:
        df: Raw telemetry DataFrame.
        strict: If True (default), any invalid row raises ``TelemetryValidationError``.
            If False, invalid rows are dropped and the per-rule counts are logged.

    Returns:
        Valid rows matching TELEMETRY_POLARS_SCHEMA.

    Raises:
        ValueError: If a required column is absent entirely.
        TelemetryValidationError: If ``strict`` and any row violates the contract.
    """
    valid, rejected = partition_telemetry(df)
    if rejected.height == 0:
        return valid

    counts = rejected.group_by("rejection_reason").len().sort("rejection_reason")
    violations = dict(zip(counts["rejection_reason"], counts["len"], strict=True))
    if strict:
        raise TelemetryValidationError(violations, rejected)

    logger.warning(
        "Dropped %d of %d telemetry rows failing validation: %s",
        rejected.height,
        df.height,
        violations,
    )
    return valid


def aggregate_telemetry_5min(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate validated telemetry into 5-minute epochs per station.

    Speed is the volume-weighted harmonic mean ``sum(q) / sum(q / v)`` over readings
    with positive volume (the space-mean speed of the observed flow):

    * A reading with positive volume and zero speed (stopped flow) drives the harmonic
      mean to its limit of 0 mph; it is never dropped, so breakdown is not biased upward.
    * Readings whose volume is null cannot be weighted. If an epoch mixes weighted and
      unweighted readings, the harmonic mean uses the weighted ones and is labelled
      ``harmonic_partial_volume``.
    * If no reading has positive volume, the unweighted arithmetic mean is used and
      labelled ``arithmetic_unweighted``.

    Optional measurements are averaged ignoring nulls and remain null when never reported.

    Args:
        df: Validated telemetry (TELEMETRY_POLARS_SCHEMA).

    Returns:
        One row per (segment_id, corridor, direction, 5-minute timestamp) with the schema
        columns plus ``n_raw_records``, ``n_volume_weighted`` and ``speed_aggregation``.
    """
    q = pl.col("volume_vph")
    v = pl.col("speed_mph")
    weighted = q > 0
    moving = weighted & (v > 0)
    has_weights = weighted.any()
    has_stopped_flow = (weighted & (v == 0)).any()
    sum_q = pl.when(moving).then(q).sum()
    sum_q_over_v = pl.when(moving).then(q / v).sum()
    has_unweighted = q.is_null().any()

    harmonic_speed = pl.when(has_stopped_flow).then(pl.lit(0.0)).otherwise(sum_q / sum_q_over_v)

    return (
        df.with_columns(pl.col("timestamp_utc").dt.truncate("5m"))
        .group_by(["segment_id", "corridor", "direction", "timestamp_utc"])
        .agg(
            speed_mph=pl.when(has_weights).then(harmonic_speed).otherwise(v.mean()),
            free_flow_speed_mph=pl.col("free_flow_speed_mph").mean(),
            volume_vph=q.mean(),
            occupancy_pct=pl.col("occupancy_pct").mean(),
            confidence_score=pl.col("confidence_score").min(),
            n_raw_records=pl.len().cast(pl.UInt32),
            n_volume_weighted=weighted.sum().cast(pl.UInt32),
            speed_aggregation=pl.when(has_weights & has_unweighted)
            .then(pl.lit(SPEED_AGG_HARMONIC_PARTIAL))
            .when(has_weights)
            .then(pl.lit(SPEED_AGG_HARMONIC))
            .otherwise(pl.lit(SPEED_AGG_ARITHMETIC)),
        )
        .sort(["segment_id", "corridor", "direction", "timestamp_utc"])
    )


def write_partitioned_telemetry(
    df: pl.DataFrame,
    output_dir: Path | str,
    run_id: str | None = None,
) -> Path:
    """Append telemetry to the Hive-partitioned lakehouse without overwriting prior runs.

    Each call writes one new file per ``corridor=<id>/year=<YYYY>/month=<M>/`` partition
    (month is not zero-padded, e.g. ``month=3``),
    named ``{run_id}.parquet``. Partition keys live in the directory names only. An
    ``ingested_at_utc`` column records the write time for de-duplication on read.

    Args:
        df: Telemetry to persist (must contain ``corridor`` and ``timestamp_utc``).
        output_dir: Lakehouse root (e.g. ``data/raw/telemetry``).
        run_id: Optional file stem; defaults to a UTC timestamp plus a random suffix.

    Returns:
        The lakehouse root directory.

    Raises:
        FileExistsError: If a file for ``run_id`` already exists in a target partition.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if df.is_empty():
        logger.info("No telemetry rows to write to %s", output_dir)
        return output_dir

    ingested_at = datetime.now(tz=UTC)
    stem = run_id or f"{ingested_at:%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"

    df = df.with_columns(
        year=pl.col("timestamp_utc").dt.year().cast(pl.Int32),
        month=pl.col("timestamp_utc").dt.month().cast(pl.Int32),
        **{INGESTED_AT_COLUMN: pl.lit(ingested_at, dtype=pl.Datetime("ms", "UTC"))},
    )

    for keys, part in df.partition_by(list(PARTITION_KEYS), as_dict=True).items():
        corridor, year, month = keys
        part_dir = output_dir / f"corridor={corridor}" / f"year={year}" / f"month={month}"
        part_dir.mkdir(parents=True, exist_ok=True)
        target = part_dir / f"{stem}.parquet"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing telemetry file: {target}")
        part.drop(list(PARTITION_KEYS)).write_parquet(target, compression="zstd")

    return output_dir


def read_partitioned_telemetry(output_dir: Path | str) -> pl.DataFrame:
    """Read the telemetry lakehouse, de-duplicating repeated observations.

    Rows sharing ``(segment_id, timestamp_utc)`` across runs are collapsed to the most
    recently ingested one. Files written by older schema versions are combined by column
    name (missing columns become null), so a schema change does not break reading.

    Args:
        output_dir: Lakehouse root written by ``write_partitioned_telemetry``.

    Returns:
        De-duplicated telemetry sorted by segment and time (empty if no files exist).
    """
    output_dir = Path(output_dir)
    files = sorted(output_dir.glob("corridor=*/year=*/month=*/*.parquet"))
    if not files:
        return pl.DataFrame(schema=TELEMETRY_POLARS_SCHEMA)

    frames = []
    for path in files:  # one read per file (not per row); tolerate schema drift
        part = pl.read_parquet(path).drop(list(PARTITION_KEYS), strict=False)
        keys = {
            k: v for k, v in (seg.split("=", 1) for seg in path.parent.parts[-3:] if "=" in seg)
        }
        frames.append(
            part.with_columns(
                corridor=pl.lit(keys.get("corridor")),
                year=pl.lit(int(keys["year"]), dtype=pl.Int32),
                month=pl.lit(int(keys["month"]), dtype=pl.Int32),
            )
        )
    df = pl.concat(frames, how="diagonal_relaxed")
    df = df.with_columns(pl.col("corridor").cast(TELEMETRY_POLARS_SCHEMA["corridor"]))
    if INGESTED_AT_COLUMN in df.columns:
        df = df.sort(INGESTED_AT_COLUMN)
    return df.unique(subset=["segment_id", "timestamp_utc"], keep="last").sort(
        ["segment_id", "timestamp_utc"]
    )


def ingest_telemetry_pipeline(
    source_path: Path | str,
    output_dir: Path | str,
    *,
    strict: bool = True,
) -> Path:
    """Read, validate, aggregate to 5-minute epochs, and append telemetry to the lakehouse.

    Args:
        source_path: Raw telemetry dump (CSV, JSON or Parquet).
        output_dir: Lakehouse root directory.
        strict: Passed to ``validate_telemetry``.

    Returns:
        The lakehouse root directory.
    """
    df = read_raw_telemetry(source_path)
    df = validate_telemetry(df, strict=strict)
    df = aggregate_telemetry_5min(df)
    return write_partitioned_telemetry(df, output_dir)
