"""Causal data assembly for the early-price value benchmark."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .core_extract import configure_read_only_connection, database_connection, file_sha256
from .early_value_config import EarlyValueConfig
from .spot_l2_chainlink_features import (
    join_closed_chainlink_candles,
    join_qualified_l2,
)

PRICE_COLUMNS = (
    "market_id", "window_start", "window_end", "label_up", "fee_rate",
    "seconds_elapsed", "observed_at", "yes_received_at", "yes_snapshot_at",
    "yes_best_ask", "yes_ask_vwap_5", "yes_ask_depth", "no_received_at",
    "no_snapshot_at", "no_best_ask", "no_ask_vwap_5", "no_ask_depth",
)
PRICE_SCHEMA = pa.schema(
    [
        ("market_id", pa.string()),
        ("window_start", pa.timestamp("us", tz="UTC")),
        ("window_end", pa.timestamp("us", tz="UTC")),
        ("label_up", pa.int32()),
        ("fee_rate", pa.float64()),
        ("seconds_elapsed", pa.int32()),
        ("observed_at", pa.timestamp("us", tz="UTC")),
        ("yes_received_at", pa.timestamp("us", tz="UTC")),
        ("yes_snapshot_at", pa.timestamp("us", tz="UTC")),
        ("yes_best_ask", pa.float64()),
        ("yes_ask_vwap_5", pa.float64()),
        ("yes_ask_depth", pa.float64()),
        ("no_received_at", pa.timestamp("us", tz="UTC")),
        ("no_snapshot_at", pa.timestamp("us", tz="UTC")),
        ("no_best_ask", pa.float64()),
        ("no_ask_vwap_5", pa.float64()),
        ("no_ask_depth", pa.float64()),
    ]
)


def load_external_source(path: Path, *, start: datetime, end: datetime) -> pl.DataFrame:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files or not all(file.is_file() for file in files):
        raise FileNotFoundError(f"external source has no parquet evidence: {path}")
    scan = pl.scan_parquet(files)
    schema = scan.collect_schema()
    time_column = next(
        (name for name in ("available_at", "open_timestamp", "second_start") if name in schema),
        None,
    )
    if time_column is None:
        raise ValueError(f"external source lacks a causal time column: {path}")
    return scan.filter(pl.col(time_column).is_between(start, end, closed="left")).collect()


def build_strict_external_frame(
    core: pl.DataFrame,
    l2_source: pl.DataFrame,
    candle_source: pl.DataFrame,
) -> pl.DataFrame:
    """Return the common cohort where both causal external sources are present."""

    with_l2 = join_qualified_l2(core, l2_source, maximum_age_seconds=2)
    combined = join_closed_chainlink_candles(with_l2, candle_source)
    if not combined.height:
        raise RuntimeError("strict L2/candle cohort is empty")
    duplicates = combined.group_by("market_id", "seconds_elapsed").len().filter(pl.col("len") != 1)
    if duplicates.height:
        raise RuntimeError("strict external cohort contains duplicate decision points")
    return combined.sort("window_start", "seconds_elapsed")


def build_partitioned_external_frame(
    core: pl.DataFrame,
    l2_source: Path,
    candle_source: Path,
) -> pl.DataFrame:
    """Build the common cohort one UTC day at a time to bound L2 memory."""

    candles = load_external_source(
        candle_source,
        start=core["window_start"].min() - timedelta(minutes=62),
        end=core["window_start"].max() + timedelta(days=1),
    )
    pieces: list[pl.DataFrame] = []
    dated = core.with_columns(pl.col("window_start").dt.date().alias("_utc_day"))
    for day in sorted(dated["_utc_day"].unique().to_list()):
        daily_core = dated.filter(pl.col("_utc_day") == day).drop("_utc_day")
        start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
        end = start + timedelta(days=1)
        daily_path = l2_source / f"{day.isoformat()}.parquet"
        daily_l2 = load_external_source(
            daily_path if daily_path.is_file() else l2_source,
            start=start,
            end=end,
        )
        if daily_l2.is_empty():
            continue
        joined = build_strict_external_frame(daily_core, daily_l2, candles)
        if joined.height:
            pieces.append(joined)
    if not pieces:
        raise RuntimeError("no strict external feature partitions were produced")
    return pl.concat(pieces, how="vertical_relaxed").sort("window_start", "seconds_elapsed")


def extract_price_evidence(config: EarlyValueConfig, *, force: bool = False) -> dict[str, Any]:
    """Extract bounded daily causal book observations without mutating the database."""

    destination = config.price_cache
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file() and not force:
        return json.loads(manifest_path.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    query = config.price_source_sql.read_text()
    records: list[dict[str, Any]] = []
    connection = database_connection()
    try:
        configure_read_only_connection(connection)
        day = config.evaluation.start
        while day < config.evaluation.end:
            batch_end = min(day + timedelta(days=1), config.evaluation.end)
            output = destination / f"{day.date().isoformat()}.parquet"
            if force or not output.is_file():
                _extract_price_day(connection, query, day, batch_end, config, output)
            frame = pl.scan_parquet(output)
            records.append(
                {
                    "path": output.name,
                    "sha256": file_sha256(output),
                    "rows": frame.select(pl.len()).collect().item(),
                    "markets": frame.select(pl.col("market_id").n_unique()).collect().item(),
                }
            )
            day = batch_end
    finally:
        connection.close()
    manifest = {
        "schema_version": "btc-early-value-price-evidence-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "read_only_source": True,
        "quantity": config.quantity,
        "freshness_seconds": config.book_freshness_seconds,
        "price_seconds": list(config.price_seconds),
        "partitions": records,
        "refprice_training_eligible": False,
        "refprice_exclusion_reason": (
            "historical RefPrice reports do not carry a proven local receipt/availability timestamp"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_price_evidence(config: EarlyValueConfig) -> pl.DataFrame:
    manifest = extract_price_evidence(config)
    files = [config.price_cache / item["path"] for item in manifest["partitions"]]
    frame = pl.scan_parquet(files).collect().sort("window_start", "seconds_elapsed")
    missing = sorted(set(PRICE_COLUMNS) - set(frame.columns))
    if missing:
        raise RuntimeError("price evidence is missing columns: " + ", ".join(missing))
    return frame


def _extract_price_day(
    connection: Any,
    query: str,
    batch_start: datetime,
    batch_end: datetime,
    config: EarlyValueConfig,
    output: Path,
) -> None:
    temporary = output.with_suffix(".parquet.partial")
    writer: pq.ParquetWriter | None = None
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                {
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "freshness_seconds": config.book_freshness_seconds,
                    "quantity": config.quantity,
                },
            )
            names = [column.name for column in cursor.description]
            while rows := cursor.fetchmany(10_000):
                table = pa.Table.from_pylist(
                    [dict(zip(names, row, strict=True)) for row in rows],
                    schema=PRICE_SCHEMA,
                )
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
        if writer is None:
            table = pa.Table.from_pylist([], schema=PRICE_SCHEMA)
            pq.write_table(table, temporary, compression="zstd")
        else:
            writer.close()
            writer = None
        temporary.replace(output)
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
