from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypeVar

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

from .core_extract import (
    configure_read_only_connection,
    database_connection,
    file_sha256,
    write_json_atomic,
)

EXECUTION_EVIDENCE_CONTRACT = "btc_execution_evidence_v1"
EXECUTION_EVIDENCE_SCHEMA_VERSION = "btc-execution-evidence-v1"
COMPACT_SNAPSHOT_SCHEMA_VERSION = "btc5m-book-250ms-v1"
DEFAULT_EXECUTION_QUANTITY = 5.0
DEFAULT_FRESHNESS_SECONDS = 2

QUALITY_UP_MISSING = 1
QUALITY_DOWN_MISSING = 1 << 1
QUALITY_UP_STALE = 1 << 2
QUALITY_DOWN_STALE = 1 << 3
QUALITY_UP_CROSSED = 1 << 4
QUALITY_DOWN_CROSSED = 1 << 5
QUALITY_UP_INSUFFICIENT_DEPTH = 1 << 6
QUALITY_DOWN_INSUFFICIENT_DEPTH = 1 << 7
STRICT_BOTH_SIDE_QUALITY_MASK = (1 << 6) - 1

EXECUTION_EVIDENCE_SCHEMA = pa.schema(
    [
        ("market_id", pa.string()),
        ("window_start", pa.timestamp("us", tz="UTC")),
        ("window_end", pa.timestamp("us", tz="UTC")),
        ("official_outcome", pa.string()),
        ("label_up", pa.int32()),
        ("min_tick_size", pa.float64()),
        ("min_order_size", pa.float64()),
        ("fee_rate", pa.float64()),
        ("observed_at", pa.timestamp("us", tz="UTC")),
        ("seconds_elapsed", pa.int32()),
        ("artifact_id", pa.string()),
        ("schema_version", pa.string()),
        ("up_provider_received_at", pa.timestamp("us", tz="UTC")),
        ("up_best_bid", pa.float64()),
        ("up_best_ask", pa.float64()),
        ("up_best_bid_size", pa.float64()),
        ("up_best_ask_size", pa.float64()),
        ("up_bid_depth", pa.float64()),
        ("up_ask_depth", pa.float64()),
        ("up_ask_vwap_5", pa.float64()),
        ("up_imbalance", pa.float64()),
        ("down_provider_received_at", pa.timestamp("us", tz="UTC")),
        ("down_best_bid", pa.float64()),
        ("down_best_ask", pa.float64()),
        ("down_best_bid_size", pa.float64()),
        ("down_best_ask_size", pa.float64()),
        ("down_bid_depth", pa.float64()),
        ("down_ask_depth", pa.float64()),
        ("down_ask_vwap_5", pa.float64()),
        ("down_imbalance", pa.float64()),
        ("quality_flags", pa.int32()),
        ("up_provider_causal", pa.bool_()),
        ("down_provider_causal", pa.bool_()),
        ("up_fields_complete", pa.bool_()),
        ("down_fields_complete", pa.bool_()),
        ("up_side_valid", pa.bool_()),
        ("down_side_valid", pa.bool_()),
        ("up_side_fresh", pa.bool_()),
        ("down_side_fresh", pa.bool_()),
        ("up_stale_initialized", pa.bool_()),
        ("down_stale_initialized", pa.bool_()),
        ("strict_both_side_eligible", pa.bool_()),
    ]
)

Number = TypeVar("Number")


@dataclass(frozen=True)
class ExecutionEvidenceConfig:
    range_start: datetime
    range_end: datetime
    output_dir: Path
    sample_interval_seconds: int = 5
    min_seconds_after_open: int = 60
    max_seconds_after_open: int = 240
    freshness_seconds: int = DEFAULT_FRESHNESS_SECONDS
    quantity: float = DEFAULT_EXECUTION_QUANTITY

    def __post_init__(self) -> None:
        for name, value in (
            ("range_start", self.range_start),
            ("range_end", self.range_end),
        ):
            if value.tzinfo is None:
                raise ValueError(f"{name} must include a UTC offset")
            if value.utcoffset() != timedelta(0):
                raise ValueError(f"{name} must be UTC")
            if value.astimezone(UTC).time() != datetime.min.time():
                raise ValueError(f"{name} must align to a UTC day")
        if self.range_end <= self.range_start:
            raise ValueError("execution-evidence range must be positive")
        if self.sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        if not 0 <= self.min_seconds_after_open <= self.max_seconds_after_open < 300:
            raise ValueError("candidate timestamps must stay within the five-minute market")
        if (
            self.min_seconds_after_open % self.sample_interval_seconds
            or self.max_seconds_after_open % self.sample_interval_seconds
        ):
            raise ValueError("candidate timestamp boundaries must align to the cadence")
        if self.freshness_seconds <= 0:
            raise ValueError("freshness_seconds must be positive")
        if self.quantity != DEFAULT_EXECUTION_QUANTITY:
            raise ValueError("execution evidence is fixed to the stored five-share VWAP")


@dataclass(frozen=True)
class SideEligibility:
    valid: bool
    fresh: bool
    stale_initialized: bool


def classify_side_eligibility(
    *,
    side: Literal["up", "down"],
    observed_at: datetime,
    provider_received_at: datetime | None,
    best_bid: float | None,
    best_ask: float | None,
    best_bid_size: float | None,
    best_ask_size: float | None,
    bid_depth: float | None,
    ask_depth: float | None,
    ask_vwap_5: float | None,
    imbalance: float | None,
    quality_flags: int,
    freshness_seconds: int = DEFAULT_FRESHNESS_SECONDS,
) -> SideEligibility:
    if quality_flags < 0:
        raise ValueError("quality_flags must be nonnegative")
    if freshness_seconds <= 0:
        raise ValueError("freshness_seconds must be positive")

    fields_complete = all(
        value is not None
        for value in (
            best_bid,
            best_ask,
            best_bid_size,
            best_ask_size,
            bid_depth,
            ask_depth,
            ask_vwap_5,
            imbalance,
        )
    )
    provider_causal = (
        provider_received_at is not None and provider_received_at <= observed_at
    )
    missing_mask, stale_mask, crossed_mask = (
        (QUALITY_UP_MISSING, QUALITY_UP_STALE, QUALITY_UP_CROSSED)
        if side == "up"
        else (QUALITY_DOWN_MISSING, QUALITY_DOWN_STALE, QUALITY_DOWN_CROSSED)
    )
    valid = (
        provider_causal
        and fields_complete
        and quality_flags & (missing_mask | crossed_mask) == 0
    )
    within_freshness = (
        provider_received_at is not None
        and provider_received_at
        >= observed_at - timedelta(seconds=freshness_seconds)
    )
    fresh = valid and quality_flags & stale_mask == 0 and within_freshness
    return SideEligibility(
        valid=valid,
        fresh=fresh,
        stale_initialized=valid and not fresh,
    )


def selected_side_eligibility(
    *,
    predicted_up: bool,
    up: SideEligibility,
    down: SideEligibility,
) -> SideEligibility:
    return up if predicted_up else down


def strict_both_side_eligible(
    *,
    up: SideEligibility,
    down: SideEligibility,
    quality_flags: int,
) -> bool:
    if quality_flags < 0:
        raise ValueError("quality_flags must be nonnegative")
    return (
        up.fresh
        and down.fresh
        and quality_flags & STRICT_BOTH_SIDE_QUALITY_MASK == 0
    )


def taker_fee_per_share(fee_rate: Number, price: Number) -> Number:
    return fee_rate * price * (1 - price)  # type: ignore[operator, return-value]


def expected_net_per_share(
    selected_probability: Number,
    execution_price: Number,
    fee_rate: Number,
) -> Number:
    return (  # type: ignore[operator, return-value]
        selected_probability
        - execution_price
        - taker_fee_per_share(fee_rate, execution_price)
    )


def realized_pnl(
    *,
    correct: Number,
    execution_price: Number,
    fee_rate: Number,
    quantity: Number = DEFAULT_EXECUTION_QUANTITY,  # type: ignore[assignment]
) -> Number:
    return quantity * (  # type: ignore[operator, return-value]
        correct
        - execution_price
        - taker_fee_per_share(fee_rate, execution_price)
    )


def extract_execution_evidence(
    config: ExecutionEvidenceConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    query_path = Path(__file__).parents[2] / "sql" / "btc-execution-evidence.sql"
    query = query_path.read_text()
    manifest_path = config.output_dir / "manifest.json"
    contract: dict[str, Any] = {
        "source_contract": EXECUTION_EVIDENCE_CONTRACT,
        "source_schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "source_schema_sha256": hashlib.sha256(
            EXECUTION_EVIDENCE_SCHEMA.to_string().encode()
        ).hexdigest(),
        "snapshot_schema_version": COMPACT_SNAPSHOT_SCHEMA_VERSION,
        "range_start": config.range_start.isoformat(),
        "range_end": config.range_end.isoformat(),
        "sample_interval_seconds": config.sample_interval_seconds,
        "min_seconds_after_open": config.min_seconds_after_open,
        "max_seconds_after_open": config.max_seconds_after_open,
        "freshness_seconds": config.freshness_seconds,
        "quantity": config.quantity,
        "primary_key": ["market_id", "observed_at"],
        "source_tables": [
            "polymarket.btc_interval_markets",
            "polymarket.btc_market_execution_snapshots",
            "polymarket.backfill_artifacts",
        ],
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
    }
    existing_manifest = _load_existing_manifest(manifest_path, contract, force=force)
    existing_partitions = {
        row["path"]: row for row in (existing_manifest or {}).get("partitions", [])
    }
    manifest: dict[str, Any] = {**contract, "partitions": []}

    connection: psycopg.Connection[Any] | None = None
    try:
        batch_start = config.range_start
        while batch_start < config.range_end:
            batch_end = min(batch_start + timedelta(days=1), config.range_end)
            destination = config.output_dir / f"{batch_start.date().isoformat()}.parquet"
            expected = existing_partitions.get(destination.name)
            if destination.exists() and not force:
                if expected is None:
                    raise RuntimeError(
                        f"{destination.name} is not recorded in {manifest_path.name}; "
                        "use force only for an intentional isolated rebuild"
                    )
                summary = execution_partition_summary(destination)
                sha256 = file_sha256(destination)
                if expected.get("sha256") != sha256 or expected.get("rows") != summary["rows"]:
                    raise RuntimeError(
                        f"{destination.name} does not match its manifest; "
                        "use force to rebuild the isolated evidence partition"
                    )
            else:
                if connection is None:
                    connection = database_connection()
                    configure_read_only_connection(connection)
                rows = _extract_execution_partition(
                    connection,
                    query,
                    destination,
                    config=config,
                    batch_start=batch_start,
                    batch_end=batch_end,
                )
                summary = execution_partition_summary(destination)
                if rows != summary["rows"]:
                    raise RuntimeError("streamed row count does not match Parquet metadata")
                sha256 = file_sha256(destination)
            manifest["partitions"].append(
                {"path": destination.name, "sha256": sha256, **summary}
            )
            batch_start = batch_end
    finally:
        if connection is not None:
            connection.close()

    manifest["totals"] = _aggregate_partition_summaries(manifest["partitions"])
    write_json_atomic(manifest_path, manifest)
    return manifest


def load_execution_evidence_manifest(
    config: ExecutionEvidenceConfig,
) -> dict[str, Any]:
    manifest_path = config.output_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("execution-evidence manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "source_contract": EXECUTION_EVIDENCE_CONTRACT,
        "source_schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "range_start": config.range_start.isoformat(),
        "range_end": config.range_end.isoformat(),
        "sample_interval_seconds": config.sample_interval_seconds,
        "min_seconds_after_open": config.min_seconds_after_open,
        "max_seconds_after_open": config.max_seconds_after_open,
        "freshness_seconds": config.freshness_seconds,
        "quantity": config.quantity,
        "primary_key": ["market_id", "observed_at"],
    }
    mismatches = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "execution-evidence manifest does not match configuration "
            f"({', '.join(mismatches)})"
        )
    for partition in manifest.get("partitions", []):
        partition_path = config.output_dir / partition["path"]
        if not partition_path.exists():
            raise RuntimeError(
                f"execution-evidence partition is missing: {partition_path.name}"
            )
        if file_sha256(partition_path) != partition["sha256"]:
            raise RuntimeError(
                f"execution-evidence partition hash mismatch: {partition_path.name}"
            )
    return manifest


def _extract_execution_partition(
    connection: psycopg.Connection[Any],
    query: str,
    destination: Path,
    *,
    config: ExecutionEvidenceConfig,
    batch_start: datetime,
    batch_end: datetime,
) -> int:
    temporary = destination.with_suffix(".parquet.partial")
    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        cursor_name = f"btc_execution_{batch_start:%Y%m%d}"
        with connection.transaction(), connection.cursor(name=cursor_name) as cursor:
            cursor.execute(
                query,
                {
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "min_seconds_after_open": config.min_seconds_after_open,
                    "max_seconds_after_open": config.max_seconds_after_open,
                    "sample_interval_milliseconds": (
                        config.sample_interval_seconds * 1_000
                    ),
                    "snapshot_schema_version": COMPACT_SNAPSHOT_SCHEMA_VERSION,
                    "freshness_seconds": config.freshness_seconds,
                },
            )
            while rows := cursor.fetchmany(10_000):
                records = [
                    dict(zip(EXECUTION_EVIDENCE_SCHEMA.names, row, strict=True))
                    for row in rows
                ]
                table = pa.Table.from_pylist(records, schema=EXECUTION_EVIDENCE_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        EXECUTION_EVIDENCE_SCHEMA,
                        compression="zstd",
                        write_statistics=True,
                        use_dictionary=[
                            "market_id",
                            "official_outcome",
                            "artifact_id",
                            "schema_version",
                        ],
                    )
                writer.write_table(table)
                row_count += len(rows)
    finally:
        if writer is not None:
            writer.close()
    if row_count == 0:
        pq.write_table(
            pa.Table.from_pylist([], schema=EXECUTION_EVIDENCE_SCHEMA),
            temporary,
            compression="zstd",
        )
    temporary.replace(destination)
    return row_count


def execution_partition_summary(path: Path) -> dict[str, Any]:
    table = pq.read_table(
        path,
        columns=[
            "market_id",
            "observed_at",
            "up_side_valid",
            "down_side_valid",
            "up_side_fresh",
            "down_side_fresh",
            "up_stale_initialized",
            "down_stale_initialized",
            "strict_both_side_eligible",
        ],
    )
    market_ids = table["market_id"].to_pylist()
    observed_at = table["observed_at"].to_pylist()
    keys = list(zip(market_ids, observed_at, strict=True))
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{path.name} contains duplicate market_id/observed_at keys")
    return {
        "rows": table.num_rows,
        "markets": len(set(market_ids)),
        "minimum_observed_at": (
            min(observed_at).isoformat() if observed_at else None
        ),
        "maximum_observed_at": (
            max(observed_at).isoformat() if observed_at else None
        ),
        "up_side_valid_rows": _true_count(table["up_side_valid"]),
        "down_side_valid_rows": _true_count(table["down_side_valid"]),
        "up_side_fresh_rows": _true_count(table["up_side_fresh"]),
        "down_side_fresh_rows": _true_count(table["down_side_fresh"]),
        "up_stale_initialized_rows": _true_count(table["up_stale_initialized"]),
        "down_stale_initialized_rows": _true_count(
            table["down_stale_initialized"]
        ),
        "strict_both_side_eligible_rows": _true_count(
            table["strict_both_side_eligible"]
        ),
    }


def _true_count(values: pa.ChunkedArray) -> int:
    return sum(value is True for value in values.to_pylist())


def _aggregate_partition_summaries(
    partitions: Sequence[dict[str, Any]],
) -> dict[str, int]:
    keys = (
        "rows",
        "markets",
        "up_side_valid_rows",
        "down_side_valid_rows",
        "up_side_fresh_rows",
        "down_side_fresh_rows",
        "up_stale_initialized_rows",
        "down_stale_initialized_rows",
        "strict_both_side_eligible_rows",
    )
    return {
        key: sum(int(partition[key]) for partition in partitions) for key in keys
    }


def _load_existing_manifest(
    manifest_path: Path,
    contract: dict[str, Any],
    *,
    force: bool,
) -> dict[str, Any] | None:
    if force:
        return None
    parquet_files = list(manifest_path.parent.glob("*.parquet"))
    if not manifest_path.exists():
        if parquet_files:
            raise RuntimeError(
                "execution-evidence partitions exist without a manifest; "
                "use force only for an intentional isolated rebuild"
            )
        return None
    existing = json.loads(manifest_path.read_text())
    mismatches = [
        key for key, expected in contract.items() if existing.get(key) != expected
    ]
    if mismatches:
        raise RuntimeError(
            "execution-evidence cache contract changed "
            f"({', '.join(mismatches)}); use a new isolated path or force"
        )
    return existing
