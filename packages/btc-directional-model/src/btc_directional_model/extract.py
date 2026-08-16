from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

from .config import TrainingConfig

SOURCE_SCHEMA = pa.schema(
    [
        ("market_id", pa.string()),
        ("window_start", pa.timestamp("us", tz="UTC")),
        ("window_end", pa.timestamp("us", tz="UTC")),
        ("official_outcome", pa.string()),
        ("label_up", pa.int32()),
        ("min_tick_size", pa.float64()),
        ("min_order_size", pa.float64()),
        ("fee_rate", pa.float64()),
        ("opening_boundary", pa.float64()),
        ("final_price", pa.float64()),
        ("observed_at", pa.timestamp("us", tz="UTC")),
        ("seconds_elapsed", pa.int32()),
        ("btc_open", pa.float64()),
        ("btc_high", pa.float64()),
        ("btc_low", pa.float64()),
        ("btc_close", pa.float64()),
        ("btc_base_volume", pa.float64()),
        ("btc_quote_volume", pa.float64()),
        ("trade_count", pa.int64()),
        ("btc_taker_buy_base_volume", pa.float64()),
        ("btc_taker_buy_quote_volume", pa.float64()),
        ("up_provider_received_at", pa.timestamp("us", tz="UTC")),
        ("up_best_bid", pa.float64()),
        ("up_best_ask", pa.float64()),
        ("up_best_bid_size", pa.float64()),
        ("up_best_ask_size", pa.float64()),
        ("up_bid_depth", pa.float64()),
        ("up_ask_depth", pa.float64()),
        ("up_ask_vwap_1", pa.float64()),
        ("up_ask_vwap_5", pa.float64()),
        ("up_ask_vwap_10", pa.float64()),
        ("up_imbalance", pa.float64()),
        ("down_provider_received_at", pa.timestamp("us", tz="UTC")),
        ("down_best_bid", pa.float64()),
        ("down_best_ask", pa.float64()),
        ("down_best_bid_size", pa.float64()),
        ("down_best_ask_size", pa.float64()),
        ("down_bid_depth", pa.float64()),
        ("down_ask_depth", pa.float64()),
        ("down_ask_vwap_1", pa.float64()),
        ("down_ask_vwap_5", pa.float64()),
        ("down_ask_vwap_10", pa.float64()),
        ("down_imbalance", pa.float64()),
        ("quality_flags", pa.int32()),
    ]
)


def extract_source(config: TrainingConfig, *, force: bool = False) -> dict[str, Any]:
    output_dir = config.paths.source_data
    output_dir.mkdir(parents=True, exist_ok=True)
    query_path = config.package_root / "sql" / "directional-source.sql"
    query = query_path.read_text()
    manifest_path = output_dir / "manifest.json"
    contract: dict[str, Any] = {
        "range_start": config.data.range_start.isoformat(),
        "range_end": config.data.range_end.isoformat(),
        "strict_final_price_audit": config.data.strict_final_price_audit,
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
    }
    existing_manifest = load_existing_manifest(manifest_path, contract, force=force)
    existing_partitions = {
        row["path"]: row for row in (existing_manifest or {}).get("partitions", [])
    }
    manifest: dict[str, Any] = {
        **contract,
        "partitions": [],
    }

    connection: psycopg.Connection[Any] | None = None
    try:
        batch_start = config.data.range_start
        while batch_start < config.data.range_end:
            batch_end = min(batch_start + timedelta(days=1), config.data.range_end)
            destination = output_dir / f"{batch_start.date().isoformat()}.parquet"
            if destination.exists() and not force:
                metadata = pq.read_metadata(destination)
                print(f"extract: reuse {destination.name} ({metadata.num_rows:,} rows)", flush=True)
                rows = metadata.num_rows
                sha256 = file_sha256(destination)
                expected = existing_partitions.get(destination.name)
                if existing_manifest is not None and expected is None:
                    raise RuntimeError(
                        f"{destination.name} is not recorded in the manifest; "
                        "rerun extraction with --force"
                    )
                if expected is not None and (
                    expected.get("sha256") != sha256 or expected.get("rows") != rows
                ):
                    raise RuntimeError(
                        f"{destination.name} does not match its manifest; rerun extraction with --force"
                    )
            else:
                if connection is None:
                    connection = database_connection()
                    connection.execute("SET default_transaction_read_only = on")
                    connection.execute("SET statement_timeout = '10min'")
                rows = extract_partition(
                    connection,
                    query,
                    destination,
                    batch_start=batch_start,
                    batch_end=batch_end,
                    strict_final_price_audit=config.data.strict_final_price_audit,
                )
                print(f"extract: wrote {destination.name} ({rows:,} rows)", flush=True)
                sha256 = file_sha256(destination)
            manifest["partitions"].append(
                {
                    "path": destination.name,
                    "rows": rows,
                    "sha256": sha256,
                }
            )
            batch_start = batch_end
    finally:
        if connection is not None:
            connection.close()

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def load_existing_manifest(
    manifest_path: Path, contract: dict[str, Any], *, force: bool
) -> dict[str, Any] | None:
    parquet_files = list(manifest_path.parent.glob("*.parquet"))
    if force:
        return None
    if not manifest_path.exists():
        if parquet_files:
            raise RuntimeError(
                "source partitions exist without a provenance manifest; rerun extraction with --force"
            )
        return None
    existing = json.loads(manifest_path.read_text())
    immutable_keys = ("strict_final_price_audit", "query_sha256")
    mismatches = [key for key in immutable_keys if existing.get(key) != contract.get(key)]
    if mismatches:
        joined = ", ".join(mismatches)
        raise RuntimeError(
            f"source cache contract changed ({joined}); rerun extraction with --force"
        )
    return existing


def database_connection() -> psycopg.Connection[Any]:
    password = os.environ.get("BTC_MODEL_DB_PASSWORD") or os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("BTC_MODEL_DB_PASSWORD or POSTGRES_PASSWORD is required")
    return psycopg.connect(
        host=os.environ.get("BTC_MODEL_DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("BTC_MODEL_DB_PORT", "6432")),
        dbname=os.environ.get("BTC_MODEL_DB_NAME", "polymarket"),
        user=os.environ.get("BTC_MODEL_DB_USER", "postgres"),
        password=password,
        autocommit=True,
    )


def extract_partition(
    connection: psycopg.Connection[Any],
    query: str,
    destination: Path,
    *,
    batch_start: Any,
    batch_end: Any,
    strict_final_price_audit: bool,
) -> int:
    temporary = destination.with_suffix(".parquet.partial")
    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        cursor_name = f"btc_model_{batch_start:%Y%m%d}"
        with connection.transaction(), connection.cursor(name=cursor_name) as cursor:
            cursor.execute(
                query,
                {
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "strict_final_price_audit": strict_final_price_audit,
                },
            )
            while rows := cursor.fetchmany(10_000):
                records = [dict(zip(SOURCE_SCHEMA.names, row, strict=True)) for row in rows]
                table = pa.Table.from_pylist(records, schema=SOURCE_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, SOURCE_SCHEMA, compression="zstd")
                writer.write_table(table)
                row_count += len(rows)
    finally:
        if writer is not None:
            writer.close()

    if row_count == 0:
        pq.write_table(
            pa.Table.from_pylist([], schema=SOURCE_SCHEMA), temporary, compression="zstd"
        )
    temporary.replace(destination)
    return row_count


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
