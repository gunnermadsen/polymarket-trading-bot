from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

from .core_config import CoreTrainingConfig, evaluation_holdout_range

CoreScope = Literal["pre_holdout", "holdout"]
CORE_SOURCE_SCHEMA_VERSION = "btc-core-source-v1"
CORE_SOURCE_SCHEMA = pa.schema(
    [
        ("market_id", pa.string()),
        ("window_start", pa.timestamp("us", tz="UTC")),
        ("window_end", pa.timestamp("us", tz="UTC")),
        ("official_outcome", pa.string()),
        ("label_up", pa.int32()),
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
    ]
)


def extract_core_source(
    config: CoreTrainingConfig,
    scope: CoreScope,
    *,
    force: bool = False,
) -> dict[str, Any]:
    range_start, range_end = scope_range(config, scope)
    output_dir = config.paths.source_data
    output_dir.mkdir(parents=True, exist_ok=True)
    query_path = config.package_root / "sql" / "btc-core-source.sql"
    query = query_path.read_text()
    manifest_path = output_dir / f"manifest-{scope}.json"
    contract: dict[str, Any] = {
        "source_contract": config.data.source_contract,
        "source_schema_version": CORE_SOURCE_SCHEMA_VERSION,
        "source_schema_sha256": hashlib.sha256(
            CORE_SOURCE_SCHEMA.to_string().encode()
        ).hexdigest(),
        "scope": scope,
        "range_start": range_start.isoformat(),
        "range_end": range_end.isoformat(),
        "strict_final_price_audit": config.data.strict_final_price_audit,
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
    }
    existing_manifest = load_existing_manifest(manifest_path, contract, force=force)
    existing_partitions = {
        row["path"]: row for row in (existing_manifest or {}).get("partitions", [])
    }
    manifest: dict[str, Any] = {**contract, "partitions": []}

    connection: psycopg.Connection[Any] | None = None
    try:
        batch_start = range_start
        while batch_start < range_end:
            batch_end = min(batch_start + timedelta(days=1), range_end)
            destination = output_dir / f"{batch_start.date().isoformat()}.parquet"
            expected = existing_partitions.get(destination.name)
            if destination.exists() and not force:
                if expected is None:
                    raise RuntimeError(
                        f"{destination.name} is not recorded in {manifest_path.name}; "
                        "use --force only for an intentional isolated rebuild"
                    )
                summary = partition_summary(destination)
                sha256 = file_sha256(destination)
                if expected.get("sha256") != sha256 or expected.get("rows") != summary["rows"]:
                    raise RuntimeError(
                        f"{destination.name} does not match its manifest; "
                        "use --force to rebuild the isolated core partition"
                    )
                print(
                    f"core extract: reuse {destination.name} "
                    f"({summary['rows']:,} rows/{summary['markets']:,} markets)",
                    flush=True,
                )
            else:
                if connection is None:
                    connection = database_connection()
                    configure_read_only_connection(connection)
                rows = extract_partition(
                    connection,
                    query,
                    destination,
                    batch_start=batch_start,
                    batch_end=batch_end,
                    strict_final_price_audit=config.data.strict_final_price_audit,
                )
                summary = partition_summary(destination)
                if rows != summary["rows"]:
                    raise RuntimeError("streamed row count does not match Parquet metadata")
                sha256 = file_sha256(destination)
                print(
                    f"core extract: wrote {destination.name} "
                    f"({rows:,} rows/{summary['markets']:,} markets)",
                    flush=True,
                )
            manifest["partitions"].append(
                {
                    "path": destination.name,
                    "sha256": sha256,
                    **summary,
                }
            )
            batch_start = batch_end
    finally:
        if connection is not None:
            connection.close()

    manifest["totals"] = aggregate_partition_summaries(manifest["partitions"])
    write_json_atomic(manifest_path, manifest)
    return manifest


def scope_range(
    config: CoreTrainingConfig, scope: CoreScope
) -> tuple[datetime, datetime]:
    if scope == "pre_holdout":
        return config.data.range_start, config.split.holdout_start
    if scope == "holdout":
        return evaluation_holdout_range(config)
    raise ValueError(f"unsupported core extraction scope: {scope}")


def configure_read_only_connection(connection: psycopg.Connection[Any]) -> None:
    connection.execute("SET default_transaction_read_only = on")
    connection.execute("SET statement_timeout = '10min'")
    connection.execute("SET lock_timeout = '5s'")
    connection.execute("SET work_mem = '32MB'")


def database_connection() -> psycopg.Connection[Any]:
    password = os.environ.get("BTC_MODEL_DB_PASSWORD") or os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("BTC_MODEL_DB_PASSWORD or POSTGRES_PASSWORD is required")
    return psycopg.connect(
        host=os.environ.get("BTC_MODEL_DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("BTC_MODEL_DB_PORT", "55433")),
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
    batch_start: datetime,
    batch_end: datetime,
    strict_final_price_audit: bool,
) -> int:
    temporary = destination.with_suffix(".parquet.partial")
    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        cursor_name = f"btc_core_{batch_start:%Y%m%d}"
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
                records = [
                    dict(zip(CORE_SOURCE_SCHEMA.names, row, strict=True)) for row in rows
                ]
                table = pa.Table.from_pylist(records, schema=CORE_SOURCE_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        CORE_SOURCE_SCHEMA,
                        compression="zstd",
                        write_statistics=True,
                    )
                writer.write_table(table)
                row_count += len(rows)
    finally:
        if writer is not None:
            writer.close()
    if row_count == 0:
        pq.write_table(
            pa.Table.from_pylist([], schema=CORE_SOURCE_SCHEMA),
            temporary,
            compression="zstd",
        )
    temporary.replace(destination)
    return row_count


def partition_summary(path: Path) -> dict[str, Any]:
    table = pq.read_table(
        path,
        columns=["market_id", "seconds_elapsed", "final_price"],
    )
    market_ids = table["market_id"].to_pylist()
    seconds = table["seconds_elapsed"].to_pylist()
    final_prices = table["final_price"].to_pylist()
    counts = Counter(market_ids)
    final_markets = {
        market_id
        for market_id, final_price in zip(market_ids, final_prices, strict=True)
        if final_price is not None
    }
    return {
        "rows": table.num_rows,
        "markets": len(counts),
        "complete_300_row_markets": sum(count == 300 for count in counts.values()),
        "incomplete_markets": sum(count != 300 for count in counts.values()),
        "markets_with_final_price": len(final_markets),
        "minimum_second": min(seconds) if seconds else None,
        "maximum_second": max(seconds) if seconds else None,
    }


def aggregate_partition_summaries(partitions: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "rows",
        "markets",
        "complete_300_row_markets",
        "incomplete_markets",
        "markets_with_final_price",
    )
    return {key: sum(int(partition[key]) for partition in partitions) for key in keys}


def load_existing_manifest(
    manifest_path: Path,
    contract: dict[str, Any],
    *,
    force: bool,
) -> dict[str, Any] | None:
    if force or not manifest_path.exists():
        return None
    existing = json.loads(manifest_path.read_text())
    mismatches = [
        key for key, expected in contract.items() if existing.get(key) != expected
    ]
    if mismatches:
        raise RuntimeError(
            "core source cache contract changed "
            f"({', '.join(mismatches)}); use a new isolated path or --force"
        )
    return existing


def load_core_manifest(
    config: CoreTrainingConfig, scope: CoreScope
) -> dict[str, Any]:
    path = config.paths.source_data / f"manifest-{scope}.json"
    if not path.exists():
        raise RuntimeError(f"{path.name} is missing; extract {scope} source first")
    manifest = json.loads(path.read_text())
    expected_start, expected_end = scope_range(config, scope)
    expected = {
        "source_contract": config.data.source_contract,
        "source_schema_version": CORE_SOURCE_SCHEMA_VERSION,
        "scope": scope,
        "range_start": expected_start.isoformat(),
        "range_end": expected_end.isoformat(),
    }
    mismatches = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"{path.name} does not match configuration ({', '.join(mismatches)})"
        )
    for partition in manifest.get("partitions", []):
        partition_path = config.paths.source_data / partition["path"]
        if not partition_path.exists():
            raise RuntimeError(f"core source partition is missing: {partition_path.name}")
        if file_sha256(partition_path) != partition["sha256"]:
            raise RuntimeError(
                f"core source partition hash mismatch: {partition_path.name}"
            )
    return manifest


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
