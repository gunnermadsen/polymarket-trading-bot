from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from psycopg.rows import dict_row

from .contract import CONTRACT_VERSION, schema
from .normalize import normalize_canonical, normalize_legacy

LEGACY_QUERY = """
SELECT checkpoint.checkpoint_id, checkpoint.source_timestamp,
  checkpoint.received_at, checkpoint.persisted_at, checkpoint.connection_id,
  checkpoint.ingest_sequence, checkpoint.market_id, checkpoint.token_id,
  checkpoint.best_bid, checkpoint.best_ask, checkpoint.tick_size,
  checkpoint.book, checkpoint.source_hash, market.condition_id,
  market.event_slug, market.window_start, market.window_end,
  CASE WHEN checkpoint.token_id = market.up_token_id THEN 'up'
       WHEN checkpoint.token_id = market.down_token_id THEN 'down' END AS outcome
FROM polymarket.orderbook_checkpoints checkpoint
LEFT JOIN polymarket.btc_interval_markets market
  ON market.market_id = checkpoint.market_id
 AND checkpoint.token_id IN (market.up_token_id, market.down_token_id)
WHERE checkpoint.source_timestamp >= %s AND checkpoint.source_timestamp < %s
ORDER BY checkpoint.source_timestamp, checkpoint.checkpoint_id
"""
CANONICAL_QUERY = """
SELECT sampled_at, source_timestamp, provider_available_at, received_at,
  source, market_id, condition_id, event_slug, window_start, window_end,
  token_id, outcome, connection_epoch, ingest_sequence, tick_size, best_bid,
  best_ask, bid_depth, ask_depth, bids, asks, source_hash, book_sha256,
  sampling_policy, sampling_policy_sha256, payload_sha256, strategy_key,
  capture_artifact_id, ingested_at
FROM market_data.polymarket_btc_five_minute_orderbook_snapshots
WHERE source_timestamp >= %s AND source_timestamp < %s
ORDER BY source_timestamp, sampled_at, market_id, token_id
"""


def connection() -> psycopg.Connection[Any]:
    password = os.environ.get("ORDERBOOK_DRAIN_DB_PASSWORD") or os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("ORDERBOOK_DRAIN_DB_PASSWORD or POSTGRES_PASSWORD is required")
    result = psycopg.connect(
        host=os.environ.get("ORDERBOOK_DRAIN_DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("ORDERBOOK_DRAIN_DB_PORT", "6432")),
        dbname=os.environ.get("ORDERBOOK_DRAIN_DB_NAME", "polymarket"),
        user=os.environ.get("ORDERBOOK_DRAIN_DB_USER", "postgres"),
        password=password,
        autocommit=True,
        row_factory=dict_row,
    )
    result.execute("SET default_transaction_read_only = on")
    result.execute("SET statement_timeout = '5min'")
    result.execute("SET lock_timeout = '1s'")
    return result


def rows(cursor: psycopg.Cursor[Any], query: str, start: datetime, end: datetime) -> Iterator[dict[str, Any]]:
    cursor.execute(query, (start, end))
    while batch := cursor.fetchmany(2_000):
        yield from batch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def drain_hour(db: psycopg.Connection[Any], root: Path, start: datetime) -> dict[str, Any]:
    end = start + timedelta(hours=1)
    directory = root / f"date={start:%Y-%m-%d}"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"hour={start:%H}.parquet"
    manifest_path = destination.with_suffix(".manifest.json")
    if destination.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["file_sha256"] == file_sha256(destination):
            return manifest
        raise RuntimeError(f"existing partition failed checksum validation: {destination}")

    records: list[dict[str, Any]] = []
    source_counts = {"legacy": 0, "canonical": 0}
    unmapped = 0
    with db.cursor() as cursor:
        for row in rows(cursor, LEGACY_QUERY, start, end):
            source_counts["legacy"] += 1
            if row["condition_id"] is None or row["outcome"] is None:
                unmapped += 1
            records.append(normalize_legacy(row))
        for row in rows(cursor, CANONICAL_QUERY, start, end):
            source_counts["canonical"] += 1
            records.append(normalize_canonical(row))
    if unmapped:
        raise RuntimeError(f"{unmapped} legacy records cannot be mapped in {start.isoformat()}")

    unique: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for record in records:
        identity = record["archive_record_sha256"]
        if identity in unique:
            duplicate_count += 1
        else:
            unique[identity] = record
    output = sorted(unique.values(), key=lambda row: (
        row["source_timestamp"], row["archive_source_relation"], row["archive_source_record_id"]
    ))
    table = pa.Table.from_pylist(output, schema=schema())
    temporary = destination.with_suffix(".parquet.partial")
    pq.write_table(table, temporary, compression="zstd", row_group_size=25_000)
    temporary.replace(destination)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "source_counts": source_counts,
        "source_total": sum(source_counts.values()),
        "duplicate_count": duplicate_count,
        "output_count": table.num_rows,
        "unmapped_count": unmapped,
        "file": str(destination),
        "file_size_bytes": destination.stat().st_size,
        "file_sha256": file_sha256(destination),
    }
    if manifest["source_total"] != manifest["output_count"] + duplicate_count:
        raise RuntimeError(f"row accounting failed for {start.isoformat()}")
    temporary_manifest = manifest_path.with_suffix(".json.partial")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_path)
    return manifest


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def hour_floor(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Drain Polymarket orderbook tables to canonical Parquet")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", required=True, type=parse_timestamp)
    parser.add_argument("--end", required=True, type=parse_timestamp)
    args = parser.parse_args()
    start, end = hour_floor(args.start), hour_floor(args.end - timedelta(microseconds=1)) + timedelta(hours=1)
    if end <= start:
        parser.error("--end must be after --start")
    args.output.mkdir(parents=True, exist_ok=True)
    totals = {"source_total": 0, "duplicate_count": 0, "output_count": 0, "file_size_bytes": 0}
    partitions = []
    with connection() as db:
        current = start
        while current < end:
            manifest = drain_hour(db, args.output, current)
            partitions.append(manifest)
            for key in totals:
                totals[key] += manifest[key]
            print(json.dumps({"completed": current.isoformat(), **totals}), flush=True)
            current += timedelta(hours=1)
    run_manifest = {
        "contract_version": CONTRACT_VERSION,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "partition_count": len(partitions),
        **totals,
        "partitions": [item["file"] for item in partitions],
    }
    (args.output / "manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
