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

from .contract import DIRECT_CONTRACT_VERSION, PMDATA_CONTRACT_VERSION, schema
from .normalize import factual_identity, identity, normalize_canonical, normalize_legacy

CANONICAL_QUERY = """
SELECT source,feed_id,source_timestamp,valid_from_timestamp,provider_available_at,
  received_at,price,bid,ask,report_sha256::text,payload_sha256::text,strategy_key,
  capture_artifact_id,ingested_at,expires_at,report_version,source_date,
  archive_row_number,backfill_artifact_id,report_hash_kind
FROM market_data.chainlink_btcusd_reference_prices
WHERE source_timestamp >= %s AND source_timestamp < %s
ORDER BY source_timestamp,feed_id,report_sha256
"""
LEGACY_QUERY = """
SELECT feed_id,source_timestamp,valid_from_timestamp,price,bid,ask,
  report_sha256,artifact_id,ingested_at
FROM polymarket.chainlink_btcusd_archive_ticks
WHERE source_timestamp >= %s AND source_timestamp < %s
ORDER BY source_timestamp,feed_id
"""


def connection() -> psycopg.Connection[Any]:
    password = os.environ.get("CHAINLINK_DRAIN_DB_PASSWORD") or os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("CHAINLINK_DRAIN_DB_PASSWORD or POSTGRES_PASSWORD is required")
    db = psycopg.connect(
        host=os.environ.get("CHAINLINK_DRAIN_DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("CHAINLINK_DRAIN_DB_PORT", "6432")),
        dbname=os.environ.get("CHAINLINK_DRAIN_DB_NAME", "polymarket"),
        user=os.environ.get("CHAINLINK_DRAIN_DB_USER", "postgres"),
        password=password,
        autocommit=True,
        row_factory=dict_row,
    )
    db.execute("SET default_transaction_read_only = on")
    db.execute("SET statement_timeout = '5min'")
    db.execute("SET lock_timeout = '1s'")
    return db


def rows(cursor: psycopg.Cursor[Any], query: str, start: datetime, end: datetime) -> Iterator[dict[str, Any]]:
    cursor.execute(query, (start, end))
    while batch := cursor.fetchmany(2_000):
        yield from batch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_product(
    root: Path,
    product: str,
    contract_version: str,
    start: datetime,
    records: list[dict[str, Any]],
    source_counts: dict[str, int],
    duplicate_count: int,
) -> dict[str, Any]:
    directory = root / product / f"date={start:%Y-%m-%d}"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"hour={start:%H}.parquet"
    table = pa.Table.from_pylist(records, schema=schema(contract_version))
    temporary = destination.with_suffix(".parquet.partial")
    pq.write_table(table, temporary, compression="zstd", row_group_size=25_000)
    temporary.replace(destination)
    manifest = {
        "contract_version": contract_version,
        "product": product,
        "window_start": start.isoformat(),
        "window_end": (start + timedelta(hours=1)).isoformat(),
        "source_counts": source_counts,
        "source_total": sum(source_counts.values()),
        "duplicate_count": duplicate_count,
        "conflict_count": 0,
        "output_count": table.num_rows,
        "file": str(destination),
        "file_size_bytes": destination.stat().st_size,
        "file_sha256": sha256(destination),
    }
    if manifest["source_total"] != manifest["output_count"] + duplicate_count:
        raise RuntimeError(f"row accounting failed for {product} {start.isoformat()}")
    manifest_path = destination.with_suffix(".manifest.json")
    partial = manifest_path.with_suffix(".json.partial")
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    partial.replace(manifest_path)
    return manifest


def existing_partition(root: Path, product: str, start: datetime) -> dict[str, Any] | None:
    destination = root / product / f"date={start:%Y-%m-%d}" / f"hour={start:%H}.parquet"
    manifest_path = destination.with_suffix(".manifest.json")
    if not destination.exists() and not manifest_path.exists():
        return None
    if not destination.exists() or not manifest_path.exists():
        raise RuntimeError(f"partial existing partition: {destination}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["file_sha256"] != sha256(destination):
        raise RuntimeError(f"checksum mismatch: {destination}")
    return manifest


def drain_hour(db: psycopg.Connection[Any], root: Path, start: datetime) -> tuple[dict[str, Any], dict[str, Any]]:
    existing_direct = existing_partition(root, "direct", start)
    existing_pmdata = existing_partition(root, "pmdata", start)
    if existing_direct is not None and existing_pmdata is not None:
        return existing_direct, existing_pmdata
    if existing_direct is not None or existing_pmdata is not None:
        raise RuntimeError(f"product partitions are incomplete for {start.isoformat()}")

    direct: dict[tuple[str, Any, str], dict[str, Any]] = {}
    direct_facts: dict[tuple[str, Any, str], tuple[Any, ...]] = {}
    pmdata: dict[tuple[str, Any, str], dict[str, Any]] = {}
    direct_counts = {"canonical_direct": 0, "legacy": 0}
    pmdata_counts = {"canonical_pmdata": 0}
    direct_duplicates = 0
    pmdata_duplicates = 0
    end = start + timedelta(hours=1)
    with db.cursor() as cursor:
        for raw in rows(cursor, CANONICAL_QUERY, start, end):
            record = normalize_canonical(raw)
            key = identity(record)
            if record["source"] == "chainlink_data_streams":
                direct_counts["canonical_direct"] += 1
                if key in direct:
                    if direct_facts[key] != factual_identity(record):
                        raise RuntimeError(f"conflicting direct identity in {start.isoformat()}: {key}")
                    direct_duplicates += 1
                else:
                    direct[key] = record
                    direct_facts[key] = factual_identity(record)
            elif record["source"] == "pmdata_chainlink_streams":
                pmdata_counts["canonical_pmdata"] += 1
                if key in pmdata:
                    if factual_identity(pmdata[key]) != factual_identity(record):
                        raise RuntimeError(f"conflicting PMData identity in {start.isoformat()}: {key}")
                    pmdata_duplicates += 1
                else:
                    pmdata[key] = record
            else:
                raise RuntimeError(f"unknown source {record['source']!r}")
        for raw in rows(cursor, LEGACY_QUERY, start, end):
            record = normalize_legacy(raw)
            key = identity(record)
            direct_counts["legacy"] += 1
            if key in direct:
                if direct_facts[key] != factual_identity(record):
                    raise RuntimeError(f"conflicting legacy identity in {start.isoformat()}: {key}")
                direct_duplicates += 1
            else:
                direct[key] = record
                direct_facts[key] = factual_identity(record)

    ordering = lambda row: (row["source_timestamp"], row["feed_id"], row["report_sha256"])
    direct_manifest = write_product(
        root,
        "direct",
        DIRECT_CONTRACT_VERSION,
        start,
        sorted(direct.values(), key=ordering),
        direct_counts,
        direct_duplicates,
    )
    pmdata_manifest = write_product(
        root,
        "pmdata",
        PMDATA_CONTRACT_VERSION,
        start,
        sorted(pmdata.values(), key=ordering),
        pmdata_counts,
        pmdata_duplicates,
    )
    return direct_manifest, pmdata_manifest


def refresh_manifest(root: Path, product: str, contract_version: str) -> dict[str, Any]:
    partitions = [
        json.loads(path.read_text())
        for path in sorted((root / product).glob("date=*/hour=*.manifest.json"))
    ]
    if not partitions:
        raise RuntimeError(f"no {product} partitions")
    source_keys = sorted({key for part in partitions for key in part["source_counts"]})
    manifest = {
        "contract_version": contract_version,
        "product": product,
        "window_start": min(part["window_start"] for part in partitions),
        "window_end": max(part["window_end"] for part in partitions),
        "partition_count": len(partitions),
        "source_counts": {
            key: sum(part["source_counts"].get(key, 0) for part in partitions)
            for key in source_keys
        },
        "source_total": sum(part["source_total"] for part in partitions),
        "duplicate_count": sum(part["duplicate_count"] for part in partitions),
        "conflict_count": sum(part["conflict_count"] for part in partitions),
        "output_count": sum(part["output_count"] for part in partitions),
        "file_size_bytes": sum(part["file_size_bytes"] for part in partitions),
        "partitions": [part["file"] for part in partitions],
    }
    temporary = root / product / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / product / "manifest.json")
    return manifest


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def hour_floor(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy Chainlink reference-price tables to canonical Parquet")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", required=True, type=parse_timestamp)
    parser.add_argument("--end", required=True, type=parse_timestamp)
    args = parser.parse_args()
    start = hour_floor(args.start)
    end = hour_floor(args.end - timedelta(microseconds=1)) + timedelta(hours=1)
    if end <= start:
        parser.error("--end must be after --start")
    args.output.mkdir(parents=True, exist_ok=True)
    with connection() as db:
        current = start
        while current < end:
            direct, pmdata = drain_hour(db, args.output, current)
            print(json.dumps({"completed": current.isoformat(), "direct": direct["output_count"], "pmdata": pmdata["output_count"]}), flush=True)
            current += timedelta(hours=1)
    refresh_manifest(args.output, "direct", DIRECT_CONTRACT_VERSION)
    refresh_manifest(args.output, "pmdata", PMDATA_CONTRACT_VERSION)


if __name__ == "__main__":
    main()
