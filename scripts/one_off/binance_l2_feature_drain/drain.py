from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from psycopg.rows import dict_row

from .contract import CONTRACT_COLUMNS, PRODUCTS, schema
from .normalize import factual_identity, identity, normalize


def connection() -> psycopg.Connection[Any]:
    password = os.environ.get("BINANCE_L2_DRAIN_DB_PASSWORD") or os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("BINANCE_L2_DRAIN_DB_PASSWORD or POSTGRES_PASSWORD is required")
    db = psycopg.connect(
        host=os.environ.get("BINANCE_L2_DRAIN_DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("BINANCE_L2_DRAIN_DB_PORT", "6432")),
        dbname=os.environ.get("BINANCE_L2_DRAIN_DB_NAME", "polymarket"),
        user=os.environ.get("BINANCE_L2_DRAIN_DB_USER", "postgres"),
        password=password, autocommit=True, row_factory=dict_row,
    )
    db.execute("SET default_transaction_read_only = on")
    db.execute("SET statement_timeout = '5min'")
    db.execute("SET lock_timeout = '1s'")
    return db


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_partition(destination: Path) -> dict[str, Any] | None:
    manifest_path = destination.with_suffix(".manifest.json")
    if not destination.exists() and not manifest_path.exists():
        return None
    if not destination.exists() or not manifest_path.exists():
        raise RuntimeError(f"partial partition exists: {destination}")
    manifest = json.loads(manifest_path.read_text())
    if sha256(destination) != manifest["file_sha256"]:
        raise RuntimeError(f"checksum mismatch: {destination}")
    return manifest


def drain_hour(db: psycopg.Connection[Any], root: Path, product: str, start: datetime) -> dict[str, Any]:
    config = PRODUCTS[product]
    destination = root / f"date={start:%Y-%m-%d}" / f"hour={start:%H}.parquet"
    if prior := existing_partition(destination):
        return prior
    end = start + timedelta(hours=1)
    records: dict[tuple[Any, Any], dict[str, Any]] = {}
    facts: dict[tuple[Any, Any], tuple[Any, ...]] = {}
    source_counts: dict[str, int] = {}
    duplicate_count = 0
    conflicts: list[dict[str, Any]] = []
    columns = ",".join(CONTRACT_COLUMNS)
    with db.cursor() as cursor:
        for table in config["tables"]:
            source_counts[table] = 0
            cursor.execute(
                f"SELECT {columns} FROM {table} WHERE second_start >= %s AND second_start < %s ORDER BY second_start,symbol,artifact_id",
                (start, end),
            )
            while batch := cursor.fetchmany(2_000):
                for raw in batch:
                    source_counts[table] += 1
                    record = normalize(raw)
                    key = identity(record)
                    fact = factual_identity(record)
                    if key in records:
                        if facts[key] != fact:
                            conflicts.append({"symbol": key[0], "second_start": key[1].isoformat(), "source_table": table})
                        else:
                            duplicate_count += 1
                    else:
                        records[key], facts[key] = record, fact
    if conflicts:
        conflict_path = root / "conflicts" / f"date={start:%Y-%m-%d}-hour={start:%H}.json"
        conflict_path.parent.mkdir(parents=True, exist_ok=True)
        conflict_path.write_text(json.dumps(conflicts, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"{len(conflicts)} conflicting identities in {product} {start.isoformat()}")
    ordered = sorted(records.values(), key=lambda row: (row["second_start"], row["symbol"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(ordered, schema=schema(config["contract_version"]))
    temporary = destination.with_suffix(".parquet.partial")
    pq.write_table(table, temporary, compression="zstd", row_group_size=25_000)
    temporary.replace(destination)
    manifest = {
        "contract_version": config["contract_version"], "product": product,
        "window_start": start.isoformat(), "window_end": end.isoformat(),
        "source_counts": source_counts, "source_total": sum(source_counts.values()),
        "output_count": table.num_rows, "duplicate_count": duplicate_count,
        "conflict_count": 0, "file": str(destination),
        "file_size_bytes": destination.stat().st_size, "file_sha256": sha256(destination),
    }
    if manifest["source_total"] != manifest["output_count"] + duplicate_count:
        raise RuntimeError(f"row accounting failed for {product} {start.isoformat()}")
    manifest_path = destination.with_suffix(".manifest.json")
    partial = manifest_path.with_suffix(".json.partial")
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    partial.replace(manifest_path)
    return manifest


def refresh_manifest(root: Path, product: str) -> None:
    parts = [json.loads(path.read_text()) for path in sorted(root.glob("date=*/hour=*.manifest.json"))]
    source_names = sorted({name for part in parts for name in part["source_counts"]})
    result = {
        "contract_version": PRODUCTS[product]["contract_version"], "product": product,
        "window_start": min(part["window_start"] for part in parts),
        "window_end": max(part["window_end"] for part in parts),
        "partition_count": len(parts),
        "source_counts": {name: sum(part["source_counts"].get(name, 0) for part in parts) for name in source_names},
        "source_total": sum(part["source_total"] for part in parts),
        "output_count": sum(part["output_count"] for part in parts),
        "duplicate_count": sum(part["duplicate_count"] for part in parts),
        "conflict_count": sum(part["conflict_count"] for part in parts),
        "file_size_bytes": sum(part["file_size_bytes"] for part in parts),
    }
    temporary = root / "manifest.json.partial"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / "manifest.json")


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed).astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy and deduplicate Binance L2 feature tables to canonical Parquet")
    parser.add_argument("product", choices=PRODUCTS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", required=True, type=parse_timestamp)
    parser.add_argument("--end", required=True, type=parse_timestamp)
    args = parser.parse_args()
    start = args.start.replace(minute=0, second=0, microsecond=0)
    end = (args.end - timedelta(microseconds=1)).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    root = args.output / args.product
    root.mkdir(parents=True, exist_ok=True)
    with connection() as db:
        current = start
        while current < end:
            manifest = drain_hour(db, root, args.product, current)
            print(json.dumps({"completed": current.isoformat(), "rows": manifest["output_count"]}), flush=True)
            current += timedelta(hours=1)
    refresh_manifest(root, args.product)


if __name__ == "__main__":
    main()
