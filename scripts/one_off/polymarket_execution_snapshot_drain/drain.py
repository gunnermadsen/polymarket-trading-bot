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

from .contract import (
    BASE_COLUMNS, CONTRACT_COLUMNS, CONTRACT_VERSION, EXPANDED_VWAP_COLUMNS,
    SOURCES, schema,
)
from .normalize import (
    compatible_measurements, expanded_fact, identity, normalize, shared_fact,
)


def connection() -> psycopg.Connection[Any]:
    password = os.environ.get("EXECUTION_SNAPSHOT_DRAIN_DB_PASSWORD") or os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("EXECUTION_SNAPSHOT_DRAIN_DB_PASSWORD or POSTGRES_PASSWORD is required")
    db = psycopg.connect(
        host=os.environ.get("EXECUTION_SNAPSHOT_DRAIN_DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("EXECUTION_SNAPSHOT_DRAIN_DB_PORT", "6432")),
        dbname=os.environ.get("EXECUTION_SNAPSHOT_DRAIN_DB_NAME", "polymarket"),
        user=os.environ.get("EXECUTION_SNAPSHOT_DRAIN_DB_USER", "postgres"),
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


def select_columns(has_expanded_vwap: bool) -> str:
    columns = BASE_COLUMNS + (EXPANDED_VWAP_COLUMNS if has_expanded_vwap else ())
    return ",".join(columns)


def drain_hour(db: psycopg.Connection[Any], root: Path, start: datetime) -> dict[str, Any]:
    destination = root / f"date={start:%Y-%m-%d}" / f"hour={start:%H}.parquet"
    if prior := existing_partition(destination):
        return prior
    end = start + timedelta(hours=1)
    records: dict[tuple[Any, Any], dict[str, Any]] = {}
    shared_facts: dict[tuple[Any, Any], tuple[Any, ...]] = {}
    expanded_facts: dict[tuple[Any, Any], tuple[Any, ...]] = {}
    preferences: dict[tuple[Any, Any], int] = {}
    source_counts: dict[str, int] = {}
    duplicate_count = 0
    exact_duplicate_count = 0
    partial_duplicate_count = 0
    conflicts: list[dict[str, Any]] = []
    with db.cursor() as cursor:
        for source in SOURCES:
            table = source["table"]
            source_counts[table] = 0
            cursor.execute(
                f"SELECT {select_columns(source['has_expanded_vwap'])} FROM {table} "
                "WHERE sampled_at >= %s AND sampled_at < %s "
                "ORDER BY sampled_at,market_id,artifact_id",
                (start, end),
            )
            while batch := cursor.fetchmany(2_000):
                for raw in batch:
                    source_counts[table] += 1
                    record = normalize(raw, source["has_expanded_vwap"])
                    key = identity(record)
                    shared = shared_fact(record)
                    expanded = expanded_fact(record)
                    if key not in records:
                        records[key] = record
                        shared_facts[key] = shared
                        expanded_facts[key] = expanded
                        preferences[key] = source["preference"]
                        continue
                    exact_shared_match = shared_facts[key] == shared
                    if not exact_shared_match and not compatible_measurements(records[key], record):
                        conflicts.append({
                            "market_id": key[0], "sampled_at": key[1].isoformat(),
                            "source_table": table, "kind": "shared_fact_mismatch",
                        })
                        continue
                    current_expanded = expanded_facts[key]
                    if source["has_expanded_vwap"] and current_expanded != expanded:
                        conflicts.append({
                            "market_id": key[0], "sampled_at": key[1].isoformat(),
                            "source_table": table, "kind": "expanded_fact_mismatch",
                        })
                        continue
                    duplicate_count += 1
                    if exact_shared_match:
                        exact_duplicate_count += 1
                    else:
                        partial_duplicate_count += 1
                    if source["preference"] < preferences[key]:
                        records[key] = record
                        expanded_facts[key] = expanded
                        preferences[key] = source["preference"]
    if conflicts:
        conflict_path = root / "conflicts" / f"date={start:%Y-%m-%d}-hour={start:%H}.json"
        conflict_path.parent.mkdir(parents=True, exist_ok=True)
        conflict_path.write_text(json.dumps(conflicts, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"{len(conflicts)} conflicting identities in {start.isoformat()}")
    ordered = sorted(records.values(), key=lambda row: (row["sampled_at"], row["market_id"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(ordered, schema=schema())
    temporary = destination.with_suffix(".parquet.partial")
    pq.write_table(table, temporary, compression="zstd", row_group_size=25_000)
    temporary.replace(destination)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "window_start": start.isoformat(), "window_end": end.isoformat(),
        "source_counts": source_counts, "source_total": sum(source_counts.values()),
        "output_count": table.num_rows, "duplicate_count": duplicate_count,
        "exact_duplicate_count": exact_duplicate_count,
        "partial_duplicate_count": partial_duplicate_count,
        "conflict_count": 0, "file": str(destination),
        "file_size_bytes": destination.stat().st_size, "file_sha256": sha256(destination),
    }
    if manifest["source_total"] != manifest["output_count"] + duplicate_count:
        raise RuntimeError(f"row accounting failed for {start.isoformat()}")
    manifest_path = destination.with_suffix(".manifest.json")
    partial = manifest_path.with_suffix(".json.partial")
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    partial.replace(manifest_path)
    return manifest


def source_watermarks(db: psycopg.Connection[Any]) -> dict[str, dict[str, Any]]:
    watermarks = {}
    with db.cursor() as cursor:
        for source in SOURCES:
            table = source["table"]
            cursor.execute(
                f"SELECT count(*)::bigint AS row_count, min(sampled_at) AS minimum_sampled_at, "
                f"max(sampled_at) AS maximum_sampled_at, array_agg(DISTINCT schema_version ORDER BY schema_version) AS schema_versions FROM {table}"
            )
            row = cursor.fetchone()
            watermarks[table] = {
                "row_count": row["row_count"],
                "minimum_sampled_at": row["minimum_sampled_at"].isoformat(),
                "maximum_sampled_at": row["maximum_sampled_at"].isoformat(),
                "schema_versions": row["schema_versions"],
            }
    return watermarks


def refresh_manifest(db: psycopg.Connection[Any], root: Path) -> None:
    parts = [json.loads(path.read_text()) for path in sorted(root.glob("date=*/hour=*.manifest.json"))]
    if not parts:
        raise RuntimeError("no execution-snapshot partitions")
    source_names = sorted({name for part in parts for name in part["source_counts"]})
    result = {
        "contract_version": CONTRACT_VERSION,
        "window_start": min(part["window_start"] for part in parts),
        "window_end": max(part["window_end"] for part in parts),
        "partition_count": len(parts),
        "source_counts": {name: sum(part["source_counts"].get(name, 0) for part in parts) for name in source_names},
        "source_total": sum(part["source_total"] for part in parts),
        "output_count": sum(part["output_count"] for part in parts),
        "duplicate_count": sum(part["duplicate_count"] for part in parts),
        "exact_duplicate_count": sum(part.get("exact_duplicate_count", part["duplicate_count"]) for part in parts),
        "partial_duplicate_count": sum(part.get("partial_duplicate_count", 0) for part in parts),
        "conflict_count": sum(part["conflict_count"] for part in parts),
        "file_size_bytes": sum(part["file_size_bytes"] for part in parts),
        "source_watermarks": source_watermarks(db),
    }
    temporary = root / "manifest.json.partial"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / "manifest.json")


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed).astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy and deduplicate Polymarket execution snapshots to canonical Parquet")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", required=True, type=parse_timestamp)
    parser.add_argument("--end", required=True, type=parse_timestamp)
    args = parser.parse_args()
    start = args.start.replace(minute=0, second=0, microsecond=0)
    end = (args.end - timedelta(microseconds=1)).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    args.output.mkdir(parents=True, exist_ok=True)
    with connection() as db:
        current = start
        while current < end:
            manifest = drain_hour(db, args.output, current)
            print(json.dumps({"completed": current.isoformat(), "rows": manifest["output_count"]}), flush=True)
            current += timedelta(hours=1)
        refresh_manifest(db, args.output)


if __name__ == "__main__":
    main()
