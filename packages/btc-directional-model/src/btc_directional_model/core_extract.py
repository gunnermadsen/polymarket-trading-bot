from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

from .core_config import CoreTrainingConfig, evaluation_holdout_range

CoreScope = Literal["pre_holdout", "holdout"]
CORE_SOURCE_SCHEMA_VERSION = "btc-core-source-v1"
IMMUTABLE_SOURCE_SNAPSHOT_KEY = "immutable_source_snapshot"
RESIDUAL_ADMISSION_SOURCE_START = datetime(2026, 3, 21, tzinfo=UTC)
RESIDUAL_ADMISSION_SOURCE_END = datetime(2026, 7, 21, tzinfo=UTC)
_COPY_FALLBACK_ERRNOS = frozenset(
    value
    for value in (
        errno.EXDEV,
        errno.EACCES,
        errno.EPERM,
        errno.EMLINK,
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
)
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


def snapshot_residual_admission_source(
    config: CoreTrainingConfig,
    source_dir: Path,
) -> dict[str, Any]:
    """Create the exact storage-light source snapshot for residual admission.

    The destination is taken from ``config.paths.source_data`` and must not
    already exist. Every selected partition and the source manifest are
    checksum verified before a new immutable manifest is written. Partitions
    are hard linked when possible and copied with exclusive creation only for
    filesystem errors where hard links are unavailable.
    """

    range_start, range_end = scope_range(config, "pre_holdout")
    if (
        range_start != RESIDUAL_ADMISSION_SOURCE_START
        or range_end != RESIDUAL_ADMISSION_SOURCE_END
        or config.data.range_end != RESIDUAL_ADMISSION_SOURCE_END
    ):
        raise ValueError(
            "residual-admission source snapshot requires exact "
            "[2026-03-21, 2026-07-21)"
        )
    if config.data.strict_final_price_audit:
        raise ValueError(
            "residual-admission source snapshot requires non-strict final-price audit"
        )

    source = source_dir.resolve(strict=True)
    destination = config.paths.source_data.resolve()
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("source snapshot paths must be isolated")
    if destination.exists():
        raise FileExistsError(
            f"source snapshot destination already exists: {destination}"
        )

    source_manifest_path = source / "manifest-pre_holdout.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"source snapshot manifest is missing: {source_manifest_path}"
        )
    source_manifest_sha256 = file_sha256(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    _validate_snapshot_source_contract(
        config,
        source_manifest,
        range_start=range_start,
        range_end=range_end,
    )
    records_by_path = _partition_records_by_path(source_manifest)
    selected_records = [
        _validated_snapshot_partition_record(
            source / partition_name,
            records_by_path.get(partition_name),
        )
        for partition_name in _daily_partition_names(range_start, range_end)
    ]
    if file_sha256(source_manifest_path) != source_manifest_sha256:
        raise RuntimeError("source snapshot manifest changed during validation")

    destination.mkdir(parents=True, exist_ok=False)
    hard_linked = 0
    copied = 0
    for record in selected_records:
        source_partition = source / record["path"]
        destination_partition = destination / record["path"]
        transfer = _snapshot_partition_exclusive(
            source_partition,
            destination_partition,
        )
        if file_sha256(destination_partition) != record["sha256"]:
            destination_partition.unlink(missing_ok=True)
            raise RuntimeError(
                f"source snapshot checksum mismatch: {record['path']}"
            )
        if transfer == "hard_link":
            hard_linked += 1
        else:
            copied += 1

    if file_sha256(source_manifest_path) != source_manifest_sha256:
        raise RuntimeError("source snapshot manifest changed during transfer")

    query = (config.package_root / "sql" / "btc-core-source.sql").read_text()
    manifest: dict[str, Any] = {
        "source_contract": config.data.source_contract,
        "source_schema_version": CORE_SOURCE_SCHEMA_VERSION,
        "source_schema_sha256": _core_source_schema_sha256(),
        "scope": "pre_holdout",
        "range_start": range_start.isoformat(),
        "range_end": range_end.isoformat(),
        "strict_final_price_audit": config.data.strict_final_price_audit,
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "partitions": selected_records,
        "totals": aggregate_partition_summaries(selected_records),
        IMMUTABLE_SOURCE_SNAPSHOT_KEY: {
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": source_manifest_sha256,
            "source_range_start": source_manifest["range_start"],
            "source_range_end": source_manifest["range_end"],
            "snapshot_range_start": range_start.isoformat(),
            "snapshot_range_end": range_end.isoformat(),
            "partition_count": len(selected_records),
            "hard_linked_partitions": hard_linked,
            "copied_partitions": copied,
            "partition_checksums_verified": True,
            "no_overwrite": True,
        },
    }
    write_json_exclusive(destination / "manifest-pre_holdout.json", manifest)
    return manifest


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
        "source_schema_sha256": _core_source_schema_sha256(),
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
    if (
        existing_manifest is not None
        and IMMUTABLE_SOURCE_SNAPSHOT_KEY in existing_manifest
    ):
        if (
            existing_manifest.get("partitions") != manifest["partitions"]
            or existing_manifest.get("totals") != manifest["totals"]
        ):
            raise RuntimeError("immutable source snapshot manifest changed")
        return existing_manifest
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


def _core_source_schema_sha256() -> str:
    return hashlib.sha256(CORE_SOURCE_SCHEMA.to_string().encode()).hexdigest()


def _daily_partition_names(
    range_start: datetime,
    range_end: datetime,
) -> tuple[str, ...]:
    names: list[str] = []
    cursor = range_start
    while cursor < range_end:
        names.append(f"{cursor.date().isoformat()}.parquet")
        cursor += timedelta(days=1)
    return tuple(names)


def _partition_records_by_path(
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    records = manifest.get("partitions")
    if not isinstance(records, list):
        raise TypeError("source snapshot manifest has no partition list")
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise TypeError("source snapshot manifest has an invalid partition")
        path = record["path"]
        if path in output:
            raise RuntimeError(
                f"source snapshot manifest has duplicate partition: {path}"
            )
        output[path] = record
    return output


def _validate_snapshot_source_contract(
    config: CoreTrainingConfig,
    manifest: dict[str, Any],
    *,
    range_start: datetime,
    range_end: datetime,
) -> None:
    query = (config.package_root / "sql" / "btc-core-source.sql").read_text()
    expected = {
        "source_contract": config.data.source_contract,
        "source_schema_version": CORE_SOURCE_SCHEMA_VERSION,
        "source_schema_sha256": _core_source_schema_sha256(),
        "scope": "pre_holdout",
        "strict_final_price_audit": config.data.strict_final_price_audit,
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
    }
    mismatches = [
        key for key, expected_value in expected.items()
        if manifest.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "source snapshot contract mismatch: " + ", ".join(mismatches)
        )
    try:
        source_start = datetime.fromisoformat(str(manifest["range_start"]))
        source_end = datetime.fromisoformat(str(manifest["range_end"]))
    except (KeyError, ValueError) as error:
        raise RuntimeError("source snapshot range is invalid") from error
    if source_start > range_start or source_end < range_end:
        raise RuntimeError(
            "source snapshot does not cover exact [2026-03-21, 2026-07-21)"
        )


def _validated_snapshot_partition_record(
    path: Path,
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    if record is None:
        raise RuntimeError(f"source snapshot partition is unrecorded: {path.name}")
    if not path.is_file():
        raise FileNotFoundError(f"source snapshot partition is missing: {path}")
    expected_sha256 = record.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise RuntimeError(
            f"source snapshot partition has invalid checksum: {path.name}"
        )
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"source snapshot partition checksum mismatch: {path.name}")
    summary = partition_summary(path)
    mismatches = [
        key for key, value in summary.items() if record.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"source snapshot partition summary mismatch for {path.name}: "
            + ", ".join(mismatches)
        )
    return {
        "path": path.name,
        "sha256": expected_sha256,
        **summary,
    }


def _hard_link_partition(source: Path, destination: Path) -> None:
    os.link(source, destination)


def _snapshot_partition_exclusive(
    source: Path,
    destination: Path,
) -> Literal["hard_link", "copy"]:
    try:
        _hard_link_partition(source, destination)
        return "hard_link"
    except OSError as error:
        if error.errno not in _COPY_FALLBACK_ERRNOS:
            raise
    try:
        with source.open("rb") as source_handle, destination.open("xb") as output:
            shutil.copyfileobj(source_handle, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return "copy"


def load_existing_manifest(
    manifest_path: Path,
    contract: dict[str, Any],
    *,
    force: bool,
) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    existing = json.loads(manifest_path.read_text())
    if force:
        if IMMUTABLE_SOURCE_SNAPSHOT_KEY in existing:
            raise RuntimeError(
                "immutable source snapshot cannot be rebuilt or overwritten"
            )
        return None
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
    if IMMUTABLE_SOURCE_SNAPSHOT_KEY in manifest:
        expected_paths = set(_daily_partition_names(expected_start, expected_end))
        partitions = manifest.get("partitions", [])
        observed_paths = [str(partition.get("path")) for partition in partitions]
        if (
            len(observed_paths) != len(expected_paths)
            or set(observed_paths) != expected_paths
            or manifest.get("totals")
            != aggregate_partition_summaries(partitions)
        ):
            raise RuntimeError(
                "immutable source snapshot does not contain the exact daily range"
            )
    return manifest


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
