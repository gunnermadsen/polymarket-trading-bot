from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .config import BenchmarkConfig, ExpectancyConfig
from .reporting import json_safe


@dataclass(frozen=True)
class Snapshot:
    path: Path
    sha256: str
    row_count: int
    first_timestamp: datetime
    last_timestamp: datetime
    manifest_path: Path


LAKE_DATASETS = (
    "trade_candles",
    "mark_candles",
    "spot_candles",
    "future_basis",
    "open_interest",
    "aggressor_differential",
    "trade_volume",
    "trade_count",
    "cvd",
    "liquidation_volume",
    "spreads",
    "liquidity",
    "slippage",
    "funding_rates",
    "instruments",
    "fee_schedules",
)
FUNDING_PROVENANCE_DATASET = "funding_rates"
FUNDING_IMPORT_POLICY = "continuous-hourly-rate-locf-15m-v1"
FUNDING_SOURCE_KINDS = frozenset(
    {
        "official_csv_archive",
        "official_historical_funding_rates_api",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(_json_text(payload), encoding="utf-8")
    os.replace(temporary, path)


def _json_text(payload: dict[str, Any]) -> str:
    return (
        json.dumps(
            json_safe(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _immutable_json(path: Path, payload: dict[str, Any]) -> None:
    """Create a manifest once and reject any later content change."""
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = _json_text(payload)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError as error:
        if path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"immutable manifest content changed: {path}") from error
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(expected)
        handle.flush()
        os.fsync(handle.fileno())


def _snapshot_from_manifest(index_path: Path) -> Snapshot | None:
    if not index_path.exists():
        return None
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    path = Path(payload["path"])
    manifest_path = Path(payload["manifest_path"])
    if not path.exists() or not manifest_path.exists():
        return None
    sha256 = _sha256(path)
    if sha256 != payload["sha256"]:
        raise RuntimeError(f"snapshot checksum mismatch: {path}")
    return Snapshot(
        path=path,
        sha256=sha256,
        row_count=int(payload["row_count"]),
        first_timestamp=datetime.fromisoformat(payload["first_timestamp"]),
        last_timestamp=datetime.fromisoformat(payload["last_timestamp"]),
        manifest_path=manifest_path,
    )


def prepare_snapshot(
    config: BenchmarkConfig | ExpectancyConfig, *, refresh: bool = False
) -> Snapshot:
    if config.dataset.source == "parquet_lake":
        return _prepare_lake_snapshot(config, refresh=refresh)
    raise ValueError(f"unsupported source {config.dataset.source}")


def _verified_lake_files(
    config: BenchmarkConfig | ExpectancyConfig,
) -> tuple[list[Path], str]:
    files: list[Path] = []
    digest = hashlib.sha256()
    for dataset in LAKE_DATASETS:
        dataset_root = config.dataset.lake_root / f"dataset={dataset}"
        candidates = sorted(dataset_root.rglob("*.parquet"))
        if not candidates:
            raise RuntimeError(f"lake contains no Parquet objects for {dataset}")
        for path in candidates:
            expected = path.stem.rsplit("_", 1)[-1]
            actual = _sha256(path)
            if len(expected) != 64 or actual != expected:
                raise RuntimeError(f"lake object checksum mismatch: {path}")
            relative = path.relative_to(config.dataset.lake_root)
            digest.update(str(relative).encode())
            digest.update(actual.encode())
        files.extend(candidates)
    return files, digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verified_relative_file(root: Path, relative_value: Any, *, label: str) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise RuntimeError(f"{label} has no relative path")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{label} path escapes its immutable root: {relative}")
    root_resolved = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as error:
        raise RuntimeError(f"{label} path escapes its immutable root: {relative}") from error
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    return path


def _manifest_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} is not an ISO-8601 timestamp")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"{label} is not an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise RuntimeError(f"{label} is not timezone-aware")
    return timestamp.astimezone(UTC)


def _funding_manifest_matches(
    payload: dict[str, Any],
    config: BenchmarkConfig | ExpectancyConfig,
) -> bool:
    configured = payload.get("configured_range")
    if not isinstance(configured, dict):
        return False
    try:
        start = _manifest_timestamp(
            configured.get("start"), label="funding manifest configured start"
        )
        end = _manifest_timestamp(
            configured.get("end_exclusive"),
            label="funding manifest configured end",
        )
    except RuntimeError:
        return False
    return (
        payload.get("provider") == "kraken_futures"
        and payload.get("dataset") == FUNDING_PROVENANCE_DATASET
        and (
            not isinstance(config, ExpectancyConfig)
            or payload.get("import_id") == config.funding_provenance.import_id
        )
        and payload.get("symbol") == config.dataset.symbol
        and payload.get("interval_seconds") == config.dataset.interval_seconds
        and start == config.dataset.start.astimezone(UTC)
        and end == config.dataset.end.astimezone(UTC)
    )


def _verified_funding_provenance(
    config: BenchmarkConfig | ExpectancyConfig,
) -> dict[str, Any]:
    """Verify and fingerprint the first-party funding import evidence chain."""
    provenance_root = (
        config.dataset.lake_root / "_provenance" / FUNDING_PROVENANCE_DATASET
    )
    manifest_root = provenance_root / "manifests"
    manifest_paths = (
        [
            manifest_root
            / f"{config.funding_provenance.import_id}.json"
        ]
        if isinstance(config, ExpectancyConfig)
        else sorted(manifest_root.glob("*.json"))
    )
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in manifest_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"funding provenance manifest is unreadable: {path}") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"funding provenance manifest is not an object: {path}")
        if _funding_manifest_matches(payload, config):
            matches.append((path, payload))
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one first-party funding provenance manifest for the "
            f"pinned import id and configured symbol/range; found {len(matches)}"
        )

    manifest_path, manifest = matches[0]
    import_id = manifest.get("import_id")
    if (
        not isinstance(import_id, str)
        or len(import_id) != 64
        or any(character not in "0123456789abcdef" for character in import_id)
        or manifest_path.stem != import_id
    ):
        raise RuntimeError("funding provenance import identity is invalid")
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported funding provenance manifest schema")

    configured = manifest["configured_range"]
    expected_rows = int(
        (config.dataset.end - config.dataset.start).total_seconds()
        // config.dataset.interval_seconds
    )
    expected_last = config.dataset.end - timedelta(
        seconds=config.dataset.interval_seconds
    )
    if (
        int(configured.get("expected_rows", -1)) != expected_rows
        or _manifest_timestamp(
            configured.get("first_timestamp"),
            label="funding manifest first timestamp",
        )
        != config.dataset.start.astimezone(UTC)
        or _manifest_timestamp(
            configured.get("last_timestamp"),
            label="funding manifest last timestamp",
        )
        != expected_last.astimezone(UTC)
    ):
        raise RuntimeError("funding provenance configured coverage does not match training")

    source_entries = manifest.get("sources")
    if not isinstance(source_entries, list) or len(source_entries) != len(
        FUNDING_SOURCE_KINDS
    ):
        raise RuntimeError("funding provenance does not bind both first-party sources")
    source_kinds = {entry.get("kind") for entry in source_entries if isinstance(entry, dict)}
    if source_kinds != FUNDING_SOURCE_KINDS:
        raise RuntimeError("funding provenance source kinds are incomplete or duplicated")

    verified_sources: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for entry in sorted(source_entries, key=lambda item: str(item.get("kind"))):
        if not isinstance(entry, dict):
            raise RuntimeError("funding provenance source entry is not an object")
        kind = str(entry["kind"])
        sha256 = entry.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise RuntimeError(f"funding provenance {kind} has an invalid SHA-256")
        source_path = _verified_relative_file(
            provenance_root,
            entry.get("immutable_relative_path"),
            label=f"funding provenance {kind}",
        )
        if _sha256(source_path) != sha256:
            raise RuntimeError(f"funding provenance source checksum mismatch: {source_path}")
        byte_size = int(entry.get("byte_size", -1))
        if byte_size < 0 or source_path.stat().st_size != byte_size:
            raise RuntimeError(f"funding provenance source size mismatch: {source_path}")
        source_hashes[kind] = sha256
        verified_sources.append(
            {
                "kind": kind,
                "origin": entry.get("origin"),
                "relative_path": str(source_path.relative_to(config.dataset.lake_root)),
                "sha256": sha256,
                "byte_size": byte_size,
                "rows": int(entry.get("rows", -1)),
                "first_timestamp": entry.get("first_timestamp"),
                "last_timestamp": entry.get("last_timestamp"),
            }
        )

    normalization = manifest.get("normalization")
    if (
        not isinstance(normalization, dict)
        or normalization.get("policy") != FUNDING_IMPORT_POLICY
    ):
        raise RuntimeError("funding provenance normalization policy changed")
    expected_import_id = _canonical_sha256(
        {
            "policy": FUNDING_IMPORT_POLICY,
            "archive_sha256": source_hashes["official_csv_archive"],
            "recent_sha256": source_hashes[
                "official_historical_funding_rates_api"
            ],
            "provider": "kraken_futures",
            "dataset": FUNDING_PROVENANCE_DATASET,
            "symbol": config.dataset.symbol,
            "interval_seconds": config.dataset.interval_seconds,
            "start": config.dataset.start.isoformat(),
            "end": config.dataset.end.isoformat(),
        }
    )
    if import_id != expected_import_id:
        raise RuntimeError("funding provenance import identity does not match its sources")

    mutation = manifest.get("lake_mutation")
    if not isinstance(mutation, dict):
        raise RuntimeError("funding provenance has no lake mutation evidence")
    published_entries = mutation.get("published_objects")
    if not isinstance(published_entries, list):
        raise RuntimeError("funding provenance published object list is invalid")
    verified_objects: list[dict[str, Any]] = []
    published_rows = 0
    for entry in sorted(published_entries, key=lambda item: str(item.get("relative_path"))):
        if not isinstance(entry, dict):
            raise RuntimeError("funding provenance published object is not an object")
        relative_path = entry.get("relative_path")
        object_path = _verified_relative_file(
            config.dataset.lake_root,
            relative_path,
            label="funding provenance published object",
        )
        expected_prefix = (
            f"dataset={FUNDING_PROVENANCE_DATASET}",
            f"symbol={config.dataset.symbol}",
            f"interval_seconds={config.dataset.interval_seconds}",
        )
        relative = object_path.relative_to(config.dataset.lake_root)
        if relative.parts[:3] != expected_prefix:
            raise RuntimeError(
                f"funding provenance published object has wrong partition: {relative}"
            )
        sha256 = entry.get("sha256")
        if not isinstance(sha256, str) or _sha256(object_path) != sha256:
            raise RuntimeError(
                f"funding provenance published object checksum mismatch: {object_path}"
            )
        row_count = int(entry.get("row_count", -1))
        if row_count < 0:
            raise RuntimeError("funding provenance published object row count is invalid")
        published_rows += row_count
        verified_objects.append(
            {
                "relative_path": str(relative),
                "sha256": sha256,
                "row_count": row_count,
                "first_timestamp": entry.get("first_timestamp"),
                "last_timestamp": entry.get("last_timestamp"),
            }
        )

    preexisting_rows = int(mutation.get("preexisting_rows", -1))
    if (
        int(mutation.get("published_rows", -1)) != published_rows
        or preexisting_rows + published_rows != expected_rows
    ):
        raise RuntimeError("funding provenance lake mutation row totals do not reconcile")
    coverage = manifest.get("coverage_validation")
    if (
        not isinstance(coverage, dict)
        or coverage.get("complete") is not True
        or int(coverage.get("validated_rows", -1)) != expected_rows
        or int(coverage.get("missing_rows", -1)) != 0
        or int(coverage.get("conflicting_rows", -1)) != 0
    ):
        raise RuntimeError("funding provenance coverage validation is incomplete")

    manifest_relative = manifest_path.relative_to(config.dataset.lake_root)
    binding = {
        "schema_version": 1,
        "import_id": import_id,
        "normalization_policy": FUNDING_IMPORT_POLICY,
        "manifest": {
            "relative_path": str(manifest_relative),
            "sha256": _sha256(manifest_path),
        },
        "sources": verified_sources,
        "published_objects": verified_objects,
        "coverage_validation": {
            "complete": True,
            "validated_rows": expected_rows,
            "missing_rows": 0,
            "conflicting_rows": 0,
        },
        "existing_lake_reconciliation": manifest.get(
            "existing_lake_reconciliation"
        ),
    }
    return {**binding, "binding_sha256": _canonical_sha256(binding)}


def validate_funding_provenance_binding(
    raw_manifest: dict[str, Any],
    config: BenchmarkConfig | ExpectancyConfig,
) -> dict[str, Any]:
    """Fail closed unless a raw snapshot binds the currently verified evidence."""
    recorded = raw_manifest.get("funding_provenance")
    if not isinstance(recorded, dict):
        raise RuntimeError(
            "raw snapshot manifest has no first-party funding provenance binding"
        )
    recorded_binding = recorded.get("binding_sha256")
    recorded_body = {
        key: value for key, value in recorded.items() if key != "binding_sha256"
    }
    if (
        not isinstance(recorded_binding, str)
        or recorded_binding != _canonical_sha256(recorded_body)
    ):
        raise RuntimeError("raw snapshot funding provenance binding checksum is invalid")
    verified = _verified_funding_provenance(config)
    if recorded != verified:
        raise RuntimeError(
            "raw snapshot funding provenance binding does not match the "
            "currently verified first-party evidence"
        )
    return verified


def _metadata_payloads(
    config: BenchmarkConfig | ExpectancyConfig, dataset: str
) -> list[dict[str, Any]]:
    root = config.dataset.lake_root / f"dataset={dataset}"
    paths = sorted(root.rglob("*.parquet"))
    frame = (
        pl.read_parquet(
            [str(path) for path in paths],
            columns=[
                "observed_at_ms",
                "dataset",
                "symbol",
                "payload_json",
            ],
        )
        .filter((pl.col("dataset") == dataset) & (pl.col("symbol") == config.dataset.symbol))
        .sort("observed_at_ms")
    )
    if frame.is_empty():
        raise RuntimeError(f"lake contains no {dataset} metadata for the symbol")
    return [json.loads(payload) for payload in frame["payload_json"].to_list()]


def _verified_contract_metadata(
    config: BenchmarkConfig | ExpectancyConfig,
) -> dict[str, Any]:
    instruments = _metadata_payloads(config, "instruments")
    instrument = instruments[-1]
    if instrument.get("symbol") != config.dataset.symbol:
        raise RuntimeError("instrument metadata symbol does not match configuration")
    fee_uid = instrument.get("feeScheduleUid")
    schedules = _metadata_payloads(config, "fee_schedules")
    matching_schedules = [schedule for schedule in schedules if schedule.get("uid") == fee_uid]
    if not matching_schedules:
        raise RuntimeError(f"no fee schedule matched instrument UID {fee_uid}")
    schedule = matching_schedules[-1]
    base_tiers = [
        tier for tier in schedule.get("tiers", []) if float(tier.get("usdVolume", -1)) == 0.0
    ]
    if len(base_tiers) != 1:
        raise RuntimeError("fee schedule does not have one zero-volume tier")
    base_tier = base_tiers[0]
    taker_fee_bps = float(base_tier["takerFee"]) * 100.0
    if abs(taker_fee_bps - config.dataset.taker_fee_bps_per_side) > 1e-9:
        raise RuntimeError(
            "configured taker fee does not match Kraken contract metadata: "
            f"{config.dataset.taker_fee_bps_per_side} != {taker_fee_bps}"
        )
    maker_fee_bps = float(base_tier["makerFee"]) * 100.0
    if isinstance(config, ExpectancyConfig):
        if abs(taker_fee_bps - config.fees.taker_bps_per_side) > 1e-9:
            raise RuntimeError("expectancy taker fee does not match Kraken contract metadata")
        if abs(maker_fee_bps - config.fees.maker_bps_per_side) > 1e-9:
            raise RuntimeError(
                "configured maker fee does not match Kraken contract metadata: "
                f"{config.fees.maker_bps_per_side} != {maker_fee_bps}"
            )
    return {
        "symbol": instrument["symbol"],
        "type": instrument.get("type"),
        "base": instrument.get("base"),
        "quote": instrument.get("quote"),
        "contract_size": instrument.get("contractSize"),
        "tick_size": instrument.get("tickSize"),
        "opening_date": instrument.get("openingDate"),
        "funding_rate_coefficient": instrument.get("fundingRateCoefficient"),
        "maximum_relative_funding_rate": instrument.get("maxRelativeFundingRate"),
        "fee_schedule_uid": fee_uid,
        "fee_schedule_name": schedule.get("name"),
        "base_maker_fee_bps": maker_fee_bps,
        "base_taker_fee_bps": taker_fee_bps,
    }


def _lake_dataset_frame(
    config: BenchmarkConfig | ExpectancyConfig,
    dataset: str,
    fields: dict[str, str],
) -> pl.DataFrame:
    root = config.dataset.lake_root / f"dataset={dataset}"
    paths = sorted(root.rglob("*.parquet"))
    frame = pl.read_parquet(
        [str(path) for path in paths],
        columns=[
            "observed_at_ms",
            "dataset",
            "symbol",
            "interval_seconds",
            "payload_json",
        ],
    ).filter(
        (pl.col("dataset") == dataset)
        & (pl.col("symbol") == config.dataset.symbol)
        & (pl.col("interval_seconds") == config.dataset.interval_seconds)
    )
    frame = frame.with_columns(
        pl.from_epoch("observed_at_ms", time_unit="ms")
        .dt.replace_time_zone("UTC")
        .alias("bucket_start")
    ).filter(
        (pl.col("bucket_start") >= config.dataset.start)
        & (pl.col("bucket_start") < config.dataset.end)
    )
    conflicting = (
        frame.group_by("bucket_start")
        .agg(pl.col("payload_json").n_unique().alias("payloads"))
        .filter(pl.col("payloads") > 1)
    )
    if conflicting.height:
        raise RuntimeError(
            f"{dataset} has {conflicting.height} timestamps with conflicting payloads"
        )
    expressions = [
        pl.col("payload_json")
        .str.json_path_match(path)
        .cast(pl.Float64, strict=False)
        .alias(column)
        for column, path in fields.items()
    ]
    return (
        frame.unique(subset=["bucket_start"], keep="last")
        .select(["bucket_start", *expressions])
        .sort("bucket_start")
    )


def _lake_source_frame(config: BenchmarkConfig | ExpectancyConfig) -> pl.DataFrame:
    definitions: dict[str, dict[str, str]] = {
        "trade_candles": {
            "trade_open": "$.open",
            "trade_high": "$.high",
            "trade_low": "$.low",
            "trade_close": "$.close",
            "trade_candle_volume": "$.volume",
        },
        "mark_candles": {
            "mark_open": "$.open",
            "mark_high": "$.high",
            "mark_low": "$.low",
            "mark_close": "$.close",
        },
        "spot_candles": {"spot_close": "$.close"},
        "future_basis": {"future_basis": "$.basis"},
        "open_interest": {
            "oi_open": "$[0]",
            "oi_high": "$[1]",
            "oi_low": "$[2]",
            "oi_close": "$[3]",
        },
        "aggressor_differential": {"aggressor_differential": "$"},
        "trade_volume": {"trade_volume": "$"},
        "trade_count": {"trade_count": "$"},
        "cvd": {
            "cvd": "$.cvd",
            "buy_volume": "$.buy_volume",
            "sell_volume": "$.sell_volume",
        },
        "liquidation_volume": {"liquidation_volume": "$"},
        "spreads": {
            "ask_best": "$.ask.best_price",
            "bid_best": "$.bid.best_price",
        },
        "liquidity": {
            "ask_liquidity_005": "$.ask.liquidity_005",
            "bid_liquidity_005": "$.bid.liquidity_005",
            "ask_liquidity_01": "$.ask.liquidity_01",
            "bid_liquidity_01": "$.bid.liquidity_01",
            "ask_liquidity_025": "$.ask.liquidity_025",
            "bid_liquidity_025": "$.bid.liquidity_025",
            "ask_liquidity_05": "$.ask.liquidity_05",
            "bid_liquidity_05": "$.bid.liquidity_05",
            "ask_liquidity_10": "$.ask.liquidity_10",
            "bid_liquidity_10": "$.bid.liquidity_10",
        },
        "slippage": {
            "ask_slippage_1k": "$.ask.slippage_1k",
            "bid_slippage_1k": "$.bid.slippage_1k",
            "ask_slippage_10k": "$.ask.slippage_10k",
            "bid_slippage_10k": "$.bid.slippage_10k",
            "ask_slippage_100k": "$.ask.slippage_100k",
            "bid_slippage_100k": "$.bid.slippage_100k",
        },
        "funding_rates": {
            "funding_rate": "$.funding_rate",
            "relative_funding_rate": "$.relative_funding_rate",
        },
    }
    frames = {
        dataset: _lake_dataset_frame(config, dataset, fields)
        for dataset, fields in definitions.items()
    }
    result = frames.pop("trade_candles")
    for dataset in (
        "mark_candles",
        "spot_candles",
        "future_basis",
        "open_interest",
        "aggressor_differential",
        "trade_volume",
        "trade_count",
        "cvd",
        "liquidation_volume",
        "spreads",
        "liquidity",
        "slippage",
        "funding_rates",
    ):
        result = result.join(frames[dataset], on="bucket_start", how="left")
    return result.sort("bucket_start")


def _prepare_lake_snapshot(
    config: BenchmarkConfig | ExpectancyConfig, *, refresh: bool = False
) -> Snapshot:
    dataset_root = config.artifacts.root / "datasets"
    files, lake_merkle_sha256 = _verified_lake_files(config)
    contract_metadata = _verified_contract_metadata(config)
    funding_provenance = (
        _verified_funding_provenance(config)
        if isinstance(config, ExpectancyConfig)
        else None
    )
    identity_version = "lake-v2" if funding_provenance is not None else "lake-v1"
    funding_binding_sha256 = (
        funding_provenance["binding_sha256"]
        if funding_provenance is not None
        else "not-required"
    )
    identity = hashlib.sha256(
        (
            f"{config.fingerprint}:{lake_merkle_sha256}:"
            f"{funding_binding_sha256}:{identity_version}"
        ).encode()
    ).hexdigest()
    index_path = dataset_root / f"index-{identity}.json"
    if not refresh:
        existing = _snapshot_from_manifest(index_path)
        if existing is not None:
            existing_manifest = json.loads(
                existing.manifest_path.read_text(encoding="utf-8")
            )
            if (
                funding_provenance is not None
                and existing_manifest.get("funding_provenance")
                != funding_provenance
            ):
                raise RuntimeError(
                    "raw snapshot funding provenance binding no longer matches "
                    "the verified first-party evidence"
                )
            validate_snapshot_frame(pl.read_parquet(existing.path), config)
            return existing

    frame = _lake_source_frame(config)
    if not frame.height:
        raise RuntimeError("Kraken lake extraction returned no rows")
    validate_snapshot_frame(frame, config)

    staging = dataset_root / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / f"raw-{config.fingerprint}-{os.getpid()}.parquet"
    frame.write_parquet(
        temporary,
        compression="zstd",
        compression_level=9,
        statistics=True,
    )
    sha256 = _sha256(temporary)
    final_path = dataset_root / "raw" / f"{sha256}.parquet"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        if _sha256(final_path) != sha256:
            raise RuntimeError(f"existing snapshot hash mismatch: {final_path}")
        temporary.unlink()
    else:
        os.replace(temporary, final_path)

    first = frame.item(0, "bucket_start")
    last = frame.item(frame.height - 1, "bucket_start")
    funding_rows = frame.filter(pl.col("relative_funding_rate").is_not_null())
    manifest_path = dataset_root / "manifests" / f"{sha256}-{identity[:16]}.json"
    manifest = {
        "schema_version": 2 if funding_provenance is not None else 1,
        "provider": "kraken_futures",
        "canonical_source": "content_addressed_parquet_lake",
        "lake_root": str(config.dataset.lake_root),
        "lake_object_count": len(files),
        "lake_merkle_sha256": lake_merkle_sha256,
        "symbol": config.dataset.symbol,
        "interval_seconds": config.dataset.interval_seconds,
        "range_start": config.dataset.start.isoformat(),
        "range_end": config.dataset.end.isoformat(),
        "row_count": frame.height,
        "first_timestamp": first.isoformat(),
        "last_timestamp": last.isoformat(),
        "columns": frame.columns,
        "contract_metadata": contract_metadata,
        "null_counts": frame.null_count().row(0, named=True),
        "funding_coverage": {
            "complete": funding_rows.height == frame.height,
            "non_null_rows": funding_rows.height,
            "missing_rows": frame.height - funding_rows.height,
            "first_timestamp": (
                funding_rows["bucket_start"].min() if funding_rows.height else None
            ),
            "last_timestamp": (funding_rows["bucket_start"].max() if funding_rows.height else None),
            "missing_rate_policy": (
                "forbidden; expectancy labels require complete first-party funding"
                if isinstance(config, ExpectancyConfig)
                else "zero realized funding cost; funding is never a model feature"
            ),
        },
        "config_fingerprint": config.fingerprint,
        "normalized_database_exclusion": {
            "reason": "verified Timescale child-index wrong-row lookup and 600-row slippage loss",
            "policy": "do not use normalized analytics rows for model training",
        },
        "parquet_sha256": sha256,
        "parquet_path": str(final_path),
    }
    if funding_provenance is not None:
        manifest["funding_provenance"] = funding_provenance
    _immutable_json(manifest_path, manifest)
    index = {
        "path": str(final_path),
        "manifest_path": str(manifest_path),
        "sha256": sha256,
        "row_count": frame.height,
        "first_timestamp": first.isoformat(),
        "last_timestamp": last.isoformat(),
    }
    _atomic_json(index_path, index)
    return Snapshot(
        path=final_path,
        sha256=sha256,
        row_count=frame.height,
        first_timestamp=first,
        last_timestamp=last,
        manifest_path=manifest_path,
    )


def validate_snapshot_frame(
    frame: pl.DataFrame, config: BenchmarkConfig | ExpectancyConfig
) -> None:
    expected_last = config.dataset.end - timedelta(seconds=config.dataset.interval_seconds)
    expected_rows = int(
        (config.dataset.end - config.dataset.start).total_seconds()
        // config.dataset.interval_seconds
    )
    if frame.height != expected_rows:
        raise RuntimeError(
            f"snapshot row count {frame.height} does not match complete configured "
            f"range count {expected_rows}"
        )
    if frame["bucket_start"].min() != config.dataset.start:
        raise RuntimeError("snapshot does not begin at the configured dataset start")
    if frame["bucket_start"].max() != expected_last:
        raise RuntimeError("snapshot does not end at the final configured interval")
    if frame["bucket_start"].n_unique() != frame.height:
        raise RuntimeError("snapshot contains duplicate timestamps")
    deltas = frame.select(pl.col("bucket_start").diff().dt.total_seconds()).drop_nulls()
    unexpected = deltas.filter(pl.col("bucket_start") != config.dataset.interval_seconds).height
    if unexpected:
        raise RuntimeError(f"snapshot contains {unexpected} non-contiguous intervals")
    required = [
        "trade_open",
        "trade_high",
        "trade_low",
        "trade_close",
        "mark_close",
        "spot_close",
        "future_basis",
        "oi_close",
        "trade_volume",
        "trade_count",
        "cvd",
        "ask_best",
        "bid_best",
    ]
    nulls = frame.select(pl.col(required).null_count()).row(0, named=True)
    missing = {column: count for column, count in nulls.items() if count}
    if missing:
        raise RuntimeError(f"required snapshot fields contain nulls: {missing}")
    slippage_columns = [
        "ask_slippage_1k",
        "bid_slippage_1k",
        "ask_slippage_10k",
        "bid_slippage_10k",
    ]
    missing_slippage = {
        column: frame[column].null_count()
        for column in slippage_columns
        if frame[column].null_count()
    }
    if missing_slippage:
        raise RuntimeError(
            f"execution-cost fields require complete two-sided coverage: {missing_slippage}"
        )
    if isinstance(config, ExpectancyConfig):
        missing_funding = frame["relative_funding_rate"].null_count()
        if missing_funding:
            raise RuntimeError(
                "expectancy labels require complete first-party funding coverage: "
                f"{missing_funding} missing buckets"
            )
