from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .config import BenchmarkConfig


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
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


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


def prepare_snapshot(config: BenchmarkConfig, *, refresh: bool = False) -> Snapshot:
    if config.dataset.source == "parquet_lake":
        return _prepare_lake_snapshot(config, refresh=refresh)
    raise ValueError(f"unsupported source {config.dataset.source}")


def _verified_lake_files(config: BenchmarkConfig) -> tuple[list[Path], str]:
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


def _metadata_payloads(config: BenchmarkConfig, dataset: str) -> list[dict[str, Any]]:
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


def _verified_contract_metadata(config: BenchmarkConfig) -> dict[str, Any]:
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
        "base_maker_fee_bps": float(base_tier["makerFee"]) * 100.0,
        "base_taker_fee_bps": taker_fee_bps,
    }


def _lake_dataset_frame(
    config: BenchmarkConfig,
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


def _lake_source_frame(config: BenchmarkConfig) -> pl.DataFrame:
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


def _prepare_lake_snapshot(config: BenchmarkConfig, *, refresh: bool = False) -> Snapshot:
    dataset_root = config.artifacts.root / "datasets"
    files, lake_merkle_sha256 = _verified_lake_files(config)
    contract_metadata = _verified_contract_metadata(config)
    identity = hashlib.sha256(
        f"{config.fingerprint}:{lake_merkle_sha256}:lake-v1".encode()
    ).hexdigest()
    index_path = dataset_root / f"index-{identity}.json"
    if not refresh:
        existing = _snapshot_from_manifest(index_path)
        if existing is not None:
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
        "schema_version": 1,
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
            "non_null_rows": funding_rows.height,
            "first_timestamp": (
                funding_rows["bucket_start"].min() if funding_rows.height else None
            ),
            "last_timestamp": (funding_rows["bucket_start"].max() if funding_rows.height else None),
            "missing_rate_policy": ("zero realized funding cost; funding is never a model feature"),
        },
        "config_fingerprint": config.fingerprint,
        "normalized_database_exclusion": {
            "reason": "verified Timescale child-index wrong-row lookup and 600-row slippage loss",
            "policy": "do not use normalized analytics rows for model training",
        },
        "parquet_sha256": sha256,
        "parquet_path": str(final_path),
    }
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


def validate_snapshot_frame(frame: pl.DataFrame, config: BenchmarkConfig) -> None:
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
