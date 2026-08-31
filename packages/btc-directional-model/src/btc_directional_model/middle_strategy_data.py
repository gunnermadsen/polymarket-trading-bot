"""Read-only, checkpointed optional-data attachment for the middle tournament."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.special import ndtr

from .core_extract import file_sha256
from .multivenue_early_entry_data import KEY_COLUMNS, TournamentDataConfig, build_panel
from .spot_l2_chainlink_features import L2_FEATURES, join_qualified_l2
from .twap60_training_data import (
    _isolated_query_frame,
    authentic_labels,
    construct_proxy_labels,
    load_source_group,
)

SCHEMA_VERSION = "btc-middle-strategy-data-v1"
NORMALIZED_SCHEMA_VERSION = "btc-twap-normalized-middle-strategy-data-v1"
BRIDGE_SCHEMA_VERSION = "btc-source-preserving-settlement-bridge-data-v1"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def middle_cache(config: TournamentDataConfig) -> Path:
    return config.package_root / config.raw["paths"]["middle_cache"]


def normalized_cache(config: TournamentDataConfig) -> Path:
    path = config.raw["paths"].get("normalized_cache")
    return config.package_root / path if path else middle_cache(config)


def bridge_cache(config: TournamentDataConfig) -> Path:
    path = config.raw["paths"].get("bridge_cache")
    return config.package_root / path if path else middle_cache(config)


def _normalized_seed(config: TournamentDataConfig, *, force: bool) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Snapshot the existing compact Binance/RefPrice reconstruction audit."""

    cache = normalized_cache(config)
    cache.mkdir(parents=True, exist_ok=True)
    source_root = Path(config.raw["sources"]["binance_twap_seed_cache"])
    source = source_root / "label-audit.parquet"
    source_manifest = source_root / "source-manifest.json"
    if not source.is_file() or not source_manifest.is_file():
        raise RuntimeError("existing normalized-label seed artifacts are unavailable")
    destination = cache / "normalization-seed-label-audit.parquet"
    manifest_path = cache / "normalization-seed-manifest.json"
    contract = {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "columns_used": [
            "market_id", "window_start", "window_end", "official_outcome",
            "proxy_label_up", "proxy_margin_bps", "binance_raw_label_up",
            "binance_raw_margin_bps", "binance_label_complete",
        ],
        "corrected_binance_columns_used": False,
        "read_only": True,
        "database_mutations": False,
    }
    if force or not destination.is_file():
        temporary = destination.with_suffix(".parquet.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    if file_sha256(destination) != contract["source_sha256"]:
        raise RuntimeError("normalized-label seed changed while being snapshotted")
    payload = {
        "contract": contract,
        "local_path": str(destination.relative_to(cache)),
        "rows": pl.scan_parquet(destination).select(pl.len()).collect().item(),
        "sha256": file_sha256(destination),
    }
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text())
        if existing != payload:
            raise RuntimeError("existing normalized-label seed contract changed")
    else:
        _write_json(manifest_path, payload)
    return pl.read_parquet(destination), payload


def _bridge_seed(
    config: TournamentDataConfig, *, force: bool
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Snapshot the existing immutable paired RefPrice/TWAP/Binance audit."""

    cache = bridge_cache(config)
    cache.mkdir(parents=True, exist_ok=True)
    source_root = Path(config.raw["sources"]["binance_twap_seed_cache"])
    source = source_root / "label-audit.parquet"
    source_manifest = source_root / "source-manifest.json"
    if not source.is_file() or not source_manifest.is_file():
        raise RuntimeError("existing settlement-bridge seed artifacts are unavailable")
    destination = cache / "settlement-source-ledger.parquet"
    manifest_path = cache / "settlement-source-ledger-manifest.json"
    contract = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "columns_used": [
            "market_id",
            "window_start",
            "window_end",
            "official_outcome",
            "legacy_open_price",
            "legacy_close_price",
            "proxy_label_up",
            "proxy_margin_bps",
            "binance_raw_label_up",
            "binance_raw_margin_bps",
            "binance_label_complete",
            "authentic_label_up",
            "authentic_margin_bps",
        ],
        "source_values_remain_distinct": True,
        "read_only": True,
        "database_mutations": False,
    }
    if force or not destination.is_file():
        temporary = destination.with_suffix(".parquet.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    if file_sha256(destination) != contract["source_sha256"]:
        raise RuntimeError("settlement source ledger changed while being snapshotted")
    payload = {
        "contract": contract,
        "local_path": str(destination.relative_to(cache)),
        "rows": pl.scan_parquet(destination).select(pl.len()).collect().item(),
        "sha256": file_sha256(destination),
    }
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text())
        if existing != payload:
            raise RuntimeError("existing settlement source ledger contract changed")
    else:
        _write_json(manifest_path, payload)
    return pl.read_parquet(destination), payload


def _residual_contract(values: np.ndarray, *, minimum_scale_bps: float) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 250:
        raise RuntimeError(f"settlement bridge has insufficient paired residuals: {len(finite)}")
    location = float(np.median(finite))
    centered = finite - location
    robust_sigma = float(1.4826 * np.median(np.abs(centered)))
    rmse = float(np.sqrt(np.mean(centered * centered)))
    scale = max(robust_sigma, rmse, minimum_scale_bps)
    return {
        "paired_markets": len(finite),
        "location_bps": location,
        "scale_bps": scale,
        "mae_bps": float(np.mean(np.abs(finite))),
        "p95_absolute_bps": float(np.quantile(np.abs(finite), 0.95)),
        "p99_absolute_bps": float(np.quantile(np.abs(finite), 0.99)),
    }


def build_source_preserving_bridge_panel(
    config: TournamentDataConfig, *, force: bool = False
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Build uncertainty-aware TWAP supervision without overwriting official history."""

    cache = bridge_cache(config)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "middle-panel.parquet"
    manifest_path = cache / "panel-manifest.json"
    if destination.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != file_sha256(destination):
            raise RuntimeError("source-preserving bridge panel changed after checkpoint")
        return pl.read_parquet(destination), manifest

    base, base_manifest = build_middle_panel(config, force=False)
    seed, seed_manifest = _bridge_seed(config, force=force)
    bridge = config.raw["settlement_bridge"]
    calibration_start = datetime.fromisoformat(bridge["calibration_start"])
    calibration_end = datetime.fromisoformat(bridge["calibration_end"])
    validation_end = datetime.fromisoformat(bridge["validation_end"])
    if not calibration_start < calibration_end <= validation_end <= config.fit_end:
        raise RuntimeError("settlement bridge chronology is invalid")
    exact_start = datetime.fromisoformat(config.raw["windows"]["exact_twap_start"])
    refprice_start = datetime.fromisoformat(config.raw["windows"]["refprice_start"])
    minimum_scale = float(bridge["minimum_scale_bps"])
    paired = seed.filter(
        pl.col("window_start").is_between(
            calibration_start, calibration_end, closed="left"
        )
        & pl.col("authentic_margin_bps").is_finite()
    )
    ref_contract = _residual_contract(
        (
            paired["authentic_margin_bps"] - paired["proxy_margin_bps"]
        ).to_numpy(),
        minimum_scale_bps=minimum_scale,
    )
    binance_contract = _residual_contract(
        (
            paired.filter(pl.col("binance_label_complete").fill_null(False))[
                "authentic_margin_bps"
            ]
            - paired.filter(pl.col("binance_label_complete").fill_null(False))[
                "binance_raw_margin_bps"
            ]
        ).to_numpy(),
        minimum_scale_bps=minimum_scale,
    )
    validation = seed.filter(
        pl.col("window_start").is_between(calibration_end, validation_end, closed="left")
        & pl.col("authentic_margin_bps").is_finite()
    )
    ref_validation = _residual_contract(
        (validation["authentic_margin_bps"] - validation["proxy_margin_bps"]).to_numpy(),
        minimum_scale_bps=minimum_scale,
    )
    binance_validation_frame = validation.filter(
        pl.col("binance_label_complete").fill_null(False)
    )
    binance_validation = _residual_contract(
        (
            binance_validation_frame["authentic_margin_bps"]
            - binance_validation_frame["binance_raw_margin_bps"]
        ).to_numpy(),
        minimum_scale_bps=minimum_scale,
    )
    ledger = seed.select(
        "market_id",
        "window_start",
        "official_outcome",
        "legacy_open_price",
        "legacy_close_price",
        "proxy_label_up",
        "proxy_margin_bps",
        "binance_raw_label_up",
        "binance_raw_margin_bps",
        "binance_label_complete",
        "authentic_label_up",
        "authentic_margin_bps",
    ).with_columns(
        (pl.col("official_outcome") == "up").cast(pl.Int8).alias("official_label_up"),
        (
            (pl.col("legacy_close_price") / pl.col("legacy_open_price")).log()
            * 10_000.0
        ).alias("official_margin_bps"),
    )
    labels = (
        base.select("market_id", "window_start", "label_up").unique("market_id")
        .rename({"label_up": "base_official_label_up"})
        .join(ledger, on=["market_id", "window_start"], how="left", validate="1:1")
        .with_columns(
            pl.col("official_label_up")
            .fill_null(pl.col("base_official_label_up"))
            .cast(pl.Int8)
            .alias("official_label_up")
        )
    )
    official_period = pl.col("window_start") >= config.fit_end
    exact_period = pl.col("window_start") >= exact_start
    refprice_period = pl.col("window_start") >= refprice_start
    has_exact = pl.col("authentic_margin_bps").is_finite()
    has_refprice = pl.col("proxy_margin_bps").is_finite()
    has_binance = (
        pl.col("binance_label_complete").fill_null(False)
        & pl.col("binance_raw_margin_bps").is_finite()
    )
    ref_location = float(ref_contract["location_bps"])
    ref_scale = float(ref_contract["scale_bps"])
    binance_location = float(binance_contract["location_bps"])
    binance_scale = float(binance_contract["scale_bps"])
    ref_floor = float(bridge["refprice_minimum_weight"])
    binance_floor = float(bridge["binance_minimum_weight"])
    fallback_weight = float(bridge["fallback_weight"])

    source_margin = (
        pl.when(official_period & has_exact)
        .then(pl.col("authentic_margin_bps"))
        .when(official_period & has_refprice)
        .then(pl.col("proxy_margin_bps") + ref_location)
        .when(exact_period & has_exact)
        .then(pl.col("authentic_margin_bps"))
        .when(refprice_period & has_refprice)
        .then(pl.col("proxy_margin_bps") + ref_location)
        .when(has_binance)
        .then(pl.col("binance_raw_margin_bps") + binance_location)
        .otherwise(pl.col("official_margin_bps"))
    )
    source_scale = (
        pl.when(refprice_period).then(ref_scale).otherwise(binance_scale)
    )
    labels = labels.with_columns(
        source_margin.alias("target_margin_bps"),
        source_scale.alias("bridge_uncertainty_bps"),
    ).with_columns(
        (pl.col("target_margin_bps") / pl.col("bridge_uncertainty_bps")).alias(
            "settlement_bridge_z"
        )
    )
    probability = ndtr(labels["settlement_bridge_z"].to_numpy().astype(np.float64))
    labels = labels.with_columns(
        pl.Series("bridge_probability_target", probability).clip(0.001, 0.999)
    ).with_columns(
        pl.when(official_period)
        .then(pl.col("official_label_up").cast(pl.Float64))
        .otherwise(pl.col("bridge_probability_target"))
        .alias("bridge_probability_target"),
        pl.when(official_period)
        .then(pl.col("official_label_up"))
        .otherwise((pl.col("target_margin_bps") >= 0).cast(pl.Int8))
        .alias("bridge_label_up"),
        pl.when(official_period).then(1.0)
        .when(exact_period & has_exact).then(1.0)
        .when(refprice_period & has_refprice).then(
            ref_floor
            + (1.0 - ref_floor)
            * (2.0 * (pl.col("bridge_probability_target") - 0.5).abs())
        )
        .when(has_binance).then(
            binance_floor
            + (1.0 - binance_floor)
            * (2.0 * (pl.col("bridge_probability_target") - 0.5).abs())
        )
        .otherwise(fallback_weight)
        .alias("label_weight"),
        pl.when(official_period & has_exact).then(pl.lit("official_twap60_exact"))
        .when(official_period).then(pl.lit("official_twap60_source_gap"))
        .when(exact_period & has_exact).then(pl.lit("exact_twap60_bridge"))
        .when(refprice_period & has_refprice).then(pl.lit("refprice_twap60_bridge"))
        .when(has_binance).then(pl.lit("binance_twap60_bridge"))
        .otherwise(pl.lit("official_legacy_auxiliary"))
        .alias("label_source"),
    )
    invalid = labels.filter(
        pl.col("bridge_label_up").is_null()
        | pl.col("bridge_probability_target").is_null()
        | ~pl.col("bridge_probability_target").is_finite()
        | (pl.col("bridge_probability_target") < 0)
        | (pl.col("bridge_probability_target") > 1)
        | pl.col("target_margin_bps").is_null()
        | ~pl.col("target_margin_bps").is_finite()
        | (pl.col("label_weight") <= 0)
    )
    if invalid.height:
        raise RuntimeError(
            f"{invalid.height} markets lack valid source-preserving bridge supervision"
        )
    panel = (
        base.drop("label_up")
        .join(
            labels.select(
                "market_id",
                pl.col("bridge_label_up").alias("label_up"),
                "official_label_up",
                "bridge_probability_target",
                "settlement_bridge_z",
                "bridge_uncertainty_bps",
                "target_margin_bps",
                "label_weight",
                "label_source",
            ),
            on="market_id",
            how="inner",
            validate="m:1",
        )
    )
    if panel.height != base.height or panel["market_id"].n_unique() != base["market_id"].n_unique():
        raise RuntimeError("settlement bridge changed retained feature coverage")
    panel.write_parquet(destination, compression="zstd", statistics=True)
    market_labels = panel.select(
        "market_id",
        "window_start",
        "label_up",
        "official_label_up",
        "label_source",
        "label_weight",
    ).unique("market_id")
    label_coverage = (
        market_labels.group_by("label_source")
        .agg(
            pl.len().alias("markets"),
            pl.col("window_start").min().alias("first_market"),
            pl.col("window_start").max().alias("last_market"),
            pl.col("label_weight").mean().alias("mean_weight"),
            (pl.col("label_up") != pl.col("official_label_up")).sum().alias(
                "different_from_historical_official"
            ),
        )
        .sort("first_market")
        .to_dicts()
    )
    manifest = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "rows": panel.height,
        "markets": panel["market_id"].n_unique(),
        "range_start": config.source_start.isoformat(),
        "range_end": config.sealed_end.isoformat(),
        "feature_groups": base_manifest["feature_groups"],
        "coverage": base_manifest["coverage"],
        "base_panel": base_manifest,
        "settlement_source_ledger": seed_manifest,
        "bridge_calibration": {
            "start_inclusive": calibration_start.isoformat(),
            "end_exclusive": calibration_end.isoformat(),
            "sealed_rows_used": False,
            "refprice_to_exact": ref_contract,
            "binance_to_exact": binance_contract,
        },
        "bridge_validation": {
            "start_inclusive": calibration_end.isoformat(),
            "end_exclusive": validation_end.isoformat(),
            "parameters_refitted": False,
            "refprice_to_exact": ref_validation,
            "binance_to_exact": binance_validation,
        },
        "label_coverage": label_coverage,
        "historical_official_label_preserved": True,
        "source_values_remain_distinct": True,
        "probabilistic_proxy_supervision": True,
        "all_markets_retained": True,
        "all_label_weights_nonzero": True,
        "twap_inference_feature": False,
        "database_mutations": False,
        "new_tables": False,
        "new_ingesters": False,
        "new_sources": False,
        "sha256": file_sha256(destination),
    }
    _write_json(manifest_path, manifest)
    return panel, manifest


def extract_normalized_twap_labels(
    config: TournamentDataConfig, *, force: bool = False
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Extract exact TWAP60 labels from existing archive and live relations."""

    cache = normalized_cache(config)
    directory = cache / "exact-twap60-labels"
    directory.mkdir(parents=True, exist_ok=True)
    query_path = config.package_root / config.raw["paths"]["normalized_label_source_sql"]
    exact_start = datetime.fromisoformat(config.raw["windows"]["exact_twap_start"])
    range_start = exact_start.replace(hour=0, minute=0, second=0, microsecond=0)
    contract = {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "range_start": range_start.isoformat(),
        "required_exact_start": exact_start.isoformat(),
        "range_end": config.sealed_end.isoformat(),
        "query_sha256": file_sha256(query_path),
        "source_relations": [
            "market_data.pmdata_chainlink_btcusd_twap",
            "market_data.polymarket_chainlink_btcusd_twap",
        ],
        "archive_precedes_live_on_overlap": True,
        "read_only": True,
        "database_mutations": False,
        "new_sources": False,
    }
    final = cache / "exact-twap60-label-manifest.json"
    partial = cache / "exact-twap60-label-manifest.partial.json"
    if final.is_file() and not force and not partial.exists():
        payload = json.loads(final.read_text())
        if payload["contract"] != contract:
            raise RuntimeError("exact TWAP60 label extraction contract changed")
        for row in payload["partitions"]:
            path = cache / row["path"]
            if not path.is_file() or file_sha256(path) != row["sha256"]:
                raise RuntimeError(f"exact TWAP60 label checkpoint changed: {path}")
        frames = [pl.read_parquet(cache / row["path"]) for row in payload["partitions"]]
        return pl.concat(frames, how="diagonal_relaxed", rechunk=True), payload

    if force:
        partial.unlink(missing_ok=True)
    records: list[dict[str, Any]] = []
    if partial.is_file():
        payload = json.loads(partial.read_text())
        if payload["contract"] != contract:
            raise RuntimeError("partial exact TWAP60 label extraction contract changed")
        records = payload["partitions"]
    completed = {Path(row["path"]).stem for row in records}
    cursor = range_start
    query = query_path.read_text()
    while cursor < config.sealed_end:
        end = min(cursor + timedelta(days=1), config.sealed_end)
        if cursor.date().isoformat() not in completed:
            frame = _isolated_query_frame(
                query,
                {"batch_start": cursor, "batch_end": end},
                cursor_name=f"normalized_twap60_labels_{cursor:%Y%m%d}",
            )
            destination = directory / f"{cursor.date().isoformat()}.parquet"
            frame.write_parquet(destination, compression="zstd", statistics=True)
            records.append({
                "path": str(destination.relative_to(cache)),
                "rows": frame.height,
                "sha256": file_sha256(destination),
            })
            _write_json(partial, {"contract": contract, "partitions": records})
            print(f"middle normalization: exact TWAP60 {cursor.date()} {frame.height:,} rows", flush=True)
        cursor = end
    payload = {"contract": contract, "partitions": records}
    _write_json(final, payload)
    partial.unlink(missing_ok=True)
    frames = [pl.read_parquet(cache / row["path"]) for row in records]
    return pl.concat(frames, how="diagonal_relaxed", rechunk=True), payload


def build_normalized_middle_panel(
    config: TournamentDataConfig, *, force: bool = False
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Replace settlement supervision without changing the established features."""

    cache = normalized_cache(config)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "middle-panel.parquet"
    manifest_path = cache / "panel-manifest.json"
    if destination.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != file_sha256(destination):
            raise RuntimeError("normalized middle panel changed after checkpoint")
        return pl.read_parquet(destination), manifest

    base, base_manifest = build_middle_panel(config, force=False)
    seed, seed_manifest = _normalized_seed(config, force=force)
    exact_raw, exact_manifest = extract_normalized_twap_labels(config, force=force)
    exact = authentic_labels(exact_raw).select(
        "market_id",
        pl.col("authentic_label_up").alias("exact_label_up"),
        pl.col("authentic_margin_bps").alias("exact_margin_bps"),
        "twap_open_source_family",
        "twap_close_source_family",
    )
    source_labels = load_source_group(config.standard, "labels")
    refprice = load_source_group(config.standard, "refprice")
    proxy = construct_proxy_labels(
        source_labels, refprice, time_column="source_timestamp"
    ).select("market_id", "proxy_label_up", "proxy_margin_bps")
    markets = base.select("market_id", "window_start", "label_up").unique("market_id")
    labels = (
        markets.rename({"label_up": "official_label_up"})
        .join(proxy, on="market_id", how="left", validate="1:1")
        .join(
            seed.select(
                "market_id", "binance_raw_label_up", "binance_raw_margin_bps",
                "binance_label_complete",
                pl.col("proxy_label_up").alias("seed_proxy_label_up"),
                pl.col("proxy_margin_bps").alias("seed_proxy_margin_bps"),
            ),
            on="market_id", how="left", validate="1:1",
        )
        .join(exact, on="market_id", how="left", validate="1:1")
        .with_columns(
            pl.when(pl.col("proxy_margin_bps").is_finite())
            .then(pl.col("proxy_label_up"))
            .otherwise(pl.col("seed_proxy_label_up"))
            .alias("proxy_label_up"),
            pl.when(pl.col("proxy_margin_bps").is_finite())
            .then(pl.col("proxy_margin_bps"))
            .otherwise(pl.col("seed_proxy_margin_bps"))
            .alias("proxy_margin_bps"),
        )
    )

    windows = config.raw["windows"]
    normalization = config.raw["normalization"]
    refprice_start = datetime.fromisoformat(windows["refprice_start"])
    exact_start = datetime.fromisoformat(windows["exact_twap_start"])
    ref_floor = float(normalization["refprice_minimum_weight"])
    ref_band = float(normalization["refprice_error_band_bps"])
    binance_floor = float(normalization["binance_minimum_weight"])
    binance_ceiling = float(normalization["binance_maximum_weight"])
    binance_band = float(normalization["binance_full_weight_margin_bps"])
    fallback_weight = float(normalization["fallback_weight"])
    exact_period = pl.col("window_start") >= exact_start
    official_period = pl.col("window_start") >= config.fit_end
    refprice_period = pl.col("window_start") >= refprice_start
    complete_binance = pl.col("binance_label_complete").fill_null(False)
    has_exact = pl.col("exact_label_up").is_not_null()
    valid_proxy = pl.col("proxy_margin_bps").is_finite() & pl.col("proxy_label_up").is_not_null()
    labels = labels.with_columns(
        pl.when(official_period).then(pl.col("official_label_up"))
        .when(exact_period & has_exact).then(pl.col("exact_label_up"))
        .when(refprice_period & valid_proxy).then(pl.col("proxy_label_up"))
        .when(complete_binance).then(pl.col("binance_raw_label_up"))
        .otherwise(pl.col("proxy_label_up"))
        .cast(pl.Int8).alias("normalized_label_up"),
        pl.when(exact_period & has_exact).then(pl.col("exact_margin_bps"))
        .when(refprice_period & valid_proxy).then(pl.col("proxy_margin_bps"))
        .when(complete_binance).then(pl.col("binance_raw_margin_bps"))
        .otherwise(pl.col("proxy_margin_bps"))
        .alias("target_margin_bps"),
        pl.when(official_period).then(1.0)
        .when(exact_period & has_exact).then(1.0)
        .when(refprice_period & valid_proxy).then(
            ref_floor
            + (1.0 - ref_floor)
            * (pl.col("proxy_margin_bps").abs() / ref_band).clip(0.0, 1.0)
        )
        .when(complete_binance).then(
            binance_floor
            + (binance_ceiling - binance_floor)
            * (pl.col("binance_raw_margin_bps").abs() / binance_band).clip(0.0, 1.0)
        )
        .otherwise(fallback_weight).alias("label_weight"),
        pl.when(official_period & has_exact).then(pl.lit("exact_chainlink_twap60"))
        .when(official_period).then(pl.lit("official_twap60_capture_gap"))
        .when(exact_period & has_exact).then(pl.lit("exact_chainlink_twap60"))
        .when(refprice_period & valid_proxy).then(pl.lit("refprice_reconstructed_twap60"))
        .when(complete_binance).then(pl.lit("binance_reconstructed_twap60"))
        .otherwise(pl.lit("refprice_low_weight_fallback"))
        .alias("label_source"),
    )
    missing_exact = labels.filter(exact_period & pl.col("exact_label_up").is_null())
    invalid = labels.filter(
        pl.col("normalized_label_up").is_null()
        | (
            ~official_period
            & (pl.col("target_margin_bps").is_null() | ~pl.col("target_margin_bps").is_finite())
        )
        | (pl.col("label_weight") <= 0)
    )
    if invalid.height:
        summary = invalid.group_by("label_source").agg(
            pl.len().alias("markets"),
            pl.col("normalized_label_up").is_null().sum().alias("missing_labels"),
            pl.col("target_margin_bps").is_null().sum().alias("missing_margins"),
            pl.col("label_weight").is_null().sum().alias("missing_weights"),
        ).to_dicts()
        raise RuntimeError(
            f"{invalid.height} markets lack nonzero normalized supervision: {summary}"
        )
    observed_exact = labels.filter(official_period & has_exact)
    disagreements = observed_exact.filter(
        pl.col("exact_label_up").cast(pl.Int8) != pl.col("official_label_up")
    )
    if disagreements.height:
        raise RuntimeError(
            f"{disagreements.height} captured TWAP60 labels disagree with official outcomes"
        )

    entry = config.raw["entry"]
    panel = (
        base.filter(
            pl.col("seconds_elapsed").is_between(
                int(entry["start_second"]), int(entry["end_second_inclusive"]), closed="both"
            )
            & ((pl.col("seconds_elapsed") - int(entry["start_second"]))
               % int(entry["cadence_seconds"]) == 0)
        )
        .drop("label_up")
        .join(
            labels.select(
                "market_id", pl.col("normalized_label_up").alias("label_up"),
                "label_weight", "label_source", "target_margin_bps",
            ),
            on="market_id", how="inner", validate="m:1",
        )
    )
    if panel["market_id"].n_unique() != base["market_id"].n_unique():
        raise RuntimeError("normalization changed market coverage")
    panel.write_parquet(destination, compression="zstd", statistics=True)
    label_coverage = (
        labels.group_by("label_source")
        .agg(
            pl.len().alias("markets"),
            pl.col("window_start").min().alias("first_market"),
            pl.col("window_start").max().alias("last_market"),
            pl.col("label_weight").mean().alias("mean_weight"),
            pl.col("label_weight").min().alias("minimum_weight"),
        )
        .sort("first_market")
        .to_dicts()
    )
    manifest = {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "rows": panel.height,
        "markets": panel["market_id"].n_unique(),
        "range_start": config.source_start.isoformat(),
        "range_end": config.sealed_end.isoformat(),
        "entry_seconds": [int(entry["start_second"]), int(entry["end_second_inclusive"])],
        "feature_groups": base_manifest["feature_groups"],
        "coverage": base_manifest["coverage"],
        "base_panel": base_manifest,
        "normalization_seed": seed_manifest,
        "exact_twap_source": exact_manifest,
        "label_coverage": label_coverage,
        "all_markets_retained": True,
        "all_label_weights_nonzero": True,
        "post_cutover_canonical_outcomes": True,
        "captured_exact_gap_markets": missing_exact.height,
        "captured_exact_gaps_use_refprice_before_policy_development": True,
        "captured_exact_gaps_use_official_outcomes_in_development_and_test": True,
        "twap_inference_feature": False,
        "authentic_only_filter": False,
        "database_mutations": False,
        "new_tables": False,
        "new_ingesters": False,
        "new_sources": False,
        "sha256": file_sha256(destination),
    }
    _write_json(manifest_path, manifest)
    return panel, manifest


def extract_spot_l2(config: TournamentDataConfig, *, force: bool = False) -> dict[str, Any]:
    """Extract only the existing qualified L2 states needed at decision timestamps."""

    cache = middle_cache(config)
    partitions = cache / "spot-l2"
    partitions.mkdir(parents=True, exist_ok=True)
    query_path = config.package_root / config.raw["paths"]["spot_l2_source_sql"]
    contract = {
        "schema_version": SCHEMA_VERSION,
        "range_start": config.source_start.isoformat(),
        "range_end": config.sealed_end.isoformat(),
        "query_sha256": file_sha256(query_path),
        "source_relation": config.raw["sources"]["binance_spot_l2_relation"],
        "read_only": True,
        "database_mutations": False,
        "new_table": False,
        "maximum_causal_age_seconds": 2,
    }
    manifest_path = cache / "spot-l2-manifest.json"
    if manifest_path.is_file() and not force:
        payload = json.loads(manifest_path.read_text())
        if payload["contract"] != contract:
            raise RuntimeError("spot-L2 extraction contract changed")
        for row in payload["partitions"]:
            path = cache / row["path"]
            if not path.is_file() or file_sha256(path) != row["sha256"]:
                raise RuntimeError(f"spot-L2 checkpoint changed: {path}")
        return payload

    records: list[dict[str, Any]] = []
    cursor = config.source_start
    while cursor < config.sealed_end:
        end = min(cursor + timedelta(days=1), config.sealed_end)
        destination = partitions / f"{cursor.date().isoformat()}.parquet"
        if force or not destination.is_file():
            frame = _isolated_query_frame(
                query_path.read_text(),
                {"batch_start": cursor, "batch_end": end},
                cursor_name=f"middle_strategy_l2_{cursor:%Y%m%d}",
            )
            frame.write_parquet(destination, compression="zstd", statistics=True)
            print(f"middle data: L2 {cursor.date()} {frame.height:,} rows", flush=True)
        record = {
            "path": str(destination.relative_to(cache)),
            "rows": pl.scan_parquet(destination).select(pl.len()).collect().item(),
            "sha256": file_sha256(destination),
        }
        records.append(record)
        _write_json(
            cache / "spot-l2-manifest.partial.json",
            {"contract": contract, "partitions": records},
        )
        cursor = end
    payload = {
        "contract": contract,
        "partitions": records,
        "rows": sum(int(row["rows"]) for row in records),
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_json(manifest_path, payload)
    (cache / "spot-l2-manifest.partial.json").unlink(missing_ok=True)
    return payload


def build_middle_panel(
    config: TournamentDataConfig, *, force: bool = False
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Preserve the complete base panel and causally attach optional spot L2."""

    cache = middle_cache(config)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "middle-panel.parquet"
    manifest_path = cache / "panel-manifest.json"
    if destination.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != file_sha256(destination):
            raise RuntimeError("middle panel changed after checkpoint")
        return pl.read_parquet(destination), manifest

    base, base_manifest = build_panel(config, force=False)
    l2_manifest = extract_spot_l2(config, force=force)
    records = [row for row in l2_manifest["partitions"] if row["rows"]]
    if records:
        pieces = []
        for row in records:
            path = cache / row["path"]
            day = datetime.fromisoformat(path.stem).replace(tzinfo=UTC)
            core = base.filter(
                pl.col("observed_at").is_between(day, day + timedelta(days=1), closed="left")
            ).select(*KEY_COLUMNS, "btc_close")
            if core.is_empty():
                continue
            source = pl.read_parquet(path).unique(subset=["second_start"], keep="last").sort(
                "available_at"
            )
            pieces.append(
                join_qualified_l2(core, source).select(*KEY_COLUMNS, *L2_FEATURES)
            )
        qualified = pl.concat(pieces, how="vertical_relaxed", rechunk=True)
        panel = base.join(qualified, on=list(KEY_COLUMNS), how="left", validate="1:1")
    else:
        panel = base.with_columns(
            *(pl.lit(None, dtype=pl.Float64).alias(name) for name in L2_FEATURES)
        )
    panel = panel.with_columns(
        pl.any_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite() for name in L2_FEATURES
        ).alias("has_spot_l2")
    )
    if panel.height != base.height or panel["market_id"].n_unique() != base["market_id"].n_unique():
        raise RuntimeError("optional L2 attachment changed base market coverage")
    panel.write_parquet(destination, compression="zstd", statistics=True)
    feature_groups = dict(base_manifest["feature_groups"])
    feature_groups["spot_l2"] = list(L2_FEATURES)
    execution = [
        name
        for name in panel.columns
        if name.startswith(("up_ask_vwap_", "down_ask_vwap_", "pm_"))
    ]
    feature_groups["execution"] = execution
    coverage = dict(base_manifest["coverage"])
    coverage["spot_l2"] = {
        "rows": panel.filter("has_spot_l2").height,
        "markets": panel.filter("has_spot_l2")["market_id"].n_unique(),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rows": panel.height,
        "markets": panel["market_id"].n_unique(),
        "range_start": config.source_start.isoformat(),
        "range_end": config.sealed_end.isoformat(),
        "feature_groups": feature_groups,
        "coverage": coverage,
        "base_panel": base_manifest,
        "spot_l2_source": l2_manifest,
        "optional_missingness_preserves_rows": True,
        "kraken_l2_included": False,
        "twap_inference_feature": False,
        "authentic_only_filter": False,
        "database_mutations": False,
        "new_tables": False,
        "new_ingesters": False,
        "new_sources": False,
        "sha256": file_sha256(destination),
    }
    _write_json(manifest_path, manifest)
    return panel, manifest
