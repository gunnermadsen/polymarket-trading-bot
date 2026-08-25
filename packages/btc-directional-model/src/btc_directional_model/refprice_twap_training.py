"""Frozen RefPrice-primary and TWAP-target training for the five paper families.

This runner is offline-only. It does not export runtime bundles, mutate trading
processes, run backfills, or alter container configuration.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tomllib
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import psycopg
import sklearn
from sklearn.metrics import log_loss

from . import continuous_edge_training as q5_lineage
from . import fair_value_challenger_tournament as specialist_lineage
from . import middle_market_ablation_tournament as middle_lineage
from .chainlink_oi_features import (
    BINANCE_OI_FEATURES,
    CHAINLINK_REFPRICE_FEATURES,
    _attach_open_interest_features,
    _attach_refprice_features,
)
from .continuous_edge_training import (
    BOOK_RAW_FEATURES,
    CORE_FEATURES,
    ORACLE_FEATURES,
    PRIMARY_FEATURES,
    VWAP_QUANTITIES,
    attach_book_features,
)
from .core_extract import configure_read_only_connection, database_connection, file_sha256
from .core_features import (
    attach_causal_oracle_rounds,
    derive_core_point_in_time_features,
    derive_oracle_point_in_time_features,
)

SCHEMA_VERSION = "btc-refprice-twap-lineage-training-v2"
DATASET_SCHEMA_VERSION = "btc-refprice-twap-lineage-dataset-v2"
ARTIFACT_SCHEMA_VERSION = "btc-refprice-twap-lineage-model-v2"
ARMS = ("R", "T", "RT")
FAMILIES = (
    "q5",
    "stratified_payoff",
    "regime_calibrated",
    "full_combined",
    "specialist_distilled",
)
SQL_FILES = (
    "btc-refprice-twap-core-source.sql",
    "btc-refprice-twap-oracle-source.sql",
    "btc-refprice-twap-capacity-source.sql",
    "btc-refprice-twap-refprice-source.sql",
    "btc-refprice-twap-input-source.sql",
    "btc-refprice-twap-open-interest-source.sql",
    "btc-refprice-twap-label-source.sql",
)
JOIN_KEYS = ("market_id", "window_start", "observed_at", "seconds_elapsed")
PRICE_BUCKET_EDGES = (0.0, 0.65, 0.75, 0.85, 1.01)
TWAP_LABEL_COLUMNS = (
    "twap_start_source_timestamp",
    "twap_start_received_at",
    "twap_start_valid_from",
    "twap_start_expires_at",
    "twap_start_price",
    "twap_end_source_timestamp",
    "twap_end_received_at",
    "twap_end_valid_from",
    "twap_end_expires_at",
    "twap_end_price",
    "twap_label_up",
)
TWAP_INPUT_FEATURES = (
    "chainlink_twap30_return_5s_bps",
    "chainlink_twap30_return_15s_bps",
    "chainlink_twap30_return_30s_bps",
    "chainlink_twap30_return_60s_bps",
    "chainlink_twap60_return_5s_bps",
    "chainlink_twap60_return_15s_bps",
    "chainlink_twap60_return_30s_bps",
    "chainlink_twap60_return_60s_bps",
    "chainlink_twap30_binance_basis_bps",
    "chainlink_twap60_binance_basis_bps",
    "chainlink_twap30_boundary_gap_bps",
    "chainlink_twap60_boundary_gap_bps",
    "chainlink_twap30_60_spread_bps",
)


@dataclass(frozen=True)
class WindowConfig:
    data_start: datetime
    twap_fit_start: datetime
    outcome_fit_end: datetime
    admission_end: datetime
    calibration_end: datetime
    evaluation_end: datetime
    watermark: date


@dataclass(frozen=True)
class ExecutionConfig:
    quantities: tuple[int, ...]
    minimum_seconds_after_open: int
    maximum_seconds_after_open: int
    training_cadence_seconds: int
    freshness_seconds: int
    refprice_freshness_seconds: int
    maximum_depth_participation: float
    execution_reserve_per_share: float
    stress_slippage_per_share: float
    evaluation_quantity: int


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    calibration_c: float
    calibration_max_iter: int
    local_calibration_minimum_rows: int
    local_calibration_shrinkage_rows: int
    loss_max_iter: int
    loss_max_leaf_nodes: int
    loss_min_samples_leaf: int
    loss_l2_regularization: float


@dataclass(frozen=True)
class PathConfig:
    data: Path
    runs: Path
    committed_results: Path


@dataclass(frozen=True)
class FrozenTrainingConfig:
    source_path: Path
    package_root: Path
    profile: str
    random_seed: int
    windows: WindowConfig
    execution: ExecutionConfig
    model: ModelConfig
    policies: dict[str, Any]
    paths: PathConfig
    incumbents: dict[str, Path]


def load_config(path: Path) -> FrozenTrainingConfig:
    source = path.resolve()
    package_root = source.parent.parent
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    training = raw["training"]
    if (
        training.get("paper_only") is not True
        or training.get("live_capital_allowed") is not False
        or training.get("runtime_exported") is not False
    ):
        raise ValueError("RefPrice/TWAP training must remain offline and paper-only")
    windows = dict(raw["windows"])
    windows["watermark"] = date.fromisoformat(str(windows["watermark"]))
    for key in tuple(windows):
        if key != "watermark":
            windows[key] = _utc(windows[key])
    paths = PathConfig(**{key: _path(package_root, value) for key, value in raw["paths"].items()})
    config = FrozenTrainingConfig(
        source_path=source,
        package_root=package_root,
        profile=str(training["profile"]),
        random_seed=int(training["random_seed"]),
        windows=WindowConfig(**windows),
        execution=ExecutionConfig(
            **{
                **raw["execution"],
                "quantities": tuple(int(value) for value in raw["execution"]["quantities"]),
            }
        ),
        model=ModelConfig(**raw["model"]),
        policies=raw["policies"],
        paths=paths,
        incumbents={key: _path(package_root, value) for key, value in raw["incumbents"].items()},
    )
    _validate_config(config)
    return config


def _validate_config(config: FrozenTrainingConfig) -> None:
    w = config.windows
    if not (
        w.data_start
        < w.twap_fit_start
        < w.outcome_fit_end
        < w.admission_end
        < w.calibration_end
        < w.evaluation_end
    ):
        raise ValueError("frozen training windows are not strictly chronological")
    if w.evaluation_end.date() != w.watermark + timedelta(days=1):
        raise ValueError("evaluation end must be the half-open day after the watermark")
    if config.execution.quantities != VWAP_QUANTITIES:
        raise ValueError("full VWAP quantity contract changed")
    if config.execution.maximum_depth_participation != 0.25:
        raise ValueError("depth participation must remain 25 percent")
    if config.execution.training_cadence_seconds != 5:
        raise ValueError("training cadence changed from the incumbent contract")
    if set(config.incumbents) != set(FAMILIES):
        raise ValueError("incumbent family contract changed")
    for path in config.incumbents.values():
        if not path.is_file():
            raise FileNotFoundError(path)


def run_frozen_training(
    config: FrozenTrainingConfig, *, force_data: bool = False
) -> tuple[Path, dict[str, Any]]:
    dataset_manifest = build_dataset(config, force=force_data)
    frame = _load_dataset(config, dataset_manifest)
    common_evaluation = _common_evaluation_frame(frame, config)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    temporary = config.paths.runs / f"{run_id}.partial"
    final = config.paths.committed_results / run_id
    if temporary.exists() or final.exists():
        raise FileExistsError(run_id)
    temporary.mkdir(parents=True)

    artifacts: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    for arm_index, arm in enumerate(ARMS):
        print(f"train arm {arm}: q5, stratified, regime, full, specialist", flush=True)
        arm_models = _train_arm(frame, config, arm, seed=config.random_seed + 10_000 * arm_index)
        artifacts[arm] = arm_models
        results[arm] = {}
        arm_dir = temporary / arm
        arm_dir.mkdir()
        for family in FAMILIES:
            model = arm_models[family]
            scored = _score_family(common_evaluation, model, family, config)
            selected = _apply_frozen_policy(scored, family, config)
            metrics = _evaluation_metrics(scored, selected, common_evaluation, config)
            results[arm][family] = metrics
            artifact_path = arm_dir / f"{family}.joblib"
            joblib.dump(
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "arm": arm,
                    "family": family,
                    "target": "official_then_twap_60" if arm in ("T", "RT") else "official_outcome",
                    "refprice_primary": True,
                    "twap_inputs": arm == "RT",
                    "runtime_exported": False,
                    "production_qualified": False,
                    "model": model,
                },
                artifact_path,
                compress=3,
            )
            selected.write_parquet(
                arm_dir / f"{family}-evaluation-ledger.parquet", compression="zstd"
            )
            _write_json(
                arm_dir / f"{family}-metrics.json",
                {**metrics, "artifact_sha256": file_sha256(artifact_path)},
            )

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": config.profile,
        "source_commit": _git_revision(config.package_root),
        "paper_only": True,
        "runtime_exported": False,
        "production_qualified": False,
        "trading_processes_changed": False,
        "container_images_changed": False,
        "configuration": {
            "path": str(config.source_path),
            "sha256": file_sha256(config.source_path),
            "windows": _jsonable(asdict(config.windows)),
            "execution": asdict(config.execution),
            "model": asdict(config.model),
            "policies": config.policies,
        },
        "dataset": dataset_manifest,
        "common_evaluation": {
            "rows": common_evaluation.height,
            "markets": common_evaluation["market_id"].n_unique(),
            "start": config.windows.calibration_end.isoformat(),
            "end": config.windows.evaluation_end.isoformat(),
            "target": "pmdata_twap_60",
        },
        "incumbents": _incumbent_identities(config),
        "results": results,
        "attribution": _attribution(results),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "R is the RefPrice refresh; T and RT use TWAP-60 labels from August 1 onward and official outcomes as pre-August auxiliary learning.",
            "Evaluation is projected five-share execution against recorded ask VWAP and does not model queue position.",
            "Incomplete source rows are excluded individually and never block the complete training run.",
            "No runtime artifact or trading-process configuration was exported.",
        ],
    }
    _write_json(temporary / "metrics.json", metrics)
    (temporary / "report.md").write_text(_render_report(metrics))
    (temporary / "dataset-manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n"
    )
    config.paths.committed_results.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    return final, metrics


def build_dataset(config: FrozenTrainingConfig, *, force: bool = False) -> dict[str, Any]:
    destination = config.paths.data
    destination.mkdir(parents=True, exist_ok=True)
    sql_root = config.package_root / "sql"
    contract = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "watermark": config.windows.watermark.isoformat(),
        "range_start": config.windows.data_start.isoformat(),
        "range_end": config.windows.evaluation_end.isoformat(),
        "training_cadence_seconds": config.execution.training_cadence_seconds,
        "vwap_quantities": list(config.execution.quantities),
        "sql_sha256": {name: file_sha256(sql_root / name) for name in SQL_FILES},
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        if any(manifest.get(key) != value for key, value in contract.items()):
            raise RuntimeError("frozen dataset cache contract changed")
        for partition in manifest["partitions"]:
            path = destination / partition["path"]
            if not path.is_file() or file_sha256(path) != partition["sha256"]:
                raise RuntimeError(f"dataset partition changed: {partition['path']}")
        return manifest

    if force:
        for path in destination.glob("*.parquet"):
            path.unlink()
    partitions: list[dict[str, Any]] = []
    current = config.windows.data_start
    while current < config.windows.evaluation_end:
        connection = database_connection()
        configure_read_only_connection(connection)
        try:
            end = min(current + timedelta(days=1), config.windows.evaluation_end)
            frame, audit = _build_daily_frame(connection, config, current, end)
            path = destination / f"{current:%Y-%m-%d}.parquet"
            frame.write_parquet(path, compression="zstd", statistics=True)
            partitions.append(
                {
                    "path": path.name,
                    "date": current.date().isoformat(),
                    "rows": frame.height,
                    "markets": frame["market_id"].n_unique() if frame.height else 0,
                    "sha256": file_sha256(path),
                    "audit": audit,
                }
            )
            print(
                f"dataset {current:%Y-%m-%d}: {frame.height:,} rows, "
                f"{partitions[-1]['markets']:,} markets",
                flush=True,
            )
            current = end
        finally:
            connection.close()
    manifest = {
        **contract,
        "created_at": datetime.now(UTC).isoformat(),
        "partitions": partitions,
        "rows": sum(item["rows"] for item in partitions),
        "markets_by_partition": sum(item["markets"] for item in partitions),
        "perfect_coverage_required": False,
        "interpolation_used": False,
        "historical_refprice_source": "market_data.chainlink_btcusd_reference_prices:pmdata_chainlink_streams",
        "twap_target_source": "market_data.pmdata_chainlink_btcusd_twap:60",
        "capacity_source": "polymarket.btc_market_capacity_execution_snapshots",
    }
    _write_json(manifest_path, manifest)
    return manifest


def _build_daily_frame(
    connection: psycopg.Connection[Any],
    config: FrozenTrainingConfig,
    start: datetime,
    end: datetime,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    sql_root = config.package_root / "sql"
    core = _query_frame(
        connection,
        (sql_root / SQL_FILES[0]).read_text(),
        {"batch_start": start, "batch_end": end},
        f"ref_twap_core_{start:%Y%m%d}",
    )
    raw_core_rows = core.height
    if core.is_empty():
        return core, {"raw_core_rows": 0, "reason": "no_core_rows"}
    complete = (
        core.filter(pl.col("seconds_elapsed").is_between(0, 240, closed="both"))
        .group_by("market_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("seconds_elapsed").n_unique().alias("seconds"),
            pl.col("seconds_elapsed").min().alias("minimum"),
            pl.col("seconds_elapsed").max().alias("maximum"),
        )
        .filter(
            (pl.col("rows") == 241)
            & (pl.col("seconds") == 241)
            & (pl.col("minimum") == 0)
            & (pl.col("maximum") == 240)
        )
        .select("market_id")
    )
    core = core.join(complete, on="market_id", how="inner")
    core = derive_core_point_in_time_features(core.sort(["market_id", "seconds_elapsed"]))

    oracle = _query_frame(
        connection,
        (sql_root / SQL_FILES[1]).read_text(),
        {"range_start": start, "range_end": end},
        f"ref_twap_oracle_{start:%Y%m%d}",
    )
    if oracle.height:
        core = attach_causal_oracle_rounds(core, oracle)
        core = derive_oracle_point_in_time_features(core).with_columns(
            pl.col("oracle_model_eligible").fill_null(False).alias("early_oracle_eligible")
        )
    else:
        core = core.with_columns(
            *[pl.lit(None, dtype=pl.Float64).alias(name) for name in ORACLE_FEATURES],
            pl.lit(False).alias("early_oracle_eligible"),
        )
    core = core.filter(
        pl.col("seconds_elapsed").is_between(
            config.execution.minimum_seconds_after_open,
            config.execution.maximum_seconds_after_open,
            closed="both",
        )
        & (
            (pl.col("seconds_elapsed") - config.execution.minimum_seconds_after_open)
            % config.execution.training_cadence_seconds
            == 0
        )
    )

    capacity = _query_frame(
        connection,
        (sql_root / SQL_FILES[2]).read_text(),
        {"batch_start": start, "batch_end": end},
        f"ref_twap_capacity_{start:%Y%m%d}",
    )
    raw_capacity_rows = capacity.height
    capacity = _strict_capacity_rows(capacity, config)
    frame = core.join(capacity, on=list(JOIN_KEYS), how="inner", validate="1:1")
    if frame.is_empty():
        return frame, {
            "raw_core_rows": raw_core_rows,
            "complete_core_markets": complete.height,
            "raw_capacity_rows": raw_capacity_rows,
            "strict_capacity_rows": capacity.height,
            "reason": "no_joined_rows",
        }
    frame = attach_book_features(frame)

    external_core = frame.select(
        *JOIN_KEYS,
        "btc_close",
        "opening_boundary",
        "btc_return_30s_bps",
        "btc_path_from_window_open_bps",
    )
    refprice = _query_frame(
        connection,
        (sql_root / SQL_FILES[3]).read_text(),
        {"range_start": start, "range_end": end},
        f"ref_twap_refprice_{start:%Y%m%d}",
    )
    complete_refprice = refprice.drop_nulls(
        [
            "source_timestamp",
            "received_at",
            "valid_from_timestamp",
            "expires_at",
            "price",
            "bid",
            "ask",
        ]
    )
    ref_features = (
        _attach_refprice_features(
            external_core,
            complete_refprice,
            max_age_seconds=config.execution.refprice_freshness_seconds,
        ).select(*JOIN_KEYS, *CHAINLINK_REFPRICE_FEATURES)
        if complete_refprice.height
        else _empty_feature_frame(external_core, CHAINLINK_REFPRICE_FEATURES)
    )
    frame = frame.join(ref_features, on=list(JOIN_KEYS), how="left", validate="1:1")

    twap_inputs = _query_frame(
        connection,
        (sql_root / SQL_FILES[4]).read_text(),
        {"range_start": start, "range_end": end},
        f"ref_twap_inputs_{start:%Y%m%d}",
    )
    twap_features = (
        _attach_twap_features(
            external_core,
            twap_inputs,
            max_age_seconds=config.execution.refprice_freshness_seconds,
        ).select(*JOIN_KEYS, *TWAP_INPUT_FEATURES)
        if twap_inputs.height
        else _empty_feature_frame(external_core, TWAP_INPUT_FEATURES)
    )
    frame = frame.join(twap_features, on=list(JOIN_KEYS), how="left", validate="1:1")

    interest = _query_frame(
        connection,
        (sql_root / SQL_FILES[5]).read_text(),
        {"range_start": start, "range_end": end},
        f"ref_twap_oi_{start:%Y%m%d}",
    )
    oi_features = (
        _attach_open_interest_features(
            external_core,
            interest,
            max_age_seconds=300,
        ).select(*JOIN_KEYS, *BINANCE_OI_FEATURES)
        if interest.height
        else _empty_feature_frame(external_core, BINANCE_OI_FEATURES)
    )
    frame = frame.join(oi_features, on=list(JOIN_KEYS), how="left", validate="1:1")

    labels = _query_frame(
        connection,
        (sql_root / SQL_FILES[6]).read_text(),
        {"range_start": start, "range_end": end},
        f"ref_twap_labels_{start:%Y%m%d}",
    )
    if labels.height:
        frame = frame.join(labels, on=["market_id", "window_start"], how="left", validate="m:1")
    else:
        frame = frame.with_columns(
            *[
                pl.lit(
                    None,
                    dtype=(
                        pl.Int8
                        if name == "twap_label_up"
                        else pl.Float64
                        if name.endswith("_price")
                        else pl.Datetime("us", "UTC")
                    ),
                ).alias(name)
                for name in TWAP_LABEL_COLUMNS
            ]
        )
    frame = frame.sort(["window_start", "market_id", "seconds_elapsed"])
    audit = {
        "raw_core_rows": raw_core_rows,
        "complete_core_markets": complete.height,
        "raw_capacity_rows": raw_capacity_rows,
        "strict_capacity_rows": capacity.height,
        "joined_rows": frame.height,
        "joined_markets": frame["market_id"].n_unique(),
        "refprice_source_rows": refprice.height,
        "incomplete_refprice_source_rows": refprice.height - complete_refprice.height,
        "refprice_eligible_rows": frame.drop_nulls(CHAINLINK_REFPRICE_FEATURES).height,
        "twap_input_source_rows": twap_inputs.height,
        "twap_input_eligible_rows": frame.drop_nulls(TWAP_INPUT_FEATURES).height,
        "oi_eligible_rows": frame.drop_nulls(BINANCE_OI_FEATURES).height,
        "twap_labeled_rows": frame.drop_nulls(["twap_label_up"]).height,
        "twap_labeled_markets": frame.drop_nulls(["twap_label_up"])["market_id"].n_unique(),
    }
    return frame, audit


def _attach_twap_features(
    core_frame: pl.DataFrame,
    source: pl.DataFrame,
    *,
    max_age_seconds: int,
) -> pl.DataFrame:
    """Attach strict point-in-time PMData TWAP features to decision rows."""

    required = (
        "source_timestamp",
        "provider_received_at",
        "valid_from_timestamp",
        "expires_at",
        "window_seconds",
        "twap_price",
    )
    prepared = (
        source.drop_nulls(required)
        .filter(
            pl.col("window_seconds").is_in([30, 60])
            & (pl.col("twap_price") > 0)
            & (pl.col("valid_from_timestamp") <= pl.col("source_timestamp"))
            & (pl.col("provider_received_at") >= pl.col("source_timestamp"))
        )
        .sort(["window_seconds", "source_timestamp", "provider_received_at"])
        .unique(["window_seconds", "source_timestamp"], keep="first", maintain_order=True)
    )
    ordered = core_frame.with_row_index("_twap_row").sort("observed_at")
    observed_us = ordered["observed_at"].cast(pl.Int64).to_numpy()
    result: dict[str, np.ndarray] = {}
    current_prices: dict[int, np.ndarray] = {}
    eligibility = np.ones(ordered.height, dtype=bool)
    for window in (30, 60):
        reports = prepared.filter(pl.col("window_seconds") == window).sort("source_timestamp")
        if reports.is_empty():
            eligibility[:] = False
            break
        source_us = reports["source_timestamp"].cast(pl.Int64).to_numpy()
        received_us = reports["provider_received_at"].cast(pl.Int64).to_numpy()
        prices = reports["twap_price"].cast(pl.Float64).to_numpy()
        indices = np.searchsorted(source_us, observed_us - 1, side="right") - 1
        for row in range(len(indices)):
            index = int(indices[row])
            while index >= 0 and received_us[index] > observed_us[row]:
                index -= 1
            indices[row] = index
        valid = indices >= 0
        age = np.full(ordered.height, np.iinfo(np.int64).max, dtype=np.int64)
        age[valid] = observed_us[valid] - source_us[indices[valid]]
        valid &= (age > 0) & (age <= max_age_seconds * 1_000_000)
        eligibility &= valid
        safe = np.maximum(indices, 0)
        current = prices[safe]
        current_prices[window] = current
        for seconds in (5, 15, 30, 60):
            lag = np.searchsorted(source_us, observed_us - seconds * 1_000_000, side="right") - 1
            for row in range(len(lag)):
                index = int(lag[row])
                while index >= 0 and received_us[index] > observed_us[row]:
                    index -= 1
                lag[row] = index
            lag_valid = lag >= 0
            target = observed_us - seconds * 1_000_000
            lag_age = np.full(ordered.height, np.iinfo(np.int64).max, dtype=np.int64)
            lag_age[lag_valid] = target[lag_valid] - source_us[lag[lag_valid]]
            lag_valid &= (lag_age >= 0) & (lag_age <= max_age_seconds * 1_000_000)
            eligibility &= lag_valid
            result[f"chainlink_twap{window}_return_{seconds}s_bps"] = (
                np.log(current / prices[np.maximum(lag, 0)]) * 10_000.0
            )
        result[f"chainlink_twap{window}_binance_basis_bps"] = (
            np.log(current / ordered["btc_close"].to_numpy()) * 10_000.0
        )
        result[f"chainlink_twap{window}_boundary_gap_bps"] = (
            np.log(current / ordered["opening_boundary"].to_numpy()) * 10_000.0
        )
    if not eligibility.any():
        return _empty_feature_frame(core_frame, TWAP_INPUT_FEATURES)
    result["chainlink_twap30_60_spread_bps"] = (
        np.log(current_prices[30] / current_prices[60]) * 10_000.0
    )
    positions = np.flatnonzero(eligibility)
    return (
        ordered[positions]
        .with_columns(*[pl.Series(name, values[positions]) for name, values in result.items()])
        .sort("_twap_row")
        .drop("_twap_row")
    )


def _strict_capacity_rows(frame: pl.DataFrame, config: FrozenTrainingConfig) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    full_curve = pl.all_horizontal(
        pl.col(name).is_not_null() & pl.col(name).is_finite() for name in BOOK_RAW_FEATURES
    )
    fresh = (
        pl.col("up_provider_received_at").is_not_null()
        & pl.col("down_provider_received_at").is_not_null()
        & (pl.col("up_provider_received_at") <= pl.col("observed_at"))
        & (pl.col("down_provider_received_at") <= pl.col("observed_at"))
        & (
            pl.col("up_provider_received_at")
            >= pl.col("observed_at") - pl.duration(seconds=config.execution.freshness_seconds)
        )
        & (
            pl.col("down_provider_received_at")
            >= pl.col("observed_at") - pl.duration(seconds=config.execution.freshness_seconds)
        )
    )
    required_depth = max(config.execution.quantities) / config.execution.maximum_depth_participation
    return frame.filter(
        ((pl.col("quality_flags") & 63) == 0)
        & full_curve
        & fresh
        & (pl.col("up_ask_depth") >= required_depth)
        & (pl.col("down_ask_depth") >= required_depth)
        & (
            (pl.col("seconds_elapsed") - config.execution.minimum_seconds_after_open)
            % config.execution.training_cadence_seconds
            == 0
        )
    )


def _empty_feature_frame(frame: pl.DataFrame, feature_names: tuple[str, ...]) -> pl.DataFrame:
    return (
        frame.head(0)
        .select(*JOIN_KEYS)
        .with_columns(*[pl.lit(None, dtype=pl.Float64).alias(name) for name in feature_names])
    )


def _query_frame(
    connection: psycopg.Connection[Any],
    query: str,
    parameters: dict[str, Any],
    cursor_name: str,
) -> pl.DataFrame:
    chunks: list[pl.DataFrame] = []
    columns: list[str] = []
    with connection.transaction():
        connection.execute("SET TRANSACTION READ ONLY")
        with connection.cursor(name=cursor_name) as cursor:
            cursor.execute(query, parameters)
            columns = [column.name for column in cursor.description or ()]
            while rows := cursor.fetchmany(25_000):
                chunks.append(
                    pl.DataFrame(
                        rows,
                        schema=columns,
                        orient="row",
                        infer_schema_length=None,
                    )
                )
    if not chunks:
        return pl.DataFrame({column: [] for column in columns})
    return pl.concat(chunks, how="vertical_relaxed", rechunk=True)


def _load_dataset(config: FrozenTrainingConfig, manifest: dict[str, Any]) -> pl.DataFrame:
    paths = [config.paths.data / item["path"] for item in manifest["partitions"] if item["rows"]]
    if not paths:
        raise RuntimeError("frozen dataset contains no eligible rows")
    frame = pl.concat(
        [pl.read_parquet(path) for path in paths],
        how="diagonal_relaxed",
        rechunk=True,
    )
    if "early_oracle_eligible" not in frame.columns:
        frame = frame.with_columns(
            pl.col("oracle_model_eligible").fill_null(False).alias("early_oracle_eligible")
        )
    return frame.sort(["window_start", "market_id", "seconds_elapsed"])


def _common_evaluation_frame(frame: pl.DataFrame, config: FrozenTrainingConfig) -> pl.DataFrame:
    evaluation = frame.filter(
        (pl.col("window_start") >= config.windows.calibration_end)
        & (pl.col("window_start") < config.windows.evaluation_end)
    ).drop_nulls(
        [
            "twap_label_up",
            *CHAINLINK_REFPRICE_FEATURES,
        ]
    )
    if evaluation.is_empty():
        raise RuntimeError("common August 23-24 TWAP evaluation cohort is empty")
    return evaluation.with_columns(pl.col("twap_label_up").cast(pl.Int8).alias("target_label"))


def _arm_frame(frame: pl.DataFrame, config: FrozenTrainingConfig, arm: str) -> pl.DataFrame:
    if arm not in ARMS:
        raise ValueError(f"unknown descendant arm: {arm}")
    selected = frame.filter(
        (pl.col("window_start") >= config.windows.data_start)
        & (pl.col("window_start") < config.windows.evaluation_end)
    ).drop_nulls(CHAINLINK_REFPRICE_FEATURES)
    target = (
        pl.col("label_up")
        if arm == "R"
        else pl.when(pl.col("window_start") < config.windows.twap_fit_start)
        .then(pl.col("label_up"))
        .otherwise(pl.col("twap_label_up"))
    )
    selected = selected.with_columns(target.cast(pl.Int8).alias("target_label")).drop_nulls(
        "target_label"
    )
    selected = selected.with_columns(
        pl.col("target_label").alias("label_up"),
        pl.when(pl.col("seconds_elapsed") < 90)
        .then(pl.lit("early"))
        .when(pl.col("seconds_elapsed") < 180)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("late"))
        .alias("time_band"),
        pl.when(pl.col("seconds_elapsed") < 120)
        .then(pl.lit("90-119"))
        .when(pl.col("seconds_elapsed") < 150)
        .then(pl.lit("120-149"))
        .otherwise(pl.lit("150-179"))
        .alias("middle_cell"),
        pl.col("pm_yes_cost_per_share").alias("up_cost_5"),
        pl.col("pm_no_cost_per_share").alias("down_cost_5"),
    ).with_columns(
        (
            pl.col("label_up").cast(pl.Float64)
            - pl.col("up_cost_5")
            - config.execution.stress_slippage_per_share
        ).alias("up_stress_reward"),
        (
            (1 - pl.col("label_up")).cast(pl.Float64)
            - pl.col("down_cost_5")
            - config.execution.stress_slippage_per_share
        ).alias("down_stress_reward"),
    )
    if selected.is_empty():
        raise RuntimeError(f"arm {arm} has no eligible rows")
    return selected


def _train_arm(
    frame: pl.DataFrame, config: FrozenTrainingConfig, arm: str, *, seed: int
) -> dict[str, Any]:
    arm_frame = _arm_frame(frame, config, arm)
    descendant_features = (
        *PRIMARY_FEATURES,
        *CHAINLINK_REFPRICE_FEATURES,
        *(TWAP_INPUT_FEATURES if arm == "RT" else ()),
    )
    exogenous_features = tuple(
        dict.fromkeys(
            (
                *CORE_FEATURES,
                *ORACLE_FEATURES,
                *CHAINLINK_REFPRICE_FEATURES,
                *(TWAP_INPUT_FEATURES if arm == "RT" else ()),
            )
        )
    )
    q5_model = _train_q5_lineage(arm_frame, config, descendant_features, seed=seed)
    middle_models = _train_middle_lineage(arm_frame, config, descendant_features, seed=seed + 1_000)
    specialist = _train_specialist_lineage(arm_frame, config, exogenous_features, seed=seed + 2_000)
    return {
        "q5": q5_model,
        **middle_models,
        "specialist_distilled": specialist,
    }


def _q5_config(config: FrozenTrainingConfig, *, seed: int) -> q5_lineage.TrainingConfig:
    base = q5_lineage.load_config(
        config.package_root
        / "configs"
        / "btc-5m-continuous-edge-payoff-aware-20260525-20260723.toml"
    )
    probability_fit_end = config.windows.outcome_fit_end - timedelta(days=2)
    windows = q5_lineage.WindowConfig(
        outcome_fit_start=config.windows.data_start,
        outcome_fit_end=probability_fit_end,
        calibration_end=config.windows.outcome_fit_end,
        admission_end=config.windows.admission_end,
        validation_start=config.windows.admission_end,
        validation_end=config.windows.calibration_end,
        test_end=config.windows.evaluation_end,
    )
    return replace(base, random_seed=seed, windows=windows)


def _train_q5_lineage(
    frame: pl.DataFrame,
    config: FrozenTrainingConfig,
    feature_names: tuple[str, ...],
    *,
    seed: int,
) -> dict[str, Any]:
    lineage_config = _q5_config(config, seed=seed)
    experts, expert_metrics = q5_lineage.fit_candidate(
        lineage_config,
        frame,
        "refprice_twap_descendant",
        feature_names,
    )
    family_proxies, proxy_metrics = q5_lineage.fit_family_proxies(lineage_config, frame)
    calibration_raw = q5_lineage.score_frame(
        q5_lineage._block(
            frame,
            lineage_config.windows.outcome_fit_end,
            lineage_config.windows.calibration_end,
        ),
        experts,
        family_proxies,
        lineage_config,
    )
    price_time = q5_lineage.fit_price_time_calibration(calibration_raw, lineage_config)
    calibrated = q5_lineage.attach_price_time_calibration(
        calibration_raw, price_time, lineage_config
    )
    guard = q5_lineage.fit_calibration_guard(calibrated, lineage_config)
    admission_frame = q5_lineage.prepare_scored_frame(
        q5_lineage._block(
            frame,
            lineage_config.windows.calibration_end,
            lineage_config.windows.admission_end,
        ),
        experts,
        family_proxies,
        price_time,
        guard,
        lineage_config,
    )
    admission_model = q5_lineage.fit_admission_model(admission_frame, lineage_config)
    policy = config.policies["q5"]
    thresholds = {
        "early": {
            "enabled": True,
            "confidence": policy["early_confidence"],
            "edge": policy["early_stress_edge"],
            "admission": policy["early_admission"],
            "payoff_lower_bound": policy["early_payoff_lower_bound"],
        },
        "mid": {"enabled": bool(policy["middle_enabled"])},
        "late": {
            "enabled": True,
            "confidence": policy["late_confidence"],
            "edge": policy["late_stress_edge"],
            "admission": policy["late_admission"],
            "payoff_lower_bound": policy["late_payoff_lower_bound"],
        },
    }
    return {
        "lineage": "continuous_edge_payoff_q5",
        "lineage_schema_version": q5_lineage.MODEL_SCHEMA_VERSION,
        "feature_names": feature_names,
        "experts": experts,
        "family_proxies": family_proxies,
        "price_time_calibration": price_time,
        "calibration_guard": guard,
        "admission_model": admission_model,
        "thresholds": thresholds,
        "lineage_config": lineage_config,
        "training_diagnostics": {
            "experts": expert_metrics,
            "family_proxies": proxy_metrics,
            "admission_oof": admission_model.oof_diagnostics,
        },
    }


def _middle_config(config: FrozenTrainingConfig, *, seed: int) -> middle_lineage.TournamentConfig:
    base = middle_lineage.load_config(
        config.package_root / "configs" / "btc-5m-middle-market-ablation-tournament-20260525.toml"
    )
    windows = middle_lineage.Windows(
        fit_start=config.windows.data_start,
        fit_end=config.windows.outcome_fit_end - timedelta(days=2),
        calibration_end=config.windows.outcome_fit_end,
        meta_end=config.windows.admission_end,
        policy_end=config.windows.calibration_end,
        holdout_end=config.windows.evaluation_end,
    )
    return replace(base, random_seed=seed, windows=windows)


def _train_middle_lineage(
    frame: pl.DataFrame,
    config: FrozenTrainingConfig,
    feature_names: tuple[str, ...],
    *,
    seed: int,
) -> dict[str, Any]:
    lineage_config = _middle_config(config, seed=seed)
    middle_frame = frame.filter(
        (pl.col("seconds_elapsed") >= 90) & (pl.col("seconds_elapsed") < 180)
    )
    fit = middle_lineage._block(
        middle_frame, lineage_config.windows.fit_start, lineage_config.windows.fit_end
    )
    calibration = middle_lineage._block(
        middle_frame,
        lineage_config.windows.fit_end,
        lineage_config.windows.calibration_end,
    )
    meta = middle_lineage._block(
        middle_frame,
        lineage_config.windows.calibration_end,
        lineage_config.windows.meta_end,
    )
    outcome = middle_lineage._fit_shared_outcome(
        fit, calibration, feature_names, lineage_config, seed=seed
    )
    oof = middle_lineage._oof_predictions(fit, feature_names, lineage_config)
    base_calibration = middle_lineage._decision_frame(calibration, outcome, lineage_config, None)
    base_meta = middle_lineage._decision_frame(meta, outcome, lineage_config, None)
    stratified = middle_lineage._fit_correctness(
        base_calibration,
        lineage_config,
        stratified=True,
        regime=False,
        seed=seed + 21,
    )
    regime = middle_lineage._fit_correctness(
        base_calibration,
        lineage_config,
        stratified=True,
        regime=True,
        seed=seed + 22,
    )
    oi_oof = oof.drop_nulls(BINANCE_OI_FEATURES)
    oi_calibration = calibration.drop_nulls(BINANCE_OI_FEATURES)
    modifier = middle_lineage._fit_probability_modifier(oi_oof, lineage_config)
    modified_calibration = middle_lineage._decision_frame(
        oi_calibration, outcome, lineage_config, modifier
    )
    full_correctness = middle_lineage._fit_correctness(
        modified_calibration,
        lineage_config,
        stratified=True,
        regime=True,
        seed=seed + 23,
    )
    loss_training = pl.concat(
        (
            middle_lineage._loss_training_frame(oof, lineage_config),
            middle_lineage._loss_training_frame(base_meta, lineage_config),
        ),
        how="diagonal_relaxed",
    )
    loss_features = middle_lineage._variable_features(loss_training, middle_lineage.LOSS_FEATURES)
    loss_model = middle_lineage._fit_regressor(
        loss_training,
        loss_features,
        "loss_severity_target",
        lineage_config,
        seed=seed + 31,
    )
    common_eligibility = tuple(CHAINLINK_REFPRICE_FEATURES)
    models = {
        "stratified_payoff": middle_lineage.Candidate(
            "refprice_stratified_payoff",
            feature_names,
            common_eligibility,
            outcome,
            stratified,
        ),
        "regime_calibrated": middle_lineage.Candidate(
            "refprice_regime_calibrated",
            feature_names,
            common_eligibility,
            outcome,
            regime,
        ),
        "full_combined": middle_lineage.Candidate(
            "refprice_full_combined",
            feature_names,
            (*common_eligibility, *BINANCE_OI_FEATURES),
            outcome,
            full_correctness,
            probability_modifier=modifier,
            loss_model=loss_model,
            loss_feature_names=loss_features,
        ),
    }
    return {
        name: {
            "lineage": "middle_market_ablation",
            "lineage_schema_version": middle_lineage.MODEL_SCHEMA_VERSION,
            "candidate": candidate,
            "lineage_config": lineage_config,
            "oof_rows": oof.height,
            "oof_markets": oof["market_id"].n_unique(),
        }
        for name, candidate in models.items()
    }


def _specialist_config(config: FrozenTrainingConfig) -> specialist_lineage.TournamentConfig:
    return specialist_lineage.load_config(
        config.package_root
        / "configs"
        / "btc-5m-fair-value-challenger-tournament-20260525-20260802.toml"
    )


def _train_specialist_lineage(
    frame: pl.DataFrame,
    config: FrozenTrainingConfig,
    feature_names: tuple[str, ...],
    *,
    seed: int,
) -> dict[str, Any]:
    lineage_config = _specialist_config(config)
    fit = frame.filter(pl.col("window_start") < config.windows.outcome_fit_end)
    calibration = frame.filter(
        (pl.col("window_start") >= config.windows.outcome_fit_end)
        & (pl.col("window_start") < config.windows.calibration_end)
    )
    fair = specialist_lineage._fit_fair_models(
        fit,
        calibration,
        lineage_config,
        seed=seed,
        feature_names=feature_names,
    )
    return {
        "lineage": "specialist_distilled_fair_value",
        "lineage_schema_version": specialist_lineage.ARTIFACT_SCHEMA_VERSION,
        "fair_models": fair,
        "lineage_config": lineage_config,
    }


def _score_family(
    frame: pl.DataFrame,
    model: dict[str, Any],
    family: str,
    config: FrozenTrainingConfig,
) -> pl.DataFrame:
    prepared = _with_lineage_columns(frame, config)
    if family == "q5":
        lineage_config = model["lineage_config"]
        lineage_scored = q5_lineage.attach_admission_probability(
            q5_lineage.prepare_scored_frame(
                prepared,
                model["experts"],
                model["family_proxies"],
                model["price_time_calibration"],
                model["calibration_guard"],
                lineage_config,
            ),
            model["admission_model"],
        )
        scored = _attach_action_columns(
            prepared, lineage_scored["probability_up"].to_numpy(), config
        )
        return scored.with_columns(
            pl.Series(
                "model_confidence",
                lineage_scored["conservative_probability_selected"].to_numpy(),
            ),
            pl.Series("model_stress_edge", lineage_scored["conservative_edge_5"].to_numpy()),
            pl.Series(
                "admission_probability",
                lineage_scored["admission_probability"].to_numpy(),
            ),
            pl.Series(
                "payoff_expected_stress_edge",
                lineage_scored["payoff_stress_edge_lower_bound"].to_numpy(),
            ),
            pl.Series(
                "price_bucket_minimum_edge",
                lineage_scored["price_bucket_minimum_edge"].to_numpy(),
            ),
        )
    if family in ("stratified_payoff", "regime_calibrated", "full_combined"):
        candidate = model["candidate"]
        lineage_scored = middle_lineage._score_candidate(
            prepared, candidate, model["lineage_config"]
        )
        scored = _attach_action_columns(
            lineage_scored,
            lineage_scored["probability_up"].to_numpy(),
            config,
        ).with_columns(
            pl.Series(
                "admission_probability",
                lineage_scored["lower_correctness_probability"].to_numpy(),
            ),
            pl.Series(
                "model_stress_edge",
                lineage_scored["stress_edge_lower_bound"].to_numpy(),
            ),
            pl.Series(
                "payoff_expected_stress_edge",
                lineage_scored["stress_edge_lower_bound"].to_numpy(),
            ),
        )
        if family == "full_combined":
            scored = scored.with_columns(
                pl.Series(
                    "predicted_loss_severity",
                    lineage_scored["loss_severity_prediction"].to_numpy(),
                )
            )
        return scored
    fair = model["fair_models"]
    probability = np.clip(
        fair["distilled"].predict(q5_lineage._matrix(prepared, fair["feature_names"])),
        1e-6,
        1 - 1e-6,
    )
    return _attach_action_columns(prepared, probability, config)


def _with_lineage_columns(frame: pl.DataFrame, config: FrozenTrainingConfig) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("target_label").cast(pl.Int8).alias("label_up"),
        pl.when(pl.col("seconds_elapsed") < 90)
        .then(pl.lit("early"))
        .when(pl.col("seconds_elapsed") < 180)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("late"))
        .alias("time_band"),
        pl.when(pl.col("seconds_elapsed") < 120)
        .then(pl.lit("90-119"))
        .when(pl.col("seconds_elapsed") < 150)
        .then(pl.lit("120-149"))
        .otherwise(pl.lit("150-179"))
        .alias("middle_cell"),
        pl.col("pm_yes_cost_per_share").alias("up_cost_5"),
        pl.col("pm_no_cost_per_share").alias("down_cost_5"),
    ).with_columns(
        (
            pl.col("label_up").cast(pl.Float64)
            - pl.col("up_cost_5")
            - config.execution.stress_slippage_per_share
        ).alias("up_stress_reward"),
        (
            (1 - pl.col("label_up")).cast(pl.Float64)
            - pl.col("down_cost_5")
            - config.execution.stress_slippage_per_share
        ).alias("down_stress_reward"),
    )


def _attach_action_columns(
    frame: pl.DataFrame, probability: np.ndarray, config: FrozenTrainingConfig
) -> pl.DataFrame:
    predicted_up = probability >= 0.5
    confidence = np.where(predicted_up, probability, 1.0 - probability)
    up_price = frame["up_ask_vwap_5"].to_numpy()
    down_price = frame["down_ask_vwap_5"].to_numpy()
    fee = frame["fee_rate"].to_numpy()
    up_cost = (
        up_price + fee * up_price * (1.0 - up_price) + config.execution.execution_reserve_per_share
    )
    down_cost = (
        down_price
        + fee * down_price * (1.0 - down_price)
        + config.execution.execution_reserve_per_share
    )
    cost = np.where(predicted_up, up_cost, down_cost)
    correct = predicted_up == frame["target_label"].to_numpy().astype(bool)
    reward = correct.astype(float) - cost
    stress = reward - config.execution.stress_slippage_per_share
    return frame.with_columns(
        pl.Series("model_probability_up", probability),
        pl.Series("predicted_up", predicted_up),
        pl.Series("model_confidence", confidence),
        pl.Series("selected_cost_5", cost),
        pl.Series(
            "model_stress_edge",
            confidence - cost - config.execution.stress_slippage_per_share,
        ),
        pl.Series("direction_correct", correct),
        pl.Series("reward_per_share", reward),
        pl.Series("stress_reward_per_share", stress),
        pl.Series("loss_severity", np.where(correct, 0.0, cost)),
    )


def _apply_frozen_policy(
    frame: pl.DataFrame, family: str, config: FrozenTrainingConfig
) -> pl.DataFrame:
    if family == "q5":
        policy = config.policies["q5"]
        early = (
            (pl.col("seconds_elapsed") < 90)
            & (pl.col("model_confidence") >= policy["early_confidence"])
            & (pl.col("model_stress_edge") >= policy["early_stress_edge"])
            & (pl.col("admission_probability") >= policy["early_admission"])
            & (pl.col("payoff_expected_stress_edge") >= policy["early_payoff_lower_bound"])
            & (pl.col("model_stress_edge") >= pl.col("price_bucket_minimum_edge"))
        )
        late = (
            (pl.col("seconds_elapsed") >= 180)
            & (pl.col("model_confidence") >= policy["late_confidence"])
            & (pl.col("model_stress_edge") >= policy["late_stress_edge"])
            & (pl.col("admission_probability") >= policy["late_admission"])
            & (pl.col("payoff_expected_stress_edge") >= policy["late_payoff_lower_bound"])
            & (pl.col("model_stress_edge") >= pl.col("price_bucket_minimum_edge"))
        )
        eligible = early | late
    elif family in ("stratified_payoff", "regime_calibrated", "full_combined"):
        policy = config.policies[family]
        eligible = (
            pl.col("seconds_elapsed").is_between(
                policy["minimum_second"], policy["maximum_second"], closed="both"
            )
            & (pl.col("admission_probability") >= policy["confidence"])
            & (pl.col("model_stress_edge") >= policy["stress_edge"])
        )
        if family == "full_combined":
            eligible &= pl.col("predicted_loss_severity") <= policy["maximum_loss_severity"]
    else:
        eligible = pl.lit(False)
        for values in config.policies["specialist_distilled"]["cells"].values():
            eligible |= (
                pl.col("seconds_elapsed").is_between(
                    values["minimum_second"], values["maximum_second"], closed="both"
                )
                & (pl.col("model_confidence") >= values["confidence"])
                & (pl.col("model_stress_edge") >= values["stress_edge"])
            )
    return (
        frame.filter(eligible)
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .unique(subset=["market_id"], keep="first", maintain_order=True)
        .sort(["window_start", "market_id"])
    )


def _evaluation_metrics(
    scored: pl.DataFrame,
    selected: pl.DataFrame,
    evaluation: pl.DataFrame,
    config: FrozenTrainingConfig,
) -> dict[str, Any]:
    probability = scored["model_probability_up"].to_numpy()
    label = scored["target_label"].to_numpy()
    quantity = config.execution.evaluation_quantity
    trades = selected.height
    pnl = selected["reward_per_share"].sum() * quantity if trades else 0.0
    stress_pnl = selected["stress_reward_per_share"].sum() * quantity if trades else 0.0
    wins = int(selected["direction_correct"].sum()) if trades else 0
    directions: dict[str, Any] = {}
    for name, value in (("UP", True), ("DOWN", False)):
        subset = selected.filter(pl.col("predicted_up") == value)
        directions[name] = {
            "trades": subset.height,
            "wins": int(subset["direction_correct"].sum()) if subset.height else 0,
            "losses": subset.height - int(subset["direction_correct"].sum())
            if subset.height
            else 0,
            "net_pnl": subset["reward_per_share"].sum() * quantity if subset.height else 0.0,
        }
    capacity: dict[str, Any] = {}
    for capacity_quantity in config.execution.quantities:
        if trades:
            selected_price = np.where(
                selected["predicted_up"].to_numpy(),
                selected[f"up_ask_vwap_{capacity_quantity}"].to_numpy(),
                selected[f"down_ask_vwap_{capacity_quantity}"].to_numpy(),
            )
            fee = selected["fee_rate"].to_numpy() * selected_price * (1.0 - selected_price)
            cost = selected_price + fee + config.execution.execution_reserve_per_share
            realized = selected["direction_correct"].to_numpy().astype(float) - cost
            capacity[str(capacity_quantity)] = {
                "trades": trades,
                "net_pnl": float(realized.sum() * capacity_quantity),
                "stress_net_pnl": float(
                    (realized - config.execution.stress_slippage_per_share).sum()
                    * capacity_quantity
                ),
                "expectancy_per_trade": float(realized.mean() * capacity_quantity),
            }
        else:
            capacity[str(capacity_quantity)] = {
                "trades": 0,
                "net_pnl": 0.0,
                "stress_net_pnl": 0.0,
                "expectancy_per_trade": None,
            }
    entry_cells = {}
    for name, start, end in _entry_cells():
        subset = selected.filter(
            (pl.col("seconds_elapsed") >= start) & (pl.col("seconds_elapsed") < end)
        )
        entry_cells[name] = {
            "trades": subset.height,
            "wins": int(subset["direction_correct"].sum()) if subset.height else 0,
            "net_pnl": float(subset["reward_per_share"].sum() * quantity) if subset.height else 0.0,
        }
    price_buckets = {}
    for lower, upper in pairwise(PRICE_BUCKET_EDGES):
        subset = selected.filter(
            (pl.col("selected_cost_5") >= lower) & (pl.col("selected_cost_5") < upper)
        )
        price_buckets[f"{lower:.2f}-{upper:.2f}"] = {
            "trades": subset.height,
            "wins": int(subset["direction_correct"].sum()) if subset.height else 0,
            "net_pnl": float(subset["reward_per_share"].sum() * quantity) if subset.height else 0.0,
        }
    evaluation_markets = evaluation["market_id"].n_unique()
    prediction_markets = scored["market_id"].n_unique()
    traded_markets = selected["market_id"].n_unique() if trades else 0
    return {
        "evaluation_rows": scored.height,
        "evaluation_markets": evaluation_markets,
        "prediction_markets": prediction_markets,
        "prediction_market_coverage": prediction_markets / evaluation_markets,
        "settled": trades,
        "wins": wins,
        "losses": trades - wins,
        "accuracy": wins / trades if trades else None,
        "market_coverage": traded_markets / evaluation_markets,
        "no_trade_markets": evaluation_markets - traded_markets,
        "net_pnl": float(pnl),
        "stress_net_pnl": float(stress_pnl),
        "expectancy_per_trade": float(pnl / trades) if trades else None,
        "brier": float(np.mean((probability - label) ** 2)),
        "log_loss": float(log_loss(label, probability, labels=[0, 1])),
        "ece_10": _ece(label, probability, 10),
        "average_probability_up": float(probability.mean()),
        "directions": directions,
        "entry_cells": entry_cells,
        "entry_price_buckets": price_buckets,
        "vwap_capacity": capacity,
    }


def _entry_cells() -> tuple[tuple[str, int, int], ...]:
    return (
        ("early_15_89", 15, 90),
        ("middle_90_119", 90, 120),
        ("middle_120_149", 120, 150),
        ("middle_150_179", 150, 180),
        ("late_180_240", 180, 241),
    )


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    value = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if mask.any():
            value += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(value)


def _attribution(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in FAMILIES:
        output[family] = {
            "existing_to_watermark_refresh": "incumbent remains fixed external benchmark",
            "W_to_R": _metric_delta(results["W"][family], results["R"][family]),
            "R_to_T": _metric_delta(results["R"][family], results["T"][family]),
            "W_to_T": _metric_delta(results["W"][family], results["T"][family]),
        }
    return output


def _metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (right[key] - left[key])
        for key in ("net_pnl", "stress_net_pnl", "brier", "log_loss", "ece_10")
        if left[key] is not None and right[key] is not None
    }


def _incumbent_identities(config: FrozenTrainingConfig) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family, path in config.incumbents.items():
        manifest = json.loads(path.read_text())
        output[family] = {
            "model_key": manifest["model_key"],
            "model_sha256": manifest["model_sha256"],
            "manifest_sha256": file_sha256(path),
            "unchanged": True,
        }
    return output


def _render_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# RefPrice-primary / TWAP-60 target training",
        "",
        f"Run: `{metrics['run_id']}`",
        f"Source commit: `{metrics['source_commit']}`",
        (
            f"Common evaluation: {metrics['common_evaluation']['markets']:,} markets, "
            f"{metrics['common_evaluation']['rows']:,} decision rows."
        ),
        "",
        "| Arm | Family | Predicted markets | Traded markets | W-L | Net P&L | Stress P&L | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for family in FAMILIES:
            row = metrics["results"][arm][family]
            lines.append(
                f"| {arm} | {family} | {row['prediction_markets']} | {row['settled']} | "
                f"{row['wins']}-{row['losses']} | "
                f"{row['net_pnl']:.2f} | {row['stress_net_pnl']:.2f} | "
                f"{row['brier']:.4f} | {row['ece_10']:.4f} |"
            )
    lines.extend(
        [
            "",
            "R is the RefPrice-primary legacy-label refresh. T changes post-August-1 labels to PMData TWAP-60. RT adds causal PMData TWAP-30/60 inputs to T.",
            "All incumbent artifacts, trading processes, runtime bundles, and deployed images remained unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _git_revision(package_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=package_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _path(package_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (package_root / path).resolve()


def _utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include UTC timezone")
    return parsed.astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser(prog="btc-refprice-twap-training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()
    run_dir, metrics = run_frozen_training(load_config(args.config), force_data=args.force_data)
    print(f"results: {run_dir}")
    print(f"common evaluation markets: {metrics['common_evaluation']['markets']:,}")


if __name__ == "__main__":
    main()
