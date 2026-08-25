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
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import psycopg
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .chainlink_oi_features import (
    BINANCE_OI_FEATURES,
    CHAINLINK_CANDLE_FEATURES,
    CHAINLINK_REFPRICE_FEATURES,
    _attach_candle_features,
    _attach_open_interest_features,
    _attach_refprice_features,
)
from .continuous_edge_training import (
    BOOK_RAW_FEATURES,
    CHAINLINK_FEATURES,
    CORE_FEATURES,
    ORACLE_FEATURES,
    PRIMARY_FEATURES,
    VWAP_QUANTITIES,
    attach_book_features,
    market_equal_weights,
)
from .core_extract import configure_read_only_connection, database_connection, file_sha256
from .core_features import (
    attach_causal_oracle_rounds,
    derive_core_point_in_time_features,
    derive_oracle_point_in_time_features,
)

SCHEMA_VERSION = "btc-refprice-twap-target-training-v1"
DATASET_SCHEMA_VERSION = "btc-refprice-twap-target-dataset-v1"
ARTIFACT_SCHEMA_VERSION = "btc-refprice-twap-target-model-v1"
ARMS = ("W", "R", "T")
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
    "btc-refprice-twap-candle-source.sql",
    "btc-refprice-twap-open-interest-source.sql",
    "btc-refprice-twap-label-source.sql",
)
JOIN_KEYS = ("market_id", "window_start", "observed_at", "seconds_elapsed")
PRICE_BUCKET_EDGES = (0.0, 0.65, 0.75, 0.85, 1.01)
REGIME_FEATURES = (
    "seconds_elapsed_scaled",
    "btc_path_from_window_open_bps",
    "btc_cross_venue_boundary_gap_bps",
    "btc_path_terminal_volatility_z",
    "btc_boundary_terminal_volatility_z",
    "btc_volatility_shock_30_vs_120",
    "hour_sin",
    "hour_cos",
)
META_FEATURES = (
    "model_probability_up",
    "model_confidence",
    "model_stress_edge",
    "selected_cost_5",
    "seconds_elapsed_scaled",
    "btc_path_from_window_open_bps",
    "btc_cross_venue_boundary_gap_bps",
    "btc_path_terminal_volatility_z",
    "btc_boundary_terminal_volatility_z",
    "btc_volatility_shock_30_vs_120",
    "pm_vwap5_overround",
    "pm_vwap200_overround",
    "pm_depth_imbalance",
    "pm_up_book_age_seconds",
    "pm_down_book_age_seconds",
)
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
    paths = PathConfig(
        **{key: _path(package_root, value) for key, value in raw["paths"].items()}
    )
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
        incumbents={
            key: _path(package_root, value) for key, value in raw["incumbents"].items()
        },
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
                    "target": "twap_60" if arm == "T" else "official_outcome",
                    "refprice_primary": arm in ("R", "T"),
                    "runtime_exported": False,
                    "production_qualified": False,
                    "model": model,
                },
                artifact_path,
                compress=3,
            )
            selected.write_parquet(arm_dir / f"{family}-evaluation-ledger.parquet", compression="zstd")
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
            "W and R are diagnostic controls; T is the TWAP-60 target generation.",
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
            pl.col("oracle_model_eligible")
            .fill_null(False)
            .alias("early_oracle_eligible")
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

    candles = _query_frame(
        connection,
        (sql_root / SQL_FILES[4]).read_text(),
        {"range_start": start, "range_end": end},
        f"ref_twap_candles_{start:%Y%m%d}",
    )
    candle_features = (
        _attach_candle_features(
            external_core,
            candles,
            max_age_seconds=60,
        ).select(*JOIN_KEYS, *CHAINLINK_CANDLE_FEATURES)
        if candles.height
        else _empty_feature_frame(external_core, CHAINLINK_CANDLE_FEATURES)
    )
    frame = frame.join(candle_features, on=list(JOIN_KEYS), how="left", validate="1:1")

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
        frame = frame.join(
            labels, on=["market_id", "window_start"], how="left", validate="m:1"
        )
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
        "candle_eligible_rows": frame.drop_nulls(CHAINLINK_CANDLE_FEATURES).height,
        "oi_eligible_rows": frame.drop_nulls(BINANCE_OI_FEATURES).height,
        "twap_labeled_rows": frame.drop_nulls(["twap_label_up"]).height,
        "twap_labeled_markets": frame.drop_nulls(["twap_label_up"])["market_id"].n_unique(),
    }
    return frame, audit


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


def _empty_feature_frame(
    frame: pl.DataFrame, feature_names: tuple[str, ...]
) -> pl.DataFrame:
    return frame.head(0).select(*JOIN_KEYS).with_columns(
        *[pl.lit(None, dtype=pl.Float64).alias(name) for name in feature_names]
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


def _common_evaluation_frame(
    frame: pl.DataFrame, config: FrozenTrainingConfig
) -> pl.DataFrame:
    evaluation = frame.filter(
        (pl.col("window_start") >= config.windows.calibration_end)
        & (pl.col("window_start") < config.windows.evaluation_end)
    ).drop_nulls(
        [
            "twap_label_up",
            *CHAINLINK_REFPRICE_FEATURES,
            *BINANCE_OI_FEATURES,
        ]
    )
    if evaluation.is_empty():
        raise RuntimeError("common August 23-24 TWAP evaluation cohort is empty")
    return evaluation.with_columns(pl.col("twap_label_up").cast(pl.Int8).alias("target_label"))


def _arm_frame(frame: pl.DataFrame, config: FrozenTrainingConfig, arm: str) -> pl.DataFrame:
    start = config.windows.twap_fit_start if arm == "T" else config.windows.data_start
    selected = frame.filter(
        (pl.col("window_start") >= start)
        & (pl.col("window_start") < config.windows.evaluation_end)
    )
    if arm in ("R", "T"):
        selected = selected.drop_nulls(CHAINLINK_REFPRICE_FEATURES)
    label = "twap_label_up" if arm == "T" else "label_up"
    selected = selected.drop_nulls([label]).with_columns(
        pl.col(label).cast(pl.Int8).alias("target_label")
    )
    if selected.is_empty():
        raise RuntimeError(f"arm {arm} has no eligible rows")
    return selected


def _train_arm(
    frame: pl.DataFrame, config: FrozenTrainingConfig, arm: str, *, seed: int
) -> dict[str, Any]:
    arm_frame = _arm_frame(frame, config, arm)
    fit = arm_frame.filter(pl.col("window_start") < config.windows.outcome_fit_end)
    admission = arm_frame.filter(
        (pl.col("window_start") >= config.windows.outcome_fit_end)
        & (pl.col("window_start") < config.windows.admission_end)
    )
    calibration = arm_frame.filter(
        (pl.col("window_start") >= config.windows.admission_end)
        & (pl.col("window_start") < config.windows.calibration_end)
    )
    if min(fit["market_id"].n_unique(), admission["market_id"].n_unique(), calibration["market_id"].n_unique()) == 0:
        raise RuntimeError(f"arm {arm} contains an empty chronological training block")

    ref_features = CHAINLINK_REFPRICE_FEATURES if arm in ("R", "T") else ()
    direction_features = tuple(dict.fromkeys((*PRIMARY_FEATURES, *CHAINLINK_FEATURES, *ref_features)))
    full_features = (*direction_features, *BINANCE_OI_FEATURES)
    exogenous_features = tuple(
        dict.fromkeys((*CORE_FEATURES, *ORACLE_FEATURES, *CHAINLINK_FEATURES, *ref_features))
    )

    base_outcome = _fit_outcome(fit, direction_features, config, seed=seed)
    full_fit = fit.drop_nulls(BINANCE_OI_FEATURES)
    full_outcome = _fit_outcome(full_fit, full_features, config, seed=seed + 100)

    q5 = _fit_family_model(
        base_outcome, admission, calibration, "global", config, seed=seed + 200
    )
    stratified = _fit_family_model(
        base_outcome, admission, calibration, "stratified", config, seed=seed + 300
    )
    regime = _fit_family_model(
        base_outcome, admission, calibration, "regime", config, seed=seed + 400
    )
    full_admission = admission.drop_nulls(BINANCE_OI_FEATURES)
    full_calibration = calibration.drop_nulls(BINANCE_OI_FEATURES)
    full = _fit_family_model(
        full_outcome,
        full_admission,
        full_calibration,
        "regime",
        config,
        seed=seed + 500,
        fit_loss=True,
    )
    models = {
        "q5": q5,
        "stratified_payoff": stratified,
        "regime_calibrated": regime,
        "full_combined": full,
    }
    models["specialist_distilled"] = _fit_specialist(
        calibration,
        models,
        exogenous_features,
        config,
        seed=seed + 600,
    )
    for model in models.values():
        model["arm"] = arm
        model["training_rows"] = fit.height
        model["training_markets"] = fit["market_id"].n_unique()
        model["admission_rows"] = admission.height
        model["calibration_rows"] = calibration.height
    return models


def _fit_outcome(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    config: FrozenTrainingConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    medians = _feature_medians(frame, features)
    estimator = HistGradientBoostingClassifier(
        learning_rate=config.model.learning_rate,
        max_iter=config.model.max_iter,
        max_leaf_nodes=config.model.max_leaf_nodes,
        min_samples_leaf=config.model.min_samples_leaf,
        l2_regularization=config.model.l2_regularization,
        random_state=seed,
        early_stopping=False,
    )
    estimator.fit(
        _matrix(frame, features, medians),
        frame["target_label"].to_numpy(),
        sample_weight=market_equal_weights(frame),
    )
    return {"estimator": estimator, "features": features, "medians": medians}


def _fit_family_model(
    outcome: dict[str, Any],
    admission: pl.DataFrame,
    calibration: pl.DataFrame,
    calibration_kind: str,
    config: FrozenTrainingConfig,
    *,
    seed: int,
    fit_loss: bool = False,
) -> dict[str, Any]:
    raw_calibration = _raw_probability(outcome, calibration)
    calibrator = _fit_probability_calibrator(
        calibration, raw_calibration, calibration_kind, config, seed=seed
    )
    raw_admission = _raw_probability(outcome, admission)
    admission_scored = _attach_action_columns(admission, raw_admission, config)
    meta_medians = _feature_medians(admission_scored, META_FEATURES)
    correctness = HistGradientBoostingClassifier(
        learning_rate=config.model.learning_rate,
        max_iter=config.model.loss_max_iter,
        max_leaf_nodes=config.model.loss_max_leaf_nodes,
        min_samples_leaf=config.model.loss_min_samples_leaf,
        l2_regularization=config.model.loss_l2_regularization,
        random_state=seed + 1,
        early_stopping=False,
    )
    correctness.fit(
        _matrix(admission_scored, META_FEATURES, meta_medians),
        admission_scored["direction_correct"].to_numpy().astype(np.int8),
        sample_weight=market_equal_weights(admission_scored),
    )
    payoff = HistGradientBoostingRegressor(
        learning_rate=config.model.learning_rate,
        max_iter=config.model.loss_max_iter,
        max_leaf_nodes=config.model.loss_max_leaf_nodes,
        min_samples_leaf=config.model.loss_min_samples_leaf,
        l2_regularization=config.model.loss_l2_regularization,
        random_state=seed + 2,
        early_stopping=False,
    )
    payoff.fit(
        _matrix(admission_scored, META_FEATURES, meta_medians),
        admission_scored["stress_reward_per_share"].to_numpy(),
        sample_weight=market_equal_weights(admission_scored),
    )
    loss_model = None
    if fit_loss:
        loss_model = HistGradientBoostingRegressor(
            learning_rate=config.model.learning_rate,
            max_iter=config.model.loss_max_iter,
            max_leaf_nodes=config.model.loss_max_leaf_nodes,
            min_samples_leaf=config.model.loss_min_samples_leaf,
            l2_regularization=config.model.loss_l2_regularization,
            random_state=seed + 3,
            early_stopping=False,
        )
        loss_model.fit(
            _matrix(admission_scored, META_FEATURES, meta_medians),
            admission_scored["loss_severity"].to_numpy(),
            sample_weight=market_equal_weights(admission_scored),
        )
    return {
        "kind": calibration_kind,
        "outcome": outcome,
        "calibrator": calibrator,
        "meta_features": META_FEATURES,
        "meta_medians": meta_medians,
        "correctness": correctness,
        "payoff": payoff,
        "loss": loss_model,
    }


def _fit_probability_calibrator(
    frame: pl.DataFrame,
    raw: np.ndarray,
    kind: str,
    config: FrozenTrainingConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    global_model = LogisticRegression(
        C=config.model.calibration_c,
        max_iter=config.model.calibration_max_iter,
        random_state=seed,
    )
    global_model.fit(
        _logit(raw).reshape(-1, 1),
        frame["target_label"].to_numpy(),
        sample_weight=market_equal_weights(frame),
    )
    result: dict[str, Any] = {"kind": kind, "global": global_model, "locals": {}}
    if kind == "stratified":
        for name, start, end in _entry_cells():
            mask = (frame["seconds_elapsed"].to_numpy() >= start) & (
                frame["seconds_elapsed"].to_numpy() < end
            )
            if mask.sum() < config.model.local_calibration_minimum_rows:
                continue
            subset = frame.filter(pl.Series(mask))
            if subset["target_label"].n_unique() < 2:
                continue
            local = LogisticRegression(
                C=config.model.calibration_c,
                max_iter=config.model.calibration_max_iter,
                random_state=seed + len(result["locals"]) + 1,
            )
            local.fit(
                _logit(raw[mask]).reshape(-1, 1),
                subset["target_label"].to_numpy(),
                sample_weight=market_equal_weights(subset),
            )
            markets = subset["market_id"].n_unique()
            result["locals"][name] = {
                "model": local,
                "weight": markets
                / (markets + config.model.local_calibration_shrinkage_rows),
            }
    elif kind == "regime":
        design, medians = _regime_matrix(frame, raw)
        regime = LogisticRegression(
            C=config.model.calibration_c,
            max_iter=config.model.calibration_max_iter,
            random_state=seed + 1,
        )
        regime.fit(
            design,
            frame["target_label"].to_numpy(),
            sample_weight=market_equal_weights(frame),
        )
        result["regime"] = regime
        result["regime_medians"] = medians
    return result


def _fit_specialist(
    calibration: pl.DataFrame,
    teachers: dict[str, Any],
    features: tuple[str, ...],
    config: FrozenTrainingConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    teacher_probabilities = np.column_stack(
        [
            _model_probability(teachers[name], calibration)
            for name in ("q5", "stratified_payoff", "regime_calibrated", "full_combined")
        ]
    )
    selector_features = np.column_stack(
        (
            _logit(teacher_probabilities),
            teacher_probabilities.std(axis=1),
            teacher_probabilities.max(axis=1) - teacher_probabilities.min(axis=1),
            calibration["seconds_elapsed_scaled"].to_numpy(),
            calibration["btc_cross_venue_boundary_gap_bps"].to_numpy(),
        )
    )
    selector = LogisticRegression(C=0.5, max_iter=2000, random_state=seed)
    selector.fit(
        np.nan_to_num(selector_features),
        calibration["target_label"].to_numpy(),
        sample_weight=market_equal_weights(calibration),
    )
    teacher = selector.predict_proba(np.nan_to_num(selector_features))[:, 1]
    medians = _feature_medians(calibration, features)
    distilled = HistGradientBoostingRegressor(
        learning_rate=config.model.learning_rate,
        max_iter=180,
        max_leaf_nodes=23,
        min_samples_leaf=120,
        l2_regularization=5.0,
        random_state=seed + 1,
        early_stopping=False,
    )
    distilled.fit(
        _matrix(calibration, features, medians),
        teacher,
        sample_weight=market_equal_weights(calibration),
    )
    raw = np.clip(distilled.predict(_matrix(calibration, features, medians)), 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(C=0.5, max_iter=2000, random_state=seed + 2)
    calibrator.fit(
        _logit(raw).reshape(-1, 1),
        calibration["target_label"].to_numpy(),
        sample_weight=market_equal_weights(calibration),
    )
    return {
        "kind": "specialist_distilled",
        "features": features,
        "medians": medians,
        "selector": selector,
        "distilled": distilled,
        "calibrator": calibrator,
        "teacher_families": (
            "q5",
            "stratified_payoff",
            "regime_calibrated",
            "full_combined",
        ),
    }


def _raw_probability(model: dict[str, Any], frame: pl.DataFrame) -> np.ndarray:
    outcome = model
    return np.clip(
        outcome["estimator"].predict_proba(
            _matrix(frame, outcome["features"], outcome["medians"])
        )[:, 1],
        1e-6,
        1 - 1e-6,
    )


def _model_probability(model: dict[str, Any], frame: pl.DataFrame) -> np.ndarray:
    if model["kind"] == "specialist_distilled":
        raw = np.clip(
            model["distilled"].predict(
                _matrix(frame, model["features"], model["medians"])
            ),
            1e-6,
            1 - 1e-6,
        )
        return model["calibrator"].predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
    raw = _raw_probability(model["outcome"], frame)
    calibration = model["calibrator"]
    global_probability = calibration["global"].predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
    if calibration["kind"] == "stratified":
        probability = global_probability.copy()
        seconds = frame["seconds_elapsed"].to_numpy()
        for name, start, end in _entry_cells():
            local = calibration["locals"].get(name)
            if local is None:
                continue
            mask = (seconds >= start) & (seconds < end)
            local_probability = local["model"].predict_proba(
                _logit(raw[mask]).reshape(-1, 1)
            )[:, 1]
            probability[mask] = (
                local["weight"] * local_probability
                + (1.0 - local["weight"]) * global_probability[mask]
            )
        return np.clip(probability, 1e-6, 1 - 1e-6)
    if calibration["kind"] == "regime":
        design, _ = _regime_matrix(frame, raw, calibration["regime_medians"])
        return np.clip(calibration["regime"].predict_proba(design)[:, 1], 1e-6, 1 - 1e-6)
    return np.clip(global_probability, 1e-6, 1 - 1e-6)


def _score_family(
    frame: pl.DataFrame,
    model: dict[str, Any],
    family: str,
    config: FrozenTrainingConfig,
) -> pl.DataFrame:
    probability = _model_probability(model, frame)
    scored = _attach_action_columns(frame, probability, config)
    if family == "specialist_distilled":
        return scored
    matrix = _matrix(scored, model["meta_features"], model["meta_medians"])
    scored = scored.with_columns(
        pl.Series(
            "admission_probability",
            model["correctness"].predict_proba(matrix)[:, 1],
        ),
        pl.Series("payoff_expected_stress_edge", model["payoff"].predict(matrix)),
    )
    if model["loss"] is not None:
        scored = scored.with_columns(
            pl.Series("predicted_loss_severity", model["loss"].predict(matrix))
        )
    return scored


def _attach_action_columns(
    frame: pl.DataFrame, probability: np.ndarray, config: FrozenTrainingConfig
) -> pl.DataFrame:
    predicted_up = probability >= 0.5
    confidence = np.where(predicted_up, probability, 1.0 - probability)
    up_price = frame["up_ask_vwap_5"].to_numpy()
    down_price = frame["down_ask_vwap_5"].to_numpy()
    fee = frame["fee_rate"].to_numpy()
    up_cost = (
        up_price
        + fee * up_price * (1.0 - up_price)
        + config.execution.execution_reserve_per_share
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
        )
        late = (
            (pl.col("seconds_elapsed") >= 180)
            & (pl.col("model_confidence") >= policy["late_confidence"])
            & (pl.col("model_stress_edge") >= policy["late_stress_edge"])
            & (pl.col("admission_probability") >= policy["late_admission"])
            & (pl.col("payoff_expected_stress_edge") >= policy["late_payoff_lower_bound"])
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
    return {
        "evaluation_rows": scored.height,
        "evaluation_markets": evaluation["market_id"].n_unique(),
        "settled": trades,
        "wins": wins,
        "losses": trades - wins,
        "accuracy": wins / trades if trades else None,
        "market_coverage": trades / evaluation["market_id"].n_unique(),
        "net_pnl": float(pnl),
        "stress_net_pnl": float(stress_pnl),
        "expectancy_per_trade": float(pnl / trades) if trades else None,
        "brier": float(np.mean((probability - label) ** 2)),
        "log_loss": float(log_loss(label, probability, labels=[0, 1])),
        "ece_10": _ece(label, probability, 10),
        "average_probability_up": float(probability.mean()),
        "directions": directions,
    }


def _feature_medians(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    values = frame.select(features).to_numpy().astype(float, copy=False)
    values[~np.isfinite(values)] = np.nan
    medians = np.nanmedian(values, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    return medians


def _matrix(
    frame: pl.DataFrame, features: tuple[str, ...], medians: np.ndarray
) -> np.ndarray:
    values = frame.select(features).to_numpy().astype(float, copy=False)
    invalid = ~np.isfinite(values)
    if invalid.any():
        values = values.copy()
        rows, columns = np.nonzero(invalid)
        values[rows, columns] = medians[columns]
    return values


def _regime_matrix(
    frame: pl.DataFrame,
    raw: np.ndarray,
    medians: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base = frame.select(REGIME_FEATURES).to_numpy().astype(float, copy=False)
    predicted = (raw >= 0.5).astype(float)[:, None]
    cost = np.where(
        predicted[:, 0].astype(bool),
        frame["up_ask_vwap_5"].to_numpy(),
        frame["down_ask_vwap_5"].to_numpy(),
    )
    buckets = np.clip(np.digitize(cost, PRICE_BUCKET_EDGES) - 1, 0, 3)
    bucket_hot = np.column_stack([(buckets == index).astype(float) for index in range(4)])
    values = np.column_stack((_logit(raw), predicted, base, bucket_hot))
    values[~np.isfinite(values)] = np.nan
    if medians is None:
        medians = np.nanmedian(values, axis=0)
        medians[~np.isfinite(medians)] = 0.0
    rows, columns = np.nonzero(~np.isfinite(values))
    if len(rows):
        values[rows, columns] = medians[columns]
    return values, medians


def _entry_cells() -> tuple[tuple[str, int, int], ...]:
    return (
        ("early_15_89", 15, 90),
        ("middle_90_119", 90, 120),
        ("middle_120_149", 120, 150),
        ("middle_150_179", 150, 180),
        ("late_180_240", 180, 241),
    )


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


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
        f"Run: `{metrics['run_id']}`  ",
        f"Source commit: `{metrics['source_commit']}`  ",
        (
            f"Common evaluation: {metrics['common_evaluation']['markets']:,} markets, "
            f"{metrics['common_evaluation']['rows']:,} decision rows."
        ),
        "",
        "| Arm | Family | Settled | W-L | Net P&L | Stress P&L | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for family in FAMILIES:
            row = metrics["results"][arm][family]
            lines.append(
                f"| {arm} | {family} | {row['settled']} | {row['wins']}-{row['losses']} | "
                f"{row['net_pnl']:.2f} | {row['stress_net_pnl']:.2f} | "
                f"{row['brier']:.4f} | {row['ece_10']:.4f} |"
            )
    lines.extend(
        [
            "",
            "W is the watermark refresh, R adds causal PMData RefPrice, and T changes the target to PMData TWAP-60.",
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
