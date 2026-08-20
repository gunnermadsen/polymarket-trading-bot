"""Offline ablation tournament for payoff-aware BTC middle-market policies."""

from __future__ import annotations

import argparse
import json
import math
import platform
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from .chainlink_oi_features import BINANCE_OI_FEATURES
from .continuous_edge_training import (
    CHAINLINK_FEATURES,
    PRIMARY_FEATURES,
    VWAP_QUANTITIES,
    ExecutionConfig,
    market_equal_weights,
    policy_metrics,
    policy_metrics_by_price_bucket,
    rolling_policy_metrics,
)
from .core_extract import (
    configure_read_only_connection,
    database_connection,
    file_sha256,
)
from .middle_market_tournament import (
    REGIME_FEATURES,
    OutcomeModel,
    _attach_decision_scores,
    _attach_external_features,
    _block,
    _bucket_indices,
    _candidate_eligible_frame,
    _finite,
    _fit_outcome,
    _load_development_frame,
    _load_holdout_frame,
    _logit,
    _matrix,
    _variable_features,
)
from .middle_market_tournament import (
    load_config as load_source_config,
)

SCHEMA_VERSION = "btc-middle-market-ablation-tournament-v1"
MODEL_SCHEMA_VERSION = "btc-middle-market-ablation-model-v1"
ENTRY_CELLS = ("90-119", "120-149", "150-179")
LOSS_FEATURES = (
    "probability_selected",
    "selected_cost_5",
    "selected_edge_5",
    "seconds_elapsed_scaled",
    "btc_path_from_window_open_bps",
    "btc_cross_venue_boundary_gap_bps",
    "btc_path_terminal_volatility_z",
    "btc_boundary_terminal_volatility_z",
    "btc_boundary_cross_count",
    "btc_seconds_since_boundary_cross",
    "btc_reversal_5_vs_30",
    "btc_volatility_shock_30_vs_120",
    "btc_signed_flow_30s",
    "btc_signed_flow_60s",
    "oracle_return_from_window_open_bps",
    "binance_oracle_basis_bps",
    "pm_vwap5_overround",
    "pm_vwap200_overround",
    "pm_depth_imbalance",
    "pm_up_book_age_seconds",
    "pm_down_book_age_seconds",
)


@dataclass(frozen=True)
class Windows:
    fit_start: datetime
    fit_end: datetime
    calibration_end: datetime
    meta_end: datetime
    policy_end: datetime
    holdout_end: datetime


@dataclass(frozen=True)
class Entry:
    start_second: int
    end_second_exclusive: int
    cadence_seconds: int
    calibration_cells: tuple[tuple[int, int], ...]


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
    oof_initial_fit_days: int
    oof_calibration_days: int
    oof_fold_days: int
    loss_max_iter: int
    loss_max_leaf_nodes: int
    loss_min_samples_leaf: int
    loss_l2_regularization: float
    wait_horizons_seconds: tuple[int, ...]


@dataclass(frozen=True)
class PolicyConfig:
    confidence_thresholds: tuple[float, ...]
    stress_edge_thresholds: tuple[float, ...]
    loss_severity_thresholds: tuple[float, ...]
    wait_advantage_thresholds: tuple[float, ...]
    coverage_targets: tuple[float, ...]
    price_bucket_edges: tuple[float, ...]
    rolling_fold_days: int


@dataclass(frozen=True)
class GateConfig:
    minimum_trades: int
    minimum_market_coverage: float
    minimum_accuracy: float
    minimum_wilson_lower: float
    minimum_profit_factor: float
    minimum_payoff_ratio: float
    minimum_profitable_fold_ratio: float
    minimum_active_days: int
    maximum_daily_pnl_concentration: float
    minimum_direction_trades: int
    minimum_price_bucket_trades: int
    maximum_average_entry_second: float
    maximum_median_entry_second: float
    minimum_early_trade_ratio: float
    minimum_active_entry_cells: int
    maximum_drawdown_to_net_pnl: float
    bootstrap_resamples: int
    bootstrap_confidence: float


@dataclass(frozen=True)
class ReadinessConfig:
    minimum_scheduled_market_coverage: float
    minimum_daily_market_coverage: float
    required_consecutive_days: int
    maximum_single_day_market_share: float
    minimum_optional_feature_coverage: float


@dataclass(frozen=True)
class Paths:
    source_tournament_config: Path
    runs: Path
    committed_results: Path


@dataclass(frozen=True)
class TournamentConfig:
    source_path: Path
    package_root: Path
    profile: str
    random_seed: int
    windows: Windows
    entry: Entry
    execution: ExecutionConfig
    model: ModelConfig
    policy: PolicyConfig
    gates: GateConfig
    readiness: ReadinessConfig
    paths: Paths


@dataclass
class LocalCalibration:
    estimator: LogisticRegression
    weight: float
    penalty: float


@dataclass
class CorrectnessCalibration:
    estimator: LogisticRegression
    scaler: StandardScaler
    regime_features: tuple[str, ...]
    stratified: bool
    locals: dict[str, LocalCalibration]
    global_penalty: float


@dataclass
class ProbabilityModifier:
    estimator: LogisticRegression
    scaler: StandardScaler
    feature_names: tuple[str, ...]


@dataclass
class Candidate:
    name: str
    feature_names: tuple[str, ...]
    eligibility_features: tuple[str, ...]
    outcome: OutcomeModel
    correctness: CorrectnessCalibration
    probability_modifier: ProbabilityModifier | None = None
    loss_model: HistGradientBoostingRegressor | None = None
    loss_feature_names: tuple[str, ...] = ()
    wait_model: HistGradientBoostingRegressor | None = None
    wait_feature_names: tuple[str, ...] = ()
    profiles: dict[str, dict[str, Any]] | None = None
    submitted_policy: dict[str, Any] | None = None


def load_config(path: Path) -> TournamentConfig:
    source = path.resolve()
    root = source.parent.parent
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    training = raw["training"]
    if training.get("paper_only") is not True or training.get("live_capital_allowed") is not False:
        raise ValueError("ablation tournament must remain offline and paper-only")
    paths = raw["paths"]
    config = TournamentConfig(
        source_path=source,
        package_root=root,
        profile=str(training["profile"]),
        random_seed=int(training["random_seed"]),
        windows=Windows(**{name: _utc(value) for name, value in raw["windows"].items()}),
        entry=Entry(
            start_second=int(raw["entry"]["start_second"]),
            end_second_exclusive=int(raw["entry"]["end_second_exclusive"]),
            cadence_seconds=int(raw["entry"]["cadence_seconds"]),
            calibration_cells=tuple(
                tuple(int(item) for item in row) for row in raw["entry"]["calibration_cells"]
            ),
        ),
        execution=ExecutionConfig(
            quantities=tuple(int(value) for value in raw["execution"]["quantities"]),
            freshness_seconds=int(raw["execution"]["freshness_seconds"]),
            maximum_depth_participation=float(raw["execution"]["maximum_depth_participation"]),
            execution_reserve_per_share=float(raw["execution"]["execution_reserve_per_share"]),
            stress_slippage_per_share=float(raw["execution"]["stress_slippage_per_share"]),
        ),
        model=ModelConfig(
            **{
                **{
                    name: raw["model"][name]
                    for name in ModelConfig.__dataclass_fields__
                    if name != "wait_horizons_seconds"
                },
                "wait_horizons_seconds": tuple(
                    int(value) for value in raw["model"]["wait_horizons_seconds"]
                ),
            }
        ),
        policy=PolicyConfig(
            **{
                name: (
                    tuple(float(value) for value in raw["policy"][name])
                    if name != "rolling_fold_days"
                    else int(raw["policy"][name])
                )
                for name in PolicyConfig.__dataclass_fields__
            }
        ),
        gates=GateConfig(**raw["gates"]),
        readiness=ReadinessConfig(**raw["readiness"]),
        paths=Paths(
            source_tournament_config=_path(root, paths["source_tournament_config"]),
            runs=_path(root, paths["runs"]),
            committed_results=_path(root, paths["committed_results"]),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: TournamentConfig) -> None:
    windows = list(asdict(config.windows).values())
    if windows != sorted(windows) or len(windows) != len(set(windows)):
        raise ValueError("tournament windows must be strictly chronological")
    if (config.entry.start_second, config.entry.end_second_exclusive) != (90, 180):
        raise ValueError("entry contract must remain 90-179 seconds")
    if config.entry.calibration_cells != ((90, 120), (120, 150), (150, 180)):
        raise ValueError("entry calibration cells changed")
    if config.execution.quantities != VWAP_QUANTITIES:
        raise ValueError("VWAP quantity contract changed")
    if config.execution.maximum_depth_participation != 0.25:
        raise ValueError("depth participation must remain 25 percent")
    if config.readiness.required_consecutive_days != 8:
        raise ValueError("independent holdout must contain eight consecutive days")
    if not config.paths.source_tournament_config.is_file():
        raise FileNotFoundError(config.paths.source_tournament_config)


def run_development_tournament(config: TournamentConfig) -> tuple[Path, dict[str, Any]]:
    print("load: permitted development evidence through August 10", flush=True)
    frame, source_manifests = _load_frame(config)
    fit = _block(frame, config.windows.fit_start, config.windows.fit_end)
    calibration = _block(frame, config.windows.fit_end, config.windows.calibration_end)
    meta = _block(frame, config.windows.calibration_end, config.windows.meta_end)
    policy = _block(frame, config.windows.meta_end, config.windows.policy_end)
    for name, block in (
        ("fit", fit),
        ("calibration", calibration),
        ("meta", meta),
        ("policy", policy),
    ):
        if block.is_empty():
            raise RuntimeError(f"{name} block is empty")

    print("audit: unopened independent holdout readiness", flush=True)
    readiness = audit_holdout_readiness(config)
    print("train: shared payoff-aware control outcome", flush=True)
    control_features = tuple(PRIMARY_FEATURES)
    control_outcome = _fit_shared_outcome(
        fit, calibration, control_features, config, seed=config.random_seed
    )
    print("train: shared Chainlink outcome", flush=True)
    chain_features = (*PRIMARY_FEATURES, *CHAINLINK_FEATURES)
    chain_fit = fit.drop_nulls(CHAINLINK_FEATURES)
    chain_calibration = calibration.drop_nulls(CHAINLINK_FEATURES)
    chain_outcome = _fit_shared_outcome(
        chain_fit, chain_calibration, chain_features, config, seed=config.random_seed + 11
    )

    print("train: chronological out-of-fold Chainlink predictions", flush=True)
    oof = _oof_predictions(fit.drop_nulls(CHAINLINK_FEATURES), chain_features, config)
    base_calibration = _decision_frame(chain_calibration, chain_outcome, config, None)
    base_meta = _decision_frame(meta.drop_nulls(CHAINLINK_FEATURES), chain_outcome, config, None)
    modifier = _fit_probability_modifier(oof.drop_nulls(BINANCE_OI_FEATURES), config)
    modified_calibration = _decision_frame(
        chain_calibration.drop_nulls(BINANCE_OI_FEATURES), chain_outcome, config, modifier
    )

    print("train: stratified, loss-severity, regime, OI, and wait components", flush=True)
    control_calibration_frame = _decision_frame(calibration, control_outcome, config, None)
    control_correctness = _fit_correctness(
        control_calibration_frame,
        config,
        stratified=False,
        regime=False,
        seed=config.random_seed + 21,
    )
    stratified_correctness = _fit_correctness(
        base_calibration, config, stratified=True, regime=False, seed=config.random_seed + 22
    )
    regime_correctness = _fit_correctness(
        base_calibration, config, stratified=True, regime=True, seed=config.random_seed + 23
    )
    oi_correctness = _fit_correctness(
        modified_calibration, config, stratified=True, regime=False, seed=config.random_seed + 24
    )
    full_correctness = _fit_correctness(
        modified_calibration, config, stratified=True, regime=True, seed=config.random_seed + 25
    )

    oof_loss = _loss_training_frame(oof, config)
    meta_loss = _loss_training_frame(base_meta, config)
    loss_training = pl.concat((oof_loss, meta_loss), how="diagonal_relaxed")
    loss_features = _variable_features(loss_training, LOSS_FEATURES)
    loss_model = _fit_regressor(
        loss_training,
        loss_features,
        "loss_severity_target",
        config,
        seed=config.random_seed + 31,
    )
    wait_training = _wait_training_frame(base_meta, config)
    wait_features = _variable_features(wait_training, LOSS_FEATURES)
    wait_model = _fit_regressor(
        wait_training,
        wait_features,
        "enter_now_advantage_target",
        config,
        seed=config.random_seed + 32,
    )

    candidates = {
        "payoff_control": Candidate(
            "payoff_control", control_features, (), control_outcome, control_correctness
        ),
        "chainlink_stratified_payoff": Candidate(
            "chainlink_stratified_payoff",
            chain_features,
            CHAINLINK_FEATURES,
            chain_outcome,
            stratified_correctness,
        ),
        "chainlink_oof_loss_veto": Candidate(
            "chainlink_oof_loss_veto",
            chain_features,
            CHAINLINK_FEATURES,
            chain_outcome,
            stratified_correctness,
            loss_model=loss_model,
            loss_feature_names=loss_features,
        ),
        "chainlink_regime_calibrated": Candidate(
            "chainlink_regime_calibrated",
            chain_features,
            CHAINLINK_FEATURES,
            chain_outcome,
            regime_correctness,
        ),
        "chainlink_oi_modifier": Candidate(
            "chainlink_oi_modifier",
            chain_features,
            (*CHAINLINK_FEATURES, *BINANCE_OI_FEATURES),
            chain_outcome,
            oi_correctness,
            probability_modifier=modifier,
        ),
        "chainlink_full_combined": Candidate(
            "chainlink_full_combined",
            chain_features,
            (*CHAINLINK_FEATURES, *BINANCE_OI_FEATURES),
            chain_outcome,
            full_correctness,
            probability_modifier=modifier,
            loss_model=loss_model,
            loss_feature_names=loss_features,
        ),
        "chainlink_revised_wait_value": Candidate(
            "chainlink_revised_wait_value",
            chain_features,
            CHAINLINK_FEATURES,
            chain_outcome,
            stratified_correctness,
            wait_model=wait_model,
            wait_feature_names=wait_features,
        ),
    }

    results: dict[str, Any] = {}
    ledgers: dict[str, pl.DataFrame] = {}
    for index, candidate in enumerate(candidates.values()):
        print(f"select: {candidate.name}", flush=True)
        scored = _score_candidate(policy, candidate, config)
        profiles, submitted, search = _select_profiles(
            scored, candidate, config, seed=config.random_seed + 100 + index
        )
        candidate.profiles = profiles
        candidate.submitted_policy = submitted["policy"]
        selected = _apply_policy(scored, submitted["policy"])
        ledgers[candidate.name] = selected
        results[candidate.name] = {
            "features": list(candidate.feature_names),
            "eligible_rows": scored.height,
            "eligible_markets": scored["market_id"].n_unique(),
            "candidate_feature_market_coverage": (
                scored["market_id"].n_unique() / policy["market_id"].n_unique()
                if policy["market_id"].n_unique()
                else 0.0
            ),
            "probability": _probability_metrics(scored),
            "profiles": profiles,
            "submitted": submitted,
            "search": search,
            "capacity": {
                str(quantity): policy_metrics(selected, config, quantity=quantity)
                for quantity in config.execution.quantities
            },
            "cells": _cell_metrics(selected, config),
            "directions": _direction_metrics(selected, config),
            "price_buckets": policy_metrics_by_price_bucket(selected, config, quantity=5),
            "loss_tail": _loss_tail_metrics(selected, config),
        }

    control_metrics = results["payoff_control"]["submitted"]["metrics"]
    for result in results.values():
        row = result["submitted"]["metrics"]
        result["control_relative"] = {
            "coverage_change": row["strict_market_coverage"]
            - control_metrics["strict_market_coverage"],
            "accuracy_change": row["accuracy"] - control_metrics["accuracy"],
            "stress_pnl_change": row["stress_net_pnl"] - control_metrics["stress_net_pnl"],
            "average_entry_second_change": (
                row["average_entry_second"] - control_metrics["average_entry_second"]
                if row["average_entry_second"] is not None
                and control_metrics["average_entry_second"] is not None
                else None
            ),
        }

    qualified = [
        (
            result["submitted"]["metrics"]["strict_market_coverage"],
            result["submitted"]["metrics"]["bootstrap_stress_expectancy_lower"],
            -result["submitted"]["metrics"]["maximum_drawdown"],
            -result["submitted"]["metrics"]["average_entry_second"],
            name,
        )
        for name, result in results.items()
        if result["submitted"]["qualified"]
    ]
    provisional = max(qualified)[-1] if qualified else None
    decision = "holdout_ready" if readiness["passed"] else "holdout_not_ready"
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    temporary = config.paths.runs / f"{run_id}.partial"
    final = config.paths.committed_results / run_id
    if temporary.exists() or final.exists():
        raise FileExistsError(run_id)
    temporary.mkdir(parents=True)
    artifact_path = temporary / "tournament.joblib"
    joblib.dump(
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "profile": config.profile,
            "candidates": candidates,
            "provisional_champion": provisional,
            "holdout_scored": False,
            "runtime_exported": False,
        },
        artifact_path,
        compress=3,
    )
    ledger_dir = temporary / "development-ledgers"
    ledger_dir.mkdir()
    for name, ledger in ledgers.items():
        ledger.write_parquet(ledger_dir / f"{name}.parquet", compression="zstd")
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": config.profile,
        "paper_only": True,
        "runtime_exported": False,
        "trading_processes_changed": False,
        "holdout_scored": False,
        "source_commit": _git_revision(config.package_root),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "configuration": {
            "path": str(config.source_path),
            "sha256": file_sha256(config.source_path),
            "windows": {name: value.isoformat() for name, value in asdict(config.windows).items()},
            "entry": asdict(config.entry),
            "execution": asdict(config.execution),
            "model": asdict(config.model),
            "policy": asdict(config.policy),
            "gates": asdict(config.gates),
            "readiness": asdict(config.readiness),
        },
        "data": {
            "rows": frame.height,
            "markets": frame["market_id"].n_unique(),
            "fit_markets": fit["market_id"].n_unique(),
            "calibration_markets": calibration["market_id"].n_unique(),
            "meta_markets": meta["market_id"].n_unique(),
            "policy_markets": policy["market_id"].n_unique(),
            "source_manifests": source_manifests,
        },
        "holdout_readiness": readiness,
        "development_results": results,
        "qualification": {
            "provisional_champion": provisional,
            "decision": decision,
            "holdout_scoring_allowed": readiness["passed"],
            "post_hoc_holdout_selection_allowed": False,
        },
        "model_artifact": {
            "path": "tournament.joblib",
            "sha256": file_sha256(artifact_path),
        },
        "limitations": [
            "The independent holdout was not scored because its frozen readiness contract did not pass.",
            "August 2-10 was consumed by the prior tournament and is used only as development evidence.",
            "Projected PnL assumes recorded ask VWAP is fillable and does not model queue position.",
            "L2 and trade-print candidates are excluded until causal feature coverage satisfies the frozen requirement.",
        ],
    }
    _write_json(temporary / "metrics.json", metrics)
    (temporary / "report.md").write_text(_render_report(metrics))
    (temporary / "tournament.sha256").write_text(file_sha256(artifact_path) + "\n")
    config.paths.committed_results.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    return final, metrics


def _load_frame(config: TournamentConfig) -> tuple[pl.DataFrame, dict[str, Any]]:
    source = load_source_config(config.paths.source_tournament_config)
    development = _load_development_frame(source)
    consumed, capacity_manifest = _load_holdout_frame(source)
    combined = pl.concat((development, consumed), how="diagonal_relaxed").sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    combined, external_manifest = _attach_external_features(combined, source)
    frame = _block(combined, config.windows.fit_start, config.windows.policy_end)
    return frame, {"capacity": capacity_manifest, "external": external_manifest}


def _fit_shared_outcome(
    fit: pl.DataFrame,
    calibration: pl.DataFrame,
    features: tuple[str, ...],
    config: TournamentConfig,
    *,
    seed: int,
) -> OutcomeModel:
    return _fit_outcome(
        fit,
        calibration,
        features,
        seed,
        (config.windows.fit_start, config.windows.fit_end),
        (config.windows.fit_end, config.windows.calibration_end),
        minimum_fit_markets=500,
        minimum_calibration_markets=200,
    )


def _oof_predictions(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    config: TournamentConfig,
) -> pl.DataFrame:
    pieces: list[pl.DataFrame] = []
    validation_start = config.windows.fit_start + timedelta(
        days=config.model.oof_initial_fit_days + config.model.oof_calibration_days
    )
    fold = 0
    while validation_start < config.windows.fit_end:
        fit_end = validation_start - timedelta(days=config.model.oof_calibration_days)
        validation_end = min(
            validation_start + timedelta(days=config.model.oof_fold_days),
            config.windows.fit_end,
        )
        fit_block = _block(frame, config.windows.fit_start, fit_end)
        calibration_block = _block(frame, fit_end, validation_start)
        validation_block = _block(frame, validation_start, validation_end)
        if (
            fit_block["market_id"].n_unique() < 500
            or calibration_block["market_id"].n_unique() < 150
            or validation_block.is_empty()
        ):
            validation_start = validation_end
            fold += 1
            continue
        model = _fit_outcome(
            fit_block,
            calibration_block,
            features,
            config.random_seed + 200 + fold,
            (config.windows.fit_start, fit_end),
            (fit_end, validation_start),
            minimum_fit_markets=500,
            minimum_calibration_markets=150,
        )
        scored = _attach_decision_scores(validation_block, model, config)
        pieces.append(scored.with_columns(pl.lit(fold).alias("oof_fold")))
        validation_start = validation_end
        fold += 1
    if not pieces:
        raise RuntimeError("OOF schedule produced no folds")
    return pl.concat(pieces, how="diagonal_relaxed")


def _fit_probability_modifier(
    frame: pl.DataFrame,
    config: TournamentConfig,
) -> ProbabilityModifier:
    features = _variable_features(frame, BINANCE_OI_FEATURES)
    matrix = np.column_stack((_logit(frame["probability_up"].to_numpy()), _matrix(frame, features)))
    scaler = StandardScaler().fit(matrix)
    estimator = LogisticRegression(
        C=config.model.calibration_c,
        max_iter=config.model.calibration_max_iter,
        random_state=config.random_seed + 300,
    )
    estimator.fit(
        scaler.transform(matrix),
        frame["label_up"].to_numpy(),
        sample_weight=market_equal_weights(frame),
    )
    if estimator.n_iter_.max() >= config.model.calibration_max_iter:
        raise RuntimeError("OI probability modifier did not converge")
    return ProbabilityModifier(estimator, scaler, features)


def _decision_frame(
    frame: pl.DataFrame,
    outcome: OutcomeModel,
    config: TournamentConfig,
    modifier: ProbabilityModifier | None,
) -> pl.DataFrame:
    scored = _attach_decision_scores(frame, outcome, config)
    if modifier is None:
        return scored
    matrix = np.column_stack(
        (_logit(scored["probability_up"].to_numpy()), _matrix(scored, modifier.feature_names))
    )
    probability_up = modifier.estimator.predict_proba(modifier.scaler.transform(matrix))[:, 1]
    return _replace_probability_scores(scored, probability_up, config)


def _replace_probability_scores(
    frame: pl.DataFrame,
    probability_up: np.ndarray,
    config: TournamentConfig,
) -> pl.DataFrame:
    predicted_up = probability_up >= 0.5
    selected = np.where(predicted_up, probability_up, 1.0 - probability_up)
    price = np.where(
        predicted_up,
        frame["up_ask_vwap_5"].to_numpy(),
        frame["down_ask_vwap_5"].to_numpy(),
    )
    fee = frame["fee_rate"].to_numpy() * price * (1.0 - price)
    cost = price + fee + config.execution.execution_reserve_per_share
    return frame.with_columns(
        pl.Series("probability_up", probability_up),
        pl.Series("predicted_up", predicted_up),
        pl.Series("probability_selected", selected),
        pl.Series("selected_cost_5", cost),
        pl.Series("selected_edge_5", selected - cost),
        pl.Series(
            "direction_correct",
            predicted_up == frame["label_up"].to_numpy().astype(bool),
        ),
        pl.Series(
            "price_bucket_index",
            _bucket_indices(cost, config.policy.price_bucket_edges).astype(np.int8),
        ),
    )


def _fit_correctness(
    frame: pl.DataFrame,
    config: TournamentConfig,
    *,
    stratified: bool,
    regime: bool,
    seed: int,
) -> CorrectnessCalibration:
    regime_features = tuple(REGIME_FEATURES) if regime else ()
    matrix = _correctness_matrix(frame, config, regime_features)
    scaler = StandardScaler().fit(matrix)
    estimator = LogisticRegression(
        C=config.model.calibration_c,
        max_iter=config.model.calibration_max_iter,
        random_state=seed,
    )
    labels = frame["direction_correct"].to_numpy().astype(np.int8)
    weights = market_equal_weights(frame)
    estimator.fit(scaler.transform(matrix), labels, sample_weight=weights)
    if estimator.n_iter_.max() >= config.model.calibration_max_iter:
        raise RuntimeError("correctness calibration did not converge")
    global_probability = estimator.predict_proba(scaler.transform(matrix))[:, 1]
    global_penalty = _calibration_penalty(
        labels, global_probability, weights, frame["market_id"].n_unique()
    )
    locals_: dict[str, LocalCalibration] = {}
    if stratified:
        for key, subset in _calibration_subsets(frame, config).items():
            if subset.height < config.model.local_calibration_minimum_rows:
                continue
            local_labels = subset["direction_correct"].to_numpy().astype(np.int8)
            if len(np.unique(local_labels)) < 2:
                continue
            local_weights = market_equal_weights(subset)
            local = LogisticRegression(
                C=config.model.calibration_c,
                max_iter=config.model.calibration_max_iter,
                random_state=seed,
            )
            local_x = _logit(subset["probability_selected"].to_numpy()).reshape(-1, 1)
            local.fit(local_x, local_labels, sample_weight=local_weights)
            probability = local.predict_proba(local_x)[:, 1]
            penalty = _calibration_penalty(
                local_labels, probability, local_weights, subset["market_id"].n_unique()
            )
            local_weight = subset["market_id"].n_unique() / (
                subset["market_id"].n_unique() + config.model.local_calibration_shrinkage_rows
            )
            locals_[key] = LocalCalibration(local, local_weight, penalty)
    return CorrectnessCalibration(
        estimator, scaler, regime_features, stratified, locals_, global_penalty
    )


def _correctness_matrix(
    frame: pl.DataFrame,
    config: TournamentConfig,
    regime_features: tuple[str, ...],
) -> np.ndarray:
    probability = np.clip(frame["probability_selected"].to_numpy(), 1e-6, 1 - 1e-6)
    side = frame["predicted_up"].to_numpy().astype(float)
    cells = frame["middle_cell"].to_numpy()
    buckets = frame["price_bucket_index"].to_numpy()
    cell_hot = np.column_stack([(cells == name).astype(float) for name in ENTRY_CELLS])
    bucket_hot = np.column_stack(
        [
            (buckets == index).astype(float)
            for index in range(len(config.policy.price_bucket_edges) - 1)
        ]
    )
    logit = _logit(probability)
    pieces = [
        logit,
        side,
        cell_hot,
        bucket_hot,
        cell_hot * logit[:, None],
        bucket_hot * logit[:, None],
    ]
    if regime_features:
        pieces.append(
            np.nan_to_num(_matrix(frame, regime_features), nan=0.0, posinf=0.0, neginf=0.0)
        )
    return np.column_stack(pieces)


def _calibration_subsets(
    frame: pl.DataFrame,
    config: TournamentConfig,
) -> dict[str, pl.DataFrame]:
    output: dict[str, pl.DataFrame] = {}
    for cell in ENTRY_CELLS:
        for side_name, side in (("up", True), ("down", False)):
            for bucket in range(len(config.policy.price_bucket_edges) - 1):
                key = f"{cell}:{side_name}:{bucket}"
                output[key] = frame.filter(
                    (pl.col("middle_cell") == cell)
                    & (pl.col("predicted_up") == side)
                    & (pl.col("price_bucket_index") == bucket)
                )
    return output


def _calibration_penalty(
    labels: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    markets: int,
) -> float:
    accuracy = float(np.average(labels, weights=weights))
    bias = abs(float(np.average(labels - probability, weights=weights)))
    return bias + math.sqrt(max(accuracy * (1.0 - accuracy), 0.01) / max(markets, 1))


def _score_correctness(
    frame: pl.DataFrame,
    model: CorrectnessCalibration,
    config: TournamentConfig,
) -> pl.DataFrame:
    matrix = _correctness_matrix(frame, config, model.regime_features)
    global_probability = model.estimator.predict_proba(model.scaler.transform(matrix))[:, 1]
    probability = global_probability.copy()
    penalties = np.full(frame.height, model.global_penalty)
    if model.stratified:
        for index, (cell, side, bucket, selected) in enumerate(
            zip(
                frame["middle_cell"].to_numpy(),
                frame["predicted_up"].to_numpy(),
                frame["price_bucket_index"].to_numpy(),
                frame["probability_selected"].to_numpy(),
                strict=True,
            )
        ):
            key = f"{cell}:{'up' if side else 'down'}:{int(bucket)}"
            local = model.locals.get(key)
            if local is None:
                continue
            local_probability = local.estimator.predict_proba(
                _logit(np.asarray([selected])).reshape(-1, 1)
            )[0, 1]
            probability[index] = (
                local.weight * local_probability + (1.0 - local.weight) * global_probability[index]
            )
            penalties[index] = (
                local.weight * local.penalty + (1.0 - local.weight) * model.global_penalty
            )
    lower = np.clip(probability - penalties, 0.0, 1.0)
    return frame.with_columns(
        pl.Series("correctness_probability", probability),
        pl.Series("correctness_uncertainty_penalty", penalties),
        pl.Series("lower_correctness_probability", lower),
        pl.Series(
            "stress_edge_lower_bound",
            lower
            - frame["selected_cost_5"].to_numpy()
            - config.execution.stress_slippage_per_share,
        ),
    )


def _loss_training_frame(
    frame: pl.DataFrame,
    config: TournamentConfig,
) -> pl.DataFrame:
    correct = frame["direction_correct"].to_numpy().astype(float)
    realized = (
        correct - frame["selected_cost_5"].to_numpy() - config.execution.stress_slippage_per_share
    )
    return frame.with_columns(pl.Series("loss_severity_target", np.maximum(-realized, 0.0)))


def _wait_training_frame(
    frame: pl.DataFrame,
    config: TournamentConfig,
) -> pl.DataFrame:
    output = frame.with_columns(
        (
            pl.col("direction_correct").cast(pl.Float64)
            - pl.col("selected_cost_5")
            - config.execution.stress_slippage_per_share
        ).alias("_realized_stress_edge")
    )
    future_columns: list[str] = []
    for lag in config.model.wait_horizons_seconds:
        name = f"_future_{lag}"
        future = output.select(
            "market_id",
            (pl.col("seconds_elapsed") - lag).alias("seconds_elapsed"),
            pl.col("_realized_stress_edge").alias(name),
        )
        output = output.join(future, on=["market_id", "seconds_elapsed"], how="left")
        future_columns.append(name)
    return output.with_columns(
        (
            pl.col("_realized_stress_edge")
            - pl.max_horizontal(*(pl.col(name) for name in future_columns))
        ).alias("enter_now_advantage_target")
    ).drop_nulls("enter_now_advantage_target")


def _fit_regressor(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    target: str,
    config: TournamentConfig,
    *,
    seed: int,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        learning_rate=config.model.learning_rate,
        max_iter=config.model.loss_max_iter,
        max_leaf_nodes=config.model.loss_max_leaf_nodes,
        min_samples_leaf=config.model.loss_min_samples_leaf,
        l2_regularization=config.model.loss_l2_regularization,
        random_state=seed,
        early_stopping=False,
    )
    model.fit(
        _matrix(frame, features),
        frame[target].to_numpy(),
        sample_weight=market_equal_weights(frame),
    )
    return model


def _score_candidate(
    frame: pl.DataFrame,
    candidate: Candidate,
    config: TournamentConfig,
) -> pl.DataFrame:
    eligible = _candidate_eligible_frame(frame, candidate.eligibility_features)
    scored = _decision_frame(eligible, candidate.outcome, config, candidate.probability_modifier)
    scored = _score_correctness(scored, candidate.correctness, config)
    if candidate.loss_model is not None:
        scored = scored.with_columns(
            pl.Series(
                "loss_severity_prediction",
                candidate.loss_model.predict(_matrix(scored, candidate.loss_feature_names)),
            )
        )
    else:
        scored = scored.with_columns(pl.lit(0.0).alias("loss_severity_prediction"))
    if candidate.wait_model is not None:
        scored = scored.with_columns(
            pl.Series(
                "wait_advantage",
                candidate.wait_model.predict(_matrix(scored, candidate.wait_feature_names)),
            )
        )
    else:
        scored = scored.with_columns(pl.lit(0.0).alias("wait_advantage"))
    return scored


def _select_profiles(
    frame: pl.DataFrame,
    candidate: Candidate,
    config: TournamentConfig,
    *,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    loss_thresholds = (
        config.policy.loss_severity_thresholds if candidate.loss_model is not None else (math.inf,)
    )
    wait_thresholds = (
        config.policy.wait_advantage_thresholds
        if candidate.wait_model is not None
        else (-math.inf,)
    )
    records: list[dict[str, Any]] = []
    eligible_markets = frame["market_id"].n_unique()
    for confidence in config.policy.confidence_thresholds:
        for edge in config.policy.stress_edge_thresholds:
            for loss in loss_thresholds:
                for wait in wait_thresholds:
                    policy = {
                        "confidence": confidence,
                        "stress_edge": edge,
                        "loss_severity": loss,
                        "wait_advantage": wait,
                    }
                    selected = _apply_policy(frame, policy)
                    metrics = _full_metrics(
                        selected,
                        frame,
                        config,
                        seed=seed + len(records),
                        include_bootstrap=False,
                    )
                    records.append({"policy": policy, "metrics": metrics})
    profiles: dict[str, dict[str, Any]] = {}
    for target in config.policy.coverage_targets:
        chosen = min(
            records,
            key=lambda row: (
                abs(row["metrics"]["strict_market_coverage"] - target),
                -row["metrics"]["stress_net_pnl"],
            ),
        )
        evaluated = _evaluate_policy_record(
            frame, chosen["policy"], config, seed=seed + 10_000 + int(target * 100)
        )
        profiles[f"coverage_{int(target * 100)}"] = _qualify_record(evaluated, config)
    preliminary = [
        record
        for record in records
        if _qualification_checks(record["metrics"], config, require_bootstrap=False)[0]
    ]
    qualified_records = []
    for index, record in enumerate(preliminary):
        evaluated = _evaluate_policy_record(
            frame, record["policy"], config, seed=seed + 20_000 + index
        )
        qualified = _qualify_record(evaluated, config)
        if qualified["qualified"]:
            qualified_records.append(qualified)
    if qualified_records:
        submitted = max(
            qualified_records,
            key=lambda row: (
                row["metrics"]["strict_market_coverage"],
                row["metrics"]["bootstrap_stress_expectancy_lower"],
                -row["metrics"]["maximum_drawdown"],
                -row["metrics"]["average_entry_second"],
            ),
        )
    else:
        preliminary_evaluated = [
            _qualify_record(record, config, require_bootstrap=False) for record in records
        ]
        chosen = max(
            preliminary_evaluated,
            key=lambda row: (
                sum(row["qualification_checks"].values()),
                row["metrics"]["stress_net_pnl"],
                row["metrics"]["strict_market_coverage"],
            ),
        )
        submitted = _qualify_record(
            _evaluate_policy_record(frame, chosen["policy"], config, seed=seed + 30_000),
            config,
        )
    profiles["maximum_qualified"] = submitted
    frontier = sorted(
        (
            {
                "policy": row["policy"],
                "coverage": row["metrics"]["strict_market_coverage"],
                "accuracy": row["metrics"]["accuracy"],
                "stress_expectancy": row["metrics"]["stress_expectancy_per_trade"],
            }
            for row in records
        ),
        key=lambda row: row["coverage"],
    )
    return (
        profiles,
        submitted,
        {
            "attempts": len(records),
            "eligible_markets": eligible_markets,
            "risk_coverage": frontier,
        },
    )


def _apply_policy(frame: pl.DataFrame, policy: dict[str, Any]) -> pl.DataFrame:
    mask = (
        (frame["lower_correctness_probability"].to_numpy() >= policy["confidence"])
        & (frame["stress_edge_lower_bound"].to_numpy() >= policy["stress_edge"])
        & (frame["loss_severity_prediction"].to_numpy() <= policy["loss_severity"])
        & (frame["wait_advantage"].to_numpy() >= policy["wait_advantage"])
    )
    indices = np.flatnonzero(mask)
    if not len(indices):
        return frame.head(0)
    markets = frame["market_id"].to_numpy()
    _, first = np.unique(markets[indices], return_index=True)
    return frame[indices[np.sort(first)].tolist()]


def _full_metrics(
    selected: pl.DataFrame,
    eligible: pl.DataFrame,
    config: TournamentConfig,
    *,
    seed: int,
    include_bootstrap: bool,
) -> dict[str, Any]:
    metrics = policy_metrics(selected, config, quantity=5)
    eligible_markets = eligible["market_id"].n_unique()
    metrics["strict_market_coverage"] = (
        selected["market_id"].n_unique() / eligible_markets if eligible_markets else 0.0
    )
    metrics["wilson_accuracy_lower"] = _wilson_lower(
        round(metrics["accuracy"] * metrics["trades"]), metrics["trades"]
    )
    metrics["bootstrap_stress_expectancy_lower"] = (
        _bootstrap_stress_lower(selected, config, seed=seed)
        if include_bootstrap
        else metrics["stress_expectancy_per_trade"]
    )
    q10 = policy_metrics(selected, config, quantity=10)
    q20 = policy_metrics(selected, config, quantity=20)
    metrics["q10_stress_net_pnl"] = q10["stress_net_pnl"]
    metrics["q20_stress_net_pnl"] = q20["stress_net_pnl"]
    folds = rolling_policy_metrics(
        selected,
        config,
        config.windows.meta_end,
        config.windows.policy_end,
        quantity=5,
    )
    metrics["profitable_fold_ratio"] = folds["profitable_fold_ratio"]
    metrics["rolling_folds"] = folds
    if selected.is_empty():
        metrics["early_trade_ratio"] = 0.0
        metrics["active_entry_cells"] = 0
    else:
        metrics["early_trade_ratio"] = float((selected["middle_cell"] == "90-119").mean())
        metrics["active_entry_cells"] = selected["middle_cell"].n_unique()
    directions = _direction_metrics(selected, config)
    buckets = policy_metrics_by_price_bucket(selected, config, quantity=5)
    metrics["enabled_directions_positive"] = all(
        row["stress_net_pnl"] > 0
        for row in directions.values()
        if row["trades"] >= config.gates.minimum_direction_trades
    )
    metrics["populated_price_buckets_nonnegative"] = all(
        row["stress_net_pnl"] >= 0
        for row in buckets.values()
        if row["trades"] >= config.gates.minimum_price_bucket_trades
    )
    metrics["drawdown_to_net_pnl"] = (
        metrics["maximum_drawdown"] / metrics["net_pnl"] if metrics["net_pnl"] > 0 else math.inf
    )
    metrics["trades_per_active_day"] = (
        metrics["trades"] / metrics["active_days"] if metrics["active_days"] else 0.0
    )
    return metrics


def _qualification_checks(
    metrics: dict[str, Any],
    config: TournamentConfig,
    *,
    require_bootstrap: bool = True,
) -> tuple[bool, dict[str, bool]]:
    gates = config.gates
    checks = {
        "minimum_trades": metrics["trades"] >= gates.minimum_trades,
        "minimum_market_coverage": metrics["strict_market_coverage"]
        >= gates.minimum_market_coverage,
        "minimum_accuracy": metrics["accuracy"] >= gates.minimum_accuracy,
        "minimum_wilson_lower": metrics["wilson_accuracy_lower"] >= gates.minimum_wilson_lower,
        "positive_stress_pnl": metrics["stress_net_pnl"] > 0,
        "positive_q10_stress": metrics["q10_stress_net_pnl"] > 0,
        "positive_q20_stress": metrics["q20_stress_net_pnl"] > 0,
        "minimum_profit_factor": (metrics["profit_factor"] or 0.0) >= gates.minimum_profit_factor,
        "minimum_payoff_ratio": metrics["payoff_ratio"] >= gates.minimum_payoff_ratio,
        "bootstrap_lower_positive": (
            metrics["bootstrap_stress_expectancy_lower"] > 0 if require_bootstrap else True
        ),
        "profitable_fold_ratio": metrics["profitable_fold_ratio"]
        >= gates.minimum_profitable_fold_ratio,
        "minimum_active_days": metrics["active_days"] >= gates.minimum_active_days,
        "daily_concentration": metrics["daily_pnl_concentration"]
        <= gates.maximum_daily_pnl_concentration,
        "directions_positive": metrics["enabled_directions_positive"],
        "price_buckets_nonnegative": metrics["populated_price_buckets_nonnegative"],
        "average_entry_time": metrics["average_entry_second"] is not None
        and metrics["average_entry_second"] <= gates.maximum_average_entry_second,
        "median_entry_time": metrics["median_entry_second"] is not None
        and metrics["median_entry_second"] <= gates.maximum_median_entry_second,
        "minimum_early_ratio": metrics["early_trade_ratio"] >= gates.minimum_early_trade_ratio,
        "minimum_active_cells": metrics["active_entry_cells"] >= gates.minimum_active_entry_cells,
        "maximum_drawdown_ratio": metrics["drawdown_to_net_pnl"]
        <= gates.maximum_drawdown_to_net_pnl,
    }
    return all(checks.values()), checks


def _qualify_record(
    record: dict[str, Any],
    config: TournamentConfig,
    *,
    require_bootstrap: bool = True,
) -> dict[str, Any]:
    qualified, checks = _qualification_checks(
        record["metrics"], config, require_bootstrap=require_bootstrap
    )
    return {
        "policy": record["policy"],
        "metrics": record["metrics"],
        "qualified": qualified,
        "qualification_checks": checks,
    }


def _evaluate_policy_record(
    frame: pl.DataFrame,
    policy: dict[str, Any],
    config: TournamentConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    selected = _apply_policy(frame, policy)
    return {
        "policy": policy,
        "metrics": _full_metrics(
            selected,
            frame,
            config,
            seed=seed,
            include_bootstrap=True,
        ),
    }


def _bootstrap_stress_lower(
    frame: pl.DataFrame,
    config: TournamentConfig,
    *,
    seed: int,
) -> float:
    if frame.is_empty():
        return 0.0
    predicted = frame["predicted_up"].to_numpy().astype(bool)
    correct = predicted == frame["label_up"].to_numpy().astype(bool)
    price = np.where(
        predicted,
        frame["up_ask_vwap_5"].to_numpy(),
        frame["down_ask_vwap_5"].to_numpy(),
    )
    fee = frame["fee_rate"].to_numpy() * price * (1.0 - price)
    stress = (
        correct.astype(float)
        - price
        - fee
        - config.execution.execution_reserve_per_share
        - config.execution.stress_slippage_per_share
    ) * 5
    rng = np.random.default_rng(seed)
    means = np.empty(config.gates.bootstrap_resamples)
    for index in range(config.gates.bootstrap_resamples):
        means[index] = rng.choice(stress, size=len(stress), replace=True).mean()
    alpha = 1.0 - config.gates.bootstrap_confidence
    return float(np.quantile(means, alpha))


def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = probability + z * z / (2.0 * total)
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    )
    return (center - radius) / denominator


def _cell_metrics(frame: pl.DataFrame, config: TournamentConfig) -> dict[str, Any]:
    return {
        name: policy_metrics(frame.filter(pl.col("middle_cell") == name), config, quantity=5)
        for name in ENTRY_CELLS
    }


def _direction_metrics(frame: pl.DataFrame, config: TournamentConfig) -> dict[str, Any]:
    return {
        name: policy_metrics(frame.filter(pl.col("predicted_up") == value), config, quantity=5)
        for name, value in (("up", True), ("down", False))
    }


def _probability_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {
            "rows": 0,
            "markets": 0,
            "accuracy": 0.0,
            "brier": None,
            "log_loss": None,
            "bias": None,
            "ece": None,
        }
    probability = np.clip(frame["probability_up"].to_numpy(), 1e-6, 1 - 1e-6)
    labels = frame["label_up"].to_numpy().astype(np.int8)
    bins = np.minimum((probability * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(probability[mask].mean()) - float(labels[mask].mean())
            )
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "accuracy": float(np.mean((probability >= 0.5) == labels.astype(bool))),
        "brier": float(brier_score_loss(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "bias": float(np.mean(probability - labels)),
        "ece": ece,
    }


def _loss_tail_metrics(
    frame: pl.DataFrame,
    config: TournamentConfig,
) -> dict[str, Any]:
    if frame.is_empty():
        return {
            "worst_trade_stress_pnl": 0.0,
            "worst_five_percent_mean_stress_pnl": 0.0,
            "loss_trade_ratio": 0.0,
        }
    predicted = frame["predicted_up"].to_numpy().astype(bool)
    correct = predicted == frame["label_up"].to_numpy().astype(bool)
    price = np.where(
        predicted,
        frame["up_ask_vwap_5"].to_numpy(),
        frame["down_ask_vwap_5"].to_numpy(),
    )
    fee = frame["fee_rate"].to_numpy() * price * (1.0 - price)
    stress = (
        correct.astype(float)
        - price
        - fee
        - config.execution.execution_reserve_per_share
        - config.execution.stress_slippage_per_share
    ) * 5
    tail_count = max(1, math.ceil(len(stress) * 0.05))
    tail = np.sort(stress)[:tail_count]
    return {
        "worst_trade_stress_pnl": float(stress.min()),
        "worst_five_percent_mean_stress_pnl": float(tail.mean()),
        "loss_trade_ratio": float((stress < 0).mean()),
    }


def audit_holdout_readiness(config: TournamentConfig) -> dict[str, Any]:
    connection = database_connection()
    configure_read_only_connection(connection)
    try:
        rows = connection.execute(
            """
            WITH days AS MATERIALIZED (
              SELECT day FROM generate_series(
                %(start)s::timestamptz,
                %(end)s::timestamptz - interval '1 day',
                interval '1 day'
              ) AS day
            ), markets AS MATERIALIZED (
              SELECT window_start::date AS day,
                     count(*)::int AS scheduled,
                     count(*) FILTER (
                       WHERE validation_status = 'valid'
                         AND official_outcome IN ('up', 'down')
                     )::int AS resolved
              FROM polymarket.btc_interval_markets
              WHERE window_start >= %(start)s AND window_start < %(end)s
              GROUP BY 1
            ), facts AS MATERIALIZED (
              SELECT source_effective_at::date AS day,
                     count(DISTINCT fact.market_id)::int AS canonical_markets
              FROM polymarket.btc_market_reference_facts fact
              JOIN polymarket.backfill_artifacts artifact USING (artifact_id)
              WHERE artifact.status = 'completed'
                AND fact.fact_type = 'opening_boundary'
                AND fact.source_effective_at >= %(start)s
                AND fact.source_effective_at < %(end)s
              GROUP BY 1
            ), capacity AS MATERIALIZED (
              SELECT minimum_source_timestamp::date AS day,
                     count(*) FILTER (WHERE record_count = 1152)::int AS complete_hours
              FROM polymarket.backfill_artifacts
              WHERE provider = 'pmxt_v2_capacity_execution_snapshots_v2'
                AND status = 'completed'
                AND minimum_source_timestamp >= %(start)s
                AND minimum_source_timestamp < %(end)s
              GROUP BY 1
            )
            SELECT days.day::date,
                   coalesce(markets.scheduled, 0),
                   coalesce(markets.resolved, 0),
                   coalesce(facts.canonical_markets, 0),
                   coalesce(capacity.complete_hours, 0)
            FROM days
            LEFT JOIN markets ON markets.day = days.day::date
            LEFT JOIN facts ON facts.day = days.day::date
            LEFT JOIN capacity ON capacity.day = days.day::date
            ORDER BY days.day
            """,
            {"start": config.windows.policy_end, "end": config.windows.holdout_end},
        ).fetchall()
    finally:
        connection.close()
    daily = []
    for day, scheduled, resolved, canonical, hours in rows:
        usable = min(resolved, canonical)
        coverage = usable / scheduled if scheduled else 0.0
        daily.append(
            {
                "day": day.isoformat(),
                "scheduled_markets": scheduled,
                "resolved_markets": resolved,
                "canonical_markets": canonical,
                "complete_capacity_hours": hours,
                "market_coverage": coverage,
                "passed": coverage >= config.readiness.minimum_daily_market_coverage
                and hours == 24,
            }
        )
    scheduled_total = sum(row["scheduled_markets"] for row in daily)
    usable_total = sum(min(row["resolved_markets"], row["canonical_markets"]) for row in daily)
    overall = usable_total / scheduled_total if scheduled_total else 0.0
    maximum_share = (
        max((min(row["resolved_markets"], row["canonical_markets"]) for row in daily), default=0)
        / usable_total
        if usable_total
        else 0.0
    )
    checks = {
        "required_days": len(daily) == config.readiness.required_consecutive_days,
        "all_daily_coverage": all(row["passed"] for row in daily),
        "overall_market_coverage": overall >= config.readiness.minimum_scheduled_market_coverage,
        "single_day_concentration": maximum_share
        <= config.readiness.maximum_single_day_market_share,
    }
    return {
        "range_start": config.windows.policy_end.isoformat(),
        "range_end": config.windows.holdout_end.isoformat(),
        "daily": daily,
        "scheduled_markets": scheduled_total,
        "usable_canonical_markets": usable_total,
        "market_coverage": overall,
        "maximum_single_day_market_share": maximum_share,
        "checks": checks,
        "passed": all(checks.values()),
        "holdout_labels_accessed": False,
    }


def _render_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# BTC Five-Minute Middle-Market Ablation Tournament",
        "",
        f"Development decision: **{metrics['qualification']['decision']}**",
        "",
        f"Provisional champion: `{metrics['qualification']['provisional_champion'] or 'none'}`",
        "",
        "The independent holdout was not scored unless its frozen readiness contract passed.",
        "",
        "## Development comparison (submitted profiles)",
        "",
        "| Candidate | Qualified | Trades | Coverage | Accuracy | Net PnL | Stress PnL | PF | Avg entry |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in metrics["development_results"].items():
        submitted = result["submitted"]
        row = submitted["metrics"]
        lines.append(
            f"| {name} | {'yes' if submitted['qualified'] else 'no'} | {row['trades']} | "
            f"{row['strict_market_coverage']:.2%} | {row['accuracy']:.2%} | "
            f"{row['net_pnl']:.2f} | {row['stress_net_pnl']:.2f} | "
            f"{_fmt(row['profit_factor'])} | {_fmt(row['average_entry_second'])} |"
        )
    readiness = metrics["holdout_readiness"]
    lines.extend(
        [
            "",
            "## Independent holdout readiness",
            "",
            (
                f"Coverage: {readiness['usable_canonical_markets']}/"
                f"{readiness['scheduled_markets']} "
                f"({readiness['market_coverage']:.2%}); passed: "
                f"**{readiness['passed']}**."
            ),
            "",
            "## Limitations",
            "",
            *(f"- {value}" for value in metrics["limitations"]),
        ]
    )
    return "\n".join(lines) + "\n"


def _git_revision(package_root: Path) -> str:
    git_file = package_root.parents[1] / ".git"
    if not git_file.is_file():
        return "unknown"
    content = git_file.read_text().strip()
    git_dir = Path(content.split(":", 1)[1].strip())
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head
    common = git_dir
    if (git_dir / "commondir").is_file():
        common = (git_dir / (git_dir / "commondir").read_text().strip()).resolve()
    return (common / head.split(" ", 1)[1]).read_text().strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_finite(payload), indent=2, sort_keys=True) + "\n")


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.3f}"


def _utc(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamps must be timezone aware")
    return parsed.astimezone(UTC)


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run_dir, metrics = run_development_tournament(load_config(args.config))
    print(f"run: {run_dir}")
    print(json.dumps(metrics["qualification"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
