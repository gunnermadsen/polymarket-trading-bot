"""Offline Q5 and trajectory-distilled optimal-stopping tournament."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import tomllib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .continuous_edge_training import (
    ADMISSION_FEATURES,
    CHAINLINK_FEATURES,
    TrainingConfig,
    _block,
    _finite,
    _matrix,
    attach_probability_stability,
    coverage_summary,
    extract_capacity_evidence,
    load_training_frame,
    market_equal_weights,
    policy_metrics,
    policy_metrics_by_price_bucket,
    variable_feature_names,
)
from .continuous_edge_training import (
    load_config as load_source_config,
)
from .core_extract import file_sha256

SCHEMA_VERSION = "btc-q5-optimal-stopping-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-q5-optimal-stopping-artifact-v1"
CANDIDATES = (
    "q5_incumbent",
    "q5_loss_veto",
    "q5_coverage_expander",
    "q5_veto_expander_composite",
    "two_sided_direct_value",
    "expected_utility_optimal_stopping",
    "distributional_optimal_stopping",
)
ENTRY_CELLS = (
    ("early_15_89", 15, 90),
    ("middle_90_119", 90, 120),
    ("middle_120_149", 120, 150),
    ("middle_150_179", 150, 180),
    ("late_180_240", 180, 241),
)
ACTION_FEATURES = tuple(
    dict.fromkeys(
        (
            *ADMISSION_FEATURES,
            *CHAINLINK_FEATURES,
            "probability_up",
            "up_cost_5",
            "down_cost_5",
            "up_ask_vwap_5",
            "down_ask_vwap_5",
            "fee_rate",
        )
    )
)
VETO_FEATURES = tuple(
    dict.fromkeys(
        (
            *ADMISSION_FEATURES,
            *CHAINLINK_FEATURES,
            "payoff_expected_stress_edge",
            "admission_probability",
        )
    )
)


@dataclass(frozen=True)
class OuterFold:
    name: str
    fit_end: datetime
    selection_start: datetime
    selection_end: datetime
    evaluation_start: datetime
    evaluation_end: datetime


@dataclass(frozen=True)
class SelectionCell:
    name: str
    start_second: int
    end_second_exclusive: int
    minimum_accuracy: float


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    quantile: float
    meta_training_start: datetime
    residual_calibration_days: int


@dataclass(frozen=True)
class PolicyConfig:
    veto_thresholds: tuple[float, ...]
    value_thresholds: tuple[float, ...]
    stopping_margins: tuple[float, ...]
    coverage_targets: tuple[float, ...]
    minimum_inner_trades: int


@dataclass(frozen=True)
class GateConfig:
    minimum_directional_trades: int
    minimum_asymmetric_trades: int
    minimum_profit_factor_directional: float
    minimum_profit_factor_asymmetric: float
    minimum_asymmetric_payoff_ratio: float
    minimum_profitable_fold_ratio: float
    minimum_q10_stress_pnl: float
    minimum_bootstrap_lower: float
    bootstrap_resamples: int
    bootstrap_confidence: float
    minimum_populated_cell_trades: int
    minimum_populated_direction_trades: int


@dataclass(frozen=True)
class TournamentPaths:
    source_training_config: Path
    incumbent_artifact: Path
    runs: Path
    committed_results: Path


@dataclass(frozen=True)
class TournamentConfig:
    source_path: Path
    package_root: Path
    profile: str
    random_seed: int
    fresh_holdout: bool
    folds: tuple[OuterFold, ...]
    cells: tuple[SelectionCell, ...]
    model: ModelConfig
    policy: PolicyConfig
    gates: GateConfig
    paths: TournamentPaths
    source: TrainingConfig


def load_config(path: Path) -> TournamentConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    training = raw["training"]
    if training.get("paper_only") is not True or training.get("live_capital_allowed") is not False:
        raise ValueError("tournament must remain offline and paper-only")
    path_values = raw["paths"]
    source_config = _path(package_root, path_values["source_training_config"])
    source = load_source_config(source_config)
    model_values = dict(raw["model"])
    model_values["meta_training_start"] = _utc(model_values["meta_training_start"])
    config = TournamentConfig(
        source_path=source_path,
        package_root=package_root,
        profile=str(training["profile"]),
        random_seed=int(training["random_seed"]),
        fresh_holdout=bool(training["fresh_holdout"]),
        folds=tuple(
            OuterFold(
                **{key: _utc(value) if key != "name" else value for key, value in row.items()}
            )
            for row in raw["outer_folds"]
        ),
        cells=tuple(SelectionCell(**row) for row in raw["selection_cells"]),
        model=ModelConfig(**model_values),
        policy=PolicyConfig(
            veto_thresholds=tuple(float(value) for value in raw["policy"]["veto_thresholds"]),
            value_thresholds=tuple(float(value) for value in raw["policy"]["value_thresholds"]),
            stopping_margins=tuple(float(value) for value in raw["policy"]["stopping_margins"]),
            coverage_targets=tuple(float(value) for value in raw["policy"]["coverage_targets"]),
            minimum_inner_trades=int(raw["policy"]["minimum_inner_trades"]),
        ),
        gates=GateConfig(**raw["gates"]),
        paths=TournamentPaths(
            source_training_config=source_config,
            incumbent_artifact=_path(package_root, path_values["incumbent_artifact"]),
            runs=_path(package_root, path_values["runs"]),
            committed_results=_path(package_root, path_values["committed_results"]),
        ),
        source=source,
    )
    _validate_config(config)
    return config


def _validate_config(config: TournamentConfig) -> None:
    if config.profile != "btc_5m_q5_optimal_stopping_tournament":
        raise ValueError("unexpected tournament profile")
    if config.fresh_holdout:
        raise ValueError("previously consumed evidence cannot be marked fresh")
    if (
        tuple((cell.name, cell.start_second, cell.end_second_exclusive) for cell in config.cells)
        != ENTRY_CELLS
    ):
        raise ValueError("selection cells changed")
    previous_end: datetime | None = None
    for fold in config.folds:
        if not (
            fold.fit_end
            <= fold.selection_start
            < fold.selection_end
            <= fold.evaluation_start
            < fold.evaluation_end
        ):
            raise ValueError(f"invalid chronology for {fold.name}")
        if previous_end is not None and fold.evaluation_start < previous_end:
            raise ValueError("outer evaluation folds overlap")
        previous_end = fold.evaluation_end
    if config.source.execution.quantities[-1] != 200:
        raise ValueError("source VWAP contract does not reach 200")
    if config.gates.bootstrap_resamples < 500:
        raise ValueError("bootstrap requires at least 500 resamples")
    if not config.paths.incumbent_artifact.is_file():
        raise FileNotFoundError(config.paths.incumbent_artifact)


def run_tournament(config: TournamentConfig) -> tuple[Path, dict[str, Any]]:
    print("load: immutable full-window capacity evidence", flush=True)
    manifest = extract_capacity_evidence(config.source)
    frame = load_training_frame(config.source, manifest)
    incumbent = joblib.load(config.paths.incumbent_artifact)
    if incumbent.get("profile") != "btc_5m_continuous_edge_payoff_aware":
        raise ValueError("unexpected Q5 incumbent profile")
    print(
        f"joined: {frame.height:,} strict rows, {frame['market_id'].n_unique():,} markets",
        flush=True,
    )
    scored = _attach_action_targets(
        _score_incumbent(frame, incumbent, config.source), config.source
    )

    fold_models: dict[str, Any] = {}
    fold_results: dict[str, dict[str, Any]] = {name: {} for name in CANDIDATES}
    ledgers: dict[str, list[pl.DataFrame]] = {name: [] for name in CANDIDATES}
    evaluated_markets: set[str] = set()
    for fold_index, fold in enumerate(config.folds):
        print(f"fold: {fold.name} fit models", flush=True)
        fit = scored.filter(pl.col("window_start") < fold.fit_end)
        selection = _block(scored, fold.selection_start, fold.selection_end)
        evaluation = _block(scored, fold.evaluation_start, fold.evaluation_end)
        if (
            min(
                fit["market_id"].n_unique(),
                selection["market_id"].n_unique(),
                evaluation["market_id"].n_unique(),
            )
            == 0
        ):
            raise RuntimeError(f"{fold.name} contains an empty block")
        models = _fit_fold_models(
            fit, selection, incumbent, config, seed=config.random_seed + 100 * fold_index
        )
        fold_models[fold.name] = models
        policies = _select_fold_policies(selection, models, incumbent, config)
        selections = _apply_all_candidates(evaluation, models, policies, incumbent, config)
        evaluated_markets.update(str(value) for value in evaluation["market_id"].unique().to_list())
        for name in CANDIDATES:
            selected = selections[name]
            ledgers[name].append(selected.with_columns(pl.lit(fold.name).alias("outer_fold")))
            fold_results[name][fold.name] = {
                "policy": policies[name],
                "metrics": _metrics(selected, evaluation, config.source),
                "stopping": _stopping_diagnostics(selected, evaluation, selections["q5_incumbent"]),
            }

    combined: dict[str, Any] = {}
    combined_ledgers: dict[str, pl.DataFrame] = {}
    total_markets = len(evaluated_markets)
    scheduled_markets = _scheduled_market_count(config, manifest)
    for name in CANDIDATES:
        combined_ledgers[name] = (
            pl.concat(ledgers[name], how="diagonal_relaxed") if ledgers[name] else scored.head(0)
        )
    for index, name in enumerate(CANDIDATES):
        ledger = combined_ledgers[name]
        result = _aggregate_candidate(
            name,
            ledger,
            fold_results[name],
            total_markets,
            scheduled_markets,
            config,
            incumbent=combined_ledgers["q5_incumbent"],
            seed=config.random_seed + 10_000 + index,
        )
        combined[name] = result

    qualified = [name for name, result in combined.items() if result["qualification"]["passed"]]
    champion = (
        max(qualified, key=lambda name: _champion_rank(combined[name])) if qualified else None
    )
    diagnostic_leader = max(CANDIDATES, key=lambda name: _diagnostic_rank(combined[name]))
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    temporary = config.paths.runs / f"{run_id}.partial"
    final = config.paths.committed_results / run_id
    if temporary.exists() or final.exists():
        raise FileExistsError(run_id)
    temporary.mkdir(parents=True)
    artifact_path = temporary / "tournament.joblib"
    joblib.dump(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "profile": config.profile,
            "incumbent_artifact": str(config.paths.incumbent_artifact),
            "incumbent_sha256": file_sha256(config.paths.incumbent_artifact),
            "fold_models": fold_models,
            "candidate_names": list(CANDIDATES),
            "champion": champion,
            "diagnostic_leader": diagnostic_leader,
            "runtime_exported": False,
            "production_qualified": False,
        },
        artifact_path,
        compress=3,
    )
    ledger_dir = temporary / "development-ledgers"
    ledger_dir.mkdir()
    for name, ledger in combined_ledgers.items():
        ledger.write_parquet(ledger_dir / f"{name}.parquet", compression="zstd")

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": config.profile,
        "paper_only": True,
        "fresh_holdout": False,
        "runtime_exported": False,
        "production_qualified": False,
        "trading_processes_changed": False,
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
            "outer_folds": [_serialize_fold(fold) for fold in config.folds],
            "selection_cells": [asdict(cell) for cell in config.cells],
            "model": _serialize_dataclass(config.model),
            "policy": asdict(config.policy),
            "gates": asdict(config.gates),
            "execution": asdict(config.source.execution),
        },
        "data": {
            "rows": frame.height,
            "markets": frame["market_id"].n_unique(),
            "evaluated_strict_markets": total_markets,
            "evaluated_scheduled_markets": scheduled_markets,
            "strict_data_coverage": total_markets / scheduled_markets if scheduled_markets else 0.0,
            "capacity_manifest": manifest,
            "coverage": coverage_summary(frame, config.source),
            "incumbent_artifact": str(config.paths.incumbent_artifact),
            "incumbent_sha256": file_sha256(config.paths.incumbent_artifact),
            "twap_used": False,
            "optional_incomplete_features_required": False,
        },
        "candidates": combined,
        "fold_results": fold_results,
        "decision": {
            "status": "qualified_development_champion" if champion else "no_qualified_champion",
            "champion": champion,
            "diagnostic_leader": diagnostic_leader,
            "production_selection_allowed": False,
            "fresh_post_august_evidence_required": True,
        },
        "model_artifact": {
            "path": artifact_path.name,
            "sha256": file_sha256(artifact_path),
        },
        "limitations": [
            "All evidence through August 2 was consumed by prior development work and is not a fresh holdout.",
            "Future trajectory values are training targets only; evaluation and student inference use causal current-state features.",
            "Projected PnL assumes recorded ask VWAP was fillable and does not model queue position.",
            "The tournament is offline only and exports no runtime model or trading-process configuration.",
        ],
    }
    _write_json(temporary / "metrics.json", metrics)
    (temporary / "report.md").write_text(_render_report(metrics))
    (temporary / "tournament.sha256").write_text(file_sha256(artifact_path) + "\n")
    config.paths.committed_results.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    return final, metrics


def _score_incumbent(
    frame: pl.DataFrame, artifact: dict[str, Any], config: TrainingConfig
) -> pl.DataFrame:
    pieces: list[pl.DataFrame] = []
    for band in config.bands:
        subset = frame.filter(pl.col("time_band") == band.name)
        if subset.is_empty():
            continue
        probability = _expert_probability(artifact["experts"][band.name], subset)
        pieces.append(subset.with_columns(pl.Series("probability_up", probability)))
    scored = pl.concat(pieces, how="vertical").sort(["market_id", "seconds_elapsed", "observed_at"])
    directional = _expert_probability(artifact["family_proxies"]["directional"], scored)
    asymmetric = _expert_probability(artifact["family_proxies"]["asymmetric"], scored)
    probability_up = scored["probability_up"].to_numpy()
    predicted_up = probability_up >= 0.5
    selected_probability = np.where(predicted_up, probability_up, 1.0 - probability_up)
    selected_price = np.where(
        predicted_up, scored["up_ask_vwap_5"].to_numpy(), scored["down_ask_vwap_5"].to_numpy()
    )
    fee = scored["fee_rate"].to_numpy() * selected_price * (1.0 - selected_price)
    selected_cost = selected_price + fee + config.execution.execution_reserve_per_share
    scored = scored.with_columns(
        pl.Series("predicted_up", predicted_up),
        pl.Series("probability_selected", selected_probability),
        pl.Series("confidence_margin", selected_probability - 0.5),
        pl.Series("selected_cost_5", selected_cost),
        pl.Series("selected_edge_5", selected_probability - selected_cost),
        pl.Series("direction_correct", predicted_up == scored["label_up"].to_numpy().astype(bool)),
        pl.Series("directional_family_probability_up", directional),
        pl.Series("asymmetric_family_probability_up", asymmetric),
        pl.Series("directional_family_disagreement", np.abs(probability_up - directional)),
        pl.Series("asymmetric_family_disagreement", np.abs(probability_up - asymmetric)),
        pl.Series("family_probability_spread", np.abs(directional - asymmetric)),
        pl.Series(
            "family_vote_agreement", ((directional >= 0.5) == (asymmetric >= 0.5)).astype(np.int8)
        ),
    )
    calibration = artifact["price_time_calibration"]
    calibrated = calibration["estimator"].predict_proba(
        _price_time_matrix(
            scored, tuple(calibration["band_names"]), tuple(calibration["price_bucket_edges"])
        )
    )[:, 1]
    probability_up = np.where(predicted_up, calibrated, 1.0 - calibrated)
    scored = scored.with_columns(
        pl.Series("probability_up", probability_up),
        pl.Series("probability_selected", calibrated),
        pl.Series("confidence_margin", calibrated - 0.5),
        pl.Series("selected_edge_5", calibrated - selected_cost),
    )
    edges = tuple(calibration["price_bucket_edges"])
    buckets = _bucket_indices(selected_cost, edges)
    guard = artifact["calibration_guard"]
    bands = scored["time_band"].to_numpy()
    penalties = np.asarray(
        [
            guard["penalties"].get(f"{band}:{bucket}", guard["global_penalty"])
            for band, bucket in zip(bands, buckets, strict=True)
        ]
    )
    conservative = np.clip(calibrated - penalties, 0.0, 1.0)
    required = np.asarray(config.policy.price_bucket_minimum_edges)[buckets]
    scored = scored.with_columns(
        pl.Series("price_bucket_index", buckets.astype(np.int8)),
        pl.Series("calibration_uncertainty_penalty", penalties),
        pl.Series("conservative_probability_selected", conservative),
        pl.Series("conservative_edge_5", conservative - selected_cost),
        pl.Series("price_bucket_minimum_edge", required),
    )
    scored = attach_probability_stability(scored)
    admission = artifact["admission_model"]
    matrix = _matrix(scored, admission["feature_names"])
    admission_probability = admission["profitable_classifier"].predict_proba(matrix)[:, 1]
    expected_stress = admission["stress_edge_regressor"].predict(matrix)
    return scored.with_columns(
        pl.Series("admission_probability", admission_probability),
        pl.Series("payoff_expected_stress_edge", expected_stress),
        pl.Series("payoff_stress_edge_lower_bound", expected_stress),
        pl.Series("payoff_loss_probability", 1.0 - admission_probability),
        pl.Series(
            "payoff_expected_shortfall",
            (1.0 - admission_probability)
            * (selected_cost + config.execution.stress_slippage_per_share),
        ),
    )


def _expert_probability(spec: dict[str, Any], frame: pl.DataFrame) -> np.ndarray:
    raw = np.clip(
        spec["estimator"].predict_proba(_matrix(frame, spec["feature_names"]))[:, 1],
        1e-7,
        1.0 - 1e-7,
    )
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    return spec["calibrator"].predict_proba(logits)[:, 1]


def _price_time_matrix(
    frame: pl.DataFrame, band_names: tuple[str, ...], edges: tuple[float, ...]
) -> np.ndarray:
    probability = np.clip(frame["probability_selected"].to_numpy(), 1e-6, 1.0 - 1e-6)
    logit = np.log(probability / (1.0 - probability))
    buckets = _bucket_indices(frame["selected_cost_5"].to_numpy(), edges)
    bands = frame["time_band"].to_numpy()
    band_hot = np.column_stack([(bands == name).astype(float) for name in band_names])
    bucket_hot = np.column_stack(
        [(buckets == index).astype(float) for index in range(len(edges) - 1)]
    )
    return np.column_stack(
        (
            logit,
            frame["predicted_up"].to_numpy().astype(float),
            band_hot,
            bucket_hot,
            band_hot * logit[:, None],
            bucket_hot * logit[:, None],
        )
    )


def _attach_action_targets(frame: pl.DataFrame, config: TrainingConfig) -> pl.DataFrame:
    label = frame["label_up"].to_numpy().astype(float)
    fee_rate = frame["fee_rate"].to_numpy()
    up_price = frame["up_ask_vwap_5"].to_numpy()
    down_price = frame["down_ask_vwap_5"].to_numpy()
    up_cost = (
        up_price
        + fee_rate * up_price * (1.0 - up_price)
        + config.execution.execution_reserve_per_share
    )
    down_cost = (
        down_price
        + fee_rate * down_price * (1.0 - down_price)
        + config.execution.execution_reserve_per_share
    )
    up_reward = label - up_cost - config.execution.stress_slippage_per_share
    down_reward = (1.0 - label) - down_cost - config.execution.stress_slippage_per_share
    best = np.maximum.reduce((up_reward, down_reward, np.zeros(frame.height)))
    future = _future_max_by_market(frame["market_id"].to_numpy(), best)
    return frame.with_columns(
        pl.Series("up_cost_5", up_cost),
        pl.Series("down_cost_5", down_cost),
        pl.Series("up_stress_reward", up_reward),
        pl.Series("down_stress_reward", down_reward),
        pl.Series("best_realized_action_value", best),
        pl.Series("future_best_action_value", future),
        pl.Series("teacher_enter_now_advantage", best - future),
    )


def _future_max_by_market(markets: np.ndarray, values: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values), dtype=float)
    if not len(values):
        return output
    starts = np.r_[0, np.flatnonzero(markets[1:] != markets[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends, strict=True):
        group = values[start:end]
        reverse = np.maximum.accumulate(group[::-1])[::-1]
        if len(group) > 1:
            output[start : end - 1] = reverse[1:]
    return output


def _fit_fold_models(
    fit: pl.DataFrame,
    calibration: pl.DataFrame,
    incumbent: dict[str, Any],
    config: TournamentConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    features = variable_feature_names(fit, ACTION_FEATURES)
    direct_up = _fit_regressor(fit, features, "up_stress_reward", config, seed=seed)
    direct_down = _fit_regressor(fit, features, "down_stress_reward", config, seed=seed + 1)
    direct_penalties = _lower_penalties(calibration, features, direct_up, direct_down, config)

    incumbent_fit = _select_q5(fit, incumbent, config.source)
    abstained = fit.filter(
        ~pl.col("market_id").is_in(incumbent_fit["market_id"].unique().implode())
    )
    expander_features = variable_feature_names(abstained, ACTION_FEATURES)
    expander_up = _fit_regressor(
        abstained, expander_features, "up_stress_reward", config, seed=seed + 2
    )
    expander_down = _fit_regressor(
        abstained, expander_features, "down_stress_reward", config, seed=seed + 3
    )
    calibration_incumbent = _select_q5(calibration, incumbent, config.source)
    abstained_calibration = calibration.filter(
        ~pl.col("market_id").is_in(calibration_incumbent["market_id"].unique().implode())
    )
    expander_penalties = _lower_penalties(
        abstained_calibration, expander_features, expander_up, expander_down, config
    )

    quantile_up = _fit_regressor(
        fit, features, "up_stress_reward", config, seed=seed + 4, quantile=config.model.quantile
    )
    quantile_down = _fit_regressor(
        fit, features, "down_stress_reward", config, seed=seed + 5, quantile=config.model.quantile
    )
    continuation = _fit_regressor(fit, features, "future_best_action_value", config, seed=seed + 6)
    continuation_penalty = _upper_penalty(
        calibration, features, continuation, "future_best_action_value", config
    )
    continuation_upper = _fit_regressor(
        fit,
        features,
        "future_best_action_value",
        config,
        seed=seed + 7,
        quantile=1.0 - config.model.quantile,
    )

    veto_rows = _select_q5(
        fit.filter(pl.col("window_start") >= config.model.meta_training_start),
        incumbent,
        config.source,
    )
    veto = _fit_veto(veto_rows, config, seed=seed + 8)
    return {
        "features": features,
        "direct_up": direct_up,
        "direct_down": direct_down,
        "direct_penalties": direct_penalties,
        "expander_features": expander_features,
        "expander_up": expander_up,
        "expander_down": expander_down,
        "expander_penalties": expander_penalties,
        "quantile_up": quantile_up,
        "quantile_down": quantile_down,
        "continuation": continuation,
        "continuation_penalty": continuation_penalty,
        "continuation_upper": continuation_upper,
        "veto": veto,
    }


def _fit_regressor(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    target: str,
    config: TournamentConfig,
    *,
    seed: int,
    quantile: float | None = None,
) -> HistGradientBoostingRegressor:
    values: dict[str, Any] = {
        "learning_rate": config.model.learning_rate,
        "max_iter": config.model.max_iter,
        "max_leaf_nodes": config.model.max_leaf_nodes,
        "min_samples_leaf": config.model.min_samples_leaf,
        "l2_regularization": config.model.l2_regularization,
        "random_state": seed,
        "early_stopping": False,
    }
    if quantile is not None:
        values.update(loss="quantile", quantile=quantile)
    model = HistGradientBoostingRegressor(**values)
    model.fit(
        _matrix(frame, features),
        frame[target].to_numpy(),
        sample_weight=market_equal_weights(frame),
    )
    return model


def _fit_veto(frame: pl.DataFrame, config: TournamentConfig, *, seed: int) -> dict[str, Any] | None:
    if frame["market_id"].n_unique() < 20:
        return None
    labels = frame["direction_correct"].to_numpy().astype(np.int8)
    if len(np.unique(labels)) < 2:
        return None
    features = variable_feature_names(frame, VETO_FEATURES)
    model = HistGradientBoostingClassifier(
        learning_rate=config.model.learning_rate,
        max_iter=config.model.max_iter,
        max_leaf_nodes=config.model.max_leaf_nodes,
        min_samples_leaf=max(10, min(config.model.min_samples_leaf, frame.height // 8)),
        l2_regularization=config.model.l2_regularization,
        random_state=seed,
        early_stopping=False,
    )
    model.fit(_matrix(frame, features), labels)
    return {"features": features, "model": model, "training_trades": frame.height}


def _lower_penalties(
    frame: pl.DataFrame, features: tuple[str, ...], up: Any, down: Any, config: TournamentConfig
) -> dict[str, float]:
    if frame.is_empty():
        return {"up": 0.25, "down": 0.25}
    quantile = 1.0 - config.model.quantile
    return {
        "up": max(
            0.0,
            float(
                np.quantile(
                    up.predict(_matrix(frame, features)) - frame["up_stress_reward"].to_numpy(),
                    quantile,
                )
            ),
        ),
        "down": max(
            0.0,
            float(
                np.quantile(
                    down.predict(_matrix(frame, features)) - frame["down_stress_reward"].to_numpy(),
                    quantile,
                )
            ),
        ),
    }


def _upper_penalty(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    model: Any,
    target: str,
    config: TournamentConfig,
) -> float:
    if frame.is_empty():
        return 0.25
    residual = frame[target].to_numpy() - model.predict(_matrix(frame, features))
    return max(0.0, float(np.quantile(residual, 1.0 - config.model.quantile)))


def _select_fold_policies(
    selection: pl.DataFrame,
    models: dict[str, Any],
    incumbent: dict[str, Any],
    config: TournamentConfig,
) -> dict[str, Any]:
    policies: dict[str, Any] = {"q5_incumbent": {"type": "frozen"}}
    q5 = _select_q5(selection, incumbent, config.source)
    veto_scored = _score_veto(q5, models["veto"])
    policies["q5_loss_veto"] = _choose_policy(
        veto_scored, config.policy.veto_thresholds, "veto_probability", config, comparator="ge"
    )
    expander_scored = _score_value(_abstained_markets(selection, q5), models, "expander")
    policies["q5_coverage_expander"] = _choose_policy(
        expander_scored, config.policy.value_thresholds, "candidate_value", config, comparator="ge"
    )
    direct_scored = _score_value(selection, models, "direct")
    policies["two_sided_direct_value"] = _choose_policy(
        direct_scored, config.policy.value_thresholds, "candidate_value", config, comparator="ge"
    )
    expected_scored = _score_stopping(selection, models, distributional=False)
    policies["expected_utility_optimal_stopping"] = _choose_policy(
        expected_scored,
        config.policy.stopping_margins,
        "stopping_advantage",
        config,
        comparator="ge",
    )
    distributional_scored = _score_stopping(selection, models, distributional=True)
    policies["distributional_optimal_stopping"] = _choose_policy(
        distributional_scored,
        config.policy.stopping_margins,
        "stopping_advantage",
        config,
        comparator="ge",
    )
    composite_records: list[dict[str, Any]] = []
    for veto_threshold, value_threshold in product(
        config.policy.veto_thresholds, config.policy.value_thresholds
    ):
        veto = _first_crossing(veto_scored, "veto_probability", veto_threshold)
        expander = _first_crossing(expander_scored, "candidate_value", value_threshold)
        selected = _combine_first(veto, expander)
        metrics = policy_metrics(selected, config.source, quantity=5)
        composite_records.append(
            {
                "policy": {"veto_threshold": veto_threshold, "value_threshold": value_threshold},
                "metrics": metrics,
            }
        )
    policies["q5_veto_expander_composite"] = _choose_record(composite_records, config)
    return policies


def _apply_all_candidates(
    frame: pl.DataFrame,
    models: dict[str, Any],
    policies: dict[str, Any],
    incumbent: dict[str, Any],
    config: TournamentConfig,
) -> dict[str, pl.DataFrame]:
    q5 = _select_q5(frame, incumbent, config.source)
    veto_scored = _score_veto(q5, models["veto"])
    veto = _first_crossing(veto_scored, "veto_probability", policies["q5_loss_veto"]["threshold"])
    expander_scored = _score_value(_abstained_markets(frame, q5), models, "expander")
    expander = _first_crossing(
        expander_scored, "candidate_value", policies["q5_coverage_expander"]["threshold"]
    )
    composite_policy = policies["q5_veto_expander_composite"]
    composite_veto = _first_crossing(
        veto_scored, "veto_probability", composite_policy["veto_threshold"]
    )
    composite_expander = _first_crossing(
        expander_scored, "candidate_value", composite_policy["value_threshold"]
    )
    direct = _first_crossing(
        _score_value(frame, models, "direct"),
        "candidate_value",
        policies["two_sided_direct_value"]["threshold"],
    )
    expected = _first_crossing(
        _score_stopping(frame, models, distributional=False),
        "stopping_advantage",
        policies["expected_utility_optimal_stopping"]["threshold"],
    )
    distributional = _first_crossing(
        _score_stopping(frame, models, distributional=True),
        "stopping_advantage",
        policies["distributional_optimal_stopping"]["threshold"],
    )
    return {
        "q5_incumbent": q5,
        "q5_loss_veto": veto,
        "q5_coverage_expander": expander,
        "q5_veto_expander_composite": _combine_first(composite_veto, composite_expander),
        "two_sided_direct_value": direct,
        "expected_utility_optimal_stopping": expected,
        "distributional_optimal_stopping": distributional,
    }


def _select_q5(
    frame: pl.DataFrame, artifact: dict[str, Any], config: TrainingConfig
) -> pl.DataFrame:
    eligible = np.zeros(frame.height, dtype=bool)
    bands = frame["time_band"].to_numpy()
    for name, values in artifact["thresholds"].items():
        if not values.get("enabled", True):
            continue
        payoff = values.get("payoff_lower_bound", values.get("payoff_edge", -math.inf))
        eligible |= (
            (bands == name)
            & (frame["conservative_probability_selected"].to_numpy() >= values["confidence"])
            & (frame["conservative_edge_5"].to_numpy() >= values["edge"])
            & (
                frame["conservative_edge_5"].to_numpy()
                >= frame["price_bucket_minimum_edge"].to_numpy()
            )
            & (frame["admission_probability"].to_numpy() >= values["admission"])
            & (frame["payoff_expected_stress_edge"].to_numpy() >= payoff)
        )
    return _first_indices(frame, np.flatnonzero(eligible))


def _abstained_markets(frame: pl.DataFrame, incumbent: pl.DataFrame) -> pl.DataFrame:
    if incumbent.is_empty():
        return frame
    return frame.filter(~pl.col("market_id").is_in(incumbent["market_id"].unique().implode()))


def _score_veto(frame: pl.DataFrame, veto: dict[str, Any] | None) -> pl.DataFrame:
    if frame.is_empty():
        return frame.with_columns(pl.Series("veto_probability", [], dtype=pl.Float64))
    probability = (
        np.ones(frame.height)
        if veto is None
        else veto["model"].predict_proba(_matrix(frame, veto["features"]))[:, 1]
    )
    return frame.with_columns(pl.Series("veto_probability", probability))


def _score_value(frame: pl.DataFrame, models: dict[str, Any], prefix: str) -> pl.DataFrame:
    features = models["features"] if prefix == "direct" else models["expander_features"]
    up = (
        models[f"{prefix}_up"].predict(_matrix(frame, features))
        - models[f"{prefix}_penalties"]["up"]
    )
    down = (
        models[f"{prefix}_down"].predict(_matrix(frame, features))
        - models[f"{prefix}_penalties"]["down"]
    )
    return _with_action(frame, up, down, np.maximum(up, down), "candidate_value")


def _score_stopping(
    frame: pl.DataFrame, models: dict[str, Any], *, distributional: bool
) -> pl.DataFrame:
    features = models["features"]
    matrix = _matrix(frame, features)
    if distributional:
        up = models["quantile_up"].predict(matrix)
        down = models["quantile_down"].predict(matrix)
        continuation = models["continuation_upper"].predict(matrix)
    else:
        up = models["direct_up"].predict(matrix) - models["direct_penalties"]["up"]
        down = models["direct_down"].predict(matrix) - models["direct_penalties"]["down"]
        continuation = models["continuation"].predict(matrix) + models["continuation_penalty"]
    advantage = np.maximum(up, down) - np.maximum(continuation, 0.0)
    return _with_action(frame, up, down, advantage, "stopping_advantage").with_columns(
        pl.Series("continuation_value", continuation)
    )


def _with_action(
    frame: pl.DataFrame, up: np.ndarray, down: np.ndarray, score: np.ndarray, score_name: str
) -> pl.DataFrame:
    predicted_up = up >= down
    selected_cost = np.where(
        predicted_up, frame["up_cost_5"].to_numpy(), frame["down_cost_5"].to_numpy()
    )
    edges = (0.0, 0.65, 0.75, 0.85, 1.01)
    return frame.with_columns(
        pl.Series("predicted_up", predicted_up),
        pl.Series("direction_correct", predicted_up == frame["label_up"].to_numpy().astype(bool)),
        pl.Series("selected_cost_5", selected_cost),
        pl.Series("price_bucket_index", _bucket_indices(selected_cost, edges).astype(np.int8)),
        pl.Series("up_action_value", up),
        pl.Series("down_action_value", down),
        pl.Series(score_name, score),
    )


def _choose_policy(
    frame: pl.DataFrame,
    thresholds: Iterable[float],
    column: str,
    config: TournamentConfig,
    *,
    comparator: str,
) -> dict[str, Any]:
    del comparator
    records = []
    markets = frame["market_id"].n_unique()
    for threshold in thresholds:
        selected = _first_crossing(frame, column, threshold)
        metrics = policy_metrics(selected, config.source, quantity=5)
        coverage = selected["market_id"].n_unique() / markets if markets else 0.0
        records.append(
            {
                "policy": {"threshold": threshold},
                "metrics": {**metrics, "market_coverage": coverage},
            }
        )
    chosen = _choose_record(records, config)
    chosen["profiles"] = _coverage_profiles(records, config.policy.coverage_targets)
    return chosen


def _choose_record(records: list[dict[str, Any]], config: TournamentConfig) -> dict[str, Any]:
    viable = [
        row
        for row in records
        if row["metrics"]["trades"] >= config.policy.minimum_inner_trades
        and row["metrics"]["stress_net_pnl"] > 0
        and (row["metrics"]["profit_factor"] or 0.0) >= 1.0
    ]
    pool = viable or records
    chosen = max(
        pool,
        key=lambda row: (
            row["metrics"]["stress_net_pnl"] > 0,
            row["metrics"]["stress_net_pnl"],
            row["metrics"]["trades"],
            -row["metrics"]["average_entry_second"]
            if row["metrics"]["average_entry_second"] is not None
            else -999.0,
        ),
    )
    return {
        **chosen["policy"],
        "inner_metrics": chosen["metrics"],
        "inner_qualified": chosen in viable,
    }


def _coverage_profiles(records: list[dict[str, Any]], targets: tuple[float, ...]) -> dict[str, Any]:
    return {
        f"coverage_{int(target * 100)}": min(
            records,
            key=lambda row: (
                abs(row["metrics"].get("market_coverage", 0.0) - target),
                -row["metrics"]["stress_net_pnl"],
            ),
        )
        for target in targets
    }


def _first_crossing(frame: pl.DataFrame, column: str, threshold: float) -> pl.DataFrame:
    return _first_indices(frame, np.flatnonzero(frame[column].to_numpy() >= threshold))


def _first_indices(frame: pl.DataFrame, indices: np.ndarray) -> pl.DataFrame:
    if not len(indices):
        return frame.head(0)
    markets = frame["market_id"].to_numpy()
    _, first = np.unique(markets[indices], return_index=True)
    return frame[indices[np.sort(first)].tolist()]


def _combine_first(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    if left.is_empty():
        return right
    if right.is_empty():
        return left
    return (
        pl.concat((left, right), how="diagonal_relaxed")
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .unique("market_id", keep="first", maintain_order=True)
    )


def _metrics(
    selected: pl.DataFrame, eligible: pl.DataFrame, source: TrainingConfig
) -> dict[str, Any]:
    metrics = policy_metrics(selected, source, quantity=5)
    markets = eligible["market_id"].n_unique()
    metrics["market_coverage"] = selected["market_id"].n_unique() / markets if markets else 0.0
    return metrics


def _aggregate_candidate(
    name: str,
    ledger: pl.DataFrame,
    folds: dict[str, Any],
    total_markets: int,
    scheduled_markets: int,
    config: TournamentConfig,
    *,
    incumbent: pl.DataFrame,
    seed: int,
) -> dict[str, Any]:
    metrics = policy_metrics(ledger, config.source, quantity=5)
    metrics["strict_market_coverage"] = (
        ledger["market_id"].n_unique() / total_markets if total_markets else 0.0
    )
    metrics["end_to_end_market_coverage"] = (
        ledger["market_id"].n_unique() / scheduled_markets if scheduled_markets else 0.0
    )
    metrics["strict_data_coverage"] = (
        total_markets / scheduled_markets if scheduled_markets else 0.0
    )
    metrics["bootstrap_stress_expectancy_lower"] = _bootstrap_lower(ledger, config, seed=seed)
    profitable = [row["metrics"]["stress_net_pnl"] > 0 for row in folds.values()]
    metrics["profitable_fold_ratio"] = sum(profitable) / len(profitable) if profitable else 0.0
    metrics["worst_fold_stress_expectancy"] = min(
        (row["metrics"]["stress_expectancy_per_trade"] for row in folds.values()), default=0.0
    )
    cells = _cell_metrics(ledger, config.source)
    directions = _direction_metrics(ledger, config.source)
    capacity = {
        str(quantity): policy_metrics(ledger, config.source, quantity=quantity)
        for quantity in config.source.execution.quantities
    }
    qualification = _qualification(name, metrics, cells, directions, capacity, config)
    return {
        "metrics": metrics,
        "capacity": capacity,
        "cells": cells,
        "directions": directions,
        "price_buckets": policy_metrics_by_price_bucket(ledger, config.source, quantity=5),
        "loss_tail": _loss_tail(ledger, config.source),
        "stopping": _stopping_diagnostics(
            ledger,
            eligible_market_count=total_markets,
            incumbent=incumbent,
        ),
        "qualification": qualification,
    }


def _qualification(
    name: str,
    metrics: dict[str, Any],
    cells: dict[str, Any],
    directions: dict[str, Any],
    capacity: dict[str, Any],
    config: TournamentConfig,
) -> dict[str, Any]:
    populated_cells = [
        row for row in cells.values() if row["trades"] >= config.gates.minimum_populated_cell_trades
    ]
    populated_directions = [
        row
        for row in directions.values()
        if row["trades"] >= config.gates.minimum_populated_direction_trades
    ]
    cell_map = {cell.name: cell for cell in config.cells}
    accuracy_cells = all(
        row["accuracy"] >= cell_map[cell].minimum_accuracy
        for cell, row in cells.items()
        if row["trades"] >= config.gates.minimum_populated_cell_trades
    )
    universal = {
        "positive_net_pnl": metrics["net_pnl"] > 0,
        "positive_stress_pnl": metrics["stress_net_pnl"] > 0,
        "minimum_profitable_fold_ratio": metrics["profitable_fold_ratio"]
        >= config.gates.minimum_profitable_fold_ratio,
        "positive_q10_stress": capacity["10"]["stress_net_pnl"]
        > config.gates.minimum_q10_stress_pnl,
        "bootstrap_lower_positive": metrics["bootstrap_stress_expectancy_lower"]
        > config.gates.minimum_bootstrap_lower,
        "populated_cells_nonnegative": all(row["stress_net_pnl"] >= 0 for row in populated_cells),
        "populated_directions_nonnegative": all(
            row["stress_net_pnl"] >= 0 for row in populated_directions
        ),
    }
    directional = {
        "minimum_trades": metrics["trades"] >= config.gates.minimum_directional_trades,
        "minimum_profit_factor": (metrics["profit_factor"] or 0.0)
        >= config.gates.minimum_profit_factor_directional,
        "cell_accuracy_targets": accuracy_cells,
    }
    asymmetric = {
        "minimum_trades": metrics["trades"] >= config.gates.minimum_asymmetric_trades,
        "minimum_profit_factor": (metrics["profit_factor"] or 0.0)
        >= config.gates.minimum_profit_factor_asymmetric,
        "minimum_payoff_ratio": metrics["payoff_ratio"]
        >= config.gates.minimum_asymmetric_payoff_ratio,
    }
    directional_passed = all(universal.values()) and all(directional.values())
    asymmetric_passed = all(universal.values()) and all(asymmetric.values())
    return {
        "passed": directional_passed or asymmetric_passed,
        "track": "directional"
        if directional_passed
        else "asymmetric"
        if asymmetric_passed
        else None,
        "universal": universal,
        "directional": directional,
        "asymmetric": asymmetric,
        "candidate": name,
    }


def _cell_metrics(frame: pl.DataFrame, source: TrainingConfig) -> dict[str, Any]:
    return {
        name: policy_metrics(
            frame.filter(pl.col("seconds_elapsed").is_between(start, end - 1)), source, quantity=5
        )
        for name, start, end in ENTRY_CELLS
    }


def _direction_metrics(frame: pl.DataFrame, source: TrainingConfig) -> dict[str, Any]:
    return {
        "up": policy_metrics(frame.filter(pl.col("predicted_up")), source, quantity=5),
        "down": policy_metrics(frame.filter(~pl.col("predicted_up")), source, quantity=5),
    }


def _loss_tail(frame: pl.DataFrame, source: TrainingConfig) -> dict[str, Any]:
    if frame.is_empty():
        return {
            "loss_trade_ratio": 0.0,
            "worst_trade_stress_pnl": 0.0,
            "worst_five_percent_mean_stress_pnl": 0.0,
        }
    predicted = frame["predicted_up"].to_numpy().astype(bool)
    correct = predicted == frame["label_up"].to_numpy().astype(bool)
    price = np.where(
        predicted, frame["up_ask_vwap_5"].to_numpy(), frame["down_ask_vwap_5"].to_numpy()
    )
    fee = frame["fee_rate"].to_numpy() * price * (1.0 - price)
    stress = (
        correct.astype(float)
        - price
        - fee
        - source.execution.execution_reserve_per_share
        - source.execution.stress_slippage_per_share
    ) * 5
    count = max(1, math.ceil(len(stress) * 0.05))
    return {
        "loss_trade_ratio": float((stress < 0).mean()),
        "worst_trade_stress_pnl": float(stress.min()),
        "worst_five_percent_mean_stress_pnl": float(np.sort(stress)[:count].mean()),
    }


def _stopping_diagnostics(
    selected: pl.DataFrame,
    eligible: pl.DataFrame | None = None,
    incumbent: pl.DataFrame | None = None,
    *,
    eligible_market_count: int | None = None,
) -> dict[str, Any]:
    if eligible_market_count is None:
        eligible_market_count = eligible["market_id"].n_unique() if eligible is not None else 0
    wait_rate = (
        1.0 - selected["market_id"].n_unique() / eligible_market_count
        if eligible_market_count
        else 0.0
    )
    if selected.is_empty():
        return {
            "wait_rate": wait_rate,
            "mean_opportunity_regret_per_share": 0.0,
            "earlier_than_q5_rate": 0.0,
            "average_entry_lead_seconds": None,
        }
    predicted = selected["predicted_up"].to_numpy().astype(bool)
    realized = np.where(
        predicted,
        selected["up_stress_reward"].to_numpy(),
        selected["down_stress_reward"].to_numpy(),
    )
    oracle = np.maximum(
        selected["best_realized_action_value"].to_numpy(),
        selected["future_best_action_value"].to_numpy(),
    )
    common = (
        selected.select("market_id", pl.col("seconds_elapsed").alias("candidate_second")).join(
            incumbent.select("market_id", pl.col("seconds_elapsed").alias("q5_second")),
            on="market_id",
            how="inner",
        )
        if incumbent is not None and not incumbent.is_empty()
        else pl.DataFrame()
    )
    return {
        "wait_rate": wait_rate,
        "mean_opportunity_regret_per_share": float(np.maximum(oracle - realized, 0.0).mean()),
        "earlier_than_q5_rate": float((common["candidate_second"] < common["q5_second"]).mean())
        if common.height
        else 0.0,
        "average_entry_lead_seconds": float(
            (common["q5_second"] - common["candidate_second"]).mean()
        )
        if common.height
        else None,
    }


def _bootstrap_lower(frame: pl.DataFrame, config: TournamentConfig, *, seed: int) -> float:
    if frame.is_empty():
        return 0.0
    predicted = frame["predicted_up"].to_numpy().astype(bool)
    correct = predicted == frame["label_up"].to_numpy().astype(bool)
    price = np.where(
        predicted, frame["up_ask_vwap_5"].to_numpy(), frame["down_ask_vwap_5"].to_numpy()
    )
    fee = frame["fee_rate"].to_numpy() * price * (1.0 - price)
    values = (
        correct.astype(float)
        - price
        - fee
        - config.source.execution.execution_reserve_per_share
        - config.source.execution.stress_slippage_per_share
    ) * 5
    days = frame["window_start"].dt.date().to_numpy()
    unique = np.unique(days)
    groups = [values[days == day] for day in unique]
    rng = np.random.default_rng(seed)
    means = np.empty(config.gates.bootstrap_resamples)
    for index in range(len(means)):
        sampled = rng.choice(len(groups), size=len(groups), replace=True)
        means[index] = np.concatenate([groups[position] for position in sampled]).mean()
    return float(np.quantile(means, 1.0 - config.gates.bootstrap_confidence))


def _champion_rank(result: dict[str, Any]) -> tuple[Any, ...]:
    metrics = result["metrics"]
    return (
        metrics["worst_fold_stress_expectancy"],
        metrics["bootstrap_stress_expectancy_lower"],
        metrics["stress_net_pnl"],
        metrics["strict_market_coverage"],
        -metrics["average_entry_second"],
    )


def _diagnostic_rank(result: dict[str, Any]) -> tuple[Any, ...]:
    metrics = result["metrics"]
    return (
        metrics["stress_net_pnl"] > 0,
        metrics["stress_net_pnl"],
        metrics["profit_factor"] or 0.0,
        metrics["strict_market_coverage"],
    )


def _render_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# BTC Q5 and Optimal-Stopping Tournament",
        "",
        "Status: **strictly offline development comparison; no runtime export**",
        "",
        f"Decision: `{metrics['decision']['status']}`",
        f"Champion: `{metrics['decision']['champion']}`",
        f"Diagnostic leader: `{metrics['decision']['diagnostic_leader']}`",
        "",
        "| Candidate | Qualified | Trades | Coverage | Accuracy | VWAP5 net | Stress net | PF | Payoff | Avg entry |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CANDIDATES:
        result = metrics["candidates"][name]
        row = result["metrics"]
        lines.append(
            f"| {name} | {'yes' if result['qualification']['passed'] else 'no'} | {row['trades']} | "
            f"{row['strict_market_coverage']:.2%} | {row['accuracy']:.2%} | {row['net_pnl']:.2f} | "
            f"{row['stress_net_pnl']:.2f} | {_fmt(row['profit_factor'])} | {_fmt(row['payoff_ratio'])} | {_fmt(row['average_entry_second'])} |"
        )
    lines.extend(["", "## Limitations", "", *(f"- {value}" for value in metrics["limitations"])])
    return "\n".join(lines) + "\n"


def _scheduled_market_count(config: TournamentConfig, manifest: dict[str, Any]) -> int:
    evidence_paths = [
        config.source.paths.capacity_evidence / part["path"] for part in manifest["partitions"]
    ]
    evidence = pl.scan_parquet(evidence_paths)
    eligible = pl.any_horizontal(
        [
            (pl.col("window_start") >= fold.evaluation_start)
            & (pl.col("window_start") < fold.evaluation_end)
            for fold in config.folds
        ]
    )
    return int(evidence.filter(eligible).select(pl.col("market_id").n_unique()).collect().item())


def _serialize_fold(fold: OuterFold) -> dict[str, Any]:
    return {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in asdict(fold).items()
    }


def _serialize_dataclass(value: Any) -> dict[str, Any]:
    return {
        key: item.isoformat() if isinstance(item, datetime) else item
        for key, item in asdict(value).items()
    }


def _bucket_indices(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    return np.clip(
        np.searchsorted(np.asarray(edges[1:-1]), values, side="right"), 0, len(edges) - 2
    )


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _git_revision(package_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=package_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_finite(payload), indent=2, sort_keys=True) + "\n")


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run_dir, metrics = run_tournament(load_config(args.config))
    print(f"run: {run_dir}")
    print(json.dumps(metrics["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
