"""Offline exogenous fair-value challenger tournament for BTC five-minute markets."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .chainlink_oi_features import BINANCE_OI_FEATURES, _attach_open_interest_features
from .continuous_edge_training import (
    CHAINLINK_FEATURES,
    CORE_FEATURES,
    ORACLE_FEATURES,
    PRIMARY_FEATURES,
    _block,
    _matrix,
    coverage_summary,
    extract_capacity_evidence,
    load_training_frame,
    market_equal_weights,
    policy_metrics,
    policy_metrics_by_price_bucket,
)
from .core_extract import file_sha256
from .middle_market_ablation_tournament import (
    LOSS_FEATURES,
)
from .middle_market_ablation_tournament import (
    Candidate as MiddleCandidate,
)
from .middle_market_ablation_tournament import (
    _decision_frame as middle_decision_frame,
)
from .middle_market_ablation_tournament import (
    _fit_correctness as fit_middle_correctness,
)
from .middle_market_ablation_tournament import (
    _fit_probability_modifier as fit_middle_probability_modifier,
)
from .middle_market_ablation_tournament import (
    _fit_regressor as fit_middle_regressor,
)
from .middle_market_ablation_tournament import (
    _loss_training_frame as middle_loss_training_frame,
)
from .middle_market_ablation_tournament import (
    _score_candidate as score_middle_candidate,
)
from .middle_market_ablation_tournament import load_config as load_middle_config
from .middle_market_tournament import _fit_outcome as fit_middle_outcome
from .q5_optimal_stopping_tournament import (
    ENTRY_CELLS,
    OuterFold,
    SelectionCell,
    _attach_action_targets,
    _cell_metrics,
    _direction_metrics,
    _loss_tail,
    _score_incumbent,
    _select_q5,
    _serialize_fold,
    _write_json,
)
from .q5_optimal_stopping_tournament import load_config as load_q5_config

SCHEMA_VERSION = "btc-fair-value-challenger-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-fair-value-challenger-artifact-v1"

INCUMBENTS = (
    "q5_incumbent",
    "chainlink_stratified_payoff",
    "chainlink_full_combined",
    "chainlink_regime_calibrated",
)
CHALLENGERS = (
    "exogenous_fair_value",
    "stratified_fair_value",
    "tail_weighted_fair_value",
    "terminal_margin_fair_value",
    "specialist_distilled_fair_value",
    "groupwise_entry_ranker",
)
CANDIDATES = (*INCUMBENTS, *CHALLENGERS)

# Polymarket prices and book state are intentionally absent. They are used only
# after fair probability estimation to decide whether the executable quote has edge.
EXOGENOUS_FEATURES = tuple(dict.fromkeys((*CORE_FEATURES, *ORACLE_FEATURES, *CHAINLINK_FEATURES)))
RANKING_FEATURES = (
    *EXOGENOUS_FEATURES,
    "probability_selected",
    "selected_cost_5",
    "pm_vwap5_overround",
    "pm_vwap200_overround",
    "pm_depth_imbalance",
    "pm_up_book_age_seconds",
    "pm_down_book_age_seconds",
)


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    tail_weight_multiplier: float
    terminal_lower_quantile: float
    terminal_upper_quantile: float
    local_calibration_shrinkage_markets: int
    ranking_lateness_penalty: float


@dataclass(frozen=True)
class PolicyConfig:
    confidence_thresholds: tuple[float, ...]
    edge_thresholds: tuple[float, ...]
    ranking_thresholds: tuple[float, ...]
    minimum_cell_selection_trades: int


@dataclass(frozen=True)
class GateConfig:
    minimum_directional_trades: int
    minimum_asymmetric_trades: int
    minimum_profit_factor: float
    minimum_asymmetric_payoff_ratio: float
    minimum_profitable_fold_ratio: float
    minimum_bootstrap_lower: float
    bootstrap_resamples: int
    bootstrap_confidence: float
    minimum_populated_cell_trades: int
    minimum_populated_direction_trades: int
    incumbent_coverage_floor_ratio: float
    incumbent_expectancy_improvement_ratio: float


@dataclass(frozen=True)
class FreshReadinessConfig:
    start: datetime
    end: datetime
    minimum_strict_markets: int
    minimum_usable_days: int
    minimum_scheduled_market_coverage: float
    maximum_single_day_market_share: float


@dataclass(frozen=True)
class TournamentPaths:
    source_tournament_config: Path
    middle_tournament_config: Path
    middle_tournament_artifact: Path
    middle_tournament_metrics: Path
    open_interest_evidence: Path
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
    readiness: FreshReadinessConfig
    paths: TournamentPaths
    q5: Any
    source: Any
    middle: Any


def load_config(path: Path) -> TournamentConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    training = raw["training"]
    if training.get("paper_only") is not True or training.get("live_capital_allowed") is not False:
        raise ValueError("fair-value tournament must remain offline and paper-only")
    paths = raw["paths"]
    q5_path = _path(package_root, paths["source_tournament_config"])
    middle_path = _path(package_root, paths["middle_tournament_config"])
    q5 = load_q5_config(q5_path)
    readiness_values = dict(raw["fresh_readiness"])
    readiness_values["start"] = _utc(readiness_values["start"])
    readiness_values["end"] = _utc(readiness_values["end"])
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
        model=ModelConfig(**raw["model"]),
        policy=PolicyConfig(
            confidence_thresholds=tuple(raw["policy"]["confidence_thresholds"]),
            edge_thresholds=tuple(raw["policy"]["edge_thresholds"]),
            ranking_thresholds=tuple(raw["policy"]["ranking_thresholds"]),
            minimum_cell_selection_trades=int(raw["policy"]["minimum_cell_selection_trades"]),
        ),
        gates=GateConfig(**raw["gates"]),
        readiness=FreshReadinessConfig(**readiness_values),
        paths=TournamentPaths(
            source_tournament_config=q5_path,
            middle_tournament_config=middle_path,
            middle_tournament_artifact=_path(package_root, paths["middle_tournament_artifact"]),
            middle_tournament_metrics=_path(package_root, paths["middle_tournament_metrics"]),
            open_interest_evidence=_path(package_root, paths["open_interest_evidence"]),
            runs=_path(package_root, paths["runs"]),
            committed_results=_path(package_root, paths["committed_results"]),
        ),
        q5=q5,
        source=q5.source,
        middle=load_middle_config(middle_path),
    )
    _validate_config(config)
    return config


def _validate_config(config: TournamentConfig) -> None:
    if config.profile != "btc_5m_fair_value_challenger_tournament":
        raise ValueError("unexpected fair-value tournament profile")
    if config.fresh_holdout:
        raise ValueError("consumed development evidence cannot be marked fresh")
    if (
        tuple((cell.name, cell.start_second, cell.end_second_exclusive) for cell in config.cells)
        != ENTRY_CELLS
    ):
        raise ValueError("fair-value entry cells changed")
    prior_end: datetime | None = None
    for fold in config.folds:
        if not (
            fold.fit_end
            <= fold.selection_start
            < fold.selection_end
            <= fold.evaluation_start
            < fold.evaluation_end
        ):
            raise ValueError(f"invalid chronology for {fold.name}")
        if prior_end is not None and fold.evaluation_start < prior_end:
            raise ValueError("outer evaluation folds overlap")
        prior_end = fold.evaluation_end
    forbidden = (
        "ask_vwap",
        "pm_",
        "book_",
        "selected_cost",
        "fee_rate",
    )
    leaked = [name for name in EXOGENOUS_FEATURES if name.startswith(forbidden)]
    if leaked:
        raise ValueError("fair-value estimator contains Polymarket features: " + ", ".join(leaked))
    for path in (
        config.paths.middle_tournament_artifact,
        config.paths.middle_tournament_metrics,
        config.paths.open_interest_evidence,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if config.gates.bootstrap_resamples < 500:
        raise ValueError("bootstrap requires at least 500 resamples")


def run_tournament(config: TournamentConfig) -> tuple[Path, dict[str, Any]]:
    print("load: immutable full-window capacity evidence", flush=True)
    manifest = extract_capacity_evidence(config.source)
    base_frame = load_training_frame(config.source, manifest)
    q5_artifact = joblib.load(config.q5.paths.incumbent_artifact)
    scored = _attach_entry_cells(
        _attach_action_targets(
            _score_incumbent(base_frame, q5_artifact, config.source), config.source
        )
    )
    scored = _attach_oi(scored, config.paths.open_interest_evidence)
    print(
        f"joined: {scored.height:,} strict rows, {scored['market_id'].n_unique():,} markets",
        flush=True,
    )

    fold_models: dict[str, Any] = {}
    fold_results: dict[str, dict[str, Any]] = {name: {} for name in CANDIDATES}
    ledgers: dict[str, list[pl.DataFrame]] = {name: [] for name in CANDIDATES}
    evaluated_markets: set[str] = set()
    for fold_index, fold in enumerate(config.folds):
        print(f"fold: {fold.name} fit fair-value challengers", flush=True)
        fit = scored.filter(pl.col("window_start") < fold.fit_end)
        selection = _block(scored, fold.selection_start, fold.selection_end)
        split = fold.selection_start + (fold.selection_end - fold.selection_start) / 2
        calibration = _block(selection, fold.selection_start, split)
        policy_frame = _block(selection, split, fold.selection_end)
        evaluation = _block(scored, fold.evaluation_start, fold.evaluation_end)
        if (
            min(
                fit["market_id"].n_unique(),
                calibration["market_id"].n_unique(),
                policy_frame["market_id"].n_unique(),
                evaluation["market_id"].n_unique(),
            )
            == 0
        ):
            raise RuntimeError(f"{fold.name} contains an empty chronological block")

        fair_models = _fit_fair_models(
            fit,
            calibration,
            config,
            seed=config.random_seed + 1000 * fold_index,
        )
        middle_models = _fit_middle_replicas(
            fit,
            calibration,
            config,
            seed=config.random_seed + 1000 * fold_index + 500,
        )
        fold_models[fold.name] = {
            "fair_value": fair_models,
            "middle_replicas": middle_models,
        }

        policy_scores = _score_all(
            policy_frame,
            fair_models,
            middle_models,
            q5_artifact,
            config,
        )
        policies = {
            name: _select_policy(
                policy_scores[name], config, ranked=name == "groupwise_entry_ranker"
            )
            for name in CANDIDATES
            if name != "q5_incumbent"
        }
        policies["q5_incumbent"] = {"type": "frozen_q5_policy"}
        evaluation_scores = _score_all(
            evaluation,
            fair_models,
            middle_models,
            q5_artifact,
            config,
        )
        selections: dict[str, pl.DataFrame] = {}
        for name in CANDIDATES:
            selected = (
                evaluation_scores[name]
                if name == "q5_incumbent"
                else _apply_policy(evaluation_scores[name], policies[name])
            )
            selections[name] = selected
            ledger = selected.with_columns(pl.lit(fold.name).alias("outer_fold"))
            ledgers[name].append(ledger)
            fold_results[name][fold.name] = {
                "policy": policies[name],
                "metrics": _fold_metrics(selected, evaluation, config),
            }
        evaluated_markets.update(str(value) for value in evaluation["market_id"].unique())

    combined_ledgers = {
        name: pl.concat(parts, how="diagonal_relaxed") if parts else scored.head(0)
        for name, parts in ledgers.items()
    }
    strict_markets = len(evaluated_markets)
    scheduled_markets = _scheduled_market_count(config, manifest)
    candidates = {
        name: _aggregate_candidate(
            name,
            combined_ledgers[name],
            fold_results[name],
            strict_markets,
            scheduled_markets,
            config,
            seed=config.random_seed + 20_000 + index,
        )
        for index, name in enumerate(CANDIDATES)
    }
    for name in CHALLENGERS:
        candidates[name]["incumbent_comparisons"] = {
            incumbent: _dominance(candidates[name], candidates[incumbent], config)
            for incumbent in INCUMBENTS
        }
        candidates[name]["overlap"] = {
            incumbent: _overlap(combined_ledgers[name], combined_ledgers[incumbent], config)
            for incumbent in INCUMBENTS
        }

    qualified = [
        name
        for name in CHALLENGERS
        if candidates[name]["qualification"]["passed"]
        and any(row["beat_incumbent"] for row in candidates[name]["incumbent_comparisons"].values())
    ]
    champion = max(qualified, key=lambda name: _rank(candidates[name])) if qualified else None
    diagnostic_leader = max(CHALLENGERS, key=lambda name: _rank(candidates[name]))
    readiness = _fresh_readiness(config)
    decision_status = (
        "fresh_holdout_not_ready"
        if not readiness["passed"]
        else "qualified_incumbent_beater"
        if champion
        else "no_qualified_challenger"
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    temporary = config.paths.runs / f"{run_id}.partial"
    final = config.paths.committed_results / run_id
    if temporary.exists() or final.exists():
        raise FileExistsError(run_id)
    temporary.mkdir(parents=True)
    artifact_path = temporary / "tournament.joblib"
    source_commit = _git_revision(config.package_root)
    joblib.dump(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "profile": config.profile,
            "source_commit": source_commit,
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
        "source_commit": source_commit,
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
            "model": asdict(config.model),
            "policy": asdict(config.policy),
            "gates": asdict(config.gates),
            "execution": asdict(config.source.execution),
        },
        "data": {
            "rows": scored.height,
            "markets": scored["market_id"].n_unique(),
            "evaluated_strict_markets": strict_markets,
            "evaluated_scheduled_markets": scheduled_markets,
            "strict_data_coverage": strict_markets / scheduled_markets,
            "capacity_manifest": manifest,
            "coverage": coverage_summary(scored, config.source),
            "open_interest_rows": scored.drop_nulls(BINANCE_OI_FEATURES).height,
            "open_interest_markets": scored.drop_nulls(BINANCE_OI_FEATURES)["market_id"].n_unique(),
            "twap_used": False,
            "polymarket_features_in_fair_value_estimator": False,
        },
        "fresh_readiness": readiness,
        "candidates": candidates,
        "fold_results": fold_results,
        "decision": {
            "status": decision_status,
            "champion": champion,
            "diagnostic_leader": diagnostic_leader,
            "production_selection_allowed": False,
        },
        "model_artifact": {
            "path": artifact_path.name,
            "sha256": file_sha256(artifact_path),
        },
        "limitations": [
            "Development evidence through August 2 is consumed and is not a fresh holdout.",
            "The Q5 control is the exact frozen deployed artifact; Chainlink controls are fold-specific algorithm replicas.",
            "The full-combined replica is eligible only where causal OI evidence exists.",
            "Terminal-margin targets use the final available Binance path margin and calibrate to canonical labels; no TWAP is used.",
            "Projected PnL assumes recorded ask VWAP was fillable and does not model queue position.",
            "The tournament exports no runtime model or trading-process configuration.",
        ],
    }
    _write_json(temporary / "metrics.json", metrics)
    (temporary / "report.md").write_text(_render_report(metrics))
    (temporary / "tournament.sha256").write_text(file_sha256(artifact_path) + "\n")
    config.paths.committed_results.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    return final, metrics


def _fit_fair_models(
    fit: pl.DataFrame,
    calibration: pl.DataFrame,
    config: TournamentConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    base = _fit_probability_model(fit, calibration, EXOGENOUS_FEATURES, config, seed=seed)
    stratified = _fit_probability_model(
        fit,
        calibration,
        EXOGENOUS_FEATURES,
        config,
        seed=seed,
        stratified=True,
    )
    hard_weights = _hard_negative_weights(fit, config, seed=seed + 10)
    tail = _fit_probability_model(
        fit,
        calibration,
        EXOGENOUS_FEATURES,
        config,
        seed=seed + 20,
        fit_weights=hard_weights,
    )
    margin = _fit_margin_model(fit, calibration, config, seed=seed + 30)

    specialist_probabilities = np.column_stack(
        (
            _score_probability(base, calibration),
            _score_probability(stratified, calibration),
            _score_probability(tail, calibration),
            _score_margin_probability(margin, calibration),
        )
    )
    selector_matrix = _selector_matrix(specialist_probabilities, calibration)
    selector = LogisticRegression(C=0.5, max_iter=2000, random_state=seed + 40)
    selector.fit(
        selector_matrix,
        calibration["label_up"].to_numpy(),
        sample_weight=market_equal_weights(calibration),
    )
    teacher = selector.predict_proba(selector_matrix)[:, 1]
    distilled = _new_regressor(config, seed + 41)
    distilled.fit(
        _matrix(calibration, EXOGENOUS_FEATURES),
        teacher,
        sample_weight=market_equal_weights(calibration),
    )

    ranked_fit = _score_action(fit, _score_probability(base, fit), config)
    reward = (
        np.where(
            ranked_fit["predicted_up"].to_numpy(),
            ranked_fit["up_stress_reward"].to_numpy(),
            ranked_fit["down_stress_reward"].to_numpy(),
        )
        - config.model.ranking_lateness_penalty * ranked_fit["seconds_elapsed"].to_numpy()
    )
    ranked_fit = ranked_fit.with_columns(pl.Series("_rank_reward", reward)).with_columns(
        (pl.col("_rank_reward") - pl.col("_rank_reward").mean().over("market_id")).alias(
            "_rank_target"
        )
    )
    ranker = _new_regressor(config, seed + 50)
    ranker.fit(
        _matrix(ranked_fit, RANKING_FEATURES),
        ranked_fit["_rank_target"].to_numpy(),
        sample_weight=market_equal_weights(ranked_fit),
    )
    return {
        "base": base,
        "stratified": stratified,
        "tail": tail,
        "margin": margin,
        "selector": selector,
        "distilled": distilled,
        "ranker": ranker,
        "feature_names": EXOGENOUS_FEATURES,
    }


def _fit_probability_model(
    fit: pl.DataFrame,
    calibration: pl.DataFrame,
    features: tuple[str, ...],
    config: TournamentConfig,
    *,
    seed: int,
    fit_weights: np.ndarray | None = None,
    stratified: bool = False,
) -> dict[str, Any]:
    estimator = _new_classifier(config, seed)
    estimator.fit(
        _matrix(fit, features),
        fit["label_up"].to_numpy(),
        sample_weight=fit_weights if fit_weights is not None else market_equal_weights(fit),
    )
    raw = estimator.predict_proba(_matrix(calibration, features))[:, 1]
    calibration_model = _fit_calibrator(raw, calibration, seed=seed + 1)
    model: dict[str, Any] = {
        "estimator": estimator,
        "calibrator": calibration_model,
        "features": features,
        "locals": {},
    }
    if stratified:
        for cell in ENTRY_CELLS:
            name, start, end = cell
            subset = calibration.filter(
                (pl.col("seconds_elapsed") >= start) & (pl.col("seconds_elapsed") < end)
            )
            if subset["market_id"].n_unique() < 50 or subset.height < 200:
                continue
            indices = np.flatnonzero(
                (calibration["seconds_elapsed"].to_numpy() >= start)
                & (calibration["seconds_elapsed"].to_numpy() < end)
            )
            local = _fit_calibrator(raw[indices], subset, seed=seed + 10 + len(model["locals"]))
            markets = subset["market_id"].n_unique()
            weight = markets / (markets + config.model.local_calibration_shrinkage_markets)
            model["locals"][name] = {"calibrator": local, "weight": weight}
    return model


def _fit_calibrator(raw: np.ndarray, frame: pl.DataFrame, *, seed: int) -> LogisticRegression:
    estimator = LogisticRegression(C=0.5, max_iter=2000, random_state=seed)
    estimator.fit(
        _logit(raw).reshape(-1, 1),
        frame["label_up"].to_numpy(),
        sample_weight=market_equal_weights(frame),
    )
    return estimator


def _score_probability(model: dict[str, Any], frame: pl.DataFrame) -> np.ndarray:
    raw = model["estimator"].predict_proba(_matrix(frame, model["features"]))[:, 1]
    global_probability = model["calibrator"].predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
    probability = global_probability.copy()
    seconds = frame["seconds_elapsed"].to_numpy()
    for name, start, end in ENTRY_CELLS:
        local = model["locals"].get(name)
        if local is None:
            continue
        mask = (seconds >= start) & (seconds < end)
        local_probability = local["calibrator"].predict_proba(_logit(raw[mask]).reshape(-1, 1))[
            :, 1
        ]
        probability[mask] = (
            local["weight"] * local_probability + (1.0 - local["weight"]) * global_probability[mask]
        )
    return np.clip(probability, 1e-6, 1 - 1e-6)


def _hard_negative_weights(
    frame: pl.DataFrame, config: TournamentConfig, *, seed: int
) -> np.ndarray:
    starts = np.sort(frame["window_start"].unique().to_numpy())
    boundaries = [starts[int(len(starts) * ratio)] for ratio in (0.50, 0.75)]
    probability = np.full(frame.height, np.nan)
    windows = (
        (None, boundaries[0], boundaries[1]),
        (None, boundaries[1], starts[-1] + np.timedelta64(1, "s")),
    )
    for index, (_, train_end, score_end) in enumerate(windows):
        train_mask = frame["window_start"].to_numpy() < train_end
        score_mask = (frame["window_start"].to_numpy() >= train_end) & (
            frame["window_start"].to_numpy() < score_end
        )
        if not score_mask.any():
            continue
        model = _new_classifier(config, seed + index)
        model.fit(
            _matrix(frame.filter(pl.Series(train_mask)), EXOGENOUS_FEATURES),
            frame.filter(pl.Series(train_mask))["label_up"].to_numpy(),
            sample_weight=market_equal_weights(frame.filter(pl.Series(train_mask))),
        )
        probability[score_mask] = model.predict_proba(
            _matrix(frame.filter(pl.Series(score_mask)), EXOGENOUS_FEATURES)
        )[:, 1]
    weights = market_equal_weights(frame)
    available = np.isfinite(probability)
    predicted = probability >= 0.5
    correct = predicted == frame["label_up"].to_numpy().astype(bool)
    confidence = np.maximum(probability, 1.0 - probability)
    selected_cost = np.where(
        predicted,
        frame["up_cost_5"].to_numpy(),
        frame["down_cost_5"].to_numpy(),
    )
    hard = available & (~correct) & (confidence >= 0.65)
    weights *= 1.0 + config.model.tail_weight_multiplier * selected_cost * hard
    return weights


def _fit_margin_model(
    fit: pl.DataFrame,
    calibration: pl.DataFrame,
    config: TournamentConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    target = _terminal_margin_targets(fit)
    models = {
        "lower": _new_regressor(
            config, seed, loss="quantile", quantile=config.model.terminal_lower_quantile
        ),
        "median": _new_regressor(config, seed + 1, loss="absolute_error"),
        "upper": _new_regressor(
            config, seed + 2, loss="quantile", quantile=config.model.terminal_upper_quantile
        ),
    }
    matrix = _matrix(fit, EXOGENOUS_FEATURES)
    weights = market_equal_weights(fit)
    for model in models.values():
        model.fit(matrix, target, sample_weight=weights)
    raw = _raw_margin_probability(models, calibration)
    return {
        "models": models,
        "features": EXOGENOUS_FEATURES,
        "calibrator": _fit_calibrator(raw, calibration, seed=seed + 3),
    }


def _terminal_margin_targets(frame: pl.DataFrame) -> np.ndarray:
    terminals = (
        frame.sort(["market_id", "seconds_elapsed", "observed_at"])
        .group_by("market_id", maintain_order=True)
        .agg(pl.col("btc_path_from_window_open_bps").last().alias("_terminal_margin"))
    )
    return (
        frame.select("market_id")
        .join(terminals, on="market_id", how="left")["_terminal_margin"]
        .to_numpy()
    )


def _raw_margin_probability(models: dict[str, Any], frame: pl.DataFrame) -> np.ndarray:
    matrix = _matrix(frame, EXOGENOUS_FEATURES)
    lower = models["lower"].predict(matrix)
    median = models["median"].predict(matrix)
    upper = models["upper"].predict(matrix)
    sigma = np.maximum(np.abs(upper - lower) / 1.6832, 0.25)
    return np.clip(ndtr(median / sigma), 1e-6, 1 - 1e-6)


def _score_margin_probability(model: dict[str, Any], frame: pl.DataFrame) -> np.ndarray:
    raw = _raw_margin_probability(model["models"], frame)
    return model["calibrator"].predict_proba(_logit(raw).reshape(-1, 1))[:, 1]


def _selector_matrix(probabilities: np.ndarray, frame: pl.DataFrame) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.column_stack(
        (
            _logit(clipped),
            clipped.std(axis=1),
            clipped.max(axis=1) - clipped.min(axis=1),
            frame["seconds_elapsed_scaled"].to_numpy(),
            frame["btc_cross_venue_boundary_gap_bps"].to_numpy(),
        )
    )


def _fit_middle_replicas(
    fit: pl.DataFrame,
    calibration: pl.DataFrame,
    config: TournamentConfig,
    *,
    seed: int,
) -> dict[str, MiddleCandidate | None]:
    features = tuple(dict.fromkeys((*PRIMARY_FEATURES, *CHAINLINK_FEATURES)))
    middle_fit = fit.filter((pl.col("seconds_elapsed") >= 90) & (pl.col("seconds_elapsed") < 180))
    middle_calibration = calibration.filter(
        (pl.col("seconds_elapsed") >= 90) & (pl.col("seconds_elapsed") < 180)
    )
    chain_fit = middle_fit.drop_nulls(CHAINLINK_FEATURES)
    chain_calibration = middle_calibration.drop_nulls(CHAINLINK_FEATURES)
    outcome = fit_middle_outcome(
        chain_fit,
        chain_calibration,
        features,
        seed,
        (chain_fit["window_start"].min(), chain_fit["window_start"].max()),
        (chain_calibration["window_start"].min(), chain_calibration["window_start"].max()),
        minimum_fit_markets=500,
        minimum_calibration_markets=100,
    )
    base_calibration = middle_decision_frame(chain_calibration, outcome, config.middle, None)
    stratified = fit_middle_correctness(
        base_calibration, config.middle, stratified=True, regime=False, seed=seed + 1
    )
    regime = fit_middle_correctness(
        base_calibration, config.middle, stratified=True, regime=True, seed=seed + 2
    )
    output: dict[str, MiddleCandidate | None] = {
        "chainlink_stratified_payoff": MiddleCandidate(
            "chainlink_stratified_payoff", features, CHAINLINK_FEATURES, outcome, stratified
        ),
        "chainlink_regime_calibrated": MiddleCandidate(
            "chainlink_regime_calibrated", features, CHAINLINK_FEATURES, outcome, regime
        ),
        "chainlink_full_combined": None,
    }
    oi_calibration = chain_calibration.drop_nulls(BINANCE_OI_FEATURES)
    if oi_calibration["market_id"].n_unique() < 100:
        return output
    modifier = fit_middle_probability_modifier(
        middle_decision_frame(oi_calibration, outcome, config.middle, None), config.middle
    )
    modified = middle_decision_frame(oi_calibration, outcome, config.middle, modifier)
    correctness = fit_middle_correctness(
        modified, config.middle, stratified=True, regime=True, seed=seed + 3
    )
    loss_training = middle_loss_training_frame(modified, config.middle)
    loss_features = tuple(
        name
        for name in LOSS_FEATURES
        if name in loss_training.columns and loss_training[name].n_unique() > 1
    )
    loss_model = fit_middle_regressor(
        loss_training,
        loss_features,
        "loss_severity_target",
        config.middle,
        seed=seed + 4,
    )
    output["chainlink_full_combined"] = MiddleCandidate(
        "chainlink_full_combined",
        features,
        (*CHAINLINK_FEATURES, *BINANCE_OI_FEATURES),
        outcome,
        correctness,
        probability_modifier=modifier,
        loss_model=loss_model,
        loss_feature_names=loss_features,
    )
    return output


def _score_all(
    frame: pl.DataFrame,
    fair: dict[str, Any],
    middle: dict[str, MiddleCandidate | None],
    q5_artifact: dict[str, Any],
    config: TournamentConfig,
) -> dict[str, pl.DataFrame]:
    base_probability = _score_probability(fair["base"], frame)
    stratified_probability = _score_probability(fair["stratified"], frame)
    tail_probability = _score_probability(fair["tail"], frame)
    margin_probability = _score_margin_probability(fair["margin"], frame)
    distilled_probability = np.clip(
        fair["distilled"].predict(_matrix(frame, EXOGENOUS_FEATURES)), 1e-6, 1 - 1e-6
    )
    output = {
        "q5_incumbent": _select_q5(frame, q5_artifact, config.q5),
        "exogenous_fair_value": _score_action(frame, base_probability, config),
        "stratified_fair_value": _score_action(frame, stratified_probability, config),
        "tail_weighted_fair_value": _score_action(frame, tail_probability, config),
        "terminal_margin_fair_value": _score_action(frame, margin_probability, config),
        "specialist_distilled_fair_value": _score_action(frame, distilled_probability, config),
    }
    ranked = _score_action(frame, base_probability, config)
    output["groupwise_entry_ranker"] = ranked.with_columns(
        pl.Series("candidate_rank", fair["ranker"].predict(_matrix(ranked, RANKING_FEATURES)))
    )
    middle_frame = frame.filter(
        (pl.col("seconds_elapsed") >= 90) & (pl.col("seconds_elapsed") < 180)
    )
    for name in INCUMBENTS[1:]:
        candidate = middle.get(name)
        if candidate is None:
            output[name] = _empty_scored(frame)
            continue
        scored = score_middle_candidate(middle_frame, candidate, config.middle)
        output[name] = scored.with_columns(
            pl.col("lower_correctness_probability").alias("candidate_confidence"),
            pl.col("stress_edge_lower_bound").alias("candidate_edge"),
            pl.lit(1.0).alias("candidate_rank"),
        )
    return output


def _score_action(
    frame: pl.DataFrame, probability_up: np.ndarray, config: TournamentConfig
) -> pl.DataFrame:
    predicted_up = probability_up >= 0.5
    confidence = np.where(predicted_up, probability_up, 1.0 - probability_up)
    cost = np.where(predicted_up, frame["up_cost_5"].to_numpy(), frame["down_cost_5"].to_numpy())
    return frame.with_columns(
        pl.Series("probability_up", probability_up),
        pl.Series("predicted_up", predicted_up),
        pl.Series("probability_selected", confidence),
        pl.Series("selected_cost_5", cost),
        pl.Series("direction_correct", predicted_up == frame["label_up"].to_numpy().astype(bool)),
        pl.Series("candidate_confidence", confidence),
        pl.Series(
            "candidate_edge",
            confidence - cost - config.source.execution.stress_slippage_per_share,
        ),
        pl.lit(1.0).alias("candidate_rank"),
    )


def _select_policy(
    frame: pl.DataFrame, config: TournamentConfig, *, ranked: bool
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for cell in config.cells:
        subset = frame.filter(
            (pl.col("seconds_elapsed") >= cell.start_second)
            & (pl.col("seconds_elapsed") < cell.end_second_exclusive)
        )
        if subset.is_empty():
            policies[cell.name] = {"enabled": False}
            continue
        rank_thresholds = config.policy.ranking_thresholds if ranked else (-math.inf,)
        records: list[dict[str, Any]] = []
        for confidence, edge, rank_threshold in product(
            config.policy.confidence_thresholds,
            config.policy.edge_thresholds,
            rank_thresholds,
        ):
            selected = _first_qualified(
                subset,
                confidence=confidence,
                edge=edge,
                rank_threshold=rank_threshold,
            )
            metrics = policy_metrics(selected, config.source, quantity=5)
            valid = (
                metrics["trades"] >= config.policy.minimum_cell_selection_trades
                and metrics["accuracy"] >= cell.minimum_accuracy
                and metrics["stress_net_pnl"] > 0
            )
            records.append(
                {
                    "enabled": True,
                    "confidence": confidence,
                    "edge": edge,
                    "rank": rank_threshold,
                    "valid": valid,
                    "metrics": metrics,
                }
            )
        chosen = max(
            records,
            key=lambda row: (
                row["valid"],
                row["metrics"]["stress_net_pnl"] > 0,
                row["metrics"]["stress_net_pnl"],
                row["metrics"]["trades"],
                row["metrics"]["accuracy"],
            ),
        )
        policies[cell.name] = {
            key: chosen[key] for key in ("enabled", "confidence", "edge", "rank")
        }
    return {"type": "cell_first_crossing", "cells": policies}


def _apply_policy(frame: pl.DataFrame, policy: dict[str, Any]) -> pl.DataFrame:
    pieces: list[pl.DataFrame] = []
    for name, start, end in ENTRY_CELLS:
        row = policy["cells"][name]
        if not row.get("enabled"):
            continue
        subset = frame.filter(
            (pl.col("seconds_elapsed") >= start) & (pl.col("seconds_elapsed") < end)
        )
        pieces.append(
            _first_qualified(
                subset,
                confidence=row["confidence"],
                edge=row["edge"],
                rank_threshold=row["rank"],
            )
        )
    if not pieces:
        return frame.head(0)
    return (
        pl.concat(pieces, how="diagonal_relaxed")
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .unique("market_id", keep="first", maintain_order=True)
    )


def _first_qualified(
    frame: pl.DataFrame, *, confidence: float, edge: float, rank_threshold: float
) -> pl.DataFrame:
    return (
        frame.filter(
            (pl.col("candidate_confidence") >= confidence)
            & (pl.col("candidate_edge") >= edge)
            & (pl.col("candidate_rank") >= rank_threshold)
        )
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .unique("market_id", keep="first", maintain_order=True)
    )


def _empty_scored(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.head(0).with_columns(
        pl.lit(None).cast(pl.Float64).alias("candidate_confidence"),
        pl.lit(None).cast(pl.Float64).alias("candidate_edge"),
        pl.lit(None).cast(pl.Float64).alias("candidate_rank"),
    )


def _attach_entry_cells(frame: pl.DataFrame) -> pl.DataFrame:
    cell = pl.lit(None).cast(pl.String)
    middle = pl.lit(None).cast(pl.String)
    for name, start, end in reversed(ENTRY_CELLS):
        cell = (
            pl.when((pl.col("seconds_elapsed") >= start) & (pl.col("seconds_elapsed") < end))
            .then(pl.lit(name))
            .otherwise(cell)
        )
    for name, start, end in (("90-119", 90, 120), ("120-149", 120, 150), ("150-179", 150, 180)):
        middle = (
            pl.when((pl.col("seconds_elapsed") >= start) & (pl.col("seconds_elapsed") < end))
            .then(pl.lit(name))
            .otherwise(middle)
        )
    return frame.with_columns(cell.alias("entry_cell"), middle.alias("middle_cell"))


def _attach_oi(frame: pl.DataFrame, path: Path) -> pl.DataFrame:
    qualified = _attach_open_interest_features(frame, pl.read_parquet(path), max_age_seconds=300)
    keys = ("market_id", "window_start", "observed_at", "seconds_elapsed", "label_up")
    features = qualified.select(*keys, *BINANCE_OI_FEATURES)
    return frame.join(features, on=list(keys), how="left", validate="1:1")


def _fold_metrics(
    selected: pl.DataFrame, eligible: pl.DataFrame, config: TournamentConfig
) -> dict[str, Any]:
    metrics = policy_metrics(selected, config.source, quantity=5)
    markets = eligible["market_id"].n_unique()
    metrics["strict_market_coverage"] = (
        selected["market_id"].n_unique() / markets if markets else 0.0
    )
    return metrics


def _aggregate_candidate(
    name: str,
    ledger: pl.DataFrame,
    folds: dict[str, Any],
    strict_markets: int,
    scheduled_markets: int,
    config: TournamentConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    metrics = policy_metrics(ledger, config.source, quantity=5)
    markets = ledger["market_id"].n_unique()
    metrics["strict_market_coverage"] = markets / strict_markets if strict_markets else 0.0
    metrics["end_to_end_market_coverage"] = (
        markets / scheduled_markets if scheduled_markets else 0.0
    )
    metrics["strict_data_coverage"] = strict_markets / scheduled_markets
    metrics["bootstrap_stress_expectancy_lower"] = _bootstrap_lower(ledger, config, seed=seed)
    profitable = [row["metrics"]["stress_net_pnl"] > 0 for row in folds.values()]
    metrics["profitable_fold_ratio"] = sum(profitable) / len(profitable)
    metrics["worst_fold_stress_expectancy"] = min(
        row["metrics"]["stress_expectancy_per_trade"] for row in folds.values()
    )
    cells = _cell_metrics(ledger, config.source)
    directions = _direction_metrics(ledger, config.source)
    capacity = {
        str(quantity): policy_metrics(ledger, config.source, quantity=quantity)
        for quantity in config.source.execution.quantities
    }
    return {
        "metrics": metrics,
        "capacity": capacity,
        "cells": cells,
        "directions": directions,
        "price_buckets": policy_metrics_by_price_bucket(ledger, config.source, quantity=5),
        "loss_tail": _loss_tail(ledger, config.source),
        "qualification": _qualification(name, metrics, cells, directions, capacity, config),
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
    universal = {
        "positive_stress_pnl": metrics["stress_net_pnl"] > 0,
        "positive_q10_stress": capacity["10"]["stress_net_pnl"] > 0,
        "positive_q20_stress": capacity["20"]["stress_net_pnl"] > 0,
        "bootstrap_lower_positive": metrics["bootstrap_stress_expectancy_lower"]
        > config.gates.minimum_bootstrap_lower,
        "minimum_profitable_fold_ratio": metrics["profitable_fold_ratio"]
        >= config.gates.minimum_profitable_fold_ratio,
        "populated_cells_nonnegative": all(row["stress_net_pnl"] >= 0 for row in populated_cells),
        "populated_directions_nonnegative": all(
            row["stress_net_pnl"] >= 0 for row in populated_directions
        ),
    }
    directional_accuracy = all(
        cells[cell.name]["trades"] < config.gates.minimum_populated_cell_trades
        or cells[cell.name]["accuracy"] >= cell.minimum_accuracy
        for cell in config.cells
    )
    directional = {
        "minimum_trades": metrics["trades"] >= config.gates.minimum_directional_trades,
        "minimum_profit_factor": (metrics["profit_factor"] or 0.0)
        >= config.gates.minimum_profit_factor,
        "cell_accuracy_targets": directional_accuracy,
    }
    asymmetric = {
        "minimum_trades": metrics["trades"] >= config.gates.minimum_asymmetric_trades,
        "minimum_profit_factor": (metrics["profit_factor"] or 0.0)
        >= config.gates.minimum_profit_factor,
        "minimum_payoff_ratio": metrics["payoff_ratio"]
        >= config.gates.minimum_asymmetric_payoff_ratio,
    }
    track = (
        "directional"
        if all(directional.values())
        else "asymmetric"
        if all(asymmetric.values())
        else None
    )
    passed = name in CHALLENGERS and all(universal.values()) and track is not None
    return {
        "candidate": name,
        "passed": passed,
        "track": track,
        "universal": universal,
        "directional": directional,
        "asymmetric": asymmetric,
    }


def _dominance(
    challenger: dict[str, Any], incumbent: dict[str, Any], config: TournamentConfig
) -> dict[str, Any]:
    left = challenger["metrics"]
    right = incumbent["metrics"]
    coverage = left["strict_market_coverage"] >= (
        config.gates.incumbent_coverage_floor_ratio * right["strict_market_coverage"]
    )
    pnl = left["stress_net_pnl"] >= right["stress_net_pnl"]
    required_expectancy = config.gates.incumbent_expectancy_improvement_ratio * max(
        right["stress_expectancy_per_trade"], 0.0
    )
    expectancy = left["stress_expectancy_per_trade"] >= required_expectancy
    accuracy = left["accuracy"] >= right["accuracy"]
    beat = bool(
        challenger["qualification"]["passed"] and coverage and pnl and (expectancy or accuracy)
    )
    return {
        "beat_incumbent": beat,
        "coverage_floor_passed": coverage,
        "stress_pnl_passed": pnl,
        "expectancy_improvement_passed": expectancy,
        "accuracy_passed": accuracy,
        "stress_pnl_delta": left["stress_net_pnl"] - right["stress_net_pnl"],
        "coverage_delta": left["strict_market_coverage"] - right["strict_market_coverage"],
        "accuracy_delta": left["accuracy"] - right["accuracy"],
    }


def _overlap(
    challenger: pl.DataFrame, incumbent: pl.DataFrame, config: TournamentConfig
) -> dict[str, Any]:
    incumbent_markets = set(incumbent["market_id"].to_list())
    challenger_markets = set(challenger["market_id"].to_list())
    shared = challenger.filter(pl.col("market_id").is_in(list(incumbent_markets)))
    incremental = challenger.filter(~pl.col("market_id").is_in(list(incumbent_markets)))
    return {
        "shared_markets": len(challenger_markets & incumbent_markets),
        "incremental_markets": len(challenger_markets - incumbent_markets),
        "shared": policy_metrics(shared, config.source, quantity=5),
        "incremental": policy_metrics(incremental, config.source, quantity=5),
    }


def _bootstrap_lower(frame: pl.DataFrame, config: TournamentConfig, *, seed: int) -> float:
    if frame.is_empty():
        return 0.0
    stress = np.where(
        frame["direction_correct"].to_numpy(),
        1.0
        - frame["selected_cost_5"].to_numpy()
        - config.source.execution.stress_slippage_per_share,
        -frame["selected_cost_5"].to_numpy() - config.source.execution.stress_slippage_per_share,
    )
    days = frame["window_start"].dt.date().to_numpy()
    unique = np.unique(days)
    grouped = [stress[days == day] for day in unique]
    rng = np.random.default_rng(seed)
    samples = np.empty(config.gates.bootstrap_resamples)
    for index in range(config.gates.bootstrap_resamples):
        chosen = rng.integers(0, len(grouped), len(grouped))
        values = np.concatenate([grouped[item] for item in chosen])
        samples[index] = values.mean()
    return float(np.quantile(samples, 1.0 - config.gates.bootstrap_confidence))


def _fresh_readiness(config: TournamentConfig) -> dict[str, Any]:
    source = json.loads(config.paths.middle_tournament_metrics.read_text())["holdout_readiness"]
    usable_days = sum(row["canonical_markets"] > 0 for row in source["daily"])
    checks = {
        "minimum_strict_markets": source["usable_canonical_markets"]
        >= config.readiness.minimum_strict_markets,
        "minimum_usable_days": usable_days >= config.readiness.minimum_usable_days,
        "minimum_scheduled_market_coverage": source["market_coverage"]
        >= config.readiness.minimum_scheduled_market_coverage,
        "maximum_single_day_market_share": source["maximum_single_day_market_share"]
        <= config.readiness.maximum_single_day_market_share,
        "labels_unopened": source["holdout_labels_accessed"] is False,
    }
    return {
        "range_start": config.readiness.start.isoformat(),
        "range_end": config.readiness.end.isoformat(),
        "strict_markets": source["usable_canonical_markets"],
        "scheduled_markets": source["scheduled_markets"],
        "market_coverage": source["market_coverage"],
        "usable_days": usable_days,
        "maximum_single_day_market_share": source["maximum_single_day_market_share"],
        "labels_accessed": source["holdout_labels_accessed"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def _scheduled_market_count(config: TournamentConfig, manifest: dict[str, Any]) -> int:
    paths = [config.source.paths.capacity_evidence / row["path"] for row in manifest["partitions"]]
    eligible = pl.any_horizontal(
        [
            (pl.col("window_start") >= fold.evaluation_start)
            & (pl.col("window_start") < fold.evaluation_end)
            for fold in config.folds
        ]
    )
    return int(
        pl.scan_parquet(paths)
        .filter(eligible)
        .select(pl.col("market_id").n_unique())
        .collect()
        .item()
    )


def _new_classifier(config: TournamentConfig, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=config.model.learning_rate,
        max_iter=config.model.max_iter,
        max_leaf_nodes=config.model.max_leaf_nodes,
        min_samples_leaf=config.model.min_samples_leaf,
        l2_regularization=config.model.l2_regularization,
        random_state=seed,
        early_stopping=False,
    )


def _new_regressor(
    config: TournamentConfig,
    seed: int,
    *,
    loss: str = "squared_error",
    quantile: float | None = None,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss,
        quantile=quantile,
        learning_rate=config.model.learning_rate,
        max_iter=config.model.max_iter,
        max_leaf_nodes=config.model.max_leaf_nodes,
        min_samples_leaf=config.model.min_samples_leaf,
        l2_regularization=config.model.l2_regularization,
        random_state=seed,
        early_stopping=False,
    )


def _rank(result: dict[str, Any]) -> tuple[Any, ...]:
    metrics = result["metrics"]
    return (
        result["qualification"]["passed"],
        metrics["bootstrap_stress_expectancy_lower"],
        metrics["stress_net_pnl"],
        metrics["profit_factor"] or 0.0,
        metrics["strict_market_coverage"],
    )


def _render_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# BTC Five-Minute Fair-Value Challenger Tournament",
        "",
        "Status: **offline development comparison; no runtime export**",
        "",
        f"Decision: `{metrics['decision']['status']}`",
        f"Champion: `{metrics['decision']['champion']}`",
        f"Diagnostic leader: `{metrics['decision']['diagnostic_leader']}`",
        "",
        "| Candidate | Qualified | Trades | Coverage | Accuracy | VWAP5 net | Stress net | PF | Payoff | Avg entry |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CANDIDATES:
        row = metrics["candidates"][name]
        values = row["metrics"]
        lines.append(
            f"| {name} | {'yes' if row['qualification']['passed'] else 'no'} | "
            f"{values['trades']} | {values['strict_market_coverage']:.2%} | "
            f"{values['accuracy']:.2%} | {values['net_pnl']:.2f} | "
            f"{values['stress_net_pnl']:.2f} | {_fmt(values['profit_factor'])} | "
            f"{_fmt(values['payoff_ratio'])} | {_fmt(values['average_entry_second'])} |"
        )
    lines.extend(
        [
            "",
            "## Fresh holdout readiness",
            "",
            (
                f"Passed: **{metrics['fresh_readiness']['passed']}**; coverage: "
                f"{metrics['fresh_readiness']['market_coverage']:.2%}."
            ),
            "",
            "## Limitations",
            "",
            *(f"- {value}" for value in metrics["limitations"]),
        ]
    )
    return "\n".join(lines) + "\n"


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.3f}"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run_dir, metrics = run_tournament(load_config(args.config))
    print(f"run: {run_dir}")
    print(json.dumps(metrics["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
