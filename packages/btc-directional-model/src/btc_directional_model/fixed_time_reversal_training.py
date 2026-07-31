from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import log_loss

from .core_config import CoreTrainingConfig
from .core_evaluation import classification_metrics, scored_prediction_rows
from .core_training import (
    MARKET_EQUAL_ROW_WEIGHT_POLICY,
    CandidateSpec,
    FittedCoreModel,
    candidate_training_weights,
    chronological_inner_split,
    estimator_converged,
    fit_model,
    histogram_parameters,
)
from .fixed_time_config import FIXED_TIME_TARGET
from .fixed_time_reversal_config import (
    BASE_PARAMETER_GRID,
    PATH_PERSISTENCE_TARGET,
    TAIL_PARAMETER_GRID,
    FixedTimeReversalCandidateConfig,
    FixedTimeReversalModelConfig,
)
from .persistence_benchmark import (
    hard_confident_error_metrics,
    path_is_directionally_eligible,
    persistence_target_labels,
    target_probability_to_up,
)

REVERSAL_ESTIMATOR_TRAINING_SECONDS = (120, 125, 130, 135, 140)
REVERSAL_DECISION_SECOND = 120
PRIMARY_REVERSAL_COVERAGE = 0.10
DIAGNOSTIC_REVERSAL_COVERAGE = 0.08
HARD_CONFIDENCE_FLOOR = 0.95
INNER_VALIDATION_FRACTION = 0.20


def reversal_candidate_spec(
    candidate_config: FixedTimeReversalCandidateConfig,
    model_config: FixedTimeReversalModelConfig,
) -> CandidateSpec:
    if not math.isclose(candidate_config.market_weight_multiplier, 1.0):
        raise ValueError("fixed-time reversal candidates require 1x market weighting")
    return CandidateSpec(
        name=candidate_config.name,
        family="histogram",
        feature_names=tuple(candidate_config.feature_names),
        row_weight_policy=MARKET_EQUAL_ROW_WEIGHT_POLICY,
        recency_half_life_days=model_config.recency_half_life_days,
    )


def reversal_parameter_grid(
    candidate_config: FixedTimeReversalCandidateConfig,
    model_config: FixedTimeReversalModelConfig,
    core_config: CoreTrainingConfig,
) -> tuple[dict[str, Any], ...]:
    if candidate_config.parameter_grid == BASE_PARAMETER_GRID:
        parameters = tuple(
            histogram_parameters(candidate)
            for candidate in core_config.model.histogram_candidates
        )
    elif candidate_config.parameter_grid == TAIL_PARAMETER_GRID:
        parameters = tuple(
            _histogram_parameter_payload(candidate)
            for candidate in model_config.tail_histogram_parameters
        )
        if len(parameters) != 3:
            raise ValueError(
                "fixed-time reversal tail grid must contain exactly three combinations"
            )
    else:
        raise ValueError(
            "unsupported fixed-time reversal parameter grid: "
            f"{candidate_config.parameter_grid}"
        )
    if not parameters:
        raise ValueError("fixed-time reversal parameter grid cannot be empty")
    if len({_parameter_identity(item) for item in parameters}) != len(parameters):
        raise ValueError("fixed-time reversal parameter grid contains duplicates")
    return parameters


def eligible_complete_market_frame(
    frame: pl.DataFrame,
    *,
    decision_seconds: tuple[int, ...] = REVERSAL_ESTIMATOR_TRAINING_SECONDS,
    expected_source_markets: int | None = None,
    expected_source_estimator_rows: int | None = None,
    expected_eligible_markets: int | None = None,
    expected_eligible_estimator_rows: int | None = None,
    expected_exact_120_eligible_markets: int | None = None,
) -> pl.DataFrame:
    required = {
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "binance_sign_up",
        "btc_path_from_window_open_bps",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "fixed-time reversal source frame is missing columns: "
            + ", ".join(missing)
        )
    if decision_seconds != REVERSAL_ESTIMATOR_TRAINING_SECONDS:
        raise ValueError("fixed-time reversal estimator seconds changed")

    estimator = frame.filter(pl.col("seconds_elapsed").is_in(decision_seconds))
    if estimator.is_empty():
        raise ValueError("fixed-time reversal estimator frame cannot be empty")
    market_quality = estimator.group_by("market_id").agg(
        pl.len().alias("rows"),
        pl.col("seconds_elapsed").n_unique().alias("unique_seconds"),
        pl.col("window_start").n_unique().alias("unique_windows"),
        pl.col("label_up").n_unique().alias("unique_labels"),
    )
    incomplete = market_quality.filter(
        (pl.col("rows") != len(decision_seconds))
        | (pl.col("unique_seconds") != len(decision_seconds))
        | (pl.col("unique_windows") != 1)
        | (pl.col("unique_labels") != 1)
    )
    if incomplete.height:
        raise ValueError(
            "every fixed-time reversal market must contain each 120-140 row "
            "exactly once with one outcome label"
        )
    source_markets = market_quality.height
    if (
        expected_source_markets is not None
        and source_markets != expected_source_markets
    ):
        raise RuntimeError(
            "fixed-time reversal source market count changed: "
            f"expected {expected_source_markets}, observed {source_markets}"
        )
    if (
        expected_source_estimator_rows is not None
        and estimator.height != expected_source_estimator_rows
    ):
        raise RuntimeError(
            "fixed-time reversal source estimator row count changed: "
            f"expected {expected_source_estimator_rows}, observed {estimator.height}"
        )

    eligible = estimator.filter(
        pl.Series(
            "_path_directionally_eligible",
            path_is_directionally_eligible(estimator),
            dtype=pl.Boolean,
        )
    ).sort(["window_start", "market_id", "observed_at"])
    if eligible.is_empty():
        raise RuntimeError("fixed-time reversal cohort has no eligible estimator rows")
    eligible_markets = eligible["market_id"].n_unique()
    if (
        expected_eligible_markets is not None
        and eligible_markets != expected_eligible_markets
    ):
        raise RuntimeError(
            "fixed-time reversal eligible market count changed: "
            f"expected {expected_eligible_markets}, observed {eligible_markets}"
        )
    if (
        expected_eligible_estimator_rows is not None
        and eligible.height != expected_eligible_estimator_rows
    ):
        raise RuntimeError(
            "fixed-time reversal eligible estimator row count changed: "
            f"expected {expected_eligible_estimator_rows}, observed {eligible.height}"
        )
    exact_120 = eligible.filter(pl.col("seconds_elapsed") == REVERSAL_DECISION_SECOND)
    exact_120_markets = exact_120["market_id"].n_unique()
    if exact_120.height != exact_120_markets:
        raise RuntimeError(
            "fixed-time reversal exact-120 cohort must contain one row per market"
        )
    if (
        expected_exact_120_eligible_markets is not None
        and exact_120_markets != expected_exact_120_eligible_markets
    ):
        raise RuntimeError(
            "fixed-time reversal exact-120 eligible market count changed: "
            f"expected {expected_exact_120_eligible_markets}, "
            f"observed {exact_120_markets}"
        )
    return eligible


def training_target_frame(frame: pl.DataFrame, target_kind: str) -> pl.DataFrame:
    _validate_target_columns(frame)
    if target_kind == FIXED_TIME_TARGET:
        return frame
    if target_kind != PATH_PERSISTENCE_TARGET:
        raise ValueError(f"unsupported fixed-time reversal target kind: {target_kind}")
    return frame.with_columns(
        pl.Series("label_up", persistence_target_labels(frame), dtype=pl.Int8)
    )


def reversal_probability_semantics(
    frame: pl.DataFrame,
    target_probability: np.ndarray,
    target_kind: str,
) -> dict[str, np.ndarray]:
    _validate_target_columns(frame)
    probability = np.asarray(target_probability, dtype=np.float64)
    _validate_probabilities(probability, expected_rows=frame.height)
    if not path_is_directionally_eligible(frame).all():
        raise ValueError(
            "fixed-time reversal probability semantics require eligible BTC paths"
        )

    probability_up = target_probability_to_up(frame, probability, target_kind)
    sign_up = frame["binance_sign_up"].cast(pl.Int8).to_numpy() == 1
    if target_kind == PATH_PERSISTENCE_TARGET:
        probability_persistence = probability
    elif target_kind == FIXED_TIME_TARGET:
        probability_persistence = np.where(
            sign_up,
            probability_up,
            1.0 - probability_up,
        )
    else:
        raise ValueError(f"unsupported fixed-time reversal target kind: {target_kind}")
    probability_reversal = 1.0 - probability_persistence
    return {
        "p_target": probability,
        "p_persistence": probability_persistence,
        "p_reversal": probability_reversal,
        "probability_up": probability_up,
    }


def target_probability_up(
    frame: pl.DataFrame,
    target_probability: np.ndarray,
    target_kind: str,
) -> np.ndarray:
    return reversal_probability_semantics(
        frame,
        target_probability,
        target_kind,
    )["probability_up"]


def tune_and_fit_reversal_model(
    frame: pl.DataFrame,
    spec: CandidateSpec,
    parameter_grid: tuple[dict[str, Any], ...],
    core_config: CoreTrainingConfig,
    target_kind: str,
    decision_second: int,
    primary_coverage: float,
    diagnostic_coverage: float,
) -> tuple[FittedCoreModel, dict[str, Any]]:
    _validate_reversal_training_frame(frame, decision_second=decision_second)
    _validate_reversal_tuning_contract(
        target_kind=target_kind,
        decision_second=decision_second,
        primary_coverage=primary_coverage,
        diagnostic_coverage=diagnostic_coverage,
        parameter_grid=parameter_grid,
    )
    inner_train, inner_validation_context = chronological_inner_split(
        frame,
        validation_fraction=INNER_VALIDATION_FRACTION,
    )
    exact_validation = _exact_decision_rows(
        inner_validation_context,
        decision_second=decision_second,
        cohort_name="fixed-time reversal inner validation",
    )
    target_inner_train = training_target_frame(inner_train, target_kind)
    target_exact_validation = training_target_frame(exact_validation, target_kind)

    history: list[dict[str, Any]] = []
    best_parameters: dict[str, Any] | None = None
    best_rank: tuple[float, ...] | None = None
    for parameters in parameter_grid:
        started = time.perf_counter()
        model = fit_model(
            target_inner_train,
            spec,
            dict(parameters),
            core_config,
        )
        target_probability = np.asarray(
            model.raw_probability(exact_validation),
            dtype=np.float64,
        )
        semantics = reversal_probability_semantics(
            exact_validation,
            target_probability,
            target_kind,
        )
        scored = _scored_reversal_rows(exact_validation, semantics)
        primary = _reversal_operating_point(
            scored,
            target_coverage=primary_coverage,
        )
        diagnostic = _reversal_operating_point(
            scored,
            target_coverage=diagnostic_coverage,
        )
        weights = candidate_training_weights(exact_validation, spec)
        target_log_loss = float(
            log_loss(
                target_exact_validation["label_up"].to_numpy(),
                target_probability,
                sample_weight=weights,
                labels=[0, 1],
            )
        )
        outcome_log_loss = float(
            log_loss(
                exact_validation["label_up"].to_numpy(),
                semantics["probability_up"],
                sample_weight=weights,
                labels=[0, 1],
            )
        )
        converged = estimator_converged(model.estimator)
        record: dict[str, Any] = {
            "hyperparameters": dict(parameters),
            "converged": converged,
            "fit_and_score_seconds": time.perf_counter() - started,
            "inner_fit": _cohort_evidence(inner_train),
            "exact_120_inner_validation": _cohort_evidence(exact_validation),
            "target_kind": target_kind,
            "estimator_positive_class": target_kind,
            "probability_semantics": _probability_semantics_evidence(target_kind),
            "exact_120_target_log_loss": target_log_loss,
            "exact_120_outcome_log_loss": outcome_log_loss,
            "primary": primary,
            "diagnostic": diagnostic,
        }
        rank = _reversal_tuning_rank(record)
        record["selection_rank"] = list(rank)
        history.append(record)
        if converged and (best_rank is None or rank > best_rank):
            best_parameters = dict(parameters)
            best_rank = rank

    if best_parameters is None or best_rank is None:
        raise RuntimeError(f"no {spec.name} reversal hyperparameter candidate converged")

    final_model = fit_model(
        training_target_frame(frame, target_kind),
        spec,
        best_parameters,
        core_config,
    )
    return final_model, {
        "selection_objective": [
            "maximize primary minimum UP/DOWN recall",
            "maximize primary outcome accuracy",
            "maximize diagnostic outcome accuracy",
            "maximize primary balanced accuracy",
            "maximize primary override precision",
            "minimize primary false-UP exposure",
            "minimize exact-120 outcome log loss",
        ],
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "estimator_training_seconds": list(REVERSAL_ESTIMATOR_TRAINING_SECONDS),
        "hyperparameter_scoring_seconds": [decision_second],
        "target_kind": target_kind,
        "estimator_positive_class": target_kind,
        "probability_semantics": _probability_semantics_evidence(target_kind),
        "primary_target_coverage": primary_coverage,
        "diagnostic_target_coverage": diagnostic_coverage,
        "selected_hyperparameters": best_parameters,
        "selected_rank": list(best_rank),
        "candidates": history,
        "final_fit": _cohort_evidence(frame),
        "optimizer_converged": estimator_converged(final_model.estimator),
    }


def _histogram_parameter_payload(candidate: Any) -> dict[str, Any]:
    payload = asdict(candidate)
    return {
        "learning_rate": float(payload["learning_rate"]),
        "max_iter": int(payload["max_iter"]),
        "max_leaf_nodes": int(payload["max_leaf_nodes"]),
        "min_samples_leaf": int(payload["min_samples_leaf"]),
        "l2_regularization": float(payload["l2_regularization"]),
    }


def _parameter_identity(parameters: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(parameters.items()))


def _validate_reversal_training_frame(
    frame: pl.DataFrame,
    *,
    decision_second: int,
) -> None:
    required = {
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "binance_sign_up",
        "btc_path_from_window_open_bps",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "fixed-time reversal estimator frame is missing columns: "
            + ", ".join(missing)
        )
    if decision_second != REVERSAL_DECISION_SECOND:
        raise ValueError("fixed-time reversal hyperparameter scoring must remain exact 120")
    observed_seconds = set(frame["seconds_elapsed"].unique().to_list())
    expected_seconds = set(REVERSAL_ESTIMATOR_TRAINING_SECONDS)
    if (
        REVERSAL_DECISION_SECOND not in observed_seconds
        or not observed_seconds.issubset(expected_seconds)
    ):
        raise ValueError(
            "fixed-time reversal estimator requires eligible 120-140 rows"
        )
    market_quality = frame.group_by("market_id").agg(
        pl.len().alias("rows"),
        pl.col("seconds_elapsed").n_unique().alias("unique_seconds"),
        pl.col("window_start").n_unique().alias("unique_windows"),
        pl.col("label_up").n_unique().alias("unique_labels"),
    )
    invalid_markets = market_quality.filter(
        (pl.col("rows") != pl.col("unique_seconds"))
        | (pl.col("rows") < 1)
        | (pl.col("rows") > len(REVERSAL_ESTIMATOR_TRAINING_SECONDS))
        | (pl.col("unique_windows") != 1)
        | (pl.col("unique_labels") != 1)
    )
    if invalid_markets.height:
        raise ValueError(
            "fixed-time reversal estimator rows must be unique per market and second"
        )
    if not path_is_directionally_eligible(frame).all():
        raise ValueError(
            "fixed-time reversal estimator frame contains a path-ineligible row"
        )
    _exact_decision_rows(
        frame,
        decision_second=decision_second,
        cohort_name="fixed-time reversal estimator",
    )


def _validate_reversal_tuning_contract(
    *,
    target_kind: str,
    decision_second: int,
    primary_coverage: float,
    diagnostic_coverage: float,
    parameter_grid: tuple[dict[str, Any], ...],
) -> None:
    if target_kind not in {FIXED_TIME_TARGET, PATH_PERSISTENCE_TARGET}:
        raise ValueError(f"unsupported fixed-time reversal target kind: {target_kind}")
    if decision_second != REVERSAL_DECISION_SECOND:
        raise ValueError("fixed-time reversal decision second must remain 120")
    if not math.isclose(primary_coverage, PRIMARY_REVERSAL_COVERAGE):
        raise ValueError("fixed-time reversal primary coverage must remain 10%")
    if not math.isclose(diagnostic_coverage, DIAGNOSTIC_REVERSAL_COVERAGE):
        raise ValueError("fixed-time reversal diagnostic coverage must remain 8%")
    if not parameter_grid:
        raise ValueError("fixed-time reversal parameter grid cannot be empty")


def _validate_target_columns(frame: pl.DataFrame) -> None:
    required = {"label_up", "binance_sign_up", "btc_path_from_window_open_bps"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "fixed-time reversal target frame is missing columns: "
            + ", ".join(missing)
        )
    for column in ("label_up", "binance_sign_up"):
        invalid = frame.filter(
            pl.col(column).is_null() | ~pl.col(column).is_in([0, 1])
        )
        if invalid.height:
            raise ValueError(f"{column} must contain only binary directions")


def _exact_decision_rows(
    frame: pl.DataFrame,
    *,
    decision_second: int,
    cohort_name: str,
) -> pl.DataFrame:
    selected = frame.filter(pl.col("seconds_elapsed") == decision_second).sort(
        ["window_start", "market_id", "observed_at"]
    )
    if selected.is_empty():
        raise RuntimeError(f"{cohort_name} has no exact decision rows")
    if selected.height != selected["market_id"].n_unique():
        raise RuntimeError(f"{cohort_name} must contain one row per market")
    return selected


def _validate_probabilities(
    probability: np.ndarray,
    *,
    expected_rows: int,
) -> None:
    if probability.ndim != 1 or len(probability) != expected_rows:
        raise ValueError("reversal probability count does not match exact-120 rows")
    if not np.isfinite(probability).all():
        raise ValueError("reversal probabilities must be finite")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("reversal probabilities must remain between zero and one")


def _scored_reversal_rows(
    frame: pl.DataFrame,
    semantics: dict[str, np.ndarray],
) -> pl.DataFrame:
    scored = scored_prediction_rows(frame, semantics["probability_up"])
    scored = scored.with_columns(
        pl.Series("p_target", semantics["p_target"]),
        pl.Series("p_persistence", semantics["p_persistence"]),
        pl.Series("p_reversal", semantics["p_reversal"]),
    ).with_columns(
        (pl.col("predicted_up") != pl.col("binance_sign_up")).alias(
            "path_overridden"
        ),
    ).with_columns(
        pl.col("path_overridden").alias("predicted_reversal"),
    )
    return scored


def _reversal_operating_point(
    scored: pl.DataFrame,
    *,
    target_coverage: float,
) -> dict[str, Any]:
    threshold, selected, selection = _empirical_confidence_quantile(
        scored,
        target_coverage=target_coverage,
    )
    return {
        "threshold_selection": selection,
        "metrics": classification_metrics(
            selected,
            eligible_markets=scored.height,
        ),
        "override_metrics": _override_metrics(
            selected,
            eligible_markets=scored.height,
        ),
        "hard_confident_errors": hard_confident_error_metrics(
            selected,
            eligible_markets=scored.height,
            confidence_floor=HARD_CONFIDENCE_FLOOR,
        ),
        "confidence_threshold": threshold,
        "selected_market_ids": selected["market_id"].to_list(),
    }


def _override_metrics(
    selected: pl.DataFrame,
    *,
    eligible_markets: int,
) -> dict[str, Any]:
    overrides = selected.filter(pl.col("path_overridden"))
    false_up = selected.filter(
        (pl.col("predicted_up") == 1) & (pl.col("label_up") == 0)
    )
    false_down = selected.filter(
        (pl.col("predicted_up") == 0) & (pl.col("label_up") == 1)
    )
    actual_reversals = selected.filter(
        pl.col("label_up") != pl.col("binance_sign_up")
    )
    missed_reversals = actual_reversals.filter(~pl.col("path_overridden"))
    missed_reversal_false_up = missed_reversals.filter(
        (pl.col("predicted_up") == 1) & (pl.col("label_up") == 0)
    )
    missed_reversal_false_down = missed_reversals.filter(
        (pl.col("predicted_up") == 0) & (pl.col("label_up") == 1)
    )
    false_overrides = overrides.filter(
        pl.col("label_up") == pl.col("binance_sign_up")
    )
    corrected = overrides.filter(pl.col("correct") & ~pl.col("baseline_correct"))
    damaged = overrides.filter(~pl.col("correct") & pl.col("baseline_correct"))
    return {
        "selected_markets": selected.height,
        "eligible_markets": eligible_markets,
        "override_markets": overrides.height,
        "override_exposure_rate": (
            overrides.height / eligible_markets if eligible_markets else 0.0
        ),
        "override_rate_selected": (
            overrides.height / selected.height if selected.height else 0.0
        ),
        "override_correct": int(overrides["correct"].sum() or 0),
        "override_precision": (
            float(overrides["correct"].mean()) if overrides.height else None
        ),
        "baseline_errors_corrected": corrected.height,
        "baseline_correct_decisions_damaged": damaged.height,
        "net_corrected_decisions": corrected.height - damaged.height,
        "actual_reversal_markets": actual_reversals.height,
        "missed_reversal_markets": missed_reversals.height,
        "missed_reversal_false_up_markets": missed_reversal_false_up.height,
        "missed_reversal_false_down_markets": missed_reversal_false_down.height,
        "false_override_markets": false_overrides.height,
        "false_up_markets": false_up.height,
        "false_up_exposure_rate": (
            false_up.height / eligible_markets if eligible_markets else 0.0
        ),
        "false_down_markets": false_down.height,
        "false_down_exposure_rate": (
            false_down.height / eligible_markets if eligible_markets else 0.0
        ),
    }


def _empirical_confidence_quantile(
    scored: pl.DataFrame,
    *,
    target_coverage: float,
) -> tuple[float, pl.DataFrame, dict[str, Any]]:
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target coverage must be between zero and one")
    if scored.is_empty():
        raise ValueError("cannot select a confidence quantile from empty rows")
    if scored.height != scored["market_id"].n_unique():
        raise ValueError("confidence quantile requires one row per market")
    if scored["confidence"].is_null().any() or not scored["confidence"].is_finite().all():
        raise ValueError("confidence quantile contains missing or non-finite values")
    target_markets = max(1, math.ceil(scored.height * target_coverage))
    ordered = scored.sort(
        ["confidence", "observed_at", "market_id"],
        descending=[True, False, False],
    )
    threshold = float(ordered[target_markets - 1, "confidence"])
    selected = scored.filter(pl.col("confidence") >= threshold).sort(
        ["observed_at", "market_id"]
    )
    return threshold, selected, {
        "method": "empirical_policy_confidence_quantile",
        "labels_used_for_threshold_selection": False,
        "eligible_markets": scored.height,
        "target_coverage": target_coverage,
        "target_markets": target_markets,
        "confidence_threshold": threshold,
        "selected_markets": selected.height,
        "realized_coverage": selected.height / scored.height,
        "tie_expansion_markets": selected.height - target_markets,
    }


def _reversal_tuning_rank(record: dict[str, Any]) -> tuple[float, ...]:
    primary = record["primary"]["metrics"]
    diagnostic = record["diagnostic"]["metrics"]
    override = record["primary"]["override_metrics"]
    override_precision = override["override_precision"]
    return (
        min(primary["up_recall"], primary["down_recall"]),
        primary["accuracy"],
        diagnostic["accuracy"],
        primary["balanced_accuracy"],
        float(override_precision) if override_precision is not None else 0.0,
        -override["false_up_exposure_rate"],
        -record["exact_120_outcome_log_loss"],
    )


def _probability_semantics_evidence(target_kind: str) -> dict[str, str]:
    return {
        "estimator_target": target_kind,
        "p_target": f"P({target_kind})",
        "p_persistence": "P(outcome direction equals Binance path direction)",
        "p_reversal": "1 - p_persistence",
        "predicted_reversal": (
            "executable path_overridden: predicted_up != binance_sign_up; "
            "probability_up >= 0.5 resolves direction ties to UP"
        ),
        "probability_up": (
            "p_target for outcome_up; otherwise p_persistence when Binance path is UP "
            "and 1 - p_persistence when Binance path is DOWN"
        ),
    }


def _cohort_evidence(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "decision_seconds": sorted(frame["seconds_elapsed"].unique().to_list()),
        "minimum_window_start": frame["window_start"].min().isoformat(),
        "maximum_window_start": frame["window_start"].max().isoformat(),
    }
