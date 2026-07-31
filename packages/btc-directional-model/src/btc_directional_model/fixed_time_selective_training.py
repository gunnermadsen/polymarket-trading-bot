from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import log_loss

from .core_config import CoreTrainingConfig, RowWeightScheduleConfig
from .core_evaluation import classification_metrics, scored_prediction_rows
from .core_training import (
    EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY,
    MARKET_EQUAL_ROW_WEIGHT_POLICY,
    CandidateSpec,
    FittedCoreModel,
    candidate_training_weights,
    chronological_inner_split,
    estimator_converged,
    fit_model,
    histogram_parameters,
)
from .fixed_time_selective_config import (
    BASE_PARAMETER_GRID,
    TAIL_PARAMETER_GRID,
    FixedTimeSelectiveCandidateConfig,
    FixedTimeSelectiveModelConfig,
)
from .persistence_benchmark import hard_confident_error_metrics

SELECTIVE_ESTIMATOR_TRAINING_SECONDS = (120, 125, 130, 135, 140)
SELECTIVE_DECISION_SECOND = 120
PRIMARY_SELECTIVE_COVERAGE = 0.15
SECONDARY_SELECTIVE_COVERAGE = 0.10
INNER_VALIDATION_FRACTION = 0.20


def selective_candidate_spec(
    candidate_config: FixedTimeSelectiveCandidateConfig,
    model_config: FixedTimeSelectiveModelConfig,
) -> CandidateSpec:
    multiplier = candidate_config.exact_120_weight_multiplier
    if math.isclose(multiplier, 1.0):
        row_weight_policy = MARKET_EQUAL_ROW_WEIGHT_POLICY
        row_weight_schedule = None
    else:
        row_weight_policy = EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY
        row_weight_schedule = _exact_decision_weight_schedule(
            model_config.decision_second,
            multiplier,
        )
    return CandidateSpec(
        name=candidate_config.name,
        family="histogram",
        feature_names=tuple(candidate_config.feature_names),
        row_weight_policy=row_weight_policy,
        row_weight_schedule=row_weight_schedule,
        recency_half_life_days=model_config.recency_half_life_days,
    )


def selective_parameter_grid(
    candidate_config: FixedTimeSelectiveCandidateConfig,
    model_config: FixedTimeSelectiveModelConfig,
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
                "fixed-time selective tail grid must contain exactly three combinations"
            )
    else:
        raise ValueError(
            f"unsupported fixed-time selective parameter grid: "
            f"{candidate_config.parameter_grid}"
        )
    if not parameters:
        raise ValueError("fixed-time selective parameter grid cannot be empty")
    if len({_parameter_identity(item) for item in parameters}) != len(parameters):
        raise ValueError("fixed-time selective parameter grid contains duplicates")
    return parameters


def tune_and_fit_selective_model(
    frame: pl.DataFrame,
    spec: CandidateSpec,
    parameter_grid: tuple[dict[str, Any], ...],
    core_config: CoreTrainingConfig,
    decision_second: int,
    primary_coverage: float,
    secondary_coverage: float,
    hard_confidence_floor: float,
) -> tuple[FittedCoreModel, dict[str, Any]]:
    _validate_selective_training_frame(frame, decision_second=decision_second)
    _validate_selective_tuning_contract(
        decision_second=decision_second,
        primary_coverage=primary_coverage,
        secondary_coverage=secondary_coverage,
        hard_confidence_floor=hard_confidence_floor,
        parameter_grid=parameter_grid,
    )
    inner_train, inner_validation_context = chronological_inner_split(
        frame,
        validation_fraction=INNER_VALIDATION_FRACTION,
    )
    exact_validation = _exact_decision_rows(
        inner_validation_context,
        decision_second=decision_second,
        cohort_name="fixed-time selective inner validation",
    )

    history: list[dict[str, Any]] = []
    best_parameters: dict[str, Any] | None = None
    best_rank: tuple[float, ...] | None = None
    for parameters in parameter_grid:
        started = time.perf_counter()
        model = fit_model(inner_train, spec, dict(parameters), core_config)
        probability = np.asarray(
            model.raw_probability(exact_validation),
            dtype=np.float64,
        )
        _validate_probabilities(probability, expected_rows=exact_validation.height)
        scored = scored_prediction_rows(exact_validation, probability)
        primary = _selective_operating_point(
            scored,
            target_coverage=primary_coverage,
            hard_confidence_floor=hard_confidence_floor,
        )
        secondary = _selective_operating_point(
            scored,
            target_coverage=secondary_coverage,
            hard_confidence_floor=hard_confidence_floor,
        )
        exact_log_loss = float(
            log_loss(
                exact_validation["label_up"].to_numpy(),
                probability,
                sample_weight=candidate_training_weights(exact_validation, spec),
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
            "exact_120_log_loss": exact_log_loss,
            "primary": primary,
            "secondary": secondary,
        }
        rank = _selective_tuning_rank(record)
        record["selection_rank"] = list(rank)
        history.append(record)
        if converged and (best_rank is None or rank > best_rank):
            best_parameters = dict(parameters)
            best_rank = rank

    if best_parameters is None or best_rank is None:
        raise RuntimeError(f"no {spec.name} selective hyperparameter candidate converged")

    final_model = fit_model(frame, spec, best_parameters, core_config)
    return final_model, {
        "selection_objective": [
            "maximize primary minimum UP/DOWN recall",
            "maximize primary accuracy",
            "maximize secondary accuracy",
            "maximize primary balanced accuracy",
            "minimize primary hard-confident selected-error rate",
            "minimize exact-120 log loss",
        ],
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "estimator_training_seconds": list(SELECTIVE_ESTIMATOR_TRAINING_SECONDS),
        "hyperparameter_scoring_seconds": [decision_second],
        "primary_target_coverage": primary_coverage,
        "secondary_target_coverage": secondary_coverage,
        "hard_confidence_floor": hard_confidence_floor,
        "selected_hyperparameters": best_parameters,
        "selected_rank": list(best_rank),
        "candidates": history,
        "final_fit": _cohort_evidence(frame),
        "optimizer_converged": estimator_converged(final_model.estimator),
    }


def predicted_side_hard_error_metrics(
    selected: pl.DataFrame,
    eligible_markets: int,
    confidence_floor: float,
) -> dict[str, Any]:
    if "predicted_up" not in selected.columns:
        raise ValueError("predicted-side hard-error evidence requires predicted_up")
    invalid_direction = selected.filter(
        pl.col("predicted_up").is_null()
        | ~pl.col("predicted_up").is_in([0, 1])
    )
    if invalid_direction.height:
        raise ValueError("predicted_up must contain only binary directions")

    all_metrics = hard_confident_error_metrics(
        selected,
        eligible_markets=eligible_markets,
        confidence_floor=confidence_floor,
    )
    up_metrics = hard_confident_error_metrics(
        selected.filter(pl.col("predicted_up") == 1),
        eligible_markets=eligible_markets,
        confidence_floor=confidence_floor,
    )
    down_metrics = hard_confident_error_metrics(
        selected.filter(pl.col("predicted_up") == 0),
        eligible_markets=eligible_markets,
        confidence_floor=confidence_floor,
    )
    all_errors = all_metrics["hard_confident_error_markets"]
    side_errors = (
        up_metrics["hard_confident_error_markets"]
        + down_metrics["hard_confident_error_markets"]
    )
    if all_errors != side_errors:
        raise RuntimeError("predicted-side hard-confident errors do not sum to all errors")
    if any(
        payload["eligible_markets"] != eligible_markets
        for payload in (all_metrics, up_metrics, down_metrics)
    ):
        raise RuntimeError("predicted-side hard-error exposure denominators changed")
    return {
        "all": all_metrics,
        "up": up_metrics,
        "down": down_metrics,
        "hard_confident_error_sum_invariant": True,
        "exposure_denominator": "global_eligible_markets",
    }


def _exact_decision_weight_schedule(
    decision_second: int,
    multiplier: float,
) -> RowWeightScheduleConfig:
    return RowWeightScheduleConfig(
        start_second=decision_second,
        end_second_inclusive=decision_second,
        multiplier=multiplier,
    )


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


def _validate_selective_training_frame(
    frame: pl.DataFrame,
    *,
    decision_second: int,
) -> None:
    if frame.is_empty():
        raise ValueError("fixed-time selective estimator frame cannot be empty")
    required = {
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "binance_sign_up",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "fixed-time selective estimator frame is missing columns: "
            + ", ".join(missing)
        )
    if decision_second != SELECTIVE_DECISION_SECOND:
        raise ValueError("fixed-time selective hyperparameter scoring must remain exact 120")
    observed_seconds = set(frame["seconds_elapsed"].unique().to_list())
    if observed_seconds != set(SELECTIVE_ESTIMATOR_TRAINING_SECONDS):
        raise ValueError("fixed-time selective estimator requires exact 120-140 rows")
    incomplete = (
        frame.group_by("market_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("seconds_elapsed").n_unique().alias("unique_seconds"),
        )
        .filter(
            (pl.col("rows") != len(SELECTIVE_ESTIMATOR_TRAINING_SECONDS))
            | (
                pl.col("unique_seconds")
                != len(SELECTIVE_ESTIMATOR_TRAINING_SECONDS)
            )
        )
    )
    if incomplete.height:
        raise ValueError("every market must contain each fixed-time estimator row exactly once")


def _validate_selective_tuning_contract(
    *,
    decision_second: int,
    primary_coverage: float,
    secondary_coverage: float,
    hard_confidence_floor: float,
    parameter_grid: tuple[dict[str, Any], ...],
) -> None:
    if decision_second != SELECTIVE_DECISION_SECOND:
        raise ValueError("fixed-time selective decision second must remain 120")
    if not math.isclose(primary_coverage, PRIMARY_SELECTIVE_COVERAGE):
        raise ValueError("fixed-time selective primary coverage must remain 15%")
    if not math.isclose(secondary_coverage, SECONDARY_SELECTIVE_COVERAGE):
        raise ValueError("fixed-time selective secondary coverage must remain 10%")
    if not 0.5 <= hard_confidence_floor <= 1.0:
        raise ValueError("hard-confidence floor must be between 0.5 and 1.0")
    if not parameter_grid:
        raise ValueError("fixed-time selective parameter grid cannot be empty")


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
        raise ValueError("selective probability count does not match exact-120 rows")
    if not np.isfinite(probability).all():
        raise ValueError("selective probabilities must be finite")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("selective probabilities must remain between zero and one")


def _selective_operating_point(
    scored: pl.DataFrame,
    *,
    target_coverage: float,
    hard_confidence_floor: float,
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
        "hard_confident_errors": predicted_side_hard_error_metrics(
            selected,
            eligible_markets=scored.height,
            confidence_floor=hard_confidence_floor,
        ),
        "confidence_threshold": threshold,
        "selected_market_ids": selected["market_id"].to_list(),
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


def _selective_tuning_rank(record: dict[str, Any]) -> tuple[float, ...]:
    primary = record["primary"]["metrics"]
    secondary = record["secondary"]["metrics"]
    hard_errors = record["primary"]["hard_confident_errors"]["all"]
    return (
        min(primary["up_recall"], primary["down_recall"]),
        primary["accuracy"],
        secondary["accuracy"],
        primary["balanced_accuracy"],
        -hard_errors["hard_confident_error_rate_selected"],
        -record["exact_120_log_loss"],
    )


def _cohort_evidence(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "decision_seconds": sorted(frame["seconds_elapsed"].unique().to_list()),
        "minimum_window_start": frame["window_start"].min().isoformat(),
        "maximum_window_start": frame["window_start"].max().isoformat(),
    }
