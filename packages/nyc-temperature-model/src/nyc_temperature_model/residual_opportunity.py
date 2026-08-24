from __future__ import annotations

import copy
import math
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .asymmetric_benchmark import _circular_block_indices

FEATURE_SCHEMA_VERSION = "nyc-temperature-market-offset-v1"
PROBABILITY_CLIP = 0.005
RIDGE_PENALTY = 1.0
WARMUP_EVENT_DAYS = 30
REFIT_INTERVAL_EVENT_DAYS = 12
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_BLOCK_DAYS = 7
BOOTSTRAP_SEED = 7_351_985
LOWER_QUANTILE = 0.10
MINIMUM_ALL_IN_COST = 0.04
MAXIMUM_ALL_IN_COST = 0.25


@dataclass(frozen=True)
class ResidualTrainingRow:
    event_date: date
    decision_time: datetime
    source_timestamp: datetime
    decision_hour_local: int
    market_id: str
    weather_probability_yes: float
    market_probability_yes: float
    resolved_yes: bool
    label_available_at: datetime


@dataclass(frozen=True)
class MarketOffsetFit:
    opportunity_fit_id: str
    origin_time: datetime
    latest_label_available_at: datetime
    training_start: date
    training_end: date
    training_event_days: int
    training_rows: int
    coefficients: tuple[float, float]
    bootstrap_coefficients: np.ndarray
    converged: bool
    fit_metrics: dict[str, Any]


def _clip_probability(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("residual probability inputs must be finite")
    return min(1.0 - PROBABILITY_CLIP, max(PROBABILITY_CLIP, value))


def _logit(value: float | np.ndarray) -> float | np.ndarray:
    clipped = np.clip(value, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    nonnegative = value >= 0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exponent = np.exp(value[~nonnegative])
    output[~nonnegative] = exponent / (1.0 + exponent)
    return float(output) if output.ndim == 0 else output


def market_offset_probability(
    weather_probability: float,
    market_probability: float,
    weather_residual_weight: float,
) -> float:
    if not 0.0 <= weather_residual_weight <= 1.0:
        raise ValueError("weather residual weight must be between zero and one")
    market_logit = float(_logit(_clip_probability(market_probability)))
    weather_logit = float(_logit(_clip_probability(weather_probability)))
    return float(
        _sigmoid(
            market_logit
            + weather_residual_weight * (weather_logit - market_logit)
        )
    )


def _event_date_weights(rows: list[ResidualTrainingRow]) -> np.ndarray:
    counts = Counter(row.event_date for row in rows)
    if not counts:
        return np.asarray([], dtype=np.float64)
    weights = np.asarray([1.0 / counts[row.event_date] for row in rows], dtype=np.float64)
    totals: dict[date, float] = defaultdict(float)
    for row, weight in zip(rows, weights, strict=True):
        totals[row.event_date] += float(weight)
    if any(not math.isclose(total, 1.0, abs_tol=1e-12) for total in totals.values()):
        raise ValueError("residual training weights must total one per event date")
    return weights


def _decision_hour_index(decision_hour_local: int) -> int:
    if decision_hour_local == 0:
        return 0
    if decision_hour_local == 12:
        return 1
    raise ValueError(
        "market-offset model supports only midnight and noon decision hours"
    )


def _fit_coefficients(
    rows: list[ResidualTrainingRow],
    weights: np.ndarray,
    *,
    ridge_penalty: float = RIDGE_PENALTY,
) -> tuple[np.ndarray, bool, float]:
    if not rows or weights.shape != (len(rows),):
        raise ValueError("market-offset fitting requires aligned training rows and weights")
    if ridge_penalty <= 0 or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("market-offset fitting requires positive finite weights and penalty")
    weather = np.asarray([row.weather_probability_yes for row in rows], dtype=np.float64)
    market = np.asarray([row.market_probability_yes for row in rows], dtype=np.float64)
    outcomes = np.asarray([float(row.resolved_yes) for row in rows], dtype=np.float64)
    hours = np.asarray(
        [_decision_hour_index(row.decision_hour_local) for row in rows],
        dtype=np.int64,
    )
    market_logit = np.asarray(_logit(market), dtype=np.float64)
    residual = np.asarray(_logit(weather), dtype=np.float64) - market_logit
    weight_total = float(weights.sum())

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        selected = coefficients[hours]
        linear = market_logit + selected * residual
        probabilities = np.asarray(_sigmoid(linear), dtype=np.float64)
        probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
        loss = -float(
            np.sum(
                weights
                * (
                    outcomes * np.log(probabilities)
                    + (1.0 - outcomes) * np.log(1.0 - probabilities)
                )
            )
            / weight_total
        )
        loss += 0.5 * ridge_penalty * float(np.dot(coefficients, coefficients))
        gradient = np.asarray(
            [
                np.sum(
                    weights[hours == index]
                    * (probabilities[hours == index] - outcomes[hours == index])
                    * residual[hours == index]
                )
                / weight_total
                + ridge_penalty * coefficients[index]
                for index in (0, 1)
            ],
            dtype=np.float64,
        )
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(2, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 200},
    )
    coefficients = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    return coefficients, bool(result.success), float(result.fun)


def _weighted_binary_scores(
    rows: list[ResidualTrainingRow],
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    outcomes = np.asarray([float(row.resolved_yes) for row in rows], dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    weight_total = float(weights.sum())
    log_loss = -float(
        np.sum(
            weights
            * (
                outcomes * np.log(probabilities)
                + (1.0 - outcomes) * np.log(1.0 - probabilities)
            )
        )
        / weight_total
    )
    brier = float(np.sum(weights * (probabilities - outcomes) ** 2) / weight_total)
    return log_loss, brier


def fit_market_offset(
    rows: list[ResidualTrainingRow],
    *,
    origin_time: datetime,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> MarketOffsetFit:
    if any(
        row.source_timestamp is None
        or row.source_timestamp > row.decision_time
        for row in rows
    ):
        raise ValueError(
            "market-offset fit requires every source time at or before its decision time"
        )
    if any(row.decision_time >= origin_time for row in rows):
        raise ValueError(
            "market-offset fit requires every decision time to precede its origin"
        )
    if any(row.label_available_at >= origin_time for row in rows):
        raise ValueError(
            "market-offset fit requires every label availability time to precede its origin"
        )
    for row in rows:
        _decision_hour_index(row.decision_hour_local)
    event_dates = sorted({row.event_date for row in rows})
    if len(event_dates) < WARMUP_EVENT_DAYS:
        raise ValueError(
            f"market-offset fit requires {WARMUP_EVENT_DAYS} event dates, found {len(event_dates)}"
        )
    if bootstrap_iterations < 100:
        raise ValueError("market-offset probability bounds require at least 100 bootstraps")
    weights = _event_date_weights(rows)
    coefficients, converged, penalized_loss = _fit_coefficients(rows, weights)
    weather = np.asarray([row.weather_probability_yes for row in rows], dtype=np.float64)
    market = np.asarray([row.market_probability_yes for row in rows], dtype=np.float64)
    hours = np.asarray(
        [_decision_hour_index(row.decision_hour_local) for row in rows],
        dtype=np.int64,
    )
    market_logits = np.asarray(_logit(market), dtype=np.float64)
    weather_residual = np.asarray(_logit(weather), dtype=np.float64) - market_logits
    fitted = np.asarray(
        _sigmoid(market_logits + coefficients[hours] * weather_residual), dtype=np.float64
    )
    fitted_log_loss, fitted_brier = _weighted_binary_scores(rows, fitted, weights)
    market_log_loss, market_brier = _weighted_binary_scores(rows, market, weights)
    weather_log_loss, weather_brier = _weighted_binary_scores(rows, weather, weights)

    block_indices = _circular_block_indices(
        len(event_dates),
        iterations=bootstrap_iterations,
        block_size=min(BOOTSTRAP_BLOCK_DAYS, len(event_dates)),
        seed=bootstrap_seed ^ int(origin_time.timestamp()),
    )
    bootstrap_coefficients = np.empty((bootstrap_iterations, 2), dtype=np.float64)
    date_position = {event_date: index for index, event_date in enumerate(event_dates)}
    base_weights = _event_date_weights(rows)
    convergence_count = 0
    for iteration, sampled_positions in enumerate(block_indices):
        multiplicity = np.bincount(sampled_positions, minlength=len(event_dates))
        sampled_weights = np.asarray(
            [base_weights[index] * multiplicity[date_position[row.event_date]] for index, row in enumerate(rows)],
            dtype=np.float64,
        )
        sampled_coefficients, sampled_converged, _ = _fit_coefficients(
            rows, sampled_weights
        )
        bootstrap_coefficients[iteration] = sampled_coefficients
        convergence_count += int(sampled_converged)
    convergence_rate = convergence_count / bootstrap_iterations
    if convergence_rate < 0.95:
        raise ValueError(
            f"market-offset bootstrap convergence rate is too low: {convergence_rate:.3f}"
        )

    return MarketOffsetFit(
        opportunity_fit_id=str(uuid.uuid4()),
        origin_time=origin_time,
        latest_label_available_at=max(row.label_available_at for row in rows),
        training_start=min(event_dates),
        training_end=max(event_dates),
        training_event_days=len(event_dates),
        training_rows=len(rows),
        coefficients=(float(coefficients[0]), float(coefficients[1])),
        bootstrap_coefficients=bootstrap_coefficients,
        converged=converged,
        fit_metrics={
            "penalized_objective": penalized_loss,
            "weighted_binary_log_loss": fitted_log_loss,
            "weighted_binary_brier_score": fitted_brier,
            "market_weighted_binary_log_loss": market_log_loss,
            "market_weighted_binary_brier_score": market_brier,
            "weather_weighted_binary_log_loss": weather_log_loss,
            "weather_weighted_binary_brier_score": weather_brier,
            "bootstrap_convergence_rate": convergence_rate,
        },
    )


def _candidate_contract_key(candidate: dict[str, Any]) -> tuple[str, str, datetime]:
    return (
        candidate["model_run_id"],
        candidate["market_id"],
        candidate["decision_time"],
    )


def _side_is_in_training_universe(candidate: dict[str, Any]) -> bool:
    cost = candidate.get("all_in_cost_per_share")
    return bool(
        candidate.get("executable")
        and cost is not None
        and MINIMUM_ALL_IN_COST <= float(cost) <= MAXIMUM_ALL_IN_COST
    )


def _has_causal_market_quality(candidate: dict[str, Any]) -> bool:
    flags = set(candidate.get("quality_flags") or [])
    if any(
        str(flag).startswith(("pmxt_archive_", "crossed_yes_", "crossed_no_"))
        for flag in flags
    ):
        return False
    source_timestamp = candidate.get("source_timestamp")
    return bool(
        source_timestamp is not None
        and source_timestamp <= candidate["decision_time"]
    )


def contract_training_rows(candidates: list[dict[str, Any]]) -> list[ResidualTrainingRow]:
    contracts: dict[tuple[str, str, datetime], dict[str, dict[str, Any]]] = defaultdict(dict)
    for candidate in candidates:
        _decision_hour_index(int(candidate["decision_hour_local"]))
        contracts[_candidate_contract_key(candidate)][candidate["side"]] = candidate
    rows = []
    for sides in contracts.values():
        yes = sides.get("YES")
        if yes is None or not any(_side_is_in_training_universe(row) for row in sides.values()):
            continue
        market_probability = yes.get("market_probability_proxy")
        label_available_at = yes.get("label_available_at")
        if market_probability is None or label_available_at is None:
            continue
        if not _has_causal_market_quality(yes):
            continue
        rows.append(
            ResidualTrainingRow(
                event_date=yes["event_date"],
                decision_time=yes["decision_time"],
                source_timestamp=yes["source_timestamp"],
                decision_hour_local=int(yes["decision_hour_local"]),
                market_id=yes["market_id"],
                weather_probability_yes=float(yes["weather_probability"]),
                market_probability_yes=float(market_probability),
                resolved_yes=bool(yes["resolved_side"]),
                label_available_at=label_available_at,
            )
        )
    return sorted(rows, key=lambda row: (row.event_date, row.decision_time, row.market_id))


def _fit_record(fit: MarketOffsetFit) -> dict[str, Any]:
    bootstrap = fit.bootstrap_coefficients
    return {
        "opportunity_fit_id": fit.opportunity_fit_id,
        "origin_time": fit.origin_time,
        "latest_label_available_at": fit.latest_label_available_at,
        "training_start": fit.training_start,
        "training_end": fit.training_end,
        "training_event_days": fit.training_event_days,
        "training_rows": fit.training_rows,
        "ridge_penalty": RIDGE_PENALTY,
        "probability_clip": PROBABILITY_CLIP,
        "coefficients": {
            "midnight_weather_residual_weight": fit.coefficients[0],
            "noon_weather_residual_weight": fit.coefficients[1],
        },
        "bootstrap_metrics": {
            "iterations": int(bootstrap.shape[0]),
            "block_days": BOOTSTRAP_BLOCK_DAYS,
            "seed": BOOTSTRAP_SEED ^ int(fit.origin_time.timestamp()),
            "lower_quantile": LOWER_QUANTILE,
            "midnight_weight_p10": float(np.quantile(bootstrap[:, 0], LOWER_QUANTILE)),
            "midnight_weight_p90": float(np.quantile(bootstrap[:, 0], 0.90)),
            "noon_weight_p10": float(np.quantile(bootstrap[:, 1], LOWER_QUANTILE)),
            "noon_weight_p90": float(np.quantile(bootstrap[:, 1], 0.90)),
        },
        "fit_metrics": fit.fit_metrics,
        "converged": fit.converged,
    }


def _replace_probability(
    candidate: dict[str, Any],
    *,
    probability: float,
    probability_lower: float,
    fit: MarketOffsetFit,
) -> None:
    market_probability = float(candidate["market_probability_proxy"])
    weather_probability = float(candidate["weather_probability"])
    candidate["probability"] = probability
    candidate["probability_lower"] = min(probability, probability_lower)
    candidate["market_probability_input"] = market_probability
    candidate["weather_market_logit_residual"] = float(
        _logit(weather_probability) - _logit(market_probability)
    )
    candidate["opportunity_fit_id"] = fit.opportunity_fit_id
    candidate["probability_source"] = "residual_market_offset"
    candidate["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    all_in = candidate.get("all_in_cost_per_share")
    if candidate.get("executable") and all_in is not None:
        point_edge = probability - float(all_in)
        robust_edge = candidate["probability_lower"] - float(all_in)
        candidate["model_edge_per_share"] = point_edge
        candidate["robust_edge_per_share"] = robust_edge
        candidate["expected_roi"] = point_edge / float(all_in)
        candidate["robust_expected_roi"] = robust_edge / float(all_in)


def apply_rolling_market_offset(
    candidates: list[dict[str, Any]],
    *,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored = copy.deepcopy(candidates)
    training_rows = contract_training_rows(scored)
    by_origin: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for candidate in scored:
        by_origin[candidate["decision_time"]].append(candidate)

    active_fit: MarketOffsetFit | None = None
    active_generation = -1
    fit_records = []
    for origin_time in sorted(by_origin):
        available_rows = [
            row
            for row in training_rows
            if row.decision_time < origin_time and row.label_available_at < origin_time
        ]
        available_dates = {row.event_date for row in available_rows}
        generation = (
            (len(available_dates) - WARMUP_EVENT_DAYS) // REFIT_INTERVAL_EVENT_DAYS
            if len(available_dates) >= WARMUP_EVENT_DAYS
            else -1
        )
        if generation >= 0 and generation > active_generation:
            active_fit = fit_market_offset(
                available_rows,
                origin_time=origin_time,
                bootstrap_iterations=bootstrap_iterations,
            )
            active_generation = generation
            fit_records.append(_fit_record(active_fit))

        for candidate in by_origin[origin_time]:
            candidate.setdefault("weather_probability", candidate["probability"])
            candidate.setdefault("weather_probability_lower", candidate["probability_lower"])
            candidate.setdefault("probability_source", "weather_distribution")
            cost = candidate.get("all_in_cost_per_share")
            if (
                not candidate.get("executable")
                or cost is None
                or not MINIMUM_ALL_IN_COST <= float(cost) <= MAXIMUM_ALL_IN_COST
            ):
                candidate["rejection_reasons"] = sorted(
                    set(candidate["rejection_reasons"])
                    | {"outside_residual_opportunity_universe"}
                )
                continue
            market_probability = candidate.get("market_probability_proxy")
            if market_probability is None:
                candidate["rejection_reasons"] = sorted(
                    set(candidate["rejection_reasons"])
                    | {"missing_causal_market_probability"}
                )
                continue
            if not _has_causal_market_quality(candidate):
                candidate["rejection_reasons"] = sorted(
                    set(candidate["rejection_reasons"])
                    | {"residual_market_quality_failure"}
                )
                continue
            if active_fit is None:
                candidate["probability_source"] = "residual_model_warmup"
                candidate["feature_schema_version"] = FEATURE_SCHEMA_VERSION
                candidate["rejection_reasons"] = sorted(
                    set(candidate["rejection_reasons"]) | {"residual_model_warmup"}
                )
                continue
            hour_index = _decision_hour_index(int(candidate["decision_hour_local"]))
            probability = market_offset_probability(
                float(candidate["weather_probability"]),
                float(market_probability),
                active_fit.coefficients[hour_index],
            )
            sampled_probabilities = np.asarray(
                [
                    market_offset_probability(
                        float(candidate["weather_probability_lower"]),
                        float(market_probability),
                        float(weight),
                    )
                    for weight in active_fit.bootstrap_coefficients[:, hour_index]
                ],
                dtype=np.float64,
            )
            lower = float(np.quantile(sampled_probabilities, LOWER_QUANTILE))
            _replace_probability(
                candidate,
                probability=probability,
                probability_lower=lower,
                fit=active_fit,
            )
    return scored, fit_records
