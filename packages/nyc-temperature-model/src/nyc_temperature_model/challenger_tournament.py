from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from typing import Any

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from . import PROCESS_ID
from .asymmetric_benchmark import (
    _build_candidates,
    _candidate_coverage,
    _date_range,
    _json_default,
    _json_safe,
    _market_rows,
    _policy_metrics,
)
from .config import Settings
from .modeling import (
    FEATURE_NAMES,
    FeatureRow,
    _impute,
    _matrix,
    _training_imputation_medians,
    build_feature_rows,
    raw_point_prediction,
)
from .residual_opportunity_benchmark import (
    FIXED_POLICY,
    _compact_policy_metrics,
    _equal_blend_candidates,
    _market_only_candidates,
    _selected_trade_stress,
)
from .sources import file_sha256

SCHEMA_VERSION = "nyc-temperature-challenger-tournament-v1"
SUPPORT_MIN_F = -20
SUPPORT_MAX_F = 130
SUPPORT = np.arange(SUPPORT_MIN_F, SUPPORT_MAX_F + 1, dtype=np.int64)
SMOOTHING = 0.5
RANDOM_SEED = 24_081_921
CONDITIONAL_NEIGHBORS = 90
ANALOG_NEIGHBORS = 60
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
ENSEMBLE_COMPONENTS = (
    "linear_bias",
    "histogram_residual",
    "conditional_residual",
    "quantile_distribution",
    "analog_ensemble",
)
MODEL_CANDIDATES = (
    "seasonal_climatology",
    "raw_hrrr",
    "linear_bias",
    "histogram_residual",
    "conditional_residual",
    "quantile_distribution",
    "analog_ensemble",
    "convex_ensemble",
)
PROBABILITY_LOWER_QUANTILE = 0.10
PROBABILITY_EFFECTIVE_DAYS = 365.0


def _runtime_provenance(weather_model_image_id: str) -> dict[str, Any]:
    revision = os.environ.get("POLYMARKET_GIT_REVISION", "")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("POLYMARKET_GIT_REVISION must be a 40-character lowercase Git revision")
    if len(weather_model_image_id) != 71 or not weather_model_image_id.startswith("sha256:"):
        raise ValueError("weather model image ID must be sha256:<64 lowercase hex characters>")
    if any(character not in "0123456789abcdef" for character in weather_model_image_id[7:]):
        raise ValueError("weather model image ID contains non-hexadecimal characters")
    return {
        "git_revision": revision,
        "runner_declared_weather_model_image_id": weather_model_image_id,
        "weather_model_image_id_source_contract": (
            "runner supplies .Id from docker image inspect for the exact launched image"
        ),
        "generated_at": datetime.now(UTC),
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _row_digest(rows_by_hour: dict[int, list[FeatureRow]]) -> str:
    values = []
    for hour in sorted(rows_by_hour):
        for row in rows_by_hour[hour]:
            values.append(
                {
                    "event_date": row.event_date,
                    "decision_time": row.decision_time,
                    "decision_hour_local": row.decision_hour_local,
                    "features": row.vector(),
                    "target_daily_max_f": row.target_daily_max_f,
                    "target_rounded_max_f": row.target_rounded_max_f,
                    "observation_count": row.observation_count,
                }
            )
    return _canonical_sha256(values)


def _rounded_samples_probability(samples: np.ndarray) -> np.ndarray:
    rounded = np.floor(np.asarray(samples, dtype=np.float64) + 0.5).astype(np.int64)
    rounded = np.clip(rounded, SUPPORT_MIN_F, SUPPORT_MAX_F)
    counts = np.bincount(rounded - SUPPORT_MIN_F, minlength=SUPPORT.size)
    probabilities = (counts + SMOOTHING) / (counts.sum() + SMOOTHING * SUPPORT.size)
    if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-12):
        raise ValueError("temperature probabilities must sum to one")
    return probabilities


def _residual_probabilities(points: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_rounded_samples_probability(float(point) + residuals) for point in points],
        dtype=np.float64,
    )


def _fit_point_models(
    train_rows: list[FeatureRow], calibration_rows: list[FeatureRow], decision_hour: int
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_matrix = _matrix(train_rows)
    medians, structurally_missing = _training_imputation_medians(train_matrix, decision_hour)
    train_matrix = _impute(train_matrix, medians)
    calibration_matrix = _impute(_matrix(calibration_rows), medians)
    target = np.asarray([row.target_daily_max_f for row in train_rows])
    ridge = Ridge(alpha=1.0).fit(train_matrix, target)
    histogram = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    ).fit(train_matrix, target)
    train_points = {
        "raw_hrrr": raw_point_prediction(train_rows),
        "linear_bias": np.asarray(ridge.predict(train_matrix), dtype=np.float64),
        "histogram_residual": np.asarray(histogram.predict(train_matrix), dtype=np.float64),
    }
    calibration_points = {
        "raw_hrrr": raw_point_prediction(calibration_rows),
        "linear_bias": np.asarray(ridge.predict(calibration_matrix), dtype=np.float64),
        "histogram_residual": np.asarray(histogram.predict(calibration_matrix), dtype=np.float64),
    }
    return (
        {
            "imputation_medians": medians,
            "structurally_missing_features": structurally_missing,
            "ridge": ridge,
            "histogram": histogram,
        },
        train_points,
        calibration_points,
    )


def _seasonal_distance(first: date, second: date) -> float:
    difference = abs(first.timetuple().tm_yday - second.timetuple().tm_yday)
    return float(min(difference, 366 - difference))


def _conditional_probability_rows(
    reference_points: np.ndarray,
    reference_targets: np.ndarray,
    reference_rows: list[FeatureRow],
    query_points: np.ndarray,
    query_rows: list[FeatureRow],
) -> np.ndarray:
    residuals = reference_targets - reference_points
    reference_boundary = np.abs((reference_points - 0.5) - np.round(reference_points - 0.5))
    output = []
    for point, row in zip(query_points, query_rows, strict=True):
        query_boundary = abs((point - 0.5) - round(point - 0.5))
        distances = np.asarray(
            [
                (_seasonal_distance(row.event_date, other.event_date) / 45.0) ** 2
                + ((point - reference_points[position]) / 5.0) ** 2
                + ((query_boundary - reference_boundary[position]) / 0.25) ** 2
                for position, other in enumerate(reference_rows)
            ],
            dtype=np.float64,
        )
        count = min(CONDITIONAL_NEIGHBORS, len(reference_rows))
        selected = np.argpartition(distances, count - 1)[:count]
        output.append(_rounded_samples_probability(float(point) + residuals[selected]))
    return np.asarray(output, dtype=np.float64)


def _quantile_samples(values: np.ndarray, sample_count: int = 399) -> np.ndarray:
    ordered = np.maximum.accumulate(np.asarray(values, dtype=np.float64))
    lower_span = max(0.25, ordered[1] - ordered[0])
    upper_span = max(0.25, ordered[-1] - ordered[-2])
    value_knots = np.concatenate(([ordered[0] - lower_span], ordered, [ordered[-1] + upper_span]))
    probability_knots = np.asarray((0.005, *QUANTILES, 0.995), dtype=np.float64)
    samples = np.linspace(0.0025, 0.9975, sample_count)
    return np.interp(samples, probability_knots, value_knots)


def _fit_quantile_models(
    train_matrix: np.ndarray,
    train_target: np.ndarray,
    calibration_matrix: np.ndarray,
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    estimators = []
    predictions = []
    for quantile in QUANTILES:
        estimator = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=RANDOM_SEED,
        ).fit(train_matrix, train_target)
        estimators.append(estimator)
        predictions.append(estimator.predict(calibration_matrix))
    quantile_predictions = np.maximum.accumulate(
        np.asarray(predictions, dtype=np.float64).T, axis=1
    )
    probabilities = np.asarray(
        [_rounded_samples_probability(_quantile_samples(row)) for row in quantile_predictions],
        dtype=np.float64,
    )
    return estimators, quantile_predictions[:, QUANTILES.index(0.50)], probabilities


def _fit_analog(
    train_rows: list[FeatureRow],
    calibration_rows: list[FeatureRow],
    medians: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    train_matrix = _impute(_matrix(train_rows), medians)
    calibration_matrix = _impute(_matrix(calibration_rows), medians)
    center = np.mean(train_matrix, axis=0)
    scale = np.std(train_matrix, axis=0)
    scale[scale < 1e-9] = 1.0
    standardized_train = (train_matrix - center) / scale
    standardized_calibration = (calibration_matrix - center) / scale
    train_raw = raw_point_prediction(train_rows)
    calibration_raw = raw_point_prediction(calibration_rows)
    train_target = np.asarray([row.target_daily_max_f for row in train_rows])
    residuals = train_target - train_raw
    probabilities = []
    points = []
    for raw, vector in zip(calibration_raw, standardized_calibration, strict=True):
        distances = np.sum((standardized_train - vector) ** 2, axis=1)
        selected = np.argpartition(distances, ANALOG_NEIGHBORS - 1)[:ANALOG_NEIGHBORS]
        samples = float(raw) + residuals[selected]
        probabilities.append(_rounded_samples_probability(samples))
        points.append(float(np.median(samples)))
    return (
        {
            "center": center,
            "scale": scale,
            "standardized_train": standardized_train,
            "train_residuals": residuals,
        },
        np.asarray(points, dtype=np.float64),
        np.asarray(probabilities, dtype=np.float64),
    )


def _climatology_probabilities(
    train_rows: list[FeatureRow], calibration_rows: list[FeatureRow]
) -> tuple[np.ndarray, np.ndarray]:
    train_target = np.asarray([row.target_daily_max_f for row in train_rows])
    output = []
    points = []
    for row in calibration_rows:
        distances = np.asarray(
            [_seasonal_distance(row.event_date, other.event_date) for other in train_rows]
        )
        selected = np.flatnonzero(distances <= 45)
        samples = train_target[selected]
        output.append(_rounded_samples_probability(samples))
        points.append(float(np.median(samples)))
    return np.asarray(points), np.asarray(output)


def _fit_ensemble_weights(
    probability_matrices: dict[str, np.ndarray], targets: np.ndarray
) -> np.ndarray:
    stacked = np.stack([probability_matrices[name] for name in ENSEMBLE_COMPONENTS], axis=1)
    target_indices = (
        np.clip(np.floor(targets + 0.5).astype(np.int64), SUPPORT_MIN_F, SUPPORT_MAX_F)
        - SUPPORT_MIN_F
    )

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        selected = stacked[np.arange(stacked.shape[0]), :, target_indices]
        mixed = np.clip(selected @ weights, 1e-12, 1.0)
        loss = -float(np.mean(np.log(mixed)))
        gradient = -np.mean(selected / mixed[:, None], axis=0)
        return loss, gradient

    initial = np.full(len(ENSEMBLE_COMPONENTS), 1.0 / len(ENSEMBLE_COMPONENTS))
    result = minimize(
        objective,
        initial,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(initial),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise ValueError(f"convex ensemble optimization failed: {result.message}")
    weights = np.clip(np.asarray(result.x), 0.0, 1.0)
    return weights / weights.sum()


def _cross_fitted_ensemble(
    probability_matrices: dict[str, np.ndarray],
    targets: np.ndarray,
    rows: list[FeatureRow],
) -> tuple[np.ndarray, np.ndarray]:
    months = np.asarray([row.event_date.month for row in rows], dtype=np.int64)
    output = np.empty_like(next(iter(probability_matrices.values())))
    fold_weights = []
    for month in sorted(set(months)):
        train_mask = months != month
        test_mask = ~train_mask
        weights = _fit_ensemble_weights(
            {name: matrix[train_mask] for name, matrix in probability_matrices.items()},
            targets[train_mask],
        )
        output[test_mask] = sum(
            weights[index] * probability_matrices[name][test_mask]
            for index, name in enumerate(ENSEMBLE_COMPONENTS)
        )
        fold_weights.append(weights)
    final_weights = _fit_ensemble_weights(probability_matrices, targets)
    return output, np.asarray((*fold_weights, final_weights), dtype=np.float64)


def _distribution_metrics(
    probabilities: np.ndarray, points: np.ndarray, rows: list[FeatureRow]
) -> tuple[dict[str, Any], np.ndarray]:
    targets = np.asarray([row.target_daily_max_f for row in rows], dtype=np.float64)
    rounded = np.clip(np.floor(targets + 0.5).astype(np.int64), SUPPORT_MIN_F, SUPPORT_MAX_F)
    target_indices = rounded - SUPPORT_MIN_F
    winner_probabilities = np.clip(probabilities[np.arange(len(rows)), target_indices], 1e-12, 1.0)
    daily_log_loss = -np.log(winner_probabilities)
    cdf = np.cumsum(probabilities, axis=1)
    observed = SUPPORT[None, :] >= rounded[:, None]
    daily_rps = np.sum((cdf - observed) ** 2, axis=1)
    point_error = points - targets
    predicted_bucket = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predicted_bucket == target_indices
    bins = np.minimum((confidence * 10).astype(int), 9)
    calibration_error = 0.0
    for bin_index in range(10):
        selected = bins == bin_index
        if selected.any():
            calibration_error += float(
                selected.mean()
                * abs(float(confidence[selected].mean()) - float(correct[selected].mean()))
            )
    point_boundary_distance = np.abs((points - 0.5) - np.round(points - 0.5))
    boundary = point_boundary_distance <= 0.25
    return (
        {
            "days": len(rows),
            "rounded_temperature_log_loss": float(np.mean(daily_log_loss)),
            "ranked_probability_score": float(np.mean(daily_rps)),
            "top_bucket_calibration_error": calibration_error,
            "bucket_boundary_days": int(boundary.sum()),
            "bucket_boundary_log_loss": (
                float(np.mean(daily_log_loss[boundary])) if boundary.any() else None
            ),
            "temperature_mae_f": float(np.mean(np.abs(point_error))),
            "temperature_rmse_f": float(np.mean(point_error**2) ** 0.5),
            "exact_degree_rate": float(np.mean(np.floor(points + 0.5) == rounded)),
            "within_one_f_rate": float(np.mean(np.abs(point_error) <= 1.0)),
            "monthly_log_loss": {
                str(month): float(
                    np.mean(
                        daily_log_loss[np.asarray([row.event_date.month == month for row in rows])]
                    )
                )
                for month in range(1, 13)
            },
        },
        daily_log_loss,
    )


def _train_hour(
    train_rows: list[FeatureRow], calibration_rows: list[FeatureRow], decision_hour: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    if len(train_rows) < 365 or len(calibration_rows) < 90:
        raise ValueError("challenger tournament requires complete training and calibration years")
    target = np.asarray([row.target_daily_max_f for row in calibration_rows])
    point_bundle, train_points, points = _fit_point_models(
        train_rows, calibration_rows, decision_hour
    )
    train_matrix = _impute(_matrix(train_rows), np.asarray(point_bundle["imputation_medians"]))
    calibration_matrix = _impute(
        _matrix(calibration_rows), np.asarray(point_bundle["imputation_medians"])
    )
    train_target = np.asarray([row.target_daily_max_f for row in train_rows])
    probability_matrices = {
        name: _residual_probabilities(points[name], train_target - train_points[name])
        for name in ("raw_hrrr", "linear_bias", "histogram_residual")
    }
    probability_matrices["conditional_residual"] = _conditional_probability_rows(
        train_points["linear_bias"],
        train_target,
        train_rows,
        points["linear_bias"],
        calibration_rows,
    )
    points["conditional_residual"] = points["linear_bias"]
    quantile_estimators, quantile_points, quantile_probabilities = _fit_quantile_models(
        train_matrix, train_target, calibration_matrix
    )
    points["quantile_distribution"] = quantile_points
    probability_matrices["quantile_distribution"] = quantile_probabilities
    analog_bundle, analog_points, analog_probabilities = _fit_analog(
        train_rows,
        calibration_rows,
        np.asarray(point_bundle["imputation_medians"]),
    )
    points["analog_ensemble"] = analog_points
    probability_matrices["analog_ensemble"] = analog_probabilities
    climatology_points, climatology_probabilities = _climatology_probabilities(
        train_rows, calibration_rows
    )
    points["seasonal_climatology"] = climatology_points
    probability_matrices["seasonal_climatology"] = climatology_probabilities
    ensemble_probabilities, ensemble_weights = _cross_fitted_ensemble(
        probability_matrices, target, calibration_rows
    )
    probability_matrices["convex_ensemble"] = ensemble_probabilities
    points["convex_ensemble"] = ensemble_probabilities @ SUPPORT.astype(np.float64)

    metrics = {}
    daily_losses = {}
    for candidate in MODEL_CANDIDATES:
        metrics[candidate], daily_losses[candidate] = _distribution_metrics(
            probability_matrices[candidate], points[candidate], calibration_rows
        )
    artifact = {
        "decision_hour_local": decision_hour,
        "feature_names": FEATURE_NAMES,
        "point_bundle": point_bundle,
        "quantile_estimators": quantile_estimators,
        "analog_bundle": analog_bundle,
        "training_rows": train_rows,
        "calibration_rows": calibration_rows,
        "calibration_points": points,
        "calibration_targets": target,
        "calibration_residuals": {
            name: target - points[name]
            for name in ("raw_hrrr", "linear_bias", "histogram_residual")
        },
        "ensemble_final_weights": ensemble_weights[-1],
        "ensemble_month_fold_weights": ensemble_weights[:-1],
    }
    return artifact, metrics, daily_losses


def _select_champion(
    metrics_by_hour: dict[int, dict[str, Any]],
    daily_losses_by_hour: dict[int, dict[str, np.ndarray]],
) -> tuple[str, dict[str, Any]]:
    ridge = np.concatenate([daily_losses_by_hour[hour]["linear_bias"] for hour in (0, 12)])
    decisions = []
    eligible = []
    for candidate in MODEL_CANDIDATES:
        losses = np.concatenate([daily_losses_by_hour[hour][candidate] for hour in (0, 12)])
        differences = ridge - losses
        standard_error = float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
        improvement = float(np.mean(differences))
        aggregate_rps = float(
            np.mean(
                [metrics_by_hour[hour][candidate]["ranked_probability_score"] for hour in (0, 12)]
            )
        )
        ridge_rps = float(
            np.mean(
                [
                    metrics_by_hour[hour]["linear_bias"]["ranked_probability_score"]
                    for hour in (0, 12)
                ]
            )
        )
        passes = (
            candidate != "seasonal_climatology"
            and improvement > standard_error
            and aggregate_rps <= ridge_rps
        )
        decisions.append(
            {
                "candidate": candidate,
                "mean_log_loss_improvement_over_linear_bias": improvement,
                "paired_daily_standard_error": standard_error,
                "aggregate_ranked_probability_score": aggregate_rps,
                "linear_bias_ranked_probability_score": ridge_rps,
                "passes_one_standard_error_rule": passes,
            }
        )
        if passes:
            eligible.append(candidate)
    champion = min(
        eligible,
        key=lambda name: np.mean(
            [metrics_by_hour[hour][name]["rounded_temperature_log_loss"] for hour in (0, 12)]
        ),
        default="linear_bias",
    )
    return champion, {"candidate_decisions": decisions, "selected_champion": champion}


def _predict_hour(
    artifact: dict[str, Any], rows: list[FeatureRow]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    medians = np.asarray(artifact["point_bundle"]["imputation_medians"])
    matrix = _impute(_matrix(rows), medians)
    raw = raw_point_prediction(rows)
    ridge = np.asarray(artifact["point_bundle"]["ridge"].predict(matrix))
    histogram = np.asarray(artifact["point_bundle"]["histogram"].predict(matrix))
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "raw_hrrr": (
            raw,
            _residual_probabilities(raw, artifact["calibration_residuals"]["raw_hrrr"]),
        ),
        "linear_bias": (
            ridge,
            _residual_probabilities(ridge, artifact["calibration_residuals"]["linear_bias"]),
        ),
        "histogram_residual": (
            histogram,
            _residual_probabilities(
                histogram,
                artifact["calibration_residuals"]["histogram_residual"],
            ),
        ),
    }
    calibration_rows = artifact["calibration_rows"]
    calibration_points = np.asarray(artifact["calibration_points"]["linear_bias"])
    calibration_targets = np.asarray(artifact["calibration_targets"])
    calibration_residuals = calibration_targets - calibration_points
    boundary = np.abs((calibration_points - 0.5) - np.round(calibration_points - 0.5))
    conditional = []
    for point, row in zip(ridge, rows, strict=True):
        point_boundary = abs((point - 0.5) - round(point - 0.5))
        distances = np.asarray(
            [
                (_seasonal_distance(row.event_date, other.event_date) / 45.0) ** 2
                + ((point - calibration_points[index]) / 5.0) ** 2
                + ((point_boundary - boundary[index]) / 0.25) ** 2
                for index, other in enumerate(calibration_rows)
            ]
        )
        selected = np.argpartition(distances, CONDITIONAL_NEIGHBORS - 1)[:CONDITIONAL_NEIGHBORS]
        conditional.append(
            _rounded_samples_probability(float(point) + calibration_residuals[selected])
        )
    predictions["conditional_residual"] = (ridge, np.asarray(conditional))

    quantiles = np.maximum.accumulate(
        np.asarray([estimator.predict(matrix) for estimator in artifact["quantile_estimators"]]).T,
        axis=1,
    )
    quantile_probabilities = np.asarray(
        [_rounded_samples_probability(_quantile_samples(row)) for row in quantiles]
    )
    predictions["quantile_distribution"] = (
        quantiles[:, QUANTILES.index(0.50)],
        quantile_probabilities,
    )
    analog = artifact["analog_bundle"]
    standardized = (matrix - analog["center"]) / analog["scale"]
    analog_points = []
    analog_probabilities = []
    for raw_point, vector in zip(raw, standardized, strict=True):
        distances = np.sum((analog["standardized_train"] - vector) ** 2, axis=1)
        selected = np.argpartition(distances, ANALOG_NEIGHBORS - 1)[:ANALOG_NEIGHBORS]
        samples = float(raw_point) + analog["train_residuals"][selected]
        analog_points.append(float(np.median(samples)))
        analog_probabilities.append(_rounded_samples_probability(samples))
    predictions["analog_ensemble"] = (
        np.asarray(analog_points),
        np.asarray(analog_probabilities),
    )
    climatology_points, climatology_probabilities = _climatology_probabilities(
        artifact["training_rows"], rows
    )
    predictions["seasonal_climatology"] = (
        climatology_points,
        climatology_probabilities,
    )
    ensemble = sum(
        artifact["ensemble_final_weights"][index] * predictions[name][1]
        for index, name in enumerate(ENSEMBLE_COMPONENTS)
    )
    predictions["convex_ensemble"] = (ensemble @ SUPPORT, ensemble)
    return predictions


def _bucket_probability(pmf: np.ndarray, lower: int | None, upper: int | None) -> float:
    selected = np.ones(SUPPORT.shape, dtype=bool)
    if lower is not None:
        selected &= SUPPORT >= lower
    if upper is not None:
        selected &= SUPPORT <= upper
    return float(pmf[selected].sum())


def _probability_lower(probability: float) -> float:
    if probability <= 0:
        return 0.0
    if probability >= 1:
        return 1.0
    lower = beta.ppf(
        PROBABILITY_LOWER_QUANTILE,
        probability * PROBABILITY_EFFECTIVE_DAYS + 0.5,
        (1.0 - probability) * PROBABILITY_EFFECTIVE_DAYS + 0.5,
    )
    return min(probability, float(lower))


def _economic_probability_rows(
    settings: Settings,
    artifacts: dict[int, dict[str, Any]],
    *,
    start: date,
    end: date,
    tournament_id: str,
) -> dict[str, list[dict[str, Any]]]:
    markets = _market_rows(settings.database_url, start, end)
    markets_by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for market in markets:
        markets_by_date[market["event_date"]].append(market)
    output = {candidate: [] for candidate in MODEL_CANDIDATES}
    for hour in (0, 12):
        rows = build_feature_rows(settings.database_url, start, end + timedelta(days=1), hour)
        predictions = _predict_hour(artifacts[hour], rows)
        for candidate, (points, probability_matrix) in predictions.items():
            for row, point, pmf in zip(rows, points, probability_matrix, strict=True):
                day_markets = markets_by_date.get(row.event_date, [])
                if not day_markets:
                    continue
                probabilities = [
                    _bucket_probability(pmf, market["bucket_lower_f"], market["bucket_upper_f"])
                    for market in day_markets
                ]
                if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
                    raise ValueError(
                        f"market buckets do not partition probability on {row.event_date}"
                    )
                if sum(int(market["resolved_yes"]) for market in day_markets) != 1:
                    raise ValueError(f"market event does not have one winner on {row.event_date}")
                for market, probability in zip(day_markets, probabilities, strict=True):
                    no_probability = 1.0 - probability
                    output[candidate].append(
                        {
                            **market,
                            "model_run_id": f"{tournament_id}:{candidate}:h{hour:02d}",
                            "decision_time": row.decision_time,
                            "decision_hour_local": hour,
                            "point_prediction_f": float(point),
                            "probability_yes": probability,
                            "probability_yes_lower": _probability_lower(probability),
                            "probability_no": no_probability,
                            "probability_no_lower": _probability_lower(no_probability),
                        }
                    )
    return output


def _economic_benchmark(
    settings: Settings,
    artifacts: dict[int, dict[str, Any]],
    *,
    champion: str,
    tournament_id: str,
    start: date,
    end: date,
    sealed_start: date,
) -> dict[str, Any]:
    probability_rows = _economic_probability_rows(
        settings, artifacts, start=start, end=end, tournament_id=tournament_id
    )
    periods = {
        "retrospective": (start, min(end, sealed_start - timedelta(days=1))),
        "sealed_post_freeze_backfill": (max(start, sealed_start), end),
    }
    output: dict[str, Any] = {"candidates": {}, "periods": {}}
    selected_by_period: dict[str, list[dict[str, Any]]] = {}
    candidates_by_model: dict[str, list[dict[str, Any]]] = {}
    for candidate, rows in probability_rows.items():
        candidates_by_model[candidate] = _build_candidates(
            settings.database_url,
            probability_rows=rows,
            start=start,
            end=end,
            quantity=5.0,
            modeled_slippage_per_share=0.01,
        )
    for period_name, (period_start, period_end) in periods.items():
        if period_start > period_end:
            output["periods"][period_name] = {
                "start": period_start,
                "end": period_end,
                "available": False,
            }
            continue
        dates = _date_range(period_start, period_end)
        output["periods"][period_name] = {
            "start": period_start,
            "end": period_end,
            "available": True,
        }
        for candidate in MODEL_CANDIDATES:
            period_candidates = [
                row
                for row in candidates_by_model[candidate]
                if period_start <= row["event_date"] <= period_end
            ]
            metrics, selected, _ = _policy_metrics(period_candidates, dates, FIXED_POLICY)
            output["candidates"].setdefault(candidate, {})[period_name] = {
                "metrics": _compact_policy_metrics(metrics),
                "candidate_coverage": _candidate_coverage(period_candidates),
            }
            if candidate == champion:
                selected_by_period[period_name] = selected
                output["candidates"][candidate][period_name]["slippage_stress"] = {
                    f"{slippage:.3f}": _selected_trade_stress(selected, slippage)
                    for slippage in (0.0, 0.005, 0.01, 0.02)
                }

    champion_candidates = candidates_by_model[champion]
    market_candidates = _market_only_candidates(champion_candidates)
    blend_input = []
    for row in champion_candidates:
        copy = dict(row)
        copy["weather_probability"] = copy["probability"]
        copy["weather_probability_lower"] = copy["probability_lower"]
        blend_input.append(copy)
    blend_candidates = _equal_blend_candidates(blend_input)
    output["comparators"] = {"no_trade": {"total_net": 0.0}}
    for comparator_name, comparator_candidates in (
        ("market_only", market_candidates),
        ("equal_logit_blend", blend_candidates),
    ):
        output["comparators"][comparator_name] = {}
        for period_name, (period_start, period_end) in periods.items():
            if period_start > period_end:
                continue
            metrics, _, _ = _policy_metrics(
                [
                    row
                    for row in comparator_candidates
                    if period_start <= row["event_date"] <= period_end
                ],
                _date_range(period_start, period_end),
                FIXED_POLICY,
            )
            output["comparators"][comparator_name][period_name] = _compact_policy_metrics(metrics)
    output["champion"] = champion
    output["execution_contract"] = asdict(FIXED_POLICY) | {
        "quantity": 5.0,
        "modeled_slippage_per_share": 0.01,
        "dynamic_captured_fee_schedule": True,
        "maximum_positions_per_event_day": 1,
    }
    output["production_qualified"] = False
    return output


def run_challenger_tournament(
    settings: Settings,
    *,
    training_start: date,
    training_end: date,
    calibration_start: date,
    calibration_end: date,
    economic_start: date,
    economic_end: date,
    sealed_start: date,
    weather_model_image_id: str,
) -> dict[str, Any]:
    if not training_start <= training_end < calibration_start <= calibration_end:
        raise ValueError("training and calibration ranges must be chronological and disjoint")
    if economic_start <= calibration_end or not economic_start <= sealed_start <= economic_end:
        raise ValueError("economic and sealed ranges must follow calibration")
    provenance = _runtime_provenance(weather_model_image_id)
    training_rows = {
        hour: build_feature_rows(
            settings.database_url,
            training_start,
            training_end + timedelta(days=1),
            hour,
        )
        for hour in (0, 12)
    }
    calibration_rows = {
        hour: build_feature_rows(
            settings.database_url,
            calibration_start,
            calibration_end + timedelta(days=1),
            hour,
        )
        for hour in (0, 12)
    }
    source_digest = _row_digest(
        {hour: training_rows[hour] + calibration_rows[hour] for hour in (0, 12)}
    )
    artifacts = {}
    metrics_by_hour = {}
    daily_losses_by_hour = {}
    for hour in (0, 12):
        artifacts[hour], metrics_by_hour[hour], daily_losses_by_hour[hour] = _train_hour(
            training_rows[hour], calibration_rows[hour], hour
        )
    champion, selection = _select_champion(metrics_by_hour, daily_losses_by_hour)
    tournament_id = str(uuid.uuid4())
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "process_id": PROCESS_ID,
        "tournament_id": tournament_id,
        "source_feature_rows_sha256": source_digest,
        "training_start": training_start,
        "training_end": training_end,
        "calibration_start": calibration_start,
        "calibration_end": calibration_end,
        "candidate_names": MODEL_CANDIDATES,
        "selected_champion": champion,
        "hours": artifacts,
        "random_seed": RANDOM_SEED,
    }
    settings.model_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = settings.model_directory / f"challenger-tournament-{tournament_id}.joblib"
    temporary_artifact = artifact_path.with_suffix(".partial")
    joblib.dump(artifact, temporary_artifact, compress=3)
    temporary_artifact.replace(artifact_path)
    artifact_sha256, artifact_size = file_sha256(artifact_path)
    economic = _economic_benchmark(
        settings,
        artifacts,
        champion=champion,
        tournament_id=tournament_id,
        start=economic_start,
        end=economic_end,
        sealed_start=sealed_start,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "process_id": PROCESS_ID,
        "tournament_id": tournament_id,
        "objective": ("calibrated_rounded_temperature_distribution_then_positive_net_expectancy"),
        "provenance": provenance,
        "data_contract": {
            "training_start": training_start,
            "training_end": training_end,
            "calibration_start": calibration_start,
            "calibration_end": calibration_end,
            "economic_start": economic_start,
            "economic_end": economic_end,
            "sealed_start": sealed_start,
            "source_feature_rows_sha256": source_digest,
            "training_days": {str(hour): len(training_rows[hour]) for hour in (0, 12)},
            "calibration_days": {str(hour): len(calibration_rows[hour]) for hour in (0, 12)},
        },
        "model_contract": {
            "support_min_f": SUPPORT_MIN_F,
            "support_max_f": SUPPORT_MAX_F,
            "smoothing": SMOOTHING,
            "conditional_neighbors": CONDITIONAL_NEIGHBORS,
            "analog_neighbors": ANALOG_NEIGHBORS,
            "quantiles": QUANTILES,
            "ensemble_components": ENSEMBLE_COMPONENTS,
            "random_seed": RANDOM_SEED,
            "market_prices_used_for_weather_training": False,
            "pnl_used_for_model_selection": False,
        },
        "forecast_metrics": {str(hour): metrics_by_hour[hour] for hour in (0, 12)},
        "selection": selection,
        "selected_champion": champion,
        "artifact_uri": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": artifact_size,
        "economic_benchmark": economic,
        "production_qualified": False,
    }
    settings.report_directory.mkdir(parents=True, exist_ok=True)
    report_path = settings.report_directory / f"challenger-tournament-{tournament_id}.json"
    report["report_uri"] = str(report_path)
    safe_report = _json_safe(report)
    temporary_report = report_path.with_suffix(".partial")
    temporary_report.write_text(
        json.dumps(safe_report, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    temporary_report.replace(report_path)
    return safe_report
