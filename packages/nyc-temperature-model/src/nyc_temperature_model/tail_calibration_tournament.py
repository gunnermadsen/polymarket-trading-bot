from __future__ import annotations

import copy
import json
import math
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from typing import Any

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta, norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import PROCESS_ID
from .asymmetric_benchmark import (
    _build_candidates,
    _candidate_coverage,
    _date_range,
    _json_default,
    _json_safe,
    _policy_metrics,
)
from .challenger_tournament import (
    RANDOM_SEED,
    SUPPORT,
    SUPPORT_MAX_F,
    SUPPORT_MIN_F,
    _bucket_probability,
    _distribution_metrics,
    _pmf_median,
    _predict_hour,
    _row_digest,
    _train_hour,
)
from .config import Settings
from .modeling import FeatureRow, _impute, _matrix, build_feature_rows
from .residual_opportunity import market_offset_probability
from .residual_opportunity_benchmark import (
    FIXED_POLICY,
    _compact_policy_metrics,
    _selected_trade_stress,
)
from .sources import file_sha256

SCHEMA_VERSION = "nyc-temperature-tail-calibration-tournament-v1"
MODEL_CANDIDATES = (
    "linear_bias",
    "histogram_residual",
    "heteroskedastic_residual",
    "ordinal_temperature_classifier",
    "tail_calibrated_ensemble",
)
POLICY_CANDIDATES = (
    "weather_only",
    "market_shrunk_50pct",
    "selection_adjusted",
)
TAIL_PROBABILITY_BANDS = (
    (0.00, 0.04, "0-4pct"),
    (0.04, 0.08, "4-8pct"),
    (0.08, 0.12, "8-12pct"),
    (0.12, 0.20, "12-20pct"),
    (0.20, 0.35, "20-35pct"),
)
SELECTION_FAMILYWISE_ALPHA = 0.10


def _validate_revision(revision: str) -> str:
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("git revision must be a 40-character lowercase hexadecimal value")
    return revision


def _validate_image_id(image_id: str) -> str:
    if len(image_id) != 71 or not image_id.startswith("sha256:"):
        raise ValueError("runner image ID must use sha256:<64 lowercase hexadecimal characters>")
    if any(character not in "0123456789abcdef" for character in image_id[7:]):
        raise ValueError("runner image ID must use lowercase hexadecimal characters")
    return image_id


def _gaussian_probabilities(points: np.ndarray, scales: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    scales = np.clip(np.asarray(scales, dtype=np.float64), 0.35, 12.0)
    upper = norm.cdf((SUPPORT[None, :] + 0.5 - points[:, None]) / scales[:, None])
    lower = norm.cdf((SUPPORT[None, :] - 0.5 - points[:, None]) / scales[:, None])
    probabilities = upper - lower
    probabilities[:, 0] += lower[:, 0]
    probabilities[:, -1] += 1.0 - upper[:, -1]
    probabilities = np.clip(probabilities, 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _classifier_probabilities(estimator: Any, matrix: np.ndarray) -> np.ndarray:
    output = np.full((len(matrix), len(SUPPORT)), 1e-12, dtype=np.float64)
    class_probabilities = estimator.predict_proba(matrix)
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    valid = (classes >= SUPPORT_MIN_F) & (classes <= SUPPORT_MAX_F)
    output[:, classes[valid] - SUPPORT_MIN_F] += class_probabilities[:, valid]
    return output / output.sum(axis=1, keepdims=True)


def _temperature_transform(probabilities: np.ndarray, exponent: float) -> np.ndarray:
    transformed = np.power(np.clip(probabilities, 1e-12, 1.0), exponent)
    return transformed / transformed.sum(axis=1, keepdims=True)


def _fit_tail_blend(
    ensemble: np.ndarray, linear: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    target_indices = (
        np.clip(np.floor(targets + 0.5).astype(np.int64), SUPPORT_MIN_F, SUPPORT_MAX_F)
        - SUPPORT_MIN_F
    )

    def objective(parameters: np.ndarray) -> float:
        weight, exponent = parameters
        mixed = weight * ensemble + (1.0 - weight) * linear
        calibrated = _temperature_transform(mixed, exponent)
        winner = calibrated[np.arange(len(target_indices)), target_indices]
        return -float(np.mean(np.log(np.clip(winner, 1e-12, 1.0))))

    result = minimize(
        objective,
        np.asarray([0.5, 1.0]),
        method="L-BFGS-B",
        bounds=((0.0, 1.0), (0.35, 2.5)),
    )
    if not result.success:
        raise ValueError(f"tail calibration optimization failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def _cross_fitted_tail_blend(
    ensemble: np.ndarray,
    linear: np.ndarray,
    targets: np.ndarray,
    rows: list[FeatureRow],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    months = np.asarray([row.event_date.month for row in rows], dtype=np.int64)
    output = np.empty_like(ensemble)
    fold_parameters = []
    for month in sorted(set(months)):
        train = months != month
        test = ~train
        parameters = _fit_tail_blend(ensemble[train], linear[train], targets[train])
        mixed = parameters[0] * ensemble[test] + (1.0 - parameters[0]) * linear[test]
        output[test] = _temperature_transform(mixed, parameters[1])
        fold_parameters.append(parameters)
    final_parameters = _fit_tail_blend(ensemble, linear, targets)
    return output, np.asarray(fold_parameters), final_parameters


def _train_followup_hour(
    train_rows: list[FeatureRow], calibration_rows: list[FeatureRow], decision_hour: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    base_artifact, _, _ = _train_hour(train_rows, calibration_rows, decision_hour)
    medians = np.asarray(base_artifact["point_bundle"]["imputation_medians"])
    train_matrix = _impute(_matrix(train_rows), medians)
    calibration_matrix = _impute(_matrix(calibration_rows), medians)
    train_targets = np.asarray([row.target_daily_max_f for row in train_rows])
    calibration_targets = np.asarray([row.target_daily_max_f for row in calibration_rows])
    train_linear = np.asarray(base_artifact["point_bundle"]["ridge"].predict(train_matrix))
    calibration_linear = np.asarray(
        base_artifact["point_bundle"]["ridge"].predict(calibration_matrix)
    )

    scale_model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=200,
        max_leaf_nodes=11,
        min_samples_leaf=45,
        l2_regularization=2.0,
        random_state=RANDOM_SEED,
    ).fit(train_matrix, np.abs(train_targets - train_linear))
    calibration_scales = np.asarray(scale_model.predict(calibration_matrix)) * 1.4826
    heteroskedastic = _gaussian_probabilities(calibration_linear, calibration_scales)

    rounded_targets = np.clip(
        np.floor(train_targets + 0.5).astype(np.int64), SUPPORT_MIN_F, SUPPORT_MAX_F
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=2.0,
        random_state=RANDOM_SEED,
    ).fit(train_matrix, rounded_targets)
    ordinal = _classifier_probabilities(classifier, calibration_matrix)

    base_probabilities = base_artifact["calibration_probability_matrices"]
    tail_blend, fold_parameters, final_parameters = _cross_fitted_tail_blend(
        np.asarray(base_probabilities["convex_ensemble"]),
        np.asarray(base_probabilities["linear_bias"]),
        calibration_targets,
        calibration_rows,
    )
    probabilities = {
        "linear_bias": np.asarray(base_probabilities["linear_bias"]),
        "histogram_residual": np.asarray(base_probabilities["histogram_residual"]),
        "heteroskedastic_residual": heteroskedastic,
        "ordinal_temperature_classifier": ordinal,
        "tail_calibrated_ensemble": tail_blend,
    }
    points = {
        "linear_bias": np.asarray(base_artifact["calibration_points"]["linear_bias"]),
        "histogram_residual": np.asarray(
            base_artifact["calibration_points"]["histogram_residual"]
        ),
        "heteroskedastic_residual": calibration_linear,
        "ordinal_temperature_classifier": _pmf_median(ordinal),
        "tail_calibrated_ensemble": _pmf_median(tail_blend),
    }
    metrics: dict[str, Any] = {}
    losses: dict[str, np.ndarray] = {}
    for candidate in MODEL_CANDIDATES:
        metrics[candidate], losses[candidate] = _distribution_metrics(
            probabilities[candidate], points[candidate], calibration_rows
        )
    base_artifact["followup"] = {
        "scale_model": scale_model,
        "classifier": classifier,
        "tail_blend_month_fold_parameters": fold_parameters,
        "tail_blend_final_parameters": final_parameters,
        "calibration_probability_matrices": probabilities,
    }
    return base_artifact, metrics, losses


def _predict_followup_hour(
    artifact: dict[str, Any], rows: list[FeatureRow]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    base = _predict_hour(artifact, rows)
    medians = np.asarray(artifact["point_bundle"]["imputation_medians"])
    matrix = _impute(_matrix(rows), medians)
    linear_points = np.asarray(artifact["point_bundle"]["ridge"].predict(matrix))
    scales = np.asarray(artifact["followup"]["scale_model"].predict(matrix)) * 1.4826
    ordinal = _classifier_probabilities(artifact["followup"]["classifier"], matrix)
    weight, exponent = artifact["followup"]["tail_blend_final_parameters"]
    mixed = weight * base["convex_ensemble"][1] + (1.0 - weight) * base["linear_bias"][1]
    calibrated = _temperature_transform(mixed, exponent)
    return {
        "linear_bias": base["linear_bias"],
        "histogram_residual": base["histogram_residual"],
        "heteroskedastic_residual": (
            linear_points,
            _gaussian_probabilities(linear_points, scales),
        ),
        "ordinal_temperature_classifier": (_pmf_median(ordinal), ordinal),
        "tail_calibrated_ensemble": (_pmf_median(calibrated), calibrated),
    }


def _tail_calibration(probabilities: np.ndarray, rows: list[FeatureRow]) -> list[dict[str, Any]]:
    rounded = np.clip(
        np.floor(np.asarray([row.target_daily_max_f for row in rows]) + 0.5).astype(np.int64),
        SUPPORT_MIN_F,
        SUPPORT_MAX_F,
    )
    output = []
    for lower, upper, label in TAIL_PROBABILITY_BANDS:
        selected = (probabilities >= lower) & (probabilities < upper)
        count = int(selected.sum())
        observed = np.zeros_like(probabilities)
        observed[np.arange(len(rows)), rounded - SUPPORT_MIN_F] = 1.0
        mean_probability = float(probabilities[selected].mean()) if count else None
        observed_rate = float(observed[selected].mean()) if count else None
        output.append(
            {
                "probability_band": label,
                "degree_predictions": count,
                "event_days": int(np.any(selected, axis=1).sum()),
                "mean_predicted_probability": mean_probability,
                "observed_hit_rate": observed_rate,
                "calibration_ratio_observed_over_predicted": (
                    observed_rate / mean_probability
                    if mean_probability and observed_rate is not None
                    else None
                ),
            }
        )
    return output


def _select_champion(
    metrics_by_hour: dict[int, dict[str, Any]],
    losses_by_hour: dict[int, dict[str, np.ndarray]],
) -> tuple[str, dict[str, Any]]:
    baseline = np.concatenate([losses_by_hour[hour]["linear_bias"] for hour in (0, 12)])
    decisions = []
    eligible = []
    for candidate in MODEL_CANDIDATES:
        losses = np.concatenate([losses_by_hour[hour][candidate] for hour in (0, 12)])
        difference = baseline - losses
        improvement = float(np.mean(difference))
        standard_error = float(np.std(difference, ddof=1) / math.sqrt(len(difference)))
        rps = float(
            np.mean([metrics_by_hour[hour][candidate]["ranked_probability_score"] for hour in (0, 12)])
        )
        baseline_rps = float(
            np.mean([metrics_by_hour[hour]["linear_bias"]["ranked_probability_score"] for hour in (0, 12)])
        )
        passes = candidate == "linear_bias" or (
            improvement > standard_error and rps <= baseline_rps
        )
        decisions.append(
            {
                "candidate": candidate,
                "mean_log_loss_improvement_over_linear_bias": improvement,
                "paired_daily_standard_error": standard_error,
                "aggregate_ranked_probability_score": rps,
                "linear_bias_ranked_probability_score": baseline_rps,
                "passes_forecast_gate": passes,
            }
        )
        if passes:
            eligible.append(candidate)
    champion = min(
        eligible,
        key=lambda name: np.mean(
            [metrics_by_hour[hour][name]["rounded_temperature_log_loss"] for hour in (0, 12)]
        ),
    )
    return champion, {"candidate_decisions": decisions, "selected_champion": champion}


def _probability_lower(probability: float, alpha: float = 0.10) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    effective_days = 365.0
    value = beta.ppf(
        alpha,
        probability * effective_days + 0.5,
        (1.0 - probability) * effective_days + 0.5,
    )
    return min(probability, float(value))


def _recompute_candidate(candidate: dict[str, Any], probability: float, lower: float) -> None:
    candidate["probability"] = probability
    candidate["probability_lower"] = min(probability, lower)
    cost = candidate.get("all_in_cost_per_share")
    if cost is None:
        return
    candidate["model_edge_per_share"] = probability - float(cost)
    candidate["robust_edge_per_share"] = candidate["probability_lower"] - float(cost)
    candidate["expected_roi"] = candidate["model_edge_per_share"] / float(cost)
    candidate["robust_expected_roi"] = candidate["robust_edge_per_share"] / float(cost)


def _policy_candidates(
    candidates: list[dict[str, Any]], policy_name: str
) -> list[dict[str, Any]]:
    output = copy.deepcopy(candidates)
    if policy_name == "weather_only":
        return output
    if policy_name == "market_shrunk_50pct":
        for candidate in output:
            market = candidate.get("market_probability_proxy")
            if market is None:
                candidate["rejection_reasons"] = sorted(
                    set(candidate["rejection_reasons"]) | {"market_shrink_unavailable"}
                )
                continue
            probability = market_offset_probability(
                float(candidate["probability"]), float(market), 0.5
            )
            lower = market_offset_probability(
                float(candidate["probability_lower"]), float(market), 0.5
            )
            _recompute_candidate(candidate, probability, lower)
        return output
    if policy_name != "selection_adjusted":
        raise ValueError(f"unsupported policy: {policy_name}")
    family_sizes: dict[tuple[date, int], int] = defaultdict(int)
    for candidate in output:
        if candidate.get("executable"):
            family_sizes[(candidate["event_date"], candidate["decision_hour_local"])] += 1
    for candidate in output:
        family_size = family_sizes[(candidate["event_date"], candidate["decision_hour_local"])]
        alpha = SELECTION_FAMILYWISE_ALPHA / max(1, family_size)
        probability = float(candidate["probability"])
        _recompute_candidate(candidate, probability, _probability_lower(probability, alpha))
        candidate["selection_family_size"] = family_size
        candidate["selection_adjusted_alpha"] = alpha
    return output


def _market_probability_rows(
    settings: Settings,
    artifacts: dict[int, dict[str, Any]],
    *,
    start: date,
    end: date,
    tournament_id: str,
) -> dict[str, list[dict[str, Any]]]:
    from .challenger_tournament import _market_rows

    markets = _market_rows(settings.database_url, start, end)
    by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for market in markets:
        by_date[market["event_date"]].append(market)
    output = {candidate: [] for candidate in MODEL_CANDIDATES}
    for hour in (0, 12):
        rows = build_feature_rows(settings.database_url, start, end + timedelta(days=1), hour)
        predictions = _predict_followup_hour(artifacts[hour], rows)
        for candidate, (points, matrices) in predictions.items():
            for row, point, pmf in zip(rows, points, matrices, strict=True):
                day_markets = by_date.get(row.event_date, [])
                probabilities = [
                    _bucket_probability(pmf, market["bucket_lower_f"], market["bucket_upper_f"])
                    for market in day_markets
                ]
                if not day_markets:
                    continue
                if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
                    raise ValueError(f"market buckets do not partition {row.event_date}")
                if sum(int(market["resolved_yes"]) for market in day_markets) != 1:
                    raise ValueError(f"market event does not have one winner on {row.event_date}")
                for market, yes_probability in zip(day_markets, probabilities, strict=True):
                    output[candidate].append(
                        {
                            **market,
                            "model_run_id": f"{tournament_id}:{candidate}:h{hour:02d}",
                            "decision_time": row.decision_time,
                            "decision_hour_local": hour,
                            "point_prediction_f": float(point),
                            "probability_yes": yes_probability,
                            "probability_yes_lower": _probability_lower(yes_probability),
                            "probability_no": 1.0 - yes_probability,
                            "probability_no_lower": _probability_lower(1.0 - yes_probability),
                        }
                    )
    return output


def _economic_benchmark(
    settings: Settings,
    artifacts: dict[int, dict[str, Any]],
    *,
    tournament_id: str,
    start: date,
    end: date,
    sealed_start: date,
) -> dict[str, Any]:
    probability_rows = _market_probability_rows(
        settings, artifacts, start=start, end=end, tournament_id=tournament_id
    )
    periods = {
        "retrospective": (start, min(end, sealed_start - timedelta(days=1))),
        "sealed_post_freeze_backfill": (max(start, sealed_start), end),
    }
    candidate_rows = {
        model: _build_candidates(
            settings.database_url,
            probability_rows=rows,
            start=start,
            end=end,
            quantity=5.0,
            modeled_slippage_per_share=0.01,
        )
        for model, rows in probability_rows.items()
    }
    output: dict[str, Any] = {"models": {}, "periods": {}}
    for period_name, (period_start, period_end) in periods.items():
        available = period_start <= period_end
        output["periods"][period_name] = {
            "start": period_start,
            "end": period_end,
            "available": available,
            "forecast_metrics": {},
        }
        if not available:
            continue
        for hour in (0, 12):
            rows = build_feature_rows(
                settings.database_url, period_start, period_end + timedelta(days=1), hour
            )
            predictions = _predict_followup_hour(artifacts[hour], rows)
            output["periods"][period_name]["forecast_metrics"][str(hour)] = {
                model: {
                    **_distribution_metrics(probabilities, points, rows)[0],
                    "tail_calibration": _tail_calibration(probabilities, rows),
                }
                for model, (points, probabilities) in predictions.items()
            }
        dates = _date_range(period_start, period_end)
        for model, all_candidates in candidate_rows.items():
            period_candidates = [
                row for row in all_candidates if period_start <= row["event_date"] <= period_end
            ]
            for policy_name in POLICY_CANDIDATES:
                transformed = _policy_candidates(period_candidates, policy_name)
                metrics, selected, _ = _policy_metrics(transformed, dates, FIXED_POLICY)
                price_bands = []
                for lower, upper, label in (
                    (0.00, 0.04, "0-4c"),
                    (0.04, 0.08, "4-8c"),
                    (0.08, 0.12, "8-12c"),
                    (0.12, 0.16, "12-16c"),
                    (0.16, 0.25, "16-25c"),
                ):
                    band_trades = [
                        trade
                        for trade in selected
                        if lower <= float(trade["ask_vwap"]) < upper
                    ]
                    band_pnl = [
                        float(trade["realized_net_per_share"]) * float(trade["quantity"])
                        for trade in band_trades
                    ]
                    price_bands.append(
                        {
                            "price_band": label,
                            "trades": len(band_trades),
                            "wins": sum(value > 0 for value in band_pnl),
                            "total_net": float(sum(band_pnl)),
                        }
                    )
                slippage_stress = {
                    f"{slippage:.3f}": _selected_trade_stress(selected, slippage)
                    for slippage in (0.0, 0.01)
                }
                positive_profit = sum(
                    max(0.0, float(band["total_net"])) for band in price_bands
                )
                maximum_band_profit_share = (
                    max(
                        max(0.0, float(band["total_net"])) for band in price_bands
                    )
                    / positive_profit
                    if positive_profit > 0
                    else None
                )
                checks = {
                    "positive_total_net": metrics["total_net"] > 0,
                    "positive_all_chronological_folds": all(
                        fold["total_net"] > 0 for fold in metrics["chronological_folds"]
                    ),
                    "positive_lower_90pct_block_bootstrap_mean_daily_net": (
                        metrics["lower_90pct_block_bootstrap_mean_daily_net"] > 0
                    ),
                    "positive_without_best_three_trades": (
                        metrics["net_without_best_three_trades"] > 0
                    ),
                    "minimum_30_distinct_event_day_trades": metrics["trades"] >= 30,
                    "selected_hit_rate_exceeds_mean_break_even_probability": (
                        metrics["hit_rate"] > metrics["mean_break_even_probability"]
                    ),
                    "no_price_band_exceeds_70pct_of_positive_profit": (
                        maximum_band_profit_share is not None
                        and maximum_band_profit_share <= 0.70
                    ),
                    "positive_at_zero_slippage": slippage_stress["0.000"]["total_net"] > 0,
                    "positive_at_one_cent_slippage": (
                        slippage_stress["0.010"]["total_net"] > 0
                    ),
                }
                entry = {
                    "metrics": _compact_policy_metrics(metrics),
                    "candidate_coverage": _candidate_coverage(transformed),
                    "selected_price_bands": price_bands,
                    "maximum_price_band_positive_profit_share": maximum_band_profit_share,
                    "slippage_stress": slippage_stress,
                    "qualification_checks": checks,
                    "qualification_supported": all(checks.values()),
                }
                output["models"].setdefault(model, {}).setdefault(policy_name, {})[
                    period_name
                ] = entry
    for model in MODEL_CANDIDATES:
        for policy_name in POLICY_CANDIDATES:
            periods_for_policy = output["models"][model][policy_name]
            retrospective_supported = periods_for_policy["retrospective"][
                "qualification_supported"
            ]
            sealed_trades = periods_for_policy["sealed_post_freeze_backfill"]["metrics"][
                "trades"
            ]
            periods_for_policy["overall_qualification"] = {
                "retrospective_checks_supported": retrospective_supported,
                "nonzero_sealed_executable_trade_evidence": sealed_trades > 0,
                "production_qualified": retrospective_supported and sealed_trades > 0,
            }
    output["execution_contract"] = asdict(FIXED_POLICY) | {
        "quantity": 5.0,
        "modeled_slippage_per_share": 0.01,
        "policy_candidates": POLICY_CANDIDATES,
        "selection_adjustment": (
            "beta lower bound with event-day/hour Bonferroni familywise alpha"
        ),
        "economic_block_bootstrap_resamples_complete_event_days": True,
    }
    output["production_qualified"] = False
    return output


def run_tail_calibration_tournament(
    settings: Settings,
    *,
    training_start: date,
    training_end: date,
    calibration_start: date,
    calibration_end: date,
    economic_start: date,
    economic_end: date,
    sealed_start: date,
    git_revision: str,
    runner_image_id: str,
) -> dict[str, Any]:
    if not training_start <= training_end < calibration_start <= calibration_end:
        raise ValueError("training and calibration ranges must be chronological and disjoint")
    if economic_start <= calibration_end or not economic_start <= sealed_start <= economic_end:
        raise ValueError("economic and sealed ranges must follow calibration")
    revision = _validate_revision(git_revision)
    image_id = _validate_image_id(runner_image_id)
    training_rows = {
        hour: build_feature_rows(
            settings.database_url, training_start, training_end + timedelta(days=1), hour
        )
        for hour in (0, 12)
    }
    calibration_rows = {
        hour: build_feature_rows(
            settings.database_url, calibration_start, calibration_end + timedelta(days=1), hour
        )
        for hour in (0, 12)
    }
    artifacts: dict[int, dict[str, Any]] = {}
    metrics: dict[int, dict[str, Any]] = {}
    losses: dict[int, dict[str, np.ndarray]] = {}
    for hour in (0, 12):
        artifacts[hour], metrics[hour], losses[hour] = _train_followup_hour(
            training_rows[hour], calibration_rows[hour], hour
        )
        for candidate in MODEL_CANDIDATES:
            metrics[hour][candidate]["tail_calibration"] = _tail_calibration(
                artifacts[hour]["followup"]["calibration_probability_matrices"][candidate],
                calibration_rows[hour],
            )
    champion, selection = _select_champion(metrics, losses)
    tournament_id = str(uuid.uuid4())
    source_digest = _row_digest(
        {hour: training_rows[hour] + calibration_rows[hour] for hour in (0, 12)}
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "process_id": PROCESS_ID,
        "tournament_id": tournament_id,
        "git_revision": revision,
        "source_feature_rows_sha256": source_digest,
        "candidate_names": MODEL_CANDIDATES,
        "selected_champion": champion,
        "hours": artifacts,
        "random_seed": RANDOM_SEED,
    }
    settings.model_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = settings.model_directory / f"tail-calibration-tournament-{tournament_id}.joblib"
    partial_artifact = artifact_path.with_suffix(".partial")
    joblib.dump(artifact, partial_artifact, compress=3)
    partial_artifact.replace(artifact_path)
    artifact_sha256, artifact_bytes = file_sha256(artifact_path)
    economic = _economic_benchmark(
        settings,
        artifacts,
        tournament_id=tournament_id,
        start=economic_start,
        end=economic_end,
        sealed_start=sealed_start,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "process_id": PROCESS_ID,
        "tournament_id": tournament_id,
        "objective": "tail_calibration_and_selection_adjusted_positive_net_expectancy",
        "provenance": {
            "git_revision": revision,
            "runner_declared_existing_image_id": image_id,
            "runner": "existing dependency image with read-only source mount",
            "container_image_rebuilt": False,
            "generated_at": datetime.now(UTC),
        },
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
            "candidate_names": MODEL_CANDIDATES,
            "policy_candidates": POLICY_CANDIDATES,
            "model_selection_uses_market_prices": False,
            "model_selection_uses_pnl": False,
            "tail_blend_is_month_cross_fitted_for_calibration_scoring": True,
            "economic_policies_are_fixed_comparators": True,
        },
        "forecast_metrics": {str(hour): metrics[hour] for hour in (0, 12)},
        "selection": selection,
        "selected_champion": champion,
        "artifact_uri": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": artifact_bytes,
        "economic_benchmark": economic,
        "production_qualified": False,
    }
    settings.report_directory.mkdir(parents=True, exist_ok=True)
    report_path = settings.report_directory / f"tail-calibration-tournament-{tournament_id}.json"
    report["report_uri"] = str(report_path)
    safe = _json_safe(report)
    partial_report = report_path.with_suffix(".partial")
    partial_report.write_text(
        json.dumps(safe, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    partial_report.replace(report_path)
    return safe
