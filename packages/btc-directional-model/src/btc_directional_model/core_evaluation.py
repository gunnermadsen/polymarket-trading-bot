from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .core_config import CoreGateConfig, CoreModelConfig

FIXED_TIME_SCORE_SECONDS = (60, 90, 120, 180, 240)
FIRST_CROSSING_TIME_BANDS = (
    ("60-89", 60, 90),
    ("90-119", 90, 120),
    ("120-179", 120, 180),
    ("180-240", 180, 241),
)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def wilson_interval(
    correct: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        (proportion * (1 - proportion) + z * z / (4 * total)) / total
    )
    return (centre - margin) / denominator, (centre + margin) / denominator


def first_prediction_rows(
    frame: pl.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pl.DataFrame:
    scored = scored_prediction_rows(frame, probabilities)
    return (
        scored.filter(pl.col("confidence") >= threshold)
        .sort(["observed_at", "market_id"])
        .group_by("market_id", maintain_order=True)
        .first()
    )


def fixed_time_prediction_rows(
    frame: pl.DataFrame,
    probabilities: np.ndarray,
) -> pl.DataFrame:
    return (
        scored_prediction_rows(frame, probabilities)
        .filter(pl.col("seconds_elapsed").is_in(FIXED_TIME_SCORE_SECONDS))
        .sort(["observed_at", "market_id"])
    )


def scored_prediction_rows(
    frame: pl.DataFrame,
    probabilities: np.ndarray,
) -> pl.DataFrame:
    if frame.height != len(probabilities):
        raise ValueError("probability count does not match feature rows")
    scored = frame.select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "binance_sign_up",
    ).with_columns(pl.Series("probability_up", probabilities))
    scored = scored.with_columns(
        (pl.col("probability_up") >= 0.5).cast(pl.Int8).alias("predicted_up"),
        pl.max_horizontal("probability_up", 1 - pl.col("probability_up")).alias(
            "confidence"
        ),
    ).with_columns(
        (pl.col("predicted_up") == pl.col("label_up")).alias("correct"),
        (pl.col("binance_sign_up") == pl.col("label_up")).alias("baseline_correct"),
    )
    return scored


def classification_metrics(
    rows: pl.DataFrame,
    *,
    eligible_markets: int | None = None,
) -> dict[str, Any]:
    if rows.is_empty():
        return empty_metrics(eligible_markets or 0)
    y_true = rows["label_up"].to_numpy()
    y_pred = rows["predicted_up"].to_numpy()
    probabilities = rows["probability_up"].to_numpy()
    total = len(y_true)
    correct = int((y_true == y_pred).sum())
    lower, upper = wilson_interval(correct, total)
    try:
        auc = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        auc = None
    total_eligible = eligible_markets if eligible_markets is not None else total
    return {
        "markets": total,
        "eligible_markets": total_eligible,
        "coverage": total / total_eligible if total_eligible else 0.0,
        "correct": correct,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "wilson_lower_95": lower,
        "wilson_upper_95": upper,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "up_precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "up_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "down_precision": float(
            precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        ),
        "down_recall": float(
            recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "roc_auc": auc,
        "expected_calibration_error": expected_calibration_error(
            y_true, probabilities, bins=10
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "predicted_up": int(y_pred.sum()),
        "predicted_down": int(total - y_pred.sum()),
        "actual_up": int(y_true.sum()),
        "actual_down": int(total - y_true.sum()),
        "maximum_consecutive_losses": maximum_consecutive_losses(rows),
    }


def baseline_metrics(
    rows: pl.DataFrame,
    *,
    eligible_markets: int | None = None,
) -> dict[str, Any]:
    if rows.is_empty():
        return empty_metrics(eligible_markets or 0)
    baseline = rows.with_columns(
        pl.col("binance_sign_up").cast(pl.Int8).alias("predicted_up"),
        pl.col("binance_sign_up").cast(pl.Float64).alias("probability_up"),
        pl.col("baseline_correct").alias("correct"),
    )
    return classification_metrics(baseline, eligible_markets=eligible_markets)


def paired_uplift(rows: pl.DataFrame) -> dict[str, Any]:
    if rows.is_empty():
        return {
            "markets": 0,
            "model_accuracy": 0.0,
            "baseline_accuracy": 0.0,
            "accuracy_uplift": 0.0,
            "model_only_correct": 0,
            "baseline_only_correct": 0,
        }
    model_correct = rows["correct"].cast(pl.Int8)
    baseline_correct = rows["baseline_correct"].cast(pl.Int8)
    return {
        "markets": rows.height,
        "model_accuracy": float(model_correct.mean()),
        "baseline_accuracy": float(baseline_correct.mean()),
        "accuracy_uplift": float((model_correct - baseline_correct).mean()),
        "model_only_correct": rows.filter(
            pl.col("correct") & ~pl.col("baseline_correct")
        ).height,
        "baseline_only_correct": rows.filter(
            ~pl.col("correct") & pl.col("baseline_correct")
        ).height,
    }


def threshold_table(
    frame: pl.DataFrame,
    probabilities: np.ndarray,
    model: CoreModelConfig,
) -> list[dict[str, Any]]:
    eligible = frame["market_id"].n_unique()
    thresholds = np.round(
        np.arange(
            model.confidence_min,
            model.confidence_max + model.confidence_step / 2,
            model.confidence_step,
        ),
        6,
    )
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = first_prediction_rows(frame, probabilities, float(threshold))
        metrics = classification_metrics(selected, eligible_markets=eligible)
        rows.append(
            {
                "threshold": float(threshold),
                **metrics,
                **paired_uplift(selected),
            }
        )
    return rows


def choose_threshold(
    table: list[dict[str, Any]],
    gates: CoreGateConfig,
    *,
    minimum_markets: int,
) -> tuple[float, bool]:
    qualifying = [
        row
        for row in table
        if row["markets"] >= minimum_markets
        and row["coverage"] >= gates.minimum_coverage
        and row["accuracy"] >= gates.target_accuracy
        and row["wilson_lower_95"] >= gates.target_wilson_lower
        and row["balanced_accuracy"] >= gates.target_balanced_accuracy
        and row["up_recall"] >= gates.minimum_direction_recall
        and row["down_recall"] >= gates.minimum_direction_recall
        and row["expected_calibration_error"] <= gates.maximum_ece
        and row["accuracy_uplift"] >= gates.minimum_same_time_path_uplift
    ]
    candidates = qualifying or [
        row
        for row in table
        if row["markets"] >= minimum_markets
        and row["coverage"] >= gates.minimum_coverage
    ]
    if not candidates:
        candidates = table
    selected = max(
        candidates,
        key=lambda row: (
            row["coverage"],
            row["wilson_lower_95"],
            row["balanced_accuracy"],
            row["accuracy"],
            row["accuracy_uplift"],
            -row["threshold"],
        ),
    )
    return float(selected["threshold"]), bool(qualifying)


def block_bootstrap_uplift(
    rows: pl.DataFrame,
    *,
    resamples: int,
    random_seed: int,
    block: str,
) -> dict[str, Any]:
    if rows.is_empty():
        return {
            "block": block,
            "resamples": resamples,
            "observed": 0.0,
            "lower_95": 0.0,
            "upper_95": 0.0,
        }
    if block == "hour":
        key_expression = pl.col("window_start").dt.truncate("1h").cast(pl.String)
    elif block == "day":
        key_expression = pl.col("window_start").dt.date().cast(pl.String)
    else:
        raise ValueError("bootstrap block must be hour or day")
    block_rows = (
        rows.with_columns(key_expression.alias("block"))
        .with_columns(
            (
                pl.col("correct").cast(pl.Int8)
                - pl.col("baseline_correct").cast(pl.Int8)
            ).alias("delta")
        )
        .group_by("block")
        .agg(
            pl.col("delta").sum().alias("delta_sum"),
            pl.len().alias("markets"),
        )
        .sort("block")
    )
    sums = block_rows["delta_sum"].to_numpy().astype(np.float64)
    counts = block_rows["markets"].to_numpy().astype(np.float64)
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.integers(
        0,
        len(sums),
        size=(resamples, len(sums)),
    )
    sampled_sum = sums[sample_indices].sum(axis=1)
    sampled_count = counts[sample_indices].sum(axis=1)
    distribution = sampled_sum / sampled_count
    observed = float(sums.sum() / counts.sum())
    return {
        "block": block,
        "blocks": len(sums),
        "resamples": resamples,
        "observed": observed,
        "lower_95": float(np.quantile(distribution, 0.025)),
        "upper_95": float(np.quantile(distribution, 0.975)),
        "mean": float(distribution.mean()),
    }


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, *, bins: int
) -> float:
    if len(y_true) == 0:
        return 0.0
    order = np.argsort(probability)
    chunks = np.array_split(order, min(bins, len(order)))
    total = len(order)
    error = 0.0
    for indices in chunks:
        if len(indices) == 0:
            continue
        error += (
            len(indices)
            / total
            * abs(float(probability[indices].mean()) - float(y_true[indices].mean()))
        )
    return float(error)


def reliability_rows(rows: pl.DataFrame, *, bins: int = 10) -> list[dict[str, Any]]:
    if rows.is_empty():
        return []
    probabilities = rows["probability_up"].to_numpy()
    order = np.argsort(probabilities)
    output: list[dict[str, Any]] = []
    for index, indices in enumerate(np.array_split(order, min(bins, len(order)))):
        if len(indices) == 0:
            continue
        subset = rows[indices.tolist()]
        output.append(
            {
                "bin": index + 1,
                "markets": subset.height,
                "mean_probability_up": float(subset["probability_up"].mean()),
                "observed_up_rate": float(subset["label_up"].mean()),
            }
        )
    return output


def daily_accuracy(rows: pl.DataFrame) -> list[dict[str, Any]]:
    if rows.is_empty():
        return []
    return (
        rows.with_columns(pl.col("window_start").dt.date().cast(pl.String).alias("date"))
        .group_by("date")
        .agg(
            pl.len().alias("markets"),
            pl.col("correct").sum().alias("correct"),
            pl.col("baseline_correct").sum().alias("baseline_correct"),
        )
        .with_columns(
            (pl.col("correct") / pl.col("markets")).alias("accuracy"),
            (pl.col("baseline_correct") / pl.col("markets")).alias(
                "baseline_accuracy"
            ),
        )
        .with_columns(
            (pl.col("accuracy") - pl.col("baseline_accuracy")).alias("uplift")
        )
        .sort("date")
        .to_dicts()
    )


def time_accuracy(rows: pl.DataFrame) -> list[dict[str, Any]]:
    if rows.is_empty():
        return []
    return (
        rows.group_by("seconds_elapsed")
        .agg(
            pl.len().alias("markets"),
            pl.col("correct").sum().alias("correct"),
            pl.col("baseline_correct").sum().alias("baseline_correct"),
        )
        .with_columns(
            (pl.col("correct") / pl.col("markets")).alias("accuracy"),
            (pl.col("baseline_correct") / pl.col("markets")).alias(
                "baseline_accuracy"
            ),
        )
        .sort("seconds_elapsed")
        .to_dicts()
    )


def first_crossing_timing(
    rows: pl.DataFrame,
    *,
    eligible_markets: int | None = None,
) -> dict[str, Any]:
    total_eligible = (
        eligible_markets
        if eligible_markets is not None
        else rows["market_id"].n_unique()
        if not rows.is_empty()
        else 0
    )
    if rows.is_empty():
        return {
            "markets": 0,
            "eligible_markets": total_eligible,
            "coverage": 0.0,
            "early_entry_markets": 0,
            "early_entry_coverage": 0.0,
            "median_first_crossing_seconds": None,
            "p90_first_crossing_seconds": None,
            "time_bands": [
                empty_time_band(label, start, end, total_eligible)
                for label, start, end in FIRST_CROSSING_TIME_BANDS
            ],
        }

    elapsed = rows["seconds_elapsed"].to_numpy().astype(np.float64)
    bands = [
        time_band_metrics(rows, label, start, end, total_eligible)
        for label, start, end in FIRST_CROSSING_TIME_BANDS
    ]
    early_entry_markets = sum(
        int(band["markets"]) for band in bands if band["end_seconds_exclusive"] <= 120
    )
    return {
        "markets": rows.height,
        "eligible_markets": total_eligible,
        "coverage": rows.height / total_eligible if total_eligible else 0.0,
        "early_entry_markets": early_entry_markets,
        "early_entry_coverage": (
            early_entry_markets / total_eligible if total_eligible else 0.0
        ),
        "median_first_crossing_seconds": float(np.median(elapsed)),
        "p90_first_crossing_seconds": float(np.quantile(elapsed, 0.90)),
        "time_bands": bands,
    }


def time_band_metrics(
    rows: pl.DataFrame,
    label: str,
    start_seconds: int,
    end_seconds_exclusive: int,
    eligible_markets: int,
) -> dict[str, Any]:
    selected = rows.filter(
        (pl.col("seconds_elapsed") >= start_seconds)
        & (pl.col("seconds_elapsed") < end_seconds_exclusive)
    )
    if selected.is_empty():
        return empty_time_band(
            label,
            start_seconds,
            end_seconds_exclusive,
            eligible_markets,
        )
    model_accuracy = float(selected["correct"].mean())
    baseline_accuracy = float(selected["baseline_correct"].mean())
    return {
        "band": label,
        "start_seconds": start_seconds,
        "end_seconds_exclusive": end_seconds_exclusive,
        "markets": selected.height,
        "coverage": selected.height / eligible_markets if eligible_markets else 0.0,
        "prediction_share": selected.height / rows.height,
        "accuracy": model_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_uplift": model_accuracy - baseline_accuracy,
    }


def empty_time_band(
    label: str,
    start_seconds: int,
    end_seconds_exclusive: int,
    eligible_markets: int,
) -> dict[str, Any]:
    return {
        "band": label,
        "start_seconds": start_seconds,
        "end_seconds_exclusive": end_seconds_exclusive,
        "markets": 0,
        "coverage": 0.0,
        "prediction_share": 0.0,
        "accuracy": None,
        "baseline_accuracy": None,
        "accuracy_uplift": None,
    }


def maximum_consecutive_losses(rows: pl.DataFrame) -> int:
    longest = 0
    current = 0
    for correct in rows.sort("observed_at")["correct"].to_list():
        if correct:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def empty_metrics(eligible_markets: int) -> dict[str, Any]:
    return {
        "markets": 0,
        "eligible_markets": eligible_markets,
        "coverage": 0.0,
        "correct": 0,
        "accuracy": 0.0,
        "wilson_lower_95": 0.0,
        "wilson_upper_95": 0.0,
        "balanced_accuracy": 0.0,
        "up_precision": 0.0,
        "up_recall": 0.0,
        "down_precision": 0.0,
        "down_recall": 0.0,
        "f1": 0.0,
        "matthews_correlation": 0.0,
        "brier_score": None,
        "log_loss": None,
        "roc_auc": None,
        "expected_calibration_error": None,
        "confusion_matrix": [[0, 0], [0, 0]],
        "predicted_up": 0,
        "predicted_down": 0,
        "actual_up": 0,
        "actual_down": 0,
        "maximum_consecutive_losses": 0,
    }
