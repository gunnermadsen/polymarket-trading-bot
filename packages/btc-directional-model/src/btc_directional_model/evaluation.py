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


def sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
    return (centre - margin) / denominator, (centre + margin) / denominator


def prediction_rows(
    frame: pl.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    *,
    require_executable: bool,
) -> pl.DataFrame:
    scored = frame.select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "binance_sign_up",
        "market_favorite_up",
        "up_executable",
        "down_executable",
        "up_ask_vwap_5",
        "down_ask_vwap_5",
    ).with_columns(pl.Series("probability_up", probabilities))
    scored = scored.with_columns(
        (pl.col("probability_up") >= 0.5).cast(pl.Int8).alias("predicted_up"),
        pl.max_horizontal("probability_up", 1 - pl.col("probability_up")).alias("confidence"),
    ).with_columns(
        (pl.col("predicted_up") == pl.col("label_up")).alias("correct"),
        pl.when(pl.col("predicted_up") == 1)
        .then(pl.col("up_executable"))
        .otherwise(pl.col("down_executable"))
        .alias("selected_side_executable"),
        pl.when(pl.col("predicted_up") == 1)
        .then(pl.col("up_ask_vwap_5"))
        .otherwise(pl.col("down_ask_vwap_5"))
        .alias("selected_ask_vwap_5"),
    )
    eligible = scored.filter(pl.col("confidence") >= threshold)
    if require_executable:
        eligible = eligible.filter(pl.col("selected_side_executable"))
    return (
        eligible.sort(["observed_at", "market_id"])
        .group_by("market_id", maintain_order=True)
        .first()
    )


def classification_metrics(rows: pl.DataFrame) -> dict[str, Any]:
    if rows.is_empty():
        return empty_metrics()
    y_true = rows["label_up"].to_numpy()
    y_pred = rows["predicted_up"].to_numpy()
    probability = rows["probability_up"].to_numpy()
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    correct = int((y_true == y_pred).sum())
    total = len(y_true)
    lower, upper = wilson_interval(correct, total)
    try:
        auc = float(roc_auc_score(y_true, probability))
    except ValueError:
        auc = None
    return {
        "markets": total,
        "correct": correct,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "wilson_lower_95": lower,
        "wilson_upper_95": upper,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "up_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "up_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "down_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "down_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "roc_auc": auc,
        "confusion_matrix": matrix.tolist(),
        "predicted_up": int(y_pred.sum()),
        "predicted_down": int(total - y_pred.sum()),
        "actual_up": int(y_true.sum()),
        "actual_down": int(total - y_true.sum()),
        "maximum_consecutive_losses": maximum_consecutive_losses(rows),
    }


def baseline_metrics(rows: pl.DataFrame, column: str) -> dict[str, Any]:
    available = rows.filter(pl.col(column).is_not_null())
    if available.is_empty():
        return empty_metrics()
    baseline = available.with_columns(
        pl.col(column).cast(pl.Int8).alias("predicted_up"),
        pl.col(column).cast(pl.Float64).alias("probability_up"),
    )
    return classification_metrics(baseline)


def threshold_table(
    frame: pl.DataFrame,
    probabilities: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    table = []
    for threshold in thresholds:
        rows = prediction_rows(frame, probabilities, threshold, require_executable=False)
        metrics = classification_metrics(rows)
        table.append({"threshold": threshold, **metrics})
    return table


def confidence_buckets(rows: pl.DataFrame) -> list[dict[str, Any]]:
    if rows.is_empty():
        return []
    return (
        rows.with_columns(
            (pl.col("confidence") * 20).floor().truediv(20).clip(0.5, 0.95).alias("bucket")
        )
        .group_by("bucket")
        .agg(
            pl.len().alias("markets"),
            pl.col("correct").sum().alias("correct"),
            pl.col("probability_up").mean().alias("mean_probability_up"),
            pl.col("label_up").mean().alias("observed_up_rate"),
        )
        .with_columns((pl.col("correct") / pl.col("markets")).alias("accuracy"))
        .sort("bucket")
        .to_dicts()
    )


def time_buckets(rows: pl.DataFrame) -> list[dict[str, Any]]:
    if rows.is_empty():
        return []
    return (
        rows.group_by("seconds_elapsed")
        .agg(pl.len().alias("markets"), pl.col("correct").sum().alias("correct"))
        .with_columns((pl.col("correct") / pl.col("markets")).alias("accuracy"))
        .sort("seconds_elapsed")
        .to_dicts()
    )


def daily_accuracy(rows: pl.DataFrame) -> list[dict[str, Any]]:
    if rows.is_empty():
        return []
    return (
        rows.with_columns(pl.col("window_start").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.len().alias("markets"), pl.col("correct").sum().alias("correct"))
        .with_columns((pl.col("correct") / pl.col("markets")).alias("accuracy"))
        .sort("date")
        .with_columns(pl.col("date").cast(pl.String))
        .to_dicts()
    )


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


def empty_metrics() -> dict[str, Any]:
    return {
        "markets": 0,
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
        "confusion_matrix": [[0, 0], [0, 0]],
        "predicted_up": 0,
        "predicted_down": 0,
        "actual_up": 0,
        "actual_down": 0,
        "maximum_consecutive_losses": 0,
    }
