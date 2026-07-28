from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from .config import BenchmarkConfig
from .models import CLASS_LABELS


@dataclass(frozen=True)
class Policy:
    probability_threshold: float
    directional_margin: float
    no_trade: bool = False


def predicted_classes(probabilities: np.ndarray) -> np.ndarray:
    return CLASS_LABELS[np.argmax(probabilities, axis=1)]


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.column_stack([y_true == label for label in CLASS_LABELS]).astype(float)
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def macro_ece(y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 10) -> float:
    class_errors: list[float] = []
    for index, label in enumerate(CLASS_LABELS):
        order = np.argsort(probabilities[:, index])
        chunks = np.array_split(order, bins)
        weighted = 0.0
        for chunk in chunks:
            if chunk.size == 0:
                continue
            confidence = float(np.mean(probabilities[chunk, index]))
            frequency = float(np.mean(y_true[chunk] == label))
            weighted += chunk.size * abs(confidence - frequency)
        class_errors.append(weighted / len(y_true))
    return float(np.mean(class_errors))


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = predicted_classes(probabilities)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predicted,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "log_loss": float(log_loss(y_true, probabilities, labels=CLASS_LABELS)),
        "brier": multiclass_brier(y_true, probabilities),
        "macro_ece": macro_ece(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=CLASS_LABELS).tolist(),
        "per_class": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(CLASS_LABELS)
        },
    }


def baseline_metrics(
    *,
    fit_labels: np.ndarray,
    evaluation_frame: pl.DataFrame,
) -> dict[str, Any]:
    y_true = evaluation_frame["label"].to_numpy()
    counts = np.array([(fit_labels == label).sum() for label in CLASS_LABELS])
    priors = counts / counts.sum()
    majority = int(CLASS_LABELS[np.argmax(counts)])
    flat = np.zeros_like(y_true)
    majority_predictions = np.full_like(y_true, majority)
    momentum = evaluation_frame["momentum_label"].to_numpy()
    contrarian = -momentum
    rules = {
        "always_flat": flat,
        "training_majority": majority_predictions,
        "momentum": momentum,
        "contrarian": contrarian,
    }
    results: dict[str, Any] = {}
    for name, predictions in rules.items():
        results[name] = {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        }
    prior_probabilities = np.tile(priors, (len(y_true), 1))
    results["class_prior_probability"] = {
        "log_loss": float(log_loss(y_true, prior_probabilities, labels=CLASS_LABELS)),
        "brier": multiclass_brier(y_true, prior_probabilities),
        "priors": priors.tolist(),
    }
    results["best_balanced_accuracy"] = max(results[name]["balanced_accuracy"] for name in rules)
    results["best_macro_f1"] = max(results[name]["macro_f1"] for name in rules)
    return results


def policy_actions(probabilities: np.ndarray, policy: Policy) -> np.ndarray:
    if policy.no_trade:
        return np.zeros(probabilities.shape[0], dtype=np.int8)
    short = probabilities[:, 0]
    flat = probabilities[:, 1]
    long = probabilities[:, 2]
    choose_long = long >= short
    direction_probability = np.where(choose_long, long, short)
    opposite_probability = np.where(choose_long, short, long)
    eligible = (
        (direction_probability > flat)
        & (direction_probability >= policy.probability_threshold)
        & (direction_probability - opposite_probability >= policy.directional_margin)
    )
    actions = np.zeros(probabilities.shape[0], dtype=np.int8)
    actions[eligible & choose_long] = 1
    actions[eligible & ~choose_long] = -1
    return actions


def trade_ledger(
    frame: pl.DataFrame,
    actions: np.ndarray,
    *,
    execution_cost_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    if frame.height != len(actions):
        raise ValueError("action count does not match frame")
    columns = frame.select(
        [
            "bucket_start",
            "entry_at",
            "label_exit_at",
            "gross_forward_bps",
            "long_market_execution_cost_bps",
            "short_market_execution_cost_bps",
            "fee_cost_bps",
            "funding_horizon_bps",
        ]
    )
    ledger: list[dict[str, Any]] = []
    busy_until = None
    for row, action in zip(columns.iter_rows(named=True), actions, strict=True):
        if action == 0:
            continue
        if busy_until is not None and row["entry_at"] < busy_until:
            continue
        gross = float(action) * row["gross_forward_bps"]
        directional_cost = (
            row["long_market_execution_cost_bps"]
            if action == 1
            else row["short_market_execution_cost_bps"]
        )
        market_cost = directional_cost * execution_cost_multiplier
        funding = -float(action) * row["funding_horizon_bps"]
        net = gross - row["fee_cost_bps"] - market_cost + funding
        ledger.append(
            {
                "decision_time": row["bucket_start"],
                "entry_at": row["entry_at"],
                "exit_at": row["label_exit_at"],
                "direction": int(action),
                "gross_bps": float(gross),
                "fee_bps": float(row["fee_cost_bps"]),
                "market_execution_bps": float(market_cost),
                "funding_bps": float(funding),
                "net_bps": float(net),
            }
        )
        busy_until = row["label_exit_at"]
    return ledger


def _daily_arrays(
    ledger: list[dict[str, Any]], start: date, end: date
) -> tuple[np.ndarray, np.ndarray]:
    days = (end - start).days + 1
    pnl = np.zeros(days, dtype=float)
    trades = np.zeros(days, dtype=int)
    for trade in ledger:
        index = (trade["decision_time"].date() - start).days
        if 0 <= index < days:
            pnl[index] += trade["net_bps"]
            trades[index] += 1
    return pnl, trades


def circular_block_expectancy_ci(
    ledger: list[dict[str, Any]],
    *,
    start: date,
    end: date,
    repetitions: int,
    seed: int,
    confidence: float = 0.95,
    block_days: int = 7,
) -> tuple[float | None, float | None]:
    if not ledger:
        return None, None
    pnl, trades = _daily_arrays(ledger, start, end)
    rng = np.random.default_rng(seed)
    sample_count = len(pnl)
    block_count = math.ceil(sample_count / block_days)
    estimates = np.empty(repetitions, dtype=float)
    offsets = np.arange(block_days)
    for repetition in range(repetitions):
        starts = rng.integers(0, sample_count, size=block_count)
        indices = ((starts[:, None] + offsets) % sample_count).ravel()[:sample_count]
        sampled_trades = trades[indices].sum()
        estimates[repetition] = pnl[indices].sum() / sampled_trades if sampled_trades else np.nan
    estimates = estimates[np.isfinite(estimates)]
    if not estimates.size:
        return None, None
    alpha = 1.0 - confidence
    return (
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )


def economic_metrics(
    frame: pl.DataFrame,
    actions: np.ndarray,
    *,
    execution_cost_multiplier: float,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    ledger = trade_ledger(
        frame,
        actions,
        execution_cost_multiplier=execution_cost_multiplier,
    )
    if not ledger:
        return {
            "trades": 0,
            "action_coverage": 0.0,
            "net_expectancy_bps": None,
            "bootstrap_95_lower_bps": None,
            "bootstrap_95_upper_bps": None,
            "profit_factor": None,
            "win_rate": None,
            "total_net_bps": 0.0,
            "max_drawdown_bps": 0.0,
            "positive_month_fraction": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }
    pnl = np.asarray([row["net_bps"] for row in ledger])
    cumulative = np.cumsum(pnl)
    drawdown = np.maximum.accumulate(np.insert(cumulative, 0, 0.0))[1:] - cumulative
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    first_timestamp = frame["bucket_start"].min()
    last_timestamp = frame["bucket_start"].max()
    monthly: dict[str, float] = {}
    year = first_timestamp.year
    month = first_timestamp.month
    while (year, month) <= (last_timestamp.year, last_timestamp.month):
        monthly[f"{year:04d}-{month:02d}"] = 0.0
        month += 1
        if month == 13:
            year += 1
            month = 1
    for row in ledger:
        month = row["decision_time"].strftime("%Y-%m")
        monthly[month] = monthly.get(month, 0.0) + row["net_bps"]
    start = first_timestamp.date()
    end = last_timestamp.date()
    lower, upper = circular_block_expectancy_ci(
        ledger,
        start=start,
        end=end,
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    return {
        "trades": len(ledger),
        "action_coverage": len(ledger) / frame.height,
        "net_expectancy_bps": float(np.mean(pnl)),
        "median_net_bps": float(np.median(pnl)),
        "bootstrap_95_lower_bps": lower,
        "bootstrap_95_upper_bps": upper,
        "profit_factor": float(gains / losses) if losses > 0 else None,
        "win_rate": float(np.mean(pnl > 0)),
        "total_net_bps": float(pnl.sum()),
        "max_drawdown_bps": float(drawdown.max(initial=0.0)),
        "positive_month_fraction": float(np.mean(np.asarray(list(monthly.values())) > 0)),
        "long_trades": sum(row["direction"] == 1 for row in ledger),
        "short_trades": sum(row["direction"] == -1 for row in ledger),
        "long_expectancy_bps": _direction_expectancy(ledger, 1),
        "short_expectancy_bps": _direction_expectancy(ledger, -1),
        "gross_bps": float(sum(row["gross_bps"] for row in ledger)),
        "fees_bps": float(sum(row["fee_bps"] for row in ledger)),
        "market_execution_bps": float(sum(row["market_execution_bps"] for row in ledger)),
        "funding_bps": float(sum(row["funding_bps"] for row in ledger)),
        "monthly_net_bps": monthly,
    }


def _direction_expectancy(ledger: list[dict[str, Any]], direction: int) -> float | None:
    values = [row["net_bps"] for row in ledger if row["direction"] == direction]
    return float(np.mean(values)) if values else None


def choose_policy(
    frame: pl.DataFrame,
    probabilities: np.ndarray,
    config: BenchmarkConfig,
    *,
    seed_offset: int = 0,
    bootstrap_repetitions: int | None = None,
) -> tuple[Policy, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    repetitions = (
        min(500, config.compute.bootstrap_resamples)
        if bootstrap_repetitions is None
        else bootstrap_repetitions
    )
    if repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    for threshold in config.selection.probability_thresholds:
        for margin in config.selection.directional_margins:
            policy = Policy(threshold, margin)
            actions = policy_actions(probabilities, policy)
            metrics = economic_metrics(
                frame,
                actions,
                execution_cost_multiplier=1.0,
                bootstrap_repetitions=repetitions,
                seed=config.compute.random_seed + seed_offset,
            )
            lower_80, _ = circular_block_expectancy_ci(
                trade_ledger(frame, actions),
                start=frame["bucket_start"].min().date(),
                end=frame["bucket_start"].max().date(),
                repetitions=repetitions,
                seed=config.compute.random_seed + seed_offset,
                confidence=0.80,
            )
            candidates.append(
                {
                    "probability_threshold": threshold,
                    "directional_margin": margin,
                    "bootstrap_80_lower_bps": lower_80,
                    **metrics,
                }
            )
    eligible = [
        row
        for row in candidates
        if row["trades"] >= config.selection.minimum_calibration_trades
        and row["bootstrap_80_lower_bps"] is not None
        and row["bootstrap_80_lower_bps"] > 0
    ]
    if not eligible:
        return Policy(1.0, 1.0, no_trade=True), candidates
    selected = max(
        eligible,
        key=lambda row: (
            row["bootstrap_80_lower_bps"],
            row["probability_threshold"],
            row["directional_margin"],
        ),
    )
    return (
        Policy(
            selected["probability_threshold"],
            selected["directional_margin"],
        ),
        candidates,
    )


def paired_accuracy_uplift_ci(
    frame: pl.DataFrame,
    model_probabilities: np.ndarray,
    baseline_predictions: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    block_days: int = 7,
) -> tuple[float, float]:
    truth = frame["label"].to_numpy()
    model_correct = (predicted_classes(model_probabilities) == truth).astype(float)
    baseline_correct = (baseline_predictions == truth).astype(float)
    timestamps = frame["bucket_start"].to_list()
    first_day = timestamps[0].date()
    last_day = timestamps[-1].date()
    days = (last_day - first_day).days + 1
    differences: list[list[float]] = [[] for _ in range(days)]
    for timestamp, model_value, baseline_value in zip(
        timestamps, model_correct, baseline_correct, strict=True
    ):
        differences[(timestamp.date() - first_day).days].append(model_value - baseline_value)
    daily_sum = np.asarray([sum(values) for values in differences])
    daily_count = np.asarray([len(values) for values in differences])
    rng = np.random.default_rng(seed)
    block_count = math.ceil(days / block_days)
    offsets = np.arange(block_days)
    estimates = np.empty(repetitions)
    for repetition in range(repetitions):
        starts = rng.integers(0, days, size=block_count)
        indices = ((starts[:, None] + offsets) % days).ravel()[:days]
        estimates[repetition] = daily_sum[indices].sum() / daily_count[indices].sum()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))
