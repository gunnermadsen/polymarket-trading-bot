from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .evaluation import circular_block_expectancy_ci, trade_ledger
from .regression_models import NET_TARGET_COLUMNS

POLICY_HURDLES_BPS = (0.0, 3.0, 6.0, 10.0)
POLICY_ADVANTAGES_BPS = (0.0, 3.0, 6.0)
MINIMUM_POLICY_TRADES = 50
FEE_COUNTERFACTUALS_BPS = {
    "taker_10_bps": 10.0,
    "hybrid_7_bps": 7.0,
    "maker_4_bps": 4.0,
    "zero_fee": 0.0,
}


@dataclass(frozen=True)
class RegressionPolicy:
    hurdle_bps: float
    advantage_bps: float
    no_trade: bool = False


def regression_policy_actions(
    predictions: np.ndarray,
    policy: RegressionPolicy,
) -> np.ndarray:
    values = _prediction_matrix(predictions)
    if policy.no_trade:
        return np.zeros(values.shape[0], dtype=np.int8)
    if not np.isfinite(policy.hurdle_bps) or not np.isfinite(policy.advantage_bps):
        raise ValueError("policy thresholds must be finite")
    if policy.hurdle_bps < 0.0 or policy.advantage_bps < 0.0:
        raise ValueError("policy thresholds must be nonnegative")

    long_prediction = values[:, 0]
    short_prediction = values[:, 1]
    choose_long = long_prediction > short_prediction
    choose_short = short_prediction > long_prediction
    best = np.maximum(long_prediction, short_prediction)
    advantage = np.abs(long_prediction - short_prediction)
    eligible = (best >= policy.hurdle_bps) & (advantage >= policy.advantage_bps)
    actions = np.zeros(values.shape[0], dtype=np.int8)
    actions[eligible & choose_long] = 1
    actions[eligible & choose_short] = -1
    return actions


def regression_diagnostics(
    frame: pl.DataFrame,
    predictions: np.ndarray,
    *,
    deciles: int = 10,
) -> dict[str, Any]:
    values = _prediction_matrix(predictions)
    if frame.height != values.shape[0]:
        raise ValueError("prediction count does not match frame")
    observed = np.column_stack(
        [_finite_vector(frame[column].to_numpy(), name=column) for column in NET_TARGET_COLUMNS]
    )
    sides = {
        "long": regression_metrics(observed[:, 0], values[:, 0], deciles=deciles),
        "short": regression_metrics(observed[:, 1], values[:, 1], deciles=deciles),
    }
    return {
        **sides,
        "pooled": regression_metrics(
            observed.reshape(-1),
            values.reshape(-1),
            deciles=deciles,
        ),
    }


def regression_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    deciles: int = 10,
) -> dict[str, Any]:
    actual = _finite_vector(observed, name="observed")
    estimate = _finite_vector(predicted, name="predicted")
    if actual.shape != estimate.shape:
        raise ValueError("observed and predicted counts do not match")
    if actual.size == 0:
        raise ValueError("regression diagnostics require at least one observation")

    rank_correlation: float | None
    if actual.size < 2 or np.ptp(actual) == 0.0 or np.ptp(estimate) == 0.0:
        rank_correlation = None
    else:
        statistic = float(spearmanr(actual, estimate).statistic)
        rank_correlation = statistic if np.isfinite(statistic) else None
    r_squared = float(r2_score(actual, estimate)) if actual.size >= 2 else None
    if r_squared is not None and not np.isfinite(r_squared):
        r_squared = None

    return {
        "observations": int(actual.size),
        "mae_bps": _finite_float(mean_absolute_error(actual, estimate)),
        "rmse_bps": _finite_float(root_mean_squared_error(actual, estimate)),
        "r2": r_squared,
        "spearman": rank_correlation,
        "decile_calibration": decile_calibration(actual, estimate, bins=deciles),
    }


def decile_calibration(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    bins: int = 10,
) -> list[dict[str, Any]]:
    actual = _finite_vector(observed, name="observed")
    estimate = _finite_vector(predicted, name="predicted")
    if actual.shape != estimate.shape:
        raise ValueError("observed and predicted counts do not match")
    if actual.size == 0:
        raise ValueError("calibration data must not be empty")
    if bins <= 0:
        raise ValueError("bins must be positive")

    order = np.argsort(estimate, kind="stable")
    chunks = np.array_split(order, min(bins, actual.size))
    calibration: list[dict[str, Any]] = []
    for bin_index, indices in enumerate(chunks, start=1):
        predicted_mean = _finite_float(np.mean(estimate[indices]))
        observed_mean = _finite_float(np.mean(actual[indices]))
        calibration.append(
            {
                "bin": bin_index,
                "count": int(indices.size),
                "predicted_mean_bps": predicted_mean,
                "observed_mean_bps": observed_mean,
                "bias_bps": _finite_float(predicted_mean - observed_mean),
            }
        )
    return calibration


def choose_regression_policy(
    frame: pl.DataFrame,
    predictions: np.ndarray,
    *,
    seed: int,
    bootstrap_repetitions: int,
    minimum_trades: int = MINIMUM_POLICY_TRADES,
) -> tuple[RegressionPolicy, list[dict[str, Any]]]:
    values = _prediction_matrix(predictions)
    if frame.height != values.shape[0]:
        raise ValueError("prediction count does not match frame")
    if frame.height == 0:
        raise ValueError("policy selection frame must not be empty")
    if bootstrap_repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    if minimum_trades <= 0:
        raise ValueError("minimum_trades must be positive")

    start = frame["bucket_start"].min().date()
    end = frame["bucket_start"].max().date()
    candidates: list[dict[str, Any]] = []
    for hurdle in POLICY_HURDLES_BPS:
        for advantage in POLICY_ADVANTAGES_BPS:
            policy = RegressionPolicy(hurdle_bps=hurdle, advantage_bps=advantage)
            actions = regression_policy_actions(values, policy)
            ledger = trade_ledger(frame, actions)
            metrics = pooled_ledger_economics(
                ledger,
                evaluation_rows=frame.height,
                start=start,
                end=end,
                bootstrap_repetitions=bootstrap_repetitions,
                seed=seed,
            )
            lower_80, upper_80 = circular_block_expectancy_ci(
                ledger,
                start=start,
                end=end,
                repetitions=bootstrap_repetitions,
                seed=seed,
                confidence=0.80,
                block_days=7,
            )
            candidates.append(
                {
                    "hurdle_bps": hurdle,
                    "advantage_bps": advantage,
                    "bootstrap_80_lower_bps": lower_80,
                    "bootstrap_80_upper_bps": upper_80,
                    **metrics,
                }
            )

    eligible = [
        candidate
        for candidate in candidates
        if candidate["trades"] >= minimum_trades
        and candidate["bootstrap_80_lower_bps"] is not None
        and candidate["bootstrap_80_lower_bps"] > 0.0
    ]
    if not eligible:
        return RegressionPolicy(0.0, 0.0, no_trade=True), candidates

    selected = max(
        eligible,
        key=lambda candidate: (
            candidate["bootstrap_80_lower_bps"],
            candidate["net_expectancy_bps"],
            -candidate["trades"],
            candidate["hurdle_bps"],
            candidate["advantage_bps"],
        ),
    )
    return (
        RegressionPolicy(
            hurdle_bps=selected["hurdle_bps"],
            advantage_bps=selected["advantage_bps"],
        ),
        candidates,
    )


def economic_metrics_for_actions(
    frame: pl.DataFrame,
    actions: np.ndarray,
    *,
    execution_cost_multiplier: float,
    bootstrap_repetitions: int,
    seed: int,
    fee_cost_bps_override: float | None = None,
) -> dict[str, Any]:
    if frame.height == 0:
        raise ValueError("evaluation frame must not be empty")
    ledger = trade_ledger(
        frame,
        actions,
        execution_cost_multiplier=execution_cost_multiplier,
        fee_cost_bps_override=fee_cost_bps_override,
    )
    return pooled_ledger_economics(
        ledger,
        evaluation_rows=frame.height,
        start=frame["bucket_start"].min().date(),
        end=frame["bucket_start"].max().date(),
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )


def pooled_ledger_economics(
    ledger: list[dict[str, Any]],
    *,
    evaluation_rows: int,
    start: date,
    end: date,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if evaluation_rows <= 0:
        raise ValueError("evaluation_rows must be positive")
    if start > end:
        raise ValueError("ledger date range is invalid")
    if bootstrap_repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    if not ledger:
        return _empty_economic_metrics()

    ordered = sorted(ledger, key=lambda row: (row["entry_at"], row["exit_at"]))
    pnl = _finite_vector(
        np.asarray([row["net_bps"] for row in ordered]),
        name="ledger net_bps",
    )
    cumulative = np.cumsum(pnl)
    peaks = np.maximum.accumulate(np.insert(cumulative, 0, 0.0))[1:]
    drawdown = peaks - cumulative
    gains = float(pnl[pnl > 0.0].sum())
    losses = float(-pnl[pnl < 0.0].sum())
    monthly = _monthly_totals(ordered, start=start, end=end)
    lower, upper = circular_block_expectancy_ci(
        ordered,
        start=start,
        end=end,
        repetitions=bootstrap_repetitions,
        seed=seed,
        confidence=0.95,
        block_days=7,
    )
    return {
        "trades": len(ordered),
        "action_coverage": _finite_float(len(ordered) / evaluation_rows),
        "net_expectancy_bps": _finite_float(np.mean(pnl)),
        "median_net_bps": _finite_float(np.median(pnl)),
        "bootstrap_95_lower_bps": lower,
        "bootstrap_95_upper_bps": upper,
        "profit_factor": _finite_float(gains / losses) if losses > 0.0 else None,
        "win_rate": _finite_float(np.mean(pnl > 0.0)),
        "total_net_bps": _finite_float(pnl.sum()),
        "max_drawdown_bps": _finite_float(drawdown.max(initial=0.0)),
        "positive_month_fraction": _finite_float(
            np.mean(np.asarray(list(monthly.values())) > 0.0)
        ),
        "long_trades": sum(row["direction"] == 1 for row in ordered),
        "short_trades": sum(row["direction"] == -1 for row in ordered),
        "long_expectancy_bps": _direction_expectancy(ordered, 1),
        "short_expectancy_bps": _direction_expectancy(ordered, -1),
        "gross_bps": _finite_float(sum(row["gross_bps"] for row in ordered)),
        "fees_bps": _finite_float(sum(row["fee_bps"] for row in ordered)),
        "market_execution_bps": _finite_float(
            sum(row["market_execution_bps"] for row in ordered)
        ),
        "funding_bps": _finite_float(sum(row["funding_bps"] for row in ordered)),
        "monthly_net_bps": monthly,
    }


def fee_counterfactuals(
    frame: pl.DataFrame,
    actions: np.ndarray,
    *,
    execution_cost_multiplier: float,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    action_vector = np.asarray(actions, dtype=np.int8)
    if action_vector.ndim != 1 or action_vector.size != frame.height:
        raise ValueError("action count does not match frame")
    results: dict[str, dict[str, Any]] = {}
    for name, round_trip_fee_bps in FEE_COUNTERFACTUALS_BPS.items():
        results[name] = economic_metrics_for_actions(
            frame,
            action_vector,
            execution_cost_multiplier=execution_cost_multiplier,
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed,
            fee_cost_bps_override=round_trip_fee_bps,
        )
        results[name]["round_trip_fee_bps"] = round_trip_fee_bps
    return results


def _empty_economic_metrics() -> dict[str, Any]:
    return {
        "trades": 0,
        "action_coverage": 0.0,
        "net_expectancy_bps": None,
        "median_net_bps": None,
        "bootstrap_95_lower_bps": None,
        "bootstrap_95_upper_bps": None,
        "profit_factor": None,
        "win_rate": None,
        "total_net_bps": 0.0,
        "max_drawdown_bps": 0.0,
        "positive_month_fraction": 0.0,
        "long_trades": 0,
        "short_trades": 0,
        "long_expectancy_bps": None,
        "short_expectancy_bps": None,
        "gross_bps": 0.0,
        "fees_bps": 0.0,
        "market_execution_bps": 0.0,
        "funding_bps": 0.0,
        "monthly_net_bps": {},
    }


def _monthly_totals(
    ledger: list[dict[str, Any]],
    *,
    start: date,
    end: date,
) -> dict[str, float]:
    monthly: dict[str, float] = {}
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        monthly[f"{year:04d}-{month:02d}"] = 0.0
        month += 1
        if month == 13:
            year += 1
            month = 1
    for trade in ledger:
        key = trade["decision_time"].strftime("%Y-%m")
        if key in monthly:
            monthly[key] = _finite_float(monthly[key] + float(trade["net_bps"]))
    return monthly


def _direction_expectancy(
    ledger: list[dict[str, Any]],
    direction: int,
) -> float | None:
    values = [float(row["net_bps"]) for row in ledger if row["direction"] == direction]
    return _finite_float(np.mean(values)) if values else None


def _prediction_matrix(predictions: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(NET_TARGET_COLUMNS):
        raise ValueError("predictions must have shape (rows, 2) in long, short order")
    if not np.isfinite(values).all():
        raise ValueError("predictions contain non-finite values")
    return values


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values")
    return vector


def _finite_float(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("metric is not finite")
    return result
