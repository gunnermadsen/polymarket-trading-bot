"""Price, calibration, and asymmetric-payout evaluation for early predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .early_value_config import EarlyValueConfig


def attach_execution_value(
    predictions: pl.DataFrame,
    prices: pl.DataFrame,
) -> pl.DataFrame:
    joined = predictions.join(
        prices,
        on=["market_id", "window_start", "observed_at", "seconds_elapsed", "label_up"],
        how="inner",
        validate="m:1",
    )
    valid = joined.filter(
        pl.col("fee_rate").is_not_null()
        & pl.col("fee_rate").is_finite()
        & (pl.col("fee_rate") >= 0)
        & pl.col("yes_ask_vwap_5").is_between(0.0, 1.0, closed="right")
        & pl.col("no_ask_vwap_5").is_between(0.0, 1.0, closed="right")
    )
    return (
        valid.with_columns(
            (pl.col("fee_rate") * pl.col("yes_ask_vwap_5") * (1.0 - pl.col("yes_ask_vwap_5"))).alias("yes_fee_per_share"),
            (pl.col("fee_rate") * pl.col("no_ask_vwap_5") * (1.0 - pl.col("no_ask_vwap_5"))).alias("no_fee_per_share"),
        )
        .with_columns(
            (pl.col("yes_ask_vwap_5") + pl.col("yes_fee_per_share")).alias("yes_cost_per_share"),
            (pl.col("no_ask_vwap_5") + pl.col("no_fee_per_share")).alias("no_cost_per_share"),
        )
        .with_columns(
            (pl.col("probability_yes") - pl.col("yes_cost_per_share")).alias("yes_edge_per_share"),
            (1.0 - pl.col("probability_yes") - pl.col("no_cost_per_share")).alias("no_edge_per_share"),
        )
    )


def policy_ledgers(scored: pl.DataFrame, config: EarlyValueConfig) -> dict[str, pl.DataFrame]:
    with_dual = scored.with_columns(
        (pl.col("yes_edge_per_share") >= pl.col("no_edge_per_share")).alias("dual_yes"),
        pl.max_horizontal("yes_edge_per_share", "no_edge_per_share").alias("dual_edge"),
    ).with_columns(
        pl.when(pl.col("dual_yes")).then(pl.col("yes_cost_per_share")).otherwise(pl.col("no_cost_per_share")).alias("dual_cost")
    )
    policies = {
        "reference_89_confidence": _select_first(
            with_dual.filter(pl.col("confidence") >= config.confidence_reference),
            side_expression=pl.col("predicted_yes") == 1,
            quantity=config.quantity,
        ),
        "predicted_side_positive_value": _select_first(
            with_dual.filter(
                pl.when(pl.col("predicted_yes") == 1)
                .then(pl.col("yes_edge_per_share"))
                .otherwise(pl.col("no_edge_per_share"))
                > config.minimum_edge_per_share
            ),
            side_expression=pl.col("predicted_yes") == 1,
            quantity=config.quantity,
        ),
        "dual_side_positive_value": _select_first(
            with_dual.filter(pl.col("dual_edge") > config.minimum_edge_per_share),
            side_expression=pl.col("dual_yes"),
            quantity=config.quantity,
        ),
        "early_cheap_dual_side_value": _select_first(
            with_dual.filter(
                (pl.col("seconds_elapsed") < 60)
                & pl.col("dual_cost").is_between(
                    config.cheap_price_min,
                    config.cheap_price_max,
                    closed="left",
                )
                & (pl.col("dual_edge") > config.minimum_edge_per_share)
            ),
            side_expression=pl.col("dual_yes"),
            quantity=config.quantity,
        ),
    }
    return policies


def ledger_metrics(ledger: pl.DataFrame) -> dict[str, Any]:
    if ledger.is_empty():
        return {
            "trades": 0,
            "accuracy": None,
            "net_profit": 0.0,
            "net_expectancy_per_trade": None,
            "profit_factor": None,
            "mean_entry_cost_per_share": None,
            "mean_entry_second": None,
        }
    profits = ledger["realized_net"].to_numpy()
    gross_profit = float(profits[profits > 0].sum())
    gross_loss = float(-profits[profits < 0].sum())
    return {
        "trades": ledger.height,
        "markets": ledger["market_id"].n_unique(),
        "accuracy": float(ledger["won"].mean()),
        "net_profit": float(profits.sum()),
        "net_expectancy_per_trade": float(profits.mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "mean_entry_cost_per_share": float(ledger["entry_cost_per_share"].mean()),
        "median_entry_cost_per_share": float(ledger["entry_cost_per_share"].median()),
        "mean_entry_second": float(ledger["seconds_elapsed"].mean()),
        "maximum_loss": float(profits.min()),
    }


def bootstrap_net_expectancy(
    ledger: pl.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float] | None:
    if ledger.is_empty():
        return None
    daily = (
        ledger.with_columns(pl.col("window_start").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("realized_net").sum().alias("net"), pl.len().alias("trades"))
        .sort("date")
    )
    net = daily["net"].to_numpy()
    trades = daily["trades"].to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, len(net), len(net))
        samples[index] = net[chosen].sum() / max(trades[chosen].sum(), 1)
    return {
        "lower_95": float(np.quantile(samples, 0.025)),
        "median": float(np.quantile(samples, 0.5)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def price_by_second(prices: pl.DataFrame) -> list[dict[str, Any]]:
    return (
        prices.group_by("seconds_elapsed")
        .agg(
            pl.len().alias("candidate_rows"),
            pl.col("market_id").n_unique().alias("markets"),
            pl.col("yes_ask_vwap_5").is_not_null().sum().alias("yes_executable_rows"),
            pl.col("no_ask_vwap_5").is_not_null().sum().alias("no_executable_rows"),
            pl.col("yes_ask_vwap_5").mean().alias("mean_yes_ask_vwap_5"),
            pl.col("no_ask_vwap_5").mean().alias("mean_no_ask_vwap_5"),
            pl.col("yes_ask_vwap_5").median().alias("median_yes_ask_vwap_5"),
            pl.col("no_ask_vwap_5").median().alias("median_no_ask_vwap_5"),
        )
        .with_columns(
            (pl.col("yes_executable_rows") / pl.col("candidate_rows")).alias("yes_execution_coverage"),
            (pl.col("no_executable_rows") / pl.col("candidate_rows")).alias("no_execution_coverage"),
        )
        .sort("seconds_elapsed")
        .to_dicts()
    )


def calibration_by_time_and_cost(scored: pl.DataFrame) -> list[dict[str, Any]]:
    long = pl.concat(
        (
            scored.select(
                "model", "market_id", "seconds_elapsed",
                pl.lit("YES").alias("side"),
                pl.col("probability_yes").alias("predicted_win_probability"),
                pl.col("label_up").alias("won"),
                pl.col("yes_cost_per_share").alias("cost_per_share"),
            ),
            scored.select(
                "model", "market_id", "seconds_elapsed",
                pl.lit("NO").alias("side"),
                (1.0 - pl.col("probability_yes")).alias("predicted_win_probability"),
                (1 - pl.col("label_up")).alias("won"),
                pl.col("no_cost_per_share").alias("cost_per_share"),
            ),
        )
    ).with_columns(
        pl.when(pl.col("seconds_elapsed") < 15).then(pl.lit("05-14"))
        .when(pl.col("seconds_elapsed") < 30).then(pl.lit("15-29"))
        .when(pl.col("seconds_elapsed") < 45).then(pl.lit("30-44"))
        .when(pl.col("seconds_elapsed") < 60).then(pl.lit("45-59"))
        .when(pl.col("seconds_elapsed") < 120).then(pl.lit("60-119"))
        .otherwise(pl.lit("120-240")).alias("time_band"),
        (pl.col("cost_per_share") * 10).floor().clip(0, 9).cast(pl.Int8).alias("price_decile"),
    )
    return (
        long.group_by("model", "side", "time_band", "price_decile")
        .agg(
            pl.len().alias("rows"),
            pl.col("market_id").n_unique().alias("markets"),
            pl.col("predicted_win_probability").mean().alias("mean_predicted_probability"),
            pl.col("won").mean().alias("actual_win_rate"),
            pl.col("cost_per_share").mean().alias("mean_cost_per_share"),
        )
        .with_columns(
            (pl.col("actual_win_rate") - pl.col("mean_cost_per_share")).alias("realized_edge_before_sampling_error")
        )
        .sort("model", "time_band", "side", "price_decile")
        .to_dicts()
    )


def _select_first(
    eligible: pl.DataFrame,
    *,
    side_expression: pl.Expr,
    quantity: float,
) -> pl.DataFrame:
    if eligible.is_empty():
        return eligible.with_columns(
            pl.lit(None, dtype=pl.Boolean).alias("selected_yes"),
            pl.lit(None, dtype=pl.Boolean).alias("won"),
            pl.lit(None, dtype=pl.Float64).alias("entry_cost_per_share"),
            pl.lit(None, dtype=pl.Float64).alias("realized_net"),
        )
    first = (
        eligible.sort("model", "market_id", "seconds_elapsed", "observed_at")
        .group_by("model", "market_id", maintain_order=True)
        .first()
        .with_columns(side_expression.alias("selected_yes"))
        .with_columns(
            pl.when(pl.col("selected_yes")).then(pl.col("yes_cost_per_share")).otherwise(pl.col("no_cost_per_share")).alias("entry_cost_per_share"),
            pl.when(pl.col("selected_yes")).then(pl.col("label_up") == 1).otherwise(pl.col("label_up") == 0).alias("won"),
        )
        .with_columns(
            ((pl.col("won").cast(pl.Float64) - pl.col("entry_cost_per_share")) * quantity).alias("realized_net")
        )
    )
    return first
