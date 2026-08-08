"""Economic policy selection for lower-cost asymmetric-value predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .asymmetric_value_config import AsymmetricValueConfig, ValuePolicy

EXECUTION_STRESS_PER_SHARE = 0.01


def score_two_sided_value(predictions: pl.DataFrame) -> pl.DataFrame:
    """Score YES and NO independently and select the greater calibrated edge."""

    required = {
        "probability_yes",
        "yes_cost_per_share",
        "no_cost_per_share",
        "yes_execution_cost_per_share",
        "no_execution_cost_per_share",
        "label_up",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError("two-sided value predictions are missing columns: " + ", ".join(missing))
    invalid = predictions.filter(
        pl.col("probability_yes").is_null()
        | ~pl.col("probability_yes").is_finite()
        | ~pl.col("probability_yes").is_between(0.0, 1.0, closed="both")
    )
    if invalid.height:
        raise ValueError("two-sided value probabilities must be finite and inside [0, 1]")

    scored = predictions.with_columns(
        (pl.col("probability_yes") - pl.col("yes_cost_per_share")).alias("yes_edge_per_share"),
        (1.0 - pl.col("probability_yes") - pl.col("no_cost_per_share")).alias(
            "no_edge_per_share"
        ),
        (pl.col("probability_yes") >= 0.5).alias("argmax_yes"),
    ).with_columns(
        (pl.col("yes_edge_per_share") >= pl.col("no_edge_per_share")).alias("selected_yes"),
        pl.max_horizontal("yes_edge_per_share", "no_edge_per_share").alias(
            "selected_edge_per_share"
        ),
    )
    return scored.with_columns(
        pl.when(pl.col("selected_yes"))
        .then(pl.col("probability_yes"))
        .otherwise(1.0 - pl.col("probability_yes"))
        .alias("selected_probability"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_cost_per_share"))
        .otherwise(pl.col("no_cost_per_share"))
        .alias("selected_admission_cost_per_share"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_execution_cost_per_share"))
        .otherwise(pl.col("no_execution_cost_per_share"))
        .alias("selected_execution_cost_per_share"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_ask_vwap_5"))
        .otherwise(pl.col("no_ask_vwap_5"))
        .alias("selected_share_price"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("label_up") == 1)
        .otherwise(pl.col("label_up") == 0)
        .alias("won"),
        (pl.col("selected_yes") != pl.col("argmax_yes")).alias("selected_underdog"),
    )


def policy_ledger(
    scored: pl.DataFrame,
    policy: ValuePolicy,
    *,
    quantity: float,
    maximum_depth_participation: float,
) -> pl.DataFrame:
    yes_eligible = (
        pl.col("yes_ask_vwap_5").is_between(
            policy.minimum_share_price,
            policy.maximum_share_price,
            closed="left",
        )
        & (pl.col("yes_cost_per_share") <= policy.maximum_cost_per_share)
        & (pl.col("yes_edge_per_share") >= policy.minimum_edge_per_share)
        & (
            quantity
            <= pl.col("yes_ask_depth") * maximum_depth_participation
        )
    )
    no_eligible = (
        pl.col("no_ask_vwap_5").is_between(
            policy.minimum_share_price,
            policy.maximum_share_price,
            closed="left",
        )
        & (pl.col("no_cost_per_share") <= policy.maximum_cost_per_share)
        & (pl.col("no_edge_per_share") >= policy.minimum_edge_per_share)
        & (
            quantity
            <= pl.col("no_ask_depth") * maximum_depth_participation
        )
    )
    eligible = scored.with_columns(
        yes_eligible.alias("_yes_policy_eligible"),
        no_eligible.alias("_no_policy_eligible"),
    ).filter(
        (pl.col("seconds_elapsed") <= policy.maximum_entry_second)
        & (pl.col("_yes_policy_eligible") | pl.col("_no_policy_eligible"))
    ).with_columns(
        pl.when(
            pl.col("_yes_policy_eligible") & pl.col("_no_policy_eligible")
        )
        .then(pl.col("yes_edge_per_share") >= pl.col("no_edge_per_share"))
        .otherwise(pl.col("_yes_policy_eligible"))
        .alias("selected_yes")
    ).with_columns(
        pl.when(pl.col("selected_yes"))
        .then(pl.col("probability_yes"))
        .otherwise(1.0 - pl.col("probability_yes"))
        .alias("selected_probability"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_cost_per_share"))
        .otherwise(pl.col("no_cost_per_share"))
        .alias("selected_admission_cost_per_share"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_execution_cost_per_share"))
        .otherwise(pl.col("no_execution_cost_per_share"))
        .alias("selected_execution_cost_per_share"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_ask_vwap_5"))
        .otherwise(pl.col("no_ask_vwap_5"))
        .alias("selected_share_price"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_edge_per_share"))
        .otherwise(pl.col("no_edge_per_share"))
        .alias("selected_edge_per_share"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("label_up") == 1)
        .otherwise(pl.col("label_up") == 0)
        .alias("won"),
        (pl.col("selected_yes") != pl.col("argmax_yes")).alias(
            "selected_underdog"
        ),
    ).drop("_yes_policy_eligible", "_no_policy_eligible")
    return (
        eligible.sort("model", "market_id", "seconds_elapsed", "observed_at")
        .group_by("model", "market_id", maintain_order=True)
        .first()
        .with_columns(
            pl.lit(policy.name).alias("policy"),
            pl.lit(quantity).alias("quantity"),
            (
                (
                    pl.col("won").cast(pl.Float64)
                    - pl.col("selected_execution_cost_per_share")
                )
                * quantity
            ).alias("realized_net"),
            (pl.col("selected_execution_cost_per_share") * quantity).alias("entry_debit"),
        )
    )


def confidence_reference_ledger(
    predictions: pl.DataFrame,
    *,
    threshold: float,
    minimum_edge_per_share: float,
    quantity: float,
) -> pl.DataFrame:
    """Return the frozen higher-probability-side confidence control."""

    with_side = predictions.with_columns(
        (pl.col("probability_yes") >= 0.5).alias("selected_yes"),
        pl.max_horizontal("probability_yes", 1.0 - pl.col("probability_yes")).alias(
            "confidence"
        ),
    ).with_columns(
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_execution_cost_per_share"))
        .otherwise(pl.col("no_execution_cost_per_share"))
        .alias("selected_execution_cost_per_share"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_cost_per_share"))
        .otherwise(pl.col("no_cost_per_share"))
        .alias("selected_admission_cost_per_share"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_ask_vwap_5"))
        .otherwise(pl.col("no_ask_vwap_5"))
        .alias("selected_share_price"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("label_up") == 1)
        .otherwise(pl.col("label_up") == 0)
        .alias("won"),
    )
    with_edge = with_side.with_columns(
        (pl.col("confidence") - pl.col("selected_admission_cost_per_share")).alias(
            "control_edge_per_share"
        )
    )
    return (
        with_edge.filter(
            (pl.col("confidence") >= threshold)
            & (pl.col("control_edge_per_share") >= minimum_edge_per_share)
        )
        .sort("model", "market_id", "seconds_elapsed", "observed_at")
        .group_by("model", "market_id", maintain_order=True)
        .first()
        .with_columns(
            pl.lit("reference_89_confidence").alias("policy"),
            pl.lit(quantity).alias("quantity"),
            pl.lit(False).alias("selected_underdog"),
            pl.col("confidence").alias("selected_probability"),
            pl.col("control_edge_per_share").alias("selected_edge_per_share"),
            (
                (
                    pl.col("won").cast(pl.Float64)
                    - pl.col("selected_execution_cost_per_share")
                )
                * quantity
            ).alias("realized_net"),
            (pl.col("selected_execution_cost_per_share") * quantity).alias("entry_debit"),
        )
    )


def current_policy_reference_ledger(
    predictions: pl.DataFrame,
    *,
    execution: pl.DataFrame | None = None,
    threshold: float,
    minimum_entry_second: int,
    maximum_entry_second: int,
    minimum_share_price: float,
    maximum_share_price: float,
    maximum_depth_participation: float,
    quantity: float,
) -> pl.DataFrame:
    """Emulate terminal first crossing, then validate contemporaneous execution.

    The runtime consumes a market after its first qualifying directional prediction,
    even when the contemporaneous quote later fails execution validation. Selecting
    the crossing before the execution join prevents a later executable row from being
    substituted for that terminal rejection.
    """

    crossings = (
        predictions.with_columns(
            (pl.col("probability_yes") >= 0.5).alias("selected_yes"),
            pl.max_horizontal("probability_yes", 1.0 - pl.col("probability_yes")).alias(
                "confidence"
            ),
        )
        .filter(
            (pl.col("confidence") >= threshold)
            & (
                pl.col("seconds_elapsed").is_between(
                    minimum_entry_second,
                    maximum_entry_second,
                    closed="both",
                )
            )
        )
        .sort("model", "market_id", "seconds_elapsed", "observed_at")
        .group_by("model", "market_id", maintain_order=True)
        .first()
    )
    if execution is not None:
        keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
        execution_columns = [
            "yes_ask_vwap_5",
            "yes_ask_depth",
            "no_ask_vwap_5",
            "no_ask_depth",
            "yes_execution_cost_per_share",
            "no_execution_cost_per_share",
        ]
        missing = sorted({*keys, *execution_columns} - set(execution.columns))
        if missing:
            raise ValueError(
                "current-policy execution evidence is missing columns: "
                + ", ".join(missing)
            )
        duplicate_execution = (
            execution.group_by(*keys).len().filter(pl.col("len") != 1)
        )
        if duplicate_execution.height:
            raise ValueError("current-policy execution evidence contains duplicate keys")
        crossings = crossings.join(
            execution.select(*keys, *execution_columns),
            on=keys,
            how="left",
            validate="1:1",
        )

    with_side = crossings.with_columns(
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_execution_cost_per_share"))
        .otherwise(pl.col("no_execution_cost_per_share"))
        .alias("selected_execution_cost_per_share"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_ask_vwap_5"))
        .otherwise(pl.col("no_ask_vwap_5"))
        .alias("selected_share_price"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_ask_depth"))
        .otherwise(pl.col("no_ask_depth"))
        .alias("selected_ask_depth"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("label_up") == 1)
        .otherwise(pl.col("label_up") == 0)
        .alias("won"),
    )
    return (
        with_side.filter(
            pl.col("selected_execution_cost_per_share").is_not_null()
            & (
                pl.col("selected_share_price").is_between(
                    minimum_share_price,
                    maximum_share_price,
                    closed="both",
                )
            )
            & (
                pl.col("selected_ask_depth") * maximum_depth_participation
                >= quantity
            )
        )
        .with_columns(
            pl.lit("frozen_current_execute_directional_89").alias("policy"),
            pl.lit(quantity).alias("quantity"),
            pl.lit(False).alias("selected_underdog"),
            pl.col("confidence").alias("selected_probability"),
            pl.col("selected_execution_cost_per_share").alias(
                "selected_admission_cost_per_share"
            ),
            (
                pl.col("confidence")
                - pl.col("selected_execution_cost_per_share")
            ).alias("selected_edge_per_share"),
            (
                (
                    pl.col("won").cast(pl.Float64)
                    - pl.col("selected_execution_cost_per_share")
                )
                * quantity
            ).alias("realized_net"),
            (pl.col("selected_execution_cost_per_share") * quantity).alias(
                "entry_debit"
            ),
        )
    )


def confidence_control_ledger(
    predictions: pl.DataFrame,
    *,
    threshold: float,
    maximum_entry_second: int,
    maximum_cost_per_share: float,
    minimum_edge_per_share: float,
    maximum_depth_participation: float,
    quantity: float,
) -> pl.DataFrame:
    """First argmax-side entry for a conventional confidence-threshold control."""

    with_side = predictions.with_columns(
        (pl.col("probability_yes") >= 0.5).alias("selected_yes"),
        pl.max_horizontal("probability_yes", 1.0 - pl.col("probability_yes")).alias(
            "confidence"
        ),
    ).with_columns(
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_execution_cost_per_share"))
        .otherwise(pl.col("no_execution_cost_per_share"))
        .alias("selected_execution_cost_per_share"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_cost_per_share"))
        .otherwise(pl.col("no_cost_per_share"))
        .alias("selected_admission_cost_per_share"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_ask_vwap_5"))
        .otherwise(pl.col("no_ask_vwap_5"))
        .alias("selected_share_price"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_ask_depth"))
        .otherwise(pl.col("no_ask_depth"))
        .alias("selected_ask_depth"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("label_up") == 1)
        .otherwise(pl.col("label_up") == 0)
        .alias("won"),
    )
    with_edge = with_side.with_columns(
        (pl.col("confidence") - pl.col("selected_admission_cost_per_share")).alias(
            "control_edge_per_share"
        )
    )
    return (
        with_edge.filter(
            (pl.col("confidence") >= threshold)
            & (pl.col("control_edge_per_share") >= minimum_edge_per_share)
            & (pl.col("seconds_elapsed") <= maximum_entry_second)
            & (
                pl.col("selected_admission_cost_per_share")
                <= maximum_cost_per_share
            )
            & (
                quantity
                <= pl.col("selected_ask_depth")
                * maximum_depth_participation
            )
        )
        .sort("model", "market_id", "seconds_elapsed", "observed_at")
        .group_by("model", "market_id", maintain_order=True)
        .first()
        .with_columns(
            pl.lit(f"confidence_{threshold:.2f}").alias("policy"),
            pl.lit(quantity).alias("quantity"),
            pl.lit(False).alias("selected_underdog"),
            pl.col("confidence").alias("selected_probability"),
            pl.col("control_edge_per_share").alias("selected_edge_per_share"),
            (
                (
                    pl.col("won").cast(pl.Float64)
                    - pl.col("selected_execution_cost_per_share")
                )
                * quantity
            ).alias("realized_net"),
            (pl.col("selected_execution_cost_per_share") * quantity).alias(
                "entry_debit"
            ),
        )
    )


def evaluate_policy_grid(
    scored: pl.DataFrame,
    config: AsymmetricValueConfig,
) -> tuple[dict[str, pl.DataFrame], dict[str, dict[str, Any]]]:
    ledgers: dict[str, pl.DataFrame] = {}
    metrics: dict[str, dict[str, Any]] = {}
    models = sorted(scored["model"].unique().to_list())
    for policy in config.policies:
        combined = policy_ledger(
            scored,
            policy,
            quantity=config.quantity,
            maximum_depth_participation=config.maximum_depth_participation,
        )
        for model in models:
            key = candidate_policy_key(model, policy.name)
            ledger = combined.filter(pl.col("model") == model)
            ledgers[key] = ledger
            metrics[key] = ledger_metrics(ledger)
    return ledgers, metrics


def select_policy_candidate(
    metrics: dict[str, dict[str, Any]],
    config: AsymmetricValueConfig,
    *,
    evidence_checks: list[dict[str, Any]] | None = None,
    evidence_checks_by_model: dict[str, list[dict[str, Any]]] | None = None,
    eligible_models: set[str] | None = None,
) -> dict[str, Any]:
    if not metrics:
        raise ValueError("asymmetric-value policy selection requires candidate metrics")
    records: list[dict[str, Any]] = []
    for key, values in metrics.items():
        model, policy = split_candidate_policy_key(key)
        policy_config = next(
            item for item in config.policies if item.name == policy
        )
        checks = [
            *policy_gate_checks(values, config, policy_window=True),
            *((evidence_checks_by_model or {}).get(model, evidence_checks or [])),
        ]
        records.append(
            {
                "key": key,
                "model": model,
                "policy": policy,
                "selection_eligible": bool(
                    policy_config.selection_eligible
                    and (eligible_models is None or model in eligible_models)
                ),
                "qualified": all(check["passed"] for check in checks),
                "checks": checks,
                "metrics": values,
            }
        )
    selectable = [record for record in records if record["selection_eligible"]]
    selected = max(selectable, key=_selection_rank)
    return {
        "selected_key": selected["key"],
        "selected_model": selected["model"],
        "selected_policy": selected["policy"],
        "qualified_on_policy_window": selected["qualified"],
        "selection_objective": (
            "qualified first, then UTC-day bootstrap lower expectancy, net profit per "
            "resolved market, capital efficiency, lower loss-recovery burden, lower "
            "admission cost, net expectancy, profit factor, and trade count"
        ),
        "frontier": records,
    }


def policy_gate_checks(
    metrics: dict[str, Any],
    config: AsymmetricValueConfig,
    *,
    policy_window: bool,
) -> list[dict[str, Any]]:
    gates = config.gates
    required_trades = (
        gates.minimum_policy_trades if policy_window else gates.minimum_evaluation_trades
    )
    values = {
        "minimum_trades": (metrics.get("trades", 0), required_trades, ">="),
        "minimum_trade_utc_days": (
            metrics.get("utc_days", 0),
            (
                gates.minimum_policy_executable_days
                if policy_window
                else gates.minimum_evaluation_executable_days
            ),
            ">=",
        ),
        "minimum_yes_trades": (
            metrics.get("yes_trades", 0),
            gates.minimum_side_trades,
            ">=",
        ),
        "minimum_no_trades": (
            metrics.get("no_trades", 0),
            gates.minimum_side_trades,
            ">=",
        ),
        "minimum_pre60_trades": (
            metrics.get("pre60_trades", 0),
            gates.minimum_pre60_trades,
            ">=",
        ),
        "minimum_20_30c_trades": (
            metrics.get("twenty_to_thirty_cent_trades", 0),
            gates.minimum_20_30c_trades,
            ">=",
        ),
        "minimum_profit_factor": (
            metrics.get("profit_factor"),
            gates.minimum_profit_factor,
            ">=",
        ),
        "minimum_net_expectancy": (
            metrics.get("net_expectancy_per_trade"),
            gates.minimum_net_expectancy_per_trade,
            ">",
        ),
        "minimum_capital_efficiency": (
            metrics.get("capital_efficiency"),
            gates.minimum_capital_efficiency,
            ">",
        ),
        "minimum_stress_expectancy": (
            metrics.get("stress_1c_net_expectancy_per_trade"),
            gates.minimum_stress_expectancy_per_trade,
            ">",
        ),
        "maximum_mean_cost": (
            metrics.get("mean_admission_cost_per_share"),
            gates.maximum_mean_cost_per_share,
            "<=",
        ),
        "maximum_mean_share_price": (
            metrics.get("mean_share_price"),
            gates.maximum_mean_share_price,
            "<=",
        ),
        "maximum_selected_calibration_bias": (
            metrics.get("absolute_selected_calibration_bias"),
            gates.maximum_selected_calibration_bias,
            "<=",
        ),
        "maximum_loss_recovery": (
            metrics.get("loss_recovery_wins"),
            gates.maximum_loss_recovery_wins,
            "<=",
        ),
        "maximum_average_loss": (
            abs(metrics["average_loss"]) if metrics.get("average_loss") is not None else None,
            gates.maximum_average_loss,
            "<=",
        ),
        "maximum_single_loss": (
            abs(metrics["maximum_loss"])
            if metrics.get("maximum_loss") is not None
            else None,
            gates.maximum_single_loss,
            "<=",
        ),
        "positive_yes_net_profit": (
            metrics.get("yes_net_profit"),
            0.0,
            ">",
        ),
        "positive_no_net_profit": (
            metrics.get("no_net_profit"),
            0.0,
            ">",
        ),
    }
    checks: list[dict[str, Any]] = []
    for name, (observed, threshold, operator) in values.items():
        if (
            name == "minimum_profit_factor"
            and metrics.get("profit_factor_no_losses") is True
        ):
            passed = True
        elif observed is None or not np.isfinite(observed):
            passed = False
        elif operator == ">=":
            passed = observed >= threshold
        elif operator == ">":
            passed = observed > threshold
        else:
            passed = observed <= threshold
        checks.append(
            {
                "name": name,
                "observed": observed,
                "threshold": threshold,
                "operator": operator,
                "passed": bool(passed),
            }
        )
    bootstrap = metrics.get("utc_day_block_bootstrap") or {}
    for metric_name in ("net_expectancy_per_trade", "capital_efficiency"):
        lower = (bootstrap.get(metric_name) or {}).get("lower_95")
        checks.append(
            {
                "name": f"positive_bootstrap_lower_{metric_name}",
                "observed": lower,
                "threshold": 0.0,
                "operator": ">",
                "passed": bool(
                    lower is not None and np.isfinite(lower) and lower > 0
                ),
            }
        )
    return checks


def evidence_gate_checks(
    frame: pl.DataFrame,
    config: AsymmetricValueConfig,
    *,
    policy_window: bool,
    source_grid_coverage: float,
    strict_grid_coverage: float,
    candidate_grid_coverage: float,
) -> list[dict[str, Any]]:
    """Keep evidence sufficiency separate from candidate economics."""

    gates = config.gates
    minimum_markets = (
        gates.minimum_policy_strict_markets
        if policy_window
        else gates.minimum_evaluation_strict_markets
    )
    minimum_days = (
        gates.minimum_policy_executable_days
        if policy_window
        else gates.minimum_evaluation_executable_days
    )
    minimum_source_grid_coverage = (
        gates.minimum_policy_source_grid_coverage
        if policy_window
        else gates.minimum_evaluation_source_grid_coverage
    )
    minimum_strict_grid_coverage = (
        gates.minimum_policy_strict_grid_coverage
        if policy_window
        else gates.minimum_evaluation_strict_grid_coverage
    )
    minimum_candidate_grid_coverage = (
        gates.minimum_policy_candidate_grid_coverage
        if policy_window
        else gates.minimum_evaluation_candidate_grid_coverage
    )
    observed = {
        "strict_markets": frame["market_id"].n_unique() if not frame.is_empty() else 0,
        "executable_utc_days": (
            frame["window_start"].dt.date().n_unique() if not frame.is_empty() else 0
        ),
    }
    return [
        {
            "name": "minimum_strict_markets",
            "observed": observed["strict_markets"],
            "threshold": minimum_markets,
            "operator": ">=",
            "passed": observed["strict_markets"] >= minimum_markets,
        },
        {
            "name": "minimum_executable_utc_days",
            "observed": observed["executable_utc_days"],
            "threshold": minimum_days,
            "operator": ">=",
            "passed": observed["executable_utc_days"] >= minimum_days,
        },
        {
            "name": "minimum_source_grid_coverage",
            "observed": source_grid_coverage,
            "threshold": minimum_source_grid_coverage,
            "operator": ">=",
            "passed": source_grid_coverage >= minimum_source_grid_coverage,
        },
        {
            "name": "minimum_strict_grid_coverage",
            "observed": strict_grid_coverage,
            "threshold": minimum_strict_grid_coverage,
            "operator": ">=",
            "passed": strict_grid_coverage >= minimum_strict_grid_coverage,
        },
        {
            "name": "minimum_candidate_prediction_grid_coverage",
            "observed": candidate_grid_coverage,
            "threshold": minimum_candidate_grid_coverage,
            "operator": ">=",
            "passed": candidate_grid_coverage >= minimum_candidate_grid_coverage,
        },
    ]


def ledger_metrics(ledger: pl.DataFrame) -> dict[str, Any]:
    if ledger.is_empty():
        return {
            "trades": 0,
            "pre60_trades": 0,
            "twenty_to_thirty_cent_trades": 0,
            "accuracy": None,
            "net_profit": 0.0,
            "net_expectancy_per_trade": None,
            "profit_factor": None,
            "capital_efficiency": None,
            "mean_admission_cost_per_share": None,
            "mean_share_price": None,
            "absolute_selected_calibration_bias": None,
            "stress_1c_net_expectancy_per_trade": None,
            "loss_recovery_wins": None,
            "average_loss": None,
            "losing_trades": 0,
            "profit_factor_no_losses": False,
        }
    profits = ledger["realized_net"].to_numpy()
    wins = profits[profits > 0]
    losses = profits[profits < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    average_win = float(wins.mean()) if len(wins) else None
    average_loss = float(losses.mean()) if len(losses) else 0.0
    chronological = ledger.sort("window_start", "observed_at")["realized_net"].to_list()
    equity = 0.0
    peak = 0.0
    maximum_drawdown = 0.0
    current_losing_streak = 0
    maximum_losing_streak = 0
    for value in chronological:
        equity += value
        peak = max(peak, equity)
        maximum_drawdown = min(maximum_drawdown, equity - peak)
        if value < 0:
            current_losing_streak += 1
            maximum_losing_streak = max(maximum_losing_streak, current_losing_streak)
        else:
            current_losing_streak = 0
    total_debit = float(ledger["entry_debit"].sum())
    net_profit = float(profits.sum())
    quantity = ledger["quantity"].to_numpy()
    stressed_cost = np.minimum(
        ledger["selected_execution_cost_per_share"].to_numpy()
        + EXECUTION_STRESS_PER_SHARE,
        1.0,
    )
    stressed_profits = quantity * (
        ledger["won"].to_numpy().astype(np.float64) - stressed_cost
    )
    stressed_debit = float(np.sum(quantity * stressed_cost))
    stressed_net = float(stressed_profits.sum())
    stressed_wins = stressed_profits[stressed_profits > 0]
    stressed_losses = stressed_profits[stressed_profits < 0]
    selected_probability = ledger["selected_probability"].to_numpy()
    won = ledger["won"].to_numpy().astype(np.float64)
    clipped_probability = np.clip(selected_probability, 1e-9, 1.0 - 1e-9)
    calibration_bias = float(np.mean(selected_probability - won))
    selected_yes = ledger["selected_yes"]
    pre60_trades = ledger.filter(pl.col("seconds_elapsed") < 60).height
    twenty_to_thirty = ledger.filter(
        pl.col("selected_share_price").is_between(
            0.20,
            0.30,
            closed="left",
        )
    ).height
    return {
        "trades": ledger.height,
        "markets": ledger["market_id"].n_unique(),
        "utc_days": ledger["window_start"].dt.date().n_unique(),
        "yes_trades": int(selected_yes.sum()),
        "no_trades": int((~selected_yes).sum()),
        "pre60_trades": pre60_trades,
        "pre60_trade_share": pre60_trades / ledger.height,
        "twenty_to_thirty_cent_trades": twenty_to_thirty,
        "twenty_to_thirty_cent_trade_share": twenty_to_thirty / ledger.height,
        "underdog_trades": int(ledger["selected_underdog"].sum()),
        "underdog_trade_share": float(ledger["selected_underdog"].mean()),
        "accuracy": float(ledger["won"].mean()),
        "net_profit": net_profit,
        "net_expectancy_per_trade": float(profits.mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "profit_factor_no_losses": bool(len(wins) and not len(losses)),
        "losing_trades": len(losses),
        "entry_debit": total_debit,
        "capital_efficiency": net_profit / total_debit if total_debit > 0 else None,
        "mean_admission_cost_per_share": float(
            ledger["selected_admission_cost_per_share"].mean()
        ),
        "mean_share_price": float(ledger["selected_share_price"].mean()),
        "median_admission_cost_per_share": float(
            ledger["selected_admission_cost_per_share"].median()
        ),
        "mean_execution_cost_per_share": float(
            ledger["selected_execution_cost_per_share"].mean()
        ),
        "mean_selected_probability": float(ledger["selected_probability"].mean()),
        "selected_calibration_bias": calibration_bias,
        "absolute_selected_calibration_bias": abs(calibration_bias),
        "selected_brier_score": float(np.mean(np.square(selected_probability - won))),
        "selected_log_loss": float(
            -np.mean(
                won * np.log(clipped_probability)
                + (1.0 - won) * np.log(1.0 - clipped_probability)
            )
        ),
        "mean_modeled_edge_per_share": float(ledger["selected_edge_per_share"].mean()),
        "mean_entry_second": float(ledger["seconds_elapsed"].mean()),
        "average_win": average_win,
        "average_loss": average_loss,
        "loss_recovery_wins": (
            abs(average_loss) / average_win
            if average_win is not None and average_win > 0
            else 0.0 if not len(losses) else None
        ),
        "maximum_loss": min(float(profits.min()), 0.0),
        "maximum_drawdown": maximum_drawdown,
        "maximum_losing_streak": maximum_losing_streak,
        "stress_1c_net_profit": stressed_net,
        "stress_1c_net_expectancy_per_trade": float(stressed_profits.mean()),
        "stress_1c_capital_efficiency": (
            stressed_net / stressed_debit if stressed_debit > 0 else None
        ),
        "stress_1c_profit_factor": (
            float(stressed_wins.sum()) / float(-stressed_losses.sum())
            if len(stressed_losses)
            else None
        ),
        "yes_net_profit": float(ledger.filter(pl.col("selected_yes"))["realized_net"].sum()),
        "no_net_profit": float(ledger.filter(~pl.col("selected_yes"))["realized_net"].sum()),
    }


def bootstrap_ledger_metrics(
    ledger: pl.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float]] | None:
    if ledger.is_empty():
        return None
    daily = (
        ledger.with_columns(pl.col("window_start").dt.date().alias("date"))
        .group_by("date")
        .agg(
            pl.col("realized_net").sum().alias("net"),
            pl.col("entry_debit").sum().alias("debit"),
            pl.len().alias("trades"),
        )
        .sort("date")
    )
    net = daily["net"].to_numpy()
    debit = daily["debit"].to_numpy()
    trades = daily["trades"].to_numpy()
    rng = np.random.default_rng(seed)
    expectancy = np.empty(resamples, dtype=np.float64)
    capital_efficiency = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, len(net), len(net))
        expectancy[index] = net[chosen].sum() / max(trades[chosen].sum(), 1)
        capital_efficiency[index] = net[chosen].sum() / max(debit[chosen].sum(), 1e-12)
    return {
        "utc_day_blocks": int(daily.height),
        "net_expectancy_per_trade": _interval(expectancy),
        "capital_efficiency": _interval(capital_efficiency),
    }


def accuracy_price_by_second(scored: pl.DataFrame) -> list[dict[str, Any]]:
    """Report prediction quality and executable economics at every model decision."""

    if scored.is_empty():
        return []
    group_columns = ["seconds_elapsed"]
    sort_columns = ["seconds_elapsed"]
    if "model" in scored.columns:
        group_columns.insert(0, "model")
        sort_columns.insert(0, "model")
    label = pl.col("label_up").cast(pl.Float64)
    probability = pl.col("probability_yes")
    clipped_probability = probability.clip(1e-15, 1.0 - 1e-15)
    return (
        scored.with_columns(
            (pl.col("argmax_yes") == (pl.col("label_up") == 1)).alias(
                "argmax_correct"
            ),
            (pl.col("selected_edge_per_share") > 0).alias("positive_value"),
            ((probability - label) ** 2).alias("probability_yes_brier"),
            (
                -(
                    label * clipped_probability.log()
                    + (1.0 - label) * (1.0 - clipped_probability).log()
                )
            ).alias("probability_yes_log_loss"),
            (probability - label).alias("probability_yes_calibration_error"),
        )
        .group_by(group_columns)
        .agg(
            pl.len().alias("rows"),
            pl.col("market_id").n_unique().alias("markets"),
            pl.col("window_start").dt.date().n_unique().alias("utc_days"),
            pl.col("argmax_correct").mean().alias("accuracy"),
            pl.col("argmax_correct").mean().alias("argmax_accuracy"),
            pl.col("probability_yes_brier").mean().alias("brier_score"),
            pl.col("probability_yes_log_loss").mean().alias("log_loss"),
            pl.col("probability_yes_calibration_error")
            .mean()
            .alias("calibration_bias"),
            pl.col("won").mean().alias("value_side_accuracy"),
            pl.col("probability_yes").mean().alias("mean_probability_yes"),
            pl.col("label_up").mean().alias("actual_yes_rate"),
            pl.col("selected_probability").mean().alias(
                "mean_selected_probability"
            ),
            pl.col("selected_admission_cost_per_share").mean().alias(
                "mean_selected_cost_per_share"
            ),
            pl.col("selected_share_price").mean().alias(
                "mean_selected_share_price"
            ),
            pl.col("selected_edge_per_share").mean().alias(
                "mean_selected_edge_per_share"
            ),
            pl.col("positive_value").mean().alias("positive_value_share"),
            pl.col("selected_underdog").mean().alias(
                "selected_underdog_share"
            ),
            pl.col("yes_best_ask").mean().alias("mean_yes_best_ask"),
            pl.col("yes_ask_vwap_5").mean().alias("mean_yes_vwap_5"),
            pl.col("yes_cost_per_share")
            .mean()
            .alias("mean_yes_all_in_cost_per_share"),
            pl.col("no_best_ask").mean().alias("mean_no_best_ask"),
            pl.col("no_ask_vwap_5").mean().alias("mean_no_vwap_5"),
            pl.col("no_cost_per_share")
            .mean()
            .alias("mean_no_all_in_cost_per_share"),
        )
        .with_columns(pl.col("calibration_bias").abs().alias("absolute_calibration_bias"))
        .sort(sort_columns)
        .to_dicts()
    )


def opportunity_calibration_by_price_band(
    scored: pl.DataFrame,
) -> list[dict[str, Any]]:
    """Measure whether selected-side probability supports each executable cost band."""

    if scored.is_empty():
        return []
    banded = _with_price_band(scored)
    group_columns = ["price_band"]
    sort_columns = ["price_band"]
    if "model" in banded.columns:
        group_columns.insert(0, "model")
        sort_columns.insert(0, "model")
    return (
        banded.group_by(group_columns)
        .agg(
            pl.len().alias("rows"),
            pl.col("market_id").n_unique().alias("markets"),
            pl.col("won").mean().alias("actual_win_rate"),
            pl.col("selected_probability").mean().alias(
                "mean_selected_probability"
            ),
            pl.col("selected_admission_cost_per_share").mean().alias(
                "mean_admission_cost_per_share"
            ),
            pl.col("selected_share_price").mean().alias("mean_share_price"),
            pl.col("selected_edge_per_share").mean().alias(
                "mean_modeled_edge_per_share"
            ),
            (
                pl.col("won").cast(pl.Float64)
                - pl.col("selected_execution_cost_per_share")
            )
            .mean()
            .alias("realized_net_per_share_if_every_point"),
            pl.col("selected_underdog").mean().alias("underdog_share"),
        )
        .sort(sort_columns)
        .to_dicts()
    )


def joint_accuracy_value_surface(scored: pl.DataFrame) -> list[dict[str, Any]]:
    """Joint model/time/side/raw-price cells for the user's interval hypothesis."""

    if scored.is_empty():
        return []
    rows = (
        _with_price_band(
            scored.with_columns(
                pl.when(pl.col("selected_yes"))
                .then(pl.lit("YES"))
                .otherwise(pl.lit("NO"))
                .alias("selected_side")
            )
        )
        .group_by("model", "seconds_elapsed", "selected_side", "price_band")
        .agg(
            pl.len().alias("rows"),
            pl.col("market_id").n_unique().alias("markets"),
            pl.col("won").sum().alias("wins"),
            pl.col("won").mean().alias("actual_win_rate"),
            pl.col("selected_probability").mean().alias(
                "mean_selected_probability"
            ),
            pl.col("selected_share_price").mean().alias("mean_share_price"),
            pl.col("selected_admission_cost_per_share").mean().alias(
                "mean_admission_cost_per_share"
            ),
            pl.col("selected_edge_per_share").mean().alias(
                "mean_modeled_edge_per_share"
            ),
            (
                pl.col("won").cast(pl.Float64)
                - pl.col("selected_execution_cost_per_share")
            )
            .mean()
            .alias("realized_net_per_share_if_every_point"),
        )
        .sort("model", "seconds_elapsed", "selected_side", "price_band")
        .to_dicts()
    )
    for row in rows:
        lower, upper = _wilson_interval(int(row["wins"]), int(row["rows"]))
        row["win_rate_lower_95"] = lower
        row["win_rate_upper_95"] = upper
        row["selected_calibration_bias"] = (
            float(row["mean_selected_probability"])
            - float(row["actual_win_rate"])
        )
    return rows


def side_accuracy_value_surface(predictions: pl.DataFrame) -> list[dict[str, Any]]:
    """Expose YES and NO probability, raw price, and realized value in every cell."""

    if predictions.is_empty():
        return []
    identity = [
        "model",
        "market_id",
        "window_start",
        "seconds_elapsed",
        "label_up",
    ]
    yes = predictions.select(
        *identity,
        pl.lit("YES").alias("side"),
        pl.col("probability_yes").alias("side_probability"),
        pl.col("yes_ask_vwap_5").alias("selected_share_price"),
        pl.col("yes_cost_per_share").alias("side_admission_cost_per_share"),
        pl.col("yes_execution_cost_per_share").alias(
            "side_execution_cost_per_share"
        ),
        (pl.col("label_up") == 1).alias("won"),
    )
    no = predictions.select(
        *identity,
        pl.lit("NO").alias("side"),
        (1.0 - pl.col("probability_yes")).alias("side_probability"),
        pl.col("no_ask_vwap_5").alias("selected_share_price"),
        pl.col("no_cost_per_share").alias("side_admission_cost_per_share"),
        pl.col("no_execution_cost_per_share").alias(
            "side_execution_cost_per_share"
        ),
        (pl.col("label_up") == 0).alias("won"),
    )
    rows = (
        _with_price_band(pl.concat((yes, no), how="vertical"))
        .with_columns(
            (
                pl.col("side_probability")
                - pl.col("side_admission_cost_per_share")
            ).alias("modeled_edge_per_share")
        )
        .group_by("model", "seconds_elapsed", "side", "price_band")
        .agg(
            pl.len().alias("rows"),
            pl.col("market_id").n_unique().alias("markets"),
            pl.col("won").sum().alias("wins"),
            pl.col("won").mean().alias("actual_win_rate"),
            pl.col("side_probability").mean().alias("mean_side_probability"),
            pl.col("selected_share_price").mean().alias("mean_share_price"),
            pl.col("side_admission_cost_per_share")
            .mean()
            .alias("mean_admission_cost_per_share"),
            pl.col("modeled_edge_per_share")
            .mean()
            .alias("mean_modeled_edge_per_share"),
            (
                pl.col("won").cast(pl.Float64)
                - pl.col("side_execution_cost_per_share")
            )
            .mean()
            .alias("realized_net_per_share_if_every_point"),
        )
        .sort("model", "seconds_elapsed", "side", "price_band")
        .to_dicts()
    )
    for row in rows:
        lower, upper = _wilson_interval(int(row["wins"]), int(row["rows"]))
        row["win_rate_lower_95"] = lower
        row["win_rate_upper_95"] = upper
        row["side_calibration_bias"] = (
            float(row["mean_side_probability"]) - float(row["actual_win_rate"])
        )
    return rows


def price_band_metrics(ledger: pl.DataFrame) -> list[dict[str, Any]]:
    if ledger.is_empty():
        return []
    banded = _with_price_band(ledger)
    return (
        banded.group_by("price_band")
        .agg(
            pl.len().alias("trades"),
            pl.col("won").mean().alias("accuracy"),
            pl.col("selected_admission_cost_per_share").mean().alias("mean_cost_per_share"),
            pl.col("selected_share_price").mean().alias("mean_share_price"),
            pl.col("realized_net").sum().alias("net_profit"),
            pl.col("realized_net").mean().alias("net_expectancy_per_trade"),
            pl.col("selected_underdog").mean().alias("underdog_trade_share"),
        )
        .sort("price_band")
        .to_dicts()
    )


def _with_price_band(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.col("selected_share_price") < 0.10)
        .then(pl.lit("00_10c"))
        .when(pl.col("selected_share_price") < 0.20)
        .then(pl.lit("10_20c"))
        .when(pl.col("selected_share_price") < 0.30)
        .then(pl.lit("20_30c"))
        .when(pl.col("selected_share_price") < 0.40)
        .then(pl.lit("30_40c"))
        .when(pl.col("selected_share_price") < 0.50)
        .then(pl.lit("40_50c"))
        .when(pl.col("selected_share_price") < 0.60)
        .then(pl.lit("50_60c"))
        .when(pl.col("selected_share_price") < 0.70)
        .then(pl.lit("60_70c"))
        .when(pl.col("selected_share_price") < 0.80)
        .then(pl.lit("70_80c"))
        .when(pl.col("selected_share_price") < 0.90)
        .then(pl.lit("80_90c"))
        .otherwise(pl.lit("90_100c"))
        .alias("price_band")
    )


def candidate_policy_key(model: str, policy: str) -> str:
    return f"{model}::{policy}"


def split_candidate_policy_key(key: str) -> tuple[str, str]:
    model, separator, policy = key.partition("::")
    if not separator or not model or not policy:
        raise ValueError(f"invalid asymmetric-value candidate-policy key: {key}")
    return model, policy


def add_bootstrap_metrics(
    metrics: dict[str, dict[str, Any]],
    ledgers: dict[str, pl.DataFrame],
    *,
    resamples: int,
    seed: int,
) -> None:
    for offset, key in enumerate(sorted(metrics)):
        metrics[key]["utc_day_block_bootstrap"] = bootstrap_ledger_metrics(
            ledgers[key],
            resamples=resamples,
            seed=seed + offset,
        )


def _selection_rank(record: dict[str, Any]) -> tuple[Any, ...]:
    metrics = record["metrics"]
    bootstrap = metrics.get("utc_day_block_bootstrap") or {}
    lower = (bootstrap.get("net_expectancy_per_trade") or {}).get("lower_95")
    return (
        record["qualified"],
        lower if lower is not None else float("-inf"),
        metrics.get("net_profit_per_resolved_market")
        if metrics.get("net_profit_per_resolved_market") is not None
        else float("-inf"),
        metrics.get("capital_efficiency")
        if metrics.get("capital_efficiency") is not None
        else float("-inf"),
        -(
            metrics.get("loss_recovery_wins")
            if metrics.get("loss_recovery_wins") is not None
            else float("inf")
        ),
        -(
            metrics.get("mean_admission_cost_per_share")
            if metrics.get("mean_admission_cost_per_share") is not None
            else float("inf")
        ),
        metrics.get("net_expectancy_per_trade")
        if metrics.get("net_expectancy_per_trade") is not None
        else float("-inf"),
        metrics.get("profit_factor")
        if metrics.get("profit_factor") is not None
        else float("-inf"),
        metrics.get("trades", 0),
        -len(record["model"]),
    )


def _interval(values: np.ndarray) -> dict[str, float]:
    return {
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _wilson_interval(successes: int, observations: int) -> tuple[float, float]:
    if observations <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / observations
    denominator = 1.0 + z * z / observations
    center = (proportion + z * z / (2.0 * observations)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / observations
            + z * z / (4.0 * observations * observations)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
