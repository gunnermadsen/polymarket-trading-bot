"""Economic policy selection for lower-cost asymmetric-value predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .asymmetric_value_config import AsymmetricValueConfig, ValuePolicy

EXECUTION_STRESS_PER_SHARE = 0.01
IMMEDIATE_FIRST_CROSSING = "immediate_first_crossing"
EDGE_POSITIVE_2_OF_LAST_3_SECONDS = "edge_positive_2_of_last_3_seconds"
MINIMUM_SELECTED_WIN_RATE_ADVANTAGE = 0.05

_POLICY_REQUIRED_COLUMNS = {
    "model",
    "market_id",
    "window_start",
    "observed_at",
    "seconds_elapsed",
    "label_up",
    "probability_yes",
    "argmax_yes",
    "yes_ask_vwap_5",
    "no_ask_vwap_5",
    "yes_ask_depth",
    "no_ask_depth",
    "yes_cost_per_share",
    "no_cost_per_share",
    "yes_execution_cost_per_share",
    "no_execution_cost_per_share",
    "yes_edge_per_share",
    "no_edge_per_share",
}


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
    staged = _with_policy_stage_flags(
        scored,
        policy,
        quantity=quantity,
        maximum_depth_participation=maximum_depth_participation,
    )
    eligible = _select_policy_side(staged, "_yes_edge_eligible", "_no_edge_eligible")
    return _first_policy_entries(eligible, policy.name, quantity)


def temporal_confirmation_policy_ledger(
    scored: pl.DataFrame,
    policy: ValuePolicy,
    *,
    quantity: float,
    maximum_depth_participation: float,
) -> pl.DataFrame:
    """Apply the fixed same-side two-of-three-second confirmation rule.

    A side must satisfy every current policy condition at the entry second and in at
    least one of the exact decision seconds ``t-1`` or ``t-2``. Prior rows only count
    when their observation timestamp precedes the candidate entry, preventing future
    or merely row-adjacent observations from leaking into the confirmation.
    """

    staged = _with_policy_stage_flags(
        scored,
        policy,
        quantity=quantity,
        maximum_depth_participation=maximum_depth_participation,
    )
    confirmed = _with_temporal_confirmation_flags(staged)
    eligible = _select_policy_side(
        confirmed,
        "_yes_temporally_confirmed",
        "_no_temporally_confirmed",
    )
    return _first_policy_entries(
        eligible,
        f"{policy.name}::{EDGE_POSITIVE_2_OF_LAST_3_SECONDS}",
        quantity,
    )


def temporal_confirmation_ablation(
    scored: pl.DataFrame,
    policy: ValuePolicy,
    *,
    quantity: float,
    maximum_depth_participation: float,
) -> tuple[dict[str, pl.DataFrame], dict[str, dict[str, Any]]]:
    """Evaluate fixed immediate and causal confirmation rules on one scored frame."""

    ledgers = {
        IMMEDIATE_FIRST_CROSSING: policy_ledger(
            scored,
            policy,
            quantity=quantity,
            maximum_depth_participation=maximum_depth_participation,
        ),
        EDGE_POSITIVE_2_OF_LAST_3_SECONDS: temporal_confirmation_policy_ledger(
            scored,
            policy,
            quantity=quantity,
            maximum_depth_participation=maximum_depth_participation,
        ),
    }
    return ledgers, {name: ledger_metrics(ledger) for name, ledger in ledgers.items()}


def _with_policy_stage_flags(
    scored: pl.DataFrame,
    policy: ValuePolicy,
    *,
    quantity: float,
    maximum_depth_participation: float,
) -> pl.DataFrame:
    _require_columns(scored, _POLICY_REQUIRED_COLUMNS, "asymmetric policy scoring")
    if quantity <= 0.0 or maximum_depth_participation <= 0.0:
        raise ValueError("policy quantity and depth participation must be positive")
    within_time = pl.col("seconds_elapsed") <= policy.maximum_entry_second
    yes_time_raw = (
        within_time
        & pl.col("yes_ask_vwap_5").is_between(
            policy.minimum_share_price,
            policy.maximum_share_price,
            closed="left",
        )
        & (pl.col("yes_cost_per_share") <= policy.maximum_cost_per_share)
    )
    no_time_raw = (
        within_time
        & pl.col("no_ask_vwap_5").is_between(
            policy.minimum_share_price,
            policy.maximum_share_price,
            closed="left",
        )
        & (pl.col("no_cost_per_share") <= policy.maximum_cost_per_share)
    )
    staged = scored.with_columns(
        yes_time_raw.fill_null(False).alias("_yes_time_raw_eligible"),
        no_time_raw.fill_null(False).alias("_no_time_raw_eligible"),
    ).with_columns(
        (
            pl.col("_yes_time_raw_eligible")
            & (
                quantity
                <= pl.col("yes_ask_depth") * maximum_depth_participation
            )
        )
        .fill_null(False)
        .alias("_yes_depth_eligible"),
        (
            pl.col("_no_time_raw_eligible")
            & (
                quantity
                <= pl.col("no_ask_depth") * maximum_depth_participation
            )
        )
        .fill_null(False)
        .alias("_no_depth_eligible"),
    )
    return staged.with_columns(
        (
            pl.col("_yes_depth_eligible")
            & (pl.col("yes_edge_per_share") >= policy.minimum_edge_per_share)
        )
        .fill_null(False)
        .alias("_yes_edge_eligible"),
        (
            pl.col("_no_depth_eligible")
            & (pl.col("no_edge_per_share") >= policy.minimum_edge_per_share)
        )
        .fill_null(False)
        .alias("_no_edge_eligible"),
    )


def _with_temporal_confirmation_flags(staged: pl.DataFrame) -> pl.DataFrame:
    keys = ["model", "market_id", "window_start", "seconds_elapsed"]
    if staged.select(*keys).is_duplicated().any():
        raise ValueError(
            "temporal confirmation requires one frozen prediction per model/market/second"
        )

    confirmed = staged
    for lag in (1, 2):
        prior = staged.select(
            *keys,
            pl.col("observed_at").alias(f"_prior_{lag}_observed_at"),
            pl.col("_yes_edge_eligible").alias(f"_prior_{lag}_yes_eligible"),
            pl.col("_no_edge_eligible").alias(f"_prior_{lag}_no_eligible"),
        ).with_columns((pl.col("seconds_elapsed") + lag).alias("seconds_elapsed"))
        confirmed = confirmed.join(
            prior,
            on=keys,
            how="left",
            validate="1:1",
        ).with_columns(
            (
                pl.col(f"_prior_{lag}_yes_eligible").fill_null(False)
                & (
                    pl.col(f"_prior_{lag}_observed_at").is_not_null()
                    & (pl.col(f"_prior_{lag}_observed_at") < pl.col("observed_at"))
                )
            ).alias(f"_causal_prior_{lag}_yes"),
            (
                pl.col(f"_prior_{lag}_no_eligible").fill_null(False)
                & (
                    pl.col(f"_prior_{lag}_observed_at").is_not_null()
                    & (pl.col(f"_prior_{lag}_observed_at") < pl.col("observed_at"))
                )
            ).alias(f"_causal_prior_{lag}_no"),
        )

    return confirmed.with_columns(
        (
            pl.col("_yes_edge_eligible")
            & (
                pl.col("_yes_edge_eligible").cast(pl.Int8)
                + pl.col("_causal_prior_1_yes").cast(pl.Int8)
                + pl.col("_causal_prior_2_yes").cast(pl.Int8)
                >= 2
            )
        ).alias("_yes_temporally_confirmed"),
        (
            pl.col("_no_edge_eligible")
            & (
                pl.col("_no_edge_eligible").cast(pl.Int8)
                + pl.col("_causal_prior_1_no").cast(pl.Int8)
                + pl.col("_causal_prior_2_no").cast(pl.Int8)
                >= 2
            )
        ).alias("_no_temporally_confirmed"),
    )


def _select_policy_side(
    staged: pl.DataFrame,
    yes_eligible_column: str,
    no_eligible_column: str,
) -> pl.DataFrame:
    eligible = staged.filter(
        pl.col(yes_eligible_column) | pl.col(no_eligible_column)
    ).with_columns(
        pl.when(pl.col(yes_eligible_column) & pl.col(no_eligible_column))
        .then(pl.col("yes_edge_per_share") >= pl.col("no_edge_per_share"))
        .otherwise(pl.col(yes_eligible_column))
        .alias("selected_yes")
    )
    internal = [
        column
        for column in eligible.columns
        if column.startswith(
            (
                "_yes_",
                "_no_",
                "_prior_",
                "_causal_prior_",
            )
        )
    ]
    return eligible.with_columns(
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
        (pl.col("selected_yes") != pl.col("argmax_yes")).alias("selected_underdog"),
    ).drop(*internal)


def _first_policy_entries(
    eligible: pl.DataFrame,
    policy_name: str,
    quantity: float,
) -> pl.DataFrame:
    return (
        eligible.sort("model", "market_id", "seconds_elapsed", "observed_at")
        .group_by("model", "market_id", maintain_order=True)
        .first()
        .with_columns(
            pl.lit(policy_name).alias("policy"),
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


def rejection_funnel(
    scored: pl.DataFrame,
    policy: ValuePolicy,
    *,
    quantity: float,
    maximum_depth_participation: float,
    confirmation_rule: str = EDGE_POSITIVE_2_OF_LAST_3_SECONDS,
) -> dict[str, Any]:
    """Count cumulative decision eligibility without double-counting aggregate markets.

    The input is contractually the already-fresh, strict execution cohort. The helper
    reports that boundary explicitly; it does not infer freshness from quote columns.
    YES and NO support may overlap, while aggregate rows and markets are always the
    union of the two side masks.
    """

    if confirmation_rule not in {
        IMMEDIATE_FIRST_CROSSING,
        EDGE_POSITIVE_2_OF_LAST_3_SECONDS,
    }:
        raise ValueError(f"unsupported temporal confirmation rule: {confirmation_rule}")
    staged = _with_policy_stage_flags(
        scored,
        policy,
        quantity=quantity,
        maximum_depth_participation=maximum_depth_participation,
    ).with_columns(
        pl.lit(True).alias("_yes_source_candidate"),
        pl.lit(True).alias("_no_source_candidate"),
        pl.lit(True).alias("_yes_fresh_strict"),
        pl.lit(True).alias("_no_fresh_strict"),
    )
    models = staged["model"].unique().to_list()
    if len(models) > 1:
        raise ValueError("rejection funnel requires one model at a time")

    if confirmation_rule == EDGE_POSITIVE_2_OF_LAST_3_SECONDS:
        staged = _with_temporal_confirmation_flags(staged)
        yes_temporal = "_yes_temporally_confirmed"
        no_temporal = "_no_temporally_confirmed"
    else:
        yes_temporal = "_yes_edge_eligible"
        no_temporal = "_no_edge_eligible"

    eligible = _select_policy_side(staged, yes_temporal, no_temporal)
    selected = _first_policy_entries(eligible, policy.name, quantity)
    stages = [
        _funnel_stage_counts(
            staged,
            "source_candidate_rows",
            "_yes_source_candidate",
            "_no_source_candidate",
        ),
        _funnel_stage_counts(
            staged,
            "fresh_strict_execution",
            "_yes_fresh_strict",
            "_no_fresh_strict",
        ),
        _funnel_stage_counts(
            staged,
            "time_raw_price",
            "_yes_time_raw_eligible",
            "_no_time_raw_eligible",
        ),
        _funnel_stage_counts(
            staged,
            "depth",
            "_yes_depth_eligible",
            "_no_depth_eligible",
        ),
        _funnel_stage_counts(
            staged,
            "modeled_edge",
            "_yes_edge_eligible",
            "_no_edge_eligible",
        ),
        _funnel_stage_counts(
            staged,
            "temporal_confirmation",
            yes_temporal,
            no_temporal,
        ),
        _selected_funnel_counts(selected),
    ]
    return {
        "model": models[0] if models else None,
        "confirmation_rule": confirmation_rule,
        "fresh_strict_execution_input": True,
        "fresh_strict_execution_contract": (
            "input frame is already filtered to fresh, strict execution evidence"
        ),
        "time_raw_price_includes_all_in_cost_cap": True,
        "aggregate_market_counting": "unique union across YES and NO; never side-count sum",
        "stages": stages,
    }


def _funnel_stage_counts(
    frame: pl.DataFrame,
    stage: str,
    yes_column: str,
    no_column: str,
) -> dict[str, Any]:
    yes = frame.filter(pl.col(yes_column))
    no = frame.filter(pl.col(no_column))
    aggregate = frame.filter(pl.col(yes_column) | pl.col(no_column))
    return {
        "stage": stage,
        "aggregate_rows": aggregate.height,
        "aggregate_markets": aggregate["market_id"].n_unique(),
        "yes_rows": yes.height,
        "yes_markets": yes["market_id"].n_unique(),
        "no_rows": no.height,
        "no_markets": no["market_id"].n_unique(),
    }


def _selected_funnel_counts(selected: pl.DataFrame) -> dict[str, Any]:
    yes = selected.filter(pl.col("selected_yes"))
    no = selected.filter(~pl.col("selected_yes"))
    return {
        "stage": "selected_one_trade_per_market",
        "aggregate_rows": selected.height,
        "aggregate_markets": selected["market_id"].n_unique(),
        "yes_rows": yes.height,
        "yes_markets": yes["market_id"].n_unique(),
        "no_rows": no.height,
        "no_markets": no["market_id"].n_unique(),
    }


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
            "selected_win_rate": None,
            "net_profit": 0.0,
            "net_expectancy_per_trade": None,
            "profit_factor": None,
            "capital_efficiency": None,
            "mean_admission_cost_per_share": None,
            "conservative_all_in_break_even_probability": None,
            "selected_win_rate_advantage": None,
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
    conservative_break_even = float(
        ledger["selected_admission_cost_per_share"].mean()
    )
    actual_win_rate = float(ledger["won"].mean())
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
        "accuracy": actual_win_rate,
        "selected_win_rate": actual_win_rate,
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
        "conservative_all_in_break_even_probability": conservative_break_even,
        "selected_win_rate_advantage": actual_win_rate - conservative_break_even,
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
) -> dict[str, Any] | None:
    if ledger.is_empty():
        return None
    daily = (
        ledger.with_columns(pl.col("window_start").dt.date().alias("date"))
        .group_by("date")
        .agg(
            pl.col("realized_net").sum().alias("net"),
            pl.col("entry_debit").sum().alias("debit"),
            pl.col("won").cast(pl.Float64).sum().alias("wins"),
            pl.col("selected_admission_cost_per_share").sum().alias(
                "admission_cost"
            ),
            pl.len().alias("trades"),
        )
        .sort("date")
    )
    net = daily["net"].to_numpy()
    debit = daily["debit"].to_numpy()
    wins = daily["wins"].to_numpy()
    admission_cost = daily["admission_cost"].to_numpy()
    trades = daily["trades"].to_numpy()
    rng = np.random.default_rng(seed)
    expectancy = np.empty(resamples, dtype=np.float64)
    capital_efficiency = np.empty(resamples, dtype=np.float64)
    selected_win_rate_advantage = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, len(net), len(net))
        expectancy[index] = net[chosen].sum() / max(trades[chosen].sum(), 1)
        capital_efficiency[index] = net[chosen].sum() / max(debit[chosen].sum(), 1e-12)
        selected_win_rate_advantage[index] = (
            wins[chosen].sum() - admission_cost[chosen].sum()
        ) / max(trades[chosen].sum(), 1)
    return {
        "utc_day_blocks": int(daily.height),
        "net_expectancy_per_trade": _interval(expectancy),
        "capital_efficiency": _interval(capital_efficiency),
        "selected_win_rate_advantage": _interval(selected_win_rate_advantage),
    }


def selected_win_rate_advantage_gate_checks(
    metrics: dict[str, Any],
    *,
    minimum_point_advantage: float = MINIMUM_SELECTED_WIN_RATE_ADVANTAGE,
) -> list[dict[str, Any]]:
    """Require both a five-point advantage and a positive UTC-day lower bound."""

    if not np.isfinite(minimum_point_advantage) or minimum_point_advantage < 0.0:
        raise ValueError("minimum selected win-rate advantage must be finite and nonnegative")
    point = metrics.get("selected_win_rate_advantage")
    bootstrap = metrics.get("utc_day_block_bootstrap") or {}
    lower = (bootstrap.get("selected_win_rate_advantage") or {}).get("lower_95")
    return [
        {
            "name": "minimum_selected_win_rate_advantage",
            "observed": point,
            "threshold": minimum_point_advantage,
            "operator": ">=",
            "passed": bool(point is not None and np.isfinite(point) and point >= minimum_point_advantage),
        },
        {
            "name": "positive_bootstrap_lower_selected_win_rate_advantage",
            "observed": lower,
            "threshold": 0.0,
            "operator": ">",
            "passed": bool(lower is not None and np.isfinite(lower) and lower > 0.0),
        },
    ]


def frequency_floor_check(
    *,
    candidate_trades: int,
    eligible_resolved_markets: int,
    incumbent_trades_per_eligible_resolved_market: float,
    minimum_incumbent_fraction: float = 0.80,
) -> dict[str, Any]:
    """Compare candidate market-level frequency with a supplied incumbent rate."""

    if eligible_resolved_markets <= 0:
        raise ValueError("frequency floor requires at least one eligible resolved market")
    if candidate_trades < 0 or candidate_trades > eligible_resolved_markets:
        raise ValueError("candidate trades must be inside the eligible resolved market count")
    if (
        not np.isfinite(incumbent_trades_per_eligible_resolved_market)
        or incumbent_trades_per_eligible_resolved_market <= 0.0
    ):
        raise ValueError("incumbent trade frequency must be finite and positive")
    if (
        not np.isfinite(minimum_incumbent_fraction)
        or not 0.0 < minimum_incumbent_fraction <= 1.0
    ):
        raise ValueError("minimum incumbent frequency fraction must be inside (0, 1]")

    candidate_rate = candidate_trades / eligible_resolved_markets
    required_rate = (
        incumbent_trades_per_eligible_resolved_market * minimum_incumbent_fraction
    )
    return {
        "name": "minimum_frequency_relative_to_incumbent",
        "candidate_trades": candidate_trades,
        "eligible_resolved_markets": eligible_resolved_markets,
        "candidate_trades_per_eligible_resolved_market": candidate_rate,
        "incumbent_trades_per_eligible_resolved_market": (
            incumbent_trades_per_eligible_resolved_market
        ),
        "candidate_to_incumbent_frequency_ratio": (
            candidate_rate / incumbent_trades_per_eligible_resolved_market
        ),
        "observed": candidate_rate,
        "threshold": required_rate,
        "minimum_incumbent_fraction": minimum_incumbent_fraction,
        "operator": ">=",
        "passed": candidate_rate >= required_rate,
    }


def matched_probability_quality(
    candidate: pl.DataFrame,
    oracle_control: pl.DataFrame,
    *,
    block_unit: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare probability quality on identical market/second observations.

    Bootstrap sampling is paired by complete market or UTC-day blocks. No economic
    outcome or PnL column participates in the calculation.
    """

    keys = ["market_id", "window_start", "seconds_elapsed"]
    required = {*keys, "label_up", "probability_yes"}
    _require_columns(candidate, required, "candidate probability quality")
    _require_columns(oracle_control, required, "Oracle control probability quality")
    if block_unit not in {"market", "utc_day"}:
        raise ValueError("probability quality block_unit must be market or utc_day")
    if resamples <= 0 or seed < 0:
        raise ValueError("probability quality bootstrap settings are invalid")
    if candidate.is_empty() or oracle_control.is_empty():
        raise ValueError("matched probability quality requires at least one observation")
    for name, frame in (("candidate", candidate), ("Oracle control", oracle_control)):
        if frame.select(*keys).is_duplicated().any():
            raise ValueError(f"{name} probability frame contains duplicate market/second keys")

    candidate_keys = candidate.select(*keys).sort(keys)
    oracle_keys = oracle_control.select(*keys).sort(keys)
    if not candidate_keys.equals(oracle_keys, null_equal=True):
        raise ValueError(
            "candidate and Oracle control require identical market/second keys"
        )
    paired = candidate.select(
        *keys,
        pl.col("label_up").alias("candidate_label_up"),
        pl.col("probability_yes").alias("candidate_probability_yes"),
    ).join(
        oracle_control.select(
            *keys,
            pl.col("label_up").alias("oracle_label_up"),
            pl.col("probability_yes").alias("oracle_probability_yes"),
        ),
        on=keys,
        how="inner",
        validate="1:1",
    )
    if paired.filter(pl.col("candidate_label_up") != pl.col("oracle_label_up")).height:
        raise ValueError("candidate and Oracle control labels differ on matched keys")

    candidate_probability = paired["candidate_probability_yes"].to_numpy()
    oracle_probability = paired["oracle_probability_yes"].to_numpy()
    labels = paired["candidate_label_up"].to_numpy().astype(np.float64)
    for name, probability in (
        ("candidate", candidate_probability),
        ("Oracle control", oracle_probability),
    ):
        if not np.all(np.isfinite(probability)) or np.any(
            (probability < 0.0) | (probability > 1.0)
        ):
            raise ValueError(f"{name} probabilities must be finite and inside [0, 1]")
    if not np.all(np.isin(labels, (0.0, 1.0))):
        raise ValueError("matched probability labels must be binary")

    candidate_brier = np.square(candidate_probability - labels)
    oracle_brier = np.square(oracle_probability - labels)
    candidate_clipped = np.clip(candidate_probability, 1e-15, 1.0 - 1e-15)
    oracle_clipped = np.clip(oracle_probability, 1e-15, 1.0 - 1e-15)
    candidate_log_loss = -(
        labels * np.log(candidate_clipped)
        + (1.0 - labels) * np.log(1.0 - candidate_clipped)
    )
    oracle_log_loss = -(
        labels * np.log(oracle_clipped)
        + (1.0 - labels) * np.log(1.0 - oracle_clipped)
    )
    paired = paired.with_columns(
        pl.Series("_brier_delta", candidate_brier - oracle_brier),
        pl.Series("_log_loss_delta", candidate_log_loss - oracle_log_loss),
    )
    bootstrap = _paired_probability_quality_bootstrap(
        paired,
        block_unit=block_unit,
        resamples=resamples,
        seed=seed,
    )
    brier_delta = float(np.mean(candidate_brier - oracle_brier))
    log_loss_delta = float(np.mean(candidate_log_loss - oracle_log_loss))
    return {
        "matched_rows": paired.height,
        "matched_markets": paired["market_id"].n_unique(),
        "matched_utc_days": paired["window_start"].dt.date().n_unique(),
        "block_unit": block_unit,
        "block_count": bootstrap["block_count"],
        "brier_score": {
            "candidate": float(np.mean(candidate_brier)),
            "oracle_control": float(np.mean(oracle_brier)),
            "candidate_minus_oracle_control": brier_delta,
            "improvement": -brier_delta,
            "candidate_minus_oracle_control_bootstrap": bootstrap["brier_delta"],
            "improvement_bootstrap": bootstrap["brier_improvement"],
        },
        "log_loss": {
            "candidate": float(np.mean(candidate_log_loss)),
            "oracle_control": float(np.mean(oracle_log_loss)),
            "candidate_minus_oracle_control": log_loss_delta,
            "improvement": -log_loss_delta,
            "candidate_minus_oracle_control_bootstrap": bootstrap["log_loss_delta"],
            "improvement_bootstrap": bootstrap["log_loss_improvement"],
        },
    }


def matched_probability_quality_gate_checks(
    metrics: dict[str, Any],
    *,
    brier_noninferiority_margin: float,
    log_loss_noninferiority_margin: float,
    minimum_brier_improvement: float = 0.0,
    minimum_log_loss_improvement: float = 0.0,
) -> list[dict[str, Any]]:
    """Return robust paired noninferiority and improvement checks."""

    thresholds = {
        "brier_score": (brier_noninferiority_margin, minimum_brier_improvement),
        "log_loss": (log_loss_noninferiority_margin, minimum_log_loss_improvement),
    }
    checks: list[dict[str, Any]] = []
    for metric_name, (margin, minimum_improvement) in thresholds.items():
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("probability noninferiority margins must be finite and nonnegative")
        if not np.isfinite(minimum_improvement) or minimum_improvement < 0.0:
            raise ValueError("minimum probability improvements must be finite and nonnegative")
        values = metrics.get(metric_name) or {}
        delta = values.get("candidate_minus_oracle_control")
        delta_upper = (
            values.get("candidate_minus_oracle_control_bootstrap") or {}
        ).get("upper_95")
        improvement = values.get("improvement")
        improvement_lower = (values.get("improvement_bootstrap") or {}).get("lower_95")
        checks.extend(
            [
                {
                    "name": f"{metric_name}_noninferior_to_oracle_control",
                    "observed": delta_upper,
                    "point_estimate": delta,
                    "threshold": margin,
                    "operator": "<=",
                    "passed": bool(
                        delta is not None
                        and delta_upper is not None
                        and np.isfinite(delta)
                        and np.isfinite(delta_upper)
                        and delta <= margin
                        and delta_upper <= margin
                    ),
                },
                {
                    "name": f"{metric_name}_improves_oracle_control",
                    "observed": improvement_lower,
                    "point_estimate": improvement,
                    "threshold": minimum_improvement,
                    "operator": ">",
                    "passed": bool(
                        improvement is not None
                        and improvement_lower is not None
                        and np.isfinite(improvement)
                        and np.isfinite(improvement_lower)
                        and improvement > minimum_improvement
                        and improvement_lower > minimum_improvement
                    ),
                },
            ]
        )
    return checks


def _paired_probability_quality_bootstrap(
    paired: pl.DataFrame,
    *,
    block_unit: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if block_unit == "utc_day":
        with_block = paired.with_columns(
            pl.col("window_start").dt.date().alias("_quality_block")
        )
    else:
        with_block = paired.with_columns(pl.col("market_id").alias("_quality_block"))
    blocks = (
        with_block.group_by("_quality_block")
        .agg(
            pl.col("_brier_delta").sum().alias("brier_delta_sum"),
            pl.col("_log_loss_delta").sum().alias("log_loss_delta_sum"),
            pl.len().alias("rows"),
        )
        .sort("_quality_block")
    )
    brier_sum = blocks["brier_delta_sum"].to_numpy()
    log_loss_sum = blocks["log_loss_delta_sum"].to_numpy()
    rows = blocks["rows"].to_numpy()
    rng = np.random.default_rng(seed)
    brier_delta = np.empty(resamples, dtype=np.float64)
    log_loss_delta = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, len(rows), len(rows))
        denominator = max(rows[chosen].sum(), 1)
        brier_delta[index] = brier_sum[chosen].sum() / denominator
        log_loss_delta[index] = log_loss_sum[chosen].sum() / denominator
    return {
        "block_count": blocks.height,
        "brier_delta": _interval(brier_delta),
        "brier_improvement": _interval(-brier_delta),
        "log_loss_delta": _interval(log_loss_delta),
        "log_loss_improvement": _interval(-log_loss_delta),
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


def _require_columns(frame: pl.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing columns: {', '.join(missing)}")


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
