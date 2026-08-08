from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from btc_directional_model.asymmetric_value_config import ValuePolicy
from btc_directional_model.asymmetric_value_evaluation import (
    EDGE_POSITIVE_2_OF_LAST_3_SECONDS,
    IMMEDIATE_FIRST_CROSSING,
    bootstrap_ledger_metrics,
    frequency_floor_check,
    ledger_metrics,
    matched_probability_quality,
    matched_probability_quality_gate_checks,
    policy_ledger,
    rejection_funnel,
    score_two_sided_value,
    selected_win_rate_advantage_gate_checks,
    temporal_confirmation_ablation,
)


def _policy() -> ValuePolicy:
    return ValuePolicy(
        name="asymmetric_20_30c",
        selection_eligible=True,
        maximum_entry_second=55,
        minimum_share_price=0.20,
        maximum_share_price=0.30,
        maximum_cost_per_share=0.35,
        minimum_edge_per_share=0.03,
    )


def _prediction(
    market_id: str,
    second: int,
    *,
    side: str,
    label_up: int = 1,
    day: int = 0,
    observed_offset: float | None = None,
    depth: float = 20.0,
) -> pl.DataFrame:
    window_start = datetime(2026, 7, 23, tzinfo=UTC) + timedelta(days=day)
    observed_at = window_start + timedelta(
        seconds=second if observed_offset is None else observed_offset
    )
    if side == "YES":
        probability_yes = 0.40
        yes_price, yes_cost = 0.25, 0.27
        no_price, no_cost = 0.75, 0.77
    elif side == "NO":
        probability_yes = 0.60
        yes_price, yes_cost = 0.75, 0.77
        no_price, no_cost = 0.25, 0.27
    elif side == "BOTH":
        probability_yes = 0.50
        yes_price = no_price = 0.25
        yes_cost = no_cost = 0.27
    else:
        raise ValueError(f"unsupported test side: {side}")
    return pl.DataFrame(
        {
            "market_id": [market_id],
            "window_start": [window_start],
            "observed_at": [observed_at],
            "seconds_elapsed": [second],
            "label_up": [label_up],
            "model": ["candidate"],
            "probability_yes": [probability_yes],
            "yes_best_ask": [yes_price],
            "yes_ask_vwap_5": [yes_price],
            "yes_ask_depth": [depth],
            "no_best_ask": [no_price],
            "no_ask_vwap_5": [no_price],
            "no_ask_depth": [depth],
            "yes_execution_cost_per_share": [yes_price],
            "no_execution_cost_per_share": [no_price],
            "yes_cost_per_share": [yes_cost],
            "no_cost_per_share": [no_cost],
            "fee_rate": [0.0],
        }
    )


def _scored(*rows: pl.DataFrame) -> pl.DataFrame:
    return score_two_sided_value(pl.concat(rows, how="vertical_relaxed"))


def _ledger(labels: list[int]) -> pl.DataFrame:
    rows = [
        _prediction(f"market-{index}", 5, side="YES", label_up=label, day=index)
        for index, label in enumerate(labels)
    ]
    return policy_ledger(
        _scored(*rows),
        _policy(),
        quantity=5.0,
        maximum_depth_participation=0.25,
    )


def test_ledger_reports_five_share_economics_and_conservative_break_even() -> None:
    ledger = _ledger([1, 0])
    metrics = ledger_metrics(ledger)

    assert ledger.sort("market_id")["realized_net"].to_list() == pytest.approx(
        [3.75, -1.25]
    )
    assert metrics["accuracy"] == pytest.approx(0.50)
    assert metrics["selected_win_rate"] == pytest.approx(0.50)
    assert metrics["conservative_all_in_break_even_probability"] == pytest.approx(0.27)
    assert metrics["selected_win_rate_advantage"] == pytest.approx(0.23)


def test_win_rate_advantage_bootstrap_is_deterministic_and_gated() -> None:
    positive = _ledger([1, 1, 1, 1])
    first = bootstrap_ledger_metrics(positive, resamples=1_000, seed=20260808)
    second = bootstrap_ledger_metrics(positive, resamples=1_000, seed=20260808)

    assert first == second
    assert first is not None
    assert first["selected_win_rate_advantage"]["lower_95"] == pytest.approx(0.73)
    positive_metrics = ledger_metrics(positive)
    positive_metrics["utc_day_block_bootstrap"] = first
    assert all(
        check["passed"]
        for check in selected_win_rate_advantage_gate_checks(positive_metrics)
    )

    negative = _ledger([0, 0, 0, 0])
    negative_bootstrap = bootstrap_ledger_metrics(
        negative,
        resamples=1_000,
        seed=20260808,
    )
    assert negative_bootstrap is not None
    assert negative_bootstrap["selected_win_rate_advantage"]["upper_95"] < 0.0
    negative_metrics = ledger_metrics(negative)
    negative_metrics["utc_day_block_bootstrap"] = negative_bootstrap
    assert not any(
        check["passed"]
        for check in selected_win_rate_advantage_gate_checks(negative_metrics)
    )


def test_temporal_confirmation_is_same_side_exact_second_and_causal() -> None:
    scored = _scored(
        _prediction("alternating", 1, side="YES"),
        _prediction("alternating", 2, side="NO"),
        _prediction("alternating", 3, side="NO"),
        _prediction("irregular-gap", 1, side="YES", observed_offset=1.2),
        _prediction("irregular-gap", 4, side="YES", observed_offset=4.7),
        _prediction("exact-gap", 1, side="YES", observed_offset=1.2),
        _prediction("exact-gap", 3, side="YES", observed_offset=3.7),
        _prediction("future-observation", 1, side="YES", observed_offset=10.0),
        _prediction("future-observation", 2, side="YES", observed_offset=2.0),
    )

    ledgers, metrics = temporal_confirmation_ablation(
        scored,
        _policy(),
        quantity=5.0,
        maximum_depth_participation=0.25,
    )
    immediate = ledgers[IMMEDIATE_FIRST_CROSSING]
    confirmed = ledgers[EDGE_POSITIVE_2_OF_LAST_3_SECONDS].sort("market_id")

    assert immediate["market_id"].n_unique() == 4
    assert confirmed["market_id"].to_list() == ["alternating", "exact-gap"]
    assert confirmed["seconds_elapsed"].to_list() == [3, 3]
    assert confirmed["selected_yes"].to_list() == [False, True]
    assert metrics[EDGE_POSITIVE_2_OF_LAST_3_SECONDS]["trades"] == 2


def test_rejection_funnel_counts_side_overlap_as_one_aggregate_market() -> None:
    scored = _scored(
        _prediction("both", 1, side="BOTH"),
        _prediction("both", 2, side="BOTH"),
        _prediction("shallow", 1, side="YES", depth=10.0),
        _prediction("shallow", 2, side="YES", depth=10.0),
    )

    funnel = rejection_funnel(
        scored,
        _policy(),
        quantity=5.0,
        maximum_depth_participation=0.25,
    )
    stages = {stage["stage"]: stage for stage in funnel["stages"]}

    assert funnel["fresh_strict_execution_input"] is True
    assert stages["source_candidate_rows"]["aggregate_rows"] == 4
    assert stages["source_candidate_rows"]["yes_rows"] == 4
    assert stages["source_candidate_rows"]["no_rows"] == 4
    assert stages["time_raw_price"]["aggregate_markets"] == 2
    assert stages["depth"]["aggregate_markets"] == 1
    assert stages["temporal_confirmation"]["yes_markets"] == 1
    assert stages["temporal_confirmation"]["no_markets"] == 1
    assert stages["temporal_confirmation"]["aggregate_markets"] == 1
    assert stages["selected_one_trade_per_market"]["aggregate_rows"] == 1


def test_frequency_floor_requires_eighty_percent_of_supplied_incumbent() -> None:
    passing = frequency_floor_check(
        candidate_trades=40,
        eligible_resolved_markets=100,
        incumbent_trades_per_eligible_resolved_market=0.50,
    )
    failing = frequency_floor_check(
        candidate_trades=39,
        eligible_resolved_markets=100,
        incumbent_trades_per_eligible_resolved_market=0.50,
    )

    assert passing["candidate_to_incumbent_frequency_ratio"] == pytest.approx(0.80)
    assert passing["passed"] is True
    assert failing["passed"] is False


def _probability_frame(probabilities: list[float]) -> pl.DataFrame:
    start = datetime(2026, 7, 23, tzinfo=UTC)
    labels = [1, 0, 1, 0]
    return pl.DataFrame(
        {
            "market_id": [f"market-{index}" for index in range(4)],
            "window_start": [start + timedelta(days=index) for index in range(4)],
            "seconds_elapsed": [5, 10, 15, 20],
            "label_up": labels,
            "probability_yes": probabilities,
        }
    )


@pytest.mark.parametrize("block_unit", ["market", "utc_day"])
def test_matched_probability_quality_improves_oracle_control_deterministically(
    block_unit: str,
) -> None:
    candidate = _probability_frame([0.90, 0.10, 0.90, 0.10])
    oracle = _probability_frame([0.70, 0.30, 0.70, 0.30])

    first = matched_probability_quality(
        candidate,
        oracle,
        block_unit=block_unit,
        resamples=1_000,
        seed=20260808,
    )
    second = matched_probability_quality(
        candidate,
        oracle,
        block_unit=block_unit,
        resamples=1_000,
        seed=20260808,
    )

    assert first == second
    assert first["brier_score"]["candidate"] == pytest.approx(0.01)
    assert first["brier_score"]["oracle_control"] == pytest.approx(0.09)
    assert first["brier_score"]["improvement"] == pytest.approx(0.08)
    assert first["log_loss"]["improvement"] > 0.0
    checks = matched_probability_quality_gate_checks(
        first,
        brier_noninferiority_margin=0.01,
        log_loss_noninferiority_margin=0.01,
    )
    assert len(checks) == 4
    assert all(check["passed"] for check in checks)


def test_matched_probability_quality_rejects_nonidentical_keys() -> None:
    candidate = _probability_frame([0.90, 0.10, 0.90, 0.10])
    oracle = _probability_frame([0.70, 0.30, 0.70, 0.30]).with_columns(
        pl.when(pl.col("market_id") == "market-3")
        .then(pl.lit(21))
        .otherwise(pl.col("seconds_elapsed"))
        .alias("seconds_elapsed")
    )

    with pytest.raises(ValueError, match="identical market/second keys"):
        matched_probability_quality(
            candidate,
            oracle,
            block_unit="utc_day",
            resamples=100,
            seed=1,
        )
