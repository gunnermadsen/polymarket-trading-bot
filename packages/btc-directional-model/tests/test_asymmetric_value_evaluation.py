from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from btc_directional_model.asymmetric_value_config import (
    ValuePolicy,
    load_asymmetric_value_config,
)
from btc_directional_model.asymmetric_value_evaluation import (
    confidence_control_ledger,
    current_policy_reference_ledger,
    joint_accuracy_value_surface,
    ledger_metrics,
    policy_ledger,
    score_two_sided_value,
    select_policy_candidate,
    side_accuracy_value_surface,
)


def _predictions(*, probability_yes: float = 0.21, label_up: int = 1) -> pl.DataFrame:
    timestamp = datetime(2026, 7, 16, tzinfo=UTC)
    return pl.DataFrame(
        {
            "market_id": ["m"],
            "window_start": [timestamp],
            "observed_at": [timestamp],
            "seconds_elapsed": [5],
            "label_up": [label_up],
            "model": ["test"],
            "probability_yes": [probability_yes],
            "yes_best_ask": [0.08],
            "yes_ask_vwap_5": [0.08],
            "yes_ask_depth": [20.0],
            "no_best_ask": [0.93],
            "no_ask_vwap_5": [0.93],
            "no_ask_depth": [20.0],
            "yes_execution_cost_per_share": [0.08],
            "no_execution_cost_per_share": [0.93],
            "yes_cost_per_share": [0.09],
            "no_cost_per_share": [0.94],
            "fee_rate": [0.0],
        }
    )


def _under30_policy() -> ValuePolicy:
    return ValuePolicy(
        name="under30",
        selection_eligible=True,
        maximum_entry_second=120,
        minimum_share_price=0.05,
        maximum_share_price=0.30,
        maximum_cost_per_share=0.35,
        minimum_edge_per_share=0.03,
    )


def test_low_probability_underdog_is_selected_when_price_creates_edge() -> None:
    scored = score_two_sided_value(_predictions())
    ledger = policy_ledger(scored, _under30_policy(), quantity=5.0)

    assert scored["selected_yes"].item() is True
    assert scored["selected_underdog"].item() is True
    assert scored["selected_edge_per_share"].item() == pytest.approx(0.12)
    assert ledger["realized_net"].item() == pytest.approx(4.60)


def test_cheap_loss_is_bounded_and_stress_is_reported() -> None:
    scored = score_two_sided_value(_predictions(label_up=0))
    ledger = policy_ledger(scored, _under30_policy(), quantity=5.0)
    metrics = ledger_metrics(ledger)

    assert ledger["realized_net"].item() == pytest.approx(-0.40)
    assert metrics["maximum_loss"] == pytest.approx(-0.40)
    assert metrics["stress_1c_net_expectancy_per_trade"] == pytest.approx(-0.45)


def test_confidence_control_requires_positive_conservative_edge() -> None:
    predictions = _predictions(probability_yes=0.60).with_columns(
        pl.lit(0.70).alias("yes_cost_per_share"),
        pl.lit(0.70).alias("yes_execution_cost_per_share"),
    )
    ledger = confidence_control_ledger(
        predictions,
        threshold=0.55,
        maximum_entry_second=240,
        maximum_cost_per_share=0.70,
        minimum_edge_per_share=0.015,
        quantity=5.0,
    )

    assert ledger.is_empty()


def test_current_policy_control_does_not_add_an_edge_gate() -> None:
    predictions = _predictions(probability_yes=0.90).with_columns(
        pl.lit(60).alias("seconds_elapsed"),
        pl.lit(0.92).alias("yes_ask_vwap_5"),
        pl.lit(0.92).alias("yes_execution_cost_per_share"),
        pl.lit(20.0).alias("yes_ask_depth"),
    )
    ledger = current_policy_reference_ledger(
        predictions,
        threshold=0.89,
        minimum_entry_second=60,
        maximum_entry_second=240,
        minimum_share_price=0.30,
        maximum_share_price=0.95,
        maximum_depth_participation=0.25,
        quantity=5.0,
    )

    assert ledger.height == 1
    assert ledger["selected_share_price"].item() == pytest.approx(0.92)


def test_current_policy_does_not_substitute_after_rejected_first_crossing() -> None:
    start = datetime(2026, 7, 16, tzinfo=UTC)
    first = _predictions(probability_yes=0.90).with_columns(
        pl.lit(60).alias("seconds_elapsed"),
        pl.lit(start + timedelta(seconds=60)).alias("observed_at"),
    )
    later = _predictions(probability_yes=0.91).with_columns(
        pl.lit(65).alias("seconds_elapsed"),
        pl.lit(start + timedelta(seconds=65)).alias("observed_at"),
        pl.lit(0.40).alias("yes_ask_vwap_5"),
        pl.lit(0.40).alias("yes_execution_cost_per_share"),
    )
    combined = pl.concat((first, later), how="vertical_relaxed")
    predictions = combined.select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "model",
        "probability_yes",
    )
    execution = later

    ledger = current_policy_reference_ledger(
        predictions,
        execution=execution,
        threshold=0.89,
        minimum_entry_second=60,
        maximum_entry_second=240,
        minimum_share_price=0.30,
        maximum_share_price=0.95,
        maximum_depth_participation=0.25,
        quantity=5.0,
    )

    assert ledger.is_empty()


def test_all_win_ledger_passes_loss_shape_with_zero_loss() -> None:
    ledger = policy_ledger(
        score_two_sided_value(_predictions(label_up=1)),
        _under30_policy(),
        quantity=5.0,
    )
    metrics = ledger_metrics(ledger)

    assert metrics["profit_factor_no_losses"] is True
    assert metrics["average_loss"] == 0.0
    assert metrics["maximum_loss"] == 0.0
    assert metrics["loss_recovery_wins"] == 0.0


def test_policy_selection_cannot_be_won_by_expensive_diagnostic() -> None:
    config = load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-hunter-20260414-20260802.toml"
    )
    primary = next(policy for policy in config.policies if policy.selection_eligible)
    diagnostic = next(policy for policy in config.policies if not policy.selection_eligible)
    weak = {"trades": 0, "net_profit": 0.0}
    strong = {
        "trades": 10_000,
        "net_profit": 10_000.0,
        "net_expectancy_per_trade": 1.0,
        "capital_efficiency": 10.0,
        "profit_factor": 10.0,
    }

    selection = select_policy_candidate(
        {
            f"test::{primary.name}": weak,
            f"test::{diagnostic.name}": strong,
        },
        config,
    )

    assert selection["selected_policy"] == primary.name


def test_joint_surface_keeps_second_side_and_raw_price_band() -> None:
    rows = joint_accuracy_value_surface(score_two_sided_value(_predictions()))

    assert rows[0]["seconds_elapsed"] == 5
    assert rows[0]["selected_side"] == "YES"
    assert rows[0]["price_band"] == "00_10c"
    assert rows[0]["actual_win_rate"] == 1.0


def test_both_side_surface_reports_yes_and_no_without_selection() -> None:
    rows = side_accuracy_value_surface(_predictions())

    assert {(row["side"], row["price_band"]) for row in rows} == {
        ("YES", "00_10c"),
        ("NO", "90_100c"),
    }
    yes = next(row for row in rows if row["side"] == "YES")
    no = next(row for row in rows if row["side"] == "NO")
    assert yes["mean_side_probability"] == pytest.approx(0.21)
    assert yes["actual_win_rate"] == 1.0
    assert no["mean_side_probability"] == pytest.approx(0.79)
    assert no["actual_win_rate"] == 0.0
