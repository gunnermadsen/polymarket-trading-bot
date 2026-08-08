from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from btc_directional_model.asymmetric_value_config import (
    ValuePolicy,
    load_asymmetric_value_config,
)
from btc_directional_model.asymmetric_value_evaluation import (
    accuracy_price_by_five_second_interval,
    accuracy_price_by_second,
    confidence_control_ledger,
    current_policy_reference_ledger,
    joint_accuracy_value_surface,
    ledger_metrics,
    policy_ledger,
    score_two_sided_value,
    select_policy_candidate,
    side_accuracy_value_surface,
    side_time_price_strata_economics,
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
    ledger = policy_ledger(
        scored,
        _under30_policy(),
        quantity=5.0,
        maximum_depth_participation=0.25,
    )

    assert scored["selected_yes"].item() is True
    assert scored["selected_underdog"].item() is True
    assert scored["selected_edge_per_share"].item() == pytest.approx(0.12)
    assert ledger["realized_net"].item() == pytest.approx(4.60)


def test_cheap_loss_is_bounded_and_stress_is_reported() -> None:
    scored = score_two_sided_value(_predictions(label_up=0))
    ledger = policy_ledger(
        scored,
        _under30_policy(),
        quantity=5.0,
        maximum_depth_participation=0.25,
    )
    metrics = ledger_metrics(ledger)

    assert ledger["realized_net"].item() == pytest.approx(-0.40)
    assert metrics["maximum_loss"] == pytest.approx(-0.40)
    assert metrics["stress_1c_net_expectancy_per_trade"] == pytest.approx(-0.45)


def test_policy_selects_best_eligible_side_before_global_edge() -> None:
    predictions = _predictions(probability_yes=0.80, label_up=0).with_columns(
        pl.lit(0.40).alias("yes_ask_vwap_5"),
        pl.lit(0.40).alias("yes_execution_cost_per_share"),
        pl.lit(0.40).alias("yes_cost_per_share"),
        pl.lit(0.10).alias("no_ask_vwap_5"),
        pl.lit(0.10).alias("no_execution_cost_per_share"),
        pl.lit(0.10).alias("no_cost_per_share"),
    )
    scored = score_two_sided_value(predictions)

    ledger = policy_ledger(
        scored,
        _under30_policy(),
        quantity=5.0,
        maximum_depth_participation=0.25,
    )

    assert scored["selected_yes"].item() is True
    assert ledger["selected_yes"].item() is False
    assert ledger["selected_share_price"].item() == pytest.approx(0.10)
    assert ledger["realized_net"].item() == pytest.approx(4.50)


def test_policy_preserves_twenty_share_depth_requirement() -> None:
    predictions = _predictions().with_columns(
        pl.lit(19.0).alias("yes_ask_depth"),
    )

    ledger = policy_ledger(
        score_two_sided_value(predictions),
        _under30_policy(),
        quantity=5.0,
        maximum_depth_participation=0.25,
    )

    accepted = policy_ledger(
        score_two_sided_value(_predictions()),
        _under30_policy(),
        quantity=5.0,
        maximum_depth_participation=0.25,
    )

    assert ledger.is_empty()
    assert ledger.schema == accepted.schema
    assert pl.concat((ledger, accepted), how="vertical_relaxed").height == 1


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
        maximum_depth_participation=0.25,
        quantity=5.0,
    )

    assert ledger.is_empty()


def test_confidence_control_preserves_depth_participation() -> None:
    predictions = _predictions(probability_yes=0.80).with_columns(
        pl.lit(19.0).alias("yes_ask_depth")
    )

    ledger = confidence_control_ledger(
        predictions,
        threshold=0.55,
        maximum_entry_second=240,
        maximum_cost_per_share=0.70,
        minimum_edge_per_share=0.015,
        maximum_depth_participation=0.25,
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
        maximum_depth_participation=0.25,
    )
    metrics = ledger_metrics(ledger)

    assert metrics["profit_factor_no_losses"] is True
    assert metrics["average_loss"] == 0.0
    assert metrics["maximum_loss"] == 0.0
    assert metrics["loss_recovery_wins"] == 0.0


def test_policy_selection_cannot_be_won_by_expensive_diagnostic() -> None:
    config = load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-one-second-20260414-20260802.toml"
    )
    primary = next(policy for policy in config.policies if policy.selection_eligible)
    diagnostic = next(
        policy for policy in config.policies if not policy.selection_eligible
    )
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


def test_per_second_report_covers_hybrid_grid_quality_prices_and_support() -> None:
    start = datetime(2026, 7, 16, tzinfo=UTC)
    second_one_yes = _predictions(probability_yes=0.80, label_up=1).with_columns(
        pl.lit(1).alias("seconds_elapsed"),
        pl.lit(start + timedelta(seconds=1)).alias("observed_at"),
    )
    second_one_no = _predictions(probability_yes=0.20, label_up=0).with_columns(
        pl.lit("m-next-day").alias("market_id"),
        pl.lit(start + timedelta(days=1)).alias("window_start"),
        pl.lit(start + timedelta(days=1, seconds=1)).alias("observed_at"),
        pl.lit(1).alias("seconds_elapsed"),
        pl.lit(0.12).alias("yes_best_ask"),
        pl.lit(0.14).alias("yes_ask_vwap_5"),
        pl.lit(0.16).alias("yes_cost_per_share"),
        pl.lit(0.85).alias("no_best_ask"),
        pl.lit(0.86).alias("no_ask_vwap_5"),
        pl.lit(0.88).alias("no_cost_per_share"),
    )
    later = [
        _predictions(probability_yes=0.60, label_up=label).with_columns(
            pl.lit(f"m-{second}").alias("market_id"),
            pl.lit(start + timedelta(seconds=second)).alias("observed_at"),
            pl.lit(second).alias("seconds_elapsed"),
        )
        for second, label in ((59, 1), (60, 1), (65, 0))
    ]
    scored = score_two_sided_value(
        pl.concat((second_one_yes, second_one_no, *later), how="vertical_relaxed")
    )

    rows = accuracy_price_by_second(scored)
    at_one = next(row for row in rows if row["seconds_elapsed"] == 1)

    assert [row["seconds_elapsed"] for row in rows] == [1, 59, 60, 65]
    assert at_one["rows"] == 2
    assert at_one["markets"] == 2
    assert at_one["utc_days"] == 2
    assert at_one["accuracy"] == 1.0
    assert at_one["argmax_accuracy"] == 1.0
    assert at_one["brier_score"] == pytest.approx(0.04)
    assert at_one["log_loss"] == pytest.approx(-math.log(0.8))
    assert at_one["calibration_bias"] == pytest.approx(0.0)
    assert at_one["absolute_calibration_bias"] == pytest.approx(0.0)
    assert at_one["mean_yes_best_ask"] == pytest.approx(0.10)
    assert at_one["mean_yes_vwap_5"] == pytest.approx(0.11)
    assert at_one["mean_yes_all_in_cost_per_share"] == pytest.approx(0.125)
    assert at_one["mean_no_best_ask"] == pytest.approx(0.89)
    assert at_one["mean_no_vwap_5"] == pytest.approx(0.895)
    assert at_one["mean_no_all_in_cost_per_share"] == pytest.approx(0.91)


def test_per_second_report_keeps_models_separate() -> None:
    scored = score_two_sided_value(
        pl.concat(
            (
                _predictions(probability_yes=0.80, label_up=1).with_columns(
                    pl.lit("model-a").alias("model")
                ),
                _predictions(probability_yes=0.20, label_up=1).with_columns(
                    pl.lit("model-b").alias("model")
                ),
            ),
            how="vertical_relaxed",
        )
    )

    rows = accuracy_price_by_second(scored)

    assert [(row["model"], row["seconds_elapsed"]) for row in rows] == [
        ("model-a", 5),
        ("model-b", 5),
    ]
    assert rows[0]["argmax_accuracy"] == 1.0
    assert rows[1]["argmax_accuracy"] == 0.0


def test_five_second_report_has_fixed_half_open_intervals_and_selected_scores() -> None:
    start = datetime(2026, 7, 16, tzinfo=UTC)
    rows = []
    for market, second, probability, label, yes_vwap, no_vwap in (
        ("m-1", 1, 0.80, 1, 0.20, 0.81),
        ("m-5", 5, 0.60, 1, 0.24, 0.77),
        ("m-6", 6, 0.25, 0, 0.22, 0.79),
        ("m-55", 55, 0.70, 1, 0.29, 0.72),
    ):
        rows.append(
            _predictions(probability_yes=probability, label_up=label).with_columns(
                pl.lit(market).alias("market_id"),
                pl.lit(start + timedelta(minutes=len(rows) * 5)).alias("window_start"),
                pl.lit(start + timedelta(seconds=second)).alias("observed_at"),
                pl.lit(second).alias("seconds_elapsed"),
                pl.lit(yes_vwap).alias("yes_ask_vwap_5"),
                pl.lit(yes_vwap + 0.01).alias("yes_execution_cost_per_share"),
                pl.lit(yes_vwap + 0.02).alias("yes_cost_per_share"),
                pl.lit(no_vwap).alias("no_ask_vwap_5"),
                pl.lit(no_vwap + 0.01).alias("no_execution_cost_per_share"),
                pl.lit(no_vwap + 0.02).alias("no_cost_per_share"),
            )
        )
    scored = score_two_sided_value(pl.concat(rows, how="vertical_relaxed"))

    report = accuracy_price_by_five_second_interval(scored)

    assert len(report) == 11
    assert [row["interval"] for row in report] == [
        "[1,6)",
        "[6,11)",
        "[11,16)",
        "[16,21)",
        "[21,26)",
        "[26,31)",
        "[31,36)",
        "[36,41)",
        "[41,46)",
        "[46,51)",
        "[51,56)",
    ]
    assert report[0]["rows"] == 2
    assert report[0]["wins"] == 2
    assert report[0]["accuracy"] == 1.0
    assert report[0]["brier_score"] == pytest.approx((0.04 + 0.16) / 2.0)
    assert report[0]["log_loss"] == pytest.approx(
        (-math.log(0.8) - math.log(0.6)) / 2.0
    )
    assert report[0]["calibration_bias"] == pytest.approx(-0.30)
    assert report[0]["mean_yes_vwap_5"] == pytest.approx(0.22)
    assert report[0]["mean_no_vwap_5"] == pytest.approx(0.79)
    assert report[0]["mean_selected_raw_share_price"] == pytest.approx(0.22)
    assert report[1]["rows"] == 1
    assert report[1]["wins"] == 0
    assert report[1]["accuracy"] == 0.0
    assert report[2]["rows"] == 0
    assert report[2]["accuracy"] is None
    assert report[-1]["rows"] == 1


def test_five_second_report_rejects_missing_columns_and_out_of_scope_seconds() -> None:
    scored = score_two_sided_value(_predictions())

    with pytest.raises(ValueError, match="missing columns: no_ask_vwap_5"):
        accuracy_price_by_five_second_interval(scored.drop("no_ask_vwap_5"))
    with pytest.raises(ValueError, match=r"seconds in \[1, 56\)"):
        accuracy_price_by_five_second_interval(
            scored.with_columns(pl.lit(56).alias("seconds_elapsed"))
        )


def _strata_ledger() -> pl.DataFrame:
    start = datetime(2026, 7, 16, tzinfo=UTC)
    rows = []
    for index, (second, price, selected_yes, won) in enumerate(
        (
            (1, 0.20, True, True),
            (14, 0.249999, False, False),
            (15, 0.25, True, False),
            (30, 0.275, False, True),
            (45, 0.299999, True, True),
            (55, 0.20, False, False),
        )
    ):
        execution_cost = price + 0.005
        quantity = 5.0
        return_value = (float(won) - execution_cost) * quantity
        rows.append(
            {
                "model": "selected-model",
                "policy": "primary",
                "market_id": f"m-{index}",
                "window_start": start + timedelta(minutes=5 * index),
                "observed_at": start + timedelta(minutes=5 * index, seconds=second),
                "seconds_elapsed": second,
                "selected_yes": selected_yes,
                "won": won,
                "quantity": quantity,
                "selected_probability": 0.40 if won else 0.35,
                "selected_share_price": price,
                "selected_admission_cost_per_share": price + 0.015,
                "selected_execution_cost_per_share": execution_cost,
                "selected_edge_per_share": 0.05,
                "selected_underdog": True,
                "realized_net": return_value,
                "entry_debit": execution_cost * quantity,
            }
        )
    return pl.DataFrame(rows)


def test_side_time_price_strata_economics_respects_every_half_open_boundary() -> None:
    report = side_time_price_strata_economics(
        _strata_ledger(),
        time_strata=((1, 15), (15, 30), (30, 45), (45, 56)),
        price_strata=((0.20, 0.25), (0.25, 0.275), (0.275, 0.30)),
    )

    assert len(report) == 24
    assert [
        (row["side"], row["time_interval"], row["price_interval"]) for row in report
    ] == sorted(
        (
            (side, time, price)
            for side in ("YES", "NO")
            for time in ("[1,15)", "[15,30)", "[30,45)", "[45,56)")
            for price in ("[0.2,0.25)", "[0.25,0.275)", "[0.275,0.3)")
        ),
        key=lambda item: (
            ("YES", "NO").index(item[0]),
            ("[1,15)", "[15,30)", "[30,45)", "[45,56)").index(item[1]),
            ("[0.2,0.25)", "[0.25,0.275)", "[0.275,0.3)").index(item[2]),
        ),
    )
    yes_second_band = next(
        row
        for row in report
        if row["side"] == "YES"
        and row["time_start_second"] == 15
        and row["price_start"] == 0.25
    )
    no_third_band = next(
        row
        for row in report
        if row["side"] == "NO"
        and row["time_start_second"] == 30
        and row["price_start"] == 0.275
    )
    assert yes_second_band["trades"] == 1
    assert yes_second_band["wins"] == 0
    assert yes_second_band["net_expectancy_per_trade"] < 0.0
    assert no_third_band["trades"] == 1
    assert no_third_band["wins"] == 1
    assert no_third_band["net_expectancy_per_trade"] > 0.0
    assert sum(row["trades"] for row in report) == 6


def test_side_time_price_strata_economics_fails_closed_on_scope_errors() -> None:
    ledger = _strata_ledger()
    kwargs = {
        "time_strata": ((1, 15), (15, 30), (30, 45), (45, 56)),
        "price_strata": ((0.20, 0.25), (0.25, 0.275), (0.275, 0.30)),
    }

    with pytest.raises(ValueError, match="one selected trade per market"):
        side_time_price_strata_economics(
            pl.concat((ledger, ledger.head(1)), how="vertical"),
            **kwargs,
        )
    with pytest.raises(ValueError, match="price strata requires every row"):
        side_time_price_strata_economics(
            ledger.with_columns(
                pl.when(pl.col("market_id") == "m-0")
                .then(pl.lit(0.30))
                .otherwise(pl.col("selected_share_price"))
                .alias("selected_share_price")
            ),
            **kwargs,
        )
    with pytest.raises(ValueError, match="exactly one model"):
        side_time_price_strata_economics(
            ledger.with_columns(
                pl.when(pl.col("market_id") == "m-0")
                .then(pl.lit("other-model"))
                .otherwise(pl.col("model"))
                .alias("model")
            ),
            **kwargs,
        )
