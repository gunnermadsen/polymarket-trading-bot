from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from kraken_ml.evaluation import (
    Policy,
    baseline_metrics,
    economic_metrics,
    policy_actions,
    trade_ledger,
)


def _ledger_frame(row_count: int = 6) -> pl.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    decisions = [start + timedelta(minutes=15 * index) for index in range(row_count)]
    return pl.DataFrame(
        {
            "bucket_start": decisions,
            "entry_at": [value + timedelta(minutes=15) for value in decisions],
            "label_exit_at": [value + timedelta(minutes=75) for value in decisions],
            "gross_forward_bps": [20.0] * row_count,
            "market_execution_cost_bps": [2.0] * row_count,
            "long_market_execution_cost_bps": [2.0] * row_count,
            "short_market_execution_cost_bps": [2.0] * row_count,
            "fee_cost_bps": [10.0] * row_count,
            "funding_horizon_bps": [1.0] * row_count,
        }
    )


def test_policy_actions_require_threshold_margin_and_direction_over_flat() -> None:
    probabilities = np.asarray(
        [
            [0.10, 0.20, 0.70],
            [0.65, 0.20, 0.15],
            [0.35, 0.40, 0.25],
            [0.45, 0.10, 0.45],
        ]
    )
    policy = Policy(probability_threshold=0.50, directional_margin=0.20)

    np.testing.assert_array_equal(
        policy_actions(probabilities, policy),
        np.asarray([1, -1, 0, 0], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        policy_actions(probabilities, Policy(1.0, 1.0, no_trade=True)),
        np.zeros(4, dtype=np.int8),
    )


def test_baseline_summary_is_computed_only_from_prediction_rules() -> None:
    frame = pl.DataFrame(
        {
            "label": [-1, 0, 1, 1, 0, -1],
            "momentum_label": [-1, 0, 1, 0, 1, -1],
        }
    )
    metrics = baseline_metrics(
        fit_labels=np.asarray([-1, -1, 0, 1, 1], dtype=np.int8),
        evaluation_frame=frame,
    )

    assert 0.0 <= metrics["best_balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["best_macro_f1"] <= 1.0
    assert metrics["class_prior_probability"]["log_loss"] > 0


def test_trade_ledger_enforces_single_position_and_dynamic_cost_multiplier() -> None:
    frame = _ledger_frame()
    actions = np.ones(frame.height, dtype=np.int8)

    nominal = trade_ledger(frame, actions, execution_cost_multiplier=1.0)
    stressed = trade_ledger(frame, actions, execution_cost_multiplier=2.0)

    assert len(nominal) == 2
    assert [trade["decision_time"] for trade in nominal] == [
        frame.item(0, "bucket_start"),
        frame.item(4, "bucket_start"),
    ]
    assert nominal[1]["entry_at"] == nominal[0]["exit_at"]
    assert all(
        current["entry_at"] >= previous["exit_at"]
        for previous, current in zip(nominal[:-1], nominal[1:], strict=True)
    )
    assert nominal[0]["gross_bps"] == pytest.approx(20.0)
    assert nominal[0]["fee_bps"] == pytest.approx(10.0)
    assert nominal[0]["market_execution_bps"] == pytest.approx(2.0)
    assert nominal[0]["funding_bps"] == pytest.approx(-1.0)
    assert nominal[0]["net_bps"] == pytest.approx(7.0)
    assert stressed[0]["market_execution_bps"] == pytest.approx(4.0)
    assert stressed[0]["net_bps"] == pytest.approx(5.0)


def test_positive_month_fraction_counts_months_without_trades() -> None:
    frame = _ledger_frame(3).with_columns(
        pl.Series(
            "bucket_start",
            [
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 2, 1, tzinfo=UTC),
                datetime(2026, 3, 1, tzinfo=UTC),
            ],
        ),
        pl.Series(
            "entry_at",
            [
                datetime(2026, 1, 1, 0, 15, tzinfo=UTC),
                datetime(2026, 2, 1, 0, 15, tzinfo=UTC),
                datetime(2026, 3, 1, 0, 15, tzinfo=UTC),
            ],
        ),
        pl.Series(
            "label_exit_at",
            [
                datetime(2026, 1, 1, 1, 15, tzinfo=UTC),
                datetime(2026, 2, 1, 1, 15, tzinfo=UTC),
                datetime(2026, 3, 1, 1, 15, tzinfo=UTC),
            ],
        ),
    )
    metrics = economic_metrics(
        frame,
        np.asarray([1, 0, 0], dtype=np.int8),
        execution_cost_multiplier=1.0,
        bootstrap_repetitions=10,
        seed=17,
    )

    assert metrics["trades"] == 1
    assert metrics["positive_month_fraction"] == pytest.approx(1 / 3)
