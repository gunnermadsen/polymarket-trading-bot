from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from kraken_ml.regression_evaluation import (
    POLICY_ADVANTAGES_BPS,
    POLICY_HURDLES_BPS,
    RegressionPolicy,
    choose_regression_policy,
    fee_counterfactuals,
    regression_diagnostics,
    regression_policy_actions,
)


def _evaluation_frame(row_count: int, *, gross_bps: float = 30.0) -> pl.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    decisions = [start + timedelta(hours=2 * index) for index in range(row_count)]
    return pl.DataFrame(
        {
            "bucket_start": decisions,
            "entry_at": [value + timedelta(minutes=15) for value in decisions],
            "label_exit_at": [value + timedelta(hours=1, minutes=15) for value in decisions],
            "gross_forward_bps": [gross_bps] * row_count,
            "long_net_bps": [gross_bps - 12.0] * row_count,
            "short_net_bps": [-gross_bps - 12.0] * row_count,
            "long_market_execution_cost_bps": [2.0] * row_count,
            "short_market_execution_cost_bps": [2.0] * row_count,
            "fee_cost_bps": [10.0] * row_count,
            "funding_horizon_bps": [0.0] * row_count,
        }
    )


def test_policy_requires_strictly_better_side_hurdle_and_advantage() -> None:
    predictions = np.asarray(
        [
            [8.0, 1.0],
            [1.0, 8.0],
            [8.0, 8.0],
            [2.9, -10.0],
            [8.0, 6.0],
        ]
    )
    policy = RegressionPolicy(hurdle_bps=3.0, advantage_bps=3.0)

    np.testing.assert_array_equal(
        regression_policy_actions(predictions, policy),
        np.asarray([1, -1, 0, 0, 0], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        regression_policy_actions(predictions, RegressionPolicy(0.0, 0.0, no_trade=True)),
        np.zeros(5, dtype=np.int8),
    )


def test_regression_diagnostics_include_finite_side_pooled_and_decile_metrics() -> None:
    frame = _evaluation_frame(20)
    observed = np.column_stack(
        [
            frame["long_net_bps"].to_numpy(),
            frame["short_net_bps"].to_numpy(),
        ]
    )
    predictions = observed + np.linspace(-1.0, 1.0, 20)[:, None]

    diagnostics = regression_diagnostics(frame, predictions)

    assert set(diagnostics) == {"long", "short", "pooled"}
    assert diagnostics["pooled"]["observations"] == 40
    assert diagnostics["long"]["mae_bps"] >= 0.0
    assert diagnostics["long"]["rmse_bps"] >= 0.0
    assert diagnostics["long"]["spearman"] is None
    assert len(diagnostics["pooled"]["decile_calibration"]) == 10
    assert all(row["count"] == 4 for row in diagnostics["pooled"]["decile_calibration"])


def test_policy_selection_evaluates_fixed_grid_and_requires_positive_lower_bound() -> None:
    frame = _evaluation_frame(80)
    predictions = np.column_stack(
        [
            np.full(frame.height, 20.0),
            np.full(frame.height, -20.0),
        ]
    )

    policy, candidates = choose_regression_policy(
        frame,
        predictions,
        seed=17,
        bootstrap_repetitions=100,
    )

    assert not policy.no_trade
    assert len(candidates) == len(POLICY_HURDLES_BPS) * len(POLICY_ADVANTAGES_BPS)
    assert all(candidate["trades"] == 80 for candidate in candidates)
    assert all(candidate["bootstrap_80_lower_bps"] > 0.0 for candidate in candidates)
    assert policy == RegressionPolicy(hurdle_bps=10.0, advantage_bps=6.0)


def test_policy_selection_returns_no_trade_when_minimum_trade_count_is_not_met() -> None:
    frame = _evaluation_frame(20)
    predictions = np.column_stack(
        [
            np.full(frame.height, 20.0),
            np.full(frame.height, -20.0),
        ]
    )

    policy, _ = choose_regression_policy(
        frame,
        predictions,
        seed=17,
        bootstrap_repetitions=20,
    )

    assert policy.no_trade


def test_fee_counterfactuals_reuse_actions_without_policy_reselection() -> None:
    frame = _evaluation_frame(60)
    actions = np.ones(frame.height, dtype=np.int8)

    scenarios = fee_counterfactuals(
        frame,
        actions,
        execution_cost_multiplier=1.0,
        bootstrap_repetitions=20,
        seed=17,
    )

    assert list(scenarios) == [
        "taker_10_bps",
        "hybrid_7_bps",
        "maker_4_bps",
        "zero_fee",
    ]
    assert {metrics["trades"] for metrics in scenarios.values()} == {60}
    assert scenarios["taker_10_bps"]["fees_bps"] == pytest.approx(600.0)
    assert scenarios["hybrid_7_bps"]["fees_bps"] == pytest.approx(420.0)
    assert (
        scenarios["hybrid_7_bps"]["total_net_bps"]
        - scenarios["taker_10_bps"]["total_net_bps"]
        == pytest.approx(180.0)
    )


def test_non_finite_predictions_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        regression_policy_actions(
            np.asarray([[1.0, np.nan]]),
            RegressionPolicy(0.0, 0.0),
        )
