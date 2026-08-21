from datetime import UTC, date, datetime

import numpy as np

from nyc_temperature_model.asymmetric_benchmark import (
    AsymmetricPolicy,
    _circular_block_indices,
    _probability_estimates,
    _select_policy_trades,
    _selection_key,
    policy_frontier,
)


def _candidate(
    *,
    market_id: str,
    hour: int,
    cost: float,
    probability: float,
    probability_lower: float,
    resolved: bool = False,
):
    edge = probability - cost
    robust_edge = probability_lower - cost
    return {
        "model_run_id": f"model-{hour}",
        "market_id": market_id,
        "event_date": date(2026, 7, 1),
        "decision_time": datetime(2026, 7, 1, 4 if hour == 0 else 16, tzinfo=UTC),
        "decision_hour_local": hour,
        "side": "YES",
        "quantity": 5.0,
        "probability": probability,
        "probability_lower": probability_lower,
        "ask_vwap": cost,
        "all_in_cost_per_share": cost,
        "model_edge_per_share": edge,
        "robust_edge_per_share": robust_edge,
        "expected_roi": edge / cost,
        "robust_expected_roi": robust_edge / cost,
        "resolved_side": resolved,
        "realized_net_per_share": (1.0 if resolved else 0.0) - cost,
        "executable": True,
        "rejection_reasons": [],
    }


def test_frontier_is_restrained_to_nine_predeclared_policies():
    assert len(policy_frontier()) == 9


def test_low_price_value_is_selected_over_high_accuracy_expensive_share():
    policy = AsymmetricPolicy(
        name="low-price",
        decision_mode="midnight_only",
        minimum_all_in_cost=0.04,
        maximum_all_in_cost=0.25,
        minimum_robust_edge=0.03,
        minimum_robust_roi=0.35,
    )
    low_price = _candidate(
        market_id="low", hour=0, cost=0.12, probability=0.24, probability_lower=0.20
    )
    expensive = _candidate(
        market_id="expensive", hour=0, cost=0.92, probability=0.90, probability_lower=0.88
    )

    selected, reasons = _select_policy_trades([low_price, expensive], policy)

    assert [row["market_id"] for row in selected] == ["low"]
    assert "all_in_cost_outside_policy" in reasons[
        ("model-0", "expensive", expensive["decision_time"], "YES")
    ]


def test_sequential_policy_keeps_qualifying_midnight_entry_even_if_noon_is_better():
    policy = AsymmetricPolicy(
        name="sequential",
        decision_mode="midnight_then_noon",
        minimum_all_in_cost=0.04,
        maximum_all_in_cost=0.25,
        minimum_robust_edge=0.03,
        minimum_robust_roi=0.35,
    )
    midnight = _candidate(
        market_id="midnight", hour=0, cost=0.12, probability=0.24, probability_lower=0.20
    )
    noon = _candidate(
        market_id="noon", hour=12, cost=0.08, probability=0.35, probability_lower=0.30
    )

    selected, _ = _select_policy_trades([midnight, noon], policy)

    assert [row["market_id"] for row in selected] == ["midnight"]


def test_probability_bounds_cover_yes_and_no_without_accuracy_gate():
    residuals = np.asarray([-2, -1, 0, 1, 2] * 20, dtype=float)
    indices = _circular_block_indices(
        residuals.size, iterations=200, block_size=7, seed=11
    )
    estimates = _probability_estimates(
        70.0,
        residuals,
        [(None, 68), (69, 71), (72, None)],
        indices,
    )

    assert np.isclose(sum(row["yes"] for row in estimates), 1.0)
    assert all(0 <= row["yes_lower"] <= row["yes"] <= 1 for row in estimates)
    assert all(0 <= row["no_lower"] <= row["no"] <= 1 for row in estimates)


def test_frontier_selection_never_prefers_nan_zero_trade_result():
    empty = {
        "policy": {"maximum_all_in_cost": 0.16, "name": "empty"},
        "metrics": {
            "trades": 0,
            "chronological_folds": [{"trades": 0}] * 3,
            "lower_90pct_block_bootstrap_mean_daily_return": float("nan"),
            "mean_daily_return_on_debit": float("nan"),
            "maximum_drawdown": 0.0,
        },
    }
    observed = {
        "policy": {"maximum_all_in_cost": 0.25, "name": "observed"},
        "metrics": {
            "trades": 3,
            "chronological_folds": [{"trades": 1}] * 3,
            "lower_90pct_block_bootstrap_mean_daily_return": 0.01,
            "mean_daily_return_on_debit": 0.02,
            "maximum_drawdown": 1.0,
        },
    }

    assert max([empty, observed], key=_selection_key) is observed
