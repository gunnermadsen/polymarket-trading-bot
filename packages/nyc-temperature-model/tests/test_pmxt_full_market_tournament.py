from datetime import UTC, date, datetime, timedelta

import numpy as np

from nyc_temperature_model.pmxt_full_market_tournament import (
    MODEL_FEATURE_SETS,
    _trade_features,
    _weighted_probability_metrics,
    fit_offset_model,
    predict_offset_model,
    select_trades,
)


def _candidate(event_date: date, side: str, cost: float, resolved: bool) -> dict:
    return {
        "event_date": event_date,
        "side": side,
        "all_in_cost_per_share": cost,
        "ask_vwap": cost - 0.01,
        "executable": True,
        "resolved_side": resolved,
        "realized_net_per_share": (1.0 if resolved else 0.0) - cost,
        "fees_enabled": False,
        "fee_rate": 0.0,
        "fee_exponent": 1.0,
    }


def _row(index: int, *, signal: float | None = None) -> dict:
    event_date = date(2026, 4, 1) + timedelta(days=index)
    decision_time = datetime.combine(event_date, datetime.min.time(), UTC)
    value = (-1.0 if index % 2 else 1.0) if signal is None else signal
    market_probability = 0.5
    resolved_yes = value > 0
    features = {name: 0.0 for names in MODEL_FEATURE_SETS.values() for name in names}
    features["weather_residual_midnight"] = value
    features["pmxt_outcome_size_imbalance_60m"] = value
    return {
        "event_date": event_date,
        "decision_time": decision_time,
        "decision_hour_local": 0,
        "market_id": f"market-{index}",
        "resolved_yes": resolved_yes,
        "weather_probability_yes": 0.7 if resolved_yes else 0.3,
        "market_probability_yes": market_probability,
        "features": features,
        "coverage_manifest": "a" * 64,
        "candidates": {
            "YES": _candidate(event_date, "YES", 0.60, resolved_yes),
            "NO": _candidate(event_date, "NO", 0.40, not resolved_yes),
        },
    }


def test_trade_features_are_causal_and_preserve_true_zero_windows():
    decision = datetime(2026, 6, 1, 12, tzinfo=UTC)
    trades = [
        {
            "outcome": "YES",
            "source_timestamp": decision - timedelta(minutes=4),
            "provider_received_at": decision - timedelta(minutes=3),
            "price": 0.25,
            "size": 10.0,
            "trade_side": "BUY",
        },
        {
            "outcome": "YES",
            "source_timestamp": decision + timedelta(seconds=1),
            "provider_received_at": decision + timedelta(seconds=1),
            "price": 0.95,
            "size": 1000.0,
            "trade_side": "BUY",
        },
    ]

    features = _trade_features(trades, decision, 0.20)

    assert np.isclose(features["pmxt_log_trade_count_5m"], np.log1p(1))
    assert np.isclose(features["pmxt_yes_last_market_divergence"], 0.05)
    assert features["pmxt_no_trade_missing"] == 1.0

    empty = _trade_features([], decision, 0.20)
    assert empty["pmxt_log_trade_count_60m"] == 0.0
    assert empty["pmxt_yes_trade_missing"] == 1.0
    assert empty["pmxt_no_trade_missing"] == 1.0


def test_regularized_offset_learns_incremental_signal_over_market():
    rows = [_row(index) for index in range(40)]
    fit = fit_offset_model(
        rows,
        ("weather_residual_midnight",),
        regularization=0.1,
    )
    predictions = predict_offset_model(fit, rows)
    market = np.asarray([row["market_probability_yes"] for row in rows])

    assert fit.converged
    assert fit.coefficients[1] > 0
    assert _weighted_probability_metrics(rows, predictions)["binary_log_loss"] < (
        _weighted_probability_metrics(rows, market)["binary_log_loss"]
    )


def test_full_market_policy_is_a_price_cap_ablation_of_same_predictions():
    rows = [_row(0), _row(1)]
    rows[0]["candidates"]["YES"]["all_in_cost_per_share"] = 0.80
    rows[0]["candidates"]["YES"]["ask_vwap"] = 0.79
    rows[1]["candidates"]["NO"]["all_in_cost_per_share"] = 0.20
    rows[1]["candidates"]["NO"]["ask_vwap"] = 0.19
    point = np.asarray([0.95, 0.10])
    lower = np.asarray([0.90, 0.05])
    upper = np.asarray([0.97, 0.15])

    capped = select_trades(
        rows,
        point,
        lower,
        upper,
        edge_threshold=0.05,
        maximum_cost=0.25,
    )
    full = select_trades(
        rows,
        point,
        lower,
        upper,
        edge_threshold=0.05,
        maximum_cost=0.999999,
    )

    assert len(capped) == 1
    assert len(full) == 2
    assert any(trade["all_in_cost_per_share"] == 0.80 for trade in full)


def test_no_side_uses_the_complementary_upper_probability_bound():
    row = _row(0)
    row["candidates"]["YES"]["all_in_cost_per_share"] = 0.70
    row["candidates"]["NO"]["all_in_cost_per_share"] = 0.20

    trades = select_trades(
        [row],
        np.asarray([0.30]),
        np.asarray([0.10]),
        np.asarray([0.40]),
        edge_threshold=0.30,
        maximum_cost=1.0,
    )

    assert len(trades) == 1
    assert trades[0]["side"] == "NO"
    assert np.isclose(trades[0]["probability_lower"], 0.60)
