from datetime import UTC, datetime, timedelta

import polars as pl

from btc_directional_model.candidate_loss_risk_tournament import (
    _history,
    sequential_replay,
    trade_metrics,
)


def _rows() -> pl.DataFrame:
    start = datetime(2026, 8, 14, tzinfo=UTC)
    return pl.DataFrame(
        {
            "champion": ["a", "a", "a"],
            "market_id": ["m1", "m1", "m2"],
            "window_start": [start, start, start + timedelta(minutes=5)],
            "observed_at": [start + timedelta(seconds=x) for x in (60, 90, 360)],
            "seconds_elapsed": [60, 90, 60],
            "probability": [0.7, 0.8, 0.6],
            "selected_probability": [0.7, 0.8, 0.6],
            "share_cost": [0.6, 0.7, 0.5],
            "fee_per_share": [0.0, 0.0, 0.0],
            "expected_edge": [0.1, 0.1, 0.1],
            "direction_correct": [False, True, True],
            "net_pnl": [-3.0, 1.5, 2.5],
            "stress_net_pnl": [-3.05, 1.45, 2.45],
            "time_bucket": ["60_89", "90_119", "60_89"],
            "side": ["UP", "UP", "UP"],
            "loss_label": [1, 0, 0],
        }
    )


def test_sequential_replay_allows_later_candidate_after_deferral() -> None:
    result = sequential_replay(_rows(), [False, True, True])
    assert result["seconds_elapsed"].to_list() == [90, 60]


def test_history_is_strictly_prior_market() -> None:
    result = _history(_rows())
    first = result.filter(pl.col("market_id") == "m1")
    second = result.filter(pl.col("market_id") == "m2")
    assert first["history_win_rate_5"].null_count() == first.height
    assert second["history_loss_streak"].item() == 1
    assert second["history_win_rate_5"].item() == 0.0


def test_trade_metrics_recovery_ratio_and_drawdown() -> None:
    metrics = trade_metrics(sequential_replay(_rows(), [True, True, True]), 2)
    assert metrics["wins"] == 1
    assert metrics["losses"] == 1
    assert metrics["net_pnl"] == -0.5
    assert metrics["recovery_wins_per_loss"] == 1.2
