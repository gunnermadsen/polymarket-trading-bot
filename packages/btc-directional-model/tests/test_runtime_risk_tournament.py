from datetime import UTC, datetime, timedelta

import polars as pl

from btc_directional_model.runtime_risk_tournament import metrics, selected_trades


def test_risk_replay_uses_first_allowed_candidate_per_market() -> None:
    start = datetime(2026, 8, 21, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "champion": ["strategy", "strategy", "strategy"],
            "market_id": ["a", "a", "b"],
            "observed_at": [start, start + timedelta(seconds=5), start],
            "seconds_elapsed": [60, 65, 60],
            "net_pnl": [-1.0, 0.5, 0.5],
        }
    )
    baseline = selected_trades(frame, None, None, None)
    with_risk = selected_trades(frame, [0.9, 0.1, 0.1], 0.5, "60_89")

    result = metrics(baseline, with_risk)
    assert result["trades"] == 2
    assert result["wins"] == 2
    assert result["pnl"] == 1.0
    assert result["delta_pnl"] == 1.5


def test_risk_replay_does_not_apply_outside_qualified_bucket() -> None:
    frame = pl.DataFrame(
        {
            "champion": ["strategy"],
            "market_id": ["a"],
            "observed_at": [datetime(2026, 8, 21, tzinfo=UTC)],
            "seconds_elapsed": [120],
            "net_pnl": [-1.0],
        }
    )
    kept = selected_trades(frame, [0.99], 0.5, "60_89")
    assert len(kept) == 1
