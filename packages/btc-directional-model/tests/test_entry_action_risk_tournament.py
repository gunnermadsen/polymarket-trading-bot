from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from btc_directional_model.entry_action_risk_tournament import (
    attach_action_targets,
    intervention_metrics,
)


def _frame() -> pl.DataFrame:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    return pl.DataFrame({
        "champion": ["s", "s", "s", "s"],
        "market_id": ["a", "a", "b", "b"],
        "window_start": [start, start, start + timedelta(minutes=5), start + timedelta(minutes=5)],
        "observed_at": [start + timedelta(seconds=x) for x in (60, 90, 360, 390)],
        "seconds_elapsed": [60, 90, 60, 90],
        "net_pnl": [-3.0, 2.0, 1.0, -2.0],
        "stress_net_pnl": [-3.05, 1.95, 0.95, -2.05],
        "share_cost": [0.6, 0.5, 0.4, 0.6],
    })


def test_action_targets_use_strictly_later_candidate_and_abstention() -> None:
    rows = attach_action_targets(_frame(), 0.1).sort(["market_id", "seconds_elapsed"])
    assert rows["future_best_net_pnl"].to_list() == [2.0, 0.0, -2.0, 0.0]
    assert rows["defer_value"].to_list() == [1.9, 0.0, 0.0, 0.0]
    assert rows["optimal_action"].to_list() == ["defer", "allow_now", "allow_now", "abstain"]


def test_action_replay_measures_deferral_separately_from_strategy_pnl() -> None:
    frame = attach_action_targets(_frame(), 0.1)
    result = intervention_metrics(frame, np.array([0.0, 1.0, 1.0, 0.0]), 0.5)
    assert result["deferred_markets"] == 1
    assert result["beneficial_deferrals"] == 1
    assert result["blocked_losses"] == 1
    assert result["blocked_winners"] == 0
    assert result["net_risk_value"] == 5.0
