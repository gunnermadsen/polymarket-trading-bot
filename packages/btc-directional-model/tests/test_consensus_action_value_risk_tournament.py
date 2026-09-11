from datetime import UTC, datetime

import polars as pl

from btc_directional_model.consensus_action_value_risk_tournament import (
    attach_action_value_targets,
    attach_consensus_features,
)


def test_action_value_wait_never_crosses_bucket() -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "champion": ["s", "s", "s"],
            "market_id": ["a", "a", "a"],
            "window_start": [start] * 3,
            "seconds_elapsed": [80, 85, 90],
            "net_pnl": [-2.0, 1.0, 4.0],
        }
    )
    result = attach_action_value_targets(frame, 0.1).sort("seconds_elapsed")
    assert result["future_best_within_bucket_pnl"].to_list() == [1.0, 0.0, 0.0]
    assert result["wait_value"].to_list() == [0.9, 0.0, 0.0]
    assert result["optimal_action"].to_list() == ["wait", "enter_now", "enter_now"]


def test_consensus_features_use_same_timestamp_strategy_outputs() -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "champion": ["a", "b", "a"],
            "market_id": ["m", "m", "m"],
            "window_start": [start] * 3,
            "seconds_elapsed": [60, 60, 65],
            "side": ["UP", "DOWN", "UP"],
            "selected_probability": [0.8, 0.6, 0.7],
            "confidence": [0.8, 0.6, 0.7],
            "expected_edge": [0.2, 0.1, 0.15],
        }
    )
    result = attach_consensus_features(frame).sort(["seconds_elapsed", "champion"])
    assert result["consensus_strategy_count"].to_list() == [2.0, 2.0, 1.0]
    assert result["consensus_probability_mean"].to_list() == [0.7, 0.7, 0.7]
    assert result["consensus_up_share"].to_list() == [0.5, 0.5, 1.0]
    assert result["strategy_side_consensus"].to_list() == [0.5, 0.5, 1.0]
