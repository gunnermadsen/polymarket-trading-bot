from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from btc_directional_model.middle_market_ablation_tournament import (
    _apply_policy,
    _probability_metrics,
    _wait_training_frame,
    _wilson_lower,
    load_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "configs" / "btc-5m-middle-market-ablation-tournament-20260525.toml"


def test_ablation_contract_freezes_execution_and_holdout() -> None:
    config = load_config(CONFIG_PATH)

    assert config.entry.start_second == 90
    assert config.entry.end_second_exclusive == 180
    assert config.execution.quantities[0] == 5
    assert config.execution.quantities[-1] == 200
    assert config.execution.maximum_depth_participation == 0.25
    assert config.windows.policy_end == datetime(2026, 8, 10, tzinfo=UTC)
    assert config.windows.holdout_end == datetime(2026, 8, 18, tzinfo=UTC)


def test_wilson_lower_is_conservative() -> None:
    assert _wilson_lower(80, 100) < 0.80
    assert _wilson_lower(0, 0) == 0.0
    assert _wilson_lower(100, 100) > 0.95


def test_wait_target_uses_the_frozen_thirty_second_horizon() -> None:
    config = load_config(CONFIG_PATH)
    frame = pl.DataFrame(
        {
            "market_id": ["m"] * 7,
            "seconds_elapsed": [90, 95, 100, 105, 110, 115, 120],
            "direction_correct": [True, False, False, False, False, False, True],
            "selected_cost_5": [0.60] * 7,
        }
    )

    result = _wait_training_frame(frame, config)

    row = result.filter(pl.col("seconds_elapsed") == 90)
    assert row.height == 1
    assert row["enter_now_advantage_target"].item() == 0.0


def test_policy_takes_first_qualifying_crossing_per_market() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "b"],
            "lower_correctness_probability": [0.70, 0.80, 0.90],
            "stress_edge_lower_bound": [0.02, 0.03, 0.04],
            "loss_severity_prediction": [0.10, 0.10, 0.10],
            "wait_advantage": [0.0, 0.0, 0.0],
        }
    )
    policy = {
        "confidence": 0.65,
        "stress_edge": 0.0,
        "loss_severity": 0.20,
        "wait_advantage": 0.0,
    }

    result = _apply_policy(frame, policy)

    assert result.height == 2
    assert result["lower_correctness_probability"].to_list() == [0.70, 0.90]


def test_probability_metrics_include_calibration_error() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "b"],
            "probability_up": [0.8, 0.2],
            "label_up": [1, 0],
        }
    )

    metrics = _probability_metrics(frame)

    assert metrics["accuracy"] == 1.0
    assert abs(metrics["brier"] - 0.04) < 1e-12
    assert abs(metrics["ece"] - 0.2) < 1e-12
