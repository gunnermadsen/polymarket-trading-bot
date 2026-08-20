from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl

from btc_directional_model.continuous_edge_training import (
    VWAP_QUANTITIES,
    ExecutionConfig,
    PathConfig,
    PolicyConfig,
    TimeBand,
    TrainingConfig,
    WindowConfig,
    _first_crossings_array,
    _price_bucket_indices,
    market_equal_weights,
    policy_metrics,
    rolling_policy_metrics,
)


def _config(tmp_path) -> TrainingConfig:
    instant = datetime(2026, 1, 1, tzinfo=UTC)
    return TrainingConfig(
        source_path=tmp_path / "config.toml",
        package_root=tmp_path,
        profile="test",
        random_seed=1,
        windows=WindowConfig(
            instant,
            instant,
            instant,
            instant,
            instant,
            instant,
            instant,
        ),
        bands=(TimeBand("early", 15, 90, 0.5, 1),),
        execution=ExecutionConfig(VWAP_QUANTITIES, 10, 0.25, 0.005, 0.01),
        policy=PolicyConfig(
            (0.5,),
            (0.0,),
            (0.5,),
            (0.0,),
            (0.0, 0.65, 0.75, 0.85, 1.01),
            (0.0, 0.005, 0.015, 0.04),
            3,
            0.75,
            1.10,
            1.0,
        ),
        paths=PathConfig(*(tmp_path for _ in range(8))),
    )


def test_market_equal_weights_give_each_market_equal_mass() -> None:
    frame = pl.DataFrame({"market_id": ["a", "a", "a", "b"]})
    weights = market_equal_weights(frame)

    assert np.isclose(weights[:3].sum(), weights[3:].sum())
    assert np.isclose(weights.mean(), 1.0)


def test_first_crossing_uses_earliest_qualified_row_per_market() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "a", "b", "b"],
            "seconds_elapsed": [15, 16, 17, 15, 16],
            "probability_selected": [0.60, 0.80, 0.90, 0.70, 0.80],
            "conservative_probability_selected": [0.60, 0.80, 0.90, 0.70, 0.80],
            "conservative_edge_5": [0.00, 0.03, 0.05, 0.04, 0.05],
            "price_bucket_minimum_edge": [0.0] * 5,
            "admission_probability": [0.90, 0.90, 0.90, 0.40, 0.90],
            "payoff_expected_stress_edge": [0.05] * 5,
        }
    )

    selected = _first_crossings_array(
        frame,
        confidence=0.75,
        edge=0.02,
        admission=0.5,
        use_admission=True,
    )

    assert selected["seconds_elapsed"].to_list() == [16, 16]


def test_policy_metrics_use_exact_vwap_and_fee(tmp_path) -> None:
    values = {
        "predicted_up": [True, False],
        "label_up": [1, 1],
        "fee_rate": [0.0, 0.0],
        "seconds_elapsed": [20, 40],
        "window_start": [
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
        ],
    }
    for quantity in VWAP_QUANTITIES:
        values[f"up_ask_vwap_{quantity}"] = [0.60, 0.60]
        values[f"down_ask_vwap_{quantity}"] = [0.40, 0.40]
    frame = pl.DataFrame(values)

    metrics = policy_metrics(frame, _config(tmp_path), quantity=5)

    assert metrics["trades"] == 2
    assert metrics["accuracy"] == 0.5
    assert np.isclose(metrics["net_pnl"], -0.05)
    assert metrics["average_entry_second"] == 30.0


def test_price_bucket_indices_apply_expensive_entry_bucket() -> None:
    values = np.array([0.10, 0.65, 0.749, 0.75, 0.85, 0.99])

    result = _price_bucket_indices(values, (0.0, 0.65, 0.75, 0.85, 1.01))

    assert result.tolist() == [0, 1, 1, 2, 3, 3]


def test_rolling_metrics_count_empty_chronological_fold(tmp_path) -> None:
    config = _config(tmp_path)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "window_start": [start],
            "predicted_up": [True],
            "label_up": [1],
            "fee_rate": [0.0],
            "seconds_elapsed": [20],
            **{
                f"{side}_ask_vwap_{quantity}": [0.60 if side == "up" else 0.40]
                for quantity in VWAP_QUANTITIES
                for side in ("up", "down")
            },
        }
    )

    metrics = rolling_policy_metrics(
        frame,
        config,
        start,
        datetime(2026, 1, 7, tzinfo=UTC),
        quantity=5,
    )

    assert metrics["fold_count"] == 2
    assert metrics["profitable_folds"] == 1
    assert metrics["profitable_fold_ratio"] == 0.5
