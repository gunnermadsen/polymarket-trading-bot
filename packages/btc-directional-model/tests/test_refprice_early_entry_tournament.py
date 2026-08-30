from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from btc_directional_model.refprice_context_data import validate_inference_columns
from btc_directional_model.refprice_early_entry_tournament import (
    BINANCE_FEATURES,
    CANDIDATES,
    CHAINLINK_FEATURES,
    CONTROL_FEATURES,
    HISTORY_ARMS,
    POLICIES,
    Calibrator,
    Policy,
    _active_indices,
    _asof_feature,
    _history,
    _wide_base_predictions,
    apply_policy,
    load_config,
    predictive_metrics,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_frozen_tournament_contract() -> None:
    config = load_config(
        _root() / "configs/btc-5m-refprice-early-entry-tournament-20260607-20260829.toml"
    )
    assert config.data_end.isoformat() == "2026-08-29T00:00:00+00:00"
    assert config.economic_end.isoformat() == "2026-08-27T00:00:00+00:00"
    assert config.run_id == "refprice-early-entry-20260829"
    assert config.checkpoint_origin_revision == "fe9f5fcf6e7c57bda0076c22bc4e8c10bd5c0040"
    assert tuple(policy.name for policy in config.policies) == POLICIES
    assert len(config.folds) == 7
    assert CANDIDATES[-1] == "refprice_nonnegative_consensus"
    assert HISTORY_ARMS[-1] == "uncertainty_weighted_hybrid"


def test_all_declared_inference_features_exclude_twap_and_completed_fields() -> None:
    validate_inference_columns(
        tuple(dict.fromkeys(CONTROL_FEATURES + CHAINLINK_FEATURES + BINANCE_FEATURES))
    )


def test_calibrator_is_monotone() -> None:
    values = Calibrator(1.2, -0.1).predict(np.array([0.1, 0.5, 0.9]))
    assert np.all(np.diff(values) > 0)


def test_active_indices_remove_all_missing_and_constant_columns() -> None:
    matrix = np.array(
        [
            [np.nan, 1.0, 0.0],
            [np.nan, 1.0, 1.0],
            [np.nan, 1.0, 0.0],
        ]
    )
    assert _active_indices(matrix) == (2,)


def test_predictive_metrics_reports_brier() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "b"],
            "official_label_up": [1, 0],
            "probability_up": [0.8, 0.2],
        }
    )
    metrics = predictive_metrics(frame)
    assert metrics["brier"] == 0.039999999999999994
    assert metrics["accuracy"] == 1.0


def test_policy_uses_existing_fee_helper_contract() -> None:
    observed = datetime(2026, 8, 21, 0, 1, tzinfo=UTC)
    prediction = pl.DataFrame(
        {
            "market_id": ["a"],
            "window_start": [datetime(2026, 8, 21, tzinfo=UTC)],
            "observed_at": [observed],
            "seconds_elapsed": [60],
            "official_label_up": [1],
            "target_margin_bps": [3.0],
            "probability_up": [0.80],
            "predicted_margin_bps": [4.0],
            "margin_uncertainty_bps": [1.0],
        }
    )
    execution = pl.DataFrame(
        {
            "market_id": ["a"],
            "seconds_elapsed": [60],
            "strict_both_side_eligible": [True],
            "fee_rate": [0.02],
            "up_ask_vwap_5": [0.50],
            "down_ask_vwap_5": [0.51],
        }
    )
    policy = Policy("probability_edge", 0.03, 0.99, 0.005, 0.0, False, 99.0)
    trades = apply_policy(prediction, execution, policy)
    assert trades.height == 1
    assert trades["fee_per_share"][0] == 0.005


def test_asof_feature_never_uses_future_availability() -> None:
    left = pl.DataFrame(
        {
            "observed_at": [
                datetime(2026, 8, 1, 0, 0, 10, tzinfo=UTC),
                datetime(2026, 8, 1, 0, 0, 20, tzinfo=UTC),
            ]
        }
    )
    right = pl.DataFrame(
        {
            "available_at": [
                datetime(2026, 8, 1, 0, 0, 5, tzinfo=UTC),
                datetime(2026, 8, 1, 0, 0, 15, tzinfo=UTC),
            ],
            "price": [100.0, 200.0],
        }
    )
    result = _asof_feature(
        left,
        right,
        left_on="observed_at",
        right_on="available_at",
        columns=("price",),
        prefix="ref_",
    )
    assert result["ref_price"].to_list() == [100.0, 200.0]


def test_history_nulls_are_excluded_and_base_predictions_join_null_labels() -> None:
    starts = pl.datetime_range(
        pl.datetime(2026, 8, 14, time_zone="UTC"),
        pl.datetime(2026, 8, 14, 0, 5, time_zone="UTC"),
        interval="5m",
        eager=True,
    )
    frame = pl.DataFrame(
        {
            "market_id": ["a", "b"],
            "window_start": starts,
            "official_label_up": [None, 1],
            "reconstructed_twap60_label_up": [0, 1],
            "binance_synthetic_label_up": [1, 1],
            "target_margin_bps": [-2.0, 3.0],
        }
    )
    labels, valid, _ = _history(frame, "authentic_only")
    assert valid.tolist() == [True, True]
    assert labels.tolist() == [0, 1]

    prediction_rows = []
    for candidate in CANDIDATES[:3]:
        prediction_rows.append(
            pl.DataFrame(
                {
                    "market_id": ["a"],
                    "window_start": starts[:1],
                    "observed_at": starts[:1],
                    "seconds_elapsed": [60],
                    "official_label_up": [None],
                    "target_margin_bps": [-2.0],
                    "history_arm": ["authentic_only"],
                    "fold": ["f"],
                    "candidate": [candidate],
                    "probability_up": [0.4],
                    "predicted_margin_bps": [-1.0],
                }
            )
        )
    assert _wide_base_predictions(pl.concat(prediction_rows)).height == 1
    assert pl.concat(prediction_rows, how="vertical_relaxed").columns == [
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "official_label_up",
        "target_margin_bps",
        "history_arm",
        "fold",
        "candidate",
        "probability_up",
        "predicted_margin_bps",
    ]
