from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from btc_directional_model.settlement_bridge_residual_tournament import (
    BASE_FEATURES,
    CANDIDATES,
    INFERENCE_FEATURES,
    RELATIVE_TWAP_FEATURES,
    SUPERVISION_FIELDS,
    _attach_capacity,
    _causal_piecewise_average,
    _market_equal_weights,
    base_specs,
    causal_feature_registry,
    load_config,
    residual_specs,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/btc-5m-settlement-bridge-residual-20260321-20260828.toml"


def test_contract_freezes_exact_candidates_windows_and_searches() -> None:
    config = load_config(CONFIG)

    assert CANDIDATES == (
        "refprice_bridge_baseline",
        "non_twap_settlement_correction",
        "relative_twap_settlement_correction",
        "twap_margin_residual_bridge",
    )
    assert config.candidate_freeze == datetime(2026, 8, 28, tzinfo=UTC)
    assert config.windows.reconstruction_start == datetime(2026, 6, 7, tzinfo=UTC)
    assert config.windows.paired_training_end == datetime(2026, 8, 1, tzinfo=UTC)
    assert len(base_specs(config)) == 12
    assert len(base_specs(config)) <= 36
    assert len(residual_specs(config)) == 4
    assert len(residual_specs(config)) <= 12


def test_inference_contract_excludes_supervision_and_date_shortcuts() -> None:
    assert not set(INFERENCE_FEATURES) & set(SUPERVISION_FIELDS)
    assert not set(INFERENCE_FEATURES) & {"window_start", "market_date", "label_source", "regime"}
    assert "hour_sin" not in BASE_FEATURES
    assert "weekday_sin" not in BASE_FEATURES


def test_every_inference_feature_has_causal_registry_metadata() -> None:
    registry = causal_feature_registry()
    by_name = {row["feature"]: row for row in registry}

    assert set(by_name) == set(INFERENCE_FEATURES)
    for row in registry:
        assert row["source_event_timestamp"]
        assert row["source_availability_timestamp"]
        assert row["lookback"]
        assert row["feature_as_of"]
        assert row["live_computable"] is True


def test_causal_twap_uses_only_reports_available_before_target() -> None:
    start = np.datetime64("2026-06-07T00:00:00", "us").astype(np.int64)
    source = start + np.array([0, 30, 60, 90], dtype=np.int64) * 1_000_000
    available = source + 1_000_000
    # The 90-second report is deliberately unavailable until after the target.
    available[-1] = start + 200 * 1_000_000
    prices = np.array([100.0, 110.0, 120.0, 1_000_000.0])
    target = np.array([start + 120 * 1_000_000], dtype=np.int64)

    result = _causal_piecewise_average(source, available, prices, target, 60)

    assert np.allclose(result, [120.0])


def test_market_equal_weights_do_not_overweight_observation_count() -> None:
    frame = pl.DataFrame({"market_id": ["a", "a", "b"]})

    weights = _market_equal_weights(frame)

    assert np.isclose(weights[:2].sum(), weights[2])


def test_read_only_queries_are_bounded_and_have_no_mutations() -> None:
    sql = (ROOT / "sql/btc-settlement-bridge-binance-label-diagnostic.sql").read_text().lower()
    assert "batch_start" in sql and "batch_end" in sql
    assert not any(token in sql for token in (
        "insert ", "update ", "delete ", "create table", "alter table", "drop table"
    ))


def test_relative_twap_roster_is_complete_and_contains_no_absolute_price() -> None:
    assert len(RELATIVE_TWAP_FEATURES) == 13
    assert not any(name in {"price", "btc_price", "twap30", "twap60"} for name in RELATIVE_TWAP_FEATURES)


def test_empty_filtered_capacity_preserves_prediction_rows_as_ineligible() -> None:
    instant = datetime(2026, 8, 26, tzinfo=UTC)
    frame = pl.DataFrame({"market_id": ["m"], "observed_at": [instant]})
    capacity = pl.DataFrame(
        {
            "market_id": ["m"],
            "observed_at": [instant],
            "seconds_elapsed": [60],
            "quality_flags": [1],
            "up_provider_received_at": [instant],
            "down_provider_received_at": [instant],
        }
    )

    result = _attach_capacity(frame, capacity, 10)

    assert result.height == 1
    assert result["up_ask_vwap_5"].null_count() == 1
