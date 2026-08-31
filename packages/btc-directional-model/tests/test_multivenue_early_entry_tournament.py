from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from btc_directional_model.multivenue_early_entry_data import (
    ENTRY_SECONDS,
    KRAKEN_FEATURES,
    _attach_kraken,
    _mask_optional_features,
    _preserving_feature_join,
    load_data_config,
)
from btc_directional_model.multivenue_early_entry_tournament import (
    FORBIDDEN_INFERENCE_TOKENS,
    _candidate_contract,
    _matrix,
    _neutralized_feature_indices,
    train_tournament,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_ROOT / "configs/btc-5m-multivenue-early-entry-tournament-20260321-20260828.toml"


def test_frozen_range_and_candidate_roster() -> None:
    config = load_data_config(CONFIG)
    assert ENTRY_SECONDS == tuple(range(60, 241, 5))
    assert config.source_start == datetime(2026, 3, 21, tzinfo=UTC)
    assert config.fit_end == datetime(2026, 8, 14, tzinfo=UTC)
    assert config.sealed_start == config.fit_end
    assert config.sealed_end == datetime(2026, 8, 29, tzinfo=UTC)
    assert config.raw["sources"]["kraken_l2_included"] is False
    assert [row["name"] for row in config.raw["candidates"]] == [
        "full_history_price_control",
        "full_history_refprice_residual",
        "binance_flow",
        "kraken_crossvenue",
        "settlement_aligned_oracle",
        "dual_venue_flow_agreement",
        "multivenue_consensus",
        "time_specialist_ensemble",
    ]


def test_sealed_frame_is_opened_only_after_selection_freeze() -> None:
    source = inspect.getsource(train_tournament)
    assert source.index("selection_frozen_at =") < source.index("sealed = panel.filter")


def test_missing_and_constant_columns_are_neutralized_for_stable_fitting() -> None:
    matrix = np.column_stack(
        (
            np.full(1_100, np.nan),
            np.ones(1_100),
            np.arange(1_100, dtype=float),
        )
    )
    assert _neutralized_feature_indices(matrix) == (0, 1)
    frame = pl.DataFrame({"missing": [None, None], "constant": [1.0, 1.0], "signal": [1.0, 2.0]})
    assert _matrix(frame, ("missing", "constant", "signal"), (0, 1)).shape == (2, 1)


def test_float64_variation_that_collapses_in_float32_is_excluded() -> None:
    matrix = np.linspace(-1.0, -0.9999999999, 1_100).reshape(-1, 1)
    assert _neutralized_feature_indices(matrix) == (0,)


def test_sparse_optional_feature_is_excluded_from_histogram_sampling() -> None:
    matrix = np.full((2_000, 1), np.nan)
    matrix[:999, 0] = np.arange(999, dtype=float)
    assert _neutralized_feature_indices(matrix) == (0,)


def test_candidate_contract_rejects_target_or_rolling_average_inputs() -> None:
    config = load_data_config(CONFIG)
    manifest = {
        "feature_groups": {
            "core": ["btc_return_30s_bps"],
            "candles": ["chainlink_candle_return_5m_bps"],
            "oracle": ["oracle_round_age_seconds_scaled"],
            "refprice": ["chainlink_ref_return_30s_bps"],
            "open_interest": ["binance_oi_change_15m_bps"],
            "binance_prints": ["binance_print_signed_share_30s"],
            "kraken": ["kraken_print_signed_share_30s"],
        }
    }
    contract = _candidate_contract(config, manifest)
    for candidate in contract.values():
        assert not any(
            token in feature.lower()
            for feature in candidate["features"]
            for token in FORBIDDEN_INFERENCE_TOKENS
        )


def test_optional_feature_join_preserves_every_base_row() -> None:
    start = datetime(2026, 4, 1, tzinfo=UTC)
    base = pl.DataFrame(
        {
            "market_id": ["a", "b"],
            "window_start": [start, start + timedelta(minutes=5)],
            "observed_at": [start + timedelta(seconds=60), start + timedelta(minutes=6)],
            "seconds_elapsed": [60, 60],
        }
    )
    subset = base.head(1).with_columns(pl.lit(2.0).alias("optional_signal"))
    joined = _preserving_feature_join(base, subset, ("optional_signal",))
    assert joined.height == base.height
    assert joined["market_id"].to_list() == ["a", "b"]
    assert joined["optional_signal"].to_list() == [2.0, None]


def test_ineligible_optional_values_are_masked_without_dropping_rows() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["before", "after"],
            "causal_eligible": [False, True],
            "external_signal": [99.0, 2.0],
        }
    )
    masked = _mask_optional_features(frame, ("external_signal",), "causal_eligible")
    assert masked.height == frame.height
    assert masked["external_signal"].to_list() == [None, 2.0]


def test_kraken_join_is_backward_asof_and_requires_bucket_close() -> None:
    start = datetime(2026, 4, 1, tzinfo=UTC)
    base = pl.DataFrame(
        {
            "market_id": ["a", "a"],
            "window_start": [start, start],
            "observed_at": [start + timedelta(seconds=60), start + timedelta(seconds=61)],
            "seconds_elapsed": [60, 61],
            "btc_close": [100.0, 100.0],
            "btc_return_30s_bps": [1.0, 1.0],
            "btc_signed_flow_30s": [1.0, 1.0],
        }
    )
    feature_values = {
        name: [1.0] for name in KRAKEN_FEATURES if not name.startswith("kraken_binance_")
    }
    kraken = pl.DataFrame(
        {
            "kraken_available_at": [start + timedelta(seconds=61)],
            "kraken_source_timestamp": [start + timedelta(seconds=60)],
            "kraken_close": [101.0],
            **feature_values,
        }
    )
    joined = _attach_kraken(base, kraken)
    assert joined["kraken_return_30s_bps"].to_list() == [None, 1.0]
    assert joined["kraken_binance_basis_bps"][0] is None
    assert joined["kraken_binance_basis_bps"][1] > 0
