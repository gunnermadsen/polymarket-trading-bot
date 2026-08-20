from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from btc_directional_model.middle_market_tournament import (
    TRADE_PRINT_FEATURES,
    _bucket_indices,
    _candidate_eligible_frame,
    _derive_trade_print_features,
    _join_trade_print_features,
    load_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_tournament_contract_loads_with_frozen_middle_window() -> None:
    config = load_config(
        PACKAGE_ROOT / "configs" / "btc-5m-middle-market-payoff-tournament-20260525-20260810.toml"
    )

    assert config.entry.start_second == 90
    assert config.entry.end_second_exclusive == 180
    assert config.execution.quantities[0] == 5
    assert config.execution.quantities[-1] == 200
    assert config.windows.policy_end == datetime(2026, 8, 2, tzinfo=UTC)
    assert config.windows.holdout_end == datetime(2026, 8, 10, tzinfo=UTC)


def test_price_bucket_indices_include_outer_edges() -> None:
    values = np.asarray([0.0, 0.649, 0.65, 0.75, 0.85, 1.0])

    result = _bucket_indices(values, (0.0, 0.65, 0.75, 0.85, 1.01))

    assert result.tolist() == [0, 0, 1, 2, 3, 3]


def test_core_candidates_keep_rows_with_causally_unavailable_long_horizons() -> None:
    frame = pl.DataFrame(
        {
            "core": [1.0, None],
            "optional": [2.0, None],
        }
    )

    assert _candidate_eligible_frame(frame, ()).height == 2
    assert _candidate_eligible_frame(frame, ("optional",)).height == 1


def test_trade_print_features_are_causal_and_stale_values_are_nulled() -> None:
    start = datetime(2026, 7, 22, tzinfo=UTC)
    source = pl.DataFrame(
        {
            "second_start": [start + timedelta(seconds=index) for index in range(61)],
            "quote_volume": [100.0] * 61,
            "base_volume": [1.0] * 61,
            "signed_taker_quote_volume": [25.0] * 61,
            "trade_count": [10] * 61,
            "trade_vwap": [100.0 + index for index in range(61)],
        }
    )
    features = _derive_trade_print_features(source)
    frame = pl.DataFrame(
        {
            "observed_at": [
                start + timedelta(seconds=61, microseconds=500_000),
                start + timedelta(seconds=65),
            ]
        }
    )

    joined = _join_trade_print_features(frame, features)

    assert joined.height == 2
    assert joined[TRADE_PRINT_FEATURES[0]][0] is not None
    assert all(joined[name][1] is None for name in TRADE_PRINT_FEATURES)
