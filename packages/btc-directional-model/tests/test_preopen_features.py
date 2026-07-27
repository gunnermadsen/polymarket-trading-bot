from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from btc_directional_model.preopen_features import (
    PREOPEN_MODEL_FEATURES,
    _derive_preopen_static_features,
    join_preopen_features,
)


def market_summaries() -> pl.DataFrame:
    start = datetime(2026, 5, 27, tzinfo=UTC)
    return pl.DataFrame(
        {
            "market_id": ["a", "b", "c", "d"],
            "window_start": [start + timedelta(minutes=5 * index) for index in range(4)],
            "open_available_close": [100.0, 101.0, 103.0, 106.0],
            "last_close": [101.0, 103.0, 106.0, 110.0],
            "window_high": [102.0, 104.0, 107.0, 111.0],
            "window_low": [99.0, 100.0, 102.0, 105.0],
            "window_quote_volume": [10.0, 20.0, 30.0, 40.0],
            "window_trade_count": [5.0, 10.0, 15.0, 20.0],
            "window_taker_buy_quote_volume": [6.0, 8.0, 18.0, 10.0],
            "window_volatility_bps": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_preopen_features_use_only_prior_completed_windows() -> None:
    derived = _derive_preopen_static_features(market_summaries())
    row = derived.filter(pl.col("market_id") == "d").row(0, named=True)

    assert row["preopen_return_5m_bps"] > 0
    assert row["preopen_return_15m_bps"] > row["preopen_return_5m_bps"]
    assert row["preopen_log_quote_volume_15m"] == pl.Series(
        [10.0 + 20.0 + 30.0]
    ).log1p()[0]
    assert row["preopen_realized_volatility_5m_bps"] == 3.0
    assert row["preopen_taker_buy_share_5m"] == pytest.approx(18.0 / 30.0)


def test_gap_breaks_preopen_contiguity_instead_of_bridging_history() -> None:
    summaries = market_summaries().with_columns(
        pl.when(pl.col("market_id") == "d")
        .then(pl.col("window_start") + pl.duration(minutes=5))
        .otherwise(pl.col("window_start"))
        .alias("window_start")
    )
    row = (
        _derive_preopen_static_features(summaries)
        .filter(pl.col("market_id") == "d")
        .row(0, named=True)
    )

    assert row["preopen_return_5m_bps"] is None
    assert row["preopen_return_15m_bps"] is None


def test_current_path_interactions_are_point_in_time_features() -> None:
    start = datetime(2026, 5, 27, tzinfo=UTC)
    core = pl.DataFrame(
        {
            "market_id": ["a"],
            "window_start": [start],
            "btc_path_from_window_open_bps": [2.0],
        }
    )
    preopen = pl.DataFrame(
        {
            "market_id": ["a"],
            "window_start": [start],
            "preopen_return_5m_bps": [-3.0],
            "preopen_return_15m_bps": [5.0],
        }
    )

    joined = join_preopen_features(core, preopen)

    assert set(PREOPEN_MODEL_FEATURES[-4:]) <= set(joined.columns)
    assert joined["btc_path_preopen_5m_agreement"][0] == -1
    assert joined["btc_path_preopen_15m_agreement"][0] == 1
    assert joined["btc_path_preopen_5m_reversal"][0] == 1
