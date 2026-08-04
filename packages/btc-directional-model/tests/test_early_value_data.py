from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from btc_directional_model.early_value_data import build_strict_external_frame
from btc_directional_model.spot_l2_chainlink_features import L2_SOURCE_FEATURE_COLUMNS


def test_external_join_uses_only_strictly_prior_availability() -> None:
    observed = datetime(2026, 7, 20, 0, 0, 5, tzinfo=UTC)
    core = pl.DataFrame({"market_id": ["m"], "window_start": [observed - timedelta(seconds=5)], "observed_at": [observed], "seconds_elapsed": [5], "btc_close": [100.0]})
    values = {name: [1.0] for name in L2_SOURCE_FEATURE_COLUMNS}
    values.update({
        "symbol": ["BTCUSDT"], "second_start": [observed - timedelta(seconds=1)],
        "source_event_timestamp": [observed - timedelta(seconds=1)],
        "provider_received_at": [observed - timedelta(milliseconds=500)],
        "available_at": [observed - timedelta(milliseconds=250)], "source_update_id": [1],
    })
    values["midpoint"] = [100.0]
    values["microprice"] = [100.0]
    for name in ("bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10", "bid_depth_20", "ask_depth_20"):
        values[name] = [10.0]
    final_close = observed - timedelta(seconds=1)
    closes = [final_close - timedelta(minutes=60 - index) for index in range(61)]
    candles = pl.DataFrame({
        "open_timestamp": [value - timedelta(minutes=1) for value in closes],
        "close_timestamp": closes,
        "available_at": closes,
        "open_price": [99.0] * 61, "high_price": [101.0] * 61,
        "low_price": [98.0] * 61, "close_price": [100.0] * 61,
    })
    result = build_strict_external_frame(core, pl.DataFrame(values), candles)
    assert result.height == 1
    assert result["seconds_elapsed"].to_list() == [5]


def test_price_query_matches_canonical_checkpoint_schema() -> None:
    query = (
        Path(__file__).parents[1] / "sql/btc-early-value-book-source.sql"
    ).read_text()
    assert "official_outcome IN ('up', 'down')" in query
    assert "checkpoint.source_timestamp AS snapshot_at" in query
    assert "checkpoint.snapshot_at" not in query
