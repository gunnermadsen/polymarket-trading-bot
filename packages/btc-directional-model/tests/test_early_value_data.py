from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from btc_directional_model.chainlink_oi_features import CHAINLINK_CANDLE_FEATURES
from btc_directional_model.early_value_data import (
    build_full_closed_candle_frame,
    build_partitioned_l2_frame,
    build_strict_external_frame,
)
from btc_directional_model.spot_l2_chainlink_features import (
    L2_FEATURES,
    L2_SOURCE_FEATURE_COLUMNS,
)


def _core(observed: datetime, *, second_observed: datetime | None = None) -> pl.DataFrame:
    observations = [observed, *([second_observed] if second_observed else [])]
    return pl.DataFrame(
        {
            "market_id": [f"m-{index}" for index in range(len(observations))],
            "window_start": [value - timedelta(seconds=5) for value in observations],
            "observed_at": observations,
            "seconds_elapsed": [5] * len(observations),
            "btc_close": [100.0] * len(observations),
        }
    )


def _l2(observed: datetime) -> pl.DataFrame:
    values = {name: [1.0] for name in L2_SOURCE_FEATURE_COLUMNS}
    values.update(
        {
            "symbol": ["BTCUSDT"],
            "second_start": [observed - timedelta(seconds=1)],
            "source_event_timestamp": [observed - timedelta(seconds=1)],
            "provider_received_at": [observed - timedelta(milliseconds=500)],
            "available_at": [observed - timedelta(milliseconds=250)],
            "source_update_id": [1],
        }
    )
    values["midpoint"] = [100.0]
    values["microprice"] = [100.0]
    for name in (
        "bid_depth_5",
        "ask_depth_5",
        "bid_depth_10",
        "ask_depth_10",
        "bid_depth_20",
        "ask_depth_20",
    ):
        values[name] = [10.0]
    return pl.DataFrame(values)


def _candles(observed: datetime) -> pl.DataFrame:
    final_close = observed - timedelta(seconds=1)
    closes = [final_close - timedelta(minutes=60 - index) for index in range(61)]
    return pl.DataFrame(
        {
            "open_timestamp": [value - timedelta(minutes=1) for value in closes],
            "close_timestamp": closes,
            "available_at": closes,
            "open_price": [99.0] * 61,
            "high_price": [101.0] * 61,
            "low_price": [98.0] * 61,
            "close_price": [100.0] * 61,
        }
    )


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


def test_partitioned_l2_builder_has_no_candle_dependency(tmp_path: Path) -> None:
    observed = datetime(2026, 7, 20, 0, 0, 5, tzinfo=UTC)
    next_day = observed + timedelta(days=1)
    _l2(observed).write_parquet(tmp_path / "2026-07-20.parquet")
    _l2(next_day).write_parquet(tmp_path / "2026-07-21.parquet")

    core = _core(observed, second_observed=next_day)
    result = build_partitioned_l2_frame(core, tmp_path)
    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]

    assert result.select(*keys).equals(core.select(*keys))
    assert set(L2_FEATURES).issubset(result.columns)
    assert set(CHAINLINK_CANDLE_FEATURES).isdisjoint(result.columns)


def test_full_candle_builder_preserves_keys_without_l2(tmp_path: Path) -> None:
    observed = datetime(2026, 7, 20, 0, 0, 5, tzinfo=UTC)
    core = _core(observed, second_observed=observed + timedelta(seconds=5))
    _candles(observed).write_parquet(tmp_path / "2026-07-20.parquet")

    result = build_full_closed_candle_frame(core, tmp_path)
    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]

    assert result.select(*keys).equals(core.select(*keys))
    assert set(CHAINLINK_CANDLE_FEATURES).issubset(result.columns)
    assert set(L2_FEATURES).isdisjoint(result.columns)


def test_full_candle_builder_fails_if_a_core_key_lacks_context(
    tmp_path: Path,
) -> None:
    observed = datetime(2026, 7, 20, 0, 0, 5, tzinfo=UTC)
    core = _core(observed, second_observed=observed + timedelta(minutes=2))
    _candles(observed).write_parquet(tmp_path / "2026-07-20.parquet")

    with pytest.raises(RuntimeError, match="changed core decision-point identity"):
        build_full_closed_candle_frame(core, tmp_path)


def test_price_query_matches_canonical_checkpoint_schema() -> None:
    query = (
        Path(__file__).parents[1] / "sql/btc-early-value-book-source.sql"
    ).read_text()
    assert "official_outcome IN ('up', 'down')" in query
    assert "checkpoint.source_timestamp AS snapshot_at" in query
    assert "checkpoint.snapshot_at" not in query
