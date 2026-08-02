from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from btc_directional_model.chainlink_oi_features import (
    BINANCE_OI_FEATURES,
    CHAINLINK_CANDLE_FEATURES,
    CHAINLINK_EXTERNAL_FEATURES,
    CHAINLINK_OI_FEATURES,
    EXTERNAL_CORE_REQUIRED_COLUMNS,
    POINT_KEY_COLUMNS,
    derive_chainlink_candle_feature_frame,
    derive_chainlink_oi_feature_frames,
)
from btc_directional_model.core_features import derive_core_point_in_time_features

DECISION = datetime(2026, 7, 20, 12, tzinfo=UTC)


def _core_frame(*, include_second_point: bool = True) -> pl.DataFrame:
    offsets = (0, 5) if include_second_point else (0,)
    return pl.DataFrame(
        [
            {
                "market_id": "btc-updown-5m-20260720-1200",
                "window_start": DECISION,
                "seconds_elapsed": 120 + offset,
                "observed_at": DECISION + timedelta(seconds=offset),
                "btc_close": 100_010.0 + offset,
                "opening_boundary": 100_000.0,
                "btc_return_30s_bps": 2.5,
                "btc_path_from_window_open_bps": 4.0,
                "label_up": 1,
            }
            for offset in offsets
        ]
    )


def _refprice_frame() -> pl.DataFrame:
    rows = []
    for offset in range(-70, 11):
        timestamp = DECISION + timedelta(seconds=offset)
        price = 100_000.0 * math.exp(offset * 0.000001)
        rows.append(
            {
                "source_timestamp": timestamp,
                "valid_from_timestamp": timestamp - timedelta(milliseconds=100),
                "price": price,
                "bid": price - 0.5,
                "ask": price + 0.5,
            }
        )
    return pl.DataFrame(rows)


def _candle_frame() -> pl.DataFrame:
    rows = []
    for offset in range(-70, 2):
        close_timestamp = DECISION + timedelta(minutes=offset)
        close = 100_000.0 * math.exp(offset * 0.00001)
        rows.append(
            {
                "open_timestamp": close_timestamp - timedelta(minutes=1),
                "close_timestamp": close_timestamp,
                "open_price": close - 0.25,
                "high_price": close + 1.0,
                "low_price": close - 1.0,
                "close_price": close,
            }
        )
    return pl.DataFrame(rows)


def _open_interest_frame() -> pl.DataFrame:
    rows = []
    for offset in range(-13, 1):
        timestamp = DECISION + timedelta(minutes=offset * 5)
        interest = 10_000.0 * math.exp(offset * 0.0001)
        rows.append(
            {
                "source_timestamp": timestamp,
                "period_seconds": 300,
                "sum_open_interest": interest,
                "sum_open_interest_value": interest * 100_000.0,
            }
        )
    return pl.DataFrame(rows)


def _core_source_frame() -> pl.DataFrame:
    window_start = DECISION - timedelta(seconds=120)
    rows = []
    for second in range(300):
        close = 100_000.0 * math.exp(second * 0.000001)
        rows.append(
            {
                "market_id": "btc-updown-5m-core-schema",
                "window_start": window_start,
                "window_end": window_start + timedelta(minutes=5),
                "official_outcome": "up",
                "label_up": 1,
                "opening_boundary": 100_000.0,
                "final_price": close + 1.0,
                "observed_at": window_start + timedelta(seconds=second),
                "seconds_elapsed": second,
                "btc_open": close - 0.1,
                "btc_high": close + 0.5,
                "btc_low": close - 0.5,
                "btc_close": close,
                "btc_base_volume": 1.0,
                "btc_quote_volume": 10_000.0,
                "trade_count": 10,
                "btc_taker_buy_base_volume": 0.5,
                "btc_taker_buy_quote_volume": 5_000.0,
            }
        )
    return pl.DataFrame(rows)


def _derive(
    core: pl.DataFrame,
    refprice: pl.DataFrame,
    candles: pl.DataFrame,
    open_interest: pl.DataFrame,
    *,
    refprice_max_age_seconds: int = 2,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return derive_chainlink_oi_feature_frames(
        core,
        refprice,
        candles,
        open_interest,
        refprice_max_age_seconds=refprice_max_age_seconds,
    )


def test_external_feature_cohorts_are_strict_and_key_identical() -> None:
    without_oi, with_oi = _derive(
        _core_frame(),
        _refprice_frame(),
        _candle_frame(),
        _open_interest_frame(),
    )

    assert without_oi.height == 2
    assert with_oi.height == 2
    assert without_oi.select(POINT_KEY_COLUMNS).equals(with_oi.select(POINT_KEY_COLUMNS))
    assert set(CHAINLINK_EXTERNAL_FEATURES).issubset(without_oi.columns)
    assert not set(BINANCE_OI_FEATURES).intersection(without_oi.columns)
    assert set(CHAINLINK_OI_FEATURES).issubset(with_oi.columns)
    assert "chainlink_ref_return_1s_bps" in with_oi.columns
    assert (
        with_oi.select(
            pl.all_horizontal(pl.col(feature).is_finite() for feature in CHAINLINK_OI_FEATURES)
        )
        .to_series()
        .all()
    )
    assert not any(
        column.startswith("_ref_") or column in {"source_timestamp", "close_timestamp"}
        for column in with_oi.columns
    )


def test_candle_only_path_has_no_refprice_or_oi_dependency() -> None:
    rows = derive_chainlink_candle_feature_frame(
        _core_frame(),
        _candle_frame(),
    )

    assert rows.height == 2
    assert set(CHAINLINK_CANDLE_FEATURES).issubset(rows.columns)
    assert not any("ref_" in column or "oi_" in column for column in rows.columns)


def test_external_requirements_are_compatible_with_core_feature_schema() -> None:
    core_features = derive_core_point_in_time_features(_core_source_frame())

    assert set(EXTERNAL_CORE_REQUIRED_COLUMNS).issubset(core_features.columns)


def test_refprice_current_tick_is_strictly_before_the_decision() -> None:
    core = _core_frame(include_second_point=False)
    original = _refprice_frame()
    changed = original.with_columns(
        pl.when(pl.col("source_timestamp") >= DECISION)
        .then(pl.col("price") * 2.0)
        .otherwise(pl.col("price"))
        .alias("price"),
        pl.when(pl.col("source_timestamp") >= DECISION)
        .then(pl.col("bid") * 2.0)
        .otherwise(pl.col("bid"))
        .alias("bid"),
        pl.when(pl.col("source_timestamp") >= DECISION)
        .then(pl.col("ask") * 2.0)
        .otherwise(pl.col("ask"))
        .alias("ask"),
    )

    original_rows, _ = _derive(core, original, _candle_frame(), _open_interest_frame())
    changed_rows, _ = _derive(core, changed, _candle_frame(), _open_interest_frame())

    assert original_rows.select(CHAINLINK_EXTERNAL_FEATURES).equals(
        changed_rows.select(CHAINLINK_EXTERNAL_FEATURES)
    )


def test_refprice_freshness_is_configurable_and_never_filled() -> None:
    refprice = _refprice_frame().filter(
        (pl.col("source_timestamp") <= DECISION - timedelta(seconds=3))
        | (pl.col("source_timestamp") > DECISION)
    )

    too_old, _ = _derive(
        _core_frame(include_second_point=False),
        refprice,
        _candle_frame(),
        _open_interest_frame(),
        refprice_max_age_seconds=2,
    )
    accepted, _ = _derive(
        _core_frame(include_second_point=False),
        refprice,
        _candle_frame(),
        _open_interest_frame(),
        refprice_max_age_seconds=3,
    )

    assert too_old.is_empty()
    assert accepted.height == 1


def test_oi_qualification_applies_to_both_candidates() -> None:
    interest_without_exact_point = _open_interest_frame().filter(
        pl.col("source_timestamp") < DECISION
    )

    without_oi, with_oi = _derive(
        _core_frame(),
        _refprice_frame(),
        _candle_frame(),
        interest_without_exact_point,
    )

    assert without_oi.height == 1
    assert with_oi.height == 1
    assert without_oi["seconds_elapsed"].to_list() == [120]
    assert with_oi["seconds_elapsed"].to_list() == [120]


def test_candles_are_closed_and_contiguous_before_use() -> None:
    incomplete = _candle_frame().filter(
        pl.col("close_timestamp") != DECISION - timedelta(minutes=30)
    )

    without_oi, with_oi = _derive(
        _core_frame(),
        _refprice_frame(),
        incomplete,
        _open_interest_frame(),
    )

    assert without_oi.is_empty()
    assert with_oi.is_empty()


def _sql(name: str) -> str:
    return (Path(__file__).parent.parent / "sql" / name).read_text().lower()


def test_external_source_sql_is_completed_bounded_and_read_only() -> None:
    refprice = _sql("btc-chainlink-refprice-source.sql")
    candles = _sql("btc-chainlink-one-minute-candles-source.sql")
    interest = _sql("btc-binance-five-minute-open-interest-source.sql")

    for sql in (refprice, candles, interest):
        assert sql.lstrip().startswith("select")
        assert "artifact.status = 'completed'" in sql
        assert "%(range_start)s" in sql
        assert "%(range_end)s" in sql
        assert "insert " not in sql
        assert "update " not in sql
        assert "delete " not in sql
        assert "experiment" not in sql

    assert "chainlink_btcusd_archive_ticks" in refprice
    assert "tick.source_timestamp < %(range_end)s" in refprice
    assert "chainlink_btcusd_one_minute_candles" in candles
    assert "candle.close_timestamp" in candles
    assert "volume" not in candles
    assert "binance_btcusdt_five_minute_open_interest" in interest
    assert "interest.source_timestamp < %(range_end)s" in interest
    assert "interest.period_seconds = 300" in interest
