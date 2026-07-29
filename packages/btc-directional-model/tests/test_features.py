from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from btc_directional_model.features import FEATURE_GROUPS, derive_point_in_time_features


def source_frame() -> pl.DataFrame:
    start = datetime(2026, 5, 1, tzinfo=UTC)
    rows = []
    for market_index, market_id in enumerate(("a", "b")):
        for second in range(70):
            price = 100_000 + market_index * 1_000 + second
            observed_at = start + timedelta(minutes=market_index * 5, seconds=second)
            rows.append(
                {
                    "market_id": market_id,
                    "seconds_elapsed": second,
                    "observed_at": observed_at,
                    "opening_boundary": float(100_000 + market_index * 1_000),
                    "btc_open": float(price - 0.5),
                    "btc_high": float(price + 1),
                    "btc_low": float(price - 1),
                    "btc_close": float(price),
                    "btc_quote_volume": float(10 + second),
                    "trade_count": 2,
                    "btc_taker_buy_quote_volume": float(5 + second / 2),
                    "up_provider_received_at": observed_at - timedelta(milliseconds=10),
                    "down_provider_received_at": observed_at - timedelta(milliseconds=15),
                    "up_best_bid": 0.50,
                    "up_best_ask": 0.52,
                    "down_best_bid": 0.47,
                    "down_best_ask": 0.49,
                    "up_bid_depth": 10.0,
                    "up_ask_depth": 11.0,
                    "down_bid_depth": 12.0,
                    "down_ask_depth": 13.0,
                    "up_ask_vwap_1": 0.52,
                    "up_ask_vwap_5": 0.53,
                    "up_ask_vwap_10": 0.54,
                    "down_ask_vwap_1": 0.49,
                    "down_ask_vwap_5": 0.50,
                    "down_ask_vwap_10": 0.51,
                    "up_imbalance": 0.1,
                    "down_imbalance": -0.1,
                    "quality_flags": 0,
                }
            )
    return pl.DataFrame(rows).sort(["market_id", "seconds_elapsed"])


def test_feature_allowlists_exclude_labels_and_final_values() -> None:
    forbidden = {"label_up", "official_outcome", "final_price", "window_end"}

    for features in FEATURE_GROUPS.values():
        assert forbidden.isdisjoint(features)


def test_point_in_time_features_do_not_change_when_future_rows_change() -> None:
    original = source_frame()
    altered = original.with_columns(
        pl.when((pl.col("market_id") == "a") & (pl.col("seconds_elapsed") > 60))
        .then(pl.col("btc_close") * 2)
        .otherwise(pl.col("btc_close"))
        .alias("btc_close")
    )

    original_features = derive_point_in_time_features(original)
    altered_features = derive_point_in_time_features(altered)
    feature_names = FEATURE_GROUPS["btc_path_and_book"]
    original_at_60 = original_features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 60)
    ).select(feature_names)
    altered_at_60 = altered_features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 60)
    ).select(feature_names)

    assert original_at_60.equals(altered_at_60, null_equal=True)


def test_lags_do_not_cross_market_boundaries() -> None:
    features = derive_point_in_time_features(source_frame())
    first_b = features.filter((pl.col("market_id") == "b") & (pl.col("seconds_elapsed") == 0))

    assert first_b["btc_return_1s_bps"][0] is None
