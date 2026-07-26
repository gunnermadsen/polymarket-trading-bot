from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from btc_directional_model.core_features import (
    CORE_MODEL_FEATURES,
    derive_core_point_in_time_features,
)


def core_source_frame() -> pl.DataFrame:
    start = datetime(2026, 5, 1, tzinfo=UTC)
    rows = []
    for market_index, market_id in enumerate(("a", "b")):
        boundary = 100_000.0 + market_index * 1_000
        for second in range(300):
            close = boundary + ((second % 40) - 20) * 0.5
            observed_at = start + timedelta(minutes=market_index * 5, seconds=second)
            rows.append(
                {
                    "market_id": market_id,
                    "window_start": start + timedelta(minutes=market_index * 5),
                    "window_end": start + timedelta(minutes=(market_index + 1) * 5),
                    "official_outcome": "up" if market_index else "down",
                    "label_up": market_index,
                    "opening_boundary": boundary,
                    "final_price": boundary + (1 if market_index else -1),
                    "observed_at": observed_at,
                    "seconds_elapsed": second,
                    "btc_open": close - 0.1,
                    "btc_high": close + 0.5,
                    "btc_low": close - 0.5,
                    "btc_close": close,
                    "btc_base_volume": 1.0,
                    "btc_quote_volume": 10_000.0 + second,
                    "trade_count": 10,
                    "btc_taker_buy_base_volume": 0.5,
                    "btc_taker_buy_quote_volume": 5_000.0 + second / 2,
                }
            )
    return pl.DataFrame(rows).sort(["market_id", "seconds_elapsed"])


def test_enriched_core_features_need_no_book_columns() -> None:
    features = derive_core_point_in_time_features(core_source_frame())

    for allowlist in CORE_MODEL_FEATURES.values():
        assert set(allowlist).issubset(features.columns)
    assert features.filter(pl.col("seconds_elapsed") == 240).height == 2
    assert (
        features.filter(pl.col("seconds_elapsed") == 240)[
            "btc_path_terminal_volatility_z"
        ].null_count()
        == 0
    )


def test_core_features_are_invariant_to_future_price_mutation() -> None:
    original = core_source_frame()
    altered = original.with_columns(
        pl.when((pl.col("market_id") == "a") & (pl.col("seconds_elapsed") > 120))
        .then(pl.col("btc_close") * 1.5)
        .otherwise(pl.col("btc_close"))
        .alias("btc_close")
    )
    original_features = derive_core_point_in_time_features(original)
    altered_features = derive_core_point_in_time_features(altered)
    allowlist = CORE_MODEL_FEATURES["histogram_enriched"]

    before = original_features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 120)
    ).select(allowlist)
    after = altered_features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 120)
    ).select(allowlist)

    assert before.equals(after, null_equal=True)


def test_core_lags_and_cross_counts_do_not_cross_markets() -> None:
    features = derive_core_point_in_time_features(core_source_frame())
    first_b = features.filter(
        (pl.col("market_id") == "b") & (pl.col("seconds_elapsed") == 0)
    )

    assert first_b["btc_return_1s_bps"][0] is None
    assert first_b["btc_path_cross_count"][0] == 0


def test_model_features_ignore_cross_venue_opening_basis() -> None:
    original = core_source_frame()
    shifted_boundary = original.with_columns(
        (pl.col("opening_boundary") * 1.02).alias("opening_boundary")
    )
    original_features = derive_core_point_in_time_features(original)
    shifted_features = derive_core_point_in_time_features(shifted_boundary)

    for allowlist in CORE_MODEL_FEATURES.values():
        assert original_features.select(allowlist).equals(
            shifted_features.select(allowlist),
            null_equal=True,
        )
    assert original_features["binance_sign_up"].equals(
        shifted_features["binance_sign_up"]
    )
