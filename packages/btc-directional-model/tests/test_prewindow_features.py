from datetime import UTC, datetime, timedelta

import polars as pl

from btc_directional_model.prewindow_features import (
    PREWINDOW_MODEL_FEATURES,
    derive_prewindow_static_features,
    join_prewindow_features,
)


def summaries(count: int = 14) -> pl.DataFrame:
    start = datetime(2026, 5, 27, tzinfo=UTC)
    return pl.DataFrame(
        {
            "market_id": [f"m-{index}" for index in range(count)],
            "window_start": [
                start + timedelta(minutes=5 * index) for index in range(count)
            ],
            "open_available_close": [100.0 + index for index in range(count)],
            "last_close": [100.5 + index for index in range(count)],
            "window_high": [101.0 + index for index in range(count)],
            "window_low": [99.0 + index for index in range(count)],
            "window_quote_volume": [1000.0 + index for index in range(count)],
            "window_trade_count": [100.0 + index for index in range(count)],
            "window_taker_buy_quote_volume": [
                550.0 + index for index in range(count)
            ],
            "window_volatility_bps": [2.0 + index / 10 for index in range(count)],
        }
    )


def test_sixty_minute_features_require_twelve_contiguous_prior_windows() -> None:
    derived = derive_prewindow_static_features(summaries())
    complete = derived.filter(pl.col("market_id") == "m-12").row(0, named=True)
    incomplete = derived.filter(pl.col("market_id") == "m-11").row(0, named=True)

    assert complete["prewindow_return_60m_bps"] is not None
    assert complete["prewindow_log_quote_volume_60m"] is not None
    assert incomplete["prewindow_return_60m_bps"] is None


def test_gap_marks_long_horizon_unknown_instead_of_bridging_history() -> None:
    frame = summaries().with_columns(
        pl.when(pl.col("market_id") == "m-12")
        .then(pl.col("window_start") + pl.duration(minutes=5))
        .otherwise(pl.col("window_start"))
        .alias("window_start")
    )
    row = (
        derive_prewindow_static_features(frame)
        .filter(pl.col("market_id") == "m-12")
        .row(0, named=True)
    )

    assert row["prewindow_return_60m_bps"] is None
    assert row["prewindow_range_60m_bps"] is None


def test_future_market_mutation_cannot_change_existing_prewindow_features() -> None:
    original = derive_prewindow_static_features(summaries())
    mutated = derive_prewindow_static_features(
        summaries().with_columns(
            pl.when(pl.col("market_id") == "m-13")
            .then(pl.lit(1_000_000.0))
            .otherwise(pl.col("window_quote_volume"))
            .alias("window_quote_volume")
        )
    )
    original_row = original.filter(pl.col("market_id") == "m-12").select(
        pl.col("^prewindow_.*$")
    )
    mutated_row = mutated.filter(pl.col("market_id") == "m-12").select(
        pl.col("^prewindow_.*$")
    )

    assert original_row.equals(mutated_row)


def test_join_requires_complete_features_for_model_eligibility() -> None:
    start = datetime(2026, 5, 27, tzinfo=UTC)
    core = pl.DataFrame(
        {
            "market_id": ["a"],
            "window_start": [start],
            "btc_path_from_window_open_bps": [2.0],
        }
    )
    values = {
        feature: [1.0]
        for feature in PREWINDOW_MODEL_FEATURES
        if not feature.startswith("btc_path_")
    }
    prewindow = pl.DataFrame(
        {
            "market_id": ["a"],
            "window_start": [start],
            **values,
        }
    )

    joined = join_prewindow_features(core, prewindow)

    assert joined["prewindow_model_eligible"][0] is True
