from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from btc_directional_model.binance_latent_twap_context import (
    KLINE_FEATURES,
    OI_FEATURES,
    apply_residual_adjustment,
    attach_open_interest_features,
    derive_kline_checkpoint_features,
    fit_residual_adjustment,
)


def _kline_rows() -> pl.DataFrame:
    start = datetime(2026, 8, 14, tzinfo=UTC)
    rows = []
    for second in range(181):
        close = 60_000.0 + second
        rows.append(
            {
                "market_id": "market",
                "window_start": start,
                "observed_at": start + timedelta(seconds=second),
                "seconds_elapsed": second,
                "btc_open": close - 0.5,
                "btc_high": close + 1.0,
                "btc_low": close - 1.0,
                "btc_close": close,
                "btc_quote_volume": 100.0 + second,
                "btc_taker_buy_quote_volume": 52.0 + second / 2.0,
            }
        )
    return pl.DataFrame(rows)


def test_kline_features_are_unchanged_by_future_source_rows() -> None:
    source = _kline_rows()
    original = derive_kline_checkpoint_features(source).filter(
        pl.col("seconds_elapsed") <= 90
    )
    perturbed = source.with_columns(
        pl.when(pl.col("seconds_elapsed") > 90)
        .then(pl.col("btc_close") * 10.0)
        .otherwise(pl.col("btc_close"))
        .alias("btc_close"),
        pl.when(pl.col("seconds_elapsed") > 90)
        .then(pl.col("btc_quote_volume") * 100.0)
        .otherwise(pl.col("btc_quote_volume"))
        .alias("btc_quote_volume"),
    )
    replay = derive_kline_checkpoint_features(perturbed).filter(
        pl.col("seconds_elapsed") <= 90
    )

    assert original.select("observed_at", *KLINE_FEATURES).equals(
        replay.select("observed_at", *KLINE_FEATURES), null_equal=True
    )


def test_open_interest_join_is_strictly_prior_and_expires() -> None:
    anchor = datetime(2026, 8, 14, tzinfo=UTC)
    source_times = [anchor - timedelta(minutes=65 - 5 * index) for index in range(14)]
    source = pl.DataFrame(
        {
            "source_timestamp": source_times,
            "period_seconds": [300] * len(source_times),
            "sum_open_interest": [100_000.0 + 100.0 * index for index in range(14)],
            "sum_open_interest_value": [6.0e9 + 1.0e7 * index for index in range(14)],
        }
    )
    frame = pl.DataFrame(
        {
            "market_id": ["exact", "new", "expired"],
            "observed_at": [anchor, anchor + timedelta(seconds=1), anchor + timedelta(seconds=301)],
            "binance_kline_path_from_window_open_bps": [1.0, 1.0, 1.0],
        }
    )

    joined = attach_open_interest_features(frame, source)

    exact = joined.filter(pl.col("market_id") == "exact")
    new = joined.filter(pl.col("market_id") == "new")
    expired = joined.filter(pl.col("market_id") == "expired")
    assert exact["binance_oi_source_timestamp"].item() == anchor - timedelta(minutes=5)
    assert new["binance_oi_source_timestamp"].item() == anchor
    assert new["binance_oi_age_seconds"].item() == 1.0
    assert expired.select(*OI_FEATURES).null_count().row(0) == (1,) * len(OI_FEATURES)
    assert expired["binance_oi_source_timestamp"].item() is None


def test_residual_model_drops_missing_context_and_round_trips() -> None:
    rows = []
    for market_index in range(4):
        label = market_index % 2
        for second in (30, 60):
            context = float(market_index + second / 100.0)
            if market_index == 3 and second == 60:
                context = None
            rows.append(
                {
                    "market_id": f"market-{market_index}",
                    "label_up": label,
                    "target_margin_bps": (8.0 if label else -8.0) + market_index,
                    "seconds_elapsed": second,
                    "expected_margin_bps": 3.0 if label else -3.0,
                    "margin_velocity_bps_per_step": 0.1 * (market_index + 1),
                    "process_uncertainty_bps2": 4.0,
                    "reversal_probability": 0.1,
                    "stable_regime_probability": 0.6,
                    "trending_regime_probability": 0.3,
                    "reversal_regime_probability": 0.1,
                    "context": context,
                }
            )
    scored = pl.DataFrame(rows)

    parameters = fit_residual_adjustment(
        scored,
        candidate_name="test-context",
        context_features=("context",),
        ridge_alpha=1.0,
    )
    adjusted = apply_residual_adjustment(
        scored,
        parameters,
        context_features=("context",),
    )

    assert parameters.fit_markets == 4
    assert parameters.fit_rows == 7
    assert adjusted.height == 7
    assert adjusted["market_id"].n_unique() == 4
    assert np.isfinite(adjusted["probability_up"].to_numpy()).all()
    assert adjusted["probability_up"].is_between(0.0, 1.0).all()
