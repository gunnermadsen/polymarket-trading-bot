from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from btc_directional_model.core_evaluation import (
    block_bootstrap_uplift,
    first_crossing_timing,
    first_prediction_rows,
    fixed_time_prediction_rows,
    paired_uplift,
)


def evaluation_frame() -> pl.DataFrame:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    rows = []
    for market_index in range(24):
        label = market_index % 2
        window_start = start + timedelta(minutes=market_index * 5)
        for second in (60, 65, 70):
            rows.append(
                {
                    "market_id": f"market-{market_index:02d}",
                    "window_start": window_start,
                    "observed_at": window_start + timedelta(seconds=second),
                    "seconds_elapsed": second,
                    "label_up": label,
                    "binance_sign_up": 1 - label,
                }
            )
    return pl.DataFrame(rows).sort(["market_id", "seconds_elapsed"])


def test_core_prediction_selects_first_confidence_crossing() -> None:
    frame = evaluation_frame()
    per_market = np.asarray([0.55, 0.80, 0.90])
    probabilities = np.tile(per_market, frame["market_id"].n_unique())

    rows = first_prediction_rows(frame, probabilities, 0.70)

    assert rows.height == 24
    assert set(rows["seconds_elapsed"]) == {65}


def test_paired_bootstrap_is_deterministic_and_same_cohort() -> None:
    frame = evaluation_frame()
    labels = frame["label_up"].to_numpy()
    probabilities = np.where(labels == 1, 0.9, 0.1)
    rows = first_prediction_rows(frame, probabilities, 0.70)

    paired = paired_uplift(rows)
    first = block_bootstrap_uplift(
        rows,
        resamples=500,
        random_seed=7,
        block="hour",
    )
    second = block_bootstrap_uplift(
        rows,
        resamples=500,
        random_seed=7,
        block="hour",
    )

    assert paired["markets"] == rows.height
    assert paired["accuracy_uplift"] == 1
    assert first == second


def test_fixed_time_rows_are_compact_and_do_not_apply_confidence_policy() -> None:
    frame = evaluation_frame()
    extra = frame.filter(pl.col("seconds_elapsed") == 60).with_columns(
        (pl.col("window_start") + pl.duration(seconds=90)).alias("observed_at"),
        pl.lit(90).alias("seconds_elapsed"),
    )
    expanded = pl.concat([frame, extra], how="vertical_relaxed").sort(
        ["market_id", "seconds_elapsed"]
    )
    probabilities = np.full(expanded.height, 0.51)

    rows = fixed_time_prediction_rows(expanded, probabilities)

    assert rows.height == 24 * 2
    assert set(rows["seconds_elapsed"]) == {60, 90}
    assert set(rows.columns) >= {
        "market_id",
        "observed_at",
        "label_up",
        "probability_up",
        "predicted_up",
        "confidence",
        "correct",
        "baseline_correct",
    }


def test_first_crossing_timing_reports_frozen_bands_and_quantiles() -> None:
    frame = evaluation_frame().filter(pl.col("seconds_elapsed") == 60)
    elapsed = np.asarray([60, 90, 120, 180] * 6)
    rows = frame.with_columns(
        pl.Series("seconds_elapsed", elapsed),
        (
            pl.col("window_start")
            + pl.duration(seconds=pl.Series("crossing_seconds", elapsed))
        ).alias("observed_at"),
        pl.lit(True).alias("correct"),
        pl.lit(False).alias("baseline_correct"),
    )

    timing = first_crossing_timing(rows, eligible_markets=48)

    assert timing["markets"] == 24
    assert timing["coverage"] == 0.5
    assert timing["early_entry_markets"] == 12
    assert timing["early_entry_coverage"] == 0.25
    assert timing["median_first_crossing_seconds"] == 105
    assert timing["p90_first_crossing_seconds"] == 180
    assert [band["band"] for band in timing["time_bands"]] == [
        "60-89",
        "90-119",
        "120-179",
        "180-240",
    ]
    assert [band["markets"] for band in timing["time_bands"]] == [6, 6, 6, 6]
