from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from btc_directional_model.core_evaluation import (
    block_bootstrap_uplift,
    first_prediction_rows,
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
