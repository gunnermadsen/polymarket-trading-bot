from __future__ import annotations

import numpy as np
import polars as pl

from btc_directional_model.early_value_training import accuracy_by_second, prediction_frame


def test_no_threshold_accuracy_is_reported_at_each_second() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "b", "a", "b"],
            "window_start": [1, 2, 1, 2],
            "observed_at": [6, 7, 11, 12],
            "seconds_elapsed": [5, 5, 10, 10],
            "label_up": [1, 0, 1, 0],
        }
    )
    predictions = prediction_frame(
        frame,
        np.array([0.60, 0.40, 0.49, 0.51]),
        model="test",
    )
    rows = accuracy_by_second(predictions)
    assert [row["seconds_elapsed"] for row in rows] == [5, 10]
    assert [row["accuracy"] for row in rows] == [1.0, 0.0]


def test_prediction_frame_keeps_low_confidence_predictions() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a"],
            "window_start": [1],
            "observed_at": [6],
            "seconds_elapsed": [5],
            "label_up": [1],
        }
    )
    predictions = prediction_frame(frame, np.array([0.51]), model="test")
    assert predictions.height == 1
    assert predictions["confidence"].item() == 0.51
