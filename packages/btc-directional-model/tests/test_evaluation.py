from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from btc_directional_model.evaluation import (
    baseline_metrics,
    empty_metrics,
    prediction_rows,
    wilson_interval,
)


def prediction_frame() -> pl.DataFrame:
    start = datetime(2026, 5, 1, tzinfo=UTC)
    rows = []
    for market_id, label in (("a", 1), ("b", 0)):
        for offset in (60, 65, 70):
            rows.append(
                {
                    "market_id": market_id,
                    "window_start": start,
                    "observed_at": start + timedelta(seconds=offset),
                    "seconds_elapsed": offset,
                    "label_up": label,
                    "binance_sign_up": label,
                    "market_favorite_up": label,
                    "up_executable": market_id == "a" and offset == 70,
                    "down_executable": market_id == "b",
                    "up_ask_vwap_5": 0.55,
                    "down_ask_vwap_5": 0.45,
                }
            )
    return pl.DataFrame(rows).sort(["market_id", "seconds_elapsed"])


def test_prediction_rows_select_first_threshold_crossing_and_first_executable() -> None:
    frame = prediction_frame()
    probabilities = np.asarray([0.55, 0.80, 0.85, 0.25, 0.20, 0.15])

    selected = prediction_rows(frame, probabilities, 0.70, require_executable=False)
    executable = prediction_rows(frame, probabilities, 0.70, require_executable=True)

    assert dict(zip(selected["market_id"], selected["seconds_elapsed"], strict=True)) == {
        "a": 65,
        "b": 60,
    }
    assert dict(zip(executable["market_id"], executable["seconds_elapsed"], strict=True)) == {
        "a": 70,
        "b": 60,
    }


def test_baseline_metrics_replace_model_probabilities() -> None:
    rows = pl.DataFrame(
        {
            "label_up": [0, 1],
            "predicted_up": [1, 0],
            "probability_up": [0.9, 0.1],
            "correct": [False, False],
            "observed_at": [
                datetime(2026, 5, 1, tzinfo=UTC),
                datetime(2026, 5, 1, tzinfo=UTC) + timedelta(minutes=5),
            ],
            "baseline": [0, 1],
        }
    )

    metrics = baseline_metrics(rows, "baseline")

    assert metrics["accuracy"] == 1
    assert metrics["brier_score"] == 0
    assert metrics["roc_auc"] == 1
    assert metrics["log_loss"] == pytest.approx(2.2204460492503136e-16)


def test_empty_metrics_are_strict_json_and_wilson_is_bounded() -> None:
    json.dumps(empty_metrics(), allow_nan=False)
    lower, upper = wilson_interval(65, 100)

    assert 0 < lower < 0.65 < upper < 1
