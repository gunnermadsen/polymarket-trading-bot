from datetime import UTC, datetime

import numpy as np
import polars as pl

from btc_directional_model.bucket_harm_risk_tournament import (
    attach_bucket_targets,
    safety_margin,
)


def test_deferral_target_never_crosses_bucket_boundary() -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    frame = pl.DataFrame({
        "champion": ["s", "s", "s"], "market_id": ["a", "a", "a"],
        "window_start": [start] * 3, "seconds_elapsed": [80, 85, 90],
        "net_pnl": [-2.0, 1.0, 4.0],
    })
    result = attach_bucket_targets(frame, 0.1).sort("seconds_elapsed")
    assert result["future_best_within_bucket_pnl"].to_list() == [1.0, 0.0, 0.0]
    assert result["within_bucket_defer_value"].to_list() == [0.9, 0.0, 0.0]


def test_bucket_thresholds_are_applied_by_candidate_bucket() -> None:
    frame = pl.DataFrame({"time_bucket": ["60_89", "90_119", "180_240"]})
    margin = safety_margin(
        frame, np.array([0.8, 0.8, 0.8]),
        {"60_89": 0.7, "90_119": 0.9, "180_240": 0.8},
    )
    assert np.allclose(margin, [0.1, -0.1, 0.0])
