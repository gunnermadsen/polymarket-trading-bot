from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from btc_directional_model.strategy_conditioned_risk_tournament import (
    _bucket_expr,
    _intervention_metrics,
)


def _candidates() -> pl.DataFrame:
    start = datetime(2026, 8, 21, tzinfo=UTC)
    return pl.DataFrame({
        "champion": ["s", "s", "s"], "market_id": ["a", "a", "b"],
        "window_start": [start, start, start + timedelta(minutes=5)],
        "seconds_elapsed": [60, 90, 180], "net_pnl": [-3.0, 2.0, 1.0],
        "stress_net_pnl": [-3.05, 1.95, 0.95], "share_cost": [0.6, 0.5, 0.4],
    })


def test_intervention_metrics_distinguish_loss_capture_and_opportunity_rejection() -> None:
    result = _intervention_metrics(_candidates(), np.array([1.0, 0.0, 0.0]), 0.5)
    assert result["blocked_losses"] == 1
    assert result["blocked_winners"] == 0
    assert result["loss_capture_rate"] == 1.0
    assert result["opportunity_rejection_rate"] == 0.0
    assert result["deferred_to_later_entry"] == 1
    assert result["net_risk_value"] == 5.0


def test_natural_time_buckets_cover_full_process_window() -> None:
    frame = pl.DataFrame({"seconds_elapsed": [15, 60, 90, 120, 150, 180, 240]})
    assert frame.with_columns(_bucket_expr().alias("bucket"))["bucket"].to_list() == [
        "15_59", "60_89", "90_119", "120_149", "150_179", "180_240", "180_240"
    ]
