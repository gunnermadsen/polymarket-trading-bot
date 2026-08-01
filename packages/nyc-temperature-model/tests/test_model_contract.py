from datetime import UTC, date, datetime

import numpy as np

from nyc_temperature_model.modeling import (
    FeatureRow,
    bucket_probability,
    normalized_bucket_probabilities,
    round_temperature,
)


def test_round_temperature_uses_half_up():
    assert round_temperature(66.5) == 67
    assert round_temperature(66.49) == 66


def test_bucket_probabilities_are_smoothed_and_ordered():
    residuals = np.asarray([-2, -1, 0, 1, 2], dtype=float)
    center = bucket_probability(70, residuals, 69, 71)
    tail = bucket_probability(70, residuals, 75, None)
    assert 0 < tail < center < 1


def test_market_bucket_probabilities_sum_to_one_after_smoothing():
    probabilities = normalized_bucket_probabilities(
        70,
        np.asarray([-2, -1, 0, 1, 2], dtype=float),
        [(None, 68), (69, 71), (72, None)],
    )
    assert np.isclose(sum(probabilities), 1.0)


def test_midnight_decision_hour_remains_zero_not_missing():
    row = FeatureRow(
        event_date=date(2026, 7, 4),
        decision_time=datetime(2026, 7, 4, 4, tzinfo=UTC),
        decision_hour_local=0,
        forecast_max_remaining_f=90,
        forecast_mean_remaining_f=80,
        forecast_min_remaining_f=70,
        observed_max_so_far_f=None,
        latest_observation_f=72,
        day_of_year_sin=0.1,
        day_of_year_cos=-0.2,
        target_daily_max_f=91,
        target_rounded_max_f=91,
        observation_count=24,
    )
    assert row.vector()[-1] == 0.0
