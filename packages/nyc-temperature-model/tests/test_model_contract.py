from datetime import UTC, date, datetime

import numpy as np

from nyc_temperature_model.modeling import (
    FeatureRow,
    _complete_forecast_values,
    _expected_forecast_times,
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


def test_expected_forecasts_cover_each_hour_until_local_midnight():
    midnight = datetime(2026, 7, 4, 4, tzinfo=UTC)
    noon = datetime(2026, 7, 4, 16, tzinfo=UTC)

    assert len(_expected_forecast_times(midnight)) == 24
    assert len(_expected_forecast_times(noon)) == 12


def test_expected_forecasts_respect_daylight_saving_day_length():
    spring_midnight = datetime(2026, 3, 8, 5, tzinfo=UTC)
    fall_midnight = datetime(2026, 11, 1, 4, tzinfo=UTC)

    assert len(_expected_forecast_times(spring_midnight)) == 23
    assert len(_expected_forecast_times(fall_midnight)) == 25


def test_incomplete_forecast_window_is_not_eligible_for_model_features():
    decision_time = datetime(2019, 3, 11, 16, tzinfo=UTC)
    expected = _expected_forecast_times(decision_time)
    forecasts = {valid_at: 50.0 for valid_at in expected}

    assert _complete_forecast_values(decision_time, forecasts) == [50.0] * 12

    del forecasts[datetime(2019, 3, 11, 20, tzinfo=UTC)]
    assert _complete_forecast_values(decision_time, forecasts) is None
