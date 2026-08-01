from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from kraken_ml.regression_models import (
    REGRESSION_MODEL_NAMES,
    NonnegativeAffineCalibrator,
    build_regression_estimator,
    fit_net_regressors,
)


def _regression_frame(
    rows: int,
    *,
    offset: float = 0.0,
    start: datetime | None = None,
) -> pl.DataFrame:
    signal = np.linspace(-2.0, 2.0, rows) + offset
    frame_start = start or datetime(2026, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "bucket_start": [
                frame_start + timedelta(minutes=15 * index) for index in range(rows)
            ],
            "signal": signal,
            "secondary": np.sin(np.arange(rows) / 7.0),
            "long_net_bps": 5.0 + 4.0 * signal,
            "short_net_bps": -3.0 - 2.0 * signal,
        }
    )


def test_nonnegative_affine_calibrator_falls_back_for_degenerate_or_negative_slope() -> None:
    degenerate = NonnegativeAffineCalibrator.fit(
        np.ones(4),
        np.asarray([1.0, 3.0, 5.0, 7.0]),
    )
    negative = NonnegativeAffineCalibrator.fit(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([4.0, 3.0, 2.0, 1.0]),
    )

    assert degenerate.slope == 0.0
    assert degenerate.intercept == pytest.approx(4.0)
    assert negative.slope == 0.0
    assert negative.intercept == pytest.approx(2.5)
    np.testing.assert_allclose(negative.predict(np.asarray([10.0, 20.0])), 2.5)


def test_positive_affine_calibrator_recovers_linear_mapping() -> None:
    calibrator = NonnegativeAffineCalibrator.fit(
        np.asarray([-1.0, 0.0, 1.0, 2.0]),
        np.asarray([1.0, 3.0, 5.0, 7.0]),
    )

    assert calibrator.slope == pytest.approx(2.0)
    assert calibrator.intercept == pytest.approx(3.0)


def test_fixed_regressor_parameters_disable_nested_search_and_honor_threads() -> None:
    histogram, histogram_parameters = build_regression_estimator(
        "histogram",
        seed=17,
        threads=1,
    )
    extra_trees, extra_parameters = build_regression_estimator(
        "extra_trees",
        seed=17,
        threads=3,
    )

    assert isinstance(histogram, HistGradientBoostingRegressor)
    assert histogram_parameters["early_stopping"] is False
    assert extra_parameters["n_estimators"] == 500
    assert extra_trees.named_steps["regressor"].n_jobs == 3


@pytest.mark.parametrize("model_name", REGRESSION_MODEL_NAMES)
def test_all_regressors_fit_sequential_sides_and_emit_finite_calibrated_predictions(
    model_name: str,
) -> None:
    fitted = fit_net_regressors(
        name=model_name,
        feature_set="test",
        feature_names=("signal", "secondary"),
        fit_frame=_regression_frame(240),
        calibration_frame=_regression_frame(
            80,
            offset=0.01,
            start=datetime(2026, 1, 4, tzinfo=UTC),
        ),
        seed=20260728,
        threads=1,
    )
    evaluation = _regression_frame(20, offset=-0.01)

    raw = fitted.predict_raw(evaluation)
    calibrated = fitted.predict(evaluation)

    assert raw.shape == (20, 2)
    assert calibrated.shape == (20, 2)
    assert np.isfinite(raw).all()
    assert np.isfinite(calibrated).all()
    assert fitted.estimators[0] is not fitted.estimators[1]
    assert len(fitted.calibrators) == 2


def test_fit_rejects_non_chronological_calibration_data() -> None:
    with pytest.raises(ValueError, match="chronologically"):
        fit_net_regressors(
            name="ridge",
            feature_set="test",
            feature_names=("signal",),
            fit_frame=_regression_frame(20),
            calibration_frame=_regression_frame(10).reverse(),
            seed=17,
            threads=1,
        )
