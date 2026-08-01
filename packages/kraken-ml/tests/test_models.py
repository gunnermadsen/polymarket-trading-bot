from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from kraken_ml.models import (
    CLASS_LABELS,
    MODEL_NAMES,
    build_estimator,
    fit_calibrated_model,
)


def _model_frame(rows_per_class: int, *, offset: float = 0.0) -> pl.DataFrame:
    labels = np.repeat(CLASS_LABELS, rows_per_class)
    within_class = np.tile(np.linspace(-0.25, 0.25, rows_per_class), 3)
    return pl.DataFrame(
        {
            "signal": labels.astype(float) + within_class + offset,
            "secondary": np.sin(np.arange(labels.size) / 7.0),
            "label": labels,
        }
    )


def test_histogram_model_disables_early_stopping() -> None:
    estimator, parameters = build_estimator("histogram", seed=17, threads=1)

    assert isinstance(estimator, HistGradientBoostingClassifier)
    assert parameters["early_stopping"] is False
    assert estimator.early_stopping is False


def test_extra_trees_honors_explicit_estimator_thread_budget() -> None:
    estimator, parameters = build_estimator("extra_trees", seed=17, threads=3)

    assert parameters["n_jobs"] == 3
    assert estimator.named_steps["classifier"].n_jobs == 3


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_all_models_fit_calibrate_and_emit_ordered_probabilities(
    model_name: str,
) -> None:
    fit_frame = _model_frame(80)
    calibration_frame = _model_frame(30, offset=0.01)
    evaluation_frame = _model_frame(8, offset=-0.01)

    fitted = fit_calibrated_model(
        name=model_name,
        feature_set="test",
        feature_names=("signal", "secondary"),
        fit_frame=fit_frame,
        calibration_frame=calibration_frame,
        seed=20260728,
        threads=1,
    )
    probabilities = fitted.predict_proba(evaluation_frame)

    assert probabilities.shape == (evaluation_frame.height, 3)
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
    assert tuple(fitted.calibrator.classes_) == (-1, 0, 1)
    assert fitted.hyperparameters == build_estimator(model_name, seed=20260728, threads=1)[1]
