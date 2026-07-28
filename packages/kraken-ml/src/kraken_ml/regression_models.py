from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .models import feature_matrix

REGRESSION_MODEL_NAMES = ("ridge", "histogram", "extra_trees")
NET_TARGET_COLUMNS = ("long_net_bps", "short_net_bps")


@dataclass(frozen=True)
class NonnegativeAffineCalibrator:
    """Least-squares affine calibration constrained to a nonnegative slope."""

    slope: float
    intercept: float

    @classmethod
    def fit(
        cls,
        raw_predictions: np.ndarray,
        targets: np.ndarray,
    ) -> NonnegativeAffineCalibrator:
        predicted = _finite_vector(raw_predictions, name="raw_predictions")
        observed = _finite_vector(targets, name="targets")
        if predicted.shape != observed.shape:
            raise ValueError("calibration prediction and target counts do not match")
        if predicted.size == 0:
            raise ValueError("calibration data must not be empty")

        predicted_mean = float(np.mean(predicted))
        target_mean = float(np.mean(observed))
        centered_prediction = predicted - predicted_mean
        denominator = float(np.dot(centered_prediction, centered_prediction))
        scale = max(1.0, float(np.dot(predicted, predicted)))
        if denominator <= np.finfo(np.float64).eps * scale:
            return cls(slope=0.0, intercept=target_mean)

        centered_target = observed - target_mean
        unconstrained_slope = float(
            np.dot(centered_prediction, centered_target) / denominator
        )
        if not np.isfinite(unconstrained_slope) or unconstrained_slope <= 0.0:
            return cls(slope=0.0, intercept=target_mean)

        intercept = target_mean - unconstrained_slope * predicted_mean
        if not np.isfinite(intercept):
            return cls(slope=0.0, intercept=target_mean)
        return cls(slope=unconstrained_slope, intercept=float(intercept))

    def predict(self, raw_predictions: np.ndarray) -> np.ndarray:
        predicted = _finite_vector(raw_predictions, name="raw_predictions")
        calibrated = self.intercept + self.slope * predicted
        if not np.isfinite(calibrated).all():
            raise RuntimeError("affine calibration produced non-finite predictions")
        return calibrated


@dataclass
class FittedNetRegressors:
    name: str
    feature_set: str
    feature_names: tuple[str, ...]
    estimators: tuple[Any, Any]
    calibrators: tuple[NonnegativeAffineCalibrator, NonnegativeAffineCalibrator]
    hyperparameters: dict[str, Any]

    def predict_raw(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = feature_matrix(frame, self.feature_names)
        predictions = np.column_stack(
            [estimator.predict(matrix) for estimator in self.estimators]
        ).astype(np.float64, copy=False)
        if predictions.shape != (frame.height, len(NET_TARGET_COLUMNS)):
            raise RuntimeError("regressors emitted an unexpected prediction shape")
        if not np.isfinite(predictions).all():
            raise RuntimeError("regressors emitted non-finite predictions")
        return predictions

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        raw = self.predict_raw(frame)
        calibrated = np.column_stack(
            [
                calibrator.predict(raw[:, index])
                for index, calibrator in enumerate(self.calibrators)
            ]
        )
        if not np.isfinite(calibrated).all():
            raise RuntimeError("calibrators emitted non-finite predictions")
        return calibrated


def regression_model_parameters(
    name: str,
    *,
    seed: int,
    threads: int,
) -> dict[str, Any]:
    if threads <= 0:
        raise ValueError("threads must be positive")
    if name == "ridge":
        return {
            "alpha": 10.0,
            "fit_intercept": True,
            "solver": "lsqr",
            "tol": 1e-4,
        }
    if name == "histogram":
        return {
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 100,
            "l2_regularization": 10.0,
            "early_stopping": False,
            "random_state": seed,
        }
    if name == "extra_trees":
        return {
            "n_estimators": 500,
            "criterion": "squared_error",
            "max_features": 0.5,
            "min_samples_leaf": 25,
            "max_depth": 18,
            "bootstrap": False,
            "n_jobs": threads,
            "random_state": seed,
        }
    raise ValueError(f"unsupported regression model {name}")


def build_regression_estimator(
    name: str,
    *,
    seed: int,
    threads: int,
) -> tuple[Any, dict[str, Any]]:
    parameters = regression_model_parameters(name, seed=seed, threads=threads)
    if name == "ridge":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(**parameters)),
            ]
        )
    elif name == "histogram":
        estimator = HistGradientBoostingRegressor(**parameters)
    elif name == "extra_trees":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("regressor", ExtraTreesRegressor(**parameters)),
            ]
        )
    else:
        raise ValueError(f"unsupported regression model {name}")
    return estimator, parameters


def fit_net_regressors(
    *,
    name: str,
    feature_set: str,
    feature_names: Sequence[str],
    fit_frame: pl.DataFrame,
    calibration_frame: pl.DataFrame,
    seed: int,
    threads: int,
) -> FittedNetRegressors:
    """Fit long then short regressors, and calibrate each on a later fixed slice."""

    if fit_frame.height == 0 or calibration_frame.height == 0:
        raise ValueError("fit and calibration frames must not be empty")
    _require_chronological(fit_frame, name="fit_frame")
    _require_chronological(calibration_frame, name="calibration_frame")
    if (
        "bucket_start" in fit_frame.columns
        and "bucket_start" in calibration_frame.columns
        and fit_frame["bucket_start"].max() >= calibration_frame["bucket_start"].min()
    ):
        raise ValueError("calibration_frame must begin after fit_frame")

    x_fit = feature_matrix(fit_frame, feature_names)
    x_calibration = feature_matrix(calibration_frame, feature_names)
    estimators: list[Any] = []
    calibrators: list[NonnegativeAffineCalibrator] = []
    parameters: dict[str, Any] | None = None

    # This loop is intentionally sequential. The caller controls concurrency
    # across candidates while each logical candidate stays within its CPU budget.
    for target_column in NET_TARGET_COLUMNS:
        estimator, target_parameters = build_regression_estimator(
            name,
            seed=seed,
            threads=threads,
        )
        y_fit = _target_vector(fit_frame, target_column)
        y_calibration = _target_vector(calibration_frame, target_column)
        with threadpool_limits(limits=threads):
            estimator.fit(x_fit, y_fit)
            raw_calibration = np.asarray(
                estimator.predict(x_calibration),
                dtype=np.float64,
            )
        calibrator = NonnegativeAffineCalibrator.fit(raw_calibration, y_calibration)
        estimators.append(estimator)
        calibrators.append(calibrator)
        if parameters is None:
            parameters = target_parameters

    if parameters is None:
        raise RuntimeError("no regressors were fitted")
    return FittedNetRegressors(
        name=name,
        feature_set=feature_set,
        feature_names=tuple(feature_names),
        estimators=(estimators[0], estimators[1]),
        calibrators=(calibrators[0], calibrators[1]),
        hyperparameters=parameters,
    )


def _target_vector(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"missing regression target column {column}")
    return _finite_vector(frame[column].to_numpy(), name=column)


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values")
    return vector


def _require_chronological(frame: pl.DataFrame, *, name: str) -> None:
    if "bucket_start" not in frame.columns or frame.height < 2:
        return
    timestamps = frame["bucket_start"]
    if not timestamps.is_sorted():
        raise ValueError(f"{name} must be sorted chronologically by bucket_start")
