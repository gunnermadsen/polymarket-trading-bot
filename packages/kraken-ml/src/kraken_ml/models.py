from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

CLASS_LABELS = np.array([-1, 0, 1], dtype=np.int8)
MODEL_NAMES = ("logistic", "histogram", "extra_trees")


@dataclass
class FittedModel:
    name: str
    feature_set: str
    feature_names: tuple[str, ...]
    estimator: Any
    calibrator: Any
    hyperparameters: dict[str, Any]

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = feature_matrix(frame, self.feature_names)
        probabilities = self.calibrator.predict_proba(matrix)
        classes = np.asarray(self.calibrator.classes_)
        order = [int(np.flatnonzero(classes == label)[0]) for label in CLASS_LABELS]
        return probabilities[:, order]


def feature_matrix(frame: pl.DataFrame, names: Sequence[str]) -> np.ndarray:
    matrix = np.array(
        frame.select(list(names)).to_numpy(),
        dtype=np.float64,
        copy=True,
    )
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def target_vector(frame: pl.DataFrame) -> np.ndarray:
    return frame["label"].to_numpy().astype(np.int8, copy=False)


def model_parameters(name: str, *, seed: int, threads: int) -> dict[str, Any]:
    if name == "logistic":
        return {
            "solver": "saga",
            "C": 0.1,
            "l1_ratio": 0.5,
            "max_iter": 5_000,
            "tol": 1e-4,
            "class_weight": None,
            "random_state": seed,
        }
    if name == "histogram":
        return {
            "loss": "log_loss",
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
            "criterion": "log_loss",
            "max_features": 0.5,
            "min_samples_leaf": 25,
            "max_depth": 18,
            "bootstrap": False,
            "class_weight": None,
            "n_jobs": threads,
            "random_state": seed,
        }
    raise ValueError(f"unsupported model {name}")


def build_estimator(name: str, *, seed: int, threads: int) -> tuple[Any, dict[str, Any]]:
    parameters = model_parameters(name, seed=seed, threads=threads)
    if name == "logistic":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**parameters)),
            ]
        )
    elif name == "histogram":
        estimator = HistGradientBoostingClassifier(**parameters)
    elif name == "extra_trees":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("classifier", ExtraTreesClassifier(**parameters)),
            ]
        )
    else:
        raise ValueError(f"unsupported model {name}")
    return estimator, parameters


def fit_calibrated_model(
    *,
    name: str,
    feature_set: str,
    feature_names: Sequence[str],
    fit_frame: pl.DataFrame,
    calibration_frame: pl.DataFrame,
    seed: int,
    threads: int,
) -> FittedModel:
    estimator, parameters = build_estimator(name, seed=seed, threads=threads)
    x_fit = feature_matrix(fit_frame, feature_names)
    y_fit = target_vector(fit_frame)
    x_calibration = feature_matrix(calibration_frame, feature_names)
    y_calibration = target_vector(calibration_frame)
    if set(np.unique(y_fit)) != {-1, 0, 1}:
        raise RuntimeError(f"{name} fit slice does not contain all target classes")
    if set(np.unique(y_calibration)) != {-1, 0, 1}:
        raise RuntimeError(f"{name} calibration slice does not contain all target classes")

    with threadpool_limits(limits=threads):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(x_fit, y_fit)
        calibrator = CalibratedClassifierCV(
            FrozenEstimator(estimator),
            method="sigmoid",
        )
        calibrator.fit(x_calibration, y_calibration)
    return FittedModel(
        name=name,
        feature_set=feature_set,
        feature_names=tuple(feature_names),
        estimator=estimator,
        calibrator=calibrator,
        hyperparameters=parameters,
    )
