"""Offline model fitting and time-banded calibration for early decisions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from .chainlink_oi_features import CHAINLINK_CANDLE_FEATURES
from .core_config import CoreTrainingConfig
from .core_features import CORE_BOUNDARY_ENRICHED_FEATURES
from .core_training import (
    MARKET_EQUAL_ROW_WEIGHT_POLICY,
    CandidateSpec,
    FittedCoreModel,
    ProbabilityCalibrator,
    candidate_calibration_weights,
    fit_model,
)
from .early_value_config import EarlyValueConfig
from .spot_l2_chainlink_features import L2_FEATURES

CORE = "core_hgb"
CORE_LOGISTIC = "core_logistic"
CORE_L2 = "core_l2_hgb"
CORE_CANDLES = "core_chainlink_candles_hgb"
COMBINED = "core_l2_chainlink_candles_hgb"


@dataclass(frozen=True)
class TimeBandCalibrator:
    start_second: int
    end_second_exclusive: int
    calibrator: ProbabilityCalibrator
    rows: int
    markets: int


@dataclass
class EarlyModel:
    name: str
    model: FittedCoreModel
    calibrators: tuple[TimeBandCalibrator, ...]

    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        logits = self.model.raw_logit(frame)
        elapsed = frame["seconds_elapsed"].to_numpy()
        output = np.full(frame.height, np.nan, dtype=np.float64)
        for band in self.calibrators:
            mask = (elapsed >= band.start_second) & (elapsed < band.end_second_exclusive)
            output[mask] = band.calibrator.probability(logits[mask])
        if not np.isfinite(output).all():
            raise RuntimeError(f"{self.name} calibration bands do not cover all rows")
        return np.clip(output, 1e-9, 1.0 - 1e-9)


def model_feature_sets() -> dict[str, tuple[str, ...]]:
    core = tuple(CORE_BOUNDARY_ENRICHED_FEATURES)
    return {
        CORE_LOGISTIC: core,
        CORE: core,
        CORE_L2: tuple(dict.fromkeys((*core, *L2_FEATURES))),
        CORE_CANDLES: tuple(dict.fromkeys((*core, *CHAINLINK_CANDLE_FEATURES))),
        COMBINED: tuple(
            dict.fromkeys((*core, *L2_FEATURES, *CHAINLINK_CANDLE_FEATURES))
        ),
    }


def fit_early_models(
    frame: pl.DataFrame,
    config: EarlyValueConfig,
    core_config: CoreTrainingConfig,
) -> tuple[dict[str, EarlyModel], dict[str, Any]]:
    required = {"market_id", "window_start", "seconds_elapsed", "label_up"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("early model frame is missing columns: " + ", ".join(missing))
    fit_frame = _window(frame, config.fit.start, config.fit.end)
    calibration_frame = _window(frame, config.calibration.start, config.calibration.end)
    policy_frame = _window(frame, config.policy.start, config.policy.end)
    if any(item.is_empty() for item in (fit_frame, calibration_frame, policy_frame)):
        raise RuntimeError("fit, calibration, and policy frames must all be non-empty")

    histogram = asdict(core_config.model.histogram_candidates[0])
    models: dict[str, EarlyModel] = {}
    summary: dict[str, Any] = {
        "selection_metric": "market_equal_policy_log_loss",
        "threshold_used_for_selection": False,
        "profiles": {},
    }
    for name, features in model_feature_sets().items():
        absent = sorted(set(features) - set(frame.columns))
        if absent:
            raise RuntimeError(f"{name} features are missing: " + ", ".join(absent))
        family = "logistic" if name == CORE_LOGISTIC else "histogram"
        parameters = {"c": core_config.model.c_candidates[0]} if family == "logistic" else histogram
        spec = CandidateSpec(
            name=name,
            family=family,
            feature_names=features,
            row_weight_policy=MARKET_EQUAL_ROW_WEIGHT_POLICY,
        )
        model = fit_model(fit_frame, spec, parameters, core_config)
        calibrators = fit_time_band_calibrators(
            model,
            calibration_frame,
            config=config,
            core_config=core_config,
            spec=spec,
        )
        bundle = EarlyModel(name=name, model=model, calibrators=calibrators)
        probability = bundle.probability(policy_frame)
        weights = _market_equal_weights(policy_frame)
        metrics = probability_metrics(policy_frame, probability, sample_weight=weights)
        summary["profiles"][name] = {
            "features": list(features),
            "feature_count": len(features),
            "fit_rows": fit_frame.height,
            "fit_markets": fit_frame["market_id"].n_unique(),
            "calibration_rows": calibration_frame.height,
            "policy_rows": policy_frame.height,
            "policy": metrics,
            "calibration_bands": [
                {
                    "start_second": item.start_second,
                    "end_second_exclusive": item.end_second_exclusive,
                    "rows": item.rows,
                    "markets": item.markets,
                    **asdict(item.calibrator),
                }
                for item in calibrators
            ],
        }
        models[name] = bundle
    selected = min(
        models,
        key=lambda name: summary["profiles"][name]["policy"]["log_loss"],
    )
    summary["selected_profile"] = selected
    return models, summary


def fit_time_band_calibrators(
    model: FittedCoreModel,
    frame: pl.DataFrame,
    *,
    config: EarlyValueConfig,
    core_config: CoreTrainingConfig,
    spec: CandidateSpec,
) -> tuple[TimeBandCalibrator, ...]:
    logits = model.raw_logit(frame)
    labels = frame["label_up"].to_numpy()
    elapsed = frame["seconds_elapsed"].to_numpy()
    weights = candidate_calibration_weights(frame, spec)
    fitted: list[TimeBandCalibrator] = []
    for start, end in config.calibration_bands:
        mask = (elapsed >= start) & (elapsed < end)
        if mask.sum() < 20 or np.unique(labels[mask]).size != 2:
            raise RuntimeError(f"calibration band {start}-{end} lacks two-class evidence")
        estimator = LogisticRegression(
            C=1_000_000,
            solver="lbfgs",
            max_iter=500,
            tol=1e-9,
            random_state=config.random_seed,
        )
        with threadpool_limits(limits=core_config.compute.threads_per_fit):
            estimator.fit(logits[mask].reshape(-1, 1), labels[mask], sample_weight=weights[mask])
        fitted.append(
            TimeBandCalibrator(
                start_second=start,
                end_second_exclusive=end,
                calibrator=ProbabilityCalibrator(
                    slope=float(estimator.coef_[0, 0]),
                    intercept=float(estimator.intercept_[0]),
                    converged=bool(estimator.n_iter_[0] < estimator.max_iter),
                    iterations=int(estimator.n_iter_[0]),
                ),
                rows=int(mask.sum()),
                markets=frame.filter(pl.Series(mask))["market_id"].n_unique(),
            )
        )
    return tuple(fitted)


def prediction_frame(frame: pl.DataFrame, probability: np.ndarray, *, model: str) -> pl.DataFrame:
    if len(probability) != frame.height or not np.isfinite(probability).all():
        raise ValueError("probability vector must be finite and row-aligned")
    return frame.select("market_id", "window_start", "observed_at", "seconds_elapsed", "label_up").with_columns(
        pl.lit(model).alias("model"),
        pl.Series("probability_yes", probability),
    ).with_columns(
        (pl.col("probability_yes") >= 0.5).cast(pl.Int8).alias("predicted_yes"),
        pl.max_horizontal("probability_yes", 1.0 - pl.col("probability_yes")).alias("confidence"),
    ).with_columns(
        (pl.col("predicted_yes") == pl.col("label_up")).alias("correct"),
    )


def probability_metrics(
    frame: pl.DataFrame,
    probability: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float | int]:
    labels = frame["label_up"].to_numpy().astype(np.int8)
    predicted = probability >= 0.5
    weight = np.ones(len(labels), dtype=np.float64) if sample_weight is None else sample_weight
    weight = weight / weight.sum()
    accuracy = float(np.sum(weight * (predicted == labels)))
    brier = float(np.sum(weight * np.square(probability - labels)))
    loss = float(log_loss(labels, probability, sample_weight=sample_weight, labels=[0, 1]))
    return {
        "rows": len(labels),
        "markets": frame["market_id"].n_unique(),
        "accuracy": accuracy,
        "brier_score": brier,
        "log_loss": loss,
        "mean_probability_yes": float(np.average(probability, weights=weight)),
        "actual_yes_rate": float(np.average(labels, weights=weight)),
    }


def accuracy_by_second(predictions: pl.DataFrame) -> list[dict[str, Any]]:
    return (
        predictions.group_by("model", "seconds_elapsed")
        .agg(
            pl.len().alias("rows"),
            pl.col("market_id").n_unique().alias("markets"),
            pl.col("correct").mean().alias("accuracy"),
            pl.col("probability_yes").mean().alias("mean_probability_yes"),
            pl.col("label_up").mean().alias("actual_yes_rate"),
            ((pl.col("probability_yes") - pl.col("label_up")) ** 2).mean().alias("brier_score"),
        )
        .sort("model", "seconds_elapsed")
        .to_dicts()
    )


def _window(frame: pl.DataFrame, start: Any, end: Any) -> pl.DataFrame:
    return frame.filter(pl.col("window_start").is_between(start, end, closed="left"))


def _market_equal_weights(frame: pl.DataFrame) -> np.ndarray:
    counts = frame.group_by("market_id").len().rename({"len": "market_rows"})
    weighted = frame.select("market_id").with_row_index("_row").join(counts, on="market_id")
    values = (1.0 / weighted.sort("_row")["market_rows"].to_numpy()).astype(np.float64)
    if not math.isfinite(values.sum()) or values.sum() <= 0:
        raise RuntimeError("market-equal weights are invalid")
    return values
