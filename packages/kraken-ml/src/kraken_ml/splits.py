from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from .config import BenchmarkConfig, FoldConfig


@dataclass(frozen=True)
class TemporalSlices:
    name: str
    fit: pl.DataFrame
    calibration: pl.DataFrame
    threshold: pl.DataFrame
    evaluation: pl.DataFrame


@dataclass(frozen=True)
class FrozenTrainingSlices:
    fit: pl.DataFrame
    calibration: pl.DataFrame
    threshold: pl.DataFrame


def shift_months(value: datetime, months: int) -> datetime:
    month_index = value.year * 12 + (value.month - 1) + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _time_range(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    return frame.filter((pl.col("bucket_start") >= start) & (pl.col("bucket_start") < end))


def _purge_before(frame: pl.DataFrame, next_slice: pl.DataFrame) -> pl.DataFrame:
    if next_slice.is_empty():
        raise RuntimeError("cannot purge against an empty next slice")
    next_observation = next_slice["bucket_start"].min()
    return frame.filter(pl.col("label_exit_at") < next_observation)


def _assert_boundary(left: pl.DataFrame, right: pl.DataFrame, name: str) -> None:
    if left.is_empty() or right.is_empty():
        raise RuntimeError(f"empty temporal slice at {name}")
    latest_dependency = left["label_exit_at"].max()
    earliest_observation = right["bucket_start"].min()
    if latest_dependency >= earliest_observation:
        raise RuntimeError(
            f"look-ahead boundary violation at {name}: "
            f"{latest_dependency} >= {earliest_observation}"
        )


def _build_slices(
    frame: pl.DataFrame,
    config: BenchmarkConfig,
    *,
    name: str,
    calibration_start: datetime,
    threshold_start: datetime,
    evaluation_start: datetime,
    evaluation_end: datetime,
) -> TemporalSlices:
    fit = frame.filter(pl.col("bucket_start") < calibration_start)
    calibration = _time_range(frame, calibration_start, threshold_start)
    threshold = _time_range(frame, threshold_start, evaluation_start)
    evaluation = _time_range(frame, evaluation_start, evaluation_end)

    fit = _purge_before(fit, calibration)
    calibration = _purge_before(calibration, threshold)
    threshold = _purge_before(threshold, evaluation)
    _assert_boundary(fit, calibration, f"{name}:fit/calibration")
    _assert_boundary(calibration, threshold, f"{name}:calibration/threshold")
    _assert_boundary(threshold, evaluation, f"{name}:threshold/evaluation")

    training_days = (
        fit["bucket_start"].max() - fit["bucket_start"].min()
    ).total_seconds() / 86_400.0
    if training_days < config.validation.minimum_training_days:
        raise RuntimeError(
            f"{name} has only {training_days:.1f} training days; "
            f"minimum is {config.validation.minimum_training_days}"
        )
    return TemporalSlices(
        name=name,
        fit=fit,
        calibration=calibration,
        threshold=threshold,
        evaluation=evaluation,
    )


def development_slices(
    frame: pl.DataFrame, config: BenchmarkConfig, fold: FoldConfig
) -> TemporalSlices:
    threshold_start = shift_months(fold.start, -1)
    calibration_start = shift_months(fold.start, -2)
    return _build_slices(
        frame,
        config,
        name=fold.name,
        calibration_start=calibration_start,
        threshold_start=threshold_start,
        evaluation_start=fold.start,
        evaluation_end=fold.end,
    )


def final_slices(frame: pl.DataFrame, config: BenchmarkConfig) -> TemporalSlices:
    return _build_slices(
        frame,
        config,
        name="locked_holdout",
        calibration_start=config.validation.calibration_start,
        threshold_start=config.validation.threshold_start,
        evaluation_start=config.validation.holdout_start,
        evaluation_end=config.validation.holdout_end,
    )


def frozen_training_slices(frame: pl.DataFrame, config: BenchmarkConfig) -> FrozenTrainingSlices:
    """Build final pre-holdout slices without reading a holdout observation."""
    fit = frame.filter(pl.col("bucket_start") < config.validation.calibration_start)
    calibration = _time_range(
        frame,
        config.validation.calibration_start,
        config.validation.threshold_start,
    )
    threshold = _time_range(
        frame,
        config.validation.threshold_start,
        config.validation.holdout_start,
    )
    fit = _purge_before(fit, calibration)
    calibration = _purge_before(calibration, threshold)
    threshold = threshold.filter(pl.col("label_exit_at") < config.validation.holdout_start)
    _assert_boundary(fit, calibration, "frozen:fit/calibration")
    _assert_boundary(calibration, threshold, "frozen:calibration/threshold")
    if threshold.is_empty():
        raise RuntimeError("empty frozen threshold slice")
    if threshold["label_exit_at"].max() >= config.validation.holdout_start:
        raise RuntimeError("frozen threshold labels cross into the holdout")
    return FrozenTrainingSlices(
        fit=fit,
        calibration=calibration,
        threshold=threshold,
    )
