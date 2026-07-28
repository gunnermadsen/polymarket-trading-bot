from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from kraken_ml.config import FoldConfig, load_config
from kraken_ml.splits import development_slices, final_slices, frozen_training_slices


def _dependency_frame(start: datetime, end: datetime) -> pl.DataFrame:
    periods = int((end - start).total_seconds() // (15 * 60))
    timestamps = [start + timedelta(minutes=15 * index) for index in range(periods)]
    return pl.DataFrame({"bucket_start": timestamps}).with_columns(
        [
            (pl.col("bucket_start") + pl.duration(minutes=15)).alias("feature_available_at"),
            (pl.col("bucket_start") + pl.duration(minutes=75)).alias("label_exit_at"),
        ]
    )


def _assert_strict_dependencies(left: pl.DataFrame, right: pl.DataFrame) -> None:
    assert left["label_exit_at"].max() < right["bucket_start"].min()


def test_development_slices_are_chronological_and_purged(config_path: Path) -> None:
    base = load_config(config_path)
    fold = FoldConfig(
        name="test_fold",
        start=datetime(2024, 6, 1, tzinfo=UTC),
        end=datetime(2024, 9, 1, tzinfo=UTC),
    )
    config = replace(
        base,
        validation=replace(
            base.validation,
            minimum_training_days=1,
            folds=(fold,),
        ),
    )
    frame = _dependency_frame(
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 9, 1, tzinfo=UTC),
    )

    slices = development_slices(frame, config, fold)

    assert slices.calibration["bucket_start"].min() == datetime(2024, 4, 1, tzinfo=UTC)
    assert slices.threshold["bucket_start"].min() == datetime(2024, 5, 1, tzinfo=UTC)
    assert slices.evaluation["bucket_start"].min() == fold.start
    assert slices.evaluation["bucket_start"].max() == fold.end - timedelta(minutes=90)
    assert slices.evaluation["label_exit_at"].max() < fold.end
    _assert_strict_dependencies(slices.fit, slices.calibration)
    _assert_strict_dependencies(slices.calibration, slices.threshold)
    _assert_strict_dependencies(slices.threshold, slices.evaluation)

    # Decisions nearest each boundary whose labels depend on the next slice are
    # removed; the last retained dependency ends before the next observation.
    assert slices.fit["bucket_start"].max() == datetime(2024, 3, 31, 22, 30, tzinfo=UTC)
    assert slices.calibration["bucket_start"].max() == datetime(2024, 4, 30, 22, 30, tzinfo=UTC)
    assert slices.threshold["bucket_start"].max() == datetime(2024, 5, 31, 22, 30, tzinfo=UTC)


def test_final_slices_keep_locked_holdout_out_of_fit_calibration_and_threshold(
    config_path: Path,
) -> None:
    base = load_config(config_path)
    config = replace(
        base,
        validation=replace(base.validation, minimum_training_days=1),
    )
    frame = _dependency_frame(
        datetime(2025, 1, 1, tzinfo=UTC),
        config.validation.holdout_end,
    )

    slices = final_slices(frame, config)

    assert slices.calibration["bucket_start"].min() == config.validation.calibration_start
    assert slices.threshold["bucket_start"].min() == config.validation.threshold_start
    assert slices.evaluation["bucket_start"].min() == config.validation.holdout_start
    assert slices.evaluation["bucket_start"].max() < config.validation.holdout_end
    assert slices.fit["bucket_start"].max() < config.validation.calibration_start
    assert slices.calibration["bucket_start"].max() < config.validation.threshold_start
    assert slices.threshold["bucket_start"].max() < config.validation.holdout_start
    _assert_strict_dependencies(slices.fit, slices.calibration)
    _assert_strict_dependencies(slices.calibration, slices.threshold)
    _assert_strict_dependencies(slices.threshold, slices.evaluation)

    sealed = frozen_training_slices(
        frame.filter(pl.col("bucket_start") < config.validation.holdout_start),
        config,
    )
    assert sealed.threshold["label_exit_at"].max() < config.validation.holdout_start
