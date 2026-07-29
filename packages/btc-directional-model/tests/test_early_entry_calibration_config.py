from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from btc_directional_model.core_config import (
    evaluation_holdout_range,
    load_core_config,
)
from btc_directional_model.core_extract import scope_range


def early_entry_calibration_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-core-early-entry-calibration-20260421-20260720.toml"
    )


def utc_day(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def test_early_entry_calibration_config_freezes_exact_ninety_day_contract() -> None:
    config = load_core_config(early_entry_calibration_config())

    assert config.data.range_start == utc_day(2026, 4, 21)
    assert config.data.range_end == utc_day(2026, 7, 20)
    assert (config.data.range_end - config.data.range_start).days == 90
    assert config.split.development_start == utc_day(2026, 4, 21)
    assert config.split.development_end == utc_day(2026, 7, 6)
    assert config.split.probability_calibration_start == utc_day(2026, 7, 6)
    assert config.split.probability_calibration_end == utc_day(2026, 7, 13)
    assert config.split.policy_selection_start == utc_day(2026, 7, 13)
    assert config.split.policy_selection_end == utc_day(2026, 7, 20)
    assert (
        config.split.holdout_start
        == config.split.holdout_end
        == config.data.range_end
    )
    assert evaluation_holdout_range(config) == (
        utc_day(2026, 7, 21),
        utc_day(2026, 8, 4),
    )
    assert scope_range(config, "pre_holdout") == (
        utc_day(2026, 4, 21),
        utc_day(2026, 7, 20),
    )
    assert scope_range(config, "holdout") == (
        utc_day(2026, 7, 21),
        utc_day(2026, 8, 4),
    )


def test_early_entry_calibration_config_freezes_walk_forward_windows() -> None:
    config = load_core_config(early_entry_calibration_config())

    assert config.split.validation_windows == (
        (utc_day(2026, 6, 1), utc_day(2026, 6, 8)),
        (utc_day(2026, 6, 8), utc_day(2026, 6, 15)),
        (utc_day(2026, 6, 15), utc_day(2026, 6, 22)),
        (utc_day(2026, 6, 22), utc_day(2026, 6, 29)),
        (utc_day(2026, 6, 29), utc_day(2026, 7, 6)),
    )


def test_early_entry_calibration_config_freezes_policy_gates_and_compute() -> None:
    config = load_core_config(early_entry_calibration_config())

    assert config.model.confidence_min == 0.87
    assert config.model.confidence_max == 0.91
    assert config.model.confidence_step == 0.01
    assert config.model.random_seed == 20260726
    assert asdict(config.gates) == {
        "target_accuracy": 0.874,
        "target_wilson_lower": 0.865,
        "target_balanced_accuracy": 0.874,
        "minimum_direction_recall": 0.874,
        "minimum_coverage": 0.55,
        "minimum_holdout_markets": 1000,
        "maximum_walk_forward_holdout_gap": 0.05,
        "minimum_same_time_path_uplift": 0.0,
        "minimum_nonnegative_uplift_folds": 5,
        "maximum_ece": 0.05,
        "bootstrap_resamples": 10000,
    }
    assert asdict(config.compute) == {
        "max_parallel_fits": 4,
        "threads_per_fit": 1,
        "polars_threads": 6,
    }
    generated_paths = {
        config.paths.source_data,
        config.paths.development_feature_data,
        config.paths.holdout_feature_data,
        config.paths.runs,
        config.paths.artifacts,
    }
    assert len(generated_paths) == 5
    assert (
        config.paths.holdout_feature_data.name
        == "independent-holdout-20260721-20260804.parquet"
    )
