from __future__ import annotations

import tomllib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from btc_directional_model.core_features import (
    CORE_MATURE_REVERSAL_ENRICHED_FEATURES,
    CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
)
from btc_directional_model.fixed_time_config import (
    EMPIRICAL_COVERAGE_THRESHOLD_SELECTION,
    EXPECTED_VALIDATION_WINDOWS,
    FIXED_TIME_ACCURACY_PROFILE,
    FIXED_TIME_CANDIDATE,
    FixedTimeOperatingPointConfig,
    load_fixed_time_accuracy_config,
    validate_fixed_time_accuracy_config,
)


def repository_config() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "btc-5m-directional-mature-reversal-fixed-120-20260321-20260729.toml"
    )


def utc_day(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def test_repository_config_freezes_identity_evidence_and_output_paths() -> None:
    config = load_fixed_time_accuracy_config(repository_config())

    assert config.benchmark.profile == FIXED_TIME_ACCURACY_PROFILE
    assert config.benchmark.candidate == FIXED_TIME_CANDIDATE
    assert not config.benchmark.evaluation_is_independent
    assert config.benchmark.core_config_sha256 == (
        "ec5262959ec33d15047943c7242d4e05ad43665f40313e304975c6644127b294"
    )
    assert config.paths.execution_manifest_sha256 == (
        "c02f14c75382c8609fb8d4d905e9ca5735fca33a352d17f6a224aa415742cc34"
    )
    assert config.paths.predecessor_benchmark_sha256 == (
        "5e4f3fabc403d108b28fc85d1dc3240fd8328341c23bb5f691bff128c141b057"
    )
    assert config.paths.execution_evidence.name == "execution-evidence"
    assert config.paths.runs.name == ("btc-mature-reversal-fixed-120-20260321-20260729")
    assert config.paths.freezes.name == "freezes"
    assert config.paths.runtime_models.name == "runtime-models"
    assert (
        len(
            {
                config.paths.runs,
                config.paths.freezes,
                config.paths.runtime_models,
            }
        )
        == 3
    )


def test_repository_config_freezes_chronological_five_fold_and_final_ranges() -> None:
    config = load_fixed_time_accuracy_config(repository_config())

    assert config.split.validation_windows == EXPECTED_VALIDATION_WINDOWS
    assert len(config.split.validation_windows) == 5
    assert config.split.development_start == utc_day(2026, 3, 21)
    assert config.split.development_end == utc_day(2026, 7, 14)
    assert config.split.probability_calibration_start == utc_day(2026, 7, 14)
    assert config.split.probability_calibration_end == utc_day(2026, 7, 21)
    assert config.split.policy_selection_start == utc_day(2026, 7, 21)
    assert config.split.policy_selection_end == utc_day(2026, 7, 29)
    assert all(
        left[1] == right[0]
        for left, right in zip(
            config.split.validation_windows,
            config.split.validation_windows[1:],
        )
    )


def test_repository_config_freezes_fixed_120_core_only_model_contract() -> None:
    config = load_fixed_time_accuracy_config(repository_config())

    assert config.model.decision_second == 120
    assert config.model.estimator_training_seconds == (120, 125, 130, 135, 140)
    assert config.model.target == "outcome_up"
    assert config.model.estimator_family == "histogram_gradient_boosting"
    assert config.model.probability_calibration == "global_platt"
    assert config.model.recency_half_life_days == 28.0
    assert config.model.feature_schema_version == (CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION)
    assert config.model.feature_names == tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)
    assert len(config.model.feature_names) == 71
    assert not config.model.include_oracle
    assert not config.model.include_book
    assert not any(
        name.startswith("oracle_") or "vwap" in name for name in config.model.feature_names
    )
    assert config.model.threshold_selection == EMPIRICAL_COVERAGE_THRESHOLD_SELECTION
    assert config.model.hard_confidence_floor == 0.95
    assert config.model.require_hard_confident_error_no_regression


def test_repository_config_freezes_primary_and_secondary_quality_contracts() -> None:
    config = load_fixed_time_accuracy_config(repository_config())

    assert config.primary == FixedTimeOperatingPointConfig(
        name="primary",
        target_coverage=0.15,
        coverage_tolerance=0.03,
        minimum_accuracy=0.91,
        minimum_balanced_accuracy=0.90,
        minimum_direction_recall=0.90,
        minimum_wilson_lower_95=0.895,
        maximum_expected_calibration_error=0.05,
    )
    assert config.secondary == FixedTimeOperatingPointConfig(
        name="secondary",
        target_coverage=0.10,
        coverage_tolerance=0.025,
        minimum_accuracy=0.93,
        minimum_balanced_accuracy=0.92,
        minimum_direction_recall=0.92,
        minimum_wilson_lower_95=0.90,
        maximum_expected_calibration_error=0.05,
    )


def test_repository_config_uses_empirical_threshold_category_not_fixed_grid() -> None:
    with repository_config().open("rb") as handle:
        raw = tomllib.load(handle)

    model = raw["model"]
    assert model["threshold_selection"] == EMPIRICAL_COVERAGE_THRESHOLD_SELECTION
    assert not {
        "confidence_min",
        "confidence_max",
        "confidence_step",
        "threshold_candidates",
    }.intersection(model)


def test_validation_rejects_feature_or_source_scope_drift() -> None:
    config = load_fixed_time_accuracy_config(repository_config())

    with pytest.raises(ValueError, match="exact 71-feature core contract"):
        validate_fixed_time_accuracy_config(
            replace(
                config,
                model=replace(
                    config.model,
                    feature_names=(
                        *config.model.feature_names[:-1],
                        "oracle_return_60s_bps",
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="excludes oracle"):
        validate_fixed_time_accuracy_config(
            replace(
                config,
                model=replace(config.model, include_oracle=True),
            )
        )
    with pytest.raises(ValueError, match="excludes oracle"):
        validate_fixed_time_accuracy_config(
            replace(
                config,
                model=replace(config.model, include_book=True),
            )
        )


def test_validation_rejects_timing_quality_and_checksum_drift() -> None:
    config = load_fixed_time_accuracy_config(repository_config())

    with pytest.raises(ValueError, match="decision second"):
        validate_fixed_time_accuracy_config(
            replace(
                config,
                model=replace(config.model, decision_second=125),
            )
        )
    with pytest.raises(ValueError, match="primary operating-point"):
        validate_fixed_time_accuracy_config(
            replace(
                config,
                primary=replace(config.primary, minimum_accuracy=0.90),
            )
        )
    with pytest.raises(ValueError, match="core_config_sha256"):
        validate_fixed_time_accuracy_config(
            replace(
                config,
                benchmark=replace(
                    config.benchmark,
                    core_config_sha256="invalid",
                ),
            )
        )
    with pytest.raises(ValueError, match="execution-evidence manifest hash"):
        validate_fixed_time_accuracy_config(
            replace(
                config,
                paths=replace(
                    config.paths,
                    execution_manifest_sha256="0" * 64,
                ),
            )
        )
