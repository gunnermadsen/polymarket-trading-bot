from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from btc_directional_model.core_features import (
    CORE_MATURE_REVERSAL_ENRICHED_FEATURES,
    CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
    CORE_REGIME_REVERSAL_ENRICHED_FEATURES,
    CORE_REGIME_REVERSAL_FEATURE_SCHEMA_VERSION,
)
from btc_directional_model.fixed_time_config import FixedTimeOperatingPointConfig
from btc_directional_model.fixed_time_selective_config import (
    BASE_PARAMETER_GRID,
    EXACT_120_SELECTIVE_TUNING_IDENTITY,
    EXPECTED_CORE_CONFIG_SHA256,
    EXPECTED_EXECUTION_MANIFEST_SHA256,
    EXPECTED_VALIDATION_WINDOWS,
    FIXED_TIME_SELECTIVE_ACCURACY_PROFILE,
    FIXED_TIME_SELECTIVE_CANDIDATE_NAMES,
    TAIL_PARAMETER_GRID,
    FixedTimeSelectiveHistogramConfig,
    load_fixed_time_selective_config,
    validate_fixed_time_selective_config,
)


def repository_config() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "btc-5m-directional-fixed-120-selective-accuracy-20260321-20260729.toml"
    )


def utc_day(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def test_repository_config_freezes_identity_evidence_and_isolated_paths() -> None:
    config = load_fixed_time_selective_config(repository_config())

    assert config.benchmark.profile == FIXED_TIME_SELECTIVE_ACCURACY_PROFILE
    assert config.benchmark.tuning_identity == EXACT_120_SELECTIVE_TUNING_IDENTITY
    assert not config.benchmark.evaluation_is_independent
    assert config.benchmark.core_config_sha256 == EXPECTED_CORE_CONFIG_SHA256
    assert config.paths.execution_manifest_sha256 == (
        EXPECTED_EXECUTION_MANIFEST_SHA256
    )
    assert config.paths.runs.name == (
        "btc-mature-reversal-fixed-120-selective-accuracy-20260321-20260729"
    )
    assert config.paths.freezes.name == "freezes"
    assert config.paths.runtime_models.name == (
        "btc-mature-reversal-fixed-120-selective-accuracy-20260321-20260729"
    )
    assert len({config.paths.runs, config.paths.freezes, config.paths.runtime_models}) == 3


def test_repository_config_freezes_seven_windows_and_excludes_july_29() -> None:
    config = load_fixed_time_selective_config(repository_config())

    assert config.split.validation_windows == EXPECTED_VALIDATION_WINDOWS
    assert len(config.split.validation_windows) == 7
    assert config.split.validation_windows[0] == (
        utc_day(2026, 6, 9),
        utc_day(2026, 6, 16),
    )
    assert config.split.validation_windows[-1] == (
        utc_day(2026, 7, 21),
        utc_day(2026, 7, 29),
    )
    assert all(
        left[1] == right[0]
        for left, right in zip(
            config.split.validation_windows,
            config.split.validation_windows[1:],
            strict=False,
        )
    )
    assert config.split.development_start == utc_day(2026, 3, 21)
    assert config.split.development_end == utc_day(2026, 7, 14)
    assert config.split.probability_calibration_start == utc_day(2026, 7, 14)
    assert config.split.probability_calibration_end == utc_day(2026, 7, 21)
    assert config.split.policy_selection_start == utc_day(2026, 7, 21)
    assert config.split.policy_selection_end == utc_day(2026, 7, 29)
    assert max(end for _, end in config.split.validation_windows) == (
        config.split.policy_selection_end
    )


def test_repository_config_freezes_exact_120_model_and_candidate_matrix() -> None:
    config = load_fixed_time_selective_config(repository_config())

    assert config.model.decision_second == 120
    assert config.model.estimator_training_seconds == (120, 125, 130, 135, 140)
    assert config.model.probability_calibration == "global_platt"
    assert config.model.recency_half_life_days == 28.0
    assert config.model.hard_confidence_floor == 0.95
    assert not config.model.include_oracle
    assert not config.model.include_book
    assert tuple(candidate.name for candidate in config.candidates) == (
        FIXED_TIME_SELECTIVE_CANDIDATE_NAMES
    )
    assert tuple(
        candidate.exact_120_weight_multiplier for candidate in config.candidates
    ) == (1.0, 2.0, 3.0, 2.0, 2.0)
    assert tuple(candidate.parameter_grid for candidate in config.candidates) == (
        BASE_PARAMETER_GRID,
        BASE_PARAMETER_GRID,
        BASE_PARAMETER_GRID,
        TAIL_PARAMETER_GRID,
        BASE_PARAMETER_GRID,
    )


def test_repository_config_freezes_base_and_six_feature_regime_schemas() -> None:
    config = load_fixed_time_selective_config(repository_config())
    base_features = tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)
    regime_features = tuple(CORE_REGIME_REVERSAL_ENRICHED_FEATURES)

    for candidate in config.candidates[:-1]:
        assert candidate.feature_schema_version == (
            CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION
        )
        assert candidate.feature_names == base_features
    regime = config.candidates[-1]
    assert regime.feature_schema_version == CORE_REGIME_REVERSAL_FEATURE_SCHEMA_VERSION
    assert regime.feature_names == regime_features
    assert len(base_features) == 71
    assert len(regime_features) == 77
    assert regime_features[:71] == base_features
    assert regime_features[71:] == (
        "btc_path_sign_normalized_return_90s_bps",
        "btc_path_sign_normalized_return_120s_bps",
        "btc_path_sign_normalized_flow_90s",
        "btc_path_sign_normalized_flow_120s",
        "btc_realized_volatility_90s_bps",
        "btc_realized_volatility_120s_bps",
    )


def test_repository_config_freezes_tail_grid_and_operating_points() -> None:
    config = load_fixed_time_selective_config(repository_config())

    assert config.model.tail_histogram_parameters == (
        FixedTimeSelectiveHistogramConfig(15, 200, 2.0, 0.05, 160),
        FixedTimeSelectiveHistogramConfig(15, 300, 5.0, 0.03, 220),
        FixedTimeSelectiveHistogramConfig(7, 200, 2.0, 0.05, 160),
    )
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


def test_validation_rejects_candidate_timing_source_and_boundary_drift() -> None:
    config = load_fixed_time_selective_config(repository_config())

    with pytest.raises(ValueError, match="candidate feature or training contract"):
        validate_fixed_time_selective_config(
            replace(
                config,
                candidates=(
                    config.candidates[0],
                    replace(
                        config.candidates[1],
                        exact_120_weight_multiplier=2.5,
                    ),
                    *config.candidates[2:],
                ),
            )
        )
    with pytest.raises(ValueError, match="decision second"):
        validate_fixed_time_selective_config(
            replace(config, model=replace(config.model, decision_second=125))
        )
    with pytest.raises(ValueError, match="exclude oracle"):
        validate_fixed_time_selective_config(
            replace(config, model=replace(config.model, include_oracle=True))
        )
    with pytest.raises(ValueError, match="frozen March 21-July 29 split"):
        validate_fixed_time_selective_config(
            replace(
                config,
                split=replace(
                    config.split,
                    policy_selection_end=utc_day(2026, 7, 30),
                ),
            )
        )
