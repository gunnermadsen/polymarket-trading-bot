from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from btc_directional_model.core_features import (
    CORE_MATURE_REVERSAL_ENRICHED_FEATURES,
    CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
)
from btc_directional_model.fixed_time_config import FixedTimeOperatingPointConfig
from btc_directional_model.fixed_time_reversal_config import (
    BASE_PARAMETER_GRID,
    DIAGNOSTIC_OPERATING_POINT,
    EXACT_120_REVERSAL_TUNING_IDENTITY,
    EXPECTED_CORE_CONFIG_SHA256,
    EXPECTED_ELIGIBLE_ESTIMATOR_ROWS,
    EXPECTED_ELIGIBLE_MARKETS,
    EXPECTED_EXACT_120_ELIGIBLE_MARKETS,
    EXPECTED_EXECUTION_MANIFEST_SHA256,
    EXPECTED_SOURCE_ESTIMATOR_ROWS,
    EXPECTED_SOURCE_MARKETS,
    EXPECTED_VALIDATION_WINDOWS,
    FIXED_TIME_REVERSAL_CANDIDATE_NAMES,
    FIXED_TIME_REVERSAL_DECISION_PROFILE,
    PATH_PERSISTENCE_TARGET,
    PATH_ZERO_EPSILON_BPS,
    TAIL_PARAMETER_GRID,
    FixedTimeReversalHistogramConfig,
    load_fixed_time_reversal_config,
    validate_fixed_time_reversal_config,
)


def repository_config() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "btc-5m-directional-fixed-120-reversal-decision-20260321-20260729.toml"
    )


def utc_day(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def test_repository_config_freezes_identity_artifacts_and_paths() -> None:
    config = load_fixed_time_reversal_config(repository_config())

    assert config.benchmark.profile == FIXED_TIME_REVERSAL_DECISION_PROFILE
    assert config.benchmark.tuning_identity == EXACT_120_REVERSAL_TUNING_IDENTITY
    assert not config.benchmark.evaluation_is_independent
    assert config.benchmark.core_config_sha256 == EXPECTED_CORE_CONFIG_SHA256
    assert config.paths.execution_manifest_sha256 == (
        EXPECTED_EXECUTION_MANIFEST_SHA256
    )
    assert config.paths.runs.name == (
        "btc-mature-reversal-fixed-120-reversal-decision-20260321-20260729"
    )
    assert config.paths.freezes.name == "freezes"
    assert config.paths.runtime_models.name == (
        "btc-mature-reversal-fixed-120-reversal-decision-20260321-20260729"
    )
    assert len({config.paths.runs, config.paths.freezes, config.paths.runtime_models}) == 3


def test_repository_config_freezes_seven_windows_and_excludes_july_29() -> None:
    config = load_fixed_time_reversal_config(repository_config())

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
    assert config.split.development_start == utc_day(2026, 3, 21)
    assert config.split.development_end == utc_day(2026, 7, 14)
    assert config.split.probability_calibration_start == utc_day(2026, 7, 14)
    assert config.split.probability_calibration_end == utc_day(2026, 7, 21)
    assert config.split.policy_selection_start == utc_day(2026, 7, 21)
    assert config.split.policy_selection_end == utc_day(2026, 7, 29)
    assert max(end for _, end in config.split.validation_windows) == (
        config.split.policy_selection_end
    )


def test_repository_config_freezes_common_features_targets_and_weights() -> None:
    config = load_fixed_time_reversal_config(repository_config())
    features = tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)

    assert config.model.decision_second == 120
    assert config.model.estimator_training_seconds == (120, 125, 130, 135, 140)
    assert config.model.decision_target == "outcome_up"
    assert config.model.reversal_estimator_positive_class == PATH_PERSISTENCE_TARGET
    assert config.model.probability_calibration == "global_platt"
    assert config.model.recency_half_life_days == 28.0
    assert config.model.path_zero_epsilon_bps == PATH_ZERO_EPSILON_BPS
    assert config.model.expected_source_markets == EXPECTED_SOURCE_MARKETS
    assert (
        config.model.expected_source_estimator_rows
        == EXPECTED_SOURCE_ESTIMATOR_ROWS
    )
    assert config.model.expected_eligible_markets == EXPECTED_ELIGIBLE_MARKETS
    assert (
        config.model.expected_eligible_estimator_rows
        == EXPECTED_ELIGIBLE_ESTIMATOR_ROWS
    )
    assert (
        config.model.expected_exact_120_eligible_markets
        == EXPECTED_EXACT_120_ELIGIBLE_MARKETS
    )
    assert not config.model.include_oracle
    assert not config.model.include_book
    assert tuple(candidate.name for candidate in config.candidates) == (
        FIXED_TIME_REVERSAL_CANDIDATE_NAMES
    )
    assert tuple(candidate.target_kind for candidate in config.candidates) == (
        "outcome_up",
        PATH_PERSISTENCE_TARGET,
        PATH_PERSISTENCE_TARGET,
    )
    assert tuple(candidate.parameter_grid for candidate in config.candidates) == (
        BASE_PARAMETER_GRID,
        BASE_PARAMETER_GRID,
        TAIL_PARAMETER_GRID,
    )
    assert all(
        candidate.feature_schema_version
        == CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION
        and candidate.feature_names == features
        and candidate.market_weight_multiplier == 1.0
        for candidate in config.candidates
    )
    assert len(features) == 71


def test_repository_config_freezes_tail_grid_and_operating_points() -> None:
    config = load_fixed_time_reversal_config(repository_config())

    assert config.model.tail_histogram_parameters == (
        FixedTimeReversalHistogramConfig(15, 200, 2.0, 0.05, 160),
        FixedTimeReversalHistogramConfig(15, 300, 5.0, 0.03, 220),
        FixedTimeReversalHistogramConfig(7, 200, 2.0, 0.05, 160),
    )
    assert config.primary == FixedTimeOperatingPointConfig(
        name="primary",
        target_coverage=0.10,
        coverage_tolerance=0.025,
        minimum_accuracy=0.93,
        minimum_balanced_accuracy=0.92,
        minimum_direction_recall=0.92,
        minimum_wilson_lower_95=0.90,
        maximum_expected_calibration_error=0.05,
    )
    assert config.diagnostic == FixedTimeOperatingPointConfig(
        name=DIAGNOSTIC_OPERATING_POINT,
        target_coverage=0.08,
        coverage_tolerance=0.02,
        minimum_accuracy=0.935,
        minimum_balanced_accuracy=0.925,
        minimum_direction_recall=0.92,
        minimum_wilson_lower_95=0.90,
        maximum_expected_calibration_error=0.05,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("expected_source_markets", 36_580, "source market count"),
        ("expected_source_estimator_rows", 182_900, "source estimator row count"),
        ("expected_eligible_markets", 36_562, "eligible market count"),
        (
            "expected_eligible_estimator_rows",
            181_752,
            "eligible estimator row count",
        ),
        (
            "expected_exact_120_eligible_markets",
            36_302,
            "exact-120 eligible market count",
        ),
    ),
)
def test_validation_rejects_cohort_count_drift(
    field: str,
    value: int,
    message: str,
) -> None:
    config = load_fixed_time_reversal_config(repository_config())

    with pytest.raises(ValueError, match=message):
        validate_fixed_time_reversal_config(
            replace(config, model=replace(config.model, **{field: value}))
        )


def test_validation_rejects_candidate_source_and_boundary_drift() -> None:
    config = load_fixed_time_reversal_config(repository_config())

    with pytest.raises(ValueError, match="feature or weighting contract"):
        validate_fixed_time_reversal_config(
            replace(
                config,
                candidates=(
                    config.candidates[0],
                    replace(
                        config.candidates[1],
                        market_weight_multiplier=2.0,
                    ),
                    config.candidates[2],
                ),
            )
        )
    with pytest.raises(ValueError, match="exclude oracle"):
        validate_fixed_time_reversal_config(
            replace(config, model=replace(config.model, include_oracle=True))
        )
    with pytest.raises(ValueError, match="frozen March 21-July 29 split"):
        validate_fixed_time_reversal_config(
            replace(
                config,
                split=replace(
                    config.split,
                    policy_selection_end=utc_day(2026, 7, 30),
                ),
            )
        )
