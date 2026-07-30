from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from btc_directional_model.core_config import (
    CORE_ORACLE_SOURCE_CONTRACT,
    CORE_SOURCE_CONTRACT,
    evaluation_holdout_range,
    load_core_config,
)


def repository_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-core-20260421-20260620.toml"
    )


def coverage_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-core-coverage-20260421-20260720.toml"
    )


def extended_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-core-20260421-20260720.toml"
    )


def balanced_coverage_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-core-coverage-088-20260421-20260720.toml"
    )


def conservative_coverage_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-core-coverage-089-20260421-20260720.toml"
    )


def oracle_early_entry_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-core-oracle-early-entry-20260321-20260729.toml"
    )


def test_expanded_core_config_has_frozen_contiguous_cohorts() -> None:
    config = load_core_config(repository_config())

    assert config.data.source_contract == CORE_SOURCE_CONTRACT
    assert config.data.range_start == config.split.development_start
    assert config.split.development_end == config.split.probability_calibration_start
    assert config.split.probability_calibration_end == config.split.policy_selection_start
    assert config.split.policy_selection_end == config.split.holdout_start
    assert config.split.holdout_end == config.data.range_end
    assert config.split.independent_holdout_start is None
    assert config.split.independent_holdout_end is None
    assert evaluation_holdout_range(config) == (
        config.split.holdout_start,
        config.split.holdout_end,
    )
    assert len(config.split.validation_windows) == 5
    assert config.gates.minimum_same_time_path_uplift == 0
    assert config.gates.minimum_nonnegative_uplift_folds == 5
    assert config.paths.development_feature_data != config.paths.holdout_feature_data


def test_oracle_early_entry_config_has_exact_range_window_and_isolated_paths() -> None:
    config = load_core_config(oracle_early_entry_config())
    legacy = load_core_config(
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-core-boundary-reversal-20260321-20260729.toml"
    )

    assert config.data.source_contract == CORE_ORACLE_SOURCE_CONTRACT
    assert config.data.range_start == datetime(2026, 3, 21, tzinfo=UTC)
    assert config.data.range_end == datetime(2026, 7, 29, tzinfo=UTC)
    assert config.data.sample_interval_seconds == 5
    assert config.data.min_seconds_after_open == 120
    assert config.data.min_seconds_before_close == 160
    assert list(
        range(
            config.data.min_seconds_after_open,
            301 - config.data.min_seconds_before_close,
            config.data.sample_interval_seconds,
        )
    ) == [120, 125, 130, 135, 140]
    assert config.split.development_start == config.data.range_start
    assert (
        config.split.development_end
        == config.split.probability_calibration_start
    )
    assert (
        config.split.probability_calibration_end
        == config.split.policy_selection_start
    )
    assert config.split.policy_selection_end == config.split.holdout_start
    assert config.split.holdout_end == config.data.range_end
    assert evaluation_holdout_range(config) == (
        datetime(2026, 7, 29, tzinfo=UTC),
        datetime(2026, 7, 29, tzinfo=UTC),
    )
    assert len(config.split.validation_windows) == 5
    assert config.model.candidate_names == ("histogram_enriched",)
    assert config.model.row_weight_schedules[0].schedule.multiplier == 1.0
    assert config.gates.minimum_coverage == 0.60
    assert config.paths.source_data != legacy.paths.source_data
    assert (
        config.paths.development_feature_data
        != legacy.paths.development_feature_data
    )
    assert config.paths.runs != legacy.paths.runs
    assert config.paths.artifacts != legacy.paths.artifacts
    assert "btc-core-oracle-early-entry-20260321-20260729" in str(
        config.paths.source_data
    )


def test_core_config_rejects_unknown_source_contract(tmp_path: Path) -> None:
    source = oracle_early_entry_config().read_text().replace(
        'source_contract = "btc_core_oracle_v1"',
        'source_contract = "btc_core_oracle_future"',
    )
    path = tmp_path / "unknown-source" / "configs" / "core.toml"
    path.parent.mkdir(parents=True)
    path.write_text(source)

    with pytest.raises(ValueError, match="data.source_contract must be one of"):
        load_core_config(path)


def test_coverage_challenger_freezes_one_stricter_policy() -> None:
    config = load_core_config(coverage_config())

    assert config.model.confidence_min == 0.87
    assert config.model.confidence_max == 0.87
    assert config.gates.minimum_coverage == 0.55
    assert config.paths.artifacts == load_core_config(extended_config()).paths.artifacts
    assert config.paths.runs != load_core_config(extended_config()).paths.runs


def test_balanced_coverage_challenger_uses_next_fixed_policy() -> None:
    config = load_core_config(balanced_coverage_config())

    assert config.model.confidence_min == 0.88
    assert config.model.confidence_max == 0.88
    assert config.gates.target_accuracy == 0.875
    assert config.gates.minimum_coverage == 0.55
    assert config.paths.artifacts == load_core_config(extended_config()).paths.artifacts
    assert config.paths.runs != load_core_config(coverage_config()).paths.runs


def test_conservative_coverage_challenger_preserves_accuracy_gates() -> None:
    config = load_core_config(conservative_coverage_config())

    assert config.model.confidence_min == 0.89
    assert config.model.confidence_max == 0.89
    assert config.gates.target_accuracy == 0.874
    assert config.gates.target_wilson_lower == 0.865
    assert config.gates.target_balanced_accuracy == 0.874
    assert config.gates.minimum_direction_recall == 0.874
    assert config.gates.minimum_coverage == 0.55
    assert config.paths.artifacts == load_core_config(extended_config()).paths.artifacts
    assert config.paths.runs != load_core_config(balanced_coverage_config()).paths.runs


def test_core_config_rejects_holdout_overlap(tmp_path: Path) -> None:
    source = repository_config().read_text()
    source = source.replace(
        'policy_selection_end = "2026-06-14T00:00:00Z"',
        'policy_selection_end = "2026-06-15T00:00:00Z"',
    )
    path = tmp_path / "package" / "configs" / "core.toml"
    path.parent.mkdir(parents=True)
    path.write_text(source)

    with pytest.raises(ValueError, match="chronological|contiguous"):
        load_core_config(path)


def test_core_config_rejects_partial_or_overlapping_independent_holdout(
    tmp_path: Path,
) -> None:
    source = repository_config().read_text().replace(
        'holdout_end = "2026-06-21T00:00:00Z"',
        'holdout_end = "2026-06-21T00:00:00Z"\n'
        'independent_holdout_start = "2026-06-22T00:00:00Z"',
    )
    path = tmp_path / "partial" / "configs" / "core.toml"
    path.parent.mkdir(parents=True)
    path.write_text(source)

    with pytest.raises(ValueError, match="both be configured"):
        load_core_config(path)

    overlapping = source.replace(
        'independent_holdout_start = "2026-06-22T00:00:00Z"',
        'independent_holdout_start = "2026-06-20T00:00:00Z"\n'
        'independent_holdout_end = "2026-06-27T00:00:00Z"',
    )
    path.write_text(overlapping)

    with pytest.raises(ValueError, match="must not overlap"):
        load_core_config(path)
