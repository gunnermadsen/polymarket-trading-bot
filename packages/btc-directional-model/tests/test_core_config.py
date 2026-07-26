from __future__ import annotations

from pathlib import Path

import pytest

from btc_directional_model.core_config import load_core_config


def repository_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-core-20260421-20260620.toml"
    )


def test_expanded_core_config_has_frozen_contiguous_cohorts() -> None:
    config = load_core_config(repository_config())

    assert config.data.source_contract == "btc_core_v1"
    assert config.data.range_start == config.split.development_start
    assert config.split.development_end == config.split.probability_calibration_start
    assert config.split.probability_calibration_end == config.split.policy_selection_start
    assert config.split.policy_selection_end == config.split.holdout_start
    assert config.split.holdout_end == config.data.range_end
    assert len(config.split.validation_windows) == 5
    assert config.gates.minimum_same_time_path_uplift == 0
    assert config.gates.minimum_nonnegative_uplift_folds == 5
    assert config.paths.development_feature_data != config.paths.holdout_feature_data


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
