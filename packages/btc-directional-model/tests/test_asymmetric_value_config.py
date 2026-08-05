from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from btc_directional_model.asymmetric_value_config import (
    load_asymmetric_value_config,
    validate_asymmetric_value_config,
)
from btc_directional_model.core_config import load_core_config


def _config_path() -> Path:
    return (
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-one-second-20260414-20260802.toml"
    )


def test_asymmetric_value_contract_is_lower_price_and_paper_only() -> None:
    config = load_asymmetric_value_config(_config_path())
    selectable = [policy for policy in config.policies if policy.selection_eligible]

    expected_grid = (*range(1, 60), *range(60, 241, 5))
    assert config.prediction_seconds == expected_grid
    assert config.price_seconds == expected_grid
    assert len(config.prediction_seconds) == 96
    assert config.calibration_bands[0] == (1, 15)
    assert config.gates.minimum_calibration_markets_per_cell == 50
    assert config.gates.minimum_calibration_days_per_cell == 5
    assert config.calibration_identity_l2 == 1.0
    assert config.maximum_depth_participation == 0.25
    assert len(selectable) == 1
    assert all(policy.maximum_share_price == 0.30 for policy in selectable)
    assert config.evaluation.start.isoformat() == "2026-07-20T00:00:00+00:00"


def test_asymmetric_value_contract_rejects_primary_expensive_policy() -> None:
    config = load_asymmetric_value_config(_config_path())
    primary = next(policy for policy in config.policies if policy.selection_eligible)
    invalid = replace(primary, maximum_share_price=0.90)

    with pytest.raises(ValueError, match="raw share-price"):
        validate_asymmetric_value_config(
            replace(config, policies=(invalid, *config.policies[1:]))
        )


def test_asymmetric_value_contract_preserves_original_frozen_windows() -> None:
    config = load_asymmetric_value_config(_config_path())

    assert config.fit.end == config.calibration.start
    assert config.calibration.end == config.policy.start
    assert config.policy.end == config.evaluation.start
    assert config.fit.end.isoformat() == "2026-07-06T00:00:00+00:00"
    assert config.calibration.end.isoformat() == "2026-07-13T00:00:00+00:00"


def test_asymmetric_value_core_builds_causal_one_second_candidates() -> None:
    config = load_asymmetric_value_config(_config_path())
    core = load_core_config(config.core_config)

    assert core.data.sample_interval_seconds == 1
    assert core.data.min_seconds_after_open == 1
    assert 300 - core.data.min_seconds_before_close == 240


def test_asymmetric_value_contract_rejects_invalid_calibration_support() -> None:
    config = load_asymmetric_value_config(_config_path())

    with pytest.raises(ValueError, match="identity L2"):
        validate_asymmetric_value_config(
            replace(config, calibration_identity_l2=0.0)
        )
    with pytest.raises(ValueError, match="day support exceeds"):
        validate_asymmetric_value_config(
            replace(
                config,
                gates=replace(config.gates, minimum_calibration_days_per_cell=8),
            )
        )
