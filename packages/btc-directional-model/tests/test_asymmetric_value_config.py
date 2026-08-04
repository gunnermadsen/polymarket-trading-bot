from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from btc_directional_model.asymmetric_value_config import (
    load_asymmetric_value_config,
    validate_asymmetric_value_config,
)


def _config_path() -> Path:
    return (
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-hunter-20260414-20260802.toml"
    )


def test_asymmetric_value_contract_is_lower_price_and_paper_only() -> None:
    config = load_asymmetric_value_config(_config_path())
    selectable = [policy for policy in config.policies if policy.selection_eligible]

    assert config.prediction_seconds == tuple(range(5, 241, 5))
    assert config.price_seconds == (*range(1, 60), *range(60, 241, 5))
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
