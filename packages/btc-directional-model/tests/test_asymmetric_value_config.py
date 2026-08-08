from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from btc_directional_model.asymmetric_value_config import (
    HYBRID_DECISION_QUALITY_TRAINING_CONTRACT,
    TARGET_CALIBRATED_TRAINING_CONTRACT,
    load_asymmetric_value_config,
    validate_asymmetric_value_config,
)
from btc_directional_model.core_config import load_core_config


def _config_path() -> Path:
    return (
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-one-second-20260414-20260802.toml"
    )


def _target_calibrated_config_path() -> Path:
    return (
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-calibrated-20260414-20260802.toml"
    )


def _decision_quality_config_path() -> Path:
    return (
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-decision-quality-20260414-20260802.toml"
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
        validate_asymmetric_value_config(replace(config, policies=(invalid, *config.policies[1:])))


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
        validate_asymmetric_value_config(replace(config, calibration_identity_l2=0.0))
    with pytest.raises(ValueError, match="day support exceeds"):
        validate_asymmetric_value_config(
            replace(
                config,
                gates=replace(config.gates, minimum_calibration_days_per_cell=8),
            )
        )


def test_target_calibrated_contract_freezes_windows_policy_and_cells() -> None:
    config = load_asymmetric_value_config(_target_calibrated_config_path())
    core = load_core_config(config.core_config)
    primary = next(policy for policy in config.policies if policy.selection_eligible)

    assert config.training_contract == TARGET_CALIBRATED_TRAINING_CONTRACT
    assert config.fit.start.isoformat() == "2026-04-14T00:00:00+00:00"
    assert config.fit.end.isoformat() == "2026-07-16T00:00:00+00:00"
    assert config.calibration.end.isoformat() == "2026-07-23T00:00:00+00:00"
    assert config.policy.end.isoformat() == "2026-08-02T00:00:00+00:00"
    assert config.evaluation is None
    assert config.target_calibration is not None
    assert config.target_calibration.required_fitted_cells == 8
    assert config.target_calibration.time_bands == config.calibration_bands[:4]
    assert config.target_calibration.sides == ("YES", "NO")
    assert primary.maximum_entry_second == 55
    assert primary.minimum_share_price == 0.20
    assert primary.maximum_share_price == 0.30
    assert primary.maximum_cost_per_share == 0.35
    assert primary.minimum_edge_per_share == 0.03
    assert config.maximum_depth_participation == 0.25
    assert config.quantity == 5.0
    assert config.gates.maximum_selected_calibration_bias == 0.03
    assert core.split.holdout_start == core.split.holdout_end
    assert core.split.holdout_start == config.policy.end
    assert core.data.source_contract == "btc_core_v1"
    assert core.paths.source_data != config.oracle_source
    assert core.paths.source_data.name == "core-market-source"
    assert core.paths.development_feature_data.name == "core-base-development.parquet"
    assert config.oracle_source.name == "core-oracle-source"
    assert "btc-asymmetric-value-calibrated" in str(config.oracle_source)


def test_target_calibrated_contract_rejects_target_cell_fallback_weakening() -> None:
    config = load_asymmetric_value_config(_target_calibrated_config_path())

    with pytest.raises(ValueError, match="at least 50 markets"):
        validate_asymmetric_value_config(
            replace(
                config,
                gates=replace(config.gates, minimum_calibration_markets_per_cell=49),
            )
        )
    with pytest.raises(ValueError, match="fresh forward evaluation"):
        validate_asymmetric_value_config(replace(config, evaluation=config.policy))


def test_decision_quality_contract_freezes_matrix_folds_and_final_chronology() -> None:
    config = load_asymmetric_value_config(_decision_quality_config_path())
    contract = config.decision_quality

    assert config.training_contract == HYBRID_DECISION_QUALITY_TRAINING_CONTRACT
    assert contract is not None
    assert tuple(fold.name for fold in contract.folds) == (
        "jun11_jun18",
        "jun18_jun25",
        "jun25_jul02",
        "jul02_jul09",
        "jul09_jul16",
    )
    assert all(fold.fit.end == fold.calibration.start for fold in contract.folds)
    assert all(fold.calibration.end == fold.validation.start for fold in contract.folds)
    assert contract.final_fit.end == contract.final_calibration.start
    assert contract.final_fit.end.isoformat() == "2026-07-23T00:00:00+00:00"
    assert contract.final_calibration.end.isoformat() == "2026-08-02T00:00:00+00:00"
    assert len(contract.candidates) == 10
    assert sum(item.selection_eligible for item in contract.candidates) == 6
    assert {item.target_weight for item in contract.candidates if item.selection_eligible} == {
        0.25,
        0.50,
    }
    assert len(contract.calibration_variants) == 6
    assert {item.parent_source for item in contract.calibration_variants} == {
        "alltime",
        "targetpool",
    }
    assert {item.identity_l2 for item in contract.calibration_variants} == {
        0.05,
        0.20,
        1.00,
    }
    assert config.evaluation is None


def test_decision_quality_contract_rejects_matrix_or_fold_weakening() -> None:
    config = load_asymmetric_value_config(_decision_quality_config_path())
    contract = config.decision_quality
    assert contract is not None

    with pytest.raises(ValueError, match="candidate matrix changed"):
        validate_asymmetric_value_config(
            replace(
                config,
                decision_quality=replace(
                    contract,
                    candidates=contract.candidates[:-1],
                ),
            )
        )
    first = contract.folds[0]
    with pytest.raises(ValueError, match="walk-forward folds changed"):
        validate_asymmetric_value_config(
            replace(
                config,
                decision_quality=replace(
                    contract,
                    folds=(
                        replace(first, validation=first.calibration),
                        *contract.folds[1:],
                    ),
                ),
            )
        )
