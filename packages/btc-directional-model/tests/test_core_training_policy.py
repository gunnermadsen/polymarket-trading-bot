from __future__ import annotations

from pathlib import Path

from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_training import qualification_checks


def repository_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-core-20260421-20260620.toml"
    )


def qualifying_metrics() -> dict[str, float | int]:
    return {
        "accuracy": 0.75,
        "wilson_lower_95": 0.70,
        "balanced_accuracy": 0.75,
        "up_recall": 0.74,
        "down_recall": 0.76,
        "coverage": 0.60,
        "markets": 1_200,
        "expected_calibration_error": 0.03,
    }


def test_zero_same_time_path_uplift_is_noninferior() -> None:
    config = load_core_config(repository_config())
    checks = qualification_checks(
        config,
        qualifying_metrics(),
        {"accuracy_uplift": 0.0},
        {"lower_95": 0.0},
        walk_forward_accuracy=0.75,
    )

    assert all(check["passed"] for check in checks)


def test_negative_same_time_path_uplift_blocks_qualification() -> None:
    config = load_core_config(repository_config())
    checks = qualification_checks(
        config,
        qualifying_metrics(),
        {"accuracy_uplift": -0.001},
        {"lower_95": -0.001},
        walk_forward_accuracy=0.75,
    )

    failed = {check["name"] for check in checks if not check["passed"]}
    assert failed == {
        "same_cohort_accuracy_uplift",
        "hourly_bootstrap_lower_95",
    }
