from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from btc_directional_model import core_training
from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_features import CORE_ENRICHED_FEATURES
from btc_directional_model.core_training import (
    EARLY_ENTRY_TRAINING_WEIGHT_MULTIPLIER,
    candidate_rank,
    candidate_spec,
    candidate_training_weights,
    market_equal_weights,
    qualification_checks,
)


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


def test_early_candidate_uses_deploy_compatible_feature_schema() -> None:
    candidate = candidate_spec("histogram_early_weighted")

    assert candidate.family == "histogram"
    assert candidate.feature_names == tuple(CORE_ENRICHED_FEATURES)


def test_early_weights_emphasize_timing_with_equal_total_per_market() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "a", "a", "b", "b"],
            "seconds_elapsed": [60, 90, 120, 180, 60, 180],
        }
    )
    candidate = candidate_spec("histogram_early_weighted")

    weights = candidate_training_weights(frame, candidate)

    assert weights[0] / weights[3] == pytest.approx(
        EARLY_ENTRY_TRAINING_WEIGHT_MULTIPLIER
    )
    assert weights[1] / weights[3] == pytest.approx(
        EARLY_ENTRY_TRAINING_WEIGHT_MULTIPLIER
    )
    assert weights[2] / weights[3] == pytest.approx(
        EARLY_ENTRY_TRAINING_WEIGHT_MULTIPLIER
    )
    assert weights[4] / weights[5] == pytest.approx(
        EARLY_ENTRY_TRAINING_WEIGHT_MULTIPLIER
    )
    assert weights[:4].sum() == pytest.approx(weights[4:].sum())


def test_existing_candidates_retain_market_equal_weights() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "a", "b", "b"],
            "seconds_elapsed": [60, 90, 180, 60, 180],
        }
    )
    expected = market_equal_weights(frame)

    for name in (
        "logistic_baseline",
        "logistic_enriched",
        "histogram_enriched",
    ):
        assert np.array_equal(
            candidate_training_weights(frame, candidate_spec(name)),
            expected,
        )


def test_candidate_rank_never_weakens_accuracy_for_earlier_timing() -> None:
    later_more_accurate = ranking_result(
        passed=True,
        accuracy=0.90,
        early_coverage=0.10,
        median_seconds=180.0,
    )
    earlier_less_accurate = ranking_result(
        passed=True,
        accuracy=0.89,
        early_coverage=0.50,
        median_seconds=90.0,
    )
    earlier_blocked = ranking_result(
        passed=False,
        accuracy=0.99,
        early_coverage=0.90,
        median_seconds=60.0,
    )

    assert candidate_rank(later_more_accurate) > candidate_rank(
        earlier_less_accurate
    )
    assert candidate_rank(later_more_accurate) > candidate_rank(earlier_blocked)


def test_candidate_task_loads_feature_cache_once_for_all_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        compute=SimpleNamespace(threads_per_fit=1),
        split=SimpleNamespace(validation_windows=(("a", "b"), ("b", "c"), ("c", "d"))),
    )
    frame = pl.DataFrame({"market_id": ["market"]})
    feature_loads = 0

    monkeypatch.setattr(core_training, "load_core_config", lambda _: config)
    monkeypatch.setattr(core_training, "configure_native_thread_limits", lambda _: None)

    def load_frame(_config: object, cohort: str) -> pl.DataFrame:
        nonlocal feature_loads
        feature_loads += 1
        assert cohort == "pre_holdout"
        return frame

    monkeypatch.setattr(core_training, "load_core_feature_frame", load_frame)
    monkeypatch.setattr(
        core_training,
        "evaluate_fold",
        lambda loaded, spec, fold_index, _config: {
            "candidate": spec.name,
            "fold_index": fold_index,
            "same_frame": loaded is frame,
        },
    )

    results = core_training.evaluate_candidate_task(
        Path("unused.toml"),
        "histogram_early_weighted",
    )

    assert feature_loads == 1
    assert [result["fold_index"] for result in results] == [0, 1, 2]
    assert all(result["same_frame"] for result in results)


def ranking_result(
    *,
    passed: bool,
    accuracy: float,
    early_coverage: float,
    median_seconds: float,
) -> dict[str, object]:
    return {
        "passed_development": passed,
        "bootstrap": {"lower_95": 0.01},
        "paired": {"accuracy_uplift": 0.01},
        "out_of_fold": {
            "wilson_lower_95": accuracy - 0.02,
            "balanced_accuracy": accuracy,
            "accuracy": accuracy,
            "coverage": 0.60,
        },
        "timing": {
            "early_entry_coverage": early_coverage,
            "median_first_crossing_seconds": median_seconds,
            "p90_first_crossing_seconds": median_seconds + 30,
        },
        "feature_count": len(CORE_ENRICHED_FEATURES),
    }
