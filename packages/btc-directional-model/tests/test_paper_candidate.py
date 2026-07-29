from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from btc_directional_model.core_training import (
    FittedCoreModel,
    ProbabilityCalibrator,
)
from btc_directional_model.paper_candidate import (
    PAPER_CONFIDENCE_THRESHOLD,
    PAPER_MINIMUM_COVERAGE,
    _benchmark_check_observed,
    choose_paper_threshold,
    load_benchmark_evidence,
    persistence_bundle_to_runtime_bundle,
    production_blocking_reasons,
    validate_mature_reversal_schema,
)
from btc_directional_model.persistence_benchmark import (
    CANDIDATE_PROFILES,
    CalibratorSet,
    PersistenceTrainingBundle,
)
from btc_directional_model.persistence_config import (
    REGIME_ROBUST_RECENCY_CANDIDATE,
    load_persistence_benchmark_config,
    persistence_config_to_dict,
)


def policy_row(
    threshold: float,
    *,
    coverage: float,
    markets: int,
    accuracy: float,
) -> dict[str, float | int]:
    return {
        "threshold": threshold,
        "coverage": coverage,
        "markets": markets,
        "wilson_lower_95": accuracy - 0.01,
        "accuracy": accuracy,
        "balanced_accuracy": accuracy,
        "up_recall": accuracy,
        "down_recall": accuracy,
        "expected_calibration_error": 0.01,
    }


def test_paper_threshold_is_locked_to_087_with_coverage_floor() -> None:
    table = [
        policy_row(0.86, coverage=0.60, markets=1_200, accuracy=0.87),
        policy_row(0.87, coverage=0.56, markets=1_120, accuracy=0.89),
        policy_row(0.88, coverage=0.55, markets=1_100, accuracy=0.92),
    ]

    selected = choose_paper_threshold(
        table,
        locked_threshold=PAPER_CONFIDENCE_THRESHOLD,
        minimum_coverage=PAPER_MINIMUM_COVERAGE,
        minimum_markets=1_100,
    )

    assert selected["threshold"] == 0.87
    assert selected["accuracy"] == 0.89


def test_locked_paper_threshold_rejects_insufficient_coverage() -> None:
    with pytest.raises(RuntimeError, match="does not retain"):
        choose_paper_threshold(
            [policy_row(0.87, coverage=0.549, markets=1_500, accuracy=0.99)],
            locked_threshold=PAPER_CONFIDENCE_THRESHOLD,
            minimum_coverage=PAPER_MINIMUM_COVERAGE,
            minimum_markets=1_000,
        )


def test_direct_outcome_global_platt_bundle_converts_without_refit() -> None:
    profile = CANDIDATE_PROFILES[REGIME_ROBUST_RECENCY_CANDIDATE]
    model = FittedCoreModel(
        candidate_name=REGIME_ROBUST_RECENCY_CANDIDATE,
        family="histogram",
        feature_names=("feature",),
        hyperparameters={"max_iter": 1},
        imputation_medians=np.asarray([0.0]),
        standardization_means=None,
        standardization_scales=None,
        estimator=object(),
    )
    calibrator = ProbabilityCalibrator(1.1, -0.02, True, 4)
    persistence = PersistenceTrainingBundle(
        model=model,
        calibrators=CalibratorSet(
            kind="global_platt",
            calibrators={"global": calibrator},
            bands=(),
        ),
        profile=profile,
        confidence_threshold=PAPER_CONFIDENCE_THRESHOLD,
    )

    runtime = persistence_bundle_to_runtime_bundle(persistence)

    assert runtime.model is model
    assert runtime.calibrator is calibrator
    assert runtime.confidence_threshold == PAPER_CONFIDENCE_THRESHOLD

    with pytest.raises(RuntimeError, match="direct outcome"):
        persistence_bundle_to_runtime_bundle(
            replace(
                persistence,
                profile=replace(profile, target_kind="path_persistence"),
            )
        )


def test_real_benchmark_shape_derives_uplift_without_own_policy_paired_key() -> None:
    own_policy = {
        "accuracy": 0.8908463278181169,
        "coverage": 0.6064390384754205,
    }
    advance_checks = [
        {
            "name": "minimum aggregate accuracy uplift",
            "observed": 0.00008603840998022694,
            "passed": False,
        }
    ]

    assert "paired" not in own_policy
    assert _benchmark_check_observed(
        advance_checks,
        "minimum aggregate accuracy uplift",
    ) == pytest.approx(0.00008603840998022694)


def test_benchmark_evidence_requires_the_exact_path_configuration(
    tmp_path: Path,
) -> None:
    package_root = Path(__file__).parent.parent
    config = load_persistence_benchmark_config(
        package_root
        / "configs"
        / "btc-5m-directional-regime-robust-accuracy-20260321-20260729.toml"
    )
    benchmark_run = tmp_path / "run"
    benchmark_run.mkdir()
    configuration = json.loads(
        json.dumps(persistence_config_to_dict(config))
    )
    benchmark = {
        "configuration": configuration,
        "candidates": {REGIME_ROBUST_RECENCY_CANDIDATE: {}},
    }
    (benchmark_run / "benchmark.json").write_text(json.dumps(benchmark))

    loaded, _ = load_benchmark_evidence(benchmark_run, config)
    assert loaded["configuration"]["core_config"] == str(config.core_config)

    benchmark["configuration"]["core_config"] = "/wrong/core-config.toml"
    (benchmark_run / "benchmark.json").write_text(json.dumps(benchmark))
    with pytest.raises(RuntimeError, match="exact benchmark configuration"):
        load_benchmark_evidence(benchmark_run, config)


def test_paper_artifact_blockers_never_claim_production_qualification() -> None:
    benchmark = {
        "benchmark_passed_candidates": [],
        "deployment_qualified_candidates": [],
    }
    reasons = production_blocking_reasons(
        benchmark,
        {"passed": False},
    )

    assert "no independent post-freeze forward cohort has been evaluated" in reasons
    assert "development benchmark advancement contract did not pass" in reasons
    assert "final-fit policy selection did not pass every production gate" in reasons


def test_recency_candidate_requires_the_71_feature_schema() -> None:
    validate_mature_reversal_schema(
        {
            "candidate_feature_schema_versions": {
                REGIME_ROBUST_RECENCY_CANDIDATE: (
                    "btc-5m-directional-mature-reversal-features-v1"
                )
            }
        }
    )
    with pytest.raises(RuntimeError, match="71-feature schema"):
        validate_mature_reversal_schema(
            {
                "candidate_feature_schema_versions": {
                    REGIME_ROBUST_RECENCY_CANDIDATE: "wrong-schema"
                }
            }
        )
