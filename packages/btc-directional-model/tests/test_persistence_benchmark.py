from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_features import (
    CORE_BOUNDARY_REVERSAL_ENRICHED_FEATURES,
)
from btc_directional_model.core_training import ProbabilityCalibrator
from btc_directional_model.persistence_benchmark import (
    CANDIDATE_PROFILES,
    CalibratorSet,
    _add_training_gates,
    _candidate_spec,
    _fold_robust_agreement_probability,
    attach_execution_evidence,
    calibrated_target_probability,
    common_selected_execution_comparison,
    hard_confident_error_metrics,
    path_is_directionally_eligible,
    persistence_target_labels,
    target_probability_to_up,
)
from btc_directional_model.persistence_config import (
    BOUNDARY_ALIGNMENT_CANDIDATE,
    BOUNDARY_ALIGNMENT_PROFILE,
    BOUNDARY_REVERSAL_ACCURACY_CANDIDATE,
    BOUNDARY_REVERSAL_ACCURACY_PROFILE,
    FOLD_ROBUST_FREQUENCY_CANDIDATE,
    FOLD_ROBUST_FREQUENCY_PROFILE,
    CalibrationBand,
    load_persistence_benchmark_config,
)


def probability_frame() -> pl.DataFrame:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "market_id": ["a", "b", "c", "d"],
            "observed_at": [
                start + timedelta(seconds=value) for value in (60, 90, 120, 180)
            ],
            "seconds_elapsed": [60, 90, 120, 180],
            "label_up": [1, 0, 0, 1],
            "binance_sign_up": [1, 0, 1, 0],
            "btc_path_from_window_open_bps": [2.0, -2.0, 3.0, -3.0],
        }
    )


def test_persistence_target_and_outcome_probability_truth_table() -> None:
    frame = probability_frame()

    assert persistence_target_labels(frame).tolist() == [1, 1, 0, 0]
    converted = target_probability_to_up(
        frame,
        np.array([0.90, 0.90, 0.10, 0.10]),
        "path_persistence",
    )

    assert np.allclose(converted, [0.90, 0.10, 0.10, 0.90])


def test_outcome_target_probability_is_unchanged() -> None:
    frame = probability_frame()
    values = np.array([0.2, 0.4, 0.6, 0.8])

    assert np.array_equal(
        target_probability_to_up(frame, values, "outcome_up"),
        values,
    )


def test_zero_path_is_ineligible_and_routes_to_no_trade() -> None:
    frame = probability_frame().with_columns(
        pl.Series(
            "btc_path_from_window_open_bps",
            [0.0, 1e-13, -1e-13, 1e-6],
        )
    )

    assert path_is_directionally_eligible(frame).tolist() == [
        False,
        False,
        False,
        True,
    ]


class StaticLogitModel:
    def raw_logit(self, frame: pl.DataFrame) -> np.ndarray:
        return np.ones(frame.height, dtype=np.float64)


def test_time_banded_calibration_uses_exact_nonoverlapping_boundaries() -> None:
    frame = probability_frame()
    bands = (
        CalibrationBand("60-89", 60, 90),
        CalibrationBand("90-119", 90, 120),
        CalibrationBand("120-179", 120, 180),
        CalibrationBand("180-240", 180, 241),
    )
    calibrators = CalibratorSet(
        "time_banded_platt",
        {
            band.name: ProbabilityCalibrator(
                slope=1.0,
                intercept=float(index),
                converged=True,
                iterations=1,
            )
            for index, band in enumerate(bands)
        },
        bands,
    )

    probability = calibrated_target_probability(
        StaticLogitModel(),
        calibrators,
        frame,
    )

    assert np.allclose(
        probability,
        1.0 / (1.0 + np.exp(-np.array([1.0, 2.0, 3.0, 4.0]))),
    )


def test_frozen_persistence_configuration_preserves_holdout_and_gates() -> None:
    package_root = Path(__file__).resolve().parents[1]
    config = load_persistence_benchmark_config(
        package_root
        / "configs"
        / "btc-5m-directional-path-persistence-20260421-20260720.toml"
    )

    assert config.evaluation_is_independent is False
    assert config.minimum_early_markets == 500
    assert config.maximum_median_entry_seconds_regression == -5.0
    assert config.calibration_bands[-1].end_second_exclusive == 241


def test_fold_robust_frequency_configuration_is_training_only() -> None:
    package_root = Path(__file__).resolve().parents[1]
    config = load_persistence_benchmark_config(
        package_root
        / "configs"
        / "btc-5m-directional-fold-robust-frequency-20260421-20260720.toml"
    )

    assert config.profile == FOLD_ROBUST_FREQUENCY_PROFILE
    assert config.candidate_names == (
        "histogram_enriched",
        FOLD_ROBUST_FREQUENCY_CANDIDATE,
    )
    assert config.evaluation_is_independent is False


def test_boundary_alignment_configuration_preserves_control_and_gates() -> None:
    package_root = Path(__file__).resolve().parents[1]
    config = load_persistence_benchmark_config(
        package_root
        / "configs"
        / "btc-5m-directional-boundary-alignment-20260421-20260720.toml"
    )

    assert config.profile == BOUNDARY_ALIGNMENT_PROFILE
    assert config.candidate_names == (
        "histogram_enriched",
        BOUNDARY_ALIGNMENT_CANDIDATE,
    )
    assert config.maximum_median_entry_seconds_regression == -5.0
    assert config.minimum_executable_markets == 500
    assert config.evaluation_is_independent is False


def test_boundary_reversal_accuracy_configuration_locks_march_july_contract() -> None:
    package_root = Path(__file__).resolve().parents[1]
    config = load_persistence_benchmark_config(
        package_root
        / "configs"
        / "btc-5m-directional-boundary-reversal-accuracy-20260321-20260729.toml"
    )
    core = load_core_config(config.core_config)

    assert config.profile == BOUNDARY_REVERSAL_ACCURACY_PROFILE
    assert config.candidate_names == (
        "histogram_enriched",
        BOUNDARY_REVERSAL_ACCURACY_CANDIDATE,
    )
    assert core.data.range_start == datetime(2026, 3, 21, tzinfo=UTC)
    assert core.data.range_end == datetime(2026, 7, 29, tzinfo=UTC)
    assert core.split.development_end == datetime(2026, 7, 14, tzinfo=UTC)
    assert core.split.probability_calibration_end == datetime(
        2026, 7, 21, tzinfo=UTC
    )
    assert core.split.policy_selection_end == datetime(2026, 7, 29, tzinfo=UTC)
    assert core.data.strict_final_price_audit is False
    assert config.evaluation_is_independent is False
    assert config.hard_confidence_floor == 0.95
    assert config.minimum_hard_confident_error_count_reduction == 1
    assert config.maximum_hard_confident_error_selected_rate_regression == 0.0


def test_boundary_reversal_candidate_uses_full_versioned_feature_contract() -> None:
    package_root = Path(__file__).resolve().parents[1]
    config = load_persistence_benchmark_config(
        package_root
        / "configs"
        / "btc-5m-directional-boundary-reversal-accuracy-20260321-20260729.toml"
    )
    profile = CANDIDATE_PROFILES[BOUNDARY_REVERSAL_ACCURACY_CANDIDATE]

    spec = _candidate_spec(profile, config)

    assert spec.feature_names == tuple(CORE_BOUNDARY_REVERSAL_ENRICHED_FEATURES)
    assert len(spec.feature_names) == 106
    assert "btc_path_from_window_open_bps" in spec.feature_names
    assert "btc_cross_venue_boundary_gap_bps" in spec.feature_names
    assert "btc_path_sign_normalized_boundary_gap_bps" in spec.feature_names


def test_hard_confident_errors_use_fixed_and_selected_denominators() -> None:
    rows = pl.DataFrame(
        {
            "market_id": ["hard-wrong", "soft-wrong", "hard-correct"],
            "correct": [False, False, True],
            "confidence": [0.96, 0.94, 0.99],
        }
    )

    metrics = hard_confident_error_metrics(
        rows,
        eligible_markets=5,
        confidence_floor=0.95,
    )

    assert metrics["hard_confident_error_markets"] == 1
    assert metrics["hard_confident_error_exposure_rate"] == 0.2
    assert metrics["hard_confident_error_rate_selected"] == 1 / 3
    assert metrics["maximum_incorrect_confidence"] == 0.96


def test_boundary_reversal_tail_gate_cannot_be_diluted_by_more_trades() -> None:
    package_root = Path(__file__).resolve().parents[1]
    config = load_persistence_benchmark_config(
        package_root
        / "configs"
        / "btc-5m-directional-boundary-reversal-accuracy-20260321-20260729.toml"
    )
    core = load_core_config(config.core_config)

    def evaluate(
        candidate_hard_errors: int,
        candidate_selected_markets: int = 100,
    ) -> dict[str, bool]:
        checkpoints = [
            {
                "seconds_elapsed": second,
                "accuracy_delta": 0.0,
                "balanced_accuracy_delta": 0.0,
                "up_recall_delta": 0.0,
                "down_recall_delta": 0.0,
            }
            for second in (60, 90, 120, 180, 240)
        ]
        benchmark = {
            "candidates": {
                "histogram_enriched": {"advance": {"checks": []}},
                BOUNDARY_REVERSAL_ACCURACY_CANDIDATE: {
                    "advance": {"checks": []}
                },
            },
            "common_comparisons": {
                BOUNDARY_REVERSAL_ACCURACY_CANDIDATE: {
                    "checkpoints": checkpoints
                }
            },
            "common_selected_execution_comparisons": {
                BOUNDARY_REVERSAL_ACCURACY_CANDIDATE: {"common_markets": 500}
            },
        }
        candidate_results = {
            "histogram_enriched": {
                "hard_confident_errors": {
                    "eligible_markets": 100,
                    "selected_markets": 80,
                    "hard_confident_error_markets": 2,
                    "hard_confident_error_rate_selected": 2 / 80,
                }
            },
            BOUNDARY_REVERSAL_ACCURACY_CANDIDATE: {
                "hard_confident_errors": {
                    "eligible_markets": 100,
                    "selected_markets": candidate_selected_markets,
                    "hard_confident_error_markets": candidate_hard_errors,
                    "hard_confident_error_rate_selected": (
                        candidate_hard_errors / candidate_selected_markets
                    ),
                },
                "early": {
                    "markets": 500,
                    "accuracy": 0.90,
                    "balanced_accuracy": 0.90,
                    "up_recall": 0.90,
                    "down_recall": 0.90,
                    "wilson_lower_95": 0.88,
                },
                "qualified_threshold_folds": 5,
                "total_folds": 5,
                "nonnegative_uplift_folds": 5,
                "bootstrap": {"lower_95": 0.0},
                "passed_development": True,
            },
        }

        _add_training_gates(
            benchmark,
            candidate_results,
            config,
            core,
        )

        return {
            check["name"]: check["passed"]
            for check in benchmark["candidates"][
                BOUNDARY_REVERSAL_ACCURACY_CANDIDATE
            ]["advance"]["checks"]
        }

    diluted = evaluate(candidate_hard_errors=2)
    improved = evaluate(candidate_hard_errors=1)
    concentrated = evaluate(
        candidate_hard_errors=1,
        candidate_selected_markets=20,
    )

    assert diluted["minimum hard-confident error count reduction"] is False
    assert (
        diluted["hard-confident error rate per selected trade does not regress"]
        is True
    )
    assert improved["minimum hard-confident error count reduction"] is True
    assert (
        improved["hard-confident error rate per selected trade does not regress"]
        is True
    )
    assert concentrated["minimum hard-confident error count reduction"] is True
    assert (
        concentrated["hard-confident error rate per selected trade does not regress"]
        is False
    )


def test_fold_robust_agreement_boost_preserves_control_direction() -> None:
    control = np.array([0.40, 0.60, 0.40, 0.60])
    auxiliary = np.array([0.10, 0.90, 0.90, 0.10])

    combined = _fold_robust_agreement_probability(control, auxiliary)

    assert np.array_equal(combined >= 0.5, control >= 0.5)
    assert combined[0] < control[0]
    assert combined[1] > control[1]
    assert combined[2] == control[2]
    assert combined[3] == control[3]


def test_execution_attachment_requires_strict_both_side_eligibility() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    predictions = pl.DataFrame(
        {
            "market_id": ["not-strict", "strict"],
            "observed_at": [start, start],
        }
    )
    evidence = pl.DataFrame(
        {
            "market_id": ["not-strict", "strict"],
            "observed_at": [start, start],
            "fee_rate": [0.02, 0.02],
            "up_ask_vwap_5": [0.6, 0.6],
            "down_ask_vwap_5": [0.4, 0.4],
            "up_side_fresh": [True, True],
            "down_side_fresh": [True, True],
            "strict_both_side_eligible": [False, True],
        }
    )

    attached = attach_execution_evidence(predictions, evidence).sort("market_id")

    assert attached["execution_evidence_available"].to_list() == [True, True]
    assert attached["up_executable"].to_list() == [False, True]
    assert attached["down_executable"].to_list() == [False, True]


def test_common_selected_execution_requires_both_policies_to_be_executable() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    common = {
        "market_id": ["a", "b"],
        "observed_at": [start, start + timedelta(minutes=5)],
        "seconds_elapsed": [90, 90],
        "predicted_up": [1, 0],
        "policy_selected": [True, True],
        "up_executable": [True, True],
        "down_executable": [True, True],
    }
    control = pl.DataFrame(common)
    candidate = pl.DataFrame(common).with_columns(
        pl.when(pl.col("market_id") == "b")
        .then(False)
        .otherwise(pl.col("down_executable"))
        .alias("down_executable")
    )

    comparison = common_selected_execution_comparison(
        control,
        candidate,
        control_name="control",
        candidate_name="candidate",
    )

    assert comparison["control_strict_executable_markets"] == 2
    assert comparison["candidate_strict_executable_markets"] == 1
    assert comparison["common_markets"] == 1
    assert comparison["exact_timestamp_markets"] == 1


def test_training_gates_fail_closed_on_checkpoint_early_fold_and_common_execution() -> None:
    package_root = Path(__file__).resolve().parents[1]
    frozen = load_persistence_benchmark_config(
        package_root
        / "configs"
        / "btc-5m-directional-path-persistence-20260421-20260720.toml"
    )
    config = replace(
        frozen,
        candidate_names=("histogram_enriched", "histogram_path_persistence"),
    )
    core = load_core_config(config.core_config)
    checkpoints = [
        {
            "seconds_elapsed": second,
            "accuracy_delta": -0.001 if second == 60 else 0.0,
            "balanced_accuracy_delta": 0.0,
            "up_recall_delta": 0.0,
            "down_recall_delta": 0.0,
        }
        for second in (60, 90, 120, 180, 240)
    ]
    benchmark = {
        "candidates": {
            "histogram_enriched": {"advance": {"checks": []}},
            "histogram_path_persistence": {"advance": {"checks": []}},
        },
        "common_comparisons": {
            "histogram_path_persistence": {"checkpoints": checkpoints}
        },
        "common_selected_execution_comparisons": {
            "histogram_path_persistence": {"common_markets": 499}
        },
    }
    result = {
        "early": {
            "markets": 500,
            "accuracy": 0.873,
            "balanced_accuracy": 0.873,
            "up_recall": 0.873,
            "down_recall": 0.873,
            "wilson_lower_95": 0.864,
        },
        "qualified_threshold_folds": 4,
        "total_folds": 5,
        "nonnegative_uplift_folds": 5,
        "bootstrap": {"lower_95": 0.0},
        "passed_development": True,
    }

    _add_training_gates(
        benchmark,
        {"histogram_path_persistence": result},
        config,
        core,
    )

    checks = {
        check["name"]: check["passed"]
        for check in benchmark["candidates"]["histogram_path_persistence"][
            "advance"
        ]["checks"]
    }
    assert checks["60s common-time accuracy does not regress"] is False
    assert checks["minimum early accuracy"] is False
    assert checks["minimum early balanced accuracy"] is False
    assert checks["minimum early UP recall"] is False
    assert checks["minimum early DOWN recall"] is False
    assert checks["minimum early Wilson lower bound"] is False
    assert checks["qualified threshold in every fold"] is False
    assert checks["minimum common selected executable markets"] is False
    assert (
        benchmark["candidates"]["histogram_path_persistence"]["advance"][
            "benchmark_passed"
        ]
        is False
    )
