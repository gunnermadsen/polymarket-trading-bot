from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from btc_directional_model import persistence_benchmark
from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_features import (
    CORE_BOUNDARY_REVERSAL_ENRICHED_FEATURES,
    CORE_ENRICHED_FEATURES,
    CORE_MATURE_REVERSAL_ENRICHED_FEATURES,
    CORE_REGIME_REVERSAL_ENRICHED_FEATURES,
)
from btc_directional_model.core_training import ProbabilityCalibrator
from btc_directional_model.persistence_benchmark import (
    CANDIDATE_PROFILES,
    CalibratorSet,
    _add_training_gates,
    _candidate_spec,
    _fit_development_finalist,
    _fit_profile_model,
    _fit_ranked_development_candidates,
    _fold_robust_agreement_probability,
    _ranked_regime_robust_candidates,
    _select_finalist,
    _validation_fold_meets_absolute_gates,
    attach_execution_evidence,
    calibrated_target_probability,
    common_selected_execution_comparison,
    configured_validation_windows,
    hard_confident_error_metrics,
    path_is_directionally_eligible,
    persistence_target_labels,
    rolling_walk_forward_fold_roles,
    target_probability_to_up,
)
from btc_directional_model.persistence_config import (
    BOUNDARY_ALIGNMENT_CANDIDATE,
    BOUNDARY_ALIGNMENT_PROFILE,
    BOUNDARY_REVERSAL_ACCURACY_CANDIDATE,
    BOUNDARY_REVERSAL_ACCURACY_PROFILE,
    FOLD_ROBUST_FREQUENCY_CANDIDATE,
    FOLD_ROBUST_FREQUENCY_PROFILE,
    MATURE_REVERSAL_ACCURACY_CANDIDATE,
    MATURE_REVERSAL_ACCURACY_PROFILE,
    REGIME_ROBUST_ACCURACY_CANDIDATES,
    REGIME_ROBUST_FEATURE_CANDIDATE,
    REGIME_ROBUST_RECENCY_CANDIDATE,
    REGIME_ROBUST_REGULARIZED_CANDIDATE,
    REGIME_ROBUST_VALIDATION_ENDS,
    REGIME_ROBUST_VALIDATION_STARTS,
    RESIDUAL_ADMISSION_SOURCE_CANDIDATES,
    RESIDUAL_ADMISSION_SOURCE_PROFILE,
    CalibrationBand,
    load_persistence_benchmark_config,
    validate_persistence_benchmark_config,
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


def regime_robust_config():
    package_root = Path(__file__).resolve().parents[1]
    return load_persistence_benchmark_config(
        package_root
        / "configs"
        / "btc-5m-directional-regime-robust-accuracy-20260321-20260729.toml"
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


def test_residual_admission_source_configuration_locks_seven_causal_folds() -> None:
    package_root = Path(__file__).resolve().parents[1]
    config = load_persistence_benchmark_config(
        package_root
        / "configs"
        / "btc-5m-directional-residual-admission-source-20260321-20260721.toml"
    )
    core = load_core_config(config.core_config)

    assert config.profile == RESIDUAL_ADMISSION_SOURCE_PROFILE
    assert config.candidate_names == RESIDUAL_ADMISSION_SOURCE_CANDIDATES
    assert core.data.range_start == datetime(2026, 3, 21, tzinfo=UTC)
    assert core.data.range_end == datetime(2026, 7, 21, tzinfo=UTC)
    assert core.split.development_end == datetime(2026, 7, 6, tzinfo=UTC)
    assert core.split.probability_calibration_end == datetime(
        2026, 7, 13, tzinfo=UTC
    )
    assert core.split.policy_selection_end == datetime(2026, 7, 21, tzinfo=UTC)
    assert len(core.split.validation_windows) == 7
    assert core.split.validation_windows[0] == (
        datetime(2026, 5, 18, tzinfo=UTC),
        datetime(2026, 5, 25, tzinfo=UTC),
    )
    assert core.split.validation_windows[-1] == (
        datetime(2026, 6, 29, tzinfo=UTC),
        datetime(2026, 7, 6, tzinfo=UTC),
    )
    assert core.gates.minimum_nonnegative_uplift_folds == 7
    assert config.quantity == 5.0
    assert config.evaluation_is_independent is False


def test_residual_admission_source_never_selects_or_freezes_a_finalist(
    tmp_path: Path,
) -> None:
    package_root = Path(__file__).resolve().parents[1]
    config = load_persistence_benchmark_config(
        package_root
        / "configs"
        / "btc-5m-directional-residual-admission-source-20260321-20260721.toml"
    )
    benchmark = {
        "benchmark_passed_candidates": [
            BOUNDARY_REVERSAL_ACCURACY_CANDIDATE
        ]
    }

    assert (
        _select_finalist(
            benchmark,
            {
                BOUNDARY_REVERSAL_ACCURACY_CANDIDATE: {
                    "out_of_fold": {"accuracy": 1.0}
                }
            },
            config,
        )
        is None
    )
    core = load_core_config(config.core_config)
    assert _fit_development_finalist(None, config, core, tmp_path) is None
    with pytest.raises(
        RuntimeError,
        match="source profile cannot create a finalist",
    ):
        _fit_development_finalist(
            BOUNDARY_REVERSAL_ACCURACY_CANDIDATE,
            config,
            core,
            tmp_path,
        )


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


def test_mature_reversal_accuracy_configuration_locks_model_accuracy_contract() -> None:
    package_root = Path(__file__).resolve().parents[1]
    config = load_persistence_benchmark_config(
        package_root
        / "configs"
        / "btc-5m-directional-mature-reversal-accuracy-20260321-20260729.toml"
    )
    core = load_core_config(config.core_config)
    profile = CANDIDATE_PROFILES[MATURE_REVERSAL_ACCURACY_CANDIDATE]
    spec = _candidate_spec(profile, config)

    assert config.profile == MATURE_REVERSAL_ACCURACY_PROFILE
    assert config.candidate_names == (
        "histogram_enriched",
        MATURE_REVERSAL_ACCURACY_CANDIDATE,
    )
    assert profile.target_kind == "outcome_up"
    assert profile.feature_kind == "core_mature_reversal"
    assert profile.calibration_kind == "global_platt"
    assert spec.feature_names == tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)
    assert len(spec.feature_names) == 71
    assert core.data.range_start == datetime(2026, 3, 21, tzinfo=UTC)
    assert core.data.range_end == datetime(2026, 7, 29, tzinfo=UTC)
    assert core.split.development_end == datetime(2026, 7, 14, tzinfo=UTC)
    assert core.split.probability_calibration_end == datetime(
        2026, 7, 21, tzinfo=UTC
    )
    assert core.split.policy_selection_end == datetime(2026, 7, 29, tzinfo=UTC)
    assert config.minimum_accuracy_uplift == 0.001
    assert config.minimum_balanced_accuracy_uplift == 0.001
    assert config.minimum_direction_recall_uplift == 0.0
    assert config.minimum_wilson_lower_uplift == 0.001
    assert config.minimum_coverage_uplift == 0.0
    assert config.minimum_hard_confident_error_count_reduction == 1


def test_regime_robust_configuration_locks_seven_validation_windows() -> None:
    config = regime_robust_config()
    core = load_core_config(config.core_config)

    validate_persistence_benchmark_config(config)
    windows = configured_validation_windows(config, core)

    assert len(windows) == 7
    assert tuple(start.isoformat() for start, _ in windows) == (
        REGIME_ROBUST_VALIDATION_STARTS
    )
    assert tuple(end.isoformat() for _, end in windows) == (
        REGIME_ROBUST_VALIDATION_ENDS
    )
    assert windows[0] == (
        datetime(2026, 6, 9, tzinfo=UTC),
        datetime(2026, 6, 16, tzinfo=UTC),
    )
    assert windows[-1] == (
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 7, 29, tzinfo=UTC),
    )


def test_regime_robust_configuration_locks_exact_ablation_matrix() -> None:
    config = regime_robust_config()

    validate_persistence_benchmark_config(config)
    assert config.candidate_names == REGIME_ROBUST_ACCURACY_CANDIDATES
    with pytest.raises(ValueError, match="frozen five-candidate"):
        validate_persistence_benchmark_config(
            replace(
                config,
                candidate_names=config.candidate_names[:-1],
            )
        )


def test_regime_robust_candidate_specs_isolate_model_training_changes() -> None:
    config = regime_robust_config()
    specs = {
        name: _candidate_spec(CANDIDATE_PROFILES[name], config)
        for name in config.candidate_names
    }

    assert specs["histogram_enriched"].feature_names == tuple(
        CORE_ENRICHED_FEATURES
    )
    assert specs[MATURE_REVERSAL_ACCURACY_CANDIDATE].feature_names == tuple(
        CORE_MATURE_REVERSAL_ENRICHED_FEATURES
    )
    assert specs[REGIME_ROBUST_RECENCY_CANDIDATE].feature_names == tuple(
        CORE_MATURE_REVERSAL_ENRICHED_FEATURES
    )
    assert specs[REGIME_ROBUST_FEATURE_CANDIDATE].feature_names == tuple(
        CORE_REGIME_REVERSAL_ENRICHED_FEATURES
    )
    assert specs[REGIME_ROBUST_REGULARIZED_CANDIDATE].feature_names == tuple(
        CORE_MATURE_REVERSAL_ENRICHED_FEATURES
    )
    assert [len(specs[name].feature_names) for name in config.candidate_names] == [
        58,
        71,
        71,
        77,
        71,
    ]
    assert specs[REGIME_ROBUST_RECENCY_CANDIDATE].recency_half_life_days == 28.0
    assert all(
        spec.recency_half_life_days is None
        for name, spec in specs.items()
        if name != REGIME_ROBUST_RECENCY_CANDIDATE
    )
    regularized = CANDIDATE_PROFILES[
        REGIME_ROBUST_REGULARIZED_CANDIDATE
    ].fixed_histogram_parameters
    assert regularized is not None
    assert regularized.min_samples_leaf == 370
    assert regularized.l2_regularization == 2.0
    for name in config.candidate_names:
        profile = CANDIDATE_PROFILES[name]
        assert profile.target_kind == "outcome_up"
        assert profile.calibration_kind == "global_platt"


def test_regime_robust_fold_roles_are_exact_disjoint_ranges() -> None:
    config = regime_robust_config()
    core = load_core_config(config.core_config)
    first = rolling_walk_forward_fold_roles(config, core, 0)
    last = rolling_walk_forward_fold_roles(config, core, 6)

    assert first.fit_start == datetime(2026, 3, 21, tzinfo=UTC)
    assert first.fit_end_exclusive == datetime(2026, 5, 26, tzinfo=UTC)
    assert first.calibration_start == datetime(2026, 5, 26, tzinfo=UTC)
    assert first.calibration_end_exclusive == datetime(2026, 6, 2, tzinfo=UTC)
    assert first.policy_start == datetime(2026, 6, 2, tzinfo=UTC)
    assert first.policy_end_exclusive == datetime(2026, 6, 9, tzinfo=UTC)
    assert first.validation_start == datetime(2026, 6, 9, tzinfo=UTC)
    assert first.validation_end_exclusive == datetime(2026, 6, 16, tzinfo=UTC)
    assert last.fit_end_exclusive == datetime(2026, 7, 7, tzinfo=UTC)
    assert last.calibration_start == datetime(2026, 7, 7, tzinfo=UTC)
    assert last.calibration_end_exclusive == datetime(2026, 7, 14, tzinfo=UTC)
    assert last.policy_start == datetime(2026, 7, 14, tzinfo=UTC)
    assert last.policy_end_exclusive == datetime(2026, 7, 21, tzinfo=UTC)
    assert last.validation_start == datetime(2026, 7, 21, tzinfo=UTC)
    assert last.validation_end_exclusive == datetime(2026, 7, 29, tzinfo=UTC)

    for fold_index in range(7):
        roles = rolling_walk_forward_fold_roles(config, core, fold_index)
        assert roles.fit_end_exclusive == roles.calibration_start
        assert roles.calibration_end_exclusive == roles.policy_start
        assert roles.policy_end_exclusive == roles.validation_start
        assert roles.fit_start < roles.fit_end_exclusive
        assert roles.calibration_start < roles.calibration_end_exclusive
        assert roles.policy_start < roles.policy_end_exclusive
        assert roles.validation_start < roles.validation_end_exclusive


def test_old_profiles_keep_core_proportional_validation_windows() -> None:
    package_root = Path(__file__).resolve().parents[1]
    config = load_persistence_benchmark_config(
        package_root
        / "configs"
        / "btc-5m-directional-mature-reversal-accuracy-20260321-20260729.toml"
    )
    core = load_core_config(config.core_config)

    assert configured_validation_windows(config, core) is core.split.validation_windows
    assert config.walk_forward_validation_starts == ()
    assert config.walk_forward_validation_ends == ()
    assert config.rolling_calibration_days is None
    assert config.rolling_policy_days is None


def test_mature_reversal_accuracy_gates_isolate_decision_quality() -> None:
    package_root = Path(__file__).resolve().parents[1]
    config = load_persistence_benchmark_config(
        package_root
        / "configs"
        / "btc-5m-directional-mature-reversal-accuracy-20260321-20260729.toml"
    )
    core = load_core_config(config.core_config)

    def evaluate(
        *,
        accuracy: float = 0.902,
        balanced_accuracy: float = 0.902,
        up_recall: float = 0.901,
        down_recall: float = 0.901,
        coverage: float = 0.60,
    ) -> tuple[dict[str, bool], dict[str, object]]:
        benchmark = {
            "candidates": {
                "histogram_enriched": {"advance": {"checks": []}},
                MATURE_REVERSAL_ACCURACY_CANDIDATE: {
                    "advance": {
                        "checks": [
                            {
                                "name": "runtime deployment contract is compatible",
                                "passed": False,
                            }
                        ]
                    }
                },
            },
            "common_comparisons": {
                MATURE_REVERSAL_ACCURACY_CANDIDATE: {
                    "checkpoints": [
                        {
                            "seconds_elapsed": 60,
                            "accuracy_delta": -0.25,
                            "balanced_accuracy_delta": -0.25,
                            "up_recall_delta": -0.25,
                            "down_recall_delta": -0.25,
                        }
                    ]
                }
            },
            "common_selected_execution_comparisons": {
                MATURE_REVERSAL_ACCURACY_CANDIDATE: {"common_markets": 0}
            },
        }
        candidate_results = {
            "histogram_enriched": {
                "out_of_fold": {
                    "accuracy": 0.900,
                    "balanced_accuracy": 0.900,
                    "up_recall": 0.900,
                    "down_recall": 0.900,
                    "wilson_lower_95": 0.880,
                    "coverage": 0.60,
                },
                "hard_confident_errors": {
                    "eligible_markets": 1_000,
                    "hard_confident_error_markets": 2,
                    "hard_confident_error_rate_selected": 2 / 800,
                },
            },
            MATURE_REVERSAL_ACCURACY_CANDIDATE: {
                "out_of_fold": {
                    "markets": 600,
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced_accuracy,
                    "up_recall": up_recall,
                    "down_recall": down_recall,
                    "wilson_lower_95": 0.882,
                    "expected_calibration_error": 0.02,
                    "coverage": coverage,
                },
                "hard_confident_errors": {
                    "eligible_markets": 1_000,
                    "hard_confident_error_markets": 1,
                    "hard_confident_error_rate_selected": 1 / 600,
                },
                "qualified_threshold_folds": 5,
                "total_folds": 5,
            },
        }

        _add_training_gates(
            benchmark,
            candidate_results,
            config,
            core,
        )

        advance = benchmark["candidates"][
            MATURE_REVERSAL_ACCURACY_CANDIDATE
        ]["advance"]
        return (
            {
                check["name"]: check["passed"]
                for check in advance["checks"]
            },
            advance,
        )

    passing, passing_advance = evaluate()
    weak_accuracy, _ = evaluate(accuracy=0.9005)
    tied_down_recall, _ = evaluate(down_recall=0.900)
    lower_coverage, _ = evaluate(coverage=0.599)

    assert all(passing.values())
    assert passing_advance["benchmark_passed"] is True
    assert passing_advance["diagnostic_only"] == [
        "early",
        "timing",
        "fixed_checkpoints",
        "path_behavior",
        "paired_uplift",
        "bootstrap_uplift",
        "execution_economics",
    ]
    assert weak_accuracy["minimum aggregate accuracy uplift"] is False
    assert tied_down_recall["positive aggregate DOWN recall uplift"] is False
    assert lower_coverage["eligible-market coverage does not regress"] is False


def test_regime_robust_accuracy_gates_require_every_configured_fold() -> None:
    config = replace(
        regime_robust_config(),
        candidate_names=(
            "histogram_enriched",
            BOUNDARY_REVERSAL_ACCURACY_CANDIDATE,
        ),
    )
    core = load_core_config(config.core_config)
    candidate_name = BOUNDARY_REVERSAL_ACCURACY_CANDIDATE

    def evaluate(qualified_validation_folds: int) -> dict[str, bool]:
        benchmark = {
            "candidates": {
                "histogram_enriched": {"advance": {"checks": []}},
                candidate_name: {"advance": {"checks": []}},
            }
        }
        candidate_results = {
            "histogram_enriched": {
                "out_of_fold": {
                    "accuracy": 0.900,
                    "balanced_accuracy": 0.900,
                    "up_recall": 0.900,
                    "down_recall": 0.900,
                    "wilson_lower_95": 0.880,
                    "coverage": 0.60,
                },
                "hard_confident_errors": {
                    "eligible_markets": 1_000,
                    "hard_confident_error_markets": 2,
                    "hard_confident_error_rate_selected": 2 / 800,
                },
            },
            candidate_name: {
                "out_of_fold": {
                    "markets": 600,
                    "accuracy": 0.902,
                    "balanced_accuracy": 0.902,
                    "up_recall": 0.901,
                    "down_recall": 0.901,
                    "wilson_lower_95": 0.882,
                    "expected_calibration_error": 0.02,
                    "coverage": 0.60,
                },
                "hard_confident_errors": {
                    "eligible_markets": 1_000,
                    "hard_confident_error_markets": 1,
                    "hard_confident_error_rate_selected": 1 / 600,
                },
                "qualified_threshold_folds": 7,
                "qualified_validation_folds": qualified_validation_folds,
                "total_folds": 7,
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
            for check in benchmark["candidates"][candidate_name]["advance"]["checks"]
        }

    passing = evaluate(7)
    one_bad_fold = evaluate(6)

    assert all(passing.values())
    assert passing["frozen chronological fold count"] is True
    assert passing["absolute accuracy standards in every validation fold"] is True
    assert one_bad_fold["absolute accuracy standards in every validation fold"] is False


def test_validation_fold_qualification_uses_all_absolute_accuracy_standards() -> None:
    config = regime_robust_config()
    core = load_core_config(config.core_config)
    passing = {
        "accuracy": core.gates.target_accuracy,
        "balanced_accuracy": core.gates.target_balanced_accuracy,
        "up_recall": core.gates.minimum_direction_recall,
        "down_recall": core.gates.minimum_direction_recall,
        "wilson_lower_95": core.gates.target_wilson_lower,
        "expected_calibration_error": core.gates.maximum_ece,
        "coverage": core.gates.minimum_coverage,
    }

    assert _validation_fold_meets_absolute_gates(passing, core) is True
    for metric, failing_value in (
        ("accuracy", core.gates.target_accuracy - 0.001),
        ("balanced_accuracy", core.gates.target_balanced_accuracy - 0.001),
        ("up_recall", core.gates.minimum_direction_recall - 0.001),
        ("down_recall", core.gates.minimum_direction_recall - 0.001),
        ("wilson_lower_95", core.gates.target_wilson_lower - 0.001),
        ("expected_calibration_error", core.gates.maximum_ece + 0.001),
        ("coverage", core.gates.minimum_coverage - 0.001),
    ):
        assert (
            _validation_fold_meets_absolute_gates(
                {**passing, metric: failing_value},
                core,
            )
            is False
        )


def test_market_regularized_candidate_uses_exact_fixed_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = regime_robust_config()
    core = load_core_config(config.core_config)
    profile = CANDIDATE_PROFILES[REGIME_ROBUST_REGULARIZED_CANDIDATE]
    spec = _candidate_spec(profile, config)
    captured: dict[str, object] = {}
    fitted = SimpleNamespace(estimator=object())

    def fake_fit_model(frame, observed_spec, parameters, observed_core):
        captured.update(
            {
                "frame": frame,
                "spec": observed_spec,
                "parameters": parameters,
                "core": observed_core,
            }
        )
        return fitted

    monkeypatch.setattr(persistence_benchmark, "fit_model", fake_fit_model)
    monkeypatch.setattr(
        persistence_benchmark,
        "estimator_converged",
        lambda estimator: estimator is fitted.estimator,
    )
    frame = pl.DataFrame({"label_up": [0, 1]})

    model, tuning = _fit_profile_model(frame, profile, spec, core)

    assert model is fitted
    assert captured["parameters"] == {
        "learning_rate": 0.05,
        "max_iter": 160,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 370,
        "l2_regularization": 2.0,
    }
    assert tuning["selected_hyperparameters"] == captured["parameters"]
    assert tuning["hyperparameter_search_consumed"] is False


def accuracy_rank_result(
    *,
    exposure: float,
    selected_rate: float,
    accuracy: float,
    median: float = 150.0,
) -> dict[str, object]:
    return {
        "hard_confident_errors": {
            "hard_confident_error_exposure_rate": exposure,
            "hard_confident_error_rate_selected": selected_rate,
        },
        "out_of_fold": {
            "accuracy": accuracy,
            "balanced_accuracy": accuracy,
            "wilson_lower_95": accuracy - 0.01,
            "coverage": 0.60,
        },
        "timing": {"median_first_crossing_seconds": median},
    }


def test_regime_robust_ranking_includes_only_oof_qualified_candidates() -> None:
    benchmark = {
        "benchmark_passed_candidates": [
            MATURE_REVERSAL_ACCURACY_CANDIDATE,
            REGIME_ROBUST_RECENCY_CANDIDATE,
            REGIME_ROBUST_FEATURE_CANDIDATE,
        ]
    }
    results = {
        MATURE_REVERSAL_ACCURACY_CANDIDATE: accuracy_rank_result(
            exposure=0.002,
            selected_rate=0.003,
            accuracy=0.91,
        ),
        REGIME_ROBUST_RECENCY_CANDIDATE: accuracy_rank_result(
            exposure=0.001,
            selected_rate=0.004,
            accuracy=0.89,
        ),
        REGIME_ROBUST_FEATURE_CANDIDATE: accuracy_rank_result(
            exposure=0.001,
            selected_rate=0.003,
            accuracy=0.90,
        ),
        REGIME_ROBUST_REGULARIZED_CANDIDATE: accuracy_rank_result(
            exposure=0.0,
            selected_rate=0.0,
            accuracy=0.99,
        ),
    }

    ranked = _ranked_regime_robust_candidates(benchmark, results)

    assert ranked == (
        REGIME_ROBUST_FEATURE_CANDIDATE,
        REGIME_ROBUST_RECENCY_CANDIDATE,
        MATURE_REVERSAL_ACCURACY_CANDIDATE,
    )
    assert REGIME_ROBUST_REGULARIZED_CANDIDATE not in ranked


def test_ranked_development_fit_tries_until_first_policy_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = regime_robust_config()
    core = load_core_config(config.core_config)
    ranked = (
        REGIME_ROBUST_FEATURE_CANDIDATE,
        REGIME_ROBUST_RECENCY_CANDIDATE,
        REGIME_ROBUST_REGULARIZED_CANDIDATE,
    )
    called: list[str] = []

    def fake_fit(candidate, observed_config, observed_core, run_dir):
        called.append(candidate)
        passed = candidate == REGIME_ROBUST_RECENCY_CANDIDATE
        return {
            "status": (
                "development_candidate_frozen"
                if passed
                else "blocked_policy_selection"
            ),
            "candidate": candidate,
            "policy_passed": passed,
            "bundle_created": passed,
        }

    monkeypatch.setattr(
        persistence_benchmark,
        "_fit_development_finalist",
        fake_fit,
    )

    finalist, bundle, attempts = _fit_ranked_development_candidates(
        ranked,
        config,
        core,
        tmp_path,
    )

    assert finalist == REGIME_ROBUST_RECENCY_CANDIDATE
    assert bundle is not None and bundle["bundle_created"] is True
    assert called == list(ranked[:2])
    assert [attempt["rank"] for attempt in attempts] == [1, 2]
    assert [attempt["bundle_created"] for attempt in attempts] == [False, True]


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
