from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_training import ProbabilityCalibrator
from btc_directional_model.persistence_benchmark import (
    CalibratorSet,
    _add_training_gates,
    attach_execution_evidence,
    calibrated_target_probability,
    common_selected_execution_comparison,
    path_is_directionally_eligible,
    persistence_target_labels,
    target_probability_to_up,
)
from btc_directional_model.persistence_config import (
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
