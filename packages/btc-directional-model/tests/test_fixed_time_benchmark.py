from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from btc_directional_model.core_features import (
    CORE_MATURE_REVERSAL_ENRICHED_FEATURES,
)
from btc_directional_model.fixed_time_benchmark import (
    _economics_checks,
    _hard_error_nonregression_checks,
    empirical_coverage_threshold,
    estimator_training_rows,
    fixed_time_candidate_spec,
    fixed_time_rows,
    operating_point_checks,
)
from btc_directional_model.fixed_time_config import (
    load_fixed_time_accuracy_config,
)


def repository_config() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "btc-5m-directional-mature-reversal-fixed-120-20260321-20260729.toml"
    )


def scored_policy_rows() -> pl.DataFrame:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    probability = [0.99, 0.96, 0.96, 0.92, 0.84, 0.51]
    return (
        pl.DataFrame(
            {
                "market_id": [f"market-{index}" for index in range(6)],
                "window_start": [start + timedelta(minutes=5 * index) for index in range(6)],
                "observed_at": [
                    start + timedelta(minutes=5 * index, seconds=120) for index in range(6)
                ],
                "seconds_elapsed": [120] * 6,
                "label_up": [1, 0, 1, 0, 1, 0],
                "binance_sign_up": [1, 0, 0, 1, 1, 0],
                "probability_up": probability,
            }
        )
        .with_columns(
            (pl.col("probability_up") >= 0.5).cast(pl.Int8).alias("predicted_up"),
            pl.max_horizontal(
                "probability_up",
                1 - pl.col("probability_up"),
            ).alias("confidence"),
        )
        .with_columns(
            (pl.col("predicted_up") == pl.col("label_up")).alias("correct"),
            (pl.col("binance_sign_up") == pl.col("label_up")).alias("baseline_correct"),
        )
    )


def test_fixed_time_candidate_uses_only_frozen_71_core_features() -> None:
    config = load_fixed_time_accuracy_config(repository_config())

    spec = fixed_time_candidate_spec(config)

    assert spec.feature_names == tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)
    assert len(spec.feature_names) == 71
    assert spec.family == "histogram"
    assert spec.recency_half_life_days == 28.0
    assert not any(
        name.startswith("oracle_") or "vwap" in name or "missing" in name or "eligible" in name
        for name in spec.feature_names
    )


def test_fixed_time_rows_requires_exactly_one_120_row_per_market() -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "b", "b"],
            "window_start": [start, start, start, start],
            "observed_at": [
                start + timedelta(seconds=120),
                start + timedelta(seconds=125),
                start + timedelta(seconds=120),
                start + timedelta(seconds=125),
            ],
            "seconds_elapsed": [120, 125, 120, 125],
        }
    )

    selected = fixed_time_rows(
        frame,
        decision_second=120,
        cohort_name="test",
    )

    assert selected.height == 2
    assert selected["market_id"].n_unique() == 2
    assert selected["seconds_elapsed"].to_list() == [120, 120]

    duplicate = pl.concat([selected, selected[:1]])
    with pytest.raises(RuntimeError, match="exactly one row per market"):
        fixed_time_rows(
            duplicate,
            decision_second=120,
            cohort_name="duplicate",
        )


def test_estimator_training_retains_all_predecessor_context_rows() -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    seconds = (120, 125, 130, 135, 140)
    frame = pl.DataFrame(
        {
            "market_id": [market_id for market_id in ("a", "b") for _ in seconds],
            "window_start": [start] * (2 * len(seconds)),
            "observed_at": [
                start + timedelta(seconds=second) for _ in ("a", "b") for second in seconds
            ],
            "seconds_elapsed": list(seconds) * 2,
        }
    )

    selected = estimator_training_rows(
        frame,
        training_seconds=seconds,
        cohort_name="estimator context",
    )

    assert selected.height == 10
    assert selected.group_by("market_id").len()["len"].to_list() == [5, 5]
    with pytest.raises(RuntimeError, match="every estimator-training second"):
        estimator_training_rows(
            frame.filter(~((pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 140))),
            training_seconds=seconds,
            cohort_name="incomplete estimator context",
        )


def test_empirical_threshold_is_label_blind_deterministic_and_expands_ties() -> None:
    scored = scored_policy_rows()

    threshold, selected, evidence = empirical_coverage_threshold(
        scored,
        target_coverage=2 / 6,
    )
    changed_labels = scored.with_columns((1 - pl.col("label_up")).alias("label_up"))
    changed_threshold, changed_selected, changed_evidence = empirical_coverage_threshold(
        changed_labels,
        target_coverage=2 / 6,
    )

    assert threshold == 0.96
    assert selected.height == 3
    assert evidence["target_markets"] == 2
    assert evidence["tie_expansion_markets"] == 1
    assert evidence["labels_used_for_threshold_selection"] is False
    assert changed_threshold == threshold
    assert changed_selected["market_id"].to_list() == selected["market_id"].to_list()
    assert changed_evidence["labels_used_for_threshold_selection"] is False


def test_empirical_threshold_rejects_duplicate_markets_and_bad_coverage() -> None:
    scored = scored_policy_rows()
    duplicate = pl.concat([scored, scored[:1]])

    with pytest.raises(ValueError, match="one row per market"):
        empirical_coverage_threshold(
            duplicate,
            target_coverage=0.15,
        )
    with pytest.raises(ValueError, match="between zero and one"):
        empirical_coverage_threshold(
            scored,
            target_coverage=1.0,
        )


def test_operating_point_checks_apply_accuracy_recall_wilson_and_ece() -> None:
    config = load_fixed_time_accuracy_config(repository_config())
    metrics = {
        "coverage": 0.15,
        "accuracy": 0.92,
        "balanced_accuracy": 0.91,
        "up_recall": 0.905,
        "down_recall": 0.915,
        "wilson_lower_95": 0.90,
        "expected_calibration_error": 0.04,
    }

    checks = operating_point_checks(metrics, config.primary)

    assert checks
    assert all(check["passed"] for check in checks)
    failed = operating_point_checks(
        {**metrics, "down_recall": 0.89},
        config.primary,
    )
    assert not all(check["passed"] for check in failed)
    assert next(check for check in failed if check["name"] == "DOWN recall")["passed"] is False


def test_hard_error_and_ten_share_economics_checks_are_selection_gates() -> None:
    candidate_tail = {
        "hard_confident_error_exposure_rate": 0.01,
        "hard_confident_error_rate_selected": 0.05,
    }
    predecessor_tail = {
        "hard_confident_error_exposure_rate": 0.011,
        "hard_confident_error_rate_selected": 0.051,
    }

    assert all(
        check["passed"]
        for check in _hard_error_nonregression_checks(
            candidate_tail,
            predecessor_tail,
        )
    )
    regressed = {
        **candidate_tail,
        "hard_confident_error_rate_selected": 0.06,
    }
    assert not all(
        check["passed"]
        for check in _hard_error_nonregression_checks(
            regressed,
            predecessor_tail,
        )
    )

    positive = {
        "economics_available": True,
        "realized_net_expectancy_per_trade": 1.0,
        "realized_net_pnl_total": 100.0,
    }
    assert all(check["passed"] for check in _economics_checks(positive))
    assert not all(
        check["passed"]
        for check in _economics_checks(
            {
                **positive,
                "realized_net_expectancy_per_trade": -0.01,
            }
        )
    )
