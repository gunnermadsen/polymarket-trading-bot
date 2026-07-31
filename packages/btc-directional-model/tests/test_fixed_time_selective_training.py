from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from btc_directional_model import fixed_time_selective_training
from btc_directional_model.core_config import HistogramCandidate
from btc_directional_model.core_features import (
    CORE_MATURE_REVERSAL_ENRICHED_FEATURES,
    CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
    CORE_REGIME_REVERSAL_ENRICHED_FEATURES,
    CORE_REGIME_REVERSAL_FEATURE_SCHEMA_VERSION,
)
from btc_directional_model.core_training import candidate_training_weights
from btc_directional_model.fixed_time_selective_config import (
    BASE_PARAMETER_GRID,
    FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE,
    FIXED_TIME_SELECTIVE_REGIME_CANDIDATE,
    FIXED_TIME_SELECTIVE_TAIL_REGULARIZED_CANDIDATE,
    FIXED_TIME_SELECTIVE_WEIGHT_2X_CANDIDATE,
    FIXED_TIME_SELECTIVE_WEIGHT_3X_CANDIDATE,
    TAIL_PARAMETER_GRID,
    FixedTimeSelectiveCandidateConfig,
    FixedTimeSelectiveHistogramConfig,
    FixedTimeSelectiveModelConfig,
)
from btc_directional_model.fixed_time_selective_training import (
    predicted_side_hard_error_metrics,
    selective_candidate_spec,
    selective_parameter_grid,
    tune_and_fit_selective_model,
)


def selective_model_config() -> FixedTimeSelectiveModelConfig:
    return FixedTimeSelectiveModelConfig(
        decision_second=120,
        estimator_training_seconds=(120, 125, 130, 135, 140),
        target="outcome_up",
        estimator_family="histogram_gradient_boosting",
        probability_calibration="global_platt",
        recency_half_life_days=28.0,
        include_oracle=False,
        include_book=False,
        threshold_selection="empirical_policy_confidence_quantile",
        hard_confidence_floor=0.95,
        require_hard_confident_error_no_regression=True,
        tail_histogram_parameters=(
            FixedTimeSelectiveHistogramConfig(15, 200, 2.0, 0.05, 160),
            FixedTimeSelectiveHistogramConfig(15, 300, 5.0, 0.03, 220),
            FixedTimeSelectiveHistogramConfig(7, 200, 2.0, 0.05, 160),
        ),
    )


def selective_candidate_configs() -> tuple[FixedTimeSelectiveCandidateConfig, ...]:
    mature = tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)
    regime = tuple(CORE_REGIME_REVERSAL_ENRICHED_FEATURES)
    return (
        FixedTimeSelectiveCandidateConfig(
            FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE,
            CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
            mature,
            1.0,
            BASE_PARAMETER_GRID,
        ),
        FixedTimeSelectiveCandidateConfig(
            FIXED_TIME_SELECTIVE_WEIGHT_2X_CANDIDATE,
            CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
            mature,
            2.0,
            BASE_PARAMETER_GRID,
        ),
        FixedTimeSelectiveCandidateConfig(
            FIXED_TIME_SELECTIVE_WEIGHT_3X_CANDIDATE,
            CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
            mature,
            3.0,
            BASE_PARAMETER_GRID,
        ),
        FixedTimeSelectiveCandidateConfig(
            FIXED_TIME_SELECTIVE_TAIL_REGULARIZED_CANDIDATE,
            CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
            mature,
            2.0,
            TAIL_PARAMETER_GRID,
        ),
        FixedTimeSelectiveCandidateConfig(
            FIXED_TIME_SELECTIVE_REGIME_CANDIDATE,
            CORE_REGIME_REVERSAL_FEATURE_SCHEMA_VERSION,
            regime,
            2.0,
            BASE_PARAMETER_GRID,
        ),
    )


def estimator_frame(markets: int = 10) -> pl.DataFrame:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    seconds = (120, 125, 130, 135, 140)
    rows = []
    for market_index in range(markets):
        window_start = start + timedelta(minutes=5 * market_index)
        for second in seconds:
            rows.append(
                {
                    "market_id": f"market-{market_index:03d}",
                    "window_start": window_start,
                    "observed_at": window_start + timedelta(seconds=second),
                    "seconds_elapsed": second,
                    "label_up": market_index % 2,
                    "binance_sign_up": market_index % 2,
                }
            )
    return pl.DataFrame(rows)


def test_candidate_specs_freeze_five_feature_and_weight_contracts() -> None:
    model = selective_model_config()
    candidates = selective_candidate_configs()
    specs = tuple(selective_candidate_spec(candidate, model) for candidate in candidates)

    assert tuple(spec.name for spec in specs) == tuple(
        candidate.name for candidate in candidates
    )
    assert tuple(len(spec.feature_names) for spec in specs) == (71, 71, 71, 71, 77)
    assert all(spec.recency_half_life_days == 28.0 for spec in specs)
    assert specs[0].row_weight_schedule.start_second is None
    assert tuple(
        (
            spec.row_weight_schedule.start_second,
            spec.row_weight_schedule.end_second_inclusive,
            spec.row_weight_schedule.multiplier,
        )
        for spec in specs[1:]
    ) == (
        (120, 120, 2.0),
        (120, 120, 3.0),
        (120, 120, 2.0),
        (120, 120, 2.0),
    )


@pytest.mark.parametrize(
    ("candidate_index", "expected_ratio"),
    [(1, 2.0), (2, 3.0)],
)
def test_exact_120_weights_preserve_equal_total_per_market(
    candidate_index: int,
    expected_ratio: float,
) -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "market_id": [market for market in ("a", "b") for _ in range(5)],
            "window_start": [start] * 10,
            "seconds_elapsed": [120, 125, 130, 135, 140] * 2,
        }
    )
    spec = selective_candidate_spec(
        selective_candidate_configs()[candidate_index],
        selective_model_config(),
    )

    weights = candidate_training_weights(frame, spec)

    assert weights[0] / weights[1] == pytest.approx(expected_ratio)
    assert weights[5] / weights[6] == pytest.approx(expected_ratio)
    assert weights[:5].sum() == pytest.approx(weights[5:].sum())
    assert weights.mean() == pytest.approx(1.0)


def test_parameter_grid_resolver_uses_base_and_exact_three_tail_combinations() -> None:
    core = SimpleNamespace(
        model=SimpleNamespace(
            histogram_candidates=(
                HistogramCandidate(0.05, 160, 15, 100, 0.1),
                HistogramCandidate(0.08, 220, 31, 150, 2.0),
            )
        )
    )
    model = selective_model_config()
    candidates = selective_candidate_configs()

    base = selective_parameter_grid(candidates[0], model, core)
    tail = selective_parameter_grid(candidates[3], model, core)

    assert base == (
        {
            "learning_rate": 0.05,
            "max_iter": 160,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 100,
            "l2_regularization": 0.1,
        },
        {
            "learning_rate": 0.08,
            "max_iter": 220,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 150,
            "l2_regularization": 2.0,
        },
    )
    assert tail == (
        {
            "learning_rate": 0.05,
            "max_iter": 160,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 200,
            "l2_regularization": 2.0,
        },
        {
            "learning_rate": 0.03,
            "max_iter": 220,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 300,
            "l2_regularization": 5.0,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 160,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 200,
            "l2_regularization": 2.0,
        },
    )


def test_selective_tuner_fits_all_context_but_scores_only_exact_120(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = estimator_frame()
    spec = selective_candidate_spec(
        selective_candidate_configs()[0],
        selective_model_config(),
    )
    parameters = (
        {
            "learning_rate": 0.05,
            "max_iter": 160,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 100,
            "l2_regularization": 0.1,
        },
        {
            "learning_rate": 0.08,
            "max_iter": 160,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 150,
            "l2_regularization": 1.0,
        },
    )
    fit_frames: list[pl.DataFrame] = []

    class FakeModel:
        estimator = object()

        @staticmethod
        def raw_probability(scoring: pl.DataFrame) -> np.ndarray:
            assert scoring["seconds_elapsed"].unique().to_list() == [120]
            return np.where(scoring["label_up"].to_numpy() == 1, 0.99, 0.01)

    def fake_fit_model(
        observed_frame: pl.DataFrame,
        _spec: object,
        _parameters: dict[str, object],
        _core: object,
    ) -> FakeModel:
        fit_frames.append(observed_frame)
        return FakeModel()

    monkeypatch.setattr(fixed_time_selective_training, "fit_model", fake_fit_model)
    monkeypatch.setattr(
        fixed_time_selective_training,
        "estimator_converged",
        lambda _estimator: True,
    )

    model, tuning = tune_and_fit_selective_model(
        frame,
        spec,
        parameters,
        SimpleNamespace(),
        120,
        0.15,
        0.10,
        0.95,
    )

    assert isinstance(model, FakeModel)
    assert len(fit_frames) == 3
    assert all(
        set(observed["seconds_elapsed"].unique().to_list())
        == {120, 125, 130, 135, 140}
        for observed in fit_frames
    )
    assert fit_frames[0].height == fit_frames[1].height == 40
    assert fit_frames[2].height == frame.height
    assert tuning["hyperparameter_scoring_seconds"] == [120]
    assert tuning["final_fit"]["rows"] == frame.height
    assert len(tuning["candidates"]) == 2
    assert all(
        record["exact_120_inner_validation"]["rows"] == 2
        and record["primary"]["threshold_selection"][
            "labels_used_for_threshold_selection"
        ]
        is False
        and record["secondary"]["threshold_selection"][
            "labels_used_for_threshold_selection"
        ]
        is False
        for record in tuning["candidates"]
    )
    assert tuning["selected_hyperparameters"] == parameters[0]


def test_tuning_rank_uses_the_frozen_lexicographic_priority() -> None:
    def record(
        *,
        up: float = 0.91,
        down: float = 0.92,
        primary_accuracy: float = 0.93,
        secondary_accuracy: float = 0.95,
        balanced: float = 0.915,
        hard_rate: float = 0.04,
        exact_log_loss: float = 0.20,
    ) -> dict[str, object]:
        return {
            "primary": {
                "metrics": {
                    "up_recall": up,
                    "down_recall": down,
                    "accuracy": primary_accuracy,
                    "balanced_accuracy": balanced,
                },
                "hard_confident_errors": {
                    "all": {"hard_confident_error_rate_selected": hard_rate}
                },
            },
            "secondary": {"metrics": {"accuracy": secondary_accuracy}},
            "exact_120_log_loss": exact_log_loss,
        }

    rank = fixed_time_selective_training._selective_tuning_rank
    base = rank(record())

    assert rank(record(up=0.90, primary_accuracy=1.0)) < base
    assert rank(record(primary_accuracy=0.92, secondary_accuracy=1.0)) < base
    assert rank(record(secondary_accuracy=0.94, balanced=1.0)) < base
    assert rank(record(balanced=0.90, hard_rate=0.0)) < base
    assert rank(record(hard_rate=0.05, exact_log_loss=0.0)) < base
    assert rank(record(exact_log_loss=0.21)) < base


def test_confidence_quantile_selection_is_label_blind_and_tie_deterministic() -> None:
    frame = estimator_frame(markets=6).filter(pl.col("seconds_elapsed") == 120)
    probability = np.array([0.99, 0.04, 0.96, 0.04, 0.80, 0.20])
    scored = fixed_time_selective_training.scored_prediction_rows(frame, probability)
    changed_labels = scored.with_columns((1 - pl.col("label_up")).alias("label_up"))

    threshold, selected, evidence = (
        fixed_time_selective_training._empirical_confidence_quantile(
            scored,
            target_coverage=2 / 6,
        )
    )
    changed_threshold, changed_selected, changed_evidence = (
        fixed_time_selective_training._empirical_confidence_quantile(
            changed_labels,
            target_coverage=2 / 6,
        )
    )

    assert threshold == changed_threshold == 0.96
    assert selected["market_id"].to_list() == changed_selected["market_id"].to_list()
    assert evidence["labels_used_for_threshold_selection"] is False
    assert changed_evidence["labels_used_for_threshold_selection"] is False


def test_predicted_side_hard_errors_sum_with_global_exposure_denominator() -> None:
    selected = pl.DataFrame(
        {
            "market_id": [f"market-{index}" for index in range(6)],
            "predicted_up": [1, 1, 1, 0, 0, 0],
            "correct": [False, False, True, False, False, True],
            "confidence": [0.99, 0.96, 0.98, 0.97, 0.80, 0.99],
        }
    )

    metrics = predicted_side_hard_error_metrics(selected, 10, 0.95)

    assert metrics["hard_confident_error_sum_invariant"] is True
    assert metrics["exposure_denominator"] == "global_eligible_markets"
    assert metrics["all"]["hard_confident_error_markets"] == 3
    assert metrics["up"]["hard_confident_error_markets"] == 2
    assert metrics["down"]["hard_confident_error_markets"] == 1
    assert metrics["all"]["hard_confident_error_exposure_rate"] == pytest.approx(0.3)
    assert metrics["up"]["hard_confident_error_exposure_rate"] == pytest.approx(0.2)
    assert metrics["down"]["hard_confident_error_exposure_rate"] == pytest.approx(0.1)
    assert {
        metrics[side]["eligible_markets"] for side in ("all", "up", "down")
    } == {10}
