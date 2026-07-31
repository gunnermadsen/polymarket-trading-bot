from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from btc_directional_model import fixed_time_reversal_training
from btc_directional_model.core_config import HistogramCandidate
from btc_directional_model.core_training import candidate_training_weights
from btc_directional_model.fixed_time_reversal_config import (
    PATH_PERSISTENCE_TARGET,
    load_fixed_time_reversal_config,
)
from btc_directional_model.fixed_time_reversal_training import (
    eligible_complete_market_frame,
    reversal_candidate_spec,
    reversal_parameter_grid,
    reversal_probability_semantics,
    target_probability_up,
    training_target_frame,
    tune_and_fit_reversal_model,
)
from btc_directional_model.persistence_benchmark import persistence_target_labels


def repository_config() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "btc-5m-directional-fixed-120-reversal-decision-20260321-20260729.toml"
    )


def estimator_frame(markets: int = 10) -> pl.DataFrame:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    seconds = (120, 125, 130, 135, 140)
    rows = []
    for market_index in range(markets):
        window_start = start + timedelta(minutes=5 * market_index)
        sign_up = market_index % 2
        reverses = market_index % 4 >= 2
        label_up = 1 - sign_up if reverses else sign_up
        path = 10.0 if sign_up else -10.0
        for second in seconds:
            rows.append(
                {
                    "market_id": f"market-{market_index:03d}",
                    "window_start": window_start,
                    "observed_at": window_start + timedelta(seconds=second),
                    "seconds_elapsed": second,
                    "label_up": label_up,
                    "binance_sign_up": sign_up,
                    "btc_path_from_window_open_bps": path,
                }
            )
    return pl.DataFrame(rows)


def test_complete_source_filters_each_path_row_without_future_conditioning() -> None:
    frame = estimator_frame(markets=4).with_columns(
        pl.when(
            (pl.col("market_id") == "market-001")
            & (pl.col("seconds_elapsed") == 130)
        )
        .then(0.0)
        .when(
            (pl.col("market_id") == "market-002")
            & (pl.col("seconds_elapsed") == 120)
        )
        .then(0.0)
        .otherwise(pl.col("btc_path_from_window_open_bps"))
        .alias("btc_path_from_window_open_bps")
    )

    eligible = eligible_complete_market_frame(
        frame,
        expected_source_markets=4,
        expected_source_estimator_rows=20,
        expected_eligible_markets=4,
        expected_eligible_estimator_rows=18,
        expected_exact_120_eligible_markets=3,
    )

    assert eligible.height == 18
    assert eligible["market_id"].n_unique() == 4
    assert (
        eligible.filter(pl.col("market_id") == "market-001")["seconds_elapsed"]
        .to_list()
        == [120, 125, 135, 140]
    )
    assert (
        eligible.filter(pl.col("market_id") == "market-002")["seconds_elapsed"]
        .to_list()
        == [125, 130, 135, 140]
    )
    assert "market-001" in eligible.filter(pl.col("seconds_elapsed") == 120)[
        "market_id"
    ].to_list()
    assert set(eligible["seconds_elapsed"].unique().to_list()) == {
        120,
        125,
        130,
        135,
        140,
    }
    with pytest.raises(RuntimeError, match="source estimator row count changed"):
        eligible_complete_market_frame(frame, expected_source_estimator_rows=19)
    with pytest.raises(RuntimeError, match="eligible estimator row count changed"):
        eligible_complete_market_frame(frame, expected_eligible_estimator_rows=19)
    with pytest.raises(RuntimeError, match="exact-120 eligible market count changed"):
        eligible_complete_market_frame(
            frame,
            expected_exact_120_eligible_markets=4,
        )


def test_complete_market_eligibility_rejects_incomplete_estimator_context() -> None:
    frame = estimator_frame(markets=3).filter(
        ~(
            (pl.col("market_id") == "market-001")
            & (pl.col("seconds_elapsed") == 135)
        )
    )

    with pytest.raises(ValueError, match="each 120-140 row exactly once"):
        eligible_complete_market_frame(frame)


def test_path_persistence_target_and_reversal_probabilities_are_explicit() -> None:
    frame = estimator_frame(markets=4).filter(pl.col("seconds_elapsed") == 120)
    target = training_target_frame(frame, PATH_PERSISTENCE_TARGET)
    target_probability = np.array([0.90, 0.80, 0.25, 0.10])

    semantics = reversal_probability_semantics(
        frame,
        target_probability,
        PATH_PERSISTENCE_TARGET,
    )

    assert target["label_up"].to_list() == [1, 1, 0, 0]
    np.testing.assert_allclose(semantics["p_target"], target_probability)
    np.testing.assert_allclose(semantics["p_persistence"], target_probability)
    np.testing.assert_allclose(semantics["p_reversal"], 1.0 - target_probability)
    np.testing.assert_allclose(
        semantics["probability_up"],
        [0.10, 0.80, 0.75, 0.10],
    )
    np.testing.assert_allclose(
        target_probability_up(frame, target_probability, PATH_PERSISTENCE_TARGET),
        semantics["probability_up"],
    )


def test_outcome_control_exposes_comparable_reversal_semantics() -> None:
    frame = estimator_frame(markets=4).filter(pl.col("seconds_elapsed") == 120)
    probability_up = np.array([0.90, 0.20, 0.25, 0.80])

    semantics = reversal_probability_semantics(
        frame,
        probability_up,
        "outcome_up",
    )

    np.testing.assert_allclose(semantics["probability_up"], probability_up)
    np.testing.assert_allclose(
        semantics["p_persistence"],
        [0.10, 0.20, 0.75, 0.80],
    )
    np.testing.assert_allclose(
        semantics["p_reversal"],
        [0.90, 0.80, 0.25, 0.20],
    )


def test_candidate_specs_use_common_71_features_and_1x_market_weights() -> None:
    config = load_fixed_time_reversal_config(repository_config())
    specs = tuple(
        reversal_candidate_spec(candidate, config.model)
        for candidate in config.candidates
    )

    assert tuple(len(spec.feature_names) for spec in specs) == (71, 71, 71)
    assert all(spec.recency_half_life_days == 28.0 for spec in specs)
    assert all(spec.row_weight_policy == "market_equal" for spec in specs)
    assert all(spec.row_weight_schedule.start_second is None for spec in specs)

    frame = estimator_frame(markets=2)
    weights = candidate_training_weights(frame, specs[0])
    assert weights[:5].max() == pytest.approx(weights[:5].min())
    assert weights[5:].max() == pytest.approx(weights[5:].min())
    assert weights.mean() == pytest.approx(1.0)


def test_parameter_grid_resolver_uses_base_and_frozen_three_tail_combinations() -> None:
    config = load_fixed_time_reversal_config(repository_config())
    core = SimpleNamespace(
        model=SimpleNamespace(
            histogram_candidates=(
                HistogramCandidate(0.05, 160, 15, 100, 0.1),
                HistogramCandidate(0.08, 220, 31, 150, 2.0),
            )
        )
    )

    base = reversal_parameter_grid(config.candidates[0], config.model, core)
    tail = reversal_parameter_grid(config.candidates[2], config.model, core)

    assert len(base) == 2
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


def test_reversal_tuner_encodes_target_and_scores_only_exact_120(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_fixed_time_reversal_config(repository_config())
    source = estimator_frame().with_columns(
        pl.when(
            ((pl.col("market_id") == "market-001") & (pl.col("seconds_elapsed") == 130))
            | ((pl.col("market_id") == "market-002") & (pl.col("seconds_elapsed") == 125))
        )
        .then(0.0)
        .otherwise(pl.col("btc_path_from_window_open_bps"))
        .alias("btc_path_from_window_open_bps")
    )
    frame = eligible_complete_market_frame(
        source,
        expected_source_markets=10,
        expected_source_estimator_rows=50,
        expected_eligible_markets=10,
        expected_eligible_estimator_rows=48,
        expected_exact_120_eligible_markets=10,
    )
    spec = reversal_candidate_spec(config.candidates[1], config.model)
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
            target = persistence_target_labels(scoring)
            return np.where(target == 1, 0.99, 0.01)

    def fake_fit_model(
        observed_frame: pl.DataFrame,
        _spec: object,
        _parameters: dict[str, object],
        _core: object,
    ) -> FakeModel:
        fit_frames.append(observed_frame)
        return FakeModel()

    monkeypatch.setattr(fixed_time_reversal_training, "fit_model", fake_fit_model)
    monkeypatch.setattr(
        fixed_time_reversal_training,
        "estimator_converged",
        lambda _estimator: True,
    )

    model, tuning = tune_and_fit_reversal_model(
        frame,
        spec,
        parameters,
        SimpleNamespace(),
        PATH_PERSISTENCE_TARGET,
        120,
        0.10,
        0.08,
    )

    assert isinstance(model, FakeModel)
    assert len(fit_frames) == 3
    assert any(
        observed.group_by("market_id").len()["len"].min() < 5
        for observed in fit_frames
    )
    assert all(
        set(observed["seconds_elapsed"].unique().to_list())
        == {120, 125, 130, 135, 140}
        for observed in fit_frames
    )
    assert all(
        observed["label_up"].to_list()
        == persistence_target_labels(
            frame.join(
                observed.select("market_id").unique(),
                on="market_id",
                how="inner",
            ).sort(["window_start", "market_id", "observed_at"])
        ).tolist()
        for observed in fit_frames
    )
    assert tuning["target_kind"] == PATH_PERSISTENCE_TARGET
    assert tuning["estimator_positive_class"] == PATH_PERSISTENCE_TARGET
    assert tuning["probability_semantics"]["p_reversal"] == "1 - p_persistence"
    assert tuning["hyperparameter_scoring_seconds"] == [120]
    assert tuning["primary_target_coverage"] == 0.10
    assert tuning["diagnostic_target_coverage"] == 0.08
    assert tuning["final_fit"]["rows"] == frame.height
    assert len(tuning["candidates"]) == 2
    assert tuning["selected_hyperparameters"] == parameters[0]


def test_reversal_ties_follow_executable_direction_override_semantics() -> None:
    frame = estimator_frame(markets=2).filter(pl.col("seconds_elapsed") == 120)
    semantics = reversal_probability_semantics(
        frame,
        np.array([0.50, 0.50]),
        PATH_PERSISTENCE_TARGET,
    )

    scored = fixed_time_reversal_training._scored_reversal_rows(frame, semantics)

    assert scored["p_reversal"].to_list() == [0.5, 0.5]
    assert scored["predicted_up"].to_list() == [1, 1]
    assert scored["path_overridden"].to_list() == [True, False]
    assert scored["predicted_reversal"].to_list() == [True, False]


def test_scored_reversal_rows_pin_override_and_false_up_evidence() -> None:
    frame = estimator_frame(markets=4).filter(pl.col("seconds_elapsed") == 120)
    semantics = reversal_probability_semantics(
        frame,
        np.array([0.90, 0.80, 0.25, 0.10]),
        PATH_PERSISTENCE_TARGET,
    )

    scored = fixed_time_reversal_training._scored_reversal_rows(frame, semantics)
    metrics = fixed_time_reversal_training._override_metrics(
        scored,
        eligible_markets=4,
    )

    assert scored["predicted_reversal"].to_list() == [False, False, True, True]
    assert scored["path_overridden"].to_list() == [False, False, True, True]
    assert metrics["override_markets"] == 2
    assert metrics["override_precision"] == 1.0
    assert metrics["actual_reversal_markets"] == 2
    assert metrics["missed_reversal_markets"] == 0
    assert metrics["false_up_markets"] == 0


def test_tuning_rank_uses_decision_and_override_objectives() -> None:
    def record(
        *,
        up: float = 0.93,
        down: float = 0.94,
        primary_accuracy: float = 0.95,
        diagnostic_accuracy: float = 0.96,
        balanced: float = 0.935,
        override_precision: float | None = 0.80,
        false_up_exposure: float = 0.01,
        outcome_log_loss: float = 0.15,
    ) -> dict[str, object]:
        return {
            "primary": {
                "metrics": {
                    "up_recall": up,
                    "down_recall": down,
                    "accuracy": primary_accuracy,
                    "balanced_accuracy": balanced,
                },
                "override_metrics": {
                    "override_precision": override_precision,
                    "false_up_exposure_rate": false_up_exposure,
                },
            },
            "diagnostic": {"metrics": {"accuracy": diagnostic_accuracy}},
            "exact_120_outcome_log_loss": outcome_log_loss,
        }

    rank = fixed_time_reversal_training._reversal_tuning_rank
    base = rank(record())

    assert rank(record(up=0.92, primary_accuracy=1.0)) < base
    assert rank(record(primary_accuracy=0.94, diagnostic_accuracy=1.0)) < base
    assert rank(record(diagnostic_accuracy=0.95, balanced=1.0)) < base
    assert rank(record(balanced=0.92, override_precision=1.0)) < base
    assert rank(record(override_precision=0.70, false_up_exposure=0.0)) < base
    assert rank(record(false_up_exposure=0.02, outcome_log_loss=0.0)) < base
    assert rank(record(outcome_log_loss=0.16)) < base
