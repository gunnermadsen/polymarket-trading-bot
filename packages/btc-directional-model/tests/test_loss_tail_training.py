from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from btc_directional_model import loss_tail_training
from btc_directional_model.core_config import HistogramCandidate
from btc_directional_model.core_training import ProbabilityCalibrator
from btc_directional_model.loss_tail_oof import BOUNDARY_CORRECTNESS_FEATURES
from btc_directional_model.loss_tail_training import (
    EXPECTED_LOSS_TAIL_FEATURE_COUNT,
    LOSS_TAIL_FEATURES,
    ExtraTreesHyperparameters,
    boundary_residual_training_weights,
    fit_boundary_correctness_logistic,
    fit_boundary_residual_economic_hgb,
    fit_direct_loss_tail_hgb,
    fit_extra_trees_rare_regime,
    fit_unweighted_platt_calibrator,
    loss_tail_training_weights,
    materialize_boundary_proposals,
    score_calibrated_candidate,
    validate_loss_tail_feature_contract,
    wrong_side_economic_severity,
)


def _core_config() -> object:
    return SimpleNamespace(
        model=SimpleNamespace(
            random_seed=37,
            c_candidates=(0.2, 1.0),
            histogram_candidates=(
                HistogramCandidate(
                    learning_rate=0.08,
                    max_iter=30,
                    max_leaf_nodes=7,
                    min_samples_leaf=4,
                    l2_regularization=1.0,
                ),
            ),
        ),
        compute=SimpleNamespace(threads_per_fit=1),
    )


def _direction_frame(rows: int, *, offset: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(900 + offset)
    labels = ((np.arange(rows) + offset) % 2).astype(np.int8)
    signed = labels * 2.0 - 1.0
    features = {
        feature: signed * (0.2 + (index % 7) * 0.03) + rng.normal(0.0, 0.8, rows)
        for index, feature in enumerate(LOSS_TAIL_FEATURES)
    }
    start = datetime(2026, 4, 13, tzinfo=UTC) + timedelta(days=offset)
    return pl.DataFrame(
        features
        | {
            "market_id": [f"market-{offset}-{index}" for index in range(rows)],
            "window_start": [start + timedelta(minutes=5 * index) for index in range(rows)],
            "seconds_elapsed": [120] * rows,
            "label_up": labels,
            "up_entry_debit_per_share": np.where(labels == 0, 0.88, 0.42),
            "down_entry_debit_per_share": np.where(labels == 1, 0.91, 0.39),
        }
    )


def test_loss_tail_feature_contract_is_exact_and_disjoint() -> None:
    assert validate_loss_tail_feature_contract() == LOSS_TAIL_FEATURES
    assert len(LOSS_TAIL_FEATURES) == EXPECTED_LOSS_TAIL_FEATURE_COUNT == 124
    assert len(set(LOSS_TAIL_FEATURES)) == len(LOSS_TAIL_FEATURES)

    with pytest.raises(ValueError, match="missing features"):
        validate_loss_tail_feature_contract(pl.DataFrame({"market_id": ["missing"]}))


def test_wrong_side_severity_and_market_equal_weighting_follow_frozen_formula() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "b"],
            "label_up": [1, 0, 1],
            "up_entry_debit_per_share": [0.20, 0.80, 0.50],
            "down_entry_debit_per_share": [0.90, 0.20, 0.50],
        }
    )

    severity = wrong_side_economic_severity(frame)
    weights = loss_tail_training_weights(frame)

    assert severity.tolist() == pytest.approx([9.0, 4.0, 1.0])
    # Equal-market weights are [0.75, 0.75, 1.5]. Multiplying by severity
    # and normalizing the result to mean one yields the vector below.
    assert weights.tolist() == pytest.approx([1.8, 0.8, 0.4])
    assert float(weights.mean()) == pytest.approx(1.0)

    capped = frame[:1].with_columns(pl.lit(0.99).alias("down_entry_debit_per_share"))
    assert wrong_side_economic_severity(capped).item() == pytest.approx(10.0)


def test_unweighted_platt_does_not_pass_training_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class _RawModel:
        def raw_logit(self, frame: pl.DataFrame) -> np.ndarray:
            return np.array([-2.0, -0.5, 0.5, 2.0])

    class _FakeLogistic:
        max_iter = 500

        def __init__(self, **kwargs: object) -> None:
            calls.append({"init": kwargs})
            self.coef_ = np.array([[1.25]])
            self.intercept_ = np.array([-0.1])
            self.n_iter_ = np.array([4])

        def fit(self, matrix: np.ndarray, labels: np.ndarray, **kwargs: object) -> object:
            calls.append({"fit": kwargs, "rows": len(labels)})
            return self

    monkeypatch.setattr(loss_tail_training, "LogisticRegression", _FakeLogistic)
    frame = pl.DataFrame({"correct": [0, 0, 1, 1]})

    calibrator = fit_unweighted_platt_calibrator(
        _RawModel(),  # type: ignore[arg-type]
        frame,
        target_column="correct",
        random_seed=37,
        threads_per_fit=1,
    )

    assert calibrator == ProbabilityCalibrator(1.25, -0.1, True, 4)
    assert calls[-1] == {"fit": {}, "rows": 4}


def test_hgb_and_extra_trees_candidates_fit_and_score_deterministically() -> None:
    fit = _direction_frame(80)
    calibration = _direction_frame(24, offset=20)
    score = _direction_frame(16, offset=40)
    config = _core_config()

    hgb_first = fit_direct_loss_tail_hgb(
        fit,
        calibration,
        core_config=config,  # type: ignore[arg-type]
    )
    hgb_second = fit_direct_loss_tail_hgb(
        fit,
        calibration,
        core_config=config,  # type: ignore[arg-type]
    )
    assert hgb_first.model.family == "histogram"
    assert hgb_first.feature_names == LOSS_TAIL_FEATURES
    assert hgb_first.tuning["calibration_weighting"] == "unweighted_rows"
    assert hgb_first.probability(score).tolist() == pytest.approx(
        hgb_second.probability(score).tolist(), abs=1e-12
    )

    tree_parameters = (
        ExtraTreesHyperparameters(
            n_estimators=24,
            max_depth=7,
            min_samples_leaf=2,
            max_features="sqrt",
        ),
    )
    trees_first = fit_extra_trees_rare_regime(
        fit,
        calibration,
        core_config=config,  # type: ignore[arg-type]
        n_jobs=1,
        parameter_candidates=tree_parameters,
    )
    trees_second = fit_extra_trees_rare_regime(
        fit,
        calibration,
        core_config=config,  # type: ignore[arg-type]
        n_jobs=1,
        parameter_candidates=tree_parameters,
    )
    assert trees_first.model.family == "extra_trees"
    assert trees_first.model.estimator.n_jobs == 1
    assert trees_first.probability(score).tolist() == pytest.approx(
        trees_second.probability(score).tolist(), abs=1e-12
    )

    scored = score_calibrated_candidate(trees_first, score)
    assert scored.height == score.height
    assert scored["raw_probability"].is_between(0.0, 1.0).all()
    assert scored["probability"].is_between(0.0, 1.0).all()


def test_correctness_logistic_uses_explicit_causal_features_and_target() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)

    def frame(rows: int, offset: int) -> pl.DataFrame:
        correct = ((np.arange(rows) + offset) % 2).astype(np.int8)
        signed = correct * 2.0 - 1.0
        return pl.DataFrame(
            {
                "market_id": [f"correct-{offset}-{index}" for index in range(rows)],
                "window_start": [
                    start + timedelta(minutes=5 * (offset + index)) for index in range(rows)
                ],
                "source_probability": 0.5 + signed * 0.2,
                "source_disagreement": 1.0 - correct,
                "selected_debit": 0.7 + (1.0 - correct) * 0.2,
                "boundary_direction_correct": correct,
            }
        )

    fit = frame(80, 0)
    calibration = frame(24, 100)
    scored = frame(12, 200)
    features = ("source_probability", "source_disagreement", "selected_debit")

    result = fit_boundary_correctness_logistic(
        fit,
        calibration,
        feature_names=features,
        core_config=_core_config(),  # type: ignore[arg-type]
    )

    assert result.target_column == "boundary_direction_correct"
    assert result.feature_names == features
    assert result.model.family == "logistic"
    assert np.all((result.probability(scored) >= 0.0) & (result.probability(scored) <= 1.0))


def _proposal_frame(rows: int, *, offset: int = 0) -> pl.DataFrame:
    correct = ((np.arange(rows) + offset) % 3 != 0).astype(np.int8)
    rng = np.random.default_rng(4100 + offset)
    start = datetime(2026, 4, 13, tzinfo=UTC) + timedelta(days=offset)
    features = {
        name: correct * 0.2 + rng.normal(0.0, 0.4, rows) for name in BOUNDARY_CORRECTNESS_FEATURES
    }
    return pl.DataFrame(
        features
        | {
            "market_id": [f"proposal-{offset}-{index}" for index in range(rows)],
            "window_start": [start + timedelta(minutes=5 * index) for index in range(rows)],
            "seconds_elapsed": [120] * rows,
            "boundary_direction_correct": correct,
            "fee_inclusive_debit": np.where(correct == 0, 0.95, 0.75),
        }
    )


def test_boundary_residual_hgb_uses_exact_proposal_severity_and_is_deterministic() -> None:
    fit = _proposal_frame(90)
    calibration = _proposal_frame(30, offset=20)
    score = _proposal_frame(15, offset=40)

    weights = boundary_residual_training_weights(fit)
    assert set(weights[fit["boundary_direction_correct"].to_numpy() == 1]) == {1.0}
    assert set(weights[fit["boundary_direction_correct"].to_numpy() == 0]) == {10.0}

    first = fit_boundary_residual_economic_hgb(
        fit,
        calibration,
        feature_names=BOUNDARY_CORRECTNESS_FEATURES,
        core_config=_core_config(),  # type: ignore[arg-type]
    )
    second = fit_boundary_residual_economic_hgb(
        fit,
        calibration,
        feature_names=BOUNDARY_CORRECTNESS_FEATURES,
        core_config=_core_config(),  # type: ignore[arg-type]
    )

    assert first.model.family == "histogram"
    assert first.target_column == "boundary_direction_correct"
    assert first.feature_names == BOUNDARY_CORRECTNESS_FEATURES
    assert first.tuning["training_weighting"] == "incorrect_proposal_severity"
    assert first.probability(score).tolist() == pytest.approx(
        second.probability(score).tolist(), abs=1e-12
    )


def test_materialize_boundary_proposals_keeps_only_first_causal_trade_per_market() -> None:
    start = datetime(2026, 6, 9, tzinfo=UTC)
    rows = 4
    frame = pl.DataFrame(
        {name: np.zeros(rows) for name in BOUNDARY_CORRECTNESS_FEATURES}
        | {
            "market_id": ["a", "a", "b", "b"],
            "window_start": [
                start,
                start,
                start + timedelta(minutes=5),
                start + timedelta(minutes=5),
            ],
            "observed_at": [
                start + timedelta(seconds=60),
                start + timedelta(seconds=65),
                start + timedelta(minutes=6),
                start + timedelta(minutes=6, seconds=5),
            ],
            "seconds_elapsed": [60, 65, 60, 65],
            "walk_forward_block": ["evaluation_jun09"] * rows,
            "label_up": [1, 1, 0, 0],
            "oof_boundary_probability_up": [0.60, 0.95, 0.05, 0.04],
            "oof_boundary_confidence": [0.60, 0.95, 0.95, 0.96],
            "oof_boundary_predicted_up": [1, 1, 0, 0],
            "oof_boundary_no_trade": [1, 0, 0, 0],
            "boundary_direction_correct": [1, 1, 1, 1],
            "up_ask_vwap_5": [0.6, 0.7, 0.1, 0.1],
            "down_ask_vwap_5": [0.4, 0.3, 0.9, 0.9],
            "up_fee_per_share": [0.01] * rows,
            "down_fee_per_share": [0.01] * rows,
            "up_entry_debit_per_share": [0.61, 0.71, 0.11, 0.11],
            "down_entry_debit_per_share": [0.41, 0.31, 0.91, 0.91],
            "realized_up_net_per_share": [0.39, 0.29, -0.11, -0.11],
            "realized_down_net_per_share": [-0.41, -0.31, 0.09, 0.09],
        }
    ).with_columns(
        pl.Series("oof_boundary_confidence", [0.60, 0.95, 0.95, 0.96]),
        pl.Series("oof_boundary_probability_up", [0.60, 0.95, 0.05, 0.04]),
        pl.Series("oof_boundary_predicted_up", [1, 1, 0, 0]),
        pl.Series("oof_boundary_no_trade", [1, 0, 0, 0]),
        pl.Series("boundary_direction_correct", [1, 1, 1, 1]),
    )

    proposals = materialize_boundary_proposals(frame)

    assert proposals.select("market_id", "seconds_elapsed").rows() == [("a", 65), ("b", 60)]
    assert proposals["market_id"].n_unique() == proposals.height == 2
    assert proposals["boundary_direction"].to_list() == [1, 0]
    assert proposals["fee_inclusive_debit"].to_list() == pytest.approx([0.71, 0.91])
