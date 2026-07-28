from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model.benchmark_config import load_entry_benchmark_config
from btc_directional_model.core_training import (
    FittedCoreModel,
    FrozenTrainingBundle,
    ProbabilityCalibrator,
)
from btc_directional_model.offline_challengers import (
    PREOPEN_CANDIDATE,
    STRICT_BOOK_DELTA_FEATURES,
    STRICT_BOOK_FEATURES,
    STRICT_BOOK_TEN_SHARE_FEATURES,
    STRICT_BOOK_V2_FEATURES,
    derive_strict_book_feature_frame,
    derive_strict_book_frame,
    evaluate_strict_cohort_candidate,
    preopen_candidate_spec,
    score_strict_cohort_candidate,
    strict_book_candidate_spec,
    strict_cohort_candidate_spec,
)


def repository_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-entry-benchmark-20260421-20260720.toml"
    )


def evidence_row(
    market_id: str,
    observed_at: datetime,
    *,
    strict: bool,
    up_shift: float = 0.0,
) -> dict[str, object]:
    return {
        "market_id": market_id,
        "observed_at": observed_at,
        "strict_both_side_eligible": strict,
        "strict_both_side_eligible_10": strict,
        "up_provider_received_at": observed_at - timedelta(milliseconds=20),
        "up_best_bid": 0.40 + up_shift,
        "up_best_ask": 0.42 + up_shift,
        "up_best_bid_size": 10.0,
        "up_best_ask_size": 10.0,
        "up_bid_depth": 100.0,
        "up_ask_depth": 90.0,
        "up_ask_vwap_5": 0.43 + up_shift,
        "up_ask_vwap_10": 0.44 + up_shift,
        "up_imbalance": 0.10,
        "down_provider_received_at": observed_at - timedelta(milliseconds=30),
        "down_best_bid": 0.56,
        "down_best_ask": 0.58,
        "down_best_bid_size": 10.0,
        "down_best_ask_size": 10.0,
        "down_bid_depth": 90.0,
        "down_ask_depth": 100.0,
        "down_ask_vwap_5": 0.59,
        "down_ask_vwap_10": 0.60,
        "down_imbalance": -0.10,
        "quality_flags": 0 if strict else 16,
    }


def test_book_features_are_derived_only_after_strict_routing() -> None:
    observed_at = datetime(2026, 6, 8, 0, 1, tzinfo=UTC)
    core = pl.DataFrame(
        {
            "market_id": ["valid", "invalid"],
            "observed_at": [observed_at, observed_at],
            "label_up": [1, 0],
        }
    )
    evidence = pl.DataFrame(
        [
            evidence_row("valid", observed_at, strict=True),
            evidence_row("invalid", observed_at, strict=False),
        ]
    )

    book = derive_strict_book_frame(core, evidence)

    assert book["market_id"].to_list() == ["valid"]
    assert set(STRICT_BOOK_FEATURES) <= set(book.columns)
    assert "quality_flags" not in book.columns
    assert book["model_eligible"].to_list() == [True]


def test_causal_book_deltas_never_bridge_an_invalid_gap() -> None:
    start = datetime(2026, 6, 8, 0, 1, tzinfo=UTC)
    observed = [start + timedelta(seconds=offset) for offset in (0, 5, 10, 15)]
    core = pl.DataFrame(
        {
            "market_id": ["market"] * 4,
            "observed_at": observed,
            "label_up": [1] * 4,
        }
    )
    evidence = pl.DataFrame(
        [
            evidence_row("market", observed[0], strict=True, up_shift=0.00),
            evidence_row("market", observed[1], strict=False, up_shift=0.40),
            evidence_row("market", observed[2], strict=True, up_shift=0.01),
            evidence_row("market", observed[3], strict=True, up_shift=0.03),
        ]
    )

    book = derive_strict_book_feature_frame(core, evidence)

    assert book["observed_at"].to_list() == [observed[3]]
    assert set(STRICT_BOOK_V2_FEATURES) <= set(book.columns)
    assert book["book_up_mid_delta_5s"].item() == pytest.approx(0.02)
    assert "quality_flags" not in book.columns
    assert not any("provider_age" in column for column in book.columns)


def test_offline_candidates_cannot_silently_enter_current_runtime_schema() -> None:
    config = load_entry_benchmark_config(repository_config())
    preopen = preopen_candidate_spec()
    book = strict_book_candidate_spec(config)
    core_ablation = strict_cohort_candidate_spec(
        config,
        candidate_name="strict_core_ablation",
        include_book_features=False,
    )
    book_ablation = strict_cohort_candidate_spec(
        config,
        candidate_name="strict_book_ablation",
        include_book_features=True,
    )

    assert preopen.name == PREOPEN_CANDIDATE
    assert set(STRICT_BOOK_FEATURES) <= set(book.feature_names)
    assert "quality_flags" not in book.feature_names
    assert not any("provider_age" in feature for feature in book.feature_names)
    assert set(STRICT_BOOK_TEN_SHARE_FEATURES) <= set(
        book_ablation.feature_names
    )
    assert set(STRICT_BOOK_DELTA_FEATURES) <= set(book_ablation.feature_names)
    assert not any(
        "provider_age" in feature for feature in book_ablation.feature_names
    )
    assert set(core_ablation.feature_names) < set(book_ablation.feature_names)
    assert core_ablation.row_weight_schedule == book_ablation.row_weight_schedule
    assert len(preopen.feature_names) > 58
    assert len(book.feature_names) > 58


class ConstantEstimator:
    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        probability = np.full(matrix.shape[0], 0.8)
        return np.column_stack((1 - probability, probability))


def test_reusable_scoring_keeps_exact_rows_and_marks_development_evidence() -> None:
    observed_at = datetime(2026, 7, 16, 0, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "market_id": ["up-market", "down-market"],
            "window_start": [
                datetime(2026, 7, 16, tzinfo=UTC),
                datetime(2026, 7, 16, 0, 5, tzinfo=UTC),
            ],
            "observed_at": [observed_at, observed_at + timedelta(minutes=5)],
            "seconds_elapsed": [60, 60],
            "label_up": [1, 0],
            "binance_sign_up": [1, 0],
            "feature": [1.0, -1.0],
        }
    )
    bundle = FrozenTrainingBundle(
        model=FittedCoreModel(
            candidate_name="strict_core_ablation",
            family="histogram",
            feature_names=("feature",),
            hyperparameters={},
            imputation_medians=np.array([0.0]),
            standardization_means=None,
            standardization_scales=None,
            estimator=ConstantEstimator(),
            row_weight_policy="early_entry_market_equal",
        ),
        calibrator=ProbabilityCalibrator(
            slope=1.0,
            intercept=0.0,
            converged=True,
            iterations=1,
        ),
        confidence_threshold=0.75,
    )

    scored = score_strict_cohort_candidate(frame, bundle)
    evaluation, evaluated_rows = evaluate_strict_cohort_candidate(
        frame,
        bundle,
    )

    assert scored.select("market_id", "observed_at").equals(
        frame.select("market_id", "observed_at")
    )
    assert evaluated_rows.select("market_id", "observed_at").equals(
        scored.select("market_id", "observed_at")
    )
    assert evaluation["evidence_kind"] == "development"
    assert evaluation["evaluation_is_independent"] is False
