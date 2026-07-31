from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model import price_aware_training
from btc_directional_model.continuous_context_features import (
    CONTINUOUS_CONTEXT_MODEL_FEATURES,
)
from btc_directional_model.core_training import ProbabilityCalibrator
from btc_directional_model.offline_challengers import (
    STRICT_BOOK_FIVE_SHARE_V2_FEATURES,
)
from btc_directional_model.price_aware_benchmark import (
    _economic_summary,
    _prediction_artifact_frame,
    select_economic_operating_point,
)
from btc_directional_model.price_aware_config import (
    PriceAwareGateConfig,
    load_price_aware_benchmark_config,
)
from btc_directional_model.price_aware_training import (
    ACCURACY_ANCHORED_ADMISSION_CANDIDATE,
    ADMISSION_FEATURES,
    CORE_ORACLE_OUTCOME_FEATURES,
    OUTCOME_DIRECTION_CORRECT_COLUMN,
    OUTCOME_FEATURES,
    OUTCOME_SIGNAL_BLOCK_COLUMN,
    PRICE_AWARE_FEATURES,
    admission_fit_frame,
    attach_five_share_economic_targets,
    attach_outcome_signals,
    fit_admission_bundle,
    mark_first_economic_action,
    score_admission_actions,
    score_probability_actions,
)


class _FixedAdmissionBundle:
    name = "accuracy_anchored_admission"

    def __init__(self, probability: np.ndarray) -> None:
        self._probability = probability

    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        assert frame.height == len(self._probability)
        return self._probability


def test_price_aware_config_preserves_frozen_data_and_timing_contract() -> None:
    config = load_price_aware_benchmark_config(
        Path("configs/btc-5m-directional-price-aware-economic-20260321-20260729.toml")
    )

    assert config.walk_forward.history_start == datetime(2026, 3, 21, tzinfo=UTC)
    evaluation_block = next(
        block
        for block in config.walk_forward.blocks
        if block.name == config.walk_forward.evaluation_block_name
    )
    assert evaluation_block.end == datetime(2026, 7, 29, tzinfo=UTC)
    assert config.model.quantity == 5.0
    assert config.model.freshness_seconds == 2
    assert not any("kraken" in feature for feature in PRICE_AWARE_FEATURES)
    assert not any(feature.startswith("quality_") for feature in PRICE_AWARE_FEATURES)


def test_outcome_ablation_adds_only_narrow_continuous_context() -> None:
    assert OUTCOME_FEATURES
    assert not set(CORE_ORACLE_OUTCOME_FEATURES) & set(CONTINUOUS_CONTEXT_MODEL_FEATURES)
    assert set(OUTCOME_FEATURES) - set(CORE_ORACLE_OUTCOME_FEATURES) == set(
        CONTINUOUS_CONTEXT_MODEL_FEATURES
    )
    assert len(CONTINUOUS_CONTEXT_MODEL_FEATURES) == 6
    assert not set(OUTCOME_FEATURES) & set(STRICT_BOOK_FIVE_SHARE_V2_FEATURES)
    assert set(OUTCOME_FEATURES) < set(ADMISSION_FEATURES)


def test_five_share_targets_include_vwap_and_fee_once() -> None:
    frame = pl.DataFrame(
        {
            "label_up": [1, 0],
            "fee_rate": [0.25, 0.25],
            "up_ask_vwap_5": [0.40, 0.40],
            "down_ask_vwap_5": [0.60, 0.60],
            "strict_both_side_eligible": [True, True],
        }
    )

    targets = attach_five_share_economic_targets(frame)

    up_fee = 0.25 * 0.40 * 0.60
    down_fee = 0.25 * 0.60 * 0.40
    assert targets["realized_up_net_per_share"].to_list() == pytest.approx(
        [1 - 0.40 - up_fee, -0.40 - up_fee]
    )
    assert targets["realized_down_net_per_share"].to_list() == pytest.approx(
        [-0.60 - down_fee, 1 - 0.60 - down_fee]
    )


def test_admission_candidate_never_flips_to_the_cheaper_opposite_side() -> None:
    feature_values = {feature: [0.0] for feature in ADMISSION_FEATURES}
    frame = attach_five_share_economic_targets(
        pl.DataFrame(
            feature_values
            | {
                "market_id": ["market"],
                "window_start": [datetime(2026, 7, 20, tzinfo=UTC)],
                "observed_at": [datetime(2026, 7, 20, tzinfo=UTC) + timedelta(seconds=120)],
                "seconds_elapsed": [120],
                "label_up": [0],
                "fee_rate": [0.07],
                "up_ask_vwap_5": [0.91],
                "down_ask_vwap_5": [0.21],
                "strict_both_side_eligible": [True],
            }
        )
    )
    with_outcome = attach_outcome_signals(
        frame,
        np.array([0.60]),
        block_name="evaluation",
    )
    scored = score_admission_actions(
        with_outcome,
        _FixedAdmissionBundle(np.array([0.70])),  # type: ignore[arg-type]
    )

    assert scored["outcome_probability_up"].item() == pytest.approx(0.60)
    assert scored["predicted_up"].item() == 1
    assert scored["economic_score"].item() == pytest.approx(0.70 - 0.91 - 0.07 * 0.91 * 0.09)
    assert scored["selected_ask_vwap_5"].item() == pytest.approx(0.91)
    assert scored["admission_probability_correct"].item() == pytest.approx(0.70)


@pytest.mark.parametrize("invalid_label", [None, 2])
def test_outcome_signals_reject_nonbinary_labels(invalid_label: int | None) -> None:
    frame = attach_five_share_economic_targets(
        pl.DataFrame(
            {
                "label_up": [1],
                "fee_rate": [0.07],
                "up_ask_vwap_5": [0.60],
                "down_ask_vwap_5": [0.40],
                "strict_both_side_eligible": [True],
            }
        )
    ).with_columns(pl.lit(invalid_label).alias("label_up"))

    with pytest.raises(ValueError, match="label_up must contain only non-null binary"):
        attach_outcome_signals(frame, np.array([0.60]), block_name="B5")


@pytest.mark.parametrize(
    ("locked_direction", "message"),
    (
        (2, "non-null binary"),
        (0, "must match outcome_probability_up"),
    ),
)
def test_admission_scoring_rejects_invalid_or_inconsistent_locked_direction(
    locked_direction: int,
    message: str,
) -> None:
    feature_values = {feature: [0.0] for feature in ADMISSION_FEATURES}
    frame = attach_five_share_economic_targets(
        pl.DataFrame(
            feature_values
            | {
                "label_up": [1],
                "fee_rate": [0.07],
                "up_ask_vwap_5": [0.60],
                "down_ask_vwap_5": [0.40],
                "strict_both_side_eligible": [True],
            }
        )
    )
    signaled = attach_outcome_signals(
        frame,
        np.array([0.60]),
        block_name="B5",
    ).with_columns(pl.lit(locked_direction).alias("outcome_predicted_up"))

    with pytest.raises(ValueError, match=message):
        score_admission_actions(
            signaled,
            _FixedAdmissionBundle(np.array([0.70])),  # type: ignore[arg-type]
        )


def test_admission_fit_frame_isolates_direction_correctness_target() -> None:
    feature_values = {feature: [0.0, 0.0] for feature in ADMISSION_FEATURES}
    economic = attach_five_share_economic_targets(
        pl.DataFrame(
            feature_values
            | {
                "label_up": [0, 0],
                "fee_rate": [0.07, 0.07],
                "up_ask_vwap_5": [0.60, 0.60],
                "down_ask_vwap_5": [0.40, 0.40],
                "strict_both_side_eligible": [True, True],
            }
        )
    )
    signaled = attach_outcome_signals(
        economic,
        np.array([0.60, 0.40]),
        block_name="fit-oof-1",
    )

    isolated = admission_fit_frame(signaled)

    assert signaled["label_up"].to_list() == [0, 0]
    assert signaled[OUTCOME_DIRECTION_CORRECT_COLUMN].to_list() == [False, True]
    assert isolated["label_up"].to_list() == [0, 1]
    assert isolated[OUTCOME_SIGNAL_BLOCK_COLUMN].to_list() == [
        "fit-oof-1",
        "fit-oof-1",
    ]
    corrupted = signaled.with_columns(pl.lit(2).alias(OUTCOME_DIRECTION_CORRECT_COLUMN))
    with pytest.raises(ValueError, match="non-null binary"):
        admission_fit_frame(corrupted)


def test_admission_fit_reuses_histogram_classifier_with_isolated_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pl.DataFrame(
        {feature: [0.0, 0.0] for feature in ADMISSION_FEATURES}
        | {
            "label_up": [1, 1],
            OUTCOME_DIRECTION_CORRECT_COLUMN: [False, True],
            OUTCOME_SIGNAL_BLOCK_COLUMN: ["oof-1", "oof-1"],
        }
    )

    class _Model:
        candidate_name = ACCURACY_ANCHORED_ADMISSION_CANDIDATE

        def raw_logit(self, scored: pl.DataFrame) -> np.ndarray:
            return np.zeros(scored.height)

    def fake_tune(
        isolated: pl.DataFrame,
        spec: object,
        config: object,
    ) -> tuple[object, dict[str, str]]:
        assert isolated["label_up"].to_list() == [0, 1]
        assert spec.family == "histogram"  # type: ignore[attr-defined]
        assert spec.feature_names == ADMISSION_FEATURES  # type: ignore[attr-defined]
        return _Model(), {"selected": "mock"}

    def fake_calibrator(
        model: object,
        isolated: pl.DataFrame,
        config: object,
        spec: object,
    ) -> ProbabilityCalibrator:
        assert isolated["label_up"].to_list() == [0, 1]
        return ProbabilityCalibrator(1.0, 0.0, True, 1)

    monkeypatch.setattr(price_aware_training, "tune_and_fit_model", fake_tune)
    monkeypatch.setattr(
        price_aware_training,
        "fit_probability_calibrator",
        fake_calibrator,
    )

    bundle, tuning = fit_admission_bundle(
        frame,
        frame,
        core_config=object(),  # type: ignore[arg-type]
        recency_half_life_days=None,
    )

    assert bundle.name == ACCURACY_ANCHORED_ADMISSION_CANDIDATE
    assert bundle.probability(frame).tolist() == pytest.approx([0.5, 0.5])
    assert tuning == {"selected": "mock"}


def test_retired_value_direction_selection_fails_closed() -> None:
    frame = attach_five_share_economic_targets(
        pl.DataFrame(
            {
                "label_up": [1],
                "fee_rate": [0.07],
                "up_ask_vwap_5": [0.91],
                "down_ask_vwap_5": [0.21],
                "strict_both_side_eligible": [True],
            }
        )
    )

    with pytest.raises(ValueError, match="direction must stay locked"):
        score_probability_actions(
            frame,
            np.array([0.60]),
            candidate="retired",
            select_by_value=True,
        )


def test_economic_policy_selects_only_first_positive_action_per_market() -> None:
    start = datetime(2026, 7, 20, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "a", "b", "b"],
            "observed_at": [
                start + timedelta(seconds=value) for value in (120, 125, 130, 120, 125)
            ],
            "seconds_elapsed": [120, 125, 130, 120, 125],
            "economic_score": [-0.01, 0.03, 0.08, 0.01, 0.02],
        }
    )

    marked = mark_first_economic_action(frame, threshold=0.02)
    selected = marked.filter(pl.col("policy_selected")).sort("market_id")

    assert selected.select("market_id", "seconds_elapsed").rows() == [("a", 125)]


def test_prediction_artifact_drops_repeated_training_features() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["market"],
            "window_start": [datetime(2026, 7, 20, tzinfo=UTC)],
            "observed_at": [datetime(2026, 7, 20, tzinfo=UTC)],
            "seconds_elapsed": [120],
            "label_up": [1],
            "candidate": ["candidate"],
            "predicted_up": [1],
            "probability_up": [0.9],
            "confidence": [0.9],
            "correct": [True],
            "policy_selected": [True],
            "btc_return_5s_bps": [12.0],
        }
    )

    artifact = _prediction_artifact_frame(frame)

    assert "market_id" in artifact.columns
    assert "policy_selected" in artifact.columns
    assert "btc_return_5s_bps" not in artifact.columns


def test_operating_point_uses_policy_data_economics_not_entry_time() -> None:
    start = datetime(2026, 7, 2, tzinfo=UTC)
    labels = [1] * 90 + [0] * 10
    base = attach_five_share_economic_targets(
        pl.DataFrame(
            {
                "market_id": [f"market-{index}" for index in range(100)],
                "window_start": [start + timedelta(minutes=5 * index) for index in range(100)],
                "observed_at": [
                    start + timedelta(minutes=5 * index, seconds=120) for index in range(100)
                ],
                "seconds_elapsed": [120] * 100,
                "label_up": labels,
                "fee_rate": [0.07] * 100,
                "up_ask_vwap_5": [0.40] * 100,
                "down_ask_vwap_5": [0.70] * 100,
                "strict_both_side_eligible": [True] * 100,
            }
        )
    )
    scored = score_probability_actions(
        base,
        np.full(100, 0.90),
        candidate="price_aware",
        select_by_value=False,
    )
    gates = PriceAwareGateConfig(
        target_accuracy=0.92,
        minimum_accuracy=0.89,
        minimum_wilson_lower=0.87,
        minimum_coverage=0.20,
        minimum_evaluation_trades=200,
        minimum_fold_trades=30,
        minimum_fold_direction_trades=10,
        minimum_profit_factor=1.05,
        maximum_expected_calibration_error=0.08,
        maximum_selected_net_bias=0.03,
    )

    selected = select_economic_operating_point(
        scored,
        thresholds=(0.0, 0.2, 0.6),
        gates=gates,
    )

    assert selected["threshold"] == 0.0
    assert selected["qualified_on_threshold_blocks"] is False
    assert selected["frontier"][0]["aggregate_metrics"]["accuracy"] == pytest.approx(0.90)
    assert selected["frontier"][0]["aggregate_metrics"][
        "median_selected_ask_vwap_5"
    ] == pytest.approx(0.40)


def test_price_aware_targets_fail_closed_on_unknown_fee() -> None:
    frame = pl.DataFrame(
        {
            "label_up": [1],
            "fee_rate": [0.0],
            "up_ask_vwap_5": [0.40],
            "down_ask_vwap_5": [0.60],
            "strict_both_side_eligible": [True],
        }
    )

    with pytest.raises(RuntimeError, match="require known fees"):
        attach_five_share_economic_targets(frame)


def test_economic_summary_publishes_both_coverage_denominators() -> None:
    start = datetime(2026, 7, 20, tzinfo=UTC)
    frame = attach_five_share_economic_targets(
        pl.DataFrame(
            {
                "market_id": ["market"],
                "window_start": [start],
                "observed_at": [start + timedelta(seconds=120)],
                "seconds_elapsed": [120],
                "label_up": [1],
                "fee_rate": [0.07],
                "up_ask_vwap_5": [0.40],
                "down_ask_vwap_5": [0.60],
                "strict_both_side_eligible": [True],
            }
        )
    )
    selected = score_probability_actions(
        frame,
        np.array([0.90]),
        candidate="price_aware",
        select_by_value=False,
    )

    summary = _economic_summary(
        selected,
        book_qualified_markets=2,
        all_core_markets=4,
    )

    assert summary["book_qualified_coverage"] == pytest.approx(0.50)
    assert summary["all_core_market_coverage"] == pytest.approx(0.25)
