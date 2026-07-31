from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

from btc_directional_model import price_aware_benchmark
from btc_directional_model.core_training import ProbabilityCalibrator
from btc_directional_model.price_aware_benchmark import (
    _candidate_result,
    _check,
    _fit_admission_policy_folds,
    _fit_final_candidates,
    _generate_outcome_oof,
    _outcome_oof_artifact_frame,
    _prediction_artifact_frame,
    _render_report,
    select_walk_forward_operating_point,
)
from btc_directional_model.price_aware_config import (
    PriceAwareGateConfig,
    load_price_aware_benchmark_config,
)
from btc_directional_model.price_aware_training import (
    ACCURACY_ANCHORED_ADMISSION_CANDIDATE,
    ADMISSION_FEATURES,
    BOUNDARY_CONTROL_CANDIDATE,
    OUTCOME_DERIVED_FEATURES,
    OUTCOME_DIRECTION_CORRECT_COLUMN,
    OUTCOME_SIGNAL_BLOCK_COLUMN,
    mark_first_economic_action,
)

CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "btc-5m-directional-price-aware-economic-20260321-20260729.toml"
)


def _config() -> Any:
    return load_price_aware_benchmark_config(CONFIG_PATH)


def _permissive_gates(**overrides: Any) -> PriceAwareGateConfig:
    values: dict[str, Any] = {
        "target_accuracy": 0.75,
        "minimum_accuracy": 0.75,
        "minimum_wilson_lower": 0.0,
        "minimum_coverage": 0.40,
        "minimum_evaluation_trades": 6,
        "minimum_fold_trades": 2,
        "minimum_fold_direction_trades": 1,
        "minimum_profit_factor": 1.01,
        "maximum_expected_calibration_error": 0.20,
        "maximum_selected_net_bias": 0.05,
    }
    values.update(overrides)
    return PriceAwareGateConfig(**values)


def _scored_block(
    block_name: str,
    rows: list[tuple[float, int, int, float]],
) -> pl.DataFrame:
    """Build one already-scored decision per market.

    Each tuple is economic score, predicted direction, official label, and VWAP5.
    """

    block_index = int(block_name.removeprefix("B"))
    start = datetime(2026, 6, 2, tzinfo=UTC) + timedelta(days=block_index * 7)
    predicted = np.array([row[1] for row in rows], dtype=np.int8)
    labels = np.array([row[2] for row in rows], dtype=np.int8)
    correct = predicted == labels
    price = np.array([row[3] for row in rows], dtype=np.float64)
    fee_rate = np.full(len(rows), 0.01)
    fee = fee_rate * price * (1.0 - price)
    probability_up = np.where(predicted == 1, 0.99, 0.01)
    probability_correct = np.where(correct, 0.99, 0.01)
    realized = correct.astype(np.float64) - price - fee
    predicted_net = probability_correct - price - fee
    return pl.DataFrame(
        {
            "market_id": [f"{block_name}-{index}" for index in range(len(rows))],
            "window_start": [start + timedelta(minutes=5 * index) for index in range(len(rows))],
            "observed_at": [
                start + timedelta(minutes=5 * index, seconds=120) for index in range(len(rows))
            ],
            "seconds_elapsed": [120] * len(rows),
            "label_up": labels,
            "candidate": [ACCURACY_ANCHORED_ADMISSION_CANDIDATE] * len(rows),
            "predicted_up": predicted,
            "probability_up": probability_up,
            "confidence": np.maximum(probability_up, 1.0 - probability_up),
            "correct": correct,
            "outcome_probability_up": probability_up,
            "outcome_predicted_up": predicted,
            "outcome_probability_selected": np.full(len(rows), 0.99),
            "outcome_signal_block": [block_name] * len(rows),
            "admission_probability_correct": probability_correct,
            "economic_score": [row[0] for row in rows],
            "direct_net_edge_per_share": predicted_net,
            "predicted_net_per_share": predicted_net,
            "selected_ask_vwap_5": price,
            "selected_fee_per_share": fee,
            "realized_selected_net_per_share": realized,
            "fee_rate": fee_rate,
            "strict_both_side_eligible": [True] * len(rows),
            "execution_evidence_available": [True] * len(rows),
            "model_eligible": [True] * len(rows),
            "walk_forward_block": [block_name] * len(rows),
        }
    )


def _admission_block(block_name: str, start: datetime) -> pl.DataFrame:
    feature_values = {feature: [0.0, 0.0, 0.0, 0.0] for feature in ADMISSION_FEATURES}
    probability = [0.90, 0.10, 0.80, 0.20]
    predicted = [1, 0, 1, 0]
    return pl.DataFrame(
        feature_values
        | {
            "market_id": [f"{block_name}-{index}" for index in range(4)],
            "window_start": [start + timedelta(minutes=index) for index in range(4)],
            "observed_at": [start + timedelta(minutes=index, seconds=120) for index in range(4)],
            "seconds_elapsed": [120] * 4,
            "label_up": [1, 1, 0, 0],
            "outcome_direction_correct": [True, False, False, True],
            "outcome_signal_block": [block_name] * 4,
            "outcome_probability_up": probability,
            "outcome_predicted_up": predicted,
            "outcome_probability_selected": [0.90, 0.90, 0.80, 0.80],
            "outcome_confidence_margin": [0.40, 0.40, 0.30, 0.30],
            "outcome_selected_ask_vwap_5": [0.40] * 4,
            "outcome_selected_entry_debit_per_share": [0.41] * 4,
            "outcome_raw_net_edge_per_share": [0.49, 0.49, 0.39, 0.39],
            "fee_rate": [0.01] * 4,
            "up_ask_vwap_5": [0.40] * 4,
            "down_ask_vwap_5": [0.40] * 4,
            "up_fee_per_share": [0.0024] * 4,
            "down_fee_per_share": [0.0024] * 4,
            "up_entry_debit_per_share": [0.4024] * 4,
            "down_entry_debit_per_share": [0.4024] * 4,
            "realized_up_net_per_share": [0.5976, 0.5976, -0.4024, -0.4024],
            "realized_down_net_per_share": [-0.4024, -0.4024, 0.5976, 0.5976],
            "strict_both_side_eligible": [True] * 4,
            "walk_forward_block": [block_name] * 4,
        }
    )


def _action_frame(frame: pl.DataFrame, candidate: str) -> pl.DataFrame:
    predicted = frame["outcome_predicted_up"].cast(pl.Int8)
    probability = frame["outcome_probability_up"].cast(pl.Float64)
    correct = predicted == frame["label_up"]
    selected_price = np.where(
        predicted.to_numpy() == 1,
        frame["up_ask_vwap_5"].to_numpy(),
        frame["down_ask_vwap_5"].to_numpy(),
    )
    fee = 0.01 * selected_price * (1.0 - selected_price)
    realized = correct.cast(pl.Float64).to_numpy() - selected_price - fee
    return frame.with_columns(
        pl.lit(candidate).alias("candidate"),
        predicted.alias("predicted_up"),
        probability.alias("probability_up"),
        pl.max_horizontal(probability, 1.0 - probability).alias("confidence"),
        correct.alias("correct"),
        pl.lit(0.20).alias("economic_score"),
        pl.lit(0.20).alias("direct_net_edge_per_share"),
        pl.lit(0.20).alias("predicted_net_per_share"),
        pl.Series("selected_ask_vwap_5", selected_price),
        pl.Series("selected_fee_per_share", fee),
        pl.Series("realized_selected_net_per_share", realized),
        pl.lit(0.80).alias("admission_probability_correct"),
        pl.lit(True).alias("model_eligible"),
    )


def test_outcome_oof_is_causal_and_preserves_exact_block_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    history_start = config.walk_forward.history_start
    history_rows = _universal_rows("history", history_start, 20)
    universal_blocks = {
        block.name: _universal_rows(block.name, block.start, 10)
        for block in config.walk_forward.blocks
    }
    universal = pl.concat([history_rows, *universal_blocks.values()], how="vertical")
    book_blocks = {name: _book_from_universal(frame) for name, frame in universal_blocks.items()}
    model_fit_maxima: list[datetime] = []
    calibration_maxima: list[datetime] = []

    class _Model:
        candidate_name = "outcome"

        def raw_logit(self, frame: pl.DataFrame) -> np.ndarray:
            return np.zeros(frame.height)

    def fake_tune(frame: pl.DataFrame, *_: Any) -> tuple[_Model, dict[str, str]]:
        model_fit_maxima.append(frame["window_start"].max())
        return _Model(), {"selected": "test"}

    def fake_calibrator(
        _model: Any,
        frame: pl.DataFrame,
        *_: Any,
    ) -> ProbabilityCalibrator:
        calibration_maxima.append(frame["window_start"].max())
        return ProbabilityCalibrator(0.0, 0.0, True, 1)

    monkeypatch.setattr(price_aware_benchmark, "tune_and_fit_model", fake_tune)
    monkeypatch.setattr(
        price_aware_benchmark,
        "fit_probability_calibrator",
        fake_calibrator,
    )

    training, attached_book, scored_universal = _generate_outcome_oof(
        universal,
        universal_blocks,
        book_blocks,
        config,
        object(),  # type: ignore[arg-type]
    )

    assert training["selected_candidate"] == BOUNDARY_CONTROL_CANDIDATE
    assert set(training["candidates"]) == {
        BOUNDARY_CONTROL_CANDIDATE,
        "boundary_oracle_outcome",
        "boundary_oracle_continuous_context_outcome",
    }
    selected_blocks = training["candidates"][training["selected_candidate"]]["blocks"]
    for index, block in enumerate(config.walk_forward.blocks):
        assert model_fit_maxima[index] < block.start
        assert calibration_maxima[index] < block.start
        assert (
            selected_blocks[block.name]["fit"]["window_start_max"]
            < selected_blocks[block.name]["calibration"]["window_start_min"]
        )
        assert (
            selected_blocks[block.name]["calibration"]["window_start_max"]
            < selected_blocks[block.name]["score"]["window_start_min"]
        )
        assert set(attached_book[block.name]["outcome_signal_block"].unique()) == {block.name}
        assert set(scored_universal[block.name]["walk_forward_block"].unique()) == {block.name}


def _universal_rows(name: str, start: datetime, markets: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "market_id": [f"{name}-{index}" for index in range(markets)],
            "window_start": [start + timedelta(minutes=5 * index) for index in range(markets)],
            "observed_at": [
                start + timedelta(minutes=5 * index, seconds=120) for index in range(markets)
            ],
            "seconds_elapsed": [120] * markets,
            "label_up": [index % 2 for index in range(markets)],
        }
    )


def _book_from_universal(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.lit(0.01).alias("fee_rate"),
        pl.lit(0.40).alias("up_ask_vwap_5"),
        pl.lit(0.40).alias("down_ask_vwap_5"),
        pl.lit(0.0024).alias("up_fee_per_share"),
        pl.lit(0.0024).alias("down_fee_per_share"),
        pl.lit(0.4024).alias("up_entry_debit_per_share"),
        pl.lit(0.4024).alias("down_entry_debit_per_share"),
        (pl.col("label_up") - 0.4024).alias("realized_up_net_per_share"),
        (1.0 - pl.col("label_up") - 0.4024).alias("realized_down_net_per_share"),
        pl.lit(True).alias("strict_both_side_eligible"),
    )


def test_admission_threshold_folds_use_only_prior_oof_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    blocks = {
        block.name: _admission_block(block.name, block.start)
        for block in config.walk_forward.blocks
    }
    fit_calls: list[tuple[set[str], set[str]]] = []
    score_calls: list[set[str]] = []
    fake_bundle = SimpleNamespace(calibrator=ProbabilityCalibrator(1.0, 0.0, True, 1))

    def fake_fit(
        fit: pl.DataFrame,
        calibration: pl.DataFrame,
        **_: Any,
    ) -> tuple[Any, dict[str, str]]:
        fit_calls.append(
            (
                set(fit["walk_forward_block"].unique()),
                set(calibration["walk_forward_block"].unique()),
            )
        )
        return fake_bundle, {"selected": "test"}

    def fake_score(frame: pl.DataFrame, _bundle: Any) -> pl.DataFrame:
        score_calls.append(set(frame["walk_forward_block"].unique()))
        return _action_frame(frame, ACCURACY_ANCHORED_ADMISSION_CANDIDATE)

    monkeypatch.setattr(price_aware_benchmark, "fit_admission_bundle", fake_fit)
    monkeypatch.setattr(price_aware_benchmark, "score_admission_actions", fake_score)

    training, scored = _fit_admission_policy_folds(
        blocks,
        config,
        object(),  # type: ignore[arg-type]
    )

    assert fit_calls == [
        ({"initial_book_history"}, {"validation_may26"}),
        (
            {"initial_book_history", "validation_may26"},
            {"validation_jun02"},
        ),
        (
            {"initial_book_history", "validation_may26", "validation_jun02"},
            {"validation_jun09"},
        ),
    ]
    assert score_calls == [
        {"validation_jun02"},
        {"validation_jun09"},
        {"validation_jul03"},
    ]
    assert list(training) == [
        "validation_jun02",
        "validation_jun09",
        "validation_jul03",
    ]
    assert list(scored) == list(training)
    assert all(
        record["fit"]["window_start_max"]
        < record["calibration"]["window_start_min"]
        < record["score"]["window_start_min"]
        for record in training.values()
    )
    assert not any(
        "confirmation_jul14" in fit | calibration
        for fit, calibration in fit_calls
    )


def test_one_threshold_is_applied_identically_across_b2_b4() -> None:
    rows = [
        (0.20, 1, 1, 0.40),
        (0.20, 0, 0, 0.40),
        (0.05, 1, 0, 0.40),
        (0.05, 0, 1, 0.40),
    ]
    blocks = {name: _scored_block(name, rows) for name in ("B2", "B3", "B4")}

    selected = select_walk_forward_operating_point(
        blocks,
        thresholds=(0.0, 0.10),
        gates=_permissive_gates(),
        all_core_markets_by_block={name: 8 for name in blocks},
    )

    assert selected["threshold"] == pytest.approx(0.10)
    assert selected["qualified_on_threshold_blocks"] is True
    assert selected["threshold_blocks"] == ["B2", "B3", "B4"]
    selected_frontier = next(
        row for row in selected["frontier"] if row["threshold"] == selected["threshold"]
    )
    assert set(selected_frontier["folds"]) == {"B2", "B3", "B4"}
    assert {fold["metrics"]["trades"] for fold in selected_frontier["folds"].values()} == {2}


def test_threshold_qualification_fails_closed_when_one_fold_lacks_a_side() -> None:
    balanced = [
        (0.20, 1, 1, 0.40),
        (0.20, 0, 0, 0.40),
        (0.20, 1, 1, 0.40),
        (0.20, 0, 0, 0.40),
    ]
    up_only = [(0.20, 1, 1, 0.40)] * 4
    blocks = {
        "B2": _scored_block("B2", balanced),
        "B3": _scored_block("B3", up_only),
        "B4": _scored_block("B4", balanced),
    }

    selected = select_walk_forward_operating_point(
        blocks,
        thresholds=(0.10,),
        gates=_permissive_gates(minimum_coverage=0.20),
        all_core_markets_by_block={name: 8 for name in blocks},
    )

    frontier = selected["frontier"][0]
    assert all(check["passed"] for check in frontier["aggregate_checks"])
    assert frontier["folds"]["B2"]["qualified"] is True
    assert frontier["folds"]["B3"]["qualified"] is False
    assert frontier["folds"]["B4"]["qualified"] is True
    assert selected["qualified_on_threshold_blocks"] is False


def test_unqualified_thresholds_use_explicit_diagnostic_fallback_ranking() -> None:
    rows = [
        (0.20, 1, 1, 0.40),
        (0.20, 0, 0, 0.40),
        (0.05, 1, 0, 0.40),
        (0.05, 0, 1, 0.40),
    ]
    blocks = {name: _scored_block(name, rows) for name in ("B2", "B3", "B4")}

    selected = select_walk_forward_operating_point(
        blocks,
        thresholds=(0.0, 0.10),
        gates=_permissive_gates(minimum_coverage=0.90),
        all_core_markets_by_block={name: 4 for name in blocks},
    )

    assert not any(row["qualified"] for row in selected["frontier"])
    expected = max(
        selected["frontier"],
        key=lambda row: (
            min(fold["metrics"]["net_pnl_per_all_core_market"] for fold in row["folds"].values()),
            row["aggregate_metrics"]["net_pnl_per_all_core_market"],
            row["aggregate_metrics"]["accuracy"],
            -row["aggregate_metrics"]["median_selected_ask_vwap_5"],
        ),
    )
    assert expected["threshold"] == pytest.approx(0.10)
    assert selected["threshold"] == expected["threshold"]
    assert selected["qualified_on_threshold_blocks"] is False


def test_b5_is_evaluation_only_for_final_candidate_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    blocks = {
        block.name: _admission_block(block.name, block.start)
        for block in config.walk_forward.blocks
    }
    admission_calls: list[tuple[set[str], set[str]]] = []
    control_fit_calls: list[set[str]] = []
    control_calibration_calls: list[set[str]] = []
    scored_calls: list[tuple[str, set[str]]] = []
    real_control_score = price_aware_benchmark.score_probability_actions
    fake_bundle = SimpleNamespace(calibrator=ProbabilityCalibrator(1.0, 0.0, True, 1))

    def fake_admission_fit(
        fit: pl.DataFrame,
        calibration: pl.DataFrame,
        **_: Any,
    ) -> tuple[Any, dict[str, str]]:
        admission_calls.append(
            (
                set(fit["walk_forward_block"].unique()),
                set(calibration["walk_forward_block"].unique()),
            )
        )
        return fake_bundle, {"selected": "admission"}

    def fake_admission_score(frame: pl.DataFrame, _bundle: Any) -> pl.DataFrame:
        scored_calls.append(("admission", set(frame["walk_forward_block"].unique())))
        return _action_frame(frame, ACCURACY_ANCHORED_ADMISSION_CANDIDATE)

    class _ControlModel:
        candidate_name = BOUNDARY_CONTROL_CANDIDATE

    def fake_control_fit(
        frame: pl.DataFrame,
        *_: Any,
    ) -> tuple[_ControlModel, dict[str, str]]:
        control_fit_calls.append(set(frame["walk_forward_block"].unique()))
        return _ControlModel(), {"selected": "control"}

    def fake_control_calibration(
        _model: Any,
        frame: pl.DataFrame,
        *_: Any,
    ) -> ProbabilityCalibrator:
        control_calibration_calls.append(set(frame["walk_forward_block"].unique()))
        return ProbabilityCalibrator(1.0, 0.0, True, 1)

    class _Frozen:
        def __init__(self, *_: Any) -> None:
            pass

        def probability(self, frame: pl.DataFrame) -> np.ndarray:
            return np.where(frame["label_up"].to_numpy() == 1, 0.90, 0.10)

    def fake_control_score(
        frame: pl.DataFrame,
        probability: np.ndarray,
        **kwargs: Any,
    ) -> pl.DataFrame:
        scored_calls.append(("control", set(frame["walk_forward_block"].unique())))
        stale_columns = {
            *OUTCOME_DERIVED_FEATURES,
            OUTCOME_DIRECTION_CORRECT_COLUMN,
            OUTCOME_SIGNAL_BLOCK_COLUMN,
        }
        assert stale_columns.isdisjoint(frame.columns)
        return real_control_score(frame, probability, **kwargs)

    monkeypatch.setattr(price_aware_benchmark, "fit_admission_bundle", fake_admission_fit)
    monkeypatch.setattr(price_aware_benchmark, "score_admission_actions", fake_admission_score)
    monkeypatch.setattr(price_aware_benchmark, "tune_and_fit_model", fake_control_fit)
    monkeypatch.setattr(
        price_aware_benchmark,
        "fit_probability_calibrator",
        fake_control_calibration,
    )
    monkeypatch.setattr(price_aware_benchmark, "FrozenTrainingBundle", _Frozen)
    monkeypatch.setattr(price_aware_benchmark, "score_probability_actions", fake_control_score)

    training, evaluation = _fit_final_candidates(
        blocks,
        config,
        object(),  # type: ignore[arg-type]
        0.10,
    )

    prior = {
        "initial_book_history",
        "validation_may26",
        "validation_jun02",
        "validation_jun09",
    }
    assert admission_calls == [(prior, {"validation_jul03"})]
    assert control_fit_calls == [prior]
    assert control_calibration_calls == [{"validation_jul03"}]
    assert scored_calls == [
        ("admission", {"confirmation_jul14"}),
        ("control", {"confirmation_jul14"}),
    ]
    assert set(evaluation) == {
        BOUNDARY_CONTROL_CANDIDATE,
        ACCURACY_ANCHORED_ADMISSION_CANDIDATE,
    }
    for record in training.values():
        assert record["fit_blocks"] == [
            "initial_book_history",
            "validation_may26",
            "validation_jun02",
            "validation_jun09",
        ]
        assert record["calibration_block"] == "validation_jul03"
        assert record["evaluation_block"] == "confirmation_jul14"
    assert "outcome_predicted_up" not in evaluation[BOUNDARY_CONTROL_CANDIDATE].columns
    assert OUTCOME_SIGNAL_BLOCK_COLUMN not in evaluation[BOUNDARY_CONTROL_CANDIDATE].columns


def test_tail_comparison_with_missing_control_value_fails_closed() -> None:
    check = _check("tail comparison", -1.0, ">=", None)

    assert check["passed"] is False


def test_candidate_metrics_publish_dual_denominators_wilson_tails_and_calibration() -> None:
    count = 200
    correct = np.array([True] * 198 + [False] * 2)
    start = datetime(2026, 7, 14, tzinfo=UTC)
    price = np.full(count, 0.40)
    fee_rate = np.full(count, 0.10)
    fee = fee_rate * price * (1.0 - price)
    realized = correct.astype(np.float64) - price - fee
    frame = pl.DataFrame(
        {
            "market_id": [f"B5-{index}" for index in range(count)],
            "window_start": [start + timedelta(minutes=5 * index) for index in range(count)],
            "observed_at": [
                start + timedelta(minutes=5 * index, seconds=120) for index in range(count)
            ],
            "seconds_elapsed": [120] * count,
            "label_up": [1] * 198 + [0] * 2,
            "predicted_up": [1] * count,
            "probability_up": [0.99] * count,
            "confidence": [0.99] * count,
            "correct": correct,
            "outcome_probability_up": [0.99] * count,
            "outcome_predicted_up": [1] * count,
            "admission_probability_correct": [0.99] * 198 + [0.01] * 2,
            "economic_score": realized,
            "direct_net_edge_per_share": realized,
            "predicted_net_per_share": realized,
            "selected_ask_vwap_5": price,
            "selected_fee_per_share": fee,
            "realized_selected_net_per_share": realized,
            "fee_rate": fee_rate,
            "strict_both_side_eligible": [True] * count,
            "execution_evidence_available": [True] * count,
            "model_eligible": [True] * count,
            "policy_selected": [True] * count,
        }
    )

    result = _candidate_result(
        frame,
        book_qualified_markets=400,
        all_core_markets=1_000,
        complete_market_ids=frame["market_id"].to_list(),
    )
    economics = result["economics"]

    assert economics["book_qualified_coverage"] == pytest.approx(0.50)
    assert economics["all_core_market_coverage"] == pytest.approx(0.20)
    assert economics["accuracy"] == pytest.approx(0.99)
    assert economics["accuracy_wilson_lower_95"] == pytest.approx(0.9642782383)
    assert economics["execution"]["worst_realized_net_pnl"] == pytest.approx(-2.12)
    assert economics["execution"]["mean_worst_one_percent_realized_net_pnl"] == pytest.approx(-2.12)
    assert result["admission_probability_calibration"][
        "expected_calibration_error"
    ] == pytest.approx(0.01)
    assert result["outcome_probability_calibration"]["expected_calibration_error"] == pytest.approx(
        0.0
    )
    assert result["selected_net_calibration"]["absolute_bias"] == pytest.approx(0.0)


def test_prediction_artifacts_keep_only_compact_audit_columns() -> None:
    start = datetime(2026, 7, 14, tzinfo=UTC)
    source = pl.DataFrame(
        {
            "market_id": ["B5-0"],
            "window_start": [start],
            "observed_at": [start + timedelta(seconds=120)],
            "seconds_elapsed": [120],
            "label_up": [1],
            "walk_forward_block": ["B5"],
            "outcome_signal_block": ["B5"],
            "candidate": [ACCURACY_ANCHORED_ADMISSION_CANDIDATE],
            "predicted_up": [1],
            "probability_up": [0.90],
            "confidence": [0.90],
            "correct": [True],
            "outcome_probability_up": [0.90],
            "outcome_predicted_up": [1],
            "outcome_probability_selected": [0.90],
            "outcome_confidence_margin": [0.40],
            "outcome_direction_correct": [True],
            "admission_probability_correct": [0.90],
            "economic_score": [0.40],
            "direct_net_edge_per_share": [0.40],
            "predicted_net_per_share": [0.40],
            "selected_ask_vwap_5": [0.40],
            "selected_fee_per_share": [0.01],
            "realized_selected_net_per_share": [0.59],
            "fee_rate": [0.01],
            "up_ask_vwap_5": [0.40],
            "down_ask_vwap_5": [0.60],
            "strict_both_side_eligible": [True],
            "execution_evidence_available": [True],
            "model_eligible": [True],
            "policy_selected": [True],
            "btc_return_5s_bps": [12.0],
            "chainlink_deviation_bps": [3.0],
        }
    )

    outcome = _outcome_oof_artifact_frame(source)
    prediction = _prediction_artifact_frame(source)

    assert outcome.columns == [
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "walk_forward_block",
        "outcome_signal_block",
        "outcome_probability_up",
        "outcome_predicted_up",
        "correct",
        "outcome_probability_selected",
        "outcome_confidence_margin",
        "outcome_direction_correct",
    ]
    assert prediction.columns == [
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "walk_forward_block",
        "outcome_signal_block",
        "candidate",
        "predicted_up",
        "probability_up",
        "confidence",
        "correct",
        "outcome_probability_up",
        "outcome_predicted_up",
        "outcome_probability_selected",
        "admission_probability_correct",
        "economic_score",
        "direct_net_edge_per_share",
        "predicted_net_per_share",
        "selected_ask_vwap_5",
        "selected_fee_per_share",
        "realized_selected_net_per_share",
        "fee_rate",
        "up_ask_vwap_5",
        "down_ask_vwap_5",
        "strict_both_side_eligible",
        "execution_evidence_available",
        "model_eligible",
        "policy_selected",
    ]
    assert "btc_return_5s_bps" not in prediction.columns
    assert "chainlink_deviation_bps" not in prediction.columns


def test_report_surfaces_walk_forward_and_calibration_evidence() -> None:
    rows = [
        (0.20, 1, 1, 0.40),
        (0.20, 0, 0, 0.40),
        (0.05, 1, 0, 0.40),
        (0.05, 0, 1, 0.40),
    ]
    blocks = {name: _scored_block(name, rows) for name in ("B2", "B3", "B4")}
    operating_point = select_walk_forward_operating_point(
        blocks,
        thresholds=(0.0, 0.10),
        gates=_permissive_gates(),
        all_core_markets_by_block={name: 8 for name in blocks},
    )
    evaluation = mark_first_economic_action(
        _scored_block("B5", rows),
        threshold=0.10,
    )
    result = _candidate_result(
        evaluation,
        book_qualified_markets=4,
        all_core_markets=8,
        complete_market_ids=evaluation["market_id"].to_list(),
    )
    payload = {
        "objective": "test economic admission",
        "model_contract": {
            "candidate_names": [
                BOUNDARY_CONTROL_CANDIDATE,
                ACCURACY_ANCHORED_ADMISSION_CANDIDATE,
            ]
        },
        "candidates": {
            BOUNDARY_CONTROL_CANDIDATE: result,
            ACCURACY_ANCHORED_ADMISSION_CANDIDATE: result,
        },
        "selection": {
            "selected_candidate": ACCURACY_ANCHORED_ADMISSION_CANDIDATE,
            "candidates": {ACCURACY_ANCHORED_ADMISSION_CANDIDATE: {"checks": []}},
        },
        "operating_points": {
            BOUNDARY_CONTROL_CANDIDATE: {
                "threshold": 0.89,
                "qualified_on_threshold_blocks": None,
            },
            ACCURACY_ANCHORED_ADMISSION_CANDIDATE: operating_point,
        },
        "outcome_head_evaluation": {
            "markets": 8,
            "rows": 16,
            "accuracy": 0.90,
            "accuracy_wilson_lower_95": 0.80,
            "predicted_up_rate": 0.50,
            "calibration": {"expected_calibration_error": 0.05},
        },
        "availability": {
            "point_qualified_rows": 16,
            "point_qualified_markets": 8,
            "complete_decision_window_markets_diagnostic_only": 4,
            "blocks": {
                "B5": {
                    "book_qualified_markets": 4,
                    "all_core_oracle_markets": 8,
                }
            },
        },
        "evaluation": {"block": "B5"},
    }

    report = _render_report(payload)

    assert "Universal outcome head" in report
    assert "Selected threshold by blocked fold" in report
    assert "Evaluation calibration" in report
    assert "B2" in report and "B3" in report and "B4" in report
