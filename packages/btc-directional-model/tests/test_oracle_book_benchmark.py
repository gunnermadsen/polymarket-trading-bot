from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_execution import EXECUTION_CONTEXT_SECONDS
from btc_directional_model.core_features import (
    CORE_MATURE_REVERSAL_ORACLE_FEATURES,
)
from btc_directional_model.offline_challengers import STRICT_BOOK_V2_FEATURES
from btc_directional_model.oracle_book_benchmark import (
    EXPECTED_RECENCY_HALF_LIFE_DAYS,
    FIXED_DECISION_SECONDS,
    TIMING_DECISION_SECONDS,
    _complete_context_market_ids,
    _evaluation_payload,
    _frame_sha256,
    _selection_payload,
    _split_experiment_frame,
    _validate_strict_execution_contract,
    book_candidate_spec,
    control_candidate_spec,
    load_oracle_book_benchmark_config,
)


def repository_config() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "btc-5m-directional-oracle-book-matched-20260526-20260721.toml"
    )


def strict_execution_row(
    market_id: str,
    window_start: datetime,
    second: int,
    *,
    strict: bool = True,
    provider_age_seconds: float = 0.1,
) -> dict[str, object]:
    observed_at = window_start + timedelta(seconds=second)
    provider_at = observed_at - timedelta(seconds=provider_age_seconds)
    return {
        "market_id": market_id,
        "window_start": window_start,
        "observed_at": observed_at,
        "seconds_elapsed": second,
        "fee_rate": 0.20,
        "quality_flags": 0,
        "up_provider_received_at": provider_at,
        "up_best_bid": 0.40,
        "up_best_ask": 0.42,
        "up_best_bid_size": 10.0,
        "up_best_ask_size": 10.0,
        "up_bid_depth": 100.0,
        "up_ask_depth": 100.0,
        "up_ask_vwap_5": 0.43,
        "up_ask_vwap_10": 0.44,
        "up_imbalance": 0.10,
        "down_provider_received_at": provider_at,
        "down_best_bid": 0.56,
        "down_best_ask": 0.58,
        "down_best_bid_size": 10.0,
        "down_best_ask_size": 10.0,
        "down_bid_depth": 100.0,
        "down_ask_depth": 100.0,
        "down_ask_vwap_5": 0.59,
        "down_ask_vwap_10": 0.60,
        "down_imbalance": -0.10,
        "up_side_fresh": strict,
        "down_side_fresh": strict,
        "strict_both_side_eligible": strict,
        "strict_both_side_eligible_10": strict,
    }


def test_repository_config_freezes_exact_matched_contract() -> None:
    config = load_oracle_book_benchmark_config(repository_config())

    assert config.model.recency_half_life_days == 28.0
    assert config.core_config_sha256 == (
        "ec5262959ec33d15047943c7242d4e05ad43665f40313e304975c6644127b294"
    )
    assert config.split.fit_start == datetime(2026, 5, 26, tzinfo=UTC)
    assert config.split.evaluation_end == datetime(2026, 7, 21, tzinfo=UTC)
    assert FIXED_DECISION_SECONDS == (120, 125)
    assert TIMING_DECISION_SECONDS == (120, 125, 130, 135, 140)


def test_candidate_specs_share_82_oracle_features_and_recency() -> None:
    control = control_candidate_spec(
        "control",
        EXPECTED_RECENCY_HALF_LIFE_DAYS,
    )
    challenger = book_candidate_spec(
        "challenger",
        EXPECTED_RECENCY_HALF_LIFE_DAYS,
    )

    assert len(control.feature_names) == 82
    assert control.feature_names == tuple(CORE_MATURE_REVERSAL_ORACLE_FEATURES)
    assert challenger.feature_names[:82] == control.feature_names
    assert challenger.feature_names[82:] == tuple(STRICT_BOOK_V2_FEATURES)
    assert control.recency_half_life_days == 28.0
    assert challenger.recency_half_life_days == 28.0
    assert control.row_weight_schedule == challenger.row_weight_schedule


def test_complete_context_requires_all_11_exact_strict_points() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    rows = [
        strict_execution_row("complete", start, second)
        for second in EXECUTION_CONTEXT_SECONDS
    ]
    rows.extend(
        strict_execution_row("missing", start + timedelta(minutes=5), second)
        for second in EXECUTION_CONTEXT_SECONDS
        if second != 115
    )
    rows.extend(
        strict_execution_row(
            "invalid",
            start + timedelta(minutes=10),
            second,
            strict=second != 120,
        )
        for second in EXECUTION_CONTEXT_SECONDS
    )

    assert _complete_context_market_ids(pl.DataFrame(rows)) == ["complete"]


def test_strict_contract_fails_closed_on_stale_claimed_row() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    valid = pl.DataFrame(
        [strict_execution_row("valid", start, 120)]
    )
    _validate_strict_execution_contract(valid)

    stale = pl.DataFrame(
        [
            strict_execution_row(
                "stale",
                start,
                120,
                provider_age_seconds=2.1,
            )
        ]
    )
    with pytest.raises(RuntimeError, match="freshness"):
        _validate_strict_execution_contract(stale)


def test_fixed_cohorts_preserve_identical_one_row_markets() -> None:
    config = load_oracle_book_benchmark_config(repository_config())
    ranges = (
        config.split.fit_start,
        config.split.calibration_start,
        config.split.threshold_start,
        config.split.evaluation_start,
    )
    rows = []
    for range_index, range_start in enumerate(ranges):
        for label in (0, 1):
            market_start = range_start + timedelta(minutes=5 * label)
            rows.append(
                {
                    "market_id": f"{range_index}-{label}",
                    "window_start": market_start,
                    "observed_at": market_start + timedelta(seconds=120),
                    "seconds_elapsed": 120,
                    "label_up": label,
                }
            )

    cohorts = _split_experiment_frame(
        pl.DataFrame(rows),
        (120,),
        config.split,
    )

    assert set(cohorts) == {
        "fit",
        "calibration",
        "threshold",
        "evaluation",
    }
    assert all(frame.height == 2 for frame in cohorts.values())
    assert all(frame["market_id"].n_unique() == 2 for frame in cohorts.values())


def test_row_hash_is_stable_under_input_order() -> None:
    start = datetime(2026, 7, 14, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "market_id": ["b", "a"],
            "observed_at": [
                start + timedelta(minutes=5, seconds=120),
                start + timedelta(seconds=120),
            ],
            "seconds_elapsed": [120, 120],
            "label_up": [1, 0],
            "feature": [2.0, 1.0],
        }
    )

    assert _frame_sha256(
        frame,
        ["market_id", "observed_at", "seconds_elapsed", "label_up", "feature"],
    ) == _frame_sha256(
        frame.reverse(),
        ["market_id", "observed_at", "seconds_elapsed", "label_up", "feature"],
    )


def test_evaluation_reports_five_and_ten_share_economics() -> None:
    start = datetime(2026, 7, 14, tzinfo=UTC)
    scored = pl.DataFrame(
        {
            "market_id": ["up", "down"],
            "window_start": [start, start + timedelta(minutes=5)],
            "observed_at": [
                start + timedelta(seconds=120),
                start + timedelta(minutes=5, seconds=120),
            ],
            "seconds_elapsed": [120, 120],
            "label_up": [1, 0],
            "predicted_up": [1, 0],
            "probability_up": [0.9, 0.1],
            "confidence": [0.9, 0.9],
            "correct": [True, True],
            "baseline_correct": [True, True],
            "fee_rate": [0.20, 0.20],
            "up_ask_vwap_5": [0.43, 0.43],
            "down_ask_vwap_5": [0.59, 0.59],
            "up_ask_vwap_10": [0.44, 0.44],
            "down_ask_vwap_10": [0.60, 0.60],
            "strict_both_side_eligible": [True, True],
            "strict_both_side_eligible_10": [True, True],
            "execution_evidence_available": [True, True],
        }
    )

    result = _evaluation_payload(
        scored,
        scored,
        expected_seconds=(120,),
        eligible_markets=2,
        universe_markets=4,
    )

    five = result["execution_by_size"]["vwap5_five_share"]
    ten = result["execution_by_size"]["vwap10_ten_share"]
    assert five["quantity"] == 5.0
    assert five["vwap_depth"] == 5
    assert ten["quantity"] == 10.0
    assert ten["vwap_depth"] == 10
    assert five["economic_markets"] == 2
    assert ten["economic_markets"] == 2
    assert result["book_availability_coverage"] == 0.5
    assert result["hard_confident_errors"]["hard_confident_error_markets"] == 0


def test_selection_requires_accuracy_coverage_tail_and_positive_economics(
) -> None:
    config = load_oracle_book_benchmark_config(repository_config())
    core = load_core_config(config.core_config)
    control_policy = {
        "coverage": 0.61,
        "accuracy": 0.88,
    }
    challenger_policy = {
        "coverage": 0.65,
        "accuracy": 0.90,
        "balanced_accuracy": 0.90,
        "up_recall": 0.90,
        "down_recall": 0.90,
        "wilson_lower_95": 0.88,
        "expected_calibration_error": 0.02,
    }
    control_tail = {
        "hard_confident_error_markets": 2,
        "hard_confident_error_rate_selected": 0.01,
    }
    challenger_tail = {
        "hard_confident_error_markets": 1,
        "hard_confident_error_rate_selected": 0.005,
    }
    experiment = {
        "arms": {
            "core_oracle": {
                "training": {"threshold_qualified": True},
                "evaluation": {
                    "policy": control_policy,
                    "hard_confident_errors": control_tail,
                    "execution_by_size": {},
                },
            },
            "core_oracle_book": {
                "training": {"threshold_qualified": True},
                "evaluation": {
                    "policy": challenger_policy,
                    "hard_confident_errors": challenger_tail,
                    "execution_by_size": {
                        "vwap10_ten_share": {
                            "realized_net_expectancy_per_trade": 0.10,
                            "realized_net_pnl_total": 10.0,
                        }
                    },
                },
            },
        },
        "paired_evaluation": {
            "checkpoints": {"125": {"accuracy_delta": 0.02}}
        },
    }

    selection = _selection_payload({"fixed_125s": experiment}, core)
    assert selection["selected_candidate"] == "fixed_125s"

    insufficient_coverage = copy.deepcopy(experiment)
    insufficient_coverage["arms"]["core_oracle_book"]["evaluation"][
        "policy"
    ]["coverage"] = 0.59
    selection = _selection_payload(
        {"fixed_125s": insufficient_coverage},
        core,
    )
    assert selection["selected_candidate"] is None
