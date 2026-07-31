from __future__ import annotations

import copy
import os
import resource
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model.loss_tail_benchmark import (
    MATCHED_BOUNDARY_CONTROL,
    _apply_worker_resources,
    _prediction_artifact_frame,
    _qualification_checks,
    _restore_environment,
    _score_boundary_correctness,
    _score_direct,
    _select_candidate,
    _set_worker_thread_environment,
    _validate_matched_prediction_frames,
)
from btc_directional_model.loss_tail_config import load_loss_tail_benchmark_config

CONFIG_PATH = (
    Path(__file__).parents[1] / "configs" / "btc-5m-directional-loss-tail-20260413-20260729.toml"
)


def _scoring_frame() -> pl.DataFrame:
    start = datetime(2026, 7, 14, tzinfo=UTC)
    seconds = [130, 120, 125, 120, 125]
    market_ids = ["market-a", "market-a", "market-a", "market-b", "market-b"]
    labels = [1, 1, 1, 0, 0]
    up_price = np.array([0.45, 0.43, 0.44, 0.55, 0.57])
    down_price = 1.0 - up_price
    fee_rate = np.full(len(seconds), 0.02)
    up_fee = fee_rate * up_price * (1.0 - up_price)
    down_fee = fee_rate * down_price * (1.0 - down_price)
    up_debit = up_price + up_fee
    down_debit = down_price + down_fee
    observed = [start + timedelta(seconds=value) for value in seconds]
    return pl.DataFrame(
        {
            "market_id": market_ids,
            "window_start": [start] * 3 + [start + timedelta(minutes=5)] * 2,
            "observed_at": observed[:3]
            + [start + timedelta(minutes=5, seconds=value) for value in seconds[3:]],
            "seconds_elapsed": seconds,
            "label_up": labels,
            "binance_sign_up": [1, 1, 1, 0, 0],
            "oof_boundary_probability_up": [0.20, 0.80, 0.25, 0.65, 0.30],
            "fee_rate": fee_rate,
            "up_ask_vwap_5": up_price,
            "down_ask_vwap_5": down_price,
            "up_ask_vwap_10": up_price + 0.01,
            "down_ask_vwap_10": down_price + 0.01,
            "up_fee_per_share": up_fee,
            "down_fee_per_share": down_fee,
            "up_entry_debit_per_share": up_debit,
            "down_entry_debit_per_share": down_debit,
            "realized_up_net_per_share": np.asarray(labels) - up_debit,
            "realized_down_net_per_share": 1.0 - np.asarray(labels) - down_debit,
            "strict_both_side_eligible": [True] * len(seconds),
            "strict_both_side_eligible_10": [True] * len(seconds),
            "execution_evidence_available": [True] * len(seconds),
        }
    )


def _benchmark_result(*, candidate: bool) -> dict[str, object]:
    if candidate:
        accuracy = 0.92
        wilson = 0.89
        profit_factor = 3.0
        pnl_per_market = 0.20
        worst_trade = -4.0
        worst_one_percent = -3.0
        drawdown = 8.0
        coverage = 0.75
        mean_loss_win = 0.72
        gross_loss_profit = 0.36
        fold_loss_ratios = (0.35, 0.35, 0.40)
    else:
        accuracy = 0.92
        wilson = 0.90
        profit_factor = 2.5
        pnl_per_market = 0.15
        worst_trade = -5.0
        worst_one_percent = -4.0
        drawdown = 10.0
        coverage = 0.90
        mean_loss_win = 0.80
        gross_loss_profit = 0.40
        fold_loss_ratios = (0.40, 0.40, 0.40)

    config = load_loss_tail_benchmark_config(CONFIG_PATH)
    folds = {
        fold.name: {
            "economics": {
                "trades": 220,
                "net_pnl_per_all_core_market": pnl_per_market,
                "execution": {"realized_net_expectancy_per_trade": 0.01},
            },
            "loss_ratios": {"gross_loss_profit": fold_loss_ratios[index]},
            "per_side": {"up": {"trades": 110}, "down": {"trades": 110}},
        }
        for index, fold in enumerate(config.walk_forward.folds)
    }
    return {
        "economics": {
            "accuracy": accuracy,
            "accuracy_wilson_lower_95": wilson,
            "wins_to_recover_average_loss": mean_loss_win,
            "book_qualified_coverage": coverage,
            "net_pnl_per_all_core_market": pnl_per_market,
            "trades": 660,
            "execution": {
                "gross_profit": 100.0,
                "gross_loss": 100.0 * gross_loss_profit,
                "profit_factor": profit_factor,
                "worst_realized_net_pnl": worst_trade,
                "mean_worst_one_percent_realized_net_pnl": worst_one_percent,
                "maximum_drawdown": drawdown,
                "realized_net_expectancy_per_trade": 0.01,
            },
        },
        "same_row_direction": {
            "accuracy": 0.92,
            "balanced_accuracy": 0.92,
            "per_true_side": {
                "up": {"accuracy": 0.92},
                "down": {"accuracy": 0.92},
            },
        },
        "decision_calibration": {"expected_calibration_error": 0.02},
        "loss_ratios": {
            "mean_loss_win": mean_loss_win,
            "gross_loss_profit": gross_loss_profit,
        },
        "folds": folds,
    }


def test_prediction_artifact_retains_outcome_direction() -> None:
    frame = _scoring_frame()
    scored = _score_direct(
        frame,
        np.array([0.99, 0.60, 0.91, 0.40, 0.10]),
        candidate="direct",
        fold_name="evaluation",
        confidence_threshold=0.89,
    )

    artifact = _prediction_artifact_frame(scored)

    assert "outcome_predicted_up" in artifact.columns
    assert artifact["outcome_predicted_up"].to_list() == artifact["predicted_up"].to_list()

    _validate_matched_prediction_frames(
        {
            MATCHED_BOUNDARY_CONTROL: artifact,
            "candidate": artifact.with_columns(pl.lit("candidate").alias("candidate")),
        }
    )
    with pytest.raises(RuntimeError, match="evaluation keys and labels"):
        _validate_matched_prediction_frames(
            {
                MATCHED_BOUNDARY_CONTROL: artifact,
                "candidate": artifact.with_columns((1 - pl.col("label_up")).alias("label_up")),
            }
        )


def test_direct_and_correctness_policies_select_first_confident_point_per_market() -> None:
    frame = _scoring_frame()
    confidence = np.array([0.99, 0.60, 0.91, 0.60, 0.80])

    direct = _score_direct(
        frame,
        confidence,
        candidate="direct",
        fold_name="evaluation",
        confidence_threshold=0.89,
    )
    correctness = _score_boundary_correctness(
        frame,
        confidence,
        candidate="correctness",
        fold_name="evaluation",
        confidence_threshold=0.89,
    )

    for scored in (direct, correctness):
        selected = scored.filter(pl.col("policy_selected"))
        assert selected.height == 1
        assert selected.select("market_id", "seconds_elapsed").row(0) == ("market-a", 125)
        assert not scored.filter(pl.col("market_id") == "market-b")["policy_selected"].any()

    # Correctness confidence only controls admission. Direction remains locked to
    # the independently generated boundary probability.
    expected_boundary_direction = (correctness["oof_boundary_probability_up"] >= 0.5).cast(pl.Int8)
    assert correctness["predicted_up"].to_list() == expected_boundary_direction.to_list()


def test_qualification_enforces_accuracy_and_loss_gates_without_fallback() -> None:
    config = load_loss_tail_benchmark_config(CONFIG_PATH)
    control = _benchmark_result(candidate=False)
    passing = _benchmark_result(candidate=True)

    passing_checks = {
        check["name"]: check
        for check in _qualification_checks(passing, control, config)  # type: ignore[arg-type]
    }
    assert all(check["passed"] for check in passing_checks.values())

    zero_expectancy = copy.deepcopy(passing)
    for fold in zero_expectancy["folds"].values():  # type: ignore[union-attr]
        fold["economics"]["execution"]["realized_net_expectancy_per_trade"] = 0.0
    zero_checks = {
        check["name"]: check
        for check in _qualification_checks(zero_expectancy, control, config)  # type: ignore[arg-type]
    }
    assert zero_checks["nonnegative expectancy folds"]["passed"]

    low_accuracy = copy.deepcopy(passing)
    low_accuracy["economics"]["accuracy"] = 0.88  # type: ignore[index]
    low_accuracy_checks = {
        check["name"]: check
        for check in _qualification_checks(low_accuracy, control, config)  # type: ignore[arg-type]
    }
    assert not low_accuracy_checks["selected accuracy"]["passed"]

    weak_loss = copy.deepcopy(passing)
    weak_loss["loss_ratios"]["gross_loss_profit"] = 0.37  # type: ignore[index]
    weak_loss_checks = {
        check["name"]: check
        for check in _qualification_checks(weak_loss, control, config)  # type: ignore[arg-type]
    }
    assert not weak_loss_checks["gross-loss/profit ratio improvement"]["passed"]

    failing_candidates = {
        MATCHED_BOUNDARY_CONTROL: control,
        **{name: copy.deepcopy(low_accuracy) for name in config.candidate_names},
    }
    selection = _select_candidate(failing_candidates, config)  # type: ignore[arg-type]
    assert selection["selected_candidate"] is None
    assert selection["qualified_candidates"] == []
    assert selection["fallback_allowed"] is False


def test_worker_resource_contract_without_lowering_process_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_loss_tail_benchmark_config(CONFIG_PATH)
    calls: list[tuple[int, tuple[int, int]]] = []
    infinity = resource.RLIM_INFINITY

    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (infinity, infinity))
    monkeypatch.setattr(resource, "setrlimit", lambda kind, limits: calls.append((kind, limits)))

    applied = _apply_worker_resources(config)
    memory_bytes = 8 * 1024**3
    supported_limits = [
        getattr(resource, name)
        for name in ("RLIMIT_AS", "RLIMIT_RSS")
        if getattr(resource, name, None) is not None
    ]
    assert applied["threads"] == 3
    assert applied["memory_limit_bytes"] == memory_bytes
    assert calls == [(kind, (memory_bytes, infinity)) for kind in supported_limits]

    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "POLARS_MAX_THREADS",
    )
    monkeypatch.setenv(thread_variables[0], "17")
    for name in thread_variables[1:]:
        monkeypatch.delenv(name, raising=False)
    previous = _set_worker_thread_environment(config.resources.threads_per_worker)
    assert all(os.environ[name] == "3" for name in thread_variables)
    _restore_environment(previous)
    assert os.environ[thread_variables[0]] == "17"
    assert all(name not in os.environ for name in thread_variables[1:])
