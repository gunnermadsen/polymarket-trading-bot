from __future__ import annotations

import copy
import os
import resource
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model.loss_tail_benchmark import (
    EXPECTED_PROPOSAL_KEY_SHA256,
    EXPECTED_PROPOSAL_ROWS,
    EXPECTED_STRICT_KEY_SHA256,
    EXPECTED_STRICT_ROWS,
    MATCHED_BOUNDARY_CONTROL,
    _apply_worker_resources,
    _fold_frames,
    _prediction_artifact_frame,
    _qualification_checks,
    _restore_environment,
    _score_boundary_proposals,
    _select_candidate,
    _set_worker_thread_environment,
    _start_max_rss_watchdog,
    _validate_matched_prediction_frames,
)
from btc_directional_model.loss_tail_config import load_loss_tail_benchmark_config


def test_direct_loss_tail_cohort_is_frozen() -> None:
    assert EXPECTED_STRICT_ROWS == 264_121
    assert (
        EXPECTED_STRICT_KEY_SHA256
        == "2e4a6481d822be3491f4d574ee0ec8d11ec7d5dd8139979a54dc581a4daf312e"
    )
    assert EXPECTED_PROPOSAL_ROWS == 5_396
    assert (
        EXPECTED_PROPOSAL_KEY_SHA256
        == "d7cc08f28298583d66daeb9d07913466f4ec2ec72ff43e4b24dbde29de6fbc67"
    )


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
            "walk_forward_block": ["confirmation_jul14"] * len(seconds),
            "label_up": labels,
            "binance_sign_up": [1, 1, 1, 0, 0],
            "oof_boundary_probability_up": [0.20, 0.80, 0.25, 0.65, 0.30],
            "boundary_proposal_probability_up": [0.20, 0.80, 0.25, 0.65, 0.30],
            "boundary_proposal_confidence": [0.80, 0.80, 0.75, 0.65, 0.70],
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
            "boundary_direction": [0, 1, 0, 1, 0],
            "boundary_direction_correct": [0, 1, 0, 0, 1],
            "fee_inclusive_debit": np.where(
                np.asarray([0, 1, 0, 1, 0], dtype=bool), up_debit, down_debit
            ),
            "realized_trade_return": np.where(
                np.asarray([0, 1, 0, 1, 0], dtype=bool),
                np.asarray(labels) - up_debit,
                1.0 - np.asarray(labels) - down_debit,
            ),
            "no_trade_return": [0.0] * len(seconds),
        }
    )


def _benchmark_result(*, candidate: bool) -> dict[str, object]:
    if candidate:
        accuracy = 0.92
        wilson = 0.89
        profit_factor = 1.50
        pnl_per_market = 0.03
        total_pnl = 300.0
        worst_trade = -4.0
        worst_one_percent = -3.0
        drawdown = 8.0
        coverage = 0.40
        mean_loss_win = 6.0
        gross_loss_profit = 0.70
        fold_loss_ratios = (0.70, 0.70, 0.80)
    else:
        accuracy = 0.92
        wilson = 0.90
        profit_factor = 1.265
        pnl_per_market = 0.02044
        total_pnl = 283.15
        worst_trade = -5.0
        worst_one_percent = -4.0
        drawdown = 10.0
        coverage = 0.5301
        mean_loss_win = 7.05
        gross_loss_profit = 0.791
        fold_loss_ratios = (0.79, 0.79, 0.79)

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
            "mean_selected_ask_vwap_5": 0.86 if candidate else 0.8697,
            "book_qualified_coverage": coverage,
            "net_pnl_per_all_core_market": pnl_per_market,
            "trades": 660,
            "execution": {
                "gross_profit": 100.0,
                "gross_loss": 100.0 * gross_loss_profit,
                "profit_factor": profit_factor,
                "realized_net_pnl_total": total_pnl,
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
        "proposal_outcomes": {
            "gross_loss_removed": 0.31 if candidate else 0.0,
            "gross_profit_retained": 0.71 if candidate else 1.0,
            "catastrophic_loss_recall": 0.31 if candidate else 0.0,
        },
        "proposal_price_bands": {
            "0_90_and_over": {"realized_net_pnl_total": 1.0},
        },
        "folds": folds,
    }


def test_prediction_artifact_retains_outcome_direction() -> None:
    frame = (
        _scoring_frame()
        .sort(["market_id", "seconds_elapsed"])
        .group_by("market_id", maintain_order=True)
        .first()
    )
    scored = _score_boundary_proposals(
        frame,
        np.array([0.99, 0.99]),
        candidate="residual",
        fold_name="evaluation",
        economic_action=True,
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


def test_residual_action_is_exact_positive_expected_net_with_boundary_locked() -> None:
    frame = _scoring_frame().filter(pl.col("seconds_elapsed") == 120).sort("market_id")
    debit = frame["fee_inclusive_debit"].to_numpy()
    probability = np.array([debit[0] + 0.01, debit[1]])

    candidate = _score_boundary_proposals(
        frame,
        probability,
        candidate="residual",
        fold_name="evaluation",
        economic_action=True,
    )
    control = _score_boundary_proposals(
        frame,
        probability,
        candidate=MATCHED_BOUNDARY_CONTROL,
        fold_name="evaluation",
        economic_action=False,
    )

    assert candidate["predicted_net_per_share"].to_list() == pytest.approx([0.01, 0.0])
    assert candidate["policy_selected"].to_list() == [True, False]
    assert control["policy_selected"].to_list() == [True, True]
    assert candidate["predicted_up"].to_list() == frame["boundary_direction"].to_list()


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
    weak_loss["loss_ratios"]["gross_loss_profit"] = 0.713  # type: ignore[index]
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


def test_fold_frames_are_one_proposal_per_market_and_strictly_chronological() -> None:
    config = load_loss_tail_benchmark_config(CONFIG_PATH)
    rows: list[dict[str, object]] = []
    for block_index, block in enumerate(config.walk_forward.blocks):
        for label in (0, 1):
            rows.append(
                {
                    "market_id": f"{block.name}-{label}",
                    "window_start": block.start + timedelta(minutes=5 * (label + 1)),
                    "seconds_elapsed": 120,
                    "walk_forward_block": block.name,
                    "label_up": label,
                    "boundary_direction_correct": label,
                    "block_index": block_index,
                }
            )
    frame = pl.DataFrame(rows)
    fold = config.walk_forward.folds[0]

    fit, calibration, evaluation = _fold_frames(frame, fold)

    assert fit["market_id"].n_unique() == fit.height
    assert fit["window_start"].max() < calibration["window_start"].min()
    assert calibration["window_start"].max() < evaluation["window_start"].min()

    leaked = replace(fold, fit_block_names=(*fold.fit_block_names, fold.evaluation_block_name))
    with pytest.raises(RuntimeError, match="chronology leaked"):
        _fold_frames(frame, leaked)


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
    assert applied["memory_enforcement"] == {"mode": "address_space_rlimit"}

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


def test_worker_resource_falls_back_to_fail_hard_rss_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_loss_tail_benchmark_config(CONFIG_PATH)
    memory_bytes = 8 * 1024**3
    watchdog_calls: list[int] = []

    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (resource.RLIM_INFINITY,) * 2)

    def unavailable(_kind: int, _limits: tuple[int, int]) -> None:
        raise ValueError("host does not support lowering this limit")

    monkeypatch.setattr(resource, "setrlimit", unavailable)
    monkeypatch.setattr(
        "btc_directional_model.loss_tail_benchmark._start_max_rss_watchdog",
        lambda limit: watchdog_calls.append(limit) or {"mode": "maximum_rss_watchdog"},
    )

    applied = _apply_worker_resources(config)

    assert watchdog_calls == [memory_bytes]
    assert applied["memory_enforcement"] == {"mode": "maximum_rss_watchdog"}


def test_max_rss_watchdog_rejects_a_worker_already_over_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "btc_directional_model.loss_tail_benchmark._maximum_resident_set_bytes",
        lambda: 9 * 1024**3,
    )
    with pytest.raises(RuntimeError, match="already exceeds"):
        _start_max_rss_watchdog(8 * 1024**3)
