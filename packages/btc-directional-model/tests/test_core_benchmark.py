from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from btc_directional_model.core_benchmark import (
    BenchmarkEvidence,
    CandidatePolicy,
    benchmark_predictions,
)


def prediction_frame(candidate: str, *, early: bool) -> pl.DataFrame:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    rows = []
    for market_index in range(12):
        label = market_index % 2
        window_start = start + timedelta(minutes=market_index * 5)
        for second in (60, 90, 120, 180, 240):
            if early:
                confidence = 0.9 if second >= 90 else 0.6
            else:
                confidence = (
                    0.9 if second >= 180 and market_index < 8 else 0.7
                )
            probability_up = confidence if label else 1.0 - confidence
            rows.append(
                {
                    "candidate": candidate,
                    "market_id": f"market-{market_index:02d}",
                    "observed_at": window_start + timedelta(seconds=second),
                    "seconds_elapsed": second,
                    "label_up": label,
                    "predicted_up": label,
                    "probability_up": probability_up,
                    "confidence": confidence,
                    "correct": True,
                    "up_executable": True,
                    "down_executable": True,
                    "up_ask_vwap_5": 0.50,
                    "down_ask_vwap_5": 0.50,
                    "direct_taker_fee_per_share": 0.01,
                }
            )
    return pl.DataFrame(rows)


def benchmark_result() -> dict:
    frames = {
        "control": prediction_frame("control", early=False),
        "early-core": prediction_frame("early-core", early=True),
    }
    return benchmark_predictions(
        frames,
        policies={
            "control": CandidatePolicy(0.8, True, 0.20, 1024),
            "early-core": CandidatePolicy(0.8, True, 0.18, 2048),
        },
        control_candidate="control",
        evidence=BenchmarkEvidence(
            label="Chronological development comparison",
            kind="development",
            independent=False,
        ),
        minimum_samples=8,
        minimum_executable_samples=8,
    )


def test_own_policy_metrics_include_timing_no_trade_and_five_share_economics() -> None:
    result = benchmark_result()
    control = result["candidates"]["control"]["own_policy"]
    early = result["candidates"]["early-core"]["own_policy"]

    assert control["markets"] == 8
    assert control["confidence_no_trade_markets"] == 4
    assert control["median_seconds_elapsed"] == 180
    assert early["markets"] == 12
    assert early["confidence_no_trade_markets"] == 0
    assert early["median_seconds_elapsed"] == 90
    assert result["candidates"]["early-core"]["time_bands"][1]["markets"] == 12
    assert early["execution"]["executable_coverage"] == 1
    assert early["execution"]["mean_direct_edge_per_share"] == pytest.approx(0.39)
    assert early["execution"]["realized_net_expectancy_per_trade"] == pytest.approx(
        2.45
    )
    assert early["execution"]["maximum_drawdown"] == 0


def test_common_comparison_uses_exact_fixed_timestamps() -> None:
    result = benchmark_result()
    comparison = result["common_comparisons"]["early-core"]

    assert [row["seconds_elapsed"] for row in comparison["checkpoints"]] == [
        60,
        90,
        120,
        180,
        240,
    ]
    assert all(row["common_markets"] == 12 for row in comparison["checkpoints"])
    assert all(row["accuracy_delta"] == 0 for row in comparison["checkpoints"])


def test_advance_gates_pass_benchmark_but_not_deployment_on_development_data() -> None:
    result = benchmark_result()
    advance = result["candidates"]["early-core"]["advance"]

    assert advance["benchmark_passed"] is True
    assert advance["deployment_qualified"] is False
    assert result["benchmark_passed_candidates"] == ["early-core"]
    assert result["deployment_qualified_candidates"] == []


def test_prediction_contract_rejects_inconsistent_correct_flag() -> None:
    invalid = prediction_frame("control", early=False).with_columns(
        pl.when(pl.col("market_id") == "market-00")
        .then(False)
        .otherwise(pl.col("correct"))
        .alias("correct")
    )

    with pytest.raises(ValueError, match="correct is inconsistent"):
        benchmark_predictions(
            {"control": invalid},
            policies={"control": CandidatePolicy(0.8, True)},
            control_candidate="control",
            evidence=BenchmarkEvidence(
                label="Development",
                kind="development",
                independent=False,
            ),
        )


def test_execution_gate_fails_when_fee_evidence_is_absent() -> None:
    frames = {
        "control": prediction_frame("control", early=False),
        "early-core": prediction_frame("early-core", early=True).drop(
            "direct_taker_fee_per_share"
        ),
    }
    result = benchmark_predictions(
        frames,
        policies={
            "control": CandidatePolicy(0.8, True, 0.20, 1024),
            "early-core": CandidatePolicy(0.8, True, 0.18, 2048),
        },
        control_candidate="control",
        evidence=BenchmarkEvidence(
            label="Development",
            kind="development",
            independent=False,
        ),
        minimum_samples=8,
        minimum_executable_samples=8,
    )

    advance = result["candidates"]["early-core"]["advance"]
    assert advance["benchmark_passed"] is False
    assert (
        result["candidates"]["early-core"]["own_policy"]["execution"][
            "realized_net_expectancy_per_trade"
        ]
        is None
    )


def test_missing_native_performance_evidence_blocks_candidate() -> None:
    frames = {
        "control": prediction_frame("control", early=False),
        "early-core": prediction_frame("early-core", early=True),
    }
    result = benchmark_predictions(
        frames,
        policies={
            "control": CandidatePolicy(0.8, True, 0.20, 1024),
            "early-core": CandidatePolicy(0.8, True),
        },
        control_candidate="control",
        evidence=BenchmarkEvidence(
            label="Independent chronological holdout",
            kind="holdout",
            independent=True,
        ),
        minimum_samples=8,
        minimum_executable_samples=8,
    )

    advance = result["candidates"]["early-core"]["advance"]
    checks = {check["name"]: check for check in advance["checks"]}
    assert checks["native inference p99 is within budget"]["passed"] is False
    assert checks["runtime model size is within budget"]["passed"] is False
    assert advance["deployment_qualified"] is False
