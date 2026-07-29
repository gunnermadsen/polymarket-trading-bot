from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from btc_directional_model.core_benchmark import (
    BenchmarkEvidence,
    CandidatePolicy,
    benchmark_predictions,
    select_chronological_policy_rows,
    select_explicit_policy_rows,
    select_time_band_policy_rows,
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
    assert early["execution"]["executable_coverage_all_selected"] == 1
    assert early["execution"]["executable_coverage_within_evidence"] == 1
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


def test_checkpoint_coverage_uses_the_universal_market_denominator() -> None:
    control = prediction_frame("control", early=False)
    candidate = prediction_frame("candidate", early=True).filter(
        ~pl.col("market_id").is_in(["market-10", "market-11"])
    )

    result = benchmark_predictions(
        {"control": control, "candidate": candidate},
        policies={
            "control": CandidatePolicy(0.8, True, 0.20, 1024),
            "candidate": CandidatePolicy(0.8, True, 0.18, 2048),
        },
        control_candidate="control",
        evidence=BenchmarkEvidence(
            label="Chronological development comparison",
            kind="development",
            independent=False,
        ),
        eligible_market_ids=[f"market-{index:02d}" for index in range(12)],
        minimum_samples=8,
        minimum_executable_samples=8,
    )

    checkpoint = result["candidates"]["candidate"]["checkpoints"][0]
    assert checkpoint["markets"] == 10
    assert checkpoint["eligible_markets"] == 12
    assert checkpoint["coverage"] == pytest.approx(10 / 12)


def test_advance_gates_pass_benchmark_but_not_deployment_on_development_data() -> None:
    result = benchmark_result()
    advance = result["candidates"]["early-core"]["advance"]
    checks = {check["name"]: check for check in advance["checks"]}

    assert advance["benchmark_passed"] is True
    assert advance["deployment_qualified"] is False
    assert checks["minimum realized net per share"]["observed"] == pytest.approx(0.49)
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


def test_chronological_policy_uses_preselected_first_crossing_and_full_scores() -> None:
    frame = prediction_frame("control", early=True).with_columns(
        pl.when(pl.col("market_id").is_in(["market-00", "market-01"]))
        .then(0.91)
        .otherwise(0.87)
        .alias("selected_confidence_threshold")
    )
    first_crossings = (
        frame.filter(
            pl.col("confidence") >= pl.col("selected_confidence_threshold")
        )
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .group_by("market_id", maintain_order=True)
        .first()
        .select("market_id", "observed_at", "seconds_elapsed")
        .with_columns(pl.lit(True).alias("policy_selected"))
    )
    frame = frame.join(
        first_crossings,
        on=["market_id", "observed_at", "seconds_elapsed"],
        how="left",
    ).with_columns(pl.col("policy_selected").fill_null(False))

    selected = select_chronological_policy_rows(frame)
    result = benchmark_predictions(
        {"control": frame},
        policies={
            "control": CandidatePolicy(
                confidence_threshold=None,
                deployment_compatible=True,
                selection_mode="chronological_preselected",
                confidence_threshold_min=0.87,
                confidence_threshold_max=0.91,
            )
        },
        control_candidate="control",
        evidence=BenchmarkEvidence(
            label="Chronological development",
            kind="development",
            independent=False,
        ),
        minimum_samples=1,
    )

    assert selected.height == 10
    assert result["candidates"]["control"]["own_policy"]["markets"] == 10
    assert result["candidates"]["control"]["own_policy"][
        "confidence_no_trade_markets"
    ] == 2
    assert result["candidates"]["control"]["checkpoints"][1][
        "seconds_elapsed"
    ] == 90


def test_chronological_policy_rejects_a_marker_that_is_not_first_crossing() -> None:
    frame = (
        prediction_frame("control", early=True)
        .filter(pl.col("market_id") == "market-00")
        .with_columns(
            pl.lit(0.80).alias("selected_confidence_threshold"),
            (pl.col("seconds_elapsed") == 120).alias("policy_selected"),
        )
    )

    with pytest.raises(ValueError, match="does not match"):
        select_chronological_policy_rows(frame)


def test_explicit_policy_accepts_one_control_priority_decision_per_market() -> None:
    frame = (
        prediction_frame("control", early=True)
        .filter(pl.col("seconds_elapsed") == 90)
        .with_columns(
            pl.lit(True).alias("model_eligible"),
            pl.lit(True).alias("policy_selected"),
        )
    )

    selected = select_explicit_policy_rows(frame)
    result = benchmark_predictions(
        {"control": frame},
        policies={
            "control": CandidatePolicy(
                confidence_threshold=None,
                deployment_compatible=False,
                selection_mode="explicit_preselected",
            )
        },
        control_candidate="control",
        evidence=BenchmarkEvidence(
            label="Explicit residual decisions",
            kind="development",
            independent=False,
        ),
        minimum_samples=1,
    )

    assert selected.height == 12
    assert result["candidates"]["control"]["own_policy"]["markets"] == 12
    assert result["candidates"]["control"]["own_policy"][
        "median_seconds_elapsed"
    ] == 90


def test_explicit_policy_rejects_duplicate_or_ineligible_decisions() -> None:
    duplicate = (
        prediction_frame("control", early=True)
        .filter(pl.col("market_id") == "market-00")
        .with_columns(
            pl.lit(True).alias("model_eligible"),
            pl.lit(True).alias("policy_selected"),
        )
    )
    with pytest.raises(ValueError, match="at most one"):
        select_explicit_policy_rows(duplicate)

    ineligible = duplicate.head(1).with_columns(
        pl.lit(False).alias("model_eligible")
    )
    with pytest.raises(ValueError, match="model-ineligible"):
        select_explicit_policy_rows(ineligible)


def test_time_band_policy_accepts_one_immutable_threshold_per_band() -> None:
    frame = prediction_frame("control", early=True).with_columns(
        pl.when(pl.col("seconds_elapsed") < 120)
        .then(0.90)
        .otherwise(0.85)
        .alias("selected_confidence_threshold"),
        pl.when(pl.col("seconds_elapsed") < 120)
        .then(pl.lit("early"))
        .otherwise(pl.lit("late"))
        .alias("policy_threshold_band"),
    )
    first_crossings = (
        frame.filter(
            pl.col("confidence") >= pl.col("selected_confidence_threshold")
        )
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .group_by("market_id", maintain_order=True)
        .first()
        .select("market_id", "observed_at", "seconds_elapsed")
        .with_columns(pl.lit(True).alias("policy_selected"))
    )
    frame = frame.join(
        first_crossings,
        on=["market_id", "observed_at", "seconds_elapsed"],
        how="left",
    ).with_columns(pl.col("policy_selected").fill_null(False))

    selected = select_time_band_policy_rows(frame)
    result = benchmark_predictions(
        {"control": frame},
        policies={
            "control": CandidatePolicy(
                confidence_threshold=None,
                deployment_compatible=False,
                selection_mode="time_band_preselected",
                confidence_threshold_min=0.85,
                confidence_threshold_max=0.90,
            )
        },
        control_candidate="control",
        evidence=BenchmarkEvidence(
            label="Chronological development",
            kind="development",
            independent=False,
        ),
        minimum_samples=1,
    )

    assert selected.height == 12
    assert result["candidates"]["control"]["own_policy"]["markets"] == 12


def test_time_band_policy_accepts_causally_frozen_thresholds_per_fold() -> None:
    first_fold = prediction_frame("control", early=True).with_columns(
        pl.lit(0).alias("fold_index"),
        pl.when(pl.col("seconds_elapsed") < 120)
        .then(0.90)
        .otherwise(0.85)
        .alias("selected_confidence_threshold"),
        pl.when(pl.col("seconds_elapsed") < 120)
        .then(pl.lit("early"))
        .otherwise(pl.lit("late"))
        .alias("policy_threshold_band"),
    )
    second_fold = prediction_frame("control", early=True).with_columns(
        pl.concat_str(pl.lit("fold-1-"), pl.col("market_id")).alias("market_id"),
        (pl.col("observed_at") + pl.duration(days=7)).alias("observed_at"),
        pl.lit(1).alias("fold_index"),
        pl.when(pl.col("seconds_elapsed") < 120)
        .then(0.85)
        .otherwise(0.80)
        .alias("selected_confidence_threshold"),
        pl.when(pl.col("seconds_elapsed") < 120)
        .then(pl.lit("early"))
        .otherwise(pl.lit("late"))
        .alias("policy_threshold_band"),
    )
    frame = pl.concat((first_fold, second_fold), how="vertical_relaxed")
    first_crossings = (
        frame.filter(
            pl.col("confidence") >= pl.col("selected_confidence_threshold")
        )
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .group_by("market_id", maintain_order=True)
        .first()
        .select("market_id", "observed_at", "seconds_elapsed")
        .with_columns(pl.lit(True).alias("policy_selected"))
    )
    frame = frame.join(
        first_crossings,
        on=["market_id", "observed_at", "seconds_elapsed"],
        how="left",
    ).with_columns(pl.col("policy_selected").fill_null(False))

    selected = select_time_band_policy_rows(frame)

    assert selected["market_id"].n_unique() == 24
    assert selected["fold_index"].n_unique() == 2


def test_time_band_policy_rejects_threshold_drift_within_a_band() -> None:
    frame = prediction_frame("control", early=True).with_columns(
        pl.when(pl.col("market_id") == "market-00")
        .then(0.91)
        .otherwise(0.90)
        .alias("selected_confidence_threshold"),
        pl.lit("60-240").alias("policy_threshold_band"),
        pl.lit(False).alias("policy_selected"),
    )

    with pytest.raises(ValueError, match="stable per band"):
        select_time_band_policy_rows(frame)
