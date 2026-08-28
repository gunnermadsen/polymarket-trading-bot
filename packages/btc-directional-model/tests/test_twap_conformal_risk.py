from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model.twap_conformal_risk import (
    CHECKPOINT_SECONDS,
    DIRECTION_TIME_PRICE,
    ConformalArtifact,
    apply_admission_contract,
    apply_conformal_bounds,
    artifact_bytes,
    conformal_quantile,
    earliest_admitted_trades,
    fit_conformal_artifact,
    load_artifact,
    write_artifact,
)
from btc_directional_model.twap_conformal_risk_tournament import (
    FoldLedger,
    candidate_metrics,
    development_checks,
    load_config,
    persist_daily_ledgers,
    validate_prediction_frame,
)


def _calibration_frame(markets: int = 105) -> pl.DataFrame:
    rows = []
    start = datetime(2026, 8, 14, tzinfo=UTC)
    for market in range(markets):
        label_up = market % 2 == 0
        margin = 20.0 if label_up else -20.0
        probability = 0.96 if label_up else 0.04
        for second in CHECKPOINT_SECONDS:
            observed = start + timedelta(minutes=5 * market, seconds=second)
            rows.append(
                {
                    "market_id": str(market),
                    "window_start": start + timedelta(minutes=5 * market),
                    "window_end": start + timedelta(minutes=5 * (market + 1)),
                    "seconds_elapsed": second,
                    "observed_at": observed,
                    "sensor_max_available_at": observed - timedelta(milliseconds=1),
                    "label_source": "authentic_official_twap60",
                    "sensor_valid": True,
                    "probability_up": probability,
                    "label_up": label_up,
                    "target_margin_bps": margin,
                    "expected_margin_bps": margin * 0.8,
                    "margin_p05_bps": margin - 2.0,
                    "margin_p50_bps": margin,
                    "margin_p95_bps": margin + 2.0,
                    "up_ask_vwap_5": 0.55 if label_up else 0.45,
                    "down_ask_vwap_5": 0.45 if label_up else 0.55,
                    "book_valid": True,
                    "fee_rate": 0.0,
                    "fold": "synthetic",
                }
            )
    return pl.DataFrame(rows)


def _artifact(frame: pl.DataFrame) -> ConformalArtifact:
    artifact, _ = fit_conformal_artifact(
        frame,
        DIRECTION_TIME_PRICE,
        alpha=0.10,
        minimum_cell_markets=100,
        calibration_start="2026-08-14T00:00:00+00:00",
        calibration_end="2026-08-18T00:00:00+00:00",
        source_identity="source",
        prediction_ledger_sha256="ledger",
        feature_registry_sha256="features",
        upstream_artifact_sha256="upstream",
    )
    return artifact


def test_finite_sample_quantile_uses_conformal_rank() -> None:
    values = np.arange(10, dtype=float)
    assert conformal_quantile(values, 0.10) == 9.0
    with pytest.raises(ValueError):
        conformal_quantile(np.array([np.nan]), 0.10)


def test_market_block_calibration_and_fallback_are_deterministic() -> None:
    frame = _calibration_frame()
    artifact = _artifact(frame)
    assert artifact.alpha == 0.10
    assert artifact.cells["global"]["global"].support_markets == 105
    # Direction/time/price cells have only about half the markets and must fall back.
    bounded = apply_conformal_bounds(frame.head(1), artifact)
    assert bounded["conformal_fallback"][0] == "global"
    assert bounded["calibration_support_markets"][0] == 105
    repeated = apply_conformal_bounds(frame.head(1), artifact)
    assert bounded.equals(repeated)


def test_direction_symmetry_and_quote_geometry() -> None:
    frame = _calibration_frame()
    artifact = _artifact(frame)
    up = frame.filter(pl.col("label_up")).head(1)
    down = frame.filter(~pl.col("label_up")).head(1)
    bounded = apply_conformal_bounds(pl.concat([up, down]), artifact)
    decisions = apply_admission_contract(
        bounded,
        reserve_per_share=0.005,
        stress_slippage_per_share=0.01,
        minimum_correctness=0.90,
        maximum_error_risk=0.10,
        maximum_recovery_ratio=3.0,
        evaluation_quantity=5,
    )
    assert decisions["correctness_lower_bound"][0] == pytest.approx(
        decisions["correctness_lower_bound"][1]
    )
    assert decisions["entry_eligible"].to_list() == [True, True]
    expensive = bounded.with_columns(
        pl.lit(0.80).alias("selected_cost_5"),
    )
    rejected = apply_admission_contract(
        expensive,
        reserve_per_share=0.005,
        stress_slippage_per_share=0.01,
        minimum_correctness=0.90,
        maximum_error_risk=0.10,
        maximum_recovery_ratio=3.0,
        evaluation_quantity=5,
    )
    assert not any(rejected["entry_eligible"])
    assert set(rejected["abstention_reason"]) == {"quote_recovery_geometry"}


def test_earliest_checkpoint_and_one_trade_per_market() -> None:
    frame = _calibration_frame(1).head(3)
    artifact, _ = fit_conformal_artifact(
        _calibration_frame(),
        DIRECTION_TIME_PRICE,
        alpha=0.10,
        minimum_cell_markets=100,
        calibration_start="2026-08-14T00:00:00+00:00",
        calibration_end="2026-08-18T00:00:00+00:00",
        source_identity="source",
        prediction_ledger_sha256="ledger",
        feature_registry_sha256="features",
        upstream_artifact_sha256="upstream",
    )
    decisions = apply_admission_contract(
        apply_conformal_bounds(frame, artifact),
        reserve_per_share=0.005,
        stress_slippage_per_share=0.01,
        minimum_correctness=0.90,
        maximum_error_risk=0.10,
        maximum_recovery_ratio=3.0,
        evaluation_quantity=5,
    )
    trades = earliest_admitted_trades(decisions)
    assert trades.height == 1
    assert trades["seconds_elapsed"][0] == 30


def test_serialization_reload_and_single_row_decisions_match(tmp_path) -> None:
    frame = _calibration_frame()
    artifact = _artifact(frame)
    path = tmp_path / "artifact.json"
    digest = write_artifact(path, artifact)
    reloaded = load_artifact(path)
    assert artifact_bytes(artifact) == artifact_bytes(reloaded)
    assert len(digest) == 64
    batch = apply_conformal_bounds(frame.head(4), reloaded)
    singles = pl.concat(
        [apply_conformal_bounds(frame.slice(index, 1), reloaded) for index in range(4)],
        how="vertical",
    )
    columns = [
        "correctness_lower_bound",
        "error_risk_upper_bound",
        "conformal_margin_lower",
        "conformal_margin_upper",
        "conformal_fallback",
    ]
    assert batch.select(columns).equals(singles.select(columns))


def test_causal_trajectory_validation_fails_closed() -> None:
    frame = _calibration_frame(1)
    fold = FoldLedger(
        "synthetic", datetime(2026, 8, 14, tzinfo=UTC), datetime(2026, 8, 16, tzinfo=UTC), "x"
    )
    validate_prediction_frame(frame, fold)
    invalid = frame.with_columns(pl.col("observed_at").alias("sensor_max_available_at"))
    with pytest.raises(RuntimeError, match="causal availability"):
        validate_prediction_frame(invalid, fold)
    partial = frame.head(frame.height - 1)
    with pytest.raises(RuntimeError, match="partial checkpoint"):
        validate_prediction_frame(partial, fold)


def test_daily_prediction_checkpoint_resumes_only_identical_inputs(tmp_path) -> None:
    frame = _calibration_frame(2)
    first = persist_daily_ledgers(frame, tmp_path / "ledgers", "input-a")
    second = persist_daily_ledgers(frame, tmp_path / "ledgers", "input-a")
    assert first == second
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        persist_daily_ledgers(frame, tmp_path / "ledgers", "input-b")


def test_zero_trade_report_retains_every_required_metric_group() -> None:
    frame = _calibration_frame(2)
    artifact = _artifact(_calibration_frame())
    decisions = apply_admission_contract(
        apply_conformal_bounds(frame, artifact),
        reserve_per_share=0.005,
        stress_slippage_per_share=0.01,
        minimum_correctness=1.0,
        maximum_error_risk=0.0,
        maximum_recovery_ratio=0.0,
        evaluation_quantity=5,
    )
    trades = earliest_admitted_trades(decisions)
    config = load_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-twap-conformal-risk-admission-20260814-20260825.toml"
    )
    metrics = candidate_metrics(frame, decisions, trades, config)
    assert metrics["trades"] == 0
    assert set(metrics["entry_time_bands"]) == {"30-59", "60-89", "90-120"}
    assert set(metrics["executable_price_bands"]) == {
        "below_0.60",
        "0.60-0.70",
        "0.70-0.80",
        "above_0.80",
    }
    assert set(metrics["loss_distribution"]) == {"average", "median", "p90", "worst"}
    paired = {"bootstrap_improvement": {"lower": None}}
    checks = development_checks(
        DIRECTION_TIME_PRICE, metrics, metrics, paired, config
    )
    assert checks["average_loss_recovery"] is False
    assert checks["average_entry"] is False
