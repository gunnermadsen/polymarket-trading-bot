from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model.champion_vwap_benchmark import (
    CHALLENGERS,
    CONTROL,
    VWAP5,
    VWAP5_DEPTH,
    ConditionedCalibrator,
    candidate_metrics,
    fit_challenger_calibrators,
    promotion_decision,
    score_candidates,
)
from btc_directional_model.champion_vwap_config import load_champion_vwap_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PACKAGE_ROOT
    / "configs"
    / "btc-5m-directional-champion-vwap-calibration-20260621-20260729.toml"
)


def test_config_pins_frozen_champion_and_oos_boundary() -> None:
    config = load_champion_vwap_config(CONFIG_PATH)

    assert str(config.source_process_id) == "3d8a8efd-fea6-41ec-b885-36ecf7877f4c"
    assert config.champion_model_key.endswith("20260421-20260620-v1")
    assert config.data.out_of_sample_score_start == datetime(2026, 6, 21, tzinfo=UTC)
    assert config.data.calibration_end == datetime(2026, 7, 14, tzinfo=UTC)
    assert config.model.confidence_threshold == 0.89
    assert config.model.quantity == 5.0


def test_challengers_fit_only_declared_price_inputs() -> None:
    config = load_champion_vwap_config(CONFIG_PATH)
    frame = _decision_frame(400)

    models = fit_challenger_calibrators(frame, config)

    assert models[VWAP5].feature_names == (
        "champion_selected_raw_logit",
        "selected_ask_vwap_5",
    )
    assert models[VWAP5_DEPTH].feature_names == (
        "champion_selected_raw_logit",
        "selected_ask_vwap_5",
        "vwap10_minus_vwap5",
    )
    assert all(model.fit_rows == frame.height for model in models.values())
    assert all(model.converged for model in models.values())


def test_missing_vwap_is_rejected_instead_of_imputed() -> None:
    config = load_champion_vwap_config(CONFIG_PATH)
    frame = _decision_frame(400).with_columns(
        pl.when(pl.int_range(pl.len()) == 7)
        .then(None)
        .otherwise(pl.col("selected_ask_vwap_5"))
        .alias("selected_ask_vwap_5")
    )

    with pytest.raises(ValueError, match="imputation is disabled"):
        fit_challenger_calibrators(frame, config)


def test_candidate_scoring_keeps_identical_rows_and_locked_direction() -> None:
    frame = _decision_frame(20)
    models = {
        VWAP5: _constant_calibrator(VWAP5, ("champion_selected_raw_logit",), 4.0),
        VWAP5_DEPTH: _constant_calibrator(
            VWAP5_DEPTH,
            ("champion_selected_raw_logit",),
            -4.0,
        ),
    }

    scored = score_candidates(frame, models, 0.89)

    assert tuple(scored) == (CONTROL, *CHALLENGERS)
    for candidate_frame in scored.values():
        assert candidate_frame["market_id"].to_list() == frame["market_id"].to_list()
        assert (
            candidate_frame["champion_selected_up"].to_list()
            == frame["champion_selected_up"].to_list()
        )
    assert scored[CONTROL]["policy_selected"].all()
    assert scored[VWAP5]["policy_selected"].all()
    assert not scored[VWAP5_DEPTH]["policy_selected"].any()


def test_metrics_keep_ten_share_economics_separate() -> None:
    frame = score_candidates(
        _decision_frame(100),
        {
            VWAP5: _constant_calibrator(
                VWAP5,
                ("champion_selected_raw_logit",),
                4.0,
            ),
            VWAP5_DEPTH: _constant_calibrator(
                VWAP5_DEPTH,
                ("champion_selected_raw_logit",),
                4.0,
            ),
        },
        0.89,
    )[VWAP5]

    metrics = candidate_metrics(frame, eligible_markets=100)

    assert metrics["five_share"]["trades"] == 100
    assert metrics["ten_share_reporting_only"]["trades"] == 100
    assert (
        metrics["five_share"]["net_pnl"]
        != metrics["ten_share_reporting_only"]["net_pnl"]
    )
    assert "worst_one_percent_mean" in metrics["five_share"]
    assert "losses_above" in metrics["price"]
    assert "bands" in metrics["depth"]


def test_promotion_retains_champion_when_coverage_collapses() -> None:
    config = load_champion_vwap_config(CONFIG_PATH)
    frame = _decision_frame(400)
    models = {
        candidate: _constant_calibrator(
            candidate,
            ("champion_selected_raw_logit",),
            -4.0,
        )
        for candidate in CHALLENGERS
    }
    scored = score_candidates(frame, models, config.model.confidence_threshold)
    metrics = {
        candidate: candidate_metrics(candidate_frame, eligible_markets=frame.height)
        for candidate, candidate_frame in scored.items()
    }
    folds = {
        fold.name: {"candidates": metrics}
        for fold in config.folds
    }

    decision = promotion_decision(metrics, folds, config)

    assert decision["selected_candidate"] is None
    assert decision["champion_retained"]
    assert all(not decision["candidates"][name]["passed"] for name in CHALLENGERS)


def _decision_frame(rows: int) -> pl.DataFrame:
    start = datetime(2026, 6, 29, tzinfo=UTC)
    index = np.arange(rows)
    price = 0.70 + (index % 25) * 0.01
    slope = (index % 5) * 0.004
    raw_logit = 1.8 + (index % 20) * 0.08
    correct = (raw_logit - 2.0 * price - 3.0 * slope) > 0.15
    selected_up = index % 2 == 0
    label_up = np.where(correct, selected_up, ~selected_up).astype(np.int8)
    fee_rate = np.full(rows, 0.25)
    fee_5 = fee_rate * price * (1.0 - price)
    price_10 = price + slope
    fee_10 = fee_rate * price_10 * (1.0 - price_10)
    return pl.DataFrame(
        {
            "market_id": [f"market-{value}" for value in index],
            "window_start": [start + timedelta(minutes=5 * int(value)) for value in index],
            "observed_at": [
                start + timedelta(minutes=5 * int(value), seconds=60) for value in index
            ],
            "seconds_elapsed": np.full(rows, 60),
            "label_up": label_up,
            "champion_selected_up": selected_up,
            "champion_selected_raw_logit": raw_logit,
            "champion_selected_probability": np.full(rows, 0.91),
            "selected_ask_vwap_5": price,
            "selected_ask_vwap_10": price_10,
            "vwap10_minus_vwap5": slope,
            "fee_rate": fee_rate,
            "correct": correct,
            "realized_net_pnl_5": (correct.astype(float) - price - fee_5) * 5.0,
            "realized_net_pnl_10": (correct.astype(float) - price_10 - fee_10) * 10.0,
        }
    )


def _constant_calibrator(
    candidate: str,
    feature_names: tuple[str, ...],
    intercept: float,
) -> ConditionedCalibrator:
    return ConditionedCalibrator(
        candidate=candidate,
        feature_names=feature_names,
        means=tuple(0.0 for _ in feature_names),
        scales=tuple(1.0 for _ in feature_names),
        coefficients=tuple(0.0 for _ in feature_names),
        intercept=intercept,
        iterations=1,
        converged=True,
        fit_rows=100,
        fit_start="2026-06-29T00:00:00+00:00",
        fit_end="2026-07-01T00:00:00+00:00",
    )
