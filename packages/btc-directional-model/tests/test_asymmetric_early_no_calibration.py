from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import btc_directional_model.asymmetric_decision_quality as decision_quality
from btc_directional_model.asymmetric_decision_quality import _fit_calibrated_bundle
from btc_directional_model.asymmetric_value_config import (
    DAY_MARKET_ROW_EQUAL_CALIBRATION_WEIGHTING,
    EARLY_NO_CALIBRATION_DECISION_QUALITY_STUDY,
    MARKET_EQUAL_CALIBRATION_WEIGHTING,
    load_asymmetric_value_config,
)
from btc_directional_model.asymmetric_value_training import (
    CORE_L2_PRICE,
    EXPECTED_MODEL_FEATURE_COUNTS,
    AsymmetricCalibrationCell,
    _day_market_row_equal_weights,
    asymmetric_value_feature_sets,
    fit_side_price_time_calibrators,
)
from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_training import ProbabilityCalibrator
from btc_directional_model.early_value_training import TimeBandCalibrator


def _config():
    return load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-early-no-calibration-20260414-20260802.toml"
    )


class _FeatureLogitModel:
    candidate_name = "feature_logit"

    def raw_logit(self, frame: pl.DataFrame) -> np.ndarray:
        return frame["raw_logit"].to_numpy()


def test_early_no_contract_freezes_architecture_matrix_and_chronology() -> None:
    config = _config()
    contract = config.decision_quality
    assert contract is not None
    assert contract.study == EARLY_NO_CALIBRATION_DECISION_QUALITY_STUDY
    assert len(contract.folds) == 10
    assert all(
        (fold.calibration.end - fold.calibration.start).days == 28 for fold in contract.folds
    )
    assert all((fold.validation.end - fold.validation.start).days == 1 for fold in contract.folds)
    assert [fold.validation.start.date().isoformat() for fold in contract.folds] == [
        "2026-07-21",
        "2026-07-22",
        "2026-07-23",
        "2026-07-24",
        "2026-07-25",
        "2026-07-28",
        "2026-07-29",
        "2026-07-30",
        "2026-07-31",
        "2026-08-01",
    ]
    assert contract.oof_evidence_scope == "consumed_cross_day_development"
    assert contract.oof_forward_proof is False
    assert contract.final_fit.end == datetime(2026, 7, 16, tzinfo=UTC)
    assert contract.final_calibration.start == datetime(2026, 7, 16, tzinfo=UTC)
    assert contract.final_calibration.end == datetime(2026, 8, 2, tzinfo=UTC)
    assert [
        (
            candidate.name,
            candidate.target_weight,
            candidate.histogram_profile,
            candidate.selection_eligible,
        )
        for candidate in contract.candidates
    ] == [
        ("broad_current", 0.0, "h0_current", False),
        ("target_only_current", 1.0, "h0_current", False),
        ("hybrid_50_h3", 0.5, "h3_regularized", True),
    ]
    assert [variant.name for variant in contract.calibration_variants] == [
        "targetpool_control",
        "targetpool_day_balanced",
        "targetpool_early_no_offset",
        "targetpool_day_balanced_early_no_offset",
    ]
    assert [variant.calibration_weighting for variant in contract.calibration_variants] == [
        MARKET_EQUAL_CALIBRATION_WEIGHTING,
        DAY_MARKET_ROW_EQUAL_CALIBRATION_WEIGHTING,
        MARKET_EQUAL_CALIBRATION_WEIGHTING,
        DAY_MARKET_ROW_EQUAL_CALIBRATION_WEIGHTING,
    ]
    assert [variant.early_no_intercept_only for variant in contract.calibration_variants] == [
        False,
        False,
        True,
        True,
    ]
    assert all(variant.parent_source == "targetpool" for variant in contract.calibration_variants)
    assert all(variant.identity_l2 == 1.0 for variant in contract.calibration_variants)
    features = asymmetric_value_feature_sets()[CORE_L2_PRICE]
    assert len(features) == EXPECTED_MODEL_FEATURE_COUNTS[CORE_L2_PRICE] == 111

    core = load_core_config(config.core_config)
    assert core.data.range_start == datetime(2026, 4, 14, tzinfo=UTC)
    assert core.data.range_end == datetime(2026, 8, 2, tzinfo=UTC)
    assert "btc-asymmetric-value-calibrated-20260414-20260802" in str(core.paths.source_data)
    assert "btc-asymmetric-value-calibrated-20260414-20260802" in str(config.feature_cache)


def test_day_market_row_weights_are_equal_at_every_hierarchy() -> None:
    market_ids = np.asarray(["a", "b", "b", "b", "c", "c"])
    utc_days = np.asarray(["2026-07-23"] * 4 + ["2026-07-24"] * 2)

    weights = _day_market_row_equal_weights(market_ids, utc_days)

    np.testing.assert_allclose(weights.sum(), 1.0)
    np.testing.assert_allclose(weights[utc_days == "2026-07-23"].sum(), 0.5)
    np.testing.assert_allclose(weights[utc_days == "2026-07-24"].sum(), 0.5)
    np.testing.assert_allclose(weights[market_ids == "a"].sum(), 0.25)
    np.testing.assert_allclose(weights[market_ids == "b"].sum(), 0.25)
    np.testing.assert_allclose(weights[market_ids == "c"].sum(), 0.5)
    np.testing.assert_allclose(weights[market_ids == "b"], np.repeat(1.0 / 12.0, 3))
    np.testing.assert_allclose(weights[market_ids == "c"], np.repeat(0.25, 2))
    with pytest.raises(ValueError, match="aligned non-empty vectors"):
        _day_market_row_equal_weights(np.asarray(["a"]), np.asarray([], dtype=str))


def test_early_no_offset_fixes_only_targeted_slope_to_identity() -> None:
    config = _config()
    start = datetime(2026, 7, 23, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for market_index in range(240):
        label = market_index % 2
        yes_target = market_index % 4 < 2
        window_start = start + timedelta(
            days=market_index % 14,
            minutes=5 * (market_index // 14),
        )
        for second in (1, 15, 30, 45, 60, 90, 120, 180):
            rows.append(
                {
                    "market_id": f"m{market_index}",
                    "window_start": window_start,
                    "seconds_elapsed": second,
                    "yes_ask_vwap_5": 0.25 if yes_target else 0.75,
                    "no_ask_vwap_5": 0.75 if yes_target else 0.25,
                    "label_up": label,
                    "raw_logit": 0.15 if label else -0.15,
                }
            )
    frame = pl.DataFrame(rows)
    calibrators = tuple(
        TimeBandCalibrator(
            start_second=band_start,
            end_second_exclusive=band_end,
            calibrator=ProbabilityCalibrator(1.0, 0.0, True, 1),
            rows=frame.height,
            markets=240,
        )
        for band_start, band_end in config.calibration_bands
    )

    cells = fit_side_price_time_calibrators(
        _FeatureLogitModel(),  # type: ignore[arg-type]
        calibrators,
        frame,
        config,
        identity_l2_strength=1.0,
        slope_bounds=(0.05, 3.0),
        intercept_bounds=(-2.0, 2.0),
        calibration_weighting=DAY_MARKET_ROW_EQUAL_CALIBRATION_WEIGHTING,
        early_no_intercept_only=True,
    )

    early_no = next(
        cell
        for cell in cells
        if cell.start_second == 1
        and cell.end_second_exclusive == 15
        and cell.side == "NO"
        and cell.minimum_price == pytest.approx(0.20)
    )
    later_no = next(
        cell
        for cell in cells
        if cell.start_second == 15
        and cell.end_second_exclusive == 30
        and cell.side == "NO"
        and cell.minimum_price == pytest.approx(0.20)
    )
    assert early_no.fitted is True
    assert early_no.slope == 1.0
    assert early_no.identity_l2_strength == 1.0
    assert later_no.fitted is True
    assert later_no.slope != pytest.approx(1.0, abs=1e-6)


def test_calibrated_bundle_routes_variant_weighting_and_parameterization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    contract = config.decision_quality
    assert contract is not None
    variant = contract.calibration_variants[-1]
    captured: dict[str, object] = {}
    calibrator = ProbabilityCalibrator(1.0, 0.0, True, 1)
    time_calibrators = tuple(
        TimeBandCalibrator(start, end, calibrator, 4_000, 500)
        for start, end in config.calibration_bands
    )
    cells = tuple(
        AsymmetricCalibrationCell(
            start_second=start,
            end_second_exclusive=end,
            minimum_price=0.20,
            maximum_price=0.30,
            side=side,
            slope=1.0 if (start, side) == (1, "NO") else 1.1,
            intercept=0.05,
            fitted=True,
            fallback=None,
            rows=500,
            markets=500,
            utc_days=14,
            positives=250,
            negatives=250,
            identity_l2_strength=1.0,
            converged=True,
            iterations=3,
            objective=0.5,
            weighted_log_loss=0.5,
        )
        for start, end in config.target_calibration.time_bands  # type: ignore[union-attr]
        for side in ("YES", "NO")
    )

    def fake_parent(*args, **kwargs):
        captured["parent"] = kwargs
        return time_calibrators

    def fake_cells(*args, **kwargs):
        captured["cells"] = kwargs
        return cells

    monkeypatch.setattr(decision_quality, "fit_asymmetric_time_band_calibrators", fake_parent)
    monkeypatch.setattr(decision_quality, "fit_side_price_time_calibrators", fake_cells)
    monkeypatch.setattr(
        decision_quality,
        "target_calibration_evidence",
        lambda *args, **kwargs: {"qualified": True},
    )
    start = datetime(2026, 7, 23, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "market_id": [f"m{index}" for index in range(500)],
            "window_start": [start + timedelta(days=index % 14) for index in range(500)],
            "seconds_elapsed": [1] * 500,
            "yes_ask_vwap_5": [0.25] * 500,
            "no_ask_vwap_5": [0.75] * 500,
            "label_up": [index % 2 for index in range(500)],
        }
    )

    _, profile = _fit_calibrated_bundle(
        object(),
        frame,
        config,
        load_core_config(config.core_config),
        base_candidate="hybrid_50_h3",
        variant=variant,
    )

    assert captured["parent"]["parent_source"] == "targetpool"  # type: ignore[index]
    assert (
        captured["parent"]["calibration_weighting"]  # type: ignore[index]
        == DAY_MARKET_ROW_EQUAL_CALIBRATION_WEIGHTING
    )
    assert (
        captured["cells"]["calibration_weighting"]  # type: ignore[index]
        == DAY_MARKET_ROW_EQUAL_CALIBRATION_WEIGHTING
    )
    assert captured["cells"]["early_no_intercept_only"] is True  # type: ignore[index]
    assert profile["calibration_variant"] == variant.name
    assert profile["early_no_intercept_only"] is True
    assert profile["target_calibration"]["early_no_parameterization"] == {
        "start_second": 1,
        "end_second_exclusive": 15,
        "minimum_price": 0.20,
        "maximum_price": 0.30,
        "side": "NO",
        "intercept_only": True,
        "slope": 1.0,
        "intercept": 0.05,
        "qualified": True,
    }
