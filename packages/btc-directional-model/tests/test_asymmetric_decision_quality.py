from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model.asymmetric_decision_quality import (
    OOF_SELECTION_COLUMNS,
    _decision_quality_gate_evidence,
    _validate_oof_selection_frame,
    _validate_parent_calibration_support,
    _window_evidence,
    calibration_variant_id,
    control_calibration_variant,
    decision_quality_metrics,
    paired_probability_delta,
    select_decision_quality_candidate,
)
from btc_directional_model.asymmetric_value_config import (
    load_asymmetric_value_config,
)
from btc_directional_model.asymmetric_value_training import (
    fit_asymmetric_time_band_calibrators,
)
from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_training import ProbabilityCalibrator
from btc_directional_model.early_value_training import TimeBandCalibrator


def _config():
    return load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-decision-quality-20260414-20260802.toml"
    )


class _FeatureLogitModel:
    candidate_name = "feature_logit"

    def raw_logit(self, frame: pl.DataFrame) -> np.ndarray:
        return frame["raw_logit"].to_numpy()


def test_targetpool_parent_replicates_across_early_runtime_bands() -> None:
    config = _config()
    core_config = load_core_config(config.core_config)
    start = datetime(2026, 7, 23, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    seconds = (1, 15, 30, 45, 60, 90, 120, 180)
    for market_index in range(600):
        label = market_index % 2
        day = market_index % 7
        for second in seconds:
            window_start = start + timedelta(days=day, minutes=5 * market_index)
            rows.append(
                {
                    "market_id": f"m{market_index}",
                    "window_start": window_start,
                    "seconds_elapsed": second,
                    "yes_ask_vwap_5": 0.25 if second <= 55 else 0.45,
                    "no_ask_vwap_5": 0.75 if second <= 55 else 0.55,
                    "label_up": label,
                    "raw_logit": 0.5 if label else -0.5,
                }
            )
    frame = pl.DataFrame(rows)

    calibrators = fit_asymmetric_time_band_calibrators(
        _FeatureLogitModel(),  # type: ignore[arg-type]
        frame,
        config,
        core_config=core_config,
        parent_source="targetpool",
    )

    early = calibrators[:4]
    assert len(calibrators) == 8
    assert all(item.markets == 600 for item in early)
    assert len({item.calibrator.slope for item in early}) == 1
    assert len({item.calibrator.intercept for item in early}) == 1
    assert all(item.calibrator.slope > 0.0 for item in calibrators)


def test_parent_calibration_support_and_window_evidence_fail_closed() -> None:
    config = _config()
    calibrators = tuple(
        TimeBandCalibrator(
            start_second=start,
            end_second_exclusive=end,
            calibrator=ProbabilityCalibrator(1.0, 0.0, True, 1),
            rows=10_000,
            markets=499,
        )
        for start, end in config.calibration_bands
    )
    with pytest.raises(RuntimeError, match="required market support"):
        _validate_parent_calibration_support(calibrators, config)

    contract = config.decision_quality
    assert contract is not None
    evidence = _window_evidence(contract.final_fit)
    assert evidence == {
        "start": "2026-04-14T00:00:00+00:00",
        "end": "2026-07-23T00:00:00+00:00",
    }
    json.dumps(evidence, allow_nan=False)


def test_decision_quality_metrics_are_market_equal_and_side_conditioned() -> None:
    start = datetime(2026, 6, 11, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "a", "b", "b", "b", "b", "b"],
            "label_up": [1, 1, 1, 0, 0, 0, 0, 0],
            "probability_yes": [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
            "yes_target_eligible": [True] * 8,
            "no_target_eligible": [True] * 8,
            "target_time_band": [
                "1_15",
                "15_30",
                "30_45",
                "1_15",
                "15_30",
                "30_45",
                "45_56",
                "45_56",
            ],
            "window_start": [start] * 8,
        }
    )

    metrics = decision_quality_metrics(frame)

    np.testing.assert_allclose(metrics["overall"]["actual_rate"], 0.5)
    np.testing.assert_allclose(metrics["overall"]["mean_probability"], 0.5)
    np.testing.assert_allclose(metrics["overall"]["brier"], 0.01)
    assert metrics["sides"]["YES"]["bias"] == pytest.approx(0.0)
    assert metrics["sides"]["NO"]["bias"] == pytest.approx(0.0)


def _synthetic_oof() -> tuple[pl.DataFrame, dict[str, object]]:
    config = _config()
    contract = config.decision_quality
    assert contract is not None
    controls = {
        item.name: calibration_variant_id(item.name, control_calibration_variant())
        for item in contract.candidates
        if not item.selection_eligible
    }
    candidate_ids = list(controls.values())
    for base in contract.candidates:
        if base.selection_eligible:
            candidate_ids.extend(
                calibration_variant_id(base.name, variant)
                for variant in contract.calibration_variants
            )
    rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(contract.folds):
        for market_index in range(40):
            label = market_index % 2
            second = (1, 15, 30, 45)[market_index % 4]
            side_yes = (market_index // 4) % 2 == 0
            window_start = fold.validation.start + timedelta(
                days=market_index % 7,
                minutes=5 * market_index,
            )
            for candidate_id in candidate_ids:
                if candidate_id == controls["broad_current"]:
                    strength = 0.70
                elif candidate_id == controls["target_only_current"]:
                    strength = 0.80
                elif candidate_id in controls.values():
                    strength = 0.75
                else:
                    strength = 0.98
                probability = strength if label else 1.0 - strength
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "base_candidate": candidate_id.split("__", maxsplit=1)[0],
                        "fold": fold.name,
                        "parent_source": (
                            "targetpool" if "targetpool" in candidate_id else "alltime"
                        ),
                        "identity_l2": (
                            0.05
                            if candidate_id.endswith("l2_0_05")
                            else (0.20 if candidate_id.endswith("l2_0_2") else 1.0)
                        ),
                        "market_id": f"{fold_index}-m{market_index}",
                        "window_start": window_start,
                        "observed_at": window_start + timedelta(seconds=second),
                        "seconds_elapsed": second,
                        "label_up": label,
                        "probability_yes": probability,
                        "yes_target_eligible": side_yes,
                        "no_target_eligible": not side_yes,
                        "target_time_band": ("1_15", "15_30", "30_45", "45_56")[market_index % 4],
                    }
                )
    oof = pl.DataFrame(rows).select(*OOF_SELECTION_COLUMNS)
    fold_profiles: dict[str, object] = {}
    for fold in contract.folds:
        candidates: dict[str, object] = {}
        for base in contract.candidates:
            calibrations: dict[str, object] = {}
            variants = (
                contract.calibration_variants
                if base.selection_eligible
                else (control_calibration_variant(),)
            )
            for variant in variants:
                candidate_id = calibration_variant_id(base.name, variant)
                calibrations[candidate_id] = {"target_calibration": {"qualified": True}}
            candidates[base.name] = {"calibrations": calibrations}
        fold_profiles[fold.name] = {"candidates": candidates}
    return oof, fold_profiles


def test_probability_only_selection_is_deterministic_and_rejects_economics() -> None:
    config = _config()
    oof, profiles = _synthetic_oof()

    first = select_decision_quality_candidate(oof, profiles, config)
    second = select_decision_quality_candidate(oof, profiles, config)

    assert first == second
    assert first["status"] == "selected"
    assert first["selected_candidate_id"] == "hybrid_25_h3__alltime_l2_1_0"
    assert first["economics_used"] is False
    _validate_oof_selection_frame(oof, config)
    with pytest.raises(RuntimeError, match="schema changed"):
        _validate_oof_selection_frame(oof.with_columns(pl.lit(1.0).alias("net_profit")), config)
    corrupted = (
        oof.with_row_index("_row")
        .with_columns(
            pl.when(pl.col("_row") == 0)
            .then(1 - pl.col("label_up"))
            .otherwise(pl.col("label_up"))
            .alias("label_up")
        )
        .drop("_row")
    )
    with pytest.raises(RuntimeError, match="exact invariant grid"):
        _validate_oof_selection_frame(corrupted, config)


def test_paired_probability_delta_fails_on_key_mismatch_and_is_repeatable() -> None:
    oof, _ = _synthetic_oof()
    candidate = oof.filter(pl.col("candidate_id") == "hybrid_25_h1__alltime_l2_0_05")
    reference = oof.filter(pl.col("candidate_id") == "target_only_current__alltime_l2_1_0")

    first = paired_probability_delta(candidate, reference, resamples=1000, seed=7)
    second = paired_probability_delta(candidate, reference, resamples=1000, seed=7)

    assert first == second
    assert first["brier_delta"]["upper_95"] < 0.0
    with pytest.raises(RuntimeError, match="keys do not match"):
        paired_probability_delta(
            candidate.head(candidate.height - 1),
            reference,
            resamples=1000,
            seed=7,
        )
    with pytest.raises(RuntimeError, match="labels do not match"):
        paired_probability_delta(
            candidate,
            reference.with_columns((1 - pl.col("label_up")).alias("label_up")),
            resamples=1000,
            seed=7,
        )


def test_paired_probability_delta_preserves_market_equal_estimand() -> None:
    start = datetime(2026, 6, 11, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for market_index in range(101):
        day = 0 if market_index < 100 else 1
        rows.append(
            {
                "fold": "fold",
                "market_id": f"m{market_index}",
                "window_start": start + timedelta(days=day, minutes=5 * market_index),
                "observed_at": start + timedelta(days=day, minutes=5 * market_index, seconds=1),
                "seconds_elapsed": 1,
                "label_up": 1,
                "probability_yes": 0.6 if day == 0 else 0.1,
            }
        )
    candidate = pl.DataFrame(rows)
    reference = candidate.with_columns(pl.lit(0.5).alias("probability_yes"))

    result = paired_probability_delta(candidate, reference, resamples=1000, seed=19)

    expected = (100 * (0.4**2 - 0.5**2) + (0.9**2 - 0.5**2)) / 101
    assert result["brier_delta"]["point"] == pytest.approx(expected)
    assert result["brier_delta"]["point"] < 0.0


def test_fold_stability_requires_four_common_folds_against_both_controls() -> None:
    config = _config()
    contract = config.decision_quality
    assert contract is not None
    oof, _ = _synthetic_oof()
    candidate_id = "hybrid_25_h1__alltime_l2_0_05"
    broad_id = "broad_current__alltime_l2_1_0"
    target_id = "target_only_current__alltime_l2_1_0"
    first_fold = contract.folds[0].name
    second_fold = contract.folds[1].name
    frame = oof.filter(pl.col("candidate_id").is_in([candidate_id, broad_id, target_id]))
    frame = frame.with_columns(
        pl.when(
            ((pl.col("candidate_id") == broad_id) & (pl.col("fold") == first_fold))
            | ((pl.col("candidate_id") == target_id) & (pl.col("fold") == second_fold))
        )
        .then(pl.when(pl.col("label_up") == 1).then(0.999).otherwise(0.001))
        .otherwise(pl.col("probability_yes"))
        .alias("probability_yes")
    )
    favorable_delta = {
        "brier_delta": {"point": -0.01, "lower_95": -0.02, "upper_95": -0.001},
        "log_loss_delta": {"point": -0.01, "lower_95": -0.02, "upper_95": -0.001},
    }

    checks = _decision_quality_gate_evidence(
        decision_quality_metrics(frame.filter(pl.col("candidate_id") == candidate_id)),
        {"broad_current": favorable_delta, "target_only_current": favorable_delta},
        candidate_id=candidate_id,
        oof=frame,
        references={"broad_current": broad_id, "target_only_current": target_id},
        contract=contract,
        calibration_ok=True,
    )
    by_name = {item["name"]: item for item in checks}

    assert by_name["noninferior_folds_to_broad_current"]["observed"] == 4
    assert by_name["noninferior_folds_to_target_only_current"]["observed"] == 4
    assert by_name["noninferior_folds_to_all_controls"]["observed"] == 3
    assert by_name["noninferior_folds_to_all_controls"]["passed"] is False
