from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model.asymmetric_decision_quality import (
    DECISION_SELECTION_SEAL_SCHEMA_VERSION,
    MATCHED_CORE_CONTROL_CANDIDATE_ID,
    OOF_SELECTION_COLUMNS,
    POST_SELECTION_ATTRIBUTION_PAIRS,
    _decision_quality_gate_evidence,
    _validate_attribution_pair_oof_grid,
    _validate_matched_core_feature_contract,
    _validate_matched_core_oof_grid,
    _validate_oof_selection_frame,
    _validate_parent_calibration_support,
    _window_evidence,
    calibration_variant_id,
    control_calibration_variant,
    decision_quality_metrics,
    fit_final_matched_core_model,
    fit_final_selected_attribution_pair,
    fit_selected_attribution_pair_walk_forward,
    fit_selected_matched_core_walk_forward,
    paired_probability_delta,
    post_selection_attribution_candidate_id,
    select_decision_quality_candidate,
)
from btc_directional_model.asymmetric_value_config import (
    load_asymmetric_value_config,
)
from btc_directional_model.asymmetric_value_training import (
    CANDLE_MATCHED_CORE_PRICE_CONTROL,
    CORE_CANDLES_PRICE,
    CORE_ORACLE_L2_PRICE,
    CORE_ORACLE_PRICE,
    L2_MATCHED_CORE_PRICE_CONTROL,
    ORACLE_MATCHED_CORE_PRICE_CONTROL,
    THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL,
    asymmetric_value_feature_sets,
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


def _selected_configuration() -> dict[str, object]:
    return {
        "status": "selected",
        "selected_candidate_id": "hybrid_25_h3__alltime_l2_1_0",
        "selected_base_candidate": "hybrid_25_h3",
        "economics_used": False,
    }


def _matched_replay_frame(primary_oof: pl.DataFrame) -> pl.DataFrame:
    selected_id = str(_selected_configuration()["selected_candidate_id"])
    selected = primary_oof.filter(pl.col("candidate_id") == selected_id)
    replay = selected.select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "yes_target_eligible",
        "no_target_eligible",
    ).with_columns(
        pl.when(pl.col("yes_target_eligible")).then(0.25).otherwise(0.75).alias("yes_ask_vwap_5"),
        pl.when(pl.col("no_target_eligible")).then(0.25).otherwise(0.75).alias("no_ask_vwap_5"),
    )
    support = pl.DataFrame(
        {
            "market_id": ["fit-support", "calibration-support", "final-calibration"],
            "window_start": [
                datetime(2026, 4, 15, tzinfo=UTC),
                datetime(2026, 6, 5, tzinfo=UTC),
                datetime(2026, 7, 24, tzinfo=UTC),
            ],
            "observed_at": [
                datetime(2026, 4, 15, 0, 0, 1, tzinfo=UTC),
                datetime(2026, 6, 5, 0, 0, 1, tzinfo=UTC),
                datetime(2026, 7, 24, 0, 0, 1, tzinfo=UTC),
            ],
            "seconds_elapsed": [1, 1, 1],
            "label_up": [0, 1, 0],
            "yes_target_eligible": [True, True, True],
            "no_target_eligible": [False, False, False],
            "yes_ask_vwap_5": [0.25, 0.25, 0.25],
            "no_ask_vwap_5": [0.75, 0.75, 0.75],
        }
    )
    return pl.concat((replay, support), how="vertical_relaxed")


class _MatchedCoreFakeBundle:
    def __init__(self) -> None:
        self.name = "fake"

    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        return np.where(frame["label_up"].to_numpy() == 1, 0.7, 0.3)


def test_matched_core_replays_selected_configuration_on_exact_oof_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    core_config = load_core_config(config.core_config)
    primary_oof, _ = _synthetic_oof()
    frame = _matched_replay_frame(primary_oof)
    fit_calls: list[dict[str, object]] = []
    calibration_calls: list[dict[str, object]] = []

    def fake_fit(
        fit_frame: pl.DataFrame,
        features: tuple[str, ...],
        candidate: object,
        *_args: object,
    ) -> tuple[object, dict[str, object]]:
        fit_calls.append(
            {
                "frame": fit_frame,
                "features": features,
                "candidate": candidate,
            }
        )
        return object(), {"fit_rows": fit_frame.height}

    def fake_calibration(
        _model: object,
        calibration_frame: pl.DataFrame,
        *_args: object,
        base_candidate: str,
        variant: object,
    ) -> tuple[_MatchedCoreFakeBundle, dict[str, object]]:
        calibration_calls.append(
            {
                "frame": calibration_frame,
                "base_candidate": base_candidate,
                "variant": variant,
            }
        )
        return _MatchedCoreFakeBundle(), {"target_calibration": {"qualified": True}}

    monkeypatch.setattr(
        "btc_directional_model.asymmetric_decision_quality.fit_hybrid_histogram_model",
        fake_fit,
    )
    monkeypatch.setattr(
        "btc_directional_model.asymmetric_decision_quality._fit_calibrated_bundle",
        fake_calibration,
    )

    control, evidence = fit_selected_matched_core_walk_forward(
        frame,
        primary_oof,
        _selected_configuration(),
        config,
        core_config,
    )
    selected = primary_oof.filter(
        pl.col("candidate_id") == _selected_configuration()["selected_candidate_id"]
    )

    assert control["candidate_id"].unique().to_list() == [MATCHED_CORE_CONTROL_CANDIDATE_ID]
    assert control["base_candidate"].unique().to_list() == [L2_MATCHED_CORE_PRICE_CONTROL]
    assert evidence["selection_eligible"] is False
    assert evidence["selected_grid_key_sha256"] == evidence["control_grid_key_sha256"]
    assert len(fit_calls) == 5
    assert len(calibration_calls) == 5
    for call in fit_calls:
        features = call["features"]
        candidate = call["candidate"]
        assert isinstance(features, tuple)
        assert features == asymmetric_value_feature_sets()[L2_MATCHED_CORE_PRICE_CONTROL]
        assert len(features) == 71
        assert candidate.name == "hybrid_25_h3"  # type: ignore[attr-defined]
        assert candidate.target_weight == pytest.approx(0.25)  # type: ignore[attr-defined]
        assert candidate.histogram_profile == "h3_regularized"  # type: ignore[attr-defined]
    assert all(
        call["base_candidate"] == L2_MATCHED_CORE_PRICE_CONTROL for call in calibration_calls
    )
    assert all(call["variant"].parent_source == "alltime" for call in calibration_calls)  # type: ignore[union-attr]
    assert all(call["variant"].identity_l2 == 1.0 for call in calibration_calls)  # type: ignore[union-attr]
    _validate_matched_core_oof_grid(control, selected, config)
    with pytest.raises(RuntimeError, match="exact selected key/label/eligibility grid"):
        _validate_matched_core_oof_grid(
            control,
            selected.with_row_index("_row")
            .with_columns(
                pl.when(pl.col("_row") == 0)
                .then(1 - pl.col("label_up"))
                .otherwise(pl.col("label_up"))
                .alias("label_up")
            )
            .drop("_row"),
            config,
        )


def test_matched_core_final_fit_uses_frozen_windows_and_rejects_bad_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    core_config = load_core_config(config.core_config)
    primary_oof, _ = _synthetic_oof()
    frame = _matched_replay_frame(primary_oof)
    observed: dict[str, object] = {}

    def fake_fit(
        fit_frame: pl.DataFrame,
        features: tuple[str, ...],
        candidate: object,
        *_args: object,
    ) -> tuple[object, dict[str, object]]:
        observed["fit_frame"] = fit_frame
        observed["features"] = features
        observed["candidate"] = candidate
        return object(), {"fit_rows": fit_frame.height}

    def fake_calibration(
        _model: object,
        calibration_frame: pl.DataFrame,
        *_args: object,
        base_candidate: str,
        variant: object,
    ) -> tuple[_MatchedCoreFakeBundle, dict[str, object]]:
        observed["calibration_frame"] = calibration_frame
        observed["base_candidate"] = base_candidate
        observed["variant"] = variant
        return _MatchedCoreFakeBundle(), {"target_calibration": {"qualified": True}}

    monkeypatch.setattr(
        "btc_directional_model.asymmetric_decision_quality.fit_hybrid_histogram_model",
        fake_fit,
    )
    monkeypatch.setattr(
        "btc_directional_model.asymmetric_decision_quality._fit_calibrated_bundle",
        fake_calibration,
    )

    bundle, evidence = fit_final_matched_core_model(
        frame,
        _selected_configuration(),
        config,
        core_config,
    )

    contract = config.decision_quality
    assert contract is not None
    fit_frame = observed["fit_frame"]
    calibration_frame = observed["calibration_frame"]
    assert isinstance(fit_frame, pl.DataFrame)
    assert isinstance(calibration_frame, pl.DataFrame)
    assert fit_frame["window_start"].min() >= contract.final_fit.start
    assert fit_frame["window_start"].max() < contract.final_fit.end
    assert calibration_frame["window_start"].min() >= contract.final_calibration.start
    assert calibration_frame["window_start"].max() < contract.final_calibration.end
    assert bundle.name == L2_MATCHED_CORE_PRICE_CONTROL
    assert evidence["feature_count"] == 71
    assert evidence["selection_eligible"] is False
    assert evidence["economics_used"] is False
    with pytest.raises(RuntimeError, match="economic evidence cannot select"):
        fit_final_matched_core_model(
            frame,
            {**_selected_configuration(), "economics_used": True},
            config,
            core_config,
        )
    features = asymmetric_value_feature_sets()[L2_MATCHED_CORE_PRICE_CONTROL]
    _validate_matched_core_feature_contract(features)
    with pytest.raises(RuntimeError, match="exact 71-feature"):
        _validate_matched_core_feature_contract(features[:-1])


def _verified_selection_seal() -> dict[str, object]:
    seal: dict[str, object] = {
        "schema_version": DECISION_SELECTION_SEAL_SCHEMA_VERSION,
        "economics_opened": False,
        "selection_uses_economics": False,
        "selection": _selected_configuration(),
    }
    seal["selection_identity_sha256"] = hashlib.sha256(
        json.dumps(
            seal,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return seal


@pytest.mark.parametrize(
    ("candidate_name", "control_name", "candidate_count", "control_count"),
    (
        (CORE_ORACLE_PRICE, ORACLE_MATCHED_CORE_PRICE_CONTROL, 75, 71),
        (CORE_CANDLES_PRICE, CANDLE_MATCHED_CORE_PRICE_CONTROL, 79, 71),
        (
            CORE_ORACLE_L2_PRICE,
            THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL,
            115,
            75,
        ),
    ),
)
def test_post_selection_attribution_replays_exact_source_cohort_probability_only(
    monkeypatch: pytest.MonkeyPatch,
    candidate_name: str,
    control_name: str,
    candidate_count: int,
    control_count: int,
) -> None:
    config = _config()
    core_config = load_core_config(config.core_config)
    primary_oof, _ = _synthetic_oof()
    frame = _matched_replay_frame(primary_oof)
    fit_calls: list[dict[str, object]] = []
    calibration_calls: list[dict[str, object]] = []

    def fake_fit(
        fit_frame: pl.DataFrame,
        features: tuple[str, ...],
        candidate: object,
        *_args: object,
    ) -> tuple[object, dict[str, object]]:
        fit_calls.append(
            {
                "frame": fit_frame,
                "features": features,
                "candidate": candidate,
            }
        )
        return object(), {"fit_rows": fit_frame.height}

    def fake_calibration(
        _model: object,
        calibration_frame: pl.DataFrame,
        *_args: object,
        base_candidate: str,
        variant: object,
    ) -> tuple[_MatchedCoreFakeBundle, dict[str, object]]:
        calibration_calls.append(
            {
                "frame": calibration_frame,
                "base_candidate": base_candidate,
                "variant": variant,
            }
        )
        return _MatchedCoreFakeBundle(), {"target_calibration": {"qualified": True}}

    monkeypatch.setattr(
        "btc_directional_model.asymmetric_decision_quality.fit_hybrid_histogram_model",
        fake_fit,
    )
    monkeypatch.setattr(
        "btc_directional_model.asymmetric_decision_quality._fit_calibrated_bundle",
        fake_calibration,
    )

    oof_by_arm, evidence = fit_selected_attribution_pair_walk_forward(
        frame,
        _verified_selection_seal(),
        config,
        core_config,
        feature_set_name=candidate_name,
    )

    assert POST_SELECTION_ATTRIBUTION_PAIRS[candidate_name] == (
        control_name,
        candidate_count,
        control_count,
    )
    assert set(oof_by_arm) == {candidate_name, control_name}
    assert len(fit_calls) == 10
    assert len(calibration_calls) == 10
    expected_features = asymmetric_value_feature_sets()
    assert sum(call["features"] == expected_features[candidate_name] for call in fit_calls) == 5
    assert sum(call["features"] == expected_features[control_name] for call in fit_calls) == 5
    assert all(call["candidate"].name == "hybrid_25_h3" for call in fit_calls)  # type: ignore[union-attr]
    assert all(call["candidate"].target_weight == 0.25 for call in fit_calls)  # type: ignore[union-attr]
    assert all(call["candidate"].histogram_profile == "h3_regularized" for call in fit_calls)  # type: ignore[union-attr]
    assert all(call["variant"].parent_source == "alltime" for call in calibration_calls)  # type: ignore[union-attr]
    assert all(call["variant"].identity_l2 == 1.0 for call in calibration_calls)  # type: ignore[union-attr]
    assert evidence["source_cohort"]["canonical_feature_set_name"] == candidate_name
    assert evidence["source_cohort"]["rows"] == frame.height
    assert evidence["candidate"]["feature_count"] == candidate_count
    assert evidence["matched_control"]["feature_count"] == control_count
    assert evidence["selection_eligible"] is False
    assert evidence["probability_only"] is True
    assert evidence["economics_used"] is False
    assert evidence["all_calibrations_qualified"] is True
    assert evidence["exact_within_pair_grid"] is True
    assert evidence["candidate_grid_key_sha256"] == evidence["control_grid_key_sha256"]
    assert evidence["candidate_minus_matched_control_probability"]["brier_delta"][
        "point"
    ] == pytest.approx(0.0)
    for arm_name, arm in oof_by_arm.items():
        assert tuple(arm.columns) == OOF_SELECTION_COLUMNS
        assert arm["candidate_id"].unique().to_list() == [
            post_selection_attribution_candidate_id(arm_name)
        ]
        assert arm["base_candidate"].unique().to_list() == [arm_name]
    _validate_attribution_pair_oof_grid(
        oof_by_arm[candidate_name],
        oof_by_arm[control_name],
        candidate_name=candidate_name,
        control_name=control_name,
        variant=calibration_calls[0]["variant"],  # type: ignore[arg-type]
        config=config,
    )
    corrupted = (
        oof_by_arm[control_name]
        .with_row_index("_row")
        .with_columns(
            pl.when(pl.col("_row") == 0)
            .then(1 - pl.col("label_up"))
            .otherwise(pl.col("label_up"))
            .alias("label_up")
        )
        .drop("_row")
    )
    with pytest.raises(RuntimeError, match="exact keys, labels, and eligibility"):
        _validate_attribution_pair_oof_grid(
            oof_by_arm[candidate_name],
            corrupted,
            candidate_name=candidate_name,
            control_name=control_name,
            variant=calibration_calls[0]["variant"],  # type: ignore[arg-type]
            config=config,
        )


@pytest.mark.parametrize("candidate_name", tuple(POST_SELECTION_ATTRIBUTION_PAIRS))
def test_final_post_selection_attribution_uses_frozen_windows_and_feature_counts(
    monkeypatch: pytest.MonkeyPatch,
    candidate_name: str,
) -> None:
    config = _config()
    core_config = load_core_config(config.core_config)
    primary_oof, _ = _synthetic_oof()
    frame = _matched_replay_frame(primary_oof)
    calls: list[dict[str, object]] = []

    def fake_fit(
        fit_frame: pl.DataFrame,
        features: tuple[str, ...],
        candidate: object,
        *_args: object,
    ) -> tuple[object, dict[str, object]]:
        calls.append({"fit_frame": fit_frame, "features": features, "candidate": candidate})
        return object(), {"fit_rows": fit_frame.height}

    def fake_calibration(
        _model: object,
        calibration_frame: pl.DataFrame,
        *_args: object,
        base_candidate: str,
        variant: object,
    ) -> tuple[_MatchedCoreFakeBundle, dict[str, object]]:
        calls.append(
            {
                "calibration_frame": calibration_frame,
                "base_candidate": base_candidate,
                "variant": variant,
            }
        )
        return _MatchedCoreFakeBundle(), {"target_calibration": {"qualified": True}}

    monkeypatch.setattr(
        "btc_directional_model.asymmetric_decision_quality.fit_hybrid_histogram_model",
        fake_fit,
    )
    monkeypatch.setattr(
        "btc_directional_model.asymmetric_decision_quality._fit_calibrated_bundle",
        fake_calibration,
    )

    models, evidence = fit_final_selected_attribution_pair(
        frame,
        _verified_selection_seal(),
        config,
        core_config,
        feature_set_name=candidate_name,
    )

    contract = config.decision_quality
    assert contract is not None
    control_name, candidate_count, control_count = POST_SELECTION_ATTRIBUTION_PAIRS[candidate_name]
    assert set(models) == {candidate_name, control_name}
    assert models[candidate_name].name == candidate_name
    assert models[control_name].name == control_name
    assert evidence["arms"][candidate_name]["feature_count"] == candidate_count
    assert evidence["arms"][control_name]["feature_count"] == control_count
    assert evidence["final_fit_window"] == {
        "start": contract.final_fit.start.isoformat(),
        "end": contract.final_fit.end.isoformat(),
    }
    assert evidence["final_calibration_window"] == {
        "start": contract.final_calibration.start.isoformat(),
        "end": contract.final_calibration.end.isoformat(),
    }
    fit_frames = [call["fit_frame"] for call in calls if "fit_frame" in call]
    calibration_frames = [
        call["calibration_frame"] for call in calls if "calibration_frame" in call
    ]
    assert len(fit_frames) == 2
    assert len(calibration_frames) == 2
    assert all(frame_["window_start"].max() < contract.final_fit.end for frame_ in fit_frames)  # type: ignore[index]
    assert all(
        frame_["window_start"].min() >= contract.final_calibration.start
        for frame_ in calibration_frames
    )  # type: ignore[index]


def test_post_selection_attribution_requires_unopened_verified_seal() -> None:
    config = _config()
    core_config = load_core_config(config.core_config)
    primary_oof, _ = _synthetic_oof()
    frame = _matched_replay_frame(primary_oof)

    with pytest.raises(RuntimeError, match="verified decision seal"):
        fit_selected_attribution_pair_walk_forward(
            frame,
            {**_verified_selection_seal(), "schema_version": "wrong"},
            config,
            core_config,
            feature_set_name=CORE_ORACLE_PRICE,
        )
    with pytest.raises(RuntimeError, match="economics to remain sealed"):
        fit_selected_attribution_pair_walk_forward(
            frame,
            {**_verified_selection_seal(), "economics_opened": True},
            config,
            core_config,
            feature_set_name=CORE_ORACLE_PRICE,
        )
    with pytest.raises(ValueError, match="unsupported post-selection attribution"):
        fit_selected_attribution_pair_walk_forward(
            frame,
            _verified_selection_seal(),
            config,
            core_config,
            feature_set_name=L2_MATCHED_CORE_PRICE_CONTROL,
        )
