from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import btc_directional_model.asymmetric_decision_quality as quality
import btc_directional_model.asymmetric_decision_quality_benchmark as benchmark
import btc_directional_model.asymmetric_value_benchmark as value_benchmark
from btc_directional_model.asymmetric_decision_quality import OOF_SELECTION_COLUMNS
from btc_directional_model.asymmetric_value_config import load_asymmetric_value_config
from btc_directional_model.asymmetric_value_training import CORE_L2_PRICE
from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_extract import file_sha256, write_json_atomic


def _config(tmp_path: Path | None = None):
    source = (
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-early-no-calibration-20260414-20260820.toml"
    )
    config = load_asymmetric_value_config(source)
    if tmp_path is None:
        return config
    return replace(
        config,
        feature_cache=tmp_path / "features",
        runs=tmp_path / "runs",
    )


def _complete_support_frame() -> pl.DataFrame:
    start = datetime(2026, 4, 14, tzinfo=UTC)
    end = datetime(2026, 8, 20, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    day = start
    while day < end:
        day_index = (day - start).days
        for market_index in range(200):
            yes_target = market_index < 100
            market_id = f"d{day_index:03d}-m{market_index:03d}"
            window_start = day + timedelta(minutes=5 * market_index)
            for second in (1, 15, 30, 45, 60, 90, 120, 180):
                rows.append(
                    {
                        "market_id": market_id,
                        "window_start": window_start,
                        "seconds_elapsed": second,
                        "label_up": market_index % 2,
                        "yes_ask_vwap_5": 0.25 if yes_target else 0.75,
                        "no_ask_vwap_5": 0.75 if yes_target else 0.25,
                    }
                )
        day += timedelta(days=1)
    return pl.DataFrame(rows)


def test_new_day_support_freezes_ten_days_and_early_no_outcomes() -> None:
    evidence = quality.audit_decision_quality_walk_forward_support(
        _complete_support_frame(),
        _config(),
    )

    assert evidence["passed"] is True
    assert len(evidence["folds"]) == 10
    aggregate = evidence["aggregate_validation"]
    assert aggregate["validation_utc_days"] == 10
    assert aggregate["strict_markets"] == 2_000
    assert aggregate["early_no"]["rows"] >= 200
    assert aggregate["early_no"]["markets"] >= 50
    assert aggregate["early_no"]["utc_days"] == 10
    assert aggregate["early_no"]["positive_markets"] >= 15
    assert aggregate["early_no"]["negative_markets"] >= 15
    early_cells = [
        cell
        for fold in evidence["folds"]
        for cell in fold["target_cells"]
        if cell["side"] == "NO" and cell["start_second"] == 1
    ]
    assert all(cell["minimum_markets"] == 100 for cell in early_cells)
    assert all(cell["minimum_utc_days"] == 14 for cell in early_cells)
    assert all(cell["minimum_markets_per_outcome"] == 25 for cell in early_cells)


def test_empty_new_day_validation_cell_blocks_before_any_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    frame = _complete_support_frame().filter(
        ~(
            (
                pl.col("window_start").dt.date()
                == datetime(2026, 8, 19, tzinfo=UTC).date()
            )
            & (pl.col("seconds_elapsed") == 45)
            & (pl.col("yes_ask_vwap_5") == 0.25)
        )
    )
    monkeypatch.setattr(
        quality,
        "fit_hybrid_histogram_model",
        lambda *args, **kwargs: pytest.fail("fit ran after empty validation cell"),
    )

    with pytest.raises(
        RuntimeError,
        match="aug19_aug20:validation_cell:YES:45-60",
    ):
        quality.fit_decision_quality_walk_forward(
            frame,
            config,
            load_core_config(config.core_config),
        )


def test_duplicate_source_key_blocks_before_any_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    source = _complete_support_frame()
    source = pl.concat((source, source.head(1)), how="vertical_relaxed")
    monkeypatch.setattr(
        quality,
        "fit_hybrid_histogram_model",
        lambda *args, **kwargs: pytest.fail("fit ran after duplicate source key"),
    )

    with pytest.raises(RuntimeError, match="source:duplicate_market_time_keys"):
        quality.fit_decision_quality_walk_forward(
            source,
            config,
            load_core_config(config.core_config),
        )


class _FinalBundle:
    def __init__(self) -> None:
        self.name = "fake"


def _selected_early_no_configuration() -> dict[str, object]:
    return {
        "status": "selected",
        "selected_candidate_id": "hybrid_50_h3__targetpool_early_no_offset",
        "selected_base_candidate": "hybrid_50_h3",
        "economics_used": False,
    }


def test_final_model_enforces_and_seals_early_no_calibration_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    fit_calls: list[int] = []

    def fake_fit(frame: pl.DataFrame, *args, **kwargs):
        fit_calls.append(frame.height)
        return object(), {"fit_rows": frame.height}

    monkeypatch.setattr(quality, "fit_hybrid_histogram_model", fake_fit)
    monkeypatch.setattr(
        quality,
        "_fit_calibrated_bundle",
        lambda *args, **kwargs: (
            _FinalBundle(),
            {"target_calibration": {"qualified": True}},
        ),
    )

    bundle, evidence = quality.fit_final_decision_quality_model(
        _complete_support_frame(),
        _selected_early_no_configuration(),
        config,
        load_core_config(config.core_config),
    )

    support = evidence["final_early_no_calibration_support"]
    assert bundle.name == CORE_L2_PRICE
    assert fit_calls
    assert support["passed"] is True
    assert support["minimum_markets"] == 100
    assert support["minimum_utc_days"] == 14
    assert support["minimum_markets_per_outcome"] == 25


def test_final_model_rejects_early_no_support_before_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    contract = config.decision_quality
    assert contract is not None
    source = _complete_support_frame().filter(
        ~(
            pl.col("window_start").is_between(
                contract.final_calibration.start,
                contract.final_calibration.end,
                closed="left",
            )
            & pl.col("seconds_elapsed").is_between(1, 15, closed="left")
            & pl.col("no_ask_vwap_5").is_between(0.20, 0.30, closed="left")
        )
    )
    monkeypatch.setattr(
        quality,
        "fit_hybrid_histogram_model",
        lambda *args, **kwargs: pytest.fail("fit ran without final early-NO support"),
    )

    with pytest.raises(
        RuntimeError,
        match="selected final early-NO calibration support did not qualify",
    ):
        quality.fit_final_decision_quality_model(
            source,
            _selected_early_no_configuration(),
            config,
            load_core_config(config.core_config),
        )


def _ranking_frame(*candidate_ids: str) -> pl.DataFrame:
    start = datetime(2026, 8, 10, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for candidate_id in candidate_ids:
        for day in range(10):
            for market in range(4):
                label = market % 2
                window_start = start + timedelta(days=day, minutes=5 * market)
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "fold": f"day-{day}",
                        "market_id": f"d{day}-m{market}",
                        "window_start": window_start,
                        "observed_at": window_start + timedelta(seconds=1),
                        "seconds_elapsed": 1,
                        "label_up": label,
                        "probability_yes": 0.75 if label else 0.25,
                        "yes_target_eligible": True,
                        "no_target_eligible": True,
                        "target_time_band": "1_15",
                    }
                )
    return pl.DataFrame(rows)


def _rank_record(
    candidate_id: str,
    *,
    early_brier: float,
    early_bias: float,
    overall_brier: float,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "histogram_profile": "h3_regularized",
        "identity_l2": 1.0,
        "target_weight": 0.5,
        "metrics": {
            "overall": {"log_loss": 0.55, "brier": overall_brier, "bias": 0.0},
            "time_cells": {
                "NO_1_15": {"brier": early_brier, "bias": early_bias}
            },
        },
    }


def test_early_no_ranking_uses_one_se_then_early_quality_and_control_tie() -> None:
    control = "hybrid_50_h3__targetpool_control"
    challenger = "hybrid_50_h3__targetpool_early_no_offset"
    frame = _ranking_frame(control, challenger)
    ranked = quality._rank_decision_quality_candidates(
        [
            _rank_record(control, early_brier=0.18, early_bias=0.02, overall_brier=0.20),
            _rank_record(
                challenger,
                early_brier=0.17,
                early_bias=0.01,
                overall_brier=0.21,
            ),
        ],
        frame,
        resamples=1_000,
        seed=7,
        early_no_study=True,
        lead_control_id=control,
    )
    assert ranked[0]["candidate_id"] == challenger

    tied = quality._rank_decision_quality_candidates(
        [
            _rank_record(control, early_brier=0.18, early_bias=0.02, overall_brier=0.20),
            _rank_record(
                challenger,
                early_brier=0.18,
                early_bias=0.02,
                overall_brier=0.20,
            ),
        ],
        frame,
        resamples=1_000,
        seed=7,
        early_no_study=True,
        lead_control_id=control,
    )
    assert tied[0]["candidate_id"] == control
    assert tied[0]["lead_control_tie_preferred"] is True


def _four_arm_oof() -> tuple[pl.DataFrame, dict[str, object]]:
    config = _config()
    contract = config.decision_quality
    assert contract is not None
    broad_id = "broad_current__alltime_l2_1_0"
    target_id = "target_only_current__alltime_l2_1_0"
    arm_ids = {
        variant.name: quality.calibration_variant_id("hybrid_50_h3", variant)
        for variant in contract.calibration_variants
    }
    candidate_ids = [broad_id, target_id, *arm_ids.values()]
    rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(contract.folds):
        for side in ("YES", "NO"):
            for band_index, (start_second, _) in enumerate(contract.time_strata):
                for market_index in range(20):
                    label = market_index % 2
                    market_id = (
                        f"f{fold_index}-{side}-{band_index}-m{market_index}"
                    )
                    window_start = fold.validation.start + timedelta(
                        minutes=market_index
                    )
                    for candidate_id in candidate_ids:
                        bias = 0.0
                        if candidate_id == broad_id:
                            strength = 0.90
                        elif candidate_id == target_id:
                            strength = 0.92
                        elif candidate_id == arm_ids["targetpool_control"]:
                            strength = 0.97
                            bias = 0.022
                        elif candidate_id == arm_ids["targetpool_day_balanced"]:
                            strength = 0.97
                            bias = 0.012
                        elif candidate_id == arm_ids["targetpool_early_no_offset"]:
                            strength = 0.98
                        else:
                            strength = 0.975
                        probability_yes = strength if label else 1.0 - strength
                        if side == "NO" and band_index == 0 and bias:
                            probability_no = (
                                strength if label == 0 else 1.0 - strength
                            ) + bias
                            probability_yes = 1.0 - probability_no
                        rows.append(
                            {
                                "candidate_id": candidate_id,
                                "base_candidate": candidate_id.split("__", 1)[0],
                                "fold": fold.name,
                                "parent_source": (
                                    "alltime"
                                    if candidate_id in {broad_id, target_id}
                                    else "targetpool"
                                ),
                                "identity_l2": 1.0,
                                "market_id": market_id,
                                "window_start": window_start,
                                "observed_at": window_start
                                + timedelta(seconds=start_second),
                                "seconds_elapsed": start_second,
                                "label_up": label,
                                "probability_yes": probability_yes,
                                "yes_target_eligible": side == "YES",
                                "no_target_eligible": side == "NO",
                                "target_time_band": (
                                    "1_15",
                                    "15_30",
                                    "30_45",
                                    "45_56",
                                )[band_index],
                            }
                        )
    profiles: dict[str, object] = {}
    for fold in contract.folds:
        calibrations = {
            candidate_id: {"target_calibration": {"qualified": True}}
            for candidate_id in arm_ids.values()
        }
        profiles[fold.name] = {
            "candidates": {
                "hybrid_50_h3": {"calibrations": calibrations},
                "broad_current": {"calibrations": {}},
                "target_only_current": {"calibrations": {}},
            }
        }
    return pl.DataFrame(rows).select(*OOF_SELECTION_COLUMNS), profiles


def test_four_arm_selection_enforces_early_no_control_gates_and_stability() -> None:
    oof, profiles = _four_arm_oof()
    selection = quality.select_decision_quality_candidate(oof, profiles, _config())

    assert len(selection["candidate_records"]) == 4
    assert selection["selected_candidate_id"] == (
        "hybrid_50_h3__targetpool_early_no_offset"
    )
    records = {
        record["candidate_id"]: record
        for record in selection["candidate_records"]
    }
    control = records["hybrid_50_h3__targetpool_control"]
    challenger = records["hybrid_50_h3__targetpool_early_no_offset"]
    weak_bias = records["hybrid_50_h3__targetpool_day_balanced"]
    control_gates = {gate["name"]: gate for gate in control["gates"]}
    challenger_gates = {gate["name"]: gate for gate in challenger["gates"]}
    weak_gates = {gate["name"]: gate for gate in weak_bias["gates"]}

    assert control_gates["challenger_early_no_bias_improvement"]["passed"] is True
    assert challenger_gates["challenger_early_no_bias_improvement"]["observed"] >= 0.02
    assert challenger_gates["early_no_brier_delta_point_noninferior_to_lead_control"][
        "threshold"
    ] == 0.0
    assert challenger_gates[
        "early_no_log_loss_delta_upper_95_noninferior_to_lead_control"
    ]["threshold"] == 0.005
    assert challenger_gates["early_no_noninferior_folds_to_lead_control"][
        "observed"
    ] == 10
    assert challenger_gates["noninferior_folds_to_all_controls"]["observed"] == 10
    assert weak_gates["challenger_early_no_bias_improvement"]["passed"] is False


def _blocked_readiness(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    payload: dict[str, object] = {
        "ready": False,
        "status": "blocked_source_readiness",
        "blocking_statuses": ["missing_complete_day"],
    }
    path = tmp_path / "new-day-readiness.json"
    write_json_atomic(path, payload)
    return path, payload


def test_unready_new_day_cohort_seals_block_without_fitting_or_economics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        benchmark,
        "fit_decision_quality_walk_forward",
        lambda *args, **kwargs: pytest.fail("fit ran before new-day readiness"),
    )
    monkeypatch.setattr(
        benchmark,
        "reveal_decision_quality_economics",
        lambda **kwargs: pytest.fail("economics opened before new-day readiness"),
    )

    run_dir, result = benchmark.run_decision_quality_benchmark(
        config=config,
        core_config=object(),
        development_model_frames={},
        oof_core=pl.DataFrame(),
        oof_grid_coverage={},
        development_coverage={},
        development_price_manifest=None,
        implementation_sha256="a" * 64,
        dependency_versions={},
        current_process={},
        readiness=_blocked_readiness(tmp_path),
    )

    assert result["selection"]["status"] == "blocked_source_readiness"
    assert result["evaluation"]["status"] == "blocked_source_readiness"
    assert result["development"]["economic_reveal_status"] == "not_opened"
    assert not (run_dir / "decision-quality-oof-predictions.parquet").exists()
    assert not (run_dir / "development-economic-reveal.json").exists()
    report = (run_dir / "benchmark-report.md").read_text()
    assert "No model was selected" in report
    assert "The model was selected" not in report
    sealed = benchmark.load_verified_decision_selection_seal(
        run_dir / "decision-selection-seal.json"
    )
    assert sealed["selection"]["status"] == "blocked_source_readiness"
    assert sealed["selection"]["readiness_failures"] == ["missing_complete_day"]
    forward = result["forward_requirements"]
    assert forward["minimum_complete_utc_days"] == 21
    assert forward["minimum_probability_noninferior_utc_days"] == 17
    assert forward["minimum_strict_markets"] == 2_000
    assert forward["minimum_selected_trades"] == 200
    assert forward["minimum_early_no_opportunity_rows"] == 100
    assert forward["executable_forward_subsystem_added"] is False


def test_early_no_runner_never_falls_back_to_legacy_readiness(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="injected sealed new-day readiness"):
        benchmark.run_decision_quality_benchmark(
            config=_config(tmp_path),
            core_config=object(),
            development_model_frames={},
            oof_core=pl.DataFrame(),
            oof_grid_coverage={},
            development_coverage={},
            development_price_manifest=None,
            implementation_sha256="a" * 64,
            dependency_versions={},
            current_process={},
        )


def test_frequency_uses_sealed_history_and_same_new_day_lead_denominator() -> None:
    historical, lead = benchmark._early_no_frequency_checks(
        candidate_trades=200,
        lead_control_trades=250,
        eligible_resolved_markets=2_000,
    )

    assert historical["observed"] == pytest.approx(0.10)
    assert historical["threshold"] == pytest.approx(0.03901625677365569)
    assert historical["historical_incumbent_trades"] == 117
    assert historical["historical_incumbent_eligible_markets"] == 2_399
    assert historical["historical_incumbent_artifact_provenance_sha256"] == (
        "40e89cfd77165d7fee57715e2988d1862c175561cb19c6e84adf5d01e031fa11"
    )
    assert historical["common_market_replay_claimed"] is False
    assert lead["threshold"] == pytest.approx(0.10)
    assert lead["passed"] is True
    assert lead["same_new_day_eligible_cohort"] is True
    assert lead["common_market_replay_claimed"] is False


def _single_oof(candidate_id: str) -> pl.DataFrame:
    window_start = datetime(2026, 8, 10, tzinfo=UTC)
    return pl.DataFrame(
        {
            "candidate_id": [candidate_id],
            "base_candidate": ["hybrid_50_h3"],
            "fold": ["aug10_aug11"],
            "parent_source": ["targetpool"],
            "identity_l2": [1.0],
            "market_id": ["m1"],
            "window_start": [window_start],
            "observed_at": [window_start + timedelta(seconds=1)],
            "seconds_elapsed": [1],
            "label_up": [0],
            "probability_yes": [0.20],
            "yes_target_eligible": [False],
            "no_target_eligible": [True],
            "target_time_band": ["1_15"],
        }
    ).select(*OOF_SELECTION_COLUMNS)


def _ready_readiness(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    payload: dict[str, object] = {
        "ready": True,
        "status": "ready",
        "blocking_statuses": [],
        "readiness_identity_sha256": "b" * 64,
        "payload_sha256": "c" * 64,
        "validation": {
            "range_start": "2026-08-10T00:00:00+00:00",
            "range_end": "2026-08-20T00:00:00+00:00",
        },
        "external_archive": {"required_for": []},
    }
    path = tmp_path / "ready.json"
    write_json_atomic(path, payload)
    return path, payload


def test_ready_runner_skips_source_attribution_and_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    selected_id = "hybrid_50_h3__targetpool_early_no_offset"
    selected = _single_oof(selected_id)
    selection = {
        "status": "selected",
        "selected_candidate_id": selected_id,
        "selected_base_candidate": "hybrid_50_h3",
        "candidate_records": [],
        "rank_trace": [{"candidate_id": selected_id}],
        "economics_used": False,
        "lead_control_candidate_id": "hybrid_50_h3__targetpool_control",
    }
    monkeypatch.setattr(
        benchmark,
        "fit_decision_quality_walk_forward",
        lambda *args, **kwargs: (
            selected,
            {"selection": selection, "walk_forward_source_support": {"passed": True}},
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "fit_selected_matched_core_walk_forward",
        lambda *args, **kwargs: (
            _single_oof(quality.MATCHED_CORE_CONTROL_CANDIDATE_ID),
            {},
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "fit_final_decision_quality_model",
        lambda *args, **kwargs: ({"model": "selected"}, {}),
    )
    monkeypatch.setattr(
        benchmark,
        "fit_final_matched_core_model",
        lambda *args, **kwargs: ({"model": "core"}, {}),
    )
    matched = {
        "utc_days": 10,
        "rows": 1,
        "brier_delta": {
            "point": 0.0,
            "lower_95": 0.0,
            "upper_95": 0.0,
            "standard_error": 0.0,
        },
        "log_loss_delta": {
            "point": 0.0,
            "lower_95": 0.0,
            "upper_95": 0.0,
            "standard_error": 0.0,
        },
    }
    monkeypatch.setattr(benchmark, "paired_probability_delta", lambda *args, **kwargs: matched)
    monkeypatch.setattr(
        benchmark,
        "_fit_post_selection_attribution",
        lambda **kwargs: pytest.fail("early-NO source attribution was executed"),
    )

    def stop_at_reveal(**kwargs):
        assert kwargs["attribution_manifest_path"] is None
        assert kwargs["oracle_frame"] is None
        raise RuntimeError("early-NO sealed reveal boundary")

    monkeypatch.setattr(benchmark, "reveal_decision_quality_economics", stop_at_reveal)
    with pytest.raises(RuntimeError, match="sealed reveal boundary"):
        benchmark.run_decision_quality_benchmark(
            config=config,
            core_config=object(),
            development_model_frames={CORE_L2_PRICE: pl.DataFrame()},
            oof_core=pl.DataFrame(),
            oof_grid_coverage={},
            development_coverage={},
            development_price_manifest={"created_at": "ignored"},
            implementation_sha256="a" * 64,
            dependency_versions={},
            current_process={},
            readiness=_ready_readiness(tmp_path),
        )


def _economic_seal(
    tmp_path: Path,
    *,
    selected_id: str,
    lead_id: str,
    matched: dict[str, object],
) -> Path:
    selection = {
        "status": "selected",
        "selected_candidate_id": selected_id,
        "selected_base_candidate": "hybrid_50_h3",
        "economics_used": False,
    }
    selection_path = tmp_path / "decision-quality-selection.json"
    write_json_atomic(selection_path, {"selection": selection})
    oof = pl.concat(
        (_single_oof(selected_id), _single_oof(lead_id)),
        how="vertical_relaxed",
    )
    oof_path = tmp_path / "decision-quality-oof-predictions.parquet"
    oof.write_parquet(oof_path)
    matched_path = tmp_path / "matched-core-oof-predictions.parquet"
    _single_oof(quality.MATCHED_CORE_CONTROL_CANDIDATE_ID).write_parquet(
        matched_path
    )
    seal_path, _ = benchmark.write_decision_selection_seal(
        tmp_path,
        {
            "run_id": "economic",
            "selection": selection,
            "selection_artifact": selection_path.name,
            "oof_artifact": oof_path.name,
            "matched_core_oof_artifact": matched_path.name,
            "matched_core_probability": matched,
            "economic_reveal_candidate_ids": [selected_id, lead_id],
            "oof": {
                "rows": oof.height,
                "markets": oof["market_id"].n_unique(),
                "key_sha256": quality.decision_quality_oof_key_digest(oof),
                "content_sha256": file_sha256(oof_path),
            },
            "artifact_sha256": {
                selection_path.name: file_sha256(selection_path),
                oof_path.name: file_sha256(oof_path),
                matched_path.name: file_sha256(matched_path),
            },
            "selection_uses_economics": False,
        },
    )
    return seal_path


def test_economic_reveal_opens_only_sealed_arms_and_uses_joint_strict_denominator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    selected_id = "hybrid_50_h3__targetpool_early_no_offset"
    lead_id = "hybrid_50_h3__targetpool_control"
    matched: dict[str, object] = {
        "brier_delta": {
            "point": 0.0,
            "lower_95": 0.0,
            "upper_95": 0.0,
        },
        "log_loss_delta": {
            "point": 0.0,
            "lower_95": 0.0,
            "upper_95": 0.0,
        },
    }
    seal_path = _economic_seal(
        tmp_path,
        selected_id=selected_id,
        lead_id=lead_id,
        matched=matched,
    )
    monkeypatch.setattr(
        benchmark,
        "paired_probability_delta",
        lambda *args, **kwargs: matched,
    )
    opened_models: list[str] = []

    def fake_join(*args, model: str, **kwargs):
        opened_models.append(model)
        return pl.DataFrame({"model": [model]})

    monkeypatch.setattr(benchmark, "_join_oof_probability_to_execution", fake_join)
    arm_trades = {
        selected_id: 200,
        lead_id: 250,
        quality.MATCHED_CORE_CONTROL_CANDIDATE_ID: 180,
    }

    def fake_economics(predictions: pl.DataFrame, config):
        model = predictions["model"][0]
        trades = arm_trades[model]
        ledger = pl.DataFrame({"trade": list(range(trades))})
        return predictions, ledger, {"trades": trades, "net_profit": trades / 10}

    monkeypatch.setattr(benchmark, "_economic_model_evidence", fake_economics)
    monkeypatch.setattr(benchmark, "policy_gate_checks", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        benchmark,
        "selected_win_rate_advantage_gate_checks",
        lambda *args, **kwargs: [],
    )
    observed_source_markets: list[int] = []

    def evidence_checks(frame: pl.DataFrame, *args, **kwargs):
        observed_source_markets.append(frame["market_id"].n_unique())
        return []

    monkeypatch.setattr(benchmark, "evidence_gate_checks", evidence_checks)
    monkeypatch.setattr(benchmark, "_matched_probability_checks", lambda *args: [])
    monkeypatch.setattr(benchmark, "_matched_economic_checks", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        benchmark,
        "temporal_confirmation_ablation",
        lambda *args, **kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        benchmark,
        "vwap10_capacity_policy_ledger",
        lambda *args, **kwargs: (pl.DataFrame(), {}),
    )
    monkeypatch.setattr(benchmark, "rejection_funnel", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        benchmark,
        "accuracy_price_by_second",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        value_benchmark,
        "_candidate_grid_summary",
        lambda *args, **kwargs: {"prediction_grid_coverage": 0.70},
    )
    monkeypatch.setattr(
        value_benchmark,
        "_paired_day_net_difference_bootstrap",
        lambda *args, **kwargs: {"lower_95": 0.0},
    )
    validation_start = datetime(2026, 8, 10, tzinfo=UTC)
    l2_frame = pl.DataFrame(
        {
            "market_id": [f"joint-{index}" for index in range(2_000)],
            "window_start": [validation_start] * 2_000,
        }
    )
    oof_core = pl.DataFrame(
        {
            "market_id": [f"core-{index}" for index in range(3_000)],
            "window_start": [validation_start] * 3_000,
        }
    )

    reveal = benchmark._reveal_early_no_economics(
        seal_path=seal_path,
        l2_frame=l2_frame,
        oof_core=oof_core,
        oof_grid_coverage={"retained_coverage": 0.90, "strict_coverage": 0.70},
        config=config,
    )

    assert opened_models == [
        selected_id,
        lead_id,
        quality.MATCHED_CORE_CONTROL_CANDIDATE_ID,
    ]
    assert observed_source_markets == [2_000]
    evidence = reveal["evidence"]
    assert evidence["selected_metrics"]["eligible_resolved_markets"] == 2_000
    assert evidence["lead_control_metrics"]["eligible_resolved_markets"] == 2_000
    assert evidence["matched_core_metrics"]["eligible_resolved_markets"] == 2_000
    assert evidence["evidence_scope"] == benchmark.EARLY_NO_DEVELOPMENT_OOF_GATE_SCOPE
    assert evidence["evidence_scope"]["minimum_trades"] == 200
    assert evidence["evidence_scope"]["minimum_trade_utc_days"] == 10
    assert evidence["evidence_scope"]["minimum_strict_markets"] == 2_000
    assert evidence["evidence_scope"]["minimum_strict_grid_coverage"] == 0.70
    assert evidence["evidence_scope"]["minimum_candidate_grid_coverage"] == 0.70
    assert evidence["frequency"]["eligible_market_count"] == 2_000
    assert evidence["frequency"]["common_market_replay_claimed"] is False
    assert evidence["source_attribution_executed"] is False


def test_outer_runner_blocks_before_any_source_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    readiness_result = _blocked_readiness(tmp_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(value_benchmark, "_implementation_digest", lambda *_: "a" * 64)
    monkeypatch.setattr(value_benchmark, "_dependency_versions", dict)
    monkeypatch.setattr(value_benchmark, "load_core_config", lambda *_: object())
    monkeypatch.setattr(value_benchmark, "_current_process_contract", lambda *_: {})
    monkeypatch.setattr(
        value_benchmark,
        "load_new_day_readiness_contract",
        lambda *_: object(),
    )
    monkeypatch.setattr(
        value_benchmark,
        "prepare_new_day_training_readiness",
        lambda *args, **kwargs: readiness_result,
    )
    monkeypatch.setattr(
        value_benchmark,
        "extract_core_source",
        lambda *args, **kwargs: pytest.fail("Core extraction preceded readiness"),
    )

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return tmp_path / "blocked-run", {"status": "blocked"}

    monkeypatch.setattr(benchmark, "run_decision_quality_benchmark", fake_runner)

    run_dir, result = value_benchmark.run_asymmetric_value_benchmark(config)

    assert run_dir == tmp_path / "blocked-run"
    assert result == {"status": "blocked"}
    assert captured["readiness"] == readiness_result
    assert captured["development_model_frames"] == {}
    assert captured["development_price_manifest"] is None


def test_ready_outer_runner_materializes_only_core_l2_and_exact_pm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    readiness_path = tmp_path / "ready.json"
    write_json_atomic(readiness_path, {"ready": True})
    readiness_result = (readiness_path, {"ready": True})
    validation_start = datetime(2026, 8, 10, tzinfo=UTC)
    development = pl.DataFrame(
        {
            "market_id": ["market-1"],
            "window_start": [validation_start],
            "seconds_elapsed": [1],
            "observed_at": [validation_start + timedelta(seconds=1)],
            "label_up": [1],
        }
    )
    captured: dict[str, object] = {}
    observed_source_families: list[str] = []

    monkeypatch.setattr(value_benchmark, "_implementation_digest", lambda *_: "a" * 64)
    monkeypatch.setattr(value_benchmark, "_dependency_versions", dict)
    monkeypatch.setattr(value_benchmark, "load_core_config", lambda *_: object())
    monkeypatch.setattr(value_benchmark, "_current_process_contract", lambda *_: {})
    monkeypatch.setattr(
        value_benchmark,
        "load_new_day_readiness_contract",
        lambda *_: object(),
    )
    monkeypatch.setattr(
        value_benchmark,
        "prepare_new_day_training_readiness",
        lambda *args, **kwargs: readiness_result,
    )
    monkeypatch.setattr(value_benchmark, "extract_core_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(value_benchmark, "build_core_features", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        value_benchmark,
        "_load_asymmetric_core_grid",
        lambda *args, **kwargs: development,
    )
    monkeypatch.setattr(
        value_benchmark,
        "extract_asymmetric_price_evidence",
        lambda *args, **kwargs: {"proxy_prices_used": False},
    )
    monkeypatch.setattr(
        value_benchmark,
        "load_asymmetric_price_evidence",
        lambda *args, **kwargs: pl.DataFrame(),
    )
    monkeypatch.setattr(
        value_benchmark,
        "attach_asymmetric_value_features",
        lambda frame, *args, **kwargs: frame,
    )
    monkeypatch.setattr(
        value_benchmark,
        "_base_coverage_summary",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        value_benchmark,
        "_project_candidate_source",
        lambda frame, *args, **kwargs: frame,
    )

    def fake_source(frame, config, *, source_family, **kwargs):
        observed_source_families.append(source_family)
        return frame

    monkeypatch.setattr(value_benchmark, "_load_or_build_source_features", fake_source)
    monkeypatch.setattr(
        value_benchmark,
        "_add_joint_source_coverage",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        value_benchmark,
        "execution_grid_coverage",
        lambda *args, **kwargs: {"strict_coverage": 1.0},
    )
    monkeypatch.setattr(
        value_benchmark,
        "oracle_source_inventory",
        lambda *args, **kwargs: pytest.fail("Oracle materialized in early-NO study"),
    )

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return tmp_path / "ready-run", {"status": "ready"}

    monkeypatch.setattr(benchmark, "run_decision_quality_benchmark", fake_runner)

    run_dir, result = value_benchmark.run_asymmetric_value_benchmark(config)

    assert run_dir == tmp_path / "ready-run"
    assert result == {"status": "ready"}
    assert observed_source_families == ["l2"]
    assert set(captured["development_model_frames"]) == {CORE_L2_PRICE}
    assert captured["readiness"] == readiness_result
    assert captured["development_price_manifest"] == {"proxy_prices_used": False}
