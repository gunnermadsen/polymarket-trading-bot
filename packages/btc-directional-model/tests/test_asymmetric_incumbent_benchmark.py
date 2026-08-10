from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

from btc_directional_model import asymmetric_incumbent_benchmark as benchmark_module
from btc_directional_model.asymmetric_incumbent_calibration import (
    DEFAULT_INCUMBENT_CALIBRATION_CONFIG,
    load_incumbent_calibration_config,
)
from btc_directional_model.asymmetric_incumbent_estimator import (
    ESTIMATOR_FALLBACK_CANDIDATES,
)
from btc_directional_model.asymmetric_incumbent_replay import (
    load_frozen_asymmetric_incumbent,
)


def _ten_day_frame() -> pl.DataFrame:
    start = datetime(2026, 7, 23, tzinfo=UTC)
    return pl.DataFrame(
        {
            "market_id": [f"m{index}" for index in range(10)],
            "window_start": [start + timedelta(days=index) for index in range(10)],
            "observed_at": [start + timedelta(days=index, seconds=5) for index in range(10)],
            "seconds_elapsed": [5] * 10,
            "label_up": [index % 2 for index in range(10)],
        }
    )


def _fit(arm_name: str, payload: dict) -> SimpleNamespace:
    cells = []
    for side in ("YES", "NO"):
        for lower, upper in ((1, 15), (15, 30), (30, 45), (45, 60)):
            cells.append(
                SimpleNamespace(
                    side=side,
                    start_second=lower,
                    end_second_exclusive=upper,
                    fitted=True,
                    converged=True,
                    fallback=None,
                    rows=100,
                    markets=60,
                    utc_days=7,
                    negatives=30,
                    positives=30,
                    slope=0.9,
                    intercept=0.0,
                    iterations=2,
                    objective=1.0,
                    weighted_log_loss=0.5,
                )
            )
    return SimpleNamespace(
        arm_name=arm_name,
        payload=payload,
        cells=tuple(cells),
        weighting="market_equal",
        identity_l2=1.0,
        converged=True,
        iterations=2,
        objective=1.0,
        payload_sha256="a" * 64,
    )


def test_no_probability_winner_never_opens_economics(monkeypatch, tmp_path: Path) -> None:
    config = replace(
        load_incumbent_calibration_config(DEFAULT_INCUMBENT_CALIBRATION_CONFIG),
        runs=tmp_path / "runs",
    )
    config = replace(
        config,
        conditional_estimator=replace(config.conditional_estimator, enabled=False),
    )
    comparison = _ten_day_frame()
    fit_row = comparison.head(1).with_columns(
        pl.lit("fit-market").alias("market_id"),
        pl.lit(datetime(2026, 7, 16, tzinfo=UTC)).alias("window_start"),
        pl.lit(datetime(2026, 7, 16, 0, 0, 5, tzinfo=UTC)).alias("observed_at"),
    )
    source_frame = pl.concat([fit_row, comparison])
    incumbent = load_frozen_asymmetric_incumbent(config.incumbent_model)
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text("{}")

    monkeypatch.setattr(
        benchmark_module,
        "load_asymmetric_value_config",
        lambda _path: SimpleNamespace(core_config=Path("core.toml")),
    )
    monkeypatch.setattr(benchmark_module, "load_core_config", lambda _path: object())
    monkeypatch.setattr(
        benchmark_module,
        "prepare_asymmetric_training_readiness",
        lambda *_args, **_kwargs: (
            readiness_path,
            {"ready": True, "readiness_identity_sha256": "b" * 64},
        ),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_load_core_oracle_value_frame",
        lambda *_args, **_kwargs: (
            source_frame,
            {
                "rows": source_frame.height,
                "markets": source_frame["market_id"].n_unique(),
                "utc_days": source_frame["window_start"].dt.date().n_unique(),
                "proxy_prices_used": False,
            },
        ),
    )
    monkeypatch.setattr(
        benchmark_module,
        "load_frozen_asymmetric_incumbent",
        lambda _path: incumbent,
    )
    monkeypatch.setattr(
        benchmark_module,
        "frozen_parent_probabilities",
        lambda _model, frame: np.full(frame.height, 0.5),
    )
    monkeypatch.setattr(
        benchmark_module,
        "fit_incumbent_calibration_arm",
        lambda arm, model, frame, cfg, **_kwargs: _fit(arm, model.payload),
    )
    monkeypatch.setattr(
        benchmark_module,
        "clone_incumbent_with_calibration",
        lambda model, _fit_result, model_key: {
            **model.payload,
            "model_key": model_key,
        },
    )
    monkeypatch.setattr(
        benchmark_module,
        "validate_incumbent_calibration_parity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "score_incumbent_calibration_payload",
        lambda _model, _payload, frame, **_kwargs: np.full(frame.height, 0.5),
    )
    monkeypatch.setattr(
        benchmark_module,
        "asymmetric_probability_frame",
        lambda frame, probability, model: frame.with_columns(
            pl.Series("probability_yes", probability),
            pl.lit(model).alias("model"),
        ),
    )
    monkeypatch.setattr(
        benchmark_module,
        "select_incumbent_calibration_challenger",
        lambda *_args, **_kwargs: {
            "schema_version": "test",
            "status": "blocked_no_quality_configuration",
            "selected_candidate_id": None,
            "candidate_records": [],
            "economics_used": False,
        },
    )
    monkeypatch.setattr(
        benchmark_module,
        "paired_incumbent_economics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("economics opened without a probability winner")
        ),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_run_conditional_estimator_if_eligible",
        lambda **_kwargs: {
            "status": "blocked_no_estimator_quality_winner",
            "economics_used_for_trigger": False,
        },
    )

    run_dir, result = benchmark_module.run_incumbent_calibration_benchmark(config)

    assert result["economics_opened"] is False
    assert result["paper_artifact"] is None
    assert result["source_process_changed"] is False
    seal = json.loads((run_dir / "probability-selection-seal.json").read_text())
    assert set(seal) == {
        "schema_version",
        "created_at",
        "source_process_id",
        "source_model_key",
        "source_model_sha256",
        "selected_candidate_id",
        "selected_candidate_payload_sha256",
        "probability_selection_sha256",
        "probability_predictions_sha256",
        "calibration_fit_sha256",
        "readiness_manifest_sha256",
        "economics_opened",
    }
    assert seal["economics_opened"] is False
    lineage_path = run_dir / "calibration-selection-lineage.json"
    lineage = json.loads(lineage_path.read_text())
    assert seal["calibration_fit_sha256"] == benchmark_module.file_sha256(lineage_path)
    assert lineage["outcome_access_seal_sha256"] == benchmark_module.file_sha256(
        run_dir / "outcome-access-seal.json"
    )
    assert lineage["development_calibration_support_sha256"] == (
        benchmark_module.file_sha256(run_dir / "calibration-support.json")
    )
    assert lineage["final_calibration_support_sha256"] is None
    assert not (run_dir / "economics-evidence.json").exists()


def test_fit_support_uses_runtime_cell_boundaries() -> None:
    incumbent = load_frozen_asymmetric_incumbent()
    payload = benchmark_module._fit_support_payload(_fit("C1_supported", incumbent.payload))
    assert set(payload["cells"]) == {
        f"{side}_{lower}_{upper}"
        for side in ("YES", "NO")
        for lower, upper in ((1, 15), (15, 30), (30, 45), (45, 60))
    }
    assert all(cell["fallback"] is False for cell in payload["cells"].values())


def test_forward_start_is_next_full_utc_day() -> None:
    requirements = benchmark_module._forward_requirements(
        {"created_at": "2026-08-09T14:25:10+00:00"}
    )
    assert requirements["start_not_before"] == "2026-08-10T00:00:00+00:00"
    assert requirements["minimum_complete_utc_days"] == 21
    assert requirements["no_pnl_based_early_stopping"] is True


def test_runner_rejects_weakened_economic_gate() -> None:
    config = load_incumbent_calibration_config(DEFAULT_INCUMBENT_CALIBRATION_CONFIG)
    weakened = replace(
        config,
        economic_gates=replace(
            config.economic_gates,
            minimum_profit_factor=1.0,
        ),
    )

    with np.testing.assert_raises_regex(ValueError, "economic qualification"):
        benchmark_module._validate_runner_gate_contract(weakened)


def test_estimator_probability_alignment_reorders_and_rejects_extra_rows() -> None:
    comparison = _ten_day_frame()
    artifact = comparison.select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
    ).with_columns(pl.Series("probability_yes", np.linspace(0.2, 0.8, 10)))
    reversed_artifact = artifact.reverse()

    aligned = benchmark_module._aligned_estimator_probability(
        reversed_artifact,
        comparison,
    )

    assert aligned.to_list() == artifact["probability_yes"].to_list()
    with np.testing.assert_raises_regex(RuntimeError, "extra or missing"):
        benchmark_module._aligned_estimator_probability(
            pl.concat([artifact, artifact.head(1)]),
            comparison,
        )


def test_outcome_join_rejects_extra_sealed_prediction_key(tmp_path: Path) -> None:
    comparison = _ten_day_frame()
    sealed = comparison.drop("label_up").with_columns(pl.lit(0.5).alias("probability_yes"))
    extra = sealed.head(1).with_columns(pl.lit("extra-market").alias("market_id"))
    path = tmp_path / "predictions.parquet"
    pl.concat([sealed, extra]).write_parquet(path)

    with np.testing.assert_raises_regex(RuntimeError, "exact outcome key grid"):
        benchmark_module._reload_predictions_with_outcomes(
            path,
            comparison,
            expected_sha256=benchmark_module.file_sha256(path),
        )


def test_calibration_selection_lineage_binds_both_support_scopes(
    tmp_path: Path,
) -> None:
    config = load_incumbent_calibration_config(DEFAULT_INCUMBENT_CALIBRATION_CONFIG)
    incumbent = load_frozen_asymmetric_incumbent(config.incumbent_model)
    candidate_id = "C1_supported"
    selection_path = tmp_path / "probability-selection.json"
    selection_path.write_text(json.dumps({"selected_candidate_id": candidate_id}))
    development_support_path = tmp_path / "calibration-support.json"
    development_support_path.write_text("{}")
    final_support_path = tmp_path / "final-calibration-support.json"
    final_support_path.write_text("{}")
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text("{}")
    payload_path = tmp_path / "candidate-model.json"
    payload_path.write_text("{}")
    prediction_path = tmp_path / "comparison-predictions.parquet"
    prediction_path.write_text("predictions")
    selected_export_payload_path = tmp_path / "selected-candidate-model.json"
    selected_export_payload_path.write_text("{}")
    payload_hashes = {candidate_id: benchmark_module.file_sha256(payload_path)}
    prediction_hashes = {candidate_id: benchmark_module.file_sha256(prediction_path)}
    outcome_access_seal_path = tmp_path / "outcome-access-seal.json"
    outcome_access_seal_path.write_text(
        json.dumps(
            {
                "economics_opened": False,
                "candidate_payload_sha256": payload_hashes,
                "comparison_prediction_sha256": prediction_hashes,
            }
        )
    )

    seal_path, seal = benchmark_module._write_selection_seal(
        tmp_path,
        config=config,
        incumbent=incumbent,
        selection={"selected_candidate_id": candidate_id},
        selection_path=selection_path,
        outcome_access_seal_path=outcome_access_seal_path,
        development_support_path=development_support_path,
        final_support_path=final_support_path,
        prediction_paths={candidate_id: prediction_path},
        prediction_hashes=prediction_hashes,
        payload_paths={candidate_id: payload_path},
        payload_hashes=payload_hashes,
        selected_export_payload_path=selected_export_payload_path,
        selected_export_payload_sha256=benchmark_module.file_sha256(selected_export_payload_path),
        readiness_path=readiness_path,
    )

    lineage_path = tmp_path / "calibration-selection-lineage.json"
    lineage = json.loads(lineage_path.read_text())
    assert benchmark_module.file_sha256(seal_path) == benchmark_module.file_sha256(
        tmp_path / "probability-selection-seal.json"
    )
    assert seal["calibration_fit_sha256"] == benchmark_module.file_sha256(lineage_path)
    assert lineage["development_calibration_support_sha256"] == (
        benchmark_module.file_sha256(development_support_path)
    )
    assert lineage["final_calibration_support_sha256"] == (
        benchmark_module.file_sha256(final_support_path)
    )


def test_conditional_quality_failure_never_opens_economics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_incumbent_calibration_config(DEFAULT_INCUMBENT_CALIBRATION_CONFIG)
    run_dir = tmp_path / "run"
    candidate_dir = run_dir / "candidates"
    candidate_dir.mkdir(parents=True)
    comparison = _ten_day_frame()
    base_prediction = comparison.drop("label_up").with_columns(pl.lit(0.5).alias("probability_yes"))
    incumbent_path = run_dir / "incumbent.parquet"
    base_prediction.write_parquet(incumbent_path)
    prediction_paths: dict[str, Path] = {}
    model_paths: dict[str, Path] = {}
    evidence_paths: dict[str, Path] = {}
    for candidate_id in ESTIMATOR_FALLBACK_CANDIDATES:
        destination = candidate_dir / candidate_id
        destination.mkdir()
        prediction_path = destination / "comparison-predictions.parquet"
        base_prediction.write_parquet(prediction_path)
        model_path = destination / "training-model.joblib"
        model_path.write_bytes(candidate_id.encode())
        evidence_path = destination / "fit-evidence.json"
        evidence_path.write_text("{}")
        prediction_paths[candidate_id] = prediction_path
        model_paths[candidate_id] = model_path
        evidence_paths[candidate_id] = evidence_path
    prediction_hashes = {
        candidate_id: benchmark_module.file_sha256(path)
        for candidate_id, path in prediction_paths.items()
    }
    model_hashes = {
        candidate_id: benchmark_module.file_sha256(path)
        for candidate_id, path in model_paths.items()
    }
    evidence_hashes = {
        candidate_id: benchmark_module.file_sha256(path)
        for candidate_id, path in evidence_paths.items()
    }
    estimator_support = {candidate_id: {} for candidate_id in ESTIMATOR_FALLBACK_CANDIDATES}
    estimator_support_path = run_dir / "estimator-calibration-support.json"
    estimator_support_path.write_text(json.dumps(estimator_support))
    outcome_seal_path = run_dir / "outcome-access-seal.json"
    outcome_seal_path.write_text(
        json.dumps(
            {
                "conditional_estimator_prediction_sha256": prediction_hashes,
                "conditional_estimator_model_sha256": model_hashes,
                "conditional_estimator_fit_evidence_sha256": evidence_hashes,
                "conditional_estimator_support_sha256": (
                    benchmark_module.file_sha256(estimator_support_path)
                ),
                "economics_opened": False,
            }
        )
    )
    readiness_path = run_dir / "readiness.json"
    readiness_path.write_text("{}")
    calibration_selection = {
        "selected_candidate_id": None,
        "economics_used": False,
        "candidate_records": [
            {
                "candidate_id": candidate_id,
                "calibration_support": {"passed": True},
            }
            for candidate_id in benchmark_module.CALIBRATION_CHALLENGERS
        ],
    }
    monkeypatch.setattr(
        benchmark_module,
        "select_incumbent_calibration_challenger",
        lambda *_args, **_kwargs: {
            "schema_version": "test",
            "status": "blocked_no_quality_configuration",
            "selected_candidate_id": None,
            "candidate_records": [],
            "economics_used": False,
        },
    )
    monkeypatch.setattr(
        benchmark_module,
        "paired_incumbent_economics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("conditional economics opened without a probability winner")
        ),
    )

    result = benchmark_module._run_conditional_estimator_if_eligible(
        config=config,
        run_id="run",
        run_dir=run_dir,
        incumbent=SimpleNamespace(),
        source_frame=comparison,
        comparison_frame=comparison,
        source_config=SimpleNamespace(),
        core_config=SimpleNamespace(),
        source_incumbent_probabilities=np.full(comparison.height, 0.5),
        calibration_selection=calibration_selection,
        outcome_access_seal_path=outcome_seal_path,
        readiness_path=readiness_path,
        incumbent_prediction_path=incumbent_path,
        incumbent_prediction_sha256=benchmark_module.file_sha256(incumbent_path),
        estimator_fallbacks={
            candidate_id: SimpleNamespace() for candidate_id in ESTIMATOR_FALLBACK_CANDIDATES
        },
        estimator_prediction_paths=prediction_paths,
        estimator_prediction_hashes=prediction_hashes,
        estimator_model_paths=model_paths,
        estimator_model_hashes=model_hashes,
        estimator_evidence_paths=evidence_paths,
        estimator_evidence_hashes=evidence_hashes,
        estimator_support=estimator_support,
        estimator_support_path=estimator_support_path,
    )

    assert result["status"] == "incumbent_retained_no_estimator_quality_winner"
    assert result["economics_opened"] is False
    assert result["selection_seal_sha256"] is not None
    assert (run_dir / "estimator-fallback-probability-selection-seal.json").is_file()
    assert not (run_dir / "estimator-fallback-economics-evidence.json").exists()


def test_estimator_economics_opens_only_after_selected_model_seal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_incumbent_calibration_config(DEFAULT_INCUMBENT_CALIBRATION_CONFIG)
    run_dir = tmp_path / "run"
    candidate_dir = run_dir / "candidates"
    candidate_dir.mkdir(parents=True)
    comparison = _ten_day_frame()
    base_prediction = comparison.drop("label_up").with_columns(pl.lit(0.5).alias("probability_yes"))
    incumbent_path = run_dir / "incumbent.parquet"
    base_prediction.write_parquet(incumbent_path)
    prediction_paths: dict[str, Path] = {}
    model_paths: dict[str, Path] = {}
    evidence_paths: dict[str, Path] = {}
    fallbacks: dict[str, SimpleNamespace] = {}
    for candidate_id in ESTIMATOR_FALLBACK_CANDIDATES:
        destination = candidate_dir / candidate_id
        destination.mkdir()
        prediction_path = destination / "comparison-predictions.parquet"
        base_prediction.write_parquet(prediction_path)
        model_path = destination / "training-model.joblib"
        model_path.write_bytes(candidate_id.encode())
        evidence_path = destination / "fit-evidence.json"
        evidence_path.write_text("{}")
        prediction_paths[candidate_id] = prediction_path
        model_paths[candidate_id] = model_path
        evidence_paths[candidate_id] = evidence_path
        fallbacks[candidate_id] = SimpleNamespace(
            semantic_sha256=("1" if candidate_id.startswith("E1") else "2") * 64
        )
    prediction_hashes = {
        candidate_id: benchmark_module.file_sha256(path)
        for candidate_id, path in prediction_paths.items()
    }
    model_hashes = {
        candidate_id: benchmark_module.file_sha256(path)
        for candidate_id, path in model_paths.items()
    }
    evidence_hashes = {
        candidate_id: benchmark_module.file_sha256(path)
        for candidate_id, path in evidence_paths.items()
    }
    estimator_support = {candidate_id: {} for candidate_id in ESTIMATOR_FALLBACK_CANDIDATES}
    estimator_support_path = run_dir / "estimator-calibration-support.json"
    estimator_support_path.write_text(json.dumps(estimator_support))
    outcome_seal_path = run_dir / "outcome-access-seal.json"
    outcome_seal_path.write_text(
        json.dumps(
            {
                "conditional_estimator_prediction_sha256": prediction_hashes,
                "conditional_estimator_model_sha256": model_hashes,
                "conditional_estimator_fit_evidence_sha256": evidence_hashes,
                "conditional_estimator_support_sha256": (
                    benchmark_module.file_sha256(estimator_support_path)
                ),
                "economics_opened": False,
            }
        )
    )
    readiness_path = run_dir / "readiness.json"
    readiness_path.write_text("{}")
    calibration_selection = {
        "selected_candidate_id": None,
        "economics_used": False,
        "candidate_records": [
            {
                "candidate_id": candidate_id,
                "calibration_support": {"passed": True},
            }
            for candidate_id in benchmark_module.CALIBRATION_CHALLENGERS
        ],
    }
    selected_candidate = ESTIMATOR_FALLBACK_CANDIDATES[0]
    monkeypatch.setattr(
        benchmark_module,
        "select_incumbent_calibration_challenger",
        lambda *_args, **_kwargs: {
            "schema_version": "test",
            "status": "selected",
            "selected_candidate_id": selected_candidate,
            "candidate_records": [],
            "economics_used": False,
        },
    )
    final_fit = SimpleNamespace(
        bundle={"candidate": selected_candidate},
        evidence={"semantic_sha256": "3" * 64},
        semantic_sha256="3" * 64,
    )
    monkeypatch.setattr(
        benchmark_module,
        "refit_selected_core_oracle_estimator_fallback",
        lambda *_args, **_kwargs: final_fit,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_estimator_support_payload",
        lambda _fit_result: {"qualified": True, "cells": {}},
    )
    monkeypatch.setattr(
        benchmark_module,
        "_validate_sealed_estimator_refit",
        lambda **_kwargs: None,
    )
    fake_ledger = comparison.select("market_id", "window_start")
    monkeypatch.setattr(
        benchmark_module,
        "_economic_ledger",
        lambda *_args, **_kwargs: fake_ledger,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_validate_incumbent_economic_reproduction",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "build_incumbent_correction_ledger",
        lambda *_args, **_kwargs: (
            pl.DataFrame({"market_id": ["m0"]}),
            {},
        ),
    )

    def sealed_economics(*_args, **_kwargs):
        seal_path = run_dir / "estimator-fallback-probability-selection-seal.json"
        assert seal_path.is_file()
        assert json.loads(seal_path.read_text())["economics_opened"] is False
        assert not (run_dir / "estimator-fallback-economics-evidence.json").exists()
        return {"status": "blocked", "gates": []}

    monkeypatch.setattr(
        benchmark_module,
        "paired_incumbent_economics",
        sealed_economics,
    )
    fit_start = datetime(2026, 4, 14, tzinfo=UTC)
    fit_end = datetime(2026, 7, 16, tzinfo=UTC)

    result = benchmark_module._run_conditional_estimator_if_eligible(
        config=config,
        run_id="run",
        run_dir=run_dir,
        incumbent=SimpleNamespace(),
        source_frame=comparison,
        comparison_frame=comparison,
        source_config=SimpleNamespace(fit=SimpleNamespace(start=fit_start, end=fit_end)),
        core_config=SimpleNamespace(),
        source_incumbent_probabilities=np.full(comparison.height, 0.5),
        calibration_selection=calibration_selection,
        outcome_access_seal_path=outcome_seal_path,
        readiness_path=readiness_path,
        incumbent_prediction_path=incumbent_path,
        incumbent_prediction_sha256=benchmark_module.file_sha256(incumbent_path),
        estimator_fallbacks=fallbacks,
        estimator_prediction_paths=prediction_paths,
        estimator_prediction_hashes=prediction_hashes,
        estimator_model_paths=model_paths,
        estimator_model_hashes=model_hashes,
        estimator_evidence_paths=evidence_paths,
        estimator_evidence_hashes=evidence_hashes,
        estimator_support=estimator_support,
        estimator_support_path=estimator_support_path,
    )

    assert result["status"] == "incumbent_retained_estimator_economic_gates_failed"
    assert result["probability_qualified"] is True
    assert result["economics_opened"] is True
    assert result["economics_qualified"] is False
    assert result["forward_requirements"]["status"] == ("not_started_no_qualified_paper_artifact")
