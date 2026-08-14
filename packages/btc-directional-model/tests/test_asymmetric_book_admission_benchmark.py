from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from btc_directional_model import asymmetric_book_admission_benchmark as benchmark
from btc_directional_model import cli
from btc_directional_model.asymmetric_book_admission import (
    BookAdmissionFitEvidence,
    BookAdmissionSupport,
    load_book_admission_config,
)
from btc_directional_model.core_extract import file_sha256, write_json_atomic

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PACKAGE_ROOT
    / "configs"
    / "btc-5m-asymmetric-core-oracle-book-admission-20260414-20260802.toml"
)


def _config():
    return load_book_admission_config(CONFIG_PATH)


def _prediction(candidate_id: str, probabilities: list[float]) -> pl.DataFrame:
    start = datetime(2026, 7, 21, tzinfo=UTC)
    return pl.DataFrame(
        {
            "market_id": [f"m{index}" for index in range(len(probabilities))],
            "window_start": [start + timedelta(minutes=5 * index) for index in range(len(probabilities))],
            "observed_at": [
                start + timedelta(minutes=5 * index, seconds=1)
                for index in range(len(probabilities))
            ],
            "seconds_elapsed": [1] * len(probabilities),
            "candidate_id": [candidate_id] * len(probabilities),
            "probability_yes": probabilities,
        },
        schema_overrides={
            "window_start": pl.Datetime("us", "UTC"),
            "observed_at": pl.Datetime("us", "UTC"),
        },
    )


def _probability_context(prediction: pl.DataFrame) -> pl.DataFrame:
    return prediction.select(*benchmark.BOOK_ADMISSION_KEY_COLUMNS).with_columns(
        pl.Series("label_up", [1] * prediction.height, dtype=pl.Int8),
        pl.Series("yes_ask_vwap_5", [0.25] * prediction.height),
        pl.Series("no_ask_vwap_5", [0.75] * prediction.height),
        pl.Series("yes_ask_depth", [100.0] * prediction.height),
        pl.Series("no_ask_depth", [100.0] * prediction.height),
        pl.Series("yes_cost_per_share", [0.26] * prediction.height),
        pl.Series("no_cost_per_share", [0.76] * prediction.height),
    )


def test_config_and_dependency_identity_are_frozen() -> None:
    config = _config()
    identity = benchmark._dependency_identity(config)

    assert config.runtime_deployable is False
    assert config.batch_forward_eligible is True
    assert set(identity["packages"]) == {
        "numpy",
        "polars",
        "scipy",
        "scikit-learn",
        "joblib",
    }
    assert identity["pyproject_sha256"] == file_sha256(PACKAGE_ROOT / "pyproject.toml")
    assert identity["requirements_lock_sha256"] == file_sha256(
        PACKAGE_ROOT / "requirements.lock"
    )


def test_root_contract_serializes_chronology_as_iso8601(tmp_path: Path) -> None:
    config = _config()
    readiness = tmp_path / "readiness.json"
    write_json_atomic(readiness, {"ready": True})
    target = _prediction("I0", [0.5]).select(*benchmark.BOOK_ADMISSION_KEY_COLUMNS)

    benchmark._write_pre_fit_contracts(
        tmp_path,
        config=config,
        readiness_path=readiness,
        readiness={"ready": True},
        source_evidence={"proxy_prices_used": False},
        dynamics_evidence={"content_sha256": "0" * 64},
        coverage={"passed": True},
        target_validation=target,
    )

    cohort = json.loads((tmp_path / "consumed-cohort-registry.json").read_text())
    assert cohort["development"]["start"] == "2026-04-14T00:00:00+00:00"
    assert cohort["folds"][0]["validation"]["start"] == "2026-07-21T00:00:00+00:00"
    root = json.loads((tmp_path / "root-training-contract.json").read_text())
    assert root["dependency_identity"]["packages"]["polars"]


def test_selected_refit_support_serializes_windows_after_selection_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_seal = tmp_path / "probability-selection-seal.json"
    write_json_atomic(selection_seal, {"selected_candidate_id": "D1"})
    dynamic = pl.DataFrame(
        {
            "market_id": ["fit", "calibration"],
            "window_start": [
                datetime(2026, 7, 1, tzinfo=UTC),
                datetime(2026, 7, 20, tzinfo=UTC),
            ],
        },
        schema_overrides={"window_start": pl.Datetime("us", "UTC")},
    )
    monkeypatch.setattr(
        benchmark,
        "select_target_opportunity_rows",
        lambda frame, *_args, **_kwargs: frame,
    )
    monkeypatch.setattr(
        benchmark,
        "_target_cell_support",
        lambda *_args, **_kwargs: ({"all": {"passed": True}}, []),
    )

    path, passed = benchmark._write_selected_refit_support(
        tmp_path,
        config=_config(),
        dynamic=dynamic,
        selection_seal_path=selection_seal,
    )

    payload = json.loads(path.read_text())
    assert passed is True
    assert payload["cohorts"]["fit"]["window"]["end"] == (
        "2026-07-16T00:00:00+00:00"
    )
    assert payload["accessed_after_probability_selection_seal_sha256"] == file_sha256(
        selection_seal
    )


def test_prediction_artifacts_are_exact_probability_only_and_strictly_bounded(
    tmp_path: Path,
) -> None:
    incumbent = _prediction("I0", [0.4, 0.6])
    static = _prediction("S0", [0.41, 0.59])
    paths, hashes = benchmark._write_prediction_artifacts(
        tmp_path,
        incumbent_predictions=[incumbent],
        candidate_predictions={"S0": [static]},
        expected_keys=incumbent.select(*benchmark.BOOK_ADMISSION_KEY_COLUMNS),
    )

    assert set(pl.read_parquet_schema(paths["I0"]).names()) == (
        benchmark.SEALED_PREDICTION_COLUMNS
    )
    assert hashes["S0"] == file_sha256(paths["S0"])

    invalid = _prediction("S0", [0.0, 0.5])
    with pytest.raises(RuntimeError, match="invalid probabilities"):
        benchmark._write_prediction_artifacts(
            tmp_path / "invalid",
            incumbent_predictions=[incumbent],
            candidate_predictions={"S0": [invalid]},
            expected_keys=incumbent.select(*benchmark.BOOK_ADMISSION_KEY_COLUMNS),
        )


def test_probability_selector_context_rejects_economic_fields(tmp_path: Path) -> None:
    prediction = _prediction("D1", [0.4, 0.6])
    path = tmp_path / "prediction.parquet"
    prediction.write_parquet(path)
    context = _probability_context(prediction)

    selected = benchmark._load_sealed_probability_frame(
        path,
        file_sha256(path),
        context,
    )
    assert set(selected.columns) == benchmark.PROBABILITY_SELECTION_FRAME_COLUMNS

    with pytest.raises(RuntimeError, match="non-allowlisted context"):
        benchmark._load_sealed_probability_frame(
            path,
            file_sha256(path),
            context.with_columns(pl.lit(0.01).alias("fee_rate")),
        )


def test_preseal_target_projection_rejects_oof_labels() -> None:
    with pytest.raises(RuntimeError, match="cannot enter a pre-seal"):
        benchmark._outcome_blind_target(
            pl.DataFrame({"label_up": [1]}),
            _config(),
        )


def test_outcome_seal_requires_exact_prediction_schema(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates" / "D1"
    candidate_dir.mkdir(parents=True)
    path = candidate_dir / "oof-predictions.parquet"
    _prediction("D1", [0.4]).with_columns(pl.lit("unexpected").alias("extra")).write_parquet(
        path
    )
    inputs = []
    for name in ("pre.json", "models.json", "predictions.json", "support.json"):
        value = tmp_path / name
        write_json_atomic(value, {"name": name})
        inputs.append(value)

    with pytest.raises(RuntimeError, match="exact allowlist"):
        benchmark._write_outcome_access_seal(
            tmp_path,
            pre_outcome_manifest_path=inputs[0],
            model_manifest_path=inputs[1],
            prediction_manifest_path=inputs[2],
            support_path=inputs[3],
            prediction_hashes={"D1": file_sha256(path)},
        )


def test_model_bytes_are_checked_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.pkl"
    path.write_bytes(b"sealed")
    serialization = {"artifact_sha256": file_sha256(path)}
    path.write_bytes(b"tampered")
    called = False

    def loader(_: Path):
        nonlocal called
        called = True
        raise AssertionError("loader must not receive tampered bytes")

    monkeypatch.setattr(benchmark, "load_book_admission_model", loader)
    with pytest.raises(RuntimeError, match="before deserialization"):
        benchmark._load_verified_model(path, serialization)
    assert called is False


def test_batch_authorization_rejects_tampered_economics_access(tmp_path: Path) -> None:
    selected = "D1"
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"model")
    selection_path = tmp_path / "probability-selection-seal.json"
    write_json_atomic(
        selection_path,
        {
            "schema_version": benchmark.BOOK_ADMISSION_SELECTION_SEAL_SCHEMA_VERSION,
            "selected_candidate_id": selected,
            "economics_opened": False,
            "selection_uses_economics": False,
        },
    )
    selected_model_path = tmp_path / "selected-model-seal.json"
    write_json_atomic(
        selected_model_path,
        {
            "schema_version": benchmark.BOOK_ADMISSION_SELECTED_MODEL_SEAL_SCHEMA_VERSION,
            "selected_candidate_id": selected,
            "selection_seal_sha256": file_sha256(selection_path),
            "model_artifact_sha256": file_sha256(model_path),
            "economics_opened": False,
            "runtime_deployable": False,
            "lineage_scope": "final_batch_forward_refit",
            "development_oof_models_reused": False,
        },
    )
    access_path = tmp_path / "economics-access.json"
    write_json_atomic(
        access_path,
        {
            "schema_version": benchmark.BOOK_ADMISSION_ECONOMICS_SCHEMA_VERSION,
            "selected_candidate_id": "D2",
            "selection_seal_sha256": file_sha256(selection_path),
            "selected_model_seal_sha256": file_sha256(selected_model_path),
            "economics_opened": True,
            "evaluated_candidates": ["I0", selected],
        },
    )
    economics_path = tmp_path / "selected-economics.json"
    write_json_atomic(
        economics_path,
        {
            "schema_version": benchmark.BOOK_ADMISSION_ECONOMICS_SCHEMA_VERSION,
            "status": "qualified",
            "selected_candidate_id": selected,
            "selection_seal_sha256": file_sha256(selection_path),
            "selected_model_seal_sha256": file_sha256(selected_model_path),
            "selected_model_artifact_sha256": file_sha256(model_path),
            "economics_access_sha256": file_sha256(access_path),
            "economic_scope": "development_oof",
            "final_batch_refit_economically_scored": False,
            "development_prediction_sha256": {"I0": "1" * 64, selected: "2" * 64},
            "evaluated_candidates": ["I0", selected],
        },
    )

    with pytest.raises(RuntimeError, match="lineage is inconsistent"):
        benchmark._authorize_batch_forward(
            tmp_path,
            config=_config(),
            selected=selected,
            final_model_path=model_path,
            selection_seal_path=selection_path,
            selected_model_seal_path=selected_model_path,
            economics_access_path=access_path,
            economics_path=economics_path,
        )

    write_json_atomic(
        access_path,
        {
            "schema_version": benchmark.BOOK_ADMISSION_ECONOMICS_SCHEMA_VERSION,
            "selected_candidate_id": selected,
            "selection_seal_sha256": file_sha256(selection_path),
            "selected_model_seal_sha256": file_sha256(selected_model_path),
            "economics_opened": True,
            "evaluated_candidates": ["I0", selected],
        },
    )
    economics = json.loads(economics_path.read_text())
    economics["economics_access_sha256"] = file_sha256(access_path)
    write_json_atomic(economics_path, economics)
    destination = benchmark._authorize_batch_forward(
        tmp_path,
        config=_config(),
        selected=selected,
        final_model_path=model_path,
        selection_seal_path=selection_path,
        selected_model_seal_path=selected_model_path,
        economics_access_path=access_path,
        economics_path=economics_path,
    )
    assert file_sha256(destination) == file_sha256(model_path)
    assert json.loads(destination.with_name("manifest.json").read_text())[
        "runtime_deployable"
    ] is False


def test_empty_challenger_economics_fails_gates_without_crashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incumbent_ledger = pl.DataFrame({"marker": [1]})
    challenger_ledger = pl.DataFrame({"marker": []}, schema={"marker": pl.Int64})
    ledgers = iter((incumbent_ledger, challenger_ledger))
    monkeypatch.setattr(benchmark, "_economic_ledger", lambda *_: next(ledgers))
    correction = pl.DataFrame(
        {
            "correction_category": [],
            "challenger_stress_1c_net": [],
        },
        schema={"correction_category": pl.String, "challenger_stress_1c_net": pl.Float64},
    )
    correction_summary = {
        "net_corrected_decisions": 0,
        "corrected_decision_improvement_utc_days": 0,
    }
    monkeypatch.setattr(
        benchmark,
        "build_incumbent_correction_ledger",
        lambda *_args, **_kwargs: (correction, correction_summary),
    )
    incumbent_metrics = {
        "trades": 10,
        "yes_trades": 5,
        "no_trades": 5,
        "accuracy": 0.30,
        "net_profit": 1.0,
        "profit_factor": 1.1,
        "profit_factor_no_losses": False,
        "mean_share_price": 0.24,
        "selected_win_rate_advantage": 0.03,
        "stress_1c_net_profit": 0.5,
        "stress_1c_net_expectancy_per_trade": 0.05,
        "loss_recovery_wins": 0.35,
        "average_loss": -1.0,
        "maximum_loss": -1.2,
        "maximum_drawdown": -4.0,
        "yes_net_profit": 0.5,
        "no_net_profit": 0.5,
    }
    metrics = iter((incumbent_metrics, {"trades": 0}))
    monkeypatch.setattr(benchmark, "ledger_metrics", lambda _: next(metrics))
    validation = pl.DataFrame(
        {
            "market_id": ["m1"],
            "window_start": [datetime(2026, 7, 21, tzinfo=UTC)],
        },
        schema_overrides={"window_start": pl.Datetime("us", "UTC")},
    )
    identity = pl.DataFrame({"candidate_id": ["I0"]})
    challenger = pl.DataFrame({"candidate_id": ["D1"]})

    result = benchmark._evaluate_selected_economics(
        _config(),
        run_dir=tmp_path,
        incumbent=identity,
        challenger=challenger,
        validation=validation,
    )

    assert result["status"] == "not_qualified"
    assert result["paired_development_diagnostic"]["status"] == (
        "not_run_empty_challenger_ledger"
    )
    assert any(not gate["passed"] for gate in result["strict_gates"])


def test_finalize_seals_result_report_and_artifact_manifest(tmp_path: Path) -> None:
    write_json_atomic(tmp_path / "fold-model-manifest.json", {"models": {}})
    result = {
        "status": "incumbent_retained_probability_gates",
        "selected_candidate_id": None,
        "economics_opened": False,
        "batch_forward_artifact": None,
    }
    selection = {
        "status": "blocked_no_quality_configuration",
        "incumbent": {"target_metrics": {"overall": {"brier": 0.2, "log_loss": 0.6}}},
        "candidate_records": [],
        "failure_trace": [],
    }

    benchmark._finalize(tmp_path, result, selection=selection, economics=None)

    manifest = json.loads((tmp_path / "artifact-manifest.json").read_text())
    bundle = json.loads((tmp_path / "benchmark-bundle-seal.json").read_text())
    assert "benchmark-result.json" in manifest["artifacts"]
    assert "benchmark-report.md" in manifest["artifacts"]
    assert bundle["artifact_manifest_sha256"] == file_sha256(
        tmp_path / "artifact-manifest.json"
    )


@pytest.mark.parametrize(
    ("winner", "expected_status", "expected_events"),
    (
        (
            None,
            "incumbent_retained_probability_gates",
            ["outcome_seal", "selection", "selection_seal", "finalize"],
        ),
        (
            "D1",
            "qualified_batch_forward_blocked_runtime_feature_parity",
            [
                "outcome_seal",
                "selection",
                "selection_seal",
                "refit_support",
                "refit",
                "economics",
                "authorize",
                "finalize",
            ],
        ),
    ),
)
def test_runner_enforces_seal_first_terminal_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    winner: str | None,
    expected_status: str,
    expected_events: list[str],
) -> None:
    config = _config()
    config = replace(
        config,
        folds=(config.folds[0],),
        paths=replace(config.paths, runs=tmp_path / "runs"),
    )
    start = datetime(2026, 7, 21, tzinfo=UTC)
    source = pl.DataFrame(
        {
            "market_id": ["m1"],
            "window_start": [start],
            "observed_at": [start + timedelta(seconds=1)],
            "seconds_elapsed": [1],
            "label_up": [1],
            "incumbent_probability_yes": [0.5],
            "fee_rate": [0.0],
            "yes_best_ask": [0.25],
            "yes_ask_vwap_5": [0.25],
            "yes_ask_depth": [100.0],
            "no_best_ask": [0.75],
            "no_ask_vwap_5": [0.75],
            "no_ask_depth": [100.0],
            "yes_cost_per_share": [0.25],
            "no_cost_per_share": [0.75],
            "yes_execution_cost_per_share": [0.25],
            "no_execution_cost_per_share": [0.75],
        },
        schema_overrides={
            "window_start": pl.Datetime("us", "UTC"),
            "observed_at": pl.Datetime("us", "UTC"),
        },
    )
    readiness_path = tmp_path / "readiness.json"
    write_json_atomic(readiness_path, {"ready": True})
    events: list[str] = []
    monkeypatch.setattr(benchmark, "load_asymmetric_value_config", lambda _: SimpleNamespace(core_config=Path("core")))
    monkeypatch.setattr(benchmark, "load_core_config", lambda _: object())
    monkeypatch.setattr(
        benchmark,
        "prepare_asymmetric_training_readiness",
        lambda *_args, **_kwargs: (readiness_path, {"ready": True}),
    )
    monkeypatch.setattr(
        benchmark,
        "_load_core_oracle_value_frame",
        lambda *_args, **_kwargs: (source.drop("incumbent_probability_yes"), {"proxy_prices_used": False}),
    )
    monkeypatch.setattr(
        benchmark,
        "load_frozen_asymmetric_incumbent",
        lambda _: SimpleNamespace(payload={}),
    )
    monkeypatch.setattr(benchmark, "_validate_incumbent", lambda *_: None)
    monkeypatch.setattr(benchmark, "frozen_parent_probabilities", lambda *_: [0.5])
    monkeypatch.setattr(
        benchmark,
        "score_incumbent_calibration_payload",
        lambda *_args, **_kwargs: [0.5],
    )
    monkeypatch.setattr(
        benchmark,
        "attach_frozen_policy_selected_side",
        lambda frame, _config: frame,
    )
    monkeypatch.setattr(benchmark, "attach_causal_book_dynamics", lambda frame, **_: frame)
    monkeypatch.setattr(
        benchmark,
        "book_dynamics_evidence",
        lambda _: {"content_sha256": "0" * 64},
    )
    monkeypatch.setattr(benchmark, "_validation_union", lambda *_: source)
    monkeypatch.setattr(benchmark, "_outcome_blind_target", lambda frame, _: frame)
    monkeypatch.setattr(
        benchmark,
        "_validate_readiness",
        lambda *_: {"minimum_conditional_dynamic_coverage": 1.0},
    )
    monkeypatch.setattr(benchmark, "_write_pre_fit_contracts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark, "_validate_fold_chronology", lambda *_: None)

    fit_evidence = BookAdmissionFitEvidence(
        fit_key_sha256="1" * 64,
        calibration_key_sha256="2" * 64,
        fit_rows=2,
        calibration_rows=2,
        fit_markets=1,
        calibration_markets=1,
        fit_days=1,
        calibration_days=1,
        fit_weight_sha256="3" * 64,
        calibration_weight_sha256="4" * 64,
        gamma=1.0,
        raw_residual_minimum=0.0,
        raw_residual_median=0.0,
        raw_residual_maximum=0.0,
        residual_cap_hit_rate=0.0,
    )

    class FakeModel:
        semantic_sha256 = "5" * 64
        evidence = fit_evidence

        def __init__(self, name: str):
            self.name = name

        def serialize(self, path: Path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"model")
            return {"artifact_sha256": file_sha256(path)}

    monkeypatch.setattr(
        benchmark,
        "fit_book_admission_candidate",
        lambda _config, name, *_: FakeModel(name),
    )
    monkeypatch.setattr(
        benchmark,
        "_load_verified_model",
        lambda path, _serialization: FakeModel(path.parents[1].name),
    )
    monkeypatch.setattr(benchmark, "_validate_serialized_model", lambda *_args, **_kwargs: None)

    def score(model, frame, _config):
        scored = frame.select(*benchmark.BOOK_ADMISSION_KEY_COLUMNS).with_columns(
            pl.lit(model.name).alias("candidate_name"),
            pl.lit(0.5).alias("incumbent_probability_yes"),
        )
        return SimpleNamespace(
            frame=scored,
            probability_sha256="6" * 64,
            model_semantic_sha256=model.semantic_sha256,
        )

    monkeypatch.setattr(benchmark, "score_book_admission_model", score)
    support = BookAdmissionSupport(2, 1, 1, 1, 1, 1, 1, "7" * 64)
    monkeypatch.setattr(benchmark, "book_admission_support", lambda *_args, **_kwargs: support)
    monkeypatch.setattr(
        benchmark,
        "_selection_support",
        lambda config, *_: {
            candidate.name: {
                "support_passed": True,
                "coverage": 1.0,
                "residual_cap_passed": True,
                "regularization_strength": candidate.l2_regularization,
                "model_complexity": 1,
            }
            for candidate in config.model.candidates
        },
    )

    def pre_outcome(run_dir: Path, **_kwargs):
        path = run_dir / "pre-outcome-artifact-manifest.json"
        write_json_atomic(path, {})
        return path

    monkeypatch.setattr(benchmark, "_write_pre_outcome_artifact_manifest", pre_outcome)

    def outcome_seal(run_dir: Path, **_kwargs):
        events.append("outcome_seal")
        path = run_dir / "outcome-access-seal.json"
        write_json_atomic(path, {"economics_opened": False})
        return path

    monkeypatch.setattr(benchmark, "_write_outcome_access_seal", outcome_seal)
    monkeypatch.setattr(
        benchmark,
        "_target_cell_support",
        lambda *_args, **_kwargs: ({"cells": {}}, []),
    )

    def select(incumbent, static, dynamics, *_args, **_kwargs):
        events.append("selection")
        for frame in (incumbent, static, *dynamics.values()):
            assert set(frame.columns) == benchmark.PROBABILITY_SELECTION_FRAME_COLUMNS
        return {
            "status": "selected" if winner else "blocked_no_quality_configuration",
            "selected_candidate_id": winner,
            "incumbent": {},
            "candidate_records": [],
            "failure_trace": [],
        }

    monkeypatch.setattr(benchmark, "select_asymmetric_book_admission_challenger", select)

    def selection_seal(run_dir: Path, **_kwargs):
        events.append("selection_seal")
        path = run_dir / "probability-selection-seal.json"
        write_json_atomic(path, {"selected_candidate_id": winner, "economics_opened": False})
        return path

    monkeypatch.setattr(benchmark, "_write_selection_seal", selection_seal)

    def refit_support(run_dir: Path, **_kwargs):
        events.append("refit_support")
        path = run_dir / "selected-refit-support.json"
        write_json_atomic(path, {"passed": True})
        return path, True

    monkeypatch.setattr(benchmark, "_write_selected_refit_support", refit_support)

    def refit(run_dir: Path, **_kwargs):
        events.append("refit")
        model = run_dir / "selected-book-admission" / "model.pkl"
        model.parent.mkdir()
        model.write_bytes(b"selected")
        seal = run_dir / "selected-model-seal.json"
        write_json_atomic(seal, {"runtime_deployable": False})
        return object(), model, seal

    monkeypatch.setattr(benchmark, "_refit_selected_model", refit)

    def economics(_config, *, run_dir, incumbent, challenger, validation):
        del _config, run_dir, validation
        events.append("economics")
        assert "fee_rate" in incumbent.columns
        assert "fee_rate" in challenger.columns
        return {
            "status": "qualified",
            "selected_candidate_id": winner,
            "strict_gates": [],
            "incumbent": {"metrics": {}},
            "challenger": {"metrics": {}},
            "evaluated_candidates": ["I0", winner],
        }

    monkeypatch.setattr(benchmark, "_evaluate_selected_economics", economics)

    def authorize(run_dir: Path, **_kwargs):
        events.append("authorize")
        return run_dir / "batch-forward" / "selected-book-admission.pkl"

    monkeypatch.setattr(benchmark, "_authorize_batch_forward", authorize)
    monkeypatch.setattr(
        benchmark,
        "_finalize",
        lambda *_args, **_kwargs: events.append("finalize"),
    )

    _, result = benchmark.run_book_admission_benchmark(config)

    assert result["status"] == expected_status
    assert events == expected_events


def test_cli_exposes_book_admission_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(
        cli,
        "load_and_run_book_admission_benchmark",
        lambda *_args, **_kwargs: (
            run_dir,
            {
                "status": "incumbent_retained_probability_gates",
                "selected_candidate_id": None,
                "batch_forward_artifact": None,
            },
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["btc-directional-model", "asymmetric-book-admission-run", "--config", str(CONFIG_PATH)],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "incumbent retained" in output
    assert "runtime deployable: false" in output
