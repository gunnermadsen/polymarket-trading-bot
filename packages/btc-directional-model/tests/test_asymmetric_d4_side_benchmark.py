from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from btc_directional_model import asymmetric_d4_side_benchmark as benchmark_module
from btc_directional_model import cli
from btc_directional_model.asymmetric_book_admission import load_book_admission_config
from btc_directional_model.asymmetric_d4_side_benchmark import (
    ARM_IDS,
    D4_BASE_ID,
    EXPECTED_PARENT_ARTIFACTS,
    INCUMBENT_ID,
    PARENT_CANDIDATES,
    _assert_reproduced_parent_d4,
    _calibration_input_sha256,
    _canonical_sha256,
    _frozen_orientation_ledger,
    _orient_parent_prediction,
    _projected_model_record,
    _strict_validation_union,
    _write_projected_pnl_csv,
    _write_projected_pnl_markdown,
    validate_parent_run_integrity,
)
from btc_directional_model.core_extract import file_sha256


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parent_config() -> object:
    return load_book_admission_config(
        package_root()
        / "configs"
        / "btc-5m-asymmetric-core-oracle-book-admission-20260414-20260802.toml"
    )


def _utc_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("window_start").cast(pl.Datetime("us", "UTC")),
        pl.col("observed_at").cast(pl.Datetime("us", "UTC")),
    )


def _parent_integrity_fixture(root: Path) -> SimpleNamespace:
    days = tuple(
        date.fromisoformat(value)
        for value in (
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
        )
    )
    root.mkdir()
    model_records: dict[str, object] = {}
    for candidate in PARENT_CANDIDATES:
        for day in days:
            relative = f"candidates/{candidate}/{day.isoformat()}/model.pkl"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{candidate}/{day.isoformat()}".encode())
            digest = file_sha256(path)
            semantic = (candidate.encode().hex() + day.isoformat().replace("-", ""))[:64]
            semantic = semantic.ljust(64, "0")
            model_records[f"{candidate}/{day.isoformat()}"] = {
                "path": relative,
                "artifact_sha256": digest,
                "semantic_sha256": semantic,
                "serialization": {
                    "artifact_sha256": digest,
                    "model_semantic_sha256": semantic,
                    "schema_version": "btc-asymmetric-book-admission-v1",
                    "candidate_name": candidate,
                },
            }
    fold_manifest = {
        "schema_version": "btc-asymmetric-book-admission-benchmark-v1",
        "usable_as_final_batch_model": False,
        "models": model_records,
    }
    (root / "fold-model-manifest.json").write_text(json.dumps(fold_manifest))

    prediction_paths: dict[str, str] = {}
    prediction_hashes: dict[str, str] = {}
    for candidate in (INCUMBENT_ID, *PARENT_CANDIDATES):
        relative = f"candidates/{candidate}/oof-predictions.parquet"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"prediction/{candidate}".encode())
        prediction_paths[candidate] = relative
        prediction_hashes[candidate] = file_sha256(path)
    prediction_manifest = {
        "schema_version": "btc-asymmetric-book-admission-benchmark-v1",
        "prediction_paths": prediction_paths,
        "prediction_sha256": prediction_hashes,
    }
    (root / "prediction-manifest.json").write_text(json.dumps(prediction_manifest))
    root_contract = {
        "profile": "btc_asymmetric_core_oracle_book_admission",
        "process_id": "process",
        "incumbent_model_key": "model",
        "incumbent_model_sha256": "model-sha",
        "paper_only": True,
        "runtime_deployable": False,
        "process_change_allowed": False,
        "recursive_search": {
            "candidate_ids": list(PARENT_CANDIDATES),
            "pnl_based_candidate_selection_allowed": False,
        },
    }
    (root / "root-training-contract.json").write_text(json.dumps(root_contract))
    for name in (
        "benchmark-report.md",
        "benchmark-result.json",
        "candidate-registry.json",
        "candidate-support-manifest.json",
        "consumed-cohort-registry.json",
        "feature-manifest.json",
        "outcome-access-seal.json",
        "pooled-oof-cell-support.json",
        "pre-outcome-artifact-manifest.json",
        "probability-selection-seal.json",
        "probability-selection.json",
        "source-readiness-manifest.json",
    ):
        (root / name).write_text("{}" if name.endswith(".json") else "report\n")
    artifacts = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    assert len(artifacts) == EXPECTED_PARENT_ARTIFACTS
    manifest = {
        "schema_version": "btc-asymmetric-book-admission-benchmark-v1",
        "artifacts": artifacts,
        "aggregate_sha256": _canonical_sha256(artifacts),
    }
    manifest_path = root / "artifact-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    bundle = {
        "schema_version": "btc-asymmetric-book-admission-benchmark-v1",
        "artifact_manifest_sha256": file_sha256(manifest_path),
        "benchmark_report_sha256": file_sha256(root / "benchmark-report.md"),
        "benchmark_result_sha256": file_sha256(root / "benchmark-result.json"),
    }
    bundle["aggregate_sha256"] = _canonical_sha256(bundle)
    bundle_path = root / "benchmark-bundle-seal.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True))
    return SimpleNamespace(
        paths=SimpleNamespace(parent_run=root),
        parent_run=SimpleNamespace(
            run_id=root.name,
            root_training_contract_sha256=file_sha256(root / "root-training-contract.json"),
            fold_model_manifest_sha256=file_sha256(root / "fold-model-manifest.json"),
            prediction_manifest_sha256=file_sha256(root / "prediction-manifest.json"),
            probability_selection_sha256=file_sha256(root / "probability-selection.json"),
            probability_selection_seal_sha256=file_sha256(
                root / "probability-selection-seal.json"
            ),
            benchmark_bundle_seal_sha256=file_sha256(bundle_path),
            d4_oof_predictions_sha256=prediction_hashes["D4"],
        ),
        incumbent=SimpleNamespace(
            process_id="process",
            model_key="model",
            model_sha256="model-sha",
        ),
        oof_utc_days=days,
    )


def test_parent_integrity_validates_all_artifacts_and_rejects_one_tampered_model(
    tmp_path: Path,
) -> None:
    config = _parent_integrity_fixture(tmp_path / "sealed-parent")

    evidence = validate_parent_run_integrity(config)

    assert evidence["validated_artifact_count"] == 71
    assert evidence["validated_fold_model_count"] == 50
    assert evidence["validated_prediction_count"] == 6
    assert evidence["all_parent_artifacts_validated_before_deserialization"] is True

    tampered = config.paths.parent_run / "candidates" / "S0" / "2026-07-21" / "model.pkl"
    tampered.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="parent artifact changed"):
        validate_parent_run_integrity(config)


def test_strict_oof_union_requires_exact_days_and_excludes_july_26_27() -> None:
    days = (
        date(2026, 7, 21),
        date(2026, 7, 22),
        date(2026, 7, 23),
        date(2026, 7, 24),
        date(2026, 7, 25),
        date(2026, 7, 28),
        date(2026, 7, 29),
        date(2026, 7, 30),
        date(2026, 7, 31),
        date(2026, 8, 1),
    )
    rows = []
    for index, day in enumerate(days):
        start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
        rows.append(
            {
                "market_id": f"market-{index}",
                "window_start": start,
                "observed_at": start + timedelta(seconds=1),
                "seconds_elapsed": 1,
            }
        )
    frame = _utc_frame(rows)
    config = SimpleNamespace(
        oof_utc_days=days,
        excluded_utc_days=(date(2026, 7, 26), date(2026, 7, 27)),
    )

    assert _strict_validation_union(frame, config).height == 10

    with pytest.raises(RuntimeError, match="UTC-day union"):
        _strict_validation_union(frame.filter(pl.col("market_id") != "market-0"), config)


def test_i0_orientation_is_rederived_while_calibrated_no_side_remains_frozen() -> None:
    config = parent_config()
    start = datetime(2026, 7, 21, tzinfo=UTC)
    context = _utc_frame(
        [
            {
                "market_id": "market",
                "window_start": start,
                "observed_at": start + timedelta(seconds=2),
                "seconds_elapsed": 2,
                "incumbent_probability_yes": 0.20,
                "selected_side": "NO",
                "selected_side_policy_eligible": True,
                "yes_ask_vwap_5": 0.25,
                "no_ask_vwap_5": 0.25,
                "yes_cost_per_share": 0.26,
                "no_cost_per_share": 0.26,
            }
        ]
    )
    i0_prediction = context.select(
        "market_id", "window_start", "observed_at", "seconds_elapsed"
    ).with_columns(
        pl.lit("I0").alias("candidate_id"),
        pl.lit(0.80).alias("probability_yes"),
    )

    oriented = _orient_parent_prediction(
        i0_prediction,
        context,
        config,
        candidate_id="I0",
    )

    assert oriented["selected_side"].to_list() == ["YES"]

    economic = _utc_frame(
        [
            {
                "market_id": "market",
                "window_start": start,
                "observed_at": start + timedelta(seconds=2),
                "seconds_elapsed": 2,
                "candidate_id": "N1",
                "probability_yes": 0.90,
                "selected_side": "NO",
                "label_up": 1,
                "yes_ask_vwap_5": 0.25,
                "no_ask_vwap_5": 0.25,
                "yes_ask_depth": 100.0,
                "no_ask_depth": 100.0,
                "yes_cost_per_share": 0.26,
                "no_cost_per_share": 0.26,
                "yes_execution_cost_per_share": 0.25,
                "no_execution_cost_per_share": 0.25,
            }
        ]
    )
    assert _frozen_orientation_ledger(economic, config.policy, arm_id="N1").is_empty()


def test_exact_d4_reproduction_rejects_one_probability_change() -> None:
    start = datetime(2026, 7, 21, tzinfo=UTC)
    base = _utc_frame(
        [
            {
                "market_id": "market",
                "window_start": start,
                "observed_at": start + timedelta(seconds=1),
                "seconds_elapsed": 1,
                "candidate_id": D4_BASE_ID,
                "probability_yes": 0.4,
                "selected_side": "NO",
            }
        ]
    )
    parent = base.select(
        "market_id", "window_start", "observed_at", "seconds_elapsed"
    ).with_columns(
        pl.lit("D4").alias("candidate_id"),
        pl.lit(0.4).alias("probability_yes"),
    )
    _assert_reproduced_parent_d4(base, parent)
    with pytest.raises(RuntimeError, match="exactly reproduce"):
        _assert_reproduced_parent_d4(
            base.with_columns(pl.lit(0.4000000001).alias("probability_yes")),
            parent,
        )


def test_calibration_input_digest_binds_probability_side_and_label() -> None:
    start = datetime(2026, 7, 21, tzinfo=UTC)
    frame = _utc_frame(
        [
            {
                "market_id": "market",
                "window_start": start,
                "observed_at": start + timedelta(seconds=2),
                "seconds_elapsed": 2,
                "incumbent_probability_yes": 0.35,
                "selected_side": "NO",
                "label_up": 0,
            }
        ]
    )
    baseline = _calibration_input_sha256(frame)

    assert baseline != _calibration_input_sha256(
        frame.with_columns(pl.lit(0.36).alias("incumbent_probability_yes"))
    )
    assert baseline != _calibration_input_sha256(
        frame.with_columns(pl.lit("YES").alias("selected_side"))
    )
    assert baseline != _calibration_input_sha256(
        frame.with_columns(pl.lit(1).alias("label_up"))
    )


def test_projected_pnl_outputs_show_every_arm_nominal_and_stressed_pf(
    tmp_path: Path,
) -> None:
    config = parent_config()
    start = datetime(2026, 7, 21, tzinfo=UTC)
    source = _utc_frame(
        [
            {
                "market_id": "market",
                "window_start": start,
                "observed_at": start + timedelta(seconds=2),
                "seconds_elapsed": 2,
                "candidate_id": arm_id,
                "probability_yes": 0.65,
                "selected_side": "NO",
                "label_up": 0,
                "yes_ask_vwap_5": 0.25,
                "no_ask_vwap_5": 0.25,
                "yes_ask_depth": 100.0,
                "no_ask_depth": 100.0,
                "yes_cost_per_share": 0.26,
                "no_cost_per_share": 0.26,
                "yes_execution_cost_per_share": 0.25,
                "no_execution_cost_per_share": 0.25,
            }
            for arm_id in ARM_IDS
        ]
    )
    models = {}
    for arm_id in ARM_IDS:
        ledger = _frozen_orientation_ledger(
            source.filter(pl.col("candidate_id") == arm_id),
            config.policy,
            arm_id=arm_id,
        )
        models[arm_id] = _projected_model_record(
            ledger,
            arm_id=arm_id,
            eligible_market_count=2_692,
            selected_candidate_id="N1",
        )
    csv_path = tmp_path / "projected-pnl.csv"
    markdown_path = tmp_path / "projected-pnl.md"
    _write_projected_pnl_csv(csv_path, models)
    _write_projected_pnl_markdown(
        markdown_path,
        {"models": models, "eligible_market_denominator": 2_692},
    )

    csv_text = csv_path.read_text()
    markdown = markdown_path.read_text()
    assert all(arm_id in csv_text for arm_id in ARM_IDS)
    assert "profit_factor" in csv_text
    assert "stress_1c_profit_factor" in csv_text
    assert "mean_share_price" in csv_text
    assert all(arm_id in markdown for arm_id in ARM_IDS)
    assert "+1c PF" in markdown
    assert "Mean price" in markdown
    assert "Eligible-market denominator: 2692" in markdown


def test_d4_side_cli_reports_projected_pnl_without_qualification(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        cli,
        "load_and_run_d4_side_calibration_benchmark",
        lambda path, force=False: (
            run_dir,
            {
                "status": "consumed_development_diagnostic_complete",
                "probability_selected_candidate_id": "N1",
            },
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "btc-directional-model",
            "asymmetric-d4-side-calibration-run",
            "--config",
            str(tmp_path / "config.toml"),
            "--force",
        ],
    )

    cli.main()

    output = capsys.readouterr().out
    assert str(run_dir / "projected-pnl.md") in output
    assert "probability-selected challenger: N1" in output
    assert "qualification eligible: false" in output


def test_runner_seals_probability_selection_before_all_arm_projected_pnl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from btc_directional_model.asymmetric_d4_side_calibration import (
        PARENT_RUN_ID,
        load_d4_side_calibration_config,
    )

    config = load_d4_side_calibration_config(
        package_root()
        / "configs"
        / "btc-5m-asymmetric-core-oracle-d4-side-calibration-20260414-20260802.toml"
    )
    parent_run = tmp_path / PARENT_RUN_ID
    parent_run.mkdir()
    (parent_run / "fold-model-manifest.json").write_text(
        json.dumps(
            {
                "models": {
                    f"D4/{fold.name}": {
                        "path": f"candidates/D4/{fold.name}/model.pkl",
                        "artifact_sha256": "a" * 64,
                        "semantic_sha256": "b" * 64,
                        "serialization": {},
                    }
                    for fold in config.folds
                }
            }
        )
    )
    (parent_run / "benchmark-bundle-seal.json").write_text("{}")
    (parent_run / "artifact-manifest.json").write_text("{}")
    config = replace(
        config,
        paths=replace(
            config.paths,
            parent_run=parent_run,
            runs=tmp_path / "runs",
        ),
    )

    oof_days = list(config.oof_utc_days)
    validation_counts = [270, 270, 269, 269, 269, 269, 269, 269, 269, 269]
    rows: list[dict[str, object]] = []
    for day_index, (day, count) in enumerate(zip(oof_days, validation_counts, strict=True)):
        day_start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
        for market_index in range(count):
            window_start = day_start + timedelta(minutes=5 * market_index)
            rows.append(
                {
                    "market_id": f"oof-{day_index}-{market_index}",
                    "window_start": window_start,
                    "observed_at": window_start + timedelta(seconds=2),
                    "seconds_elapsed": 2,
                    "label_up": (day_index + market_index) % 2,
                    "yes_ask_vwap_5": 0.25,
                    "no_ask_vwap_5": 0.25,
                    "yes_ask_depth": 100.0,
                    "no_ask_depth": 100.0,
                    "yes_cost_per_share": 0.26,
                    "no_cost_per_share": 0.26,
                    "yes_execution_cost_per_share": 0.25,
                    "no_execution_cost_per_share": 0.25,
                    "yes_best_ask": 0.24,
                    "no_best_ask": 0.24,
                    "fee_rate": 0.0,
                }
            )
    calibration_start = min(fold.calibration.start for fold in config.folds)
    first_validation = min(fold.validation.start for fold in config.folds)
    day_cursor = calibration_start
    calibration_index = 0
    while day_cursor < first_validation:
        window_start = day_cursor
        rows.append(
            {
                "market_id": f"calibration-{calibration_index}",
                "window_start": window_start,
                "observed_at": window_start + timedelta(seconds=2),
                "seconds_elapsed": 2,
                "label_up": calibration_index % 2,
                "yes_ask_vwap_5": 0.25,
                "no_ask_vwap_5": 0.25,
                "yes_ask_depth": 100.0,
                "no_ask_depth": 100.0,
                "yes_cost_per_share": 0.26,
                "no_cost_per_share": 0.26,
                "yes_execution_cost_per_share": 0.25,
                "no_execution_cost_per_share": 0.25,
                "yes_best_ask": 0.24,
                "no_best_ask": 0.24,
                "fee_rate": 0.0,
            }
        )
        day_cursor += timedelta(days=1)
        calibration_index += 1
    source = _utc_frame(rows)

    @dataclass(frozen=True)
    class CellSupport:
        name: str
        rows: int = 100
        markets: int = 100
        utc_days: int = 14
        winning_markets: int = 50
        losing_markets: int = 50

    @dataclass(frozen=True)
    class Support:
        rows: int = 200
        markets: int = 100
        utc_days: int = 14
        no_rows: int = 200
        no_markets: int = 100
        no_utc_days: int = 14
        no_winning_markets: int = 50
        no_losing_markets: int = 50
        time_cells: tuple[CellSupport, ...] = (
            CellSupport("NO_1_15"),
            CellSupport("NO_15_30"),
            CellSupport("NO_30_45"),
            CellSupport("NO_45_56"),
        )
        key_sha256: str = "c" * 64

    @dataclass(frozen=True)
    class Evidence:
        support: Support = Support()
        support_passed: bool = True
        support_failures: tuple[str, ...] = ()

    @dataclass(frozen=True)
    class Calibrator:
        arm_id: str
        semantic_sha256: str = "d" * 64
        no_logit_offsets: tuple[tuple[str, float], ...] = ()
        evidence: Evidence = Evidence()

    parent = parent_config()
    validation = source.filter(
        pl.col("window_start").dt.date().cast(pl.String).is_in(
            [day.isoformat() for day in config.oof_utc_days]
        )
    ).sort("market_id", "window_start", "observed_at", "seconds_elapsed")

    def parent_prediction(candidate_id: str) -> pl.DataFrame:
        probability = 0.40 if candidate_id == INCUMBENT_ID else 0.35
        return validation.select(
            "market_id", "window_start", "observed_at", "seconds_elapsed"
        ).with_columns(
            pl.lit(candidate_id).alias("candidate_id"),
            pl.lit(probability).alias("probability_yes"),
        ).sort("market_id", "window_start", "observed_at", "seconds_elapsed")

    monkeypatch.setattr(
        benchmark_module,
        "validate_parent_run_integrity",
        lambda _: {
            "validated_artifact_count": 71,
            "validated_fold_model_count": 50,
            "validated_prediction_count": 6,
        },
    )
    monkeypatch.setattr(
        benchmark_module,
        "_calibration_api",
        lambda: {
            "validate_config": lambda _: None,
            "fit": lambda _config, arm_id, _frame, fold_name=None: Calibrator(arm_id),
            "score": lambda calibrator, frame: SimpleNamespace(
                frame=frame.select(
                    "market_id", "window_start", "observed_at", "seconds_elapsed"
                    ).with_columns(
                        pl.lit(calibrator.arm_id).alias("candidate_name"),
                        pl.lit(
                            0.35
                            + (0.01 if calibrator.arm_id in {"N1", "N2"} else 0.0)
                        ).alias("incumbent_probability_yes"),
                )
            ),
        },
    )
    monkeypatch.setattr(benchmark_module, "load_book_admission_config", lambda _: parent)
    monkeypatch.setattr(
        benchmark_module,
        "load_asymmetric_value_config",
        lambda _: SimpleNamespace(core_config=tmp_path / "core.toml"),
    )
    monkeypatch.setattr(benchmark_module, "load_core_config", lambda _: object())
    monkeypatch.setattr(
        benchmark_module,
        "_load_core_oracle_value_frame",
        lambda *_args, **_kwargs: (source, {"proxy_prices_used": False}),
    )
    incumbent = SimpleNamespace(
        model_sha256=config.incumbent.model_sha256,
        payload={
            "model_key": config.incumbent.model_key,
            "features": {
                "schema_sha256": config.incumbent.feature_schema_sha256,
                "names": [f"feature-{index}" for index in range(75)],
            },
        },
    )
    monkeypatch.setattr(
        benchmark_module, "load_frozen_asymmetric_incumbent", lambda _: incumbent
    )
    monkeypatch.setattr(
        benchmark_module,
        "frozen_parent_probabilities",
        lambda *_args, **_kwargs: [0.4] * source.height,
    )
    monkeypatch.setattr(
        benchmark_module,
        "score_incumbent_calibration_payload",
        lambda *_args, **_kwargs: [0.4] * source.height,
    )
    monkeypatch.setattr(
        benchmark_module, "attach_causal_book_dynamics", lambda frame, **_kwargs: frame
    )
    monkeypatch.setattr(
        benchmark_module,
        "_load_verified_parent_d4_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        benchmark_module,
        "score_book_admission_model",
        lambda _model, frame, _config: SimpleNamespace(
            frame=frame.select(
                "market_id", "window_start", "observed_at", "seconds_elapsed"
            ).with_columns(
                pl.lit("D4").alias("candidate_name"),
                pl.lit(0.35).alias("incumbent_probability_yes"),
            )
        ),
    )
    monkeypatch.setattr(
        benchmark_module,
        "select_target_opportunity_rows",
        lambda frame, *_args, **_kwargs: frame.filter(
            pl.col("seconds_elapsed").is_between(1, 55)
        ),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_read_parent_prediction",
        lambda _config, candidate_id: parent_prediction(candidate_id),
    )

    import btc_directional_model.asymmetric_book_admission as admission_module

    monkeypatch.setattr(
        admission_module,
        "fit_book_admission_candidate",
        lambda *_args, **_kwargs: pytest.fail("D4 must never be refitted"),
    )
    events: list[str] = []

    def select_probability(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("probability_selection")
        run_dirs = list(config.paths.runs.iterdir())
        assert len(run_dirs) == 1
        assert not (run_dirs[0] / "probability-selection-seal.json").exists()
        assert not (run_dirs[0] / "projected-pnl-access.json").exists()
        assert not (run_dirs[0] / "projected-pnl-context.json").exists()
        return {
            "schema_version": "btc-asymmetric-d4-side-calibration-evaluation-v1",
            "status": "diagnostic_selected_consumed_evidence",
            "selected_candidate_id": "N1",
            "evidence_scope": config.evidence_scope,
            "qualification_eligible": False,
            "economics_used": False,
            "failure_trace": [
                {"candidate_id": "N1", "failed_gates": []},
                {"candidate_id": "N2", "failed_gates": ["example_gate"]},
            ],
        }

    def projected_pnl(
        ledgers: dict[str, pl.DataFrame],
        _eligible: pl.DataFrame,
        _selection: dict[str, object],
        _seal: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        events.append("projected_pnl")
        run_dir = next(config.paths.runs.iterdir())
        assert (run_dir / "probability-selection-seal.json").is_file()
        assert (run_dir / "projected-pnl-access.json").is_file()
        assert (run_dir / "projected-pnl-context.json").is_file()
        assert tuple(ledgers) == ARM_IDS
        return {
            "status": "diagnostic_only",
            "all_predeclared_models_reported": list(ARM_IDS),
        }

    monkeypatch.setattr(
        benchmark_module,
        "_evaluation_api",
        lambda: {
            "selection_thresholds": lambda **kwargs: kwargs,
            "projected_pnl_thresholds": lambda **kwargs: kwargs,
            "select": select_probability,
            "projected_pnl": projected_pnl,
        },
    )

    run_dir, result = benchmark_module.run_d4_side_calibration_benchmark(config)

    assert events == ["probability_selection", "projected_pnl"]
    assert result["qualification_eligible"] is False
    assert result["paper_artifact"] is None
    assert result["batch_forward_artifact"] is None
    assert result["runtime_deployable"] is False
    assert result["source_process_changed"] is False
    assert result["parent_d4_refitted"] is False
    assert all(
        (run_dir / "projected-pnl-ledgers" / f"{arm_id}.parquet").is_file()
        for arm_id in ARM_IDS
    )
    report = (run_dir / "benchmark-report.md").read_text()
    assert "Hard 14-day support failures" in report
    assert all(arm_id in report for arm_id in ARM_IDS)
