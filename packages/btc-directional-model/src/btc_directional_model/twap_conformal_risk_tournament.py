"""Resumable tournament runner for frozen-TWAP conformal risk admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from .twap_conformal_risk import (
    CANDIDATES,
    CHECKPOINT_SECONDS,
    CONTROL,
    DIRECTION_PRICE,
    DIRECTION_TIME_PRICE,
    GLOBAL,
    VWAP_QUANTITIES,
    ConformalArtifact,
    apply_admission_contract,
    apply_conformal_bounds,
    artifact_bytes,
    earliest_admitted_trades,
    fit_conformal_artifact,
    load_artifact,
    write_artifact,
)

SCHEMA_VERSION = "btc-twap-conformal-risk-tournament-v1"
EXPECTED_FEATURE_REGISTRY = {
    "upstream_candidate": "twap_single_regime",
    "sensor_names": ["twap30_margin_bps", "twap60_margin_bps"],
    "checkpoint_seconds": list(CHECKPOINT_SECONDS),
    "prediction_outputs": [
        "probability_up",
        "expected_margin_bps",
        "margin_p05_bps",
        "margin_p50_bps",
        "margin_p95_bps",
        "reversal_probability",
        "process_uncertainty_bps2",
        "sensor_uncertainty_bps2",
    ],
    "raw_refprice_admission_input": False,
}
BASE_COLUMNS = (
    "market_id",
    "window_start",
    "window_end",
    "label_source",
    "target_margin_bps",
    "label_up",
    "official_outcome",
    "seconds_elapsed",
    "observed_at",
    "sensor_max_available_at",
    "sensor_valid",
    "fee_rate",
    "quality_flags",
    "up_provider_received_at",
    "down_provider_received_at",
    "book_valid",
    "raw_probability_up",
    "probability_up",
    "expected_margin_bps",
    "margin_p05_bps",
    "margin_p50_bps",
    "margin_p95_bps",
    "reversal_probability",
    "process_uncertainty_bps2",
    "sensor_uncertainty_bps2",
    "fold",
)
VWAP_COLUMNS = tuple(
    f"{direction}_ask_vwap_{quantity}"
    for direction in ("up", "down")
    for quantity in VWAP_QUANTITIES
)


@dataclass(frozen=True)
class FoldLedger:
    name: str
    start: datetime
    end: datetime
    sha256: str


@dataclass(frozen=True)
class TournamentConfig:
    source_path: Path
    package_root: Path
    raw: dict[str, Any]
    upstream_root: Path
    folds: tuple[FoldLedger, ...]
    periods: dict[str, datetime]
    runs: Path
    committed_results: Path


def _utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return parsed.astimezone(UTC)


def load_config(path: Path, upstream_root: Path | None = None) -> TournamentConfig:
    source_path = path.resolve()
    package_root = source_path.parents[1]
    raw = tomllib.loads(source_path.read_text())
    configured_upstream = _path(package_root, raw["upstream"]["package_root"])
    config = TournamentConfig(
        source_path=source_path,
        package_root=package_root,
        raw=raw,
        upstream_root=(upstream_root or configured_upstream).resolve(),
        folds=tuple(
            FoldLedger(row["name"], _utc(row["start"]), _utc(row["end"]), row["sha256"])
            for row in raw["upstream"]["fold_ledgers"]
        ),
        periods={name: _utc(value) for name, value in raw["periods"].items()},
        runs=_path(package_root, raw["paths"]["runs"]),
        committed_results=_path(package_root, raw["paths"]["committed_results"]),
    )
    validate_config(config)
    return config


def validate_config(config: TournamentConfig) -> None:
    raw = config.raw
    if raw["training"] != {
        "profile": "btc_5m_twap_conformal_risk_admission",
        "model_family": "btc-5m-twap-conformal-risk-admission",
        "strictly_training_only": True,
        "live_capital_allowed": False,
        "random_seed": 20260828,
    }:
        raise ValueError("frozen training identity changed")
    if tuple(raw["conformal"]["candidates"]) != CANDIDATES:
        raise ValueError("candidate roster changed")
    if raw["conformal"]["alpha"] != 0.10:
        raise ValueError("alpha changed")
    if raw["conformal"]["minimum_cell_markets"] != 100:
        raise ValueError("conditional support changed")
    entry = raw["entry"]
    if (entry["start_second"], entry["end_second"], entry["cadence_seconds"]) != (30, 120, 5):
        raise ValueError("entry schedule changed")
    if (
        entry["minimum_correctness_lower_bound"],
        entry["maximum_error_risk_upper_bound"],
        entry["maximum_wins_per_loss"],
    ) != (0.90, 0.10, 3.0):
        raise ValueError("shared admission gates changed")
    if tuple(raw["execution"]["quantities"]) != VWAP_QUANTITIES:
        raise ValueError("capacity roster changed")
    expected_periods = {
        "calibration_start": datetime(2026, 8, 14, tzinfo=UTC),
        "calibration_end": datetime(2026, 8, 18, tzinfo=UTC),
        "development_start": datetime(2026, 8, 18, tzinfo=UTC),
        "development_end": datetime(2026, 8, 22, tzinfo=UTC),
        "recalibration_start": datetime(2026, 8, 14, tzinfo=UTC),
        "recalibration_end": datetime(2026, 8, 22, tzinfo=UTC),
        "test_start": datetime(2026, 8, 22, tzinfo=UTC),
        "test_end": datetime(2026, 8, 26, tzinfo=UTC),
    }
    if config.periods != expected_periods:
        raise ValueError("chronological split changed")
    expected_fold_boundaries = tuple(
        (datetime(2026, 8, day, tzinfo=UTC), datetime(2026, 8, day + 2, tzinfo=UTC))
        for day in (14, 16, 18, 20, 22, 24)
    )
    if tuple((fold.start, fold.end) for fold in config.folds) != expected_fold_boundaries:
        raise ValueError("upstream OOS fold roster changed")


def run_tournament(config: TournamentConfig) -> tuple[Path, dict[str, Any]]:
    source_commit = _git_revision(config.package_root)
    input_identity = _hash_payload(
        {
            "schema": SCHEMA_VERSION,
            "config": file_sha256(config.source_path),
            "source_commit": source_commit,
            "upstream_artifact": config.raw["upstream"]["artifact_sha256"],
            "folds": [fold.sha256 for fold in config.folds],
        }
    )
    workspace = config.runs / input_identity[:20]
    workspace.mkdir(parents=True, exist_ok=True)
    completion = workspace / "completion.json"
    if completion.is_file():
        record = json.loads(completion.read_text())
        result = config.package_root / record["result"]
        if file_sha256(result / "metrics.json") != record["metrics_sha256"]:
            raise RuntimeError("completed result metrics changed")
        return result, json.loads((result / "metrics.json").read_text())

    upstream = verify_upstream(config)
    _checkpoint_json(workspace / "source-verification.json", input_identity, upstream)

    pretest_folds = config.folds[:4]
    pretest, pretest_inventory = load_prediction_folds(config, pretest_folds)
    calibration = _period(
        pretest, config.periods["calibration_start"], config.periods["calibration_end"]
    )
    development = _period(
        pretest, config.periods["development_start"], config.periods["development_end"]
    )
    verify_split_separation(calibration, development, None)
    pretest_manifest = persist_daily_ledgers(
        pretest, workspace / "frozen-prediction-ledgers" / "pretest", input_identity
    )
    calibration_hash = _role_ledger_hash(
        pretest_manifest, config.periods["calibration_start"], config.periods["calibration_end"]
    )
    recalibration_hash = _role_ledger_hash(
        pretest_manifest,
        config.periods["recalibration_start"],
        config.periods["recalibration_end"],
    )

    dev_artifacts: dict[str, ConformalArtifact] = {}
    conformity: dict[str, Any] = {}
    development_metrics: dict[str, Any] = {}
    development_decisions: dict[str, pl.DataFrame] = {}
    development_trades: dict[str, pl.DataFrame] = {}
    for candidate in CANDIDATES:
        artifact_path = workspace / "candidate-calibration" / f"{candidate}.json"
        if artifact_path.is_file():
            artifact = load_artifact(artifact_path)
            if artifact.prediction_ledger_sha256 != calibration_hash:
                raise RuntimeError(f"calibration checkpoint input changed: {candidate}")
            print(f"checkpoint resume: calibration {candidate}", flush=True)
        else:
            artifact, records = fit_conformal_artifact(
                calibration,
                candidate,
                alpha=float(config.raw["conformal"]["alpha"]),
                minimum_cell_markets=int(config.raw["conformal"]["minimum_cell_markets"]),
                calibration_start=config.periods["calibration_start"].isoformat(),
                calibration_end=config.periods["calibration_end"].isoformat(),
                source_identity=upstream["source_identity"],
                prediction_ledger_sha256=calibration_hash,
                feature_registry_sha256=upstream["feature_registry_sha256"],
                upstream_artifact_sha256=upstream["artifact_sha256"],
            )
            write_artifact(artifact_path, artifact)
            if records.height:
                _write_parquet_atomic(
                    workspace / "candidate-calibration" / f"{candidate}-market-blocks.parquet",
                    records,
                )
            print(f"checkpoint complete: calibration {candidate}", flush=True)
        dev_artifacts[candidate] = artifact
        conformity[candidate] = calibration_report(calibration, artifact)
        decision_path = workspace / "development" / f"{candidate}-decisions.parquet"
        trade_path = workspace / "development" / f"{candidate}-trades.parquet"
        metrics_path = workspace / "development" / f"{candidate}-metrics.json"
        phase_hash = _hash_payload(
            {
                "artifact": hashlib.sha256(artifact_bytes(artifact)).hexdigest(),
                "development": _frame_identity(development),
                "config": file_sha256(config.source_path),
            }
        )
        if decision_path.is_file() and trade_path.is_file() and metrics_path.is_file():
            saved = json.loads(metrics_path.read_text())
            if saved["input_hash"] != phase_hash:
                raise RuntimeError(f"development checkpoint input changed: {candidate}")
            decisions = pl.read_parquet(decision_path)
            trades = pl.read_parquet(trade_path)
            metrics = saved["value"]
            print(f"checkpoint resume: development {candidate}", flush=True)
        else:
            decisions, trades, metrics = evaluate_candidate(development, artifact, config)
            _write_parquet_atomic(decision_path, decisions)
            _write_parquet_atomic(trade_path, trades)
            _write_json(metrics_path, {"input_hash": phase_hash, "value": metrics})
            print(f"checkpoint complete: development {candidate}", flush=True)
        development_decisions[candidate] = decisions
        development_trades[candidate] = trades
        development_metrics[candidate] = metrics

    paired = {
        candidate: paired_comparison(
            development,
            development_trades[candidate],
            development_trades[CONTROL],
            int(config.raw["qualification"]["bootstrap_resamples"]),
            int(config.raw["training"]["random_seed"]) + index * 101,
        )
        for index, candidate in enumerate(CANDIDATES)
    }
    selection = select_development_candidate(development_metrics, paired, config)
    _checkpoint_json(workspace / "development-selection.json", input_identity, selection)

    selected = str(selection["selected_candidate"])
    frozen_path = workspace / "frozen-selected-conformal-artifact.json"
    if frozen_path.is_file():
        frozen_artifact = load_artifact(frozen_path)
        if frozen_artifact.candidate != selected:
            raise RuntimeError("selected artifact candidate changed")
        if frozen_artifact.prediction_ledger_sha256 != recalibration_hash:
            raise RuntimeError("selected artifact ledger identity changed")
        frozen_artifact_sha = file_sha256(frozen_path)
        print("checkpoint resume: selected recalibration", flush=True)
    else:
        frozen_artifact, selected_blocks = fit_conformal_artifact(
            pretest,
            selected,
            alpha=float(config.raw["conformal"]["alpha"]),
            minimum_cell_markets=int(config.raw["conformal"]["minimum_cell_markets"]),
            calibration_start=config.periods["recalibration_start"].isoformat(),
            calibration_end=config.periods["recalibration_end"].isoformat(),
            source_identity=upstream["source_identity"],
            prediction_ledger_sha256=recalibration_hash,
            feature_registry_sha256=upstream["feature_registry_sha256"],
            upstream_artifact_sha256=upstream["artifact_sha256"],
        )
        frozen_artifact_sha = write_artifact(frozen_path, frozen_artifact)
        if selected_blocks.height:
            _write_parquet_atomic(workspace / "selected-recalibration-market-blocks.parquet", selected_blocks)
        print("checkpoint complete: selected recalibration and freeze", flush=True)

    reloaded = load_artifact(frozen_path)
    if artifact_bytes(frozen_artifact) != artifact_bytes(reloaded):
        raise RuntimeError("conformal artifact serialization changed")
    development_reload = apply_conformal_bounds(development, reloaded)
    development_original = apply_conformal_bounds(development, frozen_artifact)
    _assert_decision_columns_equal(development_original, development_reload)

    # The untouched test is not loaded until selection and artifact freezing are complete.
    test, test_inventory = load_prediction_folds(config, config.folds[4:])
    verify_split_separation(calibration, development, test)
    test_manifest = persist_daily_ledgers(
        test, workspace / "frozen-prediction-ledgers" / "test", input_identity
    )
    test_hash = _role_ledger_hash(
        test_manifest, config.periods["test_start"], config.periods["test_end"]
    )
    test_decisions, test_trades, test_metrics = evaluate_candidate(test, reloaded, config)
    _write_parquet_atomic(workspace / "test" / f"{selected}-decisions.parquet", test_decisions)
    _write_parquet_atomic(workspace / "test" / f"{selected}-trades.parquet", test_trades)
    test_qualification = apply_test_qualification(
        test_metrics, development_metrics[selected], config
    )
    conclusion = definitive_conclusion(
        selection, development_metrics, paired, test_metrics, test_qualification
    )
    all_prediction_hash = _hash_payload(
        {
            "pretest": recalibration_hash,
            "test": test_hash,
            "daily": pretest_manifest["days"] + test_manifest["days"],
        }
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_family": config.raw["training"]["model_family"],
        "source_commit": source_commit,
        "configuration_sha256": file_sha256(config.source_path),
        "upstream": upstream,
        "split_audit": split_audit(config, calibration, development, test),
        "input_inventory": {
            "calibration_and_development": pretest_inventory,
            "untouched_test": test_inventory,
            "partial_markets_excluded": pretest_inventory["partial_markets_excluded"]
            + test_inventory["partial_markets_excluded"],
        },
        "prediction_ledgers": {
            "pretest": pretest_manifest,
            "untouched_test": test_manifest,
            "recalibration_sha256": recalibration_hash,
            "test_sha256": test_hash,
            "all_sha256": all_prediction_hash,
        },
        "calibration": conformity,
        "development": {
            "candidates": development_metrics,
            "paired_against_control": paired,
            "selection": selection,
        },
        "selected_artifact": {
            "candidate": selected,
            "sha256": frozen_artifact_sha,
            "calibration_start": config.periods["recalibration_start"],
            "calibration_end": config.periods["recalibration_end"],
            "serialization_reload_identical": True,
            "batch_reload_decisions_identical": True,
        },
        "untouched_test": {
            "candidate": selected,
            "metrics": test_metrics,
            "qualification": test_qualification,
            "test_read_after_artifact_freeze": True,
            "candidate_substitution_performed": False,
            "adjustment_performed": False,
        },
        "conclusion": conclusion,
        "deployment_status": "not_deployed_training_only",
        "database_mutations": False,
        "new_data_sources": False,
        "new_ingesters": False,
        "new_tables": False,
    }
    result = config.committed_results / run_id
    temporary = result.with_name(result.name + ".partial")
    temporary.mkdir(parents=True, exist_ok=False)
    _write_json(temporary / "metrics.json", metrics)
    (temporary / "report.md").write_text(render_report(metrics))
    (temporary / "conformal-artifact.json").write_bytes(frozen_path.read_bytes())
    (temporary / "conformal-artifact.sha256").write_text(frozen_artifact_sha + "\n")
    _write_json(
        temporary / "model-provenance.json",
        {
            "schema_version": "btc-model-provenance-v1",
            "model_family": config.raw["training"]["model_family"],
            "conformal_artifact_sha256": frozen_artifact_sha,
            "upstream_model_tag": config.raw["upstream"]["model_tag"],
            "upstream_artifact_sha256": upstream["artifact_sha256"],
            "source_manifest_sha256": upstream["source_manifest_sha256"],
            "prediction_ledger_sha256": all_prediction_hash,
            "feature_registry_sha256": upstream["feature_registry_sha256"],
            "calibration_dates": "2026-08-14/2026-08-17",
            "development_dates": "2026-08-18/2026-08-21",
            "test_dates": "2026-08-22/2026-08-25",
            "candidate": selected,
            "qualification_status": test_qualification["status"],
            "deployment_status": "not_deployed_training_only",
            "producing_commit": source_commit,
            "training_run_id": run_id,
        },
    )
    ledger_root = temporary / "ledgers"
    ledger_root.mkdir()
    for candidate in CANDIDATES:
        _write_parquet_atomic(
            ledger_root / f"development-{candidate}-trades.parquet",
            development_trades[candidate],
        )
    _write_parquet_atomic(ledger_root / f"test-{selected}-trades.parquet", test_trades)
    _write_parquet_atomic(
        ledger_root / f"test-{selected}-decisions.parquet",
        test_decisions,
    )
    os.replace(temporary, result)
    completion_record = {
        "input_hash": input_identity,
        "result": str(result.relative_to(config.package_root)),
        "metrics_sha256": file_sha256(result / "metrics.json"),
    }
    _write_json(completion, completion_record)
    return result, metrics


def verify_upstream(config: TournamentConfig) -> dict[str, Any]:
    raw = config.raw["upstream"]
    artifact_path = config.upstream_root / raw["artifact_path"]
    manifest_path = config.upstream_root / raw["source_manifest_path"]
    frame_manifest_path = config.upstream_root / raw["training_frame_manifest_path"]
    checks = (
        (artifact_path, raw["artifact_sha256"]),
        (manifest_path, raw["source_manifest_sha256"]),
        (frame_manifest_path, raw["training_frame_manifest_sha256"]),
    )
    for path, expected in checks:
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"immutable upstream input changed or is missing: {path}")
    tag_commit = subprocess.run(
        ["git", "rev-parse", f"{raw['model_tag']}^{{commit}}"],
        cwd=config.package_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tag_commit != raw["artifact_recording_commit"]:
        raise RuntimeError("upstream model tag moved")
    artifact = joblib.load(artifact_path)
    if artifact["model_family"] != raw["model_family"]:
        raise RuntimeError("upstream family changed")
    model = artifact["models"].get(raw["candidate"])
    if model is None:
        raise RuntimeError("frozen upstream candidate is missing")
    candidate = model["candidate"]
    if (
        candidate.get("name") != "twap_single_regime"
        or tuple(candidate.get("sensors", ()))
        != ("twap30_margin_bps", "twap60_margin_bps")
        or candidate.get("regime_switching") is not False
    ):
        raise RuntimeError("upstream TWAP-only candidate contract changed")
    feature_hash = _hash_payload(EXPECTED_FEATURE_REGISTRY)
    if feature_hash != raw["feature_registry_sha256"]:
        raise RuntimeError("causal feature-registry identity changed")
    return {
        "model_family": raw["model_family"],
        "candidate": raw["candidate"],
        "model_tag": raw["model_tag"],
        "artifact_recording_commit": tag_commit,
        "artifact_sha256": file_sha256(artifact_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_identity": raw["source_identity"],
        "feature_registry_sha256": feature_hash,
        "refit_performed": False,
        "raw_refprice_used": False,
    }


def load_prediction_folds(
    config: TournamentConfig, folds: tuple[FoldLedger, ...]
) -> tuple[pl.DataFrame, dict[str, Any]]:
    frames: list[pl.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    checkpoint_root = config.upstream_root / config.raw["upstream"]["checkpoint_root"]
    for fold in folds:
        path = checkpoint_root / f"fold-twap_single_regime-{fold.name}.joblib"
        digest_path = path.with_suffix(".sha256")
        if not path.is_file() or file_sha256(path) != fold.sha256:
            raise RuntimeError(f"frozen OOS prediction checkpoint changed: {path}")
        if not digest_path.is_file() or digest_path.read_text().strip() != fold.sha256:
            raise RuntimeError(f"upstream checkpoint digest changed: {path}")
        payload = joblib.load(path)
        value = payload.get("value", {})
        report = value.get("report", {})
        parameters = value.get("parameters", {})
        if report.get("name") != fold.name:
            raise RuntimeError("upstream fold identity changed")
        if (_utc(report["test_start"]), _utc(report["test_end"])) != (fold.start, fold.end):
            raise RuntimeError("upstream fold date boundary changed")
        if parameters.get("candidate_name") != "twap_single_regime":
            raise RuntimeError("upstream fold used a different candidate")
        if tuple(parameters.get("sensor_names", ())) != (
            "twap30_margin_bps",
            "twap60_margin_bps",
        ):
            raise RuntimeError("upstream fold used RefPrice or changed sensors")
        scored = pl.DataFrame(value["scored"])
        missing = sorted(set(BASE_COLUMNS + VWAP_COLUMNS) - set(scored.columns))
        if missing:
            raise RuntimeError(f"upstream prediction ledger lacks columns: {missing}")
        scored = scored.select(BASE_COLUMNS + VWAP_COLUMNS).filter(
            pl.col("seconds_elapsed").is_in(CHECKPOINT_SECONDS)
        ).sort(["window_start", "market_id", "seconds_elapsed"])
        validate_prediction_frame(scored, fold)
        complete = scored["market_id"].n_unique()
        scheduled = int(report["testing_markets"])
        inventory.append(
            {
                "fold": fold.name,
                "scheduled_markets": scheduled,
                "complete_prediction_markets": complete,
                "partial_markets_excluded": scheduled - complete,
                "rows": scored.height,
                "checkpoint_sha256": fold.sha256,
                "upstream_input_hash": payload.get("input_hash"),
            }
        )
        frames.append(scored)
    combined = pl.concat(frames, how="vertical").sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    return combined, {
        "folds": inventory,
        "scheduled_markets": sum(row["scheduled_markets"] for row in inventory),
        "complete_prediction_markets": combined["market_id"].n_unique(),
        "partial_markets_excluded": sum(row["partial_markets_excluded"] for row in inventory),
        "rows": combined.height,
    }


def validate_prediction_frame(frame: pl.DataFrame, fold: FoldLedger) -> None:
    if frame.is_empty():
        raise RuntimeError("upstream OOS prediction ledger is empty")
    if frame.filter(
        ~pl.col("window_start").is_between(fold.start, fold.end, closed="left")
    ).height:
        raise RuntimeError("prediction ledger contains rows outside its OOS fold")
    if set(frame["label_source"].unique()) != {"authentic_official_twap60"}:
        raise RuntimeError("non-authentic labels entered the tournament")
    if frame.filter(pl.col("sensor_max_available_at") >= pl.col("observed_at")).height:
        raise RuntimeError("upstream predictions violate causal availability")
    if frame.filter(~pl.col("sensor_valid")).height:
        raise RuntimeError("invalid upstream sensor rows entered the ledger")
    counts = frame.group_by("market_id").agg(
        pl.len().alias("rows"), pl.col("seconds_elapsed").n_unique().alias("seconds")
    )
    if counts.filter(
        (pl.col("rows") != len(CHECKPOINT_SECONDS))
        | (pl.col("seconds") != len(CHECKPOINT_SECONDS))
    ).height:
        raise RuntimeError("partial checkpoint trajectory entered the tournament")
    if frame["market_id"].n_unique() * len(CHECKPOINT_SECONDS) != frame.height:
        raise RuntimeError("checkpoint rows are not one complete trajectory per market")
    if "refprice_margin_bps" in frame.columns:
        raise RuntimeError("raw RefPrice entered conformal admission")


def persist_daily_ledgers(frame: pl.DataFrame, root: Path, input_hash: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    days: list[dict[str, Any]] = []
    daily = frame.with_columns(pl.col("window_start").dt.date().alias("ledger_day"))
    for partition in daily.partition_by("ledger_day", maintain_order=True):
        ledger_day = partition["ledger_day"][0]
        values = partition.drop("ledger_day")
        path = root / f"{ledger_day.isoformat()}.parquet"
        manifest_path = path.with_suffix(".json")
        if path.is_file() and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest["input_hash"] != input_hash or file_sha256(path) != manifest["sha256"]:
                raise RuntimeError(f"daily prediction checkpoint changed: {ledger_day}")
            print(f"checkpoint resume: prediction ledger {ledger_day}", flush=True)
        else:
            _write_parquet_atomic(path, values)
            manifest = {
                "input_hash": input_hash,
                "day": ledger_day.isoformat(),
                "rows": values.height,
                "markets": values["market_id"].n_unique(),
                "sha256": file_sha256(path),
            }
            _write_json(manifest_path, manifest)
            print(f"checkpoint complete: prediction ledger {ledger_day}", flush=True)
        days.append(manifest)
    return {"days": days, "combined_sha256": _hash_payload(days)}


def verify_split_separation(
    calibration: pl.DataFrame, development: pl.DataFrame, test: pl.DataFrame | None
) -> None:
    roles = {"calibration": calibration, "development": development}
    if test is not None:
        roles["test"] = test
    ids = {name: set(frame["market_id"].unique()) for name, frame in roles.items()}
    names = tuple(roles)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if ids[left] & ids[right]:
                raise RuntimeError(f"market overlap between {left} and {right}")
    ordered = [frame["window_start"].min() for frame in roles.values()]
    if ordered != sorted(ordered):
        raise RuntimeError("data roles are not chronologically ordered")


def split_audit(
    config: TournamentConfig,
    calibration: pl.DataFrame,
    development: pl.DataFrame,
    test: pl.DataFrame,
) -> dict[str, Any]:
    roles = {"calibration": calibration, "development": development, "untouched_test": test}
    return {
        "passed": True,
        "chronological": True,
        "market_disjoint": True,
        "roles": {
            name: {
                "start": frame["window_start"].min(),
                "end_exclusive": frame["window_end"].max(),
                "markets": frame["market_id"].n_unique(),
                "rows": frame.height,
                "checkpoint_seconds": list(CHECKPOINT_SECONDS),
            }
            for name, frame in roles.items()
        },
        "test_read_after_freeze": True,
        "random_split": False,
        "database_reads": False,
        "expected_boundaries": config.periods,
    }


def calibration_report(frame: pl.DataFrame, artifact: ConformalArtifact) -> dict[str, Any]:
    bounded = apply_conformal_bounds(frame, artifact)
    coverage = conformal_coverage_metrics(bounded)
    cells = {
        level: {
            key: {
                "support_markets": cell.support_markets,
                "probability_residual_quantile": cell.probability_residual_quantile,
                "margin_residual_quantile_bps": cell.margin_residual_quantile_bps,
            }
            for key, cell in rows.items()
        }
        for level, rows in artifact.cells.items()
    }
    return {"candidate": artifact.candidate, "coverage": coverage, "cells": cells}


def evaluate_candidate(
    frame: pl.DataFrame, artifact: ConformalArtifact, config: TournamentConfig
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    bounded = apply_conformal_bounds(frame, artifact)
    decisions = apply_admission_contract(
        bounded,
        reserve_per_share=float(config.raw["execution"]["execution_reserve_per_share"]),
        stress_slippage_per_share=float(
            config.raw["execution"]["stress_slippage_per_share"]
        ),
        minimum_correctness=float(config.raw["entry"]["minimum_correctness_lower_bound"]),
        maximum_error_risk=float(config.raw["entry"]["maximum_error_risk_upper_bound"]),
        maximum_recovery_ratio=float(config.raw["entry"]["maximum_wins_per_loss"]),
        evaluation_quantity=int(config.raw["execution"]["evaluation_quantity"]),
    )
    trades = earliest_admitted_trades(decisions)
    metrics = candidate_metrics(frame, decisions, trades, config)
    return decisions, trades, metrics


def candidate_metrics(
    frame: pl.DataFrame,
    decisions: pl.DataFrame,
    trades: pl.DataFrame,
    config: TournamentConfig,
) -> dict[str, Any]:
    markets = frame["market_id"].n_unique()
    predictive_all = probability_metrics(frame)
    coverage = conformal_coverage_metrics(decisions)
    abstention = {
        str(row["abstention_reason"]): int(row["count"])
        for row in decisions.group_by("abstention_reason").len(name="count").sort(
            "abstention_reason"
        ).iter_rows(named=True)
    }
    fallback = {
        str(row["conformal_fallback"]): int(row["count"])
        for row in decisions.group_by("conformal_fallback").len(name="count").sort(
            "conformal_fallback"
        ).iter_rows(named=True)
    }
    if trades.is_empty():
        empty_group = group_metrics(trades, config)
        days = sorted(str(value) for value in frame["window_start"].dt.date().unique())
        folds = sorted(str(value) for value in frame["fold"].unique())
        entry_bands = {
            name: dict(empty_group) for name in ("30-59", "60-89", "90-120")
        }
        price_bands = {
            name: dict(empty_group)
            for name in ("below_0.60", "0.60-0.70", "0.70-0.80", "above_0.80")
        }
        margin_edges = tuple(
            float(value) for value in config.raw["reporting"]["predicted_margin_bands_bps"]
        )
        margin_names = [
            f"{lower:g}-{upper:g}" for lower, upper in pairwise(margin_edges)
        ] + [f"above_{margin_edges[-1]:g}"]
        return {
            "markets": markets,
            "trades": 0,
            "coverage": 0.0,
            "accuracy": None,
            "up_accuracy": None,
            "down_accuracy": None,
            "up_trade_share": 0.0,
            "down_trade_share": 0.0,
            "wins": 0,
            "losses": 0,
            "average_win": None,
            "average_loss": None,
            "wins_to_recover_average_loss": None,
            "loss_distribution": {
                "average": None,
                "median": None,
                "p90": None,
                "worst": None,
            },
            "gross_pnl": 0.0,
            "fee_adjusted_pnl": 0.0,
            "net_pnl": 0.0,
            "stressed_pnl": 0.0,
            "stressed_expectancy": None,
            "profit_factor": None,
            "maximum_drawdown": 0.0,
            "cvar_10": None,
            "mean_entry_second": None,
            "median_entry_second": None,
            "p10_entry_second": None,
            "p50_entry_second": None,
            "p90_entry_second": None,
            "predictive_all": predictive_all,
            "admitted_probability_metrics": probability_metrics(trades),
            "conformal": coverage,
            "abstention_reasons": abstention,
            "fallback_frequency": fallback,
            "daily": {day: dict(empty_group) for day in days},
            "two_day_folds": {fold: dict(empty_group) for fold in folds},
            "by_direction": {"UP": dict(empty_group), "DOWN": dict(empty_group)},
            "entry_time_bands": entry_bands,
            "executable_price_bands": price_bands,
            "predicted_margin_bands": {
                name: dict(empty_group) for name in margin_names
            },
            "capacity": capacity_metrics(trades, config),
            "maximum_positive_pnl_day_share": None,
            "maximum_quote_loss_recovery_ratio": None,
            "all_quotes_pass_recovery": False,
            "bootstrap_stressed_expectancy": empty_interval(
                int(config.raw["qualification"]["bootstrap_resamples"])
            ),
        }
    pnl = trades["stressed_pnl"].to_numpy().astype(float)
    correct = trades["direction_correct"].to_numpy().astype(bool)
    winners = pnl[pnl > 0.0]
    losers = pnl[pnl < 0.0]
    loss_magnitudes = -losers
    average_win = float(winners.mean()) if len(winners) else 0.0
    average_loss = float(losers.mean()) if len(losers) else 0.0
    ordered = trades.sort(["window_start", "market_id"])
    cumulative = np.cumsum(ordered["stressed_pnl"].to_numpy().astype(float))
    prior_peak = np.maximum.accumulate(np.r_[0.0, cumulative])[:-1]
    drawdown = prior_peak - cumulative
    by_direction = {
        name: group_metrics(trades.filter(pl.col("predicted_up") == value), config, seed_offset)
        for name, value, seed_offset in (("UP", True, 11), ("DOWN", False, 29))
    }
    entry_bands = band_metrics(
        trades,
        "seconds_elapsed",
        (("30-59", 30.0, 60.0), ("60-89", 60.0, 90.0), ("90-120", 90.0, 121.0)),
    )
    price_bands = band_metrics(
        trades,
        "selected_cost_5",
        (
            ("below_0.60", 0.0, 0.60),
            ("0.60-0.70", 0.60, 0.70),
            ("0.70-0.80", 0.70, 0.80),
            ("above_0.80", 0.80, 1.01),
        ),
    )
    margin_bands = predicted_margin_band_metrics(trades, config)
    daily = grouped_time_metrics(trades, "day")
    folds = grouped_time_metrics(trades, "fold")
    positive_daily = np.array(
        [row["stressed_pnl"] for row in daily.values() if row["stressed_pnl"] > 0], dtype=float
    )
    concentration = (
        float(positive_daily.max() / positive_daily.sum()) if len(positive_daily) else math.inf
    )
    entry = trades["seconds_elapsed"].to_numpy().astype(float)
    bootstrap = hierarchical_bootstrap(
        trades,
        "stressed_pnl",
        int(config.raw["qualification"]["bootstrap_resamples"]),
        int(config.raw["training"]["random_seed"]),
    )
    return {
        "markets": markets,
        "trades": trades.height,
        "coverage": trades.height / max(markets, 1),
        "accuracy": float(correct.mean()),
        "up_accuracy": by_direction["UP"]["accuracy"],
        "down_accuracy": by_direction["DOWN"]["accuracy"],
        "up_trade_share": by_direction["UP"]["trades"] / trades.height,
        "down_trade_share": by_direction["DOWN"]["trades"] / trades.height,
        "wins": int(correct.sum()),
        "losses": int((~correct).sum()),
        "average_win": average_win,
        "average_loss": average_loss,
        "wins_to_recover_average_loss": (
            abs(average_loss) / average_win if average_win > 0.0 else math.inf
        ),
        "loss_distribution": {
            "average": float(loss_magnitudes.mean()) if len(loss_magnitudes) else None,
            "median": float(np.median(loss_magnitudes)) if len(loss_magnitudes) else None,
            "p90": float(np.quantile(loss_magnitudes, 0.90)) if len(loss_magnitudes) else None,
            "worst": float(loss_magnitudes.max()) if len(loss_magnitudes) else None,
        },
        "gross_pnl": float(trades["gross_pnl"].sum()),
        "fee_adjusted_pnl": float(trades["fee_adjusted_pnl"].sum()),
        "net_pnl": float(trades["net_pnl"].sum()),
        "stressed_pnl": float(pnl.sum()),
        "stressed_expectancy": float(pnl.mean()),
        "profit_factor": float(winners.sum() / -losers.sum()) if len(losers) else math.inf,
        "bootstrap_stressed_expectancy": bootstrap,
        "maximum_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "cvar_10": cvar(pnl),
        "mean_entry_second": float(entry.mean()),
        "median_entry_second": float(np.median(entry)),
        "p10_entry_second": float(np.quantile(entry, 0.10)),
        "p50_entry_second": float(np.quantile(entry, 0.50)),
        "p90_entry_second": float(np.quantile(entry, 0.90)),
        "daily": daily,
        "two_day_folds": folds,
        "by_direction": by_direction,
        "entry_time_bands": entry_bands,
        "executable_price_bands": price_bands,
        "predicted_margin_bands": margin_bands,
        "capacity": capacity_metrics(trades, config),
        "maximum_positive_pnl_day_share": concentration,
        "maximum_quote_loss_recovery_ratio": float(
            trades["quoted_loss_recovery_ratio"].max()
        ),
        "all_quotes_pass_recovery": bool(
            trades["quoted_loss_recovery_ratio"].max()
            <= float(config.raw["entry"]["maximum_wins_per_loss"])
        ),
        "predictive_all": predictive_all,
        "admitted_probability_metrics": probability_metrics(trades),
        "conformal": coverage,
        "abstention_reasons": abstention,
        "fallback_frequency": fallback,
    }


def probability_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {"brier_score": None, "log_loss": None, "ece": None, "rows": 0, "markets": 0}
    probability = np.clip(frame["probability_up"].to_numpy().astype(float), 1e-9, 1 - 1e-9)
    labels = frame["label_up"].to_numpy().astype(float)
    counts = frame.group_by("market_id").len(name="market_rows")
    weighted = frame.select("market_id").join(counts, on="market_id", how="left")
    weights = 1.0 / weighted["market_rows"].to_numpy().astype(float)
    weights /= weights.sum()
    losses = -(labels * np.log(probability) + (1.0 - labels) * np.log(1.0 - probability))
    return {
        "brier_score": float(np.sum(weights * (probability - labels) ** 2)),
        "log_loss": float(np.sum(weights * losses)),
        "ece": ece(labels, probability, weights),
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
    }


def conformal_coverage_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {
            "empirical_conformal_coverage": None,
            "margin_interval_coverage": None,
            "mean_margin_interval_width_bps": None,
            "error_risk_bound_calibration": {},
            "market_blocks": 0,
        }
    eligible = frame.filter(pl.col("book_valid") & pl.col("selected_cost_5").is_finite())
    if eligible.is_empty():
        return {
            "empirical_conformal_coverage": None,
            "margin_interval_coverage": None,
            "mean_margin_interval_width_bps": None,
            "error_risk_bound_calibration": {},
            "market_blocks": 0,
        }
    blocks = eligible.with_columns(
        (
            (pl.col("predicted_up") == pl.col("label_up")).cast(pl.Float64)
            >= pl.col("correctness_lower_bound")
        ).alias("probability_covered"),
        pl.col("target_margin_bps").is_between(
            pl.col("conformal_margin_lower"), pl.col("conformal_margin_upper"), closed="both"
        ).alias("margin_covered"),
        (pl.col("conformal_margin_upper") - pl.col("conformal_margin_lower")).alias(
            "margin_width"
        ),
    ).group_by(["market_id", "direction_price_group"]).agg(
        pl.col("probability_covered").all(),
        pl.col("margin_covered").all(),
        pl.col("margin_width").mean(),
        pl.col("error_risk_upper_bound").mean(),
        (~(pl.col("predicted_up") == pl.col("label_up"))).cast(pl.Float64).mean().alias(
            "actual_error"
        ),
    )
    actual_error = float(blocks["actual_error"].mean())
    risk_bound = float(blocks["error_risk_upper_bound"].mean())
    return {
        "empirical_conformal_coverage": float(blocks["probability_covered"].mean()),
        "margin_interval_coverage": float(blocks["margin_covered"].mean()),
        "mean_margin_interval_width_bps": float(blocks["margin_width"].mean()),
        "median_margin_interval_width_bps": float(blocks["margin_width"].median()),
        "p90_margin_interval_width_bps": float(blocks["margin_width"].quantile(0.90)),
        "error_risk_bound_calibration": {
            "actual_error_rate": actual_error,
            "mean_error_risk_upper_bound": risk_bound,
            "bound_minus_actual": risk_bound - actual_error,
        },
        "market_blocks": blocks.height,
    }


def group_metrics(
    frame: pl.DataFrame, config: TournamentConfig | None = None, seed_offset: int = 0
) -> dict[str, Any]:
    if frame.is_empty():
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "accuracy": None,
            "stressed_pnl": 0.0,
            "stressed_expectancy": None,
            "bootstrap_stressed_expectancy": empty_interval(0),
        }
    pnl = frame["stressed_pnl"].to_numpy().astype(float)
    correct = frame["direction_correct"].to_numpy().astype(bool)
    resamples = int(config.raw["qualification"]["bootstrap_resamples"]) if config else 0
    seed = int(config.raw["training"]["random_seed"]) + seed_offset if config else 0
    return {
        "trades": frame.height,
        "wins": int(correct.sum()),
        "losses": int((~correct).sum()),
        "accuracy": float(correct.mean()),
        "stressed_pnl": float(pnl.sum()),
        "stressed_expectancy": float(pnl.mean()),
        "average_win": float(pnl[pnl > 0].mean()) if np.any(pnl > 0) else None,
        "average_loss": float(pnl[pnl < 0].mean()) if np.any(pnl < 0) else None,
        "bootstrap_stressed_expectancy": (
            hierarchical_bootstrap(frame, "stressed_pnl", resamples, seed)
            if resamples
            else empty_interval(0)
        ),
    }


def band_metrics(
    frame: pl.DataFrame, column: str, bands: tuple[tuple[str, float, float], ...]
) -> dict[str, Any]:
    return {
        name: group_metrics(frame.filter(pl.col(column).is_between(lower, upper, closed="left")))
        for name, lower, upper in bands
    }


def predicted_margin_band_metrics(
    frame: pl.DataFrame, config: TournamentConfig
) -> dict[str, Any]:
    edges = tuple(float(value) for value in config.raw["reporting"]["predicted_margin_bands_bps"])
    absolute = frame.with_columns(pl.col("expected_margin_bps").abs().alias("abs_margin"))
    bands = []
    for lower, upper in pairwise(edges):
        bands.append((f"{lower:g}-{upper:g}", lower, upper))
    bands.append((f"above_{edges[-1]:g}", edges[-1], math.inf))
    return band_metrics(absolute, "abs_margin", tuple(bands))


def grouped_time_metrics(frame: pl.DataFrame, grouping: str) -> dict[str, Any]:
    if grouping == "day":
        working = frame.with_columns(pl.col("window_start").dt.date().cast(pl.String).alias("group"))
    else:
        working = frame.with_columns(pl.col("fold").alias("group"))
    return {
        str(group["group"][0]): group_metrics(group)
        for group in working.partition_by("group", maintain_order=True)
    }


def capacity_metrics(frame: pl.DataFrame, config: TournamentConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    reserve = float(config.raw["execution"]["execution_reserve_per_share"])
    stress = float(config.raw["execution"]["stress_slippage_per_share"])
    for quantity in VWAP_QUANTITIES:
        if frame.is_empty():
            result[str(quantity)] = {
                "supported_trades": 0,
                "net_pnl": 0.0,
                "stressed_pnl": 0.0,
                "stressed_expectancy": None,
            }
            continue
        predicted = frame["predicted_up"].to_numpy().astype(bool)
        up = frame[f"up_ask_vwap_{quantity}"].fill_null(float("nan")).to_numpy().astype(float)
        down = frame[f"down_ask_vwap_{quantity}"].fill_null(float("nan")).to_numpy().astype(float)
        cost = np.where(predicted, up, down)
        valid = np.isfinite(cost)
        fee_rate = frame["fee_rate"].fill_null(float("nan")).to_numpy().astype(float)
        fee = fee_rate * cost * (1.0 - cost)
        correct = frame["direction_correct"].to_numpy().astype(float)
        net = quantity * (correct - cost - fee - reserve)
        stressed = net - quantity * stress
        result[str(quantity)] = {
            "supported_trades": int(valid.sum()),
            "net_pnl": float(net[valid].sum()) if valid.any() else 0.0,
            "stressed_pnl": float(stressed[valid].sum()) if valid.any() else 0.0,
            "stressed_expectancy": float(stressed[valid].mean()) if valid.any() else None,
        }
    return result


def paired_comparison(
    schedule: pl.DataFrame,
    candidate: pl.DataFrame,
    control: pl.DataFrame,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    markets = schedule.select("market_id", "window_start").unique(subset=["market_id"])
    candidate_pnl = candidate.select(
        "market_id", pl.col("stressed_pnl").alias("candidate_pnl")
    )
    control_pnl = control.select("market_id", pl.col("stressed_pnl").alias("control_pnl"))
    paired = markets.join(candidate_pnl, on="market_id", how="left").join(
        control_pnl, on="market_id", how="left"
    ).with_columns(
        pl.col("candidate_pnl").fill_null(0.0), pl.col("control_pnl").fill_null(0.0)
    ).with_columns(
        (pl.col("candidate_pnl") - pl.col("control_pnl")).alias("stressed_pnl_difference")
    )
    interval = hierarchical_bootstrap(paired, "stressed_pnl_difference", resamples, seed)
    return {
        "scheduled_markets": paired.height,
        "candidate_stressed_pnl": float(paired["candidate_pnl"].sum()),
        "control_stressed_pnl": float(paired["control_pnl"].sum()),
        "stressed_expectancy_improvement_per_scheduled_market": float(
            paired["stressed_pnl_difference"].mean()
        ),
        "bootstrap_improvement": interval,
    }


def select_development_candidate(
    metrics: dict[str, Any], paired: dict[str, Any], config: TournamentConfig
) -> dict[str, Any]:
    control = metrics[CONTROL]
    qualifications: dict[str, Any] = {}
    qualified: list[str] = []
    for candidate in CANDIDATES:
        row = metrics[candidate]
        checks = development_checks(candidate, row, control, paired[candidate], config)
        qualifications[candidate] = {
            "passed": all(checks.values()),
            "checks": checks,
            "reasons": [name for name, passed in checks.items() if not passed],
        }
        row["qualification"] = qualifications[candidate]
        if candidate != CONTROL and all(checks.values()):
            qualified.append(candidate)
    pool = qualified or [GLOBAL, DIRECTION_PRICE, DIRECTION_TIME_PRICE]
    selected = max(pool, key=lambda name: selection_key(metrics[name]))
    return {
        "status": "qualified_candidate_selected" if qualified else "diagnostic_unqualified_candidate_selected",
        "selected_candidate": selected,
        "qualified_candidates": qualified,
        "selection_pool": pool,
        "qualifications": qualifications,
        "selection_order": [
            "highest lower confidence bound for stressed expectancy",
            "highest trade count",
            "earliest average entry",
            "lowest maximum drawdown",
        ],
        "test_read": False,
    }


def development_checks(
    candidate: str,
    row: dict[str, Any],
    control: dict[str, Any],
    paired: dict[str, Any],
    config: TournamentConfig,
) -> dict[str, bool]:
    gates = config.raw["qualification"]
    directions = row.get("by_direction", {})
    direction_checks = []
    for name in ("UP", "DOWN"):
        direction = directions.get(name, {})
        lower = direction.get("bootstrap_stressed_expectancy", {}).get("lower")
        direction_checks.append(
            (direction.get("accuracy") or 0.0) >= gates["minimum_accuracy"]
            and (direction.get("stressed_expectancy") or -math.inf) > 0.0
            and lower is not None
            and lower >= 0.0
        )
    return {
        "conformal_candidate": candidate != CONTROL,
        "empirical_conformal_coverage": (
            row.get("conformal", {}).get("empirical_conformal_coverage") or 0.0
        )
        >= gates["minimum_conformal_coverage"],
        "overall_accuracy": (row.get("accuracy") or 0.0) >= gates["minimum_accuracy"],
        "up_accuracy": (row.get("up_accuracy") or 0.0) >= gates["minimum_accuracy"],
        "down_accuracy": (row.get("down_accuracy") or 0.0) >= gates["minimum_accuracy"],
        "average_loss_recovery": row.get("wins_to_recover_average_loss", math.inf) <= 3.0,
        "every_quote_recovery": row.get("all_quotes_pass_recovery", False),
        "positive_stressed_pnl": row.get("stressed_pnl", 0.0) > 0.0,
        "positive_stressed_expectancy": (row.get("stressed_expectancy") or -math.inf) > 0.0,
        "profit_factor": (row.get("profit_factor") or 0.0) >= gates["minimum_profit_factor"],
        "positive_bootstrap_lower": _interval_lower(row) > 0.0,
        "positive_each_two_day_fold": bool(row.get("two_day_folds"))
        and all(value["stressed_pnl"] > 0.0 for value in row["two_day_folds"].values()),
        "average_entry": row.get("mean_entry_second", math.inf) <= 90.0,
        "coverage": row.get("coverage", 0.0) >= gates["minimum_coverage"],
        "trade_support": row.get("trades", 0) >= gates["minimum_trades"],
        "both_directions_represented": min(
            row.get("up_trade_share", 0.0), row.get("down_trade_share", 0.0)
        )
        >= gates["minimum_direction_share"],
        "direction_economics": all(direction_checks),
        "pnl_concentration": row.get("maximum_positive_pnl_day_share", math.inf)
        <= gates["maximum_positive_pnl_day_share"],
        "positive_five_share_capacity": row.get("capacity", {}).get("5", {}).get(
            "stressed_pnl", 0.0
        )
        > 0.0,
        "maximum_drawdown_not_worse_than_control": row.get("maximum_drawdown", math.inf)
        <= control.get("maximum_drawdown", math.inf),
        "cvar_not_worse_than_control": _cvar_value(row) >= _cvar_value(control),
        "no_material_band_failure": not material_band_failure(row, config),
        "paired_improvement_lower_positive": paired["bootstrap_improvement"].get("lower")
        is not None
        and paired["bootstrap_improvement"]["lower"] > 0.0,
    }


def apply_test_qualification(
    row: dict[str, Any], development: dict[str, Any], config: TournamentConfig
) -> dict[str, Any]:
    gates = config.raw["qualification"]
    directions = row.get("by_direction", {})
    direction_valid = []
    for name in ("UP", "DOWN"):
        direction = directions.get(name, {})
        lower = direction.get("bootstrap_stressed_expectancy", {}).get("lower")
        direction_valid.append(
            (direction.get("stressed_expectancy") or -math.inf) > 0.0
            and lower is not None
            and lower >= 0.0
        )
    checks = {
        "coverage": row.get("coverage", 0.0) >= gates["minimum_coverage"],
        "trade_support": row.get("trades", 0) >= gates["minimum_trades"],
        "overall_accuracy": (row.get("accuracy") or 0.0) >= gates["minimum_accuracy"],
        "up_accuracy": (row.get("up_accuracy") or 0.0) >= gates["minimum_accuracy"],
        "down_accuracy": (row.get("down_accuracy") or 0.0) >= gates["minimum_accuracy"],
        "positive_stressed_pnl": row.get("stressed_pnl", 0.0) > 0.0,
        "positive_stressed_expectancy": (row.get("stressed_expectancy") or -math.inf) > 0.0,
        "profit_factor": (row.get("profit_factor") or 0.0) >= gates["minimum_profit_factor"],
        "positive_bootstrap_lower": _interval_lower(row) > 0.0,
        "average_loss_recovery": row.get("wins_to_recover_average_loss", math.inf) <= 3.0,
        "every_quote_recovery": row.get("all_quotes_pass_recovery", False),
        "average_entry": row.get("mean_entry_second", math.inf) <= 90.0,
        "positive_each_two_day_fold": bool(row.get("two_day_folds"))
        and all(value["stressed_pnl"] > 0.0 for value in row["two_day_folds"].values()),
        "both_directions_represented": min(
            row.get("up_trade_share", 0.0), row.get("down_trade_share", 0.0)
        )
        >= gates["minimum_direction_share"],
        "direction_economics": all(direction_valid),
        "pnl_concentration": row.get("maximum_positive_pnl_day_share", math.inf)
        <= gates["maximum_positive_pnl_day_share"],
        "positive_five_share_capacity": row.get("capacity", {}).get("5", {}).get(
            "stressed_pnl", 0.0
        )
        > 0.0,
        "empirical_conformal_coverage": (
            row.get("conformal", {}).get("empirical_conformal_coverage") or 0.0
        )
        >= gates["minimum_conformal_coverage"],
        "drawdown_not_regressed": row.get("maximum_drawdown", math.inf)
        <= development.get("maximum_drawdown", math.inf),
        "cvar_not_regressed": _cvar_value(row) >= _cvar_value(development),
    }
    return {
        "status": "qualified_on_untouched_test" if all(checks.values()) else "unqualified_on_untouched_test",
        "passed": all(checks.values()),
        "checks": checks,
        "reasons": [name for name, passed in checks.items() if not passed],
    }


def definitive_conclusion(
    selection: dict[str, Any],
    development: dict[str, Any],
    paired: dict[str, Any],
    test: dict[str, Any],
    test_qualification: dict[str, Any],
) -> dict[str, Any]:
    selected = selection["selected_candidate"]
    if test_qualification["passed"]:
        return {
            "code": 8,
            "outcome": "candidate_qualifies_on_untouched_test_separate_deployment_decision_required",
            "candidate": selected,
        }
    selected_dev = development[selected]
    if selected_dev.get("trades", 0) == 0:
        return {
            "code": 6,
            "outcome": "upstream_predictor_lacks_sufficient_resolution_for_safe_admission",
            "candidate": selected,
        }
    improvement = paired[selected]["bootstrap_improvement"].get("median")
    economics_positive = (selected_dev.get("stressed_pnl") or 0.0) > 0.0
    support_failed = selected_dev.get("trades", 0) < 50 or selected_dev.get("coverage", 0.0) < 0.03
    accuracy_improved = (selected_dev.get("accuracy") or 0.0) > (
        development[CONTROL].get("accuracy") or 0.0
    )
    if economics_positive and support_failed and (improvement or -math.inf) > 0.0:
        return {
            "code": 5,
            "outcome": "conformal_calibration_improves_economics_but_has_insufficient_trade_support",
            "candidate": selected,
        }
    if accuracy_improved and not economics_positive:
        return {
            "code": 4,
            "outcome": "conformal_calibration_improves_accuracy_but_not_economics",
            "candidate": selected,
        }
    return {"code": 7, "outcome": "no_conformal_admission_candidate_qualifies", "candidate": selected}


def selection_key(row: dict[str, Any]) -> tuple[float, int, float, float]:
    return (
        _interval_lower(row),
        int(row.get("trades", 0)),
        -float(row.get("mean_entry_second") or math.inf),
        -float(row.get("maximum_drawdown") or math.inf),
    )


def material_band_failure(row: dict[str, Any], config: TournamentConfig) -> bool:
    minimum_trades = int(config.raw["qualification"]["material_band_minimum_trades"])
    minimum_accuracy = float(config.raw["qualification"]["material_band_minimum_accuracy"])
    for section in ("entry_time_bands", "executable_price_bands"):
        for value in row.get(section, {}).values():
            if value.get("trades", 0) >= minimum_trades and (
                (value.get("accuracy") or 0.0) < minimum_accuracy
                or (value.get("stressed_expectancy") or -math.inf) <= 0.0
            ):
                return True
    return False


def hierarchical_bootstrap(
    frame: pl.DataFrame, value_column: str, resamples: int, seed: int
) -> dict[str, Any]:
    if frame.is_empty() or resamples <= 0:
        return empty_interval(resamples)
    working = frame.with_columns(pl.col("window_start").dt.date().alias("bootstrap_day"))
    day_values = [
        group[value_column].to_numpy().astype(float)
        for group in working.partition_by("bootstrap_day", maintain_order=True)
    ]
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=float)
    for index in range(resamples):
        selected_days = rng.integers(0, len(day_values), len(day_values))
        values = []
        for selected in selected_days:
            block = day_values[selected]
            values.append(block[rng.integers(0, len(block), len(block))])
        samples[index] = float(np.concatenate(values).mean())
    return {
        "lower": float(np.quantile(samples, 0.025)),
        "median": float(np.quantile(samples, 0.50)),
        "upper": float(np.quantile(samples, 0.975)),
        "resamples": resamples,
        "unit": "market_day_block_stressed_pnl_per_observation",
    }


def empty_interval(resamples: int) -> dict[str, Any]:
    return {"lower": None, "median": None, "upper": None, "resamples": resamples}


def ece(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> float:
    total = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper >= 1.0 else probabilities < upper
        )
        if not mask.any():
            continue
        weight = float(weights[mask].sum())
        total += weight * abs(
            float(np.average(labels[mask], weights=weights[mask]))
            - float(np.average(probabilities[mask], weights=weights[mask]))
        )
    return total


def cvar(values: np.ndarray, fraction: float = 0.10) -> float:
    count = max(1, math.ceil(len(values) * fraction))
    return float(np.sort(values)[:count].mean())


def render_report(metrics: dict[str, Any]) -> str:
    development = metrics["development"]
    test = metrics["untouched_test"]
    lines = [
        "# BTC 5m TWAP Conformal-Risk Admission Tournament",
        "",
        f"Run: `{metrics['run_id']}`  ",
        f"Source commit: `{metrics['source_commit']}`  ",
        f"Selected artifact: `{metrics['selected_artifact']['sha256']}`  ",
        f"Conclusion: **{metrics['conclusion']['outcome']}**  ",
        "Deployment: **not deployed; training only**",
        "",
        "## Development candidate comparison",
        "",
        "| Candidate | Trades | Coverage | Accuracy | UP | DOWN | Stressed PnL | Expectancy | PF | Bootstrap lower | Avg entry | Qualified |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in development["candidates"].items():
        lines.append(
            f"| {name} | {row.get('trades', 0)} | {_pct(row.get('coverage'))} | "
            f"{_pct(row.get('accuracy'))} | {_pct(row.get('up_accuracy'))} | "
            f"{_pct(row.get('down_accuracy'))} | {_fmt(row.get('stressed_pnl'))} | "
            f"{_fmt(row.get('stressed_expectancy'))} | {_fmt(row.get('profit_factor'))} | "
            f"{_fmt(row.get('bootstrap_stressed_expectancy', {}).get('lower'))} | "
            f"{_fmt(row.get('mean_entry_second'))} | "
            f"{row.get('qualification', {}).get('passed', False)} |"
        )
    selected = test["candidate"]
    row = test["metrics"]
    lines.extend(
        [
            "",
            "## Untouched test",
            "",
            f"Candidate: `{selected}`  ",
            f"Qualification: **{test['qualification']['status']}**  ",
            f"Trades/coverage: {row.get('trades', 0)} / {_pct(row.get('coverage'))}  ",
            f"Accuracy overall/UP/DOWN: {_pct(row.get('accuracy'))} / {_pct(row.get('up_accuracy'))} / {_pct(row.get('down_accuracy'))}  ",
            f"Gross/net/stressed PnL: {_fmt(row.get('gross_pnl'))} / {_fmt(row.get('net_pnl'))} / {_fmt(row.get('stressed_pnl'))}  ",
            f"Expectancy/profit factor: {_fmt(row.get('stressed_expectancy'))} / {_fmt(row.get('profit_factor'))}  ",
            f"Bootstrap 95% interval: `{json.dumps(row.get('bootstrap_stressed_expectancy', {}), sort_keys=True)}`  ",
            f"Failed gates: {', '.join(test['qualification']['reasons']) or 'none'}",
            "",
            "## Full metric inventory",
            "",
            "`metrics.json` contains every candidate's predictive scores, conformal coverage, risk-bound calibration, interval widths, loss distribution, economics, bootstrap intervals, drawdown, CVaR, entries, day/fold/direction/band breakdowns, calibration-cell support, fallback counts, abstention reasons, 5–200 share capacity, concentration, and paired control comparison.",
            "",
            "## Integrity",
            "",
            "- Calibration, development, and untouched test market IDs are disjoint and chronological.",
            "- The untouched test was loaded only after candidate selection and artifact freezing.",
            "- The upstream predictor was not refit; raw RefPrice was excluded from admission.",
            "- No database, data source, ingester, table, trading process, or runtime service was changed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _assert_decision_columns_equal(left: pl.DataFrame, right: pl.DataFrame) -> None:
    columns = (
        "correctness_lower_bound",
        "error_risk_upper_bound",
        "conformal_margin_lower",
        "conformal_margin_upper",
        "conformal_fallback",
    )
    if not left.select(columns).equals(right.select(columns)):
        raise RuntimeError("serialization/reload decisions differ")


def _period(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    return frame.filter(pl.col("window_start").is_between(start, end, closed="left"))


def _role_ledger_hash(manifest: dict[str, Any], start: datetime, end: datetime) -> str:
    selected = [
        row
        for row in manifest["days"]
        if start.date() <= date.fromisoformat(row["day"]) < end.date()
    ]
    return _hash_payload(selected)


def _interval_lower(row: dict[str, Any]) -> float:
    value = row.get("bootstrap_stressed_expectancy", {}).get("lower")
    return float(value) if value is not None else -math.inf


def _cvar_value(row: dict[str, Any]) -> float:
    value = row.get("cvar_10")
    return float(value) if value is not None else -math.inf


def _frame_identity(frame: pl.DataFrame) -> str:
    return _hash_payload(
        {
            "rows": frame.height,
            "markets": frame["market_id"].n_unique(),
            "first": frame["window_start"].min(),
            "last": frame["window_start"].max(),
            "market_ids": sorted(frame["market_id"].unique().to_list()),
        }
    )


def _checkpoint_json(path: Path, input_hash: str, value: dict[str, Any]) -> None:
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved["input_hash"] != input_hash or saved["value"] != _json_roundtrip(value):
            raise RuntimeError(f"checkpoint input or value changed: {path}")
        print(f"checkpoint resume: {path.stem}", flush=True)
        return
    _write_json(path, {"input_hash": input_hash, "value": value})
    print(f"checkpoint complete: {path.stem}", flush=True)


def _json_roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default, sort_keys=True))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _write_parquet_atomic(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date, Path)):
        return value.isoformat() if not isinstance(value, Path) else str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(type(value).__name__)


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf"
    return f"{float(value):.4f}"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path)
    args = parser.parse_args()
    result, metrics = run_tournament(load_config(args.config, args.upstream_root))
    print(
        json.dumps(
            {
                "result": str(result),
                "artifact_sha256": metrics["selected_artifact"]["sha256"],
                "selected_candidate": metrics["development"]["selection"]["selected_candidate"],
                "test_status": metrics["untouched_test"]["qualification"]["status"],
                "conclusion": metrics["conclusion"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
