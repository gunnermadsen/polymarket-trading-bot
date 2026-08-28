"""Frozen latent-TWAP settlement-risk tournament.

This runner reads only the existing immutable TWAP/refprice/orderbook extract.
It creates local training checkpoints and committed model/report artifacts; it
does not issue SQL, mutate the database, export a runtime model, or alter a
trading process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import scipy
import sklearn

from .latent_twap_state_space import (
    CANDIDATES,
    CandidateSpec,
    ProbabilityCalibrator,
    SearchConfiguration,
    StateSpaceParameters,
    filter_sequence,
    fit_chronological_mle,
    fit_probability_calibrator,
    predetermined_configurations,
)
from .twap60_training_data import canonical_refprice_path

SCHEMA_VERSION = "btc-latent-twap-settlement-risk-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-latent-twap-settlement-risk-artifact-v1"
CHECKPOINT_SECONDS = tuple(range(30, 151, 5))
SENSOR_COLUMNS = (
    "twap30_margin_bps",
    "twap60_margin_bps",
    "refprice_margin_bps",
)
VWAP_QUANTITIES = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200)


@dataclass(frozen=True)
class Fold:
    name: str
    test_start: datetime
    test_end: datetime


@dataclass(frozen=True)
class TournamentConfig:
    source_path: Path
    package_root: Path
    raw: dict[str, Any]
    profile: str
    model_family: str
    random_seed: int
    freeze_at: datetime
    historical_start: datetime
    historical_end: datetime
    calibration_start: datetime
    calibration_end: datetime
    development_start: datetime
    development_end: datetime
    prospective_start: datetime
    folds: tuple[Fold, ...]
    source_cache: Path
    source_frame: Path
    source_manifest: Path
    label_audit: Path
    comparator_metrics: Path
    runs: Path
    committed_results: Path


def load_config(path: Path) -> TournamentConfig:
    source_path = path.resolve()
    package_root = source_path.parents[1]
    raw = tomllib.loads(source_path.read_text())
    periods = raw["periods"]
    paths = raw["paths"]
    config = TournamentConfig(
        source_path=source_path,
        package_root=package_root,
        raw=raw,
        profile=str(raw["training"]["profile"]),
        model_family=str(raw["training"]["model_family"]),
        random_seed=int(raw["training"]["random_seed"]),
        freeze_at=_utc(raw["training"]["freeze_at"]),
        historical_start=_utc(periods["historical_fit_start"]),
        historical_end=_utc(periods["historical_fit_end"]),
        calibration_start=_utc(periods["calibration_start"]),
        calibration_end=_utc(periods["calibration_end"]),
        development_start=_utc(periods["development_start"]),
        development_end=_utc(periods["development_end"]),
        prospective_start=_utc(periods["prospective_start"]),
        folds=tuple(
            Fold(str(row["name"]), _utc(row["test_start"]), _utc(row["test_end"]))
            for row in raw["folds"]
        ),
        source_cache=_path(package_root, paths["source_cache"]),
        source_frame=_path(package_root, paths["source_frame"]),
        source_manifest=_path(package_root, paths["source_manifest"]),
        label_audit=_path(package_root, paths["label_audit"]),
        comparator_metrics=_path(package_root, paths["corrected_comparator_metrics"]),
        runs=_path(package_root, paths["runs"]),
        committed_results=_path(package_root, paths["committed_results"]),
    )
    _validate_config(config)
    return config


def _validate_config(config: TournamentConfig) -> None:
    training = config.raw["training"]
    entry = config.raw["entry"]
    execution = config.raw["execution"]
    search = config.raw["search"]
    expected_periods = (
        config.historical_start,
        config.historical_end,
        config.calibration_start,
        config.calibration_end,
        config.development_start,
        config.development_end,
        config.prospective_start,
        config.freeze_at,
    )
    if config.profile != "btc_5m_latent_twap_settlement_risk":
        raise ValueError("unexpected latent-TWAP profile")
    if config.model_family != "btc-5m-latent-twap-settlement-risk":
        raise ValueError("unexpected model family")
    if not (
        training["paper_only"]
        and training["strictly_training_only"]
        and not training["live_capital_allowed"]
    ):
        raise ValueError("latent-TWAP execution must remain training-only and paper-only")
    if expected_periods != (
        datetime(2026, 6, 7, tzinfo=UTC),
        datetime(2026, 8, 1, tzinfo=UTC),
        datetime(2026, 8, 1, tzinfo=UTC),
        datetime(2026, 8, 14, tzinfo=UTC),
        datetime(2026, 8, 14, tzinfo=UTC),
        datetime(2026, 8, 28, tzinfo=UTC),
        datetime(2026, 8, 28, tzinfo=UTC),
        datetime(2026, 8, 28, tzinfo=UTC),
    ):
        raise ValueError("frozen data periods changed")
    if (
        int(entry["start_second"]),
        int(entry["end_second"]),
        int(entry["cadence_seconds"]),
    ) != (30, 150, 5):
        raise ValueError("shared entry checkpoint contract changed")
    if float(entry["minimum_calibrated_correctness"]) != 0.90:
        raise ValueError("correctness admission threshold changed")
    if float(entry["maximum_wins_per_loss"]) != 3.0:
        raise ValueError("loss-recovery gate changed")
    if tuple(int(value) for value in execution["quantities"]) != VWAP_QUANTITIES:
        raise ValueError("VWAP capacity quantities changed")
    if int(search["configurations_per_candidate"]) != 12:
        raise ValueError("search budget must remain exactly twelve rows per candidate")
    if len(predetermined_configurations()) != 12 or len(config.folds) != 7:
        raise ValueError("frozen search or fold roster changed")
    previous = config.development_start
    for fold in config.folds:
        if fold.test_start != previous:
            raise ValueError("development folds are not contiguous")
        if (fold.test_end - fold.test_start).total_seconds() != 2 * 86_400:
            raise ValueError("development folds must be two UTC days")
        previous = fold.test_end
    if previous != config.development_end:
        raise ValueError("development folds do not end at the freeze boundary")
    hashes = (
        (config.source_frame, config.raw["paths"]["source_frame_sha256"]),
        (config.source_manifest, config.raw["paths"]["source_manifest_sha256"]),
        (config.label_audit, config.raw["paths"]["label_audit_sha256"]),
    )
    for file_path, expected_hash in hashes:
        if not file_path.is_file() or file_sha256(file_path) != expected_hash:
            raise RuntimeError(f"immutable training input changed or is missing: {file_path}")
    if not config.comparator_metrics.is_file():
        raise FileNotFoundError(config.comparator_metrics)


class CheckpointStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get_or_compute(
        self,
        key: str,
        input_hash: str,
        compute: Callable[[], Any],
    ) -> Any:
        path = self.root / f"{key}.joblib"
        digest_path = self.root / f"{key}.sha256"
        if path.is_file() and digest_path.is_file():
            if file_sha256(path) != digest_path.read_text().strip():
                raise RuntimeError(f"checkpoint hash mismatch: {key}")
            payload = joblib.load(path)
            if payload.get("input_hash") != input_hash:
                raise RuntimeError(f"checkpoint input identity changed: {key}")
            print(f"checkpoint resume: {key}", flush=True)
            return payload["value"]
        value = compute()
        temporary = path.with_suffix(".joblib.partial")
        joblib.dump({"input_hash": input_hash, "value": value}, temporary, compress=3)
        os.replace(temporary, path)
        digest_path.write_text(file_sha256(path) + "\n")
        print(f"checkpoint complete: {key}", flush=True)
        return value


def run_tournament(config: TournamentConfig) -> tuple[Path, dict[str, Any]]:
    source_identity = hashlib.sha256(
        (
            file_sha256(config.source_path)
            + file_sha256(config.source_manifest)
            + file_sha256(config.source_frame)
            + file_sha256(config.label_audit)
            + _git_revision(config.package_root)
        ).encode()
    ).hexdigest()
    workspace = config.runs / source_identity[:20]
    completion = workspace / "completion.json"
    if completion.is_file():
        record = json.loads(completion.read_text())
        result = config.package_root / record["result"]
        metrics = json.loads((result / "metrics.json").read_text())
        return result, metrics
    workspace.mkdir(parents=True, exist_ok=True)
    store = CheckpointStore(workspace / "checkpoints")
    frame_path = workspace / "training-frame.parquet"
    frame_manifest_path = workspace / "training-frame-manifest.json"
    if frame_path.is_file() and frame_manifest_path.is_file():
        frame_manifest = json.loads(frame_manifest_path.read_text())
        if file_sha256(frame_path) != frame_manifest["sha256"]:
            raise RuntimeError("training-frame checkpoint hash mismatch")
        frame = pl.read_parquet(frame_path)
        print("checkpoint resume: causal training frame", flush=True)
    else:
        frame, frame_manifest = build_training_frame(config)
        temporary = frame_path.with_suffix(".parquet.partial")
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        os.replace(temporary, frame_path)
        frame_manifest["sha256"] = file_sha256(frame_path)
        _write_json(frame_manifest_path, frame_manifest)
    split_audit = verify_split_separation(frame, config)
    search_results: dict[str, Any] = {}
    selected: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        candidate_rows: list[dict[str, Any]] = []
        for search_config in predetermined_configurations():
            key = f"search-{candidate.name}-{search_config.identifier}"
            value = store.get_or_compute(
                key,
                _hash_payload(
                    {
                        "source": frame_manifest["sha256"],
                        "candidate": asdict(candidate),
                        "configuration": asdict(search_config),
                        "historical_end": config.historical_end,
                        "calibration_end": config.calibration_end,
                    }
                ),
                lambda candidate=candidate, search_config=search_config: _fit_and_calibrate(
                    frame, config, candidate, search_config
                ),
            )
            candidate_rows.append(value)
        winner = min(
            candidate_rows,
            key=lambda row: (
                row["calibration_metrics"]["brier_score"],
                row["calibration_metrics"]["log_loss"],
                row["calibration_metrics"]["margin_mae_bps"],
                row["configuration"]["identifier"],
            ),
        )
        search_results[candidate.name] = {
            "budget": len(candidate_rows),
            "rows": [_search_report_row(row) for row in candidate_rows],
            "selected_configuration": winner["configuration"]["identifier"],
        }
        selected[candidate.name] = winner

    candidate_results: dict[str, Any] = {}
    ledgers: dict[str, pl.DataFrame] = {}
    scored_frames: dict[str, pl.DataFrame] = {}
    fold_reports: dict[str, list[dict[str, Any]]] = {}
    for candidate in CANDIDATES:
        calibrator = ProbabilityCalibrator.from_dict(selected[candidate.name]["calibrator"])
        search_config = SearchConfiguration(**selected[candidate.name]["configuration"])
        candidate_ledgers: list[pl.DataFrame] = []
        candidate_scored: list[pl.DataFrame] = []
        candidate_folds: list[dict[str, Any]] = []
        for fold in config.folds:
            key = f"fold-{candidate.name}-{fold.name}"
            fold_value = store.get_or_compute(
                key,
                _hash_payload(
                    {
                        "source": frame_manifest["sha256"],
                        "candidate": asdict(candidate),
                        "configuration": asdict(search_config),
                        "calibrator": calibrator.to_dict(),
                        "fold": asdict(fold),
                    }
                ),
                lambda candidate=candidate, search_config=search_config, calibrator=calibrator, fold=fold: _run_fold(
                    frame, config, candidate, search_config, calibrator, fold
                ),
            )
            scored = pl.DataFrame(fold_value["scored"])
            ledger = pl.DataFrame(fold_value["ledger"])
            if not scored.is_empty():
                candidate_scored.append(scored)
            if not ledger.is_empty():
                candidate_ledgers.append(ledger)
            candidate_folds.append(fold_value["report"])
        scored_all = pl.concat(candidate_scored, how="diagonal_relaxed") if candidate_scored else pl.DataFrame()
        ledger_all = pl.concat(candidate_ledgers, how="diagonal_relaxed") if candidate_ledgers else pl.DataFrame()
        scheduled = frame.filter(
            pl.col("window_start").is_between(
                config.development_start, config.development_end, closed="left"
            )
        )["market_id"].n_unique()
        result = trading_metrics(
            ledger_all,
            scored_all,
            scheduled_markets=scheduled,
            config=config,
            fold_reports=candidate_folds,
        )
        result["selected_configuration"] = search_config.identifier
        result["calibration_period"] = selected[candidate.name]["calibration_metrics"]
        candidate_results[candidate.name] = result
        ledgers[candidate.name] = ledger_all
        scored_frames[candidate.name] = scored_all
        fold_reports[candidate.name] = candidate_folds

    paired_tests = refprice_contribution_tests(
        candidate_results, ledgers, scored_frames, config
    )
    comparator = _load_corrected_comparator(config.comparator_metrics)
    development_evidence = _development_evidence(frame, config)
    selection = apply_development_qualification(
        candidate_results, paired_tests, comparator, development_evidence, config
    )
    final_models = _fit_final_models(frame, config, selected, selection, store, frame_manifest)
    prospective = prospective_status(frame, selection, config)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_family": config.model_family,
        "freeze_at": config.freeze_at.isoformat(),
        "source_commit": _git_revision(config.package_root),
        "source_identity": frame_manifest["sha256"],
        "selection": selection,
        "prospective": prospective,
        "entry_contract": {
            "seconds": list(CHECKPOINT_SECONDS),
            "minimum_calibrated_correctness": 0.90,
            "positive_lower_stressed_payoff_required": True,
            "maximum_wins_per_loss": 3.0,
            "symmetric_directions": True,
            "maximum_one_trade_per_market": 1,
            "execution_evidence": "VWAP-5 executable ask",
        },
        "models": final_models,
        "deployment_status": "not_deployed",
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    temporary = config.committed_results / f"{run_id}.partial"
    final = config.committed_results / run_id
    temporary.mkdir(parents=True, exist_ok=False)
    artifact_path = temporary / "model-family.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    artifact_sha = file_sha256(artifact_path)
    ledger_root = temporary / "development-ledgers"
    ledger_root.mkdir()
    for name, ledger in ledgers.items():
        ledger.write_parquet(ledger_root / f"{name}.parquet", compression="zstd", statistics=True)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": config.profile,
        "model_family": config.model_family,
        "paper_only": True,
        "strictly_training_only": True,
        "database_mutations": False,
        "new_data_sources": False,
        "new_ingesters": False,
        "new_tables": False,
        "migrations": False,
        "runtime_exported": False,
        "trading_processes_changed": False,
        "source_commit": _git_revision(config.package_root),
        "configuration": {
            "path": str(config.source_path.relative_to(config.package_root)),
            "sha256": file_sha256(config.source_path),
            "freeze_at": config.freeze_at.isoformat(),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "data": {
            "immutable_source_manifest_sha256": file_sha256(config.source_manifest),
            "immutable_source_frame_sha256": file_sha256(config.source_frame),
            "immutable_label_audit_sha256": file_sha256(config.label_audit),
            "training_frame": frame_manifest,
            "split_separation": split_audit,
        },
        "search": search_results,
        "development": {
            "folds": fold_reports,
            "candidates": candidate_results,
            "refprice_contribution": paired_tests,
            "corrected_twap_comparator": comparator,
            "official_evidence_completeness": development_evidence,
            "selection": selection,
        },
        "prospective": prospective,
        "evidence_separation": evidence_separation_metrics(frame, selected, config),
        "model_artifact": {
            "path": "model-family.joblib",
            "sha256": artifact_sha,
            "training_run_id": run_id,
            "qualification_status": selection["status"],
            "deployment_status": "not_deployed",
        },
        "limitations": _limitations(frame, prospective, selection, config),
    }
    _write_json(temporary / "metrics.json", metrics)
    (temporary / "report.md").write_text(render_report(metrics))
    (temporary / "model-family.sha256").write_text(artifact_sha + "\n")
    _write_json(
        temporary / "model-provenance.json",
        {
            "schema_version": "btc-model-provenance-v1",
            "model_artifact_sha256": artifact_sha,
            "artifact_path": "model-family.joblib",
            "producing_commit": _git_revision(config.package_root),
            "training_run_id": run_id,
            "source_identity": frame_manifest["sha256"],
            "source_manifest_sha256": file_sha256(config.source_manifest),
            "configuration_sha256": file_sha256(config.source_path),
            "qualification_status": selection["status"],
            "prospective_status": prospective["status"],
            "deployment_status": "not_deployed",
        },
    )
    config.committed_results.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    _write_json(
        completion,
        {"result": str(final.relative_to(config.package_root)), "artifact_sha256": artifact_sha},
    )
    return final, metrics


def build_training_frame(config: TournamentConfig) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Build causal 30..150 second sensors from the existing source cache."""

    labels = (
        pl.read_parquet(config.label_audit)
        .filter(
            pl.col("window_start").is_between(
                config.historical_start, config.development_end, closed="left"
            )
            & (
                (
                    (pl.col("window_start") < config.historical_end)
                    & (pl.col("label_source") == "chainlink_reconstructed_twap60")
                )
                | (
                    pl.col("window_start").is_between(
                        config.calibration_start, config.calibration_end, closed="left"
                    )
                    & (pl.col("label_source") == "authentic_counterfactual_twap60")
                )
                | (
                    pl.col("window_start").is_between(
                        config.development_start, config.development_end, closed="left"
                    )
                    & (pl.col("label_source") == "authentic_official_twap60")
                )
            )
            & pl.col("target_margin_bps").is_not_null()
            & pl.col("target_margin_bps").is_finite()
        )
        .select(
            "market_id",
            "window_start",
            "window_end",
            "label_source",
            "target_margin_bps",
            "label_up",
            "official_outcome",
        )
        .unique(subset=["market_id"], keep="last")
        .sort(["window_start", "market_id"])
    )
    if labels.is_empty():
        raise RuntimeError("immutable label audit has no eligible markets")
    repeated = labels.select(
        pl.all().repeat_by(pl.lit(len(CHECKPOINT_SECONDS))).explode()
    ).with_columns(
        pl.Series("seconds_elapsed", np.tile(CHECKPOINT_SECONDS, labels.height))
    ).with_columns(
        (pl.col("window_start") + pl.duration(seconds=pl.col("seconds_elapsed")))
        .alias("observed_at")
    )
    refprice = _load_source_partitions(
        config.source_cache,
        "refprice",
        (
            "source_timestamp",
            "valid_from_timestamp",
            "provider_available_at",
            "received_at",
            "price",
            "bid",
            "ask",
            "archive_row_number",
            "artifact_id",
            "report_sha256",
        ),
    ).filter(
        pl.col("source_timestamp").is_between(
            config.historical_start - timedelta(minutes=2),
            config.development_end,
            closed="left",
        )
    )
    path = (
        canonical_refprice_path(refprice)
        .sort(["provider_available_at", "source_timestamp", "archive_row_number"])
        .filter(pl.col("source_timestamp") == pl.col("source_timestamp").cum_max())
        .unique(subset=["source_timestamp"], keep="last")
        .sort("source_timestamp")
    )
    if path.height < 2:
        raise RuntimeError("causal refprice path is too short")
    source_us = path["source_timestamp"].to_numpy().astype("datetime64[us]").astype(np.int64)
    available_us = (
        path["provider_available_at"].to_numpy().astype("datetime64[us]").astype(np.int64)
    )
    prices = path["price"].to_numpy().astype(float)
    if np.any(np.diff(source_us) <= 0) or np.any(np.diff(available_us) < 0):
        raise RuntimeError("causal refprice path chronology is not monotonic")
    cumulative = np.zeros(len(prices), dtype=float)
    cumulative[1:] = np.cumsum(prices[:-1] * np.diff(source_us))
    decisions = repeated["observed_at"].to_numpy().astype("datetime64[us]").astype(np.int64)
    openings = repeated["window_start"].to_numpy().astype("datetime64[us]").astype(np.int64)
    available_index = np.searchsorted(available_us, decisions, side="left") - 1

    def integral_at(points: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(source_us, points, side="right") - 1
        indices = np.minimum(indices, available_index)
        safe = np.maximum(indices, 0)
        values = cumulative[safe] + prices[safe] * (points - source_us[safe])
        values[indices < 0] = np.nan
        return values

    open_twap60 = (integral_at(openings) - integral_at(openings - 60_000_000)) / 60_000_000
    current_twap30 = (integral_at(decisions) - integral_at(decisions - 30_000_000)) / 30_000_000
    current_twap60 = (integral_at(decisions) - integral_at(decisions - 60_000_000)) / 60_000_000
    safe_available = np.maximum(available_index, 0)
    current_refprice = prices[safe_available]
    source_age = (decisions - source_us[safe_available]) / 1_000_000.0
    sensor_available_at = available_us[safe_available].astype("datetime64[us]")
    sensor_valid = (
        (available_index >= 0)
        & (available_us[safe_available] < decisions)
        & (source_age > 0.0)
        & (source_age <= 5.0)
        & np.isfinite(open_twap60)
        & np.isfinite(current_twap30)
        & np.isfinite(current_twap60)
        & (open_twap60 > 0)
    )
    repeated = repeated.with_columns(
        pl.Series("twap30_margin_bps", np.log(current_twap30 / open_twap60) * 10_000.0),
        pl.Series("twap60_margin_bps", np.log(current_twap60 / open_twap60) * 10_000.0),
        pl.Series("refprice_margin_bps", np.log(current_refprice / open_twap60) * 10_000.0),
        pl.Series("sensor_source_age_seconds", source_age),
        pl.Series("sensor_max_available_at", sensor_available_at).dt.replace_time_zone("UTC"),
        pl.Series("sensor_valid", sensor_valid),
    )
    execution_columns = [
        "market_id",
        "seconds_elapsed",
        "observed_at",
        "fee_rate",
        "quality_flags",
        "up_provider_received_at",
        "down_provider_received_at",
        *[f"up_ask_vwap_{quantity}" for quantity in VWAP_QUANTITIES],
        *[f"down_ask_vwap_{quantity}" for quantity in VWAP_QUANTITIES],
    ]
    execution = (
        _load_source_partitions(config.source_cache, "execution", tuple(execution_columns) + ("window_start",))
        .filter(
            pl.col("window_start").is_between(
                config.development_start, config.development_end, closed="left"
            )
            & pl.col("seconds_elapsed").is_in(CHECKPOINT_SECONDS)
        )
        .select(execution_columns)
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .unique(subset=["market_id", "seconds_elapsed"], keep="last")
    )
    freshness = float(config.raw["execution"]["freshness_seconds"])
    execution = execution.with_columns(
        (
            ((pl.col("quality_flags") & 63) == 0)
            & pl.col("up_provider_received_at").is_not_null()
            & pl.col("down_provider_received_at").is_not_null()
            & (pl.col("up_provider_received_at") <= pl.col("observed_at"))
            & (pl.col("down_provider_received_at") <= pl.col("observed_at"))
            & (
                pl.col("up_provider_received_at")
                >= pl.col("observed_at") - pl.duration(seconds=freshness)
            )
            & (
                pl.col("down_provider_received_at")
                >= pl.col("observed_at") - pl.duration(seconds=freshness)
            )
        ).alias("book_valid")
    ).drop("observed_at")
    frame = repeated.join(
        execution, on=["market_id", "seconds_elapsed"], how="left", validate="m:1"
    ).with_columns(pl.col("book_valid").fill_null(False))
    vwap_columns = [
        f"{direction}_ask_vwap_{quantity}"
        for direction in ("up", "down")
        for quantity in VWAP_QUANTITIES
    ]
    frame = frame.with_columns(
        *[
            pl.when(pl.col("book_valid")).then(pl.col(column)).otherwise(None).alias(column)
            for column in vwap_columns
        ]
    ).sort(["window_start", "market_id", "seconds_elapsed"])
    expected_rows = labels.height * len(CHECKPOINT_SECONDS)
    if frame.height != expected_rows:
        raise RuntimeError("causal training frame lost scheduled checkpoints")
    causality_failures = frame.filter(
        pl.col("sensor_max_available_at") >= pl.col("observed_at")
    ).height
    if causality_failures:
        raise RuntimeError(f"{causality_failures} sensor rows violate causal availability")
    inventory = {
        str(row["label_source"]): {
            "markets": int(row["markets"]),
            "rows": int(row["rows"]),
        }
        for row in frame.group_by("label_source").agg(
            pl.col("market_id").n_unique().alias("markets"), pl.len().alias("rows")
        ).iter_rows(named=True)
    }
    manifest = {
        "schema_version": "btc-latent-twap-causal-frame-v1",
        "rows": frame.height,
        "markets": labels.height,
        "checkpoints_per_market": len(CHECKPOINT_SECONDS),
        "checkpoint_seconds": list(CHECKPOINT_SECONDS),
        "sensor_valid_rows": int(frame["sensor_valid"].sum()),
        "book_valid_rows": int(frame["book_valid"].sum()),
        "label_inventory": inventory,
        "source_manifest_sha256": file_sha256(config.source_manifest),
        "source_frame_sha256": file_sha256(config.source_frame),
        "label_audit_sha256": file_sha256(config.label_audit),
        "features_use_completed_targets": False,
        "labels_supervision_only": True,
        "read_only_sources": True,
        "database_mutations": False,
    }
    return frame, manifest


def _load_source_partitions(
    cache: Path, name: str, columns: tuple[str, ...]
) -> pl.DataFrame:
    manifest = json.loads((cache / "source-manifest.json").read_text())
    paths = [cache / row["path"] for row in manifest["partitions"][name]]
    frames = [pl.read_parquet(path, columns=list(columns)) for path in paths]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed", rechunk=True)


def verify_split_separation(frame: pl.DataFrame, config: TournamentConfig) -> dict[str, Any]:
    blocks = {
        "historical_fit": _period(frame, config.historical_start, config.historical_end),
        "authentic_calibration": _period(frame, config.calibration_start, config.calibration_end),
        "official_development": _period(frame, config.development_start, config.development_end),
        "prospective": frame.filter(pl.col("window_start") >= config.prospective_start),
    }
    market_sets = {name: set(block["market_id"].unique()) for name, block in blocks.items()}
    overlaps = {
        f"{left}__{right}": len(market_sets[left] & market_sets[right])
        for index, left in enumerate(blocks)
        for right in tuple(blocks)[index + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError("market identity overlaps fitting, calibration, or evaluation")
    fold_overlaps: dict[str, int] = {}
    for fold in config.folds:
        training = set(frame.filter(pl.col("window_start") < fold.test_start)["market_id"].unique())
        testing = set(_period(frame, fold.test_start, fold.test_end)["market_id"].unique())
        fold_overlaps[fold.name] = len(training & testing)
    if any(fold_overlaps.values()):
        raise RuntimeError("rolling fold contains same-market train/test overlap")
    return {
        "passed": True,
        "block_markets": {name: len(values) for name, values in market_sets.items()},
        "block_overlaps": overlaps,
        "fold_train_test_overlaps": fold_overlaps,
        "completed_market_values_supervision_only": True,
        "features_at_or_before_checkpoint": True,
    }


def _fit_and_calibrate(
    frame: pl.DataFrame,
    config: TournamentConfig,
    candidate: CandidateSpec,
    search_config: SearchConfiguration,
) -> dict[str, Any]:
    historical = _period(frame, config.historical_start, config.historical_end)
    calibration = _period(frame, config.calibration_start, config.calibration_end)
    parameters = _fit_parameters(historical, candidate, search_config, config)
    raw_scored = score_frame(calibration, candidate, parameters, None)
    calibrator = fit_probability_calibrator(
        raw_scored["raw_probability_up"].to_numpy(),
        raw_scored["label_up"].to_numpy(),
        raw_scored["market_id"].to_list(),
    )
    scored = _apply_calibrator(raw_scored, calibrator)
    metrics = predictive_metrics(scored)
    metrics["margin_mae_bps"] = float(
        (scored["expected_margin_bps"] - scored["target_margin_bps"]).abs().mean()
    )
    return {
        "configuration": asdict(search_config),
        "parameters": parameters.to_dict(),
        "calibrator": calibrator.to_dict(),
        "calibration_metrics": metrics,
    }


def _run_fold(
    frame: pl.DataFrame,
    config: TournamentConfig,
    candidate: CandidateSpec,
    search_config: SearchConfiguration,
    calibrator: ProbabilityCalibrator,
    fold: Fold,
) -> dict[str, Any]:
    training = frame.filter(
        (pl.col("window_start") >= config.historical_start)
        & (pl.col("window_start") < fold.test_start)
    )
    testing = _period(frame, fold.test_start, fold.test_end)
    parameters = _fit_parameters(training, candidate, search_config, config)
    scored = score_frame(testing, candidate, parameters, calibrator).with_columns(
        pl.lit(fold.name).alias("fold")
    )
    ledger = apply_shared_entry_controller(scored, calibrator, config)
    scheduled = testing["market_id"].n_unique()
    report = {
        "name": fold.name,
        "test_start": fold.test_start.isoformat(),
        "test_end": fold.test_end.isoformat(),
        "training_markets": training["market_id"].n_unique(),
        "testing_markets": scheduled,
        "predictive": predictive_metrics(scored),
        "economic": _compact_economic_metrics(ledger, scheduled),
    }
    return {
        "parameters": parameters.to_dict(),
        "scored": scored.to_dict(as_series=False),
        "ledger": ledger.to_dict(as_series=False),
        "report": report,
    }


def _fit_parameters(
    frame: pl.DataFrame,
    candidate: CandidateSpec,
    search_config: SearchConfiguration,
    config: TournamentConfig,
) -> StateSpaceParameters:
    market_ids, observations, targets = _sequence_arrays(frame, candidate)
    if not len(market_ids):
        raise RuntimeError(f"no complete fitting markets for {candidate.name}")
    return fit_chronological_mle(
        [observations[index] for index in range(len(observations))],
        targets,
        candidate,
        search_config,
        reconstruction_p99_bps=float(config.raw["search"]["reconstruction_p99_error_bps"]),
    )


def score_frame(
    frame: pl.DataFrame,
    candidate: CandidateSpec,
    parameters: StateSpaceParameters,
    calibrator: ProbabilityCalibrator | None,
) -> pl.DataFrame:
    market_ids, observations, _ = _sequence_arrays(frame, candidate)
    if not len(market_ids):
        return pl.DataFrame()
    eligible = frame.filter(pl.col("market_id").is_in(market_ids)).sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    outputs: dict[str, list[np.ndarray]] = {
        "raw_probability_up": [],
        "expected_margin_bps": [],
        "margin_p05_bps": [],
        "margin_p50_bps": [],
        "margin_p95_bps": [],
        "reversal_probability": [],
        "margin_velocity_bps_per_step": [],
        "process_uncertainty_bps2": [],
        "sensor_uncertainty_bps2": [],
        "stable_regime_probability": [],
        "trending_regime_probability": [],
        "reversal_regime_probability": [],
    }
    for sequence in observations:
        result = filter_sequence(sequence, candidate, parameters)
        outputs["raw_probability_up"].append(result.probability_up)
        outputs["expected_margin_bps"].append(result.expected_margin_bps)
        outputs["margin_p05_bps"].append(result.margin_p05_bps)
        outputs["margin_p50_bps"].append(result.margin_p50_bps)
        outputs["margin_p95_bps"].append(result.margin_p95_bps)
        outputs["reversal_probability"].append(result.reversal_probability)
        outputs["margin_velocity_bps_per_step"].append(result.margin_velocity_bps_per_step)
        outputs["process_uncertainty_bps2"].append(result.process_uncertainty_bps2)
        outputs["sensor_uncertainty_bps2"].append(result.sensor_uncertainty_bps2)
        outputs["stable_regime_probability"].append(result.regime_probabilities[:, 0])
        outputs["trending_regime_probability"].append(result.regime_probabilities[:, 1])
        outputs["reversal_regime_probability"].append(result.regime_probabilities[:, 2])
    scored = eligible.with_columns(
        *[pl.Series(name, np.concatenate(values)) for name, values in outputs.items()]
    )
    return _apply_calibrator(scored, calibrator) if calibrator is not None else scored


def _apply_calibrator(
    scored: pl.DataFrame, calibrator: ProbabilityCalibrator | None
) -> pl.DataFrame:
    if calibrator is None:
        return scored
    probability = calibrator.transform(scored["raw_probability_up"].to_numpy())
    return scored.with_columns(pl.Series("probability_up", probability))


def _sequence_arrays(
    frame: pl.DataFrame, candidate: CandidateSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = frame.filter(
        pl.col("sensor_valid")
        & pl.all_horizontal(pl.col(sensor).is_finite() for sensor in candidate.sensors)
    ).sort(["window_start", "market_id", "seconds_elapsed"])
    complete = (
        eligible.group_by("market_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("seconds_elapsed").n_unique().alias("seconds"),
        )
        .filter(
            (pl.col("rows") == len(CHECKPOINT_SECONDS))
            & (pl.col("seconds") == len(CHECKPOINT_SECONDS))
        )["market_id"]
    )
    eligible = eligible.filter(pl.col("market_id").is_in(complete)).sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    if eligible.is_empty():
        return np.array([], dtype=object), np.empty((0, 0, len(candidate.sensors))), np.array([])
    rows = len(CHECKPOINT_SECONDS)
    observations = eligible.select(candidate.sensors).to_numpy().reshape(-1, rows, len(candidate.sensors))
    market_ids = eligible["market_id"].to_numpy()[::rows]
    targets = eligible["target_margin_bps"].to_numpy()[::rows].astype(float)
    return market_ids, observations, targets


def apply_shared_entry_controller(
    scored: pl.DataFrame,
    calibrator: ProbabilityCalibrator,
    config: TournamentConfig,
) -> pl.DataFrame:
    if scored.is_empty():
        return scored
    execution = config.raw["execution"]
    threshold = float(config.raw["entry"]["minimum_calibrated_correctness"])
    maximum_ratio = float(config.raw["entry"]["maximum_wins_per_loss"])
    reserve = float(execution["execution_reserve_per_share"])
    stress = float(execution["stress_slippage_per_share"])
    probability = scored["probability_up"].to_numpy()
    predicted_up = probability >= 0.5
    correctness = np.where(predicted_up, probability, 1.0 - probability)
    up_cost = scored["up_ask_vwap_5"].to_numpy().astype(float)
    down_cost = scored["down_ask_vwap_5"].to_numpy().astype(float)
    selected_cost = np.where(predicted_up, up_cost, down_cost)
    fee_rate = scored["fee_rate"].fill_null(float("nan")).to_numpy().astype(float)
    selected_fee = fee_rate * selected_cost * (1.0 - selected_cost)
    all_in_stressed = selected_cost + selected_fee + reserve + stress
    stressed_profit = 1.0 - all_in_stressed
    stressed_loss = all_in_stressed
    recovery_ratio = stressed_loss / np.maximum(stressed_profit, 1e-12)
    support = max(calibrator.support_markets, 1)
    correctness_lower = np.clip(
        correctness - 1.96 * np.sqrt(correctness * (1.0 - correctness) / support),
        0.0,
        1.0,
    )
    up_fee = fee_rate * up_cost * (1.0 - up_cost)
    down_fee = fee_rate * down_cost * (1.0 - down_cost)
    up_all_in = up_cost + up_fee + reserve + stress
    down_all_in = down_cost + down_fee + reserve + stress
    up_expected = probability - up_all_in
    down_expected = (1.0 - probability) - down_all_in
    probability_lower_up = np.clip(
        probability - 1.96 * np.sqrt(probability * (1.0 - probability) / support), 0.0, 1.0
    )
    probability_upper_up = np.clip(
        probability + 1.96 * np.sqrt(probability * (1.0 - probability) / support), 0.0, 1.0
    )
    up_tail = probability_lower_up - up_all_in
    down_tail = (1.0 - probability_upper_up) - down_all_in
    tail_selected = np.where(predicted_up, up_tail, down_tail)
    direction_correct = predicted_up == scored["label_up"].to_numpy().astype(bool)
    eligible = (
        scored["book_valid"].to_numpy().astype(bool)
        & np.isfinite(selected_cost)
        & np.isfinite(selected_fee)
        & (correctness >= threshold)
        & (tail_selected > 0.0)
        & (stressed_profit > 0.0)
        & (recovery_ratio <= maximum_ratio)
    )
    enriched = scored.with_columns(
        pl.Series("predicted_up", predicted_up),
        pl.Series("calibrated_correctness", correctness),
        pl.Series("lower_calibrated_correctness", correctness_lower),
        pl.Series("expected_payoff_up", up_expected),
        pl.Series("expected_payoff_down", down_expected),
        pl.Series("tail_risk_adjusted_payoff_up", up_tail),
        pl.Series("tail_risk_adjusted_payoff_down", down_tail),
        pl.Series("selected_tail_risk_adjusted_payoff", tail_selected),
        pl.Series("selected_cost_5", selected_cost),
        pl.Series("selected_fee_5", selected_fee),
        pl.Series("stressed_profit_if_correct_per_share", stressed_profit),
        pl.Series("stressed_loss_if_wrong_per_share", stressed_loss),
        pl.Series("quoted_loss_recovery_ratio", recovery_ratio),
        pl.Series("direction_correct", direction_correct),
        pl.Series("entry_eligible", eligible),
    )
    selected = (
        enriched.filter(pl.col("entry_eligible"))
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .unique(subset=["market_id"], keep="first", maintain_order=True)
        .sort(["window_start", "market_id"])
    )
    if selected.is_empty():
        return selected
    quantity = int(execution["evaluation_quantity"])
    gross = quantity * (selected["direction_correct"].cast(pl.Float64) - selected["selected_cost_5"])
    fee_net = gross - quantity * selected["selected_fee_5"]
    reserve_net = fee_net - quantity * reserve
    stressed = reserve_net - quantity * stress
    return selected.with_columns(
        gross.alias("gross_pnl"),
        fee_net.alias("fee_adjusted_pnl"),
        reserve_net.alias("net_pnl"),
        stressed.alias("stressed_pnl"),
    )


def predictive_metrics(scored: pl.DataFrame) -> dict[str, Any]:
    if scored.is_empty():
        return {
            "markets": 0,
            "rows": 0,
            "brier_score": None,
            "log_loss": None,
            "expected_calibration_error": None,
        }
    probability_column = "probability_up" if "probability_up" in scored.columns else "raw_probability_up"
    probability = np.clip(scored[probability_column].to_numpy().astype(float), 1e-9, 1 - 1e-9)
    labels = scored["label_up"].to_numpy().astype(float)
    market_counts = scored.group_by("market_id").len().rename({"len": "market_rows"})
    weighted = scored.join(market_counts, on="market_id", how="left")
    weights = 1.0 / weighted["market_rows"].to_numpy().astype(float)
    weights /= weights.sum()
    log_losses = -(labels * np.log(probability) + (1 - labels) * np.log(1 - probability))
    return {
        "markets": scored["market_id"].n_unique(),
        "rows": scored.height,
        "brier_score": float(np.sum(weights * (probability - labels) ** 2)),
        "log_loss": float(np.sum(weights * log_losses)),
        "expected_calibration_error": _ece(labels, probability, weights),
        "mean_probability_up": float(np.sum(weights * probability)),
    }


def trading_metrics(
    ledger: pl.DataFrame,
    scored: pl.DataFrame,
    *,
    scheduled_markets: int,
    config: TournamentConfig,
    fold_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    probability = predictive_metrics(scored)
    if ledger.is_empty():
        return {
            "markets": scheduled_markets,
            "trades": 0,
            "coverage": 0.0,
            "accuracy": None,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "stressed_pnl": 0.0,
            "stressed_expectancy": None,
            "profit_factor": None,
            "maximum_drawdown": 0.0,
            "cvar_10": None,
            "predictive": probability,
            "folds": fold_reports,
        }
    pnl = ledger["stressed_pnl"].to_numpy().astype(float)
    correct = ledger["direction_correct"].to_numpy().astype(bool)
    winning_pnl = pnl[pnl > 0]
    losing_pnl = pnl[pnl < 0]
    average_win = float(winning_pnl.mean()) if len(winning_pnl) else 0.0
    average_loss = float(losing_pnl.mean()) if len(losing_pnl) else 0.0
    loss_recovery = abs(average_loss) / average_win if average_win > 0 else math.inf
    ordered = ledger.sort(["window_start", "market_id"])
    cumulative = np.cumsum(ordered["stressed_pnl"].to_numpy().astype(float))
    peak = np.maximum.accumulate(np.r_[0.0, cumulative])[:-1]
    drawdown = peak - cumulative
    daily_frame = ledger.with_columns(pl.col("window_start").dt.date().alias("day")).group_by(
        "day"
    ).agg(
        pl.len().alias("trades"),
        pl.col("direction_correct").sum().alias("wins"),
        pl.col("stressed_pnl").sum().alias("stressed_pnl"),
        pl.col("net_pnl").sum().alias("net_pnl"),
    ).sort("day")
    positive_daily = daily_frame.filter(pl.col("stressed_pnl") > 0)["stressed_pnl"].to_numpy()
    concentration = (
        float(positive_daily.max() / positive_daily.sum()) if len(positive_daily) else math.inf
    )
    bootstrap = _day_bootstrap(
        daily_frame,
        int(config.raw["gates"]["bootstrap_resamples"]),
        config.random_seed,
    )
    fold_pnl = {
        row["fold"]: float(row["pnl"])
        for row in ledger.group_by("fold").agg(pl.col("stressed_pnl").sum().alias("pnl")).iter_rows(named=True)
    }
    profitable_folds = sum(value > 0 for value in fold_pnl.values())
    directions = {
        name: _group_trade_metrics(ledger.filter(pl.col("predicted_up") == value))
        for name, value in (("UP", True), ("DOWN", False))
    }
    entry_bands = _band_metrics(
        ledger,
        "seconds_elapsed",
        ((30, 60), (60, 90), (90, 120), (120, 151)),
    )
    price_bands = _band_metrics(
        ledger,
        "selected_cost_5",
        ((0.0, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.01)),
    )
    entry = ledger["seconds_elapsed"].to_numpy().astype(float)
    loss_percentiles = {
        f"p{percentile}": float(np.quantile(losing_pnl, percentile / 100.0))
        if len(losing_pnl)
        else None
        for percentile in (5, 25, 50, 75, 95)
    }
    return {
        "markets": scheduled_markets,
        "trades": ledger.height,
        "coverage": ledger.height / max(scheduled_markets, 1),
        "accuracy": float(correct.mean()),
        "wins": int(correct.sum()),
        "losses": int((~correct).sum()),
        "up_trade_share": directions["UP"]["trades"] / ledger.height,
        "down_trade_share": directions["DOWN"]["trades"] / ledger.height,
        "per_direction": directions,
        "average_win": average_win,
        "average_loss": average_loss,
        "wins_to_recover_average_loss": loss_recovery,
        "worst_loss": float(pnl.min()),
        "loss_percentiles": loss_percentiles,
        "gross_pnl": float(ledger["gross_pnl"].sum()),
        "net_pnl": float(ledger["net_pnl"].sum()),
        "stressed_pnl": float(pnl.sum()),
        "stressed_expectancy": float(pnl.mean()),
        "profit_factor": float(winning_pnl.sum() / -losing_pnl.sum())
        if len(losing_pnl)
        else math.inf,
        "maximum_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "cvar_10": _cvar(pnl),
        "mean_entry_second": float(entry.mean()),
        "median_entry_second": float(np.median(entry)),
        "p10_entry_second": float(np.quantile(entry, 0.10)),
        "p25_entry_second": float(np.quantile(entry, 0.25)),
        "p75_entry_second": float(np.quantile(entry, 0.75)),
        "p90_entry_second": float(np.quantile(entry, 0.90)),
        "entry_bands": entry_bands,
        "price_bands": price_bands,
        "daily": {
            str(row["day"]): {
                "trades": int(row["trades"]),
                "wins": int(row["wins"]),
                "net_pnl": float(row["net_pnl"]),
                "stressed_pnl": float(row["stressed_pnl"]),
            }
            for row in daily_frame.iter_rows(named=True)
        },
        "folds": fold_reports,
        "fold_stressed_pnl": fold_pnl,
        "profitable_fold_ratio": profitable_folds / max(len(config.folds), 1),
        "maximum_positive_pnl_day_share": concentration,
        "bootstrap_stressed_expectancy": bootstrap,
        "capacity": _capacity_metrics(ledger, config),
        "maximum_quote_loss_recovery_ratio": float(ledger["quoted_loss_recovery_ratio"].max()),
        "predictive": probability,
    }


def _compact_economic_metrics(ledger: pl.DataFrame, scheduled: int) -> dict[str, Any]:
    if ledger.is_empty():
        return {"trades": 0, "coverage": 0.0, "accuracy": None, "stressed_pnl": 0.0}
    return {
        "trades": ledger.height,
        "coverage": ledger.height / max(scheduled, 1),
        "accuracy": float(ledger["direction_correct"].mean()),
        "stressed_pnl": float(ledger["stressed_pnl"].sum()),
        "mean_entry_second": float(ledger["seconds_elapsed"].mean()),
    }


def _group_trade_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "accuracy": None,
            "stressed_pnl": 0.0,
            "stressed_expectancy": None,
            "average_win": None,
            "average_loss": None,
            "mean_entry_second": None,
        }
    pnl = frame["stressed_pnl"].to_numpy().astype(float)
    correct = frame["direction_correct"].to_numpy().astype(bool)
    return {
        "trades": frame.height,
        "wins": int(correct.sum()),
        "losses": int((~correct).sum()),
        "accuracy": float(correct.mean()),
        "stressed_pnl": float(pnl.sum()),
        "stressed_expectancy": float(pnl.mean()),
        "average_win": float(pnl[pnl > 0].mean()) if np.any(pnl > 0) else None,
        "average_loss": float(pnl[pnl < 0].mean()) if np.any(pnl < 0) else None,
        "mean_entry_second": float(frame["seconds_elapsed"].mean()),
    }


def _band_metrics(
    frame: pl.DataFrame,
    column: str,
    bands: tuple[tuple[float, float], ...],
) -> dict[str, Any]:
    result = {}
    for lower, upper in bands:
        subset = frame.filter((pl.col(column) >= lower) & (pl.col(column) < upper))
        result[f"{lower:g}-{upper:g}"] = _group_trade_metrics(subset)
    return result


def _capacity_metrics(ledger: pl.DataFrame, config: TournamentConfig) -> dict[str, Any]:
    reserve = float(config.raw["execution"]["execution_reserve_per_share"])
    stress = float(config.raw["execution"]["stress_slippage_per_share"])
    result: dict[str, Any] = {}
    for quantity in VWAP_QUANTITIES:
        up = ledger[f"up_ask_vwap_{quantity}"].to_numpy().astype(float)
        down = ledger[f"down_ask_vwap_{quantity}"].to_numpy().astype(float)
        selected = np.where(ledger["predicted_up"].to_numpy().astype(bool), up, down)
        valid = np.isfinite(selected)
        fee_rate = ledger["fee_rate"].to_numpy().astype(float)
        fee = fee_rate * selected * (1.0 - selected)
        correct = ledger["direction_correct"].to_numpy().astype(float)
        net = quantity * (correct - selected - fee - reserve)
        stressed = net - quantity * stress
        result[str(quantity)] = {
            "supported_trades": int(valid.sum()),
            "net_pnl": float(net[valid].sum()) if valid.any() else 0.0,
            "stressed_pnl": float(stressed[valid].sum()) if valid.any() else 0.0,
            "stressed_expectancy": float(stressed[valid].mean()) if valid.any() else None,
        }
    return result


def refprice_contribution_tests(
    candidate_results: dict[str, Any],
    ledgers: dict[str, pl.DataFrame],
    scored: dict[str, pl.DataFrame],
    config: TournamentConfig,
) -> dict[str, Any]:
    pairs = (
        ("single_regime", "twap_refprice_single_regime", "twap_single_regime"),
        ("regime_switching", "twap_refprice_regime_switching", "twap_regime_switching"),
    )
    return {
        name: _paired_contribution(
            refprice_name,
            twap_name,
            candidate_results,
            ledgers,
            scored,
            config,
        )
        for name, refprice_name, twap_name in pairs
    }


def _paired_contribution(
    refprice_name: str,
    twap_name: str,
    results: dict[str, Any],
    ledgers: dict[str, pl.DataFrame],
    scored: dict[str, pl.DataFrame],
    config: TournamentConfig,
) -> dict[str, Any]:
    ref_scored = scored[refprice_name].select(
        "market_id", "window_start", "seconds_elapsed", "probability_up", "label_up"
    ).rename({"probability_up": "ref_probability"})
    twap_scored = scored[twap_name].select(
        "market_id", "seconds_elapsed", "probability_up"
    ).rename({"probability_up": "twap_probability"})
    paired = ref_scored.join(
        twap_scored, on=["market_id", "seconds_elapsed"], how="inner", validate="1:1"
    ).with_columns(
        (
            (pl.col("ref_probability") - pl.col("label_up")) ** 2
            - (pl.col("twap_probability") - pl.col("label_up")) ** 2
        ).alias("brier_difference")
    ).group_by("market_id").agg(
        pl.col("window_start").first(), pl.col("brier_difference").mean()
    )
    brier_interval = _paired_day_interval(
        paired, "brier_difference", config.raw["gates"]["bootstrap_resamples"], config.random_seed
    )
    schedule = pl.concat(
        [
            frame.select("market_id", "window_start")
            for frame in (scored[refprice_name], scored[twap_name])
            if not frame.is_empty()
        ],
        how="vertical",
    ).unique(subset=["market_id"])
    economic = schedule
    for name, ledger in (("ref", ledgers[refprice_name]), ("twap", ledgers[twap_name])):
        values = ledger.select("market_id", pl.col("stressed_pnl").alias(f"{name}_pnl"))
        economic = economic.join(values, on="market_id", how="left")
    economic = economic.with_columns(
        pl.col("ref_pnl").fill_null(0.0),
        pl.col("twap_pnl").fill_null(0.0),
    ).with_columns((pl.col("ref_pnl") - pl.col("twap_pnl")).alias("pnl_difference"))
    expectancy_interval = _paired_day_interval(
        economic,
        "pnl_difference",
        config.raw["gates"]["bootstrap_resamples"],
        config.random_seed + 17,
    )
    fold_differences = {}
    for fold in config.folds:
        block = economic.filter(
            pl.col("window_start").is_between(fold.test_start, fold.test_end, closed="left")
        )
        fold_differences[fold.name] = float(block["pnl_difference"].sum()) if block.height else 0.0
    direction_differences = {}
    for direction, value in (("UP", True), ("DOWN", False)):
        ref_pnl = ledgers[refprice_name].filter(pl.col("predicted_up") == value)["stressed_pnl"].sum()
        twap_pnl = ledgers[twap_name].filter(pl.col("predicted_up") == value)["stressed_pnl"].sum()
        direction_differences[direction] = float((ref_pnl or 0.0) - (twap_pnl or 0.0))
    ref = results[refprice_name]
    twap = results[twap_name]
    checks = {
        "brier_upper_below_zero": brier_interval["upper"] < 0,
        "calibration_not_degraded": (
            ref["predictive"]["expected_calibration_error"]
            <= twap["predictive"]["expected_calibration_error"]
        ),
        "stressed_expectancy_lower_positive": expectancy_interval["lower"] > 0,
        "loss_recovery_not_worse": (
            ref.get("wins_to_recover_average_loss", math.inf)
            <= twap.get("wins_to_recover_average_loss", math.inf)
        ),
        "drawdown_not_worse": ref.get("maximum_drawdown", math.inf) <= twap.get("maximum_drawdown", math.inf),
        "multiple_positive_folds": sum(value > 0 for value in fold_differences.values()) >= 2,
        "both_directions_improve": all(value > 0 for value in direction_differences.values()),
    }
    return {
        "refprice_candidate": refprice_name,
        "matching_twap_candidate": twap_name,
        "paired_markets": paired.height,
        "brier_score_difference_refprice_minus_twap": brier_interval,
        "stressed_pnl_per_scheduled_market_difference": expectancy_interval,
        "fold_differences": fold_differences,
        "direction_differences": direction_differences,
        "checks": checks,
        "admitted": all(checks.values()),
    }


def apply_development_qualification(
    results: dict[str, Any],
    paired_tests: dict[str, Any],
    comparator: dict[str, Any],
    development_evidence: dict[str, Any],
    config: TournamentConfig,
) -> dict[str, Any]:
    gates = config.raw["gates"]
    refprice_admission = {
        "twap_refprice_single_regime": paired_tests["single_regime"]["admitted"],
        "twap_refprice_regime_switching": paired_tests["regime_switching"]["admitted"],
    }
    qualified = []
    for name, row in results.items():
        average_win = row.get("average_win", 0.0) or 0.0
        average_loss = row.get("average_loss", 0.0) or 0.0
        capacity_five = row.get("capacity", {}).get("5", {})
        checks = {
            "official_development_period_complete": development_evidence["complete"],
            "accuracy": (row.get("accuracy") or 0.0) >= float(gates["minimum_accuracy"]),
            "profit_factor": (row.get("profit_factor") or 0.0) >= float(gates["minimum_profit_factor"]),
            "positive_stressed_pnl": row.get("stressed_pnl", 0.0) > 0,
            "positive_stressed_expectancy": (row.get("stressed_expectancy") or 0.0) > 0,
            "positive_bootstrap_lower": (
                row.get("bootstrap_stressed_expectancy", {}).get("lower", -math.inf) > 0
            ),
            "average_loss_recovery": (
                abs(average_loss) / average_win <= float(gates["maximum_wins_per_loss"])
                if average_win > 0
                else False
            ),
            "every_quote_recovery": row.get("maximum_quote_loss_recovery_ratio", math.inf)
            <= float(gates["maximum_wins_per_loss"]),
            "average_entry": row.get("mean_entry_second", math.inf)
            <= float(gates["maximum_average_entry_second"]),
            "profitable_folds": row.get("profitable_fold_ratio", 0.0)
            >= float(gates["minimum_profitable_fold_ratio"]),
            "both_directions": min(
                row.get("up_trade_share", 0.0), row.get("down_trade_share", 0.0)
            )
            >= float(gates["minimum_direction_share"]),
            "pnl_day_concentration": row.get("maximum_positive_pnl_day_share", math.inf)
            <= float(gates["maximum_positive_pnl_day_share"]),
            "positive_vwap5_capacity": capacity_five.get("stressed_pnl", 0.0) > 0,
            "maximum_drawdown_nonworse": row.get("maximum_drawdown", math.inf)
            <= comparator["maximum_drawdown"],
            "cvar_nonworse": (row.get("cvar_10") or -math.inf) >= comparator["cvar_10"],
            "coverage": row.get("coverage", 0.0) >= float(gates["minimum_coverage"]),
            "trade_support": row.get("trades", 0) >= int(gates["minimum_trades"]),
            "refprice_admission": refprice_admission.get(name, True),
        }
        row["qualification"] = {
            "passed": all(checks.values()),
            "checks": checks,
            "reasons": [key for key, passed in checks.items() if not passed],
        }
        if all(checks.values()):
            qualified.append(name)
    if qualified:
        winner = max(
            qualified,
            key=lambda name: (
                results[name]["bootstrap_stressed_expectancy"]["lower"],
                results[name]["trades"],
                -results[name]["mean_entry_second"],
                -results[name]["maximum_drawdown"],
            ),
        )
        status = "development_candidate_frozen_pending_prospective_evidence"
    else:
        winner = None
        status = "no_candidate_qualified"
    return {
        "status": status,
        "selected_candidate": winner,
        "qualified_candidates": qualified,
        "selection_order": [
            "highest bootstrap lower stressed expectancy",
            "highest trade count",
            "earliest average entry",
            "lowest maximum drawdown",
        ],
        "refprice_conclusion": (
            "RefPrice does not provide qualified incremental value under the current TWAP settlement mechanism."
            if not any(refprice_admission.values())
            else "At least one RefPrice arm passed the frozen development admission rule."
        ),
    }


def _development_evidence(frame: pl.DataFrame, config: TournamentConfig) -> dict[str, Any]:
    official = _period(frame, config.development_start, config.development_end).select(
        "market_id", "window_start"
    ).unique(subset=["market_id"])
    daily = official.with_columns(pl.col("window_start").dt.date().alias("day")).group_by(
        "day"
    ).agg(pl.len().alias("markets")).sort("day")
    expected_days = int(
        (config.development_end - config.development_start).total_seconds() // 86_400
    )
    expected_markets_per_day = 288
    daily_counts = {
        str(row["day"]): int(row["markets"]) for row in daily.iter_rows(named=True)
    }
    return {
        "complete": (
            daily.height == expected_days
            and all(value == expected_markets_per_day for value in daily_counts.values())
        ),
        "expected_complete_utc_days": expected_days,
        "observed_utc_days": daily.height,
        "expected_markets_per_day": expected_markets_per_day,
        "observed_markets": official.height,
        "daily_market_counts": daily_counts,
    }


def _fit_final_models(
    frame: pl.DataFrame,
    config: TournamentConfig,
    selected: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    store: CheckpointStore,
    frame_manifest: dict[str, Any],
) -> dict[str, Any]:
    frozen_training = frame.filter(pl.col("window_start") < config.freeze_at)
    names = [selection["selected_candidate"]] if selection["selected_candidate"] else [
        candidate.name for candidate in CANDIDATES
    ]
    models = {}
    for name in names:
        candidate = next(candidate for candidate in CANDIDATES if candidate.name == name)
        search_config = SearchConfiguration(**selected[name]["configuration"])
        calibrator = ProbabilityCalibrator.from_dict(selected[name]["calibrator"])
        parameters = store.get_or_compute(
            f"final-{name}",
            _hash_payload(
                {
                    "source": frame_manifest["sha256"],
                    "candidate": asdict(candidate),
                    "configuration": asdict(search_config),
                    "freeze": config.freeze_at,
                }
            ),
            lambda candidate=candidate, search_config=search_config: _fit_parameters(
                frozen_training, candidate, search_config, config
            ).to_dict(),
        )
        models[name] = {
            "candidate": asdict(candidate),
            "parameters": parameters,
            "probability_calibrator": calibrator.to_dict(),
            "configuration": asdict(search_config),
            "fit_through_exclusive": config.freeze_at.isoformat(),
        }
    return models


def prospective_status(
    frame: pl.DataFrame, selection: dict[str, Any], config: TournamentConfig
) -> dict[str, Any]:
    prospective = frame.filter(pl.col("window_start") >= config.prospective_start)
    markets = prospective["market_id"].n_unique() if prospective.height else 0
    days = prospective["window_start"].dt.date().n_unique() if prospective.height else 0
    requirements = config.raw["prospective"]
    evidence = {
        "markets": markets,
        "complete_utc_days": days,
        "chronological_blocks": 0,
        "trades": 0,
    }
    checks = {
        "development_candidate_exists": selection["selected_candidate"] is not None,
        "minimum_markets": markets >= int(requirements["minimum_markets"]),
        "minimum_complete_days": days >= int(requirements["minimum_complete_days"]),
        "minimum_blocks": False,
        "minimum_trades": False,
        "development_gates_repeat": False,
        "refprice_admission_repeats_if_applicable": False,
    }
    return {
        "status": "paper_only_pending_evidence"
        if selection["selected_candidate"] is not None
        else "not_applicable_no_development_candidate",
        "untouched": True,
        "tuning_performed": False,
        "evidence": evidence,
        "requirements": requirements,
        "checks": checks,
    }


def evidence_separation_metrics(
    frame: pl.DataFrame,
    selected: dict[str, dict[str, Any]],
    config: TournamentConfig,
) -> dict[str, Any]:
    periods = {
        "synthetic_chainlink_historical_fit": _period(
            frame, config.historical_start, config.historical_end
        ),
        "authentic_counterfactual_calibration": _period(
            frame, config.calibration_start, config.calibration_end
        ),
        "official_development": _period(frame, config.development_start, config.development_end),
        "prospective": frame.filter(pl.col("window_start") >= config.prospective_start),
    }
    result = {}
    for name, block in periods.items():
        result[name] = {
            "markets": block["market_id"].n_unique() if block.height else 0,
            "rows": block.height,
            "label_sources": sorted(block["label_source"].unique().to_list()) if block.height else [],
            "evidentiary_role": {
                "synthetic_chainlink_historical_fit": "historical fitting only; cannot qualify",
                "authentic_counterfactual_calibration": "selection and calibration only; cannot qualify",
                "official_development": "official development qualification",
                "prospective": "untouched prospective qualification only",
            }[name],
        }
    return result


def _load_corrected_comparator(path: Path) -> dict[str, Any]:
    metrics = json.loads(path.read_text())
    row = metrics["tournament"]["results"]["frozen_chainlink_stratified_payoff"]
    return {
        "identity": "frozen_chainlink_stratified_payoff",
        "source_metrics": str(path),
        "maximum_drawdown": float(row["maximum_drawdown"]),
        "cvar_10": float(row["tail_loss_cvar"]),
        "accuracy": float(row["accuracy"]),
        "stressed_pnl": float(row["stressed_pnl"]),
    }


def _search_report_row(value: dict[str, Any]) -> dict[str, Any]:
    parameters = StateSpaceParameters.from_dict(value["parameters"])
    return {
        "configuration": value["configuration"],
        "calibration_metrics": value["calibration_metrics"],
        "fit_markets": parameters.fit_markets,
        "fit_rows": parameters.fit_rows,
        "chronological_log_likelihood": parameters.chronological_log_likelihood,
        "sensor_variances": dict(zip(parameters.sensor_names, parameters.sensor_variances, strict=True)),
        "transition_phi": parameters.transition_phi,
        "process_margin_variance": parameters.process_margin_variance,
        "process_velocity_variance": parameters.process_velocity_variance,
        "regime_transition": parameters.regime_transition,
    }


def _limitations(
    frame: pl.DataFrame,
    prospective: dict[str, Any],
    selection: dict[str, Any],
    config: TournamentConfig,
) -> list[str]:
    limitations = [
        "Synthetic Chainlink-reconstructed TWAP evidence was used only for historical fitting and did not qualify a candidate.",
        "Projected economics assume the recorded executable ask VWAP was fillable at each selected checkpoint.",
        "No model was deployed and no trading process, ingester, SQL source, table, migration, or database row was changed.",
    ]
    if prospective["evidence"]["markets"] == 0:
        limitations.append(
            "The immutable extract ends at the 2026-08-28 freeze boundary, so no prospective market was read or scored."
        )
    development_evidence = _development_evidence(frame, config)
    if not development_evidence["complete"]:
        limitations.append(
            "Official development evidence is incomplete: "
            f"{development_evidence['observed_utc_days']} of "
            f"{development_evidence['expected_complete_utc_days']} UTC days and "
            f"{development_evidence['observed_markets']} official markets were available."
        )
    if selection["selected_candidate"] is None:
        limitations.append("No development candidate passed every frozen qualification gate.")
    return limitations


def render_report(metrics: dict[str, Any]) -> str:
    development = metrics["development"]
    lines = [
        "# Latent TWAP Settlement-Risk Tournament",
        "",
        f"Run: `{metrics['run_id']}`  ",
        f"Source commit: `{metrics['source_commit']}`  ",
        f"Artifact SHA-256: `{metrics['model_artifact']['sha256']}`  ",
        f"Development result: **{development['selection']['status']}**  ",
        f"Prospective result: **{metrics['prospective']['status']}**",
        "",
        "## Candidate results",
        "",
        "| Candidate | Trades | Coverage | Accuracy | UP / DOWN | Stressed PnL | Expectancy | Profit factor | Avg entry | Max DD | CVaR | Qualified |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in development["candidates"].items():
        lines.append(
            "| "
            + " | ".join(
                (
                    name,
                    str(row.get("trades", 0)),
                    _fmt_pct(row.get("coverage")),
                    _fmt_pct(row.get("accuracy")),
                    f"{row.get('per_direction', {}).get('UP', {}).get('trades', 0)} / {row.get('per_direction', {}).get('DOWN', {}).get('trades', 0)}",
                    _fmt(row.get("stressed_pnl")),
                    _fmt(row.get("stressed_expectancy")),
                    _fmt(row.get("profit_factor")),
                    _fmt(row.get("mean_entry_second")),
                    _fmt(row.get("maximum_drawdown")),
                    _fmt(row.get("cvar_10")),
                    str(row.get("qualification", {}).get("passed", False)),
                )
            )
            + " |"
        )
    lines.extend(["", "## Detailed performance", ""])
    for name, row in development["candidates"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Wins/losses: {row.get('wins', 0)} / {row.get('losses', 0)}; average win {_fmt(row.get('average_win'))}; average loss {_fmt(row.get('average_loss'))}; wins to recover one average loss {_fmt(row.get('wins_to_recover_average_loss'))}.",
                f"- Worst loss {_fmt(row.get('worst_loss'))}; loss percentiles `{json.dumps(row.get('loss_percentiles', {}), sort_keys=True)}`.",
                f"- Gross/net/stressed PnL: {_fmt(row.get('gross_pnl'))} / {_fmt(row.get('net_pnl'))} / {_fmt(row.get('stressed_pnl'))}.",
                f"- Entry seconds mean/median/p10/p90: {_fmt(row.get('mean_entry_second'))} / {_fmt(row.get('median_entry_second'))} / {_fmt(row.get('p10_entry_second'))} / {_fmt(row.get('p90_entry_second'))}.",
                f"- Brier/log loss/ECE: {_fmt(row.get('predictive', {}).get('brier_score'))} / {_fmt(row.get('predictive', {}).get('log_loss'))} / {_fmt(row.get('predictive', {}).get('expected_calibration_error'))}.",
                f"- Positive-fold ratio {_fmt_pct(row.get('profitable_fold_ratio'))}; maximum positive-PnL day share {_fmt_pct(row.get('maximum_positive_pnl_day_share'))}.",
                f"- Qualification failures: {', '.join(row.get('qualification', {}).get('reasons', [])) or 'none'}.",
                "",
                "Direction metrics:",
                "",
                "```json",
                json.dumps(row.get("per_direction", {}), indent=2, sort_keys=True),
                "```",
                "",
                "Entry bands, price bands, daily/fold results, and capacity from 5 through 200 shares are retained in `metrics.json`.",
                "",
            ]
        )
    lines.extend(
        [
            "## RefPrice contribution",
            "",
            development["selection"]["refprice_conclusion"],
            "",
            "```json",
            json.dumps(development["refprice_contribution"], indent=2, sort_keys=True),
            "```",
            "",
            "## Evidence separation",
            "",
            "```json",
            json.dumps(metrics["evidence_separation"], indent=2, sort_keys=True),
            "```",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in metrics["limitations"])
    return "\n".join(lines) + "\n"


def _day_bootstrap(
    daily: pl.DataFrame, resamples: int, seed: int
) -> dict[str, Any]:
    if daily.is_empty() or not int(daily["trades"].sum()):
        return {"lower": None, "median": None, "upper": None, "resamples": resamples}
    pnl = daily["stressed_pnl"].to_numpy().astype(float)
    trades = daily["trades"].to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=float)
    for index in range(resamples):
        selected = rng.integers(0, len(pnl), len(pnl))
        samples[index] = pnl[selected].sum() / max(trades[selected].sum(), 1.0)
    return {
        "lower": float(np.quantile(samples, 0.025)),
        "median": float(np.quantile(samples, 0.50)),
        "upper": float(np.quantile(samples, 0.975)),
        "resamples": resamples,
        "unit": "stressed_pnl_per_trade",
    }


def _paired_day_interval(
    frame: pl.DataFrame,
    value_column: str,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    if frame.is_empty():
        return {"mean": math.nan, "lower": math.nan, "upper": math.nan, "resamples": resamples}
    daily = frame.with_columns(pl.col("window_start").dt.date().alias("day")).group_by(
        "day"
    ).agg(pl.col(value_column).mean().alias("value")).sort("day")
    values = daily["value"].to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=float)
    for index in range(resamples):
        samples[index] = values[rng.integers(0, len(values), len(values))].mean()
    return {
        "mean": float(frame[value_column].mean()),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "resamples": resamples,
    }


def _ece(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> float:
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


def _cvar(values: np.ndarray, fraction: float = 0.10) -> float:
    if not len(values):
        return math.nan
    count = max(1, math.ceil(len(values) * fraction))
    return float(np.sort(values)[:count].mean())


def _period(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    return frame.filter(pl.col("window_start").is_between(start, end, closed="left"))


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(type(value).__name__)


def _git_revision(package_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=package_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _path(package_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else package_root / path


def _utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "inf" if number > 0 else "n/a"
    return f"{number:.4f}"


def _fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(prog="latent-twap-settlement-risk-tournament")
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    result, metrics = run_tournament(load_config(arguments.config))
    print(
        json.dumps(
            {
                "result": str(result),
                "status": metrics["development"]["selection"]["status"],
                "artifact_sha256": metrics["model_artifact"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
