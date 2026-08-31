"""Resumable Binance-context challenger tournament for latent TWAP settlement risk."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import scipy
import sklearn

from . import latent_twap_tournament as latent
from .binance_latent_twap_context import (
    KLINE_FEATURES,
    OI_FEATURES,
    ContextSourceCache,
    apply_residual_adjustment,
    attach_l2_features,
    attach_open_interest_features,
    context_coverage,
    derive_kline_checkpoint_features,
    fit_residual_adjustment,
    load_or_extract_context_sources,
    verify_context_causality,
)
from .core_extract import file_sha256
from .latent_twap_state_space import (
    CANDIDATES,
    CandidateSpec,
    ProbabilityCalibrator,
    SearchConfiguration,
    StateSpaceParameters,
    fit_probability_calibrator,
    predetermined_configurations,
)
from .spot_l2_chainlink_features import L2_FEATURES

SCHEMA_VERSION = "btc-binance-latent-twap-context-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-binance-latent-twap-context-artifact-v1"


@dataclass(frozen=True)
class ChallengerSpec:
    name: str
    feature_family: str
    features: tuple[str, ...]
    base_candidate: str = "twap_regime_switching"
    research_only: bool = False


CHALLENGERS = (
    ChallengerSpec(
        "twap_regime_kline_residual",
        "kline",
        KLINE_FEATURES,
    ),
    ChallengerSpec(
        "twap_regime_open_interest_residual",
        "open_interest",
        OI_FEATURES,
    ),
    ChallengerSpec(
        "twap_regime_kline_open_interest_residual",
        "kline_open_interest",
        (*KLINE_FEATURES, *OI_FEATURES),
    ),
    ChallengerSpec(
        "twap_regime_l2_residual_research",
        "l2",
        tuple(L2_FEATURES),
        research_only=True,
    ),
)


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
    folds: tuple[latent.Fold, ...]
    source_cache: Path
    source_frame: Path
    source_manifest: Path
    label_audit: Path
    comparator_metrics: Path
    context_cache: Path
    runs: Path
    committed_results: Path
    open_interest_source_sql: Path
    l2_source_sql: Path
    l2_extract_end: datetime


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
            latent.Fold(str(row["name"]), _utc(row["test_start"]), _utc(row["test_end"]))
            for row in raw["folds"]
        ),
        source_cache=_path(package_root, paths["source_cache"]),
        source_frame=_path(package_root, paths["source_frame"]),
        source_manifest=_path(package_root, paths["source_manifest"]),
        label_audit=_path(package_root, paths["label_audit"]),
        comparator_metrics=_path(package_root, paths["corrected_comparator_metrics"]),
        context_cache=_path(package_root, paths["context_cache"]),
        runs=_path(package_root, paths["runs"]),
        committed_results=_path(package_root, paths["committed_results"]),
        open_interest_source_sql=_path(package_root, paths["open_interest_source_sql"]),
        l2_source_sql=_path(package_root, paths["l2_source_sql"]),
        l2_extract_end=_utc(raw["binance"]["l2_extract_end"]),
    )
    _validate_config(config)
    return config


def _validate_config(config: TournamentConfig) -> None:
    training = config.raw["training"]
    expected_periods = (
        datetime(2026, 6, 7, tzinfo=UTC),
        datetime(2026, 8, 1, tzinfo=UTC),
        datetime(2026, 8, 1, tzinfo=UTC),
        datetime(2026, 8, 14, tzinfo=UTC),
        datetime(2026, 8, 14, tzinfo=UTC),
        datetime(2026, 8, 28, tzinfo=UTC),
        datetime(2026, 8, 28, tzinfo=UTC),
        datetime(2026, 8, 28, tzinfo=UTC),
    )
    observed_periods = (
        config.historical_start,
        config.historical_end,
        config.calibration_start,
        config.calibration_end,
        config.development_start,
        config.development_end,
        config.prospective_start,
        config.freeze_at,
    )
    if config.profile != "btc_5m_binance_latent_twap_context":
        raise ValueError("unexpected Binance latent-TWAP profile")
    if config.model_family != "btc-5m-binance-latent-twap-context":
        raise ValueError("unexpected Binance latent-TWAP model family")
    if not (
        training["paper_only"]
        and training["strictly_training_only"]
        and not training["live_capital_allowed"]
    ):
        raise ValueError("Binance latent-TWAP work must remain training-only")
    if observed_periods != expected_periods:
        raise ValueError("frozen evidence periods changed")
    if config.l2_extract_end != datetime(2026, 8, 2, tzinfo=UTC):
        raise ValueError("audited L2 availability boundary changed")
    if tuple(config.raw["binance"]["kline_horizons_seconds"]) != (5, 15, 30):
        raise ValueError("kline feature horizons changed")
    if int(config.raw["binance"]["open_interest_max_age_seconds"]) != 300:
        raise ValueError("OI age contract changed")
    if int(config.raw["binance"]["l2_max_age_seconds"]) != 2:
        raise ValueError("L2 age contract changed")
    alphas = tuple(float(value) for value in config.raw["search"]["residual_ridge_alphas"])
    if len(alphas) != 12 or len(set(alphas)) != 12 or any(value <= 0 for value in alphas):
        raise ValueError("residual search must contain twelve distinct positive rows")
    if len(predetermined_configurations()) != 12 or len(config.folds) != 7:
        raise ValueError("control search or fold roster changed")
    if len(CHALLENGERS) != 4 or sum(spec.research_only for spec in CHALLENGERS) != 1:
        raise ValueError("Binance challenger roster changed")
    previous = config.development_start
    for fold in config.folds:
        if fold.test_start != previous or fold.test_end - fold.test_start != timedelta(days=2):
            raise ValueError("development folds must be contiguous two-day UTC blocks")
        previous = fold.test_end
    if previous != config.development_end:
        raise ValueError("development folds do not end at the freeze boundary")
    for file_path, expected in (
        (config.source_frame, config.raw["paths"]["source_frame_sha256"]),
        (config.source_manifest, config.raw["paths"]["source_manifest_sha256"]),
        (config.label_audit, config.raw["paths"]["label_audit_sha256"]),
    ):
        if not file_path.is_file() or file_sha256(file_path) != expected:
            raise RuntimeError(f"immutable latent-TWAP input changed or is missing: {file_path}")
    for path in (
        config.comparator_metrics,
        config.open_interest_source_sql,
        config.l2_source_sql,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def run_tournament(config: TournamentConfig) -> tuple[Path, dict[str, Any]]:
    context_sources, context_source_manifest = load_or_extract_context_sources(config)
    source_identity = hashlib.sha256(
        (
            file_sha256(config.source_path)
            + file_sha256(config.source_manifest)
            + file_sha256(config.source_frame)
            + file_sha256(config.label_audit)
            + file_sha256(context_sources.manifest)
            + _git_revision(config.package_root)
        ).encode()
    ).hexdigest()
    workspace = config.runs / source_identity[:20]
    completion = workspace / "completion.json"
    if completion.is_file():
        record = json.loads(completion.read_text())
        result = config.package_root / record["result"]
        return result, json.loads((result / "metrics.json").read_text())
    workspace.mkdir(parents=True, exist_ok=True)
    store = latent.CheckpointStore(workspace / "checkpoints")
    frame, frame_manifest = _load_or_build_feature_frame(
        config, context_sources, context_source_manifest, workspace
    )
    split_audit = latent.verify_split_separation(frame, config)
    causality_audit = verify_context_causality(frame)
    if not split_audit["passed"] or not causality_audit["passed"]:
        raise RuntimeError("training split or Binance context causality audit failed")

    control_search, control_selected = _search_controls(frame, config, store, frame_manifest)
    control_results, control_ledgers, control_scored, control_folds = _evaluate_controls(
        frame, config, store, frame_manifest, control_selected
    )
    challenger_search, challenger_selected = _search_challengers(
        frame, config, store, frame_manifest, control_selected
    )
    challenger_results, challenger_ledgers, challenger_scored, challenger_folds = (
        _evaluate_challengers(
            frame,
            config,
            store,
            frame_manifest,
            control_selected,
            challenger_selected,
        )
    )
    results = {**control_results, **challenger_results}
    ledgers = {**control_ledgers, **challenger_ledgers}
    scored = {**control_scored, **challenger_scored}
    folds = {**control_folds, **challenger_folds}
    refprice_tests = latent.refprice_contribution_tests(results, ledgers, scored, config)
    context_tests = _context_contribution_tests(results, ledgers, scored, config)
    comparator = latent._load_corrected_comparator(config.comparator_metrics)
    development_evidence = latent._development_evidence(frame, config)
    selection = _apply_context_qualification(
        results,
        refprice_tests,
        context_tests,
        comparator,
        development_evidence,
        config,
    )
    final_models = _fit_final_models(
        frame,
        config,
        control_selected,
        challenger_selected,
        selection,
        store,
        frame_manifest,
    )
    prospective = latent.prospective_status(frame, selection, config)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_family": config.model_family,
        "freeze_at": config.freeze_at.isoformat(),
        "source_commit": _git_revision(config.package_root),
        "source_identity": frame_manifest["sha256"],
        "selection": selection,
        "prospective": prospective,
        "entry_contract": {
            "seconds": list(latent.CHECKPOINT_SECONDS),
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
        ledger.write_parquet(
            ledger_root / f"{name}.parquet", compression="zstd", statistics=True
        )
    coverage = context_coverage(frame, config)
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
            "immutable_twap_source_manifest_sha256": file_sha256(config.source_manifest),
            "immutable_twap_source_frame_sha256": file_sha256(config.source_frame),
            "immutable_label_audit_sha256": file_sha256(config.label_audit),
            "immutable_binance_context_manifest_sha256": file_sha256(
                context_sources.manifest
            ),
            "training_frame": frame_manifest,
            "split_separation": split_audit,
            "context_causality": causality_audit,
            "context_coverage": coverage,
        },
        "search": {"controls": control_search, "challengers": challenger_search},
        "development": {
            "folds": folds,
            "candidates": results,
            "refprice_contribution": refprice_tests,
            "binance_context_contribution": context_tests,
            "corrected_twap_comparator": comparator,
            "official_evidence_completeness": development_evidence,
            "selection": selection,
        },
        "prospective": prospective,
        "evidence_separation": latent.evidence_separation_metrics(
            frame, control_selected, config
        ),
        "model_artifact": {
            "path": "model-family.joblib",
            "sha256": artifact_sha,
            "training_run_id": run_id,
            "qualification_status": selection["status"],
            "deployment_status": "not_deployed",
        },
        "limitations": _limitations(frame, selection, prospective, coverage, config),
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
            "twap_source_manifest_sha256": file_sha256(config.source_manifest),
            "binance_context_manifest_sha256": file_sha256(context_sources.manifest),
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


def _load_or_build_feature_frame(
    config: TournamentConfig,
    context_sources: ContextSourceCache,
    context_manifest: dict[str, Any],
    workspace: Path,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    path = workspace / "binance-context-training-frame.parquet"
    manifest_path = workspace / "binance-context-training-frame-manifest.json"
    if path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if file_sha256(path) != manifest["sha256"]:
            raise RuntimeError("Binance context training-frame checkpoint changed")
        print("checkpoint resume: Binance context training frame", flush=True)
        return pl.read_parquet(path), manifest

    base, base_manifest = latent.build_training_frame(config)
    core_files = sorted((config.source_cache / "core_current").glob("*.parquet"))
    if not core_files:
        raise FileNotFoundError("immutable one-second Binance kline partitions are missing")
    support_start = config.historical_start - timedelta(minutes=1)
    kline_source = (
        pl.scan_parquet(core_files)
        .filter(
            (pl.col("observed_at") >= support_start)
            & (pl.col("observed_at") < config.freeze_at)
        )
        .select(
            "market_id",
            "window_start",
            "observed_at",
            "seconds_elapsed",
            "btc_open",
            "btc_high",
            "btc_low",
            "btc_close",
            "btc_quote_volume",
            "btc_taker_buy_quote_volume",
        )
        .collect()
    )
    kline = derive_kline_checkpoint_features(kline_source)
    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
    frame = base.join(kline, on=keys, how="left", validate="1:1")
    frame = attach_open_interest_features(
        frame,
        pl.read_parquet(context_sources.open_interest),
        maximum_age_seconds=int(config.raw["binance"]["open_interest_max_age_seconds"]),
    )
    frame = attach_l2_features(frame, context_sources.l2)
    frame = frame.with_columns(
        pl.all_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite() for name in KLINE_FEATURES
        ).alias("kline_context_eligible"),
        pl.all_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite() for name in OI_FEATURES
        ).alias("open_interest_context_eligible"),
        pl.all_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite() for name in L2_FEATURES
        ).alias("l2_context_eligible"),
    )
    causality = verify_context_causality(frame)
    if not causality["passed"]:
        raise RuntimeError("Binance context feature frame failed causality checks")
    temporary = path.with_suffix(".parquet.partial")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)
    manifest = {
        "schema_version": "btc-binance-latent-twap-context-frame-v1",
        "sha256": file_sha256(path),
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "twap_frame": base_manifest,
        "twap_source_manifest_sha256": file_sha256(config.source_manifest),
        "binance_context_manifest_sha256": file_sha256(context_sources.manifest),
        "binance_context_contract": context_manifest["contract"],
        "feature_families": {
            "kline": list(KLINE_FEATURES),
            "open_interest": list(OI_FEATURES),
            "l2": list(L2_FEATURES),
        },
        "features_use_completed_targets": False,
        "labels_supervision_only": True,
        "read_only_sources": True,
        "database_mutations": False,
        "causality": causality,
    }
    _write_json(manifest_path, manifest)
    return frame, manifest


def _search_controls(
    frame: pl.DataFrame,
    config: TournamentConfig,
    store: latent.CheckpointStore,
    frame_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    search: dict[str, Any] = {}
    selected: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        rows = []
        for search_config in predetermined_configurations():
            value = store.get_or_compute(
                f"control-search-{candidate.name}-{search_config.identifier}",
                latent._hash_payload(
                    {
                        "source": frame_manifest["sha256"],
                        "candidate": asdict(candidate),
                        "configuration": asdict(search_config),
                        "historical_end": config.historical_end,
                        "calibration_end": config.calibration_end,
                    }
                ),
                lambda candidate=candidate, search_config=search_config: latent._fit_and_calibrate(
                    frame, config, candidate, search_config
                ),
            )
            rows.append(value)
        winner = min(
            rows,
            key=lambda row: (
                row["calibration_metrics"]["brier_score"],
                row["calibration_metrics"]["log_loss"],
                row["calibration_metrics"]["margin_mae_bps"],
                row["configuration"]["identifier"],
            ),
        )
        search[candidate.name] = {
            "budget": len(rows),
            "rows": [latent._search_report_row(row) for row in rows],
            "selected_configuration": winner["configuration"]["identifier"],
        }
        selected[candidate.name] = winner
    return search, selected


def _evaluate_controls(
    frame: pl.DataFrame,
    config: TournamentConfig,
    store: latent.CheckpointStore,
    frame_manifest: dict[str, Any],
    selected: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, pl.DataFrame], dict[str, pl.DataFrame], dict[str, Any]]:
    results: dict[str, Any] = {}
    ledgers: dict[str, pl.DataFrame] = {}
    scored_frames: dict[str, pl.DataFrame] = {}
    reports: dict[str, Any] = {}
    scheduled = _development_markets(frame, config)
    for candidate in CANDIDATES:
        calibrator = ProbabilityCalibrator.from_dict(selected[candidate.name]["calibrator"])
        search_config = SearchConfiguration(**selected[candidate.name]["configuration"])
        candidate_ledgers: list[pl.DataFrame] = []
        candidate_scored: list[pl.DataFrame] = []
        candidate_reports: list[dict[str, Any]] = []
        for fold in config.folds:
            value = store.get_or_compute(
                f"control-fold-{candidate.name}-{fold.name}",
                latent._hash_payload(
                    {
                        "source": frame_manifest["sha256"],
                        "candidate": asdict(candidate),
                        "configuration": asdict(search_config),
                        "calibrator": calibrator.to_dict(),
                        "fold": asdict(fold),
                    }
                ),
                lambda candidate=candidate,
                search_config=search_config,
                calibrator=calibrator,
                fold=fold: latent._run_fold(
                    frame, config, candidate, search_config, calibrator, fold
                ),
            )
            scored = pl.DataFrame(value["scored"])
            ledger = pl.DataFrame(value["ledger"])
            if not scored.is_empty():
                candidate_scored.append(scored)
            if not ledger.is_empty():
                candidate_ledgers.append(ledger)
            candidate_reports.append(value["report"])
        scored_all = _concat(candidate_scored)
        ledger_all = _concat(candidate_ledgers)
        result = latent.trading_metrics(
            ledger_all,
            scored_all,
            scheduled_markets=scheduled,
            config=config,
            fold_reports=candidate_reports,
        )
        result["selected_configuration"] = search_config.identifier
        result["calibration_period"] = selected[candidate.name]["calibration_metrics"]
        result["candidate_role"] = "frozen_latent_control"
        results[candidate.name] = result
        ledgers[candidate.name] = ledger_all
        scored_frames[candidate.name] = scored_all
        reports[candidate.name] = candidate_reports
    return results, ledgers, scored_frames, reports


def _search_challengers(
    frame: pl.DataFrame,
    config: TournamentConfig,
    store: latent.CheckpointStore,
    frame_manifest: dict[str, Any],
    control_selected: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    base_spec = _control("twap_regime_switching")
    base_selected = control_selected[base_spec.name]
    base_parameters = StateSpaceParameters.from_dict(base_selected["parameters"])
    historical = latent._period(frame, config.historical_start, config.historical_end)
    calibration = latent._period(frame, config.calibration_start, config.calibration_end)
    historical_scored = latent.score_frame(historical, base_spec, base_parameters, None)
    calibration_scored = latent.score_frame(calibration, base_spec, base_parameters, None)
    alphas = tuple(float(value) for value in config.raw["search"]["residual_ridge_alphas"])
    search: dict[str, Any] = {}
    selected: dict[str, dict[str, Any]] = {}
    for spec in CHALLENGERS:
        rows = []
        for index, alpha in enumerate(alphas, start=1):
            identifier = f"residual_{index:02d}"
            value = store.get_or_compute(
                f"challenger-search-{spec.name}-{identifier}",
                latent._hash_payload(
                    {
                        "source": frame_manifest["sha256"],
                        "spec": asdict(spec),
                        "base_parameters": base_parameters.to_dict(),
                        "ridge_alpha": alpha,
                    }
                ),
                lambda spec=spec, alpha=alpha, identifier=identifier: _fit_and_calibrate_challenger(
                    historical_scored,
                    calibration_scored,
                    spec,
                    alpha,
                    identifier,
                ),
            )
            rows.append(value)
        eligible_rows = [row for row in rows if row["calibration_metrics"]["rows"] > 0]
        if not eligible_rows:
            raise RuntimeError(f"no calibration evidence for challenger {spec.name}")
        winner = min(
            eligible_rows,
            key=lambda row: (
                row["calibration_metrics"]["brier_score"],
                row["calibration_metrics"]["log_loss"],
                row["calibration_metrics"]["margin_mae_bps"],
                row["configuration"],
            ),
        )
        search[spec.name] = {
            "budget": len(rows),
            "base_candidate": spec.base_candidate,
            "feature_family": spec.feature_family,
            "features": list(spec.features),
            "research_only": spec.research_only,
            "rows": [
                {
                    "configuration": row["configuration"],
                    "ridge_alpha": row["ridge_alpha"],
                    "fit_markets": row["residual_parameters"]["fit_markets"],
                    "fit_rows": row["residual_parameters"]["fit_rows"],
                    "residual_variance_bps2": row["residual_parameters"][
                        "residual_variance_bps2"
                    ],
                    "calibration_metrics": row["calibration_metrics"],
                }
                for row in rows
            ],
            "selected_configuration": winner["configuration"],
        }
        selected[spec.name] = winner
    return search, selected


def _fit_and_calibrate_challenger(
    historical_scored: pl.DataFrame,
    calibration_scored: pl.DataFrame,
    spec: ChallengerSpec,
    alpha: float,
    identifier: str,
) -> dict[str, Any]:
    residual = fit_residual_adjustment(
        historical_scored,
        candidate_name=spec.name,
        context_features=spec.features,
        ridge_alpha=alpha,
    )
    raw = apply_residual_adjustment(
        calibration_scored, residual, context_features=spec.features
    )
    if raw.is_empty():
        predictive = {
            "markets": 0,
            "rows": 0,
            "brier_score": math.inf,
            "log_loss": math.inf,
            "expected_calibration_error": math.inf,
            "margin_mae_bps": math.inf,
        }
        calibrator = ProbabilityCalibrator(1.0, 0, 0)
    else:
        calibrator = fit_probability_calibrator(
            raw["raw_probability_up"].to_numpy(),
            raw["label_up"].to_numpy(),
            raw["market_id"].to_list(),
        )
        calibrated = apply_residual_adjustment(
            calibration_scored,
            residual,
            context_features=spec.features,
            calibrator=calibrator,
        )
        predictive = latent.predictive_metrics(calibrated)
        predictive["margin_mae_bps"] = float(
            (calibrated["expected_margin_bps"] - calibrated["target_margin_bps"])
            .abs()
            .mean()
        )
    return {
        "configuration": identifier,
        "ridge_alpha": alpha,
        "residual_parameters": residual.to_dict(),
        "calibrator": calibrator.to_dict(),
        "calibration_metrics": predictive,
    }


def _evaluate_challengers(
    frame: pl.DataFrame,
    config: TournamentConfig,
    store: latent.CheckpointStore,
    frame_manifest: dict[str, Any],
    control_selected: dict[str, dict[str, Any]],
    selected: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, pl.DataFrame], dict[str, pl.DataFrame], dict[str, Any]]:
    base_spec = _control("twap_regime_switching")
    base_search = SearchConfiguration(
        **control_selected[base_spec.name]["configuration"]
    )
    results: dict[str, Any] = {spec.name: {} for spec in CHALLENGERS}
    ledgers: dict[str, list[pl.DataFrame]] = {spec.name: [] for spec in CHALLENGERS}
    scored_frames: dict[str, list[pl.DataFrame]] = {spec.name: [] for spec in CHALLENGERS}
    reports: dict[str, list[dict[str, Any]]] = {spec.name: [] for spec in CHALLENGERS}
    for fold in config.folds:
        training = frame.filter(
            (pl.col("window_start") >= config.historical_start)
            & (pl.col("window_start") < fold.test_start)
        )
        testing = latent._period(frame, fold.test_start, fold.test_end)
        base_parameters_payload = store.get_or_compute(
            f"challenger-base-{fold.name}",
            latent._hash_payload(
                {
                    "source": frame_manifest["sha256"],
                    "base_candidate": asdict(base_spec),
                    "configuration": asdict(base_search),
                    "fold": asdict(fold),
                }
            ),
            lambda training=training,
            base_spec=base_spec,
            base_search=base_search: latent._fit_parameters(
                training, base_spec, base_search, config
            ).to_dict(),
        )
        base_parameters = StateSpaceParameters.from_dict(base_parameters_payload)
        training_scored = latent.score_frame(training, base_spec, base_parameters, None)
        testing_scored = latent.score_frame(testing, base_spec, base_parameters, None)
        for spec in CHALLENGERS:
            calibrator = ProbabilityCalibrator.from_dict(selected[spec.name]["calibrator"])
            alpha = float(selected[spec.name]["ridge_alpha"])
            value = store.get_or_compute(
                f"challenger-fold-{spec.name}-{fold.name}",
                latent._hash_payload(
                    {
                        "source": frame_manifest["sha256"],
                        "spec": asdict(spec),
                        "base_parameters": base_parameters.to_dict(),
                        "ridge_alpha": alpha,
                        "calibrator": calibrator.to_dict(),
                        "fold": asdict(fold),
                    }
                ),
                lambda training_scored=training_scored,
                testing_scored=testing_scored,
                spec=spec,
                alpha=alpha,
                calibrator=calibrator,
                fold=fold: _run_challenger_fold(
                    training_scored,
                    testing_scored,
                    spec,
                    alpha,
                    calibrator,
                    fold,
                    config,
                ),
            )
            scored = pl.DataFrame(value["scored"])
            ledger = pl.DataFrame(value["ledger"])
            if not scored.is_empty():
                scored_frames[spec.name].append(scored)
            if not ledger.is_empty():
                ledgers[spec.name].append(ledger)
            reports[spec.name].append(value["report"])
    scheduled = _development_markets(frame, config)
    final_ledgers: dict[str, pl.DataFrame] = {}
    final_scored: dict[str, pl.DataFrame] = {}
    for spec in CHALLENGERS:
        ledger = _concat(ledgers[spec.name])
        scored = _concat(scored_frames[spec.name])
        result = latent.trading_metrics(
            ledger,
            scored,
            scheduled_markets=scheduled,
            config=config,
            fold_reports=reports[spec.name],
        )
        result["selected_configuration"] = selected[spec.name]["configuration"]
        result["ridge_alpha"] = selected[spec.name]["ridge_alpha"]
        result["calibration_period"] = selected[spec.name]["calibration_metrics"]
        result["candidate_role"] = "binance_context_challenger"
        result["feature_family"] = spec.feature_family
        result["research_only"] = spec.research_only
        results[spec.name] = result
        final_ledgers[spec.name] = ledger
        final_scored[spec.name] = scored
    return results, final_ledgers, final_scored, reports


def _run_challenger_fold(
    training_scored: pl.DataFrame,
    testing_scored: pl.DataFrame,
    spec: ChallengerSpec,
    alpha: float,
    calibrator: ProbabilityCalibrator,
    fold: latent.Fold,
    config: TournamentConfig,
) -> dict[str, Any]:
    residual = fit_residual_adjustment(
        training_scored,
        candidate_name=spec.name,
        context_features=spec.features,
        ridge_alpha=alpha,
    )
    scored = apply_residual_adjustment(
        testing_scored,
        residual,
        context_features=spec.features,
        calibrator=calibrator,
    ).with_columns(pl.lit(fold.name).alias("fold"))
    ledger = latent.apply_shared_entry_controller(scored, calibrator, config)
    scheduled = testing_scored["market_id"].n_unique() if testing_scored.height else 0
    return {
        "residual_parameters": residual.to_dict(),
        "scored": scored.to_dict(as_series=False),
        "ledger": ledger.to_dict(as_series=False),
        "report": {
            "name": fold.name,
            "test_start": fold.test_start.isoformat(),
            "test_end": fold.test_end.isoformat(),
            "training_markets": training_scored["market_id"].n_unique(),
            "testing_markets": scheduled,
            "source_eligible_testing_markets": scored["market_id"].n_unique()
            if scored.height
            else 0,
            "predictive": latent.predictive_metrics(scored),
            "economic": latent._compact_economic_metrics(ledger, scheduled),
        },
    }


def _context_contribution_tests(
    results: dict[str, Any],
    ledgers: dict[str, pl.DataFrame],
    scored: dict[str, pl.DataFrame],
    config: TournamentConfig,
) -> dict[str, Any]:
    return {
        spec.name: _paired_context_contribution(
            spec, results, ledgers, scored, config
        )
        for spec in CHALLENGERS
    }


def _paired_context_contribution(
    spec: ChallengerSpec,
    results: dict[str, Any],
    ledgers: dict[str, pl.DataFrame],
    scored: dict[str, pl.DataFrame],
    config: TournamentConfig,
) -> dict[str, Any]:
    challenger = scored[spec.name]
    control = scored[spec.base_candidate]
    if challenger.is_empty():
        return {
            "candidate": spec.name,
            "matching_control": spec.base_candidate,
            "feature_family": spec.feature_family,
            "paired_markets": 0,
            "source_development_market_coverage": 0.0,
            "checks": {"source_evidence_available": False},
            "admitted": False,
            "research_only": spec.research_only,
        }
    paired = (
        challenger.select(
            "market_id", "window_start", "seconds_elapsed", "probability_up", "label_up"
        )
        .rename({"probability_up": "challenger_probability"})
        .join(
            control.select("market_id", "seconds_elapsed", "probability_up").rename(
                {"probability_up": "control_probability"}
            ),
            on=["market_id", "seconds_elapsed"],
            how="inner",
            validate="1:1",
        )
        .with_columns(
            (
                (pl.col("challenger_probability") - pl.col("label_up")) ** 2
                - (pl.col("control_probability") - pl.col("label_up")) ** 2
            ).alias("brier_difference")
        )
        .group_by("market_id")
        .agg(pl.col("window_start").first(), pl.col("brier_difference").mean())
    )
    brier = latent._paired_day_interval(
        paired,
        "brier_difference",
        int(config.raw["gates"]["bootstrap_resamples"]),
        config.random_seed + 101,
    )
    schedule = challenger.select("market_id", "window_start").unique(subset=["market_id"])
    economic = schedule
    for prefix, ledger in (
        ("challenger", ledgers[spec.name]),
        ("control", ledgers[spec.base_candidate]),
    ):
        values = ledger.select(
            "market_id", pl.col("stressed_pnl").alias(f"{prefix}_pnl")
        )
        economic = economic.join(values, on="market_id", how="left")
    economic = economic.with_columns(
        pl.col("challenger_pnl").fill_null(0.0),
        pl.col("control_pnl").fill_null(0.0),
    ).with_columns(
        (pl.col("challenger_pnl") - pl.col("control_pnl")).alias("pnl_difference")
    )
    expectancy = latent._paired_day_interval(
        economic,
        "pnl_difference",
        int(config.raw["gates"]["bootstrap_resamples"]),
        config.random_seed + 211,
    )
    fold_differences = {
        fold.name: float(
            economic.filter(
                pl.col("window_start").is_between(
                    fold.test_start, fold.test_end, closed="left"
                )
            )["pnl_difference"].sum()
            or 0.0
        )
        for fold in config.folds
    }
    scheduled = _development_markets(control, config)
    source_coverage = schedule.height / max(scheduled, 1)
    challenger_result = results[spec.name]
    control_result = results[spec.base_candidate]
    challenger_ece = challenger_result.get("predictive", {}).get(
        "expected_calibration_error"
    )
    control_ece = control_result.get("predictive", {}).get("expected_calibration_error")
    checks = {
        "source_evidence_available": paired.height > 0,
        "source_development_coverage": source_coverage
        >= float(config.raw["binance"]["minimum_source_development_market_coverage"]),
        "brier_upper_below_zero": math.isfinite(float(brier["upper"]))
        and float(brier["upper"]) < 0,
        "calibration_not_degraded": challenger_ece is not None
        and control_ece is not None
        and challenger_ece <= control_ece,
        "stressed_pnl_difference_lower_positive": math.isfinite(float(expectancy["lower"]))
        and float(expectancy["lower"]) > 0,
        "loss_recovery_not_worse": (
            challenger_result.get("wins_to_recover_average_loss", math.inf)
            <= control_result.get("wins_to_recover_average_loss", math.inf)
        ),
        "drawdown_not_worse": challenger_result.get("maximum_drawdown", math.inf)
        <= control_result.get("maximum_drawdown", math.inf),
        "multiple_positive_folds": sum(value > 0 for value in fold_differences.values()) >= 2,
        "deployment_eligible_source": not spec.research_only,
    }
    return {
        "candidate": spec.name,
        "matching_control": spec.base_candidate,
        "feature_family": spec.feature_family,
        "paired_markets": paired.height,
        "source_development_market_coverage": source_coverage,
        "brier_score_difference_challenger_minus_control": brier,
        "stressed_pnl_per_scheduled_market_difference": expectancy,
        "fold_differences": fold_differences,
        "checks": checks,
        "admitted": all(checks.values()),
        "research_only": spec.research_only,
    }


def _apply_context_qualification(
    results: dict[str, Any],
    refprice_tests: dict[str, Any],
    context_tests: dict[str, Any],
    comparator: dict[str, Any],
    development_evidence: dict[str, Any],
    config: TournamentConfig,
) -> dict[str, Any]:
    baseline = latent.apply_development_qualification(
        results, refprice_tests, comparator, development_evidence, config
    )
    isolated = {
        "kline": context_tests["twap_regime_kline_residual"]["admitted"],
        "open_interest": context_tests["twap_regime_open_interest_residual"]["admitted"],
    }
    qualified = []
    for spec in CHALLENGERS:
        row = results[spec.name]
        checks = row["qualification"]["checks"]
        checks["binance_context_admission"] = context_tests[spec.name]["admitted"]
        checks["not_research_only"] = not spec.research_only
        if spec.feature_family == "kline_open_interest":
            checks["isolated_sources_admitted"] = all(isolated.values())
        row["qualification"]["passed"] = all(checks.values())
        row["qualification"]["reasons"] = [
            name for name, passed in checks.items() if not passed
        ]
    for name, row in results.items():
        if row["qualification"]["passed"]:
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
    baseline.update(
        {
            "status": status,
            "selected_candidate": winner,
            "qualified_candidates": qualified,
            "binance_context_conclusion": (
                "No Binance context family supplied qualified incremental value."
                if not any(test["admitted"] for test in context_tests.values())
                else "At least one Binance context family passed incremental admission."
            ),
            "l2_conclusion": (
                "Binance L2 is research-only because its immutable history ends before "
                "the official-development interval."
            ),
        }
    )
    return baseline


def _fit_final_models(
    frame: pl.DataFrame,
    config: TournamentConfig,
    control_selected: dict[str, dict[str, Any]],
    challenger_selected: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    store: latent.CheckpointStore,
    frame_manifest: dict[str, Any],
) -> dict[str, Any]:
    frozen = frame.filter(pl.col("window_start") < config.freeze_at)
    names = (
        [selection["selected_candidate"]]
        if selection["selected_candidate"]
        else [candidate.name for candidate in CANDIDATES] + [spec.name for spec in CHALLENGERS]
    )
    models: dict[str, Any] = {}
    for name in names:
        if name in control_selected:
            candidate = _control(name)
            search = SearchConfiguration(**control_selected[name]["configuration"])
            parameters = store.get_or_compute(
                f"final-control-{name}",
                latent._hash_payload(
                    {
                        "source": frame_manifest["sha256"],
                        "candidate": asdict(candidate),
                        "configuration": asdict(search),
                        "freeze": config.freeze_at,
                    }
                ),
                lambda candidate=candidate, search=search: latent._fit_parameters(
                    frozen, candidate, search, config
                ).to_dict(),
            )
            models[name] = {
                "candidate_role": "frozen_latent_control",
                "candidate": asdict(candidate),
                "parameters": parameters,
                "probability_calibrator": control_selected[name]["calibrator"],
                "configuration": asdict(search),
                "fit_through_exclusive": config.freeze_at.isoformat(),
            }
            continue
        spec = next(spec for spec in CHALLENGERS if spec.name == name)
        base = _control(spec.base_candidate)
        base_search = SearchConfiguration(
            **control_selected[base.name]["configuration"]
        )
        base_parameters_payload = store.get_or_compute(
            f"final-challenger-base-{spec.name}",
            latent._hash_payload(
                {
                    "source": frame_manifest["sha256"],
                    "base": asdict(base),
                    "configuration": asdict(base_search),
                    "freeze": config.freeze_at,
                }
            ),
            lambda frozen=frozen,
            base=base,
            base_search=base_search: latent._fit_parameters(
                frozen, base, base_search, config
            ).to_dict(),
        )
        base_parameters = StateSpaceParameters.from_dict(base_parameters_payload)
        base_scored = latent.score_frame(frozen, base, base_parameters, None)
        alpha = float(challenger_selected[spec.name]["ridge_alpha"])
        residual_payload = store.get_or_compute(
            f"final-challenger-residual-{spec.name}",
            latent._hash_payload(
                {
                    "source": frame_manifest["sha256"],
                    "spec": asdict(spec),
                    "base_parameters": base_parameters.to_dict(),
                    "ridge_alpha": alpha,
                    "freeze": config.freeze_at,
                }
            ),
            lambda base_scored=base_scored,
            spec=spec,
            alpha=alpha: fit_residual_adjustment(
                base_scored,
                candidate_name=spec.name,
                context_features=spec.features,
                ridge_alpha=alpha,
            ).to_dict(),
        )
        models[name] = {
            "candidate_role": "binance_context_challenger",
            "candidate": asdict(spec),
            "base_state_parameters": base_parameters_payload,
            "residual_parameters": residual_payload,
            "probability_calibrator": challenger_selected[spec.name]["calibrator"],
            "configuration": challenger_selected[spec.name]["configuration"],
            "fit_through_exclusive": config.freeze_at.isoformat(),
        }
    return models


def render_report(metrics: dict[str, Any]) -> str:
    development = metrics["development"]
    selection = development["selection"]
    lines = [
        "# Binance-Context Latent TWAP Tournament",
        "",
        f"Run: `{metrics['run_id']}`  ",
        f"Source commit: `{metrics['source_commit']}`  ",
        f"Artifact SHA-256: `{metrics['model_artifact']['sha256']}`  ",
        f"Development result: **{selection['status']}**  ",
        f"Prospective result: **{metrics['prospective']['status']}**",
        "",
        "## Candidate results",
        "",
        "| Candidate | Role | Trades | Coverage | Accuracy | UP / DOWN | Stressed PnL | Expectancy | PF | Avg entry | Max DD | CVaR | Qualified |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in development["candidates"].items():
        up = row.get("per_direction", {}).get("UP", {}).get("trades", 0)
        down = row.get("per_direction", {}).get("DOWN", {}).get("trades", 0)
        lines.append(
            f"| {name} | {row.get('candidate_role', '—')} | {row.get('trades', 0)} | "
            f"{_pct(row.get('coverage'))} | {_pct(row.get('accuracy'))} | {up} / {down} | "
            f"{_num(row.get('stressed_pnl'))} | {_num(row.get('stressed_expectancy'))} | "
            f"{_num(row.get('profit_factor'))} | {_num(row.get('mean_entry_second'))} | "
            f"{_num(row.get('maximum_drawdown'))} | {_num(row.get('cvar_10'))} | "
            f"{row.get('qualification', {}).get('passed', False)} |"
        )
    lines.extend(["", "## Binance source coverage", ""])
    for family, periods in metrics["data"]["context_coverage"].items():
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| Evidence block | Markets | Scheduled | Coverage | Rows |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, row in periods.items():
            lines.append(
                f"| {name} | {row['markets']} | {row['scheduled_markets']} | "
                f"{_pct(row['market_coverage'])} | {row['rows']} |"
            )
        lines.append("")
    lines.extend(["## Detailed performance", ""])
    for name, row in development["candidates"].items():
        predictive = row.get("predictive", {})
        lines.extend(
            [
                f"### {name}",
                "",
                (
                    f"- Wins/losses: {row.get('wins', 0)} / {row.get('losses', 0)}; "
                    f"average win {_num(row.get('average_win'))}; average loss "
                    f"{_num(row.get('average_loss'))}; wins to recover one average loss "
                    f"{_num(row.get('wins_to_recover_average_loss'))}."
                ),
                (
                    f"- Worst loss {_num(row.get('worst_loss'))}; loss percentiles "
                    f"`{json.dumps(row.get('loss_percentiles', {}), sort_keys=True)}`."
                ),
                (
                    f"- Gross/net/stressed PnL: {_num(row.get('gross_pnl'))} / "
                    f"{_num(row.get('net_pnl'))} / {_num(row.get('stressed_pnl'))}."
                ),
                (
                    "- Entry seconds mean/median/p10/p90: "
                    f"{_num(row.get('mean_entry_second'))} / "
                    f"{_num(row.get('median_entry_second'))} / "
                    f"{_num(row.get('p10_entry_second'))} / "
                    f"{_num(row.get('p90_entry_second'))}."
                ),
                (
                    f"- Brier/log loss/ECE: {_num(predictive.get('brier_score'))} / "
                    f"{_num(predictive.get('log_loss'))} / "
                    f"{_num(predictive.get('expected_calibration_error'))}."
                ),
                (
                    "- Qualification failures: "
                    f"{', '.join(row.get('qualification', {}).get('reasons', [])) or 'none'}."
                ),
                (
                    "- Per-direction, entry-band, price-band, daily, fold, bootstrap, and "
                    "capacity results from 5 through 200 shares are retained in `metrics.json`."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Incremental Binance contribution",
            "",
            "```json",
            json.dumps(development["binance_context_contribution"], indent=2, sort_keys=True),
            "```",
            "",
            "## Evidence and provenance boundaries",
            "",
            f"- Split audit passed: {metrics['data']['split_separation']['passed']}.",
            f"- Context causality audit passed: {metrics['data']['context_causality']['passed']}.",
            "- August 14–27 is development evidence; August 28 onward remains untouched prospective evidence.",
            "- No runtime model was exported and no trading process was changed.",
            "- No data source, ingester, table, migration, or database row was created or changed.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in metrics["limitations"])
    return "\n".join(lines) + "\n"


def _limitations(
    frame: pl.DataFrame,
    selection: dict[str, Any],
    prospective: dict[str, Any],
    coverage: dict[str, Any],
    config: TournamentConfig,
) -> list[str]:
    limitations = [
        "Synthetic Chainlink-reconstructed TWAP evidence was used only for historical fitting and cannot qualify a model.",
        "Projected economics assume the recorded executable ask VWAP was fillable at each selected checkpoint.",
        "Binance L2 history ends before official development and is research-only; no missing L2 state was imputed.",
        "Open-interest history begins July 3, so the OI residual was fitted only on its causally eligible historical rows.",
        "No model was deployed and no trading process, ingester, SQL source, table, migration, or database row was changed.",
    ]
    if prospective["evidence"]["markets"] == 0:
        limitations.append(
            "The immutable extract ends at the August 28 freeze boundary, so no prospective market was read or scored."
        )
    evidence = latent._development_evidence(frame, config)
    if not evidence["complete"]:
        limitations.append(
            "Official development evidence is incomplete: "
            f"{evidence['observed_utc_days']} of {evidence['expected_complete_utc_days']} UTC "
            f"days and {evidence['observed_markets']} official markets were available."
        )
    if selection["selected_candidate"] is None:
        limitations.append("No development candidate passed every frozen qualification gate.")
    if coverage["l2"]["official_development"]["markets"]:
        raise RuntimeError("L2 unexpectedly acquired official-development coverage after freeze")
    return limitations


def _control(name: str) -> CandidateSpec:
    return next(candidate for candidate in CANDIDATES if candidate.name == name)


def _development_markets(frame: pl.DataFrame, config: TournamentConfig) -> int:
    return frame.filter(
        pl.col("window_start").is_between(
            config.development_start, config.development_end, closed="left"
        )
    )["market_id"].n_unique()


def _concat(frames: list[pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _git_revision(package_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=package_root, text=True
    ).strip()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=latent._json_default))
    os.replace(temporary, path)


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def _num(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.4f}" if math.isfinite(number) else "—"


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(prog="binance-latent-twap-context-tournament")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result, metrics = run_tournament(load_config(args.config))
    print(result)
    print(metrics["development"]["selection"]["status"])


if __name__ == "__main__":
    main()
