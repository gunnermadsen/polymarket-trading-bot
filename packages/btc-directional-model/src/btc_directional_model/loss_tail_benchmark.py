from __future__ import annotations

import html
import json
import math
import multiprocessing as mp
import os
import pickle
import resource
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .core_config import (
    CORE_ORACLE_SOURCE_CONTRACT,
    CoreComputeConfig,
    CoreTrainingConfig,
    load_core_config,
)
from .core_execution import (
    LEGACY_SNAPSHOT_SCHEMA_VERSION,
    ExecutionEvidenceConfig,
    extract_execution_evidence,
)
from .core_extract import file_sha256, write_json_atomic
from .core_features import (
    CORE_BOUNDARY_ENRICHED_FEATURES,
    build_core_features,
    feature_destination,
    validate_core_feature_cache,
)
from .core_training import (
    MARKET_EQUAL_ROW_WEIGHT_POLICY,
    CandidateSpec,
    FrozenTrainingBundle,
    fit_probability_calibrator,
    range_frame,
    tune_and_fit_model,
)
from .loss_tail_config import (
    BOUNDARY_RESIDUAL_ECONOMIC_HGB_CANDIDATE,
    LossTailBenchmarkConfig,
    LossTailFoldConfig,
    load_loss_tail_benchmark_config,
)
from .loss_tail_training import (
    fit_boundary_residual_economic_hgb,
    materialize_boundary_proposals,
    validate_loss_tail_feature_contract,
)
from .offline_challengers import derive_strict_book_feature_frame
from .oracle_book_benchmark import _execution_economics_frame, _load_execution_evidence
from .price_aware_benchmark import (
    _candidate_result,
    _cohort_lineage,
    _complete_market_ids,
    _direction_metrics,
    _key_fingerprint,
    _probability_calibration,
)
from .price_aware_training import (
    attach_five_share_economic_targets,
    mark_first_confident_action,
    score_probability_actions,
)
from .provenance import runtime_provenance

LOSS_TAIL_BENCHMARK_SCHEMA_VERSION = "btc-boundary-residual-economic-benchmark-v1"
MATCHED_BOUNDARY_CONTROL = "matched_boundary_alignment_control"

# These values were recorded from the audited direct core + oracle + strict PMXT
# cohort.  The cohort deliberately is not censored by continuous-context
# availability because those features are outside the frozen 124-feature
# contract.  Freezing the direct cohort makes silent source drift fail closed.
EXPECTED_CORE_FEATURE_SHA256 = "1fc8d07b1b3389c2c3ac8286a2ef78a389d0302b80b21c90384f7528093d256f"
EXPECTED_EXECUTION_MANIFEST_SHA256 = (
    "6bb946690c2ab5ab5ebe855556c2bcf941a5d92a9485f6814b2fb6f9c104ed1a"
)
EXPECTED_STRICT_KEY_SHA256 = "2e4a6481d822be3491f4d574ee0ec8d11ec7d5dd8139979a54dc581a4daf312e"
EXPECTED_TRAINING_SHA256 = "e4b7a796801f9a64f0fa998d2b14118858db58d719b0164ad6fc714cb89075c0"
EXPECTED_CACHE_CONFIG_SHA256 = "ac5b22ad510cf2a34392a18a9722f605ab217cb35a07719066d39d3b1ff04eb2"
EXPECTED_STRICT_ROWS = 264_121
EXPECTED_STRICT_MARKETS = 9_116
EXPECTED_PROPOSAL_ROWS = 5_396
EXPECTED_PROPOSAL_KEY_SHA256 = "d7cc08f28298583d66daeb9d07913466f4ec2ec72ff43e4b24dbde29de6fbc67"

_CACHE_KEYS = ("market_id", "window_start", "observed_at", "seconds_elapsed")
_PREDICTION_COLUMNS = (
    "market_id",
    "window_start",
    "observed_at",
    "seconds_elapsed",
    "walk_forward_block",
    "label_up",
    "binance_sign_up",
    "candidate",
    "benchmark_fold",
    "predicted_up",
    "outcome_predicted_up",
    "probability_up",
    "confidence",
    "decision_probability_correct",
    "correct",
    "outcome_probability_up",
    "economic_score",
    "direct_net_edge_per_share",
    "predicted_net_per_share",
    "selected_ask_vwap_5",
    "selected_fee_per_share",
    "realized_selected_net_per_share",
    "fee_rate",
    "up_ask_vwap_5",
    "down_ask_vwap_5",
    "up_ask_vwap_10",
    "down_ask_vwap_10",
    "strict_both_side_eligible",
    "strict_both_side_eligible_10",
    "execution_evidence_available",
    "policy_selected",
    "boundary_direction",
    "boundary_proposal_probability_up",
    "boundary_proposal_confidence",
    "boundary_direction_correct",
    "fee_inclusive_debit",
    "realized_trade_return",
    "no_trade_return",
)


def run_loss_tail_benchmark(
    config: LossTailBenchmarkConfig,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Benchmark one residual action head on frozen boundary proposals."""

    core_config = _validated_core_config(config)
    cache_path, cache_manifest = prepare_loss_tail_training_cache(
        config,
        core_config,
        force=force,
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    source_frame = pl.read_parquet(cache_path)
    print("loss-tail: fitting matched boundary control", flush=True)
    control_metadata = _run_one_worker(
        _train_control_worker,
        config.source_path,
        cache_path,
        run_dir,
        resources=config.resources,
    )
    control_decisions = pl.read_parquet(run_dir / "matched-boundary-control-decision-rows.parquet")
    proposals = materialize_boundary_proposals(
        source_frame,
        matched_evaluation=control_decisions,
    )
    proposal_path = run_dir / "boundary-proposals.parquet"
    _write_parquet_atomic(proposals, proposal_path)
    proposal_manifest = _proposal_cache_manifest(proposals, proposal_path)
    write_json_atomic(run_dir / "boundary-proposals-manifest.json", proposal_manifest)
    _write_matched_control_predictions(proposals, config, run_dir)

    print(
        "loss-tail: launching isolated residual HGB worker "
        f"({config.resources.threads_per_worker} threads, "
        f"{config.resources.memory_limit_gib} GiB each)",
        flush=True,
    )
    training: dict[str, Any] = {}
    previous_environment = _set_worker_thread_environment(config.resources.threads_per_worker)
    try:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=config.resources.workers,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(
                    _train_candidate_worker,
                    candidate,
                    config.source_path,
                    proposal_path,
                    run_dir,
                ): candidate
                for candidate in config.candidate_names
            }
            for future in as_completed(futures):
                candidate = futures[future]
                training[candidate] = future.result()
                print(f"loss-tail: completed {candidate}", flush=True)
    finally:
        _restore_environment(previous_environment)

    candidates = (MATCHED_BOUNDARY_CONTROL, *config.candidate_names)
    frames = {
        candidate: pl.read_parquet(run_dir / f"{candidate}-predictions.parquet")
        for candidate in candidates
    }
    _validate_matched_prediction_frames(frames)
    results = _evaluate_candidates(frames, source_frame, config, core_config)
    selection = _select_candidate(results, config)
    if selection["selected_candidate"] is None:
        promotion_artifacts: dict[str, Any] = {
            "exported": False,
            "reason": "candidate did not pass every promotion requirement",
        }
    else:
        promotion_artifacts = _run_one_worker(
            _export_promoted_candidate_worker,
            config.source_path,
            proposal_path,
            run_dir,
            resources=config.resources,
        )
    payload: dict[str, Any] = {
        "schema_version": LOSS_TAIL_BENCHMARK_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "complete",
        "objective": (
            "preserve boundary-alignment decision accuracy while teaching the model "
            "to avoid fee-inclusive high-debit wrong-direction loss tails"
        ),
        "evaluation_note": config.evaluation_note,
        "data_contract": {
            "range": {
                "history_start": config.walk_forward.history_start.isoformat(),
                "book_start": config.walk_forward.blocks[0].start.isoformat(),
                "end_exclusive": config.walk_forward.blocks[-1].end.isoformat(),
            },
            "decision_seconds": list(config.data.decision_seconds),
            "context_seconds": list(config.data.context_seconds),
            "quantity": config.data.quantity,
            "book_freshness_seconds": config.data.book_freshness_seconds,
            "strict_point_qualified_strategy": True,
            "complete_paths": "diagnostic_only",
            "book_missingness": "unavailable; never imputed and never a feature",
        },
        "walk_forward": {
            "blocks": [
                {
                    "name": block.name,
                    "start": block.start.isoformat(),
                    "end": block.end.isoformat(),
                }
                for block in config.walk_forward.blocks
            ],
            "folds": [asdict(fold) for fold in config.walk_forward.folds],
        },
        "resource_contract": {
            **asdict(config.resources),
            "worker_process_isolation": True,
            "native_thread_limits": True,
            "host_logical_cpu_headroom": max(
                0,
                (os.cpu_count() or 0)
                - config.resources.workers * config.resources.threads_per_worker,
            ),
            "cpu_affinity": "not available on this macOS host",
            "memory_enforcement": (
                "per-worker address-space limit when supported; otherwise a fail-hard "
                "OS maximum-RSS watchdog"
            ),
        },
        "cache": cache_manifest,
        "proposal_cache": proposal_manifest,
        "training": {
            MATCHED_BOUNDARY_CONTROL: control_metadata,
            **training,
        },
        "candidates": results,
        "selection": selection,
        "promotion_artifacts": promotion_artifacts,
        "runtime_provenance": runtime_provenance(config.package_root),
        "input_provenance": {
            "benchmark_config": str(config.source_path),
            "benchmark_config_sha256": file_sha256(config.source_path),
            "core_config": str(config.core_config),
            "core_config_sha256": file_sha256(config.core_config),
            "shared_cache": str(cache_path),
            "shared_cache_sha256": file_sha256(cache_path),
            "proposal_cache": str(proposal_path),
            "proposal_cache_sha256": file_sha256(proposal_path),
        },
        "deployment": {
            "runtime_exported": False,
            "paper_process_created": False,
            "live_capital_authorized": False,
            "reason": "benchmark-only training; runtime support is intentionally out of scope",
        },
    }
    write_json_atomic(run_dir / "benchmark.json", payload)
    _write_text_atomic(run_dir / "report.html", _render_report(payload))
    return run_dir, payload


def prepare_loss_tail_training_cache(
    config: LossTailBenchmarkConfig,
    core_config: CoreTrainingConfig,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Materialize one strict 124-feature cache and attach causal OOF signals."""

    from .loss_tail_oof import build_causal_oof_signals

    cache_dir = config.paths.shared_cache
    raw_path = cache_dir / "strict-loss-tail-rows.parquet"
    oof_path = cache_dir / "causal-source-oof.parquet"
    training_path = cache_dir / "training-rows.parquet"
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text())
        _validate_cache_manifest(manifest, training_path, config)
        print(f"loss-tail: reuse immutable cache {cache_dir}", flush=True)
        return training_path, manifest
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise RuntimeError(
            "loss-tail cache is partial or force-rebuild was requested; use a new isolated "
            "cache path instead of overwriting training evidence"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("loss-tail: building/reusing core + Chainlink features", flush=True)
    build_core_features(core_config, "pre_holdout", force=False)
    core_metadata = validate_core_feature_cache(core_config, "pre_holdout")
    core_path = feature_destination(core_config, "pre_holdout")
    if core_path.resolve() != config.paths.core_features.resolve():
        raise RuntimeError("loss-tail core feature path does not match the pinned core config")
    core_sha256 = file_sha256(core_path)
    if core_sha256 != EXPECTED_CORE_FEATURE_SHA256:
        raise RuntimeError(
            "regenerated core feature cache does not match the previously audited artifact"
        )

    print("loss-tail: building/reusing 55-240 strict execution evidence", flush=True)
    execution_config = _execution_config(config, core_config)
    execution_manifest = extract_execution_evidence(execution_config, force=False)
    execution_manifest_path = execution_config.output_dir / "manifest.json"
    execution_manifest_sha256 = file_sha256(execution_manifest_path)
    if execution_manifest_sha256 != EXPECTED_EXECUTION_MANIFEST_SHA256:
        raise RuntimeError(
            "regenerated execution evidence does not match the previously audited artifact"
        )

    core = pl.read_parquet(core_path).sort(["window_start", "market_id", "seconds_elapsed"])
    execution = _load_execution_evidence(execution_config, execution_manifest)
    strict = derive_strict_book_feature_frame(
        core,
        execution,
        require_ten_share=False,
        include_deltas=True,
    ).filter(
        pl.col("seconds_elapsed").is_in(config.data.decision_seconds)
        & (pl.col("window_start") >= config.walk_forward.blocks[0].start)
        & (pl.col("window_start") < config.walk_forward.blocks[-1].end)
    )
    economics = _execution_economics_frame(execution)
    strict = attach_five_share_economic_targets(
        strict.join(
            economics,
            on=["market_id", "observed_at"],
            how="inner",
            validate="1:1",
        )
    ).sort(["window_start", "market_id", "seconds_elapsed"])
    validate_loss_tail_feature_contract(strict)
    _validate_strict_cache(strict)
    _write_parquet_atomic(strict, raw_path)

    print("loss-tail: generating block-causal 58/68/71 OOF signals", flush=True)
    oof_metadata = build_causal_oof_signals(
        core_path=core_path,
        strict_path=raw_path,
        output_path=oof_path,
        config=config,
        core_config=core_config,
    )
    oof = pl.read_parquet(oof_path)
    training = strict.join(
        oof,
        on=list(_CACHE_KEYS),
        how="inner",
        validate="1:1",
    )
    if training.height != strict.height:
        raise RuntimeError("causal OOF signals do not cover every strict training row")
    _write_parquet_atomic(training, training_path)
    manifest = {
        "schema_version": "btc-boundary-loss-tail-cache-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_config_sha256": file_sha256(config.source_path),
        "core_feature_path": str(core_path),
        "core_feature_sha256": core_sha256,
        "core_feature_metadata": core_metadata,
        "execution_manifest_path": str(execution_manifest_path),
        "execution_manifest_sha256": execution_manifest_sha256,
        "execution_totals": execution_manifest["totals"],
        "raw_strict_path": str(raw_path),
        "raw_strict_sha256": file_sha256(raw_path),
        "training_path": str(training_path),
        "training_sha256": file_sha256(training_path),
        "strict_rows": training.height,
        "strict_markets": training["market_id"].n_unique(),
        "strict_key_sha256": _key_fingerprint(training),
        "direct_feature_count": len(validate_loss_tail_feature_contract()),
        "oof": oof_metadata,
    }
    write_json_atomic(manifest_path, manifest)
    return training_path, manifest


def _train_control_worker(
    config_path: Path,
    cache_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    config = load_loss_tail_benchmark_config(config_path)
    resources = _apply_worker_resources(config)
    core_config = _worker_core_config(config)
    frame = pl.read_parquet(cache_path)
    outputs: list[pl.DataFrame] = []
    folds: dict[str, Any] = {}
    spec = CandidateSpec(
        MATCHED_BOUNDARY_CONTROL,
        "histogram",
        tuple(CORE_BOUNDARY_ENRICHED_FEATURES),
        MARKET_EQUAL_ROW_WEIGHT_POLICY,
    )
    started = time.perf_counter()
    for fold in config.walk_forward.folds:
        fit = frame.filter(pl.col("walk_forward_block").is_in(fold.fit_block_names)).sort(
            ["window_start", "market_id", "seconds_elapsed"]
        )
        calibration = frame.filter(
            pl.col("walk_forward_block") == fold.calibration_block_name
        ).sort(["window_start", "market_id", "seconds_elapsed"])
        evaluation = frame.filter(pl.col("walk_forward_block") == fold.evaluation_block_name).sort(
            ["window_start", "market_id", "seconds_elapsed"]
        )
        model, tuning = tune_and_fit_model(fit, spec, core_config)
        calibrator = fit_probability_calibrator(model, calibration, core_config, spec)
        bundle = FrozenTrainingBundle(model, calibrator, 0.89)
        scored = _score_direct(
            evaluation,
            bundle.probability(evaluation),
            candidate=MATCHED_BOUNDARY_CONTROL,
            fold_name=fold.name,
            confidence_threshold=0.89,
        )
        outputs.append(
            scored.select(
                *_CACHE_KEYS,
                "walk_forward_block",
                "label_up",
                "predicted_up",
                "probability_up",
                "policy_selected",
                "benchmark_fold",
            )
        )
        folds[fold.name] = {
            "fit": _cohort_lineage(fit),
            "calibration": _cohort_lineage(calibration),
            "evaluation": _cohort_lineage(evaluation),
            "tuning": tuning,
            "calibrator": asdict(calibrator),
            "proposal_policy": "first boundary decision at frozen 0.89 confidence",
        }
    output = pl.concat(outputs, how="vertical")
    _write_parquet_atomic(output, run_dir / "matched-boundary-control-decision-rows.parquet")
    return {
        "candidate": MATCHED_BOUNDARY_CONTROL,
        "worker": resources,
        "elapsed_seconds": time.perf_counter() - started,
        "feature_count": len(CORE_BOUNDARY_ENRICHED_FEATURES),
        "features": list(CORE_BOUNDARY_ENRICHED_FEATURES),
        "folds": folds,
    }


def _write_matched_control_predictions(
    proposals: pl.DataFrame,
    config: LossTailBenchmarkConfig,
    run_dir: Path,
) -> None:
    outputs: list[pl.DataFrame] = []
    for fold in config.walk_forward.folds:
        evaluation = proposals.filter(pl.col("walk_forward_block") == fold.evaluation_block_name)
        scored = _score_boundary_proposals(
            evaluation,
            evaluation["boundary_proposal_confidence"].to_numpy(),
            candidate=MATCHED_BOUNDARY_CONTROL,
            fold_name=fold.name,
            economic_action=False,
        )
        outputs.append(_prediction_artifact_frame(scored))
    _write_parquet_atomic(
        pl.concat(outputs, how="vertical"),
        run_dir / f"{MATCHED_BOUNDARY_CONTROL}-predictions.parquet",
    )


def _train_candidate_worker(
    candidate: str,
    config_path: Path,
    cache_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    from .loss_tail_oof import BOUNDARY_CORRECTNESS_FEATURES

    config = load_loss_tail_benchmark_config(config_path)
    if candidate not in config.candidate_names:
        raise ValueError(f"unsupported loss-tail candidate: {candidate}")
    resources = _apply_worker_resources(config)
    core_config = _worker_core_config(config)
    frame = pl.read_parquet(cache_path)
    outputs: list[pl.DataFrame] = []
    folds: dict[str, Any] = {}
    started = time.perf_counter()
    for fold in config.walk_forward.folds:
        fit, calibration, evaluation = _fold_frames(frame, fold)
        if candidate != BOUNDARY_RESIDUAL_ECONOMIC_HGB_CANDIDATE:
            raise ValueError(f"unsupported residual candidate: {candidate}")
        bundle = fit_boundary_residual_economic_hgb(
            fit,
            calibration,
            feature_names=BOUNDARY_CORRECTNESS_FEATURES,
            core_config=core_config,
            candidate_name=candidate,
        )
        scored = _score_boundary_proposals(
            evaluation,
            bundle.probability(evaluation),
            candidate=candidate,
            fold_name=fold.name,
            economic_action=True,
        )
        outputs.append(_prediction_artifact_frame(scored))
        folds[fold.name] = {
            "fit": _cohort_lineage(fit),
            "calibration": _cohort_lineage(calibration),
            "evaluation": _cohort_lineage(evaluation),
            "tuning": bundle.tuning,
            "calibrator": asdict(bundle.calibrator),
        }
    output = pl.concat(outputs, how="vertical")
    _write_parquet_atomic(output, run_dir / f"{candidate}-predictions.parquet")
    return {
        "candidate": candidate,
        "worker": resources,
        "elapsed_seconds": time.perf_counter() - started,
        "feature_count": len(bundle.feature_names),
        "features": list(bundle.feature_names),
        "folds": folds,
    }


def _export_promoted_candidate_worker(
    config_path: Path,
    cache_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Export the confirmation-fold model only after every benchmark gate passes."""

    from .loss_tail_oof import BOUNDARY_CORRECTNESS_FEATURES

    config = load_loss_tail_benchmark_config(config_path)
    resources = _apply_worker_resources(config)
    core_config = _worker_core_config(config)
    frame = pl.read_parquet(cache_path)
    fold = config.walk_forward.folds[-1]
    fit, calibration, _evaluation = _fold_frames(frame, fold)
    bundle = fit_boundary_residual_economic_hgb(
        fit,
        calibration,
        feature_names=BOUNDARY_CORRECTNESS_FEATURES,
        core_config=core_config,
        candidate_name=BOUNDARY_RESIDUAL_ECONOMIC_HGB_CANDIDATE,
    )
    model_path = run_dir / "promoted-boundary-residual-model.pkl"
    temporary_model_path = model_path.with_suffix(model_path.suffix + ".partial")
    with temporary_model_path.open("wb") as handle:
        pickle.dump(bundle.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_model_path.replace(model_path)
    calibrator_path = run_dir / "promoted-boundary-residual-calibrator.json"
    write_json_atomic(calibrator_path, asdict(bundle.calibrator))
    return {
        "exported": True,
        "runtime_exported": False,
        "candidate": bundle.candidate_name,
        "model_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "calibrator_path": str(calibrator_path),
        "calibrator_sha256": file_sha256(calibrator_path),
        "feature_count": len(bundle.feature_names),
        "features": list(bundle.feature_names),
        "target": bundle.target_column,
        "fit": _cohort_lineage(fit),
        "calibration": _cohort_lineage(calibration),
        "tuning": bundle.tuning,
        "worker": resources,
        "note": "offline benchmark artifact; no runtime integration or process modification",
    }


def _score_direct(
    frame: pl.DataFrame,
    probability_up: np.ndarray,
    *,
    candidate: str,
    fold_name: str,
    confidence_threshold: float,
) -> pl.DataFrame:
    scored = score_probability_actions(
        frame,
        probability_up,
        candidate=candidate,
        select_by_value=False,
    ).with_columns(
        pl.max_horizontal("probability_up", 1.0 - pl.col("probability_up")).alias(
            "decision_probability_correct"
        ),
        pl.lit(fold_name).alias("benchmark_fold"),
    )
    return mark_first_confident_action(
        scored,
        confidence_threshold=confidence_threshold,
    )


def _score_boundary_proposals(
    frame: pl.DataFrame,
    probability_correct: np.ndarray,
    *,
    candidate: str,
    fold_name: str,
    economic_action: bool,
) -> pl.DataFrame:
    boundary_probability = frame["boundary_proposal_probability_up"].to_numpy()
    scored = score_probability_actions(
        frame,
        boundary_probability,
        candidate=candidate,
        select_by_value=False,
    )
    selected_debit = np.where(
        boundary_probability >= 0.5,
        frame["up_entry_debit_per_share"].to_numpy(),
        frame["down_entry_debit_per_share"].to_numpy(),
    )
    scored = scored.with_columns(
        pl.Series("decision_probability_correct", probability_correct),
        pl.Series("confidence", probability_correct),
        pl.Series("economic_score", probability_correct - selected_debit),
        pl.Series("predicted_net_per_share", probability_correct - selected_debit),
        pl.lit(fold_name).alias("benchmark_fold"),
    )
    return scored.with_columns(
        (pl.col("economic_score") > 0.0 if economic_action else pl.lit(True)).alias(
            "policy_selected"
        )
    )


def _fold_frames(
    frame: pl.DataFrame,
    fold: LossTailFoldConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    fit = frame.filter(pl.col("walk_forward_block").is_in(fold.fit_block_names)).sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    calibration = frame.filter(pl.col("walk_forward_block") == fold.calibration_block_name).sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    evaluation = frame.filter(pl.col("walk_forward_block") == fold.evaluation_block_name).sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    for name, cohort in (
        ("fit", fit),
        ("calibration", calibration),
        ("evaluation", evaluation),
    ):
        if (
            cohort.is_empty()
            or cohort["label_up"].n_unique() != 2
            or cohort["boundary_direction_correct"].n_unique() != 2
        ):
            raise RuntimeError(f"{fold.name} {name} cohort is empty or single-class")
        if cohort["market_id"].n_unique() != cohort.height:
            raise RuntimeError(f"{fold.name} {name} cohort is not one proposal per market")
    if not (
        fit["window_start"].max() < calibration["window_start"].min()
        and calibration["window_start"].max() < evaluation["window_start"].min()
    ):
        raise RuntimeError(f"{fold.name} fit/calibration/evaluation chronology leaked")
    return fit, calibration, evaluation


def _evaluate_candidates(
    frames: dict[str, pl.DataFrame],
    source_frame: pl.DataFrame,
    config: LossTailBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    evaluation_blocks = {fold.evaluation_block_name for fold in config.walk_forward.folds}
    core = pl.read_parquet(config.paths.core_features).filter(
        pl.col("window_start").is_between(
            min(
                block.start
                for block in config.walk_forward.blocks
                if block.name in evaluation_blocks
            ),
            max(
                block.end for block in config.walk_forward.blocks if block.name in evaluation_blocks
            ),
            closed="left",
        )
    )
    all_core_markets = core["market_id"].n_unique()
    evaluation_source = source_frame.filter(pl.col("walk_forward_block").is_in(evaluation_blocks))
    book_qualified_markets = evaluation_source["market_id"].n_unique()
    complete_market_ids = _complete_market_ids(evaluation_source)
    core_markets_by_fold = {
        fold.name: range_frame(
            core,
            next(
                block.start
                for block in config.walk_forward.blocks
                if block.name == fold.evaluation_block_name
            ),
            next(
                block.end
                for block in config.walk_forward.blocks
                if block.name == fold.evaluation_block_name
            ),
        )["market_id"].n_unique()
        for fold in config.walk_forward.folds
    }
    output: dict[str, Any] = {}
    for candidate, frame in frames.items():
        result = _candidate_result(
            frame,
            book_qualified_markets=book_qualified_markets,
            all_core_markets=all_core_markets,
            complete_market_ids=complete_market_ids,
        )
        result["economics"]["net_pnl_per_boundary_proposal"] = (
            result["economics"]["execution"]["realized_net_pnl_total"] / frame.height
            if frame.height
            else 0.0
        )
        result["same_row_direction"] = _direction_metrics(frame)
        result["decision_calibration"] = _probability_calibration(
            frame,
            probability_column="decision_probability_correct",
            target_column="correct",
        )
        result["loss_ratios"] = _loss_ratios(result["economics"])
        result["proposal_price_bands"] = _proposal_price_band_metrics(frame)
        result["folds"] = {}
        for fold in config.walk_forward.folds:
            rows = frame.filter(pl.col("benchmark_fold") == fold.name)
            source_rows = source_frame.filter(
                pl.col("walk_forward_block") == fold.evaluation_block_name
            )
            fold_result = _candidate_result(
                rows,
                book_qualified_markets=source_rows["market_id"].n_unique(),
                all_core_markets=core_markets_by_fold[fold.name],
                complete_market_ids=_complete_market_ids(source_rows),
            )
            fold_result["economics"]["net_pnl_per_boundary_proposal"] = (
                fold_result["economics"]["execution"]["realized_net_pnl_total"] / rows.height
                if rows.height
                else 0.0
            )
            fold_result["same_row_direction"] = _direction_metrics(rows)
            fold_result["decision_calibration"] = _probability_calibration(
                rows,
                probability_column="decision_probability_correct",
                target_column="correct",
            )
            fold_result["loss_ratios"] = _loss_ratios(fold_result["economics"])
            fold_result["proposal_price_bands"] = _proposal_price_band_metrics(rows)
            result["folds"][fold.name] = fold_result
        output[candidate] = result
    control_frame = frames[MATCHED_BOUNDARY_CONTROL]
    control_result = output[MATCHED_BOUNDARY_CONTROL]
    for candidate, frame in frames.items():
        output[candidate]["proposal_outcomes"] = _proposal_outcome_metrics(
            frame,
            control_frame,
            output[candidate],
            control_result,
            high_debit_threshold=config.gates.high_debit_threshold,
        )
    return output


def _proposal_outcome_metrics(
    candidate: pl.DataFrame,
    control: pl.DataFrame,
    candidate_result: dict[str, Any],
    control_result: dict[str, Any],
    *,
    high_debit_threshold: float,
) -> dict[str, Any]:
    """Measure abstention outcomes on the exact predecessor proposal denominator."""

    ordered_candidate = candidate.sort(["window_start", "market_id", "seconds_elapsed"])
    ordered_control = control.sort(["window_start", "market_id", "seconds_elapsed"])
    if not ordered_candidate.select(*_CACHE_KEYS).equals(ordered_control.select(*_CACHE_KEYS)):
        raise RuntimeError("proposal outcome metrics require exact matched proposal keys")
    selected = ordered_candidate["policy_selected"].to_numpy().astype(bool)
    correct = ordered_control["correct"].to_numpy().astype(bool)
    debit = ordered_control["fee_inclusive_debit"].to_numpy()
    rejected = ~selected
    high_debit_losses = (~correct) & (debit >= high_debit_threshold)
    candidate_execution = candidate_result["economics"]["execution"]
    control_execution = control_result["economics"]["execution"]
    candidate_gross_profit = candidate_execution["gross_profit"] or 0.0
    control_gross_profit = control_execution["gross_profit"] or 0.0
    candidate_gross_loss = candidate_execution["gross_loss"] or 0.0
    control_gross_loss = control_execution["gross_loss"] or 0.0
    return {
        "boundary_proposals": len(selected),
        "trades": int(np.sum(selected)),
        "no_trades": int(np.sum(rejected)),
        "no_trade_precision": (float(np.mean(~correct[rejected])) if np.any(rejected) else None),
        "winner_retention": (
            float(np.sum(selected & correct) / np.sum(correct)) if np.any(correct) else None
        ),
        "false_abstention_rate": (
            float(np.sum(rejected & correct) / np.sum(correct)) if np.any(correct) else None
        ),
        "incorrect_proposals_converted_to_no_trade": int(np.sum(rejected & ~correct)),
        "high_debit_threshold": high_debit_threshold,
        "high_debit_incorrect_proposals": int(np.sum(high_debit_losses)),
        "high_debit_incorrect_converted_to_no_trade": int(np.sum(rejected & high_debit_losses)),
        "catastrophic_loss_recall": (
            float(np.sum(rejected & high_debit_losses) / np.sum(high_debit_losses))
            if np.any(high_debit_losses)
            else None
        ),
        "gross_profit_retained": (
            candidate_gross_profit / control_gross_profit if control_gross_profit > 0.0 else None
        ),
        "gross_loss_removed": (
            1.0 - candidate_gross_loss / control_gross_loss if control_gross_loss > 0.0 else None
        ),
    }


def _proposal_price_band_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    from .price_aware_config import PRICE_AWARE_PRICE_BANDS

    output: dict[str, Any] = {}
    selected_metrics = _candidate_result(
        frame,
        book_qualified_markets=frame.height,
        all_core_markets=frame.height,
        complete_market_ids=[],
    )["price_bands_diagnostic_only"]
    for name, lower, upper in PRICE_AWARE_PRICE_BANDS:
        rows = frame.filter(pl.col("selected_ask_vwap_5").is_between(lower, upper, closed="left"))
        rejected_incorrect = rows.filter((~pl.col("policy_selected")) & (~pl.col("correct"))).height
        output[name] = {
            **selected_metrics[name],
            "boundary_proposals": rows.height,
            "incorrect_trades_converted_to_no_trade": rejected_incorrect,
            "probability_calibration": _probability_calibration(
                rows,
                probability_column="decision_probability_correct",
                target_column="correct",
            ),
        }
    return output


def _select_candidate(
    results: dict[str, Any],
    config: LossTailBenchmarkConfig,
) -> dict[str, Any]:
    control = results[MATCHED_BOUNDARY_CONTROL]
    checks_by_candidate: dict[str, Any] = {}
    qualified: list[str] = []
    for candidate in config.candidate_names:
        checks = _qualification_checks(results[candidate], control, config)
        passed = all(check["passed"] for check in checks)
        checks_by_candidate[candidate] = {"passed": passed, "checks": checks}
        if passed:
            qualified.append(candidate)
    selected: str | None = None
    if qualified:
        selected = max(
            qualified,
            key=lambda candidate: (
                min(
                    results[candidate]["folds"][fold.name]["economics"][
                        "net_pnl_per_all_core_market"
                    ]
                    for fold in config.walk_forward.folds
                ),
                results[candidate]["economics"]["net_pnl_per_all_core_market"],
                _none_to_infinity(results[candidate]["economics"]["execution"]["profit_factor"]),
                results[candidate]["economics"]["book_qualified_coverage"],
            ),
        )
    return {
        "selected_candidate": selected,
        "qualified_candidates": qualified,
        "fallback_allowed": config.gates.allow_fallback_winner,
        "ranking": (
            "best worst-fold net PnL per all-core market, then pooled PnL, "
            "profit factor, and coverage"
        ),
        "candidates": checks_by_candidate,
    }


def _qualification_checks(
    candidate: dict[str, Any],
    control: dict[str, Any],
    config: LossTailBenchmarkConfig,
) -> list[dict[str, Any]]:
    gates = config.gates
    economics = candidate["economics"]
    control_economics = control["economics"]
    loss = candidate["loss_ratios"]
    proposal_outcomes = candidate["proposal_outcomes"]
    execution = economics["execution"]
    confirmation = candidate["folds"][config.walk_forward.folds[-1].name]
    fold_economics = [
        candidate["folds"][fold.name]["economics"] for fold in config.walk_forward.folds
    ]
    nonnegative_folds = sum(
        row["execution"]["realized_net_expectancy_per_trade"] is not None
        and row["execution"]["realized_net_expectancy_per_trade"] >= 0.0
        for row in fold_economics
    )
    improved_loss_folds = sum(
        _ratio_at_most(
            row_loss["gross_loss_profit"],
            control_row_loss["gross_loss_profit"],
            1.0,
        )
        for row_loss, control_row_loss in zip(
            (candidate["folds"][fold.name]["loss_ratios"] for fold in config.walk_forward.folds),
            (control["folds"][fold.name]["loss_ratios"] for fold in config.walk_forward.folds),
            strict=True,
        )
    )
    fold_direction_ok = all(
        min(
            candidate["folds"][fold.name]["per_side"]["up"]["trades"],
            candidate["folds"][fold.name]["per_side"]["down"]["trades"],
        )
        >= gates.minimum_fold_direction_trades
        for fold in config.walk_forward.folds
    )
    checks = [
        _check("selected accuracy", economics["accuracy"], ">=", gates.minimum_selected_accuracy),
        _check(
            "selected Wilson lower",
            economics["accuracy_wilson_lower_95"],
            ">=",
            gates.minimum_wilson_lower,
        ),
        _check(
            "control accuracy noninferiority",
            economics["accuracy"],
            ">=",
            control_economics["accuracy"] - gates.maximum_control_accuracy_regression,
        ),
        _check(
            "decision ECE",
            candidate["decision_calibration"]["expected_calibration_error"],
            "<=",
            gates.maximum_expected_calibration_error,
        ),
        _check(
            "mean-loss/win ratio improvement",
            loss["mean_loss_win"],
            "<=",
            gates.maximum_wins_per_average_loss,
        ),
        _check(
            "gross-loss/profit ratio improvement",
            loss["gross_loss_profit"],
            "<=",
            gates.maximum_gross_loss_profit_ratio,
        ),
        _check(
            "minimum profit factor",
            _none_to_infinity(execution["profit_factor"]),
            ">",
            gates.minimum_profit_factor,
        ),
        _check(
            "minimum PnL per all-core market",
            economics["net_pnl_per_all_core_market"],
            ">=",
            gates.minimum_pnl_per_all_core_market,
        ),
        _check(
            "minimum total five-share PnL",
            execution["realized_net_pnl_total"],
            ">=",
            gates.minimum_total_pnl,
        ),
        _check(
            "maximum mean selected price",
            economics["mean_selected_ask_vwap_5"],
            "<=",
            gates.maximum_mean_selected_price,
        ),
        _check(
            "worst trade",
            execution["worst_realized_net_pnl"],
            ">=",
            gates.minimum_worst_trade,
        ),
        _check(
            "mean worst one percent",
            execution["mean_worst_one_percent_realized_net_pnl"],
            ">=",
            gates.minimum_mean_worst_one_percent,
        ),
        _check(
            "maximum drawdown",
            execution["maximum_drawdown"],
            "<=",
            gates.maximum_drawdown,
        ),
        _check(
            "minimum book-qualified coverage",
            economics["book_qualified_coverage"],
            ">=",
            gates.minimum_book_qualified_coverage,
        ),
        _check("pooled trades", economics["trades"], ">=", gates.minimum_pooled_trades),
        _check(
            "confirmation trades",
            confirmation["economics"]["trades"],
            ">=",
            gates.minimum_confirmation_trades,
        ),
        _check(
            "both directions represented in every fold",
            int(fold_direction_ok),
            ">=",
            1,
        ),
        _check(
            "nonnegative expectancy folds",
            nonnegative_folds,
            ">=",
            gates.required_nonnegative_expectancy_folds,
        ),
        _check(
            "loss-improvement folds",
            improved_loss_folds,
            ">=",
            gates.minimum_loss_improvement_folds,
        ),
        _check(
            "gross loss removed",
            proposal_outcomes["gross_loss_removed"],
            ">=",
            gates.minimum_gross_loss_reduction,
        ),
        _check(
            "gross profit retained",
            proposal_outcomes["gross_profit_retained"],
            ">=",
            gates.minimum_gross_profit_retention,
        ),
        _check(
            "high-debit catastrophic-loss recall",
            proposal_outcomes["catastrophic_loss_recall"],
            ">=",
            gates.minimum_high_debit_loss_recall,
        ),
        _check(
            "90c-and-over band net positive",
            candidate["proposal_price_bands"]["0_90_and_over"]["realized_net_pnl_total"],
            ">",
            0.0,
        ),
    ]
    return checks


def _loss_ratios(economics: dict[str, Any]) -> dict[str, float | None]:
    execution = economics["execution"]
    gross_profit = execution["gross_profit"]
    gross_loss = execution["gross_loss"]
    return {
        "mean_loss_win": economics["wins_to_recover_average_loss"],
        "gross_loss_profit": (
            gross_loss / gross_profit
            if gross_profit is not None and gross_profit > 0.0 and gross_loss is not None
            else None
        ),
    }


def _ratio_at_most(observed: float | None, control: float | None, scale: float) -> bool:
    required = _scaled_ratio(control, scale)
    return observed is not None and required is not None and observed <= required


def _scaled_ratio(value: float | None, scale: float) -> float | None:
    return None if value is None else value * scale


def _check(name: str, observed: Any, operator: str, required: Any) -> dict[str, Any]:
    if observed is None or required is None:
        passed = False
    elif operator == ">=":
        passed = observed >= required
    elif operator == ">":
        passed = observed > required
    elif operator == "<=":
        passed = observed <= required
    else:
        raise ValueError(f"unsupported qualification operator: {operator}")
    return {
        "name": name,
        "observed": _json_safe_metric(observed),
        "operator": operator,
        "required": _json_safe_metric(required),
        "passed": bool(passed),
    }


def _json_safe_metric(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0 else "negative_infinity"
    return value


def _worker_core_config(config: LossTailBenchmarkConfig) -> CoreTrainingConfig:
    core = _validated_core_config(config)
    return replace(
        core,
        compute=CoreComputeConfig(
            max_parallel_fits=1,
            threads_per_fit=config.resources.threads_per_worker,
            polars_threads=config.resources.threads_per_worker,
        ),
    )


def _validated_core_config(config: LossTailBenchmarkConfig) -> CoreTrainingConfig:
    core = load_core_config(config.core_config)
    if core.data.source_contract != CORE_ORACLE_SOURCE_CONTRACT:
        raise RuntimeError("loss-tail training requires the causal Chainlink core source")
    if (
        core.data.range_start != config.walk_forward.history_start
        or core.data.range_end != config.walk_forward.blocks[-1].end
        or core.data.min_seconds_after_open != config.data.decision_start_seconds
        or 300 - core.data.min_seconds_before_close != config.data.decision_end_seconds
        or core.data.sample_interval_seconds != config.data.decision_interval_seconds
    ):
        raise RuntimeError("loss-tail core range/timing does not match the frozen contract")
    return core


def _execution_config(
    config: LossTailBenchmarkConfig,
    core: CoreTrainingConfig,
) -> ExecutionEvidenceConfig:
    return ExecutionEvidenceConfig(
        range_start=core.data.range_start,
        range_end=core.data.range_end,
        output_dir=config.paths.execution_evidence,
        sample_interval_seconds=config.data.decision_interval_seconds,
        min_seconds_after_open=config.data.context_start_seconds,
        max_seconds_after_open=config.data.decision_end_seconds,
        decision_min_seconds_after_open=config.data.decision_start_seconds,
        freshness_seconds=config.data.book_freshness_seconds,
        quantity=config.data.quantity,
        snapshot_schema_versions=(LEGACY_SNAPSHOT_SCHEMA_VERSION,),
    )


def _validate_strict_cache(frame: pl.DataFrame) -> None:
    duplicate = frame.group_by(list(_CACHE_KEYS)).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise RuntimeError("loss-tail strict cache has duplicate decision keys")
    if frame.height != EXPECTED_STRICT_ROWS:
        raise RuntimeError(
            f"loss-tail strict row count drifted: {frame.height} != {EXPECTED_STRICT_ROWS}"
        )
    if frame["market_id"].n_unique() != EXPECTED_STRICT_MARKETS:
        raise RuntimeError("loss-tail strict market count drifted")
    if _key_fingerprint(frame) != EXPECTED_STRICT_KEY_SHA256:
        raise RuntimeError("loss-tail strict decision-key cohort drifted")


def _proposal_cache_manifest(frame: pl.DataFrame, path: Path) -> dict[str, Any]:
    """Describe and fail closed on the frozen predecessor proposal universe."""

    proposal_key_sha256 = _key_fingerprint(frame)
    if frame.height != EXPECTED_PROPOSAL_ROWS or frame["market_id"].n_unique() != frame.height:
        raise RuntimeError("boundary proposal count drifted from the audited predecessor cohort")
    if proposal_key_sha256 != EXPECTED_PROPOSAL_KEY_SHA256:
        raise RuntimeError("boundary proposal keys drifted from the audited predecessor cohort")
    by_block = {
        row["walk_forward_block"]: int(row["proposals"])
        for row in frame.group_by("walk_forward_block")
        .agg(pl.len().alias("proposals"))
        .sort("walk_forward_block")
        .to_dicts()
    }
    direction_counts = {
        "up": int(frame.filter(pl.col("boundary_direction") == 1).height),
        "down": int(frame.filter(pl.col("boundary_direction") == 0).height),
    }
    correct = int(frame["boundary_direction_correct"].sum())
    return {
        "schema_version": "btc-boundary-proposal-cache-v1",
        "path": str(path),
        "sha256": file_sha256(path),
        "proposal_key_sha256": proposal_key_sha256,
        "proposals": frame.height,
        "markets": frame["market_id"].n_unique(),
        "proposals_by_block": by_block,
        "direction_counts": direction_counts,
        "correct_proposals": correct,
        "incorrect_proposals": frame.height - correct,
        "high_debit_threshold": 0.90,
        "high_debit_incorrect_proposals": frame.filter(
            (pl.col("boundary_direction_correct") == 0) & (pl.col("fee_inclusive_debit") >= 0.90)
        ).height,
        "source_training_sha256": EXPECTED_TRAINING_SHA256,
        "materialization": (
            "first causal OOF boundary proposal for fit/calibration history and exact matched "
            "predecessor proposal keys for B3-B5 at frozen 0.89 confidence; no predecessor "
            "NoTrade markets admitted"
        ),
    }


def _validate_cache_manifest(
    manifest: dict[str, Any],
    training_path: Path,
    config: LossTailBenchmarkConfig,
) -> None:
    expected = {
        "schema_version": "btc-boundary-loss-tail-cache-v1",
        "benchmark_config_sha256": EXPECTED_CACHE_CONFIG_SHA256,
        "core_feature_sha256": EXPECTED_CORE_FEATURE_SHA256,
        "execution_manifest_sha256": EXPECTED_EXECUTION_MANIFEST_SHA256,
        "strict_rows": EXPECTED_STRICT_ROWS,
        "strict_markets": EXPECTED_STRICT_MARKETS,
        "strict_key_sha256": EXPECTED_STRICT_KEY_SHA256,
        "training_sha256": EXPECTED_TRAINING_SHA256,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"immutable loss-tail cache manifest mismatch: {key}")
    if not training_path.is_file() or file_sha256(training_path) != manifest.get("training_sha256"):
        raise RuntimeError("immutable loss-tail training cache checksum mismatch")


def _prediction_artifact_frame(frame: pl.DataFrame) -> pl.DataFrame:
    frame = frame.with_columns(pl.col("predicted_up").alias("outcome_predicted_up"))
    missing = sorted(set(_PREDICTION_COLUMNS) - set(frame.columns))
    if missing:
        raise RuntimeError("prediction artifact is missing columns: " + ", ".join(missing))
    return frame.select(_PREDICTION_COLUMNS).sort(["window_start", "market_id", "seconds_elapsed"])


def _validate_matched_prediction_frames(
    frames: dict[str, pl.DataFrame],
) -> None:
    if MATCHED_BOUNDARY_CONTROL not in frames:
        raise RuntimeError("matched boundary control predictions are missing")
    comparison_columns = (*_CACHE_KEYS, "label_up")
    reference = (
        frames[MATCHED_BOUNDARY_CONTROL]
        .select(comparison_columns)
        .sort(["window_start", "market_id", "seconds_elapsed"])
    )
    if reference.is_empty() or reference.select(_CACHE_KEYS).is_duplicated().any():
        raise RuntimeError("matched boundary control prediction keys are empty or duplicated")
    for candidate, frame in frames.items():
        observed = frame.select(comparison_columns).sort(
            ["window_start", "market_id", "seconds_elapsed"]
        )
        if not observed.equals(reference):
            raise RuntimeError(
                f"{candidate} predictions do not share the control evaluation keys and labels"
            )


def _run_one_worker(
    function: Any,
    config_path: Path,
    cache_path: Path,
    run_dir: Path,
    *,
    resources: Any,
) -> dict[str, Any]:
    previous_environment = _set_worker_thread_environment(resources.threads_per_worker)
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as executor:
            return executor.submit(function, config_path, cache_path, run_dir).result()
    finally:
        _restore_environment(previous_environment)


def _apply_worker_resources(config: LossTailBenchmarkConfig) -> dict[str, Any]:
    memory_bytes = config.resources.memory_limit_gib * 1024**3
    applied: dict[str, Any] = {
        "pid": os.getpid(),
        "threads": config.resources.threads_per_worker,
        "memory_limit_bytes": memory_bytes,
        "memory_limits": {},
    }
    for name in ("RLIMIT_AS", "RLIMIT_RSS"):
        kind = getattr(resource, name, None)
        if kind is None:
            applied["memory_limits"][name] = "unsupported"
            continue
        _soft, hard = resource.getrlimit(kind)
        target = memory_bytes if hard == resource.RLIM_INFINITY else min(memory_bytes, hard)
        try:
            resource.setrlimit(kind, (target, hard))
            applied["memory_limits"][name] = target
        except (OSError, ValueError) as error:
            applied["memory_limits"][name] = f"unavailable: {error}"
    if any(isinstance(value, int) for value in applied["memory_limits"].values()):
        applied["memory_enforcement"] = {"mode": "address_space_rlimit"}
    else:
        applied["memory_enforcement"] = _start_max_rss_watchdog(memory_bytes)
    return applied


def _start_max_rss_watchdog(memory_bytes: int) -> dict[str, Any]:
    poll_seconds = 0.10
    baseline_bytes = _maximum_resident_set_bytes()
    if baseline_bytes > memory_bytes:
        raise RuntimeError("worker already exceeds its configured memory limit")

    def enforce_limit() -> None:
        while True:
            if _maximum_resident_set_bytes() > memory_bytes:
                os._exit(137)
            time.sleep(poll_seconds)

    threading.Thread(
        target=enforce_limit,
        name="loss-tail-memory-limit",
        daemon=True,
    ).start()
    return {
        "mode": "maximum_rss_watchdog",
        "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "baseline_bytes": baseline_bytes,
        "poll_interval_milliseconds": int(poll_seconds * 1_000),
        "exit_code_on_limit": 137,
    }


def _maximum_resident_set_bytes() -> int:
    maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum_rss if sys.platform == "darwin" else maximum_rss * 1_024


def _set_worker_thread_environment(threads: int) -> dict[str, str | None]:
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "POLARS_MAX_THREADS",
    )
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = str(threads)
    return previous


def _restore_environment(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(path)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(content)
    temporary.replace(path)


def _none_to_infinity(value: float | None) -> float:
    return math.inf if value is None else value


def _render_report(payload: dict[str, Any]) -> str:
    rows: list[str] = []
    for candidate, result in payload["candidates"].items():
        economics = result["economics"]
        execution = economics["execution"]
        decision = payload["selection"]["candidates"].get(candidate, {})
        proposal = result["proposal_outcomes"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(candidate)}</td>"
            f"<td>{'yes' if decision.get('passed') else 'control' if candidate == MATCHED_BOUNDARY_CONTROL else 'no'}</td>"
            f"<td>{economics['trades']}</td>"
            f"<td>{_format(economics['accuracy'])}</td>"
            f"<td>{_format(economics['accuracy_wilson_lower_95'])}</td>"
            f"<td>{_format(economics['book_qualified_coverage'])}</td>"
            f"<td>{_format(economics['mean_selected_ask_vwap_5'])}</td>"
            f"<td>{_format(execution['realized_net_pnl_total'])}</td>"
            f"<td>{_format(execution['profit_factor'])}</td>"
            f"<td>{_format(result['loss_ratios']['mean_loss_win'])}</td>"
            f"<td>{_format(result['loss_ratios']['gross_loss_profit'])}</td>"
            f"<td>{_format(proposal['no_trade_precision'])}</td>"
            f"<td>{_format(proposal['catastrophic_loss_recall'])}</td>"
            f"<td>{_format(proposal['winner_retention'])}</td>"
            f"<td>{_format(proposal['gross_profit_retained'])}</td>"
            f"<td>{_format(proposal['gross_loss_removed'])}</td>"
            f"<td>{_format(execution['maximum_drawdown'])}</td>"
            f"<td>{_format(execution['mean_worst_one_percent_realized_net_pnl'])}</td>"
            "</tr>"
        )
    selected = payload["selection"]["selected_candidate"] or "none"
    candidate_name = next(iter(payload["selection"]["candidates"]))
    candidate = payload["candidates"][candidate_name]
    gate_rows = "".join(
        "<tr>"
        f"<td>{html.escape(check['name'])}</td>"
        f"<td>{_format(check['observed'])}</td>"
        f"<td>{html.escape(check['operator'])}</td>"
        f"<td>{_format(check['required'])}</td>"
        f"<td>{'PASS' if check['passed'] else 'FAIL'}</td>"
        "</tr>"
        for check in payload["selection"]["candidates"][candidate_name]["checks"]
    )
    band_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{metrics['boundary_proposals']}</td>"
        f"<td>{metrics['trades']}</td>"
        f"<td>{_format(metrics['accuracy'])}</td>"
        f"<td>{_format(metrics['fee_inclusive_break_even_accuracy'])}</td>"
        f"<td>{_format(metrics['accuracy_minus_fee_inclusive_break_even'])}</td>"
        f"<td>{_format(metrics['realized_net_pnl_total'])}</td>"
        f"<td>{_format(metrics['profit_factor'])}</td>"
        f"<td>{metrics['incorrect_trades_converted_to_no_trade']}</td>"
        f"<td>{_format(metrics['probability_calibration']['brier_score'])}</td>"
        f"<td>{_format(metrics['probability_calibration']['expected_calibration_error'])}</td>"
        "</tr>"
        for name, metrics in candidate["proposal_price_bands"].items()
    )
    fold_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{metrics['economics']['trades']}</td>"
        f"<td>{_format(metrics['economics']['accuracy'])}</td>"
        f"<td>{_format(metrics['economics']['book_qualified_coverage'])}</td>"
        f"<td>{_format(metrics['economics']['execution']['realized_net_pnl_total'])}</td>"
        f"<td>{_format(metrics['economics']['execution']['profit_factor'])}</td>"
        f"<td>{_format(metrics['loss_ratios']['gross_loss_profit'])}</td>"
        "</tr>"
        for name, metrics in candidate["folds"].items()
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>BTC boundary loss-tail benchmark</title>
<style>body{{font:14px system-ui;margin:2rem;color:#17202a}}table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}th,td{{border:1px solid #ccd1d1;padding:.45rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#f4f6f7;padding:.15rem .3rem}}.PASS{{color:#176b32}}.FAIL{{color:#a61b1b}}</style></head>
<body><h1>BTC boundary loss-tail benchmark</h1>
<p><strong>Selected:</strong> {html.escape(selected)}. No fallback winner is permitted.</p>
<p>{html.escape(payload["objective"])}</p>
<table><thead><tr><th>Candidate</th><th>Qualified</th><th>Trades</th><th>Accuracy</th><th>Wilson lower</th><th>Coverage</th><th>Mean VWAP5</th><th>Net PnL</th><th>Profit factor</th><th>Loss/win</th><th>Gross loss/profit</th><th>NoTrade precision</th><th>Catastrophic recall</th><th>Winner retention</th><th>Gross profit retained</th><th>Gross loss removed</th><th>Drawdown</th><th>Worst 1%</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<h2>Promotion gates: {html.escape(candidate_name)}</h2>
<table><thead><tr><th>Requirement</th><th>Observed</th><th>Operator</th><th>Required</th><th>Result</th></tr></thead><tbody>{gate_rows}</tbody></table>
<h2>Walk-forward folds</h2>
<table><thead><tr><th>Fold</th><th>Trades</th><th>Accuracy</th><th>Coverage</th><th>Net PnL</th><th>Profit factor</th><th>Gross loss/profit</th></tr></thead><tbody>{fold_rows}</tbody></table>
<h2>Executable price bands</h2>
<table><thead><tr><th>Band</th><th>Proposals</th><th>Trades</th><th>Accuracy</th><th>Break-even</th><th>Margin</th><th>Net PnL</th><th>Profit factor</th><th>Incorrect→NoTrade</th><th>Brier</th><th>ECE</th></tr></thead><tbody>{band_rows}</tbody></table>
<p>Proposal cache: {payload["proposal_cache"]["proposals"]} rows, key SHA-256 <code>{payload["proposal_cache"]["proposal_key_sha256"]}</code>.</p>
<p>Evidence through July 28, 2026 is consumed development evidence. Runtime export, process creation, image rebuilds, and service changes were intentionally excluded.</p>
</body></html>"""


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isinf(value):
            return "∞"
        return f"{value:.6f}"
    return html.escape(str(value))
