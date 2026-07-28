from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from .core_benchmark import (
    AdvancementCriteria,
    BenchmarkEvidence,
    CandidatePolicy,
    benchmark_predictions,
)
from .core_config import CoreTrainingConfig, load_core_config
from .core_evaluation import (
    baseline_metrics,
    block_bootstrap_uplift,
    choose_threshold,
    classification_metrics,
    first_crossing_timing,
    paired_uplift,
    scored_prediction_rows,
    threshold_table,
)
from .core_execution import (
    ExecutionEvidenceConfig,
    load_execution_evidence_manifest,
)
from .core_extract import file_sha256, write_json_atomic
from .core_features import (
    CORE_ENRICHED_FEATURES,
    load_core_feature_frame,
    validate_core_feature_cache,
)
from .core_training import (
    CandidateSpec,
    FittedCoreModel,
    ProbabilityCalibrator,
    chronological_subsplit,
    configure_native_thread_limits,
    development_gate_passed,
    fit_probability_calibrator,
    range_frame,
    tune_and_fit_model,
)
from .persistence_config import (
    CalibrationBand,
    PersistenceBenchmarkConfig,
    load_persistence_benchmark_config,
    persistence_config_to_dict,
)
from .prewindow_features import (
    PREWINDOW_MODEL_FEATURES,
    build_prewindow_features,
    join_prewindow_features,
)
from .provenance import runtime_provenance

PERSISTENCE_BENCHMARK_SCHEMA_VERSION = "btc-path-persistence-benchmark-v1"
PATH_ZERO_EPSILON_BPS = 1e-12
DEFERRED_RUNTIME_CHECKS = {
    "runtime deployment contract is compatible",
    "native inference p99 is within budget",
    "runtime model size is within budget",
}
PERSISTENCE_TRAINING_CHECKS = {
    "minimum accepted markets by 120 seconds",
    "minimum early accuracy",
    "minimum early balanced accuracy",
    "minimum early UP recall",
    "minimum early DOWN recall",
    "minimum early Wilson lower bound",
    "qualified threshold in every fold",
    "minimum common selected executable markets",
    "nonnegative same-time uplift in every fold",
    "nonnegative hourly bootstrap lower bound",
    "walk-forward development gates",
}


@dataclass(frozen=True)
class CandidateProfile:
    name: str
    target_kind: Literal["outcome_up", "path_persistence"]
    feature_kind: Literal["core", "core_prewindow"]
    calibration_kind: Literal["global_platt", "time_banded_platt"]


@dataclass(frozen=True)
class CalibratorSet:
    kind: str
    calibrators: dict[str, ProbabilityCalibrator]
    bands: tuple[CalibrationBand, ...]


@dataclass
class PersistenceTrainingBundle:
    model: FittedCoreModel
    calibrators: CalibratorSet
    profile: CandidateProfile
    confidence_threshold: float

    def probability_up(self, frame: pl.DataFrame) -> np.ndarray:
        target_probability = calibrated_target_probability(
            self.model,
            self.calibrators,
            frame,
        )
        return target_probability_to_up(
            frame,
            target_probability,
            self.profile.target_kind,
        )


CANDIDATE_PROFILES = {
    profile.name: profile
    for profile in (
        CandidateProfile(
            "histogram_enriched",
            "outcome_up",
            "core",
            "global_platt",
        ),
        CandidateProfile(
            "histogram_path_persistence",
            "path_persistence",
            "core",
            "global_platt",
        ),
        CandidateProfile(
            "histogram_path_persistence_prewindow",
            "path_persistence",
            "core_prewindow",
            "global_platt",
        ),
        CandidateProfile(
            "histogram_path_persistence_time_calibrated",
            "path_persistence",
            "core_prewindow",
            "time_banded_platt",
        ),
    )
}


def run_persistence_benchmark(
    config: PersistenceBenchmarkConfig,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    _configure_compute(config)
    core_config = load_core_config(config.core_config)
    feature_metadata = validate_core_feature_cache(core_config, "pre_holdout")
    _assert_locked_development_range(core_config)
    prewindow_metadata = build_prewindow_features(
        core_config,
        config.prewindow_features,
        force=force,
    )
    execution_config = _execution_config(config)
    execution_manifest = load_execution_evidence_manifest(execution_config)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _update_progress(
        run_dir,
        "walk_forward_training",
        0.03,
        {
            "candidate_names": list(config.candidate_names),
            "runtime_export_enabled": False,
            "holdout_access_enabled": False,
        },
    )

    candidate_results = _train_candidate_matrix(config, run_dir)
    scored_frames = {
        name: result.pop("scored_rows")
        for name, result in candidate_results.items()
    }
    execution_frame = load_execution_evidence(execution_config)
    scored_with_execution = {
        name: attach_execution_evidence(frame, execution_frame)
        for name, frame in scored_frames.items()
    }
    universe = sorted(
        {
            market_id
            for frame in scored_frames.values()
            for market_id in frame["market_id"].unique().to_list()
        }
    )
    policies = {
        name: CandidatePolicy(
            confidence_threshold=None,
            deployment_compatible=name == config.control_candidate,
            selection_mode="chronological_preselected",
            confidence_threshold_min=core_config.model.confidence_min,
            confidence_threshold_max=core_config.model.confidence_max,
        )
        for name in config.candidate_names
    }
    benchmark = benchmark_predictions(
        scored_with_execution,
        policies=policies,
        control_candidate=config.control_candidate,
        evidence=BenchmarkEvidence(
            label=(
                "Five-fold chronological April 21-July 6 development evidence; "
                "historical labels consumed; compact-book execution evidence "
                "available only on the intersecting May 27-June 11 cohort"
            ),
            kind="development",
            independent=config.evaluation_is_independent,
        ),
        eligible_market_ids=universe,
        minimum_samples=config.minimum_common_markets,
        minimum_executable_samples=config.minimum_executable_markets,
        quantity=config.quantity,
        criteria=_advancement_criteria(config, core_config),
    )
    benchmark["common_selected_execution_comparisons"] = {
        name: common_selected_execution_comparison(
            scored_with_execution[config.control_candidate],
            scored_with_execution[name],
            control_name=config.control_candidate,
            candidate_name=name,
        )
        for name in config.candidate_names
        if name != config.control_candidate
    }
    _add_training_gates(
        benchmark,
        candidate_results,
        config,
        core_config,
    )
    finalist = _select_finalist(benchmark, candidate_results, config)
    freeze_record = _fit_development_finalist(
        finalist,
        config,
        core_config,
        run_dir,
    )

    benchmark.update(
        {
            "run_schema_version": PERSISTENCE_BENCHMARK_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "configuration": persistence_config_to_dict(config),
            "evaluation_note": config.evaluation_note,
            "runtime_provenance": runtime_provenance(config.package_root),
            "data_evidence": {
                "training_range": {
                    "start": core_config.data.range_start.isoformat(),
                    "end_exclusive": core_config.data.range_end.isoformat(),
                    "calendar_days": (
                        core_config.data.range_end - core_config.data.range_start
                    ).days,
                },
                "core_features": feature_metadata,
                "preopen": prewindow_metadata,
                "execution": execution_manifest,
                "execution_cohort": {
                    "start": execution_manifest["range_start"],
                    "end_exclusive": execution_manifest["range_end"],
                },
                "historical_compact_book_use": {
                    "model_feature_role": "excluded",
                    "execution_economics_role": (
                        "strict-both-side cached execution evidence on the "
                        "intersecting May 27-June 11 development cohort"
                    ),
                    "source": "compact 250 ms execution snapshots",
                    "raw_pmxt_archive_read": False,
                },
                "recent_backfill_book_use": {
                    "model_fitting_role": "excluded",
                    "probability_calibration_role": "excluded",
                    "policy_selection_role": "excluded",
                    "outcome_scoring_role": "excluded",
                    "execution_economics_role": "none",
                    "diagnostic_role": (
                        "read-only coverage and strict-validity audit only; "
                        "reported separately when an audited cohort is supplied"
                    ),
                    "reason": (
                        "recent clean shards are noncontiguous and below the "
                        "pre-registered independent sample gate"
                    ),
                    "raw_pmxt_archive_read": False,
                },
                "independent_holdout": {
                    "start": (
                        core_config.split.independent_holdout_start.isoformat()
                        if core_config.split.independent_holdout_start
                        else None
                    ),
                    "end_exclusive": (
                        core_config.split.independent_holdout_end.isoformat()
                        if core_config.split.independent_holdout_end
                        else None
                    ),
                    "outcome_labels_accessed": False,
                    "directional_features_accessed": False,
                    "model_evaluation_accessed": False,
                    "book_quality_diagnostics_accessed": False,
                },
            },
            "training_evidence": {
                "core_candidates": candidate_results,
            },
            "persistence_analysis": {
                name: {
                    "target_kind": result["target_kind"],
                    "feature_kind": result["feature_kind"],
                    "calibration_kind": result["calibration_kind"],
                    "path_behavior": result["path_behavior"],
                    "early": result["early"],
                    "calibration": result["calibration"],
                }
                for name, result in candidate_results.items()
            },
            "training_selection": {
                "finalist": finalist,
                "runtime_freeze_created": False,
                "development_bundle": freeze_record,
                "candidates": {
                    name: {
                        "passed": benchmark["candidates"][name]["advance"][
                            "benchmark_passed"
                        ],
                        "checks": benchmark["candidates"][name]["advance"]["checks"],
                    }
                    for name in config.candidate_names
                    if name != config.control_candidate
                },
            },
            "deployment": {
                "status": "blocked",
                "action": "training evidence only; no runtime export or deployment",
                "reasons": [
                    (
                        "July 21-August 4 outcome labels and directional features "
                        "remain untouched"
                    ),
                    (
                        "path-persistence target conversion, time-banded calibration, "
                        "and pre-window features are not runtime-v1 contracts"
                    ),
                    "no Rust, image, process, playbook, or adapter change is authorized",
                ],
            },
        }
    )
    for name, frame in scored_frames.items():
        frame.write_parquet(
            run_dir / f"{name}-scored-probabilities.parquet",
            compression="zstd",
            statistics=True,
        )
    write_json_atomic(run_dir / "benchmark.json", benchmark)
    from .core_benchmark_report import generate_benchmark_report

    generate_benchmark_report(benchmark, run_dir / "report.html")
    _update_progress(
        run_dir,
        "benchmark_complete",
        1.0,
        {
            "finalist": finalist,
            "benchmark_passed_candidates": benchmark[
                "benchmark_passed_candidates"
            ],
            "holdout_accessed": False,
            "holdout_outcome_labels_accessed": False,
            "holdout_directional_features_accessed": False,
            "holdout_book_quality_diagnostics_accessed": False,
            "runtime_changed": False,
        },
    )
    return run_dir, benchmark


def persistence_target_labels(frame: pl.DataFrame) -> np.ndarray:
    return (
        frame["label_up"].cast(pl.Int8).to_numpy()
        == frame["binance_sign_up"].cast(pl.Int8).to_numpy()
    ).astype(np.int8)


def target_probability_to_up(
    frame: pl.DataFrame,
    target_probability: np.ndarray,
    target_kind: str,
) -> np.ndarray:
    if frame.height != len(target_probability):
        raise ValueError("target probability count does not match frame")
    if target_kind == "outcome_up":
        return np.asarray(target_probability, dtype=np.float64)
    if target_kind != "path_persistence":
        raise ValueError(f"unsupported target kind: {target_kind}")
    sign_up = frame["binance_sign_up"].cast(pl.Int8).to_numpy() == 1
    probability = np.asarray(target_probability, dtype=np.float64)
    return np.where(sign_up, probability, 1.0 - probability)


def path_is_directionally_eligible(frame: pl.DataFrame) -> np.ndarray:
    path = frame["btc_path_from_window_open_bps"].to_numpy()
    return np.isfinite(path) & (np.abs(path) > PATH_ZERO_EPSILON_BPS)


def calibrated_target_probability(
    model: FittedCoreModel,
    calibrators: CalibratorSet,
    frame: pl.DataFrame,
) -> np.ndarray:
    raw_logit = model.raw_logit(frame)
    if calibrators.kind == "global_platt":
        return calibrators.calibrators["global"].probability(raw_logit)
    if calibrators.kind != "time_banded_platt":
        raise ValueError(f"unsupported calibration kind: {calibrators.kind}")
    elapsed = frame["seconds_elapsed"].to_numpy()
    output = np.full(frame.height, np.nan, dtype=np.float64)
    for band in calibrators.bands:
        mask = (elapsed >= band.start_second) & (
            elapsed < band.end_second_exclusive
        )
        output[mask] = calibrators.calibrators[band.name].probability(
            raw_logit[mask]
        )
    if not np.isfinite(output).all():
        raise RuntimeError("time calibration bands do not cover every scored row")
    return output


def load_execution_evidence(config: ExecutionEvidenceConfig) -> pl.DataFrame:
    manifest = load_execution_evidence_manifest(config)
    files = [
        config.output_dir / partition["path"]
        for partition in manifest["partitions"]
    ]
    return (
        pl.scan_parquet(files)
        .select(
            "market_id",
            "observed_at",
            "fee_rate",
            "up_ask_vwap_5",
            "down_ask_vwap_5",
            "up_side_fresh",
            "down_side_fresh",
            "strict_both_side_eligible",
        )
        .collect()
    )


def attach_execution_evidence(
    predictions: pl.DataFrame,
    evidence: pl.DataFrame,
) -> pl.DataFrame:
    return predictions.join(
        evidence.select(
            "market_id",
            "observed_at",
            "fee_rate",
            "up_ask_vwap_5",
            "down_ask_vwap_5",
            "up_side_fresh",
            "down_side_fresh",
            "strict_both_side_eligible",
        ),
        on=["market_id", "observed_at"],
        how="left",
        validate="m:1",
    ).with_columns(
        pl.col("strict_both_side_eligible")
        .is_not_null()
        .alias("execution_evidence_available"),
        (
            pl.col("strict_both_side_eligible").fill_null(False)
            & pl.col("up_side_fresh").fill_null(False)
        ).alias("up_executable"),
        (
            pl.col("strict_both_side_eligible").fill_null(False)
            & pl.col("down_side_fresh").fill_null(False)
        ).alias("down_executable"),
    )


def common_selected_execution_comparison(
    control: pl.DataFrame,
    candidate: pl.DataFrame,
    *,
    control_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    control_rows = _strict_selected_execution_rows(control, "control")
    candidate_rows = _strict_selected_execution_rows(candidate, "candidate")
    common = control_rows.join(
        candidate_rows,
        on="market_id",
        how="inner",
        validate="1:1",
    )
    exact_timestamp_markets = common.filter(
        pl.col("control_observed_at") == pl.col("candidate_observed_at")
    ).height
    same_direction_markets = common.filter(
        pl.col("control_predicted_up") == pl.col("candidate_predicted_up")
    ).height
    return {
        "control_candidate": control_name,
        "candidate": candidate_name,
        "cohort": (
            "same market selected by both policies with strict-both-side execution "
            "evidence available at each policy's causal decision timestamp"
        ),
        "control_strict_executable_markets": control_rows.height,
        "candidate_strict_executable_markets": candidate_rows.height,
        "common_markets": common.height,
        "exact_timestamp_markets": exact_timestamp_markets,
        "same_direction_markets": same_direction_markets,
    }


def _strict_selected_execution_rows(
    frame: pl.DataFrame,
    prefix: str,
) -> pl.DataFrame:
    required = {
        "market_id",
        "observed_at",
        "seconds_elapsed",
        "predicted_up",
        "policy_selected",
        "up_executable",
        "down_executable",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "selected execution comparison is missing columns: "
            + ", ".join(missing)
        )
    executable = (
        pl.when(pl.col("predicted_up") == 1)
        .then(pl.col("up_executable"))
        .otherwise(pl.col("down_executable"))
        .fill_null(False)
    )
    return (
        frame.filter(pl.col("policy_selected") & executable)
        .select(
            "market_id",
            pl.col("observed_at").alias(f"{prefix}_observed_at"),
            pl.col("seconds_elapsed").alias(f"{prefix}_seconds_elapsed"),
            pl.col("predicted_up").alias(f"{prefix}_predicted_up"),
        )
        .sort("market_id")
    )


def _train_candidate_matrix(
    config: PersistenceBenchmarkConfig,
    run_dir: Path,
) -> dict[str, dict[str, Any]]:
    max_workers = min(
        load_core_config(config.core_config).compute.max_parallel_fits,
        len(config.candidate_names),
    )
    results: dict[str, dict[str, Any]] = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_candidate_task,
                config.source_path,
                candidate_name,
            ): candidate_name
            for candidate_name in config.candidate_names
        }
        for future in as_completed(futures):
            candidate_name = futures[future]
            results[candidate_name] = future.result()
            completed += 1
            metrics = results[candidate_name]["out_of_fold"]
            print(
                f"persistence benchmark: {candidate_name} "
                f"accuracy={metrics['accuracy']:.4f} "
                f"coverage={metrics['coverage']:.4f}",
                flush=True,
            )
            _update_progress(
                run_dir,
                "walk_forward_training",
                0.05 + 0.55 * completed / len(config.candidate_names),
                {
                    "completed_candidates": completed,
                    "total_candidates": len(config.candidate_names),
                    "latest_candidate": candidate_name,
                },
            )
    return {name: results[name] for name in config.candidate_names}


def _evaluate_candidate_task(
    config_path: Path,
    candidate_name: str,
) -> dict[str, Any]:
    config = load_persistence_benchmark_config(config_path)
    core_config = load_core_config(config.core_config)
    configure_native_thread_limits(core_config)
    frame = load_core_feature_frame(core_config, "pre_holdout")
    profile = CANDIDATE_PROFILES[candidate_name]
    if profile.feature_kind == "core_prewindow":
        prewindow = pl.read_parquet(config.prewindow_features)
        frame = join_prewindow_features(frame, prewindow)
    with threadpool_limits(limits=core_config.compute.threads_per_fit):
        folds = [
            _evaluate_fold(frame, profile, fold_index, config, core_config)
            for fold_index in range(len(core_config.split.validation_windows))
        ]
    return _aggregate_candidate(profile, folds, config, core_config)


def _evaluate_fold(
    frame: pl.DataFrame,
    profile: CandidateProfile,
    fold_index: int,
    config: PersistenceBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    validation_start, validation_end = core_config.split.validation_windows[
        fold_index
    ]
    history = _candidate_eligible_frame(
        range_frame(frame, core_config.split.development_start, validation_start),
        profile,
    )
    validation_source = range_frame(frame, validation_start, validation_end)
    universal_validation = _path_eligible_frame(validation_source)
    validation = _candidate_eligible_frame(
        validation_source,
        profile,
    )
    fit_frame, calibration_frame, policy_frame = chronological_subsplit(
        history,
        fit_fraction=0.70,
        calibration_fraction=0.15,
    )
    spec = _candidate_spec(profile)
    started = time.perf_counter()
    model, tuning = tune_and_fit_model(
        _training_target_frame(fit_frame, profile),
        spec,
        core_config,
    )
    calibrators, calibration_evidence = _fit_calibrators(
        model,
        calibration_frame,
        profile,
        config,
        core_config,
        spec,
    )
    policy_probability = target_probability_to_up(
        policy_frame,
        calibrated_target_probability(model, calibrators, policy_frame),
        profile.target_kind,
    )
    thresholds = threshold_table(
        policy_frame,
        policy_probability,
        core_config.model,
    )
    minimum_markets = max(
        50,
        math.ceil(
            policy_frame["market_id"].n_unique()
            * core_config.gates.minimum_coverage
        ),
    )
    threshold, threshold_qualified = choose_threshold(
        thresholds,
        core_config.gates,
        minimum_markets=minimum_markets,
    )
    validation_target_probability = calibrated_target_probability(
        model,
        calibrators,
        validation,
    )
    validation_probability_up = target_probability_to_up(
        validation,
        validation_target_probability,
        profile.target_kind,
    )
    scored = _scored_rows(
        validation,
        validation_probability_up,
        validation_target_probability,
        profile,
        fold_index,
        threshold,
    )
    selected = scored.filter(pl.col("policy_selected"))
    eligible = universal_validation["market_id"].n_unique()
    metrics = classification_metrics(selected, eligible_markets=eligible)
    paired = paired_uplift(selected)
    bootstrap = block_bootstrap_uplift(
        selected,
        resamples=core_config.gates.bootstrap_resamples,
        random_seed=core_config.model.random_seed + fold_index,
        block="hour",
    )
    return {
        "candidate": profile.name,
        "fold_index": fold_index,
        "fit_range_start": fit_frame["window_start"].min().isoformat(),
        "fit_range_end": fit_frame["window_start"].max().isoformat(),
        "calibration_range_start": calibration_frame["window_start"].min().isoformat(),
        "calibration_range_end": calibration_frame["window_start"].max().isoformat(),
        "policy_range_start": policy_frame["window_start"].min().isoformat(),
        "policy_range_end": policy_frame["window_start"].max().isoformat(),
        "validation_range_start": validation_start.isoformat(),
        "validation_range_end": validation_end.isoformat(),
        "eligible_markets": eligible,
        "confidence_threshold": threshold,
        "threshold_qualified": threshold_qualified,
        "threshold_history": thresholds,
        "tuning": tuning,
        "calibration": calibration_evidence,
        "metrics": metrics,
        "baseline": baseline_metrics(selected, eligible_markets=eligible),
        "paired": paired,
        "bootstrap": bootstrap,
        "timing": first_crossing_timing(selected, eligible_markets=eligible),
        "elapsed_seconds": time.perf_counter() - started,
        "scored_rows": scored,
        "selected_rows": selected,
    }


def _aggregate_candidate(
    profile: CandidateProfile,
    folds: list[dict[str, Any]],
    config: PersistenceBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    scored = pl.concat(
        [fold["scored_rows"] for fold in folds],
        how="vertical_relaxed",
    ).sort(["observed_at", "market_id"])
    selected = pl.concat(
        [fold["selected_rows"] for fold in folds],
        how="vertical_relaxed",
    ).sort(["observed_at", "market_id"])
    eligible = sum(int(fold["eligible_markets"]) for fold in folds)
    metrics = classification_metrics(selected, eligible_markets=eligible)
    paired = paired_uplift(selected)
    bootstrap = block_bootstrap_uplift(
        selected,
        resamples=core_config.gates.bootstrap_resamples,
        random_seed=core_config.model.random_seed,
        block="hour",
    )
    nonnegative_folds = sum(
        fold["paired"]["accuracy_uplift"]
        >= core_config.gates.minimum_same_time_path_uplift
        for fold in folds
    )
    qualified_threshold_folds = sum(
        bool(fold["threshold_qualified"]) for fold in folds
    )
    early_rows = selected.filter(
        pl.col("seconds_elapsed") <= config.early_cutoff_second
    )
    early = classification_metrics(early_rows, eligible_markets=eligible)
    path_behavior = {
        "followed": _path_behavior_metrics(
            selected.filter(pl.col("path_followed")),
            eligible,
        ),
        "reversed": _path_behavior_metrics(
            selected.filter(~pl.col("path_followed")),
            eligible,
        ),
    }
    passed = development_gate_passed(
        core_config,
        metrics,
        paired,
        bootstrap,
        nonnegative_folds,
    )
    fold_summaries = []
    for fold in folds:
        fold_summaries.append(
            {
                key: value
                for key, value in fold.items()
                if key not in {"scored_rows", "selected_rows", "threshold_history"}
            }
        )
    return {
        "candidate": profile.name,
        "family": "histogram",
        "target_kind": profile.target_kind,
        "feature_kind": profile.feature_kind,
        "calibration_kind": profile.calibration_kind,
        "feature_count": len(_candidate_spec(profile).feature_names),
        "features": list(_candidate_spec(profile).feature_names),
        "folds": fold_summaries,
        "out_of_fold": metrics,
        "baseline": baseline_metrics(selected, eligible_markets=eligible),
        "paired": paired,
        "bootstrap": bootstrap,
        "timing": first_crossing_timing(selected, eligible_markets=eligible),
        "early": early,
        "path_behavior": path_behavior,
        "calibration": [fold["calibration"] for fold in folds],
        "nonnegative_uplift_folds": nonnegative_folds,
        "qualified_threshold_folds": qualified_threshold_folds,
        "total_folds": len(folds),
        "passed_development": passed,
        "scored_rows": scored,
    }


def _candidate_spec(profile: CandidateProfile) -> CandidateSpec:
    features = tuple(CORE_ENRICHED_FEATURES)
    if profile.feature_kind == "core_prewindow":
        features += tuple(PREWINDOW_MODEL_FEATURES)
    return CandidateSpec(
        profile.name,
        "histogram",
        features,
    )


def _candidate_eligible_frame(
    frame: pl.DataFrame,
    profile: CandidateProfile,
) -> pl.DataFrame:
    eligible = _path_eligible_frame(frame)
    if profile.feature_kind == "core_prewindow":
        eligible = eligible.filter(pl.col("prewindow_model_eligible"))
    if eligible.is_empty():
        raise RuntimeError(f"{profile.name} has no eligible point-in-time rows")
    return eligible


def _path_eligible_frame(frame: pl.DataFrame) -> pl.DataFrame:
    eligible = frame.filter(
        pl.Series("_path_eligible", path_is_directionally_eligible(frame))
    )
    if eligible.is_empty():
        raise RuntimeError("cohort has no directionally eligible BTC path rows")
    return eligible


def _training_target_frame(
    frame: pl.DataFrame,
    profile: CandidateProfile,
) -> pl.DataFrame:
    if profile.target_kind == "outcome_up":
        return frame
    return frame.with_columns(
        pl.Series("label_up", persistence_target_labels(frame), dtype=pl.Int8)
    )


def _fit_calibrators(
    model: FittedCoreModel,
    calibration_frame: pl.DataFrame,
    profile: CandidateProfile,
    config: PersistenceBenchmarkConfig,
    core_config: CoreTrainingConfig,
    spec: CandidateSpec,
) -> tuple[CalibratorSet, list[dict[str, Any]]]:
    target_frame = _training_target_frame(calibration_frame, profile)
    if profile.calibration_kind == "global_platt":
        calibrator = fit_probability_calibrator(
            model,
            target_frame,
            core_config,
            spec,
        )
        _validate_calibrator(calibrator, target_frame, "global", config)
        return (
            CalibratorSet("global_platt", {"global": calibrator}, ()),
            [_calibration_evidence("global", target_frame, calibrator)],
        )
    fitted: dict[str, ProbabilityCalibrator] = {}
    evidence: list[dict[str, Any]] = []
    for band in config.calibration_bands:
        band_frame = target_frame.filter(
            pl.col("seconds_elapsed").is_between(
                band.start_second,
                band.end_second_exclusive,
                closed="left",
            )
        )
        calibrator = fit_probability_calibrator(
            model,
            band_frame,
            core_config,
            spec,
        )
        _validate_calibrator(calibrator, band_frame, band.name, config)
        fitted[band.name] = calibrator
        evidence.append(_calibration_evidence(band.name, band_frame, calibrator))
    return (
        CalibratorSet("time_banded_platt", fitted, config.calibration_bands),
        evidence,
    )


def _validate_calibrator(
    calibrator: ProbabilityCalibrator,
    target_frame: pl.DataFrame,
    name: str,
    config: PersistenceBenchmarkConfig,
) -> None:
    labels = set(target_frame["label_up"].unique().to_list())
    if target_frame.height < config.minimum_calibration_rows_per_band:
        raise RuntimeError(f"calibration band {name} has insufficient rows")
    if labels != {0, 1}:
        raise RuntimeError(f"calibration band {name} does not contain both classes")
    if not calibrator.converged or calibrator.slope <= 0.0:
        raise RuntimeError(
            f"calibration band {name} failed convergence/monotonicity"
        )


def _calibration_evidence(
    name: str,
    frame: pl.DataFrame,
    calibrator: ProbabilityCalibrator,
) -> dict[str, Any]:
    return {
        "band": name,
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "positive_target_rate": float(frame["label_up"].mean()),
        **asdict(calibrator),
    }


def _scored_rows(
    frame: pl.DataFrame,
    probability_up: np.ndarray,
    probability_target: np.ndarray,
    profile: CandidateProfile,
    fold_index: int,
    threshold: float,
) -> pl.DataFrame:
    scored = scored_prediction_rows(frame, probability_up).with_columns(
        pl.Series("probability_target", probability_target),
        pl.lit(profile.name).alias("candidate"),
        pl.lit(profile.target_kind).alias("target_kind"),
        pl.lit(profile.calibration_kind).alias("calibration_kind"),
        pl.lit(fold_index).cast(pl.Int32).alias("fold_index"),
        pl.lit(threshold).alias("selected_confidence_threshold"),
        pl.lit(True).alias("model_eligible"),
    )
    scored = scored.with_columns(
        (pl.col("predicted_up") == pl.col("binance_sign_up")).alias(
            "path_followed"
        )
    )
    keys = [
        "candidate",
        "fold_index",
        "market_id",
        "observed_at",
        "seconds_elapsed",
    ]
    selected_keys = (
        scored.filter(pl.col("confidence") >= threshold)
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .group_by("market_id", maintain_order=True)
        .first()
        .select(keys)
        .with_columns(pl.lit(True).alias("policy_selected"))
    )
    return (
        scored.join(
            selected_keys,
            on=keys,
            how="left",
            validate="1:1",
        )
        .with_columns(pl.col("policy_selected").fill_null(False))
        .sort(["market_id", "seconds_elapsed", "observed_at"])
    )


def _path_behavior_metrics(
    rows: pl.DataFrame,
    eligible_markets: int,
) -> dict[str, Any]:
    metrics = classification_metrics(rows, eligible_markets=eligible_markets)
    return {
        "markets": metrics["markets"],
        "coverage": metrics["coverage"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "up_recall": metrics["up_recall"],
        "down_recall": metrics["down_recall"],
    }


def _add_training_gates(
    benchmark: dict[str, Any],
    candidate_results: dict[str, dict[str, Any]],
    config: PersistenceBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> None:
    for name in config.candidate_names:
        if name == config.control_candidate:
            continue
        result = candidate_results[name]
        advance = benchmark["candidates"][name]["advance"]
        deferred = [
            check
            for check in advance["checks"]
            if check["name"] in DEFERRED_RUNTIME_CHECKS
        ]
        checks = [
            check
            for check in advance["checks"]
            if check["name"] not in DEFERRED_RUNTIME_CHECKS
            and check["name"] not in PERSISTENCE_TRAINING_CHECKS
        ]
        common_checkpoints = benchmark["common_comparisons"][name]["checkpoints"]
        for checkpoint in common_checkpoints:
            second = int(checkpoint["seconds_elapsed"])
            checks.extend(
                (
                    _check(
                        f"{second}s common-time accuracy does not regress",
                        checkpoint["accuracy_delta"],
                        ">=",
                        -config.maximum_accuracy_regression,
                    ),
                    _check(
                        f"{second}s common-time balanced accuracy does not regress",
                        checkpoint["balanced_accuracy_delta"],
                        ">=",
                        -config.maximum_balanced_accuracy_regression,
                    ),
                    _check(
                        f"{second}s common-time UP recall does not regress",
                        checkpoint["up_recall_delta"],
                        ">=",
                        -config.maximum_direction_recall_regression,
                    ),
                    _check(
                        f"{second}s common-time DOWN recall does not regress",
                        checkpoint["down_recall_delta"],
                        ">=",
                        -config.maximum_direction_recall_regression,
                    ),
                )
            )
        common_execution = benchmark["common_selected_execution_comparisons"][name]
        checks.extend(
            (
                _check(
                    "minimum accepted markets by 120 seconds",
                    result["early"]["markets"],
                    ">=",
                    config.minimum_early_markets,
                ),
                _check(
                    "minimum early accuracy",
                    result["early"]["accuracy"],
                    ">=",
                    core_config.gates.target_accuracy,
                ),
                _check(
                    "minimum early balanced accuracy",
                    result["early"]["balanced_accuracy"],
                    ">=",
                    core_config.gates.target_balanced_accuracy,
                ),
                _check(
                    "minimum early UP recall",
                    result["early"]["up_recall"],
                    ">=",
                    core_config.gates.minimum_direction_recall,
                ),
                _check(
                    "minimum early DOWN recall",
                    result["early"]["down_recall"],
                    ">=",
                    core_config.gates.minimum_direction_recall,
                ),
                _check(
                    "minimum early Wilson lower bound",
                    result["early"]["wilson_lower_95"],
                    ">=",
                    core_config.gates.target_wilson_lower,
                ),
                _check(
                    "qualified threshold in every fold",
                    result["qualified_threshold_folds"],
                    "==",
                    result["total_folds"],
                ),
                _check(
                    "minimum common selected executable markets",
                    common_execution["common_markets"],
                    ">=",
                    config.minimum_executable_markets,
                ),
                _check(
                    "nonnegative same-time uplift in every fold",
                    result["nonnegative_uplift_folds"],
                    ">=",
                    core_config.gates.minimum_nonnegative_uplift_folds,
                ),
                _check(
                    "nonnegative hourly bootstrap lower bound",
                    result["bootstrap"]["lower_95"],
                    ">=",
                    0.0,
                ),
                _check(
                    "walk-forward development gates",
                    int(result["passed_development"]),
                    "==",
                    1,
                ),
            )
        )
        advance["checks"] = checks
        advance["deferred_runtime_checks"] = deferred
        advance["benchmark_passed"] = all(check["passed"] for check in checks)
        advance["deployment_qualified"] = False
    benchmark["benchmark_passed_candidates"] = [
        name
        for name in config.candidate_names
        if name != config.control_candidate
        and benchmark["candidates"][name]["advance"]["benchmark_passed"]
    ]
    benchmark["deployment_qualified_candidates"] = []


def _check(
    name: str,
    observed: float,
    operator: str,
    required: float,
) -> dict[str, Any]:
    if operator == ">=":
        passed = observed >= required
    elif operator == "==":
        passed = observed == required
    else:
        raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "required": required,
        "passed": bool(passed),
    }


def _select_finalist(
    benchmark: dict[str, Any],
    candidate_results: dict[str, dict[str, Any]],
    config: PersistenceBenchmarkConfig,
) -> str | None:
    passing = benchmark["benchmark_passed_candidates"]
    if not passing:
        return None
    return max(
        passing,
        key=lambda name: (
            candidate_results[name]["out_of_fold"]["coverage"],
            candidate_results[name]["out_of_fold"]["wilson_lower_95"],
            candidate_results[name]["out_of_fold"]["balanced_accuracy"],
            candidate_results[name]["out_of_fold"]["accuracy"],
            -(
                candidate_results[name]["timing"]["median_first_crossing_seconds"]
                or float("inf")
            ),
        ),
    )


def _fit_development_finalist(
    finalist: str | None,
    config: PersistenceBenchmarkConfig,
    core_config: CoreTrainingConfig,
    run_dir: Path,
) -> dict[str, Any] | None:
    if finalist is None:
        return None
    profile = CANDIDATE_PROFILES[finalist]
    frame = load_core_feature_frame(core_config, "pre_holdout")
    if profile.feature_kind == "core_prewindow":
        frame = join_prewindow_features(
            frame,
            pl.read_parquet(config.prewindow_features),
        )
    development = _candidate_eligible_frame(
        range_frame(
            frame,
            core_config.split.development_start,
            core_config.split.development_end,
        ),
        profile,
    )
    calibration = _candidate_eligible_frame(
        range_frame(
            frame,
            core_config.split.probability_calibration_start,
            core_config.split.probability_calibration_end,
        ),
        profile,
    )
    policy_source = range_frame(
        frame,
        core_config.split.policy_selection_start,
        core_config.split.policy_selection_end,
    )
    policy_universe = _path_eligible_frame(policy_source)
    policy = _candidate_eligible_frame(
        policy_source,
        profile,
    )
    spec = _candidate_spec(profile)
    model, tuning = tune_and_fit_model(
        _training_target_frame(development, profile),
        spec,
        core_config,
    )
    calibrators, calibration_evidence = _fit_calibrators(
        model,
        calibration,
        profile,
        config,
        core_config,
        spec,
    )
    policy_target_probability = calibrated_target_probability(
        model,
        calibrators,
        policy,
    )
    policy_probability_up = target_probability_to_up(
        policy,
        policy_target_probability,
        profile.target_kind,
    )
    thresholds = threshold_table(
        policy,
        policy_probability_up,
        core_config.model,
    )
    threshold, threshold_qualified = choose_threshold(
        thresholds,
        core_config.gates,
        minimum_markets=max(
            100,
            math.ceil(
                policy_universe["market_id"].n_unique()
                * core_config.gates.minimum_coverage
            ),
        ),
    )
    policy_rows = _scored_rows(
        policy,
        policy_probability_up,
        policy_target_probability,
        profile,
        -1,
        threshold,
    ).filter(pl.col("policy_selected"))
    metrics = classification_metrics(
        policy_rows,
        eligible_markets=policy_universe["market_id"].n_unique(),
    )
    policy_passed = bool(
        threshold_qualified
        and metrics["accuracy"] >= core_config.gates.target_accuracy
        and metrics["balanced_accuracy"]
        >= core_config.gates.target_balanced_accuracy
        and metrics["up_recall"] >= core_config.gates.minimum_direction_recall
        and metrics["down_recall"] >= core_config.gates.minimum_direction_recall
        and metrics["wilson_lower_95"] >= core_config.gates.target_wilson_lower
        and metrics["expected_calibration_error"] <= core_config.gates.maximum_ece
    )
    if not policy_passed:
        return {
            "status": "blocked_policy_selection",
            "candidate": finalist,
            "threshold": threshold,
            "threshold_qualified": threshold_qualified,
            "metrics": metrics,
            "bundle_created": False,
        }
    bundle = PersistenceTrainingBundle(
        model=model,
        calibrators=calibrators,
        profile=profile,
        confidence_threshold=threshold,
    )
    destination = run_dir / "development-training-candidate.joblib"
    joblib.dump(bundle, destination, compress=3)
    return {
        "status": "development_candidate_frozen",
        "candidate": finalist,
        "training_only": True,
        "runtime_v1_compatible": False,
        "model_file": destination.name,
        "model_sha256": file_sha256(destination),
        "threshold": threshold,
        "threshold_qualified": threshold_qualified,
        "metrics": metrics,
        "tuning": tuning,
        "calibration": calibration_evidence,
        "bundle_created": True,
    }


def _advancement_criteria(
    config: PersistenceBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> AdvancementCriteria:
    return AdvancementCriteria(
        minimum_accuracy=core_config.gates.target_accuracy,
        minimum_balanced_accuracy=core_config.gates.target_balanced_accuracy,
        minimum_direction_recall=core_config.gates.minimum_direction_recall,
        minimum_wilson_lower_95=core_config.gates.target_wilson_lower,
        maximum_expected_calibration_error=core_config.gates.maximum_ece,
        minimum_coverage=core_config.gates.minimum_coverage,
        minimum_coverage_uplift=config.minimum_coverage_uplift,
        maximum_accuracy_regression=config.maximum_accuracy_regression,
        maximum_balanced_accuracy_regression=(
            config.maximum_balanced_accuracy_regression
        ),
        maximum_direction_recall_regression=(
            config.maximum_direction_recall_regression
        ),
        maximum_median_entry_seconds_regression=(
            config.maximum_median_entry_seconds_regression
        ),
        minimum_mean_direct_edge_per_share=(
            config.minimum_mean_direct_edge_per_share
        ),
        minimum_realized_net_per_share=config.minimum_realized_net_per_share,
        minimum_common_time_markets=config.minimum_common_markets,
    )


def _execution_config(
    config: PersistenceBenchmarkConfig,
) -> ExecutionEvidenceConfig:
    manifest_path = config.execution_evidence / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"execution manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    return ExecutionEvidenceConfig(
        range_start=datetime.fromisoformat(manifest["range_start"]),
        range_end=datetime.fromisoformat(manifest["range_end"]),
        output_dir=config.execution_evidence,
        sample_interval_seconds=int(manifest["sample_interval_seconds"]),
        min_seconds_after_open=int(manifest["min_seconds_after_open"]),
        max_seconds_after_open=int(manifest["max_seconds_after_open"]),
        freshness_seconds=int(manifest["freshness_seconds"]),
        quantity=float(manifest["quantity"]),
    )


def _assert_locked_development_range(config: CoreTrainingConfig) -> None:
    metadata = validate_core_feature_cache(config, "pre_holdout")
    contract = metadata.get("build_contract", {})
    if (
        datetime.fromisoformat(str(contract.get("range_start")))
        != config.data.range_start
        or datetime.fromisoformat(str(contract.get("range_end")))
        != config.data.range_end
    ):
        raise RuntimeError("development cache escapes the locked training range")
    holdout_path = config.paths.holdout_feature_data
    access_parent = config.paths.artifacts
    holdout_access = (
        list(access_parent.glob("holdout-access-2026-07-21-2026-08-04.json"))
        if access_parent.exists()
        else []
    )
    if holdout_access:
        raise RuntimeError("independent holdout access was already recorded")
    if holdout_path.exists():
        raise RuntimeError(
            "independent holdout feature cache exists before candidate freeze"
        )


def _configure_compute(config: PersistenceBenchmarkConfig) -> None:
    core = load_core_config(config.core_config)
    threads = str(core.compute.threads_per_fit)
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["NUMEXPR_NUM_THREADS"] = threads
    os.environ.setdefault("POLARS_MAX_THREADS", str(core.compute.polars_threads))


def _update_progress(
    run_dir: Path,
    state: str,
    fraction: float,
    details: dict[str, Any] | None = None,
) -> None:
    write_json_atomic(
        run_dir / "progress.json",
        {
            "schema_version": "btc-path-persistence-progress-v1",
            "updated_at": datetime.now(UTC).isoformat(),
            "state": state,
            "fraction": fraction,
            "details": details or {},
        },
    )
