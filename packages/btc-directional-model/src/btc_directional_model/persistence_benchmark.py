from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
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
    fixed_time_prediction_rows,
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
    CORE_BOUNDARY_ENRICHED_FEATURES,
    CORE_BOUNDARY_REVERSAL_ENRICHED_FEATURES,
    CORE_ENRICHED_FEATURES,
    CORE_MATURE_REVERSAL_ENRICHED_FEATURES,
    CORE_REGIME_REVERSAL_ENRICHED_FEATURES,
    load_core_feature_frame,
    validate_core_feature_cache,
)
from .core_training import (
    EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY,
    MARKET_EQUAL_ROW_WEIGHT_POLICY,
    CandidateSpec,
    FittedCoreModel,
    ProbabilityCalibrator,
    chronological_subsplit,
    configure_native_thread_limits,
    development_gate_passed,
    estimator_converged,
    fit_model,
    fit_probability_calibrator,
    histogram_parameters,
    range_frame,
    tune_and_fit_model,
)
from .persistence_config import (
    BOUNDARY_ALIGNMENT_CANDIDATE,
    BOUNDARY_REVERSAL_ACCURACY_CANDIDATE,
    BOUNDARY_REVERSAL_ACCURACY_PROFILE,
    FOLD_ROBUST_FREQUENCY_CANDIDATE,
    FOLD_ROBUST_FREQUENCY_PROFILE,
    MATURE_REVERSAL_ACCURACY_CANDIDATE,
    MATURE_REVERSAL_ACCURACY_PROFILE,
    PATH_PERSISTENCE_PROFILE,
    REGIME_ROBUST_ACCURACY_PROFILE,
    REGIME_ROBUST_FEATURE_CANDIDATE,
    REGIME_ROBUST_RECENCY_CANDIDATE,
    REGIME_ROBUST_REGULARIZED_CANDIDATE,
    CalibrationBand,
    PersistenceBenchmarkConfig,
    load_persistence_benchmark_config,
    persistence_config_to_dict,
    persistence_row_weight_schedule,
    walk_forward_validation_windows,
)
from .prewindow_features import (
    PREWINDOW_MODEL_FEATURES,
    build_prewindow_features,
    join_prewindow_features,
)
from .provenance import runtime_provenance

PERSISTENCE_BENCHMARK_SCHEMA_VERSION = "btc-path-persistence-benchmark-v1"
SAVED_POLICY_PROBABILITY_SCHEMA_VERSION = "btc-saved-policy-probabilities-v1"
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
class FixedHistogramParameters:
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float


@dataclass(frozen=True)
class CandidateProfile:
    name: str
    target_kind: Literal["outcome_up", "path_persistence"]
    feature_kind: Literal[
        "core",
        "core_boundary",
        "core_boundary_reversal",
        "core_mature_reversal",
        "core_regime_reversal",
        "core_prewindow",
    ]
    calibration_kind: Literal["global_platt", "time_banded_platt"]
    recency_half_life_days: float | None = None
    fixed_histogram_parameters: FixedHistogramParameters | None = None


@dataclass(frozen=True)
class WalkForwardFoldRoles:
    fit_start: datetime
    fit_end_exclusive: datetime
    calibration_start: datetime
    calibration_end_exclusive: datetime
    policy_start: datetime
    policy_end_exclusive: datetime
    validation_start: datetime
    validation_end_exclusive: datetime

    def as_dict(self) -> dict[str, dict[str, str]]:
        return {
            "fit": {
                "start": self.fit_start.isoformat(),
                "end_exclusive": self.fit_end_exclusive.isoformat(),
            },
            "calibration": {
                "start": self.calibration_start.isoformat(),
                "end_exclusive": self.calibration_end_exclusive.isoformat(),
            },
            "policy": {
                "start": self.policy_start.isoformat(),
                "end_exclusive": self.policy_end_exclusive.isoformat(),
            },
            "validation": {
                "start": self.validation_start.isoformat(),
                "end_exclusive": self.validation_end_exclusive.isoformat(),
            },
        }


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
        CandidateProfile(
            "histogram_path_persistence_time_calibrated_60_120",
            "path_persistence",
            "core_prewindow",
            "time_banded_platt",
        ),
        CandidateProfile(
            "histogram_path_persistence_time_calibrated_90_120",
            "path_persistence",
            "core_prewindow",
            "time_banded_platt",
        ),
        CandidateProfile(
            BOUNDARY_ALIGNMENT_CANDIDATE,
            "outcome_up",
            "core_boundary",
            "time_banded_platt",
        ),
        CandidateProfile(
            BOUNDARY_REVERSAL_ACCURACY_CANDIDATE,
            "path_persistence",
            "core_boundary_reversal",
            "time_banded_platt",
        ),
        CandidateProfile(
            MATURE_REVERSAL_ACCURACY_CANDIDATE,
            "outcome_up",
            "core_mature_reversal",
            "global_platt",
        ),
        CandidateProfile(
            REGIME_ROBUST_RECENCY_CANDIDATE,
            "outcome_up",
            "core_mature_reversal",
            "global_platt",
            recency_half_life_days=28.0,
        ),
        CandidateProfile(
            REGIME_ROBUST_FEATURE_CANDIDATE,
            "outcome_up",
            "core_regime_reversal",
            "global_platt",
        ),
        CandidateProfile(
            REGIME_ROBUST_REGULARIZED_CANDIDATE,
            "outcome_up",
            "core_mature_reversal",
            "global_platt",
            fixed_histogram_parameters=FixedHistogramParameters(
                learning_rate=0.05,
                max_iter=160,
                max_leaf_nodes=15,
                min_samples_leaf=370,
                l2_regularization=2.0,
            ),
        ),
        CandidateProfile(
            FOLD_ROBUST_FREQUENCY_CANDIDATE,
            "outcome_up",
            "core_prewindow",
            "time_banded_platt",
        ),
    )
}


def configured_validation_windows(
    config: PersistenceBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> tuple[tuple[datetime, datetime], ...]:
    if config.profile == REGIME_ROBUST_ACCURACY_PROFILE:
        return walk_forward_validation_windows(config)
    return core_config.split.validation_windows


def rolling_walk_forward_fold_roles(
    config: PersistenceBenchmarkConfig,
    core_config: CoreTrainingConfig,
    fold_index: int,
) -> WalkForwardFoldRoles:
    if config.profile != REGIME_ROBUST_ACCURACY_PROFILE:
        raise ValueError("rolling fold roles require the regime-robust accuracy profile")
    if (
        config.rolling_calibration_days is None
        or config.rolling_policy_days is None
    ):
        raise ValueError("rolling calibration and policy durations are required")
    validation_start, validation_end = configured_validation_windows(
        config,
        core_config,
    )[fold_index]
    policy_start = validation_start - timedelta(days=config.rolling_policy_days)
    calibration_start = policy_start - timedelta(
        days=config.rolling_calibration_days
    )
    roles = WalkForwardFoldRoles(
        fit_start=core_config.split.development_start,
        fit_end_exclusive=calibration_start,
        calibration_start=calibration_start,
        calibration_end_exclusive=policy_start,
        policy_start=policy_start,
        policy_end_exclusive=validation_start,
        validation_start=validation_start,
        validation_end_exclusive=validation_end,
    )
    _validate_walk_forward_fold_roles(roles)
    return roles


def _validate_walk_forward_fold_roles(roles: WalkForwardFoldRoles) -> None:
    ordered = (
        roles.fit_start,
        roles.fit_end_exclusive,
        roles.calibration_start,
        roles.calibration_end_exclusive,
        roles.policy_start,
        roles.policy_end_exclusive,
        roles.validation_start,
        roles.validation_end_exclusive,
    )
    if ordered != tuple(sorted(ordered)):
        raise ValueError("rolling walk-forward fold roles must be chronological")
    if (
        roles.fit_start >= roles.fit_end_exclusive
        or roles.calibration_start >= roles.calibration_end_exclusive
        or roles.policy_start >= roles.policy_end_exclusive
        or roles.validation_start >= roles.validation_end_exclusive
    ):
        raise ValueError("rolling walk-forward fold roles must have positive ranges")
    if (
        roles.fit_end_exclusive != roles.calibration_start
        or roles.calibration_end_exclusive != roles.policy_start
        or roles.policy_end_exclusive != roles.validation_start
    ):
        raise ValueError("rolling walk-forward fold roles must be disjoint and contiguous")


def run_persistence_benchmark(
    config: PersistenceBenchmarkConfig,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    _configure_compute(config)
    core_config = load_core_config(config.core_config)
    feature_metadata = validate_core_feature_cache(core_config, "pre_holdout")
    _assert_locked_development_range(
        core_config,
        enforce_external_holdout_contract=(config.profile == PATH_PERSISTENCE_PROFILE),
    )
    configured_profiles = tuple(
        CANDIDATE_PROFILES[name] for name in config.candidate_names
    )
    if any(profile.feature_kind == "core_prewindow" for profile in configured_profiles):
        prewindow_metadata = build_prewindow_features(
            core_config,
            config.prewindow_features,
            force=force,
        )
    else:
        prewindow_metadata = {
            "status": "not_required",
            "reason": "the configured candidate matrix has no pre-window feature consumer",
        }
    execution_config = _execution_config(config)
    execution_manifest = load_execution_evidence_manifest(execution_config)
    validation_windows = configured_validation_windows(config, core_config)

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
    probability_evidence = write_saved_probability_manifest(
        run_dir,
        candidate_results,
        config,
    )
    scored_frames = {name: result.pop("scored_rows") for name, result in candidate_results.items()}
    for result in candidate_results.values():
        result.pop("policy_scored_rows")
        result.pop("validation_probability_rows")
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
                f"{len(validation_windows)}-fold chronological "
                f"{validation_windows[0][0].date().isoformat()} through "
                f"{validation_windows[-1][1].date().isoformat()} "
                "development evidence; historical labels consumed; cached compact-book "
                f"execution evidence intersected over [{execution_manifest['range_start']}, "
                f"{execution_manifest['range_end']})"
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
    ranked_candidates: tuple[str, ...] = ()
    development_attempts: list[dict[str, Any]] = []
    if config.profile == REGIME_ROBUST_ACCURACY_PROFILE:
        ranked_candidates = _ranked_regime_robust_candidates(
            benchmark,
            candidate_results,
        )
        oof_finalist = ranked_candidates[0] if ranked_candidates else None
        finalist, freeze_record, development_attempts = (
            _fit_ranked_development_candidates(
                ranked_candidates,
                config,
                core_config,
                run_dir,
            )
        )
    else:
        finalist = _select_finalist(benchmark, candidate_results, config)
        oof_finalist = finalist
        freeze_record = _fit_development_finalist(
            finalist,
            config,
            core_config,
            run_dir,
        )

    data_evidence = {
        "training_range": {
            "start": core_config.data.range_start.isoformat(),
            "end_exclusive": core_config.data.range_end.isoformat(),
            "calendar_days": (core_config.data.range_end - core_config.data.range_start).days,
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
                "strict-both-side cached execution evidence intersected by market and "
                f"timestamp over [{execution_manifest['range_start']}, "
                f"{execution_manifest['range_end']})"
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
        "saved_probability_evidence": probability_evidence,
    }
    if config.profile == PATH_PERSISTENCE_PROFILE:
        data_evidence["independent_holdout"] = {
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
        }
    if config.profile == BOUNDARY_REVERSAL_ACCURACY_PROFILE:
        runtime_contract_gap = (
            "the challenger's path-persistence target conversion, time-banded "
            "calibration, and 106-feature boundary-reversal schema are not "
            "runtime-v1 contracts"
        )
    elif config.profile == MATURE_REVERSAL_ACCURACY_PROFILE:
        runtime_contract_gap = (
            "the challenger's 71-feature mature-reversal schema is not a "
            "runtime-v1 contract; its direct outcome target and global Platt "
            "calibration remain runtime-v1 compatible"
        )
    elif config.profile == REGIME_ROBUST_ACCURACY_PROFILE:
        runtime_contract_gap = (
            "each regime-robust challenger requires its own runtime feature and "
            "calibration compatibility review"
        )
    else:
        runtime_contract_gap = (
            "path-persistence target conversion, time-banded calibration, "
            "and pre-window features are not runtime-v1 contracts"
        )
    deployment_reasons = [
        runtime_contract_gap,
        "this benchmark does not modify Rust, images, processes, playbooks, or adapters",
    ]
    if config.profile == PATH_PERSISTENCE_PROFILE:
        deployment_reasons.insert(
            0,
            ("July 21-August 4 outcome labels and directional features remain untouched"),
        )
    training_selection: dict[str, Any] = {
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
    }
    if config.profile == REGIME_ROBUST_ACCURACY_PROFILE:
        training_selection.update(
            {
                "oof_finalist": oof_finalist,
                "oof_qualified_candidates": list(ranked_candidates),
                "development_attempts": development_attempts,
            }
        )
    benchmark.update(
        {
            "run_schema_version": PERSISTENCE_BENCHMARK_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "configuration": persistence_config_to_dict(config),
            "evaluation_note": config.evaluation_note,
            "runtime_provenance": runtime_provenance(config.package_root),
            "data_evidence": data_evidence,
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
            "training_selection": training_selection,
            "deployment": {
                "status": "blocked",
                "action": "training evidence only; no runtime export or deployment",
                "reasons": deployment_reasons,
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
            "benchmark_passed_candidates": benchmark["benchmark_passed_candidates"],
            "holdout_accessed": False,
            "holdout_outcome_labels_accessed": False,
            "holdout_directional_features_accessed": False,
            "holdout_book_quality_diagnostics_accessed": False,
            "runtime_changed": False,
        },
    )
    return run_dir, benchmark


def write_saved_probability_manifest(
    run_dir: Path,
    candidate_results: dict[str, dict[str, Any]],
    config: PersistenceBenchmarkConfig,
) -> dict[str, Any]:
    destination = run_dir / "saved-policy-probabilities"
    destination.mkdir(parents=True, exist_ok=False)
    candidate_records: dict[str, Any] = {}
    for candidate_name in config.candidate_names:
        result = candidate_results[candidate_name]
        policy_rows = result["policy_scored_rows"]
        validation_rows = result["validation_probability_rows"]
        fold_records = []
        fold_indexes = sorted(
            set(policy_rows["fold_index"].unique().to_list())
            | set(validation_rows["fold_index"].unique().to_list())
        )
        for fold_index in fold_indexes:
            policy_fold = policy_rows.filter(pl.col("fold_index") == fold_index)
            validation_fold = validation_rows.filter(pl.col("fold_index") == fold_index)
            _validate_probability_evidence_rows(
                policy_fold,
                candidate_name,
                int(fold_index),
                "policy_selection",
            )
            _validate_probability_evidence_rows(
                validation_fold,
                candidate_name,
                int(fold_index),
                "validation",
            )
            policy_end = policy_fold["window_start"].max()
            validation_start = validation_fold["window_start"].min()
            if policy_end >= validation_start:
                raise RuntimeError(
                    f"{candidate_name} fold {fold_index} policy rows are not "
                    "strictly earlier than validation"
                )
            candidate_dir = destination / candidate_name
            candidate_dir.mkdir(parents=True, exist_ok=True)
            policy_path = candidate_dir / f"fold-{int(fold_index):02d}-policy-selection.parquet"
            validation_path = candidate_dir / f"fold-{int(fold_index):02d}-validation.parquet"
            _write_parquet_atomic(policy_fold, policy_path)
            _write_parquet_atomic(validation_fold, validation_path)
            fold_records.append(
                {
                    "fold_index": int(fold_index),
                    "causal_order_verified": True,
                    "policy_selection": _probability_file_record(
                        policy_path,
                        policy_fold,
                        destination,
                    ),
                    "validation": _probability_file_record(
                        validation_path,
                        validation_fold,
                        destination,
                    ),
                }
            )
        profile = CANDIDATE_PROFILES[candidate_name]
        schedule = persistence_row_weight_schedule(config, candidate_name)
        candidate_records[candidate_name] = {
            "target_kind": profile.target_kind,
            "feature_kind": profile.feature_kind,
            "calibration_kind": profile.calibration_kind,
            "row_weight_schedule": asdict(schedule),
            "folds": fold_records,
        }
    manifest = {
        "schema_version": SAVED_POLICY_PROBABILITY_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_benchmark_profile": config.profile,
        "source_config": str(config.source_path),
        "source_config_sha256": file_sha256(config.source_path),
        "control_candidate": config.control_candidate,
        "candidate_names": list(config.candidate_names),
        "fold_count": len(
            next(iter(candidate_records.values()))["folds"] if candidate_records else ()
        ),
        "causal_contract": (
            "thresholds may be selected only from each fold's policy-selection "
            "rows; the corresponding validation rows may be scored exactly once"
        ),
        "candidates": candidate_records,
    }
    manifest_path = destination / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return {
        "schema_version": SAVED_POLICY_PROBABILITY_SCHEMA_VERSION,
        "manifest_path": str(manifest_path.relative_to(run_dir)),
        "manifest_sha256": file_sha256(manifest_path),
        "candidate_count": len(candidate_records),
        "fold_count": manifest["fold_count"],
    }


def _validate_probability_evidence_rows(
    rows: pl.DataFrame,
    candidate_name: str,
    fold_index: int,
    role: str,
) -> None:
    required = {
        "candidate",
        "fold_index",
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "probability_up",
        "confidence",
        "predicted_up",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise RuntimeError(f"{role} probability evidence is missing columns: " + ", ".join(missing))
    if rows.is_empty():
        raise RuntimeError(f"{candidate_name} fold {fold_index} has no {role} probability rows")
    if set(rows["candidate"].unique().to_list()) != {candidate_name}:
        raise RuntimeError(f"{role} probability candidate identity changed")
    if set(rows["fold_index"].unique().to_list()) != {fold_index}:
        raise RuntimeError(f"{role} probability fold identity changed")
    keys = ["market_id", "observed_at", "seconds_elapsed"]
    if rows.select(keys).is_duplicated().any():
        raise RuntimeError(f"{role} probability evidence contains duplicate rows")
    probability = rows["probability_up"].to_numpy()
    if not np.isfinite(probability).all() or ((probability < 0.0) | (probability > 1.0)).any():
        raise RuntimeError(f"{role} probability evidence contains invalid values")


def _write_parquet_atomic(frame: pl.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    frame.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )
    os.replace(temporary, destination)


def _probability_file_record(
    path: Path,
    frame: pl.DataFrame,
    manifest_root: Path,
) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(manifest_root)),
        "sha256": file_sha256(path),
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "window_start": frame["window_start"].min().isoformat(),
        "window_end": frame["window_start"].max().isoformat(),
        "observed_at_start": frame["observed_at"].min().isoformat(),
        "observed_at_end": frame["observed_at"].max().isoformat(),
    }


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
        mask = (elapsed >= band.start_second) & (elapsed < band.end_second_exclusive)
        output[mask] = calibrators.calibrators[band.name].probability(raw_logit[mask])
    if not np.isfinite(output).all():
        raise RuntimeError("time calibration bands do not cover every scored row")
    return output


def load_execution_evidence(config: ExecutionEvidenceConfig) -> pl.DataFrame:
    manifest = load_execution_evidence_manifest(config)
    files = [config.output_dir / partition["path"] for partition in manifest["partitions"]]
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
        pl.col("strict_both_side_eligible").is_not_null().alias("execution_evidence_available"),
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
        raise ValueError("selected execution comparison is missing columns: " + ", ".join(missing))
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
            for fold_index in range(
                len(configured_validation_windows(config, core_config))
            )
        ]
    return _aggregate_candidate(profile, folds, config, core_config)


def _evaluate_fold(
    frame: pl.DataFrame,
    profile: CandidateProfile,
    fold_index: int,
    config: PersistenceBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    if profile.name == FOLD_ROBUST_FREQUENCY_CANDIDATE:
        return _evaluate_fold_robust_agreement(
            frame,
            profile,
            fold_index,
            config,
            core_config,
        )
    fold_roles: WalkForwardFoldRoles | None = None
    if config.profile == REGIME_ROBUST_ACCURACY_PROFILE:
        fold_roles = rolling_walk_forward_fold_roles(
            config,
            core_config,
            fold_index,
        )
        validation_start = fold_roles.validation_start
        validation_end = fold_roles.validation_end_exclusive
        fit_frame = _candidate_eligible_frame(
            range_frame(
                frame,
                fold_roles.fit_start,
                fold_roles.fit_end_exclusive,
            ),
            profile,
        )
        calibration_frame = _candidate_eligible_frame(
            range_frame(
                frame,
                fold_roles.calibration_start,
                fold_roles.calibration_end_exclusive,
            ),
            profile,
        )
        policy_frame = _candidate_eligible_frame(
            range_frame(
                frame,
                fold_roles.policy_start,
                fold_roles.policy_end_exclusive,
            ),
            profile,
        )
        validation_source = range_frame(frame, validation_start, validation_end)
        universal_validation = _path_eligible_frame(validation_source)
        validation = _candidate_eligible_frame(
            validation_source,
            profile,
        )
    else:
        validation_start, validation_end = core_config.split.validation_windows[fold_index]
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
    spec = _candidate_spec(profile, config)
    started = time.perf_counter()
    model, tuning = _fit_profile_model(
        _training_target_frame(fit_frame, profile),
        profile,
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
    policy_target_probability = calibrated_target_probability(
        model,
        calibrators,
        policy_frame,
    )
    policy_probability = target_probability_to_up(
        policy_frame,
        policy_target_probability,
        profile.target_kind,
    )
    policy_scored = _probability_rows(
        policy_frame,
        policy_probability,
        policy_target_probability,
        profile,
        fold_index,
    )
    thresholds = threshold_table(
        policy_frame,
        policy_probability,
        core_config.model,
    )
    minimum_markets = max(
        50,
        math.ceil(policy_frame["market_id"].n_unique() * core_config.gates.minimum_coverage),
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
    validation_probability_rows = _probability_rows(
        validation,
        validation_probability_up,
        validation_target_probability,
        profile,
        fold_index,
    )
    scored = _apply_threshold_policy(
        validation_probability_rows,
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
    result = {
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
        "policy_scored_rows": policy_scored,
        "validation_probability_rows": validation_probability_rows,
        "scored_rows": scored,
        "selected_rows": selected,
    }
    if fold_roles is not None:
        result["fold_role_boundaries"] = fold_roles.as_dict()
    return result


def _evaluate_fold_robust_agreement(
    frame: pl.DataFrame,
    profile: CandidateProfile,
    fold_index: int,
    config: PersistenceBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    control_profile = CANDIDATE_PROFILES["histogram_enriched"]
    fold_roles: WalkForwardFoldRoles | None = None
    if config.profile == REGIME_ROBUST_ACCURACY_PROFILE:
        fold_roles = rolling_walk_forward_fold_roles(
            config,
            core_config,
            fold_index,
        )
        validation_start = fold_roles.validation_start
        validation_end = fold_roles.validation_end_exclusive
        validation_source = range_frame(frame, validation_start, validation_end)
        universal_validation = _path_eligible_frame(validation_source)
        challenger_validation = _candidate_eligible_frame(validation_source, profile)
        challenger_fit = _candidate_eligible_frame(
            range_frame(
                frame,
                fold_roles.fit_start,
                fold_roles.fit_end_exclusive,
            ),
            profile,
        )
        challenger_calibration = _candidate_eligible_frame(
            range_frame(
                frame,
                fold_roles.calibration_start,
                fold_roles.calibration_end_exclusive,
            ),
            profile,
        )
        challenger_policy = _candidate_eligible_frame(
            range_frame(
                frame,
                fold_roles.policy_start,
                fold_roles.policy_end_exclusive,
            ),
            profile,
        )
        control_fit = _candidate_eligible_frame(
            range_frame(
                frame,
                fold_roles.fit_start,
                fold_roles.fit_end_exclusive,
            ),
            control_profile,
        )
        control_calibration = _candidate_eligible_frame(
            range_frame(
                frame,
                fold_roles.calibration_start,
                fold_roles.calibration_end_exclusive,
            ),
            control_profile,
        )
    else:
        validation_start, validation_end = core_config.split.validation_windows[fold_index]
        history_source = range_frame(
            frame,
            core_config.split.development_start,
            validation_start,
        )
        validation_source = range_frame(frame, validation_start, validation_end)
        universal_validation = _path_eligible_frame(validation_source)
        challenger_history = _candidate_eligible_frame(history_source, profile)
        challenger_validation = _candidate_eligible_frame(validation_source, profile)
        challenger_fit, challenger_calibration, challenger_policy = chronological_subsplit(
            challenger_history,
            fit_fraction=0.70,
            calibration_fraction=0.15,
        )

        control_history = _candidate_eligible_frame(history_source, control_profile)
        control_fit, control_calibration, _ = chronological_subsplit(
            control_history,
            fit_fraction=0.70,
            calibration_fraction=0.15,
        )
    challenger_spec = _candidate_spec(profile, config)
    control_spec = _candidate_spec(control_profile, config)
    started = time.perf_counter()
    challenger_model, tuning = _tune_and_fit_fold_robust_model(
        _training_target_frame(challenger_fit, profile),
        challenger_spec,
        core_config,
    )
    control_model, control_tuning = tune_and_fit_model(
        _training_target_frame(control_fit, control_profile),
        control_spec,
        core_config,
    )
    challenger_calibrators, calibration_evidence = _fit_calibrators(
        challenger_model,
        challenger_calibration,
        profile,
        config,
        core_config,
        challenger_spec,
    )
    control_calibrators, _ = _fit_calibrators(
        control_model,
        control_calibration,
        control_profile,
        config,
        core_config,
        control_spec,
    )

    policy_control_probability = calibrated_target_probability(
        control_model,
        control_calibrators,
        challenger_policy,
    )
    policy_probability = _fold_robust_agreement_probability(
        policy_control_probability,
        calibrated_target_probability(
            challenger_model, challenger_calibrators, challenger_policy
        ),
    )
    _assert_control_direction_preserved(
        policy_control_probability,
        policy_probability,
        role="policy-selection",
    )
    policy_scored = _probability_rows(
        challenger_policy,
        policy_probability,
        policy_probability,
        profile,
        fold_index,
    )
    thresholds = threshold_table(
        challenger_policy,
        policy_probability,
        core_config.model,
    )
    minimum_markets = max(
        50,
        math.ceil(
            challenger_policy["market_id"].n_unique()
            * core_config.gates.minimum_coverage
        ),
    )
    threshold, threshold_qualified = choose_threshold(
        thresholds,
        core_config.gates,
        minimum_markets=minimum_markets,
    )
    validation_control_probability = calibrated_target_probability(
        control_model,
        control_calibrators,
        challenger_validation,
    )
    validation_probability = _fold_robust_agreement_probability(
        validation_control_probability,
        calibrated_target_probability(
            challenger_model, challenger_calibrators, challenger_validation
        ),
    )
    _assert_control_direction_preserved(
        validation_control_probability,
        validation_probability,
        role="validation",
    )
    validation_probability_rows = _probability_rows(
        challenger_validation,
        validation_probability,
        validation_probability,
        profile,
        fold_index,
    )
    scored = _apply_threshold_policy(validation_probability_rows, threshold)
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
    result = {
        "candidate": profile.name,
        "fold_index": fold_index,
        "fit_range_start": challenger_fit["window_start"].min().isoformat(),
        "fit_range_end": challenger_fit["window_start"].max().isoformat(),
        "calibration_range_start": (
            challenger_calibration["window_start"].min().isoformat()
        ),
        "calibration_range_end": (
            challenger_calibration["window_start"].max().isoformat()
        ),
        "policy_range_start": challenger_policy["window_start"].min().isoformat(),
        "policy_range_end": challenger_policy["window_start"].max().isoformat(),
        "validation_range_start": validation_start.isoformat(),
        "validation_range_end": validation_end.isoformat(),
        "eligible_markets": eligible,
        "confidence_threshold": threshold,
        "threshold_qualified": threshold_qualified,
        "threshold_history": thresholds,
        "tuning": {
            **tuning,
            "control_selected_hyperparameters": control_tuning[
                "selected_hyperparameters"
            ],
            "composition": {
                "kind": "control_direction_auxiliary_agreement_boost",
                "agreement_boost": 0.5,
                "control_direction_preserved": True,
                "checkpoint_nonregression_by_construction": True,
            },
        },
        "calibration": calibration_evidence,
        "metrics": metrics,
        "baseline": baseline_metrics(selected, eligible_markets=eligible),
        "paired": paired,
        "bootstrap": bootstrap,
        "timing": first_crossing_timing(selected, eligible_markets=eligible),
        "elapsed_seconds": time.perf_counter() - started,
        "direction_preservation": {
            "control_candidate": control_profile.name,
            "all_scored_directions_preserved": True,
            "agreement_only_confidence_boost": 0.5,
        },
        "policy_scored_rows": policy_scored,
        "validation_probability_rows": validation_probability_rows,
        "scored_rows": scored,
        "selected_rows": selected,
    }
    if fold_roles is not None:
        result["fold_role_boundaries"] = fold_roles.as_dict()
    return result


def _tune_and_fit_fold_robust_model(
    frame: pl.DataFrame,
    spec: CandidateSpec,
    core_config: CoreTrainingConfig,
) -> tuple[FittedCoreModel, dict[str, Any]]:
    splits = _fold_robust_tuning_splits(frame)
    history: list[dict[str, Any]] = []
    best_parameters: dict[str, Any] | None = None
    best_rank: tuple[float, ...] | None = None
    for histogram_candidate in core_config.model.histogram_candidates:
        parameters = histogram_parameters(histogram_candidate)
        started = time.perf_counter()
        fold_records: list[dict[str, Any]] = []
        converged = True
        for robust_fold, (train, validation) in enumerate(splits):
            model = fit_model(train, spec, parameters, core_config)
            converged = converged and estimator_converged(model.estimator)
            probability = model.raw_probability(validation)
            checkpoint_rows = fixed_time_prediction_rows(validation, probability)
            checkpoint_metrics = []
            for second in (60, 90, 120, 180, 240):
                metrics = classification_metrics(
                    checkpoint_rows.filter(pl.col("seconds_elapsed") == second)
                )
                checkpoint_metrics.append(
                    {
                        "seconds_elapsed": second,
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "up_recall": metrics["up_recall"],
                        "down_recall": metrics["down_recall"],
                    }
                )
            direction_quality = [
                value
                for metrics in checkpoint_metrics
                for value in (
                    metrics["accuracy"],
                    metrics["balanced_accuracy"],
                    metrics["up_recall"],
                    metrics["down_recall"],
                )
            ]
            fold_records.append(
                {
                    "fold_index": robust_fold,
                    "training_markets": train["market_id"].n_unique(),
                    "validation_markets": validation["market_id"].n_unique(),
                    "worst_checkpoint_direction_metric": min(direction_quality),
                    "mean_brier_score": float(
                        np.mean(
                            (
                                probability
                                - validation["label_up"].cast(pl.Float64).to_numpy()
                            )
                            ** 2
                        )
                    ),
                    "checkpoints": checkpoint_metrics,
                }
            )
        rank = (
            min(
                record["worst_checkpoint_direction_metric"]
                for record in fold_records
            ),
            -max(record["mean_brier_score"] for record in fold_records),
            -float(
                np.mean(
                    [record["mean_brier_score"] for record in fold_records]
                )
            ),
        )
        history.append(
            {
                "hyperparameters": parameters,
                "selection_rank": list(rank),
                "worst_checkpoint_direction_metric": rank[0],
                "worst_fold_brier_score": -rank[1],
                "mean_fold_brier_score": -rank[2],
                "folds": fold_records,
                "fit_seconds": time.perf_counter() - started,
                "converged": converged,
            }
        )
        if converged and (best_rank is None or rank > best_rank):
            best_rank = rank
            best_parameters = parameters
    if best_parameters is None or best_rank is None:
        raise RuntimeError(f"no {spec.name} fold-robust hyperparameter candidate converged")
    final_model = fit_model(frame, spec, best_parameters, core_config)
    return final_model, {
        "selection_objective": (
            "maximize the worst exact-checkpoint direction metric across three "
            "expanding chronological training folds; minimize worst and mean "
            "Brier score only as tie-breakers"
        ),
        "selected_hyperparameters": best_parameters,
        "selection_rank": list(best_rank),
        "candidates": history,
        "optimizer_converged": estimator_converged(final_model.estimator),
        "validation_consumed_for_tuning": False,
    }


def _fold_robust_tuning_splits(
    frame: pl.DataFrame,
) -> tuple[tuple[pl.DataFrame, pl.DataFrame], ...]:
    markets = (
        frame.select("market_id", "window_start")
        .unique(subset=["market_id"])
        .sort("window_start")
    )
    boundaries = (
        (0.55, 0.70),
        (0.70, 0.85),
        (0.85, 1.00),
    )
    splits = []
    for training_fraction, validation_fraction in boundaries:
        training_end = max(1, int(markets.height * training_fraction))
        validation_end = min(
            markets.height,
            max(training_end + 1, int(markets.height * validation_fraction)),
        )
        training_ids = markets[:training_end]["market_id"]
        validation_ids = markets[training_end:validation_end]["market_id"]
        if validation_ids.is_empty():
            raise RuntimeError("fold-robust tuning leaves an empty validation fold")
        splits.append(
            (
                frame.filter(pl.col("market_id").is_in(training_ids.implode())),
                frame.filter(pl.col("market_id").is_in(validation_ids.implode())),
            )
        )
    return tuple(splits)


def _fold_robust_agreement_probability(
    control_probability: np.ndarray,
    auxiliary_probability: np.ndarray,
) -> np.ndarray:
    control = np.asarray(control_probability, dtype=np.float64)
    auxiliary = np.asarray(auxiliary_probability, dtype=np.float64)
    if control.shape != auxiliary.shape:
        raise ValueError("control and auxiliary probabilities must have the same shape")
    control_up = control >= 0.5
    auxiliary_up = auxiliary >= 0.5
    control_strength = np.abs(2.0 * control - 1.0)
    auxiliary_strength = np.abs(2.0 * auxiliary - 1.0)
    boosted_strength = control_strength + (
        0.5 * (1.0 - control_strength) * auxiliary_strength
    )
    strength = np.where(control_up == auxiliary_up, boosted_strength, control_strength)
    combined = np.where(control_up, 0.5 + strength / 2.0, 0.5 - strength / 2.0)
    return np.clip(combined, 0.0, 1.0)


def _assert_control_direction_preserved(
    control_probability: np.ndarray,
    candidate_probability: np.ndarray,
    *,
    role: str,
) -> None:
    control_up = np.asarray(control_probability) >= 0.5
    candidate_up = np.asarray(candidate_probability) >= 0.5
    if not np.array_equal(control_up, candidate_up):
        raise RuntimeError(
            f"fold-robust agreement candidate changed a {role} control direction"
        )


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
    policy_scored = pl.concat(
        [fold["policy_scored_rows"] for fold in folds],
        how="vertical_relaxed",
    ).sort(["fold_index", "observed_at", "market_id"])
    validation_probability_rows = pl.concat(
        [fold["validation_probability_rows"] for fold in folds],
        how="vertical_relaxed",
    ).sort(["fold_index", "observed_at", "market_id"])
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
        fold["paired"]["accuracy_uplift"] >= core_config.gates.minimum_same_time_path_uplift
        for fold in folds
    )
    qualified_threshold_folds = sum(bool(fold["threshold_qualified"]) for fold in folds)
    qualified_validation_folds = sum(
        _validation_fold_meets_absolute_gates(fold["metrics"], core_config)
        for fold in folds
    )
    early_rows = selected.filter(pl.col("seconds_elapsed") <= config.early_cutoff_second)
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
    hard_confident_errors = hard_confident_error_metrics(
        selected,
        eligible_markets=eligible,
        confidence_floor=config.hard_confidence_floor,
    )
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
                if key
                not in {
                    "policy_scored_rows",
                    "validation_probability_rows",
                    "scored_rows",
                    "selected_rows",
                    "threshold_history",
                }
            }
        )
    spec = _candidate_spec(profile, config)
    result = {
        "candidate": profile.name,
        "family": "histogram",
        "target_kind": profile.target_kind,
        "feature_kind": profile.feature_kind,
        "calibration_kind": profile.calibration_kind,
        "feature_count": len(spec.feature_names),
        "features": list(spec.feature_names),
        "row_weight_policy": spec.row_weight_policy,
        "row_weight_schedule": asdict(spec.row_weight_schedule),
        "recency_half_life_days": spec.recency_half_life_days,
        "estimator_weighting": {
            "within_market": "equal total scheduled weight per market",
            "recency_decay_applied": spec.recency_half_life_days is not None,
            "recency_half_life_days": spec.recency_half_life_days,
            "probability_calibration_recency_decay_applied": False,
        },
        "fixed_histogram_parameters": (
            asdict(profile.fixed_histogram_parameters)
            if profile.fixed_histogram_parameters is not None
            else None
        ),
        "folds": fold_summaries,
        "out_of_fold": metrics,
        "baseline": baseline_metrics(selected, eligible_markets=eligible),
        "paired": paired,
        "bootstrap": bootstrap,
        "timing": first_crossing_timing(selected, eligible_markets=eligible),
        "early": early,
        "path_behavior": path_behavior,
        "hard_confident_errors": hard_confident_errors,
        "calibration": [fold["calibration"] for fold in folds],
        "nonnegative_uplift_folds": nonnegative_folds,
        "qualified_threshold_folds": qualified_threshold_folds,
        "total_folds": len(folds),
        "passed_development": passed,
        "policy_scored_rows": policy_scored,
        "validation_probability_rows": validation_probability_rows,
        "scored_rows": scored,
    }
    if config.profile == REGIME_ROBUST_ACCURACY_PROFILE:
        result["qualified_validation_folds"] = qualified_validation_folds
    return result


def _validation_fold_meets_absolute_gates(
    metrics: dict[str, Any],
    core_config: CoreTrainingConfig,
) -> bool:
    gates = core_config.gates
    return bool(
        metrics["accuracy"] >= gates.target_accuracy
        and metrics["balanced_accuracy"] >= gates.target_balanced_accuracy
        and metrics["up_recall"] >= gates.minimum_direction_recall
        and metrics["down_recall"] >= gates.minimum_direction_recall
        and metrics["wilson_lower_95"] >= gates.target_wilson_lower
        and metrics["expected_calibration_error"] <= gates.maximum_ece
        and metrics["coverage"] >= gates.minimum_coverage
    )


def _candidate_spec(
    profile: CandidateProfile,
    config: PersistenceBenchmarkConfig,
) -> CandidateSpec:
    if profile.feature_kind == "core_boundary":
        features = tuple(CORE_BOUNDARY_ENRICHED_FEATURES)
    elif profile.feature_kind == "core_boundary_reversal":
        features = tuple(CORE_BOUNDARY_REVERSAL_ENRICHED_FEATURES)
    elif profile.feature_kind == "core_mature_reversal":
        features = tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)
    elif profile.feature_kind == "core_regime_reversal":
        features = tuple(CORE_REGIME_REVERSAL_ENRICHED_FEATURES)
    else:
        features = tuple(CORE_ENRICHED_FEATURES)
    if profile.feature_kind == "core_prewindow":
        features += tuple(PREWINDOW_MODEL_FEATURES)
    schedule = persistence_row_weight_schedule(config, profile.name)
    row_weight_policy = (
        EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY
        if schedule.start_second is not None
        else MARKET_EQUAL_ROW_WEIGHT_POLICY
    )
    return CandidateSpec(
        name=profile.name,
        family="histogram",
        feature_names=features,
        row_weight_policy=row_weight_policy,
        row_weight_schedule=schedule,
        recency_half_life_days=profile.recency_half_life_days,
    )


def _fit_profile_model(
    frame: pl.DataFrame,
    profile: CandidateProfile,
    spec: CandidateSpec,
    core_config: CoreTrainingConfig,
) -> tuple[FittedCoreModel, dict[str, Any]]:
    fixed = profile.fixed_histogram_parameters
    if fixed is None:
        return tune_and_fit_model(frame, spec, core_config)
    parameters = asdict(fixed)
    started = time.perf_counter()
    model = fit_model(frame, spec, parameters, core_config)
    converged = estimator_converged(model.estimator)
    if not converged:
        raise RuntimeError(
            f"{profile.name} fixed market-regularized estimator did not converge"
        )
    fit_seconds = time.perf_counter() - started
    return model, {
        "selection_objective": (
            "fixed pre-registered market-scale regularization ablation; "
            "no fold-local hyperparameter search"
        ),
        "selected_hyperparameters": parameters,
        "validation_log_loss": None,
        "candidates": [
            {
                "hyperparameters": parameters,
                "fit_seconds": fit_seconds,
                "converged": True,
            }
        ],
        "optimizer_converged": True,
        "hyperparameter_search_consumed": False,
    }


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
    eligible = frame.filter(pl.Series("_path_eligible", path_is_directionally_eligible(frame)))
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
        raise RuntimeError(f"calibration band {name} failed convergence/monotonicity")


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
    return _apply_threshold_policy(
        _probability_rows(
            frame,
            probability_up,
            probability_target,
            profile,
            fold_index,
        ),
        threshold,
    )


def _probability_rows(
    frame: pl.DataFrame,
    probability_up: np.ndarray,
    probability_target: np.ndarray,
    profile: CandidateProfile,
    fold_index: int,
) -> pl.DataFrame:
    return (
        scored_prediction_rows(frame, probability_up)
        .with_columns(
            pl.Series("probability_target", probability_target),
            pl.lit(profile.name).alias("candidate"),
            pl.lit(profile.target_kind).alias("target_kind"),
            pl.lit(profile.calibration_kind).alias("calibration_kind"),
            pl.lit(fold_index).cast(pl.Int32).alias("fold_index"),
            pl.lit(True).alias("model_eligible"),
        )
        .with_columns((pl.col("predicted_up") == pl.col("binance_sign_up")).alias("path_followed"))
    )


def _apply_threshold_policy(
    probability_rows: pl.DataFrame,
    threshold: float,
) -> pl.DataFrame:
    scored = probability_rows.with_columns(pl.lit(threshold).alias("selected_confidence_threshold"))
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


def hard_confident_error_metrics(
    rows: pl.DataFrame,
    *,
    eligible_markets: int,
    confidence_floor: float,
) -> dict[str, Any]:
    if not 0.5 <= confidence_floor <= 1.0:
        raise ValueError("hard-confidence floor must be between 0.5 and 1.0")
    if eligible_markets < 0:
        raise ValueError("eligible market count cannot be negative")
    if rows.height > eligible_markets:
        raise ValueError("selected rows cannot exceed the eligible market universe")
    if not rows.is_empty() and rows["market_id"].n_unique() != rows.height:
        raise ValueError("hard-confident-error evidence requires one row per market")
    incorrect = rows.filter(~pl.col("correct"))
    hard = incorrect.filter(pl.col("confidence") >= confidence_floor)
    selected_markets = rows.height
    hard_errors = hard.height
    return {
        "confidence_floor": confidence_floor,
        "eligible_markets": eligible_markets,
        "selected_markets": selected_markets,
        "hard_confident_error_markets": hard_errors,
        "hard_confident_error_exposure_rate": (
            hard_errors / eligible_markets if eligible_markets else 0.0
        ),
        "hard_confident_error_rate_selected": (
            hard_errors / selected_markets if selected_markets else 0.0
        ),
        "maximum_incorrect_confidence": (
            float(incorrect["confidence"].max()) if not incorrect.is_empty() else None
        ),
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
        if config.profile in {
            MATURE_REVERSAL_ACCURACY_PROFILE,
            REGIME_ROBUST_ACCURACY_PROFILE,
        }:
            _add_accuracy_uplift_gates(
                advance,
                candidate_results,
                config,
                core_config,
                candidate_name=name,
                configured_fold_count=len(
                    configured_validation_windows(config, core_config)
                ),
                require_each_validation_fold=(
                    config.profile == REGIME_ROBUST_ACCURACY_PROFILE
                ),
            )
            continue
        deferred = [
            check for check in advance["checks"] if check["name"] in DEFERRED_RUNTIME_CHECKS
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
        if config.profile == BOUNDARY_REVERSAL_ACCURACY_PROFILE:
            control_tail = candidate_results[config.control_candidate][
                "hard_confident_errors"
            ]
            candidate_tail = result["hard_confident_errors"]
            checks.append(
                _check(
                    "hard-confident-error eligible universe matches control",
                    candidate_tail["eligible_markets"],
                    "==",
                    control_tail["eligible_markets"],
                )
            )
            if control_tail["hard_confident_error_markets"] > 0:
                checks.append(
                    _check(
                        "minimum hard-confident error count reduction",
                        (
                            control_tail["hard_confident_error_markets"]
                            - candidate_tail["hard_confident_error_markets"]
                        ),
                        ">=",
                        config.minimum_hard_confident_error_count_reduction,
                    )
                )
            else:
                checks.append(
                    _check(
                        "no hard-confident errors when control has none",
                        candidate_tail["hard_confident_error_markets"],
                        "==",
                        0,
                    )
                )
            checks.append(
                _check(
                    "hard-confident error rate per selected trade does not regress",
                    (
                        control_tail["hard_confident_error_rate_selected"]
                        - candidate_tail["hard_confident_error_rate_selected"]
                    ),
                    ">=",
                    -config.maximum_hard_confident_error_selected_rate_regression,
                )
            )
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
    if config.profile in {
        MATURE_REVERSAL_ACCURACY_PROFILE,
        REGIME_ROBUST_ACCURACY_PROFILE,
    }:
        benchmark["advancement_contract"] = {
            "profile": config.profile,
            "optimization_target": "model_directional_decision_accuracy",
            "advancement_only": [
                "absolute aggregate accuracy standards",
                "aggregate accuracy, balanced-accuracy, direction-recall, and Wilson uplift",
                "eligible-market coverage non-regression",
                "hard-confident error count and selected-rate improvement",
                "qualified threshold in every chronological fold",
                *(
                    ["absolute accuracy standards in every validation fold"]
                    if config.profile == REGIME_ROBUST_ACCURACY_PROFILE
                    else []
                ),
            ],
            "diagnostic_only": [
                "early-entry metrics",
                "fixed-time checkpoint comparisons",
                "decision timing",
                "path-persistence uplift",
                "execution economics",
            ],
        }
    benchmark["benchmark_passed_candidates"] = [
        name
        for name in config.candidate_names
        if name != config.control_candidate
        and benchmark["candidates"][name]["advance"]["benchmark_passed"]
    ]
    benchmark["deployment_qualified_candidates"] = []


def _add_accuracy_uplift_gates(
    advance: dict[str, Any],
    candidate_results: dict[str, dict[str, Any]],
    config: PersistenceBenchmarkConfig,
    core_config: CoreTrainingConfig,
    *,
    candidate_name: str,
    configured_fold_count: int,
    require_each_validation_fold: bool,
) -> None:
    control = candidate_results[config.control_candidate]
    candidate = candidate_results[candidate_name]
    control_metrics = control["out_of_fold"]
    candidate_metrics = candidate["out_of_fold"]
    control_tail = control["hard_confident_errors"]
    candidate_tail = candidate["hard_confident_errors"]
    deferred = [
        check for check in advance["checks"] if check["name"] in DEFERRED_RUNTIME_CHECKS
    ]
    checks = [
        _check(
            "minimum accepted samples",
            candidate_metrics["markets"],
            ">=",
            config.minimum_common_markets,
        ),
        _check(
            "minimum accuracy",
            candidate_metrics["accuracy"],
            ">=",
            core_config.gates.target_accuracy,
        ),
        _check(
            "minimum balanced accuracy",
            candidate_metrics["balanced_accuracy"],
            ">=",
            core_config.gates.target_balanced_accuracy,
        ),
        _check(
            "minimum UP recall",
            candidate_metrics["up_recall"],
            ">=",
            core_config.gates.minimum_direction_recall,
        ),
        _check(
            "minimum DOWN recall",
            candidate_metrics["down_recall"],
            ">=",
            core_config.gates.minimum_direction_recall,
        ),
        _check(
            "minimum Wilson lower bound",
            candidate_metrics["wilson_lower_95"],
            ">=",
            core_config.gates.target_wilson_lower,
        ),
        _check(
            "maximum expected calibration error",
            candidate_metrics["expected_calibration_error"],
            "<=",
            core_config.gates.maximum_ece,
        ),
        _check(
            "minimum eligible-market coverage",
            candidate_metrics["coverage"],
            ">=",
            core_config.gates.minimum_coverage,
        ),
        _check(
            "minimum aggregate accuracy uplift",
            candidate_metrics["accuracy"] - control_metrics["accuracy"],
            ">=",
            config.minimum_accuracy_uplift,
        ),
        _check(
            "minimum aggregate balanced accuracy uplift",
            candidate_metrics["balanced_accuracy"]
            - control_metrics["balanced_accuracy"],
            ">=",
            config.minimum_balanced_accuracy_uplift,
        ),
        _check(
            "positive aggregate UP recall uplift",
            candidate_metrics["up_recall"] - control_metrics["up_recall"],
            ">",
            config.minimum_direction_recall_uplift,
        ),
        _check(
            "positive aggregate DOWN recall uplift",
            candidate_metrics["down_recall"] - control_metrics["down_recall"],
            ">",
            config.minimum_direction_recall_uplift,
        ),
        _check(
            "minimum aggregate Wilson lower-bound uplift",
            candidate_metrics["wilson_lower_95"]
            - control_metrics["wilson_lower_95"],
            ">=",
            config.minimum_wilson_lower_uplift,
        ),
        _check(
            "eligible-market coverage does not regress",
            candidate_metrics["coverage"] - control_metrics["coverage"],
            ">=",
            config.minimum_coverage_uplift,
        ),
        _check(
            "hard-confident-error eligible universe matches control",
            candidate_tail["eligible_markets"],
            "==",
            control_tail["eligible_markets"],
        ),
    ]
    if control_tail["hard_confident_error_markets"] > 0:
        checks.append(
            _check(
                "minimum hard-confident error count reduction",
                control_tail["hard_confident_error_markets"]
                - candidate_tail["hard_confident_error_markets"],
                ">=",
                config.minimum_hard_confident_error_count_reduction,
            )
        )
    else:
        checks.append(
            _check(
                "no hard-confident errors when control has none",
                candidate_tail["hard_confident_error_markets"],
                "==",
                0,
            )
        )
    checks.extend(
        (
            _check(
                "hard-confident error rate per selected trade does not regress",
                control_tail["hard_confident_error_rate_selected"]
                - candidate_tail["hard_confident_error_rate_selected"],
                ">=",
                -config.maximum_hard_confident_error_selected_rate_regression,
            ),
            _check(
                "frozen chronological fold count",
                candidate["total_folds"],
                "==",
                configured_fold_count,
            ),
            _check(
                "qualified threshold in every fold",
                candidate["qualified_threshold_folds"],
                "==",
                candidate["total_folds"],
            ),
        )
    )
    if require_each_validation_fold:
        checks.append(
            _check(
                "absolute accuracy standards in every validation fold",
                candidate["qualified_validation_folds"],
                "==",
                configured_fold_count,
            )
        )
    advance["checks"] = checks
    advance["deferred_runtime_checks"] = deferred
    advance["diagnostic_only"] = [
        "early",
        "timing",
        "fixed_checkpoints",
        "path_behavior",
        "paired_uplift",
        "bootstrap_uplift",
        "execution_economics",
    ]
    advance["benchmark_passed"] = all(check["passed"] for check in checks)
    advance["deployment_qualified"] = False


def _check(
    name: str,
    observed: float,
    operator: str,
    required: float,
) -> dict[str, Any]:
    if operator == ">=":
        passed = observed >= required
    elif operator == ">":
        passed = observed > required
    elif operator == "<=":
        passed = observed <= required
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
    if config.profile == FOLD_ROBUST_FREQUENCY_PROFILE:
        return None
    passing = benchmark["benchmark_passed_candidates"]
    if not passing:
        return None
    if config.profile in {
        BOUNDARY_REVERSAL_ACCURACY_PROFILE,
        MATURE_REVERSAL_ACCURACY_PROFILE,
        REGIME_ROBUST_ACCURACY_PROFILE,
    }:
        return max(passing, key=lambda name: _accuracy_finalist_rank(candidate_results[name]))
    return max(
        passing,
        key=lambda name: (
            candidate_results[name]["out_of_fold"]["coverage"],
            candidate_results[name]["out_of_fold"]["wilson_lower_95"],
            candidate_results[name]["out_of_fold"]["balanced_accuracy"],
            candidate_results[name]["out_of_fold"]["accuracy"],
            -(candidate_results[name]["timing"]["median_first_crossing_seconds"] or float("inf")),
        ),
    )


def _accuracy_finalist_rank(result: dict[str, Any]) -> tuple[float, ...]:
    median_crossing = result["timing"]["median_first_crossing_seconds"]
    return (
        -result["hard_confident_errors"]["hard_confident_error_exposure_rate"],
        -result["hard_confident_errors"]["hard_confident_error_rate_selected"],
        result["out_of_fold"]["accuracy"],
        result["out_of_fold"]["balanced_accuracy"],
        result["out_of_fold"]["wilson_lower_95"],
        result["out_of_fold"]["coverage"],
        -(median_crossing if median_crossing is not None else float("inf")),
    )


def _ranked_regime_robust_candidates(
    benchmark: dict[str, Any],
    candidate_results: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    passing = benchmark["benchmark_passed_candidates"]
    return tuple(
        sorted(
            passing,
            key=lambda name: _accuracy_finalist_rank(candidate_results[name]),
            reverse=True,
        )
    )


def _fit_ranked_development_candidates(
    ranked_candidates: tuple[str, ...],
    config: PersistenceBenchmarkConfig,
    core_config: CoreTrainingConfig,
    run_dir: Path,
) -> tuple[str | None, dict[str, Any] | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for rank, candidate_name in enumerate(ranked_candidates, start=1):
        record = _fit_development_finalist(
            candidate_name,
            config,
            core_config,
            run_dir,
        )
        if record is None:
            raise RuntimeError(
                f"ranked development candidate {candidate_name} produced no fit record"
            )
        attempts.append({"rank": rank, **record})
        if record["bundle_created"]:
            return candidate_name, record, attempts
    return None, None, attempts


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
    spec = _candidate_spec(profile, config)
    model, tuning = _fit_profile_model(
        _training_target_frame(development, profile),
        profile,
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
            math.ceil(policy_universe["market_id"].n_unique() * core_config.gates.minimum_coverage),
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
        and metrics["balanced_accuracy"] >= core_config.gates.target_balanced_accuracy
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
            "policy_passed": False,
            "metrics": metrics,
            "tuning": tuning,
            "calibration": calibration_evidence,
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
        "policy_passed": True,
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
        maximum_balanced_accuracy_regression=(config.maximum_balanced_accuracy_regression),
        maximum_direction_recall_regression=(config.maximum_direction_recall_regression),
        maximum_median_entry_seconds_regression=(config.maximum_median_entry_seconds_regression),
        minimum_mean_direct_edge_per_share=(config.minimum_mean_direct_edge_per_share),
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


def _assert_locked_development_range(
    config: CoreTrainingConfig,
    *,
    enforce_external_holdout_contract: bool = True,
) -> None:
    metadata = validate_core_feature_cache(config, "pre_holdout")
    contract = metadata.get("build_contract", {})
    if (
        datetime.fromisoformat(str(contract.get("range_start"))) != config.data.range_start
        or datetime.fromisoformat(str(contract.get("range_end"))) != config.data.range_end
    ):
        raise RuntimeError("development cache escapes the locked training range")
    if not enforce_external_holdout_contract:
        return
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
        raise RuntimeError("independent holdout feature cache exists before candidate freeze")


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
