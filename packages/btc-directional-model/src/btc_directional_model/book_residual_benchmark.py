from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from .benchmark_config import (
    EntryBenchmarkConfig,
    benchmark_config_to_dict,
)
from .book_residual import (
    BOOK_RESIDUAL_FEATURES,
    DirectionTimeCalibrator,
    ResidualBookModel,
    derive_strict_book_residual_features,
    fit_direction_time_calibrator,
    fit_residual_book_model,
    market_equal_weights,
)
from .core_benchmark import (
    BenchmarkEvidence,
    CandidatePolicy,
    benchmark_predictions,
)
from .core_benchmark_report import generate_benchmark_report
from .core_config import CoreTrainingConfig
from .core_evaluation import (
    choose_threshold,
    scored_prediction_rows,
    threshold_table,
)
from .core_extract import file_sha256, write_json_atomic
from .core_features import (
    CORE_ENRICHED_FEATURES,
    load_core_feature_frame,
    validate_core_feature_cache,
)
from .core_training import (
    CandidateSpec,
    FrozenTrainingBundle,
    fit_probability_calibrator,
    range_frame,
    tune_and_fit_model,
)
from .offline_challengers import derive_strict_book_feature_frame
from .provenance import runtime_provenance

BOOK_RESIDUAL_BENCHMARK_SCHEMA_VERSION = "btc-book-residual-benchmark-v1"
BOOK_RESIDUAL_FREEZE_SCHEMA_VERSION = "btc-book-residual-freeze-v1"
BOOK_RESIDUAL_SEALED_HOLDOUT_SCHEMA_VERSION = (
    "btc-book-residual-sealed-holdout-v1"
)


def run_book_residual_benchmark(
    config: EntryBenchmarkConfig,
    core_config: CoreTrainingConfig,
    run_id: str,
    run_dir: Path,
    *,
    force: bool,
) -> tuple[Path, dict[str, Any]]:
    """Train one compact book residual without opening the sealed holdout."""

    residual_split = config.residual_split
    residual_config = config.residual_model
    if residual_split is None or residual_config is None:
        raise RuntimeError("book-residual benchmark lost its residual configuration")

    # Imported lazily to keep the existing entry-benchmark module as the single
    # orchestration surface without creating a second CLI or runtime path.
    from .core_execution import extract_execution_evidence
    from .entry_benchmark import (
        OFFLINE_RUNTIME_CHECKS,
        _advancement_criteria,
        _attach_execution_evidence,
        _execution_config,
        _load_execution_evidence,
        _update_progress,
    )

    feature_metadata = validate_core_feature_cache(core_config, "pre_holdout")
    _validate_residual_core_range(core_config, config)
    _validate_oof_probability_artifact(config)
    _update_progress(run_dir, "extracting_residual_evidence", 0.08)

    execution_config = _execution_config(config)
    execution_manifest = extract_execution_evidence(
        execution_config,
        force=force,
    )
    execution_frame = _load_execution_evidence(
        execution_config,
        require_v2=True,
    )
    core_frame = load_core_feature_frame(core_config, "pre_holdout")
    execution_core = range_frame(
        core_frame,
        config.execution.range_start,
        config.execution.range_end,
    )
    strict_frame = derive_strict_book_feature_frame(
        execution_core,
        execution_frame,
        require_ten_share=True,
        include_deltas=True,
    )
    _update_progress(
        run_dir,
        "fitting_residual_offset",
        0.24,
        {
            "execution_rows": execution_manifest["totals"]["rows"],
            "strict_10_rows": execution_manifest["totals"][
                "strict_both_side_eligible_10_rows"
            ],
            "delta_ready_rows": strict_frame.height,
            "delta_ready_markets": strict_frame["market_id"].n_unique(),
        },
    )

    oof_frame = _load_oof_probabilities(config)
    oof_strict = _join_oof_strict_rows(strict_frame, oof_frame)
    residual_fit = range_frame(
        oof_strict,
        residual_split.residual_fit_start,
        residual_split.residual_fit_end,
    )
    lambda_selection = range_frame(
        oof_strict,
        residual_split.lambda_selection_start,
        residual_split.lambda_selection_end,
    )
    stability = range_frame(
        oof_strict,
        residual_split.stability_start,
        residual_split.stability_end,
    )
    _require_nonempty_residual_cohort("residual fit", residual_fit)
    _require_nonempty_residual_cohort("L2 selection", lambda_selection)
    _require_nonempty_residual_cohort("stability", stability)
    lambda_history, selected_l2 = _select_l2_strength(
        residual_fit,
        lambda_selection,
        residual_config.l2_candidates,
    )
    stability_evidence = _residual_fit_evidence(
        residual_fit,
        stability,
        l2_strength=selected_l2,
    )
    residual_final_frame = pl.concat(
        [residual_fit, lambda_selection, stability],
        how="vertical_relaxed",
    ).sort(["window_start", "seconds_elapsed", "market_id"])
    residual_model, residual_diagnostics = _fit_residual_from_joined(
        residual_final_frame,
        l2_strength=selected_l2,
    )

    _update_progress(
        run_dir,
        "fitting_universal_core",
        0.42,
        {
            "selected_l2": selected_l2,
            "residual_fit_rows": residual_final_frame.height,
            "residual_fit_markets": residual_final_frame[
                "market_id"
            ].n_unique(),
        },
    )
    core_fit = range_frame(
        core_frame,
        residual_split.core_fit_start,
        residual_split.core_fit_end,
    )
    core_calibration = range_frame(
        core_frame,
        residual_split.core_calibration_start,
        residual_split.core_calibration_end,
    )
    control_spec = CandidateSpec(
        config.benchmark.control_candidate,
        "histogram",
        tuple(CORE_ENRICHED_FEATURES),
    )
    core_model, core_tuning = tune_and_fit_model(
        core_fit,
        control_spec,
        core_config,
    )
    if not core_tuning.get("optimizer_converged"):
        raise RuntimeError("universal BTC core optimizer did not converge")
    core_probability_calibrator = fit_probability_calibrator(
        core_model,
        core_calibration,
        core_config,
        control_spec,
    )
    if not core_probability_calibrator.converged:
        raise RuntimeError(
            "universal BTC core probability calibrator did not converge"
        )
    core_bundle = FrozenTrainingBundle(
        model=core_model,
        calibrator=core_probability_calibrator,
        confidence_threshold=config.book_model.confidence_min,
    )

    calibration_core = range_frame(
        core_frame,
        residual_split.direction_time_calibration_start,
        residual_split.direction_time_calibration_end,
    )
    calibration_strict = range_frame(
        strict_frame,
        residual_split.direction_time_calibration_start,
        residual_split.direction_time_calibration_end,
    )
    calibration_route = _routed_raw_logits(
        calibration_core,
        calibration_strict,
        core_bundle,
        residual_model,
    )
    control_time_calibrator, control_time_diagnostics = (
        fit_direction_time_calibrator(
            calibration_route["core_raw_logit"],
            calibration_core["seconds_elapsed"].to_numpy(),
            calibration_core["label_up"].to_numpy(),
            calibration_core["market_id"].to_list(),
            identity_l2_strength=residual_config.calibration_l2,
            minimum_markets_per_cell=(
                residual_config.minimum_calibration_cell_markets
            ),
        )
    )
    strict_calibration_mask = calibration_route["strict_mask"]
    residual_time_calibrator, residual_time_diagnostics = (
        fit_direction_time_calibrator(
            calibration_route["residual_raw_logit"][
                strict_calibration_mask
            ],
            calibration_core["seconds_elapsed"].to_numpy()[
                strict_calibration_mask
            ],
            calibration_core["label_up"].to_numpy()[
                strict_calibration_mask
            ],
            calibration_core["market_id"].to_numpy()[
                strict_calibration_mask
            ],
            identity_l2_strength=residual_config.calibration_l2,
            minimum_markets_per_cell=(
                residual_config.minimum_calibration_cell_markets
            ),
        )
    )
    _require_calibrator_convergence(
        "universal BTC core direction/time",
        control_time_calibrator,
    )
    _require_calibrator_convergence(
        "strict book residual direction/time",
        residual_time_calibrator,
    )

    _update_progress(
        run_dir,
        "selecting_frozen_policies",
        0.62,
        {
            "calibration_markets": calibration_core["market_id"].n_unique(),
            "strict_calibration_markets": calibration_strict[
                "market_id"
            ].n_unique(),
        },
    )
    threshold_core = range_frame(
        core_frame,
        residual_split.threshold_selection_start,
        residual_split.threshold_selection_end,
    )
    threshold_strict = range_frame(
        strict_frame,
        residual_split.threshold_selection_start,
        residual_split.threshold_selection_end,
    )
    threshold_frames, threshold_route = _score_candidate_pair(
        threshold_core,
        threshold_strict,
        core_bundle,
        residual_model,
        control_time_calibrator,
        residual_time_calibrator,
        config,
    )
    threshold_model_config = replace(
        core_config.model,
        confidence_min=config.book_model.confidence_min,
        confidence_max=config.book_model.confidence_max,
        confidence_step=config.book_model.confidence_step,
    )
    threshold_core_config = replace(
        core_config,
        model=threshold_model_config,
    )
    thresholds: dict[str, float] = {}
    threshold_qualified: dict[str, bool] = {}
    threshold_histories: dict[str, list[dict[str, Any]]] = {}
    for candidate_name, frame in threshold_frames.items():
        probability = frame["probability_up"].to_numpy()
        history = threshold_table(
            threshold_core,
            probability,
            threshold_core_config.model,
        )
        selected, qualified = choose_threshold(
            history,
            threshold_core_config.gates,
            minimum_markets=config.gates.minimum_common_time_markets,
        )
        thresholds[candidate_name] = selected
        threshold_qualified[candidate_name] = qualified
        threshold_histories[candidate_name] = history

    freeze_path = run_dir / "book-residual-candidate-pair.joblib"
    joblib.dump(
        {
            "schema_version": BOOK_RESIDUAL_FREEZE_SCHEMA_VERSION,
            "core_bundle": core_bundle,
            "residual_model": residual_model,
            "control_time_calibrator": control_time_calibrator,
            "residual_time_calibrator": residual_time_calibrator,
            "thresholds": thresholds,
            "candidate_names": list(config.benchmark.candidate_names),
        },
        freeze_path,
        compress=3,
    )
    freeze_record = {
        "schema_version": BOOK_RESIDUAL_FREEZE_SCHEMA_VERSION,
        "candidate_pair_file": freeze_path.name,
        "candidate_pair_sha256": file_sha256(freeze_path),
        "configuration_sha256": file_sha256(config.source_path),
        "oof_probability_sha256": residual_config.oof_probability_sha256,
        "oof_benchmark_sha256": residual_config.oof_benchmark_sha256,
        "oof_run_id": residual_config.oof_run_id,
        "execution_manifest_sha256": file_sha256(
            execution_config.output_dir / "manifest.json"
        ),
        "candidate_names": list(config.benchmark.candidate_names),
        "thresholds": thresholds,
        "threshold_qualified": threshold_qualified,
        "sealed_holdout": {
            "range_start": residual_split.sealed_holdout_start.isoformat(),
            "range_end": residual_split.sealed_holdout_end.isoformat(),
            "status": "not_accessed",
        },
        "runtime_provenance": runtime_provenance(config.package_root),
    }
    freeze_manifest_path = run_dir / "freeze-manifest.json"
    write_json_atomic(freeze_manifest_path, freeze_record)
    freeze_evidence = {
        **freeze_record,
        "freeze_manifest_sha256": file_sha256(
            freeze_manifest_path
        ),
    }

    sealed_holdout_record = {
        "schema_version": BOOK_RESIDUAL_SEALED_HOLDOUT_SCHEMA_VERSION,
        "range_start": residual_split.sealed_holdout_start.isoformat(),
        "range_end": residual_split.sealed_holdout_end.isoformat(),
        "status": "sealed_not_accessed",
        "minimum_required_markets": config.gates.minimum_common_time_markets,
        "access_marker": None,
        "reason": (
            "aggregate-only coverage audit found 275 eligible markets, below "
            "the unchanged 500-market evaluation minimum"
        ),
    }
    sealed_holdout_path = run_dir / "sealed-holdout.json"
    write_json_atomic(sealed_holdout_path, sealed_holdout_record)

    # Only after the complete candidate pair and policies are frozen do we
    # score the later, already-consumed development diagnostic.
    policy_core = range_frame(
        core_frame,
        residual_split.policy_diagnostic_start,
        residual_split.policy_diagnostic_end,
    )
    policy_strict = range_frame(
        strict_frame,
        residual_split.policy_diagnostic_start,
        residual_split.policy_diagnostic_end,
    )
    policy_frames, policy_route = _score_candidate_pair(
        policy_core,
        policy_strict,
        core_bundle,
        residual_model,
        control_time_calibrator,
        residual_time_calibrator,
        config,
    )
    policy_frames = {
        name: _attach_execution_evidence(frame, execution_frame)
        for name, frame in policy_frames.items()
    }
    policies = {
        name: CandidatePolicy(
            confidence_threshold=thresholds[name],
            deployment_compatible=False,
        )
        for name in config.benchmark.candidate_names
    }
    eligible_market_ids = sorted(
        policy_core["market_id"].unique().to_list()
    )
    benchmark = benchmark_predictions(
        policy_frames,
        policies=policies,
        control_candidate=config.benchmark.control_candidate,
        evidence=BenchmarkEvidence(
            label=(
                "July 18-19 consumed chronological development diagnostic; "
                "the July 22 sealed cohort was not accessed"
            ),
            kind="development",
            independent=False,
        ),
        eligible_market_ids=eligible_market_ids,
        minimum_samples=config.gates.minimum_common_time_markets,
        minimum_executable_samples=config.gates.minimum_executable_markets,
        quantity=config.benchmark.quantity,
        criteria=_advancement_criteria(config),
    )
    strict_ablation = _strict_policy_ablation(
        policy_frames,
        policy_route["strict_keys"],
        policies,
        config,
    )
    candidate_name = config.benchmark.strict_book_candidate
    statistical_checks = [
        {
            "name": "residual threshold qualified before policy diagnostic",
            "observed": threshold_qualified[candidate_name],
            "operator": "=",
            "required": True,
            "passed": threshold_qualified[candidate_name],
        },
        *[
            check
            for check in benchmark["candidates"][candidate_name]["advance"][
                "checks"
            ]
            if check["name"] not in OFFLINE_RUNTIME_CHECKS
        ],
    ]
    selected = all(check["passed"] for check in statistical_checks)

    prediction_evidence: dict[str, Any] = {}
    for name, frame in policy_frames.items():
        destination = run_dir / f"{name}-predictions.parquet"
        frame.write_parquet(destination, compression="zstd")
        prediction_evidence[name] = {
            "path": destination.name,
            "sha256": file_sha256(destination),
            "rows": frame.height,
            "markets": frame["market_id"].n_unique(),
        }
    benchmark.update(
        {
            "run_schema_version": BOOK_RESIDUAL_BENCHMARK_SCHEMA_VERSION,
            "run_id": run_id,
            "configuration": benchmark_config_to_dict(config),
            "evaluation_note": config.benchmark.evaluation_note,
            "runtime_provenance": runtime_provenance(config.package_root),
            "data_evidence": {
                "core_features": feature_metadata,
                "execution": execution_manifest,
                "execution_manifest_sha256": freeze_record[
                    "execution_manifest_sha256"
                ],
                "cohorts": {
                    "residual_fit": _cohort_summary(residual_fit),
                    "lambda_selection": _cohort_summary(lambda_selection),
                    "stability": _cohort_summary(stability),
                    "direction_time_calibration": _paired_cohort_summary(
                        calibration_core,
                        calibration_strict,
                    ),
                    "threshold_selection": _paired_cohort_summary(
                        threshold_core,
                        threshold_strict,
                    ),
                    "policy_diagnostic": _paired_cohort_summary(
                        policy_core,
                        policy_strict,
                    ),
                },
                "sealed_holdout": sealed_holdout_record,
            },
            "training_evidence": {
                "book_residual": {
                    "feature_contract": {
                        "features": list(BOOK_RESIDUAL_FEATURES),
                        "strict_ten_share_required": True,
                        "exact_previous_five_second_row_required": True,
                        "invalid_book_route": "unchanged universal BTC core",
                        "quality_flags_are_features": False,
                        "provider_age_is_a_feature": False,
                    },
                    "oof_core": {
                        "candidate": residual_config.oof_candidate,
                        "path": str(residual_config.oof_probability_path),
                        "sha256": residual_config.oof_probability_sha256,
                        "benchmark_path": str(
                            residual_config.oof_benchmark_path
                        ),
                        "benchmark_sha256": (
                            residual_config.oof_benchmark_sha256
                        ),
                        "run_id": residual_config.oof_run_id,
                        "joined_rows": oof_strict.height,
                        "joined_markets": oof_strict["market_id"].n_unique(),
                    },
                    "lambda_selection": lambda_history,
                    "selected_l2": selected_l2,
                    "stability": stability_evidence,
                    "final_residual": {
                        "model": residual_model.to_dict(),
                        "diagnostics": residual_diagnostics.to_dict(),
                    },
                    "universal_core": {
                        "fit_rows": core_fit.height,
                        "fit_markets": core_fit["market_id"].n_unique(),
                        "calibration_rows": core_calibration.height,
                        "calibration_markets": core_calibration[
                            "market_id"
                        ].n_unique(),
                        "tuning": core_tuning,
                        "probability_calibrator": asdict(
                            core_probability_calibrator
                        ),
                    },
                    "direction_time_calibration": {
                        config.benchmark.control_candidate: (
                            control_time_diagnostics.to_dict()
                        ),
                        candidate_name: (
                            residual_time_diagnostics.to_dict()
                        ),
                    },
                    "threshold_selection": {
                        name: {
                            "threshold": thresholds[name],
                            "qualified": threshold_qualified[name],
                            "history": threshold_histories[name],
                        }
                        for name in config.benchmark.candidate_names
                    },
                    "routes": {
                        "calibration": calibration_route["diagnostics"],
                        "threshold_selection": threshold_route["diagnostics"],
                        "policy_diagnostic": policy_route["diagnostics"],
                    },
                    "freeze": freeze_evidence,
                    "predictions": prediction_evidence,
                },
            },
            "strict_row_ablation": strict_ablation,
            "book_residual_selection": {
                "status": (
                    "selected_for_sealed_holdout"
                    if selected
                    else "blocked_by_frozen_development_gates"
                ),
                "winner": candidate_name if selected else None,
                "checks": statistical_checks,
                "sealed_holdout_accessed": False,
                "deployment_authorized": False,
            },
            "deployment": {
                "status": "not_authorized",
                "candidates": [],
                "action": (
                    "no runtime bundle, Rust contract, trading process, image, "
                    "database write, or restart"
                ),
                "reasons": [
                    "the only scored result is consumed development evidence",
                    (
                        "the sealed July 22 cohort remains below the unchanged "
                        "500-market evaluation minimum"
                    ),
                    "native Rust residual inference is outside this training pass",
                ],
            },
        }
    )
    write_json_atomic(run_dir / "benchmark.json", benchmark)
    report_path = generate_benchmark_report(
        benchmark,
        run_dir / "report.html",
    )
    artifact_manifest = {
        "schema_version": BOOK_RESIDUAL_BENCHMARK_SCHEMA_VERSION,
        "benchmark_sha256": file_sha256(run_dir / "benchmark.json"),
        "report_sha256": file_sha256(report_path),
        "freeze_manifest_sha256": file_sha256(freeze_manifest_path),
        "sealed_holdout_sha256": file_sha256(sealed_holdout_path),
        "predictions": prediction_evidence,
    }
    write_json_atomic(run_dir / "artifact-manifest.json", artifact_manifest)
    _update_progress(
        run_dir,
        "book_residual_complete",
        1.0,
        {
            "report": report_path.name,
            "winner": benchmark["book_residual_selection"]["winner"],
            "sealed_holdout_accessed": False,
        },
    )
    return run_dir, benchmark


def _validate_residual_core_range(
    core_config: CoreTrainingConfig,
    config: EntryBenchmarkConfig,
) -> None:
    split = config.residual_split
    if split is None:
        raise AssertionError("residual core validation lost its split")
    if (
        split.core_fit_start < core_config.data.range_start
        or split.policy_diagnostic_end > core_config.data.range_end
    ):
        raise RuntimeError(
            "residual development cohorts escape the frozen universal BTC core"
        )
    if split.sealed_holdout_start < core_config.data.range_end:
        raise RuntimeError(
            "sealed residual holdout overlaps the loaded universal core cache"
        )


def _validate_oof_probability_artifact(config: EntryBenchmarkConfig) -> None:
    residual = config.residual_model
    if residual is None:
        raise AssertionError("OOF validation lost its residual config")
    for label, path, expected in (
        (
            "probability",
            residual.oof_probability_path,
            residual.oof_probability_sha256,
        ),
        (
            "benchmark",
            residual.oof_benchmark_path,
            residual.oof_benchmark_sha256,
        ),
    ):
        if not path.is_file():
            raise RuntimeError(f"residual OOF {label} artifact is missing: {path}")
        observed = file_sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"residual OOF {label} artifact checksum mismatch: "
                f"expected {expected}, observed {observed}"
            )


def _load_oof_probabilities(config: EntryBenchmarkConfig) -> pl.DataFrame:
    residual = config.residual_model
    if residual is None:
        raise AssertionError("OOF loading lost its residual config")
    frame = pl.read_parquet(residual.oof_probability_path)
    required = {
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "candidate",
        "fold_index",
        "probability_up",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "residual OOF artifact is missing columns: " + ", ".join(missing)
        )
    candidates = set(frame["candidate"].unique().to_list())
    if candidates != {residual.oof_candidate}:
        raise RuntimeError(
            "residual OOF artifact candidate does not match the pinned contract"
        )
    invalid_probability_rows = frame.filter(
        pl.col("probability_up").is_null()
        | ~pl.col("probability_up").is_finite()
        | (pl.col("probability_up") <= 0.0)
        | (pl.col("probability_up") >= 1.0)
    ).height
    if invalid_probability_rows:
        raise RuntimeError(
            "residual OOF probabilities must be finite and strictly within (0, 1)"
        )
    _validate_oof_training_provenance(config, frame)
    return frame.select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "fold_index",
        pl.col("probability_up").alias("core_probability_up"),
    )


def _validate_oof_training_provenance(
    config: EntryBenchmarkConfig,
    frame: pl.DataFrame,
) -> None:
    residual = config.residual_model
    if residual is None:
        raise AssertionError("OOF provenance validation lost its residual config")
    benchmark = json.loads(residual.oof_benchmark_path.read_text())
    if benchmark.get("run_id") != residual.oof_run_id:
        raise RuntimeError("residual OOF benchmark run id does not match")
    try:
        candidate = benchmark["training_evidence"]["core_candidates"][
            residual.oof_candidate
        ]
        folds = candidate["folds"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "residual OOF benchmark is missing candidate fold provenance"
        ) from error
    if (
        candidate.get("candidate") != residual.oof_candidate
        or candidate.get("total_folds") != len(folds)
        or not folds
    ):
        raise RuntimeError(
            "residual OOF benchmark candidate/fold contract does not match"
        )

    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
    if frame.select(keys).is_duplicated().any():
        raise RuntimeError("residual OOF probabilities contain duplicate row keys")
    multi_fold_markets = (
        frame.group_by("market_id")
        .agg(pl.col("fold_index").n_unique().alias("folds"))
        .filter(pl.col("folds") != 1)
    )
    if not multi_fold_markets.is_empty():
        raise RuntimeError("residual OOF markets cross validation folds")

    reported_indices = sorted(int(fold["fold_index"]) for fold in folds)
    observed_indices = sorted(int(value) for value in frame["fold_index"].unique())
    if (
        reported_indices != list(range(len(folds)))
        or observed_indices != reported_indices
    ):
        raise RuntimeError(
            "residual OOF fold indices are incomplete or non-contiguous"
        )

    previous_validation_end: datetime | None = None
    for fold in sorted(folds, key=lambda value: int(value["fold_index"])):
        fold_index = int(fold["fold_index"])
        fit_start = _report_datetime(fold["fit_range_start"])
        fit_end = _report_datetime(fold["fit_range_end"])
        calibration_start = _report_datetime(
            fold["calibration_range_start"]
        )
        calibration_end = _report_datetime(fold["calibration_range_end"])
        policy_start = _report_datetime(fold["policy_range_start"])
        policy_end = _report_datetime(fold["policy_range_end"])
        validation_start = _report_datetime(
            fold["validation_range_start"]
        )
        validation_end = _report_datetime(fold["validation_range_end"])
        if not (
            fit_start
            < fit_end
            < calibration_start
            <= calibration_end
            < policy_start
            <= policy_end
            < validation_start
            < validation_end
        ):
            raise RuntimeError(
                f"residual OOF fold {fold_index} is not chronologically trained"
            )
        if (
            previous_validation_end is not None
            and validation_start < previous_validation_end
        ):
            raise RuntimeError("residual OOF validation folds overlap")
        previous_validation_end = validation_end

        observed = frame.filter(pl.col("fold_index") == fold_index)
        if observed.is_empty():
            raise RuntimeError(f"residual OOF fold {fold_index} has no rows")
        if (
            observed["window_start"].min() < validation_start
            or observed["window_start"].max() >= validation_end
            or observed["market_id"].n_unique()
            != int(fold["eligible_markets"])
        ):
            raise RuntimeError(
                f"residual OOF fold {fold_index} rows escape reported validation"
            )


def _report_datetime(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as error:
        raise RuntimeError(
            f"invalid timestamp in residual OOF benchmark: {value}"
        ) from error
    if parsed.tzinfo is None:
        raise RuntimeError(
            f"residual OOF benchmark timestamp is not timezone-aware: {value}"
        )
    return parsed


def _join_oof_strict_rows(
    strict_frame: pl.DataFrame,
    oof_frame: pl.DataFrame,
) -> pl.DataFrame:
    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
    joined = strict_frame.join(
        oof_frame,
        on=keys,
        how="inner",
        validate="1:1",
        suffix="_oof",
    )
    if joined.is_empty():
        raise RuntimeError("strict book rows do not overlap the pinned OOF core")
    if joined.filter(pl.col("label_up") != pl.col("label_up_oof")).height:
        raise RuntimeError("strict book and OOF core labels disagree")
    return joined.drop("label_up_oof").sort(
        ["window_start", "seconds_elapsed", "market_id"]
    )


def _select_l2_strength(
    fit_frame: pl.DataFrame,
    selection_frame: pl.DataFrame,
    candidates: tuple[float, ...],
) -> tuple[list[dict[str, Any]], float]:
    history = [
        _residual_fit_evidence(
            fit_frame,
            selection_frame,
            l2_strength=value,
        )
        for value in candidates
    ]
    selected = min(
        history,
        key=lambda row: (
            row["evaluation"]["weighted_log_loss"],
            -row["l2_strength"],
        ),
    )
    return history, float(selected["l2_strength"])


def _residual_fit_evidence(
    fit_frame: pl.DataFrame,
    evaluation_frame: pl.DataFrame,
    *,
    l2_strength: float,
) -> dict[str, Any]:
    model, diagnostics = _fit_residual_from_joined(
        fit_frame,
        l2_strength=l2_strength,
    )
    core_logit = _probability_logit(
        evaluation_frame["core_probability_up"].to_numpy()
    )
    features = derive_strict_book_residual_features(
        evaluation_frame,
        core_logit,
    )
    probability = model.probability(core_logit, features)
    weights, markets = market_equal_weights(
        evaluation_frame["market_id"].to_list(),
        expected_length=evaluation_frame.height,
    )
    labels = evaluation_frame["label_up"].to_numpy().astype(np.float64)
    weighted_log_loss = float(
        np.sum(
            weights
            * (
                np.logaddexp(0.0, _probability_logit(probability))
                - labels * _probability_logit(probability)
            )
        )
    )
    core_weighted_log_loss = float(
        np.sum(
            weights
            * (
                np.logaddexp(0.0, core_logit)
                - labels * core_logit
            )
        )
    )
    return {
        "l2_strength": l2_strength,
        "fit": diagnostics.to_dict(),
        "evaluation": {
            "rows": evaluation_frame.height,
            "markets": markets,
            "weighted_log_loss": weighted_log_loss,
            "core_weighted_log_loss": core_weighted_log_loss,
            "weighted_log_loss_uplift": (
                core_weighted_log_loss - weighted_log_loss
            ),
            "row_accuracy": float(
                np.mean((probability >= 0.5) == labels)
            ),
            "core_row_accuracy": float(
                np.mean(
                    (evaluation_frame["core_probability_up"].to_numpy() >= 0.5)
                    == labels
                )
            ),
        },
    }


def _fit_residual_from_joined(
    frame: pl.DataFrame,
    *,
    l2_strength: float,
) -> tuple[ResidualBookModel, Any]:
    core_logit = _probability_logit(frame["core_probability_up"].to_numpy())
    features = derive_strict_book_residual_features(frame, core_logit)
    model, diagnostics = fit_residual_book_model(
        core_logit,
        features,
        frame["label_up"].to_numpy(),
        frame["market_id"].to_list(),
        l2_strength=l2_strength,
    )
    if not diagnostics.converged:
        raise RuntimeError(
            f"book residual optimizer did not converge at L2={l2_strength}"
        )
    return model, diagnostics


def _routed_raw_logits(
    core_frame: pl.DataFrame,
    strict_frame: pl.DataFrame,
    core_bundle: FrozenTrainingBundle,
    residual_model: ResidualBookModel,
) -> dict[str, Any]:
    if core_frame.is_empty():
        raise RuntimeError("universal routed cohort is empty")
    indexed = core_frame.with_row_index("_row_index")
    core_probability = core_bundle.probability(indexed)
    core_raw_logit = _probability_logit(core_probability)
    core_with_probability = indexed.with_columns(
        pl.Series("core_probability_up", core_probability)
    )
    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
    strict = strict_frame.join(
        core_with_probability.select(
            *keys,
            "_row_index",
            "core_probability_up",
            pl.col("label_up").alias("core_label_up"),
        ),
        on=keys,
        how="inner",
        validate="1:1",
    ).sort("_row_index")
    if strict.filter(pl.col("label_up") != pl.col("core_label_up")).height:
        raise RuntimeError("strict book and universal core labels disagree")
    residual_raw_logit = core_raw_logit.copy()
    strict_mask = np.zeros(indexed.height, dtype=np.bool_)
    if not strict.is_empty():
        positions = strict["_row_index"].to_numpy().astype(np.int64)
        strict_core_logit = _probability_logit(
            strict["core_probability_up"].to_numpy()
        )
        strict_features = derive_strict_book_residual_features(
            strict,
            strict_core_logit,
        )
        residual_raw_logit[positions] = residual_model.raw_logit(
            strict_core_logit,
            strict_features,
        )
        strict_mask[positions] = True
        correction = (
            residual_raw_logit[positions] - core_raw_logit[positions]
        )
        correction_summary = _numeric_summary(correction)
    else:
        correction_summary = None
    return {
        "core_raw_logit": core_raw_logit,
        "residual_raw_logit": residual_raw_logit,
        "strict_mask": strict_mask,
        "strict_keys": strict.select(keys),
        "diagnostics": {
            "rows": indexed.height,
            "markets": indexed["market_id"].n_unique(),
            "residual_rows": int(strict_mask.sum()),
            "residual_markets": strict["market_id"].n_unique(),
            "core_fallback_rows": int((~strict_mask).sum()),
            "correction_logit": correction_summary,
        },
    }


def _score_candidate_pair(
    core_frame: pl.DataFrame,
    strict_frame: pl.DataFrame,
    core_bundle: FrozenTrainingBundle,
    residual_model: ResidualBookModel,
    control_calibrator: DirectionTimeCalibrator,
    residual_calibrator: DirectionTimeCalibrator,
    config: EntryBenchmarkConfig,
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    route = _routed_raw_logits(
        core_frame,
        strict_frame,
        core_bundle,
        residual_model,
    )
    seconds = core_frame["seconds_elapsed"].to_numpy()
    control_probability = control_calibrator.probability(
        route["core_raw_logit"],
        seconds,
    )
    strict_residual_probability = residual_calibrator.probability(
        route["residual_raw_logit"],
        seconds,
    )
    residual_probability = np.where(
        route["strict_mask"],
        strict_residual_probability,
        control_probability,
    )
    control = scored_prediction_rows(
        core_frame,
        control_probability,
    ).with_columns(
        pl.lit(config.benchmark.control_candidate).alias("candidate"),
        pl.lit(True).alias("model_eligible"),
        pl.lit("universal_btc_core").alias("prediction_route"),
    )
    residual_routes = np.where(
        route["strict_mask"],
        "book_residual",
        "universal_btc_core_fallback",
    )
    challenger = scored_prediction_rows(
        core_frame,
        residual_probability,
    ).with_columns(
        pl.lit(config.benchmark.strict_book_candidate).alias("candidate"),
        pl.lit(True).alias("model_eligible"),
        pl.Series("prediction_route", residual_routes),
    )
    return (
        {
            config.benchmark.control_candidate: control,
            config.benchmark.strict_book_candidate: challenger,
        },
        route,
    )


def _strict_policy_ablation(
    candidate_frames: dict[str, pl.DataFrame],
    strict_keys: pl.DataFrame,
    policies: dict[str, CandidatePolicy],
    config: EntryBenchmarkConfig,
) -> dict[str, Any]:
    keys = ["market_id", "observed_at", "seconds_elapsed"]
    strict = strict_keys.select(keys).unique()
    strict_frames = {
        name: frame.join(strict, on=keys, how="inner", validate="1:1")
        for name, frame in candidate_frames.items()
    }
    if any(frame.is_empty() for frame in strict_frames.values()):
        return {
            "status": "unavailable",
            "reason": "policy cohort contains no strict residual rows",
        }
    strict_markets = sorted(
        strict_frames[config.benchmark.control_candidate][
            "market_id"
        ].unique().to_list()
    )
    return benchmark_predictions(
        strict_frames,
        policies=policies,
        control_candidate=config.benchmark.control_candidate,
        evidence=BenchmarkEvidence(
            label="July 18-19 exact-row strict residual ablation",
            kind="development",
            independent=False,
        ),
        eligible_market_ids=strict_markets,
        minimum_samples=min(
            config.gates.minimum_common_time_markets,
            len(strict_markets),
        ),
        minimum_executable_samples=min(
            config.gates.minimum_executable_markets,
            len(strict_markets),
        ),
        quantity=config.benchmark.quantity,
    )


def _require_nonempty_residual_cohort(
    name: str,
    frame: pl.DataFrame,
) -> None:
    if frame.is_empty() or frame["market_id"].n_unique() < 100:
        raise RuntimeError(
            f"{name} requires at least 100 strict OOF markets"
        )


def _require_calibrator_convergence(
    name: str,
    calibrator: DirectionTimeCalibrator,
) -> None:
    failed = [cell.key for cell in calibrator.cells if not cell.converged]
    if failed:
        raise RuntimeError(
            f"{name} calibrator did not converge for cells: "
            + ", ".join(failed)
        )


def _cohort_summary(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "range_start": frame["window_start"].min().isoformat(),
        "range_end": frame["window_start"].max().isoformat(),
    }


def _paired_cohort_summary(
    universal: pl.DataFrame,
    strict: pl.DataFrame,
) -> dict[str, Any]:
    universal_markets = universal["market_id"].n_unique()
    strict_markets = strict["market_id"].n_unique()
    return {
        "universal_rows": universal.height,
        "universal_markets": universal_markets,
        "strict_rows": strict.height,
        "strict_markets": strict_markets,
        "strict_market_coverage": (
            strict_markets / universal_markets if universal_markets else 0.0
        ),
    }


def _probability_logit(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("probabilities must be a finite vector")
    clipped = np.clip(values, 1e-9, 1 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("summary values must be a non-empty finite vector")
    return {
        "minimum": float(vector.min()),
        "p05": float(np.quantile(vector, 0.05)),
        "median": float(np.median(vector)),
        "p95": float(np.quantile(vector, 0.95)),
        "maximum": float(vector.max()),
        "mean": float(vector.mean()),
        "standard_deviation": float(vector.std()),
    }
