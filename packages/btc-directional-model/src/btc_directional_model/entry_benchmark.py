from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import polars as pl

from .benchmark_config import (
    EntryBenchmarkConfig,
    benchmark_config_to_dict,
)
from .core_benchmark import (
    AdvancementCriteria,
    BenchmarkEvidence,
    CandidatePolicy,
    benchmark_predictions,
)
from .core_benchmark_report import generate_benchmark_report
from .core_config import load_core_config
from .core_execution import (
    ExecutionEvidenceConfig,
    extract_execution_evidence,
    load_execution_evidence_manifest,
)
from .core_extract import file_sha256, write_json_atomic
from .core_features import load_core_feature_frame
from .core_training import develop_core_models
from .offline_challengers import (
    PREOPEN_CANDIDATE,
    derive_strict_book_frame,
    preopen_candidate_spec,
    train_strict_book_candidate,
    walk_forward_offline_candidate,
)
from .preopen_features import build_preopen_features, join_preopen_features
from .provenance import runtime_provenance

ENTRY_BENCHMARK_RUN_SCHEMA_VERSION = "btc-entry-benchmark-run-v1"


def run_entry_benchmark(
    config: EntryBenchmarkConfig,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    _configure_compute(config)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _update_progress(run_dir, "preparing_data", 0.02)
    core_config = load_core_config(config.benchmark.core_config)
    preopen_metadata = build_preopen_features(
        core_config,
        config.paths.preopen_features,
        force=force,
    )
    execution_config = _execution_config(config)
    execution_manifest = extract_execution_evidence(
        execution_config,
        force=force,
    )
    _update_progress(
        run_dir,
        "training_core_candidates",
        0.15,
        {
            "execution_rows": execution_manifest["totals"]["rows"],
            "strict_book_rows": execution_manifest["totals"][
                "strict_both_side_eligible_rows"
            ],
        },
    )
    core_run_dir, core_freeze_dir, core_metrics = develop_core_models(
        core_config
    )
    for candidate_name in (
        config.benchmark.control_candidate,
        config.benchmark.early_candidate,
    ):
        _require_exact_policy_threshold(
            candidate_name,
            core_metrics["candidates"][candidate_name],
            core_config.model.confidence_min,
        )
    control_and_early = _core_candidate_prediction_frames(
        core_run_dir,
        config,
    )
    core_frame = load_core_feature_frame(core_config, "pre_holdout")
    policy_core_frame = _benchmark_policy_frame(core_frame, config)
    eligible_market_ids = sorted(policy_core_frame["market_id"].unique().to_list())

    _update_progress(run_dir, "training_preopen_candidate", 0.50)
    preopen_frame = pl.read_parquet(config.paths.preopen_features)
    enriched_core = join_preopen_features(core_frame, preopen_frame)
    preopen_metrics, preopen_predictions = walk_forward_offline_candidate(
        enriched_core,
        preopen_candidate_spec(),
        core_config,
    )
    _require_exact_policy_threshold(
        PREOPEN_CANDIDATE,
        preopen_metrics,
        core_config.model.confidence_min,
    )
    preopen_predictions = _benchmark_policy_frame(
        preopen_predictions,
        config,
    )

    _update_progress(run_dir, "training_strict_book_candidate", 0.68)
    execution_frame = _load_execution_evidence(execution_config)
    strict_book_frame = derive_strict_book_frame(
        _execution_range_frame(core_frame, config),
        execution_frame,
    )
    strict_book_metrics, strict_book_predictions, strict_book_bundle = (
        train_strict_book_candidate(
            strict_book_frame,
            core_frame,
            config,
            core_config,
        )
    )
    strict_model_path = run_dir / "strict-book-training-model.joblib"
    joblib.dump(strict_book_bundle, strict_model_path, compress=3)
    strict_book_metrics["training_model_file"] = strict_model_path.name
    strict_book_metrics["training_model_sha256"] = file_sha256(
        strict_model_path
    )

    candidate_frames = {
        **control_and_early,
        PREOPEN_CANDIDATE: preopen_predictions,
        config.benchmark.strict_book_candidate: strict_book_predictions,
    }
    candidate_frames = {
        name: _attach_execution_evidence(frame, execution_frame)
        for name, frame in candidate_frames.items()
    }
    policies = {
        config.benchmark.control_candidate: CandidatePolicy(
            confidence_threshold=core_config.model.confidence_min,
            deployment_compatible=True,
        ),
        config.benchmark.early_candidate: CandidatePolicy(
            confidence_threshold=core_config.model.confidence_min,
            deployment_compatible=True,
        ),
        PREOPEN_CANDIDATE: CandidatePolicy(
            confidence_threshold=core_config.model.confidence_min,
            deployment_compatible=False,
        ),
        config.benchmark.strict_book_candidate: CandidatePolicy(
            confidence_threshold=float(
                strict_book_metrics["confidence_threshold"]
            ),
            deployment_compatible=False,
        ),
    }
    benchmark = benchmark_predictions(
        candidate_frames,
        policies=policies,
        control_candidate=config.benchmark.control_candidate,
        evidence=BenchmarkEvidence(
            label=(
                "June 8-11 clean-book chronological development cohort; "
                "historical labels previously accessed"
            ),
            kind="development",
            independent=config.benchmark.evaluation_is_independent,
        ),
        eligible_market_ids=eligible_market_ids,
        minimum_samples=config.gates.minimum_common_time_markets,
        minimum_executable_samples=config.gates.minimum_executable_markets,
        quantity=config.benchmark.quantity,
        criteria=_advancement_criteria(config),
    )
    benchmark.update(
        {
            "run_schema_version": ENTRY_BENCHMARK_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "configuration": benchmark_config_to_dict(config),
            "evaluation_note": config.benchmark.evaluation_note,
            "runtime_provenance": runtime_provenance(config.package_root),
            "data_evidence": {
                "core_run": str(core_run_dir),
                "core_freeze": (
                    str(core_freeze_dir) if core_freeze_dir is not None else None
                ),
                "core_selected_candidate": core_metrics["selected_candidate"],
                "preopen": preopen_metadata,
                "execution": execution_manifest,
            },
            "training_evidence": {
                "core_candidates": {
                    name: core_metrics["candidates"][name]
                    for name in (
                        config.benchmark.control_candidate,
                        config.benchmark.early_candidate,
                    )
                },
                "preopen_candidate": preopen_metrics,
                "strict_book_candidate": strict_book_metrics,
            },
            "deployment": _deployment_decision(benchmark),
        }
    )
    for name, frame in candidate_frames.items():
        frame.write_parquet(
            run_dir / f"{name}-predictions.parquet",
            compression="zstd",
        )
    write_json_atomic(run_dir / "benchmark.json", benchmark)
    report_path = generate_benchmark_report(
        benchmark,
        run_dir / "report.html",
    )
    _update_progress(
        run_dir,
        "benchmark_complete",
        1.0,
        {
            "report": report_path.name,
            "benchmark_passed_candidates": benchmark[
                "benchmark_passed_candidates"
            ],
            "deployment_qualified_candidates": benchmark[
                "deployment_qualified_candidates"
            ],
        },
    )
    return run_dir, benchmark


def _core_candidate_prediction_frames(
    core_run_dir: Path,
    config: EntryBenchmarkConfig,
) -> dict[str, pl.DataFrame]:
    selected = pl.read_parquet(
        core_run_dir / "walk-forward-predictions.parquet"
    )
    fixed = pl.read_parquet(
        core_run_dir / "walk-forward-fixed-time-predictions.parquet"
    )
    output = {}
    for name in (
        config.benchmark.control_candidate,
        config.benchmark.early_candidate,
    ):
        rows = pl.concat(
            [
                selected.filter(pl.col("candidate") == name),
                fixed.filter(pl.col("candidate") == name),
            ],
            how="diagonal_relaxed",
        ).unique(
            subset=[
                "candidate",
                "market_id",
                "observed_at",
                "seconds_elapsed",
            ],
            keep="first",
        )
        output[name] = _benchmark_policy_frame(rows, config)
    return output


def _benchmark_policy_frame(
    frame: pl.DataFrame,
    config: EntryBenchmarkConfig,
) -> pl.DataFrame:
    return frame.filter(
        (pl.col("window_start") >= config.book_split.policy_start)
        & (pl.col("window_start") < config.book_split.policy_end)
    ).sort(["market_id", "seconds_elapsed"])


def _execution_range_frame(
    frame: pl.DataFrame,
    config: EntryBenchmarkConfig,
) -> pl.DataFrame:
    return frame.filter(
        (pl.col("window_start") >= config.execution.range_start)
        & (pl.col("window_start") < config.execution.range_end)
    )


def _execution_config(
    config: EntryBenchmarkConfig,
) -> ExecutionEvidenceConfig:
    return ExecutionEvidenceConfig(
        range_start=config.execution.range_start,
        range_end=config.execution.range_end,
        output_dir=config.execution.output_dir,
        sample_interval_seconds=config.execution.sample_interval_seconds,
        min_seconds_after_open=config.execution.min_seconds_after_open,
        max_seconds_after_open=config.execution.max_seconds_after_open,
        freshness_seconds=config.execution.stale_after_seconds,
        quantity=config.benchmark.quantity,
    )


def _load_execution_evidence(
    config: ExecutionEvidenceConfig,
) -> pl.DataFrame:
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
            "up_provider_received_at",
            "up_best_bid",
            "up_best_ask",
            "up_best_bid_size",
            "up_best_ask_size",
            "up_bid_depth",
            "up_ask_depth",
            "up_imbalance",
            "down_provider_received_at",
            "down_best_bid",
            "down_best_ask",
            "down_best_bid_size",
            "down_best_ask_size",
            "down_bid_depth",
            "down_ask_depth",
            "down_imbalance",
        )
        .collect()
    )


def _attach_execution_evidence(
    predictions: pl.DataFrame,
    evidence: pl.DataFrame,
) -> pl.DataFrame:
    economics = evidence.select(
        "market_id",
        "observed_at",
        "fee_rate",
        "up_ask_vwap_5",
        "down_ask_vwap_5",
        pl.col("up_side_fresh").alias("up_executable"),
        pl.col("down_side_fresh").alias("down_executable"),
    )
    return predictions.join(
        economics,
        on=["market_id", "observed_at"],
        how="left",
        validate="m:1",
    ).with_columns(
        pl.col("up_executable").fill_null(False),
        pl.col("down_executable").fill_null(False),
    )


def _advancement_criteria(
    config: EntryBenchmarkConfig,
) -> AdvancementCriteria:
    gates = config.gates
    return AdvancementCriteria(
        minimum_accuracy=gates.minimum_accuracy,
        minimum_balanced_accuracy=gates.minimum_balanced_accuracy,
        minimum_direction_recall=gates.minimum_direction_recall,
        minimum_wilson_lower_95=gates.minimum_wilson_lower_95,
        maximum_expected_calibration_error=(
            gates.maximum_expected_calibration_error
        ),
        minimum_coverage=gates.minimum_coverage,
        minimum_coverage_uplift=gates.minimum_coverage_uplift,
        maximum_accuracy_regression=gates.maximum_accuracy_regression,
        maximum_balanced_accuracy_regression=(
            gates.maximum_balanced_accuracy_regression
        ),
        maximum_direction_recall_regression=(
            gates.maximum_direction_recall_regression
        ),
        maximum_median_entry_seconds_regression=(
            gates.maximum_median_entry_seconds_regression
        ),
        minimum_mean_direct_edge_per_share=(
            gates.minimum_mean_direct_edge_per_share
        ),
        minimum_realized_net_per_share=gates.minimum_realized_net_per_share,
        minimum_common_time_markets=gates.minimum_common_time_markets,
        maximum_native_p99_milliseconds=(
            gates.maximum_native_p99_milliseconds
        ),
        maximum_runtime_model_bytes=gates.maximum_runtime_model_bytes,
    )


def _require_exact_policy_threshold(
    candidate_name: str,
    training_metrics: dict[str, Any],
    benchmark_threshold: float,
) -> None:
    fold_thresholds = [
        float(fold["confidence_threshold"])
        for fold in training_metrics.get("folds", [])
    ]
    if not fold_thresholds:
        raise RuntimeError(
            f"{candidate_name} has no chronological fold threshold evidence"
        )
    mismatched = [
        threshold
        for threshold in fold_thresholds
        if not math.isclose(
            threshold,
            benchmark_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if mismatched:
        observed = ", ".join(f"{value:.12g}" for value in sorted(set(mismatched)))
        raise RuntimeError(
            f"{candidate_name} fold thresholds ({observed}) do not match "
            f"benchmark threshold {benchmark_threshold:.12g}; full five-second "
            "scores are required before benchmarking a different policy"
        )


def _deployment_decision(benchmark: dict[str, Any]) -> dict[str, Any]:
    qualified = benchmark["deployment_qualified_candidates"]
    if qualified:
        return {
            "status": "qualified_candidate_available",
            "candidates": qualified,
            "action": "package only through the existing immutable runtime contract",
        }
    evaluation = benchmark["evaluation"]
    reasons = ["no candidate passed every advancement and deployment gate"]
    if evaluation["development_only"]:
        reasons.extend(
            [
                "the benchmark cohort is development-only and non-independent",
                "a new post-July 20 holdout is required for qualification",
            ]
        )
    return {
        "status": "blocked",
        "candidates": [],
        "action": "no runtime bundle, migration, process, image, or restart",
        "reasons": reasons,
    }


def _configure_compute(config: EntryBenchmarkConfig) -> None:
    threads = str(config.compute.threads_per_fit)
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["NUMEXPR_NUM_THREADS"] = threads
    os.environ.setdefault(
        "POLARS_MAX_THREADS",
        str(config.compute.polars_threads),
    )


def _update_progress(
    run_dir: Path,
    stage: str,
    completion: float,
    details: dict[str, Any] | None = None,
) -> None:
    write_json_atomic(
        run_dir / "progress.json",
        {
            "stage": stage,
            "completion": completion,
            "updated_at": datetime.now(UTC).isoformat(),
            "details": details or {},
        },
    )
