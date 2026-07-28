from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import polars as pl

from .benchmark_config import (
    CORE_ONLY_REUSE_DIAGNOSTICS_MODE,
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
from .core_config import CoreTrainingConfig, load_core_config
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
EARLY_ENTRY_CORE_CANDIDATES = (
    "histogram_enriched",
    "histogram_early_weighted",
    "histogram_early_weighted_moderate",
    "histogram_early_90_120",
)
NATIVE_EVIDENCE_CHECKS = {
    "native inference p99 is within budget",
    "runtime model size is within budget",
}


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
    if config.benchmark.mode == CORE_ONLY_REUSE_DIAGNOSTICS_MODE:
        return _run_core_only_benchmark(
            config,
            core_config,
            run_id,
            run_dir,
            force=force,
        )
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


def _run_core_only_benchmark(
    config: EntryBenchmarkConfig,
    core_config: CoreTrainingConfig,
    run_id: str,
    run_dir: Path,
    *,
    force: bool,
) -> tuple[Path, dict[str, Any]]:
    _validate_core_only_contract(config, core_config)
    prior_diagnostics = _load_prior_diagnostics(config)
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
            "candidate_names": list(config.benchmark.candidate_names),
            "runtime_freeze_enabled": False,
            "final_candidate_fitting_enabled": False,
        },
    )
    core_run_dir, core_freeze_dir, core_metrics = develop_core_models(
        core_config,
        freeze_if_ready=False,
        fit_final_candidate=False,
    )
    if core_freeze_dir is not None:
        raise RuntimeError(
            "core-only benchmark created a premature model freeze artifact"
        )
    candidate_frames = _chronological_core_candidate_prediction_frames(
        core_run_dir,
        core_metrics,
        core_config,
        config,
    )
    eligible_market_ids = (
        pl.scan_parquet(core_config.paths.development_feature_data)
        .filter(
            (pl.col("window_start") >= config.book_split.policy_start)
            & (pl.col("window_start") < config.book_split.policy_end)
        )
        .select("market_id")
        .unique()
        .collect()["market_id"]
        .sort()
        .to_list()
    )
    if not eligible_market_ids:
        raise RuntimeError("core-only benchmark policy cohort has no eligible markets")
    execution_frame = _load_execution_evidence(execution_config)
    candidate_frames = {
        name: _attach_execution_evidence(frame, execution_frame)
        for name, frame in candidate_frames.items()
    }
    policies = {
        name: CandidatePolicy(
            confidence_threshold=None,
            deployment_compatible=True,
            selection_mode="chronological_preselected",
            confidence_threshold_min=core_config.model.confidence_min,
            confidence_threshold_max=core_config.model.confidence_max,
        )
        for name in config.benchmark.candidate_names
    }
    benchmark = benchmark_predictions(
        candidate_frames,
        policies=policies,
        control_candidate=config.benchmark.control_candidate,
        evidence=BenchmarkEvidence(
            label=(
                "June 8-11 clean execution-economics development cohort; "
                "exact 90-calendar-day training range; historical labels consumed"
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
    training_selection = _training_selection(
        benchmark,
        core_metrics,
        config,
        core_config,
    )
    prior_evidence = prior_diagnostics["training_evidence"]
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
                "core_freeze": None,
                "core_selected_candidate": core_metrics["selected_candidate"],
                "training_range": {
                    "start": core_config.data.range_start.isoformat(),
                    "end_exclusive": core_config.data.range_end.isoformat(),
                    "calendar_days": (
                        core_config.data.range_end - core_config.data.range_start
                    ).days,
                },
                "execution_cohort": {
                    "start": config.book_split.policy_start.isoformat(),
                    "end_exclusive": config.book_split.policy_end.isoformat(),
                },
                "preopen": prior_diagnostics.get("data_evidence", {}).get(
                    "preopen",
                    {},
                ),
                "execution": execution_manifest,
                "prior_diagnostics": _prior_diagnostics_provenance(
                    config,
                    prior_diagnostics,
                ),
            },
            "training_evidence": {
                "core_candidates": {
                    name: core_metrics["candidates"][name]
                    for name in config.benchmark.candidate_names
                },
                "preopen_candidate": prior_evidence["preopen_candidate"],
                "strict_book_candidate": prior_evidence[
                    "strict_book_candidate"
                ],
                "prior_diagnostics": _prior_diagnostics_provenance(
                    config,
                    prior_diagnostics,
                ),
            },
            "training_selection": training_selection,
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
            "training_finalist": training_selection["finalist"],
            "runtime_freeze_created": False,
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


def _chronological_core_candidate_prediction_frames(
    core_run_dir: Path,
    core_metrics: dict[str, Any],
    core_config: CoreTrainingConfig,
    config: EntryBenchmarkConfig,
) -> dict[str, pl.DataFrame]:
    thresholds = _chronological_threshold_frame(
        core_metrics,
        core_config,
        config.benchmark.candidate_names,
    )
    scores = (
        pl.scan_parquet(
            core_run_dir / "walk-forward-scored-probabilities.parquet"
        )
        .filter(
            pl.col("candidate").is_in(config.benchmark.candidate_names)
            & (pl.col("window_start") >= config.book_split.policy_start)
            & (pl.col("window_start") < config.book_split.policy_end)
        )
        .collect()
    )
    if scores.is_empty():
        raise RuntimeError("core-only benchmark has no scored probability rows")
    output: dict[str, pl.DataFrame] = {}
    for candidate_name in config.benchmark.candidate_names:
        rows = scores.filter(pl.col("candidate") == candidate_name).join(
            thresholds.filter(pl.col("candidate") == candidate_name),
            on=["candidate", "fold_index"],
            how="left",
            validate="m:1",
        )
        if (
            rows.is_empty()
            or rows["selected_confidence_threshold"].null_count()
        ):
            raise RuntimeError(
                f"{candidate_name} is missing scored rows or threshold lineage"
            )
        selection_keys = [
            "candidate",
            "fold_index",
            "market_id",
            "observed_at",
            "seconds_elapsed",
        ]
        first_crossings = (
            rows.filter(
                pl.col("confidence")
                >= pl.col("selected_confidence_threshold")
            )
            .sort(["market_id", "seconds_elapsed", "observed_at"])
            .group_by("market_id", maintain_order=True)
            .first()
            .select(selection_keys)
            .with_columns(pl.lit(True).alias("policy_selected"))
        )
        output[candidate_name] = (
            rows.join(
                first_crossings,
                on=selection_keys,
                how="left",
                validate="1:1",
            )
            .with_columns(pl.col("policy_selected").fill_null(False))
            .sort(["market_id", "seconds_elapsed", "observed_at"])
        )
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


def _validate_core_only_contract(
    config: EntryBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> None:
    exact_start = datetime(2026, 4, 21, tzinfo=UTC)
    exact_end = datetime(2026, 7, 20, tzinfo=UTC)
    if (
        core_config.data.range_start != exact_start
        or core_config.data.range_end != exact_end
        or (core_config.data.range_end - core_config.data.range_start).days != 90
    ):
        raise RuntimeError(
            "core-only benchmark requires exact [2026-04-21, 2026-07-20) "
            "90-calendar-day training data"
        )
    if (
        core_config.split.holdout_start != exact_end
        or core_config.split.holdout_end != exact_end
    ):
        raise RuntimeError(
            "core-only benchmark must remain development-only without a "
            "consumed in-range holdout"
        )
    if (
        tuple(core_config.model.candidate_names) != EARLY_ENTRY_CORE_CANDIDATES
        or config.benchmark.candidate_names != EARLY_ENTRY_CORE_CANDIDATES
    ):
        raise RuntimeError(
            "core-only benchmark candidate contract does not match the four "
            "deploy-compatible histogram candidates"
        )
    if (
        not math.isclose(core_config.model.confidence_min, 0.87)
        or not math.isclose(core_config.model.confidence_max, 0.91)
        or not math.isclose(core_config.model.confidence_step, 0.01)
    ):
        raise RuntimeError(
            "core-only benchmark requires the chronological 0.87-0.91 "
            "confidence grid"
        )
    if (
        len(core_config.split.validation_windows) != 5
        or core_config.gates.minimum_nonnegative_uplift_folds != 5
        or core_config.gates.bootstrap_resamples != 10_000
    ):
        raise RuntimeError(
            "core-only benchmark requires five chronological folds and "
            "10,000 bootstrap resamples"
        )
    gates = config.gates
    if (
        not math.isclose(gates.maximum_accuracy_regression, 0.0)
        or not math.isclose(
            gates.maximum_balanced_accuracy_regression,
            0.0,
        )
        or not math.isclose(
            gates.maximum_direction_recall_regression,
            0.0,
        )
        or gates.maximum_median_entry_seconds_regression > -5.0
        or gates.minimum_executable_markets < 500
        or gates.minimum_common_time_markets < 500
        or gates.minimum_mean_direct_edge_per_share < 0.0
        or gates.minimum_realized_net_per_share < 0.0
    ):
        raise RuntimeError(
            "core-only benchmark advancement gates weaken the frozen "
            "non-regression, timing, sample, or economics contract"
        )
    if (
        config.benchmark.fixed_evaluation_seconds
        != (60, 90, 120, 180, 240)
        or not math.isclose(config.benchmark.quantity, 5.0)
    ):
        raise RuntimeError(
            "core-only benchmark requires five-share economics and fixed "
            "60/90/120/180/240-second diagnostics"
        )


def _chronological_threshold_frame(
    core_metrics: dict[str, Any],
    core_config: CoreTrainingConfig,
    candidate_names: tuple[str, ...],
) -> pl.DataFrame:
    expected_fold_indexes = set(range(len(core_config.split.validation_windows)))
    grid = {
        round(
            core_config.model.confidence_min
            + index * core_config.model.confidence_step,
            6,
        )
        for index in range(
            round(
                (
                    core_config.model.confidence_max
                    - core_config.model.confidence_min
                )
                / core_config.model.confidence_step
            )
            + 1
        )
    }
    rows: list[dict[str, Any]] = []
    for candidate_name in candidate_names:
        candidate = core_metrics.get("candidates", {}).get(candidate_name)
        if not isinstance(candidate, dict):
            raise TypeError(
                f"{candidate_name} has no chronological training metrics"
            )
        folds = candidate.get("folds", [])
        indexes = {int(fold["fold_index"]) for fold in folds}
        if indexes != expected_fold_indexes:
            raise RuntimeError(
                f"{candidate_name} does not contain every chronological fold"
            )
        for fold in folds:
            threshold = round(float(fold["confidence_threshold"]), 6)
            policy_end = datetime.fromisoformat(fold["policy_range_end"])
            validation_start = datetime.fromisoformat(
                fold["validation_range_start"]
            )
            if policy_end >= validation_start:
                raise RuntimeError(
                    f"{candidate_name} fold {fold['fold_index']} threshold "
                    "was not selected strictly before validation"
                )
            if threshold not in grid:
                raise RuntimeError(
                    f"{candidate_name} fold {fold['fold_index']} threshold "
                    "escapes the configured confidence grid"
                )
            rows.append(
                {
                    "candidate": candidate_name,
                    "fold_index": int(fold["fold_index"]),
                    "selected_confidence_threshold": threshold,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "candidate": pl.String,
            "fold_index": pl.Int32,
            "selected_confidence_threshold": pl.Float64,
        },
    )


def _load_prior_diagnostics(
    config: EntryBenchmarkConfig,
) -> dict[str, Any]:
    diagnostics = config.prior_diagnostics
    if diagnostics is None:
        raise RuntimeError("core-only benchmark lost its prior diagnostics contract")
    if not diagnostics.record.is_file():
        raise RuntimeError(
            f"pinned prior diagnostics record is missing: {diagnostics.record}"
        )
    observed_sha256 = file_sha256(diagnostics.record)
    if observed_sha256 != diagnostics.sha256:
        raise RuntimeError(
            "pinned prior diagnostics sha256 does not match the configured record"
        )
    record = json.loads(diagnostics.record.read_text())
    if record.get("run_id") != diagnostics.run_id:
        raise RuntimeError(
            "pinned prior diagnostics run_id does not match the configured record"
        )
    training_evidence = record.get("training_evidence", {})
    for key in ("preopen_candidate", "strict_book_candidate"):
        if not isinstance(training_evidence.get(key), dict):
            raise TypeError(f"pinned prior diagnostics are missing {key}")
    return record


def _prior_diagnostics_provenance(
    config: EntryBenchmarkConfig,
    record: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = config.prior_diagnostics
    if diagnostics is None:
        raise RuntimeError("prior diagnostics provenance is unavailable")
    return {
        "run_id": record["run_id"],
        "record": str(diagnostics.record),
        "record_sha256": diagnostics.sha256,
        "reused_without_retraining": True,
        "selection_eligible": False,
    }


def _training_selection(
    benchmark: dict[str, Any],
    core_metrics: dict[str, Any],
    config: EntryBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    passing: list[str] = []
    for name in config.benchmark.candidate_names:
        if name == config.benchmark.control_candidate:
            continue
        benchmark_checks = [
            check
            for check in benchmark["candidates"][name]["advance"]["checks"]
            if check["name"] not in NATIVE_EVIDENCE_CHECKS
        ]
        training_metrics = core_metrics["candidates"][name]
        supplemental = [
            _selection_check(
                "walk-forward development gates",
                bool(training_metrics["passed_development"]),
                "=",
                True,
                bool(training_metrics["passed_development"]),
            ),
            _selection_check(
                "nonnegative same-time uplift folds",
                int(training_metrics["nonnegative_uplift_folds"]),
                ">=",
                core_config.gates.minimum_nonnegative_uplift_folds,
                int(training_metrics["nonnegative_uplift_folds"])
                >= core_config.gates.minimum_nonnegative_uplift_folds,
            ),
            _selection_check(
                "hour-block bootstrap lower 95",
                float(training_metrics["bootstrap"]["lower_95"]),
                ">=",
                0.0,
                float(training_metrics["bootstrap"]["lower_95"]) >= 0.0,
            ),
        ]
        checks = [*benchmark_checks, *supplemental]
        passed = all(check["passed"] for check in checks)
        candidates[name] = {
            "passed": passed,
            "checks": checks,
            "runtime_evidence_deferred": sorted(NATIVE_EVIDENCE_CHECKS),
        }
        if passed:
            passing.append(name)
    finalist = (
        max(
            passing,
            key=lambda name: _training_finalist_rank(
                benchmark["candidates"][name]["own_policy"]
            ),
        )
        if passing
        else None
    )
    return {
        "status": "finalist_available" if finalist is not None else "blocked",
        "finalist": finalist,
        "passing_candidates": passing,
        "runtime_freeze_created": False,
        "candidates": candidates,
    }


def _selection_check(
    name: str,
    observed: Any,
    operator: str,
    required: Any,
    passed: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "required": required,
        "passed": bool(passed),
    }


def _training_finalist_rank(metrics: dict[str, Any]) -> tuple[float, ...]:
    execution = metrics["execution"]
    median_seconds = metrics["median_seconds_elapsed"]
    net_expectancy = execution["realized_net_expectancy_per_trade"]
    direct_edge = execution["mean_direct_edge_per_share"]
    return (
        float(metrics["coverage"]),
        -float(median_seconds if median_seconds is not None else math.inf),
        float(net_expectancy if net_expectancy is not None else -math.inf),
        float(direct_edge if direct_edge is not None else -math.inf),
        float(metrics["wilson_lower_95"]),
        float(metrics["accuracy"]),
        float(metrics["balanced_accuracy"]),
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
