from __future__ import annotations

import html
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import polars as pl
from threadpoolctl import threadpool_limits

from .core_benchmark import _execution_metrics
from .core_config import CoreTrainingConfig, load_core_config
from .core_evaluation import classification_metrics, scored_prediction_rows
from .core_execution import ExecutionEvidenceConfig
from .core_extract import file_sha256, write_json_atomic
from .core_features import load_core_feature_frame, validate_core_feature_cache
from .core_training import (
    CORE_FREEZE_SCHEMA_VERSION,
    TRAINING_MODEL_FILENAME,
    FrozenTrainingBundle,
    chronological_subsplit,
    configure_native_thread_limits,
    fit_probability_calibrator,
    model_candidate_spec,
    model_summary_payload,
    range_frame,
    row_weight_schedule_payload,
)
from .fixed_time_benchmark import (
    GOLDEN_FEATURES_FILENAME,
    _cohort_payload,
    _economics_checks,
    _entry_price_band_metrics,
    _json_value,
    _write_text_atomic,
    empirical_coverage_threshold,
    estimator_training_rows,
    fixed_time_rows,
    operating_point_checks,
)
from .fixed_time_selective_config import (
    FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE,
    FIXED_TIME_SELECTIVE_REGIME_CANDIDATE,
    FixedTimeSelectiveCandidateConfig,
    FixedTimeSelectiveConfig,
    load_fixed_time_selective_config,
)
from .fixed_time_selective_training import (
    predicted_side_hard_error_metrics,
    selective_candidate_spec,
    selective_parameter_grid,
    tune_and_fit_selective_model,
)
from .paper_candidate import write_golden_feature_sample
from .persistence_benchmark import attach_execution_evidence, load_execution_evidence
from .provenance import runtime_provenance
from .runtime_export import export_runtime_model

FIXED_TIME_SELECTIVE_BENCHMARK_SCHEMA_VERSION = (
    "btc-mature-reversal-fixed-time-selective-benchmark-v1"
)
FIXED_TIME_SELECTIVE_FREEZE_SCHEMA_VERSION = (
    "btc-mature-reversal-fixed-time-selective-freeze-v1"
)
FIXED_TIME_SELECTIVE_ASSESSMENT_SCHEMA_VERSION = (
    "btc-mature-reversal-fixed-time-selective-freeze-assessment-v1"
)
FIXED_TIME_SELECTIVE_PAPER_AUTHORIZATION = (
    "explicit-fixed-120-selective-development-paper-only"
)
SCORED_PROBABILITY_SUFFIX = "-scored-probabilities.parquet"


def run_fixed_time_selective_benchmark(
    config: FixedTimeSelectiveConfig,
) -> tuple[Path, dict[str, Any]]:
    core_config = load_core_config(config.benchmark.core_config)
    configure_native_thread_limits(core_config)
    feature_metadata = validate_core_feature_cache(core_config, "pre_holdout")
    _validate_feature_metadata(feature_metadata, config)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    results = _run_candidate_fold_matrix(config, core_config)
    by_candidate: dict[str, list[dict[str, Any]]] = {
        candidate.name: [] for candidate in config.candidates
    }
    scored_by_candidate: dict[str, pl.DataFrame] = {}
    for result in results:
        by_candidate[result["candidate"]].append(result)
    for candidate in config.candidates:
        folds = sorted(by_candidate[candidate.name], key=lambda item: item["fold_index"])
        scored_by_candidate[candidate.name] = pl.concat(
            [fold.pop("scored_rows") for fold in folds],
            how="vertical_relaxed",
        ).sort(["fold_index", "observed_at", "market_id"])
    _assert_identical_scored_universes(scored_by_candidate)

    execution = load_execution_evidence(_execution_config(config))
    candidates: dict[str, dict[str, Any]] = {}
    control_scored = scored_by_candidate[FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE]
    for candidate in config.candidates:
        scored = scored_by_candidate[candidate.name]
        candidate_payload = _candidate_payload(
            candidate=candidate,
            folds=by_candidate[candidate.name],
            scored=scored,
            control_scored=control_scored,
            execution=execution,
            config=config,
        )
        candidates[candidate.name] = candidate_payload
        scored.write_parquet(
            run_dir / f"{candidate.name}{SCORED_PROBABILITY_SUFFIX}",
            compression="zstd",
        )

    selection = _select_candidate(config, candidates)
    benchmark: dict[str, Any] = {
        "schema_version": FIXED_TIME_SELECTIVE_BENCHMARK_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": (
            "development_candidate_selected"
            if selection["selected_candidate"] is not None
            else "no_development_candidate_selected"
        ),
        "evaluation": {
            "kind": "consumed_development",
            "independent": config.benchmark.evaluation_is_independent,
            "note": config.benchmark.evaluation_note,
            "fresh_evidence_start": config.split.policy_selection_end.isoformat(),
        },
        "configuration": _config_payload(config),
        "data_contract": {
            "range_start": core_config.data.range_start.isoformat(),
            "range_end_exclusive": core_config.data.range_end.isoformat(),
            "feature_cache_source_contract": core_config.data.source_contract,
            "post_july_28_rows_used": False,
            "oracle_role": "excluded_from_model_inputs",
            "orderbook_role": "post_prediction_execution_economics_only",
            "feature_metadata": feature_metadata,
        },
        "training_contract": {
            "decision_second": config.model.decision_second,
            "estimator_training_seconds": list(
                config.model.estimator_training_seconds
            ),
            "inner_tuning_scoring_seconds": [config.model.decision_second],
            "probability_calibration_seconds": [config.model.decision_second],
            "policy_selection_seconds": [config.model.decision_second],
            "validation_seconds": [config.model.decision_second],
            "recency_half_life_days": config.model.recency_half_life_days,
            "probability_calibration": config.model.probability_calibration,
            "threshold_selection": config.model.threshold_selection,
            "candidate_order": [candidate.name for candidate in config.candidates],
            "fold_count": len(config.split.validation_windows),
            "primary_advancement_role": "qualification_objective",
            "secondary_advancement_role": "diagnostic_only",
        },
        "candidates": candidates,
        "selection": selection,
        "provenance": {
            "benchmark_config": str(config.source_path),
            "benchmark_config_sha256": file_sha256(config.source_path),
            "core_config": str(config.benchmark.core_config),
            "core_config_sha256": file_sha256(config.benchmark.core_config),
            "development_feature": str(core_config.paths.development_feature_data),
            "development_feature_sha256": file_sha256(
                core_config.paths.development_feature_data
            ),
            "execution_manifest_sha256": config.paths.execution_manifest_sha256,
            "runtime_provenance": runtime_provenance(config.package_root),
        },
    }
    write_json_atomic(run_dir / "benchmark.json", benchmark)
    _write_text_atomic(run_dir / "report.html", _render_report(benchmark))
    return run_dir, benchmark


def _run_candidate_fold_matrix(
    config: FixedTimeSelectiveConfig,
    core_config: CoreTrainingConfig,
) -> list[dict[str, Any]]:
    tasks = [
        (candidate.name, fold_index)
        for candidate in config.candidates
        for fold_index in range(len(config.split.validation_windows))
    ]
    max_workers = min(core_config.compute.max_parallel_fits, len(tasks))
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_candidate_fold_task,
                config.source_path,
                candidate_name,
                fold_index,
            ): (candidate_name, fold_index)
            for candidate_name, fold_index in tasks
        }
        for future in as_completed(futures):
            key = futures[future]
            result = future.result()
            completed[key] = result
            metrics = result["primary"]["metrics"]
            print(
                f"{key[0]} fold {key[1] + 1}/{len(config.split.validation_windows)}: "
                f"accuracy={metrics['accuracy']:.4f} "
                f"coverage={metrics['coverage']:.4f} "
                f"UP={metrics['up_recall']:.4f} "
                f"DOWN={metrics['down_recall']:.4f}",
                flush=True,
            )
    return [completed[key] for key in tasks]


def _evaluate_candidate_fold_task(
    config_path: Path,
    candidate_name: str,
    fold_index: int,
) -> dict[str, Any]:
    config = load_fixed_time_selective_config(config_path)
    core_config = load_core_config(config.benchmark.core_config)
    configure_native_thread_limits(core_config)
    candidate = _candidate_config(config, candidate_name)
    with threadpool_limits(limits=core_config.compute.threads_per_fit):
        frame = estimator_training_rows(
            load_core_feature_frame(core_config, "pre_holdout"),
            training_seconds=config.model.estimator_training_seconds,
            cohort_name="fixed-time selective feature cache",
        )
        return _evaluate_candidate_fold(
            frame,
            candidate=candidate,
            fold_index=fold_index,
            config=config,
            core_config=core_config,
        )


def _evaluate_candidate_fold(
    frame: pl.DataFrame,
    *,
    candidate: FixedTimeSelectiveCandidateConfig,
    fold_index: int,
    config: FixedTimeSelectiveConfig,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    validation_start, validation_end = config.split.validation_windows[fold_index]
    history = range_frame(frame, config.split.development_start, validation_start)
    validation_context = range_frame(frame, validation_start, validation_end)
    fit, calibration_context, policy_context = chronological_subsplit(
        history,
        fit_fraction=0.70,
        calibration_fraction=0.15,
    )
    fit = estimator_training_rows(
        fit,
        training_seconds=config.model.estimator_training_seconds,
        cohort_name=f"{candidate.name} fold {fold_index} estimator fit",
    )
    calibration = fixed_time_rows(
        calibration_context,
        decision_second=config.model.decision_second,
        cohort_name=f"{candidate.name} fold {fold_index} calibration",
    )
    policy = fixed_time_rows(
        policy_context,
        decision_second=config.model.decision_second,
        cohort_name=f"{candidate.name} fold {fold_index} policy",
    )
    validation = fixed_time_rows(
        validation_context,
        decision_second=config.model.decision_second,
        cohort_name=f"{candidate.name} fold {fold_index} validation",
    )

    started = time.perf_counter()
    spec = selective_candidate_spec(candidate, config.model)
    parameter_grid = selective_parameter_grid(candidate, config.model, core_config)
    model, tuning = tune_and_fit_selective_model(
        fit,
        spec,
        parameter_grid,
        core_config,
        config.model.decision_second,
        config.primary.target_coverage,
        config.secondary.target_coverage,
        config.model.hard_confidence_floor,
    )
    calibrator = fit_probability_calibrator(model, calibration, core_config, spec)
    if not calibrator.converged or calibrator.slope <= 0.0:
        raise RuntimeError(
            f"{candidate.name} fold {fold_index} probability calibration failed"
        )

    policy_probability = calibrator.probability(model.raw_logit(policy))
    policy_scored = scored_prediction_rows(policy, policy_probability)
    primary_threshold, primary_policy, primary_selection = (
        empirical_coverage_threshold(
            policy_scored,
            target_coverage=config.primary.target_coverage,
        )
    )
    secondary_threshold, secondary_policy, secondary_selection = (
        empirical_coverage_threshold(
            policy_scored,
            target_coverage=config.secondary.target_coverage,
        )
    )
    probability = calibrator.probability(model.raw_logit(validation))
    scored = scored_prediction_rows(validation, probability).with_columns(
        pl.lit(candidate.name).alias("candidate"),
        pl.lit(fold_index).cast(pl.Int32).alias("fold_index"),
        pl.lit(primary_threshold).alias("primary_confidence_threshold"),
        pl.lit(secondary_threshold).alias("secondary_confidence_threshold"),
        (pl.col("confidence") >= primary_threshold).alias("primary_selected"),
        (pl.col("confidence") >= secondary_threshold).alias("secondary_selected"),
    )
    primary_selected = scored.filter(pl.col("primary_selected"))
    secondary_selected = scored.filter(pl.col("secondary_selected"))
    return {
        "candidate": candidate.name,
        "fold_index": fold_index,
        "validation_window": {
            "start": validation_start.isoformat(),
            "end_exclusive": validation_end.isoformat(),
            "consumed_development": True,
        },
        "fit": {
            **_cohort_payload(fit),
            "role": "estimator_training",
            "expected_seconds": list(config.model.estimator_training_seconds),
        },
        "calibration": _cohort_payload(calibration),
        "policy": _cohort_payload(policy),
        "validation": _cohort_payload(validation),
        "model_contract": {
            "feature_schema_version": candidate.feature_schema_version,
            "feature_count": len(spec.feature_names),
            "feature_names": list(spec.feature_names),
            "row_weight_policy": spec.row_weight_policy,
            "row_weight_schedule": row_weight_schedule_payload(spec),
            "recency_half_life_days": spec.recency_half_life_days,
            "parameter_grid": candidate.parameter_grid,
        },
        "tuning": tuning,
        "calibrator": asdict(calibrator),
        "primary": _fold_operating_point(
            primary_selected,
            eligible_markets=validation.height,
            policy_selected=primary_policy,
            policy_eligible_markets=policy.height,
            threshold_selection=primary_selection,
            contract=config.primary,
            hard_confidence_floor=config.model.hard_confidence_floor,
        ),
        "secondary": _fold_operating_point(
            secondary_selected,
            eligible_markets=validation.height,
            policy_selected=secondary_policy,
            policy_eligible_markets=policy.height,
            threshold_selection=secondary_selection,
            contract=config.secondary,
            hard_confidence_floor=config.model.hard_confidence_floor,
        ),
        "threshold_selection_used_validation_labels": False,
        "elapsed_seconds": time.perf_counter() - started,
        "scored_rows": scored,
    }


def _fold_operating_point(
    selected: pl.DataFrame,
    *,
    eligible_markets: int,
    policy_selected: pl.DataFrame,
    policy_eligible_markets: int,
    threshold_selection: dict[str, Any],
    contract: Any,
    hard_confidence_floor: float,
) -> dict[str, Any]:
    metrics = classification_metrics(selected, eligible_markets=eligible_markets)
    return {
        "threshold_selection": threshold_selection,
        "policy_metrics_after_selection": classification_metrics(
            policy_selected,
            eligible_markets=policy_eligible_markets,
        ),
        "metrics": metrics,
        "predicted_side_hard_confident_errors": (
            predicted_side_hard_error_metrics(
                selected,
                eligible_markets=eligible_markets,
                confidence_floor=hard_confidence_floor,
            )
        ),
        "checks": operating_point_checks(metrics, contract),
    }


def _candidate_payload(
    *,
    candidate: FixedTimeSelectiveCandidateConfig,
    folds: list[dict[str, Any]],
    scored: pl.DataFrame,
    control_scored: pl.DataFrame,
    execution: pl.DataFrame,
    config: FixedTimeSelectiveConfig,
) -> dict[str, Any]:
    with_execution = attach_execution_evidence(scored, execution)
    primary_selected = with_execution.filter(pl.col("primary_selected"))
    secondary_selected = with_execution.filter(pl.col("secondary_selected"))
    eligible_markets = scored.height
    primary = _aggregate_operating_point(
        primary_selected,
        eligible_markets=eligible_markets,
        contract=config.primary,
        hard_confidence_floor=config.model.hard_confidence_floor,
    )
    secondary = _aggregate_operating_point(
        secondary_selected,
        eligible_markets=eligible_markets,
        contract=config.secondary,
        hard_confidence_floor=config.model.hard_confidence_floor,
    )

    control_with_execution = attach_execution_evidence(control_scored, execution)
    control_primary = control_with_execution.filter(pl.col("primary_selected"))
    control_metrics = classification_metrics(
        control_primary,
        eligible_markets=eligible_markets,
    )
    control_hard = predicted_side_hard_error_metrics(
        control_primary,
        eligible_markets=eligible_markets,
        confidence_floor=config.model.hard_confidence_floor,
    )
    control_on_candidate_rows = _control_predictions_on_selected_rows(
        primary_selected,
        control_scored,
        keys=("fold_index", "market_id", "observed_at"),
    )
    control_same_rows_hard = predicted_side_hard_error_metrics(
        control_on_candidate_rows,
        eligible_markets=eligible_markets,
        confidence_floor=config.model.hard_confidence_floor,
    )
    primary_same_rows = _same_row_control_comparison(
        primary_selected,
        control_scored,
    )
    raw_comparison = _raw_control_comparison(scored, control_scored)
    fold_summary = _fold_summary(folds)
    adverse = folds[-1]["primary"]

    checks = list(primary["checks"])
    checks.extend(_prefixed_checks("adverse fold", adverse["checks"]))
    checks.extend(
        [
            _minimum_check(
                "minimum fold UP recall",
                fold_summary["minimum_up_recall"],
                config.primary.minimum_direction_recall,
            ),
            _minimum_check(
                "minimum fold DOWN recall",
                fold_summary["minimum_down_recall"],
                config.primary.minimum_direction_recall,
            ),
            _minimum_check(
                "candidate accuracy on its selected rows versus control",
                primary_same_rows["accuracy_delta"],
                0.0,
            ),
            _minimum_check(
                "candidate raw exact-120 accuracy versus control",
                raw_comparison["accuracy_delta"],
                0.0,
            ),
            _maximum_check(
                "same-row hard-error exposure does not regress control",
                primary["predicted_side_hard_confident_errors"]["all"][
                    "hard_confident_error_exposure_rate"
                ],
                control_same_rows_hard["all"][
                    "hard_confident_error_exposure_rate"
                ],
            ),
            _maximum_check(
                "same-row hard-error selected rate does not regress control",
                primary["predicted_side_hard_confident_errors"]["all"][
                    "hard_confident_error_rate_selected"
                ],
                control_same_rows_hard["all"][
                    "hard_confident_error_rate_selected"
                ],
            ),
            _maximum_check(
                "same-row predicted-UP hard-error selected rate does not regress control",
                primary["predicted_side_hard_confident_errors"]["up"][
                    "hard_confident_error_rate_selected"
                ],
                control_same_rows_hard["up"][
                    "hard_confident_error_rate_selected"
                ],
            ),
        ]
    )
    checks.extend(_economics_checks(primary["execution_by_size"]["ten_share_vwap10"]))
    is_control = candidate.name == FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE
    advancement_qualified = (not is_control) and all(
        check["passed"] for check in checks
    )
    return {
        "candidate": candidate.name,
        "is_control": is_control,
        "feature_schema_version": candidate.feature_schema_version,
        "feature_count": len(candidate.feature_names),
        "feature_names": list(candidate.feature_names),
        "exact_120_weight_multiplier": candidate.exact_120_weight_multiplier,
        "parameter_grid": candidate.parameter_grid,
        "folds": folds,
        "fold_summary": fold_summary,
        "adverse_fold": adverse,
        "primary": primary,
        "secondary": secondary,
        "secondary_advancement_role": "diagnostic_only",
        "control_primary_reference": {
            "metrics": control_metrics,
            "predicted_side_hard_confident_errors": control_hard,
        },
        "control_on_candidate_primary_rows": {
            "predicted_side_hard_confident_errors": control_same_rows_hard,
        },
        "same_selected_rows_control_comparison": primary_same_rows,
        "raw_exact_120_control_comparison": raw_comparison,
        "checks": checks,
        "advancement_qualified": advancement_qualified,
    }


def _aggregate_operating_point(
    selected: pl.DataFrame,
    *,
    eligible_markets: int,
    contract: Any,
    hard_confidence_floor: float,
) -> dict[str, Any]:
    metrics = classification_metrics(selected, eligible_markets=eligible_markets)
    execution_by_size = {
        "five_share_vwap5": _execution_metrics(
            selected,
            quantity=5.0,
            vwap_depth=5,
        ),
        "ten_share_vwap10": _execution_metrics(
            selected,
            quantity=10.0,
            vwap_depth=10,
        ),
    }
    checks = operating_point_checks(metrics, contract)
    return {
        "contract": asdict(contract),
        "metrics": metrics,
        "predicted_side_hard_confident_errors": (
            predicted_side_hard_error_metrics(
                selected,
                eligible_markets=eligible_markets,
                confidence_floor=hard_confidence_floor,
            )
        ),
        "execution_by_size": execution_by_size,
        "ten_share_entry_price_bands": _entry_price_band_metrics(selected),
        "checks": checks,
        "quality_qualified": all(check["passed"] for check in checks),
    }


def _fold_summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [fold["primary"]["metrics"] for fold in folds]
    return {
        "fold_count": len(folds),
        "minimum_accuracy": min(item["accuracy"] for item in metrics),
        "maximum_accuracy": max(item["accuracy"] for item in metrics),
        "minimum_balanced_accuracy": min(
            item["balanced_accuracy"] for item in metrics
        ),
        "minimum_up_recall": min(item["up_recall"] for item in metrics),
        "minimum_down_recall": min(item["down_recall"] for item in metrics),
        "maximum_expected_calibration_error": max(
            item["expected_calibration_error"] for item in metrics
        ),
        "minimum_coverage": min(item["coverage"] for item in metrics),
        "maximum_coverage": max(item["coverage"] for item in metrics),
    }


def _same_row_control_comparison(
    candidate_selected: pl.DataFrame,
    control_scored: pl.DataFrame,
) -> dict[str, Any]:
    keys = ["fold_index", "market_id", "observed_at"]
    control = control_scored.select(
        *keys,
        pl.col("predicted_up").alias("control_predicted_up"),
        pl.col("correct").alias("control_correct"),
    )
    joined = candidate_selected.join(control, on=keys, how="inner", validate="1:1")
    if joined.height != candidate_selected.height:
        raise RuntimeError("candidate selected rows do not match the control universe")
    candidate_correct = joined["correct"].cast(pl.Int64).sum()
    control_correct = joined["control_correct"].cast(pl.Int64).sum()
    markets = joined.height
    return {
        "markets": markets,
        "candidate_accuracy": candidate_correct / markets if markets else 0.0,
        "control_accuracy": control_correct / markets if markets else 0.0,
        "accuracy_delta": (
            (candidate_correct - control_correct) / markets if markets else 0.0
        ),
        "candidate_only_correct": joined.filter(
            pl.col("correct") & ~pl.col("control_correct")
        ).height,
        "control_only_correct": joined.filter(
            ~pl.col("correct") & pl.col("control_correct")
        ).height,
        "direction_disagreements": joined.filter(
            pl.col("predicted_up") != pl.col("control_predicted_up")
        ).height,
    }


def _control_predictions_on_selected_rows(
    candidate_selected: pl.DataFrame,
    control_scored: pl.DataFrame,
    *,
    keys: tuple[str, ...],
) -> pl.DataFrame:
    control = control_scored.select(
        *keys,
        "label_up",
        "probability_up",
        "predicted_up",
        "confidence",
        "correct",
    )
    selected_keys = candidate_selected.select(*keys)
    joined = selected_keys.join(control, on=list(keys), how="inner", validate="1:1")
    if joined.height != candidate_selected.height:
        raise RuntimeError(
            "candidate selected rows do not match the control prediction universe"
        )
    return joined.sort(list(keys))


def _raw_control_comparison(
    candidate_scored: pl.DataFrame,
    control_scored: pl.DataFrame,
) -> dict[str, Any]:
    keys = ["fold_index", "market_id", "observed_at"]
    control = control_scored.select(
        *keys,
        pl.col("correct").alias("control_correct"),
        pl.col("predicted_up").alias("control_predicted_up"),
    )
    joined = candidate_scored.join(control, on=keys, how="inner", validate="1:1")
    if joined.height != candidate_scored.height:
        raise RuntimeError("candidate raw rows do not match the control universe")
    markets = joined.height
    candidate_correct = joined["correct"].cast(pl.Int64).sum()
    control_correct = joined["control_correct"].cast(pl.Int64).sum()
    return {
        "markets": markets,
        "candidate_accuracy": candidate_correct / markets,
        "control_accuracy": control_correct / markets,
        "accuracy_delta": (candidate_correct - control_correct) / markets,
        "direction_disagreements": joined.filter(
            pl.col("predicted_up") != pl.col("control_predicted_up")
        ).height,
    }


def _prefixed_checks(prefix: str, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**check, "name": f"{prefix} {check['name']}"} for check in checks]


def _minimum_check(name: str, observed: float, required: float) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "operator": ">=",
        "required": required,
        "passed": bool(observed >= required - 1e-12),
    }


def _maximum_check(name: str, observed: float, required: float) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "operator": "<=",
        "required": required,
        "passed": bool(observed <= required + 1e-12),
    }


def _select_candidate(
    config: FixedTimeSelectiveConfig,
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    qualified = [
        payload
        for payload in candidates.values()
        if payload["advancement_qualified"]
    ]
    ranked = sorted(qualified, key=_candidate_rank, reverse=True)
    selected = ranked[0]["candidate"] if ranked else None
    return {
        "control_candidate": FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE,
        "qualified_candidates": [payload["candidate"] for payload in ranked],
        "selected_candidate": selected,
        "selection_rank": [
            {
                "candidate": payload["candidate"],
                "rank": list(_candidate_rank(payload)),
            }
            for payload in ranked
        ],
        "development_evidence_only": True,
        "independently_qualified": False,
        "exploratory_paper_export_authorized": selected is not None,
        "live_capital_authorized": False,
        "fresh_forward_evidence_required": True,
        "fresh_forward_evidence_start": (
            config.split.policy_selection_end.isoformat()
        ),
    }


def _candidate_rank(candidate: dict[str, Any]) -> tuple[float, ...]:
    fold = candidate["fold_summary"]
    primary = candidate["primary"]["metrics"]
    hard = candidate["primary"]["predicted_side_hard_confident_errors"]["all"]
    return (
        fold["minimum_accuracy"],
        min(fold["minimum_up_recall"], fold["minimum_down_recall"]),
        primary["accuracy"],
        primary["balanced_accuracy"],
        -hard["hard_confident_error_exposure_rate"],
        -float(candidate["feature_count"]),
    )


def _assert_identical_scored_universes(
    scored_by_candidate: dict[str, pl.DataFrame],
) -> None:
    control = scored_by_candidate[FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE]
    keys = ["fold_index", "market_id", "observed_at", "seconds_elapsed", "label_up"]
    expected = control.select(keys).rows()
    if len(expected) != len(set(expected)):
        raise RuntimeError("control scored universe contains duplicate identities")
    for candidate, scored in scored_by_candidate.items():
        observed = scored.select(keys).rows()
        if observed != expected:
            raise RuntimeError(
                f"{candidate} scored universe differs from the fixed-120 control"
            )


def _candidate_config(
    config: FixedTimeSelectiveConfig,
    candidate_name: str,
) -> FixedTimeSelectiveCandidateConfig:
    for candidate in config.candidates:
        if candidate.name == candidate_name:
            return candidate
    raise ValueError(f"unknown fixed-time selective candidate: {candidate_name}")


def _validate_feature_metadata(
    metadata: dict[str, Any],
    config: FixedTimeSelectiveConfig,
) -> None:
    schemas = metadata.get("candidate_feature_schema_versions", {})
    expected_schemas = {
        candidate.feature_schema_version for candidate in config.candidates
    }
    observed_schemas = set(schemas.values())
    if not expected_schemas.issubset(observed_schemas):
        raise RuntimeError("feature cache lost a fixed-time selective schema")
    if metadata.get("scope") != "pre_holdout":
        raise RuntimeError("fixed-time selective cache scope changed")
    if metadata.get("range_start") != config.split.development_start.isoformat():
        raise RuntimeError("fixed-time selective cache start changed")
    if metadata.get("range_end") != config.split.policy_selection_end.isoformat():
        raise RuntimeError("fixed-time selective cache includes post-July-28 data")
    if int(metadata.get("core_complete_candidate_markets", 0)) <= 0:
        raise RuntimeError("fixed-time selective cache has no complete markets")
    if int(metadata.get("expected_candidate_rows_per_market", 0)) != len(
        config.model.estimator_training_seconds
    ):
        raise RuntimeError("fixed-time selective cache row cadence changed")


def _execution_config(config: FixedTimeSelectiveConfig) -> ExecutionEvidenceConfig:
    manifest = json.loads(
        (config.paths.execution_evidence / "manifest.json").read_text()
    )
    return ExecutionEvidenceConfig(
        range_start=datetime.fromisoformat(manifest["range_start"]),
        range_end=datetime.fromisoformat(manifest["range_end"]),
        output_dir=config.paths.execution_evidence,
        sample_interval_seconds=int(manifest["sample_interval_seconds"]),
        min_seconds_after_open=int(manifest["min_seconds_after_open"]),
        max_seconds_after_open=int(manifest["max_seconds_after_open"]),
        freshness_seconds=int(manifest["freshness_seconds"]),
        quantity=float(manifest["quantity"]),
    )


def _config_payload(config: FixedTimeSelectiveConfig) -> dict[str, Any]:
    return _json_value(asdict(config))


def _render_report(benchmark: dict[str, Any]) -> str:
    selected = benchmark["selection"]["selected_candidate"]
    rows: list[str] = []
    for name, candidate in benchmark["candidates"].items():
        primary = candidate["primary"]["metrics"]
        adverse = candidate["adverse_fold"]["metrics"]
        hard = candidate["primary"]["predicted_side_hard_confident_errors"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{'yes' if candidate['advancement_qualified'] else 'no'}</td>"
            f"<td>{primary['coverage']:.2%}</td>"
            f"<td>{primary['accuracy']:.2%}</td>"
            f"<td>{primary['up_recall']:.2%}</td>"
            f"<td>{primary['down_recall']:.2%}</td>"
            f"<td>{adverse['accuracy']:.2%}</td>"
            f"<td>{adverse['down_recall']:.2%}</td>"
            f"<td>{hard['all']['hard_confident_error_markets']}</td>"
            f"<td>{hard['up']['hard_confident_error_markets']}</td>"
            "</tr>"
        )
    selection_text = selected or "none"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC fixed-120 selective-accuracy benchmark</title>
<style>
body {{ font: 15px system-ui; margin: 2rem; color: #17202a; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d5d8dc; padding: .5rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
</style>
</head>
<body>
<h1>BTC fixed-120 selective-accuracy benchmark</h1>
<p><strong>Selected development candidate: {html.escape(selection_text)}</strong></p>
<p>Evidence through July 28 is consumed development evidence. July 29 onward is excluded.</p>
<p>Oracle and order-book fields are excluded from all model inputs.</p>
<p>The 15% operating point qualifies candidates; the 10% operating point is diagnostic.</p>
<table>
<thead><tr><th>Candidate</th><th>Qualified</th><th>Coverage</th>
<th>Accuracy</th><th>UP recall</th><th>DOWN recall</th>
<th>Adverse accuracy</th><th>Adverse DOWN</th>
<th>Hard errors</th><th>Hard false-UP</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>
"""


def freeze_and_export_fixed_time_selective_paper_candidate(
    *,
    config: FixedTimeSelectiveConfig,
    benchmark_run: Path,
    model_key: str,
    authorization: str,
) -> tuple[Path, Path, dict[str, Any]]:
    if authorization != FIXED_TIME_SELECTIVE_PAPER_AUTHORIZATION:
        raise RuntimeError(
            "fixed-time selective export requires explicit paper-only authorization"
        )
    benchmark_path = benchmark_run.resolve() / "benchmark.json"
    if not benchmark_path.is_file():
        raise RuntimeError("fixed-time selective benchmark evidence is missing")
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark.get("schema_version") != (
        FIXED_TIME_SELECTIVE_BENCHMARK_SCHEMA_VERSION
    ):
        raise RuntimeError("fixed-time selective benchmark schema changed")
    if benchmark.get("configuration") != _config_payload(config):
        raise RuntimeError("fixed-time selective benchmark configuration changed")
    selected_name = benchmark.get("selection", {}).get("selected_candidate")
    if not isinstance(selected_name, str):
        raise TypeError("fixed-time selective benchmark selected no candidate")
    selected_evidence = benchmark.get("candidates", {}).get(selected_name)
    if not isinstance(selected_evidence, dict) or not selected_evidence.get(
        "advancement_qualified"
    ):
        raise RuntimeError("selected candidate did not pass development checks")

    core_config = load_core_config(config.benchmark.core_config)
    feature_metadata = validate_core_feature_cache(core_config, "pre_holdout")
    _validate_feature_metadata(feature_metadata, config)
    frame = estimator_training_rows(
        load_core_feature_frame(core_config, "pre_holdout"),
        training_seconds=config.model.estimator_training_seconds,
        cohort_name="final fixed-time selective feature cache",
    )
    selected_config = _candidate_config(config, selected_name)
    if selected_config.name == FIXED_TIME_SELECTIVE_REGIME_CANDIDATE:
        raise RuntimeError(
            "the selected regime challenger requires Rust 77-feature runtime "
            "parity before paper export"
        )
    final = _fit_final_policy(
        frame=frame,
        candidate=selected_config,
        config=config,
        core_config=core_config,
    )
    control_config = _candidate_config(
        config,
        FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE,
    )
    control = _fit_final_policy(
        frame=frame,
        candidate=control_config,
        config=config,
        core_config=core_config,
    )

    execution = load_execution_evidence(_execution_config(config))
    selected_with_execution = attach_execution_evidence(final["selected"], execution)
    policy_metrics = classification_metrics(
        selected_with_execution,
        eligible_markets=final["policy"].height,
    )
    side_hard = predicted_side_hard_error_metrics(
        selected_with_execution,
        eligible_markets=final["policy"].height,
        confidence_floor=config.model.hard_confidence_floor,
    )
    control_same_rows = _control_predictions_on_selected_rows(
        final["selected"],
        control["scored"],
        keys=("market_id", "observed_at"),
    )
    control_hard = predicted_side_hard_error_metrics(
        control_same_rows,
        eligible_markets=control["policy"].height,
        confidence_floor=config.model.hard_confidence_floor,
    )
    same_rows = _same_row_final_control_comparison(
        final["selected"],
        control["scored"],
    )
    ten_share = _execution_metrics(
        selected_with_execution,
        quantity=10.0,
        vwap_depth=10,
    )
    policy_checks = operating_point_checks(policy_metrics, config.primary)
    policy_checks.extend(
        [
            _minimum_check(
                "final candidate accuracy on selected rows versus control",
                same_rows["accuracy_delta"],
                0.0,
            ),
            _maximum_check(
                "final same-row hard-error exposure does not regress control",
                side_hard["all"]["hard_confident_error_exposure_rate"],
                control_hard["all"]["hard_confident_error_exposure_rate"],
            ),
            _maximum_check(
                "final same-row hard-error selected rate does not regress control",
                side_hard["all"]["hard_confident_error_rate_selected"],
                control_hard["all"]["hard_confident_error_rate_selected"],
            ),
            _maximum_check(
                "final same-row predicted-UP hard-error rate does not regress control",
                side_hard["up"][
                    "hard_confident_error_rate_selected"
                ],
                control_hard["up"][
                    "hard_confident_error_rate_selected"
                ],
            ),
        ]
    )
    policy_checks.extend(_economics_checks(ten_share))
    passed = all(check["passed"] for check in policy_checks)
    assessment_path = benchmark_run.resolve() / "paper-freeze-assessment.json"
    assessment: dict[str, Any] = {
        "schema_version": FIXED_TIME_SELECTIVE_ASSESSMENT_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_run": str(benchmark_run.resolve()),
        "benchmark_sha256": file_sha256(benchmark_path),
        "candidate": selected_name,
        "feature_schema_version": selected_config.feature_schema_version,
        "decision_second": config.model.decision_second,
        "estimator_training_seconds": list(config.model.estimator_training_seconds),
        "development_evidence_only": True,
        "independently_qualified": False,
        "status": (
            "qualified_for_exploratory_paper_freeze"
            if passed
            else "blocked_by_final_policy"
        ),
        "threshold_selection": final["threshold_selection"],
        "policy_metrics": policy_metrics,
        "predicted_side_hard_confident_errors": side_hard,
        "control_same_rows_hard_confident_errors": control_hard,
        "same_selected_rows_control_comparison": same_rows,
        "ten_share_execution": ten_share,
        "policy_checks": policy_checks,
        "tuning": final["tuning"],
        "calibrator": asdict(final["calibrator"]),
        "runtime_model_created": False,
        "trading_process_created": False,
        "fresh_forward_evidence_required": True,
        "fresh_forward_evidence_start": (
            config.split.policy_selection_end.isoformat()
        ),
    }
    write_json_atomic(assessment_path, assessment)
    if not passed:
        raise RuntimeError(
            "final fixed-time selective policy did not pass paper checks; "
            f"assessment: {assessment_path}"
        )

    bundle = FrozenTrainingBundle(
        model=final["model"],
        calibrator=final["calibrator"],
        confidence_threshold=final["threshold"],
    )
    created_at = datetime.now(UTC)
    freeze_id = (
        f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{selected_name}-paper"
    )
    freeze_dir = config.paths.freezes / freeze_id
    freeze_dir.mkdir(parents=True, exist_ok=False)
    model_path = freeze_dir / TRAINING_MODEL_FILENAME
    joblib.dump(bundle, model_path, compress=3)
    summary_path = freeze_dir / "model-summary.json"
    summary = model_summary_payload(bundle)
    summary.update(
        {
            "fixed_time_selective_freeze_schema_version": (
                FIXED_TIME_SELECTIVE_FREEZE_SCHEMA_VERSION
            ),
            "deployment_scope": "paper_only",
            "production_qualified": False,
            "live_capital_allowed": False,
        }
    )
    write_json_atomic(summary_path, summary)
    golden_path = freeze_dir / GOLDEN_FEATURES_FILENAME
    write_golden_feature_sample(
        final["policy"],
        final["probability"],
        bundle,
        golden_path,
        feature_schema_version=selected_config.feature_schema_version,
    )

    provenance = runtime_provenance(config.package_root)
    frozen_spec = model_candidate_spec(bundle.model)
    feature_path = core_config.paths.development_feature_data
    feature_metadata_path = feature_path.with_suffix(".metadata.json")
    manifest: dict[str, Any] = {
        "schema_version": CORE_FREEZE_SCHEMA_VERSION,
        "fixed_time_selective_freeze_schema_version": (
            FIXED_TIME_SELECTIVE_FREEZE_SCHEMA_VERSION
        ),
        "freeze_id": freeze_id,
        "created_at": created_at.isoformat(),
        "status": "exploratory_paper_candidate_frozen",
        "deployment_status": "paper_only_forward_evaluation",
        "deployment_scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
        "paper_only_authorization": authorization,
        "model_file": TRAINING_MODEL_FILENAME,
        "model_sha256": file_sha256(model_path),
        "model_summary_sha256": file_sha256(summary_path),
        "candidate": bundle.model.candidate_name,
        "family": bundle.model.family,
        "row_weight_policy": frozen_spec.row_weight_policy,
        "row_weight_schedule": row_weight_schedule_payload(frozen_spec),
        "recency_half_life_days": frozen_spec.recency_half_life_days,
        "feature_schema_version": selected_config.feature_schema_version,
        "feature_names": list(bundle.model.feature_names),
        "hyperparameters": bundle.model.hyperparameters,
        "calibrator": asdict(bundle.calibrator),
        "confidence_threshold": bundle.confidence_threshold,
        "prediction_policy": {
            "type": "first_confidence_crossing",
            "minimum_seconds_after_open": config.model.decision_second,
            "maximum_seconds_after_open": config.model.decision_second,
            "cadence_seconds": 5,
        },
        "configuration_sha256": file_sha256(config.source_path),
        "core_configuration_sha256": file_sha256(config.benchmark.core_config),
        "development_feature_sha256": file_sha256(feature_path),
        "development_feature_metadata_sha256": file_sha256(
            feature_metadata_path
        ),
        "golden_feature_file": golden_path.name,
        "golden_feature_sha256": file_sha256(golden_path),
        "golden_feature_metadata_sha256": file_sha256(
            golden_path.with_suffix(".metadata.json")
        ),
        "source_tree_sha256": provenance["source_tree_sha256"],
        "git": provenance["git"],
        "runtime_provenance": provenance,
        "random_seed": core_config.model.random_seed,
        "training_ranges": {
            "development": _range_payload(
                config.split.development_start,
                config.split.development_end,
            ),
            "probability_calibration": _range_payload(
                config.split.probability_calibration_start,
                config.split.probability_calibration_end,
            ),
            "policy_selection": _range_payload(
                config.split.policy_selection_start,
                config.split.policy_selection_end,
            ),
        },
        "holdout_range": {
            "start": config.split.policy_selection_end.isoformat(),
            "end": config.split.policy_selection_end.isoformat(),
        },
        "forward_paper_evaluation": {
            "start": config.split.policy_selection_end.isoformat(),
            "end": None,
            "intended_independent": True,
            "independent_evidence_available": False,
            "status": "pending",
        },
        "policy_selection": {
            **final["threshold_selection"],
            "metrics": policy_metrics,
            "checks": policy_checks,
        },
        "benchmark_evidence": {
            "run": str(benchmark_run.resolve()),
            "benchmark_sha256": file_sha256(benchmark_path),
            "development_candidate_selected": True,
            "independently_qualified": False,
        },
        "tuning": final["tuning"],
    }
    manifest_path = freeze_dir / "freeze-manifest.json"
    write_json_atomic(manifest_path, manifest)
    (freeze_dir / "freeze-manifest.sha256").write_text(
        file_sha256(manifest_path) + "\n"
    )
    runtime_dir = export_runtime_model(
        freeze_dir=freeze_dir,
        golden_features=golden_path,
        output_root=config.paths.runtime_models,
        model_key=model_key,
    )
    assessment.update(
        {
            "status": "exploratory_paper_runtime_exported",
            "runtime_model_created": True,
            "runtime_model": str(runtime_dir),
        }
    )
    write_json_atomic(assessment_path, assessment)
    return freeze_dir, runtime_dir, manifest


def _fit_final_policy(
    *,
    frame: pl.DataFrame,
    candidate: FixedTimeSelectiveCandidateConfig,
    config: FixedTimeSelectiveConfig,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    development = estimator_training_rows(
        range_frame(
            frame,
            config.split.development_start,
            config.split.development_end,
        ),
        training_seconds=config.model.estimator_training_seconds,
        cohort_name=f"{candidate.name} final development",
    )
    calibration = fixed_time_rows(
        range_frame(
            frame,
            config.split.probability_calibration_start,
            config.split.probability_calibration_end,
        ),
        decision_second=config.model.decision_second,
        cohort_name=f"{candidate.name} final calibration",
    )
    policy = fixed_time_rows(
        range_frame(
            frame,
            config.split.policy_selection_start,
            config.split.policy_selection_end,
        ),
        decision_second=config.model.decision_second,
        cohort_name=f"{candidate.name} final policy",
    )
    spec = selective_candidate_spec(candidate, config.model)
    parameter_grid = selective_parameter_grid(candidate, config.model, core_config)
    model, tuning = tune_and_fit_selective_model(
        development,
        spec,
        parameter_grid,
        core_config,
        config.model.decision_second,
        config.primary.target_coverage,
        config.secondary.target_coverage,
        config.model.hard_confidence_floor,
    )
    calibrator = fit_probability_calibrator(model, calibration, core_config, spec)
    if not calibrator.converged or calibrator.slope <= 0.0:
        raise RuntimeError(f"{candidate.name} final calibrator failed")
    probability = calibrator.probability(model.raw_logit(policy))
    scored = scored_prediction_rows(policy, probability)
    threshold, selected, threshold_selection = empirical_coverage_threshold(
        scored,
        target_coverage=config.primary.target_coverage,
    )
    return {
        "candidate": candidate.name,
        "model": model,
        "tuning": tuning,
        "calibrator": calibrator,
        "policy": policy,
        "probability": probability,
        "scored": scored,
        "threshold": threshold,
        "threshold_selection": threshold_selection,
        "selected": selected,
    }


def _same_row_final_control_comparison(
    candidate_selected: pl.DataFrame,
    control_scored: pl.DataFrame,
) -> dict[str, Any]:
    keys = ["market_id", "observed_at"]
    control = control_scored.select(
        *keys,
        pl.col("predicted_up").alias("control_predicted_up"),
        pl.col("correct").alias("control_correct"),
    )
    joined = candidate_selected.join(control, on=keys, how="inner", validate="1:1")
    if joined.height != candidate_selected.height:
        raise RuntimeError("final candidate rows do not match final control rows")
    markets = joined.height
    candidate_correct = joined["correct"].cast(pl.Int64).sum()
    control_correct = joined["control_correct"].cast(pl.Int64).sum()
    return {
        "markets": markets,
        "candidate_accuracy": candidate_correct / markets if markets else 0.0,
        "control_accuracy": control_correct / markets if markets else 0.0,
        "accuracy_delta": (
            (candidate_correct - control_correct) / markets if markets else 0.0
        ),
        "direction_disagreements": joined.filter(
            pl.col("predicted_up") != pl.col("control_predicted_up")
        ).height,
    }


def _range_payload(start: datetime, end: datetime) -> dict[str, str]:
    return {"start": start.isoformat(), "end": end.isoformat()}
