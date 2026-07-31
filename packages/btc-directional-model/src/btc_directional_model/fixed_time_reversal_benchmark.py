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
    FrozenCalibrationBand,
    FrozenTimeBandedTrainingBundle,
    chronological_subsplit,
    configure_native_thread_limits,
    fit_probability_calibrator,
    model_candidate_spec,
    row_weight_schedule_payload,
)
from .fixed_time_benchmark import (
    GOLDEN_FEATURES_FILENAME,
    _cohort_payload,
    _economics_checks,
    _json_value,
    _write_text_atomic,
    empirical_coverage_threshold,
    fixed_time_rows,
    operating_point_checks,
)
from .fixed_time_reversal_config import (
    FIXED_TIME_REVERSAL_CONTROL_CANDIDATE,
    PATH_PERSISTENCE_TARGET,
    FixedTimeReversalCandidateConfig,
    FixedTimeReversalConfig,
    load_fixed_time_reversal_config,
)
from .fixed_time_reversal_training import (
    eligible_complete_market_frame,
    reversal_candidate_spec,
    reversal_parameter_grid,
    reversal_probability_semantics,
    training_target_frame,
    tune_and_fit_reversal_model,
)
from .paper_candidate import write_time_banded_golden_feature_sample
from .persistence_benchmark import attach_execution_evidence, load_execution_evidence
from .provenance import runtime_provenance
from .runtime_export import export_runtime_model, frozen_calibration_bands_payload

FIXED_TIME_REVERSAL_BENCHMARK_SCHEMA_VERSION = "btc-mature-reversal-fixed-120-decision-benchmark-v1"
FIXED_TIME_REVERSAL_FREEZE_SCHEMA_VERSION = "btc-mature-reversal-fixed-120-decision-freeze-v1"
FIXED_TIME_REVERSAL_ASSESSMENT_SCHEMA_VERSION = (
    "btc-mature-reversal-fixed-120-decision-freeze-assessment-v1"
)
FIXED_TIME_REVERSAL_PAPER_AUTHORIZATION = "explicit-fixed-120-reversal-development-paper-only"
SCORED_PROBABILITY_SUFFIX = "-scored-probabilities.parquet"
ADVERSE_MINIMUM_ACCURACY = 0.90
ADVERSE_MINIMUM_BALANCED_ACCURACY = 0.89
ADVERSE_MINIMUM_DIRECTION_RECALL = 0.88
ADVERSE_MAXIMUM_ECE = 0.07


def run_fixed_time_reversal_benchmark(
    config: FixedTimeReversalConfig,
) -> tuple[Path, dict[str, Any]]:
    core_config = load_core_config(config.benchmark.core_config)
    configure_native_thread_limits(core_config)
    feature_metadata = validate_core_feature_cache(core_config, "pre_holdout")
    _validate_feature_metadata(feature_metadata, config)
    full_frame = eligible_complete_market_frame(
        load_core_feature_frame(core_config, "pre_holdout"),
        decision_seconds=config.model.estimator_training_seconds,
        expected_source_markets=config.model.expected_source_markets,
        expected_source_estimator_rows=(config.model.expected_source_estimator_rows),
        expected_eligible_markets=config.model.expected_eligible_markets,
        expected_eligible_estimator_rows=(config.model.expected_eligible_estimator_rows),
        expected_exact_120_eligible_markets=(config.model.expected_exact_120_eligible_markets),
    )
    _validate_eligible_cohort(full_frame, config)

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
        by_candidate[candidate.name] = folds
        scored_by_candidate[candidate.name] = pl.concat(
            [fold.pop("scored_rows") for fold in folds],
            how="vertical_relaxed",
        ).sort(["fold_index", "observed_at", "market_id"])
    _assert_identical_scored_universes(scored_by_candidate)

    execution = load_execution_evidence(_execution_config(config))
    control_scored = scored_by_candidate[FIXED_TIME_REVERSAL_CONTROL_CANDIDATE]
    candidates: dict[str, dict[str, Any]] = {}
    for candidate in config.candidates:
        scored = scored_by_candidate[candidate.name]
        payload = _candidate_payload(
            candidate=candidate,
            folds=by_candidate[candidate.name],
            scored=scored,
            control_scored=control_scored,
            execution=execution,
            config=config,
        )
        candidates[candidate.name] = payload
        scored.write_parquet(
            run_dir / f"{candidate.name}{SCORED_PROBABILITY_SUFFIX}",
            compression="zstd",
        )

    selection = _select_candidate(config, candidates)
    benchmark: dict[str, Any] = {
        "schema_version": FIXED_TIME_REVERSAL_BENCHMARK_SCHEMA_VERSION,
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
            "source_complete_estimator_markets": (config.model.expected_source_markets),
            "source_complete_estimator_rows": (config.model.expected_source_estimator_rows),
            "contemporaneously_eligible_estimator_markets": (full_frame["market_id"].n_unique()),
            "contemporaneously_eligible_estimator_rows": full_frame.height,
            "exact_120_eligible_markets": full_frame.filter(
                pl.col("seconds_elapsed") == config.model.decision_second
            ).height,
            "path_eligibility_policy": "contemporaneous_row_only",
            "future_path_conditioning": False,
            "estimator_rows_per_market_after_path_filter": "variable",
            "candidate_control_exact_120_universe_identical": True,
            "post_july_28_rows_used": False,
            "official_outcome_label_preserved_for_scoring": True,
            "target_label_transformed_only_for_reversal_fit_and_calibration": True,
            "oracle_role": "excluded_from_model_inputs",
            "orderbook_role": "common_vwap10_execution_economics_only",
            "feature_metadata": feature_metadata,
        },
        "training_contract": {
            "decision_second": config.model.decision_second,
            "estimator_training_seconds": list(config.model.estimator_training_seconds),
            "calibration_policy_validation_seconds": [config.model.decision_second],
            "candidate_count": len(config.candidates),
            "fold_count": len(config.split.validation_windows),
            "primary_coverage": config.primary.target_coverage,
            "diagnostic_coverage": config.diagnostic.target_coverage,
            "candidate_order": [candidate.name for candidate in config.candidates],
            "control_never_advances": True,
            "every_fold_raw_uplift_is_ranking_evidence_not_a_gate": True,
        },
        "candidates": candidates,
        "selection": selection,
        "provenance": {
            "benchmark_config": str(config.source_path),
            "benchmark_config_sha256": file_sha256(config.source_path),
            "core_config": str(config.benchmark.core_config),
            "core_config_sha256": file_sha256(config.benchmark.core_config),
            "development_feature": str(core_config.paths.development_feature_data),
            "development_feature_sha256": file_sha256(core_config.paths.development_feature_data),
            "execution_manifest_sha256": config.paths.execution_manifest_sha256,
            "runtime_provenance": runtime_provenance(config.package_root),
        },
    }
    write_json_atomic(run_dir / "benchmark.json", benchmark)
    _write_text_atomic(run_dir / "report.html", _render_report(benchmark))
    return run_dir, benchmark


def _run_candidate_fold_matrix(
    config: FixedTimeReversalConfig,
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
            primary = result["primary"]["metrics"]
            raw = result["raw"]["sign_override_metrics"]
            print(
                f"{key[0]} fold {key[1] + 1}/{len(config.split.validation_windows)}: "
                f"accuracy={primary['accuracy']:.4f} "
                f"coverage={primary['coverage']:.4f} "
                f"raw_uplift={raw['sign_baseline_accuracy_uplift']:+.4f}",
                flush=True,
            )
    return [completed[key] for key in tasks]


def _evaluate_candidate_fold_task(
    config_path: Path,
    candidate_name: str,
    fold_index: int,
) -> dict[str, Any]:
    config = load_fixed_time_reversal_config(config_path)
    core_config = load_core_config(config.benchmark.core_config)
    configure_native_thread_limits(core_config)
    candidate = _candidate_config(config, candidate_name)
    with threadpool_limits(limits=core_config.compute.threads_per_fit):
        frame = eligible_complete_market_frame(
            load_core_feature_frame(core_config, "pre_holdout"),
            decision_seconds=config.model.estimator_training_seconds,
            expected_source_markets=config.model.expected_source_markets,
            expected_source_estimator_rows=(config.model.expected_source_estimator_rows),
            expected_eligible_markets=config.model.expected_eligible_markets,
            expected_eligible_estimator_rows=(config.model.expected_eligible_estimator_rows),
            expected_exact_120_eligible_markets=(config.model.expected_exact_120_eligible_markets),
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
    candidate: FixedTimeReversalCandidateConfig,
    fold_index: int,
    config: FixedTimeReversalConfig,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    validation_start, validation_end = config.split.validation_windows[fold_index]
    history = _range_frame(frame, config.split.development_start, validation_start)
    validation_context = _range_frame(frame, validation_start, validation_end)
    fit, calibration_context, policy_context = chronological_subsplit(
        history,
        fit_fraction=0.70,
        calibration_fraction=0.15,
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
    spec = reversal_candidate_spec(candidate, config.model)
    parameter_grid = reversal_parameter_grid(candidate, config.model, core_config)
    model, tuning = tune_and_fit_reversal_model(
        fit,
        spec,
        parameter_grid,
        core_config,
        candidate.target_kind,
        config.model.decision_second,
        config.primary.target_coverage,
        config.diagnostic.target_coverage,
    )
    calibrator = fit_probability_calibrator(
        model,
        training_target_frame(calibration, candidate.target_kind),
        core_config,
        spec,
    )
    if not calibrator.converged or calibrator.slope <= 0.0:
        raise RuntimeError(f"{candidate.name} fold {fold_index} calibration failed")

    policy_scored = _score_candidate_rows(
        policy,
        candidate=candidate,
        model=model,
        calibrator=calibrator,
        fold_index=fold_index,
    )
    primary_threshold, primary_policy, primary_selection = empirical_coverage_threshold(
        policy_scored,
        target_coverage=config.primary.target_coverage,
    )
    diagnostic_threshold, diagnostic_policy, diagnostic_selection = empirical_coverage_threshold(
        policy_scored,
        target_coverage=config.diagnostic.target_coverage,
    )
    scored = _score_candidate_rows(
        validation,
        candidate=candidate,
        model=model,
        calibrator=calibrator,
        fold_index=fold_index,
    ).with_columns(
        pl.lit(primary_threshold).alias("primary_confidence_threshold"),
        pl.lit(diagnostic_threshold).alias("diagnostic_confidence_threshold"),
        (pl.col("confidence") >= primary_threshold).alias("primary_selected"),
        (pl.col("confidence") >= diagnostic_threshold).alias("diagnostic_selected"),
    )
    primary_selected = scored.filter(pl.col("primary_selected"))
    diagnostic_selected = scored.filter(pl.col("diagnostic_selected"))
    return {
        "candidate": candidate.name,
        "target_kind": candidate.target_kind,
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
            "target_kind": candidate.target_kind,
            "feature_schema_version": candidate.feature_schema_version,
            "feature_count": len(spec.feature_names),
            "feature_names": list(spec.feature_names),
            "parameter_grid": candidate.parameter_grid,
        },
        "tuning": tuning,
        "calibrator": asdict(calibrator),
        "raw": _operating_point(
            scored,
            eligible_markets=validation.height,
            contract=None,
            hard_confidence_floor=config.model.hard_confidence_floor,
        ),
        "primary": _fold_operating_point(
            primary_selected,
            eligible_markets=validation.height,
            policy_selected=primary_policy,
            policy_eligible_markets=policy.height,
            threshold_selection=primary_selection,
            contract=config.primary,
            hard_confidence_floor=config.model.hard_confidence_floor,
        ),
        "diagnostic": _fold_operating_point(
            diagnostic_selected,
            eligible_markets=validation.height,
            policy_selected=diagnostic_policy,
            policy_eligible_markets=policy.height,
            threshold_selection=diagnostic_selection,
            contract=config.diagnostic,
            hard_confidence_floor=config.model.hard_confidence_floor,
        ),
        "threshold_selection_used_validation_labels": False,
        "elapsed_seconds": time.perf_counter() - started,
        "scored_rows": scored,
    }


def _score_candidate_rows(
    frame: pl.DataFrame,
    *,
    candidate: FixedTimeReversalCandidateConfig,
    model: Any,
    calibrator: Any,
    fold_index: int,
) -> pl.DataFrame:
    target_probability = calibrator.probability(model.raw_logit(frame))
    semantics = reversal_probability_semantics(
        frame,
        target_probability,
        candidate.target_kind,
    )
    scored = (
        scored_prediction_rows(frame, semantics["probability_up"])
        .with_columns(
            pl.Series("p_target", semantics["p_target"]),
            pl.Series("p_persistence", semantics["p_persistence"]),
            pl.Series("p_reversal", semantics["p_reversal"]),
            pl.lit(candidate.name).alias("candidate"),
            pl.lit(candidate.target_kind).alias("target_kind"),
            pl.lit(fold_index).cast(pl.Int32).alias("fold_index"),
        )
        .with_columns(
            (pl.col("predicted_up") != pl.col("binance_sign_up")).alias("path_overridden"),
        )
        .with_columns(pl.col("path_overridden").alias("predicted_reversal"))
    )
    return scored


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
    result = _operating_point(
        selected,
        eligible_markets=eligible_markets,
        contract=contract,
        hard_confidence_floor=hard_confidence_floor,
    )
    result.update(
        {
            "threshold_selection": threshold_selection,
            "policy_metrics_after_selection": classification_metrics(
                policy_selected,
                eligible_markets=policy_eligible_markets,
            ),
            "policy_sign_override_metrics_after_selection": sign_override_metrics(
                policy_selected,
                eligible_markets=policy_eligible_markets,
            ),
        }
    )
    return result


def _operating_point(
    rows: pl.DataFrame,
    *,
    eligible_markets: int,
    contract: Any | None,
    hard_confidence_floor: float,
) -> dict[str, Any]:
    metrics = classification_metrics(rows, eligible_markets=eligible_markets)
    checks = operating_point_checks(metrics, contract) if contract is not None else []
    return {
        "metrics": metrics,
        "sign_override_metrics": sign_override_metrics(
            rows,
            eligible_markets=eligible_markets,
        ),
        "hard_false_up_metrics": hard_false_up_metrics(
            rows,
            eligible_markets=eligible_markets,
            confidence_floor=hard_confidence_floor,
        ),
        "checks": checks,
        "quality_qualified": all(check["passed"] for check in checks),
    }


def sign_override_metrics(
    rows: pl.DataFrame,
    *,
    eligible_markets: int,
) -> dict[str, Any]:
    selected = rows.height
    if eligible_markets < selected:
        raise ValueError("eligible markets cannot be fewer than selected markets")
    actual_reversal = pl.col("binance_sign_up") != pl.col("label_up")
    override = pl.col("predicted_up") != pl.col("binance_sign_up")
    correct_override = override & pl.col("correct")
    incorrect_override = override & ~pl.col("correct")
    false_up = (pl.col("predicted_up") == 1) & (pl.col("label_up") == 0)
    actual_reversals = rows.filter(actual_reversal).height
    overrides = rows.filter(override).height
    correct_overrides = rows.filter(correct_override).height
    incorrect_overrides = rows.filter(incorrect_override).height
    missed_reversals = rows.filter(actual_reversal & ~override).height
    model_correct = int(rows["correct"].sum() or 0) if selected else 0
    baseline_correct = int(rows["baseline_correct"].sum() or 0) if selected else 0
    uplift_numerator = correct_overrides - incorrect_overrides
    observed_uplift_numerator = model_correct - baseline_correct
    if uplift_numerator != observed_uplift_numerator:
        raise RuntimeError("sign-baseline uplift identity failed")
    model_accuracy = _ratio(model_correct, selected)
    sign_baseline_accuracy = _ratio(baseline_correct, selected)
    selected_conditional_uplift = _ratio(uplift_numerator, selected)
    eligible_exposure_uplift = _ratio(uplift_numerator, eligible_markets)
    selected_coverage = _ratio(selected, eligible_markets)
    selected_accuracy_delta = model_accuracy - sign_baseline_accuracy
    model_correct_exposure = _ratio(model_correct, eligible_markets)
    sign_baseline_correct_exposure = _ratio(
        baseline_correct,
        eligible_markets,
    )
    correct_exposure_delta = model_correct_exposure - sign_baseline_correct_exposure
    coverage_scaled_uplift = selected_coverage * selected_conditional_uplift
    if abs(selected_accuracy_delta - selected_conditional_uplift) > 1e-12:
        raise RuntimeError("selected-conditional uplift identity failed")
    if abs(correct_exposure_delta - eligible_exposure_uplift) > 1e-12:
        raise RuntimeError("eligible-exposure uplift identity failed")
    if abs(coverage_scaled_uplift - eligible_exposure_uplift) > 1e-12:
        raise RuntimeError("coverage-scaled uplift identity failed")
    false_up_rows = rows.filter(false_up)
    false_up_followed = false_up_rows.filter(
        pl.col("predicted_up") == pl.col("binance_sign_up")
    ).height
    false_up_bad_override = false_up_rows.filter(
        pl.col("predicted_up") != pl.col("binance_sign_up")
    ).height
    return {
        "markets": selected,
        "eligible_markets": eligible_markets,
        "actual_reversals": actual_reversals,
        "actual_reversal_rate": _ratio(actual_reversals, selected),
        "overrides": overrides,
        "override_exposure_rate": _ratio(overrides, eligible_markets),
        "override_rate_selected": _ratio(overrides, selected),
        "correct_overrides": correct_overrides,
        "incorrect_overrides": incorrect_overrides,
        "override_precision": _optional_ratio(correct_overrides, overrides),
        "reversal_recall": _optional_ratio(correct_overrides, actual_reversals),
        "missed_reversals": missed_reversals,
        "followed_sign_decisions": selected - overrides,
        "model_accuracy": model_accuracy,
        "sign_baseline_accuracy": sign_baseline_accuracy,
        "sign_baseline_accuracy_uplift": selected_conditional_uplift,
        "selected_conditional_accuracy_uplift": selected_conditional_uplift,
        "eligible_exposure_accuracy_uplift": eligible_exposure_uplift,
        "uplift_identity": {
            "correct_overrides_minus_incorrect_overrides": uplift_numerator,
            "model_correct_minus_sign_baseline_correct": observed_uplift_numerator,
            "verified": True,
        },
        "selected_conditional_uplift_identity": {
            "denominator_markets": selected,
            "override_net_correct": uplift_numerator,
            "override_net_accuracy_uplift": selected_conditional_uplift,
            "model_minus_sign_baseline_accuracy": selected_accuracy_delta,
            "verified": True,
        },
        "eligible_exposure_uplift_identity": {
            "denominator_markets": eligible_markets,
            "override_net_correct": uplift_numerator,
            "override_net_accuracy_uplift": eligible_exposure_uplift,
            "model_minus_sign_baseline_correct_exposure": (correct_exposure_delta),
            "selected_coverage_times_conditional_uplift": (coverage_scaled_uplift),
            "verified": True,
        },
        "false_up_markets": false_up_rows.height,
        "false_up_followed_sign": false_up_followed,
        "false_up_bad_override": false_up_bad_override,
        "false_up_rate_selected": _ratio(false_up_rows.height, selected),
    }


def hard_false_up_metrics(
    rows: pl.DataFrame,
    *,
    eligible_markets: int,
    confidence_floor: float,
) -> dict[str, Any]:
    hard = rows.filter((pl.col("confidence") >= confidence_floor) & ~pl.col("correct"))
    false_up = hard.filter((pl.col("predicted_up") == 1) & (pl.col("label_up") == 0))
    followed = false_up.filter(pl.col("predicted_up") == pl.col("binance_sign_up"))
    bad_override = false_up.filter(pl.col("predicted_up") != pl.col("binance_sign_up"))
    return {
        "confidence_floor": confidence_floor,
        "selected_markets": rows.height,
        "eligible_markets": eligible_markets,
        "hard_error_markets": hard.height,
        "hard_error_exposure_rate": _ratio(hard.height, eligible_markets),
        "hard_error_rate_selected": _ratio(hard.height, rows.height),
        "hard_false_up_markets": false_up.height,
        "hard_false_up_exposure_rate": _ratio(false_up.height, eligible_markets),
        "hard_false_up_rate_selected": _ratio(false_up.height, rows.height),
        "hard_false_up_followed_sign": followed.height,
        "hard_false_up_bad_override": bad_override.height,
    }


def _candidate_payload(
    *,
    candidate: FixedTimeReversalCandidateConfig,
    folds: list[dict[str, Any]],
    scored: pl.DataFrame,
    control_scored: pl.DataFrame,
    execution: pl.DataFrame,
    config: FixedTimeReversalConfig,
) -> dict[str, Any]:
    with_execution = attach_execution_evidence(scored, execution)
    primary_selected = with_execution.filter(pl.col("primary_selected"))
    diagnostic_selected = with_execution.filter(pl.col("diagnostic_selected"))
    eligible_markets = scored.height
    primary = _aggregate_operating_point(
        primary_selected,
        eligible_markets=eligible_markets,
        contract=config.primary,
        hard_confidence_floor=config.model.hard_confidence_floor,
        include_execution=True,
    )
    diagnostic = _aggregate_operating_point(
        diagnostic_selected,
        eligible_markets=eligible_markets,
        contract=config.diagnostic,
        hard_confidence_floor=config.model.hard_confidence_floor,
        include_execution=False,
    )
    raw = _aggregate_operating_point(
        scored,
        eligible_markets=eligible_markets,
        contract=None,
        hard_confidence_floor=config.model.hard_confidence_floor,
        include_execution=False,
    )

    fold_execution = _attach_fold_execution_diagnostics(folds, with_execution)
    fold_summary = _fold_summary(folds)
    adverse = folds[-1]
    same_rows = _same_row_control_comparison(
        primary_selected,
        control_scored,
        hard_confidence_floor=config.model.hard_confidence_floor,
    )
    raw_control = _raw_control_comparison(scored, control_scored)
    raw_uplift = raw["sign_override_metrics"]["sign_baseline_accuracy_uplift"]
    adverse_raw_uplift = adverse["raw"]["sign_override_metrics"]["sign_baseline_accuracy_uplift"]
    override = primary["sign_override_metrics"]
    override_precision = override["override_precision"]
    adverse_metrics = adverse["primary"]["metrics"]

    checks = list(primary["checks"])
    checks.extend(
        [
            _minimum_check(
                "aggregate raw accuracy uplift over Binance sign",
                raw_uplift,
                0.0,
                strict=True,
            ),
            _minimum_check(
                "adverse-fold raw accuracy uplift over Binance sign",
                adverse_raw_uplift,
                0.0,
            ),
            _minimum_check(
                "selected overrides exist",
                float(override["overrides"]),
                0.0,
                strict=True,
            ),
            _minimum_check(
                "selected override precision exceeds chance",
                float(override_precision) if override_precision is not None else 0.0,
                0.5,
                strict=True,
            ),
            _maximum_check(
                "same-row hard-error rate does not regress control",
                same_rows["candidate_hard"]["hard_error_rate_selected"],
                same_rows["control_hard"]["hard_error_rate_selected"],
            ),
            _maximum_check(
                "same-row hard false-UP rate does not regress control",
                same_rows["candidate_hard"]["hard_false_up_rate_selected"],
                same_rows["control_hard"]["hard_false_up_rate_selected"],
            ),
            _minimum_check(
                "adverse-fold primary accuracy",
                adverse_metrics["accuracy"],
                ADVERSE_MINIMUM_ACCURACY,
            ),
            _minimum_check(
                "adverse-fold primary balanced accuracy",
                adverse_metrics["balanced_accuracy"],
                ADVERSE_MINIMUM_BALANCED_ACCURACY,
            ),
            _minimum_check(
                "adverse-fold primary UP recall",
                adverse_metrics["up_recall"],
                ADVERSE_MINIMUM_DIRECTION_RECALL,
            ),
            _minimum_check(
                "adverse-fold primary DOWN recall",
                adverse_metrics["down_recall"],
                ADVERSE_MINIMUM_DIRECTION_RECALL,
            ),
            _maximum_check(
                "adverse-fold primary expected calibration error",
                adverse_metrics["expected_calibration_error"],
                ADVERSE_MAXIMUM_ECE,
            ),
        ]
    )
    checks.extend(_economics_checks(primary["ten_share_vwap10_execution"]))
    is_control = candidate.name == FIXED_TIME_REVERSAL_CONTROL_CANDIDATE
    is_reversal = candidate.target_kind == PATH_PERSISTENCE_TARGET
    advancement_qualified = (
        not is_control and is_reversal and all(check["passed"] for check in checks)
    )
    return {
        "candidate": candidate.name,
        "target_kind": candidate.target_kind,
        "is_control": is_control,
        "is_reversal_candidate": is_reversal,
        "feature_schema_version": candidate.feature_schema_version,
        "feature_count": len(candidate.feature_names),
        "feature_names": list(candidate.feature_names),
        "parameter_grid": candidate.parameter_grid,
        "folds": folds,
        "fold_summary": fold_summary,
        "fold_execution_diagnostics": fold_execution,
        "adverse_fold": adverse,
        "raw": raw,
        "primary": primary,
        "diagnostic": diagnostic,
        "diagnostic_advancement_role": "diagnostic_only",
        "same_selected_rows_control_comparison": same_rows,
        "raw_exact_120_control_comparison": raw_control,
        "checks": checks,
        "advancement_qualified": advancement_qualified,
    }


def _aggregate_operating_point(
    rows: pl.DataFrame,
    *,
    eligible_markets: int,
    contract: Any | None,
    hard_confidence_floor: float,
    include_execution: bool,
) -> dict[str, Any]:
    result = _operating_point(
        rows,
        eligible_markets=eligible_markets,
        contract=contract,
        hard_confidence_floor=hard_confidence_floor,
    )
    if include_execution:
        result["ten_share_vwap10_execution"] = _execution_metrics(
            rows,
            quantity=10.0,
            vwap_depth=10,
        )
    return result


def _attach_fold_execution_diagnostics(
    folds: list[dict[str, Any]],
    with_execution: pl.DataFrame,
) -> dict[str, Any]:
    available: list[int] = []
    unavailable: list[int] = []
    for fold in folds:
        fold_index = int(fold["fold_index"])
        rows = with_execution.filter(
            (pl.col("fold_index") == fold_index) & pl.col("primary_selected")
        )
        execution = _execution_metrics(rows, quantity=10.0, vwap_depth=10)
        fold["primary"]["ten_share_vwap10_execution_diagnostic"] = execution
        if execution["economics_available"]:
            available.append(fold_index)
        else:
            unavailable.append(fold_index)
    return {
        "role": "diagnostic_only_at_fold_level",
        "common_execution_contract": "strict two-sided executable VWAP10",
        "available_fold_indices": available,
        "unavailable_fold_indices": unavailable,
        "unavailable_book_folds_fail_advancement": False,
    }


def _fold_summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [fold["primary"]["metrics"] for fold in folds]
    raw_uplifts = [
        fold["raw"]["sign_override_metrics"]["sign_baseline_accuracy_uplift"] for fold in folds
    ]
    return {
        "fold_count": len(folds),
        "minimum_primary_accuracy": min(item["accuracy"] for item in primary),
        "maximum_primary_accuracy": max(item["accuracy"] for item in primary),
        "minimum_primary_balanced_accuracy": min(item["balanced_accuracy"] for item in primary),
        "minimum_primary_up_recall": min(item["up_recall"] for item in primary),
        "minimum_primary_down_recall": min(item["down_recall"] for item in primary),
        "maximum_primary_expected_calibration_error": max(
            item["expected_calibration_error"] for item in primary
        ),
        "minimum_primary_coverage": min(item["coverage"] for item in primary),
        "maximum_primary_coverage": max(item["coverage"] for item in primary),
        "raw_sign_baseline_uplift_by_fold": [
            {
                "fold_index": int(fold["fold_index"]),
                "accuracy_uplift": uplift,
            }
            for fold, uplift in zip(folds, raw_uplifts, strict=True)
        ],
        "minimum_raw_sign_baseline_accuracy_uplift": min(raw_uplifts),
        "maximum_raw_sign_baseline_accuracy_uplift": max(raw_uplifts),
        "nonnegative_raw_uplift_folds": sum(value >= 0.0 for value in raw_uplifts),
        "all_fold_raw_uplift_is_ranking_evidence_only": True,
    }


def _same_row_control_comparison(
    candidate_selected: pl.DataFrame,
    control_scored: pl.DataFrame,
    *,
    hard_confidence_floor: float,
) -> dict[str, Any]:
    keys = ["fold_index", "market_id", "observed_at"]
    control = control_scored.select(
        *keys,
        "label_up",
        "binance_sign_up",
        pl.col("probability_up").alias("control_probability_up"),
        pl.col("predicted_up").alias("control_predicted_up"),
        pl.col("confidence").alias("control_confidence"),
        pl.col("correct").alias("control_correct"),
    )
    joined = candidate_selected.join(
        control,
        on=keys,
        how="inner",
        validate="1:1",
        coalesce=True,
    )
    if joined.height != candidate_selected.height:
        raise RuntimeError("candidate selected rows do not match the control universe")
    control_rows = joined.select(
        *keys,
        "label_up",
        "binance_sign_up",
        pl.col("control_probability_up").alias("probability_up"),
        pl.col("control_predicted_up").alias("predicted_up"),
        pl.col("control_confidence").alias("confidence"),
        pl.col("control_correct").alias("correct"),
    ).with_columns((pl.col("binance_sign_up") == pl.col("label_up")).alias("baseline_correct"))
    candidate_hard = hard_false_up_metrics(
        candidate_selected,
        eligible_markets=candidate_selected.height,
        confidence_floor=hard_confidence_floor,
    )
    control_hard = hard_false_up_metrics(
        control_rows,
        eligible_markets=candidate_selected.height,
        confidence_floor=hard_confidence_floor,
    )
    candidate_correct = int(candidate_selected["correct"].sum() or 0)
    control_correct = int(control_rows["correct"].sum() or 0)
    markets = candidate_selected.height
    return {
        "markets": markets,
        "candidate_accuracy": _ratio(candidate_correct, markets),
        "control_accuracy": _ratio(control_correct, markets),
        "accuracy_delta": _ratio(candidate_correct - control_correct, markets),
        "candidate_only_correct": joined.filter(
            pl.col("correct") & ~pl.col("control_correct")
        ).height,
        "control_only_correct": joined.filter(
            ~pl.col("correct") & pl.col("control_correct")
        ).height,
        "direction_disagreements": joined.filter(
            pl.col("predicted_up") != pl.col("control_predicted_up")
        ).height,
        "candidate_hard": candidate_hard,
        "control_hard": control_hard,
    }


def _raw_control_comparison(
    candidate_scored: pl.DataFrame,
    control_scored: pl.DataFrame,
) -> dict[str, Any]:
    keys = ["fold_index", "market_id", "observed_at"]
    joined = candidate_scored.join(
        control_scored.select(
            *keys,
            pl.col("correct").alias("control_correct"),
            pl.col("predicted_up").alias("control_predicted_up"),
        ),
        on=keys,
        how="inner",
        validate="1:1",
    )
    if joined.height != candidate_scored.height:
        raise RuntimeError("candidate raw rows do not match the control universe")
    markets = joined.height
    candidate_correct = int(joined["correct"].sum() or 0)
    control_correct = int(joined["control_correct"].sum() or 0)
    return {
        "markets": markets,
        "candidate_accuracy": _ratio(candidate_correct, markets),
        "control_accuracy": _ratio(control_correct, markets),
        "accuracy_delta": _ratio(candidate_correct - control_correct, markets),
        "direction_disagreements": joined.filter(
            pl.col("predicted_up") != pl.col("control_predicted_up")
        ).height,
    }


def _select_candidate(
    config: FixedTimeReversalConfig,
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    qualified = [payload for payload in candidates.values() if payload["advancement_qualified"]]
    ranked = sorted(qualified, key=_candidate_rank, reverse=True)
    selected = ranked[0]["candidate"] if ranked else None
    return {
        "control_candidate": FIXED_TIME_REVERSAL_CONTROL_CANDIDATE,
        "qualified_candidates": [payload["candidate"] for payload in ranked],
        "selected_candidate": selected,
        "selection_rank": [
            {
                "candidate": payload["candidate"],
                "rank": list(_candidate_rank(payload)),
            }
            for payload in ranked
        ],
        "rank_contract": [
            "minimum fold raw sign-baseline accuracy uplift",
            "aggregate raw sign-baseline accuracy uplift",
            "minimum primary fold accuracy",
            "aggregate primary accuracy",
            "aggregate selected override precision",
            "negative aggregate hard false-UP rate",
            "negative feature count",
        ],
        "every_fold_raw_uplift_reported_and_ranked_not_gated": True,
        "development_evidence_only": True,
        "independently_qualified": False,
        "exploratory_paper_export_authorized": selected is not None,
        "trading_process_creation_authorized": False,
        "live_capital_authorized": False,
        "fresh_forward_evidence_required": True,
        "fresh_forward_evidence_start": (config.split.policy_selection_end.isoformat()),
    }


def _candidate_rank(candidate: dict[str, Any]) -> tuple[float, ...]:
    fold = candidate["fold_summary"]
    raw = candidate["raw"]["sign_override_metrics"]
    primary = candidate["primary"]
    override_precision = primary["sign_override_metrics"]["override_precision"]
    hard = primary["hard_false_up_metrics"]
    return (
        fold["minimum_raw_sign_baseline_accuracy_uplift"],
        raw["sign_baseline_accuracy_uplift"],
        fold["minimum_primary_accuracy"],
        primary["metrics"]["accuracy"],
        float(override_precision) if override_precision is not None else 0.0,
        -hard["hard_false_up_rate_selected"],
        -float(candidate["feature_count"]),
    )


def _assert_identical_scored_universes(
    scored_by_candidate: dict[str, pl.DataFrame],
) -> None:
    control = scored_by_candidate[FIXED_TIME_REVERSAL_CONTROL_CANDIDATE]
    if control.is_empty() or set(control["seconds_elapsed"].unique().to_list()) != {120}:
        raise RuntimeError("fixed-time reversal control is not an exact-120 universe")
    keys = [
        "fold_index",
        "market_id",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "binance_sign_up",
    ]
    expected = control.select(keys).rows()
    if len(expected) != len(set(expected)):
        raise RuntimeError("fixed-time reversal control universe contains duplicates")
    for candidate, scored in scored_by_candidate.items():
        observed = scored.select(keys).rows()
        if observed != expected:
            raise RuntimeError(f"{candidate} scored universe differs from the reversal control")


def _candidate_config(
    config: FixedTimeReversalConfig,
    candidate_name: str,
) -> FixedTimeReversalCandidateConfig:
    for candidate in config.candidates:
        if candidate.name == candidate_name:
            return candidate
    raise ValueError(f"unknown fixed-time reversal candidate: {candidate_name}")


def _validate_eligible_cohort(
    frame: pl.DataFrame,
    config: FixedTimeReversalConfig,
) -> None:
    markets = frame["market_id"].n_unique()
    expected_markets = config.model.expected_eligible_markets
    expected_seconds = set(config.model.estimator_training_seconds)
    if markets != expected_markets:
        raise RuntimeError(
            f"fixed-time reversal expected {expected_markets} markets, observed {markets}"
        )
    if frame.height != config.model.expected_eligible_estimator_rows:
        raise RuntimeError(
            "fixed-time reversal contemporaneously eligible estimator row count changed"
        )
    if set(frame["seconds_elapsed"].unique().to_list()) != expected_seconds:
        raise RuntimeError("fixed-time reversal eligible cohort cadence changed")
    if frame.select("market_id", "seconds_elapsed").unique().height != frame.height:
        raise RuntimeError("fixed-time reversal eligible cohort contains duplicate market seconds")
    inconsistent_market_facts = (
        frame.group_by("market_id")
        .agg(
            pl.col("window_start").n_unique().alias("unique_windows"),
            pl.col("label_up").n_unique().alias("unique_labels"),
        )
        .filter((pl.col("unique_windows") != 1) | (pl.col("unique_labels") != 1))
    )
    if inconsistent_market_facts.height:
        raise RuntimeError("fixed-time reversal eligible cohort has inconsistent market facts")
    ineligible_path = frame.filter(
        pl.col("btc_path_from_window_open_bps").abs() <= config.model.path_zero_epsilon_bps
    )
    if ineligible_path.height:
        raise RuntimeError("fixed-time reversal cohort contains a contemporaneously ambiguous path")
    exact_decisions = frame.filter(pl.col("seconds_elapsed") == config.model.decision_second)
    expected_exact_decisions = config.model.expected_exact_120_eligible_markets
    if exact_decisions.height != expected_exact_decisions:
        raise RuntimeError(
            "fixed-time reversal exact-120 eligible market count changed: "
            f"expected {expected_exact_decisions}, observed {exact_decisions.height}"
        )
    if exact_decisions["market_id"].n_unique() != exact_decisions.height:
        raise RuntimeError("fixed-time reversal exact-120 cohort must have one row per market")
    path_sign_mismatch = frame.filter(
        (pl.col("btc_path_from_window_open_bps") > 0.0)
        != pl.col("binance_sign_up").cast(pl.Boolean)
    )
    if path_sign_mismatch.height:
        raise RuntimeError("Binance sign differs from the runtime path polarity")
    if frame["window_start"].min() < config.split.development_start:
        raise RuntimeError("fixed-time reversal cohort starts before March 21")
    latest = frame["window_start"].max()
    if latest >= config.split.policy_selection_end:
        raise RuntimeError("fixed-time reversal cohort contains July 29 or later")


def _validate_feature_metadata(
    metadata: dict[str, Any],
    config: FixedTimeReversalConfig,
) -> None:
    schemas = set(metadata.get("candidate_feature_schema_versions", {}).values())
    expected = {candidate.feature_schema_version for candidate in config.candidates}
    if not expected.issubset(schemas):
        raise RuntimeError("feature cache lost the fixed-time reversal schema")
    if metadata.get("scope") != "pre_holdout":
        raise RuntimeError("fixed-time reversal cache scope changed")
    if metadata.get("range_start") != config.split.development_start.isoformat():
        raise RuntimeError("fixed-time reversal cache start changed")
    if metadata.get("range_end") != config.split.policy_selection_end.isoformat():
        raise RuntimeError("fixed-time reversal cache includes post-July-28 data")
    if int(metadata.get("expected_candidate_rows_per_market", 0)) != 37:
        raise RuntimeError("fixed-time reversal source cache cadence changed")


def _execution_config(config: FixedTimeReversalConfig) -> ExecutionEvidenceConfig:
    manifest = json.loads((config.paths.execution_evidence / "manifest.json").read_text())
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


def _config_payload(config: FixedTimeReversalConfig) -> dict[str, Any]:
    return _json_value(asdict(config))


def _range_frame(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    selected = frame.filter((pl.col("window_start") >= start) & (pl.col("window_start") < end))
    if selected.is_empty():
        raise RuntimeError(f"fixed-time reversal range is empty: {start} to {end}")
    return selected


def _minimum_check(
    name: str,
    observed: float,
    required: float,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    passed = observed > required if strict else observed >= required - 1e-12
    return {
        "name": name,
        "observed": observed,
        "operator": ">" if strict else ">=",
        "required": required,
        "passed": bool(passed),
    }


def _maximum_check(name: str, observed: float, required: float) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "operator": "<=",
        "required": required,
        "passed": bool(observed <= required + 1e-12),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _optional_ratio(
    numerator: float,
    denominator: float,
) -> float | None:
    return float(numerator / denominator) if denominator else None


def _render_report(benchmark: dict[str, Any]) -> str:
    selected = benchmark["selection"]["selected_candidate"] or "none"
    data_contract = benchmark["data_contract"]
    rows: list[str] = []
    for name, candidate in benchmark["candidates"].items():
        primary = candidate["primary"]["metrics"]
        raw = candidate["raw"]["sign_override_metrics"]
        overrides = candidate["primary"]["sign_override_metrics"]
        hard = candidate["primary"]["hard_false_up_metrics"]
        economics = candidate["primary"]["ten_share_vwap10_execution"]
        precision = overrides["override_precision"]
        precision_text = "n/a" if precision is None else f"{precision:.2%}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(candidate['target_kind'])}</td>"
            f"<td>{'yes' if candidate['advancement_qualified'] else 'no'}</td>"
            f"<td>{primary['coverage']:.2%}</td>"
            f"<td>{primary['accuracy']:.2%}</td>"
            f"<td>{raw['sign_baseline_accuracy_uplift']:+.2%}</td>"
            f"<td>{overrides['overrides']}</td>"
            f"<td>{precision_text}</td>"
            f"<td>{hard['hard_false_up_markets']}</td>"
            f"<td>{_format_number(economics['realized_net_expectancy_per_trade'])}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC fixed-120 reversal-decision benchmark</title>
<style>
body {{ font: 15px system-ui; margin: 2rem; color: #17202a; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d5d8dc; padding: .5rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
</style>
</head>
<body>
<h1>BTC fixed-120 reversal-decision benchmark</h1>
<p><strong>Selected development candidate: {html.escape(selected)}</strong></p>
<p>Three candidates are evaluated across seven chronological folds on {data_contract["exact_120_eligible_markets"]:,} causal exact-120 decisions. Estimator fitting uses {data_contract["contemporaneously_eligible_estimator_rows"]:,} contemporaneously eligible rows across {data_contract["contemporaneously_eligible_estimator_markets"]:,} markets; later path state never determines earlier eligibility. July 29 onward is excluded.</p>
<p>The 10% operating point qualifies candidates; 8% is diagnostic. The direct-outcome control never advances.</p>
<p>Order-book data is used only for common strict VWAP10 economics and never as a model input.</p>
<table>
<thead><tr><th>Candidate</th><th>Target</th><th>Qualified</th>
<th>Coverage</th><th>Accuracy</th><th>Raw uplift</th>
<th>Overrides</th><th>Override precision</th><th>Hard false-UP</th>
<th>10-share expectancy</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</body>
</html>
"""


def _format_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def freeze_and_export_fixed_time_reversal_paper_candidate(
    *,
    config: FixedTimeReversalConfig,
    benchmark_run: Path,
    model_key: str,
    authorization: str,
) -> tuple[Path, Path, dict[str, Any]]:
    if authorization != FIXED_TIME_REVERSAL_PAPER_AUTHORIZATION:
        raise RuntimeError("fixed-time reversal export requires explicit paper-only authorization")
    benchmark_path = benchmark_run.resolve() / "benchmark.json"
    if not benchmark_path.is_file():
        raise RuntimeError("fixed-time reversal benchmark evidence is missing")
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark.get("schema_version") != FIXED_TIME_REVERSAL_BENCHMARK_SCHEMA_VERSION:
        raise RuntimeError("fixed-time reversal benchmark schema changed")
    if benchmark.get("configuration") != _config_payload(config):
        raise RuntimeError("fixed-time reversal benchmark configuration changed")
    selected_name = benchmark.get("selection", {}).get("selected_candidate")
    if not isinstance(selected_name, str):
        raise TypeError("fixed-time reversal benchmark selected no candidate")
    selected_evidence = benchmark.get("candidates", {}).get(selected_name)
    if not isinstance(selected_evidence, dict) or not selected_evidence.get(
        "advancement_qualified"
    ):
        raise RuntimeError("selected reversal candidate did not pass all checks")

    selected_config = _candidate_config(config, selected_name)
    if selected_config.target_kind != PATH_PERSISTENCE_TARGET:
        raise RuntimeError("only a path-persistence reversal candidate may be exported")
    core_config = load_core_config(config.benchmark.core_config)
    feature_metadata = validate_core_feature_cache(core_config, "pre_holdout")
    _validate_feature_metadata(feature_metadata, config)
    frame = eligible_complete_market_frame(
        load_core_feature_frame(core_config, "pre_holdout"),
        decision_seconds=config.model.estimator_training_seconds,
        expected_source_markets=config.model.expected_source_markets,
        expected_source_estimator_rows=(config.model.expected_source_estimator_rows),
        expected_eligible_markets=config.model.expected_eligible_markets,
        expected_eligible_estimator_rows=(config.model.expected_eligible_estimator_rows),
        expected_exact_120_eligible_markets=(config.model.expected_exact_120_eligible_markets),
    )
    _validate_eligible_cohort(frame, config)
    final = _fit_final_policy(
        frame=frame,
        candidate=selected_config,
        config=config,
        core_config=core_config,
    )
    control = _fit_final_policy(
        frame=frame,
        candidate=_candidate_config(
            config,
            FIXED_TIME_REVERSAL_CONTROL_CANDIDATE,
        ),
        config=config,
        core_config=core_config,
    )

    execution = load_execution_evidence(_execution_config(config))
    selected_with_execution = attach_execution_evidence(final["selected"], execution)
    primary = _aggregate_operating_point(
        selected_with_execution,
        eligible_markets=final["policy"].height,
        contract=config.primary,
        hard_confidence_floor=config.model.hard_confidence_floor,
        include_execution=True,
    )
    raw = _aggregate_operating_point(
        final["scored"],
        eligible_markets=final["policy"].height,
        contract=None,
        hard_confidence_floor=config.model.hard_confidence_floor,
        include_execution=False,
    )
    same_rows = _same_row_final_control_comparison(
        final["selected"],
        control["scored"],
        hard_confidence_floor=config.model.hard_confidence_floor,
    )
    override = primary["sign_override_metrics"]
    override_precision = override["override_precision"]
    checks = list(primary["checks"])
    checks.extend(
        [
            _minimum_check(
                "final raw accuracy uplift over Binance sign",
                raw["sign_override_metrics"]["sign_baseline_accuracy_uplift"],
                0.0,
                strict=True,
            ),
            _minimum_check(
                "final selected overrides exist",
                float(override["overrides"]),
                0.0,
                strict=True,
            ),
            _minimum_check(
                "final selected override precision exceeds chance",
                float(override_precision) if override_precision is not None else 0.0,
                0.5,
                strict=True,
            ),
            _maximum_check(
                "final same-row hard-error rate does not regress control",
                same_rows["candidate_hard"]["hard_error_rate_selected"],
                same_rows["control_hard"]["hard_error_rate_selected"],
            ),
            _maximum_check(
                "final same-row hard false-UP rate does not regress control",
                same_rows["candidate_hard"]["hard_false_up_rate_selected"],
                same_rows["control_hard"]["hard_false_up_rate_selected"],
            ),
        ]
    )
    checks.extend(_economics_checks(primary["ten_share_vwap10_execution"]))
    passed = all(check["passed"] for check in checks)
    assessment_path = benchmark_run.resolve() / "paper-freeze-assessment.json"
    assessment: dict[str, Any] = {
        "schema_version": FIXED_TIME_REVERSAL_ASSESSMENT_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_run": str(benchmark_run.resolve()),
        "benchmark_sha256": file_sha256(benchmark_path),
        "candidate": selected_name,
        "target_kind": selected_config.target_kind,
        "feature_schema_version": selected_config.feature_schema_version,
        "decision_second": config.model.decision_second,
        "development_evidence_only": True,
        "independently_qualified": False,
        "status": (
            "qualified_for_exploratory_paper_freeze" if passed else "blocked_by_final_policy"
        ),
        "threshold_selection": final["threshold_selection"],
        "raw": raw,
        "primary": primary,
        "same_selected_rows_control_comparison": same_rows,
        "policy_checks": checks,
        "tuning": final["tuning"],
        "calibrator": asdict(final["calibrator"]),
        "runtime_model_created": False,
        "trading_process_created": False,
        "live_capital_authorized": False,
        "fresh_forward_evidence_required": True,
        "fresh_forward_evidence_start": config.split.policy_selection_end.isoformat(),
    }
    write_json_atomic(assessment_path, assessment)
    if not passed:
        raise RuntimeError(
            "final fixed-time reversal policy did not pass every paper check; "
            f"assessment: {assessment_path}"
        )

    band = FrozenCalibrationBand(
        name="fixed-120",
        start_second=config.model.decision_second,
        end_second_exclusive=config.model.decision_second + 1,
        calibrator=final["calibrator"],
        confidence_threshold=final["threshold"],
    )
    bundle = FrozenTimeBandedTrainingBundle(
        model=final["model"],
        target_kind=PATH_PERSISTENCE_TARGET,
        bands=(band,),
    )
    created_at = datetime.now(UTC)
    freeze_id = f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{selected_name}-paper"
    freeze_dir = config.paths.freezes / freeze_id
    freeze_dir.mkdir(parents=True, exist_ok=False)
    model_path = freeze_dir / TRAINING_MODEL_FILENAME
    joblib.dump(bundle, model_path, compress=3)
    calibration_bands = frozen_calibration_bands_payload(bundle.bands)
    summary = {
        "schema_version": "btc-core-training-model-summary-v1",
        "fixed_time_reversal_freeze_schema_version": (FIXED_TIME_REVERSAL_FREEZE_SCHEMA_VERSION),
        "training_only": True,
        "deployment_scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
        "candidate": bundle.model.candidate_name,
        "family": bundle.model.family,
        "feature_names": list(bundle.model.feature_names),
        "hyperparameters": bundle.model.hyperparameters,
        "imputation_medians": bundle.model.imputation_medians.tolist(),
        "calibration_kind": "time_banded_platt",
        "calibration_bands": calibration_bands,
        "target_kind": bundle.target_kind,
    }
    summary_path = freeze_dir / "model-summary.json"
    write_json_atomic(summary_path, summary)
    golden_path = freeze_dir / GOLDEN_FEATURES_FILENAME
    write_time_banded_golden_feature_sample(
        final["policy"],
        final["probability_up"],
        bundle,
        selected_config.feature_schema_version,
        golden_path,
    )

    provenance = runtime_provenance(config.package_root)
    frozen_spec = model_candidate_spec(bundle.model)
    feature_path = core_config.paths.development_feature_data
    feature_metadata_path = feature_path.with_suffix(".metadata.json")
    manifest: dict[str, Any] = {
        "schema_version": CORE_FREEZE_SCHEMA_VERSION,
        "fixed_time_reversal_freeze_schema_version": (FIXED_TIME_REVERSAL_FREEZE_SCHEMA_VERSION),
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
        "target_kind": bundle.target_kind,
        "row_weight_policy": frozen_spec.row_weight_policy,
        "row_weight_schedule": row_weight_schedule_payload(frozen_spec),
        "recency_half_life_days": frozen_spec.recency_half_life_days,
        "feature_schema_version": selected_config.feature_schema_version,
        "feature_names": list(bundle.model.feature_names),
        "hyperparameters": bundle.model.hyperparameters,
        "calibration_kind": "time_banded_platt",
        "calibration_bands": calibration_bands,
        "prediction_policy": {
            "type": "first_confidence_crossing",
            "minimum_seconds_after_open": config.model.decision_second,
            "maximum_seconds_after_open": config.model.decision_second,
            "cadence_seconds": 5,
        },
        "configuration_sha256": file_sha256(config.source_path),
        "core_configuration_sha256": file_sha256(config.benchmark.core_config),
        "development_feature_sha256": file_sha256(feature_path),
        "development_feature_metadata_sha256": file_sha256(feature_metadata_path),
        "golden_feature_file": golden_path.name,
        "golden_feature_sha256": file_sha256(golden_path),
        "golden_feature_metadata_sha256": file_sha256(golden_path.with_suffix(".metadata.json")),
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
            "metrics": primary["metrics"],
            "sign_override_metrics": primary["sign_override_metrics"],
            "checks": checks,
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
    (freeze_dir / "freeze-manifest.sha256").write_text(file_sha256(manifest_path) + "\n")
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
    candidate: FixedTimeReversalCandidateConfig,
    config: FixedTimeReversalConfig,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    development = _range_frame(
        frame,
        config.split.development_start,
        config.split.development_end,
    )
    calibration = fixed_time_rows(
        _range_frame(
            frame,
            config.split.probability_calibration_start,
            config.split.probability_calibration_end,
        ),
        decision_second=config.model.decision_second,
        cohort_name=f"{candidate.name} final calibration",
    )
    policy = fixed_time_rows(
        _range_frame(
            frame,
            config.split.policy_selection_start,
            config.split.policy_selection_end,
        ),
        decision_second=config.model.decision_second,
        cohort_name=f"{candidate.name} final policy",
    )
    spec = reversal_candidate_spec(candidate, config.model)
    parameter_grid = reversal_parameter_grid(candidate, config.model, core_config)
    model, tuning = tune_and_fit_reversal_model(
        development,
        spec,
        parameter_grid,
        core_config,
        candidate.target_kind,
        config.model.decision_second,
        config.primary.target_coverage,
        config.diagnostic.target_coverage,
    )
    calibrator = fit_probability_calibrator(
        model,
        training_target_frame(calibration, candidate.target_kind),
        core_config,
        spec,
    )
    if not calibrator.converged or calibrator.slope <= 0.0:
        raise RuntimeError(f"{candidate.name} final calibrator failed")
    scored = _score_candidate_rows(
        policy,
        candidate=candidate,
        model=model,
        calibrator=calibrator,
        fold_index=-1,
    )
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
        "probability_up": scored["probability_up"].to_numpy(),
        "scored": scored,
        "threshold": threshold,
        "threshold_selection": threshold_selection,
        "selected": selected,
    }


def _same_row_final_control_comparison(
    candidate_selected: pl.DataFrame,
    control_scored: pl.DataFrame,
    *,
    hard_confidence_floor: float,
) -> dict[str, Any]:
    candidate = candidate_selected.with_columns(pl.lit(-1).alias("fold_index"))
    control = control_scored.with_columns(pl.lit(-1).alias("fold_index"))
    return _same_row_control_comparison(
        candidate,
        control,
        hard_confidence_floor=hard_confidence_floor,
    )


def _range_payload(start: datetime, end: datetime) -> dict[str, str]:
    return {"start": start.isoformat(), "end": end.isoformat()}
