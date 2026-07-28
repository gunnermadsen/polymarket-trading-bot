from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from .core_config import (
    CoreTrainingConfig,
    HistogramCandidate,
    RowWeightScheduleConfig,
    config_to_dict,
    evaluation_holdout_range,
    load_core_config,
)
from .core_evaluation import (
    baseline_metrics,
    block_bootstrap_uplift,
    choose_threshold,
    classification_metrics,
    daily_accuracy,
    first_crossing_timing,
    first_prediction_rows,
    fixed_time_prediction_rows,
    paired_uplift,
    reliability_rows,
    scored_prediction_rows,
    sigmoid,
    threshold_table,
    time_accuracy,
)
from .core_extract import file_sha256, write_json_atomic
from .core_features import (
    CORE_BASELINE_FEATURES,
    CORE_ENRICHED_FEATURES,
    CORE_FEATURE_SCHEMA_VERSION,
    load_core_feature_frame,
    validate_core_feature_cache,
)
from .provenance import runtime_provenance

CORE_TRAINING_SCHEMA_VERSION = "btc-core-training-v1"
CORE_FREEZE_SCHEMA_VERSION = "btc-core-freeze-v1"
TRAINING_MODEL_FILENAME = "training-model.joblib"
MARKET_EQUAL_ROW_WEIGHT_POLICY = "market_equal"
EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY = "early_entry_market_equal"
MARKET_EQUAL_ROW_WEIGHT_SCHEDULE = RowWeightScheduleConfig(
    start_second=None,
    end_second_inclusive=None,
    multiplier=1.0,
)
EARLY_ENTRY_ROW_WEIGHT_SCHEDULE = RowWeightScheduleConfig(
    start_second=60,
    end_second_inclusive=120,
    multiplier=3.0,
)
MODERATE_EARLY_ENTRY_ROW_WEIGHT_SCHEDULE = RowWeightScheduleConfig(
    start_second=60,
    end_second_inclusive=120,
    multiplier=1.5,
)
FOCUSED_EARLY_ENTRY_ROW_WEIGHT_SCHEDULE = RowWeightScheduleConfig(
    start_second=90,
    end_second_inclusive=120,
    multiplier=2.0,
)
EARLY_ENTRY_TRAINING_START_SECONDS = EARLY_ENTRY_ROW_WEIGHT_SCHEDULE.start_second
EARLY_ENTRY_TRAINING_END_SECONDS_INCLUSIVE = (
    EARLY_ENTRY_ROW_WEIGHT_SCHEDULE.end_second_inclusive
)
EARLY_ENTRY_TRAINING_WEIGHT_MULTIPLIER = EARLY_ENTRY_ROW_WEIGHT_SCHEDULE.multiplier


def legacy_row_weight_schedule(policy: str) -> RowWeightScheduleConfig:
    if policy == MARKET_EQUAL_ROW_WEIGHT_POLICY:
        return MARKET_EQUAL_ROW_WEIGHT_SCHEDULE
    if policy == EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY:
        return EARLY_ENTRY_ROW_WEIGHT_SCHEDULE
    raise ValueError(f"unsupported candidate row-weight policy: {policy}")


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    feature_names: tuple[str, ...]
    row_weight_policy: str = MARKET_EQUAL_ROW_WEIGHT_POLICY
    row_weight_schedule: RowWeightScheduleConfig | None = None

    def __post_init__(self) -> None:
        if self.row_weight_policy not in {
            MARKET_EQUAL_ROW_WEIGHT_POLICY,
            EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY,
        }:
            raise ValueError(
                f"unsupported candidate row-weight policy: {self.row_weight_policy}"
            )
        schedule = self.row_weight_schedule
        if schedule is None:
            schedule = legacy_row_weight_schedule(self.row_weight_policy)
            object.__setattr__(self, "row_weight_schedule", schedule)
        if (
            self.row_weight_policy == MARKET_EQUAL_ROW_WEIGHT_POLICY
            and schedule != MARKET_EQUAL_ROW_WEIGHT_SCHEDULE
        ):
            raise ValueError("market-equal candidates cannot configure a timed multiplier")
        if (
            self.row_weight_policy == EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY
            and schedule.start_second is None
        ):
            raise ValueError("early-entry candidates require a timed multiplier")


CANDIDATES = (
    CandidateSpec(
        "logistic_baseline",
        "logistic",
        tuple(CORE_BASELINE_FEATURES),
    ),
    CandidateSpec(
        "logistic_enriched",
        "logistic",
        tuple(CORE_ENRICHED_FEATURES),
    ),
    CandidateSpec(
        "histogram_enriched",
        "histogram",
        tuple(CORE_ENRICHED_FEATURES),
    ),
    CandidateSpec(
        "histogram_early_weighted",
        "histogram",
        tuple(CORE_ENRICHED_FEATURES),
        EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY,
        EARLY_ENTRY_ROW_WEIGHT_SCHEDULE,
    ),
)

ADDITIONAL_CANDIDATES = (
    CandidateSpec(
        "histogram_early_weighted_moderate",
        "histogram",
        tuple(CORE_ENRICHED_FEATURES),
        EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY,
        MODERATE_EARLY_ENTRY_ROW_WEIGHT_SCHEDULE,
    ),
    CandidateSpec(
        "histogram_early_90_120",
        "histogram",
        tuple(CORE_ENRICHED_FEATURES),
        EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY,
        FOCUSED_EARLY_ENTRY_ROW_WEIGHT_SCHEDULE,
    ),
)


@dataclass
class FittedCoreModel:
    candidate_name: str
    family: str
    feature_names: tuple[str, ...]
    hyperparameters: dict[str, Any]
    imputation_medians: np.ndarray
    standardization_means: np.ndarray | None
    standardization_scales: np.ndarray | None
    estimator: Any
    row_weight_policy: str | None = None
    row_weight_schedule: RowWeightScheduleConfig | None = None

    def raw_probability(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = feature_matrix(frame, self.feature_names)
        transformed = transform_for_model(
            matrix,
            self.imputation_medians,
            self.standardization_means,
            self.standardization_scales,
        )
        return self.estimator.predict_proba(transformed)[:, 1]

    def raw_logit(self, frame: pl.DataFrame) -> np.ndarray:
        probability = np.clip(self.raw_probability(frame), 1e-9, 1 - 1e-9)
        return np.log(probability / (1 - probability))


@dataclass(frozen=True)
class ProbabilityCalibrator:
    slope: float
    intercept: float
    converged: bool
    iterations: int

    def probability(self, raw_logit: np.ndarray) -> np.ndarray:
        return sigmoid(raw_logit * self.slope + self.intercept)


@dataclass
class FrozenTrainingBundle:
    model: FittedCoreModel
    calibrator: ProbabilityCalibrator
    confidence_threshold: float

    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        return self.calibrator.probability(self.model.raw_logit(frame))


def develop_core_models(
    config: CoreTrainingConfig,
    *,
    freeze_if_ready: bool = True,
    fit_final_candidate: bool = True,
) -> tuple[Path, Path | None, dict[str, Any]]:
    if freeze_if_ready and not fit_final_candidate:
        raise ValueError(
            "freeze_if_ready requires final candidate fitting to remain enabled"
        )
    feature_metadata = validate_core_feature_cache(config, "pre_holdout")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    update_progress(run_dir, "walk_forward", 0.05)
    candidate_names = [
        candidate.name for candidate in configured_candidate_specs(config)
    ]
    total_fits = len(candidate_names) * len(config.split.validation_windows)
    fold_results: list[dict[str, Any]] = []
    max_workers = min(config.compute.max_parallel_fits, len(candidate_names))
    configure_native_thread_limits(config)
    print(
        f"core develop: {total_fits} candidate/fold fits with {max_workers} workers",
        flush=True,
    )
    completed = 0
    if max_workers == 1:
        for candidate_name in candidate_names:
            candidate_fold_results = evaluate_candidate_task(
                config.source_path,
                candidate_name,
            )
            for result in candidate_fold_results:
                completed += 1
                fold_results.append(result)
                report_fold_completion(result, completed, total_fits, run_dir)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_candidate_task,
                    config.source_path,
                    candidate_name,
                ): candidate_name
                for candidate_name in candidate_names
            }
            for future in as_completed(futures):
                candidate_fold_results = future.result()
                for result in candidate_fold_results:
                    completed += 1
                    fold_results.append(result)
                    report_fold_completion(result, completed, total_fits, run_dir)

    walk_forward_rows = combined_fold_rows(fold_results)
    fixed_time_rows = combined_fixed_time_rows(fold_results)
    walk_forward_probability_rows = combined_scored_probability_rows(fold_results)
    candidate_results = aggregate_candidate_results(config, fold_results)
    selected_name = max(
        candidate_results,
        key=lambda name: candidate_rank(candidate_results[name]),
    )
    selected_development = candidate_results[selected_name]
    print(
        "core develop: selected "
        f"{selected_name} accuracy={selected_development['out_of_fold']['accuracy']:.4f} "
        f"uplift={selected_development['paired']['accuracy_uplift']:+.4f} "
        "nonnegative_uplift_folds="
        f"{selected_development['nonnegative_uplift_folds']}",
        flush=True,
    )

    if not fit_final_candidate:
        metrics = {
            "schema_version": CORE_TRAINING_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "walk_forward_evidence_ready",
            "data": feature_metadata,
            "configuration": config_to_dict(config),
            "runtime_provenance": runtime_provenance(config.package_root),
            "selected_candidate": selected_name,
            "candidates": candidate_results,
            "final_tuning": None,
            "probability_calibration": None,
            "policy": None,
            "ready_for_holdout": False,
            "freeze_if_ready": False,
            "fit_final_candidate": False,
            "blocking_reasons": [
                (
                    "final candidate fitting and calibration are disabled for "
                    "the execution benchmark"
                )
            ],
        }
        walk_forward_rows.write_parquet(
            run_dir / "walk-forward-predictions.parquet",
            compression="zstd",
        )
        fixed_time_rows.write_parquet(
            run_dir / "walk-forward-fixed-time-predictions.parquet",
            compression="zstd",
        )
        walk_forward_probability_rows.write_parquet(
            run_dir / "walk-forward-scored-probabilities.parquet",
            compression="zstd",
        )
        write_json_atomic(run_dir / "development-metrics.json", metrics)
        update_progress(
            run_dir,
            "walk_forward_evidence_ready",
            1.0,
            {
                "selected_candidate": selected_name,
                "fit_final_candidate": False,
                "freeze_if_ready": False,
            },
        )
        return run_dir, None, metrics

    frame = load_core_feature_frame(config, "pre_holdout")
    development = range_frame(
        frame,
        config.split.development_start,
        config.split.development_end,
    )
    probability_calibration = range_frame(
        frame,
        config.split.probability_calibration_start,
        config.split.probability_calibration_end,
    )
    policy_selection = range_frame(
        frame,
        config.split.policy_selection_start,
        config.split.policy_selection_end,
    )
    selected_spec = candidate_spec(selected_name, config)
    final_model, tuning = tune_and_fit_model(development, selected_spec, config)
    calibrator = fit_probability_calibrator(
        final_model,
        probability_calibration,
        config,
        selected_spec,
    )
    policy_probability = calibrator.probability(
        final_model.raw_logit(policy_selection)
    )
    thresholds = threshold_table(policy_selection, policy_probability, config.model)
    policy_minimum_markets = max(
        100,
        math.ceil(
            policy_selection["market_id"].n_unique()
            * config.gates.minimum_coverage
        ),
    )
    confidence_threshold, threshold_qualified = choose_threshold(
        thresholds,
        config.gates,
        minimum_markets=policy_minimum_markets,
    )
    policy_rows = first_prediction_rows(
        policy_selection,
        policy_probability,
        confidence_threshold,
    ).with_columns(
        pl.lit(selected_name).alias("candidate"),
    )
    policy_eligible = policy_selection["market_id"].n_unique()
    policy_metrics = classification_metrics(
        policy_rows,
        eligible_markets=policy_eligible,
    )
    policy_baseline = baseline_metrics(
        policy_rows,
        eligible_markets=policy_eligible,
    )
    policy_paired = paired_uplift(policy_rows)
    policy_passed = bool(
        threshold_qualified
        and calibrator.converged
        and policy_metrics["expected_calibration_error"] <= config.gates.maximum_ece
        and policy_paired["accuracy_uplift"]
        >= config.gates.minimum_same_time_path_uplift
    )
    ready_for_holdout = bool(
        selected_development["passed_development"]
        and policy_passed
    )
    runtime = runtime_provenance(config.package_root)
    metrics: dict[str, Any] = {
        "schema_version": CORE_TRAINING_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": (
            "candidate_ready_for_freeze"
            if ready_for_holdout and freeze_if_ready
            else (
                "candidate_ready_for_execution_benchmark"
                if ready_for_holdout
                else "blocked_pre_holdout"
            )
        ),
        "data": feature_metadata,
        "configuration": config_to_dict(config),
        "runtime_provenance": runtime,
        "selected_candidate": selected_name,
        "candidates": candidate_results,
        "final_tuning": tuning,
        "probability_calibration": asdict(calibrator),
        "policy": {
            "confidence_threshold": confidence_threshold,
            "threshold_qualified": threshold_qualified,
            "metrics": policy_metrics,
            "baseline": policy_baseline,
            "paired": policy_paired,
            "timing": first_crossing_timing(
                policy_rows,
                eligible_markets=policy_eligible,
            ),
            "threshold_history": thresholds,
            "passed": policy_passed,
        },
        "ready_for_holdout": ready_for_holdout,
        "freeze_if_ready": freeze_if_ready,
        "fit_final_candidate": fit_final_candidate,
        "blocking_reasons": pre_holdout_blocking_reasons(
            selected_development,
            policy_passed,
            calibrator,
        ),
    }
    walk_forward_rows.write_parquet(
        run_dir / "walk-forward-predictions.parquet",
        compression="zstd",
    )
    fixed_time_rows.write_parquet(
        run_dir / "walk-forward-fixed-time-predictions.parquet",
        compression="zstd",
    )
    walk_forward_probability_rows.write_parquet(
        run_dir / "walk-forward-scored-probabilities.parquet",
        compression="zstd",
    )
    policy_rows.write_parquet(
        run_dir / "policy-predictions.parquet",
        compression="zstd",
    )
    write_json_atomic(run_dir / "development-metrics.json", metrics)
    freeze_dir: Path | None = None
    if ready_for_holdout and freeze_if_ready:
        bundle = FrozenTrainingBundle(
            model=final_model,
            calibrator=calibrator,
            confidence_threshold=confidence_threshold,
        )
        freeze_dir = freeze_candidate(
            config,
            run_dir,
            metrics,
            bundle,
        )
        metrics["freeze_dir"] = str(freeze_dir)
        write_json_atomic(run_dir / "development-metrics.json", metrics)
        update_progress(
            run_dir,
            "candidate_frozen",
            0.75,
            {"freeze_dir": str(freeze_dir)},
        )
    elif ready_for_holdout:
        update_progress(
            run_dir,
            "freeze_deferred_for_execution_benchmark",
            1.0,
            {
                "selected_candidate": selected_name,
                "freeze_if_ready": False,
            },
        )
    else:
        update_progress(
            run_dir,
            "blocked_pre_holdout",
            1.0,
            {"blocking_reasons": metrics["blocking_reasons"]},
        )
    return run_dir, freeze_dir, metrics


def evaluate_candidate_task(
    config_path: Path,
    candidate_name: str,
) -> list[dict[str, Any]]:
    config = load_core_config(config_path)
    configure_native_thread_limits(config)
    with threadpool_limits(limits=config.compute.threads_per_fit):
        frame = load_core_feature_frame(config, "pre_holdout")
        spec = candidate_spec(candidate_name, config)
        return [
            evaluate_fold(
                frame,
                spec,
                fold_index,
                config,
                include_scored_probabilities=True,
            )
            for fold_index in range(len(config.split.validation_windows))
        ]


def evaluate_fold_task(
    config_path: Path,
    candidate_name: str,
    fold_index: int,
) -> dict[str, Any]:
    config = load_core_config(config_path)
    configure_native_thread_limits(config)
    with threadpool_limits(limits=config.compute.threads_per_fit):
        frame = load_core_feature_frame(config, "pre_holdout")
        return evaluate_fold(
            frame,
            candidate_spec(candidate_name, config),
            fold_index,
            config,
            include_scored_probabilities=True,
        )


def evaluate_fold(
    frame: pl.DataFrame,
    spec: CandidateSpec,
    fold_index: int,
    config: CoreTrainingConfig,
    *,
    include_scored_probabilities: bool = False,
) -> dict[str, Any]:
    validation_start, validation_end = config.split.validation_windows[fold_index]
    history = range_frame(
        frame,
        config.split.development_start,
        validation_start,
    )
    validation = range_frame(frame, validation_start, validation_end)
    fit_frame, calibration_frame, policy_frame = chronological_subsplit(
        history,
        fit_fraction=0.70,
        calibration_fraction=0.15,
    )
    started = time.perf_counter()
    model, tuning = tune_and_fit_model(fit_frame, spec, config)
    calibrator = fit_probability_calibrator(
        model,
        calibration_frame,
        config,
        spec,
    )
    policy_probability = calibrator.probability(model.raw_logit(policy_frame))
    thresholds = threshold_table(policy_frame, policy_probability, config.model)
    minimum_markets = max(
        50,
        math.ceil(
            policy_frame["market_id"].n_unique() * config.gates.minimum_coverage
        ),
    )
    threshold, threshold_qualified = choose_threshold(
        thresholds,
        config.gates,
        minimum_markets=minimum_markets,
    )
    probability = calibrator.probability(model.raw_logit(validation))
    selected = first_prediction_rows(
        validation,
        probability,
        threshold,
    ).with_columns(
        pl.lit(spec.name).alias("candidate"),
        pl.lit(fold_index).cast(pl.Int32).alias("fold_index"),
    )
    fixed_time = fixed_time_prediction_rows(
        validation,
        probability,
    ).with_columns(
        pl.lit(spec.name).alias("candidate"),
        pl.lit(fold_index).cast(pl.Int32).alias("fold_index"),
    )
    eligible = validation["market_id"].n_unique()
    metrics = classification_metrics(selected, eligible_markets=eligible)
    baseline = baseline_metrics(selected, eligible_markets=eligible)
    paired = paired_uplift(selected)
    result = {
        "candidate": spec.name,
        "family": spec.family,
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
        "tuning": tuning,
        "calibrator": asdict(calibrator),
        "metrics": metrics,
        "baseline": baseline,
        "paired": paired,
        "timing": first_crossing_timing(
            selected,
            eligible_markets=eligible,
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "prediction_rows": selected.to_dicts(),
        "fixed_time_prediction_rows": fixed_time.to_dicts(),
    }
    if include_scored_probabilities:
        result["scored_probability_rows"] = scored_fold_probability_rows(
            validation,
            probability,
            spec,
            fold_index,
        )
    return result


def aggregate_candidate_results(
    config: CoreTrainingConfig,
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for spec in configured_candidate_specs(config):
        folds = sorted(
            (
                result
                for result in fold_results
                if result["candidate"] == spec.name
            ),
            key=lambda result: result["fold_index"],
        )
        rows = pl.DataFrame(
            [
                row
                for result in folds
                for row in result["prediction_rows"]
            ]
        )
        summarized_folds = [
            {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "prediction_rows",
                    "fixed_time_prediction_rows",
                    "scored_probability_rows",
                }
            }
            for result in folds
        ]
        eligible = sum(result["eligible_markets"] for result in folds)
        metrics = classification_metrics(rows, eligible_markets=eligible)
        baseline = baseline_metrics(rows, eligible_markets=eligible)
        paired = paired_uplift(rows)
        bootstrap = block_bootstrap_uplift(
            rows,
            resamples=config.gates.bootstrap_resamples,
            random_seed=config.model.random_seed,
            block="hour",
        )
        nonnegative_folds = sum(
            result["paired"]["accuracy_uplift"]
            >= config.gates.minimum_same_time_path_uplift
            for result in folds
        )
        passed = development_gate_passed(
            config,
            metrics,
            paired,
            bootstrap,
            nonnegative_folds,
        )
        output[spec.name] = {
            "candidate": spec.name,
            "family": spec.family,
            "feature_count": len(spec.feature_names),
            "features": list(spec.feature_names),
            "row_weight_policy": spec.row_weight_policy,
            "row_weight_schedule": row_weight_schedule_payload(spec),
            "folds": summarized_folds,
            "out_of_fold": metrics,
            "baseline": baseline,
            "paired": paired,
            "bootstrap": bootstrap,
            "timing": first_crossing_timing(
                rows,
                eligible_markets=eligible,
            ),
            "nonnegative_uplift_folds": nonnegative_folds,
            "passed_development": passed,
        }
    return output


def development_gate_passed(
    config: CoreTrainingConfig,
    metrics: dict[str, Any],
    paired: dict[str, Any],
    bootstrap: dict[str, Any],
    nonnegative_folds: int,
) -> bool:
    return bool(
        nonnegative_folds >= config.gates.minimum_nonnegative_uplift_folds
        and metrics["accuracy"] >= config.gates.target_accuracy
        and metrics["wilson_lower_95"] >= config.gates.target_wilson_lower
        and metrics["balanced_accuracy"] >= config.gates.target_balanced_accuracy
        and metrics["up_recall"] >= config.gates.minimum_direction_recall
        and metrics["down_recall"] >= config.gates.minimum_direction_recall
        and metrics["coverage"] >= config.gates.minimum_coverage
        and metrics["expected_calibration_error"] <= config.gates.maximum_ece
        and paired["accuracy_uplift"]
        >= config.gates.minimum_same_time_path_uplift
        and bootstrap["lower_95"] >= 0.0
    )


def candidate_rank(result: dict[str, Any]) -> tuple[Any, ...]:
    timing = result["timing"]
    median_crossing = timing["median_first_crossing_seconds"]
    p90_crossing = timing["p90_first_crossing_seconds"]
    return (
        result["passed_development"],
        result["bootstrap"]["lower_95"],
        result["paired"]["accuracy_uplift"],
        result["out_of_fold"]["wilson_lower_95"],
        result["out_of_fold"]["balanced_accuracy"],
        result["out_of_fold"]["accuracy"],
        timing["early_entry_coverage"],
        -(median_crossing if median_crossing is not None else float("inf")),
        -(p90_crossing if p90_crossing is not None else float("inf")),
        result["out_of_fold"]["coverage"],
        -result["feature_count"],
    )


def tune_and_fit_model(
    frame: pl.DataFrame,
    spec: CandidateSpec,
    config: CoreTrainingConfig,
) -> tuple[FittedCoreModel, dict[str, Any]]:
    train, validation = chronological_inner_split(frame, validation_fraction=0.20)
    candidates: list[dict[str, Any]]
    if spec.family == "logistic":
        candidates = [{"c": value} for value in config.model.c_candidates]
    elif spec.family == "histogram":
        candidates = [
            histogram_parameters(candidate)
            for candidate in config.model.histogram_candidates
        ]
    else:
        raise ValueError(f"unsupported candidate family: {spec.family}")
    history: list[dict[str, Any]] = []
    best_parameters: dict[str, Any] | None = None
    best_loss = float("inf")
    for parameters in candidates:
        started = time.perf_counter()
        model = fit_model(train, spec, parameters, config)
        probability = model.raw_probability(validation)
        weights = candidate_training_weights(validation, spec)
        candidate_loss = float(
            log_loss(
                validation["label_up"].to_numpy(),
                probability,
                sample_weight=weights,
                labels=[0, 1],
            )
        )
        record = {
            "hyperparameters": parameters,
            "validation_log_loss": candidate_loss,
            "fit_seconds": time.perf_counter() - started,
            "converged": estimator_converged(model.estimator),
        }
        history.append(record)
        if record["converged"] and candidate_loss < best_loss:
            best_loss = candidate_loss
            best_parameters = parameters
    if best_parameters is None:
        raise RuntimeError(f"no {spec.name} hyperparameter candidate converged")
    final_model = fit_model(frame, spec, best_parameters, config)
    return final_model, {
        "selected_hyperparameters": best_parameters,
        "validation_log_loss": best_loss,
        "candidates": history,
        "optimizer_converged": estimator_converged(final_model.estimator),
    }


def fit_model(
    frame: pl.DataFrame,
    spec: CandidateSpec,
    parameters: dict[str, Any],
    config: CoreTrainingConfig,
) -> FittedCoreModel:
    matrix = feature_matrix(frame, spec.feature_names)
    labels = frame["label_up"].to_numpy()
    weights = candidate_training_weights(frame, spec)
    medians = finite_medians(matrix)
    filled = np.where(np.isfinite(matrix), matrix, medians)
    if spec.family == "logistic":
        means = np.average(filled, axis=0, weights=weights)
        variance = np.average((filled - means) ** 2, axis=0, weights=weights)
        scales = np.sqrt(np.maximum(variance, 0.0))
        scales = np.where(scales > 1e-12, scales, 1.0)
        transformed = (filled - means) / scales
        estimator = LogisticRegression(
            C=float(parameters["c"]),
            solver="lbfgs",
            max_iter=2_000,
            tol=1e-7,
            random_state=config.model.random_seed,
        )
    else:
        means = None
        scales = None
        transformed = filled
        estimator = HistGradientBoostingClassifier(
            learning_rate=float(parameters["learning_rate"]),
            max_iter=int(parameters["max_iter"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            l2_regularization=float(parameters["l2_regularization"]),
            early_stopping=False,
            random_state=config.model.random_seed,
        )
    with threadpool_limits(limits=config.compute.threads_per_fit):
        estimator.fit(transformed, labels, sample_weight=weights)
    return FittedCoreModel(
        candidate_name=spec.name,
        family=spec.family,
        feature_names=spec.feature_names,
        hyperparameters=parameters,
        imputation_medians=medians,
        standardization_means=means,
        standardization_scales=scales,
        estimator=estimator,
        row_weight_policy=spec.row_weight_policy,
        row_weight_schedule=resolved_row_weight_schedule(spec),
    )


def fit_probability_calibrator(
    model: FittedCoreModel,
    frame: pl.DataFrame,
    config: CoreTrainingConfig,
    spec: CandidateSpec | None = None,
) -> ProbabilityCalibrator:
    logits = model.raw_logit(frame).reshape(-1, 1)
    labels = frame["label_up"].to_numpy()
    weights = candidate_training_weights(
        frame,
        spec or model_candidate_spec(model),
    )
    calibrator = LogisticRegression(
        C=1_000_000,
        solver="lbfgs",
        max_iter=500,
        tol=1e-9,
        random_state=config.model.random_seed,
    )
    with threadpool_limits(limits=config.compute.threads_per_fit):
        calibrator.fit(logits, labels, sample_weight=weights)
    return ProbabilityCalibrator(
        slope=float(calibrator.coef_[0, 0]),
        intercept=float(calibrator.intercept_[0]),
        converged=bool(calibrator.n_iter_[0] < calibrator.max_iter),
        iterations=int(calibrator.n_iter_[0]),
    )


def freeze_candidate(
    config: CoreTrainingConfig,
    run_dir: Path,
    metrics: dict[str, Any],
    bundle: FrozenTrainingBundle,
) -> Path:
    holdout_start, holdout_end = evaluation_holdout_range(config)
    if holdout_start >= holdout_end:
        raise RuntimeError("cannot freeze a candidate without a positive holdout range")
    freeze_id = (
        f"{metrics['run_id']}-{bundle.model.candidate_name}"
    )
    freeze_dir = config.paths.artifacts / freeze_id
    freeze_dir.mkdir(parents=True, exist_ok=False)
    model_path = freeze_dir / TRAINING_MODEL_FILENAME
    joblib.dump(bundle, model_path, compress=3)
    model_summary = model_summary_payload(bundle)
    frozen_spec = model_candidate_spec(bundle.model)
    write_json_atomic(freeze_dir / "model-summary.json", model_summary)
    feature_metadata_path = config.paths.development_feature_data.with_suffix(
        ".metadata.json"
    )
    manifest: dict[str, Any] = {
        "schema_version": CORE_FREEZE_SCHEMA_VERSION,
        "freeze_id": freeze_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "candidate_frozen",
        "deployment_status": "blocked_pending_execution_economics",
        "development_run": str(run_dir),
        "model_file": TRAINING_MODEL_FILENAME,
        "model_sha256": file_sha256(model_path),
        "model_summary_sha256": file_sha256(freeze_dir / "model-summary.json"),
        "candidate": bundle.model.candidate_name,
        "family": bundle.model.family,
        "row_weight_policy": frozen_spec.row_weight_policy,
        "row_weight_schedule": row_weight_schedule_payload(frozen_spec),
        "feature_schema_version": CORE_FEATURE_SCHEMA_VERSION,
        "feature_names": list(bundle.model.feature_names),
        "hyperparameters": bundle.model.hyperparameters,
        "calibrator": asdict(bundle.calibrator),
        "confidence_threshold": bundle.confidence_threshold,
        "prediction_policy": {
            "type": "first_confidence_crossing",
            "minimum_seconds_after_open": config.data.min_seconds_after_open,
            "maximum_seconds_after_open": 300
            - config.data.min_seconds_before_close,
            "cadence_seconds": config.data.sample_interval_seconds,
        },
        "configuration_sha256": file_sha256(config.source_path),
        "development_feature_sha256": file_sha256(
            config.paths.development_feature_data
        ),
        "development_feature_metadata_sha256": file_sha256(feature_metadata_path),
        "source_tree_sha256": metrics["runtime_provenance"]["source_tree_sha256"],
        "git": metrics["runtime_provenance"]["git"],
        "random_seed": config.model.random_seed,
        "holdout_range": {
            "start": holdout_start.isoformat(),
            "end": holdout_end.isoformat(),
        },
        "gates": asdict(config.gates),
        "walk_forward_accuracy": metrics["candidates"][
            bundle.model.candidate_name
        ]["out_of_fold"]["accuracy"],
        "walk_forward_uplift": metrics["candidates"][
            bundle.model.candidate_name
        ]["paired"]["accuracy_uplift"],
    }
    manifest_path = freeze_dir / "freeze-manifest.json"
    write_json_atomic(manifest_path, manifest)
    (freeze_dir / "freeze-manifest.sha256").write_text(
        file_sha256(manifest_path) + "\n"
    )
    return freeze_dir


def evaluate_core_holdout(
    config: CoreTrainingConfig,
    freeze_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    holdout_start, holdout_end = evaluation_holdout_range(config)
    if holdout_start >= holdout_end:
        raise RuntimeError("holdout evaluation requires a positive holdout range")
    manifest_path = freeze_dir / "freeze-manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("freeze manifest is missing")
    expected_manifest_sha = (freeze_dir / "freeze-manifest.sha256").read_text().strip()
    if file_sha256(manifest_path) != expected_manifest_sha:
        raise RuntimeError("freeze manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != CORE_FREEZE_SCHEMA_VERSION:
        raise RuntimeError("unsupported freeze schema")
    model_path = freeze_dir / manifest["model_file"]
    if file_sha256(model_path) != manifest["model_sha256"]:
        raise RuntimeError("frozen training model hash mismatch")
    if file_sha256(config.source_path) != manifest["configuration_sha256"]:
        raise RuntimeError("configuration changed after freeze")
    current_source_hash = runtime_provenance(config.package_root)["source_tree_sha256"]
    if current_source_hash != manifest["source_tree_sha256"]:
        raise RuntimeError("training source changed after freeze")
    validate_core_feature_cache(config, "holdout")
    access_path = holdout_access_path(config)
    freeze_hash = expected_manifest_sha
    establish_holdout_access(config, access_path, freeze_hash, freeze_dir)
    bundle: FrozenTrainingBundle = joblib.load(model_path)
    frame = load_core_feature_frame(config, "holdout")
    if (
        frame["window_start"].min() < holdout_start
        or frame["window_start"].max() >= holdout_end
    ):
        raise RuntimeError("holdout feature file escapes the frozen holdout range")
    probability = bundle.probability(frame)
    selected = first_prediction_rows(
        frame,
        probability,
        bundle.confidence_threshold,
    ).with_columns(
        pl.lit(bundle.model.candidate_name).alias("candidate"),
    )
    eligible = frame["market_id"].n_unique()
    metrics = classification_metrics(selected, eligible_markets=eligible)
    baseline = baseline_metrics(selected, eligible_markets=eligible)
    paired = paired_uplift(selected)
    hourly_bootstrap = block_bootstrap_uplift(
        selected,
        resamples=config.gates.bootstrap_resamples,
        random_seed=config.model.random_seed,
        block="hour",
    )
    daily_bootstrap = block_bootstrap_uplift(
        selected,
        resamples=config.gates.bootstrap_resamples,
        random_seed=config.model.random_seed + 1,
        block="day",
    )
    walk_forward_accuracy = float(manifest["walk_forward_accuracy"])
    checks = qualification_checks(
        config,
        metrics,
        paired,
        hourly_bootstrap,
        walk_forward_accuracy,
    )
    qualified = all(check["passed"] for check in checks)
    development_run = Path(manifest["development_run"])
    development_metrics = json.loads(
        (development_run / "development-metrics.json").read_text()
    )
    holdout: dict[str, Any] = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "status": "prediction_qualified" if qualified else "blocked",
        "trading_status": "blocked_pending_execution_economics",
        "eligible_markets": eligible,
        "metrics": metrics,
        "baseline": baseline,
        "paired": paired,
        "hourly_block_bootstrap": hourly_bootstrap,
        "daily_block_bootstrap": daily_bootstrap,
        "daily_accuracy": daily_accuracy(selected),
        "time_accuracy": time_accuracy(selected),
        "timing": first_crossing_timing(
            selected,
            eligible_markets=eligible,
        ),
        "reliability": reliability_rows(selected),
        "qualification_checks": checks,
        "qualified": qualified,
    }
    combined = {
        **development_metrics,
        "status": holdout["status"],
        "trading_status": holdout["trading_status"],
        "freeze": manifest,
        "holdout": holdout,
    }
    selected.write_parquet(
        development_run / "holdout-predictions.parquet",
        compression="zstd",
    )
    write_json_atomic(development_run / "metrics.json", combined)
    update_progress(
        development_run,
        "holdout_complete",
        0.95,
        {
            "qualified": qualified,
            "accuracy": metrics["accuracy"],
            "uplift": paired["accuracy_uplift"],
        },
    )
    return development_run, combined


def qualification_checks(
    config: CoreTrainingConfig,
    metrics: dict[str, Any],
    paired: dict[str, Any],
    bootstrap: dict[str, Any],
    walk_forward_accuracy: float,
) -> list[dict[str, Any]]:
    checks = [
        gate("accuracy", metrics["accuracy"], config.gates.target_accuracy, ">="),
        gate(
            "wilson_lower_95",
            metrics["wilson_lower_95"],
            config.gates.target_wilson_lower,
            ">=",
        ),
        gate(
            "balanced_accuracy",
            metrics["balanced_accuracy"],
            config.gates.target_balanced_accuracy,
            ">=",
        ),
        gate(
            "up_recall",
            metrics["up_recall"],
            config.gates.minimum_direction_recall,
            ">=",
        ),
        gate(
            "down_recall",
            metrics["down_recall"],
            config.gates.minimum_direction_recall,
            ">=",
        ),
        gate(
            "coverage",
            metrics["coverage"],
            config.gates.minimum_coverage,
            ">=",
        ),
        gate(
            "accepted_markets",
            metrics["markets"],
            config.gates.minimum_holdout_markets,
            ">=",
        ),
        gate(
            "same_cohort_accuracy_uplift",
            paired["accuracy_uplift"],
            config.gates.minimum_same_time_path_uplift,
            ">=",
        ),
        gate(
            "hourly_bootstrap_lower_95",
            bootstrap["lower_95"],
            0.0,
            ">=",
        ),
        gate(
            "expected_calibration_error",
            metrics["expected_calibration_error"],
            config.gates.maximum_ece,
            "<=",
        ),
        gate(
            "walk_forward_holdout_accuracy_gap",
            abs(metrics["accuracy"] - walk_forward_accuracy),
            config.gates.maximum_walk_forward_holdout_gap,
            "<=",
        ),
    ]
    return checks


def gate(
    name: str,
    observed: float,
    target: float,
    operator: str,
) -> dict[str, Any]:
    if operator == ">=":
        passed = observed >= target
    elif operator == ">":
        passed = observed > target
    elif operator == "<=":
        passed = observed <= target
    else:
        raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "target": target,
        "passed": bool(passed),
    }


def establish_holdout_access(
    config: CoreTrainingConfig,
    access_path: Path,
    freeze_hash: str,
    freeze_dir: Path,
) -> None:
    holdout_start, holdout_end = evaluation_holdout_range(config)
    payload = {
        "schema_version": "btc-core-holdout-access-v1",
        "range_start": holdout_start.isoformat(),
        "range_end": holdout_end.isoformat(),
        "freeze_manifest_sha256": freeze_hash,
        "freeze_dir": str(freeze_dir),
        "accessed_at": datetime.now(UTC).isoformat(),
    }
    if access_path.exists():
        existing = json.loads(access_path.read_text())
        if existing.get("freeze_manifest_sha256") != freeze_hash:
            raise RuntimeError(
                "holdout range was already consumed by a different frozen candidate"
            )
        return
    write_json_atomic(access_path, payload)


def holdout_access_path(config: CoreTrainingConfig) -> Path:
    holdout_start, holdout_end = evaluation_holdout_range(config)
    start = holdout_start.date().isoformat()
    end = holdout_end.date().isoformat()
    return config.paths.artifacts / f"holdout-access-{start}-{end}.json"


def model_summary_payload(bundle: FrozenTrainingBundle) -> dict[str, Any]:
    model = bundle.model
    spec = model_candidate_spec(model)
    payload: dict[str, Any] = {
        "schema_version": "btc-core-training-model-summary-v1",
        "training_only": True,
        "candidate": model.candidate_name,
        "family": model.family,
        "row_weight_policy": spec.row_weight_policy,
        "row_weight_schedule": row_weight_schedule_payload(spec),
        "feature_names": list(model.feature_names),
        "hyperparameters": model.hyperparameters,
        "imputation_medians": model.imputation_medians.tolist(),
        "calibrator": asdict(bundle.calibrator),
        "confidence_threshold": bundle.confidence_threshold,
    }
    if model.family == "logistic":
        payload.update(
            {
                "standardization_means": model.standardization_means.tolist(),
                "standardization_scales": model.standardization_scales.tolist(),
                "coefficients": model.estimator.coef_[0].tolist(),
                "intercept": float(model.estimator.intercept_[0]),
            }
        )
    return payload


def pre_holdout_blocking_reasons(
    development: dict[str, Any],
    policy_passed: bool,
    calibrator: ProbabilityCalibrator,
) -> list[str]:
    reasons: list[str] = []
    if not development["passed_development"]:
        reasons.append("selected candidate did not pass walk-forward development gates")
    if not policy_passed:
        reasons.append("frozen policy-selection cohort did not pass its gates")
    if not calibrator.converged:
        reasons.append("probability calibrator did not converge")
    return reasons


def configure_native_thread_limits(config: CoreTrainingConfig) -> None:
    thread_count = str(config.compute.threads_per_fit)
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count
    os.environ["VECLIB_MAXIMUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["NUMEXPR_NUM_THREADS"] = thread_count
    os.environ.setdefault("POLARS_MAX_THREADS", str(config.compute.polars_threads))


def candidate_spec(
    name: str,
    config: CoreTrainingConfig | None = None,
) -> CandidateSpec:
    base: CandidateSpec | None = None
    for candidate in CANDIDATES + ADDITIONAL_CANDIDATES:
        if candidate.name == name:
            base = candidate
            break
    if base is None:
        raise ValueError(f"unknown core candidate: {name}")
    if config is None:
        return base
    model_config = getattr(config, "model", None)
    if model_config is None:
        return base
    schedules = {
        entry.candidate: entry.schedule
        for entry in getattr(model_config, "row_weight_schedules", ())
    }
    schedule = schedules.get(name)
    if schedule is None:
        return base
    policy = (
        MARKET_EQUAL_ROW_WEIGHT_POLICY
        if schedule.start_second is None
        else EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY
    )
    return CandidateSpec(
        base.name,
        base.family,
        base.feature_names,
        policy,
        schedule,
    )


def configured_candidate_specs(
    config: CoreTrainingConfig,
) -> tuple[CandidateSpec, ...]:
    return tuple(
        candidate_spec(name, config)
        for name in config.model.candidate_names
    )


def model_candidate_spec(model: FittedCoreModel) -> CandidateSpec:
    base = candidate_spec(model.candidate_name)
    schedule = vars(model).get("row_weight_schedule")
    policy = vars(model).get("row_weight_policy") or base.row_weight_policy
    if schedule is None:
        return base
    return CandidateSpec(
        base.name,
        base.family,
        base.feature_names,
        policy,
        schedule,
    )


def resolved_row_weight_schedule(
    spec: CandidateSpec,
) -> RowWeightScheduleConfig:
    schedule = spec.row_weight_schedule
    if schedule is None:
        raise RuntimeError(f"{spec.name} does not have a resolved row-weight schedule")
    return schedule


def row_weight_schedule_payload(spec: CandidateSpec) -> dict[str, Any]:
    return asdict(resolved_row_weight_schedule(spec))


def histogram_parameters(candidate: HistogramCandidate) -> dict[str, Any]:
    return {
        "learning_rate": candidate.learning_rate,
        "max_iter": candidate.max_iter,
        "max_leaf_nodes": candidate.max_leaf_nodes,
        "min_samples_leaf": candidate.min_samples_leaf,
        "l2_regularization": candidate.l2_regularization,
    }


def estimator_converged(estimator: Any) -> bool:
    if isinstance(estimator, LogisticRegression):
        return bool(estimator.n_iter_[0] < estimator.max_iter)
    if isinstance(estimator, HistGradientBoostingClassifier):
        return bool(estimator.n_iter_ <= estimator.max_iter)
    return True


def feature_matrix(
    frame: pl.DataFrame, feature_names: tuple[str, ...]
) -> np.ndarray:
    return frame.select(pl.col(list(feature_names)).cast(pl.Float64)).to_numpy()


def finite_medians(matrix: np.ndarray) -> np.ndarray:
    medians = np.nanmedian(matrix, axis=0)
    return np.where(np.isfinite(medians), medians, 0.0)


def transform_for_model(
    matrix: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray | None,
    scales: np.ndarray | None,
) -> np.ndarray:
    filled = np.where(np.isfinite(matrix), matrix, medians)
    if means is None or scales is None:
        return filled
    return (filled - means) / scales


def market_equal_weights(frame: pl.DataFrame) -> np.ndarray:
    return scheduled_market_equal_weights(frame, MARKET_EQUAL_ROW_WEIGHT_SCHEDULE)


def candidate_training_weights(
    frame: pl.DataFrame,
    spec: CandidateSpec,
) -> np.ndarray:
    return scheduled_market_equal_weights(
        frame,
        resolved_row_weight_schedule(spec),
    )


def early_entry_market_equal_weights(frame: pl.DataFrame) -> np.ndarray:
    return scheduled_market_equal_weights(frame, EARLY_ENTRY_ROW_WEIGHT_SCHEDULE)


def scheduled_market_equal_weights(
    frame: pl.DataFrame,
    schedule: RowWeightScheduleConfig,
) -> np.ndarray:
    if frame.is_empty():
        raise ValueError("cannot calculate row weights for an empty frame")
    if schedule.start_second is None:
        time_weight = pl.lit(schedule.multiplier)
    else:
        time_weight = (
            pl.when(
                pl.col("seconds_elapsed").is_between(
                    schedule.start_second,
                    schedule.end_second_inclusive,
                    closed="both",
                )
            )
            .then(pl.lit(schedule.multiplier))
            .otherwise(pl.lit(1.0))
        )
    weighted = frame.select(
        "market_id",
        time_weight.alias("time_weight"),
    ).with_columns(
        pl.col("time_weight").sum().over("market_id").alias("market_weight_total")
    )
    raw = (
        weighted["time_weight"].to_numpy()
        / weighted["market_weight_total"].to_numpy()
    )
    return raw / raw.mean()


def range_frame(
    frame: pl.DataFrame,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    selected = frame.filter(
        (pl.col("window_start") >= start) & (pl.col("window_start") < end)
    )
    if selected.is_empty():
        raise RuntimeError(
            f"configured cohort is empty: {start.isoformat()} to {end.isoformat()}"
        )
    return selected


def chronological_inner_split(
    frame: pl.DataFrame,
    *,
    validation_fraction: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    markets = (
        frame.select("market_id", "window_start")
        .unique(subset=["market_id"])
        .sort("window_start")
    )
    cut = max(1, int(markets.height * (1 - validation_fraction)))
    if cut >= markets.height:
        raise RuntimeError("inner split requires at least one validation market")
    train_ids = markets[:cut]["market_id"]
    validation_ids = markets[cut:]["market_id"]
    return (
        frame.filter(pl.col("market_id").is_in(train_ids.implode())),
        frame.filter(pl.col("market_id").is_in(validation_ids.implode())),
    )


def chronological_subsplit(
    frame: pl.DataFrame,
    *,
    fit_fraction: float,
    calibration_fraction: float,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    markets = (
        frame.select("market_id", "window_start")
        .unique(subset=["market_id"])
        .sort("window_start")
    )
    fit_end = max(1, int(markets.height * fit_fraction))
    calibration_end = fit_end + max(
        1, int(markets.height * calibration_fraction)
    )
    if calibration_end >= markets.height:
        raise RuntimeError("chronological subsplit leaves no policy markets")
    fit_ids = markets[:fit_end]["market_id"]
    calibration_ids = markets[fit_end:calibration_end]["market_id"]
    policy_ids = markets[calibration_end:]["market_id"]
    return (
        frame.filter(pl.col("market_id").is_in(fit_ids.implode())),
        frame.filter(pl.col("market_id").is_in(calibration_ids.implode())),
        frame.filter(pl.col("market_id").is_in(policy_ids.implode())),
    )


def combined_fold_rows(fold_results: list[dict[str, Any]]) -> pl.DataFrame:
    rows = [
        row
        for result in fold_results
        for row in result.get("prediction_rows", [])
    ]
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(
        ["observed_at", "market_id", "candidate", "fold_index"]
    )


def combined_fixed_time_rows(
    fold_results: list[dict[str, Any]],
) -> pl.DataFrame:
    rows = [
        row
        for result in fold_results
        for row in result.get("fixed_time_prediction_rows", [])
    ]
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(
        ["observed_at", "market_id", "candidate", "fold_index"]
    )


def scored_fold_probability_rows(
    validation: pl.DataFrame,
    probabilities: np.ndarray,
    spec: CandidateSpec,
    fold_index: int,
) -> pl.DataFrame:
    return scored_prediction_rows(validation, probabilities).with_columns(
        pl.lit(spec.name).alias("candidate"),
        pl.lit(fold_index).cast(pl.Int32).alias("fold_index"),
    )


def combined_scored_probability_rows(
    fold_results: list[dict[str, Any]],
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for result in fold_results:
        rows = result.get("scored_probability_rows")
        if isinstance(rows, pl.DataFrame):
            frames.append(rows)
        elif rows:
            frames.append(pl.DataFrame(rows))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed").sort(
        ["observed_at", "market_id", "candidate", "fold_index"]
    )


def report_fold_completion(
    result: dict[str, Any],
    completed: int,
    total: int,
    run_dir: Path,
) -> None:
    print(
        f"core fold: {result['candidate']} fold={result['fold_index'] + 1} "
        f"accuracy={result['metrics']['accuracy']:.4f} "
        f"uplift={result['paired']['accuracy_uplift']:+.4f} "
        f"coverage={result['metrics']['coverage']:.4f}",
        flush=True,
    )
    update_progress(
        run_dir,
        "walk_forward",
        0.05 + 0.50 * completed / total,
        {
            "completed_fits": completed,
            "total_fits": total,
            "latest_candidate": result["candidate"],
            "latest_fold": result["fold_index"] + 1,
        },
    )


def update_progress(
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
