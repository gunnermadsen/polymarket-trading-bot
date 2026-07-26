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
    config_to_dict,
    load_core_config,
)
from .core_evaluation import (
    baseline_metrics,
    block_bootstrap_uplift,
    choose_threshold,
    classification_metrics,
    daily_accuracy,
    first_prediction_rows,
    paired_uplift,
    reliability_rows,
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


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    feature_names: tuple[str, ...]


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
) -> tuple[Path, Path | None, dict[str, Any]]:
    feature_metadata = validate_core_feature_cache(config, "pre_holdout")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    update_progress(run_dir, "walk_forward", 0.05)
    tasks = [
        (candidate.name, fold_index)
        for candidate in CANDIDATES
        for fold_index in range(len(config.split.validation_windows))
    ]
    fold_results: list[dict[str, Any]] = []
    max_workers = min(config.compute.max_parallel_fits, len(tasks))
    configure_native_thread_limits(config)
    print(
        f"core develop: {len(tasks)} candidate/fold fits with {max_workers} workers",
        flush=True,
    )
    if max_workers == 1:
        for completed, (candidate_name, fold_index) in enumerate(tasks, start=1):
            result = evaluate_fold_task(
                config.source_path,
                candidate_name,
                fold_index,
            )
            fold_results.append(result)
            report_fold_completion(result, completed, len(tasks), run_dir)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_fold_task,
                    config.source_path,
                    candidate_name,
                    fold_index,
                ): (candidate_name, fold_index)
                for candidate_name, fold_index in tasks
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                fold_results.append(result)
                report_fold_completion(result, completed, len(tasks), run_dir)

    walk_forward_rows = combined_fold_rows(fold_results)
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
        f"positive_folds={selected_development['positive_uplift_folds']}",
        flush=True,
    )

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
    selected_spec = candidate_spec(selected_name)
    final_model, tuning = tune_and_fit_model(development, selected_spec, config)
    calibrator = fit_probability_calibrator(
        final_model,
        probability_calibration,
        config,
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
        and policy_paired["accuracy_uplift"] > 0
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
        "status": "candidate_ready_for_freeze" if ready_for_holdout else "blocked_pre_holdout",
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
            "threshold_history": thresholds,
            "passed": policy_passed,
        },
        "ready_for_holdout": ready_for_holdout,
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
    policy_rows.write_parquet(
        run_dir / "policy-predictions.parquet",
        compression="zstd",
    )
    write_json_atomic(run_dir / "development-metrics.json", metrics)
    freeze_dir: Path | None = None
    if ready_for_holdout:
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
    else:
        update_progress(
            run_dir,
            "blocked_pre_holdout",
            1.0,
            {"blocking_reasons": metrics["blocking_reasons"]},
        )
    return run_dir, freeze_dir, metrics


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
            candidate_spec(candidate_name),
            fold_index,
            config,
        )


def evaluate_fold(
    frame: pl.DataFrame,
    spec: CandidateSpec,
    fold_index: int,
    config: CoreTrainingConfig,
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
    calibrator = fit_probability_calibrator(model, calibration_frame, config)
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
    selected = first_prediction_rows(validation, probability, threshold)
    eligible = validation["market_id"].n_unique()
    metrics = classification_metrics(selected, eligible_markets=eligible)
    baseline = baseline_metrics(selected, eligible_markets=eligible)
    paired = paired_uplift(selected)
    return {
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
        "elapsed_seconds": time.perf_counter() - started,
        "prediction_rows": selected.to_dicts(),
    }


def aggregate_candidate_results(
    config: CoreTrainingConfig,
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for spec in CANDIDATES:
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
            {key: value for key, value in result.items() if key != "prediction_rows"}
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
        positive_folds = sum(
            result["paired"]["accuracy_uplift"] > 0 for result in folds
        )
        passed = bool(
            positive_folds >= config.gates.minimum_positive_folds
            and metrics["accuracy"] >= config.gates.target_accuracy
            and metrics["balanced_accuracy"] >= config.gates.target_balanced_accuracy
            and metrics["up_recall"] >= config.gates.minimum_direction_recall
            and metrics["down_recall"] >= config.gates.minimum_direction_recall
            and metrics["coverage"] >= config.gates.minimum_coverage
            and paired["accuracy_uplift"] > 0
        )
        output[spec.name] = {
            "candidate": spec.name,
            "family": spec.family,
            "feature_count": len(spec.feature_names),
            "features": list(spec.feature_names),
            "folds": summarized_folds,
            "out_of_fold": metrics,
            "baseline": baseline,
            "paired": paired,
            "bootstrap": bootstrap,
            "positive_uplift_folds": positive_folds,
            "passed_development": passed,
        }
    return output


def candidate_rank(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        result["passed_development"],
        result["bootstrap"]["lower_95"],
        result["paired"]["accuracy_uplift"],
        result["out_of_fold"]["balanced_accuracy"],
        result["out_of_fold"]["accuracy"],
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
        weights = market_equal_weights(validation)
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
    weights = market_equal_weights(frame)
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
    )


def fit_probability_calibrator(
    model: FittedCoreModel,
    frame: pl.DataFrame,
    config: CoreTrainingConfig,
) -> ProbabilityCalibrator:
    logits = model.raw_logit(frame).reshape(-1, 1)
    labels = frame["label_up"].to_numpy()
    weights = market_equal_weights(frame)
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
    freeze_id = (
        f"{metrics['run_id']}-{bundle.model.candidate_name}"
    )
    freeze_dir = config.paths.artifacts / freeze_id
    freeze_dir.mkdir(parents=True, exist_ok=False)
    model_path = freeze_dir / TRAINING_MODEL_FILENAME
    joblib.dump(bundle, model_path, compress=3)
    model_summary = model_summary_payload(bundle)
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
            "start": config.split.holdout_start.isoformat(),
            "end": config.split.holdout_end.isoformat(),
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
        frame["window_start"].min() < config.split.holdout_start
        or frame["window_start"].max() >= config.split.holdout_end
    ):
        raise RuntimeError("holdout feature file escapes the frozen holdout range")
    probability = bundle.probability(frame)
    selected = first_prediction_rows(
        frame,
        probability,
        bundle.confidence_threshold,
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
            config.gates.minimum_baseline_uplift,
            ">=",
        ),
        gate(
            "hourly_bootstrap_lower_95",
            bootstrap["lower_95"],
            0.0,
            ">",
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
    payload = {
        "schema_version": "btc-core-holdout-access-v1",
        "range_start": config.split.holdout_start.isoformat(),
        "range_end": config.split.holdout_end.isoformat(),
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
    start = config.split.holdout_start.date().isoformat()
    end = (config.split.holdout_end.date()).isoformat()
    return config.paths.artifacts / f"holdout-access-{start}-{end}.json"


def model_summary_payload(bundle: FrozenTrainingBundle) -> dict[str, Any]:
    model = bundle.model
    payload: dict[str, Any] = {
        "schema_version": "btc-core-training-model-summary-v1",
        "training_only": True,
        "candidate": model.candidate_name,
        "family": model.family,
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


def candidate_spec(name: str) -> CandidateSpec:
    for candidate in CANDIDATES:
        if candidate.name == name:
            return candidate
    raise ValueError(f"unknown core candidate: {name}")


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
    counts = frame.group_by("market_id").len().rename({"len": "market_rows"})
    market_rows = frame.join(counts, on="market_id", how="left")[
        "market_rows"
    ].to_numpy()
    raw = 1.0 / market_rows
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
    return pl.DataFrame(rows).sort(["observed_at", "market_id"])


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
