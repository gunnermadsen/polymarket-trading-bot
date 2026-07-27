from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .config import TrainingConfig
from .evaluation import (
    baseline_metrics,
    classification_metrics,
    confidence_buckets,
    daily_accuracy,
    prediction_rows,
    sigmoid,
    threshold_table,
    time_buckets,
)
from .features import (
    FEATURE_GROUPS,
    FEATURE_SCHEMA_VERSION,
    file_sha256,
    validate_feature_cache,
)
from .provenance import runtime_provenance

MODEL_SCHEMA_VERSION = "capitonic-logistic-artifact-v1"
THRESHOLD_SELECTION_OBJECTIVE = "maximize_calibration_wilson_lower"


def train_models(config: TrainingConfig) -> tuple[Path, dict[str, Any]]:
    feature_metadata = validate_feature_cache(config)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    update_progress(run_dir, "loading_features", 0.05)
    frame = pl.read_parquet(config.paths.feature_data).sort(["window_start", "seconds_elapsed"])
    splits, split_summary = chronological_split(frame, config)
    print(
        "split: "
        + ", ".join(
            f"{name}={part['market_id'].n_unique():,} markets/{part.height:,} rows"
            for name, part in splits.items()
        ),
        flush=True,
    )

    group_results: dict[str, Any] = {}
    group_rows: dict[str, pl.DataFrame] = {}
    for index, (group_name, feature_names) in enumerate(FEATURE_GROUPS.items(), start=1):
        print(f"train: fitting {group_name} ({len(feature_names)} features)", flush=True)
        update_progress(
            run_dir,
            f"training_{group_name}",
            0.10 + (index - 1) * (0.55 / len(FEATURE_GROUPS)),
        )
        result, rows = fit_feature_group(group_name, feature_names, splits, config)
        group_results[group_name] = result
        group_rows[group_name] = rows
        print(
            f"train: {group_name} test accuracy={result['test']['accuracy']:.4f} "
            f"markets={result['test']['markets']:,} threshold={result['confidence_threshold']:.2f}",
            flush=True,
        )

    selected_group = max(
        group_results,
        key=lambda name: (
            group_results[name]["calibration"]["wilson_lower_95"],
            group_results[name]["calibration"]["accuracy"],
            group_results[name]["calibration"]["markets"],
        ),
    )
    selected = group_results[selected_group]
    signed_accuracy_gap = selected["train"]["accuracy"] - selected["test"]["accuracy"]
    accuracy_gap = abs(signed_accuracy_gap)
    qualification = {
        "passed": bool(
            config.evaluation.holdout_is_independent
            and selected["threshold_qualified_on_calibration"]
            and selected["converged"]
            and selected["calibrator_converged"]
            and selected["test"]["accuracy"] >= config.model.target_accuracy
            and selected["test"]["wilson_lower_95"] >= config.model.target_wilson_lower
            and selected["test"]["markets"] >= config.model.minimum_test_markets
            and accuracy_gap <= config.model.maximum_train_test_accuracy_gap
        ),
        "holdout_is_independent": config.evaluation.holdout_is_independent,
        "holdout_note": config.evaluation.holdout_note,
        "calibration_threshold_passed": selected["threshold_qualified_on_calibration"],
        "optimizer_converged": selected["converged"],
        "calibrator_converged": selected["calibrator_converged"],
        "target_accuracy": config.model.target_accuracy,
        "target_wilson_lower": config.model.target_wilson_lower,
        "minimum_test_markets": config.model.minimum_test_markets,
        "maximum_train_test_accuracy_gap": config.model.maximum_train_test_accuracy_gap,
        "observed_train_test_accuracy_gap": accuracy_gap,
        "signed_train_test_accuracy_gap": signed_accuracy_gap,
    }

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "training_backend": {
            "name": "scikit-learn-logistic-regression",
            "device": "cpu",
            "host": platform.platform(),
            "reason": (
                "The canonical L2 logistic solver has no Metal backend; Apple GPU acceleration "
                "does not improve model quality and is deferred to a separately verified benchmark."
            ),
        },
        "runtime_provenance": runtime_provenance(config.package_root),
        "evaluation": {
            "holdout_is_independent": config.evaluation.holdout_is_independent,
            "holdout_note": config.evaluation.holdout_note,
        },
        "split": split_summary,
        "data": feature_metadata,
        "selected_feature_group": selected_group,
        "groups": group_results,
        "qualification": qualification,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    selected_rows = group_rows[selected_group]
    selected_rows.write_parquet(run_dir / "test-predictions.parquet", compression="zstd")
    matrix = selected["test"]["confusion_matrix"]
    (run_dir / "confusion-matrix.csv").write_text(
        "actual,predicted_down,predicted_up\n"
        f"down,{matrix[0][0]},{matrix[0][1]}\n"
        f"up,{matrix[1][0]},{matrix[1][1]}\n"
    )

    artifact = selected["artifact"]
    artifact["training_run_id"] = run_id
    artifact["training_data_sha256"] = file_sha256(config.paths.feature_data)
    artifact["training_config_sha256"] = hashlib.sha256(config.source_path.read_bytes()).hexdigest()
    artifact["qualification_passed"] = qualification["passed"]
    artifact["deployment_status"] = "qualified" if qualification["passed"] else "blocked"
    artifact["holdout_is_independent"] = config.evaluation.holdout_is_independent
    artifact["holdout_note"] = config.evaluation.holdout_note
    artifact["source_tree_sha256"] = metrics["runtime_provenance"]["source_tree_sha256"]
    artifact_path = run_dir / "model.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (run_dir / "model.sha256").write_text(file_sha256(artifact_path) + "\n")
    update_progress(run_dir, "trained", 0.80, {"selected_feature_group": selected_group})
    return run_dir, metrics


def fit_feature_group(
    group_name: str,
    feature_names: list[str],
    splits: dict[str, pl.DataFrame],
    config: TrainingConfig,
) -> tuple[dict[str, Any], pl.DataFrame]:
    train_frame = splits["train"]
    calibration_frame = splits["calibration"]
    test_frame = splits["test"]
    calibration_fit_frame, policy_selection_frame, calibration_subsplit = (
        split_probability_calibration(calibration_frame)
    )
    train_weight = market_equal_weights(train_frame)
    calibration_fit_weight = market_equal_weights(calibration_fit_frame)

    train_matrix = feature_matrix(train_frame, feature_names)
    calibration_fit_matrix = feature_matrix(calibration_fit_frame, feature_names)
    policy_selection_matrix = feature_matrix(policy_selection_frame, feature_names)
    test_matrix = feature_matrix(test_frame, feature_names)
    train_labels = train_frame["label_up"].to_numpy()
    calibration_fit_labels = calibration_fit_frame["label_up"].to_numpy()

    inner_market_count = train_frame["market_id"].n_unique()
    inner_cut = max(1, int(inner_market_count * 0.8))
    inner_markets = (
        train_frame.select("market_id", "window_start")
        .unique(subset=["market_id"])
        .sort("window_start")["market_id"]
        .to_list()
    )
    inner_train_mask = train_frame["market_id"].is_in(inner_markets[:inner_cut]).to_numpy()
    inner_medians, inner_means, inner_scales = fit_preprocessor(
        train_matrix[inner_train_mask], train_weight[inner_train_mask]
    )
    inner_train_scaled = transform(
        train_matrix[inner_train_mask], inner_medians, inner_means, inner_scales
    )
    inner_validation_scaled = transform(
        train_matrix[~inner_train_mask], inner_medians, inner_means, inner_scales
    )
    tune_history = []
    best_c: float | None = None
    best_loss = float("inf")
    for c_value in config.model.c_candidates:
        started = time.perf_counter()
        candidate = logistic_model(c_value, config.model.random_seed)
        candidate.fit(
            inner_train_scaled,
            train_labels[inner_train_mask],
            sample_weight=train_weight[inner_train_mask],
        )
        probability = candidate.predict_proba(inner_validation_scaled)[:, 1]
        candidate_loss = float(
            log_loss(
                train_labels[~inner_train_mask],
                probability,
                sample_weight=train_weight[~inner_train_mask],
                labels=[0, 1],
            )
        )
        record = {
            "c": c_value,
            "validation_log_loss": candidate_loss,
            "fit_seconds": time.perf_counter() - started,
            "iterations": int(candidate.n_iter_[0]),
            "converged": bool(candidate.n_iter_[0] < candidate.max_iter),
        }
        tune_history.append(record)
        print(
            f"tune: {group_name} C={c_value:g} log_loss={candidate_loss:.6f} "
            f"seconds={record['fit_seconds']:.2f}",
            flush=True,
        )
        if record["converged"] and candidate_loss < best_loss:
            best_loss = candidate_loss
            best_c = c_value
    if best_c is None:
        raise RuntimeError(f"no {group_name} regularization candidate converged")

    medians, means, scales = fit_preprocessor(train_matrix, train_weight)
    train_scaled = transform(train_matrix, medians, means, scales)
    calibration_fit_scaled = transform(calibration_fit_matrix, medians, means, scales)
    policy_selection_scaled = transform(policy_selection_matrix, medians, means, scales)
    test_scaled = transform(test_matrix, medians, means, scales)

    model_started = time.perf_counter()
    model = logistic_model(best_c, config.model.random_seed)
    model.fit(train_scaled, train_labels, sample_weight=train_weight)
    fit_seconds = time.perf_counter() - model_started

    calibration_logits = model.decision_function(calibration_fit_scaled).reshape(-1, 1)
    calibrator = LogisticRegression(C=1_000_000, solver="lbfgs", max_iter=300, tol=1e-9)
    calibrator.fit(
        calibration_logits,
        calibration_fit_labels,
        sample_weight=calibration_fit_weight,
    )
    calibration_slope = float(calibrator.coef_[0, 0])
    calibration_intercept = float(calibrator.intercept_[0])
    calibrator_converged = bool(calibrator.n_iter_[0] < calibrator.max_iter)

    probability_train = calibrated_probability(
        model.decision_function(train_scaled), calibration_slope, calibration_intercept
    )
    probability_calibration_fit = calibrated_probability(
        model.decision_function(calibration_fit_scaled),
        calibration_slope,
        calibration_intercept,
    )
    probability_policy_selection = calibrated_probability(
        model.decision_function(policy_selection_scaled),
        calibration_slope,
        calibration_intercept,
    )
    probability_test = calibrated_probability(
        model.decision_function(test_scaled), calibration_slope, calibration_intercept
    )

    thresholds = list(
        np.round(
            np.arange(
                config.model.confidence_min,
                config.model.confidence_max + config.model.confidence_step / 2,
                config.model.confidence_step,
            ),
            6,
        )
    )
    threshold_results = threshold_table(
        policy_selection_frame, probability_policy_selection, thresholds
    )
    selected_threshold, threshold_qualified = select_confidence_threshold(threshold_results, config)

    train_rows = prediction_rows(
        train_frame, probability_train, selected_threshold, require_executable=False
    )
    calibration_fit_rows = prediction_rows(
        calibration_fit_frame,
        probability_calibration_fit,
        selected_threshold,
        require_executable=False,
    )
    policy_selection_rows = prediction_rows(
        policy_selection_frame,
        probability_policy_selection,
        selected_threshold,
        require_executable=False,
    )
    test_rows = prediction_rows(
        test_frame, probability_test, selected_threshold, require_executable=False
    )
    selected_prediction_executable_rows = test_rows.filter(pl.col("selected_side_executable"))
    first_executable_test_rows = prediction_rows(
        test_frame, probability_test, selected_threshold, require_executable=True
    )

    test_metrics = classification_metrics(test_rows)
    result: dict[str, Any] = {
        "feature_group": group_name,
        "feature_count": len(feature_names),
        "features": feature_names,
        "selected_c": best_c,
        "fit_seconds": fit_seconds,
        "iterations": int(model.n_iter_[0]),
        "converged": bool(model.n_iter_[0] < model.max_iter),
        "calibrator_iterations": int(calibrator.n_iter_[0]),
        "calibrator_converged": calibrator_converged,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "confidence_threshold": selected_threshold,
        "threshold_selection_objective": THRESHOLD_SELECTION_OBJECTIVE,
        "threshold_qualified_on_calibration": threshold_qualified,
        "threshold_history": threshold_results,
        "tuning_history": tune_history,
        "train": classification_metrics(train_rows),
        "probability_calibration": classification_metrics(calibration_fit_rows),
        "calibration": classification_metrics(policy_selection_rows),
        "calibration_subsplit": calibration_subsplit,
        "test": test_metrics,
        "selected_prediction_executable": classification_metrics(
            selected_prediction_executable_rows
        ),
        "first_executable_test": classification_metrics(first_executable_test_rows),
        "test_baselines": {
            "binance_sign": baseline_metrics(test_rows, "binance_sign_up"),
            "polymarket_favorite": baseline_metrics(test_rows, "market_favorite_up"),
            "majority_up": baseline_metrics(
                test_rows.with_columns(
                    pl.lit(int(train_labels.mean() >= 0.5)).alias("majority_up")
                ),
                "majority_up",
            ),
        },
        "confidence_buckets": confidence_buckets(test_rows),
        "time_buckets": time_buckets(test_rows),
        "daily_accuracy": daily_accuracy(test_rows),
        "coefficient_ranking": sorted(
            (
                {"feature": feature, "coefficient": float(coefficient)}
                for feature, coefficient in zip(feature_names, model.coef_[0], strict=True)
            ),
            key=lambda row: abs(row["coefficient"]),
            reverse=True,
        ),
        "artifact": {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_id": f"btc-5m-directional-logistic-{group_name}-v1",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": feature_names,
            "imputation_medians": medians.tolist(),
            "standardization_means": means.tolist(),
            "standardization_scales": scales.tolist(),
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "calibration_slope": calibration_slope,
            "calibration_intercept": calibration_intercept,
            "confidence_threshold": selected_threshold,
            "threshold_selection_objective": THRESHOLD_SELECTION_OBJECTIVE,
            "optimizer_converged": bool(model.n_iter_[0] < model.max_iter),
            "calibrator_converged": calibrator_converged,
            "golden_vectors": golden_inference_vectors(test_frame, test_matrix, probability_test),
        },
    }
    return result, test_rows


def split_probability_calibration(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    markets = (
        frame.select("market_id", "window_start").unique(subset=["market_id"]).sort("window_start")
    )
    cut = markets.height // 2
    if cut < 1 or cut >= markets.height:
        raise ValueError("calibration split needs at least two markets")
    calibration_ids = markets[:cut]["market_id"]
    policy_ids = markets[cut:]["market_id"]
    calibration_fit = frame.filter(pl.col("market_id").is_in(calibration_ids.implode()))
    policy_selection = frame.filter(pl.col("market_id").is_in(policy_ids.implode()))
    summary = {
        "probability_calibration": summarize_frame(calibration_fit),
        "policy_selection": summarize_frame(policy_selection),
    }
    return calibration_fit, policy_selection, summary


def chronological_split(
    frame: pl.DataFrame, config: TrainingConfig
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    markets = (
        frame.select("market_id", "window_start", "label_up")
        .unique(subset=["market_id"])
        .sort("window_start")
    )
    total = markets.height
    train_end = int(total * config.split.train_fraction)
    calibration_end = train_end + int(total * config.split.calibration_fraction)
    identifiers = {
        "train": markets[:train_end]["market_id"],
        "calibration": markets[train_end:calibration_end]["market_id"],
        "test": markets[calibration_end:]["market_id"],
    }
    splits = {
        name: frame.filter(pl.col("market_id").is_in(values.implode()))
        for name, values in identifiers.items()
    }
    summary = {name: summarize_frame(part) for name, part in splits.items()}
    return splits, summary


def summarize_frame(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "markets": frame["market_id"].n_unique(),
        "rows": frame.height,
        "range_start": frame["window_start"].min().isoformat(),
        "range_end": frame["window_start"].max().isoformat(),
        "up_markets": frame.filter(pl.col("label_up") == 1)["market_id"].n_unique(),
        "down_markets": frame.filter(pl.col("label_up") == 0)["market_id"].n_unique(),
    }


def market_equal_weights(frame: pl.DataFrame) -> np.ndarray:
    counts = frame.group_by("market_id").len().rename({"len": "market_rows"})
    weights = frame.join(counts, on="market_id", how="left")["market_rows"].to_numpy()
    raw = 1.0 / weights
    return raw / raw.mean()


def feature_matrix(frame: pl.DataFrame, feature_names: list[str]) -> np.ndarray:
    return frame.select(pl.col(feature_names).cast(pl.Float64)).to_numpy()


def select_confidence_threshold(
    threshold_results: list[dict[str, Any]], config: TrainingConfig
) -> tuple[float, bool]:
    qualifying = [
        row
        for row in threshold_results
        if row["markets"] >= config.model.minimum_calibration_markets
        and row["accuracy"] >= config.model.target_accuracy
        and row["wilson_lower_95"] >= config.model.target_wilson_lower
    ]
    candidates = qualifying or threshold_results
    selected = max(
        candidates,
        key=lambda row: (
            row["wilson_lower_95"],
            row["accuracy"],
            row["markets"],
            -row["threshold"],
        ),
    )
    return selected["threshold"], bool(qualifying)


def fit_preprocessor(
    matrix: np.ndarray, sample_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(matrix, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(matrix), matrix, medians)
    means = np.average(filled, axis=0, weights=sample_weight)
    variance = np.average((filled - means) ** 2, axis=0, weights=sample_weight)
    scales = np.sqrt(np.maximum(variance, 0.0))
    scales = np.where(scales > 1e-12, scales, 1.0)
    return medians, means, scales


def transform(
    matrix: np.ndarray, medians: np.ndarray, means: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    filled = np.where(np.isfinite(matrix), matrix, medians)
    return (filled - means) / scales


def logistic_model(c_value: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=2_000,
        tol=1e-7,
        random_state=seed,
    )


def calibrated_probability(logits: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return sigmoid(logits * slope + intercept)


def golden_inference_vectors(
    frame: pl.DataFrame, matrix: np.ndarray, probabilities: np.ndarray
) -> list[dict[str, Any]]:
    indices = sorted({0, frame.height // 2, frame.height - 1})
    vectors = []
    for index in indices:
        values = [
            float(value) if np.isfinite(value) else None
            for value in matrix[index].astype(np.float64)
        ]
        vectors.append(
            {
                "market_id": frame[index, "market_id"],
                "observed_at": frame[index, "observed_at"].isoformat(),
                "feature_values": values,
                "expected_probability_up": float(probabilities[index]),
            }
        )
    return vectors


def update_progress(
    run_dir: Path, stage: str, completion: float, details: dict[str, Any] | None = None
) -> None:
    payload = {
        "stage": stage,
        "completion": completion,
        "updated_at": datetime.now(UTC).isoformat(),
        "details": details or {},
    }
    (run_dir / "progress.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
