from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .admission_config import (
    AdmissionAdvancementGates,
    AdmissionBenchmarkConfig,
    admission_config_to_dict,
)
from .core_benchmark import (
    AdvancementCriteria,
    BenchmarkEvidence,
    CandidatePolicy,
    benchmark_predictions,
)
from .core_config import load_core_config
from .core_evaluation import classification_metrics, first_crossing_timing
from .core_execution import (
    ExecutionEvidenceConfig,
    load_execution_evidence_manifest,
)
from .core_extract import file_sha256, write_json_atomic
from .core_features import (
    CORE_BOUNDARY_FEATURES,
    feature_destination,
    load_core_feature_frame,
)
from .core_training import ProbabilityCalibrator
from .persistence_benchmark import (
    attach_execution_evidence,
    load_execution_evidence,
)
from .policy_benchmark import (
    TimeBandPolicySelection,
    absolute_policy_checks,
    apply_time_band_policy,
    load_probability_evidence,
    load_saved_probability_manifest,
    select_causal_time_band_thresholds,
)
from .provenance import runtime_provenance

ADMISSION_BENCHMARK_SCHEMA_VERSION = "btc-correctness-admission-benchmark-v1"
SELECTOR_ROW_WEIGHT_POLICY = "equal_total_per_market"
SELECTOR_BASE_FEATURES = (
    "selector_base_probability_up",
    "selector_base_confidence",
    "selector_base_predicted_up",
    "selector_seconds_elapsed_fraction",
    "selector_probability_delta_5s",
    "selector_probability_delta_15s",
    "selector_confidence_delta_5s",
    "selector_direction_persistence_15s",
    "selector_confidence_mean_15s",
)
SELECTOR_FEATURES = (*SELECTOR_BASE_FEATURES, *CORE_BOUNDARY_FEATURES)
SELECTOR_JOIN_KEYS = (
    "market_id",
    "window_start",
    "observed_at",
    "seconds_elapsed",
)


@dataclass
class FittedAdmissionSelector:
    feature_names: tuple[str, ...]
    imputation_medians: np.ndarray
    standardization_means: np.ndarray
    standardization_scales: np.ndarray
    estimator: LogisticRegression
    regularization_c: float
    row_weight_policy: str = SELECTOR_ROW_WEIGHT_POLICY

    def raw_logit(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = _feature_matrix(frame, self.feature_names)
        filled = np.where(np.isfinite(matrix), matrix, self.imputation_medians)
        transformed = (
            filled - self.standardization_means
        ) / self.standardization_scales
        return self.estimator.decision_function(transformed).astype(np.float64)


def run_admission_benchmark(
    config: AdmissionBenchmarkConfig,
) -> tuple[Path, dict[str, Any]]:
    """Fit and evaluate the development-only rolling correctness selector.

    The function consumes saved out-of-fold probabilities and the pre-holdout core
    feature cache. It never reads a holdout feature cache, queries a database, exports
    a runtime artifact, or changes the trading runtime.
    """

    manifest = load_saved_probability_manifest(config.probability_manifest)
    core_config = load_core_config(config.core_config)
    core_features = load_core_feature_frame(core_config, "pre_holdout")
    core_feature_path = feature_destination(core_config, "pre_holdout")
    core_feature_metadata = core_feature_path.with_suffix(".metadata.json")

    control_folds = load_validation_folds(
        config,
        manifest,
        config.control_candidate,
    )
    boundary_folds = load_validation_folds(
        config,
        manifest,
        config.base_candidate,
    )
    validation_universes = {
        fold_index: set(control_folds[fold_index]["market_id"].to_list())
        for fold_index in config.evaluation_folds
    }
    _validate_boundary_universes(
        boundary_folds,
        validation_universes,
        config.evaluation_folds,
    )

    control_scored, control_fold_results = evaluate_source_candidate(
        config,
        manifest,
        config.control_candidate,
        control_folds,
        validation_universes,
    )
    boundary_scored, boundary_fold_results = evaluate_source_candidate(
        config,
        manifest,
        config.base_candidate,
        boundary_folds,
        validation_universes,
    )
    (
        selector_scored,
        selector_fold_results,
        selector_training_summary,
    ) = evaluate_correctness_selector(
        config,
        boundary_folds,
        core_features,
        validation_universes,
    )

    execution_config = _execution_config(config)
    execution_manifest = load_execution_evidence_manifest(execution_config)
    execution = load_execution_evidence(execution_config)
    scored_frames = {
        config.control_candidate: attach_execution_evidence(
            control_scored,
            execution,
        ),
        config.base_candidate: attach_execution_evidence(
            boundary_scored,
            execution,
        ),
        config.selector_candidate: attach_execution_evidence(
            selector_scored,
            execution,
        ),
    }
    eligible_market_ids = sorted(
        market_id
        for fold_index in config.evaluation_folds
        for market_id in validation_universes[fold_index]
    )
    policies = {
        name: CandidatePolicy(
            confidence_threshold=None,
            deployment_compatible=False,
            selection_mode="time_band_preselected",
            confidence_threshold_min=min(config.threshold_candidates),
            confidence_threshold_max=max(config.threshold_candidates),
        )
        for name in config.candidate_names
    }
    diagnostics = benchmark_predictions(
        scored_frames,
        policies=policies,
        control_candidate=config.control_candidate,
        evidence=BenchmarkEvidence(
            label=(
                "Rolling correctness-admission evaluation on three consumed "
                "chronological development folds"
            ),
            kind="development",
            independent=False,
        ),
        eligible_market_ids=eligible_market_ids,
        minimum_samples=config.gates.minimum_selected_markets,
        minimum_executable_samples=config.gates.minimum_executable_markets,
        quantity=config.quantity,
        criteria=_benchmark_criteria(config.gates),
    )

    fold_results = {
        config.control_candidate: control_fold_results,
        config.base_candidate: boundary_fold_results,
        config.selector_candidate: selector_fold_results,
    }
    candidate_results: dict[str, dict[str, Any]] = {}
    control_metrics = diagnostics["candidates"][config.control_candidate][
        "own_policy"
    ]
    for candidate_name in config.candidate_names:
        candidate_diagnostics = diagnostics["candidates"][candidate_name]
        own_policy = candidate_diagnostics["own_policy"]
        comparison = diagnostics["common_comparisons"].get(candidate_name)
        advancement = admission_advancement_checks(
            candidate_name=candidate_name,
            control_candidate=config.control_candidate,
            metrics=own_policy,
            control_metrics=control_metrics,
            fold_results=fold_results[candidate_name],
            comparison=comparison,
            gates=config.gates,
            quantity=config.quantity,
            evidence_is_independent=config.evaluation_is_independent,
        )
        candidate_results[candidate_name] = {
            "candidate": candidate_name,
            "fold_count": len(fold_results[candidate_name]),
            "evaluation_folds": list(config.evaluation_folds),
            "out_of_fold": own_policy,
            "timing": {
                "median_first_crossing_seconds": own_policy[
                    "median_seconds_elapsed"
                ],
                "p90_first_crossing_seconds": own_policy["p90_seconds_elapsed"],
            },
            "no_trade_rate": own_policy["no_trade_rate"],
            "execution": own_policy["execution"],
            "time_bands": candidate_diagnostics["time_bands"],
            "checkpoints": candidate_diagnostics["checkpoints"],
            "folds": fold_results[candidate_name],
            "advance": advancement,
        }
    candidate_results[config.selector_candidate][
        "selector_training"
    ] = selector_training_summary

    passing = [
        name
        for name in (config.base_candidate, config.selector_candidate)
        if candidate_results[name]["advance"]["benchmark_passed"]
    ]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for candidate_name, frame in scored_frames.items():
        selected = frame.filter(pl.col("policy_selected"))
        destination = run_dir / f"{candidate_name}-selected-validation.parquet"
        _write_parquet_atomic(selected, destination)
        candidate_results[candidate_name]["selected_validation_evidence"] = {
            "path": destination.name,
            "sha256": file_sha256(destination),
            "rows": selected.height,
            "markets": selected["market_id"].n_unique(),
        }

    benchmark = {
        "schema_version": ADMISSION_BENCHMARK_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": admission_config_to_dict(config),
        "evaluation_note": config.evaluation_note,
        "evaluation_is_independent": config.evaluation_is_independent,
        "evaluation_folds": list(config.evaluation_folds),
        "probability_evidence": {
            "manifest": str(config.probability_manifest),
            "manifest_sha256": file_sha256(config.probability_manifest),
            "schema_version": manifest["schema_version"],
            "source_benchmark_profile": manifest.get("source_benchmark_profile"),
            "source_config": manifest.get("source_config"),
            "source_config_sha256": manifest.get("source_config_sha256"),
            "candidate_names": manifest.get("candidate_names"),
            "fold_count": manifest["fold_count"],
            "checksums_verified": True,
        },
        "core_feature_evidence": {
            "scope": "pre_holdout",
            "path": str(core_feature_path),
            "sha256": file_sha256(core_feature_path),
            "metadata": str(core_feature_metadata),
            "metadata_sha256": file_sha256(core_feature_metadata),
            "feature_names": list(CORE_BOUNDARY_FEATURES),
            "holdout_accessed": False,
        },
        "execution_evidence": {
            "manifest": str(config.execution_evidence / "manifest.json"),
            "manifest_sha256": file_sha256(
                config.execution_evidence / "manifest.json"
            ),
            "source_contract": execution_manifest.get("source_contract"),
            "source_schema_version": execution_manifest.get(
                "source_schema_version"
            ),
            "range_start": execution_manifest["range_start"],
            "range_end": execution_manifest["range_end"],
            "quantity": execution_manifest["quantity"],
            "checksums_verified": True,
        },
        "selector_contract": {
            "target": "base boundary direction is correct",
            "feature_names": list(SELECTOR_FEATURES),
            "feature_scope": "core-only point-in-time inputs",
            "row_weight_policy": SELECTOR_ROW_WEIGHT_POLICY,
            "rolling_fit_rule": "all validation folds earlier than eval_fold - 1",
            "calibration_rule": (
                "first chronological half of the immediately prior validation fold"
            ),
            "policy_rule": (
                "second chronological half of the immediately prior validation fold"
            ),
            "calibration_kind": "four_band_platt",
            "admission_rule": "q below 0.5 abstains; admitted rows retain base direction",
            "direction_reversal_allowed": False,
        },
        "runtime_provenance": runtime_provenance(config.package_root),
        "eligible_markets": len(eligible_market_ids),
        "control_candidate": config.control_candidate,
        "base_candidate": config.base_candidate,
        "selector_candidate": config.selector_candidate,
        "candidate_order": list(config.candidate_names),
        "candidates": candidate_results,
        "common_comparisons": diagnostics["common_comparisons"],
        "benchmark_diagnostics": diagnostics,
        "benchmark_passed_candidates": passing,
        "deployment_qualified_candidates": [],
        "winner": (
            config.selector_candidate
            if config.selector_candidate in passing
            else config.base_candidate
            if config.base_candidate in passing
            else None
        ),
        "deployment": {
            "status": "not_qualified",
            "reason": (
                "the selector has three rolling validation folds; at least "
                f"{config.gates.required_deployment_validation_folds} are required"
            ),
            "selector_validation_folds": len(config.evaluation_folds),
            "required_validation_folds": (
                config.gates.required_deployment_validation_folds
            ),
            "development_evidence": True,
            "holdout_accessed": False,
            "database_accessed": False,
            "runtime_exported": False,
            "runtime_changed": False,
        },
    }
    write_json_atomic(run_dir / "benchmark.json", benchmark)
    from .admission_report import generate_admission_report

    generate_admission_report(benchmark, run_dir / "report.html")
    return run_dir, benchmark


def load_validation_folds(
    config: AdmissionBenchmarkConfig,
    manifest: dict[str, Any],
    candidate_name: str,
) -> dict[int, pl.DataFrame]:
    candidate = manifest["candidates"].get(candidate_name)
    if candidate is None:
        raise ValueError(f"saved probability manifest is missing {candidate_name}")
    output: dict[int, pl.DataFrame] = {}
    for record in candidate["folds"]:
        fold_index = int(record["fold_index"])
        output[fold_index] = load_probability_evidence(
            config.probability_manifest,
            record["validation"],
            candidate_name,
            fold_index,
            "validation",
        )
    if tuple(sorted(output)) != tuple(range(int(manifest["fold_count"]))):
        raise RuntimeError(f"{candidate_name} validation fold sequence changed")
    return output


def evaluate_source_candidate(
    config: AdmissionBenchmarkConfig,
    manifest: dict[str, Any],
    candidate_name: str,
    validation_folds: dict[int, pl.DataFrame],
    validation_universes: dict[int, set[str]],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    records = {
        int(record["fold_index"]): record
        for record in manifest["candidates"][candidate_name]["folds"]
    }
    scored_frames: list[pl.DataFrame] = []
    fold_results: list[dict[str, Any]] = []
    for fold_index in config.evaluation_folds:
        policy_rows = load_probability_evidence(
            config.probability_manifest,
            records[fold_index]["policy_selection"],
            candidate_name,
            fold_index,
            "policy_selection",
        )
        policy_rows = _model_eligible_rows(policy_rows)
        validation_rows = _model_eligible_rows(validation_folds[fold_index])
        if policy_rows["window_start"].max() >= validation_rows["window_start"].min():
            raise RuntimeError(
                f"{candidate_name} fold {fold_index} source policy is not causal"
            )
        selection = select_causal_time_band_thresholds(policy_rows, config)
        scored = apply_time_band_policy(
            validation_rows,
            config.bands,
            selection.threshold_map(),
        )
        selected = scored.filter(pl.col("policy_selected"))
        eligible_markets = len(validation_universes[fold_index])
        metrics = classification_metrics(
            selected,
            eligible_markets=eligible_markets,
        )
        timing = first_crossing_timing(
            selected,
            eligible_markets=eligible_markets,
        )
        checks = absolute_policy_checks(metrics, timing, config.gates)
        scored_frames.append(scored)
        fold_results.append(
            {
                "fold_index": fold_index,
                "causal_order_verified": True,
                "policy_selection": _selection_payload(selection),
                "policy_selection_range": _frame_range(policy_rows),
                "evaluation_range": _frame_range(validation_rows),
                "validation": {
                    "metrics": metrics,
                    "timing": timing,
                    "checks": checks,
                    "qualified": all(check["passed"] for check in checks),
                    "thresholds_frozen_before_access": True,
                    "threshold_search_performed": False,
                },
            }
        )
    return (
        pl.concat(scored_frames, how="vertical_relaxed").sort(
            ["observed_at", "market_id", "fold_index"]
        ),
        fold_results,
    )


def evaluate_correctness_selector(
    config: AdmissionBenchmarkConfig,
    boundary_folds: dict[int, pl.DataFrame],
    core_features: pl.DataFrame,
    validation_universes: dict[int, set[str]],
) -> tuple[pl.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    prepared = {
        fold_index: prepare_selector_features(frame, core_features)
        for fold_index, frame in boundary_folds.items()
    }
    scored_frames: list[pl.DataFrame] = []
    fold_results: list[dict[str, Any]] = []
    fit_summaries: list[dict[str, Any]] = []
    for eval_fold in config.evaluation_folds:
        calibration_fold = eval_fold - 1
        fit_fold_indexes = tuple(range(calibration_fold))
        if not fit_fold_indexes:
            raise RuntimeError("selector evaluation has no earlier fitting fold")
        fit_rows = pl.concat(
            [prepared[index] for index in fit_fold_indexes],
            how="vertical_relaxed",
        )
        calibration_rows, policy_rows, split_metadata = (
            split_prior_fold_chronologically(prepared[calibration_fold])
        )
        evaluation_rows = prepared[eval_fold]
        if fit_rows["window_start"].max() >= calibration_rows["window_start"].min():
            raise RuntimeError(f"selector eval fold {eval_fold} fitting is not causal")
        if calibration_rows["window_start"].max() >= policy_rows["window_start"].min():
            raise RuntimeError(
                f"selector eval fold {eval_fold} calibration is not causal"
            )
        if policy_rows["window_start"].max() >= evaluation_rows["window_start"].min():
            raise RuntimeError(
                f"selector eval fold {eval_fold} policy selection is not causal"
            )

        selector_model = fit_admission_selector(
            fit_rows,
            regularization_c=config.selector.regularization_c,
            random_seed=config.selector.random_seed,
        )
        calibrators, calibration_diagnostics = fit_time_band_platt_calibrators(
            selector_model,
            calibration_rows,
            config,
        )
        policy_q = score_selector_q(
            selector_model,
            calibrators,
            policy_rows,
            config,
        )
        policy_scored = selector_probability_frame(
            policy_rows,
            policy_q,
            selector_candidate=config.selector_candidate,
            admission_floor=config.selector.admission_floor,
        )
        selection = select_causal_time_band_thresholds(policy_scored, config)

        evaluation_q = score_selector_q(
            selector_model,
            calibrators,
            evaluation_rows,
            config,
        )
        evaluation_probability = selector_probability_frame(
            evaluation_rows,
            evaluation_q,
            selector_candidate=config.selector_candidate,
            admission_floor=config.selector.admission_floor,
        )
        validation_scored = apply_time_band_policy(
            evaluation_probability,
            config.bands,
            selection.threshold_map(),
        )
        _validate_selector_direction_contract(validation_scored)
        selected = validation_scored.filter(pl.col("policy_selected"))
        eligible_markets = len(validation_universes[eval_fold])
        metrics = classification_metrics(
            selected,
            eligible_markets=eligible_markets,
        )
        timing = first_crossing_timing(
            selected,
            eligible_markets=eligible_markets,
        )
        checks = absolute_policy_checks(metrics, timing, config.gates)
        admitted = validation_scored.filter(pl.col("model_eligible"))
        all_markets = validation_scored["market_id"].n_unique()
        admitted_markets = admitted["market_id"].n_unique()
        scored_frames.append(validation_scored)
        model_summary = selector_model_summary(selector_model)
        fit_summaries.append(
            {
                "evaluation_fold": eval_fold,
                "fit_folds": list(fit_fold_indexes),
                "model": model_summary,
                "calibrators": calibration_diagnostics,
            }
        )
        fold_results.append(
            {
                "fold_index": eval_fold,
                "causal_order_verified": True,
                "selector_fit_folds": list(fit_fold_indexes),
                "selector_fit_range": _frame_range(fit_rows),
                "prior_fold_index": calibration_fold,
                "prior_fold_split": split_metadata,
                "calibration_range": _frame_range(calibration_rows),
                "policy_selection_range": _frame_range(policy_rows),
                "evaluation_range": _frame_range(evaluation_rows),
                "selector_model": model_summary,
                "calibration": {
                    "kind": "four_band_platt",
                    "bands": calibration_diagnostics,
                },
                "policy_selection": _selection_payload(selection),
                "validation": {
                    "metrics": metrics,
                    "timing": timing,
                    "checks": checks,
                    "qualified": all(check["passed"] for check in checks),
                    "thresholds_frozen_before_access": True,
                    "threshold_search_performed": False,
                    "validation_score_passes": 1,
                    "direction_reversals": 0,
                    "q_below_floor_selected_rows": validation_scored.filter(
                        (pl.col("selector_q") < config.selector.admission_floor)
                        & pl.col("policy_selected")
                    ).height,
                    "admitted_rows": admitted.height,
                    "abstained_rows": validation_scored.height - admitted.height,
                    "admitted_markets": admitted_markets,
                    "fully_abstained_markets": all_markets - admitted_markets,
                },
            }
        )
    return (
        pl.concat(scored_frames, how="vertical_relaxed").sort(
            ["observed_at", "market_id", "fold_index"]
        ),
        fold_results,
        {
            "feature_names": list(SELECTOR_FEATURES),
            "row_weight_policy": SELECTOR_ROW_WEIGHT_POLICY,
            "evaluation_models": fit_summaries,
            "validation_fold_count": len(fold_results),
            "deployment_required_fold_count": (
                config.gates.required_deployment_validation_folds
            ),
            "deployment_qualified": False,
        },
    )


def prepare_selector_features(
    probability_rows: pl.DataFrame,
    core_features: pl.DataFrame,
) -> pl.DataFrame:
    required_probability = {
        *SELECTOR_JOIN_KEYS,
        "probability_up",
        "confidence",
        "predicted_up",
        "correct",
        "label_up",
        "candidate",
        "fold_index",
    }
    missing_probability = sorted(
        required_probability - set(probability_rows.columns)
    )
    if missing_probability:
        raise ValueError(
            "selector probability rows are missing columns: "
            + ", ".join(missing_probability)
        )
    missing_core = sorted(
        set(SELECTOR_JOIN_KEYS).union(CORE_BOUNDARY_FEATURES)
        - set(core_features.columns)
    )
    if missing_core:
        raise ValueError(
            "selector core cache is missing columns: " + ", ".join(missing_core)
        )
    core_join = core_features.select(
        *SELECTOR_JOIN_KEYS,
        *CORE_BOUNDARY_FEATURES,
    ).with_columns(pl.lit(True).alias("_selector_core_joined"))
    joined = probability_rows.join(
        core_join,
        on=list(SELECTOR_JOIN_KEYS),
        how="left",
        validate="m:1",
    )
    if joined["_selector_core_joined"].null_count():
        missing = joined.filter(pl.col("_selector_core_joined").is_null()).height
        raise RuntimeError(
            f"selector core feature join missed {missing} probability rows"
        )
    return build_causal_selector_features(
        joined.drop("_selector_core_joined")
    )


def build_causal_selector_features(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "market_id",
        "seconds_elapsed",
        "probability_up",
        "confidence",
        "predicted_up",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "causal selector features are missing columns: " + ", ".join(missing)
        )
    sorted_frame = frame.sort(
        ["market_id", "seconds_elapsed", "observed_at"]
    )
    lag_expressions: list[pl.Expr] = []
    for lag in (1, 2, 3):
        lag_expressions.extend(
            (
                pl.col("seconds_elapsed")
                .shift(lag)
                .over("market_id")
                .alias(f"_selector_seconds_lag_{lag}"),
                pl.col("probability_up")
                .shift(lag)
                .over("market_id")
                .alias(f"_selector_probability_lag_{lag}"),
                pl.col("confidence")
                .shift(lag)
                .over("market_id")
                .alias(f"_selector_confidence_lag_{lag}"),
                pl.col("predicted_up")
                .shift(lag)
                .over("market_id")
                .alias(f"_selector_direction_lag_{lag}"),
            )
        )
    with_lags = sorted_frame.with_columns(lag_expressions)
    valid_5s = (
        pl.col("seconds_elapsed") - pl.col("_selector_seconds_lag_1")
    ) == 5
    valid_15s = (
        ((pl.col("seconds_elapsed") - pl.col("_selector_seconds_lag_1")) == 5)
        & ((pl.col("seconds_elapsed") - pl.col("_selector_seconds_lag_2")) == 10)
        & ((pl.col("seconds_elapsed") - pl.col("_selector_seconds_lag_3")) == 15)
    )
    persistence_sum = pl.lit(1.0)
    confidence_sum = pl.col("confidence")
    for lag in (1, 2, 3):
        persistence_sum = persistence_sum + (
            pl.col("predicted_up") == pl.col(f"_selector_direction_lag_{lag}")
        ).cast(pl.Float64)
        confidence_sum = confidence_sum + pl.col(
            f"_selector_confidence_lag_{lag}"
        )
    output = with_lags.with_columns(
        pl.col("probability_up")
        .cast(pl.Float64)
        .alias("selector_base_probability_up"),
        pl.col("confidence")
        .cast(pl.Float64)
        .alias("selector_base_confidence"),
        pl.col("predicted_up")
        .cast(pl.Float64)
        .alias("selector_base_predicted_up"),
        (pl.col("seconds_elapsed").cast(pl.Float64) / 300.0).alias(
            "selector_seconds_elapsed_fraction"
        ),
        pl.when(valid_5s)
        .then(
            pl.col("probability_up")
            - pl.col("_selector_probability_lag_1")
        )
        .otherwise(None)
        .cast(pl.Float64)
        .alias("selector_probability_delta_5s"),
        pl.when(valid_15s)
        .then(
            pl.col("probability_up")
            - pl.col("_selector_probability_lag_3")
        )
        .otherwise(None)
        .cast(pl.Float64)
        .alias("selector_probability_delta_15s"),
        pl.when(valid_5s)
        .then(pl.col("confidence") - pl.col("_selector_confidence_lag_1"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("selector_confidence_delta_5s"),
        pl.when(valid_15s)
        .then(persistence_sum / 4.0)
        .otherwise(None)
        .cast(pl.Float64)
        .alias("selector_direction_persistence_15s"),
        pl.when(valid_15s)
        .then(confidence_sum / 4.0)
        .otherwise(None)
        .cast(pl.Float64)
        .alias("selector_confidence_mean_15s"),
    )
    temporary_columns = [
        f"_selector_{kind}_lag_{lag}"
        for lag in (1, 2, 3)
        for kind in ("seconds", "probability", "confidence", "direction")
    ]
    return output.drop(temporary_columns)


def split_prior_fold_chronologically(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    markets = (
        frame.select("market_id", "window_start")
        .unique()
        .sort(["window_start", "market_id"])
    )
    if markets.height < 2:
        raise ValueError("prior fold needs at least two markets for a 50/50 split")
    split_index = markets.height // 2
    calibration_markets = markets.head(split_index)
    policy_markets = markets.slice(split_index)
    calibration = frame.join(
        calibration_markets.select("market_id"),
        on="market_id",
        how="semi",
    )
    policy = frame.join(
        policy_markets.select("market_id"),
        on="market_id",
        how="semi",
    )
    if calibration["window_start"].max() >= policy["window_start"].min():
        raise RuntimeError("chronological prior-fold split overlaps")
    if calibration["market_id"].n_unique() + policy["market_id"].n_unique() != (
        frame["market_id"].n_unique()
    ):
        raise RuntimeError("chronological prior-fold split lost or duplicated markets")
    return (
        calibration,
        policy,
        {
            "method": "chronological_market_50_50",
            "source_markets": markets.height,
            "calibration_markets": calibration_markets.height,
            "policy_markets": policy_markets.height,
            "calibration_fraction": calibration_markets.height / markets.height,
            "policy_fraction": policy_markets.height / markets.height,
        },
    )


def fit_admission_selector(
    frame: pl.DataFrame,
    *,
    regularization_c: float,
    random_seed: int,
) -> FittedAdmissionSelector:
    if regularization_c <= 0.0:
        raise ValueError("selector regularization_c must be positive")
    labels = frame["correct"].cast(pl.Int8).to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("selector fitting rows must contain correct and incorrect labels")
    matrix = _feature_matrix(frame, SELECTOR_FEATURES)
    medians = _finite_medians(matrix)
    filled = np.where(np.isfinite(matrix), matrix, medians)
    weights = equal_market_row_weights(frame)
    means = np.average(filled, axis=0, weights=weights)
    variance = np.average((filled - means) ** 2, axis=0, weights=weights)
    scales = np.sqrt(np.maximum(variance, 0.0))
    scales = np.where(scales > 1e-12, scales, 1.0)
    transformed = (filled - means) / scales
    estimator = LogisticRegression(
        C=regularization_c,
        solver="lbfgs",
        max_iter=2_000,
        tol=1e-7,
        random_state=random_seed,
    )
    with threadpool_limits(limits=1):
        estimator.fit(transformed, labels, sample_weight=weights)
    if int(estimator.n_iter_[0]) >= estimator.max_iter:
        raise RuntimeError("correctness selector did not converge")
    return FittedAdmissionSelector(
        feature_names=SELECTOR_FEATURES,
        imputation_medians=medians,
        standardization_means=means,
        standardization_scales=scales,
        estimator=estimator,
        regularization_c=regularization_c,
    )


def fit_time_band_platt_calibrators(
    selector: FittedAdmissionSelector,
    calibration_rows: pl.DataFrame,
    config: AdmissionBenchmarkConfig,
) -> tuple[dict[str, ProbabilityCalibrator], list[dict[str, Any]]]:
    raw_logit = selector.raw_logit(calibration_rows)
    elapsed = calibration_rows["seconds_elapsed"].to_numpy()
    correct = calibration_rows["correct"].cast(pl.Int8).to_numpy()
    calibrators: dict[str, ProbabilityCalibrator] = {}
    diagnostics: list[dict[str, Any]] = []
    for band in config.bands:
        mask = (elapsed >= band.start_second) & (
            elapsed < band.end_second_exclusive
        )
        band_rows = calibration_rows.filter(pl.Series(mask))
        if band_rows.height < config.selector.minimum_calibration_rows_per_band:
            raise RuntimeError(f"selector calibration band {band.name} has too few rows")
        if (
            band_rows["market_id"].n_unique()
            < config.selector.minimum_calibration_markets_per_band
        ):
            raise RuntimeError(
                f"selector calibration band {band.name} has too few markets"
            )
        band_labels = correct[mask]
        if set(np.unique(band_labels)) != {0, 1}:
            raise RuntimeError(
                f"selector calibration band {band.name} needs both target classes"
            )
        weights = equal_market_row_weights(band_rows)
        estimator = LogisticRegression(
            C=1_000_000,
            solver="lbfgs",
            max_iter=500,
            tol=1e-9,
            random_state=config.selector.random_seed,
        )
        with threadpool_limits(limits=1):
            estimator.fit(
                raw_logit[mask].reshape(-1, 1),
                band_labels,
                sample_weight=weights,
            )
        calibrator = ProbabilityCalibrator(
            slope=float(estimator.coef_[0, 0]),
            intercept=float(estimator.intercept_[0]),
            converged=bool(estimator.n_iter_[0] < estimator.max_iter),
            iterations=int(estimator.n_iter_[0]),
        )
        if not calibrator.converged:
            raise RuntimeError(
                f"selector calibration band {band.name} did not converge"
            )
        calibrators[band.name] = calibrator
        diagnostics.append(
            {
                "name": band.name,
                "start_second": band.start_second,
                "end_second_exclusive": band.end_second_exclusive,
                "rows": band_rows.height,
                "markets": band_rows["market_id"].n_unique(),
                **asdict(calibrator),
            }
        )
    return calibrators, diagnostics


def score_selector_q(
    selector: FittedAdmissionSelector,
    calibrators: dict[str, ProbabilityCalibrator],
    frame: pl.DataFrame,
    config: AdmissionBenchmarkConfig,
) -> np.ndarray:
    if set(calibrators) != {band.name for band in config.bands}:
        raise ValueError("selector calibrators do not match the frozen time bands")
    raw_logit = selector.raw_logit(frame)
    elapsed = frame["seconds_elapsed"].to_numpy()
    q = np.full(frame.height, np.nan, dtype=np.float64)
    for band in config.bands:
        mask = (elapsed >= band.start_second) & (
            elapsed < band.end_second_exclusive
        )
        q[mask] = calibrators[band.name].probability(raw_logit[mask])
    if not np.isfinite(q).all() or np.any((q < 0.0) | (q > 1.0)):
        raise RuntimeError("selector calibration bands did not produce valid q")
    return q


def selector_probability_frame(
    base_rows: pl.DataFrame,
    selector_q: np.ndarray,
    *,
    selector_candidate: str,
    admission_floor: float,
) -> pl.DataFrame:
    if len(selector_q) != base_rows.height:
        raise ValueError("selector q length differs from the base probability rows")
    if not np.isfinite(selector_q).all() or np.any(
        (selector_q < 0.0) | (selector_q > 1.0)
    ):
        raise ValueError("selector q must be finite probabilities")
    if admission_floor != 0.5:
        raise ValueError("selector admission floor must remain 0.5")
    floor_for_direction = float(np.nextafter(admission_floor, 1.0))
    scored = base_rows.with_columns(
        pl.col("probability_up").alias("selector_base_probability_up_output"),
        pl.col("confidence").alias("selector_base_confidence_output"),
        pl.col("predicted_up").alias("selector_base_predicted_up_output"),
        pl.col("correct").alias("selector_base_correct_output"),
        pl.Series("selector_q", selector_q, dtype=pl.Float64),
    ).with_columns(
        (pl.col("selector_q") >= admission_floor).alias("model_eligible"),
        pl.max_horizontal(
            pl.col("selector_q"),
            pl.lit(floor_for_direction),
        ).alias("_selector_effective_q"),
    )
    scored = scored.with_columns(
        pl.when(pl.col("selector_base_predicted_up_output") == 1)
        .then(pl.col("_selector_effective_q"))
        .otherwise(1.0 - pl.col("_selector_effective_q"))
        .alias("probability_up"),
        pl.col("_selector_effective_q").alias("confidence"),
        pl.col("selector_base_predicted_up_output")
        .cast(pl.Int8)
        .alias("predicted_up"),
        pl.col("selector_base_correct_output").cast(pl.Boolean).alias("correct"),
        pl.lit(selector_candidate).alias("candidate"),
        pl.lit("base_direction_correct").alias("selector_target"),
    ).drop("_selector_effective_q")
    _validate_selector_direction_contract(scored)
    return scored


def equal_market_row_weights(frame: pl.DataFrame) -> np.ndarray:
    market_ids = frame["market_id"].cast(pl.String).to_numpy()
    _, inverse, counts = np.unique(
        market_ids,
        return_inverse=True,
        return_counts=True,
    )
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights *= len(weights) / weights.sum()
    return weights


def selector_model_summary(model: FittedAdmissionSelector) -> dict[str, Any]:
    return {
        "family": "logistic",
        "target": "base boundary direction is correct",
        "feature_names": list(model.feature_names),
        "regularization_c": model.regularization_c,
        "row_weight_policy": model.row_weight_policy,
        "coefficients": {
            feature: float(coefficient)
            for feature, coefficient in zip(
                model.feature_names,
                model.estimator.coef_[0],
                strict=True,
            )
        },
        "intercept": float(model.estimator.intercept_[0]),
        "iterations": int(model.estimator.n_iter_[0]),
        "converged": int(model.estimator.n_iter_[0]) < model.estimator.max_iter,
    }


def admission_advancement_checks(
    *,
    candidate_name: str,
    control_candidate: str,
    metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    fold_results: list[dict[str, Any]],
    comparison: dict[str, Any] | None,
    gates: AdmissionAdvancementGates,
    quantity: float,
    evidence_is_independent: bool,
) -> dict[str, Any]:
    if candidate_name == control_candidate:
        return {
            "is_control": True,
            "checks": [],
            "benchmark_passed": None,
            "development_qualified": None,
            "deployment_qualified": None,
        }
    checks = [
        _check(
            "minimum selected markets",
            metrics["markets"],
            ">=",
            gates.minimum_selected_markets,
        ),
        _check(
            "minimum accuracy",
            metrics["accuracy"],
            ">=",
            gates.minimum_accuracy,
        ),
        _check(
            "minimum balanced accuracy",
            metrics["balanced_accuracy"],
            ">=",
            gates.minimum_balanced_accuracy,
        ),
        _check(
            "minimum UP recall",
            metrics["up_recall"],
            ">=",
            gates.minimum_direction_recall,
        ),
        _check(
            "minimum DOWN recall",
            metrics["down_recall"],
            ">=",
            gates.minimum_direction_recall,
        ),
        _check(
            "minimum Wilson lower bound",
            metrics["wilson_lower_95"],
            ">=",
            gates.minimum_wilson_lower_95,
        ),
        _check(
            "maximum expected calibration error",
            metrics["expected_calibration_error"],
            "<=",
            gates.maximum_expected_calibration_error,
        ),
        _check(
            "minimum eligible-market coverage",
            metrics["coverage"],
            ">=",
            gates.minimum_coverage,
        ),
        _check(
            "coverage improves over control",
            metrics["coverage"] - control_metrics["coverage"],
            ">=",
            gates.minimum_coverage_uplift,
        ),
        _check(
            "accuracy does not regress from control",
            metrics["accuracy"] - control_metrics["accuracy"],
            ">=",
            -gates.maximum_accuracy_regression,
        ),
        _check(
            "balanced accuracy does not regress from control",
            metrics["balanced_accuracy"] - control_metrics["balanced_accuracy"],
            ">=",
            -gates.maximum_balanced_accuracy_regression,
        ),
        _check(
            "UP recall does not regress from control",
            metrics["up_recall"] - control_metrics["up_recall"],
            ">=",
            -gates.maximum_direction_recall_regression,
        ),
        _check(
            "DOWN recall does not regress from control",
            metrics["down_recall"] - control_metrics["down_recall"],
            ">=",
            -gates.maximum_direction_recall_regression,
        ),
        _check(
            "median entry is at least five seconds earlier",
            _difference(
                control_metrics["median_seconds_elapsed"],
                metrics["median_seconds_elapsed"],
            ),
            ">=",
            gates.minimum_median_entry_improvement_seconds,
        ),
        _check(
            "maximum median entry second",
            metrics["median_seconds_elapsed"],
            "<=",
            gates.maximum_median_entry_second,
        ),
    ]
    qualified_folds = sum(
        bool(record["validation"]["qualified"]) for record in fold_results
    )
    if gates.require_every_fold:
        checks.append(
            _check(
                "absolute quality gates pass every evaluation fold",
                qualified_folds,
                "==",
                len(fold_results),
            )
        )
    if comparison is None:
        raise RuntimeError(f"{candidate_name} lost its exact-time comparison")
    minimum_common = min(
        int(row["common_markets"]) for row in comparison["checkpoints"]
    )
    checks.append(
        _check(
            "minimum common exact-time checkpoint markets",
            minimum_common,
            ">=",
            gates.minimum_common_checkpoint_markets,
        )
    )
    execution = metrics["execution"]
    realized_per_share = (
        execution["realized_net_expectancy_per_trade"] / quantity
        if execution["realized_net_expectancy_per_trade"] is not None
        else None
    )
    checks.extend(
        (
            _check(
                "minimum executable economics markets",
                execution["economic_markets"],
                ">=",
                gates.minimum_executable_markets,
            ),
            _check(
                "positive mean direct edge per share",
                execution["mean_direct_edge_per_share"],
                ">",
                gates.minimum_mean_direct_edge_per_share,
            ),
            _check(
                "positive realized net expectancy per share",
                realized_per_share,
                ">",
                gates.minimum_realized_net_per_share,
            ),
        )
    )
    if candidate_name.endswith("correctness_admission"):
        reversals = sum(
            int(record["validation"]["direction_reversals"])
            for record in fold_results
        )
        below_floor_selected = sum(
            int(record["validation"]["q_below_floor_selected_rows"])
            for record in fold_results
        )
        checks.extend(
            (
                _check("selector direction reversals", reversals, "==", 0),
                _check(
                    "q below 0.5 selected rows",
                    below_floor_selected,
                    "==",
                    0,
                ),
            )
        )
    benchmark_passed = all(check["passed"] for check in checks)
    deployment_fold_count_passed = (
        len(fold_results) >= gates.required_deployment_validation_folds
    )
    return {
        "is_control": False,
        "checks": checks,
        "benchmark_passed": benchmark_passed,
        "development_qualified": benchmark_passed,
        "deployment_qualified": bool(
            benchmark_passed
            and deployment_fold_count_passed
            and evidence_is_independent
        ),
        "deployment_checks": [
            _check(
                "minimum selector validation folds for deployment",
                len(fold_results),
                ">=",
                gates.required_deployment_validation_folds,
            ),
            _check(
                "independent evidence required for deployment",
                int(evidence_is_independent),
                "==",
                1,
            ),
        ],
    }


def _benchmark_criteria(
    gates: AdmissionAdvancementGates,
) -> AdvancementCriteria:
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
            -gates.minimum_median_entry_improvement_seconds
        ),
        minimum_mean_direct_edge_per_share=(
            gates.minimum_mean_direct_edge_per_share
        ),
        minimum_realized_net_per_share=gates.minimum_realized_net_per_share,
        minimum_common_time_markets=gates.minimum_common_checkpoint_markets,
    )


def _validate_boundary_universes(
    boundary_folds: dict[int, pl.DataFrame],
    validation_universes: dict[int, set[str]],
    evaluation_folds: tuple[int, ...],
) -> None:
    for fold_index in evaluation_folds:
        boundary_markets = set(boundary_folds[fold_index]["market_id"].to_list())
        outside_control = boundary_markets - validation_universes[fold_index]
        if outside_control:
            raise RuntimeError(
                f"boundary fold {fold_index} escapes the control validation universe"
            )


def _validate_selector_direction_contract(frame: pl.DataFrame) -> None:
    required = {
        "selector_q",
        "selector_base_predicted_up_output",
        "predicted_up",
        "model_eligible",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "selector direction contract is missing columns: " + ", ".join(missing)
        )
    if frame.filter(
        pl.col("predicted_up") != pl.col("selector_base_predicted_up_output")
    ).height:
        raise RuntimeError("correctness selector reversed the base model direction")
    if frame.filter(
        (pl.col("selector_q") < 0.5) & pl.col("model_eligible")
    ).height:
        raise RuntimeError("correctness selector admitted q below 0.5")
    if (
        "policy_selected" in frame.columns
        and frame.filter(
            (pl.col("selector_q") < 0.5) & pl.col("policy_selected")
        ).height
    ):
        raise RuntimeError("correctness selector selected q below 0.5")


def _model_eligible_rows(frame: pl.DataFrame) -> pl.DataFrame:
    if "model_eligible" not in frame.columns:
        return frame
    eligible = frame.filter(pl.col("model_eligible"))
    if eligible.is_empty():
        raise RuntimeError("candidate has no model-eligible probability rows")
    return eligible


def _feature_matrix(
    frame: pl.DataFrame,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError("selector features are missing: " + ", ".join(missing))
    forbidden = {
        "label_up",
        "correct",
        "official_outcome",
        "final_price",
        "window_end",
    }
    if forbidden.intersection(feature_names):
        raise RuntimeError("selector feature allowlist contains target or audit data")
    return frame.select(feature_names).to_numpy().astype(np.float64)


def _finite_medians(matrix: np.ndarray) -> np.ndarray:
    output = np.zeros(matrix.shape[1], dtype=np.float64)
    for index in range(matrix.shape[1]):
        finite = matrix[np.isfinite(matrix[:, index]), index]
        output[index] = float(np.median(finite)) if finite.size else 0.0
    return output


def _selection_payload(selection: TimeBandPolicySelection) -> dict[str, Any]:
    return {
        **asdict(selection),
        "thresholds": dict(selection.thresholds),
    }


def _frame_range(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "start": frame["window_start"].min().isoformat(),
        "end": frame["window_start"].max().isoformat(),
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
    }


def _execution_config(
    config: AdmissionBenchmarkConfig,
) -> ExecutionEvidenceConfig:
    manifest = json.loads(
        (config.execution_evidence / "manifest.json").read_text()
    )
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


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _check(
    name: str,
    observed: float | None,
    operator: str,
    required: float,
) -> dict[str, Any]:
    if observed is None:
        passed = False
    elif operator == ">=":
        passed = observed >= required
    elif operator == ">":
        passed = observed > required
    elif operator == "<=":
        passed = observed <= required
    elif operator == "==":
        passed = observed == required
    else:
        raise ValueError(f"unsupported admission gate operator: {operator}")
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "required": required,
        "passed": bool(passed),
    }


def _write_parquet_atomic(frame: pl.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    frame.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )
    os.replace(temporary, destination)
