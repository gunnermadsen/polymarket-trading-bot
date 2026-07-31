from __future__ import annotations

import html
import json
import math
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
from .core_evaluation import (
    classification_metrics,
    paired_uplift,
    scored_prediction_rows,
)
from .core_execution import ExecutionEvidenceConfig
from .core_extract import file_sha256, write_json_atomic
from .core_features import (
    CORE_MATURE_REVERSAL_ENRICHED_FEATURES,
    CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
    load_core_feature_frame,
    validate_core_feature_cache,
)
from .core_training import (
    CORE_FREEZE_SCHEMA_VERSION,
    MARKET_EQUAL_ROW_WEIGHT_POLICY,
    TRAINING_MODEL_FILENAME,
    CandidateSpec,
    FrozenTrainingBundle,
    chronological_subsplit,
    configure_native_thread_limits,
    fit_probability_calibrator,
    model_candidate_spec,
    model_summary_payload,
    range_frame,
    row_weight_schedule_payload,
    tune_and_fit_model,
)
from .fixed_time_config import (
    FIXED_TIME_CANDIDATE,
    FixedTimeAccuracyConfig,
    FixedTimeOperatingPointConfig,
    load_fixed_time_accuracy_config,
)
from .paper_candidate import write_golden_feature_sample
from .persistence_benchmark import (
    attach_execution_evidence,
    hard_confident_error_metrics,
    load_execution_evidence,
)
from .provenance import runtime_provenance
from .runtime_export import export_runtime_model

FIXED_TIME_BENCHMARK_SCHEMA_VERSION = "btc-mature-reversal-fixed-time-benchmark-v1"
FIXED_TIME_FREEZE_SCHEMA_VERSION = "btc-mature-reversal-fixed-time-freeze-v1"
FIXED_TIME_PAPER_AUTHORIZATION = "explicit-fixed-120-paper-only-forward-evaluation"
PREDECESSOR_CANDIDATE = "histogram_mature_reversal_recency_28d_oracle_control"
SCORED_PROBABILITY_SUFFIX = "-scored-probabilities.parquet"
GOLDEN_FEATURES_FILENAME = "golden-features.parquet"
ENTRY_PRICE_BANDS = (
    ("under_0_35", 0.00, 0.35),
    ("0_35_to_0_50", 0.35, 0.50),
    ("0_50_to_0_65", 0.50, 0.65),
    ("0_65_to_0_80", 0.65, 0.80),
    ("0_80_to_1_00", 0.80, 1.01),
)


def fixed_time_candidate_spec(config: FixedTimeAccuracyConfig) -> CandidateSpec:
    feature_names = tuple(config.model.feature_names)
    if feature_names != tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES):
        raise RuntimeError("fixed-time training feature allowlist changed")
    forbidden = [
        name
        for name in feature_names
        if name.startswith("oracle_")
        or "vwap" in name
        or "orderbook" in name
        or "best_bid" in name
        or "best_ask" in name
        or "missing" in name
        or "eligible" in name
    ]
    if forbidden:
        raise RuntimeError(
            "fixed-time training contains forbidden oracle, book, or routing features: "
            + ", ".join(forbidden)
        )
    return CandidateSpec(
        name=FIXED_TIME_CANDIDATE,
        family="histogram",
        feature_names=feature_names,
        row_weight_policy=MARKET_EQUAL_ROW_WEIGHT_POLICY,
        recency_half_life_days=config.model.recency_half_life_days,
    )


def fixed_time_rows(
    frame: pl.DataFrame,
    *,
    decision_second: int,
    cohort_name: str,
) -> pl.DataFrame:
    selected = frame.filter(pl.col("seconds_elapsed") == decision_second).sort(
        ["window_start", "market_id", "observed_at"]
    )
    if selected.is_empty():
        raise RuntimeError(f"{cohort_name} has no fixed-time rows")
    if selected["market_id"].n_unique() != selected.height:
        raise RuntimeError(f"{cohort_name} must contain exactly one row per market")
    if selected["seconds_elapsed"].unique().to_list() != [decision_second]:
        raise RuntimeError(f"{cohort_name} contains a non-fixed decision second")
    return selected


def estimator_training_rows(
    frame: pl.DataFrame,
    *,
    training_seconds: tuple[int, ...],
    cohort_name: str,
) -> pl.DataFrame:
    selected = frame.filter(pl.col("seconds_elapsed").is_in(training_seconds)).sort(
        ["window_start", "market_id", "seconds_elapsed", "observed_at"]
    )
    if selected.is_empty():
        raise RuntimeError(f"{cohort_name} has no estimator-training rows")
    expected = set(training_seconds)
    invalid = (
        selected.group_by("market_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("seconds_elapsed").n_unique().alias("unique_seconds"),
            pl.col("seconds_elapsed").unique().alias("seconds"),
        )
        .filter(
            (pl.col("rows") != len(training_seconds))
            | (pl.col("unique_seconds") != len(training_seconds))
        )
    )
    if invalid.height:
        raise RuntimeError(f"{cohort_name} must contain every estimator-training second per market")
    observed = set(selected["seconds_elapsed"].unique().to_list())
    if observed != expected:
        raise RuntimeError(f"{cohort_name} estimator-training seconds changed")
    return selected


def empirical_coverage_threshold(
    scored: pl.DataFrame,
    *,
    target_coverage: float,
) -> tuple[float, pl.DataFrame, dict[str, Any]]:
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target coverage must be between zero and one")
    if scored.is_empty():
        raise ValueError("cannot select a threshold from an empty policy cohort")
    if scored["market_id"].n_unique() != scored.height:
        raise ValueError("empirical threshold selection requires one row per market")
    if scored["confidence"].is_null().any() or not scored["confidence"].is_finite().all():
        raise ValueError("policy confidence contains missing or non-finite values")

    target_markets = max(1, math.ceil(scored.height * target_coverage))
    ordered = scored.sort(
        ["confidence", "observed_at", "market_id"],
        descending=[True, False, False],
    )
    threshold = float(ordered[target_markets - 1, "confidence"])
    selected = scored.filter(pl.col("confidence") >= threshold).sort(["observed_at", "market_id"])
    return (
        threshold,
        selected,
        {
            "method": "empirical_policy_confidence_quantile",
            "labels_used_for_threshold_selection": False,
            "eligible_markets": scored.height,
            "target_coverage": target_coverage,
            "target_markets": target_markets,
            "confidence_threshold": threshold,
            "selected_markets": selected.height,
            "realized_coverage": selected.height / scored.height,
            "tie_expansion_markets": selected.height - target_markets,
        },
    )


def operating_point_checks(
    metrics: dict[str, Any],
    contract: FixedTimeOperatingPointConfig,
) -> list[dict[str, Any]]:
    minimum_coverage = contract.target_coverage - contract.coverage_tolerance
    maximum_coverage = contract.target_coverage + contract.coverage_tolerance
    requirements = (
        ("minimum coverage", metrics["coverage"], minimum_coverage, ">="),
        ("maximum coverage", metrics["coverage"], maximum_coverage, "<="),
        ("accuracy", metrics["accuracy"], contract.minimum_accuracy, ">="),
        (
            "balanced accuracy",
            metrics["balanced_accuracy"],
            contract.minimum_balanced_accuracy,
            ">=",
        ),
        ("UP recall", metrics["up_recall"], contract.minimum_direction_recall, ">="),
        (
            "DOWN recall",
            metrics["down_recall"],
            contract.minimum_direction_recall,
            ">=",
        ),
        (
            "Wilson lower 95%",
            metrics["wilson_lower_95"],
            contract.minimum_wilson_lower_95,
            ">=",
        ),
        (
            "expected calibration error",
            metrics["expected_calibration_error"],
            contract.maximum_expected_calibration_error,
            "<=",
        ),
    )
    return [
        {
            "name": name,
            "observed": observed,
            "operator": operator,
            "required": required,
            "passed": bool(observed >= required if operator == ">=" else observed <= required),
        }
        for name, observed, required, operator in requirements
    ]


def run_fixed_time_accuracy_benchmark(
    config: FixedTimeAccuracyConfig,
) -> tuple[Path, dict[str, Any]]:
    core_config = load_core_config(config.benchmark.core_config)
    configure_native_thread_limits(core_config)
    feature_metadata = validate_core_feature_cache(core_config, "pre_holdout")
    _validate_feature_metadata(feature_metadata)
    spec = fixed_time_candidate_spec(config)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    fold_results = _run_fold_matrix(config, core_config)
    scored = pl.concat(
        [result.pop("scored_rows") for result in fold_results],
        how="vertical_relaxed",
    ).sort(["fold_index", "observed_at", "market_id"])

    execution_config = _execution_config(config)
    execution = load_execution_evidence(execution_config)
    scored_with_execution = attach_execution_evidence(scored, execution)
    primary_rows = scored_with_execution.filter(pl.col("primary_selected"))
    secondary_rows = scored_with_execution.filter(pl.col("secondary_selected"))
    eligible_markets = scored.height
    primary = _operating_point_payload(
        primary_rows,
        eligible_markets=eligible_markets,
        contract=config.primary,
        hard_confidence_floor=config.model.hard_confidence_floor,
    )
    secondary = _operating_point_payload(
        secondary_rows,
        eligible_markets=eligible_markets,
        contract=config.secondary,
        hard_confidence_floor=config.model.hard_confidence_floor,
    )
    predecessor = _predecessor_payload(config, execution)
    hard_error_checks = _hard_error_nonregression_checks(
        primary["hard_confident_errors"],
        predecessor["hard_confident_errors"],
    )
    economics_checks = _economics_checks(primary["execution_by_size"]["ten_share_vwap10"])
    primary["checks"].extend(hard_error_checks)
    primary["checks"].extend(economics_checks)
    primary["qualified"] = all(check["passed"] for check in primary["checks"])

    benchmark: dict[str, Any] = {
        "schema_version": FIXED_TIME_BENCHMARK_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "qualified_for_paper_freeze" if primary["qualified"] else "not_qualified",
        "evaluation": {
            "kind": "development",
            "independent": config.benchmark.evaluation_is_independent,
            "note": config.benchmark.evaluation_note,
        },
        "configuration": _config_payload(config),
        "model_contract": {
            "candidate": spec.name,
            "family": spec.family,
            "target": config.model.target,
            "probability_calibration": config.model.probability_calibration,
            "decision_second": config.model.decision_second,
            "estimator_training_seconds": list(config.model.estimator_training_seconds),
            "calibration_policy_validation_seconds": [config.model.decision_second],
            "feature_schema_version": config.model.feature_schema_version,
            "feature_count": len(spec.feature_names),
            "feature_names": list(spec.feature_names),
            "recency_half_life_days": spec.recency_half_life_days,
            "row_weight_policy": spec.row_weight_policy,
            "oracle_feature_count": 0,
            "book_feature_count": 0,
            "missingness_or_routing_feature_count": 0,
        },
        "data_contract": {
            "range_start": core_config.data.range_start.isoformat(),
            "range_end_exclusive": core_config.data.range_end.isoformat(),
            "feature_cache_source_contract": core_config.data.source_contract,
            "model_input_tables": [
                "polymarket.btc_interval_markets",
                "polymarket.btc_market_reference_facts",
                "polymarket.binance_one_second_klines",
            ],
            "oracle_role": "excluded_from_model_inputs",
            "orderbook_role": "post_prediction_execution_economics_only",
            "feature_metadata": feature_metadata,
        },
        "walk_forward": {
            "folds": fold_results,
            "eligible_markets": eligible_markets,
            "one_row_per_market": scored["market_id"].n_unique() == scored.height,
            "decision_seconds": scored["seconds_elapsed"].unique().to_list(),
        },
        "operating_points": {
            "primary": primary,
            "secondary": secondary,
        },
        "predecessor": predecessor,
        "selection": {
            "paper_candidate_qualified": primary["qualified"],
            "primary_is_deployment_selector": True,
            "secondary_is_diagnostic": True,
            "runtime_export_authorized": primary["qualified"],
            "paper_process_authorized": primary["qualified"],
            "live_capital_authorized": False,
            "forward_paper_evidence_required": True,
        },
        "provenance": {
            "benchmark_config": str(config.source_path),
            "benchmark_config_sha256": file_sha256(config.source_path),
            "core_config": str(config.benchmark.core_config),
            "core_config_sha256": file_sha256(config.benchmark.core_config),
            "development_feature": str(core_config.paths.development_feature_data),
            "development_feature_sha256": file_sha256(core_config.paths.development_feature_data),
            "execution_manifest_sha256": config.paths.execution_manifest_sha256,
            "predecessor_benchmark_sha256": (config.paths.predecessor_benchmark_sha256),
        },
    }
    scored.write_parquet(
        run_dir / f"{FIXED_TIME_CANDIDATE}{SCORED_PROBABILITY_SUFFIX}",
        compression="zstd",
    )
    write_json_atomic(run_dir / "benchmark.json", benchmark)
    _write_text_atomic(run_dir / "report.html", _render_report(benchmark))
    return run_dir, benchmark


def _run_fold_matrix(
    config: FixedTimeAccuracyConfig,
    core_config: CoreTrainingConfig,
) -> list[dict[str, Any]]:
    max_workers = min(
        core_config.compute.max_parallel_fits,
        len(config.split.validation_windows),
    )
    completed: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_fixed_time_fold_task,
                config.source_path,
                fold_index,
            ): fold_index
            for fold_index in range(len(config.split.validation_windows))
        }
        for future in as_completed(futures):
            fold_index = futures[future]
            result = future.result()
            completed[fold_index] = result
            primary = result["primary"]["metrics"]
            secondary = result["secondary"]["metrics"]
            print(
                "fixed-120 fold "
                f"{fold_index + 1}/{len(config.split.validation_windows)}: "
                f"primary accuracy={primary['accuracy']:.4f} "
                f"coverage={primary['coverage']:.4f}; "
                f"secondary accuracy={secondary['accuracy']:.4f} "
                f"coverage={secondary['coverage']:.4f}",
                flush=True,
            )
    return [completed[index] for index in sorted(completed)]


def _evaluate_fixed_time_fold_task(
    config_path: Path,
    fold_index: int,
) -> dict[str, Any]:
    config = load_fixed_time_accuracy_config(config_path)
    core_config = load_core_config(config.benchmark.core_config)
    configure_native_thread_limits(core_config)
    with threadpool_limits(limits=core_config.compute.threads_per_fit):
        frame = estimator_training_rows(
            load_core_feature_frame(core_config, "pre_holdout"),
            training_seconds=config.model.estimator_training_seconds,
            cohort_name="fixed-time estimator feature cache",
        )
        return _evaluate_fixed_time_fold(
            frame,
            fold_index=fold_index,
            config=config,
            core_config=core_config,
        )


def _evaluate_fixed_time_fold(
    frame: pl.DataFrame,
    *,
    fold_index: int,
    config: FixedTimeAccuracyConfig,
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
    estimator_training_rows(
        fit,
        training_seconds=config.model.estimator_training_seconds,
        cohort_name=f"fold {fold_index} estimator fit",
    )
    calibration = fixed_time_rows(
        calibration_context,
        decision_second=config.model.decision_second,
        cohort_name=f"fold {fold_index} calibration",
    )
    policy = fixed_time_rows(
        policy_context,
        decision_second=config.model.decision_second,
        cohort_name=f"fold {fold_index} policy",
    )
    validation = fixed_time_rows(
        validation_context,
        decision_second=config.model.decision_second,
        cohort_name=f"fold {fold_index} validation",
    )

    started = time.perf_counter()
    spec = fixed_time_candidate_spec(config)
    model, tuning = tune_and_fit_model(fit, spec, core_config)
    calibrator = fit_probability_calibrator(
        model,
        calibration,
        core_config,
        spec,
    )
    if not calibrator.converged or calibrator.slope <= 0.0:
        raise RuntimeError(f"fold {fold_index} probability calibration failed")

    policy_probability = calibrator.probability(model.raw_logit(policy))
    policy_scored = scored_prediction_rows(policy, policy_probability)
    primary_threshold, primary_policy, primary_selection = empirical_coverage_threshold(
        policy_scored,
        target_coverage=config.primary.target_coverage,
    )
    secondary_threshold, secondary_policy, secondary_selection = empirical_coverage_threshold(
        policy_scored,
        target_coverage=config.secondary.target_coverage,
    )
    probability = calibrator.probability(model.raw_logit(validation))
    scored = scored_prediction_rows(validation, probability).with_columns(
        pl.lit(FIXED_TIME_CANDIDATE).alias("candidate"),
        pl.lit(fold_index).cast(pl.Int32).alias("fold_index"),
        pl.lit(primary_threshold).alias("primary_confidence_threshold"),
        pl.lit(secondary_threshold).alias("secondary_confidence_threshold"),
        (pl.col("confidence") >= primary_threshold).alias("primary_selected"),
        (pl.col("confidence") >= secondary_threshold).alias("secondary_selected"),
    )
    primary_selected = scored.filter(pl.col("primary_selected"))
    secondary_selected = scored.filter(pl.col("secondary_selected"))
    eligible_markets = validation.height
    primary_metrics = classification_metrics(
        primary_selected,
        eligible_markets=eligible_markets,
    )
    secondary_metrics = classification_metrics(
        secondary_selected,
        eligible_markets=eligible_markets,
    )
    return {
        "fold_index": fold_index,
        "fit": {
            **_cohort_payload(fit),
            "role": "estimator_training",
            "expected_seconds": list(config.model.estimator_training_seconds),
        },
        "calibration": _cohort_payload(calibration),
        "policy": _cohort_payload(policy),
        "validation": _cohort_payload(validation),
        "tuning": tuning,
        "calibrator": asdict(calibrator),
        "primary": {
            "threshold_selection": primary_selection,
            "policy_metrics_after_selection": classification_metrics(
                primary_policy,
                eligible_markets=policy.height,
            ),
            "metrics": primary_metrics,
            "checks": operating_point_checks(primary_metrics, config.primary),
        },
        "secondary": {
            "threshold_selection": secondary_selection,
            "policy_metrics_after_selection": classification_metrics(
                secondary_policy,
                eligible_markets=policy.height,
            ),
            "metrics": secondary_metrics,
            "checks": operating_point_checks(secondary_metrics, config.secondary),
        },
        "threshold_selection_used_validation_labels": False,
        "elapsed_seconds": time.perf_counter() - started,
        "scored_rows": scored,
    }


def _operating_point_payload(
    selected: pl.DataFrame,
    *,
    eligible_markets: int,
    contract: FixedTimeOperatingPointConfig,
    hard_confidence_floor: float,
) -> dict[str, Any]:
    metrics = classification_metrics(selected, eligible_markets=eligible_markets)
    checks = operating_point_checks(metrics, contract)
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
    return {
        "contract": asdict(contract),
        "metrics": metrics,
        "baseline": paired_uplift(selected),
        "hard_confident_errors": hard_confident_error_metrics(
            selected,
            eligible_markets=eligible_markets,
            confidence_floor=hard_confidence_floor,
        ),
        "execution_by_size": execution_by_size,
        "ten_share_entry_price_bands": _entry_price_band_metrics(selected),
        "checks": checks,
        "qualified": all(check["passed"] for check in checks),
    }


def _predecessor_payload(
    config: FixedTimeAccuracyConfig,
    execution: pl.DataFrame,
) -> dict[str, Any]:
    benchmark = json.loads(config.paths.predecessor_benchmark.read_text())
    if PREDECESSOR_CANDIDATE not in benchmark.get("candidates", {}):
        raise RuntimeError("predecessor benchmark lost the mature-reversal control")
    scored_path = (
        config.paths.predecessor_benchmark.parent
        / f"{PREDECESSOR_CANDIDATE}{SCORED_PROBABILITY_SUFFIX}"
    )
    if not scored_path.is_file():
        raise RuntimeError("predecessor scored-probability evidence is missing")
    raw = pl.read_parquet(scored_path).filter(
        pl.col("seconds_elapsed") == config.model.decision_second
    )
    if raw["market_id"].n_unique() != raw.height:
        raise RuntimeError("predecessor fixed-120 evidence is not one row per market")
    selected = attach_execution_evidence(
        raw.filter(pl.col("policy_selected")),
        execution,
    )
    eligible = raw.height
    return {
        "candidate": PREDECESSOR_CANDIDATE,
        "benchmark": str(config.paths.predecessor_benchmark),
        "benchmark_sha256": file_sha256(config.paths.predecessor_benchmark),
        "scored_probability_sha256": file_sha256(scored_path),
        "metrics": classification_metrics(selected, eligible_markets=eligible),
        "hard_confident_errors": hard_confident_error_metrics(
            selected,
            eligible_markets=eligible,
            confidence_floor=config.model.hard_confidence_floor,
        ),
        "execution_by_size": {
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
        },
    }


def _hard_error_nonregression_checks(
    candidate: dict[str, Any],
    predecessor: dict[str, Any],
) -> list[dict[str, Any]]:
    requirements = (
        (
            "hard-confident error exposure rate does not regress",
            candidate["hard_confident_error_exposure_rate"],
            predecessor["hard_confident_error_exposure_rate"],
        ),
        (
            "hard-confident selected-error rate does not regress",
            candidate["hard_confident_error_rate_selected"],
            predecessor["hard_confident_error_rate_selected"],
        ),
    )
    return [
        {
            "name": name,
            "observed": observed,
            "operator": "<=",
            "required": required,
            "passed": bool(observed <= required + 1e-12),
        }
        for name, observed, required in requirements
    ]


def _economics_checks(ten_share: dict[str, Any]) -> list[dict[str, Any]]:
    expectancy = ten_share["realized_net_expectancy_per_trade"]
    total = ten_share["realized_net_pnl_total"]
    requirements = (
        ("ten-share execution economics are available", ten_share["economics_available"]),
        (
            "ten-share realized net expectancy is positive",
            expectancy is not None and expectancy > 0.0,
        ),
        (
            "ten-share realized net PnL is positive",
            total is not None and total > 0.0,
        ),
    )
    return [
        {
            "name": name,
            "observed": passed,
            "operator": "==",
            "required": True,
            "passed": bool(passed),
        }
        for name, passed in requirements
    ]


def _entry_price_band_metrics(selected: pl.DataFrame) -> dict[str, Any]:
    price = (
        pl.when(pl.col("predicted_up") == 1)
        .then(pl.col("up_ask_vwap_10"))
        .otherwise(pl.col("down_ask_vwap_10"))
        .alias("_selected_price")
    )
    executable = selected.with_columns(price).filter(
        pl.col("strict_both_side_eligible_10").fill_null(False)
        & pl.col("_selected_price").is_not_null()
        & pl.col("_selected_price").is_finite()
    )
    output: dict[str, Any] = {}
    for name, lower, upper in ENTRY_PRICE_BANDS:
        rows = executable.filter(
            pl.col("_selected_price").is_between(
                lower,
                upper,
                closed="left",
            )
        )
        output[name] = {
            "minimum_price_inclusive": lower,
            "maximum_price_exclusive": upper,
            "metrics": classification_metrics(rows),
            "ten_share_execution": _execution_metrics(
                rows,
                quantity=10.0,
                vwap_depth=10,
            ),
        }
    return output


def _validate_feature_metadata(metadata: dict[str, Any]) -> None:
    candidate_schemas = metadata.get("candidate_feature_schema_versions", {})
    if candidate_schemas.get(FIXED_TIME_CANDIDATE) != CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION:
        raise RuntimeError("feature cache lost the mature-reversal schema")
    core_markets = int(metadata.get("core_complete_candidate_markets", -1))
    oracle = metadata.get("oracle", {})
    oracle_markets = int(oracle.get("complete_candidate_markets", -2))
    oracle_incomplete = int(oracle.get("incomplete_candidate_markets", -1))
    unmatched = int(oracle.get("unmatched_core_rows", -1))
    if (
        core_markets <= 0
        or oracle_markets != core_markets
        or oracle_incomplete != 0
        or unmatched != 0
    ):
        raise RuntimeError(
            "oracle-source cache provenance would change the core-only market universe"
        )


def _execution_config(config: FixedTimeAccuracyConfig) -> ExecutionEvidenceConfig:
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


def _cohort_payload(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "minimum_window_start": frame["window_start"].min().isoformat(),
        "maximum_window_start": frame["window_start"].max().isoformat(),
        "decision_seconds": frame["seconds_elapsed"].unique().to_list(),
        "one_row_per_market": frame["market_id"].n_unique() == frame.height,
    }


def _config_payload(config: FixedTimeAccuracyConfig) -> dict[str, Any]:
    return _json_value(asdict(config))


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _write_text_atomic(path: Path, contents: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents)
    temporary.replace(path)


def _render_report(benchmark: dict[str, Any]) -> str:
    primary = benchmark["operating_points"]["primary"]
    secondary = benchmark["operating_points"]["secondary"]
    predecessor = benchmark["predecessor"]
    rows = []
    for name, payload in (
        ("Predecessor fixed-120 subset", predecessor),
        ("New primary 15% objective", primary),
        ("New secondary 10% objective", secondary),
    ):
        metrics = payload["metrics"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{metrics['markets']:,}</td>"
            f"<td>{metrics['coverage']:.2%}</td>"
            f"<td>{metrics['accuracy']:.2%}</td>"
            f"<td>{metrics['balanced_accuracy']:.2%}</td>"
            f"<td>{metrics['up_recall']:.2%}</td>"
            f"<td>{metrics['down_recall']:.2%}</td>"
            f"<td>{metrics['wilson_lower_95']:.2%}</td>"
            f"<td>{metrics['expected_calibration_error']:.2%}</td>"
            "</tr>"
        )
    checks = "".join(
        "<li class='pass'>PASS</li>"
        if check["passed"]
        else ("<li class='fail'>FAIL — " + html.escape(check["name"]) + "</li>")
        for check in primary["checks"]
    )
    status = (
        "Qualified for paper-only freeze"
        if benchmark["selection"]["paper_candidate_qualified"]
        else "Not qualified for paper-only freeze"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC fixed-120 mature-reversal benchmark</title>
<style>
body {{ font: 15px system-ui; margin: 2rem; color: #17202a; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d5d8dc; padding: .5rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.pass {{ color: #117864; }} .fail {{ color: #b03a2e; }}
</style>
</head>
<body>
<h1>BTC mature-reversal fixed-120 benchmark</h1>
<p><strong>{html.escape(status)}</strong></p>
<p>Model inputs: 71 causal core features. Oracle and order-book fields: excluded.</p>
<table>
<thead><tr><th>Operating point</th><th>Markets</th><th>Coverage</th>
<th>Accuracy</th><th>Balanced</th><th>UP recall</th><th>DOWN recall</th>
<th>Wilson lower</th><th>ECE</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
<h2>Primary qualification checks</h2><ul>{checks}</ul>
</body>
</html>
"""


def freeze_and_export_fixed_time_paper_candidate(
    *,
    config: FixedTimeAccuracyConfig,
    benchmark_run: Path,
    model_key: str,
    authorization: str,
) -> tuple[Path, Path, dict[str, Any]]:
    if authorization != FIXED_TIME_PAPER_AUTHORIZATION:
        raise RuntimeError("fixed-time export requires explicit paper-only authorization")
    benchmark_path = benchmark_run.resolve() / "benchmark.json"
    if not benchmark_path.is_file():
        raise RuntimeError("fixed-time benchmark evidence is missing")
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark.get("schema_version") != FIXED_TIME_BENCHMARK_SCHEMA_VERSION:
        raise RuntimeError("fixed-time benchmark schema changed")
    if benchmark.get("configuration") != _config_payload(config):
        raise RuntimeError("fixed-time benchmark does not match the exact configuration")
    if not benchmark.get("selection", {}).get("paper_candidate_qualified"):
        raise RuntimeError("fixed-time benchmark did not qualify for paper export")

    core_config = load_core_config(config.benchmark.core_config)
    feature_metadata = validate_core_feature_cache(core_config, "pre_holdout")
    _validate_feature_metadata(feature_metadata)
    frame = estimator_training_rows(
        load_core_feature_frame(core_config, "pre_holdout"),
        training_seconds=config.model.estimator_training_seconds,
        cohort_name="final fixed-time estimator feature cache",
    )
    development = range_frame(
        frame,
        config.split.development_start,
        config.split.development_end,
    )
    estimator_training_rows(
        development,
        training_seconds=config.model.estimator_training_seconds,
        cohort_name="final estimator development",
    )
    calibration = fixed_time_rows(
        range_frame(
            frame,
            config.split.probability_calibration_start,
            config.split.probability_calibration_end,
        ),
        decision_second=config.model.decision_second,
        cohort_name="final probability calibration",
    )
    policy = fixed_time_rows(
        range_frame(
            frame,
            config.split.policy_selection_start,
            config.split.policy_selection_end,
        ),
        decision_second=config.model.decision_second,
        cohort_name="final policy selection",
    )
    spec = fixed_time_candidate_spec(config)
    model, tuning = tune_and_fit_model(development, spec, core_config)
    calibrator = fit_probability_calibrator(
        model,
        calibration,
        core_config,
        spec,
    )
    if not calibrator.converged or calibrator.slope <= 0.0:
        raise RuntimeError("final fixed-time calibrator failed")
    policy_probability = calibrator.probability(model.raw_logit(policy))
    policy_scored = scored_prediction_rows(policy, policy_probability)
    threshold, selected, threshold_selection = empirical_coverage_threshold(
        policy_scored,
        target_coverage=config.primary.target_coverage,
    )
    policy_metrics = classification_metrics(
        selected,
        eligible_markets=policy.height,
    )
    policy_checks = operating_point_checks(policy_metrics, config.primary)
    if not all(check["passed"] for check in policy_checks):
        raise RuntimeError("final fixed-time policy cohort did not pass primary gates")

    bundle = FrozenTrainingBundle(
        model=model,
        calibrator=calibrator,
        confidence_threshold=threshold,
    )
    created_at = datetime.now(UTC)
    freeze_id = f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{FIXED_TIME_CANDIDATE}-fixed-120-paper"
    freeze_dir = config.paths.freezes / freeze_id
    freeze_dir.mkdir(parents=True, exist_ok=False)
    model_path = freeze_dir / TRAINING_MODEL_FILENAME
    joblib.dump(bundle, model_path, compress=3)
    summary_path = freeze_dir / "model-summary.json"
    summary = model_summary_payload(bundle)
    summary.update(
        {
            "fixed_time_freeze_schema_version": FIXED_TIME_FREEZE_SCHEMA_VERSION,
            "deployment_scope": "paper_only",
            "production_qualified": False,
            "live_capital_allowed": False,
        }
    )
    write_json_atomic(summary_path, summary)
    golden_path = freeze_dir / GOLDEN_FEATURES_FILENAME
    write_golden_feature_sample(
        policy,
        policy_probability,
        bundle,
        golden_path,
    )

    provenance = runtime_provenance(config.package_root)
    frozen_spec = model_candidate_spec(bundle.model)
    feature_path = core_config.paths.development_feature_data
    feature_metadata_path = feature_path.with_suffix(".metadata.json")
    manifest: dict[str, Any] = {
        "schema_version": CORE_FREEZE_SCHEMA_VERSION,
        "fixed_time_freeze_schema_version": FIXED_TIME_FREEZE_SCHEMA_VERSION,
        "freeze_id": freeze_id,
        "created_at": created_at.isoformat(),
        "status": "paper_candidate_frozen",
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
        "feature_schema_version": CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
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
        "core_configuration_sha256": file_sha256(core_config.source_path),
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
            **threshold_selection,
            "metrics": policy_metrics,
            "checks": policy_checks,
        },
        "benchmark_evidence": {
            "run": str(benchmark_run.resolve()),
            "benchmark_sha256": file_sha256(benchmark_path),
            "paper_candidate_qualified": True,
        },
        "tuning": tuning,
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
    return freeze_dir, runtime_dir, manifest


def _range_payload(start: datetime, end: datetime) -> dict[str, str]:
    return {"start": start.isoformat(), "end": end.isoformat()}
