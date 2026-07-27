from __future__ import annotations

import math
import time
from dataclasses import asdict, replace
from typing import Any

import polars as pl

from .benchmark_config import EntryBenchmarkConfig
from .core_config import CoreTrainingConfig
from .core_evaluation import (
    baseline_metrics,
    block_bootstrap_uplift,
    choose_threshold,
    classification_metrics,
    first_crossing_timing,
    first_prediction_rows,
    paired_uplift,
    scored_prediction_rows,
    threshold_table,
)
from .core_features import CORE_ENRICHED_FEATURES
from .core_training import (
    EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY,
    CandidateSpec,
    FrozenTrainingBundle,
    combined_fixed_time_rows,
    combined_fold_rows,
    evaluate_fold,
    fit_probability_calibrator,
    range_frame,
    tune_and_fit_model,
)
from .preopen_features import PREOPEN_MODEL_FEATURES

PREOPEN_CANDIDATE = "histogram_preopen_early_weighted"
STRICT_BOOK_FEATURES = [
    "book_up_mid",
    "book_down_mid",
    "book_up_spread",
    "book_down_spread",
    "book_up_ask_vwap_5",
    "book_down_ask_vwap_5",
    "book_up_vwap_slippage_5",
    "book_down_vwap_slippage_5",
    "book_up_imbalance",
    "book_down_imbalance",
    "book_up_log_bid_depth",
    "book_down_log_bid_depth",
    "book_up_log_ask_depth",
    "book_down_log_ask_depth",
    "book_mid_complement_residual",
    "book_ask_complement_residual",
    "book_mid_difference",
    "book_up_provider_age_ms",
    "book_down_provider_age_ms",
    "book_provider_age_skew_ms",
]


def preopen_candidate_spec() -> CandidateSpec:
    return CandidateSpec(
        PREOPEN_CANDIDATE,
        "histogram",
        tuple(CORE_ENRICHED_FEATURES + PREOPEN_MODEL_FEATURES),
        EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY,
    )


def strict_book_candidate_spec(config: EntryBenchmarkConfig) -> CandidateSpec:
    return CandidateSpec(
        config.benchmark.strict_book_candidate,
        "histogram",
        tuple(CORE_ENRICHED_FEATURES + STRICT_BOOK_FEATURES),
        EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY,
    )


def walk_forward_offline_candidate(
    frame: pl.DataFrame,
    spec: CandidateSpec,
    config: CoreTrainingConfig,
) -> tuple[dict[str, Any], pl.DataFrame]:
    started = time.perf_counter()
    fold_results = [
        evaluate_fold(frame, spec, fold_index, config)
        for fold_index in range(len(config.split.validation_windows))
    ]
    selected = combined_fold_rows(fold_results)
    fixed = combined_fixed_time_rows(fold_results)
    eligible = sum(int(result["eligible_markets"]) for result in fold_results)
    metrics = classification_metrics(selected, eligible_markets=eligible)
    baseline = baseline_metrics(selected, eligible_markets=eligible)
    paired = paired_uplift(selected)
    bootstrap = block_bootstrap_uplift(
        selected,
        resamples=config.gates.bootstrap_resamples,
        random_seed=config.model.random_seed,
        block="hour",
    )
    nonnegative_folds = sum(
        result["paired"]["accuracy_uplift"]
        >= config.gates.minimum_same_time_path_uplift
        for result in fold_results
    )
    passed = bool(
        nonnegative_folds >= config.gates.minimum_nonnegative_uplift_folds
        and metrics["accuracy"] >= config.gates.target_accuracy
        and metrics["balanced_accuracy"] >= config.gates.target_balanced_accuracy
        and metrics["up_recall"] >= config.gates.minimum_direction_recall
        and metrics["down_recall"] >= config.gates.minimum_direction_recall
        and metrics["coverage"] >= config.gates.minimum_coverage
        and paired["accuracy_uplift"]
        >= config.gates.minimum_same_time_path_uplift
    )
    result = {
        "candidate": spec.name,
        "family": spec.family,
        "deployment_compatible": False,
        "deployment_blocker": "feature schema is not available in the current Rust hot path",
        "feature_count": len(spec.feature_names),
        "features": list(spec.feature_names),
        "row_weight_policy": spec.row_weight_policy,
        "folds": [
            {
                key: value
                for key, value in fold.items()
                if key not in {"prediction_rows", "fixed_time_prediction_rows"}
            }
            for fold in fold_results
        ],
        "out_of_fold": metrics,
        "baseline": baseline,
        "paired": paired,
        "bootstrap": bootstrap,
        "timing": first_crossing_timing(selected, eligible_markets=eligible),
        "nonnegative_uplift_folds": nonnegative_folds,
        "passed_development": passed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    predictions = _combine_selected_and_fixed(selected, fixed)
    return result, predictions


def derive_strict_book_frame(
    core_frame: pl.DataFrame,
    execution_evidence: pl.DataFrame,
) -> pl.DataFrame:
    strict = execution_evidence.filter(pl.col("strict_both_side_eligible"))
    strict = strict.with_columns(
        ((pl.col("up_best_bid") + pl.col("up_best_ask")) / 2).alias(
            "book_up_mid"
        ),
        ((pl.col("down_best_bid") + pl.col("down_best_ask")) / 2).alias(
            "book_down_mid"
        ),
        (pl.col("up_best_ask") - pl.col("up_best_bid")).alias(
            "book_up_spread"
        ),
        (pl.col("down_best_ask") - pl.col("down_best_bid")).alias(
            "book_down_spread"
        ),
        pl.col("up_ask_vwap_5").alias("book_up_ask_vwap_5"),
        pl.col("down_ask_vwap_5").alias("book_down_ask_vwap_5"),
        (pl.col("up_ask_vwap_5") - pl.col("up_best_ask")).alias(
            "book_up_vwap_slippage_5"
        ),
        (pl.col("down_ask_vwap_5") - pl.col("down_best_ask")).alias(
            "book_down_vwap_slippage_5"
        ),
        pl.col("up_imbalance").alias("book_up_imbalance"),
        pl.col("down_imbalance").alias("book_down_imbalance"),
        pl.col("up_bid_depth").log1p().alias("book_up_log_bid_depth"),
        pl.col("down_bid_depth").log1p().alias("book_down_log_bid_depth"),
        pl.col("up_ask_depth").log1p().alias("book_up_log_ask_depth"),
        pl.col("down_ask_depth").log1p().alias("book_down_log_ask_depth"),
        (
            (pl.col("observed_at") - pl.col("up_provider_received_at"))
            .dt.total_milliseconds()
            .cast(pl.Float64)
        ).alias("book_up_provider_age_ms"),
        (
            (pl.col("observed_at") - pl.col("down_provider_received_at"))
            .dt.total_milliseconds()
            .cast(pl.Float64)
        ).alias("book_down_provider_age_ms"),
    ).with_columns(
        (pl.col("book_up_mid") + pl.col("book_down_mid") - 1).alias(
            "book_mid_complement_residual"
        ),
        (pl.col("up_best_ask") + pl.col("down_best_ask") - 1).alias(
            "book_ask_complement_residual"
        ),
        (pl.col("book_up_mid") - pl.col("book_down_mid")).alias(
            "book_mid_difference"
        ),
        (
            pl.col("book_up_provider_age_ms")
            - pl.col("book_down_provider_age_ms")
        )
        .abs()
        .alias("book_provider_age_skew_ms"),
    )
    book = strict.select(
        "market_id",
        "observed_at",
        *STRICT_BOOK_FEATURES,
    )
    joined = core_frame.join(
        book,
        on=["market_id", "observed_at"],
        how="inner",
        validate="1:1",
    )
    invalid = joined.select(
        pl.any_horizontal(
            [pl.col(feature).is_null() for feature in STRICT_BOOK_FEATURES]
        ).alias("invalid")
    )["invalid"].sum()
    if invalid:
        raise RuntimeError("strict book cohort contains missing model features")
    return joined.with_columns(pl.lit(True).alias("model_eligible"))


def train_strict_book_candidate(
    strict_frame: pl.DataFrame,
    universal_core_frame: pl.DataFrame,
    benchmark_config: EntryBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> tuple[dict[str, Any], pl.DataFrame, FrozenTrainingBundle]:
    spec = strict_book_candidate_spec(benchmark_config)
    split = benchmark_config.book_split
    fit = range_frame(strict_frame, split.fit_start, split.fit_end)
    calibration_all = range_frame(
        strict_frame,
        split.calibration_start,
        split.calibration_end,
    )
    policy = range_frame(strict_frame, split.policy_start, split.policy_end)
    calibration, threshold_selection = _split_markets_in_half(calibration_all)
    policy_universe = range_frame(
        universal_core_frame,
        split.policy_start,
        split.policy_end,
    )["market_id"].unique()
    started = time.perf_counter()
    model, tuning = tune_and_fit_model(fit, spec, core_config)
    calibrator = fit_probability_calibrator(
        model,
        calibration,
        core_config,
        spec,
    )
    selection_probability = calibrator.probability(
        model.raw_logit(threshold_selection)
    )
    threshold_model_config = replace(
        core_config.model,
        confidence_min=benchmark_config.book_model.confidence_min,
        confidence_max=benchmark_config.book_model.confidence_max,
        confidence_step=benchmark_config.book_model.confidence_step,
    )
    threshold_core_config = replace(
        core_config,
        model=threshold_model_config,
    )
    thresholds = threshold_table(
        threshold_selection,
        selection_probability,
        threshold_core_config.model,
    )
    minimum_markets = max(
        100,
        math.ceil(
            threshold_selection["market_id"].n_unique()
            * core_config.gates.minimum_coverage
        ),
    )
    threshold, threshold_qualified = choose_threshold(
        thresholds,
        core_config.gates,
        minimum_markets=minimum_markets,
    )
    policy_probability = calibrator.probability(model.raw_logit(policy))
    scored = scored_prediction_rows(policy, policy_probability).with_columns(
        pl.lit(spec.name).alias("candidate"),
        pl.lit(True).alias("model_eligible"),
    )
    selected = first_prediction_rows(
        policy,
        policy_probability,
        threshold,
    ).with_columns(pl.lit(spec.name).alias("candidate"))
    eligible_markets = len(policy_universe)
    metrics = classification_metrics(
        selected,
        eligible_markets=eligible_markets,
    )
    timing = first_crossing_timing(
        selected,
        eligible_markets=eligible_markets,
    )
    latency = _python_batch_latency(
        FrozenTrainingBundle(
            model=model,
            calibrator=calibrator,
            confidence_threshold=threshold,
        ),
        policy,
    )
    result = {
        "candidate": spec.name,
        "family": spec.family,
        "deployment_compatible": False,
        "deployment_blocker": (
            "strict book features are intentionally outside the current Rust "
            "58-feature runtime contract"
        ),
        "feature_count": len(spec.feature_names),
        "features": list(spec.feature_names),
        "quality_columns_are_features": False,
        "row_weight_policy": spec.row_weight_policy,
        "cohort": {
            "fit_start": split.fit_start.isoformat(),
            "fit_end": split.fit_end.isoformat(),
            "calibration_start": calibration["window_start"].min().isoformat(),
            "calibration_end": calibration["window_start"].max().isoformat(),
            "threshold_start": (
                threshold_selection["window_start"].min().isoformat()
            ),
            "threshold_end": (
                threshold_selection["window_start"].max().isoformat()
            ),
            "policy_start": split.policy_start.isoformat(),
            "policy_end": split.policy_end.isoformat(),
            "fit_markets": fit["market_id"].n_unique(),
            "calibration_markets": calibration["market_id"].n_unique(),
            "threshold_markets": threshold_selection["market_id"].n_unique(),
            "policy_strict_book_markets": policy["market_id"].n_unique(),
            "policy_universal_markets": eligible_markets,
        },
        "tuning": tuning,
        "calibrator": asdict(calibrator),
        "confidence_threshold": threshold,
        "threshold_qualified": threshold_qualified,
        "threshold_history": thresholds,
        "metrics": metrics,
        "timing": timing,
        "python_scoring": latency,
        "elapsed_seconds": time.perf_counter() - started,
    }
    bundle = FrozenTrainingBundle(
        model=model,
        calibrator=calibrator,
        confidence_threshold=threshold,
    )
    return result, scored, bundle


def _split_markets_in_half(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    markets = (
        frame.select("market_id", "window_start")
        .unique(subset=["market_id"])
        .sort("window_start")
    )
    split_index = markets.height // 2
    if split_index < 100 or markets.height - split_index < 100:
        raise RuntimeError("book calibration cohort is too small to split")
    calibration_ids = markets[:split_index]["market_id"]
    threshold_ids = markets[split_index:]["market_id"]
    return (
        frame.filter(pl.col("market_id").is_in(calibration_ids.implode())),
        frame.filter(pl.col("market_id").is_in(threshold_ids.implode())),
    )


def _combine_selected_and_fixed(
    selected: pl.DataFrame,
    fixed: pl.DataFrame,
) -> pl.DataFrame:
    return (
        pl.concat([selected, fixed], how="diagonal_relaxed")
        .unique(
            subset=["candidate", "market_id", "observed_at", "seconds_elapsed"],
            keep="first",
        )
        .sort(["candidate", "observed_at", "market_id"])
    )


def _python_batch_latency(
    bundle: FrozenTrainingBundle,
    frame: pl.DataFrame,
    *,
    repetitions: int = 20,
) -> dict[str, Any]:
    sample = frame.head(min(frame.height, 4096))
    bundle.probability(sample)
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        bundle.probability(sample)
        durations.append(time.perf_counter_ns() - started)
    durations.sort()
    p50 = durations[len(durations) // 2]
    p99 = durations[min(len(durations) - 1, math.ceil(len(durations) * 0.99) - 1)]
    return {
        "runtime": "python_sklearn_training_diagnostic",
        "native_rust_latency_qualified": False,
        "rows_per_batch": sample.height,
        "repetitions": repetitions,
        "batch_p50_milliseconds": p50 / 1_000_000,
        "batch_p99_milliseconds": p99 / 1_000_000,
        "p99_microseconds_per_row": p99 / 1_000 / sample.height,
    }
