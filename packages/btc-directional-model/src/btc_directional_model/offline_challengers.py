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
# The legacy five-share feature set remains available to old diagnostic
# wrappers. Quality flags and provider age are deliberately absent: they route
# rows into a cohort but must never become directional predictors.
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
]
STRICT_BOOK_TEN_SHARE_FEATURES = [
    "book_up_ask_vwap_10",
    "book_down_ask_vwap_10",
    "book_up_vwap_slippage_10",
    "book_down_vwap_slippage_10",
    "book_vwap_10_complement_residual",
]
STRICT_BOOK_DELTA_SOURCE_FEATURES = [
    "book_up_mid",
    "book_down_mid",
    "book_up_spread",
    "book_down_spread",
    "book_up_ask_vwap_5",
    "book_down_ask_vwap_5",
    "book_up_ask_vwap_10",
    "book_down_ask_vwap_10",
    "book_up_imbalance",
    "book_down_imbalance",
    "book_up_log_bid_depth",
    "book_down_log_bid_depth",
    "book_up_log_ask_depth",
    "book_down_log_ask_depth",
    "book_mid_complement_residual",
    "book_ask_complement_residual",
    "book_vwap_10_complement_residual",
    "book_mid_difference",
]
STRICT_BOOK_DELTA_FEATURES = [
    f"{feature}_delta_5s" for feature in STRICT_BOOK_DELTA_SOURCE_FEATURES
]
STRICT_BOOK_V2_FEATURES = [
    *STRICT_BOOK_FEATURES,
    *STRICT_BOOK_TEN_SHARE_FEATURES,
    *STRICT_BOOK_DELTA_FEATURES,
]


def preopen_candidate_spec() -> CandidateSpec:
    return CandidateSpec(
        PREOPEN_CANDIDATE,
        "histogram",
        tuple(CORE_ENRICHED_FEATURES + PREOPEN_MODEL_FEATURES),
        EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY,
    )


def strict_book_candidate_spec(config: EntryBenchmarkConfig) -> CandidateSpec:
    """Legacy five-share diagnostic specification."""

    return CandidateSpec(
        config.benchmark.strict_book_candidate,
        "histogram",
        tuple(CORE_ENRICHED_FEATURES + STRICT_BOOK_FEATURES),
        EARLY_ENTRY_MARKET_EQUAL_ROW_WEIGHT_POLICY,
    )


def strict_cohort_candidate_spec(
    config: EntryBenchmarkConfig,
    *,
    candidate_name: str,
    include_book_features: bool,
) -> CandidateSpec:
    """Build an ablation spec for the same quality-qualified row cohort."""

    if not candidate_name.strip():
        raise ValueError("strict-cohort candidate name must be non-empty")
    features = list(CORE_ENRICHED_FEATURES)
    if include_book_features:
        features.extend(STRICT_BOOK_V2_FEATURES)
    return CandidateSpec(
        candidate_name,
        "histogram",
        tuple(features),
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
    """Preserve the legacy five-share diagnostic-frame behavior."""

    return derive_strict_book_feature_frame(
        core_frame,
        execution_evidence,
        require_ten_share=False,
        include_deltas=False,
    )


def derive_strict_book_feature_frame(
    core_frame: pl.DataFrame,
    execution_evidence: pl.DataFrame,
    *,
    require_ten_share: bool = True,
    include_deltas: bool = True,
) -> pl.DataFrame:
    """Derive model features only after strict point-in-time routing.

    Ten-share eligibility is the v2 fitting contract. Five-share prices remain
    in the frame for the unchanged five-share economic evaluation. Delta rows
    exist only when the immediately preceding strict observation for the same
    market is exactly five seconds earlier.
    """

    if include_deltas and not require_ten_share:
        raise ValueError("causal book deltas require strict ten-share evidence")
    eligibility_column = (
        "strict_both_side_eligible_10"
        if require_ten_share
        else "strict_both_side_eligible"
    )
    required_columns = {
        "market_id",
        "observed_at",
        eligibility_column,
        "up_best_bid",
        "up_best_ask",
        "up_bid_depth",
        "up_ask_depth",
        "up_ask_vwap_5",
        "up_imbalance",
        "down_best_bid",
        "down_best_ask",
        "down_bid_depth",
        "down_ask_depth",
        "down_ask_vwap_5",
        "down_imbalance",
    }
    if require_ten_share:
        required_columns.update({"up_ask_vwap_10", "down_ask_vwap_10"})
    missing = sorted(required_columns - set(execution_evidence.columns))
    if missing:
        raise ValueError(
            "execution evidence is missing strict-book columns: "
            + ", ".join(missing)
        )

    # Filtering before every derived expression prevents invalid reconstruction
    # states from influencing levels, complements, or temporal changes.
    strict = (
        execution_evidence.filter(pl.col(eligibility_column))
        .sort(["market_id", "observed_at"])
    )
    duplicate_keys = (
        strict.group_by("market_id", "observed_at")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_keys:
        raise RuntimeError("strict book evidence contains duplicate row keys")

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
    )
    model_features = list(STRICT_BOOK_FEATURES)
    if require_ten_share:
        strict = strict.with_columns(
            pl.col("up_ask_vwap_10").alias("book_up_ask_vwap_10"),
            pl.col("down_ask_vwap_10").alias("book_down_ask_vwap_10"),
            (pl.col("up_ask_vwap_10") - pl.col("up_best_ask")).alias(
                "book_up_vwap_slippage_10"
            ),
            (pl.col("down_ask_vwap_10") - pl.col("down_best_ask")).alias(
                "book_down_vwap_slippage_10"
            ),
            (
                pl.col("up_ask_vwap_10")
                + pl.col("down_ask_vwap_10")
                - 1
            ).alias("book_vwap_10_complement_residual"),
        )
        model_features.extend(STRICT_BOOK_TEN_SHARE_FEATURES)
    if include_deltas:
        previous_columns = [
            pl.col("observed_at")
            .shift(1)
            .over("market_id")
            .alias("_previous_observed_at"),
            *[
                pl.col(feature)
                .shift(1)
                .over("market_id")
                .alias(f"_previous_{feature}")
                for feature in STRICT_BOOK_DELTA_SOURCE_FEATURES
            ],
        ]
        strict = strict.with_columns(*previous_columns).with_columns(
            *[
                pl.when(
                    pl.col("_previous_observed_at")
                    == pl.col("observed_at") - pl.duration(seconds=5)
                )
                .then(pl.col(feature) - pl.col(f"_previous_{feature}"))
                .otherwise(None)
                .alias(f"{feature}_delta_5s")
                for feature in STRICT_BOOK_DELTA_SOURCE_FEATURES
            ]
        )
        model_features.extend(STRICT_BOOK_DELTA_FEATURES)

    book = strict.select(
        "market_id",
        "observed_at",
        *model_features,
    ).filter(
        ~pl.any_horizontal(
            [
                pl.col(feature).is_null() | ~pl.col(feature).is_finite()
                for feature in model_features
            ]
        )
    )
    joined = core_frame.join(
        book,
        on=["market_id", "observed_at"],
        how="inner",
        validate="1:1",
    )
    invalid = joined.select(
        pl.any_horizontal(
            [
                pl.col(feature).is_null() | ~pl.col(feature).is_finite()
                for feature in model_features
            ]
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
    """Legacy wrapper around the reusable strict-cohort fit/evaluate APIs."""

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
    training, bundle = fit_strict_cohort_candidate(
        fit,
        calibration,
        threshold_selection,
        spec,
        benchmark_config,
        core_config,
    )
    evaluation, scored = evaluate_strict_cohort_candidate(
        policy,
        bundle,
        eligible_markets=len(policy_universe),
    )
    result = {
        **training,
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
            "policy_universal_markets": len(policy_universe),
        },
        "metrics": evaluation["metrics"],
        "timing": evaluation["timing"],
        "python_scoring": evaluation["python_scoring"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    return result, scored, bundle


def fit_strict_cohort_candidate(
    fit_frame: pl.DataFrame,
    calibration_frame: pl.DataFrame,
    threshold_selection_frame: pl.DataFrame,
    spec: CandidateSpec,
    benchmark_config: EntryBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> tuple[dict[str, Any], FrozenTrainingBundle]:
    """Fit one candidate on an already frozen strict-valid row cohort.

    Passing a BTC-only spec and a BTC-plus-book spec with the same three frames
    gives an exact-row ablation with identical chronology and weighting.
    """

    for name, frame in (
        ("fit", fit_frame),
        ("calibration", calibration_frame),
        ("threshold selection", threshold_selection_frame),
    ):
        _validate_strict_cohort_frame(frame, spec.feature_names, name=name)

    started = time.perf_counter()
    model, tuning = tune_and_fit_model(fit_frame, spec, core_config)
    calibrator = fit_probability_calibrator(
        model,
        calibration_frame,
        core_config,
        spec,
    )
    selection_probability = calibrator.probability(
        model.raw_logit(threshold_selection_frame)
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
        threshold_selection_frame,
        selection_probability,
        threshold_core_config.model,
    )
    minimum_markets = max(
        100,
        math.ceil(
            threshold_selection_frame["market_id"].n_unique()
            * core_config.gates.minimum_coverage
        ),
    )
    threshold, threshold_qualified = choose_threshold(
        thresholds,
        core_config.gates,
        minimum_markets=minimum_markets,
    )
    bundle = FrozenTrainingBundle(
        model=model,
        calibrator=calibrator,
        confidence_threshold=threshold,
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
        "provider_age_columns_are_features": False,
        "row_weight_policy": spec.row_weight_policy,
        "training_rows": {
            "fit": fit_frame.height,
            "fit_markets": fit_frame["market_id"].n_unique(),
            "calibration": calibration_frame.height,
            "calibration_markets": calibration_frame["market_id"].n_unique(),
            "threshold_selection": threshold_selection_frame.height,
            "threshold_markets": threshold_selection_frame[
                "market_id"
            ].n_unique(),
        },
        "tuning": tuning,
        "calibrator": asdict(calibrator),
        "confidence_threshold": threshold,
        "threshold_qualified": threshold_qualified,
        "threshold_history": thresholds,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return result, bundle


def score_strict_cohort_candidate(
    frame: pl.DataFrame,
    bundle: FrozenTrainingBundle,
) -> pl.DataFrame:
    """Score every supplied row without changing the strict cohort."""

    _validate_strict_cohort_frame(
        frame,
        bundle.model.feature_names,
        name="score",
    )
    probabilities = bundle.probability(frame)
    return scored_prediction_rows(frame, probabilities).with_columns(
        pl.lit(bundle.model.candidate_name).alias("candidate"),
        pl.lit(True).alias("model_eligible"),
    )


def evaluate_strict_cohort_candidate(
    frame: pl.DataFrame,
    bundle: FrozenTrainingBundle,
    *,
    eligible_markets: int | None = None,
) -> tuple[dict[str, Any], pl.DataFrame]:
    """Evaluate a frozen candidate on development-only strict-valid rows."""

    scored = score_strict_cohort_candidate(frame, bundle)
    eligible = (
        eligible_markets
        if eligible_markets is not None
        else frame["market_id"].n_unique()
    )
    if eligible < frame["market_id"].n_unique():
        raise ValueError(
            "eligible_markets cannot be smaller than the scored cohort"
        )
    selected = (
        scored.filter(pl.col("confidence") >= bundle.confidence_threshold)
        .sort(["observed_at", "market_id"])
        .group_by("market_id", maintain_order=True)
        .first()
    )
    result = {
        "candidate": bundle.model.candidate_name,
        "evidence_kind": "development",
        "evaluation_is_independent": False,
        "scored_rows": scored.height,
        "strict_markets": frame["market_id"].n_unique(),
        "eligible_markets": eligible,
        "confidence_threshold": bundle.confidence_threshold,
        "metrics": classification_metrics(
            selected,
            eligible_markets=eligible,
        ),
        "timing": first_crossing_timing(
            selected,
            eligible_markets=eligible,
        ),
        "python_scoring": _python_batch_latency(bundle, frame),
    }
    return result, scored


def _validate_strict_cohort_frame(
    frame: pl.DataFrame,
    feature_names: tuple[str, ...],
    *,
    name: str,
) -> None:
    if frame.is_empty():
        raise RuntimeError(f"{name} strict cohort is empty")
    required = {
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "binance_sign_up",
        *feature_names,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{name} strict cohort is missing model columns: "
            + ", ".join(missing)
        )
    duplicate_keys = (
        frame.group_by("market_id", "observed_at")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_keys:
        raise RuntimeError(f"{name} strict cohort contains duplicate row keys")


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
