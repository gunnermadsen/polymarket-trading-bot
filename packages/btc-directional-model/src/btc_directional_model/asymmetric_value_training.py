"""Champion-family probability models for lower-cost asymmetric opportunities."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl

from .asymmetric_value_config import AsymmetricValueConfig
from .asymmetric_value_data import (
    EARLY_CAUSAL_ORACLE_FEATURES,
    POLYMARKET_VALUE_FEATURES,
)
from .chainlink_oi_features import CHAINLINK_CANDLE_FEATURES
from .core_config import CoreTrainingConfig
from .core_features import CORE_ENRICHED_FEATURES
from .core_training import MARKET_EQUAL_ROW_WEIGHT_POLICY, CandidateSpec, fit_model
from .early_value_training import (
    EarlyModel,
    fit_time_band_calibrators,
    probability_metrics,
)
from .spot_l2_chainlink_features import L2_FEATURES

PRICE_LOGISTIC = "price_logistic_control"
CORE_CONTROL = "original_core_early_safe_hgb_control"
PAIRED_CORE_CONTROL = "paired_core_early_safe_hgb_control"
CORE_PRICE = "core_price_hgb"
CORE_L2_PRICE = "core_l2_price_hgb"
CORE_CANDLES_PRICE = "core_chainlink_candles_price_hgb"
CORE_L2_CANDLES_PRICE = "core_l2_chainlink_price_hgb"
CORE_ORACLE_PRICE = "core_oracle_price_hgb"
ORACLE_MATCHED_CORE_PRICE_CONTROL = "oracle_matched_core_price_hgb_control"
CORE_ORACLE_L2_CANDLES_PRICE = "core_oracle_l2_chainlink_price_hgb"
L2_CANDLES_MATCHED_CORE_PRICE_CONTROL = (
    "l2_candles_matched_core_price_hgb_control"
)
ORACLE_L2_CANDLES_MATCHED_CORE_ORACLE_PRICE_CONTROL = (
    "oracle_l2_candles_matched_core_oracle_price_hgb_control"
)

ASYMMETRIC_VALUE_CANDIDATES = (
    PRICE_LOGISTIC,
    CORE_CONTROL,
    PAIRED_CORE_CONTROL,
    CORE_PRICE,
    ORACLE_MATCHED_CORE_PRICE_CONTROL,
    CORE_ORACLE_PRICE,
    CORE_L2_PRICE,
    CORE_CANDLES_PRICE,
    L2_CANDLES_MATCHED_CORE_PRICE_CONTROL,
    CORE_L2_CANDLES_PRICE,
    ORACLE_L2_CANDLES_MATCHED_CORE_ORACLE_PRICE_CONTROL,
    CORE_ORACLE_L2_CANDLES_PRICE,
)

MODEL_SELECTION_ELIGIBLE = frozenset(
    set(ASYMMETRIC_VALUE_CANDIDATES)
    - {
        L2_CANDLES_MATCHED_CORE_PRICE_CONTROL,
        ORACLE_MATCHED_CORE_PRICE_CONTROL,
        ORACLE_L2_CANDLES_MATCHED_CORE_ORACLE_PRICE_CONTROL,
    }
)
ORACLE_FEATURE_CANDIDATES = frozenset(
    {
        CORE_ORACLE_PRICE,
        ORACLE_L2_CANDLES_MATCHED_CORE_ORACLE_PRICE_CONTROL,
        CORE_ORACLE_L2_CANDLES_PRICE,
    }
)


def asymmetric_value_feature_sets() -> dict[str, tuple[str, ...]]:
    core = tuple(CORE_ENRICHED_FEATURES)
    price = tuple(POLYMARKET_VALUE_FEATURES)
    oracle = tuple(EARLY_CAUSAL_ORACLE_FEATURES)
    price_control = tuple(dict.fromkeys(("seconds_elapsed_scaled", *price)))
    return {
        PRICE_LOGISTIC: price_control,
        CORE_CONTROL: core,
        PAIRED_CORE_CONTROL: core,
        CORE_PRICE: tuple(dict.fromkeys((*core, *price))),
        ORACLE_MATCHED_CORE_PRICE_CONTROL: tuple(
            dict.fromkeys((*core, *price))
        ),
        CORE_ORACLE_PRICE: tuple(dict.fromkeys((*core, *oracle, *price))),
        CORE_L2_PRICE: tuple(dict.fromkeys((*core, *L2_FEATURES, *price))),
        CORE_CANDLES_PRICE: tuple(
            dict.fromkeys((*core, *CHAINLINK_CANDLE_FEATURES, *price))
        ),
        L2_CANDLES_MATCHED_CORE_PRICE_CONTROL: tuple(
            dict.fromkeys((*core, *price))
        ),
        CORE_L2_CANDLES_PRICE: tuple(
            dict.fromkeys((*core, *L2_FEATURES, *CHAINLINK_CANDLE_FEATURES, *price))
        ),
        ORACLE_L2_CANDLES_MATCHED_CORE_ORACLE_PRICE_CONTROL: tuple(
            dict.fromkeys((*core, *oracle, *price))
        ),
        CORE_ORACLE_L2_CANDLES_PRICE: tuple(
            dict.fromkeys(
                (
                    *core,
                    *oracle,
                    *L2_FEATURES,
                    *CHAINLINK_CANDLE_FEATURES,
                    *price,
                )
            )
        ),
    }


def fit_asymmetric_value_models(
    model_frames: dict[str, pl.DataFrame],
    original_core_frame: pl.DataFrame,
    config: AsymmetricValueConfig,
    core_config: CoreTrainingConfig,
) -> tuple[dict[str, EarlyModel], dict[str, Any]]:
    required = {"market_id", "window_start", "seconds_elapsed", "label_up"}
    expected = set(ASYMMETRIC_VALUE_CANDIDATES)
    if set(model_frames) != expected:
        raise ValueError("asymmetric-value model frames do not match the frozen candidates")
    for name, frame in {**model_frames, "original_core": original_core_frame}.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} frame is missing columns: " + ", ".join(missing))

    histogram = asdict(core_config.model.histogram_candidates[0])
    feature_sets = asymmetric_value_feature_sets()
    models: dict[str, EarlyModel] = {}
    summary: dict[str, Any] = {
        "selection_metric": "economic_policy_contract",
        "accuracy_threshold_used_for_selection": False,
        "market_equal_row_weights": True,
        "earliest_decision_second": 5,
        "minimum_finite_feature_fraction_at_earliest_decision": 0.95,
        "opening_boundary_features_used": False,
        "opening_boundary_exclusion_reason": (
            "historical opening-boundary facts lack a proven decision-time availability timestamp"
        ),
        "oracle_feature_contract": list(EARLY_CAUSAL_ORACLE_FEATURES),
        "profiles": {},
    }
    for name in ASYMMETRIC_VALUE_CANDIDATES:
        scoring_source = model_frames[name]
        training_source = original_core_frame if name == CORE_CONTROL else scoring_source
        fit_frame = _window(training_source, config.fit.start, config.fit.end)
        calibration_frame = _window(
            training_source,
            config.calibration.start,
            config.calibration.end,
        )
        policy_frame = _window(scoring_source, config.policy.start, config.policy.end)
        if any(item.is_empty() for item in (fit_frame, calibration_frame, policy_frame)):
            raise RuntimeError(f"{name} fit, calibration, and policy frames must be non-empty")
        features, feature_availability = _earliest_available_feature_set(
            fit_frame,
            feature_sets[name],
        )
        if name in ORACLE_FEATURE_CANDIDATES and not set(
            EARLY_CAUSAL_ORACLE_FEATURES
        ).issubset(features):
            raise RuntimeError(f"{name} lost its causal oracle feature contract")
        required_external = {
            CORE_L2_PRICE: set(L2_FEATURES),
            CORE_CANDLES_PRICE: set(CHAINLINK_CANDLE_FEATURES),
            CORE_L2_CANDLES_PRICE: {
                *L2_FEATURES,
                *CHAINLINK_CANDLE_FEATURES,
            },
            CORE_ORACLE_L2_CANDLES_PRICE: {
                *L2_FEATURES,
                *CHAINLINK_CANDLE_FEATURES,
            },
        }.get(name, set())
        if not required_external.issubset(features):
            raise RuntimeError(f"{name} lost its external feature contract")
        absent = sorted(set(features) - set(training_source.columns))
        if absent:
            raise RuntimeError(f"{name} features are missing: " + ", ".join(absent))
        calibration_coverage = _calibration_coverage(
            calibration_frame,
            config,
            model=name,
        )
        family = "logistic" if name == PRICE_LOGISTIC else "histogram"
        parameters = (
            {"c": core_config.model.c_candidates[0]}
            if family == "logistic"
            else histogram
        )
        spec = CandidateSpec(
            name=name,
            family=family,
            feature_names=features,
            row_weight_policy=MARKET_EQUAL_ROW_WEIGHT_POLICY,
        )
        fitted = fit_model(fit_frame, spec, parameters, core_config)
        calibrators = fit_time_band_calibrators(
            fitted,
            calibration_frame,
            config=config,  # type: ignore[arg-type]
            core_config=core_config,
            spec=spec,
        )
        bundle = EarlyModel(name=name, model=fitted, calibrators=calibrators)
        policy_probability = bundle.probability(policy_frame)
        summary["profiles"][name] = {
            "family": family,
            "features": list(features),
            "feature_count": len(features),
            "training_cohort": _training_cohort(name),
            "scoring_cohort": _scoring_cohort(name),
            "fit_rows": fit_frame.height,
            "fit_markets": fit_frame["market_id"].n_unique(),
            "calibration_rows": calibration_frame.height,
            "calibration_markets": calibration_frame["market_id"].n_unique(),
            "calibration_evidence": calibration_coverage,
            "policy_rows": policy_frame.height,
            "policy_markets": policy_frame["market_id"].n_unique(),
            "feature_availability_at_second_5": feature_availability,
            "policy_probability_metrics": probability_metrics(
                policy_frame,
                policy_probability,
                sample_weight=_market_equal_weights(policy_frame),
            ),
            "calibration_bands": [
                {
                    "start_second": item.start_second,
                    "end_second_exclusive": item.end_second_exclusive,
                    "rows": item.rows,
                    "markets": item.markets,
                    **asdict(item.calibrator),
                }
                for item in calibrators
            ],
        }
        models[name] = bundle
    return models, summary


def _earliest_available_feature_set(
    fit_frame: pl.DataFrame,
    candidates: tuple[str, ...],
    *,
    minimum_finite_fraction: float = 0.95,
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    earliest = fit_frame.filter(pl.col("seconds_elapsed") == 5)
    if earliest.is_empty():
        raise RuntimeError("asymmetric-value fit evidence lacks second-5 rows")
    missing = sorted(set(candidates) - set(earliest.columns))
    if missing:
        raise RuntimeError(
            "asymmetric-value earliest feature audit is missing: " + ", ".join(missing)
        )
    matrix = earliest.select(*candidates).to_numpy()
    fractions = np.isfinite(matrix).mean(axis=0)
    availability = {
        feature: {
            "finite_fraction": float(fraction),
            "eligible": bool(fraction >= minimum_finite_fraction),
        }
        for feature, fraction in zip(candidates, fractions, strict=True)
    }
    filtered = tuple(
        feature for feature in candidates if availability[feature]["eligible"]
    )
    if not filtered:
        raise RuntimeError("earliest-available feature filtering emptied a candidate")
    return filtered, availability


def _calibration_coverage(
    frame: pl.DataFrame,
    config: AsymmetricValueConfig,
    *,
    model: str,
) -> list[dict[str, int]]:
    coverage: list[dict[str, int]] = []
    for start, end in config.calibration_bands:
        band = frame.filter(
            pl.col("seconds_elapsed").is_between(start, end, closed="left")
        )
        markets = band["market_id"].n_unique()
        coverage.append(
            {
                "start_second": start,
                "end_second_exclusive": end,
                "rows": band.height,
                "markets": markets,
            }
        )
        if markets < config.gates.minimum_calibration_markets_per_band:
            raise RuntimeError(
                f"{model} calibration band {start}-{end} has {markets} markets; "
                f"requires {config.gates.minimum_calibration_markets_per_band}"
            )
    return coverage


def _training_cohort(model: str) -> str:
    if model == CORE_CONTROL:
        return "universal_original_core"
    if model in {PRICE_LOGISTIC, PAIRED_CORE_CONTROL, CORE_PRICE}:
        return "exact_execution_price_cohort"
    if model in {ORACLE_MATCHED_CORE_PRICE_CONTROL, CORE_ORACLE_PRICE}:
        return "causal_oracle_exact_execution_cohort"
    if model in {
        CORE_L2_PRICE,
        CORE_CANDLES_PRICE,
        L2_CANDLES_MATCHED_CORE_PRICE_CONTROL,
        CORE_L2_CANDLES_PRICE,
    }:
        return "paired_l2_candle_exact_execution_cohort"
    return "causal_oracle_paired_l2_candle_exact_execution_cohort"


def _scoring_cohort(model: str) -> str:
    if model in {PRICE_LOGISTIC, CORE_CONTROL, PAIRED_CORE_CONTROL, CORE_PRICE}:
        return "exact_execution_price_cohort"
    if model in {ORACLE_MATCHED_CORE_PRICE_CONTROL, CORE_ORACLE_PRICE}:
        return "causal_oracle_exact_execution_cohort"
    if model in {
        CORE_L2_PRICE,
        CORE_CANDLES_PRICE,
        L2_CANDLES_MATCHED_CORE_PRICE_CONTROL,
        CORE_L2_CANDLES_PRICE,
    }:
        return "paired_l2_candle_exact_execution_cohort"
    return "causal_oracle_paired_l2_candle_exact_execution_cohort"


def asymmetric_probability_frame(
    frame: pl.DataFrame,
    probability_yes: Any,
    *,
    model: str,
) -> pl.DataFrame:
    return frame.select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "fee_rate",
        "yes_best_ask",
        "yes_ask_vwap_5",
        "yes_ask_depth",
        "no_best_ask",
        "no_ask_vwap_5",
        "no_ask_depth",
        "yes_cost_per_share",
        "no_cost_per_share",
        "yes_execution_cost_per_share",
        "no_execution_cost_per_share",
    ).with_columns(
        pl.lit(model).alias("model"),
        pl.Series("probability_yes", probability_yes),
    )


def _window(frame: pl.DataFrame, start: Any, end: Any) -> pl.DataFrame:
    return frame.filter(pl.col("window_start").is_between(start, end, closed="left"))


def _market_equal_weights(frame: pl.DataFrame) -> Any:
    counts = frame.group_by("market_id").len().rename({"len": "_market_rows"})
    return (
        frame.select("market_id")
        .join(counts, on="market_id", how="left", validate="m:1")["_market_rows"]
        .cast(pl.Float64)
        .pow(-1.0)
        .to_numpy()
    )
