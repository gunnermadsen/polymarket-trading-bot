"""Bounded Core+Oracle estimator fallbacks for incumbent calibration failure.

This module deliberately does not decide whether the fallback round may run.  The
caller must enforce the pre-sealed trigger.  Once called, it fits only the two
configured Core+Oracle candidates and emits probability-only, hash-addressed
evidence for the matched comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .asymmetric_value_config import AsymmetricValueConfig, EvidenceWindow
from .asymmetric_value_training import (
    CORE_ORACLE_PRICE,
    HYBRID_MARKET_EQUAL_ROW_WEIGHT_POLICY,
    POLICY_INACTIVE_IMPUTATION_STRATEGY,
    PRICE_BAND_COUNT,
    PRICE_BAND_WIDTH,
    TARGET_POLICY_INACTIVE_FEATURE_MATURITY,
    AsymmetricCalibrationCell,
    AsymmetricValueModel,
    asymmetric_value_feature_sets,
    fit_asymmetric_time_band_calibrators,
    hybrid_market_equal_weights,
    hybrid_target_mask,
    target_calibration_evidence,
)
from .core_config import CoreTrainingConfig
from .core_extract import file_sha256
from .core_training import FittedCoreModel, feature_matrix, finite_medians
from .early_value_training import TimeBandCalibrator

ESTIMATOR_FALLBACK_SCHEMA_VERSION = "btc-asymmetric-incumbent-estimator-fallback-v1"
ESTIMATOR_FALLBACK_PREDICTION_SCHEMA_VERSION = "btc-asymmetric-incumbent-estimator-predictions-v1"
E1_HYBRID50_H3 = "E1_hybrid50_h3"
E2_HYBRID50_H3_BOUNDARY = "E2_hybrid50_h3_boundary_weighted"
ESTIMATOR_FALLBACK_CANDIDATES = (E1_HYBRID50_H3, E2_HYBRID50_H3_BOUNDARY)
EXPECTED_CORE_ORACLE_PRICE_FEATURES = 75
EXPECTED_HISTOGRAM_PARAMETERS = {
    "learning_rate": 0.02,
    "max_iter": 240,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 300,
    "l2_regularization": 10.0,
}
EXPECTED_TARGET_WEIGHT = 0.50
EXPECTED_BOUNDARY_MINIMUM_EDGE = 0.02
EXPECTED_BOUNDARY_MAXIMUM_EDGE = 0.04
EXPECTED_BOUNDARY_MULTIPLIER = 2.0
EXPECTED_TARGET_PRICE_BAND = (0.20, 0.30)
EXPECTED_TARGET_TIME_BANDS = ((1, 15), (15, 30), (30, 45), (45, 60))
EXPECTED_TARGET_SIDES = ("YES", "NO")
TARGET_CELL_COUNT = 8
NON_TARGET_FALLBACK = "target_contract_identity"
BOUNDARY_WEIGHT_POLICY = "hybrid_broad_target_market_equal_boundary_renormalized"
PROBABILITY_COLUMNS = (
    "candidate_id",
    "market_id",
    "window_start",
    "observed_at",
    "seconds_elapsed",
    "probability_yes",
)


class _WindowedEstimatorConfig(Protocol):
    """Runtime shape consumed from ``IncumbentCalibrationConfig``."""

    calibration_fit: EvidenceWindow
    matched_comparison: EvidenceWindow
    final_refit: EvidenceWindow
    target_price_band: tuple[float, float]
    target_time_bands: tuple[tuple[int, int], ...]
    sides: tuple[str, ...]
    minimum_markets_per_cell: int
    minimum_days_per_cell: int
    identity_l2: float
    slope_bounds: tuple[float, float]
    intercept_bounds: tuple[float, float]
    random_seed: int
    conditional_estimator: Any
    asymmetric_value_config: Any


@dataclass
class FittedEstimatorFallback:
    """One runtime-exportable candidate and deterministic training evidence."""

    candidate_id: str
    bundle: AsymmetricValueModel
    evidence: dict[str, Any]
    semantic_sha256: str

    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        return self.bundle.probability(frame)


@dataclass(frozen=True)
class EstimatorProbabilityArtifact:
    """Probability-only matched-grid payload; outcomes and economics stay external."""

    candidate_id: str
    predictions: pl.DataFrame
    key_sha256: str
    probability_sha256: str
    artifact_sha256: str


def fit_configured_core_oracle_estimator_fallbacks(
    frame: pl.DataFrame,
    incumbent_config: _WindowedEstimatorConfig,
    asymmetric_config: AsymmetricValueConfig,
    core_config: CoreTrainingConfig,
    *,
    incumbent_probabilities: np.ndarray,
) -> dict[str, FittedEstimatorFallback]:
    """Fit the exact configured E1/E2 fallback matrix.

    Trigger evaluation intentionally belongs to the caller.  This function only
    validates the sealed estimator contract and executes its two candidates.
    """

    contract = incumbent_config.conditional_estimator
    _validate_configured_contract(incumbent_config, asymmetric_config, contract)
    return fit_core_oracle_estimator_fallbacks(
        frame,
        asymmetric_config=replace(
            asymmetric_config,
            random_seed=int(incumbent_config.random_seed),
        ),
        core_config=core_config,
        fit_window=asymmetric_config.fit,
        calibration_window=incumbent_config.calibration_fit,
        estimator_contract=contract,
        target_price_band=incumbent_config.target_price_band,
        target_time_bands=incumbent_config.target_time_bands,
        target_sides=incumbent_config.sides,
        minimum_markets_per_cell=incumbent_config.minimum_markets_per_cell,
        minimum_days_per_cell=incumbent_config.minimum_days_per_cell,
        identity_l2=incumbent_config.identity_l2,
        slope_bounds=incumbent_config.slope_bounds,
        intercept_bounds=incumbent_config.intercept_bounds,
        incumbent_probabilities=incumbent_probabilities,
    )


def refit_selected_core_oracle_estimator_fallback(
    frame: pl.DataFrame,
    incumbent_config: _WindowedEstimatorConfig,
    asymmetric_config: AsymmetricValueConfig,
    core_config: CoreTrainingConfig,
    *,
    candidate_id: str,
    incumbent_probabilities: np.ndarray,
) -> FittedEstimatorFallback:
    """Refit one already-selected arm with the sealed final calibration evidence.

    This does not select or compare a new candidate.  The estimator fit window
    remains frozen; only the selected candidate's calibration window expands to
    ``final_refit`` after development selection.
    """

    contract = incumbent_config.conditional_estimator
    _validate_configured_contract(incumbent_config, asymmetric_config, contract)
    fitted = fit_core_oracle_estimator_fallbacks(
        frame,
        asymmetric_config=replace(
            asymmetric_config,
            random_seed=int(incumbent_config.random_seed),
        ),
        core_config=core_config,
        fit_window=asymmetric_config.fit,
        calibration_window=incumbent_config.final_refit,
        estimator_contract=contract,
        target_price_band=incumbent_config.target_price_band,
        target_time_bands=incumbent_config.target_time_bands,
        target_sides=incumbent_config.sides,
        minimum_markets_per_cell=incumbent_config.minimum_markets_per_cell,
        minimum_days_per_cell=incumbent_config.minimum_days_per_cell,
        identity_l2=incumbent_config.identity_l2,
        slope_bounds=incumbent_config.slope_bounds,
        intercept_bounds=incumbent_config.intercept_bounds,
        incumbent_probabilities=incumbent_probabilities,
        candidate_ids=(candidate_id,),
    )
    return fitted[candidate_id]


def fit_core_oracle_estimator_fallbacks(
    frame: pl.DataFrame,
    *,
    asymmetric_config: AsymmetricValueConfig,
    core_config: CoreTrainingConfig,
    fit_window: EvidenceWindow,
    calibration_window: EvidenceWindow,
    estimator_contract: Any,
    target_price_band: tuple[float, float],
    target_time_bands: tuple[tuple[int, int], ...],
    target_sides: tuple[str, ...],
    minimum_markets_per_cell: int,
    minimum_days_per_cell: int,
    identity_l2: float,
    slope_bounds: tuple[float, float],
    intercept_bounds: tuple[float, float],
    incumbent_probabilities: np.ndarray,
    candidate_ids: tuple[str, ...] = ESTIMATOR_FALLBACK_CANDIDATES,
) -> dict[str, FittedEstimatorFallback]:
    """Fit exactly two sealed H3 candidates from one exact Core+Oracle+PM frame."""

    features = _core_oracle_feature_contract()
    _validate_estimator_contract(estimator_contract)
    if (
        not candidate_ids
        or len(set(candidate_ids)) != len(candidate_ids)
        or any(name not in ESTIMATOR_FALLBACK_CANDIDATES for name in candidate_ids)
        or tuple(name for name in ESTIMATOR_FALLBACK_CANDIDATES if name in candidate_ids)
        != candidate_ids
    ):
        raise ValueError("estimator fallback candidate subset is invalid")
    required = {
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "yes_ask_vwap_5",
        "no_ask_vwap_5",
        "yes_cost_per_share",
        "no_cost_per_share",
        *features,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("estimator fallback frame is missing: " + ", ".join(missing))
    probabilities = np.asarray(incumbent_probabilities, dtype=np.float64)
    if (
        probabilities.shape != (frame.height,)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities <= 0.0) | (probabilities >= 1.0))
    ):
        raise ValueError("incumbent probabilities must be aligned, finite, and inside (0, 1)")
    indexed = frame.with_row_index("__fallback_source_index")
    fit_indexed = _window(indexed, fit_window).sort(
        "window_start", "market_id", "seconds_elapsed", "observed_at"
    )
    calibration_frame = _window(frame, calibration_window).sort(
        "window_start", "market_id", "seconds_elapsed", "observed_at"
    )
    if fit_indexed.is_empty() or calibration_frame.is_empty():
        raise RuntimeError("estimator fallback fit and calibration windows must be non-empty")
    _validate_unique_grid(fit_indexed, cohort="fit")
    _validate_unique_grid(calibration_frame, cohort="calibration")
    fit_indices = fit_indexed["__fallback_source_index"].to_numpy().astype(np.int64)
    fit_frame = fit_indexed.drop("__fallback_source_index")
    fit_incumbent_probability = probabilities[fit_indices]
    labels = fit_frame["label_up"].to_numpy()
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise RuntimeError("estimator fallback fit window requires both outcomes")

    common_evidence = {
        "schema_version": ESTIMATOR_FALLBACK_SCHEMA_VERSION,
        "probability_only_selection": True,
        "economics_used_for_fitting": False,
        "feature_contract": CORE_ORACLE_PRICE,
        "feature_count": len(features),
        "features_sha256": _canonical_sha256(list(features)),
        "fit_window": _window_evidence(fit_window),
        "calibration_window": _window_evidence(calibration_window),
        "fit_rows": fit_frame.height,
        "fit_markets": fit_frame["market_id"].n_unique(),
        "calibration_rows": calibration_frame.height,
        "calibration_markets": calibration_frame["market_id"].n_unique(),
        "fit_key_sha256": _grid_key_sha256(fit_frame),
        "fit_content_sha256": _frame_content_sha256(
            fit_frame,
            features=features,
            incumbent_probabilities=fit_incumbent_probability,
        ),
        "calibration_key_sha256": _grid_key_sha256(calibration_frame),
        "calibration_content_sha256": _frame_content_sha256(
            calibration_frame,
            features=features,
            incumbent_probabilities=None,
        ),
        "target_price_band": list(target_price_band),
        "target_time_bands": [list(band) for band in target_time_bands],
        "target_sides": list(target_sides),
    }
    output: dict[str, FittedEstimatorFallback] = {}
    for candidate in estimator_contract.candidates:
        if str(candidate.name) not in candidate_ids:
            continue
        weights, weight_evidence = fallback_training_weights(
            fit_frame,
            asymmetric_config,
            candidate,
            incumbent_probabilities=fit_incumbent_probability,
        )
        model = _fit_histogram_estimator(
            fit_frame,
            features,
            weights,
            candidate=candidate,
            estimator_contract=estimator_contract,
            random_seed=asymmetric_config.random_seed,
            threads=core_config.compute.threads_per_fit,
        )
        calibrators = fit_asymmetric_time_band_calibrators(
            model,
            calibration_frame,
            asymmetric_config,
            core_config=core_config,
            parent_source="alltime",
        )
        cells, calibration_evidence = fit_target_side_price_time_calibrators(
            model,
            calibrators,
            calibration_frame,
            asymmetric_config,
            target_price_band=target_price_band,
            target_time_bands=target_time_bands,
            target_sides=target_sides,
            minimum_markets_per_cell=minimum_markets_per_cell,
            minimum_days_per_cell=minimum_days_per_cell,
            identity_l2=identity_l2,
            slope_bounds=slope_bounds,
            intercept_bounds=intercept_bounds,
        )
        bundle = AsymmetricValueModel(
            name=str(candidate.name),
            model=model,
            time_calibrators=calibrators,
            cells=cells,
            parent_calibration_source="alltime",
            identity_l2_strength=identity_l2,
        )
        semantic_sha256 = _bundle_semantic_sha256(bundle)
        evidence = {
            **common_evidence,
            "candidate_id": str(candidate.name),
            "candidate_contract": _candidate_evidence(candidate),
            "histogram": dict(EXPECTED_HISTOGRAM_PARAMETERS),
            "random_seed": asymmetric_config.random_seed,
            "training_weights": weight_evidence,
            "training_weight_sha256": _float_vector_sha256(weights),
            "policy_inactive_imputation": POLICY_INACTIVE_IMPUTATION_STRATEGY,
            "parent_time_calibrators": [
                {
                    "start_second": band.start_second,
                    "end_second_exclusive": band.end_second_exclusive,
                    "rows": band.rows,
                    "markets": band.markets,
                    **asdict(band.calibrator),
                }
                for band in calibrators
            ],
            "side_price_time_calibration": calibration_evidence,
            "semantic_sha256": semantic_sha256,
            "runtime_exportable": True,
        }
        output[str(candidate.name)] = FittedEstimatorFallback(
            candidate_id=str(candidate.name),
            bundle=bundle,
            evidence=evidence,
            semantic_sha256=semantic_sha256,
        )
    if tuple(output) != candidate_ids:
        raise RuntimeError("estimator fallback did not fit the requested sealed candidate matrix")
    return output


def score_core_oracle_estimator_fallback(
    fitted: FittedEstimatorFallback,
    frame: pl.DataFrame,
    *,
    window: EvidenceWindow | None = None,
) -> EstimatorProbabilityArtifact:
    """Score a candidate into a deterministic probability-only comparison payload."""

    scoring_frame = (_window(frame, window) if window is not None else frame).sort(
        "window_start", "market_id", "seconds_elapsed", "observed_at"
    )
    if scoring_frame.is_empty():
        raise ValueError("estimator fallback scoring frame is empty")
    _validate_unique_grid(scoring_frame, cohort="scoring")
    probability = fitted.probability(scoring_frame)
    if (
        probability.shape != (scoring_frame.height,)
        or not np.isfinite(probability).all()
        or np.any((probability <= 0.0) | (probability >= 1.0))
    ):
        raise RuntimeError("estimator fallback produced invalid probabilities")
    predictions = (
        scoring_frame.select(
            "market_id",
            "window_start",
            "observed_at",
            "seconds_elapsed",
        )
        .with_columns(
            pl.lit(fitted.candidate_id).alias("candidate_id"),
            pl.Series("probability_yes", probability),
        )
        .select(*PROBABILITY_COLUMNS)
    )
    key_sha256 = _grid_key_sha256(predictions)
    probability_sha256 = _float_vector_sha256(probability)
    artifact_payload = {
        "schema_version": ESTIMATOR_FALLBACK_PREDICTION_SCHEMA_VERSION,
        "candidate_id": fitted.candidate_id,
        "rows": predictions.height,
        "markets": predictions["market_id"].n_unique(),
        "key_sha256": key_sha256,
        "probability_sha256": probability_sha256,
        "model_semantic_sha256": fitted.semantic_sha256,
    }
    return EstimatorProbabilityArtifact(
        candidate_id=fitted.candidate_id,
        predictions=predictions,
        key_sha256=key_sha256,
        probability_sha256=probability_sha256,
        artifact_sha256=_canonical_sha256(artifact_payload),
    )


def fallback_training_weights(
    frame: pl.DataFrame,
    asymmetric_config: AsymmetricValueConfig,
    candidate: Any,
    *,
    incumbent_probabilities: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return E1 hybrid weights or E2 boundary-focused, market-renormalized weights."""

    if str(candidate.name) not in ESTIMATOR_FALLBACK_CANDIDATES:
        raise ValueError(f"unknown estimator fallback candidate: {candidate.name}")
    if not math.isclose(float(candidate.target_weight), EXPECTED_TARGET_WEIGHT):
        raise ValueError("estimator fallback target weight changed")
    base = hybrid_market_equal_weights(
        frame,
        asymmetric_config,
        target_weight=EXPECTED_TARGET_WEIGHT,
    )
    target = hybrid_target_mask(frame, asymmetric_config)
    if not bool(candidate.boundary_weighted):
        if str(candidate.name) != E1_HYBRID50_H3:
            raise ValueError("only E1 may omit boundary weighting")
        return base, _weight_evidence(
            frame,
            base,
            base,
            target,
            np.zeros(frame.height, dtype=bool),
            candidate,
        )

    if str(candidate.name) != E2_HYBRID50_H3_BOUNDARY:
        raise ValueError("only E2 may use boundary weighting")
    boundary = incumbent_admission_boundary_mask(
        frame,
        incumbent_probabilities,
        minimum_edge=float(candidate.boundary_minimum_edge),
        maximum_edge=float(candidate.boundary_maximum_edge),
        target_mask=target,
    )
    if not boundary.any():
        raise RuntimeError("E2 has no incumbent admission-boundary rows")
    adjusted = base * np.where(boundary, float(candidate.boundary_multiplier), 1.0)
    weights = _restore_market_weight_exposure(frame, base, adjusted)
    return weights, _weight_evidence(frame, base, weights, target, boundary, candidate)


def incumbent_admission_boundary_mask(
    frame: pl.DataFrame,
    incumbent_probabilities: np.ndarray,
    *,
    minimum_edge: float = EXPECTED_BOUNDARY_MINIMUM_EDGE,
    maximum_edge: float = EXPECTED_BOUNDARY_MAXIMUM_EDGE,
    target_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Identify target-eligible rows with incumbent edge inside the closed 2–4¢ band."""

    required = {
        "seconds_elapsed",
        "yes_ask_vwap_5",
        "no_ask_vwap_5",
        "yes_cost_per_share",
        "no_cost_per_share",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("incumbent admission boundary is missing: " + ", ".join(missing))
    probability = np.asarray(incumbent_probabilities, dtype=np.float64)
    if probability.shape != (frame.height,) or not np.isfinite(probability).all():
        raise ValueError("incumbent admission probabilities are invalid")
    yes_cost = frame["yes_cost_per_share"].to_numpy().astype(np.float64)
    no_cost = frame["no_cost_per_share"].to_numpy().astype(np.float64)
    if not np.isfinite(yes_cost).all() or not np.isfinite(no_cost).all():
        raise ValueError("incumbent admission costs must be finite")
    selected_target = (
        np.ones(frame.height, dtype=bool)
        if target_mask is None
        else np.asarray(target_mask, dtype=bool)
    )
    if selected_target.shape != (frame.height,):
        raise ValueError("incumbent target mask is invalid")
    elapsed = frame["seconds_elapsed"].to_numpy().astype(np.int64)
    yes_price = frame["yes_ask_vwap_5"].to_numpy().astype(np.float64)
    no_price = frame["no_ask_vwap_5"].to_numpy().astype(np.float64)
    within_time = (elapsed >= EXPECTED_TARGET_TIME_BANDS[0][0]) & (
        elapsed <= EXPECTED_TARGET_TIME_BANDS[-1][1] - 5
    )
    yes_eligible = (
        within_time
        & (yes_price >= EXPECTED_TARGET_PRICE_BAND[0])
        & (yes_price < EXPECTED_TARGET_PRICE_BAND[1])
    )
    no_eligible = (
        within_time
        & (no_price >= EXPECTED_TARGET_PRICE_BAND[0])
        & (no_price < EXPECTED_TARGET_PRICE_BAND[1])
    )
    modeled_edge = np.maximum(
        np.where(yes_eligible, probability - yes_cost, -np.inf),
        np.where(no_eligible, 1.0 - probability - no_cost, -np.inf),
    )
    boundary_tolerance = 1e-12
    return (
        selected_target
        & (modeled_edge >= minimum_edge - boundary_tolerance)
        & (modeled_edge <= maximum_edge + boundary_tolerance)
    )


def fit_target_side_price_time_calibrators(
    model: FittedCoreModel,
    time_calibrators: tuple[TimeBandCalibrator, ...],
    frame: pl.DataFrame,
    asymmetric_config: AsymmetricValueConfig,
    *,
    target_price_band: tuple[float, float],
    target_time_bands: tuple[tuple[int, int], ...],
    target_sides: tuple[str, ...],
    minimum_markets_per_cell: int,
    minimum_days_per_cell: int,
    identity_l2: float,
    slope_bounds: tuple[float, float],
    intercept_bounds: tuple[float, float],
) -> tuple[tuple[AsymmetricCalibrationCell, ...], dict[str, Any]]:
    """Fit only eight coherent target cells; leave the other 152 cells at identity."""

    _validate_target_contract(
        asymmetric_config,
        target_price_band=target_price_band,
        target_time_bands=target_time_bands,
        target_sides=target_sides,
        minimum_markets_per_cell=minimum_markets_per_cell,
        minimum_days_per_cell=minimum_days_per_cell,
        identity_l2=identity_l2,
        slope_bounds=slope_bounds,
        intercept_bounds=intercept_bounds,
    )
    required = {
        "market_id",
        "window_start",
        "seconds_elapsed",
        "label_up",
        "yes_ask_vwap_5",
        "no_ask_vwap_5",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("target calibration frame is missing: " + ", ".join(missing))
    parent_yes = _time_calibrated_probability(model, time_calibrators, frame)
    parent_logit = _logit(parent_yes)
    elapsed = frame["seconds_elapsed"].to_numpy().astype(np.int64)
    market_ids = frame["market_id"].cast(pl.String).to_numpy()
    utc_days = frame["window_start"].dt.date().cast(pl.String).to_numpy()
    labels = frame["label_up"].to_numpy().astype(np.int8)
    yes_price_indices = _price_band_indices(frame["yes_ask_vwap_5"].to_numpy())
    no_price_indices = _price_band_indices(frame["no_ask_vwap_5"].to_numpy())
    target_price_index = round(target_price_band[0] / PRICE_BAND_WIDTH)
    target_band_set = set(target_time_bands)
    cells: list[AsymmetricCalibrationCell] = []
    optimizer_evidence: list[dict[str, Any]] = []
    for time_band in time_calibrators:
        time_key = (time_band.start_second, time_band.end_second_exclusive)
        selected_band = (elapsed >= time_key[0]) & (elapsed < time_key[1])
        band_logit = parent_logit[selected_band]
        band_labels = labels[selected_band]
        band_ids = market_ids[selected_band]
        band_days = utc_days[selected_band]
        band_yes_price = yes_price_indices[selected_band]
        band_no_price = no_price_indices[selected_band]
        band_weights = _market_equal_weights(band_ids)
        parameters = np.asarray((1.0, 0.0, 1.0, 0.0), dtype=np.float64)
        result = None
        if time_key in target_band_set:
            active_keys = ((target_price_index, 0), (target_price_index, 1))
            support = [
                _cell_support(
                    side=side,
                    price_index=target_price_index,
                    yes_price_indices=band_yes_price,
                    no_price_indices=band_no_price,
                    labels=band_labels,
                    market_ids=band_ids,
                    utc_days=band_days,
                )
                for side in target_sides
            ]
            _require_target_support(
                support,
                time_key=time_key,
                minimum_markets=minimum_markets_per_cell,
                minimum_days=minimum_days_per_cell,
            )
            penalty_weights = np.asarray(
                [
                    band_weights[
                        (band_yes_price if side == "YES" else band_no_price) == target_price_index
                    ].sum()
                    for side in target_sides
                ],
                dtype=np.float64,
            )
            bounds = (
                slope_bounds,
                intercept_bounds,
                slope_bounds,
                intercept_bounds,
            )
            result = minimize(
                _target_calibration_objective,
                parameters,
                args=(
                    band_logit,
                    band_labels.astype(np.float64),
                    band_weights,
                    band_yes_price,
                    band_no_price,
                    active_keys,
                    penalty_weights,
                    identity_l2,
                ),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-9, "maxls": 50},
            )
            if not result.success or not np.isfinite(result.x).all():
                raise RuntimeError(
                    f"target calibration {time_key[0]}-{time_key[1]} did not converge: "
                    f"{result.message}"
                )
            parameters = result.x
            optimizer_evidence.append(
                {
                    "start_second": time_key[0],
                    "end_second_exclusive": time_key[1],
                    "converged": True,
                    "iterations": int(result.nit),
                    "objective": float(result.fun),
                }
            )
        slopes = np.ones((PRICE_BAND_COUNT, 2), dtype=np.float64)
        intercepts = np.zeros_like(slopes)
        slopes[target_price_index] = parameters[0::2]
        intercepts[target_price_index] = parameters[1::2]
        coherent_probability = _coherent_probability(
            band_logit,
            band_yes_price,
            band_no_price,
            slopes,
            intercepts,
        )
        for price_index in range(PRICE_BAND_COUNT):
            for side_index, side in enumerate(EXPECTED_TARGET_SIDES):
                support = _cell_support(
                    side=side,
                    price_index=price_index,
                    yes_price_indices=band_yes_price,
                    no_price_indices=band_no_price,
                    labels=band_labels,
                    market_ids=band_ids,
                    utc_days=band_days,
                )
                is_target = time_key in target_band_set and price_index == target_price_index
                weighted_log_loss = None
                if is_target:
                    selected = support["selected"]
                    side_probability = (
                        coherent_probability[selected]
                        if side == "YES"
                        else 1.0 - coherent_probability[selected]
                    )
                    side_labels = support["side_labels"]
                    cell_weights = _market_equal_weights(band_ids[selected])
                    clipped = np.clip(side_probability, 1e-9, 1.0 - 1e-9)
                    weighted_log_loss = float(
                        -np.sum(
                            cell_weights
                            * (
                                side_labels * np.log(clipped)
                                + (1.0 - side_labels) * np.log(1.0 - clipped)
                            )
                        )
                    )
                cells.append(
                    AsymmetricCalibrationCell(
                        start_second=time_key[0],
                        end_second_exclusive=time_key[1],
                        minimum_price=price_index * PRICE_BAND_WIDTH,
                        maximum_price=(price_index + 1) * PRICE_BAND_WIDTH,
                        side=side,
                        slope=float(slopes[price_index, side_index]),
                        intercept=float(intercepts[price_index, side_index]),
                        fitted=is_target,
                        fallback=None if is_target else NON_TARGET_FALLBACK,
                        rows=support["rows"],
                        markets=support["markets"],
                        utc_days=support["utc_days"],
                        positives=support["positives"],
                        negatives=support["negatives"],
                        identity_l2_strength=identity_l2,
                        converged=is_target,
                        iterations=int(result.nit) if is_target and result is not None else 0,
                        objective=(float(result.fun) if is_target and result is not None else None),
                        weighted_log_loss=weighted_log_loss,
                    )
                )
    expected_cells = len(time_calibrators) * PRICE_BAND_COUNT * len(target_sides)
    if len(cells) != expected_cells:
        raise RuntimeError("target calibration did not materialize the runtime grid")
    target_cells = [cell for cell in cells if cell.fitted]
    non_target_cells = [cell for cell in cells if not cell.fitted]
    if (
        len(target_cells) != TARGET_CELL_COUNT
        or any(cell.fallback is not None or not cell.converged for cell in target_cells)
        or any(cell.fitted for cell in non_target_cells)
        or any(
            not math.isclose(cell.slope, 1.0) or not math.isclose(cell.intercept, 0.0)
            for cell in non_target_cells
        )
    ):
        raise RuntimeError("target calibration did not preserve its exact eight-cell contract")
    frozen_cells = tuple(cells)
    target_evidence = target_calibration_evidence(frozen_cells, asymmetric_config)
    if not target_evidence["qualified"]:
        raise RuntimeError("target calibration evidence did not qualify all eight cells")
    evidence = {
        "fitted_cells": len(target_cells),
        "identity_non_target_cells": len(non_target_cells),
        "target_cells_have_no_fallback": True,
        "non_target_behavior": NON_TARGET_FALLBACK,
        "identity_l2": identity_l2,
        "slope_bounds": list(slope_bounds),
        "intercept_bounds": list(intercept_bounds),
        "optimizers": optimizer_evidence,
        "target_contract": target_evidence,
        "cells_sha256": _canonical_sha256([asdict(cell) for cell in frozen_cells]),
    }
    return frozen_cells, evidence


def _fit_histogram_estimator(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    weights: np.ndarray,
    *,
    candidate: Any,
    estimator_contract: Any,
    random_seed: int,
    threads: int,
) -> FittedCoreModel:
    matrix = feature_matrix(frame, features)
    labels = frame["label_up"].to_numpy()
    medians = finite_medians(matrix)
    policy_inactive = tuple(
        feature for feature in TARGET_POLICY_INACTIVE_FEATURE_MATURITY if feature in features
    )
    for feature in policy_inactive:
        medians[features.index(feature)] = 0.0
    transformed = np.where(np.isfinite(matrix), matrix, medians)
    parameters = _histogram_parameters(estimator_contract)
    estimator = HistGradientBoostingClassifier(
        **parameters,
        early_stopping=False,
        random_state=random_seed,
    )
    with threadpool_limits(limits=threads):
        estimator.fit(transformed, labels, sample_weight=weights)
    if estimator.n_iter_ > estimator.max_iter:
        raise RuntimeError(f"{candidate.name} histogram estimator did not converge")
    boundary_contract = (
        {
            "minimum_edge": float(candidate.boundary_minimum_edge),
            "maximum_edge": float(candidate.boundary_maximum_edge),
            "closed": str(candidate.boundary_closed),
            "multiplier": float(candidate.boundary_multiplier),
            "weighting": str(candidate.weighting),
        }
        if bool(candidate.boundary_weighted)
        else None
    )
    return FittedCoreModel(
        candidate_name=str(candidate.name),
        family="histogram",
        feature_names=features,
        hyperparameters={
            **parameters,
            "target_weight": float(candidate.target_weight),
            "histogram_profile": "h3_regularized",
            "boundary_weighted": bool(candidate.boundary_weighted),
            "boundary_contract": boundary_contract,
        },
        imputation_medians=medians,
        standardization_means=None,
        standardization_scales=None,
        estimator=estimator,
        row_weight_policy=(
            BOUNDARY_WEIGHT_POLICY
            if bool(candidate.boundary_weighted)
            else HYBRID_MARKET_EQUAL_ROW_WEIGHT_POLICY
        ),
        row_weight_schedule=None,
        recency_half_life_days=None,
    )


def _validate_configured_contract(
    config: _WindowedEstimatorConfig,
    asymmetric_config: AsymmetricValueConfig,
    estimator_contract: Any,
) -> None:
    if config.asymmetric_value_config.resolve() != asymmetric_config.source_path.resolve():
        raise RuntimeError("conditional estimator did not load its pinned asymmetric config")
    if asymmetric_config.fit.end != config.calibration_fit.start:
        raise RuntimeError("conditional estimator fit must end at calibration start")
    if (
        asymmetric_config.calibration.start != config.calibration_fit.start
        or asymmetric_config.calibration.end != config.calibration_fit.end
    ):
        raise RuntimeError("conditional estimator calibration window changed")
    if (
        asymmetric_config.policy.start != config.matched_comparison.start
        or asymmetric_config.policy.end != config.matched_comparison.end
    ):
        raise RuntimeError("conditional estimator matched comparison window changed")
    _validate_estimator_contract(estimator_contract)


def _validate_estimator_contract(contract: Any) -> None:
    if tuple(str(item.name) for item in contract.candidates) != ESTIMATOR_FALLBACK_CANDIDATES:
        raise RuntimeError("conditional estimator candidate matrix changed")
    if int(contract.maximum_total_challengers) != 5:
        raise RuntimeError("conditional estimator total challenger budget changed")
    if _histogram_parameters(contract) != EXPECTED_HISTOGRAM_PARAMETERS:
        raise RuntimeError("conditional estimator H3 contract changed")
    first, second = contract.candidates
    if (
        not math.isclose(float(first.target_weight), EXPECTED_TARGET_WEIGHT)
        or bool(first.boundary_weighted)
        or not math.isclose(float(second.target_weight), EXPECTED_TARGET_WEIGHT)
        or not bool(second.boundary_weighted)
        or not math.isclose(float(second.boundary_minimum_edge), EXPECTED_BOUNDARY_MINIMUM_EDGE)
        or not math.isclose(float(second.boundary_maximum_edge), EXPECTED_BOUNDARY_MAXIMUM_EDGE)
        or str(second.boundary_closed) != "both"
        or not math.isclose(float(second.boundary_multiplier), EXPECTED_BOUNDARY_MULTIPLIER)
        or str(second.weighting) != "market_equal_renormalized"
    ):
        raise RuntimeError("conditional estimator E1/E2 contract changed")


def _validate_target_contract(
    config: AsymmetricValueConfig,
    *,
    target_price_band: tuple[float, float],
    target_time_bands: tuple[tuple[int, int], ...],
    target_sides: tuple[str, ...],
    minimum_markets_per_cell: int,
    minimum_days_per_cell: int,
    identity_l2: float,
    slope_bounds: tuple[float, float],
    intercept_bounds: tuple[float, float],
) -> None:
    if (
        target_price_band != EXPECTED_TARGET_PRICE_BAND
        or target_time_bands != EXPECTED_TARGET_TIME_BANDS
        or target_sides != EXPECTED_TARGET_SIDES
    ):
        raise RuntimeError("conditional estimator target calibration contract changed")
    target = config.target_calibration
    if target is None or (
        not math.isclose(target.minimum_price, target_price_band[0])
        or not math.isclose(target.maximum_price, target_price_band[1])
        or target.time_bands != target_time_bands
        or target.sides != target_sides
        or target.required_fitted_cells != TARGET_CELL_COUNT
    ):
        raise RuntimeError("asymmetric target calibration does not match fallback contract")
    if minimum_markets_per_cell < 1 or minimum_days_per_cell < 1:
        raise ValueError("target calibration support thresholds must be positive")
    if not np.isfinite(identity_l2) or identity_l2 <= 0.0:
        raise ValueError("target calibration identity L2 must be positive")
    if (
        not all(np.isfinite(value) for value in (*slope_bounds, *intercept_bounds))
        or slope_bounds[0] <= 0.0
        or slope_bounds[0] >= slope_bounds[1]
        or intercept_bounds[0] >= intercept_bounds[1]
    ):
        raise ValueError("target calibration parameter bounds are invalid")
    if not set(target_time_bands).issubset(set(config.calibration_bands)):
        raise RuntimeError("target calibration time bands are absent from runtime bands")


def _histogram_parameters(contract: Any) -> dict[str, Any]:
    histogram = contract.histogram
    return {
        "learning_rate": float(histogram.learning_rate),
        "max_iter": int(histogram.max_iter),
        "max_leaf_nodes": int(histogram.max_leaf_nodes),
        "min_samples_leaf": int(histogram.min_samples_leaf),
        "l2_regularization": float(histogram.l2_regularization),
    }


def _core_oracle_feature_contract() -> tuple[str, ...]:
    features = asymmetric_value_feature_sets()[CORE_ORACLE_PRICE]
    if len(features) != EXPECTED_CORE_ORACLE_PRICE_FEATURES or len(set(features)) != len(features):
        raise RuntimeError("Core+Oracle+Polymarket feature contract changed")
    return features


def _restore_market_weight_exposure(
    frame: pl.DataFrame,
    base: np.ndarray,
    adjusted: np.ndarray,
) -> np.ndarray:
    market_ids = frame["market_id"].cast(pl.String).to_numpy()
    _, inverse = np.unique(market_ids, return_inverse=True)
    output = np.asarray(adjusted, dtype=np.float64).copy()
    adjusted_totals = np.bincount(inverse, weights=output)
    base_totals = np.bincount(inverse, weights=base)
    if np.any(adjusted_totals <= 0.0) or np.any(base_totals <= 0.0):
        raise RuntimeError("boundary weighting produced invalid market exposure")
    output *= base_totals[inverse] / adjusted_totals[inverse]
    if (
        not np.isfinite(output).all()
        or np.any(output <= 0.0)
        or not math.isclose(float(output.sum()), float(base.sum()), rel_tol=0.0, abs_tol=1e-9)
    ):
        raise RuntimeError("boundary weighting did not preserve total market exposure")
    return output


def _weight_evidence(
    frame: pl.DataFrame,
    base: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
    boundary: np.ndarray,
    candidate: Any,
) -> dict[str, Any]:
    market_ids = frame["market_id"].cast(pl.String).to_numpy()
    _, inverse = np.unique(market_ids, return_inverse=True)
    market_deltas = np.abs(
        np.bincount(inverse, weights=weights) - np.bincount(inverse, weights=base)
    )
    total = float(weights.sum())
    return {
        "formula": (
            "hybrid50_market_equal"
            if not bool(candidate.boundary_weighted)
            else "hybrid50_market_equal * boundary_multiplier; preserve_base_market_total"
        ),
        "target_weight": EXPECTED_TARGET_WEIGHT,
        "source_rows": frame.height,
        "source_markets": frame["market_id"].n_unique(),
        "target_rows": int(target.sum()),
        "target_markets": frame.filter(pl.Series(target))["market_id"].n_unique(),
        "boundary_weighted": bool(candidate.boundary_weighted),
        "boundary_rows": int(boundary.sum()),
        "boundary_markets": (
            frame.filter(pl.Series(boundary))["market_id"].n_unique() if boundary.any() else 0
        ),
        "boundary_edge_interval": (
            [EXPECTED_BOUNDARY_MINIMUM_EDGE, EXPECTED_BOUNDARY_MAXIMUM_EDGE]
            if bool(candidate.boundary_weighted)
            else None
        ),
        "boundary_interval_closed": "both" if bool(candidate.boundary_weighted) else None,
        "boundary_multiplier": (
            EXPECTED_BOUNDARY_MULTIPLIER if bool(candidate.boundary_weighted) else None
        ),
        "market_exposure_renormalization": (
            "preserve_each_market_base_hybrid_total"
            if bool(candidate.boundary_weighted)
            else "not_required"
        ),
        "maximum_market_weight_total_delta": float(market_deltas.max(initial=0.0)),
        "weight_sum": total,
        "mean_weight": float(weights.mean()),
        "realized_weight_on_target_rows": float(weights[target].sum() / total),
        "realized_weight_on_boundary_rows": (
            float(weights[boundary].sum() / total) if boundary.any() else 0.0
        ),
    }


def _time_calibrated_probability(
    model: FittedCoreModel,
    calibrators: tuple[TimeBandCalibrator, ...],
    frame: pl.DataFrame,
) -> np.ndarray:
    logits = model.raw_logit(frame)
    elapsed = frame["seconds_elapsed"].to_numpy()
    output = np.full(frame.height, np.nan, dtype=np.float64)
    for band in calibrators:
        selected = (elapsed >= band.start_second) & (elapsed < band.end_second_exclusive)
        output[selected] = band.calibrator.probability(logits[selected])
    if not np.isfinite(output).all():
        raise RuntimeError("parent calibration does not cover the target calibration frame")
    return np.clip(output, 1e-9, 1.0 - 1e-9)


def _target_calibration_objective(
    parameters: np.ndarray,
    parent_logit: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    yes_price_indices: np.ndarray,
    no_price_indices: np.ndarray,
    active_keys: tuple[tuple[int, int], ...],
    penalty_weights: np.ndarray,
    identity_l2: float,
) -> tuple[float, np.ndarray]:
    slopes = np.ones((PRICE_BAND_COUNT, 2), dtype=np.float64)
    intercepts = np.zeros_like(slopes)
    for offset, key in enumerate(active_keys):
        slopes[key] = parameters[2 * offset]
        intercepts[key] = parameters[2 * offset + 1]
    yes_eta = parent_logit * slopes[yes_price_indices, 0] + intercepts[yes_price_indices, 0]
    no_eta = -parent_logit * slopes[no_price_indices, 1] + intercepts[no_price_indices, 1]
    eta = 0.5 * (yes_eta - no_eta)
    loss = float(np.sum(weights * (np.logaddexp(0.0, eta) - labels * eta)))
    delta = parameters.copy()
    delta[0::2] -= 1.0
    parameter_weights = np.repeat(penalty_weights, 2)
    penalty = 0.5 * identity_l2 * float((parameter_weights * delta) @ delta)
    error = weights * (_sigmoid(eta) - labels)
    gradient = np.empty_like(parameters)
    for offset, (price_index, side_index) in enumerate(active_keys):
        selected = (
            yes_price_indices == price_index if side_index == 0 else no_price_indices == price_index
        )
        intercept_sign = 1.0 if side_index == 0 else -1.0
        gradient[2 * offset] = (
            np.sum(error[selected] * 0.5 * parent_logit[selected])
            + identity_l2 * penalty_weights[offset] * delta[2 * offset]
        )
        gradient[2 * offset + 1] = (
            np.sum(error[selected] * 0.5 * intercept_sign)
            + identity_l2 * penalty_weights[offset] * delta[2 * offset + 1]
        )
    return loss + penalty, gradient


def _coherent_probability(
    parent_logit: np.ndarray,
    yes_price_indices: np.ndarray,
    no_price_indices: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    yes_eta = parent_logit * slopes[yes_price_indices, 0] + intercepts[yes_price_indices, 0]
    no_eta = -parent_logit * slopes[no_price_indices, 1] + intercepts[no_price_indices, 1]
    return _sigmoid(0.5 * (yes_eta - no_eta))


def _cell_support(
    *,
    side: str,
    price_index: int,
    yes_price_indices: np.ndarray,
    no_price_indices: np.ndarray,
    labels: np.ndarray,
    market_ids: np.ndarray,
    utc_days: np.ndarray,
) -> dict[str, Any]:
    selected = (
        yes_price_indices == price_index if side == "YES" else no_price_indices == price_index
    )
    side_labels = labels[selected] if side == "YES" else 1 - labels[selected]
    rows = int(selected.sum())
    positives = int(side_labels.sum()) if rows else 0
    return {
        "side": side,
        "selected": selected,
        "side_labels": side_labels.astype(np.float64),
        "rows": rows,
        "markets": int(np.unique(market_ids[selected]).size),
        "utc_days": int(np.unique(utc_days[selected]).size),
        "positives": positives,
        "negatives": rows - positives,
    }


def _require_target_support(
    support: list[dict[str, Any]],
    *,
    time_key: tuple[int, int],
    minimum_markets: int,
    minimum_days: int,
) -> None:
    failures = []
    for item in support:
        reasons = []
        if item["markets"] < minimum_markets:
            reasons.append("markets")
        if item["utc_days"] < minimum_days:
            reasons.append("days")
        if item["positives"] == 0 or item["negatives"] == 0:
            reasons.append("single_class")
        if reasons:
            failures.append(f"{item['side']}:{'+'.join(reasons)}")
    if failures:
        raise RuntimeError(
            f"target calibration {time_key[0]}-{time_key[1]} support failed: " + ", ".join(failures)
        )


def _market_equal_weights(market_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(market_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("market-equal weighting requires non-empty market IDs")
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights / weights.sum()


def _price_band_indices(values: np.ndarray) -> np.ndarray:
    prices = np.asarray(values, dtype=np.float64)
    if prices.ndim != 1 or not np.isfinite(prices).all():
        raise ValueError("calibration prices must be finite")
    if np.any((prices < 0.0) | (prices > 1.0)):
        raise ValueError("calibration prices must be inside [0, 1]")
    return np.clip(
        np.floor(prices * PRICE_BAND_COUNT + 1e-12),
        0,
        PRICE_BAND_COUNT - 1,
    ).astype(np.int16)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -700.0, 700.0)))


def _window(frame: pl.DataFrame, window: EvidenceWindow) -> pl.DataFrame:
    return frame.filter(pl.col("window_start").is_between(window.start, window.end, closed="left"))


def _window_evidence(window: EvidenceWindow) -> dict[str, str]:
    return {"start": window.start.isoformat(), "end_exclusive": window.end.isoformat()}


def _validate_unique_grid(frame: pl.DataFrame, *, cohort: str) -> None:
    keys = ("market_id", "window_start", "seconds_elapsed", "observed_at")
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ValueError(f"{cohort} grid is missing: " + ", ".join(missing))
    if frame.select(*keys).is_duplicated().any():
        raise ValueError(f"{cohort} grid contains duplicate market-second keys")


def _candidate_evidence(candidate: Any) -> dict[str, Any]:
    return {
        "name": str(candidate.name),
        "target_weight": float(candidate.target_weight),
        "boundary_weighted": bool(candidate.boundary_weighted),
        "boundary_minimum_edge": (
            float(candidate.boundary_minimum_edge)
            if candidate.boundary_minimum_edge is not None
            else None
        ),
        "boundary_maximum_edge": (
            float(candidate.boundary_maximum_edge)
            if candidate.boundary_maximum_edge is not None
            else None
        ),
        "boundary_closed": candidate.boundary_closed,
        "boundary_multiplier": (
            float(candidate.boundary_multiplier)
            if candidate.boundary_multiplier is not None
            else None
        ),
        "weighting": candidate.weighting,
    }


def _grid_key_sha256(frame: pl.DataFrame) -> str:
    required = ("market_id", "window_start", "observed_at", "seconds_elapsed")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError("prediction key digest is missing: " + ", ".join(missing))
    ordered = frame.select(*required).sort(*required)
    digest = hashlib.sha256(b"btc-asymmetric-estimator-grid-key-v1\n")
    for row in ordered.iter_rows():
        for value in row:
            rendered = value.isoformat() if isinstance(value, datetime) else str(value)
            encoded = rendered.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _frame_content_sha256(
    frame: pl.DataFrame,
    *,
    features: tuple[str, ...],
    incumbent_probabilities: np.ndarray | None,
) -> str:
    ordered = frame.with_row_index("__content_index").sort(
        "window_start", "market_id", "seconds_elapsed", "observed_at"
    )
    order = ordered["__content_index"].to_numpy().astype(np.int64)
    ordered = ordered.drop("__content_index")
    digest = hashlib.sha256(b"btc-asymmetric-estimator-frame-content-v1\n")
    digest.update(_grid_key_sha256(ordered).encode())
    for column in (
        "label_up",
        "yes_ask_vwap_5",
        "no_ask_vwap_5",
        "yes_cost_per_share",
        "no_cost_per_share",
        *features,
    ):
        digest.update(column.encode())
        values = ordered[column].cast(pl.Float64).to_numpy().astype("<f8", copy=False)
        normalized = np.where(np.isnan(values), np.nan, values).astype("<f8", copy=False)
        digest.update(normalized.tobytes())
    if incumbent_probabilities is not None:
        aligned = np.asarray(incumbent_probabilities, dtype=np.float64)[order]
        digest.update(b"incumbent_probability")
        digest.update(aligned.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _float_vector_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8")
    digest = hashlib.sha256(b"btc-asymmetric-estimator-float-vector-v1\n")
    digest.update(array.shape[0].to_bytes(8, "big"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _bundle_semantic_sha256(bundle: AsymmetricValueModel) -> str:
    digest = hashlib.sha256(b"btc-asymmetric-estimator-bundle-v1\n")
    model = bundle.model
    digest.update(
        _canonical_bytes(
            {
                "name": bundle.name,
                "candidate_name": model.candidate_name,
                "family": model.family,
                "feature_names": list(model.feature_names),
                "hyperparameters": model.hyperparameters,
                "row_weight_policy": model.row_weight_policy,
                "parent_calibration_source": bundle.parent_calibration_source,
                "identity_l2_strength": bundle.identity_l2_strength,
                "time_calibrators": [asdict(item) for item in bundle.time_calibrators],
                "cells": [asdict(item) for item in bundle.cells],
            }
        )
    )
    digest.update(np.asarray(model.imputation_medians, dtype="<f8").tobytes())
    estimator = model.estimator
    digest.update(np.asarray(estimator._baseline_prediction, dtype="<f8").tobytes())
    digest.update(np.asarray(estimator.classes_, dtype="<i8").tobytes())
    for iteration in estimator._predictors:
        predictor = iteration[0]
        for field in predictor.nodes.dtype.names or ():
            digest.update(field.encode())
            values = np.ascontiguousarray(predictor.nodes[field])
            digest.update(str(values.dtype).encode())
            digest.update(values.tobytes())
        for attribute in ("binned_left_cat_bitsets", "raw_left_cat_bitsets"):
            values = np.ascontiguousarray(getattr(predictor, attribute))
            digest.update(attribute.encode())
            digest.update(str(values.dtype).encode())
            digest.update(values.tobytes())
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def implementation_sha256() -> str:
    """Return the source digest for run-level provenance."""

    return file_sha256(Path(__file__))
