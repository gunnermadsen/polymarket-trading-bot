"""Champion-family probability models for lower-cost asymmetric opportunities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .asymmetric_value_config import AsymmetricValueConfig
from .asymmetric_value_data import (
    EARLY_CAUSAL_ORACLE_FEATURES,
    POLYMARKET_VALUE_FEATURES,
)
from .chainlink_oi_features import CHAINLINK_CANDLE_FEATURES
from .core_config import CoreTrainingConfig
from .core_features import CORE_ENRICHED_FEATURES
from .core_training import (
    MARKET_EQUAL_ROW_WEIGHT_POLICY,
    CandidateSpec,
    FittedCoreModel,
    ProbabilityCalibrator,
    fit_model,
)
from .early_value_training import (
    TimeBandCalibrator,
    probability_metrics,
)
from .spot_l2_chainlink_features import L2_FEATURES

PRICE_LOGISTIC = "price_logistic_control"
CORE_PRICE = "core_price_hgb"
L2_MATCHED_CORE_PRICE_CONTROL = "l2_matched_core_price_hgb_control"
CORE_L2_PRICE = "core_l2_price_hgb"
CORE_CANDLES_PRICE = "core_chainlink_candles_price_hgb"
CORE_ORACLE_PRICE = "core_oracle_price_hgb"
ORACLE_MATCHED_CORE_PRICE_CONTROL = "oracle_matched_core_price_hgb_control"

ASYMMETRIC_VALUE_CANDIDATES = (
    PRICE_LOGISTIC,
    CORE_PRICE,
    L2_MATCHED_CORE_PRICE_CONTROL,
    CORE_L2_PRICE,
    CORE_CANDLES_PRICE,
    ORACLE_MATCHED_CORE_PRICE_CONTROL,
    CORE_ORACLE_PRICE,
)

MODEL_SELECTION_ELIGIBLE = frozenset(
    set(ASYMMETRIC_VALUE_CANDIDATES)
    - {
        L2_MATCHED_CORE_PRICE_CONTROL,
        ORACLE_MATCHED_CORE_PRICE_CONTROL,
    }
)
ORACLE_FEATURE_CANDIDATES = frozenset({CORE_ORACLE_PRICE})
PRICE_BAND_WIDTH = 0.10
PRICE_BAND_COUNT = 10


@dataclass(frozen=True)
class AsymmetricCalibrationCell:
    start_second: int
    end_second_exclusive: int
    minimum_price: float
    maximum_price: float
    side: str
    slope: float
    intercept: float
    fitted: bool
    fallback: str | None
    rows: int
    markets: int
    utc_days: int
    positives: int
    negatives: int
    identity_l2_strength: float
    converged: bool
    iterations: int
    objective: float | None
    weighted_log_loss: float | None


@dataclass
class AsymmetricValueModel:
    """Direction model with parent time calibration and side/price corrections."""

    name: str
    model: FittedCoreModel
    time_calibrators: tuple[TimeBandCalibrator, ...]
    cells: tuple[AsymmetricCalibrationCell, ...]

    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        parent_yes = _time_calibrated_probability(
            self.model,
            self.time_calibrators,
            frame,
            model_name=self.name,
        )
        elapsed = frame["seconds_elapsed"].to_numpy()
        time_index = _time_band_indices(elapsed, self.time_calibrators)
        yes_price_index = _price_band_indices(frame["yes_ask_vwap_5"].to_numpy())
        no_price_index = _price_band_indices(frame["no_ask_vwap_5"].to_numpy())
        slopes, intercepts = _cell_parameter_arrays(self.cells, self.time_calibrators)
        yes_logit = _logit(parent_yes)
        no_logit = _logit(1.0 - parent_yes)
        yes_eta = (
            yes_logit * slopes[time_index, yes_price_index, 0]
            + intercepts[time_index, yes_price_index, 0]
        )
        no_eta = (
            no_logit * slopes[time_index, no_price_index, 1]
            + intercepts[time_index, no_price_index, 1]
        )
        coherent_eta = 0.5 * (yes_eta - no_eta)
        if not np.isfinite(coherent_eta).all():
            raise RuntimeError(f"{self.name} produced invalid side calibration")
        return np.clip(_sigmoid(coherent_eta), 1e-9, 1.0 - 1e-9)


def asymmetric_value_feature_sets() -> dict[str, tuple[str, ...]]:
    core = tuple(CORE_ENRICHED_FEATURES)
    price = tuple(POLYMARKET_VALUE_FEATURES)
    oracle = tuple(EARLY_CAUSAL_ORACLE_FEATURES)
    price_control = tuple(dict.fromkeys(("seconds_elapsed_scaled", *price)))
    return {
        PRICE_LOGISTIC: price_control,
        CORE_PRICE: tuple(dict.fromkeys((*core, *price))),
        L2_MATCHED_CORE_PRICE_CONTROL: tuple(dict.fromkeys((*core, *price))),
        CORE_L2_PRICE: tuple(dict.fromkeys((*core, *L2_FEATURES, *price))),
        CORE_CANDLES_PRICE: tuple(
            dict.fromkeys((*core, *CHAINLINK_CANDLE_FEATURES, *price))
        ),
        ORACLE_MATCHED_CORE_PRICE_CONTROL: tuple(
            dict.fromkeys((*core, *price))
        ),
        CORE_ORACLE_PRICE: tuple(dict.fromkeys((*core, *oracle, *price))),
    }


def fit_asymmetric_value_models(
    model_frames: dict[str, pl.DataFrame],
    config: AsymmetricValueConfig,
    core_config: CoreTrainingConfig,
) -> tuple[dict[str, AsymmetricValueModel], dict[str, Any]]:
    required = {"market_id", "window_start", "seconds_elapsed", "label_up"}
    expected = set(ASYMMETRIC_VALUE_CANDIDATES)
    if set(model_frames) != expected:
        raise ValueError("asymmetric-value model frames do not match the frozen candidates")
    for name, frame in model_frames.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} frame is missing columns: " + ", ".join(missing))

    histogram = asdict(core_config.model.histogram_candidates[0])
    feature_sets = asymmetric_value_feature_sets()
    models: dict[str, AsymmetricValueModel] = {}
    summary: dict[str, Any] = {
        "selection_metric": "economic_policy_contract",
        "accuracy_threshold_used_for_selection": False,
        "market_equal_row_weights": True,
        "parent_time_calibration_weighting": (
            "market equal independently within each causal time band"
        ),
        "earliest_decision_second": 1,
        "prediction_grid_points_per_market": len(config.prediction_seconds),
        "unavailable_horizons": "fit-only median imputation; no future filling",
        "opening_boundary_features_used": False,
        "opening_boundary_exclusion_reason": (
            "historical opening-boundary facts lack a proven decision-time availability timestamp"
        ),
        "oracle_feature_contract": list(EARLY_CAUSAL_ORACLE_FEATURES),
        "profiles": {},
    }
    for name in ASYMMETRIC_VALUE_CANDIDATES:
        scoring_source = model_frames[name]
        training_source = scoring_source
        fit_frame = _window(training_source, config.fit.start, config.fit.end)
        calibration_frame = _window(
            scoring_source,
            config.calibration.start,
            config.calibration.end,
        )
        policy_frame = _window(scoring_source, config.policy.start, config.policy.end)
        if any(item.is_empty() for item in (fit_frame, calibration_frame, policy_frame)):
            raise RuntimeError(f"{name} fit, calibration, and policy frames must be non-empty")
        features, feature_availability = _causal_feature_availability(
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
        calibrators = fit_asymmetric_time_band_calibrators(
            fitted,
            calibration_frame,
            config,
            core_config=core_config,
        )
        cells = fit_side_price_time_calibrators(
            fitted,
            calibrators,
            calibration_frame,
            config,
        )
        target_calibration = target_calibration_evidence(cells, config)
        bundle = AsymmetricValueModel(
            name=name,
            model=fitted,
            time_calibrators=calibrators,
            cells=cells,
        )
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
            "feature_availability": feature_availability,
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
            "side_price_time_calibration": {
                "price_band_width": PRICE_BAND_WIDTH,
                "minimum_markets_per_cell": (
                    config.gates.minimum_calibration_markets_per_cell
                ),
                "minimum_utc_days_per_cell": (
                    config.gates.minimum_calibration_days_per_cell
                ),
                "identity_l2_strength": config.calibration_identity_l2,
                "identity_l2_normalization": (
                    "market-equal time-band weight exposure per side/price cell"
                ),
                "fitted_cells": sum(cell.fitted for cell in cells),
                "fallback_cells": sum(not cell.fitted for cell in cells),
                "target_contract": target_calibration,
                "cells": [asdict(cell) for cell in cells],
            },
        }
        models[name] = bundle
    return models, summary


def target_calibration_evidence(
    cells: tuple[AsymmetricCalibrationCell, ...],
    config: AsymmetricValueConfig,
) -> dict[str, Any]:
    """Report whether every policy-driving calibration cell was genuinely fitted."""

    target = config.target_calibration
    if target is None:
        return {
            "required": False,
            "qualified": True,
            "required_fitted_cells": 0,
            "fitted_cells": 0,
            "fallback_cells": 0,
            "cells": [],
        }

    indexed = {
        (
            cell.start_second,
            cell.end_second_exclusive,
            round(cell.minimum_price, 10),
            round(cell.maximum_price, 10),
            cell.side,
        ): cell
        for cell in cells
    }
    evidence: list[dict[str, Any]] = []
    for start, end in target.time_bands:
        for side in target.sides:
            key = (
                start,
                end,
                round(target.minimum_price, 10),
                round(target.maximum_price, 10),
                side,
            )
            cell = indexed.get(key)
            failure_reasons: list[str] = []
            if cell is None:
                failure_reasons.append("missing_cell")
                evidence.append(
                    {
                        "start_second": start,
                        "end_second_exclusive": end,
                        "minimum_price": target.minimum_price,
                        "maximum_price": target.maximum_price,
                        "side": side,
                        "present": False,
                        "fitted": False,
                        "fallback": "missing_cell",
                        "markets": 0,
                        "utc_days": 0,
                        "positives": 0,
                        "negatives": 0,
                        "failure_reasons": failure_reasons,
                        "passed": False,
                    }
                )
                continue
            if cell.markets < config.gates.minimum_calibration_markets_per_cell:
                failure_reasons.append("insufficient_markets")
            if cell.utc_days < config.gates.minimum_calibration_days_per_cell:
                failure_reasons.append("insufficient_utc_days")
            if cell.positives <= 0 or cell.negatives <= 0:
                failure_reasons.append("single_class")
            if not cell.fitted:
                failure_reasons.append("parent_fallback")
            if cell.fallback is not None:
                failure_reasons.append(f"fallback:{cell.fallback}")
            if not cell.converged:
                failure_reasons.append("optimizer_not_converged")
            if cell.objective is None or not np.isfinite(cell.objective):
                failure_reasons.append("invalid_objective")
            if cell.weighted_log_loss is None or not np.isfinite(
                cell.weighted_log_loss
            ):
                failure_reasons.append("invalid_weighted_log_loss")
            evidence.append(
                {
                    "start_second": start,
                    "end_second_exclusive": end,
                    "minimum_price": target.minimum_price,
                    "maximum_price": target.maximum_price,
                    "side": side,
                    "present": True,
                    "fitted": cell.fitted,
                    "fallback": cell.fallback,
                    "markets": cell.markets,
                    "utc_days": cell.utc_days,
                    "positives": cell.positives,
                    "negatives": cell.negatives,
                    "failure_reasons": failure_reasons,
                    "passed": not failure_reasons,
                }
            )
    fitted_cells = sum(bool(item["passed"]) for item in evidence)
    qualified = bool(
        len(evidence) == target.required_fitted_cells
        and fitted_cells == target.required_fitted_cells
    )
    return {
        "required": True,
        "qualified": qualified,
        "required_fitted_cells": target.required_fitted_cells,
        "fitted_cells": fitted_cells,
        "fallback_cells": len(evidence) - fitted_cells,
        "minimum_markets_per_cell": (
            config.gates.minimum_calibration_markets_per_cell
        ),
        "minimum_utc_days_per_cell": (
            config.gates.minimum_calibration_days_per_cell
        ),
        "both_outcomes_required": True,
        "cells": evidence,
    }


def target_calibration_gate_checks(
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Translate target-cell evidence into explicit model qualification checks."""

    target = profile["side_price_time_calibration"]["target_contract"]
    if not target["required"]:
        return []
    checks = [
        {
            "name": "target_calibration_cells_genuinely_fitted",
            "observed": int(target["fitted_cells"]),
            "threshold": int(target["required_fitted_cells"]),
            "operator": "==",
            "passed": bool(target["qualified"]),
            "fallback_cells": int(target["fallback_cells"]),
        }
    ]
    checks.extend(
        {
            "name": (
                "target_calibration_cell_"
                f"{str(cell['side']).lower()}_"
                f"{int(cell['start_second'])}_{int(cell['end_second_exclusive'])}_"
                "20_30c"
            ),
            "observed": bool(cell["passed"]),
            "threshold": True,
            "operator": "==",
            "passed": bool(cell["passed"]),
            "markets": int(cell["markets"]),
            "utc_days": int(cell["utc_days"]),
            "positives": int(cell["positives"]),
            "negatives": int(cell["negatives"]),
            "fallback": cell["fallback"],
            "failure_reasons": list(cell["failure_reasons"]),
        }
        for cell in target["cells"]
    )
    return checks


def _causal_feature_availability(
    fit_frame: pl.DataFrame,
    candidates: tuple[str, ...],
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    earliest = fit_frame.filter(pl.col("seconds_elapsed") == 1)
    if earliest.is_empty():
        raise RuntimeError("asymmetric-value fit evidence lacks second-1 rows")
    missing = sorted(set(candidates) - set(earliest.columns))
    if missing:
        raise RuntimeError(
            "asymmetric-value earliest feature audit is missing: " + ", ".join(missing)
        )
    earliest_matrix = earliest.select(
        pl.col(list(candidates)).cast(pl.Float64)
    ).to_numpy()
    fit_matrix = fit_frame.select(
        pl.col(list(candidates)).cast(pl.Float64)
    ).to_numpy()
    earliest_fractions = np.isfinite(earliest_matrix).mean(axis=0)
    fit_fractions = np.isfinite(fit_matrix).mean(axis=0)
    availability = {
        feature: {
            "second_1_finite_fraction": float(earliest_fraction),
            "fit_finite_fraction": float(fit_fraction),
            "second_1_imputed_fraction": float(1.0 - earliest_fraction),
            "fit_imputed_fraction": float(1.0 - fit_fraction),
            "retained": True,
        }
        for feature, earliest_fraction, fit_fraction in zip(
            candidates,
            earliest_fractions,
            fit_fractions,
            strict=True,
        )
    }
    unavailable = [
        feature
        for feature in candidates
        if availability[feature]["fit_finite_fraction"] <= 0.0
    ]
    if unavailable:
        raise RuntimeError(
            "asymmetric-value features are nonfinite throughout fitting: "
            + ", ".join(unavailable)
        )
    return candidates, availability


def fit_asymmetric_time_band_calibrators(
    model: FittedCoreModel,
    frame: pl.DataFrame,
    config: AsymmetricValueConfig,
    *,
    core_config: CoreTrainingConfig,
) -> tuple[TimeBandCalibrator, ...]:
    """Fit parent Platt scaling with market equality local to each time band."""

    logits = model.raw_logit(frame)
    labels = frame["label_up"].to_numpy()
    elapsed = frame["seconds_elapsed"].to_numpy()
    market_ids = frame["market_id"].cast(pl.String).to_numpy()
    fitted: list[TimeBandCalibrator] = []
    for start, end in config.calibration_bands:
        selected = (elapsed >= start) & (elapsed < end)
        if selected.sum() < 20 or np.unique(labels[selected]).size != 2:
            raise RuntimeError(
                f"calibration band {start}-{end} lacks two-class evidence"
            )
        weights = _market_equal_weights_for_ids(market_ids[selected])
        estimator = LogisticRegression(
            C=1_000_000,
            solver="lbfgs",
            max_iter=500,
            tol=1e-9,
            random_state=config.random_seed,
        )
        with threadpool_limits(limits=core_config.compute.threads_per_fit):
            estimator.fit(
                logits[selected].reshape(-1, 1),
                labels[selected],
                sample_weight=weights,
            )
        fitted.append(
            TimeBandCalibrator(
                start_second=start,
                end_second_exclusive=end,
                calibrator=ProbabilityCalibrator(
                    slope=float(estimator.coef_[0, 0]),
                    intercept=float(estimator.intercept_[0]),
                    converged=bool(estimator.n_iter_[0] < estimator.max_iter),
                    iterations=int(estimator.n_iter_[0]),
                ),
                rows=int(selected.sum()),
                markets=int(np.unique(market_ids[selected]).size),
            )
        )
    return tuple(fitted)


def fit_side_price_time_calibrators(
    model: FittedCoreModel,
    time_calibrators: tuple[TimeBandCalibrator, ...],
    frame: pl.DataFrame,
    config: AsymmetricValueConfig,
) -> tuple[AsymmetricCalibrationCell, ...]:
    """Fit coherent monotone side/price corrections by causal time band."""

    parent_yes = _time_calibrated_probability(
        model,
        time_calibrators,
        frame,
        model_name=model.candidate_name,
    )
    elapsed = frame["seconds_elapsed"].to_numpy()
    market_ids = frame["market_id"].cast(pl.String).to_numpy()
    utc_days = frame["window_start"].dt.date().cast(pl.String).to_numpy()
    labels_yes = frame["label_up"].to_numpy().astype(np.int8)
    parent_logit = _logit(parent_yes)
    yes_price_indices = _price_band_indices(
        frame["yes_ask_vwap_5"].to_numpy()
    )
    no_price_indices = _price_band_indices(
        frame["no_ask_vwap_5"].to_numpy()
    )
    fitted: list[AsymmetricCalibrationCell] = []
    for time_band in time_calibrators:
        time_mask = (elapsed >= time_band.start_second) & (
            elapsed < time_band.end_second_exclusive
        )
        band_logit = parent_logit[time_mask]
        band_labels = labels_yes[time_mask]
        band_ids = market_ids[time_mask]
        band_days = utc_days[time_mask]
        band_yes_prices = yes_price_indices[time_mask]
        band_no_prices = no_price_indices[time_mask]
        band_weights = _market_equal_weights_for_ids(band_ids)
        cell_records: list[dict[str, Any]] = []
        active_keys: list[tuple[int, int]] = []
        for price_index in range(PRICE_BAND_COUNT):
            minimum_price = price_index * PRICE_BAND_WIDTH
            maximum_price = (price_index + 1) * PRICE_BAND_WIDTH
            for side_index, side in enumerate(("YES", "NO")):
                price_indices = (
                    band_yes_prices if side == "YES" else band_no_prices
                )
                cell_labels = (
                    band_labels if side == "YES" else 1 - band_labels
                )
                selected = price_indices == price_index
                selected_labels = cell_labels[selected]
                cell_ids = band_ids[selected]
                cell_days = band_days[selected]
                rows = int(selected.sum())
                markets = int(np.unique(cell_ids).size)
                days = int(np.unique(cell_days).size)
                positives = int(selected_labels.sum()) if rows else 0
                negatives = rows - positives
                fallback_reasons: list[str] = []
                if markets < config.gates.minimum_calibration_markets_per_cell:
                    fallback_reasons.append("insufficient_markets")
                if days < config.gates.minimum_calibration_days_per_cell:
                    fallback_reasons.append("insufficient_utc_days")
                if positives == 0 or negatives == 0:
                    fallback_reasons.append("single_class")
                record = {
                    "price_index": price_index,
                    "side_index": side_index,
                    "side": side,
                    "minimum_price": minimum_price,
                    "maximum_price": maximum_price,
                    "selected": selected,
                    "labels": selected_labels.astype(np.float64),
                    "ids": cell_ids,
                    "rows": rows,
                    "markets": markets,
                    "days": days,
                    "positives": positives,
                    "negatives": negatives,
                    "fallback": "+".join(fallback_reasons) or None,
                }
                cell_records.append(record)
                if not fallback_reasons:
                    active_keys.append((price_index, side_index))

        slopes = np.ones((PRICE_BAND_COUNT, 2), dtype=np.float64)
        intercepts = np.zeros_like(slopes)
        result = None
        optimizer_valid = True
        if active_keys:
            penalty_weights = np.asarray(
                [
                    np.sum(
                        band_weights[
                            (
                                band_yes_prices
                                if side_index == 0
                                else band_no_prices
                            )
                            == price_index
                        ]
                    )
                    for price_index, side_index in active_keys
                ],
                dtype=np.float64,
            )
            initial = np.tile(
                np.asarray((1.0, 0.0), dtype=np.float64),
                len(active_keys),
            )
            result = minimize(
                _coherent_calibration_objective,
                initial,
                args=(
                    band_logit,
                    band_labels.astype(np.float64),
                    band_weights,
                    band_yes_prices,
                    band_no_prices,
                    tuple(active_keys),
                    penalty_weights,
                    config.calibration_identity_l2,
                ),
                method="L-BFGS-B",
                jac=True,
                bounds=tuple(
                    bound
                    for _ in active_keys
                    for bound in ((0.0, None), (None, None))
                ),
                options={
                    "maxiter": 500,
                    "ftol": 1e-9,
                    "gtol": 1e-9,
                    "maxls": 50,
                },
            )
            optimizer_valid = bool(
                result.success
                and np.isfinite(result.x).all()
                and np.all(result.x[0::2] >= 0.0)
            )
            if optimizer_valid:
                for offset, key in enumerate(active_keys):
                    slopes[key] = float(result.x[2 * offset])
                    intercepts[key] = float(result.x[2 * offset + 1])

        coherent_probability = _coherent_probability_from_parameters(
            band_logit,
            band_yes_prices,
            band_no_prices,
            slopes,
            intercepts,
        )
        for record in cell_records:
            fallback = record["fallback"]
            key = (record["price_index"], record["side_index"])
            if fallback is not None or not optimizer_valid:
                fitted.append(
                    _fallback_cell(
                        time_band,
                        record["minimum_price"],
                        record["maximum_price"],
                        record["side"],
                        record["rows"],
                        record["markets"],
                        record["days"],
                        record["positives"],
                        record["negatives"],
                        config.calibration_identity_l2,
                        fallback or "optimizer_not_converged",
                    )
                )
                continue
            selected = record["selected"]
            side_probability = (
                coherent_probability[selected]
                if record["side"] == "YES"
                else 1.0 - coherent_probability[selected]
            )
            weights = _market_equal_weights_for_ids(record["ids"])
            cell_labels = record["labels"]
            clipped = np.clip(side_probability, 1e-9, 1.0 - 1e-9)
            weighted_loss = float(
                -np.sum(
                    weights
                    * (
                        cell_labels * np.log(clipped)
                        + (1.0 - cell_labels) * np.log(1.0 - clipped)
                    )
                )
            )
            fitted.append(
                AsymmetricCalibrationCell(
                    start_second=time_band.start_second,
                    end_second_exclusive=time_band.end_second_exclusive,
                    minimum_price=record["minimum_price"],
                    maximum_price=record["maximum_price"],
                    side=record["side"],
                    slope=float(slopes[key]),
                    intercept=float(intercepts[key]),
                    fitted=True,
                    fallback=None,
                    rows=record["rows"],
                    markets=record["markets"],
                    utc_days=record["days"],
                    positives=record["positives"],
                    negatives=record["negatives"],
                    identity_l2_strength=config.calibration_identity_l2,
                    converged=True,
                    iterations=int(result.nit) if result is not None else 0,
                    objective=float(result.fun) if result is not None else None,
                    weighted_log_loss=weighted_loss,
                )
            )
    expected = len(time_calibrators) * PRICE_BAND_COUNT * 2
    if len(fitted) != expected:
        raise RuntimeError("side/price/time calibration did not materialize every frozen cell")
    return tuple(fitted)


def _fallback_cell(
    time_band: TimeBandCalibrator,
    minimum_price: float,
    maximum_price: float,
    side: str,
    rows: int,
    markets: int,
    utc_days: int,
    positives: int,
    negatives: int,
    identity_l2_strength: float,
    reason: str,
) -> AsymmetricCalibrationCell:
    return AsymmetricCalibrationCell(
        start_second=time_band.start_second,
        end_second_exclusive=time_band.end_second_exclusive,
        minimum_price=minimum_price,
        maximum_price=maximum_price,
        side=side,
        slope=1.0,
        intercept=0.0,
        fitted=False,
        fallback=reason,
        rows=rows,
        markets=markets,
        utc_days=utc_days,
        positives=positives,
        negatives=negatives,
        identity_l2_strength=identity_l2_strength,
        converged=False,
        iterations=0,
        objective=None,
        weighted_log_loss=None,
    )


def _time_calibrated_probability(
    model: FittedCoreModel,
    calibrators: tuple[TimeBandCalibrator, ...],
    frame: pl.DataFrame,
    *,
    model_name: str,
) -> np.ndarray:
    logits = model.raw_logit(frame)
    elapsed = frame["seconds_elapsed"].to_numpy()
    output = np.full(frame.height, np.nan, dtype=np.float64)
    for band in calibrators:
        selected = (elapsed >= band.start_second) & (
            elapsed < band.end_second_exclusive
        )
        output[selected] = band.calibrator.probability(logits[selected])
    if not np.isfinite(output).all():
        raise RuntimeError(f"{model_name} time calibration does not cover all rows")
    return np.clip(output, 1e-9, 1.0 - 1e-9)


def _time_band_indices(
    elapsed: np.ndarray,
    calibrators: tuple[TimeBandCalibrator, ...],
) -> np.ndarray:
    indices = np.full(len(elapsed), -1, dtype=np.int16)
    for index, band in enumerate(calibrators):
        selected = (elapsed >= band.start_second) & (
            elapsed < band.end_second_exclusive
        )
        indices[selected] = index
    if np.any(indices < 0):
        raise RuntimeError("time calibration indices do not cover the prediction frame")
    return indices


def _price_band_indices(values: Any) -> np.ndarray:
    prices = np.asarray(values, dtype=np.float64)
    if prices.ndim != 1 or not np.isfinite(prices).all():
        raise ValueError("calibration prices must be a finite vector")
    if np.any((prices < 0.0) | (prices > 1.0)):
        raise ValueError("calibration prices must be inside [0, 1]")
    indices = np.floor(prices * PRICE_BAND_COUNT + 1e-12)
    return np.clip(indices, 0, PRICE_BAND_COUNT - 1).astype(np.int16)


def _cell_parameter_arrays(
    cells: tuple[AsymmetricCalibrationCell, ...],
    calibrators: tuple[TimeBandCalibrator, ...],
) -> tuple[np.ndarray, np.ndarray]:
    slopes = np.full((len(calibrators), PRICE_BAND_COUNT, 2), np.nan)
    intercepts = np.full_like(slopes, np.nan)
    time_keys = {
        (band.start_second, band.end_second_exclusive): index
        for index, band in enumerate(calibrators)
    }
    side_indices = {"YES": 0, "NO": 1}
    for cell in cells:
        time_index = time_keys[(cell.start_second, cell.end_second_exclusive)]
        price_index = round(cell.minimum_price / PRICE_BAND_WIDTH)
        side_index = side_indices[cell.side]
        slopes[time_index, price_index, side_index] = cell.slope
        intercepts[time_index, price_index, side_index] = cell.intercept
    if not np.isfinite(slopes).all() or not np.isfinite(intercepts).all():
        raise RuntimeError("side/price/time calibration cells are incomplete")
    return slopes, intercepts


def _market_equal_weights_for_ids(market_ids: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(market_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights / weights.sum()


def _coherent_probability_from_parameters(
    parent_logit: np.ndarray,
    yes_price_indices: np.ndarray,
    no_price_indices: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    yes_eta = (
        parent_logit * slopes[yes_price_indices, 0]
        + intercepts[yes_price_indices, 0]
    )
    no_eta = (
        -parent_logit * slopes[no_price_indices, 1]
        + intercepts[no_price_indices, 1]
    )
    return _sigmoid(0.5 * (yes_eta - no_eta))


def _coherent_calibration_objective(
    parameters: np.ndarray,
    parent_logit: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    yes_price_indices: np.ndarray,
    no_price_indices: np.ndarray,
    active_keys: tuple[tuple[int, int], ...],
    penalty_weights: np.ndarray,
    identity_l2_strength: float,
) -> tuple[float, np.ndarray]:
    slopes = np.ones((PRICE_BAND_COUNT, 2), dtype=np.float64)
    intercepts = np.zeros_like(slopes)
    for offset, key in enumerate(active_keys):
        slopes[key] = parameters[2 * offset]
        intercepts[key] = parameters[2 * offset + 1]
    yes_eta = (
        parent_logit * slopes[yes_price_indices, 0]
        + intercepts[yes_price_indices, 0]
    )
    no_eta = (
        -parent_logit * slopes[no_price_indices, 1]
        + intercepts[no_price_indices, 1]
    )
    eta = 0.5 * (yes_eta - no_eta)
    weighted_loss = np.sum(weights * (np.logaddexp(0.0, eta) - labels * eta))
    delta = parameters.copy()
    delta[0::2] -= 1.0
    parameter_penalty_weights = np.repeat(penalty_weights, 2)
    penalty = 0.5 * identity_l2_strength * float(
        (parameter_penalty_weights * delta) @ delta
    )
    error = weights * (_sigmoid(eta) - labels)
    gradient = np.empty_like(parameters)
    for offset, (price_index, side_index) in enumerate(active_keys):
        selected = (
            yes_price_indices == price_index
            if side_index == 0
            else no_price_indices == price_index
        )
        intercept_sign = 1.0 if side_index == 0 else -1.0
        gradient[2 * offset] = (
            np.sum(error[selected] * 0.5 * parent_logit[selected])
            + identity_l2_strength
            * penalty_weights[offset]
            * delta[2 * offset]
        )
        gradient[2 * offset + 1] = (
            np.sum(error[selected] * 0.5 * intercept_sign)
            + identity_l2_strength
            * penalty_weights[offset]
            * delta[2 * offset + 1]
        )
    return float(weighted_loss + penalty), gradient


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-clipped))


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
    if model in {PRICE_LOGISTIC, CORE_PRICE, CORE_CANDLES_PRICE}:
        return "exact_execution_price_cohort"
    if model in {L2_MATCHED_CORE_PRICE_CONTROL, CORE_L2_PRICE}:
        return "l2_exact_execution_cohort"
    if model in {ORACLE_MATCHED_CORE_PRICE_CONTROL, CORE_ORACLE_PRICE}:
        return "causal_oracle_exact_execution_cohort"
    raise ValueError(f"unknown asymmetric-value model: {model}")


def _scoring_cohort(model: str) -> str:
    return _training_cohort(model)


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
    ).with_columns(
        (1.0 - pl.col("probability_yes")).alias("probability_no"),
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
