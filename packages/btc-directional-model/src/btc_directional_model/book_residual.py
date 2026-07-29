from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize

BOOK_MID_DISAGREEMENT_FEATURE = "book_core_mid_logit_disagreement"
STANDARDIZED_BOOK_RESIDUAL_FEATURES = (
    "book_vwap10_logit_minus_mid",
    "book_imbalance_difference",
    "book_log_bid_depth_difference",
    "book_log_ask_depth_difference",
    "book_spread_difference",
    "book_mid_logit_delta_5s",
    "book_imbalance_difference_delta_5s",
)
BOOK_RESIDUAL_FEATURES = (
    BOOK_MID_DISAGREEMENT_FEATURE,
    *STANDARDIZED_BOOK_RESIDUAL_FEATURES,
)

# These columns are produced only after the strict ten-share quality route in
# derive_strict_book_feature_frame. Quality flags and provider age deliberately
# remain routing inputs, never residual-model features.
STRICT_BOOK_RESIDUAL_INPUT_COLUMNS = (
    "model_eligible",
    "book_up_mid",
    "book_down_mid",
    "book_up_ask_vwap_10",
    "book_down_ask_vwap_10",
    "book_up_imbalance",
    "book_down_imbalance",
    "book_up_log_bid_depth",
    "book_down_log_bid_depth",
    "book_up_log_ask_depth",
    "book_down_log_ask_depth",
    "book_up_spread",
    "book_down_spread",
    "book_up_mid_delta_5s",
    "book_down_mid_delta_5s",
    "book_up_imbalance_delta_5s",
    "book_down_imbalance_delta_5s",
)

RAW_DOWN = "DOWN"
RAW_UP = "UP"
RAW_DIRECTIONS = (RAW_DOWN, RAW_UP)
LOGIT_LIMIT = 40.0
MINIMUM_SCALE = 1e-12


@dataclass(frozen=True)
class CalibrationBand:
    name: str
    start_second: int
    end_second_exclusive: int

    def contains(self, second: int | np.integer[Any]) -> bool:
        return self.start_second <= int(second) < self.end_second_exclusive


CALIBRATION_BANDS = (
    CalibrationBand("60-89", 60, 90),
    CalibrationBand("90-119", 90, 120),
    CalibrationBand("120-179", 120, 180),
    CalibrationBand("180-240", 180, 241),
)


@dataclass(frozen=True)
class DistributionDiagnostics:
    minimum: float
    p05: float
    median: float
    p95: float
    maximum: float
    mean: float
    standard_deviation: float

    @classmethod
    def from_values(cls, values: np.ndarray) -> DistributionDiagnostics:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("diagnostic values must be a non-empty finite vector")
        quantiles = np.quantile(values, (0.05, 0.50, 0.95))
        return cls(
            minimum=float(values.min()),
            p05=float(quantiles[0]),
            median=float(quantiles[1]),
            p95=float(quantiles[2]),
            maximum=float(values.max()),
            mean=float(values.mean()),
            standard_deviation=float(values.std()),
        )


@dataclass(frozen=True)
class ResidualFitDiagnostics:
    converged: bool
    optimizer_status: int
    optimizer_message: str
    iterations: int
    function_evaluations: int
    rows: int
    markets: int
    positive_rate: float
    l2_strength: float
    objective: float
    weighted_log_loss: float
    penalty: float
    gradient_infinity_norm: float
    gamma: float
    beta_l2_norm: float
    correction_logit: DistributionDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualBookModel:
    """Compact additive correction to a pre-existing universal BTC core.

    The seven book residuals use zero-centred RMS scaling. Scaling about zero,
    rather than subtracting a sample mean, preserves their exact UP/DOWN
    antisymmetry and prevents an implicit residual intercept.
    """

    gamma: float
    beta: tuple[float, ...]
    feature_scales: tuple[float, ...]
    l2_strength: float
    feature_names: tuple[str, ...] = BOOK_RESIDUAL_FEATURES

    def __post_init__(self) -> None:
        if self.feature_names != BOOK_RESIDUAL_FEATURES:
            raise ValueError("residual model feature order must match the frozen contract")
        if len(self.beta) != len(STANDARDIZED_BOOK_RESIDUAL_FEATURES):
            raise ValueError("residual model requires seven standardized coefficients")
        if len(self.feature_scales) != len(STANDARDIZED_BOOK_RESIDUAL_FEATURES):
            raise ValueError("residual model requires seven feature scales")
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not math.isfinite(self.l2_strength) or self.l2_strength < 0.0:
            raise ValueError("l2_strength must be finite and nonnegative")
        if not all(math.isfinite(value) for value in self.beta):
            raise ValueError("residual coefficients must be finite")
        if not all(math.isfinite(value) and value > 0.0 for value in self.feature_scales):
            raise ValueError("residual feature scales must be finite and positive")

    @classmethod
    def identity(
        cls,
        *,
        feature_scales: Sequence[float] | None = None,
        l2_strength: float = 0.0,
    ) -> ResidualBookModel:
        scales = (
            tuple(float(value) for value in feature_scales)
            if feature_scales is not None
            else (1.0,) * len(STANDARDIZED_BOOK_RESIDUAL_FEATURES)
        )
        return cls(
            gamma=0.0,
            beta=(0.0,) * len(STANDARDIZED_BOOK_RESIDUAL_FEATURES),
            feature_scales=scales,
            l2_strength=float(l2_strength),
        )

    def correction_logit(self, features: np.ndarray) -> np.ndarray:
        matrix = _finite_matrix(
            features,
            columns=len(BOOK_RESIDUAL_FEATURES),
            name="book residual features",
        )
        standardized = matrix[:, 1:] / np.asarray(self.feature_scales, dtype=np.float64)
        return self.gamma * matrix[:, 0] + standardized @ np.asarray(self.beta, dtype=np.float64)

    def raw_logit(self, core_logit: np.ndarray, features: np.ndarray) -> np.ndarray:
        core = _finite_vector(core_logit, name="core logits")
        if len(core) != len(features):
            raise ValueError("core logit count does not match book feature rows")
        return core + self.correction_logit(features)

    def probability(self, core_logit: np.ndarray, features: np.ndarray) -> np.ndarray:
        return _sigmoid(self.raw_logit(core_logit, features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "beta": list(self.beta),
            "feature_scales": list(self.feature_scales),
            "l2_strength": self.l2_strength,
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResidualBookModel:
        return cls(
            gamma=float(value["gamma"]),
            beta=tuple(float(item) for item in value["beta"]),
            feature_scales=tuple(float(item) for item in value["feature_scales"]),
            l2_strength=float(value["l2_strength"]),
            feature_names=tuple(str(item) for item in value["feature_names"]),
        )


@dataclass(frozen=True)
class CalibrationCell:
    band: str
    raw_direction: str
    slope: float
    intercept: float
    converged: bool
    optimizer_status: int
    optimizer_message: str
    iterations: int
    function_evaluations: int
    rows: int
    markets: int
    positives: int
    identity_l2_strength: float
    objective: float
    weighted_log_loss: float
    penalty: float

    @property
    def key(self) -> str:
        return calibration_cell_key(self.band, self.raw_direction)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CalibrationCell:
        return cls(
            band=str(value["band"]),
            raw_direction=str(value["raw_direction"]),
            slope=float(value["slope"]),
            intercept=float(value["intercept"]),
            converged=bool(value["converged"]),
            optimizer_status=int(value["optimizer_status"]),
            optimizer_message=str(value["optimizer_message"]),
            iterations=int(value["iterations"]),
            function_evaluations=int(value["function_evaluations"]),
            rows=int(value["rows"]),
            markets=int(value["markets"]),
            positives=int(value["positives"]),
            identity_l2_strength=float(value["identity_l2_strength"]),
            objective=float(value["objective"]),
            weighted_log_loss=float(value["weighted_log_loss"]),
            penalty=float(value["penalty"]),
        )


@dataclass(frozen=True)
class DirectionTimeCalibrator:
    """Eight deterministic, monotone cells selected by pre-calibration score."""

    cells: tuple[CalibrationCell, ...]
    bands: tuple[CalibrationBand, ...] = CALIBRATION_BANDS

    def __post_init__(self) -> None:
        expected = {
            calibration_cell_key(band.name, direction)
            for band in self.bands
            for direction in RAW_DIRECTIONS
        }
        actual = {cell.key for cell in self.cells}
        if actual != expected or len(actual) != len(self.cells):
            raise ValueError("direction/time calibrator must contain exactly eight cells")
        if any(
            not math.isfinite(cell.slope) or cell.slope < 0.0 or not math.isfinite(cell.intercept)
            for cell in self.cells
        ):
            raise ValueError("calibration cells require finite nonnegative slopes")

    def calibrated_logit(
        self,
        raw_logit: np.ndarray,
        seconds_elapsed: np.ndarray,
    ) -> np.ndarray:
        raw = _finite_vector(raw_logit, name="raw residual logits")
        seconds = _seconds_vector(seconds_elapsed, expected_length=len(raw))
        keys = calibration_route_keys(raw, seconds, bands=self.bands)
        calibrated = np.empty_like(raw)
        cells = {cell.key: cell for cell in self.cells}
        for key, cell in cells.items():
            selected = keys == key
            calibrated[selected] = raw[selected] * cell.slope + cell.intercept
        return calibrated

    def probability(
        self,
        raw_logit: np.ndarray,
        seconds_elapsed: np.ndarray,
    ) -> np.ndarray:
        return _sigmoid(self.calibrated_logit(raw_logit, seconds_elapsed))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "bands": [asdict(band) for band in self.bands],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DirectionTimeCalibrator:
        return cls(
            cells=tuple(CalibrationCell.from_dict(item) for item in value["cells"]),
            bands=tuple(CalibrationBand(**item) for item in value["bands"]),
        )


@dataclass(frozen=True)
class DirectionTimeCalibrationDiagnostics:
    cells: tuple[CalibrationCell, ...]
    raw_to_calibrated_direction_flip_rate: float
    raw_logit: DistributionDiagnostics
    calibrated_logit: DistributionDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "raw_to_calibrated_direction_flip_rate": (self.raw_to_calibrated_direction_flip_rate),
            "raw_logit": asdict(self.raw_logit),
            "calibrated_logit": asdict(self.calibrated_logit),
        }


@dataclass(frozen=True)
class ResidualRouteDiagnostics:
    rows: int
    residual_rows: int
    core_fallback_rows: int
    correction_logit: DistributionDiagnostics | None
    calibrated_direction_flip_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_strict_book_residual_features(
    frame: pl.DataFrame,
    core_logit: np.ndarray,
) -> np.ndarray:
    """Build the frozen eight-feature residual matrix from strict-v2 rows.

    The exact-five-second midpoint-logit delta is reconstructed from the
    current midpoint and its separately derived exact-five-second level delta.
    It is intentionally not approximated by subtracting the two midpoint
    deltas, because a logit is nonlinear.
    """

    missing = sorted(set(STRICT_BOOK_RESIDUAL_INPUT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("strict residual frame is missing required columns: " + ", ".join(missing))
    if frame.is_empty():
        raise ValueError("strict residual frame cannot be empty")
    if frame.filter(~pl.col("model_eligible").fill_null(False)).height:
        raise ValueError("residual features require strict ten-share exact-delta rows")
    if {"market_id", "observed_at"} <= set(frame.columns):
        duplicates = (
            frame.group_by("market_id", "observed_at").len().filter(pl.col("len") > 1).height
        )
        if duplicates:
            raise ValueError("strict residual frame contains duplicate row keys")

    core = _finite_vector(core_logit, name="core logits")
    if len(core) != frame.height:
        raise ValueError("core logit count does not match strict residual rows")

    up_mid = _frame_column(frame, "book_up_mid", positive=True)
    down_mid = _frame_column(frame, "book_down_mid", positive=True)
    up_vwap = _frame_column(frame, "book_up_ask_vwap_10", positive=True)
    down_vwap = _frame_column(frame, "book_down_ask_vwap_10", positive=True)
    up_mid_delta = _frame_column(frame, "book_up_mid_delta_5s")
    down_mid_delta = _frame_column(frame, "book_down_mid_delta_5s")
    previous_up_mid = up_mid - up_mid_delta
    previous_down_mid = down_mid - down_mid_delta
    if np.any(previous_up_mid <= 0.0) or np.any(previous_down_mid <= 0.0):
        raise ValueError("reconstructed prior book midpoint must be positive")

    mid_logit = _pair_logit(up_mid, down_mid)
    previous_mid_logit = _pair_logit(previous_up_mid, previous_down_mid)
    vwap_logit = _pair_logit(up_vwap, down_vwap)

    up_imbalance = _frame_column(frame, "book_up_imbalance")
    down_imbalance = _frame_column(frame, "book_down_imbalance")
    up_imbalance_delta = _frame_column(frame, "book_up_imbalance_delta_5s")
    down_imbalance_delta = _frame_column(frame, "book_down_imbalance_delta_5s")
    imbalance_difference = up_imbalance - down_imbalance
    previous_imbalance_difference = (up_imbalance - up_imbalance_delta) - (
        down_imbalance - down_imbalance_delta
    )

    matrix = np.column_stack(
        (
            mid_logit - core,
            vwap_logit - mid_logit,
            imbalance_difference,
            _frame_column(frame, "book_up_log_bid_depth")
            - _frame_column(frame, "book_down_log_bid_depth"),
            _frame_column(frame, "book_up_log_ask_depth")
            - _frame_column(frame, "book_down_log_ask_depth"),
            _frame_column(frame, "book_up_spread") - _frame_column(frame, "book_down_spread"),
            mid_logit - previous_mid_logit,
            imbalance_difference - previous_imbalance_difference,
        )
    )
    return _finite_matrix(
        matrix,
        columns=len(BOOK_RESIDUAL_FEATURES),
        name="derived book residual features",
    )


def fit_residual_book_model(
    core_logit: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    market_ids: Sequence[Any],
    *,
    l2_strength: float,
    maximum_iterations: int = 500,
    tolerance: float = 1e-9,
) -> tuple[ResidualBookModel, ResidualFitDiagnostics]:
    core = _finite_vector(core_logit, name="core logits")
    matrix = _finite_matrix(
        features,
        columns=len(BOOK_RESIDUAL_FEATURES),
        name="book residual features",
    )
    target = _binary_labels(labels, expected_length=len(core))
    if matrix.shape[0] != len(core):
        raise ValueError("book feature count does not match core logits")
    weights, markets = market_equal_weights(market_ids, expected_length=len(core))
    strength = _nonnegative_float(l2_strength, name="l2_strength")
    _optimizer_controls(maximum_iterations, tolerance)

    scales = np.sqrt(np.sum(weights[:, None] * np.square(matrix[:, 1:]), axis=0))
    scales = np.where(scales > MINIMUM_SCALE, scales, 1.0)
    design = np.column_stack((matrix[:, 0], matrix[:, 1:] / scales))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        eta = core + design @ parameters
        weighted_loss = np.sum(weights * (np.logaddexp(0.0, eta) - target * eta))
        penalty = 0.5 * strength * float(parameters @ parameters)
        gradient = design.T @ (weights * (_sigmoid(eta) - target))
        gradient += strength * parameters
        return float(weighted_loss + penalty), np.asarray(gradient, dtype=np.float64)

    result = minimize(
        objective,
        np.zeros(design.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=((0.0, 1.0), *((None, None),) * (design.shape[1] - 1)),
        options={
            "maxiter": int(maximum_iterations),
            "ftol": float(tolerance),
            "gtol": float(tolerance),
            "maxls": 50,
        },
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    model = ResidualBookModel(
        gamma=float(parameters[0]),
        beta=tuple(float(value) for value in parameters[1:]),
        feature_scales=tuple(float(value) for value in scales),
        l2_strength=strength,
    )
    correction = model.correction_logit(matrix)
    fitted_eta = core + correction
    weighted_log_loss = float(
        np.sum(weights * (np.logaddexp(0.0, fitted_eta) - target * fitted_eta))
    )
    penalty = 0.5 * strength * float(parameters @ parameters)
    diagnostics = ResidualFitDiagnostics(
        converged=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        iterations=int(result.nit),
        function_evaluations=int(result.nfev),
        rows=len(core),
        markets=markets,
        positive_rate=float(np.sum(weights * target)),
        l2_strength=strength,
        objective=weighted_log_loss + penalty,
        weighted_log_loss=weighted_log_loss,
        penalty=penalty,
        gradient_infinity_norm=float(np.max(np.abs(result.jac))),
        gamma=model.gamma,
        beta_l2_norm=float(np.linalg.norm(parameters[1:])),
        correction_logit=DistributionDiagnostics.from_values(correction),
    )
    return model, diagnostics


def fit_direction_time_calibrator(
    raw_logit: np.ndarray,
    seconds_elapsed: np.ndarray,
    labels: np.ndarray,
    market_ids: Sequence[Any],
    *,
    identity_l2_strength: float,
    minimum_rows_per_cell: int = 1,
    minimum_markets_per_cell: int = 1,
    maximum_iterations: int = 500,
    tolerance: float = 1e-9,
) -> tuple[DirectionTimeCalibrator, DirectionTimeCalibrationDiagnostics]:
    raw = _finite_vector(raw_logit, name="raw residual logits")
    seconds = _seconds_vector(seconds_elapsed, expected_length=len(raw))
    target = _binary_labels(labels, expected_length=len(raw))
    ids = _market_id_vector(market_ids, expected_length=len(raw))
    strength = _nonnegative_float(identity_l2_strength, name="identity_l2_strength")
    if minimum_rows_per_cell <= 0 or minimum_markets_per_cell <= 0:
        raise ValueError("calibration cell minimums must be positive")
    _optimizer_controls(maximum_iterations, tolerance)

    route_keys = calibration_route_keys(raw, seconds)
    fitted_cells: list[CalibrationCell] = []
    for band in CALIBRATION_BANDS:
        for direction in RAW_DIRECTIONS:
            key = calibration_cell_key(band.name, direction)
            selected = route_keys == key
            cell_raw = raw[selected]
            cell_target = target[selected]
            cell_ids = ids[selected]
            if len(cell_raw) < minimum_rows_per_cell:
                raise ValueError(f"calibration cell {key} has insufficient rows")
            if np.unique(cell_target).size != 2:
                raise ValueError(f"calibration cell {key} must contain both classes")
            weights, markets = market_equal_weights(cell_ids, expected_length=len(cell_raw))
            if markets < minimum_markets_per_cell:
                raise ValueError(f"calibration cell {key} has insufficient markets")

            result = minimize(
                _calibration_objective,
                np.asarray((1.0, 0.0), dtype=np.float64),
                args=(cell_raw, cell_target, weights, strength),
                method="L-BFGS-B",
                jac=True,
                bounds=((0.0, None), (None, None)),
                options={
                    "maxiter": int(maximum_iterations),
                    "ftol": float(tolerance),
                    "gtol": float(tolerance),
                    "maxls": 50,
                },
            )
            slope, intercept = (float(value) for value in result.x)
            calibrated = slope * cell_raw + intercept
            weighted_log_loss = float(
                np.sum(weights * (np.logaddexp(0.0, calibrated) - cell_target * calibrated))
            )
            penalty = 0.5 * strength * ((slope - 1.0) ** 2 + intercept**2)
            fitted_cells.append(
                CalibrationCell(
                    band=band.name,
                    raw_direction=direction,
                    slope=slope,
                    intercept=intercept,
                    converged=bool(result.success),
                    optimizer_status=int(result.status),
                    optimizer_message=str(result.message),
                    iterations=int(result.nit),
                    function_evaluations=int(result.nfev),
                    rows=len(cell_raw),
                    markets=markets,
                    positives=int(cell_target.sum()),
                    identity_l2_strength=strength,
                    objective=weighted_log_loss + penalty,
                    weighted_log_loss=weighted_log_loss,
                    penalty=penalty,
                )
            )

    calibrator = DirectionTimeCalibrator(tuple(fitted_cells))
    calibrated = calibrator.calibrated_logit(raw, seconds)
    diagnostics = DirectionTimeCalibrationDiagnostics(
        cells=calibrator.cells,
        raw_to_calibrated_direction_flip_rate=float(np.mean((raw >= 0.0) != (calibrated >= 0.0))),
        raw_logit=DistributionDiagnostics.from_values(raw),
        calibrated_logit=DistributionDiagnostics.from_values(calibrated),
    )
    return calibrator, diagnostics


def route_book_residual_probability(
    model: ResidualBookModel,
    core_logit: np.ndarray,
    features: np.ndarray,
    strict_eligible: np.ndarray,
    *,
    seconds_elapsed: np.ndarray | None = None,
    calibrator: DirectionTimeCalibrator | None = None,
) -> tuple[np.ndarray, ResidualRouteDiagnostics]:
    """Apply the residual only to strict rows; every other value is core-identical."""

    core = _finite_vector(core_logit, name="core logits")
    eligible = np.asarray(strict_eligible, dtype=np.bool_)
    if eligible.ndim != 1 or len(eligible) != len(core):
        raise ValueError("strict eligibility must match core logits")
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (len(core), len(BOOK_RESIDUAL_FEATURES)):
        raise ValueError("book feature matrix must match core logits and frozen features")
    if calibrator is not None and seconds_elapsed is None:
        raise ValueError("calibrated residual routing requires seconds_elapsed")

    output = _sigmoid(core)
    correction_distribution: DistributionDiagnostics | None = None
    flip_rate = 0.0
    if eligible.any():
        strict_features = _finite_matrix(
            matrix[eligible],
            columns=len(BOOK_RESIDUAL_FEATURES),
            name="eligible book residual features",
        )
        strict_core = core[eligible]
        strict_raw = model.raw_logit(strict_core, strict_features)
        correction_distribution = DistributionDiagnostics.from_values(strict_raw - strict_core)
        strict_final = strict_raw
        if calibrator is not None:
            seconds = _seconds_vector(seconds_elapsed, expected_length=len(core))
            strict_final = calibrator.calibrated_logit(
                strict_raw,
                seconds[eligible],
            )
            flip_rate = float(np.mean((strict_raw >= 0.0) != (strict_final >= 0.0)))
        output[eligible] = _sigmoid(strict_final)

    diagnostics = ResidualRouteDiagnostics(
        rows=len(core),
        residual_rows=int(eligible.sum()),
        core_fallback_rows=int((~eligible).sum()),
        correction_logit=correction_distribution,
        calibrated_direction_flip_rate=flip_rate,
    )
    return output, diagnostics


def calibration_route_keys(
    raw_logit: np.ndarray,
    seconds_elapsed: np.ndarray,
    *,
    bands: tuple[CalibrationBand, ...] = CALIBRATION_BANDS,
) -> np.ndarray:
    raw = _finite_vector(raw_logit, name="raw residual logits")
    seconds = _seconds_vector(seconds_elapsed, expected_length=len(raw))
    keys = np.empty(len(raw), dtype=object)
    assigned = np.zeros(len(raw), dtype=np.bool_)
    for band in bands:
        in_band = (seconds >= band.start_second) & (seconds < band.end_second_exclusive)
        keys[in_band & (raw < 0.0)] = calibration_cell_key(band.name, RAW_DOWN)
        keys[in_band & (raw >= 0.0)] = calibration_cell_key(band.name, RAW_UP)
        assigned |= in_band
    if not assigned.all():
        invalid = sorted({int(value) for value in seconds[~assigned]})
        raise ValueError(f"seconds_elapsed outside calibration bands: {invalid}")
    return keys


def calibration_cell_key(band: str, raw_direction: str) -> str:
    if raw_direction not in RAW_DIRECTIONS:
        raise ValueError(f"unsupported raw direction: {raw_direction}")
    return f"{band}:{raw_direction}"


def market_equal_weights(
    market_ids: Sequence[Any],
    *,
    expected_length: int | None = None,
) -> tuple[np.ndarray, int]:
    ids = _market_id_vector(market_ids, expected_length=expected_length)
    if len(ids) == 0:
        raise ValueError("cannot calculate market-equal weights for an empty cohort")
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights /= weights.sum()
    return weights, len(counts)


def _pair_logit(up_value: np.ndarray, down_value: np.ndarray) -> np.ndarray:
    if (
        not np.isfinite(up_value).all()
        or not np.isfinite(down_value).all()
        or np.any(up_value <= 0.0)
        or np.any(down_value <= 0.0)
    ):
        raise ValueError("paired book values must be finite and positive")
    # logit(up / (up + down)) simplifies to log(up / down). This form avoids
    # a lossy normalization and remains exactly antisymmetric under side swap.
    return np.log(up_value) - np.log(down_value)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -LOGIT_LIMIT, LOGIT_LIMIT)
    return 1.0 / (1.0 + np.exp(-clipped))


def _frame_column(
    frame: pl.DataFrame,
    name: str,
    *,
    positive: bool = False,
) -> np.ndarray:
    values = frame[name].cast(pl.Float64).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    return values


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector")
    return result


def _finite_matrix(values: np.ndarray, *, columns: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != columns or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite matrix with {columns} columns")
    return result


def _binary_labels(values: np.ndarray, *, expected_length: int) -> np.ndarray:
    labels = np.asarray(values, dtype=np.float64)
    if labels.ndim != 1 or len(labels) != expected_length or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("labels must be a binary vector matching the scored rows")
    if np.unique(labels).size != 2:
        raise ValueError("training labels must contain both classes")
    return labels


def _seconds_vector(values: np.ndarray, *, expected_length: int) -> np.ndarray:
    seconds_float = np.asarray(values, dtype=np.float64)
    if (
        seconds_float.ndim != 1
        or len(seconds_float) != expected_length
        or not np.isfinite(seconds_float).all()
        or not np.equal(seconds_float, np.floor(seconds_float)).all()
    ):
        raise ValueError("seconds_elapsed must be finite integral values matching rows")
    return seconds_float.astype(np.int64)


def _market_id_vector(
    values: Sequence[Any],
    *,
    expected_length: int | None,
) -> np.ndarray:
    ids = np.asarray(values, dtype=str)
    if ids.ndim != 1 or (expected_length is not None and len(ids) != expected_length):
        raise ValueError("market identifiers must be a vector matching the scored rows")
    if np.any(np.char.str_len(ids) == 0):
        raise ValueError("market identifiers must be non-empty")
    return ids


def _nonnegative_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _optimizer_controls(maximum_iterations: int, tolerance: float) -> None:
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")


def _calibration_objective(
    parameters: np.ndarray,
    raw_logit: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    identity_l2_strength: float,
) -> tuple[float, np.ndarray]:
    slope, intercept = parameters
    eta = slope * raw_logit + intercept
    weighted_loss = np.sum(weights * (np.logaddexp(0.0, eta) - labels * eta))
    delta = np.asarray((slope - 1.0, intercept), dtype=np.float64)
    penalty = 0.5 * identity_l2_strength * float(delta @ delta)
    error = weights * (_sigmoid(eta) - labels)
    gradient = np.asarray(
        (
            np.sum(error * raw_logit) + identity_l2_strength * (slope - 1.0),
            np.sum(error) + identity_l2_strength * intercept,
        ),
        dtype=np.float64,
    )
    return float(weighted_loss + penalty), gradient
