"""Offline side-conditioned residual model for lower-price PM opportunities.

The model treats the normalized executable Polymarket cost as a prior and
learns only a compact, strongly regularized correction.  Every decision is
represented by an exactly antisymmetric YES/NO pair.  The two corrected side
logits are reconciled to one coherent YES probability at score time.

This module deliberately contains no benchmark selection, policy/PnL tuning,
runtime export, or trading-process integration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Self

import numpy as np
import polars as pl
from scipy.optimize import minimize

from .spot_l2_chainlink_features import (
    L2_CAUSAL_AGE_COLUMNS,
    L2_CAUSAL_TIMESTAMP_COLUMNS,
    L2_MAXIMUM_CAUSAL_AGE_SECONDS,
)

YES = "YES"
NO = "NO"
SIDES = (YES, NO)
PRIOR_CLIP = 1e-6
RANDOM_SEED = 73
MINIMUM_SCALE = 1e-9
LOGIT_LIMIT = 40.0
SIDE_CONDITIONED_RESIDUAL_MODEL = "side_conditioned_residual_value_offline"

CORE_ORACLE_ANCHOR = "core_oracle_anchor"
L2_CONFIRMATION = "l2_confirmation"
CROSS_SOURCE_CONFIRMATION = "cross_source_confirmation"
POLYMARKET_STRUCTURE = "polymarket_structure"
HORIZON_MATURITY = "horizon_maturity"
SOURCE_AVAILABILITY = "source_availability"

# Penalties act on fit-window standardized coefficients.  L2 and every group
# containing L2 confirmation are deliberately no weaker than the Core+Oracle
# anchor.  These are estimator constants, not benchmark-tuned parameters.
GROUP_PENALTIES: dict[str, float] = {
    CORE_ORACLE_ANCHOR: 2.0,
    L2_CONFIRMATION: 6.0,
    CROSS_SOURCE_CONFIRMATION: 6.0,
    POLYMARKET_STRUCTURE: 4.0,
    HORIZON_MATURITY: 8.0,
    SOURCE_AVAILABILITY: 8.0,
}


@dataclass(frozen=True)
class ResidualFeatureSpec:
    name: str
    group: str
    source_columns: tuple[str, ...]


def _feature(
    name: str,
    group: str,
    *source_columns: str,
) -> ResidualFeatureSpec:
    return ResidualFeatureSpec(name, group, tuple(source_columns))


RESIDUAL_FEATURE_SPECS = (
    _feature(
        "side_core_path_from_open_bps",
        CORE_ORACLE_ANCHOR,
        "btc_path_from_window_open_bps",
    ),
    *(
        _feature(
            f"side_core_return_{seconds}s_bps",
            CORE_ORACLE_ANCHOR,
            f"btc_return_{seconds}s_bps",
            "seconds_elapsed",
        )
        for seconds in (1, 5, 15, 30, 60)
    ),
    *(
        _feature(
            f"side_core_signed_flow_{seconds}s",
            CORE_ORACLE_ANCHOR,
            f"btc_signed_flow_{seconds}s",
            "seconds_elapsed",
        )
        for seconds in (5, 30, 60)
    ),
    _feature(
        "side_oracle_return_from_open_bps",
        CORE_ORACLE_ANCHOR,
        "oracle_return_from_window_open_bps",
    ),
    _feature(
        "side_binance_oracle_basis_bps",
        CORE_ORACLE_ANCHOR,
        "binance_oracle_basis_bps",
    ),
    _feature(
        "side_l2_microprice_to_midpoint_bps",
        L2_CONFIRMATION,
        "spot_l2_microprice_to_midpoint_bps",
    ),
    *(
        _feature(
            f"side_l2_midpoint_change_{seconds}s_bps",
            L2_CONFIRMATION,
            f"spot_l2_midpoint_change_{seconds}s_bps",
            "seconds_elapsed",
        )
        for seconds in (1, 5, 15, 30, 60)
    ),
    _feature(
        "side_l2_imbalance_5",
        L2_CONFIRMATION,
        "spot_l2_imbalance_5",
    ),
    _feature(
        "side_l2_imbalance_20",
        L2_CONFIRMATION,
        "spot_l2_imbalance_20",
    ),
    _feature(
        "side_cross_source_consensus",
        CROSS_SOURCE_CONFIRMATION,
        "btc_path_from_window_open_bps",
        "oracle_return_from_window_open_bps",
        "spot_l2_microprice_to_midpoint_bps",
    ),
    _feature(
        "side_core_oracle_agreement",
        CROSS_SOURCE_CONFIRMATION,
        "btc_path_from_window_open_bps",
        "oracle_return_from_window_open_bps",
    ),
    _feature(
        "side_core_oracle_contradiction",
        CROSS_SOURCE_CONFIRMATION,
        "btc_path_from_window_open_bps",
        "oracle_return_from_window_open_bps",
    ),
    _feature(
        "side_core_l2_agreement",
        CROSS_SOURCE_CONFIRMATION,
        "btc_path_from_window_open_bps",
        "spot_l2_microprice_to_midpoint_bps",
    ),
    _feature(
        "side_core_l2_contradiction",
        CROSS_SOURCE_CONFIRMATION,
        "btc_path_from_window_open_bps",
        "spot_l2_microprice_to_midpoint_bps",
    ),
    *(
        _feature(
            f"side_horizon_mature_{seconds}s",
            HORIZON_MATURITY,
            "seconds_elapsed",
        )
        for seconds in (1, 5, 15, 30, 60)
    ),
    _feature(
        "side_pm_depth_advantage",
        POLYMARKET_STRUCTURE,
        "pm_yes_depth_log",
        "pm_no_depth_log",
    ),
    _feature(
        "side_pm_vwap_slippage_disadvantage",
        POLYMARKET_STRUCTURE,
        "pm_yes_vwap_slippage",
        "pm_no_vwap_slippage",
    ),
    _feature(
        "side_pm_book_age_disadvantage_seconds",
        POLYMARKET_STRUCTURE,
        "pm_yes_book_age_seconds",
        "pm_no_book_age_seconds",
    ),
    _feature(
        "side_pm_cost_overround",
        POLYMARKET_STRUCTURE,
        "pm_cost_overround",
    ),
    _feature(
        "side_oracle_age_scaled",
        SOURCE_AVAILABILITY,
        "oracle_age_seconds",
    ),
    _feature(
        "side_oracle_available",
        SOURCE_AVAILABILITY,
        "early_oracle_eligible",
    ),
)
RESIDUAL_FEATURE_NAMES = tuple(spec.name for spec in RESIDUAL_FEATURE_SPECS)
RESIDUAL_FEATURE_GROUPS = tuple(spec.group for spec in RESIDUAL_FEATURE_SPECS)

IDENTITY_COLUMNS = (
    "market_id",
    "window_start",
    "observed_at",
    "seconds_elapsed",
)
CAUSAL_TIMESTAMP_COLUMNS = (
    "yes_received_at",
    "no_received_at",
    "oracle_source_timestamp",
    "oracle_block_timestamp",
    *L2_CAUSAL_TIMESTAMP_COLUMNS,
)
PRIOR_COLUMNS = ("pm_yes_cost_per_share", "pm_no_cost_per_share")
NUMERIC_SOURCE_COLUMNS = tuple(
    dict.fromkeys(
        (
            *PRIOR_COLUMNS,
            *(
                column
                for spec in RESIDUAL_FEATURE_SPECS
                for column in spec.source_columns
                if column not in {"seconds_elapsed", "early_oracle_eligible"}
            ),
        )
    )
)
MATURITY_GATED_SOURCE_COLUMNS = {
    **{f"btc_return_{seconds}s_bps": seconds for seconds in (1, 5, 15, 30, 60)},
    **{f"btc_signed_flow_{seconds}s": seconds for seconds in (5, 30, 60)},
}
REQUIRED_FEATURE_COLUMNS = tuple(
    dict.fromkeys(
        (
            *IDENTITY_COLUMNS,
            *CAUSAL_TIMESTAMP_COLUMNS,
            *L2_CAUSAL_AGE_COLUMNS,
            "early_oracle_eligible",
            *NUMERIC_SOURCE_COLUMNS,
        )
    )
)


@dataclass(frozen=True)
class SideConditionedRows:
    """Exactly two side rows per decision, ordered YES then NO."""

    market_ids: tuple[str, ...]
    decision_indices: tuple[int, ...]
    sides: tuple[str, ...]
    prior_probability: np.ndarray
    prior_logit: np.ndarray
    features: np.ndarray
    row_weights: np.ndarray
    labels: np.ndarray | None

    def __post_init__(self) -> None:
        rows = len(self.market_ids)
        if rows == 0 or rows % 2:
            raise ValueError("side-conditioned rows require non-empty YES/NO pairs")
        if not (
            len(self.decision_indices)
            == len(self.sides)
            == len(self.prior_probability)
            == len(self.prior_logit)
            == len(self.features)
            == len(self.row_weights)
            == rows
        ):
            raise ValueError("side-conditioned arrays have inconsistent lengths")
        if self.features.shape != (rows, len(RESIDUAL_FEATURE_NAMES)):
            raise ValueError("side-conditioned feature matrix has the wrong schema")
        if self.labels is not None and len(self.labels) != rows:
            raise ValueError("side-conditioned labels have the wrong length")
        if not np.isfinite(self.prior_probability).all():
            raise ValueError("side prior probabilities must be finite")
        if not np.isfinite(self.prior_logit).all():
            raise ValueError("side prior logits must be finite")
        if not np.isfinite(self.features).all():
            raise ValueError("side-conditioned features must be finite")
        if not np.isfinite(self.row_weights).all() or np.any(self.row_weights <= 0.0):
            raise ValueError("side row weights must be finite and positive")


@dataclass(frozen=True)
class ResidualValueFitDiagnostics:
    converged: bool
    optimizer_status: int
    optimizer_message: str
    iterations: int
    function_evaluations: int
    decision_rows: int
    side_rows: int
    markets: int
    weighted_log_loss: float
    penalty: float
    objective: float
    gradient_infinity_norm: float
    correction_l2_norm: float
    maximum_abs_correction: float
    market_weight_total_min: float
    market_weight_total_max: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SideConditionedResidualModel:
    """Frozen offline correction to the normalized executable-price prior."""

    coefficients: tuple[float, ...]
    feature_scales: tuple[float, ...]
    group_penalties: tuple[tuple[str, float], ...]
    random_seed: int = RANDOM_SEED
    feature_names: tuple[str, ...] = RESIDUAL_FEATURE_NAMES
    feature_groups: tuple[str, ...] = RESIDUAL_FEATURE_GROUPS
    prior_clip: float = PRIOR_CLIP

    selection_eligible: ClassVar[bool] = False
    runtime_exportable: ClassVar[bool] = False

    def __post_init__(self) -> None:
        count = len(RESIDUAL_FEATURE_NAMES)
        if self.feature_names != RESIDUAL_FEATURE_NAMES:
            raise ValueError("residual feature order must match the frozen contract")
        if self.feature_groups != RESIDUAL_FEATURE_GROUPS:
            raise ValueError("residual feature groups must match the frozen contract")
        if len(self.coefficients) != count or len(self.feature_scales) != count:
            raise ValueError("residual coefficient and scale counts must match features")
        if not all(math.isfinite(value) for value in self.coefficients):
            raise ValueError("residual coefficients must be finite")
        if not all(math.isfinite(value) and value > 0.0 for value in self.feature_scales):
            raise ValueError("residual feature scales must be finite and positive")
        penalties = dict(self.group_penalties)
        if penalties != GROUP_PENALTIES:
            raise ValueError("residual group penalties must match the frozen contract")
        if penalties[L2_CONFIRMATION] < penalties[CORE_ORACLE_ANCHOR]:
            raise ValueError("L2 confirmation cannot be less regularized than the anchor")
        if penalties[CROSS_SOURCE_CONFIRMATION] < penalties[CORE_ORACLE_ANCHOR]:
            raise ValueError("cross-source confirmation cannot be less regularized than the anchor")
        if self.random_seed != RANDOM_SEED:
            raise ValueError(f"residual random seed is fixed at {RANDOM_SEED}")
        if self.prior_clip != PRIOR_CLIP:
            raise ValueError(f"residual prior clipping is fixed at {PRIOR_CLIP}")

    @classmethod
    def fit(
        cls,
        frame: pl.DataFrame,
    ) -> tuple[Self, ResidualValueFitDiagnostics]:
        _validate_input_frame(frame, require_labels=True)
        prior = normalized_executable_cost_prior(frame)[:, 0]
        prior_logit = np.log(prior) - np.log1p(-prior)
        matrix = _derive_yes_features(frame)
        weights = _market_equal_decision_weights(frame)
        scales = _weighted_rms(matrix, weights)
        matrix /= scales
        penalties = _penalty_vector()
        labels = _finite_vector(
            frame["label_up"].to_numpy().astype(np.float64, copy=False),
            name="decision labels",
        )
        total_weight = float(weights.sum())

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            logits = prior_logit + matrix @ beta
            losses = np.logaddexp(0.0, logits) - labels * logits
            weighted_loss = float(np.dot(weights, losses) / total_weight)
            penalty = 0.5 * float(np.dot(penalties, beta * beta))
            probability = _sigmoid(logits)
            gradient = matrix.T @ (weights * (probability - labels)) / total_weight
            gradient += penalties * beta
            return weighted_loss + penalty, gradient

        initial = np.random.default_rng(RANDOM_SEED).normal(
            loc=0.0,
            scale=1e-7,
            size=matrix.shape[1],
        )
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 1_000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
        )
        coefficients = np.asarray(result.x, dtype=np.float64)
        value, gradient = objective(coefficients)
        logits = prior_logit + matrix @ coefficients
        losses = np.logaddexp(0.0, logits) - labels * logits
        weighted_loss = float(np.dot(weights, losses) / total_weight)
        penalty = 0.5 * float(np.dot(penalties, coefficients * coefficients))
        correction = matrix @ coefficients
        market_totals = _decision_market_weight_totals(frame, weights)
        model = cls(
            coefficients=tuple(float(value) for value in coefficients),
            feature_scales=tuple(float(value) for value in scales),
            group_penalties=tuple(GROUP_PENALTIES.items()),
        )
        diagnostics = ResidualValueFitDiagnostics(
            converged=bool(result.success),
            optimizer_status=int(result.status),
            optimizer_message=str(result.message),
            iterations=int(result.nit),
            function_evaluations=int(result.nfev),
            decision_rows=frame.height,
            side_rows=frame.height * 2,
            markets=frame["market_id"].n_unique(),
            weighted_log_loss=weighted_loss,
            penalty=penalty,
            objective=float(value),
            gradient_infinity_norm=float(np.max(np.abs(gradient))),
            correction_l2_norm=float(np.linalg.norm(coefficients)),
            maximum_abs_correction=float(np.max(np.abs(correction))),
            market_weight_total_min=float(market_totals.min()),
            market_weight_total_max=float(market_totals.max()),
        )
        if not diagnostics.converged:
            raise RuntimeError(
                "side-conditioned residual fit did not converge: "
                f"{diagnostics.optimizer_message}"
            )
        return model, diagnostics

    def predict_yes_probability(self, frame: pl.DataFrame) -> np.ndarray:
        _validate_input_frame(frame, require_labels=False)
        prior = normalized_executable_cost_prior(frame)[:, 0]
        yes_prior_logit = np.log(prior) - np.log1p(-prior)
        matrix = _derive_yes_features(frame) / np.asarray(
            self.feature_scales,
            dtype=np.float64,
        )
        beta = np.asarray(self.coefficients, dtype=np.float64)
        yes_logits = yes_prior_logit + matrix @ beta
        # The frozen row contract derives NO as the exact negative of YES,
        # including its complementary price-prior offset.  Reconcile the pair
        # explicitly while retaining the compact decision-level computation.
        no_logits = -yes_logits
        coherent_yes_logit = 0.5 * (yes_logits - no_logits)
        probability = _sigmoid(coherent_yes_logit)
        if not np.isfinite(probability).all():
            raise RuntimeError("residual model produced non-finite YES probabilities")
        return np.clip(probability, PRIOR_CLIP, 1.0 - PRIOR_CLIP)

    def predict_side_probability(self, frame: pl.DataFrame) -> np.ndarray:
        yes = self.predict_yes_probability(frame)
        return np.column_stack((yes, 1.0 - yes))

    def manifest(self) -> dict[str, Any]:
        return {
            "model_class": type(self).__name__,
            "selection_eligible": self.selection_eligible,
            "runtime_exportable": self.runtime_exportable,
            "prior": {
                "definition": (
                    "side admission cost divided by YES plus NO admission costs"
                ),
                "clip": self.prior_clip,
                "offset": "log_odds",
            },
            "features": [asdict(spec) for spec in RESIDUAL_FEATURE_SPECS],
            "group_penalties": dict(self.group_penalties),
            "l2_not_weaker_than_anchor": (
                dict(self.group_penalties)[L2_CONFIRMATION]
                >= dict(self.group_penalties)[CORE_ORACLE_ANCHOR]
            ),
            "market_weighting": "each market totals one across all decisions and both sides",
            "pair_computation": (
                "exact antisymmetric YES/NO loss identity; compact decision-level fit"
            ),
            "random_seed": self.random_seed,
            "no_policy_or_pnl_objective": True,
            "no_forward_fill_or_interpolation": True,
            "causal_validation": {
                "polymarket": "receipt timestamps and derived book ages revalidated",
                "oracle": "source/block/decision ordering and derived age revalidated",
                "spot_l2": (
                    "source-event/availability/decision ordering, two-second "
                    "freshness, and derived ages revalidated"
                ),
            },
        }


def normalized_executable_cost_prior(frame: pl.DataFrame) -> np.ndarray:
    """Return complementary YES/NO priors from fee/reserve-inclusive costs."""

    _require_columns(frame, PRIOR_COLUMNS, "executable-price frame")
    costs = frame.select(*PRIOR_COLUMNS).to_numpy().astype(np.float64, copy=False)
    if costs.ndim != 2 or costs.shape[1] != 2 or not np.isfinite(costs).all():
        raise ValueError("executable PM costs must be a finite two-column matrix")
    if np.any(costs <= 0.0):
        raise ValueError("executable PM costs must be positive")
    totals = costs.sum(axis=1)
    if not np.isfinite(totals).all() or np.any(totals <= 0.0):
        raise ValueError("YES plus NO executable cost must be finite and positive")
    yes = np.clip(costs[:, 0] / totals, PRIOR_CLIP, 1.0 - PRIOR_CLIP)
    return np.column_stack((yes, 1.0 - yes))


def derive_side_conditioned_rows(
    frame: pl.DataFrame,
    *,
    require_labels: bool,
) -> SideConditionedRows:
    """Derive an exact antisymmetric YES/NO pair for every decision row."""

    _validate_input_frame(frame, require_labels=require_labels)
    decision_count = frame.height
    prior = normalized_executable_cost_prior(frame)
    yes_features = _derive_yes_features(frame)
    features = np.empty((decision_count * 2, yes_features.shape[1]), dtype=np.float64)
    features[0::2] = yes_features
    features[1::2] = -yes_features
    prior_probability = np.empty(decision_count * 2, dtype=np.float64)
    prior_probability[0::2] = prior[:, 0]
    prior_probability[1::2] = prior[:, 1]
    prior_logit = np.log(prior_probability) - np.log1p(-prior_probability)
    market_ids_by_decision = tuple(str(value) for value in frame["market_id"].to_list())
    market_ids = tuple(value for market in market_ids_by_decision for value in (market, market))
    weights = market_equal_side_weights(market_ids)
    labels: np.ndarray | None = None
    if require_labels:
        decision_labels = frame["label_up"].to_numpy().astype(np.float64, copy=False)
        labels = np.empty(decision_count * 2, dtype=np.float64)
        labels[0::2] = decision_labels
        labels[1::2] = 1.0 - decision_labels
    return SideConditionedRows(
        market_ids=market_ids,
        decision_indices=tuple(index for index in range(decision_count) for _ in SIDES),
        sides=SIDES * decision_count,
        prior_probability=prior_probability,
        prior_logit=prior_logit,
        features=features,
        row_weights=weights,
        labels=labels,
    )


def market_equal_side_weights(market_ids: Sequence[str]) -> np.ndarray:
    """Give every market total weight one across all decisions and sides."""

    if not market_ids:
        raise ValueError("market-equal weighting requires at least one side row")
    counts: dict[str, int] = {}
    for market_id in market_ids:
        value = str(market_id)
        if not value:
            raise ValueError("market ids must be non-empty")
        counts[value] = counts.get(value, 0) + 1
    return np.asarray([1.0 / counts[str(market_id)] for market_id in market_ids])


def _market_equal_decision_weights(frame: pl.DataFrame) -> np.ndarray:
    return (
        frame.select((1.0 / pl.len().over("market_id")).alias("weight"))["weight"]
        .to_numpy()
        .astype(np.float64, copy=False)
    )


def market_equal_decision_weights(frame: pl.DataFrame) -> np.ndarray:
    """Give every market total decision weight one."""

    return _market_equal_decision_weights(frame)


def _derive_yes_features(frame: pl.DataFrame) -> np.ndarray:
    elapsed = frame["seconds_elapsed"].to_numpy().astype(np.int64, copy=False)
    values: dict[str, np.ndarray] = {
        name: frame[name].to_numpy().astype(np.float64, copy=False)
        for name in NUMERIC_SOURCE_COLUMNS
    }

    def mature(seconds: int) -> np.ndarray:
        return (elapsed >= seconds).astype(np.float64)

    def mature_value(name: str, seconds: int) -> np.ndarray:
        available = elapsed >= seconds
        return np.where(available, values[name], 0.0)

    columns: list[np.ndarray] = [values["btc_path_from_window_open_bps"]]
    columns.extend(
        mature_value(f"btc_return_{seconds}s_bps", seconds)
        for seconds in (1, 5, 15, 30, 60)
    )
    columns.extend(
        mature_value(f"btc_signed_flow_{seconds}s", seconds)
        for seconds in (5, 30, 60)
    )
    columns.extend(
        (
            values["oracle_return_from_window_open_bps"],
            values["binance_oracle_basis_bps"],
            values["spot_l2_microprice_to_midpoint_bps"],
        )
    )
    columns.extend(
        values[f"spot_l2_midpoint_change_{seconds}s_bps"]
        for seconds in (1, 5, 15, 30, 60)
    )
    columns.extend(
        (
            values["spot_l2_imbalance_5"],
            values["spot_l2_imbalance_20"],
        )
    )

    core_sign = np.sign(values["btc_path_from_window_open_bps"])
    oracle_sign = np.sign(values["oracle_return_from_window_open_bps"])
    l2_sign = np.sign(values["spot_l2_microprice_to_midpoint_bps"])
    core_oracle_same = (core_sign == oracle_sign) & (core_sign != 0.0)
    core_l2_same = (core_sign == l2_sign) & (core_sign != 0.0)
    columns.extend(
        (
            (core_sign + oracle_sign + l2_sign) / 3.0,
            core_sign * core_oracle_same,
            core_sign * (~core_oracle_same) * (core_sign != 0.0) * (oracle_sign != 0.0),
            core_sign * core_l2_same,
            core_sign * (~core_l2_same) * (core_sign != 0.0) * (l2_sign != 0.0),
        )
    )
    columns.extend(mature(seconds) for seconds in (1, 5, 15, 30, 60))
    columns.extend(
        (
            values["pm_yes_depth_log"] - values["pm_no_depth_log"],
            values["pm_yes_vwap_slippage"] - values["pm_no_vwap_slippage"],
            values["pm_yes_book_age_seconds"] - values["pm_no_book_age_seconds"],
            values["pm_cost_overround"],
            values["oracle_age_seconds"] / 300.0,
            frame["early_oracle_eligible"].cast(pl.Float64).to_numpy(),
        )
    )
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    if matrix.shape != (frame.height, len(RESIDUAL_FEATURE_NAMES)):
        raise RuntimeError("derived residual features do not match the frozen manifest")
    if not np.isfinite(matrix).all():
        raise ValueError("derived residual features contain non-finite values")
    return matrix


def _validate_input_frame(frame: pl.DataFrame, *, require_labels: bool) -> None:
    columns = (*REQUIRED_FEATURE_COLUMNS, *(("label_up",) if require_labels else ()))
    _require_columns(frame, columns, "side-conditioned residual frame")
    if frame.is_empty():
        raise ValueError("side-conditioned residual frame is empty")
    if frame.select(pl.struct("market_id", "observed_at").n_unique()).item() != frame.height:
        raise ValueError("side-conditioned residual decisions must be unique")
    if frame.filter(pl.col("market_id").is_null() | (pl.col("market_id").str.len_chars() == 0)).height:
        raise ValueError("market_id must be non-null and non-empty")

    numeric = (
        *(
            name
            for name in NUMERIC_SOURCE_COLUMNS
            if name not in MATURITY_GATED_SOURCE_COLUMNS
        ),
        *L2_CAUSAL_AGE_COLUMNS,
        "seconds_elapsed",
    )
    non_finite = frame.filter(
        pl.any_horizontal(
            [
                pl.col(name).is_null() | ~pl.col(name).cast(pl.Float64).is_finite()
                for name in numeric
            ]
        )
    )
    if non_finite.height:
        raise ValueError("side-conditioned residual inputs contain missing or non-finite values")
    for name, minimum_elapsed in MATURITY_GATED_SOURCE_COLUMNS.items():
        invalid_mature = frame.filter(
            (pl.col(name).is_not_null() & ~pl.col(name).cast(pl.Float64).is_finite())
            | (
                (pl.col("seconds_elapsed") >= minimum_elapsed)
                & pl.col(name).is_null()
            )
        )
        if invalid_mature.height:
            raise ValueError(
                "side-conditioned residual mature inputs contain missing or "
                f"non-finite values: {name}"
            )
    elapsed = frame["seconds_elapsed"].cast(pl.Float64)
    if frame.filter(
        (pl.col("seconds_elapsed") < 1)
        | (pl.col("seconds_elapsed") > 299)
        | (elapsed != elapsed.floor())
    ).height:
        raise ValueError("seconds_elapsed must be an integer from 1 through 299")
    if frame.filter(
        pl.col("window_start").is_null()
        | pl.col("observed_at").is_null()
        | (
            (pl.col("observed_at") - pl.col("window_start")).dt.total_microseconds()
            != pl.col("seconds_elapsed") * 1_000_000
        )
    ).height:
        raise ValueError("decision timestamps must equal window start plus elapsed seconds")
    if frame.filter(
        pl.any_horizontal([pl.col(name).is_null() for name in CAUSAL_TIMESTAMP_COLUMNS])
        | (pl.col("yes_received_at") > pl.col("observed_at"))
        | (pl.col("no_received_at") > pl.col("observed_at"))
        | (pl.col("oracle_source_timestamp") > pl.col("oracle_block_timestamp"))
        | (pl.col("oracle_block_timestamp") > pl.col("observed_at"))
    ).height:
        raise ValueError("source timestamps are missing or noncausal")
    if frame.filter(
        (pl.col("spot_l2_source_event_timestamp") > pl.col("spot_l2_available_at"))
        | (pl.col("spot_l2_source_event_timestamp") >= pl.col("observed_at"))
        | (pl.col("spot_l2_available_at") >= pl.col("observed_at"))
    ).height:
        raise ValueError("spot-L2 timestamps are noncausal")
    if frame.filter(
        ~pl.col("spot_l2_availability_age_seconds").is_between(
            0.0,
            float(L2_MAXIMUM_CAUSAL_AGE_SECONDS),
            closed="right",
        )
        | ~pl.col("spot_l2_state_age_seconds").is_between(
            0.0,
            float(L2_MAXIMUM_CAUSAL_AGE_SECONDS),
            closed="right",
        )
    ).height:
        raise ValueError("spot-L2 ages violate the causal freshness contract")
    if frame.filter(~pl.col("early_oracle_eligible").fill_null(False)).height:
        raise ValueError("residual challenger requires a qualified causal Oracle join")
    if frame.filter(
        ~pl.col("oracle_age_seconds").is_between(2.0, 300.0, closed="both")
    ).height:
        raise ValueError("Oracle age must satisfy the causal two-to-300-second contract")
    derived_yes_age = (
        pl.col("observed_at") - pl.col("yes_received_at")
    ).dt.total_milliseconds().cast(pl.Float64) / 1_000.0
    derived_no_age = (
        pl.col("observed_at") - pl.col("no_received_at")
    ).dt.total_milliseconds().cast(pl.Float64) / 1_000.0
    derived_oracle_age = (
        pl.col("observed_at") - pl.col("oracle_block_timestamp")
    ).dt.total_seconds().cast(pl.Float64)
    derived_l2_availability_age = (
        (pl.col("observed_at") - pl.col("spot_l2_available_at"))
        .dt.total_microseconds()
        .cast(pl.Float64)
        / 1_000_000.0
    )
    derived_l2_state_age = (
        (pl.col("observed_at") - pl.col("spot_l2_source_event_timestamp"))
        .dt.total_microseconds()
        .cast(pl.Float64)
        / 1_000_000.0
    )
    if frame.filter(
        ((derived_yes_age - pl.col("pm_yes_book_age_seconds")).abs() > 1e-6)
        | ((derived_no_age - pl.col("pm_no_book_age_seconds")).abs() > 1e-6)
        | ((derived_oracle_age - pl.col("oracle_age_seconds")).abs() > 1e-6)
        | (
            (
                derived_l2_availability_age
                - pl.col("spot_l2_availability_age_seconds")
            ).abs()
            > 1e-6
        )
        | (
            (
                derived_l2_state_age
                - pl.col("spot_l2_state_age_seconds")
            ).abs()
            > 1e-6
        )
    ).height:
        raise ValueError("source ages disagree with their causal timestamps")
    if require_labels:
        labels = frame["label_up"].cast(pl.Float64)
        if frame.filter(labels.is_null() | ~labels.is_in((0.0, 1.0))).height:
            raise ValueError("label_up must be binary")


def _weighted_rms(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    scales = np.sqrt((weights[:, None] * matrix * matrix).sum(axis=0) / total)
    return np.maximum(scales, MINIMUM_SCALE)


def _penalty_vector() -> np.ndarray:
    return np.asarray([GROUP_PENALTIES[group] for group in RESIDUAL_FEATURE_GROUPS])


def _market_weight_totals(market_ids: Sequence[str], weights: np.ndarray) -> np.ndarray:
    totals: dict[str, float] = {}
    for market_id, weight in zip(market_ids, weights, strict=True):
        totals[market_id] = totals.get(market_id, 0.0) + float(weight)
    return np.asarray(list(totals.values()), dtype=np.float64)


def _decision_market_weight_totals(
    frame: pl.DataFrame,
    weights: np.ndarray,
) -> np.ndarray:
    return (
        frame.select("market_id")
        .with_columns(pl.Series("_weight", weights))
        .group_by("market_id")
        .agg(pl.col("_weight").sum())
        ["_weight"]
        .to_numpy()
        .astype(np.float64, copy=False)
    )


def _finite_vector(values: np.ndarray | None, *, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} are required")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector")
    return array


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -LOGIT_LIMIT, LOGIT_LIMIT)
    positive = clipped >= 0.0
    result = np.empty_like(clipped)
    result[positive] = 1.0 / (1.0 + np.exp(-clipped[positive]))
    exponent = np.exp(clipped[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: " + ", ".join(missing))


def feature_penalty_manifest() -> dict[str, Any]:
    """Return the immutable estimator-only feature and penalty contract."""

    return {
        "features": [asdict(spec) for spec in RESIDUAL_FEATURE_SPECS],
        "group_penalties": dict(GROUP_PENALTIES),
        "selection_eligible": False,
        "runtime_exportable": False,
    }
