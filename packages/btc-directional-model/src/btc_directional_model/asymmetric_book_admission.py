"""Offline residual admission models for the frozen asymmetric Core+Oracle incumbent.

The model in this module is deliberately an additive correction to the incumbent
YES logit.  It does not replace, refit, or export the incumbent estimator and is
not a runtime trading model.  Every fit is restricted to the incumbent's exact
20--30 cent, seconds 1--55 opportunity cohort.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from joblib import hash as joblib_hash
from scipy.optimize import minimize, minimize_scalar
from sklearn.ensemble import HistGradientBoostingRegressor

from .asymmetric_book_dynamics import (
    BOOK_DYNAMICS_FEATURES,
    BOOK_DYNAMICS_MATURITY_FEATURES,
    book_dynamics_schema_sha256,
)
from .asymmetric_value_data import POLYMARKET_VALUE_FEATURES

BOOK_ADMISSION_PROFILE = "btc_asymmetric_core_oracle_book_admission"
BOOK_ADMISSION_SCHEMA_VERSION = "btc-asymmetric-book-admission-v1"
BOOK_ADMISSION_KEY_COLUMNS = (
    "market_id",
    "window_start",
    "observed_at",
    "seconds_elapsed",
)
STATIC_FEATURES = tuple(POLYMARKET_VALUE_FEATURES)
DYNAMIC_FEATURES = tuple(BOOK_DYNAMICS_FEATURES)
EXPECTED_STATIC_FEATURES = 13
EXPECTED_DYNAMIC_FEATURES = 40
INCUMBENT_PROCESS_ID = "81f82de7-002b-4ac7-814b-236c6742d81c"
INCUMBENT_MODEL_KEY = "btc-5m-asymmetric-core-oracle-paper-20260805-v1"
INCUMBENT_MODEL_SHA256 = "2c91e894356f6fee7fe9514e24c39da6e11602ffcb7f961f64848bde72418db9"
INCUMBENT_FEATURE_SCHEMA_SHA256 = (
    "fe2a5aaee3df1ef899d2553712555091aa29b7481b3fed7805ba140dc8aa5014"
)
SOURCE_CONFIG_NAME = (
    "btc-5m-directional-asymmetric-value-calibrated-20260414-20260802.toml"
)
OOF_DAYS = tuple(
    date.fromisoformat(value)
    for value in (
        "2026-07-21",
        "2026-07-22",
        "2026-07-23",
        "2026-07-24",
        "2026-07-25",
        "2026-07-28",
        "2026-07-29",
        "2026-07-30",
        "2026-07-31",
        "2026-08-01",
    )
)
EXCLUDED_DAYS = (date(2026, 7, 26), date(2026, 7, 27))
MARKET_EQUAL = "market_equal"
DAY_MARKET_ROW_EQUAL = "day_market_row_equal"
PROBABILITY_COLUMN = "incumbent_probability_yes"
LABEL_COLUMN = "label_up"
SELECTED_SIDE_COLUMN = "selected_side"
SELECTED_SIDE_POLICY_ELIGIBLE_COLUMN = "selected_side_policy_eligible"
MINIMUM_IQR = 1e-12
NEWTON_RESPONSE_CAP = 5.0


@dataclass(frozen=True)
class EvidenceWindow:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class BookAdmissionFold:
    name: str
    fit: EvidenceWindow
    calibration: EvidenceWindow
    validation: EvidenceWindow


@dataclass(frozen=True)
class FrozenIncumbent:
    process_id: str
    model_key: str
    model_sha256: str
    feature_schema_sha256: str
    feature_count: int


@dataclass(frozen=True)
class BookAdmissionPolicy:
    name: str
    quantity: float
    vwap_quantity: float
    maximum_depth_participation: float
    book_freshness_seconds: float
    execution_reserve_per_share: float
    minimum_edge_per_share: float
    minimum_share_price: float
    maximum_share_price: float
    maximum_cost_per_share: float
    minimum_entry_second: int
    maximum_entry_second: int
    price_interval_closed: str


@dataclass(frozen=True)
class BookAdmissionCandidate:
    name: str
    feature_set: Literal["static", "dynamic"]
    estimator: Literal["linear", "histogram_gradient_boosting"]
    selection_eligible: bool
    weighting: Literal["market_equal", "day_market_row_equal"]
    l2_regularization: float
    learning_rate: float | None = None
    max_iter: int | None = None
    max_leaf_nodes: int | None = None
    min_samples_leaf: int | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return STATIC_FEATURES if self.feature_set == "static" else DYNAMIC_FEATURES


@dataclass(frozen=True)
class BookAdmissionModelContract:
    residual_logit_cap: float
    probability_clip: float
    gamma_minimum: float
    gamma_maximum: float
    preprocessing: str
    random_seed: int
    optimizer_max_iterations: int
    optimizer_ftol: float
    candidates: tuple[BookAdmissionCandidate, ...]


@dataclass(frozen=True)
class ProbabilityGates:
    maximum_paired_degradation_upper_95: float
    maximum_selected_opportunity_bias: float
    maximum_cell_bias: float
    minimum_noninferior_days: int
    required_comparison_days: int
    require_brier_noninferiority: bool
    require_log_loss_noninferiority: bool
    require_one_proper_score_improvement: bool
    require_dynamic_noninferiority_to_static: bool


@dataclass(frozen=True)
class CorrectionGates:
    minimum_net_corrected_decisions: int
    minimum_improvement_days: int
    minimum_candidate_accuracy: float
    minimum_accuracy_delta: float
    minimum_correctness_margin: float
    require_positive_candidate_only_stressed_expectancy: bool


@dataclass(frozen=True)
class EconomicGates:
    minimum_incumbent_frequency_fraction: float
    minimum_yes_entries: int
    minimum_no_entries: int
    minimum_stressed_expectancy_per_trade: float
    minimum_profit_factor: float
    minimum_profit_factor_fraction_of_incumbent: float
    maximum_mean_share_price: float
    maximum_loss_recovery_burden: float
    maximum_average_loss: float
    maximum_single_loss: float
    maximum_drawdown: float
    maximum_primary_metric_regression_fraction: float
    require_positive_both_sides: bool
    development_lower_95_is_report_only: bool


@dataclass(frozen=True)
class ReadinessGates:
    minimum_dynamic_coverage: float
    minimum_strict_markets: int
    minimum_target_markets: int
    maximum_book_age_seconds: float
    require_complete_exact_keys: bool
    require_no_duplicate_keys: bool
    require_no_future_data: bool


@dataclass(frozen=True)
class BookAdmissionPaths:
    source_asymmetric_value_config: Path
    incumbent_runtime_dir: Path
    incumbent_model: Path
    runs: Path


@dataclass(frozen=True)
class BookAdmissionConfig:
    source_path: Path
    package_root: Path
    profile: str
    paper_only: bool
    live_capital_allowed: bool
    runtime_deployable: bool
    batch_forward_eligible: bool
    evidence_scope: str
    process_change_allowed: bool
    incumbent: FrozenIncumbent
    development: EvidenceWindow
    calibration_days: int
    excluded_utc_days: tuple[date, ...]
    oof_utc_days: tuple[date, ...]
    folds: tuple[BookAdmissionFold, ...]
    final_fit: EvidenceWindow
    final_calibration: EvidenceWindow
    policy: BookAdmissionPolicy
    model: BookAdmissionModelContract
    probability_gates: ProbabilityGates
    correction_gates: CorrectionGates
    economic_gates: EconomicGates
    readiness_gates: ReadinessGates
    paths: BookAdmissionPaths

    def candidate(self, name: str) -> BookAdmissionCandidate:
        matches = [candidate for candidate in self.model.candidates if candidate.name == name]
        if len(matches) != 1:
            raise ValueError(f"unknown book-admission candidate: {name}")
        return matches[0]


@dataclass(frozen=True)
class RobustPreprocessor:
    feature_names: tuple[str, ...]
    medians: tuple[float, ...]
    iqrs: tuple[float, ...]

    @classmethod
    def fit(cls, frame: pl.DataFrame, feature_names: tuple[str, ...]) -> RobustPreprocessor:
        matrix = _feature_matrix(frame, feature_names)
        medians = np.median(matrix, axis=0)
        lower, upper = np.quantile(matrix, (0.25, 0.75), axis=0)
        iqrs = upper - lower
        iqrs = np.where(iqrs > MINIMUM_IQR, iqrs, 1.0)
        return cls(
            feature_names=feature_names,
            medians=tuple(float(value) for value in medians),
            iqrs=tuple(float(value) for value in iqrs),
        )

    def transform(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = _feature_matrix(frame, self.feature_names)
        return (matrix - np.asarray(self.medians)) / np.asarray(self.iqrs)


@dataclass(frozen=True)
class BookAdmissionFitEvidence:
    fit_key_sha256: str
    calibration_key_sha256: str
    fit_rows: int
    calibration_rows: int
    fit_markets: int
    calibration_markets: int
    fit_days: int
    calibration_days: int
    fit_weight_sha256: str
    calibration_weight_sha256: str
    gamma: float
    raw_residual_minimum: float
    raw_residual_median: float
    raw_residual_maximum: float
    residual_cap_hit_rate: float


@dataclass
class FittedBookAdmissionModel:
    schema_version: str
    candidate: BookAdmissionCandidate
    incumbent: FrozenIncumbent
    preprocessor: RobustPreprocessor
    residual_logit_cap: float
    probability_clip: float
    gamma: float
    linear_intercept: float | None
    linear_coefficients: tuple[float, ...] | None
    histogram_estimator: HistGradientBoostingRegressor | None
    dynamics_schema_sha256: str
    evidence: BookAdmissionFitEvidence

    def raw_residual(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(frame)
        if self.candidate.estimator == "linear":
            if self.linear_intercept is None or self.linear_coefficients is None:
                raise RuntimeError("linear book-admission model is incomplete")
            values = self.linear_intercept + matrix @ np.asarray(self.linear_coefficients)
        else:
            if self.histogram_estimator is None:
                raise RuntimeError("histogram book-admission model is incomplete")
            values = self.histogram_estimator.predict(matrix)
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (frame.height,) or not np.isfinite(values).all():
            raise RuntimeError("book-admission model produced invalid residual logits")
        return np.clip(values, -self.residual_logit_cap, self.residual_logit_cap)

    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        incumbent_yes = _incumbent_probabilities(frame, self.probability_clip)
        selected_yes = _selected_yes_mask(frame)
        incumbent_selected = np.where(selected_yes, incumbent_yes, 1.0 - incumbent_yes)
        logits = _logit(incumbent_selected) + self.gamma * self.raw_residual(frame)
        selected_probability = _sigmoid(logits)
        probability_yes = np.where(
            selected_yes,
            selected_probability,
            1.0 - selected_probability,
        )
        return np.clip(
            probability_yes,
            self.probability_clip,
            1.0 - self.probability_clip,
        )

    def semantic_payload(self) -> dict[str, Any]:
        estimator_sha = None
        if self.histogram_estimator is not None:
            estimator_sha = _histogram_estimator_semantic_sha256(
                self.histogram_estimator
            )
        return {
            "schema_version": self.schema_version,
            "candidate": asdict(self.candidate),
            "incumbent": asdict(self.incumbent),
            "preprocessor": asdict(self.preprocessor),
            "residual_logit_cap": self.residual_logit_cap,
            "probability_clip": self.probability_clip,
            "gamma": self.gamma,
            "linear_intercept": self.linear_intercept,
            "linear_coefficients": self.linear_coefficients,
            "histogram_estimator_sha256": estimator_sha,
            "dynamics_schema_sha256": self.dynamics_schema_sha256,
            "newton_response_cap": NEWTON_RESPONSE_CAP,
        }

    @property
    def semantic_sha256(self) -> str:
        return _canonical_sha256(self.semantic_payload())

    def serialize(self, path: Path) -> dict[str, Any]:
        return serialize_book_admission_model(self, path)


@dataclass(frozen=True)
class BookAdmissionScore:
    candidate_name: str
    frame: pl.DataFrame
    key_sha256: str
    probability_sha256: str
    model_semantic_sha256: str


@dataclass(frozen=True)
class BookAdmissionSupport:
    rows: int
    markets: int
    utc_days: int
    positive_selected_outcomes: int
    negative_selected_outcomes: int
    yes_oriented_rows: int
    no_oriented_rows: int
    key_sha256: str


def load_book_admission_config(path: Path) -> BookAdmissionConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    benchmark = raw["benchmark"]
    incumbent = raw["incumbent"]
    chronology = raw["chronology"]
    policy = raw["policy"]
    model = raw["model"]
    paths = raw["paths"]

    folds = tuple(
        BookAdmissionFold(
            name=str(value["name"]),
            fit=_window(value, "fit"),
            calibration=_window(value, "calibration"),
            validation=_window(value, "validation"),
        )
        for value in chronology["folds"]
    )
    final = chronology["final"]
    candidates = tuple(_load_candidate(value) for value in model["candidates"])
    config = BookAdmissionConfig(
        source_path=source_path,
        package_root=package_root,
        profile=str(benchmark["profile"]),
        paper_only=bool(benchmark["paper_only"]),
        live_capital_allowed=bool(benchmark["live_capital_allowed"]),
        runtime_deployable=bool(benchmark["runtime_deployable"]),
        batch_forward_eligible=bool(benchmark["batch_forward_eligible"]),
        evidence_scope=str(benchmark["evidence_scope"]),
        process_change_allowed=bool(benchmark["process_change_allowed"]),
        incumbent=FrozenIncumbent(
            process_id=str(incumbent["process_id"]),
            model_key=str(incumbent["model_key"]),
            model_sha256=str(incumbent["model_sha256"]),
            feature_schema_sha256=str(incumbent["feature_schema_sha256"]),
            feature_count=int(incumbent["feature_count"]),
        ),
        development=EvidenceWindow(
            _parse_utc(chronology["development_start"]),
            _parse_utc(chronology["development_end"]),
        ),
        calibration_days=int(chronology["calibration_days"]),
        excluded_utc_days=tuple(_parse_date(value) for value in chronology["excluded_utc_days"]),
        oof_utc_days=tuple(_parse_date(value) for value in chronology["oof_utc_days"]),
        folds=folds,
        final_fit=_window(final, "fit"),
        final_calibration=_window(final, "calibration"),
        policy=BookAdmissionPolicy(**policy),
        model=BookAdmissionModelContract(
            residual_logit_cap=float(model["residual_logit_cap"]),
            probability_clip=float(model["probability_clip"]),
            gamma_minimum=float(model["gamma_minimum"]),
            gamma_maximum=float(model["gamma_maximum"]),
            preprocessing=str(model["preprocessing"]),
            random_seed=int(model["random_seed"]),
            optimizer_max_iterations=int(model["optimizer_max_iterations"]),
            optimizer_ftol=float(model["optimizer_ftol"]),
            candidates=candidates,
        ),
        probability_gates=ProbabilityGates(**raw["probability_gates"]),
        correction_gates=CorrectionGates(**raw["correction_gates"]),
        economic_gates=EconomicGates(**raw["economic_gates"]),
        readiness_gates=ReadinessGates(**raw["readiness_gates"]),
        paths=BookAdmissionPaths(
            source_asymmetric_value_config=package_root
            / str(paths["source_asymmetric_value_config"]),
            incumbent_runtime_dir=package_root / str(paths["incumbent_runtime_dir"]),
            incumbent_model=package_root / str(paths["incumbent_model"]),
            runs=package_root / str(paths["runs"]),
        ),
    )
    validate_book_admission_config(config)
    return config


def validate_book_admission_config(config: BookAdmissionConfig) -> None:
    if (
        config.profile != BOOK_ADMISSION_PROFILE
        or not config.paper_only
        or config.live_capital_allowed
        or config.runtime_deployable
        or not config.batch_forward_eligible
        or config.process_change_allowed
        or config.evidence_scope != "consumed_cross_day_development"
    ):
        raise ValueError("book-admission benchmark deployment boundary changed")
    if config.incumbent != FrozenIncumbent(
        process_id=INCUMBENT_PROCESS_ID,
        model_key=INCUMBENT_MODEL_KEY,
        model_sha256=INCUMBENT_MODEL_SHA256,
        feature_schema_sha256=INCUMBENT_FEATURE_SCHEMA_SHA256,
        feature_count=75,
    ):
        raise ValueError("frozen book-admission incumbent changed")
    if config.paths.source_asymmetric_value_config.name != SOURCE_CONFIG_NAME:
        raise ValueError("book-admission source config changed")
    if not config.paths.source_asymmetric_value_config.is_file():
        raise ValueError("book-admission source config is missing")
    if config.paths.incumbent_model != config.paths.incumbent_runtime_dir / "model.json":
        raise ValueError("book-admission incumbent model path changed")
    if not config.paths.incumbent_model.is_file():
        raise ValueError("book-admission incumbent model is missing")
    if _file_sha256(config.paths.incumbent_model) != config.incumbent.model_sha256:
        raise RuntimeError("book-admission incumbent model bytes changed")
    if config.development != EvidenceWindow(_utc(2026, 4, 14), _utc(2026, 8, 2)):
        raise ValueError("book-admission development window changed")
    if config.calibration_days != 14 or config.excluded_utc_days != EXCLUDED_DAYS:
        raise ValueError("book-admission calibration/exclusion chronology changed")
    if config.oof_utc_days != OOF_DAYS or len(config.folds) != 10:
        raise ValueError("book-admission OOF day registry changed")
    for expected_day, fold in zip(OOF_DAYS, config.folds, strict=True):
        start = datetime.combine(expected_day, datetime.min.time(), tzinfo=UTC)
        if (
            fold.name != expected_day.isoformat()
            or fold.fit != EvidenceWindow(config.development.start, start - timedelta(days=14))
            or fold.calibration != EvidenceWindow(start - timedelta(days=14), start)
            or fold.validation != EvidenceWindow(start, start + timedelta(days=1))
        ):
            raise ValueError(f"book-admission fold chronology changed: {fold.name}")
    if config.final_fit != EvidenceWindow(_utc(2026, 4, 14), _utc(2026, 7, 16)):
        raise ValueError("book-admission final fit window changed")
    if config.final_calibration != EvidenceWindow(_utc(2026, 7, 16), _utc(2026, 8, 2)):
        raise ValueError("book-admission final calibration window changed")
    expected_policy = BookAdmissionPolicy(
        name="raw20_30_by55_edge_3c",
        quantity=5.0,
        vwap_quantity=5.0,
        maximum_depth_participation=0.25,
        book_freshness_seconds=2.0,
        execution_reserve_per_share=0.01,
        minimum_edge_per_share=0.03,
        minimum_share_price=0.20,
        maximum_share_price=0.30,
        maximum_cost_per_share=0.35,
        minimum_entry_second=1,
        maximum_entry_second=55,
        price_interval_closed="left",
    )
    if config.policy != expected_policy:
        raise ValueError("frozen book-admission policy changed")
    if (
        config.model.residual_logit_cap != 0.5
        or config.model.probability_clip != 1e-6
        or (config.model.gamma_minimum, config.model.gamma_maximum) != (0.0, 1.0)
        or config.model.preprocessing != "fit_window_median_iqr"
        or config.model.random_seed != 20260809
    ):
        raise ValueError("book-admission model safety contract changed")
    _validate_candidate_matrix(config.model.candidates)
    if len(STATIC_FEATURES) != EXPECTED_STATIC_FEATURES:
        raise RuntimeError("static book-admission feature count changed")
    if len(DYNAMIC_FEATURES) != EXPECTED_DYNAMIC_FEATURES:
        raise RuntimeError("dynamic book-admission feature count changed")
    _validate_gate_contract(config)


def select_target_opportunity_rows(
    frame: pl.DataFrame,
    config: BookAdmissionConfig,
    *,
    require_label: bool,
    feature_names: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Validate and select only seconds 1--55 with either raw VWAP5 in [0.20, 0.30)."""

    features = feature_names or STATIC_FEATURES
    _validate_source_frame(frame, config, require_label=require_label, feature_names=features)
    policy = config.policy
    excluded = [value.isoformat() for value in config.excluded_utc_days]
    selected = frame.filter(
        pl.col("seconds_elapsed").is_between(
            policy.minimum_entry_second,
            policy.maximum_entry_second,
            closed="both",
        )
        & (
            pl.col("yes_ask_vwap_5").is_between(
                policy.minimum_share_price,
                policy.maximum_share_price,
                closed="left",
            )
            | pl.col("no_ask_vwap_5").is_between(
                policy.minimum_share_price,
                policy.maximum_share_price,
                closed="left",
            )
        )
        & ~pl.col("window_start").dt.date().cast(pl.String).is_in(excluded)
    )
    if selected.is_empty():
        raise ValueError("book-admission frame has no target opportunity rows")
    return selected


def attach_frozen_policy_selected_side(
    frame: pl.DataFrame,
    config: BookAdmissionConfig,
) -> pl.DataFrame:
    """Orient the full strict grid to the frozen policy's executable-price side.

    This helper must run before ``attach_causal_book_dynamics`` and before target
    filtering.  When exactly one side is in the frozen raw-price interval that
    side is selected.  When both (or neither, for lag-only rows) are in range,
    the higher frozen-incumbent edge wins with YES as the deterministic tie break.
    Target training later rejects rows whose selected side is not price eligible.
    """

    _require_columns(
        frame,
        (
            *BOOK_ADMISSION_KEY_COLUMNS,
            "yes_ask_vwap_5",
            "no_ask_vwap_5",
            "yes_cost_per_share",
            "no_cost_per_share",
            PROBABILITY_COLUMN,
        ),
        "book-admission side orientation",
    )
    if frame.select(*BOOK_ADMISSION_KEY_COLUMNS).is_duplicated().any():
        raise ValueError("book-admission side orientation contains duplicate decision keys")
    probability = _incumbent_probabilities(frame, config.model.probability_clip)
    yes_price = frame["yes_ask_vwap_5"].cast(pl.Float64).to_numpy()
    no_price = frame["no_ask_vwap_5"].cast(pl.Float64).to_numpy()
    yes_cost = frame["yes_cost_per_share"].cast(pl.Float64).to_numpy()
    no_cost = frame["no_cost_per_share"].cast(pl.Float64).to_numpy()
    if not all(
        np.isfinite(values).all()
        for values in (yes_price, no_price, yes_cost, no_cost)
    ):
        raise ValueError("book-admission side orientation requires finite prices and costs")
    for side, canonical, feature in (
        ("YES", yes_cost, "pm_yes_cost_per_share"),
        ("NO", no_cost, "pm_no_cost_per_share"),
    ):
        if feature in frame.columns and not np.allclose(
            canonical,
            frame[feature].cast(pl.Float64).to_numpy(),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{side} canonical admission cost disagrees with PM features")
    minimum = config.policy.minimum_share_price
    maximum = config.policy.maximum_share_price
    yes_eligible = (yes_price >= minimum) & (yes_price < maximum)
    no_eligible = (no_price >= minimum) & (no_price < maximum)
    yes_edge = probability - yes_cost
    no_edge = (1.0 - probability) - no_cost
    choose_yes = (yes_eligible & ~no_eligible) | (
        (yes_eligible == no_eligible) & (yes_edge >= no_edge)
    )
    derived_side = np.where(choose_yes, "YES", "NO")
    derived_eligible = np.where(choose_yes, yes_eligible, no_eligible)
    if SELECTED_SIDE_COLUMN in frame.columns:
        observed = frame[SELECTED_SIDE_COLUMN].cast(pl.String).to_numpy()
        if not np.array_equal(observed, derived_side):
            raise ValueError("selected_side does not match the frozen policy orientation")
    if SELECTED_SIDE_POLICY_ELIGIBLE_COLUMN in frame.columns:
        observed_eligible = (
            frame[SELECTED_SIDE_POLICY_ELIGIBLE_COLUMN].cast(pl.Boolean).to_numpy()
        )
        if not np.array_equal(observed_eligible, derived_eligible):
            raise ValueError("selected-side eligibility does not match the frozen policy")
    return frame.with_columns(
        pl.Series(SELECTED_SIDE_COLUMN, derived_side, dtype=pl.String),
        pl.Series(
            SELECTED_SIDE_POLICY_ELIGIBLE_COLUMN,
            derived_eligible,
            dtype=pl.Boolean,
        ),
    )


def book_admission_support(
    frame: pl.DataFrame,
    config: BookAdmissionConfig,
    candidate_name: str,
    *,
    require_label: bool = True,
) -> BookAdmissionSupport:
    candidate = config.candidate(candidate_name)
    selected = select_target_opportunity_rows(
        frame,
        config,
        require_label=require_label,
        feature_names=candidate.feature_names,
    )
    labels = _selected_labels(selected) if require_label else np.asarray([], dtype=np.float64)
    return BookAdmissionSupport(
        rows=selected.height,
        markets=selected["market_id"].n_unique(),
        utc_days=selected["window_start"].dt.date().n_unique(),
        positive_selected_outcomes=int(labels.sum()) if require_label else 0,
        negative_selected_outcomes=int(selected.height - labels.sum()) if require_label else 0,
        yes_oriented_rows=int((selected[SELECTED_SIDE_COLUMN] == "YES").sum()),
        no_oriented_rows=int((selected[SELECTED_SIDE_COLUMN] == "NO").sum()),
        key_sha256=book_admission_key_sha256(selected),
    )


def book_admission_weights(frame: pl.DataFrame, weighting: str) -> np.ndarray:
    """Return exact equal-market or equal-day/equal-market/equal-row weights."""

    if frame.is_empty():
        raise ValueError("book-admission weighting frame is empty")
    market_ids = frame["market_id"].cast(pl.String).to_numpy()
    if weighting == MARKET_EQUAL:
        _, inverse, counts = np.unique(market_ids, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse]
    elif weighting == DAY_MARKET_ROW_EQUAL:
        days = frame["window_start"].dt.date().cast(pl.String).to_numpy()
        day_values, day_inverse, day_counts = np.unique(days, return_inverse=True, return_counts=True)
        del day_counts
        weights = np.zeros(frame.height, dtype=np.float64)
        for day_index in range(len(day_values)):
            day_mask = day_inverse == day_index
            day_markets = market_ids[day_mask]
            unique_markets, inverse, counts = np.unique(
                day_markets, return_inverse=True, return_counts=True
            )
            weights[day_mask] = 1.0 / (len(day_values) * len(unique_markets) * counts[inverse])
    else:
        raise ValueError(f"unsupported book-admission weighting: {weighting}")
    weights *= frame.height / weights.sum()
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise RuntimeError("book-admission weights are invalid")
    return weights


def fit_book_admission_candidate(
    config: BookAdmissionConfig,
    candidate_name: str,
    fit_frame: pl.DataFrame,
    calibration_frame: pl.DataFrame,
) -> FittedBookAdmissionModel:
    """Fit one frozen candidate and its calibration-window scalar shrinkage."""

    candidate = config.candidate(candidate_name)
    feature_names = candidate.feature_names
    fit = select_target_opportunity_rows(
        fit_frame, config, require_label=True, feature_names=feature_names
    )
    calibration = select_target_opportunity_rows(
        calibration_frame, config, require_label=True, feature_names=feature_names
    )
    if fit["window_start"].max() >= calibration["window_start"].min():
        raise ValueError("book-admission fit evidence must strictly precede calibration evidence")
    fit_labels = _selected_labels(fit)
    calibration_labels = _selected_labels(calibration)
    _require_both_outcomes(fit_labels, "fit")
    _require_both_outcomes(calibration_labels, "calibration")
    preprocessor = RobustPreprocessor.fit(fit, feature_names)
    fit_matrix = preprocessor.transform(fit)
    fit_probability = _incumbent_selected_probabilities(
        fit, config.model.probability_clip
    )
    fit_weights = book_admission_weights(fit, candidate.weighting)

    linear_intercept: float | None = None
    linear_coefficients: tuple[float, ...] | None = None
    histogram: HistGradientBoostingRegressor | None = None
    if candidate.estimator == "linear":
        linear_intercept, linear_coefficients = _fit_linear_residual(
            fit_matrix,
            fit_labels,
            fit_probability,
            fit_weights,
            candidate,
            config,
        )
        fit_raw = linear_intercept + fit_matrix @ np.asarray(linear_coefficients)
    else:
        histogram = _fit_histogram_residual(
            fit_matrix,
            fit_labels,
            fit_probability,
            fit_weights,
            candidate,
            config,
        )
        fit_raw = histogram.predict(fit_matrix)
    fit_raw = np.clip(fit_raw, -config.model.residual_logit_cap, config.model.residual_logit_cap)

    calibration_matrix = preprocessor.transform(calibration)
    if candidate.estimator == "linear":
        calibration_raw = linear_intercept + calibration_matrix @ np.asarray(
            linear_coefficients
        )
    else:
        if histogram is None:
            raise RuntimeError("histogram residual fit was not materialized")
        calibration_raw = histogram.predict(calibration_matrix)
    calibration_raw = np.clip(
        calibration_raw,
        -config.model.residual_logit_cap,
        config.model.residual_logit_cap,
    )
    calibration_probability = _incumbent_selected_probabilities(
        calibration, config.model.probability_clip
    )
    calibration_weights = book_admission_weights(calibration, candidate.weighting)
    gamma = _fit_gamma(
        _logit(calibration_probability),
        calibration_raw,
        calibration_labels,
        calibration_weights,
        config,
    )
    evidence = BookAdmissionFitEvidence(
        fit_key_sha256=book_admission_key_sha256(fit),
        calibration_key_sha256=book_admission_key_sha256(calibration),
        fit_rows=fit.height,
        calibration_rows=calibration.height,
        fit_markets=fit["market_id"].n_unique(),
        calibration_markets=calibration["market_id"].n_unique(),
        fit_days=fit["window_start"].dt.date().n_unique(),
        calibration_days=calibration["window_start"].dt.date().n_unique(),
        fit_weight_sha256=_array_sha256(fit_weights),
        calibration_weight_sha256=_array_sha256(calibration_weights),
        gamma=gamma,
        raw_residual_minimum=float(np.min(fit_raw)),
        raw_residual_median=float(np.median(fit_raw)),
        raw_residual_maximum=float(np.max(fit_raw)),
        residual_cap_hit_rate=float(
            np.mean(np.abs(fit_raw) >= config.model.residual_logit_cap - 1e-12)
        ),
    )
    return FittedBookAdmissionModel(
        schema_version=BOOK_ADMISSION_SCHEMA_VERSION,
        candidate=candidate,
        incumbent=config.incumbent,
        preprocessor=preprocessor,
        residual_logit_cap=config.model.residual_logit_cap,
        probability_clip=config.model.probability_clip,
        gamma=gamma,
        linear_intercept=linear_intercept,
        linear_coefficients=linear_coefficients,
        histogram_estimator=histogram,
        dynamics_schema_sha256=book_dynamics_schema_sha256(),
        evidence=evidence,
    )


def score_book_admission_model(
    model: FittedBookAdmissionModel,
    frame: pl.DataFrame,
    config: BookAdmissionConfig,
) -> BookAdmissionScore:
    if model.incumbent != config.incumbent:
        raise RuntimeError("book-admission model does not belong to the configured incumbent")
    if model.dynamics_schema_sha256 != book_dynamics_schema_sha256():
        raise RuntimeError("book-admission dynamics schema changed")
    selected = select_target_opportunity_rows(
        frame,
        config,
        require_label=False,
        feature_names=model.preprocessor.feature_names,
    )
    probabilities = model.probability(selected)
    scored = selected.select(*BOOK_ADMISSION_KEY_COLUMNS).with_columns(
        pl.lit(model.candidate.name).alias("candidate_name"),
        pl.Series(PROBABILITY_COLUMN, probabilities, dtype=pl.Float64),
    )
    return BookAdmissionScore(
        candidate_name=model.candidate.name,
        frame=scored,
        key_sha256=book_admission_key_sha256(scored),
        probability_sha256=book_admission_probability_sha256(scored),
        model_semantic_sha256=model.semantic_sha256,
    )


def serialize_book_admission_model(
    model: FittedBookAdmissionModel, path: Path
) -> dict[str, Any]:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        "schema_version": BOOK_ADMISSION_SCHEMA_VERSION,
        "model_semantic_sha256": model.semantic_sha256,
        "candidate_name": model.candidate.name,
        "incumbent_model_sha256": model.incumbent.model_sha256,
        "dynamics_schema_sha256": model.dynamics_schema_sha256,
    }
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        pickle.dump({"model": model, "evidence": evidence}, handle, protocol=5)
    temporary.replace(destination)
    evidence["artifact_sha256"] = _file_sha256(destination)
    return evidence


def load_book_admission_model(path: Path) -> FittedBookAdmissionModel:
    with path.resolve().open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("model"), FittedBookAdmissionModel
    ):
        raise TypeError("book-admission artifact is unsupported")
    model = payload["model"]
    evidence = payload.get("evidence", {})
    if (
        evidence.get("schema_version") != BOOK_ADMISSION_SCHEMA_VERSION
        or evidence.get("model_semantic_sha256") != model.semantic_sha256
        or evidence.get("incumbent_model_sha256") != model.incumbent.model_sha256
        or evidence.get("dynamics_schema_sha256") != model.dynamics_schema_sha256
    ):
        raise RuntimeError("book-admission artifact semantic evidence changed")
    return model


def book_admission_key_sha256(frame: pl.DataFrame) -> str:
    _require_columns(frame, BOOK_ADMISSION_KEY_COLUMNS, "book-admission key evidence")
    ordered = frame.select(*BOOK_ADMISSION_KEY_COLUMNS).sort(*BOOK_ADMISSION_KEY_COLUMNS)
    digest = hashlib.sha256(b"btc-asymmetric-book-admission-key-v1\n")
    for row in ordered.iter_rows():
        for value in row:
            _update_digest(digest, value)
    return digest.hexdigest()


def book_admission_probability_sha256(frame: pl.DataFrame) -> str:
    _require_columns(
        frame,
        (*BOOK_ADMISSION_KEY_COLUMNS, PROBABILITY_COLUMN),
        "book-admission probability evidence",
    )
    ordered = frame.sort(*BOOK_ADMISSION_KEY_COLUMNS)
    values = ordered[PROBABILITY_COLUMN].cast(pl.Float64).to_numpy()
    if not np.isfinite(values).all() or np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("book-admission probability evidence is invalid")
    digest = hashlib.sha256(b"btc-asymmetric-book-admission-probability-v1\n")
    digest.update(book_admission_key_sha256(ordered).encode())
    digest.update(values.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _load_candidate(raw: dict[str, Any]) -> BookAdmissionCandidate:
    return BookAdmissionCandidate(
        name=str(raw["name"]),
        feature_set=str(raw["feature_set"]),  # type: ignore[arg-type]
        estimator=str(raw["estimator"]),  # type: ignore[arg-type]
        selection_eligible=bool(raw["selection_eligible"]),
        weighting=str(raw["weighting"]),  # type: ignore[arg-type]
        l2_regularization=float(raw["l2_regularization"]),
        learning_rate=(float(raw["learning_rate"]) if "learning_rate" in raw else None),
        max_iter=(int(raw["max_iter"]) if "max_iter" in raw else None),
        max_leaf_nodes=(int(raw["max_leaf_nodes"]) if "max_leaf_nodes" in raw else None),
        min_samples_leaf=(
            int(raw["min_samples_leaf"]) if "min_samples_leaf" in raw else None
        ),
    )


def _validate_candidate_matrix(candidates: tuple[BookAdmissionCandidate, ...]) -> None:
    expected = (
        BookAdmissionCandidate("S0", "static", "linear", False, MARKET_EQUAL, 30.0),
        BookAdmissionCandidate("D1", "dynamic", "linear", True, MARKET_EQUAL, 30.0),
        BookAdmissionCandidate("D2", "dynamic", "linear", True, MARKET_EQUAL, 10.0),
        BookAdmissionCandidate("D3", "dynamic", "linear", True, DAY_MARKET_ROW_EQUAL, 30.0),
        BookAdmissionCandidate(
            "D4",
            "dynamic",
            "histogram_gradient_boosting",
            True,
            MARKET_EQUAL,
            20.0,
            0.03,
            120,
            5,
            250,
        ),
    )
    if candidates != expected:
        raise ValueError("book-admission candidate matrix changed")


def _validate_gate_contract(config: BookAdmissionConfig) -> None:
    probability = config.probability_gates
    correction = config.correction_gates
    economic = config.economic_gates
    readiness = config.readiness_gates
    if (
        probability.maximum_paired_degradation_upper_95 > 0.005
        or probability.maximum_selected_opportunity_bias > 0.03
        or probability.maximum_cell_bias > 0.05
        or probability.minimum_noninferior_days < 8
        or probability.required_comparison_days != 10
        or not all(
            (
                probability.require_brier_noninferiority,
                probability.require_log_loss_noninferiority,
                probability.require_one_proper_score_improvement,
                probability.require_dynamic_noninferiority_to_static,
            )
        )
    ):
        raise ValueError("book-admission probability gates weakened")
    if (
        correction.minimum_net_corrected_decisions < 3
        or correction.minimum_improvement_days < 3
        or correction.minimum_candidate_accuracy < 0.315
        or correction.minimum_accuracy_delta < 0.02
        or correction.minimum_correctness_margin < 0.05
        or not correction.require_positive_candidate_only_stressed_expectancy
    ):
        raise ValueError("book-admission correction gates weakened")
    if (
        economic.minimum_incumbent_frequency_fraction < 0.80
        or economic.minimum_yes_entries < 20
        or economic.minimum_no_entries < 20
        or economic.minimum_stressed_expectancy_per_trade < 0.0
        or economic.minimum_profit_factor < 1.05
        or economic.minimum_profit_factor_fraction_of_incumbent < 0.90
        or economic.maximum_mean_share_price > 0.2466
        or economic.maximum_loss_recovery_burden > 0.40
        or economic.maximum_average_loss > 1.390
        or economic.maximum_single_loss > 1.674
        or economic.maximum_drawdown > 19.18
        or economic.maximum_primary_metric_regression_fraction > 0.10
        or not economic.require_positive_both_sides
        or not economic.development_lower_95_is_report_only
    ):
        raise ValueError("book-admission economic gates weakened")
    if (
        readiness.minimum_dynamic_coverage < 0.90
        or readiness.minimum_strict_markets < 2000
        or readiness.minimum_target_markets < 750
        or readiness.maximum_book_age_seconds > 2.0
        or not all(
            (
                readiness.require_complete_exact_keys,
                readiness.require_no_duplicate_keys,
                readiness.require_no_future_data,
            )
        )
    ):
        raise ValueError("book-admission readiness gates weakened")


def _validate_source_frame(
    frame: pl.DataFrame,
    config: BookAdmissionConfig,
    *,
    require_label: bool,
    feature_names: tuple[str, ...],
) -> None:
    if frame.is_empty():
        raise ValueError("book-admission source frame is empty")
    required = {
        *BOOK_ADMISSION_KEY_COLUMNS,
        "yes_ask_vwap_5",
        "no_ask_vwap_5",
        "yes_cost_per_share",
        "no_cost_per_share",
        PROBABILITY_COLUMN,
        SELECTED_SIDE_COLUMN,
        *feature_names,
    }
    if require_label:
        required.add(LABEL_COLUMN)
    _require_columns(frame, tuple(sorted(required)), "book-admission source")
    if frame.select(*BOOK_ADMISSION_KEY_COLUMNS).is_duplicated().any():
        raise ValueError("book-admission source contains duplicate decision keys")
    for timestamp in ("window_start", "observed_at"):
        dtype = frame.schema[timestamp]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
            raise TypeError(f"{timestamp} must be a UTC Datetime")
    if not frame.schema["seconds_elapsed"].is_integer():
        raise TypeError("seconds_elapsed must be an integer")
    misaligned = frame.filter(
        (pl.col("observed_at") - pl.col("window_start")).dt.total_microseconds()
        != pl.col("seconds_elapsed").cast(pl.Int64) * 1_000_000
    )
    if misaligned.height:
        raise ValueError("book-admission timestamps are non-causal or misaligned")
    numeric = (
        *feature_names,
        "yes_ask_vwap_5",
        "no_ask_vwap_5",
        "yes_cost_per_share",
        "no_cost_per_share",
        PROBABILITY_COLUMN,
    )
    invalid = frame.filter(
        ~pl.all_horizontal(
            pl.col(column).is_not_null() & pl.col(column).cast(pl.Float64).is_finite()
            for column in numeric
        )
    )
    if invalid.height:
        raise ValueError("book-admission source contains unsupported non-finite values")
    for maturity in BOOK_DYNAMICS_MATURITY_FEATURES:
        if maturity in feature_names and frame.schema[maturity] != pl.Boolean:
            raise TypeError(f"{maturity} must be boolean")
    prices = frame.select("yes_ask_vwap_5", "no_ask_vwap_5").to_numpy()
    if np.any((prices < 0.0) | (prices > 1.0)):
        raise ValueError("book-admission raw prices must be within [0, 1]")
    _incumbent_probabilities(frame, config.model.probability_clip)
    attach_frozen_policy_selected_side(frame, config)
    if require_label:
        labels = _labels(frame)
        if not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError("book-admission labels must be binary")
        inconsistent = (
            frame.group_by("market_id")
            .agg(
                pl.col("window_start").n_unique().alias("starts"),
                pl.col(LABEL_COLUMN).n_unique().alias("labels"),
            )
            .filter((pl.col("starts") != 1) | (pl.col("labels") != 1))
        )
        if inconsistent.height:
            raise ValueError("book-admission market identity or label is inconsistent")
    _validate_book_causality(frame, config.readiness_gates.maximum_book_age_seconds)


def _validate_book_causality(frame: pl.DataFrame, maximum_age: float) -> None:
    for age in ("pm_yes_book_age_seconds", "pm_no_book_age_seconds"):
        if age in frame.columns and frame.filter(
            ~pl.col(age).is_between(0.0, maximum_age, closed="both")
        ).height:
            raise ValueError("book-admission source contains non-causal or stale books")
    receipt_columns = ("yes_received_at", "no_received_at")
    present = [column in frame.columns for column in receipt_columns]
    if any(present) and not all(present):
        raise ValueError("book-admission source must include both receipt timestamps or neither")
    for receipt in receipt_columns if all(present) else ():
        dtype = frame.schema[receipt]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
            raise TypeError(f"{receipt} must be a UTC Datetime")
        if frame.filter(
            pl.col(receipt).is_null()
            | (pl.col(receipt) > pl.col("observed_at"))
            | (
                (pl.col("observed_at") - pl.col(receipt)).dt.total_microseconds()
                > maximum_age * 1_000_000
            )
        ).height:
            raise ValueError("book-admission source contains a future or stale receipt")


def _fit_linear_residual(
    matrix: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    candidate: BookAdmissionCandidate,
    config: BookAdmissionConfig,
) -> tuple[float, tuple[float, ...]]:
    parent_logit = _logit(probabilities)
    rows = matrix.shape[0]

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        raw = parameters[0] + matrix @ parameters[1:]
        clipped = np.clip(raw, -config.model.residual_logit_cap, config.model.residual_logit_cap)
        eta = parent_logit + clipped
        probability = _sigmoid(eta)
        loss = np.sum(weights * (np.logaddexp(0.0, eta) - labels * eta)) / weights.sum()
        penalty = 0.5 * candidate.l2_regularization * np.dot(
            parameters[1:], parameters[1:]
        ) / rows
        active = np.abs(raw) < config.model.residual_logit_cap
        score = weights * (probability - labels) * active / weights.sum()
        gradient = np.empty_like(parameters)
        gradient[0] = score.sum()
        gradient[1:] = matrix.T @ score + candidate.l2_regularization * parameters[1:] / rows
        return float(loss + penalty), gradient

    result = minimize(
        objective,
        np.zeros(matrix.shape[1] + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": config.model.optimizer_max_iterations,
            "ftol": config.model.optimizer_ftol,
        },
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"{candidate.name} residual optimizer failed: {result.message}")
    return float(result.x[0]), tuple(float(value) for value in result.x[1:])


def _fit_histogram_residual(
    matrix: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    candidate: BookAdmissionCandidate,
    config: BookAdmissionConfig,
) -> HistGradientBoostingRegressor:
    if None in (
        candidate.learning_rate,
        candidate.max_iter,
        candidate.max_leaf_nodes,
        candidate.min_samples_leaf,
    ):
        raise ValueError("histogram book-admission candidate is incomplete")
    variance = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
    response = np.clip((labels - probabilities) / variance, -NEWTON_RESPONSE_CAP, NEWTON_RESPONSE_CAP)
    estimator = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(candidate.learning_rate),
        max_iter=int(candidate.max_iter),
        max_leaf_nodes=int(candidate.max_leaf_nodes),
        min_samples_leaf=int(candidate.min_samples_leaf),
        l2_regularization=candidate.l2_regularization,
        early_stopping=False,
        random_state=config.model.random_seed,
    )
    estimator.fit(matrix, response, sample_weight=weights * variance)
    predicted = estimator.predict(matrix)
    if not np.isfinite(predicted).all():
        raise RuntimeError("histogram book-admission fit produced non-finite residuals")
    return estimator


def _fit_gamma(
    parent_logit: np.ndarray,
    residual: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    config: BookAdmissionConfig,
) -> float:
    def objective(gamma: float) -> float:
        eta = parent_logit + gamma * residual
        return float(
            np.sum(weights * (np.logaddexp(0.0, eta) - labels * eta)) / weights.sum()
        )

    result = minimize_scalar(
        objective,
        bounds=(config.model.gamma_minimum, config.model.gamma_maximum),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": config.model.optimizer_max_iterations},
    )
    choices = (config.model.gamma_minimum, float(result.x), config.model.gamma_maximum)
    gamma = min(choices, key=lambda value: (objective(value), value))
    if not result.success or not math.isfinite(gamma):
        raise RuntimeError("book-admission gamma optimization failed")
    return float(gamma)


def _feature_matrix(frame: pl.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    _require_columns(frame, feature_names, "book-admission feature matrix")
    matrix = np.asarray(frame.select(*feature_names).to_numpy(), dtype=np.float64)
    if matrix.shape != (frame.height, len(feature_names)) or not np.isfinite(matrix).all():
        raise ValueError("book-admission feature matrix is unsupported")
    return matrix


def _incumbent_probabilities(frame: pl.DataFrame, probability_clip: float) -> np.ndarray:
    values = frame[PROBABILITY_COLUMN].cast(pl.Float64).to_numpy()
    if not np.isfinite(values).all() or np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("incumbent YES probabilities must be finite and strictly within (0, 1)")
    return np.clip(values, probability_clip, 1.0 - probability_clip)


def _incumbent_selected_probabilities(
    frame: pl.DataFrame, probability_clip: float
) -> np.ndarray:
    probability_yes = _incumbent_probabilities(frame, probability_clip)
    return np.where(_selected_yes_mask(frame), probability_yes, 1.0 - probability_yes)


def _labels(frame: pl.DataFrame) -> np.ndarray:
    return frame[LABEL_COLUMN].cast(pl.Float64).to_numpy()


def _selected_labels(frame: pl.DataFrame) -> np.ndarray:
    labels_yes = _labels(frame)
    return np.where(_selected_yes_mask(frame), labels_yes, 1.0 - labels_yes)


def _selected_yes_mask(frame: pl.DataFrame) -> np.ndarray:
    _require_columns(frame, (SELECTED_SIDE_COLUMN,), "book-admission side orientation")
    sides = frame[SELECTED_SIDE_COLUMN].cast(pl.String).to_numpy()
    if not np.isin(sides, ("YES", "NO")).all():
        raise ValueError("book-admission selected_side must contain only YES or NO")
    return sides == "YES"


def _require_both_outcomes(labels: np.ndarray, label: str) -> None:
    if set(np.unique(labels).tolist()) != {0.0, 1.0}:
        raise ValueError(f"book-admission {label} evidence must contain both outcomes")


def _require_columns(frame: pl.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _window(raw: dict[str, Any], prefix: str) -> EvidenceWindow:
    return EvidenceWindow(
        _parse_utc(raw[f"{prefix}_start"]),
        _parse_utc(raw[f"{prefix}_end"]),
    )


def _parse_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("book-admission timestamps must be UTC")
    return parsed.astimezone(UTC)


def _parse_date(value: Any) -> date:
    return value if isinstance(value, date) and not isinstance(value, datetime) else date.fromisoformat(str(value))


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _logit(values: np.ndarray) -> np.ndarray:
    return np.log(values) - np.log1p(-values)


def _histogram_estimator_semantic_sha256(
    estimator: HistGradientBoostingRegressor,
) -> str:
    """Hash fitted estimator state without relying on unstable pickle memoization."""

    canonical_estimator = pickle.loads(pickle.dumps(estimator, protocol=5))
    state_token = joblib_hash(
        canonical_estimator,
        hash_name="sha1",
        coerce_mmap=True,
    )
    digest = hashlib.sha256(b"btc-asymmetric-book-admission-hgb-state-v1\n")
    digest.update(state_token.encode())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<f8").tobytes()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, datetime):
        rendered = value.astimezone(UTC).isoformat(timespec="microseconds")
    else:
        rendered = str(value)
    encoded = rendered.encode()
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
