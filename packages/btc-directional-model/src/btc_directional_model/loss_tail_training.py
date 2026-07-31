from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from .core_config import CoreTrainingConfig
from .core_features import (
    CORE_BOUNDARY_ENRICHED_FEATURES,
    CORE_MATURE_REVERSAL_FEATURES,
    CORE_ORACLE_FEATURES,
)
from .core_training import (
    MARKET_EQUAL_ROW_WEIGHT_POLICY,
    MARKET_EQUAL_ROW_WEIGHT_SCHEDULE,
    FittedCoreModel,
    ProbabilityCalibrator,
    chronological_inner_split,
    feature_matrix,
    finite_medians,
    histogram_parameters,
    market_equal_weights,
)
from .offline_challengers import STRICT_BOOK_FIVE_SHARE_V2_FEATURES

DIRECT_LOSS_TAIL_HGB_CANDIDATE = "direct_loss_tail_hgb"
EXTRA_TREES_RARE_REGIME_CANDIDATE = "extra_trees_rare_regime"
BOUNDARY_CORRECTNESS_LOGISTIC_CANDIDATE = "boundary_correctness_logistic"

EXPECTED_LOSS_TAIL_FEATURE_COUNT = 124
LOSS_TAIL_FEATURES = (
    *CORE_BOUNDARY_ENRICHED_FEATURES,
    *CORE_MATURE_REVERSAL_FEATURES,
    *CORE_ORACLE_FEATURES,
    *STRICT_BOOK_FIVE_SHARE_V2_FEATURES,
)


@dataclass(frozen=True)
class ExtraTreesHyperparameters:
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    max_features: float | str

    def __post_init__(self) -> None:
        if self.n_estimators <= 0:
            raise ValueError("ExtraTrees n_estimators must be positive")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("ExtraTrees max_depth must be positive when configured")
        if self.min_samples_leaf <= 0:
            raise ValueError("ExtraTrees min_samples_leaf must be positive")
        if isinstance(self.max_features, float) and not 0.0 < self.max_features <= 1.0:
            raise ValueError("ExtraTrees numeric max_features must be in (0, 1]")
        if isinstance(self.max_features, str) and self.max_features not in {"sqrt", "log2"}:
            raise ValueError("ExtraTrees string max_features must be sqrt or log2")


DEFAULT_EXTRA_TREES_CANDIDATES = (
    ExtraTreesHyperparameters(384, 14, 12, "sqrt"),
    ExtraTreesHyperparameters(384, 18, 20, 0.35),
    ExtraTreesHyperparameters(512, None, 32, 0.50),
)


@dataclass
class CalibratedCandidateResult:
    """A benchmark-only fitted classifier with an unweighted Platt layer."""

    model: FittedCoreModel
    calibrator: ProbabilityCalibrator
    target_column: str
    tuning: dict[str, Any]

    @property
    def candidate_name(self) -> str:
        return self.model.candidate_name

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.model.feature_names

    def raw_probability(self, frame: pl.DataFrame) -> np.ndarray:
        return self.model.raw_probability(frame)

    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        return self.calibrator.probability(self.model.raw_logit(frame))


def validate_loss_tail_feature_contract(
    frame: pl.DataFrame | None = None,
) -> tuple[str, ...]:
    """Validate the frozen 68 + 13 + 11 + 32 causal feature contract."""

    component_lengths = (
        len(CORE_BOUNDARY_ENRICHED_FEATURES),
        len(CORE_MATURE_REVERSAL_FEATURES),
        len(CORE_ORACLE_FEATURES),
        len(STRICT_BOOK_FIVE_SHARE_V2_FEATURES),
    )
    if component_lengths != (68, 13, 11, 32):
        raise RuntimeError(
            "loss-tail feature component drift: expected 68 + 13 + 11 + 32, "
            f"found {' + '.join(str(value) for value in component_lengths)}"
        )
    if len(LOSS_TAIL_FEATURES) != EXPECTED_LOSS_TAIL_FEATURE_COUNT:
        raise RuntimeError(
            "loss-tail feature count drift: "
            f"expected {EXPECTED_LOSS_TAIL_FEATURE_COUNT}, found {len(LOSS_TAIL_FEATURES)}"
        )
    duplicates = _duplicates(LOSS_TAIL_FEATURES)
    if duplicates:
        raise RuntimeError("loss-tail feature blocks overlap: " + ", ".join(duplicates))
    if frame is not None:
        _validate_feature_frame(frame, LOSS_TAIL_FEATURES)
    return LOSS_TAIL_FEATURES


def wrong_side_economic_severity(
    frame: pl.DataFrame,
    *,
    label_column: str = "label_up",
    up_debit_column: str = "up_entry_debit_per_share",
    down_debit_column: str = "down_entry_debit_per_share",
) -> np.ndarray:
    """Return capped loss-to-upside severity for the direction opposite the label.

    Entry debit must already include executable VWAP and fees.  If UP wins, the
    wrong-side debit is DOWN; if DOWN wins, it is UP.
    """

    required = {label_column, up_debit_column, down_debit_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("economic-severity frame is missing columns: " + ", ".join(missing))
    labels = _validated_binary_target(frame, label_column)
    up_debit = _validated_debit(frame, up_debit_column)
    down_debit = _validated_debit(frame, down_debit_column)
    wrong_debit = np.where(labels == 1, down_debit, up_debit)
    severity = wrong_debit / np.maximum(1.0 - wrong_debit, 0.05)
    return np.clip(severity, 1.0, 10.0)


def loss_tail_training_weights(
    frame: pl.DataFrame,
    *,
    label_column: str = "label_up",
    up_debit_column: str = "up_entry_debit_per_share",
    down_debit_column: str = "down_entry_debit_per_share",
) -> np.ndarray:
    """Combine equal-market row weights with capped wrong-side severity.

    The final vector is normalized to mean one, preserving the absolute scale
    expected by the existing training code while retaining all relative ratios.
    """

    base = market_equal_weights(frame)
    severity = wrong_side_economic_severity(
        frame,
        label_column=label_column,
        up_debit_column=up_debit_column,
        down_debit_column=down_debit_column,
    )
    weighted = base * severity
    mean = float(weighted.mean())
    if not math.isfinite(mean) or mean <= 0.0:
        raise RuntimeError("loss-tail training weights have an invalid mean")
    return weighted / mean


def fit_direct_loss_tail_hgb(
    fit_frame: pl.DataFrame,
    calibration_frame: pl.DataFrame,
    *,
    core_config: CoreTrainingConfig,
    candidate_name: str = DIRECT_LOSS_TAIL_HGB_CANDIDATE,
) -> CalibratedCandidateResult:
    """Tune and fit the economically weighted 124-feature HGB candidate."""

    validate_loss_tail_feature_contract(fit_frame)
    validate_loss_tail_feature_contract(calibration_frame)
    parameter_candidates = tuple(
        histogram_parameters(candidate) for candidate in core_config.model.histogram_candidates
    )
    return _fit_tuned_candidate(
        fit_frame,
        calibration_frame,
        candidate_name=candidate_name,
        family="histogram",
        feature_names=LOSS_TAIL_FEATURES,
        target_column="label_up",
        parameter_candidates=parameter_candidates,
        random_seed=core_config.model.random_seed,
        threads_per_fit=core_config.compute.threads_per_fit,
        training_weight_kind="loss_tail",
        extra_trees_n_jobs=None,
    )


def fit_extra_trees_rare_regime(
    fit_frame: pl.DataFrame,
    calibration_frame: pl.DataFrame,
    *,
    core_config: CoreTrainingConfig,
    n_jobs: int,
    parameter_candidates: Sequence[ExtraTreesHyperparameters] = DEFAULT_EXTRA_TREES_CANDIDATES,
    candidate_name: str = EXTRA_TREES_RARE_REGIME_CANDIDATE,
) -> CalibratedCandidateResult:
    """Tune and fit the economically weighted 124-feature ExtraTrees candidate."""

    if n_jobs <= 0:
        raise ValueError("ExtraTrees n_jobs must be positive")
    validate_loss_tail_feature_contract(fit_frame)
    validate_loss_tail_feature_contract(calibration_frame)
    parameters = tuple(asdict(candidate) for candidate in parameter_candidates)
    if not parameters:
        raise ValueError("ExtraTrees requires at least one hyperparameter candidate")
    return _fit_tuned_candidate(
        fit_frame,
        calibration_frame,
        candidate_name=candidate_name,
        family="extra_trees",
        feature_names=LOSS_TAIL_FEATURES,
        target_column="label_up",
        parameter_candidates=parameters,
        random_seed=core_config.model.random_seed,
        threads_per_fit=core_config.compute.threads_per_fit,
        training_weight_kind="loss_tail",
        extra_trees_n_jobs=n_jobs,
    )


def fit_boundary_correctness_logistic(
    fit_frame: pl.DataFrame,
    calibration_frame: pl.DataFrame,
    *,
    feature_names: tuple[str, ...],
    core_config: CoreTrainingConfig,
    target_column: str = "boundary_direction_correct",
    candidate_name: str = BOUNDARY_CORRECTNESS_LOGISTIC_CANDIDATE,
) -> CalibratedCandidateResult:
    """Fit a regularized logistic model for P(locked boundary direction is correct)."""

    _validate_feature_names(feature_names)
    _validate_feature_frame(fit_frame, feature_names)
    _validate_feature_frame(calibration_frame, feature_names)
    parameters = tuple({"c": float(value)} for value in core_config.model.c_candidates)
    return _fit_tuned_candidate(
        fit_frame,
        calibration_frame,
        candidate_name=candidate_name,
        family="logistic",
        feature_names=feature_names,
        target_column=target_column,
        parameter_candidates=parameters,
        random_seed=core_config.model.random_seed,
        threads_per_fit=core_config.compute.threads_per_fit,
        training_weight_kind="market_equal",
        extra_trees_n_jobs=None,
    )


def fit_unweighted_platt_calibrator(
    model: FittedCoreModel,
    calibration_frame: pl.DataFrame,
    *,
    target_column: str,
    random_seed: int,
    threads_per_fit: int,
) -> ProbabilityCalibrator:
    """Fit row-unweighted Platt calibration on a chronologically separate frame."""

    if threads_per_fit <= 0:
        raise ValueError("threads_per_fit must be positive")
    labels = _validated_binary_target(calibration_frame, target_column)
    _require_both_classes(labels, context="Platt calibration")
    logits = model.raw_logit(calibration_frame).reshape(-1, 1)
    estimator = LogisticRegression(
        C=1_000_000,
        solver="lbfgs",
        max_iter=500,
        tol=1e-9,
        random_state=random_seed,
    )
    with threadpool_limits(limits=threads_per_fit):
        # Calibration is intentionally unweighted. Economic severity belongs to
        # the decision model fit, not to probability calibration.
        estimator.fit(logits, labels)
    calibrator = ProbabilityCalibrator(
        slope=float(estimator.coef_[0, 0]),
        intercept=float(estimator.intercept_[0]),
        converged=bool(estimator.n_iter_[0] < estimator.max_iter),
        iterations=int(estimator.n_iter_[0]),
    )
    if not calibrator.converged:
        raise RuntimeError("unweighted Platt calibration did not converge")
    return calibrator


def score_calibrated_candidate(
    result: CalibratedCandidateResult,
    frame: pl.DataFrame,
    *,
    raw_probability_column: str = "raw_probability",
    probability_column: str = "probability",
) -> pl.DataFrame:
    """Attach deterministic raw and calibrated probabilities to a row frame."""

    if not raw_probability_column or not probability_column:
        raise ValueError("probability column names must be non-empty")
    if raw_probability_column == probability_column:
        raise ValueError("raw and calibrated probability columns must differ")
    raw = result.raw_probability(frame)
    calibrated = result.probability(frame)
    _validate_probability(raw, frame.height, context="raw candidate probability")
    _validate_probability(calibrated, frame.height, context="calibrated probability")
    return frame.with_columns(
        pl.Series(raw_probability_column, raw),
        pl.Series(probability_column, calibrated),
    )


def _fit_tuned_candidate(
    fit_frame: pl.DataFrame,
    calibration_frame: pl.DataFrame,
    *,
    candidate_name: str,
    family: str,
    feature_names: tuple[str, ...],
    target_column: str,
    parameter_candidates: tuple[dict[str, Any], ...],
    random_seed: int,
    threads_per_fit: int,
    training_weight_kind: str,
    extra_trees_n_jobs: int | None,
) -> CalibratedCandidateResult:
    if not candidate_name.strip():
        raise ValueError("candidate name must be non-empty")
    if threads_per_fit <= 0:
        raise ValueError("threads_per_fit must be positive")
    if not parameter_candidates:
        raise ValueError(f"{candidate_name} requires at least one hyperparameter candidate")
    _validate_feature_frame(fit_frame, feature_names)
    _validate_feature_frame(calibration_frame, feature_names)
    fit_labels = _validated_binary_target(fit_frame, target_column)
    calibration_labels = _validated_binary_target(calibration_frame, target_column)
    _require_both_classes(fit_labels, context=f"{candidate_name} fitting")
    _require_both_classes(calibration_labels, context=f"{candidate_name} calibration")
    train, validation = chronological_inner_split(fit_frame, validation_fraction=0.20)
    _require_both_classes(
        _validated_binary_target(train, target_column),
        context=f"{candidate_name} inner training",
    )

    history: list[dict[str, Any]] = []
    best_parameters: dict[str, Any] | None = None
    best_loss = float("inf")
    for parameters in parameter_candidates:
        started = time.perf_counter()
        model = _fit_candidate_model(
            train,
            candidate_name=candidate_name,
            family=family,
            feature_names=feature_names,
            target_column=target_column,
            parameters=parameters,
            random_seed=random_seed,
            threads_per_fit=threads_per_fit,
            training_weight_kind=training_weight_kind,
            extra_trees_n_jobs=extra_trees_n_jobs,
        )
        probability = model.raw_probability(validation)
        validation_weights = _training_weights(
            validation,
            kind=training_weight_kind,
            target_column=target_column,
        )
        validation_loss = float(
            log_loss(
                _validated_binary_target(validation, target_column),
                probability,
                sample_weight=validation_weights,
                labels=[0, 1],
            )
        )
        record = {
            "hyperparameters": dict(parameters),
            "validation_log_loss": validation_loss,
            "fit_seconds": time.perf_counter() - started,
        }
        history.append(record)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_parameters = dict(parameters)
    if best_parameters is None:
        raise RuntimeError(f"{candidate_name} did not produce a fitted candidate")

    final_model = _fit_candidate_model(
        fit_frame,
        candidate_name=candidate_name,
        family=family,
        feature_names=feature_names,
        target_column=target_column,
        parameters=best_parameters,
        random_seed=random_seed,
        threads_per_fit=threads_per_fit,
        training_weight_kind=training_weight_kind,
        extra_trees_n_jobs=extra_trees_n_jobs,
    )
    calibrator = fit_unweighted_platt_calibrator(
        final_model,
        calibration_frame,
        target_column=target_column,
        random_seed=random_seed,
        threads_per_fit=threads_per_fit,
    )
    return CalibratedCandidateResult(
        model=final_model,
        calibrator=calibrator,
        target_column=target_column,
        tuning={
            "selected_hyperparameters": best_parameters,
            "validation_log_loss": best_loss,
            "candidates": history,
            "training_weighting": training_weight_kind,
            "training_weight_formula": (
                "equal_total_per_market_x_fee_inclusive_wrong_side_debit_odds_capped_1_10"
                if training_weight_kind == "loss_tail"
                else "equal_total_per_market"
            ),
            "calibration_weighting": "unweighted_rows",
        },
    )


def _fit_candidate_model(
    frame: pl.DataFrame,
    *,
    candidate_name: str,
    family: str,
    feature_names: tuple[str, ...],
    target_column: str,
    parameters: dict[str, Any],
    random_seed: int,
    threads_per_fit: int,
    training_weight_kind: str,
    extra_trees_n_jobs: int | None,
) -> FittedCoreModel:
    labels = _validated_binary_target(frame, target_column)
    weights = _training_weights(frame, kind=training_weight_kind, target_column=target_column)
    matrix = feature_matrix(frame, feature_names)
    medians = finite_medians(matrix)
    filled = np.where(np.isfinite(matrix), matrix, medians)
    if family == "logistic":
        means = np.average(filled, axis=0, weights=weights)
        variance = np.average((filled - means) ** 2, axis=0, weights=weights)
        scales = np.sqrt(np.maximum(variance, 0.0))
        scales = np.where(scales > 1e-12, scales, 1.0)
        transformed = (filled - means) / scales
        estimator: Any = LogisticRegression(
            C=float(parameters["c"]),
            solver="lbfgs",
            max_iter=2_000,
            tol=1e-7,
            random_state=random_seed,
        )
    elif family == "histogram":
        means = None
        scales = None
        transformed = filled
        estimator = HistGradientBoostingClassifier(
            learning_rate=float(parameters["learning_rate"]),
            max_iter=int(parameters["max_iter"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            l2_regularization=float(parameters["l2_regularization"]),
            early_stopping=False,
            random_state=random_seed,
        )
    elif family == "extra_trees":
        if extra_trees_n_jobs is None or extra_trees_n_jobs <= 0:
            raise ValueError("ExtraTrees requires a positive n_jobs")
        means = None
        scales = None
        transformed = filled
        estimator = ExtraTreesClassifier(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=(
                int(parameters["max_depth"]) if parameters["max_depth"] is not None else None
            ),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=parameters["max_features"],
            criterion="log_loss",
            bootstrap=False,
            n_jobs=extra_trees_n_jobs,
            random_state=random_seed,
        )
    else:
        raise ValueError(f"unsupported loss-tail candidate family: {family}")

    with threadpool_limits(limits=threads_per_fit):
        estimator.fit(transformed, labels, sample_weight=weights)
    if (
        isinstance(estimator, LogisticRegression)
        and int(estimator.n_iter_[0]) >= estimator.max_iter
    ):
        raise RuntimeError(f"{candidate_name} logistic optimizer did not converge")
    return FittedCoreModel(
        candidate_name=candidate_name,
        family=family,
        feature_names=feature_names,
        hyperparameters=dict(parameters),
        imputation_medians=medians,
        standardization_means=means,
        standardization_scales=scales,
        estimator=estimator,
        row_weight_policy=MARKET_EQUAL_ROW_WEIGHT_POLICY,
        row_weight_schedule=MARKET_EQUAL_ROW_WEIGHT_SCHEDULE,
        recency_half_life_days=None,
    )


def _training_weights(
    frame: pl.DataFrame,
    *,
    kind: str,
    target_column: str,
) -> np.ndarray:
    if kind == "loss_tail":
        if target_column != "label_up":
            raise ValueError("loss-tail severity weights require the directional label_up target")
        return loss_tail_training_weights(frame)
    if kind == "market_equal":
        return market_equal_weights(frame)
    raise ValueError(f"unsupported training weight kind: {kind}")


def _validated_binary_target(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"training frame is missing target column: {column}")
    values = frame[column].cast(pl.Float64, strict=False).to_numpy()
    if (
        values.ndim != 1
        or len(values) != frame.height
        or not np.isfinite(values).all()
        or not np.isin(values, (0.0, 1.0)).all()
    ):
        raise ValueError(f"{column} must contain only non-null binary values")
    return values.astype(np.int8)


def _validated_debit(frame: pl.DataFrame, column: str) -> np.ndarray:
    values = frame[column].cast(pl.Float64, strict=False).to_numpy()
    if (
        values.ndim != 1
        or len(values) != frame.height
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError(f"{column} must contain finite values in (0, 1]")
    return values


def _require_both_classes(labels: np.ndarray, *, context: str) -> None:
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError(f"{context} requires both target classes")


def _validate_feature_frame(frame: pl.DataFrame, feature_names: tuple[str, ...]) -> None:
    _validate_feature_names(feature_names)
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError("training frame is missing features: " + ", ".join(missing))


def _validate_feature_names(feature_names: tuple[str, ...]) -> None:
    if not feature_names:
        raise ValueError("candidate feature set must be non-empty")
    duplicates = _duplicates(feature_names)
    if duplicates:
        raise ValueError("candidate feature set contains duplicates: " + ", ".join(duplicates))
    if any(not feature.strip() for feature in feature_names):
        raise ValueError("candidate feature names must be non-empty")


def _duplicates(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return sorted(duplicate)


def _validate_probability(values: np.ndarray, expected_length: int, *, context: str) -> None:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 1 or len(probability) != expected_length:
        raise RuntimeError(f"{context} has the wrong shape")
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise RuntimeError(f"{context} must stay inside [0, 1]")


# Fail immediately if one of the upstream feature blocks drifts.  This benchmark
# is intentionally tied to an exact, reviewable schema rather than silently
# changing its training inputs.
validate_loss_tail_feature_contract()
