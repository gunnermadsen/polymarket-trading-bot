"""Conservative NO-side calibration for the frozen D4 admission predictions.

This module is intentionally downstream of the sealed D4 cross-day run.  It does
not fit D4, change its feature contract, widen its price policy, or create a
runtime model.  The only trainable parameters are non-positive NO-side logit
intercepts; YES probabilities and every calibration slope remain identity.  The
caller must supply the outcome-blind side orientation frozen from D4-base before
calibration; calibrated probabilities never recursively reorient that side.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

from .asymmetric_book_admission import (
    BOOK_ADMISSION_KEY_COLUMNS,
    INCUMBENT_FEATURE_SCHEMA_SHA256,
    INCUMBENT_MODEL_KEY,
    INCUMBENT_MODEL_SHA256,
    INCUMBENT_PROCESS_ID,
    LABEL_COLUMN,
    MARKET_EQUAL,
    PROBABILITY_COLUMN,
    SELECTED_SIDE_COLUMN,
    BookAdmissionFold,
    BookAdmissionPolicy,
    EvidenceWindow,
    FrozenIncumbent,
    book_admission_key_sha256,
    book_admission_weights,
    load_book_admission_config,
)

D4_SIDE_CALIBRATION_PROFILE = "btc_asymmetric_core_oracle_d4_side_calibration"
D4_SIDE_CALIBRATION_SCHEMA_VERSION = "btc-asymmetric-d4-side-calibration-v1"
PARENT_BOOK_ADMISSION_CONFIG = "btc-5m-asymmetric-core-oracle-book-admission-20260414-20260802.toml"
PARENT_BOOK_ADMISSION_CONFIG_SHA256 = (
    "c43758e5dc77d01cbb148cefa4da1bc16222b32e37ce6de473c363ca50dea486"
)
PARENT_RUN_ID = "20260810T200017635552Z"
CONSUMED_EVIDENCE_END = datetime(2026, 8, 2, tzinfo=UTC)
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
POOLED_NO_CELL = "NO_1_56"


@dataclass(frozen=True)
class FrozenD4Contract:
    candidate_name: str
    feature_set: str
    estimator: str
    weighting: str
    l2_regularization: float
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    residual_logit_cap: float
    gamma_minimum: float
    gamma_maximum: float
    preprocessing: str
    random_seed: int


@dataclass(frozen=True)
class ParentD4Run:
    run_id: str
    root_training_contract_sha256: str
    fold_model_manifest_sha256: str
    prediction_manifest_sha256: str
    probability_selection_sha256: str
    probability_selection_seal_sha256: str
    benchmark_bundle_seal_sha256: str
    d4_oof_predictions_sha256: str


@dataclass(frozen=True)
class NoTimeBand:
    name: str
    start_second: int
    end_second_exclusive: int

    def contains(self, seconds: np.ndarray) -> np.ndarray:
        return (seconds >= self.start_second) & (seconds < self.end_second_exclusive)


@dataclass(frozen=True)
class D4SideCalibrationArm:
    name: str
    strategy: Literal["identity", "pooled_no_intercept", "time_local_no_intercepts"]
    selection_eligible: bool


@dataclass(frozen=True)
class D4SideCalibrationContract:
    probability_clip: float
    fixed_slope: float
    identity_l2: float
    minimum_no_logit_offset: float
    maximum_no_logit_offset: float
    weighting: str
    optimizer_max_iterations: int
    optimizer_absolute_tolerance: float
    time_bands: tuple[NoTimeBand, ...]
    arms: tuple[D4SideCalibrationArm, ...]


@dataclass(frozen=True)
class D4SideCalibrationSupportGates:
    minimum_calibration_no_markets: int
    minimum_calibration_utc_days: int
    minimum_no_winning_markets: int
    minimum_no_losing_markets: int
    require_both_outcomes_per_time_cell_for_local: bool
    minimum_validation_strict_markets: int
    minimum_validation_no_opportunities: int
    minimum_validation_days_with_no_opportunities: int
    required_validation_days: int
    require_exact_candidate_control_keys: bool
    require_no_duplicate_keys: bool
    require_no_future_data: bool


@dataclass(frozen=True)
class D4SideCalibrationProbabilityGates:
    maximum_paired_degradation_upper_95: float
    maximum_selected_opportunity_bias: float
    maximum_no_selected_bias: float
    maximum_yes_selected_bias: float
    minimum_no_selected_bias_reduction: float
    minimum_noninferior_days: int
    required_comparison_days: int
    require_brier_noninferiority_to_d4: bool
    require_log_loss_noninferiority_to_d4: bool
    require_brier_noninferiority_to_incumbent: bool
    require_log_loss_noninferiority_to_incumbent: bool
    require_preserved_yes_quality: bool
    require_no_harmful_cell_overconfidence: bool
    selection_uses_economics: bool


@dataclass(frozen=True)
class D4SideCalibrationEconomicGates:
    minimum_stressed_expectancy_per_trade: float
    minimum_profit_factor: float
    minimum_yes_stressed_pnl: float
    minimum_no_stressed_pnl: float
    minimum_no_stressed_pnl_improvement: float
    maximum_yes_stressed_pnl_regression_fraction: float
    maximum_drawdown_regression_fraction: float
    require_accuracy_noninferior_to_d4: bool
    require_positive_paired_net_profit_per_eligible_market: bool
    report_projected_pnl_for_every_arm: bool


@dataclass(frozen=True)
class D4SideCalibrationPaths:
    parent_book_admission_config: Path
    parent_run: Path
    runs: Path


@dataclass(frozen=True)
class D4SideCalibrationConfig:
    source_path: Path
    package_root: Path
    profile: str
    paper_only: bool
    live_capital_allowed: bool
    runtime_deployable: bool
    batch_forward_eligible: bool
    qualification_eligible: bool
    support_qualification_allowed: bool
    projected_pnl_diagnostic: bool
    process_change_allowed: bool
    evidence_scope: str
    incumbent: FrozenIncumbent
    parent_d4: FrozenD4Contract
    parent_run: ParentD4Run
    development: EvidenceWindow
    calibration_days: int
    excluded_utc_days: tuple[date, ...]
    oof_utc_days: tuple[date, ...]
    folds: tuple[BookAdmissionFold, ...]
    consumed_validation_role: str
    policy: BookAdmissionPolicy
    calibration: D4SideCalibrationContract
    support_gates: D4SideCalibrationSupportGates
    probability_gates: D4SideCalibrationProbabilityGates
    economic_gates: D4SideCalibrationEconomicGates
    paths: D4SideCalibrationPaths

    def arm(self, name: str) -> D4SideCalibrationArm:
        matches = [arm for arm in self.calibration.arms if arm.name == name]
        if len(matches) != 1:
            raise ValueError(f"unknown D4 side-calibration arm: {name}")
        return matches[0]


@dataclass(frozen=True)
class D4NoCellSupport:
    name: str
    rows: int
    markets: int
    utc_days: int
    winning_markets: int
    losing_markets: int


@dataclass(frozen=True)
class D4SideCalibrationSupport:
    rows: int
    markets: int
    utc_days: int
    no_rows: int
    no_markets: int
    no_utc_days: int
    no_winning_markets: int
    no_losing_markets: int
    time_cells: tuple[D4NoCellSupport, ...]
    key_sha256: str


@dataclass(frozen=True)
class D4SideCalibrationFitEvidence:
    calibration_window: EvidenceWindow
    calibration_key_sha256: str
    calibration_weight_sha256: str
    support: D4SideCalibrationSupport
    support_passed: bool
    support_failures: tuple[str, ...]
    identity_objective: float
    fitted_objective: float


@dataclass(frozen=True)
class FittedD4SideCalibrator:
    schema_version: str
    arm: D4SideCalibrationArm
    parent_run_id: str
    parent_d4_oof_predictions_sha256: str
    fixed_slope: float
    probability_clip: float
    no_logit_offsets: tuple[tuple[str, float], ...]
    time_bands: tuple[NoTimeBand, ...]
    evidence: D4SideCalibrationFitEvidence

    def probability_yes(self, frame: pl.DataFrame) -> np.ndarray:
        _validate_probability_frame(frame, self.probability_clip, require_label=False)
        parent_yes = frame[PROBABILITY_COLUMN].cast(pl.Float64).to_numpy().copy()
        sides = frame[SELECTED_SIDE_COLUMN].cast(pl.String).to_numpy()
        no_mask = sides == "NO"
        if not no_mask.any() or self.arm.strategy == "identity":
            return parent_yes
        seconds = frame["seconds_elapsed"].cast(pl.Int64).to_numpy()
        calibrated_yes = parent_yes.copy()
        parent_no = 1.0 - parent_yes
        offset_map = dict(self.no_logit_offsets)
        if self.arm.strategy == "pooled_no_intercept":
            offsets = np.full(frame.height, offset_map[POOLED_NO_CELL], dtype=np.float64)
        else:
            offsets = np.zeros(frame.height, dtype=np.float64)
            assigned = np.zeros(frame.height, dtype=bool)
            for band in self.time_bands:
                selected = no_mask & band.contains(seconds)
                offsets[selected] = offset_map[band.name]
                assigned |= selected
            if not np.array_equal(assigned, no_mask):
                raise RuntimeError("NO rows escaped the frozen calibration time bands")
        no_eta = _logit(parent_no, self.probability_clip) + offsets
        fitted_no = np.clip(_sigmoid(no_eta), self.probability_clip, 1.0 - self.probability_clip)
        if np.any(fitted_no[no_mask] > parent_no[no_mask] + 1e-12):
            raise RuntimeError("D4 NO calibration became more aggressive")
        calibrated_yes[no_mask] = 1.0 - fitted_no[no_mask]
        if not np.array_equal(calibrated_yes[~no_mask], parent_yes[~no_mask]):
            raise RuntimeError("D4 YES probabilities changed")
        return calibrated_yes

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "arm": asdict(self.arm),
            "parent_run_id": self.parent_run_id,
            "parent_d4_oof_predictions_sha256": (self.parent_d4_oof_predictions_sha256),
            "fixed_slope": self.fixed_slope,
            "probability_clip": self.probability_clip,
            "no_logit_offsets": self.no_logit_offsets,
            "time_bands": [asdict(value) for value in self.time_bands],
        }

    @property
    def semantic_sha256(self) -> str:
        return _canonical_sha256(self.semantic_payload())


@dataclass(frozen=True)
class D4SideCalibrationScore:
    arm_name: str
    frame: pl.DataFrame
    key_sha256: str
    probability_sha256: str
    calibrator_semantic_sha256: str


def load_d4_side_calibration_config(path: Path) -> D4SideCalibrationConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    benchmark = raw["benchmark"]
    incumbent = raw["incumbent"]
    chronology = raw["chronology"]
    calibration = raw["calibration"]
    paths = raw["paths"]
    oof_days = tuple(_parse_date(value) for value in chronology["oof_utc_days"])
    development = EvidenceWindow(
        _parse_utc(chronology["development_start"]),
        _parse_utc(chronology["development_end"]),
    )
    calibration_days = int(chronology["calibration_days"])
    folds = tuple(_fold_for_day(day, development.start, calibration_days) for day in oof_days)
    config = D4SideCalibrationConfig(
        source_path=source_path,
        package_root=package_root,
        profile=str(benchmark["profile"]),
        paper_only=bool(benchmark["paper_only"]),
        live_capital_allowed=bool(benchmark["live_capital_allowed"]),
        runtime_deployable=bool(benchmark["runtime_deployable"]),
        batch_forward_eligible=bool(benchmark["batch_forward_eligible"]),
        qualification_eligible=bool(benchmark["qualification_eligible"]),
        support_qualification_allowed=bool(benchmark["support_qualification_allowed"]),
        projected_pnl_diagnostic=bool(benchmark["projected_pnl_diagnostic"]),
        process_change_allowed=bool(benchmark["process_change_allowed"]),
        evidence_scope=str(benchmark["evidence_scope"]),
        incumbent=FrozenIncumbent(**incumbent),
        parent_d4=FrozenD4Contract(**raw["parent_d4"]),
        parent_run=ParentD4Run(**raw["parent_run"]),
        development=development,
        calibration_days=calibration_days,
        excluded_utc_days=tuple(_parse_date(value) for value in chronology["excluded_utc_days"]),
        oof_utc_days=oof_days,
        folds=folds,
        consumed_validation_role=str(chronology["consumed_validation_role"]),
        policy=BookAdmissionPolicy(**raw["policy"]),
        calibration=D4SideCalibrationContract(
            probability_clip=float(calibration["probability_clip"]),
            fixed_slope=float(calibration["fixed_slope"]),
            identity_l2=float(calibration["identity_l2"]),
            minimum_no_logit_offset=float(calibration["minimum_no_logit_offset"]),
            maximum_no_logit_offset=float(calibration["maximum_no_logit_offset"]),
            weighting=str(calibration["weighting"]),
            optimizer_max_iterations=int(calibration["optimizer_max_iterations"]),
            optimizer_absolute_tolerance=float(calibration["optimizer_absolute_tolerance"]),
            time_bands=tuple(NoTimeBand(**value) for value in calibration["time_bands"]),
            arms=tuple(D4SideCalibrationArm(**value) for value in calibration["arms"]),
        ),
        support_gates=D4SideCalibrationSupportGates(**raw["support_gates"]),
        probability_gates=D4SideCalibrationProbabilityGates(**raw["probability_gates"]),
        economic_gates=D4SideCalibrationEconomicGates(**raw["economic_gates"]),
        paths=D4SideCalibrationPaths(
            parent_book_admission_config=(
                package_root / str(paths["parent_book_admission_config"])
            ),
            parent_run=package_root / str(paths["parent_run"]),
            runs=package_root / str(paths["runs"]),
        ),
    )
    validate_d4_side_calibration_config(config)
    return config


def validate_d4_side_calibration_config(config: D4SideCalibrationConfig) -> None:
    if (
        config.profile != D4_SIDE_CALIBRATION_PROFILE
        or not config.paper_only
        or config.live_capital_allowed
        or config.runtime_deployable
        or config.batch_forward_eligible
        or config.qualification_eligible
        or config.support_qualification_allowed
        or not config.projected_pnl_diagnostic
        or config.process_change_allowed
        or config.evidence_scope != "consumed_architecture_feedback"
        or config.consumed_validation_role != "diagnostic_only_not_forward_proof"
    ):
        raise ValueError("D4 side-calibration diagnostic boundary changed")
    expected_incumbent = FrozenIncumbent(
        process_id=INCUMBENT_PROCESS_ID,
        model_key=INCUMBENT_MODEL_KEY,
        model_sha256=INCUMBENT_MODEL_SHA256,
        feature_schema_sha256=INCUMBENT_FEATURE_SCHEMA_SHA256,
        feature_count=75,
    )
    if config.incumbent != expected_incumbent:
        raise ValueError("D4 side-calibration incumbent changed")
    _validate_parent_contract(config)
    expected_development = EvidenceWindow(datetime(2026, 4, 14, tzinfo=UTC), CONSUMED_EVIDENCE_END)
    if (
        config.development != expected_development
        or config.calibration_days != 14
        or config.excluded_utc_days != EXCLUDED_DAYS
        or config.oof_utc_days != OOF_DAYS
        or len(config.folds) != 10
    ):
        raise ValueError("D4 side-calibration consumed chronology changed")
    for day, fold in zip(OOF_DAYS, config.folds, strict=True):
        if fold != _fold_for_day(day, expected_development.start, 14):
            raise ValueError(f"D4 side-calibration fold chronology changed: {fold.name}")
    parent_config = load_book_admission_config(config.paths.parent_book_admission_config)
    if config.policy != parent_config.policy:
        raise ValueError("D4 side-calibration policy differs from frozen D4")
    _validate_calibration_contract(config.calibration)
    _validate_gate_contract(config)


def validate_d4_parent_run_artifacts(
    config: D4SideCalibrationConfig,
) -> dict[str, str]:
    """Fail closed unless every pinned parent-run artifact is present and exact."""

    expected = {
        "root-training-contract.json": config.parent_run.root_training_contract_sha256,
        "fold-model-manifest.json": config.parent_run.fold_model_manifest_sha256,
        "prediction-manifest.json": config.parent_run.prediction_manifest_sha256,
        "probability-selection.json": config.parent_run.probability_selection_sha256,
        "probability-selection-seal.json": (config.parent_run.probability_selection_seal_sha256),
        "benchmark-bundle-seal.json": config.parent_run.benchmark_bundle_seal_sha256,
        "candidates/D4/oof-predictions.parquet": (config.parent_run.d4_oof_predictions_sha256),
    }
    observed: dict[str, str] = {}
    for relative, expected_sha256 in expected.items():
        path = config.paths.parent_run / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned D4 parent artifact is missing: {path}")
        observed_sha256 = _file_sha256(path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(f"pinned D4 parent artifact changed: {relative}")
        observed[relative] = observed_sha256
    return observed


def d4_side_calibration_support(
    frame: pl.DataFrame,
    config: D4SideCalibrationConfig,
    arm_name: str,
) -> D4SideCalibrationSupport:
    config.arm(arm_name)
    _validate_probability_frame(frame, config.calibration.probability_clip, require_label=True)
    no_frame = frame.filter(pl.col(SELECTED_SIDE_COLUMN) == "NO")
    cells = tuple(_cell_support(no_frame, band) for band in config.calibration.time_bands)
    no_wins, no_losses = _market_outcome_counts(no_frame)
    return D4SideCalibrationSupport(
        rows=frame.height,
        markets=frame["market_id"].n_unique(),
        utc_days=frame["window_start"].dt.date().n_unique(),
        no_rows=no_frame.height,
        no_markets=no_frame["market_id"].n_unique(),
        no_utc_days=no_frame["window_start"].dt.date().n_unique(),
        no_winning_markets=no_wins,
        no_losing_markets=no_losses,
        time_cells=cells,
        key_sha256=book_admission_key_sha256(frame),
    )


def fit_d4_side_calibrator(
    config: D4SideCalibrationConfig,
    arm_name: str,
    calibration_frame: pl.DataFrame,
    *,
    fold_name: str | None = None,
) -> FittedD4SideCalibrator:
    """Fit only bounded, non-positive NO logit offsets on one causal window."""

    arm = config.arm(arm_name)
    calibration_window = _matching_calibration_window(
        calibration_frame, config, fold_name=fold_name
    )
    support = d4_side_calibration_support(calibration_frame, config, arm_name)
    support_failures = _calibration_support_failures(support, config, arm)
    if support_failures and config.qualification_eligible:
        raise RuntimeError("D4 side-calibration support failed: " + ", ".join(support_failures))
    no_frame = calibration_frame.filter(pl.col(SELECTED_SIDE_COLUMN) == "NO")
    weights = book_admission_weights(no_frame, config.calibration.weighting)
    probability_no = 1.0 - no_frame[PROBABILITY_COLUMN].cast(pl.Float64).to_numpy()
    labels_no = 1.0 - no_frame[LABEL_COLUMN].cast(pl.Float64).to_numpy()
    seconds = no_frame["seconds_elapsed"].cast(pl.Int64).to_numpy()

    if arm.strategy == "identity":
        offsets: tuple[tuple[str, float], ...] = ()
    elif arm.strategy == "pooled_no_intercept":
        offsets = (
            (
                POOLED_NO_CELL,
                _fit_conservative_offset(probability_no, labels_no, weights, config.calibration),
            ),
        )
    else:
        fitted: list[tuple[str, float]] = []
        for band in config.calibration.time_bands:
            selected = band.contains(seconds)
            fitted.append(
                (
                    band.name,
                    _fit_conservative_offset(
                        probability_no[selected],
                        labels_no[selected],
                        weights[selected],
                        config.calibration,
                    ),
                )
            )
        offsets = tuple(fitted)

    identity_objective = _calibration_objective(
        probability_no, labels_no, weights, 0.0, config.calibration
    )
    fitted_objective = _joint_objective(
        probability_no, labels_no, weights, seconds, offsets, arm, config.calibration
    )
    return FittedD4SideCalibrator(
        schema_version=D4_SIDE_CALIBRATION_SCHEMA_VERSION,
        arm=arm,
        parent_run_id=config.parent_run.run_id,
        parent_d4_oof_predictions_sha256=(config.parent_run.d4_oof_predictions_sha256),
        fixed_slope=config.calibration.fixed_slope,
        probability_clip=config.calibration.probability_clip,
        no_logit_offsets=offsets,
        time_bands=config.calibration.time_bands,
        evidence=D4SideCalibrationFitEvidence(
            calibration_window=calibration_window,
            calibration_key_sha256=support.key_sha256,
            calibration_weight_sha256=_array_sha256(weights),
            support=support,
            support_passed=not support_failures,
            support_failures=support_failures,
            identity_objective=identity_objective,
            fitted_objective=fitted_objective,
        ),
    )


def score_d4_side_calibrator(
    calibrator: FittedD4SideCalibrator,
    frame: pl.DataFrame,
) -> D4SideCalibrationScore:
    probabilities = calibrator.probability_yes(frame)
    scored = frame.select(*BOOK_ADMISSION_KEY_COLUMNS).with_columns(
        pl.lit(calibrator.arm.name).alias("candidate_name"),
        pl.Series(PROBABILITY_COLUMN, probabilities, dtype=pl.Float64),
    )
    return D4SideCalibrationScore(
        arm_name=calibrator.arm.name,
        frame=scored,
        key_sha256=book_admission_key_sha256(scored),
        probability_sha256=d4_side_calibration_probability_sha256(scored),
        calibrator_semantic_sha256=calibrator.semantic_sha256,
    )


def d4_side_calibration_probability_sha256(frame: pl.DataFrame) -> str:
    _require_columns(
        frame,
        (*BOOK_ADMISSION_KEY_COLUMNS, PROBABILITY_COLUMN),
        "D4 side-calibration probability evidence",
    )
    ordered = frame.sort(*BOOK_ADMISSION_KEY_COLUMNS)
    probabilities = ordered[PROBABILITY_COLUMN].cast(pl.Float64).to_numpy()
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities <= 0.0) | (probabilities >= 1.0)
    ):
        raise ValueError("D4 side-calibration probabilities are invalid")
    digest = hashlib.sha256(b"btc-asymmetric-d4-side-calibration-probability-v1\n")
    digest.update(book_admission_key_sha256(ordered).encode())
    digest.update(probabilities.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _validate_parent_contract(config: D4SideCalibrationConfig) -> None:
    expected_run = ParentD4Run(
        run_id=PARENT_RUN_ID,
        root_training_contract_sha256=(
            "82879fee81ff3ae91291088f5a9d20c06f171be3bafd3ed466d36b0479bbcf01"
        ),
        fold_model_manifest_sha256=(
            "87c9633464b6e387056b814e24125f6a70c1f76b8aa483d036be1ad8ca5e7b0a"
        ),
        prediction_manifest_sha256=(
            "7248ea84cd4ec1a91f32ff400deb24f432df2de46638499ca013d2c85c6f4802"
        ),
        probability_selection_sha256=(
            "5b1bbedbce4dc51818b96723267a1a700eaadf9de2276356763e78aac36be9ae"
        ),
        probability_selection_seal_sha256=(
            "32a43c883d5d7dae0a01d4f1f084cb81d57ef0f378411b21599c9030c04fc1c3"
        ),
        benchmark_bundle_seal_sha256=(
            "0b98ab3bd4ca3d04982a89104a04ce4fea15ece2a5920d0acd2284ee348e238d"
        ),
        d4_oof_predictions_sha256=(
            "0faea82a211b968d5b182da56ab07a9fcfe0d0d909079d1f409946d7b7d0e6d3"
        ),
    )
    if config.parent_run != expected_run:
        raise ValueError("pinned D4 parent-run identity changed")
    if (
        config.paths.parent_book_admission_config.name != PARENT_BOOK_ADMISSION_CONFIG
        or not config.paths.parent_book_admission_config.is_file()
        or _file_sha256(config.paths.parent_book_admission_config)
        != PARENT_BOOK_ADMISSION_CONFIG_SHA256
        or config.paths.parent_run.name != PARENT_RUN_ID
    ):
        raise ValueError("pinned D4 parent path or config changed")
    parent = load_book_admission_config(config.paths.parent_book_admission_config)
    candidate = parent.candidate("D4")
    expected_d4 = FrozenD4Contract(
        candidate_name=candidate.name,
        feature_set=candidate.feature_set,
        estimator=candidate.estimator,
        weighting=candidate.weighting,
        l2_regularization=candidate.l2_regularization,
        learning_rate=float(candidate.learning_rate),
        max_iter=int(candidate.max_iter),
        max_leaf_nodes=int(candidate.max_leaf_nodes),
        min_samples_leaf=int(candidate.min_samples_leaf),
        residual_logit_cap=parent.model.residual_logit_cap,
        gamma_minimum=parent.model.gamma_minimum,
        gamma_maximum=parent.model.gamma_maximum,
        preprocessing=parent.model.preprocessing,
        random_seed=parent.model.random_seed,
    )
    if config.parent_d4 != expected_d4:
        raise ValueError("frozen D4 estimator contract changed")


def _validate_calibration_contract(contract: D4SideCalibrationContract) -> None:
    expected_bands = (
        NoTimeBand("NO_1_15", 1, 15),
        NoTimeBand("NO_15_30", 15, 30),
        NoTimeBand("NO_30_45", 30, 45),
        NoTimeBand("NO_45_56", 45, 56),
    )
    expected_arms = (
        D4SideCalibrationArm("D4-base", "identity", False),
        D4SideCalibrationArm("N1", "pooled_no_intercept", True),
        D4SideCalibrationArm("N2", "time_local_no_intercepts", True),
    )
    if (
        contract.probability_clip != 1e-6
        or contract.fixed_slope != 1.0
        or contract.identity_l2 != 1.0
        or contract.minimum_no_logit_offset != -0.5
        or contract.maximum_no_logit_offset != 0.0
        or contract.weighting != MARKET_EQUAL
        or contract.optimizer_max_iterations != 500
        or contract.optimizer_absolute_tolerance > 1e-12
        or contract.time_bands != expected_bands
        or contract.arms != expected_arms
    ):
        raise ValueError("D4 side-calibration model contract changed")


def _validate_gate_contract(config: D4SideCalibrationConfig) -> None:
    support = config.support_gates
    probability = config.probability_gates
    economic = config.economic_gates
    if (
        support.minimum_calibration_no_markets < 100
        or support.minimum_calibration_utc_days < 14
        or support.minimum_no_winning_markets < 25
        or support.minimum_no_losing_markets < 25
        or not support.require_both_outcomes_per_time_cell_for_local
        or support.minimum_validation_strict_markets < 2000
        or support.minimum_validation_no_opportunities < 200
        or support.minimum_validation_days_with_no_opportunities < 8
        or support.required_validation_days != 10
        or not support.require_exact_candidate_control_keys
        or not support.require_no_duplicate_keys
        or not support.require_no_future_data
    ):
        raise ValueError("D4 side-calibration support gates weakened")
    if (
        probability.maximum_paired_degradation_upper_95 > 0.005
        or probability.maximum_selected_opportunity_bias > 0.03
        or probability.maximum_no_selected_bias > 0.05
        or probability.maximum_yes_selected_bias > 0.03
        or probability.minimum_no_selected_bias_reduction < 0.01
        or probability.minimum_noninferior_days < 8
        or probability.required_comparison_days != 10
        or not all(
            (
                probability.require_brier_noninferiority_to_d4,
                probability.require_log_loss_noninferiority_to_d4,
                probability.require_brier_noninferiority_to_incumbent,
                probability.require_log_loss_noninferiority_to_incumbent,
                probability.require_preserved_yes_quality,
                probability.require_no_harmful_cell_overconfidence,
            )
        )
        or probability.selection_uses_economics
    ):
        raise ValueError("D4 side-calibration probability gates weakened")
    if (
        economic.minimum_stressed_expectancy_per_trade < 0.0
        or economic.minimum_profit_factor < 1.05
        or economic.minimum_yes_stressed_pnl < 0.0
        or economic.minimum_no_stressed_pnl < 0.0
        or economic.minimum_no_stressed_pnl_improvement < 0.0
        or economic.maximum_yes_stressed_pnl_regression_fraction > 0.10
        or economic.maximum_drawdown_regression_fraction > 0.10
        or not economic.require_accuracy_noninferior_to_d4
        or not economic.require_positive_paired_net_profit_per_eligible_market
        or not economic.report_projected_pnl_for_every_arm
    ):
        raise ValueError("D4 side-calibration economic gates weakened")


def _validate_probability_frame(
    frame: pl.DataFrame, probability_clip: float, *, require_label: bool
) -> None:
    if frame.is_empty():
        raise ValueError("D4 side-calibration frame is empty")
    required = {
        *BOOK_ADMISSION_KEY_COLUMNS,
        PROBABILITY_COLUMN,
        SELECTED_SIDE_COLUMN,
    }
    if require_label:
        required.add(LABEL_COLUMN)
    _require_columns(frame, tuple(sorted(required)), "D4 side-calibration frame")
    if frame.select(*BOOK_ADMISSION_KEY_COLUMNS).is_duplicated().any():
        raise ValueError("D4 side-calibration frame contains duplicate decision keys")
    for timestamp in ("window_start", "observed_at"):
        dtype = frame.schema[timestamp]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
            raise TypeError(f"{timestamp} must be a UTC Datetime")
    if not frame.schema["seconds_elapsed"].is_integer():
        raise TypeError("seconds_elapsed must be an integer")
    if frame.filter(
        (pl.col("observed_at") - pl.col("window_start")).dt.total_microseconds()
        != pl.col("seconds_elapsed").cast(pl.Int64) * 1_000_000
    ).height:
        raise ValueError("D4 side-calibration timestamps are non-causal or misaligned")
    if frame.filter(~pl.col("seconds_elapsed").is_between(1, 55)).height:
        raise ValueError("D4 side-calibration frame escaped seconds 1--55")
    sides = frame[SELECTED_SIDE_COLUMN].cast(pl.String).to_numpy()
    if not np.isin(sides, ("YES", "NO")).all():
        raise ValueError("selected_side must contain only YES or NO")
    probabilities = frame[PROBABILITY_COLUMN].cast(pl.Float64).to_numpy()
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < probability_clip) | (probabilities > 1.0 - probability_clip)
    ):
        raise ValueError("D4 probabilities violate the frozen probability clip")
    if require_label:
        labels = frame[LABEL_COLUMN].cast(pl.Float64).to_numpy()
        if not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError("D4 side-calibration labels must be binary")
        inconsistent = (
            frame.group_by("market_id")
            .agg(
                pl.col("window_start").n_unique().alias("starts"),
                pl.col(LABEL_COLUMN).n_unique().alias("labels"),
            )
            .filter((pl.col("starts") != 1) | (pl.col("labels") != 1))
        )
        if inconsistent.height:
            raise ValueError("D4 side-calibration market labels are inconsistent")


def _matching_calibration_window(
    frame: pl.DataFrame,
    config: D4SideCalibrationConfig,
    *,
    fold_name: str | None,
) -> EvidenceWindow:
    _validate_probability_frame(frame, config.calibration.probability_clip, require_label=True)
    observed_days = set(frame["window_start"].dt.date().cast(pl.String).to_list())
    if fold_name is not None:
        folds = [fold for fold in config.folds if fold.name == fold_name]
        if len(folds) != 1:
            raise ValueError(f"unknown D4 side-calibration fold: {fold_name}")
        fold = folds[0]
        if not observed_days or not all(
            fold.calibration.start.date() <= date.fromisoformat(day) < fold.calibration.end.date()
            for day in observed_days
        ):
            raise ValueError("calibration evidence escaped the requested fold window")
        return fold.calibration
    matches: list[EvidenceWindow] = []
    for fold in config.folds:
        required_days = {
            (fold.calibration.start + timedelta(days=offset)).date().isoformat()
            for offset in range(config.calibration_days)
        }
        if observed_days == required_days:
            matches.append(fold.calibration)
    if len(matches) != 1:
        raise ValueError("calibration evidence must match exactly one frozen 14-day fold window")
    return matches[0]


def _calibration_support_failures(
    support: D4SideCalibrationSupport,
    config: D4SideCalibrationConfig,
    arm: D4SideCalibrationArm,
) -> tuple[str, ...]:
    gates = config.support_gates
    failures: list[str] = []
    if support.no_markets < gates.minimum_calibration_no_markets:
        failures.append("NO markets")
    if support.no_utc_days < gates.minimum_calibration_utc_days:
        failures.append("NO UTC days")
    if support.no_winning_markets < gates.minimum_no_winning_markets:
        failures.append("NO winning markets")
    if support.no_losing_markets < gates.minimum_no_losing_markets:
        failures.append("NO losing markets")
    if (
        arm.strategy == "time_local_no_intercepts"
        and gates.require_both_outcomes_per_time_cell_for_local
    ):
        for cell in support.time_cells:
            if cell.winning_markets == 0 or cell.losing_markets == 0:
                failures.append(f"{cell.name} both outcomes")
    return tuple(failures)


def _cell_support(frame: pl.DataFrame, band: NoTimeBand) -> D4NoCellSupport:
    selected = frame.filter(
        pl.col("seconds_elapsed").is_between(
            band.start_second, band.end_second_exclusive, closed="left"
        )
    )
    wins, losses = _market_outcome_counts(selected)
    return D4NoCellSupport(
        name=band.name,
        rows=selected.height,
        markets=selected["market_id"].n_unique(),
        utc_days=selected["window_start"].dt.date().n_unique(),
        winning_markets=wins,
        losing_markets=losses,
    )


def _market_outcome_counts(frame: pl.DataFrame) -> tuple[int, int]:
    if frame.is_empty():
        return 0, 0
    markets = frame.select("market_id", LABEL_COLUMN).unique()
    wins = int((markets[LABEL_COLUMN].cast(pl.Float64) == 0.0).sum())
    return wins, markets.height - wins


def _fit_conservative_offset(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    contract: D4SideCalibrationContract,
) -> float:
    if probabilities.size == 0 or probabilities.shape != labels.shape:
        raise ValueError("NO calibration arrays must be non-empty and aligned")

    def objective(offset: float) -> float:
        return _calibration_objective(probabilities, labels, weights, offset, contract)

    result = minimize_scalar(
        objective,
        bounds=(
            contract.minimum_no_logit_offset,
            contract.maximum_no_logit_offset,
        ),
        method="bounded",
        options={
            "xatol": contract.optimizer_absolute_tolerance,
            "maxiter": contract.optimizer_max_iterations,
        },
    )
    choices = (
        contract.minimum_no_logit_offset,
        float(result.x),
        contract.maximum_no_logit_offset,
    )
    offset = min(choices, key=lambda value: (objective(value), -value))
    if not result.success or not math.isfinite(offset):
        raise RuntimeError("D4 NO offset optimizer failed")
    if offset > 0.0 or offset < contract.minimum_no_logit_offset:
        raise RuntimeError("D4 NO offset escaped conservative bounds")
    return float(offset)


def _calibration_objective(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    offset: float,
    contract: D4SideCalibrationContract,
) -> float:
    if not (
        probabilities.shape == labels.shape == weights.shape
        and probabilities.ndim == 1
        and probabilities.size
        and np.isfinite(weights).all()
        and np.all(weights > 0.0)
    ):
        raise ValueError("D4 NO calibration objective arrays are invalid")
    eta = _logit(probabilities, contract.probability_clip) + offset
    exposure = float(weights.sum())
    loss = float(np.sum(weights * (np.logaddexp(0.0, eta) - labels * eta)))
    penalty = 0.5 * contract.identity_l2 * exposure * offset**2
    return loss + penalty


def _joint_objective(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    seconds: np.ndarray,
    offsets: tuple[tuple[str, float], ...],
    arm: D4SideCalibrationArm,
    contract: D4SideCalibrationContract,
) -> float:
    if arm.strategy == "identity":
        return _calibration_objective(probabilities, labels, weights, 0.0, contract)
    offset_map = dict(offsets)
    if arm.strategy == "pooled_no_intercept":
        return _calibration_objective(
            probabilities, labels, weights, offset_map[POOLED_NO_CELL], contract
        )
    total = 0.0
    assigned = np.zeros(probabilities.size, dtype=bool)
    for band in contract.time_bands:
        selected = band.contains(seconds)
        assigned |= selected
        total += _calibration_objective(
            probabilities[selected],
            labels[selected],
            weights[selected],
            offset_map[band.name],
            contract,
        )
    if not assigned.all():
        raise RuntimeError("NO calibration objective escaped time bands")
    return total


def _fold_for_day(
    day: date, development_start: datetime, calibration_days: int
) -> BookAdmissionFold:
    validation_start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    calibration_start = validation_start - timedelta(days=calibration_days)
    return BookAdmissionFold(
        name=day.isoformat(),
        fit=EvidenceWindow(development_start, calibration_start),
        calibration=EvidenceWindow(calibration_start, validation_start),
        validation=EvidenceWindow(validation_start, validation_start + timedelta(days=1)),
    )


def _parse_utc(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("D4 side-calibration timestamps must be UTC")
    return parsed.astimezone(UTC)


def _parse_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _logit(values: np.ndarray, probability_clip: float) -> np.ndarray:
    clipped = np.clip(
        np.asarray(values, dtype=np.float64), probability_clip, 1.0 - probability_clip
    )
    return np.log(clipped) - np.log1p(-clipped)


def _require_columns(frame: pl.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
