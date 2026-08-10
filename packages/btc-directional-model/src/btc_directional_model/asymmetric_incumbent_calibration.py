"""Frozen-estimator calibration for the Core+Oracle asymmetric incumbent."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize

from .asymmetric_incumbent_replay import (
    FROZEN_ASYMMETRIC_INCUMBENT_KEY,
    FROZEN_ASYMMETRIC_INCUMBENT_MODEL_SHA256,
    FrozenAsymmetricRuntimeModel,
    load_frozen_asymmetric_incumbent,
    validate_asymmetric_runtime_model,
)
from .asymmetric_value_config import EvidenceWindow
from .core_config import parse_utc_day
from .runtime_export import reached_leaf_value

INCUMBENT_CALIBRATION_PROFILE = "btc_asymmetric_incumbent_calibration"
INCUMBENT_ARM = "I0_incumbent"
SUPPORTED_ARM = "C1_supported"
COMPRESSED_ARM = "C2_compressed"
DAY_BALANCED_ARM = "C3_day_balanced"
INCUMBENT_CALIBRATION_ARM_NAMES = (
    INCUMBENT_ARM,
    SUPPORTED_ARM,
    COMPRESSED_ARM,
    DAY_BALANCED_ARM,
)
MARKET_EQUAL = "market_equal"
DAY_MARKET_ROW_EQUAL = "day_market_row_equal"
DEFAULT_INCUMBENT_CALIBRATION_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "btc-5m-asymmetric-core-oracle-gen2-calibration-20260716-20260802.toml"
)
_TARGET_SIDES = ("YES", "NO")
_PRICE_BANDS = 10
_PRICE_BAND_WIDTH = 0.1
_PROBABILITY_CLIP = (1e-9, 1.0 - 1e-9)


@dataclass(frozen=True)
class IncumbentCalibrationArm:
    name: str
    kind: str
    weighting: str


@dataclass(frozen=True)
class IncumbentPolicyContract:
    name: str
    quantity: float
    vwap_quantity: float
    maximum_depth_participation: float
    execution_reserve_per_share: float
    minimum_edge_per_share: float
    minimum_share_price: float
    maximum_share_price: float
    maximum_cost_per_share: float
    minimum_entry_second: int
    maximum_entry_second: int


@dataclass(frozen=True)
class IncumbentProbabilityGates:
    maximum_paired_degradation_upper_95: float
    maximum_selected_opportunity_bias: float
    maximum_cell_bias: float
    minimum_noninferior_days: int
    required_comparison_days: int
    require_brier_noninferiority: bool
    require_log_loss_noninferiority: bool
    require_one_proper_score_improvement: bool
    require_all_target_cells_fitted: bool
    require_both_outcomes_per_cell: bool


@dataclass(frozen=True)
class IncumbentCorrectionGates:
    minimum_net_corrected_decisions: int
    minimum_improvement_days: int
    minimum_correctness_margin: float
    minimum_correctness_margin_delta: float
    require_positive_candidate_only_stressed_pnl: bool


@dataclass(frozen=True)
class IncumbentEconomicGates:
    minimum_incumbent_frequency_fraction: float
    minimum_yes_entries: int
    minimum_no_entries: int
    minimum_stressed_expectancy_per_trade: float
    minimum_profit_factor: float
    maximum_mean_share_price: float
    maximum_loss_recovery_burden: float
    maximum_average_loss: float
    maximum_single_loss: float
    maximum_drawdown: float
    maximum_primary_metric_regression_fraction: float
    minimum_paired_profit_per_market_lower_95: float


@dataclass(frozen=True)
class IncumbentHistoricalBaseline:
    trades: int
    wins: int
    losses: int
    win_rate: float
    net_profit: float
    expectancy_per_trade: float
    profit_factor: float
    maximum_drawdown: float
    loss_recovery_burden: float


@dataclass(frozen=True)
class ConditionalHistogramContract:
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float


@dataclass(frozen=True)
class ConditionalEstimatorCandidate:
    name: str
    target_weight: float
    boundary_weighted: bool
    boundary_minimum_edge: float | None
    boundary_maximum_edge: float | None
    boundary_closed: str | None
    boundary_multiplier: float | None
    weighting: str | None


@dataclass(frozen=True)
class ConditionalEstimatorContract:
    enabled: bool
    trigger: str
    maximum_total_challengers: int
    forbidden_triggers: tuple[str, ...]
    histogram: ConditionalHistogramContract
    candidates: tuple[ConditionalEstimatorCandidate, ...]


@dataclass(frozen=True)
class IncumbentCalibrationConfig:
    source_path: Path
    package_root: Path
    evidence_scope: str
    process_id: str
    model_key: str
    model_sha256: str
    feature_schema_sha256: str
    feature_count: int
    estimator_sha256: str
    features_sha256: str
    parent_time_calibrators_sha256: str
    non_target_cells_sha256: str
    total_cells: int
    target_cells: int
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
    compression_slope_bounds: tuple[float, float]
    intercept_bounds: tuple[float, float]
    arms: tuple[IncumbentCalibrationArm, ...]
    random_seed: int
    bootstrap_resamples: int
    optimizer_max_iterations: int
    optimizer_ftol: float
    optimizer_gtol: float
    optimizer_max_line_search_steps: int
    policy: IncumbentPolicyContract
    probability_gates: IncumbentProbabilityGates
    correction_gates: IncumbentCorrectionGates
    economic_gates: IncumbentEconomicGates
    historical_baseline: IncumbentHistoricalBaseline
    conditional_estimator: ConditionalEstimatorContract
    asymmetric_value_config: Path
    incumbent_runtime_dir: Path
    incumbent_model: Path
    runs: Path


@dataclass(frozen=True)
class IncumbentCalibrationCellFit:
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
    converged: bool
    iterations: int
    objective: float | None
    weighted_log_loss: float | None


@dataclass(frozen=True)
class IncumbentCalibrationFit:
    arm_name: str
    payload: dict[str, Any]
    cells: tuple[IncumbentCalibrationCellFit, ...]
    weighting: str
    identity_l2: float
    converged: bool
    iterations: int
    objective: float | None
    payload_sha256: str

    def probability(
        self,
        model: FrozenAsymmetricRuntimeModel,
        frame: pl.DataFrame,
        *,
        parent_probabilities: np.ndarray | None = None,
    ) -> np.ndarray:
        return score_incumbent_calibration_payload(
            model,
            self.payload,
            frame,
            parent_probabilities=parent_probabilities,
        )


def load_incumbent_calibration_config(
    path: Path = DEFAULT_INCUMBENT_CALIBRATION_CONFIG,
) -> IncumbentCalibrationConfig:
    """Load and validate the immutable incumbent-only calibration contract."""

    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    benchmark = raw["benchmark"]
    if benchmark.get("profile") != INCUMBENT_CALIBRATION_PROFILE:
        raise ValueError("incumbent calibration profile identity changed")
    if (
        benchmark.get("paper_only") is not True
        or benchmark.get("live_capital_allowed") is not False
    ):
        raise ValueError("incumbent calibration must remain offline and paper-only")

    def window(name: str) -> EvidenceWindow:
        values = raw["windows"][name]
        return EvidenceWindow(
            start=parse_utc_day(values["start"]),
            end=parse_utc_day(values["end"]),
        )

    incumbent = raw["incumbent"]
    calibration = raw["calibration"]
    model = raw["model"]
    policy = raw["policy"]
    probability = raw["probability_gates"]
    correction = raw["correction_gates"]
    economics = raw["economic_gates"]
    baseline = raw["historical_baseline"]
    conditional = raw["conditional_estimator"]
    histogram = conditional["histogram"]
    paths = raw["paths"]
    config = IncumbentCalibrationConfig(
        source_path=source_path,
        package_root=package_root,
        evidence_scope=str(benchmark["evidence_scope"]),
        process_id=str(incumbent["process_id"]),
        model_key=str(incumbent["model_key"]),
        model_sha256=str(incumbent["model_sha256"]),
        feature_schema_sha256=str(incumbent["feature_schema_sha256"]),
        feature_count=int(incumbent["feature_count"]),
        estimator_sha256=str(incumbent["estimator_sha256"]),
        features_sha256=str(incumbent["features_sha256"]),
        parent_time_calibrators_sha256=str(incumbent["parent_time_calibrators_sha256"]),
        non_target_cells_sha256=str(incumbent["non_target_cells_sha256"]),
        total_cells=int(incumbent["total_cells"]),
        target_cells=int(incumbent["target_cells"]),
        calibration_fit=window("calibration_fit"),
        matched_comparison=window("matched_comparison"),
        final_refit=window("final_refit"),
        target_price_band=(
            float(calibration["minimum_price"]),
            float(calibration["maximum_price"]),
        ),
        target_time_bands=tuple(
            (int(item["start_second"]), int(item["end_second_exclusive"]))
            for item in calibration["time_bands"]
        ),
        sides=tuple(str(side) for side in calibration["sides"]),
        minimum_markets_per_cell=int(calibration["minimum_markets_per_cell"]),
        minimum_days_per_cell=int(calibration["minimum_days_per_cell"]),
        identity_l2=float(calibration["identity_l2"]),
        slope_bounds=(
            float(calibration["slope_minimum"]),
            float(calibration["slope_maximum"]),
        ),
        compression_slope_bounds=(
            float(calibration["compression_slope_minimum"]),
            float(calibration["compression_slope_maximum"]),
        ),
        intercept_bounds=(
            float(calibration["intercept_minimum"]),
            float(calibration["intercept_maximum"]),
        ),
        arms=tuple(
            IncumbentCalibrationArm(
                name=str(item["name"]),
                kind=str(item["kind"]),
                weighting=str(item["weighting"]),
            )
            for item in calibration["arms"]
        ),
        random_seed=int(model["random_seed"]),
        bootstrap_resamples=int(model["bootstrap_resamples"]),
        optimizer_max_iterations=int(model["optimizer_max_iterations"]),
        optimizer_ftol=float(model["optimizer_ftol"]),
        optimizer_gtol=float(model["optimizer_gtol"]),
        optimizer_max_line_search_steps=int(model["optimizer_max_line_search_steps"]),
        policy=IncumbentPolicyContract(
            name=str(policy["name"]),
            quantity=float(policy["quantity"]),
            vwap_quantity=float(policy["vwap_quantity"]),
            maximum_depth_participation=float(policy["maximum_depth_participation"]),
            execution_reserve_per_share=float(policy["execution_reserve_per_share"]),
            minimum_edge_per_share=float(policy["minimum_edge_per_share"]),
            minimum_share_price=float(policy["minimum_share_price"]),
            maximum_share_price=float(policy["maximum_share_price"]),
            maximum_cost_per_share=float(policy["maximum_cost_per_share"]),
            minimum_entry_second=int(policy["minimum_entry_second"]),
            maximum_entry_second=int(policy["maximum_entry_second"]),
        ),
        probability_gates=IncumbentProbabilityGates(
            maximum_paired_degradation_upper_95=float(
                probability["maximum_paired_degradation_upper_95"]
            ),
            maximum_selected_opportunity_bias=float(
                probability["maximum_selected_opportunity_bias"]
            ),
            maximum_cell_bias=float(probability["maximum_cell_bias"]),
            minimum_noninferior_days=int(probability["minimum_noninferior_days"]),
            required_comparison_days=int(probability["required_comparison_days"]),
            require_brier_noninferiority=bool(probability["require_brier_noninferiority"]),
            require_log_loss_noninferiority=bool(probability["require_log_loss_noninferiority"]),
            require_one_proper_score_improvement=bool(
                probability["require_one_proper_score_improvement"]
            ),
            require_all_target_cells_fitted=bool(probability["require_all_target_cells_fitted"]),
            require_both_outcomes_per_cell=bool(probability["require_both_outcomes_per_cell"]),
        ),
        correction_gates=IncumbentCorrectionGates(
            minimum_net_corrected_decisions=int(correction["minimum_net_corrected_decisions"]),
            minimum_improvement_days=int(correction["minimum_improvement_days"]),
            minimum_correctness_margin=float(correction["minimum_correctness_margin"]),
            minimum_correctness_margin_delta=float(correction["minimum_correctness_margin_delta"]),
            require_positive_candidate_only_stressed_pnl=bool(
                correction["require_positive_candidate_only_stressed_pnl"]
            ),
        ),
        economic_gates=IncumbentEconomicGates(
            minimum_incumbent_frequency_fraction=float(
                economics["minimum_incumbent_frequency_fraction"]
            ),
            minimum_yes_entries=int(economics["minimum_yes_entries"]),
            minimum_no_entries=int(economics["minimum_no_entries"]),
            minimum_stressed_expectancy_per_trade=float(
                economics["minimum_stressed_expectancy_per_trade"]
            ),
            minimum_profit_factor=float(economics["minimum_profit_factor"]),
            maximum_mean_share_price=float(economics["maximum_mean_share_price"]),
            maximum_loss_recovery_burden=float(economics["maximum_loss_recovery_burden"]),
            maximum_average_loss=float(economics["maximum_average_loss"]),
            maximum_single_loss=float(economics["maximum_single_loss"]),
            maximum_drawdown=float(economics["maximum_drawdown"]),
            maximum_primary_metric_regression_fraction=float(
                economics["maximum_primary_metric_regression_fraction"]
            ),
            minimum_paired_profit_per_market_lower_95=float(
                economics["minimum_paired_profit_per_market_lower_95"]
            ),
        ),
        historical_baseline=IncumbentHistoricalBaseline(
            trades=int(baseline["trades"]),
            wins=int(baseline["wins"]),
            losses=int(baseline["losses"]),
            win_rate=float(baseline["win_rate"]),
            net_profit=float(baseline["net_profit"]),
            expectancy_per_trade=float(baseline["expectancy_per_trade"]),
            profit_factor=float(baseline["profit_factor"]),
            maximum_drawdown=float(baseline["maximum_drawdown"]),
            loss_recovery_burden=float(baseline["loss_recovery_burden"]),
        ),
        conditional_estimator=ConditionalEstimatorContract(
            enabled=bool(conditional["enabled"]),
            trigger=str(conditional["trigger"]),
            maximum_total_challengers=int(conditional["maximum_total_challengers"]),
            forbidden_triggers=tuple(str(value) for value in conditional["forbidden_triggers"]),
            histogram=ConditionalHistogramContract(
                learning_rate=float(histogram["learning_rate"]),
                max_iter=int(histogram["max_iter"]),
                max_leaf_nodes=int(histogram["max_leaf_nodes"]),
                min_samples_leaf=int(histogram["min_samples_leaf"]),
                l2_regularization=float(histogram["l2_regularization"]),
            ),
            candidates=tuple(
                ConditionalEstimatorCandidate(
                    name=str(item["name"]),
                    target_weight=float(item["target_weight"]),
                    boundary_weighted=bool(item["boundary_weighted"]),
                    boundary_minimum_edge=(
                        float(item["boundary_minimum_edge"])
                        if "boundary_minimum_edge" in item
                        else None
                    ),
                    boundary_maximum_edge=(
                        float(item["boundary_maximum_edge"])
                        if "boundary_maximum_edge" in item
                        else None
                    ),
                    boundary_closed=(
                        str(item["boundary_closed"]) if "boundary_closed" in item else None
                    ),
                    boundary_multiplier=(
                        float(item["boundary_multiplier"])
                        if "boundary_multiplier" in item
                        else None
                    ),
                    weighting=(str(item["weighting"]) if "weighting" in item else None),
                )
                for item in conditional["candidates"]
            ),
        ),
        asymmetric_value_config=package_root / str(paths["asymmetric_value_config"]),
        incumbent_runtime_dir=package_root / str(paths["incumbent_runtime_dir"]),
        incumbent_model=package_root / str(paths["incumbent_model"]),
        runs=package_root / str(paths["runs"]),
    )
    validate_incumbent_calibration_config(config)
    return config


def validate_incumbent_calibration_config(config: IncumbentCalibrationConfig) -> None:
    """Fail closed if the incumbent, chronology, policy, or candidate registry drifts."""

    expected_windows = (
        (config.calibration_fit.start.isoformat(), config.calibration_fit.end.isoformat()),
        (config.matched_comparison.start.isoformat(), config.matched_comparison.end.isoformat()),
        (config.final_refit.start.isoformat(), config.final_refit.end.isoformat()),
    )
    if expected_windows != (
        ("2026-07-16T00:00:00+00:00", "2026-07-23T00:00:00+00:00"),
        ("2026-07-23T00:00:00+00:00", "2026-08-02T00:00:00+00:00"),
        ("2026-07-16T00:00:00+00:00", "2026-08-02T00:00:00+00:00"),
    ):
        raise ValueError("incumbent calibration evidence chronology changed")
    if config.evidence_scope != "consumed_cross_day_development":
        raise ValueError("incumbent calibration must disclose consumed development evidence")
    if (
        config.process_id != "81f82de7-002b-4ac7-814b-236c6742d81c"
        or config.model_key != FROZEN_ASYMMETRIC_INCUMBENT_KEY
        or config.model_sha256 != FROZEN_ASYMMETRIC_INCUMBENT_MODEL_SHA256
        or config.feature_count != 75
        or config.total_cells != 160
        or config.target_cells != 8
    ):
        raise ValueError("frozen incumbent identity changed")
    if config.target_price_band != (0.20, 0.30) or config.sides != _TARGET_SIDES:
        raise ValueError("incumbent target side/price contract changed")
    if config.target_time_bands != ((1, 15), (15, 30), (30, 45), (45, 60)):
        raise ValueError("incumbent target time-cell contract changed")
    if (
        config.minimum_markets_per_cell != 50
        or config.minimum_days_per_cell != 7
        or config.identity_l2 != 1.0
        or config.slope_bounds != (0.05, 3.0)
        or config.compression_slope_bounds != (0.5, 1.0)
        or config.intercept_bounds != (-2.0, 2.0)
    ):
        raise ValueError("incumbent calibration optimization contract changed")
    if tuple(arm.name for arm in config.arms) != INCUMBENT_CALIBRATION_ARM_NAMES:
        raise ValueError("incumbent calibration candidate registry changed")
    if tuple((arm.kind, arm.weighting) for arm in config.arms) != (
        ("incumbent_identity", "none"),
        ("independent_cells", MARKET_EQUAL),
        ("shared_compression", MARKET_EQUAL),
        ("independent_cells", DAY_MARKET_ROW_EQUAL),
    ):
        raise ValueError("incumbent calibration arm semantics changed")
    expected_policy = IncumbentPolicyContract(
        name="raw20_30_by55_edge_3c",
        quantity=5.0,
        vwap_quantity=5.0,
        maximum_depth_participation=0.25,
        execution_reserve_per_share=0.01,
        minimum_edge_per_share=0.03,
        minimum_share_price=0.20,
        maximum_share_price=0.30,
        maximum_cost_per_share=0.35,
        minimum_entry_second=1,
        maximum_entry_second=55,
    )
    if config.policy != expected_policy:
        raise ValueError("frozen incumbent policy changed")
    if config.incumbent_model != config.incumbent_runtime_dir / "model.json":
        raise ValueError("frozen incumbent runtime directory/model path changed")
    _validate_conditional_estimator(config.conditional_estimator)

    _validate_source_asymmetric_value_config(config)
    incumbent = load_frozen_asymmetric_incumbent(config.incumbent_model)
    payload = incumbent.payload
    calibration = payload["asymmetric_value_calibration"]
    non_target = [
        cell
        for cell in calibration["side_price_cells"]
        if not _is_target_runtime_cell(cell, config)
    ]
    hashes = {
        "features": _canonical_sha256(payload["features"]),
        "estimator": _canonical_sha256(payload["estimator"]),
        "parent": _canonical_sha256(calibration["time_bands"]),
        "non_target": _canonical_sha256(non_target),
    }
    if (
        payload["features"]["schema_sha256"] != config.feature_schema_sha256
        or len(payload["features"]["names"]) != config.feature_count
        or hashes["features"] != config.features_sha256
        or hashes["estimator"] != config.estimator_sha256
        or hashes["parent"] != config.parent_time_calibrators_sha256
        or hashes["non_target"] != config.non_target_cells_sha256
        or len(non_target) != config.total_cells - config.target_cells
    ):
        raise RuntimeError("frozen incumbent estimator/calibration parity changed")
    target = [
        cell for cell in calibration["side_price_cells"] if _is_target_runtime_cell(cell, config)
    ]
    if len(target) != config.target_cells or any(
        cell["fitted"] is not False
        or float(cell["slope"]) != 1.0
        or float(cell["intercept"]) != 0.0
        for cell in target
    ):
        raise RuntimeError("incumbent target cells no longer use fallback identity calibration")


def frozen_parent_probabilities(
    model: FrozenAsymmetricRuntimeModel,
    frame: pl.DataFrame,
) -> np.ndarray:
    """Score the frozen trees and frozen parent time calibrators exactly once."""

    _validate_scoring_frame(model, frame)
    feature_matrix = np.asarray(frame.select(*model.feature_names).to_numpy(), dtype=np.float64)
    medians = np.asarray(model.payload["features"]["imputation_medians"], dtype=np.float64)
    feature_matrix = np.where(np.isfinite(feature_matrix), feature_matrix, medians)
    estimator = model.payload["estimator"]
    raw_logits = np.full(frame.height, float(estimator["baseline_logit"]), dtype=np.float64)
    for row_index, row in enumerate(feature_matrix):
        raw_logits[row_index] += sum(
            reached_leaf_value(tree["nodes"], row) for tree in estimator["trees"]
        )
    if not np.isfinite(raw_logits).all():
        raise RuntimeError("frozen incumbent estimator produced non-finite logits")
    elapsed = frame["seconds_elapsed"].to_numpy().astype(np.int64)
    probabilities = np.full(frame.height, np.nan, dtype=np.float64)
    for band in model.payload["asymmetric_value_calibration"]["time_bands"]:
        selected = (elapsed >= int(band["start_seconds"])) & (
            elapsed < int(band["end_seconds_exclusive"])
        )
        eta = raw_logits[selected] * float(band["slope"]) + float(band["intercept"])
        probabilities[selected] = _sigmoid(eta)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("frozen parent time calibration does not cover the frame")
    return np.clip(probabilities, *_PROBABILITY_CLIP)


def score_incumbent_calibration_payload(
    model: FrozenAsymmetricRuntimeModel,
    payload: dict[str, Any],
    frame: pl.DataFrame,
    *,
    parent_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a cloned payload's side/price cells to cached frozen-parent probabilities."""

    _validate_scoring_frame(model, frame)
    validate_asymmetric_runtime_model(payload)
    _validate_payload_parent_parity(model.payload, payload)
    parents = (
        frozen_parent_probabilities(model, frame)
        if parent_probabilities is None
        else np.asarray(parent_probabilities, dtype=np.float64)
    )
    if parents.shape != (frame.height,) or not np.isfinite(parents).all():
        raise ValueError("cached frozen-parent probabilities are invalid")
    parents = np.clip(parents, *_PROBABILITY_CLIP)
    calibration = payload["asymmetric_value_calibration"]
    bands = calibration["time_bands"]
    time_indices = _runtime_time_indices(frame["seconds_elapsed"].to_numpy(), bands)
    yes_indices = _price_band_indices(frame["yes_ask_vwap_5"].to_numpy())
    no_indices = _price_band_indices(frame["no_ask_vwap_5"].to_numpy())
    slopes, intercepts = _runtime_cell_arrays(calibration["side_price_cells"], bands)
    parent_logit = _logit(parents)
    yes_eta = (
        parent_logit * slopes[time_indices, yes_indices, 0]
        + intercepts[time_indices, yes_indices, 0]
    )
    no_eta = (
        -parent_logit * slopes[time_indices, no_indices, 1]
        + intercepts[time_indices, no_indices, 1]
    )
    probability = _sigmoid(0.5 * (yes_eta - no_eta))
    if not np.isfinite(probability).all():
        raise RuntimeError("cloned incumbent calibration produced non-finite probabilities")
    return np.clip(probability, *_PROBABILITY_CLIP)


def fit_incumbent_calibration_arm(
    arm_name: str,
    model: FrozenAsymmetricRuntimeModel,
    frame: pl.DataFrame,
    config: IncumbentCalibrationConfig,
    *,
    window: EvidenceWindow | None = None,
    parent_probabilities: np.ndarray | None = None,
) -> IncumbentCalibrationFit:
    """Fit exactly eight target cells while retaining all other incumbent bytes logically."""

    arms = {arm.name: arm for arm in config.arms}
    if arm_name not in arms:
        raise ValueError(f"unknown incumbent calibration arm: {arm_name}")
    if model.model_sha256 != config.model_sha256 or model.payload["model_key"] != config.model_key:
        raise RuntimeError("calibration fit did not receive the frozen incumbent")
    _validate_scoring_frame(model, frame)
    if "label_up" not in frame.columns:
        raise ValueError("incumbent calibration fit requires label_up")
    selected_window = window or config.calibration_fit
    indexed = frame.with_row_index("__incumbent_row_index").filter(
        (pl.col("window_start") >= selected_window.start)
        & (pl.col("window_start") < selected_window.end)
    )
    if indexed.is_empty():
        raise ValueError("incumbent calibration window has no rows")
    indexed = indexed.sort("window_start", "market_id", "seconds_elapsed")
    invalid_identity = indexed.filter(
        pl.col("market_id").is_null()
        | pl.col("window_start").is_null()
        | pl.col("label_up").is_null()
    )
    if invalid_identity.height:
        raise ValueError("incumbent calibration market identity or outcome is incomplete")
    inconsistent = (
        indexed.group_by("market_id")
        .agg(
            pl.col("window_start").n_unique().alias("window_starts"),
            pl.col("label_up").n_unique().alias("labels"),
        )
        .filter((pl.col("window_starts") != 1) | (pl.col("labels") != 1))
    )
    if inconsistent.height:
        raise ValueError("incumbent calibration market identity or outcome is inconsistent")
    duplicate = (
        indexed.group_by("market_id", "window_start", "seconds_elapsed")
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise ValueError("incumbent calibration contains duplicate market-second keys")
    labels = indexed["label_up"].to_numpy()
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("incumbent calibration requires resolved binary outcomes")
    parents_all = (
        frozen_parent_probabilities(model, frame)
        if parent_probabilities is None
        else np.asarray(parent_probabilities, dtype=np.float64)
    )
    if parents_all.shape != (frame.height,) or not np.isfinite(parents_all).all():
        raise ValueError("cached frozen-parent probabilities are invalid")
    source_indices = indexed["__incumbent_row_index"].to_numpy().astype(np.int64)
    parents = np.clip(parents_all[source_indices], *_PROBABILITY_CLIP)
    cohort = indexed.drop("__incumbent_row_index")
    prepared = _prepare_target_cohort(cohort, parents, config)
    support = _target_cell_support(prepared, config)
    _validate_target_support(support, config)
    arm = arms[arm_name]
    payload = copy.deepcopy(model.payload)
    if arm_name == INCUMBENT_ARM:
        cells = _incumbent_cell_diagnostics(payload, support, config)
        return IncumbentCalibrationFit(
            arm_name=arm_name,
            payload=payload,
            cells=cells,
            weighting=arm.weighting,
            identity_l2=config.identity_l2,
            converged=True,
            iterations=0,
            objective=None,
            payload_sha256=_canonical_sha256(payload),
        )

    weights = _calibration_weights(
        prepared["market_ids"],
        prepared["utc_days"],
        weighting=arm.weighting,
    )
    if arm.kind == "independent_cells":
        result, slopes, intercepts = _fit_independent_cells(prepared, weights, config)
    elif arm.kind == "shared_compression":
        result, slopes, intercepts = _fit_shared_compression(prepared, weights, config)
    else:
        raise ValueError(f"unsupported incumbent calibration arm kind: {arm.kind}")
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"{arm_name} target-cell optimizer did not converge: {result.message}")
    _replace_target_cells(payload, slopes, intercepts, config)
    validate_incumbent_calibration_parity(model.payload, payload, config)
    probability = score_incumbent_calibration_payload(
        model,
        payload,
        prepared["frame"],
        parent_probabilities=prepared["parents"],
    )
    cells = _fitted_cell_diagnostics(
        support,
        slopes,
        intercepts,
        probability,
        prepared,
        arm.weighting,
        int(result.nit),
        float(result.fun),
        config,
    )
    return IncumbentCalibrationFit(
        arm_name=arm_name,
        payload=payload,
        cells=cells,
        weighting=arm.weighting,
        identity_l2=config.identity_l2,
        converged=True,
        iterations=int(result.nit),
        objective=float(result.fun),
        payload_sha256=_canonical_sha256(payload),
    )


def clone_incumbent_with_calibration(
    model: FrozenAsymmetricRuntimeModel,
    fit: IncumbentCalibrationFit,
    *,
    model_key: str,
) -> dict[str, Any]:
    """Create a distinct paper candidate while changing only key and target cells."""

    if not model_key or model_key == model.payload["model_key"]:
        raise ValueError("Gen2 paper model key must be new and non-empty")
    candidate = copy.deepcopy(fit.payload)
    candidate["model_key"] = model_key
    return candidate


def validate_incumbent_calibration_parity(
    incumbent_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    config: IncumbentCalibrationConfig,
) -> None:
    """Prove estimator, features, parent calibration, and 152 other cells are unchanged."""

    validate_asymmetric_runtime_model(incumbent_payload)
    validate_asymmetric_runtime_model(candidate_payload)
    incumbent = copy.deepcopy(incumbent_payload)
    candidate = copy.deepcopy(candidate_payload)
    incumbent_key = incumbent.pop("model_key")
    candidate_key = candidate.pop("model_key")
    incumbent_calibration = incumbent.pop("asymmetric_value_calibration")
    candidate_calibration = candidate.pop("asymmetric_value_calibration")
    if incumbent != candidate:
        raise RuntimeError("Gen2 candidate changed frozen estimator/runtime behavior")
    if not isinstance(incumbent_key, str) or not isinstance(candidate_key, str):
        raise TypeError("Gen2 candidate model identity is invalid")
    if incumbent_calibration["time_bands"] != candidate_calibration["time_bands"]:
        raise RuntimeError("Gen2 candidate changed parent time calibrators")
    incumbent_cells = incumbent_calibration["side_price_cells"]
    candidate_cells = candidate_calibration["side_price_cells"]
    if len(incumbent_cells) != config.total_cells or len(candidate_cells) != config.total_cells:
        raise RuntimeError("Gen2 candidate changed runtime cell cardinality")
    changed = 0
    for before, after in zip(incumbent_cells, candidate_cells, strict=True):
        if _runtime_cell_key(before) != _runtime_cell_key(after):
            raise RuntimeError("Gen2 candidate reordered or changed runtime cell routing")
        if _is_target_runtime_cell(before, config):
            changed += int(before != after)
        elif before != after:
            raise RuntimeError("Gen2 candidate changed a non-target calibration cell")
    if changed not in {0, config.target_cells}:
        raise RuntimeError("Gen2 candidate must change either zero or all eight target cells")


def _prepare_target_cohort(
    frame: pl.DataFrame,
    parents: np.ndarray,
    config: IncumbentCalibrationConfig,
) -> dict[str, Any]:
    elapsed = frame["seconds_elapsed"].to_numpy().astype(np.int64)
    time_indices = np.full(frame.height, -1, dtype=np.int16)
    for index, (start, end) in enumerate(config.target_time_bands):
        time_indices[(elapsed >= start) & (elapsed < end)] = index
    minimum, maximum = config.target_price_band
    yes_prices = frame["yes_ask_vwap_5"].to_numpy().astype(np.float64)
    no_prices = frame["no_ask_vwap_5"].to_numpy().astype(np.float64)
    if (
        not np.isfinite(yes_prices).all()
        or not np.isfinite(no_prices).all()
        or np.any((yes_prices < 0.0) | (yes_prices > 1.0))
        or np.any((no_prices < 0.0) | (no_prices > 1.0))
    ):
        raise ValueError("incumbent calibration prices must be finite within [0, 1]")
    yes_target = (yes_prices >= minimum) & (yes_prices < maximum)
    no_target = (no_prices >= minimum) & (no_prices < maximum)
    selected = (time_indices >= 0) & (yes_target | no_target)
    if not selected.any():
        raise ValueError("incumbent calibration has no target side/price/time rows")
    target_frame = frame.filter(pl.Series(selected))
    return {
        "frame": target_frame,
        "parents": parents[selected],
        "parent_logit": _logit(parents[selected]),
        "labels": frame["label_up"].to_numpy().astype(np.float64)[selected],
        "market_ids": frame["market_id"].cast(pl.String).to_numpy()[selected],
        "utc_days": frame["window_start"].dt.date().cast(pl.String).to_numpy()[selected],
        "time_indices": time_indices[selected],
        "yes_target": yes_target[selected],
        "no_target": no_target[selected],
    }


def _target_cell_support(
    prepared: dict[str, Any],
    config: IncumbentCalibrationConfig,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for time_index, (start, end) in enumerate(config.target_time_bands):
        for side_index, side in enumerate(config.sides):
            selected = (prepared["time_indices"] == time_index) & prepared[
                "yes_target" if side == "YES" else "no_target"
            ]
            side_labels = (
                prepared["labels"][selected]
                if side == "YES"
                else 1.0 - prepared["labels"][selected]
            )
            records.append(
                {
                    "cell_index": time_index * 2 + side_index,
                    "start_second": start,
                    "end_second_exclusive": end,
                    "side": side,
                    "selected": selected,
                    "rows": int(selected.sum()),
                    "markets": int(np.unique(prepared["market_ids"][selected]).size),
                    "utc_days": int(np.unique(prepared["utc_days"][selected]).size),
                    "positives": int(side_labels.sum()),
                    "negatives": int(selected.sum() - side_labels.sum()),
                }
            )
    return tuple(records)


def _validate_target_support(
    support: tuple[dict[str, Any], ...],
    config: IncumbentCalibrationConfig,
) -> None:
    if len(support) != config.target_cells:
        raise RuntimeError("incumbent calibration did not materialize eight target cells")
    failures: list[str] = []
    for cell in support:
        reasons = []
        if cell["markets"] < config.minimum_markets_per_cell:
            reasons.append("markets")
        if cell["utc_days"] < config.minimum_days_per_cell:
            reasons.append("days")
        if cell["positives"] == 0 or cell["negatives"] == 0:
            reasons.append("single_class")
        if reasons:
            failures.append(
                f"{cell['side']} {cell['start_second']}-{cell['end_second_exclusive']}:"
                + "+".join(reasons)
            )
    if failures:
        raise RuntimeError("incumbent target calibration support failed: " + ", ".join(failures))


def _fit_independent_cells(
    prepared: dict[str, Any],
    weights: np.ndarray,
    config: IncumbentCalibrationConfig,
) -> tuple[Any, np.ndarray, np.ndarray]:
    penalty_weights = _cell_exposure_weights(prepared, weights)
    initial = np.tile(np.asarray((1.0, 0.0)), config.target_cells)
    bounds = tuple(
        bound
        for _ in range(config.target_cells)
        for bound in (config.slope_bounds, config.intercept_bounds)
    )
    result = minimize(
        _independent_objective,
        initial,
        args=(prepared, weights, penalty_weights, config.identity_l2),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options=_optimizer_options(config),
    )
    slopes = result.x[0::2].reshape(len(config.target_time_bands), 2)
    intercepts = result.x[1::2].reshape(len(config.target_time_bands), 2)
    return result, slopes, intercepts


def _fit_shared_compression(
    prepared: dict[str, Any],
    weights: np.ndarray,
    config: IncumbentCalibrationConfig,
) -> tuple[Any, np.ndarray, np.ndarray]:
    penalty_weights = _cell_exposure_weights(prepared, weights)
    initial = np.concatenate((np.asarray((1.0,)), np.zeros(config.target_cells)))
    bounds = (config.compression_slope_bounds,) + (config.intercept_bounds,) * config.target_cells
    result = minimize(
        _shared_compression_objective,
        initial,
        args=(prepared, weights, penalty_weights, config.identity_l2),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options=_optimizer_options(config),
    )
    slopes = np.full((len(config.target_time_bands), 2), float(result.x[0]))
    intercepts = result.x[1:].reshape(len(config.target_time_bands), 2)
    return result, slopes, intercepts


def _independent_objective(
    parameters: np.ndarray,
    prepared: dict[str, Any],
    weights: np.ndarray,
    penalty_weights: np.ndarray,
    identity_l2: float,
) -> tuple[float, np.ndarray]:
    slopes = parameters[0::2].reshape(-1, 2)
    intercepts = parameters[1::2].reshape(-1, 2)
    eta = _target_eta(prepared, slopes, intercepts)
    labels = prepared["labels"]
    loss = float(np.sum(weights * (np.logaddexp(0.0, eta) - labels * eta)))
    delta = parameters.copy()
    delta[0::2] -= 1.0
    parameter_weights = np.repeat(penalty_weights, 2)
    penalty = 0.5 * identity_l2 * float((parameter_weights * delta) @ delta)
    error = weights * (_sigmoid(eta) - labels)
    gradient = np.empty_like(parameters)
    for cell_index in range(len(penalty_weights)):
        time_index, side_index = divmod(cell_index, 2)
        selected = (prepared["time_indices"] == time_index) & prepared[
            "yes_target" if side_index == 0 else "no_target"
        ]
        gradient[2 * cell_index] = (
            np.sum(error[selected] * 0.5 * prepared["parent_logit"][selected])
            + identity_l2 * penalty_weights[cell_index] * delta[2 * cell_index]
        )
        sign = 1.0 if side_index == 0 else -1.0
        gradient[2 * cell_index + 1] = (
            np.sum(error[selected] * 0.5 * sign)
            + identity_l2 * penalty_weights[cell_index] * delta[2 * cell_index + 1]
        )
    return loss + penalty, gradient


def _shared_compression_objective(
    parameters: np.ndarray,
    prepared: dict[str, Any],
    weights: np.ndarray,
    penalty_weights: np.ndarray,
    identity_l2: float,
) -> tuple[float, np.ndarray]:
    slope = float(parameters[0])
    slopes = np.full((len(penalty_weights) // 2, 2), slope)
    intercepts = parameters[1:].reshape(-1, 2)
    eta = _target_eta(prepared, slopes, intercepts)
    labels = prepared["labels"]
    loss = float(np.sum(weights * (np.logaddexp(0.0, eta) - labels * eta)))
    slope_weight = float(penalty_weights.sum())
    penalty = (
        0.5
        * identity_l2
        * (slope_weight * (slope - 1.0) ** 2 + float((penalty_weights * parameters[1:] ** 2).sum()))
    )
    error = weights * (_sigmoid(eta) - labels)
    any_target = prepared["yes_target"].astype(np.float64) + prepared["no_target"].astype(
        np.float64
    )
    gradient = np.empty_like(parameters)
    gradient[0] = np.sum(
        error * 0.5 * prepared["parent_logit"] * any_target
    ) + identity_l2 * slope_weight * (slope - 1.0)
    for cell_index in range(len(penalty_weights)):
        time_index, side_index = divmod(cell_index, 2)
        selected = (prepared["time_indices"] == time_index) & prepared[
            "yes_target" if side_index == 0 else "no_target"
        ]
        sign = 1.0 if side_index == 0 else -1.0
        gradient[cell_index + 1] = (
            np.sum(error[selected] * 0.5 * sign)
            + identity_l2 * penalty_weights[cell_index] * parameters[cell_index + 1]
        )
    return loss + penalty, gradient


def _target_eta(
    prepared: dict[str, Any],
    slopes: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    parent_logit = prepared["parent_logit"]
    time_indices = prepared["time_indices"]
    yes_eta = parent_logit.copy()
    no_eta = -parent_logit.copy()
    yes_selected = prepared["yes_target"]
    no_selected = prepared["no_target"]
    yes_eta[yes_selected] = (
        parent_logit[yes_selected] * slopes[time_indices[yes_selected], 0]
        + intercepts[time_indices[yes_selected], 0]
    )
    no_eta[no_selected] = (
        -parent_logit[no_selected] * slopes[time_indices[no_selected], 1]
        + intercepts[time_indices[no_selected], 1]
    )
    return 0.5 * (yes_eta - no_eta)


def _cell_exposure_weights(prepared: dict[str, Any], weights: np.ndarray) -> np.ndarray:
    exposure = np.empty(8, dtype=np.float64)
    for cell_index in range(8):
        time_index, side_index = divmod(cell_index, 2)
        selected = (prepared["time_indices"] == time_index) & prepared[
            "yes_target" if side_index == 0 else "no_target"
        ]
        exposure[cell_index] = weights[selected].sum()
    if not np.isfinite(exposure).all() or np.any(exposure <= 0.0):
        raise RuntimeError("incumbent calibration cell exposure weights are invalid")
    return exposure


def _calibration_weights(
    market_ids: np.ndarray,
    utc_days: np.ndarray,
    *,
    weighting: str,
) -> np.ndarray:
    ids = np.asarray(market_ids)
    days = np.asarray(utc_days)
    if ids.ndim != 1 or days.shape != ids.shape or ids.size == 0:
        raise ValueError("incumbent calibration weighting requires aligned non-empty IDs")
    weights = np.zeros(ids.size, dtype=np.float64)
    if weighting == MARKET_EQUAL:
        _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse].astype(np.float64)
    elif weighting == DAY_MARKET_ROW_EQUAL:
        unique_days = np.unique(days)
        for day in unique_days:
            day_selected = days == day
            day_markets = np.unique(ids[day_selected])
            for market_id in day_markets:
                market_selected = day_selected & (ids == market_id)
                weights[market_selected] = 1.0 / (
                    unique_days.size * day_markets.size * int(market_selected.sum())
                )
    else:
        raise ValueError(f"unsupported incumbent calibration weighting: {weighting}")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise RuntimeError("incumbent calibration produced invalid row weights")
    return weights / weights.sum()


def _replace_target_cells(
    payload: dict[str, Any],
    slopes: np.ndarray,
    intercepts: np.ndarray,
    config: IncumbentCalibrationConfig,
) -> None:
    changed = 0
    time_indices = {band: index for index, band in enumerate(config.target_time_bands)}
    side_indices = {side.lower(): index for index, side in enumerate(config.sides)}
    for cell in payload["asymmetric_value_calibration"]["side_price_cells"]:
        if not _is_target_runtime_cell(cell, config):
            continue
        time_index = time_indices[(cell["start_seconds"], cell["end_seconds_exclusive"])]
        side_index = side_indices[cell["side"]]
        cell["slope"] = float(slopes[time_index, side_index])
        cell["intercept"] = float(intercepts[time_index, side_index])
        cell["fitted"] = True
        cell["fallback"] = None
        changed += 1
    if changed != config.target_cells:
        raise RuntimeError("Gen2 calibration did not replace exactly eight target cells")


def _incumbent_cell_diagnostics(
    payload: dict[str, Any],
    support: tuple[dict[str, Any], ...],
    config: IncumbentCalibrationConfig,
) -> tuple[IncumbentCalibrationCellFit, ...]:
    runtime = {
        _runtime_cell_key(cell): cell
        for cell in payload["asymmetric_value_calibration"]["side_price_cells"]
        if _is_target_runtime_cell(cell, config)
    }
    output = []
    for record in support:
        key = (
            record["start_second"],
            record["end_second_exclusive"],
            config.target_price_band[0],
            config.target_price_band[1],
            record["side"].lower(),
        )
        cell = runtime[key]
        output.append(
            IncumbentCalibrationCellFit(
                **{
                    key: record[key]
                    for key in (
                        "start_second",
                        "end_second_exclusive",
                        "side",
                        "rows",
                        "markets",
                        "utc_days",
                        "positives",
                        "negatives",
                    )
                },
                minimum_price=config.target_price_band[0],
                maximum_price=config.target_price_band[1],
                slope=float(cell["slope"]),
                intercept=float(cell["intercept"]),
                fitted=False,
                fallback=str(cell["fallback"]),
                converged=False,
                iterations=0,
                objective=None,
                weighted_log_loss=None,
            )
        )
    return tuple(output)


def _fitted_cell_diagnostics(
    support: tuple[dict[str, Any], ...],
    slopes: np.ndarray,
    intercepts: np.ndarray,
    probability_up: np.ndarray,
    prepared: dict[str, Any],
    weighting: str,
    iterations: int,
    objective: float,
    config: IncumbentCalibrationConfig,
) -> tuple[IncumbentCalibrationCellFit, ...]:
    output = []
    for record in support:
        cell_index = int(record["cell_index"])
        time_index, side_index = divmod(cell_index, 2)
        selected = record["selected"]
        labels = (
            prepared["labels"][selected] if side_index == 0 else 1.0 - prepared["labels"][selected]
        )
        probability = (
            probability_up[selected] if side_index == 0 else 1.0 - probability_up[selected]
        )
        weights = _calibration_weights(
            prepared["market_ids"][selected],
            prepared["utc_days"][selected],
            weighting=weighting,
        )
        probability = np.clip(probability, *_PROBABILITY_CLIP)
        weighted_log_loss = float(
            -np.sum(
                weights
                * (labels * np.log(probability) + (1.0 - labels) * np.log(1.0 - probability))
            )
        )
        output.append(
            IncumbentCalibrationCellFit(
                **{
                    key: record[key]
                    for key in (
                        "start_second",
                        "end_second_exclusive",
                        "side",
                        "rows",
                        "markets",
                        "utc_days",
                        "positives",
                        "negatives",
                    )
                },
                minimum_price=config.target_price_band[0],
                maximum_price=config.target_price_band[1],
                slope=float(slopes[time_index, side_index]),
                intercept=float(intercepts[time_index, side_index]),
                fitted=True,
                fallback=None,
                converged=True,
                iterations=iterations,
                objective=objective,
                weighted_log_loss=weighted_log_loss,
            )
        )
    return tuple(output)


def _runtime_cell_arrays(
    cells: list[dict[str, Any]],
    bands: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    slopes = np.full((len(bands), _PRICE_BANDS, 2), np.nan, dtype=np.float64)
    intercepts = np.full_like(slopes, np.nan)
    time_indices = {
        (band["start_seconds"], band["end_seconds_exclusive"]): index
        for index, band in enumerate(bands)
    }
    side_indices = {"yes": 0, "no": 1}
    for cell in cells:
        time_index = time_indices[(cell["start_seconds"], cell["end_seconds_exclusive"])]
        price_index = round(float(cell["minimum_price"]) / _PRICE_BAND_WIDTH)
        side_index = side_indices[cell["side"]]
        slopes[time_index, price_index, side_index] = float(cell["slope"])
        intercepts[time_index, price_index, side_index] = float(cell["intercept"])
    if not np.isfinite(slopes).all() or not np.isfinite(intercepts).all():
        raise RuntimeError("runtime calibration cells are incomplete")
    return slopes, intercepts


def _runtime_time_indices(values: Any, bands: list[dict[str, Any]]) -> np.ndarray:
    elapsed = np.asarray(values, dtype=np.int64)
    indices = np.full(elapsed.size, -1, dtype=np.int16)
    for index, band in enumerate(bands):
        indices[
            (elapsed >= int(band["start_seconds"])) & (elapsed < int(band["end_seconds_exclusive"]))
        ] = index
    if np.any(indices < 0):
        raise ValueError("runtime time calibration does not cover the scoring frame")
    return indices


def _price_band_indices(values: Any) -> np.ndarray:
    prices = np.asarray(values, dtype=np.float64)
    if prices.ndim != 1 or not np.isfinite(prices).all():
        raise ValueError("runtime calibration prices must be finite")
    if np.any((prices < 0.0) | (prices > 1.0)):
        raise ValueError("runtime calibration prices must be within [0, 1]")
    return np.clip(
        np.floor(prices * _PRICE_BANDS + 1e-12),
        0,
        _PRICE_BANDS - 1,
    ).astype(np.int16)


def _validate_scoring_frame(model: FrozenAsymmetricRuntimeModel, frame: pl.DataFrame) -> None:
    required = {
        *model.feature_names,
        "market_id",
        "window_start",
        "seconds_elapsed",
        "yes_ask_vwap_5",
        "no_ask_vwap_5",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("incumbent calibration frame is missing columns: " + ", ".join(missing))
    if frame.is_empty():
        raise ValueError("incumbent calibration frame is empty")
    if not frame.schema["seconds_elapsed"].is_integer():
        raise TypeError("incumbent calibration seconds_elapsed must be integral")
    if "label_up" in frame.columns and not frame.schema["label_up"].is_integer():
        raise TypeError("incumbent calibration label_up must be integral")
    elapsed = frame["seconds_elapsed"].to_numpy().astype(np.int64)
    policy = model.payload["prediction_policy"]
    early = elapsed <= int(policy["early_end_second"])
    valid = (
        (elapsed >= int(policy["minimum_seconds_after_open"]))
        & (elapsed <= int(policy["maximum_seconds_after_open"]))
        & np.where(
            early,
            (elapsed - int(policy["minimum_seconds_after_open"]))
            % int(policy["early_cadence_seconds"])
            == 0,
            (elapsed - int(policy["early_end_second"]) - 1) % int(policy["cadence_seconds"]) == 0,
        )
    )
    if not valid.all():
        raise ValueError("incumbent calibration frame violates the runtime prediction schedule")


def _validate_payload_parent_parity(
    incumbent_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> None:
    incumbent = copy.deepcopy(incumbent_payload)
    candidate = copy.deepcopy(candidate_payload)
    incumbent.pop("model_key")
    candidate.pop("model_key")
    incumbent_calibration = incumbent.pop("asymmetric_value_calibration")
    candidate_calibration = candidate.pop("asymmetric_value_calibration")
    if incumbent != candidate:
        raise RuntimeError("calibration payload changed frozen estimator/runtime behavior")
    if incumbent_calibration["time_bands"] != candidate_calibration["time_bands"]:
        raise RuntimeError("calibration payload changed frozen parent time calibrators")


def _is_target_runtime_cell(
    cell: dict[str, Any],
    config: IncumbentCalibrationConfig,
) -> bool:
    minimum, maximum = config.target_price_band
    return (
        (int(cell["start_seconds"]), int(cell["end_seconds_exclusive"])) in config.target_time_bands
        and math.isclose(float(cell["minimum_price"]), minimum, abs_tol=1e-12)
        and math.isclose(float(cell["maximum_price"]), maximum, abs_tol=1e-12)
        and str(cell["side"]).upper() in config.sides
    )


def _runtime_cell_key(cell: dict[str, Any]) -> tuple[int, int, float, float, str]:
    return (
        int(cell["start_seconds"]),
        int(cell["end_seconds_exclusive"]),
        round(float(cell["minimum_price"]), 12),
        round(float(cell["maximum_price"]), 12),
        str(cell["side"]),
    )


def _optimizer_options(config: IncumbentCalibrationConfig) -> dict[str, Any]:
    return {
        "maxiter": config.optimizer_max_iterations,
        "ftol": config.optimizer_ftol,
        "gtol": config.optimizer_gtol,
        "maxls": config.optimizer_max_line_search_steps,
    }


def _validate_conditional_estimator(contract: ConditionalEstimatorContract) -> None:
    if (
        contract.enabled is not True
        or contract.trigger != "no_supported_calibration_challenger_passed_probability_gates"
        or contract.maximum_total_challengers != 5
        or set(contract.forbidden_triggers)
        != {"support_failure", "technical_failure", "economic_failure", "forward_failure"}
        or contract.histogram
        != ConditionalHistogramContract(
            learning_rate=0.02,
            max_iter=240,
            max_leaf_nodes=7,
            min_samples_leaf=300,
            l2_regularization=10.0,
        )
        or tuple(candidate.name for candidate in contract.candidates)
        != ("E1_hybrid50_h3", "E2_hybrid50_h3_boundary_weighted")
    ):
        raise ValueError("conditional estimator contingency changed")
    e1, e2 = contract.candidates
    if e1.target_weight != 0.5 or e1.boundary_weighted:
        raise ValueError("E1 conditional estimator contract changed")
    if (
        e2.target_weight != 0.5
        or not e2.boundary_weighted
        or e2.boundary_minimum_edge != 0.02
        or e2.boundary_maximum_edge != 0.04
        or e2.boundary_closed != "both"
        or e2.boundary_multiplier != 2.0
        or e2.weighting != "market_equal_renormalized"
    ):
        raise ValueError("E2 conditional estimator boundary contract changed")


def _validate_source_asymmetric_value_config(config: IncumbentCalibrationConfig) -> None:
    """Validate only the Core+Oracle/PM-book source contract; L2 is not required here."""

    with config.asymmetric_value_config.open("rb") as handle:
        raw = tomllib.load(handle)

    def source_window(name: str) -> EvidenceWindow:
        values = raw["windows"][name]
        return EvidenceWindow(
            start=parse_utc_day(values["start"]),
            end=parse_utc_day(values["end"]),
        )

    target = raw["model"]["target_calibration"]
    target_bands = tuple(
        (int(item["start_second"]), int(item["end_second_exclusive"]))
        for item in target["time_bands"]
    )
    if (
        source_window("calibration") != config.calibration_fit
        or source_window("policy") != config.matched_comparison
        or float(target["minimum_price"]) != config.target_price_band[0]
        or float(target["maximum_price"]) != config.target_price_band[1]
        or tuple(str(side) for side in target["sides"]) != config.sides
        or target_bands != config.target_time_bands
        or int(target["required_fitted_cells"]) != config.target_cells
    ):
        raise ValueError("source asymmetric-value cache contract does not match Gen2")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), *_PROBABILITY_CLIP)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-clipped))
