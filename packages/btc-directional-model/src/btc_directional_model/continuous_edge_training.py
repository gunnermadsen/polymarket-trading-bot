"""Offline, time-conditioned BTC five-minute continuous-edge training.

This module is deliberately isolated from runtime inference and trading-process
configuration.  It consumes immutable feature caches and completed PMXT capacity
artifacts, selects policy parameters on chronological validation evidence, and
opens the final test block exactly once.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import tomllib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .core_extract import (
    configure_read_only_connection,
    database_connection,
    file_sha256,
)

SCHEMA_VERSION = "btc-continuous-edge-payoff-training-v2"
MODEL_SCHEMA_VERSION = "btc-continuous-edge-payoff-development-artifact-v2"
VWAP_QUANTITIES = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200)

CORE_FEATURES = (
    "seconds_elapsed_scaled",
    "seconds_remaining_scaled",
    "btc_path_from_window_open_bps",
    "btc_cross_venue_boundary_gap_bps",
    "btc_window_open_cross_venue_basis_bps",
    "btc_return_1s_bps",
    "btc_return_5s_bps",
    "btc_return_15s_bps",
    "btc_return_30s_bps",
    "btc_return_60s_bps",
    "btc_return_90s_bps",
    "btc_return_120s_bps",
    "btc_return_180s_bps",
    "btc_realized_volatility_5s_bps",
    "btc_realized_volatility_15s_bps",
    "btc_realized_volatility_30s_bps",
    "btc_realized_volatility_60s_bps",
    "btc_realized_volatility_90s_bps",
    "btc_realized_volatility_120s_bps",
    "btc_realized_volatility_180s_bps",
    "btc_range_5s_bps",
    "btc_range_30s_bps",
    "btc_range_60s_bps",
    "btc_path_efficiency_30s",
    "btc_path_efficiency_60s",
    "btc_range_position_30s",
    "btc_range_position_60s",
    "btc_signed_flow_5s",
    "btc_signed_flow_30s",
    "btc_signed_flow_60s",
    "btc_signed_flow_90s",
    "btc_signed_flow_120s",
    "btc_signed_flow_180s",
    "btc_path_cross_count",
    "btc_boundary_cross_count",
    "btc_seconds_since_path_cross",
    "btc_seconds_since_boundary_cross",
    "btc_path_terminal_volatility_z",
    "btc_boundary_terminal_volatility_z",
    "btc_boundary_distance_velocity_5s_bps",
    "btc_boundary_momentum_alignment_5s",
    "btc_momentum_multihorizon_score",
    "btc_momentum_acceleration_5_vs_30",
    "btc_momentum_acceleration_15_vs_60",
    "btc_reversal_5_vs_30",
    "btc_path_max_favorable_excursion_bps",
    "btc_path_max_adverse_excursion_bps",
    "btc_path_pullback_from_favorable_extreme_bps",
    "btc_path_recovery_from_adverse_extreme_bps",
    "btc_seconds_since_path_high_scaled",
    "btc_seconds_since_path_low_scaled",
    "btc_volatility_shock_30_vs_120",
    "btc_volatility_shock_60_vs_180",
    "btc_path_sign_normalized_return_5s_bps",
    "btc_path_sign_normalized_return_30s_bps",
    "btc_path_sign_normalized_return_60s_bps",
    "btc_path_sign_normalized_return_90s_bps",
    "btc_path_sign_normalized_return_120s_bps",
    "btc_path_sign_normalized_flow_5s",
    "btc_path_sign_normalized_flow_30s",
    "btc_path_sign_normalized_flow_60s",
    "btc_path_sign_normalized_flow_90s",
    "btc_path_sign_normalized_flow_120s",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)

ORACLE_FEATURES = (
    "oracle_return_from_window_open_bps",
    "oracle_round_age_seconds_scaled",
    "oracle_update_count_since_open_scaled",
    "binance_oracle_basis_bps",
    "early_oracle_eligible",
)

CHAINLINK_FEATURES = (
    "chainlink_candle_return_5m_bps",
    "chainlink_candle_return_15m_bps",
    "chainlink_candle_return_30m_bps",
    "chainlink_candle_return_60m_bps",
    "chainlink_candle_realized_volatility_15m_bps",
    "chainlink_candle_realized_volatility_60m_bps",
    "chainlink_candle_range_15m_bps",
    "chainlink_candle_range_60m_bps",
)

L2_FEATURES = (
    "spot_l2_midpoint_to_kline_close_bps",
    "spot_l2_microprice_to_midpoint_bps",
    "spot_l2_spread_bps",
    "spot_l2_imbalance_5",
    "spot_l2_imbalance_20",
    "spot_l2_bid_depth_20_log",
    "spot_l2_ask_depth_20_log",
    "spot_l2_bid_depth_slope_20",
    "spot_l2_ask_depth_slope_20",
    "spot_l2_midpoint_change_5s_bps",
    "spot_l2_imbalance_20_change_5s",
    "spot_l2_midpoint_change_30s_bps",
    "spot_l2_imbalance_20_change_30s",
)

BOOK_RAW_FEATURES = tuple(
    f"{side}_ask_vwap_{quantity}"
    for side in ("up", "down")
    for quantity in VWAP_QUANTITIES
)
BOOK_DERIVED_FEATURES = (
    "pm_vwap5_overround",
    "pm_vwap50_overround",
    "pm_vwap200_overround",
    "pm_up_slope_5_25",
    "pm_up_slope_25_100",
    "pm_up_slope_100_200",
    "pm_down_slope_5_25",
    "pm_down_slope_25_100",
    "pm_down_slope_100_200",
    "pm_up_depth_log",
    "pm_down_depth_log",
    "pm_depth_imbalance",
    "pm_up_book_age_seconds",
    "pm_down_book_age_seconds",
)
PRIMARY_FEATURES = tuple(dict.fromkeys((*CORE_FEATURES, *ORACLE_FEATURES, *BOOK_RAW_FEATURES, *BOOK_DERIVED_FEATURES)))

ADMISSION_FEATURES = (
    "probability_selected",
    "confidence_margin",
    "selected_edge_5",
    "selected_cost_5",
    "pm_vwap5_overround",
    "pm_vwap200_overround",
    "pm_depth_imbalance",
    "pm_up_book_age_seconds",
    "pm_down_book_age_seconds",
    "seconds_elapsed_scaled",
    "btc_path_from_window_open_bps",
    "btc_cross_venue_boundary_gap_bps",
    "btc_path_terminal_volatility_z",
    "btc_boundary_terminal_volatility_z",
    "btc_reversal_5_vs_30",
    "btc_volatility_shock_30_vs_120",
    "btc_signed_flow_30s",
    "btc_signed_flow_60s",
    "oracle_return_from_window_open_bps",
    "binance_oracle_basis_bps",
    "directional_family_probability_up",
    "asymmetric_family_probability_up",
    "directional_family_disagreement",
    "asymmetric_family_disagreement",
    "family_probability_spread",
    "family_vote_agreement",
    "probability_change_5s",
    "probability_change_15s",
    "probability_change_30s",
    "probability_instability_30s",
    "conservative_probability_selected",
    "conservative_edge_5",
    "price_bucket_index",
)

MODEL_PROFILES = (
    {
        "name": "smooth_15",
        "learning_rate": 0.05,
        "max_iter": 180,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 100,
        "l2_regularization": 3.0,
    },
    {
        "name": "balanced_31",
        "learning_rate": 0.04,
        "max_iter": 240,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 140,
        "l2_regularization": 4.0,
    },
)


@dataclass(frozen=True)
class WindowConfig:
    outcome_fit_start: datetime
    outcome_fit_end: datetime
    calibration_end: datetime
    admission_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_end: datetime


@dataclass(frozen=True)
class TimeBand:
    name: str
    start_second: int
    end_second_exclusive: int
    minimum_validation_accuracy: float
    minimum_validation_trades: int


@dataclass(frozen=True)
class ExecutionConfig:
    quantities: tuple[int, ...]
    freshness_seconds: int
    maximum_depth_participation: float
    execution_reserve_per_share: float
    stress_slippage_per_share: float


@dataclass(frozen=True)
class PolicyConfig:
    confidence_thresholds: tuple[float, ...]
    edge_thresholds: tuple[float, ...]
    admission_thresholds: tuple[float, ...]
    payoff_lower_bound_thresholds: tuple[float, ...]
    price_bucket_edges: tuple[float, ...]
    price_bucket_minimum_edges: tuple[float, ...]
    rolling_fold_days: int
    minimum_profitable_fold_ratio: float
    minimum_profit_factor: float
    minimum_stress_expectancy_per_trade: float
    minimum_payoff_ratio: float
    minimum_active_days: int
    maximum_daily_pnl_concentration: float
    conservative_z_score: float
    payoff_lower_bound_quantile: float


@dataclass(frozen=True)
class PathConfig:
    oracle_features: Path
    chainlink_features: Path
    l2_features: Path
    directional_family_model: Path
    asymmetric_family_model: Path
    capacity_evidence: Path
    runs: Path
    committed_results: Path


@dataclass(frozen=True)
class TrainingConfig:
    source_path: Path
    package_root: Path
    profile: str
    random_seed: int
    fresh_holdout: bool
    windows: WindowConfig
    bands: tuple[TimeBand, ...]
    execution: ExecutionConfig
    policy: PolicyConfig
    paths: PathConfig


@dataclass
class Expert:
    band: TimeBand
    feature_names: tuple[str, ...]
    profile: dict[str, Any]
    estimator: HistGradientBoostingClassifier
    calibrator: LogisticRegression

    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        raw = np.clip(self.estimator.predict_proba(_matrix(frame, self.feature_names))[:, 1], 1e-7, 1 - 1e-7)
        logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
        return self.calibrator.predict_proba(logits)[:, 1]


@dataclass
class PayoffAdmissionModel:
    feature_names: tuple[str, ...]
    profitable_classifier: HistGradientBoostingClassifier
    stress_edge_regressor: HistGradientBoostingRegressor
    lower_bound_penalties: dict[str, float]
    global_lower_bound_penalty: float
    oof_diagnostics: dict[str, Any]
    stress_slippage_per_share: float


@dataclass
class PriceTimeCalibration:
    estimator: LogisticRegression
    band_names: tuple[str, ...]
    price_bucket_edges: tuple[float, ...]


@dataclass
class CalibrationGuard:
    penalties: dict[str, float]
    global_penalty: float


CAPACITY_SCHEMA = pa.schema(
    [
        pa.field("market_id", pa.string(), nullable=False),
        pa.field("window_start", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("window_end", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("label_up", pa.int8(), nullable=False),
        pa.field("fee_rate", pa.float64(), nullable=False),
        pa.field("observed_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("seconds_elapsed", pa.int32(), nullable=False),
        pa.field("artifact_id", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("up_provider_received_at", pa.timestamp("us", tz="UTC")),
        pa.field("up_best_ask", pa.float64()),
        pa.field("up_ask_depth", pa.float64()),
        *(pa.field(f"up_ask_vwap_{q}", pa.float64()) for q in VWAP_QUANTITIES),
        pa.field("down_provider_received_at", pa.timestamp("us", tz="UTC")),
        pa.field("down_best_ask", pa.float64()),
        pa.field("down_ask_depth", pa.float64()),
        *(pa.field(f"down_ask_vwap_{q}", pa.float64()) for q in VWAP_QUANTITIES),
        pa.field("quality_flags", pa.int32(), nullable=False),
    ]
)


def load_config(path: Path) -> TrainingConfig:
    source = path.resolve()
    package_root = source.parent.parent
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    training = raw["training"]
    if training.get("paper_only") is not True or training.get("live_capital_allowed") is not False:
        raise ValueError("continuous-edge training must remain offline and paper-only")
    windows = raw["windows"]
    config = TrainingConfig(
        source_path=source,
        package_root=package_root,
        profile=str(training["profile"]),
        random_seed=int(training["random_seed"]),
        fresh_holdout=bool(training.get("fresh_holdout", False)),
        windows=WindowConfig(**{name: _utc(value) for name, value in windows.items()}),
        bands=tuple(TimeBand(**values) for values in raw["time_bands"]),
        execution=ExecutionConfig(
            quantities=tuple(int(q) for q in raw["execution"]["quantities"]),
            freshness_seconds=int(raw["execution"]["freshness_seconds"]),
            maximum_depth_participation=float(raw["execution"]["maximum_depth_participation"]),
            execution_reserve_per_share=float(raw["execution"]["execution_reserve_per_share"]),
            stress_slippage_per_share=float(raw["execution"]["stress_slippage_per_share"]),
        ),
        policy=PolicyConfig(
            confidence_thresholds=tuple(float(v) for v in raw["policy"]["confidence_thresholds"]),
            edge_thresholds=tuple(float(v) for v in raw["policy"]["edge_thresholds"]),
            admission_thresholds=tuple(float(v) for v in raw["policy"]["admission_thresholds"]),
            payoff_lower_bound_thresholds=tuple(
                float(v)
                for v in raw["policy"].get(
                    "payoff_lower_bound_thresholds",
                    raw["policy"].get("payoff_edge_thresholds", ()),
                )
            ),
            price_bucket_edges=tuple(float(v) for v in raw["policy"]["price_bucket_edges"]),
            price_bucket_minimum_edges=tuple(
                float(v) for v in raw["policy"]["price_bucket_minimum_edges"]
            ),
            rolling_fold_days=int(raw["policy"]["rolling_fold_days"]),
            minimum_profitable_fold_ratio=float(
                raw["policy"]["minimum_profitable_fold_ratio"]
            ),
            minimum_profit_factor=float(raw["policy"]["minimum_profit_factor"]),
            minimum_stress_expectancy_per_trade=float(
                raw["policy"].get("minimum_stress_expectancy_per_trade", 0.0)
            ),
            minimum_payoff_ratio=float(raw["policy"].get("minimum_payoff_ratio", 0.0)),
            minimum_active_days=int(raw["policy"].get("minimum_active_days", 1)),
            maximum_daily_pnl_concentration=float(
                raw["policy"].get("maximum_daily_pnl_concentration", 1.0)
            ),
            conservative_z_score=float(raw["policy"]["conservative_z_score"]),
            payoff_lower_bound_quantile=float(
                raw["policy"].get("payoff_lower_bound_quantile", 0.20)
            ),
        ),
        paths=PathConfig(**{name: _path(package_root, value) for name, value in raw["paths"].items()}),
    )
    _validate_config(config)
    return config


def _validate_config(config: TrainingConfig) -> None:
    window_values = list(asdict(config.windows).values())
    if window_values != sorted(window_values) or len(set(window_values)) != len(window_values):
        raise ValueError("training windows must be strictly chronological")
    if config.execution.quantities != VWAP_QUANTITIES:
        raise ValueError("continuous-edge training requires the exact VWAP 5-200 contract")
    if config.execution.maximum_depth_participation != 0.25:
        raise ValueError("maximum depth participation must remain 25 percent")
    if len(config.policy.price_bucket_edges) != len(config.policy.price_bucket_minimum_edges) + 1:
        raise ValueError("price bucket edges and minimum edges do not align")
    if config.policy.price_bucket_edges != tuple(sorted(config.policy.price_bucket_edges)):
        raise ValueError("price bucket edges must be sorted")
    if not 0 < config.policy.minimum_profitable_fold_ratio <= 1:
        raise ValueError("profitable fold ratio must be in (0, 1]")
    if not 0 < config.policy.payoff_lower_bound_quantile < 0.5:
        raise ValueError("payoff lower-bound quantile must be in (0, 0.5)")
    if not 0 < config.policy.maximum_daily_pnl_concentration <= 1:
        raise ValueError("maximum daily PnL concentration must be in (0, 1]")
    if tuple((band.start_second, band.end_second_exclusive) for band in config.bands) != (
        (15, 90),
        (90, 180),
        (180, 241),
    ):
        raise ValueError("time bands must cover the exact 15-240 second policy range")
    for path in (
        config.paths.oracle_features,
        config.paths.chainlink_features,
        config.paths.l2_features,
        config.paths.directional_family_model,
        config.paths.asymmetric_family_model,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def run_training(config: TrainingConfig) -> tuple[Path, dict[str, Any]]:
    evidence_manifest = extract_capacity_evidence(config)
    print("load: capacity evidence", flush=True)
    frame = load_training_frame(config, evidence_manifest)
    coverage = coverage_summary(frame, config)
    print(
        "joined: "
        f"{frame.height:,} strict rows, {frame['market_id'].n_unique():,} markets",
        flush=True,
    )

    candidate_specs = {
        "core_oracle_vwap_curve": PRIMARY_FEATURES,
        "core_oracle_vwap_curve_chainlink": (*PRIMARY_FEATURES, *CHAINLINK_FEATURES),
        "core_oracle_vwap_curve_l2": (*PRIMARY_FEATURES, *L2_FEATURES),
    }
    candidates: dict[str, dict[str, Expert]] = {}
    candidate_metrics: dict[str, Any] = {}
    for candidate_name, features in candidate_specs.items():
        print(f"train: {candidate_name}", flush=True)
        experts, metrics = fit_candidate(config, frame, candidate_name, tuple(features))
        candidates[candidate_name] = experts
        candidate_metrics[candidate_name] = metrics

    selected_candidate, challenger_decision = choose_candidate(
        config,
        frame,
        candidates,
        candidate_metrics,
    )
    selected_experts = candidates[selected_candidate]
    print(f"selected probability candidate: {selected_candidate}", flush=True)

    print("train: chronological directional/asymmetric family proxies", flush=True)
    family_proxies, family_proxy_metrics = fit_family_proxies(config, frame)
    calibration_raw = score_frame(
        _block(frame, config.windows.outcome_fit_end, config.windows.calibration_end),
        selected_experts,
        family_proxies,
        config,
    )
    price_time_calibration = fit_price_time_calibration(calibration_raw, config)
    calibrated_calibration = attach_price_time_calibration(
        calibration_raw,
        price_time_calibration,
        config,
    )
    calibration_guard = fit_calibration_guard(calibrated_calibration, config)
    scored_admission = prepare_scored_frame(
        _block(frame, config.windows.calibration_end, config.windows.admission_end),
        selected_experts,
        family_proxies,
        price_time_calibration,
        calibration_guard,
        config,
    )
    admission_model = fit_admission_model(scored_admission, config)
    validation = attach_admission_probability(
        prepare_scored_frame(
            _block(frame, config.windows.validation_start, config.windows.validation_end),
            selected_experts,
            family_proxies,
            price_time_calibration,
            calibration_guard,
            config,
        ),
        admission_model,
    )
    thresholds, threshold_search = select_policy_thresholds(validation, config)
    validation_control = select_first_crossings(validation, thresholds, use_admission=False)
    validation_selected = select_first_crossings(validation, thresholds, use_admission=True)

    print(
        "evaluation: opening frozen chronological holdout "
        f"{config.windows.validation_end.isoformat()} to {config.windows.test_end.isoformat()}",
        flush=True,
    )
    test_scored = attach_admission_probability(
        prepare_scored_frame(
            _block(frame, config.windows.validation_end, config.windows.test_end),
            selected_experts,
            family_proxies,
            price_time_calibration,
            calibration_guard,
            config,
        ),
        admission_model,
    )
    test_control = select_first_crossings(test_scored, thresholds, use_admission=False)
    test_selected = select_first_crossings(test_scored, thresholds, use_admission=True)

    validation_metrics = {
        "without_admission_veto": policy_metrics(validation_control, config, quantity=5),
        "with_admission_veto": policy_metrics(validation_selected, config, quantity=5),
    }
    test_metrics = {
        "without_admission_veto": policy_metrics(test_control, config, quantity=5),
        "with_admission_veto": policy_metrics(test_selected, config, quantity=5),
    }
    capacity_curve = {
        str(quantity): policy_metrics(test_selected, config, quantity=quantity)
        for quantity in config.execution.quantities
    }
    band_metrics = {
        band.name: policy_metrics(
            test_selected.filter(pl.col("time_band") == band.name),
            config,
            quantity=5,
        )
        for band in config.bands
    }
    validation_folds = rolling_policy_metrics(
        validation_selected,
        config,
        config.windows.validation_start,
        config.windows.validation_end,
        quantity=5,
    )
    test_folds = rolling_policy_metrics(
        test_selected,
        config,
        config.windows.validation_end,
        config.windows.test_end,
        quantity=5,
    )
    price_bucket_metrics = policy_metrics_by_price_bucket(test_selected, config, quantity=5)
    test_market_count = test_scored["market_id"].n_unique()
    scheduled_test_market_count = int(
        pl.scan_parquet(config.paths.capacity_evidence / "test.parquet")
        .select(pl.col("market_id").n_unique())
        .collect()
        .item()
    )
    test_metrics["with_admission_veto"]["market_coverage"] = (
        test_selected["market_id"].n_unique() / test_market_count if test_market_count else 0.0
    )
    test_metrics["with_admission_veto"]["strict_data_coverage"] = (
        test_market_count / scheduled_test_market_count if scheduled_test_market_count else 0.0
    )
    test_metrics["with_admission_veto"]["end_to_end_market_coverage"] = (
        test_selected["market_id"].n_unique() / scheduled_test_market_count
        if scheduled_test_market_count
        else 0.0
    )
    test_metrics["with_admission_veto"]["scheduled_markets"] = scheduled_test_market_count
    test_metrics["with_admission_veto"]["strict_markets"] = test_market_count
    qualification_checks = {
        "positive_test_net_pnl": test_metrics["with_admission_veto"]["net_pnl"] > 0,
        "positive_test_stress_expectancy": (
            test_metrics["with_admission_veto"]["stress_expectancy_per_trade"] > 0
        ),
        "test_profit_factor_at_least_target": (
            (test_metrics["with_admission_veto"]["profit_factor"] or 0.0)
            >= config.policy.minimum_profit_factor
        ),
        "payoff_admission_improves_test_net_pnl": (
            test_metrics["with_admission_veto"]["net_pnl"]
            > test_metrics["without_admission_veto"]["net_pnl"]
        ),
        "early_test_accuracy_floor": (
            band_metrics["early"]["accuracy"]
            >= next(band.minimum_validation_accuracy for band in config.bands if band.name == "early")
        ),
        "early_test_positive_stress_expectancy": (
            band_metrics["early"]["stress_expectancy_per_trade"] > 0
        ),
        "positive_test_q10_expectancy": policy_metrics(
            test_selected, config, quantity=10
        )["expectancy_per_trade"] > 0,
        "positive_test_q20_expectancy": policy_metrics(
            test_selected, config, quantity=20
        )["expectancy_per_trade"] > 0,
        "profitable_test_fold_ratio": (
            test_folds["profitable_fold_ratio"]
            >= config.policy.minimum_profitable_fold_ratio
        ),
        "no_enabled_negative_price_bucket": all(
            row["expectancy_per_trade"] >= 0
            for row in price_bucket_metrics.values()
            if row["trades"] > 0
        ),
        "test_payoff_ratio_at_least_target": (
            test_metrics["with_admission_veto"]["payoff_ratio"]
            >= config.policy.minimum_payoff_ratio
        ),
        "test_daily_pnl_concentration_within_limit": (
            test_metrics["with_admission_veto"]["daily_pnl_concentration"]
            <= config.policy.maximum_daily_pnl_concentration
        ),
        "test_trades_at_least_50": test_metrics["with_admission_veto"]["trades"] >= 50,
        "fresh_untouched_holdout_available": config.fresh_holdout,
    }
    qualification = {
        "passed": all(qualification_checks.values()),
        "checks": qualification_checks,
        "decision": "qualified" if all(qualification_checks.values()) else "development_only",
    }

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    temporary = config.paths.runs / f"{run_id}.partial"
    final = config.paths.committed_results / run_id
    if temporary.exists() or final.exists():
        raise FileExistsError(run_id)
    temporary.mkdir(parents=True)
    artifact = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "profile": config.profile,
        "selected_candidate": selected_candidate,
        "feature_names": list(candidate_specs[selected_candidate]),
        "bands": [asdict(band) for band in config.bands],
        "experts": {
            name: {
                "band": asdict(expert.band),
                "feature_names": list(expert.feature_names),
                "profile": expert.profile,
                "estimator": expert.estimator,
                "calibrator": expert.calibrator,
            }
            for name, expert in selected_experts.items()
        },
        "family_proxies": {
            name: {
                "feature_names": list(expert.feature_names),
                "profile": expert.profile,
                "estimator": expert.estimator,
                "calibrator": expert.calibrator,
            }
            for name, expert in family_proxies.items()
        },
        "price_time_calibration": {
            "estimator": price_time_calibration.estimator,
            "band_names": list(price_time_calibration.band_names),
            "price_bucket_edges": list(price_time_calibration.price_bucket_edges),
        },
        "calibration_guard": {
            "penalties": calibration_guard.penalties,
            "global_penalty": calibration_guard.global_penalty,
        },
        "admission_model": {
            "feature_names": list(admission_model.feature_names),
            "profitable_classifier": admission_model.profitable_classifier,
            "stress_edge_regressor": admission_model.stress_edge_regressor,
            "lower_bound_penalties": admission_model.lower_bound_penalties,
            "global_lower_bound_penalty": admission_model.global_lower_bound_penalty,
            "stress_slippage_per_share": admission_model.stress_slippage_per_share,
        },
        "thresholds": thresholds,
        "execution": asdict(config.execution),
        "runtime_exported": False,
        "production_qualified": False,
    }
    joblib_path = temporary / "model.joblib"
    joblib.dump(artifact, joblib_path, compress=3)
    test_selected.select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "time_band",
        "label_up",
        "predicted_up",
        "probability_up",
        "probability_selected",
        "conservative_probability_selected",
        "calibration_uncertainty_penalty",
        "admission_probability",
        "payoff_expected_stress_edge",
        "payoff_lower_bound_penalty",
        "payoff_stress_edge_lower_bound",
        "payoff_loss_probability",
        "payoff_conditional_loss",
        "payoff_expected_shortfall",
        "selected_edge_5",
        "conservative_edge_5",
        "selected_cost_5",
        "price_bucket_index",
        "directional_family_probability_up",
        "asymmetric_family_probability_up",
        *BOOK_RAW_FEATURES,
        "fee_rate",
    ).write_parquet(temporary / "test-trades.parquet", compression="zstd")

    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": config.profile,
        "paper_only": True,
        "runtime_exported": False,
        "trading_processes_changed": False,
        "production_qualified": False,
        "source_commit": _git_revision(config.package_root),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "configuration": {
            "path": str(config.source_path),
            "sha256": file_sha256(config.source_path),
            "windows": {name: value.isoformat() for name, value in asdict(config.windows).items()},
            "bands": [asdict(band) for band in config.bands],
            "execution": asdict(config.execution),
            "policy": asdict(config.policy),
            "fresh_holdout": config.fresh_holdout,
        },
        "data": {
            "oracle_features": _source_identity(config.paths.oracle_features),
            "chainlink_features": _source_identity(config.paths.chainlink_features),
            "l2_features": _source_identity(config.paths.l2_features),
            "directional_family_contract": _source_identity(
                config.paths.directional_family_model
            ),
            "asymmetric_family_contract": _source_identity(
                config.paths.asymmetric_family_model
            ),
            "capacity_manifest": evidence_manifest,
            "coverage": coverage,
            "twap_challenger": {
                "evaluated": False,
                "reason": "No persisted, settlement-aligned historical TWAP source was available.",
            },
        },
        "candidate_metrics": candidate_metrics,
        "challenger_decision": challenger_decision,
        "selected_candidate": selected_candidate,
        "family_proxy_metrics": family_proxy_metrics,
        "calibration": {
            "type": "regularized_time_side_price_logistic_with_lower_bound_guard",
            "global_uncertainty_penalty": calibration_guard.global_penalty,
            "cell_penalties": calibration_guard.penalties,
        },
        "payoff_lower_bound": {
            **admission_model.oof_diagnostics,
            "cell_penalties": admission_model.lower_bound_penalties,
        },
        "threshold_search": threshold_search,
        "selected_thresholds": thresholds,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_by_time_band": band_metrics,
        "validation_rolling_folds": validation_folds,
        "test_rolling_folds": test_folds,
        "test_by_price_bucket": price_bucket_metrics,
        "test_fixed_entry_capacity_curve": capacity_curve,
        "qualification": qualification,
        "model_artifact": {
            "path": "model.joblib",
            "sha256": file_sha256(joblib_path),
        },
        "limitations": [
            "Capacity evidence ends at second 240; seconds 241-299 are not evaluated.",
            "TWAP was excluded because no persisted, settlement-aligned historical TWAP source was available.",
            "Open interest begins after the outcome-fit window and is diagnostic only.",
            "Aggregate trade prints overlap only the end of the test block and are diagnostic only.",
            "Directional and asymmetric family signals are chronological proxy refits using the frozen family feature contracts; they are not replays of future-trained runtime artifacts.",
            "Projected PnL assumes the recorded VWAP is fillable at the sampled timestamp and does not model queue position.",
        ],
    }
    _write_json(temporary / "metrics.json", metrics)
    (temporary / "report.md").write_text(render_report(metrics))
    (temporary / "model.sha256").write_text(file_sha256(joblib_path) + "\n")
    config.paths.committed_results.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    return final, metrics


def extract_capacity_evidence(config: TrainingConfig) -> dict[str, Any]:
    destination = config.paths.capacity_evidence
    destination.mkdir(parents=True, exist_ok=True)
    query_path = config.package_root / "sql" / "btc-continuous-edge-capacity-evidence.sql"
    query = query_path.read_text()
    intervals = _evidence_intervals(config)
    contract = {
        "schema_version": "btc-continuous-edge-capacity-evidence-v1",
        "query_sha256": file_sha256(query_path),
        "intervals": [
            {"name": name, "start": start.isoformat(), "end": end.isoformat()}
            for name, start, end in intervals
        ],
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if any(manifest.get(key) != value for key, value in contract.items()):
            raise RuntimeError("capacity evidence cache contract changed")
        for part in manifest["partitions"]:
            path = destination / part["path"]
            if not path.is_file() or file_sha256(path) != part["sha256"]:
                raise RuntimeError(f"capacity evidence partition changed: {part['path']}")
        return manifest

    connection = database_connection()
    configure_read_only_connection(connection)
    partitions: list[dict[str, Any]] = []
    try:
        for name, start, end in intervals:
            _require_complete_hours(connection, start, end)
            path = destination / f"{name}.parquet"
            rows = _extract_capacity_partition(connection, query, path, start, end)
            partitions.append({"path": path.name, "rows": rows, "sha256": file_sha256(path)})
            print(f"extract: {name} {rows:,} rows", flush=True)
    finally:
        connection.close()
    manifest = {
        **contract,
        "created_at": datetime.now(UTC).isoformat(),
        "completed_artifacts_only": True,
        "source_table": "polymarket.btc_market_capacity_execution_snapshots",
        "partitions": partitions,
        "rows": sum(part["rows"] for part in partitions),
    }
    _write_json(manifest_path, manifest)
    return manifest


def _evidence_intervals(config: TrainingConfig) -> tuple[tuple[str, datetime, datetime], ...]:
    w = config.windows
    return (
        ("outcome-fit", w.outcome_fit_start, w.outcome_fit_end),
        ("calibration", w.outcome_fit_end, w.calibration_end),
        ("admission", w.calibration_end, w.admission_end),
        ("validation", w.validation_start, w.validation_end),
        ("test", w.validation_end, w.test_end),
    )


def _require_complete_hours(connection: Any, start: datetime, end: datetime) -> None:
    row = connection.execute(
        """
        WITH expected AS MATERIALIZED (
          SELECT hour FROM generate_series(
            %(start)s::timestamptz,
            %(end)s::timestamptz - interval '1 hour',
            interval '1 hour'
          ) AS hour
        ), completed AS MATERIALIZED (
          SELECT DISTINCT date_trunc('hour', minimum_source_timestamp) AS hour
          FROM polymarket.backfill_artifacts
          WHERE provider = 'pmxt_v2_capacity_execution_snapshots_v2'
            AND status = 'completed'
            AND record_count = 1152
            AND minimum_source_timestamp >= %(start)s
            AND minimum_source_timestamp < %(end)s
        )
        SELECT count(*)::bigint, min(expected.hour)
        FROM expected LEFT JOIN completed USING (hour)
        WHERE completed.hour IS NULL
        """,
        {"start": start, "end": end},
    ).fetchone()
    missing, first_missing = row
    if missing:
        raise RuntimeError(f"capacity interval has {missing} missing hours; first={first_missing}")


def _extract_capacity_partition(
    connection: Any,
    query: str,
    destination: Path,
    start: datetime,
    end: datetime,
) -> int:
    temporary = destination.with_suffix(".parquet.partial")
    writer: pq.ParquetWriter | None = None
    count = 0
    try:
        with connection.transaction(), connection.cursor(name=f"continuous_edge_{start:%Y%m%d}") as cursor:
            cursor.execute(query, {"batch_start": start, "batch_end": end})
            while rows := cursor.fetchmany(10_000):
                table = pa.Table.from_pylist(
                    [dict(zip(CAPACITY_SCHEMA.names, row, strict=True)) for row in rows],
                    schema=CAPACITY_SCHEMA,
                )
                writer = writer or pq.ParquetWriter(temporary, CAPACITY_SCHEMA, compression="zstd")
                writer.write_table(table)
                count += len(rows)
    finally:
        if writer:
            writer.close()
    if count == 0:
        pq.write_table(pa.Table.from_pylist([], schema=CAPACITY_SCHEMA), temporary)
    temporary.replace(destination)
    return count


def load_training_frame(config: TrainingConfig, manifest: dict[str, Any]) -> pl.DataFrame:
    evidence_paths = [config.paths.capacity_evidence / part["path"] for part in manifest["partitions"]]
    evidence = pl.read_parquet(evidence_paths)
    duplicates = evidence.group_by("market_id", "observed_at").len().filter(pl.col("len") != 1)
    if duplicates.height:
        raise RuntimeError("capacity evidence contains duplicate decision points")
    clean = (pl.col("quality_flags") & 63) == 0
    fresh = (
        pl.col("up_provider_received_at").is_not_null()
        & pl.col("down_provider_received_at").is_not_null()
        & (pl.col("up_provider_received_at") <= pl.col("observed_at"))
        & (pl.col("down_provider_received_at") <= pl.col("observed_at"))
        & (pl.col("up_provider_received_at") >= pl.col("observed_at") - pl.duration(seconds=config.execution.freshness_seconds))
        & (pl.col("down_provider_received_at") >= pl.col("observed_at") - pl.duration(seconds=config.execution.freshness_seconds))
    )
    full_curve = pl.all_horizontal(
        [pl.col(column).is_not_null() & pl.col(column).is_finite() for column in BOOK_RAW_FEATURES]
    )
    strict = (
        clean
        & fresh
        & full_curve
        & (pl.col("up_ask_depth") >= 200 / config.execution.maximum_depth_participation)
        & (pl.col("down_ask_depth") >= 200 / config.execution.maximum_depth_participation)
    )
    evidence = evidence.filter(strict)

    key_columns = ("market_id", "window_start", "observed_at", "seconds_elapsed", "label_up")
    family_features = tuple(
        name for name in _family_feature_names(config) if not name.startswith("pm_")
    )
    source_columns = tuple(
        dict.fromkeys((*key_columns, *CORE_FEATURES, *ORACLE_FEATURES, *family_features))
    )
    oracle = (
        pl.scan_parquet(config.paths.oracle_features)
        .filter(_window_filter(config))
        .select(*source_columns)
        .collect()
    )
    frame = oracle.join(evidence, on=list(key_columns), how="inner", validate="1:1")
    if frame.is_empty():
        raise RuntimeError("oracle feature and capacity evidence keys do not overlap")
    frame = attach_book_features(frame)
    frame = _join_optional_features(frame, config.paths.chainlink_features, CHAINLINK_FEATURES)
    frame = _join_optional_features(frame, config.paths.l2_features, L2_FEATURES)
    frame = frame.with_columns(
        pl.when(pl.col("seconds_elapsed") < 90)
        .then(pl.lit("early"))
        .when(pl.col("seconds_elapsed") < 180)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("late"))
        .alias("time_band")
    ).sort(["window_start", "market_id", "seconds_elapsed", "observed_at"])
    return frame


def _window_filter(config: TrainingConfig) -> pl.Expr:
    return pl.any_horizontal(
        [
            (pl.col("window_start") >= start) & (pl.col("window_start") < end)
            for _, start, end in _evidence_intervals(config)
        ]
    )


def _join_optional_features(frame: pl.DataFrame, path: Path, features: tuple[str, ...]) -> pl.DataFrame:
    keys = ("market_id", "window_start", "observed_at", "seconds_elapsed", "label_up")
    optional = pl.scan_parquet(path).select(*keys, *features).collect()
    return frame.join(optional, on=list(keys), how="left", validate="1:1")


def attach_book_features(frame: pl.DataFrame) -> pl.DataFrame:
    reserve = 0.005
    enriched = frame.with_columns(
        (pl.col("up_ask_vwap_5") + pl.col("down_ask_vwap_5") - 1.0).alias("pm_vwap5_overround"),
        (pl.col("up_ask_vwap_50") + pl.col("down_ask_vwap_50") - 1.0).alias("pm_vwap50_overround"),
        (pl.col("up_ask_vwap_200") + pl.col("down_ask_vwap_200") - 1.0).alias("pm_vwap200_overround"),
        (pl.col("up_ask_vwap_25") - pl.col("up_ask_vwap_5")).alias("pm_up_slope_5_25"),
        (pl.col("up_ask_vwap_100") - pl.col("up_ask_vwap_25")).alias("pm_up_slope_25_100"),
        (pl.col("up_ask_vwap_200") - pl.col("up_ask_vwap_100")).alias("pm_up_slope_100_200"),
        (pl.col("down_ask_vwap_25") - pl.col("down_ask_vwap_5")).alias("pm_down_slope_5_25"),
        (pl.col("down_ask_vwap_100") - pl.col("down_ask_vwap_25")).alias("pm_down_slope_25_100"),
        (pl.col("down_ask_vwap_200") - pl.col("down_ask_vwap_100")).alias("pm_down_slope_100_200"),
        pl.col("up_ask_depth").log1p().alias("pm_up_depth_log"),
        pl.col("down_ask_depth").log1p().alias("pm_down_depth_log"),
        ((pl.col("up_ask_depth") - pl.col("down_ask_depth")) / (pl.col("up_ask_depth") + pl.col("down_ask_depth")).clip(1e-9)).alias("pm_depth_imbalance"),
        ((pl.col("observed_at") - pl.col("up_provider_received_at")).dt.total_milliseconds() / 1000.0).alias("pm_up_book_age_seconds"),
        ((pl.col("observed_at") - pl.col("down_provider_received_at")).dt.total_milliseconds() / 1000.0).alias("pm_down_book_age_seconds"),
    )
    enriched = enriched.with_columns(
        pl.col("up_ask_vwap_5").alias("yes_ask_vwap_5"),
        pl.col("down_ask_vwap_5").alias("no_ask_vwap_5"),
        (
            pl.col("up_ask_vwap_5")
            + pl.col("fee_rate") * pl.col("up_ask_vwap_5") * (1.0 - pl.col("up_ask_vwap_5"))
            + reserve
        ).alias("pm_yes_cost_per_share"),
        (
            pl.col("down_ask_vwap_5")
            + pl.col("fee_rate")
            * pl.col("down_ask_vwap_5")
            * (1.0 - pl.col("down_ask_vwap_5"))
            + reserve
        ).alias("pm_no_cost_per_share"),
        (pl.col("up_ask_vwap_5") - pl.col("up_best_ask")).alias("pm_yes_vwap_slippage"),
        (pl.col("down_ask_vwap_5") - pl.col("down_best_ask")).alias("pm_no_vwap_slippage"),
        pl.col("pm_up_depth_log").alias("pm_yes_depth_log"),
        pl.col("pm_down_depth_log").alias("pm_no_depth_log"),
        pl.col("pm_up_book_age_seconds").alias("pm_yes_book_age_seconds"),
        pl.col("pm_down_book_age_seconds").alias("pm_no_book_age_seconds"),
    )
    epsilon = 1e-6
    return enriched.with_columns(
        (
            pl.col("pm_yes_cost_per_share").clip(epsilon, 1.0 - epsilon).log()
            - (1.0 - pl.col("pm_yes_cost_per_share").clip(epsilon, 1.0 - epsilon)).log()
        ).alias("pm_yes_cost_logit"),
        (
            pl.col("pm_no_cost_per_share").clip(epsilon, 1.0 - epsilon).log()
            - (1.0 - pl.col("pm_no_cost_per_share").clip(epsilon, 1.0 - epsilon)).log()
        ).alias("pm_no_cost_logit"),
        (pl.col("pm_yes_cost_per_share") + pl.col("pm_no_cost_per_share") - 1.0).alias(
            "pm_cost_overround"
        ),
        (pl.col("pm_yes_cost_per_share") - pl.col("pm_no_cost_per_share")).alias(
            "pm_yes_minus_no_cost"
        ),
    )


def _family_feature_names(config: TrainingConfig) -> tuple[str, ...]:
    directional = joblib.load(config.paths.directional_family_model)
    asymmetric = joblib.load(config.paths.asymmetric_family_model)
    return tuple(
        dict.fromkeys((*directional.model.feature_names, *asymmetric.model.feature_names))
    )


def coverage_summary(frame: pl.DataFrame, config: TrainingConfig) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, start, end in _evidence_intervals(config):
        block = _block(frame, start, end)
        output[name] = {
            "strict_full_curve_rows": block.height,
            "strict_full_curve_markets": block["market_id"].n_unique(),
            "chainlink_rows": block.drop_nulls(CHAINLINK_FEATURES).height,
            "chainlink_markets": block.drop_nulls(CHAINLINK_FEATURES)["market_id"].n_unique(),
            "l2_rows": block.drop_nulls(L2_FEATURES).height,
            "l2_markets": block.drop_nulls(L2_FEATURES)["market_id"].n_unique(),
        }
    return output


def fit_candidate(
    config: TrainingConfig,
    frame: pl.DataFrame,
    candidate_name: str,
    feature_names: tuple[str, ...],
) -> tuple[dict[str, Expert], dict[str, Any]]:
    extra = tuple(name for name in feature_names if name not in PRIMARY_FEATURES)
    fit = _block(frame, config.windows.outcome_fit_start, config.windows.outcome_fit_end)
    calibration = _block(frame, config.windows.outcome_fit_end, config.windows.calibration_end)
    validation = _block(frame, config.windows.validation_start, config.windows.validation_end)
    if extra:
        fit = fit.drop_nulls(extra)
        calibration = calibration.drop_nulls(extra)
        validation = validation.drop_nulls(extra)
    experts: dict[str, Expert] = {}
    metrics: dict[str, Any] = {"feature_count": len(feature_names), "bands": {}}
    for band in config.bands:
        band_fit = fit.filter(pl.col("time_band") == band.name)
        band_cal = calibration.filter(pl.col("time_band") == band.name)
        band_validation = validation.filter(pl.col("time_band") == band.name)
        if band_fit["market_id"].n_unique() < 300 or band_cal["market_id"].n_unique() < 100:
            raise RuntimeError(f"{candidate_name}/{band.name} lacks chronological training coverage")
        expert, tuning = fit_expert(
            band_fit,
            band_cal,
            band,
            feature_names,
            config.random_seed,
        )
        experts[band.name] = expert
        cal_probability = expert.probability(band_cal)
        validation_probability = expert.probability(band_validation)
        metrics["bands"][band.name] = {
            "fit_rows": band_fit.height,
            "fit_markets": band_fit["market_id"].n_unique(),
            "calibration_rows": band_cal.height,
            "calibration_markets": band_cal["market_id"].n_unique(),
            "validation_rows": band_validation.height,
            "validation_markets": band_validation["market_id"].n_unique(),
            "tuning": tuning,
            "calibration": probability_metrics(band_cal, cal_probability),
            "validation": probability_metrics(band_validation, validation_probability),
        }
    return experts, metrics


def fit_expert(
    fit: pl.DataFrame,
    calibration: pl.DataFrame,
    band: TimeBand,
    feature_names: tuple[str, ...],
    random_seed: int,
) -> tuple[Expert, list[dict[str, Any]]]:
    feature_names = variable_feature_names(fit, feature_names)
    fit_matrix = _matrix(fit, feature_names)
    cal_matrix = _matrix(calibration, feature_names)
    fit_labels = fit["label_up"].to_numpy()
    fit_weights = market_equal_weights(fit)
    cal_weights = market_equal_weights(calibration)
    history: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], HistGradientBoostingClassifier] | None = None
    for profile in MODEL_PROFILES:
        model = HistGradientBoostingClassifier(
            learning_rate=profile["learning_rate"],
            max_iter=profile["max_iter"],
            max_leaf_nodes=profile["max_leaf_nodes"],
            min_samples_leaf=profile["min_samples_leaf"],
            l2_regularization=profile["l2_regularization"],
            random_state=random_seed,
            early_stopping=False,
        )
        model.fit(fit_matrix, fit_labels, sample_weight=fit_weights)
        probability = np.clip(model.predict_proba(cal_matrix)[:, 1], 1e-7, 1 - 1e-7)
        loss = float(log_loss(calibration["label_up"].to_numpy(), probability, sample_weight=cal_weights, labels=[0, 1]))
        record = {"profile": profile["name"], "calibration_raw_log_loss": loss}
        history.append(record)
        if best is None or loss < best[0]:
            best = (loss, profile, model)
    assert best is not None
    _, profile, model = best
    raw = np.clip(model.predict_proba(cal_matrix)[:, 1], 1e-7, 1 - 1e-7)
    raw_logit = np.log(raw / (1 - raw)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=400, random_state=random_seed)
    calibrator.fit(raw_logit, calibration["label_up"].to_numpy(), sample_weight=cal_weights)
    return Expert(band, feature_names, dict(profile), model, calibrator), history


def choose_candidate(
    config: TrainingConfig,
    frame: pl.DataFrame,
    candidates: dict[str, dict[str, Expert]],
    candidate_metrics: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    primary_name = "core_oracle_vwap_curve"
    primary = candidates[primary_name]
    validation = _block(frame, config.windows.validation_start, config.windows.validation_end)
    decisions: dict[str, Any] = {}
    qualified: list[tuple[float, str]] = []
    for name, experts in candidates.items():
        if name == primary_name:
            continue
        extras = CHAINLINK_FEATURES if "chainlink" in name else L2_FEATURES
        common = validation.drop_nulls(extras)
        band_rows: dict[str, Any] = {}
        improvements = 0
        total_rows = 0
        weighted_delta = 0.0
        for band in config.bands:
            subset = common.filter(pl.col("time_band") == band.name)
            if subset.is_empty():
                continue
            primary_metrics = probability_metrics(subset, primary[band.name].probability(subset))
            challenger_metrics = probability_metrics(subset, experts[band.name].probability(subset))
            delta = primary_metrics["log_loss"] - challenger_metrics["log_loss"]
            if delta > 0 and challenger_metrics["brier"] < primary_metrics["brier"]:
                improvements += 1
            total_rows += subset.height
            weighted_delta += delta * subset.height
            band_rows[band.name] = {
                "rows": subset.height,
                "primary": primary_metrics,
                "challenger": challenger_metrics,
                "log_loss_improvement": delta,
            }
        primary_rows = sum(
            candidate_metrics[primary_name]["bands"][band.name]["validation_rows"]
            for band in config.bands
        )
        coverage_ratio = total_rows / primary_rows if primary_rows else 0.0
        mean_delta = weighted_delta / total_rows if total_rows else -math.inf
        is_qualified = improvements >= 2 and mean_delta > 0 and coverage_ratio >= 0.50
        decisions[name] = {
            "common_subset_bands": band_rows,
            "improving_bands": improvements,
            "weighted_log_loss_improvement": mean_delta,
            "validation_row_coverage_ratio": coverage_ratio,
            "qualified": is_qualified,
        }
        if is_qualified:
            qualified.append((mean_delta, name))
    selected = max(qualified)[1] if qualified else primary_name
    return selected, {"primary": primary_name, "challengers": decisions, "selected": selected}


def fit_family_proxies(
    config: TrainingConfig,
    frame: pl.DataFrame,
) -> tuple[dict[str, Expert], dict[str, Any]]:
    fit = _block(frame, config.windows.outcome_fit_start, config.windows.outcome_fit_end)
    calibration = _block(frame, config.windows.outcome_fit_end, config.windows.calibration_end)
    contract_band = TimeBand("family_proxy", 15, 241, 0.0, 0)
    directional_contract = joblib.load(config.paths.directional_family_model)
    asymmetric_contract = joblib.load(config.paths.asymmetric_family_model)
    proxies: dict[str, Expert] = {}
    histories: dict[str, Any] = {}
    for offset, (name, contract) in enumerate(
        (
            ("directional", directional_contract.model),
            ("asymmetric", asymmetric_contract.model),
        )
    ):
        features = variable_feature_names(fit, contract.feature_names)
        proxy, history = fit_expert(
            fit,
            calibration,
            contract_band,
            features,
            config.random_seed + 100 + offset,
        )
        proxies[name] = proxy
        histories[name] = {
            "source_contract": str(
                config.paths.directional_family_model
                if name == "directional"
                else config.paths.asymmetric_family_model
            ),
            "source_contract_sha256": file_sha256(
                config.paths.directional_family_model
                if name == "directional"
                else config.paths.asymmetric_family_model
            ),
            "chronological_refit": True,
            "feature_count": len(features),
            "profile_search": history,
        }
    return proxies, histories


def score_frame(
    frame: pl.DataFrame,
    experts: dict[str, Expert],
    family_proxies: dict[str, Expert],
    config: TrainingConfig,
) -> pl.DataFrame:
    pieces = []
    for band in config.bands:
        subset = frame.filter(pl.col("time_band") == band.name)
        if subset.is_empty():
            continue
        probability = experts[band.name].probability(subset)
        pieces.append(subset.with_columns(pl.Series("probability_up", probability)))
    if not pieces:
        return frame.head(0)
    scored = pl.concat(pieces, how="vertical").sort(
        ["market_id", "seconds_elapsed", "observed_at"]
    )
    directional_family = family_proxies["directional"].probability(scored)
    asymmetric_family = family_proxies["asymmetric"].probability(scored)
    predicted_up = scored["probability_up"].to_numpy() >= 0.5
    selected_probability = np.where(predicted_up, scored["probability_up"].to_numpy(), 1 - scored["probability_up"].to_numpy())
    selected_price = np.where(predicted_up, scored["up_ask_vwap_5"].to_numpy(), scored["down_ask_vwap_5"].to_numpy())
    fee_rate = scored["fee_rate"].to_numpy()
    selected_cost = selected_price + fee_rate * selected_price * (1 - selected_price) + config.execution.execution_reserve_per_share
    return scored.with_columns(
        pl.Series("predicted_up", predicted_up),
        pl.Series("probability_selected", selected_probability),
        pl.Series("confidence_margin", selected_probability - 0.5),
        pl.Series("selected_cost_5", selected_cost),
        pl.Series("selected_edge_5", selected_probability - selected_cost),
        pl.Series("direction_correct", predicted_up == scored["label_up"].to_numpy().astype(bool)),
        pl.Series("directional_family_probability_up", directional_family),
        pl.Series("asymmetric_family_probability_up", asymmetric_family),
        pl.Series(
            "directional_family_disagreement",
            np.abs(scored["probability_up"].to_numpy() - directional_family),
        ),
        pl.Series(
            "asymmetric_family_disagreement",
            np.abs(scored["probability_up"].to_numpy() - asymmetric_family),
        ),
        pl.Series("family_probability_spread", np.abs(directional_family - asymmetric_family)),
        pl.Series(
            "family_vote_agreement",
            (
                (directional_family >= 0.5)
                == (asymmetric_family >= 0.5)
            ).astype(np.int8),
        ),
    )


def fit_price_time_calibration(
    frame: pl.DataFrame,
    config: TrainingConfig,
) -> PriceTimeCalibration:
    band_names = tuple(band.name for band in config.bands)
    matrix = _price_time_calibration_matrix(
        frame,
        band_names,
        config.policy.price_bucket_edges,
    )
    estimator = LogisticRegression(
        C=0.25,
        solver="lbfgs",
        max_iter=500,
        random_state=config.random_seed,
    )
    estimator.fit(
        matrix,
        frame["direction_correct"].to_numpy().astype(np.int8),
        sample_weight=market_equal_weights(frame),
    )
    return PriceTimeCalibration(estimator, band_names, config.policy.price_bucket_edges)


def attach_price_time_calibration(
    frame: pl.DataFrame,
    calibration: PriceTimeCalibration,
    config: TrainingConfig,
) -> pl.DataFrame:
    selected_probability = calibration.estimator.predict_proba(
        _price_time_calibration_matrix(
            frame,
            calibration.band_names,
            calibration.price_bucket_edges,
        )
    )[:, 1]
    predicted_up = frame["predicted_up"].to_numpy().astype(bool)
    probability_up = np.where(predicted_up, selected_probability, 1.0 - selected_probability)
    selected_cost = frame["selected_cost_5"].to_numpy()
    return frame.with_columns(
        pl.Series("probability_up", probability_up),
        pl.Series("probability_selected", selected_probability),
        pl.Series("confidence_margin", selected_probability - 0.5),
        pl.Series("selected_edge_5", selected_probability - selected_cost),
    )


def _price_time_calibration_matrix(
    frame: pl.DataFrame,
    band_names: tuple[str, ...],
    price_bucket_edges: tuple[float, ...],
) -> np.ndarray:
    probability = np.clip(frame["probability_selected"].to_numpy(), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1.0 - probability))
    price = frame["selected_cost_5"].to_numpy()
    buckets = _price_bucket_indices(price, price_bucket_edges)
    bands = frame["time_band"].to_numpy()
    band_one_hot = np.column_stack([(bands == name).astype(float) for name in band_names])
    bucket_one_hot = np.column_stack(
        [(buckets == index).astype(float) for index in range(len(price_bucket_edges) - 1)]
    )
    return np.column_stack(
        (
            logit,
            frame["predicted_up"].to_numpy().astype(float),
            band_one_hot,
            bucket_one_hot,
            band_one_hot * logit[:, None],
            bucket_one_hot * logit[:, None],
        )
    )


def fit_calibration_guard(frame: pl.DataFrame, config: TrainingConfig) -> CalibrationGuard:
    price = frame["selected_cost_5"].to_numpy()
    bucket = _price_bucket_indices(price, config.policy.price_bucket_edges)
    probability = frame["probability_selected"].to_numpy()
    correct = frame["direction_correct"].to_numpy().astype(float)
    weights = market_equal_weights(frame)
    global_gap = abs(float(np.average(correct - probability, weights=weights)))
    global_markets = max(frame["market_id"].n_unique(), 1)
    global_se = math.sqrt(max(float(np.average(correct, weights=weights)) * (1.0 - float(np.average(correct, weights=weights))), 0.01) / global_markets)
    global_penalty = global_gap + config.policy.conservative_z_score * global_se
    penalties: dict[str, float] = {}
    bands = frame["time_band"].to_numpy()
    for band in config.bands:
        for index in range(len(config.policy.price_bucket_edges) - 1):
            mask = (bands == band.name) & (bucket == index)
            if not mask.any():
                penalties[f"{band.name}:{index}"] = global_penalty
                continue
            cell_weights = weights[mask]
            cell_accuracy = float(np.average(correct[mask], weights=cell_weights))
            cell_gap = abs(float(np.average(correct[mask] - probability[mask], weights=cell_weights)))
            markets = frame.filter(pl.Series(mask))["market_id"].n_unique()
            cell_se = math.sqrt(max(cell_accuracy * (1.0 - cell_accuracy), 0.01) / max(markets, 1))
            cell_penalty = cell_gap + config.policy.conservative_z_score * cell_se
            shrink = markets / (markets + 200.0)
            penalties[f"{band.name}:{index}"] = shrink * cell_penalty + (1.0 - shrink) * global_penalty
    return CalibrationGuard(penalties, global_penalty)


def attach_calibration_guard(
    frame: pl.DataFrame,
    guard: CalibrationGuard,
    config: TrainingConfig,
) -> pl.DataFrame:
    price = frame["selected_cost_5"].to_numpy()
    buckets = _price_bucket_indices(price, config.policy.price_bucket_edges)
    bands = frame["time_band"].to_numpy()
    penalties = np.array(
        [guard.penalties.get(f"{band}:{bucket}", guard.global_penalty) for band, bucket in zip(bands, buckets, strict=True)]
    )
    conservative_probability = np.clip(
        frame["probability_selected"].to_numpy() - penalties,
        0.0,
        1.0,
    )
    required_edge = np.array(config.policy.price_bucket_minimum_edges)[buckets]
    return frame.with_columns(
        pl.Series("price_bucket_index", buckets.astype(np.int8)),
        pl.Series("calibration_uncertainty_penalty", penalties),
        pl.Series("conservative_probability_selected", conservative_probability),
        pl.Series("conservative_edge_5", conservative_probability - price),
        pl.Series("price_bucket_minimum_edge", required_edge),
    )


def attach_probability_stability(frame: pl.DataFrame) -> pl.DataFrame:
    output = frame
    for lag in (5, 15, 30):
        lagged = frame.select(
            "market_id",
            (pl.col("seconds_elapsed") + lag).alias("seconds_elapsed"),
            pl.col("probability_selected").alias(f"_probability_lag_{lag}"),
        )
        output = output.join(lagged, on=["market_id", "seconds_elapsed"], how="left")
        output = output.with_columns(
            (pl.col("probability_selected") - pl.col(f"_probability_lag_{lag}")).alias(
                f"probability_change_{lag}s"
            )
        ).drop(f"_probability_lag_{lag}")
    return output.with_columns(
        pl.max_horizontal(
            pl.col("probability_change_5s").abs(),
            pl.col("probability_change_15s").abs(),
            pl.col("probability_change_30s").abs(),
        ).alias("probability_instability_30s")
    )


def prepare_scored_frame(
    frame: pl.DataFrame,
    experts: dict[str, Expert],
    family_proxies: dict[str, Expert],
    price_time_calibration: PriceTimeCalibration,
    calibration_guard: CalibrationGuard,
    config: TrainingConfig,
) -> pl.DataFrame:
    scored = score_frame(frame, experts, family_proxies, config)
    scored = attach_price_time_calibration(scored, price_time_calibration, config)
    scored = attach_calibration_guard(scored, calibration_guard, config)
    return attach_probability_stability(scored)


def _price_bucket_indices(
    values: np.ndarray,
    edges: tuple[float, ...],
) -> np.ndarray:
    return np.clip(np.searchsorted(np.asarray(edges[1:-1]), values, side="right"), 0, len(edges) - 2)


def fit_admission_model(
    frame: pl.DataFrame,
    config: TrainingConfig,
) -> PayoffAdmissionModel:
    if frame["market_id"].n_unique() < 250:
        raise RuntimeError("admission model lacks held-out markets")
    feature_names = variable_feature_names(frame, ADMISSION_FEATURES)
    oof = _chronological_payoff_oof(frame, feature_names, config)
    residual_quantile = 1.0 - config.policy.payoff_lower_bound_quantile
    residual = (
        oof["payoff_expected_stress_edge"].to_numpy()
        - oof["realized_stress_edge"].to_numpy()
    )
    global_penalty = float(np.quantile(residual, residual_quantile))
    penalties: dict[str, float] = {}
    for band in config.bands:
        for bucket in range(len(config.policy.price_bucket_edges) - 1):
            cell = oof.filter(
                (pl.col("time_band") == band.name)
                & (pl.col("price_bucket_index") == bucket)
            )
            markets = cell["market_id"].n_unique()
            if markets == 0:
                penalties[f"{band.name}:{bucket}"] = global_penalty
                continue
            cell_residual = (
                cell["payoff_expected_stress_edge"].to_numpy()
                - cell["realized_stress_edge"].to_numpy()
            )
            cell_penalty = float(np.quantile(cell_residual, residual_quantile))
            shrink = markets / (markets + 200.0)
            penalties[f"{band.name}:{bucket}"] = (
                shrink * cell_penalty + (1.0 - shrink) * global_penalty
            )

    oof_bands = oof["time_band"].to_numpy()
    oof_buckets = oof["price_bucket_index"].to_numpy()
    oof_penalties = np.asarray(
        [
            penalties.get(f"{band}:{bucket}", global_penalty)
            for band, bucket in zip(oof_bands, oof_buckets, strict=True)
        ]
    )
    oof_lower_bound = oof["payoff_expected_stress_edge"].to_numpy() - oof_penalties
    oof_diagnostics = {
        "method": "three_expanding_chronological_residual_folds",
        "rows": oof.height,
        "markets": oof["market_id"].n_unique(),
        "target_coverage": 1.0 - config.policy.payoff_lower_bound_quantile,
        "empirical_coverage": float(
            np.mean(oof["realized_stress_edge"].to_numpy() >= oof_lower_bound)
        ),
        "global_penalty": global_penalty,
    }

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=5.0,
        random_state=config.random_seed,
        early_stopping=False,
    )
    realized_stress_edge = _realized_stress_edge(frame, config)
    profitable = realized_stress_edge > 0
    weights = market_equal_weights(frame)
    classifier.fit(
        _matrix(frame, feature_names),
        profitable.astype(np.int8),
        sample_weight=weights,
    )
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=5.0,
        random_state=config.random_seed + 1,
        early_stopping=False,
    )
    regressor.fit(_matrix(frame, feature_names), realized_stress_edge, sample_weight=weights)
    return PayoffAdmissionModel(
        feature_names,
        classifier,
        regressor,
        penalties,
        global_penalty,
        oof_diagnostics,
        config.execution.stress_slippage_per_share,
    )


def _chronological_payoff_oof(
    frame: pl.DataFrame,
    feature_names: tuple[str, ...],
    config: TrainingConfig,
) -> pl.DataFrame:
    windows = frame["window_start"].unique().sort().to_list()
    boundaries = [windows[min(int(len(windows) * fraction), len(windows) - 1)] for fraction in (0.40, 0.60, 0.80)]
    folds: list[pl.DataFrame] = []
    starts = boundaries
    ends = [boundaries[1], boundaries[2], config.windows.admission_end]
    for fold_index, (train_end, validation_end) in enumerate(zip(starts, ends, strict=True)):
        fit = frame.filter(pl.col("window_start") < train_end)
        validation = frame.filter(
            (pl.col("window_start") >= train_end)
            & (pl.col("window_start") < validation_end)
        )
        if fit["market_id"].n_unique() < 100 or validation["market_id"].n_unique() < 40:
            raise RuntimeError("admission lower-bound fold lacks chronological coverage")
        regressor = _fit_stress_edge_regressor(
            fit,
            feature_names,
            config,
            random_seed=config.random_seed + 200 + fold_index,
        )
        predicted = regressor.predict(_matrix(validation, feature_names))
        folds.append(
            validation.select(
                "market_id",
                "time_band",
                "price_bucket_index",
            ).with_columns(
                pl.Series("payoff_expected_stress_edge", predicted),
                pl.Series("realized_stress_edge", _realized_stress_edge(validation, config)),
            )
        )
    if not folds:
        raise RuntimeError("admission lower-bound calibration produced no folds")
    return pl.concat(folds, how="vertical")


def _fit_stress_edge_regressor(
    frame: pl.DataFrame,
    feature_names: tuple[str, ...],
    config: TrainingConfig,
    *,
    random_seed: int,
) -> HistGradientBoostingRegressor:
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=5.0,
        random_state=random_seed,
        early_stopping=False,
    )
    regressor.fit(
        _matrix(frame, feature_names),
        _realized_stress_edge(frame, config),
        sample_weight=market_equal_weights(frame),
    )
    return regressor


def _realized_stress_edge(frame: pl.DataFrame, config: TrainingConfig) -> np.ndarray:
    return (
        frame["direction_correct"].to_numpy().astype(float)
        - frame["selected_cost_5"].to_numpy()
        - config.execution.stress_slippage_per_share
    )


def attach_admission_probability(
    frame: pl.DataFrame,
    model: PayoffAdmissionModel,
) -> pl.DataFrame:
    matrix = _matrix(frame, model.feature_names)
    probability = model.profitable_classifier.predict_proba(matrix)[:, 1]
    expected_stress_edge = model.stress_edge_regressor.predict(matrix)
    bands = frame["time_band"].to_numpy()
    buckets = frame["price_bucket_index"].to_numpy()
    penalties = np.asarray(
        [
            model.lower_bound_penalties.get(
                f"{band}:{bucket}", model.global_lower_bound_penalty
            )
            for band, bucket in zip(bands, buckets, strict=True)
        ]
    )
    conditional_loss = (
        frame["selected_cost_5"].to_numpy()
        + model.stress_slippage_per_share
    )
    loss_probability = 1.0 - probability
    return frame.with_columns(
        pl.Series("admission_probability", probability),
        pl.Series("payoff_expected_stress_edge", expected_stress_edge),
        pl.Series("payoff_lower_bound_penalty", penalties),
        pl.Series("payoff_stress_edge_lower_bound", expected_stress_edge - penalties),
        pl.Series("payoff_loss_probability", loss_probability),
        pl.Series("payoff_conditional_loss", conditional_loss),
        pl.Series("payoff_expected_shortfall", loss_probability * conditional_loss),
    )


def select_policy_thresholds(frame: pl.DataFrame, config: TrainingConfig) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    selected: dict[str, dict[str, float]] = {}
    history: dict[str, Any] = {}
    for band in config.bands:
        subset = frame.filter(pl.col("time_band") == band.name).sort(["market_id", "seconds_elapsed", "observed_at"])
        best: tuple[tuple[float, float, float], dict[str, float], dict[str, Any]] | None = None
        qualified_summaries: list[dict[str, Any]] = []
        attempts = 0
        qualified_attempts = 0
        for confidence in config.policy.confidence_thresholds:
            for edge in config.policy.edge_thresholds:
                for admission in config.policy.admission_thresholds:
                    for payoff_lower_bound in config.policy.payoff_lower_bound_thresholds:
                        attempts += 1
                        trades = _first_crossings_array(
                            subset,
                            confidence,
                            edge,
                            admission,
                            payoff_lower_bound,
                            use_admission=True,
                        )
                        metrics = policy_metrics(trades, config, quantity=5)
                        folds = rolling_policy_metrics(
                            trades,
                            config,
                            config.windows.validation_start,
                            config.windows.validation_end,
                            quantity=5,
                        )
                        values = {
                            "confidence": confidence,
                            "edge": edge,
                            "admission": admission,
                            "payoff_lower_bound": payoff_lower_bound,
                        }
                        score = (
                            metrics["trades"],
                            folds["median_fold_stress_expectancy"],
                            metrics["stress_net_pnl"],
                        )
                        record = {**metrics, "rolling_folds": folds}
                        qualified = (
                            metrics["trades"] >= band.minimum_validation_trades
                            and metrics["accuracy"] >= band.minimum_validation_accuracy
                            and metrics["stress_expectancy_per_trade"]
                            >= config.policy.minimum_stress_expectancy_per_trade
                            and (metrics["profit_factor"] or 0.0)
                            >= config.policy.minimum_profit_factor
                            and metrics["payoff_ratio"] >= config.policy.minimum_payoff_ratio
                            and metrics["active_days"] >= config.policy.minimum_active_days
                            and metrics["daily_pnl_concentration"]
                            <= config.policy.maximum_daily_pnl_concentration
                            and folds["profitable_fold_ratio"]
                            >= config.policy.minimum_profitable_fold_ratio
                        )
                        if not qualified:
                            continue
                        qualified_attempts += 1
                        qualified_summaries.append(
                            {
                                **values,
                                "trades": metrics["trades"],
                                "accuracy": metrics["accuracy"],
                                "stress_expectancy_per_trade": metrics[
                                    "stress_expectancy_per_trade"
                                ],
                                "stress_net_pnl": metrics["stress_net_pnl"],
                                "profit_factor": metrics["profit_factor"],
                                "payoff_ratio": metrics["payoff_ratio"],
                                "profitable_fold_ratio": folds["profitable_fold_ratio"],
                            }
                        )
                        if best is None or score > best[0]:
                            best = (score, values, record)
        if best is None:
            selected[band.name] = {"enabled": False}
            history[band.name] = {
                "attempts": attempts,
                "qualified_attempts": qualified_attempts,
                "enabled": False,
                "reason": "no rolling-validation-qualified policy",
                "coverage_expectancy_frontier": [],
            }
            continue
        _, values, metrics = best
        values = {**values, "enabled": True}
        selected[band.name] = values
        history[band.name] = {
            "attempts": attempts,
            "qualified_attempts": qualified_attempts,
            "selected": values,
            "validation_metrics": metrics,
            "strictly_qualified": True,
            "coverage_expectancy_frontier": _coverage_expectancy_frontier(
                qualified_summaries
            ),
        }
    return selected, history


def _coverage_expectancy_frontier(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for candidate in records:
        dominated = any(
            other["trades"] >= candidate["trades"]
            and other["stress_expectancy_per_trade"]
            >= candidate["stress_expectancy_per_trade"]
            and (
                other["trades"] > candidate["trades"]
                or other["stress_expectancy_per_trade"]
                > candidate["stress_expectancy_per_trade"]
            )
            for other in records
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda row: (row["trades"], row["stress_expectancy_per_trade"]),
        reverse=True,
    )[:25]


def _first_crossings_array(
    frame: pl.DataFrame,
    confidence: float,
    edge: float,
    admission: float,
    payoff_lower_bound: float = -math.inf,
    *,
    use_admission: bool,
) -> pl.DataFrame:
    conservative_edge = frame["conservative_edge_5"].to_numpy()
    required_edge = frame["price_bucket_minimum_edge"].to_numpy()
    mask = (
        (frame["conservative_probability_selected"].to_numpy() >= confidence)
        & (conservative_edge >= edge)
        & (conservative_edge >= required_edge)
    )
    if use_admission:
        mask &= (
            (frame["admission_probability"].to_numpy() >= admission)
            & (
                frame["payoff_stress_edge_lower_bound"].to_numpy()
                >= payoff_lower_bound
            )
        )
    indices = np.flatnonzero(mask)
    if not len(indices):
        return frame.head(0)
    markets = frame["market_id"].to_numpy()
    _, first_positions = np.unique(markets[indices], return_index=True)
    return frame[indices[np.sort(first_positions)].tolist()]


def select_first_crossings(
    frame: pl.DataFrame,
    thresholds: dict[str, dict[str, float]],
    *,
    use_admission: bool,
) -> pl.DataFrame:
    eligible = np.zeros(frame.height, dtype=bool)
    bands = frame["time_band"].to_numpy()
    confidence = frame["conservative_probability_selected"].to_numpy()
    edge = frame["conservative_edge_5"].to_numpy()
    required_edge = frame["price_bucket_minimum_edge"].to_numpy()
    admission = frame["admission_probability"].to_numpy()
    payoff_lower_bound = frame["payoff_stress_edge_lower_bound"].to_numpy()
    for name, values in thresholds.items():
        if not values.get("enabled", True):
            continue
        mask = (
            (bands == name)
            & (confidence >= values["confidence"])
            & (edge >= values["edge"])
            & (edge >= required_edge)
        )
        if use_admission:
            mask &= (admission >= values["admission"]) & (
                payoff_lower_bound >= values["payoff_lower_bound"]
            )
        eligible |= mask
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return frame.head(0)
    markets = frame["market_id"].to_numpy()
    _, first_positions = np.unique(markets[indices], return_index=True)
    return frame[indices[np.sort(first_positions)].tolist()]


def probability_metrics(frame: pl.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    if frame.is_empty():
        return {"rows": 0, "markets": 0, "accuracy": 0.0, "log_loss": None, "brier": None, "bias": None}
    labels = frame["label_up"].to_numpy()
    weights = market_equal_weights(frame)
    prediction = probability >= 0.5
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "accuracy": float(np.average(prediction == labels.astype(bool), weights=weights)),
        "log_loss": float(log_loss(labels, probability, sample_weight=weights, labels=[0, 1])),
        "brier": float(np.average((probability - labels) ** 2, weights=weights)),
        "bias": float(np.average(probability - labels, weights=weights)),
    }


def policy_metrics(frame: pl.DataFrame, config: TrainingConfig, *, quantity: int) -> dict[str, Any]:
    if frame.is_empty():
        return {
            "trades": 0,
            "accuracy": 0.0,
            "net_pnl": 0.0,
            "expectancy_per_trade": 0.0,
            "profit_factor": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "payoff_ratio": 0.0,
            "maximum_drawdown": 0.0,
            "stress_net_pnl": 0.0,
            "stress_expectancy_per_trade": 0.0,
            "average_entry_second": None,
            "median_entry_second": None,
            "profitable_day_ratio": 0.0,
            "worst_day_net_pnl": 0.0,
            "daily_pnl_concentration": 0.0,
            "active_days": 0,
        }
    if quantity not in VWAP_QUANTITIES:
        raise ValueError(quantity)
    predicted_up = frame["predicted_up"].to_numpy().astype(bool)
    correct = predicted_up == frame["label_up"].to_numpy().astype(bool)
    price = np.where(predicted_up, frame[f"up_ask_vwap_{quantity}"].to_numpy(), frame[f"down_ask_vwap_{quantity}"].to_numpy())
    fee = frame["fee_rate"].to_numpy() * price * (1 - price)
    pnl = (correct.astype(float) - price - fee - config.execution.execution_reserve_per_share) * quantity
    stress = pnl - config.execution.stress_slippage_per_share * quantity
    gains = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    cumulative = np.cumsum(pnl)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
    drawdown = peaks - np.concatenate(([0.0], cumulative))
    entry = frame["seconds_elapsed"].to_numpy()
    days = frame["window_start"].dt.date().to_numpy()
    daily_pnl: dict[Any, float] = {}
    for day, value in zip(days, pnl, strict=True):
        daily_pnl[day] = daily_pnl.get(day, 0.0) + float(value)
    daily_values = np.asarray(list(daily_pnl.values()), dtype=float)
    absolute_daily_sum = float(np.abs(daily_values).sum())
    return {
        "trades": len(pnl),
        "accuracy": float(correct.mean()),
        "net_pnl": float(pnl.sum()),
        "expectancy_per_trade": float(pnl.mean()),
        "profit_factor": float(gains.sum() / -losses.sum()) if len(losses) else None,
        "average_win": float(gains.mean()) if len(gains) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "payoff_ratio": (
            float(gains.mean() / -losses.mean()) if len(gains) and len(losses) else 0.0
        ),
        "maximum_drawdown": float(drawdown.max()),
        "stress_net_pnl": float(stress.sum()),
        "stress_expectancy_per_trade": float(stress.mean()),
        "average_entry_second": float(entry.mean()),
        "median_entry_second": float(np.median(entry)),
        "entry_second_p25": float(np.quantile(entry, 0.25)),
        "entry_second_p75": float(np.quantile(entry, 0.75)),
        "mean_vwap": float(price.mean()),
        "gross_notional": float((price * quantity).sum()),
        "return_on_entry_cost": float(pnl.sum() / (price * quantity).sum()),
        "up_trades": int(predicted_up.sum()),
        "down_trades": int((~predicted_up).sum()),
        "profitable_day_ratio": float((daily_values > 0).mean()),
        "worst_day_net_pnl": float(daily_values.min()),
        "active_days": len(daily_values),
        "daily_pnl_concentration": (
            float(np.abs(daily_values).max() / absolute_daily_sum)
            if absolute_daily_sum
            else 0.0
        ),
    }


def rolling_policy_metrics(
    frame: pl.DataFrame,
    config: TrainingConfig,
    start: datetime,
    end: datetime,
    *,
    quantity: int,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    cursor = start
    width = timedelta(days=config.policy.rolling_fold_days)
    while cursor < end:
        fold_end = min(cursor + width, end)
        subset = _block(frame, cursor, fold_end)
        metrics = policy_metrics(subset, config, quantity=quantity)
        folds.append(
            {
                "start": cursor.isoformat(),
                "end": fold_end.isoformat(),
                **metrics,
            }
        )
        cursor = fold_end
    expectations = np.asarray(
        [row["stress_expectancy_per_trade"] for row in folds],
        dtype=float,
    )
    profitable = np.asarray([row["stress_net_pnl"] > 0 for row in folds], dtype=bool)
    return {
        "fold_days": config.policy.rolling_fold_days,
        "fold_count": len(folds),
        "profitable_folds": int(profitable.sum()),
        "profitable_fold_ratio": float(profitable.mean()) if len(profitable) else 0.0,
        "median_fold_stress_expectancy": (
            float(np.median(expectations)) if len(expectations) else 0.0
        ),
        "folds": folds,
    }


def policy_metrics_by_price_bucket(
    frame: pl.DataFrame,
    config: TrainingConfig,
    *,
    quantity: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for index, (lower, upper) in enumerate(
        zip(
            config.policy.price_bucket_edges[:-1],
            config.policy.price_bucket_edges[1:],
            strict=True,
        )
    ):
        subset = frame.filter(pl.col("price_bucket_index") == index)
        output[f"{lower:.2f}-{upper:.2f}"] = policy_metrics(
            subset,
            config,
            quantity=quantity,
        )
    return output


def market_equal_weights(frame: pl.DataFrame) -> np.ndarray:
    counts = frame.group_by("market_id").len().rename({"len": "_market_rows"})
    weights = frame.select("market_id").join(counts, on="market_id", how="left")["_market_rows"].to_numpy()
    output = 1.0 / weights.astype(float)
    return output / output.mean()


def variable_feature_names(
    frame: pl.DataFrame,
    feature_names: Iterable[str],
) -> tuple[str, ...]:
    selected: list[str] = []
    for name in feature_names:
        values = frame[name].cast(pl.Float64).drop_nulls()
        values = values.filter(values.is_finite())
        if values.n_unique() >= 2:
            selected.append(name)
    if not selected:
        raise RuntimeError("model frame has no variable finite features")
    return tuple(selected)


def _matrix(frame: pl.DataFrame, feature_names: Iterable[str]) -> np.ndarray:
    matrix = frame.select(list(feature_names)).cast(pl.Float64).to_numpy()
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def _block(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    return frame.filter((pl.col("window_start") >= start) & (pl.col("window_start") < end))


def render_report(metrics: dict[str, Any]) -> str:
    test = metrics["test"]["with_admission_veto"]
    control = metrics["test"]["without_admission_veto"]
    lines = [
        "# BTC Five-Minute Payoff-Robust Continuous-Edge Training",
        "",
        (
            "Status: **development qualified; not runtime exported**"
            if metrics["qualification"]["passed"]
            else "Status: **development-only evidence; not runtime exported**"
        ),
        "",
        f"Selected probability candidate: `{metrics['selected_candidate']}`",
        "",
        (
            "The candidate passed every frozen fresh-holdout gate."
            if metrics["qualification"]["passed"]
            else "The candidate did not pass every frozen fresh-holdout gate; review the failed checks before considering deployment."
        ),
        "",
        "## Fresh chronological holdout",
        "",
        "| Policy | Trades | Accuracy | Net PnL (VWAP5) | Expectancy | PF | Avg entry | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Without payoff admission | {control['trades']} | {control['accuracy']:.2%} | {control['net_pnl']:.2f} | {control['expectancy_per_trade']:.4f} | {_fmt(control['profit_factor'])} | {_fmt(control['average_entry_second'])} | — |",
        f"| Selected payoff-aware edge | {test['trades']} | {test['accuracy']:.2%} | {test['net_pnl']:.2f} | {test['expectancy_per_trade']:.4f} | {_fmt(test['profit_factor'])} | {_fmt(test['average_entry_second'])} | {test.get('market_coverage', 0):.2%} |",
        "",
        f"Stress PnL: **{test['stress_net_pnl']:.2f}**; stress expectancy: **{test['stress_expectancy_per_trade']:.4f}**; strict-data coverage: **{test.get('strict_data_coverage', 0):.2%}**; end-to-end trade coverage: **{test.get('end_to_end_market_coverage', 0):.2%}**.",
        "",
        f"Average win: **{test['average_win']:.3f}**; average loss: **{test['average_loss']:.3f}**; payoff ratio: **{test['payoff_ratio']:.3f}**; active days: **{test['active_days']}**; daily PnL concentration: **{test['daily_pnl_concentration']:.2%}**.",
        "",
        f"Chronological payoff lower-bound calibration coverage: **{metrics['payoff_lower_bound']['empirical_coverage']:.2%}** versus **{metrics['payoff_lower_bound']['target_coverage']:.2%}** target.",
        "",
        "## Fixed-entry VWAP capacity curve",
        "",
        "The side and timestamp are frozen from the VWAP5 policy; only exact execution size changes.",
        "",
        "| Shares | Trades | Accuracy | Mean VWAP | Net PnL | Expectancy | PF | Max drawdown |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for quantity, row in metrics["test_fixed_entry_capacity_curve"].items():
        lines.append(
            f"| {quantity} | {row['trades']} | {row['accuracy']:.2%} | {row.get('mean_vwap', 0):.4f} | "
            f"{row['net_pnl']:.2f} | {row['expectancy_per_trade']:.4f} | {_fmt(row['profit_factor'])} | {row['maximum_drawdown']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Test results by entry band (VWAP5)",
            "",
            "| Band | Trades | Accuracy | Net PnL | Expectancy | Average entry |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for band, row in metrics["test_by_time_band"].items():
        lines.append(
            f"| {band} | {row['trades']} | {row['accuracy']:.2%} | {row['net_pnl']:.2f} | "
            f"{row['expectancy_per_trade']:.4f} | {_fmt(row['average_entry_second'])} |"
        )
    lines.extend(
        [
            "",
            "## Rolling three-day comparison",
            "",
            "| Fold | Trades | Accuracy | Stress PnL | Stress expectancy |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for fold in metrics["test_rolling_folds"]["folds"]:
        lines.append(
            f"| {fold['start'][:10]} to {fold['end'][:10]} | {fold['trades']} | "
            f"{fold['accuracy']:.2%} | {fold['stress_net_pnl']:.2f} | "
            f"{fold['stress_expectancy_per_trade']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Comparison results by entry-price bucket (VWAP5)",
            "",
            "| Entry price | Trades | Accuracy | Net PnL | Stress PnL |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for bucket, row in metrics["test_by_price_bucket"].items():
        lines.append(
            f"| {bucket} | {row['trades']} | {row['accuracy']:.2%} | "
            f"{row['net_pnl']:.2f} | {row['stress_net_pnl']:.2f} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {value}" for value in metrics["limitations"])
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.3f}"


def _source_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _git_revision(package_root: Path) -> str:
    head = package_root.parents[1] / ".git"
    if head.is_file():
        content = head.read_text().strip()
        if content.startswith("gitdir:"):
            git_dir = Path(content.split(":", 1)[1].strip())
            revision = (git_dir / "HEAD").read_text().strip()
            if revision.startswith("ref:"):
                common_dir = git_dir
                commondir_path = git_dir / "commondir"
                if commondir_path.is_file():
                    common_dir = (git_dir / commondir_path.read_text().strip()).resolve()
                return (common_dir / revision.split(" ", 1)[1]).read_text().strip()
            return revision
    return "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_finite(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _utc(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamps must be timezone aware")
    return parsed.astimezone(UTC)


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run_dir, metrics = run_training(load_config(args.config))
    print(f"run: {run_dir}")
    print(json.dumps(metrics["test"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
