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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .core_extract import (
    configure_read_only_connection,
    database_connection,
    file_sha256,
)

SCHEMA_VERSION = "btc-continuous-edge-training-v1"
MODEL_SCHEMA_VERSION = "btc-continuous-edge-development-artifact-v1"
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


@dataclass(frozen=True)
class PathConfig:
    oracle_features: Path
    chainlink_features: Path
    l2_features: Path
    capacity_evidence: Path
    runs: Path
    committed_results: Path


@dataclass(frozen=True)
class TrainingConfig:
    source_path: Path
    package_root: Path
    profile: str
    random_seed: int
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
    if tuple((band.start_second, band.end_second_exclusive) for band in config.bands) != (
        (15, 90),
        (90, 180),
        (180, 241),
    ):
        raise ValueError("time bands must cover the exact 15-240 second policy range")
    for path in (config.paths.oracle_features, config.paths.chainlink_features, config.paths.l2_features):
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

    scored_admission = score_frame(
        _block(frame, config.windows.calibration_end, config.windows.admission_end),
        selected_experts,
        config,
    )
    admission_model, admission_feature_names = fit_admission_model(scored_admission, config)
    validation = attach_admission_probability(
        score_frame(
            _block(frame, config.windows.validation_start, config.windows.validation_end),
            selected_experts,
            config,
        ),
        admission_model,
        admission_feature_names,
    )
    thresholds, threshold_search = select_policy_thresholds(validation, config)
    validation_control = select_first_crossings(validation, thresholds, use_admission=False)
    validation_selected = select_first_crossings(validation, thresholds, use_admission=True)

    print("test: opening untouched chronological block", flush=True)
    test_scored = attach_admission_probability(
        score_frame(
            _block(frame, config.windows.validation_end, config.windows.test_end),
            selected_experts,
            config,
        ),
        admission_model,
        admission_feature_names,
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
    test_market_count = test_scored["market_id"].n_unique()
    test_metrics["with_admission_veto"]["market_coverage"] = (
        test_selected["market_id"].n_unique() / test_market_count if test_market_count else 0.0
    )
    qualification_checks = {
        "positive_test_net_pnl": test_metrics["with_admission_veto"]["net_pnl"] > 0,
        "positive_test_stress_expectancy": (
            test_metrics["with_admission_veto"]["stress_expectancy_per_trade"] > 0
        ),
        "test_profit_factor_at_least_one": (
            (test_metrics["with_admission_veto"]["profit_factor"] or 0.0) >= 1.0
        ),
        "correctness_veto_improves_test_net_pnl": (
            test_metrics["with_admission_veto"]["net_pnl"]
            > test_metrics["without_admission_veto"]["net_pnl"]
        ),
        "early_test_accuracy_floor": (
            band_metrics["early"]["accuracy"]
            >= next(band.minimum_validation_accuracy for band in config.bands if band.name == "early")
        ),
        "early_test_positive_expectancy": band_metrics["early"]["expectancy_per_trade"] > 0,
        "late_test_accuracy_floor": (
            band_metrics["late"]["accuracy"]
            >= next(band.minimum_validation_accuracy for band in config.bands if band.name == "late")
        ),
    }
    qualification = {
        "passed": all(qualification_checks.values()),
        "checks": qualification_checks,
        "decision": "qualified" if all(qualification_checks.values()) else "rejected",
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
        "admission_model": admission_model,
        "admission_feature_names": list(admission_feature_names),
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
        "admission_probability",
        "selected_edge_5",
        "selected_cost_5",
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
        },
        "data": {
            "oracle_features": _source_identity(config.paths.oracle_features),
            "chainlink_features": _source_identity(config.paths.chainlink_features),
            "l2_features": _source_identity(config.paths.l2_features),
            "capacity_manifest": evidence_manifest,
            "coverage": coverage,
        },
        "candidate_metrics": candidate_metrics,
        "challenger_decision": challenger_decision,
        "selected_candidate": selected_candidate,
        "threshold_search": threshold_search,
        "selected_thresholds": thresholds,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_by_time_band": band_metrics,
        "test_fixed_entry_capacity_curve": capacity_curve,
        "qualification": qualification,
        "model_artifact": {
            "path": "model.joblib",
            "sha256": file_sha256(joblib_path),
        },
        "limitations": [
            "Capacity evidence ends at second 240; seconds 241-299 are not evaluated.",
            "Open interest begins after the outcome-fit window and is diagnostic only.",
            "Aggregate trade prints overlap only the end of the test block and are diagnostic only.",
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
    source_columns = (*key_columns, *CORE_FEATURES, *ORACLE_FEATURES)
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
    return frame.with_columns(
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


def score_frame(frame: pl.DataFrame, experts: dict[str, Expert], config: TrainingConfig) -> pl.DataFrame:
    pieces = []
    for band in config.bands:
        subset = frame.filter(pl.col("time_band") == band.name)
        if subset.is_empty():
            continue
        probability = experts[band.name].probability(subset)
        pieces.append(subset.with_columns(pl.Series("probability_up", probability)))
    if not pieces:
        return frame.head(0)
    scored = pl.concat(pieces, how="vertical").sort(["market_id", "seconds_elapsed", "observed_at"])
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
    )


def fit_admission_model(
    frame: pl.DataFrame,
    config: TrainingConfig,
) -> tuple[HistGradientBoostingClassifier, tuple[str, ...]]:
    if frame["market_id"].n_unique() < 250:
        raise RuntimeError("admission model lacks held-out markets")
    feature_names = variable_feature_names(frame, ADMISSION_FEATURES)
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=5.0,
        random_state=config.random_seed,
        early_stopping=False,
    )
    model.fit(
        _matrix(frame, feature_names),
        frame["direction_correct"].to_numpy().astype(np.int8),
        sample_weight=market_equal_weights(frame),
    )
    return model, feature_names


def attach_admission_probability(
    frame: pl.DataFrame,
    model: HistGradientBoostingClassifier,
    feature_names: tuple[str, ...],
) -> pl.DataFrame:
    probability = model.predict_proba(_matrix(frame, feature_names))[:, 1]
    return frame.with_columns(pl.Series("admission_probability", probability))


def select_policy_thresholds(frame: pl.DataFrame, config: TrainingConfig) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    selected: dict[str, dict[str, float]] = {}
    history: dict[str, Any] = {}
    for band in config.bands:
        subset = frame.filter(pl.col("time_band") == band.name).sort(["market_id", "seconds_elapsed", "observed_at"])
        best: tuple[tuple[float, float, float], dict[str, float], dict[str, Any]] | None = None
        attempts = 0
        qualified_attempts = 0
        for confidence in config.policy.confidence_thresholds:
            for edge in config.policy.edge_thresholds:
                for admission in config.policy.admission_thresholds:
                    attempts += 1
                    trades = _first_crossings_array(subset, confidence, edge, admission, use_admission=True)
                    metrics = policy_metrics(trades, config, quantity=5)
                    qualified = (
                        metrics["trades"] >= band.minimum_validation_trades
                        and metrics["accuracy"] >= band.minimum_validation_accuracy
                        and metrics["stress_expectancy_per_trade"] > 0
                    )
                    if not qualified:
                        continue
                    qualified_attempts += 1
                    score = (metrics["stress_net_pnl"], metrics["accuracy"], metrics["trades"])
                    values = {"confidence": confidence, "edge": edge, "admission": admission}
                    if best is None or score > best[0]:
                        best = (score, values, metrics)
        if best is None:
            raise RuntimeError(f"no validation-qualified policy threshold for {band.name}")
        _, values, metrics = best
        selected[band.name] = values
        history[band.name] = {
            "attempts": attempts,
            "qualified_attempts": qualified_attempts,
            "selected": values,
            "validation_metrics": metrics,
        }
    return selected, history


def _first_crossings_array(
    frame: pl.DataFrame,
    confidence: float,
    edge: float,
    admission: float,
    *,
    use_admission: bool,
) -> pl.DataFrame:
    mask = (frame["probability_selected"].to_numpy() >= confidence) & (frame["selected_edge_5"].to_numpy() >= edge)
    if use_admission:
        mask &= frame["admission_probability"].to_numpy() >= admission
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
    confidence = frame["probability_selected"].to_numpy()
    edge = frame["selected_edge_5"].to_numpy()
    admission = frame["admission_probability"].to_numpy()
    for name, values in thresholds.items():
        mask = (bands == name) & (confidence >= values["confidence"]) & (edge >= values["edge"])
        if use_admission:
            mask &= admission >= values["admission"]
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
            "maximum_drawdown": 0.0,
            "stress_net_pnl": 0.0,
            "stress_expectancy_per_trade": 0.0,
            "average_entry_second": None,
            "median_entry_second": None,
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
    return {
        "trades": len(pnl),
        "accuracy": float(correct.mean()),
        "net_pnl": float(pnl.sum()),
        "expectancy_per_trade": float(pnl.mean()),
        "profit_factor": float(gains.sum() / -losses.sum()) if len(losses) else None,
        "average_win": float(gains.mean()) if len(gains) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
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
    }


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
        "# BTC Five-Minute Continuous-Edge Directional Training",
        "",
        (
            "Status: **development qualified; not runtime exported**"
            if metrics["qualification"]["passed"]
            else "Status: **development candidate rejected; not runtime exported**"
        ),
        "",
        f"Selected probability candidate: `{metrics['selected_candidate']}`",
        "",
        (
            "The candidate passed every frozen development gate."
            if metrics["qualification"]["passed"]
            else "The candidate failed frozen untouched-test expectancy gates and must not be deployed."
        ),
        "",
        "## Untouched chronological test",
        "",
        "| Policy | Trades | Accuracy | Net PnL (VWAP5) | Expectancy | PF | Avg entry | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Without correctness veto | {control['trades']} | {control['accuracy']:.2%} | {control['net_pnl']:.2f} | {control['expectancy_per_trade']:.4f} | {_fmt(control['profit_factor'])} | {_fmt(control['average_entry_second'])} | — |",
        f"| Selected continuous edge | {test['trades']} | {test['accuracy']:.2%} | {test['net_pnl']:.2f} | {test['expectancy_per_trade']:.4f} | {_fmt(test['profit_factor'])} | {_fmt(test['average_entry_second'])} | {test.get('market_coverage', 0):.2%} |",
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
