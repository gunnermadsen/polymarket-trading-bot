"""Offline tournament for selective BTC five-minute middle-market policies."""

from __future__ import annotations

import argparse
import json
import math
import platform
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .chainlink_oi_features import (
    BINANCE_OI_FEATURES,
    _attach_candle_features,
    _attach_open_interest_features,
    _query_frame,
)
from .continuous_edge_training import (
    BOOK_RAW_FEATURES,
    CHAINLINK_FEATURES,
    L2_FEATURES,
    PRIMARY_FEATURES,
    VWAP_QUANTITIES,
    ExecutionConfig,
    _extract_capacity_partition,
    _require_complete_hours,
    attach_book_features,
    load_training_frame,
    market_equal_weights,
    policy_metrics,
    policy_metrics_by_price_bucket,
    rolling_policy_metrics,
)
from .continuous_edge_training import (
    load_config as load_development_config,
)
from .core_config import load_core_config
from .core_extract import (
    configure_read_only_connection,
    database_connection,
    file_sha256,
)
from .spot_l2_chainlink_features import join_qualified_l2

SCHEMA_VERSION = "btc-middle-market-payoff-tournament-v1"
MODEL_SCHEMA_VERSION = "btc-middle-market-tournament-development-artifact-v1"

RISK_FEATURES = (
    "probability_selected",
    "correctness_probability",
    "lower_correctness_probability",
    "selected_cost_5",
    "stress_edge_lower_bound",
    "seconds_elapsed_scaled",
    "btc_path_from_window_open_bps",
    "btc_cross_venue_boundary_gap_bps",
    "btc_path_terminal_volatility_z",
    "btc_boundary_terminal_volatility_z",
    "btc_boundary_cross_count",
    "btc_seconds_since_boundary_cross",
    "btc_reversal_5_vs_30",
    "btc_volatility_shock_30_vs_120",
    "btc_signed_flow_30s",
    "btc_signed_flow_60s",
    "oracle_return_from_window_open_bps",
    "binance_oracle_basis_bps",
    "pm_vwap5_overround",
    "pm_vwap200_overround",
    "pm_depth_imbalance",
    "pm_up_book_age_seconds",
    "pm_down_book_age_seconds",
)

REGIME_FEATURES = (
    "btc_path_terminal_volatility_z",
    "btc_boundary_terminal_volatility_z",
    "btc_boundary_cross_count",
    "btc_volatility_shock_30_vs_120",
    "btc_path_efficiency_60s",
    "pm_depth_imbalance",
    "pm_vwap200_overround",
)

TRADE_PRINT_FEATURES = tuple(
    name
    for seconds in (5, 15, 30, 60)
    for name in (
        f"binance_print_return_{seconds}s_bps",
        f"binance_print_signed_share_{seconds}s",
        f"binance_print_log_quote_volume_{seconds}s",
        f"binance_print_log_trade_count_{seconds}s",
    )
)


@dataclass(frozen=True)
class TournamentWindows:
    fit_start: datetime
    fit_end: datetime
    calibration_end: datetime
    risk_end: datetime
    policy_end: datetime
    holdout_end: datetime
    oi_fit_start: datetime
    oi_fit_end: datetime
    oi_calibration_end: datetime
    print_fit_start: datetime
    print_fit_end: datetime
    print_calibration_end: datetime


@dataclass(frozen=True)
class EntryConfig:
    start_second: int
    end_second_exclusive: int
    cadence_seconds: int
    calibration_cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TournamentPolicy:
    confidence_thresholds: tuple[float, ...]
    stress_edge_thresholds: tuple[float, ...]
    wait_advantage_thresholds: tuple[float, ...]
    price_bucket_edges: tuple[float, ...]
    rolling_fold_days: int
    minimum_policy_trades: int
    minimum_accuracy: float
    minimum_profit_factor: float
    minimum_profitable_fold_ratio: float
    minimum_payoff_ratio: float
    minimum_side_trades: int
    maximum_daily_pnl_concentration: float
    minimum_active_days: int
    conservative_z_score: float
    coverage_targets: tuple[float, ...]


@dataclass(frozen=True)
class TournamentPaths:
    development_config: Path
    holdout_core_config: Path
    capacity_evidence: Path
    external_evidence: Path
    runs: Path
    committed_results: Path
    incumbent_artifact: Path
    l2_source_sql: Path
    trade_print_source_sql: Path


@dataclass(frozen=True)
class TournamentConfig:
    source_path: Path
    package_root: Path
    profile: str
    random_seed: int
    windows: TournamentWindows
    entry: EntryConfig
    execution: ExecutionConfig
    policy: TournamentPolicy
    paths: TournamentPaths


@dataclass
class OutcomeModel:
    feature_names: tuple[str, ...]
    estimator: HistGradientBoostingClassifier
    calibrator: LogisticRegression
    fit_window: tuple[datetime, datetime]
    calibration_window: tuple[datetime, datetime]


@dataclass
class CorrectnessModel:
    estimator: LogisticRegression
    regime_features: tuple[str, ...]
    cell_names: tuple[str, ...]
    price_bucket_edges: tuple[float, ...]
    penalties: dict[str, float]
    global_penalty: float


@dataclass
class FrozenCandidate:
    name: str
    outcome: OutcomeModel
    correctness: CorrectnessModel
    feature_names: tuple[str, ...]
    eligibility_features: tuple[str, ...]
    score_mode: str
    risk_model: HistGradientBoostingClassifier | None = None
    risk_feature_names: tuple[str, ...] = ()
    wait_model: HistGradientBoostingRegressor | None = None
    wait_feature_names: tuple[str, ...] = ()
    policy: dict[str, Any] | None = None


def load_config(path: Path) -> TournamentConfig:
    source = path.resolve()
    package_root = source.parent.parent
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    training = raw["training"]
    if training.get("paper_only") is not True or training.get("live_capital_allowed") is not False:
        raise ValueError("middle-market tournament must remain offline and paper-only")
    paths = raw["paths"]
    config = TournamentConfig(
        source_path=source,
        package_root=package_root,
        profile=str(training["profile"]),
        random_seed=int(training["random_seed"]),
        windows=TournamentWindows(**{name: _utc(value) for name, value in raw["windows"].items()}),
        entry=EntryConfig(
            start_second=int(raw["entry"]["start_second"]),
            end_second_exclusive=int(raw["entry"]["end_second_exclusive"]),
            cadence_seconds=int(raw["entry"]["cadence_seconds"]),
            calibration_cells=tuple(
                (int(values[0]), int(values[1])) for values in raw["entry"]["calibration_cells"]
            ),
        ),
        execution=ExecutionConfig(
            quantities=tuple(int(value) for value in raw["execution"]["quantities"]),
            freshness_seconds=int(raw["execution"]["freshness_seconds"]),
            maximum_depth_participation=float(raw["execution"]["maximum_depth_participation"]),
            execution_reserve_per_share=float(raw["execution"]["execution_reserve_per_share"]),
            stress_slippage_per_share=float(raw["execution"]["stress_slippage_per_share"]),
        ),
        policy=TournamentPolicy(
            confidence_thresholds=tuple(
                float(value) for value in raw["policy"]["confidence_thresholds"]
            ),
            stress_edge_thresholds=tuple(
                float(value) for value in raw["policy"]["stress_edge_thresholds"]
            ),
            wait_advantage_thresholds=tuple(
                float(value) for value in raw["policy"]["wait_advantage_thresholds"]
            ),
            price_bucket_edges=tuple(float(value) for value in raw["policy"]["price_bucket_edges"]),
            rolling_fold_days=int(raw["policy"]["rolling_fold_days"]),
            minimum_policy_trades=int(raw["policy"]["minimum_policy_trades"]),
            minimum_accuracy=float(raw["policy"]["minimum_accuracy"]),
            minimum_profit_factor=float(raw["policy"]["minimum_profit_factor"]),
            minimum_profitable_fold_ratio=float(raw["policy"]["minimum_profitable_fold_ratio"]),
            minimum_payoff_ratio=float(raw["policy"]["minimum_payoff_ratio"]),
            minimum_side_trades=int(raw["policy"]["minimum_side_trades"]),
            maximum_daily_pnl_concentration=float(raw["policy"]["maximum_daily_pnl_concentration"]),
            minimum_active_days=int(raw["policy"]["minimum_active_days"]),
            conservative_z_score=float(raw["policy"]["conservative_z_score"]),
            coverage_targets=tuple(float(value) for value in raw["policy"]["coverage_targets"]),
        ),
        paths=TournamentPaths(
            **{name: _path(package_root, value) for name, value in paths.items()}
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: TournamentConfig) -> None:
    values = list(asdict(config.windows).values())[:6]
    if values != sorted(values) or len(set(values)) != len(values):
        raise ValueError("primary tournament windows must be strictly chronological")
    if (config.entry.start_second, config.entry.end_second_exclusive) != (90, 180):
        raise ValueError("tournament entry window must remain 90-179 seconds")
    if config.entry.calibration_cells != ((90, 120), (120, 150), (150, 180)):
        raise ValueError("middle-market calibration cells changed")
    if config.execution.quantities != VWAP_QUANTITIES:
        raise ValueError("tournament requires the exact VWAP 5-200 contract")
    if config.execution.maximum_depth_participation != 0.25:
        raise ValueError("maximum depth participation must remain 25 percent")
    if not config.paths.development_config.is_file():
        raise FileNotFoundError(config.paths.development_config)
    if not config.paths.holdout_core_config.is_file():
        raise FileNotFoundError(config.paths.holdout_core_config)
    if not config.paths.incumbent_artifact.is_file():
        raise FileNotFoundError(config.paths.incumbent_artifact)
    for path in (config.paths.l2_source_sql, config.paths.trade_print_source_sql):
        if not path.is_file():
            raise FileNotFoundError(path)


def run_tournament(config: TournamentConfig) -> tuple[Path, dict[str, Any]]:
    print("load: development middle-market frame", flush=True)
    frame = _load_development_frame(config)
    print("load: new post-August-2 holdout frame", flush=True)
    holdout, capacity_manifest = _load_holdout_frame(config)
    combined = pl.concat((frame, holdout), how="diagonal_relaxed").sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    print("attach: OI and trade-print evidence", flush=True)
    combined, external_manifest = _attach_external_features(combined, config)
    frame = _block(combined, config.windows.fit_start, config.windows.policy_end)
    holdout = _block(combined, config.windows.policy_end, config.windows.holdout_end)
    if holdout.is_empty():
        raise RuntimeError("new holdout is empty")

    candidate_specs = (
        ("mid_core_control", PRIMARY_FEATURES, "control", "core"),
        ("mid_payoff_coverage", PRIMARY_FEATURES, "payoff", "core"),
        ("mid_failure_risk", PRIMARY_FEATURES, "failure_risk", "core"),
        ("mid_direction_qualified", PRIMARY_FEATURES, "direction", "core"),
        ("mid_regime_calibrated", PRIMARY_FEATURES, "regime", "core"),
        ("mid_wait_value", PRIMARY_FEATURES, "wait", "core"),
        (
            "mid_chainlink",
            (*PRIMARY_FEATURES, *CHAINLINK_FEATURES),
            "payoff",
            "core",
        ),
        ("mid_binance_l2", (*PRIMARY_FEATURES, *L2_FEATURES), "payoff", "core"),
        (
            "mid_open_interest",
            (*PRIMARY_FEATURES, *BINANCE_OI_FEATURES),
            "payoff",
            "oi",
        ),
        (
            "mid_trade_prints",
            (*PRIMARY_FEATURES, *TRADE_PRINT_FEATURES),
            "payoff",
            "prints",
        ),
    )
    candidates: dict[str, FrozenCandidate] = {}
    development_results: dict[str, Any] = {}
    policy_frames: dict[str, pl.DataFrame] = {}
    for index, (name, features, mode, window_family) in enumerate(candidate_specs):
        print(f"train: {name}", flush=True)
        candidate, scored_policy, result = _train_candidate(
            name,
            tuple(features),
            mode,
            window_family,
            frame,
            config,
            seed=config.random_seed + index * 17,
        )
        candidates[name] = candidate
        policy_frames[name] = scored_policy
        development_results[name] = result

    qualified = [
        (
            development_results[name]["selected_policy_metrics"]["trades"],
            development_results[name]["selected_policy_metrics"]["stress_net_pnl"],
            name,
        )
        for name in candidates
        if development_results[name]["qualified"]
    ]
    provisional_champion = max(qualified)[2] if qualified else None
    print(f"frozen provisional champion: {provisional_champion or 'none'}", flush=True)

    print("evaluation: score all frozen candidates on new holdout", flush=True)
    holdout_results: dict[str, Any] = {}
    holdout_ledgers: dict[str, pl.DataFrame] = {}
    for name, candidate in candidates.items():
        scored = _score_candidate(holdout, candidate, config)
        selected = _apply_frozen_policy(scored, candidate.policy or {})
        holdout_ledgers[name] = selected
        holdout_results[name] = _evaluation_bundle(
            selected,
            scored,
            config,
            config.windows.policy_end,
            config.windows.holdout_end,
        )
    incumbent = {
        "artifact": str(config.paths.incumbent_artifact),
        "middle_policy_enabled": False,
        "reason": "The frozen incumbent disables its 90-179-second policy.",
        "metrics": policy_metrics(holdout.head(0), config, quantity=5),
    }

    qualification = {
        "provisional_champion": provisional_champion,
        "champion_holdout_passed": (
            _qualification_checks(
                holdout_ledgers[provisional_champion],
                config,
                config.windows.policy_end,
                config.windows.holdout_end,
            )[0]
            if provisional_champion is not None
            else False
        ),
        "decision": (
            "development_qualified"
            if provisional_champion is not None
            and _qualification_checks(
                holdout_ledgers[provisional_champion],
                config,
                config.windows.policy_end,
                config.windows.holdout_end,
            )[0]
            else "no_qualified_model"
        ),
        "post_hoc_holdout_selection_allowed": False,
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
        "candidates": candidates,
        "provisional_champion": provisional_champion,
        "runtime_exported": False,
        "production_qualified": False,
    }
    artifact_path = temporary / "tournament.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    ledger_dir = temporary / "holdout-ledgers"
    ledger_dir.mkdir()
    for name, ledger in holdout_ledgers.items():
        _ledger_columns(ledger).write_parquet(ledger_dir / f"{name}.parquet", compression="zstd")

    strict_holdout_markets = holdout["market_id"].n_unique()
    scheduled_holdout_markets = int(
        pl.scan_parquet(config.paths.capacity_evidence / "holdout.parquet")
        .select(pl.col("market_id").n_unique())
        .collect()
        .item()
    )
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
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "configuration": {
            "path": str(config.source_path),
            "sha256": file_sha256(config.source_path),
            "windows": {name: value.isoformat() for name, value in asdict(config.windows).items()},
            "entry": asdict(config.entry),
            "execution": asdict(config.execution),
            "policy": asdict(config.policy),
        },
        "data": {
            "development_rows": frame.height,
            "development_markets": frame["market_id"].n_unique(),
            "holdout_rows": holdout.height,
            "holdout_strict_markets": strict_holdout_markets,
            "holdout_scheduled_markets": scheduled_holdout_markets,
            "holdout_strict_data_coverage": (
                strict_holdout_markets / scheduled_holdout_markets
                if scheduled_holdout_markets
                else 0.0
            ),
            "capacity_manifest": capacity_manifest,
            "external_manifest": external_manifest,
            "twap": {
                "used": False,
                "reason": "No persisted settlement-aligned historical TWAP was available.",
            },
        },
        "development_results": development_results,
        "holdout_results": holdout_results,
        "incumbent_control": incumbent,
        "qualification": qualification,
        "model_artifact": {
            "path": "tournament.joblib",
            "sha256": file_sha256(artifact_path),
        },
        "limitations": [
            "All candidates are offline development artifacts and are not runtime exported.",
            "The frozen incumbent has no enabled middle-market policy and therefore makes no common-window trades.",
            "Open-interest history starts July 3 and trade-print history starts July 22, so those candidates use shorter causal fitting windows.",
            "Projected PnL assumes recorded ask VWAP is fillable and does not model queue position.",
        ],
    }
    _write_json(temporary / "metrics.json", metrics)
    (temporary / "report.md").write_text(_render_report(metrics))
    (temporary / "tournament.sha256").write_text(file_sha256(artifact_path) + "\n")
    config.paths.committed_results.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    return final, metrics


def _load_development_frame(config: TournamentConfig) -> pl.DataFrame:
    development = load_development_config(config.paths.development_config)
    manifest = json.loads((development.paths.capacity_evidence / "manifest.json").read_text())
    frame = load_training_frame(development, manifest)
    return _middle_rows(frame, config)


def _load_holdout_frame(
    config: TournamentConfig,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    core_config = load_core_config(config.paths.holdout_core_config)
    core_paths = (
        core_config.paths.development_feature_data,
        core_config.paths.holdout_feature_data,
    )
    for path in core_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"holdout core features are missing; run core extraction first: {path}"
            )
    core = pl.read_parquet(core_paths).with_columns(
        pl.col("oracle_return_from_window_open_bps")
        .is_not_null()
        .cast(pl.Int8)
        .alias("early_oracle_eligible")
    )
    destination = config.paths.capacity_evidence
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "holdout.parquet"
    manifest_path = destination / "manifest.json"
    query_path = config.package_root / "sql" / "btc-continuous-edge-capacity-evidence.sql"
    contract = {
        "schema_version": "btc-middle-market-capacity-holdout-v1",
        "start": config.windows.policy_end.isoformat(),
        "end": config.windows.holdout_end.isoformat(),
        "query_sha256": file_sha256(query_path),
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("contract") != contract:
            raise RuntimeError("middle-market capacity cache contract changed")
        if not path.is_file() or manifest.get("sha256") != file_sha256(path):
            raise RuntimeError("middle-market capacity cache hash changed")
    else:
        connection = database_connection()
        configure_read_only_connection(connection)
        try:
            _require_complete_hours(
                connection, config.windows.policy_end, config.windows.holdout_end
            )
            rows = _extract_capacity_partition(
                connection,
                query_path.read_text(),
                path,
                config.windows.policy_end,
                config.windows.holdout_end,
            )
        finally:
            connection.close()
        manifest = {
            "contract": contract,
            "rows": rows,
            "sha256": file_sha256(path),
            "created_at": datetime.now(UTC).isoformat(),
        }
        _write_json(manifest_path, manifest)
    evidence = pl.read_parquet(path)
    evidence = _strict_capacity(evidence, config)
    keys = ("market_id", "window_start", "observed_at", "seconds_elapsed", "label_up")
    frame = core.join(evidence, on=list(keys), how="inner", validate="1:1")
    frame = attach_book_features(frame)
    frame = frame.with_columns(
        *(pl.lit(None).cast(pl.Float64).alias(name) for name in (*CHAINLINK_FEATURES, *L2_FEATURES))
    )
    return _middle_rows(frame, config), manifest


def _strict_capacity(frame: pl.DataFrame, config: TournamentConfig) -> pl.DataFrame:
    clean = (pl.col("quality_flags") & 63) == 0
    fresh = (
        pl.col("up_provider_received_at").is_not_null()
        & pl.col("down_provider_received_at").is_not_null()
        & (pl.col("up_provider_received_at") <= pl.col("observed_at"))
        & (pl.col("down_provider_received_at") <= pl.col("observed_at"))
        & (
            pl.col("up_provider_received_at")
            >= pl.col("observed_at") - pl.duration(seconds=config.execution.freshness_seconds)
        )
        & (
            pl.col("down_provider_received_at")
            >= pl.col("observed_at") - pl.duration(seconds=config.execution.freshness_seconds)
        )
    )
    curve = pl.all_horizontal(
        [pl.col(name).is_not_null() & pl.col(name).is_finite() for name in BOOK_RAW_FEATURES]
    )
    return frame.filter(
        clean
        & fresh
        & curve
        & (pl.col("up_ask_depth") >= 200 / config.execution.maximum_depth_participation)
        & (pl.col("down_ask_depth") >= 200 / config.execution.maximum_depth_participation)
    )


def _middle_rows(frame: pl.DataFrame, config: TournamentConfig) -> pl.DataFrame:
    return (
        frame.filter(
            pl.col("seconds_elapsed").is_between(
                config.entry.start_second,
                config.entry.end_second_exclusive - 1,
                closed="both",
            )
            & (pl.col("seconds_elapsed") % config.entry.cadence_seconds == 0)
        )
        .with_columns(
            pl.lit("mid").alias("time_band"),
            pl.when(pl.col("seconds_elapsed") < 120)
            .then(pl.lit("90-119"))
            .when(pl.col("seconds_elapsed") < 150)
            .then(pl.lit("120-149"))
            .otherwise(pl.lit("150-179"))
            .alias("middle_cell"),
        )
        .sort(["window_start", "market_id", "seconds_elapsed", "observed_at"])
    )


def _attach_external_features(
    frame: pl.DataFrame,
    config: TournamentConfig,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    cache = config.paths.external_evidence
    cache.mkdir(parents=True, exist_ok=True)
    oi_path = cache / "open-interest.parquet"
    candle_path = cache / "holdout-chainlink-candles.parquet"
    l2_path = cache / "holdout-l2.parquet"
    trade_path = cache / "trade-print-seconds.parquet"
    manifest_path = cache / "manifest.json"
    contract = {
        "schema_version": "btc-middle-market-external-evidence-v1",
        "start": config.windows.oi_fit_start.isoformat(),
        "end": config.windows.holdout_end.isoformat(),
        "candle_sql_sha256": file_sha256(
            config.package_root / "sql" / "btc-chainlink-one-minute-candles-source.sql"
        ),
        "oi_sql_sha256": file_sha256(
            config.package_root / "sql" / "btc-binance-five-minute-open-interest-source.sql"
        ),
        "l2_sql_sha256": file_sha256(config.paths.l2_source_sql),
        "trade_sql_sha256": file_sha256(config.paths.trade_print_source_sql),
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("contract") != contract:
            raise RuntimeError("external evidence cache contract changed")
        for name, path in (
            ("oi", oi_path),
            ("candles", candle_path),
            ("l2", l2_path),
            ("prints", trade_path),
        ):
            if not path.is_file() or manifest["files"][name]["sha256"] != file_sha256(path):
                raise RuntimeError(f"external evidence cache changed: {name}")
    else:
        connection = database_connection()
        configure_read_only_connection(connection)
        try:
            oi = _query_frame(
                connection,
                (
                    config.package_root / "sql" / "btc-binance-five-minute-open-interest-source.sql"
                ).read_text(),
                {
                    "range_start": config.windows.oi_fit_start,
                    "range_end": config.windows.holdout_end,
                    "history_minutes": 65,
                    "open_interest_symbol": "BTCUSDT",
                },
                cursor_name="middle_market_open_interest",
            )
            oi.write_parquet(oi_path, compression="zstd")
            candles = _query_frame(
                connection,
                (
                    config.package_root / "sql" / "btc-chainlink-one-minute-candles-source.sql"
                ).read_text(),
                {
                    "range_start": config.windows.policy_end,
                    "range_end": config.windows.holdout_end,
                    "history_minutes": 61,
                    "candle_symbol": "BTCUSD",
                },
                cursor_name="middle_market_chainlink_candles",
            )
            candles.write_parquet(candle_path, compression="zstd")
            l2 = _query_frame(
                connection,
                config.paths.l2_source_sql.read_text(),
                {
                    "batch_start": config.windows.policy_end - timedelta(seconds=2),
                    "batch_end": config.windows.holdout_end,
                },
                cursor_name="middle_market_holdout_l2",
            )
            l2.write_parquet(l2_path, compression="zstd")
            prints = _extract_trade_print_seconds(connection, config)
            prints.write_parquet(trade_path, compression="zstd")
        finally:
            connection.close()
        manifest = {
            "contract": contract,
            "created_at": datetime.now(UTC).isoformat(),
            "files": {
                "oi": _file_record(oi_path),
                "candles": _file_record(candle_path),
                "l2": _file_record(l2_path),
                "prints": _file_record(trade_path),
            },
        }
        _write_json(manifest_path, manifest)

    oi = pl.read_parquet(oi_path)
    joined = _attach_open_interest_features(frame, oi, max_age_seconds=300)
    oi_keys = ("market_id", "window_start", "observed_at", "seconds_elapsed", "label_up")
    oi_frame = joined.select(*oi_keys, *BINANCE_OI_FEATURES)
    output = frame.join(oi_frame, on=list(oi_keys), how="left", validate="1:1")

    holdout = _block(output, config.windows.policy_end, config.windows.holdout_end)
    holdout_without_optional = holdout.drop(*CHAINLINK_FEATURES, *L2_FEATURES)
    qualified_chainlink = _attach_candle_features(
        holdout_without_optional,
        pl.read_parquet(candle_path),
        max_age_seconds=60,
    ).select(*oi_keys, *CHAINLINK_FEATURES)
    holdout_with_chainlink = holdout_without_optional.join(
        qualified_chainlink,
        on=list(oi_keys),
        how="left",
        validate="1:1",
    )
    qualified_l2 = join_qualified_l2(
        holdout_with_chainlink,
        pl.read_parquet(l2_path),
    ).select(*oi_keys, *L2_FEATURES)
    holdout_with_l2 = holdout_with_chainlink.join(
        qualified_l2, on=list(oi_keys), how="left", validate="1:1"
    )
    development = _block(output, config.windows.fit_start, config.windows.policy_end)
    output = pl.concat((development, holdout_with_l2), how="diagonal_relaxed")

    print_features = _derive_trade_print_features(pl.read_parquet(trade_path))
    output = _join_trade_print_features(output, print_features)
    return output.sort(["window_start", "market_id", "seconds_elapsed"]), manifest


def _extract_trade_print_seconds(
    connection: Any,
    config: TournamentConfig,
) -> pl.DataFrame:
    query = config.paths.trade_print_source_sql.read_text()
    pieces: list[pl.DataFrame] = []
    cursor = config.windows.print_fit_start
    while cursor < config.windows.holdout_end:
        end = min(cursor + timedelta(days=1), config.windows.holdout_end)
        piece = _query_frame(
            connection,
            query,
            {"batch_start": cursor, "batch_end": end},
            cursor_name=f"middle_prints_{cursor:%Y%m%d}",
        )
        pieces.append(piece)
        cursor = end
    return pl.concat(pieces, how="vertical_relaxed").sort("second_start")


def _derive_trade_print_features(frame: pl.DataFrame) -> pl.DataFrame:
    ordered = frame.sort("second_start").with_columns(
        (pl.col("second_start") + pl.duration(seconds=1)).alias("available_at")
    )
    expressions: list[pl.Expr] = []
    for seconds in (5, 15, 30, 60):
        exact = pl.col("second_start") - pl.col("second_start").shift(seconds) == pl.duration(
            seconds=seconds
        )
        quote = pl.col("quote_volume").rolling_sum(seconds)
        signed = pl.col("signed_taker_quote_volume").rolling_sum(seconds)
        count = pl.col("trade_count").rolling_sum(seconds)
        expressions.extend(
            (
                pl.when(exact)
                .then((pl.col("trade_vwap") / pl.col("trade_vwap").shift(seconds)).log() * 10_000)
                .otherwise(None)
                .alias(f"binance_print_return_{seconds}s_bps"),
                pl.when(exact)
                .then(signed / (quote + 1e-9))
                .otherwise(None)
                .alias(f"binance_print_signed_share_{seconds}s"),
                pl.when(exact)
                .then(quote.log1p())
                .otherwise(None)
                .alias(f"binance_print_log_quote_volume_{seconds}s"),
                pl.when(exact)
                .then(count.log1p())
                .otherwise(None)
                .alias(f"binance_print_log_trade_count_{seconds}s"),
            )
        )
    return ordered.with_columns(*expressions).select("available_at", *TRADE_PRINT_FEATURES)


def _join_trade_print_features(
    frame: pl.DataFrame,
    features: pl.DataFrame,
) -> pl.DataFrame:
    joined = (
        frame.with_row_index("_row")
        .sort("observed_at")
        .join_asof(
            features.sort("available_at"),
            left_on="observed_at",
            right_on="available_at",
            strategy="backward",
        )
    )
    fresh = pl.col("available_at").is_not_null() & (
        pl.col("observed_at") - pl.col("available_at") <= pl.duration(seconds=2)
    )
    return (
        joined.with_columns(
            *(
                pl.when(fresh).then(pl.col(name)).otherwise(None).alias(name)
                for name in TRADE_PRINT_FEATURES
            )
        )
        .sort("_row")
        .drop("_row", "available_at")
    )


def _train_candidate(
    name: str,
    features: tuple[str, ...],
    mode: str,
    window_family: str,
    frame: pl.DataFrame,
    config: TournamentConfig,
    *,
    seed: int,
) -> tuple[FrozenCandidate, pl.DataFrame, dict[str, Any]]:
    fit_start, fit_end, calibration_end = _candidate_windows(config, window_family)
    minimum_fit_markets = 250 if window_family == "prints" else 500
    minimum_calibration_markets = 100 if window_family == "prints" else 200
    eligibility_features = tuple(name for name in features if name not in PRIMARY_FEATURES)
    eligible = _candidate_eligible_frame(frame, eligibility_features)
    outcome = _fit_outcome(
        _block(eligible, fit_start, fit_end),
        _block(eligible, fit_end, calibration_end),
        features,
        seed,
        (fit_start, fit_end),
        (fit_end, calibration_end),
        minimum_fit_markets=minimum_fit_markets,
        minimum_calibration_markets=minimum_calibration_markets,
    )
    calibration = _attach_decision_scores(
        _block(eligible, fit_end, calibration_end), outcome, config
    )
    correctness = _fit_correctness(
        calibration,
        config,
        regime=(mode == "regime"),
        seed=seed + 1,
    )
    candidate = FrozenCandidate(
        name=name,
        outcome=outcome,
        correctness=correctness,
        feature_names=features,
        eligibility_features=eligibility_features,
        score_mode=mode,
    )
    if mode == "failure_risk":
        risk = _score_base(
            _block(eligible, calibration_end, config.windows.risk_end),
            candidate,
            config,
        )
        candidate.risk_feature_names = _variable_features(risk, RISK_FEATURES)
        candidate.risk_model = _fit_binary_model(
            risk,
            candidate.risk_feature_names,
            risk["direction_correct"].to_numpy().astype(np.int8),
            seed + 2,
        )
    if mode == "wait":
        risk = _score_base(
            _block(eligible, calibration_end, config.windows.risk_end),
            candidate,
            config,
        )
        wait_frame = _wait_training_frame(risk, config)
        candidate.wait_feature_names = _variable_features(wait_frame, RISK_FEATURES)
        candidate.wait_model = HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=5.0,
            random_state=seed + 3,
            early_stopping=False,
        )
        candidate.wait_model.fit(
            _matrix(wait_frame, candidate.wait_feature_names),
            wait_frame["wait_advantage_target"].to_numpy(),
            sample_weight=market_equal_weights(wait_frame),
        )
    policy_start = calibration_end if window_family in {"oi", "prints"} else config.windows.risk_end
    scored_policy = _score_candidate(
        _block(eligible, policy_start, config.windows.policy_end),
        candidate,
        config,
    )
    frozen_policy, search = _select_policy(scored_policy, candidate, config)
    candidate.policy = frozen_policy
    selected = _apply_frozen_policy(scored_policy, frozen_policy)
    qualified, checks = _qualification_checks(
        selected,
        config,
        policy_start,
        config.windows.policy_end,
    )
    result = {
        "features": list(features),
        "feature_count": len(features),
        "fit_window": [fit_start.isoformat(), fit_end.isoformat()],
        "calibration_window": [fit_end.isoformat(), calibration_end.isoformat()],
        "coverage_guard": {
            "minimum_fit_markets": minimum_fit_markets,
            "minimum_calibration_markets": minimum_calibration_markets,
        },
        "fit_rows": _block(eligible, fit_start, fit_end).height,
        "fit_markets": _block(eligible, fit_start, fit_end)["market_id"].n_unique(),
        "policy_rows": scored_policy.height,
        "policy_markets": scored_policy["market_id"].n_unique(),
        "selected_policy": frozen_policy,
        "selected_policy_metrics": policy_metrics(selected, config, quantity=5),
        "capacity": {
            str(quantity): policy_metrics(selected, config, quantity=quantity)
            for quantity in config.execution.quantities
        },
        "search": search,
        "qualified": qualified,
        "qualification_checks": checks,
    }
    return candidate, scored_policy, result


def _candidate_windows(
    config: TournamentConfig,
    family: str,
) -> tuple[datetime, datetime, datetime]:
    if family == "oi":
        return (
            config.windows.oi_fit_start,
            config.windows.oi_fit_end,
            config.windows.oi_calibration_end,
        )
    if family == "prints":
        return (
            config.windows.print_fit_start,
            config.windows.print_fit_end,
            config.windows.print_calibration_end,
        )
    return (
        config.windows.fit_start,
        config.windows.fit_end,
        config.windows.calibration_end,
    )


def _fit_outcome(
    fit: pl.DataFrame,
    calibration: pl.DataFrame,
    features: tuple[str, ...],
    seed: int,
    fit_window: tuple[datetime, datetime],
    calibration_window: tuple[datetime, datetime],
    *,
    minimum_fit_markets: int,
    minimum_calibration_markets: int,
) -> OutcomeModel:
    if (
        fit["market_id"].n_unique() < minimum_fit_markets
        or calibration["market_id"].n_unique() < minimum_calibration_markets
    ):
        raise RuntimeError("candidate lacks minimum chronological outcome coverage")
    feature_names = _variable_features(fit, features)
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=240,
        max_leaf_nodes=31,
        min_samples_leaf=140,
        l2_regularization=4.0,
        random_state=seed,
        early_stopping=False,
    )
    estimator.fit(
        _matrix(fit, feature_names),
        fit["label_up"].to_numpy(),
        sample_weight=market_equal_weights(fit),
    )
    raw = np.clip(
        estimator.predict_proba(_matrix(calibration, feature_names))[:, 1], 1e-6, 1 - 1e-6
    )
    calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    calibrator.fit(
        _logit(raw).reshape(-1, 1),
        calibration["label_up"].to_numpy(),
        sample_weight=market_equal_weights(calibration),
    )
    return OutcomeModel(feature_names, estimator, calibrator, fit_window, calibration_window)


def _attach_decision_scores(
    frame: pl.DataFrame,
    model: OutcomeModel,
    config: TournamentConfig,
) -> pl.DataFrame:
    raw = np.clip(
        model.estimator.predict_proba(_matrix(frame, model.feature_names))[:, 1], 1e-6, 1 - 1e-6
    )
    probability_up = model.calibrator.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
    predicted_up = probability_up >= 0.5
    probability_selected = np.where(predicted_up, probability_up, 1.0 - probability_up)
    price = np.where(
        predicted_up,
        frame["up_ask_vwap_5"].to_numpy(),
        frame["down_ask_vwap_5"].to_numpy(),
    )
    fee_rate = frame["fee_rate"].to_numpy()
    cost = price + fee_rate * price * (1.0 - price) + config.execution.execution_reserve_per_share
    buckets = _bucket_indices(cost, config.policy.price_bucket_edges)
    return frame.with_columns(
        pl.Series("probability_up", probability_up),
        pl.Series("predicted_up", predicted_up),
        pl.Series("probability_selected", probability_selected),
        pl.Series("selected_cost_5", cost),
        pl.Series("selected_edge_5", probability_selected - cost),
        pl.Series(
            "direction_correct",
            predicted_up == frame["label_up"].to_numpy().astype(bool),
        ),
        pl.Series("price_bucket_index", buckets.astype(np.int8)),
    )


def _fit_correctness(
    frame: pl.DataFrame,
    config: TournamentConfig,
    *,
    regime: bool,
    seed: int,
) -> CorrectnessModel:
    regime_features = REGIME_FEATURES if regime else ()
    matrix = _correctness_matrix(
        frame,
        config,
        regime_features,
    )
    estimator = LogisticRegression(
        C=0.25,
        max_iter=600,
        random_state=seed,
    )
    labels = frame["direction_correct"].to_numpy().astype(np.int8)
    weights = market_equal_weights(frame)
    estimator.fit(matrix, labels, sample_weight=weights)
    probability = estimator.predict_proba(matrix)[:, 1]
    global_accuracy = float(np.average(labels, weights=weights))
    global_bias = abs(float(np.average(labels - probability, weights=weights)))
    markets = max(frame["market_id"].n_unique(), 1)
    global_penalty = global_bias + config.policy.conservative_z_score * math.sqrt(
        max(global_accuracy * (1.0 - global_accuracy), 0.01) / markets
    )
    penalties: dict[str, float] = {}
    for cell in ("90-119", "120-149", "150-179"):
        for side in (0, 1):
            for bucket in range(len(config.policy.price_bucket_edges) - 1):
                mask = (
                    (frame["middle_cell"].to_numpy() == cell)
                    & (frame["predicted_up"].to_numpy().astype(np.int8) == side)
                    & (frame["price_bucket_index"].to_numpy() == bucket)
                )
                key = f"{cell}:{side}:{bucket}"
                if not mask.any():
                    penalties[key] = global_penalty
                    continue
                cell_weights = weights[mask]
                accuracy = float(np.average(labels[mask], weights=cell_weights))
                bias = abs(
                    float(np.average(labels[mask] - probability[mask], weights=cell_weights))
                )
                cell_markets = frame.filter(pl.Series(mask))["market_id"].n_unique()
                raw_penalty = bias + config.policy.conservative_z_score * math.sqrt(
                    max(accuracy * (1.0 - accuracy), 0.01) / max(cell_markets, 1)
                )
                shrink = cell_markets / (cell_markets + 200.0)
                penalties[key] = shrink * raw_penalty + (1.0 - shrink) * global_penalty
    return CorrectnessModel(
        estimator,
        tuple(regime_features),
        ("90-119", "120-149", "150-179"),
        config.policy.price_bucket_edges,
        penalties,
        global_penalty,
    )


def _correctness_matrix(
    frame: pl.DataFrame,
    config: TournamentConfig,
    regime_features: tuple[str, ...],
) -> np.ndarray:
    probability = np.clip(frame["probability_selected"].to_numpy(), 1e-6, 1 - 1e-6)
    cells = frame["middle_cell"].to_numpy()
    buckets = frame["price_bucket_index"].to_numpy()
    cell_hot = np.column_stack(
        [(cells == name).astype(float) for name in ("90-119", "120-149", "150-179")]
    )
    bucket_hot = np.column_stack(
        [
            (buckets == index).astype(float)
            for index in range(len(config.policy.price_bucket_edges) - 1)
        ]
    )
    pieces = [
        _logit(probability),
        frame["predicted_up"].to_numpy().astype(float),
        cell_hot,
        bucket_hot,
        cell_hot * _logit(probability)[:, None],
        bucket_hot * _logit(probability)[:, None],
    ]
    if regime_features:
        regime = _matrix(frame, regime_features)
        regime = np.nan_to_num(regime, nan=0.0, posinf=0.0, neginf=0.0)
        pieces.append(regime)
    return np.column_stack(pieces)


def _score_base(
    frame: pl.DataFrame,
    candidate: FrozenCandidate,
    config: TournamentConfig,
) -> pl.DataFrame:
    scored = _attach_decision_scores(frame, candidate.outcome, config)
    matrix = _correctness_matrix(
        scored,
        config,
        candidate.correctness.regime_features,
    )
    correctness = candidate.correctness.estimator.predict_proba(matrix)[:, 1]
    penalties = np.asarray(
        [
            candidate.correctness.penalties.get(
                f"{cell}:{int(side)}:{bucket}",
                candidate.correctness.global_penalty,
            )
            for cell, side, bucket in zip(
                scored["middle_cell"].to_numpy(),
                scored["predicted_up"].to_numpy(),
                scored["price_bucket_index"].to_numpy(),
                strict=True,
            )
        ]
    )
    lower = np.clip(correctness - penalties, 0.0, 1.0)
    if candidate.score_mode == "control":
        lower = correctness
    return scored.with_columns(
        pl.Series("correctness_probability", correctness),
        pl.Series("correctness_uncertainty_penalty", penalties),
        pl.Series("lower_correctness_probability", lower),
        pl.Series(
            "stress_edge_lower_bound",
            lower
            - scored["selected_cost_5"].to_numpy()
            - config.execution.stress_slippage_per_share,
        ),
    )


def _score_candidate(
    frame: pl.DataFrame,
    candidate: FrozenCandidate,
    config: TournamentConfig,
) -> pl.DataFrame:
    eligible = _candidate_eligible_frame(frame, candidate.eligibility_features)
    if eligible.is_empty():
        return _empty_scored_frame(eligible)
    scored = _score_base(eligible, candidate, config)
    if candidate.risk_model is not None:
        probability = candidate.risk_model.predict_proba(
            _matrix(scored, candidate.risk_feature_names)
        )[:, 1]
        lower = np.minimum(scored["lower_correctness_probability"].to_numpy(), probability)
        scored = scored.with_columns(
            pl.Series("failure_risk_correctness_probability", probability),
            pl.Series("lower_correctness_probability", lower),
            pl.Series(
                "stress_edge_lower_bound",
                lower
                - scored["selected_cost_5"].to_numpy()
                - config.execution.stress_slippage_per_share,
            ),
        )
    if candidate.wait_model is not None:
        scored = scored.with_columns(
            pl.Series(
                "wait_advantage",
                candidate.wait_model.predict(_matrix(scored, candidate.wait_feature_names)),
            )
        )
    else:
        scored = scored.with_columns(pl.lit(0.0).alias("wait_advantage"))
    return scored


def _empty_scored_frame(frame: pl.DataFrame) -> pl.DataFrame:
    float_columns = (
        "probability_up",
        "probability_selected",
        "selected_cost_5",
        "selected_edge_5",
        "correctness_probability",
        "correctness_uncertainty_penalty",
        "lower_correctness_probability",
        "stress_edge_lower_bound",
        "wait_advantage",
    )
    return frame.with_columns(
        *(pl.lit(None).cast(pl.Float64).alias(name) for name in float_columns),
        pl.lit(None).cast(pl.Boolean).alias("predicted_up"),
        pl.lit(None).cast(pl.Boolean).alias("direction_correct"),
        pl.lit(None).cast(pl.Int8).alias("price_bucket_index"),
    )


def _fit_binary_model(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    labels: np.ndarray,
    seed: int,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=5.0,
        random_state=seed,
        early_stopping=False,
    )
    model.fit(
        _matrix(frame, features),
        labels,
        sample_weight=market_equal_weights(frame),
    )
    return model


def _wait_training_frame(
    frame: pl.DataFrame,
    config: TournamentConfig,
) -> pl.DataFrame:
    output = frame.with_columns(
        (
            pl.col("direction_correct").cast(pl.Float64)
            - pl.col("selected_cost_5")
            - config.execution.stress_slippage_per_share
        ).alias("_now_stress_edge")
    )
    future_columns: list[str] = []
    for lag in (5, 10, 15):
        name = f"_future_{lag}"
        future = output.select(
            "market_id",
            (pl.col("seconds_elapsed") - lag).alias("seconds_elapsed"),
            pl.col("_now_stress_edge").alias(name),
        )
        output = output.join(future, on=["market_id", "seconds_elapsed"], how="left")
        future_columns.append(name)
    return output.with_columns(
        (
            pl.col("_now_stress_edge")
            - pl.max_horizontal(*(pl.col(name) for name in future_columns))
        ).alias("wait_advantage_target")
    ).drop_nulls("wait_advantage_target")


def _select_policy(
    frame: pl.DataFrame,
    candidate: FrozenCandidate,
    config: TournamentConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if candidate.score_mode == "direction":
        directions: dict[str, Any] = {}
        histories: dict[str, Any] = {}
        for side_name, side_value in (("up", True), ("down", False)):
            policy, history = _select_policy_grid(
                frame.filter(pl.col("predicted_up") == side_value),
                candidate,
                config,
                minimum_trades=config.policy.minimum_side_trades,
                direction_gate=True,
            )
            directions[side_name] = policy
            histories[side_name] = history
        return {"type": "direction", "directions": directions}, histories
    policy, history = _select_policy_grid(
        frame,
        candidate,
        config,
        minimum_trades=config.policy.minimum_policy_trades,
        direction_gate=False,
    )
    return {"type": "global", **policy}, history


def _select_policy_grid(
    frame: pl.DataFrame,
    candidate: FrozenCandidate,
    config: TournamentConfig,
    *,
    minimum_trades: int,
    direction_gate: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    wait_thresholds = (
        config.policy.wait_advantage_thresholds if candidate.score_mode == "wait" else (-math.inf,)
    )
    for confidence in config.policy.confidence_thresholds:
        for edge in config.policy.stress_edge_thresholds:
            for wait in wait_thresholds:
                selected = _first_crossings(frame, confidence, edge, wait)
                metrics = policy_metrics(selected, config, quantity=5)
                folds = rolling_policy_metrics(
                    selected,
                    config,
                    frame["window_start"].min(),
                    frame["window_start"].max() + timedelta(minutes=5),
                    quantity=5,
                )
                qualified = (
                    metrics["trades"] >= minimum_trades
                    and metrics["accuracy"] >= config.policy.minimum_accuracy
                    and metrics["stress_net_pnl"] > 0
                    and (metrics["profit_factor"] or 0.0) >= config.policy.minimum_profit_factor
                    and metrics["payoff_ratio"] >= config.policy.minimum_payoff_ratio
                    and (
                        direction_gate
                        or folds["profitable_fold_ratio"]
                        >= config.policy.minimum_profitable_fold_ratio
                    )
                )
                records.append(
                    {
                        "confidence": confidence,
                        "stress_edge": edge,
                        "wait_advantage": wait,
                        "qualified": qualified,
                        "metrics": metrics,
                        "profitable_fold_ratio": folds["profitable_fold_ratio"],
                    }
                )
    qualified_records = [record for record in records if record["qualified"]]
    if qualified_records:
        chosen = max(
            qualified_records,
            key=lambda record: (
                record["metrics"]["trades"],
                record["metrics"]["stress_net_pnl"],
            ),
        )
    else:
        eligible = [record for record in records if record["metrics"]["trades"] >= 20]
        chosen = max(
            eligible or records,
            key=lambda record: (
                record["metrics"]["stress_net_pnl"],
                record["metrics"]["accuracy"],
                record["metrics"]["trades"],
            ),
        )
    policy = {
        "enabled": chosen["qualified"] if direction_gate else True,
        "confidence": chosen["confidence"],
        "stress_edge": chosen["stress_edge"],
        "wait_advantage": chosen["wait_advantage"],
        "development_qualified": chosen["qualified"],
    }
    frontier = _risk_coverage_frontier(records)
    return policy, {"attempts": len(records), "selected": chosen, "frontier": frontier}


def _risk_coverage_frontier(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        metrics = record["metrics"]
        dominated = any(
            other["metrics"]["trades"] >= metrics["trades"]
            and other["metrics"]["stress_expectancy_per_trade"]
            >= metrics["stress_expectancy_per_trade"]
            and (
                other["metrics"]["trades"] > metrics["trades"]
                or other["metrics"]["stress_expectancy_per_trade"]
                > metrics["stress_expectancy_per_trade"]
            )
            for other in records
        )
        if not dominated:
            output.append(
                {
                    "confidence": record["confidence"],
                    "stress_edge": record["stress_edge"],
                    "wait_advantage": record["wait_advantage"],
                    "trades": metrics["trades"],
                    "accuracy": metrics["accuracy"],
                    "stress_expectancy_per_trade": metrics["stress_expectancy_per_trade"],
                }
            )
    return sorted(output, key=lambda row: row["trades"], reverse=True)[:25]


def _first_crossings(
    frame: pl.DataFrame,
    confidence: float,
    stress_edge: float,
    wait_advantage: float,
) -> pl.DataFrame:
    mask = (
        (frame["lower_correctness_probability"].to_numpy() >= confidence)
        & (frame["stress_edge_lower_bound"].to_numpy() >= stress_edge)
        & (frame["wait_advantage"].to_numpy() >= wait_advantage)
    )
    indices = np.flatnonzero(mask)
    if not len(indices):
        return frame.head(0)
    markets = frame["market_id"].to_numpy()
    _, first = np.unique(markets[indices], return_index=True)
    return frame[indices[np.sort(first)].tolist()]


def _apply_frozen_policy(frame: pl.DataFrame, policy: dict[str, Any]) -> pl.DataFrame:
    if not policy:
        return frame.head(0)
    if policy.get("type") == "direction":
        pieces: list[pl.DataFrame] = []
        for name, side in (("up", True), ("down", False)):
            values = policy["directions"][name]
            if not values.get("enabled"):
                continue
            pieces.append(
                _first_crossings(
                    frame.filter(pl.col("predicted_up") == side),
                    values["confidence"],
                    values["stress_edge"],
                    values["wait_advantage"],
                )
            )
        if not pieces:
            return frame.head(0)
        combined = pl.concat(pieces, how="vertical").sort(
            ["market_id", "seconds_elapsed", "observed_at"]
        )
        return combined.unique(subset="market_id", keep="first", maintain_order=True)
    return _first_crossings(
        frame,
        policy["confidence"],
        policy["stress_edge"],
        policy["wait_advantage"],
    )


def _qualification_checks(
    selected: pl.DataFrame,
    config: TournamentConfig,
    start: datetime,
    end: datetime,
) -> tuple[bool, dict[str, bool]]:
    metrics = policy_metrics(selected, config, quantity=5)
    q10 = policy_metrics(selected, config, quantity=10)
    q20 = policy_metrics(selected, config, quantity=20)
    folds = rolling_policy_metrics(selected, config, start, end, quantity=5)
    side_metrics = {
        side: policy_metrics(selected.filter(pl.col("predicted_up") == value), config, quantity=5)
        for side, value in (("up", True), ("down", False))
    }
    bucket_metrics = policy_metrics_by_price_bucket(selected, config, quantity=5)
    checks = {
        "minimum_trades": metrics["trades"] >= config.policy.minimum_policy_trades,
        "minimum_accuracy": metrics["accuracy"] >= config.policy.minimum_accuracy,
        "positive_stress_pnl": metrics["stress_net_pnl"] > 0,
        "minimum_profit_factor": (metrics["profit_factor"] or 0.0)
        >= config.policy.minimum_profit_factor,
        "minimum_payoff_ratio": metrics["payoff_ratio"] >= config.policy.minimum_payoff_ratio,
        "minimum_active_days": metrics["active_days"] >= config.policy.minimum_active_days,
        "daily_concentration": metrics["daily_pnl_concentration"]
        <= config.policy.maximum_daily_pnl_concentration,
        "profitable_fold_ratio": folds["profitable_fold_ratio"]
        >= config.policy.minimum_profitable_fold_ratio,
        "positive_q10_stress": q10["stress_net_pnl"] > 0,
        "positive_q20_stress": q20["stress_net_pnl"] > 0,
        "enabled_directions_positive": all(
            row["stress_net_pnl"] > 0 for row in side_metrics.values() if row["trades"] > 0
        ),
        "no_negative_price_bucket": all(
            row["stress_net_pnl"] >= 0 for row in bucket_metrics.values() if row["trades"] > 0
        ),
    }
    return all(checks.values()), checks


def _evaluation_bundle(
    selected: pl.DataFrame,
    scored: pl.DataFrame,
    config: TournamentConfig,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    base = policy_metrics(selected, config, quantity=5)
    strict_markets = scored["market_id"].n_unique()
    base["strict_market_coverage"] = (
        selected["market_id"].n_unique() / strict_markets if strict_markets else 0.0
    )
    probability = scored["probability_up"].to_numpy()
    labels = scored["label_up"].to_numpy()
    probability_metrics = {
        "rows": scored.height,
        "markets": strict_markets,
        "accuracy": float(np.mean((probability >= 0.5) == labels.astype(bool)))
        if scored.height
        else 0.0,
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])) if scored.height else None,
        "brier": float(brier_score_loss(labels, probability)) if scored.height else None,
        "bias": float(np.mean(probability - labels)) if scored.height else None,
    }
    return {
        "vwap5": base,
        "capacity": {
            str(quantity): policy_metrics(selected, config, quantity=quantity)
            for quantity in config.execution.quantities
        },
        "probability": probability_metrics,
        "cells": {
            name: policy_metrics(selected.filter(pl.col("middle_cell") == name), config, quantity=5)
            for name in ("90-119", "120-149", "150-179")
        },
        "directions": {
            name: policy_metrics(
                selected.filter(pl.col("predicted_up") == value), config, quantity=5
            )
            for name, value in (("up", True), ("down", False))
        },
        "price_buckets": policy_metrics_by_price_bucket(selected, config, quantity=5),
        "rolling_folds": rolling_policy_metrics(selected, config, start, end, quantity=5),
        "qualification": _qualification_checks(selected, config, start, end)[1],
    }


def _ledger_columns(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "middle_cell",
        "label_up",
        "predicted_up",
        "probability_up",
        "probability_selected",
        "correctness_probability",
        "lower_correctness_probability",
        "stress_edge_lower_bound",
        "selected_cost_5",
        "price_bucket_index",
        "wait_advantage",
        *BOOK_RAW_FEATURES,
        "fee_rate",
    ]
    return frame.select(*(name for name in columns if name in frame.columns))


def _render_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# BTC Five-Minute Middle-Market Payoff Tournament",
        "",
        f"Provisional champion: `{metrics['qualification']['provisional_champion'] or 'none'}`",
        "",
        f"Qualification decision: **{metrics['qualification']['decision']}**",
        "",
        "All candidates were frozen before the post-August-2 holdout was scored. Holdout results cannot replace the provisional champion.",
        "",
        "## New holdout comparison (VWAP5)",
        "",
        "| Candidate | Trades | Coverage | Accuracy | Net PnL | Stress PnL | PF | Avg entry | Median entry |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in metrics["holdout_results"].items():
        row = result["vwap5"]
        lines.append(
            f"| {name} | {row['trades']} | {row['strict_market_coverage']:.2%} | "
            f"{row['accuracy']:.2%} | {row['net_pnl']:.2f} | {row['stress_net_pnl']:.2f} | "
            f"{_fmt(row['profit_factor'])} | {_fmt(row['average_entry_second'])} | "
            f"{_fmt(row['median_entry_second'])} |"
        )
    lines.extend(
        [
            "",
            "## Fixed-entry VWAP PnL",
            "",
            "| Candidate | VWAP5 | VWAP10 | VWAP20 | VWAP50 | VWAP100 | VWAP200 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in metrics["holdout_results"].items():
        curve = result["capacity"]
        lines.append(
            f"| {name} | {curve['5']['net_pnl']:.2f} | {curve['10']['net_pnl']:.2f} | "
            f"{curve['20']['net_pnl']:.2f} | {curve['50']['net_pnl']:.2f} | "
            f"{curve['100']['net_pnl']:.2f} | {curve['200']['net_pnl']:.2f} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {value}" for value in metrics["limitations"])
    return "\n".join(lines) + "\n"


def _variable_features(frame: pl.DataFrame, features: tuple[str, ...]) -> tuple[str, ...]:
    selected: list[str] = []
    for name in features:
        if name not in frame.columns:
            continue
        values = frame[name].cast(pl.Float64).drop_nulls()
        values = values.filter(values.is_finite())
        if values.n_unique() >= 2:
            selected.append(name)
    if not selected:
        raise RuntimeError("candidate has no variable finite features")
    return tuple(selected)


def _matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    matrix = frame.select(*features).cast(pl.Float64).to_numpy()
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def _candidate_eligible_frame(
    frame: pl.DataFrame,
    eligibility_features: tuple[str, ...],
) -> pl.DataFrame:
    if not eligibility_features:
        return frame
    return frame.drop_nulls(eligibility_features)


def _bucket_indices(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    return np.clip(
        np.searchsorted(np.asarray(edges[1:-1]), values, side="right"),
        0,
        len(edges) - 2,
    )


def _block(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    return frame.filter((pl.col("window_start") >= start) & (pl.col("window_start") < end))


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _file_record(path: Path) -> dict[str, Any]:
    frame = pl.scan_parquet(path)
    return {
        "path": path.name,
        "rows": frame.select(pl.len()).collect().item(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _git_revision(package_root: Path) -> str:
    git_file = package_root.parents[1] / ".git"
    if not git_file.is_file():
        return "unknown"
    content = git_file.read_text().strip()
    if not content.startswith("gitdir:"):
        return "unknown"
    git_dir = Path(content.split(":", 1)[1].strip())
    revision = (git_dir / "HEAD").read_text().strip()
    if not revision.startswith("ref:"):
        return revision
    common = git_dir
    if (git_dir / "commondir").is_file():
        common = (git_dir / (git_dir / "commondir").read_text().strip()).resolve()
    return (common / revision.split(" ", 1)[1]).read_text().strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_finite(payload), indent=2, sort_keys=True) + "\n")


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.3f}"


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
    run_dir, metrics = run_tournament(load_config(args.config))
    print(f"run: {run_dir}")
    print(json.dumps(metrics["qualification"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
