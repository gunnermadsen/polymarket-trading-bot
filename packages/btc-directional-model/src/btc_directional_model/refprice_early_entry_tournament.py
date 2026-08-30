"""RefPrice-first BTC five-minute early-entry training tournament.

The workflow is deliberately training-only.  It consumes checksum-sealed local
artifacts produced from existing database relations, builds causal observations
at 60..180 seconds, evaluates four frozen candidates and four label-history
arms with chronological out-of-fold predictions, and writes resumable model and
report artifacts.  TWAP is supervision only and is rejected from every feature
matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .core_execution import taker_fee_per_share
from .core_extract import configure_read_only_connection, database_connection
from .early_entry_settlement_consensus_tournament import economic_metrics
from .refprice_context_data import (
    ENTRY_SECONDS,
    extract_source_artifacts,
    file_sha256,
    validate_inference_columns,
)
from .refprice_context_data import (
    load_config as load_source_config,
)
from .refprice_twap_training import _query_frame

PROFILE = "btc_5m_refprice_early_entry_tournament"
CANDIDATES = (
    "refprice_path_control",
    "refprice_chainlink_crossvenue_context",
    "refprice_binance_microstructure_context",
    "refprice_nonnegative_consensus",
)
HISTORY_ARMS = (
    "authentic_only",
    "chainlink_reconstructed",
    "binance_synthetic_extension",
    "uncertainty_weighted_hybrid",
)
POLICIES = (
    "probability_edge",
    "edge_debit",
    "edge_uncertainty_margin",
    "edge_loss_compensation_tail",
)
KEYS = ("market_id", "window_start", "observed_at", "seconds_elapsed")


@dataclass(frozen=True)
class Fold:
    name: str
    test_start: datetime
    test_end: datetime


@dataclass(frozen=True)
class TreeSpec:
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_iter: int


@dataclass(frozen=True)
class Policy:
    name: str
    minimum_edge: float
    maximum_debit: float
    slippage_reserve: float
    uncertainty_scale: float
    require_margin_excludes_zero: bool
    maximum_loss_recovery_wins: float


@dataclass(frozen=True)
class Config:
    source_path: Path
    package_root: Path
    source_config: Path
    source_cache: Path
    execution_cache: Path
    run_root: Path
    result_root: Path
    run_id: str
    checkpoint_origin_revision: str
    random_seed: int
    data_end: datetime
    official_start: datetime
    economic_end: datetime
    policy_development_end: datetime
    sealed_start: datetime
    prospective_start: datetime
    folds: tuple[Fold, ...]
    tree: TreeSpec
    policies: tuple[Policy, ...]
    primary_history_arm: str


@dataclass
class Calibrator:
    slope: float
    intercept: float

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        logits = _logit(raw_probability)
        return _sigmoid(self.intercept + self.slope * logits)


@dataclass
class ModelBundle:
    candidate: str
    history_arm: str
    feature_names: tuple[str, ...]
    active_indices: tuple[int, ...]
    classifier: HistGradientBoostingClassifier
    calibrator: Calibrator
    margin: HistGradientBoostingRegressor
    margin_error_quantile: float
    fit_end: datetime


@dataclass
class ConsensusBundle:
    candidate: str
    history_arm: str
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    intercept: float
    margin_coefficients: np.ndarray
    margin_error_quantile: float
    fit_end: datetime


def _utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def load_config(path: Path) -> Config:
    source = path.resolve()
    package_root = source.parent.parent
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    training = raw["training"]
    for key, expected in {
        "profile": PROFILE,
        "training_only": True,
        "runtime_exported": False,
        "database_mutations": False,
        "twap_inference_allowed": False,
    }.items():
        if training.get(key) != expected:
            raise ValueError(f"invalid frozen tournament setting: {key}")
    if tuple(training["candidates"]) != CANDIDATES:
        raise ValueError("frozen candidate roster changed")
    if tuple(training["history_arms"]) != HISTORY_ARMS:
        raise ValueError("frozen history-arm roster changed")
    paths = raw["paths"]
    windows = raw["windows"]
    config = Config(
        source_path=source,
        package_root=package_root,
        source_config=package_root / paths["source_config"],
        source_cache=package_root / paths["source_cache"],
        execution_cache=package_root / paths["execution_cache"],
        run_root=package_root / paths["runs"],
        result_root=package_root / paths["committed_results"],
        run_id=str(training["run_id"]),
        checkpoint_origin_revision=str(training["checkpoint_origin_revision"]),
        random_seed=int(training["random_seed"]),
        data_end=_utc(windows["data_end"]),
        official_start=_utc(windows["official_start"]),
        economic_end=_utc(windows["economic_end"]),
        policy_development_end=_utc(windows["policy_development_end"]),
        sealed_start=_utc(windows["sealed_start"]),
        prospective_start=_utc(windows["prospective_start"]),
        folds=tuple(
            Fold(row["name"], _utc(row["test_start"]), _utc(row["test_end"]))
            for row in raw["oof_folds"]
        ),
        tree=TreeSpec(**raw["tree"]),
        policies=tuple(Policy(**row) for row in raw["policies"]),
        primary_history_arm=str(training["primary_history_arm"]),
    )
    _validate_config(config)
    return config


def _validate_config(config: Config) -> None:
    if config.primary_history_arm not in HISTORY_ARMS:
        raise ValueError("unknown primary history arm")
    if not config.run_id or "/" in config.run_id or ".." in config.run_id:
        raise ValueError("invalid frozen run ID")
    if tuple(policy.name for policy in config.policies) != POLICIES:
        raise ValueError("frozen policy roster changed")
    if len(config.folds) != 7:
        raise ValueError("expected seven chronological folds")
    if any(left.test_end > right.test_start for left, right in zip(config.folds, config.folds[1:])):
        raise ValueError("OOF folds overlap")
    if config.folds[-1].test_end != config.data_end:
        raise ValueError("OOF folds must consume evidence through August 28")
    if not (
        config.official_start
        < config.policy_development_end
        == config.sealed_start
        < config.economic_end
        == config.prospective_start
        < config.data_end
    ):
        raise ValueError("evaluation windows changed")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(path)


def _write_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def _git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _tracked_dirty(root: Path) -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
        ).strip()
    )


def _load_group(cache: Path, manifest: dict[str, Any], name: str) -> pl.DataFrame:
    rows = manifest["partitions"][name]
    paths = [cache / row["path"] for row in rows]
    for row, path in zip(rows, paths, strict=True):
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise RuntimeError(f"source partition identity changed: {row['path']}")
    frames = [pl.read_parquet(path) for path in paths]
    nonempty = [frame for frame in frames if frame.width]
    return pl.concat(nonempty, how="diagonal_relaxed") if nonempty else pl.DataFrame()


def _asof_feature(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    left_on: str,
    right_on: str,
    columns: tuple[str, ...],
    prefix: str,
) -> pl.DataFrame:
    if right.is_empty():
        return left.with_columns(
            *(pl.lit(None, dtype=pl.Float64).alias(prefix + name) for name in columns)
        )
    right_key = prefix + right_on
    selected = right.select(right_on, *columns).sort(right_on)
    renamed = selected.rename({right_on: right_key, **{name: prefix + name for name in columns}})
    return left.sort(left_on).join_asof(
        renamed,
        left_on=left_on,
        right_on=right_key,
        strategy="backward",
        check_sortedness=False,
    )


def build_panel(config: Config, manifest: dict[str, Any]) -> pl.DataFrame:
    cache = config.source_cache
    labels = _load_group(cache, manifest, "labels")
    core = (
        _load_group(cache, manifest, "core")
        .rename({"trade_count": "btc_trade_count"})
        .sort(["market_id", "seconds_elapsed"])
    )
    ref = _load_group(cache, manifest, "refprice")
    candles = _load_group(cache, manifest, "candles")
    oracle = _load_group(cache, manifest, "oracle")
    interest = _load_group(cache, manifest, "open_interest")
    trades = _load_group(cache, manifest, "aggregate_trades")
    l2 = _load_group(cache, manifest, "spot_l2")

    core = core.with_columns(
        *(
            (
                10_000.0
                * (pl.col("btc_close") / pl.col("btc_close").shift(h).over("market_id")).log()
            ).alias(f"btc_return_{h}s_bps")
            for h in (1, 5, 15, 30, 60)
        ),
        (
            (2.0 * pl.col("btc_taker_buy_base_volume") / pl.col("btc_base_volume").replace(0, None))
            - 1.0
        ).alias("btc_taker_flow_ratio_1s"),
        pl.col("btc_base_volume")
        .rolling_sum(30, min_samples=1)
        .over("market_id")
        .alias("btc_base_volume_30s"),
        pl.col("btc_trade_count")
        .cast(pl.Float64)
        .rolling_sum(30, min_samples=1)
        .over("market_id")
        .alias("btc_trade_count_30s"),
    ).with_columns(
        pl.col("btc_return_1s_bps")
        .rolling_std(30, min_samples=5)
        .over("market_id")
        .alias("btc_realized_vol_30s_bps"),
    )
    synthetic = (
        core.group_by("market_id")
        .agg(
            pl.col("btc_open").sort_by("seconds_elapsed").first().alias("binance_open"),
            pl.col("btc_close").sort_by("seconds_elapsed").last().alias("binance_close"),
        )
        .with_columns(
            (pl.col("binance_close") >= pl.col("binance_open"))
            .cast(pl.Int8)
            .alias("binance_synthetic_label_up")
        )
    )
    observations = (
        core.filter(pl.col("seconds_elapsed").is_in(ENTRY_SECONDS))
        .select(
            *KEYS,
            "opening_boundary",
            "btc_close",
            "btc_base_volume",
            "btc_quote_volume",
            "btc_trade_count",
            "btc_taker_flow_ratio_1s",
            "btc_base_volume_30s",
            "btc_trade_count_30s",
            "btc_realized_vol_30s_bps",
            *(f"btc_return_{h}s_bps" for h in (1, 5, 15, 30, 60)),
        )
        .join(synthetic.select("market_id", "binance_synthetic_label_up"), on="market_id")
    )
    panel = observations.join(
        labels.select(
            "market_id",
            "official_label_up",
            "target_label_up",
            "target_margin_bps",
            "target_label_source",
            "reconstructed_twap60_label_up",
        ),
        on="market_id",
        how="inner",
        validate="m:1",
    )

    ref = (
        ref.filter(
            pl.col("available_at").is_not_null()
            & pl.col("price").is_finite()
            & (pl.col("price") > 0)
        )
        .sort(["available_at", "source_timestamp", "report_sha256"])
        .unique(subset=["available_at"], keep="last", maintain_order=True)
    )
    panel = _asof_feature(
        panel,
        ref,
        left_on="observed_at",
        right_on="available_at",
        columns=("price", "bid", "ask", "source_timestamp", "expires_at"),
        prefix="ref_",
    )
    panel = panel.with_columns(pl.col("window_start").alias("ref_open_lookup"))
    panel = _asof_feature(
        panel,
        ref,
        left_on="ref_open_lookup",
        right_on="available_at",
        columns=("price",),
        prefix="ref_open_",
    )
    for horizon in (5, 15, 30, 60, 90, 120):
        lookup = f"ref_lookup_{horizon}s"
        panel = panel.with_columns(
            (pl.col("observed_at") - pl.duration(seconds=horizon)).alias(lookup)
        )
        panel = _asof_feature(
            panel,
            ref,
            left_on=lookup,
            right_on="available_at",
            columns=("price",),
            prefix=f"ref_{horizon}s_",
        )
    panel = panel.with_columns(
        (10_000.0 * (pl.col("ref_price") / pl.col("ref_open_price")).log()).alias(
            "ref_path_from_open_bps"
        ),
        (10_000.0 * (pl.col("ref_price") / pl.col("opening_boundary")).log()).alias(
            "ref_vs_boundary_bps"
        ),
        (10_000.0 * (pl.col("ref_ask") - pl.col("ref_bid")) / pl.col("ref_price")).alias(
            "ref_spread_bps"
        ),
        (pl.col("observed_at") - pl.col("ref_source_timestamp"))
        .dt.total_milliseconds()
        .cast(pl.Float64)
        .truediv(1000)
        .alias("ref_source_age_seconds"),
        (pl.col("ref_expires_at") - pl.col("observed_at"))
        .dt.total_milliseconds()
        .cast(pl.Float64)
        .truediv(1000)
        .alias("ref_expiry_remaining_seconds"),
        *(
            (10_000.0 * (pl.col("ref_price") / pl.col(f"ref_{h}s_price")).log()).alias(
                f"ref_return_{h}s_bps"
            )
            for h in (5, 15, 30, 60, 90, 120)
        ),
    )
    panel = panel.with_columns(
        (pl.col("ref_return_15s_bps") - pl.col("ref_return_30s_bps") / 2.0).alias(
            "ref_acceleration_bps"
        ),
        (-pl.col("ref_return_15s_bps") * pl.col("ref_return_60s_bps")).alias(
            "ref_reversal_interaction"
        ),
        pl.col("seconds_elapsed").cast(pl.Float64).alias("decision_second"),
    )

    if not candles.is_empty():
        panel = _asof_feature(
            panel,
            candles.filter(pl.col("close_timestamp") <= pl.col("available_at")),
            left_on="observed_at",
            right_on="available_at",
            columns=("open_price", "high_price", "low_price", "close_price", "close_timestamp"),
            prefix="candle_",
        )
        panel = panel.with_columns(
            (10_000.0 * (pl.col("candle_close_price") / pl.col("candle_open_price")).log()).alias(
                "chainlink_candle_return_bps"
            ),
            (10_000.0 * (pl.col("candle_high_price") / pl.col("candle_low_price")).log()).alias(
                "chainlink_candle_range_bps"
            ),
            (10_000.0 * (pl.col("ref_price") / pl.col("candle_close_price")).log()).alias(
                "ref_candle_basis_bps"
            ),
        )
    if not oracle.is_empty():
        panel = _asof_feature(
            panel,
            oracle,
            left_on="observed_at",
            right_on="oracle_block_timestamp",
            columns=("oracle_price", "oracle_source_timestamp"),
            prefix="chain_",
        )
        panel = panel.with_columns(
            (10_000.0 * (pl.col("ref_price") / pl.col("chain_oracle_price")).log()).alias(
                "ref_oracle_basis_bps"
            ),
            (pl.col("observed_at") - pl.col("chain_oracle_source_timestamp"))
            .dt.total_milliseconds()
            .cast(pl.Float64)
            .truediv(1000)
            .alias("oracle_age_seconds"),
        )
    if not interest.is_empty():
        panel = _asof_feature(
            panel,
            interest,
            left_on="observed_at",
            right_on="available_at",
            columns=("sum_open_interest", "sum_open_interest_value", "source_timestamp"),
            prefix="oi_",
        )
        panel = panel.with_columns(
            (pl.col("oi_sum_open_interest").pct_change().over("market_id") * 10_000.0).alias(
                "oi_change_bps"
            ),
            (pl.col("observed_at") - pl.col("oi_source_timestamp"))
            .dt.total_milliseconds()
            .cast(pl.Float64)
            .truediv(1000)
            .alias("oi_age_seconds"),
        )
    if not trades.is_empty():
        panel = _asof_feature(
            panel,
            trades,
            left_on="observed_at",
            right_on="available_at",
            columns=(
                "quote_volume",
                "base_volume",
                "signed_taker_quote_volume",
                "trade_count",
                "trade_vwap",
            ),
            prefix="prints_",
        )
        panel = panel.with_columns(
            (
                pl.col("prints_signed_taker_quote_volume")
                / pl.col("prints_quote_volume").replace(0, None)
            ).alias("prints_signed_flow_ratio"),
            (10_000.0 * (pl.col("prints_trade_vwap") / pl.col("ref_price")).log()).alias(
                "prints_ref_basis_bps"
            ),
        )
    if not l2.is_empty():
        l2_columns = tuple(
            name
            for name in (
                "midpoint",
                "microprice",
                "spread_bps",
                "imbalance_5",
                "imbalance_10",
                "imbalance_20",
                "bid_depth_20",
                "ask_depth_20",
                "bid_quote_replenishment_1s",
                "ask_quote_replenishment_1s",
                "bid_quote_churn_1s",
                "ask_quote_churn_1s",
                "midpoint_change_bps_5s",
                "midpoint_change_bps_15s",
                "midpoint_change_bps_30s",
                "midpoint_change_bps_60s",
                "imbalance_20_delta_15s",
                "depth_20_change_bps_15s",
            )
            if name in l2.columns
        )
        panel = _asof_feature(
            panel,
            l2,
            left_on="observed_at",
            right_on="available_at",
            columns=l2_columns,
            prefix="l2_",
        )
        panel = panel.with_columns(
            (10_000.0 * (pl.col("l2_microprice") / pl.col("ref_price")).log()).alias(
                "l2_ref_basis_bps"
            )
        )

    panel = panel.filter(
        pl.col("target_label_up").is_not_null()
        & pl.col("target_margin_bps").is_finite()
        & pl.col("ref_price").is_finite()
    ).sort(["window_start", "market_id", "seconds_elapsed"])
    if panel.select(pl.struct(KEYS).is_duplicated().any()).item():
        raise RuntimeError("duplicate observation keys")
    return panel


CONTROL_FEATURES = (
    "decision_second",
    "ref_path_from_open_bps",
    "ref_vs_boundary_bps",
    "ref_spread_bps",
    "ref_source_age_seconds",
    "ref_expiry_remaining_seconds",
    "ref_return_5s_bps",
    "ref_return_15s_bps",
    "ref_return_30s_bps",
    "ref_return_60s_bps",
    "ref_return_90s_bps",
    "ref_return_120s_bps",
    "ref_acceleration_bps",
    "ref_reversal_interaction",
)
CHAINLINK_FEATURES = CONTROL_FEATURES + (
    "chainlink_candle_return_bps",
    "chainlink_candle_range_bps",
    "ref_candle_basis_bps",
    "ref_oracle_basis_bps",
    "oracle_age_seconds",
)
BINANCE_FEATURES = CONTROL_FEATURES + (
    "btc_return_1s_bps",
    "btc_return_5s_bps",
    "btc_return_15s_bps",
    "btc_return_30s_bps",
    "btc_return_60s_bps",
    "btc_realized_vol_30s_bps",
    "btc_taker_flow_ratio_1s",
    "btc_base_volume_30s",
    "btc_trade_count_30s",
    "oi_sum_open_interest",
    "oi_sum_open_interest_value",
    "oi_change_bps",
    "oi_age_seconds",
    "prints_quote_volume",
    "prints_base_volume",
    "prints_signed_taker_quote_volume",
    "prints_trade_count",
    "prints_signed_flow_ratio",
    "prints_ref_basis_bps",
    "l2_spread_bps",
    "l2_imbalance_5",
    "l2_imbalance_10",
    "l2_imbalance_20",
    "l2_bid_depth_20",
    "l2_ask_depth_20",
    "l2_bid_quote_replenishment_1s",
    "l2_ask_quote_replenishment_1s",
    "l2_bid_quote_churn_1s",
    "l2_ask_quote_churn_1s",
    "l2_midpoint_change_bps_5s",
    "l2_midpoint_change_bps_15s",
    "l2_midpoint_change_bps_30s",
    "l2_midpoint_change_bps_60s",
    "l2_imbalance_20_delta_15s",
    "l2_depth_20_change_bps_15s",
    "l2_ref_basis_bps",
)


def feature_names(candidate: str, panel: pl.DataFrame) -> tuple[str, ...]:
    desired = {
        CANDIDATES[0]: CONTROL_FEATURES,
        CANDIDATES[1]: CHAINLINK_FEATURES,
        CANDIDATES[2]: BINANCE_FEATURES,
    }[candidate]
    names = tuple(name for name in desired if name in panel.columns)
    validate_inference_columns(names)
    return names


def _matrix(frame: pl.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    values = frame.select(*(pl.col(name).cast(pl.Float64) for name in names)).to_numpy()
    missing = ~np.isfinite(values)
    values[missing] = np.nan
    return np.column_stack((values, missing.astype(np.float64)))


def _active_indices(matrix: np.ndarray) -> tuple[int, ...]:
    active = tuple(
        index
        for index in range(matrix.shape[1])
        if np.unique(matrix[np.isfinite(matrix[:, index]), index]).size > 1
    )
    if not active:
        raise RuntimeError("feature matrix has no non-constant finite columns")
    return active


def _history(frame: pl.DataFrame, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    official = frame["official_label_up"].to_numpy()
    reconstructed = frame["reconstructed_twap60_label_up"].to_numpy()
    synthetic = frame["binance_synthetic_label_up"].to_numpy()
    margins = np.abs(frame["target_margin_bps"].to_numpy().astype(float))
    official_valid = frame["official_label_up"].is_not_null().to_numpy()
    reconstructed_valid = frame["reconstructed_twap60_label_up"].is_not_null().to_numpy()
    synthetic_valid = frame["binance_synthetic_label_up"].is_not_null().to_numpy()
    if arm == "authentic_only":
        authentic_counterfactual = (
            frame["window_start"].to_numpy() >= np.datetime64("2026-08-01T00:00:00")
        ) & reconstructed_valid
        labels = np.where(official_valid, official, reconstructed)
        valid = official_valid | authentic_counterfactual
        reliability = np.where(official_valid, 1.0, 0.90)
    elif arm == "chainlink_reconstructed":
        labels = np.where(official_valid, official, reconstructed)
        valid = official_valid | reconstructed_valid
        reliability = np.where(official_valid, 1.0, np.clip(margins / 2.0, 0.25, 0.95))
    elif arm == "binance_synthetic_extension":
        labels = np.where(official_valid, official, synthetic)
        valid = official_valid | synthetic_valid
        reliability = np.where(official_valid, 1.0, 0.55)
    elif arm == "uncertainty_weighted_hybrid":
        labels = np.where(
            official_valid, official, np.where(reconstructed_valid, reconstructed, synthetic)
        )
        valid = official_valid | reconstructed_valid | synthetic_valid
        reliability = np.where(
            official_valid,
            1.0,
            np.where(reconstructed_valid, np.clip(margins / 2.0, 0.30, 0.90), 0.45),
        )
    else:
        raise ValueError(f"unknown history arm: {arm}")
    return np.asarray(labels[valid], dtype=np.int8), valid, reliability[valid]


def _weights(frame: pl.DataFrame, valid: np.ndarray, reliability: np.ndarray) -> np.ndarray:
    selected = frame.filter(pl.Series(valid))
    counts = selected.group_by("market_id").len().rename({"len": "market_rows"})
    per_row = selected.join(counts, on="market_id", validate="m:1")["market_rows"].to_numpy()
    weights = reliability / per_row
    return weights / np.mean(weights)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def _logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _fit_calibrator(raw: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> Calibrator:
    logits = _logit(raw)

    def objective(params: np.ndarray) -> float:
        probabilities = _sigmoid(params[0] + params[1] * logits)
        loss = -(
            labels * np.log(np.clip(probabilities, 1e-9, 1))
            + (1 - labels) * np.log(np.clip(1 - probabilities, 1e-9, 1))
        )
        return float(np.average(loss, weights=weights))

    fitted = minimize(objective, np.array([0.0, 1.0]), method="L-BFGS-B", bounds=[(-8, 8), (0, 10)])
    if not fitted.success:
        raise RuntimeError("probability calibration failed")
    return Calibrator(float(fitted.x[1]), float(fitted.x[0]))


def fit_model(
    candidate: str, arm: str, frame: pl.DataFrame, fit_end: datetime, config: Config
) -> ModelBundle:
    names = feature_names(candidate, frame)
    labels, valid, reliability = _history(frame, arm)
    selected = frame.filter(pl.Series(valid))
    if selected["market_id"].n_unique() < 200 or len(np.unique(labels)) != 2:
        raise RuntimeError(f"insufficient training markets/classes for {candidate}/{arm}")
    weights = _weights(frame, valid, reliability)
    market_starts = selected.select("market_id", "window_start").unique().sort("window_start")
    split_index = max(1, int(market_starts.height * 0.8))
    calibration_ids = set(market_starts["market_id"].to_list()[split_index:])
    calibration_mask = np.array(
        [value in calibration_ids for value in selected["market_id"].to_list()]
    )
    fit_mask = ~calibration_mask
    spec = config.tree
    classifier = HistGradientBoostingClassifier(
        learning_rate=spec.learning_rate,
        max_leaf_nodes=spec.max_leaf_nodes,
        min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2_regularization,
        max_iter=spec.max_iter,
        random_state=config.random_seed,
        early_stopping=True,
        validation_fraction=0.1,
    )
    matrix = _matrix(selected, names)
    active_indices = _active_indices(matrix[fit_mask])
    active_matrix = matrix[:, active_indices]
    classifier.fit(active_matrix[fit_mask], labels[fit_mask], sample_weight=weights[fit_mask])
    raw_cal = classifier.predict_proba(active_matrix[calibration_mask])[:, 1]
    calibrator = _fit_calibrator(raw_cal, labels[calibration_mask], weights[calibration_mask])
    margin = HistGradientBoostingRegressor(
        learning_rate=spec.learning_rate,
        max_leaf_nodes=spec.max_leaf_nodes,
        min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2_regularization,
        max_iter=spec.max_iter,
        random_state=config.random_seed + 1,
        early_stopping=True,
        validation_fraction=0.1,
    )
    target_margin = selected["target_margin_bps"].to_numpy()
    margin.fit(active_matrix, target_margin, sample_weight=weights)
    residual = np.abs(
        target_margin[calibration_mask] - margin.predict(active_matrix[calibration_mask])
    )
    return ModelBundle(
        candidate,
        arm,
        names,
        active_indices,
        classifier,
        calibrator,
        margin,
        float(np.quantile(residual, 0.90)),
        fit_end,
    )


def score_model(frame: pl.DataFrame, bundle: ModelBundle) -> pl.DataFrame:
    matrix = _matrix(frame, bundle.feature_names)[:, bundle.active_indices]
    probability = bundle.calibrator.predict(bundle.classifier.predict_proba(matrix)[:, 1])
    margin = bundle.margin.predict(matrix)
    return frame.select(*KEYS, "official_label_up", "target_margin_bps").with_columns(
        pl.Series("probability_up", probability),
        pl.Series("predicted_margin_bps", margin),
        pl.lit(bundle.margin_error_quantile).alias("margin_uncertainty_bps"),
        pl.lit(bundle.candidate).alias("candidate"),
        pl.lit(bundle.history_arm).alias("history_arm"),
    )


def fit_consensus(prior: pl.DataFrame, arm: str, fit_end: datetime) -> ConsensusBundle:
    names = tuple(f"p_{name}" for name in CANDIDATES[:3])
    if prior.is_empty():
        coefficients = np.ones(3) / 3
        intercept = 0.0
        margin_coefficients = np.ones(3) / 3
        error = 5.0
    else:
        wide = _wide_base_predictions(prior)
        probabilities = wide.select(*names).to_numpy()
        labels = wide["official_label_up"].to_numpy().astype(float)
        logits = _logit(probabilities)

        def objective(params: np.ndarray) -> float:
            predicted = _sigmoid(params[0] + logits @ params[1:])
            return float(
                np.mean(
                    -(
                        labels * np.log(np.clip(predicted, 1e-9, 1))
                        + (1 - labels) * np.log(np.clip(1 - predicted, 1e-9, 1))
                    )
                )
            )

        result = minimize(
            objective,
            np.r_[0.0, np.ones(3) / 3],
            method="L-BFGS-B",
            bounds=[(-8, 8), (0, 8), (0, 8), (0, 8)],
        )
        if not result.success:
            raise RuntimeError("nonnegative consensus fit failed")
        intercept, coefficients = float(result.x[0]), result.x[1:]
        margin_matrix = wide.select(*(f"m_{name}" for name in CANDIDATES[:3])).to_numpy()
        target = wide["target_margin_bps"].to_numpy()
        margin_result = minimize(
            lambda value: float(np.mean((target - margin_matrix @ value) ** 2)),
            np.ones(3) / 3,
            method="L-BFGS-B",
            bounds=[(0, 8)] * 3,
        )
        margin_coefficients = margin_result.x
        error = float(np.quantile(np.abs(target - margin_matrix @ margin_coefficients), 0.90))
    return ConsensusBundle(
        CANDIDATES[3], arm, names, coefficients, intercept, margin_coefficients, error, fit_end
    )


def _wide_base_predictions(frame: pl.DataFrame) -> pl.DataFrame:
    join_keys = list(KEYS) + ["history_arm", "fold"]
    pieces = []
    for index, name in enumerate(CANDIDATES[:3]):
        columns: list[Any] = [*join_keys]
        if index == 0:
            columns.extend(["official_label_up", "target_margin_bps"])
        pieces.append(
            frame.filter(pl.col("candidate") == name).select(
                *columns,
                pl.col("probability_up").alias(f"p_{name}"),
                pl.col("predicted_margin_bps").alias(f"m_{name}"),
            )
        )
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.join(piece, on=join_keys, how="inner", validate="1:1")
    return result


def score_consensus(base: pl.DataFrame, bundle: ConsensusBundle) -> pl.DataFrame:
    wide = _wide_base_predictions(base)
    probability_matrix = wide.select(*bundle.feature_names).to_numpy()
    margin_matrix = wide.select(*(f"m_{name}" for name in CANDIDATES[:3])).to_numpy()
    probability = _sigmoid(bundle.intercept + _logit(probability_matrix) @ bundle.coefficients)
    margin = margin_matrix @ bundle.margin_coefficients
    fold_name = wide["fold"][0]
    return wide.select(*KEYS, "official_label_up", "target_margin_bps").with_columns(
        pl.Series("probability_up", probability),
        pl.Series("predicted_margin_bps", margin),
        pl.lit(bundle.margin_error_quantile).alias("margin_uncertainty_bps"),
        pl.lit(CANDIDATES[3]).alias("candidate"),
        pl.lit(bundle.history_arm).alias("history_arm"),
        pl.lit(fold_name).alias("fold"),
    )


def predictive_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    valid = frame.filter(pl.col("official_label_up").is_not_null())
    if valid.is_empty():
        return {
            "rows": 0,
            "markets": 0,
            "brier": None,
            "log_loss": None,
            "accuracy": None,
            "ece": None,
        }
    labels = valid["official_label_up"].to_numpy().astype(float)
    probability = np.clip(valid["probability_up"].to_numpy(), 1e-9, 1 - 1e-9)
    brier = float(np.mean((probability - labels) ** 2))
    log_loss = float(
        np.mean(-(labels * np.log(probability) + (1 - labels) * np.log(1 - probability)))
    )
    accuracy = float(np.mean((probability >= 0.5) == labels))
    ece = 0.0
    for low in np.linspace(0, 1, 11)[:-1]:
        mask = (probability >= low) & (probability < low + 0.1 if low < 0.9 else probability <= 1)
        if mask.any():
            ece += float(mask.mean()) * abs(float(probability[mask].mean() - labels[mask].mean()))
    return {
        "rows": valid.height,
        "markets": valid["market_id"].n_unique(),
        "brier": brier,
        "log_loss": log_loss,
        "accuracy": accuracy,
        "ece": ece,
    }


def _load_execution(config: Config) -> tuple[dict[str, Any], pl.DataFrame]:
    cache = config.execution_cache
    cache.mkdir(parents=True, exist_ok=True)
    query_path = config.package_root / "sql" / "btc-refprice-twap-capacity-source.sql"
    query = query_path.read_text()
    contract = {
        "schema_version": "btc-refprice-early-entry-capacity-v1",
        "range_start": config.official_start.isoformat(),
        "range_end": config.economic_end.isoformat(),
        "entry_seconds": list(ENTRY_SECONDS),
        "quantity": 5,
        "maximum_depth_participation": 0.25,
        "freshness_seconds": 2,
        "query_sha256": file_sha256(query_path),
        "source_table": "polymarket.btc_market_capacity_execution_snapshots",
        "source_providers": [
            "pmxt_v2_capacity_execution_snapshots_v2",
            "polymarket_local_orderbook_capacity_execution_snapshots_v1",
        ],
    }
    manifest_path = cache / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        mismatches = [key for key, value in contract.items() if manifest.get(key) != value]
        if mismatches:
            raise RuntimeError(
                "capacity execution manifest contract changed: " + ", ".join(mismatches)
            )
        for row in manifest["partitions"]:
            path = cache / row["path"]
            if not path.is_file() or file_sha256(path) != row["sha256"]:
                raise RuntimeError(f"capacity execution partition changed: {row['path']}")
    else:
        partitions: list[dict[str, Any]] = []
        connection = database_connection()
        configure_read_only_connection(connection)
        try:
            day = config.official_start
            while day < config.economic_end:
                end = min(day + timedelta(days=1), config.economic_end)
                raw = _query_frame(
                    connection,
                    query,
                    {"batch_start": day, "batch_end": end},
                    f"refprice_early_capacity_{day:%Y%m%d}",
                )
                frame = _qualify_capacity_execution(raw)
                path = cache / f"{day.date().isoformat()}.parquet"
                _write_parquet(path, frame)
                partitions.append(
                    {
                        "date": day.date().isoformat(),
                        "path": path.name,
                        "rows": frame.height,
                        "markets": frame["market_id"].n_unique() if frame.height else 0,
                        "strict_rows": (
                            frame.filter(pl.col("strict_both_side_eligible")).height
                            if frame.height
                            else 0
                        ),
                        "sha256": file_sha256(path),
                    }
                )
                print(
                    f"capacity execution {day.date()}: {frame.height} rows, "
                    f"{partitions[-1]['strict_rows']} strict",
                    flush=True,
                )
                day = end
        finally:
            connection.close()
        manifest = {
            **contract,
            "created_at": datetime.now(UTC).isoformat(),
            "partitions": partitions,
            "totals": {
                "rows": sum(row["rows"] for row in partitions),
                "markets": sum(row["markets"] for row in partitions),
                "strict_both_side_eligible_rows": sum(row["strict_rows"] for row in partitions),
            },
        }
        _write_json(manifest_path, manifest)
    frames = [pl.read_parquet(cache / row["path"]) for row in manifest["partitions"]]
    return manifest, pl.concat(frames, how="vertical_relaxed")


def _qualify_capacity_execution(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame.with_columns(pl.lit(False).alias("strict_both_side_eligible"))
    fresh = (
        pl.col("up_provider_received_at").is_not_null()
        & pl.col("down_provider_received_at").is_not_null()
        & (pl.col("up_provider_received_at") <= pl.col("observed_at"))
        & (pl.col("down_provider_received_at") <= pl.col("observed_at"))
        & (pl.col("up_provider_received_at") >= pl.col("observed_at") - pl.duration(seconds=2))
        & (pl.col("down_provider_received_at") >= pl.col("observed_at") - pl.duration(seconds=2))
    )
    executable = (
        pl.col("up_ask_vwap_5").is_finite()
        & pl.col("down_ask_vwap_5").is_finite()
        & pl.col("up_ask_vwap_5").is_between(0.0, 1.0, closed="none")
        & pl.col("down_ask_vwap_5").is_between(0.0, 1.0, closed="none")
        & (pl.col("up_ask_depth") >= 20.0)
        & (pl.col("down_ask_depth") >= 20.0)
    )
    return frame.filter(pl.col("seconds_elapsed").is_in(ENTRY_SECONDS)).with_columns(
        (((pl.col("quality_flags") & 63) == 0) & fresh & executable).alias(
            "strict_both_side_eligible"
        )
    )


def apply_policy(frame: pl.DataFrame, execution: pl.DataFrame, policy: Policy) -> pl.DataFrame:
    panel = frame.join(
        execution.select(
            "market_id",
            "seconds_elapsed",
            "strict_both_side_eligible",
            "fee_rate",
            "up_ask_vwap_5",
            "down_ask_vwap_5",
        ),
        on=["market_id", "seconds_elapsed"],
        how="inner",
        validate="m:1",
    ).with_columns(
        (pl.col("probability_up") >= 0.5).alias("predicted_up"),
        pl.max_horizontal(pl.col("probability_up"), 1 - pl.col("probability_up")).alias(
            "side_probability"
        ),
        pl.when(pl.col("probability_up") >= 0.5)
        .then(pl.col("up_ask_vwap_5"))
        .otherwise(pl.col("down_ask_vwap_5"))
        .alias("contract_cost"),
    )
    cost = panel["contract_cost"].to_numpy()
    fee = panel["fee_rate"].to_numpy()
    panel = panel.with_columns(
        pl.Series(
            "fee_per_share",
            np.asarray(taker_fee_per_share(fee_rate=fee, price=cost), dtype=float),
        ),
        pl.lit(policy.slippage_reserve).alias("slippage_reserve"),
        (pl.col("margin_uncertainty_bps") * policy.uncertainty_scale / 10_000.0).alias(
            "uncertainty_reserve"
        ),
    ).with_columns(
        (
            pl.col("side_probability")
            - pl.col("contract_cost")
            - pl.col("fee_per_share")
            - pl.col("slippage_reserve")
            - pl.col("uncertainty_reserve")
        ).alias("stressed_edge"),
        (pl.col("predicted_margin_bps").abs() > pl.col("margin_uncertainty_bps")).alias(
            "margin_excludes_zero"
        ),
        (pl.col("contract_cost") / (1 - pl.col("contract_cost")).clip(0.01, 1.0)).alias(
            "loss_recovery_wins_at_entry"
        ),
        pl.when(pl.col("predicted_up"))
        .then(pl.col("official_label_up") == 1)
        .otherwise(pl.col("official_label_up") == 0)
        .alias("direction_correct"),
    )
    eligible = panel.filter(
        pl.col("strict_both_side_eligible")
        & pl.col("official_label_up").is_not_null()
        & (pl.col("stressed_edge") >= policy.minimum_edge)
        & (pl.col("contract_cost") <= policy.maximum_debit)
        & (pl.col("loss_recovery_wins_at_entry") <= policy.maximum_loss_recovery_wins)
        & (pl.col("margin_excludes_zero") if policy.require_margin_excludes_zero else pl.lit(True))
    )
    return (
        eligible.sort(
            ["window_start", "market_id", "seconds_elapsed", "stressed_edge"],
            descending=[False, False, False, True],
        )
        .group_by("market_id", maintain_order=True)
        .first()
        .sort("window_start")
        .with_columns(pl.lit(policy.name).alias("policy"))
    )


def run(config: Config) -> dict[str, Any]:
    repository_root = config.package_root.parents[1]
    if _tracked_dirty(repository_root):
        raise RuntimeError("training requires a clean committed producing worktree")
    revision = _git_revision(repository_root)
    source_config = load_source_config(config.source_config)
    source_manifest = extract_source_artifacts(source_config)
    if source_manifest["readiness"]["status"] != "ready":
        raise RuntimeError("source manifest contains a genuine blocking integrity failure")
    execution_manifest, execution = _load_execution(config)
    identity = config.run_id
    run_dir = config.run_root / identity
    run_dir.mkdir(parents=True, exist_ok=True)
    panel_path = run_dir / "training-panel.parquet"
    if panel_path.is_file():
        panel = pl.read_parquet(panel_path)
    else:
        panel = build_panel(config, source_manifest)
        _write_parquet(panel_path, panel)
    integrity = integrity_audit(panel, config)
    _write_json(run_dir / "integrity-audit.json", integrity)
    if not integrity["passed"]:
        raise RuntimeError("pre-training integrity audit failed")

    all_predictions: list[pl.DataFrame] = []
    fold_models: dict[str, dict[str, dict[str, Any]]] = {}
    for fold_index, fold in enumerate(config.folds):
        raw_train = panel.filter(pl.col("window_start") < fold.test_start)
        test = panel.filter(
            pl.col("window_start").is_between(fold.test_start, fold.test_end, closed="left")
        )
        fold_models[fold.name] = {}
        for arm_index, arm in enumerate(HISTORY_ARMS):
            if arm == "authentic_only" and fold.test_start < config.official_start:
                continue
            fold_models[fold.name][arm] = {}
            base_predictions: list[pl.DataFrame] = []
            for candidate_index, candidate in enumerate(CANDIDATES[:3]):
                checkpoint = run_dir / "checkpoints" / fold.name / arm / f"{candidate}.joblib"
                if checkpoint.is_file():
                    bundle = joblib.load(checkpoint)
                else:
                    seeded = replace(
                        config,
                        random_seed=(
                            config.random_seed
                            + 1000 * fold_index
                            + 100 * arm_index
                            + candidate_index
                        ),
                    )
                    bundle = fit_model(candidate, arm, raw_train, fold.test_start, seeded)
                    _write_joblib(checkpoint, bundle)
                fold_models[fold.name][arm][candidate] = bundle
                scored = score_model(test, bundle).with_columns(pl.lit(fold.name).alias("fold"))
                base_predictions.append(scored)
                all_predictions.append(scored)
            base = pl.concat(base_predictions, how="vertical_relaxed")
            prior = (
                pl.concat(all_predictions, how="vertical_relaxed").filter(
                    (pl.col("history_arm") == arm)
                    & (pl.col("fold") != fold.name)
                    & (pl.col("official_label_up").is_not_null())
                )
                if all_predictions
                else pl.DataFrame()
            )
            consensus = fit_consensus(prior, arm, fold.test_start)
            fold_models[fold.name][arm][CANDIDATES[3]] = consensus
            all_predictions.append(score_consensus(base, consensus))
        _write_parquet(
            run_dir / "checkpoints" / fold.name / "predictions.parquet",
            pl.concat(all_predictions, how="vertical_relaxed").filter(pl.col("fold") == fold.name),
        )
        print(f"OOF fold complete: {fold.name}", flush=True)

    predictions = pl.concat(all_predictions, how="vertical_relaxed").sort(
        ["candidate", "history_arm", "window_start", "market_id", "seconds_elapsed"]
    )
    _write_parquet(run_dir / "oof-predictions.parquet", predictions)
    candidate_rows: list[dict[str, Any]] = []
    economic_rows: list[dict[str, Any]] = []
    ledgers: dict[str, pl.DataFrame] = {}
    for candidate in CANDIDATES:
        for arm in HISTORY_ARMS:
            selected = predictions.filter(
                (pl.col("candidate") == candidate)
                & (pl.col("history_arm") == arm)
                & (pl.col("window_start") >= config.official_start)
            )
            overall = predictive_metrics(selected)
            development_predictive = predictive_metrics(
                selected.filter(pl.col("window_start") < config.policy_development_end)
            )
            sealed_predictive = predictive_metrics(
                selected.filter(
                    pl.col("window_start").is_between(
                        config.sealed_start, config.economic_end, closed="left"
                    )
                )
            )
            prospective_predictive = predictive_metrics(
                selected.filter(pl.col("window_start") >= config.prospective_start)
            )
            candidate_rows.append(
                {
                    "candidate": candidate,
                    "history_arm": arm,
                    "overall": overall,
                    "development": development_predictive,
                    "sealed": sealed_predictive,
                    "prospective": prospective_predictive,
                }
            )
            economic_source = selected.filter(pl.col("window_start") < config.economic_end)
            for policy in config.policies:
                trades = apply_policy(economic_source, execution, policy)
                key = f"{candidate}__{arm}__{policy.name}"
                ledgers[key] = trades
                development = trades.filter(pl.col("window_start") < config.policy_development_end)
                sealed = trades.filter(pl.col("window_start") >= config.sealed_start)
                scheduled_development = panel.filter(
                    pl.col("official_label_up").is_not_null()
                    & pl.col("window_start").is_between(
                        config.official_start, config.policy_development_end, closed="left"
                    )
                )["market_id"].n_unique()
                scheduled_sealed = panel.filter(
                    pl.col("official_label_up").is_not_null()
                    & pl.col("window_start").is_between(
                        config.sealed_start, config.economic_end, closed="left"
                    )
                )["market_id"].n_unique()
                economic_rows.append(
                    {
                        "candidate": candidate,
                        "history_arm": arm,
                        "policy": policy.name,
                        "development": economic_metrics(development, scheduled_development),
                        "sealed": economic_metrics(sealed, scheduled_sealed),
                    }
                )
    ranking = sorted(
        economic_rows,
        key=lambda row: (
            row["development"]["stressed_pnl"],
            row["development"]["profit_factor"] or 0.0,
            -next(
                item["development"]["brier"]
                for item in candidate_rows
                if item["candidate"] == row["candidate"]
                and item["history_arm"] == row["history_arm"]
            ),
        ),
        reverse=True,
    )
    selected = ranking[0]
    for key, ledger in ledgers.items():
        _write_parquet(run_dir / "ledgers" / f"{key}.parquet", ledger)

    final_models: dict[str, Any] = {}
    training = panel.filter(pl.col("window_start") < config.data_end)
    if selected["candidate"] == CANDIDATES[3]:
        final_base = {}
        for candidate in CANDIDATES[:3]:
            final_base[candidate] = fit_model(
                candidate, selected["history_arm"], training, config.data_end, config
            )
        prior = predictions.filter(
            (pl.col("history_arm") == selected["history_arm"])
            & pl.col("official_label_up").is_not_null()
        )
        final_models = {
            "base": final_base,
            "consensus": fit_consensus(prior, selected["history_arm"], config.data_end),
        }
    else:
        final_models = {
            "model": fit_model(
                selected["candidate"], selected["history_arm"], training, config.data_end, config
            )
        }
    artifact_path = run_dir / "refprice-early-entry-tournament.joblib"
    artifact = {
        "schema_version": "btc-refprice-early-entry-tournament-v1",
        "producing_revision": revision,
        "source_identity": identity,
        "selected": selected,
        "models": final_models,
        "observation_seconds": ENTRY_SECONDS,
        "runtime_exported": False,
    }
    _write_joblib(artifact_path, artifact)
    loaded = joblib.load(artifact_path)
    if loaded["producing_revision"] != revision or loaded["selected"] != selected:
        raise RuntimeError("post-training artifact parity failure; do not retrain")
    artifact_sha = file_sha256(artifact_path)
    prospective_metrics = predictive_metrics(
        predictions.filter(
            (pl.col("candidate") == selected["candidate"])
            & (pl.col("history_arm") == selected["history_arm"])
            & (pl.col("window_start") >= config.prospective_start)
        )
    )
    report = {
        "schema_version": "btc-refprice-early-entry-report-v1",
        "run_id": identity,
        "producing_revision": revision,
        "oof_checkpoint_origin_revision": config.checkpoint_origin_revision,
        "artifact_path": str(artifact_path.relative_to(repository_root)),
        "artifact_sha256": artifact_sha,
        "source_manifest_sha256": file_sha256(config.source_cache / "source-manifest.json"),
        "execution_manifest_sha256": file_sha256(config.execution_cache / "manifest.json"),
        "source_readiness": source_manifest["readiness"],
        "execution_totals": execution_manifest["totals"],
        "integrity": integrity,
        "candidate_metrics": candidate_rows,
        "economic_metrics": economic_rows,
        "selected": selected,
        "prospective_predictive_only": prospective_metrics,
        "training_rows": panel.height,
        "training_markets": panel["market_id"].n_unique(),
        "feature_dimensions": {name: list(feature_names(name, panel)) for name in CANDIDATES[:3]},
        "twap_inference_allowed": False,
        "runtime_exported": False,
        "deployment_status": "not_deployed",
    }
    _write_json(run_dir / "metrics.json", report)
    (run_dir / "report.md").write_text(render_report(report))
    committed = config.result_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    committed.mkdir(parents=True, exist_ok=False)
    for name in (
        "metrics.json",
        "report.md",
        "refprice-early-entry-tournament.joblib",
        "oof-predictions.parquet",
        "integrity-audit.json",
    ):
        source = run_dir / name
        destination = committed / name
        destination.write_bytes(source.read_bytes())
    _write_json(
        committed / "provenance.json",
        {
            "producing_revision": revision,
            "run_id": identity,
            "artifact_sha256": artifact_sha,
            "source_manifest_sha256": report["source_manifest_sha256"],
            "execution_manifest_sha256": report["execution_manifest_sha256"],
            "qualification_status": "historical_tournament_complete",
            "deployment_status": "not_deployed",
        },
    )
    print(str(committed), flush=True)
    return {**report, "committed_result": str(committed), "run_dir": str(run_dir)}


def recompute_post_training_evaluation(config: Config, result_dir: Path) -> dict[str, Any]:
    """Correct economics from immutable OOF predictions without fitting any model."""

    result_dir = result_dir.resolve()
    original = json.loads((result_dir / "metrics.json").read_text())
    run_dir = config.run_root / config.run_id
    predictions = pl.read_parquet(run_dir / "oof-predictions.parquet")
    panel = pl.read_parquet(run_dir / "training-panel.parquet")
    execution_manifest, execution = _load_execution(config)
    if execution.is_empty() or not execution["strict_both_side_eligible"].any():
        raise RuntimeError("corrected capacity execution evidence is empty or ineligible")

    candidate_rows: list[dict[str, Any]] = []
    economic_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for arm in HISTORY_ARMS:
            selected = predictions.filter(
                (pl.col("candidate") == candidate)
                & (pl.col("history_arm") == arm)
                & (pl.col("window_start") >= config.official_start)
            )
            candidate_rows.append(
                {
                    "candidate": candidate,
                    "history_arm": arm,
                    "overall": predictive_metrics(selected),
                    "development": predictive_metrics(
                        selected.filter(pl.col("window_start") < config.policy_development_end)
                    ),
                    "sealed": predictive_metrics(
                        selected.filter(
                            pl.col("window_start").is_between(
                                config.sealed_start,
                                config.economic_end,
                                closed="left",
                            )
                        )
                    ),
                    "prospective": predictive_metrics(
                        selected.filter(pl.col("window_start") >= config.prospective_start)
                    ),
                }
            )
            economic_source = selected.filter(pl.col("window_start") < config.economic_end)
            for policy in config.policies:
                trades = apply_policy(economic_source, execution, policy)
                development = trades.filter(pl.col("window_start") < config.policy_development_end)
                sealed = trades.filter(pl.col("window_start") >= config.sealed_start)
                scheduled_development = panel.filter(
                    pl.col("official_label_up").is_not_null()
                    & pl.col("window_start").is_between(
                        config.official_start,
                        config.policy_development_end,
                        closed="left",
                    )
                )["market_id"].n_unique()
                scheduled_sealed = panel.filter(
                    pl.col("official_label_up").is_not_null()
                    & pl.col("window_start").is_between(
                        config.sealed_start, config.economic_end, closed="left"
                    )
                )["market_id"].n_unique()
                economic_rows.append(
                    {
                        "candidate": candidate,
                        "history_arm": arm,
                        "policy": policy.name,
                        "development": economic_metrics(development, scheduled_development),
                        "sealed": economic_metrics(sealed, scheduled_sealed),
                    }
                )
    ranking = sorted(
        economic_rows,
        key=lambda row: (
            row["development"]["stressed_pnl"],
            row["development"]["profit_factor"] or 0.0,
            -next(
                item["development"]["brier"]
                for item in candidate_rows
                if item["candidate"] == row["candidate"]
                and item["history_arm"] == row["history_arm"]
            ),
        ),
        reverse=True,
    )
    corrected = {
        **original,
        "schema_version": "btc-refprice-early-entry-corrected-evaluation-v1",
        "evaluation_status": "corrected_without_model_retraining",
        "original_evaluation_invalid_reason": (
            "retired execution artifact identity produced zero execution rows"
        ),
        "model_artifact_qualified": False,
        "model_artifact_qualification_reason": (
            "post-training evaluation defect changed the selection evidence; "
            "no retraining was performed"
        ),
        "execution_totals": execution_manifest["totals"],
        "execution_manifest_sha256": file_sha256(config.execution_cache / "manifest.json"),
        "candidate_metrics": candidate_rows,
        "economic_metrics": economic_rows,
        "selected": ranking[0],
    }
    _write_json(result_dir / "corrected-metrics.json", corrected)
    warning = (
        "# Post-training evaluation correction\n\n"
        "No model was retrained. The original zero-trade economics were invalid "
        "because they queried a retired execution artifact identity. The table "
        "below uses the existing capacity execution table. The model artifact "
        "remains unqualified and receives no model tag.\n\n"
    )
    (result_dir / "corrected-report.md").write_text(warning + render_report(corrected))
    return corrected


def integrity_audit(panel: pl.DataFrame, config: Config) -> dict[str, Any]:
    forbidden = [
        name
        for name in (*CONTROL_FEATURES, *CHAINLINK_FEATURES, *BINANCE_FEATURES)
        if any(
            token in name.lower() for token in ("twap", "official", "target", "final", "resolution")
        )
    ]
    duplicate_keys = int(panel.select(pl.struct(KEYS).is_duplicated().sum()).item())
    future_ref = int(panel.filter(pl.col("ref_source_timestamp") > pl.col("observed_at")).height)
    outside_seconds = int(panel.filter(~pl.col("seconds_elapsed").is_in(ENTRY_SECONDS)).height)
    fold_rows = []
    overlap = False
    for fold in config.folds:
        train_ids = set(
            panel.filter(pl.col("window_start") < fold.test_start)["market_id"].unique().to_list()
        )
        test_ids = set(
            panel.filter(
                pl.col("window_start").is_between(fold.test_start, fold.test_end, closed="left")
            )["market_id"]
            .unique()
            .to_list()
        )
        intersection = len(train_ids & test_ids)
        overlap |= bool(intersection)
        fold_rows.append(
            {
                "fold": fold.name,
                "train_markets": len(train_ids),
                "test_markets": len(test_ids),
                "overlap": intersection,
            }
        )
    passed = (
        not forbidden
        and not duplicate_keys
        and not future_ref
        and not outside_seconds
        and not overlap
    )
    return {
        "passed": passed,
        "forbidden_feature_names": forbidden,
        "duplicate_keys": duplicate_keys,
        "future_refprice_rows": future_ref,
        "outside_entry_window_rows": outside_seconds,
        "folds": fold_rows,
        "market_disjoint": not overlap,
    }


def render_report(report: dict[str, Any]) -> str:
    selected = report["selected"]
    sealed = selected["sealed"]
    predictive = next(
        row
        for row in report["candidate_metrics"]
        if row["candidate"] == selected["candidate"]
        and row["history_arm"] == selected["history_arm"]
    )
    lines = [
        "# RefPrice Early-Entry Tournament Report",
        "",
        "## Selected sealed result",
        "",
        "| Candidate | History arm | Policy | PnL | Coverage | Wins | Losses | Loss-recovery wins | Brier | Profit factor |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {selected['candidate']} | {selected['history_arm']} | {selected['policy']} | {sealed['stressed_pnl']:.4f} | {sealed['market_coverage']:.2%} | {sealed['wins']} | {sealed['losses']} | {_fmt(sealed['loss_recovery_wins'])} | {_fmt(predictive['sealed']['brier'])} | {_fmt(sealed['profit_factor'])} |",
        "",
        "## Full candidate predictive table",
        "",
        "| Candidate | History arm | Markets | Rows | Brier | Log loss | Accuracy | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(report["candidate_metrics"], key=lambda value: value["sealed"]["brier"]):
        metrics = row["sealed"]
        lines.append(
            f"| {row['candidate']} | {row['history_arm']} | {metrics['markets']} | {metrics['rows']} | {_fmt(metrics['brier'])} | {_fmt(metrics['log_loss'])} | {_fmt(metrics['accuracy'])} | {_fmt(metrics['ece'])} |"
        )
    lines += [
        "",
        "## Economic candidate-policy table",
        "",
        "| Candidate | History arm | Policy | Dev PnL | Sealed PnL | Coverage | Wins | Losses | Recovery wins | Profit factor |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        report["economic_metrics"], key=lambda value: value["sealed"]["stressed_pnl"], reverse=True
    ):
        sealed_row = row["sealed"]
        lines.append(
            f"| {row['candidate']} | {row['history_arm']} | {row['policy']} | {row['development']['stressed_pnl']:.4f} | {sealed_row['stressed_pnl']:.4f} | {sealed_row['market_coverage']:.2%} | {sealed_row['wins']} | {sealed_row['losses']} | {_fmt(sealed_row['loss_recovery_wins'])} | {_fmt(sealed_row['profit_factor'])} |"
        )
    lines += [
        "",
        "## Integrity and provenance",
        "",
        f"- Producing revision: `{report['producing_revision']}`",
        f"- Artifact SHA-256: `{report['artifact_sha256']}`",
        f"- Training observations: {report['training_rows']:,} across {report['training_markets']:,} markets",
        "- TWAP-30/TWAP-60 inference: prohibited",
        "- Runtime export/deployment: none",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--evaluation-only-result", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.evaluation_only_result:
        result = recompute_post_training_evaluation(config, args.evaluation_only_result)
        print(json.dumps({"selected": result["selected"]}, indent=2, default=str))
        return
    result = run(config)
    print(
        json.dumps(
            {"selected": result["selected"], "committed_result": result["committed_result"]},
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
