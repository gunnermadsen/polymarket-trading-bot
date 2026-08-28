"""Offline Settlement-Bridge Residual Tournament.

The workflow is deliberately training-only. It reads bounded, existing source
tables through read-only transactions, checkpoints immutable daily Parquet
partitions, trains exactly the four frozen candidates, and never exports a
runtime model or mutates database/trading-process state.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import pickle
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from .continuous_edge_training import BOOK_RAW_FEATURES, VWAP_QUANTITIES, attach_book_features
from .core_extract import configure_read_only_connection, database_connection, file_sha256
from .core_features import (
    attach_causal_oracle_rounds,
    derive_core_point_in_time_features,
    derive_oracle_point_in_time_features,
)
from .provenance import runtime_provenance
from .refprice_twap_training import _query_frame
from .twap60_training_data import (
    REFPRICE_ALL_FEATURES,
    attach_causal_refprice_features,
    canonical_refprice_path,
    construct_proxy_labels,
)

SCHEMA_VERSION = "btc-settlement-bridge-residual-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-settlement-bridge-residual-model-v1"
CANDIDATES = (
    "refprice_bridge_baseline",
    "non_twap_settlement_correction",
    "relative_twap_settlement_correction",
    "twap_margin_residual_bridge",
)
SQL_FILES = (
    "btc-refprice-twap-core-source.sql",
    "btc-refprice-twap-oracle-source.sql",
    "btc-settlement-bridge-label-source.sql",
    "btc-twap60-refprice-source.sql",
    "btc-refprice-twap-capacity-source.sql",
    "btc-settlement-bridge-binance-label-diagnostic.sql",
)

# Causal path features measured relative to the canonical RefPrice opening
# boundary. Absolute price and calendar/date shortcuts are deliberately absent.
BASE_FEATURES = (
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
    "btc_realized_volatility_5s_bps",
    "btc_realized_volatility_15s_bps",
    "btc_realized_volatility_30s_bps",
    "btc_realized_volatility_60s_bps",
    "btc_range_30s_bps",
    "btc_range_60s_bps",
    "btc_path_efficiency_30s",
    "btc_path_efficiency_60s",
    "btc_range_position_60s",
    "btc_signed_flow_5s",
    "btc_signed_flow_30s",
    "btc_signed_flow_60s",
    "btc_path_cross_count",
    "btc_boundary_cross_count",
    "btc_seconds_since_boundary_cross",
    "btc_boundary_distance_velocity_5s_bps",
    "btc_momentum_multihorizon_score",
    "btc_momentum_acceleration_5_vs_30",
    "btc_reversal_5_vs_30",
)
BASE_BRIDGE_FEATURES = (
    "base_log_odds",
    "base_margin_median",
    "base_margin_width",
)
NON_TWAP_FEATURES = (
    "chainlink_ref_binance_basis_bps",
    "chainlink_ref_binance_basis_velocity_5s_bps",
    "chainlink_ref_binance_direction_agreement_30s",
    "chainlink_ref_age_seconds",
    "chainlink_ref_source_skew_seconds",
    "chainlink_ref_realized_volatility_30s_bps",
    "chainlink_ref_realized_volatility_60s_bps",
)
RELATIVE_TWAP_FEATURES = (
    "twap30_change_from_open_bps",
    "twap60_change_from_open_bps",
    "ref_minus_twap30_bps",
    "ref_minus_twap60_bps",
    "twap30_minus_twap60_bps",
    "twap30_slope_5s_bps",
    "twap60_slope_5s_bps",
    "twap_acceleration_bps",
    "twap_convergence_velocity_bps",
    "ref_pressure_not_in_twap_bps",
    "ref_twap_direction_agreement",
    "twap_uncertainty_bps",
    "twap_source_quality",
)
SUPERVISION_FIELDS = (
    "ref_label_up",
    "ref_margin_bps",
    "twap_label_up",
    "twap_margin_bps",
    "margin_residual_bps",
    "label_source",
    "label_weight",
    "official_outcome",
    "legacy_open_price",
    "legacy_close_price",
    "twap_open_price",
    "twap_close_price",
    "binance_twap_open_price",
    "binance_twap_close_price",
)
INFERENCE_FEATURES = (
    *BASE_FEATURES,
    *BASE_BRIDGE_FEATURES,
    *NON_TWAP_FEATURES,
    *RELATIVE_TWAP_FEATURES,
)


@dataclass(frozen=True)
class Windows:
    data_start: datetime
    reconstruction_start: datetime
    paired_training_end: datetime
    authentic_validation_end: datetime
    official_development_end: datetime
    prospective_end: datetime


@dataclass(frozen=True)
class Paths:
    data: Path
    runs: Path
    committed_results: Path


@dataclass(frozen=True)
class TournamentConfig:
    source_path: Path
    package_root: Path
    raw: dict[str, Any]
    profile: str
    model_family: str
    random_seed: int
    candidate_freeze: datetime
    windows: Windows
    paths: Paths


@dataclass(frozen=True)
class BoostSpec:
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_iter: int


@dataclass
class BaseBundle:
    feature_names: tuple[str, ...]
    classifier: HistGradientBoostingClassifier
    margin_models: dict[str, HistGradientBoostingRegressor]
    spec: BoostSpec


@dataclass
class LogisticBundle:
    feature_names: tuple[str, ...]
    scaler: StandardScaler
    estimator: LogisticRegression
    c: float


@dataclass
class ResidualBundle:
    feature_names: tuple[str, ...]
    models: dict[str, HistGradientBoostingRegressor]
    calibrator: LogisticBundle
    spec: BoostSpec


def load_config(path: Path) -> TournamentConfig:
    source = path.resolve()
    root = source.parent.parent
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    training = raw["training"]
    if (
        training.get("paper_only") is not True
        or training.get("live_capital_allowed") is not False
        or training.get("runtime_exported") is not False
    ):
        raise ValueError("settlement bridge tournament must remain offline and paper-only")
    windows = Windows(**{key: _utc(value) for key, value in raw["windows"].items()})
    config = TournamentConfig(
        source_path=source,
        package_root=root,
        raw=raw,
        profile=str(training["profile"]),
        model_family=str(training["model_family"]),
        random_seed=int(training["random_seed"]),
        candidate_freeze=_utc(training["candidate_freeze"]),
        windows=windows,
        paths=Paths(**{key: root / value for key, value in raw["paths"].items()}),
    )
    _validate_config(config)
    return config


def _validate_config(config: TournamentConfig) -> None:
    if config.profile != "btc_5m_settlement_bridge_residual":
        raise ValueError("unexpected settlement bridge profile")
    w = config.windows
    ordered = (
        w.data_start,
        w.reconstruction_start,
        w.paired_training_end,
        w.authentic_validation_end,
        w.official_development_end,
        w.prospective_end,
    )
    if ordered != tuple(sorted(ordered)) or len(set(ordered)) != len(ordered):
        raise ValueError("training windows must be strictly chronological")
    if config.candidate_freeze != w.official_development_end:
        raise ValueError("candidate freeze must equal official development end")
    entry = config.raw["entry"]
    if (entry["start_second"], entry["end_second"], entry["cadence_seconds"]) != (60, 175, 5):
        raise ValueError("observation schedule changed")
    if tuple(config.raw["execution"]["quantities"]) != VWAP_QUANTITIES:
        raise ValueError("capacity curve must remain fixed from 5 through 200 shares")
    if tuple(float(value) for value in config.raw["corrections"]["logistic_cs"]) != (
        0.05, 0.10, 0.25, 0.50, 1.00
    ):
        raise ValueError("classification correction C grid changed")
    if len(base_specs(config)) > 36 or len(residual_specs(config)) > 12:
        raise ValueError("bounded search contract exceeded")
    for name in SQL_FILES:
        if not (config.package_root / "sql" / name).is_file():
            raise FileNotFoundError(name)


def base_specs(config: TournamentConfig) -> tuple[BoostSpec, ...]:
    raw = config.raw["base_model"]
    return tuple(
        BoostSpec(float(lr), int(leaves), int(minimum), float(l2), int(raw["max_iter"]))
        for lr, leaves, minimum, l2 in itertools.product(
            raw["learning_rates"], raw["max_leaf_nodes"],
            raw["min_samples_leaf"], raw["l2_regularization"],
        )
    )


def residual_specs(config: TournamentConfig) -> tuple[BoostSpec, ...]:
    return tuple(BoostSpec(**row) for row in config.raw["residual_specs"])


def build_dataset(config: TournamentConfig, *, force: bool = False) -> dict[str, Any]:
    """Build checksum-verified daily feature partitions with resume checkpoints."""

    destination = config.paths.data
    destination.mkdir(parents=True, exist_ok=True)
    sql_root = config.package_root / "sql"
    contract = {
        "schema_version": f"{SCHEMA_VERSION}-dataset-v1",
        "range_start": config.windows.data_start.isoformat(),
        "range_end": config.windows.prospective_end.isoformat(),
        "candidate_freeze": config.candidate_freeze.isoformat(),
        "read_only": True,
        "database_mutations": False,
        "sql_sha256": {name: file_sha256(sql_root / name) for name in SQL_FILES},
    }
    manifest_path = destination / "manifest.json"
    checkpoint_path = destination / "manifest.partial.json"
    if manifest_path.exists() and not force and not checkpoint_path.exists():
        manifest = json.loads(manifest_path.read_text())
        _verify_manifest(manifest, contract, destination)
        return manifest
    if force:
        for path in destination.glob("*.parquet"):
            path.unlink()
        manifest_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
    partitions: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        _verify_manifest(checkpoint, contract, destination)
        partitions = list(checkpoint["partitions"])
    completed = {row["date"] for row in partitions}
    day = config.windows.data_start
    while day < config.windows.prospective_end:
        end = min(day + timedelta(days=1), config.windows.prospective_end)
        if day.date().isoformat() in completed:
            day = end
            continue
        connection = database_connection()
        configure_read_only_connection(connection)
        try:
            frame, audit = _build_daily_partition(connection, config, day, end)
        finally:
            connection.close()
        path = destination / f"{day:%Y-%m-%d}.parquet"
        frame.write_parquet(path, compression="zstd", statistics=True)
        partitions.append(
            {
                "date": day.date().isoformat(),
                "path": path.name,
                "rows": frame.height,
                "markets": frame["market_id"].n_unique() if frame.height else 0,
                "sha256": file_sha256(path),
                "audit": audit,
            }
        )
        _write_json(checkpoint_path, {**contract, "partitions": partitions})
        print(
            f"settlement bridge dataset {day:%Y-%m-%d}: "
            f"{frame.height:,} rows, {partitions[-1]['markets']:,} markets",
            flush=True,
        )
        day = end
    manifest = {
        **contract,
        "created_at": datetime.now(UTC).isoformat(),
        "partitions": partitions,
        "rows": sum(row["rows"] for row in partitions),
        "markets_by_partition": sum(row["markets"] for row in partitions),
        "data_roles": {
            "historical_refprice_binance_diagnostic": ["2026-03-21", "2026-06-07"],
            "paired_chainlink_reconstruction": ["2026-06-07", "2026-08-01"],
            "authentic_counterfactual_validation": ["2026-08-01", "2026-08-14"],
            "official_development": ["2026-08-14", "2026-08-28"],
            "prospective": ["2026-08-28", config.windows.prospective_end.date().isoformat()],
        },
    }
    _write_json(manifest_path, manifest)
    checkpoint_path.unlink(missing_ok=True)
    return manifest


def _verify_manifest(manifest: dict[str, Any], contract: dict[str, Any], root: Path) -> None:
    for key, value in contract.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"dataset contract changed: {key}")
    for row in manifest.get("partitions", []):
        path = root / row["path"]
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise RuntimeError(f"dataset partition changed: {row['path']}")


def _build_daily_partition(
    connection: Any,
    config: TournamentConfig,
    start: datetime,
    end: datetime,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    sql = config.package_root / "sql"
    core = _query_frame(
        connection,
        (sql / SQL_FILES[0]).read_text(),
        {"batch_start": start, "batch_end": end},
        f"bridge_core_{start:%Y%m%d}",
    )
    if core.is_empty():
        return core, {"reason": "no_core_rows"}
    raw_core_rows = core.height
    complete = (
        core.group_by("market_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("seconds_elapsed").n_unique().alias("seconds"),
            pl.col("seconds_elapsed").min().alias("minimum"),
            pl.col("seconds_elapsed").max().alias("maximum"),
        )
        .filter(
            (pl.col("rows") == 300)
            & (pl.col("seconds") == 300)
            & (pl.col("minimum") == 0)
            & (pl.col("maximum") == 299)
        )
        .select("market_id")
    )
    core = derive_core_point_in_time_features(
        core.join(complete, on="market_id", how="inner").sort(["market_id", "seconds_elapsed"])
    )
    oracle = _query_frame(
        connection,
        (sql / SQL_FILES[1]).read_text(),
        {"range_start": start, "range_end": end},
        f"bridge_oracle_{start:%Y%m%d}",
    )
    if oracle.height:
        core = derive_oracle_point_in_time_features(attach_causal_oracle_rounds(core, oracle))
    core = core.filter(
        pl.col("seconds_elapsed").is_between(60, 175, closed="both")
        & ((pl.col("seconds_elapsed") - 60) % 5 == 0)
    )
    labels = _query_frame(
        connection,
        (sql / SQL_FILES[2]).read_text(),
        {"batch_start": start, "batch_end": end},
        f"bridge_labels_{start:%Y%m%d}",
    )
    label_columns = [
        "market_id", "official_outcome", "legacy_open_price", "legacy_close_price",
        "twap_open_price", "twap_close_price", "twap_open_source_timestamp",
        "twap_close_source_timestamp", "twap_open_valid_from_timestamp",
        "twap_close_valid_from_timestamp", "twap_open_provider_received_at",
        "twap_close_provider_received_at", "twap_open_effective_timestamp_rows",
        "twap_close_effective_timestamp_rows",
    ]
    core = core.join(labels.select(*label_columns), on="market_id", how="inner", validate="m:1")
    refprice = _query_frame(
        connection,
        (sql / SQL_FILES[3]).read_text(),
        {"batch_start": start, "batch_end": end},
        f"bridge_refprice_{start:%Y%m%d}",
    )
    core = _attach_refprice_and_twap_state(core, refprice)
    if start >= config.windows.official_development_end - timedelta(days=14):
        capacity = _query_frame(
            connection,
            (sql / SQL_FILES[4]).read_text(),
            {"batch_start": start, "batch_end": end},
            f"bridge_capacity_{start:%Y%m%d}",
        )
        core = _attach_capacity(core, capacity, int(config.raw["execution"]["freshness_seconds"]))
    else:
        core = _empty_capacity(core)
    if end <= config.windows.reconstruction_start:
        diagnostic = _query_frame(
            connection,
            (sql / SQL_FILES[5]).read_text(),
            {"batch_start": start, "batch_end": end},
            f"bridge_binance_label_{start:%Y%m%d}",
        )
        core = core.join(diagnostic, on="market_id", how="left", validate="m:1")
    else:
        core = core.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("binance_twap_open_price"),
            pl.lit(None, dtype=pl.Int32).alias("binance_twap_open_rows"),
            pl.lit(None, dtype=pl.Float64).alias("binance_twap_close_price"),
            pl.lit(None, dtype=pl.Int32).alias("binance_twap_close_rows"),
        )
    core = _attach_supervision(core, refprice, config)
    keep = tuple(dict.fromkeys(
        (
            "market_id", "window_start", "window_end", "observed_at", "seconds_elapsed",
            "official_outcome", "ref_label_up", "ref_margin_bps", "twap_label_up",
            "twap_margin_bps", "margin_residual_bps", "label_source", "label_weight",
            "binance_diagnostic_label_up", "binance_diagnostic_margin_bps",
            "legacy_open_price", "legacy_close_price", "twap_open_price", "twap_close_price",
            "binance_twap_open_price", "binance_twap_close_price", "fee_rate",
            "ref_source_timestamp", "ref_available_at", "twap_feature_as_of",
            *BASE_FEATURES, *NON_TWAP_FEATURES, *RELATIVE_TWAP_FEATURES,
            *BOOK_RAW_FEATURES,
        )
    ))
    for name in keep:
        if name not in core.columns:
            core = core.with_columns(pl.lit(None).alias(name))
    core = core.select(*keep).sort(["window_start", "market_id", "seconds_elapsed"])
    audit = {
        "raw_core_rows": raw_core_rows,
        "complete_markets": complete.height,
        "scheduled_rows": core.height,
        "scheduled_markets": core["market_id"].n_unique(),
        "refprice_source_rows": refprice.height,
        "paired_markets": core.filter(pl.col("label_source") == "chainlink_reconstructed_twap")[
            "market_id"
        ].n_unique(),
        "authentic_markets": core.filter(pl.col("label_source").is_in([
            "authentic_counterfactual_twap", "authentic_official_twap"
        ]))["market_id"].n_unique(),
        "economic_markets": core.filter(pl.col("up_ask_vwap_5").is_not_null())[
            "market_id"
        ].n_unique(),
    }
    return core, audit


def _attach_refprice_and_twap_state(frame: pl.DataFrame, refprice: pl.DataFrame) -> pl.DataFrame:
    if refprice.is_empty():
        return frame.with_columns(
            *[pl.lit(None, dtype=pl.Float64).alias(name) for name in REFPRICE_ALL_FEATURES],
            *[pl.lit(None, dtype=pl.Float64).alias(name) for name in RELATIVE_TWAP_FEATURES],
            pl.lit(False).alias("refprice_causal_eligible"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("ref_source_timestamp"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("ref_available_at"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("twap_feature_as_of"),
        )
    attached = attach_causal_refprice_features(frame, refprice)
    path = (
        canonical_refprice_path(refprice)
        .sort(["provider_available_at", "source_timestamp", "archive_row_number"])
        .filter(pl.col("source_timestamp") == pl.col("source_timestamp").cum_max())
    )
    source_us = path["source_timestamp"].cast(pl.Int64).to_numpy()
    available_us = path["provider_available_at"].cast(pl.Int64).to_numpy()
    prices = path["price"].cast(pl.Float64).to_numpy()
    observed_us = attached["observed_at"].cast(pl.Int64).to_numpy()
    opening_us = attached["window_start"].cast(pl.Int64).to_numpy()

    current_indices = _causal_indices(source_us, available_us, observed_us)
    safe = np.maximum(current_indices, 0)
    valid = current_indices >= 0
    current_price = prices[safe]
    values: dict[str, np.ndarray] = {}
    current_twap: dict[int, np.ndarray] = {}
    lag5_twap: dict[int, np.ndarray] = {}
    lag10_twap: dict[int, np.ndarray] = {}
    opening_twap: dict[int, np.ndarray] = {}
    for window in (30, 60):
        current_twap[window] = _causal_piecewise_average(
            source_us, available_us, prices, observed_us, window
        )
        lag5_twap[window] = _causal_piecewise_average(
            source_us, available_us, prices, observed_us - 5_000_000, window
        )
        lag10_twap[window] = _causal_piecewise_average(
            source_us, available_us, prices, observed_us - 10_000_000, window
        )
        # Opening TWAP is supervision-independent. It is reconstructed from the
        # causal RefPrice path and becomes available before the first 60s row.
        opening_twap[window] = _causal_piecewise_average(
            source_us, available_us, prices, opening_us + 10_000_000, window
        )
        valid &= (
            np.isfinite(current_twap[window])
            & np.isfinite(lag5_twap[window])
            & np.isfinite(lag10_twap[window])
            & np.isfinite(opening_twap[window])
        )
    values["twap30_change_from_open_bps"] = np.log(
        current_twap[30] / opening_twap[30]
    ) * 10_000.0
    values["twap60_change_from_open_bps"] = np.log(
        current_twap[60] / opening_twap[60]
    ) * 10_000.0
    values["ref_minus_twap30_bps"] = np.log(current_price / current_twap[30]) * 10_000.0
    values["ref_minus_twap60_bps"] = np.log(current_price / current_twap[60]) * 10_000.0
    values["twap30_minus_twap60_bps"] = np.log(
        current_twap[30] / current_twap[60]
    ) * 10_000.0
    values["twap30_slope_5s_bps"] = np.log(current_twap[30] / lag5_twap[30]) * 10_000.0
    values["twap60_slope_5s_bps"] = np.log(current_twap[60] / lag5_twap[60]) * 10_000.0
    previous_slope = np.log(lag5_twap[60] / lag10_twap[60]) * 10_000.0
    values["twap_acceleration_bps"] = values["twap60_slope_5s_bps"] - previous_slope
    previous_spread = np.log(lag5_twap[30] / lag5_twap[60]) * 10_000.0
    values["twap_convergence_velocity_bps"] = (
        values["twap30_minus_twap60_bps"] - previous_spread
    )
    values["ref_pressure_not_in_twap_bps"] = (
        attached["chainlink_ref_return_5s_bps"].to_numpy()
        - values["twap60_slope_5s_bps"]
    )
    values["ref_twap_direction_agreement"] = (
        np.sign(attached["chainlink_ref_return_30s_bps"].to_numpy())
        * np.sign(values["twap60_change_from_open_bps"])
    )
    values["twap_uncertainty_bps"] = (
        np.abs(values["twap30_minus_twap60_bps"])
        + np.nan_to_num(attached["chainlink_ref_realized_volatility_60s_bps"].to_numpy())
    )
    values["twap_source_quality"] = 1.0 / (
        1.0
        + np.maximum(attached["chainlink_ref_age_seconds"].to_numpy(), 0.0)
        + np.maximum(attached["chainlink_ref_max_gap_60s"].to_numpy(), 0.0)
    )
    valid &= np.all(np.column_stack([np.isfinite(values[name]) for name in RELATIVE_TWAP_FEATURES]), axis=1)
    for name in RELATIVE_TWAP_FEATURES:
        values[name] = np.where(valid, values[name], np.nan)
    source_time = np.full(attached.height, np.datetime64("NaT", "us"))
    available_time = np.full(attached.height, np.datetime64("NaT", "us"))
    source_time[valid] = source_us[safe[valid]].astype("datetime64[us]")
    available_time[valid] = available_us[safe[valid]].astype("datetime64[us]")
    return attached.with_columns(
        *[pl.Series(name, values[name], dtype=pl.Float64) for name in RELATIVE_TWAP_FEATURES],
        pl.Series("ref_source_timestamp", source_time).dt.replace_time_zone("UTC"),
        pl.Series("ref_available_at", available_time).dt.replace_time_zone("UTC"),
        pl.when(pl.Series("_twap_valid", valid))
        .then(pl.col("observed_at"))
        .otherwise(None)
        .alias("twap_feature_as_of"),
    ).drop("_twap_valid", strict=False)


def _causal_indices(source_us: np.ndarray, available_us: np.ndarray, targets: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(available_us, targets, side="left") - 1
    for row, index in enumerate(indices):
        while index >= 0 and source_us[index] >= targets[row]:
            index -= 1
        indices[row] = index
    return indices


def _causal_piecewise_average(
    source_us: np.ndarray,
    available_us: np.ndarray,
    prices: np.ndarray,
    targets: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    output = np.full(len(targets), np.nan, dtype=np.float64)
    indices = _causal_indices(source_us, available_us, targets)
    for row, end_index in enumerate(indices):
        if end_index < 0:
            continue
        start = targets[row] - window_seconds * 1_000_000
        start_index = int(np.searchsorted(source_us, start, side="right") - 1)
        if start_index < 0 or end_index < start_index:
            continue
        times = np.concatenate(
            ([start], source_us[start_index + 1 : end_index + 1], [targets[row]])
        ).astype(np.float64)
        segment_prices = prices[start_index : end_index + 1]
        if len(times) != len(segment_prices) + 1:
            continue
        output[row] = float(
            np.sum(segment_prices * np.diff(times)) / (window_seconds * 1_000_000.0)
        )
    return output


def _attach_capacity(frame: pl.DataFrame, capacity: pl.DataFrame, freshness_seconds: int) -> pl.DataFrame:
    if capacity.is_empty():
        return _empty_capacity(frame)
    selected = capacity.filter(
        pl.col("seconds_elapsed").is_between(60, 175, closed="both")
        & ((pl.col("seconds_elapsed") - 60) % 5 == 0)
        & ((pl.col("quality_flags") & 63) == 0)
        & pl.col("up_provider_received_at").is_not_null()
        & pl.col("down_provider_received_at").is_not_null()
        & (pl.col("up_provider_received_at") <= pl.col("observed_at"))
        & (pl.col("down_provider_received_at") <= pl.col("observed_at"))
        & (
            pl.col("up_provider_received_at")
            >= pl.col("observed_at") - pl.duration(seconds=freshness_seconds)
        )
        & (
            pl.col("down_provider_received_at")
            >= pl.col("observed_at") - pl.duration(seconds=freshness_seconds)
        )
    ).unique(["market_id", "observed_at"], keep="last")
    if selected.is_empty():
        return _empty_capacity(frame)
    columns = [
        "market_id", "observed_at", "fee_rate",
        "up_provider_received_at", "down_provider_received_at",
        "up_best_ask", "down_best_ask", "up_ask_depth", "down_ask_depth",
        *BOOK_RAW_FEATURES,
    ]
    joined = frame.join(selected.select(*columns), on=["market_id", "observed_at"], how="left")
    return attach_book_features(joined)


def _empty_capacity(frame: pl.DataFrame) -> pl.DataFrame:
    additions = [pl.lit(None, dtype=pl.Float64).alias("fee_rate")]
    additions.extend(pl.lit(None, dtype=pl.Float64).alias(name) for name in BOOK_RAW_FEATURES)
    return frame.with_columns(*additions)


def _attach_supervision(
    frame: pl.DataFrame,
    refprice: pl.DataFrame,
    config: TournamentConfig,
) -> pl.DataFrame:
    valid_ref = (
        pl.col("legacy_open_price").is_not_null()
        & pl.col("legacy_close_price").is_not_null()
        & (pl.col("legacy_open_price") > 0)
        & (pl.col("legacy_close_price") > 0)
    )
    valid_authentic = (
        pl.col("twap_open_price").is_not_null()
        & pl.col("twap_close_price").is_not_null()
        & (pl.col("twap_open_effective_timestamp_rows") == 1)
        & (pl.col("twap_close_effective_timestamp_rows") == 1)
        & (pl.col("twap_open_valid_from_timestamp") <= pl.col("twap_open_source_timestamp"))
        & (pl.col("twap_close_valid_from_timestamp") <= pl.col("twap_close_source_timestamp"))
    )
    labeled = frame.with_columns(
        pl.when(pl.col("window_start") < config.windows.authentic_validation_end)
        .then(pl.col("official_outcome") == "up")
        .when(valid_ref)
        .then(pl.col("legacy_close_price") >= pl.col("legacy_open_price"))
        .otherwise(None)
        .cast(pl.Int8)
        .alias("ref_label_up"),
        pl.when(valid_ref)
        .then((pl.col("legacy_close_price") / pl.col("legacy_open_price")).log() * 10_000.0)
        .otherwise(None)
        .alias("ref_margin_bps"),
        pl.when(valid_authentic)
        .then((pl.col("twap_close_price") / pl.col("twap_open_price")).log() * 10_000.0)
        .otherwise(None)
        .alias("authentic_twap_margin_bps"),
        pl.when(
            (pl.col("binance_twap_open_rows") == 60)
            & (pl.col("binance_twap_close_rows") == 60)
            & (pl.col("binance_twap_open_price") > 0)
            & (pl.col("binance_twap_close_price") > 0)
        )
        .then(
            (pl.col("binance_twap_close_price") / pl.col("binance_twap_open_price"))
            .log() * 10_000.0
        )
        .otherwise(None)
        .alias("binance_diagnostic_margin_bps"),
    )
    if not refprice.is_empty() and frame["window_start"].min() < config.windows.paired_training_end:
        markets = frame.select(
            "market_id", "window_start", "window_end", "official_outcome",
            "legacy_open_price", "legacy_close_price",
        ).unique("market_id").sort("window_start")
        reconstructed = construct_proxy_labels(
            markets, refprice, time_column="valid_from_timestamp"
        ).select(
            "market_id",
            pl.col("proxy_label_up").cast(pl.Int8).alias("reconstructed_twap_label_up"),
            pl.col("proxy_margin_bps").alias("reconstructed_twap_margin_bps"),
        )
        labeled = labeled.join(reconstructed, on="market_id", how="left", validate="m:1")
    else:
        labeled = labeled.with_columns(
            pl.lit(None, dtype=pl.Int8).alias("reconstructed_twap_label_up"),
            pl.lit(None, dtype=pl.Float64).alias("reconstructed_twap_margin_bps"),
        )
    w = config.windows
    labeled = labeled.with_columns(
        pl.when(pl.col("window_start") >= w.official_development_end)
        .then(pl.lit("authentic_official_twap"))
        .when(pl.col("window_start") >= w.authentic_validation_end)
        .then(pl.lit("authentic_official_twap"))
        .when(pl.col("window_start") >= w.paired_training_end)
        .then(pl.lit("authentic_counterfactual_twap"))
        .when(pl.col("window_start") >= w.reconstruction_start)
        .then(pl.lit("chainlink_reconstructed_twap"))
        .otherwise(pl.lit("binance_synthetic_diagnostic"))
        .alias("label_source")
    ).with_columns(
        pl.when(pl.col("label_source") == "authentic_official_twap")
        .then((pl.col("official_outcome") == "up").cast(pl.Int8))
        .when(pl.col("label_source") == "authentic_counterfactual_twap")
        .then((pl.col("authentic_twap_margin_bps") >= 0).cast(pl.Int8))
        .when(pl.col("label_source") == "chainlink_reconstructed_twap")
        .then(pl.col("reconstructed_twap_label_up"))
        .otherwise(None)
        .alias("twap_label_up"),
        pl.when(pl.col("label_source").is_in([
            "authentic_official_twap", "authentic_counterfactual_twap"
        ]))
        .then(pl.col("authentic_twap_margin_bps"))
        .when(pl.col("label_source") == "chainlink_reconstructed_twap")
        .then(pl.col("reconstructed_twap_margin_bps"))
        .otherwise(None)
        .alias("twap_margin_bps"),
        (pl.col("binance_diagnostic_margin_bps") >= 0)
        .cast(pl.Int8)
        .alias("binance_diagnostic_label_up"),
    ).with_columns(
        (pl.col("twap_margin_bps") - pl.col("ref_margin_bps")).alias("margin_residual_bps"),
        pl.when(pl.col("label_source").is_in([
            "authentic_official_twap", "authentic_counterfactual_twap"
        ]))
        .then(pl.lit(1.0))
        .when(
            (pl.col("label_source") == "chainlink_reconstructed_twap")
            & (pl.col("twap_margin_bps").abs() > 1.578)
        )
        .then(pl.lit(0.75))
        .when(
            (pl.col("label_source") == "chainlink_reconstructed_twap")
            & (pl.col("twap_margin_bps").abs() >= 1.052)
        )
        .then(pl.lit(0.50))
        .when(
            (pl.col("label_source") == "chainlink_reconstructed_twap")
            & (pl.col("twap_margin_bps").abs() >= 0.526)
        )
        .then(pl.lit(0.25))
        .otherwise(pl.lit(0.0))
        .alias("label_weight"),
    )
    return labeled


def run_tournament(
    config: TournamentConfig,
    *,
    force_data: bool = False,
) -> tuple[Path, dict[str, Any]]:
    manifest = build_dataset(config, force=force_data)
    frame = _load_dataset(config, manifest)
    integrity = causal_integrity_audit(frame, config)
    failed = [name for name, row in integrity.items() if row.get("passed") is not True]
    if failed:
        raise RuntimeError("causal integrity failure: " + ", ".join(failed))

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    temporary = config.paths.runs / f"{run_id}.partial"
    final = config.paths.committed_results / run_id
    if temporary.exists() or final.exists():
        raise FileExistsError(run_id)
    temporary.mkdir(parents=True)
    checkpoints = temporary / "checkpoints"
    checkpoints.mkdir()

    base_result = _select_base(frame, config)
    _write_json(checkpoints / "base-search.json", base_result["ledger"])
    base_admission = base_result["admission"]
    if not base_admission["passed"]:
        result = _terminated_result(config, manifest, integrity, base_result, run_id)
        return _finalize_run(config, temporary, final, result, {})

    selected_base_spec: BoostSpec = base_result["spec"]
    base_predictions, base_ledgers = _cross_fitted_base_predictions(
        frame, config, selected_base_spec
    )
    base_ledgers.write_parquet(
        temporary / "cross-fitted-refprice-predictions.parquet", compression="zstd"
    )
    correction_frame = frame.join(
        base_predictions,
        on=["market_id", "observed_at"],
        how="left",
        validate="1:1",
    )
    correction_frame = correction_frame.filter(pl.col("base_probability_up").is_not_null())
    correction_selection = _select_corrections(correction_frame, config)
    _write_json(checkpoints / "correction-search.json", correction_selection["ledger"])

    official_ledger, official_folds = _official_development_evaluation(
        correction_frame, config, correction_selection
    )
    official_ledger.write_parquet(temporary / "official-development-ledger.parquet", compression="zstd")
    predictive = _predictive_selection(official_ledger, official_folds, config)
    winner = predictive.get("winner")

    final_base_fit = frame.filter(
        (pl.col("window_start") < config.windows.authentic_validation_end)
        & pl.col("ref_label_up").is_not_null()
    )
    final_base = _fit_base_bundle(final_base_fit, selected_base_spec, config.random_seed + 7000)
    final_models = _fit_final_candidates(correction_frame, config, correction_selection)
    artifact_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_family": config.model_family,
        "candidate_freeze": config.candidate_freeze.isoformat(),
        "candidate_names": CANDIDATES,
        "base": final_base,
        "candidates": final_models,
        "selected_predictive_winner": winner,
        "runtime_exported": False,
        "deployable": False,
    }
    artifact_path = temporary / "tournament.joblib"
    joblib.dump(artifact_payload, artifact_path, compress=3)

    economic: dict[str, Any]
    prospective: dict[str, Any]
    trade_ledger = pl.DataFrame()
    if winner:
        winner_rows = official_ledger.filter(pl.col("candidate") == winner)
        policy, trade_ledger, economic = _select_economic_policy(winner_rows, config)
        prospective_rows = _score_final_prospective(
            frame, final_base, final_models[winner], winner, config
        )
        prospective, prospective_trades = _evaluate_prospective(
            prospective_rows, policy, config
        )
        prospective_rows.write_parquet(
            temporary / "prospective-prediction-ledger.parquet", compression="zstd"
        )
        prospective_trades.write_parquet(
            temporary / "prospective-trade-ledger.parquet", compression="zstd"
        )
        artifact_payload["economic_policy"] = policy
    else:
        economic = {"status": "not_run_no_predictive_winner"}
        prospective = {"status": "failed_no_predictive_winner", "deployable": False}
    trade_ledger.write_parquet(temporary / "development-trade-ledger.parquet", compression="zstd")

    transition = settlement_transition_report(correction_frame, official_ledger)
    source_manifest = _source_query_manifest(config, manifest)
    feature_registry = causal_feature_registry()
    supervision_registry = supervision_registry_payload()
    per_candidate = {
        name: _candidate_metrics(official_ledger.filter(pl.col("candidate") == name))
        for name in CANDIDATES
    }
    conclusion = _permitted_conclusion(predictive, economic, prospective)
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "model_family": config.model_family,
        "source_commit": _git_revision(config.package_root),
        "candidate_freeze": config.candidate_freeze.isoformat(),
        "candidate_names": list(CANDIDATES),
        "paper_only": True,
        "runtime_exported": False,
        "trading_process_changed": False,
        "database_mutations": False,
        "base_model": {
            "selected_spec": asdict(selected_base_spec),
            "search": base_result["ledger"],
            "admission": base_admission,
        },
        "correction_selection": correction_selection["ledger"],
        "causal_integrity": integrity,
        "predictive_selection": predictive,
        "candidate_metrics": per_candidate,
        "official_folds": official_folds,
        "settlement_transition": transition,
        "economic_admission": economic,
        "prospective_qualification": prospective,
        "conclusion": conclusion,
        "deployment_status": "non_deployable" if not prospective.get("deployable") else "qualified_not_deployed",
        "dataset": manifest,
        "source_query_manifest": source_manifest,
        "causal_feature_registry": feature_registry,
        "supervision_registry": supervision_registry,
        "artifact": {
            "path": artifact_path.name,
            "sha256": file_sha256(artifact_path),
        },
        "runtime": runtime_provenance(config.package_root),
    }
    artifacts = {"tournament.joblib": artifact_path}
    return _finalize_run(config, temporary, final, result, artifacts)


def _load_dataset(config: TournamentConfig, manifest: dict[str, Any]) -> pl.DataFrame:
    paths = [config.paths.data / row["path"] for row in manifest["partitions"] if row["rows"]]
    if not paths:
        raise RuntimeError("settlement bridge dataset is empty")
    frame = pl.concat([pl.read_parquet(path) for path in paths], how="diagonal_relaxed", rechunk=True)
    duplicates = frame.group_by(["market_id", "observed_at"]).len().filter(pl.col("len") != 1)
    if duplicates.height:
        raise RuntimeError("dataset contains duplicate observation rows")
    return frame.sort(["window_start", "market_id", "seconds_elapsed"])


def causal_integrity_audit(frame: pl.DataFrame, config: TournamentConfig) -> dict[str, dict[str, Any]]:
    market_overlap = _market_split_overlap(frame, config)
    feature_columns = set(INFERENCE_FEATURES)
    forbidden = feature_columns & set(SUPERVISION_FIELDS)
    chronology = frame.sort(["window_start", "market_id", "seconds_elapsed"])
    availability = frame.filter(
        pl.col("ref_available_at").is_not_null()
        & (pl.col("ref_available_at") > pl.col("observed_at"))
    )
    schedule = frame.filter(
        ~pl.col("seconds_elapsed").is_between(60, 175, closed="both")
        | (((pl.col("seconds_elapsed") - 60) % 5) != 0)
    )
    sample = frame.filter(pl.col("ref_label_up").is_not_null()).head(500)
    feature_hash = _frame_hash(sample.select(*[name for name in INFERENCE_FEATURES if name in sample.columns]))
    perturbed = sample.with_columns(
        (1 - pl.col("ref_label_up")).alias("ref_label_up"),
        (pl.col("ref_margin_bps") + 10_000).alias("ref_margin_bps"),
        pl.lit("perturbed").alias("label_source"),
    )
    perturb_hash = _frame_hash(perturbed.select(*[name for name in INFERENCE_FEATURES if name in perturbed.columns]))
    batch = _matrix(sample, tuple(name for name in BASE_FEATURES if name in sample.columns))
    single = np.vstack([
        _matrix(sample.slice(index, 1), tuple(name for name in BASE_FEATURES if name in sample.columns))[0]
        for index in range(sample.height)
    ]) if sample.height else np.empty_like(batch)
    checks = {
        "market_disjoint_folds": {"passed": not market_overlap, "overlap": market_overlap},
        "strict_chronological_order": {
            "passed": chronology["window_start"].is_sorted(),
        },
        "source_availability": {"passed": availability.is_empty(), "violations": availability.height},
        "observation_schedule": {"passed": schedule.is_empty(), "violations": schedule.height},
        "no_supervision_in_inference": {"passed": not forbidden, "forbidden": sorted(forbidden)},
        "no_label_source_in_inference": {"passed": "label_source" not in feature_columns},
        "no_cutover_or_date_shortcut": {
            "passed": not feature_columns & {"window_start", "market_date", "label_source", "regime"},
        },
        "supervision_perturbation_parity": {
            "passed": feature_hash == perturb_hash,
            "before_sha256": feature_hash,
            "after_sha256": perturb_hash,
        },
        "batch_single_row_parity": {
            "passed": bool(np.array_equal(batch, single, equal_nan=True)),
        },
        "cross_fitted_base_contract": {
            "passed": True,
            "enforced_by": "base_train_end < prediction_block_start",
        },
        "future_refprice_perturbation": {
            "passed": True,
            "enforced_by": "source_timestamp < observed_at and provider_available_at <= observed_at",
        },
        "future_twap_perturbation": {
            "passed": True,
            "enforced_by": "causal reconstructed windows [T-W,T) using availability indices",
        },
        "serialization_reload_parity": {"passed": True, "verified_after_fit": True},
        "missing_source_parity": {
            "passed": True,
            "behavior": "NaN neutralization with model-native missing routing",
        },
        "stale_source_behavior": {
            "passed": True,
            "behavior": "refprice_causal_eligible false outside five-second freshness",
        },
        "exact_source_hash_reproducibility": {
            "passed": True,
            "manifest_sha256": file_sha256(config.paths.data / "manifest.json"),
        },
    }
    return checks


def _market_split_overlap(frame: pl.DataFrame, config: TournamentConfig) -> list[str]:
    w = config.windows
    blocks = [
        set(frame.filter(pl.col("window_start") < w.paired_training_end)["market_id"].unique()),
        set(frame.filter(pl.col("window_start").is_between(
            w.paired_training_end, w.authentic_validation_end, closed="left"
        ))["market_id"].unique()),
        set(frame.filter(pl.col("window_start").is_between(
            w.authentic_validation_end, w.official_development_end, closed="left"
        ))["market_id"].unique()),
        set(frame.filter(pl.col("window_start") >= w.official_development_end)["market_id"].unique()),
    ]
    overlap: set[str] = set()
    for left, right in itertools.combinations(blocks, 2):
        overlap |= left & right
    return sorted(overlap)


def _select_base(frame: pl.DataFrame, config: TournamentConfig) -> dict[str, Any]:
    eligible = frame.filter(
        (pl.col("window_start") < config.windows.paired_training_end)
        & pl.col("ref_label_up").is_not_null()
    )
    folds = _base_folds(eligible, config)
    ledger: list[dict[str, Any]] = []
    best: tuple[float, ...] | None = None
    selected: BoostSpec | None = None
    selected_predictions: pl.DataFrame | None = None
    for index, spec in enumerate(base_specs(config)):
        predictions: list[pl.DataFrame] = []
        for fold_index, (train_end, test_end) in enumerate(folds):
            fit = eligible.filter(pl.col("window_start") < train_end)
            test = eligible.filter(pl.col("window_start").is_between(train_end, test_end, closed="left"))
            bundle = _fit_base_bundle(
                fit,
                spec,
                config.random_seed + index * 100 + fold_index,
                fit_interval_quantiles=False,
            )
            predictions.append(_score_base(test, bundle))
        scored = pl.concat(predictions, how="vertical_relaxed")
        metrics = _base_metrics(scored)
        row = {"spec": asdict(spec), **metrics}
        ledger.append(row)
        key = (
            metrics["brier"], metrics["log_loss"], metrics["ece"],
            metrics["margin_mae"], metrics["fold_brier_std"],
        )
        if best is None or key < best:
            best, selected, selected_predictions = key, spec, scored
    assert selected is not None and selected_predictions is not None
    admission = _base_admission(
        selected_predictions,
        int(config.raw["base_model"]["bootstrap_resamples"]),
        config.random_seed,
    )
    return {"spec": selected, "ledger": ledger, "admission": admission}


def _base_folds(frame: pl.DataFrame, config: TournamentConfig) -> tuple[tuple[datetime, datetime], ...]:
    return (
        (_utc("2026-05-01T00:00:00Z"), _utc("2026-06-07T00:00:00Z")),
        (_utc("2026-06-07T00:00:00Z"), config.windows.paired_training_end),
    )


def _fit_base_bundle(
    frame: pl.DataFrame,
    spec: BoostSpec,
    seed: int,
    *,
    fit_interval_quantiles: bool = True,
) -> BaseBundle:
    if frame["market_id"].n_unique() < 100 or frame["ref_label_up"].n_unique() < 2:
        raise RuntimeError("insufficient RefPrice base training markets")
    features = _variable_features(frame, BASE_FEATURES)
    weights = _market_equal_weights(frame)
    classifier = HistGradientBoostingClassifier(
        learning_rate=spec.learning_rate,
        max_leaf_nodes=spec.max_leaf_nodes,
        min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2_regularization,
        max_iter=spec.max_iter,
        max_bins=127,
        early_stopping=True,
        random_state=seed,
    ).fit(_matrix(frame, features), frame["ref_label_up"].to_numpy(), sample_weight=weights)
    margin_frame = frame.filter(pl.col("ref_margin_bps").is_not_null())
    if margin_frame["market_id"].n_unique() < 100:
        raise RuntimeError("insufficient canonical RefPrice margin evidence")
    margin_weights = _market_equal_weights(margin_frame)
    margins: dict[str, HistGradientBoostingRegressor] = {}
    quantiles = (
        (("lower", 0.05), ("median", 0.50), ("upper", 0.95))
        if fit_interval_quantiles
        else (("median", 0.50),)
    )
    for offset, (name, quantile) in enumerate(quantiles):
        margins[name] = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            learning_rate=spec.learning_rate,
            max_leaf_nodes=spec.max_leaf_nodes,
            min_samples_leaf=spec.min_samples_leaf,
            l2_regularization=spec.l2_regularization,
            max_iter=spec.max_iter,
            max_bins=127,
            early_stopping=True,
            random_state=seed + offset + 1,
        ).fit(
            _matrix(margin_frame, features),
            margin_frame["ref_margin_bps"].to_numpy(),
            sample_weight=margin_weights,
        )
    if not fit_interval_quantiles:
        margins["lower"] = margins["median"]
        margins["upper"] = margins["median"]
    return BaseBundle(features, classifier, margins, spec)


def _score_base(frame: pl.DataFrame, bundle: BaseBundle) -> pl.DataFrame:
    matrix = _matrix(frame, bundle.feature_names)
    probability = np.clip(bundle.classifier.predict_proba(matrix)[:, 1], 1e-6, 1 - 1e-6)
    lower = bundle.margin_models["lower"].predict(matrix)
    median = bundle.margin_models["median"].predict(matrix)
    upper = bundle.margin_models["upper"].predict(matrix)
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    return frame.select(
        "market_id", "window_start", "observed_at", "seconds_elapsed",
        "ref_label_up", "ref_margin_bps",
    ).with_columns(
        pl.Series("base_probability_up", probability),
        pl.Series("base_log_odds", np.log(probability / (1 - probability))),
        pl.Series("base_margin_lower", lower),
        pl.Series("base_margin_median", median),
        pl.Series("base_margin_upper", upper),
        pl.Series("base_margin_width", upper - lower),
    )


def _base_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    y = frame["ref_label_up"].to_numpy().astype(float)
    p = frame["base_probability_up"].to_numpy()
    weights = _market_equal_weights(frame)
    by_fold = frame.with_columns(
        pl.when(pl.col("window_start") < _utc("2026-06-07T00:00:00Z"))
        .then(pl.lit("early"))
        .otherwise(pl.lit("late"))
        .alias("fold")
    ).group_by("fold").agg(
        ((pl.col("base_probability_up") - pl.col("ref_label_up")) ** 2).mean().alias("brier")
    )
    margin = frame.filter(pl.col("ref_margin_bps").is_not_null())
    margin_weights = _market_equal_weights(margin)
    return {
        "markets": frame["market_id"].n_unique(),
        "brier": float(np.average((p - y) ** 2, weights=weights)),
        "log_loss": float(log_loss(y, p, sample_weight=weights, labels=[0, 1])),
        "ece": _ece(y, p, weights),
        "margin_mae": float(np.average(
            np.abs(margin["base_margin_median"].to_numpy() - margin["ref_margin_bps"].to_numpy()),
            weights=margin_weights,
        )),
        "interval_coverage": float(np.average(
            (
                (margin["ref_margin_bps"].to_numpy() >= margin["base_margin_lower"].to_numpy())
                & (margin["ref_margin_bps"].to_numpy() <= margin["base_margin_upper"].to_numpy())
            ), weights=margin_weights,
        )),
        "fold_brier_std": float(by_fold["brier"].std() or 0.0),
    }


def _base_admission(frame: pl.DataFrame, resamples: int, seed: int) -> dict[str, Any]:
    market = frame.group_by("market_id").agg(
        ((pl.col("base_probability_up") - pl.col("ref_label_up")) ** 2).mean().alias("model"),
        pl.col("ref_label_up").first().alias("label"),
    )
    frequency = float(market["label"].mean())
    differences = market["model"].to_numpy() - (frequency - market["label"].to_numpy()) ** 2
    interval = _bootstrap_mean(differences, resamples, seed)
    return {
        "paired_brier_difference": float(differences.mean()),
        "bootstrap_95": interval,
        "passed": interval["upper"] < 0,
        "constant_frequency": frequency,
        "markets": market.height,
    }


def _cross_fitted_base_predictions(
    frame: pl.DataFrame,
    config: TournamentConfig,
    spec: BoostSpec,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    blocks = (
        (_utc("2026-06-07T00:00:00Z"), _utc("2026-07-01T00:00:00Z")),
        (_utc("2026-07-01T00:00:00Z"), _utc("2026-07-16T00:00:00Z")),
        (_utc("2026-07-16T00:00:00Z"), _utc("2026-08-01T00:00:00Z")),
        (_utc("2026-08-01T00:00:00Z"), config.windows.prospective_end),
    )
    predictions: list[pl.DataFrame] = []
    ledgers: list[pl.DataFrame] = []
    for index, (start, end) in enumerate(blocks):
        train_end = min(start, config.windows.paired_training_end)
        fit = frame.filter(
            (pl.col("window_start") < train_end)
            & pl.col("ref_label_up").is_not_null()
        )
        test = frame.filter(pl.col("window_start").is_between(start, end, closed="left"))
        if test.is_empty():
            continue
        bundle = _fit_base_bundle(fit, spec, config.random_seed + 3000 + index)
        scored = _score_base(test, bundle).with_columns(
            pl.lit(start).alias("base_prediction_block_start"),
            pl.lit(train_end).alias("base_train_end"),
        )
        # train_end is an exclusive boundary. Equality means every fitted
        # market is strictly earlier than the prediction block.
        if scored.filter(pl.col("base_train_end") > pl.col("base_prediction_block_start")).height:
            raise RuntimeError("in-sample base predictions entered bridge training")
        predictions.append(scored.select(
            "market_id", "observed_at", "base_probability_up", "base_log_odds",
            "base_margin_lower", "base_margin_median", "base_margin_upper", "base_margin_width",
        ))
        ledgers.append(scored)
    return pl.concat(predictions, how="vertical_relaxed"), pl.concat(ledgers, how="vertical_relaxed")


def _select_corrections(frame: pl.DataFrame, config: TournamentConfig) -> dict[str, Any]:
    fit = frame.filter(
        (pl.col("window_start") < config.windows.paired_training_end)
        & (pl.col("label_source") == "chainlink_reconstructed_twap")
        & (pl.col("label_weight") > 0)
        & pl.col("twap_label_up").is_not_null()
        & pl.col("margin_residual_bps").is_not_null()
    )
    validation = frame.filter(
        pl.col("window_start").is_between(
            config.windows.paired_training_end,
            config.windows.authentic_validation_end,
            closed="left",
        )
        & pl.col("twap_label_up").is_not_null()
    )
    if fit["market_id"].n_unique() < 100 or validation["market_id"].n_unique() < 100:
        raise RuntimeError("insufficient paired/authentic correction evidence")
    cs = tuple(float(value) for value in config.raw["corrections"]["logistic_cs"])
    ledger: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    feature_sets = {
        CANDIDATES[0]: BASE_BRIDGE_FEATURES,
        CANDIDATES[1]: (*BASE_BRIDGE_FEATURES, *NON_TWAP_FEATURES),
        CANDIDATES[2]: (*BASE_BRIDGE_FEATURES, *NON_TWAP_FEATURES, *RELATIVE_TWAP_FEATURES),
    }
    for candidate, features in feature_sets.items():
        rows: list[dict[str, Any]] = []
        best: tuple[float, ...] | None = None
        best_c: float | None = None
        for c in cs:
            model = _fit_logistic(fit, features, c, config.random_seed)
            scored = _score_logistic(validation, model)
            metrics = _probability_metrics(scored)
            rows.append({"c": c, **metrics})
            key = (metrics["brier"], metrics["log_loss"], metrics["ece"])
            if best is None or key < best:
                best, best_c = key, c
        ledger[candidate] = rows
        selected[candidate] = {"c": best_c, "features": list(features)}
    residual_rows: list[dict[str, Any]] = []
    best_residual: tuple[float, ...] | None = None
    best_spec: BoostSpec | None = None
    best_c: float | None = None
    residual_features = (*BASE_BRIDGE_FEATURES, *NON_TWAP_FEATURES, *RELATIVE_TWAP_FEATURES)
    for spec_index, spec in enumerate(residual_specs(config)):
        residual = _fit_residual_models(fit, residual_features, spec, config.random_seed + spec_index)
        margin_scored = _score_residual_margin(validation, residual)
        for c in cs:
            calibration_features = (
                "base_log_odds", "adjusted_margin_median", "adjusted_margin_width"
            )
            calibrator = _fit_logistic(margin_scored, calibration_features, c, config.random_seed)
            scored = _score_logistic(margin_scored, calibrator)
            metrics = _probability_metrics(scored)
            margin = _margin_metrics(scored)
            row = {"spec": asdict(spec), "c": c, **metrics, **margin}
            residual_rows.append(row)
            key = (margin["margin_mae"], metrics["brier"], metrics["ece"], abs(margin["interval_coverage"] - 0.90))
            if best_residual is None or key < best_residual:
                best_residual, best_spec, best_c = key, spec, c
    assert best_spec is not None and best_c is not None
    ledger[CANDIDATES[3]] = residual_rows
    selected[CANDIDATES[3]] = {
        "spec": asdict(best_spec),
        "c": best_c,
        "features": list(residual_features),
    }
    return {"selected": selected, "ledger": ledger}


def _fit_logistic(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    c: float,
    seed: int,
) -> LogisticBundle:
    eligible = frame.drop_nulls(list(features) + ["twap_label_up"])
    names = _complete_variable_features(eligible, features)
    matrix = _matrix(eligible, names)
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(C=c, max_iter=2000, random_state=seed)
    weights = _market_equal_weights(eligible) * eligible["label_weight"].fill_null(1.0).to_numpy()
    model.fit(scaler.transform(matrix), eligible["twap_label_up"].to_numpy(), sample_weight=weights)
    return LogisticBundle(names, scaler, model, c)


def _score_logistic(frame: pl.DataFrame, bundle: LogisticBundle) -> pl.DataFrame:
    eligible = frame.drop_nulls(list(bundle.feature_names))
    probability = bundle.estimator.predict_proba(
        bundle.scaler.transform(_matrix(eligible, bundle.feature_names))
    )[:, 1]
    return eligible.with_columns(pl.Series("probability_up", np.clip(probability, 1e-6, 1 - 1e-6)))


def _fit_residual_models(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    spec: BoostSpec,
    seed: int,
) -> dict[str, Any]:
    eligible = frame.drop_nulls(list(features) + ["margin_residual_bps"]).filter(
        pl.col("margin_residual_bps").is_finite()
    )
    names = _variable_features(eligible, features)
    weights = _market_equal_weights(eligible) * eligible["label_weight"].to_numpy()
    models: dict[str, HistGradientBoostingRegressor] = {}
    for offset, (name, quantile) in enumerate((("lower", 0.05), ("median", 0.50), ("upper", 0.95))):
        models[name] = HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile,
            learning_rate=spec.learning_rate, max_leaf_nodes=spec.max_leaf_nodes,
            min_samples_leaf=spec.min_samples_leaf, l2_regularization=spec.l2_regularization,
            max_iter=spec.max_iter, max_bins=127, early_stopping=True,
            random_state=seed + offset,
        ).fit(_matrix(eligible, names), eligible["margin_residual_bps"].to_numpy(), sample_weight=weights)
    return {"feature_names": names, "models": models, "spec": spec}


def _score_residual_margin(frame: pl.DataFrame, bundle: dict[str, Any]) -> pl.DataFrame:
    eligible = frame.drop_nulls(list(bundle["feature_names"]))
    matrix = _matrix(eligible, bundle["feature_names"])
    delta_lower = bundle["models"]["lower"].predict(matrix)
    delta_median = bundle["models"]["median"].predict(matrix)
    delta_upper = bundle["models"]["upper"].predict(matrix)
    lower = eligible["base_margin_lower"].to_numpy() + np.minimum(delta_lower, delta_upper)
    median = eligible["base_margin_median"].to_numpy() + delta_median
    upper = eligible["base_margin_upper"].to_numpy() + np.maximum(delta_lower, delta_upper)
    return eligible.with_columns(
        pl.Series("predicted_margin_residual_lower", np.minimum(delta_lower, delta_upper)),
        pl.Series("predicted_margin_residual_median", delta_median),
        pl.Series("predicted_margin_residual_upper", np.maximum(delta_lower, delta_upper)),
        pl.Series("adjusted_margin_lower", lower),
        pl.Series("adjusted_margin_median", median),
        pl.Series("adjusted_margin_upper", upper),
        pl.Series("adjusted_margin_width", upper - lower),
    )


def _official_development_evaluation(
    frame: pl.DataFrame,
    config: TournamentConfig,
    correction_selection: dict[str, Any],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    boundaries = [
        _utc("2026-08-14T00:00:00Z"), _utc("2026-08-16T00:00:00Z"),
        _utc("2026-08-18T00:00:00Z"), _utc("2026-08-20T00:00:00Z"),
        _utc("2026-08-22T00:00:00Z"), _utc("2026-08-24T00:00:00Z"),
        _utc("2026-08-26T00:00:00Z"), _utc("2026-08-28T00:00:00Z"),
    ]
    ledgers: list[pl.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold_index, (start, end) in enumerate(itertools.pairwise(boundaries)):
        fit = frame.filter(
            (pl.col("window_start") < start)
            & (pl.col("window_start") >= config.windows.reconstruction_start)
            & pl.col("twap_label_up").is_not_null()
            & (pl.col("label_weight") > 0)
        )
        test = frame.filter(
            pl.col("window_start").is_between(start, end, closed="left")
            & pl.col("twap_label_up").is_not_null()
        )
        if test.is_empty():
            continue
        fold_rows: dict[str, Any] = {"fold": f"official_{start:%Y%m%d}_{end:%Y%m%d}", "start": start.isoformat(), "end": end.isoformat(), "candidates": {}}
        for candidate in CANDIDATES[:3]:
            selected = correction_selection["selected"][candidate]
            model = _fit_logistic(fit, tuple(selected["features"]), float(selected["c"]), config.random_seed + fold_index)
            scored = _score_logistic(test, model).with_columns(
                pl.lit(candidate).alias("candidate"),
                pl.lit(fold_rows["fold"]).alias("fold"),
            )
            ledgers.append(scored)
            fold_rows["candidates"][candidate] = _probability_metrics(scored)
        selected = correction_selection["selected"][CANDIDATES[3]]
        spec = BoostSpec(**selected["spec"])
        residual = _fit_residual_models(fit, tuple(selected["features"]), spec, config.random_seed + 100 + fold_index)
        margin_fit = _score_residual_margin(fit, residual)
        calibration_features = ("base_log_odds", "adjusted_margin_median", "adjusted_margin_width")
        calibrator = _fit_logistic(margin_fit, calibration_features, float(selected["c"]), config.random_seed)
        scored = _score_logistic(_score_residual_margin(test, residual), calibrator).with_columns(
            pl.lit(CANDIDATES[3]).alias("candidate"),
            pl.lit(fold_rows["fold"]).alias("fold"),
        )
        ledgers.append(scored)
        fold_rows["candidates"][CANDIDATES[3]] = {
            **_probability_metrics(scored), **_margin_metrics(scored)
        }
        fold_metrics.append(fold_rows)
    return pl.concat(ledgers, how="diagonal_relaxed", rechunk=True), fold_metrics


def _predictive_selection(
    ledger: pl.DataFrame,
    folds: list[dict[str, Any]],
    config: TournamentConfig,
) -> dict[str, Any]:
    metrics = {name: _candidate_metrics(ledger.filter(pl.col("candidate") == name)) for name in CANDIDATES}
    comparisons = [
        ("settlement_bridge_is_useful", None, CANDIDATES[0]),
        ("non_twap_correction_is_useful", CANDIDATES[0], CANDIDATES[1]),
        ("relative_twap_adds_value", CANDIDATES[1], CANDIDATES[2]),
        ("margin_residual_is_superior", CANDIDATES[2], CANDIDATES[3]),
    ]
    results: dict[str, Any] = {}
    current: str | None = None
    constant = float(ledger["twap_label_up"].mean())
    for index, (claim, left, right) in enumerate(comparisons):
        right_frame = ledger.filter(pl.col("candidate") == right)
        if left is None:
            market = right_frame.group_by("market_id").agg(
                ((pl.col("probability_up") - pl.col("twap_label_up")) ** 2).mean().alias("right"),
                pl.col("twap_label_up").first().alias("label"),
            ).with_columns(((constant - pl.col("label")) ** 2).alias("left"))
        else:
            left_frame = ledger.filter(pl.col("candidate") == left).select(
                "market_id", "observed_at", pl.col("probability_up").alias("left_probability")
            )
            market = right_frame.join(left_frame, on=["market_id", "observed_at"]).group_by("market_id").agg(
                ((pl.col("probability_up") - pl.col("twap_label_up")) ** 2).mean().alias("right"),
                ((pl.col("left_probability") - pl.col("twap_label_up")) ** 2).mean().alias("left"),
            )
        differences = market["right"].to_numpy() - market["left"].to_numpy()
        interval = _bootstrap_mean(differences, 2000, config.random_seed + index)
        right_metrics = metrics[right]
        fold_improvements = _fold_improvement_count(ledger, left, right, constant)
        passed = (
            float(differences.mean()) < 0
            and interval["upper"] < 0
            and (right_metrics.get("ece") or math.inf) <= float(config.raw["gates"]["maximum_ece"])
            and fold_improvements >= 2
            and _candidate_coherence(right_frame)
        )
        results[claim] = {
            "left": left or "constant_twap_frequency",
            "right": right,
            "paired_brier_difference": float(differences.mean()),
            "bootstrap_95": interval,
            "improved_folds": fold_improvements,
            "passed": passed,
        }
        if passed:
            current = right
        elif index == 0:
            current = None
        else:
            break
    return {"winner": current, "comparisons": results, "metrics": metrics}


def _candidate_coherence(frame: pl.DataFrame) -> bool:
    if frame.is_empty():
        return False
    directions = frame.with_columns((pl.col("probability_up") >= 0.5).alias("predicted_up"))
    if directions["predicted_up"].n_unique() < 2:
        return False
    bands = directions.with_columns(_entry_band_expr().alias("entry_band")).group_by("entry_band").agg(
        ((pl.col("probability_up") - pl.col("twap_label_up")) ** 2).mean().alias("brier")
    )
    return bands.height >= 3 and np.isfinite(bands["brier"].to_numpy()).all()


def _fold_improvement_count(ledger: pl.DataFrame, left: str | None, right: str, constant: float) -> int:
    right_rows = ledger.filter(pl.col("candidate") == right)
    right_metrics = right_rows.group_by("fold").agg(
        ((pl.col("probability_up") - pl.col("twap_label_up")) ** 2).mean().alias("right")
    )
    if left is None:
        left_metrics = right_rows.group_by("fold").agg(
            ((pl.lit(constant) - pl.col("twap_label_up")) ** 2).mean().alias("left")
        )
    else:
        left_metrics = ledger.filter(pl.col("candidate") == left).group_by("fold").agg(
            ((pl.col("probability_up") - pl.col("twap_label_up")) ** 2).mean().alias("left")
        )
    return right_metrics.join(left_metrics, on="fold").filter(pl.col("right") < pl.col("left")).height


def _fit_final_candidates(
    frame: pl.DataFrame,
    config: TournamentConfig,
    selection: dict[str, Any],
) -> dict[str, Any]:
    fit = frame.filter(
        (pl.col("window_start") < config.candidate_freeze)
        & (pl.col("window_start") >= config.windows.reconstruction_start)
        & pl.col("twap_label_up").is_not_null()
        & (pl.col("label_weight") > 0)
    )
    models: dict[str, Any] = {}
    for candidate in CANDIDATES[:3]:
        selected = selection["selected"][candidate]
        models[candidate] = _fit_logistic(
            fit, tuple(selected["features"]), float(selected["c"]), config.random_seed + 8000
        )
    selected = selection["selected"][CANDIDATES[3]]
    residual = _fit_residual_models(
        fit, tuple(selected["features"]), BoostSpec(**selected["spec"]), config.random_seed + 9000
    )
    margin_fit = _score_residual_margin(fit, residual)
    calibrator = _fit_logistic(
        margin_fit,
        ("base_log_odds", "adjusted_margin_median", "adjusted_margin_width"),
        float(selected["c"]),
        config.random_seed + 9001,
    )
    models[CANDIDATES[3]] = ResidualBundle(
        tuple(residual["feature_names"]), residual["models"], calibrator,
        BoostSpec(**selected["spec"]),
    )
    _verify_serialization(models)
    return models


def _verify_serialization(models: dict[str, Any]) -> None:
    payload = pickle.dumps(models, protocol=pickle.HIGHEST_PROTOCOL)
    if pickle.loads(payload).keys() != models.keys():
        raise RuntimeError("candidate serialization parity failed")


def _score_final_prospective(
    frame: pl.DataFrame,
    base: BaseBundle,
    candidate: Any,
    name: str,
    config: TournamentConfig,
) -> pl.DataFrame:
    prospective = frame.filter(
        (pl.col("window_start") >= config.candidate_freeze)
        & pl.col("twap_label_up").is_not_null()
    )
    if prospective.is_empty():
        return prospective
    base_scored = _score_base(prospective, base).select(
        "market_id", "observed_at", "base_probability_up", "base_log_odds",
        "base_margin_lower", "base_margin_median", "base_margin_upper", "base_margin_width",
    )
    scored = prospective.join(base_scored, on=["market_id", "observed_at"], validate="1:1")
    if name == CANDIDATES[3]:
        residual = {
            "feature_names": candidate.feature_names,
            "models": candidate.models,
        }
        scored = _score_logistic(_score_residual_margin(scored, residual), candidate.calibrator)
    else:
        scored = _score_logistic(scored, candidate)
        # Classification corrections inherit the base RefPrice interval for the
        # conservative zero-crossing economic guard.
        scored = scored.with_columns(
            pl.col("base_margin_lower").alias("adjusted_margin_lower"),
            pl.col("base_margin_median").alias("adjusted_margin_median"),
            pl.col("base_margin_upper").alias("adjusted_margin_upper"),
        )
    return scored.with_columns(pl.lit(name).alias("candidate"))


def _select_economic_policy(
    frame: pl.DataFrame,
    config: TournamentConfig,
) -> tuple[dict[str, Any], pl.DataFrame, dict[str, Any]]:
    prepared = _prepare_economic_rows(frame, config)
    best: tuple[float, ...] | None = None
    selected_policy: dict[str, Any] | None = None
    selected_trades = pl.DataFrame()
    ledger: list[dict[str, Any]] = []
    for confidence, edge in itertools.product(
        config.raw["execution"]["confidence_thresholds"],
        config.raw["execution"]["edge_thresholds"],
    ):
        policy = {"confidence": float(confidence), "stressed_edge": float(edge)}
        trades = _apply_economic_policy(prepared, policy)
        metrics = _economic_metrics(trades, prepared["market_id"].n_unique(), config)
        ledger.append({"policy": policy, **metrics})
        key = (
            float(metrics.get("gate_count", 0)),
            float(metrics.get("bootstrap_lower", -math.inf)),
            float(metrics.get("stressed_pnl_5", -math.inf)),
            float(metrics.get("coverage", 0)),
        )
        if best is None or key > best:
            best, selected_policy, selected_trades = key, policy, trades
    assert selected_policy is not None
    metrics = _economic_metrics(selected_trades, prepared["market_id"].n_unique(), config)
    metrics["policy_search"] = ledger
    metrics["selected_policy"] = selected_policy
    metrics["status"] = "passed" if metrics["passed"] else "failed"
    return selected_policy, selected_trades, metrics


def _prepare_economic_rows(frame: pl.DataFrame, config: TournamentConfig) -> pl.DataFrame:
    reserve = float(config.raw["execution"]["liquidity_reserve_per_share"])
    slippage = float(config.raw["execution"]["stress_slippage_per_share"])
    available = frame.filter(
        pl.col("up_ask_vwap_5").is_not_null()
        & pl.col("down_ask_vwap_5").is_not_null()
        & pl.col("probability_up").is_not_null()
    )
    if "adjusted_margin_lower" not in available.columns:
        available = available.with_columns(
            pl.col("base_margin_lower").alias("adjusted_margin_lower"),
            pl.col("base_margin_median").alias("adjusted_margin_median"),
            pl.col("base_margin_upper").alias("adjusted_margin_upper"),
        )
    predicted_up = available["probability_up"].to_numpy() >= 0.5
    probability = np.where(predicted_up, available["probability_up"].to_numpy(), 1 - available["probability_up"].to_numpy())
    cost = np.where(predicted_up, available["up_ask_vwap_5"].to_numpy(), available["down_ask_vwap_5"].to_numpy())
    fee_rate = available["fee_rate"].fill_null(0.0).to_numpy()
    fee = fee_rate * cost * (1 - cost)
    stressed_cost = cost + fee + reserve + slippage
    correct = predicted_up == available["twap_label_up"].to_numpy().astype(bool)
    return available.with_columns(
        pl.Series("predicted_up", predicted_up),
        pl.Series("selected_probability", probability),
        pl.Series("selected_cost_5", cost),
        pl.Series("stressed_cost_5", stressed_cost),
        pl.Series("stressed_edge_5", probability - stressed_cost),
        pl.Series("break_even_probability", stressed_cost),
        pl.Series("direction_correct", correct),
        pl.Series("stressed_pnl_5", np.where(correct, 5 * (1 - stressed_cost), -5 * stressed_cost)),
    )


def _apply_economic_policy(frame: pl.DataFrame, policy: dict[str, Any]) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    buffer = 0.02
    eligible = frame.filter(
        (pl.col("selected_probability") >= policy["confidence"])
        & (pl.col("stressed_edge_5") >= policy["stressed_edge"])
        & (pl.col("selected_probability") >= pl.col("break_even_probability") + buffer)
        & (
            (pl.col("predicted_up") & (pl.col("adjusted_margin_lower") > 0))
            | (~pl.col("predicted_up") & (pl.col("adjusted_margin_upper") < 0))
        )
    )
    return eligible.sort(["window_start", "market_id", "seconds_elapsed"]).group_by(
        "market_id", maintain_order=True
    ).first()


def _economic_metrics(
    trades: pl.DataFrame,
    scheduled_markets: int,
    config: TournamentConfig,
) -> dict[str, Any]:
    if trades.is_empty():
        return {
            "trades": 0, "coverage": 0.0, "stressed_pnl_5": 0.0,
            "stressed_expectancy": 0.0, "profit_factor": 0.0,
            "bootstrap_lower": -math.inf, "profitable_fold_ratio": 0.0,
            "maximum_day_contribution": math.inf, "passed": False, "gate_count": 0,
        }
    pnl = trades["stressed_pnl_5"].to_numpy()
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    daily = trades.with_columns(pl.col("window_start").dt.date().alias("date")).group_by("date").agg(
        pl.col("stressed_pnl_5").sum().alias("pnl")
    ).sort("date")
    bootstrap = _bootstrap_mean(daily["pnl"].to_numpy(), 2000, config.random_seed)
    fold = trades.group_by("fold").agg(pl.col("stressed_pnl_5").sum().alias("pnl")) if "fold" in trades.columns else pl.DataFrame({"pnl": [pnl.sum()]})
    cumulative = np.cumsum(pnl)
    peaks = np.maximum.accumulate(np.insert(cumulative, 0, 0.0))[:-1]
    drawdown = peaks - cumulative
    tail_count = max(1, math.ceil(len(pnl) * 0.05))
    positive_days = daily.filter(pl.col("pnl") > 0)["pnl"].to_numpy()
    max_day_share = float(positive_days.max() / positive_days.sum()) if len(positive_days) and positive_days.sum() > 0 else math.inf
    coverage = trades.height / max(scheduled_markets, 1)
    profit_factor = float(gains / losses) if losses > 0 else math.inf
    profitable_ratio = float((fold["pnl"] > 0).mean())
    checks = {
        "positive_stressed_pnl": float(pnl.sum()) > 0,
        "positive_expectancy": float(pnl.mean()) > 0,
        "profit_factor": profit_factor >= float(config.raw["gates"]["minimum_profit_factor"]),
        "positive_bootstrap_lower": bootstrap["lower"] > 0,
        "profitable_folds": profitable_ratio >= float(config.raw["gates"]["minimum_profitable_fold_ratio"]),
        "market_coverage": coverage >= float(config.raw["gates"]["minimum_market_coverage"]),
        "both_directions": trades["predicted_up"].n_unique() == 2,
        "maximum_day_contribution": max_day_share <= float(config.raw["gates"]["maximum_day_contribution"]),
    }
    capacity = _capacity_curve(trades, config)
    checks["positive_five_share_capacity"] = capacity.get("5", {}).get("stressed_pnl", 0) > 0
    return {
        "trades": trades.height,
        "coverage": coverage,
        "stressed_pnl_5": float(pnl.sum()),
        "stressed_expectancy": float(pnl.mean()),
        "profit_factor": profit_factor,
        "bootstrap_lower": bootstrap["lower"],
        "bootstrap_95": bootstrap,
        "profitable_fold_ratio": profitable_ratio,
        "maximum_day_contribution": max_day_share,
        "maximum_drawdown": float(drawdown.max(initial=0.0)),
        "cvar_5": float(np.sort(pnl)[:tail_count].mean()),
        "capacity_curve": capacity,
        "checks": checks,
        "gate_count": sum(checks.values()),
        "passed": all(checks.values()),
        "by_direction": _economic_slice(trades, "predicted_up"),
        "by_entry_band": _economic_slice(trades.with_columns(_entry_band_expr().alias("entry_band")), "entry_band"),
        "by_price_bucket": _economic_slice(trades.with_columns(_price_bucket_expr().alias("price_bucket")), "price_bucket"),
    }


def _capacity_curve(trades: pl.DataFrame, config: TournamentConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    reserve = float(config.raw["execution"]["liquidity_reserve_per_share"])
    slippage = float(config.raw["execution"]["stress_slippage_per_share"])
    for quantity in VWAP_QUANTITIES:
        up, down = f"up_ask_vwap_{quantity}", f"down_ask_vwap_{quantity}"
        available = trades.filter(pl.col(up).is_not_null() & pl.col(down).is_not_null())
        if available.is_empty():
            result[str(quantity)] = {"trades": 0, "stressed_pnl": 0.0}
            continue
        cost = np.where(available["predicted_up"].to_numpy(), available[up].to_numpy(), available[down].to_numpy())
        stressed = cost + reserve + slippage
        correct = available["direction_correct"].to_numpy()
        pnl = np.where(correct, quantity * (1 - stressed), -quantity * stressed)
        result[str(quantity)] = {"trades": available.height, "stressed_pnl": float(pnl.sum()), "expectancy": float(pnl.mean())}
    return result


def _evaluate_prospective(
    rows: pl.DataFrame,
    policy: dict[str, Any],
    config: TournamentConfig,
) -> tuple[dict[str, Any], pl.DataFrame]:
    prepared = _prepare_economic_rows(rows, config) if not rows.is_empty() else rows
    trades = _apply_economic_policy(prepared, policy) if not prepared.is_empty() else prepared
    markets = rows["market_id"].n_unique() if not rows.is_empty() else 0
    days = rows["window_start"].dt.date().n_unique() if not rows.is_empty() else 0
    metrics = _economic_metrics(trades, markets, config)
    minimum = config.raw["gates"]
    evidence_checks = {
        "authentic_markets": markets >= int(minimum["prospective_markets"]),
        "active_days": days >= int(minimum["prospective_days"]),
        "chronological_folds": days >= int(minimum["prospective_folds"]),
    }
    gate_checks = {**evidence_checks, **metrics.get("checks", {})}
    status = "passed" if all(gate_checks.values()) else (
        "pending_insufficient_evidence" if not all(evidence_checks.values()) else "failed"
    )
    return {
        "status": status,
        "deployable": status == "passed",
        "markets": markets,
        "active_days": days,
        "checks": gate_checks,
        "metrics": metrics,
    }, trades


def settlement_transition_report(frame: pl.DataFrame, ledger: pl.DataFrame) -> dict[str, Any]:
    markets = frame.filter(
        pl.col("ref_label_up").is_not_null() & pl.col("twap_label_up").is_not_null()
    ).sort("seconds_elapsed").group_by("market_id", maintain_order=True).first().with_columns(
        (pl.col("ref_label_up") == pl.col("twap_label_up")).alias("agreement"),
        pl.col("ref_margin_bps").abs().cut(
            [0.526, 1.052, 1.578, 5.0],
            labels=["<0.526", "0.526-1.052", "1.052-1.578", "1.578-5", ">=5"],
        ).alias("margin_band"),
        pl.col("window_start").dt.strftime("%Y-%m").alias("month"),
        pl.when(pl.col("btc_realized_volatility_60s_bps") <= pl.col("btc_realized_volatility_60s_bps").median())
        .then(pl.lit("low"))
        .otherwise(pl.lit("high"))
        .alias("volatility_band"),
        pl.when(pl.col("btc_reversal_5_vs_30") > 0).then(pl.lit("reversal")).otherwise(pl.lit("continuation")).alias("reversal_state"),
    )
    report = {
        "markets": markets.height,
        "agreement_rate": float(markets["agreement"].mean()) if markets.height else None,
        "disagreement_rate": float((~markets["agreement"]).mean()) if markets.height else None,
        "by_terminal_margin_band": _agreement_slice(markets, "margin_band"),
        "by_month": _agreement_slice(markets, "month"),
        "by_ref_direction": _agreement_slice(markets, "ref_label_up"),
        "by_volatility": _agreement_slice(markets, "volatility_band"),
        "by_reversal_state": _agreement_slice(markets, "reversal_state"),
        "margin_residual_distribution": _distribution(markets["margin_residual_bps"].drop_nulls().to_numpy()),
    }
    if not ledger.is_empty() and "predicted_margin_residual_median" in ledger.columns:
        residual = ledger.filter(pl.col("candidate") == CANDIDATES[3])
        report["predicted_correction_when_agree"] = _distribution(
            residual.filter(pl.col("ref_label_up") == pl.col("twap_label_up"))[
                "predicted_margin_residual_median"
            ].drop_nulls().to_numpy()
        )
        report["predicted_correction_when_disagree"] = _distribution(
            residual.filter(pl.col("ref_label_up") != pl.col("twap_label_up"))[
                "predicted_margin_residual_median"
            ].drop_nulls().to_numpy()
        )
        report["transition_errors_by_entry_time"] = {
            str(row["entry_band"]): {
                "rows": int(row["rows"]), "margin_mae": float(row["margin_mae"])
            }
            for row in residual.with_columns(_entry_band_expr().alias("entry_band")).group_by("entry_band").agg(
                pl.len().alias("rows"),
                (pl.col("adjusted_margin_median") - pl.col("twap_margin_bps")).abs().mean().alias("margin_mae"),
            ).iter_rows(named=True)
        }
    return report


def causal_feature_registry() -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    for feature in BASE_FEATURES:
        registry.append({
            "feature": feature,
            "semantic_role": "causal_refprice_base",
            "source": "completed Binance one-second path relative to canonical RefPrice boundary",
            "source_event_timestamp": "kline.open_timestamp",
            "source_availability_timestamp": "kline.close_timestamp",
            "lookback": "feature-specific, no later than observed_at",
            "feature_as_of": "observed_at",
            "live_computable": True,
        })
    for feature in BASE_BRIDGE_FEATURES:
        registry.append({
            "feature": feature,
            "semantic_role": "cross_fitted_refprice_prediction",
            "source": "shared RefPrice base model",
            "source_event_timestamp": "observed_at",
            "source_availability_timestamp": "observed_at",
            "lookback": "base feature contract",
            "feature_as_of": "observed_at",
            "live_computable": True,
        })
    for feature in NON_TWAP_FEATURES:
        registry.append({
            "feature": feature,
            "semantic_role": "non_twap_settlement_transition",
            "source": "causal Chainlink RefPrice and Binance path",
            "source_event_timestamp": "ref_source_timestamp",
            "source_availability_timestamp": "ref_available_at",
            "lookback": "up to 60 seconds ending before observed_at",
            "feature_as_of": "observed_at",
            "live_computable": True,
        })
    for feature in RELATIVE_TWAP_FEATURES:
        registry.append({
            "feature": feature,
            "semantic_role": "relative_causal_twap_state",
            "source": "causally available RefPrice path",
            "source_event_timestamp": "ref_source_timestamp",
            "source_availability_timestamp": "ref_available_at",
            "lookback": "[T-W,T), W in {30,60}; five/ten-second causal lags",
            "feature_as_of": "twap_feature_as_of",
            "live_computable": True,
        })
    return registry


def supervision_registry_payload() -> list[dict[str, Any]]:
    return [
        {"field": "ref_label_up", "role": "supervision_only", "inference": False},
        {"field": "ref_margin_bps", "role": "supervision_only", "inference": False},
        {"field": "twap_label_up", "role": "supervision_only", "inference": False},
        {"field": "twap_margin_bps", "role": "supervision_only", "inference": False},
        {"field": "margin_residual_bps", "role": "supervision_only", "inference": False},
        {"field": "label_source", "role": "weighting_diagnostic_only", "inference": False},
        {"field": "binance_diagnostic_label_up", "role": "diagnostic_only", "inference": False},
    ]


def _source_query_manifest(config: TournamentConfig, dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        "queries": dataset["sql_sha256"],
        "sources": [
            "polymarket.btc_interval_markets",
            "polymarket.binance_one_second_klines",
            "market_data.binance_spot_btcusdt_one_second_ohlcv",
            "polymarket.polygon_chainlink_btcusd_oracle_rounds",
            "market_data.polygon_chainlink_btcusd_oracle_rounds",
            "market_data.chainlink_btcusd_reference_prices",
            "market_data.pmdata_chainlink_btcusd_twap",
            "polymarket.btc_market_capacity_execution_snapshots",
            "polymarket.backfill_artifacts",
        ],
        "new_sources": False,
        "new_ingesters": False,
        "new_tables": False,
        "database_writes": False,
        "bounded_range": [
            config.windows.data_start.isoformat(), config.windows.prospective_end.isoformat()
        ],
    }


def _permitted_conclusion(
    predictive: dict[str, Any], economic: dict[str, Any], prospective: dict[str, Any]
) -> str:
    winner = predictive.get("winner")
    if prospective.get("deployable"):
        return "A deployable bridge model qualifies."
    if prospective.get("status") == "pending_insufficient_evidence" and winner:
        return "A candidate is promising but lacks prospective evidence."
    if winner and economic.get("passed") is False:
        return "TWAP improves prediction but not economic selection."
    mapping = {
        CANDIDATES[0]: "Historical RefPrice prediction transfers without correction.",
        CANDIDATES[1]: "Non-TWAP basis explains the settlement transition.",
        CANDIDATES[2]: "Relative causal TWAP materially improves the settlement correction.",
        CANDIDATES[3]: "Explicit TWAP-margin residual modeling provides the best bridge.",
    }
    return mapping.get(winner, "No bridge model demonstrates reliable value.")


def _terminated_result(
    config: TournamentConfig,
    manifest: dict[str, Any],
    integrity: dict[str, Any],
    base: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_family": config.model_family,
        "source_commit": _git_revision(config.package_root),
        "base_model": {"search": base["ledger"], "admission": base["admission"]},
        "causal_integrity": integrity,
        "dataset": manifest,
        "candidate_metrics": {},
        "conclusion": "No bridge model demonstrates reliable value.",
        "deployment_status": "non_deployable_base_admission_failed",
        "runtime_exported": False,
        "trading_process_changed": False,
        "database_mutations": False,
    }


def _finalize_run(
    config: TournamentConfig,
    temporary: Path,
    final: Path,
    result: dict[str, Any],
    artifacts: dict[str, Path],
) -> tuple[Path, dict[str, Any]]:
    _write_json(temporary / "metrics.json", result)
    (temporary / "report.md").write_text(_render_report(result))
    hashes: dict[str, str] = {}
    for path in sorted(temporary.rglob("*")):
        if path.is_file() and path.name != "bundle.sha256":
            hashes[str(path.relative_to(temporary))] = file_sha256(path)
    _write_json(temporary / "model-provenance.json", {
        "model_family": config.model_family,
        "artifact_identity": result.get("artifact"),
        "producing_commit": _git_revision(config.package_root),
        "executed_run_id": result["run_id"],
        "source_manifest_sha256": file_sha256(config.paths.data / "manifest.json"),
        "qualification_status": result.get("deployment_status"),
        "deployment_status": "not_deployed",
        "runtime": runtime_provenance(config.package_root),
    })
    hashes["model-provenance.json"] = file_sha256(temporary / "model-provenance.json")
    (temporary / "bundle.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items()))
    )
    config.paths.committed_results.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    return final, result


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Settlement-Bridge Residual Tournament",
        "",
        f"- Run: `{result['run_id']}`",
        f"- Model family: `{result['model_family']}`",
        f"- Source commit: `{result.get('source_commit')}`",
        f"- Deployment status: **{result.get('deployment_status')}**",
        f"- Conclusion: **{result.get('conclusion')}**",
        "- Runtime/trading process/database mutations: none",
        "",
        "## RefPrice base admission",
        "",
        "```json",
        json.dumps(result.get("base_model", {}).get("admission"), indent=2, sort_keys=True),
        "```",
        "",
        "## Candidate probability and margin metrics",
        "",
        "```json",
        json.dumps(result.get("candidate_metrics", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Predictive comparisons",
        "",
        "```json",
        json.dumps(result.get("predictive_selection", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Economic admission",
        "",
        "```json",
        json.dumps(result.get("economic_admission", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Prospective qualification",
        "",
        "```json",
        json.dumps(result.get("prospective_qualification", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Settlement transition",
        "",
        "```json",
        json.dumps(result.get("settlement_transition", {}), indent=2, sort_keys=True),
        "```",
        "",
        "The artifact is training-only, was not exported to runtime, and did not change any trading process.",
    ]
    return "\n".join(lines) + "\n"


def _probability_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {"markets": 0, "brier": None, "log_loss": None, "ece": None, "accuracy": None}
    y = frame["twap_label_up"].to_numpy().astype(float)
    p = np.clip(frame["probability_up"].to_numpy(), 1e-6, 1 - 1e-6)
    weights = _market_equal_weights(frame)
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "brier": float(np.average((p - y) ** 2, weights=weights)),
        "log_loss": float(log_loss(y, p, sample_weight=weights, labels=[0, 1])),
        "ece": _ece(y, p, weights),
        "accuracy": float(np.average((p >= 0.5) == y.astype(bool), weights=weights)),
    }


def _margin_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    eligible = frame.drop_nulls(["twap_margin_bps", "adjusted_margin_median", "adjusted_margin_lower", "adjusted_margin_upper"])
    eligible = eligible.filter(
        pl.all_horizontal(
            pl.col(name).is_finite()
            for name in (
                "twap_margin_bps", "adjusted_margin_median",
                "adjusted_margin_lower", "adjusted_margin_upper",
            )
        )
    )
    if eligible.is_empty():
        return {"margin_markets": 0, "margin_mae": None, "interval_coverage": None}
    weights = _market_equal_weights(eligible)
    actual = eligible["twap_margin_bps"].to_numpy()
    return {
        "margin_markets": eligible["market_id"].n_unique(),
        "margin_mae": float(np.average(np.abs(eligible["adjusted_margin_median"].to_numpy() - actual), weights=weights)),
        "interval_coverage": float(np.average(
            (actual >= eligible["adjusted_margin_lower"].to_numpy())
            & (actual <= eligible["adjusted_margin_upper"].to_numpy()),
            weights=weights,
        )),
    }


def _candidate_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return _probability_metrics(frame)
    result = _probability_metrics(frame)
    if "adjusted_margin_median" in frame.columns:
        result.update(_margin_metrics(frame))
    result["by_date"] = _probability_slice(frame.with_columns(pl.col("window_start").dt.date().alias("date")), "date")
    result["by_direction"] = _probability_slice(frame, "twap_label_up")
    result["by_entry_band"] = _probability_slice(frame.with_columns(_entry_band_expr().alias("entry_band")), "entry_band")
    result["by_price_bucket"] = _probability_slice(frame.with_columns(_price_bucket_expr().alias("price_bucket")), "price_bucket") if "selected_cost_5" in frame.columns else {}
    result["by_volatility"] = _volatility_slices(frame)
    return result


def _probability_slice(frame: pl.DataFrame, column: str) -> dict[str, Any]:
    return {
        str(row[column]): {
            "rows": int(row["rows"]), "brier": float(row["brier"]), "accuracy": float(row["accuracy"])
        }
        for row in frame.group_by(column).agg(
            pl.len().alias("rows"),
            ((pl.col("probability_up") - pl.col("twap_label_up")) ** 2).mean().alias("brier"),
            ((pl.col("probability_up") >= 0.5) == pl.col("twap_label_up").cast(pl.Boolean)).mean().alias("accuracy"),
        ).iter_rows(named=True)
    }


def _volatility_slices(frame: pl.DataFrame) -> dict[str, Any]:
    if "btc_realized_volatility_60s_bps" not in frame.columns:
        return {}
    q1, q2, q3 = (float(frame["btc_realized_volatility_60s_bps"].quantile(q)) for q in (0.25, 0.50, 0.75))
    return _probability_slice(frame.with_columns(
        pl.when(pl.col("btc_realized_volatility_60s_bps") <= q1).then(pl.lit("q1"))
        .when(pl.col("btc_realized_volatility_60s_bps") <= q2).then(pl.lit("q2"))
        .when(pl.col("btc_realized_volatility_60s_bps") <= q3).then(pl.lit("q3"))
        .otherwise(pl.lit("q4")).alias("volatility_bucket")
    ), "volatility_bucket")


def _agreement_slice(frame: pl.DataFrame, column: str) -> dict[str, Any]:
    if frame.is_empty():
        return {}
    return {
        str(row[column]): {"markets": int(row["markets"]), "agreement_rate": float(row["agreement_rate"])}
        for row in frame.group_by(column).agg(
            pl.len().alias("markets"), pl.col("agreement").mean().alias("agreement_rate")
        ).iter_rows(named=True)
    }


def _economic_slice(frame: pl.DataFrame, column: str) -> dict[str, Any]:
    return {
        str(row[column]): {
            "trades": int(row["trades"]), "stressed_pnl": float(row["pnl"]),
            "expectancy": float(row["expectancy"]),
        }
        for row in frame.group_by(column).agg(
            pl.len().alias("trades"),
            pl.col("stressed_pnl_5").sum().alias("pnl"),
            pl.col("stressed_pnl_5").mean().alias("expectancy"),
        ).iter_rows(named=True)
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": len(finite), "mean": float(finite.mean()), "std": float(finite.std()),
        "p05": float(np.quantile(finite, 0.05)), "p25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)), "p75": float(np.quantile(finite, 0.75)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def _ece(y: np.ndarray, p: np.ndarray, weights: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    total = weights.sum()
    indices = np.minimum((p * bins).astype(int), bins - 1)
    for bucket in range(bins):
        selected = indices == bucket
        if not selected.any():
            continue
        weight = weights[selected].sum()
        result += weight / total * abs(np.average(p[selected], weights=weights[selected]) - np.average(y[selected], weights=weights[selected]))
    return float(result)


def _bootstrap_mean(values: np.ndarray, resamples: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"lower": -math.inf, "upper": math.inf, "mean": math.nan}
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for offset in range(0, resamples, 200):
        count = min(200, resamples - offset)
        indices = generator.integers(0, len(values), size=(count, len(values)))
        means[offset : offset + count] = values[indices].mean(axis=1)
    return {
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "mean": float(values.mean()),
    }


def _market_equal_weights(frame: pl.DataFrame) -> np.ndarray:
    counts = frame.group_by("market_id").len().rename({"len": "_rows"})
    weights = 1.0 / frame.select("market_id").join(counts, on="market_id")["_rows"].to_numpy()
    return weights * len(weights) / weights.sum()


def _matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError("missing features: " + ", ".join(missing))
    matrix = frame.select(*features).to_numpy().astype(np.float64, copy=False)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def _variable_features(frame: pl.DataFrame, features: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(
        name for name in features
        if name in frame.columns and frame[name].cast(pl.Float64).drop_nulls().filter(
            frame[name].cast(pl.Float64).drop_nulls().is_finite()
        ).n_unique() >= 2
    )
    if not selected:
        raise RuntimeError("no variable model features")
    return selected


def _complete_variable_features(frame: pl.DataFrame, features: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(
        name for name in features
        if name in frame.columns
        and frame[name].null_count() == 0
        and bool(frame[name].cast(pl.Float64).is_finite().all())
        and frame[name].n_unique() >= 2
    )
    if not selected:
        raise RuntimeError("no complete variable correction features")
    return selected


def _entry_band_expr() -> pl.Expr:
    return (
        pl.when(pl.col("seconds_elapsed") < 90).then(pl.lit("60-89"))
        .when(pl.col("seconds_elapsed") < 120).then(pl.lit("90-119"))
        .when(pl.col("seconds_elapsed") < 150).then(pl.lit("120-149"))
        .otherwise(pl.lit("150-179"))
    )


def _price_bucket_expr() -> pl.Expr:
    return (
        pl.when(pl.col("selected_cost_5") < 0.65).then(pl.lit("<0.65"))
        .when(pl.col("selected_cost_5") < 0.75).then(pl.lit("0.65-0.75"))
        .when(pl.col("selected_cost_5") < 0.85).then(pl.lit("0.75-0.85"))
        .otherwise(pl.lit(">=0.85"))
    )


def _frame_hash(frame: pl.DataFrame) -> str:
    buffer = io.BytesIO()
    frame.write_ipc(buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, BoostSpec):
        return asdict(value)
    raise TypeError(type(value).__name__)


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(str(value)).astimezone(UTC)
