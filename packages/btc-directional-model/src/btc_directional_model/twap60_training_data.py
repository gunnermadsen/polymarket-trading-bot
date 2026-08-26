"""Bounded, immutable training data for the TWAP60 challenger tournament.

Label construction and causal feature construction intentionally live in separate
functions.  The module reads only existing source facts and writes local Parquet
caches; it never mutates the database.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import pyarrow as pa

from .chainlink_oi_features import (
    _attach_candle_features,
)
from .continuous_edge_training import (
    BOOK_RAW_FEATURES,
    CAPACITY_SCHEMA,
    attach_book_features,
)
from .core_extract import (
    CORE_ORACLE_ROUND_SCHEMA,
    POLYGON_CHAINLINK_BTCUSD_PROXY,
    configure_read_only_connection,
    database_connection,
    file_sha256,
    write_json_atomic,
)
from .core_features import (
    attach_causal_oracle_rounds,
    derive_core_point_in_time_features,
    derive_oracle_point_in_time_features,
    prepare_causal_oracle_rounds,
)

REFPRICE_RUNTIME_FEATURES = (
    "chainlink_ref_return_1s_bps",
    "chainlink_ref_return_5s_bps",
    "chainlink_ref_return_15s_bps",
    "chainlink_ref_return_30s_bps",
    "chainlink_ref_return_60s_bps",
    "chainlink_ref_binance_basis_bps",
    "chainlink_ref_boundary_gap_bps",
    "chainlink_ref_spread_bps",
    "chainlink_ref_spread_change_30s_bps",
    "chainlink_ref_binance_direction_agreement_30s",
)
REFPRICE_DIAGNOSTIC_FEATURES = (
    "chainlink_ref_return_90s_bps",
    "chainlink_ref_return_120s_bps",
    "chainlink_ref_momentum_alignment_5_30",
    "chainlink_ref_reversal_5_vs_30",
    "chainlink_ref_reversal_15_vs_60",
    "chainlink_ref_boundary_velocity_5s_bps",
    "chainlink_ref_realized_volatility_5s_bps",
    "chainlink_ref_realized_volatility_15s_bps",
    "chainlink_ref_realized_volatility_30s_bps",
    "chainlink_ref_realized_volatility_60s_bps",
    "chainlink_ref_realized_volatility_90s_bps",
    "chainlink_ref_realized_volatility_120s_bps",
    "chainlink_ref_range_15s_bps",
    "chainlink_ref_range_30s_bps",
    "chainlink_ref_range_60s_bps",
    "chainlink_ref_range_90s_bps",
    "chainlink_ref_range_120s_bps",
    "chainlink_ref_path_efficiency_30s",
    "chainlink_ref_path_efficiency_60s",
    "chainlink_ref_range_position_60s",
    "chainlink_ref_boundary_cross_count_60s",
    "chainlink_ref_direction_changes_60s",
    "chainlink_ref_age_seconds",
    "chainlink_ref_reports_60s",
    "chainlink_ref_max_gap_60s",
    "chainlink_ref_binance_basis_velocity_5s_bps",
    "chainlink_ref_binance_disagreement_30s",
    "chainlink_ref_source_skew_seconds",
)
REFPRICE_ALL_FEATURES = (*REFPRICE_RUNTIME_FEATURES, *REFPRICE_DIAGNOSTIC_FEATURES)


@dataclass(frozen=True)
class DataPaths:
    package_root: Path
    cache: Path
    core_features: Path
    core_current_sql: Path
    oracle_sql: Path
    label_sql: Path
    refprice_sql: Path
    candle_sql: Path
    execution_sql: Path


@dataclass(frozen=True)
class ProxyConvention:
    time_column: Literal["valid_from_timestamp", "source_timestamp"]
    calibration_mae_bps: float
    calibration_p99_bps: float
    error_band_bps: float


def extract_tournament_sources(
    paths: DataPaths,
    *,
    range_start: datetime,
    range_end: datetime,
    current_start: datetime,
    force: bool = False,
) -> dict[str, Any]:
    """Extract bounded daily source caches using read-only transactions."""

    if range_start.tzinfo is None or range_end.tzinfo is None or range_start >= range_end:
        raise ValueError("source interval must be a positive timezone-aware range")
    paths.cache.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": "btc-twap60-training-source-cache-v1",
        "range_start": range_start.isoformat(),
        "range_end": range_end.isoformat(),
        "current_start": current_start.isoformat(),
        "queries": {
            "labels": file_sha256(paths.label_sql),
            "refprice": file_sha256(paths.refprice_sql),
            "core_current": file_sha256(paths.core_current_sql),
            "oracle": file_sha256(paths.oracle_sql),
            "candles": file_sha256(paths.candle_sql),
            "execution": file_sha256(paths.execution_sql),
        },
        "read_only": True,
        "database_mutations": False,
        "completed_artifacts_only": True,
    }
    manifest_path = paths.cache / "source-manifest.json"
    checkpoint_path = paths.cache / "source-manifest.partial.json"
    if manifest_path.exists() and not force and not checkpoint_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("contract") != contract:
            raise RuntimeError("existing TWAP60 source cache contract changed")
        for group in existing.get("partitions", {}).values():
            for row in group:
                target = paths.cache / row["path"]
                if not target.is_file() or file_sha256(target) != row["sha256"]:
                    raise RuntimeError(f"cached source partition changed: {target.name}")
        return existing
    if force:
        checkpoint_path.unlink(missing_ok=True)

    sql = {
        "labels": paths.label_sql.read_text(),
        "refprice": paths.refprice_sql.read_text(),
        "core_current": paths.core_current_sql.read_text(),
        "oracle": paths.oracle_sql.read_text(),
        "candles": paths.candle_sql.read_text(),
        "execution": paths.execution_sql.read_text(),
    }
    partitions: dict[str, list[dict[str, Any]]] = {name: [] for name in sql}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("contract") != contract:
            raise RuntimeError("partial TWAP60 source cache contract changed")
        partitions = checkpoint["partitions"]
        for group in partitions.values():
            for row in group:
                target = paths.cache / row["path"]
                if not target.is_file() or file_sha256(target) != row["sha256"]:
                    raise RuntimeError(f"partial source partition changed: {target.name}")
    completed_dates = {
        Path(row["path"]).stem for row in partitions["labels"]
    }
    day = range_start
    while day < range_end:
        if day.date().isoformat() in completed_dates:
            day = min(day + timedelta(days=1), range_end)
            continue
        end = min(day + timedelta(days=1), range_end)
        parameters = {"batch_start": day, "batch_end": end}
        frames = {
            "labels": _isolated_query_frame(
                sql["labels"], parameters,
                cursor_name=f"btc_twap60_labels_{day:%Y%m%d}",
            ),
            "refprice": _isolated_query_frame(
                sql["refprice"], parameters,
                cursor_name=f"btc_twap60_refprice_{day:%Y%m%d}",
            ),
            "core_current": _isolated_query_frame(
                sql["core_current"],
                {
                    "batch_start": day,
                    "batch_end": end,
                    "current_start": current_start,
                },
                cursor_name=f"btc_twap60_core_{day:%Y%m%d}",
            ),
            "oracle": _isolated_query_frame(
                sql["oracle"],
                {
                    "batch_start": day,
                    "batch_end": end,
                    "oracle_feed_proxy_address": POLYGON_CHAINLINK_BTCUSD_PROXY,
                    "oracle_max_publication_delay_seconds": 300,
                },
                cursor_name=f"btc_twap60_oracle_{day:%Y%m%d}",
            ),
            "candles": _query_completed_candles(
                sql["candles"],
                {
                    "history_start": day - timedelta(minutes=121),
                    "range_end": end,
                    "candle_symbol": "BTCUSD",
                },
                cursor_name=f"btc_twap60_candles_{day:%Y%m%d}",
            ),
            "execution": _isolated_query_frame(
                sql["execution"], parameters,
                cursor_name=f"btc_twap60_execution_{day:%Y%m%d}",
                capacity=True,
            ),
        }
        for name, frame in frames.items():
            directory = paths.cache / name
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"{day.date().isoformat()}.parquet"
            frame.write_parquet(destination, compression="zstd", statistics=True)
            partitions[name].append(
                {
                    "path": str(destination.relative_to(paths.cache)),
                    "rows": frame.height,
                    "sha256": file_sha256(destination),
                }
            )
            print(f"twap60 extract: {name} {day.date()} {frame.height:,} rows", flush=True)
        write_json_atomic(
            checkpoint_path,
            {"contract": contract, "partitions": partitions},
        )
        day = end
    manifest = {"contract": contract, "partitions": partitions}
    write_json_atomic(manifest_path, manifest)
    checkpoint_path.unlink(missing_ok=True)
    return manifest


def load_source_group(paths: DataPaths, name: str) -> pl.DataFrame:
    manifest = json.loads((paths.cache / "source-manifest.json").read_text())
    rows = manifest["partitions"][name]
    frames = [pl.read_parquet(paths.cache / row["path"]) for row in rows]
    return pl.concat(frames, how="diagonal_relaxed", rechunk=True) if frames else pl.DataFrame()


def _query_capacity_frame(
    connection: Any,
    query: str,
    parameters: dict[str, Any],
    *,
    cursor_name: str,
) -> pl.DataFrame:
    """Decode nullable capacity columns with the repository's fixed Arrow schema."""

    chunks: list[pl.DataFrame] = []
    with connection.transaction():
        connection.execute("SET TRANSACTION READ ONLY")
        with connection.cursor(name=cursor_name) as cursor:
            cursor.execute(query, parameters)
            while rows := cursor.fetchmany(25_000):
                records = [
                    dict(zip(CAPACITY_SCHEMA.names, row, strict=True)) for row in rows
                ]
                chunks.append(pl.from_arrow(pa.Table.from_pylist(records, schema=CAPACITY_SCHEMA)))
    if not chunks:
        return pl.from_arrow(pa.Table.from_pylist([], schema=CAPACITY_SCHEMA))
    return pl.concat(chunks, how="vertical", rechunk=True)


def _isolated_query_frame(
    query: str,
    parameters: dict[str, Any],
    *,
    cursor_name: str,
    capacity: bool = False,
) -> pl.DataFrame:
    connection = database_connection()
    configure_read_only_connection(connection)
    try:
        if capacity:
            return _query_capacity_frame(
                connection, query, parameters, cursor_name=cursor_name
            )
        return _query_stable_frame(
            connection, query, parameters, cursor_name=cursor_name
        )
    finally:
        connection.close()


def _query_stable_frame(
    connection: Any,
    query: str,
    parameters: dict[str, Any],
    *,
    cursor_name: str,
) -> pl.DataFrame:
    chunks: list[pl.DataFrame] = []
    columns: list[str] = []
    with connection.transaction():
        connection.execute("SET TRANSACTION READ ONLY")
        with connection.cursor(name=cursor_name) as cursor:
            cursor.execute(query, parameters)
            columns = [column.name for column in cursor.description or ()]
            while rows := cursor.fetchmany(25_000):
                chunks.append(
                    pl.DataFrame(
                        rows,
                        schema=columns,
                        orient="row",
                        infer_schema_length=None,
                    )
                )
    if not chunks:
        return pl.DataFrame({column: [] for column in columns})
    return pl.concat(chunks, how="vertical_relaxed", rechunk=True)


def _query_completed_candles(
    query: str,
    parameters: dict[str, Any],
    *,
    cursor_name: str,
) -> pl.DataFrame:
    candles = _isolated_query_frame(
        query, parameters, cursor_name=cursor_name
    )
    if candles.is_empty():
        return candles
    artifact_ids = candles["artifact_id"].unique().to_list()
    connection = database_connection()
    configure_read_only_connection(connection)
    try:
        rows = connection.execute(
            """
            SELECT artifact_id::text
            FROM polymarket.backfill_artifacts
            WHERE artifact_id = ANY(%s::uuid[])
              AND status = 'completed'
            """,
            (artifact_ids,),
        ).fetchall()
    finally:
        connection.close()
    completed = [row[0] for row in rows]
    filtered = candles.filter(pl.col("artifact_id").is_in(completed))
    if filtered.height != candles.height:
        raise RuntimeError("candle source includes a non-completed artifact")
    return filtered


def build_current_core_features(
    raw: pl.DataFrame,
    labels: pl.DataFrame,
    oracle: pl.DataFrame,
) -> pl.DataFrame:
    """Apply the established core/oracle formulas to current TWAP60 markets."""

    if raw.is_empty():
        raise RuntimeError("current-regime Binance core source is empty")
    boundaries = labels.filter(
        pl.col("twap_open_price").is_not_null()
        & pl.col("twap_open_source_timestamp").is_not_null()
        & (pl.col("twap_open_effective_timestamp_rows") == 1)
    ).select(
        "market_id",
        pl.col("twap_open_price").alias("twap_opening_boundary"),
    )
    frame = (
        raw.drop("opening_boundary")
        .join(boundaries, on="market_id", how="inner", validate="m:1")
        .rename({"twap_opening_boundary": "opening_boundary"})
        .sort(["market_id", "seconds_elapsed"])
    )
    complete = (
        frame.group_by("market_id")
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
    frame = frame.join(complete, on="market_id", how="inner")
    rounds = prepare_causal_oracle_rounds(oracle.select(CORE_ORACLE_ROUND_SCHEMA.names))
    frame = attach_causal_oracle_rounds(frame, rounds)
    frame = derive_core_point_in_time_features(frame)
    frame = derive_oracle_point_in_time_features(frame)
    return frame.filter(
        pl.col("seconds_elapsed").is_between(60, 179, closed="both")
        & ((pl.col("seconds_elapsed") - 60) % 5 == 0)
        & pl.col("oracle_model_eligible")
    ).drop(
        "official_outcome", "final_price", "btc_path_positive", "btc_path_crossed",
        "btc_last_path_cross_second", "btc_boundary_positive", "btc_boundary_crossed",
        "btc_last_boundary_cross_second", "oracle_price", "oracle_source_timestamp",
        "oracle_block_timestamp", "oracle_phase_id", "oracle_round_id",
        "oracle_block_number", "oracle_log_index", "oracle_window_open_price",
        "oracle_round_changed", strict=False,
    )


def canonical_refprice_path(frame: pl.DataFrame) -> pl.DataFrame:
    """Deterministically deduplicate archive reports by effective timestamp."""

    required = {
        "source_timestamp", "valid_from_timestamp", "provider_available_at",
        "received_at", "price", "archive_row_number", "report_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("refprice source is missing columns: " + ", ".join(missing))
    clean = frame.filter(
        pl.col("source_timestamp").is_not_null()
        & pl.col("valid_from_timestamp").is_not_null()
        & pl.col("provider_available_at").is_not_null()
        & pl.col("received_at").is_not_null()
        & pl.col("price").is_not_null()
        & pl.col("price").is_finite()
        & (pl.col("price") > 0)
        & (pl.col("valid_from_timestamp") <= pl.col("source_timestamp"))
        & (pl.col("source_timestamp") <= pl.col("provider_available_at"))
    )
    clean = clean.unique(
        subset=["report_sha256", "source_timestamp", "archive_row_number"], keep="last"
    )
    canonical = (
        clean.sort(
            ["valid_from_timestamp", "source_timestamp", "provider_available_at",
             "archive_row_number", "report_sha256"]
        )
        .unique(subset=["valid_from_timestamp"], keep="last")
        .sort("valid_from_timestamp")
    )
    if canonical.is_empty():
        raise RuntimeError("canonical refprice path is empty")
    if canonical["valid_from_timestamp"].n_unique() != canonical.height:
        raise RuntimeError("canonical refprice path still has duplicate effective timestamps")
    return canonical


def piecewise_average(
    timestamps: np.ndarray,
    prices: np.ndarray,
    boundaries: np.ndarray,
    *,
    window_seconds: float = 60.0,
) -> np.ndarray:
    """Integrate a left-continuous piecewise-constant path over trailing windows."""

    times = timestamps.astype("datetime64[us]").astype(np.int64) / 1_000_000.0
    query = boundaries.astype("datetime64[us]").astype(np.int64) / 1_000_000.0
    values = np.asarray(prices, dtype=np.float64)
    if len(times) != len(values) or len(times) == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("piecewise path must have unique increasing timestamps")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("piecewise path prices must be finite and positive")
    cumulative = np.zeros(len(times), dtype=np.float64)
    if len(times) > 1:
        cumulative[1:] = np.cumsum(values[:-1] * np.diff(times))

    def integral_at(points: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(times, points, side="right") - 1
        output = np.full(len(points), np.nan, dtype=np.float64)
        valid = indices >= 0
        output[valid] = (
            cumulative[indices[valid]]
            + values[indices[valid]] * (points[valid] - times[indices[valid]])
        )
        return output

    end_integral = integral_at(query)
    start_integral = integral_at(query - window_seconds)
    return (end_integral - start_integral) / window_seconds


def construct_proxy_labels(
    labels: pl.DataFrame,
    refprice: pl.DataFrame,
    *,
    time_column: Literal["valid_from_timestamp", "source_timestamp"],
) -> pl.DataFrame:
    path = canonical_refprice_path(refprice)
    ordered = path.sort(time_column).unique(subset=[time_column], keep="last").sort(time_column)
    opens = piecewise_average(
        ordered[time_column].to_numpy(), ordered["price"].to_numpy(),
        labels["window_start"].to_numpy(),
    )
    closes = piecewise_average(
        ordered[time_column].to_numpy(), ordered["price"].to_numpy(),
        labels["window_end"].to_numpy(),
    )
    margin_bps = np.log(closes / opens) * 10_000.0
    return labels.with_columns(
        pl.Series("proxy_open_price", opens),
        pl.Series("proxy_close_price", closes),
        pl.Series("proxy_margin_bps", margin_bps),
        pl.Series("proxy_label_up", closes >= opens),
    )


def authentic_labels(labels: pl.DataFrame) -> pl.DataFrame:
    complete = labels.filter(
        pl.col("twap_open_price").is_not_null()
        & pl.col("twap_close_price").is_not_null()
        & pl.col("twap_open_source_timestamp").is_not_null()
        & pl.col("twap_close_source_timestamp").is_not_null()
        & (pl.col("twap_open_effective_timestamp_rows") == 1)
        & (pl.col("twap_close_effective_timestamp_rows") == 1)
        & (pl.col("twap_open_valid_from_timestamp") <= pl.col("twap_open_source_timestamp"))
        & (pl.col("twap_close_valid_from_timestamp") <= pl.col("twap_close_source_timestamp"))
    ).with_columns(
        (pl.col("twap_close_price") >= pl.col("twap_open_price")).alias(
            "authentic_label_up"
        ),
        (pl.col("twap_close_price") / pl.col("twap_open_price"))
        .log().mul(10_000.0).alias("authentic_margin_bps"),
        (pl.col("twap_close_price") == pl.col("twap_open_price")).alias("authentic_equality"),
    )
    duplicate_ids = complete.group_by("market_id").len().filter(pl.col("len") != 1)
    if duplicate_ids.height:
        raise RuntimeError("authentic TWAP label source contains duplicate markets")
    return complete


def select_proxy_convention(
    labels: pl.DataFrame,
    refprice: pl.DataFrame,
    *,
    calibration_start: datetime,
    calibration_end: datetime,
) -> tuple[ProxyConvention, dict[str, pl.DataFrame]]:
    authentic = authentic_labels(labels)
    candidates: dict[str, pl.DataFrame] = {}
    summaries: list[ProxyConvention] = []
    for column in ("valid_from_timestamp", "source_timestamp"):
        candidate = construct_proxy_labels(authentic, refprice, time_column=column)
        calibration = candidate.filter(
            pl.col("window_start").is_between(
                calibration_start, calibration_end, closed="left"
            )
        ).with_columns(
            (pl.col("proxy_open_price") / pl.col("twap_open_price"))
            .log().abs().mul(10_000.0).alias("open_error_bps"),
            (pl.col("proxy_close_price") / pl.col("twap_close_price"))
            .log().abs().mul(10_000.0).alias("close_error_bps"),
        )
        errors = np.concatenate(
            (calibration["open_error_bps"].to_numpy(), calibration["close_error_bps"].to_numpy())
        )
        errors = errors[np.isfinite(errors)]
        if not len(errors):
            raise RuntimeError("proxy convention calibration has no finite errors")
        summary = ProxyConvention(
            time_column=column,
            calibration_mae_bps=float(np.mean(errors)),
            calibration_p99_bps=float(np.quantile(errors, 0.99)),
            error_band_bps=float(max(np.quantile(errors, 0.995), 0.10)),
        )
        summaries.append(summary)
        candidates[column] = candidate
    selected = min(summaries, key=lambda item: (item.calibration_mae_bps, item.time_column))
    return selected, candidates


def attach_training_labels(
    frame: pl.DataFrame,
    labels: pl.DataFrame,
    refprice: pl.DataFrame,
    *,
    convention: ProxyConvention,
    authentic_start: datetime,
    current_start: datetime,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    proxy = construct_proxy_labels(labels, refprice, time_column=convention.time_column)
    valid_authentic = (
        pl.col("twap_open_price").is_not_null()
        & pl.col("twap_close_price").is_not_null()
        & (pl.col("twap_open_effective_timestamp_rows") == 1)
        & (pl.col("twap_close_effective_timestamp_rows") == 1)
        & (pl.col("twap_open_valid_from_timestamp") <= pl.col("twap_open_source_timestamp"))
        & (pl.col("twap_close_valid_from_timestamp") <= pl.col("twap_close_source_timestamp"))
    )
    joined_labels = proxy.with_columns(
        pl.when(valid_authentic)
        .then(pl.col("twap_close_price") >= pl.col("twap_open_price"))
        .otherwise(None)
        .alias("authentic_label_up"),
        pl.when(valid_authentic)
        .then((pl.col("twap_close_price") / pl.col("twap_open_price")).log() * 10_000.0)
        .otherwise(None)
        .alias("authentic_margin_bps"),
    ).with_columns(
        pl.when(pl.col("window_start") >= current_start)
        .then(pl.lit("official_current_twap60"))
        .when(pl.col("window_start") >= authentic_start)
        .then(pl.lit("authentic_counterfactual_twap60"))
        .otherwise(pl.lit("proxy_twap60"))
        .alias("label_regime"),
        pl.when(pl.col("window_start") >= authentic_start)
        .then(pl.col("authentic_label_up"))
        .otherwise(pl.col("proxy_label_up"))
        .alias("training_label_up"),
        pl.when(pl.col("window_start") >= authentic_start)
        .then(pl.lit(1.0))
        .when(pl.col("proxy_margin_bps").abs() >= 3.0 * convention.error_band_bps)
        .then(pl.lit(0.50))
        .when(pl.col("proxy_margin_bps").abs() >= 2.0 * convention.error_band_bps)
        .then(pl.lit(0.25))
        .when(pl.col("proxy_margin_bps").abs() >= convention.error_band_bps)
        .then(pl.lit(0.10))
        .otherwise(pl.lit(0.0))
        .alias("base_label_weight"),
    )
    selected = joined_labels.filter(
        pl.col("training_label_up").is_not_null() & (pl.col("base_label_weight") > 0)
    ).select(
        "market_id", "label_regime", "training_label_up", "base_label_weight",
        "authentic_label_up", "authentic_margin_bps", "proxy_label_up", "proxy_margin_bps",
        "official_outcome", "legacy_open_price", "legacy_close_price",
    )
    current = joined_labels.filter(
        (pl.col("window_start") >= current_start)
        & pl.col("authentic_label_up").is_not_null()
    )
    disagreement = current.filter(
        pl.col("authentic_label_up")
        != (pl.col("official_outcome") == pl.lit("up"))
    )
    if disagreement.height:
        raise RuntimeError(
            f"{disagreement.height} current TWAP60 labels disagree with official outcomes"
        )
    return (
        frame.drop("label_up", strict=False)
        .join(selected, on="market_id", how="inner", validate="m:1")
        .with_columns(pl.col("training_label_up").cast(pl.Int8).alias("label_up")),
        joined_labels,
    )


def attach_execution(frame: pl.DataFrame, execution: pl.DataFrame) -> pl.DataFrame:
    if execution.is_empty():
        raise RuntimeError("capacity execution source is empty")
    evidence = execution.filter(
        pl.col("seconds_elapsed").is_between(60, 179, closed="both")
        & ((pl.col("seconds_elapsed") - 60) % 5 == 0)
        & ((pl.col("quality_flags") & 63) == 0)
        & pl.col("up_provider_received_at").is_not_null()
        & pl.col("down_provider_received_at").is_not_null()
        & (pl.col("up_provider_received_at") <= pl.col("observed_at"))
        & (pl.col("down_provider_received_at") <= pl.col("observed_at"))
        & (pl.col("up_provider_received_at") >= pl.col("observed_at") - pl.duration(seconds=10))
        & (pl.col("down_provider_received_at") >= pl.col("observed_at") - pl.duration(seconds=10))
        & pl.all_horizontal(
            pl.col(column).is_not_null() & pl.col(column).is_finite()
            for column in BOOK_RAW_FEATURES
        )
    )
    evidence = evidence.unique(subset=["market_id", "observed_at"], keep="last")
    columns = [
        "market_id", "observed_at", "fee_rate", "up_provider_received_at",
        "down_provider_received_at", "up_best_ask", "down_best_ask",
        "up_ask_depth", "down_ask_depth", "quality_flags", *BOOK_RAW_FEATURES,
    ]
    joined = frame.join(
        evidence.select(*columns), on=["market_id", "observed_at"], how="left", validate="m:1"
    )
    return attach_book_features(joined)


def attach_candle_context(frame: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    return _attach_candle_features(frame, candles, max_age_seconds=120)


def attach_causal_refprice_features(
    frame: pl.DataFrame,
    refprice: pl.DataFrame,
    *,
    additional_delay_seconds: float = 0.0,
    maximum_age_seconds: float = 5.0,
) -> pl.DataFrame:
    """Attach causal features using receipt availability, never terminal labels."""

    if additional_delay_seconds < 0:
        raise ValueError("additional causal delay cannot be negative")
    path = (
        canonical_refprice_path(refprice)
        .sort(["provider_available_at", "source_timestamp", "archive_row_number"])
        .filter(
            pl.col("source_timestamp")
            == pl.col("source_timestamp").cum_max()
        )
    )
    available = path["provider_available_at"].to_numpy().astype("datetime64[us]").astype(np.int64)
    source = path["source_timestamp"].to_numpy().astype("datetime64[us]").astype(np.int64)
    valid = path["valid_from_timestamp"].to_numpy().astype("datetime64[us]").astype(np.int64)
    if np.any(np.diff(available) < 0) or np.any(np.diff(source) < 0):
        raise RuntimeError("refprice availability/source chronology is not monotonic")
    price = path["price"].to_numpy().astype(float)
    bid = path["bid"].to_numpy().astype(float)
    ask = path["ask"].to_numpy().astype(float)
    decision = frame["observed_at"].to_numpy().astype("datetime64[us]").astype(np.int64)
    effective_decision = decision - int(additional_delay_seconds * 1_000_000)
    current = np.searchsorted(available, effective_decision, side="right") - 1
    output: dict[str, np.ndarray] = {}
    eligible = current >= 0
    safe_current = np.maximum(current, 0)
    age = (decision - source[safe_current]) / 1_000_000.0
    eligible &= (age > 0) & (age <= maximum_age_seconds + additional_delay_seconds)
    output["chainlink_ref_age_seconds"] = age
    output["chainlink_ref_source_skew_seconds"] = (
        (source[safe_current] - valid[safe_current]) / 1_000_000.0
    )
    anchors: dict[int, np.ndarray] = {}
    for seconds in (1, 5, 15, 30, 60, 90, 120):
        target = decision - seconds * 1_000_000
        index = np.minimum(np.searchsorted(source, target, side="right") - 1, current)
        anchor_age = (target - source[np.maximum(index, 0)]) / 1_000_000.0
        good = (index >= 0) & (anchor_age >= 0) & (anchor_age <= maximum_age_seconds)
        eligible &= good
        anchors[seconds] = np.maximum(index, 0)
        output[f"chainlink_ref_return_{seconds}s_bps"] = (
            np.log(price[safe_current] / price[np.maximum(index, 0)]) * 10_000.0
        )
    spread = (ask - bid) / price * 10_000.0
    output["chainlink_ref_binance_basis_bps"] = (
        np.log(price[safe_current] / frame["btc_close"].to_numpy()) * 10_000.0
    )
    output["chainlink_ref_boundary_gap_bps"] = (
        np.log(price[safe_current] / frame["opening_boundary"].to_numpy()) * 10_000.0
    )
    output["chainlink_ref_spread_bps"] = spread[safe_current]
    output["chainlink_ref_spread_change_30s_bps"] = (
        spread[safe_current] - spread[anchors[30]]
    )
    output["chainlink_ref_binance_direction_agreement_30s"] = (
        np.sign(frame["btc_return_30s_bps"].to_numpy())
        * np.sign(output["chainlink_ref_return_30s_bps"])
    )
    output["chainlink_ref_momentum_alignment_5_30"] = (
        np.sign(output["chainlink_ref_return_5s_bps"])
        * np.sign(output["chainlink_ref_return_30s_bps"])
    )
    output["chainlink_ref_reversal_5_vs_30"] = (
        np.sign(output["chainlink_ref_return_5s_bps"])
        * np.sign(output["chainlink_ref_return_30s_bps"]) < 0
    ).astype(float)
    output["chainlink_ref_reversal_15_vs_60"] = (
        np.sign(output["chainlink_ref_return_15s_bps"])
        * np.sign(output["chainlink_ref_return_60s_bps"]) < 0
    ).astype(float)
    output["chainlink_ref_boundary_velocity_5s_bps"] = output["chainlink_ref_return_5s_bps"]
    output["chainlink_ref_binance_basis_velocity_5s_bps"] = (
        output["chainlink_ref_binance_basis_bps"]
        - (
            np.log(
                price[anchors[5]]
                / (
                    frame["btc_close"].to_numpy()
                    / np.exp(frame["btc_return_5s_bps"].to_numpy() / 10_000.0)
                )
            ) * 10_000.0
        )
    )
    output["chainlink_ref_binance_disagreement_30s"] = (
        output["chainlink_ref_binance_direction_agreement_30s"] < 0
    ).astype(float)

    for horizon in (5, 15, 30, 60, 90, 120):
        volatility = np.full(frame.height, np.nan)
        price_range = np.full(frame.height, np.nan)
        count = np.zeros(frame.height)
        maximum_gap = np.full(frame.height, np.nan)
        crosses = np.zeros(frame.height)
        changes = np.zeros(frame.height)
        efficiency = np.full(frame.height, np.nan)
        range_position = np.full(frame.height, np.nan)
        for row, end_index in enumerate(current):
            if end_index < 1:
                continue
            start_index = int(np.searchsorted(source, decision[row] - horizon * 1_000_000))
            sample = price[start_index : end_index + 1]
            sample_times = source[start_index : end_index + 1]
            if len(sample) < 2:
                continue
            returns = np.diff(np.log(sample)) * 10_000.0
            volatility[row] = float(np.sqrt(np.sum(returns * returns)))
            price_range[row] = float(np.log(sample.max() / sample.min()) * 10_000.0)
            count[row] = len(sample)
            maximum_gap[row] = float(np.diff(sample_times).max() / 1_000_000.0)
            boundary = frame["opening_boundary"][row]
            signs = sample >= boundary
            crosses[row] = np.count_nonzero(signs[1:] != signs[:-1])
            directions = np.sign(returns)
            changes[row] = np.count_nonzero(directions[1:] != directions[:-1])
            path_length = float(np.abs(returns).sum())
            efficiency[row] = abs(float(np.log(sample[-1] / sample[0]) * 10_000.0)) / max(
                path_length, 1e-9
            )
            range_position[row] = (sample[-1] - sample.min()) / max(
                sample.max() - sample.min(), 1e-9
            )
        output[f"chainlink_ref_realized_volatility_{horizon}s_bps"] = volatility
        if horizon >= 15:
            output[f"chainlink_ref_range_{horizon}s_bps"] = price_range
        if horizon in (30, 60):
            output[f"chainlink_ref_path_efficiency_{horizon}s"] = efficiency
        if horizon == 60:
            output["chainlink_ref_range_position_60s"] = range_position
            output["chainlink_ref_boundary_cross_count_60s"] = crosses
            output["chainlink_ref_direction_changes_60s"] = changes
            output["chainlink_ref_reports_60s"] = count
            output["chainlink_ref_max_gap_60s"] = maximum_gap

    for name in REFPRICE_ALL_FEATURES:
        values = output[name]
        eligible &= np.isfinite(values)
    attached = frame.with_columns(
        *[pl.Series(name, output[name]) for name in REFPRICE_ALL_FEATURES],
        pl.Series("refprice_causal_eligible", eligible),
        pl.lit(float(additional_delay_seconds)).alias("refprice_delay_stress_seconds"),
    )
    return attached


def verify_runtime_refprice_golden_vectors(
    frame: pl.DataFrame,
    refprice: pl.DataFrame,
    *,
    maximum_rows: int = 512,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Independently replay the existing Rust 10-feature refprice contract."""

    sample = frame.filter(pl.col("refprice_causal_eligible")).head(maximum_rows)
    path = canonical_refprice_path(refprice).sort("source_timestamp")
    source = path["source_timestamp"].to_numpy().astype("datetime64[us]").astype(np.int64)
    available = (
        path["provider_available_at"].to_numpy().astype("datetime64[us]").astype(np.int64)
    )
    price = path["price"].to_numpy().astype(float)
    spread = (path["ask"].to_numpy() - path["bid"].to_numpy()) / price * 10_000.0
    maximum_error = 0.0
    compared = 0
    vectors: list[dict[str, Any]] = []
    for row in sample.iter_rows(named=True):
        decision = np.datetime64(
            row["observed_at"].replace(tzinfo=None), "us"
        ).astype(np.int64)

        def report_at(target: int, available_by: int, *, strict: bool) -> int:
            side = "left" if strict else "right"
            index = int(np.searchsorted(source, target, side=side) - 1)
            while index >= 0 and available[index] > available_by:
                index -= 1
            return index

        current = report_at(decision, decision, strict=True)
        anchors = [
            report_at(decision - seconds * 1_000_000, decision, strict=False)
            for seconds in (1, 5, 15, 30, 60)
        ]
        if current < 0 or min(anchors) < 0:
            continue
        ages = [
            (decision - seconds * 1_000_000 - source[index]) / 1_000_000.0
            for index, seconds in zip(anchors, (1, 5, 15, 30, 60), strict=True)
        ]
        current_age = (decision - source[current]) / 1_000_000.0
        if not (0 < current_age <= 5 and all(0 <= age <= 5 for age in ages)):
            continue
        returns = [math.log(price[current] / price[index]) * 10_000.0 for index in anchors]
        expected = np.array(
            [
                *returns,
                math.log(price[current] / float(row["btc_close"])) * 10_000.0,
                math.log(price[current] / float(row["opening_boundary"])) * 10_000.0,
                spread[current],
                spread[current] - spread[anchors[3]],
                np.sign(float(row["btc_return_30s_bps"])) * np.sign(returns[3]),
            ],
            dtype=float,
        )
        actual = np.array([float(row[name]) for name in REFPRICE_RUNTIME_FEATURES])
        error = float(np.max(np.abs(expected - actual)))
        maximum_error = max(maximum_error, error)
        compared += 1
        if len(vectors) < 25:
            vectors.append(
                {
                    "market_id": row["market_id"],
                    "observed_at": row["observed_at"].isoformat(),
                    "expected": expected.tolist(),
                    "actual": actual.tolist(),
                }
            )
    vector_sha = hashlib.sha256(
        json.dumps(vectors, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "passed": compared > 0 and maximum_error <= tolerance,
        "compared_rows": compared,
        "maximum_absolute_error": maximum_error,
        "tolerance": tolerance,
        "feature_order": list(REFPRICE_RUNTIME_FEATURES),
        "golden_vector_sha256": vector_sha,
        "golden_vectors": vectors,
    }


def ensure_oracle_eligibility_compatibility(frame: pl.DataFrame) -> pl.DataFrame:
    if "early_oracle_eligible" in frame.columns:
        return frame
    if "oracle_model_eligible" not in frame.columns:
        raise RuntimeError("training core is missing oracle eligibility evidence")
    return frame.with_columns(
        pl.col("oracle_model_eligible").alias("early_oracle_eligible")
    )


def build_tournament_frame(
    paths: DataPaths,
    *,
    authentic_start: datetime,
    current_start: datetime,
    proxy_calibration_start: datetime,
    proxy_calibration_end: datetime,
) -> tuple[pl.DataFrame, pl.DataFrame, ProxyConvention, dict[str, Any]]:
    labels = load_source_group(paths, "labels")
    historical_core = pl.read_parquet(paths.core_features).filter(
        pl.col("seconds_elapsed").is_between(60, 179, closed="both")
    )
    current_core = build_current_core_features(
        load_source_group(paths, "core_current"), labels, load_source_group(paths, "oracle")
    )
    core = pl.concat(
        [historical_core, current_core], how="diagonal_relaxed", rechunk=True
    ).unique(subset=["market_id", "observed_at"], keep="last")
    core = ensure_oracle_eligibility_compatibility(core)
    refprice = load_source_group(paths, "refprice")
    candles = load_source_group(paths, "candles").unique(
        subset=["close_timestamp"], keep="last"
    )
    execution = load_source_group(paths, "execution").filter(
        pl.col("window_start") >= current_start
    )
    convention, _ = select_proxy_convention(
        labels, refprice,
        calibration_start=proxy_calibration_start,
        calibration_end=proxy_calibration_end,
    )
    labeled, label_audit = attach_training_labels(
        core, labels, refprice, convention=convention,
        authentic_start=authentic_start, current_start=current_start,
    )
    joined = attach_execution(labeled, execution)
    joined = attach_candle_context(joined, candles)
    joined = attach_causal_refprice_features(joined, refprice)
    joined = joined.sort(["window_start", "market_id", "seconds_elapsed"])
    manifest = {
        "schema_version": "btc-twap60-training-frame-v1",
        "rows": joined.height,
        "markets": joined["market_id"].n_unique(),
        "historical_core_rows": historical_core.height,
        "historical_core_markets": historical_core["market_id"].n_unique(),
        "current_core_rows": current_core.height,
        "current_core_markets": current_core["market_id"].n_unique(),
        "economic_rows": joined.filter(pl.col("up_ask_vwap_5").is_not_null()).height,
        "economic_markets": joined.filter(pl.col("up_ask_vwap_5").is_not_null())["market_id"].n_unique(),
        "execution_source_rows": execution.height,
        "execution_schema_versions": sorted(execution["schema_version"].unique().to_list()),
        "execution_artifact_ids": sorted(execution["artifact_id"].unique().to_list()),
        "proxy_convention": {
            "time_column": convention.time_column,
            "calibration_mae_bps": convention.calibration_mae_bps,
            "calibration_p99_bps": convention.calibration_p99_bps,
            "error_band_bps": convention.error_band_bps,
        },
        "feature_label_code_paths_separate": True,
        "database_mutations": False,
        "refprice_runtime_features": list(REFPRICE_RUNTIME_FEATURES),
        "refprice_diagnostic_features": list(REFPRICE_DIAGNOSTIC_FEATURES),
    }
    return joined, label_audit, convention, manifest


def proxy_validation_metrics(frame: pl.DataFrame, start: datetime, end: datetime) -> dict[str, Any]:
    block = frame.filter(
        pl.col("window_start").is_between(start, end, closed="left")
        & pl.col("authentic_label_up").is_not_null()
    ).with_columns(
        (pl.col("proxy_open_price") - pl.col("twap_open_price")).abs().alias("open_abs_error"),
        (pl.col("proxy_close_price") - pl.col("twap_close_price")).abs().alias("close_abs_error"),
        (pl.col("proxy_open_price") / pl.col("twap_open_price"))
        .log().abs().mul(10_000).alias("open_error_bps"),
        (pl.col("proxy_close_price") / pl.col("twap_close_price"))
        .log().abs().mul(10_000).alias("close_error_bps"),
    )
    if block.is_empty():
        return {"markets": 0}
    result = {
        "markets": block.height,
        "opening_absolute_price_error_mean": float(block["open_abs_error"].mean()),
        "opening_absolute_price_error_p95": float(block["open_abs_error"].quantile(0.95)),
        "closing_absolute_price_error_mean": float(block["close_abs_error"].mean()),
        "closing_absolute_price_error_p95": float(block["close_abs_error"].quantile(0.95)),
        "opening_error_bps_mean": float(block["open_error_bps"].mean()),
        "opening_error_bps_p95": float(block["open_error_bps"].quantile(0.95)),
        "closing_error_bps_mean": float(block["close_error_bps"].mean()),
        "closing_error_bps_p95": float(block["close_error_bps"].quantile(0.95)),
        "outcome_agreement": float(
            (block["proxy_label_up"] == block["authentic_label_up"]).mean()
        ),
        "near_boundary_agreement": float(
            (
                block.filter(pl.col("authentic_margin_bps").abs() < 5)["proxy_label_up"]
                == block.filter(pl.col("authentic_margin_bps").abs() < 5)["authentic_label_up"]
            ).mean()
        ) if block.filter(pl.col("authentic_margin_bps").abs() < 5).height else None,
    }
    block = block.with_columns(
        (pl.col("proxy_margin_bps") - pl.col("authentic_margin_bps"))
        .abs()
        .alias("proxy_twap_divergence_bps")
    )
    for feature, key in (
        ("btc_realized_volatility_60s_bps", "by_volatility_quartile"),
        ("chainlink_ref_reversal_5_vs_30", "by_refprice_reversal"),
        ("proxy_twap_divergence_bps", "by_refprice_twap_divergence_quartile"),
    ):
        if feature not in block.columns:
            continue
        if feature == "chainlink_ref_reversal_5_vs_30":
            grouped = block.with_columns(
                pl.when(pl.col(feature) > 0).then(pl.lit("reversal")).otherwise(pl.lit("no_reversal")).alias("slice")
            )
        else:
            q1, q2, q3 = (float(block[feature].quantile(q)) for q in (0.25, 0.50, 0.75))
            grouped = block.with_columns(
                pl.when(pl.col(feature) <= q1).then(pl.lit("q1"))
                .when(pl.col(feature) <= q2).then(pl.lit("q2"))
                .when(pl.col(feature) <= q3).then(pl.lit("q3"))
                .otherwise(pl.lit("q4")).alias("slice")
            )
        result[key] = {
            str(row["slice"]): {
                "markets": int(row["markets"]),
                "outcome_agreement": float(row["outcome_agreement"]),
                "opening_error_bps_mean": float(row["opening_error_bps_mean"]),
                "closing_error_bps_mean": float(row["closing_error_bps_mean"]),
            }
            for row in grouped.group_by("slice").agg(
                pl.len().alias("markets"),
                (pl.col("proxy_label_up") == pl.col("authentic_label_up"))
                .mean().alias("outcome_agreement"),
                pl.col("open_error_bps").mean().alias("opening_error_bps_mean"),
                pl.col("close_error_bps").mean().alias("closing_error_bps_mean"),
            ).iter_rows(named=True)
        }
    return result
