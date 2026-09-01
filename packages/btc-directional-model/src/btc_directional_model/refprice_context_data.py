"""Immutable, read-only source artifacts for a RefPrice-primary tournament.

The extractor reads only existing relations. It writes checksum-sealed daily
Parquet partitions and a resumable local manifest; it never writes to the
database, creates a data source, or exports a runtime model. TWAP-60 is retained
only as target provenance and is explicitly forbidden from inference inputs.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .core_extract import (
    configure_read_only_connection,
    database_connection,
    file_sha256,
    write_json_atomic,
)
from .refprice_twap_training import _query_frame
from .twap60_training_data import piecewise_average

SCHEMA_VERSION = "btc-refprice-context-source-cache-v2"
PROFILE = "btc_5m_refprice_context_source"
ENTRY_SECONDS = tuple(range(60, 181, 5))
SOURCE_SQL = {
    "labels": "btc-refprice-context-label-source.sql",
    "core": "btc-refprice-context-core-source.sql",
    "refprice": "btc-refprice-context-refprice-source.sql",
    "oracle": "btc-refprice-context-oracle-source.sql",
    "candles": "btc-refprice-context-candle-source.sql",
    "open_interest": "btc-refprice-context-open-interest-source.sql",
    "aggregate_trades": "btc-refprice-context-trade-print-source.sql",
    "spot_l2": "btc-refprice-context-l2-source.sql",
}
SOURCE_RELATIONS = {
    "labels": ("polymarket.btc_interval_markets",),
    "core": (
        "polymarket.binance_one_second_klines",
        "market_data.binance_spot_btcusdt_one_second_ohlcv",
    ),
    "refprice": ("market_data.chainlink_btcusd_reference_prices",),
    "oracle": (
        "polymarket.polygon_chainlink_btcusd_oracle_rounds",
        "market_data.polygon_chainlink_btcusd_oracle_rounds",
    ),
    "candles": (
        "polymarket.chainlink_btcusd_one_minute_candles",
        "market_data.chainlink_btcusd_one_minute_candles",
    ),
    "open_interest": (
        "polymarket.binance_btcusdt_five_minute_open_interest",
        "market_data.binance_futures_btcusdt_open_interest",
    ),
    "aggregate_trades": (
        "polymarket.binance_aggregate_trades",
        "market_data.binance_spot_btcusdt_aggregate_trades",
    ),
    "spot_l2": ("polymarket.binance_spot_btcusdt_l2_training_features",),
}
TIMESTAMP_COLUMNS = {
    "labels": "window_start",
    "core": "observed_at",
    "refprice": "source_timestamp",
    "oracle": "oracle_source_timestamp",
    "candles": "close_timestamp",
    "open_interest": "source_timestamp",
    "aggregate_trades": "second_start",
    "spot_l2": "available_at",
}
SUPERVISION_ONLY_PREFIXES = (
    "twap_",
    "reconstructed_twap60_",
    "official_",
    "target_",
    "legacy_",
)
FORBIDDEN_INFERENCE_TOKENS = (
    "twap30",
    "twap60",
    "twap_30",
    "twap_60",
    "synthetic_label",
    "official_outcome",
    "target_label",
    "resolution_price",
    "final_price",
)


@dataclass(frozen=True)
class SourceConfig:
    source_path: Path
    package_root: Path
    data_start: datetime
    data_end: datetime
    official_twap_start: datetime
    l2_feature_end: datetime
    reconstruction_time_column: str
    reconstruction_calibration_mae_bps: float
    reconstruction_calibration_p99_bps: float
    reconstruction_error_band_bps: float
    reconstruction_source_run: str
    open_interest_start: datetime
    open_interest_full_day_start: datetime
    aggregate_trades_start: datetime
    observation_seconds: tuple[int, ...]
    destination: Path


def load_config(path: Path) -> SourceConfig:
    source = path.resolve()
    package_root = source.parent.parent
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    training = raw["training"]
    required_flags = {
        "training_only": True,
        "paper_only": True,
        "live_capital_allowed": False,
        "runtime_exported": False,
        "database_mutations": False,
        "twap_inference_allowed": False,
    }
    if training.get("profile") != PROFILE:
        raise ValueError("unexpected RefPrice-context source profile")
    for name, expected in required_flags.items():
        if training.get(name) is not expected:
            raise ValueError(f"invalid training-only source flag: {name}")
    windows = raw["windows"]
    labels = raw["labels"]
    availability = raw["availability"]
    config = SourceConfig(
        source_path=source,
        package_root=package_root,
        data_start=_utc(windows["data_start"]),
        data_end=_utc(windows["data_end"]),
        official_twap_start=_utc(windows["official_twap_start"]),
        l2_feature_end=_utc(windows["l2_feature_end"]),
        reconstruction_time_column=str(labels["reconstruction_time_column"]),
        reconstruction_calibration_mae_bps=float(labels["reconstruction_calibration_mae_bps"]),
        reconstruction_calibration_p99_bps=float(labels["reconstruction_calibration_p99_bps"]),
        reconstruction_error_band_bps=float(labels["reconstruction_error_band_bps"]),
        reconstruction_source_run=str(labels["reconstruction_source_run"]),
        open_interest_start=_utc(availability["open_interest_start"]),
        open_interest_full_day_start=_utc(availability["open_interest_full_day_start"]),
        aggregate_trades_start=_utc(availability["aggregate_trades_start"]),
        observation_seconds=tuple(int(value) for value in raw["observations"]["seconds"]),
        destination=package_root / str(raw["paths"]["data"]),
    )
    _validate_config(config)
    return config


def _validate_config(config: SourceConfig) -> None:
    if not (
        config.data_start < config.l2_feature_end < config.official_twap_start < config.data_end
    ):
        raise ValueError("source windows must be strictly chronological")
    if any(
        value.hour or value.minute or value.second or value.microsecond
        for value in (config.data_start, config.data_end)
    ):
        raise ValueError("source boundaries must be UTC day boundaries")
    if config.observation_seconds != ENTRY_SECONDS:
        raise ValueError("RefPrice-context observation schedule changed")
    if config.reconstruction_time_column != "source_timestamp":
        raise ValueError("RefPrice target reconstruction convention changed")
    if not (
        config.data_start
        < config.open_interest_start
        < config.open_interest_full_day_start
        < config.aggregate_trades_start
        < config.data_end
    ):
        raise ValueError("optional source applicability windows changed")
    sql_root = config.package_root / "sql"
    missing = [name for name in SOURCE_SQL.values() if not (sql_root / name).is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))


def extract_source_artifacts(config: SourceConfig) -> dict[str, Any]:
    """Extract daily source groups with checksum-verified resume checkpoints."""

    destination = config.destination
    destination.mkdir(parents=True, exist_ok=True)
    sql_root = config.package_root / "sql"
    contract = {
        "schema_version": SCHEMA_VERSION,
        "range_start": config.data_start.isoformat(),
        "range_end": config.data_end.isoformat(),
        "watermark_semantics": "half_open_exclusive",
        "official_twap_start": config.official_twap_start.isoformat(),
        "l2_feature_end": config.l2_feature_end.isoformat(),
        "observation_seconds": list(config.observation_seconds),
        "read_only": True,
        "database_mutations": False,
        "new_sources": False,
        "new_tables": False,
        "twap_inference_allowed": False,
        "twap_role": "supervision_only",
        "label_reconstruction": {
            "time_column": config.reconstruction_time_column,
            "calibration_mae_bps": config.reconstruction_calibration_mae_bps,
            "calibration_p99_bps": config.reconstruction_calibration_p99_bps,
            "error_band_bps": config.reconstruction_error_band_bps,
            "source_run": config.reconstruction_source_run,
        },
        "source_applicability": {
            "open_interest_start": config.open_interest_start.isoformat(),
            "open_interest_full_day_start": (config.open_interest_full_day_start.isoformat()),
            "aggregate_trades_start": config.aggregate_trades_start.isoformat(),
        },
        "source_relations": {name: list(relations) for name, relations in SOURCE_RELATIONS.items()},
        "query_sha256": {
            name: file_sha256(sql_root / sql_name) for name, sql_name in SOURCE_SQL.items()
        },
    }
    final_path = destination / "source-manifest.json"
    checkpoint_path = destination / "source-manifest.partial.json"
    if final_path.is_file() and not checkpoint_path.exists():
        manifest = json.loads(final_path.read_text())
        _verify_manifest(manifest, contract, destination)
        return manifest

    partitions: dict[str, list[dict[str, Any]]] = {name: [] for name in SOURCE_SQL}
    days: list[dict[str, Any]] = []
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text())
        _verify_manifest(checkpoint, contract, destination)
        partitions = checkpoint["partitions"]
        days = checkpoint["days"]
    completed = {row["date"] for row in days}
    queries = {name: (sql_root / sql_name).read_text() for name, sql_name in SOURCE_SQL.items()}

    day = config.data_start
    while day < config.data_end:
        end = min(day + timedelta(days=1), config.data_end)
        date_key = day.date().isoformat()
        if date_key in completed:
            day = end
            continue
        connection = database_connection()
        configure_read_only_connection(connection)
        try:
            frames = _extract_day(connection, config, queries, day, end)
        finally:
            connection.close()
        day_audit = _daily_audit(frames, config, day)
        for name, frame in frames.items():
            group_dir = destination / name
            group_dir.mkdir(parents=True, exist_ok=True)
            path = group_dir / f"{date_key}.parquet"
            temporary = path.with_suffix(".parquet.partial")
            frame.write_parquet(temporary, compression="zstd", statistics=True)
            temporary.replace(path)
            partitions[name].append(
                {
                    "date": date_key,
                    "path": str(path.relative_to(destination)),
                    "rows": frame.height,
                    "sha256": file_sha256(path),
                    "audit": _frame_audit(frame, name),
                }
            )
        days.append(day_audit)
        write_json_atomic(
            checkpoint_path,
            {**contract, "partitions": partitions, "days": days},
        )
        print(
            f"RefPrice source {date_key}: "
            f"{day_audit['target_markets']} targets, "
            f"{day_audit['complete_core_markets']} complete core markets, "
            f"{day_audit['status']}",
            flush=True,
        )
        day = end

    manifest = {
        **contract,
        "created_at": datetime.now(UTC).isoformat(),
        "partitions": partitions,
        "days": days,
        "totals": {
            name: {
                "partitions": len(rows),
                "rows": sum(int(row["rows"]) for row in rows),
            }
            for name, rows in partitions.items()
        },
        "readiness": _overall_readiness(days),
        "inference_contract": {
            "primary_signal": "Chainlink RefPrice",
            "forbidden_tokens": list(FORBIDDEN_INFERENCE_TOKENS),
            "supervision_only_prefixes": list(SUPERVISION_ONLY_PREFIXES),
            "economic_evaluation_start": config.official_twap_start.isoformat(),
            "economic_reason": ("pre-cutover contracts resolved under a different settlement rule"),
            "l2_scope": ("matched historical ablation only; raw live snapshots are excluded"),
        },
    }
    write_json_atomic(final_path, manifest)
    checkpoint_path.unlink(missing_ok=True)
    return manifest


def _extract_day(
    connection: Any,
    config: SourceConfig,
    queries: dict[str, str],
    start: datetime,
    end: datetime,
) -> dict[str, pl.DataFrame]:
    batch = {"batch_start": start, "batch_end": end}
    range_parameters = {"range_start": start, "range_end": end}
    frames = {
        "labels": _query_frame(
            connection,
            queries["labels"],
            batch,
            f"ref_context_labels_{start:%Y%m%d}",
        ),
        "core": _query_frame(
            connection, queries["core"], batch, f"ref_context_core_{start:%Y%m%d}"
        ),
        "refprice": _query_frame(
            connection,
            queries["refprice"],
            range_parameters,
            f"ref_context_refprice_{start:%Y%m%d}",
        ),
        "oracle": _query_frame(
            connection,
            queries["oracle"],
            range_parameters,
            f"ref_context_oracle_{start:%Y%m%d}",
        ),
        "candles": _query_frame(
            connection,
            queries["candles"],
            {**range_parameters, "history_minutes": 65, "candle_symbol": "BTCUSD"},
            f"ref_context_candles_{start:%Y%m%d}",
        ),
        "open_interest": _query_frame(
            connection,
            queries["open_interest"],
            range_parameters,
            f"ref_context_oi_{start:%Y%m%d}",
        ),
        "aggregate_trades": _query_frame(
            connection,
            queries["aggregate_trades"],
            batch,
            f"ref_context_trades_{start:%Y%m%d}",
        ),
    }
    if start < config.l2_feature_end:
        frames["spot_l2"] = _query_frame(
            connection,
            queries["spot_l2"],
            batch,
            f"ref_context_l2_{start:%Y%m%d}",
        )
    else:
        frames["spot_l2"] = pl.DataFrame()
    frames["labels"] = _attach_refprice_reconstructed_targets(
        frames["labels"],
        frames["refprice"],
        official_twap_start=config.official_twap_start,
        time_column=config.reconstruction_time_column,
    )
    return frames


def _attach_refprice_reconstructed_targets(
    labels: pl.DataFrame,
    refprice: pl.DataFrame,
    *,
    official_twap_start: datetime,
    time_column: str,
) -> pl.DataFrame:
    """Attach the frozen historical TWAP-60 target reconstructed from RefPrice."""

    if time_column != "source_timestamp":
        raise ValueError("unsupported RefPrice reconstruction clock")
    required = {time_column, "price", "available_at", "report_sha256"}
    missing = sorted(required - set(refprice.columns))
    if missing:
        raise RuntimeError("RefPrice reconstruction columns missing: " + ", ".join(missing))
    path = (
        refprice.filter(
            pl.col(time_column).is_not_null()
            & pl.col("price").is_not_null()
            & pl.col("price").is_finite()
            & (pl.col("price") > 0)
        )
        .sort([time_column, "available_at", "report_sha256"])
        .unique(subset=[time_column], keep="last", maintain_order=True)
        .sort(time_column)
    )
    if path.is_empty():
        return labels.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("reconstructed_twap60_open_price"),
            pl.lit(None, dtype=pl.Float64).alias("reconstructed_twap60_close_price"),
            pl.lit(None, dtype=pl.Float64).alias("reconstructed_twap60_margin_bps"),
            pl.lit(None, dtype=pl.Int8).alias("reconstructed_twap60_label_up"),
            pl.lit(None, dtype=pl.Int8).alias("target_label_up"),
            pl.lit(None, dtype=pl.Float64).alias("target_margin_bps"),
            pl.lit("unavailable").alias("target_label_source"),
        )
    opens = piecewise_average(
        path[time_column].to_numpy(),
        path["price"].to_numpy(),
        labels["window_start"].to_numpy(),
    )
    closes = piecewise_average(
        path[time_column].to_numpy(),
        path["price"].to_numpy(),
        labels["window_end"].to_numpy(),
    )
    timestamps = path[time_column].to_numpy().astype("datetime64[us]").astype(np.int64)
    starts = labels["window_start"].to_numpy().astype("datetime64[us]").astype(np.int64)
    ends = labels["window_end"].to_numpy().astype("datetime64[us]").astype(np.int64)
    start_index = np.searchsorted(timestamps, starts, side="right") - 1
    end_index = np.searchsorted(timestamps, ends, side="right") - 1
    start_age = np.full(labels.height, np.inf, dtype=np.float64)
    end_age = np.full(labels.height, np.inf, dtype=np.float64)
    valid_start = start_index >= 0
    valid_end = end_index >= 0
    start_age[valid_start] = (
        starts[valid_start] - timestamps[start_index[valid_start]]
    ) / 1_000_000.0
    end_age[valid_end] = (ends[valid_end] - timestamps[end_index[valid_end]]) / 1_000_000.0
    finite = (
        np.isfinite(opens)
        & np.isfinite(closes)
        & (opens > 0)
        & (closes > 0)
        & (start_age >= 0)
        & (start_age <= 5.0)
        & (end_age >= 0)
        & (end_age <= 5.0)
    )
    margins = np.full(labels.height, np.nan, dtype=np.float64)
    margins[finite] = np.log(closes[finite] / opens[finite]) * 10_000.0
    reconstructed = [
        int(closes[index] >= opens[index]) if finite[index] else None
        for index in range(labels.height)
    ]
    with_proxy = labels.with_columns(
        pl.Series("reconstructed_twap60_open_price", opens),
        pl.Series("reconstructed_twap60_close_price", closes),
        pl.Series("reconstructed_twap60_margin_bps", margins),
        pl.Series("reconstructed_twap60_label_up", reconstructed, dtype=pl.Int8),
    )
    official = pl.col("window_start") >= official_twap_start
    official_available = official & pl.col("official_label_up").is_not_null()
    return with_proxy.with_columns(
        pl.when(official_available)
        .then(pl.col("official_label_up"))
        .otherwise(pl.col("reconstructed_twap60_label_up"))
        .alias("target_label_up"),
        pl.when(pl.col("reconstructed_twap60_margin_bps").is_finite())
        .then(pl.col("reconstructed_twap60_margin_bps"))
        .otherwise(None)
        .alias("target_margin_bps"),
        pl.when(official_available)
        .then(pl.lit("official_twap60"))
        .when(pl.col("reconstructed_twap60_label_up").is_not_null())
        .then(
            pl.when(official)
            .then(pl.lit("refprice_reconstructed_training_fallback"))
            .otherwise(pl.lit("refprice_reconstructed_twap60"))
        )
        .otherwise(pl.lit("unavailable"))
        .alias("target_label_source"),
    )


def _daily_audit(
    frames: dict[str, pl.DataFrame], config: SourceConfig, day: datetime
) -> dict[str, Any]:
    labels = frames["labels"]
    duplicate_labels = labels.height - labels["market_id"].n_unique() if labels.height else 0
    target = labels.filter(pl.col("target_label_up").is_not_null()) if labels.height else labels
    official_disagreements = 0
    if labels.height and day >= config.official_twap_start:
        official_disagreements = labels.filter(
            pl.col("official_label_up").is_not_null()
            & pl.col("reconstructed_twap60_label_up").is_not_null()
            & (pl.col("official_label_up") != pl.col("reconstructed_twap60_label_up"))
        ).height

    core = frames["core"]
    if core.height:
        complete = (
            core.filter(pl.col("seconds_elapsed").is_in(ENTRY_SECONDS))
            .group_by("market_id")
            .agg(pl.col("seconds_elapsed").n_unique().alias("required_seconds"))
            .filter(pl.col("required_seconds") == len(ENTRY_SECONDS))
        )
        complete_markets = complete.height
    else:
        complete = pl.DataFrame({"market_id": []})
        complete_markets = 0

    candles = frames["candles"]
    interest = frames["open_interest"]
    scheduled_ids = set(labels["market_id"].to_list()) if labels.height else set()
    target_ids = set(target["market_id"].to_list()) if target.height else set()
    complete_ids = set(complete["market_id"].to_list()) if core.height else set()
    missing_target_ids = sorted(scheduled_ids - target_ids)
    missing_core_ids = sorted(target_ids - complete_ids)
    excluded_market_ids = sorted(set(missing_target_ids) | set(missing_core_ids))
    reasons: list[str] = []
    diagnostics: list[str] = []
    if duplicate_labels:
        reasons.append("duplicate_label_market")
    if missing_target_ids:
        diagnostics.append("explicit_missing_target_exclusion")
    if missing_core_ids:
        diagnostics.append("explicit_core_market_exclusion")
    if official_disagreements:
        diagnostics.append("refprice_proxy_official_disagreement")
    if candles.height != 1_505:
        diagnostics.append("incomplete_chainlink_candles")
    if day == config.open_interest_start and interest.height < 288:
        diagnostics.append("incomplete_open_interest_start_day")
    if day >= config.open_interest_full_day_start and interest.height != 301:
        diagnostics.append("incomplete_open_interest")
    in_day_refprice = frames["refprice"].filter(
        pl.col("source_timestamp").is_between(day, day + timedelta(days=1), closed="left")
    )
    if in_day_refprice.is_empty():
        diagnostics.append("missing_in_day_refprice")
    if frames["oracle"].is_empty():
        diagnostics.append("missing_oracle")
    if day >= config.aggregate_trades_start and frames["aggregate_trades"].is_empty():
        diagnostics.append("missing_aggregate_trades")
    return {
        "date": day.date().isoformat(),
        "scheduled_markets": labels.height,
        "target_markets": target.height,
        "complete_core_markets": complete_markets,
        "official_label_disagreements": official_disagreements,
        "spot_l2_rows": frames["spot_l2"].height,
        "excluded_market_ids": excluded_market_ids,
        "missing_target_market_ids": missing_target_ids,
        "missing_core_market_ids": missing_core_ids,
        "status": ("blocked" if reasons else "ready_with_exclusions" if diagnostics else "ready"),
        "reasons": reasons,
        "diagnostics": diagnostics,
    }


def _frame_audit(frame: pl.DataFrame, source_name: str) -> dict[str, Any]:
    audit: dict[str, Any] = {"rows": frame.height}
    if frame.is_empty():
        return audit
    timestamp = TIMESTAMP_COLUMNS[source_name]
    if timestamp in frame.columns:
        audit["minimum_timestamp"] = _json_value(frame[timestamp].min())
        audit["maximum_timestamp"] = _json_value(frame[timestamp].max())
    if "market_id" in frame.columns:
        audit["markets"] = frame["market_id"].n_unique()
    for column in ("source_relation", "kline_source_relation", "target_label_source"):
        if column in frame.columns:
            counts = frame.group_by(column).len().sort(column)
            audit[f"{column}_counts"] = {
                str(row[column]): int(row["len"]) for row in counts.to_dicts()
            }
    return audit


def _overall_readiness(days: list[dict[str, Any]]) -> dict[str, Any]:
    blocked = [row for row in days if row["status"] == "blocked"]
    exclusions = [
        {"date": row["date"], "market_ids": row["excluded_market_ids"]}
        for row in days
        if row["excluded_market_ids"]
    ]
    return {
        "status": "ready" if not blocked else "blocked",
        "ready_days": len(days) - len(blocked),
        "blocked_days": len(blocked),
        "blocked_dates": [row["date"] for row in blocked],
        "explicit_market_exclusions": exclusions,
        "training_must_exclude_blocked_dates": bool(blocked),
        "coverage_findings_do_not_block_training": True,
    }


def _verify_manifest(manifest: dict[str, Any], contract: dict[str, Any], root: Path) -> None:
    for key, expected in contract.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"source artifact contract changed: {key}")
    for rows in manifest.get("partitions", {}).values():
        for row in rows:
            path = root / row["path"]
            if not path.is_file() or file_sha256(path) != row["sha256"]:
                raise RuntimeError(f"source artifact partition changed: {row['path']}")


def validate_inference_columns(columns: tuple[str, ...]) -> None:
    """Reject target, completed-market, or TWAP signals from a model matrix."""

    invalid = sorted(
        name
        for name in columns
        if name.startswith(SUPERVISION_ONLY_PREFIXES)
        or any(token in name for token in FORBIDDEN_INFERENCE_TOKENS)
    )
    if invalid:
        raise ValueError("forbidden RefPrice-context inference columns: " + ", ".join(invalid))


def _json_value(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = extract_source_artifacts(config)
    print(json.dumps(manifest["readiness"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
