"""Checkpointed causal features from the existing Kraken L2 update archive."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .core_extract import file_sha256
from .multivenue_early_entry_data import KEY_COLUMNS

SCHEMA_VERSION = "btc-kraken-l2-update-flow-v1"
HORIZONS = (5, 15, 30, 60)
KRAKEN_L2_FEATURES = tuple(
    name
    for seconds in HORIZONS
    for name in (
        f"kraken_l2_update_imbalance_{seconds}s",
        f"kraken_l2_quantity_imbalance_{seconds}s",
        f"kraken_l2_cancel_imbalance_{seconds}s",
        f"kraken_l2_log_update_count_{seconds}s",
        f"kraken_l2_log_quote_quantity_{seconds}s",
    )
) + (
    "kraken_l2_last_update_spread_bps",
    "kraken_l2_last_update_mid_to_binance_bps",
    "kraken_l2_age_seconds",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _raw_files(root: Path, start: datetime, end: datetime) -> dict[str, list[Path]]:
    output: dict[str, list[Path]] = {}
    for path in root.glob("????-??-??/??/BTC_USD_orderbook.parquet"):
        day = path.parents[1].name
        when = datetime.fromisoformat(day).replace(tzinfo=UTC)
        if start <= when < end:
            output.setdefault(day, []).append(path)
    return {day: sorted(paths) for day, paths in sorted(output.items())}


def _source_identity(paths: list[Path]) -> list[dict[str, Any]]:
    identity = []
    for path in paths:
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        identity.append(
            {
                "path": str(path),
                "sha256": manifest["sha256"],
                "rows": int(manifest["parquet_rows"]),
                "bytes": int(manifest["compressed_bytes"]),
            }
        )
    return identity


def _daily_seconds(paths: list[Path]) -> pl.DataFrame:
    source = (
        pl.scan_parquet(paths, hive_partitioning=False)
        .select(
            pl.col("event_time").cast(pl.Int64),
            pl.col("side").cast(pl.String).str.to_lowercase(),
            pl.col("price").cast(pl.Float64),
            pl.col("quantity").cast(pl.Float64),
        )
        .filter(pl.col("side").is_in(["bid", "ask"]))
        .with_columns(
            # A second is available only after it closes, preventing same-second look-ahead.
            (((pl.col("event_time") // 1_000_000_000) + 1) * 1_000_000_000).alias("available_ns"),
            (pl.col("quantity") == 0).alias("is_cancel"),
        )
        .group_by("available_ns")
        .agg(
            (pl.col("side") == "bid").sum().alias("bid_updates"),
            (pl.col("side") == "ask").sum().alias("ask_updates"),
            pl.when(pl.col("side") == "bid")
            .then(pl.col("quantity"))
            .otherwise(0.0)
            .sum()
            .alias("bid_quantity"),
            pl.when(pl.col("side") == "ask")
            .then(pl.col("quantity"))
            .otherwise(0.0)
            .sum()
            .alias("ask_quantity"),
            (pl.col("is_cancel") & (pl.col("side") == "bid")).sum().alias("bid_cancels"),
            (pl.col("is_cancel") & (pl.col("side") == "ask")).sum().alias("ask_cancels"),
            pl.when(pl.col("side") == "bid")
            .then(pl.col("price"))
            .otherwise(None)
            .max()
            .alias("last_bid_update_price"),
            pl.when(pl.col("side") == "ask")
            .then(pl.col("price"))
            .otherwise(None)
            .min()
            .alias("last_ask_update_price"),
        )
        .sort("available_ns")
        .collect(engine="streaming")
    )
    if source.is_empty():
        return source
    start_ns = int(source["available_ns"].min())
    end_ns = int(source["available_ns"].max())
    grid = pl.DataFrame(
        {"available_ns": pl.int_range(start_ns, end_ns + 1_000_000_000, 1_000_000_000, eager=True)}
    )
    frame = grid.join(source, on="available_ns", how="left").with_columns(
        pl.col("bid_updates", "ask_updates", "bid_cancels", "ask_cancels").fill_null(0),
        pl.col("bid_quantity", "ask_quantity").fill_null(0.0),
        pl.col("last_bid_update_price", "last_ask_update_price").forward_fill(),
    )
    expressions: list[pl.Expr] = []
    for seconds in HORIZONS:
        bid_updates = pl.col("bid_updates").rolling_sum(seconds)
        ask_updates = pl.col("ask_updates").rolling_sum(seconds)
        bid_quantity = pl.col("bid_quantity").rolling_sum(seconds)
        ask_quantity = pl.col("ask_quantity").rolling_sum(seconds)
        bid_cancels = pl.col("bid_cancels").rolling_sum(seconds)
        ask_cancels = pl.col("ask_cancels").rolling_sum(seconds)
        expressions.extend(
            (
                ((bid_updates - ask_updates) / (bid_updates + ask_updates + 1.0)).alias(
                    f"kraken_l2_update_imbalance_{seconds}s"
                ),
                ((bid_quantity - ask_quantity) / (bid_quantity + ask_quantity + 1e-12)).alias(
                    f"kraken_l2_quantity_imbalance_{seconds}s"
                ),
                ((ask_cancels - bid_cancels) / (bid_cancels + ask_cancels + 1.0)).alias(
                    f"kraken_l2_cancel_imbalance_{seconds}s"
                ),
                (bid_updates + ask_updates).log1p().alias(f"kraken_l2_log_update_count_{seconds}s"),
                (bid_quantity + ask_quantity)
                .log1p()
                .alias(f"kraken_l2_log_quote_quantity_{seconds}s"),
            )
        )
    return frame.with_columns(*expressions).select(
        "available_ns", *KRAKEN_L2_FEATURES[:-3], "last_bid_update_price", "last_ask_update_price"
    )


def build_kraken_l2_features(
    *,
    raw_root: Path,
    cache: Path,
    start: datetime,
    end: datetime,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Build resumable daily update-flow partitions without altering the source archive."""

    cache.mkdir(parents=True, exist_ok=True)
    source_by_day = _raw_files(raw_root, start, end)
    if not source_by_day:
        raise RuntimeError("Kraken L2 archive contains no files in the configured interval")
    partitions: list[dict[str, Any]] = []
    for day, paths in source_by_day.items():
        destination = cache / "kraken-l2-daily" / f"{day}.parquet"
        source = _source_identity(paths)
        source_key = [(row["sha256"], row["rows"], row["bytes"]) for row in source]
        sidecar = destination.with_suffix(".json")
        valid = False
        if destination.is_file() and sidecar.is_file() and not force:
            cached = json.loads(sidecar.read_text())
            valid = cached.get("source_key") == source_key and cached.get("sha256") == file_sha256(
                destination
            )
        if not valid:
            daily = _daily_seconds(paths)
            destination.parent.mkdir(parents=True, exist_ok=True)
            daily.write_parquet(destination, compression="zstd", statistics=True)
            _write_json(
                sidecar,
                {
                    "schema_version": SCHEMA_VERSION,
                    "source_key": source_key,
                    "source": source,
                    "rows": daily.height,
                    "sha256": file_sha256(destination),
                },
            )
            print(f"Kraken L2 features: {day} {daily.height:,} seconds", flush=True)
        metadata = json.loads(sidecar.read_text())
        partitions.append({"day": day, "path": str(destination), **metadata})
        _write_json(
            cache / "kraken-l2-manifest.partial.json",
            {"schema_version": SCHEMA_VERSION, "partitions": partitions},
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "range_start": start.isoformat(),
        "range_end_exclusive": end.isoformat(),
        "availability_rule": "event second close plus one second",
        "representation": "incremental update-flow; not reconstructed full-book depth",
        "days_present": len(partitions),
        "partitions": partitions,
        "read_only_source": True,
        "database_mutations": False,
    }
    _write_json(cache / "kraken-l2-manifest.json", manifest)
    (cache / "kraken-l2-manifest.partial.json").unlink(missing_ok=True)
    return cache / "kraken-l2-manifest.json", manifest


def attach_kraken_l2(panel: pl.DataFrame, manifest: dict[str, Any]) -> pl.DataFrame:
    """Causally attach update-flow features while preserving every input row."""

    pieces: list[pl.DataFrame] = []
    for row in manifest["partitions"]:
        day = datetime.fromisoformat(row["day"]).replace(tzinfo=UTC)
        targets = panel.filter(
            pl.col("observed_at").is_between(day, day + timedelta(days=1), closed="left")
        ).select(*KEY_COLUMNS, "btc_close")
        if targets.is_empty():
            continue
        source = pl.read_parquet(row["path"]).with_columns(
            pl.from_epoch("available_ns", time_unit="ns")
            .dt.replace_time_zone("UTC")
            .cast(pl.Datetime("us", "UTC"))
            .alias("available_at")
        )
        joined = targets.sort("observed_at").join_asof(
            source.sort("available_at"),
            left_on="observed_at",
            right_on="available_at",
            strategy="backward",
        )
        pieces.append(
            joined.with_columns(
                (
                    (pl.col("last_ask_update_price") - pl.col("last_bid_update_price"))
                    / ((pl.col("last_ask_update_price") + pl.col("last_bid_update_price")) / 2.0)
                    * 10_000.0
                ).alias("kraken_l2_last_update_spread_bps"),
                (
                    ((pl.col("last_ask_update_price") + pl.col("last_bid_update_price")) / 2.0)
                    / pl.col("btc_close")
                )
                .log()
                .mul(10_000.0)
                .alias("kraken_l2_last_update_mid_to_binance_bps"),
                (pl.col("observed_at") - pl.col("available_at"))
                .dt.total_seconds()
                .alias("kraken_l2_age_seconds"),
            ).select(*KEY_COLUMNS, *KRAKEN_L2_FEATURES)
        )
    if pieces:
        features = pl.concat(pieces, how="vertical_relaxed", rechunk=True)
        output = panel.join(features, on=list(KEY_COLUMNS), how="left", validate="1:1")
    else:
        output = panel.with_columns(
            *(pl.lit(None, dtype=pl.Float64).alias(name) for name in KRAKEN_L2_FEATURES)
        )
    output = output.with_columns(
        pl.any_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite() for name in KRAKEN_L2_FEATURES
        ).alias("has_kraken_l2")
    )
    if (
        output.height != panel.height
        or output["market_id"].n_unique() != panel["market_id"].n_unique()
    ):
        raise RuntimeError("optional Kraken L2 attachment changed market coverage")
    return output
