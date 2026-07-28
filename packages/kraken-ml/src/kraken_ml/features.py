from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .config import BenchmarkConfig
from .dataset import Snapshot, _atomic_json, _immutable_json, _sha256

PRICE_FEATURES = [
    "return_1_bps",
    "return_2_bps",
    "return_4_bps",
    "return_8_bps",
    "return_16_bps",
    "return_32_bps",
    "return_96_bps",
    "range_bps",
    "body_bps",
    "close_location",
    "volatility_4_bps",
    "volatility_16_bps",
    "volatility_96_bps",
    "trade_candle_volume_log",
    "trade_candle_volume_z16",
    "trade_candle_volume_z96",
    "mark_spot_basis_bps",
    "trade_mark_basis_bps",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

FLOW_FEATURES = PRICE_FEATURES + [
    "future_basis",
    "future_basis_delta_1",
    "future_basis_z96",
    "oi_log",
    "oi_change_1_bps",
    "oi_change_4_bps",
    "oi_change_16_bps",
    "oi_z96",
    "aggressor_differential",
    "aggressor_z16",
    "aggressor_z96",
    "trade_volume_log",
    "trade_volume_z16",
    "trade_count_log",
    "trade_count_z16",
    "cvd_delta_1",
    "cvd_delta_4",
    "cvd_delta_z96",
    "buy_sell_imbalance",
    "liquidation_log",
    "liquidation_z96",
]

FULL_FEATURES = FLOW_FEATURES + [
    "spread_bps",
    "execution_cost_1k_bps",
    "execution_cost_10k_bps",
    "ask_slippage_1k_bps",
    "bid_slippage_1k_bps",
    "ask_slippage_10k_bps",
    "bid_slippage_10k_bps",
    "liquidity_005_log",
    "liquidity_005_imbalance",
    "liquidity_01_log",
    "liquidity_01_imbalance",
    "liquidity_025_log",
    "liquidity_025_imbalance",
    "liquidity_05_log",
    "liquidity_05_imbalance",
    "liquidity_10_log",
    "liquidity_10_imbalance",
]

FEATURE_SETS = {
    "price": PRICE_FEATURES,
    "flow": FLOW_FEATURES,
    "full": FULL_FEATURES,
}
FEATURE_SCHEMA_VERSION = 4

TARGET_COLUMNS = [
    "label",
    "long_net_bps",
    "short_net_bps",
    "gross_forward_bps",
    "execution_cost_bps",
    "market_execution_cost_bps",
    "long_market_execution_cost_bps",
    "short_market_execution_cost_bps",
    "fee_cost_bps",
    "funding_horizon_bps",
    "momentum_label",
    "feature_available_at",
    "entry_at",
    "label_exit_at",
]


@dataclass(frozen=True)
class FeatureSnapshot:
    path: Path
    sha256: str
    row_count: int
    manifest_path: Path


def _zscore(column: str, window: int, alias: str) -> pl.Expr:
    value = pl.col(column)
    mean = value.rolling_mean(window_size=window)
    std = value.rolling_std(window_size=window)
    return ((value - mean) / std.clip(lower_bound=1e-12)).alias(alias)


def _imbalance(bid: str, ask: str, alias: str) -> pl.Expr:
    denominator = pl.col(bid) + pl.col(ask)
    return ((pl.col(bid) - pl.col(ask)) / denominator.clip(lower_bound=1e-12)).alias(alias)


def build_feature_frame(raw: pl.DataFrame, config: BenchmarkConfig) -> pl.DataFrame:
    horizon = config.dataset.horizon_bars
    exit_shift = horizon + 1
    fee_bps = config.dataset.taker_fee_bps_per_side * 2.0
    interval = config.dataset.interval_seconds

    mid = (pl.col("ask_best") + pl.col("bid_best")) / 2.0
    frame = raw.sort("bucket_start").with_columns(
        [
            (pl.col("bucket_start") + pl.duration(seconds=interval)).alias("feature_available_at"),
            pl.col("bucket_start").shift(-1).alias("entry_at"),
            pl.col("bucket_start").shift(-exit_shift).alias("label_exit_at"),
            (10_000.0 * (pl.col("ask_best") - pl.col("bid_best")) / mid).alias("spread_bps"),
            (10_000.0 * (pl.col("ask_slippage_1k") - pl.col("bid_slippage_1k")) / mid).alias(
                "execution_cost_1k_bps"
            ),
            (10_000.0 * (pl.col("ask_slippage_10k") - pl.col("bid_slippage_10k")) / mid).alias(
                "execution_cost_10k_bps"
            ),
            (10_000.0 * (pl.col("ask_slippage_1k") - mid) / mid).alias("ask_slippage_1k_bps"),
            (10_000.0 * (mid - pl.col("bid_slippage_1k")) / mid).alias("bid_slippage_1k_bps"),
            (10_000.0 * (pl.col("ask_slippage_10k") - mid) / mid).alias("ask_slippage_10k_bps"),
            (10_000.0 * (mid - pl.col("bid_slippage_10k")) / mid).alias("bid_slippage_10k_bps"),
        ]
    )

    if config.dataset.reference_notional_usd not in (1_000, 10_000):
        raise ValueError("reference_notional_usd must be 1000 or 10000")
    notional_suffix = "1k" if config.dataset.reference_notional_usd == 1_000 else "10k"
    ask_cost = f"ask_slippage_{notional_suffix}_bps"
    bid_cost = f"bid_slippage_{notional_suffix}_bps"

    funding_terms = [
        (pl.col("relative_funding_rate").shift(-offset).fill_null(0.0) * (interval / 3_600.0))
        for offset in range(1, horizon + 1)
    ]
    frame = frame.with_columns(
        [
            (
                10_000.0
                * (pl.col("trade_open").shift(-exit_shift) / pl.col("trade_open").shift(-1) - 1.0)
            ).alias("gross_forward_bps"),
            # Analytics timestamps identify 15-minute buckets. At a next-open
            # fill, the just-completed bucket is the last observable book-cost
            # estimate; using the new bucket would require post-fill data.
            (pl.col(ask_cost) + pl.col(bid_cost).shift(-horizon)).alias(
                "long_market_execution_cost_bps"
            ),
            (pl.col(bid_cost) + pl.col(ask_cost).shift(-horizon)).alias(
                "short_market_execution_cost_bps"
            ),
            pl.lit(fee_bps).alias("fee_cost_bps"),
            (10_000.0 * pl.sum_horizontal(funding_terms)).alias("funding_horizon_bps"),
        ]
    )
    frame = frame.with_columns(
        [
            (
                (
                    pl.col("long_market_execution_cost_bps")
                    + pl.col("short_market_execution_cost_bps")
                )
                / 2.0
            ).alias("market_execution_cost_bps"),
        ]
    )
    frame = frame.with_columns(
        [
            (pl.col("market_execution_cost_bps") + pl.col("fee_cost_bps")).alias(
                "execution_cost_bps"
            ),
        ]
    )
    frame = frame.with_columns(
        [
            (
                pl.col("gross_forward_bps")
                - pl.col("long_market_execution_cost_bps")
                - pl.col("fee_cost_bps")
                - pl.col("funding_horizon_bps")
            ).alias("long_net_bps"),
            (
                -pl.col("gross_forward_bps")
                - pl.col("short_market_execution_cost_bps")
                - pl.col("fee_cost_bps")
                + pl.col("funding_horizon_bps")
            ).alias("short_net_bps"),
        ]
    )
    buffer = config.dataset.label_uncertainty_buffer_bps
    frame = frame.with_columns(
        [
            pl.when(
                (pl.col("long_net_bps") > buffer)
                & (pl.col("long_net_bps") >= pl.col("short_net_bps"))
            )
            .then(pl.lit(1))
            .when(pl.col("short_net_bps") > buffer)
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("label"),
            (10_000.0 * (pl.col("trade_close") / pl.col("trade_close").shift(horizon)).log()).alias(
                "trailing_return_4_bps"
            ),
        ]
    )
    baseline_hurdle = fee_bps + buffer
    frame = frame.with_columns(
        pl.when(pl.col("trailing_return_4_bps") > baseline_hurdle)
        .then(pl.lit(1))
        .when(pl.col("trailing_return_4_bps") < -baseline_hurdle)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("momentum_label")
    )

    return_expressions = [
        (10_000.0 * (pl.col("trade_close") / pl.col("trade_close").shift(lag)).log()).alias(
            f"return_{lag}_bps"
        )
        for lag in (1, 2, 4, 8, 16, 32, 96)
    ]
    frame = frame.with_columns(
        return_expressions
        + [
            (10_000.0 * (pl.col("trade_high") - pl.col("trade_low")) / pl.col("trade_close")).alias(
                "range_bps"
            ),
            (
                10_000.0 * (pl.col("trade_close") - pl.col("trade_open")) / pl.col("trade_open")
            ).alias("body_bps"),
            (
                (pl.col("trade_close") - pl.col("trade_low"))
                / (pl.col("trade_high") - pl.col("trade_low")).clip(lower_bound=1e-12)
            ).alias("close_location"),
            pl.col("trade_candle_volume").log1p().alias("trade_candle_volume_log"),
            (10_000.0 * (pl.col("mark_close") / pl.col("spot_close")).log()).alias(
                "mark_spot_basis_bps"
            ),
            (10_000.0 * (pl.col("trade_close") / pl.col("mark_close")).log()).alias(
                "trade_mark_basis_bps"
            ),
            pl.col("oi_close").log1p().alias("oi_log"),
            (10_000.0 * (pl.col("oi_close") / pl.col("oi_close").shift(1)).log()).alias(
                "oi_change_1_bps"
            ),
            (10_000.0 * (pl.col("oi_close") / pl.col("oi_close").shift(4)).log()).alias(
                "oi_change_4_bps"
            ),
            (10_000.0 * (pl.col("oi_close") / pl.col("oi_close").shift(16)).log()).alias(
                "oi_change_16_bps"
            ),
            pl.col("future_basis").diff().alias("future_basis_delta_1"),
            pl.col("trade_volume").log1p().alias("trade_volume_log"),
            pl.col("trade_count").log1p().alias("trade_count_log"),
            # Kraken's archived CVD level restarts at every archive-object
            # boundary. Derive the mathematically equivalent per-bar flow from
            # the underlying aggressive volumes so those restarts cannot
            # create artificial feature spikes.
            (pl.col("buy_volume") - pl.col("sell_volume")).alias("cvd_delta_1"),
            (pl.col("buy_volume") - pl.col("sell_volume"))
            .rolling_sum(window_size=4)
            .alias("cvd_delta_4"),
            (
                (pl.col("buy_volume") - pl.col("sell_volume"))
                / (pl.col("buy_volume") + pl.col("sell_volume")).clip(lower_bound=1e-12)
            ).alias("buy_sell_imbalance"),
            pl.col("liquidation_volume").log1p().alias("liquidation_log"),
            ((2.0 * math.pi * pl.col("bucket_start").dt.hour() / 24.0).sin()).alias("hour_sin"),
            ((2.0 * math.pi * pl.col("bucket_start").dt.hour() / 24.0).cos()).alias("hour_cos"),
            ((2.0 * math.pi * pl.col("bucket_start").dt.weekday() / 7.0).sin()).alias(
                "weekday_sin"
            ),
            ((2.0 * math.pi * pl.col("bucket_start").dt.weekday() / 7.0).cos()).alias(
                "weekday_cos"
            ),
        ]
    )

    frame = frame.with_columns(
        [
            (pl.col("return_1_bps").rolling_std(window_size=window) * math.sqrt(window)).alias(
                f"volatility_{window}_bps"
            )
            for window in (4, 16, 96)
        ]
        + [
            _zscore("trade_candle_volume_log", 16, "trade_candle_volume_z16"),
            _zscore("trade_candle_volume_log", 96, "trade_candle_volume_z96"),
            _zscore("future_basis", 96, "future_basis_z96"),
            _zscore("oi_log", 96, "oi_z96"),
            _zscore("aggressor_differential", 16, "aggressor_z16"),
            _zscore("aggressor_differential", 96, "aggressor_z96"),
            _zscore("trade_volume_log", 16, "trade_volume_z16"),
            _zscore("trade_count_log", 16, "trade_count_z16"),
            _zscore("cvd_delta_1", 96, "cvd_delta_z96"),
            _zscore("liquidation_log", 96, "liquidation_z96"),
        ]
    )

    liquidity_expressions: list[pl.Expr] = []
    for depth in ("005", "01", "025", "05", "10"):
        ask = f"ask_liquidity_{depth}"
        bid = f"bid_liquidity_{depth}"
        liquidity_expressions.extend(
            [
                (pl.col(ask) + pl.col(bid)).log1p().alias(f"liquidity_{depth}_log"),
                _imbalance(bid, ask, f"liquidity_{depth}_imbalance"),
            ]
        )
    frame = frame.with_columns(liquidity_expressions)

    maximum_lookback = 96
    frame = frame.slice(maximum_lookback)
    frame = frame.filter(
        pl.col("label_exit_at").is_not_null()
        & pl.col("long_net_bps").is_not_null()
        & pl.col("short_net_bps").is_not_null()
    )
    selected = [
        "bucket_start",
        *TARGET_COLUMNS,
        *FULL_FEATURES,
    ]
    result = frame.select(selected)
    validate_feature_frame(result, config)
    return result


def validate_feature_frame(frame: pl.DataFrame, config: BenchmarkConfig) -> None:
    if not frame.height:
        raise RuntimeError("feature engineering produced no rows")
    if frame["bucket_start"].n_unique() != frame.height:
        raise RuntimeError("feature frame contains duplicate timestamps")
    expected_interval = config.dataset.interval_seconds
    deltas = frame.select(pl.col("bucket_start").diff().dt.total_seconds()).drop_nulls()
    unexpected = deltas.filter(pl.col("bucket_start") != expected_interval).height
    if unexpected:
        raise RuntimeError(f"feature frame contains {unexpected} non-contiguous intervals")
    expected_first = config.dataset.start + timedelta(seconds=96 * expected_interval)
    expected_last = config.dataset.end - timedelta(
        seconds=(config.dataset.horizon_bars + 2) * expected_interval
    )
    expected_rows = int((expected_last - expected_first).total_seconds() // expected_interval) + 1
    if frame.height != expected_rows:
        raise RuntimeError(
            f"feature row count {frame.height} does not match expected {expected_rows}"
        )
    if frame["bucket_start"].min() != expected_first:
        raise RuntimeError("feature frame does not begin after the complete lookback")
    if frame["bucket_start"].max() != expected_last:
        raise RuntimeError("feature frame does not end at the final complete target")
    if frame.select(pl.col("label").is_null().any()).item():
        raise RuntimeError("feature frame contains null labels")
    nulls = frame.select(pl.col(FULL_FEATURES).null_count()).row(0, named=True)
    missing = {column: count for column, count in nulls.items() if count}
    if missing:
        raise RuntimeError(f"feature columns contain nulls: {missing}")
    classes = set(frame["label"].unique().to_list())
    if classes != {-1, 0, 1}:
        raise RuntimeError(f"expected three target classes, received {classes}")
    invalid_availability = frame.filter(pl.col("feature_available_at") > pl.col("entry_at")).height
    if invalid_availability:
        raise RuntimeError("feature availability occurs after entry")
    expected_exit_seconds = config.dataset.interval_seconds * config.dataset.horizon_bars
    invalid_horizon = frame.filter(
        (pl.col("label_exit_at") - pl.col("entry_at")).dt.total_seconds() != expected_exit_seconds
    ).height
    if invalid_horizon:
        raise RuntimeError("label horizon does not match configuration")


def prepare_feature_snapshot(
    config: BenchmarkConfig, raw_snapshot: Snapshot, *, refresh: bool = False
) -> FeatureSnapshot:
    feature_root = config.artifacts.root / "features"
    identity = hashlib.sha256(
        (f"{raw_snapshot.sha256}:{config.fingerprint}:features-v{FEATURE_SCHEMA_VERSION}").encode()
    ).hexdigest()
    index_path = feature_root / f"index-{identity}.json"
    if index_path.exists() and not refresh:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        path = Path(payload["path"])
        manifest_path = Path(payload["manifest_path"])
        if path.exists() and manifest_path.exists():
            sha256 = _sha256(path)
            if sha256 != payload["sha256"]:
                raise RuntimeError(f"feature snapshot checksum mismatch: {path}")
            cached = pl.read_parquet(path)
            validate_feature_frame(cached, config)
            return FeatureSnapshot(
                path=path,
                sha256=sha256,
                row_count=int(payload["row_count"]),
                manifest_path=manifest_path,
            )

    raw = pl.read_parquet(raw_snapshot.path)
    features = build_feature_frame(raw, config)
    staging = feature_root / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / f"{identity}-{os.getpid()}.parquet"
    features.write_parquet(
        temporary,
        compression="zstd",
        compression_level=9,
        statistics=True,
    )
    sha256 = _sha256(temporary)
    final_path = feature_root / "objects" / f"{sha256}.parquet"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        if _sha256(final_path) != sha256:
            raise RuntimeError(f"feature object hash mismatch: {final_path}")
        temporary.unlink()
    else:
        os.replace(temporary, final_path)

    class_counts = {
        str(row["label"]): row["len"]
        for row in features.group_by("label").len().sort("label").iter_rows(named=True)
    }
    manifest_path = feature_root / "manifests" / f"{sha256}-{identity[:16]}.json"
    manifest: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "raw_snapshot_sha256": raw_snapshot.sha256,
        "feature_sha256": sha256,
        "config_fingerprint": config.fingerprint,
        "path": str(final_path),
        "row_count": features.height,
        "first_timestamp": features.item(0, "bucket_start").isoformat(),
        "last_timestamp": features.item(features.height - 1, "bucket_start").isoformat(),
        "feature_sets": FEATURE_SETS,
        "class_counts": class_counts,
        "null_counts": features.null_count().row(0, named=True),
    }
    _immutable_json(manifest_path, manifest)
    _atomic_json(
        index_path,
        {
            "path": str(final_path),
            "manifest_path": str(manifest_path),
            "sha256": sha256,
            "row_count": features.height,
        },
    )
    return FeatureSnapshot(
        path=final_path,
        sha256=sha256,
        row_count=features.height,
        manifest_path=manifest_path,
    )
