from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from .core_config import CoreTrainingConfig
from .core_extract import file_sha256, load_core_manifest, write_json_atomic

PREOPEN_FEATURE_SCHEMA_VERSION = "btc-5m-preopen-features-v1"
PREOPEN_STATIC_FEATURES = [
    "preopen_return_5m_bps",
    "preopen_return_10m_bps",
    "preopen_return_15m_bps",
    "preopen_range_5m_bps",
    "preopen_realized_volatility_5m_bps",
    "preopen_log_quote_volume_5m",
    "preopen_log_quote_volume_15m",
    "preopen_log_trade_count_5m",
    "preopen_log_trade_count_15m",
    "preopen_taker_buy_share_5m",
    "preopen_taker_buy_share_15m",
    "preopen_signed_flow_5m",
    "preopen_signed_flow_15m",
    "preopen_momentum_agreement_5_15",
    "preopen_momentum_acceleration_5_vs_15",
]
PREOPEN_INTERACTION_FEATURES = [
    "btc_path_preopen_5m_agreement",
    "btc_path_preopen_15m_agreement",
    "btc_path_preopen_5m_reversal",
    "btc_path_preopen_15m_reversal",
]
PREOPEN_MODEL_FEATURES = PREOPEN_STATIC_FEATURES + PREOPEN_INTERACTION_FEATURES


def build_preopen_features(
    config: CoreTrainingConfig,
    destination: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    manifest = load_core_manifest(config, "pre_holdout")
    source_manifest = config.paths.source_data / "manifest-pre_holdout.json"
    metadata_path = destination.with_suffix(".metadata.json")
    contract = {
        "feature_schema_version": PREOPEN_FEATURE_SCHEMA_VERSION,
        "feature_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_manifest_sha256": file_sha256(source_manifest),
        "point_in_time_rule": (
            "current window t=-1 close plus aggregates from fully completed prior windows"
        ),
        "feature_names": PREOPEN_STATIC_FEATURES,
    }
    if destination.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("build_contract") != contract:
            raise RuntimeError("pre-open feature cache contract changed")
        if metadata.get("feature_file_sha256") != file_sha256(destination):
            raise RuntimeError("pre-open feature cache hash mismatch")
        return metadata

    source_files = [
        config.paths.source_data / partition["path"]
        for partition in manifest["partitions"]
    ]
    rows = (
        pl.scan_parquet(source_files)
        .select(
            "market_id",
            "window_start",
            "seconds_elapsed",
            "btc_high",
            "btc_low",
            "btc_close",
            "btc_quote_volume",
            "trade_count",
            "btc_taker_buy_quote_volume",
        )
        .sort(["market_id", "seconds_elapsed"])
        .with_columns(
            pl.col("btc_close")
            .log()
            .diff()
            .over("market_id")
            .alias("btc_log_return_1s")
        )
        .collect()
    )
    summaries = (
        rows.group_by("market_id", "window_start")
        .agg(
            pl.first("btc_close").alias("open_available_close"),
            pl.last("btc_close").alias("last_close"),
            pl.max("btc_high").alias("window_high"),
            pl.min("btc_low").alias("window_low"),
            pl.sum("btc_quote_volume").alias("window_quote_volume"),
            pl.sum("trade_count").cast(pl.Float64).alias("window_trade_count"),
            pl.sum("btc_taker_buy_quote_volume").alias(
                "window_taker_buy_quote_volume"
            ),
            pl.std("btc_log_return_1s").mul(10_000).alias("window_volatility_bps"),
            pl.len().alias("source_rows"),
            pl.n_unique("seconds_elapsed").alias("unique_seconds"),
        )
        .filter(
            (pl.col("source_rows") == 300)
            & (pl.col("unique_seconds") == 300)
        )
        .sort("window_start")
    )
    summaries = _derive_preopen_static_features(summaries).select(
        "market_id",
        "window_start",
        *PREOPEN_STATIC_FEATURES,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.partial")
    summaries.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(destination)
    metadata = {
        "build_contract": contract,
        "markets": summaries.height,
        "minimum_window_start": summaries["window_start"].min().isoformat(),
        "maximum_window_start": summaries["window_start"].max().isoformat(),
        "complete_feature_markets": summaries.drop_nulls(
            PREOPEN_STATIC_FEATURES
        ).height,
        "feature_file_sha256": file_sha256(destination),
    }
    write_json_atomic(metadata_path, metadata)
    return metadata


def join_preopen_features(
    core_frame: pl.DataFrame,
    preopen_frame: pl.DataFrame,
) -> pl.DataFrame:
    joined = core_frame.join(
        preopen_frame,
        on=["market_id", "window_start"],
        how="left",
        validate="m:1",
    )
    return joined.with_columns(
        (
            pl.col("btc_path_from_window_open_bps").sign()
            * pl.col("preopen_return_5m_bps").sign()
        ).alias("btc_path_preopen_5m_agreement"),
        (
            pl.col("btc_path_from_window_open_bps").sign()
            * pl.col("preopen_return_15m_bps").sign()
        ).alias("btc_path_preopen_15m_agreement"),
        (
            pl.col("btc_path_from_window_open_bps")
            * pl.col("preopen_return_5m_bps")
            < 0
        )
        .cast(pl.Float64)
        .alias("btc_path_preopen_5m_reversal"),
        (
            pl.col("btc_path_from_window_open_bps")
            * pl.col("preopen_return_15m_bps")
            < 0
        )
        .cast(pl.Float64)
        .alias("btc_path_preopen_15m_reversal"),
    )


def _derive_preopen_static_features(frame: pl.DataFrame) -> pl.DataFrame:
    frame = frame.with_columns(
        *[
            pl.col("window_start").shift(offset).alias(
                f"window_start_lag_{offset}"
            )
            for offset in (1, 2, 3)
        ],
        *[
            pl.col("open_available_close").shift(offset).alias(
                f"open_close_lag_{offset}"
            )
            for offset in (1, 2, 3)
        ],
        pl.col("window_high").shift(1).alias("window_high_lag_1"),
        pl.col("window_low").shift(1).alias("window_low_lag_1"),
        pl.col("last_close").shift(1).alias("last_close_lag_1"),
        pl.col("window_volatility_bps")
        .shift(1)
        .alias("window_volatility_lag_1"),
        *[
            pl.col(column).shift(offset).alias(f"{column}_lag_{offset}")
            for column in (
                "window_quote_volume",
                "window_trade_count",
                "window_taker_buy_quote_volume",
            )
            for offset in (1, 2, 3)
        ],
    )
    contiguous = {
        offset: (
            pl.col("window_start")
            - pl.col(f"window_start_lag_{offset}")
            == pl.duration(minutes=5 * offset)
        )
        for offset in (1, 2, 3)
    }
    quote_volume_15m = pl.sum_horizontal(
        [pl.col(f"window_quote_volume_lag_{offset}") for offset in (1, 2, 3)]
    )
    trade_count_15m = pl.sum_horizontal(
        [pl.col(f"window_trade_count_lag_{offset}") for offset in (1, 2, 3)]
    )
    taker_quote_15m = pl.sum_horizontal(
        [
            pl.col(f"window_taker_buy_quote_volume_lag_{offset}")
            for offset in (1, 2, 3)
        ]
    )
    frame = frame.with_columns(
        pl.when(contiguous[1])
        .then(
            (pl.col("open_available_close") / pl.col("open_close_lag_1"))
            .log()
            .mul(10_000)
        )
        .alias("preopen_return_5m_bps"),
        pl.when(contiguous[2])
        .then(
            (pl.col("open_available_close") / pl.col("open_close_lag_2"))
            .log()
            .mul(10_000)
        )
        .alias("preopen_return_10m_bps"),
        pl.when(contiguous[3])
        .then(
            (pl.col("open_available_close") / pl.col("open_close_lag_3"))
            .log()
            .mul(10_000)
        )
        .alias("preopen_return_15m_bps"),
        pl.when(contiguous[1])
        .then(
            (pl.col("window_high_lag_1") - pl.col("window_low_lag_1"))
            / pl.col("last_close_lag_1")
            * 10_000
        )
        .alias("preopen_range_5m_bps"),
        pl.when(contiguous[1])
        .then(pl.col("window_volatility_lag_1"))
        .alias("preopen_realized_volatility_5m_bps"),
        pl.when(contiguous[1])
        .then(pl.col("window_quote_volume_lag_1").log1p())
        .alias("preopen_log_quote_volume_5m"),
        pl.when(contiguous[3])
        .then(quote_volume_15m.log1p())
        .alias("preopen_log_quote_volume_15m"),
        pl.when(contiguous[1])
        .then(pl.col("window_trade_count_lag_1").log1p())
        .alias("preopen_log_trade_count_5m"),
        pl.when(contiguous[3])
        .then(trade_count_15m.log1p())
        .alias("preopen_log_trade_count_15m"),
        pl.when(contiguous[1])
        .then(
            pl.col("window_taker_buy_quote_volume_lag_1")
            / (pl.col("window_quote_volume_lag_1") + 1e-9)
        )
        .alias("preopen_taker_buy_share_5m"),
        pl.when(contiguous[3])
        .then(taker_quote_15m / (quote_volume_15m + 1e-9))
        .alias("preopen_taker_buy_share_15m"),
        pl.when(contiguous[1])
        .then(
            (
                2 * pl.col("window_taker_buy_quote_volume_lag_1")
                - pl.col("window_quote_volume_lag_1")
            )
            / (pl.col("window_quote_volume_lag_1") + 1e-9)
        )
        .alias("preopen_signed_flow_5m"),
        pl.when(contiguous[3])
        .then((2 * taker_quote_15m - quote_volume_15m) / (quote_volume_15m + 1e-9))
        .alias("preopen_signed_flow_15m"),
    )
    return frame.with_columns(
        (
            pl.col("preopen_return_5m_bps").sign()
            * pl.col("preopen_return_15m_bps").sign()
        ).alias("preopen_momentum_agreement_5_15"),
        (
            pl.col("preopen_return_5m_bps")
            - pl.col("preopen_return_15m_bps") / 3.0
        ).alias("preopen_momentum_acceleration_5_vs_15"),
    )
