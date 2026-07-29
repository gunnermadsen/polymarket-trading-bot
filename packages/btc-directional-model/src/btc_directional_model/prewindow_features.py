from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from .core_config import CoreTrainingConfig
from .core_extract import file_sha256, load_core_manifest, write_json_atomic

PREWINDOW_FEATURE_SCHEMA_VERSION = "btc-5m-prewindow-regime-v1"
PREWINDOW_HORIZONS_MINUTES = (5, 15, 30, 60)
PREWINDOW_STATIC_FEATURES = [
    feature
    for minutes in PREWINDOW_HORIZONS_MINUTES
    for feature in (
        f"prewindow_return_{minutes}m_bps",
        f"prewindow_range_{minutes}m_bps",
        f"prewindow_realized_volatility_{minutes}m_bps",
        f"prewindow_log_quote_volume_{minutes}m",
        f"prewindow_log_trade_count_{minutes}m",
        f"prewindow_taker_buy_share_{minutes}m",
        f"prewindow_signed_flow_{minutes}m",
    )
] + [
    "prewindow_momentum_agreement_5_15",
    "prewindow_momentum_agreement_5_30",
    "prewindow_momentum_agreement_5_60",
    "prewindow_momentum_acceleration_5_vs_15",
    "prewindow_momentum_acceleration_5_vs_30",
    "prewindow_momentum_acceleration_5_vs_60",
]
PREWINDOW_INTERACTION_FEATURES = [
    feature
    for minutes in PREWINDOW_HORIZONS_MINUTES
    for feature in (
        f"btc_path_prewindow_{minutes}m_agreement",
        f"btc_path_prewindow_{minutes}m_reversal",
    )
]
PREWINDOW_MODEL_FEATURES = PREWINDOW_STATIC_FEATURES + PREWINDOW_INTERACTION_FEATURES


def build_prewindow_features(
    config: CoreTrainingConfig,
    destination: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    manifest = load_core_manifest(config, "pre_holdout")
    source_manifest = config.paths.source_data / "manifest-pre_holdout.json"
    metadata_path = destination.with_suffix(".metadata.json")
    contract = {
        "feature_schema_version": PREWINDOW_FEATURE_SCHEMA_VERSION,
        "feature_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_manifest_sha256": file_sha256(source_manifest),
        "point_in_time_rule": (
            "market-open close and aggregates from fully completed, contiguous "
            "prior five-minute windows only"
        ),
        "horizons_minutes": list(PREWINDOW_HORIZONS_MINUTES),
        "feature_names": PREWINDOW_STATIC_FEATURES,
        "missing_history_policy": "model_ineligible; never directional imputation",
    }
    if destination.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("build_contract") != contract:
            raise RuntimeError("pre-window feature cache contract changed")
        if metadata.get("feature_file_sha256") != file_sha256(destination):
            raise RuntimeError("pre-window feature cache hash mismatch")
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
    features = derive_prewindow_static_features(summaries).select(
        "market_id",
        "window_start",
        *PREWINDOW_STATIC_FEATURES,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.partial")
    features.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(destination)
    complete = features.drop_nulls(PREWINDOW_STATIC_FEATURES)
    metadata = {
        "build_contract": contract,
        "markets": features.height,
        "minimum_window_start": features["window_start"].min().isoformat(),
        "maximum_window_start": features["window_start"].max().isoformat(),
        "complete_feature_markets": complete.height,
        "incomplete_feature_markets": features.height - complete.height,
        "feature_file_sha256": file_sha256(destination),
    }
    write_json_atomic(metadata_path, metadata)
    return metadata


def join_prewindow_features(
    core_frame: pl.DataFrame,
    prewindow_frame: pl.DataFrame,
) -> pl.DataFrame:
    joined = core_frame.join(
        prewindow_frame,
        on=["market_id", "window_start"],
        how="left",
        validate="m:1",
    )
    interactions: list[pl.Expr] = []
    for minutes in PREWINDOW_HORIZONS_MINUTES:
        prior_return = pl.col(f"prewindow_return_{minutes}m_bps")
        current_path = pl.col("btc_path_from_window_open_bps")
        interactions.extend(
            (
                (current_path.sign() * prior_return.sign()).alias(
                    f"btc_path_prewindow_{minutes}m_agreement"
                ),
                (current_path * prior_return < 0)
                .cast(pl.Float64)
                .alias(f"btc_path_prewindow_{minutes}m_reversal"),
            )
        )
    return joined.with_columns(*interactions).with_columns(
        pl.all_horizontal(
            [pl.col(feature).is_not_null() for feature in PREWINDOW_MODEL_FEATURES]
        ).alias("prewindow_model_eligible")
    )


def derive_prewindow_static_features(frame: pl.DataFrame) -> pl.DataFrame:
    maximum_lag = max(PREWINDOW_HORIZONS_MINUTES) // 5
    sorted_frame = frame.sort("window_start")
    lag_columns: list[pl.Expr] = []
    for offset in range(1, maximum_lag + 1):
        lag_columns.extend(
            (
                pl.col("window_start")
                .shift(offset)
                .alias(f"window_start_lag_{offset}"),
                pl.col("window_high").shift(offset).alias(f"window_high_lag_{offset}"),
                pl.col("window_low").shift(offset).alias(f"window_low_lag_{offset}"),
                pl.col("open_available_close")
                .shift(offset)
                .alias(f"open_available_close_lag_{offset}"),
                pl.col("last_close").shift(offset).alias(f"last_close_lag_{offset}"),
                pl.col("window_quote_volume")
                .shift(offset)
                .alias(f"window_quote_volume_lag_{offset}"),
                pl.col("window_trade_count")
                .shift(offset)
                .alias(f"window_trade_count_lag_{offset}"),
                pl.col("window_taker_buy_quote_volume")
                .shift(offset)
                .alias(f"window_taker_buy_quote_volume_lag_{offset}"),
                pl.col("window_volatility_bps")
                .shift(offset)
                .alias(f"window_volatility_bps_lag_{offset}"),
            )
        )
    derived = sorted_frame.with_columns(*lag_columns)
    horizon_expressions: list[pl.Expr] = []
    for minutes in PREWINDOW_HORIZONS_MINUTES:
        windows = minutes // 5
        contiguous = (
            pl.col("window_start") - pl.col(f"window_start_lag_{windows}")
            == pl.duration(minutes=minutes)
        )
        quote_volume = pl.sum_horizontal(
            [pl.col(f"window_quote_volume_lag_{offset}") for offset in range(1, windows + 1)]
        )
        trade_count = pl.sum_horizontal(
            [pl.col(f"window_trade_count_lag_{offset}") for offset in range(1, windows + 1)]
        )
        taker_quote = pl.sum_horizontal(
            [
                pl.col(f"window_taker_buy_quote_volume_lag_{offset}")
                for offset in range(1, windows + 1)
            ]
        )
        volatility = pl.mean_horizontal(
            [
                pl.col(f"window_volatility_bps_lag_{offset}").pow(2)
                for offset in range(1, windows + 1)
            ]
        ).sqrt()
        horizon_expressions.extend(
            (
                pl.when(contiguous)
                .then(
                    (
                        pl.col("open_available_close")
                        / pl.col(f"open_available_close_lag_{windows}")
                    )
                    .log()
                    .mul(10_000)
                )
                .alias(f"prewindow_return_{minutes}m_bps"),
                pl.when(contiguous)
                .then(
                    (
                        pl.max_horizontal(
                            [
                                pl.col(f"window_high_lag_{offset}")
                                for offset in range(1, windows + 1)
                            ]
                        )
                        - pl.min_horizontal(
                            [
                                pl.col(f"window_low_lag_{offset}")
                                for offset in range(1, windows + 1)
                            ]
                        )
                    )
                    / pl.col("open_available_close")
                    * 10_000
                )
                .alias(f"prewindow_range_{minutes}m_bps"),
                pl.when(contiguous)
                .then(volatility)
                .alias(f"prewindow_realized_volatility_{minutes}m_bps"),
                pl.when(contiguous)
                .then(quote_volume.log1p())
                .alias(f"prewindow_log_quote_volume_{minutes}m"),
                pl.when(contiguous)
                .then(trade_count.log1p())
                .alias(f"prewindow_log_trade_count_{minutes}m"),
                pl.when(contiguous)
                .then(taker_quote / (quote_volume + 1e-9))
                .alias(f"prewindow_taker_buy_share_{minutes}m"),
                pl.when(contiguous)
                .then((2 * taker_quote - quote_volume) / (quote_volume + 1e-9))
                .alias(f"prewindow_signed_flow_{minutes}m"),
            )
        )
    derived = derived.with_columns(*horizon_expressions)
    return derived.with_columns(
        *[
            (
                pl.col("prewindow_return_5m_bps").sign()
                * pl.col(f"prewindow_return_{minutes}m_bps").sign()
            ).alias(f"prewindow_momentum_agreement_5_{minutes}")
            for minutes in (15, 30, 60)
        ],
        *[
            (
                pl.col("prewindow_return_5m_bps")
                - pl.col(f"prewindow_return_{minutes}m_bps") / (minutes / 5)
            ).alias(f"prewindow_momentum_acceleration_5_vs_{minutes}")
            for minutes in (15, 30, 60)
        ],
    )
