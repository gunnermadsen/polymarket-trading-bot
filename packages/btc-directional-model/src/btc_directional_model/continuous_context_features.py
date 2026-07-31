from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from .core_config import CoreTrainingConfig
from .core_extract import file_sha256, load_core_manifest, write_json_atomic

CONTINUOUS_CONTEXT_FEATURE_SCHEMA_VERSION = "btc-continuous-binance-context-90-120s-v1"
CONTINUOUS_CONTEXT_HORIZONS_SECONDS = (90, 120)
CONTINUOUS_CONTEXT_MODEL_FEATURES = [
    f"continuous_return_{seconds}s_bps"
    for seconds in CONTINUOUS_CONTEXT_HORIZONS_SECONDS
] + [
    f"continuous_realized_volatility_{seconds}s_bps"
    for seconds in CONTINUOUS_CONTEXT_HORIZONS_SECONDS
] + [
    f"continuous_signed_flow_{seconds}s"
    for seconds in CONTINUOUS_CONTEXT_HORIZONS_SECONDS
]


def build_continuous_context_features(
    config: CoreTrainingConfig,
    destination: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build six causal Binance context features across market boundaries."""

    manifest = load_core_manifest(config, "pre_holdout")
    source_manifest = config.paths.source_data / "manifest-pre_holdout.json"
    metadata_path = destination.with_suffix(".metadata.json")
    contract = {
        "feature_schema_version": CONTINUOUS_CONTEXT_FEATURE_SCHEMA_VERSION,
        "feature_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_manifest_sha256": file_sha256(source_manifest),
        "point_in_time_rule": (
            "only Binance one-second rows at or before observed_at; exact contiguous "
            "90-second and 120-second histories"
        ),
        "horizons_seconds": list(CONTINUOUS_CONTEXT_HORIZONS_SECONDS),
        "feature_names": CONTINUOUS_CONTEXT_MODEL_FEATURES,
        "missing_history_policy": "model_ineligible; never imputed or used as a predictor",
    }
    if destination.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("build_contract") != contract:
            raise RuntimeError("continuous-context feature cache contract changed")
        if metadata.get("feature_file_sha256") != file_sha256(destination):
            raise RuntimeError("continuous-context feature cache hash mismatch")
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
            "observed_at",
            "seconds_elapsed",
            "btc_close",
            "btc_quote_volume",
            "btc_taker_buy_quote_volume",
        )
        .sort("observed_at")
        .collect()
    )
    features = derive_continuous_context_features(rows).select(
        "market_id",
        "window_start",
        "observed_at",
        *CONTINUOUS_CONTEXT_MODEL_FEATURES,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.partial")
    features.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(destination)
    complete = features.drop_nulls(CONTINUOUS_CONTEXT_MODEL_FEATURES)
    metadata = {
        "build_contract": contract,
        "rows": features.height,
        "markets": features["market_id"].n_unique(),
        "minimum_observed_at": features["observed_at"].min().isoformat(),
        "maximum_observed_at": features["observed_at"].max().isoformat(),
        "complete_feature_rows": complete.height,
        "incomplete_feature_rows": features.height - complete.height,
        "feature_file_sha256": file_sha256(destination),
    }
    write_json_atomic(metadata_path, metadata)
    return metadata


def join_continuous_context_features(
    core_frame: pl.DataFrame,
    context_frame: pl.DataFrame,
) -> pl.DataFrame:
    joined = core_frame.join(
        context_frame,
        on=["market_id", "window_start", "observed_at"],
        how="left",
        validate="1:1",
    )
    return joined.with_columns(
        pl.all_horizontal(
            [
                pl.col(feature).is_not_null() & pl.col(feature).is_finite()
                for feature in CONTINUOUS_CONTEXT_MODEL_FEATURES
            ]
        ).alias("continuous_context_model_eligible")
    )


def derive_continuous_context_features(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "btc_close",
        "btc_quote_volume",
        "btc_taker_buy_quote_volume",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("continuous-context source is missing columns: " + ", ".join(missing))
    ordered = frame.sort("observed_at")
    duplicate = ordered.group_by("observed_at").len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise RuntimeError("continuous-context source contains duplicate timestamps")
    ordered = ordered.with_columns(
        pl.col("btc_close").log().diff().alias("_log_return_1s"),
        (2.0 * pl.col("btc_taker_buy_quote_volume") - pl.col("btc_quote_volume")).alias(
            "_signed_quote_volume"
        ),
    )
    expressions: list[pl.Expr] = []
    for seconds in CONTINUOUS_CONTEXT_HORIZONS_SECONDS:
        exact_history = (
            pl.col("observed_at") - pl.col("observed_at").shift(seconds)
            == pl.duration(seconds=seconds)
        )
        expressions.extend(
            (
                pl.when(exact_history)
                .then((pl.col("btc_close") / pl.col("btc_close").shift(seconds)).log() * 10_000)
                .otherwise(None)
                .alias(f"continuous_return_{seconds}s_bps"),
                pl.when(exact_history)
                .then(pl.col("_log_return_1s").rolling_std(window_size=seconds) * 10_000)
                .otherwise(None)
                .alias(f"continuous_realized_volatility_{seconds}s_bps"),
                pl.when(exact_history)
                .then(
                    pl.col("_signed_quote_volume").rolling_sum(window_size=seconds)
                    / (pl.col("btc_quote_volume").rolling_sum(window_size=seconds) + 1e-9)
                )
                .otherwise(None)
                .alias(f"continuous_signed_flow_{seconds}s"),
            )
        )
    return ordered.with_columns(*expressions).drop("_log_return_1s", "_signed_quote_volume")
