from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import polars as pl

from .core_config import CoreTrainingConfig
from .core_extract import (
    file_sha256,
    load_core_manifest,
    scope_range,
    write_json_atomic,
)

CoreFeatureScope = Literal["pre_holdout", "holdout"]
CORE_FEATURE_SCHEMA_VERSION = "btc-5m-directional-core-features-v2"
CORE_BOUNDARY_FEATURE_SCHEMA_VERSION = "btc-5m-directional-boundary-features-v1"

CORE_BASELINE_FEATURES = [
    "seconds_elapsed_scaled",
    "seconds_remaining_scaled",
    "btc_path_from_window_open_bps",
    "btc_return_1s_bps",
    "btc_return_5s_bps",
    "btc_return_15s_bps",
    "btc_return_30s_bps",
    "btc_return_60s_bps",
    "btc_realized_volatility_5s_bps",
    "btc_realized_volatility_15s_bps",
    "btc_realized_volatility_30s_bps",
    "btc_realized_volatility_60s_bps",
    "btc_range_5s_bps",
    "btc_range_30s_bps",
    "btc_range_60s_bps",
    "btc_path_efficiency_30s",
    "btc_path_efficiency_60s",
    "btc_range_position_30s",
    "btc_volatility_expansion_5_to_30",
    "btc_log_quote_volume_5s",
    "btc_log_quote_volume_30s",
    "btc_log_quote_volume_60s",
    "btc_log_trade_count_5s",
    "btc_log_trade_count_30s",
    "btc_taker_buy_share_5s",
    "btc_taker_buy_share_30s",
    "btc_signed_flow_5s",
    "btc_signed_flow_30s",
    "btc_volume_surprise_5_to_60",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

CORE_ENRICHMENT_FEATURES = [
    "btc_path_terminal_volatility_z",
    "btc_path_abs_terminal_volatility_z",
    "btc_path_cross_count",
    "btc_seconds_since_path_cross",
    "btc_fraction_time_path_positive",
    "btc_fraction_time_path_negative",
    "btc_momentum_agreement_5_15",
    "btc_momentum_agreement_15_30",
    "btc_momentum_multihorizon_score",
    "btc_momentum_acceleration_5_vs_30",
    "btc_momentum_acceleration_15_vs_60",
    "btc_reversal_5_vs_30",
    "btc_range_position_60s",
    "btc_distance_from_high_30s_bps",
    "btc_distance_from_low_30s_bps",
    "btc_distance_from_high_60s_bps",
    "btc_distance_from_low_60s_bps",
    "btc_volatility_regime_60_vs_elapsed",
    "btc_log_trade_count_60s",
    "btc_taker_buy_share_60s",
    "btc_signed_flow_60s",
    "btc_flow_persistence_5_30",
    "btc_flow_persistence_30_60",
    "btc_price_flow_agreement_30s",
    "btc_price_flow_divergence_30s",
]

CORE_ENRICHED_FEATURES = CORE_BASELINE_FEATURES + CORE_ENRICHMENT_FEATURES
CORE_BOUNDARY_FEATURES = [
    "btc_cross_venue_boundary_gap_bps",
    "btc_window_open_cross_venue_basis_bps",
    "btc_boundary_terminal_volatility_z",
    "btc_boundary_abs_terminal_volatility_z",
    "btc_boundary_cross_count",
    "btc_seconds_since_boundary_cross",
    "btc_fraction_time_boundary_positive",
    "btc_fraction_time_boundary_negative",
    "btc_boundary_distance_velocity_5s_bps",
    "btc_boundary_momentum_alignment_5s",
]
CORE_BOUNDARY_ENRICHED_FEATURES = CORE_ENRICHED_FEATURES + CORE_BOUNDARY_FEATURES
CORE_MODEL_FEATURES = {
    "logistic_baseline": CORE_BASELINE_FEATURES,
    "logistic_enriched": CORE_ENRICHED_FEATURES,
    "histogram_enriched": CORE_ENRICHED_FEATURES,
    "histogram_boundary_enriched": CORE_BOUNDARY_ENRICHED_FEATURES,
}


def build_core_features(
    config: CoreTrainingConfig,
    scope: CoreFeatureScope,
    *,
    force: bool = False,
) -> dict[str, Any]:
    manifest = load_core_manifest(config, scope)
    destination = feature_destination(config, scope)
    metadata_path = destination.with_suffix(".metadata.json")
    manifest_path = config.paths.source_data / f"manifest-{scope}.json"
    build_contract = feature_build_contract(config, scope, manifest_path)
    if destination.exists() and metadata_path.exists() and not force:
        metadata = validate_core_feature_cache(config, scope)
        print(
            f"core features: reuse {destination.relative_to(config.package_root)}",
            flush=True,
        )
        return metadata

    source_files = [
        config.paths.source_data / partition["path"]
        for partition in manifest["partitions"]
    ]
    frame = (
        pl.scan_parquet(source_files)
        .sort(["market_id", "seconds_elapsed"])
        .collect()
    )
    source_rows = frame.height
    source_markets = frame["market_id"].n_unique()
    frame = derive_core_point_in_time_features(frame)
    final_audit = audit_final_prices(frame)
    mismatch_ids = final_audit.filter(
        pl.col("has_final_price") & ~pl.col("final_price_matches_official")
    )["market_id"]
    mismatch_markets = len(mismatch_ids)
    if mismatch_markets:
        frame = frame.filter(~pl.col("market_id").is_in(mismatch_ids.implode()))

    maximum = 300 - config.data.min_seconds_before_close
    minimum = config.data.min_seconds_after_open
    cadence = config.data.sample_interval_seconds
    history = (
        frame.filter(pl.col("seconds_elapsed").is_between(0, maximum, closed="both"))
        .group_by("market_id")
        .agg(
            pl.len().alias("history_rows"),
            pl.col("seconds_elapsed").n_unique().alias("history_unique_seconds"),
            pl.col("seconds_elapsed").min().alias("history_min_second"),
            pl.col("seconds_elapsed").max().alias("history_max_second"),
        )
    )
    history_complete = history.filter(
        (pl.col("history_rows") == maximum + 1)
        & (pl.col("history_unique_seconds") == maximum + 1)
        & (pl.col("history_min_second") == 0)
        & (pl.col("history_max_second") == maximum)
    ).select("market_id")
    candidates = frame.filter(
        pl.col("seconds_elapsed").is_between(minimum, maximum, closed="both")
        & ((pl.col("seconds_elapsed") - minimum) % cadence == 0)
    ).join(history_complete, on="market_id", how="inner")
    expected_rows = ((maximum - minimum) // cadence) + 1
    complete_candidates = (
        candidates.group_by("market_id")
        .agg(
            pl.len().alias("candidate_rows"),
            pl.col("seconds_elapsed").n_unique().alias("candidate_unique_seconds"),
        )
        .filter(
            (pl.col("candidate_rows") == expected_rows)
            & (pl.col("candidate_unique_seconds") == expected_rows)
        )
        .select("market_id")
    )
    candidates = (
        candidates.join(complete_candidates, on="market_id", how="inner")
        .drop(
            "official_outcome",
            "final_price",
            "btc_path_positive",
            "btc_path_crossed",
            "btc_last_path_cross_second",
            "btc_boundary_positive",
            "btc_boundary_crossed",
            "btc_last_boundary_cross_second",
            strict=False,
        )
        .sort(["window_start", "seconds_elapsed"])
    )
    validate_feature_allowlists(candidates)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.partial")
    candidates.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )
    temporary.replace(destination)
    range_start, range_end = scope_range(config, scope)
    audited_with_final = final_audit.filter(pl.col("has_final_price")).height
    audited_matching = final_audit.filter(pl.col("final_price_matches_official")).height
    daily_counts = (
        candidates.select("market_id", "window_start", "label_up")
        .unique(subset=["market_id"])
        .with_columns(pl.col("window_start").dt.date().cast(pl.String).alias("date"))
        .group_by("date")
        .agg(
            pl.len().alias("markets"),
            pl.col("label_up").sum().alias("up_markets"),
        )
        .with_columns((pl.col("markets") - pl.col("up_markets")).alias("down_markets"))
        .sort("date")
        .to_dicts()
    )
    metadata: dict[str, Any] = {
        "build_contract": build_contract,
        "feature_schema_version": CORE_FEATURE_SCHEMA_VERSION,
        "candidate_feature_schema_versions": {
            "histogram_boundary_enriched": CORE_BOUNDARY_FEATURE_SCHEMA_VERSION,
        },
        "scope": scope,
        "range_start": range_start.isoformat(),
        "range_end": range_end.isoformat(),
        "source_partitions": len(source_files),
        "source_rows": source_rows,
        "source_markets": source_markets,
        "history_complete_markets": history_complete.height,
        "history_incomplete_markets": history.height - history_complete.height,
        "final_price_mismatch_markets": mismatch_markets,
        "candidate_rows": candidates.height,
        "candidate_markets": candidates["market_id"].n_unique(),
        "expected_candidate_rows_per_market": expected_rows,
        "class_up_markets": candidates.filter(pl.col("label_up") == 1)[
            "market_id"
        ].n_unique(),
        "class_down_markets": candidates.filter(pl.col("label_up") == 0)[
            "market_id"
        ].n_unique(),
        "markets_with_final_price": audited_with_final,
        "matching_final_price_labels": audited_matching,
        "daily_counts": daily_counts,
        "feature_groups": CORE_MODEL_FEATURES,
        "feature_file_sha256": file_sha256(destination),
    }
    write_json_atomic(metadata_path, metadata)
    print(
        f"core features: wrote {candidates.height:,} rows across "
        f"{metadata['candidate_markets']:,} markets ({scope})",
        flush=True,
    )
    return metadata


def derive_core_point_in_time_features(frame: pl.DataFrame) -> pl.DataFrame:
    frame = frame.with_columns(
        pl.col("btc_close").log().alias("btc_log_close"),
        pl.first("btc_close").over("market_id").alias("btc_window_open_close"),
    ).with_columns(
        (pl.col("btc_close") / pl.col("opening_boundary"))
        .log()
        .mul(10_000)
        .alias("btc_cross_venue_boundary_gap_bps"),
        (pl.col("btc_window_open_close") / pl.col("opening_boundary"))
        .log()
        .mul(10_000)
        .alias("btc_window_open_cross_venue_basis_bps"),
        (pl.col("btc_close") / pl.col("btc_window_open_close"))
        .log()
        .mul(10_000)
        .alias("btc_path_from_window_open_bps"),
        (pl.col("seconds_elapsed") / 300.0).alias("seconds_elapsed_scaled"),
        ((300 - pl.col("seconds_elapsed")) / 300.0).alias(
            "seconds_remaining_scaled"
        ),
    ).with_columns(
        (pl.col("btc_path_from_window_open_bps") >= 0).alias("btc_path_positive"),
        (pl.col("btc_cross_venue_boundary_gap_bps") >= 0).alias(
            "btc_boundary_positive"
        ),
    )
    frame = frame.with_columns(
        (
            pl.col("btc_path_positive")
            != pl.col("btc_path_positive").shift(1).over("market_id")
        )
        .fill_null(False)
        .alias("btc_path_crossed"),
        (
            pl.col("btc_boundary_positive")
            != pl.col("btc_boundary_positive").shift(1).over("market_id")
        )
        .fill_null(False)
        .alias("btc_boundary_crossed"),
    )
    for seconds in (1, 5, 15, 30, 60):
        frame = frame.with_columns(
            (
                pl.col("btc_log_close")
                - pl.col("btc_log_close").shift(seconds).over("market_id")
            )
            .mul(10_000)
            .alias(f"btc_return_{seconds}s_bps")
        )
    frame = frame.with_columns(
        (
            pl.col("btc_log_close")
            - pl.col("btc_log_close").shift(1).over("market_id")
        ).alias("btc_log_return_1s")
    )
    for seconds in (5, 15, 30, 60):
        minimum_samples = max(2, seconds // 2)
        frame = frame.with_columns(
            pl.col("btc_log_return_1s")
            .rolling_std(window_size=seconds, min_samples=minimum_samples)
            .over("market_id")
            .mul(10_000)
            .alias(f"btc_realized_volatility_{seconds}s_bps"),
            (
                pl.col("btc_high")
                .rolling_max(window_size=seconds, min_samples=minimum_samples)
                .over("market_id")
                - pl.col("btc_low")
                .rolling_min(window_size=seconds, min_samples=minimum_samples)
                .over("market_id")
            )
            .truediv(pl.col("btc_close"))
            .mul(10_000)
            .alias(f"btc_range_{seconds}s_bps"),
            pl.col("btc_quote_volume")
            .rolling_sum(window_size=seconds, min_samples=max(1, seconds // 2))
            .over("market_id")
            .alias(f"btc_quote_volume_{seconds}s"),
            pl.col("trade_count")
            .rolling_sum(window_size=seconds, min_samples=max(1, seconds // 2))
            .over("market_id")
            .alias(f"btc_trade_count_{seconds}s"),
            pl.col("btc_taker_buy_quote_volume")
            .rolling_sum(window_size=seconds, min_samples=max(1, seconds // 2))
            .over("market_id")
            .alias(f"btc_taker_buy_quote_volume_{seconds}s"),
        )
    frame = frame.with_columns(
        (
            pl.col("btc_return_30s_bps").abs()
            / (
                pl.col("btc_log_return_1s")
                .abs()
                .rolling_sum(window_size=30, min_samples=15)
                .over("market_id")
                * 10_000
                + 1e-9
            )
        ).alias("btc_path_efficiency_30s"),
        (
            pl.col("btc_return_60s_bps").abs()
            / (
                pl.col("btc_log_return_1s")
                .abs()
                .rolling_sum(window_size=60, min_samples=30)
                .over("market_id")
                * 10_000
                + 1e-9
            )
        ).alias("btc_path_efficiency_60s"),
        (
            (
                pl.col("btc_close")
                - pl.col("btc_low")
                .rolling_min(window_size=30, min_samples=15)
                .over("market_id")
            )
            / (
                pl.col("btc_high")
                .rolling_max(window_size=30, min_samples=15)
                .over("market_id")
                - pl.col("btc_low")
                .rolling_min(window_size=30, min_samples=15)
                .over("market_id")
                + 1e-9
            )
        ).alias("btc_range_position_30s"),
        (
            (
                pl.col("btc_close")
                - pl.col("btc_low")
                .rolling_min(window_size=60, min_samples=30)
                .over("market_id")
            )
            / (
                pl.col("btc_high")
                .rolling_max(window_size=60, min_samples=30)
                .over("market_id")
                - pl.col("btc_low")
                .rolling_min(window_size=60, min_samples=30)
                .over("market_id")
                + 1e-9
            )
        ).alias("btc_range_position_60s"),
        (
            pl.col("btc_realized_volatility_5s_bps")
            / (pl.col("btc_realized_volatility_30s_bps") + 1e-9)
        ).alias("btc_volatility_expansion_5_to_30"),
    )
    for seconds in (5, 30, 60):
        frame = frame.with_columns(
            pl.col(f"btc_quote_volume_{seconds}s")
            .log1p()
            .alias(f"btc_log_quote_volume_{seconds}s"),
            pl.col(f"btc_trade_count_{seconds}s")
            .cast(pl.Float64)
            .log1p()
            .alias(f"btc_log_trade_count_{seconds}s"),
            (
                pl.col(f"btc_taker_buy_quote_volume_{seconds}s")
                / (pl.col(f"btc_quote_volume_{seconds}s") + 1e-9)
            ).alias(f"btc_taker_buy_share_{seconds}s"),
            (
                2 * pl.col(f"btc_taker_buy_quote_volume_{seconds}s")
                - pl.col(f"btc_quote_volume_{seconds}s")
            )
            .truediv(pl.col(f"btc_quote_volume_{seconds}s") + 1e-9)
            .alias(f"btc_signed_flow_{seconds}s"),
        )
    frame = frame.with_columns(
        pl.when(pl.col("btc_path_crossed"))
        .then(pl.col("seconds_elapsed"))
        .otherwise(None)
        .forward_fill()
        .over("market_id")
        .alias("btc_last_path_cross_second"),
        pl.col("btc_path_crossed")
        .cast(pl.Int32)
        .cum_sum()
        .over("market_id")
        .cast(pl.Float64)
        .alias("btc_path_cross_count"),
        (
            pl.col("btc_path_positive").cast(pl.Int32).cum_sum().over("market_id")
            / (pl.col("seconds_elapsed") + 1)
        ).alias("btc_fraction_time_path_positive"),
        pl.when(pl.col("btc_boundary_crossed"))
        .then(pl.col("seconds_elapsed"))
        .otherwise(None)
        .forward_fill()
        .over("market_id")
        .alias("btc_last_boundary_cross_second"),
        pl.col("btc_boundary_crossed")
        .cast(pl.Int32)
        .cum_sum()
        .over("market_id")
        .cast(pl.Float64)
        .alias("btc_boundary_cross_count"),
        (
            pl.col("btc_boundary_positive")
            .cast(pl.Int32)
            .cum_sum()
            .over("market_id")
            / (pl.col("seconds_elapsed") + 1)
        ).alias("btc_fraction_time_boundary_positive"),
    )
    frame = frame.with_columns(
        (
            pl.col("seconds_elapsed")
            - pl.col("btc_last_path_cross_second")
            .fill_null(pl.col("seconds_elapsed"))
        )
        .cast(pl.Float64)
        .alias("btc_seconds_since_path_cross"),
        (1.0 - pl.col("btc_fraction_time_path_positive")).alias(
            "btc_fraction_time_path_negative"
        ),
        (
            pl.col("seconds_elapsed")
            - pl.col("btc_last_boundary_cross_second")
            .fill_null(pl.col("seconds_elapsed"))
        )
        .cast(pl.Float64)
        .alias("btc_seconds_since_boundary_cross"),
        (1.0 - pl.col("btc_fraction_time_boundary_positive")).alias(
            "btc_fraction_time_boundary_negative"
        ),
        (
            pl.col("btc_realized_volatility_60s_bps").cum_sum().over("market_id")
            / pl.col("btc_realized_volatility_60s_bps")
            .is_not_null()
            .cast(pl.Int32)
            .cum_sum()
            .over("market_id")
            .clip(lower_bound=1)
        ).alias("btc_elapsed_volatility_mean"),
    )
    frame = frame.with_columns(
        (
            pl.col("btc_path_from_window_open_bps")
            / (
                pl.col("btc_realized_volatility_60s_bps")
                * (300 - pl.col("seconds_elapsed")).clip(lower_bound=1).sqrt()
                + 1e-9
            )
        ).alias("btc_path_terminal_volatility_z"),
        (
            pl.col("btc_path_from_window_open_bps").abs()
            / (
                pl.col("btc_realized_volatility_60s_bps")
                * (300 - pl.col("seconds_elapsed")).clip(lower_bound=1).sqrt()
                + 1e-9
            )
        ).alias("btc_path_abs_terminal_volatility_z"),
        (
            pl.col("btc_cross_venue_boundary_gap_bps")
            / (
                pl.col("btc_realized_volatility_60s_bps")
                * (300 - pl.col("seconds_elapsed")).clip(lower_bound=1).sqrt()
                + 1e-9
            )
        ).alias("btc_boundary_terminal_volatility_z"),
        (
            pl.col("btc_cross_venue_boundary_gap_bps").abs()
            / (
                pl.col("btc_realized_volatility_60s_bps")
                * (300 - pl.col("seconds_elapsed")).clip(lower_bound=1).sqrt()
                + 1e-9
            )
        ).alias("btc_boundary_abs_terminal_volatility_z"),
        (
            pl.col("btc_cross_venue_boundary_gap_bps").abs()
            - pl.col("btc_cross_venue_boundary_gap_bps")
            .abs()
            .shift(5)
            .over("market_id")
        ).alias("btc_boundary_distance_velocity_5s_bps"),
        (
            pl.col("btc_cross_venue_boundary_gap_bps").sign()
            * pl.col("btc_return_5s_bps").sign()
        ).alias("btc_boundary_momentum_alignment_5s"),
        (
            pl.col("btc_return_5s_bps").sign()
            * pl.col("btc_return_15s_bps").sign()
        ).alias("btc_momentum_agreement_5_15"),
        (
            pl.col("btc_return_15s_bps").sign()
            * pl.col("btc_return_30s_bps").sign()
        ).alias("btc_momentum_agreement_15_30"),
        (
            pl.col("btc_return_5s_bps").sign()
            + pl.col("btc_return_15s_bps").sign()
            + pl.col("btc_return_30s_bps").sign()
            + pl.col("btc_return_60s_bps").sign()
        )
        .truediv(4.0)
        .alias("btc_momentum_multihorizon_score"),
        (
            pl.col("btc_return_5s_bps")
            - pl.col("btc_return_30s_bps") * (5.0 / 30.0)
        ).alias("btc_momentum_acceleration_5_vs_30"),
        (
            pl.col("btc_return_15s_bps")
            - pl.col("btc_return_60s_bps") * (15.0 / 60.0)
        ).alias("btc_momentum_acceleration_15_vs_60"),
        (
            (pl.col("btc_return_5s_bps") * pl.col("btc_return_30s_bps")) < 0
        )
        .cast(pl.Float64)
        .alias("btc_reversal_5_vs_30"),
    )
    frame = frame.with_columns(
        (
            (
                pl.col("btc_high")
                .rolling_max(window_size=30, min_samples=15)
                .over("market_id")
                - pl.col("btc_close")
            )
            / pl.col("btc_close")
            * 10_000
        ).alias("btc_distance_from_high_30s_bps"),
        (
            (
                pl.col("btc_close")
                - pl.col("btc_low")
                .rolling_min(window_size=30, min_samples=15)
                .over("market_id")
            )
            / pl.col("btc_close")
            * 10_000
        ).alias("btc_distance_from_low_30s_bps"),
        (
            (
                pl.col("btc_high")
                .rolling_max(window_size=60, min_samples=30)
                .over("market_id")
                - pl.col("btc_close")
            )
            / pl.col("btc_close")
            * 10_000
        ).alias("btc_distance_from_high_60s_bps"),
        (
            (
                pl.col("btc_close")
                - pl.col("btc_low")
                .rolling_min(window_size=60, min_samples=30)
                .over("market_id")
            )
            / pl.col("btc_close")
            * 10_000
        ).alias("btc_distance_from_low_60s_bps"),
        (
            pl.col("btc_realized_volatility_60s_bps")
            / (pl.col("btc_elapsed_volatility_mean") + 1e-9)
        ).alias("btc_volatility_regime_60_vs_elapsed"),
        (
            pl.col("btc_signed_flow_5s") * pl.col("btc_signed_flow_30s")
        ).alias("btc_flow_persistence_5_30"),
        (
            pl.col("btc_signed_flow_30s") * pl.col("btc_signed_flow_60s")
        ).alias("btc_flow_persistence_30_60"),
        (
            pl.col("btc_return_30s_bps").sign() * pl.col("btc_signed_flow_30s")
        ).alias("btc_price_flow_agreement_30s"),
        (
            pl.col("btc_return_30s_bps").sign() * -pl.col("btc_signed_flow_30s")
        ).alias("btc_price_flow_divergence_30s"),
        (
            pl.col("btc_quote_volume_5s")
            / (pl.col("btc_quote_volume_60s") / 12.0 + 1e-9)
        ).alias("btc_volume_surprise_5_to_60"),
        ((pl.col("observed_at").dt.hour() * 2 * math.pi / 24).sin()).alias(
            "hour_sin"
        ),
        ((pl.col("observed_at").dt.hour() * 2 * math.pi / 24).cos()).alias(
            "hour_cos"
        ),
        ((pl.col("observed_at").dt.weekday() * 2 * math.pi / 7).sin()).alias(
            "weekday_sin"
        ),
        ((pl.col("observed_at").dt.weekday() * 2 * math.pi / 7).cos()).alias(
            "weekday_cos"
        ),
        (pl.col("btc_path_from_window_open_bps") >= 0)
        .cast(pl.Int8)
        .alias("binance_sign_up"),
    )
    return frame


def audit_final_prices(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.group_by("market_id")
        .agg(
            pl.first("label_up").alias("label_up"),
            pl.first("opening_boundary").alias("opening_boundary"),
            pl.first("final_price").alias("final_price"),
        )
        .with_columns(
            pl.col("final_price").is_not_null().alias("has_final_price"),
            pl.when(pl.col("final_price").is_not_null())
            .then((pl.col("final_price") >= pl.col("opening_boundary")).cast(pl.Int32))
            .otherwise(None)
            .alias("audited_label_up"),
        )
        .with_columns(
            (
                pl.col("has_final_price")
                & (pl.col("audited_label_up") == pl.col("label_up"))
            ).alias("final_price_matches_official")
        )
    )


def feature_destination(
    config: CoreTrainingConfig, scope: CoreFeatureScope
) -> Path:
    if scope == "pre_holdout":
        return config.paths.development_feature_data
    if scope == "holdout":
        return config.paths.holdout_feature_data
    raise ValueError(f"unsupported core feature scope: {scope}")


def feature_build_contract(
    config: CoreTrainingConfig,
    scope: CoreFeatureScope,
    source_manifest_path: Path,
) -> dict[str, Any]:
    range_start, range_end = scope_range(config, scope)
    return {
        "feature_schema_version": CORE_FEATURE_SCHEMA_VERSION,
        "feature_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "scope": scope,
        "range_start": range_start.isoformat(),
        "range_end": range_end.isoformat(),
        "sample_interval_seconds": config.data.sample_interval_seconds,
        "minimum_seconds_after_open": config.data.min_seconds_after_open,
        "minimum_seconds_before_close": config.data.min_seconds_before_close,
        "strict_final_price_audit": config.data.strict_final_price_audit,
    }


def validate_core_feature_cache(
    config: CoreTrainingConfig, scope: CoreFeatureScope
) -> dict[str, Any]:
    destination = feature_destination(config, scope)
    metadata_path = destination.with_suffix(".metadata.json")
    if not destination.exists() or not metadata_path.exists():
        raise RuntimeError(f"{scope} feature cache is missing")
    source_manifest_path = config.paths.source_data / f"manifest-{scope}.json"
    load_core_manifest(config, scope)
    expected = feature_build_contract(config, scope, source_manifest_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("build_contract") != expected:
        raise RuntimeError(f"{scope} feature cache contract changed")
    if metadata.get("feature_file_sha256") != file_sha256(destination):
        raise RuntimeError(f"{scope} feature file hash mismatch")
    return metadata


def validate_feature_allowlists(frame: pl.DataFrame) -> None:
    forbidden = {
        "label_up",
        "official_outcome",
        "final_price",
        "window_end",
        "opening_boundary",
    }
    for name, features in CORE_MODEL_FEATURES.items():
        if forbidden.intersection(features):
            raise RuntimeError(f"{name} feature allowlist contains label/audit data")
        missing = [feature for feature in features if feature not in frame.columns]
        if missing:
            raise RuntimeError(f"{name} features are missing: {', '.join(missing)}")


def load_core_feature_frame(
    config: CoreTrainingConfig, scope: CoreFeatureScope
) -> pl.DataFrame:
    validate_core_feature_cache(config, scope)
    return pl.read_parquet(feature_destination(config, scope)).sort(
        ["window_start", "seconds_elapsed"]
    )
