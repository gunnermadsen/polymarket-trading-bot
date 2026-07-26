from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from .config import TrainingConfig

FEATURE_SCHEMA_VERSION = "btc-5m-directional-features-v2"

CORE_FEATURES = [
    "seconds_elapsed_scaled",
    "seconds_remaining_scaled",
    "btc_gap_from_open_bps",
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

BOOK_FEATURES = [
    "up_mid",
    "down_mid",
    "up_spread",
    "down_spread",
    "up_ask_vwap_1",
    "down_ask_vwap_1",
    "up_ask_vwap_5",
    "down_ask_vwap_5",
    "up_ask_vwap_10",
    "down_ask_vwap_10",
    "up_vwap_slippage_5",
    "down_vwap_slippage_5",
    "up_imbalance",
    "down_imbalance",
    "up_log_bid_depth",
    "down_log_bid_depth",
    "up_log_ask_depth",
    "down_log_ask_depth",
    "mid_complement_residual",
    "ask_complement_residual",
    "book_mid_difference",
    "up_mid_change_5s",
    "down_mid_change_5s",
    "up_imbalance_change_5s",
    "down_imbalance_change_5s",
    "up_provider_age_ms",
    "down_provider_age_ms",
    "provider_age_skew_ms",
    "quality_up_missing",
    "quality_down_missing",
    "quality_up_stale",
    "quality_down_stale",
    "quality_up_crossed",
    "quality_down_crossed",
    "quality_up_insufficient_depth",
    "quality_down_insufficient_depth",
    "up_book_missing",
    "down_book_missing",
]

FEATURE_GROUPS = {
    "btc_path": CORE_FEATURES,
    "btc_path_and_book": CORE_FEATURES + BOOK_FEATURES,
}


def build_features(config: TrainingConfig, *, force: bool = False) -> dict[str, Any]:
    destination = config.paths.feature_data
    metadata_path = destination.with_suffix(".metadata.json")
    source_manifest_path = config.paths.source_data / "manifest.json"
    source_manifest = load_source_manifest(source_manifest_path, config)
    build_contract = feature_build_contract(config, source_manifest_path)
    if destination.exists() and metadata_path.exists() and not force:
        metadata = validate_feature_cache(config)
        print(f"features: reuse {destination.relative_to(config.package_root)}", flush=True)
        return metadata

    source_files = verified_source_files(config.paths.source_data, source_manifest)
    frame = pl.scan_parquet(source_files).sort(["market_id", "seconds_elapsed"]).collect()
    frame = derive_point_in_time_features(frame)

    minimum = config.data.min_seconds_after_open
    maximum = 300 - config.data.min_seconds_before_close
    cadence = config.data.sample_interval_seconds
    history_complete_markets = (
        frame.filter(pl.col("seconds_elapsed").is_between(0, maximum, closed="both"))
        .group_by("market_id")
        .agg(
            pl.len().alias("history_rows"),
            pl.col("seconds_elapsed").n_unique().alias("history_unique_seconds"),
            pl.col("seconds_elapsed").min().alias("history_min_second"),
            pl.col("seconds_elapsed").max().alias("history_max_second"),
        )
        .filter(
            (pl.col("history_rows") == maximum + 1)
            & (pl.col("history_unique_seconds") == maximum + 1)
            & (pl.col("history_min_second") == 0)
            & (pl.col("history_max_second") == maximum)
        )
        .select("market_id")
    )
    candidates = frame.filter(
        pl.col("seconds_elapsed").is_between(minimum, maximum, closed="both")
        & ((pl.col("seconds_elapsed") - minimum) % cadence == 0)
    ).join(history_complete_markets, on="market_id", how="inner")

    expected_rows = ((maximum - minimum) // cadence) + 1
    complete_markets = (
        candidates.group_by("market_id")
        .agg(pl.len().alias("candidate_rows"))
        .filter(pl.col("candidate_rows") == expected_rows)
        .select("market_id")
    )
    candidates = candidates.join(complete_markets, on="market_id", how="inner")

    final_audit = (
        candidates.group_by("market_id")
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
            (pl.col("has_final_price") & (pl.col("audited_label_up") == pl.col("label_up"))).alias(
                "final_price_matches_official"
            )
        )
    )
    audit_rows = final_audit.select(
        pl.len().alias("markets"),
        pl.col("has_final_price").sum().alias("markets_with_final_price"),
        pl.col("final_price_matches_official").sum().alias("matching_final_price_labels"),
    ).row(0, named=True)
    candidates = candidates.drop("official_outcome", "final_price")

    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates.write_parquet(destination, compression="zstd", statistics=True)
    metadata: dict[str, Any] = {
        "build_contract": build_contract,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_partitions": len(source_files),
        "source_rows": frame.height,
        "candidate_rows": candidates.height,
        "candidate_markets": candidates["market_id"].n_unique(),
        "history_complete_markets": history_complete_markets.height,
        "expected_candidate_rows_per_market": expected_rows,
        "range_start": config.data.range_start.isoformat(),
        "range_end": config.data.range_end.isoformat(),
        "sample_interval_seconds": cadence,
        "minimum_seconds_after_open": minimum,
        "minimum_seconds_before_close": config.data.min_seconds_before_close,
        "class_up_markets": candidates.filter(pl.col("label_up") == 1)["market_id"].n_unique(),
        "class_down_markets": candidates.filter(pl.col("label_up") == 0)["market_id"].n_unique(),
        "markets_with_final_price": int(audit_rows["markets_with_final_price"] or 0),
        "matching_final_price_labels": int(audit_rows["matching_final_price_labels"] or 0),
        "feature_groups": FEATURE_GROUPS,
        "feature_file_sha256": file_sha256(destination),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        f"features: wrote {candidates.height:,} rows across "
        f"{metadata['candidate_markets']:,} markets",
        flush=True,
    )
    return metadata


def feature_build_contract(config: TrainingConfig, source_manifest_path: Path) -> dict[str, Any]:
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "range_start": config.data.range_start.isoformat(),
        "range_end": config.data.range_end.isoformat(),
        "sample_interval_seconds": config.data.sample_interval_seconds,
        "minimum_seconds_after_open": config.data.min_seconds_after_open,
        "minimum_seconds_before_close": config.data.min_seconds_before_close,
        "strict_final_price_audit": config.data.strict_final_price_audit,
    }


def validate_feature_cache(config: TrainingConfig) -> dict[str, Any]:
    destination = config.paths.feature_data
    metadata_path = destination.with_suffix(".metadata.json")
    if not destination.exists() or not metadata_path.exists():
        raise RuntimeError("feature cache is missing; build features before training")
    source_manifest_path = config.paths.source_data / "manifest.json"
    load_source_manifest(source_manifest_path, config)
    expected_contract = feature_build_contract(config, source_manifest_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("build_contract") != expected_contract:
        raise RuntimeError("feature cache contract changed; rebuild features with --force")
    if metadata.get("feature_file_sha256") != file_sha256(destination):
        raise RuntimeError(
            "feature file does not match its metadata; rebuild features with --force"
        )
    return metadata


def load_source_manifest(manifest_path: Path, config: TrainingConfig) -> dict[str, Any]:
    if not manifest_path.exists():
        raise RuntimeError("source manifest is missing; run extraction first")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "range_start": config.data.range_start.isoformat(),
        "range_end": config.data.range_end.isoformat(),
        "strict_final_price_audit": config.data.strict_final_price_audit,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        joined = ", ".join(mismatches)
        raise RuntimeError(f"source manifest does not match training config ({joined})")
    if not manifest.get("partitions"):
        raise RuntimeError("source manifest contains no partitions")
    return manifest


def verified_source_files(source_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    paths = []
    for partition in manifest["partitions"]:
        if partition["rows"] <= 0:
            raise RuntimeError(
                f"source partition has no complete four-table rows: {partition['path']}"
            )
        path = source_dir / partition["path"]
        if not path.exists():
            raise RuntimeError(f"source partition is missing: {path.name}")
        if file_sha256(path) != partition["sha256"]:
            raise RuntimeError(f"source partition hash mismatch: {path.name}")
        paths.append(path)
    return paths


def derive_point_in_time_features(frame: pl.DataFrame) -> pl.DataFrame:
    frame = frame.with_columns(
        pl.col("btc_close").log().alias("btc_log_close"),
        (pl.col("btc_close") / pl.col("opening_boundary"))
        .log()
        .mul(10_000)
        .alias("btc_gap_from_open_bps"),
        (pl.col("seconds_elapsed") / 300.0).alias("seconds_elapsed_scaled"),
        ((300 - pl.col("seconds_elapsed")) / 300.0).alias("seconds_remaining_scaled"),
        ((pl.col("btc_close") - pl.col("btc_open")) / pl.col("btc_open") * 10_000).alias(
            "btc_candle_body_bps"
        ),
        ((pl.col("btc_high") - pl.col("btc_low")) / pl.col("btc_open") * 10_000).alias(
            "btc_candle_range_bps"
        ),
    )

    for seconds in (1, 5, 15, 30, 60):
        frame = frame.with_columns(
            (pl.col("btc_log_close") - pl.col("btc_log_close").shift(seconds).over("market_id"))
            .mul(10_000)
            .alias(f"btc_return_{seconds}s_bps")
        )

    one_second_log_return = pl.col("btc_log_close") - pl.col("btc_log_close").shift(1).over(
        "market_id"
    )
    frame = frame.with_columns(one_second_log_return.alias("btc_log_return_1s"))
    for seconds in (5, 15, 30, 60):
        frame = frame.with_columns(
            pl.col("btc_log_return_1s")
            .rolling_std(window_size=seconds, min_samples=max(2, seconds // 2))
            .over("market_id")
            .mul(10_000)
            .alias(f"btc_realized_volatility_{seconds}s_bps"),
            (
                pl.col("btc_high")
                .rolling_max(window_size=seconds, min_samples=max(2, seconds // 2))
                .over("market_id")
                - pl.col("btc_low")
                .rolling_min(window_size=seconds, min_samples=max(2, seconds // 2))
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
                - pl.col("btc_low").rolling_min(window_size=30, min_samples=15).over("market_id")
            )
            / (
                pl.col("btc_high").rolling_max(window_size=30, min_samples=15).over("market_id")
                - pl.col("btc_low").rolling_min(window_size=30, min_samples=15).over("market_id")
                + 1e-9
            )
        ).alias("btc_range_position_30s"),
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
        (pl.col("btc_quote_volume_5s") / (pl.col("btc_quote_volume_60s") / 12.0 + 1e-9)).alias(
            "btc_volume_surprise_5_to_60"
        ),
        ((pl.col("observed_at").dt.hour() * 2 * math.pi / 24).sin()).alias("hour_sin"),
        ((pl.col("observed_at").dt.hour() * 2 * math.pi / 24).cos()).alias("hour_cos"),
        ((pl.col("observed_at").dt.weekday() * 2 * math.pi / 7).sin()).alias("weekday_sin"),
        ((pl.col("observed_at").dt.weekday() * 2 * math.pi / 7).cos()).alias("weekday_cos"),
        ((pl.col("up_best_bid") + pl.col("up_best_ask")) / 2).alias("up_mid"),
        ((pl.col("down_best_bid") + pl.col("down_best_ask")) / 2).alias("down_mid"),
        (pl.col("up_best_ask") - pl.col("up_best_bid")).alias("up_spread"),
        (pl.col("down_best_ask") - pl.col("down_best_bid")).alias("down_spread"),
        (pl.col("up_ask_vwap_5") - pl.col("up_best_ask")).alias("up_vwap_slippage_5"),
        (pl.col("down_ask_vwap_5") - pl.col("down_best_ask")).alias("down_vwap_slippage_5"),
        pl.col("up_bid_depth").log1p().alias("up_log_bid_depth"),
        pl.col("down_bid_depth").log1p().alias("down_log_bid_depth"),
        pl.col("up_ask_depth").log1p().alias("up_log_ask_depth"),
        pl.col("down_ask_depth").log1p().alias("down_log_ask_depth"),
        (
            (pl.col("observed_at") - pl.col("up_provider_received_at"))
            .dt.total_milliseconds()
            .cast(pl.Float64)
        ).alias("up_provider_age_ms"),
        (
            (pl.col("observed_at") - pl.col("down_provider_received_at"))
            .dt.total_milliseconds()
            .cast(pl.Float64)
        ).alias("down_provider_age_ms"),
    )
    frame = frame.with_columns(
        (pl.col("up_mid") + pl.col("down_mid") - 1).alias("mid_complement_residual"),
        (pl.col("up_best_ask") + pl.col("down_best_ask") - 1).alias("ask_complement_residual"),
        (pl.col("up_mid") - pl.col("down_mid")).alias("book_mid_difference"),
        (pl.col("up_mid") - pl.col("up_mid").shift(5).over("market_id")).alias("up_mid_change_5s"),
        (pl.col("down_mid") - pl.col("down_mid").shift(5).over("market_id")).alias(
            "down_mid_change_5s"
        ),
        (pl.col("up_imbalance") - pl.col("up_imbalance").shift(5).over("market_id")).alias(
            "up_imbalance_change_5s"
        ),
        (pl.col("down_imbalance") - pl.col("down_imbalance").shift(5).over("market_id")).alias(
            "down_imbalance_change_5s"
        ),
        (pl.col("up_provider_age_ms") - pl.col("down_provider_age_ms"))
        .abs()
        .alias("provider_age_skew_ms"),
        pl.col("up_best_bid").is_null().cast(pl.Int8).alias("up_book_missing"),
        pl.col("down_best_bid").is_null().cast(pl.Int8).alias("down_book_missing"),
        (pl.col("quality_flags").fill_null(255).cast(pl.Int32) & 1 != 0)
        .cast(pl.Int8)
        .alias("quality_up_missing"),
        (pl.col("quality_flags").fill_null(255).cast(pl.Int32) & 2 != 0)
        .cast(pl.Int8)
        .alias("quality_down_missing"),
        (pl.col("quality_flags").fill_null(255).cast(pl.Int32) & 4 != 0)
        .cast(pl.Int8)
        .alias("quality_up_stale"),
        (pl.col("quality_flags").fill_null(255).cast(pl.Int32) & 8 != 0)
        .cast(pl.Int8)
        .alias("quality_down_stale"),
        (pl.col("quality_flags").fill_null(255).cast(pl.Int32) & 16 != 0)
        .cast(pl.Int8)
        .alias("quality_up_crossed"),
        (pl.col("quality_flags").fill_null(255).cast(pl.Int32) & 32 != 0)
        .cast(pl.Int8)
        .alias("quality_down_crossed"),
        (pl.col("quality_flags").fill_null(255).cast(pl.Int32) & 64 != 0)
        .cast(pl.Int8)
        .alias("quality_up_insufficient_depth"),
        (pl.col("quality_flags").fill_null(255).cast(pl.Int32) & 128 != 0)
        .cast(pl.Int8)
        .alias("quality_down_insufficient_depth"),
        (
            pl.col("up_ask_vwap_5").is_not_null()
            & (pl.col("quality_flags").fill_null(255).cast(pl.Int32) & (1 | 4 | 16) == 0)
        ).alias("up_executable"),
        (
            pl.col("down_ask_vwap_5").is_not_null()
            & (pl.col("quality_flags").fill_null(255).cast(pl.Int32) & (2 | 8 | 32) == 0)
        ).alias("down_executable"),
        (pl.col("btc_gap_from_open_bps") >= 0).cast(pl.Int8).alias("binance_sign_up"),
        (pl.col("up_mid") >= pl.col("down_mid")).cast(pl.Int8).alias("market_favorite_up"),
    )
    return frame


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
