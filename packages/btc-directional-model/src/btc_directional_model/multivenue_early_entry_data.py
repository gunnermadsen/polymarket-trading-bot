"""Checkpointed, read-only data preparation for the multi-venue tournament."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .chainlink_oi_features import BINANCE_OI_FEATURES, _attach_open_interest_features
from .continuous_edge_training import (
    BOOK_RAW_FEATURES,
    CHAINLINK_FEATURES,
    CORE_FEATURES,
    ORACLE_FEATURES,
    attach_book_features,
)
from .core_extract import file_sha256
from .core_features import (
    attach_causal_oracle_rounds,
    derive_core_point_in_time_features,
    derive_oracle_point_in_time_features,
    prepare_causal_oracle_rounds,
)
from .counterfactual_twap_state_data import attach_causal_refprice_features
from .middle_market_tournament import (
    TRADE_PRINT_FEATURES,
    _derive_trade_print_features,
    _join_trade_print_features,
)
from .twap60_training_data import (
    DataPaths,
    _isolated_query_frame,
    attach_candle_context,
    extract_tournament_sources,
    load_source_group,
)

SCHEMA_VERSION = "btc-multivenue-early-entry-data-v1"
KEY_COLUMNS = ("market_id", "window_start", "observed_at", "seconds_elapsed")
ENTRY_SECONDS = tuple(range(60, 241, 5))
KRAKEN_FEATURES = (
    "kraken_return_5s_bps",
    "kraken_return_15s_bps",
    "kraken_return_30s_bps",
    "kraken_return_60s_bps",
    "kraken_return_120s_bps",
    "kraken_realized_volatility_30s_bps",
    "kraken_realized_volatility_60s_bps",
    "kraken_log_quote_volume_30s",
    "kraken_log_quote_volume_60s",
    "kraken_print_signed_share_5s",
    "kraken_print_signed_share_30s",
    "kraken_print_signed_share_60s",
    "kraken_log_trade_count_30s",
    "kraken_log_trade_count_60s",
    "kraken_binance_basis_bps",
    "kraken_binance_return_agreement_30s",
    "kraken_binance_flow_agreement_30s",
)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class TournamentDataConfig:
    package_root: Path
    raw: dict[str, Any]
    source_start: datetime
    kraken_start: datetime
    open_interest_start: datetime
    binance_print_start: datetime
    fit_end: datetime
    sealed_start: datetime
    sealed_end: datetime
    cache: Path
    results: Path
    standard: DataPaths
    oi_sql: Path
    print_sql: Path
    kraken_candle_root: Path
    kraken_print_root: Path


def load_data_config(path: Path) -> TournamentDataConfig:
    package_root = path.resolve().parent.parent
    raw = tomllib.loads(path.read_text())
    windows = raw["windows"]
    paths = raw["paths"]
    sources = raw["sources"]
    cache = package_root / paths["cache"]
    standard = DataPaths(
        package_root=package_root,
        cache=cache,
        core_features=cache / "unused-core-features.parquet",
        core_current_sql=package_root / paths["core_source_sql"],
        oracle_sql=package_root / paths["oracle_source_sql"],
        label_sql=package_root / paths["label_source_sql"],
        refprice_sql=package_root / paths["refprice_source_sql"],
        candle_sql=package_root / paths["candle_source_sql"],
        execution_sql=package_root / paths["execution_source_sql"],
    )
    root = Path(sources["kraken_spot_root"])
    symbol = sources["kraken_symbol"]
    return TournamentDataConfig(
        package_root=package_root,
        raw=raw,
        source_start=_dt(windows["source_start"]),
        kraken_start=_dt(windows["kraken_start"]),
        open_interest_start=_dt(windows["open_interest_start"]),
        binance_print_start=_dt(windows["binance_print_start"]),
        fit_end=_dt(windows["fit_end"]),
        sealed_start=_dt(windows["sealed_start"]),
        sealed_end=_dt(windows["sealed_end"]),
        cache=cache,
        results=package_root / paths["committed_results"],
        standard=standard,
        oi_sql=package_root / paths["open_interest_source_sql"],
        print_sql=package_root / paths["binance_print_source_sql"],
        kraken_candle_root=root
        / f"dataset={sources['kraken_candle_dataset']}"
        / f"symbol={symbol}",
        kraken_print_root=root / f"dataset={sources['kraken_print_dataset']}" / f"symbol={symbol}",
    )


def _date_from_kraken_path(path: Path) -> datetime:
    stem = path.name.split("T", 1)[0]
    return datetime.strptime(stem, "%Y%m%d").replace(tzinfo=UTC)


def _kraken_files(root: Path, start: datetime, end: datetime) -> list[Path]:
    files = []
    for path in root.rglob("*.parquet"):
        day = _date_from_kraken_path(path)
        if start <= day < end:
            files.append(path)
    return sorted(files, key=_date_from_kraken_path)


def _extract_auxiliary_sources(config: TournamentDataConfig, *, force: bool) -> dict[str, Any]:
    config.cache.mkdir(parents=True, exist_ok=True)
    manifest_path = config.cache / "auxiliary-source-manifest.json"
    contract = {
        "schema_version": SCHEMA_VERSION,
        "range_start": config.source_start.isoformat(),
        "range_end": config.sealed_end.isoformat(),
        "oi_sql_sha256": file_sha256(config.oi_sql),
        "print_sql_sha256": file_sha256(config.print_sql),
        "read_only": True,
        "database_mutations": False,
    }
    if manifest_path.is_file() and not force:
        payload = json.loads(manifest_path.read_text())
        if payload["contract"] != contract:
            raise RuntimeError("auxiliary source extraction contract changed")
        return payload

    oi_path = config.cache / "open-interest.parquet"
    if force or not oi_path.is_file():
        oi = _isolated_query_frame(
            config.oi_sql.read_text(),
            {
                "open_interest_symbol": "BTCUSDT",
                "range_start": config.open_interest_start,
                "range_end": config.sealed_end,
                "history_minutes": 60,
            },
            cursor_name="multivenue_open_interest",
        )
        oi.write_parquet(oi_path, compression="zstd", statistics=True)

    print_dir = config.cache / "binance-prints"
    print_dir.mkdir(parents=True, exist_ok=True)
    partitions: list[dict[str, Any]] = []
    cursor = config.binance_print_start
    while cursor < config.sealed_end:
        end = min(cursor + timedelta(days=1), config.sealed_end)
        destination = print_dir / f"{cursor.date().isoformat()}.parquet"
        if force or not destination.is_file():
            frame = _isolated_query_frame(
                config.print_sql.read_text(),
                {"batch_start": cursor, "batch_end": end},
                cursor_name=f"multivenue_prints_{cursor:%Y%m%d}",
            )
            frame.write_parquet(destination, compression="zstd", statistics=True)
            print(
                f"multivenue extract: Binance prints {cursor.date()} {frame.height:,} rows",
                flush=True,
            )
        partitions.append(
            {
                "path": str(destination.relative_to(config.cache)),
                "rows": pl.scan_parquet(destination).select(pl.len()).collect().item(),
                "sha256": file_sha256(destination),
            }
        )
        _write_json(
            config.cache / "auxiliary-source-manifest.partial.json",
            {"contract": contract, "binance_print_partitions": partitions},
        )
        cursor = end
    payload = {
        "contract": contract,
        "open_interest": {
            "path": str(oi_path.relative_to(config.cache)),
            "rows": pl.scan_parquet(oi_path).select(pl.len()).collect().item(),
            "sha256": file_sha256(oi_path),
        },
        "binance_print_partitions": partitions,
    }
    _write_json(manifest_path, payload)
    (config.cache / "auxiliary-source-manifest.partial.json").unlink(missing_ok=True)
    return payload


def _snapshot_kraken_sources(config: TournamentDataConfig) -> dict[str, Any]:
    manifest_path = config.cache / "kraken-source-manifest.json"
    candles = _kraken_files(config.kraken_candle_root, config.kraken_start, config.sealed_end)
    prints = _kraken_files(config.kraken_print_root, config.kraken_start, config.sealed_end)
    if not candles or not prints:
        raise RuntimeError("Kraken candle or trade-print files are absent for the configured range")
    rows = {
        "candles": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in candles
        ],
        "prints": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in prints
        ],
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "range_start": config.kraken_start.isoformat(),
        "range_end": config.sealed_end.isoformat(),
        "availability_rule": "event or one-second bucket close plus one second",
        "l2_included": False,
        **rows,
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing != payload:
            raise RuntimeError("Kraken source files changed after the tournament snapshot")
    else:
        _write_json(manifest_path, payload)
    return payload


def extract_sources(config: TournamentDataConfig, *, force: bool = False) -> dict[str, Any]:
    standard = extract_tournament_sources(
        config.standard,
        range_start=config.source_start,
        range_end=config.sealed_end,
        current_start=config.source_start,
        force=force,
    )
    auxiliary = _extract_auxiliary_sources(config, force=force)
    kraken = _snapshot_kraken_sources(config)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "standard": standard,
        "auxiliary": auxiliary,
        "kraken": kraken,
        "database_mutations": False,
        "new_tables": False,
        "new_sources": False,
    }
    _write_json(config.cache / "complete-source-manifest.json", payload)
    return payload


def _load_binance_prints(config: TournamentDataConfig) -> pl.DataFrame:
    payload = json.loads((config.cache / "auxiliary-source-manifest.json").read_text())
    frames = [
        pl.read_parquet(config.cache / row["path"]) for row in payload["binance_print_partitions"]
    ]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed", rechunk=True).sort("second_start")


def build_kraken_features(config: TournamentDataConfig, *, force: bool = False) -> pl.DataFrame:
    destination = config.cache / "kraken-features.parquet"
    if destination.is_file() and not force:
        return pl.read_parquet(destination)
    manifest = json.loads((config.cache / "kraken-source-manifest.json").read_text())
    candle_paths = [row["path"] for row in manifest["candles"]]
    print_paths = [row["path"] for row in manifest["prints"]]
    candles = (
        pl.scan_parquet(candle_paths, hive_partitioning=False)
        .select(
            pl.col("bucket_start_ns").cast(pl.Int64),
            pl.col("close").cast(pl.Float64).alias("kraken_close"),
            pl.col("quote_volume").cast(pl.Float64).alias("kraken_candle_quote_volume"),
        )
        .sort("bucket_start_ns")
    )
    prints = (
        pl.scan_parquet(print_paths, hive_partitioning=False)
        .select(
            ((pl.col("event_timestamp_ns").cast(pl.Int64) // 1_000_000_000) * 1_000_000_000).alias(
                "bucket_start_ns"
            ),
            pl.col("price").cast(pl.Float64),
            pl.col("base_volume").cast(pl.Float64),
            pl.col("side").cast(pl.String).str.to_lowercase(),
        )
        .with_columns((pl.col("price") * pl.col("base_volume")).alias("quote"))
        .group_by("bucket_start_ns")
        .agg(
            pl.col("quote").sum().alias("kraken_print_quote_volume"),
            pl.when(pl.col("side") == "buy")
            .then(pl.col("quote"))
            .otherwise(-pl.col("quote"))
            .sum()
            .alias("kraken_signed_quote_volume"),
            pl.len().alias("kraken_trade_count"),
        )
    )
    frame = (
        candles.join(prints, on="bucket_start_ns", how="left")
        .with_columns(
            pl.col("kraken_print_quote_volume").fill_null(0.0),
            pl.col("kraken_signed_quote_volume").fill_null(0.0),
            pl.col("kraken_trade_count").fill_null(0),
        )
        .collect(engine="streaming")
        .sort("bucket_start_ns")
    )
    expressions: list[pl.Expr] = []
    for seconds in (5, 15, 30, 60, 120):
        exact = (
            pl.col("bucket_start_ns") - pl.col("bucket_start_ns").shift(seconds)
            == seconds * 1_000_000_000
        )
        expressions.append(
            pl.when(exact)
            .then((pl.col("kraken_close") / pl.col("kraken_close").shift(seconds)).log() * 10_000.0)
            .otherwise(None)
            .alias(f"kraken_return_{seconds}s_bps")
        )
    log_return = (pl.col("kraken_close") / pl.col("kraken_close").shift(1)).log() * 10_000.0
    frame = frame.with_columns(log_return.alias("_kraken_log_return_1s"), *expressions)
    extra: list[pl.Expr] = []
    for seconds in (30, 60):
        extra.extend(
            (
                pl.col("_kraken_log_return_1s")
                .rolling_std(seconds)
                .alias(f"kraken_realized_volatility_{seconds}s_bps"),
                pl.col("kraken_candle_quote_volume")
                .rolling_sum(seconds)
                .log1p()
                .alias(f"kraken_log_quote_volume_{seconds}s"),
                pl.col("kraken_trade_count")
                .rolling_sum(seconds)
                .log1p()
                .alias(f"kraken_log_trade_count_{seconds}s"),
            )
        )
    for seconds in (5, 30, 60):
        quote = pl.col("kraken_print_quote_volume").rolling_sum(seconds)
        signed = pl.col("kraken_signed_quote_volume").rolling_sum(seconds)
        extra.append((signed / (quote + 1e-9)).alias(f"kraken_print_signed_share_{seconds}s"))
    frame = (
        frame.with_columns(*extra)
        .with_columns(
            pl.from_epoch("bucket_start_ns", time_unit="ns")
            .dt.replace_time_zone("UTC")
            .alias("kraken_source_timestamp"),
            (
                pl.from_epoch("bucket_start_ns", time_unit="ns").dt.replace_time_zone("UTC")
                + pl.duration(seconds=1)
            ).alias("kraken_available_at"),
        )
        .select(
            "kraken_available_at",
            "kraken_source_timestamp",
            "kraken_close",
            *[
                name
                for name in KRAKEN_FEATURES
                if name
                not in {
                    "kraken_binance_basis_bps",
                    "kraken_binance_return_agreement_30s",
                    "kraken_binance_flow_agreement_30s",
                }
            ],
        )
    )
    frame.write_parquet(destination, compression="zstd", statistics=True)
    _write_json(
        config.cache / "kraken-feature-manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "rows": frame.height,
            "sha256": file_sha256(destination),
            "availability_rule": "one-second bucket close plus one second",
            "causal": True,
        },
    )
    return frame


def _preserving_feature_join(
    base: pl.DataFrame,
    subset: pl.DataFrame,
    feature_names: tuple[str, ...],
) -> pl.DataFrame:
    available = tuple(name for name in feature_names if name in subset.columns)
    if not available:
        return base.with_columns(
            *(pl.lit(None, dtype=pl.Float64).alias(name) for name in feature_names)
        )
    selected = subset.select(*KEY_COLUMNS, *available).unique(subset=list(KEY_COLUMNS), keep="last")
    output = base.join(selected, on=list(KEY_COLUMNS), how="left", validate="1:1")
    missing = tuple(name for name in feature_names if name not in output.columns)
    if missing:
        output = output.with_columns(
            *(pl.lit(None, dtype=pl.Float64).alias(name) for name in missing)
        )
    return output


def _mask_optional_features(
    frame: pl.DataFrame,
    feature_names: tuple[str, ...],
    eligibility_column: str,
) -> pl.DataFrame:
    """Preserve rows while removing values whose causal eligibility failed."""

    available = tuple(name for name in feature_names if name in frame.columns)
    if eligibility_column not in frame.columns:
        raise RuntimeError(f"missing optional-source eligibility column: {eligibility_column}")
    return frame.with_columns(
        *(
            pl.when(pl.col(eligibility_column)).then(pl.col(name)).otherwise(None).alias(name)
            for name in available
        )
    )


def _attach_execution_60_240(
    frame: pl.DataFrame, execution: pl.DataFrame, freshness: int
) -> pl.DataFrame:
    if execution.is_empty():
        return attach_book_features(
            frame.with_columns(
                *(pl.lit(None, dtype=pl.Float64).alias(name) for name in BOOK_RAW_FEATURES)
            )
        )
    evidence = execution.filter(
        pl.col("seconds_elapsed").is_in(ENTRY_SECONDS)
        & ((pl.col("quality_flags") & 63) == 0)
        & pl.col("up_provider_received_at").is_not_null()
        & pl.col("down_provider_received_at").is_not_null()
        & (pl.col("up_provider_received_at") <= pl.col("observed_at"))
        & (pl.col("down_provider_received_at") <= pl.col("observed_at"))
        & (
            pl.col("up_provider_received_at")
            >= pl.col("observed_at") - pl.duration(seconds=freshness)
        )
        & (
            pl.col("down_provider_received_at")
            >= pl.col("observed_at") - pl.duration(seconds=freshness)
        )
        & pl.all_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite() for name in BOOK_RAW_FEATURES
        )
    ).unique(subset=["market_id", "observed_at"], keep="last")
    columns = (
        "market_id",
        "observed_at",
        "fee_rate",
        "up_provider_received_at",
        "down_provider_received_at",
        "up_best_ask",
        "down_best_ask",
        "up_ask_depth",
        "down_ask_depth",
        "quality_flags",
        *BOOK_RAW_FEATURES,
    )
    joined = frame.join(
        evidence.select(*columns), on=["market_id", "observed_at"], how="left", validate="m:1"
    )
    return attach_book_features(joined)


def _attach_kraken(frame: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    joined = (
        frame.with_row_index("_row")
        .sort("observed_at")
        .join_asof(
            features.sort("kraken_available_at"),
            left_on="observed_at",
            right_on="kraken_available_at",
            strategy="backward",
        )
    )
    fresh = (
        pl.col("kraken_available_at").is_not_null()
        & (pl.col("kraken_available_at") <= pl.col("observed_at"))
        & (pl.col("observed_at") - pl.col("kraken_available_at") <= pl.duration(seconds=2))
    )
    output = (
        joined.with_columns(
            pl.when(fresh)
            .then((pl.col("kraken_close") / pl.col("btc_close")).log() * 10_000.0)
            .otherwise(None)
            .alias("kraken_binance_basis_bps"),
            pl.when(fresh)
            .then(pl.col("kraken_return_30s_bps").sign() * pl.col("btc_return_30s_bps").sign())
            .otherwise(None)
            .alias("kraken_binance_return_agreement_30s"),
            pl.when(fresh)
            .then(
                pl.col("kraken_print_signed_share_30s").sign()
                * pl.col("btc_signed_flow_30s").sign()
            )
            .otherwise(None)
            .alias("kraken_binance_flow_agreement_30s"),
            *(
                pl.when(fresh).then(pl.col(name)).otherwise(None).alias(name)
                for name in KRAKEN_FEATURES
                if name in joined.columns
                and name
                not in {
                    "kraken_binance_basis_bps",
                    "kraken_binance_return_agreement_30s",
                    "kraken_binance_flow_agreement_30s",
                }
            ),
        )
        .sort("_row")
        .drop("_row", "kraken_available_at", "kraken_source_timestamp", "kraken_close")
    )
    return output


def build_panel(
    config: TournamentDataConfig, *, force: bool = False
) -> tuple[pl.DataFrame, dict[str, Any]]:
    destination = config.cache / "multivenue-panel.parquet"
    manifest_path = config.cache / "panel-manifest.json"
    if destination.is_file() and manifest_path.is_file() and not force:
        return pl.read_parquet(destination), json.loads(manifest_path.read_text())

    raw = load_source_group(config.standard, "core_current").sort(["market_id", "seconds_elapsed"])
    raw = raw.unique(subset=["market_id", "observed_at"], keep="last", maintain_order=True)
    complete = (
        raw.group_by("market_id")
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
    boundaries = (
        raw.sort(["market_id", "seconds_elapsed"])
        .group_by("market_id", maintain_order=True)
        .agg(pl.first("btc_open").alias("opening_boundary"))
    )
    core = (
        raw.drop("opening_boundary")
        .join(complete, on="market_id", how="inner")
        .join(boundaries, on="market_id", how="inner", validate="m:1")
    )
    oracle = load_source_group(config.standard, "oracle")
    if oracle.is_empty():
        core = derive_core_point_in_time_features(core)
    else:
        rounds = prepare_causal_oracle_rounds(oracle)
        core = derive_oracle_point_in_time_features(
            derive_core_point_in_time_features(attach_causal_oracle_rounds(core, rounds))
        )
        core = _mask_optional_features(
            core,
            tuple(name for name in ORACLE_FEATURES if name != "early_oracle_eligible"),
            "oracle_model_eligible",
        )
    panel = core.filter(pl.col("seconds_elapsed").is_in(ENTRY_SECONDS)).sort(list(KEY_COLUMNS))
    panel = panel.with_columns(
        (pl.col("seconds_elapsed") / 300.0).alias("seconds_elapsed_scaled"),
        ((300 - pl.col("seconds_elapsed")) / 300.0).alias("seconds_remaining_scaled"),
    )

    refprice = load_source_group(config.standard, "refprice")
    if not refprice.is_empty():
        panel = attach_causal_refprice_features(panel, refprice)
        panel = _mask_optional_features(
            panel,
            tuple(name for name in panel.columns if name.startswith("chainlink_ref_")),
            "refprice_causal_eligible",
        )
    candles = load_source_group(config.standard, "candles")
    if not candles.is_empty():
        candle_subset = attach_candle_context(
            panel, candles.unique(subset=["close_timestamp"], keep="last")
        )
        panel = _preserving_feature_join(panel, candle_subset, tuple(CHAINLINK_FEATURES))

    oi = pl.read_parquet(config.cache / "open-interest.parquet")
    if not oi.is_empty():
        oi_subset = _attach_open_interest_features(panel, oi, max_age_seconds=600)
        panel = _preserving_feature_join(panel, oi_subset, tuple(BINANCE_OI_FEATURES))
    prints = _load_binance_prints(config)
    if not prints.is_empty():
        panel = _join_trade_print_features(panel, _derive_trade_print_features(prints))
    kraken = build_kraken_features(config, force=force)
    panel = _attach_kraken(panel, kraken)
    panel = _attach_execution_60_240(
        panel,
        load_source_group(config.standard, "execution"),
        int(config.raw["execution"]["freshness_seconds"]),
    )

    groups = {
        "core": tuple(name for name in CORE_FEATURES if name in panel.columns),
        "candles": tuple(name for name in CHAINLINK_FEATURES if name in panel.columns),
        "oracle": tuple(name for name in ORACLE_FEATURES if name in panel.columns),
        "refprice": tuple(name for name in panel.columns if name.startswith("chainlink_ref_")),
        "open_interest": tuple(name for name in BINANCE_OI_FEATURES if name in panel.columns),
        "binance_prints": tuple(name for name in TRADE_PRINT_FEATURES if name in panel.columns),
        "kraken": tuple(name for name in KRAKEN_FEATURES if name in panel.columns),
    }
    for group, features in groups.items():
        if group == "refprice":
            availability = pl.col("refprice_causal_eligible")
        elif group == "oracle":
            availability = pl.col("oracle_model_eligible")
        elif features:
            availability = pl.any_horizontal(
                pl.col(name).is_not_null() & pl.col(name).is_finite() for name in features
            )
        else:
            availability = pl.lit(False)
        panel = panel.with_columns(availability.fill_null(False).alias(f"has_{group}"))
    market_schedule = panel.group_by("market_id").agg(
        pl.col("seconds_elapsed").sort().alias("schedule"),
        pl.len().alias("rows"),
    )
    bad_schedule = market_schedule.filter(
        (pl.col("rows") != len(ENTRY_SECONDS)) | (pl.col("schedule") != pl.lit(list(ENTRY_SECONDS)))
    )
    if bad_schedule.height:
        raise RuntimeError(f"{bad_schedule.height} markets failed the 60..240 entry schedule")
    panel.write_parquet(destination, compression="zstd", statistics=True)
    coverage = {}
    for group in groups:
        coverage[group] = {
            "rows": panel.filter(pl.col(f"has_{group}")).height,
            "markets": panel.filter(pl.col(f"has_{group}")).select("market_id").n_unique(),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rows": panel.height,
        "markets": panel["market_id"].n_unique(),
        "entry_seconds": list(ENTRY_SECONDS),
        "source_markets": raw["market_id"].n_unique(),
        "complete_core_markets": complete.height,
        "feature_groups": {name: list(values) for name, values in groups.items()},
        "coverage": coverage,
        "label_contract": "official resolved up/down outcome for every retained historical market",
        "optional_missingness_preserves_rows": True,
        "kraken_l2_included": False,
        "database_mutations": False,
        "new_tables": False,
        "sha256": file_sha256(destination),
    }
    _write_json(manifest_path, manifest)
    return panel, manifest
