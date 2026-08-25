from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psycopg

POINT_KEY_COLUMNS = ("market_id", "window_start", "seconds_elapsed")
EXTERNAL_CORE_REQUIRED_COLUMNS = (
    *POINT_KEY_COLUMNS,
    "observed_at",
    "btc_close",
    "opening_boundary",
    "btc_return_30s_bps",
    "btc_path_from_window_open_bps",
)
CHAINLINK_REFPRICE_FEATURES = (
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
CHAINLINK_CANDLE_FEATURES = (
    "chainlink_candle_return_5m_bps",
    "chainlink_candle_return_15m_bps",
    "chainlink_candle_return_30m_bps",
    "chainlink_candle_return_60m_bps",
    "chainlink_candle_realized_volatility_15m_bps",
    "chainlink_candle_realized_volatility_60m_bps",
    "chainlink_candle_range_15m_bps",
    "chainlink_candle_range_60m_bps",
)
CHAINLINK_EXTERNAL_FEATURES = (
    *CHAINLINK_REFPRICE_FEATURES,
    *CHAINLINK_CANDLE_FEATURES,
)
BINANCE_OI_FEATURES = (
    "binance_oi_change_5m_bps",
    "binance_oi_change_15m_bps",
    "binance_oi_change_30m_bps",
    "binance_oi_change_60m_bps",
    "binance_oi_value_change_15m_bps",
    "binance_oi_value_change_60m_bps",
    "binance_oi_acceleration_5_vs_30_bps",
    "binance_oi_path_agreement_15m",
    "binance_oi_path_agreement_60m",
)
CHAINLINK_OI_FEATURES = (*CHAINLINK_EXTERNAL_FEATURES, *BINANCE_OI_FEATURES)

REFPRICE_SQL = "btc-chainlink-refprice-source.sql"
CANDLE_SQL = "btc-chainlink-one-minute-candles-source.sql"
OPEN_INTEREST_SQL = "btc-binance-five-minute-open-interest-source.sql"

_REFPRICE_HORIZONS_SECONDS = (1, 5, 15, 30, 60)
_CANDLE_HORIZONS_MINUTES = (5, 15, 30, 60)
_OI_HORIZONS_MINUTES = (5, 15, 30, 60)


@dataclass(frozen=True)
class ExternalSourceFrames:
    refprice: pl.DataFrame
    candles: pl.DataFrame
    open_interest: pl.DataFrame


def extract_external_source_frames(
    connection: psycopg.Connection[Any],
    package_root: Path,
    *,
    range_start: datetime,
    range_end: datetime,
    refprice_feed_id: str,
    refprice_history_seconds: int = 65,
    candle_history_minutes: int = 61,
    open_interest_history_minutes: int = 65,
    candle_symbol: str = "BTCUSD",
    open_interest_symbol: str = "BTCUSDT",
) -> ExternalSourceFrames:
    """Read only the bounded source intervals needed by the feature builder."""

    if range_start >= range_end:
        raise ValueError("external-source range_start must precede range_end")
    for name, value in (
        ("refprice_history_seconds", refprice_history_seconds),
        ("candle_history_minutes", candle_history_minutes),
        ("open_interest_history_minutes", open_interest_history_minutes),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    sql_root = package_root / "sql"
    refprice = _query_frame(
        connection,
        (sql_root / REFPRICE_SQL).read_text(),
        {
            "range_start": range_start,
            "range_end": range_end,
            "history_seconds": refprice_history_seconds,
            "refprice_feed_id": refprice_feed_id,
        },
        cursor_name="btc_chainlink_refprice_source",
    )
    candles = _query_frame(
        connection,
        (sql_root / CANDLE_SQL).read_text(),
        {
            "range_start": range_start,
            "range_end": range_end,
            "history_minutes": candle_history_minutes,
            "candle_symbol": candle_symbol,
        },
        cursor_name="btc_chainlink_candle_source",
    )
    open_interest = _query_frame(
        connection,
        (sql_root / OPEN_INTEREST_SQL).read_text(),
        {
            "range_start": range_start,
            "range_end": range_end,
            "history_minutes": open_interest_history_minutes,
            "open_interest_symbol": open_interest_symbol,
        },
        cursor_name="btc_binance_oi_source",
    )
    return ExternalSourceFrames(
        refprice=refprice,
        candles=candles,
        open_interest=open_interest,
    )


def derive_chainlink_oi_feature_frames(
    core_frame: pl.DataFrame,
    refprice_frame: pl.DataFrame,
    candle_frame: pl.DataFrame,
    open_interest_frame: pl.DataFrame,
    *,
    refprice_max_age_seconds: int,
    candle_max_age_seconds: int = 60,
    open_interest_max_age_seconds: int = 300,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return key-identical strict Chainlink and Chainlink-plus-OI cohorts.

    Source rows are joined point in time. A row is returned only when every
    allowlisted Chainlink and OI value exists and is finite. The no-OI frame is
    deliberately qualified by OI too, keeping the A/B comparison row matched.
    """

    for name, value in (
        ("refprice_max_age_seconds", refprice_max_age_seconds),
        ("candle_max_age_seconds", candle_max_age_seconds),
        ("open_interest_max_age_seconds", open_interest_max_age_seconds),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    _validate_core_frame(core_frame)
    original_columns = tuple(core_frame.columns)
    conflicts = sorted(set(CHAINLINK_OI_FEATURES) & set(original_columns))
    if conflicts:
        raise RuntimeError(
            "core frame already contains external feature columns: " + ", ".join(conflicts)
        )

    joined = _attach_refprice_features(
        core_frame,
        refprice_frame,
        max_age_seconds=refprice_max_age_seconds,
    )
    joined = _attach_candle_features(
        joined,
        candle_frame,
        max_age_seconds=candle_max_age_seconds,
    )
    joined = _attach_open_interest_features(
        joined,
        open_interest_frame,
        max_age_seconds=open_interest_max_age_seconds,
    )
    strict = _strict_feature_rows(joined, CHAINLINK_OI_FEATURES).sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    without_oi = strict.select(*original_columns, *CHAINLINK_EXTERNAL_FEATURES)
    with_oi = strict.select(*original_columns, *CHAINLINK_OI_FEATURES)
    _validate_key_identity(without_oi, with_oi)
    return without_oi, with_oi


def derive_chainlink_candle_feature_frame(
    core_frame: pl.DataFrame,
    candle_frame: pl.DataFrame,
    *,
    candle_max_age_seconds: int = 60,
) -> pl.DataFrame:
    """Attach only closed-candle context for the long-history candidate."""

    if candle_max_age_seconds <= 0:
        raise ValueError("candle_max_age_seconds must be positive")
    _validate_point_frame(core_frame)
    original_columns = tuple(core_frame.columns)
    conflicts = sorted(set(CHAINLINK_CANDLE_FEATURES) & set(original_columns))
    if conflicts:
        raise RuntimeError(
            "core frame already contains candle feature columns: " + ", ".join(conflicts)
        )
    joined = _attach_candle_features(
        core_frame,
        candle_frame,
        max_age_seconds=candle_max_age_seconds,
    )
    return (
        _strict_feature_rows(joined, CHAINLINK_CANDLE_FEATURES)
        .select(*original_columns, *CHAINLINK_CANDLE_FEATURES)
        .sort(["window_start", "market_id", "seconds_elapsed"])
    )


def _query_frame(
    connection: psycopg.Connection[Any],
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
                chunks.append(pl.DataFrame(rows, schema=columns, orient="row"))
    if not chunks:
        return pl.DataFrame({column: [] for column in columns})
    return pl.concat(chunks, how="vertical_relaxed", rechunk=True)


def _attach_refprice_features(
    core_frame: pl.DataFrame,
    source: pl.DataFrame,
    *,
    max_age_seconds: int,
) -> pl.DataFrame:
    refprice = _prepare_refprice(source).with_columns(
        ((pl.col("ask") - pl.col("bid")) / pl.col("price") * 10_000.0).alias("_ref_spread_bps")
    )
    ordered = core_frame.with_row_index("_external_source_row").sort("observed_at")
    observed_us = ordered["observed_at"].cast(pl.Int64).to_numpy()
    source_us = refprice["source_timestamp"].cast(pl.Int64).to_numpy()
    received_us = refprice["received_at"].cast(pl.Int64).to_numpy()
    prices = refprice["price"].to_numpy()
    spreads = refprice["_ref_spread_bps"].to_numpy()

    current_indices = _causal_refprice_indices(
        source_us,
        received_us,
        observed_us,
        observed_us - 1,
    )
    horizon_indices = {
        seconds: _causal_refprice_indices(
            source_us,
            received_us,
            observed_us,
            observed_us - seconds * 1_000_000,
        )
        for seconds in _REFPRICE_HORIZONS_SECONDS
    }
    eligible = current_indices >= 0
    current_age = np.full(len(observed_us), np.iinfo(np.int64).max, dtype=np.int64)
    current_age[eligible] = observed_us[eligible] - source_us[current_indices[eligible]]
    eligible &= (current_age > 0) & (current_age <= max_age_seconds * 1_000_000)
    for seconds, indices in horizon_indices.items():
        valid = indices >= 0
        target_us = observed_us - seconds * 1_000_000
        age = np.full(len(observed_us), np.iinfo(np.int64).max, dtype=np.int64)
        age[valid] = target_us[valid] - source_us[indices[valid]]
        eligible &= valid & (age >= 0) & (age <= max_age_seconds * 1_000_000)
    positions = np.flatnonzero(eligible)
    if not len(positions):
        return ordered.head(0).drop("_external_source_row").with_columns(
            *[pl.lit(None, dtype=pl.Float64).alias(name) for name in CHAINLINK_REFPRICE_FEATURES]
        )

    current = current_indices[positions]
    current_price = prices[current]
    horizon_price = {seconds: prices[indices[positions]] for seconds, indices in horizon_indices.items()}
    selected = ordered[positions].with_columns(
        *[
            pl.Series(
                f"chainlink_ref_return_{seconds}s_bps",
                np.log(current_price / horizon_price[seconds]) * 10_000.0,
            )
            for seconds in _REFPRICE_HORIZONS_SECONDS
        ],
        pl.Series(
            "chainlink_ref_binance_basis_bps",
            np.log(current_price / ordered["btc_close"].to_numpy()[positions]) * 10_000.0,
        ),
        pl.Series(
            "chainlink_ref_boundary_gap_bps",
            np.log(current_price / ordered["opening_boundary"].to_numpy()[positions]) * 10_000.0,
        ),
        pl.Series("chainlink_ref_spread_bps", spreads[current]),
        pl.Series(
            "chainlink_ref_spread_change_30s_bps",
            spreads[current] - spreads[horizon_indices[30][positions]],
        ),
        pl.Series(
            "chainlink_ref_binance_direction_agreement_30s",
            np.sign(ordered["btc_return_30s_bps"].to_numpy()[positions])
            * np.sign(np.log(current_price / horizon_price[30])),
        ),
    )
    return selected.sort("_external_source_row").drop("_external_source_row")


def _causal_refprice_indices(
    source_us: np.ndarray,
    received_us: np.ndarray,
    observed_us: np.ndarray,
    target_source_us: np.ndarray,
) -> np.ndarray:
    """Find the newest source-time anchor that was received by each decision."""

    indices = np.searchsorted(source_us, target_source_us, side="right") - 1
    for row in range(len(indices)):
        index = int(indices[row])
        while index >= 0 and received_us[index] > observed_us[row]:
            index -= 1
        indices[row] = index
    return indices


def _attach_candle_features(
    frame: pl.DataFrame,
    source: pl.DataFrame,
    *,
    max_age_seconds: int,
) -> pl.DataFrame:
    candles = _derive_candle_source_features(_prepare_candles(source))
    joined = (
        frame.with_row_index("_external_source_row")
        .sort("observed_at")
        .join_asof(
            candles.select("available_at", "close_timestamp", *CHAINLINK_CANDLE_FEATURES),
            left_on="observed_at",
            right_on="available_at",
            strategy="backward",
        )
    )
    age = (pl.col("observed_at") - pl.col("close_timestamp")).dt.total_microseconds()
    joined = joined.filter(
        pl.col("available_at").is_not_null()
        & (pl.col("available_at") <= pl.col("observed_at"))
        & pl.col("close_timestamp").is_not_null()
        & (pl.col("close_timestamp") <= pl.col("observed_at"))
        & (age >= 0)
        & (age < max_age_seconds * 1_000_000)
    )
    return joined.sort("_external_source_row").drop(
        "_external_source_row", "available_at", "close_timestamp"
    )


def _attach_open_interest_features(
    frame: pl.DataFrame,
    source: pl.DataFrame,
    *,
    max_age_seconds: int,
) -> pl.DataFrame:
    interest = _derive_open_interest_source_features(_prepare_open_interest(source))
    source_features = tuple(
        feature
        for feature in BINANCE_OI_FEATURES
        if not feature.startswith("binance_oi_path_agreement_")
    )
    joined = (
        frame.with_row_index("_external_source_row")
        .sort("observed_at")
        .join_asof(
            interest.select("available_at", "source_timestamp", *source_features),
            left_on="observed_at",
            right_on="available_at",
            strategy="backward",
            allow_exact_matches=False,
        )
    )
    age = (pl.col("observed_at") - pl.col("source_timestamp")).dt.total_microseconds()
    joined = joined.filter(
        pl.col("available_at").is_not_null()
        & (pl.col("available_at") <= pl.col("observed_at"))
        & pl.col("source_timestamp").is_not_null()
        & (pl.col("source_timestamp") < pl.col("observed_at"))
        & (age > 0)
        & (age <= max_age_seconds * 1_000_000)
    ).with_columns(
        (
            pl.col("binance_oi_change_15m_bps").sign()
            * pl.col("btc_path_from_window_open_bps").sign()
        ).alias("binance_oi_path_agreement_15m"),
        (
            pl.col("binance_oi_change_60m_bps").sign()
            * pl.col("btc_path_from_window_open_bps").sign()
        ).alias("binance_oi_path_agreement_60m"),
    )
    return joined.sort("_external_source_row").drop(
        "_external_source_row", "available_at", "source_timestamp"
    )


def _prepare_refprice(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "source_timestamp",
        "received_at",
        "valid_from_timestamp",
        "expires_at",
        "price",
        "bid",
        "ask",
    }
    _require_columns(frame, required, "RefPrice")
    prepared = frame.select(
        "source_timestamp",
        "received_at",
        "valid_from_timestamp",
        "expires_at",
        pl.col("price").cast(pl.Float64),
        pl.col("bid").cast(pl.Float64),
        pl.col("ask").cast(pl.Float64),
    )
    invalid = (
        pl.any_horizontal(pl.col(column).is_null() for column in required)
        | pl.any_horizontal(~pl.col(column).is_finite() for column in ("price", "bid", "ask"))
        | (pl.col("price") <= 0)
        | (pl.col("bid") <= 0)
        | (pl.col("ask") <= 0)
        | (pl.col("bid") > pl.col("price"))
        | (pl.col("price") > pl.col("ask"))
        | (pl.col("valid_from_timestamp") > pl.col("source_timestamp"))
        | (pl.col("received_at") < pl.col("source_timestamp"))
        | (pl.col("expires_at") < pl.col("valid_from_timestamp"))
    )
    prepared = (
        prepared.filter(~invalid)
        .sort(["source_timestamp", "received_at"])
        .unique(subset=["source_timestamp"], keep="first", maintain_order=True)
    )
    if prepared.is_empty():
        raise RuntimeError("RefPrice source contains no complete causal rows")
    _validate_unique_timestamp(prepared, "source_timestamp", "RefPrice")
    return prepared.sort("source_timestamp")


def _prepare_candles(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "open_timestamp",
        "close_timestamp",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    }
    _require_columns(frame, required, "Chainlink candle")
    available = (
        pl.col("available_at") if "available_at" in frame.columns else pl.col("close_timestamp")
    )
    prepared = frame.select(
        "open_timestamp",
        "close_timestamp",
        available.alias("available_at"),
        *[
            pl.col(column).cast(pl.Float64)
            for column in (
                "open_price",
                "high_price",
                "low_price",
                "close_price",
            )
        ],
    )
    invalid = prepared.filter(
        pl.any_horizontal(pl.col(column).is_null() for column in required)
        | pl.any_horizontal(
            ~pl.col(column).is_finite()
            for column in (
                "open_price",
                "high_price",
                "low_price",
                "close_price",
            )
        )
        | (pl.col("close_timestamp") != pl.col("open_timestamp") + pl.duration(minutes=1))
        | (pl.col("available_at") < pl.col("close_timestamp"))
        | (pl.col("low_price") <= 0)
        | (pl.col("high_price") < pl.col("open_price"))
        | (pl.col("high_price") < pl.col("close_price"))
        | (pl.col("high_price") < pl.col("low_price"))
        | (pl.col("low_price") > pl.col("open_price"))
        | (pl.col("low_price") > pl.col("close_price"))
    )
    if invalid.height:
        raise RuntimeError("Chainlink candle source contains invalid rows")
    _validate_unique_timestamp(prepared, "close_timestamp", "Chainlink candle")
    return prepared.sort("close_timestamp")


def _prepare_open_interest(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "source_timestamp",
        "period_seconds",
        "sum_open_interest",
        "sum_open_interest_value",
    }
    _require_columns(frame, required, "Binance open interest")
    available = (
        pl.col("available_at")
        if "available_at" in frame.columns
        else pl.col("source_timestamp")
    )
    prepared = frame.select(
        "source_timestamp",
        available.alias("available_at"),
        pl.col("period_seconds").cast(pl.Int32),
        pl.col("sum_open_interest").cast(pl.Float64),
        pl.col("sum_open_interest_value").cast(pl.Float64),
    )
    invalid = prepared.filter(
        pl.any_horizontal(pl.col(column).is_null() for column in required)
        | pl.any_horizontal(
            ~pl.col(column).is_finite()
            for column in ("sum_open_interest", "sum_open_interest_value")
        )
        | (pl.col("period_seconds") != 300)
        | (pl.col("available_at") < pl.col("source_timestamp"))
        | (pl.col("sum_open_interest") <= 0)
        | (pl.col("sum_open_interest_value") <= 0)
    )
    if invalid.height:
        raise RuntimeError("Binance open-interest source contains invalid rows")
    _validate_unique_timestamp(prepared, "source_timestamp", "Binance open interest")
    return prepared.sort("source_timestamp")


def _derive_candle_source_features(frame: pl.DataFrame) -> pl.DataFrame:
    enriched = frame.with_columns(
        pl.col("close_price").log().diff().alias("_candle_log_return_1m"),
        *[
            pl.col("close_timestamp").shift(minutes).alias(f"_candle_timestamp_lag_{minutes}m")
            for minutes in _CANDLE_HORIZONS_MINUTES
        ],
        *[
            pl.col("close_price").shift(minutes).alias(f"_candle_close_lag_{minutes}m")
            for minutes in _CANDLE_HORIZONS_MINUTES
        ],
    )
    enriched = enriched.with_columns(
        *[
            pl.when(
                pl.col("close_timestamp") - pl.col(f"_candle_timestamp_lag_{minutes}m")
                == pl.duration(minutes=minutes)
            )
            .then(
                (pl.col("close_price") / pl.col(f"_candle_close_lag_{minutes}m"))
                .log()
                .mul(10_000.0)
            )
            .alias(f"chainlink_candle_return_{minutes}m_bps")
            for minutes in _CANDLE_HORIZONS_MINUTES
        ],
        *[
            pl.when(
                pl.col("close_timestamp") - pl.col(f"_candle_timestamp_lag_{minutes}m")
                == pl.duration(minutes=minutes)
            )
            .then(
                pl.col("_candle_log_return_1m")
                .rolling_std(window_size=minutes, min_samples=minutes)
                .mul(10_000.0)
            )
            .alias(f"chainlink_candle_realized_volatility_{minutes}m_bps")
            for minutes in (15, 60)
        ],
        *[
            pl.when(
                pl.col("close_timestamp") - pl.col(f"_candle_timestamp_lag_{minutes}m")
                == pl.duration(minutes=minutes)
            )
            .then(
                (
                    pl.col("high_price").rolling_max(window_size=minutes, min_samples=minutes)
                    / pl.col("low_price").rolling_min(window_size=minutes, min_samples=minutes)
                )
                .log()
                .mul(10_000.0)
            )
            .alias(f"chainlink_candle_range_{minutes}m_bps")
            for minutes in (15, 60)
        ],
    )
    return enriched.select("available_at", "close_timestamp", *CHAINLINK_CANDLE_FEATURES)


def _derive_open_interest_source_features(frame: pl.DataFrame) -> pl.DataFrame:
    enriched = frame.with_columns(
        *[
            pl.col("source_timestamp").shift(minutes // 5).alias(f"_oi_timestamp_lag_{minutes}m")
            for minutes in _OI_HORIZONS_MINUTES
        ],
        *[
            pl.col("sum_open_interest").shift(minutes // 5).alias(f"_oi_lag_{minutes}m")
            for minutes in _OI_HORIZONS_MINUTES
        ],
        *[
            pl.col("sum_open_interest_value").shift(minutes // 5).alias(f"_oi_value_lag_{minutes}m")
            for minutes in (15, 60)
        ],
    )
    enriched = enriched.with_columns(
        *[
            pl.when(
                pl.col("source_timestamp") - pl.col(f"_oi_timestamp_lag_{minutes}m")
                == pl.duration(minutes=minutes)
            )
            .then((pl.col("sum_open_interest") / pl.col(f"_oi_lag_{minutes}m")).log().mul(10_000.0))
            .alias(f"binance_oi_change_{minutes}m_bps")
            for minutes in _OI_HORIZONS_MINUTES
        ],
        *[
            pl.when(
                pl.col("source_timestamp") - pl.col(f"_oi_timestamp_lag_{minutes}m")
                == pl.duration(minutes=minutes)
            )
            .then(
                (pl.col("sum_open_interest_value") / pl.col(f"_oi_value_lag_{minutes}m"))
                .log()
                .mul(10_000.0)
            )
            .alias(f"binance_oi_value_change_{minutes}m_bps")
            for minutes in (15, 60)
        ],
    ).with_columns(
        (pl.col("binance_oi_change_5m_bps") - pl.col("binance_oi_change_30m_bps") / 6.0).alias(
            "binance_oi_acceleration_5_vs_30_bps"
        )
    )
    return enriched.select(
        "available_at",
        "source_timestamp",
        *[
            feature
            for feature in BINANCE_OI_FEATURES
            if not feature.startswith("binance_oi_path_agreement_")
        ],
    )


def _strict_feature_rows(
    frame: pl.DataFrame,
    feature_names: tuple[str, ...],
) -> pl.DataFrame:
    return frame.drop_nulls(feature_names).filter(
        pl.all_horizontal(pl.col(feature).is_finite() for feature in feature_names)
    )


def _validate_core_frame(frame: pl.DataFrame) -> None:
    _require_columns(
        frame,
        set(EXTERNAL_CORE_REQUIRED_COLUMNS),
        "core feature",
    )
    _validate_point_frame(frame)


def _validate_point_frame(frame: pl.DataFrame) -> None:
    _require_columns(
        frame,
        {*POINT_KEY_COLUMNS, "observed_at"},
        "point-in-time feature",
    )
    if frame.select(pl.struct(POINT_KEY_COLUMNS).n_unique()).item() != frame.height:
        raise RuntimeError("point-in-time feature frame contains duplicate point keys")


def _validate_key_identity(
    without_oi: pl.DataFrame,
    with_oi: pl.DataFrame,
) -> None:
    left = without_oi.select(POINT_KEY_COLUMNS)
    right = with_oi.select(POINT_KEY_COLUMNS)
    if not left.equals(right, null_equal=True):
        raise RuntimeError("Chainlink A/B feature cohorts are not key-identical")


def _validate_unique_timestamp(
    frame: pl.DataFrame,
    column: str,
    role: str,
) -> None:
    if frame[column].n_unique() != frame.height:
        raise RuntimeError(f"{role} source contains duplicate timestamps")


def _require_columns(
    frame: pl.DataFrame,
    required: set[str],
    role: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{role} frame is missing required columns: " + ", ".join(missing))
