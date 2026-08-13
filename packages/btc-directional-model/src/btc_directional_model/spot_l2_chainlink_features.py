"""Causal feature construction for the fixed spot-L2/candle benchmark.

The source contract is the qualified, immutable Binance spot BTCUSDT
one-second training view.  Source identity, availability timestamps, and
quality lineage remain audit data; only the forty numeric information
dimensions below can reach a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from .chainlink_oi_features import (
    CHAINLINK_CANDLE_FEATURES,
    _derive_candle_source_features,
    _prepare_candles,
)

L2_SOURCE_FEATURE_COLUMNS = (
    "midpoint",
    "microprice",
    "spread_bps",
    "bid_depth_5",
    "ask_depth_5",
    "imbalance_5",
    "bid_depth_10",
    "ask_depth_10",
    "imbalance_10",
    "bid_depth_20",
    "ask_depth_20",
    "imbalance_20",
    "bid_depth_slope_20",
    "ask_depth_slope_20",
    "bid_depth_concentration_20",
    "ask_depth_concentration_20",
    "bid_quote_replenishment_1s",
    "ask_quote_replenishment_1s",
    "bid_quote_churn_1s",
    "ask_quote_churn_1s",
    "midpoint_change_bps_1s",
    "spread_bps_delta_1s",
    "depth_20_change_bps_1s",
    "imbalance_20_delta_1s",
    "midpoint_change_bps_5s",
    "spread_bps_delta_5s",
    "depth_20_change_bps_5s",
    "imbalance_20_delta_5s",
    "midpoint_change_bps_15s",
    "spread_bps_delta_15s",
    "depth_20_change_bps_15s",
    "imbalance_20_delta_15s",
    "midpoint_change_bps_30s",
    "spread_bps_delta_30s",
    "depth_20_change_bps_30s",
    "imbalance_20_delta_30s",
    "midpoint_change_bps_60s",
    "spread_bps_delta_60s",
    "depth_20_change_bps_60s",
    "imbalance_20_delta_60s",
)

L2_FEATURES = (
    "spot_l2_midpoint_to_kline_close_bps",
    "spot_l2_microprice_to_midpoint_bps",
    "spot_l2_spread_bps",
    "spot_l2_bid_depth_5_log",
    "spot_l2_ask_depth_5_log",
    "spot_l2_imbalance_5",
    "spot_l2_bid_depth_10_log",
    "spot_l2_ask_depth_10_log",
    "spot_l2_imbalance_10",
    "spot_l2_bid_depth_20_log",
    "spot_l2_ask_depth_20_log",
    "spot_l2_imbalance_20",
    "spot_l2_bid_depth_slope_20",
    "spot_l2_ask_depth_slope_20",
    "spot_l2_bid_depth_concentration_20",
    "spot_l2_ask_depth_concentration_20",
    "spot_l2_bid_quote_replenishment_1s_log",
    "spot_l2_ask_quote_replenishment_1s_log",
    "spot_l2_bid_quote_churn_1s_log",
    "spot_l2_ask_quote_churn_1s_log",
    "spot_l2_midpoint_change_1s_bps",
    "spot_l2_spread_change_1s_bps",
    "spot_l2_depth_20_change_1s_bps",
    "spot_l2_imbalance_20_change_1s",
    "spot_l2_midpoint_change_5s_bps",
    "spot_l2_spread_change_5s_bps",
    "spot_l2_depth_20_change_5s_bps",
    "spot_l2_imbalance_20_change_5s",
    "spot_l2_midpoint_change_15s_bps",
    "spot_l2_spread_change_15s_bps",
    "spot_l2_depth_20_change_15s_bps",
    "spot_l2_imbalance_20_change_15s",
    "spot_l2_midpoint_change_30s_bps",
    "spot_l2_spread_change_30s_bps",
    "spot_l2_depth_20_change_30s_bps",
    "spot_l2_imbalance_20_change_30s",
    "spot_l2_midpoint_change_60s_bps",
    "spot_l2_spread_change_60s_bps",
    "spot_l2_depth_20_change_60s_bps",
    "spot_l2_imbalance_20_change_60s",
)

L2_AUDIT_COLUMNS = (
    "symbol",
    "second_start",
    "source_event_timestamp",
    "provider_received_at",
    "available_at",
    "source_update_id",
)

L2_MAXIMUM_CAUSAL_AGE_SECONDS = 2
L2_CAUSAL_TIMESTAMP_COLUMNS = (
    "spot_l2_source_event_timestamp",
    "spot_l2_available_at",
)
L2_CAUSAL_AGE_COLUMNS = (
    "spot_l2_availability_age_seconds",
    "spot_l2_state_age_seconds",
)
L2_CAUSAL_AUDIT_COLUMNS = (
    *L2_CAUSAL_TIMESTAMP_COLUMNS,
    *L2_CAUSAL_AGE_COLUMNS,
)

_SOURCE_TO_MODEL = {
    "spread_bps": "spot_l2_spread_bps",
    "imbalance_5": "spot_l2_imbalance_5",
    "imbalance_10": "spot_l2_imbalance_10",
    "imbalance_20": "spot_l2_imbalance_20",
    "bid_depth_slope_20": "spot_l2_bid_depth_slope_20",
    "ask_depth_slope_20": "spot_l2_ask_depth_slope_20",
    "bid_depth_concentration_20": "spot_l2_bid_depth_concentration_20",
    "ask_depth_concentration_20": "spot_l2_ask_depth_concentration_20",
}
for _seconds in (1, 5, 15, 30, 60):
    _SOURCE_TO_MODEL.update(
        {
            f"midpoint_change_bps_{_seconds}s": f"spot_l2_midpoint_change_{_seconds}s_bps",
            f"spread_bps_delta_{_seconds}s": f"spot_l2_spread_change_{_seconds}s_bps",
            f"depth_20_change_bps_{_seconds}s": f"spot_l2_depth_20_change_{_seconds}s_bps",
            f"imbalance_20_delta_{_seconds}s": f"spot_l2_imbalance_20_change_{_seconds}s",
        }
    )


@dataclass(frozen=True)
class L2Normalizer:
    """Fit-interval-only standardization for the transformed L2 dimensions."""

    means: dict[str, float]
    scales: dict[str, float]

    @classmethod
    def fit(cls, frame: pl.DataFrame) -> L2Normalizer:
        _require_finite(frame, L2_FEATURES, "transformed L2 fitting frame")
        if frame.height < 2:
            raise RuntimeError("L2 normalization requires at least two fitting rows")
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for name in L2_FEATURES:
            values = frame[name].cast(pl.Float64)
            mean = values.mean()
            std = values.std(ddof=0)
            if mean is None or std is None or not math.isfinite(mean) or not math.isfinite(std):
                raise RuntimeError(f"L2 fitting statistic is not finite: {name}")
            means[name] = float(mean)
            scales[name] = max(float(std), 1e-12)
        return cls(means=means, scales=scales)

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        _require_finite(frame, L2_FEATURES, "transformed L2 frame")
        if set(self.means) != set(L2_FEATURES) or set(self.scales) != set(L2_FEATURES):
            raise RuntimeError("L2 normalizer schema does not match the fixed feature contract")
        return frame.with_columns(
            *(
                ((pl.col(name) - self.means[name]) / self.scales[name]).alias(name)
                for name in L2_FEATURES
            )
        )


def join_qualified_l2(
    core: pl.DataFrame,
    source: pl.DataFrame,
    *,
    maximum_age_seconds: int = L2_MAXIMUM_CAUSAL_AGE_SECONDS,
) -> pl.DataFrame:
    """Attach the latest strictly prior qualified state without filling gaps."""

    if maximum_age_seconds != L2_MAXIMUM_CAUSAL_AGE_SECONDS:
        raise ValueError("spot-L2 maximum age is fixed at two seconds")
    _require_columns(core, ("observed_at", "btc_close"), "core frame")
    _validate_l2_source(source)
    original_columns = tuple(core.columns)
    conflicts = sorted(
        (set(L2_FEATURES) | set(L2_CAUSAL_AUDIT_COLUMNS))
        & set(original_columns)
    )
    if conflicts:
        raise RuntimeError(
            "core frame already contains spot-L2 features or causal audit columns: "
            + ", ".join(conflicts)
        )

    joined = (
        core.with_row_index("_benchmark_row")
        .sort("observed_at")
        .join_asof(
            source.sort("available_at"),
            left_on="observed_at",
            right_on="available_at",
            strategy="backward",
            allow_exact_matches=False,
        )
    )
    availability_age_microseconds = (
        pl.col("observed_at") - pl.col("available_at")
    ).dt.total_microseconds()
    state_age_microseconds = (
        pl.col("observed_at") - pl.col("source_event_timestamp")
    ).dt.total_microseconds()
    joined = joined.filter(
        pl.col("available_at").is_not_null()
        & pl.col("source_event_timestamp").is_not_null()
        & (pl.col("available_at") < pl.col("observed_at"))
        & (availability_age_microseconds > 0)
        & (availability_age_microseconds <= maximum_age_seconds * 1_000_000)
        & (state_age_microseconds > 0)
        & (state_age_microseconds <= maximum_age_seconds * 1_000_000)
    )

    expressions: list[pl.Expr] = [
        (pl.col("midpoint") / pl.col("btc_close"))
        .log()
        .mul(10_000.0)
        .alias("spot_l2_midpoint_to_kline_close_bps"),
        (pl.col("microprice") / pl.col("midpoint"))
        .log()
        .mul(10_000.0)
        .alias("spot_l2_microprice_to_midpoint_bps"),
    ]
    for depth in (5, 10, 20):
        expressions.extend(
            (
                pl.col(f"bid_depth_{depth}").log().alias(f"spot_l2_bid_depth_{depth}_log"),
                pl.col(f"ask_depth_{depth}").log().alias(f"spot_l2_ask_depth_{depth}_log"),
            )
        )
    expressions.extend(
        (
            pl.col("bid_quote_replenishment_1s")
            .log1p()
            .alias("spot_l2_bid_quote_replenishment_1s_log"),
            pl.col("ask_quote_replenishment_1s")
            .log1p()
            .alias("spot_l2_ask_quote_replenishment_1s_log"),
            pl.col("bid_quote_churn_1s").log1p().alias("spot_l2_bid_quote_churn_1s_log"),
            pl.col("ask_quote_churn_1s").log1p().alias("spot_l2_ask_quote_churn_1s_log"),
        )
    )
    expressions.extend(
        pl.col(source_name).alias(model_name)
        for source_name, model_name in _SOURCE_TO_MODEL.items()
    )
    result = joined.with_columns(
        *expressions,
        pl.col("source_event_timestamp").alias(
            "spot_l2_source_event_timestamp"
        ),
        pl.col("available_at").alias("spot_l2_available_at"),
        (
            availability_age_microseconds.cast(pl.Float64) / 1_000_000.0
        ).alias("spot_l2_availability_age_seconds"),
        (state_age_microseconds.cast(pl.Float64) / 1_000_000.0).alias(
            "spot_l2_state_age_seconds"
        ),
    ).sort("_benchmark_row")
    result = result.select(
        *original_columns,
        *L2_FEATURES,
        *L2_CAUSAL_AUDIT_COLUMNS,
    )
    _require_finite(result, L2_FEATURES, "qualified spot-L2 join")
    _require_finite(
        result,
        L2_CAUSAL_AGE_COLUMNS,
        "qualified spot-L2 causal ages",
    )
    return result


def join_closed_chainlink_candles(
    core: pl.DataFrame,
    source: pl.DataFrame,
    *,
    maximum_age_seconds: int = 60,
) -> pl.DataFrame:
    """Attach a complete candle context whose close was strictly before ``t``."""

    if maximum_age_seconds != 60:
        raise ValueError("Chainlink candle maximum age is fixed at sixty seconds")
    _require_columns(core, ("observed_at",), "core frame")
    _require_columns(source, ("close_timestamp", "available_at"), "Chainlink candle source")
    original_columns = tuple(core.columns)
    conflicts = sorted(set(CHAINLINK_CANDLE_FEATURES) & set(original_columns))
    if conflicts:
        raise RuntimeError(
            "core frame already contains Chainlink candle features: " + ", ".join(conflicts)
        )
    availability = source.select("close_timestamp", "available_at").sort("close_timestamp")
    if availability.filter(
        pl.col("available_at").is_null() | (pl.col("available_at") < pl.col("close_timestamp"))
    ).height:
        raise RuntimeError("Chainlink candle source contains invalid availability timestamps")
    if availability["available_at"].n_unique() != availability.height:
        raise RuntimeError("Chainlink candle availability timestamps must be unique")
    if availability.filter(pl.col("available_at").diff() <= pl.duration(microseconds=0)).height:
        raise RuntimeError("Chainlink candle availability must increase with candle close")
    candles = (
        _derive_candle_source_features(_prepare_candles(source))
        .join(availability, on="close_timestamp", how="inner", validate="1:1")
        .sort("available_at")
    )
    joined = (
        core.with_row_index("_benchmark_row")
        .sort("observed_at")
        .join_asof(
            candles,
            left_on="observed_at",
            right_on="available_at",
            strategy="backward",
            allow_exact_matches=False,
        )
    )
    age_microseconds = (pl.col("observed_at") - pl.col("close_timestamp")).dt.total_microseconds()
    result = (
        joined.filter(
            pl.col("close_timestamp").is_not_null()
            & pl.col("available_at").is_not_null()
            & (pl.col("close_timestamp") < pl.col("observed_at"))
            & (pl.col("available_at") < pl.col("observed_at"))
            & (age_microseconds > 0)
            & (age_microseconds <= maximum_age_seconds * 1_000_000)
        )
        .filter(
            pl.all_horizontal([pl.col(name).is_not_null() for name in CHAINLINK_CANDLE_FEATURES])
        )
        .sort("_benchmark_row")
        .select(*original_columns, *CHAINLINK_CANDLE_FEATURES)
    )
    _require_finite(result, CHAINLINK_CANDLE_FEATURES, "closed Chainlink candle join")
    return result


def _validate_l2_source(frame: pl.DataFrame) -> None:
    _require_columns(frame, (*L2_AUDIT_COLUMNS, *L2_SOURCE_FEATURE_COLUMNS), "spot-L2 source")
    if frame.is_empty():
        raise RuntimeError("spot-L2 source is empty")
    if frame.select(pl.struct("second_start").n_unique()).item() != frame.height:
        raise RuntimeError("spot-L2 source contains more than one qualified state per second")
    numeric = (*L2_SOURCE_FEATURE_COLUMNS, "source_update_id")
    invalid = frame.filter(
        pl.any_horizontal([pl.col(name).is_null() for name in (*L2_AUDIT_COLUMNS, *numeric)])
        | pl.any_horizontal([~pl.col(name).cast(pl.Float64).is_finite() for name in numeric])
        | (pl.col("symbol") != "BTCUSDT")
        | (pl.col("source_event_timestamp") > pl.col("available_at"))
        | (pl.col("provider_received_at") > pl.col("available_at"))
        | (pl.col("second_start") > pl.col("available_at"))
        | (pl.col("available_at") >= pl.col("second_start") + pl.duration(seconds=1))
        | (pl.col("source_update_id") < 0)
        | (pl.col("midpoint") <= 0)
        | (pl.col("microprice") <= 0)
        | (pl.col("spread_bps") < 0)
        | (pl.col("bid_depth_5") <= 0)
        | (pl.col("ask_depth_5") <= 0)
        | (pl.col("bid_depth_5") > pl.col("bid_depth_10"))
        | (pl.col("bid_depth_10") > pl.col("bid_depth_20"))
        | (pl.col("ask_depth_5") > pl.col("ask_depth_10"))
        | (pl.col("ask_depth_10") > pl.col("ask_depth_20"))
        | ~pl.col("imbalance_5").is_between(-1.0, 1.0, closed="both")
        | ~pl.col("imbalance_10").is_between(-1.0, 1.0, closed="both")
        | ~pl.col("imbalance_20").is_between(-1.0, 1.0, closed="both")
        | (pl.col("bid_depth_slope_20") < 0)
        | (pl.col("ask_depth_slope_20") < 0)
        | ~pl.col("bid_depth_concentration_20").is_between(0.0, 1.0, closed="both")
        | ~pl.col("ask_depth_concentration_20").is_between(0.0, 1.0, closed="both")
        | (pl.col("bid_quote_replenishment_1s") < 0)
        | (pl.col("ask_quote_replenishment_1s") < 0)
        | (pl.col("bid_quote_churn_1s") < 0)
        | (pl.col("ask_quote_churn_1s") < 0)
    )
    if invalid.height:
        raise RuntimeError("spot-L2 source contains stale, invalid, or unqualified rows")


def _require_columns(frame: pl.DataFrame, names: tuple[str, ...], label: str) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing columns: {', '.join(missing)}")


def _require_finite(frame: pl.DataFrame, names: tuple[str, ...], label: str) -> None:
    _require_columns(frame, names, label)
    if frame.select(
        pl.any_horizontal(
            [pl.col(name).is_null() | ~pl.col(name).cast(pl.Float64).is_finite() for name in names]
        ).any()
    ).item():
        raise RuntimeError(f"{label} contains null or non-finite values")
