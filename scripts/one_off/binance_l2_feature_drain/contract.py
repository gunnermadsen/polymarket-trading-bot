from __future__ import annotations

import pyarrow as pa

CONTRACT_COLUMNS = (
    "symbol", "second_start", "source_event_timestamp", "provider_received_at",
    "available_at", "source_update_id", "feature_schema_version", "quality_status",
    "artifact_id", "midpoint", "microprice", "spread_bps", "bid_depth_5",
    "ask_depth_5", "imbalance_5", "bid_depth_10", "ask_depth_10", "imbalance_10",
    "bid_depth_20", "ask_depth_20", "imbalance_20", "bid_depth_slope_20",
    "ask_depth_slope_20", "bid_depth_concentration_20",
    "ask_depth_concentration_20", "bid_quote_replenishment_1s",
    "ask_quote_replenishment_1s", "bid_quote_churn_1s", "ask_quote_churn_1s",
    "midpoint_change_bps_1s", "spread_bps_delta_1s", "depth_20_change_bps_1s",
    "imbalance_20_delta_1s", "midpoint_change_bps_5s", "spread_bps_delta_5s",
    "depth_20_change_bps_5s", "imbalance_20_delta_5s", "midpoint_change_bps_15s",
    "spread_bps_delta_15s", "depth_20_change_bps_15s", "imbalance_20_delta_15s",
    "midpoint_change_bps_30s", "spread_bps_delta_30s", "depth_20_change_bps_30s",
    "imbalance_20_delta_30s", "midpoint_change_bps_60s", "spread_bps_delta_60s",
    "depth_20_change_bps_60s", "imbalance_20_delta_60s", "ingested_at",
)

FACT_COLUMNS = tuple(
    column for column in CONTRACT_COLUMNS if column not in {"artifact_id", "ingested_at"}
)

PRODUCTS = {
    "spot": {
        "contract_version": "binance-spot-btcusdt-l2-one-second-features-v1",
        "tables": (
            "polymarket.binance_spot_btcusdt_l2_one_second_features",
            "polymarket.binance_spot_btcusdt_l2_one_second_features_staging",
        ),
    },
    "futures": {
        "contract_version": "binance-btcusdt-l2-one-second-features-v1",
        "tables": (
            "polymarket.binance_btcusdt_l2_one_second_features",
            "polymarket.binance_btcusdt_l2_one_second_features_staging",
        ),
    },
}


def schema(contract_version: str) -> pa.Schema:
    timestamp = pa.timestamp("us", tz="UTC")
    decimal_30 = pa.decimal128(30, 10)
    decimal_20 = pa.decimal128(20, 10)
    decimal_20_names = {
        "spread_bps", "imbalance_5", "imbalance_10", "imbalance_20",
        "bid_depth_slope_20", "ask_depth_slope_20", "bid_depth_concentration_20",
        "ask_depth_concentration_20", "imbalance_20_delta_1s",
        "imbalance_20_delta_5s", "imbalance_20_delta_15s",
        "imbalance_20_delta_30s", "imbalance_20_delta_60s",
    }
    fields = []
    for column in CONTRACT_COLUMNS:
        if column in {"second_start", "source_event_timestamp", "provider_received_at", "available_at", "ingested_at"}:
            field = pa.field(column, timestamp, nullable=False)
        elif column == "source_update_id":
            field = pa.field(column, pa.int64(), nullable=False)
        elif column in {"symbol", "feature_schema_version", "quality_status", "artifact_id"}:
            field = pa.field(column, pa.string(), nullable=False)
        else:
            field = pa.field(column, decimal_20 if column in decimal_20_names else decimal_30, nullable=False)
        fields.append(field)
    return pa.schema(fields, metadata={b"contract_version": contract_version.encode()})
