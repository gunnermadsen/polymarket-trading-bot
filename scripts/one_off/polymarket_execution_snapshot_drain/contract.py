from __future__ import annotations

import pyarrow as pa

CONTRACT_VERSION = "polymarket-btc-market-capacity-execution-snapshots-v2"

BASE_COLUMNS = (
    "market_id", "sampled_at", "artifact_id", "schema_version",
    "up_source_row_number", "up_source_timestamp", "up_provider_received_at",
    "up_best_bid", "up_best_ask", "up_best_bid_size", "up_best_ask_size",
    "up_bid_depth", "up_ask_depth", "up_ask_vwap_1", "up_ask_vwap_5",
    "up_ask_vwap_10", "up_imbalance", "down_source_row_number",
    "down_source_timestamp", "down_provider_received_at", "down_best_bid",
    "down_best_ask", "down_best_bid_size", "down_best_ask_size",
    "down_bid_depth", "down_ask_depth", "down_ask_vwap_1",
    "down_ask_vwap_5", "down_ask_vwap_10", "down_imbalance",
    "quality_flags", "created_at",
)

EXPANDED_VWAP_COLUMNS = tuple(
    f"{side}_ask_vwap_{quantity}"
    for side in ("up", "down")
    for quantity in (15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200)
)

# Preserve the existing PostgreSQL table's physical column order.
CONTRACT_COLUMNS = BASE_COLUMNS + EXPANDED_VWAP_COLUMNS
LINEAGE_COLUMNS = {"artifact_id", "created_at", "schema_version"}
SHARED_FACT_COLUMNS = tuple(
    column for column in BASE_COLUMNS if column not in LINEAGE_COLUMNS
)
MEASUREMENT_COLUMNS = tuple(
    column for column in SHARED_FACT_COLUMNS if column != "quality_flags"
)

SOURCES = (
    {
        "table": "polymarket.btc_market_capacity_execution_snapshots",
        "has_expanded_vwap": True,
        "preference": 0,
    },
    {
        "table": "polymarket.btc_market_execution_snapshots",
        "has_expanded_vwap": False,
        "preference": 1,
    },
    {
        "table": "polymarket.btc_market_decision_execution_snapshots",
        "has_expanded_vwap": False,
        "preference": 2,
    },
)


def schema() -> pa.Schema:
    timestamp = pa.timestamp("us", tz="UTC")
    decimal = pa.decimal128(18, 8)
    timestamp_columns = {
        "sampled_at", "up_source_timestamp", "up_provider_received_at",
        "down_source_timestamp", "down_provider_received_at", "created_at",
    }
    integer64_columns = {"up_source_row_number", "down_source_row_number"}
    fields = []
    for column in CONTRACT_COLUMNS:
        if column in timestamp_columns:
            data_type = timestamp
        elif column in integer64_columns:
            data_type = pa.int64()
        elif column == "quality_flags":
            data_type = pa.int32()
        elif column in {"market_id", "artifact_id", "schema_version"}:
            data_type = pa.string()
        else:
            data_type = decimal
        nullable = column not in {
            "market_id", "sampled_at", "artifact_id", "schema_version",
            "quality_flags", "created_at",
        }
        fields.append(pa.field(column, data_type, nullable=nullable))
    return pa.schema(
        fields, metadata={b"contract_version": CONTRACT_VERSION.encode()}
    )
