from __future__ import annotations

import pyarrow as pa

DIRECT_CONTRACT_VERSION = "chainlink-btcusd-reference-prices-v1"
PMDATA_CONTRACT_VERSION = "pmdata-chainlink-btcusd-reference-prices-v1"

CANONICAL_COLUMNS = (
    "source",
    "feed_id",
    "source_timestamp",
    "valid_from_timestamp",
    "provider_available_at",
    "received_at",
    "price",
    "bid",
    "ask",
    "report_sha256",
    "payload_sha256",
    "strategy_key",
    "capture_artifact_id",
    "ingested_at",
    "expires_at",
    "report_version",
    "source_date",
    "archive_row_number",
    "backfill_artifact_id",
    "report_hash_kind",
)


def schema(contract_version: str) -> pa.Schema:
    timestamp = pa.timestamp("us", tz="UTC")
    decimal = pa.decimal128(38, 18)
    return pa.schema(
        [
            pa.field("source", pa.string(), nullable=False),
            pa.field("feed_id", pa.string(), nullable=False),
            pa.field("source_timestamp", timestamp, nullable=False),
            pa.field("valid_from_timestamp", timestamp),
            pa.field("provider_available_at", timestamp),
            pa.field("received_at", timestamp, nullable=False),
            pa.field("price", decimal, nullable=False),
            pa.field("bid", decimal),
            pa.field("ask", decimal),
            pa.field("report_sha256", pa.string(), nullable=False),
            pa.field("payload_sha256", pa.string(), nullable=False),
            pa.field("strategy_key", pa.string(), nullable=False),
            pa.field("capture_artifact_id", pa.string()),
            pa.field("ingested_at", timestamp, nullable=False),
            pa.field("expires_at", timestamp),
            pa.field("report_version", pa.string()),
            pa.field("source_date", pa.date32()),
            pa.field("archive_row_number", pa.int64()),
            pa.field("backfill_artifact_id", pa.string()),
            pa.field("report_hash_kind", pa.string(), nullable=False),
        ],
        metadata={b"contract_version": contract_version.encode()},
    )
