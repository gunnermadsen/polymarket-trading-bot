from __future__ import annotations

import pyarrow as pa

CONTRACT_VERSION = "polymarket-orderbook-snapshot-parquet-v1"
CANONICAL_COLUMNS = (
    "sampled_at",
    "source_timestamp",
    "provider_available_at",
    "received_at",
    "source",
    "market_id",
    "condition_id",
    "event_slug",
    "window_start",
    "window_end",
    "token_id",
    "outcome",
    "connection_epoch",
    "ingest_sequence",
    "tick_size",
    "best_bid",
    "best_ask",
    "bid_depth",
    "ask_depth",
    "bids",
    "asks",
    "source_hash",
    "book_sha256",
    "sampling_policy",
    "sampling_policy_sha256",
    "payload_sha256",
    "strategy_key",
    "capture_artifact_id",
    "ingested_at",
)


def schema() -> pa.Schema:
    timestamp = pa.timestamp("us", tz="UTC")
    decimal = pa.decimal128(18, 8)
    fields = [
        pa.field("sampled_at", timestamp, nullable=False),
        pa.field("source_timestamp", timestamp, nullable=False),
        pa.field("provider_available_at", timestamp, nullable=False),
        pa.field("received_at", timestamp, nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("market_id", pa.string(), nullable=False),
        pa.field("condition_id", pa.string()),
        pa.field("event_slug", pa.string()),
        pa.field("window_start", timestamp),
        pa.field("window_end", timestamp),
        pa.field("token_id", pa.string(), nullable=False),
        pa.field("outcome", pa.string()),
        pa.field("connection_epoch", pa.string(), nullable=False),
        pa.field("ingest_sequence", pa.int64(), nullable=False),
        pa.field("tick_size", decimal, nullable=False),
        pa.field("best_bid", decimal),
        pa.field("best_ask", decimal),
        pa.field("bid_depth", pa.int32(), nullable=False),
        pa.field("ask_depth", pa.int32(), nullable=False),
        pa.field("bids", pa.string(), nullable=False),
        pa.field("asks", pa.string(), nullable=False),
        pa.field("source_hash", pa.string()),
        pa.field("book_sha256", pa.string(), nullable=False),
        pa.field("sampling_policy", pa.string(), nullable=False),
        pa.field("sampling_policy_sha256", pa.string(), nullable=False),
        pa.field("payload_sha256", pa.string(), nullable=False),
        pa.field("strategy_key", pa.string(), nullable=False),
        pa.field("capture_artifact_id", pa.string()),
        pa.field("ingested_at", timestamp, nullable=False),
        pa.field("archive_source_relation", pa.string(), nullable=False),
        pa.field("archive_source_record_id", pa.string(), nullable=False),
        pa.field("archive_source_contract", pa.string(), nullable=False),
        pa.field("archive_record_sha256", pa.string(), nullable=False),
    ]
    return pa.schema(fields, metadata={b"contract_version": CONTRACT_VERSION.encode()})
