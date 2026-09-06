from __future__ import annotations

from typing import Any

DIRECT_REALTIME = "chainlink_btcusd_reference_price"
DIRECT_BACKFILL = "chainlink_btcusd_reference_ticks_backfill"
PMDATA_BACKFILL = "pmdata_chainlink_btcusd_refprice_backfill"


def normalize_canonical(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for key in ("capture_artifact_id", "backfill_artifact_id"):
        if result[key] is not None:
            result[key] = str(result[key])
    if result["source"] == "pmdata_chainlink_streams":
        result["strategy_key"] = PMDATA_BACKFILL
    return result


def normalize_legacy(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "chainlink_data_streams",
        "feed_id": row["feed_id"],
        "source_timestamp": row["source_timestamp"],
        "valid_from_timestamp": row["valid_from_timestamp"],
        "provider_available_at": None,
        "received_at": row["ingested_at"],
        "price": row["price"],
        "bid": row["bid"],
        "ask": row["ask"],
        "report_sha256": row["report_sha256"],
        "payload_sha256": row["report_sha256"],
        "strategy_key": DIRECT_BACKFILL,
        "capture_artifact_id": None,
        "ingested_at": row["ingested_at"],
        "expires_at": None,
        "report_version": None,
        "source_date": row["source_timestamp"].date(),
        "archive_row_number": None,
        "backfill_artifact_id": str(row["artifact_id"]),
        "report_hash_kind": "signed_report",
    }


def identity(row: dict[str, Any]) -> tuple[str, Any, str]:
    return row["feed_id"], row["source_timestamp"], row["report_sha256"]


def factual_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["feed_id"],
        row["source_timestamp"],
        row["valid_from_timestamp"],
        row["price"],
        row["bid"],
        row["ask"],
        row["report_sha256"],
    )
