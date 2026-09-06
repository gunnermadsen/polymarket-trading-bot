from __future__ import annotations

import hashlib
import json
from datetime import timezone
from decimal import Decimal
from typing import Any

STRATEGY_KEY = "polymarket_btc_five_minute_orderbooks"
POLICY_VERSION = "polymarket-clob-btc-5m-orderbook-top-n-v1"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _utc(value: Any) -> Any:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _levels(book: dict[str, Any], side: str) -> list[list[str]]:
    return [[str(level["price"]), str(level["size"])] for level in book.get(side, [])]


def _policy(*, top_n: int, legacy: bool) -> dict[str, Any]:
    policy = {
        "market_interval_seconds": 300,
        "sample_interval_ms": 1000,
        "selection": "latest_valid_subscribed_market_book_at_aligned_wall_clock_slot",
        "source": "polymarket_clob_market",
        "top_n": max(1, top_n),
        "version": POLICY_VERSION,
    }
    if legacy:
        policy["legacy_event_driven_checkpoint"] = True
    return policy


def normalize_legacy(row: dict[str, Any]) -> dict[str, Any]:
    book = row["book"]
    bids = _levels(book, "bids")
    asks = _levels(book, "asks")
    policy = _policy(top_n=max(len(bids), len(asks)), legacy=True)
    bids_json, asks_json, policy_json = _json(bids), _json(asks), _json(policy)
    source_record_id = f'{row["checkpoint_id"]}:{_utc(row["source_timestamp"]).isoformat()}'
    payload_hash = _sha256(f'{row["checkpoint_id"]}|{_json(book)}')
    result = {
        "sampled_at": _utc(row["persisted_at"]),
        "source_timestamp": _utc(row["source_timestamp"]),
        "provider_available_at": _utc(row["source_timestamp"]),
        "received_at": _utc(row["received_at"]),
        "source": "polymarket_clob_market",
        "market_id": row["market_id"],
        "condition_id": row["condition_id"],
        "event_slug": row["event_slug"],
        "window_start": _utc(row["window_start"]),
        "window_end": _utc(row["window_end"]),
        "token_id": row["token_id"],
        "outcome": row["outcome"],
        "connection_epoch": str(row["connection_id"]),
        "ingest_sequence": row["ingest_sequence"],
        "tick_size": Decimal(row["tick_size"]),
        "best_bid": Decimal(row["best_bid"]) if row["best_bid"] is not None else None,
        "best_ask": Decimal(row["best_ask"]) if row["best_ask"] is not None else None,
        "bid_depth": len(bids),
        "ask_depth": len(asks),
        "bids": bids_json,
        "asks": asks_json,
        "source_hash": row["source_hash"],
        "book_sha256": _sha256(f"{bids_json}|{asks_json}"),
        "sampling_policy": policy_json,
        "sampling_policy_sha256": _sha256(policy_json),
        "payload_sha256": payload_hash,
        "strategy_key": STRATEGY_KEY,
        "capture_artifact_id": None,
        "ingested_at": _utc(row["persisted_at"]),
        "archive_source_relation": "polymarket.orderbook_checkpoints",
        "archive_source_record_id": source_record_id,
        "archive_source_contract": "legacy-event-driven-checkpoint-v1",
    }
    result["archive_record_sha256"] = record_hash(result)
    return result


def normalize_canonical(
    row: dict[str, Any], *, source_relation: str
) -> dict[str, Any]:
    result = dict(row)
    for key in ("sampled_at", "source_timestamp", "provider_available_at", "received_at", "window_start", "window_end", "ingested_at"):
        result[key] = _utc(result[key])
    for key in ("connection_epoch", "capture_artifact_id"):
        if result[key] is not None:
            result[key] = str(result[key])
    for key in ("bids", "asks", "sampling_policy"):
        result[key] = _json(result[key])
    source_record_id = ":".join((
        result["sampled_at"].isoformat(), result["market_id"], result["token_id"],
        result["sampling_policy_sha256"],
    ))
    result.update({
        "archive_source_relation": source_relation,
        "archive_source_record_id": source_record_id,
        "archive_source_contract": "canonical-sampled-snapshot-v1",
    })
    result["archive_record_sha256"] = record_hash(result)
    return result


def record_hash(row: dict[str, Any]) -> str:
    values = {}
    for key, value in row.items():
        if key == "archive_record_sha256":
            continue
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        elif isinstance(value, Decimal):
            value = format(value, "f")
        values[key] = value
    return _sha256(_json(values))
