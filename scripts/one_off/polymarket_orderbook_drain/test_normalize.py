from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

from .contract import CANONICAL_COLUMNS, schema
from .normalize import normalize_legacy


def test_legacy_normalizes_to_canonical_prefix() -> None:
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    row = {
        "checkpoint_id": UUID("00000000-0000-0000-0000-000000000001"),
        "source_timestamp": now,
        "received_at": now,
        "persisted_at": now,
        "connection_id": UUID("00000000-0000-0000-0000-000000000002"),
        "ingest_sequence": 7,
        "market_id": "market",
        "token_id": "1",
        "best_bid": Decimal("0.40000000"),
        "best_ask": Decimal("0.60000000"),
        "tick_size": Decimal("0.01000000"),
        "book": {"bids": [{"price": "0.4", "size": "2"}], "asks": [{"price": "0.6", "size": "3"}]},
        "source_hash": "source",
        "condition_id": "0x" + "a" * 64,
        "event_slug": "btc-updown-5m-1788220800",
        "window_start": now,
        "window_end": now,
        "outcome": "up",
    }
    normalized = normalize_legacy(row)
    assert tuple(normalized)[: len(CANONICAL_COLUMNS)] == CANONICAL_COLUMNS
    assert normalized["bids"] == '[["0.4","2"]]'
    assert normalized["archive_source_record_id"].startswith(str(row["checkpoint_id"]))
    assert len(normalized["archive_record_sha256"]) == 64
    assert schema().metadata[b"contract_version"]
