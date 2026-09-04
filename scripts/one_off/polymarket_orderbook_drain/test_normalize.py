from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

from .contract import CANONICAL_COLUMNS, schema
from .drain import refresh_root_manifest
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


def test_refresh_root_manifest_keeps_existing_partitions(tmp_path) -> None:
    first = tmp_path / "date=2026-09-04" / "hour=19.manifest.json"
    second = tmp_path / "date=2026-09-04" / "hour=20.manifest.json"
    first.parent.mkdir()
    base = {
        "contract_version": "polymarket-orderbook-snapshot-parquet-v1",
        "source_counts": {"legacy": 0, "canonical": 2},
        "source_total": 2,
        "duplicate_count": 0,
        "output_count": 2,
        "file_size_bytes": 10,
    }
    first.write_text(__import__("json").dumps({
        **base, "window_start": "2026-09-04T19:00:00+00:00",
        "window_end": "2026-09-04T20:00:00+00:00", "file": "first.parquet",
    }))
    second.write_text(__import__("json").dumps({
        **base, "window_start": "2026-09-04T20:00:00+00:00",
        "window_end": "2026-09-04T21:00:00+00:00", "file": "second.parquet",
    }))
    manifest = refresh_root_manifest(tmp_path)
    assert manifest["partition_count"] == 2
    assert manifest["source_counts"] == {"legacy": 0, "canonical": 4}
    assert manifest["source_total"] == manifest["output_count"] == 4
    assert manifest["window_start"] == "2026-09-04T19:00:00+00:00"
    assert manifest["window_end"] == "2026-09-04T21:00:00+00:00"
