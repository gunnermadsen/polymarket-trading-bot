from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

from .contract import CANONICAL_COLUMNS
from .normalize import normalize_legacy


def test_legacy_normalizes_to_exact_canonical_columns() -> None:
    timestamp = datetime(2026, 8, 1, tzinfo=timezone.utc)
    row = {
        "feed_id": "0x" + "1" * 64,
        "source_timestamp": timestamp,
        "valid_from_timestamp": timestamp,
        "price": Decimal(1),
        "bid": Decimal("0.9"),
        "ask": Decimal("1.1"),
        "report_sha256": "a" * 64,
        "artifact_id": UUID("00000000-0000-0000-0000-000000000001"),
        "ingested_at": timestamp,
    }
    assert tuple(normalize_legacy(row)) == CANONICAL_COLUMNS


def test_drain_queries_are_copy_only() -> None:
    from . import drain

    sql = f"{drain.CANONICAL_QUERY}\n{drain.LEGACY_QUERY}".upper()
    for mutation in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "CREATE "):
        assert mutation not in sql
