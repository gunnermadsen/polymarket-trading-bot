from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from .contract import CONTRACT_COLUMNS
from .normalize import factual_identity, identity, normalize


def row():
    value = {column: Decimal("1") for column in CONTRACT_COLUMNS}
    for column in ("second_start", "source_event_timestamp", "provider_received_at", "available_at", "ingested_at"):
        value[column] = datetime(2026, 1, 1, tzinfo=timezone.utc)
    value.update(symbol="BTCUSDT", source_update_id=1, feature_schema_version="v1", quality_status="qualified", artifact_id=uuid4())
    return value


def test_artifact_lineage_does_not_change_fact_identity():
    left = normalize(row())
    right_raw = row()
    right_raw["artifact_id"] = uuid4()
    right_raw["ingested_at"] = datetime(2026, 1, 2, tzinfo=timezone.utc)
    right = normalize(right_raw)
    assert identity(left) == identity(right)
    assert factual_identity(left) == factual_identity(right)


def test_drain_is_copy_only():
    source = (__file__.replace("test_normalize.py", "drain.py"))
    text = open(source).read().lower()
    for statement in ("insert into", "update ", "delete from", "truncate ", "alter table", "drop table"):
        assert statement not in text
