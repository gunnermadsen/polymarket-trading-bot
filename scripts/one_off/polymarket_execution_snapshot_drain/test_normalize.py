from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from .contract import BASE_COLUMNS, CONTRACT_COLUMNS, EXPANDED_VWAP_COLUMNS
from .normalize import (
    compatible_measurements, expanded_fact, identity, normalize, shared_fact,
)


def row(expanded: bool = True):
    value = {column: Decimal("0.5") for column in CONTRACT_COLUMNS}
    for column in (
        "sampled_at", "up_source_timestamp", "up_provider_received_at",
        "down_source_timestamp", "down_provider_received_at", "created_at",
    ):
        value[column] = datetime(2026, 1, 1, tzinfo=timezone.utc)
    value.update(
        market_id="btc-updown-5m", artifact_id=uuid4(), schema_version="v1",
        up_source_row_number=1, down_source_row_number=2, quality_flags=0,
    )
    if not expanded:
        value = {column: value[column] for column in BASE_COLUMNS}
    return value


def test_legacy_rows_map_to_capacity_contract_without_inventing_vwap():
    record = normalize(row(expanded=False), has_expanded_vwap=False)
    assert tuple(record) == CONTRACT_COLUMNS
    assert all(record[column] is None for column in EXPANDED_VWAP_COLUMNS)


def test_lineage_does_not_change_natural_or_shared_fact_identity():
    left = normalize(row(), has_expanded_vwap=True)
    right_raw = row()
    right_raw["artifact_id"] = uuid4()
    right_raw["created_at"] = datetime(2026, 1, 2, tzinfo=timezone.utc)
    right_raw["schema_version"] = "legacy-compatible-v2"
    right = normalize(right_raw, has_expanded_vwap=True)
    assert identity(left) == identity(right)
    assert shared_fact(left) == shared_fact(right)
    assert expanded_fact(left) == expanded_fact(right)


def test_drain_is_copy_only():
    source = __file__.replace("test_normalize.py", "drain.py")
    text = open(source).read().lower()
    for statement in ("insert into", "update ", "delete from", "truncate ", "alter table", "drop table"):
        assert statement not in text


def test_partial_rows_are_compatible_only_when_populated_measurements_agree():
    complete = normalize(row(), has_expanded_vwap=True)
    partial = normalize(row(expanded=False), has_expanded_vwap=False)
    partial["up_best_bid"] = None
    partial["quality_flags"] = 33
    assert compatible_measurements(complete, partial)
    partial["down_best_bid"] = Decimal("0.7")
    assert not compatible_measurements(complete, partial)
