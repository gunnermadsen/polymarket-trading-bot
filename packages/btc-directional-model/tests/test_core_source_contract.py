from __future__ import annotations

from pathlib import Path

from btc_directional_model.core_extract import CORE_SOURCE_SCHEMA


def source_sql() -> str:
    return (
        Path(__file__).parent.parent / "sql" / "btc-core-source.sql"
    ).read_text()


def test_core_sql_uses_only_canonical_backfill_inputs() -> None:
    sql = source_sql().lower()

    assert "btc_interval_markets" in sql
    assert "btc_market_reference_facts" in sql
    assert "binance_one_second_klines" in sql
    assert "backfill_artifacts" in sql
    assert "btc_market_execution_snapshots" not in sql
    assert "btc_orderbook_archive_events" not in sql
    assert "binance_aggregate_trades" not in sql
    assert "chainlink" not in sql
    assert "experiment" not in sql


def test_core_sql_uses_only_a_fully_completed_prior_second() -> None:
    sql = source_sql().lower()

    assert "open_timestamp + interval '1 second'" in sql
    assert "close_timestamp < kline.open_timestamp + interval '1 second'" in sql
    assert "market.window_start - interval '1 second'" in sql
    assert "market.window_end - interval '1 second'" in sql


def test_core_arrow_schema_contains_no_book_or_live_fields() -> None:
    names = set(CORE_SOURCE_SCHEMA.names)

    assert "opening_boundary" in names
    assert "label_up" in names
    assert "btc_close" in names
    assert not any("vwap" in name or "book" in name or "provider" in name for name in names)
