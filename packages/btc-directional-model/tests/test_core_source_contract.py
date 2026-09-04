from __future__ import annotations

from pathlib import Path

from btc_directional_model.core_extract import (
    CORE_ORACLE_ROUND_SCHEMA,
    CORE_SOURCE_SCHEMA,
)


def source_sql() -> str:
    return (
        Path(__file__).parent.parent / "sql" / "btc-core-source.sql"
    ).read_text()


def oracle_source_sql() -> str:
    return (
        Path(__file__).parent.parent / "sql" / "btc-core-oracle-source.sql"
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


def test_oracle_sql_uses_only_canonical_completed_oracle_rounds() -> None:
    sql = oracle_source_sql().lower()

    assert "market_data.polygon_chainlink_btcusd_oracle_rounds" in sql
    assert "polymarket.backfill_artifacts" in sql
    assert "artifact.status = 'completed'" in sql
    assert "round.feed_proxy_address = %(oracle_feed_proxy_address)s" in sql
    assert "btc_market_execution_snapshots" not in sql
    assert "btc_market_decision_execution_snapshots" not in sql
    assert "chainlink_btcusd_archive_ticks" not in sql
    assert "experiment" not in sql


def test_oracle_sql_routes_rounds_by_causal_block_availability() -> None:
    sql = oracle_source_sql().lower()

    assert "round.block_timestamp >= %(batch_start)s" in sql
    assert "round.block_timestamp < %(batch_end)s" in sql
    assert "round.source_timestamp < %(batch_end)s" in sql
    assert "%(oracle_max_publication_delay_seconds)s" in sql
    assert "round.source_timestamp < %(batch_start)s" in sql
    assert "round.block_timestamp <= %(batch_start)s" in sql
    assert "round.source_timestamp desc" in sql
    assert "round.block_number desc" in sql
    assert "round.log_index desc" in sql
    assert "limit 1" in sql


def test_oracle_round_arrow_schema_is_provenance_only_and_book_free() -> None:
    names = set(CORE_ORACLE_ROUND_SCHEMA.names)

    assert names == {
        "oracle_price",
        "oracle_source_timestamp",
        "oracle_block_timestamp",
        "oracle_phase_id",
        "oracle_round_id",
        "oracle_block_number",
        "oracle_log_index",
    }
    assert not any(
        "vwap" in name
        or "book" in name
        or "quality" in name
        or "missing" in name
        for name in names
    )
