from __future__ import annotations

from pathlib import Path

import pytest

from btc_directional_model.refprice_context_data import (
    ENTRY_SECONDS,
    SOURCE_RELATIONS,
    SOURCE_SQL,
    load_config,
    validate_inference_columns,
)


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_frozen_source_contract_uses_half_open_august_29_watermark() -> None:
    config = load_config(
        _package_root() / "configs/btc-5m-refprice-context-source-20260607-20260829.toml"
    )
    assert config.data_start.isoformat() == "2026-06-07T00:00:00+00:00"
    assert config.data_end.isoformat() == "2026-08-29T00:00:00+00:00"
    assert config.observation_seconds == ENTRY_SECONDS
    assert config.destination.name == "btc-refprice-context-source-20260607-20260829"


def test_union_queries_name_both_existing_relations_and_deduplicate() -> None:
    root = _package_root() / "sql"
    for source in ("core", "candles", "open_interest", "oracle", "aggregate_trades"):
        query = (root / SOURCE_SQL[source]).read_text()
        for relation in SOURCE_RELATIONS[source]:
            assert relation in query
        assert "UNION ALL" in query
        assert "DISTINCT ON" in query


def test_live_availability_uses_later_provider_or_receipt_clock() -> None:
    root = _package_root() / "sql"
    for source in ("core", "candles", "open_interest", "oracle", "aggregate_trades"):
        query = (root / SOURCE_SQL[source]).read_text().lower()
        assert "greatest(" in query


def test_twap_and_completed_market_fields_are_supervision_only() -> None:
    validate_inference_columns(
        (
            "chainlink_ref_return_30s_bps",
            "btc_signed_flow_30s",
            "binance_oi_change_15m_bps",
        )
    )
    for column in (
        "twap60_margin_bps",
        "target_label_up",
        "official_outcome",
        "final_price",
        "synthetic_label_error",
    ):
        with pytest.raises(ValueError, match="forbidden"):
            validate_inference_columns((column,))


def test_l2_source_is_qualified_historical_feature_relation_only() -> None:
    query = (_package_root() / "sql" / SOURCE_SQL["spot_l2"]).read_text()
    assert "polymarket.binance_spot_btcusdt_l2_training_features" in query
    assert "market_data.binance_spot_btcusdt_l2_snapshots" not in query
    assert "interval '2 seconds'" in query


def test_refprice_query_freezes_live_precedence_over_identical_archive_prices() -> None:
    query = (_package_root() / "sql" / SOURCE_SQL["refprice"]).read_text()
    assert "pmdata_chainlink_streams" in query
    assert "chainlink_data_streams" in query
    assert "WHEN 'chainlink_data_streams' THEN 2" in query
    assert "DISTINCT ON (source_timestamp)" in query
    assert "coalesce(report.provider_available_at, report.received_at)" in query
    assert "report.provider_available_at IS NOT NULL" not in query
