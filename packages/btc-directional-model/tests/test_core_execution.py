from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from btc_directional_model.core_execution import (
    DEFAULT_EXECUTION_QUANTITY,
    EXECUTION_EVIDENCE_CONTRACT,
    EXECUTION_EVIDENCE_SCHEMA,
    EXECUTION_EVIDENCE_SCHEMA_VERSION,
    LEGACY_EXECUTION_EVIDENCE_CONTRACT,
    LEGACY_EXECUTION_EVIDENCE_SCHEMA_VERSION,
    QUALITY_DOWN_INSUFFICIENT_DEPTH,
    QUALITY_UP_CROSSED,
    QUALITY_UP_STALE,
    ExecutionEvidenceConfig,
    classify_side_eligibility,
    expected_net_per_share,
    load_execution_evidence_manifest,
    realized_pnl,
    selected_side_eligibility,
    strict_both_side_eligible,
    strict_both_side_eligible_10,
    taker_fee_per_share,
)


def execution_sql() -> str:
    return (
        Path(__file__).parent.parent / "sql" / "btc-execution-evidence.sql"
    ).read_text()


def complete_side(
    *,
    side: str = "up",
    observed_at: datetime,
    provider_received_at: datetime | None,
    quality_flags: int = 0,
    ask_vwap_10: float | None = None,
):
    return classify_side_eligibility(
        side=side,  # type: ignore[arg-type]
        observed_at=observed_at,
        provider_received_at=provider_received_at,
        best_bid=0.40,
        best_ask=0.41,
        best_bid_size=10.0,
        best_ask_size=10.0,
        bid_depth=100.0,
        ask_depth=100.0,
        ask_vwap_5=0.42,
        ask_vwap_10=ask_vwap_10,
        imbalance=0.0,
        quality_flags=quality_flags,
    )


def test_execution_sql_is_bounded_to_canonical_compact_backfill_inputs() -> None:
    sql = execution_sql().lower()

    assert "polymarket.btc_interval_markets" in sql
    assert "polymarket.btc_market_decision_execution_snapshots" in sql
    assert "polymarket.backfill_artifacts" in sql
    assert "snapshot_artifact.status = 'completed'" in sql
    assert "polymarket_btc_five_minute_execution_snapshots" in sql
    assert "btc_orderbook_archive_events" not in sql
    assert "btc_market_execution_snapshots_one_second" not in sql
    assert "trading_process" not in sql
    assert "experiment" not in sql


def test_execution_sql_uses_exact_configured_candidate_timestamps() -> None:
    sql = execution_sql().lower()

    assert "%(batch_start)s" in sql
    assert "%(batch_end)s" in sql
    assert "%(min_seconds_after_open)s" in sql
    assert "%(max_seconds_after_open)s" in sql
    assert "%(sample_interval_milliseconds)s" in sql
    assert "snapshot.schema_version = %(snapshot_schema_version)s" in sql
    assert "snapshot.up_ask_vwap_10" in sql
    assert "snapshot.down_ask_vwap_10" in sql
    assert "(cohort.quality_flags & 255) = 0" in sql


def test_execution_schema_keeps_quality_as_eligibility_evidence() -> None:
    names = set(EXECUTION_EVIDENCE_SCHEMA.names)

    assert {"market_id", "observed_at", "quality_flags"} <= names
    assert {
        "up_side_valid",
        "down_side_valid",
        "up_side_fresh",
        "down_side_fresh",
        "up_stale_initialized",
        "down_stale_initialized",
        "strict_both_side_eligible",
    } <= names
    assert "up_ask_vwap_5" in names
    assert "down_ask_vwap_5" in names
    assert "up_ask_vwap_10" in names
    assert "down_ask_vwap_10" in names
    assert "strict_both_side_eligible_10" in names
    assert EXECUTION_EVIDENCE_CONTRACT == "btc_execution_evidence_v2"
    assert EXECUTION_EVIDENCE_SCHEMA_VERSION == "btc-execution-evidence-v2"


def test_side_quality_cohorts_require_causal_complete_books() -> None:
    observed_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)
    fresh = complete_side(
        observed_at=observed_at,
        provider_received_at=observed_at - timedelta(seconds=1),
    )
    stale = complete_side(
        observed_at=observed_at,
        provider_received_at=observed_at - timedelta(seconds=3),
        quality_flags=QUALITY_UP_STALE,
    )
    future = complete_side(
        observed_at=observed_at,
        provider_received_at=observed_at + timedelta(microseconds=1),
    )
    crossed = complete_side(
        observed_at=observed_at,
        provider_received_at=observed_at,
        quality_flags=QUALITY_UP_CROSSED,
    )

    assert fresh.valid and fresh.fresh and not fresh.stale_initialized
    assert stale.valid and not stale.fresh and stale.stale_initialized
    assert not future.valid
    assert not crossed.valid


def test_strict_both_side_uses_quality_bits_zero_through_five_only() -> None:
    observed_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)
    up = complete_side(
        observed_at=observed_at,
        provider_received_at=observed_at,
    )
    down = complete_side(
        side="down",
        observed_at=observed_at,
        provider_received_at=observed_at,
        quality_flags=QUALITY_DOWN_INSUFFICIENT_DEPTH,
    )

    assert selected_side_eligibility(predicted_up=True, up=up, down=down) is up
    assert strict_both_side_eligible(
        up=up,
        down=down,
        quality_flags=QUALITY_DOWN_INSUFFICIENT_DEPTH,
    )
    assert not strict_both_side_eligible_10(
        up=up,
        down=down,
        quality_flags=QUALITY_DOWN_INSUFFICIENT_DEPTH,
    )


def test_ten_share_eligibility_is_explicit_and_preserves_five_share_route() -> None:
    observed_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)
    up = complete_side(
        observed_at=observed_at,
        provider_received_at=observed_at,
        ask_vwap_10=0.43,
    )
    down = complete_side(
        side="down",
        observed_at=observed_at,
        provider_received_at=observed_at,
        ask_vwap_10=0.59,
    )

    assert strict_both_side_eligible(up=up, down=down, quality_flags=0)
    assert strict_both_side_eligible_10(up=up, down=down, quality_flags=0)


def test_v1_execution_manifest_remains_loadable(tmp_path: Path) -> None:
    start = datetime(2026, 5, 27, tzinfo=UTC)
    config = ExecutionEvidenceConfig(
        range_start=start,
        range_end=start + timedelta(days=1),
        output_dir=tmp_path,
    )
    legacy_manifest = {
        "source_contract": LEGACY_EXECUTION_EVIDENCE_CONTRACT,
        "source_schema_version": LEGACY_EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "range_start": config.range_start.isoformat(),
        "range_end": config.range_end.isoformat(),
        "sample_interval_seconds": config.sample_interval_seconds,
        "min_seconds_after_open": config.min_seconds_after_open,
        "max_seconds_after_open": config.max_seconds_after_open,
        "freshness_seconds": config.freshness_seconds,
        "quantity": config.quantity,
        "primary_key": ["market_id", "observed_at"],
        "partitions": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(legacy_manifest))

    loaded = load_execution_evidence_manifest(config)

    assert loaded["source_contract"] == LEGACY_EXECUTION_EVIDENCE_CONTRACT


def test_execution_config_is_fixed_to_five_share_vwap() -> None:
    start = datetime(2026, 4, 21, tzinfo=UTC)
    config = ExecutionEvidenceConfig(
        range_start=start,
        range_end=start + timedelta(days=1),
        output_dir=Path("generated/evidence"),
    )

    assert config.quantity == DEFAULT_EXECUTION_QUANTITY
    with pytest.raises(ValueError, match="five-share VWAP"):
        ExecutionEvidenceConfig(
            range_start=start,
            range_end=start + timedelta(days=1),
            output_dir=Path("generated/evidence"),
            quantity=10,
        )
    with pytest.raises(ValueError, match="must be UTC"):
        ExecutionEvidenceConfig(
            range_start=datetime(2026, 4, 21, tzinfo=timezone(timedelta(hours=-5))),
            range_end=datetime(2026, 4, 22, tzinfo=timezone(timedelta(hours=-5))),
            output_dir=Path("generated/evidence"),
        )


def test_economics_match_rust_dynamic_taker_fee_and_binary_payout() -> None:
    fee = taker_fee_per_share(0.25, 0.40)

    assert fee == pytest.approx(0.06)
    assert expected_net_per_share(0.80, 0.40, 0.25) == pytest.approx(0.34)
    assert realized_pnl(
        correct=1.0,
        execution_price=0.40,
        fee_rate=0.25,
    ) == pytest.approx(2.70)
    assert realized_pnl(
        correct=0.0,
        execution_price=0.40,
        fee_rate=0.25,
    ) == pytest.approx(-2.30)
