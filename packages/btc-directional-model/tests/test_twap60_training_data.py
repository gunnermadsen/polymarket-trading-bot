from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from btc_directional_model.twap60_training_data import (
    attach_causal_refprice_features,
    authentic_labels,
    canonical_refprice_path,
    piecewise_average,
    verify_runtime_refprice_golden_vectors,
)


def test_piecewise_average_integrates_trailing_sixty_seconds() -> None:
    start = np.datetime64("2026-08-01T00:00:00", "us")
    timestamps = start + np.array([0, 30, 60], dtype="timedelta64[s]")
    prices = np.array([100.0, 110.0, 120.0])
    boundaries = start + np.array([60, 90], dtype="timedelta64[s]")

    result = piecewise_average(timestamps, prices, boundaries)

    assert np.allclose(result, [105.0, 115.0])


def test_canonical_refprice_path_is_deterministic() -> None:
    instant = datetime(2026, 8, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "source_timestamp": [instant, instant + timedelta(milliseconds=1)],
            "valid_from_timestamp": [instant, instant],
            "provider_available_at": [instant + timedelta(seconds=1)] * 2,
            "received_at": [instant + timedelta(seconds=1)] * 2,
            "price": [100.0, 101.0],
            "archive_row_number": [1, 2],
            "report_sha256": ["a", "b"],
        }
    )

    result = canonical_refprice_path(frame)

    assert result.height == 1
    assert result["price"].item() == 101.0


def test_authentic_twap60_uses_up_on_exact_equality() -> None:
    instant = datetime(2026, 8, 14, tzinfo=UTC)
    labels = pl.DataFrame(
        {
            "market_id": ["m"],
            "twap_open_price": [100.0],
            "twap_close_price": [100.0],
            "twap_open_source_timestamp": [instant],
            "twap_close_source_timestamp": [instant + timedelta(minutes=5)],
            "twap_open_valid_from_timestamp": [instant],
            "twap_close_valid_from_timestamp": [instant + timedelta(minutes=5)],
            "twap_open_effective_timestamp_rows": [1],
            "twap_close_effective_timestamp_rows": [1],
        }
    )

    result = authentic_labels(labels)

    assert result["authentic_label_up"].item() is True
    assert result["authentic_equality"].item() is True


def test_tournament_source_queries_are_bounded_read_only_selects() -> None:
    root = Path(__file__).resolve().parents[1] / "sql"
    names = (
        "btc-twap60-label-source.sql",
        "btc-twap60-refprice-source.sql",
        "btc-twap60-core-current-source.sql",
        "btc-capacity-execution-evidence.sql",
    )
    forbidden = ("insert ", "update ", "delete ", "create table", "alter table", "drop table")
    for name in names:
        sql = (root / name).read_text().lower()
        assert "% (" not in sql
        assert "batch_start" in sql and "batch_end" in sql
        assert not any(token in sql for token in forbidden)


def test_runtime_refprice_golden_vectors_match_existing_formula_contract() -> None:
    start = datetime(2026, 8, 24, tzinfo=UTC)
    source_times = [start + timedelta(seconds=second) for second in range(130)]
    refprice = pl.DataFrame(
        {
            "source_timestamp": source_times,
            "valid_from_timestamp": source_times,
            "provider_available_at": [value + timedelta(milliseconds=100) for value in source_times],
            "received_at": [value + timedelta(milliseconds=200) for value in source_times],
            "price": [100_000.0 + second for second in range(130)],
            "bid": [99_999.5 + second for second in range(130)],
            "ask": [100_000.5 + second for second in range(130)],
            "archive_row_number": list(range(130)),
            "report_sha256": [f"row-{second}" for second in range(130)],
        }
    )
    frame = pl.DataFrame(
        {
            "market_id": ["m"],
            "observed_at": [start + timedelta(seconds=125)],
            "opening_boundary": [100_000.0],
            "btc_close": [100_110.0],
            "btc_return_5s_bps": [1.0],
            "btc_return_30s_bps": [2.0],
        }
    )

    featured = attach_causal_refprice_features(frame, refprice)
    result = verify_runtime_refprice_golden_vectors(featured, refprice)

    assert result["passed"] is True
    assert result["compared_rows"] == 1
    assert result["maximum_absolute_error"] <= 1e-9
