from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from btc_directional_model.benchmark_config import load_entry_benchmark_config
from btc_directional_model.offline_challengers import (
    PREOPEN_CANDIDATE,
    STRICT_BOOK_FEATURES,
    derive_strict_book_frame,
    preopen_candidate_spec,
    strict_book_candidate_spec,
)


def repository_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-entry-benchmark-20260421-20260720.toml"
    )


def evidence_row(
    market_id: str,
    observed_at: datetime,
    *,
    strict: bool,
) -> dict[str, object]:
    return {
        "market_id": market_id,
        "observed_at": observed_at,
        "strict_both_side_eligible": strict,
        "up_provider_received_at": observed_at - timedelta(milliseconds=20),
        "up_best_bid": 0.40,
        "up_best_ask": 0.42,
        "up_best_bid_size": 10.0,
        "up_best_ask_size": 10.0,
        "up_bid_depth": 100.0,
        "up_ask_depth": 90.0,
        "up_ask_vwap_5": 0.43,
        "up_imbalance": 0.10,
        "down_provider_received_at": observed_at - timedelta(milliseconds=30),
        "down_best_bid": 0.56,
        "down_best_ask": 0.58,
        "down_best_bid_size": 10.0,
        "down_best_ask_size": 10.0,
        "down_bid_depth": 90.0,
        "down_ask_depth": 100.0,
        "down_ask_vwap_5": 0.59,
        "down_imbalance": -0.10,
        "quality_flags": 0 if strict else 16,
    }


def test_book_features_are_derived_only_after_strict_routing() -> None:
    observed_at = datetime(2026, 6, 8, 0, 1, tzinfo=UTC)
    core = pl.DataFrame(
        {
            "market_id": ["valid", "invalid"],
            "observed_at": [observed_at, observed_at],
            "label_up": [1, 0],
        }
    )
    evidence = pl.DataFrame(
        [
            evidence_row("valid", observed_at, strict=True),
            evidence_row("invalid", observed_at, strict=False),
        ]
    )

    book = derive_strict_book_frame(core, evidence)

    assert book["market_id"].to_list() == ["valid"]
    assert set(STRICT_BOOK_FEATURES) <= set(book.columns)
    assert "quality_flags" not in book.columns
    assert book["model_eligible"].to_list() == [True]


def test_offline_candidates_cannot_silently_enter_current_runtime_schema() -> None:
    config = load_entry_benchmark_config(repository_config())
    preopen = preopen_candidate_spec()
    book = strict_book_candidate_spec(config)

    assert preopen.name == PREOPEN_CANDIDATE
    assert set(STRICT_BOOK_FEATURES) <= set(book.feature_names)
    assert "quality_flags" not in book.feature_names
    assert len(preopen.feature_names) > 58
    assert len(book.feature_names) > 58
