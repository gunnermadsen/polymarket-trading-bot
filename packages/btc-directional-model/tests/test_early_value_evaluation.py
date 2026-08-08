from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from btc_directional_model.early_value_config import EarlyValueConfig, EvidenceWindow
from btc_directional_model.early_value_evaluation import (
    attach_execution_value,
    ledger_metrics,
    policy_ledgers,
)


def _config() -> EarlyValueConfig:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return EarlyValueConfig(
        source_path=Path("x"), package_root=Path("."), core_config=Path("x"),
        l2_source=Path("x"), candle_source=Path("x"), refprice_source=Path("x"),
        price_source_sql=Path("x"), price_cache=Path("x"), runs=Path("x"),
        champion_model=Path("x"), fit=EvidenceWindow(start, start),
        calibration=EvidenceWindow(start, start), policy=EvidenceWindow(start, start),
        evaluation=EvidenceWindow(start, start), prediction_seconds=tuple(range(5, 241, 5)),
        price_seconds=(*range(1, 60), *range(60, 241, 5)),
        calibration_bands=((5, 15),), quantity=5.0, book_freshness_seconds=2,
        minimum_edge_per_share=0.015, cheap_price_min=0.20, cheap_price_max=0.30,
        confidence_reference=0.89, bootstrap_resamples=1000, random_seed=7,
    )


def test_dual_side_policy_uses_probability_minus_cost() -> None:
    timestamp = datetime(2026, 7, 20, tzinfo=UTC)
    keys = {
        "market_id": ["m"], "window_start": [timestamp], "observed_at": [timestamp],
        "seconds_elapsed": [5], "label_up": [1],
    }
    predictions = pl.DataFrame({
        **keys, "model": ["test"], "probability_yes": [0.25],
        "predicted_yes": [0], "confidence": [0.75], "correct": [False],
    })
    prices = pl.DataFrame({
        **keys, "window_end": [timestamp], "fee_rate": [0.0],
        "yes_ask_vwap_5": [0.20], "no_ask_vwap_5": [0.80],
    })
    scored = attach_execution_value(predictions, prices)
    ledger = policy_ledgers(scored, _config())["early_cheap_dual_side_value"]
    assert ledger.height == 1
    assert ledger["selected_yes"].item() is True
    assert ledger["realized_net"].item() == 4.0


def test_twenty_five_percent_accuracy_is_negative_at_thirty_cent_cost() -> None:
    ledger = pl.DataFrame(
        {
            "market_id": [str(index) for index in range(100)],
            "won": [True] * 25 + [False] * 75,
            "realized_net": [3.5] * 25 + [-1.5] * 75,
            "entry_cost_per_share": [0.30] * 100,
            "seconds_elapsed": [30] * 100,
        }
    )
    metrics = ledger_metrics(ledger)
    assert metrics["accuracy"] == 0.25
    assert metrics["net_profit"] == -25.0


def test_break_even_is_cost_not_fifty_percent() -> None:
    ledger = pl.DataFrame(
        {
            "market_id": [str(index) for index in range(100)],
            "won": [True] * 31 + [False] * 69,
            "realized_net": [3.5] * 31 + [-1.5] * 69,
            "entry_cost_per_share": [0.30] * 100,
            "seconds_elapsed": [30] * 100,
        }
    )
    metrics = ledger_metrics(ledger)
    assert metrics["accuracy"] == 0.31
    assert metrics["net_profit"] == 5.0
