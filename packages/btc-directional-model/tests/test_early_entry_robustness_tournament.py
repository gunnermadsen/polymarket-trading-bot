from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from btc_directional_model.early_entry_robustness_tournament import (
    ALL_CANDIDATES,
    L2_SIGNAL_FEATURES,
    L2ResidualModel,
    _attach_l2_confirmation,
    _bands,
    _load_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_ROOT / "configs/btc-5m-early-entry-robustness-20260321-20260901.toml"


def test_frozen_roster_and_chronological_windows() -> None:
    _, raw = _load_config(CONFIG)
    assert tuple(row["name"] for row in raw["candidates"]) == ALL_CANDIDATES
    assert datetime.fromisoformat(raw["windows"]["source_start"]) == datetime(
        2026, 3, 21, tzinfo=UTC
    )
    assert datetime.fromisoformat(raw["windows"]["fit_end"]) == datetime(2026, 8, 14, tzinfo=UTC)
    assert datetime.fromisoformat(raw["windows"]["development_end"]) == datetime(
        2026, 8, 20, tzinfo=UTC
    )
    assert datetime.fromisoformat(raw["windows"]["sealed_end"]) == datetime(2026, 9, 1, tzinfo=UTC)
    assert _bands(raw) == (
        ("60_89", 60, 89),
        ("90_119", 90, 119),
        ("120_149", 120, 149),
        ("150_180", 150, 180),
    )


def test_training_contract_has_no_database_mutation_or_coverage_lock() -> None:
    text = CONFIG.read_text().lower()
    assert "authentic_only" not in text
    assert "create table" not in text
    assert "insert into" not in text
    assert "update " not in text


def test_l2_availability_uses_numeric_evidence_not_boolean_flags() -> None:
    assert "has_spot_l2" not in L2_SIGNAL_FEATURES
    assert "has_kraken_l2" not in L2_SIGNAL_FEATURES
    frame = pl.DataFrame(
        {
            "side": ["up", "down"],
            "spot_l2_imbalance_20": [None, 0.2],
            "kraken_l2_update_imbalance_30s": [None, None],
        },
        schema_overrides={
            "spot_l2_imbalance_20": pl.Float64,
            "kraken_l2_update_imbalance_30s": pl.Float64,
        },
    )
    attached = _attach_l2_confirmation(frame)
    assert attached["has_any_l2"].to_list() == [False, True]


def test_residual_artifact_type_is_importable() -> None:
    assert L2ResidualModel.__module__ == ("btc_directional_model.early_entry_robustness_tournament")
