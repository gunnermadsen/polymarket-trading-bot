from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from btc_directional_model.champion_admission_tournament import (
    ALL_CANDIDATES,
    AdmissionModel,
    _admission_frame,
    _bands,
    _capacity_metrics,
    _fit_admission,
    _load_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_ROOT / "configs/btc-5m-champion-admission-tournament-20260321-20260901.toml"


def test_frozen_roster_and_windows() -> None:
    _, raw = _load_config(CONFIG)
    assert tuple(row["name"] for row in raw["candidates"]) == ALL_CANDIDATES
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


def test_admission_is_separate_and_importable() -> None:
    assert AdmissionModel.__module__ == "btc_directional_model.champion_admission_tournament"
    assert {"profitable", "stress_edge", "loss_severity"}.issubset(
        AdmissionModel.__dataclass_fields__
    )


def test_admission_has_no_best_later_target() -> None:
    source = (inspect.getsource(_admission_frame) + inspect.getsource(_fit_admission)).lower()
    assert "best_later" not in source
    assert "realized_stress_edge" in source


def test_config_has_no_authentic_lock_or_database_mutation() -> None:
    text = CONFIG.read_text().lower()
    assert "authentic_only" not in text
    assert "create table" not in text
    assert "insert into" not in text
    assert "update " not in text


def test_capacity_metrics_reprices_fee_for_each_vwap_quantity() -> None:
    _, raw = _load_config(CONFIG)
    row = {
        "side": "up",
        "won": True,
        "fee_rate": 0.10,
        **{f"up_ask_vwap_{quantity}": 0.5 for quantity in range(5, 201)},
        **{f"down_ask_vwap_{quantity}": 0.5 for quantity in range(5, 201)},
    }
    row["up_ask_vwap_10"] = 0.8
    metrics = _capacity_metrics(pl.DataFrame([row]), raw)
    assert abs(metrics["5"]["net_pnl"] - 2.35) < 1e-12
    assert abs(metrics["10"]["net_pnl"] - 1.79) < 1e-12
