from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from btc_directional_model.extended_specialist_strategy_tournament import (
    ALL_CANDIDATES,
    CalibratedClassifier,
    DistilledSpecialist,
    DualHeadAdmission,
    ResidualSpecialist,
    _bands,
    _candidate_contract,
    _load_config,
    _realized_opportunities,
    _select_dual_policy,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PACKAGE_ROOT
    / "configs/btc-5m-extended-specialist-strategy-tournament-20260321-20260826.toml"
)


def test_serialized_model_classes_use_importable_module_identity() -> None:
    expected = "btc_directional_model.extended_specialist_strategy_tournament"
    assert {
        CalibratedClassifier.__module__,
        DistilledSpecialist.__module__,
        ResidualSpecialist.__module__,
        DualHeadAdmission.__module__,
    } == {expected}


def test_frozen_roster_and_chronological_split() -> None:
    _, raw = _load_config(CONFIG)
    assert tuple(row["name"] for row in raw["candidates"]) == ALL_CANDIDATES
    assert datetime.fromisoformat(raw["windows"]["fit_end"]) == datetime(
        2026, 8, 14, tzinfo=UTC
    )
    assert datetime.fromisoformat(raw["windows"]["sealed_start"]) == datetime(
        2026, 8, 20, tzinfo=UTC
    )
    assert datetime.fromisoformat(raw["windows"]["sealed_end"]) == datetime(
        2026, 8, 26, tzinfo=UTC
    )
    assert _bands(raw) == (
        ("60_89", 60, 89),
        ("90_119", 90, 119),
        ("120_149", 120, 149),
        ("150_180", 150, 180),
    )


def test_settlement_agnostic_contract_removes_boundary_and_settlement_features() -> None:
    _, raw = _load_config(CONFIG)
    manifest = {
        "feature_groups": {
            "core": ["btc_return_5s_bps", "btc_boundary_terminal_volatility_z"],
            "candles": ["chainlink_candle_return_5m_bps"],
            "oracle": ["oracle_return_from_window_open_bps"],
            "refprice": ["chainlink_ref_return_5s_bps"],
            "open_interest": ["binance_oi_change_5m_bps"],
            "binance_prints": ["binance_print_return_5s_bps"],
            "kraken": ["kraken_return_5s_bps"],
            "spot_l2": ["spot_l2_imbalance_20"],
            "kraken_l2": ["kraken_l2_update_imbalance_30s"],
        }
    }
    contract = _candidate_contract(raw, manifest)
    features = contract["settlement_agnostic_trajectory"]["features"]
    assert "btc_return_5s_bps" in features
    assert "btc_boundary_terminal_volatility_z" not in features
    assert all("oracle" not in name and "ref" not in name for name in features)


def test_dual_head_uses_enter_now_outcome_without_best_later_target() -> None:
    source = inspect.getsource(_select_dual_policy)
    realized = inspect.getsource(_realized_opportunities)
    assert "best_later" not in source
    assert "best_later" not in realized
    assert "realized_stress_edge" in realized


def test_config_contains_no_authentic_only_lock_or_database_mutation() -> None:
    text = CONFIG.read_text().lower()
    assert "authentic_only" not in text
    assert "create table" not in text
    assert "insert into" not in text
    assert "update " not in text


def test_realized_opportunity_target_is_current_trade_economics() -> None:
    _, raw = _load_config(CONFIG)
    predictions = pl.DataFrame(
        {
            "market_id": ["a"],
            "window_start": [datetime(2026, 8, 1, tzinfo=UTC)],
            "observed_at": [datetime(2026, 8, 1, 0, 1, tzinfo=UTC)],
            "seconds_elapsed": [60],
            "label_up": [1],
            "fold": ["x"],
            "candidate": ["x"],
            "probability": [0.7],
            "eligible_signal": [True],
        }
    )
    panel = pl.DataFrame(
        {
            "market_id": ["a"],
            "window_start": [datetime(2026, 8, 1, tzinfo=UTC)],
            "observed_at": [datetime(2026, 8, 1, 0, 1, tzinfo=UTC)],
            "seconds_elapsed": [60],
            "up_ask_vwap_5": [0.60],
            "down_ask_vwap_5": [0.41],
            "fee_rate": [0.0],
            "pm_up_book_age_seconds": [1.0],
            "pm_down_book_age_seconds": [1.0],
        }
    )
    result = _realized_opportunities(predictions, panel, raw)
    assert result["realized_stress_edge"][0] == 0.385
