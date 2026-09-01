from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from btc_directional_model.hybrid_admission_tournament import (
    _candidate_contract,
    _features,
    _hybrid_trades,
    build_hybrid_panel,
)
from btc_directional_model.kraken_l2_training_data import (
    KRAKEN_L2_FEATURES,
    _daily_seconds,
    attach_kraken_l2,
)
from btc_directional_model.multivenue_early_entry_data import load_data_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_ROOT / "configs/btc-5m-hybrid-payoff-admission-tournament-20260321-20260827.toml"


def test_contract_uses_full_history_seven_candidates_and_no_authentic_lock() -> None:
    config = load_data_config(CONFIG)
    assert config.source_start == datetime(2026, 3, 21, tzinfo=UTC)
    assert config.fit_end == datetime(2026, 8, 21, tzinfo=UTC)
    assert config.sealed_start == datetime(2026, 8, 21, tzinfo=UTC)
    assert config.sealed_end == datetime(2026, 8, 27, tzinfo=UTC)
    assert len(config.raw["candidates"]) == 7
    assert "authentic_only" not in CONFIG.read_text().lower()


def test_directional_contract_adds_kraken_l2_only_to_prescribed_bases() -> None:
    config = load_data_config(CONFIG)
    groups = {
        "core": ["btc_return_30s_bps"],
        "candles": ["chainlink_candle_return_5m_bps"],
        "oracle": ["oracle_round_age_seconds_scaled"],
        "refprice": ["chainlink_ref_return_30s_bps"],
        "open_interest": ["binance_oi_change_15m_bps"],
        "binance_prints": ["binance_print_signed_share_30s"],
        "kraken": ["kraken_print_signed_share_30s"],
        "spot_l2": ["spot_l2_imbalance_20"],
        "kraken_l2": ["kraken_l2_update_imbalance_30s"],
    }
    contract = _candidate_contract(config, {"feature_groups": groups})
    assert "kraken_l2_update_imbalance_30s" in contract["external_flow_only_control"]["features"]
    assert "kraken_l2_update_imbalance_30s" not in contract["middle_specialist_refit"]["features"]
    assert "kraken_l2_update_imbalance_30s" not in contract["middle_q5_admission"]["features"]
    assert "kraken_l2_update_imbalance_30s" in contract["crossvenue_middle_specialist"]["features"]


def test_kraken_l2_availability_is_delayed_to_next_second(tmp_path: Path) -> None:
    source = tmp_path / "hour.parquet"
    pl.DataFrame(
        {
            "event_time": [1_000_000_001, 1_500_000_000, 2_000_000_000],
            "side": ["bid", "ask", "bid"],
            "price": [100.0, 101.0, 100.5],
            "quantity": [2.0, 1.0, 0.0],
        }
    ).write_parquet(source)
    result = _daily_seconds([source])
    assert result["available_ns"].min() == 2_000_000_000
    assert result["available_ns"].max() == 3_000_000_000


def test_optional_kraken_l2_left_join_preserves_markets() -> None:
    source = inspect.getsource(attach_kraken_l2)
    builder = inspect.getsource(build_hybrid_panel)
    assert 'how="left"' in source
    assert "changed market coverage" in source
    assert '"authentic_only_filter": False' in builder


def test_every_hybrid_veto_can_use_dual_l2_but_never_settlement_features() -> None:
    frame = pl.DataFrame(
        {
            "probability": [0.6, 0.7],
            "selected_probability": [0.6, 0.7],
            "share_cost": [0.5, 0.6],
            "spot_l2_imbalance_20": [0.1, 0.2],
            "kraken_l2_update_imbalance_30s": [0.2, 0.3],
            "twap60": [1.0, 2.0],
            "label_up": [1, 0],
        }
    )
    features = _features(frame, uses_l2=True)
    assert "spot_l2_imbalance_20" in features
    assert "kraken_l2_update_imbalance_30s" in features
    assert not any("twap" in name or "label" in name for name in features)


def test_hybrid_veto_never_changes_predicted_side() -> None:
    source = inspect.getsource(_hybrid_trades)
    assert ".filter(" in source
    assert 'alias("side")' not in source
    assert "_select_trades" in source


def test_kraken_feature_contract_is_update_flow_not_full_depth() -> None:
    assert all("depth" not in name for name in KRAKEN_L2_FEATURES)
