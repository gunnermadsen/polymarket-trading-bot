from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from btc_directional_model.continuous_edge_training import VWAP_QUANTITIES
from btc_directional_model.multivenue_early_entry_data import load_data_config
from btc_directional_model.vwap_curve_admission_tournament import (
    ADMISSION_MODES,
    TARGET_ONLY_COLUMNS,
    _attach_curve_features,
    _attach_enter_now_target,
    _features,
    _mode_replays,
    build_vwap_panel,
    train_tournament,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PACKAGE_ROOT
    / "configs/btc-5m-full-august-vwap-admission-tournament-20260321-20260901.toml"
)


def test_frozen_split_uses_pre_august_fit_and_full_august_holdout() -> None:
    config = load_data_config(CONFIG)
    assert config.source_start == datetime(2026, 3, 21, tzinfo=UTC)
    assert config.fit_end == datetime(2026, 8, 1, tzinfo=UTC)
    assert config.sealed_start == datetime(2026, 8, 1, tzinfo=UTC)
    assert config.sealed_end == datetime(2026, 9, 1, tzinfo=UTC)
    assert len(config.raw["folds"]) == 4
    assert all(
        datetime.fromisoformat(fold["test_end"]) <= config.fit_end
        for fold in config.raw["folds"]
    )


def test_contract_preserves_seven_candidates_without_bridge_or_authentic_lock() -> None:
    text = CONFIG.read_text().lower()
    config = load_data_config(CONFIG)
    assert len(config.raw["candidates"]) == 7
    assert "settlement_bridge" not in config.raw
    assert "normalization" not in config.raw
    assert "authentic_only" not in text
    assert set(ADMISSION_MODES) == {
        "hybrid_vwap5",
        "hybrid_full_vwap_no_l2",
        "hybrid_full_vwap_dual_l2",
    }


def test_full_curve_engineering_uses_every_frozen_vwap_quantity() -> None:
    values: dict[str, list[object]] = {
        "side": ["up", "down"],
        "pm_up_depth_log": [2.0, 3.0],
        "pm_down_depth_log": [1.0, 2.0],
        "pm_depth_imbalance": [0.2, -0.3],
        "pm_up_book_age_seconds": [0.5, None],
        "pm_down_book_age_seconds": [0.6, None],
    }
    for quantity in VWAP_QUANTITIES:
        values[f"up_ask_vwap_{quantity}"] = [0.50 + quantity / 10_000, 0.55]
        values[f"down_ask_vwap_{quantity}"] = [0.52, 0.45 + quantity / 10_000]
    result = _attach_curve_features(pl.DataFrame(values))
    for quantity in VWAP_QUANTITIES:
        assert f"selected_vwap_{quantity}" in result.columns
        assert f"opposite_vwap_{quantity}" in result.columns
        assert f"vwap_overround_{quantity}" in result.columns
    assert "selected_curve_convexity" in result.columns
    assert "capacity_slippage_imbalance" in result.columns


def test_enter_now_target_compares_only_with_later_rows_in_same_market() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "a", "b", "b"],
            "seconds_elapsed": [60, 65, 70, 60, 65],
            "realized_stress_edge": [0.10, 0.30, -0.20, -0.10, 0.20],
        }
    )
    result = _attach_enter_now_target(frame)
    assert result.filter(pl.col("market_id") == "a")[
        "realized_enter_now_advantage"
    ].to_list() == [-0.19999999999999998, 0.3, -0.2]
    assert result.filter(pl.col("market_id") == "b")[
        "realized_enter_now_advantage"
    ].to_list() == [-0.30000000000000004, 0.2]


def test_full_curve_and_l2_are_admission_only_and_targets_are_forbidden() -> None:
    frame = pl.DataFrame(
        {
            "probability": [0.6, 0.7],
            "selected_probability": [0.6, 0.7],
            "share_cost": [0.5, 0.6],
            "selected_vwap_5": [0.5, 0.6],
            "selected_vwap_200": [0.55, 0.70],
            "vwap_overround_200": [0.03, 0.05],
            "spot_l2_imbalance_20": [0.1, 0.2],
            "kraken_l2_update_imbalance_30s": [0.2, 0.3],
            "realized_stress_edge": [0.1, -0.2],
            "label_up": [1, 0],
        }
    )
    no_l2 = _features(frame, "hybrid_full_vwap_no_l2")
    dual = _features(frame, "hybrid_full_vwap_dual_l2")
    assert "selected_vwap_200" in no_l2
    assert "spot_l2_imbalance_20" not in no_l2
    assert "spot_l2_imbalance_20" in dual
    assert "kraken_l2_update_imbalance_30s" in dual
    assert not set(TARGET_ONLY_COLUMNS) & set(dual)
    assert "realized_stress_edge" not in dual
    assert "label_up" not in dual


def test_each_time_band_is_replayed_independently_before_combined_selection() -> None:
    source = inspect.getsource(_mode_replays)
    assert "for band, start, end in mst._calibration_cells(config)" in source
    assert 'output["combined_60_180"]' in source
    assert "is_between(start, end" in source


def test_resume_and_scope_contracts_are_explicit() -> None:
    panel_source = inspect.getsource(build_vwap_panel)
    training_source = inspect.getsource(train_tournament)
    assert "resume split differs from its frozen checkpoint" in training_source
    assert "admission-{candidate}-{mode}.joblib" in training_source
    assert "heldout_metrics_accessed" in training_source
    assert '"database_mutations": False' in panel_source
    assert '"new_tables": False' in panel_source
    assert '"new_ingesters": False' in panel_source
