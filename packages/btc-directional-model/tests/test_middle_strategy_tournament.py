from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from btc_directional_model.middle_strategy_data import (
    build_middle_panel,
    build_source_preserving_bridge_panel,
)
from btc_directional_model.middle_strategy_tournament import (
    ALL_NAMES,
    BridgeEnsembleCalibrator,
    BridgeTreeModel,
    _agreement_predictions,
    _candidate_contract,
    _counterfactual_economics,
    _fit_candidate_tree,
    _opportunities,
    _qualification_status,
    _select_trades,
    _validate_artifact,
    train_tournament,
)
from btc_directional_model.multivenue_early_entry_data import load_data_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_ROOT / "configs/btc-5m-middle-strategy-tournament-20260321-20260828.toml"
NORMALIZED_CONFIG = (
    PACKAGE_ROOT
    / "configs/btc-5m-twap-normalized-middle-strategy-tournament-20260321-20260828.toml"
)
BRIDGE_CONFIG = (
    PACKAGE_ROOT
    / "configs/btc-5m-source-preserving-settlement-bridge-middle-strategy-tournament-20260321-20260828.toml"
)


def test_frozen_contract_uses_full_range_and_five_challengers() -> None:
    config = load_data_config(CONFIG)
    assert config.source_start == datetime(2026, 3, 21, tzinfo=UTC)
    assert config.fit_end == datetime(2026, 8, 14, tzinfo=UTC)
    assert config.sealed_end == datetime(2026, 8, 29, tzinfo=UTC)
    assert tuple(row["name"] for row in config.raw["candidates"]) == ALL_NAMES
    assert config.raw["entry"]["economic_start_second"] == 90
    assert config.raw["entry"]["economic_end_second_inclusive"] == 179
    assert config.raw["sources"]["kraken_l2_included"] is False
    assert "authentic" not in CONFIG.read_text().lower()


def test_contract_rejects_twap_and_target_features() -> None:
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
        "execution": ["pm_depth_imbalance"],
    }
    contract = _candidate_contract(config, {"feature_groups": groups})
    assert tuple(contract) == ALL_NAMES
    for row in contract.values():
        assert not any("twap" in feature.lower() for feature in row.get("features", ()))


def test_agreement_ensemble_marks_disagreement_ineligible() -> None:
    start = datetime(2026, 8, 1, tzinfo=UTC)
    wide = pl.DataFrame(
        {
            "market_id": ["a", "b"],
            "window_start": [start, start + timedelta(minutes=5)],
            "observed_at": [start + timedelta(seconds=90), start + timedelta(minutes=6, seconds=30)],
            "seconds_elapsed": [90, 90],
            "label_up": [1, 0],
            "fold": ["x", "x"],
            "middle_specialist_refit": [0.7, 0.7],
            "middle_q5_admission": [0.8, 0.3],
            "crossvenue_middle_specialist": [0.6, 0.4],
        }
    )
    result = _agreement_predictions(wide)
    assert result["eligible_signal"].to_list() == [True, False]
    assert result.columns[-3:] == ["candidate", "probability", "eligible_signal"]


def test_economics_require_middle_window_and_two_second_books() -> None:
    config = load_data_config(CONFIG)
    start = datetime(2026, 8, 1, tzinfo=UTC)
    keys = {
        "market_id": ["fresh", "stale", "early"],
        "window_start": [start, start + timedelta(minutes=5), start + timedelta(minutes=10)],
        "observed_at": [start + timedelta(seconds=90), start + timedelta(minutes=6, seconds=30), start + timedelta(minutes=11)],
        "seconds_elapsed": [90, 90, 60],
    }
    predictions = pl.DataFrame(
        {**keys, "label_up": [1, 1, 1], "fold": ["x"] * 3, "candidate": ["x"] * 3,
         "probability": [0.8] * 3, "eligible_signal": [True] * 3}
    )
    panel = pl.DataFrame(
        {**keys, "up_ask_vwap_5": [0.5] * 3, "down_ask_vwap_5": [0.5] * 3,
         "fee_rate": [0.0] * 3, "pm_up_book_age_seconds": [1.0, 3.0, 1.0],
         "pm_down_book_age_seconds": [1.0, 3.0, 1.0]}
    )
    opportunities = _opportunities(predictions, panel, config)
    trades = _select_trades(opportunities, {"minimum_edge": 0.01, "minimum_confidence": 0.52, "maximum_share_cost": 0.95}, config)
    assert trades["market_id"].to_list() == ["fresh"]


def test_sealed_replay_is_opened_only_after_selection_freeze() -> None:
    source = inspect.getsource(train_tournament)
    assert source.index("selection_frozen_at =") < source.index("sealed = panel.filter")


def test_round_trip_validation_uses_absolute_package_source_path() -> None:
    source = inspect.getsource(_validate_artifact)
    assert 'str(config.package_root / "src")' in source


def test_bridge_estimators_have_stable_import_identity() -> None:
    assert BridgeTreeModel.__module__ == "btc_directional_model.middle_strategy_tournament"
    assert BridgeEnsembleCalibrator.__module__ == "btc_directional_model.middle_strategy_tournament"


def test_negative_expectancy_is_not_promoted() -> None:
    row = {"net_pnl": -1.0, "stress_net_pnl": -2.0, "profit_factor": 0.9}
    assert _qualification_status({"candidate": row}).endswith("negative_expectancy")


def test_optional_l2_builder_contract_preserves_base_panel() -> None:
    source = inspect.getsource(build_middle_panel)
    assert 'how="left"' in source
    assert "changed base market coverage" in source


def test_normalized_contract_has_three_pristine_chronological_blocks() -> None:
    config = load_data_config(NORMALIZED_CONFIG)
    assert config.source_start == datetime(2026, 3, 21, tzinfo=UTC)
    assert config.fit_end == datetime(2026, 8, 14, tzinfo=UTC)
    assert config.raw["windows"]["development_start"] == "2026-08-14T00:00:00Z"
    assert config.raw["windows"]["development_end"] == "2026-08-21T00:00:00Z"
    assert config.sealed_start == datetime(2026, 8, 21, tzinfo=UTC)
    assert config.sealed_end == datetime(2026, 8, 29, tzinfo=UTC)
    assert config.raw["entry"]["start_second"] == 60
    assert config.raw["entry"]["end_second_inclusive"] == 180
    assert tuple(row["name"] for row in config.raw["candidates"]) == ALL_NAMES
    assert "authentic_only" not in NORMALIZED_CONFIG.read_text().lower()


def test_normalized_weights_are_nonzero_and_applied_to_both_model_layers() -> None:
    config = load_data_config(NORMALIZED_CONFIG)
    normalization = config.raw["normalization"]
    assert min(
        normalization["refprice_minimum_weight"],
        normalization["binance_minimum_weight"],
        normalization["fallback_weight"],
    ) > 0
    source = inspect.getsource(_fit_candidate_tree)
    assert source.count("sample_weight=") >= 2


def test_exact_label_query_uses_existing_archive_and_live_sources() -> None:
    query = (PACKAGE_ROOT / "sql/btc-normalized-twap60-label-source.sql").read_text()
    assert "market_data.pmdata_chainlink_btcusd_twap" in query
    assert "market_data.polymarket_chainlink_btcusd_twap" in query
    assert "archive_precedes_live" not in query


def test_normalized_policy_freezes_before_sealed_test_is_opened() -> None:
    source = inspect.getsource(train_tournament)
    assert source.index("development = panel.filter") < source.index("selection_frozen_at =")
    assert source.index("selection_frozen_at =") < source.index("sealed = panel.filter")


def test_bridge_contract_preserves_full_range_and_pristine_splits() -> None:
    config = load_data_config(BRIDGE_CONFIG)
    assert config.source_start == datetime(2026, 3, 21, tzinfo=UTC)
    assert config.fit_end == datetime(2026, 8, 14, tzinfo=UTC)
    assert config.sealed_start == datetime(2026, 8, 21, tzinfo=UTC)
    assert config.sealed_end == datetime(2026, 8, 29, tzinfo=UTC)
    assert config.raw["settlement_bridge"]["calibration_end"] == "2026-08-08T00:00:00Z"
    assert config.raw["settlement_bridge"]["validation_end"] == config.raw["windows"][
        "fit_end"
    ]
    assert config.raw["folds"][-1]["test_start"] == config.raw["settlement_bridge"][
        "calibration_end"
    ]
    assert tuple(row["name"] for row in config.raw["candidates"]) == ALL_NAMES


def test_bridge_builder_preserves_sources_and_never_uses_sealed_calibration() -> None:
    source = inspect.getsource(build_source_preserving_bridge_panel)
    assert '"official_label_up"' in source
    assert '"authentic_margin_bps"' in source
    assert '"proxy_margin_bps"' in source
    assert '"binance_raw_margin_bps"' in source
    assert 'calibration_end = datetime.fromisoformat(bridge["calibration_end"])' in source
    assert 'closed="left"' in source
    assert '"source_values_remain_distinct": True' in source


def test_bridge_model_uses_probabilistic_target_without_doubling_rows() -> None:
    source = inspect.getsource(_fit_candidate_tree)
    assert "HistGradientBoostingRegressor" in source
    assert 'fit["bridge_probability_target"]' in source
    assert "np.vstack" not in source


def test_prescribed_counterfactuals_are_reporting_only() -> None:
    source = inspect.getsource(_counterfactual_economics)
    assert 'is_between(150, 180' in source
    assert 'pl.col("selected_probability") >= 0.80' in source
    training = inspect.getsource(train_tournament)
    assert training.index("selection_frozen_at =") < training.index(
        "counterfactual_economic[name]"
    )
