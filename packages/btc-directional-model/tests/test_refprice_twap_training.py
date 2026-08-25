from __future__ import annotations

from pathlib import Path

import polars as pl

from btc_directional_model.continuous_edge_training import VWAP_QUANTITIES
from btc_directional_model.refprice_twap_training import (
    ARMS,
    FAMILIES,
    TWAP_INPUT_FEATURES,
    _apply_frozen_policy,
    load_config,
)

PACKAGE_ROOT = Path(__file__).parents[1]
CONFIG_PATH = (
    PACKAGE_ROOT / "configs" / "btc-5m-refprice-twap-target-training-20260607-20260824.toml"
)


def test_frozen_config_keeps_the_scope_and_watermark_contract() -> None:
    config = load_config(CONFIG_PATH)

    assert ARMS == ("R", "T", "RT")
    assert len(ARMS) * len(FAMILIES) == 15
    assert config.windows.watermark.isoformat() == "2026-08-24"
    assert config.execution.quantities == VWAP_QUANTITIES
    assert config.execution.training_cadence_seconds == 5
    assert config.execution.refprice_freshness_seconds == 5
    assert config.execution.maximum_depth_participation == 0.25


def test_sql_contract_is_pmdata_primary_causal_and_read_only() -> None:
    sql_root = PACKAGE_ROOT / "sql"
    refprice = (sql_root / "btc-refprice-twap-refprice-source.sql").read_text()
    label = (sql_root / "btc-refprice-twap-label-source.sql").read_text()
    twap_input = (sql_root / "btc-refprice-twap-input-source.sql").read_text()
    capacity = (sql_root / "btc-refprice-twap-capacity-source.sql").read_text()
    all_sql = "\n".join(
        path.read_text() for path in sql_root.glob("btc-refprice-twap-*-source.sql")
    ).lower()

    assert "market_data.chainlink_btcusd_reference_prices" in refprice
    assert "pmdata_chainlink_streams" in refprice
    assert "received_at" in refprice
    assert "market_data.pmdata_chainlink_btcusd_twap" in label
    assert "window_seconds = 60" in label
    assert "source_timestamp = market.window_start" in label
    assert "source_timestamp = market.window_end" in label
    assert "market_data.pmdata_chainlink_btcusd_twap" in twap_input
    assert "provider_received_at" in twap_input
    assert "window_seconds IN (30, 60)" in twap_input
    assert len(TWAP_INPUT_FEATURES) == 13
    assert "btc5m-capacity-book-1-240s-v2" in capacity
    assert "btc5m-capacity-local-orderbook-1-240s-v1" in capacity
    for quantity in VWAP_QUANTITIES:
        assert f"up_ask_vwap_{quantity}" in capacity
        assert f"down_ask_vwap_{quantity}" in capacity
    for mutation in ("insert ", "update ", "delete ", "alter ", "create "):
        assert mutation not in all_sql


def test_admission_is_symmetric_and_has_no_streak_rule() -> None:
    config = load_config(CONFIG_PATH)
    frame = pl.DataFrame(
        {
            "market_id": ["up", "down"],
            "window_start": [
                config.windows.calibration_end,
                config.windows.calibration_end,
            ],
            "observed_at": [
                config.windows.calibration_end,
                config.windows.calibration_end,
            ],
            "seconds_elapsed": [120, 120],
            "predicted_up": [True, False],
            "model_confidence": [0.90, 0.90],
            "model_stress_edge": [0.10, 0.10],
            "admission_probability": [0.90, 0.90],
            "payoff_expected_stress_edge": [0.10, 0.10],
        }
    )

    selected = _apply_frozen_policy(frame, "stratified_payoff", config)

    assert selected["market_id"].to_list() == ["down", "up"]
    assert "streak" not in str(config.policies).lower()
