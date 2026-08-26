from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from btc_directional_model.counterfactual_twap_state_data import (
    CHAINLINK_UNCERTAINTY_BPS,
    construct_binance_labels,
)
from btc_directional_model.counterfactual_twap_state_tournament import (
    CANDIDATE_NAMES,
    FEATURE_TREATMENTS,
    HISTORY_ARMS,
    feature_names,
    load_config,
    predetermined_hyperparameters,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_ROOT / "configs/btc-5m-counterfactual-twap-state-20260321-20260826.toml"


def test_contract_is_new_training_only_family_with_exact_schedule() -> None:
    config = load_config(CONFIG)

    assert config.model_family == "btc-5m-counterfactual-twap-state"
    assert config.start == datetime(2026, 3, 21, tzinfo=UTC)
    assert config.chainlink_start == datetime(2026, 6, 7, tzinfo=UTC)
    assert config.authentic_start == datetime(2026, 8, 1, tzinfo=UTC)
    assert config.current_start == datetime(2026, 8, 14, tzinfo=UTC)
    assert config.candidate_freeze == datetime(2026, 8, 25, tzinfo=UTC)
    assert config.raw["training"]["training_only"] is True
    assert config.raw["training"]["live_capital_allowed"] is False
    assert tuple(tuple(value) for value in config.raw["entry"]["cells"]) == (
        (60, 90), (90, 120), (120, 150), (150, 180)
    )


def test_tournament_dimensions_and_search_are_frozen() -> None:
    config = load_config(CONFIG)

    assert FEATURE_TREATMENTS == (
        "refprice_control", "twap30", "twap60", "dual_twap", "combined",
        "combined_disagreement",
    )
    assert len(HISTORY_ARMS) == 5
    assert len(CANDIDATE_NAMES) == 6
    assert len(predetermined_hyperparameters(config)) == 36
    assert len(set(predetermined_hyperparameters(config))) == 36
    assert not set(feature_names("combined_disagreement")) & {
        "label_source", "label_up", "target_margin_bps", "window_start"
    }


def test_binance_twap_uses_completed_left_closed_right_open_closes() -> None:
    start = datetime(2026, 3, 21, tzinfo=UTC)
    times = [start - timedelta(seconds=60) + timedelta(seconds=i) for i in range(360)]
    binance = pl.DataFrame(
        {
            "market_id": ["m"] * 360,
            "window_start": [start] * 360,
            "window_end": [start + timedelta(minutes=5)] * 360,
            "open_timestamp": times,
            "available_at": [value + timedelta(seconds=1) for value in times],
            "close_price": np.arange(100.0, 460.0),
        }
    )
    labels = pl.DataFrame(
        {
            "market_id": ["m"], "window_start": [start],
            "window_end": [start + timedelta(minutes=5)],
        }
    )

    result = construct_binance_labels(labels, binance)

    assert result["binance_open_prints"].item() == 60
    assert result["binance_close_prints"].item() == 60
    assert result["binance_open_twap60"].item() == np.mean(np.arange(100.0, 160.0))
    assert result["binance_close_twap60"].item() == np.mean(np.arange(400.0, 460.0))


def test_source_queries_are_bounded_select_only_and_add_no_schema() -> None:
    names = (
        "btc-twap60-label-source.sql", "btc-twap60-refprice-source.sql",
        "btc-twap60-core-current-source.sql", "btc-core-oracle-source.sql",
        "btc-twap60-candle-source.sql", "btc-capacity-execution-evidence.sql",
        "btc-counterfactual-twap-binance-source.sql",
    )
    forbidden = ("insert ", "update ", "delete ", "create ", "alter ", "drop ")
    for name in names:
        sql = (PACKAGE_ROOT / "sql" / name).read_text().lower()
        assert "select" in sql
        assert not any(token in sql for token in forbidden)
        assert "batch_start" in sql or "history_start" in sql
        assert "batch_end" in sql or "range_end" in sql


def test_chainlink_uncertainty_exclusion_boundary_is_not_lowered() -> None:
    assert CHAINLINK_UNCERTAINTY_BPS == 0.526
