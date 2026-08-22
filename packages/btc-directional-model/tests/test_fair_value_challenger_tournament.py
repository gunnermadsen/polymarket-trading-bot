from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from btc_directional_model.fair_value_challenger_tournament import (
    CHALLENGERS,
    EXOGENOUS_FEATURES,
    _apply_policy,
    _attach_entry_cells,
    _fresh_readiness,
    _terminal_margin_targets,
    load_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PACKAGE_ROOT / "configs" / "btc-5m-fair-value-challenger-tournament-20260525-20260802.toml"
)


def test_contract_is_offline_chronological_and_complete() -> None:
    config = load_config(CONFIG_PATH)

    assert config.fresh_holdout is False
    assert len(config.folds) == 5
    assert len(CHALLENGERS) == 6
    assert config.folds[0].evaluation_start == datetime(2026, 6, 29, tzinfo=UTC)
    assert config.folds[-1].evaluation_end == datetime(2026, 8, 2, tzinfo=UTC)
    assert config.source.execution.quantities[-1] == 200


def test_fair_value_features_exclude_polymarket_price_and_book_state() -> None:
    forbidden = ("ask_vwap", "pm_", "book_", "selected_cost", "fee_rate")

    assert not [name for name in EXOGENOUS_FEATURES if name.startswith(forbidden)]


def test_terminal_margin_target_uses_last_causal_row_per_market() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "b", "a", "b"],
            "seconds_elapsed": [15, 15, 240, 240],
            "observed_at": [1, 1, 2, 2],
            "btc_path_from_window_open_bps": [1.0, -2.0, 7.0, -9.0],
        }
    )

    assert _terminal_margin_targets(frame).tolist() == [7.0, -9.0, 7.0, -9.0]


def test_cell_policy_takes_first_crossing_across_cells() -> None:
    frame = _attach_entry_cells(
        pl.DataFrame(
            {
                "market_id": ["a", "a", "b"],
                "seconds_elapsed": [30, 100, 130],
                "observed_at": [1, 2, 3],
                "candidate_confidence": [0.75, 0.90, 0.80],
                "candidate_edge": [0.02, 0.04, 0.03],
                "candidate_rank": [0.0, 0.0, 0.0],
            }
        )
    )
    policy = {
        "type": "cell_first_crossing",
        "cells": {
            "early_15_89": {"enabled": True, "confidence": 0.7, "edge": 0.0, "rank": -1.0},
            "middle_90_119": {"enabled": True, "confidence": 0.7, "edge": 0.0, "rank": -1.0},
            "middle_120_149": {"enabled": True, "confidence": 0.7, "edge": 0.0, "rank": -1.0},
            "middle_150_179": {"enabled": False},
            "late_180_240": {"enabled": False},
        },
    }

    selected = _apply_policy(frame, policy)

    assert selected["market_id"].to_list() == ["a", "b"]
    assert selected["seconds_elapsed"].to_list() == [30, 130]


def test_fresh_holdout_remains_unopened_and_not_ready() -> None:
    readiness = _fresh_readiness(load_config(CONFIG_PATH))

    assert readiness["labels_accessed"] is False
    assert readiness["strict_markets"] == 600
    assert readiness["passed"] is False
