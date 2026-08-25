from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from btc_directional_model.continuous_edge_training import BOOK_RAW_FEATURES
from btc_directional_model.twap60_challenger_tournament import (
    FEATURE_TREATMENTS,
    _matrix,
    _model_eligible,
    feature_names,
    load_config,
    predetermined_hyperparameters,
)
from btc_directional_model.twap60_training_data import REFPRICE_RUNTIME_FEATURES

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PACKAGE_ROOT
    / "configs"
    / "btc-5m-twap60-challenger-tournament-20260607-20260825.toml"
)


def test_tournament_contract_freezes_regimes_folds_and_execution() -> None:
    config = load_config(CONFIG)

    assert config.authentic_start == datetime(2026, 8, 1, tzinfo=UTC)
    assert config.transition_start == datetime(2026, 8, 7, tzinfo=UTC)
    assert config.current_start == datetime(2026, 8, 14, tzinfo=UTC)
    assert config.end == datetime(2026, 8, 25, tzinfo=UTC)
    assert len(config.folds) == 6
    assert next(iter(config.raw["execution"]["quantities"])) == 5
    assert tuple(config.raw["execution"]["quantities"])[-1] == 200
    assert config.raw["training"]["paper_only"] is True
    assert config.raw["training"]["live_capital_allowed"] is False


def test_search_is_exactly_36_unique_non_cartesian_rows() -> None:
    config = load_config(CONFIG)

    rows = predetermined_hyperparameters(config)

    assert len(rows) == 36
    assert len(set(rows)) == 36
    assert {row.learning_rate for row in rows} == {0.02, 0.04, 0.06}
    assert {row.max_leaf_nodes for row in rows} == {7, 15, 31}


def test_bakeoff_treatments_keep_historical_share_prices_out_of_outcome_fit() -> None:
    for treatment in FEATURE_TREATMENTS:
        assert not set(feature_names(treatment)) & set(BOOK_RAW_FEATURES)
    assert set(REFPRICE_RUNTIME_FEATURES).issubset(feature_names("refprice_path"))
    assert set(REFPRICE_RUNTIME_FEATURES).issubset(
        feature_names("refprice_oracle_candle_combined")
    )


def test_outcome_matrix_retains_early_rows_with_unavailable_long_lookbacks() -> None:
    frame = pl.DataFrame(
        {
            "short_feature": [1.0, 2.0],
            "long_lookback_feature": [None, None],
            "refprice_causal_eligible": [True, True],
        }
    )
    features = ("short_feature", "long_lookback_feature")

    eligible = _model_eligible(frame, features)
    matrix = _matrix(eligible, features)

    assert eligible.height == 2
    assert np.isnan(matrix[:, 1]).all()
