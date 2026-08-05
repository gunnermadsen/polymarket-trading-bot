from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from btc_directional_model.asymmetric_value_benchmark import (
    _calibration_report_summary,
    _candidate_frames,
    _candidate_grid_summary,
    _evaluation_economics_table,
    _frame_content_digest,
    _matched_control_noninferiority_checks,
    _selected_matched_control,
    _validate_external_core_keys,
)
from btc_directional_model.asymmetric_value_config import (
    load_asymmetric_value_config,
)
from btc_directional_model.asymmetric_value_training import (
    ASYMMETRIC_VALUE_CANDIDATES,
    CORE_CANDLES_PRICE,
    CORE_L2_PRICE,
    CORE_ORACLE_PRICE,
    CORE_PRICE,
    L2_MATCHED_CORE_PRICE_CONTROL,
    ORACLE_MATCHED_CORE_PRICE_CONTROL,
    PRICE_LOGISTIC,
)


def _core_frame() -> pl.DataFrame:
    start = datetime(2026, 7, 16, tzinfo=UTC)
    return pl.DataFrame(
        {
            "market_id": ["a", "a"],
            "window_start": [start, start],
            "observed_at": [
                start + timedelta(seconds=5),
                start + timedelta(seconds=10),
            ],
            "seconds_elapsed": [5, 10],
            "label_up": [1, 1],
            "btc_close": [100_001.0, 100_002.0],
        }
    )


def test_core_content_digest_is_order_invariant_and_value_sensitive() -> None:
    core = _core_frame()
    reversed_core = core.reverse()
    changed = core.with_columns(
        pl.when(pl.col("seconds_elapsed") == 10)
        .then(999.0)
        .otherwise(pl.col("btc_close"))
        .alias("btc_close")
    )

    assert _frame_content_digest(core) == _frame_content_digest(reversed_core)
    assert _frame_content_digest(core) != _frame_content_digest(changed)


def test_external_cache_validation_binds_inherited_core_values() -> None:
    core = _core_frame()
    external = core.with_columns(pl.lit(1.0).alias("external_feature"))
    changed = external.with_columns(pl.lit(999.0).alias("btc_close"))

    _validate_external_core_keys(external, core)
    with pytest.raises(RuntimeError, match="core values changed"):
        _validate_external_core_keys(changed, core)


def test_selected_enriched_models_have_predeclared_matched_controls() -> None:
    assert _selected_matched_control(CORE_ORACLE_PRICE) == (
        ORACLE_MATCHED_CORE_PRICE_CONTROL
    )
    assert _selected_matched_control(CORE_L2_PRICE) == (
        L2_MATCHED_CORE_PRICE_CONTROL
    )
    assert _selected_matched_control(CORE_CANDLES_PRICE) == CORE_PRICE


def test_candidate_frames_predeclare_exact_seven_models() -> None:
    frame = _core_frame()

    candidates = _candidate_frames(
        price=frame,
        l2_price=frame,
        candle_price=frame,
        oracle_price=frame,
    )

    assert tuple(candidates) == ASYMMETRIC_VALUE_CANDIDATES


def test_evaluation_economics_table_includes_every_predeclared_model() -> None:
    metrics = {
        name: {
            "trades": 1,
            "resolved_markets": 10,
            "strict_executable_markets": 8,
            "strict_market_coverage": 0.8,
            "trades_per_resolved_market": 0.1,
            "net_profit_per_resolved_market": float(index) / 10.0,
            "net_expectancy_per_trade": float(index),
            "utc_day_block_bootstrap": {
                "net_expectancy_per_trade": {
                    "lower_95": float(index) - 1.0,
                    "upper_95": float(index) + 1.0,
                }
            },
        }
        for index, name in enumerate(ASYMMETRIC_VALUE_CANDIDATES)
    }

    table = _evaluation_economics_table(
        metrics,
        selected_model=CORE_PRICE,
        policy="raw20_30_by55_edge_3c",
    )

    assert {row["model"] for row in table} == set(ASYMMETRIC_VALUE_CANDIDATES)
    assert sum(row["selected_on_policy_window"] for row in table) == 1
    assert table[0]["net_profit_per_resolved_market"] == pytest.approx(0.6)
    assert all(row["strict_market_coverage"] == 0.8 for row in table)


def test_sparse_enriched_arm_must_beat_its_same_key_control() -> None:
    config = load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-one-second-20260414-20260802.toml"
    )
    policy = next(item for item in config.policies if item.selection_eligible)
    metrics = {
        f"{name}::{policy.name}": {
            "net_expectancy_per_trade": 0.10,
            "net_profit_per_resolved_market": 0.05,
        }
        for name in ASYMMETRIC_VALUE_CANDIDATES
    }
    metrics[f"{CORE_L2_PRICE}::{policy.name}"] = {
        "net_expectancy_per_trade": -0.10,
        "net_profit_per_resolved_market": -0.05,
    }
    empty = pl.DataFrame(
        schema={"window_start": pl.Datetime("us", "UTC"), "realized_net": pl.Float64}
    )
    ledgers = {
        f"{name}::{policy.name}": empty
        for name in ASYMMETRIC_VALUE_CANDIDATES
    }
    ledgers[f"{CORE_L2_PRICE}::{policy.name}"] = pl.DataFrame(
        {
            "window_start": [config.policy.start],
            "realized_net": [-1.0],
        }
    )

    checks, eligible = _matched_control_noninferiority_checks(
        metrics,
        ledgers,
        policy.name,
        config,
    )

    assert CORE_L2_PRICE not in eligible
    assert CORE_PRICE in eligible
    assert PRICE_LOGISTIC in eligible
    assert not all(check["passed"] for check in checks[CORE_L2_PRICE])


def test_candidate_grid_materializes_missing_prediction_seconds() -> None:
    config = load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-one-second-20260414-20260802.toml"
    )
    core = _core_frame()

    summary = _candidate_grid_summary(core, core, config)

    by_second = {
        row["seconds_elapsed"]: row for row in summary["by_second"]
    }
    assert by_second[5]["markets"] == 1
    assert by_second[15]["markets"] == 0
    assert summary["minimum_second_market_coverage"] == 0.0


def test_calibration_report_summary_discloses_parent_and_cell_fallbacks() -> None:
    parent = {
        "converged": True,
        "slope": 1.1,
        "rows": 500,
        "markets": 100,
    }
    fallback = {
        "fitted": False,
        "fallback": "insufficient_markets+insufficient_utc_days",
        "utc_days": 3,
    }
    fitted = {"fitted": True, "fallback": None, "utc_days": 5}
    training = {
        "profiles": {
            "first": {
                "calibration_bands": [parent],
                "side_price_time_calibration": {
                    "minimum_utc_days_per_cell": 5,
                    "cells": [fallback, fitted],
                },
            },
            "second": {
                "calibration_bands": [
                    {**parent, "rows": 600, "markets": 120}
                ],
                "side_price_time_calibration": {
                    "minimum_utc_days_per_cell": 5,
                    "cells": [fallback],
                },
            },
        }
    }

    summary = _calibration_report_summary(training)

    assert summary["valid_parent_calibrators"] == 2
    assert summary["parent_calibrators"] == 2
    assert summary["minimum_parent_rows"] == 500
    assert summary["maximum_parent_rows"] == 600
    assert summary["minimum_parent_markets"] == 100
    assert summary["maximum_parent_markets"] == 120
    assert summary["fitted_cells"] == 1
    assert summary["fallback_cells"] == 2
    assert summary["maximum_fallback_cell_utc_days"] == 3
    assert summary["minimum_cell_utc_days"] == 5
    assert summary["fallback_reason_counts"] == {
        "insufficient_markets": 2,
        "insufficient_utc_days": 2,
    }
