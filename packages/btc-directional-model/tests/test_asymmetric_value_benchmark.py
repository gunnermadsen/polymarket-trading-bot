from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from btc_directional_model.asymmetric_value_benchmark import (
    _calibration_report_summary,
    _candidate_frames,
    _candidate_grid_summary,
    _configured_windows,
    _development_markdown_report,
    _evaluation_economics_table,
    _frame_content_digest,
    _join_oracle_l2_candidate_features,
    _matched_control_noninferiority_checks,
    _matched_feature_attribution,
    _selected_matched_control,
    _validate_external_core_keys,
)
from btc_directional_model.asymmetric_value_config import (
    AsymmetricValueConfig,
    load_asymmetric_value_config,
)
from btc_directional_model.asymmetric_value_data import EARLY_CAUSAL_ORACLE_FEATURES
from btc_directional_model.asymmetric_value_training import (
    ASYMMETRIC_VALUE_CANDIDATES,
    CANDLE_MATCHED_CORE_PRICE_CONTROL,
    CORE_CANDLES_PRICE,
    CORE_L2_PRICE,
    CORE_ORACLE_L2_PRICE,
    CORE_ORACLE_PRICE,
    CORE_PRICE,
    L2_MATCHED_CORE_PRICE_CONTROL,
    ORACLE_MATCHED_CORE_PRICE_CONTROL,
    PRICE_LOGISTIC,
    THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL,
)
from btc_directional_model.spot_l2_chainlink_features import L2_FEATURES


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
    assert _selected_matched_control(CORE_CANDLES_PRICE) == (
        CANDLE_MATCHED_CORE_PRICE_CONTROL
    )
    assert _selected_matched_control(CORE_ORACLE_L2_PRICE) == (
        THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL
    )


def test_candidate_frames_predeclare_model_matrix_and_matched_controls() -> None:
    frame = _core_frame()

    candidates = _candidate_frames(
        price=frame,
        l2_price=frame,
        candle_price=frame,
        oracle_price=frame,
        three_source_price=frame,
    )

    assert tuple(candidates) == ASYMMETRIC_VALUE_CANDIDATES


def test_three_source_join_is_exact_key_intersection_without_filling() -> None:
    start = datetime(2026, 7, 23, tzinfo=UTC)

    def keyed(seconds: list[int]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "market_id": [f"m-{second}" for second in seconds],
                "window_start": [start] * len(seconds),
                "observed_at": [
                    start + timedelta(seconds=second) for second in seconds
                ],
                "seconds_elapsed": seconds,
                "label_up": [1] * len(seconds),
            }
        )

    oracle = keyed([5, 10]).with_columns(
        *(
            pl.lit(float(index + 1)).alias(name)
            for index, name in enumerate(EARLY_CAUSAL_ORACLE_FEATURES)
        )
    )
    l2 = keyed([10, 15]).with_columns(
        *(
            pl.lit(float(index + 1)).alias(name)
            for index, name in enumerate(L2_FEATURES)
        )
    )

    joined = _join_oracle_l2_candidate_features(oracle, l2)

    assert joined["seconds_elapsed"].to_list() == [10]
    assert set(EARLY_CAUSAL_ORACLE_FEATURES).issubset(joined.columns)
    assert set(L2_FEATURES).issubset(joined.columns)


def _attribution_inputs() -> tuple[
    AsymmetricValueConfig,
    str,
    dict[str, dict[str, float]],
    dict[str, pl.DataFrame],
    dict[str, pl.DataFrame],
]:
    config = replace(
        load_asymmetric_value_config(
            Path(__file__).parents[1]
            / "configs/btc-5m-directional-asymmetric-value-one-second-20260414-20260802.toml"
        ),
        bootstrap_resamples=20,
    )
    policy = next(item.name for item in config.policies if item.selection_eligible)
    metrics = {
        f"{name}::{policy}": {
            "accuracy": 0.30,
            "net_expectancy_per_trade": 0.10,
            "stress_1c_net_expectancy_per_trade": 0.05,
            "net_profit_per_resolved_market": 0.02,
            "capital_efficiency": 0.04,
            "profit_factor": 1.10,
            "selected_calibration_bias": 0.01,
            "trades_per_resolved_market": 0.05,
        }
        for name in ASYMMETRIC_VALUE_CANDIDATES
    }
    empty_ledger = pl.DataFrame(
        schema={
            "window_start": pl.Datetime("us", "UTC"),
            "realized_net": pl.Float64,
        }
    )
    ledgers = {
        f"{name}::{policy}": empty_ledger
        for name in ASYMMETRIC_VALUE_CANDIDATES
    }
    frames = {name: _core_frame() for name in ASYMMETRIC_VALUE_CANDIDATES}
    return config, policy, metrics, ledgers, frames


def test_policy_attribution_rejects_mismatched_market_second_keys() -> None:
    config, policy, metrics, ledgers, frames = _attribution_inputs()
    frames[ORACLE_MATCHED_CORE_PRICE_CONTROL] = frames[
        ORACLE_MATCHED_CORE_PRICE_CONTROL
    ].with_columns((pl.col("seconds_elapsed") + 1).alias("seconds_elapsed"))

    with pytest.raises(RuntimeError, match="does not share exact market/second keys"):
        _matched_feature_attribution(
            metrics,
            ledgers,
            frames,
            policy_name=policy,
            config=config,
            window=config.policy,
            seed_offset=0,
        )


def test_policy_attribution_marks_combined_candidate_offline_only() -> None:
    config, policy, metrics, ledgers, frames = _attribution_inputs()

    evidence = _matched_feature_attribution(
        metrics,
        ledgers,
        frames,
        policy_name=policy,
        config=config,
        window=config.policy,
        seed_offset=0,
    )

    combined = evidence["comparisons"][CORE_ORACLE_L2_PRICE]
    assert combined["matched_control"] == (
        THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL
    )
    assert combined["candidate_feature_count"] == 115
    assert combined["control_feature_count"] == 75
    assert len(combined["added_features"]) == 40
    assert combined["identical_market_second_keys_verified"]
    assert not combined["selection_eligible"]
    assert not combined["runtime_exportable"]


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
    assert table[0]["net_profit_per_resolved_market"] == pytest.approx(
        (len(ASYMMETRIC_VALUE_CANDIDATES) - 1) / 10.0
    )
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
    assert PRICE_LOGISTIC not in eligible
    assert CORE_CANDLES_PRICE not in eligible
    assert CORE_ORACLE_L2_PRICE not in eligible
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
    assert summary["target_required_models"] == 0
    assert summary["target_qualified"] is False


def test_calibration_report_exposes_target_cell_qualification() -> None:
    target = {
        "required": True,
        "qualified": False,
        "required_fitted_cells": 8,
        "fitted_cells": 7,
        "fallback_cells": 1,
    }
    training = {
        "profiles": {
            "candidate": {
                "calibration_bands": [
                    {
                        "converged": True,
                        "slope": 1.0,
                        "rows": 500,
                        "markets": 100,
                    }
                ],
                "side_price_time_calibration": {
                    "minimum_utc_days_per_cell": 5,
                    "target_contract": target,
                    "cells": [
                        {"fitted": False, "fallback": "single_class", "utc_days": 7}
                    ],
                },
            }
        }
    }

    summary = _calibration_report_summary(training)

    assert summary["target_required_models"] == 1
    assert summary["target_qualified_models"] == 0
    assert summary["target_required_cells"] == 8
    assert summary["target_fitted_cells"] == 7
    assert summary["target_fallback_cells"] == 1
    assert summary["target_qualified"] is False


def test_target_calibrated_window_inventory_excludes_fake_evaluation() -> None:
    config = load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-calibrated-20260414-20260802.toml"
    )

    windows = _configured_windows(config)

    assert tuple(name for name, _ in windows) == ("fit", "calibration", "policy")
    assert windows[-1][1].end == config.policy.end


def test_development_report_never_claims_independent_proof() -> None:
    result = {
        "selection": {"qualified_on_policy_window": True},
        "historical_development": {
            "selected_key": "core_oracle_price_hgb::raw20_30_by55_edge_3c",
            "selected_metrics": {
                "trades": 100,
                "accuracy": 0.30,
                "mean_share_price": 0.25,
                "mean_entry_second": 20.0,
                "net_profit": 10.0,
                "net_expectancy_per_trade": 0.10,
                "stress_1c_net_expectancy_per_trade": 0.05,
                "profit_factor": 1.10,
                "loss_recovery_wins": 0.40,
            },
        },
        "evaluation": {
            "earliest_fresh_full_utc_day": "2026-08-09T00:00:00+00:00"
        },
    }

    report = _development_markdown_report(result)

    assert "not independent proof" in report
    assert "no deployment is authorized" in report
    assert "no PnL-based early stopping" in report
