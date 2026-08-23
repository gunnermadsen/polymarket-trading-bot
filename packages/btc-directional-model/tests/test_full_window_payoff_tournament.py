from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from btc_directional_model.full_window_payoff_tournament import (
    SPECS,
    _apply_policy,
    _nonempty_fold_summary,
    _policy_mask,
    load_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PACKAGE_ROOT / "configs" / "btc-5m-full-window-payoff-challenger-20260525-20260802.toml"
)


def test_challenger_contract_is_offline_and_full_window() -> None:
    config = load_config(CONFIG_PATH)

    assert config.training.fresh_holdout is False
    assert config.evidence_training.windows.outcome_fit_end == datetime(2026, 6, 8, tzinfo=UTC)
    assert config.training.windows.outcome_fit_end == datetime(2026, 7, 8, tzinfo=UTC)
    assert config.training.execution.quantities[0] == 5
    assert config.training.execution.quantities[-1] == 200
    assert [(band.start_second, band.end_second_exclusive) for band in config.training.bands] == [
        (15, 90),
        (90, 180),
        (180, 241),
    ]
    assert config.training.windows.test_end == datetime(2026, 8, 2, tzinfo=UTC)
    assert [cell.name for cell in config.selection_cells] == [
        "early_15_89",
        "middle_90_119",
        "middle_120_149",
        "middle_150_179",
        "late_180_240",
    ]


def test_baseline_policy_takes_first_crossing_across_selection_cells() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "b"],
            "seconds_elapsed": [30, 100, 130],
            "probability_selected": [0.70, 0.90, 0.80],
            "selected_edge_5": [0.02, 0.10, 0.05],
        }
    )
    values = {
        "confidence": 0.65,
        "edge": 0.0,
        "admission": float("-inf"),
        "payoff_lower_bound": float("-inf"),
        "maximum_loss_probability": float("inf"),
        "maximum_expected_shortfall": float("inf"),
        "maximum_family_spread": float("inf"),
        "require_expert_agreement": False,
        "enabled": True,
    }
    policy = {
        "early_15_89": values,
        "middle_90_119": values,
        "middle_120_149": values,
        "middle_150_179": {**values, "enabled": False},
        "late_180_240": {**values, "enabled": False},
    }

    selected = _apply_policy(frame, SPECS["continuous_payoff_baseline"], policy)

    assert selected["market_id"].to_list() == ["a", "b"]
    assert selected["probability_selected"].to_list() == [0.70, 0.80]


def test_consensus_policy_requires_all_direction_votes() -> None:
    frame = pl.DataFrame(
        {
            "conservative_probability_selected": [0.80, 0.80],
            "conservative_edge_5": [0.10, 0.10],
            "price_bucket_minimum_edge": [0.01, 0.01],
            "admission_probability": [0.80, 0.80],
            "payoff_stress_edge_lower_bound": [0.05, 0.05],
            "predicted_up": [True, True],
            "directional_family_probability_up": [0.70, 0.70],
            "asymmetric_family_probability_up": [0.70, 0.30],
            "family_probability_spread": [0.00, 0.40],
        }
    )
    values = {
        "confidence": 0.75,
        "edge": 0.0,
        "admission": 0.65,
        "payoff_lower_bound": 0.0,
        "maximum_loss_probability": float("inf"),
        "maximum_expected_shortfall": float("inf"),
        "maximum_family_spread": 0.20,
        "require_expert_agreement": True,
    }

    mask = _policy_mask(frame, SPECS["oof_expert_distilled_admission"], values)

    assert mask.tolist() == [True, False]


def test_nonempty_fold_summary_does_not_treat_missing_data_as_a_loss() -> None:
    summary = _nonempty_fold_summary(
        {
            "folds": [
                {"trades": 10, "stress_net_pnl": 2.0, "stress_expectancy_per_trade": 0.2},
                {"trades": 0, "stress_net_pnl": 0.0, "stress_expectancy_per_trade": 0.0},
                {"trades": 5, "stress_net_pnl": -0.5, "stress_expectancy_per_trade": -0.1},
            ]
        }
    )

    assert summary["folds"] == 2
    assert summary["profitable_fold_ratio"] == 0.5
    assert summary["worst_stress_expectancy"] == -0.1
