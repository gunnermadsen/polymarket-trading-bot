from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from btc_directional_model.q5_optimal_stopping_tournament import (
    CANDIDATES,
    _abstained_markets,
    _first_crossing,
    _future_max_by_market,
    _stopping_diagnostics,
    _with_action,
    load_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PACKAGE_ROOT / "configs" / "btc-5m-q5-optimal-stopping-tournament-20260525-20260802.toml"
)


def test_tournament_contract_is_offline_and_chronological() -> None:
    config = load_config(CONFIG_PATH)

    assert config.fresh_holdout is False
    assert config.source.execution.quantities[0] == 5
    assert config.source.execution.quantities[-1] == 200
    assert config.folds[0].evaluation_start == datetime(2026, 6, 29, tzinfo=UTC)
    assert config.folds[-1].evaluation_end == datetime(2026, 8, 2, tzinfo=UTC)
    assert len(CANDIDATES) == 7


def test_future_max_uses_only_later_rows_in_same_market() -> None:
    markets = np.asarray(["a", "a", "a", "b", "b"])
    values = np.asarray([1.0, 3.0, 2.0, 4.0, 1.0])

    result = _future_max_by_market(markets, values)

    assert result.tolist() == [3.0, 2.0, 0.0, 1.0, 0.0]


def test_first_crossing_takes_earliest_qualified_row_per_market() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "b", "b"],
            "score": [0.2, 0.8, 0.9, 0.95],
            "seconds_elapsed": [20, 30, 40, 50],
        }
    )

    selected = _first_crossing(frame, "score", 0.7)

    assert selected["seconds_elapsed"].to_list() == [30, 40]


def test_action_scoring_sets_side_and_executable_cost() -> None:
    frame = pl.DataFrame(
        {
            "label_up": [1, 0],
            "up_cost_5": [0.4, 0.7],
            "down_cost_5": [0.6, 0.3],
        }
    )

    scored = _with_action(
        frame,
        np.asarray([0.2, -0.1]),
        np.asarray([-0.1, 0.3]),
        np.asarray([0.2, 0.3]),
        "candidate_value",
    )

    assert scored["predicted_up"].to_list() == [True, False]
    assert scored["selected_cost_5"].to_list() == [0.4, 0.3]
    assert scored["direction_correct"].to_list() == [True, True]


def test_expander_eligibility_excludes_incumbent_markets() -> None:
    frame = pl.DataFrame({"market_id": ["a", "a", "b"], "seconds_elapsed": [20, 30, 40]})
    incumbent = frame.filter(pl.col("market_id") == "a").head(1)

    eligible = _abstained_markets(frame, incumbent)

    assert eligible["market_id"].to_list() == ["b"]


def test_aggregate_stopping_diagnostics_use_full_market_denominator_and_q5_timing() -> None:
    selected = pl.DataFrame(
        {
            "market_id": ["a", "b"],
            "seconds_elapsed": [30, 120],
            "predicted_up": [True, False],
            "up_stress_reward": [0.4, -0.6],
            "down_stress_reward": [-0.4, 0.6],
            "best_realized_action_value": [0.5, 0.7],
            "future_best_action_value": [0.6, 0.8],
        }
    )
    incumbent = pl.DataFrame(
        {
            "market_id": ["a", "b"],
            "seconds_elapsed": [60, 90],
        }
    )

    diagnostics = _stopping_diagnostics(
        selected,
        eligible_market_count=4,
        incumbent=incumbent,
    )

    assert diagnostics["wait_rate"] == 0.5
    assert diagnostics["earlier_than_q5_rate"] == 0.5
    assert diagnostics["average_entry_lead_seconds"] == 0.0
