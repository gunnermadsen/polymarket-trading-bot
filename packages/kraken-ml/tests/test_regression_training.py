from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from kraken_ml.config import load_config, load_expectancy_config
from kraken_ml.dataset import Snapshot
from kraken_ml.regression_training import (
    _assert_complete_funding,
    _confirmation_gates,
    _consensus_policy,
    _fold_jobs,
    _holdout_gates,
    _pooled_regression_diagnostics,
    _regression_policy_payload,
    apply_oi_qualification,
    development_gates,
    select_candidate,
)
from kraken_ml.training import _holdout_identity as classifier_holdout_identity


def _economics(
    *,
    expectancy: float = 4.0,
    trades: int = 100,
    total: float = 400.0,
    lower: float = 1.0,
) -> dict:
    return {
        "trades": trades,
        "net_expectancy_bps": expectancy,
        "bootstrap_95_lower_bps": lower,
        "profit_factor": 1.3,
        "win_rate": 0.6,
        "positive_month_fraction": 0.75,
        "total_net_bps": total,
        "action_coverage": 0.1,
    }


def _aggregate(candidate_id: str, feature_set: str = "price") -> dict:
    aggregate = {
        "candidate_id": candidate_id,
        "horizon_bars": 16,
        "model": "extra_trees",
        "feature_set": feature_set,
        "positive_nominal_folds": 5,
        "positive_stress_folds": 5,
        "positive_fold_pnl_fraction": 0.2,
        "median_fold_stress_expectancy_bps": 3.0,
        "turnover": 0.1,
        "pooled_economics": _economics(),
        "pooled_stress": _economics(expectancy=2.0),
        "oi_qualification": None,
    }
    return aggregate


@pytest.fixture(scope="module")
def expectancy_config():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "pf_xbtusd_15m_expectancy.toml"
    )
    return load_expectancy_config(path)


def test_exact_candidate_fold_job_count(expectancy_config, tmp_path: Path) -> None:
    snapshots = {
        4: replace(
            Snapshot(
                path=tmp_path / "raw",
                sha256="0" * 64,
                row_count=1,
                first_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                last_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                manifest_path=tmp_path / "manifest",
            ),
        ),
        16: replace(
            Snapshot(
                path=tmp_path / "raw",
                sha256="0" * 64,
                row_count=1,
                first_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                last_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                manifest_path=tmp_path / "manifest",
            ),
        ),
    }
    assert len(_fold_jobs(expectancy_config, snapshots)) == 48


def test_development_gates_enforce_all_frozen_economic_requirements(
    expectancy_config,
) -> None:
    aggregate = _aggregate("h4_extra_trees_price")
    gates = development_gates(aggregate, expectancy_config)
    assert gates["pass"]
    assert set(gates["checks"]) == {
        "nominal_positive_fold_stability",
        "minimum_pooled_net_expectancy",
        "cost_stress_positive_fold_stability",
        "positive_daily_block_bootstrap_lower",
        "minimum_profit_factor",
        "positive_calendar_month_fraction",
        "positive_fold_pnl_concentration",
    }

    aggregate["positive_fold_pnl_fraction"] = 0.41
    assert not development_gates(aggregate, expectancy_config)["pass"]


def test_oi_candidate_requires_paired_wins_and_noninferior_stress(
    expectancy_config,
) -> None:
    baseline = _aggregate("h4_extra_trees_price")
    oi = _aggregate("h4_extra_trees_price_oi", "price_oi")
    for aggregate in (baseline, oi):
        aggregate["gates"] = development_gates(aggregate, expectancy_config)
        aggregate["qualified"] = aggregate["gates"]["pass"]
    folds = []
    for index, fold in enumerate(expectancy_config.validation.folds):
        folds.extend(
            [
                {
                    "candidate_id": baseline["candidate_id"],
                    "fold": fold.name,
                    "economics": {"net_expectancy_bps": 2.0},
                },
                {
                    "candidate_id": oi["candidate_id"],
                    "fold": fold.name,
                    "economics": {
                        "net_expectancy_bps": 3.0 if index < 5 else 1.0
                    },
                },
            ]
        )

    apply_oi_qualification([baseline, oi], folds, expectancy_config)

    assert oi["oi_qualification"]["pass"]
    assert oi["qualified"]
    oi["pooled_stress"]["net_expectancy_bps"] = 1.0
    apply_oi_qualification([baseline, oi], folds, expectancy_config)
    assert not oi["qualified"]


def test_selection_uses_stress_then_turnover_then_simplicity_then_id() -> None:
    ridge = _aggregate("ridge")
    ridge.update(model="ridge", median_fold_stress_expectancy_bps=3.0, turnover=0.05)
    histogram = _aggregate("histogram")
    histogram.update(
        model="histogram",
        median_fold_stress_expectancy_bps=3.0,
        turnover=0.05,
    )
    for aggregate in (ridge, histogram):
        aggregate["qualified"] = True
    selected, diagnostic = select_candidate([histogram, ridge])
    assert selected["candidate_id"] == "ridge"
    assert not diagnostic

    ridge["qualified"] = False
    histogram["qualified"] = False
    selected, diagnostic = select_candidate([histogram, ridge])
    assert selected["candidate_id"] == "ridge"
    assert diagnostic


def test_funding_guard_fails_closed_on_partial_coverage(
    expectancy_config,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "funding_coverage": {
                    "non_null_rows": 9,
                    "missing_rows": 1,
                    "first_timestamp": "2024-01-01T00:00:00+00:00",
                    "last_timestamp": "2024-01-01T02:15:00+00:00",
                }
            }
        )
    )
    snapshot = Snapshot(
        path=tmp_path / "raw.parquet",
        sha256="0" * 64,
        row_count=10,
        first_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        last_timestamp=datetime(2024, 1, 1, 2, 15, tzinfo=UTC),
        manifest_path=manifest,
    )
    with pytest.raises(RuntimeError, match="complete funding coverage"):
        _assert_complete_funding(snapshot, expectancy_config)


def test_regression_holdout_identity_matches_existing_global_seal(
    expectancy_config,
) -> None:
    classifier_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "pf_xbtusd_15m_1h.toml"
    )
    classifier_config = load_config(classifier_path)

    assert classifier_holdout_identity(expectancy_config) == classifier_holdout_identity(
        classifier_config
    )


def test_holdout_gates_require_minimum_trade_count(expectancy_config) -> None:
    economics = _economics(trades=199)
    stress = _economics(expectancy=1.0, trades=199)
    gates = _holdout_gates(economics, stress, expectancy_config)
    assert not gates["pass"]
    assert not gates["checks"]["minimum_trades"]["pass"]


def test_final_confirmation_uses_only_agreed_full_window_checks(
    expectancy_config,
) -> None:
    gates = _confirmation_gates(
        _economics(trades=50),
        _economics(expectancy=1.0, trades=50),
        expectancy_config,
    )
    assert gates["pass"]
    assert set(gates["checks"]) == {
        "minimum_confirmation_trades",
        "minimum_net_expectancy",
        "positive_daily_block_bootstrap_lower",
        "minimum_profit_factor",
        "positive_calendar_month_fraction",
        "positive_cost_stress_expectancy",
    }


def test_consensus_policy_uses_development_mode_and_conservative_ties() -> None:
    fold_results = [
        {
            "policy": {
                "hurdle_bps": hurdle,
                "advantage_bps": advantage,
                "no_trade": no_trade,
            }
        }
        for hurdle, advantage, no_trade in (
            (3.0, 0.0, False),
            (3.0, 0.0, False),
            (6.0, 3.0, False),
            (6.0, 3.0, False),
            (10.0, 6.0, True),
        )
    ]

    policy = _consensus_policy(fold_results)

    assert policy.hurdle_bps == 6.0
    assert policy.advantage_bps == 3.0
    assert not policy.no_trade


def test_consensus_policy_fails_closed_when_every_fold_is_no_trade() -> None:
    fold_results = [
        {
            "policy": {
                "hurdle_bps": 0.0,
                "advantage_bps": 0.0,
                "no_trade": True,
            }
        }
        for _ in range(6)
    ]

    policy = _consensus_policy(fold_results)

    assert policy.no_trade


def test_frozen_and_preregistered_policy_payloads_have_one_schema() -> None:
    policy = _consensus_policy(
        [
            {
                "policy": {
                    "hurdle_bps": 6.0,
                    "advantage_bps": 3.0,
                    "no_trade": False,
                }
            }
        ]
    )

    assert _regression_policy_payload(policy) == {
        "schema_version": 1,
        "hurdle_bps": 6.0,
        "advantage_bps": 3.0,
        "no_trade": False,
    }


def test_pooled_regression_diagnostics_are_exact_across_folds() -> None:
    results = [
        {
            "_regression_observed": np.asarray(
                [[1.0, -1.0], [2.0, -2.0]],
            ),
            "_regression_predictions": np.asarray(
                [[1.0, -1.0], [2.0, -2.0]],
            ),
        },
        {
            "_regression_observed": np.asarray([[3.0, -3.0]]),
            "_regression_predictions": np.asarray([[3.0, -3.0]]),
        },
    ]

    diagnostics = _pooled_regression_diagnostics(results)

    assert diagnostics["pooled"]["observations"] == 6
    assert diagnostics["pooled"]["mae_bps"] == 0.0
    assert diagnostics["pooled"]["r2"] == 1.0
