from __future__ import annotations

import json
from pathlib import Path

from kraken_ml.regression_reporting import (
    render_regression_development_report,
    render_regression_holdout_report,
    write_regression_development_report,
    write_regression_holdout_report,
)


def _development_report() -> dict:
    gates = {
        "pass": True,
        "checks": {
            "pooled_expectancy": {
                "pass": True,
                "actual": 4.25,
                "required": 3.0,
            }
        },
    }
    return {
        "run_id": "20260728T120000Z-source00-config00",
        "generated_at": "2026-07-28T12:00:00+00:00",
        "verdict": "qualified_for_holdout",
        "hashes": {
            "config_sha256": "config",
            "source_sha256": "source",
            "code_sha256": "code",
        },
        "candidate_aggregates": [
            {
                "candidate_id": "h4_ridge_price",
                "horizon_bars": 16,
                "model": "ridge",
                "feature_set": "price",
                "pooled_regression": {
                    "pooled": {
                        "mae_bps": 12.5,
                        "rmse_bps": 18.0,
                        "r2": 0.01,
                        "spearman": 0.09,
                    }
                },
                "positive_nominal_folds": 6,
                "median_fold_stress_expectancy_bps": 3.5,
                "pooled_economics": {"net_expectancy_bps": 4.25, "trades": 300},
                "pooled_stress": {"net_expectancy_bps": 3.5},
                "fee_counterfactuals": {
                    "taker_10_bps": {
                        "round_trip_fee_bps": 10.0,
                        "trades": 300,
                        "net_expectancy_bps": 4.25,
                        "bootstrap_95_lower_bps": 1.0,
                        "profit_factor": 1.3,
                    }
                },
                "qualified": True,
                "gates": gates,
            }
        ],
        "fold_results": [
            {
                "candidate_id": "h4_ridge_price",
                "fold": "fold_a",
                "policy": {
                    "expected_return_hurdle_bps": 3.0,
                    "directional_advantage_bps": 3.0,
                },
                "economics": {"trades": 50, "net_expectancy_bps": 4.25},
                "cost_stress": {"net_expectancy_bps": 3.5},
            }
        ],
        "selected": {
            "candidate_id": "h4_ridge_price",
            "horizon_bars": 16,
            "model": "ridge",
            "feature_set": "price",
            "diagnostic_only": False,
            "gates": gates,
        },
        "final_confirmation": {
            "status": "passed",
            "gates": gates,
            "economics": {"net_expectancy_bps": 4.0},
            "cost_stress": {"net_expectancy_bps": 3.2},
        },
        "holdout": {
            "status": "sealed_ready",
            "opened": False,
            "identity": "holdout",
            "reason": "frozen",
        },
    }


def test_development_report_has_fixed_artifact_names(tmp_path: Path) -> None:
    report = _development_report()
    json_path, markdown_path = write_regression_development_report(tmp_path, report)

    assert json_path.name == "regression-development-benchmark.json"
    assert markdown_path.name == "regression-development-benchmark.md"
    assert json.loads(json_path.read_text())["selected"]["candidate_id"] == "h4_ridge_price"
    markdown = markdown_path.read_text()
    assert "Net-Expectancy Development Benchmark" in markdown
    assert "h4_ridge_price" in markdown
    assert "Pooled expectancy" in markdown
    assert "Training prerequisites" in markdown
    assert "Active-candidate fixed-action fee counterfactuals" in markdown
    assert "Open-interest qualification" in markdown
    assert "Historical classifier control" in markdown


def test_holdout_report_has_fixed_artifact_names(tmp_path: Path) -> None:
    report = {
        "run_id": "20260728T120000Z-source00-config00",
        "generated_at": "2026-07-28T12:00:00+00:00",
        "verdict": "qualified_positive_net_edge",
        "hashes": {"config_sha256": "config", "source_sha256": "source"},
        "selected": {
            "candidate_id": "h4_ridge_price",
            "horizon_bars": 16,
            "model": "ridge",
            "feature_set": "price",
            "policy": {
                "hurdle_bps": 3.0,
                "advantage_bps": 3.0,
                "no_trade": False,
            },
        },
        "economics": {
            "trades": 220,
            "net_expectancy_bps": 4.0,
            "bootstrap_95_lower_bps": 1.0,
            "profit_factor": 1.3,
            "positive_month_fraction": 0.8,
        },
        "cost_stress": {"trades": 220, "net_expectancy_bps": 2.0},
        "gates": {
            "pass": True,
            "checks": {
                "minimum_trades": {"pass": True, "actual": 220, "required": 200}
            },
        },
    }
    json_path, markdown_path = write_regression_holdout_report(tmp_path, report)

    assert json_path.name == "regression-holdout-benchmark.json"
    assert markdown_path.name == "regression-holdout-benchmark.md"
    assert "LOCKED-HOLDOUT" in render_regression_holdout_report(report).upper()
    assert "QUALIFIED_POSITIVE_NET_EDGE" in markdown_path.read_text()
    assert "Expected-return hurdle bps | 3.0000" in markdown_path.read_text()


def test_render_development_report_is_stable() -> None:
    report = _development_report()
    assert render_regression_development_report(report) == render_regression_development_report(
        report
    )
