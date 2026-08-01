from __future__ import annotations

from kraken_ml.reporting import render_development_report


def test_development_report_renders_aggregate_metrics_and_nested_gates() -> None:
    report = {
        "run_id": "test-run",
        "selected": {"model": "extra_trees", "feature_set": "price"},
        "model_comparison": {
            "extra_trees": {
                "model": "extra_trees",
                "feature_set": "full",
                "mean_balanced_accuracy": 0.42,
                "mean_macro_f1": 0.41,
                "mean_log_loss": 1.05,
                "positive_economic_folds": 0,
                "selected": True,
            }
        },
        "selection_parameters": {"minimum_calibration_trades": 50},
        "selected_fold_results": [
            {
                "fold": "fold_a",
                "model": "extra_trees",
                "feature_set": "price",
                "ranges": {"evaluation_start": "2025-01-01T00:00:00+00:00"},
                "classification": {
                    "balanced_accuracy": 0.42,
                    "macro_f1": 0.41,
                    "log_loss": 1.05,
                },
                "balanced_accuracy_uplift": 0.04,
                "economics": {"trades": 0, "net_expectancy_bps": None},
                "policy": {
                    "probability_threshold": 1.0,
                    "directional_margin": 1.0,
                    "no_trade": True,
                },
                "policy_grid": [
                    {
                        "trades": 80,
                        "net_expectancy_bps": -3.0,
                        "bootstrap_80_lower_bps": -8.0,
                        "profit_factor": 0.9,
                        "probability_threshold": 0.5,
                        "directional_margin": 0.1,
                    }
                ],
            }
        ],
        "gates": {
            "pass": False,
            "checks": {
                "positive_net_expectancy": {
                    "pass": False,
                    "actual": 0,
                    "required": 5,
                }
            },
        },
        "holdout_state": {"status": "sealed_not_qualified", "opened": False},
    }

    markdown = render_development_report(report)

    assert "42.00%" in markdown
    assert "1.05" in markdown
    assert "Best net bps/trade" in markdown
    assert "-3" in markdown
    assert "positive net expectancy" in markdown
    assert "sealed_not_qualified" in markdown
