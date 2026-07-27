from __future__ import annotations

from pathlib import Path

from btc_directional_model.core_benchmark import BENCHMARK_SCHEMA_VERSION
from btc_directional_model.core_benchmark_report import (
    generate_benchmark_report,
    render_benchmark_report,
)


def report_record() -> dict:
    metrics = {
        "markets": 100,
        "eligible_markets": 120,
        "coverage": 5 / 6,
        "available_markets": 120,
        "data_unavailable_markets": 0,
        "confidence_no_trade_markets": 20,
        "no_trade_markets": 20,
        "accuracy": 0.9,
        "balanced_accuracy": 0.89,
        "up_recall": 0.88,
        "down_recall": 0.90,
        "wilson_lower_95": 0.83,
        "expected_calibration_error": 0.02,
        "median_seconds_elapsed": 90.0,
        "p90_seconds_elapsed": 120.0,
        "execution": {
            "executable_coverage": 0.95,
            "median_selected_ask_vwap_5": 0.60,
            "mean_fee_per_share": 0.01,
            "mean_direct_edge_per_share": 0.08,
            "realized_net_expectancy_per_trade": 1.45,
            "maximum_net_loss_streak": 2,
            "maximum_drawdown": 3.20,
        },
    }
    band = {
        "band": "90-119",
        "markets": 100,
        "coverage": 5 / 6,
        "accuracy": 0.9,
        "balanced_accuracy": 0.89,
        "up_recall": 0.88,
        "down_recall": 0.90,
        "wilson_lower_95": 0.83,
    }
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "evaluation": {
            "label": "Chronological development comparison",
            "kind": "development",
            "independent": False,
            "development_only": True,
        },
        "control_candidate": "control",
        "quantity": 5.0,
        "eligible_markets": 120,
        "minimum_samples": 80,
        "minimum_executable_samples": 80,
        "fixed_checkpoints": [60, 90, 120, 180, 240],
        "time_bands": [],
        "candidate_order": ["control", "early-core"],
        "candidates": {
            "control": {
                "policy": {
                    "confidence_threshold": 0.89,
                    "deployment_compatible": True,
                },
                "own_policy": metrics,
                "time_bands": [band],
                "checkpoints": [],
                "advance": {
                    "is_control": True,
                    "benchmark_passed": None,
                    "deployment_qualified": None,
                    "checks": [],
                },
            },
            "early-core": {
                "policy": {
                    "confidence_threshold": 0.89,
                    "deployment_compatible": True,
                },
                "own_policy": metrics,
                "time_bands": [band],
                "checkpoints": [],
                "advance": {
                    "is_control": False,
                    "benchmark_passed": True,
                    "deployment_qualified": False,
                    "checks": [
                        {
                            "name": "minimum accepted samples",
                            "observed": 100,
                            "operator": ">=",
                            "required": 80,
                            "passed": True,
                        }
                    ],
                },
            },
        },
        "common_comparisons": {
            "early-core": {
                "control_candidate": "control",
                "candidate": "early-core",
                "cohort": "same market and timestamp",
                "checkpoints": [
                    {
                        "seconds_elapsed": 90,
                        "common_markets": 120,
                        "control": {"accuracy": 0.88},
                        "candidate": {"accuracy": 0.90},
                        "accuracy_delta": 0.02,
                        "balanced_accuracy_delta": 0.01,
                        "up_recall_delta": 0.01,
                        "down_recall_delta": 0.01,
                    }
                ],
            }
        },
        "benchmark_passed_candidates": ["early-core"],
        "deployment_qualified_candidates": [],
    }


def test_report_is_deterministic_and_labels_non_independent_evidence() -> None:
    record = report_record()

    first = render_benchmark_report(record)
    second = render_benchmark_report(record)

    assert first == second
    assert "development only" in first
    assert "non-independent development evidence" in first
    assert "same-market, exact-timestamp checkpoint comparisons" in first
    assert "early-core" in first


def test_generate_report_writes_self_contained_html(tmp_path: Path) -> None:
    destination = generate_benchmark_report(report_record(), tmp_path / "report.html")
    contents = destination.read_text()

    assert destination.exists()
    assert "<!doctype html>" in contents
    assert "plotly.js" in contents
