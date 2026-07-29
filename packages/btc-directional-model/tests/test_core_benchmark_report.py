from __future__ import annotations

from copy import deepcopy
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
            "execution_evidence_coverage": 0.80,
            "executable_coverage_within_evidence": 0.95,
            "executable_coverage_all_selected": 0.76,
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
    assert "Evidence / selected" in first
    assert "Executable / evidence" in first
    assert "Executable / all selected" in first
    assert "early-core" in first


def test_generate_report_writes_self_contained_html(tmp_path: Path) -> None:
    destination = generate_benchmark_report(report_record(), tmp_path / "report.html")
    contents = destination.read_text()

    assert destination.exists()
    assert "<!doctype html>" in contents
    assert "plotly.js" in contents


def test_report_uses_recorded_walk_forward_fold_count() -> None:
    record = deepcopy(report_record())
    control = record["candidates"]["control"]["own_policy"]
    record["training_evidence"] = {
        "core_candidates": {
            "control": {
                "out_of_fold": control,
                "timing": {
                    "median_first_crossing_seconds": 90,
                    "early_entry_coverage": 0.5,
                },
                "total_folds": 7,
                "passed_development": False,
            }
        }
    }

    document = render_benchmark_report(record)

    assert "7-fold walk-forward" in document
    assert "five-fold walk-forward" not in document


def test_report_labels_chronological_policy_and_reused_diagnostics() -> None:
    record = deepcopy(report_record())
    record["candidates"]["control"]["policy"].update(
        {
            "confidence_threshold": None,
            "selection_mode": "chronological_preselected",
            "confidence_threshold_min": 0.87,
            "confidence_threshold_max": 0.91,
        }
    )
    record["training_evidence"] = {
        "core_candidates": {},
        "prior_diagnostics": {"reused_without_retraining": True},
        "preopen_candidate": {
            "candidate": "preopen",
            "out_of_fold": record["candidates"]["control"]["own_policy"],
            "timing": {
                "median_first_crossing_seconds": 90,
                "early_entry_coverage": 0.5,
            },
            "passed_development": False,
        },
    }
    record["training_selection"] = {
        "finalist": None,
        "candidates": {
            "early-core": {
                "passed": False,
                "checks": [
                    {
                        "name": "minimum executable economics samples",
                        "passed": False,
                    }
                ],
            }
        },
    }

    document = render_benchmark_report(record)

    assert "chronological 0.87–0.91" in document
    assert "prior diagnostic — reused, not retrained, not eligible" in document
    assert "Frozen training selection" in document
    assert "no runtime freeze created" in document


def test_report_renders_strict_book_chronology_without_runtime_claim() -> None:
    record = report_record()
    metrics = record["candidates"]["control"]["own_policy"]
    timing = {
        "median_first_crossing_seconds": 90,
        "early_entry_coverage": 0.60,
    }
    candidate_record = {
        "training": {
            "confidence_threshold": 0.89,
            "threshold_qualified": True,
        },
        "policy_diagnostic": {
            "metrics": metrics,
            "timing": timing,
        },
        "later_vintage_evaluation": {
            "metrics": metrics,
            "timing": timing,
        },
    }
    record["training_evidence"] = {
        "strict_book_chronology": {
            "exact_row_ablation": True,
            "candidates": {
                "control": candidate_record,
                "strict-book": candidate_record,
            },
        }
    }
    record["strict_book_selection"] = {
        "status": "selected_for_future_independent_validation",
        "winner": "strict-book",
        "statistical_checks": [],
    }

    document = render_benchmark_report(record)

    assert "Strict-book chronological challenge" in document
    assert "identical strict-valid rows" in document
    assert "strict-book" in document
    assert "no deployment artifact was exported" in document


def test_report_renders_compact_residual_and_sealed_holdout() -> None:
    record = report_record()
    calibration_cell = {
        "band": "60-89",
        "raw_direction": "UP",
        "markets": 120,
        "positives": 80,
        "slope": 0.9,
        "intercept": 0.1,
        "converged": True,
    }
    record["training_evidence"] = {
        "book_residual": {
            "final_residual": {
                "model": {
                    "gamma": 0.4,
                    "beta": [0.1] * 7,
                    "feature_scales": [1.0] * 7,
                    "feature_names": [
                        "book_core_mid_logit_disagreement",
                        "book_vwap10_logit_minus_mid",
                        "book_imbalance_difference",
                        "book_log_bid_depth_difference",
                        "book_log_ask_depth_difference",
                        "book_spread_difference",
                        "book_mid_logit_delta_5s",
                        "book_imbalance_difference_delta_5s",
                    ],
                }
            },
            "direction_time_calibration": {
                "control": {"cells": [calibration_cell]},
                "residual": {"cells": [calibration_cell]},
            },
            "threshold_selection": {
                "control": {
                    "threshold": 0.87,
                    "qualified": False,
                    "history": [{"markets": 300}],
                },
                "residual": {
                    "threshold": 0.87,
                    "qualified": False,
                    "history": [{"markets": 320}],
                },
            },
        }
    }
    record["data_evidence"] = {
        "cohorts": {
            "residual_fit": {
                "rows": 1000,
                "markets": 200,
            },
            "policy_diagnostic": {
                "universal_rows": 2000,
                "universal_markets": 300,
                "strict_rows": 1800,
                "strict_markets": 280,
                "strict_market_coverage": 280 / 300,
            },
        },
        "sealed_holdout": {
            "range_start": "2026-07-22T00:00:00+00:00",
            "range_end": "2026-07-23T00:00:00+00:00",
            "status": "sealed_not_accessed",
            "reason": "below the unchanged minimum",
        },
    }
    record["book_residual_selection"] = {
        "status": "blocked_by_frozen_development_gates",
        "winner": None,
    }

    document = render_benchmark_report(record)

    assert "Compact orderbook residual challenge" in document
    assert "strict 10-share books with an exact prior five-second row" in document
    assert "unchanged BTC-core fallback" in document
    assert "book_core_mid_logit_disagreement" in document
    assert "sealed_not_accessed" in document
    assert "winner: <strong>none</strong>" in document
    assert "No runtime artifact was exported" in document
    assert "Chronological training evidence" not in document
