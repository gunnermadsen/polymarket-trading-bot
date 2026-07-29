from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from btc_directional_model.residual_admission_benchmark import (
    RESIDUAL_ADMISSION_SCHEMA_VERSION,
)
from btc_directional_model.residual_admission_report import (
    generate_residual_admission_report,
    render_residual_admission_report,
)

CONTROL = "histogram_enriched"
EARLY = "histogram_boundary_residual_early"
RESCUE = "histogram_boundary_residual_rescue"
COMBINED = "histogram_boundary_residual_combined"


def _execution(selected: int, economic: int) -> dict[str, object]:
    evidence = min(selected, economic + 20)
    executable = min(evidence, economic + 10)
    economic = min(executable, economic)
    return {
        "execution_price_source": "strict_snapshot",
        "fee_source": "polymarket",
        "selected_markets": selected,
        "execution_evidence_markets": evidence,
        "execution_evidence_coverage": evidence / selected if selected else 0.0,
        "executable_markets": executable,
        "executable_coverage": executable / selected if selected else 0.0,
        "executable_coverage_all_selected": executable / selected if selected else 0.0,
        "executable_coverage_within_evidence": (
            executable / evidence if evidence else 0.0
        ),
        "median_selected_ask_vwap_5": 0.51,
        "p90_selected_ask_vwap_5": 0.61,
        "economics_available": True,
        "economic_markets": economic,
        "mean_fee_per_share": 0.001,
        "total_fees": 2.5,
        "mean_direct_edge_per_share": 0.031,
        "positive_direct_edge_markets": max(0, economic - 10),
        "positive_direct_edge_rate": (
            max(0, economic - 10) / economic if economic else None
        ),
        "realized_net_pnl_total": 75.0,
        "realized_net_expectancy_per_trade": 0.15,
        "realized_net_expectancy_per_selected_market": 0.02,
        "maximum_net_loss_streak": 3,
        "maximum_drawdown": 2.5,
    }


def _metrics(
    *,
    markets: int,
    coverage: float,
    accuracy: float,
    median: float,
) -> dict[str, object]:
    return {
        "markets": markets,
        "eligible_markets": 1_000,
        "coverage": coverage,
        "correct": round(markets * accuracy),
        "accuracy": accuracy,
        "wilson_lower_95": 0.866,
        "wilson_upper_95": 0.91,
        "balanced_accuracy": accuracy,
        "up_precision": accuracy,
        "up_recall": 0.878,
        "down_precision": accuracy,
        "down_recall": 0.879,
        "f1": accuracy,
        "matthews_correlation": 0.75,
        "brier_score": 0.10,
        "log_loss": 0.30,
        "roc_auc": 0.93,
        "expected_calibration_error": 0.03,
        "confusion_matrix": [[390, 50], [50, 390]],
        "predicted_up": markets // 2,
        "predicted_down": markets - markets // 2,
        "actual_up": 500,
        "actual_down": 500,
        "maximum_consecutive_losses": 3,
        "no_trade_markets": 1_000 - markets,
        "no_trade_rate": 1.0 - coverage,
        "median_seconds_elapsed": median,
        "p90_seconds_elapsed": 210.0,
        "execution": _execution(markets, min(markets, 510)),
    }


def _residual(markets: int) -> dict[str, object]:
    metrics = _metrics(
        markets=markets,
        coverage=markets / 1_000,
        accuracy=0.88,
        median=110.0,
    )
    metrics["execution"] = _execution(markets, min(markets, 505))
    metrics["hourly_net_bootstrap"] = {
        "blocks": 100,
        "resamples": 10_000,
        "observed": 0.03,
        "lower_95": 0.002,
        "upper_95": 0.06,
    }
    return metrics


def _attribution(
    *,
    advanced: int,
    rescued: int,
) -> dict[str, object]:
    preserved = max(0, 700 - advanced)
    category_counts = {
        "preserved_control": preserved,
        "earlier_same_direction": advanced - 10,
        "earlier_preempted_direction_change": 10,
        "rescued_control_no_trade": rescued,
        "lost_control": 0,
        "still_no_trade": 1_000 - preserved - advanced - rescued,
    }
    categories = {
        name: {
            "markets": markets,
            "accuracy": 0.88 if markets else None,
            "median_entry_difference_seconds": (
                15.0 if name.startswith("earlier") else None
            ),
            "added_wrong_trades": round(markets * 0.12),
        }
        for name, markets in category_counts.items()
    }
    return {
        "eligible_markets": 1_000,
        "categories": categories,
        "advanced_markets": advanced,
        "median_advancement_seconds": 15.0 if advanced else None,
        "lost_control_markets": 0,
        "rescued_markets": rescued,
    }


def _check(name: str, passed: bool = True) -> dict[str, object]:
    return {
        "name": name,
        "observed": 0.88 if passed else 0.86,
        "operator": ">=",
        "required": 0.874,
        "passed": passed,
    }


def _candidate(
    name: str,
    *,
    coverage: float,
    median: float,
    residual_markets: int,
    advanced: int,
    rescued: int,
    passed: bool | None,
) -> dict[str, object]:
    selected = round(1_000 * coverage)
    metrics = _metrics(
        markets=selected,
        coverage=coverage,
        accuracy=0.882,
        median=median,
    )
    attribution = (
        None
        if name == CONTROL
        else _attribution(advanced=advanced, rescued=rescued)
    )
    return {
        "candidate": name,
        "out_of_fold": metrics,
        "timing": {
            "median_first_crossing_seconds": median,
            "p90_first_crossing_seconds": 210.0,
        },
        "no_trade_rate": 1.0 - coverage,
        "checkpoints": {
            "60": 0.02,
            "90": 0.20 if name == CONTROL else 0.24,
            "120": 0.49 if name == CONTROL else 0.51,
            "180": 0.61 if name == CONTROL else 0.67,
            "240": coverage,
        },
        "residual_cohort": _residual(residual_markets),
        "paired_attribution": attribution,
        "folds": [],
        "advance": {
            "is_control": name == CONTROL,
            "checks": [] if name == CONTROL else [_check("minimum accuracy")],
            "benchmark_passed": passed,
            "development_qualified": bool(passed),
            "deployment_qualified": False,
        },
    }


def _fold_candidate(
    name: str,
    candidate: dict[str, object],
) -> dict[str, object]:
    return {
        "metrics": candidate["out_of_fold"],
        "checkpoints": candidate["checkpoints"],
        "residual_cohort": candidate["residual_cohort"],
        "paired_attribution": candidate["paired_attribution"],
    }


def _threshold_diagnostic(
    *,
    threshold: float,
    qualified: bool,
    metrics: dict[str, object],
    residual_markets: int,
    residual_accuracy: float,
    by_120_uplift: float,
    rescued_markets: int,
    failed_gate: str | None = None,
) -> dict[str, object]:
    checks = [_check("minimum policy residual rows")]
    if failed_gate is not None:
        checks.append(_check(failed_gate, False))
    return {
        "threshold": threshold,
        "qualified": qualified,
        "metrics": metrics,
        "objective": {
            "coverage": metrics["coverage"],
            "decisions_by_120": 0.49 + by_120_uplift,
            "decisions_by_120_uplift": by_120_uplift,
            "median_seconds_elapsed": metrics["median_seconds_elapsed"],
            "residual_markets": residual_markets,
            "residual_accuracy": residual_accuracy,
            "rescued_markets": rescued_markets,
        },
        "checks": checks,
    }


def _benchmark() -> dict[str, object]:
    candidates = {
        CONTROL: _candidate(
            CONTROL,
            coverage=0.70,
            median=130.0,
            residual_markets=0,
            advanced=0,
            rescued=0,
            passed=None,
        ),
        EARLY: _candidate(
            EARLY,
            coverage=0.72,
            median=120.0,
            residual_markets=520,
            advanced=510,
            rescued=20,
            passed=True,
        ),
        RESCUE: _candidate(
            RESCUE,
            coverage=0.76,
            median=130.0,
            residual_markets=520,
            advanced=0,
            rescued=200,
            passed=True,
        ),
        COMBINED: _candidate(
            COMBINED,
            coverage=0.78,
            median=120.0,
            residual_markets=700,
            advanced=510,
            rescued=200,
            passed=True,
        ),
    }
    threshold_check = _check("minimum policy residual rows")
    early_blocked_metrics = _metrics(
        markets=750,
        coverage=0.75,
        accuracy=0.861,
        median=115.0,
    )
    early_blocked_metrics.update(
        {
            "up_recall": 0.860,
            "down_recall": 0.862,
            "expected_calibration_error": 0.061,
        }
    )
    rescue_blocked_metrics = _metrics(
        markets=790,
        coverage=0.79,
        accuracy=0.868,
        median=128.0,
    )
    fold = {
        "fold_index": 2,
        "causal_order_verified": True,
        "selector_fit_folds": [0],
        "prior_fold_index": 1,
        "prior_fold_split": {
            "method": "chronological_market_50_50",
            "source_markets": 2_000,
            "calibration_markets": 1_000,
            "policy_markets": 1_000,
        },
        "thresholds": {
            "early": {
                "head": "early",
                "threshold": 0.91,
                "qualified": True,
                "thresholds_evaluated": 4,
                "qualifying_thresholds": 2,
                "metrics": candidates[EARLY]["out_of_fold"],
                "objective": {
                    "residual_markets": 520,
                    "residual_accuracy": 0.88,
                    "rescued_markets": 20,
                },
                "checks": [threshold_check],
                "threshold_diagnostics": [
                    _threshold_diagnostic(
                        threshold=0.87,
                        qualified=False,
                        metrics=early_blocked_metrics,
                        residual_markets=610,
                        residual_accuracy=0.852,
                        by_120_uplift=0.025,
                        rescued_markets=30,
                        failed_gate="minimum residual accuracy",
                    ),
                    _threshold_diagnostic(
                        threshold=0.91,
                        qualified=True,
                        metrics=candidates[EARLY]["out_of_fold"],
                        residual_markets=520,
                        residual_accuracy=0.88,
                        by_120_uplift=0.02,
                        rescued_markets=20,
                    ),
                ],
            },
            "rescue": {
                "head": "rescue",
                "threshold": 0.93,
                "qualified": True,
                "thresholds_evaluated": 4,
                "qualifying_thresholds": 1,
                "metrics": candidates[RESCUE]["out_of_fold"],
                "objective": {
                    "residual_markets": 520,
                    "residual_accuracy": 0.88,
                    "rescued_markets": 520,
                },
                "checks": [threshold_check],
                "threshold_diagnostics": [
                    _threshold_diagnostic(
                        threshold=0.89,
                        qualified=False,
                        metrics=rescue_blocked_metrics,
                        residual_markets=590,
                        residual_accuracy=0.86,
                        by_120_uplift=0.01,
                        rescued_markets=240,
                        failed_gate="minimum policy accuracy",
                    ),
                    _threshold_diagnostic(
                        threshold=0.93,
                        qualified=True,
                        metrics=candidates[RESCUE]["out_of_fold"],
                        residual_markets=520,
                        residual_accuracy=0.88,
                        by_120_uplift=0.02,
                        rescued_markets=520,
                    ),
                ],
            },
        },
        "candidates": {
            name: _fold_candidate(name, candidate)
            for name, candidate in candidates.items()
        },
    }
    for candidate in candidates.values():
        candidate["folds"] = [fold["candidates"][candidate["candidate"]]]
    return {
        "schema_version": RESIDUAL_ADMISSION_SCHEMA_VERSION,
        "run_id": "20260729T120000Z",
        "created_at": "2026-07-29T12:00:00+00:00",
        "configuration": {
            "quantity": 5.0,
            "candidate_names": [CONTROL, EARLY, RESCUE, COMBINED],
            "early_head": {"candidate": EARLY},
            "rescue_head": {"candidate": RESCUE},
            "combined_candidate": COMBINED,
            "gates": {
                "minimum_accuracy": 0.874,
                "minimum_balanced_accuracy": 0.874,
                "minimum_direction_recall": 0.874,
                "minimum_wilson_lower_95": 0.865,
                "maximum_expected_calibration_error": 0.05,
            },
        },
        "evaluation_note": "Consumed chronological development evidence.",
        "evaluation_is_independent": False,
        "runtime_provenance": {},
        "probability_evidence": {
            "manifest": "/tmp/probabilities/manifest.json",
            "manifest_sha256": "a" * 64,
            "source_profile": "residual_admission_source",
            "source_candidates": [CONTROL, "histogram_boundary_reversal"],
            "source_fold_count": 7,
            "checksums_verified": True,
        },
        "core_features": {
            "path": "/tmp/features.parquet",
            "sha256": "b" * 64,
            "rows": 350_000,
            "markets": 10_000,
            "holdout_accessed": False,
        },
        "execution_evidence": {
            "path": "/tmp/execution",
            "manifest": {},
            "role": "strict decision-time economics diagnostic only",
            "known_gap": "no strict cached rows on June 15-29",
        },
        "selector_training": {},
        "control_candidate": CONTROL,
        "proposal_candidate": "histogram_boundary_reversal",
        "candidates": candidates,
        "folds": [fold],
        "benchmark_passed_candidates": [EARLY, RESCUE, COMBINED],
        "winner": COMBINED,
        "deployment": {
            "status": "blocked",
            "runtime_exported": False,
            "runtime_changed": False,
            "reasons": [
                "evaluation evidence is consumed development evidence",
                "five-fold strict execution evidence is incomplete",
                "a new forward post-freeze cohort is required",
            ],
        },
    }


def test_report_is_self_contained_and_surfaces_residual_contract(
    tmp_path: Path,
) -> None:
    destination = generate_residual_admission_report(
        _benchmark(),
        tmp_path / "nested" / "report.html",
    )
    contents = destination.read_text()

    assert destination.is_file()
    assert "Development evidence only" in contents
    assert "Control vs residual outcomes" in contents
    assert "Early · histogram_boundary_residual_early" in contents
    assert "Rescue · histogram_boundary_residual_rescue" in contents
    assert "Combined · histogram_boundary_residual_combined" in contents
    assert "Cumulative decision coverage" in contents
    assert "60 sec" in contents
    assert "90 sec" in contents
    assert "120 sec" in contents
    assert "180 sec" in contents
    assert "240 sec" in contents
    assert "51.00%" in contents
    assert "+2.00 pp" in contents
    assert "Paired control attribution" in contents
    assert "Rescued control NoTrade" in contents
    assert "Per-fold frozen thresholds" in contents
    assert "0.910" in contents
    assert "Threshold grid diagnostics" in contents
    assert "Every frozen q candidate" in contents
    assert "Policy accuracy" in contents
    assert "UP recall" in contents
    assert "DOWN recall" in contents
    assert "Δ120 vs control" in contents
    assert "Failed gates" in contents
    assert "0.870" in contents
    assert "86.10%" in contents
    assert "86.00%" in contents
    assert "86.20%" in contents
    assert "6.10%" in contents
    assert "75.00%" in contents
    assert "610" in contents
    assert "85.20%" in contents
    assert "+2.50 pp" in contents
    assert "minimum residual accuracy" in contents
    assert "minimum policy accuracy" in contents
    assert "Absolute quality" in contents
    assert "Five-share execution economics" in contents
    assert "no strict cached rows on June 15-29" in contents
    assert "Deployment remains blocked" in contents
    assert "a new forward post-freeze cohort is required" in contents
    assert "Deterministic benchmark record" in contents
    assert "<script" not in contents
    assert "<link" not in contents
    assert "https://" not in contents


def test_report_lists_failed_candidate_gates() -> None:
    benchmark = _benchmark()
    candidate = benchmark["candidates"][EARLY]
    candidate["advance"]["benchmark_passed"] = False
    candidate["advance"]["checks"] = [_check("minimum residual accuracy", False)]

    contents = render_residual_admission_report(benchmark)

    assert "minimum residual accuracy" in contents
    assert "0.860000" in contents
    assert "blocked" in contents


def test_report_rejects_unknown_schema() -> None:
    benchmark = deepcopy(_benchmark())
    benchmark["schema_version"] = "btc-residual-admission-benchmark-v999"

    with pytest.raises(
        ValueError,
        match="unsupported residual-admission benchmark schema",
    ):
        render_residual_admission_report(benchmark)
