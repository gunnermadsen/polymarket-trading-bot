from __future__ import annotations

import sys
from pathlib import Path

from btc_directional_model import cli
from btc_directional_model.policy_benchmark import (
    SAVED_POLICY_BENCHMARK_SCHEMA_VERSION,
)
from btc_directional_model.policy_report import generate_saved_policy_report


def _metrics(
    *,
    coverage: float,
    accuracy: float,
) -> dict[str, object]:
    return {
        "markets": 700,
        "eligible_markets": 1_000,
        "coverage": coverage,
        "correct": round(700 * accuracy),
        "accuracy": accuracy,
        "wilson_lower_95": 0.865,
        "wilson_upper_95": 0.91,
        "balanced_accuracy": accuracy,
        "up_precision": accuracy,
        "up_recall": accuracy,
        "down_precision": accuracy,
        "down_recall": accuracy,
        "f1": accuracy,
        "matthews_correlation": 0.75,
        "brier_score": 0.10,
        "log_loss": 0.30,
        "roc_auc": 0.93,
        "expected_calibration_error": 0.03,
        "confusion_matrix": [[310, 40], [40, 310]],
        "predicted_up": 350,
        "predicted_down": 350,
        "actual_up": 350,
        "actual_down": 350,
        "maximum_consecutive_losses": 2,
    }


def _candidate(
    name: str,
    *,
    benchmark_passed: bool,
) -> dict[str, object]:
    metrics = _metrics(coverage=0.70, accuracy=0.886)
    timing = {
        "markets": 700,
        "eligible_markets": 1_000,
        "coverage": 0.70,
        "median_first_crossing_seconds": 120.0,
        "p90_first_crossing_seconds": 180.0,
        "early_markets": 500,
        "early_coverage": 0.50,
        "bands": [],
    }
    check = {
        "name": "minimum accuracy",
        "observed": 0.886,
        "operator": ">=",
        "required": 0.874,
        "passed": True,
    }
    return {
        "candidate": name,
        "target_kind": "outcome",
        "feature_kind": "btc_path",
        "calibration_kind": "time_banded_platt",
        "row_weight_schedule": {},
        "folds": [
            {
                "fold_index": 0,
                "causal_order_verified": True,
                "policy_selection_range": {
                    "start": "2026-05-01T00:00:00+00:00",
                    "end": "2026-05-07T23:55:00+00:00",
                },
                "validation_range": {
                    "start": "2026-05-08T00:00:00+00:00",
                    "end": "2026-05-14T23:55:00+00:00",
                },
                "policy_selection": {
                    "thresholds": {
                        "60-89": 0.90,
                        "90-119": 0.90,
                        "120-179": 0.91,
                        "180-240": 0.92,
                    },
                    "qualified": True,
                    "combinations_evaluated": 10_000,
                    "qualifying_combinations": 5,
                    "metrics": metrics,
                    "timing": timing,
                    "checks": [check],
                },
                "validation": {
                    "metrics": metrics,
                    "timing": timing,
                    "checks": [check],
                    "qualified": True,
                    "thresholds_frozen_before_access": True,
                    "threshold_search_performed": False,
                },
            }
        ],
        "fold_count": 1,
        "policy_qualified_folds": 1,
        "validation_qualified_folds": 1,
        "out_of_fold": metrics,
        "timing": timing,
        "no_trade_rate": 0.30,
        "validation_threshold_searches": 0,
        "validation_score_passes": 1,
        "advance": {
            "checks": [check],
            "benchmark_passed": benchmark_passed,
            "deployment_qualified": False,
        },
    }


def _benchmark() -> dict[str, object]:
    control = _candidate("histogram_enriched", benchmark_passed=False)
    challenger = _candidate("early_weighted", benchmark_passed=True)
    return {
        "schema_version": SAVED_POLICY_BENCHMARK_SCHEMA_VERSION,
        "run_id": "20260728T120000Z",
        "created_at": "2026-07-28T12:00:00+00:00",
        "configuration": {},
        "evaluation_note": "Consumed chronological development evidence.",
        "evaluation_is_independent": False,
        "probability_evidence": {
            "manifest": "/tmp/source/saved-policy-probabilities/manifest.json",
            "manifest_sha256": "a" * 64,
            "schema_version": "btc-saved-policy-probabilities-v1",
            "created_at": "2026-07-28T11:00:00+00:00",
            "source_benchmark_profile": "accuracy_timing",
            "source_config": "/tmp/source.toml",
            "source_config_sha256": "b" * 64,
            "control_candidate": "histogram_enriched",
            "candidate_names": ["histogram_enriched", "early_weighted"],
            "fold_count": 1,
            "causal_contract": "selection rows precede validation rows",
            "checksums_verified": True,
            "causal_contract_verified": True,
        },
        "runtime_provenance": {},
        "control_candidate": "histogram_enriched",
        "candidates": {
            "histogram_enriched": control,
            "early_weighted": challenger,
        },
        "benchmark_passed_candidates": ["early_weighted"],
        "winner": "early_weighted",
        "deployment": {
            "status": "not_authorized",
            "runtime_exported": False,
            "runtime_changed": False,
            "scope": "offline training and saved-probability policy evidence only",
        },
    }


def test_saved_policy_report_is_self_contained_and_explicitly_development_only(
    tmp_path: Path,
) -> None:
    destination = generate_saved_policy_report(
        _benchmark(),
        tmp_path / "report.html",
    )
    contents = destination.read_text()

    assert "Development-only · non-independent validation" in contents
    assert "cannot authorize deployment" in contents
    assert "early_weighted" in contents
    assert "88.60%" in contents
    assert "30.00%" in contents
    assert "not_authorized" in contents
    assert "a" * 64 in contents
    assert "selection rows precede validation rows" in contents
    assert "Deterministic benchmark record" in contents
    assert "<script src=" not in contents


def test_saved_policy_cli_dispatches_and_reports_runtime_boundary(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "report.html").write_text("<html></html>")
    config = object()
    benchmark = _benchmark()
    monkeypatch.setattr(
        cli,
        "load_saved_policy_benchmark_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        cli,
        "run_saved_policy_benchmark",
        lambda loaded: (run_dir, benchmark),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "btc-directional-model",
            "persistence-policy-benchmark-run",
            "--config",
            str(tmp_path / "policy.toml"),
        ],
    )

    cli.main()

    output = capsys.readouterr().out
    assert f"report: {run_dir / 'report.html'}" in output
    assert "benchmark passed: early_weighted" in output
    assert "winner: early_weighted" in output
    assert "evidence: non-independent development validation" in output
    assert "runtime: unchanged" in output
