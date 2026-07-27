from __future__ import annotations

from pathlib import Path

from btc_directional_model.report import generate_report


def compact_metrics() -> dict:
    score = {
        "accuracy": 0.62,
        "wilson_lower_95": 0.59,
        "markets": 100,
        "maximum_consecutive_losses": 4,
        "confusion_matrix": [[30, 20], [18, 32]],
    }
    group = {
        "test": score,
        "train": {**score, "accuracy": 0.63},
        "calibration": {**score, "accuracy": 0.66},
        "first_executable_test": {**score, "accuracy": 0.68, "markets": 70},
        "selected_prediction_executable": {**score, "accuracy": 0.64, "markets": 20},
        "test_baselines": {
            "binance_sign": {**score, "accuracy": 0.55},
            "polymarket_favorite": {**score, "accuracy": 0.70, "markets": 10},
            "majority_up": {**score, "accuracy": 0.50},
        },
        "threshold_history": [
            {"threshold": 0.5, "accuracy": 0.66, "markets": 100},
            {"threshold": 0.6, "accuracy": 0.70, "markets": 80},
        ],
        "confidence_buckets": [{"bucket": 0.6, "accuracy": 0.62, "markets": 100}],
        "time_buckets": [{"seconds_elapsed": 60, "accuracy": 0.62, "markets": 100}],
        "daily_accuracy": [{"date": "2026-05-20", "accuracy": 0.62, "markets": 100}],
        "coefficient_ranking": [{"feature": "btc_gap_from_open_bps", "coefficient": 1.0}],
        "tuning_history": [{"c": 0.1, "validation_log_loss": 0.6}],
        "confidence_threshold": 0.5,
        "threshold_qualified_on_calibration": True,
        "converged": True,
    }
    split = {
        "train": {
            "range_start": "2026-04-21T00:00:00+00:00",
            "range_end": "2026-05-09T00:00:00+00:00",
            "markets": 300,
        },
        "calibration": {
            "range_start": "2026-05-09T00:05:00+00:00",
            "range_end": "2026-05-15T00:00:00+00:00",
            "markets": 100,
        },
        "test": {
            "range_start": "2026-05-15T00:05:00+00:00",
            "range_end": "2026-05-20T23:55:00+00:00",
            "markets": 100,
        },
    }
    return {
        "run_id": "test-run",
        "selected_feature_group": "btc_path",
        "groups": {"btc_path": group},
        "qualification": {
            "passed": False,
            "calibration_threshold_passed": True,
            "target_accuracy": 0.65,
            "target_wilson_lower": 0.60,
            "minimum_test_markets": 50,
            "maximum_train_test_accuracy_gap": 0.05,
            "observed_train_test_accuracy_gap": 0.01,
        },
        "data": {
            "candidate_markets": 500,
            "candidate_rows": 18_500,
            "markets_with_final_price": 490,
            "matching_final_price_labels": 490,
            "class_up_markets": 250,
            "class_down_markets": 250,
        },
        "split": split,
        "training_backend": {
            "name": "scikit-learn-logistic-regression",
            "device": "cpu",
            "reason": "deterministic linear baseline",
        },
    }


def test_generate_self_contained_report(tmp_path: Path) -> None:
    destination = generate_report(tmp_path, compact_metrics())
    contents = destination.read_text()

    assert "NOT QUALIFIED" in contents
    assert "First-executable policy accuracy" in contents
    assert "Chronological-holdout confusion matrix" in contents
    assert "plotly.js" in contents
    assert (tmp_path / "progress.json").exists()
