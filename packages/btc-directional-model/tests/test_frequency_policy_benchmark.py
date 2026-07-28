from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from btc_directional_model.core_extract import file_sha256
from btc_directional_model.frequency_policy_benchmark import (
    evaluate_frequency_policy_candidate,
    frequency_policy_advancement_checks,
)
from btc_directional_model.frequency_policy_config import (
    FrequencyPolicyAdvancementGates,
    FrequencyPolicyBenchmarkConfig,
)
from btc_directional_model.persistence_benchmark import (
    SAVED_POLICY_PROBABILITY_SCHEMA_VERSION,
)
from btc_directional_model.policy_config import PolicyThresholdBand

BANDS = (
    PolicyThresholdBand("60-89", 60, 90),
    PolicyThresholdBand("90-119", 90, 120),
    PolicyThresholdBand("120-179", 120, 180),
    PolicyThresholdBand("180-240", 180, 241),
)


def frequency_config(tmp_path: Path) -> FrequencyPolicyBenchmarkConfig:
    return FrequencyPolicyBenchmarkConfig(
        source_path=tmp_path / "frequency.toml",
        package_root=tmp_path,
        probability_manifest=tmp_path / "manifest.json",
        qualification_objective="frequency",
        policy_selection_mode="single_frozen",
        policy_anchor_fold=0,
        control_candidate="histogram_enriched",
        candidate_names=(
            "histogram_enriched",
            "histogram_path_persistence_time_calibrated",
            "histogram_path_persistence_time_calibrated_60_120",
            "histogram_path_persistence_time_calibrated_90_120",
        ),
        evaluation_note="Consumed chronological development evidence.",
        evaluation_is_independent=False,
        quantity=5.0,
        threshold_candidates=(0.80, 0.90),
        bands=BANDS,
        gates=frequency_gates(minimum_markets=2),
        execution_evidence=tmp_path / "execution",
        runs=tmp_path / "runs",
    )


def frequency_gates(
    *,
    minimum_markets: int,
) -> FrequencyPolicyAdvancementGates:
    return FrequencyPolicyAdvancementGates(
        minimum_accuracy=0.874,
        minimum_balanced_accuracy=0.874,
        minimum_direction_recall=0.874,
        minimum_wilson_lower_95=0.0,
        maximum_expected_calibration_error=0.05,
        minimum_coverage=0.55,
        minimum_selected_markets=minimum_markets,
        minimum_coverage_uplift=1e-9,
        maximum_accuracy_regression=0.0,
        maximum_balanced_accuracy_regression=0.0,
        maximum_direction_recall_regression=0.0,
        minimum_common_checkpoint_markets=minimum_markets,
        minimum_executable_markets=minimum_markets,
        minimum_mean_direct_edge_per_share=0.0,
        minimum_realized_net_per_share=0.0,
        require_every_fold=True,
    )


def probability_rows(
    candidate: str,
    fold_index: int,
    start: datetime,
    *,
    markets: int = 20,
) -> pl.DataFrame:
    rows = []
    for market_index in range(markets):
        window_start = start + timedelta(minutes=5 * market_index)
        label = market_index % 2
        for second in (60, 90, 120, 180, 240):
            confidence = 0.96
            probability_up = confidence if label else 1.0 - confidence
            rows.append(
                {
                    "candidate": candidate,
                    "fold_index": fold_index,
                    "market_id": f"fold-{fold_index}-market-{market_index}",
                    "window_start": window_start,
                    "observed_at": window_start + timedelta(seconds=second),
                    "seconds_elapsed": second,
                    "label_up": label,
                    "probability_up": probability_up,
                    "predicted_up": label,
                    "confidence": confidence,
                    "correct": True,
                    "baseline_correct": True,
                }
            )
    return pl.DataFrame(rows)


def probability_manifest(
    tmp_path: Path,
    candidate: str,
) -> dict[str, object]:
    candidate_dir = tmp_path / candidate
    candidate_dir.mkdir()
    fold_records = []
    for fold_index in (0, 1):
        policy = probability_rows(
            candidate,
            fold_index,
            datetime(2026, 5, 1, tzinfo=UTC),
        )
        validation = probability_rows(
            candidate,
            fold_index,
            datetime(2026, 6, 1 + 7 * fold_index, tzinfo=UTC),
        )
        records = {}
        for role, frame in (
            ("policy_selection", policy),
            ("validation", validation),
        ):
            path = candidate_dir / f"fold-{fold_index}-{role}.parquet"
            frame.write_parquet(path)
            records[role] = {
                "path": str(path.relative_to(tmp_path)),
                "sha256": file_sha256(path),
                "rows": frame.height,
                "markets": frame["market_id"].n_unique(),
            }
        fold_records.append(
            {
                "fold_index": fold_index,
                "causal_order_verified": True,
                **records,
            }
        )
    return {
        "schema_version": SAVED_POLICY_PROBABILITY_SCHEMA_VERSION,
        "fold_count": 2,
        "candidates": {
            candidate: {
                "target_kind": "path_persistence",
                "feature_kind": "core_prewindow",
                "calibration_kind": "time_banded_platt",
                "row_weight_schedule": {
                    "start_second": 60,
                    "end_second_inclusive": 120,
                    "multiplier": 1.5,
                },
                "folds": fold_records,
            }
        },
    }


def test_one_anchor_policy_is_applied_unchanged_to_every_validation_fold(
    tmp_path: Path,
) -> None:
    candidate = "histogram_path_persistence_time_calibrated_60_120"
    config = frequency_config(tmp_path)
    manifest = probability_manifest(tmp_path, candidate)

    result, scored, _ = evaluate_frequency_policy_candidate(
        config,
        manifest,
        candidate,
    )

    expected = result["single_policy"]["thresholds"]
    assert result["single_policy"]["source_fold_index"] == 0
    assert result["single_policy"]["applied_validation_folds"] == 2
    assert result["validation_threshold_searches"] == 0
    assert result["validation_score_passes"] == 2
    assert result["single_policy_qualified"] is True
    assert scored.filter(pl.col("policy_selected"))["seconds_elapsed"].unique().to_list() == [60]
    for fold in result["folds"]:
        assert fold["policy_selection"]["thresholds"] == expected
        assert fold["policy_selection"]["source_fold_index"] == 0
        assert fold["validation"]["threshold_search_performed"] is False
        assert fold["validation"]["thresholds_frozen_before_access"] is True


def test_frequency_advancement_keeps_quality_fold_checkpoint_and_economics() -> None:
    metrics = {
        "markets": 700,
        "eligible_markets": 1_000,
        "coverage": 0.70,
        "accuracy": 0.89,
        "balanced_accuracy": 0.89,
        "up_recall": 0.89,
        "down_recall": 0.89,
        "wilson_lower_95": 0.88,
        "expected_calibration_error": 0.02,
    }
    control_metrics = {**metrics, "coverage": 0.60}
    fold = {
        "fold_index": 0,
        "validation": {
            "metrics": metrics,
        },
    }
    control_fold = {
        "fold_index": 0,
        "validation": {
            "metrics": metrics,
        },
    }
    candidate = {
        "out_of_fold": metrics,
        "timing": {"median_first_crossing_seconds": 200.0},
        "single_policy_qualified": True,
        "validation_qualified_folds": 1,
        "fold_count": 1,
        "folds": [fold],
        "execution": {
            "economic_markets": 700,
            "mean_direct_edge_per_share": 0.05,
            "realized_net_expectancy_per_trade": 0.10,
        },
    }
    control = {
        "out_of_fold": control_metrics,
        "timing": {"median_first_crossing_seconds": 100.0},
        "folds": [control_fold],
    }
    comparison = {
        "checkpoints": [
            {
                "seconds_elapsed": second,
                "common_markets": 1_000,
                "accuracy_delta": 0.0,
                "balanced_accuracy_delta": 0.0,
                "up_recall_delta": 0.0,
                "down_recall_delta": 0.0,
            }
            for second in (60, 90, 120, 180, 240)
        ]
    }

    advancement = frequency_policy_advancement_checks(
        candidate,
        control,
        frequency_gates(minimum_markets=500),
        comparison=comparison,
        is_control=False,
        quantity=5.0,
    )

    names = {check["name"] for check in advancement["checks"]}
    assert advancement["benchmark_passed"] is True
    assert advancement["timing_gates_applied"] is False
    assert not any("median" in name for name in names)
    assert "validation qualified in every fold" in names
    assert "60s common-time accuracy does not regress" in names
    assert "minimum executable evaluation markets" in names
    assert "minimum realized net expectancy per share" in names
