from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from btc_directional_model.core_config import (
    CandidateRowWeightScheduleConfig,
    RowWeightScheduleConfig,
)
from btc_directional_model.core_extract import file_sha256
from btc_directional_model.persistence_benchmark import (
    write_saved_probability_manifest,
)
from btc_directional_model.persistence_config import (
    ACCURACY_TIMING_CANDIDATES,
    ACCURACY_TIMING_PROFILE,
    load_persistence_benchmark_config,
)
from btc_directional_model.policy_benchmark import (
    apply_time_band_policy,
    select_causal_time_band_thresholds,
)
from btc_directional_model.policy_config import (
    PolicyAdvancementGates,
    PolicyThresholdBand,
    SavedPolicyBenchmarkConfig,
)

BANDS = (
    PolicyThresholdBand("60-89", 60, 90),
    PolicyThresholdBand("90-119", 90, 120),
    PolicyThresholdBand("120-179", 120, 180),
    PolicyThresholdBand("180-240", 180, 241),
)


def policy_config(tmp_path: Path) -> SavedPolicyBenchmarkConfig:
    return SavedPolicyBenchmarkConfig(
        source_path=tmp_path / "policy.toml",
        package_root=tmp_path,
        probability_manifest=tmp_path / "manifest.json",
        control_candidate="histogram_enriched",
        candidate_names=(
            "histogram_enriched",
            "histogram_path_persistence_time_calibrated",
            "histogram_path_persistence_time_calibrated_60_120",
            "histogram_path_persistence_time_calibrated_90_120",
        ),
        evaluation_note="Consumed chronological development evidence.",
        evaluation_is_independent=False,
        threshold_candidates=(0.80, 0.90),
        bands=BANDS,
        gates=PolicyAdvancementGates(
            minimum_accuracy=0.874,
            minimum_balanced_accuracy=0.874,
            minimum_direction_recall=0.874,
            minimum_wilson_lower_95=0.865,
            maximum_expected_calibration_error=0.05,
            minimum_coverage=0.55,
            minimum_selected_markets=500,
            minimum_coverage_uplift=1e-9,
            maximum_accuracy_regression=0.0,
            maximum_balanced_accuracy_regression=0.0,
            maximum_direction_recall_regression=0.0,
            maximum_median_entry_second=125.0,
            minimum_median_entry_improvement_seconds=5.0,
            require_every_fold=True,
        ),
        runs=tmp_path / "runs",
    )


def probability_rows(markets: int = 2) -> pl.DataFrame:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    rows = []
    for market_index in range(markets):
        window_start = start + timedelta(minutes=5 * market_index)
        label = market_index % 2
        for second, confidence in (
            (60, 0.82),
            (90, 0.91),
            (120, 0.94),
            (180, 0.96),
        ):
            probability_up = confidence if label else 1.0 - confidence
            rows.append(
                {
                    "candidate": "histogram_enriched",
                    "fold_index": 0,
                    "market_id": f"market-{market_index}",
                    "window_start": window_start,
                    "observed_at": window_start + timedelta(seconds=second),
                    "seconds_elapsed": second,
                    "label_up": label,
                    "binance_sign_up": label,
                    "probability_up": probability_up,
                    "predicted_up": label,
                    "confidence": confidence,
                    "correct": True,
                    "baseline_correct": True,
                }
            )
    return pl.DataFrame(rows)


def test_time_band_policy_uses_the_first_causal_crossing() -> None:
    scored = apply_time_band_policy(
        probability_rows(),
        BANDS,
        {band.name: 0.90 for band in BANDS},
    )
    selected = scored.filter(pl.col("policy_selected")).sort("market_id")

    assert selected["seconds_elapsed"].to_list() == [90, 90]
    assert selected["policy_threshold_band"].to_list() == [
        "90-119",
        "90-119",
    ]


def test_threshold_search_returns_four_frozen_band_values(
    tmp_path: Path,
) -> None:
    selection = select_causal_time_band_thresholds(
        probability_rows(),
        policy_config(tmp_path),
    )

    assert selection.combinations_evaluated == 16
    assert tuple(name for name, _ in selection.thresholds) == tuple(band.name for band in BANDS)
    assert selection.metrics["markets"] == 2
    assert selection.qualified is False


def test_later_validation_rows_cannot_enter_threshold_selection(
    tmp_path: Path,
) -> None:
    policy = probability_rows()
    selection = select_causal_time_band_thresholds(
        policy,
        policy_config(tmp_path),
    )
    validation = policy.with_columns(
        (1 - pl.col("probability_up")).alias("probability_up"),
        (1 - pl.col("predicted_up")).alias("predicted_up"),
        pl.lit(False).alias("correct"),
    )
    scored = apply_time_band_policy(
        validation,
        BANDS,
        selection.threshold_map(),
    )

    assert selection.combinations_evaluated == 16
    assert scored.filter(pl.col("policy_selected")).height == 2


def test_fold_probability_manifest_persists_both_causal_roles_with_checksums(
    tmp_path: Path,
) -> None:
    package_root = Path(__file__).resolve().parents[1]
    frozen = load_persistence_benchmark_config(
        package_root / "configs" / "btc-5m-directional-path-persistence-20260421-20260720.toml"
    )
    schedules = (
        CandidateRowWeightScheduleConfig(
            ACCURACY_TIMING_CANDIDATES[0],
            RowWeightScheduleConfig(None, None, 1.0),
        ),
        CandidateRowWeightScheduleConfig(
            ACCURACY_TIMING_CANDIDATES[1],
            RowWeightScheduleConfig(None, None, 1.0),
        ),
        CandidateRowWeightScheduleConfig(
            ACCURACY_TIMING_CANDIDATES[2],
            RowWeightScheduleConfig(60, 120, 1.5),
        ),
        CandidateRowWeightScheduleConfig(
            ACCURACY_TIMING_CANDIDATES[3],
            RowWeightScheduleConfig(90, 120, 2.0),
        ),
    )
    config = replace(
        frozen,
        profile=ACCURACY_TIMING_PROFILE,
        candidate_names=ACCURACY_TIMING_CANDIDATES,
        row_weight_schedules=schedules,
    )
    policy = probability_rows()
    validation = policy.with_columns(
        pl.col("window_start") + timedelta(days=7),
        pl.col("observed_at") + timedelta(days=7),
    )
    results = {}
    for candidate_name in ACCURACY_TIMING_CANDIDATES:
        results[candidate_name] = {
            "policy_scored_rows": policy.with_columns(pl.lit(candidate_name).alias("candidate")),
            "validation_probability_rows": validation.with_columns(
                pl.lit(candidate_name).alias("candidate")
            ),
        }
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    record = write_saved_probability_manifest(run_dir, results, config)
    manifest_path = run_dir / record["manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    first_fold = manifest["candidates"][ACCURACY_TIMING_CANDIDATES[0]]["folds"][0]

    assert first_fold["causal_order_verified"] is True
    for role in ("policy_selection", "validation"):
        evidence = first_fold[role]
        evidence_path = manifest_path.parent / evidence["path"]
        assert evidence_path.is_file()
        assert file_sha256(evidence_path) == evidence["sha256"]
