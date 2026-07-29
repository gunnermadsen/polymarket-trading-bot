from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model.admission_benchmark import (
    SELECTOR_BASE_FEATURES,
    SELECTOR_FEATURES,
    admission_advancement_checks,
    build_causal_selector_features,
    equal_market_row_weights,
    fit_admission_selector,
    selector_probability_frame,
    split_prior_fold_chronologically,
)
from btc_directional_model.admission_config import (
    ADMISSION_BASE_CANDIDATE,
    ADMISSION_CONTROL_CANDIDATE,
    ADMISSION_EVALUATION_FOLDS,
    ADMISSION_SELECTOR_CANDIDATE,
    AdmissionAdvancementGates,
    AdmissionBenchmarkConfig,
    AdmissionSelectorConfig,
    validate_admission_benchmark_config,
)
from btc_directional_model.core_features import CORE_BOUNDARY_FEATURES
from btc_directional_model.persistence_benchmark import (
    SAVED_POLICY_PROBABILITY_SCHEMA_VERSION,
)
from btc_directional_model.policy_benchmark import apply_time_band_policy
from btc_directional_model.policy_config import PolicyThresholdBand

FROZEN_BANDS = (
    PolicyThresholdBand("60-89", 60, 90),
    PolicyThresholdBand("90-119", 90, 120),
    PolicyThresholdBand("120-179", 120, 180),
    PolicyThresholdBand("180-240", 180, 241),
)


def test_causal_selector_features_do_not_change_before_future_mutation() -> None:
    rows = _base_probability_rows(markets=2, seconds=(60, 65, 70, 75, 80))
    original = build_causal_selector_features(rows)
    mutated = rows.with_columns(
        pl.when(pl.col("seconds_elapsed") >= 75)
        .then(pl.lit(0.99))
        .otherwise(pl.col("probability_up"))
        .alias("probability_up"),
        pl.when(pl.col("seconds_elapsed") >= 75)
        .then(pl.lit(0.99))
        .otherwise(pl.col("confidence"))
        .alias("confidence"),
        pl.when(pl.col("seconds_elapsed") >= 75)
        .then(pl.lit(1))
        .otherwise(pl.col("predicted_up"))
        .alias("predicted_up"),
    )
    mutated_features = build_causal_selector_features(mutated)
    keys_and_features = [
        "market_id",
        "seconds_elapsed",
        *SELECTOR_BASE_FEATURES,
    ]
    expected = original.filter(pl.col("seconds_elapsed") <= 70).select(
        keys_and_features
    )
    observed = mutated_features.filter(pl.col("seconds_elapsed") <= 70).select(
        keys_and_features
    )
    assert expected.to_dicts() == observed.to_dicts()


def test_prior_fold_split_is_market_disjoint_and_chronological() -> None:
    frame = _base_probability_rows(
        markets=6,
        seconds=(60, 65),
    )
    calibration, policy, metadata = split_prior_fold_chronologically(frame)
    calibration_markets = set(calibration["market_id"].to_list())
    policy_markets = set(policy["market_id"].to_list())
    assert calibration_markets.isdisjoint(policy_markets)
    assert calibration_markets | policy_markets == set(frame["market_id"].to_list())
    assert calibration["window_start"].max() < policy["window_start"].min()
    assert metadata["calibration_markets"] == 3
    assert metadata["policy_markets"] == 3


def test_selector_abstains_below_half_and_never_reverses_base_direction() -> None:
    frame = _base_probability_rows(
        markets=4,
        seconds=(60,),
    ).with_columns(
        pl.Series("predicted_up", [0, 0, 1, 1], dtype=pl.Int8),
        pl.Series("label_up", [0, 1, 1, 0], dtype=pl.Int32),
        pl.Series("probability_up", [0.2, 0.3, 0.8, 0.7], dtype=pl.Float64),
        pl.Series("confidence", [0.8, 0.7, 0.8, 0.7], dtype=pl.Float64),
        pl.Series("correct", [True, False, True, False], dtype=pl.Boolean),
    )
    scored = selector_probability_frame(
        frame,
        np.array([0.40, 0.50, 0.80, 0.90]),
        selector_candidate=ADMISSION_SELECTOR_CANDIDATE,
        admission_floor=0.5,
    )
    assert scored["predicted_up"].to_list() == [0, 0, 1, 1]
    assert scored["model_eligible"].to_list() == [False, True, True, True]
    assert scored["probability_up"][1] < 0.5
    thresholds = {band.name: 0.55 for band in FROZEN_BANDS}
    policy = apply_time_band_policy(scored, FROZEN_BANDS, thresholds)
    assert policy.filter(
        (pl.col("selector_q") < 0.5) & pl.col("policy_selected")
    ).is_empty()
    assert policy.filter(
        pl.col("predicted_up") != pl.col("selector_base_predicted_up_output")
    ).is_empty()


def test_selector_fit_uses_only_frozen_core_feature_allowlist() -> None:
    rng = np.random.default_rng(20260728)
    markets = 40
    rows_per_market = 4
    total = markets * rows_per_market
    payload: dict[str, object] = {
        "market_id": [
            f"market-{market:03d}"
            for market in range(markets)
            for _ in range(rows_per_market)
        ],
        "correct": np.tile([True, False, True, False], markets),
    }
    for feature in SELECTOR_FEATURES:
        payload[feature] = rng.normal(size=total)
    frame = pl.DataFrame(payload)
    model = fit_admission_selector(
        frame,
        regularization_c=1.0,
        random_seed=20260728,
    )
    logits = model.raw_logit(frame)
    assert model.feature_names == (
        *SELECTOR_BASE_FEATURES,
        *CORE_BOUNDARY_FEATURES,
    )
    assert np.isfinite(logits).all()
    weights = equal_market_row_weights(frame)
    assert weights.sum() == pytest.approx(total)
    assert np.unique(weights).size == 1


def test_development_gate_can_pass_but_three_folds_never_qualify_deployment() -> None:
    gates = _gates()
    control = _metrics(
        coverage=0.55,
        accuracy=0.88,
        balanced_accuracy=0.88,
        median=130.0,
    )
    selector = _metrics(
        coverage=0.60,
        accuracy=0.89,
        balanced_accuracy=0.89,
        median=120.0,
    )
    folds = [
        {
            "validation": {
                "qualified": True,
                "direction_reversals": 0,
                "q_below_floor_selected_rows": 0,
            }
        }
        for _ in ADMISSION_EVALUATION_FOLDS
    ]
    comparison = {
        "checkpoints": [
            {"common_markets": 600, "seconds_elapsed": second}
            for second in (60, 90, 120, 180, 240)
        ]
    }
    result = admission_advancement_checks(
        candidate_name=ADMISSION_SELECTOR_CANDIDATE,
        control_candidate=ADMISSION_CONTROL_CANDIDATE,
        metrics=selector,
        control_metrics=control,
        fold_results=folds,
        comparison=comparison,
        gates=gates,
        quantity=5.0,
        evidence_is_independent=False,
    )
    assert result["benchmark_passed"] is True
    assert result["development_qualified"] is True
    assert result["deployment_qualified"] is False
    assert result["deployment_checks"][0]["passed"] is False
    assert result["deployment_checks"][1]["passed"] is False


def test_config_requires_boundary_alignment_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    validate_admission_benchmark_config(config)
    manifest = json.loads(config.probability_manifest.read_text())
    manifest["source_benchmark_profile"] = "wrong_profile"
    config.probability_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="boundary_alignment"):
        validate_admission_benchmark_config(config)


def test_config_rejects_unresolved_boundary_run_id(tmp_path: Path) -> None:
    config = _config(tmp_path)
    unresolved = replace(
        config,
        probability_manifest=(
            tmp_path
            / "__BOUNDARY_RUN_ID__"
            / "saved-policy-probabilities"
            / "manifest.json"
        ),
    )
    with pytest.raises(ValueError, match="replace __BOUNDARY_RUN_ID__"):
        validate_admission_benchmark_config(unresolved)


def _base_probability_rows(
    *,
    markets: int,
    seconds: tuple[int, ...],
) -> pl.DataFrame:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    rows = []
    for market_index in range(markets):
        window_start = start + timedelta(minutes=5 * market_index)
        label_up = market_index % 2
        for second in seconds:
            probability_up = 0.65 if (market_index + second // 5) % 2 else 0.35
            predicted_up = int(probability_up >= 0.5)
            rows.append(
                {
                    "market_id": f"market-{market_index:03d}",
                    "window_start": window_start,
                    "observed_at": window_start + timedelta(seconds=second),
                    "seconds_elapsed": second,
                    "label_up": label_up,
                    "probability_up": probability_up,
                    "predicted_up": predicted_up,
                    "confidence": max(probability_up, 1.0 - probability_up),
                    "correct": predicted_up == label_up,
                    "candidate": ADMISSION_BASE_CANDIDATE,
                    "fold_index": 0,
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("label_up").cast(pl.Int32),
        pl.col("predicted_up").cast(pl.Int8),
        pl.col("fold_index").cast(pl.Int32),
    )


def _gates() -> AdmissionAdvancementGates:
    return AdmissionAdvancementGates(
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
        minimum_common_checkpoint_markets=500,
        minimum_executable_markets=500,
        minimum_mean_direct_edge_per_share=0.0,
        minimum_realized_net_per_share=0.0,
        require_every_fold=True,
        required_deployment_validation_folds=5,
    )


def _metrics(
    *,
    coverage: float,
    accuracy: float,
    balanced_accuracy: float,
    median: float,
) -> dict[str, object]:
    return {
        "markets": 600,
        "coverage": coverage,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "up_recall": 0.89,
        "down_recall": 0.89,
        "wilson_lower_95": 0.87,
        "expected_calibration_error": 0.03,
        "median_seconds_elapsed": median,
        "execution": {
            "economic_markets": 550,
            "mean_direct_edge_per_share": 0.01,
            "realized_net_expectancy_per_trade": 0.05,
        },
    }


def _config(tmp_path: Path) -> AdmissionBenchmarkConfig:
    manifest_path = tmp_path / "saved-policy-probabilities" / "manifest.json"
    manifest_path.parent.mkdir()
    candidate = {
        "folds": [{"fold_index": index} for index in range(5)],
    }
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SAVED_POLICY_PROBABILITY_SCHEMA_VERSION,
                "source_benchmark_profile": "boundary_alignment",
                "candidate_names": [
                    ADMISSION_CONTROL_CANDIDATE,
                    ADMISSION_BASE_CANDIDATE,
                ],
                "control_candidate": ADMISSION_CONTROL_CANDIDATE,
                "fold_count": 5,
                "candidates": {
                    ADMISSION_CONTROL_CANDIDATE: candidate,
                    ADMISSION_BASE_CANDIDATE: candidate,
                },
            }
        )
    )
    core_config = tmp_path / "core.toml"
    core_config.write_text("synthetic = true\n")
    execution = tmp_path / "execution"
    execution.mkdir()
    (execution / "manifest.json").write_text("{}")
    source_path = tmp_path / "admission.toml"
    source_path.write_text("synthetic = true\n")
    return AdmissionBenchmarkConfig(
        source_path=source_path,
        package_root=tmp_path,
        profile="correctness_admission",
        probability_manifest=manifest_path,
        core_config=core_config,
        control_candidate=ADMISSION_CONTROL_CANDIDATE,
        base_candidate=ADMISSION_BASE_CANDIDATE,
        selector_candidate=ADMISSION_SELECTOR_CANDIDATE,
        evaluation_folds=ADMISSION_EVALUATION_FOLDS,
        evaluation_note="synthetic consumed development evidence",
        evaluation_is_independent=False,
        quantity=5.0,
        selector=AdmissionSelectorConfig(
            regularization_c=1.0,
            minimum_calibration_rows_per_band=100,
            minimum_calibration_markets_per_band=25,
            admission_floor=0.5,
            random_seed=20260728,
        ),
        threshold_candidates=(0.55, 0.65, 0.75),
        bands=FROZEN_BANDS,
        gates=_gates(),
        execution_evidence=execution,
        runs=tmp_path / "runs",
    )
