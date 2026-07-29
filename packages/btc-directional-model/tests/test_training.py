from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model.config import (
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    PathConfig,
    SplitConfig,
    TrainingConfig,
)
from btc_directional_model.features import FEATURE_GROUPS
from btc_directional_model.inference import artifact_prediction, artifact_probability
from btc_directional_model.train import (
    chronological_split,
    fit_feature_group,
    fit_preprocessor,
    market_equal_weights,
    select_confidence_threshold,
    transform,
)


def training_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        source_path=tmp_path / "config.toml",
        package_root=tmp_path,
        data=DataConfig(
            range_start=datetime(2026, 4, 21, tzinfo=UTC),
            range_end=datetime(2026, 5, 21, tzinfo=UTC),
            sample_interval_seconds=5,
            min_seconds_after_open=60,
            min_seconds_before_close=60,
            strict_final_price_audit=False,
        ),
        split=SplitConfig(0.6, 0.2, 0.2),
        model=ModelConfig((0.1,), 0.5, 0.9, 0.01, 0.65, 0.6, 1, 1, 0.05, 1),
        evaluation=EvaluationConfig(True, "fresh chronological holdout"),
        paths=PathConfig(
            tmp_path / "source",
            tmp_path / "features.parquet",
            tmp_path / "runs",
            tmp_path / "artifacts",
        ),
    )


def market_frame() -> pl.DataFrame:
    start = datetime(2026, 4, 21, tzinfo=UTC)
    rows = []
    for market in range(10):
        for second in (60, 65):
            rows.append(
                {
                    "market_id": f"market-{market:02d}",
                    "window_start": start + timedelta(minutes=market * 5),
                    "seconds_elapsed": second,
                    "label_up": market % 2,
                }
            )
    return pl.DataFrame(rows)


def test_chronological_split_is_market_disjoint_and_ordered(tmp_path: Path) -> None:
    splits, summary = chronological_split(market_frame(), training_config(tmp_path))
    identifiers = {name: set(frame["market_id"]) for name, frame in splits.items()}

    assert [summary[name]["markets"] for name in ("train", "calibration", "test")] == [6, 2, 2]
    assert identifiers["train"].isdisjoint(identifiers["calibration"])
    assert identifiers["train"].isdisjoint(identifiers["test"])
    assert identifiers["calibration"].isdisjoint(identifiers["test"])
    assert summary["train"]["range_end"] < summary["calibration"]["range_start"]
    assert summary["calibration"]["range_end"] < summary["test"]["range_start"]


def test_market_equal_weights_give_each_market_equal_total_weight() -> None:
    frame = pl.DataFrame({"market_id": ["a", "a", "a", "b"]})
    weights = market_equal_weights(frame)

    assert weights[:3].sum() == pytest.approx(weights[3])
    assert weights.mean() == pytest.approx(1)


def test_preprocessor_is_fit_only_from_supplied_matrix() -> None:
    train = np.asarray([[1.0, np.nan], [3.0, 8.0]])
    weights = np.ones(2)
    future = np.asarray([[1_000.0, 1_000.0]])
    medians, means, scales = fit_preprocessor(train, weights)

    transformed = transform(future, medians, means, scales)

    assert medians.tolist() == [2.0, 8.0]
    assert means.tolist() == [2.0, 8.0]
    assert transformed[0, 0] > 900


def test_threshold_selection_is_accuracy_first_after_contract_filters(tmp_path: Path) -> None:
    base = training_config(tmp_path)
    config = replace(
        base,
        model=replace(base.model, minimum_calibration_markets=300),
    )
    thresholds = [
        {
            "threshold": 0.5,
            "markets": 1_000,
            "accuracy": 0.66,
            "wilson_lower_95": 0.63,
        },
        {
            "threshold": 0.8,
            "markets": 600,
            "accuracy": 0.80,
            "wilson_lower_95": 0.76,
        },
        {
            "threshold": 0.9,
            "markets": 200,
            "accuracy": 0.90,
            "wilson_lower_95": 0.85,
        },
    ]

    threshold, qualified = select_confidence_threshold(thresholds, config)

    assert qualified
    assert threshold == 0.8


def test_json_artifact_inference_matches_manual_calculation() -> None:
    artifact = {
        "feature_names": ["a", "b"],
        "imputation_medians": [2.0, 4.0],
        "standardization_means": [1.0, 2.0],
        "standardization_scales": [2.0, 4.0],
        "coefficients": [0.5, -1.0],
        "intercept": 0.25,
        "calibration_slope": 1.2,
        "calibration_intercept": -0.1,
        "confidence_threshold": 0.6,
    }
    standardized = np.asarray([(3.0 - 1.0) / 2.0, (4.0 - 2.0) / 4.0])
    raw_logit = float(standardized @ np.asarray([0.5, -1.0]) + 0.25)
    expected = 1 / (1 + np.exp(-(raw_logit * 1.2 - 0.1)))

    probability = artifact_probability(artifact, {"a": 3.0, "b": None})
    prediction = artifact_prediction(artifact, [3.0, None])

    assert probability == pytest.approx(expected, abs=1e-15)
    assert prediction["probability_up"] == pytest.approx(expected, abs=1e-15)
    with pytest.raises(ValueError, match="missing artifact features"):
        artifact_probability(artifact, {"a": 3.0})


def test_feature_group_fit_emits_reconstructable_golden_vectors(tmp_path: Path) -> None:
    start = datetime(2026, 4, 21, tzinfo=UTC)
    rows = []
    core = FEATURE_GROUPS["btc_path"]
    for market in range(30):
        for second in (60, 65, 70):
            row = {
                "market_id": f"market-{market:02d}",
                "window_start": start + timedelta(minutes=market * 5),
                "observed_at": start + timedelta(minutes=market * 5, seconds=second),
                "seconds_elapsed": second,
                "label_up": market % 2,
                "binance_sign_up": market % 2,
                "market_favorite_up": market % 2,
                "up_executable": True,
                "down_executable": True,
                "up_ask_vwap_5": 0.55,
                "down_ask_vwap_5": 0.45,
            }
            row.update(
                {
                    feature: market / 10 + second / 1_000 + index / 100
                    for index, feature in enumerate(core)
                }
            )
            rows.append(row)
    frame = pl.DataFrame(rows)
    config = training_config(tmp_path)
    splits, _ = chronological_split(frame, config)

    result, predictions = fit_feature_group("btc_path", core, splits, config)

    assert result["converged"]
    assert result["calibrator_converged"]
    assert predictions["market_id"].n_unique() > 0
    artifact = result["artifact"]
    assert artifact["optimizer_converged"]
    assert artifact["calibrator_converged"]
    for vector in artifact["golden_vectors"]:
        assert artifact_probability(artifact, vector["feature_values"]) == pytest.approx(
            vector["expected_probability_up"], abs=1e-12
        )
