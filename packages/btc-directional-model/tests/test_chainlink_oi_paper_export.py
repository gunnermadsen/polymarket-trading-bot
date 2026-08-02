from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from btc_directional_model.chainlink_oi_benchmark import (
    ChainlinkOiCandidateBundle,
    _candidate_feature_sets,
)
from btc_directional_model.chainlink_oi_config import (
    CHAINLINK_FULL_CANDIDATE,
    CHAINLINK_FULL_OI_CANDIDATE,
    LONG_HISTORY_CANDLE_CANDIDATE,
)
from btc_directional_model.chainlink_oi_paper_export import (
    PAPER_EXPORT_SPECS,
    export_chainlink_oi_paper_candidates,
    time_banded_runtime_bundle,
)
from btc_directional_model.core_evaluation import FIRST_CROSSING_TIME_BANDS
from btc_directional_model.core_training import (
    FittedCoreModel,
    ProbabilityCalibrator,
)
from btc_directional_model.runtime_export import (
    TIME_BANDED_RUNTIME_MODEL_SCHEMA_VERSION,
    runtime_model_payload,
)


def test_export_specs_freeze_exact_candidate_schema_contracts() -> None:
    feature_sets = _candidate_feature_sets()
    specs = {spec.candidate: spec for spec in PAPER_EXPORT_SPECS}

    assert tuple(specs) == (
        CHAINLINK_FULL_CANDIDATE,
        CHAINLINK_FULL_OI_CANDIDATE,
        LONG_HISTORY_CANDLE_CANDIDATE,
    )
    assert {
        candidate: len(feature_sets[candidate]) for candidate in specs
    } == {
        CHAINLINK_FULL_CANDIDATE: 97,
        CHAINLINK_FULL_OI_CANDIDATE: 106,
        LONG_HISTORY_CANDLE_CANDIDATE: 87,
    }
    assert specs[LONG_HISTORY_CANDLE_CANDIDATE].feature_schema_version == (
        "btc-5m-directional-boundary-oracle-chainlink-candle-features-v1"
    )
    assert specs[CHAINLINK_FULL_CANDIDATE].feature_schema_version == (
        "btc-5m-directional-boundary-oracle-chainlink-refprice-candle-features-v1"
    )
    assert specs[CHAINLINK_FULL_OI_CANDIDATE].feature_schema_version == (
        "btc-5m-directional-boundary-oracle-chainlink-refprice-candle-oi-features-v1"
    )
    assert len({spec.model_key for spec in PAPER_EXPORT_SPECS}) == 3
    assert all(spec.model_key.endswith("-paper-v1") for spec in PAPER_EXPORT_SPECS)


def test_saved_bundle_converts_without_refitting_and_preserves_calibration() -> None:
    model = _fitted_model()
    calibrators = _calibrators()
    source = ChainlinkOiCandidateBundle(
        name=CHAINLINK_FULL_CANDIDATE,
        feature_names=model.feature_names,
        model=model,
        calibrators=calibrators,
        confidence_threshold=0.89,
    )

    converted = time_banded_runtime_bundle(source)

    assert converted.model is model
    assert converted.target_kind == "outcome_up"
    assert tuple(band.name for band in converted.bands) == tuple(calibrators)
    assert all(band.calibrator is calibrators[band.name] for band in converted.bands)
    assert all(band.confidence_threshold == 0.89 for band in converted.bands)


def test_converted_bundle_uses_existing_verified_runtime_v2_contract() -> None:
    model = _fitted_model()
    source = ChainlinkOiCandidateBundle(
        name=CHAINLINK_FULL_CANDIDATE,
        feature_names=model.feature_names,
        model=model,
        calibrators=_calibrators(),
        confidence_threshold=0.89,
    )
    bundle = time_banded_runtime_bundle(source)
    freeze = {
        "feature_schema_version": PAPER_EXPORT_SPECS[0].feature_schema_version,
        "prediction_policy": {
            "type": "first_confidence_crossing",
            "minimum_seconds_after_open": 60,
            "maximum_seconds_after_open": 240,
            "cadence_seconds": 5,
        },
        "freeze_id": "frozen-chainlink-full-paper",
        "created_at": "2026-08-02T00:18:17+00:00",
        "model_sha256": "1" * 64,
        "model_summary_sha256": "2" * 64,
        "configuration_sha256": "3" * 64,
        "development_feature_sha256": "4" * 64,
        "development_feature_metadata_sha256": "5" * 64,
        "source_tree_sha256": "6" * 64,
        "git": {"commit": "7" * 40, "branch": "feature", "dirty": False},
        "holdout_range": {
            "start": "2026-07-21T00:00:00+00:00",
            "end": "2026-07-29T00:00:00+00:00",
        },
        "random_seed": 20260801,
        "deployment_scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
    }

    payload = runtime_model_payload(
        bundle=bundle,
        freeze=freeze,
        freeze_sha256="8" * 64,
        model_key=PAPER_EXPORT_SPECS[0].model_key,
    )

    assert payload["schema_version"] == TIME_BANDED_RUNTIME_MODEL_SCHEMA_VERSION
    assert payload["target"] == {"type": "outcome_up"}
    assert [band["name"] for band in payload["time_bands"]] == [
        "60-89",
        "90-119",
        "120-179",
        "180-240",
    ]
    assert {band["confidence_threshold"] for band in payload["time_bands"]} == {
        0.89
    }
    assert payload["deployment"] == {
        "scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
    }


def test_conversion_rejects_reordered_or_invalid_calibration() -> None:
    model = _fitted_model()
    reordered = dict(reversed(tuple(_calibrators().items())))
    source = ChainlinkOiCandidateBundle(
        name=CHAINLINK_FULL_CANDIDATE,
        feature_names=model.feature_names,
        model=model,
        calibrators=reordered,
        confidence_threshold=0.89,
    )
    with pytest.raises(RuntimeError, match="missing or reordered"):
        time_banded_runtime_bundle(source)

    invalid = _calibrators()
    invalid["60-89"] = ProbabilityCalibrator(
        slope=-1.0,
        intercept=0.0,
        converged=True,
        iterations=1,
    )
    source.calibrators = invalid
    with pytest.raises(RuntimeError, match="calibration band is invalid"):
        time_banded_runtime_bundle(source)


def test_export_fails_closed_without_exact_paper_authorization(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="explicit paper-only authorization"):
        export_chainlink_oi_paper_candidates(
            config=SimpleNamespace(),
            benchmark_run=tmp_path,
            freeze_root=tmp_path / "freezes",
            runtime_output_root=tmp_path / "runtime-models",
            authorization="paper",
        )
    assert not (tmp_path / "freezes").exists()
    assert not (tmp_path / "runtime-models").exists()


def _calibrators() -> dict[str, ProbabilityCalibrator]:
    return {
        name: ProbabilityCalibrator(
            slope=1.0 + index / 100.0,
            intercept=index / 1000.0,
            converged=True,
            iterations=5,
        )
        for index, (name, _, _) in enumerate(FIRST_CROSSING_TIME_BANDS)
    }


def _fitted_model() -> FittedCoreModel:
    matrix = np.asarray(
        [
            [-2.0, -1.0],
            [-1.0, -0.5],
            [-0.5, -1.5],
            [0.5, 1.5],
            [1.0, 0.5],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=3,
        max_leaf_nodes=3,
        min_samples_leaf=2,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=20260801,
    ).fit(matrix, labels)
    return FittedCoreModel(
        candidate_name=CHAINLINK_FULL_CANDIDATE,
        family="histogram",
        feature_names=("signal", "context"),
        hyperparameters={
            "learning_rate": 0.1,
            "max_iter": 3,
            "max_leaf_nodes": 3,
            "min_samples_leaf": 2,
            "l2_regularization": 0.1,
        },
        imputation_medians=np.asarray([0.0, 0.0]),
        standardization_means=None,
        standardization_scales=None,
        estimator=estimator,
    )
