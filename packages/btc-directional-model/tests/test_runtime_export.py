from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from btc_directional_model.core_extract import file_sha256, write_json_atomic
from btc_directional_model.core_training import (
    CORE_FREEZE_SCHEMA_VERSION,
    TRAINING_MODEL_FILENAME,
    FittedCoreModel,
    FrozenTrainingBundle,
    ProbabilityCalibrator,
    model_summary_payload,
)
from btc_directional_model.runtime_export import (
    GOLDEN_VECTORS_FILENAME,
    GOLDEN_VECTORS_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    RUNTIME_MANIFEST_SCHEMA_VERSION,
    RUNTIME_MODEL_SCHEMA_VERSION,
    export_runtime_model,
    score_runtime_model,
)

MODEL_KEY = "btc-test-histogram-v1"


def make_frozen_candidate(tmp_path: Path) -> tuple[Path, Path, FrozenTrainingBundle]:
    feature_names = ("a", "b")
    a = np.linspace(-4.0, 4.0, 201)
    b = np.sin(a)
    matrix = np.column_stack([a, b])
    probability = 1.0 / (1.0 + np.exp(-a))
    labels = (
        np.random.default_rng(7).random(len(a)) < probability
    ).astype(np.int32)
    parameters = {
        "learning_rate": 0.2,
        "max_iter": 30,
        "max_leaf_nodes": 3,
        "min_samples_leaf": 2,
        "l2_regularization": 0.1,
    }
    estimator = HistGradientBoostingClassifier(
        **parameters,
        early_stopping=False,
        random_state=7,
    ).fit(matrix, labels)
    fitted = FittedCoreModel(
        candidate_name="histogram_enriched",
        family="histogram",
        feature_names=feature_names,
        hyperparameters=parameters,
        imputation_medians=np.asarray([0.0, 0.0]),
        standardization_means=None,
        standardization_scales=None,
        estimator=estimator,
    )
    calibrator = ProbabilityCalibrator(
        slope=1.0,
        intercept=0.0,
        converged=True,
        iterations=1,
    )
    bundle = FrozenTrainingBundle(
        model=fitted,
        calibrator=calibrator,
        confidence_threshold=0.75,
    )

    freeze_dir = tmp_path / "freeze"
    freeze_dir.mkdir()
    model_path = freeze_dir / TRAINING_MODEL_FILENAME
    joblib.dump(bundle, model_path, compress=3)
    summary_path = freeze_dir / "model-summary.json"
    write_json_atomic(summary_path, model_summary_payload(bundle))
    freeze = {
        "schema_version": CORE_FREEZE_SCHEMA_VERSION,
        "freeze_id": "test-freeze",
        "created_at": "2026-07-26T23:54:22+00:00",
        "status": "candidate_frozen",
        "deployment_status": "blocked_pending_execution_economics",
        "model_file": TRAINING_MODEL_FILENAME,
        "model_sha256": file_sha256(model_path),
        "model_summary_sha256": file_sha256(summary_path),
        "candidate": fitted.candidate_name,
        "family": fitted.family,
        "feature_schema_version": "test-features-v1",
        "feature_names": list(feature_names),
        "hyperparameters": parameters,
        "calibrator": asdict(calibrator),
        "confidence_threshold": bundle.confidence_threshold,
        "prediction_policy": {
            "type": "first_confidence_crossing",
            "minimum_seconds_after_open": 60,
            "maximum_seconds_after_open": 240,
            "cadence_seconds": 5,
        },
        "configuration_sha256": "1" * 64,
        "development_feature_sha256": "2" * 64,
        "development_feature_metadata_sha256": "3" * 64,
        "source_tree_sha256": "4" * 64,
        "git": {"branch": "test", "commit": "5" * 40, "dirty": False},
        "random_seed": 7,
        "holdout_range": {
            "start": "2026-06-14T00:00:00+00:00",
            "end": "2026-06-21T00:00:00+00:00",
        },
    }
    manifest_path = freeze_dir / "freeze-manifest.json"
    write_json_atomic(manifest_path, freeze)
    (freeze_dir / "freeze-manifest.sha256").write_text(
        file_sha256(manifest_path) + "\n"
    )

    start = datetime(2026, 6, 14, tzinfo=UTC)
    feature_path = tmp_path / "holdout.parquet"
    feature_frame = pl.DataFrame(
        {
            "market_id": [f"market-{index:03d}" for index in range(len(a))],
            "observed_at": [
                start + timedelta(seconds=index * 5) for index in range(len(a))
            ],
            "a": a,
            "b": b,
        }
    )
    feature_frame.write_parquet(feature_path)
    write_json_atomic(
        feature_path.with_suffix(".metadata.json"),
        {
            "feature_schema_version": "test-features-v1",
            "feature_file_sha256": file_sha256(feature_path),
        },
    )
    return freeze_dir, feature_path, bundle


def test_runtime_export_is_deterministic_and_reconstructable(tmp_path: Path) -> None:
    freeze_dir, feature_path, bundle = make_frozen_candidate(tmp_path)
    output_root = tmp_path / "runtime-models"

    destination = export_runtime_model(
        freeze_dir=freeze_dir,
        golden_features=feature_path,
        output_root=output_root,
        model_key=MODEL_KEY,
    )
    original = {
        name: (destination / name).read_bytes()
        for name in (MODEL_FILENAME, MANIFEST_FILENAME, GOLDEN_VECTORS_FILENAME)
    }
    assert destination.stat().st_mode & 0o777 == 0o755
    assert {
        (destination / name).stat().st_mode & 0o777 for name in original
    } == {0o644}
    destination.chmod(0o700)
    for name in original:
        (destination / name).chmod(0o600)
    repeated = export_runtime_model(
        freeze_dir=freeze_dir,
        golden_features=feature_path,
        output_root=output_root,
        model_key=MODEL_KEY,
    )

    assert repeated == destination
    assert {
        name: (destination / name).read_bytes() for name in original
    } == original
    assert destination.stat().st_mode & 0o777 == 0o755
    assert {
        (destination / name).stat().st_mode & 0o777 for name in original
    } == {0o644}
    model = json.loads((destination / MODEL_FILENAME).read_text())
    manifest = json.loads((destination / MANIFEST_FILENAME).read_text())
    golden = json.loads((destination / GOLDEN_VECTORS_FILENAME).read_text())
    assert model["schema_version"] == RUNTIME_MODEL_SCHEMA_VERSION
    assert manifest["schema_version"] == RUNTIME_MANIFEST_SCHEMA_VERSION
    assert golden["schema_version"] == GOLDEN_VECTORS_SCHEMA_VERSION
    assert manifest["model_sha256"] == file_sha256(destination / MODEL_FILENAME)
    assert manifest["golden_vectors_sha256"] == file_sha256(
        destination / GOLDEN_VECTORS_FILENAME
    )
    assert {vector["expected"]["action"] for vector in golden["vectors"]} >= {
        "up",
        "down",
        "no_trade",
    }
    for vector in golden["vectors"]:
        actual = score_runtime_model(model, vector["feature_values"])
        assert actual["raw_logit"] == vector["expected"]["raw_logit"]
        assert actual["probability_up"] == vector["expected"]["probability_up"]
        assert actual["confidence"] == vector["expected"]["confidence"]
        assert actual["action"] == vector["expected"]["action"]

    frame = pl.DataFrame({"a": [2.5], "b": [float(np.sin(2.5))]})
    expected = float(bundle.probability(frame)[0])
    actual = score_runtime_model(model, [2.5, float(np.sin(2.5))])
    assert actual["probability_up"] == pytest.approx(expected, abs=1e-15)


def test_runtime_export_rejects_freeze_manifest_hash_mismatch(tmp_path: Path) -> None:
    freeze_dir, feature_path, _ = make_frozen_candidate(tmp_path)
    (freeze_dir / "freeze-manifest.sha256").write_text("0" * 64 + "\n")

    with pytest.raises(RuntimeError, match="freeze manifest hash mismatch"):
        export_runtime_model(
            freeze_dir=freeze_dir,
            golden_features=feature_path,
            output_root=tmp_path / "runtime-models",
            model_key=MODEL_KEY,
        )


def test_runtime_export_rejects_model_key_with_trailing_dash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model key"):
        export_runtime_model(
            freeze_dir=tmp_path / "missing-freeze",
            golden_features=tmp_path / "missing-features.parquet",
            output_root=tmp_path / "runtime-models",
            model_key="btc-invalid-",
        )


def test_checked_in_runtime_model_is_self_consistent() -> None:
    package_root = Path(__file__).parent.parent
    model_dir = (
        package_root
        / "runtime-models"
        / "btc-5m-directional-histogram-enriched-20260421-20260620-v1"
    )
    model = json.loads((model_dir / MODEL_FILENAME).read_text())
    manifest = json.loads((model_dir / MANIFEST_FILENAME).read_text())
    golden = json.loads((model_dir / GOLDEN_VECTORS_FILENAME).read_text())

    assert manifest["model_sha256"] == file_sha256(model_dir / MODEL_FILENAME)
    assert manifest["golden_vectors_sha256"] == file_sha256(
        model_dir / GOLDEN_VECTORS_FILENAME
    )
    assert len(model["features"]["names"]) == 58
    assert len(model["estimator"]["trees"]) == 160
    assert model["decision"]["below_confidence_action"] == "no_trade"
    assert model["prediction_policy"] == {
        "type": "first_confidence_crossing",
        "minimum_seconds_after_open": 60,
        "maximum_seconds_after_open": 240,
        "cadence_seconds": 5,
    }
    for vector in golden["vectors"]:
        assert score_runtime_model(model, vector["feature_values"]) == vector["expected"]
