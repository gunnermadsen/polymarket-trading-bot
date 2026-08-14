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
    FrozenCalibrationBand,
    FrozenTimeBandedTrainingBundle,
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
    TIME_BANDED_GOLDEN_VECTORS_SCHEMA_VERSION,
    TIME_BANDED_RUNTIME_MODEL_SCHEMA_VERSION,
    export_runtime_model,
    promote_runtime_model_for_live_pilot,
    score_runtime_model,
    validate_bundle_against_freeze,
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
        "deployment_scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
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


def make_time_banded_candidate(
    tmp_path: Path,
    *,
    fixed_120: bool = False,
) -> tuple[Path, Path, FrozenTimeBandedTrainingBundle]:
    feature_names = ("btc_path_from_window_open_bps", "signal")
    per_band = 100
    seconds_to_materialize = [120, 125] if fixed_120 else [60, 90, 120, 180]
    seconds = np.repeat(seconds_to_materialize, per_band)
    signal = np.tile(
        np.linspace(-4.0, 4.0, per_band),
        len(seconds_to_materialize),
    )
    path = np.where(np.arange(len(signal)) % 2 == 0, 8.0, -8.0)
    training_matrix = np.column_stack([np.full(len(signal), 8.0), signal])
    probability = 1.0 / (1.0 + np.exp(-signal))
    labels = (
        np.random.default_rng(17).random(len(signal)) < probability
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
        random_state=17,
    ).fit(training_matrix, labels)
    fitted = FittedCoreModel(
        candidate_name=(
            "histogram_path_persistence_exact_120"
            if fixed_120
            else "histogram_path_persistence_time_calibrated_60_120"
        ),
        family="histogram",
        feature_names=feature_names,
        hyperparameters=parameters,
        imputation_medians=np.asarray([8.0, 0.0]),
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
    ranges = (
        (("120-120", 120, 121, 0.77),)
        if fixed_120
        else (
            ("60-89", 60, 90, 0.75),
            ("90-119", 90, 120, 0.76),
            ("120-179", 120, 180, 0.77),
            ("180-240", 180, 241, 0.78),
        )
    )
    bundle = FrozenTimeBandedTrainingBundle(
        model=fitted,
        target_kind="path_persistence",
        bands=tuple(
            FrozenCalibrationBand(
                name=name,
                start_second=start,
                end_second_exclusive=end,
                calibrator=calibrator,
                confidence_threshold=threshold,
            )
            for name, start, end, threshold in ranges
        ),
    )
    calibration_bands = [
        {
            "name": band.name,
            "start_second": band.start_second,
            "end_second_exclusive": band.end_second_exclusive,
            "calibrator": asdict(band.calibrator),
            "confidence_threshold": band.confidence_threshold,
        }
        for band in bundle.bands
    ]
    freeze_dir = tmp_path / ("fixed-time-freeze" if fixed_120 else "time-freeze")
    freeze_dir.mkdir()
    model_path = freeze_dir / TRAINING_MODEL_FILENAME
    joblib.dump(bundle, model_path, compress=3)
    summary_path = freeze_dir / "model-summary.json"
    write_json_atomic(
        summary_path,
        {
            "schema_version": "btc-core-training-model-summary-v1",
            "training_only": True,
            "candidate": fitted.candidate_name,
            "family": fitted.family,
            "feature_names": list(feature_names),
            "hyperparameters": parameters,
            "imputation_medians": fitted.imputation_medians.tolist(),
            "calibration_kind": "time_banded_platt",
            "calibration_bands": calibration_bands,
            "target_kind": "path_persistence",
        },
    )
    freeze = {
        "schema_version": CORE_FREEZE_SCHEMA_VERSION,
        "freeze_id": "test-time-freeze",
        "created_at": "2026-07-29T00:00:00+00:00",
        "deployment_scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
        "model_file": TRAINING_MODEL_FILENAME,
        "model_sha256": file_sha256(model_path),
        "model_summary_sha256": file_sha256(summary_path),
        "candidate": fitted.candidate_name,
        "family": fitted.family,
        "feature_schema_version": "test-path-features-v1",
        "feature_names": list(feature_names),
        "hyperparameters": parameters,
        "calibration_kind": "time_banded_platt",
        "calibration_bands": calibration_bands,
        "target_kind": "path_persistence",
        "prediction_policy": {
            "type": "first_confidence_crossing",
            "minimum_seconds_after_open": 120 if fixed_120 else 60,
            "maximum_seconds_after_open": 120 if fixed_120 else 240,
            "cadence_seconds": 5,
        },
        "configuration_sha256": "1" * 64,
        "development_feature_sha256": "2" * 64,
        "development_feature_metadata_sha256": "3" * 64,
        "source_tree_sha256": "4" * 64,
        "git": {"branch": "test", "commit": "5" * 40, "dirty": False},
        "random_seed": 17,
        "holdout_range": {
            "start": "2026-07-20T00:00:00+00:00",
            "end": "2026-07-20T00:00:00+00:00",
        },
    }
    manifest_path = freeze_dir / "freeze-manifest.json"
    write_json_atomic(manifest_path, freeze)
    (freeze_dir / "freeze-manifest.sha256").write_text(
        file_sha256(manifest_path) + "\n"
    )
    start = datetime(2026, 7, 13, tzinfo=UTC)
    feature_path = tmp_path / (
        "fixed-time-golden.parquet" if fixed_120 else "time-golden.parquet"
    )
    pl.DataFrame(
        {
            "market_id": [
                f"time-market-{index:03d}" for index in range(len(signal))
            ],
            "observed_at": [
                start + timedelta(seconds=index * 5)
                for index in range(len(signal))
            ],
            "seconds_elapsed": seconds,
            "btc_path_from_window_open_bps": path,
            "signal": signal,
        }
    ).write_parquet(feature_path)
    write_json_atomic(
        feature_path.with_suffix(".metadata.json"),
        {
            "feature_schema_version": "test-path-features-v1",
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
    assert model["deployment"] == {
        "scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
    }
    assert manifest["deployment_scope"] == "paper_only"
    assert manifest["production_qualified"] is False
    assert manifest["live_capital_allowed"] is False
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


def test_time_banded_runtime_export_preserves_target_and_policy(
    tmp_path: Path,
) -> None:
    freeze_dir, feature_path, _ = make_time_banded_candidate(tmp_path)
    destination = export_runtime_model(
        freeze_dir=freeze_dir,
        golden_features=feature_path,
        output_root=tmp_path / "runtime-models",
        model_key="btc-test-time-banded-v2",
    )
    model = json.loads((destination / MODEL_FILENAME).read_text())
    golden = json.loads((destination / GOLDEN_VECTORS_FILENAME).read_text())

    assert model["schema_version"] == TIME_BANDED_RUNTIME_MODEL_SCHEMA_VERSION
    assert model["target"] == {
        "type": "path_persistence",
        "path_direction_feature": "btc_path_from_window_open_bps",
        "zero_path_epsilon_bps": 1e-12,
        "ineligible_action": "no_trade",
    }
    assert [band["confidence_threshold"] for band in model["time_bands"]] == [
        0.75,
        0.76,
        0.77,
        0.78,
    ]
    assert "confidence_threshold" not in model["decision"]
    assert "calibration" not in model
    assert (
        golden["schema_version"]
        == TIME_BANDED_GOLDEN_VECTORS_SCHEMA_VERSION
    )
    assert {
        vector["seconds_elapsed"] // 30 for vector in golden["vectors"][:-1]
    } >= {2, 3, 4, 6}
    for vector in golden["vectors"]:
        assert set(vector["expected"]) == {
            "raw_logit",
            "probability_up",
            "confidence",
            "action",
        }
        assert (
            score_runtime_model(
                model,
                vector["feature_values"],
                seconds_elapsed=vector["seconds_elapsed"],
            )
            == vector["expected"]
        )

    positive = score_runtime_model(
        model,
        [8.0, 3.0],
        seconds_elapsed=60,
    )
    negative = score_runtime_model(
        model,
        [-8.0, 3.0],
        seconds_elapsed=60,
    )
    assert negative["probability_up"] == pytest.approx(
        1.0 - positive["probability_up"],
        abs=1e-15,
    )
    ineligible = score_runtime_model(
        model,
        [0.0, 3.0],
        seconds_elapsed=60,
    )
    assert ineligible == {
        "raw_logit": ineligible["raw_logit"],
        "probability_up": 0.5,
        "confidence": 0.5,
        "action": "no_trade",
    }
    zero_path_vector = next(
        vector for vector in golden["vectors"] if vector["id"] == "zero_path_ineligible"
    )
    assert zero_path_vector["expected"]["probability_up"] == 0.5
    assert zero_path_vector["expected"]["confidence"] == 0.5
    assert zero_path_vector["expected"]["action"] == "no_trade"


def test_exact_120_time_banded_runtime_export_is_reconstructable(
    tmp_path: Path,
) -> None:
    freeze_dir, feature_path, bundle = make_time_banded_candidate(
        tmp_path,
        fixed_120=True,
    )
    destination = export_runtime_model(
        freeze_dir=freeze_dir,
        golden_features=feature_path,
        output_root=tmp_path / "runtime-models",
        model_key="btc-test-path-persistence-exact-120-v2",
    )
    model = json.loads((destination / MODEL_FILENAME).read_text())
    golden = json.loads((destination / GOLDEN_VECTORS_FILENAME).read_text())

    assert model["prediction_policy"] == {
        "type": "first_confidence_crossing",
        "minimum_seconds_after_open": 120,
        "maximum_seconds_after_open": 120,
        "cadence_seconds": 5,
    }
    assert model["time_bands"] == [
        {
            "name": "120-120",
            "start_seconds": 120,
            "end_seconds_exclusive": 121,
            "calibration": model["time_bands"][0]["calibration"],
            "confidence_threshold": 0.77,
        }
    ]
    assert {vector["seconds_elapsed"] for vector in golden["vectors"]} == {120}
    assert {vector["id"] for vector in golden["vectors"]} >= {
        "120-120-minimum",
        "120-120-maximum",
        "zero_path_ineligible",
        "all_features_non_finite",
    }
    for vector in golden["vectors"]:
        assert (
            score_runtime_model(
                model,
                vector["feature_values"],
                seconds_elapsed=vector["seconds_elapsed"],
            )
            == vector["expected"]
        )

    exact_row = pl.DataFrame(
        {
            "seconds_elapsed": [120],
            "btc_path_from_window_open_bps": [8.0],
            "signal": [3.0],
        }
    )
    expected_probability = float(bundle.probability_up(exact_row)[0])
    actual = score_runtime_model(model, [8.0, 3.0], seconds_elapsed=120)
    assert actual["probability_up"] == pytest.approx(
        expected_probability,
        abs=1e-15,
    )
    zero_path = score_runtime_model(model, [0.0, 3.0], seconds_elapsed=120)
    assert zero_path["probability_up"] == 0.5
    assert zero_path["confidence"] == 0.5
    assert zero_path["action"] == "no_trade"
    with pytest.raises(ValueError, match="outside the runtime time bands"):
        score_runtime_model(model, [8.0, 3.0], seconds_elapsed=125)


def test_time_banded_runtime_export_rejects_incomplete_rolling_bands(
    tmp_path: Path,
) -> None:
    freeze_dir, _, bundle = make_time_banded_candidate(tmp_path)
    freeze = json.loads((freeze_dir / "freeze-manifest.json").read_text())
    summary = json.loads((freeze_dir / "model-summary.json").read_text())
    incomplete_bundle = FrozenTimeBandedTrainingBundle(
        model=bundle.model,
        target_kind=bundle.target_kind,
        bands=bundle.bands[:-1],
    )
    incomplete_bands = [
        {
            "name": band.name,
            "start_second": band.start_second,
            "end_second_exclusive": band.end_second_exclusive,
            "calibrator": asdict(band.calibrator),
            "confidence_threshold": band.confidence_threshold,
        }
        for band in incomplete_bundle.bands
    ]
    freeze["calibration_bands"] = incomplete_bands
    summary["calibration_bands"] = incomplete_bands

    with pytest.raises(RuntimeError, match="exactly cover the 60-240"):
        validate_bundle_against_freeze(incomplete_bundle, freeze, summary)


def test_exact_120_policy_rejects_legacy_four_band_bundle(tmp_path: Path) -> None:
    freeze_dir, _, bundle = make_time_banded_candidate(tmp_path)
    freeze = json.loads((freeze_dir / "freeze-manifest.json").read_text())
    summary = json.loads((freeze_dir / "model-summary.json").read_text())
    freeze["prediction_policy"] = {
        "type": "first_confidence_crossing",
        "minimum_seconds_after_open": 120,
        "maximum_seconds_after_open": 120,
        "cadence_seconds": 5,
    }

    with pytest.raises(RuntimeError, match="exactly one 120-121 band"):
        validate_bundle_against_freeze(bundle, freeze, summary)


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


def test_runtime_export_accepts_exact_fixed_120_policy(tmp_path: Path) -> None:
    freeze_dir, feature_path, _ = make_frozen_candidate(tmp_path)
    manifest_path = freeze_dir / "freeze-manifest.json"
    freeze = json.loads(manifest_path.read_text())
    freeze["prediction_policy"] = {
        "type": "first_confidence_crossing",
        "minimum_seconds_after_open": 120,
        "maximum_seconds_after_open": 120,
        "cadence_seconds": 7,
    }
    write_json_atomic(manifest_path, freeze)
    (freeze_dir / "freeze-manifest.sha256").write_text(
        file_sha256(manifest_path) + "\n"
    )

    destination = export_runtime_model(
        freeze_dir=freeze_dir,
        golden_features=feature_path,
        output_root=tmp_path / "runtime-models",
        model_key=MODEL_KEY,
    )
    model = json.loads((destination / MODEL_FILENAME).read_text())

    assert model["prediction_policy"] == freeze["prediction_policy"]


@pytest.mark.parametrize(
    "prediction_policy",
    (
        {
            "type": "first_confidence_crossing",
            "minimum_seconds_after_open": 120,
            "maximum_seconds_after_open": 125,
            "cadence_seconds": 5,
        },
        {
            "type": "first_confidence_crossing",
            "minimum_seconds_after_open": 125,
            "maximum_seconds_after_open": 125,
            "cadence_seconds": 5,
        },
        {
            "type": "first_confidence_crossing",
            "minimum_seconds_after_open": 120,
            "maximum_seconds_after_open": 120,
            "cadence_seconds": 0,
        },
        {
            "type": "first_confidence_crossing",
            "minimum_seconds_after_open": 120,
            "maximum_seconds_after_open": 120,
            "cadence_seconds": 5.0,
        },
    ),
)
def test_runtime_export_rejects_other_prediction_policies(
    tmp_path: Path,
    prediction_policy: dict[str, object],
) -> None:
    freeze_dir, feature_path, _ = make_frozen_candidate(tmp_path)
    manifest_path = freeze_dir / "freeze-manifest.json"
    freeze = json.loads(manifest_path.read_text())
    freeze["prediction_policy"] = prediction_policy
    write_json_atomic(manifest_path, freeze)
    (freeze_dir / "freeze-manifest.sha256").write_text(
        file_sha256(manifest_path) + "\n"
    )

    with pytest.raises(RuntimeError, match="exact fixed 120-second policy"):
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "production_qualified",
            "false",
            "qualification fields must be booleans",
        ),
        (
            "live_capital_allowed",
            True,
            "paper-only frozen models cannot",
        ),
    ),
)
def test_runtime_export_rejects_incoherent_paper_deployment_metadata(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    freeze_dir, feature_path, _ = make_frozen_candidate(tmp_path)
    manifest_path = freeze_dir / "freeze-manifest.json"
    freeze = json.loads(manifest_path.read_text())
    freeze[field] = value
    write_json_atomic(manifest_path, freeze)
    (freeze_dir / "freeze-manifest.sha256").write_text(
        file_sha256(manifest_path) + "\n"
    )

    with pytest.raises((RuntimeError, TypeError), match=message):
        export_runtime_model(
            freeze_dir=freeze_dir,
            golden_features=feature_path,
            output_root=tmp_path / "runtime-models",
            model_key=MODEL_KEY,
        )


def test_runtime_model_live_pilot_promotion_changes_only_identity_and_authorization(
    tmp_path: Path,
) -> None:
    freeze_dir, feature_path, _ = make_frozen_candidate(tmp_path)
    output_root = tmp_path / "runtime-models"
    paper_dir = export_runtime_model(
        freeze_dir=freeze_dir,
        golden_features=feature_path,
        output_root=output_root,
        model_key=MODEL_KEY,
    )
    live_key = "btc-test-histogram-development-live-pilot-v1"
    live_dir = promote_runtime_model_for_live_pilot(
        source_runtime=paper_dir,
        output_root=output_root,
        model_key=live_key,
    )

    paper_model = json.loads((paper_dir / MODEL_FILENAME).read_text())
    live_model = json.loads((live_dir / MODEL_FILENAME).read_text())
    paper_golden = json.loads((paper_dir / GOLDEN_VECTORS_FILENAME).read_text())
    live_golden = json.loads((live_dir / GOLDEN_VECTORS_FILENAME).read_text())
    live_manifest = json.loads((live_dir / MANIFEST_FILENAME).read_text())

    assert paper_model.pop("model_key") == MODEL_KEY
    assert live_model.pop("model_key") == live_key
    assert paper_model.pop("deployment") == {
        "scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
    }
    assert live_model.pop("deployment") == {
        "scope": "development_live_pilot",
        "production_qualified": False,
        "live_capital_allowed": True,
    }
    assert live_model == paper_model
    assert paper_golden["vectors"] == live_golden["vectors"]
    assert live_manifest["model_key"] == live_key
    assert live_manifest["model_sha256"] == file_sha256(live_dir / MODEL_FILENAME)
    assert live_manifest["golden_vectors_sha256"] == file_sha256(
        live_dir / GOLDEN_VECTORS_FILENAME
    )
    assert live_manifest["deployment_scope"] == "development_live_pilot"
    assert live_manifest["production_qualified"] is False
    assert live_manifest["live_capital_allowed"] is True

    (live_dir / MODEL_FILENAME).write_text("different")
    with pytest.raises(RuntimeError, match="immutable model key"):
        promote_runtime_model_for_live_pilot(
            source_runtime=paper_dir,
            output_root=output_root,
            model_key=live_key,
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


def test_checked_in_asymmetric_oracle_live_pilot_preserves_paper_model_behavior() -> None:
    package_root = Path(__file__).parent.parent
    paper_dir = (
        package_root
        / "runtime-models"
        / "btc-5m-asymmetric-core-oracle-paper-20260805-v1"
    )
    live_dir = (
        package_root
        / "runtime-models"
        / "btc-5m-asymmetric-core-oracle-live-pilot-20260814"
    )
    paper_model = json.loads((paper_dir / MODEL_FILENAME).read_text())
    live_model = json.loads((live_dir / MODEL_FILENAME).read_text())
    paper_golden = json.loads((paper_dir / GOLDEN_VECTORS_FILENAME).read_text())
    live_golden = json.loads((live_dir / GOLDEN_VECTORS_FILENAME).read_text())
    live_manifest = json.loads((live_dir / MANIFEST_FILENAME).read_text())

    assert paper_model.pop("model_key") == "btc-5m-asymmetric-core-oracle-paper-20260805-v1"
    assert live_model.pop("model_key") == "btc-5m-asymmetric-core-oracle-live-pilot-20260814"
    assert paper_model.pop("deployment") == {
        "scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
    }
    assert live_model.pop("deployment") == {
        "scope": "development_live_pilot",
        "production_qualified": False,
        "live_capital_allowed": True,
    }
    assert live_model == paper_model
    assert paper_golden["vectors"] == live_golden["vectors"]
    assert live_manifest["model_sha256"] == file_sha256(live_dir / MODEL_FILENAME)
    assert live_manifest["golden_vectors_sha256"] == file_sha256(
        live_dir / GOLDEN_VECTORS_FILENAME
    )
