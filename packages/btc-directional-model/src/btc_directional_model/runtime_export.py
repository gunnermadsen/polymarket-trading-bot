from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from .core_extract import file_sha256
from .core_training import (
    CORE_FREEZE_SCHEMA_VERSION,
    FrozenCalibrationBand,
    FrozenTimeBandedTrainingBundle,
    FrozenTrainingBundle,
)

RUNTIME_MODEL_SCHEMA_VERSION = "capitonic-btc-directional-runtime-model-v1"
TIME_BANDED_RUNTIME_MODEL_SCHEMA_VERSION = (
    "capitonic-btc-directional-runtime-model-v2"
)
RUNTIME_MANIFEST_SCHEMA_VERSION = "capitonic-btc-directional-runtime-manifest-v1"
GOLDEN_VECTORS_SCHEMA_VERSION = "capitonic-btc-directional-golden-vectors-v1"
TIME_BANDED_GOLDEN_VECTORS_SCHEMA_VERSION = (
    "capitonic-btc-directional-golden-vectors-v2"
)
MODEL_FILENAME = "model.json"
MANIFEST_FILENAME = "manifest.json"
GOLDEN_VECTORS_FILENAME = "golden-vectors.json"
MODEL_KEY_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,126}[a-z0-9])?")
RAW_PROBABILITY_CLIP = (1e-9, 1.0 - 1e-9)
CALIBRATION_LOGIT_CLIP = (-40.0, 40.0)
RUNTIME_DIRECTORY_MODE = 0o755
RUNTIME_FILE_MODE = 0o644


def export_runtime_model(
    *,
    freeze_dir: Path,
    golden_features: Path,
    output_root: Path,
    model_key: str,
) -> Path:
    """Export one immutable, native-runtime model directory.

    The source joblib is verified before it is loaded. The resulting JSON contains
    no Python or scikit-learn runtime dependency and is byte-deterministic for the
    same frozen model and golden feature cache.
    """

    if MODEL_KEY_PATTERN.fullmatch(model_key) is None:
        raise ValueError(
            "model key must contain only lowercase letters, digits, and hyphens"
        )
    bundle, freeze, freeze_sha256 = verify_frozen_bundle(freeze_dir)
    model = runtime_model_payload(
        bundle=bundle,
        freeze=freeze,
        freeze_sha256=freeze_sha256,
        model_key=model_key,
    )
    golden = golden_vectors_payload(
        bundle=bundle,
        model=model,
        golden_features=golden_features,
    )
    model_bytes = canonical_json_bytes(model)
    golden_bytes = canonical_json_bytes(golden)
    manifest = {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "model_key": model_key,
        "model_file": MODEL_FILENAME,
        "model_sha256": sha256_bytes(model_bytes),
        "golden_vectors_file": GOLDEN_VECTORS_FILENAME,
        "golden_vectors_sha256": sha256_bytes(golden_bytes),
        "feature_schema_version": model["features"]["schema_version"],
        "feature_schema_sha256": model["features"]["schema_sha256"],
        "source_freeze_manifest_sha256": freeze_sha256,
        "source_training_model_sha256": freeze["model_sha256"],
    }
    deployment = validate_deployment_metadata(freeze)
    if deployment is not None:
        manifest["deployment_scope"] = deployment["scope"]
        manifest["production_qualified"] = deployment["production_qualified"]
        manifest["live_capital_allowed"] = deployment["live_capital_allowed"]
    files = {
        MODEL_FILENAME: model_bytes,
        MANIFEST_FILENAME: canonical_json_bytes(manifest),
        GOLDEN_VECTORS_FILENAME: golden_bytes,
    }
    destination = output_root / model_key
    write_immutable_directory(destination, files)
    return destination


def verify_frozen_bundle(
    freeze_dir: Path,
) -> tuple[
    FrozenTrainingBundle | FrozenTimeBandedTrainingBundle,
    dict[str, Any],
    str,
]:
    freeze_dir = freeze_dir.resolve()
    manifest_path = freeze_dir / "freeze-manifest.json"
    manifest_hash_path = freeze_dir / "freeze-manifest.sha256"
    summary_path = freeze_dir / "model-summary.json"
    for path in (manifest_path, manifest_hash_path, summary_path):
        if not path.is_file():
            raise RuntimeError(f"frozen candidate file is missing: {path.name}")

    expected_manifest_hash = manifest_hash_path.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_hash):
        raise RuntimeError("freeze manifest checksum is malformed")
    actual_manifest_hash = file_sha256(manifest_path)
    if actual_manifest_hash != expected_manifest_hash:
        raise RuntimeError("freeze manifest hash mismatch")
    freeze = read_json_object(manifest_path)
    if freeze.get("schema_version") != CORE_FREEZE_SCHEMA_VERSION:
        raise RuntimeError("unsupported freeze manifest schema")
    validate_deployment_metadata(freeze)

    model_filename = freeze.get("model_file")
    if not isinstance(model_filename, str) or Path(model_filename).name != model_filename:
        raise RuntimeError("freeze manifest model file is unsafe")
    model_path = freeze_dir / model_filename
    if not model_path.is_file():
        raise RuntimeError("frozen training model is missing")
    if file_sha256(model_path) != freeze.get("model_sha256"):
        raise RuntimeError("frozen training model hash mismatch")
    if file_sha256(summary_path) != freeze.get("model_summary_sha256"):
        raise RuntimeError("frozen model summary hash mismatch")

    bundle = joblib.load(model_path)
    if not isinstance(
        bundle,
        (FrozenTrainingBundle, FrozenTimeBandedTrainingBundle),
    ):
        raise TypeError("unsupported frozen training bundle")
    validate_bundle_against_freeze(bundle, freeze, read_json_object(summary_path))
    return bundle, freeze, actual_manifest_hash


def validate_bundle_against_freeze(
    bundle: FrozenTrainingBundle | FrozenTimeBandedTrainingBundle,
    freeze: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    model = bundle.model
    if model.family != "histogram" or freeze.get("family") != "histogram":
        raise RuntimeError("runtime export requires a histogram model")
    if not isinstance(model.estimator, HistGradientBoostingClassifier):
        raise TypeError("unsupported histogram estimator")
    if model.standardization_means is not None or model.standardization_scales is not None:
        raise RuntimeError("histogram runtime model must not be standardized")
    if tuple(freeze.get("feature_names", [])) != tuple(model.feature_names):
        raise RuntimeError("frozen feature order does not match training model")
    if freeze.get("candidate") != model.candidate_name:
        raise RuntimeError("frozen candidate does not match training model")
    if freeze.get("hyperparameters") != model.hyperparameters:
        raise RuntimeError("frozen hyperparameters do not match training model")
    if summary.get("training_only") is not True:
        raise RuntimeError("frozen model summary is not marked training-only")
    expected_summary = {
        "candidate": model.candidate_name,
        "family": model.family,
        "feature_names": list(model.feature_names),
        "hyperparameters": model.hyperparameters,
        "imputation_medians": model.imputation_medians.tolist(),
    }
    if isinstance(bundle, FrozenTrainingBundle):
        if freeze.get("calibrator") != asdict(bundle.calibrator):
            raise RuntimeError("frozen calibrator does not match training model")
        if freeze.get("confidence_threshold") != bundle.confidence_threshold:
            raise RuntimeError(
                "frozen confidence threshold does not match training model"
            )
        expected_summary.update(
            {
                "calibrator": asdict(bundle.calibrator),
                "confidence_threshold": bundle.confidence_threshold,
            }
        )
    else:
        validate_time_banded_bundle(bundle)
        expected_bands = frozen_calibration_bands_payload(bundle.bands)
        if freeze.get("calibration_kind") != "time_banded_platt":
            raise RuntimeError("frozen calibration kind is not time-banded Platt")
        if freeze.get("calibration_bands") != expected_bands:
            raise RuntimeError(
                "frozen calibration bands do not match training model"
            )
        if freeze.get("target_kind") != bundle.target_kind:
            raise RuntimeError("frozen target kind does not match training model")
        expected_summary.update(
            {
                "calibration_kind": "time_banded_platt",
                "calibration_bands": expected_bands,
                "target_kind": bundle.target_kind,
            }
        )
    for name, expected in expected_summary.items():
        if summary.get(name) != expected:
            raise RuntimeError(f"frozen model summary field does not match: {name}")

    feature_count = len(model.feature_names)
    medians = np.asarray(model.imputation_medians, dtype=np.float64)
    if medians.shape != (feature_count,) or not np.isfinite(medians).all():
        raise RuntimeError("runtime export requires one finite median per feature")
    estimator = model.estimator
    if estimator.n_trees_per_iteration_ != 1:
        raise RuntimeError("runtime export supports one binary tree per iteration")
    if estimator.classes_.tolist() != [0, 1]:
        raise RuntimeError("runtime export requires class order [0, 1]")
    if np.asarray(estimator._baseline_prediction).shape != (1, 1):
        raise RuntimeError("runtime export requires one binary baseline logit")
    if not estimator._predictors:
        raise RuntimeError("runtime export requires at least one fitted tree")

    policy = freeze.get("prediction_policy")
    if policy != {
        "type": "first_confidence_crossing",
        "minimum_seconds_after_open": 60,
        "maximum_seconds_after_open": 240,
        "cadence_seconds": 5,
    }:
        raise RuntimeError("frozen prediction policy is not the supported 60-240/5 policy")


def runtime_model_payload(
    *,
    bundle: FrozenTrainingBundle | FrozenTimeBandedTrainingBundle,
    freeze: dict[str, Any],
    freeze_sha256: str,
    model_key: str,
) -> dict[str, Any]:
    model = bundle.model
    estimator = model.estimator
    feature_names = list(model.feature_names)
    feature_schema_version = require_string(freeze, "feature_schema_version")
    feature_hash = feature_schema_sha256(feature_schema_version, feature_names)
    trees = [
        export_tree(predictors[0], len(feature_names))
        for predictors in estimator._predictors
    ]
    baseline = float(estimator._baseline_prediction[0, 0])
    if not math.isfinite(baseline):
        raise RuntimeError("histogram baseline is not finite")
    time_banded = isinstance(bundle, FrozenTimeBandedTrainingBundle)
    payload: dict[str, Any] = {
        "schema_version": (
            TIME_BANDED_RUNTIME_MODEL_SCHEMA_VERSION
            if time_banded
            else RUNTIME_MODEL_SCHEMA_VERSION
        ),
        "model_key": model_key,
        "features": {
            "schema_version": feature_schema_version,
            "schema_sha256": feature_hash,
            "names": feature_names,
            "numeric_type": "float64",
            "non_finite_policy": "median_imputation",
            "imputation_medians": [
                finite_float(value, "imputation median")
                for value in model.imputation_medians
            ],
        },
        "estimator": {
            "type": "histogram_gradient_boosting_binary_classifier",
            "class_order": [0, 1],
            "output": "raw_logit",
            "baseline_logit": baseline,
            "tree_values_include_learning_rate": True,
            "split_comparison": "less_than_or_equal",
            "trees": trees,
        },
        "decision": {
            "probability_up_threshold": 0.5,
            "below_confidence_action": "no_trade",
            "up_action": "up",
            "down_action": "down",
        },
        "prediction_policy": dict(freeze["prediction_policy"]),
        "provenance": {
            "freeze_id": require_string(freeze, "freeze_id"),
            "freeze_created_at": require_string(freeze, "created_at"),
            "candidate": model.candidate_name,
            "family": model.family,
            "training_model_sha256": require_string(freeze, "model_sha256"),
            "training_model_summary_sha256": require_string(
                freeze,
                "model_summary_sha256",
            ),
            "freeze_manifest_sha256": freeze_sha256,
            "configuration_sha256": require_string(
                freeze,
                "configuration_sha256",
            ),
            "development_feature_sha256": require_string(
                freeze,
                "development_feature_sha256",
            ),
            "development_feature_metadata_sha256": require_string(
                freeze,
                "development_feature_metadata_sha256",
            ),
            "source_tree_sha256": require_string(freeze, "source_tree_sha256"),
            "training_git": freeze["git"],
            "holdout_range": freeze["holdout_range"],
            "random_seed": freeze["random_seed"],
            "training_hyperparameters": model.hyperparameters,
        },
    }
    if time_banded:
        payload["target"] = runtime_target_payload(bundle)
        payload["time_bands"] = [
            runtime_time_band_payload(band) for band in bundle.bands
        ]
    else:
        payload["calibration"] = runtime_calibration_payload(bundle.calibrator)
        payload["decision"]["confidence_threshold"] = finite_float(
            bundle.confidence_threshold,
            "confidence threshold",
        )
    deployment = validate_deployment_metadata(freeze)
    if deployment is not None:
        payload["deployment"] = deployment
    return payload


def validate_time_banded_bundle(
    bundle: FrozenTimeBandedTrainingBundle,
) -> None:
    if bundle.target_kind not in {"outcome_up", "path_persistence"}:
        raise RuntimeError("unsupported frozen time-banded target kind")
    expected_ranges = (
        ("60-89", 60, 90),
        ("90-119", 90, 120),
        ("120-179", 120, 180),
        ("180-240", 180, 241),
    )
    observed_ranges = tuple(
        (band.name, band.start_second, band.end_second_exclusive)
        for band in bundle.bands
    )
    if observed_ranges != expected_ranges:
        raise RuntimeError(
            "frozen time bands must exactly cover the 60-240 second runtime policy"
        )
    for band in bundle.bands:
        if not band.calibrator.converged or band.calibrator.slope <= 0.0:
            raise RuntimeError(
                f"frozen time-band calibrator is invalid: {band.name}"
            )
        threshold = finite_float(
            band.confidence_threshold,
            f"{band.name} confidence threshold",
        )
        if not 0.5 <= threshold <= 1.0:
            raise RuntimeError(
                f"frozen time-band confidence threshold is invalid: {band.name}"
            )
    if (
        bundle.target_kind == "path_persistence"
        and "btc_path_from_window_open_bps"
        not in bundle.model.feature_names
    ):
        raise RuntimeError(
            "path-persistence runtime model is missing its path-direction feature"
        )


def frozen_calibration_bands_payload(
    bands: Sequence[FrozenCalibrationBand],
) -> list[dict[str, Any]]:
    return [
        {
            "name": band.name,
            "start_second": band.start_second,
            "end_second_exclusive": band.end_second_exclusive,
            "calibrator": asdict(band.calibrator),
            "confidence_threshold": band.confidence_threshold,
        }
        for band in bands
    ]


def runtime_calibration_payload(calibrator: Any) -> dict[str, Any]:
    return {
        "type": "platt_logit",
        "slope": finite_float(calibrator.slope, "calibration slope"),
        "intercept": finite_float(
            calibrator.intercept,
            "calibration intercept",
        ),
        "input_probability_clip": {
            "minimum": RAW_PROBABILITY_CLIP[0],
            "maximum": RAW_PROBABILITY_CLIP[1],
        },
        "output_logit_clip": {
            "minimum": CALIBRATION_LOGIT_CLIP[0],
            "maximum": CALIBRATION_LOGIT_CLIP[1],
        },
    }


def runtime_time_band_payload(
    band: FrozenCalibrationBand,
) -> dict[str, Any]:
    return {
        "name": band.name,
        "start_seconds": band.start_second,
        "end_seconds_exclusive": band.end_second_exclusive,
        "calibration": runtime_calibration_payload(band.calibrator),
        "confidence_threshold": finite_float(
            band.confidence_threshold,
            f"{band.name} confidence threshold",
        ),
    }


def runtime_target_payload(
    bundle: FrozenTimeBandedTrainingBundle,
) -> dict[str, Any]:
    if bundle.target_kind == "outcome_up":
        return {"type": "outcome_up"}
    return {
        "type": "path_persistence",
        "path_direction_feature": "btc_path_from_window_open_bps",
        "zero_path_epsilon_bps": 1e-12,
        "ineligible_action": "no_trade",
    }


def validate_deployment_metadata(
    freeze: dict[str, Any],
) -> dict[str, Any] | None:
    fields = (
        "deployment_scope",
        "production_qualified",
        "live_capital_allowed",
    )
    present = tuple(name in freeze for name in fields)
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError("frozen deployment metadata is incomplete")
    scope = freeze["deployment_scope"]
    production_qualified = freeze["production_qualified"]
    live_capital_allowed = freeze["live_capital_allowed"]
    if not isinstance(scope, str) or not scope:
        raise TypeError("frozen deployment scope must be a non-empty string")
    if (
        not isinstance(production_qualified, bool)
        or not isinstance(live_capital_allowed, bool)
    ):
        raise TypeError("frozen deployment qualification fields must be booleans")
    if scope == "paper_only" and (
        production_qualified or live_capital_allowed
    ):
        raise RuntimeError(
            "paper-only frozen models cannot be production-qualified "
            "or allow live capital"
        )
    if live_capital_allowed and not production_qualified:
        raise RuntimeError(
            "live-capital permission requires production qualification"
        )
    return {
        "scope": scope,
        "production_qualified": production_qualified,
        "live_capital_allowed": live_capital_allowed,
    }


def export_tree(predictor: Any, feature_count: int) -> dict[str, Any]:
    source_nodes = predictor.nodes
    if len(source_nodes) == 0:
        raise RuntimeError("histogram tree is empty")
    nodes: list[dict[str, Any]] = []
    for index, source in enumerate(source_nodes):
        if bool(source["is_leaf"]):
            nodes.append(
                {
                    "kind": "leaf",
                    "value": finite_float(source["value"], "leaf value"),
                }
            )
            continue
        if bool(source["is_categorical"]):
            raise RuntimeError("categorical histogram splits are not supported")
        feature_index = int(source["feature_idx"])
        left = int(source["left"])
        right = int(source["right"])
        if not 0 <= feature_index < feature_count:
            raise RuntimeError("histogram split feature index is out of range")
        if not 0 <= left < len(source_nodes) or not 0 <= right < len(source_nodes):
            raise RuntimeError("histogram split child index is out of range")
        if left == index or right == index:
            raise RuntimeError("histogram split contains a self-reference")
        nodes.append(
            {
                "kind": "split",
                "feature_index": feature_index,
                "threshold": finite_float(source["num_threshold"], "split threshold"),
                "missing_go_to_left": bool(source["missing_go_to_left"]),
                "left": left,
                "right": right,
            }
        )
    validate_tree_graph(nodes)
    return {"nodes": nodes}


def validate_tree_graph(nodes: list[dict[str, Any]]) -> None:
    visited: set[int] = set()
    active: set[int] = set()

    def visit(index: int) -> None:
        if index in active:
            raise RuntimeError("histogram tree contains a cycle")
        if index in visited:
            return
        active.add(index)
        node = nodes[index]
        if node["kind"] == "split":
            visit(node["left"])
            visit(node["right"])
        active.remove(index)
        visited.add(index)

    visit(0)
    if len(visited) != len(nodes):
        raise RuntimeError("histogram tree contains unreachable nodes")


def golden_vectors_payload(
    *,
    bundle: FrozenTrainingBundle | FrozenTimeBandedTrainingBundle,
    model: dict[str, Any],
    golden_features: Path,
) -> dict[str, Any]:
    if isinstance(bundle, FrozenTimeBandedTrainingBundle):
        return time_banded_golden_vectors_payload(
            bundle=bundle,
            model=model,
            golden_features=golden_features,
        )
    golden_features = golden_features.resolve()
    if not golden_features.is_file():
        raise RuntimeError("golden feature cache is missing")
    feature_names = list(model["features"]["names"])
    available = set(pl.scan_parquet(golden_features).collect_schema().names())
    required = {"market_id", "observed_at", *feature_names}
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"golden feature cache is missing columns: {', '.join(missing)}")
    validate_golden_feature_metadata(
        golden_features,
        model["features"]["schema_version"],
    )
    frame = pl.read_parquet(
        golden_features,
        columns=["market_id", "observed_at", *feature_names],
    ).sort(["observed_at", "market_id"])
    if frame.is_empty():
        raise RuntimeError("golden feature cache is empty")
    matrix = frame.select(feature_names).cast(pl.Float64).to_numpy()
    medians = np.asarray(bundle.model.imputation_medians, dtype=np.float64)
    transformed = np.where(np.isfinite(matrix), matrix, medians)
    sklearn_raw = bundle.model.estimator._raw_predict(transformed)[:, 0]
    sklearn_probability = bundle.probability(frame)
    targets = (
        ("minimum_probability_up", float(sklearn_probability.min())),
        (
            "down_confidence_boundary",
            1.0 - float(bundle.confidence_threshold),
        ),
        ("minimum_confidence", 0.5),
        ("up_confidence_boundary", float(bundle.confidence_threshold)),
        ("maximum_probability_up", float(sklearn_probability.max())),
    )
    selected: set[int] = set()
    vectors: list[dict[str, Any]] = []
    for vector_id, target in targets:
        index = nearest_unselected_index(sklearn_probability, target, selected)
        selected.add(index)
        feature_values = json_feature_values(matrix[index])
        scored = score_runtime_model(model, feature_values)
        if scored["raw_logit"] != float(sklearn_raw[index]):
            raise RuntimeError("portable tree traversal does not match scikit-learn")
        if not math.isclose(
            scored["probability_up"],
            float(sklearn_probability[index]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError("portable calibration does not match training bundle")
        vectors.append(
            {
                "id": vector_id,
                "source": {
                    "market_id": str(frame[index, "market_id"]),
                    "observed_at": frame[index, "observed_at"].isoformat(),
                },
                "feature_values": feature_values,
                "expected": scored,
            }
        )
    vectors.append(
        {
            "id": "all_features_non_finite",
            "source": None,
            "feature_values": [None] * len(feature_names),
            "expected": score_runtime_model(model, [None] * len(feature_names)),
        }
    )
    actions = {vector["expected"]["action"] for vector in vectors}
    if not {"up", "down", "no_trade"}.issubset(actions):
        raise RuntimeError("golden vectors do not cover up, down, and no_trade")
    return {
        "schema_version": GOLDEN_VECTORS_SCHEMA_VERSION,
        "model_key": model["model_key"],
        "feature_schema_sha256": model["features"]["schema_sha256"],
        "vectors": vectors,
    }


def time_banded_golden_vectors_payload(
    *,
    bundle: FrozenTimeBandedTrainingBundle,
    model: dict[str, Any],
    golden_features: Path,
) -> dict[str, Any]:
    golden_features = golden_features.resolve()
    if not golden_features.is_file():
        raise RuntimeError("golden feature cache is missing")
    feature_names = list(model["features"]["names"])
    available = set(pl.scan_parquet(golden_features).collect_schema().names())
    required = {"market_id", "observed_at", "seconds_elapsed", *feature_names}
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(
            f"golden feature cache is missing columns: {', '.join(missing)}"
        )
    validate_golden_feature_metadata(
        golden_features,
        model["features"]["schema_version"],
    )
    frame = pl.read_parquet(
        golden_features,
        columns=["market_id", "observed_at", "seconds_elapsed", *feature_names],
    ).sort(["observed_at", "market_id"])
    if frame.is_empty():
        raise RuntimeError("golden feature cache is empty")
    matrix = frame.select(feature_names).cast(pl.Float64).to_numpy()
    medians = np.asarray(bundle.model.imputation_medians, dtype=np.float64)
    transformed = np.where(np.isfinite(matrix), matrix, medians)
    sklearn_raw = bundle.model.estimator._raw_predict(transformed)[:, 0]
    sklearn_probability = bundle.probability_up(frame)
    elapsed = frame["seconds_elapsed"].cast(pl.Int64).to_numpy()

    selected: set[int] = set()
    vector_specs: list[tuple[str, int]] = []
    for band in bundle.bands:
        in_band = np.flatnonzero(
            (elapsed >= band.start_second)
            & (elapsed < band.end_second_exclusive)
        )
        if not len(in_band):
            raise RuntimeError(f"golden feature cache has no rows for {band.name}")
        band_probability = sklearn_probability[in_band]
        for suffix, target in (
            ("minimum", float(band_probability.min())),
            ("down_boundary", 1.0 - band.confidence_threshold),
            ("no_trade", 0.5),
            ("up_boundary", band.confidence_threshold),
            ("maximum", float(band_probability.max())),
        ):
            index = nearest_unselected_subset_index(
                sklearn_probability,
                target,
                selected,
                in_band,
            )
            selected.add(index)
            vector_specs.append((f"{band.name}-{suffix}", index))

    vectors: list[dict[str, Any]] = []
    for vector_id, index in vector_specs:
        feature_values = json_feature_values(matrix[index])
        seconds_elapsed = int(elapsed[index])
        scored = score_runtime_model(
            model,
            feature_values,
            seconds_elapsed=seconds_elapsed,
        )
        if scored["raw_logit"] != float(sklearn_raw[index]):
            raise RuntimeError(
                "portable tree traversal does not match scikit-learn"
            )
        if not math.isclose(
            scored["probability_up"],
            float(sklearn_probability[index]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(
                "portable calibration does not match training bundle"
            )
        vectors.append(
            {
                "id": vector_id,
                "source": {
                    "market_id": str(frame[index, "market_id"]),
                    "observed_at": frame[index, "observed_at"].isoformat(),
                },
                "seconds_elapsed": seconds_elapsed,
                "feature_values": feature_values,
                "expected": scored,
            }
        )

    non_finite_seconds = bundle.bands[0].start_second
    vectors.append(
        {
            "id": "all_features_non_finite",
            "source": None,
            "seconds_elapsed": non_finite_seconds,
            "feature_values": [None] * len(feature_names),
            "expected": score_runtime_model(
                model,
                [None] * len(feature_names),
                seconds_elapsed=non_finite_seconds,
            ),
        }
    )
    actions = {vector["expected"]["action"] for vector in vectors}
    if not {"up", "down", "no_trade"}.issubset(actions):
        raise RuntimeError(
            "time-banded golden vectors do not cover up, down, and no_trade"
        )
    return {
        "schema_version": TIME_BANDED_GOLDEN_VECTORS_SCHEMA_VERSION,
        "model_key": model["model_key"],
        "feature_schema_sha256": model["features"]["schema_sha256"],
        "vectors": vectors,
    }


def validate_golden_feature_metadata(
    feature_path: Path,
    feature_schema_version: str,
) -> None:
    metadata_path = feature_path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise RuntimeError("golden feature metadata is missing")
    metadata = read_json_object(metadata_path)
    if metadata.get("feature_schema_version") != feature_schema_version:
        raise RuntimeError("golden feature schema does not match frozen model")
    if metadata.get("feature_file_sha256") != file_sha256(feature_path):
        raise RuntimeError("golden feature cache hash mismatch")


def score_runtime_model(
    model: dict[str, Any],
    feature_values: Sequence[float | int | None],
    *,
    seconds_elapsed: int | None = None,
) -> dict[str, Any]:
    features = model["features"]
    if len(feature_values) != len(features["names"]):
        raise ValueError("runtime feature count does not match model")
    values = []
    for value, median in zip(
        feature_values,
        features["imputation_medians"],
        strict=True,
    ):
        numeric = float(value) if value is not None and not isinstance(value, bool) else math.nan
        values.append(numeric if math.isfinite(numeric) else float(median))

    estimator = model["estimator"]
    raw_logit = float(estimator["baseline_logit"])
    for tree in estimator["trees"]:
        raw_logit += reached_leaf_value(tree["nodes"], values)
    raw_probability = stable_sigmoid(raw_logit)
    if model["schema_version"] == TIME_BANDED_RUNTIME_MODEL_SCHEMA_VERSION:
        if seconds_elapsed is None or isinstance(seconds_elapsed, bool):
            raise ValueError("time-banded runtime scoring requires seconds_elapsed")
        band = runtime_band_for_second(model, int(seconds_elapsed))
        target_probability = calibrated_probability(
            raw_probability,
            band["calibration"],
        )
        target = model["target"]
        if target["type"] == "outcome_up":
            probability_up = target_probability
        elif target["type"] == "path_persistence":
            feature_index = features["names"].index(
                target["path_direction_feature"]
            )
            raw_path_value = feature_values[feature_index]
            path = (
                float(raw_path_value)
                if raw_path_value is not None
                and not isinstance(raw_path_value, bool)
                else math.nan
            )
            epsilon = float(target["zero_path_epsilon_bps"])
            if not math.isfinite(path) or abs(path) <= epsilon:
                return {
                    "raw_logit": raw_logit,
                    "probability_up": 0.5,
                    "confidence": 0.5,
                    "action": target["ineligible_action"],
                }
            probability_up = (
                target_probability if path > 0.0 else 1.0 - target_probability
            )
        else:
            raise ValueError("unsupported runtime target type")
        confidence = max(probability_up, 1.0 - probability_up)
        decision = model["decision"]
        if confidence < float(band["confidence_threshold"]):
            action = decision["below_confidence_action"]
        elif probability_up >= float(decision["probability_up_threshold"]):
            action = decision["up_action"]
        else:
            action = decision["down_action"]
        return {
            "raw_logit": raw_logit,
            "probability_up": probability_up,
            "confidence": confidence,
            "action": action,
        }

    calibration = model["calibration"]
    probability_up = calibrated_probability(raw_probability, calibration)
    confidence = max(probability_up, 1.0 - probability_up)
    decision = model["decision"]
    if confidence < float(decision["confidence_threshold"]):
        action = decision["below_confidence_action"]
    elif probability_up >= float(decision["probability_up_threshold"]):
        action = decision["up_action"]
    else:
        action = decision["down_action"]
    return {
        "raw_logit": raw_logit,
        "probability_up": probability_up,
        "confidence": confidence,
        "action": action,
    }


def calibrated_probability(
    raw_probability: float,
    calibration: dict[str, Any],
) -> float:
    probability_clip = calibration["input_probability_clip"]
    clipped_probability = min(
        max(raw_probability, float(probability_clip["minimum"])),
        float(probability_clip["maximum"]),
    )
    calibration_input = math.log(clipped_probability / (1.0 - clipped_probability))
    calibrated_logit = (
        calibration_input * float(calibration["slope"])
        + float(calibration["intercept"])
    )
    output_clip = calibration["output_logit_clip"]
    calibrated_logit = min(
        max(calibrated_logit, float(output_clip["minimum"])),
        float(output_clip["maximum"]),
    )
    return stable_sigmoid(calibrated_logit)


def runtime_band_for_second(
    model: dict[str, Any],
    seconds_elapsed: int,
) -> dict[str, Any]:
    for band in model["time_bands"]:
        if (
            int(band["start_seconds"])
            <= seconds_elapsed
            < int(band["end_seconds_exclusive"])
        ):
            return band
    raise ValueError("seconds_elapsed is outside the runtime time bands")


def reached_leaf_value(
    nodes: Sequence[dict[str, Any]],
    values: Sequence[float],
) -> float:
    index = 0
    traversed = 0
    while True:
        if traversed >= len(nodes):
            raise RuntimeError("runtime tree traversal did not reach a leaf")
        node = nodes[index]
        if node["kind"] == "leaf":
            return float(node["value"])
        value = values[int(node["feature_index"])]
        if math.isnan(value):
            go_left = bool(node["missing_go_to_left"])
        else:
            go_left = value <= float(node["threshold"])
        index = int(node["left"] if go_left else node["right"])
        traversed += 1


def nearest_unselected_index(
    probabilities: np.ndarray,
    target: float,
    selected: set[int],
) -> int:
    distances = np.abs(probabilities - target)
    for index in np.argsort(distances, kind="stable"):
        candidate = int(index)
        if candidate not in selected:
            return candidate
    raise RuntimeError("golden feature cache has too few distinct rows")


def nearest_unselected_subset_index(
    probabilities: np.ndarray,
    target: float,
    selected: set[int],
    indexes: np.ndarray,
) -> int:
    distances = np.abs(probabilities[indexes] - target)
    for position in np.argsort(distances, kind="stable"):
        candidate = int(indexes[int(position)])
        if candidate not in selected:
            return candidate
    raise RuntimeError("golden feature cache has too few distinct rows in a time band")


def json_feature_values(values: np.ndarray) -> list[float | None]:
    return [
        float(value) if math.isfinite(float(value)) else None
        for value in values.astype(np.float64)
    ]


def feature_schema_sha256(schema_version: str, feature_names: list[str]) -> str:
    payload = {
        "schema_version": schema_version,
        "names": feature_names,
    }
    compact = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256_bytes(compact)


def stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def finite_float(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"{name} is not finite")
    return numeric


def require_string(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"freeze manifest field is invalid: {name}")
    return value


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_immutable_directory(
    destination: Path,
    files: dict[str, bytes],
) -> None:
    if destination.exists():
        validate_immutable_directory(destination, files)
        normalize_runtime_permissions(destination, files)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.partial-",
            dir=destination.parent,
        )
    )
    try:
        for name, payload in files.items():
            (staging / name).write_bytes(payload)
        normalize_runtime_permissions(staging, files)
        try:
            staging.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            validate_immutable_directory(destination, files)
            normalize_runtime_permissions(destination, files)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validate_immutable_directory(
    destination: Path,
    files: dict[str, bytes],
) -> None:
    existing_names = {path.name for path in destination.iterdir() if path.is_file()}
    if existing_names != set(files):
        raise RuntimeError(
            f"immutable model key already exists with different files: {destination}"
        )
    for name, payload in files.items():
        if (destination / name).read_bytes() != payload:
            raise RuntimeError(
                f"immutable model key already exists with different content: {destination}"
            )


def normalize_runtime_permissions(
    destination: Path,
    files: dict[str, bytes],
) -> None:
    destination.chmod(RUNTIME_DIRECTORY_MODE)
    for name in files:
        (destination / name).chmod(RUNTIME_FILE_MODE)
