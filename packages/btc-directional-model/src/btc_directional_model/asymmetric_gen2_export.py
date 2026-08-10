"""Immutable paper export for the calibrated Core+Oracle incumbent."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asymmetric_incumbent_replay import (
    FROZEN_ASYMMETRIC_INCUMBENT_KEY,
    FROZEN_ASYMMETRIC_INCUMBENT_MODEL_SHA256,
    FrozenAsymmetricRuntimeModel,
    load_frozen_asymmetric_incumbent,
    score_asymmetric_runtime_row,
    validate_asymmetric_runtime_model,
)
from .core_extract import file_sha256
from .runtime_export import (
    ASYMMETRIC_VALUE_GOLDEN_VECTORS_SCHEMA_VERSION,
    GOLDEN_VECTORS_FILENAME,
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    MODEL_KEY_PATTERN,
    RUNTIME_MANIFEST_SCHEMA_VERSION,
    canonical_json_bytes,
    read_json_object,
    sha256_bytes,
    write_immutable_directory,
)

GEN2_SELECTION_SEAL_SCHEMA_VERSION = (
    "capitonic-btc-asymmetric-gen2-selection-seal-v1"
)
GEN2_EXPORT_AUTHORIZATION_SCHEMA_VERSION = (
    "capitonic-btc-asymmetric-gen2-export-authorization-v1"
)
GEN2_EXPORT_PROVENANCE_SCHEMA_VERSION = (
    "capitonic-btc-asymmetric-gen2-export-provenance-v1"
)
EXPORT_PROVENANCE_FILENAME = "export-provenance.json"
EXPORT_PROVENANCE_SHA256_FILENAME = "export-provenance.sha256"

FROZEN_CORE_ORACLE_PROCESS_ID = "81f82de7-002b-4ac7-814b-236c6742d81c"
FROZEN_CORE_ORACLE_SOURCE_MANIFEST_SHA256 = (
    "c47886dc0a9cfc20694ec93d3fce777c19f5e2201af1f3bd78003c0c42a5f043"
)
FROZEN_CORE_ORACLE_GOLDEN_VECTORS_SHA256 = (
    "9295a49203b23fe87e48f5acd3cdd3e6e5369a66d841b3c317c9ab50445d2a3a"
)
FROZEN_CORE_ORACLE_FEATURE_SCHEMA_VERSION = (
    "btc-5m-asymmetric-core-oracle-paper-20260805-v1-features-v1"
)
FROZEN_CORE_ORACLE_FEATURE_SCHEMA_SHA256 = (
    "fe2a5aaee3df1ef899d2553712555091aa29b7481b3fed7805ba140dc8aa5014"
)
FROZEN_CORE_ORACLE_SOURCE_BENCHMARK_SHA256 = (
    "37a47df0eb7e764b69fdbc539bf658f27e31126d78dff73cbf7908229dffaba6"
)
FROZEN_CORE_ORACLE_SOURCE_TRAINING_MODEL_SHA256 = (
    "f17dfca17a69c229a19ad35ba593f7a58904ba65530f750760eb4ea8c26e6b46"
)

_PAPER_DEPLOYMENT = {
    "scope": "paper_only",
    "production_qualified": False,
    "live_capital_allowed": False,
}
_TARGET_TIME_BANDS = ((1, 15), (15, 30), (30, 45), (45, 60))
_TARGET_SIDES = ("yes", "no")
_TARGET_MINIMUM_PRICE = 0.20
_TARGET_MAXIMUM_PRICE = 0.30
_CELL_FIELDS = {
    "start_seconds",
    "end_seconds_exclusive",
    "minimum_price",
    "maximum_price",
    "side",
    "slope",
    "intercept",
    "fitted",
    "fallback",
}
_SELECTION_SEAL_FIELDS = {
    "schema_version",
    "created_at",
    "source_process_id",
    "source_model_key",
    "source_model_sha256",
    "selected_candidate_id",
    "selected_candidate_payload_sha256",
    "probability_selection_sha256",
    "probability_predictions_sha256",
    "calibration_fit_sha256",
    "readiness_manifest_sha256",
    "economics_opened",
}
_EXPORT_AUTHORIZATION_FIELDS = {
    "schema_version",
    "created_at",
    "selection_seal_sha256",
    "economics_evidence_sha256",
    "selected_candidate_id",
    "selected_candidate_payload_sha256",
    "probability_qualified",
    "economics_qualified",
    "deployment_scope",
    "live_capital_allowed",
    "production_qualified",
}
_EXPORT_PROVENANCE_FIELDS = {
    "schema_version",
    "created_at",
    "source",
    "selection",
    "economics",
    "deployment",
    "changed_calibration_cells",
}
_EXPORT_SOURCE_FIELDS = {
    "process_id",
    "process_metadata_sha256",
    "model_key",
    "model_sha256",
    "manifest_sha256",
    "golden_vectors_sha256",
}
_EXPORT_SELECTION_FIELDS = {
    "selected_candidate_id",
    "candidate_payload_sha256",
    "selection_seal_sha256",
    "probability_selection_sha256",
    "probability_predictions_sha256",
    "calibration_fit_sha256",
    "readiness_manifest_sha256",
}
_EXPORT_ECONOMICS_FIELDS = {
    "evidence_sha256",
    "export_authorization_sha256",
    "probability_qualified",
    "economics_qualified",
}
_CHANGED_CELL_PROVENANCE_FIELDS = {
    "start_seconds",
    "end_seconds_exclusive",
    "minimum_price",
    "maximum_price",
    "side",
}


@dataclass(frozen=True)
class AsymmetricGen2PaperExport:
    """A fully verified paper-only Gen2 runtime directory."""

    path: Path
    runtime_model: FrozenAsymmetricRuntimeModel
    manifest: dict[str, Any]
    golden_vectors: dict[str, Any]
    export_provenance: dict[str, Any]
    export_provenance_sha256: str


def frozen_core_oracle_process_metadata() -> dict[str, Any]:
    """Return the exact incumbent process metadata identity used by this export."""

    return {
        "comparison_cohort": "btc5m-asymmetric-value-paper-v1-20260806",
        "comparison_arm": "core_oracle",
        "strategy_family": "btc_5m_asymmetric_value_model",
        "model_key": FROZEN_ASYMMETRIC_INCUMBENT_KEY,
        "model_artifact_sha256": FROZEN_ASYMMETRIC_INCUMBENT_MODEL_SHA256,
        "model_feature_schema_version": FROZEN_CORE_ORACLE_FEATURE_SCHEMA_VERSION,
        "model_feature_schema_sha256": FROZEN_CORE_ORACLE_FEATURE_SCHEMA_SHA256,
        "model_feature_count": 75,
        "model_golden_vectors_sha256": FROZEN_CORE_ORACLE_GOLDEN_VECTORS_SHA256,
        "model_source_benchmark_sha256": (
            FROZEN_CORE_ORACLE_SOURCE_BENCHMARK_SHA256
        ),
        "source_training_model_sha256": (
            FROZEN_CORE_ORACLE_SOURCE_TRAINING_MODEL_SHA256
        ),
        "planned_run_id": "ba06d359-5861-5d82-8e29-c97cb1a52757",
        "planned_run_key": "btc-5m-asymmetric-core-oracle-paper-run-20260806-v1",
        "policy": "raw20_30_by55_edge_3c",
        "deployment_scope": "paper_only",
        "live_capital_allowed": False,
        "production_qualified": False,
        "preregistration": (
            "2026-08-06|btc-5m-asymmetric-core-oracle-paper-20260805-v1|"
            "paper-only|asymmetric-value|seconds-1-through-55|"
            "share-price-0.20-through-0.30|quantity-5|"
            "depth-participation-0.25|execution-reserve-0.01|"
            "minimum-edge-per-share-0.03"
        ),
    }


def build_asymmetric_gen2_candidate_payload(
    *,
    source_model_payload: Mapping[str, Any],
    model_key: str,
    replacement_cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Clone the incumbent payload, changing only its key and eight target cells."""

    if MODEL_KEY_PATTERN.fullmatch(model_key) is None:
        raise ValueError(
            "model key must contain only lowercase letters, digits, and hyphens"
        )
    if model_key == FROZEN_ASYMMETRIC_INCUMBENT_KEY:
        raise ValueError("Gen2 model key must differ from the incumbent model key")
    source = copy.deepcopy(dict(source_model_payload))
    validate_asymmetric_runtime_model(source)
    if source.get("model_key") != FROZEN_ASYMMETRIC_INCUMBENT_KEY:
        raise ValueError("Gen2 source payload is not the frozen incumbent")
    if sha256_bytes(canonical_json_bytes(source)) != FROZEN_ASYMMETRIC_INCUMBENT_MODEL_SHA256:
        raise RuntimeError("Gen2 source payload bytes do not match the frozen incumbent")
    if source.get("deployment") != _PAPER_DEPLOYMENT:
        raise RuntimeError("Gen2 source payload is not paper-only")

    replacements = _validated_replacement_cells(replacement_cells)
    cells = source["asymmetric_value_calibration"]["side_price_cells"]
    source_target_indices: dict[tuple[int, int, str], int] = {}
    for index, cell in enumerate(cells):
        key = _target_cell_key(cell)
        if key is None:
            continue
        if key in source_target_indices:
            raise RuntimeError("frozen incumbent contains duplicate target cells")
        if (
            cell.get("fitted") is not False
            or cell.get("slope") != 1.0
            or cell.get("intercept") != 0.0
        ):
            raise RuntimeError("frozen incumbent target cell is no longer identity fallback")
        source_target_indices[key] = index
    if set(source_target_indices) != set(replacements):
        raise RuntimeError("frozen incumbent does not contain exactly eight target cells")

    candidate = copy.deepcopy(source)
    candidate["model_key"] = model_key
    candidate_cells = candidate["asymmetric_value_calibration"]["side_price_cells"]
    for key, replacement in replacements.items():
        index = source_target_indices[key]
        source_cell = cells[index]
        candidate_cells[index] = {
            **source_cell,
            "slope": replacement["slope"],
            "intercept": replacement["intercept"],
            "fitted": True,
            "fallback": None,
        }
    validate_asymmetric_runtime_model(candidate)
    _validate_candidate_preservation(source, candidate)
    return candidate


def export_asymmetric_gen2_paper_model(
    *,
    source_runtime_dir: Path,
    source_process_id: str,
    source_process_metadata: Mapping[str, Any],
    replacement_cells: Sequence[Mapping[str, Any]],
    selection_seal_path: Path,
    expected_selection_seal_sha256: str,
    export_authorization_path: Path,
    expected_export_authorization_sha256: str,
    economics_evidence_path: Path,
    expected_economics_evidence_sha256: str,
    output_root: Path,
    model_key: str,
) -> Path:
    """Export one qualified Gen2 candidate without mutating its source process."""

    source = _load_exact_source_runtime(source_runtime_dir)
    _validate_source_process_identity(source_process_id, source_process_metadata)
    candidate = build_asymmetric_gen2_candidate_payload(
        source_model_payload=source.payload,
        model_key=model_key,
        replacement_cells=replacement_cells,
    )
    candidate_bytes = canonical_json_bytes(candidate)
    candidate_sha256 = sha256_bytes(candidate_bytes)

    selection_seal, selection_seal_sha256 = _verified_json_evidence(
        selection_seal_path,
        expected_selection_seal_sha256,
        "selection seal",
    )
    _validate_selection_seal(selection_seal, candidate_sha256)
    economics_evidence, economics_evidence_sha256 = _verified_json_evidence(
        economics_evidence_path,
        expected_economics_evidence_sha256,
        "economics evidence",
    )
    if not economics_evidence:
        raise RuntimeError("economics evidence must be a non-empty JSON object")
    authorization, authorization_sha256 = _verified_json_evidence(
        export_authorization_path,
        expected_export_authorization_sha256,
        "export authorization",
    )
    _validate_export_authorization(
        authorization,
        selection_seal=selection_seal,
        selection_seal_sha256=selection_seal_sha256,
        candidate_sha256=candidate_sha256,
        economics_evidence_sha256=economics_evidence_sha256,
    )

    source_metadata_sha256 = sha256_bytes(
        canonical_json_bytes(dict(source_process_metadata))
    )
    golden = _golden_vectors_payload(candidate, source)
    golden_bytes = canonical_json_bytes(golden)
    manifest = _manifest_payload(
        candidate,
        candidate_sha256,
        sha256_bytes(golden_bytes),
    )
    provenance = {
        "schema_version": GEN2_EXPORT_PROVENANCE_SCHEMA_VERSION,
        "created_at": authorization["created_at"],
        "source": {
            "process_id": source_process_id,
            "process_metadata_sha256": source_metadata_sha256,
            "model_key": source.payload["model_key"],
            "model_sha256": source.model_sha256,
            "manifest_sha256": source.manifest_sha256,
            "golden_vectors_sha256": FROZEN_CORE_ORACLE_GOLDEN_VECTORS_SHA256,
        },
        "selection": {
            "selected_candidate_id": selection_seal["selected_candidate_id"],
            "candidate_payload_sha256": candidate_sha256,
            "selection_seal_sha256": selection_seal_sha256,
            "probability_selection_sha256": selection_seal[
                "probability_selection_sha256"
            ],
            "probability_predictions_sha256": selection_seal[
                "probability_predictions_sha256"
            ],
            "calibration_fit_sha256": selection_seal["calibration_fit_sha256"],
            "readiness_manifest_sha256": selection_seal[
                "readiness_manifest_sha256"
            ],
        },
        "economics": {
            "evidence_sha256": economics_evidence_sha256,
            "export_authorization_sha256": authorization_sha256,
            "probability_qualified": True,
            "economics_qualified": True,
        },
        "deployment": copy.deepcopy(_PAPER_DEPLOYMENT),
        "changed_calibration_cells": [
            {
                "start_seconds": start,
                "end_seconds_exclusive": end,
                "minimum_price": _TARGET_MINIMUM_PRICE,
                "maximum_price": _TARGET_MAXIMUM_PRICE,
                "side": side,
            }
            for start, end in _TARGET_TIME_BANDS
            for side in _TARGET_SIDES
        ],
    }
    provenance_bytes = canonical_json_bytes(provenance)
    provenance_sha256 = sha256_bytes(provenance_bytes)
    destination = output_root.resolve() / model_key
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite runtime model: {destination}")
    write_immutable_directory(
        destination,
        {
            MODEL_FILENAME: candidate_bytes,
            MANIFEST_FILENAME: canonical_json_bytes(manifest),
            GOLDEN_VECTORS_FILENAME: golden_bytes,
            EXPORT_PROVENANCE_FILENAME: provenance_bytes,
            EXPORT_PROVENANCE_SHA256_FILENAME: (provenance_sha256 + "\n").encode(),
        },
    )
    load_asymmetric_gen2_paper_model(destination)
    return destination


def load_asymmetric_gen2_paper_model(
    runtime_dir: Path,
) -> AsymmetricGen2PaperExport:
    """Load and independently verify a Gen2 paper export and its golden vectors."""

    directory = runtime_dir.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Gen2 runtime model directory is missing: {directory}")
    required = (
        MODEL_FILENAME,
        MANIFEST_FILENAME,
        GOLDEN_VECTORS_FILENAME,
        EXPORT_PROVENANCE_FILENAME,
        EXPORT_PROVENANCE_SHA256_FILENAME,
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise RuntimeError("Gen2 runtime model files are missing: " + ", ".join(missing))

    model_path = directory / MODEL_FILENAME
    manifest_path = directory / MANIFEST_FILENAME
    golden_path = directory / GOLDEN_VECTORS_FILENAME
    provenance_path = directory / EXPORT_PROVENANCE_FILENAME
    model = read_json_object(model_path)
    manifest = read_json_object(manifest_path)
    golden = read_json_object(golden_path)
    provenance = read_json_object(provenance_path)
    model_sha256 = file_sha256(model_path)
    manifest_sha256 = file_sha256(manifest_path)
    golden_sha256 = file_sha256(golden_path)
    provenance_sha256 = file_sha256(provenance_path)
    recorded_provenance_sha256 = (
        directory / EXPORT_PROVENANCE_SHA256_FILENAME
    ).read_text().strip()
    if (
        not _is_sha256(recorded_provenance_sha256)
        or recorded_provenance_sha256 != provenance_sha256
    ):
        raise RuntimeError("Gen2 export provenance SHA-256 mismatch")

    source = _load_exact_source_runtime(
        Path(__file__).resolve().parents[2]
        / "runtime-models"
        / FROZEN_ASYMMETRIC_INCUMBENT_KEY
    )
    validate_asymmetric_runtime_model(model)
    _validate_candidate_preservation(source.payload, model)
    if directory.name != model["model_key"]:
        raise RuntimeError("Gen2 runtime directory does not match its model key")
    expected_manifest = _manifest_payload(model, model_sha256, golden_sha256)
    if manifest != expected_manifest:
        raise RuntimeError("Gen2 runtime manifest does not match its artifact")
    _validate_export_provenance(
        provenance,
        model=model,
        model_sha256=model_sha256,
        source=source,
    )

    runtime_model = FrozenAsymmetricRuntimeModel(
        path=model_path,
        payload=model,
        model_sha256=model_sha256,
        manifest_sha256=manifest_sha256,
        feature_contract=validate_asymmetric_runtime_model(model),
    )
    _validate_golden_vectors(golden, runtime_model, source)
    return AsymmetricGen2PaperExport(
        path=directory,
        runtime_model=runtime_model,
        manifest=manifest,
        golden_vectors=golden,
        export_provenance=provenance,
        export_provenance_sha256=provenance_sha256,
    )


def _load_exact_source_runtime(source_runtime_dir: Path) -> FrozenAsymmetricRuntimeModel:
    directory = source_runtime_dir.resolve()
    source = load_frozen_asymmetric_incumbent(directory / MODEL_FILENAME)
    if source.manifest_sha256 != FROZEN_CORE_ORACLE_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("frozen Core+Oracle source manifest SHA-256 changed")
    golden_path = directory / GOLDEN_VECTORS_FILENAME
    if (
        not golden_path.is_file()
        or file_sha256(golden_path) != FROZEN_CORE_ORACLE_GOLDEN_VECTORS_SHA256
    ):
        raise RuntimeError("frozen Core+Oracle source golden vectors SHA-256 changed")
    golden = read_json_object(golden_path)
    if (
        golden.get("schema_version")
        != ASYMMETRIC_VALUE_GOLDEN_VECTORS_SCHEMA_VERSION
        or golden.get("model_key") != FROZEN_ASYMMETRIC_INCUMBENT_KEY
        or golden.get("feature_schema_version")
        != FROZEN_CORE_ORACLE_FEATURE_SCHEMA_VERSION
        or golden.get("feature_schema_sha256")
        != FROZEN_CORE_ORACLE_FEATURE_SCHEMA_SHA256
    ):
        raise RuntimeError("frozen Core+Oracle source golden-vector identity changed")
    return source


def _validate_source_process_identity(
    source_process_id: str,
    source_process_metadata: Mapping[str, Any],
) -> None:
    if source_process_id != FROZEN_CORE_ORACLE_PROCESS_ID:
        raise RuntimeError("Gen2 source process ID does not match the incumbent")
    if dict(source_process_metadata) != frozen_core_oracle_process_metadata():
        raise RuntimeError("Gen2 source process metadata does not match the incumbent")


def _validated_replacement_cells(
    replacement_cells: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    if len(replacement_cells) != len(_TARGET_TIME_BANDS) * len(_TARGET_SIDES):
        raise ValueError("Gen2 export requires exactly eight replacement cells")
    indexed: dict[tuple[int, int, str], dict[str, Any]] = {}
    for raw in replacement_cells:
        cell = dict(raw)
        if set(cell) != _CELL_FIELDS:
            raise ValueError("Gen2 replacement cell fields do not match runtime schema")
        key = _target_cell_key(cell)
        if key is None:
            raise ValueError("Gen2 replacement is outside the eight target cells")
        if key in indexed:
            raise ValueError("Gen2 replacement cells contain a duplicate")
        slope = cell.get("slope")
        intercept = cell.get("intercept")
        if (
            not _is_finite_number(slope)
            or float(slope) <= 0.0
            or not _is_finite_number(intercept)
            or cell.get("fitted") is not True
            or cell.get("fallback") is not None
        ):
            raise ValueError("Gen2 replacement cell is not genuinely fitted")
        indexed[key] = {
            **cell,
            "slope": float(slope),
            "intercept": float(intercept),
        }
    expected = {
        (start, end, side)
        for start, end in _TARGET_TIME_BANDS
        for side in _TARGET_SIDES
    }
    if set(indexed) != expected:
        raise ValueError("Gen2 replacements do not cover all eight target cells")
    return indexed


def _target_cell_key(cell: Mapping[str, Any]) -> tuple[int, int, str] | None:
    start = cell.get("start_seconds")
    end = cell.get("end_seconds_exclusive")
    side = cell.get("side")
    minimum = cell.get("minimum_price")
    maximum = cell.get("maximum_price")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or side not in _TARGET_SIDES
        or not _is_finite_number(minimum)
        or not _is_finite_number(maximum)
        or not math.isclose(float(minimum), _TARGET_MINIMUM_PRICE, abs_tol=1e-12)
        or not math.isclose(float(maximum), _TARGET_MAXIMUM_PRICE, abs_tol=1e-12)
        or (start, end) not in _TARGET_TIME_BANDS
    ):
        return None
    return start, end, side


def _validate_candidate_preservation(
    source: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    if candidate.get("model_key") == source.get("model_key"):
        raise RuntimeError("Gen2 candidate reused the incumbent model key")
    if candidate.get("deployment") != _PAPER_DEPLOYMENT:
        raise RuntimeError("Gen2 candidate is not restricted to paper-only use")
    for field, source_value in source.items():
        if field in {"model_key", "asymmetric_value_calibration"}:
            continue
        if candidate.get(field) != source_value:
            raise RuntimeError(f"Gen2 candidate changed frozen field: {field}")
    if set(candidate) != set(source):
        raise RuntimeError("Gen2 candidate changed the runtime model schema")
    source_calibration = source["asymmetric_value_calibration"]
    candidate_calibration = candidate["asymmetric_value_calibration"]
    if candidate_calibration.get("time_bands") != source_calibration["time_bands"]:
        raise RuntimeError("Gen2 candidate changed parent time calibration")
    source_cells = source_calibration["side_price_cells"]
    candidate_cells = candidate_calibration.get("side_price_cells")
    if not isinstance(candidate_cells, list) or len(candidate_cells) != len(source_cells):
        raise RuntimeError("Gen2 candidate changed calibration-cell topology")
    changed = 0
    for source_cell, candidate_cell in zip(source_cells, candidate_cells, strict=True):
        target = _target_cell_key(source_cell)
        if target is None:
            if candidate_cell != source_cell:
                raise RuntimeError("Gen2 candidate changed a non-target calibration cell")
            continue
        changed += 1
        expected_identity = {
            key: value
            for key, value in source_cell.items()
            if key not in {"slope", "intercept", "fitted", "fallback"}
        }
        candidate_identity = {
            key: value
            for key, value in candidate_cell.items()
            if key not in {"slope", "intercept", "fitted", "fallback"}
        }
        if (
            expected_identity != candidate_identity
            or candidate_cell.get("fitted") is not True
            or candidate_cell.get("fallback") is not None
            or not _is_finite_number(candidate_cell.get("slope"))
            or float(candidate_cell["slope"]) <= 0.0
            or not _is_finite_number(candidate_cell.get("intercept"))
        ):
            raise RuntimeError("Gen2 candidate target calibration cell is invalid")
    if changed != 8:
        raise RuntimeError("Gen2 candidate did not replace exactly eight target cells")


def _verified_json_evidence(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], str]:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"expected {label} SHA-256 is malformed")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    actual = file_sha256(resolved)
    if actual != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 mismatch")
    return read_json_object(resolved), actual


def _validate_selection_seal(
    seal: dict[str, Any],
    candidate_sha256: str,
) -> None:
    if set(seal) != _SELECTION_SEAL_FIELDS:
        raise RuntimeError("Gen2 selection seal fields are invalid")
    if seal.get("schema_version") != GEN2_SELECTION_SEAL_SCHEMA_VERSION:
        raise RuntimeError("Gen2 selection seal schema is unsupported")
    if not isinstance(seal.get("created_at"), str) or not seal["created_at"]:
        raise RuntimeError("Gen2 selection seal timestamp is invalid")
    if (
        seal.get("source_process_id") != FROZEN_CORE_ORACLE_PROCESS_ID
        or seal.get("source_model_key") != FROZEN_ASYMMETRIC_INCUMBENT_KEY
        or seal.get("source_model_sha256")
        != FROZEN_ASYMMETRIC_INCUMBENT_MODEL_SHA256
    ):
        raise RuntimeError("Gen2 selection seal source identity changed")
    if (
        not isinstance(seal.get("selected_candidate_id"), str)
        or not seal["selected_candidate_id"]
        or seal.get("selected_candidate_payload_sha256") != candidate_sha256
        or seal.get("economics_opened") is not False
    ):
        raise RuntimeError("Gen2 selection seal candidate identity is invalid")
    for field in (
        "probability_selection_sha256",
        "probability_predictions_sha256",
        "calibration_fit_sha256",
        "readiness_manifest_sha256",
    ):
        if not _is_sha256(seal.get(field)):
            raise RuntimeError(f"Gen2 selection seal hash is invalid: {field}")


def _validate_export_authorization(
    authorization: dict[str, Any],
    *,
    selection_seal: dict[str, Any],
    selection_seal_sha256: str,
    candidate_sha256: str,
    economics_evidence_sha256: str,
) -> None:
    if set(authorization) != _EXPORT_AUTHORIZATION_FIELDS:
        raise RuntimeError("Gen2 export authorization fields are invalid")
    if (
        authorization.get("schema_version")
        != GEN2_EXPORT_AUTHORIZATION_SCHEMA_VERSION
        or not isinstance(authorization.get("created_at"), str)
        or not authorization["created_at"]
        or authorization.get("selection_seal_sha256") != selection_seal_sha256
        or authorization.get("economics_evidence_sha256")
        != economics_evidence_sha256
        or authorization.get("selected_candidate_id")
        != selection_seal["selected_candidate_id"]
        or authorization.get("selected_candidate_payload_sha256")
        != candidate_sha256
        or authorization.get("probability_qualified") is not True
        or authorization.get("economics_qualified") is not True
        or authorization.get("deployment_scope") != "paper_only"
        or authorization.get("live_capital_allowed") is not False
        or authorization.get("production_qualified") is not False
    ):
        raise RuntimeError("Gen2 export authorization did not qualify this paper candidate")


def _manifest_payload(
    model: dict[str, Any],
    model_sha256: str,
    golden_sha256: str,
) -> dict[str, Any]:
    features = model["features"]
    provenance = model["provenance"]
    return {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "model_key": model["model_key"],
        "model_file": MODEL_FILENAME,
        "model_sha256": model_sha256,
        "golden_vectors_file": GOLDEN_VECTORS_FILENAME,
        "golden_vectors_sha256": golden_sha256,
        "feature_schema_version": features["schema_version"],
        "feature_schema_sha256": features["schema_sha256"],
        "source_freeze_manifest_sha256": provenance["source_benchmark_sha256"],
        "source_training_model_sha256": provenance["source_training_model_sha256"],
        "deployment_scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
    }


def _golden_vectors_payload(
    candidate: dict[str, Any],
    source: FrozenAsymmetricRuntimeModel,
) -> dict[str, Any]:
    source_golden = read_json_object(source.path.with_name(GOLDEN_VECTORS_FILENAME))
    vectors = source_golden.get("vectors")
    if not isinstance(vectors, list) or len(vectors) != 1:
        raise RuntimeError("frozen source must contain one canonical feature vector")
    feature_values = vectors[0].get("feature_values")
    if not isinstance(feature_values, list) or len(feature_values) != 75:
        raise RuntimeError("frozen source golden feature vector is invalid")
    candidate_model = FrozenAsymmetricRuntimeModel(
        path=Path(MODEL_FILENAME),
        payload=candidate,
        model_sha256=sha256_bytes(canonical_json_bytes(candidate)),
        manifest_sha256="0" * 64,
        feature_contract=validate_asymmetric_runtime_model(candidate),
    )
    routes: list[tuple[str, int, float, float]] = []
    representative_second = {(1, 15): 7, (15, 30): 22, (30, 45): 37, (45, 60): 52}
    for start, end in _TARGET_TIME_BANDS:
        second = representative_second[(start, end)]
        routes.extend(
            (
                (f"target-yes-{start:03d}-{end - 1:03d}", second, 0.25, 0.75),
                (f"target-no-{start:03d}-{end - 1:03d}", second, 0.75, 0.25),
            )
        )
    routes.extend(
        (
            ("non-target-price-030-044", 37, 0.75, 0.75),
            ("non-target-time-060-089", 60, 0.25, 0.75),
        )
    )
    golden_vectors = []
    for vector_id, second, yes_price, no_price in routes:
        scored = score_asymmetric_runtime_row(
            candidate_model,
            feature_values,
            seconds_elapsed=second,
            yes_ask_vwap=yes_price,
            no_ask_vwap=no_price,
        )
        golden_vectors.append(
            {
                "id": vector_id,
                "source": None,
                "seconds_elapsed": second,
                "feature_values": feature_values,
                "yes_ask_vwap": yes_price,
                "no_ask_vwap": no_price,
                "expected": {
                    "raw_logit": scored["raw_logit"],
                    "probability_up": scored["probability_up"],
                    "confidence": scored["confidence"],
                    "action": scored["action"],
                },
            }
        )
    return {
        "schema_version": ASYMMETRIC_VALUE_GOLDEN_VECTORS_SCHEMA_VERSION,
        "model_key": candidate["model_key"],
        "feature_schema_version": candidate["features"]["schema_version"],
        "feature_schema_sha256": candidate["features"]["schema_sha256"],
        "vectors": golden_vectors,
    }


def _validate_golden_vectors(
    golden: dict[str, Any],
    model: FrozenAsymmetricRuntimeModel,
    source: FrozenAsymmetricRuntimeModel,
) -> None:
    if (
        golden.get("schema_version")
        != ASYMMETRIC_VALUE_GOLDEN_VECTORS_SCHEMA_VERSION
        or golden.get("model_key") != model.payload["model_key"]
        or golden.get("feature_schema_version")
        != model.payload["features"]["schema_version"]
        or golden.get("feature_schema_sha256")
        != model.payload["features"]["schema_sha256"]
    ):
        raise RuntimeError("Gen2 golden-vector identity is invalid")
    vectors = golden.get("vectors")
    if not isinstance(vectors, list) or len(vectors) != 10:
        raise RuntimeError("Gen2 export requires eight target and two parity vectors")
    ids = [vector.get("id") for vector in vectors if isinstance(vector, dict)]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise RuntimeError("Gen2 golden-vector routes are invalid")
    if sum(str(vector_id).startswith("target-") for vector_id in ids) != 8:
        raise RuntimeError("Gen2 golden vectors do not cover all eight target routes")
    if sum(str(vector_id).startswith("non-target-") for vector_id in ids) != 2:
        raise RuntimeError("Gen2 golden vectors do not cover non-target parity routes")
    for vector in vectors:
        if not isinstance(vector, dict) or set(vector) != {
            "id",
            "source",
            "seconds_elapsed",
            "feature_values",
            "yes_ask_vwap",
            "no_ask_vwap",
            "expected",
        }:
            raise RuntimeError("Gen2 golden vector has invalid fields")
        if vector["source"] is not None:
            raise RuntimeError("Gen2 golden vector source must be null")
        expected = vector.get("expected")
        if not isinstance(expected, dict) or set(expected) != {
            "raw_logit",
            "probability_up",
            "confidence",
            "action",
        }:
            raise RuntimeError("Gen2 golden expected result is invalid")
        scored = score_asymmetric_runtime_row(
            model,
            vector["feature_values"],
            seconds_elapsed=vector["seconds_elapsed"],
            yes_ask_vwap=vector["yes_ask_vwap"],
            no_ask_vwap=vector["no_ask_vwap"],
        )
        for field in ("raw_logit", "probability_up", "confidence", "action"):
            if scored[field] != expected[field]:
                raise RuntimeError(f"Gen2 golden-vector mismatch: {vector['id']}:{field}")
        if str(vector["id"]).startswith("non-target-"):
            source_scored = score_asymmetric_runtime_row(
                source,
                vector["feature_values"],
                seconds_elapsed=vector["seconds_elapsed"],
                yes_ask_vwap=vector["yes_ask_vwap"],
                no_ask_vwap=vector["no_ask_vwap"],
            )
            if scored != source_scored:
                raise RuntimeError("Gen2 non-target golden route lost incumbent parity")


def _validate_export_provenance(
    provenance: dict[str, Any],
    *,
    model: dict[str, Any],
    model_sha256: str,
    source: FrozenAsymmetricRuntimeModel,
) -> None:
    if set(provenance) != _EXPORT_PROVENANCE_FIELDS:
        raise RuntimeError("Gen2 export provenance fields are invalid")
    if provenance.get("schema_version") != GEN2_EXPORT_PROVENANCE_SCHEMA_VERSION:
        raise RuntimeError("Gen2 export provenance schema is unsupported")
    if not isinstance(provenance.get("created_at"), str) or not provenance["created_at"]:
        raise RuntimeError("Gen2 export provenance timestamp is invalid")
    expected_source = {
        "process_id": FROZEN_CORE_ORACLE_PROCESS_ID,
        "process_metadata_sha256": sha256_bytes(
            canonical_json_bytes(frozen_core_oracle_process_metadata())
        ),
        "model_key": FROZEN_ASYMMETRIC_INCUMBENT_KEY,
        "model_sha256": FROZEN_ASYMMETRIC_INCUMBENT_MODEL_SHA256,
        "manifest_sha256": FROZEN_CORE_ORACLE_SOURCE_MANIFEST_SHA256,
        "golden_vectors_sha256": FROZEN_CORE_ORACLE_GOLDEN_VECTORS_SHA256,
    }
    source_provenance = provenance.get("source")
    if (
        not isinstance(source_provenance, dict)
        or set(source_provenance) != _EXPORT_SOURCE_FIELDS
        or source_provenance != expected_source
    ):
        raise RuntimeError("Gen2 export source provenance changed")
    if source.model_sha256 != expected_source["model_sha256"]:
        raise RuntimeError("Gen2 export source artifact is unavailable")
    selection = provenance.get("selection")
    economics = provenance.get("economics")
    if (
        not isinstance(selection, dict)
        or set(selection) != _EXPORT_SELECTION_FIELDS
        or not isinstance(selection.get("selected_candidate_id"), str)
        or not selection["selected_candidate_id"]
        or selection.get("candidate_payload_sha256") != model_sha256
        or any(
            not _is_sha256(selection.get(field))
            for field in (
                "selection_seal_sha256",
                "probability_selection_sha256",
                "probability_predictions_sha256",
                "calibration_fit_sha256",
                "readiness_manifest_sha256",
            )
        )
    ):
        raise RuntimeError("Gen2 selection provenance is invalid")
    if (
        not isinstance(economics, dict)
        or set(economics) != _EXPORT_ECONOMICS_FIELDS
        or not _is_sha256(economics.get("evidence_sha256"))
        or not _is_sha256(economics.get("export_authorization_sha256"))
        or economics.get("probability_qualified") is not True
        or economics.get("economics_qualified") is not True
    ):
        raise RuntimeError("Gen2 economics provenance is invalid")
    if provenance.get("deployment") != _PAPER_DEPLOYMENT:
        raise RuntimeError("Gen2 export provenance is not paper-only")
    changed = provenance.get("changed_calibration_cells")
    if (
        not isinstance(changed, list)
        or len(changed) != 8
        or any(
            not isinstance(cell, dict)
            or set(cell) != _CHANGED_CELL_PROVENANCE_FIELDS
            for cell in changed
        )
    ):
        raise RuntimeError("Gen2 changed-cell provenance is incomplete")
    observed = {
        (
            cell.get("start_seconds"),
            cell.get("end_seconds_exclusive"),
            cell.get("side"),
        )
        for cell in changed
        if isinstance(cell, dict)
        and math.isclose(float(cell.get("minimum_price", math.nan)), 0.20, abs_tol=1e-12)
        and math.isclose(float(cell.get("maximum_price", math.nan)), 0.30, abs_tol=1e-12)
    }
    expected = {
        (start, end, side)
        for start, end in _TARGET_TIME_BANDS
        for side in _TARGET_SIDES
    }
    if observed != expected:
        raise RuntimeError("Gen2 changed-cell provenance is invalid")
    if model.get("deployment") != _PAPER_DEPLOYMENT:
        raise RuntimeError("Gen2 model deployment scope changed")


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
