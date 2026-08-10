from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from btc_directional_model.asymmetric_gen2_export import (
    EXPORT_PROVENANCE_FILENAME,
    EXPORT_PROVENANCE_SHA256_FILENAME,
    FROZEN_CORE_ORACLE_PROCESS_ID,
    GEN2_EXPORT_AUTHORIZATION_SCHEMA_VERSION,
    GEN2_SELECTION_SEAL_SCHEMA_VERSION,
    build_asymmetric_gen2_candidate_payload,
    export_asymmetric_gen2_paper_model,
    frozen_core_oracle_process_metadata,
    load_asymmetric_gen2_paper_model,
)
from btc_directional_model.asymmetric_incumbent_replay import (
    DEFAULT_FROZEN_ASYMMETRIC_INCUMBENT_MODEL,
    FROZEN_ASYMMETRIC_INCUMBENT_KEY,
    FROZEN_ASYMMETRIC_INCUMBENT_MODEL_SHA256,
    load_frozen_asymmetric_incumbent,
    score_asymmetric_runtime_row,
)
from btc_directional_model.core_extract import file_sha256
from btc_directional_model.runtime_export import (
    GOLDEN_VECTORS_FILENAME,
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    canonical_json_bytes,
    sha256_bytes,
)

MODEL_KEY = "btc-5m-asymmetric-core-oracle-gen2-paper-test-v1"
SOURCE_RUNTIME_DIR = DEFAULT_FROZEN_ASYMMETRIC_INCUMBENT_MODEL.parent


def _replacement_cells() -> list[dict[str, Any]]:
    source = load_frozen_asymmetric_incumbent()
    cells = []
    for index, cell in enumerate(
        source.payload["asymmetric_value_calibration"]["side_price_cells"]
    ):
        if (
            cell["start_seconds"] in {1, 15, 30, 45}
            and cell["minimum_price"] == 0.2
            and round(cell["maximum_price"], 10) == 0.3
        ):
            replacement = dict(cell)
            replacement.update(
                slope=0.72 + 0.01 * index,
                intercept=0.04 if cell["side"] == "yes" else -0.03,
                fitted=True,
                fallback=None,
            )
            cells.append(replacement)
    assert len(cells) == 8
    return cells


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_bytes(canonical_json_bytes(payload))
    return file_sha256(path)


def _export_inputs(tmp_path: Path) -> dict[str, Any]:
    source = load_frozen_asymmetric_incumbent()
    replacements = _replacement_cells()
    candidate = build_asymmetric_gen2_candidate_payload(
        source_model_payload=source.payload,
        model_key=MODEL_KEY,
        replacement_cells=replacements,
    )
    candidate_sha256 = sha256_bytes(canonical_json_bytes(candidate))
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True)
    economics_path = evidence_dir / "economics-evidence.json"
    economics_sha256 = _write_json(
        economics_path,
        {
            "schema_version": "test-economic-qualification-v1",
            "selected_candidate_id": "c1-supported",
            "qualified": True,
            "stressed_expectancy": 0.01,
        },
    )
    seal_path = evidence_dir / "selection-seal.json"
    seal_sha256 = _write_json(
        seal_path,
        {
            "schema_version": GEN2_SELECTION_SEAL_SCHEMA_VERSION,
            "created_at": "2026-08-09T12:00:00+00:00",
            "source_process_id": FROZEN_CORE_ORACLE_PROCESS_ID,
            "source_model_key": FROZEN_ASYMMETRIC_INCUMBENT_KEY,
            "source_model_sha256": FROZEN_ASYMMETRIC_INCUMBENT_MODEL_SHA256,
            "selected_candidate_id": "c1-supported",
            "selected_candidate_payload_sha256": candidate_sha256,
            "probability_selection_sha256": "1" * 64,
            "probability_predictions_sha256": "2" * 64,
            "calibration_fit_sha256": "3" * 64,
            "readiness_manifest_sha256": "4" * 64,
            "economics_opened": False,
        },
    )
    authorization_path = evidence_dir / "export-authorization.json"
    authorization_sha256 = _write_json(
        authorization_path,
        {
            "schema_version": GEN2_EXPORT_AUTHORIZATION_SCHEMA_VERSION,
            "created_at": "2026-08-09T12:30:00+00:00",
            "selection_seal_sha256": seal_sha256,
            "economics_evidence_sha256": economics_sha256,
            "selected_candidate_id": "c1-supported",
            "selected_candidate_payload_sha256": candidate_sha256,
            "probability_qualified": True,
            "economics_qualified": True,
            "deployment_scope": "paper_only",
            "live_capital_allowed": False,
            "production_qualified": False,
        },
    )
    return {
        "source_runtime_dir": SOURCE_RUNTIME_DIR,
        "source_process_id": FROZEN_CORE_ORACLE_PROCESS_ID,
        "source_process_metadata": frozen_core_oracle_process_metadata(),
        "replacement_cells": replacements,
        "selection_seal_path": seal_path,
        "expected_selection_seal_sha256": seal_sha256,
        "export_authorization_path": authorization_path,
        "expected_export_authorization_sha256": authorization_sha256,
        "economics_evidence_path": economics_path,
        "expected_economics_evidence_sha256": economics_sha256,
        "output_root": tmp_path / "runtime-models",
        "model_key": MODEL_KEY,
    }


def test_export_replaces_only_eight_cells_and_preserves_runtime_contract(
    tmp_path: Path,
) -> None:
    inputs = _export_inputs(tmp_path)
    destination = export_asymmetric_gen2_paper_model(**inputs)
    exported = load_asymmetric_gen2_paper_model(destination)
    source = load_frozen_asymmetric_incumbent()
    candidate = exported.runtime_model.payload

    assert destination.name == MODEL_KEY
    assert candidate["model_key"] == MODEL_KEY
    assert candidate["features"] == source.payload["features"]
    assert candidate["estimator"] == source.payload["estimator"]
    assert candidate["prediction_policy"] == source.payload["prediction_policy"]
    assert candidate["decision"] == source.payload["decision"]
    assert candidate["provenance"] == source.payload["provenance"]
    assert candidate["deployment"] == {
        "scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
    }
    assert (
        candidate["asymmetric_value_calibration"]["time_bands"]
        == source.payload["asymmetric_value_calibration"]["time_bands"]
    )
    changed = [
        (before, after)
        for before, after in zip(
            source.payload["asymmetric_value_calibration"]["side_price_cells"],
            candidate["asymmetric_value_calibration"]["side_price_cells"],
            strict=True,
        )
        if before != after
    ]
    assert len(changed) == 8
    assert all(after["fitted"] is True for _, after in changed)
    assert all(after["fallback"] is None for _, after in changed)
    assert exported.manifest["deployment_scope"] == "paper_only"
    assert exported.manifest["production_qualified"] is False
    assert exported.manifest["live_capital_allowed"] is False
    assert exported.export_provenance["source"]["process_id"] == (
        FROZEN_CORE_ORACLE_PROCESS_ID
    )
    assert exported.export_provenance["economics"]["economics_qualified"] is True


def test_exported_golden_vectors_cover_targets_and_non_target_parity(
    tmp_path: Path,
) -> None:
    destination = export_asymmetric_gen2_paper_model(**_export_inputs(tmp_path))
    exported = load_asymmetric_gen2_paper_model(destination)
    source = load_frozen_asymmetric_incumbent()
    vectors = exported.golden_vectors["vectors"]

    target_vectors = [vector for vector in vectors if vector["id"].startswith("target-")]
    parity_vectors = [
        vector for vector in vectors if vector["id"].startswith("non-target-")
    ]
    assert len(target_vectors) == 8
    assert len(parity_vectors) == 2
    assert {
        (vector["id"].split("-")[1], vector["seconds_elapsed"])
        for vector in target_vectors
    } == {
        (side, second)
        for second in (7, 22, 37, 52)
        for side in ("yes", "no")
    }
    for vector in target_vectors:
        incumbent = score_asymmetric_runtime_row(
            source,
            vector["feature_values"],
            seconds_elapsed=vector["seconds_elapsed"],
            yes_ask_vwap=vector["yes_ask_vwap"],
            no_ask_vwap=vector["no_ask_vwap"],
        )
        assert vector["expected"]["raw_logit"] == incumbent["raw_logit"]
        assert vector["expected"]["probability_up"] != incumbent["probability_up"]
    for vector in parity_vectors:
        incumbent = score_asymmetric_runtime_row(
            source,
            vector["feature_values"],
            seconds_elapsed=vector["seconds_elapsed"],
            yes_ask_vwap=vector["yes_ask_vwap"],
            no_ask_vwap=vector["no_ask_vwap"],
        )
        assert vector["expected"] == {
            key: incumbent[key]
            for key in ("raw_logit", "probability_up", "confidence", "action")
        }


def test_export_is_byte_deterministic_across_output_roots(tmp_path: Path) -> None:
    inputs = _export_inputs(tmp_path)
    first = export_asymmetric_gen2_paper_model(**inputs)
    second_inputs = {**inputs, "output_root": tmp_path / "second-runtime-models"}
    second = export_asymmetric_gen2_paper_model(**second_inputs)

    assert sorted(path.name for path in first.iterdir()) == sorted(
        path.name for path in second.iterdir()
    )
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()


def test_export_refuses_to_overwrite_even_identical_destination(tmp_path: Path) -> None:
    inputs = _export_inputs(tmp_path)
    export_asymmetric_gen2_paper_model(**inputs)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_asymmetric_gen2_paper_model(**inputs)


def test_export_rejects_source_process_metadata_drift(tmp_path: Path) -> None:
    inputs = _export_inputs(tmp_path)
    metadata = copy.deepcopy(inputs["source_process_metadata"])
    metadata["policy"] = "changed"

    with pytest.raises(RuntimeError, match="process metadata"):
        export_asymmetric_gen2_paper_model(
            **{**inputs, "source_process_metadata": metadata}
        )


def test_export_rejects_tampered_source_manifest(tmp_path: Path) -> None:
    inputs = _export_inputs(tmp_path)
    copied_source = tmp_path / "copied-source"
    shutil.copytree(SOURCE_RUNTIME_DIR, copied_source)
    manifest = json.loads((copied_source / MANIFEST_FILENAME).read_text())
    manifest["production_qualified"] = True
    (copied_source / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(RuntimeError, match="manifest"):
        export_asymmetric_gen2_paper_model(
            **{**inputs, "source_runtime_dir": copied_source}
        )


def test_export_rejects_selection_or_economics_tampering(tmp_path: Path) -> None:
    inputs = _export_inputs(tmp_path)
    inputs["selection_seal_path"].write_text(
        inputs["selection_seal_path"].read_text() + "\n"
    )
    with pytest.raises(RuntimeError, match="selection seal SHA-256 mismatch"):
        export_asymmetric_gen2_paper_model(**inputs)

    inputs = _export_inputs(tmp_path / "economics")
    authorization = json.loads(inputs["export_authorization_path"].read_text())
    authorization["economics_qualified"] = False
    inputs["expected_export_authorization_sha256"] = _write_json(
        inputs["export_authorization_path"], authorization
    )
    with pytest.raises(RuntimeError, match="did not qualify"):
        export_asymmetric_gen2_paper_model(**inputs)


def test_export_rejects_incomplete_or_fallback_replacements(tmp_path: Path) -> None:
    inputs = _export_inputs(tmp_path)
    with pytest.raises(ValueError, match="exactly eight"):
        export_asymmetric_gen2_paper_model(
            **{**inputs, "replacement_cells": inputs["replacement_cells"][:-1]}
        )

    fallback = copy.deepcopy(inputs["replacement_cells"])
    fallback[0]["fitted"] = False
    fallback[0]["fallback"] = "optimizer_not_converged"
    with pytest.raises(ValueError, match="not genuinely fitted"):
        build_asymmetric_gen2_candidate_payload(
            source_model_payload=load_frozen_asymmetric_incumbent().payload,
            model_key=MODEL_KEY,
            replacement_cells=fallback,
        )


def test_loader_rejects_model_and_provenance_tampering(tmp_path: Path) -> None:
    destination = export_asymmetric_gen2_paper_model(**_export_inputs(tmp_path))
    model = json.loads((destination / MODEL_FILENAME).read_text())
    model["prediction_policy"]["maximum_seconds_after_open"] = 239
    (destination / MODEL_FILENAME).write_bytes(canonical_json_bytes(model))
    with pytest.raises((RuntimeError, ValueError), match="schedule|manifest"):
        load_asymmetric_gen2_paper_model(destination)

    destination = export_asymmetric_gen2_paper_model(
        **_export_inputs(tmp_path / "provenance")
    )
    provenance = json.loads((destination / EXPORT_PROVENANCE_FILENAME).read_text())
    provenance["economics"]["economics_qualified"] = False
    (destination / EXPORT_PROVENANCE_FILENAME).write_bytes(
        canonical_json_bytes(provenance)
    )
    with pytest.raises(RuntimeError, match="provenance SHA-256 mismatch"):
        load_asymmetric_gen2_paper_model(destination)

    (destination / EXPORT_PROVENANCE_SHA256_FILENAME).write_text(
        file_sha256(destination / EXPORT_PROVENANCE_FILENAME) + "\n"
    )
    with pytest.raises(RuntimeError, match="economics provenance"):
        load_asymmetric_gen2_paper_model(destination)


def test_candidate_hash_binds_exact_cell_payload() -> None:
    source = load_frozen_asymmetric_incumbent()
    replacements = _replacement_cells()
    candidate = build_asymmetric_gen2_candidate_payload(
        source_model_payload=source.payload,
        model_key=MODEL_KEY,
        replacement_cells=replacements,
    )
    first_hash = sha256_bytes(canonical_json_bytes(candidate))
    replacements[0]["intercept"] += 1e-6
    modified = build_asymmetric_gen2_candidate_payload(
        source_model_payload=source.payload,
        model_key=MODEL_KEY,
        replacement_cells=replacements,
    )

    assert sha256_bytes(canonical_json_bytes(modified)) != first_hash
    assert (candidate["features"], candidate["estimator"]) == (
        modified["features"],
        modified["estimator"],
    )


def test_export_contains_only_runtime_and_hashed_provenance_files(tmp_path: Path) -> None:
    destination = export_asymmetric_gen2_paper_model(**_export_inputs(tmp_path))

    assert {path.name for path in destination.iterdir()} == {
        MODEL_FILENAME,
        MANIFEST_FILENAME,
        GOLDEN_VECTORS_FILENAME,
        EXPORT_PROVENANCE_FILENAME,
        EXPORT_PROVENANCE_SHA256_FILENAME,
    }
