from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .artifact import LinearFeatureTransform, LinearLogitArtifact, build_artifact
from .dataset import SnapshotRow


RUNTIME_CONTRACT_VERSION = "btc_5m_ml_runtime_contract_v2"
DEFAULT_RUNTIME_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "runtime_v2_contract.json"
)


def load_runtime_contract_fixture(
    path: str | Path = DEFAULT_RUNTIME_CONTRACT_PATH,
) -> dict[str, Any]:
    """Load and minimally validate the shared Rust/Python runtime contract fixture."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("contract_version") != RUNTIME_CONTRACT_VERSION:
        raise ValueError("unsupported BTC ML runtime contract fixture version")
    manifest = value.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("runtime contract fixture is missing its manifest")
    expected_manifest_hash = hashlib.sha256(str(manifest.get("seed", "")).encode("utf-8")).hexdigest()
    if manifest.get("sha256") != expected_manifest_hash:
        raise ValueError("runtime contract manifest sentinel hash mismatch")
    if not isinstance(value.get("artifacts"), list) or not value["artifacts"]:
        raise ValueError("runtime contract fixture has no artifacts")
    if not isinstance(value.get("vector"), dict):
        raise ValueError("runtime contract fixture has no vector")
    return value


def artifact_from_contract(
    specification: Mapping[str, Any], manifest_sha256: str
) -> LinearLogitArtifact:
    """Build an artifact through the same Python contract used by offline training."""
    return build_artifact(
        model_version=str(specification["model_version"]),
        task=str(specification["task"]),
        feature_schema_version=str(specification["feature_schema_version"]),
        features=(
            LinearFeatureTransform(
                name=str(name), mean=0.0, scale=1.0, coefficient=0.0
            )
            for name in specification["feature_names"]
        ),
        intercept=0.0,
        dataset_manifest_sha256=manifest_sha256,
        trained_through_ms=None,
    )


def snapshot_from_contract(specification: Mapping[str, Any]) -> SnapshotRow:
    """Build the point-in-time row whose canonical vector is shared with Rust."""
    return SnapshotRow.from_mapping(specification)
