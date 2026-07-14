"""Point-in-time, execution-isolated BTC Up/Down shadow ML helpers."""

from .artifact import LinearFeatureTransform, LinearLogitArtifact, schema_canary_v0
from .dataset import (
    DatasetManifest,
    FeatureObservation,
    SnapshotRow,
    build_dataset_manifest,
    canonical_feature_vector,
    feature_vector_sha256,
)
from .evaluator import ProbabilityMetrics, evaluate_probability_artifact
from .runtime_contract import (
    artifact_from_contract,
    load_runtime_contract_fixture,
    snapshot_from_contract,
)
from .splits import WalkForwardFold, chronological_grouped_walk_forward
from .trainer import fit_logistic_residual

__all__ = [
    "DatasetManifest",
    "FeatureObservation",
    "LinearFeatureTransform",
    "LinearLogitArtifact",
    "ProbabilityMetrics",
    "SnapshotRow",
    "WalkForwardFold",
    "build_dataset_manifest",
    "artifact_from_contract",
    "canonical_feature_vector",
    "feature_vector_sha256",
    "load_runtime_contract_fixture",
    "snapshot_from_contract",
    "chronological_grouped_walk_forward",
    "evaluate_probability_artifact",
    "fit_logistic_residual",
    "schema_canary_v0",
]
