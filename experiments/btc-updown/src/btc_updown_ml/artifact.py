from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import struct
from typing import Any, Iterable, Mapping


ARTIFACT_VERSION = "linear_logit_artifact_v1"
VALID_TASKS = {
    "settlement_probability_residual",
    "fok_fill_probability",
    "fok_post_fill_toxicity_probability",
}


@dataclass(frozen=True)
class LinearFeatureTransform:
    name: str
    mean: float
    scale: float
    coefficient: float

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("feature transform name must not be empty")
        if not all(math.isfinite(value) for value in (self.mean, self.scale, self.coefficient)):
            raise ValueError(f"feature transform {self.name!r} contains a non-finite value")
        if self.scale <= 0.0:
            raise ValueError(f"feature transform {self.name!r} scale must be positive")


@dataclass(frozen=True)
class LinearLogitArtifact:
    artifact_version: str
    model_version: str
    task: str
    prior_kind: str
    feature_schema_version: str
    feature_schema_sha256: str
    dataset_manifest_sha256: str | None
    trained_through_ms: int | None
    intercept: float
    features: tuple[LinearFeatureTransform, ...]
    artifact_sha256: str

    def validate(self) -> None:
        if self.artifact_version != ARTIFACT_VERSION:
            raise ValueError(f"unsupported artifact version {self.artifact_version!r}")
        if not self.model_version.strip() or not self.feature_schema_version.strip():
            raise ValueError("model and feature schema versions must not be empty")
        if self.task not in VALID_TASKS:
            raise ValueError(f"unsupported ML task {self.task!r}")
        if self.prior_kind != "provided_probability":
            raise ValueError(f"unsupported prior kind {self.prior_kind!r}")
        if not math.isfinite(self.intercept):
            raise ValueError("intercept must be finite")
        if not self.features:
            raise ValueError("artifact features must not be empty")
        names: set[str] = set()
        for feature in self.features:
            feature.validate()
            if feature.name in names:
                raise ValueError(f"duplicate artifact feature {feature.name!r}")
            names.add(feature.name)
        expected_schema_hash = hashlib.sha256(
            canonical_feature_schema(
                self.feature_schema_version, [feature.name for feature in self.features]
            ).encode("utf-8")
        ).hexdigest()
        if self.feature_schema_sha256 != expected_schema_hash:
            raise ValueError("feature schema hash mismatch")
        if self.dataset_manifest_sha256 is not None and not _is_sha256(
            self.dataset_manifest_sha256
        ):
            raise ValueError("dataset manifest hash must be a SHA-256 hex digest")
        expected_artifact_hash = hashlib.sha256(
            canonical_artifact(self).encode("utf-8")
        ).hexdigest()
        if self.artifact_sha256 != expected_artifact_hash:
            raise ValueError("artifact hash mismatch")

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["features"] = [asdict(feature) for feature in self.features]
        return value

    def to_json(self) -> str:
        self.validate()
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True, allow_nan=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LinearLogitArtifact":
        artifact = cls(
            artifact_version=str(value["artifact_version"]),
            model_version=str(value["model_version"]),
            task=str(value["task"]),
            prior_kind=str(value["prior_kind"]),
            feature_schema_version=str(value["feature_schema_version"]),
            feature_schema_sha256=str(value["feature_schema_sha256"]),
            dataset_manifest_sha256=(
                None
                if value.get("dataset_manifest_sha256") is None
                else str(value["dataset_manifest_sha256"])
            ),
            trained_through_ms=(
                None if value.get("trained_through_ms") is None else int(value["trained_through_ms"])
            ),
            intercept=float(value["intercept"]),
            features=tuple(
                LinearFeatureTransform(
                    name=str(feature["name"]),
                    mean=float(feature["mean"]),
                    scale=float(feature["scale"]),
                    coefficient=float(feature["coefficient"]),
                )
                for feature in value["features"]
            ),
            artifact_sha256=str(value["artifact_sha256"]),
        )
        artifact.validate()
        return artifact


def build_artifact(
    *,
    model_version: str,
    task: str,
    feature_schema_version: str,
    features: Iterable[LinearFeatureTransform],
    intercept: float,
    dataset_manifest_sha256: str | None,
    trained_through_ms: int | None,
) -> LinearLogitArtifact:
    feature_tuple = tuple(features)
    schema_hash = hashlib.sha256(
        canonical_feature_schema(
            feature_schema_version, [feature.name for feature in feature_tuple]
        ).encode("utf-8")
    ).hexdigest()
    unsigned = LinearLogitArtifact(
        artifact_version=ARTIFACT_VERSION,
        model_version=model_version,
        task=task,
        prior_kind="provided_probability",
        feature_schema_version=feature_schema_version,
        feature_schema_sha256=schema_hash,
        dataset_manifest_sha256=dataset_manifest_sha256,
        trained_through_ms=trained_through_ms,
        intercept=intercept,
        features=feature_tuple,
        artifact_sha256="",
    )
    artifact = LinearLogitArtifact(
        **{**unsigned.to_mapping(), "features": feature_tuple, "artifact_sha256": hashlib.sha256(
            canonical_artifact(unsigned).encode("utf-8")
        ).hexdigest()}
    )
    artifact.validate()
    return artifact


def schema_canary_v0(
    *, task: str, feature_schema_version: str, feature_names: Iterable[str]
) -> LinearLogitArtifact:
    """Return a non-alpha artifact that exactly reproduces its supplied prior."""
    return build_artifact(
        model_version="schema_canary_v0",
        task=task,
        feature_schema_version=feature_schema_version,
        features=(
            LinearFeatureTransform(name=name, mean=0.0, scale=1.0, coefficient=0.0)
            for name in feature_names
        ),
        intercept=0.0,
        dataset_manifest_sha256=None,
        trained_through_ms=None,
    )


def canonical_feature_schema(schema_version: str, feature_names: Iterable[str]) -> str:
    names = tuple(feature_names)
    output = "ml_feature_schema_v1\n"
    output += _text("schema_version", schema_version)
    output += f"feature_count={len(names)}\n"
    for index, name in enumerate(names):
        output += _text(f"feature[{index}]", name)
    return output


def canonical_artifact(artifact: LinearLogitArtifact) -> str:
    output = "linear_logit_artifact_canonical_v1\n"
    output += _text("artifact_version", artifact.artifact_version)
    output += _text("model_version", artifact.model_version)
    output += _text("task", artifact.task)
    output += _text("prior_kind", artifact.prior_kind)
    output += _text("feature_schema_version", artifact.feature_schema_version)
    output += _text("feature_schema_sha256", artifact.feature_schema_sha256)
    output += _optional_text("dataset_manifest_sha256", artifact.dataset_manifest_sha256)
    output += _optional_i64("trained_through_ms", artifact.trained_through_ms)
    output += _float("intercept", artifact.intercept)
    output += f"feature_count={len(artifact.features)}\n"
    for index, feature in enumerate(artifact.features):
        output += _text(f"feature[{index}].name", feature.name)
        output += _float(f"feature[{index}].mean", feature.mean)
        output += _float(f"feature[{index}].scale", feature.scale)
        output += _float(f"feature[{index}].coefficient", feature.coefficient)
    return output


def _text(key: str, value: str) -> str:
    return f"{key}.utf8_bytes={len(value.encode('utf-8'))}:{value}\n"


def _float(key: str, value: float) -> str:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return f"{key}.f64_bits={bits:016x}\n"


def _optional_text(key: str, value: str | None) -> str:
    if value is None:
        return f"{key}.present=0\n"
    return f"{key}.present=1\n" + _text(key, value)


def _optional_i64(key: str, value: int | None) -> str:
    if value is None:
        return f"{key}.present=0\n"
    return f"{key}.present=1\n{key}.value={value}\n"


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)
