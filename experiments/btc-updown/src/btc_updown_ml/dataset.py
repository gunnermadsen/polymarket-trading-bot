from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class FeatureObservation:
    name: str
    value: float
    source_event_at_ms: int
    source_received_at_ms: int

    def validate(self, *, feature_as_of_ms: int, feature_received_at_ms: int) -> None:
        if not self.name.strip():
            raise ValueError("feature name must not be empty")
        if not _is_finite(self.value):
            raise ValueError(f"feature {self.name!r} must be finite")
        if self.source_event_at_ms > feature_as_of_ms:
            raise ValueError(
                f"feature {self.name!r} source event is after feature_as_of_ms"
            )
        if self.source_received_at_ms > feature_received_at_ms:
            raise ValueError(
                f"feature {self.name!r} was received after feature_received_at_ms"
            )


@dataclass(frozen=True)
class SnapshotRow:
    snapshot_id: str
    market_id: str
    feature_as_of_ms: int
    feature_received_at_ms: int
    label_available_at_ms: int
    label: int
    task: str
    label_version: str
    prior_probability: float
    schema_version: str
    features: tuple[FeatureObservation, ...]

    def validate(self) -> None:
        for field, value in (
            ("snapshot_id", self.snapshot_id),
            ("market_id", self.market_id),
            ("schema_version", self.schema_version),
            ("task", self.task),
            ("label_version", self.label_version),
        ):
            if not value.strip():
                raise ValueError(f"{field} must not be empty")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        if self.task not in {
            "settlement_probability_residual",
            "fok_fill_probability",
            "fok_post_fill_toxicity_probability",
        }:
            raise ValueError(f"unsupported task {self.task!r}")
        if self.feature_received_at_ms > self.feature_as_of_ms:
            raise ValueError("feature receipt cutoff must not follow decision cutoff")
        if (
            self.feature_as_of_ms >= self.label_available_at_ms
            or self.feature_received_at_ms >= self.label_available_at_ms
        ):
            raise ValueError("features must be fixed before the label is available")
        if not _is_finite(self.prior_probability) or not 0.0 <= self.prior_probability <= 1.0:
            raise ValueError("prior_probability must be finite and in [0, 1]")
        if not self.features:
            raise ValueError("features must not be empty")
        seen: set[str] = set()
        for feature in self.features:
            feature.validate(
                feature_as_of_ms=self.feature_as_of_ms,
                feature_received_at_ms=self.feature_received_at_ms,
            )
            if feature.name in seen:
                raise ValueError(f"duplicate feature {feature.name!r}")
            seen.add(feature.name)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @property
    def feature_values(self) -> tuple[float, ...]:
        return tuple(feature.value for feature in self.features)

    def eligible_for_training(self, training_cutoff_ms: int) -> bool:
        return self.label_available_at_ms <= training_cutoff_ms

    def to_mapping(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "market_id": self.market_id,
            "feature_as_of_ms": self.feature_as_of_ms,
            "feature_received_at_ms": self.feature_received_at_ms,
            "label_available_at_ms": self.label_available_at_ms,
            "label": self.label,
            "task": self.task,
            "label_version": self.label_version,
            "prior_probability": self.prior_probability,
            "schema_version": self.schema_version,
            "features": [asdict(feature) for feature in self.features],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SnapshotRow":
        row = cls(
            snapshot_id=str(value["snapshot_id"]),
            market_id=str(value["market_id"]),
            feature_as_of_ms=int(value["feature_as_of_ms"]),
            feature_received_at_ms=int(value["feature_received_at_ms"]),
            label_available_at_ms=int(value["label_available_at_ms"]),
            label=int(value["label"]),
            task=str(value["task"]),
            label_version=str(value["label_version"]),
            prior_probability=float(value["prior_probability"]),
            schema_version=str(value["schema_version"]),
            features=tuple(
                FeatureObservation(
                    name=str(feature["name"]),
                    value=float(feature["value"]),
                    source_event_at_ms=int(feature["source_event_at_ms"]),
                    source_received_at_ms=int(feature["source_received_at_ms"]),
                )
                for feature in value["features"]
            ),
        )
        row.validate()
        return row


@dataclass(frozen=True)
class DatasetManifest:
    manifest_version: str
    task: str
    schema_version: str
    feature_names: tuple[str, ...]
    label_version: str
    source_lineage_version: str
    exclusion_policy: str
    training_cutoff_ms: int
    row_count: int
    market_count: int
    min_feature_as_of_ms: int
    max_feature_as_of_ms: int
    dataset_sha256: str
    manifest_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["feature_names"] = list(self.feature_names)
        return value


def build_dataset_manifest(
    rows: Sequence[SnapshotRow], *, training_cutoff_ms: int
) -> DatasetManifest:
    if not rows:
        raise ValueError("cannot build a manifest for an empty dataset")
    for row in rows:
        row.validate()
        if not row.eligible_for_training(training_cutoff_ms):
            raise ValueError(
                f"label for snapshot {row.snapshot_id!r} was unavailable at training cutoff"
            )
    schema_versions = {row.schema_version for row in rows}
    tasks = {row.task for row in rows}
    label_versions = {row.label_version for row in rows}
    feature_orders = {row.feature_names for row in rows}
    if (
        len(schema_versions) != 1
        or len(tasks) != 1
        or len(label_versions) != 1
        or len(feature_orders) != 1
    ):
        raise ValueError(
            "all rows in a manifest must share one task, label, schema, and feature order"
        )

    canonical_rows = sorted(
        (row.to_mapping() for row in rows),
        key=lambda row: (row["feature_as_of_ms"], row["market_id"], row["snapshot_id"]),
    )
    canonical = json.dumps(
        canonical_rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    manifest_fields = dict(
        manifest_version="btc_updown_dataset_manifest_v1",
        task=next(iter(tasks)),
        schema_version=next(iter(schema_versions)),
        feature_names=next(iter(feature_orders)),
        label_version=next(iter(label_versions)),
        source_lineage_version="btc_realtime_lineage_v1",
        exclusion_policy="label_available_at_lte_training_cutoff_and_features_strictly_prelabel_v1",
        training_cutoff_ms=training_cutoff_ms,
        row_count=len(rows),
        market_count=len({row.market_id for row in rows}),
        min_feature_as_of_ms=min(row.feature_as_of_ms for row in rows),
        max_feature_as_of_ms=max(row.feature_as_of_ms for row in rows),
        dataset_sha256=hashlib.sha256(canonical).hexdigest(),
    )
    canonical_manifest = dict(manifest_fields)
    canonical_manifest["feature_names"] = list(canonical_manifest["feature_names"])
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            canonical_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return DatasetManifest(**manifest_fields, manifest_sha256=manifest_sha256)


def canonical_feature_vector(row: SnapshotRow) -> str:
    """Canonical representation shared with Rust's `MlFeatureVector`."""
    row.validate()
    output = "ml_feature_vector_v1\n"
    output += _text("snapshot_id", row.snapshot_id)
    output += _text("market_id", row.market_id)
    output += f"feature_as_of_ms={row.feature_as_of_ms}\n"
    output += f"feature_received_at_ms={row.feature_received_at_ms}\n"
    output += _text("schema_version", row.schema_version)
    output += _float("prior_probability", row.prior_probability)
    output += f"feature_count={len(row.features)}\n"
    for index, feature in enumerate(row.features):
        output += _text(f"feature[{index}].name", feature.name)
        output += _float(f"feature[{index}].value", feature.value)
        output += f"feature[{index}].source_event_at_ms={feature.source_event_at_ms}\n"
        output += f"feature[{index}].source_received_at_ms={feature.source_received_at_ms}\n"
    return output


def feature_vector_sha256(row: SnapshotRow) -> str:
    return hashlib.sha256(canonical_feature_vector(row).encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path) -> list[SnapshotRow]:
    rows: list[SnapshotRow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(SnapshotRow.from_mapping(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid dataset row at line {line_number}: {error}") from error
    return rows


def write_jsonl(path: str | Path, rows: Iterable[SnapshotRow]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            row.validate()
            handle.write(
                json.dumps(row.to_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False)
            )
            handle.write("\n")


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _text(key: str, value: str) -> str:
    return f"{key}.utf8_bytes={len(value.encode('utf-8'))}:{value}\n"


def _float(key: str, value: float) -> str:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return f"{key}.f64_bits={bits:016x}\n"
