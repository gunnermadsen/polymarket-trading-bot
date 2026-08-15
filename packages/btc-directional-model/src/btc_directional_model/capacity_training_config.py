from __future__ import annotations

import tomllib
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class CapacityWindows:
    development_start: datetime
    calibration_start: datetime
    policy_start: datetime
    freeze_at: datetime


@dataclass(frozen=True)
class CapacityExecution:
    quantities: tuple[int, ...]
    freshness_seconds: int
    maximum_depth_participation: float
    execution_reserve_per_share: float
    confidence_threshold: float


@dataclass(frozen=True)
class CapacityGates:
    minimum_training_rows: int
    minimum_policy_trades: int
    minimum_profit_factor: float
    minimum_stress_expectancy_per_trade: float
    minimum_improving_folds: int


@dataclass(frozen=True)
class CapacityLineage:
    name: str
    hypothesis: str
    process_id: uuid.UUID
    model: Path
    manifest: Path
    features: tuple[Path, ...]


@dataclass(frozen=True)
class CapacityTrainingConfig:
    source_path: Path
    package_root: Path
    windows: CapacityWindows
    execution: CapacityExecution
    gates: CapacityGates
    evidence: Path
    runs: Path
    lineages: tuple[CapacityLineage, ...]


def load_capacity_training_config(path: Path) -> CapacityTrainingConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    windows = raw["windows"]
    execution = raw["execution"]
    gates = raw["gates"]
    paths = raw["paths"]
    config = CapacityTrainingConfig(
        source_path=source_path,
        package_root=package_root,
        windows=CapacityWindows(
            development_start=_utc(windows["development_start"]),
            calibration_start=_utc(windows["calibration_start"]),
            policy_start=_utc(windows["policy_start"]),
            freeze_at=_utc(windows["freeze_at"]),
        ),
        execution=CapacityExecution(
            quantities=tuple(int(value) for value in execution["quantities"]),
            freshness_seconds=int(execution["freshness_seconds"]),
            maximum_depth_participation=float(
                execution["maximum_depth_participation"]
            ),
            execution_reserve_per_share=float(
                execution["execution_reserve_per_share"]
            ),
            confidence_threshold=float(execution["confidence_threshold"]),
        ),
        gates=CapacityGates(
            minimum_training_rows=int(gates["minimum_training_rows"]),
            minimum_policy_trades=int(gates["minimum_policy_trades"]),
            minimum_profit_factor=float(gates["minimum_profit_factor"]),
            minimum_stress_expectancy_per_trade=float(
                gates["minimum_stress_expectancy_per_trade"]
            ),
            minimum_improving_folds=int(gates["minimum_improving_folds"]),
        ),
        evidence=_path(package_root, paths["evidence"]),
        runs=_path(package_root, paths["runs"]),
        lineages=tuple(
            CapacityLineage(
                name=str(lineage["name"]),
                hypothesis=str(lineage["hypothesis"]),
                process_id=uuid.UUID(str(lineage["process_id"])),
                model=_path(package_root, lineage["model"]),
                manifest=_path(package_root, lineage["manifest"]),
                features=tuple(
                    _path(package_root, value) for value in lineage["features"]
                ),
            )
            for lineage in raw["lineages"]
        ),
    )
    _validate(config)
    return config


def _validate(config: CapacityTrainingConfig) -> None:
    ordered = (
        config.windows.development_start,
        config.windows.calibration_start,
        config.windows.policy_start,
        config.windows.freeze_at,
    )
    if any(value.tzinfo is None or value.utcoffset() != timedelta(0) for value in ordered):
        raise ValueError("capacity training windows must be UTC")
    if list(ordered) != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise ValueError("capacity training windows must be strictly chronological")
    if config.execution.quantities != (10, 15, 20):
        raise ValueError("capacity training is fixed to VWAP 10/15/20")
    if config.execution.freshness_seconds <= 0:
        raise ValueError("capacity freshness must be positive")
    if config.execution.maximum_depth_participation != 0.25:
        raise ValueError("capacity training is fixed to 25% maximum depth participation")
    if config.execution.execution_reserve_per_share < 0:
        raise ValueError("capacity execution reserve must be nonnegative")
    if not 0.5 < config.execution.confidence_threshold < 1:
        raise ValueError("capacity confidence threshold must be inside (0.5, 1)")
    if not 1 <= config.gates.minimum_improving_folds <= 5:
        raise ValueError("minimum_improving_folds must be inside [1, 5]")
    if len(config.lineages) != 4:
        raise ValueError("capacity training requires exactly four frozen lineages")
    if sum(lineage.hypothesis == "asymmetric_value" for lineage in config.lineages) != 1:
        raise ValueError("capacity training requires one asymmetric-value lineage")
    if any(
        lineage.hypothesis not in {"directional", "asymmetric_value"}
        for lineage in config.lineages
    ):
        raise ValueError("unsupported capacity hypothesis")
    names = [lineage.name for lineage in config.lineages]
    if len(set(names)) != len(names):
        raise ValueError("capacity lineage names must be unique")
    if any(lineage.process_id.int == 0 for lineage in config.lineages):
        raise ValueError("capacity process_id values must not be nil")
    for lineage in config.lineages:
        for label, artifact in (("model", lineage.model), ("manifest", lineage.manifest)):
            if not artifact.is_file():
                raise ValueError(f"{lineage.name} {label} is missing: {artifact}")


def _utc(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("timestamp must use UTC")
    return parsed


def _path(root: Path, value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path
