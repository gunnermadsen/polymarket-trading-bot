from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include a timezone: {value}")
    return parsed


@dataclass(frozen=True)
class DatasetConfig:
    source: str
    lake_root: Path
    symbol: str
    interval_seconds: int
    start: datetime
    end: datetime
    horizon_bars: int
    reference_notional_usd: int
    taker_fee_bps_per_side: float
    label_uncertainty_buffer_bps: float


@dataclass(frozen=True)
class FoldConfig:
    name: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ValidationConfig:
    minimum_training_days: int
    purge_bars: int
    calibration_start: datetime
    threshold_start: datetime
    holdout_start: datetime
    holdout_end: datetime
    folds: tuple[FoldConfig, ...]


@dataclass(frozen=True)
class ComputeConfig:
    reserve_cores: int
    max_parallel_fits: int
    comparison_estimator_threads: int
    final_refit_threads: int
    random_seed: int
    bootstrap_resamples: int
    holdout_bootstrap_resamples: int

    @property
    def available_cores(self) -> int:
        detected = os.cpu_count() or 1
        return max(1, min(self.max_parallel_fits, detected - self.reserve_cores))


@dataclass(frozen=True)
class SelectionConfig:
    probability_thresholds: tuple[float, ...]
    directional_margins: tuple[float, ...]
    minimum_calibration_trades: int


@dataclass(frozen=True)
class GateConfig:
    minimum_balanced_accuracy_uplift: float
    minimum_development_positive_folds: int
    minimum_holdout_trades: int
    minimum_net_expectancy_bps: float
    minimum_profit_factor: float
    minimum_positive_month_fraction: float
    execution_cost_stress_multiplier: float


@dataclass(frozen=True)
class ArtifactConfig:
    root: Path
    curated_report_directory: Path


@dataclass(frozen=True)
class BenchmarkConfig:
    source_path: Path
    dataset: DatasetConfig
    validation: ValidationConfig
    compute: ComputeConfig
    selection: SelectionConfig
    gates: GateConfig
    artifacts: ArtifactConfig

    def canonical_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("source_path", None)

        def normalize(value: Any) -> Any:
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            return value

        return normalize(payload)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path) -> BenchmarkConfig:
    source_path = Path(path).resolve()
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    dataset_raw = dict(raw["dataset"])
    dataset_raw["lake_root"] = Path(dataset_raw["lake_root"])
    dataset_raw["start"] = parse_utc(dataset_raw["start"])
    dataset_raw["end"] = parse_utc(dataset_raw["end"])
    dataset = DatasetConfig(**dataset_raw)

    validation_raw = dict(raw["validation"])
    folds = tuple(
        FoldConfig(
            name=fold["name"],
            start=parse_utc(fold["start"]),
            end=parse_utc(fold["end"]),
        )
        for fold in validation_raw.pop("folds")
    )
    for key in ("calibration_start", "threshold_start", "holdout_start", "holdout_end"):
        validation_raw[key] = parse_utc(validation_raw[key])
    validation = ValidationConfig(folds=folds, **validation_raw)

    compute = ComputeConfig(**raw["compute"])
    selection = SelectionConfig(
        probability_thresholds=tuple(raw["selection"]["probability_thresholds"]),
        directional_margins=tuple(raw["selection"]["directional_margins"]),
        minimum_calibration_trades=raw["selection"]["minimum_calibration_trades"],
    )
    gates = GateConfig(**raw["gates"])
    artifact_raw = raw["artifacts"]
    artifacts = ArtifactConfig(
        root=Path(artifact_raw["root"]),
        curated_report_directory=Path(artifact_raw["curated_report_directory"]),
    )
    config = BenchmarkConfig(
        source_path=source_path,
        dataset=dataset,
        validation=validation,
        compute=compute,
        selection=selection,
        gates=gates,
        artifacts=artifacts,
    )
    validate_config(config)
    return config


def validate_config(config: BenchmarkConfig) -> None:
    if config.dataset.interval_seconds <= 0 or config.dataset.horizon_bars <= 0:
        raise ValueError("dataset interval and horizon must be positive")
    if config.dataset.source != "parquet_lake":
        raise ValueError("the frozen benchmark requires source='parquet_lake'")
    if config.dataset.start >= config.validation.folds[0].start:
        raise ValueError("dataset must begin before the first validation fold")
    if config.validation.folds[-1].end != config.validation.calibration_start:
        raise ValueError("last development fold must end at calibration_start")
    if not (
        config.validation.calibration_start
        < config.validation.threshold_start
        < config.validation.holdout_start
        < config.validation.holdout_end
        <= config.dataset.end
    ):
        raise ValueError("calibration and holdout boundaries are inconsistent")
    minimum_purge = config.dataset.horizon_bars + 1
    if config.validation.purge_bars < minimum_purge:
        raise ValueError(f"purge_bars must be at least {minimum_purge} for next-open labels")
    previous_end: datetime | None = None
    for fold in config.validation.folds:
        if fold.start >= fold.end:
            raise ValueError(f"invalid fold {fold.name}")
        if previous_end is not None and fold.start != previous_end:
            raise ValueError("development folds must be contiguous")
        previous_end = fold.end
    if config.compute.reserve_cores < 1:
        raise ValueError("at least one CPU core must remain reserved")
    if config.compute.available_cores < 1:
        raise ValueError("no CPU cores remain for training")
