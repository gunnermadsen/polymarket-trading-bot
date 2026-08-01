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
        return min(self.max_parallel_fits, max(0, detected - self.reserve_cores))


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


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    horizon_bars: int
    model: str
    feature_set: str


@dataclass(frozen=True)
class ExpectancySelectionConfig:
    expected_return_hurdles_bps: tuple[float, ...]
    directional_advantages_bps: tuple[float, ...]
    minimum_threshold_trades: int


@dataclass(frozen=True)
class ExpectancyGateConfig:
    minimum_development_positive_folds: int
    minimum_net_expectancy_bps: float
    minimum_profit_factor: float
    minimum_positive_month_fraction: float
    maximum_positive_fold_pnl_fraction: float
    execution_cost_stress_multiplier: float
    minimum_oi_paired_wins: int
    minimum_holdout_trades: int


@dataclass(frozen=True)
class FeeConfig:
    taker_bps_per_side: float
    maker_bps_per_side: float


@dataclass(frozen=True)
class FundingProvenanceConfig:
    import_id: str


@dataclass(frozen=True)
class ExpectancyConfig:
    source_path: Path
    dataset: DatasetConfig
    validation: ValidationConfig
    compute: ComputeConfig
    candidates: tuple[CandidateConfig, ...]
    selection: ExpectancySelectionConfig
    gates: ExpectancyGateConfig
    fees: FeeConfig
    funding_provenance: FundingProvenanceConfig
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


def _dataset_config(raw: dict[str, Any]) -> DatasetConfig:
    dataset_raw = dict(raw["dataset"])
    dataset_raw["lake_root"] = Path(dataset_raw["lake_root"])
    dataset_raw["start"] = parse_utc(dataset_raw["start"])
    dataset_raw["end"] = parse_utc(dataset_raw["end"])
    return DatasetConfig(**dataset_raw)


def _validation_config(raw: dict[str, Any]) -> ValidationConfig:
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
    return ValidationConfig(folds=folds, **validation_raw)


def load_expectancy_config(path: str | Path) -> ExpectancyConfig:
    source_path = Path(path).resolve()
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    artifact_raw = raw["artifacts"]
    config = ExpectancyConfig(
        source_path=source_path,
        dataset=_dataset_config(raw),
        validation=_validation_config(raw),
        compute=ComputeConfig(**raw["compute"]),
        candidates=tuple(CandidateConfig(**candidate) for candidate in raw["candidates"]),
        selection=ExpectancySelectionConfig(
            expected_return_hurdles_bps=tuple(
                raw["selection"]["expected_return_hurdles_bps"]
            ),
            directional_advantages_bps=tuple(
                raw["selection"]["directional_advantages_bps"]
            ),
            minimum_threshold_trades=raw["selection"]["minimum_threshold_trades"],
        ),
        gates=ExpectancyGateConfig(**raw["gates"]),
        fees=FeeConfig(**raw["fees"]),
        funding_provenance=FundingProvenanceConfig(**raw["funding_provenance"]),
        artifacts=ArtifactConfig(
            root=Path(artifact_raw["root"]),
            curated_report_directory=Path(artifact_raw["curated_report_directory"]),
        ),
    )
    validate_expectancy_config(config)
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


EXPECTED_EXPECTANCY_CANDIDATES = {
    "h1_ridge_price": (4, "ridge", "price"),
    "h1_histogram_price": (4, "histogram", "price"),
    "h1_extra_trees_price": (4, "extra_trees", "price"),
    "h4_ridge_price": (16, "ridge", "price"),
    "h4_histogram_price": (16, "histogram", "price"),
    "h4_extra_trees_price": (16, "extra_trees", "price"),
    "h4_extra_trees_oi": (16, "extra_trees", "oi"),
    "h4_extra_trees_price_oi": (16, "extra_trees", "price_oi"),
}


def validate_expectancy_config(config: ExpectancyConfig) -> None:
    if config.dataset.source != "parquet_lake":
        raise ValueError("the expectancy benchmark requires source='parquet_lake'")
    funding_import_id = config.funding_provenance.import_id
    if (
        len(funding_import_id) != 64
        or any(character not in "0123456789abcdef" for character in funding_import_id)
    ):
        raise ValueError("funding provenance import id must be a lowercase SHA-256")
    if config.dataset.interval_seconds != 900:
        raise ValueError("the expectancy benchmark requires 15-minute source buckets")
    if config.dataset.horizon_bars != 16:
        raise ValueError("dataset.horizon_bars must equal the maximum candidate horizon (16)")
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

    registry: dict[str, tuple[int, str, str]] = {}
    for candidate in config.candidates:
        if candidate.candidate_id in registry:
            raise ValueError(f"duplicate candidate id: {candidate.candidate_id}")
        registry[candidate.candidate_id] = (
            candidate.horizon_bars,
            candidate.model,
            candidate.feature_set,
        )
    if registry != EXPECTED_EXPECTANCY_CANDIDATES:
        raise ValueError("expectancy candidate registry differs from the frozen eight candidates")

    maximum_horizon = max(candidate.horizon_bars for candidate in config.candidates)
    if config.validation.purge_bars < maximum_horizon + 1:
        raise ValueError(
            f"purge_bars must be at least {maximum_horizon + 1} for next-open labels"
        )
    previous_end: datetime | None = None
    for fold in config.validation.folds:
        if fold.start >= fold.end:
            raise ValueError(f"invalid fold {fold.name}")
        if previous_end is not None and fold.start != previous_end:
            raise ValueError("development folds must be contiguous")
        previous_end = fold.end

    if config.selection.expected_return_hurdles_bps != (0.0, 3.0, 6.0, 10.0):
        raise ValueError("expected-return hurdles differ from the frozen grid")
    if config.selection.directional_advantages_bps != (0.0, 3.0, 6.0):
        raise ValueError("directional advantages differ from the frozen grid")
    if config.selection.minimum_threshold_trades != 50:
        raise ValueError("minimum threshold trades must remain 50")
    if config.dataset.taker_fee_bps_per_side != config.fees.taker_bps_per_side:
        raise ValueError("dataset and fee configuration disagree on the taker fee")
    if config.fees.taker_bps_per_side != 5.0 or config.fees.maker_bps_per_side != 2.0:
        raise ValueError("expectancy fee assumptions differ from the frozen schedule")
    if config.gates.minimum_development_positive_folds != 5:
        raise ValueError("development stability requires five positive folds")
    if config.gates.minimum_oi_paired_wins != 5:
        raise ValueError("OI qualification requires five paired fold wins")
    if config.gates.minimum_net_expectancy_bps != 3.0:
        raise ValueError("minimum net expectancy must remain 3 bps/trade")
    if config.gates.minimum_profit_factor != 1.15:
        raise ValueError("minimum profit factor must remain 1.15")
    if config.gates.minimum_positive_month_fraction != 0.60:
        raise ValueError("minimum positive-month fraction must remain 60%")
    if config.gates.maximum_positive_fold_pnl_fraction != 0.40:
        raise ValueError("maximum positive fold P&L concentration must remain 40%")
    if config.gates.execution_cost_stress_multiplier != 1.5:
        raise ValueError("execution-cost stress multiplier must remain 1.5")
    if config.compute.reserve_cores < 1:
        raise ValueError("at least one CPU core must remain reserved")
    if config.compute.available_cores < 1:
        raise ValueError("no CPU cores remain for training")
