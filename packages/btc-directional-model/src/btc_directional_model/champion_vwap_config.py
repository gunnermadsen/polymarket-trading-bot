from __future__ import annotations

import math
import tomllib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .core_extract import file_sha256


@dataclass(frozen=True)
class ChampionVwapFoldConfig:
    name: str
    fit_end: datetime
    evaluation_end: datetime


@dataclass(frozen=True)
class ChampionVwapDataConfig:
    source_range_start: datetime
    source_range_end: datetime
    order_book_coverage_start: datetime
    out_of_sample_score_start: datetime
    calibration_end: datetime
    holdout_end: datetime


@dataclass(frozen=True)
class ChampionVwapModelConfig:
    quantity: float
    confidence_threshold: float
    minimum_seconds_after_open: int
    maximum_seconds_after_open: int
    cadence_seconds: int
    logistic_c: float
    maximum_iterations: int


@dataclass(frozen=True)
class ChampionVwapPromotionConfig:
    accuracy_tolerance: float
    minimum_champion_coverage: float
    minimum_improving_folds: int
    minimum_calibration_rows: int
    minimum_holdout_rows: int


@dataclass(frozen=True)
class ChampionVwapPathConfig:
    features: Path
    feature_metadata: Path
    execution_evidence: Path
    champion_model: Path
    champion_manifest: Path
    runs: Path


@dataclass(frozen=True)
class ChampionVwapBenchmarkConfig:
    source_path: Path
    package_root: Path
    source_process_id: uuid.UUID
    champion_model_key: str
    champion_model_sha256: str
    champion_feature_schema_sha256: str
    evaluation_note: str
    data: ChampionVwapDataConfig
    model: ChampionVwapModelConfig
    promotion: ChampionVwapPromotionConfig
    folds: tuple[ChampionVwapFoldConfig, ...]
    paths: ChampionVwapPathConfig


def load_champion_vwap_config(path: Path) -> ChampionVwapBenchmarkConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    benchmark = raw["benchmark"]
    data = raw["data"]
    model = raw["model"]
    promotion = raw["promotion"]
    paths = raw["paths"]
    config = ChampionVwapBenchmarkConfig(
        source_path=source_path,
        package_root=package_root,
        source_process_id=uuid.UUID(str(benchmark["source_process_id"])),
        champion_model_key=str(benchmark["champion_model_key"]),
        champion_model_sha256=str(benchmark["champion_model_sha256"]),
        champion_feature_schema_sha256=str(
            benchmark["champion_feature_schema_sha256"]
        ),
        evaluation_note=str(benchmark["evaluation_note"]).strip(),
        data=ChampionVwapDataConfig(
            source_range_start=_parse_utc(data["source_range_start"]),
            source_range_end=_parse_utc(data["source_range_end"]),
            order_book_coverage_start=_parse_utc(data["order_book_coverage_start"]),
            out_of_sample_score_start=_parse_utc(data["out_of_sample_score_start"]),
            calibration_end=_parse_utc(data["calibration_end"]),
            holdout_end=_parse_utc(data["holdout_end"]),
        ),
        model=ChampionVwapModelConfig(
            quantity=float(model["quantity"]),
            confidence_threshold=float(model["confidence_threshold"]),
            minimum_seconds_after_open=int(model["minimum_seconds_after_open"]),
            maximum_seconds_after_open=int(model["maximum_seconds_after_open"]),
            cadence_seconds=int(model["cadence_seconds"]),
            logistic_c=float(model["logistic_c"]),
            maximum_iterations=int(model["maximum_iterations"]),
        ),
        promotion=ChampionVwapPromotionConfig(
            accuracy_tolerance=float(promotion["accuracy_tolerance"]),
            minimum_champion_coverage=float(
                promotion["minimum_champion_coverage"]
            ),
            minimum_improving_folds=int(promotion["minimum_improving_folds"]),
            minimum_calibration_rows=int(promotion["minimum_calibration_rows"]),
            minimum_holdout_rows=int(promotion["minimum_holdout_rows"]),
        ),
        folds=tuple(
            ChampionVwapFoldConfig(
                name=str(fold["name"]),
                fit_end=_parse_utc(fold["fit_end"]),
                evaluation_end=_parse_utc(fold["evaluation_end"]),
            )
            for fold in raw["folds"]
        ),
        paths=ChampionVwapPathConfig(
            features=package_root / str(paths["features"]),
            feature_metadata=package_root / str(paths["feature_metadata"]),
            execution_evidence=package_root / str(paths["execution_evidence"]),
            champion_model=package_root / str(paths["champion_model"]),
            champion_manifest=package_root / str(paths["champion_manifest"]),
            runs=package_root / str(paths["runs"]),
        ),
    )
    validate_champion_vwap_config(config)
    return config


def validate_champion_vwap_config(config: ChampionVwapBenchmarkConfig) -> None:
    if config.source_process_id.int == 0:
        raise ValueError("source_process_id must not be nil")
    if not config.evaluation_note:
        raise ValueError("evaluation_note must be non-empty")
    _validate_sha("champion_model_sha256", config.champion_model_sha256)
    _validate_sha(
        "champion_feature_schema_sha256",
        config.champion_feature_schema_sha256,
    )
    if not config.paths.champion_model.is_file():
        raise ValueError("champion model is missing")
    if not config.paths.champion_manifest.is_file():
        raise ValueError("champion manifest is missing")
    if file_sha256(config.paths.champion_model) != config.champion_model_sha256:
        raise ValueError("pinned champion model hash mismatch")

    data = config.data
    boundaries = (
        data.source_range_start,
        data.order_book_coverage_start,
        data.out_of_sample_score_start,
        data.calibration_end,
        data.holdout_end,
        data.source_range_end,
    )
    if boundaries != tuple(sorted(boundaries)):
        raise ValueError("data boundaries must remain chronological")
    if data.holdout_end != data.source_range_end:
        raise ValueError("holdout end must preserve the complete source range")

    model = config.model
    if not math.isclose(model.quantity, 5.0):
        raise ValueError("the decision benchmark is fixed to five shares")
    if not 0.5 < model.confidence_threshold < 1.0:
        raise ValueError("confidence_threshold must be inside (0.5, 1)")
    if (
        model.minimum_seconds_after_open != 60
        or model.maximum_seconds_after_open != 240
        or model.cadence_seconds != 5
    ):
        raise ValueError("champion first-crossing timing must remain 60-240s at 5s")
    if not math.isfinite(model.logistic_c) or model.logistic_c <= 0:
        raise ValueError("logistic_c must be positive and finite")
    if model.maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")

    promotion = config.promotion
    if not 0 <= promotion.accuracy_tolerance <= 1:
        raise ValueError("accuracy_tolerance must be in [0, 1]")
    if not 0 < promotion.minimum_champion_coverage <= 1:
        raise ValueError("minimum_champion_coverage must be in (0, 1]")
    if not 1 <= promotion.minimum_improving_folds <= len(config.folds):
        raise ValueError("minimum_improving_folds must fit the configured folds")
    if promotion.minimum_calibration_rows <= 0 or promotion.minimum_holdout_rows <= 0:
        raise ValueError("minimum row gates must be positive")

    previous_end = data.out_of_sample_score_start
    names: set[str] = set()
    for fold in config.folds:
        if not fold.name or fold.name in names:
            raise ValueError("fold names must be non-empty and unique")
        names.add(fold.name)
        if not previous_end <= fold.fit_end < fold.evaluation_end <= data.calibration_end:
            raise ValueError("fold ranges must be expanding and pre-holdout")
        previous_end = fold.evaluation_end


def _parse_utc(value: str | datetime) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(value)
    )
    if parsed.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def _validate_sha(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal digits")
