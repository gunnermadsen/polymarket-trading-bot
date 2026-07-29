from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    range_start: datetime
    range_end: datetime
    sample_interval_seconds: int
    min_seconds_after_open: int
    min_seconds_before_close: int
    strict_final_price_audit: bool


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float
    calibration_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    c_candidates: tuple[float, ...]
    confidence_min: float
    confidence_max: float
    confidence_step: float
    target_accuracy: float
    target_wilson_lower: float
    minimum_calibration_markets: int
    minimum_test_markets: int
    maximum_train_test_accuracy_gap: float
    random_seed: int


@dataclass(frozen=True)
class EvaluationConfig:
    holdout_is_independent: bool
    holdout_note: str


@dataclass(frozen=True)
class PathConfig:
    source_data: Path
    feature_data: Path
    runs: Path
    artifacts: Path


@dataclass(frozen=True)
class TrainingConfig:
    source_path: Path
    package_root: Path
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    paths: PathConfig


def load_config(path: Path) -> TrainingConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    data = raw["data"]
    split = raw["split"]
    model = raw["model"]
    evaluation = raw["evaluation"]
    paths = raw["paths"]

    range_start = parse_utc(data["range_start"])
    range_end = parse_utc(data["range_end"])
    if range_end <= range_start:
        raise ValueError("data.range_end must be later than data.range_start")
    if any(value.time() != time() for value in (range_start, range_end)):
        raise ValueError("data range boundaries must align to UTC days")
    if data["sample_interval_seconds"] <= 0:
        raise ValueError("data.sample_interval_seconds must be positive")
    if data["min_seconds_after_open"] < 0 or data["min_seconds_before_close"] <= 0:
        raise ValueError("entry-window boundaries are invalid")
    if data["min_seconds_after_open"] + data["min_seconds_before_close"] >= 300:
        raise ValueError("entry-window boundaries leave no candidate prediction time")
    if (300 - data["min_seconds_before_close"] - data["min_seconds_after_open"]) % data[
        "sample_interval_seconds"
    ]:
        raise ValueError("sample cadence must evenly divide the configured prediction window")

    fractions = (
        float(split["train_fraction"]),
        float(split["calibration_fraction"]),
        float(split["test_fraction"]),
    )
    if abs(sum(fractions) - 1.0) > 1e-9 or any(value <= 0 for value in fractions):
        raise ValueError("split fractions must be positive and sum to one")

    c_candidates = tuple(float(value) for value in model["c_candidates"])
    if not c_candidates or any(value <= 0 for value in c_candidates):
        raise ValueError("model.c_candidates must contain positive values")
    confidence_min = float(model["confidence_min"])
    confidence_max = float(model["confidence_max"])
    confidence_step = float(model["confidence_step"])
    if not 0.5 <= confidence_min <= confidence_max < 1:
        raise ValueError("confidence bounds must satisfy 0.5 <= min <= max < 1")
    if confidence_step <= 0:
        raise ValueError("model.confidence_step must be positive")
    target_accuracy = float(model["target_accuracy"])
    target_wilson_lower = float(model["target_wilson_lower"])
    if not 0.5 <= target_accuracy <= 1 or not 0 <= target_wilson_lower <= 1:
        raise ValueError("accuracy acceptance thresholds are invalid")
    minimum_calibration_markets = int(model["minimum_calibration_markets"])
    minimum_test_markets = int(model["minimum_test_markets"])
    if minimum_calibration_markets <= 0 or minimum_test_markets <= 0:
        raise ValueError("minimum market counts must be positive")
    maximum_train_test_accuracy_gap = float(model["maximum_train_test_accuracy_gap"])
    if not 0 <= maximum_train_test_accuracy_gap <= 1:
        raise ValueError("maximum train/test accuracy gap must be between zero and one")
    holdout_is_independent = bool(evaluation["holdout_is_independent"])
    holdout_note = str(evaluation["holdout_note"]).strip()
    if not holdout_note:
        raise ValueError("evaluation.holdout_note must explain holdout status")

    return TrainingConfig(
        source_path=source_path,
        package_root=package_root,
        data=DataConfig(
            range_start=range_start,
            range_end=range_end,
            sample_interval_seconds=int(data["sample_interval_seconds"]),
            min_seconds_after_open=int(data["min_seconds_after_open"]),
            min_seconds_before_close=int(data["min_seconds_before_close"]),
            strict_final_price_audit=bool(data["strict_final_price_audit"]),
        ),
        split=SplitConfig(*fractions),
        model=ModelConfig(
            c_candidates=c_candidates,
            confidence_min=confidence_min,
            confidence_max=confidence_max,
            confidence_step=confidence_step,
            target_accuracy=target_accuracy,
            target_wilson_lower=target_wilson_lower,
            minimum_calibration_markets=minimum_calibration_markets,
            minimum_test_markets=minimum_test_markets,
            maximum_train_test_accuracy_gap=maximum_train_test_accuracy_gap,
            random_seed=int(model["random_seed"]),
        ),
        evaluation=EvaluationConfig(
            holdout_is_independent=holdout_is_independent,
            holdout_note=holdout_note,
        ),
        paths=PathConfig(
            source_data=package_root / paths["source_data"],
            feature_data=package_root / paths["feature_data"],
            runs=package_root / paths["runs"],
            artifacts=package_root / paths["artifacts"],
        ),
    )


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include an offset: {value}")
    return parsed.astimezone(UTC)
