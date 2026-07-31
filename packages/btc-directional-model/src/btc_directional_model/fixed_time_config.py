from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from .core_config import CORE_ORACLE_SOURCE_CONTRACT, load_core_config, parse_utc_day
from .core_features import (
    CORE_MATURE_REVERSAL_ENRICHED_FEATURES,
    CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
)

FIXED_TIME_ACCURACY_PROFILE = "mature_reversal_fixed_120_accuracy"
FIXED_TIME_CANDIDATE = "histogram_mature_reversal_recency_28d"
FIXED_TIME_FEATURE_CONTRACT = "core_mature_reversal_71_v1"
FIXED_TIME_ESTIMATOR_FAMILY = "histogram_gradient_boosting"
FIXED_TIME_TARGET = "outcome_up"
FIXED_TIME_CALIBRATION = "global_platt"
EMPIRICAL_COVERAGE_THRESHOLD_SELECTION = "empirical_policy_confidence_quantile"
PRIMARY_OPERATING_POINT = "primary"
SECONDARY_OPERATING_POINT = "secondary"

EXPECTED_VALIDATION_WINDOWS = (
    (
        datetime(2026, 6, 9, tzinfo=UTC),
        datetime(2026, 6, 16, tzinfo=UTC),
    ),
    (
        datetime(2026, 6, 16, tzinfo=UTC),
        datetime(2026, 6, 23, tzinfo=UTC),
    ),
    (
        datetime(2026, 6, 23, tzinfo=UTC),
        datetime(2026, 6, 30, tzinfo=UTC),
    ),
    (
        datetime(2026, 6, 30, tzinfo=UTC),
        datetime(2026, 7, 7, tzinfo=UTC),
    ),
    (
        datetime(2026, 7, 7, tzinfo=UTC),
        datetime(2026, 7, 14, tzinfo=UTC),
    ),
)


@dataclass(frozen=True)
class FixedTimeIdentityConfig:
    profile: str
    candidate: str
    core_config: Path
    core_config_sha256: str
    evaluation_is_independent: bool
    evaluation_note: str


@dataclass(frozen=True)
class FixedTimeSplitConfig:
    development_start: datetime
    development_end: datetime
    probability_calibration_start: datetime
    probability_calibration_end: datetime
    policy_selection_start: datetime
    policy_selection_end: datetime
    validation_windows: tuple[tuple[datetime, datetime], ...]


@dataclass(frozen=True)
class FixedTimeModelConfig:
    decision_second: int
    estimator_training_seconds: tuple[int, ...]
    target: str
    estimator_family: str
    probability_calibration: str
    recency_half_life_days: float
    feature_contract: str
    feature_schema_version: str
    feature_names: tuple[str, ...]
    include_oracle: bool
    include_book: bool
    threshold_selection: str
    hard_confidence_floor: float
    require_hard_confident_error_no_regression: bool


@dataclass(frozen=True)
class FixedTimeOperatingPointConfig:
    name: str
    target_coverage: float
    coverage_tolerance: float
    minimum_accuracy: float
    minimum_balanced_accuracy: float
    minimum_direction_recall: float
    minimum_wilson_lower_95: float
    maximum_expected_calibration_error: float


@dataclass(frozen=True)
class FixedTimePathConfig:
    execution_evidence: Path
    execution_manifest_sha256: str
    predecessor_benchmark: Path
    predecessor_benchmark_sha256: str
    runs: Path
    freezes: Path
    runtime_models: Path


@dataclass(frozen=True)
class FixedTimeAccuracyConfig:
    source_path: Path
    package_root: Path
    benchmark: FixedTimeIdentityConfig
    split: FixedTimeSplitConfig
    model: FixedTimeModelConfig
    primary: FixedTimeOperatingPointConfig
    secondary: FixedTimeOperatingPointConfig
    paths: FixedTimePathConfig


def load_fixed_time_accuracy_config(path: Path) -> FixedTimeAccuracyConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    benchmark = raw["benchmark"]
    split = raw["split"]
    model = raw["model"]
    paths = raw["paths"]
    validation_starts = tuple(parse_utc_day(value) for value in split["validation_starts"])
    validation_ends = tuple(parse_utc_day(value) for value in split["validation_ends"])
    if len(validation_starts) != len(validation_ends):
        raise ValueError("validation start/end lists must have equal length")

    config = FixedTimeAccuracyConfig(
        source_path=source_path,
        package_root=package_root,
        benchmark=FixedTimeIdentityConfig(
            profile=str(benchmark["profile"]),
            candidate=str(benchmark["candidate"]),
            core_config=package_root / str(benchmark["core_config"]),
            core_config_sha256=str(benchmark["core_config_sha256"]).lower(),
            evaluation_is_independent=bool(benchmark["evaluation_is_independent"]),
            evaluation_note=str(benchmark["evaluation_note"]).strip(),
        ),
        split=FixedTimeSplitConfig(
            development_start=parse_utc_day(split["development_start"]),
            development_end=parse_utc_day(split["development_end"]),
            probability_calibration_start=parse_utc_day(split["probability_calibration_start"]),
            probability_calibration_end=parse_utc_day(split["probability_calibration_end"]),
            policy_selection_start=parse_utc_day(split["policy_selection_start"]),
            policy_selection_end=parse_utc_day(split["policy_selection_end"]),
            validation_windows=tuple(zip(validation_starts, validation_ends, strict=True)),
        ),
        model=FixedTimeModelConfig(
            decision_second=int(model["decision_second"]),
            estimator_training_seconds=tuple(
                int(value) for value in model["estimator_training_seconds"]
            ),
            target=str(model["target"]),
            estimator_family=str(model["estimator_family"]),
            probability_calibration=str(model["probability_calibration"]),
            recency_half_life_days=float(model["recency_half_life_days"]),
            feature_contract=str(model["feature_contract"]),
            feature_schema_version=str(model["feature_schema_version"]),
            feature_names=tuple(str(value) for value in model["feature_names"]),
            include_oracle=bool(model["include_oracle"]),
            include_book=bool(model["include_book"]),
            threshold_selection=str(model["threshold_selection"]),
            hard_confidence_floor=float(model["hard_confidence_floor"]),
            require_hard_confident_error_no_regression=bool(
                model["require_hard_confident_error_no_regression"]
            ),
        ),
        primary=_load_operating_point(
            PRIMARY_OPERATING_POINT,
            raw["primary_operating_point"],
        ),
        secondary=_load_operating_point(
            SECONDARY_OPERATING_POINT,
            raw["secondary_operating_point"],
        ),
        paths=FixedTimePathConfig(
            execution_evidence=package_root / str(paths["execution_evidence"]),
            execution_manifest_sha256=str(paths["execution_manifest_sha256"]).lower(),
            predecessor_benchmark=package_root / str(paths["predecessor_benchmark"]),
            predecessor_benchmark_sha256=str(paths["predecessor_benchmark_sha256"]).lower(),
            runs=package_root / str(paths["runs"]),
            freezes=package_root / str(paths["freezes"]),
            runtime_models=package_root / str(paths["runtime_models"]),
        ),
    )
    validate_fixed_time_accuracy_config(config)
    return config


def validate_fixed_time_accuracy_config(config: FixedTimeAccuracyConfig) -> None:
    benchmark = config.benchmark
    if benchmark.profile != FIXED_TIME_ACCURACY_PROFILE:
        raise ValueError("fixed-time profile identity changed")
    if benchmark.candidate != FIXED_TIME_CANDIDATE:
        raise ValueError("fixed-time mature-reversal candidate identity changed")
    if benchmark.evaluation_is_independent:
        raise ValueError("fixed-time validation is consumed development evidence")
    if not benchmark.evaluation_note:
        raise ValueError("fixed-time evaluation_note is required")
    if not benchmark.core_config.is_file():
        raise ValueError(f"fixed-time core config is missing: {benchmark.core_config}")
    _validate_checksum("core_config_sha256", benchmark.core_config_sha256)
    if _file_sha256(benchmark.core_config) != benchmark.core_config_sha256:
        raise ValueError("pinned core training config hash mismatch")

    expected_final_split = FixedTimeSplitConfig(
        development_start=datetime(2026, 3, 21, tzinfo=UTC),
        development_end=datetime(2026, 7, 14, tzinfo=UTC),
        probability_calibration_start=datetime(2026, 7, 14, tzinfo=UTC),
        probability_calibration_end=datetime(2026, 7, 21, tzinfo=UTC),
        policy_selection_start=datetime(2026, 7, 21, tzinfo=UTC),
        policy_selection_end=datetime(2026, 7, 29, tzinfo=UTC),
        validation_windows=EXPECTED_VALIDATION_WINDOWS,
    )
    if config.split != expected_final_split:
        raise ValueError(
            "fixed-time benchmark requires the frozen March 21–July 29 chronological split contract"
        )
    for window in config.split.validation_windows:
        if window[0] >= window[1]:
            raise ValueError("fixed-time validation windows must be positive")
    for previous, current in pairwise(config.split.validation_windows):
        if previous[1] != current[0]:
            raise ValueError("fixed-time validation windows must be contiguous and chronological")

    model = config.model
    if model.decision_second != 120:
        raise ValueError("fixed-time decision second must remain 120")
    if model.estimator_training_seconds != (120, 125, 130, 135, 140):
        raise ValueError(
            "fixed-time estimator must retain the predecessor 120-140 second training context"
        )
    if (
        model.target != FIXED_TIME_TARGET
        or model.estimator_family != FIXED_TIME_ESTIMATOR_FAMILY
        or model.probability_calibration != FIXED_TIME_CALIBRATION
    ):
        raise ValueError("fixed-time estimator, target, and calibration are frozen")
    if not math.isclose(model.recency_half_life_days, 28.0):
        raise ValueError("fixed-time estimator requires a 28-day recency half-life")
    if (
        model.feature_contract != FIXED_TIME_FEATURE_CONTRACT
        or model.feature_schema_version != CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION
        or model.feature_names != tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)
        or len(model.feature_names) != 71
    ):
        raise ValueError("fixed-time model requires the exact 71-feature core contract")
    if model.include_oracle or model.include_book:
        raise ValueError("fixed-time model excludes oracle and order-book features")
    if any(
        name.startswith("oracle_")
        or "orderbook" in name
        or "vwap" in name
        or "best_bid" in name
        or "best_ask" in name
        for name in model.feature_names
    ):
        raise ValueError("fixed-time feature contract contains oracle or book data")
    if model.threshold_selection != EMPIRICAL_COVERAGE_THRESHOLD_SELECTION:
        raise ValueError("fixed-time thresholds must be selected empirically from policy coverage")
    if not math.isclose(model.hard_confidence_floor, 0.95):
        raise ValueError("fixed-time hard-confidence floor must remain 0.95")
    if not model.require_hard_confident_error_no_regression:
        raise ValueError("fixed-time hard-confident errors must not regress")

    _validate_operating_point(
        config.primary,
        expected=FixedTimeOperatingPointConfig(
            name=PRIMARY_OPERATING_POINT,
            target_coverage=0.15,
            coverage_tolerance=0.03,
            minimum_accuracy=0.91,
            minimum_balanced_accuracy=0.90,
            minimum_direction_recall=0.90,
            minimum_wilson_lower_95=0.895,
            maximum_expected_calibration_error=0.05,
        ),
    )
    _validate_operating_point(
        config.secondary,
        expected=FixedTimeOperatingPointConfig(
            name=SECONDARY_OPERATING_POINT,
            target_coverage=0.10,
            coverage_tolerance=0.025,
            minimum_accuracy=0.93,
            minimum_balanced_accuracy=0.92,
            minimum_direction_recall=0.92,
            minimum_wilson_lower_95=0.90,
            maximum_expected_calibration_error=0.05,
        ),
    )
    if config.primary.target_coverage <= config.secondary.target_coverage:
        raise ValueError("primary coverage must exceed secondary coverage")

    core = load_core_config(benchmark.core_config)
    if (
        core.data.source_contract != CORE_ORACLE_SOURCE_CONTRACT
        or core.data.range_start != config.split.development_start
        or core.data.range_end != config.split.policy_selection_end
        or core.data.sample_interval_seconds != 5
        or core.data.min_seconds_after_open != 120
        or core.data.min_seconds_before_close != 160
        or core.split.development_start != config.split.development_start
        or core.split.development_end != config.split.development_end
        or core.split.probability_calibration_start != config.split.probability_calibration_start
        or core.split.probability_calibration_end != config.split.probability_calibration_end
        or core.split.policy_selection_start != config.split.policy_selection_start
        or core.split.policy_selection_end != config.split.policy_selection_end
        or core.split.validation_windows != config.split.validation_windows
    ):
        raise ValueError("fixed-time benchmark core cache and chronological contract changed")

    paths = config.paths
    execution_manifest = paths.execution_evidence / "manifest.json"
    if not execution_manifest.is_file():
        raise ValueError(f"fixed-time execution manifest is missing: {execution_manifest}")
    _validate_checksum(
        "execution_manifest_sha256",
        paths.execution_manifest_sha256,
    )
    if _file_sha256(execution_manifest) != paths.execution_manifest_sha256:
        raise ValueError("pinned execution-evidence manifest hash mismatch")
    if not paths.predecessor_benchmark.is_file():
        raise ValueError(
            f"fixed-time predecessor benchmark is missing: {paths.predecessor_benchmark}"
        )
    _validate_checksum(
        "predecessor_benchmark_sha256",
        paths.predecessor_benchmark_sha256,
    )
    if _file_sha256(paths.predecessor_benchmark) != paths.predecessor_benchmark_sha256:
        raise ValueError("pinned predecessor benchmark hash mismatch")
    generated_paths = (paths.runs, paths.freezes, paths.runtime_models)
    if len(set(generated_paths)) != len(generated_paths):
        raise ValueError("fixed-time run, freeze, and runtime paths must be isolated")
    if paths.execution_evidence in generated_paths:
        raise ValueError("execution evidence cannot be used as an output path")


def _load_operating_point(
    name: str,
    raw: dict[str, object],
) -> FixedTimeOperatingPointConfig:
    return FixedTimeOperatingPointConfig(
        name=name,
        target_coverage=float(raw["target_coverage"]),
        coverage_tolerance=float(raw["coverage_tolerance"]),
        minimum_accuracy=float(raw["minimum_accuracy"]),
        minimum_balanced_accuracy=float(raw["minimum_balanced_accuracy"]),
        minimum_direction_recall=float(raw["minimum_direction_recall"]),
        minimum_wilson_lower_95=float(raw["minimum_wilson_lower_95"]),
        maximum_expected_calibration_error=float(raw["maximum_expected_calibration_error"]),
    )


def _validate_operating_point(
    observed: FixedTimeOperatingPointConfig,
    *,
    expected: FixedTimeOperatingPointConfig,
) -> None:
    values = (
        observed.target_coverage,
        observed.coverage_tolerance,
        observed.minimum_accuracy,
        observed.minimum_balanced_accuracy,
        observed.minimum_direction_recall,
        observed.minimum_wilson_lower_95,
        observed.maximum_expected_calibration_error,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{observed.name} operating point contains non-finite values")
    if observed != expected:
        raise ValueError(f"{expected.name} operating-point contract changed")


def _validate_checksum(name: str, checksum: str) -> None:
    if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal digits")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
