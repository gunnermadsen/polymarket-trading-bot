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
    CORE_REGIME_REVERSAL_ENRICHED_FEATURES,
    CORE_REGIME_REVERSAL_FEATURE_SCHEMA_VERSION,
)
from .fixed_time_config import (
    EMPIRICAL_COVERAGE_THRESHOLD_SELECTION,
    FIXED_TIME_CALIBRATION,
    FIXED_TIME_ESTIMATOR_FAMILY,
    FIXED_TIME_TARGET,
    PRIMARY_OPERATING_POINT,
    SECONDARY_OPERATING_POINT,
    FixedTimeOperatingPointConfig,
)

FIXED_TIME_SELECTIVE_ACCURACY_PROFILE = "mature_reversal_fixed_120_selective_accuracy"
EXACT_120_SELECTIVE_TUNING_IDENTITY = "exact_120_selective_accuracy"

FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE = (
    "histogram_mature_reversal_fixed_120_control"
)
FIXED_TIME_SELECTIVE_WEIGHT_2X_CANDIDATE = (
    "histogram_mature_reversal_fixed_120_weight_2x"
)
FIXED_TIME_SELECTIVE_WEIGHT_3X_CANDIDATE = (
    "histogram_mature_reversal_fixed_120_weight_3x"
)
FIXED_TIME_SELECTIVE_TAIL_REGULARIZED_CANDIDATE = (
    "histogram_mature_reversal_fixed_120_weight_2x_tail_regularized"
)
FIXED_TIME_SELECTIVE_REGIME_CANDIDATE = (
    "histogram_regime_reversal_fixed_120_weight_2x"
)
FIXED_TIME_SELECTIVE_CANDIDATE_NAMES = (
    FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE,
    FIXED_TIME_SELECTIVE_WEIGHT_2X_CANDIDATE,
    FIXED_TIME_SELECTIVE_WEIGHT_3X_CANDIDATE,
    FIXED_TIME_SELECTIVE_TAIL_REGULARIZED_CANDIDATE,
    FIXED_TIME_SELECTIVE_REGIME_CANDIDATE,
)

BASE_PARAMETER_GRID = "base"
TAIL_PARAMETER_GRID = "tail"
EXPECTED_CORE_CONFIG_SHA256 = (
    "ec5262959ec33d15047943c7242d4e05ad43665f40313e304975c6644127b294"
)
EXPECTED_EXECUTION_MANIFEST_SHA256 = (
    "c02f14c75382c8609fb8d4d905e9ca5735fca33a352d17f6a224aa415742cc34"
)

EXPECTED_VALIDATION_WINDOWS = (
    (datetime(2026, 6, 9, tzinfo=UTC), datetime(2026, 6, 16, tzinfo=UTC)),
    (datetime(2026, 6, 16, tzinfo=UTC), datetime(2026, 6, 23, tzinfo=UTC)),
    (datetime(2026, 6, 23, tzinfo=UTC), datetime(2026, 6, 30, tzinfo=UTC)),
    (datetime(2026, 6, 30, tzinfo=UTC), datetime(2026, 7, 7, tzinfo=UTC)),
    (datetime(2026, 7, 7, tzinfo=UTC), datetime(2026, 7, 14, tzinfo=UTC)),
    (datetime(2026, 7, 14, tzinfo=UTC), datetime(2026, 7, 21, tzinfo=UTC)),
    (datetime(2026, 7, 21, tzinfo=UTC), datetime(2026, 7, 29, tzinfo=UTC)),
)


@dataclass(frozen=True)
class FixedTimeSelectiveIdentityConfig:
    profile: str
    tuning_identity: str
    core_config: Path
    core_config_sha256: str
    evaluation_is_independent: bool
    evaluation_note: str


@dataclass(frozen=True)
class FixedTimeSelectiveSplitConfig:
    development_start: datetime
    development_end: datetime
    probability_calibration_start: datetime
    probability_calibration_end: datetime
    policy_selection_start: datetime
    policy_selection_end: datetime
    validation_windows: tuple[tuple[datetime, datetime], ...]


@dataclass(frozen=True)
class FixedTimeSelectiveHistogramConfig:
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    learning_rate: float
    max_iter: int


@dataclass(frozen=True)
class FixedTimeSelectiveModelConfig:
    decision_second: int
    estimator_training_seconds: tuple[int, ...]
    target: str
    estimator_family: str
    probability_calibration: str
    recency_half_life_days: float
    include_oracle: bool
    include_book: bool
    threshold_selection: str
    hard_confidence_floor: float
    require_hard_confident_error_no_regression: bool
    tail_histogram_parameters: tuple[FixedTimeSelectiveHistogramConfig, ...]


@dataclass(frozen=True)
class FixedTimeSelectiveCandidateConfig:
    name: str
    feature_schema_version: str
    feature_names: tuple[str, ...]
    exact_120_weight_multiplier: float
    parameter_grid: str


@dataclass(frozen=True)
class FixedTimeSelectivePathConfig:
    execution_evidence: Path
    execution_manifest_sha256: str
    runs: Path
    freezes: Path
    runtime_models: Path


@dataclass(frozen=True)
class FixedTimeSelectiveConfig:
    source_path: Path
    package_root: Path
    benchmark: FixedTimeSelectiveIdentityConfig
    split: FixedTimeSelectiveSplitConfig
    model: FixedTimeSelectiveModelConfig
    candidates: tuple[FixedTimeSelectiveCandidateConfig, ...]
    primary: FixedTimeOperatingPointConfig
    secondary: FixedTimeOperatingPointConfig
    paths: FixedTimeSelectivePathConfig


def load_fixed_time_selective_config(path: Path) -> FixedTimeSelectiveConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    benchmark = raw["benchmark"]
    split = raw["split"]
    model = raw["model"]
    paths = raw["paths"]
    validation_starts = tuple(
        parse_utc_day(value) for value in split["validation_starts"]
    )
    validation_ends = tuple(
        parse_utc_day(value) for value in split["validation_ends"]
    )
    if len(validation_starts) != len(validation_ends):
        raise ValueError("validation start/end lists must have equal length")

    config = FixedTimeSelectiveConfig(
        source_path=source_path,
        package_root=package_root,
        benchmark=FixedTimeSelectiveIdentityConfig(
            profile=str(benchmark["profile"]),
            tuning_identity=str(benchmark["tuning_identity"]),
            core_config=package_root / str(benchmark["core_config"]),
            core_config_sha256=str(benchmark["core_config_sha256"]).lower(),
            evaluation_is_independent=bool(benchmark["evaluation_is_independent"]),
            evaluation_note=str(benchmark["evaluation_note"]).strip(),
        ),
        split=FixedTimeSelectiveSplitConfig(
            development_start=parse_utc_day(split["development_start"]),
            development_end=parse_utc_day(split["development_end"]),
            probability_calibration_start=parse_utc_day(
                split["probability_calibration_start"]
            ),
            probability_calibration_end=parse_utc_day(
                split["probability_calibration_end"]
            ),
            policy_selection_start=parse_utc_day(split["policy_selection_start"]),
            policy_selection_end=parse_utc_day(split["policy_selection_end"]),
            validation_windows=tuple(
                zip(validation_starts, validation_ends, strict=True)
            ),
        ),
        model=FixedTimeSelectiveModelConfig(
            decision_second=int(model["decision_second"]),
            estimator_training_seconds=tuple(
                int(value) for value in model["estimator_training_seconds"]
            ),
            target=str(model["target"]),
            estimator_family=str(model["estimator_family"]),
            probability_calibration=str(model["probability_calibration"]),
            recency_half_life_days=float(model["recency_half_life_days"]),
            include_oracle=bool(model["include_oracle"]),
            include_book=bool(model["include_book"]),
            threshold_selection=str(model["threshold_selection"]),
            hard_confidence_floor=float(model["hard_confidence_floor"]),
            require_hard_confident_error_no_regression=bool(
                model["require_hard_confident_error_no_regression"]
            ),
            tail_histogram_parameters=tuple(
                FixedTimeSelectiveHistogramConfig(
                    max_leaf_nodes=int(parameters["max_leaf_nodes"]),
                    min_samples_leaf=int(parameters["min_samples_leaf"]),
                    l2_regularization=float(parameters["l2_regularization"]),
                    learning_rate=float(parameters["learning_rate"]),
                    max_iter=int(parameters["max_iter"]),
                )
                for parameters in raw["tail_histogram_parameters"]
            ),
        ),
        candidates=tuple(
            FixedTimeSelectiveCandidateConfig(
                name=str(candidate["name"]),
                feature_schema_version=str(candidate["feature_schema_version"]),
                feature_names=_feature_names_for_schema(
                    str(candidate["feature_schema_version"])
                ),
                exact_120_weight_multiplier=float(
                    candidate["exact_120_weight_multiplier"]
                ),
                parameter_grid=str(candidate["parameter_grid"]),
            )
            for candidate in raw["candidates"]
        ),
        primary=_load_operating_point(
            PRIMARY_OPERATING_POINT,
            raw["primary_operating_point"],
        ),
        secondary=_load_operating_point(
            SECONDARY_OPERATING_POINT,
            raw["secondary_operating_point"],
        ),
        paths=FixedTimeSelectivePathConfig(
            execution_evidence=package_root / str(paths["execution_evidence"]),
            execution_manifest_sha256=str(
                paths["execution_manifest_sha256"]
            ).lower(),
            runs=package_root / str(paths["runs"]),
            freezes=package_root / str(paths["freezes"]),
            runtime_models=package_root / str(paths["runtime_models"]),
        ),
    )
    validate_fixed_time_selective_config(config)
    return config


def validate_fixed_time_selective_config(config: FixedTimeSelectiveConfig) -> None:
    benchmark = config.benchmark
    if benchmark.profile != FIXED_TIME_SELECTIVE_ACCURACY_PROFILE:
        raise ValueError("fixed-time selective profile identity changed")
    if benchmark.tuning_identity != EXACT_120_SELECTIVE_TUNING_IDENTITY:
        raise ValueError("fixed-time selective tuning identity changed")
    if benchmark.evaluation_is_independent:
        raise ValueError("fixed-time selective validation is consumed development evidence")
    if not benchmark.evaluation_note:
        raise ValueError("fixed-time selective evaluation_note is required")
    if not benchmark.core_config.is_file():
        raise ValueError(f"fixed-time selective core config is missing: {benchmark.core_config}")
    _validate_checksum("core_config_sha256", benchmark.core_config_sha256)
    if benchmark.core_config_sha256 != EXPECTED_CORE_CONFIG_SHA256:
        raise ValueError("fixed-time selective core_config_sha256 identity changed")
    if _file_sha256(benchmark.core_config) != benchmark.core_config_sha256:
        raise ValueError("pinned core training config hash mismatch")

    expected_split = FixedTimeSelectiveSplitConfig(
        development_start=datetime(2026, 3, 21, tzinfo=UTC),
        development_end=datetime(2026, 7, 14, tzinfo=UTC),
        probability_calibration_start=datetime(2026, 7, 14, tzinfo=UTC),
        probability_calibration_end=datetime(2026, 7, 21, tzinfo=UTC),
        policy_selection_start=datetime(2026, 7, 21, tzinfo=UTC),
        policy_selection_end=datetime(2026, 7, 29, tzinfo=UTC),
        validation_windows=EXPECTED_VALIDATION_WINDOWS,
    )
    if config.split != expected_split:
        raise ValueError(
            "fixed-time selective benchmark requires the frozen March 21-July 29 split"
        )
    for previous, current in pairwise(config.split.validation_windows):
        if previous[1] != current[0]:
            raise ValueError(
                "fixed-time selective validation windows must be contiguous and chronological"
            )

    model = config.model
    if model.decision_second != 120:
        raise ValueError("fixed-time selective decision second must remain 120")
    if model.estimator_training_seconds != (120, 125, 130, 135, 140):
        raise ValueError(
            "fixed-time selective estimator must retain the 120-140 second context"
        )
    if (
        model.target != FIXED_TIME_TARGET
        or model.estimator_family != FIXED_TIME_ESTIMATOR_FAMILY
        or model.probability_calibration != FIXED_TIME_CALIBRATION
    ):
        raise ValueError("fixed-time selective estimator, target, and calibration are frozen")
    if not math.isclose(model.recency_half_life_days, 28.0):
        raise ValueError("fixed-time selective estimator requires 28-day recency")
    if model.include_oracle or model.include_book:
        raise ValueError("fixed-time selective candidates exclude oracle and book features")
    if model.threshold_selection != EMPIRICAL_COVERAGE_THRESHOLD_SELECTION:
        raise ValueError("fixed-time selective thresholds require empirical policy coverage")
    if not math.isclose(model.hard_confidence_floor, 0.95):
        raise ValueError("fixed-time selective hard-confidence floor must remain 0.95")
    if not model.require_hard_confident_error_no_regression:
        raise ValueError("fixed-time selective hard-confident errors must not regress")
    _validate_tail_histogram_parameters(model.tail_histogram_parameters)

    _validate_candidates(config.candidates)
    _validate_operating_points(config.primary, config.secondary)

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
        or core.split.probability_calibration_start
        != config.split.probability_calibration_start
        or core.split.probability_calibration_end
        != config.split.probability_calibration_end
        or core.split.policy_selection_start != config.split.policy_selection_start
        or core.split.policy_selection_end != config.split.policy_selection_end
        or core.split.holdout_start != config.split.policy_selection_end
        or core.split.holdout_end != config.split.policy_selection_end
        or core.split.independent_holdout_start is not None
        or core.split.independent_holdout_end is not None
    ):
        raise ValueError("fixed-time selective core cache or chronology changed")

    paths = config.paths
    execution_manifest = paths.execution_evidence / "manifest.json"
    if not execution_manifest.is_file():
        raise ValueError(
            f"fixed-time selective execution manifest is missing: {execution_manifest}"
        )
    _validate_checksum("execution_manifest_sha256", paths.execution_manifest_sha256)
    if paths.execution_manifest_sha256 != EXPECTED_EXECUTION_MANIFEST_SHA256:
        raise ValueError("fixed-time selective execution_manifest_sha256 identity changed")
    if _file_sha256(execution_manifest) != paths.execution_manifest_sha256:
        raise ValueError("pinned execution-evidence manifest hash mismatch")

    expected_generated_paths = (
        config.package_root
        / "runs/btc-mature-reversal-fixed-120-selective-accuracy-20260321-20260729",
        config.package_root
        / "artifacts/btc-mature-reversal-fixed-120-selective-accuracy-20260321-20260729/freezes",
        config.package_root
        / "runtime-models/btc-mature-reversal-fixed-120-selective-accuracy-20260321-20260729",
    )
    generated_paths = (paths.runs, paths.freezes, paths.runtime_models)
    if generated_paths != expected_generated_paths:
        raise ValueError("fixed-time selective output paths changed")
    if len(set(generated_paths)) != len(generated_paths):
        raise ValueError("fixed-time selective output paths must be isolated")
    if paths.execution_evidence in generated_paths:
        raise ValueError("execution evidence cannot be used as an output path")


def _validate_candidates(
    candidates: tuple[FixedTimeSelectiveCandidateConfig, ...],
) -> None:
    if tuple(candidate.name for candidate in candidates) != (
        FIXED_TIME_SELECTIVE_CANDIDATE_NAMES
    ):
        raise ValueError("fixed-time selective candidate matrix changed")

    base_features = tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)
    regime_features = tuple(CORE_REGIME_REVERSAL_ENRICHED_FEATURES)
    expected = (
        (
            CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
            base_features,
            1.0,
            BASE_PARAMETER_GRID,
        ),
        (
            CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
            base_features,
            2.0,
            BASE_PARAMETER_GRID,
        ),
        (
            CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
            base_features,
            3.0,
            BASE_PARAMETER_GRID,
        ),
        (
            CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
            base_features,
            2.0,
            TAIL_PARAMETER_GRID,
        ),
        (
            CORE_REGIME_REVERSAL_FEATURE_SCHEMA_VERSION,
            regime_features,
            2.0,
            BASE_PARAMETER_GRID,
        ),
    )
    observed = tuple(
        (
            candidate.feature_schema_version,
            candidate.feature_names,
            candidate.exact_120_weight_multiplier,
            candidate.parameter_grid,
        )
        for candidate in candidates
    )
    if observed != expected:
        raise ValueError("fixed-time selective candidate feature or training contract changed")
    if len(base_features) != 71 or len(regime_features) != 77:
        raise ValueError("fixed-time selective feature schema widths changed")
    if any(
        name.startswith("oracle_")
        or "orderbook" in name
        or "vwap" in name
        or "best_bid" in name
        or "best_ask" in name
        for candidate in candidates
        for name in candidate.feature_names
    ):
        raise ValueError("fixed-time selective feature contract contains oracle or book data")


def _validate_tail_histogram_parameters(
    observed: tuple[FixedTimeSelectiveHistogramConfig, ...],
) -> None:
    expected = (
        FixedTimeSelectiveHistogramConfig(15, 200, 2.0, 0.05, 160),
        FixedTimeSelectiveHistogramConfig(15, 300, 5.0, 0.03, 220),
        FixedTimeSelectiveHistogramConfig(7, 200, 2.0, 0.05, 160),
    )
    if observed != expected:
        raise ValueError("fixed-time selective tail histogram grid changed")


def _validate_operating_points(
    primary: FixedTimeOperatingPointConfig,
    secondary: FixedTimeOperatingPointConfig,
) -> None:
    expected_primary = FixedTimeOperatingPointConfig(
        name=PRIMARY_OPERATING_POINT,
        target_coverage=0.15,
        coverage_tolerance=0.03,
        minimum_accuracy=0.91,
        minimum_balanced_accuracy=0.90,
        minimum_direction_recall=0.90,
        minimum_wilson_lower_95=0.895,
        maximum_expected_calibration_error=0.05,
    )
    expected_secondary = FixedTimeOperatingPointConfig(
        name=SECONDARY_OPERATING_POINT,
        target_coverage=0.10,
        coverage_tolerance=0.025,
        minimum_accuracy=0.93,
        minimum_balanced_accuracy=0.92,
        minimum_direction_recall=0.92,
        minimum_wilson_lower_95=0.90,
        maximum_expected_calibration_error=0.05,
    )
    if primary != expected_primary or secondary != expected_secondary:
        raise ValueError("fixed-time selective operating-point contract changed")


def _feature_names_for_schema(schema_version: str) -> tuple[str, ...]:
    if schema_version == CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION:
        return tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)
    if schema_version == CORE_REGIME_REVERSAL_FEATURE_SCHEMA_VERSION:
        return tuple(CORE_REGIME_REVERSAL_ENRICHED_FEATURES)
    raise ValueError(f"unsupported fixed-time selective feature schema: {schema_version}")


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
        maximum_expected_calibration_error=float(
            raw["maximum_expected_calibration_error"]
        ),
    )


def _validate_checksum(name: str, checksum: str) -> None:
    if len(checksum) != 64 or any(
        character not in "0123456789abcdef" for character in checksum
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal digits")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
