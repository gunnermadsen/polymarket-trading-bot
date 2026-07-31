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
from .fixed_time_config import (
    EMPIRICAL_COVERAGE_THRESHOLD_SELECTION,
    FIXED_TIME_CALIBRATION,
    FIXED_TIME_ESTIMATOR_FAMILY,
    FIXED_TIME_TARGET,
    PRIMARY_OPERATING_POINT,
    FixedTimeOperatingPointConfig,
)

FIXED_TIME_REVERSAL_DECISION_PROFILE = (
    "mature_reversal_fixed_120_reversal_decision"
)
EXACT_120_REVERSAL_TUNING_IDENTITY = "exact_120_reversal_decision"
DIAGNOSTIC_OPERATING_POINT = "diagnostic"
PATH_PERSISTENCE_TARGET = "path_persistence"

FIXED_TIME_REVERSAL_CONTROL_CANDIDATE = (
    "histogram_mature_reversal_fixed_120_outcome_control"
)
FIXED_TIME_REVERSAL_BASE_CANDIDATE = (
    "histogram_mature_reversal_fixed_120_path_persistence"
)
FIXED_TIME_REVERSAL_TAIL_CANDIDATE = (
    "histogram_mature_reversal_fixed_120_path_persistence_tail_regularized"
)
FIXED_TIME_REVERSAL_CANDIDATE_NAMES = (
    FIXED_TIME_REVERSAL_CONTROL_CANDIDATE,
    FIXED_TIME_REVERSAL_BASE_CANDIDATE,
    FIXED_TIME_REVERSAL_TAIL_CANDIDATE,
)

BASE_PARAMETER_GRID = "base"
TAIL_PARAMETER_GRID = "tail"
EXPECTED_SOURCE_MARKETS = 36_579
EXPECTED_SOURCE_ESTIMATOR_ROWS = 182_895
EXPECTED_ELIGIBLE_MARKETS = 36_561
EXPECTED_ELIGIBLE_ESTIMATOR_ROWS = 181_751
EXPECTED_EXACT_120_ELIGIBLE_MARKETS = 36_301
PATH_ZERO_EPSILON_BPS = 1e-12
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
class FixedTimeReversalIdentityConfig:
    profile: str
    tuning_identity: str
    core_config: Path
    core_config_sha256: str
    evaluation_is_independent: bool
    evaluation_note: str


@dataclass(frozen=True)
class FixedTimeReversalSplitConfig:
    development_start: datetime
    development_end: datetime
    probability_calibration_start: datetime
    probability_calibration_end: datetime
    policy_selection_start: datetime
    policy_selection_end: datetime
    validation_windows: tuple[tuple[datetime, datetime], ...]


@dataclass(frozen=True)
class FixedTimeReversalHistogramConfig:
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    learning_rate: float
    max_iter: int


@dataclass(frozen=True)
class FixedTimeReversalModelConfig:
    decision_second: int
    estimator_training_seconds: tuple[int, ...]
    decision_target: str
    reversal_estimator_positive_class: str
    estimator_family: str
    probability_calibration: str
    recency_half_life_days: float
    include_oracle: bool
    include_book: bool
    threshold_selection: str
    path_zero_epsilon_bps: float
    expected_source_markets: int
    expected_source_estimator_rows: int
    expected_eligible_markets: int
    expected_eligible_estimator_rows: int
    expected_exact_120_eligible_markets: int
    hard_confidence_floor: float
    tail_histogram_parameters: tuple[FixedTimeReversalHistogramConfig, ...]


@dataclass(frozen=True)
class FixedTimeReversalCandidateConfig:
    name: str
    target_kind: str
    feature_schema_version: str
    feature_names: tuple[str, ...]
    market_weight_multiplier: float
    parameter_grid: str


@dataclass(frozen=True)
class FixedTimeReversalPathConfig:
    execution_evidence: Path
    execution_manifest_sha256: str
    runs: Path
    freezes: Path
    runtime_models: Path


@dataclass(frozen=True)
class FixedTimeReversalConfig:
    source_path: Path
    package_root: Path
    benchmark: FixedTimeReversalIdentityConfig
    split: FixedTimeReversalSplitConfig
    model: FixedTimeReversalModelConfig
    candidates: tuple[FixedTimeReversalCandidateConfig, ...]
    primary: FixedTimeOperatingPointConfig
    diagnostic: FixedTimeOperatingPointConfig
    paths: FixedTimeReversalPathConfig


def load_fixed_time_reversal_config(path: Path) -> FixedTimeReversalConfig:
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

    config = FixedTimeReversalConfig(
        source_path=source_path,
        package_root=package_root,
        benchmark=FixedTimeReversalIdentityConfig(
            profile=str(benchmark["profile"]),
            tuning_identity=str(benchmark["tuning_identity"]),
            core_config=package_root / str(benchmark["core_config"]),
            core_config_sha256=str(benchmark["core_config_sha256"]).lower(),
            evaluation_is_independent=bool(benchmark["evaluation_is_independent"]),
            evaluation_note=str(benchmark["evaluation_note"]).strip(),
        ),
        split=FixedTimeReversalSplitConfig(
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
        model=FixedTimeReversalModelConfig(
            decision_second=int(model["decision_second"]),
            estimator_training_seconds=tuple(
                int(value) for value in model["estimator_training_seconds"]
            ),
            decision_target=str(model["decision_target"]),
            reversal_estimator_positive_class=str(
                model["reversal_estimator_positive_class"]
            ),
            estimator_family=str(model["estimator_family"]),
            probability_calibration=str(model["probability_calibration"]),
            recency_half_life_days=float(model["recency_half_life_days"]),
            include_oracle=bool(model["include_oracle"]),
            include_book=bool(model["include_book"]),
            threshold_selection=str(model["threshold_selection"]),
            path_zero_epsilon_bps=float(model["path_zero_epsilon_bps"]),
            expected_source_markets=int(model["expected_source_markets"]),
            expected_source_estimator_rows=int(
                model["expected_source_estimator_rows"]
            ),
            expected_eligible_markets=int(model["expected_eligible_markets"]),
            expected_eligible_estimator_rows=int(
                model["expected_eligible_estimator_rows"]
            ),
            expected_exact_120_eligible_markets=int(
                model["expected_exact_120_eligible_markets"]
            ),
            hard_confidence_floor=float(model["hard_confidence_floor"]),
            tail_histogram_parameters=tuple(
                FixedTimeReversalHistogramConfig(
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
            FixedTimeReversalCandidateConfig(
                name=str(candidate["name"]),
                target_kind=str(candidate["target_kind"]),
                feature_schema_version=str(candidate["feature_schema_version"]),
                feature_names=_feature_names_for_schema(
                    str(candidate["feature_schema_version"])
                ),
                market_weight_multiplier=float(
                    candidate["market_weight_multiplier"]
                ),
                parameter_grid=str(candidate["parameter_grid"]),
            )
            for candidate in raw["candidates"]
        ),
        primary=_load_operating_point(
            PRIMARY_OPERATING_POINT,
            raw["primary_operating_point"],
        ),
        diagnostic=_load_operating_point(
            DIAGNOSTIC_OPERATING_POINT,
            raw["diagnostic_operating_point"],
        ),
        paths=FixedTimeReversalPathConfig(
            execution_evidence=package_root / str(paths["execution_evidence"]),
            execution_manifest_sha256=str(
                paths["execution_manifest_sha256"]
            ).lower(),
            runs=package_root / str(paths["runs"]),
            freezes=package_root / str(paths["freezes"]),
            runtime_models=package_root / str(paths["runtime_models"]),
        ),
    )
    validate_fixed_time_reversal_config(config)
    return config


def validate_fixed_time_reversal_config(config: FixedTimeReversalConfig) -> None:
    benchmark = config.benchmark
    if benchmark.profile != FIXED_TIME_REVERSAL_DECISION_PROFILE:
        raise ValueError("fixed-time reversal profile identity changed")
    if benchmark.tuning_identity != EXACT_120_REVERSAL_TUNING_IDENTITY:
        raise ValueError("fixed-time reversal tuning identity changed")
    if benchmark.evaluation_is_independent:
        raise ValueError("fixed-time reversal validation is consumed development evidence")
    if not benchmark.evaluation_note:
        raise ValueError("fixed-time reversal evaluation_note is required")
    if not benchmark.core_config.is_file():
        raise ValueError(
            f"fixed-time reversal core config is missing: {benchmark.core_config}"
        )
    _validate_checksum("core_config_sha256", benchmark.core_config_sha256)
    if benchmark.core_config_sha256 != EXPECTED_CORE_CONFIG_SHA256:
        raise ValueError("fixed-time reversal core_config_sha256 identity changed")
    if _file_sha256(benchmark.core_config) != benchmark.core_config_sha256:
        raise ValueError("pinned core training config hash mismatch")

    expected_split = FixedTimeReversalSplitConfig(
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
            "fixed-time reversal benchmark requires the frozen March 21-July 29 split"
        )
    for previous, current in pairwise(config.split.validation_windows):
        if previous[1] != current[0]:
            raise ValueError(
                "fixed-time reversal validation windows must be contiguous and "
                "chronological"
            )

    _validate_model(config.model)
    _validate_candidates(config.candidates)
    _validate_operating_points(config.primary, config.diagnostic)
    _validate_core_contract(config)
    _validate_paths(config)


def _validate_model(model: FixedTimeReversalModelConfig) -> None:
    if model.decision_second != 120:
        raise ValueError("fixed-time reversal decision second must remain 120")
    if model.estimator_training_seconds != (120, 125, 130, 135, 140):
        raise ValueError(
            "fixed-time reversal estimator must retain the 120-140 second context"
        )
    if model.decision_target != FIXED_TIME_TARGET:
        raise ValueError("fixed-time reversal decision target must remain outcome_up")
    if model.reversal_estimator_positive_class != PATH_PERSISTENCE_TARGET:
        raise ValueError(
            "fixed-time reversal estimator-positive class must remain path_persistence"
        )
    if (
        model.estimator_family != FIXED_TIME_ESTIMATOR_FAMILY
        or model.probability_calibration != FIXED_TIME_CALIBRATION
    ):
        raise ValueError("fixed-time reversal estimator and calibration are frozen")
    if not math.isclose(model.recency_half_life_days, 28.0):
        raise ValueError("fixed-time reversal estimator requires 28-day recency")
    if model.include_oracle or model.include_book:
        raise ValueError("fixed-time reversal candidates exclude oracle and book features")
    if model.threshold_selection != EMPIRICAL_COVERAGE_THRESHOLD_SELECTION:
        raise ValueError("fixed-time reversal thresholds require empirical policy coverage")
    if not math.isclose(
        model.path_zero_epsilon_bps,
        PATH_ZERO_EPSILON_BPS,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("fixed-time reversal path-zero epsilon changed")
    expected_counts = (
        ("source market", model.expected_source_markets, EXPECTED_SOURCE_MARKETS),
        (
            "source estimator row",
            model.expected_source_estimator_rows,
            EXPECTED_SOURCE_ESTIMATOR_ROWS,
        ),
        (
            "eligible market",
            model.expected_eligible_markets,
            EXPECTED_ELIGIBLE_MARKETS,
        ),
        (
            "eligible estimator row",
            model.expected_eligible_estimator_rows,
            EXPECTED_ELIGIBLE_ESTIMATOR_ROWS,
        ),
        (
            "exact-120 eligible market",
            model.expected_exact_120_eligible_markets,
            EXPECTED_EXACT_120_ELIGIBLE_MARKETS,
        ),
    )
    for name, observed, expected in expected_counts:
        if observed != expected:
            raise ValueError(f"fixed-time reversal expected {name} count changed")
    if model.expected_source_estimator_rows != (
        model.expected_source_markets * len(model.estimator_training_seconds)
    ):
        raise ValueError(
            "fixed-time reversal source estimator rows must be a complete cadence"
        )
    if (
        model.expected_eligible_markets > model.expected_source_markets
        or model.expected_eligible_estimator_rows
        > model.expected_source_estimator_rows
        or model.expected_exact_120_eligible_markets
        > model.expected_eligible_markets
    ):
        raise ValueError("fixed-time reversal expected cohort counts are inconsistent")
    if not math.isclose(model.hard_confidence_floor, 0.95):
        raise ValueError("fixed-time reversal hard-confidence floor must remain 0.95")
    _validate_tail_histogram_parameters(model.tail_histogram_parameters)


def _validate_candidates(
    candidates: tuple[FixedTimeReversalCandidateConfig, ...],
) -> None:
    if tuple(candidate.name for candidate in candidates) != (
        FIXED_TIME_REVERSAL_CANDIDATE_NAMES
    ):
        raise ValueError("fixed-time reversal candidate matrix changed")

    features = tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)
    expected = (
        (FIXED_TIME_TARGET, BASE_PARAMETER_GRID),
        (PATH_PERSISTENCE_TARGET, BASE_PARAMETER_GRID),
        (PATH_PERSISTENCE_TARGET, TAIL_PARAMETER_GRID),
    )
    observed = tuple(
        (candidate.target_kind, candidate.parameter_grid)
        for candidate in candidates
    )
    if observed != expected:
        raise ValueError("fixed-time reversal target or parameter-grid matrix changed")
    if len(features) != 71:
        raise ValueError("fixed-time reversal feature schema width changed")
    for candidate in candidates:
        if (
            candidate.feature_schema_version
            != CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION
            or candidate.feature_names != features
            or not math.isclose(candidate.market_weight_multiplier, 1.0)
        ):
            raise ValueError(
                "fixed-time reversal candidate feature or weighting contract changed"
            )
    if any(
        name.startswith("oracle_")
        or "orderbook" in name
        or "vwap" in name
        or "best_bid" in name
        or "best_ask" in name
        for candidate in candidates
        for name in candidate.feature_names
    ):
        raise ValueError("fixed-time reversal feature contract contains oracle or book data")


def _validate_tail_histogram_parameters(
    observed: tuple[FixedTimeReversalHistogramConfig, ...],
) -> None:
    expected = (
        FixedTimeReversalHistogramConfig(15, 200, 2.0, 0.05, 160),
        FixedTimeReversalHistogramConfig(15, 300, 5.0, 0.03, 220),
        FixedTimeReversalHistogramConfig(7, 200, 2.0, 0.05, 160),
    )
    if observed != expected:
        raise ValueError("fixed-time reversal tail histogram grid changed")


def _validate_operating_points(
    primary: FixedTimeOperatingPointConfig,
    diagnostic: FixedTimeOperatingPointConfig,
) -> None:
    expected_primary = FixedTimeOperatingPointConfig(
        name=PRIMARY_OPERATING_POINT,
        target_coverage=0.10,
        coverage_tolerance=0.025,
        minimum_accuracy=0.93,
        minimum_balanced_accuracy=0.92,
        minimum_direction_recall=0.92,
        minimum_wilson_lower_95=0.90,
        maximum_expected_calibration_error=0.05,
    )
    expected_diagnostic = FixedTimeOperatingPointConfig(
        name=DIAGNOSTIC_OPERATING_POINT,
        target_coverage=0.08,
        coverage_tolerance=0.02,
        minimum_accuracy=0.935,
        minimum_balanced_accuracy=0.925,
        minimum_direction_recall=0.92,
        minimum_wilson_lower_95=0.90,
        maximum_expected_calibration_error=0.05,
    )
    if primary != expected_primary or diagnostic != expected_diagnostic:
        raise ValueError("fixed-time reversal operating-point contract changed")


def _validate_core_contract(config: FixedTimeReversalConfig) -> None:
    core = load_core_config(config.benchmark.core_config)
    split = config.split
    if (
        core.data.source_contract != CORE_ORACLE_SOURCE_CONTRACT
        or core.data.range_start != split.development_start
        or core.data.range_end != split.policy_selection_end
        or core.data.sample_interval_seconds != 5
        or core.data.min_seconds_after_open != 120
        or core.data.min_seconds_before_close != 160
        or core.split.development_start != split.development_start
        or core.split.development_end != split.development_end
        or core.split.probability_calibration_start
        != split.probability_calibration_start
        or core.split.probability_calibration_end
        != split.probability_calibration_end
        or core.split.policy_selection_start != split.policy_selection_start
        or core.split.policy_selection_end != split.policy_selection_end
        or core.split.holdout_start != split.policy_selection_end
        or core.split.holdout_end != split.policy_selection_end
        or core.split.independent_holdout_start is not None
        or core.split.independent_holdout_end is not None
    ):
        raise ValueError("fixed-time reversal core cache or chronology changed")


def _validate_paths(config: FixedTimeReversalConfig) -> None:
    paths = config.paths
    execution_manifest = paths.execution_evidence / "manifest.json"
    if not execution_manifest.is_file():
        raise ValueError(
            f"fixed-time reversal execution manifest is missing: {execution_manifest}"
        )
    _validate_checksum("execution_manifest_sha256", paths.execution_manifest_sha256)
    if paths.execution_manifest_sha256 != EXPECTED_EXECUTION_MANIFEST_SHA256:
        raise ValueError("fixed-time reversal execution manifest identity changed")
    if _file_sha256(execution_manifest) != paths.execution_manifest_sha256:
        raise ValueError("pinned execution-evidence manifest hash mismatch")

    expected_generated_paths = (
        config.package_root
        / "runs/btc-mature-reversal-fixed-120-reversal-decision-20260321-20260729",
        config.package_root
        / "artifacts/btc-mature-reversal-fixed-120-reversal-decision-20260321-20260729/freezes",
        config.package_root
        / "runtime-models/btc-mature-reversal-fixed-120-reversal-decision-20260321-20260729",
    )
    generated_paths = (paths.runs, paths.freezes, paths.runtime_models)
    if generated_paths != expected_generated_paths:
        raise ValueError("fixed-time reversal output paths changed")
    if len(set(generated_paths)) != len(generated_paths):
        raise ValueError("fixed-time reversal output paths must be isolated")
    if paths.execution_evidence in generated_paths:
        raise ValueError("execution evidence cannot be used as an output path")


def _feature_names_for_schema(schema_version: str) -> tuple[str, ...]:
    if schema_version == CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION:
        return tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)
    raise ValueError(f"unsupported fixed-time reversal feature schema: {schema_version}")


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
