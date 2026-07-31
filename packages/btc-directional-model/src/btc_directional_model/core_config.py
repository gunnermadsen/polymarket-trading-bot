from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

EQUAL_TOTAL_PER_MARKET_NORMALIZATION = "equal_total_per_market"
CORE_SOURCE_CONTRACT = "btc_core_v1"
CORE_ORACLE_SOURCE_CONTRACT = "btc_core_oracle_v1"
SUPPORTED_CORE_SOURCE_CONTRACTS = (
    CORE_SOURCE_CONTRACT,
    CORE_ORACLE_SOURCE_CONTRACT,
)
HISTORICAL_CORE_CANDIDATE_NAMES = (
    "logistic_baseline",
    "logistic_enriched",
    "histogram_enriched",
    "histogram_early_weighted",
)
KNOWN_CORE_CANDIDATE_FAMILIES = {
    "logistic_baseline": "logistic",
    "logistic_enriched": "logistic",
    "histogram_enriched": "histogram",
    "histogram_early_weighted": "histogram",
    "histogram_early_weighted_moderate": "histogram",
    "histogram_early_90_120": "histogram",
}


@dataclass(frozen=True)
class RowWeightScheduleConfig:
    start_second: int | None
    end_second_inclusive: int | None
    multiplier: float
    normalization: str = EQUAL_TOTAL_PER_MARKET_NORMALIZATION

    def __post_init__(self) -> None:
        has_start = self.start_second is not None
        has_end = self.end_second_inclusive is not None
        if has_start != has_end:
            raise ValueError(
                "row-weight schedule start and end seconds must both be configured"
            )
        if has_start and (
            self.start_second < 0
            or self.end_second_inclusive < self.start_second
            or self.end_second_inclusive > 300
        ):
            raise ValueError("row-weight schedule interval is invalid")
        if self.multiplier <= 0:
            raise ValueError("row-weight schedule multiplier must be positive")
        if self.normalization != EQUAL_TOTAL_PER_MARKET_NORMALIZATION:
            raise ValueError(
                "row-weight schedule must use equal-total-per-market normalization"
            )


@dataclass(frozen=True)
class CandidateRowWeightScheduleConfig:
    candidate: str
    schedule: RowWeightScheduleConfig


@dataclass(frozen=True)
class CoreDataConfig:
    source_contract: str
    range_start: datetime
    range_end: datetime
    sample_interval_seconds: int
    min_seconds_after_open: int
    min_seconds_before_close: int
    strict_final_price_audit: bool


@dataclass(frozen=True)
class CoreSplitConfig:
    development_start: datetime
    development_end: datetime
    probability_calibration_start: datetime
    probability_calibration_end: datetime
    policy_selection_start: datetime
    policy_selection_end: datetime
    holdout_start: datetime
    holdout_end: datetime
    independent_holdout_start: datetime | None
    independent_holdout_end: datetime | None
    validation_windows: tuple[tuple[datetime, datetime], ...]


@dataclass(frozen=True)
class HistogramCandidate:
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float


@dataclass(frozen=True)
class CoreModelConfig:
    c_candidates: tuple[float, ...]
    histogram_candidates: tuple[HistogramCandidate, ...]
    confidence_min: float
    confidence_max: float
    confidence_step: float
    random_seed: int
    candidate_names: tuple[str, ...]
    row_weight_schedules: tuple[CandidateRowWeightScheduleConfig, ...]


@dataclass(frozen=True)
class CoreGateConfig:
    target_accuracy: float
    target_wilson_lower: float
    target_balanced_accuracy: float
    minimum_direction_recall: float
    minimum_coverage: float
    minimum_holdout_markets: int
    maximum_walk_forward_holdout_gap: float
    minimum_same_time_path_uplift: float
    minimum_nonnegative_uplift_folds: int
    maximum_ece: float
    bootstrap_resamples: int


@dataclass(frozen=True)
class CoreComputeConfig:
    max_parallel_fits: int
    threads_per_fit: int
    polars_threads: int


@dataclass(frozen=True)
class CorePathConfig:
    source_data: Path
    development_feature_data: Path
    holdout_feature_data: Path
    runs: Path
    artifacts: Path


@dataclass(frozen=True)
class CoreTrainingConfig:
    source_path: Path
    package_root: Path
    data: CoreDataConfig
    split: CoreSplitConfig
    model: CoreModelConfig
    gates: CoreGateConfig
    compute: CoreComputeConfig
    paths: CorePathConfig


def load_core_config(path: Path) -> CoreTrainingConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    data_raw = raw["data"]
    split_raw = raw["split"]
    model_raw = raw["model"]
    gates_raw = raw["gates"]
    compute_raw = raw["compute"]
    paths_raw = raw["paths"]

    data = CoreDataConfig(
        source_contract=str(data_raw["source_contract"]),
        range_start=parse_utc_day(data_raw["range_start"]),
        range_end=parse_utc_day(data_raw["range_end"]),
        sample_interval_seconds=int(data_raw["sample_interval_seconds"]),
        min_seconds_after_open=int(data_raw["min_seconds_after_open"]),
        min_seconds_before_close=int(data_raw["min_seconds_before_close"]),
        strict_final_price_audit=bool(data_raw["strict_final_price_audit"]),
    )
    validation_starts = tuple(parse_utc_day(value) for value in split_raw["validation_starts"])
    validation_ends = tuple(parse_utc_day(value) for value in split_raw["validation_ends"])
    if len(validation_starts) != len(validation_ends):
        raise ValueError("validation start/end lists must have equal length")
    split = CoreSplitConfig(
        development_start=parse_utc_day(split_raw["development_start"]),
        development_end=parse_utc_day(split_raw["development_end"]),
        probability_calibration_start=parse_utc_day(
            split_raw["probability_calibration_start"]
        ),
        probability_calibration_end=parse_utc_day(split_raw["probability_calibration_end"]),
        policy_selection_start=parse_utc_day(split_raw["policy_selection_start"]),
        policy_selection_end=parse_utc_day(split_raw["policy_selection_end"]),
        holdout_start=parse_utc_day(split_raw["holdout_start"]),
        holdout_end=parse_utc_day(split_raw["holdout_end"]),
        independent_holdout_start=(
            parse_utc_day(split_raw["independent_holdout_start"])
            if "independent_holdout_start" in split_raw
            else None
        ),
        independent_holdout_end=(
            parse_utc_day(split_raw["independent_holdout_end"])
            if "independent_holdout_end" in split_raw
            else None
        ),
        validation_windows=tuple(zip(validation_starts, validation_ends, strict=True)),
    )
    model = CoreModelConfig(
        c_candidates=tuple(float(value) for value in model_raw["c_candidates"]),
        histogram_candidates=tuple(
            HistogramCandidate(
                learning_rate=float(candidate["learning_rate"]),
                max_iter=int(candidate["max_iter"]),
                max_leaf_nodes=int(candidate["max_leaf_nodes"]),
                min_samples_leaf=int(candidate["min_samples_leaf"]),
                l2_regularization=float(candidate["l2_regularization"]),
            )
            for candidate in model_raw["histogram_candidates"]
        ),
        confidence_min=float(model_raw["confidence_min"]),
        confidence_max=float(model_raw["confidence_max"]),
        confidence_step=float(model_raw["confidence_step"]),
        random_seed=int(model_raw["random_seed"]),
        candidate_names=tuple(
            str(value)
            for value in model_raw.get(
                "candidate_names",
                HISTORICAL_CORE_CANDIDATE_NAMES,
            )
        ),
        row_weight_schedules=tuple(
            CandidateRowWeightScheduleConfig(
                candidate=str(schedule["candidate"]),
                schedule=RowWeightScheduleConfig(
                    start_second=(
                        int(schedule["start_second"])
                        if "start_second" in schedule
                        else None
                    ),
                    end_second_inclusive=(
                        int(schedule["end_second_inclusive"])
                        if "end_second_inclusive" in schedule
                        else None
                    ),
                    multiplier=float(schedule["multiplier"]),
                    normalization=str(
                        schedule.get(
                            "normalization",
                            EQUAL_TOTAL_PER_MARKET_NORMALIZATION,
                        )
                    ),
                ),
            )
            for schedule in model_raw.get("row_weight_schedules", ())
        ),
    )
    gates = CoreGateConfig(
        target_accuracy=float(gates_raw["target_accuracy"]),
        target_wilson_lower=float(gates_raw["target_wilson_lower"]),
        target_balanced_accuracy=float(gates_raw["target_balanced_accuracy"]),
        minimum_direction_recall=float(gates_raw["minimum_direction_recall"]),
        minimum_coverage=float(gates_raw["minimum_coverage"]),
        minimum_holdout_markets=int(gates_raw["minimum_holdout_markets"]),
        maximum_walk_forward_holdout_gap=float(
            gates_raw["maximum_walk_forward_holdout_gap"]
        ),
        minimum_same_time_path_uplift=float(
            gates_raw["minimum_same_time_path_uplift"]
        ),
        minimum_nonnegative_uplift_folds=int(
            gates_raw["minimum_nonnegative_uplift_folds"]
        ),
        maximum_ece=float(gates_raw["maximum_ece"]),
        bootstrap_resamples=int(gates_raw["bootstrap_resamples"]),
    )
    compute = CoreComputeConfig(
        max_parallel_fits=int(compute_raw["max_parallel_fits"]),
        threads_per_fit=int(compute_raw["threads_per_fit"]),
        polars_threads=int(compute_raw["polars_threads"]),
    )
    paths = CorePathConfig(
        source_data=package_root / paths_raw["source_data"],
        development_feature_data=package_root / paths_raw["development_feature_data"],
        holdout_feature_data=package_root / paths_raw["holdout_feature_data"],
        runs=package_root / paths_raw["runs"],
        artifacts=package_root / paths_raw["artifacts"],
    )
    config = CoreTrainingConfig(
        source_path=source_path,
        package_root=package_root,
        data=data,
        split=split,
        model=model,
        gates=gates,
        compute=compute,
        paths=paths,
    )
    validate_core_config(config)
    return config


def validate_core_config(config: CoreTrainingConfig) -> None:
    data = config.data
    split = config.split
    if data.source_contract not in SUPPORTED_CORE_SOURCE_CONTRACTS:
        raise ValueError(
            "data.source_contract must be one of "
            + ", ".join(SUPPORTED_CORE_SOURCE_CONTRACTS)
        )
    if data.range_end <= data.range_start:
        raise ValueError("data range must be positive")
    if data.sample_interval_seconds <= 0:
        raise ValueError("sample interval must be positive")
    if data.min_seconds_after_open < 0 or data.min_seconds_before_close <= 0:
        raise ValueError("entry-window boundaries are invalid")
    if data.min_seconds_after_open + data.min_seconds_before_close >= 300:
        raise ValueError("entry-window boundaries leave no prediction time")
    prediction_span = (
        300 - data.min_seconds_before_close - data.min_seconds_after_open
    )
    if prediction_span % data.sample_interval_seconds:
        raise ValueError("sample cadence must divide the prediction window")

    ordered_boundaries = (
        split.development_start,
        split.development_end,
        split.probability_calibration_start,
        split.probability_calibration_end,
        split.policy_selection_start,
        split.policy_selection_end,
        split.holdout_start,
        split.holdout_end,
    )
    if ordered_boundaries != tuple(sorted(ordered_boundaries)):
        raise ValueError("core split boundaries must be chronological")
    if (
        data.range_start != split.development_start
        or split.development_end != split.probability_calibration_start
        or split.probability_calibration_end != split.policy_selection_start
        or split.policy_selection_end != split.holdout_start
        or split.holdout_end != data.range_end
    ):
        raise ValueError("core split boundaries must be contiguous and span the data range")
    independent_start = split.independent_holdout_start
    independent_end = split.independent_holdout_end
    if (independent_start is None) != (independent_end is None):
        raise ValueError(
            "independent holdout start and end must both be configured"
        )
    if independent_start is not None:
        if independent_start < data.range_end:
            raise ValueError(
                "independent holdout must not overlap the configured training range"
            )
        if independent_start >= independent_end:
            raise ValueError("independent holdout range must be positive")
    if any(start >= end for start, end in split.validation_windows):
        raise ValueError("validation windows must be positive")
    if any(
        start < split.development_start or end > split.development_end
        for start, end in split.validation_windows
    ):
        raise ValueError("validation windows must stay inside development")
    for previous, current in zip(
        split.validation_windows, split.validation_windows[1:], strict=False
    ):
        if previous[1] > current[0]:
            raise ValueError("validation windows must not overlap")

    if not config.model.c_candidates or any(
        candidate <= 0 for candidate in config.model.c_candidates
    ):
        raise ValueError("model.c_candidates must be positive")
    if not config.model.histogram_candidates:
        raise ValueError("at least one histogram candidate is required")
    candidate_names = config.model.candidate_names
    if not candidate_names:
        raise ValueError("model.candidate_names must not be empty")
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("model.candidate_names must be unique")
    unknown_candidates = sorted(
        set(candidate_names) - KNOWN_CORE_CANDIDATE_FAMILIES.keys()
    )
    if unknown_candidates:
        raise ValueError(
            "model.candidate_names contains unknown candidates: "
            + ", ".join(unknown_candidates)
        )
    schedule_candidates = tuple(
        entry.candidate for entry in config.model.row_weight_schedules
    )
    if len(set(schedule_candidates)) != len(schedule_candidates):
        raise ValueError("model.row_weight_schedules candidates must be unique")
    if (
        candidate_names != HISTORICAL_CORE_CANDIDATE_NAMES
        and not schedule_candidates
    ):
        raise ValueError(
            "non-historical candidate profiles require explicit row-weight schedules"
        )
    if schedule_candidates:
        if set(schedule_candidates) != set(candidate_names):
            raise ValueError(
                "model.row_weight_schedules must configure every selected candidate"
            )
        non_histogram_candidates = [
            name
            for name in candidate_names
            if KNOWN_CORE_CANDIDATE_FAMILIES[name] != "histogram"
        ]
        if non_histogram_candidates:
            raise ValueError(
                "configured row-weight candidate profiles must be histogram-only"
            )
    scoring_start = data.min_seconds_after_open
    scoring_end = 300 - data.min_seconds_before_close
    cadence = data.sample_interval_seconds
    for entry in config.model.row_weight_schedules:
        schedule = entry.schedule
        if schedule.start_second is None:
            if schedule.multiplier != 1.0:
                raise ValueError(
                    "untimed row-weight schedules must use a 1.0 multiplier"
                )
            continue
        if (
            schedule.start_second < scoring_start
            or schedule.end_second_inclusive > scoring_end
        ):
            raise ValueError(
                "row-weight schedule must stay inside the configured scoring window"
            )
        if (
            (schedule.start_second - scoring_start) % cadence
            or (schedule.end_second_inclusive - scoring_start) % cadence
        ):
            raise ValueError(
                "row-weight schedule boundaries must align to the scoring cadence"
            )
    if not 0.5 <= config.model.confidence_min <= config.model.confidence_max < 1:
        raise ValueError("confidence bounds must satisfy 0.5 <= min <= max < 1")
    if config.model.confidence_step <= 0:
        raise ValueError("confidence step must be positive")
    probability_gates = (
        config.gates.target_accuracy,
        config.gates.target_wilson_lower,
        config.gates.target_balanced_accuracy,
        config.gates.minimum_direction_recall,
        config.gates.minimum_coverage,
        config.gates.maximum_walk_forward_holdout_gap,
        config.gates.minimum_same_time_path_uplift,
        config.gates.maximum_ece,
    )
    if any(value < 0 or value > 1 for value in probability_gates):
        raise ValueError("probability and accuracy gates must be within [0, 1]")
    if config.gates.minimum_holdout_markets <= 0:
        raise ValueError("minimum holdout markets must be positive")
    if not (
        1
        <= config.gates.minimum_nonnegative_uplift_folds
        <= len(split.validation_windows)
    ):
        raise ValueError(
            "minimum nonnegative-uplift folds is incompatible with validation windows"
        )
    if config.gates.bootstrap_resamples < 100:
        raise ValueError("bootstrap resamples must be at least 100")
    if config.compute.max_parallel_fits <= 0 or config.compute.threads_per_fit <= 0:
        raise ValueError("compute worker limits must be positive")
    if config.compute.polars_threads <= 0:
        raise ValueError("Polars thread limit must be positive")
    generated_paths = (
        config.paths.source_data,
        config.paths.development_feature_data,
        config.paths.holdout_feature_data,
        config.paths.runs,
        config.paths.artifacts,
    )
    if len(set(generated_paths)) != len(generated_paths):
        raise ValueError("generated paths must be isolated")


def parse_utc_day(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include an offset: {value}")
    parsed = parsed.astimezone(UTC)
    if parsed.time() != time():
        raise ValueError(f"timestamp must align to a UTC day: {value}")
    return parsed


def config_to_dict(config: CoreTrainingConfig) -> dict[str, Any]:
    return {
        "data": {
            "source_contract": config.data.source_contract,
            "range_start": config.data.range_start.isoformat(),
            "range_end": config.data.range_end.isoformat(),
        },
        "split": {
            "development_start": config.split.development_start.isoformat(),
            "development_end": config.split.development_end.isoformat(),
            "probability_calibration_start": (
                config.split.probability_calibration_start.isoformat()
            ),
            "probability_calibration_end": (
                config.split.probability_calibration_end.isoformat()
            ),
            "policy_selection_start": config.split.policy_selection_start.isoformat(),
            "policy_selection_end": config.split.policy_selection_end.isoformat(),
            "holdout_start": config.split.holdout_start.isoformat(),
            "holdout_end": config.split.holdout_end.isoformat(),
            "independent_holdout_start": (
                config.split.independent_holdout_start.isoformat()
                if config.split.independent_holdout_start is not None
                else None
            ),
            "independent_holdout_end": (
                config.split.independent_holdout_end.isoformat()
                if config.split.independent_holdout_end is not None
                else None
            ),
            "validation_windows": [
                {"start": start.isoformat(), "end": end.isoformat()}
                for start, end in config.split.validation_windows
            ],
        },
        "model": {
            "candidate_names": list(config.model.candidate_names),
            "row_weight_schedules": [
                {
                    "candidate": entry.candidate,
                    **asdict(entry.schedule),
                }
                for entry in config.model.row_weight_schedules
            ],
            "c_candidates": list(config.model.c_candidates),
            "histogram_candidates": [
                asdict(candidate)
                for candidate in config.model.histogram_candidates
            ],
            "confidence_min": config.model.confidence_min,
            "confidence_max": config.model.confidence_max,
            "confidence_step": config.model.confidence_step,
            "random_seed": config.model.random_seed,
        },
        "gates": asdict(config.gates),
        "compute": asdict(config.compute),
    }


def evaluation_holdout_range(
    config: CoreTrainingConfig,
) -> tuple[datetime, datetime]:
    split = config.split
    if split.independent_holdout_start is not None:
        if split.independent_holdout_end is None:
            raise RuntimeError("independent holdout end is unavailable")
        return split.independent_holdout_start, split.independent_holdout_end
    return split.holdout_start, split.holdout_end
