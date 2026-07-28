from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .core_config import (
    EQUAL_TOTAL_PER_MARKET_NORMALIZATION,
    CandidateRowWeightScheduleConfig,
    RowWeightScheduleConfig,
    load_core_config,
)

PERSISTENCE_CANDIDATES = (
    "histogram_enriched",
    "histogram_path_persistence",
    "histogram_path_persistence_prewindow",
    "histogram_path_persistence_time_calibrated",
)
ACCURACY_TIMING_CANDIDATES = (
    "histogram_enriched",
    "histogram_path_persistence_time_calibrated",
    "histogram_path_persistence_time_calibrated_60_120",
    "histogram_path_persistence_time_calibrated_90_120",
)
FOLD_ROBUST_FREQUENCY_CANDIDATE = "histogram_outcome_fold_robust_agreement"
FOLD_ROBUST_FREQUENCY_CANDIDATES = (
    "histogram_enriched",
    FOLD_ROBUST_FREQUENCY_CANDIDATE,
)
PATH_PERSISTENCE_PROFILE = "path_persistence"
ACCURACY_TIMING_PROFILE = "accuracy_timing"
FOLD_ROBUST_FREQUENCY_PROFILE = "fold_robust_frequency"


@dataclass(frozen=True)
class CalibrationBand:
    name: str
    start_second: int
    end_second_exclusive: int


@dataclass(frozen=True)
class PersistenceBenchmarkConfig:
    source_path: Path
    package_root: Path
    profile: str
    core_config: Path
    control_candidate: str
    candidate_names: tuple[str, ...]
    row_weight_schedules: tuple[CandidateRowWeightScheduleConfig, ...]
    evaluation_note: str
    evaluation_is_independent: bool
    quantity: float
    early_cutoff_second: int
    minimum_early_markets: int
    minimum_common_markets: int
    minimum_executable_markets: int
    minimum_coverage_uplift: float
    maximum_accuracy_regression: float
    maximum_balanced_accuracy_regression: float
    maximum_direction_recall_regression: float
    maximum_median_entry_seconds_regression: float
    minimum_mean_direct_edge_per_share: float
    minimum_realized_net_per_share: float
    calibration_bands: tuple[CalibrationBand, ...]
    minimum_calibration_rows_per_band: int
    prewindow_features: Path
    execution_evidence: Path
    runs: Path


def load_persistence_benchmark_config(path: Path) -> PersistenceBenchmarkConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    benchmark = raw["benchmark"]
    calibration = raw["calibration"]
    gates = raw["advancement_gates"]
    paths = raw["paths"]
    config = PersistenceBenchmarkConfig(
        source_path=source_path,
        package_root=package_root,
        profile=str(benchmark.get("profile", PATH_PERSISTENCE_PROFILE)),
        core_config=package_root / str(benchmark["core_config"]),
        control_candidate=str(benchmark["control_candidate"]),
        candidate_names=tuple(str(value) for value in benchmark["candidate_names"]),
        row_weight_schedules=tuple(
            CandidateRowWeightScheduleConfig(
                candidate=str(schedule["candidate"]),
                schedule=RowWeightScheduleConfig(
                    start_second=(
                        int(schedule["start_second"]) if "start_second" in schedule else None
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
            for schedule in raw.get("training", {}).get(
                "row_weight_schedules",
                (),
            )
        ),
        evaluation_note=str(benchmark["evaluation_note"]).strip(),
        evaluation_is_independent=bool(benchmark["evaluation_is_independent"]),
        quantity=float(benchmark["quantity"]),
        early_cutoff_second=int(benchmark["early_cutoff_second"]),
        minimum_early_markets=int(gates["minimum_early_markets"]),
        minimum_common_markets=int(gates["minimum_common_markets"]),
        minimum_executable_markets=int(gates["minimum_executable_markets"]),
        minimum_coverage_uplift=float(gates["minimum_coverage_uplift"]),
        maximum_accuracy_regression=float(gates["maximum_accuracy_regression"]),
        maximum_balanced_accuracy_regression=float(gates["maximum_balanced_accuracy_regression"]),
        maximum_direction_recall_regression=float(gates["maximum_direction_recall_regression"]),
        maximum_median_entry_seconds_regression=float(
            gates["maximum_median_entry_seconds_regression"]
        ),
        minimum_mean_direct_edge_per_share=float(gates["minimum_mean_direct_edge_per_share"]),
        minimum_realized_net_per_share=float(gates["minimum_realized_net_per_share"]),
        calibration_bands=tuple(
            CalibrationBand(
                name=str(band["name"]),
                start_second=int(band["start_second"]),
                end_second_exclusive=int(band["end_second_exclusive"]),
            )
            for band in calibration["bands"]
        ),
        minimum_calibration_rows_per_band=int(calibration["minimum_rows_per_band"]),
        prewindow_features=package_root / str(paths["prewindow_features"]),
        execution_evidence=package_root / str(paths["execution_evidence"]),
        runs=package_root / str(paths["runs"]),
    )
    validate_persistence_benchmark_config(config)
    return config


def validate_persistence_benchmark_config(
    config: PersistenceBenchmarkConfig,
) -> None:
    if not config.core_config.is_file():
        raise ValueError(f"core config is missing: {config.core_config}")
    if config.profile == PATH_PERSISTENCE_PROFILE:
        if config.candidate_names != PERSISTENCE_CANDIDATES:
            raise ValueError("persistence benchmark requires the frozen four-candidate matrix")
        if config.row_weight_schedules:
            raise ValueError("the historical path-persistence profile cannot configure row weights")
    elif config.profile == ACCURACY_TIMING_PROFILE:
        if config.candidate_names != ACCURACY_TIMING_CANDIDATES:
            raise ValueError("accuracy-timing benchmark requires the frozen four-candidate matrix")
        _validate_accuracy_timing_weights(config)
    elif config.profile == FOLD_ROBUST_FREQUENCY_PROFILE:
        if config.candidate_names != FOLD_ROBUST_FREQUENCY_CANDIDATES:
            raise ValueError(
                "fold-robust frequency benchmark requires its frozen two-candidate matrix"
            )
        _validate_fold_robust_frequency_weights(config)
    else:
        raise ValueError(f"unsupported persistence benchmark profile: {config.profile}")
    if config.control_candidate != config.candidate_names[0]:
        raise ValueError("histogram_enriched must remain the control candidate")
    if config.evaluation_is_independent:
        raise ValueError("the consumed development range cannot be independent")
    if not config.evaluation_note:
        raise ValueError("evaluation_note is required")
    if config.quantity != 5.0:
        raise ValueError("execution economics require exactly five shares")
    if config.early_cutoff_second != 120:
        raise ValueError("the early-entry checkpoint must remain 120 seconds")
    if (
        config.minimum_early_markets < 500
        or config.minimum_common_markets < 500
        or config.minimum_executable_markets < 500
    ):
        raise ValueError("sample gates cannot be weakened below 500 markets")
    if (
        config.minimum_coverage_uplift <= 0.0
        or config.maximum_accuracy_regression > 0.0
        or config.maximum_balanced_accuracy_regression > 0.0
        or config.maximum_direction_recall_regression > 0.0
        or config.maximum_median_entry_seconds_regression > -5.0
        or config.minimum_mean_direct_edge_per_share < 0.0
        or config.minimum_realized_net_per_share < 0.0
    ):
        raise ValueError("advancement gates weaken the frozen non-regression contract")
    expected_bands = (
        CalibrationBand("60-89", 60, 90),
        CalibrationBand("90-119", 90, 120),
        CalibrationBand("120-179", 120, 180),
        CalibrationBand("180-240", 180, 241),
    )
    if config.calibration_bands != expected_bands:
        raise ValueError("time calibration must use the frozen four causal bands")
    if config.minimum_calibration_rows_per_band < 500:
        raise ValueError("each calibration band requires at least 500 rows")

    core = load_core_config(config.core_config)
    if (
        core.data.range_start.isoformat() != "2026-04-21T00:00:00+00:00"
        or core.data.range_end.isoformat() != "2026-07-20T00:00:00+00:00"
        or (core.data.range_end - core.data.range_start).days != 90
    ):
        raise ValueError("persistence benchmark requires exact [2026-04-21, 2026-07-20)")
    if (
        core.split.holdout_start != core.data.range_end
        or core.split.holdout_end != core.data.range_end
    ):
        raise ValueError("the in-range holdout must remain disabled")
    independent_start = core.split.independent_holdout_start
    independent_end = core.split.independent_holdout_end
    if config.profile == PATH_PERSISTENCE_PROFILE:
        if (
            independent_start is None
            or independent_end is None
            or independent_start.isoformat() != "2026-07-21T00:00:00+00:00"
            or independent_end.isoformat() != "2026-08-04T00:00:00+00:00"
        ):
            raise ValueError("the untouched July 21-August 4 holdout contract changed")
    elif independent_start is not None or independent_end is not None:
        raise ValueError(
            "development-only training evidence cannot configure an external holdout"
        )
    if (
        core.model.confidence_min != 0.87
        or core.model.confidence_max != 0.91
        or core.model.confidence_step != 0.01
        or core.model.random_seed != 20260726
        or len(core.model.histogram_candidates) != 4
        or len(core.split.validation_windows) != 5
    ):
        raise ValueError("model search, seed, threshold, or fold contract changed")


def persistence_row_weight_schedule(
    config: PersistenceBenchmarkConfig,
    candidate_name: str,
) -> RowWeightScheduleConfig:
    for entry in config.row_weight_schedules:
        if entry.candidate == candidate_name:
            return entry.schedule
    return RowWeightScheduleConfig(
        start_second=None,
        end_second_inclusive=None,
        multiplier=1.0,
    )


def _validate_accuracy_timing_weights(
    config: PersistenceBenchmarkConfig,
) -> None:
    schedule_candidates = tuple(entry.candidate for entry in config.row_weight_schedules)
    if len(set(schedule_candidates)) != len(schedule_candidates):
        raise ValueError("training row-weight candidates must be unique")
    if set(schedule_candidates) != set(ACCURACY_TIMING_CANDIDATES):
        raise ValueError("accuracy-timing row weights must configure every candidate")
    expected = {
        "histogram_enriched": RowWeightScheduleConfig(None, None, 1.0),
        "histogram_path_persistence_time_calibrated": RowWeightScheduleConfig(
            None,
            None,
            1.0,
        ),
        "histogram_path_persistence_time_calibrated_60_120": (
            RowWeightScheduleConfig(60, 120, 1.5)
        ),
        "histogram_path_persistence_time_calibrated_90_120": (
            RowWeightScheduleConfig(90, 120, 2.0)
        ),
    }
    observed = {entry.candidate: entry.schedule for entry in config.row_weight_schedules}
    if observed != expected:
        raise ValueError(
            "accuracy-timing row weights must preserve the frozen equal-total-per-market schedules"
        )


def _validate_fold_robust_frequency_weights(
    config: PersistenceBenchmarkConfig,
) -> None:
    schedule_candidates = tuple(entry.candidate for entry in config.row_weight_schedules)
    if len(set(schedule_candidates)) != len(schedule_candidates):
        raise ValueError("training row-weight candidates must be unique")
    expected = {
        candidate: RowWeightScheduleConfig(None, None, 1.0)
        for candidate in FOLD_ROBUST_FREQUENCY_CANDIDATES
    }
    observed = {entry.candidate: entry.schedule for entry in config.row_weight_schedules}
    if observed != expected:
        raise ValueError(
            "fold-robust frequency training must preserve equal total weight per market"
        )


def persistence_config_to_dict(
    config: PersistenceBenchmarkConfig,
) -> dict[str, Any]:
    payload = asdict(config)
    for key in (
        "source_path",
        "package_root",
        "core_config",
        "prewindow_features",
        "execution_evidence",
        "runs",
    ):
        payload[key] = str(payload[key])
    return payload
