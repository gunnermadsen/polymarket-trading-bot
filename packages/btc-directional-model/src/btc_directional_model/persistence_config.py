from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

from .core_config import (
    CORE_ORACLE_SOURCE_CONTRACT,
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
BOUNDARY_ALIGNMENT_CANDIDATE = "histogram_boundary_enriched"
BOUNDARY_ALIGNMENT_CANDIDATES = (
    "histogram_enriched",
    BOUNDARY_ALIGNMENT_CANDIDATE,
)
BOUNDARY_REVERSAL_ACCURACY_CANDIDATE = "histogram_boundary_reversal"
BOUNDARY_REVERSAL_ACCURACY_CANDIDATES = (
    "histogram_enriched",
    BOUNDARY_REVERSAL_ACCURACY_CANDIDATE,
)
RESIDUAL_ADMISSION_SOURCE_CANDIDATES = BOUNDARY_REVERSAL_ACCURACY_CANDIDATES
MATURE_REVERSAL_ACCURACY_CANDIDATE = "histogram_mature_reversal"
MATURE_REVERSAL_ACCURACY_CANDIDATES = (
    "histogram_enriched",
    MATURE_REVERSAL_ACCURACY_CANDIDATE,
)
REGIME_ROBUST_RECENCY_CANDIDATE = "histogram_mature_reversal_recency_28d"
REGIME_ROBUST_FEATURE_CANDIDATE = "histogram_regime_reversal"
REGIME_ROBUST_REGULARIZED_CANDIDATE = (
    "histogram_mature_reversal_market_regularized"
)
MATURE_REVERSAL_ORACLE_CONTROL_CANDIDATE = (
    "histogram_mature_reversal_recency_28d_oracle_control"
)
MATURE_REVERSAL_ORACLE_CANDIDATE = (
    "histogram_mature_reversal_oracle_recency_28d"
)
MATURE_REVERSAL_ORACLE_ACCURACY_CANDIDATES = (
    MATURE_REVERSAL_ORACLE_CONTROL_CANDIDATE,
    MATURE_REVERSAL_ORACLE_CANDIDATE,
)
REGIME_ROBUST_ACCURACY_CANDIDATES = (
    "histogram_enriched",
    MATURE_REVERSAL_ACCURACY_CANDIDATE,
    REGIME_ROBUST_RECENCY_CANDIDATE,
    REGIME_ROBUST_FEATURE_CANDIDATE,
    REGIME_ROBUST_REGULARIZED_CANDIDATE,
)
PATH_PERSISTENCE_PROFILE = "path_persistence"
ACCURACY_TIMING_PROFILE = "accuracy_timing"
FOLD_ROBUST_FREQUENCY_PROFILE = "fold_robust_frequency"
BOUNDARY_ALIGNMENT_PROFILE = "boundary_alignment"
BOUNDARY_REVERSAL_ACCURACY_PROFILE = "boundary_reversal_accuracy"
RESIDUAL_ADMISSION_SOURCE_PROFILE = "residual_admission_source"
MATURE_REVERSAL_ACCURACY_PROFILE = "mature_reversal_accuracy"
MATURE_REVERSAL_ORACLE_ACCURACY_PROFILE = (
    "mature_reversal_oracle_accuracy"
)
REGIME_ROBUST_ACCURACY_PROFILE = "regime_robust_accuracy"
DEFAULT_FIXED_EVALUATION_SECONDS = (60, 90, 120, 180, 240)
ORACLE_EARLY_ENTRY_FIXED_EVALUATION_SECONDS = (120, 125, 130, 135, 140)
REGIME_ROBUST_VALIDATION_STARTS = (
    "2026-06-09T00:00:00+00:00",
    "2026-06-16T00:00:00+00:00",
    "2026-06-23T00:00:00+00:00",
    "2026-06-30T00:00:00+00:00",
    "2026-07-07T00:00:00+00:00",
    "2026-07-14T00:00:00+00:00",
    "2026-07-21T00:00:00+00:00",
)
REGIME_ROBUST_VALIDATION_ENDS = (
    "2026-06-16T00:00:00+00:00",
    "2026-06-23T00:00:00+00:00",
    "2026-06-30T00:00:00+00:00",
    "2026-07-07T00:00:00+00:00",
    "2026-07-14T00:00:00+00:00",
    "2026-07-21T00:00:00+00:00",
    "2026-07-29T00:00:00+00:00",
)


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
    fixed_evaluation_seconds: tuple[int, ...]
    minimum_early_markets: int
    minimum_common_markets: int
    minimum_executable_markets: int
    minimum_coverage_uplift: float
    hard_confidence_floor: float
    minimum_hard_confident_error_count_reduction: int
    maximum_hard_confident_error_selected_rate_regression: float
    minimum_accuracy_uplift: float
    minimum_balanced_accuracy_uplift: float
    minimum_direction_recall_uplift: float
    minimum_wilson_lower_uplift: float
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
    walk_forward_validation_starts: tuple[str, ...] = ()
    walk_forward_validation_ends: tuple[str, ...] = ()
    rolling_calibration_days: int | None = None
    rolling_policy_days: int | None = None


def load_persistence_benchmark_config(path: Path) -> PersistenceBenchmarkConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    benchmark = raw["benchmark"]
    calibration = raw["calibration"]
    gates = raw["advancement_gates"]
    paths = raw["paths"]
    walk_forward = raw.get("walk_forward", {})
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
        fixed_evaluation_seconds=_parse_fixed_evaluation_seconds(
            benchmark.get(
                "fixed_evaluation_seconds",
                DEFAULT_FIXED_EVALUATION_SECONDS,
            ),
        ),
        minimum_early_markets=int(gates["minimum_early_markets"]),
        minimum_common_markets=int(gates["minimum_common_markets"]),
        minimum_executable_markets=int(gates["minimum_executable_markets"]),
        minimum_coverage_uplift=float(gates["minimum_coverage_uplift"]),
        hard_confidence_floor=float(gates.get("hard_confidence_floor", 0.95)),
        minimum_hard_confident_error_count_reduction=int(
            gates.get("minimum_hard_confident_error_count_reduction", 0)
        ),
        maximum_hard_confident_error_selected_rate_regression=float(
            gates.get(
                "maximum_hard_confident_error_selected_rate_regression",
                0.0,
            )
        ),
        minimum_accuracy_uplift=float(gates.get("minimum_accuracy_uplift", 0.0)),
        minimum_balanced_accuracy_uplift=float(
            gates.get("minimum_balanced_accuracy_uplift", 0.0)
        ),
        minimum_direction_recall_uplift=float(
            gates.get("minimum_direction_recall_uplift", 0.0)
        ),
        minimum_wilson_lower_uplift=float(
            gates.get("minimum_wilson_lower_uplift", 0.0)
        ),
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
        walk_forward_validation_starts=tuple(
            str(value) for value in walk_forward.get("validation_starts", ())
        ),
        walk_forward_validation_ends=tuple(
            str(value) for value in walk_forward.get("validation_ends", ())
        ),
        rolling_calibration_days=(
            int(walk_forward["rolling_calibration_days"])
            if "rolling_calibration_days" in walk_forward
            else None
        ),
        rolling_policy_days=(
            int(walk_forward["rolling_policy_days"])
            if "rolling_policy_days" in walk_forward
            else None
        ),
    )
    validate_persistence_benchmark_config(config)
    return config


def walk_forward_validation_windows(
    config: PersistenceBenchmarkConfig,
) -> tuple[tuple[datetime, datetime], ...]:
    starts = tuple(
        _parse_walk_forward_timestamp(value)
        for value in config.walk_forward_validation_starts
    )
    ends = tuple(
        _parse_walk_forward_timestamp(value)
        for value in config.walk_forward_validation_ends
    )
    if len(starts) != len(ends):
        raise ValueError("walk-forward validation start/end lists must have equal length")
    return tuple(zip(starts, ends, strict=True))


def _parse_walk_forward_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"invalid walk-forward ISO timestamp: {value}") from error
    if parsed.tzinfo is None:
        raise ValueError("walk-forward timestamps must include a UTC offset")
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("walk-forward timestamps must use UTC")
    return parsed.astimezone(UTC)


def _parse_fixed_evaluation_seconds(values: Any) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
    ):
        raise ValueError("fixed evaluation checkpoints must contain only integers")
    return tuple(values)


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
    elif config.profile == BOUNDARY_ALIGNMENT_PROFILE:
        if config.candidate_names != BOUNDARY_ALIGNMENT_CANDIDATES:
            raise ValueError(
                "boundary-alignment benchmark requires its frozen two-candidate matrix"
            )
        if config.row_weight_schedules:
            raise ValueError(
                "boundary-alignment benchmark preserves equal market weighting"
            )
    elif config.profile == BOUNDARY_REVERSAL_ACCURACY_PROFILE:
        if config.candidate_names != BOUNDARY_REVERSAL_ACCURACY_CANDIDATES:
            raise ValueError(
                "boundary-reversal accuracy benchmark requires its frozen "
                "two-candidate matrix"
            )
        if config.row_weight_schedules:
            raise ValueError(
                "boundary-reversal accuracy benchmark preserves equal market weighting"
            )
        if (
            config.hard_confidence_floor != 0.95
            or config.minimum_hard_confident_error_count_reduction != 1
            or config.maximum_hard_confident_error_selected_rate_regression != 0.0
        ):
            raise ValueError(
                "boundary-reversal accuracy benchmark requires the frozen "
                "hard-confident-error contract"
            )
    elif config.profile == RESIDUAL_ADMISSION_SOURCE_PROFILE:
        if config.candidate_names != RESIDUAL_ADMISSION_SOURCE_CANDIDATES:
            raise ValueError(
                "residual-admission source benchmark requires its frozen "
                "two-candidate matrix"
            )
        if config.row_weight_schedules:
            raise ValueError(
                "residual-admission source benchmark preserves equal market weighting"
            )
        if (
            config.hard_confidence_floor != 0.95
            or config.minimum_hard_confident_error_count_reduction != 1
            or config.maximum_hard_confident_error_selected_rate_regression != 0.0
        ):
            raise ValueError(
                "residual-admission source benchmark requires the frozen "
                "hard-confident-error contract"
            )
    elif config.profile == MATURE_REVERSAL_ACCURACY_PROFILE:
        if config.candidate_names != MATURE_REVERSAL_ACCURACY_CANDIDATES:
            raise ValueError(
                "mature-reversal accuracy benchmark requires its frozen "
                "two-candidate matrix"
            )
        if config.row_weight_schedules:
            raise ValueError(
                "mature-reversal accuracy benchmark preserves equal market weighting"
            )
        if (
            config.hard_confidence_floor != 0.95
            or config.minimum_hard_confident_error_count_reduction != 1
            or config.maximum_hard_confident_error_selected_rate_regression != 0.0
            or config.minimum_accuracy_uplift != 0.001
            or config.minimum_balanced_accuracy_uplift != 0.001
            or config.minimum_direction_recall_uplift != 0.0
            or config.minimum_wilson_lower_uplift != 0.001
            or config.minimum_coverage_uplift != 0.0
            or config.maximum_accuracy_regression != 0.0
            or config.maximum_balanced_accuracy_regression != 0.0
            or config.maximum_direction_recall_regression != 0.0
            or config.maximum_median_entry_seconds_regression != 0.0
        ):
            raise ValueError(
                "mature-reversal accuracy benchmark requires its frozen "
                "accuracy-uplift and diagnostic-tolerance contract"
            )
    elif config.profile == MATURE_REVERSAL_ORACLE_ACCURACY_PROFILE:
        if (
            config.candidate_names
            != MATURE_REVERSAL_ORACLE_ACCURACY_CANDIDATES
        ):
            raise ValueError(
                "mature-reversal oracle accuracy benchmark requires its "
                "frozen recency-matched two-candidate matrix"
            )
        if config.row_weight_schedules:
            raise ValueError(
                "mature-reversal oracle accuracy benchmark preserves equal "
                "total weight per market"
            )
        if (
            config.hard_confidence_floor != 0.95
            or config.minimum_hard_confident_error_count_reduction != 1
            or config.maximum_hard_confident_error_selected_rate_regression
            != 0.0
            or config.minimum_accuracy_uplift != 0.001
            or config.minimum_balanced_accuracy_uplift != 0.001
            or config.minimum_direction_recall_uplift != 0.0
            or config.minimum_wilson_lower_uplift != 0.001
            or config.minimum_coverage_uplift != 0.0
            or config.maximum_accuracy_regression != 0.0
            or config.maximum_balanced_accuracy_regression != 0.0
            or config.maximum_direction_recall_regression != 0.0
            or config.maximum_median_entry_seconds_regression != 0.0
        ):
            raise ValueError(
                "mature-reversal oracle accuracy benchmark requires its "
                "frozen accuracy-uplift and hard-error contract"
            )
    elif config.profile == REGIME_ROBUST_ACCURACY_PROFILE:
        if config.candidate_names != REGIME_ROBUST_ACCURACY_CANDIDATES:
            raise ValueError(
                "regime-robust accuracy benchmark requires its frozen "
                "five-candidate ablation matrix"
            )
        if config.row_weight_schedules:
            raise ValueError(
                "regime-robust accuracy benchmark preserves the untimed "
                "within-market row schedule"
            )
        if (
            config.hard_confidence_floor != 0.95
            or config.minimum_hard_confident_error_count_reduction != 1
            or config.maximum_hard_confident_error_selected_rate_regression != 0.0
            or config.minimum_accuracy_uplift != 0.001
            or config.minimum_balanced_accuracy_uplift != 0.001
            or config.minimum_direction_recall_uplift != 0.0
            or config.minimum_wilson_lower_uplift != 0.001
            or config.minimum_coverage_uplift != 0.0
            or config.maximum_accuracy_regression != 0.0
            or config.maximum_balanced_accuracy_regression != 0.0
            or config.maximum_direction_recall_regression != 0.0
            or config.maximum_median_entry_seconds_regression != 0.0
        ):
            raise ValueError(
                "regime-robust accuracy benchmark requires its frozen "
                "accuracy-uplift contract"
            )
    else:
        raise ValueError(f"unsupported persistence benchmark profile: {config.profile}")
    if config.control_candidate != config.candidate_names[0]:
        raise ValueError("the first candidate must remain the control")
    if config.evaluation_is_independent:
        raise ValueError("the consumed development range cannot be independent")
    if not config.evaluation_note:
        raise ValueError("evaluation_note is required")
    if config.quantity != 5.0:
        raise ValueError("execution economics require exactly five shares")
    if config.early_cutoff_second != 120:
        raise ValueError("the early-entry checkpoint must remain 120 seconds")
    expected_fixed_seconds = (
        ORACLE_EARLY_ENTRY_FIXED_EVALUATION_SECONDS
        if config.profile == MATURE_REVERSAL_ORACLE_ACCURACY_PROFILE
        else DEFAULT_FIXED_EVALUATION_SECONDS
    )
    if config.fixed_evaluation_seconds != expected_fixed_seconds:
        raise ValueError(
            "fixed evaluation checkpoints do not match the frozen profile"
        )
    if (
        config.minimum_early_markets < 500
        or config.minimum_common_markets < 500
        or config.minimum_executable_markets < 500
    ):
        raise ValueError("sample gates cannot be weakened below 500 markets")
    if config.profile in {
        MATURE_REVERSAL_ACCURACY_PROFILE,
        MATURE_REVERSAL_ORACLE_ACCURACY_PROFILE,
        REGIME_ROBUST_ACCURACY_PROFILE,
    }:
        weakens_contract = (
            config.minimum_coverage_uplift < 0.0
            or not 0.5 <= config.hard_confidence_floor <= 1.0
            or config.minimum_hard_confident_error_count_reduction < 1
            or config.maximum_hard_confident_error_selected_rate_regression < 0.0
            or config.minimum_accuracy_uplift <= 0.0
            or config.minimum_balanced_accuracy_uplift <= 0.0
            or config.minimum_direction_recall_uplift < 0.0
            or config.minimum_wilson_lower_uplift <= 0.0
            or config.maximum_accuracy_regression < 0.0
            or config.maximum_balanced_accuracy_regression < 0.0
            or config.maximum_direction_recall_regression < 0.0
            or config.maximum_median_entry_seconds_regression < 0.0
            or config.minimum_mean_direct_edge_per_share < 0.0
            or config.minimum_realized_net_per_share < 0.0
        )
    else:
        weakens_contract = (
            config.minimum_coverage_uplift <= 0.0
            or not 0.5 <= config.hard_confidence_floor <= 1.0
            or config.minimum_hard_confident_error_count_reduction < 0
            or config.maximum_hard_confident_error_selected_rate_regression < 0.0
            or config.minimum_accuracy_uplift < 0.0
            or config.minimum_balanced_accuracy_uplift < 0.0
            or config.minimum_direction_recall_uplift < 0.0
            or config.minimum_wilson_lower_uplift < 0.0
            or config.maximum_accuracy_regression > 0.0
            or config.maximum_balanced_accuracy_regression > 0.0
            or config.maximum_direction_recall_regression > 0.0
            or config.maximum_median_entry_seconds_regression > -5.0
            or config.minimum_mean_direct_edge_per_share < 0.0
            or config.minimum_realized_net_per_share < 0.0
        )
    if weakens_contract:
        raise ValueError("advancement gates weaken the frozen profile contract")
    if config.profile == REGIME_ROBUST_ACCURACY_PROFILE:
        windows = walk_forward_validation_windows(config)
        expected_windows = tuple(
            zip(
                (
                    _parse_walk_forward_timestamp(value)
                    for value in REGIME_ROBUST_VALIDATION_STARTS
                ),
                (
                    _parse_walk_forward_timestamp(value)
                    for value in REGIME_ROBUST_VALIDATION_ENDS
                ),
                strict=True,
            )
        )
        if (
            windows != expected_windows
            or config.rolling_calibration_days != 7
            or config.rolling_policy_days != 7
        ):
            raise ValueError(
                "regime-robust accuracy benchmark requires the frozen seven-window "
                "rolling calibration and policy contract"
            )
        for previous, current in pairwise(windows):
            if previous[1] > current[0]:
                raise ValueError("walk-forward validation windows must not overlap")
    elif (
        config.walk_forward_validation_starts
        or config.walk_forward_validation_ends
        or config.rolling_calibration_days is not None
        or config.rolling_policy_days is not None
    ):
        raise ValueError(
            "rolling walk-forward fields are reserved for regime-robust accuracy"
        )
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
        config.profile == MATURE_REVERSAL_ORACLE_ACCURACY_PROFILE
        and (
            core.data.source_contract != CORE_ORACLE_SOURCE_CONTRACT
            or core.data.sample_interval_seconds != 5
            or core.data.min_seconds_after_open != 120
            or core.data.min_seconds_before_close != 160
        )
    ):
        raise ValueError(
            "mature-reversal oracle accuracy requires the frozen causal "
            "oracle contract and exact 120-140 second window"
        )
    if config.profile == MATURE_REVERSAL_ORACLE_ACCURACY_PROFILE:
        _validate_oracle_accuracy_core_contract(core)
    if config.profile == BOUNDARY_REVERSAL_ACCURACY_PROFILE:
        expected_range = (
            "2026-03-21T00:00:00+00:00",
            "2026-07-29T00:00:00+00:00",
            130,
        )
    elif config.profile == RESIDUAL_ADMISSION_SOURCE_PROFILE:
        expected_range = (
            "2026-03-21T00:00:00+00:00",
            "2026-07-21T00:00:00+00:00",
            122,
        )
    elif config.profile in {
        MATURE_REVERSAL_ACCURACY_PROFILE,
        MATURE_REVERSAL_ORACLE_ACCURACY_PROFILE,
        REGIME_ROBUST_ACCURACY_PROFILE,
    }:
        expected_range = (
            "2026-03-21T00:00:00+00:00",
            "2026-07-29T00:00:00+00:00",
            130,
        )
    else:
        expected_range = (
            "2026-04-21T00:00:00+00:00",
            "2026-07-20T00:00:00+00:00",
            90,
        )
    if (
        core.data.range_start.isoformat() != expected_range[0]
        or core.data.range_end.isoformat() != expected_range[1]
        or (core.data.range_end - core.data.range_start).days != expected_range[2]
    ):
        raise ValueError(
            "persistence benchmark training range does not match its frozen profile"
        )
    if config.profile in {
        BOUNDARY_REVERSAL_ACCURACY_PROFILE,
        MATURE_REVERSAL_ACCURACY_PROFILE,
        MATURE_REVERSAL_ORACLE_ACCURACY_PROFILE,
        REGIME_ROBUST_ACCURACY_PROFILE,
    }:
        expected_split = (
            "2026-03-21T00:00:00+00:00",
            "2026-07-14T00:00:00+00:00",
            "2026-07-14T00:00:00+00:00",
            "2026-07-21T00:00:00+00:00",
            "2026-07-21T00:00:00+00:00",
            "2026-07-29T00:00:00+00:00",
        )
        observed_split = (
            core.split.development_start.isoformat(),
            core.split.development_end.isoformat(),
            core.split.probability_calibration_start.isoformat(),
            core.split.probability_calibration_end.isoformat(),
            core.split.policy_selection_start.isoformat(),
            core.split.policy_selection_end.isoformat(),
        )
        expected_validation_windows = tuple(
            zip(
                REGIME_ROBUST_VALIDATION_STARTS[:5],
                REGIME_ROBUST_VALIDATION_ENDS[:5],
                strict=True,
            )
        )
        observed_validation_windows = tuple(
            (start.isoformat(), end.isoformat()) for start, end in core.split.validation_windows
        )
        if (
            observed_split != expected_split
            or observed_validation_windows != expected_validation_windows
            or core.data.strict_final_price_audit
        ):
            raise ValueError(
                "boundary-reversal accuracy split or source-selection contract changed"
            )
    elif config.profile == RESIDUAL_ADMISSION_SOURCE_PROFILE:
        expected_split = (
            "2026-03-21T00:00:00+00:00",
            "2026-07-06T00:00:00+00:00",
            "2026-07-06T00:00:00+00:00",
            "2026-07-13T00:00:00+00:00",
            "2026-07-13T00:00:00+00:00",
            "2026-07-21T00:00:00+00:00",
        )
        observed_split = (
            core.split.development_start.isoformat(),
            core.split.development_end.isoformat(),
            core.split.probability_calibration_start.isoformat(),
            core.split.probability_calibration_end.isoformat(),
            core.split.policy_selection_start.isoformat(),
            core.split.policy_selection_end.isoformat(),
        )
        expected_validation_windows = (
            (
                "2026-05-18T00:00:00+00:00",
                "2026-05-25T00:00:00+00:00",
            ),
            (
                "2026-05-25T00:00:00+00:00",
                "2026-06-01T00:00:00+00:00",
            ),
            (
                "2026-06-01T00:00:00+00:00",
                "2026-06-08T00:00:00+00:00",
            ),
            (
                "2026-06-08T00:00:00+00:00",
                "2026-06-15T00:00:00+00:00",
            ),
            (
                "2026-06-15T00:00:00+00:00",
                "2026-06-22T00:00:00+00:00",
            ),
            (
                "2026-06-22T00:00:00+00:00",
                "2026-06-29T00:00:00+00:00",
            ),
            (
                "2026-06-29T00:00:00+00:00",
                "2026-07-06T00:00:00+00:00",
            ),
        )
        observed_validation_windows = tuple(
            (start.isoformat(), end.isoformat())
            for start, end in core.split.validation_windows
        )
        if (
            observed_split != expected_split
            or observed_validation_windows != expected_validation_windows
            or core.data.strict_final_price_audit
        ):
            raise ValueError(
                "residual-admission source split or source-selection contract changed"
            )
        _validate_residual_admission_core_contract(core)
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
    expected_fold_count = (
        7 if config.profile == RESIDUAL_ADMISSION_SOURCE_PROFILE else 5
    )
    expected_random_seed = (
        20260730
        if config.profile == MATURE_REVERSAL_ORACLE_ACCURACY_PROFILE
        else 20260726
    )
    if (
        core.model.confidence_min != 0.87
        or core.model.confidence_max != 0.91
        or core.model.confidence_step != 0.01
        or core.model.random_seed != expected_random_seed
        or len(core.model.histogram_candidates) != 4
        or len(core.split.validation_windows) != expected_fold_count
    ):
        raise ValueError("model search, seed, threshold, or fold contract changed")


def _validate_oracle_accuracy_core_contract(core: Any) -> None:
    histogram_search = tuple(
        (
            candidate.learning_rate,
            candidate.max_iter,
            candidate.max_leaf_nodes,
            candidate.min_samples_leaf,
            candidate.l2_regularization,
        )
        for candidate in core.model.histogram_candidates
    )
    expected_histogram_search = (
        (0.05, 160, 15, 100, 0.10),
        (0.05, 220, 31, 100, 1.0),
        (0.08, 160, 15, 150, 1.0),
        (0.08, 220, 31, 150, 2.0),
    )
    gates = core.gates
    gate_contract = (
        gates.target_accuracy,
        gates.target_wilson_lower,
        gates.target_balanced_accuracy,
        gates.minimum_direction_recall,
        gates.minimum_coverage,
        gates.minimum_holdout_markets,
        gates.maximum_walk_forward_holdout_gap,
        gates.minimum_same_time_path_uplift,
        gates.minimum_nonnegative_uplift_folds,
        gates.maximum_ece,
        gates.bootstrap_resamples,
    )
    expected_gate_contract = (
        0.874,
        0.865,
        0.874,
        0.874,
        0.60,
        500,
        0.05,
        0.0,
        5,
        0.05,
        10_000,
    )
    if (
        histogram_search != expected_histogram_search
        or gate_contract != expected_gate_contract
    ):
        raise ValueError(
            "mature-reversal oracle accuracy requires the frozen histogram "
            "search and absolute accuracy gates"
        )


def _validate_residual_admission_core_contract(core: Any) -> None:
    histogram_search = tuple(
        (
            candidate.learning_rate,
            candidate.max_iter,
            candidate.max_leaf_nodes,
            candidate.min_samples_leaf,
            candidate.l2_regularization,
        )
        for candidate in core.model.histogram_candidates
    )
    expected_histogram_search = (
        (0.05, 160, 15, 100, 0.10),
        (0.05, 220, 31, 100, 1.0),
        (0.08, 160, 15, 150, 1.0),
        (0.08, 220, 31, 150, 2.0),
    )
    gates = core.gates
    gate_contract = (
        gates.target_accuracy,
        gates.target_wilson_lower,
        gates.target_balanced_accuracy,
        gates.minimum_direction_recall,
        gates.minimum_coverage,
        gates.minimum_holdout_markets,
        gates.maximum_walk_forward_holdout_gap,
        gates.minimum_same_time_path_uplift,
        gates.minimum_nonnegative_uplift_folds,
        gates.maximum_ece,
        gates.bootstrap_resamples,
    )
    if (
        core.data.sample_interval_seconds != 5
        or core.data.min_seconds_after_open != 60
        or core.data.min_seconds_before_close != 60
        or core.model.c_candidates != (0.01, 0.1, 1.0, 10.0)
        or histogram_search != expected_histogram_search
        or core.model.candidate_names
        != (
            "histogram_enriched",
            "histogram_early_weighted",
            "histogram_early_weighted_moderate",
            "histogram_early_90_120",
        )
        or gate_contract
        != (
            0.874,
            0.865,
            0.874,
            0.874,
            0.55,
            1000,
            0.05,
            0.0,
            7,
            0.05,
            10000,
        )
        or core.compute.max_parallel_fits != 2
        or core.compute.threads_per_fit != 1
        or core.compute.polars_threads != 6
    ):
        raise ValueError(
            "residual-admission source search, cadence, gates, or compute contract changed"
        )


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
