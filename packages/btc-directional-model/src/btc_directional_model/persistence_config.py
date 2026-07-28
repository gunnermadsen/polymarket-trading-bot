from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .core_config import load_core_config

PERSISTENCE_CANDIDATES = (
    "histogram_enriched",
    "histogram_path_persistence",
    "histogram_path_persistence_prewindow",
    "histogram_path_persistence_time_calibrated",
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
    core_config: Path
    control_candidate: str
    candidate_names: tuple[str, ...]
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
        core_config=package_root / str(benchmark["core_config"]),
        control_candidate=str(benchmark["control_candidate"]),
        candidate_names=tuple(str(value) for value in benchmark["candidate_names"]),
        evaluation_note=str(benchmark["evaluation_note"]).strip(),
        evaluation_is_independent=bool(benchmark["evaluation_is_independent"]),
        quantity=float(benchmark["quantity"]),
        early_cutoff_second=int(benchmark["early_cutoff_second"]),
        minimum_early_markets=int(gates["minimum_early_markets"]),
        minimum_common_markets=int(gates["minimum_common_markets"]),
        minimum_executable_markets=int(gates["minimum_executable_markets"]),
        minimum_coverage_uplift=float(gates["minimum_coverage_uplift"]),
        maximum_accuracy_regression=float(gates["maximum_accuracy_regression"]),
        maximum_balanced_accuracy_regression=float(
            gates["maximum_balanced_accuracy_regression"]
        ),
        maximum_direction_recall_regression=float(
            gates["maximum_direction_recall_regression"]
        ),
        maximum_median_entry_seconds_regression=float(
            gates["maximum_median_entry_seconds_regression"]
        ),
        minimum_mean_direct_edge_per_share=float(
            gates["minimum_mean_direct_edge_per_share"]
        ),
        minimum_realized_net_per_share=float(
            gates["minimum_realized_net_per_share"]
        ),
        calibration_bands=tuple(
            CalibrationBand(
                name=str(band["name"]),
                start_second=int(band["start_second"]),
                end_second_exclusive=int(band["end_second_exclusive"]),
            )
            for band in calibration["bands"]
        ),
        minimum_calibration_rows_per_band=int(
            calibration["minimum_rows_per_band"]
        ),
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
    if config.candidate_names != PERSISTENCE_CANDIDATES:
        raise ValueError(
            "persistence benchmark requires the frozen four-candidate matrix"
        )
    if config.control_candidate != PERSISTENCE_CANDIDATES[0]:
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
        raise ValueError(
            "persistence benchmark requires exact [2026-04-21, 2026-07-20)"
        )
    if (
        core.split.holdout_start != core.data.range_end
        or core.split.holdout_end != core.data.range_end
    ):
        raise ValueError("the in-range holdout must remain disabled")
    independent_start = core.split.independent_holdout_start
    independent_end = core.split.independent_holdout_end
    if (
        independent_start is None
        or independent_end is None
        or independent_start.isoformat() != "2026-07-21T00:00:00+00:00"
        or independent_end.isoformat() != "2026-08-04T00:00:00+00:00"
    ):
        raise ValueError("the untouched July 21-August 4 holdout contract changed")
    if (
        core.model.confidence_min != 0.87
        or core.model.confidence_max != 0.91
        or core.model.confidence_step != 0.01
        or core.model.random_seed != 20260726
        or len(core.model.histogram_candidates) != 4
        or len(core.split.validation_windows) != 5
    ):
        raise ValueError("model search, seed, threshold, or fold contract changed")


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
