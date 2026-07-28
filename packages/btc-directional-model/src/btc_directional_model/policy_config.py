from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .persistence_benchmark import SAVED_POLICY_PROBABILITY_SCHEMA_VERSION
from .persistence_config import ACCURACY_TIMING_CANDIDATES


@dataclass(frozen=True)
class PolicyThresholdBand:
    name: str
    start_second: int
    end_second_exclusive: int


@dataclass(frozen=True)
class PolicyAdvancementGates:
    minimum_accuracy: float
    minimum_balanced_accuracy: float
    minimum_direction_recall: float
    minimum_wilson_lower_95: float
    maximum_expected_calibration_error: float
    minimum_coverage: float
    minimum_selected_markets: int
    minimum_coverage_uplift: float
    maximum_accuracy_regression: float
    maximum_balanced_accuracy_regression: float
    maximum_direction_recall_regression: float
    maximum_median_entry_second: float
    minimum_median_entry_improvement_seconds: float
    require_every_fold: bool


@dataclass(frozen=True)
class SavedPolicyBenchmarkConfig:
    source_path: Path
    package_root: Path
    probability_manifest: Path
    control_candidate: str
    candidate_names: tuple[str, ...]
    evaluation_note: str
    evaluation_is_independent: bool
    threshold_candidates: tuple[float, ...]
    bands: tuple[PolicyThresholdBand, ...]
    gates: PolicyAdvancementGates
    runs: Path


def load_saved_policy_benchmark_config(
    path: Path,
) -> SavedPolicyBenchmarkConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    benchmark = raw["benchmark"]
    policy = raw["policy"]
    gates = raw["advancement_gates"]
    paths = raw["paths"]
    config = SavedPolicyBenchmarkConfig(
        source_path=source_path,
        package_root=package_root,
        probability_manifest=package_root / str(benchmark["probability_manifest"]),
        control_candidate=str(benchmark["control_candidate"]),
        candidate_names=tuple(str(value) for value in benchmark["candidate_names"]),
        evaluation_note=str(benchmark["evaluation_note"]).strip(),
        evaluation_is_independent=bool(benchmark["evaluation_is_independent"]),
        threshold_candidates=tuple(float(value) for value in policy["threshold_candidates"]),
        bands=tuple(
            PolicyThresholdBand(
                name=str(band["name"]),
                start_second=int(band["start_second"]),
                end_second_exclusive=int(band["end_second_exclusive"]),
            )
            for band in policy["bands"]
        ),
        gates=PolicyAdvancementGates(
            minimum_accuracy=float(gates["minimum_accuracy"]),
            minimum_balanced_accuracy=float(gates["minimum_balanced_accuracy"]),
            minimum_direction_recall=float(gates["minimum_direction_recall"]),
            minimum_wilson_lower_95=float(gates["minimum_wilson_lower_95"]),
            maximum_expected_calibration_error=float(gates["maximum_expected_calibration_error"]),
            minimum_coverage=float(gates["minimum_coverage"]),
            minimum_selected_markets=int(gates["minimum_selected_markets"]),
            minimum_coverage_uplift=float(gates["minimum_coverage_uplift"]),
            maximum_accuracy_regression=float(gates["maximum_accuracy_regression"]),
            maximum_balanced_accuracy_regression=float(
                gates["maximum_balanced_accuracy_regression"]
            ),
            maximum_direction_recall_regression=float(gates["maximum_direction_recall_regression"]),
            maximum_median_entry_second=float(gates["maximum_median_entry_second"]),
            minimum_median_entry_improvement_seconds=float(
                gates["minimum_median_entry_improvement_seconds"]
            ),
            require_every_fold=bool(gates["require_every_fold"]),
        ),
        runs=package_root / str(paths["runs"]),
    )
    validate_saved_policy_benchmark_config(config)
    return config


def validate_saved_policy_benchmark_config(
    config: SavedPolicyBenchmarkConfig,
) -> None:
    if not config.probability_manifest.is_file():
        raise ValueError(f"saved probability manifest is missing: {config.probability_manifest}")
    manifest = json.loads(config.probability_manifest.read_text())
    if manifest.get("schema_version") != SAVED_POLICY_PROBABILITY_SCHEMA_VERSION:
        raise ValueError("saved probability manifest schema is unsupported")
    if config.candidate_names != ACCURACY_TIMING_CANDIDATES:
        raise ValueError("saved policy benchmark requires the frozen accuracy-timing matrix")
    if config.control_candidate != ACCURACY_TIMING_CANDIDATES[0]:
        raise ValueError("histogram_enriched must remain the policy control")
    if tuple(manifest.get("candidate_names", ())) != config.candidate_names:
        raise ValueError("saved probability candidate order does not match policy config")
    if manifest.get("control_candidate") != config.control_candidate:
        raise ValueError("saved probability control does not match policy config")
    if config.evaluation_is_independent:
        raise ValueError("saved validation probabilities are development evidence")
    if not config.evaluation_note:
        raise ValueError("evaluation_note is required")
    expected_bands = (
        PolicyThresholdBand("60-89", 60, 90),
        PolicyThresholdBand("90-119", 90, 120),
        PolicyThresholdBand("120-179", 120, 180),
        PolicyThresholdBand("180-240", 180, 241),
    )
    if config.bands != expected_bands:
        raise ValueError("policy selection requires four frozen causal bands")
    thresholds = config.threshold_candidates
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or tuple(sorted(thresholds)) != thresholds
        or any(not math.isfinite(value) or value < 0.5 or value > 1.0 for value in thresholds)
    ):
        raise ValueError("policy threshold candidates must be unique, ordered probabilities")
    if len(thresholds) ** len(config.bands) > 100_000:
        raise ValueError("policy threshold matrix exceeds the compute budget")
    gates = config.gates
    if (
        gates.minimum_accuracy < 0.874
        or gates.minimum_balanced_accuracy < 0.874
        or gates.minimum_direction_recall < 0.874
        or gates.minimum_wilson_lower_95 < 0.865
        or gates.maximum_expected_calibration_error > 0.05
        or gates.minimum_coverage < 0.55
        or gates.minimum_selected_markets < 500
        or gates.minimum_coverage_uplift <= 0.0
        or gates.maximum_accuracy_regression > 0.0
        or gates.maximum_balanced_accuracy_regression > 0.0
        or gates.maximum_direction_recall_regression > 0.0
        or gates.maximum_median_entry_second > 125.0
        or gates.minimum_median_entry_improvement_seconds < 5.0
        or not gates.require_every_fold
    ):
        raise ValueError("saved policy advancement gates weaken the frozen contract")


def policy_config_to_dict(
    config: SavedPolicyBenchmarkConfig,
) -> dict[str, Any]:
    payload = asdict(config)
    for key in (
        "source_path",
        "package_root",
        "probability_manifest",
        "runs",
    ):
        payload[key] = str(payload[key])
    return payload
