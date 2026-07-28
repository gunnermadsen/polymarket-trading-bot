from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .persistence_benchmark import SAVED_POLICY_PROBABILITY_SCHEMA_VERSION
from .persistence_config import ACCURACY_TIMING_CANDIDATES, ACCURACY_TIMING_PROFILE
from .policy_config import PolicyThresholdBand

FREQUENCY_POLICY_OBJECTIVE = "frequency"
SINGLE_FROZEN_POLICY_MODE = "single_frozen"
FREQUENCY_POLICY_CANDIDATES = (
    "histogram_enriched",
    "histogram_path_persistence_time_calibrated_60_120",
)
FIXED_FREQUENCY_CHECKPOINTS = (60, 90, 120, 180, 240)


@dataclass(frozen=True)
class FrequencyPolicyAdvancementGates:
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
    minimum_common_checkpoint_markets: int
    minimum_executable_markets: int
    minimum_mean_direct_edge_per_share: float
    minimum_realized_net_per_share: float
    require_every_fold: bool


@dataclass(frozen=True)
class FrequencyPolicyBenchmarkConfig:
    source_path: Path
    package_root: Path
    probability_manifest: Path
    qualification_objective: str
    policy_selection_mode: str
    policy_anchor_fold: int
    control_candidate: str
    candidate_names: tuple[str, ...]
    evaluation_note: str
    evaluation_is_independent: bool
    quantity: float
    threshold_candidates: tuple[float, ...]
    bands: tuple[PolicyThresholdBand, ...]
    gates: FrequencyPolicyAdvancementGates
    execution_evidence: Path
    runs: Path


def load_frequency_policy_benchmark_config(
    path: Path,
) -> FrequencyPolicyBenchmarkConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    benchmark = raw["benchmark"]
    policy = raw["policy"]
    gates = raw["advancement_gates"]
    paths = raw["paths"]
    config = FrequencyPolicyBenchmarkConfig(
        source_path=source_path,
        package_root=package_root,
        probability_manifest=package_root / str(benchmark["probability_manifest"]),
        qualification_objective=str(benchmark["qualification_objective"]),
        policy_selection_mode=str(benchmark["policy_selection_mode"]),
        policy_anchor_fold=int(benchmark["policy_anchor_fold"]),
        control_candidate=str(benchmark["control_candidate"]),
        candidate_names=tuple(str(value) for value in benchmark["candidate_names"]),
        evaluation_note=str(benchmark["evaluation_note"]).strip(),
        evaluation_is_independent=bool(benchmark["evaluation_is_independent"]),
        quantity=float(benchmark["quantity"]),
        threshold_candidates=tuple(float(value) for value in policy["threshold_candidates"]),
        bands=tuple(
            PolicyThresholdBand(
                name=str(band["name"]),
                start_second=int(band["start_second"]),
                end_second_exclusive=int(band["end_second_exclusive"]),
            )
            for band in policy["bands"]
        ),
        gates=FrequencyPolicyAdvancementGates(
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
            minimum_common_checkpoint_markets=int(gates["minimum_common_checkpoint_markets"]),
            minimum_executable_markets=int(gates["minimum_executable_markets"]),
            minimum_mean_direct_edge_per_share=float(gates["minimum_mean_direct_edge_per_share"]),
            minimum_realized_net_per_share=float(gates["minimum_realized_net_per_share"]),
            require_every_fold=bool(gates["require_every_fold"]),
        ),
        execution_evidence=package_root / str(paths["execution_evidence"]),
        runs=package_root / str(paths["runs"]),
    )
    validate_frequency_policy_benchmark_config(config)
    return config


def validate_frequency_policy_benchmark_config(
    config: FrequencyPolicyBenchmarkConfig,
) -> None:
    if not config.probability_manifest.is_file():
        raise ValueError(f"saved probability manifest is missing: {config.probability_manifest}")
    manifest = json.loads(config.probability_manifest.read_text())
    if manifest.get("schema_version") != SAVED_POLICY_PROBABILITY_SCHEMA_VERSION:
        raise ValueError("saved probability manifest schema is unsupported")
    if manifest.get("source_benchmark_profile") != ACCURACY_TIMING_PROFILE:
        raise ValueError("frequency qualification requires accuracy-timing probability evidence")
    if config.qualification_objective != FREQUENCY_POLICY_OBJECTIVE:
        raise ValueError("frequency policy qualification_objective must remain frequency")
    if config.policy_selection_mode != SINGLE_FROZEN_POLICY_MODE:
        raise ValueError("frequency policy selection must remain single_frozen")
    if config.policy_anchor_fold != 0:
        raise ValueError("the earliest chronological fold must anchor the frozen policy")
    if config.candidate_names != FREQUENCY_POLICY_CANDIDATES:
        raise ValueError("frequency policy requires the frozen control and 60-120 candidate")
    if config.control_candidate != FREQUENCY_POLICY_CANDIDATES[0]:
        raise ValueError("histogram_enriched must remain the frequency control")
    manifest_candidates = tuple(manifest.get("candidate_names", ()))
    if manifest_candidates != ACCURACY_TIMING_CANDIDATES:
        raise ValueError("saved probability evidence lost the frozen accuracy-timing matrix")
    if any(name not in manifest_candidates for name in config.candidate_names):
        raise ValueError("frequency candidate is missing from saved probability evidence")
    if manifest.get("control_candidate") != config.control_candidate:
        raise ValueError("saved probability control does not match frequency config")
    if config.evaluation_is_independent:
        raise ValueError("saved validation probabilities are development evidence")
    if not config.evaluation_note:
        raise ValueError("evaluation_note is required")
    if config.quantity != 5.0:
        raise ValueError("frequency economics require exactly five shares")
    expected_bands = (
        PolicyThresholdBand("60-89", 60, 90),
        PolicyThresholdBand("90-119", 90, 120),
        PolicyThresholdBand("120-179", 120, 180),
        PolicyThresholdBand("180-240", 180, 241),
    )
    if config.bands != expected_bands:
        raise ValueError("frequency policy requires four frozen causal bands")
    thresholds = config.threshold_candidates
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or tuple(sorted(thresholds)) != thresholds
        or any(not math.isfinite(value) or value < 0.5 or value > 1.0 for value in thresholds)
    ):
        raise ValueError("frequency thresholds must be unique, ordered probabilities")
    if len(thresholds) ** len(config.bands) > 100_000:
        raise ValueError("frequency threshold matrix exceeds the compute budget")
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
        or gates.minimum_common_checkpoint_markets < 500
        or gates.minimum_executable_markets < 500
        or gates.minimum_mean_direct_edge_per_share < 0.0
        or gates.minimum_realized_net_per_share < 0.0
        or not gates.require_every_fold
    ):
        raise ValueError("frequency advancement gates weaken the frozen quality contract")
    if not (config.execution_evidence / "manifest.json").is_file():
        raise ValueError("frequency execution-evidence manifest is missing")


def frequency_policy_config_to_dict(
    config: FrequencyPolicyBenchmarkConfig,
) -> dict[str, Any]:
    payload = asdict(config)
    for key in (
        "source_path",
        "package_root",
        "probability_manifest",
        "execution_evidence",
        "runs",
    ):
        payload[key] = str(payload[key])
    return payload
