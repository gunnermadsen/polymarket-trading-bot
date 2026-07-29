from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .persistence_benchmark import SAVED_POLICY_PROBABILITY_SCHEMA_VERSION
from .policy_config import PolicyThresholdBand

CORRECTNESS_ADMISSION_PROFILE = "correctness_admission"
BOUNDARY_ALIGNMENT_PROFILE = "boundary_alignment"
ADMISSION_CONTROL_CANDIDATE = "histogram_enriched"
ADMISSION_BASE_CANDIDATE = "histogram_boundary_enriched"
ADMISSION_SELECTOR_CANDIDATE = "histogram_boundary_correctness_admission"
ADMISSION_SOURCE_CANDIDATES = (
    ADMISSION_CONTROL_CANDIDATE,
    ADMISSION_BASE_CANDIDATE,
)
ADMISSION_EVALUATION_FOLDS = (2, 3, 4)
ADMISSION_REQUIRED_DEPLOYMENT_FOLDS = 5


@dataclass(frozen=True)
class AdmissionSelectorConfig:
    regularization_c: float
    minimum_calibration_rows_per_band: int
    minimum_calibration_markets_per_band: int
    admission_floor: float
    random_seed: int


@dataclass(frozen=True)
class AdmissionAdvancementGates:
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
    minimum_common_checkpoint_markets: int
    minimum_executable_markets: int
    minimum_mean_direct_edge_per_share: float
    minimum_realized_net_per_share: float
    require_every_fold: bool
    required_deployment_validation_folds: int


@dataclass(frozen=True)
class AdmissionBenchmarkConfig:
    source_path: Path
    package_root: Path
    profile: str
    probability_manifest: Path
    core_config: Path
    control_candidate: str
    base_candidate: str
    selector_candidate: str
    evaluation_folds: tuple[int, ...]
    evaluation_note: str
    evaluation_is_independent: bool
    quantity: float
    selector: AdmissionSelectorConfig
    threshold_candidates: tuple[float, ...]
    bands: tuple[PolicyThresholdBand, ...]
    gates: AdmissionAdvancementGates
    execution_evidence: Path
    runs: Path

    @property
    def candidate_names(self) -> tuple[str, ...]:
        return (
            self.control_candidate,
            self.base_candidate,
            self.selector_candidate,
        )


def load_admission_benchmark_config(path: Path) -> AdmissionBenchmarkConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    benchmark = raw["benchmark"]
    selector = raw["selector"]
    policy = raw["policy"]
    gates = raw["advancement_gates"]
    paths = raw["paths"]
    config = AdmissionBenchmarkConfig(
        source_path=source_path,
        package_root=package_root,
        profile=str(benchmark["profile"]),
        probability_manifest=package_root / str(benchmark["probability_manifest"]),
        core_config=package_root / str(benchmark["core_config"]),
        control_candidate=str(benchmark["control_candidate"]),
        base_candidate=str(benchmark["base_candidate"]),
        selector_candidate=str(benchmark["selector_candidate"]),
        evaluation_folds=tuple(int(value) for value in benchmark["evaluation_folds"]),
        evaluation_note=str(benchmark["evaluation_note"]).strip(),
        evaluation_is_independent=bool(benchmark["evaluation_is_independent"]),
        quantity=float(benchmark["quantity"]),
        selector=AdmissionSelectorConfig(
            regularization_c=float(selector["regularization_c"]),
            minimum_calibration_rows_per_band=int(
                selector["minimum_calibration_rows_per_band"]
            ),
            minimum_calibration_markets_per_band=int(
                selector["minimum_calibration_markets_per_band"]
            ),
            admission_floor=float(selector["admission_floor"]),
            random_seed=int(selector["random_seed"]),
        ),
        threshold_candidates=tuple(
            float(value) for value in policy["threshold_candidates"]
        ),
        bands=tuple(
            PolicyThresholdBand(
                name=str(band["name"]),
                start_second=int(band["start_second"]),
                end_second_exclusive=int(band["end_second_exclusive"]),
            )
            for band in policy["bands"]
        ),
        gates=AdmissionAdvancementGates(
            minimum_accuracy=float(gates["minimum_accuracy"]),
            minimum_balanced_accuracy=float(gates["minimum_balanced_accuracy"]),
            minimum_direction_recall=float(gates["minimum_direction_recall"]),
            minimum_wilson_lower_95=float(gates["minimum_wilson_lower_95"]),
            maximum_expected_calibration_error=float(
                gates["maximum_expected_calibration_error"]
            ),
            minimum_coverage=float(gates["minimum_coverage"]),
            minimum_selected_markets=int(gates["minimum_selected_markets"]),
            minimum_coverage_uplift=float(gates["minimum_coverage_uplift"]),
            maximum_accuracy_regression=float(gates["maximum_accuracy_regression"]),
            maximum_balanced_accuracy_regression=float(
                gates["maximum_balanced_accuracy_regression"]
            ),
            maximum_direction_recall_regression=float(
                gates["maximum_direction_recall_regression"]
            ),
            maximum_median_entry_second=float(
                gates["maximum_median_entry_second"]
            ),
            minimum_median_entry_improvement_seconds=float(
                gates["minimum_median_entry_improvement_seconds"]
            ),
            minimum_common_checkpoint_markets=int(
                gates["minimum_common_checkpoint_markets"]
            ),
            minimum_executable_markets=int(
                gates["minimum_executable_markets"]
            ),
            minimum_mean_direct_edge_per_share=float(
                gates["minimum_mean_direct_edge_per_share"]
            ),
            minimum_realized_net_per_share=float(
                gates["minimum_realized_net_per_share"]
            ),
            require_every_fold=bool(gates["require_every_fold"]),
            required_deployment_validation_folds=int(
                gates["required_deployment_validation_folds"]
            ),
        ),
        execution_evidence=package_root / str(paths["execution_evidence"]),
        runs=package_root / str(paths["runs"]),
    )
    validate_admission_benchmark_config(config)
    return config


def validate_admission_benchmark_config(config: AdmissionBenchmarkConfig) -> None:
    if config.profile != CORRECTNESS_ADMISSION_PROFILE:
        raise ValueError("admission benchmark profile must remain correctness_admission")
    if "__BOUNDARY_RUN_ID__" in str(config.probability_manifest):
        raise ValueError(
            "replace __BOUNDARY_RUN_ID__ with the completed boundary-alignment run id"
        )
    if not config.probability_manifest.is_file():
        raise ValueError(
            f"saved probability manifest is missing: {config.probability_manifest}"
        )
    manifest = json.loads(config.probability_manifest.read_text())
    if manifest.get("schema_version") != SAVED_POLICY_PROBABILITY_SCHEMA_VERSION:
        raise ValueError("saved probability manifest schema is unsupported")
    if manifest.get("source_benchmark_profile") != BOUNDARY_ALIGNMENT_PROFILE:
        raise ValueError(
            "correctness admission requires boundary_alignment probability evidence"
        )
    if tuple(manifest.get("candidate_names", ())) != ADMISSION_SOURCE_CANDIDATES:
        raise ValueError(
            "correctness admission requires the frozen boundary-alignment matrix"
        )
    if manifest.get("control_candidate") != ADMISSION_CONTROL_CANDIDATE:
        raise ValueError("histogram_enriched must remain the admission control")
    if config.control_candidate != ADMISSION_CONTROL_CANDIDATE:
        raise ValueError("histogram_enriched must remain the admission control")
    if config.base_candidate != ADMISSION_BASE_CANDIDATE:
        raise ValueError(
            "histogram_boundary_enriched must remain the admission base model"
        )
    if config.selector_candidate != ADMISSION_SELECTOR_CANDIDATE:
        raise ValueError("correctness selector identity is frozen")
    if config.evaluation_folds != ADMISSION_EVALUATION_FOLDS:
        raise ValueError("correctness admission requires rolling evaluation folds 2, 3, 4")
    if int(manifest.get("fold_count", 0)) < ADMISSION_REQUIRED_DEPLOYMENT_FOLDS:
        raise ValueError("boundary-alignment evidence must retain all five source folds")
    for candidate_name in ADMISSION_SOURCE_CANDIDATES:
        candidate = manifest.get("candidates", {}).get(candidate_name)
        if candidate is None:
            raise ValueError(f"saved probability evidence is missing {candidate_name}")
        fold_indexes = tuple(
            sorted(int(record["fold_index"]) for record in candidate.get("folds", ()))
        )
        if fold_indexes != tuple(range(int(manifest["fold_count"]))):
            raise ValueError(f"{candidate_name} saved probability folds are incomplete")
    if config.evaluation_is_independent:
        raise ValueError("rolling selector validation is consumed development evidence")
    if not config.evaluation_note:
        raise ValueError("evaluation_note is required")
    if config.quantity != 5.0:
        raise ValueError("admission economics require exactly five shares")
    if not config.core_config.is_file():
        raise ValueError(f"boundary core config is missing: {config.core_config}")
    if not (config.execution_evidence / "manifest.json").is_file():
        raise ValueError("admission execution-evidence manifest is missing")

    selector = config.selector
    if (
        not math.isfinite(selector.regularization_c)
        or selector.regularization_c <= 0.0
    ):
        raise ValueError("selector regularization_c must be positive")
    if selector.minimum_calibration_rows_per_band < 100:
        raise ValueError("selector calibration bands require at least 100 rows")
    if selector.minimum_calibration_markets_per_band < 25:
        raise ValueError("selector calibration bands require at least 25 markets")
    if selector.admission_floor != 0.5:
        raise ValueError("the correctness admission floor must remain q=0.5")

    expected_bands = (
        PolicyThresholdBand("60-89", 60, 90),
        PolicyThresholdBand("90-119", 90, 120),
        PolicyThresholdBand("120-179", 120, 180),
        PolicyThresholdBand("180-240", 180, 241),
    )
    if config.bands != expected_bands:
        raise ValueError("correctness admission requires four frozen causal bands")
    thresholds = config.threshold_candidates
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or tuple(sorted(thresholds)) != thresholds
        or any(
            not math.isfinite(value) or value <= selector.admission_floor or value > 1.0
            for value in thresholds
        )
    ):
        raise ValueError(
            "admission thresholds must be unique, ordered, and strictly above q=0.5"
        )
    if len(thresholds) ** len(config.bands) > 100_000:
        raise ValueError("admission threshold matrix exceeds the compute budget")

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
        or gates.minimum_common_checkpoint_markets < 500
        or gates.minimum_executable_markets < 500
        or gates.minimum_mean_direct_edge_per_share < 0.0
        or gates.minimum_realized_net_per_share < 0.0
        or not gates.require_every_fold
        or gates.required_deployment_validation_folds
        < ADMISSION_REQUIRED_DEPLOYMENT_FOLDS
    ):
        raise ValueError("admission advancement gates weaken the frozen quality contract")


def admission_config_to_dict(config: AdmissionBenchmarkConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in (
        "source_path",
        "package_root",
        "probability_manifest",
        "core_config",
        "execution_evidence",
        "runs",
    ):
        payload[key] = str(payload[key])
    payload["candidate_names"] = list(config.candidate_names)
    return payload
