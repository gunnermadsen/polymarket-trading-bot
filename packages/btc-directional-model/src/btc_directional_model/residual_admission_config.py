from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .persistence_benchmark import SAVED_POLICY_PROBABILITY_SCHEMA_VERSION

RESIDUAL_ADMISSION_PROFILE = "residual_admission"
RESIDUAL_ADMISSION_SOURCE_PROFILE = "residual_admission_source"
RESIDUAL_CONTROL_CANDIDATE = "histogram_enriched"
RESIDUAL_PROPOSAL_CANDIDATE = "histogram_boundary_reversal"
EARLY_RESIDUAL_CANDIDATE = "histogram_boundary_residual_early"
RESCUE_RESIDUAL_CANDIDATE = "histogram_boundary_residual_rescue"
COMBINED_RESIDUAL_CANDIDATE = "histogram_boundary_residual_combined"
RESIDUAL_SOURCE_CANDIDATES = (
    RESIDUAL_CONTROL_CANDIDATE,
    RESIDUAL_PROPOSAL_CANDIDATE,
)
RESIDUAL_EVALUATION_FOLDS = (2, 3, 4, 5, 6)
RESIDUAL_REQUIRED_SOURCE_FOLDS = 7
RESIDUAL_THRESHOLD_GRID = (0.87, 0.89, 0.91, 0.93)


@dataclass(frozen=True)
class ResidualSelectorConfig:
    regularization_c: float
    minimum_calibration_rows_per_direction: int
    minimum_calibration_markets_per_direction: int
    agreement_cadences: int
    base_confidence_threshold: float
    random_seed: int


@dataclass(frozen=True)
class ResidualHeadConfig:
    name: str
    candidate: str
    start_second: int
    end_second_exclusive: int
    threshold_candidates: tuple[float, ...]


@dataclass(frozen=True)
class ResidualAdmissionGates:
    minimum_accuracy: float
    minimum_balanced_accuracy: float
    minimum_direction_recall: float
    minimum_wilson_lower_95: float
    maximum_expected_calibration_error: float
    minimum_selected_markets: int
    minimum_executable_markets: int
    maximum_accuracy_regression: float
    maximum_balanced_accuracy_regression: float
    maximum_direction_recall_regression: float
    maximum_median_entry_second: float
    minimum_median_entry_improvement_seconds: float
    minimum_decisions_by_120_uplift: float
    minimum_early_residual_markets: int
    minimum_median_advancement_seconds: float
    minimum_no_trade_reduction: float
    minimum_rescued_markets: int
    minimum_residual_accuracy: float
    minimum_mean_direct_edge_per_share: float
    minimum_realized_net_per_share: float
    require_every_fold: bool
    required_evaluation_folds: int


@dataclass(frozen=True)
class ResidualAdmissionBenchmarkConfig:
    source_path: Path
    package_root: Path
    profile: str
    probability_manifest: Path
    core_config: Path
    control_candidate: str
    proposal_candidate: str
    early_head: ResidualHeadConfig
    rescue_head: ResidualHeadConfig
    combined_candidate: str
    evaluation_folds: tuple[int, ...]
    evaluation_note: str
    evaluation_is_independent: bool
    quantity: float
    selector: ResidualSelectorConfig
    gates: ResidualAdmissionGates
    execution_evidence: Path
    runs: Path

    @property
    def candidate_names(self) -> tuple[str, ...]:
        return (
            self.control_candidate,
            self.early_head.candidate,
            self.rescue_head.candidate,
            self.combined_candidate,
        )


def load_residual_admission_config(
    path: Path,
) -> ResidualAdmissionBenchmarkConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    benchmark = raw["benchmark"]
    selector = raw["selector"]
    early = raw["early_head"]
    rescue = raw["rescue_head"]
    gates = raw["advancement_gates"]
    paths = raw["paths"]
    config = ResidualAdmissionBenchmarkConfig(
        source_path=source_path,
        package_root=package_root,
        profile=str(benchmark["profile"]),
        probability_manifest=package_root / str(benchmark["probability_manifest"]),
        core_config=package_root / str(benchmark["core_config"]),
        control_candidate=str(benchmark["control_candidate"]),
        proposal_candidate=str(benchmark["proposal_candidate"]),
        early_head=_load_head(early),
        rescue_head=_load_head(rescue),
        combined_candidate=str(benchmark["combined_candidate"]),
        evaluation_folds=tuple(int(value) for value in benchmark["evaluation_folds"]),
        evaluation_note=str(benchmark["evaluation_note"]).strip(),
        evaluation_is_independent=bool(benchmark["evaluation_is_independent"]),
        quantity=float(benchmark["quantity"]),
        selector=ResidualSelectorConfig(
            regularization_c=float(selector["regularization_c"]),
            minimum_calibration_rows_per_direction=int(
                selector["minimum_calibration_rows_per_direction"]
            ),
            minimum_calibration_markets_per_direction=int(
                selector["minimum_calibration_markets_per_direction"]
            ),
            agreement_cadences=int(selector["agreement_cadences"]),
            base_confidence_threshold=float(selector["base_confidence_threshold"]),
            random_seed=int(selector["random_seed"]),
        ),
        gates=ResidualAdmissionGates(
            minimum_accuracy=float(gates["minimum_accuracy"]),
            minimum_balanced_accuracy=float(gates["minimum_balanced_accuracy"]),
            minimum_direction_recall=float(gates["minimum_direction_recall"]),
            minimum_wilson_lower_95=float(gates["minimum_wilson_lower_95"]),
            maximum_expected_calibration_error=float(
                gates["maximum_expected_calibration_error"]
            ),
            minimum_selected_markets=int(gates["minimum_selected_markets"]),
            minimum_executable_markets=int(gates["minimum_executable_markets"]),
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
            minimum_decisions_by_120_uplift=float(
                gates["minimum_decisions_by_120_uplift"]
            ),
            minimum_early_residual_markets=int(
                gates["minimum_early_residual_markets"]
            ),
            minimum_median_advancement_seconds=float(
                gates["minimum_median_advancement_seconds"]
            ),
            minimum_no_trade_reduction=float(
                gates["minimum_no_trade_reduction"]
            ),
            minimum_rescued_markets=int(gates["minimum_rescued_markets"]),
            minimum_residual_accuracy=float(gates["minimum_residual_accuracy"]),
            minimum_mean_direct_edge_per_share=float(
                gates["minimum_mean_direct_edge_per_share"]
            ),
            minimum_realized_net_per_share=float(
                gates["minimum_realized_net_per_share"]
            ),
            require_every_fold=bool(gates["require_every_fold"]),
            required_evaluation_folds=int(gates["required_evaluation_folds"]),
        ),
        execution_evidence=package_root / str(paths["execution_evidence"]),
        runs=package_root / str(paths["runs"]),
    )
    validate_residual_admission_config(config)
    return config


def _load_head(raw: dict[str, Any]) -> ResidualHeadConfig:
    return ResidualHeadConfig(
        name=str(raw["name"]),
        candidate=str(raw["candidate"]),
        start_second=int(raw["start_second"]),
        end_second_exclusive=int(raw["end_second_exclusive"]),
        threshold_candidates=tuple(
            float(value) for value in raw["threshold_candidates"]
        ),
    )


def validate_residual_admission_config(
    config: ResidualAdmissionBenchmarkConfig,
) -> None:
    if config.profile != RESIDUAL_ADMISSION_PROFILE:
        raise ValueError("residual benchmark profile must remain residual_admission")
    if "__SOURCE_RUN_ID__" in str(config.probability_manifest):
        raise ValueError(
            "replace __SOURCE_RUN_ID__ with the completed seven-fold source run id"
        )
    if not config.probability_manifest.is_file():
        raise ValueError(
            f"saved probability manifest is missing: {config.probability_manifest}"
        )
    manifest = json.loads(config.probability_manifest.read_text())
    if manifest.get("schema_version") != SAVED_POLICY_PROBABILITY_SCHEMA_VERSION:
        raise ValueError("saved probability manifest schema is unsupported")
    if manifest.get("source_benchmark_profile") != RESIDUAL_ADMISSION_SOURCE_PROFILE:
        raise ValueError(
            "residual admission requires residual_admission_source evidence"
        )
    if tuple(manifest.get("candidate_names", ())) != RESIDUAL_SOURCE_CANDIDATES:
        raise ValueError("residual admission source candidate matrix changed")
    if manifest.get("control_candidate") != RESIDUAL_CONTROL_CANDIDATE:
        raise ValueError("histogram_enriched must remain the residual control")
    if int(manifest.get("fold_count", 0)) != RESIDUAL_REQUIRED_SOURCE_FOLDS:
        raise ValueError("residual admission requires exactly seven source folds")
    for candidate_name in RESIDUAL_SOURCE_CANDIDATES:
        candidate = manifest.get("candidates", {}).get(candidate_name)
        if candidate is None:
            raise ValueError(f"source evidence is missing {candidate_name}")
        indexes = tuple(
            sorted(int(row["fold_index"]) for row in candidate.get("folds", ()))
        )
        if indexes != tuple(range(RESIDUAL_REQUIRED_SOURCE_FOLDS)):
            raise ValueError(f"{candidate_name} source folds are incomplete")
    if config.control_candidate != RESIDUAL_CONTROL_CANDIDATE:
        raise ValueError("histogram_enriched must remain the residual control")
    if config.proposal_candidate != RESIDUAL_PROPOSAL_CANDIDATE:
        raise ValueError(
            "histogram_boundary_reversal must remain the residual proposal model"
        )
    if config.early_head != ResidualHeadConfig(
        name="early",
        candidate=EARLY_RESIDUAL_CANDIDATE,
        start_second=60,
        end_second_exclusive=120,
        threshold_candidates=RESIDUAL_THRESHOLD_GRID,
    ):
        raise ValueError("early residual head contract changed")
    if config.rescue_head != ResidualHeadConfig(
        name="rescue",
        candidate=RESCUE_RESIDUAL_CANDIDATE,
        start_second=120,
        end_second_exclusive=241,
        threshold_candidates=RESIDUAL_THRESHOLD_GRID,
    ):
        raise ValueError("rescue residual head contract changed")
    if config.combined_candidate != COMBINED_RESIDUAL_CANDIDATE:
        raise ValueError("combined residual candidate identity changed")
    if config.evaluation_folds != RESIDUAL_EVALUATION_FOLDS:
        raise ValueError("residual admission requires evaluation folds 2 through 6")
    if config.evaluation_is_independent:
        raise ValueError("rolling residual validation is consumed development evidence")
    if not config.evaluation_note:
        raise ValueError("evaluation_note is required")
    if not math.isclose(config.quantity, 5.0):
        raise ValueError("residual economics require exactly five shares")
    if not config.core_config.is_file():
        raise ValueError(f"residual core config is missing: {config.core_config}")
    if not (config.execution_evidence / "manifest.json").is_file():
        raise ValueError("residual execution-evidence manifest is missing")

    selector = config.selector
    if (
        not math.isclose(selector.regularization_c, 1.0)
        or selector.minimum_calibration_rows_per_direction < 500
        or selector.minimum_calibration_markets_per_direction < 100
        or selector.agreement_cadences != 2
        or not math.isclose(selector.base_confidence_threshold, 0.89)
        or selector.random_seed != 20260728
    ):
        raise ValueError("residual selector parameters changed")

    gates = config.gates
    if (
        gates.minimum_accuracy < 0.874
        or gates.minimum_balanced_accuracy < 0.874
        or gates.minimum_direction_recall < 0.874
        or gates.minimum_wilson_lower_95 < 0.865
        or gates.maximum_expected_calibration_error > 0.05
        or gates.minimum_selected_markets < 500
        or gates.minimum_executable_markets < 500
        or gates.maximum_accuracy_regression > 0.0
        or gates.maximum_balanced_accuracy_regression > 0.0
        or gates.maximum_direction_recall_regression > 0.0
        or gates.maximum_median_entry_second > 125.0
        or gates.minimum_median_entry_improvement_seconds < 5.0
        or gates.minimum_decisions_by_120_uplift < 0.02
        or gates.minimum_early_residual_markets < 500
        or gates.minimum_median_advancement_seconds < 10.0
        or gates.minimum_no_trade_reduction < 0.02
        or gates.minimum_rescued_markets < 500
        or gates.minimum_residual_accuracy < 0.874
        or gates.minimum_mean_direct_edge_per_share < 0.0
        or gates.minimum_realized_net_per_share < 0.0
        or not gates.require_every_fold
        or gates.required_evaluation_folds != len(RESIDUAL_EVALUATION_FOLDS)
    ):
        raise ValueError("residual advancement gates weaken the frozen contract")


def residual_admission_config_to_dict(
    config: ResidualAdmissionBenchmarkConfig,
) -> dict[str, Any]:
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
