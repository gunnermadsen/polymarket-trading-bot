from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from .core_config import CoreGateConfig, CoreTrainingConfig, load_core_config
from .core_evaluation import (
    choose_threshold,
    classification_metrics,
    first_prediction_rows,
    paired_uplift,
    threshold_table,
)
from .core_extract import file_sha256, write_json_atomic
from .core_features import (
    CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
    load_core_feature_frame,
    validate_core_feature_cache,
)
from .core_training import (
    CORE_FREEZE_SCHEMA_VERSION,
    TRAINING_MODEL_FILENAME,
    FrozenTrainingBundle,
    model_candidate_spec,
    model_summary_payload,
    range_frame,
    row_weight_schedule_payload,
)
from .persistence_benchmark import (
    CANDIDATE_PROFILES,
    PersistenceTrainingBundle,
    _candidate_eligible_frame,
    _candidate_spec,
    _fit_calibrators,
    _fit_profile_model,
    _path_eligible_frame,
    _training_target_frame,
    calibrated_target_probability,
    target_probability_to_up,
)
from .persistence_config import (
    REGIME_ROBUST_ACCURACY_PROFILE,
    REGIME_ROBUST_RECENCY_CANDIDATE,
    PersistenceBenchmarkConfig,
    load_persistence_benchmark_config,
    persistence_config_to_dict,
)
from .provenance import runtime_provenance
from .runtime_export import export_runtime_model

PAPER_CANDIDATE_SCHEMA_VERSION = "btc-directional-paper-candidate-v1"
PAPER_ONLY_AUTHORIZATION = "explicit-paper-only-forward-evaluation"
PAPER_MINIMUM_COVERAGE = 0.55
PAPER_CONFIDENCE_THRESHOLD = 0.88
PAPER_GOLDEN_FEATURES_FILENAME = "golden-features.parquet"


@dataclass
class PaperCandidateFit:
    bundle: FrozenTrainingBundle
    tuning: dict[str, Any]
    calibration: list[dict[str, Any]]
    threshold_history: list[dict[str, Any]]
    paper_policy: dict[str, Any]
    production_policy: dict[str, Any]
    policy_frame: pl.DataFrame
    policy_probability_up: np.ndarray


def freeze_and_export_paper_candidate(
    *,
    config: PersistenceBenchmarkConfig,
    benchmark_run: Path,
    freeze_root: Path,
    runtime_output_root: Path,
    model_key: str,
    authorization: str,
) -> tuple[Path, Path, dict[str, Any]]:
    """Fit, truthfully freeze, and natively export the recency paper candidate.

    This path is deliberately separate from production qualification. It never
    changes or bypasses the production gates and it marks every artifact as
    paper-only and ineligible for live capital.
    """

    if authorization != PAPER_ONLY_AUTHORIZATION:
        raise RuntimeError(
            "paper candidate export requires the explicit paper-only authorization"
        )
    if config.profile != REGIME_ROBUST_ACCURACY_PROFILE:
        raise ValueError("paper candidate export requires the regime-robust profile")
    if REGIME_ROBUST_RECENCY_CANDIDATE not in config.candidate_names:
        raise ValueError("recency candidate is not present in the frozen benchmark matrix")

    core_config = load_core_config(config.core_config)
    feature_metadata = validate_core_feature_cache(core_config, "pre_holdout")
    validate_mature_reversal_schema(feature_metadata)
    benchmark, benchmark_path = load_benchmark_evidence(benchmark_run, config)
    fit = fit_paper_candidate(config, core_config)
    provenance = runtime_provenance(config.package_root)

    created_at = datetime.now(UTC)
    freeze_id = (
        f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{REGIME_ROBUST_RECENCY_CANDIDATE}-paper"
    )
    freeze_dir = freeze_root.resolve() / freeze_id
    freeze_dir.mkdir(parents=True, exist_ok=False)

    model_path = freeze_dir / TRAINING_MODEL_FILENAME
    joblib.dump(fit.bundle, model_path, compress=3)
    summary = model_summary_payload(fit.bundle)
    summary.update(
        {
            "paper_candidate_schema_version": PAPER_CANDIDATE_SCHEMA_VERSION,
            "deployment_scope": "paper_only",
            "production_qualified": False,
            "live_capital_allowed": False,
        }
    )
    summary_path = freeze_dir / "model-summary.json"
    write_json_atomic(summary_path, summary)

    golden_path = freeze_dir / PAPER_GOLDEN_FEATURES_FILENAME
    write_golden_feature_sample(
        fit.policy_frame,
        fit.policy_probability_up,
        fit.bundle,
        golden_path,
    )
    feature_path = core_config.paths.development_feature_data
    feature_metadata_path = feature_path.with_suffix(".metadata.json")
    candidate_evidence = benchmark["candidates"][
        REGIME_ROBUST_RECENCY_CANDIDATE
    ]
    benchmark_policy = candidate_evidence["own_policy"]
    frozen_spec = model_candidate_spec(fit.bundle.model)
    manifest: dict[str, Any] = {
        "schema_version": CORE_FREEZE_SCHEMA_VERSION,
        "paper_candidate_schema_version": PAPER_CANDIDATE_SCHEMA_VERSION,
        "freeze_id": freeze_id,
        "created_at": created_at.isoformat(),
        "status": "paper_candidate_frozen",
        "deployment_status": "paper_only_forward_evaluation",
        "deployment_scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
        "paper_only_authorization": authorization,
        "model_file": TRAINING_MODEL_FILENAME,
        "model_sha256": file_sha256(model_path),
        "model_summary_sha256": file_sha256(summary_path),
        "candidate": fit.bundle.model.candidate_name,
        "family": fit.bundle.model.family,
        "row_weight_policy": frozen_spec.row_weight_policy,
        "row_weight_schedule": row_weight_schedule_payload(frozen_spec),
        "recency_half_life_days": frozen_spec.recency_half_life_days,
        "feature_schema_version": CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
        "feature_names": list(fit.bundle.model.feature_names),
        "hyperparameters": fit.bundle.model.hyperparameters,
        "calibrator": asdict(fit.bundle.calibrator),
        "confidence_threshold": fit.bundle.confidence_threshold,
        "prediction_policy": {
            "type": "first_confidence_crossing",
            "minimum_seconds_after_open": core_config.data.min_seconds_after_open,
            "maximum_seconds_after_open": (
                300 - core_config.data.min_seconds_before_close
            ),
            "cadence_seconds": core_config.data.sample_interval_seconds,
        },
        "configuration_sha256": file_sha256(config.source_path),
        "core_configuration_sha256": file_sha256(core_config.source_path),
        "development_feature_sha256": file_sha256(feature_path),
        "development_feature_metadata_sha256": file_sha256(
            feature_metadata_path
        ),
        "golden_feature_file": golden_path.name,
        "golden_feature_sha256": file_sha256(golden_path),
        "golden_feature_metadata_sha256": file_sha256(
            golden_path.with_suffix(".metadata.json")
        ),
        "source_tree_sha256": provenance["source_tree_sha256"],
        "git": provenance["git"],
        "runtime_provenance": provenance,
        "random_seed": core_config.model.random_seed,
        "training_ranges": {
            "development": _range_payload(
                core_config.split.development_start,
                core_config.split.development_end,
            ),
            "probability_calibration": _range_payload(
                core_config.split.probability_calibration_start,
                core_config.split.probability_calibration_end,
            ),
            "policy_selection": _range_payload(
                core_config.split.policy_selection_start,
                core_config.split.policy_selection_end,
            ),
        },
        "holdout_range": _range_payload(
            core_config.split.holdout_start,
            core_config.split.holdout_end,
        ),
        "forward_paper_evaluation": {
            "start": core_config.split.policy_selection_end.isoformat(),
            "end": None,
            "intended_independent": True,
            "independent_evidence_available": False,
            "status": "pending",
        },
        "production_gates": asdict(core_config.gates),
        "paper_policy": fit.paper_policy,
        "production_policy_evaluation": fit.production_policy,
        "production_blocking_reasons": production_blocking_reasons(
            benchmark,
            fit.production_policy,
        ),
        "benchmark_evidence": {
            "run": str(benchmark_run.resolve()),
            "benchmark_sha256": file_sha256(benchmark_path),
            "evaluation_is_independent": bool(
                benchmark["evaluation"]["independent"]
            ),
            "benchmark_passed": (
                REGIME_ROBUST_RECENCY_CANDIDATE
                in benchmark["benchmark_passed_candidates"]
            ),
            "deployment_qualified": (
                REGIME_ROBUST_RECENCY_CANDIDATE
                in benchmark["deployment_qualified_candidates"]
            ),
            "aggregate_policy_metrics": benchmark_policy,
        },
        "walk_forward_accuracy": benchmark_policy["accuracy"],
        "walk_forward_uplift": _benchmark_check_observed(
            candidate_evidence["advance"]["checks"],
            "minimum aggregate accuracy uplift",
        ),
        "tuning": fit.tuning,
        "calibration": fit.calibration,
    }
    manifest_path = freeze_dir / "freeze-manifest.json"
    write_json_atomic(manifest_path, manifest)
    (freeze_dir / "freeze-manifest.sha256").write_text(
        file_sha256(manifest_path) + "\n"
    )

    runtime_dir = export_runtime_model(
        freeze_dir=freeze_dir,
        golden_features=golden_path,
        output_root=runtime_output_root,
        model_key=model_key,
    )
    return freeze_dir, runtime_dir, manifest


def fit_paper_candidate(
    config: PersistenceBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> PaperCandidateFit:
    """Fit the selected model once and retain that exact estimator for export."""

    profile = CANDIDATE_PROFILES[REGIME_ROBUST_RECENCY_CANDIDATE]
    if profile.target_kind != "outcome_up" or profile.calibration_kind != "global_platt":
        raise RuntimeError(
            "paper runtime conversion requires direct-outcome/global-Platt training"
        )
    frame = load_core_feature_frame(core_config, "pre_holdout")
    development = _candidate_eligible_frame(
        range_frame(
            frame,
            core_config.split.development_start,
            core_config.split.development_end,
        ),
        profile,
    )
    calibration = _candidate_eligible_frame(
        range_frame(
            frame,
            core_config.split.probability_calibration_start,
            core_config.split.probability_calibration_end,
        ),
        profile,
    )
    policy_source = range_frame(
        frame,
        core_config.split.policy_selection_start,
        core_config.split.policy_selection_end,
    )
    policy_universe = _path_eligible_frame(policy_source)
    policy = _candidate_eligible_frame(policy_source, profile)
    spec = _candidate_spec(profile, config)

    model, tuning = _fit_profile_model(
        _training_target_frame(development, profile),
        profile,
        spec,
        core_config,
    )
    calibrators, calibration_evidence = _fit_calibrators(
        model,
        calibration,
        profile,
        config,
        core_config,
        spec,
    )
    target_probability = calibrated_target_probability(
        model,
        calibrators,
        policy,
    )
    probability_up = target_probability_to_up(
        policy,
        target_probability,
        profile.target_kind,
    )
    table = threshold_table(policy, probability_up, core_config.model)
    eligible_markets = policy_universe["market_id"].n_unique()
    paper_row = choose_paper_threshold(
        table,
        locked_threshold=PAPER_CONFIDENCE_THRESHOLD,
        minimum_coverage=PAPER_MINIMUM_COVERAGE,
        minimum_markets=max(
            100,
            math.ceil(eligible_markets * PAPER_MINIMUM_COVERAGE),
        ),
    )
    production_threshold, production_threshold_qualified = choose_threshold(
        table,
        core_config.gates,
        minimum_markets=max(
            100,
            math.ceil(eligible_markets * core_config.gates.minimum_coverage),
        ),
    )
    production_row = _threshold_row(table, production_threshold)
    production_checks = policy_gate_checks(
        production_row,
        core_config.gates,
        minimum_markets=max(
            100,
            math.ceil(eligible_markets * core_config.gates.minimum_coverage),
        ),
    )
    production_passed = bool(
        production_threshold_qualified
        and all(check["passed"] for check in production_checks)
        and all(calibrator.converged for calibrator in calibrators.calibrators.values())
    )

    persistence_bundle = PersistenceTrainingBundle(
        model=model,
        calibrators=calibrators,
        profile=profile,
        confidence_threshold=float(paper_row["threshold"]),
    )
    runtime_bundle = persistence_bundle_to_runtime_bundle(persistence_bundle)
    selected_rows = first_prediction_rows(
        policy,
        probability_up,
        runtime_bundle.confidence_threshold,
    )
    selected_metrics = classification_metrics(
        selected_rows,
        eligible_markets=eligible_markets,
    )
    return PaperCandidateFit(
        bundle=runtime_bundle,
        tuning=tuning,
        calibration=calibration_evidence,
        threshold_history=table,
        paper_policy={
            "selection_objective": (
                "pre-registered 0.88 confidence threshold, accepted only when "
                "it retains at least 55% eligible-market coverage"
            ),
            "threshold_locked_before_fit": True,
            "minimum_coverage": PAPER_MINIMUM_COVERAGE,
            "threshold": runtime_bundle.confidence_threshold,
            "metrics": {
                **selected_metrics,
                **paired_uplift(selected_rows),
            },
            "threshold_history": table,
        },
        production_policy={
            "threshold": production_threshold,
            "threshold_qualified": production_threshold_qualified,
            "metrics": production_row,
            "checks": production_checks,
            "passed": production_passed,
        },
        policy_frame=policy,
        policy_probability_up=probability_up,
    )


def persistence_bundle_to_runtime_bundle(
    bundle: PersistenceTrainingBundle,
) -> FrozenTrainingBundle:
    if bundle.profile.target_kind != "outcome_up":
        raise RuntimeError("runtime bundle conversion requires a direct outcome target")
    if bundle.calibrators.kind != "global_platt":
        raise RuntimeError("runtime bundle conversion requires global Platt calibration")
    if set(bundle.calibrators.calibrators) != {"global"}:
        raise RuntimeError("global Platt bundle must contain exactly one calibrator")
    return FrozenTrainingBundle(
        model=bundle.model,
        calibrator=bundle.calibrators.calibrators["global"],
        confidence_threshold=bundle.confidence_threshold,
    )


def choose_paper_threshold(
    table: list[dict[str, Any]],
    *,
    locked_threshold: float,
    minimum_coverage: float,
    minimum_markets: int,
) -> dict[str, Any]:
    candidates = [
        row
        for row in table
        if math.isclose(
            float(row["threshold"]),
            locked_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if not candidates:
        raise RuntimeError(
            f"locked paper confidence threshold {locked_threshold:.2f} "
            "is absent from the configured policy table"
        )
    selected = candidates[0]
    if (
        selected["coverage"] < minimum_coverage
        or selected["markets"] < minimum_markets
    ):
        raise RuntimeError(
            f"locked paper confidence threshold {locked_threshold:.2f} "
            "does not retain the paper coverage floor"
        )
    return selected


def policy_gate_checks(
    metrics: dict[str, Any],
    gates: CoreGateConfig,
    *,
    minimum_markets: int,
) -> list[dict[str, Any]]:
    contracts = (
        ("markets", metrics["markets"], minimum_markets, ">="),
        ("coverage", metrics["coverage"], gates.minimum_coverage, ">="),
        ("accuracy", metrics["accuracy"], gates.target_accuracy, ">="),
        (
            "wilson_lower_95",
            metrics["wilson_lower_95"],
            gates.target_wilson_lower,
            ">=",
        ),
        (
            "balanced_accuracy",
            metrics["balanced_accuracy"],
            gates.target_balanced_accuracy,
            ">=",
        ),
        (
            "up_recall",
            metrics["up_recall"],
            gates.minimum_direction_recall,
            ">=",
        ),
        (
            "down_recall",
            metrics["down_recall"],
            gates.minimum_direction_recall,
            ">=",
        ),
        (
            "expected_calibration_error",
            metrics["expected_calibration_error"],
            gates.maximum_ece,
            "<=",
        ),
        (
            "accuracy_uplift",
            metrics["accuracy_uplift"],
            gates.minimum_same_time_path_uplift,
            ">=",
        ),
    )
    return [
        {
            "name": name,
            "observed": observed,
            "operator": operator,
            "required": required,
            "passed": (
                observed >= required if operator == ">=" else observed <= required
            ),
        }
        for name, observed, required, operator in contracts
    ]


def validate_mature_reversal_schema(metadata: dict[str, Any]) -> None:
    candidate_schemas = metadata.get("candidate_feature_schema_versions", {})
    observed = candidate_schemas.get(REGIME_ROBUST_RECENCY_CANDIDATE)
    if observed != CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION:
        raise RuntimeError("recency candidate feature schema is not the frozen 71-feature schema")
    feature_count = len(model_candidate_feature_names())
    if feature_count != 71:
        raise RuntimeError("recency candidate runtime schema must contain 71 features")


def model_candidate_feature_names() -> tuple[str, ...]:
    from .core_features import CORE_MATURE_REVERSAL_ENRICHED_FEATURES

    return tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)


def load_benchmark_evidence(
    benchmark_run: Path,
    config: PersistenceBenchmarkConfig,
) -> tuple[dict[str, Any], Path]:
    benchmark_path = benchmark_run.resolve() / "benchmark.json"
    if not benchmark_path.is_file():
        raise RuntimeError("benchmark evidence is missing")
    benchmark = json.loads(benchmark_path.read_text())
    expected_configuration = json.loads(
        json.dumps(persistence_config_to_dict(config))
    )
    if benchmark.get("configuration") != expected_configuration:
        raise RuntimeError(
            "benchmark evidence does not match the exact benchmark configuration"
        )
    if REGIME_ROBUST_RECENCY_CANDIDATE not in benchmark.get("candidates", {}):
        raise RuntimeError("benchmark evidence does not contain the recency candidate")
    return benchmark, benchmark_path


def write_golden_feature_sample(
    frame: pl.DataFrame,
    probability_up: np.ndarray,
    bundle: FrozenTrainingBundle,
    destination: Path,
) -> None:
    if frame.height != len(probability_up):
        raise ValueError("golden probability count does not match feature rows")
    order = np.argsort(probability_up)
    sample_count = min(256, len(order))
    quantile_positions = np.linspace(0, len(order) - 1, sample_count).round().astype(int)
    indices = {int(order[position]) for position in quantile_positions}
    for target in (
        float(probability_up.min()),
        1.0 - bundle.confidence_threshold,
        0.5,
        bundle.confidence_threshold,
        float(probability_up.max()),
    ):
        indices.add(int(np.argmin(np.abs(probability_up - target))))
    selected = frame[sorted(indices)].select(
        "market_id",
        "observed_at",
        *bundle.model.feature_names,
    )
    selected.write_parquet(destination, compression="zstd")
    write_json_atomic(
        destination.with_suffix(".metadata.json"),
        {
            "feature_schema_version": CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
            "feature_file_sha256": file_sha256(destination),
            "rows": selected.height,
            "source": "policy_selection_feature_sample",
        },
    )


def production_blocking_reasons(
    benchmark: dict[str, Any],
    production_policy: dict[str, Any],
) -> list[str]:
    reasons = [
        "artifact is explicitly authorized only for forward paper evaluation",
        "no independent post-freeze forward cohort has been evaluated",
    ]
    if REGIME_ROBUST_RECENCY_CANDIDATE not in benchmark["benchmark_passed_candidates"]:
        reasons.append("development benchmark advancement contract did not pass")
    if (
        REGIME_ROBUST_RECENCY_CANDIDATE
        not in benchmark["deployment_qualified_candidates"]
    ):
        reasons.append("development benchmark did not qualify production deployment")
    if not production_policy["passed"]:
        reasons.append("final-fit policy selection did not pass every production gate")
    return reasons


def _threshold_row(
    table: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    for row in table:
        if math.isclose(float(row["threshold"]), threshold, abs_tol=1e-12):
            return row
    raise RuntimeError("selected threshold is missing from its policy table")


def _range_payload(start: datetime, end: datetime) -> dict[str, str]:
    return {"start": start.isoformat(), "end": end.isoformat()}


def _benchmark_check_observed(
    checks: list[dict[str, Any]],
    name: str,
) -> float:
    for check in checks:
        if check.get("name") == name:
            return float(check["observed"])
    raise RuntimeError(f"benchmark evidence is missing required check: {name}")


def run_paper_candidate_export(
    *,
    config_path: Path,
    benchmark_run: Path,
    freeze_root: Path,
    runtime_output_root: Path,
    model_key: str,
    authorization: str,
) -> tuple[Path, Path, dict[str, Any]]:
    return freeze_and_export_paper_candidate(
        config=load_persistence_benchmark_config(config_path),
        benchmark_run=benchmark_run,
        freeze_root=freeze_root,
        runtime_output_root=runtime_output_root,
        model_key=model_key,
        authorization=authorization,
    )
