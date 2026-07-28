from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from .core_benchmark import (
    BenchmarkEvidence,
    CandidatePolicy,
    benchmark_predictions,
)
from .core_evaluation import classification_metrics, first_crossing_timing
from .core_execution import (
    ExecutionEvidenceConfig,
    load_execution_evidence_manifest,
)
from .core_extract import file_sha256, write_json_atomic
from .frequency_policy_config import (
    FIXED_FREQUENCY_CHECKPOINTS,
    FrequencyPolicyAdvancementGates,
    FrequencyPolicyBenchmarkConfig,
    frequency_policy_config_to_dict,
)
from .persistence_benchmark import (
    attach_execution_evidence,
    load_execution_evidence,
)
from .policy_benchmark import (
    absolute_policy_checks,
    apply_time_band_policy,
    load_probability_evidence,
    load_saved_probability_manifest,
    select_causal_time_band_thresholds,
)
from .provenance import runtime_provenance

FREQUENCY_POLICY_BENCHMARK_SCHEMA_VERSION = "btc-frequency-policy-benchmark-v1"


def run_frequency_policy_benchmark(
    config: FrequencyPolicyBenchmarkConfig,
) -> tuple[Path, dict[str, Any]]:
    manifest = load_saved_probability_manifest(config.probability_manifest)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    candidate_results: dict[str, dict[str, Any]] = {}
    scored_frames: dict[str, pl.DataFrame] = {}
    validation_universes: dict[int, set[str]] | None = None
    for candidate_name in config.candidate_names:
        result, scored, observed_universes = evaluate_frequency_policy_candidate(
            config,
            manifest,
            candidate_name,
            validation_universes=validation_universes,
        )
        if validation_universes is None:
            validation_universes = observed_universes
        candidate_results[candidate_name] = result
        scored_frames[candidate_name] = scored
    if validation_universes is None:
        raise RuntimeError("frequency policy benchmark has no validation universe")

    execution_config = _execution_config(config)
    execution_manifest = load_execution_evidence_manifest(execution_config)
    execution = load_execution_evidence(execution_config)
    scored_with_execution = {
        name: attach_execution_evidence(frame, execution) for name, frame in scored_frames.items()
    }
    eligible_market_ids = sorted(
        market_id for market_ids in validation_universes.values() for market_id in market_ids
    )
    policies = {
        name: CandidatePolicy(
            confidence_threshold=None,
            deployment_compatible=False,
            selection_mode="time_band_preselected",
            confidence_threshold_min=min(config.threshold_candidates),
            confidence_threshold_max=max(config.threshold_candidates),
        )
        for name in config.candidate_names
    }
    diagnostics = benchmark_predictions(
        scored_with_execution,
        policies=policies,
        control_candidate=config.control_candidate,
        evidence=BenchmarkEvidence(
            label=(
                "Single frozen frequency policy over five consumed chronological "
                "development validation folds"
            ),
            kind="development",
            independent=False,
        ),
        eligible_market_ids=eligible_market_ids,
        minimum_samples=config.gates.minimum_selected_markets,
        minimum_executable_samples=config.gates.minimum_executable_markets,
        quantity=config.quantity,
    )

    control_result = candidate_results[config.control_candidate]
    for candidate_name in config.candidate_names:
        result = candidate_results[candidate_name]
        candidate_diagnostics = diagnostics["candidates"][candidate_name]
        result["out_of_fold"] = candidate_diagnostics["own_policy"]
        result["timing"] = first_crossing_timing(
            scored_frames[candidate_name].filter(pl.col("policy_selected")),
            eligible_markets=len(eligible_market_ids),
        )
        result["no_trade_rate"] = 1.0 - result["out_of_fold"]["coverage"]
        result["execution"] = candidate_diagnostics["own_policy"]["execution"]
        result["checkpoints"] = candidate_diagnostics["checkpoints"]
        result["timing_qualification_role"] = "diagnostic_only"
        if candidate_name == config.control_candidate:
            comparison = None
        else:
            comparison = diagnostics["common_comparisons"][candidate_name]
        result["advance"] = frequency_policy_advancement_checks(
            result,
            control_result,
            config.gates,
            comparison=comparison,
            is_control=candidate_name == config.control_candidate,
            quantity=config.quantity,
        )
        selected = scored_with_execution[candidate_name].filter(pl.col("policy_selected"))
        selected_path = run_dir / f"{candidate_name}-single-policy-validation.parquet"
        _write_parquet_atomic(selected, selected_path)
        result["selected_validation_evidence"] = {
            "path": selected_path.name,
            "sha256": file_sha256(selected_path),
            "rows": selected.height,
            "markets": selected["market_id"].n_unique(),
        }

    passing = [
        name
        for name in config.candidate_names
        if name != config.control_candidate
        and candidate_results[name]["advance"]["benchmark_passed"]
    ]
    winner = (
        max(
            passing,
            key=lambda name: (
                candidate_results[name]["out_of_fold"]["coverage"],
                candidate_results[name]["out_of_fold"]["wilson_lower_95"],
                candidate_results[name]["out_of_fold"]["balanced_accuracy"],
                candidate_results[name]["out_of_fold"]["accuracy"],
            ),
        )
        if passing
        else None
    )
    benchmark = {
        "schema_version": FREQUENCY_POLICY_BENCHMARK_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "qualification_objective": config.qualification_objective,
        "policy_selection_mode": config.policy_selection_mode,
        "policy_anchor_fold": config.policy_anchor_fold,
        "timing_qualification_role": "diagnostic_only",
        "fixed_checkpoints": list(FIXED_FREQUENCY_CHECKPOINTS),
        "configuration": frequency_policy_config_to_dict(config),
        "evaluation_note": config.evaluation_note,
        "evaluation_is_independent": config.evaluation_is_independent,
        "probability_evidence": {
            "manifest": str(config.probability_manifest),
            "manifest_sha256": file_sha256(config.probability_manifest),
            "schema_version": manifest["schema_version"],
            "created_at": manifest.get("created_at"),
            "source_benchmark_profile": manifest.get("source_benchmark_profile"),
            "source_config": manifest.get("source_config"),
            "source_config_sha256": manifest.get("source_config_sha256"),
            "control_candidate": manifest.get("control_candidate"),
            "candidate_names": manifest.get("candidate_names"),
            "fold_count": manifest["fold_count"],
            "causal_contract": manifest.get("causal_contract"),
            "checksums_verified": True,
            "causal_contract_verified": True,
        },
        "execution_evidence": {
            "manifest": str(config.execution_evidence / "manifest.json"),
            "manifest_sha256": file_sha256(config.execution_evidence / "manifest.json"),
            "source_contract": execution_manifest.get("source_contract"),
            "source_schema_version": execution_manifest.get("source_schema_version"),
            "range_start": execution_manifest["range_start"],
            "range_end": execution_manifest["range_end"],
            "quantity": execution_manifest["quantity"],
            "checksums_verified": True,
        },
        "runtime_provenance": runtime_provenance(config.package_root),
        "control_candidate": config.control_candidate,
        "candidates": candidate_results,
        "common_comparisons": diagnostics["common_comparisons"],
        "benchmark_passed_candidates": passing,
        "winner": winner,
        "deployment": {
            "status": "not_authorized",
            "runtime_exported": False,
            "runtime_changed": False,
            "scope": (
                "offline frequency qualification over consumed development "
                "probabilities and cached execution evidence only"
            ),
        },
    }
    write_json_atomic(run_dir / "benchmark.json", benchmark)
    from .policy_report import generate_saved_policy_report

    generate_saved_policy_report(benchmark, run_dir / "report.html")
    return run_dir, benchmark


def evaluate_frequency_policy_candidate(
    config: FrequencyPolicyBenchmarkConfig,
    manifest: dict[str, Any],
    candidate_name: str,
    *,
    validation_universes: dict[int, set[str]] | None = None,
) -> tuple[dict[str, Any], pl.DataFrame, dict[int, set[str]]]:
    candidate = manifest["candidates"].get(candidate_name)
    if candidate is None:
        raise ValueError(f"saved probability manifest is missing {candidate_name}")
    fold_records = {int(record["fold_index"]): record for record in candidate["folds"]}
    if len(fold_records) != int(manifest["fold_count"]):
        raise RuntimeError(f"{candidate_name} lost a saved-probability fold")
    anchor_record = fold_records.get(config.policy_anchor_fold)
    if anchor_record is None:
        raise RuntimeError(f"{candidate_name} is missing the policy anchor fold")
    anchor_rows = load_probability_evidence(
        config.probability_manifest,
        anchor_record["policy_selection"],
        candidate_name,
        config.policy_anchor_fold,
        "policy_selection",
    )
    selection = select_causal_time_band_thresholds(
        anchor_rows,
        config,
        include_timing_gates=False,
    )
    anchor_end = anchor_rows["window_start"].max()

    fold_results: list[dict[str, Any]] = []
    scored_frames: list[pl.DataFrame] = []
    observed_universes: dict[int, set[str]] = {}
    for fold_index, fold_record in sorted(fold_records.items()):
        validation_rows = load_probability_evidence(
            config.probability_manifest,
            fold_record["validation"],
            candidate_name,
            fold_index,
            "validation",
        )
        validation_start = validation_rows["window_start"].min()
        if anchor_end >= validation_start:
            raise RuntimeError(
                f"{candidate_name} anchor policy is not causal for fold {fold_index}"
            )
        observed_markets = set(validation_rows["market_id"].to_list())
        observed_universes[fold_index] = observed_markets
        if validation_universes is None:
            eligible_markets = len(observed_markets)
        else:
            expected = validation_universes.get(fold_index)
            if expected is None:
                raise RuntimeError(f"control validation universe lost fold {fold_index}")
            outside_control = observed_markets - expected
            if outside_control:
                raise RuntimeError(
                    f"{candidate_name} fold {fold_index} escapes the control universe"
                )
            eligible_markets = len(expected)
        validation_scored = apply_time_band_policy(
            validation_rows,
            config.bands,
            selection.threshold_map(),
        )
        selected = validation_scored.filter(pl.col("policy_selected"))
        metrics = classification_metrics(
            selected,
            eligible_markets=eligible_markets,
        )
        timing = first_crossing_timing(
            selected,
            eligible_markets=eligible_markets,
        )
        validation_checks = absolute_policy_checks(
            metrics,
            timing,
            config.gates,
            include_timing=False,
        )
        validation_qualified = all(check["passed"] for check in validation_checks)
        scored_frames.append(validation_scored)
        fold_results.append(
            {
                "fold_index": fold_index,
                "causal_order_verified": True,
                "policy_selection_range": {
                    "start": anchor_rows["window_start"].min().isoformat(),
                    "end": anchor_end.isoformat(),
                },
                "validation_range": {
                    "start": validation_start.isoformat(),
                    "end": validation_rows["window_start"].max().isoformat(),
                },
                "policy_selection": {
                    **asdict(selection),
                    "thresholds": dict(selection.thresholds),
                    "source_fold_index": config.policy_anchor_fold,
                    "reused_for_validation_fold": fold_index,
                    "single_frozen_policy": True,
                    "timing_qualification_role": "diagnostic_only",
                },
                "validation": {
                    "metrics": metrics,
                    "timing": timing,
                    "checks": validation_checks,
                    "qualified": validation_qualified,
                    "thresholds_frozen_before_access": True,
                    "threshold_search_performed": False,
                    "single_frozen_policy": True,
                },
            }
        )
    scored_all = pl.concat(scored_frames, how="vertical_relaxed").sort(
        ["observed_at", "market_id", "fold_index"]
    )
    selected_all = scored_all.filter(pl.col("policy_selected"))
    eligible_total = sum(
        len(validation_universes[index])
        if validation_universes is not None
        else len(observed_universes[index])
        for index in sorted(observed_universes)
    )
    metrics = classification_metrics(
        selected_all,
        eligible_markets=eligible_total,
    )
    timing = first_crossing_timing(
        selected_all,
        eligible_markets=eligible_total,
    )
    validation_qualified_folds = sum(bool(fold["validation"]["qualified"]) for fold in fold_results)
    result = {
        "candidate": candidate_name,
        "target_kind": candidate["target_kind"],
        "feature_kind": candidate["feature_kind"],
        "calibration_kind": candidate["calibration_kind"],
        "row_weight_schedule": candidate["row_weight_schedule"],
        "policy_selection_mode": config.policy_selection_mode,
        "single_policy_qualified": selection.qualified,
        "single_policy": {
            **asdict(selection),
            "thresholds": dict(selection.thresholds),
            "source_fold_index": config.policy_anchor_fold,
            "selection_range": {
                "start": anchor_rows["window_start"].min().isoformat(),
                "end": anchor_end.isoformat(),
            },
            "applied_validation_folds": len(fold_results),
            "timing_qualification_role": "diagnostic_only",
        },
        "folds": fold_results,
        "fold_count": len(fold_results),
        "policy_qualified_folds": (len(fold_results) if selection.qualified else 0),
        "validation_qualified_folds": validation_qualified_folds,
        "out_of_fold": metrics,
        "timing": timing,
        "no_trade_rate": 1.0 - metrics["coverage"],
        "validation_threshold_searches": 0,
        "validation_score_passes": len(fold_results),
    }
    return result, scored_all, observed_universes


def frequency_policy_advancement_checks(
    candidate: dict[str, Any],
    control: dict[str, Any],
    gates: FrequencyPolicyAdvancementGates,
    *,
    comparison: dict[str, Any] | None,
    is_control: bool,
    quantity: float,
) -> dict[str, Any]:
    metrics = candidate["out_of_fold"]
    checks = absolute_policy_checks(
        metrics,
        candidate["timing"],
        gates,
        include_timing=False,
    )
    checks.extend(
        (
            _check(
                "single frozen policy qualified before validation",
                int(candidate["single_policy_qualified"]),
                "==",
                1,
            ),
            _check(
                "validation qualified in every fold",
                candidate["validation_qualified_folds"],
                "==",
                candidate["fold_count"],
            ),
        )
    )
    if not is_control:
        control_metrics = control["out_of_fold"]
        checks.extend(
            (
                _check(
                    "coverage improves over control",
                    metrics["coverage"] - control_metrics["coverage"],
                    ">=",
                    gates.minimum_coverage_uplift,
                ),
                _check(
                    "accuracy does not regress from control",
                    metrics["accuracy"] - control_metrics["accuracy"],
                    ">=",
                    -gates.maximum_accuracy_regression,
                ),
                _check(
                    "balanced accuracy does not regress from control",
                    metrics["balanced_accuracy"] - control_metrics["balanced_accuracy"],
                    ">=",
                    -gates.maximum_balanced_accuracy_regression,
                ),
                _check(
                    "UP recall does not regress from control",
                    metrics["up_recall"] - control_metrics["up_recall"],
                    ">=",
                    -gates.maximum_direction_recall_regression,
                ),
                _check(
                    "DOWN recall does not regress from control",
                    metrics["down_recall"] - control_metrics["down_recall"],
                    ">=",
                    -gates.maximum_direction_recall_regression,
                ),
                _check(
                    "Wilson lower bound does not regress from control",
                    metrics["wilson_lower_95"] - control_metrics["wilson_lower_95"],
                    ">=",
                    0.0,
                ),
            )
        )
        control_folds = {
            int(fold["fold_index"]): fold["validation"]["metrics"] for fold in control["folds"]
        }
        for fold in candidate["folds"]:
            fold_index = int(fold["fold_index"])
            fold_metrics = fold["validation"]["metrics"]
            control_fold = control_folds[fold_index]
            checks.extend(
                (
                    _check(
                        f"fold {fold_index} accuracy does not regress",
                        fold_metrics["accuracy"] - control_fold["accuracy"],
                        ">=",
                        -gates.maximum_accuracy_regression,
                    ),
                    _check(
                        f"fold {fold_index} balanced accuracy does not regress",
                        fold_metrics["balanced_accuracy"] - control_fold["balanced_accuracy"],
                        ">=",
                        -gates.maximum_balanced_accuracy_regression,
                    ),
                    _check(
                        f"fold {fold_index} UP recall does not regress",
                        fold_metrics["up_recall"] - control_fold["up_recall"],
                        ">=",
                        -gates.maximum_direction_recall_regression,
                    ),
                    _check(
                        f"fold {fold_index} DOWN recall does not regress",
                        fold_metrics["down_recall"] - control_fold["down_recall"],
                        ">=",
                        -gates.maximum_direction_recall_regression,
                    ),
                )
            )
        if comparison is None:
            raise RuntimeError("frequency challenger lost fixed-checkpoint comparison")
        observed_checkpoints = tuple(
            int(row["seconds_elapsed"]) for row in comparison["checkpoints"]
        )
        if observed_checkpoints != FIXED_FREQUENCY_CHECKPOINTS:
            raise RuntimeError("frequency checkpoint contract changed")
        for checkpoint in comparison["checkpoints"]:
            second = int(checkpoint["seconds_elapsed"])
            checks.extend(
                (
                    _check(
                        f"{second}s minimum common exact-time markets",
                        checkpoint["common_markets"],
                        ">=",
                        gates.minimum_common_checkpoint_markets,
                    ),
                    _check(
                        f"{second}s common-time accuracy does not regress",
                        checkpoint["accuracy_delta"],
                        ">=",
                        -gates.maximum_accuracy_regression,
                    ),
                    _check(
                        f"{second}s common-time balanced accuracy does not regress",
                        checkpoint["balanced_accuracy_delta"],
                        ">=",
                        -gates.maximum_balanced_accuracy_regression,
                    ),
                    _check(
                        f"{second}s common-time UP recall does not regress",
                        checkpoint["up_recall_delta"],
                        ">=",
                        -gates.maximum_direction_recall_regression,
                    ),
                    _check(
                        f"{second}s common-time DOWN recall does not regress",
                        checkpoint["down_recall_delta"],
                        ">=",
                        -gates.maximum_direction_recall_regression,
                    ),
                )
            )
        execution = candidate["execution"]
        realized_net_per_share = (
            execution["realized_net_expectancy_per_trade"] / quantity
            if execution["realized_net_expectancy_per_trade"] is not None
            else None
        )
        checks.extend(
            (
                _check(
                    "minimum executable evaluation markets",
                    execution["economic_markets"],
                    ">=",
                    gates.minimum_executable_markets,
                ),
                _check(
                    "minimum mean direct edge per share",
                    execution["mean_direct_edge_per_share"],
                    ">",
                    gates.minimum_mean_direct_edge_per_share,
                ),
                _check(
                    "minimum realized net expectancy per share",
                    realized_net_per_share,
                    ">",
                    gates.minimum_realized_net_per_share,
                ),
            )
        )
    return {
        "checks": checks,
        "benchmark_passed": (not is_control and all(check["passed"] for check in checks)),
        "deployment_qualified": False,
        "timing_gates_applied": False,
        "timing_reported": True,
    }


def _execution_config(
    config: FrequencyPolicyBenchmarkConfig,
) -> ExecutionEvidenceConfig:
    manifest = json.loads((config.execution_evidence / "manifest.json").read_text())
    return ExecutionEvidenceConfig(
        range_start=datetime.fromisoformat(manifest["range_start"]),
        range_end=datetime.fromisoformat(manifest["range_end"]),
        output_dir=config.execution_evidence,
        sample_interval_seconds=int(manifest["sample_interval_seconds"]),
        min_seconds_after_open=int(manifest["min_seconds_after_open"]),
        max_seconds_after_open=int(manifest["max_seconds_after_open"]),
        freshness_seconds=int(manifest["freshness_seconds"]),
        quantity=float(manifest["quantity"]),
    )


def _check(
    name: str,
    observed: float | None,
    operator: str,
    required: float,
) -> dict[str, Any]:
    if observed is None:
        passed = False
    elif operator == ">=":
        passed = observed >= required
    elif operator == ">":
        passed = observed > required
    elif operator == "==":
        passed = observed == required
    else:
        raise ValueError(f"unsupported frequency gate operator: {operator}")
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "required": required,
        "passed": bool(passed),
    }


def _write_parquet_atomic(frame: pl.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    frame.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )
    os.replace(temporary, destination)
