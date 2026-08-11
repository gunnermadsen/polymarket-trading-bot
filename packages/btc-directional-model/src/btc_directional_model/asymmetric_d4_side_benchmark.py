"""Consumed-development benchmark for frozen-D4 side calibration.

The runner reuses the exact sealed D4 fold models and the incumbent OOF
predictions from one pinned parent run.  It never refits D4, changes the frozen
20--30 cent policy, exports a model, or authorizes a trading-process change.
Only the predeclared conservative NO intercepts are fitted.  The active evidence
is already-consumed development evidence, so every economic result is explicitly
diagnostic even when an arm clears its probability or economic thresholds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import struct
import sys
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .asymmetric_book_admission import (
    BOOK_ADMISSION_KEY_COLUMNS,
    BOOK_ADMISSION_SCHEMA_VERSION,
    DYNAMIC_FEATURES,
    LABEL_COLUMN,
    PROBABILITY_COLUMN,
    SELECTED_SIDE_COLUMN,
    SELECTED_SIDE_POLICY_ELIGIBLE_COLUMN,
    attach_frozen_policy_selected_side,
    book_admission_key_sha256,
    load_book_admission_config,
    load_book_admission_model,
    score_book_admission_model,
    select_target_opportunity_rows,
)
from .asymmetric_book_dynamics import (
    attach_causal_book_dynamics,
    book_dynamics_schema_sha256,
)
from .asymmetric_incumbent_benchmark import _load_core_oracle_value_frame
from .asymmetric_incumbent_calibration import (
    frozen_parent_probabilities,
    score_incumbent_calibration_payload,
)
from .asymmetric_incumbent_replay import load_frozen_asymmetric_incumbent
from .asymmetric_value_config import load_asymmetric_value_config
from .asymmetric_value_evaluation import EXECUTION_STRESS_PER_SHARE, ledger_metrics
from .core_config import load_core_config
from .core_extract import file_sha256, write_json_atomic

D4_SIDE_BENCHMARK_SCHEMA_VERSION = "btc-asymmetric-d4-side-benchmark-v1"
D4_SIDE_OUTCOME_SEAL_SCHEMA_VERSION = "btc-asymmetric-d4-side-outcome-access-v1"
D4_SIDE_SELECTION_SEAL_SCHEMA_VERSION = "btc-asymmetric-d4-side-selection-seal-v1"
D4_SIDE_BUNDLE_SCHEMA_VERSION = "btc-asymmetric-d4-side-bundle-v1"
INCUMBENT_ID = "I0"
D4_BASE_ID = "D4-base"
CALIBRATION_IDS = ("N1", "N2")
ARM_IDS = (INCUMBENT_ID, D4_BASE_ID, *CALIBRATION_IDS)
PARENT_CANDIDATES = ("S0", "D1", "D2", "D3", "D4")
EXPECTED_PARENT_ARTIFACTS = 71
EXPECTED_PARENT_FOLD_MODELS = 50
EXPECTED_STRICT_OOF_MARKETS = 2_692
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260810
SEALED_PREDICTION_COLUMNS = (
    *BOOK_ADMISSION_KEY_COLUMNS,
    "candidate_id",
    "probability_yes",
    SELECTED_SIDE_COLUMN,
)
PROBABILITY_CONTEXT_COLUMNS = (
    *BOOK_ADMISSION_KEY_COLUMNS,
    LABEL_COLUMN,
    "yes_ask_vwap_5",
    "no_ask_vwap_5",
    "yes_ask_depth",
    "no_ask_depth",
    "yes_cost_per_share",
    "no_cost_per_share",
)
PROJECTED_PNL_CONTEXT_COLUMNS = (
    *BOOK_ADMISSION_KEY_COLUMNS,
    LABEL_COLUMN,
    "fee_rate",
    "yes_best_ask",
    "yes_ask_vwap_5",
    "yes_ask_depth",
    "no_best_ask",
    "no_ask_vwap_5",
    "no_ask_depth",
    "yes_cost_per_share",
    "no_cost_per_share",
    "yes_execution_cost_per_share",
    "no_execution_cost_per_share",
)
IMPLEMENTATION_FILES = (
    "asymmetric_d4_side_benchmark.py",
    "asymmetric_d4_side_calibration.py",
    "asymmetric_d4_calibration_evaluation.py",
    "asymmetric_book_admission.py",
    "asymmetric_book_admission_benchmark.py",
    "asymmetric_book_admission_evaluation.py",
    "asymmetric_book_dynamics.py",
    "asymmetric_incumbent_benchmark.py",
    "asymmetric_incumbent_calibration.py",
    "asymmetric_incumbent_replay.py",
    "asymmetric_value_evaluation.py",
    "cli.py",
)


def run_d4_side_calibration_benchmark(
    config: Any,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Run the diagnostic I0/D4-base/N1/N2 comparison end to end."""

    calibration_api = _calibration_api()
    evaluation_api = _evaluation_api()
    calibration_api["validate_config"](config)

    # This must be the first parent-run operation.  In particular, no pickle is
    # deserialized until all 71 artifacts and both terminal parent seals verify.
    parent_integrity = validate_parent_run_integrity(config)
    parent_config = load_book_admission_config(
        config.paths.parent_book_admission_config
    )
    source_config = load_asymmetric_value_config(
        parent_config.paths.source_asymmetric_value_config
    )
    core_config = load_core_config(source_config.core_config)
    source, source_evidence = _load_core_oracle_value_frame(
        source_config,
        core_config=core_config,
        force=force,
    )
    if source_evidence.get("proxy_prices_used") is not False:
        raise RuntimeError("D4 side calibration requires exact PMXT prices")
    source = _window(source, config.development.start, config.development.end)
    incumbent = load_frozen_asymmetric_incumbent(parent_config.paths.incumbent_model)
    _validate_incumbent_identity(config, incumbent)
    incumbent_probability = score_incumbent_calibration_payload(
        incumbent,
        incumbent.payload,
        source,
        parent_probabilities=frozen_parent_probabilities(incumbent, source),
    )
    incumbent_oriented = attach_frozen_policy_selected_side(
        source.with_columns(
            pl.Series(PROBABILITY_COLUMN, incumbent_probability, dtype=pl.Float64)
        ),
        parent_config,
    )
    dynamic = attach_causal_book_dynamics(
        incumbent_oriented,
        maximum_book_age_seconds=parent_config.readiness_gates.maximum_book_age_seconds,
    )
    strict_validation = _strict_validation_union(dynamic, config)
    eligible_markets = (
        strict_validation.select("market_id", "window_start")
        .unique()
        .sort("window_start", "market_id")
    )
    if eligible_markets.height != EXPECTED_STRICT_OOF_MARKETS:
        raise RuntimeError(
            "frozen strict OOF universe changed: "
            f"{eligible_markets.height} != {EXPECTED_STRICT_OOF_MARKETS}"
        )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = config.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir()
    write_json_atomic(run_dir / "parent-integrity.json", parent_integrity)
    _write_root_contract(
        run_dir,
        config=config,
        parent_integrity=parent_integrity,
        source_evidence=source_evidence,
        strict_validation=strict_validation,
        eligible_markets=eligible_markets,
    )

    parent_manifest = _read_json(config.paths.parent_run / "fold-model-manifest.json")
    parent_d4_predictions = _read_parent_prediction(config, "D4")
    parent_i0_predictions = _read_parent_prediction(config, INCUMBENT_ID)
    validation_parts: list[pl.DataFrame] = []
    scored_parts: dict[str, list[pl.DataFrame]] = {
        D4_BASE_ID: [],
        "N1": [],
        "N2": [],
    }
    fit_records: list[dict[str, Any]] = []
    for fold in config.folds:
        _validate_fold(fold, config)
        model_record = parent_manifest["models"][f"D4/{fold.name}"]
        model = _load_verified_parent_d4_model(
            config,
            parent_config,
            fold_name=fold.name,
            record=model_record,
        )
        calibration_source = _window(dynamic, fold.calibration.start, fold.calibration.end)
        validation_source = _window(dynamic, fold.validation.start, fold.validation.end)
        calibration_score = score_book_admission_model(
            model,
            calibration_source.drop(LABEL_COLUMN),
            parent_config,
        )
        validation_score = score_book_admission_model(
            model,
            validation_source.drop(LABEL_COLUMN),
            parent_config,
        )
        calibration_frame = _attach_probability_orientation(
            calibration_source,
            calibration_score.frame,
            parent_config,
            include_label=True,
        )
        validation_frame = _attach_probability_orientation(
            validation_source,
            validation_score.frame,
            parent_config,
            include_label=False,
        )
        validation_parts.append(validation_frame)
        calibration_input_sha256 = _calibration_input_sha256(calibration_frame)

        for arm_id in (D4_BASE_ID, *CALIBRATION_IDS):
            calibrator = calibration_api["fit"](
                config,
                arm_id,
                calibration_frame,
                fold_name=fold.name,
            )
            scored = calibration_api["score"](calibrator, validation_frame)
            sealed = (
                scored.frame.rename(
                    {"candidate_name": "candidate_id", PROBABILITY_COLUMN: "probability_yes"}
                )
                .join(
                    validation_frame.select(
                        *BOOK_ADMISSION_KEY_COLUMNS, SELECTED_SIDE_COLUMN
                    ),
                    on=list(BOOK_ADMISSION_KEY_COLUMNS),
                    how="inner",
                    validate="1:1",
                )
                .select(*SEALED_PREDICTION_COLUMNS)
                .sort(*BOOK_ADMISSION_KEY_COLUMNS)
            )
            _assert_orientation_frozen(sealed, validation_frame, arm_id)
            if arm_id in CALIBRATION_IDS:
                _assert_yes_probability_unchanged(
                    sealed,
                    validation_frame,
                    arm_id,
                )
            scored_parts[arm_id].append(sealed)
            fit_records.append(
                {
                    "fold": fold.name,
                    "arm_id": arm_id,
                    "calibrator_semantic_sha256": calibrator.semantic_sha256,
                    "no_logit_offsets": [list(value) for value in calibrator.no_logit_offsets],
                    "calibration_input_content_sha256": calibration_input_sha256,
                    "fit_evidence": _json_safe(asdict(calibrator.evidence)),
                    "support_qualification_allowed": False,
                    "qualification_eligible": False,
                }
            )
        del model

    d4_context = pl.concat(validation_parts).sort(*BOOK_ADMISSION_KEY_COLUMNS)
    d4_base = pl.concat(scored_parts[D4_BASE_ID]).sort(*BOOK_ADMISSION_KEY_COLUMNS)
    _assert_reproduced_parent_d4(d4_base, parent_d4_predictions)
    i0 = _orient_parent_prediction(
        parent_i0_predictions,
        d4_context,
        parent_config,
        candidate_id=INCUMBENT_ID,
    )
    arm_predictions = {
        INCUMBENT_ID: i0,
        D4_BASE_ID: d4_base,
        "N1": pl.concat(scored_parts["N1"]).sort(*BOOK_ADMISSION_KEY_COLUMNS),
        "N2": pl.concat(scored_parts["N2"]).sort(*BOOK_ADMISSION_KEY_COLUMNS),
    }
    prediction_paths, prediction_hashes = _write_prediction_artifacts(
        run_dir,
        arm_predictions,
    )
    fit_manifest = {
        "schema_version": D4_SIDE_BENCHMARK_SCHEMA_VERSION,
        "evidence_scope": config.evidence_scope,
        "consumed_validation_role": config.consumed_validation_role,
        "parent_d4_refitted": False,
        "selected_side_source": "D4-base probability and frozen policy",
        "selected_side_reoriented_after_calibration": False,
        "qualification_eligible": False,
        "support_qualification_allowed": False,
        "fits": fit_records,
        "hard_support_failures": [
            {
                "fold": record["fold"],
                "arm_id": record["arm_id"],
                "failures": record["fit_evidence"]["support_failures"],
            }
            for record in fit_records
            if record["fit_evidence"]["support_failures"]
        ],
    }
    fit_manifest_path = run_dir / "calibration-fit-manifest.json"
    write_json_atomic(fit_manifest_path, fit_manifest)
    probability_support_evidence = _probability_support_evidence(fit_records)
    support_manifest_path = run_dir / "calibration-support-manifest.json"
    write_json_atomic(
        support_manifest_path,
        {
            "schema_version": D4_SIDE_BENCHMARK_SCHEMA_VERSION,
            "evidence_scope": config.evidence_scope,
            "consumed_validation_role": config.consumed_validation_role,
            "qualification_eligible": False,
            "support_qualification_allowed": False,
            "arms": probability_support_evidence,
            "all_causal_folds_passed": all(
                value["support_passed"]
                for value in probability_support_evidence.values()
            ),
        },
    )
    prediction_manifest_path = run_dir / "prediction-manifest.json"
    write_json_atomic(
        prediction_manifest_path,
        {
            "schema_version": D4_SIDE_BENCHMARK_SCHEMA_VERSION,
            "arm_ids": list(ARM_IDS),
            "prediction_paths": {
                arm_id: str(path.relative_to(run_dir))
                for arm_id, path in prediction_paths.items()
            },
            "prediction_sha256": prediction_hashes,
            "parent_d4_oof_probability_reproduced_exactly": True,
            "parent_d4_oof_prediction_sha256": (
                config.parent_run.d4_oof_predictions_sha256
            ),
            "selected_side_reoriented_after_calibration": False,
            "labels_present": False,
        },
    )
    probability_manifest_path = _write_probability_artifact_manifest(
        run_dir,
        config=config,
        prediction_manifest_path=prediction_manifest_path,
        fit_manifest_path=fit_manifest_path,
        support_manifest_path=support_manifest_path,
    )
    outcome_seal_path = _write_outcome_access_seal(
        run_dir,
        probability_manifest_path=probability_manifest_path,
        prediction_manifest_path=prediction_manifest_path,
        fit_manifest_path=fit_manifest_path,
        support_manifest_path=support_manifest_path,
        prediction_paths=prediction_paths,
        prediction_hashes=prediction_hashes,
    )

    # OOF outcomes become accessible only after every arm's probability and
    # frozen orientation have been materialized and sealed above.
    target_with_outcomes = select_target_opportunity_rows(
        strict_validation,
        parent_config,
        require_label=True,
        feature_names=DYNAMIC_FEATURES,
    )
    probability_context = target_with_outcomes.select(*PROBABILITY_CONTEXT_COLUMNS)
    probability_context_digest = _frame_content_sha256(
        probability_context,
        columns=PROBABILITY_CONTEXT_COLUMNS,
        domain="btc-asymmetric-d4-side-probability-context-v1",
    )
    probability_context_path = run_dir / "probability-context.json"
    write_json_atomic(
        probability_context_path,
        {
            "schema_version": D4_SIDE_BENCHMARK_SCHEMA_VERSION,
            "accessed_after_outcome_seal_sha256": file_sha256(outcome_seal_path),
            "columns": list(PROBABILITY_CONTEXT_COLUMNS),
            "rows": probability_context.height,
            "markets": probability_context["market_id"].n_unique(),
            "content_sha256": probability_context_digest,
            "economic_outcomes_present": False,
        },
    )
    probability_frames = {
        arm_id: _reload_sealed_prediction(
            prediction_paths[arm_id],
            prediction_hashes[arm_id],
            probability_context,
        )
        for arm_id in ARM_IDS
    }
    support_evidence = probability_support_evidence
    selection_thresholds = evaluation_api["selection_thresholds"](
        noninferiority_margin=(
            config.probability_gates.maximum_paired_degradation_upper_95
        ),
        maximum_selected_bias=(
            config.probability_gates.maximum_selected_opportunity_bias
        ),
        maximum_selected_yes_bias=config.probability_gates.maximum_yes_selected_bias,
        maximum_selected_no_bias=config.probability_gates.maximum_no_selected_bias,
        minimum_no_bias_reduction=(
            config.probability_gates.minimum_no_selected_bias_reduction
        ),
        maximum_cell_overconfidence=0.05,
        maximum_cell_overconfidence_regression=0.0,
        minimum_joint_noninferior_days=config.probability_gates.minimum_noninferior_days,
        required_validation_days=config.probability_gates.required_comparison_days,
        minimum_selected_yes=20,
        minimum_selected_no=20,
    )
    selection = evaluation_api["select"](
        probability_frames[INCUMBENT_ID],
        probability_frames[D4_BASE_ID],
        {arm_id: probability_frames[arm_id] for arm_id in CALIBRATION_IDS},
        support_evidence,
        selection_thresholds,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        evidence_scope=config.evidence_scope,
        qualification_eligible=False,
    )
    if selection.get("economics_used") not in (False, None) or selection.get(
        "selection_uses_economics"
    ) not in (False, None):
        raise RuntimeError("D4 side probability selection accessed economics")
    selection["qualification_eligible"] = False
    selection["evidence_scope"] = config.evidence_scope
    selection["consumed_validation_role"] = config.consumed_validation_role
    selection["economics_used"] = False
    selection_path = run_dir / "probability-selection.json"
    write_json_atomic(selection_path, selection)
    selection_seal_path, selection_seal = _write_probability_selection_seal(
        run_dir,
        config=config,
        selection_path=selection_path,
        outcome_seal_path=outcome_seal_path,
        prediction_manifest_path=prediction_manifest_path,
        fit_manifest_path=fit_manifest_path,
        probability_context_path=probability_context_path,
        selected_candidate_id=_selected_candidate_id(selection),
    )

    outcome_context = target_with_outcomes.select(*PROJECTED_PNL_CONTEXT_COLUMNS)
    projected_pnl_context_digest = _frame_content_sha256(
        outcome_context,
        columns=PROJECTED_PNL_CONTEXT_COLUMNS,
        domain="btc-asymmetric-d4-side-projected-pnl-context-v1",
    )
    projected_pnl_context_path = run_dir / "projected-pnl-context.json"
    write_json_atomic(
        projected_pnl_context_path,
        {
            "schema_version": D4_SIDE_BENCHMARK_SCHEMA_VERSION,
            "probability_selection_seal_sha256": file_sha256(selection_seal_path),
            "columns": list(PROJECTED_PNL_CONTEXT_COLUMNS),
            "rows": outcome_context.height,
            "markets": outcome_context["market_id"].n_unique(),
            "content_sha256": projected_pnl_context_digest,
            "opened_for_projected_pnl_only_after_probability_selection": True,
        },
    )
    economics_access_path = run_dir / "projected-pnl-access.json"
    write_json_atomic(
        economics_access_path,
        {
            "schema_version": D4_SIDE_BENCHMARK_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "probability_selection_seal_sha256": file_sha256(selection_seal_path),
            "probability_selection_completed_before_pnl": True,
            "selection_uses_projected_pnl": False,
            "evaluated_arm_ids": list(ARM_IDS),
            "projected_pnl_context_sha256": file_sha256(projected_pnl_context_path),
            "projected_pnl_context_content_sha256": projected_pnl_context_digest,
            "diagnostic_only": True,
            "qualification_eligible": False,
        },
    )
    economic_frames = {
        arm_id: _reload_sealed_prediction(
            prediction_paths[arm_id],
            prediction_hashes[arm_id],
            outcome_context,
        )
        for arm_id in ARM_IDS
    }
    ledgers = {
        arm_id: _frozen_orientation_ledger(
            economic_frames[arm_id],
            parent_config.policy,
            arm_id=arm_id,
        )
        for arm_id in ARM_IDS
    }
    ledger_dir = run_dir / "projected-pnl-ledgers"
    ledger_dir.mkdir()
    ledger_hashes: dict[str, str] = {}
    for arm_id, ledger in ledgers.items():
        path = ledger_dir / f"{arm_id}.parquet"
        ledger.write_parquet(path, compression="zstd", statistics=True)
        ledger_hashes[arm_id] = file_sha256(path)
    projected_thresholds = evaluation_api["projected_pnl_thresholds"](
        minimum_profit_factor=config.economic_gates.minimum_profit_factor,
        minimum_no_stressed_pnl_improvement=(
            config.economic_gates.minimum_no_stressed_pnl_improvement
        ),
        maximum_yes_stressed_pnl_regression_fraction=(
            config.economic_gates.maximum_yes_stressed_pnl_regression_fraction
        ),
        maximum_drawdown_regression_fraction=(
            config.economic_gates.maximum_drawdown_regression_fraction
        ),
        maximum_loss_recovery_regression_fraction=0.10,
    )
    projected = evaluation_api["projected_pnl"](
        ledgers,
        eligible_markets,
        selection,
        selection_seal,
        selection_seal_sha256=file_sha256(selection_seal_path),
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        thresholds=projected_thresholds,
    )
    projected_models = {
        arm_id: _projected_model_record(
            ledgers[arm_id],
            arm_id=arm_id,
            eligible_market_count=eligible_markets.height,
            selected_candidate_id=_selected_candidate_id(selection),
        )
        for arm_id in ARM_IDS
    }
    projected_payload = {
        "schema_version": D4_SIDE_BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "evidence_scope": config.evidence_scope,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "forward_proof": False,
        "runtime_deployable": False,
        "batch_forward_eligible": False,
        "source_process_changed": False,
        "selection_uses_projected_pnl": False,
        "probability_selection_seal_sha256": file_sha256(selection_seal_path),
        "projected_pnl_access_sha256": file_sha256(economics_access_path),
        "probability_context_sha256": file_sha256(probability_context_path),
        "probability_context_content_sha256": probability_context_digest,
        "projected_pnl_context_sha256": file_sha256(projected_pnl_context_path),
        "projected_pnl_context_content_sha256": projected_pnl_context_digest,
        "eligible_market_denominator": eligible_markets.height,
        "eligible_market_scope": "full strict parent OOF validation universe",
        "arm_order": list(ARM_IDS),
        "models": projected_models,
        "paired_evaluation": projected,
        "ledger_sha256": ledger_hashes,
    }
    projected_path = run_dir / "projected-pnl.json"
    write_json_atomic(projected_path, projected_payload)
    _write_projected_pnl_csv(run_dir / "projected-pnl.csv", projected_models)
    _write_projected_pnl_markdown(
        run_dir / "projected-pnl.md",
        projected_payload,
    )

    result = {
        "schema_version": D4_SIDE_BENCHMARK_SCHEMA_VERSION,
        "status": "consumed_development_diagnostic_complete",
        "evidence_scope": config.evidence_scope,
        "consumed_validation_role": config.consumed_validation_role,
        "probability_selected_candidate_id": _selected_candidate_id(selection),
        "projected_pnl_reported_for_all_arms": True,
        "eligible_market_denominator": eligible_markets.height,
        "qualification_eligible": False,
        "support_qualification_allowed": False,
        "forward_proof": False,
        "paper_artifact": None,
        "batch_forward_artifact": None,
        "runtime_deployable": False,
        "source_process_changed": False,
        "parent_d4_refitted": False,
        "selected_side_reoriented_after_calibration": False,
        "economics_used_for_selection": False,
        "probability_selection_seal_sha256": file_sha256(selection_seal_path),
        "projected_pnl_sha256": file_sha256(projected_path),
    }
    _finalize_run(
        run_dir,
        result=result,
        selection=selection,
        projected=projected_payload,
    )
    return run_dir, result


def validate_parent_run_integrity(config: Any) -> dict[str, Any]:
    """Verify the complete sealed parent run before any model deserialization."""

    parent = config.paths.parent_run.resolve()
    bundle_path = parent / "benchmark-bundle-seal.json"
    if file_sha256(bundle_path) != config.parent_run.benchmark_bundle_seal_sha256:
        raise RuntimeError("pinned parent benchmark bundle seal changed")
    bundle = _read_json(bundle_path)
    aggregate = bundle.get("aggregate_sha256")
    if aggregate != _canonical_sha256(
        {key: value for key, value in bundle.items() if key != "aggregate_sha256"}
    ):
        raise RuntimeError("parent benchmark bundle aggregate is invalid")
    manifest_path = parent / "artifact-manifest.json"
    if file_sha256(manifest_path) != bundle.get("artifact_manifest_sha256"):
        raise RuntimeError("parent artifact manifest is not bound by its bundle seal")
    manifest = _validate_parent_artifact_manifest(
        parent,
        manifest_path=manifest_path,
        expected_artifacts=EXPECTED_PARENT_ARTIFACTS,
    )
    pinned = {
        "root-training-contract.json": config.parent_run.root_training_contract_sha256,
        "fold-model-manifest.json": config.parent_run.fold_model_manifest_sha256,
        "prediction-manifest.json": config.parent_run.prediction_manifest_sha256,
        "probability-selection.json": config.parent_run.probability_selection_sha256,
        "probability-selection-seal.json": (
            config.parent_run.probability_selection_seal_sha256
        ),
        "candidates/D4/oof-predictions.parquet": (
            config.parent_run.d4_oof_predictions_sha256
        ),
    }
    for relative, expected in pinned.items():
        if manifest["artifacts"].get(relative) != expected:
            raise RuntimeError(f"parent manifest lost pinned artifact: {relative}")
    _validate_parent_root_contract(parent / "root-training-contract.json", config)
    prediction_evidence = _validate_parent_prediction_manifest(parent, manifest)
    fold_evidence = _validate_parent_fold_manifest(parent, manifest, config)
    return {
        "schema_version": D4_SIDE_BENCHMARK_SCHEMA_VERSION,
        "validated_at": datetime.now(UTC).isoformat(),
        "parent_run": str(parent),
        "parent_run_id": config.parent_run.run_id,
        "artifact_manifest_sha256": file_sha256(manifest_path),
        "artifact_manifest_aggregate_sha256": manifest["aggregate_sha256"],
        "benchmark_bundle_seal_sha256": file_sha256(bundle_path),
        "validated_artifact_count": len(manifest["artifacts"]),
        "validated_fold_model_count": fold_evidence["model_count"],
        "validated_prediction_count": prediction_evidence["prediction_count"],
        "all_parent_artifacts_validated_before_deserialization": True,
        "parent_d4_refit_allowed": False,
    }


def _validate_parent_artifact_manifest(
    parent: Path,
    *,
    manifest_path: Path,
    expected_artifacts: int,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != expected_artifacts:
        raise RuntimeError(
            f"parent artifact manifest must contain exactly {expected_artifacts} artifacts"
        )
    if manifest.get("aggregate_sha256") != _canonical_sha256(artifacts):
        raise RuntimeError("parent artifact manifest aggregate is invalid")
    parent_resolved = parent.resolve()
    for relative, expected in sorted(artifacts.items()):
        path = (parent_resolved / relative).resolve()
        if not path.is_relative_to(parent_resolved):
            raise RuntimeError("parent artifact manifest contains an unsafe path")
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"parent artifact changed: {relative}")
    return manifest


def _validate_parent_root_contract(path: Path, config: Any) -> None:
    root = _read_json(path)
    if (
        root.get("process_id") != config.incumbent.process_id
        or root.get("incumbent_model_key") != config.incumbent.model_key
        or root.get("incumbent_model_sha256") != config.incumbent.model_sha256
        or root.get("profile") != "btc_asymmetric_core_oracle_book_admission"
        or root.get("paper_only") is not True
        or root.get("runtime_deployable") is not False
        or root.get("process_change_allowed") is not False
        or root.get("recursive_search", {}).get("candidate_ids")
        != list(PARENT_CANDIDATES)
        or root.get("recursive_search", {}).get("pnl_based_candidate_selection_allowed")
        is not False
    ):
        raise RuntimeError("parent root training contract changed")


def _validate_parent_prediction_manifest(
    parent: Path,
    artifact_manifest: Mapping[str, Any],
) -> dict[str, int]:
    payload = _read_json(parent / "prediction-manifest.json")
    expected_ids = {INCUMBENT_ID, *PARENT_CANDIDATES}
    paths = payload.get("prediction_paths", {})
    hashes = payload.get("prediction_sha256", {})
    if set(paths) != expected_ids or set(hashes) != expected_ids:
        raise RuntimeError("parent prediction manifest candidate set changed")
    for candidate_id in sorted(expected_ids):
        relative = paths[candidate_id]
        if hashes[candidate_id] != artifact_manifest["artifacts"].get(relative):
            raise RuntimeError(f"parent prediction lineage changed: {candidate_id}")
    return {"prediction_count": len(expected_ids)}


def _validate_parent_fold_manifest(
    parent: Path,
    artifact_manifest: Mapping[str, Any],
    config: Any,
) -> dict[str, int]:
    payload = _read_json(parent / "fold-model-manifest.json")
    models = payload.get("models", {})
    expected = {
        f"{candidate}/{day.isoformat()}"
        for candidate in PARENT_CANDIDATES
        for day in config.oof_utc_days
    }
    if (
        payload.get("usable_as_final_batch_model") is not False
        or set(models) != expected
        or len(models) != EXPECTED_PARENT_FOLD_MODELS
    ):
        raise RuntimeError("parent fold-model manifest changed")
    for identity, record in sorted(models.items()):
        relative = record.get("path")
        artifact_hash = record.get("artifact_sha256")
        serialization = record.get("serialization", {})
        if (
            artifact_manifest["artifacts"].get(relative) != artifact_hash
            or serialization.get("artifact_sha256") != artifact_hash
            or serialization.get("model_semantic_sha256")
            != record.get("semantic_sha256")
            or serialization.get("schema_version") != BOOK_ADMISSION_SCHEMA_VERSION
            or serialization.get("candidate_name") != identity.split("/", 1)[0]
        ):
            raise RuntimeError(f"parent fold-model lineage changed: {identity}")
    return {"model_count": len(models)}


def _load_verified_parent_d4_model(
    config: Any,
    parent_config: Any,
    *,
    fold_name: str,
    record: Mapping[str, Any],
) -> Any:
    path = config.paths.parent_run / str(record["path"])
    if file_sha256(path) != record.get("artifact_sha256"):
        raise RuntimeError("D4 model bytes changed before deserialization")
    model = load_book_admission_model(path)
    expected_candidate = parent_config.candidate("D4")
    if (
        model.schema_version != BOOK_ADMISSION_SCHEMA_VERSION
        or model.candidate != expected_candidate
        or model.incumbent != parent_config.incumbent
        or model.dynamics_schema_sha256 != book_dynamics_schema_sha256()
        or model.semantic_sha256 != record.get("semantic_sha256")
        or record.get("serialization", {}).get("model_semantic_sha256")
        != model.semantic_sha256
        or not str(record.get("path", "")).endswith(f"D4/{fold_name}/model.pkl")
    ):
        raise RuntimeError(f"pinned D4 fold model contract changed: {fold_name}")
    return model


def _read_parent_prediction(config: Any, candidate_id: str) -> pl.DataFrame:
    manifest = _read_json(config.paths.parent_run / "prediction-manifest.json")
    path = config.paths.parent_run / manifest["prediction_paths"][candidate_id]
    expected = manifest["prediction_sha256"][candidate_id]
    if file_sha256(path) != expected:
        raise RuntimeError(f"parent {candidate_id} OOF predictions changed")
    frame = pl.read_parquet(path).sort(*BOOK_ADMISSION_KEY_COLUMNS)
    required = {*BOOK_ADMISSION_KEY_COLUMNS, "candidate_id", "probability_yes"}
    if set(frame.columns) != required or frame.select(
        *BOOK_ADMISSION_KEY_COLUMNS
    ).is_duplicated().any():
        raise RuntimeError(f"parent {candidate_id} prediction schema changed")
    if frame["candidate_id"].unique().to_list() != [candidate_id]:
        raise RuntimeError(f"parent {candidate_id} prediction identity changed")
    return frame


def _attach_probability_orientation(
    source: pl.DataFrame,
    score: pl.DataFrame,
    parent_config: Any,
    *,
    include_label: bool,
) -> pl.DataFrame:
    keys = list(BOOK_ADMISSION_KEY_COLUMNS)
    score_probability = score.select(*keys, PROBABILITY_COLUMN)
    drop_columns = [
        value
        for value in (
            PROBABILITY_COLUMN,
            SELECTED_SIDE_COLUMN,
            SELECTED_SIDE_POLICY_ELIGIBLE_COLUMN,
            LABEL_COLUMN,
        )
        if value in source.columns and (value != LABEL_COLUMN or not include_label)
    ]
    context = source.drop(*drop_columns)
    joined = context.join(
        score_probability,
        on=keys,
        how="inner",
        validate="1:1",
    )
    if joined.height != score.height:
        raise RuntimeError("D4 score keys do not match their source context")
    oriented = attach_frozen_policy_selected_side(joined, parent_config)
    if include_label and LABEL_COLUMN not in oriented.columns:
        labels = source.select(*keys, LABEL_COLUMN)
        oriented = oriented.join(labels, on=keys, how="inner", validate="1:1")
    return oriented.sort(*BOOK_ADMISSION_KEY_COLUMNS)


def _orient_parent_prediction(
    prediction: pl.DataFrame,
    d4_context: pl.DataFrame,
    parent_config: Any,
    *,
    candidate_id: str,
) -> pl.DataFrame:
    keys = list(BOOK_ADMISSION_KEY_COLUMNS)
    drop = [
        value
        for value in (
            PROBABILITY_COLUMN,
            SELECTED_SIDE_COLUMN,
            SELECTED_SIDE_POLICY_ELIGIBLE_COLUMN,
        )
        if value in d4_context.columns
    ]
    context = d4_context.drop(*drop)
    joined = context.join(
        prediction.select(*keys, "probability_yes").rename(
            {"probability_yes": PROBABILITY_COLUMN}
        ),
        on=keys,
        how="inner",
        validate="1:1",
    )
    if joined.height != prediction.height:
        raise RuntimeError("parent I0 and D4 prediction grids differ")
    return (
        attach_frozen_policy_selected_side(joined, parent_config)
        .select(
            *keys,
            pl.lit(candidate_id).alias("candidate_id"),
            pl.col(PROBABILITY_COLUMN).alias("probability_yes"),
            SELECTED_SIDE_COLUMN,
        )
        .sort(*keys)
    )


def _assert_reproduced_parent_d4(
    reproduced: pl.DataFrame,
    parent: pl.DataFrame,
) -> None:
    keys = list(BOOK_ADMISSION_KEY_COLUMNS)
    reproduced = reproduced.sort(*keys)
    parent = parent.sort(*keys)
    if (
        reproduced.height != parent.height
        or not reproduced.select(*keys).equals(parent.select(*keys), null_equal=True)
        or not np.array_equal(
            reproduced["probability_yes"].to_numpy(),
            parent["probability_yes"].to_numpy(),
        )
    ):
        raise RuntimeError("reused D4 fold models did not exactly reproduce parent OOF")


def _assert_orientation_frozen(
    scored: pl.DataFrame,
    base_context: pl.DataFrame,
    arm_id: str,
) -> None:
    expected = base_context.select(
        *BOOK_ADMISSION_KEY_COLUMNS, SELECTED_SIDE_COLUMN
    ).sort(*BOOK_ADMISSION_KEY_COLUMNS)
    observed = scored.select(
        *BOOK_ADMISSION_KEY_COLUMNS, SELECTED_SIDE_COLUMN
    ).sort(*BOOK_ADMISSION_KEY_COLUMNS)
    if not observed.equals(expected, null_equal=True):
        raise RuntimeError(f"{arm_id} recursively changed D4-base side orientation")


def _assert_yes_probability_unchanged(
    scored: pl.DataFrame,
    base_context: pl.DataFrame,
    arm_id: str,
) -> None:
    keys = list(BOOK_ADMISSION_KEY_COLUMNS)
    comparison = (
        scored.join(
            base_context.select(*keys, PROBABILITY_COLUMN),
            on=keys,
            how="inner",
            validate="1:1",
        )
        .filter(pl.col(SELECTED_SIDE_COLUMN) == "YES")
        .sort(*keys)
    )
    if not np.array_equal(
        comparison["probability_yes"].to_numpy(),
        comparison[PROBABILITY_COLUMN].to_numpy(),
    ):
        raise RuntimeError(f"{arm_id} changed a frozen D4 YES probability")


def _calibration_input_sha256(frame: pl.DataFrame) -> str:
    required = (
        *BOOK_ADMISSION_KEY_COLUMNS,
        PROBABILITY_COLUMN,
        SELECTED_SIDE_COLUMN,
        LABEL_COLUMN,
    )
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError("calibration input digest missing: " + ", ".join(missing))
    ordered = frame.select(*required).sort(*BOOK_ADMISSION_KEY_COLUMNS)
    digest = hashlib.sha256(b"btc-asymmetric-d4-side-calibration-input-v1\n")
    digest.update(book_admission_key_sha256(ordered).encode())
    digest.update(
        ordered[PROBABILITY_COLUMN]
        .cast(pl.Float64)
        .to_numpy()
        .astype("<f8", copy=False)
        .tobytes()
    )
    for side in ordered[SELECTED_SIDE_COLUMN].cast(pl.String).to_list():
        if side not in {"YES", "NO"}:
            raise ValueError("calibration input digest received an invalid side")
        digest.update(b"Y" if side == "YES" else b"N")
    digest.update(
        ordered[LABEL_COLUMN]
        .cast(pl.Float64)
        .to_numpy()
        .astype("<f8", copy=False)
        .tobytes()
    )
    return digest.hexdigest()


def _frame_content_sha256(
    frame: pl.DataFrame,
    *,
    columns: tuple[str, ...],
    domain: str,
) -> str:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{domain} digest missing: " + ", ".join(missing))
    ordered = frame.select(*columns).sort(*BOOK_ADMISSION_KEY_COLUMNS)
    digest = hashlib.sha256(f"{domain}\n".encode())
    for row in ordered.iter_rows():
        for value in row:
            _update_content_digest(digest, value)
    return digest.hexdigest()


def _update_content_digest(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"n")
    elif isinstance(value, bool):
        digest.update(b"b1" if value else b"b0")
    elif isinstance(value, datetime):
        encoded = value.isoformat().encode()
        digest.update(b"t" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, int):
        digest.update(b"i" + int(value).to_bytes(16, "big", signed=True))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("content digest cannot bind a non-finite float")
        digest.update(b"f" + struct.pack("<d", value))
    else:
        encoded = str(value).encode()
        digest.update(b"s" + len(encoded).to_bytes(8, "big") + encoded)


def _write_prediction_artifacts(
    run_dir: Path,
    predictions: Mapping[str, pl.DataFrame],
) -> tuple[dict[str, Path], dict[str, str]]:
    if tuple(predictions) != ARM_IDS:
        raise RuntimeError("D4 side candidate order or identity changed")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    control_keys: pl.DataFrame | None = None
    for arm_id in ARM_IDS:
        frame = predictions[arm_id].select(*SEALED_PREDICTION_COLUMNS).sort(
            *BOOK_ADMISSION_KEY_COLUMNS
        )
        if set(frame.columns) != set(SEALED_PREDICTION_COLUMNS):
            raise RuntimeError("sealed D4 side prediction schema changed")
        if frame.select(*BOOK_ADMISSION_KEY_COLUMNS).is_duplicated().any():
            raise RuntimeError("sealed D4 side predictions contain duplicate keys")
        if frame["candidate_id"].unique().to_list() != [arm_id]:
            raise RuntimeError("sealed D4 side prediction identity changed")
        keys = frame.select(*BOOK_ADMISSION_KEY_COLUMNS)
        if control_keys is None:
            control_keys = keys
        elif not keys.equals(control_keys, null_equal=True):
            raise RuntimeError("D4 side candidate predictions do not share exact keys")
        path = run_dir / "candidates" / arm_id / "oof-predictions.parquet"
        path.parent.mkdir(parents=True, exist_ok=False)
        frame.write_parquet(path, compression="zstd", statistics=True)
        paths[arm_id] = path
        hashes[arm_id] = file_sha256(path)
    return paths, hashes


def _write_probability_artifact_manifest(
    run_dir: Path,
    *,
    config: Any,
    prediction_manifest_path: Path,
    fit_manifest_path: Path,
    support_manifest_path: Path,
) -> Path:
    internal_names = (
        "parent-integrity.json",
        "root-training-contract.json",
        prediction_manifest_path.name,
        fit_manifest_path.name,
        support_manifest_path.name,
    )
    implementation_root = Path(__file__).resolve().parent
    implementation = {
        name: file_sha256(implementation_root / name) for name in IMPLEMENTATION_FILES
    }
    payload = {
        "schema_version": D4_SIDE_BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "internal_artifact_sha256": {
            name: file_sha256(run_dir / name) for name in internal_names
        },
        "external_contract_sha256": {
            "config": file_sha256(config.source_path),
            "parent_book_admission_config": file_sha256(
                config.paths.parent_book_admission_config
            ),
            "parent_bundle_seal": file_sha256(
                config.paths.parent_run / "benchmark-bundle-seal.json"
            ),
            "parent_artifact_manifest": file_sha256(
                config.paths.parent_run / "artifact-manifest.json"
            ),
            "implementation": implementation,
            "dependency_identity": _dependency_identity(config),
        },
        "validation_outcomes_opened": False,
        "projected_pnl_opened": False,
    }
    payload["aggregate_sha256"] = _canonical_sha256(
        {
            "internal": payload["internal_artifact_sha256"],
            "external": payload["external_contract_sha256"],
        }
    )
    path = run_dir / "probability-artifact-manifest.json"
    write_json_atomic(path, payload)
    return path


def _write_outcome_access_seal(
    run_dir: Path,
    *,
    probability_manifest_path: Path,
    prediction_manifest_path: Path,
    fit_manifest_path: Path,
    support_manifest_path: Path,
    prediction_paths: Mapping[str, Path],
    prediction_hashes: Mapping[str, str],
) -> Path:
    for arm_id in ARM_IDS:
        path = prediction_paths[arm_id]
        if file_sha256(path) != prediction_hashes[arm_id]:
            raise RuntimeError("D4 side prediction changed before outcome seal")
        if tuple(pl.read_parquet_schema(path).names()) != SEALED_PREDICTION_COLUMNS:
            raise RuntimeError("D4 side sealed prediction allowlist changed")
    path = run_dir / "outcome-access-seal.json"
    write_json_atomic(
        path,
        {
            "schema_version": D4_SIDE_OUTCOME_SEAL_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "probability_artifact_manifest_sha256": file_sha256(
                probability_manifest_path
            ),
            "prediction_manifest_sha256": file_sha256(prediction_manifest_path),
            "calibration_fit_manifest_sha256": file_sha256(fit_manifest_path),
            "calibration_support_manifest_sha256": file_sha256(
                support_manifest_path
            ),
            "prediction_sha256": dict(prediction_hashes),
            "validation_labels_opened_after_this_seal": True,
            "strictly_prior_calibration_labels_used_before_this_seal": True,
            "projected_pnl_opened": False,
            "selection_uses_projected_pnl": False,
        },
    )
    return path


def _reload_sealed_prediction(
    path: Path,
    expected_sha256: str,
    context: pl.DataFrame,
) -> pl.DataFrame:
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("sealed D4 side prediction changed")
    prediction = pl.read_parquet(path)
    keys = list(BOOK_ADMISSION_KEY_COLUMNS)
    if (
        prediction.height != context.height
        or prediction.select(*keys).is_duplicated().any()
        or context.select(*keys).is_duplicated().any()
        or not prediction.select(*keys)
        .sort(*keys)
        .equals(context.select(*keys).sort(*keys), null_equal=True)
    ):
        raise RuntimeError("sealed D4 side predictions do not match outcome context")
    joined = prediction.join(context, on=keys, how="inner", validate="1:1")
    if joined.height != context.height:
        raise RuntimeError("D4 side outcome join changed sealed keys")
    return joined.sort(*keys)


def _write_probability_selection_seal(
    run_dir: Path,
    *,
    config: Any,
    selection_path: Path,
    outcome_seal_path: Path,
    prediction_manifest_path: Path,
    fit_manifest_path: Path,
    probability_context_path: Path,
    selected_candidate_id: str | None,
) -> tuple[Path, dict[str, Any]]:
    payload = {
        "schema_version": D4_SIDE_SELECTION_SEAL_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "selected_candidate_id": selected_candidate_id,
        "probability_selection_sha256": file_sha256(selection_path),
        "outcome_access_seal_sha256": file_sha256(outcome_seal_path),
        "prediction_manifest_sha256": file_sha256(prediction_manifest_path),
        "calibration_fit_manifest_sha256": file_sha256(fit_manifest_path),
        "probability_context_sha256": file_sha256(probability_context_path),
        "economics_opened": False,
        "selection_uses_economics": False,
        "evidence_scope": config.evidence_scope,
        "qualification_eligible": False,
        "forward_proof": False,
        "paper_export_allowed": False,
        "process_change_allowed": False,
    }
    path = run_dir / "probability-selection-seal.json"
    write_json_atomic(path, payload)
    return path, payload


def _frozen_orientation_ledger(
    frame: pl.DataFrame,
    policy: Any,
    *,
    arm_id: str,
) -> pl.DataFrame:
    """Apply the fixed policy without allowing calibrated probabilities to switch side."""

    required = {
        *BOOK_ADMISSION_KEY_COLUMNS,
        "candidate_id",
        "probability_yes",
        SELECTED_SIDE_COLUMN,
        LABEL_COLUMN,
        "yes_ask_vwap_5",
        "no_ask_vwap_5",
        "yes_ask_depth",
        "no_ask_depth",
        "yes_cost_per_share",
        "no_cost_per_share",
        "yes_execution_cost_per_share",
        "no_execution_cost_per_share",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("frozen-orientation ledger missing: " + ", ".join(missing))
    if frame["candidate_id"].unique().to_list() != [arm_id]:
        raise RuntimeError("frozen-orientation ledger candidate identity changed")
    selected_yes = pl.col(SELECTED_SIDE_COLUMN) == "YES"
    staged = frame.with_columns(
        selected_yes.alias("selected_yes"),
        (pl.col("probability_yes") >= 0.5).alias("argmax_yes"),
    ).with_columns(
        pl.when(pl.col("selected_yes"))
        .then(pl.col("probability_yes"))
        .otherwise(1.0 - pl.col("probability_yes"))
        .alias("selected_probability"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_ask_vwap_5"))
        .otherwise(pl.col("no_ask_vwap_5"))
        .alias("selected_share_price"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_ask_depth"))
        .otherwise(pl.col("no_ask_depth"))
        .alias("selected_depth"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_cost_per_share"))
        .otherwise(pl.col("no_cost_per_share"))
        .alias("selected_admission_cost_per_share"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("yes_execution_cost_per_share"))
        .otherwise(pl.col("no_execution_cost_per_share"))
        .alias("selected_execution_cost_per_share"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col(LABEL_COLUMN) == 1)
        .otherwise(pl.col(LABEL_COLUMN) == 0)
        .alias("won"),
    ).with_columns(
        (
            pl.col("selected_probability")
            - pl.col("selected_admission_cost_per_share")
        ).alias("selected_edge_per_share"),
        (pl.col("selected_yes") != pl.col("argmax_yes")).alias("selected_underdog"),
        pl.lit(arm_id).alias("model"),
    )
    eligible = staged.filter(
        pl.col("seconds_elapsed").is_between(
            policy.minimum_entry_second,
            policy.maximum_entry_second,
            closed="both",
        )
        & pl.col("selected_share_price").is_between(
            policy.minimum_share_price,
            policy.maximum_share_price,
            closed="left",
        )
        & (pl.col("selected_admission_cost_per_share") <= policy.maximum_cost_per_share)
        & (pl.col("selected_depth") * policy.maximum_depth_participation >= policy.quantity)
        & (pl.col("selected_edge_per_share") >= policy.minimum_edge_per_share)
    )
    ledger = (
        eligible.sort("model", "market_id", "seconds_elapsed", "observed_at")
        .group_by("model", "market_id", maintain_order=True)
        .first()
        .with_columns(
            pl.lit(policy.name).alias("policy"),
            pl.lit(policy.quantity).alias("quantity"),
            (
                (
                    pl.col("won").cast(pl.Float64)
                    - pl.col("selected_execution_cost_per_share")
                )
                * policy.quantity
            ).alias("realized_net"),
            (
                pl.col("selected_execution_cost_per_share") * policy.quantity
            ).alias("entry_debit"),
        )
    )
    if ledger.filter(pl.col(SELECTED_SIDE_COLUMN) == "YES").height != int(
        ledger["selected_yes"].sum()
    ):
        raise RuntimeError("ledger side orientation changed")
    return ledger


def _projected_model_record(
    ledger: pl.DataFrame,
    *,
    arm_id: str,
    eligible_market_count: int,
    selected_candidate_id: str | None,
) -> dict[str, Any]:
    metrics = ledger_metrics(ledger)
    stressed = _stressed_profit(ledger)
    yes = ledger.filter(pl.col("selected_yes"))
    no = ledger.filter(~pl.col("selected_yes"))
    trades = int(metrics["trades"])
    wins = int(ledger["won"].sum()) if trades else 0
    return {
        "arm_id": arm_id,
        "probability_selected": arm_id == selected_candidate_id,
        "projected_pnl_used_for_probability_selection": False,
        "projected_pnl_opened_after_probability_selection": True,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "forward_proof": False,
        "trades": trades,
        "wins": wins,
        "losses": trades - wins,
        "accuracy": metrics.get("accuracy"),
        "coverage_per_eligible_market": trades / eligible_market_count,
        "covered_market_fraction": (
            int(metrics.get("markets", 0)) / eligible_market_count
        ),
        "eligible_market_denominator": eligible_market_count,
        "net_profit": float(metrics.get("net_profit", 0.0)),
        "net_profit_per_eligible_market": (
            float(metrics.get("net_profit", 0.0)) / eligible_market_count
        ),
        "net_expectancy_per_trade": metrics.get("net_expectancy_per_trade"),
        "profit_factor": metrics.get("profit_factor"),
        "stress_1c_profit_factor": metrics.get("stress_1c_profit_factor"),
        "maximum_drawdown": metrics.get("maximum_drawdown"),
        "stress_1c_net_profit": stressed,
        "stress_1c_net_profit_per_eligible_market": stressed / eligible_market_count,
        "stress_1c_net_expectancy_per_trade": (
            stressed / trades if trades else None
        ),
        "yes_trades": yes.height,
        "yes_net_profit": float(yes["realized_net"].sum()) if yes.height else 0.0,
        "yes_stress_1c_net_profit": _stressed_profit(yes),
        "no_trades": no.height,
        "no_net_profit": float(no["realized_net"].sum()) if no.height else 0.0,
        "no_stress_1c_net_profit": _stressed_profit(no),
        "mean_share_price": metrics.get("mean_share_price"),
        "mean_entry_second": metrics.get("mean_entry_second"),
        "loss_recovery_wins": metrics.get("loss_recovery_wins"),
    }


def _stressed_profit(ledger: pl.DataFrame) -> float:
    if ledger.is_empty():
        return 0.0
    stressed_cost = np.minimum(
        ledger["selected_execution_cost_per_share"].to_numpy()
        + EXECUTION_STRESS_PER_SHARE,
        1.0,
    )
    return float(
        np.sum(
            ledger["quantity"].to_numpy()
            * (ledger["won"].to_numpy().astype(np.float64) - stressed_cost)
        )
    )


def _write_projected_pnl_csv(
    path: Path,
    models: Mapping[str, Mapping[str, Any]],
) -> None:
    fields = (
        "arm_id",
        "trades",
        "wins",
        "losses",
        "accuracy",
        "coverage_per_eligible_market",
        "net_profit",
        "net_profit_per_eligible_market",
        "net_expectancy_per_trade",
        "profit_factor",
        "stress_1c_profit_factor",
        "maximum_drawdown",
        "stress_1c_net_profit",
        "stress_1c_net_profit_per_eligible_market",
        "stress_1c_net_expectancy_per_trade",
        "yes_trades",
        "yes_stress_1c_net_profit",
        "no_trades",
        "no_stress_1c_net_profit",
        "mean_share_price",
        "diagnostic_only",
        "qualification_eligible",
        "probability_selected",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for arm_id in ARM_IDS:
            writer.writerow(models[arm_id])


def _write_projected_pnl_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# D4 side-calibration projected PnL",
        "",
        (
            "Consumed development diagnostic only. These projections were opened "
            "after probability selection and cannot qualify an arm, export a model, "
            "or change the source process."
        ),
        "",
        (
            "| Arm | Trades | Coverage | Wins | Accuracy | PnL | +1c PnL | "
            "+1c EV/trade | PF | +1c PF | Mean price | Max drawdown | YES +1c | NO +1c |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_id in ARM_IDS:
        record = payload["models"][arm_id]
        lines.append(
            "| {arm} | {trades} | {coverage:.4%} | {wins} | {accuracy} | "
            "{pnl:.4f} | {stress:.4f} | {ev} | {pf} | {stress_pf} | {price} | "
            "{dd} | {yes:.4f} | "
            "{no:.4f} |".format(
                arm=arm_id,
                trades=record["trades"],
                coverage=record["coverage_per_eligible_market"],
                wins=record["wins"],
                accuracy=_format_optional(record["accuracy"]),
                pnl=record["net_profit"],
                stress=record["stress_1c_net_profit"],
                ev=_format_optional(record["stress_1c_net_expectancy_per_trade"]),
                pf=_format_optional(record["profit_factor"]),
                stress_pf=_format_optional(record["stress_1c_profit_factor"]),
                price=_format_optional(record["mean_share_price"]),
                dd=_format_optional(record["maximum_drawdown"]),
                yes=record["yes_stress_1c_net_profit"],
                no=record["no_stress_1c_net_profit"],
            )
        )
    lines.extend(
        [
            "",
            f"Eligible-market denominator: {payload['eligible_market_denominator']}",
            "",
            "Projected PnL was not used for probability selection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _write_root_contract(
    run_dir: Path,
    *,
    config: Any,
    parent_integrity: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    strict_validation: pl.DataFrame,
    eligible_markets: pl.DataFrame,
) -> None:
    observed_days = sorted(
        value.isoformat()
        for value in strict_validation["window_start"].dt.date().unique().to_list()
    )
    payload = {
        "schema_version": D4_SIDE_BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": config.profile,
        "process_id": config.incumbent.process_id,
        "incumbent_model_key": config.incumbent.model_key,
        "parent_d4_run_id": config.parent_run.run_id,
        "parent_d4_refitted": False,
        "parent_integrity_sha256": file_sha256(run_dir / "parent-integrity.json"),
        "parent_artifact_count": parent_integrity["validated_artifact_count"],
        "parent_fold_model_count": parent_integrity["validated_fold_model_count"],
        "arms": list(ARM_IDS),
        "selected_side_source": "D4-base probability and frozen 20--30c policy",
        "selected_side_reoriented_after_calibration": False,
        "policy": asdict(config.policy),
        "oof_utc_days": observed_days,
        "excluded_utc_days": [value.isoformat() for value in config.excluded_utc_days],
        "strict_oof_markets": eligible_markets.height,
        "expected_strict_oof_markets": EXPECTED_STRICT_OOF_MARKETS,
        "source_proxy_prices_used": source_evidence.get("proxy_prices_used"),
        "source_evidence_sha256": _canonical_sha256(_json_safe(source_evidence)),
        "evidence_scope": config.evidence_scope,
        "consumed_validation_role": config.consumed_validation_role,
        "paper_only": True,
        "qualification_eligible": False,
        "support_qualification_allowed": False,
        "forward_proof": False,
        "runtime_deployable": False,
        "batch_forward_eligible": False,
        "process_change_allowed": False,
        "pnl_based_selection_allowed": False,
        "projected_pnl_reported_for_every_arm": True,
        "dependency_identity": _dependency_identity(config),
    }
    write_json_atomic(run_dir / "root-training-contract.json", payload)


def _strict_validation_union(frame: pl.DataFrame, config: Any) -> pl.DataFrame:
    expected = [value.isoformat() for value in config.oof_utc_days]
    excluded = {value.isoformat() for value in config.excluded_utc_days}
    validation = frame.filter(
        pl.col("window_start").dt.date().cast(pl.String).is_in(expected)
    ).sort(*BOOK_ADMISSION_KEY_COLUMNS)
    observed = sorted(
        value.isoformat()
        for value in validation["window_start"].dt.date().unique().to_list()
    )
    if observed != expected:
        raise RuntimeError("strict D4 OOF UTC-day union changed")
    if excluded.intersection(observed):
        raise RuntimeError("excluded July 26/27 evidence entered D4 OOF")
    if validation.select(*BOOK_ADMISSION_KEY_COLUMNS).is_duplicated().any():
        raise RuntimeError("strict D4 OOF universe contains duplicate exact keys")
    return validation


def _validate_fold(fold: Any, config: Any) -> None:
    if not (
        config.development.start
        <= fold.fit.start
        < fold.fit.end
        <= fold.calibration.start
        < fold.calibration.end
        <= fold.validation.start
        < fold.validation.end
        <= config.development.end
    ):
        raise RuntimeError(f"D4 side fold is not causal: {fold.name}")
    if (fold.validation.end - fold.validation.start).total_seconds() != 86_400:
        raise RuntimeError(f"D4 side validation is not exactly one day: {fold.name}")


def _validate_incumbent_identity(config: Any, incumbent: Any) -> None:
    payload = incumbent.payload
    if (
        incumbent.model_sha256 != config.incumbent.model_sha256
        or payload.get("model_key") != config.incumbent.model_key
        or payload.get("features", {}).get("schema_sha256")
        != config.incumbent.feature_schema_sha256
        or len(payload.get("features", {}).get("names", []))
        != config.incumbent.feature_count
    ):
        raise RuntimeError("D4 side incumbent identity changed")


def _selected_candidate_id(selection: Mapping[str, Any]) -> str | None:
    for key in (
        "selected_candidate_id",
        "selected_challenger_id",
        "selected_challenger",
    ):
        value = selection.get(key)
        if value is not None:
            if value not in CALIBRATION_IDS:
                raise RuntimeError("probability selector returned an undeclared challenger")
            return str(value)
    return None


def _probability_support_evidence(
    fit_records: list[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Reduce causal fold support to a fail-closed per-challenger contract."""

    result: dict[str, dict[str, Any]] = {}
    for arm_id in CALIBRATION_IDS:
        records = [record for record in fit_records if record["arm_id"] == arm_id]
        if len(records) != 10:
            raise RuntimeError(f"{arm_id} must contain ten causal calibration fits")
        supports = [record["fit_evidence"]["support"] for record in records]
        failures = [
            f"{record['fold']}/{failure}"
            for record in records
            for failure in record["fit_evidence"]["support_failures"]
        ]
        cell_names = {
            str(cell["name"])
            for support in supports
            for cell in support["time_cells"]
        }
        cells = []
        for name in sorted(cell_names):
            matching = [
                cell
                for support in supports
                for cell in support["time_cells"]
                if cell["name"] == name
            ]
            cells.append(
                {
                    "name": name,
                    "markets": min(int(cell["markets"]) for cell in matching),
                    "utc_days": min(int(cell["utc_days"]) for cell in matching),
                    "winning_markets": min(
                        int(cell["winning_markets"]) for cell in matching
                    ),
                    "losing_markets": min(
                        int(cell["losing_markets"]) for cell in matching
                    ),
                }
            )
        result[arm_id] = {
            "support_passed": not failures
            and all(record["fit_evidence"]["support_passed"] for record in records),
            "support_failures": failures,
            "no_markets": min(int(support["no_markets"]) for support in supports),
            "no_utc_days": min(int(support["no_utc_days"]) for support in supports),
            "no_winning_markets": min(
                int(support["no_winning_markets"]) for support in supports
            ),
            "no_losing_markets": min(
                int(support["no_losing_markets"]) for support in supports
            ),
            "time_cells": cells,
            "causal_fold_count": len(records),
            "qualification_eligible": False,
        }
    return result


def _finalize_run(
    run_dir: Path,
    *,
    result: Mapping[str, Any],
    selection: Mapping[str, Any],
    projected: Mapping[str, Any],
) -> None:
    write_json_atomic(run_dir / "benchmark-result.json", dict(result))
    support = _read_json(run_dir / "calibration-support-manifest.json")
    probability_failures = {
        str(record.get("candidate_id")): list(record.get("failed_gates", []))
        for record in selection.get("failure_trace", [])
    }
    hard_support = {
        arm_id: list(support.get("arms", {}).get(arm_id, {}).get("support_failures", []))
        for arm_id in CALIBRATION_IDS
    }
    report = [
        "# Frozen D4 side-calibration benchmark",
        "",
        "Outcome: consumed-development diagnostic complete.",
        "",
        f"- Probability-selected challenger: `{_selected_candidate_id(selection) or 'none'}`",
        f"- Eligible strict OOF markets: `{projected['eligible_market_denominator']}`",
        "- Parent D4 refitted: `false`",
        "- D4-base side reoriented after calibration: `false`",
        "- Projected PnL used for selection: `false`",
        "- Qualification eligible: `false`",
        "- Forward proof: `false`",
        "- Runtime or source-process change: `none`",
        "",
        "## Probability result and support",
        "",
        f"- Probability status: `{selection.get('status')}`",
        (
            "- Probability failed gates: `"
            f"{json.dumps(probability_failures, sort_keys=True)}`"
        ),
        (
            "- Hard 14-day support failures: `"
            f"{json.dumps(hard_support, sort_keys=True)}`"
        ),
        "- Any listed support failure is hard and cannot be waived by projected PnL.",
        "",
        "## Complete projected PnL by model",
        "",
        *(
            run_dir / "projected-pnl.md"
        ).read_text().splitlines()[4:],
    ]
    (run_dir / "benchmark-report.md").write_text("\n".join(report) + "\n")
    manifest = _artifact_manifest(run_dir)
    write_json_atomic(run_dir / "artifact-manifest.json", manifest)
    bundle = {
        "schema_version": D4_SIDE_BUNDLE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_manifest_sha256": file_sha256(run_dir / "artifact-manifest.json"),
        "benchmark_result_sha256": file_sha256(run_dir / "benchmark-result.json"),
        "benchmark_report_sha256": file_sha256(run_dir / "benchmark-report.md"),
        "probability_selection_seal_sha256": file_sha256(
            run_dir / "probability-selection-seal.json"
        ),
        "projected_pnl_sha256": file_sha256(run_dir / "projected-pnl.json"),
        "projected_pnl_csv_sha256": file_sha256(run_dir / "projected-pnl.csv"),
        "projected_pnl_markdown_sha256": file_sha256(run_dir / "projected-pnl.md"),
        "qualification_eligible": False,
        "runtime_deployable": False,
        "source_process_changed": False,
    }
    bundle["aggregate_sha256"] = _canonical_sha256(bundle)
    write_json_atomic(run_dir / "benchmark-bundle-seal.json", bundle)


def _artifact_manifest(run_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, str] = {}
    for path in sorted(value for value in run_dir.rglob("*") if value.is_file()):
        if path.name in {"artifact-manifest.json", "benchmark-bundle-seal.json"}:
            continue
        artifacts[str(path.relative_to(run_dir))] = file_sha256(path)
    return {
        "schema_version": D4_SIDE_BENCHMARK_SCHEMA_VERSION,
        "artifacts": artifacts,
        "aggregate_sha256": _canonical_sha256(artifacts),
    }


def _window(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    return frame.filter((pl.col("window_start") >= start) & (pl.col("window_start") < end))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return payload


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(nested) for nested in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dependency_identity(config: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": {
            distribution: version(distribution)
            for distribution in ("numpy", "polars", "scipy", "scikit-learn", "joblib")
        },
        "pyproject_sha256": file_sha256(config.package_root / "pyproject.toml"),
        "requirements_lock_sha256": file_sha256(
            config.package_root / "requirements.lock"
        ),
    }


def _format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    return f"{float(value):.6f}"


def _calibration_api() -> dict[str, Any]:
    # Imported lazily so this one-commit branch remains testable before the
    # sibling calibration branch is merged into the shared base.
    from .asymmetric_d4_side_calibration import (
        fit_d4_side_calibrator,
        score_d4_side_calibrator,
        validate_d4_side_calibration_config,
    )

    return {
        "fit": fit_d4_side_calibrator,
        "score": score_d4_side_calibrator,
        "validate_config": validate_d4_side_calibration_config,
    }


def _evaluation_api() -> dict[str, Any]:
    # Imported lazily for the same isolated-branch reason as the calibrator API.
    from .asymmetric_d4_calibration_evaluation import (
        D4ProjectedPnlThresholds,
        D4SideCalibrationSelectionThresholds,
        build_d4_projected_pnl_report,
        select_d4_side_calibration_challenger,
    )

    return {
        "select": select_d4_side_calibration_challenger,
        "projected_pnl": build_d4_projected_pnl_report,
        "selection_thresholds": D4SideCalibrationSelectionThresholds,
        "projected_pnl_thresholds": D4ProjectedPnlThresholds,
    }


def load_and_run_d4_side_calibration_benchmark(
    config_path: Path,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    from .asymmetric_d4_side_calibration import load_d4_side_calibration_config

    return run_d4_side_calibration_benchmark(
        load_d4_side_calibration_config(config_path),
        force=force,
    )


__all__ = [
    "load_and_run_d4_side_calibration_benchmark",
    "run_d4_side_calibration_benchmark",
    "validate_parent_run_integrity",
]
