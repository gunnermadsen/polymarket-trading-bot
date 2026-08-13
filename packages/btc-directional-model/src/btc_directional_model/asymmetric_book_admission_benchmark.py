"""Seal-first offline benchmark for causal Polymarket-book admission models."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import sys
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl

from .asymmetric_book_admission import (
    BOOK_ADMISSION_KEY_COLUMNS,
    BOOK_ADMISSION_SCHEMA_VERSION,
    DYNAMIC_FEATURES,
    BookAdmissionConfig,
    attach_frozen_policy_selected_side,
    book_admission_support,
    fit_book_admission_candidate,
    load_book_admission_config,
    load_book_admission_model,
    score_book_admission_model,
    select_target_opportunity_rows,
)
from .asymmetric_book_admission_evaluation import (
    BookAdmissionSelectionThresholds,
    select_asymmetric_book_admission_challenger,
)
from .asymmetric_book_dynamics import (
    BOOK_DYNAMICS_HORIZONS_SECONDS,
    attach_causal_book_dynamics,
    book_dynamics_evidence,
    book_dynamics_schema_sha256,
)
from .asymmetric_incumbent_benchmark import _load_core_oracle_value_frame
from .asymmetric_incumbent_calibration import (
    frozen_parent_probabilities,
    score_incumbent_calibration_payload,
)
from .asymmetric_incumbent_evaluation import (
    build_incumbent_correction_ledger,
    paired_incumbent_economics,
)
from .asymmetric_incumbent_replay import (
    FROZEN_ASYMMETRIC_INCUMBENT_MAXIMUM_DEPTH_PARTICIPATION,
    FROZEN_ASYMMETRIC_INCUMBENT_POLICY,
    FROZEN_ASYMMETRIC_INCUMBENT_QUANTITY,
    load_frozen_asymmetric_incumbent,
)
from .asymmetric_training_readiness import prepare_asymmetric_training_readiness
from .asymmetric_value_config import load_asymmetric_value_config
from .asymmetric_value_evaluation import ledger_metrics, policy_ledger, score_two_sided_value
from .asymmetric_value_training import asymmetric_probability_frame
from .core_config import load_core_config
from .core_extract import file_sha256, write_json_atomic

BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION = "btc-asymmetric-book-admission-benchmark-v1"
BOOK_ADMISSION_ROOT_CONTRACT_SCHEMA_VERSION = "btc-asymmetric-book-admission-root-contract-v1"
BOOK_ADMISSION_OUTCOME_SEAL_SCHEMA_VERSION = "btc-asymmetric-book-admission-outcome-access-v1"
BOOK_ADMISSION_SELECTION_SEAL_SCHEMA_VERSION = "btc-asymmetric-book-admission-selection-seal-v1"
BOOK_ADMISSION_SELECTED_MODEL_SEAL_SCHEMA_VERSION = (
    "btc-asymmetric-book-admission-selected-model-seal-v1"
)
BOOK_ADMISSION_ECONOMICS_SCHEMA_VERSION = "btc-asymmetric-book-admission-economics-v1"
BOOK_ADMISSION_FORWARD_SCHEMA_VERSION = "btc-asymmetric-book-admission-batch-forward-v1"
INCUMBENT_ID = "I0"
STATIC_CONTROL_ID = "S0"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260809
FORWARD_REQUIREMENTS = {
    "minimum_complete_utc_days": 21,
    "minimum_strict_markets": 2_000,
    "minimum_candidate_trades": 200,
    "minimum_yes_entries": 20,
    "minimum_no_entries": 20,
    "minimum_probability_noninferior_days": 17,
    "pnl_based_early_stopping_allowed": False,
    "require_positive_lower_95_stressed_expectancy": True,
    "require_positive_stressed_capital_efficiency": True,
    "require_positive_net_corrected_decisions": True,
    "require_no_worse_loss_recovery_or_drawdown": True,
}
SEALED_PREDICTION_COLUMNS = {
    *BOOK_ADMISSION_KEY_COLUMNS,
    "candidate_id",
    "probability_yes",
}
PROBABILITY_SELECTION_FRAME_COLUMNS = {
    *SEALED_PREDICTION_COLUMNS,
    "label_up",
    "yes_ask_vwap_5",
    "no_ask_vwap_5",
    "yes_ask_depth",
    "no_ask_depth",
    "yes_cost_per_share",
    "no_cost_per_share",
}
IMPLEMENTATION_FILES = (
    "asymmetric_book_admission_benchmark.py",
    "asymmetric_book_admission.py",
    "asymmetric_book_admission_evaluation.py",
    "asymmetric_book_dynamics.py",
    "asymmetric_incumbent_benchmark.py",
    "asymmetric_incumbent_calibration.py",
    "asymmetric_incumbent_evaluation.py",
    "asymmetric_incumbent_replay.py",
    "asymmetric_training_readiness.py",
    "asymmetric_value_benchmark.py",
    "asymmetric_value_config.py",
    "asymmetric_value_data.py",
    "asymmetric_value_evaluation.py",
    "asymmetric_value_training.py",
    "core_config.py",
    "core_extract.py",
    "core_features.py",
    "cli.py",
)


def run_book_admission_benchmark(
    config: BookAdmissionConfig,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Run the frozen S0/D1--D4 matrix and open economics for one winner only."""

    source_config = load_asymmetric_value_config(config.paths.source_asymmetric_value_config)
    core_config = load_core_config(source_config.core_config)
    readiness_path, readiness = prepare_asymmetric_training_readiness(
        source_config,
        output_dir=config.paths.runs.parent / "btc-asymmetric-book-admission-readiness",
    )
    if readiness.get("ready") is not True:
        raise RuntimeError("book-admission source readiness did not pass")
    source, source_evidence = _load_core_oracle_value_frame(
        source_config,
        core_config=core_config,
        force=force,
    )
    if source_evidence.get("proxy_prices_used") is not False:
        raise RuntimeError("book admission requires exact PMXT execution evidence")
    source = _window(source, config.development.start, config.development.end)
    incumbent = load_frozen_asymmetric_incumbent(config.paths.incumbent_model)
    _validate_incumbent(config, incumbent)
    parent_probability = frozen_parent_probabilities(incumbent, source)
    incumbent_probability = score_incumbent_calibration_payload(
        incumbent,
        incumbent.payload,
        source,
        parent_probabilities=parent_probability,
    )
    oriented = attach_frozen_policy_selected_side(
        source.with_columns(
            pl.Series("incumbent_probability_yes", incumbent_probability, dtype=pl.Float64)
        ),
        config,
    )
    dynamic = attach_causal_book_dynamics(
        oriented,
        maximum_book_age_seconds=config.readiness_gates.maximum_book_age_seconds,
    )
    dynamics_evidence = book_dynamics_evidence(dynamic)
    validation = _validation_union(dynamic, config)
    oof_validation_without_outcomes = validation.drop("label_up")
    target_validation = _outcome_blind_target(
        oof_validation_without_outcomes,
        config,
    )
    coverage = _validate_readiness(config, dynamic, target_validation)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = config.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir()
    _write_pre_fit_contracts(
        run_dir,
        config=config,
        readiness_path=readiness_path,
        readiness=readiness,
        source_evidence=source_evidence,
        dynamics_evidence=dynamics_evidence,
        coverage=coverage,
        target_validation=target_validation,
    )

    candidate_predictions: dict[str, list[pl.DataFrame]] = {
        candidate.name: [] for candidate in config.model.candidates
    }
    model_manifest: dict[str, Any] = {}
    support_trace: dict[str, list[dict[str, Any]]] = {
        candidate.name: [] for candidate in config.model.candidates
    }
    incumbent_predictions: list[pl.DataFrame] = []
    for fold in config.folds:
        _validate_fold_chronology(fold, config)
        fit_frame = _window(dynamic, fold.fit.start, fold.fit.end)
        calibration_frame = _window(dynamic, fold.calibration.start, fold.calibration.end)
        validation_frame = _window(dynamic, fold.validation.start, fold.validation.end)
        fold_validation_without_outcomes = validation_frame.drop("label_up")
        target_fold = _outcome_blind_target(
            fold_validation_without_outcomes,
            config,
        )
        incumbent_predictions.append(
            target_fold.select(
                *BOOK_ADMISSION_KEY_COLUMNS,
                pl.col("incumbent_probability_yes").alias("probability_yes"),
            ).with_columns(
                pl.lit(INCUMBENT_ID).alias("candidate_id"),
            )
        )
        for candidate in config.model.candidates:
            fitted = fit_book_admission_candidate(
                config,
                candidate.name,
                fit_frame,
                calibration_frame,
            )
            candidate_path = candidates_dir / candidate.name / fold.name
            artifact_path = candidate_path / "model.pkl"
            serialization = fitted.serialize(artifact_path)
            before = score_book_admission_model(
                fitted,
                fold_validation_without_outcomes,
                config,
            )
            reloaded = _load_verified_model(artifact_path, serialization)
            _validate_serialized_model(
                reloaded,
                config=config,
                candidate_name=candidate.name,
                artifact_path=artifact_path,
                serialization=serialization,
            )
            after = score_book_admission_model(
                reloaded,
                fold_validation_without_outcomes,
                config,
            )
            if (
                before.probability_sha256 != after.probability_sha256
                or before.model_semantic_sha256 != after.model_semantic_sha256
            ):
                raise RuntimeError(f"{candidate.name} fold serialization changed predictions")
            candidate_predictions[candidate.name].append(
                after.frame.rename({"incumbent_probability_yes": "probability_yes"})
                .rename({"candidate_name": "candidate_id"})
            )
            fit_support = book_admission_support(
                fit_frame, config, candidate.name, require_label=True
            )
            calibration_support = book_admission_support(
                calibration_frame, config, candidate.name, require_label=True
            )
            support_trace[candidate.name].append(
                {
                    "fold": fold.name,
                    "fit": asdict(fit_support),
                    "calibration": asdict(calibration_support),
                    "fit_evidence": asdict(fitted.evidence),
                }
            )
            model_manifest[f"{candidate.name}/{fold.name}"] = {
                "path": str(artifact_path.relative_to(run_dir)),
                "artifact_sha256": file_sha256(artifact_path),
                "semantic_sha256": fitted.semantic_sha256,
                "serialization": serialization,
            }

    prediction_paths, prediction_hashes = _write_prediction_artifacts(
        run_dir,
        incumbent_predictions=incumbent_predictions,
        candidate_predictions=candidate_predictions,
        expected_keys=target_validation.select(*BOOK_ADMISSION_KEY_COLUMNS),
    )
    model_manifest_path = run_dir / "fold-model-manifest.json"
    write_json_atomic(
        model_manifest_path,
        {
            "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
            "lineage_scope": "development_oof_probability_selection_only",
            "usable_as_final_batch_model": False,
            "models": model_manifest,
        },
    )
    support_evidence = _selection_support(config, coverage, support_trace)
    support_path = run_dir / "candidate-support-manifest.json"
    write_json_atomic(support_path, support_evidence)
    prediction_manifest_path = run_dir / "prediction-manifest.json"
    write_json_atomic(
        prediction_manifest_path,
        {
            "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
            "prediction_sha256": prediction_hashes,
            "prediction_paths": {
                name: str(path.relative_to(run_dir)) for name, path in prediction_paths.items()
            },
        },
    )
    pre_outcome_manifest_path = _write_pre_outcome_artifact_manifest(
        run_dir,
        config=config,
        readiness_path=readiness_path,
        model_manifest_path=model_manifest_path,
        prediction_manifest_path=prediction_manifest_path,
        support_path=support_path,
    )
    outcome_seal_path = _write_outcome_access_seal(
        run_dir,
        pre_outcome_manifest_path=pre_outcome_manifest_path,
        model_manifest_path=model_manifest_path,
        prediction_manifest_path=prediction_manifest_path,
        support_path=support_path,
        prediction_hashes=prediction_hashes,
    )
    target_validation_with_outcomes = target_validation.join(
        validation.select(*BOOK_ADMISSION_KEY_COLUMNS, "label_up"),
        on=list(BOOK_ADMISSION_KEY_COLUMNS),
        how="inner",
        validate="1:1",
    )
    if target_validation_with_outcomes.height != target_validation.height:
        raise RuntimeError("post-seal OOF label join changed the target validation grid")
    pooled_oof_cells, pooled_oof_failures = _target_cell_support(
        target_validation_with_outcomes,
        require_two_outcomes=True,
    )
    pooled_oof_support_path = run_dir / "pooled-oof-cell-support.json"
    write_json_atomic(
        pooled_oof_support_path,
        {
            "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
            "accessed_after_outcome_seal_sha256": file_sha256(outcome_seal_path),
            "side_time_cells": pooled_oof_cells,
            "failures": pooled_oof_failures,
            "passed": not pooled_oof_failures,
        },
    )
    if pooled_oof_failures:
        selection = {
            "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
            "status": "insufficient_probability_support",
            "selected_candidate_id": None,
            "incumbent": {},
            "candidate_records": [],
            "failure_trace": [
                {
                    "candidate_id": None,
                    "failed_gates": pooled_oof_failures,
                }
            ],
            "economics_used": False,
        }
        selection_path = run_dir / "probability-selection.json"
        write_json_atomic(selection_path, selection)
        _write_selection_seal(
            run_dir,
            selection_path=selection_path,
            outcome_seal_path=outcome_seal_path,
            model_manifest_path=model_manifest_path,
            prediction_manifest_path=prediction_manifest_path,
            pooled_oof_support_path=pooled_oof_support_path,
            selected_candidate_id=None,
        )
        result = {
            "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
            "status": "incumbent_retained_probability_support",
            "selected_candidate_id": None,
            "economics_opened": False,
            "batch_forward_artifact": None,
            "runtime_deployable": False,
            "source_process_changed": False,
        }
        _finalize(run_dir, result, selection=selection, economics=None)
        return run_dir, result

    probability_context = target_validation_with_outcomes.select(
        *BOOK_ADMISSION_KEY_COLUMNS,
        "label_up",
        "yes_ask_vwap_5",
        "yes_ask_depth",
        "no_ask_vwap_5",
        "no_ask_depth",
        "yes_cost_per_share",
        "no_cost_per_share",
    )
    probability_frames = {
        candidate_id: _load_sealed_probability_frame(
            prediction_paths[candidate_id],
            prediction_hashes[candidate_id],
            probability_context,
        )
        for candidate_id in prediction_paths
    }
    thresholds = BookAdmissionSelectionThresholds(
        noninferiority_margin=config.probability_gates.maximum_paired_degradation_upper_95,
        maximum_selected_bias=config.probability_gates.maximum_selected_opportunity_bias,
        maximum_cell_bias=config.probability_gates.maximum_cell_bias,
        minimum_joint_noninferior_days=config.probability_gates.minimum_noninferior_days,
        required_validation_days=config.probability_gates.required_comparison_days,
        minimum_feature_coverage=config.readiness_gates.minimum_dynamic_coverage,
    )
    dynamic_frames = {
        candidate.name: probability_frames[candidate.name]
        for candidate in config.model.candidates
        if candidate.selection_eligible
    }
    selection = select_asymmetric_book_admission_challenger(
        probability_frames[INCUMBENT_ID],
        probability_frames[STATIC_CONTROL_ID],
        dynamic_frames,
        support_evidence,
        thresholds,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        static_candidate_id=STATIC_CONTROL_ID,
    )
    selection_path = run_dir / "probability-selection.json"
    write_json_atomic(selection_path, selection)
    selection_seal_path = _write_selection_seal(
        run_dir,
        selection_path=selection_path,
        outcome_seal_path=outcome_seal_path,
        model_manifest_path=model_manifest_path,
        prediction_manifest_path=prediction_manifest_path,
        pooled_oof_support_path=pooled_oof_support_path,
        selected_candidate_id=selection.get("selected_candidate_id"),
    )
    selected = selection.get("selected_candidate_id")
    if selected is None:
        result = {
            "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
            "status": "incumbent_retained_probability_gates",
            "selected_candidate_id": None,
            "economics_opened": False,
            "batch_forward_artifact": None,
            "runtime_deployable": False,
            "source_process_changed": False,
        }
        _finalize(run_dir, result, selection=selection, economics=None)
        return run_dir, result

    refit_support_path, refit_support_passed = _write_selected_refit_support(
        run_dir,
        config=config,
        dynamic=dynamic,
        selection_seal_path=selection_seal_path,
    )
    if not refit_support_passed:
        result = {
            "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
            "status": "incumbent_retained_selected_refit_support",
            "selected_candidate_id": selected,
            "economics_opened": False,
            "batch_forward_artifact": None,
            "runtime_deployable": False,
            "source_process_changed": False,
        }
        _finalize(run_dir, result, selection=selection, economics=None)
        return run_dir, result
    final_model, final_model_path, selected_model_seal_path = _refit_selected_model(
        run_dir,
        config=config,
        selected=selected,
        dynamic=dynamic,
        selection_seal_path=selection_seal_path,
        refit_support_path=refit_support_path,
    )
    del final_model
    economics_open_path = run_dir / "economics-access.json"
    write_json_atomic(
        economics_open_path,
        {
            "schema_version": BOOK_ADMISSION_ECONOMICS_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "selected_candidate_id": selected,
            "selection_seal_sha256": file_sha256(selection_seal_path),
            "selected_model_seal_sha256": file_sha256(selected_model_seal_path),
            "economics_opened": True,
            "evaluated_candidates": [INCUMBENT_ID, selected],
        },
    )
    economic_context = target_validation_with_outcomes.select(
        *BOOK_ADMISSION_KEY_COLUMNS,
        "label_up",
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
    economic_frames = {
        candidate_id: _load_sealed_probability_frame(
            prediction_paths[candidate_id],
            prediction_hashes[candidate_id],
            economic_context,
            probability_only=False,
        )
        for candidate_id in (INCUMBENT_ID, selected)
    }
    economics = _evaluate_selected_economics(
        config,
        run_dir=run_dir,
        incumbent=economic_frames[INCUMBENT_ID],
        challenger=economic_frames[selected],
        validation=oof_validation_without_outcomes,
    )
    economics["selection_seal_sha256"] = file_sha256(selection_seal_path)
    economics["selected_model_seal_sha256"] = file_sha256(selected_model_seal_path)
    economics["selected_model_artifact_sha256"] = file_sha256(final_model_path)
    economics["economics_access_sha256"] = file_sha256(economics_open_path)
    economics["economic_scope"] = "development_oof"
    economics["development_prediction_sha256"] = {
        INCUMBENT_ID: prediction_hashes[INCUMBENT_ID],
        selected: prediction_hashes[selected],
    }
    economics["development_prediction_manifest_sha256"] = file_sha256(
        prediction_manifest_path
    )
    economics["final_batch_refit_economically_scored"] = False
    economics_path = run_dir / "selected-economics.json"
    write_json_atomic(economics_path, economics)
    forward_artifact = None
    if economics["status"] == "qualified":
        forward_artifact = _authorize_batch_forward(
            run_dir,
            config=config,
            selected=selected,
            final_model_path=final_model_path,
            selection_seal_path=selection_seal_path,
            selected_model_seal_path=selected_model_seal_path,
            economics_access_path=economics_open_path,
            economics_path=economics_path,
        )
    result = {
        "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
        "status": (
            "qualified_batch_forward_blocked_runtime_feature_parity"
            if forward_artifact is not None
            else "incumbent_retained_economic_gates"
        ),
        "selected_candidate_id": selected,
        "economics_opened": True,
        "economic_qualified": economics["status"] == "qualified",
        "batch_forward_artifact": str(forward_artifact) if forward_artifact else None,
        "runtime_deployable": False,
        "blocked_runtime_feature_parity": True,
        "source_process_changed": False,
        "economic_scope": "development_oof",
        "development_prediction_sha256": {
            INCUMBENT_ID: prediction_hashes[INCUMBENT_ID],
            selected: prediction_hashes[selected],
        },
        "final_batch_refit_sha256": file_sha256(final_model_path),
        "final_batch_refit_economically_scored": False,
    }
    _finalize(run_dir, result, selection=selection, economics=economics)
    return run_dir, result


def _validate_incumbent(config: BookAdmissionConfig, incumbent: Any) -> None:
    payload = incumbent.payload
    if (
        incumbent.model_sha256 != config.incumbent.model_sha256
        or payload.get("model_key") != config.incumbent.model_key
        or tuple(payload["features"]["names"]).__len__() != config.incumbent.feature_count
        or payload["features"].get("schema_sha256")
        != config.incumbent.feature_schema_sha256
    ):
        raise RuntimeError("book-admission frozen incumbent identity changed")


def _window(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    return frame.filter((pl.col("window_start") >= start) & (pl.col("window_start") < end))


def _outcome_blind_target(
    frame: pl.DataFrame,
    config: BookAdmissionConfig,
) -> pl.DataFrame:
    if "label_up" in frame.columns:
        raise RuntimeError("OOF validation labels cannot enter a pre-seal target projection")
    return select_target_opportunity_rows(
        frame,
        config,
        require_label=False,
        feature_names=DYNAMIC_FEATURES,
    )


def _validate_fold_chronology(fold: Any, config: BookAdmissionConfig) -> None:
    if not (
        config.development.start <= fold.fit.start < fold.fit.end
        <= fold.calibration.start < fold.calibration.end
        <= fold.validation.start < fold.validation.end <= config.development.end
    ):
        raise RuntimeError(f"book-admission fold is not strictly causal: {fold.name}")
    if (fold.validation.end - fold.validation.start).total_seconds() != 86_400:
        raise RuntimeError(f"book-admission fold is not exactly one UTC day: {fold.name}")


def _validate_serialized_model(
    model: Any,
    *,
    config: BookAdmissionConfig,
    candidate_name: str,
    artifact_path: Path,
    serialization: Mapping[str, Any],
) -> None:
    if file_sha256(artifact_path) != serialization.get("artifact_sha256"):
        raise RuntimeError(f"{candidate_name} serialized model bytes changed before reload")
    if (
        model.schema_version != BOOK_ADMISSION_SCHEMA_VERSION
        or model.candidate != config.candidate(candidate_name)
        or model.incumbent != config.incumbent
        or model.dynamics_schema_sha256 != book_dynamics_schema_sha256()
    ):
        raise RuntimeError(f"{candidate_name} serialized model contract changed")


def _load_verified_model(
    artifact_path: Path,
    serialization: Mapping[str, Any],
) -> Any:
    if file_sha256(artifact_path) != serialization.get("artifact_sha256"):
        raise RuntimeError("serialized book-admission model bytes changed before deserialization")
    return load_book_admission_model(artifact_path)


def _validation_union(frame: pl.DataFrame, config: BookAdmissionConfig) -> pl.DataFrame:
    days = [value.isoformat() for value in config.oof_utc_days]
    validation = frame.filter(pl.col("window_start").dt.date().cast(pl.String).is_in(days))
    observed = sorted(
        value.isoformat() for value in validation["window_start"].dt.date().unique().to_list()
    )
    if observed != days:
        raise RuntimeError("book-admission validation does not contain the exact ten UTC days")
    if validation.select(*BOOK_ADMISSION_KEY_COLUMNS).is_duplicated().any():
        raise RuntimeError("book-admission validation contains duplicate exact keys")
    return validation.sort(*BOOK_ADMISSION_KEY_COLUMNS)


def _validate_readiness(
    config: BookAdmissionConfig,
    dynamic: pl.DataFrame,
    target_validation: pl.DataFrame,
) -> dict[str, Any]:
    gates = config.readiness_gates
    strict_validation = _validation_union(dynamic, config)
    strict_markets = strict_validation["market_id"].n_unique()
    target_markets = target_validation["market_id"].n_unique()
    failures: list[str] = []
    if strict_markets < gates.minimum_strict_markets:
        failures.append(
            f"validation strict markets {strict_markets} < {gates.minimum_strict_markets}"
        )
    if target_markets < gates.minimum_target_markets:
        failures.append(
            f"validation target markets {target_markets} < {gates.minimum_target_markets}"
        )
    cohort_evidence: dict[str, Any] = {}
    minimum_rate = 1.0
    cohort_frames: list[tuple[str, str, pl.DataFrame, bool]] = []
    for fold in config.folds:
        cohort_frames.extend(
            (
                (fold.name, "fit", _window(dynamic, fold.fit.start, fold.fit.end), True),
                (
                    fold.name,
                    "calibration",
                    _window(dynamic, fold.calibration.start, fold.calibration.end),
                    True,
                ),
                (
                    fold.name,
                    "validation",
                    _window(dynamic, fold.validation.start, fold.validation.end),
                    False,
                ),
            )
        )
    cohort_frames.extend(
        (
            (
                "batch_refit",
                "fit",
                _window(dynamic, config.final_fit.start, config.final_fit.end),
                False,
            ),
            (
                "batch_refit",
                "calibration",
                _window(
                    dynamic,
                    config.final_calibration.start,
                    config.final_calibration.end,
                ),
                False,
            ),
        )
    )
    for lineage, cohort_name, frame, require_two_outcomes in cohort_frames:
        selection_frame = frame if require_two_outcomes else frame.drop("label_up")
        target = (
            select_target_opportunity_rows(
                selection_frame,
                config,
                require_label=True,
                feature_names=DYNAMIC_FEATURES,
            )
            if require_two_outcomes
            else _outcome_blind_target(selection_frame, config)
        )
        horizons: dict[str, Any] = {}
        for horizon in BOOK_DYNAMICS_HORIZONS_SECONDS:
            mature = target.filter(pl.col("seconds_elapsed") >= horizon)
            available = int(mature[f"pm_book_horizon_{horizon}s_mature"].sum())
            rate = available / mature.height if mature.height else 0.0
            minimum_rate = min(minimum_rate, rate)
            passed = rate >= gates.minimum_dynamic_coverage
            if not passed:
                failures.append(
                    f"{lineage}/{cohort_name}/{horizon}s coverage {rate:.6f} "
                    f"< {gates.minimum_dynamic_coverage:.6f}"
                )
            horizons[f"{horizon}s"] = {
                "causally_mature_rows": mature.height,
                "available_rows": available,
                "conditional_coverage": rate,
                "passed": passed,
            }
        cells, cell_failures = _target_cell_support(
            target,
            require_two_outcomes=require_two_outcomes,
        )
        failures.extend(
            f"{lineage}/{cohort_name}/{failure}" for failure in cell_failures
        )
        cohort_evidence[f"{lineage}/{cohort_name}"] = {
            "rows": target.height,
            "markets": target["market_id"].n_unique(),
            "horizons": horizons,
            "side_time_cells": cells,
            "required_two_outcomes_per_cell": require_two_outcomes,
        }
    if failures:
        raise RuntimeError("book-admission readiness failed: " + "; ".join(failures))
    return {
        "strict_rows": strict_validation.height,
        "strict_markets": strict_markets,
        "target_validation_rows": target_validation.height,
        "target_validation_markets": target_markets,
        "target_validation_days": target_validation["window_start"].dt.date().n_unique(),
        "minimum_conditional_dynamic_coverage": minimum_rate,
        "required_minimum_conditional_dynamic_coverage": gates.minimum_dynamic_coverage,
        "cohorts": cohort_evidence,
        "failures": [],
        "passed": True,
    }


def _target_cell_support(
    frame: pl.DataFrame,
    *,
    require_two_outcomes: bool,
) -> tuple[dict[str, Any], list[str]]:
    cells: dict[str, Any] = {}
    failures: list[str] = []
    for side in ("YES", "NO"):
        for lower, upper in ((1, 15), (15, 30), (30, 45), (45, 56)):
            cell = frame.filter(
                (pl.col("selected_side") == side)
                & pl.col("seconds_elapsed").is_between(lower, upper, closed="left")
            )
            outcomes: list[int] = []
            if require_two_outcomes and not cell.is_empty():
                selected_labels = (
                    cell["label_up"]
                    if side == "YES"
                    else 1 - cell["label_up"]
                )
                outcomes = sorted(int(value) for value in selected_labels.unique().to_list())
            name = f"{side}_{lower}_{upper}"
            passed = cell.height > 0 and (not require_two_outcomes or outcomes == [0, 1])
            if not passed:
                failures.append(
                    f"{name} "
                    + ("lacks both selected outcomes" if require_two_outcomes else "is empty")
                )
            cells[name] = {
                "rows": cell.height,
                "markets": cell["market_id"].n_unique() if cell.height else 0,
                "selected_outcomes": outcomes if require_two_outcomes else None,
                "passed": passed,
            }
    return cells, failures


def _write_pre_fit_contracts(
    run_dir: Path,
    *,
    config: BookAdmissionConfig,
    readiness_path: Path,
    readiness: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    dynamics_evidence: Mapping[str, Any],
    coverage: Mapping[str, Any],
    target_validation: pl.DataFrame,
) -> None:
    config_sha256 = file_sha256(config.source_path)
    root = {
        "schema_version": BOOK_ADMISSION_ROOT_CONTRACT_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": config.profile,
        "process_id": config.incumbent.process_id,
        "incumbent_model_key": config.incumbent.model_key,
        "incumbent_model_sha256": config.incumbent.model_sha256,
        "config_sha256": config_sha256,
        "implementation_sha256": _implementation_sha256(),
        "dependency_identity": _dependency_identity(config),
        "evidence_scope": config.evidence_scope,
        "paper_only": True,
        "runtime_deployable": False,
        "batch_forward_eligible": True,
        "process_change_allowed": False,
        "recursive_search": {
            "kind": "bounded_predeclared_candidate_matrix",
            "candidate_ids": [candidate.name for candidate in config.model.candidates],
            "gate_relaxation_allowed": False,
            "pnl_based_candidate_selection_allowed": False,
        },
    }
    write_json_atomic(run_dir / "root-training-contract.json", root)
    write_json_atomic(
        run_dir / "source-readiness-manifest.json",
        {
            "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
            "readiness_path": str(readiness_path),
            "readiness_sha256": file_sha256(readiness_path),
            "readiness_ready": readiness.get("ready") is True,
            "source_evidence": dict(source_evidence),
            "exact_pmxt_only": source_evidence.get("proxy_prices_used") is False,
        },
    )
    write_json_atomic(
        run_dir / "consumed-cohort-registry.json",
        {
            "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
            "development": _window_payload(config.development),
            "excluded_utc_days": [value.isoformat() for value in config.excluded_utc_days],
            "oof_utc_days": [value.isoformat() for value in config.oof_utc_days],
            "folds": [_fold_payload(fold) for fold in config.folds],
            "oof_label_reuse_contract": (
                "a later fold may consume an earlier OOF day's label only after that "
                "day is strictly prior to the later fold validation start"
            ),
            "final_fit": _window_payload(config.final_fit),
            "final_calibration": _window_payload(config.final_calibration),
            "target_definition": "seconds 1-55; either raw YES/NO VWAP5 in [0.20,0.30)",
            "target_key_sha256": _key_sha256(target_validation),
        },
    )
    write_json_atomic(
        run_dir / "candidate-registry.json",
        {
            "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
            "incumbent": INCUMBENT_ID,
            "static_attribution_control": STATIC_CONTROL_ID,
            "candidates": [asdict(candidate) for candidate in config.model.candidates],
            "selection_eligible": [
                candidate.name for candidate in config.model.candidates if candidate.selection_eligible
            ],
        },
    )
    write_json_atomic(
        run_dir / "feature-manifest.json",
        {
            "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
            "dynamics_schema_sha256": book_dynamics_schema_sha256(),
            "dynamics_content_sha256": dynamics_evidence["content_sha256"],
            "dynamics": dict(dynamics_evidence),
            "coverage": dict(coverage),
            "runtime_feature_parity": False,
        },
    )


def _write_prediction_artifacts(
    run_dir: Path,
    *,
    incumbent_predictions: list[pl.DataFrame],
    candidate_predictions: Mapping[str, list[pl.DataFrame]],
    expected_keys: pl.DataFrame,
) -> tuple[dict[str, Path], dict[str, str]]:
    all_frames = {INCUMBENT_ID: incumbent_predictions, **candidate_predictions}
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    expected = expected_keys.sort(*BOOK_ADMISSION_KEY_COLUMNS)
    for candidate_id, frames in all_frames.items():
        prediction = pl.concat(frames, how="vertical").sort(*BOOK_ADMISSION_KEY_COLUMNS)
        if set(prediction.columns) != SEALED_PREDICTION_COLUMNS:
            raise RuntimeError(f"{candidate_id} prediction artifact contains non-probability fields")
        if prediction["candidate_id"].unique().to_list() != [candidate_id]:
            raise RuntimeError(f"{candidate_id} prediction identity changed")
        if not prediction.select(*BOOK_ADMISSION_KEY_COLUMNS).equals(expected, null_equal=True):
            raise RuntimeError(f"{candidate_id} does not preserve the exact OOF target keys")
        if (
            prediction["probability_yes"].null_count()
            or not prediction["probability_yes"].is_finite().all()
            or not prediction["probability_yes"].is_between(
                0.0, 1.0, closed="none"
            ).all()
        ):
            raise RuntimeError(f"{candidate_id} contains invalid probabilities")
        path = run_dir / "candidates" / candidate_id / "oof-predictions.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_parquet(path, compression="zstd", statistics=True)
        paths[candidate_id] = path
        hashes[candidate_id] = file_sha256(path)
    return paths, hashes


def _selection_support(
    config: BookAdmissionConfig,
    coverage: Mapping[str, Any],
    trace: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for candidate in config.model.candidates:
        folds = trace[candidate.name]
        both_outcomes = all(
            value[cohort]["positive_selected_outcomes"] > 0
            and value[cohort]["negative_selected_outcomes"] > 0
            for value in folds
            for cohort in ("fit", "calibration")
        )
        cap_passed = all(
            value["fit_evidence"]["raw_residual_minimum"]
            >= -config.model.residual_logit_cap - 1e-12
            and value["fit_evidence"]["raw_residual_maximum"]
            <= config.model.residual_logit_cap + 1e-12
            for value in folds
        )
        feature_coverage = (
            1.0
            if candidate.feature_set == "static"
            else float(coverage["minimum_conditional_dynamic_coverage"])
        )
        complexity = (
            len(candidate.feature_names) + 1
            if candidate.estimator == "linear"
            else int(candidate.max_iter or 0) * int(candidate.max_leaf_nodes or 0)
        )
        result[candidate.name] = {
            "support_passed": bool(both_outcomes and len(folds) == len(config.folds)),
            "coverage": feature_coverage,
            "residual_cap_passed": cap_passed,
            "regularization_strength": candidate.l2_regularization,
            "model_complexity": complexity,
            "fold_count": len(folds),
            "both_outcomes_in_every_fit_and_calibration": both_outcomes,
            "fold_support": folds,
        }
    return result


def _write_pre_outcome_artifact_manifest(
    run_dir: Path,
    *,
    config: BookAdmissionConfig,
    readiness_path: Path,
    model_manifest_path: Path,
    prediction_manifest_path: Path,
    support_path: Path,
) -> Path:
    internal_names = (
        "root-training-contract.json",
        "source-readiness-manifest.json",
        "consumed-cohort-registry.json",
        "candidate-registry.json",
        "feature-manifest.json",
    )
    internal = {
        name: file_sha256(run_dir / name)
        for name in internal_names
    }
    internal.update(
        {
            model_manifest_path.name: file_sha256(model_manifest_path),
            prediction_manifest_path.name: file_sha256(prediction_manifest_path),
            support_path.name: file_sha256(support_path),
        }
    )
    incumbent_files = {
        name: file_sha256(config.paths.incumbent_runtime_dir / name)
        for name in ("model.json", "manifest.json", "golden-vectors.json")
    }
    implementation_root = Path(__file__).resolve().parent
    implementation = {
        name: file_sha256(implementation_root / name)
        for name in IMPLEMENTATION_FILES
    }
    payload = {
        "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "internal_artifact_sha256": internal,
        "external_contract_sha256": {
            "book_admission_config": file_sha256(config.source_path),
            "source_asymmetric_value_config": file_sha256(
                config.paths.source_asymmetric_value_config
            ),
            "readiness_manifest": file_sha256(readiness_path),
            "pyproject": file_sha256(config.package_root / "pyproject.toml"),
            "requirements_lock": file_sha256(
                config.package_root / "requirements.lock"
            ),
            "dependency_identity": _dependency_identity(config),
            "incumbent_runtime": incumbent_files,
            "implementation": implementation,
        },
    }
    payload["aggregate_sha256"] = _canonical_sha256(
        {
            "internal": internal,
            "external": payload["external_contract_sha256"],
        }
    )
    path = run_dir / "pre-outcome-artifact-manifest.json"
    write_json_atomic(path, payload)
    return path


def _write_outcome_access_seal(
    run_dir: Path,
    *,
    pre_outcome_manifest_path: Path,
    model_manifest_path: Path,
    prediction_manifest_path: Path,
    support_path: Path,
    prediction_hashes: Mapping[str, str],
) -> Path:
    for candidate_id, expected in prediction_hashes.items():
        path = run_dir / "candidates" / candidate_id / "oof-predictions.parquet"
        if file_sha256(path) != expected:
            raise RuntimeError("prediction changed before the outcome-access seal")
        observed_schema = set(pl.read_parquet_schema(path).names())
        if observed_schema != SEALED_PREDICTION_COLUMNS:
            raise RuntimeError("pre-seal prediction artifact does not match its exact allowlist")
    path = run_dir / "outcome-access-seal.json"
    write_json_atomic(
        path,
        {
            "schema_version": BOOK_ADMISSION_OUTCOME_SEAL_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "pre_outcome_artifact_manifest_sha256": file_sha256(
                pre_outcome_manifest_path
            ),
            "fold_model_manifest_sha256": file_sha256(model_manifest_path),
            "prediction_manifest_sha256": file_sha256(prediction_manifest_path),
            "candidate_support_manifest_sha256": file_sha256(support_path),
            "prediction_sha256": dict(prediction_hashes),
            "probability_selection_labels_opened_after_this_seal": True,
            "strictly_prior_oof_labels_may_enter_later_fold_fit_or_calibration": True,
            "economics_opened": False,
        },
    )
    return path


def _load_sealed_probability_frame(
    path: Path,
    expected_sha256: str,
    outcome_context: pl.DataFrame,
    *,
    probability_only: bool = True,
) -> pl.DataFrame:
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("sealed OOF predictions changed")
    prediction = pl.read_parquet(path)
    keys = list(BOOK_ADMISSION_KEY_COLUMNS)
    if (
        prediction.height != outcome_context.height
        or prediction.select(*keys).is_duplicated().any()
        or outcome_context.select(*keys).is_duplicated().any()
        or not prediction.select(*keys)
        .sort(keys)
        .equals(outcome_context.select(*keys).sort(keys), null_equal=True)
    ):
        raise RuntimeError("sealed OOF predictions do not match validation outcomes")
    joined = prediction.join(outcome_context, on=keys, how="inner", validate="1:1")
    if joined.height != outcome_context.height:
        raise RuntimeError("validation outcome join changed the sealed target grid")
    if probability_only and set(joined.columns) != PROBABILITY_SELECTION_FRAME_COLUMNS:
        raise RuntimeError("probability selector frame contains non-allowlisted context")
    return joined.sort(*keys)


def _write_selection_seal(
    run_dir: Path,
    *,
    selection_path: Path,
    outcome_seal_path: Path,
    model_manifest_path: Path,
    prediction_manifest_path: Path,
    pooled_oof_support_path: Path,
    selected_candidate_id: str | None,
) -> Path:
    path = run_dir / "probability-selection-seal.json"
    write_json_atomic(
        path,
        {
            "schema_version": BOOK_ADMISSION_SELECTION_SEAL_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "selected_candidate_id": selected_candidate_id,
            "probability_selection_sha256": file_sha256(selection_path),
            "outcome_access_seal_sha256": file_sha256(outcome_seal_path),
            "fold_model_manifest_sha256": file_sha256(model_manifest_path),
            "prediction_manifest_sha256": file_sha256(prediction_manifest_path),
            "pooled_oof_cell_support_sha256": file_sha256(pooled_oof_support_path),
            "economics_opened": False,
            "selection_uses_economics": False,
        },
    )
    return path


def _write_selected_refit_support(
    run_dir: Path,
    *,
    config: BookAdmissionConfig,
    dynamic: pl.DataFrame,
    selection_seal_path: Path,
) -> tuple[Path, bool]:
    evidence: dict[str, Any] = {}
    failures: list[str] = []
    for cohort_name, evidence_window in (
        ("fit", config.final_fit),
        ("calibration", config.final_calibration),
    ):
        target = select_target_opportunity_rows(
            _window(dynamic, evidence_window.start, evidence_window.end),
            config,
            require_label=True,
            feature_names=DYNAMIC_FEATURES,
        )
        cells, cell_failures = _target_cell_support(
            target,
            require_two_outcomes=True,
        )
        failures.extend(f"{cohort_name}/{failure}" for failure in cell_failures)
        evidence[cohort_name] = {
            "window": _window_payload(evidence_window),
            "rows": target.height,
            "markets": target["market_id"].n_unique(),
            "side_time_cells": cells,
        }
    path = run_dir / "selected-refit-support.json"
    write_json_atomic(
        path,
        {
            "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
            "accessed_after_probability_selection_seal_sha256": file_sha256(
                selection_seal_path
            ),
            "lineage_scope": "selected_batch_refit",
            "cohorts": evidence,
            "failures": failures,
            "passed": not failures,
        },
    )
    return path, not failures


def _refit_selected_model(
    run_dir: Path,
    *,
    config: BookAdmissionConfig,
    selected: str,
    dynamic: pl.DataFrame,
    selection_seal_path: Path,
    refit_support_path: Path,
) -> tuple[Any, Path, Path]:
    if not (
        config.development.start <= config.final_fit.start < config.final_fit.end
        <= config.final_calibration.start < config.final_calibration.end
        <= config.development.end
    ):
        raise RuntimeError("selected batch-refit chronology is not strictly causal")
    fit = _window(dynamic, config.final_fit.start, config.final_fit.end)
    calibration = _window(
        dynamic,
        config.final_calibration.start,
        config.final_calibration.end,
    )
    model = fit_book_admission_candidate(config, selected, fit, calibration)
    artifact_path = run_dir / "selected-book-admission" / "model.pkl"
    serialization = model.serialize(artifact_path)
    calibration_without_outcomes = calibration.drop("label_up")
    before = score_book_admission_model(model, calibration_without_outcomes, config)
    reloaded = _load_verified_model(artifact_path, serialization)
    _validate_serialized_model(
        reloaded,
        config=config,
        candidate_name=selected,
        artifact_path=artifact_path,
        serialization=serialization,
    )
    after = score_book_admission_model(reloaded, calibration_without_outcomes, config)
    if (
        before.probability_sha256 != after.probability_sha256
        or before.model_semantic_sha256 != after.model_semantic_sha256
    ):
        raise RuntimeError("selected final model changed across serialization")
    seal_path = run_dir / "selected-model-seal.json"
    write_json_atomic(
        seal_path,
        {
            "schema_version": BOOK_ADMISSION_SELECTED_MODEL_SEAL_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "lineage_scope": "final_batch_forward_refit",
            "development_oof_models_reused": False,
            "selected_candidate_id": selected,
            "selection_seal_sha256": file_sha256(selection_seal_path),
            "selected_refit_support_sha256": file_sha256(refit_support_path),
            "model_path": str(artifact_path.relative_to(run_dir)),
            "model_artifact_sha256": file_sha256(artifact_path),
            "model_semantic_sha256": model.semantic_sha256,
            "calibration_prediction_sha256": after.probability_sha256,
            "serialization": serialization,
            "fit_evidence": asdict(model.evidence),
            "economics_opened": False,
            "runtime_deployable": False,
        },
    )
    return reloaded, artifact_path, seal_path


def _evaluate_selected_economics(
    config: BookAdmissionConfig,
    *,
    run_dir: Path,
    incumbent: pl.DataFrame,
    challenger: pl.DataFrame,
    validation: pl.DataFrame,
) -> dict[str, Any]:
    incumbent_ledger = _economic_ledger(incumbent, INCUMBENT_ID)
    challenger_id = challenger["candidate_id"].unique().item()
    challenger_ledger = _economic_ledger(challenger, challenger_id)
    incumbent_ledger_path = run_dir / "incumbent-oof-policy-ledger.parquet"
    challenger_ledger_path = run_dir / "selected-oof-policy-ledger.parquet"
    incumbent_ledger.write_parquet(
        incumbent_ledger_path, compression="zstd", statistics=True
    )
    challenger_ledger.write_parquet(
        challenger_ledger_path, compression="zstd", statistics=True
    )
    correction, correction_summary = build_incumbent_correction_ledger(
        incumbent_ledger,
        challenger_ledger,
        quantity=FROZEN_ASYMMETRIC_INCUMBENT_QUANTITY,
    )
    correction_path = run_dir / "incumbent-correction-ledger.parquet"
    correction.write_parquet(correction_path, compression="zstd", statistics=True)
    eligible_markets = validation.select("market_id", "window_start").unique().sort(
        "window_start", "market_id"
    )
    paired = (
        {
            "status": "not_run_empty_challenger_ledger",
            "reason": "paired helper requires at least one challenger trade",
        }
        if challenger_ledger.is_empty()
        else paired_incumbent_economics(
            incumbent_ledger,
            challenger_ledger,
            eligible_markets=eligible_markets,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
        )
    )
    incumbent_metrics = _normalized_ledger_metrics(ledger_metrics(incumbent_ledger))
    challenger_metrics = _normalized_ledger_metrics(ledger_metrics(challenger_ledger))
    correction_candidate_only = correction.filter(
        pl.col("correction_category").is_in(["candidate_only_win", "candidate_only_loss"])
    )
    candidate_only_stressed = float(
        correction_candidate_only["challenger_stress_1c_net"].sum()
    )
    candidate_only_expectancy = (
        candidate_only_stressed / correction_candidate_only.height
        if correction_candidate_only.height
        else None
    )
    candidate_only_passed = bool(
        correction_candidate_only.is_empty()
        or (candidate_only_expectancy is not None and candidate_only_expectancy > 0.0)
    )
    incumbent_trades = int(incumbent_metrics["trades"])
    challenger_trades = int(challenger_metrics["trades"])
    incumbent_profit_per_market = (
        float(incumbent_metrics["stress_1c_net_profit"]) / eligible_markets.height
    )
    challenger_profit_per_market = float(challenger_metrics["stress_1c_net_profit"]) / (
        eligible_markets.height
    )
    incumbent_ev = float(incumbent_metrics["stress_1c_net_expectancy_per_trade"])
    challenger_ev = challenger_metrics["stress_1c_net_expectancy_per_trade"]
    incumbent_pf = incumbent_metrics["profit_factor"]
    challenger_pf = challenger_metrics["profit_factor"]
    pf_passed = bool(
        challenger_metrics.get("profit_factor_no_losses")
        or (
            challenger_pf is not None
            and challenger_pf >= config.economic_gates.minimum_profit_factor
            and (
                incumbent_pf is None
                or challenger_pf
                >= config.economic_gates.minimum_profit_factor_fraction_of_incumbent
                * incumbent_pf
            )
        )
    )
    incumbent_accuracy = float(incumbent_metrics["accuracy"])
    challenger_accuracy = challenger_metrics["accuracy"]
    challenger_margin = challenger_metrics["selected_win_rate_advantage"]
    average_loss_limit = min(
        config.economic_gates.maximum_average_loss,
        1.10 * abs(float(incumbent_metrics["average_loss"] or 0.0)),
    )
    maximum_loss_limit = min(
        config.economic_gates.maximum_single_loss,
        1.10 * abs(float(incumbent_metrics["maximum_loss"] or 0.0)),
    )
    drawdown_limit = min(
        config.economic_gates.maximum_drawdown,
        1.10 * abs(float(incumbent_metrics["maximum_drawdown"] or 0.0)),
    )
    price_limit = min(
        config.economic_gates.maximum_mean_share_price,
        float(incumbent_metrics["mean_share_price"]) + 0.005,
    )
    regression = config.economic_gates.maximum_primary_metric_regression_fraction
    ev_improves = challenger_ev is not None and challenger_ev > incumbent_ev
    profit_improves = challenger_profit_per_market > incumbent_profit_per_market
    profit_within_limit = challenger_profit_per_market >= (
        incumbent_profit_per_market - regression * abs(incumbent_profit_per_market)
    )
    ev_within_limit = challenger_ev is not None and challenger_ev >= (
        incumbent_ev - regression * abs(incumbent_ev)
    )
    gates = [
        _gate(
            "minimum_incumbent_frequency",
            challenger_trades,
            math.ceil(config.economic_gates.minimum_incumbent_frequency_fraction * incumbent_trades),
            ">=",
        ),
        _gate("minimum_yes_entries", challenger_metrics.get("yes_trades", 0), config.economic_gates.minimum_yes_entries, ">="),
        _gate("minimum_no_entries", challenger_metrics.get("no_trades", 0), config.economic_gates.minimum_no_entries, ">="),
        _gate("positive_stress_1c_expectancy", challenger_ev, config.economic_gates.minimum_stressed_expectancy_per_trade, ">"),
        _gate("profit_factor", pf_passed, True, "=="),
        _gate("maximum_mean_share_price", challenger_metrics["mean_share_price"], price_limit, "<="),
        _gate("maximum_loss_recovery_burden", challenger_metrics["loss_recovery_wins"], config.economic_gates.maximum_loss_recovery_burden, "<="),
        _gate("maximum_average_loss", abs(float(challenger_metrics["average_loss"] or 0.0)), average_loss_limit, "<="),
        _gate("maximum_single_loss", abs(float(challenger_metrics["maximum_loss"] or 0.0)), maximum_loss_limit, "<="),
        _gate("maximum_drawdown", abs(float(challenger_metrics["maximum_drawdown"] or 0.0)), drawdown_limit, "<="),
        _gate("minimum_candidate_accuracy", challenger_accuracy, config.correction_gates.minimum_candidate_accuracy, ">="),
        _gate(
            "minimum_accuracy_delta",
            challenger_accuracy - incumbent_accuracy if challenger_accuracy is not None else None,
            config.correction_gates.minimum_accuracy_delta,
            ">=",
        ),
        _gate("minimum_selected_win_rate_margin", challenger_margin, config.correction_gates.minimum_correctness_margin, ">="),
        _gate("minimum_net_corrected_decisions", correction_summary["net_corrected_decisions"], config.correction_gates.minimum_net_corrected_decisions, ">="),
        _gate("minimum_corrected_decision_days", correction_summary["corrected_decision_improvement_utc_days"], config.correction_gates.minimum_improvement_days, ">="),
        _gate("candidate_only_positive_stressed_expectancy", candidate_only_passed, True, "=="),
        _gate("positive_yes_net_profit", challenger_metrics["yes_net_profit"], 0.0, ">"),
        _gate("positive_no_net_profit", challenger_metrics["no_net_profit"], 0.0, ">"),
        _gate(
            "expectancy_profit_tradeoff",
            (ev_improves and profit_within_limit) or (profit_improves and ev_within_limit),
            True,
            "==",
        ),
    ]
    return {
        "schema_version": BOOK_ADMISSION_ECONOMICS_SCHEMA_VERSION,
        "status": "qualified" if all(gate["passed"] for gate in gates) else "not_qualified",
        "selected_candidate_id": challenger_id,
        "eligible_resolved_markets": eligible_markets.height,
        "incumbent": {
            "metrics": incumbent_metrics,
            "stress_1c_net_profit_per_eligible_market": incumbent_profit_per_market,
        },
        "challenger": {
            "metrics": challenger_metrics,
            "stress_1c_net_profit_per_eligible_market": challenger_profit_per_market,
        },
        "challenger_minus_incumbent_stress_1c_net_profit_per_eligible_market": (
            challenger_profit_per_market - incumbent_profit_per_market
        ),
        "correction_summary": correction_summary,
        "correction_ledger": {
            "path": correction_path.name,
            "sha256": file_sha256(correction_path),
        },
        "policy_ledgers": {
            "incumbent": {
                "path": incumbent_ledger_path.name,
                "sha256": file_sha256(incumbent_ledger_path),
            },
            "challenger": {
                "path": challenger_ledger_path.name,
                "sha256": file_sha256(challenger_ledger_path),
            },
        },
        "candidate_only_stress_1c_expectancy": candidate_only_expectancy,
        "candidate_only_entries": correction_candidate_only.height,
        "strict_gates": gates,
        "paired_development_diagnostic": paired,
        "paired_lower_95_role": "report_only_not_a_development_qualification_gate",
        "evaluated_candidates": [INCUMBENT_ID, challenger_id],
    }


def _normalized_ledger_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(metrics)
    defaults = {
        "trades": 0,
        "yes_trades": 0,
        "no_trades": 0,
        "accuracy": None,
        "net_profit": 0.0,
        "profit_factor": None,
        "profit_factor_no_losses": False,
        "mean_share_price": None,
        "selected_win_rate_advantage": None,
        "stress_1c_net_profit": 0.0,
        "stress_1c_net_expectancy_per_trade": None,
        "loss_recovery_wins": None,
        "average_loss": 0.0,
        "maximum_loss": 0.0,
        "maximum_drawdown": 0.0,
        "yes_net_profit": 0.0,
        "no_net_profit": 0.0,
    }
    for key, value in defaults.items():
        normalized.setdefault(key, value)
    return normalized


def _economic_ledger(frame: pl.DataFrame, candidate_id: str) -> pl.DataFrame:
    probability = frame["probability_yes"].to_numpy()
    source = frame.drop("probability_yes", "candidate_id")
    predictions = asymmetric_probability_frame(source, probability, model=candidate_id)
    return policy_ledger(
        score_two_sided_value(predictions),
        FROZEN_ASYMMETRIC_INCUMBENT_POLICY,
        quantity=FROZEN_ASYMMETRIC_INCUMBENT_QUANTITY,
        maximum_depth_participation=FROZEN_ASYMMETRIC_INCUMBENT_MAXIMUM_DEPTH_PARTICIPATION,
    )


def _authorize_batch_forward(
    run_dir: Path,
    *,
    config: BookAdmissionConfig,
    selected: str,
    final_model_path: Path,
    selection_seal_path: Path,
    selected_model_seal_path: Path,
    economics_access_path: Path,
    economics_path: Path,
) -> Path:
    selection_seal = json.loads(selection_seal_path.read_text())
    selected_model_seal = json.loads(selected_model_seal_path.read_text())
    economics_access = json.loads(economics_access_path.read_text())
    economics = json.loads(economics_path.read_text())
    model_sha256 = file_sha256(final_model_path)
    if (
        selection_seal.get("schema_version")
        != BOOK_ADMISSION_SELECTION_SEAL_SCHEMA_VERSION
        or selection_seal.get("selected_candidate_id") != selected
        or selection_seal.get("economics_opened") is not False
        or selection_seal.get("selection_uses_economics") is not False
        or selected_model_seal.get("schema_version")
        != BOOK_ADMISSION_SELECTED_MODEL_SEAL_SCHEMA_VERSION
        or selected_model_seal.get("selected_candidate_id") != selected
        or selected_model_seal.get("selection_seal_sha256")
        != file_sha256(selection_seal_path)
        or selected_model_seal.get("model_artifact_sha256") != model_sha256
        or selected_model_seal.get("economics_opened") is not False
        or selected_model_seal.get("runtime_deployable") is not False
        or selected_model_seal.get("lineage_scope") != "final_batch_forward_refit"
        or selected_model_seal.get("development_oof_models_reused") is not False
        or economics_access.get("schema_version")
        != BOOK_ADMISSION_ECONOMICS_SCHEMA_VERSION
        or economics_access.get("selected_candidate_id") != selected
        or economics_access.get("selection_seal_sha256")
        != file_sha256(selection_seal_path)
        or economics_access.get("selected_model_seal_sha256")
        != file_sha256(selected_model_seal_path)
        or economics_access.get("economics_opened") is not True
        or economics_access.get("evaluated_candidates") != [INCUMBENT_ID, selected]
        or economics.get("schema_version") != BOOK_ADMISSION_ECONOMICS_SCHEMA_VERSION
        or economics.get("status") != "qualified"
        or economics.get("selected_candidate_id") != selected
        or economics.get("selection_seal_sha256") != file_sha256(selection_seal_path)
        or economics.get("selected_model_seal_sha256")
        != file_sha256(selected_model_seal_path)
        or economics.get("selected_model_artifact_sha256") != model_sha256
        or economics.get("economics_access_sha256")
        != file_sha256(economics_access_path)
        or economics.get("economic_scope") != "development_oof"
        or economics.get("final_batch_refit_economically_scored") is not False
        or set(economics.get("development_prediction_sha256", {}))
        != {INCUMBENT_ID, selected}
        or economics.get("evaluated_candidates") != [INCUMBENT_ID, selected]
    ):
        raise RuntimeError("batch-forward authorization lineage is inconsistent")
    destination = run_dir / "batch-forward" / "selected-book-admission.pkl"
    destination.parent.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(final_model_path, destination)
    if file_sha256(destination) != model_sha256:
        raise RuntimeError("batch-forward artifact copy changed model bytes")
    manifest_path = destination.with_name("manifest.json")
    write_json_atomic(
        manifest_path,
        {
            "schema_version": BOOK_ADMISSION_FORWARD_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "process_id": config.incumbent.process_id,
            "parent_model_key": config.incumbent.model_key,
            "parent_model_sha256": config.incumbent.model_sha256,
            "selected_candidate_id": selected,
            "artifact": destination.name,
            "artifact_sha256": file_sha256(destination),
            "selection_seal_sha256": file_sha256(selection_seal_path),
            "selected_model_seal_sha256": file_sha256(selected_model_seal_path),
            "economics_access_sha256": file_sha256(economics_access_path),
            "economics_sha256": file_sha256(economics_path),
            "economic_scope": "development_oof",
            "development_prediction_sha256": economics[
                "development_prediction_sha256"
            ],
            "final_batch_refit_sha256": model_sha256,
            "final_batch_refit_economically_scored": False,
            "probability_qualified": True,
            "economics_qualified": True,
            "runtime_deployable": False,
            "batch_forward_eligible": True,
            "blocked_runtime_feature_parity": True,
            "paper_process_change_authorized": False,
            "live_capital_allowed": False,
            "forward_requirements": FORWARD_REQUIREMENTS,
        },
    )
    return destination


def _finalize(
    run_dir: Path,
    result: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    economics: Mapping[str, Any] | None,
) -> None:
    selected_record = next(
        (
            record
            for record in selection.get("candidate_records", [])
            if record.get("candidate_id") == result.get("selected_candidate_id")
        ),
        None,
    )
    incumbent_probability = selection.get("incumbent", {}).get("target_metrics", {}).get(
        "overall", {}
    )
    selected_probability = (
        selected_record.get("target_metrics", {}).get("overall", {})
        if selected_record is not None
        else {}
    )
    probability_failures = selection.get("failure_trace", [])
    economic_failures = (
        [gate["name"] for gate in economics["strict_gates"] if not gate["passed"]]
        if economics is not None
        else []
    )
    report = [
        "# Core+Oracle Polymarket Book Admission Benchmark",
        "",
        f"- Status: `{result['status']}`",
        f"- Selected probability challenger: `{result.get('selected_candidate_id') or 'none'}`",
        f"- Economics opened: `{str(bool(result.get('economics_opened'))).lower()}`",
        f"- Batch-forward artifact: `{result.get('batch_forward_artifact') or 'none'}`",
        "- Runtime deployable: `false`",
        "- Existing paper process changed: `false`",
        f"- Development OOF model manifest SHA-256: `{file_sha256(run_dir / 'fold-model-manifest.json')}`",
        (
            "- Selected batch-refit model seal SHA-256: `"
            + (
                file_sha256(run_dir / "selected-model-seal.json")
                if (run_dir / "selected-model-seal.json").is_file()
                else "none"
            )
            + "`"
        ),
        "",
        "## Probability evidence",
        "",
        f"- Incumbent target Brier: `{incumbent_probability.get('brier')}`",
        f"- Incumbent target log loss: `{incumbent_probability.get('log_loss')}`",
        f"- Selected target Brier: `{selected_probability.get('brier')}`",
        f"- Selected target log loss: `{selected_probability.get('log_loss')}`",
        f"- Probability failure trace: `{json.dumps(probability_failures, sort_keys=True)}`",
        "",
        "The candidate matrix was frozen before outcome access. PnL was opened only for the",
        "probability-selected challenger, and no probability or economic gate was relaxed.",
        "Dynamic PMXT history is available only in the offline batch contract, so even a",
        "qualified artifact remains blocked from the current Rust paper runtime.",
    ]
    if economics is not None:
        incumbent_metrics = economics["incumbent"]["metrics"]
        challenger_metrics = economics["challenger"]["metrics"]
        report.extend(
            [
                "",
                "## Sealed economic evidence",
                "",
                f"- Economic scope: `{economics.get('economic_scope')}`",
                (
                    "- Development OOF prediction SHA-256: `"
                    f"{json.dumps(economics.get('development_prediction_sha256'), sort_keys=True)}`"
                ),
                "- Selected batch refit economically scored: `false`",
                (
                    "| Arm | Trades | Accuracy | PnL | +1c EV/trade | PF | Max drawdown |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|",
                _metric_row("Incumbent", incumbent_metrics),
                _metric_row(str(result.get("selected_candidate_id")), challenger_metrics),
                "",
                f"- Economic failed gates: `{json.dumps(economic_failures)}`",
                (
                    "- Paired development lower-95: report-only; it did not alter "
                    "development qualification."
                ),
            ]
        )
    (run_dir / "benchmark-report.md").write_text("\n".join(report) + "\n")
    write_json_atomic(
        run_dir / "benchmark-result.json",
        {
            **dict(result),
            "selection_status": selection.get("status"),
            "failed_probability_gates": probability_failures,
            "failed_economic_gates": economic_failures,
            "development_oof_model_manifest_sha256": file_sha256(
                run_dir / "fold-model-manifest.json"
            ),
            "selected_batch_refit_model_seal_sha256": (
                file_sha256(run_dir / "selected-model-seal.json")
                if (run_dir / "selected-model-seal.json").is_file()
                else None
            ),
        },
    )
    manifest = _artifact_manifest(run_dir)
    write_json_atomic(run_dir / "artifact-manifest.json", manifest)
    bundle = {
        "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_manifest_sha256": file_sha256(run_dir / "artifact-manifest.json"),
        "benchmark_result_sha256": file_sha256(run_dir / "benchmark-result.json"),
        "benchmark_report_sha256": file_sha256(run_dir / "benchmark-report.md"),
    }
    bundle["aggregate_sha256"] = _canonical_sha256(bundle)
    write_json_atomic(run_dir / "benchmark-bundle-seal.json", bundle)


def _metric_row(name: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"| {name} | {metrics.get('trades')} | {metrics.get('accuracy')} | "
        f"{metrics.get('net_profit')} | {metrics.get('stress_1c_net_expectancy_per_trade')} | "
        f"{metrics.get('profit_factor')} | {metrics.get('maximum_drawdown')} |"
    )


def _artifact_manifest(run_dir: Path) -> dict[str, Any]:
    artifacts = {}
    for path in sorted(value for value in run_dir.rglob("*") if value.is_file()):
        if path.name == "artifact-manifest.json":
            continue
        artifacts[str(path.relative_to(run_dir))] = file_sha256(path)
    return {
        "schema_version": BOOK_ADMISSION_BENCHMARK_SCHEMA_VERSION,
        "artifacts": artifacts,
        "aggregate_sha256": _canonical_sha256(artifacts),
    }


def _gate(name: str, observed: Any, threshold: Any, operator: str) -> dict[str, Any]:
    passed = False
    if observed is not None:
        if operator == ">=":
            passed = bool(observed >= threshold)
        elif operator == ">":
            passed = bool(observed > threshold)
        elif operator == "<=":
            passed = bool(observed <= threshold)
        elif operator == "==":
            passed = bool(observed == threshold)
        else:
            raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "operator": operator,
        "passed": passed,
    }


def _key_sha256(frame: pl.DataFrame) -> str:
    ordered = frame.select(*BOOK_ADMISSION_KEY_COLUMNS).sort(*BOOK_ADMISSION_KEY_COLUMNS)
    digest = hashlib.sha256(b"btc-asymmetric-book-admission-runner-keys-v1\n")
    for row in ordered.iter_rows():
        for value in row:
            encoded = (value.isoformat() if isinstance(value, datetime) else str(value)).encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _window_payload(value: Any) -> dict[str, str]:
    return {
        "start": value.start.isoformat(),
        "end": value.end.isoformat(),
    }


def _fold_payload(value: Any) -> dict[str, Any]:
    return {
        "name": value.name,
        "fit": _window_payload(value.fit),
        "calibration": _window_payload(value.calibration),
        "validation": _window_payload(value.validation),
    }


def _implementation_sha256() -> str:
    root = Path(__file__).resolve().parent
    paths = tuple(root / name for name in IMPLEMENTATION_FILES)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _dependency_identity(config: BookAdmissionConfig) -> dict[str, Any]:
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
        "requirements_lock_sha256": file_sha256(config.package_root / "requirements.lock"),
    }


def load_and_run_book_admission_benchmark(
    config_path: Path,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    return run_book_admission_benchmark(load_book_admission_config(config_path), force=force)


__all__ = [
    "load_and_run_book_admission_benchmark",
    "run_book_admission_benchmark",
]
