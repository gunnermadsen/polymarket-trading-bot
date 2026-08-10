"""Seal-first calibration benchmark for the deployed asymmetric Core+Oracle model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from .asymmetric_gen2_export import (
    GEN2_EXPORT_AUTHORIZATION_SCHEMA_VERSION,
    GEN2_SELECTION_SEAL_SCHEMA_VERSION,
    build_asymmetric_gen2_candidate_payload,
    export_asymmetric_gen2_paper_model,
    frozen_core_oracle_process_metadata,
)
from .asymmetric_incumbent_calibration import (
    IncumbentCalibrationConfig,
    clone_incumbent_with_calibration,
    fit_incumbent_calibration_arm,
    frozen_parent_probabilities,
    load_incumbent_calibration_config,
    score_incumbent_calibration_payload,
    validate_incumbent_calibration_parity,
)
from .asymmetric_incumbent_estimator import (
    ESTIMATOR_FALLBACK_CANDIDATES,
    FittedEstimatorFallback,
    fit_configured_core_oracle_estimator_fallbacks,
    refit_selected_core_oracle_estimator_fallback,
    score_core_oracle_estimator_fallback,
)
from .asymmetric_incumbent_evaluation import (
    build_incumbent_correction_ledger,
    paired_incumbent_economics,
    select_incumbent_calibration_challenger,
)
from .asymmetric_incumbent_replay import (
    FrozenAsymmetricRuntimeModel,
    load_frozen_asymmetric_incumbent,
)
from .asymmetric_training_readiness import (
    oracle_source_inventory,
    prepare_asymmetric_training_readiness,
)
from .asymmetric_value_benchmark import (
    DEVELOPMENT_ORACLE_CACHE,
    _frame_content_digest,
    _load_asymmetric_core_grid,
    _load_or_build_oracle_core,
    _project_candidate_source,
)
from .asymmetric_value_config import load_asymmetric_value_config
from .asymmetric_value_data import (
    attach_asymmetric_value_features,
    extract_asymmetric_price_evidence,
    load_asymmetric_price_evidence,
    price_manifest_identity_sha256,
)
from .asymmetric_value_evaluation import ledger_metrics, policy_ledger, score_two_sided_value
from .asymmetric_value_training import (
    CORE_ORACLE_PRICE,
    ORACLE_MATCHED_CORE_PRICE_CONTROL,
    asymmetric_probability_frame,
)
from .core_config import load_core_config
from .core_extract import extract_core_source, file_sha256, write_json_atomic
from .core_features import build_core_features
from .runtime_export import canonical_json_bytes, export_asymmetric_value_runtime_model

INCUMBENT_BENCHMARK_SCHEMA_VERSION = "btc-asymmetric-incumbent-calibration-benchmark-v1"
INCUMBENT_ROOT_CONTRACT_SCHEMA_VERSION = "btc-asymmetric-incumbent-root-contract-v1"
INCUMBENT_CANDIDATE_REGISTRY_SCHEMA_VERSION = "btc-asymmetric-incumbent-candidate-registry-v1"
INCUMBENT_OUTCOME_ACCESS_SEAL_SCHEMA_VERSION = "btc-asymmetric-incumbent-outcome-access-seal-v1"
INCUMBENT_CALIBRATION_SELECTION_LINEAGE_SCHEMA_VERSION = (
    "btc-asymmetric-incumbent-calibration-selection-lineage-v1"
)
INCUMBENT_ECONOMICS_SCHEMA_VERSION = "btc-asymmetric-incumbent-economics-v1"
INCUMBENT_ESTIMATOR_SELECTION_SEAL_SCHEMA_VERSION = (
    "btc-asymmetric-incumbent-estimator-selection-seal-v1"
)
INCUMBENT_ESTIMATOR_ECONOMICS_SCHEMA_VERSION = "btc-asymmetric-incumbent-estimator-economics-v1"
INCUMBENT_ESTIMATOR_EXPORT_AUTHORIZATION_SCHEMA_VERSION = (
    "btc-asymmetric-incumbent-estimator-export-authorization-v1"
)
INCUMBENT_EXPORT_AUTHORIZATION_SCHEMA_VERSION = GEN2_EXPORT_AUTHORIZATION_SCHEMA_VERSION
CALIBRATION_CHALLENGERS = (
    "C1_supported",
    "C2_compressed",
    "C3_day_balanced",
)
INCUMBENT_CONTROL = "I0_incumbent"
GEN2_MODEL_KEY = "btc-5m-asymmetric-core-oracle-gen2-calibrated-paper-20260809-v1"
ESTIMATOR_MODEL_KEYS = {
    "E1_hybrid50_h3": "btc-5m-asymmetric-core-oracle-hybrid50-h3-paper-20260809-v1",
    "E2_hybrid50_h3_boundary_weighted": (
        "btc-5m-asymmetric-core-oracle-hybrid50-h3-boundary-paper-20260809-v1"
    ),
}


def run_incumbent_calibration_benchmark(
    config: IncumbentCalibrationConfig,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Run the bounded calibration lineage without changing its source process."""

    _validate_runner_gate_contract(config)
    source_config = load_asymmetric_value_config(config.asymmetric_value_config)
    core_config = load_core_config(source_config.core_config)
    readiness_path, readiness = prepare_asymmetric_training_readiness(
        source_config,
        output_dir=config.runs.parent / "btc-asymmetric-core-oracle-gen2-readiness",
    )
    if readiness.get("ready") is not True:
        raise RuntimeError("incumbent calibration source readiness did not pass")

    source_frame, source_evidence = _load_core_oracle_value_frame(
        source_config,
        core_config=core_config,
        force=force,
    )
    if source_evidence.get("proxy_prices_used") is not False:
        raise RuntimeError("incumbent calibration requires exact PMXT execution prices")
    fit_frame = _window(source_frame, config.calibration_fit.start, config.calibration_fit.end)
    comparison_frame = _window(
        source_frame,
        config.matched_comparison.start,
        config.matched_comparison.end,
    )
    if fit_frame.is_empty() or comparison_frame.is_empty():
        raise RuntimeError("incumbent calibration windows must be non-empty")
    _validate_comparison_days(comparison_frame, expected=10)

    incumbent = load_frozen_asymmetric_incumbent(config.incumbent_model)
    _validate_incumbent_identity(config, incumbent)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = config.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    candidate_dir = run_dir / "candidates"
    candidate_dir.mkdir()

    root_contract = _root_contract(
        config,
        readiness_path=readiness_path,
        readiness=readiness,
        source_evidence=source_evidence,
    )
    root_contract_path = run_dir / "root-training-contract.json"
    write_json_atomic(root_contract_path, root_contract)
    write_json_atomic(
        run_dir / "consumed-cohort-registry.json",
        _consumed_cohort_registry(config),
    )
    write_json_atomic(
        run_dir / "incumbent-reproduction.json",
        _incumbent_reproduction(incumbent, comparison_frame),
    )
    eligible_keys_path = run_dir / "eligible-comparison-keys.parquet"
    comparison_frame.select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
    ).write_parquet(eligible_keys_path, compression="zstd", statistics=True)

    fits: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {INCUMBENT_CONTROL: incumbent.payload}
    fit_parent_probabilities = frozen_parent_probabilities(incumbent, fit_frame)
    parent_probabilities = frozen_parent_probabilities(incumbent, comparison_frame)
    prediction_paths: dict[str, Path] = {}
    payload_paths: dict[str, Path] = {}
    prediction_hashes: dict[str, str] = {}
    payload_hashes: dict[str, str] = {}
    estimator_fallbacks: dict[str, FittedEstimatorFallback] = {}
    estimator_prediction_paths: dict[str, Path] = {}
    estimator_prediction_hashes: dict[str, str] = {}
    estimator_model_paths: dict[str, Path] = {}
    estimator_model_hashes: dict[str, str] = {}
    estimator_evidence_paths: dict[str, Path] = {}
    estimator_evidence_hashes: dict[str, str] = {}
    estimator_support: dict[str, dict[str, Any]] = {}

    for arm_name in CALIBRATION_CHALLENGERS:
        fit = fit_incumbent_calibration_arm(
            arm_name,
            incumbent,
            fit_frame,
            config,
            parent_probabilities=fit_parent_probabilities,
        )
        candidate_payload = clone_incumbent_with_calibration(
            incumbent,
            fit,
            model_key=_candidate_model_key(arm_name),
        )
        validate_incumbent_calibration_parity(
            incumbent.payload,
            candidate_payload,
            config,
        )
        fits[arm_name] = fit
        payloads[arm_name] = candidate_payload

    for candidate_id, payload in payloads.items():
        candidate_path = candidate_dir / candidate_id
        candidate_path.mkdir()
        payload_path = candidate_path / "model.json"
        payload_path.write_bytes(canonical_json_bytes(payload))
        payload_paths[candidate_id] = payload_path
        payload_hashes[candidate_id] = file_sha256(payload_path)
        probability = score_incumbent_calibration_payload(
            incumbent,
            payload,
            comparison_frame,
            parent_probabilities=parent_probabilities,
        )
        predictions = asymmetric_probability_frame(
            comparison_frame,
            probability,
            model=candidate_id,
        ).with_columns(pl.lit(candidate_id).alias("candidate_id"))
        prediction_path = candidate_path / "comparison-predictions.parquet"
        _without_outcomes(predictions).write_parquet(
            prediction_path,
            compression="zstd",
            statistics=True,
        )
        prediction_paths[candidate_id] = prediction_path
        prediction_hashes[candidate_id] = file_sha256(prediction_path)

    source_parent_probabilities = frozen_parent_probabilities(incumbent, source_frame)
    source_incumbent_probabilities = score_incumbent_calibration_payload(
        incumbent,
        incumbent.payload,
        source_frame,
        parent_probabilities=source_parent_probabilities,
    )
    if not np.allclose(
        source_parent_probabilities,
        source_incumbent_probabilities,
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError(
            "frozen incumbent cells are not identity; fallback admission baseline changed"
        )
    if config.conditional_estimator.enabled:
        print(
            "incumbent-gen2: pre-fitting sealed E1/E2 estimator contingencies",
            flush=True,
        )
        estimator_fallbacks = fit_configured_core_oracle_estimator_fallbacks(
            source_frame,
            config,
            source_config,
            core_config,
            incumbent_probabilities=source_incumbent_probabilities,
        )
        for candidate_id in ESTIMATOR_FALLBACK_CANDIDATES:
            fitted = estimator_fallbacks[candidate_id]
            candidate_path = candidate_dir / candidate_id
            candidate_path.mkdir()
            model_path = candidate_path / "training-model.joblib"
            joblib.dump(fitted.bundle, model_path, compress=3)
            estimator_model_paths[candidate_id] = model_path
            estimator_model_hashes[candidate_id] = file_sha256(model_path)
            sealed_bundle = joblib.load(model_path)
            sealed_fitted = FittedEstimatorFallback(
                candidate_id=fitted.candidate_id,
                bundle=sealed_bundle,
                evidence=fitted.evidence,
                semantic_sha256=fitted.semantic_sha256,
            )

            evidence_path = candidate_path / "fit-evidence.json"
            write_json_atomic(evidence_path, fitted.evidence)
            estimator_evidence_paths[candidate_id] = evidence_path
            estimator_evidence_hashes[candidate_id] = file_sha256(evidence_path)
            estimator_support[candidate_id] = _estimator_support_payload(fitted)

            in_memory_artifact = score_core_oracle_estimator_fallback(
                fitted,
                source_frame,
                window=config.matched_comparison,
            )
            artifact = score_core_oracle_estimator_fallback(
                sealed_fitted,
                source_frame,
                window=config.matched_comparison,
            )
            if (
                artifact.key_sha256 != in_memory_artifact.key_sha256
                or artifact.probability_sha256 != in_memory_artifact.probability_sha256
                or artifact.artifact_sha256 != in_memory_artifact.artifact_sha256
            ):
                raise RuntimeError(f"{candidate_id} sealed joblib changed comparison probabilities")
            estimator_fallbacks[candidate_id] = sealed_fitted
            estimator_probability = _aligned_estimator_probability(
                artifact.predictions,
                comparison_frame,
            )
            predictions = asymmetric_probability_frame(
                comparison_frame,
                estimator_probability,
                model=candidate_id,
            ).with_columns(pl.lit(candidate_id).alias("candidate_id"))
            prediction_path = candidate_path / "comparison-predictions.parquet"
            _without_outcomes(predictions).write_parquet(
                prediction_path,
                compression="zstd",
                statistics=True,
            )
            estimator_prediction_paths[candidate_id] = prediction_path
            estimator_prediction_hashes[candidate_id] = file_sha256(prediction_path)
            write_json_atomic(
                candidate_path / "probability-artifact.json",
                {
                    "candidate_id": candidate_id,
                    "key_sha256": artifact.key_sha256,
                    "probability_sha256": artifact.probability_sha256,
                    "artifact_sha256": artifact.artifact_sha256,
                    "model_semantic_sha256": fitted.semantic_sha256,
                    "comparison_prediction_sha256": (estimator_prediction_hashes[candidate_id]),
                },
            )

    support_path = run_dir / "calibration-support.json"
    support = {candidate_id: _fit_support_payload(fit) for candidate_id, fit in fits.items()}
    write_json_atomic(support_path, support)
    estimator_support_path = run_dir / "estimator-calibration-support.json"
    write_json_atomic(estimator_support_path, estimator_support)
    registry = {
        "schema_version": INCUMBENT_CANDIDATE_REGISTRY_SCHEMA_VERSION,
        "process_id": config.process_id,
        "incumbent_control": INCUMBENT_CONTROL,
        "calibration_challengers": list(CALIBRATION_CHALLENGERS),
        "candidate_payload_sha256": payload_hashes,
        "comparison_prediction_sha256": prediction_hashes,
        "conditional_estimator_model_sha256": estimator_model_hashes,
        "conditional_estimator_fit_evidence_sha256": estimator_evidence_hashes,
        "conditional_estimator_prediction_sha256": estimator_prediction_hashes,
        "conditional_estimator_support_sha256": file_sha256(estimator_support_path),
        "calibration_support_sha256": file_sha256(support_path),
        "eligible_comparison_keys_sha256": file_sha256(eligible_keys_path),
        "conditional_estimator_candidates": _fallback_candidate_registry(config),
        "maximum_total_challengers": 5,
    }
    registry_path = run_dir / "candidate-registry.json"
    write_json_atomic(registry_path, registry)
    outcome_access_seal = {
        "schema_version": INCUMBENT_OUTCOME_ACCESS_SEAL_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "process_id": config.process_id,
        "root_training_contract_sha256": file_sha256(root_contract_path),
        "candidate_registry_sha256": file_sha256(registry_path),
        "readiness_manifest_sha256": file_sha256(readiness_path),
        "comparison_key_sha256": _frame_key_sha256(comparison_frame),
        "eligible_comparison_keys_sha256": file_sha256(eligible_keys_path),
        "candidate_payload_sha256": payload_hashes,
        "comparison_prediction_sha256": prediction_hashes,
        "conditional_estimator_model_sha256": estimator_model_hashes,
        "conditional_estimator_fit_evidence_sha256": estimator_evidence_hashes,
        "conditional_estimator_prediction_sha256": estimator_prediction_hashes,
        "conditional_estimator_support_sha256": file_sha256(estimator_support_path),
        "economics_opened": False,
    }
    outcome_access_seal_path = run_dir / "outcome-access-seal.json"
    write_json_atomic(outcome_access_seal_path, outcome_access_seal)

    matched_predictions = {
        candidate_id: _reload_predictions_with_outcomes(
            path,
            comparison_frame,
            expected_sha256=prediction_hashes[candidate_id],
        )
        for candidate_id, path in prediction_paths.items()
    }
    selection = select_incumbent_calibration_challenger(
        matched_predictions[INCUMBENT_CONTROL],
        {name: matched_predictions[name] for name in CALIBRATION_CHALLENGERS},
        support,
        resamples=config.bootstrap_resamples,
        seed=config.random_seed,
        incumbent_selected_bias=config.probability_gates.maximum_selected_opportunity_bias,
        noninferiority_margin=(config.probability_gates.maximum_paired_degradation_upper_95),
        minimum_noninferior_days=config.probability_gates.minimum_noninferior_days,
        maximum_cell_bias=config.probability_gates.maximum_cell_bias,
    )
    selection_path = run_dir / "probability-selection.json"
    write_json_atomic(selection_path, selection)
    selected_candidate = selection.get("selected_candidate_id")
    selected_export_payload_path: Path | None = None
    selected_export_payload_sha256: str | None = None
    selected_export_payload: dict[str, Any] | None = None
    final_support_path: Path | None = None
    if selected_candidate is not None:
        final_frame = _window(
            source_frame,
            config.final_refit.start,
            config.final_refit.end,
        )
        final_parent_probabilities = frozen_parent_probabilities(incumbent, final_frame)
        final_fit = fit_incumbent_calibration_arm(
            selected_candidate,
            incumbent,
            final_frame,
            config,
            window=config.final_refit,
            parent_probabilities=final_parent_probabilities,
        )
        final_payload = clone_incumbent_with_calibration(
            incumbent,
            final_fit,
            model_key=GEN2_MODEL_KEY,
        )
        validate_incumbent_calibration_parity(
            incumbent.payload,
            final_payload,
            config,
        )
        selected_export_payload = build_asymmetric_gen2_candidate_payload(
            source_model_payload=incumbent.payload,
            model_key=GEN2_MODEL_KEY,
            replacement_cells=_target_replacement_cells(final_payload),
        )
        selected_export_payload_path = run_dir / "selected-candidate-model.json"
        selected_export_payload_path.write_bytes(canonical_json_bytes(selected_export_payload))
        selected_export_payload_sha256 = file_sha256(selected_export_payload_path)
        final_support_path = run_dir / "final-calibration-support.json"
        write_json_atomic(
            final_support_path,
            {selected_candidate: _fit_support_payload(final_fit)},
        )
    selection_seal_path, selection_seal = _write_selection_seal(
        run_dir,
        config=config,
        incumbent=incumbent,
        selection=selection,
        selection_path=selection_path,
        outcome_access_seal_path=outcome_access_seal_path,
        development_support_path=support_path,
        final_support_path=final_support_path,
        prediction_paths=prediction_paths,
        prediction_hashes=prediction_hashes,
        payload_paths=payload_paths,
        payload_hashes=payload_hashes,
        selected_export_payload_path=selected_export_payload_path,
        selected_export_payload_sha256=selected_export_payload_sha256,
        readiness_path=readiness_path,
    )

    if selected_candidate is None:
        estimator_fallback = _run_conditional_estimator_if_eligible(
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            incumbent=incumbent,
            source_frame=source_frame,
            comparison_frame=comparison_frame,
            source_config=source_config,
            core_config=core_config,
            source_incumbent_probabilities=source_incumbent_probabilities,
            calibration_selection=selection,
            outcome_access_seal_path=outcome_access_seal_path,
            readiness_path=readiness_path,
            incumbent_prediction_path=prediction_paths[INCUMBENT_CONTROL],
            incumbent_prediction_sha256=prediction_hashes[INCUMBENT_CONTROL],
            estimator_fallbacks=estimator_fallbacks,
            estimator_prediction_paths=estimator_prediction_paths,
            estimator_prediction_hashes=estimator_prediction_hashes,
            estimator_model_paths=estimator_model_paths,
            estimator_model_hashes=estimator_model_hashes,
            estimator_evidence_paths=estimator_evidence_paths,
            estimator_evidence_hashes=estimator_evidence_hashes,
            estimator_support=estimator_support,
            estimator_support_path=estimator_support_path,
        )
        result = _blocked_result(
            run_id=run_id,
            config=config,
            selection=selection,
            selection_seal_path=selection_seal_path,
            selection_seal=selection_seal,
            status=str(estimator_fallback["status"]),
            estimator_fallback_status=estimator_fallback,
        )
        result.update(
            {
                "selected_candidate_id": estimator_fallback.get("selected_candidate_id"),
                "probability_qualified": bool(
                    estimator_fallback.get("probability_qualified", False)
                ),
                "economics_opened": bool(estimator_fallback.get("economics_opened", False)),
                "economics_qualified": bool(estimator_fallback.get("economics_qualified", False)),
                "paper_artifact": estimator_fallback.get("paper_artifact"),
                "calibration_selection_seal_sha256": file_sha256(selection_seal_path),
                "selection_seal_sha256": (
                    estimator_fallback.get("selection_seal_sha256")
                    or file_sha256(selection_seal_path)
                ),
                "economics_evidence_sha256": estimator_fallback.get("economics_evidence_sha256"),
                "forward_requirements": estimator_fallback.get(
                    "forward_requirements",
                    _inactive_forward_requirements("no_qualified_estimator_fallback_artifact"),
                ),
            }
        )
        _finalize_report(run_dir, result)
        return run_dir, result

    if selected_export_payload is None or selected_export_payload_sha256 is None:
        raise RuntimeError("selected calibration did not produce a sealed final-refit payload")
    selected_predictions = matched_predictions[selected_candidate]
    incumbent_ledger = _economic_ledger(
        matched_predictions[INCUMBENT_CONTROL],
        source_config,
    )
    _validate_incumbent_economic_reproduction(
        incumbent_ledger,
        config,
    )
    challenger_ledger = _economic_ledger(selected_predictions, source_config)
    eligible_markets = (
        comparison_frame.filter(pl.col("seconds_elapsed") <= config.policy.maximum_entry_second)
        .select("market_id", "window_start")
        .unique()
    )
    correction_ledger, _ = build_incumbent_correction_ledger(
        incumbent_ledger,
        challenger_ledger,
    )
    correction_path = run_dir / "incumbent-correction-ledger.parquet"
    correction_ledger.write_parquet(correction_path, compression="zstd", statistics=True)
    incumbent_ledger_path = run_dir / "incumbent-policy-ledger.parquet"
    challenger_ledger_path = run_dir / "challenger-policy-ledger.parquet"
    incumbent_ledger.write_parquet(
        incumbent_ledger_path,
        compression="zstd",
        statistics=True,
    )
    challenger_ledger.write_parquet(
        challenger_ledger_path,
        compression="zstd",
        statistics=True,
    )
    economics = paired_incumbent_economics(
        incumbent_ledger,
        challenger_ledger,
        eligible_markets=eligible_markets,
        resamples=config.bootstrap_resamples,
        seed=config.random_seed,
        incumbent_correctness_margin=config.correction_gates.minimum_correctness_margin,
    )
    economics = {
        **economics,
        "schema_version": INCUMBENT_ECONOMICS_SCHEMA_VERSION,
        "selected_candidate_id": selected_candidate,
        "selection_seal_sha256": file_sha256(selection_seal_path),
        "development_candidate_payload_sha256": payload_hashes[selected_candidate],
        "development_comparison_prediction_sha256": prediction_hashes[selected_candidate],
        "development_economics_scope": _window_payload(config.matched_comparison),
        "final_refit_candidate_payload_sha256": selected_export_payload_sha256,
        "final_refit_calibration_scope": _window_payload(config.final_refit),
        "ledger_sha256": {
            incumbent_ledger_path.name: file_sha256(incumbent_ledger_path),
            challenger_ledger_path.name: file_sha256(challenger_ledger_path),
            correction_path.name: file_sha256(correction_path),
        },
    }
    economics_path = run_dir / "economics-evidence.json"
    write_json_atomic(economics_path, economics)

    exported_model: Path | None = None
    if economics.get("status") == "qualified":
        authorization_path = _write_export_authorization(
            run_dir,
            selection_seal_path=selection_seal_path,
            economics_path=economics_path,
            selected_candidate=selected_candidate,
            selected_payload_sha256=selected_export_payload_sha256,
        )
        exported_model = export_asymmetric_gen2_paper_model(
            source_runtime_dir=incumbent.path.parent,
            source_process_id=config.process_id,
            source_process_metadata=frozen_core_oracle_process_metadata(),
            replacement_cells=_target_replacement_cells(selected_export_payload),
            selection_seal_path=selection_seal_path,
            expected_selection_seal_sha256=file_sha256(selection_seal_path),
            export_authorization_path=authorization_path,
            expected_export_authorization_sha256=file_sha256(authorization_path),
            economics_evidence_path=economics_path,
            expected_economics_evidence_sha256=file_sha256(economics_path),
            output_root=config.package_root / "runtime-models",
            model_key=GEN2_MODEL_KEY,
        )

    result = {
        "schema_version": INCUMBENT_BENCHMARK_SCHEMA_VERSION,
        "run_id": run_id,
        "status": (
            "qualified_paper_artifact_exported"
            if exported_model is not None
            else "incumbent_retained_economic_gates_failed"
        ),
        "process_id": config.process_id,
        "source_model_key": config.model_key,
        "selected_candidate_id": selected_candidate,
        "probability_qualified": True,
        "economics_opened": True,
        "economics_qualified": economics.get("status") == "qualified",
        "paper_artifact": str(exported_model) if exported_model else None,
        "source_process_changed": False,
        "live_capital_allowed": False,
        "selection_seal_sha256": file_sha256(selection_seal_path),
        "economics_evidence_sha256": file_sha256(economics_path),
        "forward_requirements": (
            _forward_requirements(selection_seal)
            if exported_model is not None
            else _inactive_forward_requirements("calibration_development_economic_gates_failed")
        ),
    }
    _finalize_report(run_dir, result)
    return run_dir, result


def _load_core_oracle_value_frame(
    config: Any,
    *,
    core_config: Any,
    force: bool,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    print("incumbent-gen2: validating cached Core and exact PMXT evidence", flush=True)
    extract_core_source(core_config, "pre_holdout", force=force)
    build_core_features(core_config, "pre_holdout", force=force)
    core = _load_asymmetric_core_grid(core_config, "pre_holdout", config)
    core_content_sha256 = _frame_content_digest(core)
    price_manifest = extract_asymmetric_price_evidence(
        config,
        scope="development",
        force=force,
    )
    prices = load_asymmetric_price_evidence(config, scope="development")
    inventory = oracle_source_inventory(
        config.oracle_source,
        core["window_start"].dt.date().unique().to_list(),
    )
    oracle = _load_or_build_oracle_core(
        core,
        config,
        destination=config.feature_cache / DEVELOPMENT_ORACLE_CACHE,
        source_inventory=inventory,
        core_content_sha256=core_content_sha256,
        expected_range_start=config.fit.start,
        expected_range_end=config.policy.end,
        force=force,
    )
    value = _project_candidate_source(
        attach_asymmetric_value_features(oracle, prices, config).filter(
            pl.col("early_oracle_eligible")
        ),
        ORACLE_MATCHED_CORE_PRICE_CONTROL,
        CORE_ORACLE_PRICE,
    ).sort("window_start", "market_id", "seconds_elapsed")
    return value, {
        "rows": value.height,
        "markets": value["market_id"].n_unique(),
        "utc_days": value["window_start"].dt.date().n_unique(),
        "core_content_sha256": core_content_sha256,
        "oracle_inventory_sha256": inventory["inventory_sha256"],
        "price_manifest_identity_sha256": price_manifest_identity_sha256(price_manifest),
        "proxy_prices_used": bool(price_manifest.get("proxy_prices_used", False)),
    }


def _window(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    return frame.filter((pl.col("window_start") >= start) & (pl.col("window_start") < end))


def _without_outcomes(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.drop("label_up")


def _reload_predictions_with_outcomes(
    path: Path,
    outcomes: pl.DataFrame,
    *,
    expected_sha256: str,
) -> pl.DataFrame:
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("sealed comparison predictions changed")
    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
    labels = outcomes.select(*keys, "label_up")
    sealed_predictions = pl.read_parquet(path)
    if (
        sealed_predictions.height != labels.height
        or sealed_predictions.select(*keys).is_duplicated().any()
        or labels.select(*keys).is_duplicated().any()
        or not sealed_predictions.select(*keys)
        .sort(keys)
        .equals(labels.select(*keys).sort(keys), null_equal=True)
    ):
        raise RuntimeError("sealed predictions do not match the exact outcome key grid")
    predictions = sealed_predictions.join(
        labels,
        on=keys,
        how="inner",
        validate="1:1",
    )
    if predictions.height != labels.height:
        raise RuntimeError("outcome join changed the sealed prediction grid")
    return predictions


def _aligned_estimator_probability(
    probability_artifact: pl.DataFrame,
    comparison_frame: pl.DataFrame,
) -> pl.Series:
    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
    required = {*keys, "probability_yes"}
    missing = sorted(required - set(probability_artifact.columns))
    if missing:
        raise ValueError(
            "conditional estimator probability artifact is missing: " + ", ".join(missing)
        )
    if probability_artifact.height != comparison_frame.height:
        raise RuntimeError("conditional estimator probability artifact has extra or missing rows")
    if probability_artifact.select(*keys).is_duplicated().any():
        raise RuntimeError("conditional estimator probability artifact contains duplicate keys")
    aligned = (
        comparison_frame.select(*keys)
        .with_row_index("__comparison_order")
        .join(
            probability_artifact.select(*keys, "probability_yes"),
            on=keys,
            how="inner",
            validate="1:1",
        )
        .sort("__comparison_order")
    )
    if aligned.height != comparison_frame.height:
        raise RuntimeError("conditional estimator predictions do not preserve the comparison grid")
    values = aligned["probability_yes"]
    if values.null_count() or not values.is_finite().all():
        raise RuntimeError("conditional estimator probabilities are invalid")
    return values


def _economic_ledger(predictions: pl.DataFrame, config: Any) -> pl.DataFrame:
    policy = next(policy for policy in config.policies if policy.selection_eligible)
    return policy_ledger(
        score_two_sided_value(predictions),
        policy,
        quantity=config.quantity,
        maximum_depth_participation=config.maximum_depth_participation,
    )


def _write_selection_seal(
    run_dir: Path,
    *,
    config: IncumbentCalibrationConfig,
    incumbent: FrozenAsymmetricRuntimeModel,
    selection: Mapping[str, Any],
    selection_path: Path,
    outcome_access_seal_path: Path,
    development_support_path: Path,
    final_support_path: Path | None,
    prediction_paths: Mapping[str, Path],
    prediction_hashes: Mapping[str, str],
    payload_paths: Mapping[str, Path],
    payload_hashes: Mapping[str, str],
    selected_export_payload_path: Path | None,
    selected_export_payload_sha256: str | None,
    readiness_path: Path,
) -> tuple[Path, dict[str, Any]]:
    selected = selection.get("selected_candidate_id")
    selected_prediction_sha256 = prediction_hashes.get(selected) if selected else None
    outcome_access_seal = json.loads(outcome_access_seal_path.read_text())
    if (
        outcome_access_seal.get("economics_opened") is not False
        or outcome_access_seal.get("candidate_payload_sha256") != dict(payload_hashes)
        or outcome_access_seal.get("comparison_prediction_sha256") != dict(prediction_hashes)
    ):
        raise RuntimeError("calibration outcome-access seal does not match candidates")
    lineage = {
        "schema_version": INCUMBENT_CALIBRATION_SELECTION_LINEAGE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_process_id": config.process_id,
        "outcome_access_seal_sha256": file_sha256(outcome_access_seal_path),
        "development_calibration_support_sha256": file_sha256(development_support_path),
        "final_calibration_support_sha256": (
            file_sha256(final_support_path) if final_support_path is not None else None
        ),
        "candidate_payload_sha256": dict(payload_hashes),
        "comparison_prediction_sha256": dict(prediction_hashes),
        "selected_final_payload_sha256": selected_export_payload_sha256,
    }
    lineage_path = run_dir / "calibration-selection-lineage.json"
    write_json_atomic(lineage_path, lineage)
    seal = {
        "schema_version": GEN2_SELECTION_SEAL_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_process_id": config.process_id,
        "source_model_key": config.model_key,
        "source_model_sha256": incumbent.model_sha256,
        "selected_candidate_id": selected,
        "selected_candidate_payload_sha256": selected_export_payload_sha256,
        "probability_selection_sha256": file_sha256(selection_path),
        "probability_predictions_sha256": selected_prediction_sha256,
        "calibration_fit_sha256": file_sha256(lineage_path),
        "readiness_manifest_sha256": file_sha256(readiness_path),
        "economics_opened": False,
    }
    artifact_manifest = {
        "selection": {"path": selection_path.name, "sha256": file_sha256(selection_path)},
        "calibration_selection_lineage": {
            "path": lineage_path.name,
            "sha256": file_sha256(lineage_path),
        },
        "development_calibration_support": {
            "path": development_support_path.name,
            "sha256": file_sha256(development_support_path),
        },
        "final_calibration_support": (
            {
                "path": final_support_path.name,
                "sha256": file_sha256(final_support_path),
            }
            if final_support_path is not None
            else None
        ),
        "candidate_payloads": {
            name: {
                "path": str(payload_paths[name].relative_to(run_dir)),
                "sha256": payload_hashes[name],
            }
            for name in sorted(payload_paths)
        },
        "candidate_predictions": {
            name: {
                "path": str(prediction_paths[name].relative_to(run_dir)),
                "sha256": prediction_hashes[name],
            }
            for name in sorted(prediction_paths)
        },
        "selected_export_payload": (
            {
                "path": str(selected_export_payload_path.relative_to(run_dir)),
                "sha256": selected_export_payload_sha256,
            }
            if selected_export_payload_path is not None
            else None
        ),
    }
    write_json_atomic(run_dir / "selection-artifact-manifest.json", artifact_manifest)
    path = run_dir / "probability-selection-seal.json"
    write_json_atomic(path, seal)
    reloaded = json.loads(path.read_text())
    if reloaded != seal:
        raise RuntimeError("probability selection seal failed disk round trip")
    for name, digest in payload_hashes.items():
        if file_sha256(payload_paths[name]) != digest:
            raise RuntimeError("sealed candidate payload changed")
    for name, digest in prediction_hashes.items():
        if file_sha256(prediction_paths[name]) != digest:
            raise RuntimeError("sealed candidate predictions changed")
    if (
        file_sha256(outcome_access_seal_path) != lineage["outcome_access_seal_sha256"]
        or file_sha256(development_support_path)
        != lineage["development_calibration_support_sha256"]
        or (
            final_support_path is not None
            and file_sha256(final_support_path) != lineage["final_calibration_support_sha256"]
        )
    ):
        raise RuntimeError("calibration selection lineage changed")
    return path, reloaded


def _write_export_authorization(
    run_dir: Path,
    *,
    selection_seal_path: Path,
    economics_path: Path,
    selected_candidate: str,
    selected_payload_sha256: str,
) -> Path:
    payload = {
        "schema_version": INCUMBENT_EXPORT_AUTHORIZATION_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "selection_seal_sha256": file_sha256(selection_seal_path),
        "economics_evidence_sha256": file_sha256(economics_path),
        "selected_candidate_id": selected_candidate,
        "selected_candidate_payload_sha256": selected_payload_sha256,
        "probability_qualified": True,
        "economics_qualified": True,
        "deployment_scope": "paper_only",
        "live_capital_allowed": False,
        "production_qualified": False,
    }
    path = run_dir / "export-authorization.json"
    write_json_atomic(path, payload)
    return path


def _target_replacement_cells(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(cell)
        for cell in payload["asymmetric_value_calibration"]["side_price_cells"]
        if float(cell["minimum_price"]) == 0.20
        and float(cell["maximum_price"]) == 0.30
        and int(cell["start_seconds"]) < 60
    ]


def _root_contract(
    config: IncumbentCalibrationConfig,
    *,
    readiness_path: Path,
    readiness: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": INCUMBENT_ROOT_CONTRACT_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "process_id": config.process_id,
        "source_model_key": config.model_key,
        "source_model_sha256": config.model_sha256,
        "configuration_sha256": file_sha256(config.source_path),
        "implementation_sha256": _implementation_sha256(),
        "evidence_scope": "consumed_cross_day_development",
        "calibration_fit": _window_payload(config.calibration_fit),
        "matched_comparison": _window_payload(config.matched_comparison),
        "final_refit": _window_payload(config.final_refit),
        "policy": _dataclass_payload(config.policy),
        "calibration_challengers": list(CALIBRATION_CHALLENGERS),
        "conditional_estimator_fallback": _fallback_candidate_registry(config),
        "candidate_budget": 5,
        "no_gate_relaxation": True,
        "no_pnl_based_iteration": True,
        "source_readiness_sha256": file_sha256(readiness_path),
        "source_readiness_identity_sha256": readiness["readiness_identity_sha256"],
        "source_evidence": dict(source_evidence),
        "dependency_versions": _dependency_versions(),
    }


def _consumed_cohort_registry(config: IncumbentCalibrationConfig) -> dict[str, Any]:
    return {
        "schema_version": "btc-asymmetric-consumed-cohort-registry-v1",
        "process_id": config.process_id,
        "calibration_fit": {
            **_window_payload(config.calibration_fit),
            "purpose": "calibration_fit",
            "independent_proof": False,
        },
        "matched_comparison": {
            **_window_payload(config.matched_comparison),
            "purpose": "one_shot_probability_then_economics",
            "independent_proof": False,
        },
        "reuse_rule": (
            "failed evidence may enter later training but can never be represented "
            "again as unseen proof"
        ),
    }


def _incumbent_reproduction(
    incumbent: FrozenAsymmetricRuntimeModel,
    comparison_frame: pl.DataFrame,
) -> dict[str, Any]:
    calibration = incumbent.payload["asymmetric_value_calibration"]
    fitted = sum(bool(cell.get("fitted")) for cell in calibration["side_price_cells"])
    return {
        "process_scope": "81f82de7-002b-4ac7-814b-236c6742d81c",
        "model_key": incumbent.payload["model_key"],
        "model_sha256": incumbent.model_sha256,
        "manifest_sha256": incumbent.manifest_sha256,
        "feature_contract": incumbent.feature_contract,
        "feature_count": len(incumbent.feature_names),
        "parent_time_calibrators": len(calibration["time_bands"]),
        "side_price_time_cells": len(calibration["side_price_cells"]),
        "fitted_side_price_time_cells": fitted,
        "comparison_rows": comparison_frame.height,
        "comparison_markets": comparison_frame["market_id"].n_unique(),
        "comparison_key_sha256": _frame_key_sha256(comparison_frame),
    }


def _validate_incumbent_identity(
    config: IncumbentCalibrationConfig,
    incumbent: FrozenAsymmetricRuntimeModel,
) -> None:
    if (
        incumbent.model_sha256 != config.model_sha256
        or incumbent.payload["model_key"] != config.model_key
        or len(incumbent.feature_names) != config.feature_count
        or incumbent.payload["features"]["schema_sha256"] != config.feature_schema_sha256
    ):
        raise RuntimeError("configured incumbent identity does not match the frozen artifact")


def _validate_comparison_days(frame: pl.DataFrame, *, expected: int) -> None:
    observed = frame["window_start"].dt.date().n_unique()
    if observed != expected:
        raise RuntimeError(
            f"matched incumbent comparison requires exactly {expected} UTC days; got {observed}"
        )


def _validate_runner_gate_contract(config: IncumbentCalibrationConfig) -> None:
    probability = config.probability_gates
    correction = config.correction_gates
    economics = config.economic_gates
    if (
        probability.maximum_paired_degradation_upper_95 != 0.005
        or probability.maximum_selected_opportunity_bias != 0.02694
        or probability.maximum_cell_bias != 0.05
        or probability.minimum_noninferior_days != 8
        or probability.required_comparison_days != 10
        or not probability.require_brier_noninferiority
        or not probability.require_log_loss_noninferiority
        or not probability.require_one_proper_score_improvement
        or not probability.require_all_target_cells_fitted
        or not probability.require_both_outcomes_per_cell
    ):
        raise ValueError("incumbent probability qualification gates changed")
    if (
        correction.minimum_net_corrected_decisions != 2
        or correction.minimum_improvement_days != 3
        or correction.minimum_correctness_margin != 0.026231
        or correction.minimum_correctness_margin_delta != 0.0
        or not correction.require_positive_candidate_only_stressed_pnl
    ):
        raise ValueError("incumbent correction qualification gates changed")
    if (
        economics.minimum_incumbent_frequency_fraction != 0.80
        or economics.minimum_yes_entries != 20
        or economics.minimum_no_entries != 20
        or economics.minimum_stressed_expectancy_per_trade != 0.0
        or economics.minimum_profit_factor != 1.05
        or economics.maximum_mean_share_price != 0.2466
        or economics.maximum_loss_recovery_burden != 0.40
        or economics.maximum_average_loss != 1.390
        or economics.maximum_single_loss != 1.674
        or economics.maximum_drawdown != 19.18
        or economics.maximum_primary_metric_regression_fraction != 0.10
        or economics.minimum_paired_profit_per_market_lower_95 != 0.0
    ):
        raise ValueError("incumbent economic qualification gates changed")


def _validate_incumbent_economic_reproduction(
    ledger: pl.DataFrame,
    config: IncumbentCalibrationConfig,
) -> None:
    metrics = ledger_metrics(ledger)
    baseline = config.historical_baseline
    wins = int(ledger["won"].sum())
    checks = {
        "trades": (metrics["trades"], baseline.trades, 0.0),
        "wins": (wins, baseline.wins, 0.0),
        "losses": (metrics["trades"] - wins, baseline.losses, 0.0),
        "net_profit": (metrics["net_profit"], baseline.net_profit, 0.02),
        "expectancy_per_trade": (
            metrics["net_expectancy_per_trade"],
            baseline.expectancy_per_trade,
            0.001,
        ),
        "profit_factor": (metrics["profit_factor"], baseline.profit_factor, 0.002),
        "maximum_drawdown": (
            abs(metrics["maximum_drawdown"]),
            baseline.maximum_drawdown,
            0.02,
        ),
        "loss_recovery_burden": (
            metrics["loss_recovery_wins"],
            baseline.loss_recovery_burden,
            0.002,
        ),
    }
    failed = [
        name
        for name, (observed, expected, tolerance) in checks.items()
        if observed is None or abs(float(observed) - float(expected)) > tolerance
    ]
    if failed:
        raise RuntimeError("frozen incumbent economic reproduction changed: " + ", ".join(failed))


def _fit_support_payload(fit: Any) -> dict[str, Any]:
    if not hasattr(fit, "cells"):
        raise TypeError("calibration fit does not expose target-cell evidence")
    cells: dict[str, Any] = {}
    for cell in fit.cells:
        name = f"{cell.side}_{cell.start_second}_{cell.end_second_exclusive}"
        cells[name] = {
            "fitted": bool(cell.fitted),
            "converged": bool(cell.converged),
            "fallback": bool(cell.fallback is not None),
            "rows": int(cell.rows),
            "markets": int(cell.markets),
            "utc_days": int(cell.utc_days),
            "negative_outcomes": int(cell.negatives),
            "positive_outcomes": int(cell.positives),
            "slope": float(cell.slope),
            "intercept": float(cell.intercept),
            "iterations": int(cell.iterations),
            "objective": cell.objective,
            "weighted_log_loss": cell.weighted_log_loss,
        }
    return {
        "arm_name": fit.arm_name,
        "weighting": fit.weighting,
        "identity_l2": fit.identity_l2,
        "converged": fit.converged,
        "iterations": fit.iterations,
        "objective": fit.objective,
        "payload_sha256": fit.payload_sha256,
        "cells": cells,
    }


def _estimator_support_payload(fitted: FittedEstimatorFallback) -> dict[str, Any]:
    calibration = fitted.evidence.get("side_price_time_calibration", {})
    target = calibration.get("target_contract", {})
    raw_cells = target.get("cells", [])
    if target.get("qualified") is not True or len(raw_cells) != 8:
        raise RuntimeError(f"{fitted.candidate_id} does not have eight qualified calibration cells")
    cells: dict[str, Any] = {}
    for cell in raw_cells:
        name = f"{cell['side']}_{int(cell['start_second'])}_{int(cell['end_second_exclusive'])}"
        cells[name] = {
            "fitted": bool(cell["fitted"]),
            "converged": bool(cell["passed"]),
            "fallback": bool(cell.get("fallback") is not None),
            "markets": int(cell["markets"]),
            "utc_days": int(cell["utc_days"]),
            "negative_outcomes": int(cell["negatives"]),
            "positive_outcomes": int(cell["positives"]),
            "failure_reasons": list(cell.get("failure_reasons", [])),
        }
    expected = {
        f"{side}_{lower}_{upper}"
        for side in ("YES", "NO")
        for lower, upper in ((1, 15), (15, 30), (30, 45), (45, 60))
    }
    if set(cells) != expected:
        raise RuntimeError(f"{fitted.candidate_id} calibration cell boundaries changed")
    return {
        "candidate_id": fitted.candidate_id,
        "model_semantic_sha256": fitted.semantic_sha256,
        "qualified": True,
        "cells": cells,
    }


def _validate_sealed_estimator_refit(
    *,
    development_fit: FittedEstimatorFallback,
    final_fit: FittedEstimatorFallback,
    sealed_final_bundle: Any,
    comparison_frame: pl.DataFrame,
) -> None:
    reuse = final_fit.evidence.get("selected_estimator_reuse", {})
    required_true = (
        "estimator_identity_unchanged",
        "only_calibration_window_expanded",
    )
    if (
        final_fit.candidate_id != development_fit.candidate_id
        or final_fit.evidence.get("semantic_sha256") != final_fit.semantic_sha256
        or any(reuse.get(name) is not True for name in required_true)
        or reuse.get("estimator_refit_performed") is not False
        or reuse.get("training_weights_recomputed") is not False
        or reuse.get("development_bundle_semantic_sha256") != development_fit.semantic_sha256
    ):
        raise RuntimeError("selected estimator refit reuse evidence is invalid")
    development_raw = development_fit.bundle.model.raw_logit(comparison_frame)
    final_raw = final_fit.bundle.model.raw_logit(comparison_frame)
    sealed_raw = sealed_final_bundle.model.raw_logit(comparison_frame)
    final_probability = final_fit.bundle.probability(comparison_frame)
    sealed_probability = sealed_final_bundle.probability(comparison_frame)
    if (
        not np.array_equal(development_raw, final_raw)
        or not np.array_equal(final_raw, sealed_raw)
        or not np.array_equal(final_probability, sealed_probability)
    ):
        raise RuntimeError("sealed estimator refit changed model behavior")


def _fallback_candidate_registry(config: IncumbentCalibrationConfig) -> dict[str, Any]:
    contingency = getattr(config, "conditional_estimator", None)
    if contingency is None:
        return {"enabled": False, "candidates": []}
    return _dataclass_payload(contingency)


def _run_conditional_estimator_if_eligible(
    *,
    config: IncumbentCalibrationConfig,
    run_id: str,
    run_dir: Path,
    incumbent: FrozenAsymmetricRuntimeModel,
    source_frame: pl.DataFrame,
    comparison_frame: pl.DataFrame,
    source_config: Any,
    core_config: Any,
    source_incumbent_probabilities: Any,
    calibration_selection: Mapping[str, Any],
    outcome_access_seal_path: Path,
    readiness_path: Path,
    incumbent_prediction_path: Path,
    incumbent_prediction_sha256: str,
    estimator_fallbacks: Mapping[str, FittedEstimatorFallback],
    estimator_prediction_paths: Mapping[str, Path],
    estimator_prediction_hashes: Mapping[str, str],
    estimator_model_paths: Mapping[str, Path],
    estimator_model_hashes: Mapping[str, str],
    estimator_evidence_paths: Mapping[str, Path],
    estimator_evidence_hashes: Mapping[str, str],
    estimator_support: Mapping[str, Mapping[str, Any]],
    estimator_support_path: Path,
) -> dict[str, Any]:
    """Execute the predeclared estimator fallback after a pure quality failure."""

    trigger = config.conditional_estimator.trigger
    trigger_path = run_dir / "estimator-fallback-trigger.json"
    trigger_evidence = {
        "schema_version": "btc-asymmetric-incumbent-estimator-trigger-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "process_id": config.process_id,
        "trigger": trigger,
        "calibration_selection_sha256": _canonical_payload_sha256(calibration_selection),
        "calibration_selected_candidate_id": calibration_selection.get("selected_candidate_id"),
        "calibration_support_passed": _calibration_support_passed(calibration_selection),
        "economics_used_for_trigger": False,
        "enabled": config.conditional_estimator.enabled,
    }
    write_json_atomic(trigger_path, trigger_evidence)
    if not config.conditional_estimator.enabled:
        return _estimator_fallback_status(
            "incumbent_retained_estimator_fallback_disabled",
            trigger=trigger,
        )
    if calibration_selection.get("selected_candidate_id") is not None:
        raise RuntimeError("estimator fallback cannot run after a calibration winner")
    if calibration_selection.get("economics_used") is not False:
        raise RuntimeError("estimator fallback trigger cannot use economics")
    if not _calibration_support_passed(calibration_selection):
        return _estimator_fallback_status(
            "incumbent_retained_calibration_support_failure",
            trigger=trigger,
        )
    expected_ids = tuple(ESTIMATOR_FALLBACK_CANDIDATES)
    mappings = (
        estimator_fallbacks,
        estimator_prediction_paths,
        estimator_prediction_hashes,
        estimator_model_paths,
        estimator_model_hashes,
        estimator_evidence_paths,
        estimator_evidence_hashes,
        estimator_support,
    )
    if any(tuple(mapping) != expected_ids for mapping in mappings):
        raise RuntimeError("conditional estimator artifact registry changed")
    _verify_estimator_artifacts(
        outcome_access_seal_path=outcome_access_seal_path,
        prediction_paths=estimator_prediction_paths,
        prediction_hashes=estimator_prediction_hashes,
        model_paths=estimator_model_paths,
        model_hashes=estimator_model_hashes,
        evidence_paths=estimator_evidence_paths,
        evidence_hashes=estimator_evidence_hashes,
        support_path=estimator_support_path,
        expected_support=estimator_support,
    )

    incumbent_predictions = _reload_predictions_with_outcomes(
        incumbent_prediction_path,
        comparison_frame,
        expected_sha256=incumbent_prediction_sha256,
    )
    matched_predictions = {
        candidate_id: _reload_predictions_with_outcomes(
            estimator_prediction_paths[candidate_id],
            comparison_frame,
            expected_sha256=estimator_prediction_hashes[candidate_id],
        )
        for candidate_id in expected_ids
    }
    selection = select_incumbent_calibration_challenger(
        incumbent_predictions,
        matched_predictions,
        estimator_support,
        resamples=config.bootstrap_resamples,
        seed=config.random_seed,
        incumbent_selected_bias=(config.probability_gates.maximum_selected_opportunity_bias),
        noninferiority_margin=(config.probability_gates.maximum_paired_degradation_upper_95),
        minimum_noninferior_days=config.probability_gates.minimum_noninferior_days,
        maximum_cell_bias=config.probability_gates.maximum_cell_bias,
    )
    selection_path = run_dir / "estimator-fallback-probability-selection.json"
    write_json_atomic(selection_path, selection)
    selected_candidate = selection.get("selected_candidate_id")
    final_model_path: Path | None = None
    final_model_sha256: str | None = None
    final_model_semantic_sha256: str | None = None
    final_evidence_path: Path | None = None
    final_support_path: Path | None = None
    if selected_candidate is not None:
        final_fit = refit_selected_core_oracle_estimator_fallback(
            source_frame,
            config,
            source_config,
            core_config,
            candidate_id=selected_candidate,
            incumbent_probabilities=source_incumbent_probabilities,
            selected_fit=estimator_fallbacks[selected_candidate],
        )
        final_model_path = run_dir / "selected-estimator-refit.joblib"
        joblib.dump(final_fit.bundle, final_model_path, compress=3)
        final_model_sha256 = file_sha256(final_model_path)
        final_model_semantic_sha256 = final_fit.semantic_sha256
        sealed_final_bundle = joblib.load(final_model_path)
        _validate_sealed_estimator_refit(
            development_fit=estimator_fallbacks[selected_candidate],
            final_fit=final_fit,
            sealed_final_bundle=sealed_final_bundle,
            comparison_frame=comparison_frame,
        )
        final_evidence_path = run_dir / "selected-estimator-refit-evidence.json"
        write_json_atomic(final_evidence_path, final_fit.evidence)
        final_support_path = run_dir / "selected-estimator-calibration-support.json"
        write_json_atomic(
            final_support_path,
            {selected_candidate: _estimator_support_payload(final_fit)},
        )
    selection_seal_path = _write_estimator_selection_seal(
        run_dir,
        config=config,
        selection_path=selection_path,
        selection=selection,
        trigger_path=trigger_path,
        outcome_access_seal_path=outcome_access_seal_path,
        readiness_path=readiness_path,
        prediction_paths=estimator_prediction_paths,
        prediction_hashes=estimator_prediction_hashes,
        model_paths=estimator_model_paths,
        model_hashes=estimator_model_hashes,
        evidence_paths=estimator_evidence_paths,
        evidence_hashes=estimator_evidence_hashes,
        final_model_path=final_model_path,
        final_model_sha256=final_model_sha256,
        final_model_semantic_sha256=final_model_semantic_sha256,
        final_evidence_path=final_evidence_path,
        final_support_path=final_support_path,
        estimator_support_path=estimator_support_path,
        estimator_support=estimator_support,
    )
    if selected_candidate is None:
        return _estimator_fallback_status(
            "incumbent_retained_no_estimator_quality_winner",
            trigger=trigger,
            selection_seal_path=selection_seal_path,
        )
    if final_model_path is None or final_model_sha256 is None:
        raise RuntimeError("selected estimator fallback did not produce a final model")

    incumbent_ledger = _economic_ledger(incumbent_predictions, source_config)
    _validate_incumbent_economic_reproduction(incumbent_ledger, config)
    challenger_ledger = _economic_ledger(
        matched_predictions[selected_candidate],
        source_config,
    )
    eligible_markets = (
        comparison_frame.filter(pl.col("seconds_elapsed") <= config.policy.maximum_entry_second)
        .select("market_id", "window_start")
        .unique()
    )
    correction_ledger, _ = build_incumbent_correction_ledger(
        incumbent_ledger,
        challenger_ledger,
    )
    incumbent_ledger_path = run_dir / "estimator-fallback-incumbent-ledger.parquet"
    challenger_ledger_path = run_dir / "estimator-fallback-challenger-ledger.parquet"
    correction_path = run_dir / "estimator-fallback-correction-ledger.parquet"
    incumbent_ledger.write_parquet(
        incumbent_ledger_path,
        compression="zstd",
        statistics=True,
    )
    challenger_ledger.write_parquet(
        challenger_ledger_path,
        compression="zstd",
        statistics=True,
    )
    correction_ledger.write_parquet(
        correction_path,
        compression="zstd",
        statistics=True,
    )
    economics = paired_incumbent_economics(
        incumbent_ledger,
        challenger_ledger,
        eligible_markets=eligible_markets,
        resamples=config.bootstrap_resamples,
        seed=config.random_seed,
        incumbent_correctness_margin=config.correction_gates.minimum_correctness_margin,
    )
    economics = {
        **economics,
        "schema_version": INCUMBENT_ESTIMATOR_ECONOMICS_SCHEMA_VERSION,
        "selected_candidate_id": selected_candidate,
        "selection_seal_sha256": file_sha256(selection_seal_path),
        "development_model_sha256": estimator_model_hashes[selected_candidate],
        "development_model_semantic_sha256": estimator_fallbacks[
            selected_candidate
        ].semantic_sha256,
        "development_comparison_prediction_sha256": (
            estimator_prediction_hashes[selected_candidate]
        ),
        "development_economics_scope": _window_payload(config.matched_comparison),
        "final_model_sha256": final_model_sha256,
        "final_model_semantic_sha256": final_model_semantic_sha256,
        "final_paper_refit_scope": {
            "estimator_fit": _window_payload(source_config.fit),
            "calibration": _window_payload(config.final_refit),
        },
        "ledger_sha256": {
            incumbent_ledger_path.name: file_sha256(incumbent_ledger_path),
            challenger_ledger_path.name: file_sha256(challenger_ledger_path),
            correction_path.name: file_sha256(correction_path),
        },
    }
    economics_path = run_dir / "estimator-fallback-economics-evidence.json"
    write_json_atomic(economics_path, economics)

    exported_model: Path | None = None
    if economics.get("status") == "qualified":
        authorization_path = _write_estimator_export_authorization(
            run_dir,
            run_id=run_id,
            config=config,
            selected_candidate=selected_candidate,
            selection_seal_path=selection_seal_path,
            economics_path=economics_path,
            final_model_path=final_model_path,
        )
        exported_model = export_asymmetric_value_runtime_model(
            model_path=final_model_path,
            output_root=config.package_root / "runtime-models",
            model_key=ESTIMATOR_MODEL_KEYS[selected_candidate],
            source_run_id=run_id,
            source_benchmark_sha256=file_sha256(authorization_path),
        )
    return _estimator_fallback_status(
        (
            "qualified_estimator_paper_artifact_exported"
            if exported_model is not None
            else "incumbent_retained_estimator_economic_gates_failed"
        ),
        trigger=trigger,
        selected_candidate_id=selected_candidate,
        probability_qualified=True,
        economics_opened=True,
        economics_qualified=economics.get("status") == "qualified",
        paper_artifact=str(exported_model) if exported_model is not None else None,
        selection_seal_path=selection_seal_path,
        economics_path=economics_path,
    )


def _calibration_support_passed(selection: Mapping[str, Any]) -> bool:
    records = selection.get("candidate_records", [])
    if len(records) != len(CALIBRATION_CHALLENGERS):
        return False
    by_name = {record.get("candidate_id"): record for record in records}
    if set(by_name) != set(CALIBRATION_CHALLENGERS):
        return False
    return all(
        record.get("calibration_support", {}).get("passed") is True for record in by_name.values()
    )


def _verify_estimator_artifacts(
    *,
    outcome_access_seal_path: Path,
    prediction_paths: Mapping[str, Path],
    prediction_hashes: Mapping[str, str],
    model_paths: Mapping[str, Path],
    model_hashes: Mapping[str, str],
    evidence_paths: Mapping[str, Path],
    evidence_hashes: Mapping[str, str],
    support_path: Path,
    expected_support: Mapping[str, Mapping[str, Any]],
) -> None:
    outcome_seal = json.loads(outcome_access_seal_path.read_text())
    if outcome_seal.get("economics_opened") is not False:
        raise RuntimeError("conditional estimator outcome seal already opened economics")
    expected = {
        "conditional_estimator_prediction_sha256": dict(prediction_hashes),
        "conditional_estimator_model_sha256": dict(model_hashes),
        "conditional_estimator_fit_evidence_sha256": dict(evidence_hashes),
    }
    for key, values in expected.items():
        if outcome_seal.get(key) != values:
            raise RuntimeError(f"conditional estimator outcome seal changed: {key}")
    for paths, hashes in (
        (prediction_paths, prediction_hashes),
        (model_paths, model_hashes),
        (evidence_paths, evidence_hashes),
    ):
        for candidate_id, expected_hash in hashes.items():
            if file_sha256(paths[candidate_id]) != expected_hash:
                raise RuntimeError(f"sealed conditional estimator artifact changed: {candidate_id}")
    expected_support_sha256 = outcome_seal.get("conditional_estimator_support_sha256")
    if (
        not isinstance(expected_support_sha256, str)
        or file_sha256(support_path) != expected_support_sha256
        or json.loads(support_path.read_text()) != dict(expected_support)
    ):
        raise RuntimeError("sealed conditional estimator support changed")


def _write_estimator_selection_seal(
    run_dir: Path,
    *,
    config: IncumbentCalibrationConfig,
    selection_path: Path,
    selection: Mapping[str, Any],
    trigger_path: Path,
    outcome_access_seal_path: Path,
    readiness_path: Path,
    prediction_paths: Mapping[str, Path],
    prediction_hashes: Mapping[str, str],
    model_paths: Mapping[str, Path],
    model_hashes: Mapping[str, str],
    evidence_paths: Mapping[str, Path],
    evidence_hashes: Mapping[str, str],
    final_model_path: Path | None,
    final_model_sha256: str | None,
    final_model_semantic_sha256: str | None,
    final_evidence_path: Path | None,
    final_support_path: Path | None,
    estimator_support_path: Path,
    estimator_support: Mapping[str, Mapping[str, Any]],
) -> Path:
    selected = selection.get("selected_candidate_id")
    if (selected is None) != (final_model_path is None):
        raise RuntimeError("estimator selection and final model presence disagree")
    seal = {
        "schema_version": INCUMBENT_ESTIMATOR_SELECTION_SEAL_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_process_id": config.process_id,
        "source_model_key": config.model_key,
        "source_model_sha256": config.model_sha256,
        "trigger_sha256": file_sha256(trigger_path),
        "outcome_access_seal_sha256": file_sha256(outcome_access_seal_path),
        "readiness_manifest_sha256": file_sha256(readiness_path),
        "probability_selection_sha256": file_sha256(selection_path),
        "selected_candidate_id": selected,
        "selected_comparison_prediction_sha256": (
            prediction_hashes[selected] if selected is not None else None
        ),
        "selected_final_model_sha256": final_model_sha256,
        "selected_final_model_semantic_sha256": final_model_semantic_sha256,
        "selected_final_fit_evidence_sha256": (
            file_sha256(final_evidence_path) if final_evidence_path is not None else None
        ),
        "selected_final_calibration_support_sha256": (
            file_sha256(final_support_path) if final_support_path is not None else None
        ),
        "candidate_model_sha256": dict(model_hashes),
        "candidate_fit_evidence_sha256": dict(evidence_hashes),
        "candidate_prediction_sha256": dict(prediction_hashes),
        "economics_opened": False,
    }
    manifest = {
        "selection": {
            "path": selection_path.name,
            "sha256": file_sha256(selection_path),
        },
        "trigger": {
            "path": trigger_path.name,
            "sha256": file_sha256(trigger_path),
        },
        "candidate_models": {
            candidate_id: {
                "path": str(model_paths[candidate_id].relative_to(run_dir)),
                "sha256": model_hashes[candidate_id],
            }
            for candidate_id in sorted(model_paths)
        },
        "candidate_fit_evidence": {
            candidate_id: {
                "path": str(evidence_paths[candidate_id].relative_to(run_dir)),
                "sha256": evidence_hashes[candidate_id],
            }
            for candidate_id in sorted(evidence_paths)
        },
        "candidate_predictions": {
            candidate_id: {
                "path": str(prediction_paths[candidate_id].relative_to(run_dir)),
                "sha256": prediction_hashes[candidate_id],
            }
            for candidate_id in sorted(prediction_paths)
        },
        "selected_final_model": (
            {
                "path": final_model_path.name,
                "sha256": final_model_sha256,
            }
            if final_model_path is not None
            else None
        ),
        "selected_final_fit_evidence": (
            {
                "path": final_evidence_path.name,
                "sha256": file_sha256(final_evidence_path),
            }
            if final_evidence_path is not None
            else None
        ),
        "selected_final_calibration_support": (
            {
                "path": final_support_path.name,
                "sha256": file_sha256(final_support_path),
            }
            if final_support_path is not None
            else None
        ),
    }
    write_json_atomic(
        run_dir / "estimator-fallback-selection-artifact-manifest.json",
        manifest,
    )
    path = run_dir / "estimator-fallback-probability-selection-seal.json"
    write_json_atomic(path, seal)
    if json.loads(path.read_text()) != seal:
        raise RuntimeError("estimator probability selection seal failed disk round trip")
    _verify_estimator_artifacts(
        outcome_access_seal_path=outcome_access_seal_path,
        prediction_paths=prediction_paths,
        prediction_hashes=prediction_hashes,
        model_paths=model_paths,
        model_hashes=model_hashes,
        evidence_paths=evidence_paths,
        evidence_hashes=evidence_hashes,
        support_path=estimator_support_path,
        expected_support=estimator_support,
    )
    if final_model_path is not None and file_sha256(final_model_path) != final_model_sha256:
        raise RuntimeError("sealed final estimator model changed")
    return path


def _write_estimator_export_authorization(
    run_dir: Path,
    *,
    run_id: str,
    config: IncumbentCalibrationConfig,
    selected_candidate: str,
    selection_seal_path: Path,
    economics_path: Path,
    final_model_path: Path,
) -> Path:
    payload = {
        "schema_version": INCUMBENT_ESTIMATOR_EXPORT_AUTHORIZATION_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "source_process_id": config.process_id,
        "source_model_key": config.model_key,
        "selected_candidate_id": selected_candidate,
        "selected_final_model_sha256": file_sha256(final_model_path),
        "selection_seal_sha256": file_sha256(selection_seal_path),
        "economics_evidence_sha256": file_sha256(economics_path),
        "probability_qualified": True,
        "economics_qualified": True,
        "deployment_scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
        "source_process_changed": False,
    }
    path = run_dir / "estimator-fallback-export-authorization.json"
    write_json_atomic(path, payload)
    return path


def _estimator_fallback_status(
    status: str,
    *,
    trigger: str,
    selected_candidate_id: str | None = None,
    probability_qualified: bool = False,
    economics_opened: bool = False,
    economics_qualified: bool = False,
    paper_artifact: str | None = None,
    selection_seal_path: Path | None = None,
    economics_path: Path | None = None,
) -> dict[str, Any]:
    forward_requirements = (
        _forward_requirements(json.loads(selection_seal_path.read_text()))
        if paper_artifact is not None and selection_seal_path is not None
        else _inactive_forward_requirements(status)
    )
    return {
        "status": status,
        "trigger": trigger,
        "economics_used_for_trigger": False,
        "selected_candidate_id": selected_candidate_id,
        "probability_qualified": probability_qualified,
        "economics_opened": economics_opened,
        "economics_qualified": economics_qualified,
        "paper_artifact": paper_artifact,
        "selection_seal_sha256": (
            file_sha256(selection_seal_path) if selection_seal_path is not None else None
        ),
        "economics_evidence_sha256": (
            file_sha256(economics_path) if economics_path is not None else None
        ),
        "forward_requirements": forward_requirements,
    }


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _blocked_result(
    *,
    run_id: str,
    config: IncumbentCalibrationConfig,
    selection: Mapping[str, Any],
    selection_seal_path: Path,
    selection_seal: Mapping[str, Any],
    status: str,
    estimator_fallback_status: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": INCUMBENT_BENCHMARK_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "process_id": config.process_id,
        "source_model_key": config.model_key,
        "selected_candidate_id": selection.get("selected_candidate_id"),
        "probability_qualified": False,
        "economics_opened": False,
        "economics_qualified": False,
        "paper_artifact": None,
        "source_process_changed": False,
        "live_capital_allowed": False,
        "selection_seal_sha256": file_sha256(selection_seal_path),
        "selection_seal_economics_opened": selection_seal["economics_opened"],
        "estimator_fallback": dict(estimator_fallback_status),
        "forward_requirements": _inactive_forward_requirements(status),
    }


def _forward_requirements(selection_seal: Mapping[str, Any]) -> dict[str, Any]:
    created = datetime.fromisoformat(str(selection_seal["created_at"]))
    next_day = datetime(created.year, created.month, created.day, tzinfo=UTC)
    if created > next_day:
        from datetime import timedelta

        next_day += timedelta(days=1)
    return {
        "status": "awaiting_fresh_forward_evidence",
        "start_not_before": next_day.isoformat(),
        "minimum_complete_utc_days": 21,
        "minimum_strict_markets": 2_000,
        "minimum_candidate_trades": 200,
        "minimum_yes_trades": 20,
        "minimum_no_trades": 20,
        "minimum_noninferior_days": 17,
        "positive_lower_95_stressed_expectancy": True,
        "positive_lower_95_capital_efficiency": True,
        "net_corrected_decisions_positive": True,
        "loss_recovery_burden_no_worse_than_incumbent": True,
        "maximum_drawdown_no_worse_than_incumbent": True,
        "reapply_all_development_probability_and_economic_gates": True,
        "no_pnl_based_early_stopping": True,
        "incumbent_process_remains_unchanged": True,
    }


def _inactive_forward_requirements(reason: str) -> dict[str, Any]:
    return {
        "status": "not_started_no_qualified_paper_artifact",
        "reason": reason,
        "incumbent_process_remains_unchanged": True,
        "no_pnl_based_early_stopping": True,
    }


def _finalize_report(run_dir: Path, result: Mapping[str, Any]) -> None:
    write_json_atomic(run_dir / "benchmark.json", dict(result))
    lines = [
        "# Core+Oracle Gen2 calibration benchmark",
        "",
        f"- Status: `{result['status']}`",
        f"- Process scope: `{result['process_id']}`",
        f"- Incumbent: `{result['source_model_key']}`",
        f"- Selected candidate: `{result.get('selected_candidate_id') or 'none'}`",
        f"- Probability qualified: `{str(result['probability_qualified']).lower()}`",
        f"- Economics opened: `{str(result['economics_opened']).lower()}`",
        f"- Economics qualified: `{str(result['economics_qualified']).lower()}`",
        f"- Paper artifact: `{result.get('paper_artifact') or 'none'}`",
        "- Source paper process changed: `false`",
        "- Evidence scope: consumed development; not independent forward proof",
        "",
        "A failed gate retains the incumbent. No least-bad candidate is promoted.",
    ]
    _append_probability_report(
        lines,
        run_dir / "probability-selection.json",
        title="Calibration-family probability comparison",
    )
    _append_probability_report(
        lines,
        run_dir / "estimator-fallback-probability-selection.json",
        title="Conditional estimator probability comparison",
    )
    _append_economics_report(
        lines,
        run_dir / "economics-evidence.json",
        title="Calibration-family post-seal economics",
    )
    _append_economics_report(
        lines,
        run_dir / "estimator-fallback-economics-evidence.json",
        title="Conditional estimator post-seal economics",
    )
    (run_dir / "benchmark-report.md").write_text("\n".join(lines) + "\n")


def _append_probability_report(lines: list[str], path: Path, *, title: str) -> None:
    if not path.is_file():
        return
    selection = json.loads(path.read_text())
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            (
                "| Candidate | Passed | Target log loss | Target Brier | "
                "Selected bias | Noninferior days | Failed gates |"
            ),
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for record in selection.get("candidate_records", []):
        overall = record["metrics"]["overall"]
        selected = record.get("selected_opportunity_metrics") or {"overall": {}}
        failed = [gate["name"] for gate in record.get("gates", []) if not gate["passed"]]
        lines.append(
            "| {candidate} | {passed} | {log_loss} | {brier} | {bias} | {days} | {failed} |".format(
                candidate=record["candidate_id"],
                passed=str(record["passed"]).lower(),
                log_loss=_format_number(overall.get("log_loss"), 6),
                brier=_format_number(overall.get("brier"), 6),
                bias=_format_percent(selected["overall"].get("bias")),
                days=record.get("noninferior_utc_days", "n/a"),
                failed=", ".join(failed) or "none",
            )
        )


def _append_economics_report(lines: list[str], path: Path, *, title: str) -> None:
    if not path.is_file():
        return
    economics = json.loads(path.read_text())
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            (
                "| Model | Trades | W/L accuracy | Net PnL | Stress EV/trade | "
                "PF | Mean share | Max drawdown |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("incumbent", "challenger"):
        metrics = economics[label]["metrics"]
        lines.append(
            "| {label} | {trades} | {accuracy} | {net} | {stress} | {pf} | "
            "{price} | {drawdown} |".format(
                label=label,
                trades=metrics.get("trades", 0),
                accuracy=_format_percent(metrics.get("accuracy")),
                net=_format_money(metrics.get("net_profit")),
                stress=_format_money(metrics.get("stress_1c_net_expectancy_per_trade")),
                pf=_format_number(metrics.get("profit_factor"), 3),
                price=_format_money(metrics.get("mean_share_price")),
                drawdown=_format_money(metrics.get("maximum_drawdown")),
            )
        )
    correction = economics.get("correction_summary", {})
    lines.extend(
        [
            "",
            "### Direct decision corrections",
            "",
            f"- Net corrected decisions: `{correction.get('net_corrected_decisions', 0)}`",
            (
                "- Improvement UTC days: "
                f"`{correction.get('corrected_decision_improvement_utc_days', 0)}`"
            ),
            f"- Incumbent losses avoided: `{correction.get('avoided_losses', 0)}`",
            f"- Incumbent wins suppressed: `{correction.get('suppressed_wins', 0)}`",
            f"- Correct side changes: `{correction.get('correct_side_changes', 0)}`",
            f"- Incorrect side changes: `{correction.get('incorrect_side_changes', 0)}`",
            (
                "- Candidate-only wins/losses: "
                f"`{correction.get('candidate_only_wins', 0)}/"
                f"{correction.get('candidate_only_losses', 0)}`"
            ),
            (
                "- +1c-stressed incremental PnL: "
                f"`{_format_money(correction.get('incremental_pnl_stress_1c'))}`"
            ),
        ]
    )


def _format_number(value: Any, digits: int) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _format_percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def _format_money(value: Any) -> str:
    return "n/a" if value is None else f"${float(value):.4f}"


def _frame_key_sha256(frame: pl.DataFrame) -> str:
    keys = frame.select(
        pl.col("market_id").cast(pl.String),
        pl.col("window_start").dt.strftime("%Y-%m-%dT%H:%M:%S%.6fZ"),
        pl.col("observed_at").dt.strftime("%Y-%m-%dT%H:%M:%S%.6fZ"),
        pl.col("seconds_elapsed").cast(pl.Int64),
    ).sort("window_start", "market_id", "seconds_elapsed")
    digest = hashlib.sha256()
    for row in keys.iter_rows():
        digest.update("\x1f".join(map(str, row)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _candidate_model_key(candidate_id: str) -> str:
    return (
        "btc-5m-asymmetric-core-oracle-gen2-"
        + candidate_id.lower().replace("_", "-")
        + "-development-v1"
    )


def _window_payload(window: Any) -> dict[str, str]:
    return {"start": window.start.isoformat(), "end": window.end.isoformat()}


def _dataclass_payload(value: Any) -> dict[str, Any]:
    from dataclasses import asdict, is_dataclass

    if not is_dataclass(value):
        raise TypeError("frozen benchmark contract must be a dataclass")
    payload = asdict(value)
    return _json_safe(payload)


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _dependency_versions() -> dict[str, str]:
    return {
        package: version(package)
        for package in (
            "joblib",
            "numpy",
            "polars",
            "scikit-learn",
            "scipy",
            "threadpoolctl",
        )
    }


def _implementation_sha256() -> str:
    root = Path(__file__).resolve().parent
    paths = (
        Path(__file__).resolve(),
        root / "asymmetric_incumbent_calibration.py",
        root / "asymmetric_incumbent_evaluation.py",
        root / "asymmetric_incumbent_estimator.py",
        root / "asymmetric_gen2_export.py",
        root / "asymmetric_incumbent_replay.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_and_run_incumbent_calibration_benchmark(
    config_path: Path,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    return run_incumbent_calibration_benchmark(
        load_incumbent_calibration_config(config_path),
        force=force,
    )


__all__ = [
    "load_and_run_incumbent_calibration_benchmark",
    "run_incumbent_calibration_benchmark",
]
