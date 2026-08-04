"""End-to-end paper benchmark for lower-cost asymmetric-value opportunities."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from .asymmetric_value_config import AsymmetricValueConfig
from .asymmetric_value_data import (
    EARLY_CAUSAL_ORACLE_FEATURES,
    ORACLE_MAXIMUM_AGE_SECONDS,
    ORACLE_MINIMUM_PROPAGATION_SECONDS,
    attach_asymmetric_value_features,
    attach_early_causal_oracle_features,
    exact_price_by_second,
    execution_grid_coverage,
    extract_asymmetric_price_evidence,
    load_asymmetric_price_evidence,
)
from .asymmetric_value_evaluation import (
    accuracy_price_by_second,
    add_bootstrap_metrics,
    bootstrap_ledger_metrics,
    confidence_control_ledger,
    confidence_reference_ledger,
    current_policy_reference_ledger,
    evaluate_policy_grid,
    evidence_gate_checks,
    joint_accuracy_value_surface,
    ledger_metrics,
    opportunity_calibration_by_price_band,
    policy_gate_checks,
    policy_ledger,
    price_band_metrics,
    score_two_sided_value,
    select_policy_candidate,
    side_accuracy_value_surface,
)
from .asymmetric_value_training import (
    CORE_CANDLES_PRICE,
    CORE_CONTROL,
    CORE_L2_CANDLES_PRICE,
    CORE_L2_PRICE,
    CORE_ORACLE_L2_CANDLES_PRICE,
    CORE_ORACLE_PRICE,
    CORE_PRICE,
    L2_CANDLES_MATCHED_CORE_PRICE_CONTROL,
    MODEL_SELECTION_ELIGIBLE,
    ORACLE_L2_CANDLES_MATCHED_CORE_ORACLE_PRICE_CONTROL,
    ORACLE_MATCHED_CORE_PRICE_CONTROL,
    PAIRED_CORE_CONTROL,
    PRICE_LOGISTIC,
    asymmetric_probability_frame,
    fit_asymmetric_value_models,
)
from .core_config import load_core_config
from .core_extract import extract_core_source, file_sha256, write_json_atomic
from .core_features import build_core_features, load_core_feature_frame
from .early_value_data import build_partitioned_external_frame
from .runtime_export import score_runtime_model

ASYMMETRIC_VALUE_SCHEMA_VERSION = "btc-asymmetric-value-hunter-benchmark-v1"
FROZEN_CHAMPION = "frozen_champion_reference_60s_plus"
DEVELOPMENT_ORACLE_CACHE = "development-oracle-propagation-2s.parquet"
EVALUATION_ORACLE_CACHE = "evaluation-oracle-propagation-2s.parquet"


def run_asymmetric_value_benchmark(
    config: AsymmetricValueConfig,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    implementation_sha256 = _implementation_digest(config)
    dependency_versions = _dependency_versions()
    core_config = load_core_config(config.core_config)
    current_process = _current_process_contract(config)
    print("asymmetric-value: preparing pre-evaluation causal features", flush=True)
    extract_core_source(core_config, "pre_holdout", force=force)
    build_core_features(core_config, "pre_holdout", force=force)
    development = load_core_feature_frame(core_config, "pre_holdout")
    development_core_content_sha256 = _frame_content_digest(development)
    development_oracle_inventory = _oracle_source_inventory(
        development,
        config.oracle_source,
    )
    development_oracle = _load_or_build_oracle_core(
        development,
        config,
        destination=config.feature_cache / DEVELOPMENT_ORACLE_CACHE,
        source_inventory=development_oracle_inventory,
        core_content_sha256=development_core_content_sha256,
        force=force,
    )
    development_external = _load_or_build_external(
        development,
        config,
        destination=config.feature_cache / "development-external.parquet",
        core_content_sha256=development_core_content_sha256,
        force=force,
    )
    development_external_oracle = _join_early_oracle(
        development_external,
        development_oracle,
    )

    print("asymmetric-value: extracting pre-evaluation executable books", flush=True)
    development_price_manifest = extract_asymmetric_price_evidence(
        config,
        scope="development",
        force=force,
    )
    development_prices = load_asymmetric_price_evidence(
        config,
        scope="development",
    )
    development_price_features = attach_asymmetric_value_features(
        development,
        development_prices,
        config,
    )
    development_oracle_price_features = attach_asymmetric_value_features(
        development_oracle,
        development_prices,
        config,
    ).filter(pl.col("early_oracle_eligible"))
    development_external_price_features = attach_asymmetric_value_features(
        development_external,
        development_prices,
        config,
    )
    development_external_oracle_price_features = attach_asymmetric_value_features(
        development_external_oracle,
        development_prices,
        config,
    ).filter(pl.col("early_oracle_eligible"))
    development_model_frames = _candidate_frames(
        price=development_price_features,
        oracle_price=development_oracle_price_features,
        l2_candles_price=development_external_price_features,
        oracle_l2_candles_price=development_external_oracle_price_features,
    )

    print("asymmetric-value: fitting price-aware champion-family candidates", flush=True)
    models, training = fit_asymmetric_value_models(
        development_model_frames,
        development,
        config,
        core_config,
    )
    policy_frames = {
        name: _window(frame, config.policy.start, config.policy.end)
        for name, frame in development_model_frames.items()
    }
    policy_predictions = _prediction_surface(policy_frames, models)
    policy_scored = score_two_sided_value(policy_predictions)
    policy_joint_surface = joint_accuracy_value_surface(policy_scored)
    policy_both_side_surface = side_accuracy_value_surface(policy_predictions)

    print("asymmetric-value: selecting model and economic policy without an accuracy gate", flush=True)
    policy_ledgers, policy_metrics = evaluate_policy_grid(policy_scored, config)
    add_bootstrap_metrics(
        policy_metrics,
        policy_ledgers,
        resamples=config.bootstrap_resamples,
        seed=config.random_seed,
    )
    _add_opportunity_denominators(
        policy_metrics,
        resolved_markets=_window(
            development,
            config.policy.start,
            config.policy.end,
        )["market_id"].n_unique(),
        strict_markets_by_model={
            name: frame["market_id"].n_unique()
            for name, frame in policy_frames.items()
        },
    )
    policy_grid_coverage = execution_grid_coverage(
        config,
        scope="development",
        core=_window(development, config.policy.start, config.policy.end),
    )
    policy_core = _window(development, config.policy.start, config.policy.end)
    policy_candidate_coverage = {
        name: _candidate_grid_summary(frame, policy_core, config)
        for name, frame in policy_frames.items()
    }
    policy_evidence_checks_by_model = {
        name: evidence_gate_checks(
            frame,
            config,
            policy_window=True,
            source_grid_coverage=float(policy_grid_coverage["retained_coverage"]),
            strict_grid_coverage=float(policy_grid_coverage["strict_coverage"]),
            candidate_grid_coverage=float(
                policy_candidate_coverage[name]["prediction_grid_coverage"]
            ),
        )
        for name, frame in policy_frames.items()
    }
    selection = select_policy_candidate(
        policy_metrics,
        config,
        evidence_checks_by_model=policy_evidence_checks_by_model,
        eligible_models=set(MODEL_SELECTION_ELIGIBLE),
    )
    selected_matched_control = _selected_matched_control(
        selection["selected_model"]
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir = run_dir / "models"
    model_dir.mkdir()
    model_hashes: dict[str, str] = {}
    for name, bundle in models.items():
        path = model_dir / f"{name}.joblib"
        joblib.dump(bundle, path, compress=3)
        model_hashes[name] = file_sha256(path)
    policy_predictions.write_parquet(
        run_dir / "policy-predictions.parquet",
        compression="zstd",
    )
    write_json_atomic(run_dir / "policy-selection.json", selection)
    selection_seal = {
        "schema_version": "btc-asymmetric-value-selection-seal-v1",
        "sealed_at": datetime.now(UTC).isoformat(),
        "evaluation_opened": False,
        "selected_key": selection["selected_key"],
        "selected_model": selection["selected_model"],
        "selected_policy": selection["selected_policy"],
        "predeclared_evaluation_controls": [
            CORE_CONTROL,
            *([selected_matched_control] if selected_matched_control else []),
        ],
        "predeclared_frozen_evaluation_diagnostics": [
            "full_current_policy_reference",
            "selected_market_paired_current_policy_reference",
            "frozen_champion_threshold_only_sweep",
            "frozen_champion_edge_guarded_reference",
            "retrained_core_control_threshold_sweep",
        ],
        "qualified_on_policy_window": selection["qualified_on_policy_window"],
        "policy_evidence_checks_by_model": policy_evidence_checks_by_model,
        "policy_source_grid": policy_grid_coverage,
        "policy_candidate_grid": policy_candidate_coverage,
        "policy_cohort_key_sha256": {
            name: _frame_key_digest(frame) for name, frame in policy_frames.items()
        },
        "implementation_sha256": implementation_sha256,
        "dependency_versions": dependency_versions,
        "development_core_content_sha256": development_core_content_sha256,
        "policy_predictions_sha256": file_sha256(
            run_dir / "policy-predictions.parquet"
        ),
        "policy_selection_sha256": file_sha256(
            run_dir / "policy-selection.json"
        ),
        "benchmark_config_sha256": file_sha256(config.source_path),
        "core_config_sha256": file_sha256(config.core_config),
        "champion_model_sha256": file_sha256(config.champion_model),
        "champion_process_sha256": file_sha256(config.champion_process),
        "frozen_current_policy_contract": current_process,
        "price_query_sha256": file_sha256(config.price_source_sql),
        "development_price_manifest_sha256": file_sha256(
            config.price_cache / "development" / "manifest.json"
        ),
        "development_external_features_sha256": file_sha256(
            config.feature_cache / "development-external.parquet"
        ),
        "development_oracle_features_sha256": file_sha256(
            config.feature_cache / DEVELOPMENT_ORACLE_CACHE
        ),
        "development_oracle_source_inventory": development_oracle_inventory,
        "model_artifact_sha256": model_hashes,
    }
    write_json_atomic(run_dir / "selection-seal.json", selection_seal)
    selection_seal_sha256 = file_sha256(run_dir / "selection-seal.json")

    print("asymmetric-value: selection sealed; opening frozen evaluation", flush=True)
    extract_core_source(core_config, "holdout", force=force)
    build_core_features(core_config, "holdout", force=force)
    evaluation = load_core_feature_frame(core_config, "holdout")
    evaluation_core_content_sha256 = _frame_content_digest(evaluation)
    evaluation_oracle_inventory = _oracle_source_inventory(
        evaluation,
        config.oracle_source,
    )
    evaluation_oracle = _load_or_build_oracle_core(
        evaluation,
        config,
        destination=config.feature_cache / EVALUATION_ORACLE_CACHE,
        source_inventory=evaluation_oracle_inventory,
        core_content_sha256=evaluation_core_content_sha256,
        force=force,
    )
    evaluation_external = _load_or_build_external(
        evaluation,
        config,
        destination=config.feature_cache / "evaluation-external.parquet",
        core_content_sha256=evaluation_core_content_sha256,
        force=force,
    )
    evaluation_external_oracle = _join_early_oracle(
        evaluation_external,
        evaluation_oracle,
    )
    evaluation_price_manifest = extract_asymmetric_price_evidence(
        config,
        scope="evaluation",
        force=force,
    )
    evaluation_prices = load_asymmetric_price_evidence(
        config,
        scope="evaluation",
    )
    evaluation_price_features = attach_asymmetric_value_features(
        evaluation,
        evaluation_prices,
        config,
    )
    evaluation_oracle_price_features = attach_asymmetric_value_features(
        evaluation_oracle,
        evaluation_prices,
        config,
    ).filter(pl.col("early_oracle_eligible"))
    evaluation_external_price_features = attach_asymmetric_value_features(
        evaluation_external,
        evaluation_prices,
        config,
    )
    evaluation_external_oracle_price_features = attach_asymmetric_value_features(
        evaluation_external_oracle,
        evaluation_prices,
        config,
    ).filter(pl.col("early_oracle_eligible"))
    evaluation_model_frames = _candidate_frames(
        price=evaluation_price_features,
        oracle_price=evaluation_oracle_price_features,
        l2_candles_price=evaluation_external_price_features,
        oracle_l2_candles_price=evaluation_external_oracle_price_features,
    )
    selected_evaluation_frame = evaluation_model_frames[selection["selected_model"]]
    coverage = _coverage_summary(
        pl.concat(
            (development, evaluation),
            how="vertical_relaxed",
        ),
        pl.concat(
            (development_external, evaluation_external),
            how="vertical_relaxed",
        ),
        pl.concat(
            (development_price_features, evaluation_price_features),
            how="vertical_relaxed",
        ),
        config,
    )
    coverage["policy"]["source_grid"] = policy_grid_coverage
    coverage["policy"]["candidate_strict_markets"] = {
        name: frame["market_id"].n_unique() for name, frame in policy_frames.items()
    }
    coverage["policy"]["candidate_prediction_grid"] = policy_candidate_coverage
    evaluation_grid_coverage = execution_grid_coverage(
        config,
        scope="evaluation",
        core=evaluation,
    )
    coverage["evaluation"]["source_grid"] = evaluation_grid_coverage
    coverage["evaluation"]["candidate_strict_markets"] = {
        name: frame["market_id"].n_unique()
        for name, frame in evaluation_model_frames.items()
    }
    evaluation_candidate_coverage = {
        name: _candidate_grid_summary(frame, evaluation, config)
        for name, frame in evaluation_model_frames.items()
    }
    coverage["evaluation"]["candidate_prediction_grid"] = (
        evaluation_candidate_coverage
    )

    evaluation_prediction_frames = {
        selection["selected_model"]: selected_evaluation_frame,
        CORE_CONTROL: evaluation_model_frames[CORE_CONTROL],
    }
    if selected_matched_control is not None:
        evaluation_prediction_frames[selected_matched_control] = (
            selected_evaluation_frame
        )
    evaluation_model_names = set(evaluation_prediction_frames)
    evaluation_predictions = _prediction_surface(
        evaluation_prediction_frames,
        {
            name: model
            for name, model in models.items()
            if name in evaluation_model_names
        },
    )
    selected_evaluation_predictions = evaluation_predictions.filter(
        pl.col("model") == selection["selected_model"]
    )
    selected_evaluation_scored = score_two_sided_value(
        selected_evaluation_predictions
    )
    selected_key = selection["selected_key"]
    selected_policy = next(
        policy
        for policy in config.policies
        if policy.name == selection["selected_policy"]
    )
    selected_ledger = policy_ledger(
        selected_evaluation_scored,
        selected_policy,
        quantity=config.quantity,
    )
    selected_metrics = ledger_metrics(selected_ledger)
    selected_metrics["utc_day_block_bootstrap"] = bootstrap_ledger_metrics(
        selected_ledger,
        resamples=config.bootstrap_resamples,
        seed=config.random_seed + 10_000,
    )
    _add_opportunity_denominators(
        {selected_key: selected_metrics},
        resolved_markets=evaluation["market_id"].n_unique(),
        strict_markets_by_model={
            selection["selected_model"]: selected_evaluation_frame[
                "market_id"
            ].n_unique()
        },
    )
    matched_control_metrics: dict[str, Any] | None = None
    matched_control_ledger: pl.DataFrame | None = None
    if selected_matched_control is not None:
        matched_control_scored = score_two_sided_value(
            evaluation_predictions.filter(
                pl.col("model") == selected_matched_control
            )
        )
        matched_control_ledger = policy_ledger(
            matched_control_scored,
            selected_policy,
            quantity=config.quantity,
        )
        matched_control_metrics = ledger_metrics(matched_control_ledger)
        matched_control_metrics["utc_day_block_bootstrap"] = (
            bootstrap_ledger_metrics(
                matched_control_ledger,
                resamples=config.bootstrap_resamples,
                seed=config.random_seed + 11_000,
            )
        )
        _add_opportunity_denominators(
            {
                f"{selected_matched_control}::{selected_policy.name}": (
                    matched_control_metrics
                )
            },
            resolved_markets=evaluation["market_id"].n_unique(),
            strict_markets_by_model={
                selected_matched_control: selected_evaluation_frame[
                    "market_id"
                ].n_unique()
            },
        )
    evaluation_evidence_checks = evidence_gate_checks(
        selected_evaluation_frame,
        config,
        policy_window=False,
        source_grid_coverage=float(evaluation_grid_coverage["retained_coverage"]),
        strict_grid_coverage=float(evaluation_grid_coverage["strict_coverage"]),
        candidate_grid_coverage=float(
            evaluation_candidate_coverage[selection["selected_model"]][
                "prediction_grid_coverage"
            ]
        ),
    )
    evaluation_economic_checks = policy_gate_checks(
        selected_metrics,
        config,
        policy_window=False,
    )
    evaluation_checks = [
        *evaluation_economic_checks,
        *evaluation_evidence_checks,
    ]
    evaluation_qualified = bool(
        selection["qualified_on_policy_window"]
        and all(check["passed"] for check in evaluation_checks)
    )
    evaluation_evidence_sufficient = all(
        check["passed"] for check in evaluation_evidence_checks
    )
    evaluation_status = (
        "qualified"
        if evaluation_qualified
        else (
            "insufficient_execution_evidence"
            if not evaluation_evidence_sufficient
            else "not_qualified"
        )
    )

    full_champion_predictions = _champion_probability_frame(
        config,
        evaluation,
    )
    selected_market_ids = selected_evaluation_frame.select("market_id").unique()
    paired_champion_predictions = full_champion_predictions.join(
        selected_market_ids,
        on="market_id",
        how="inner",
        validate="m:1",
    )
    paired_execution = evaluation_price_features.join(
        selected_market_ids,
        on="market_id",
        how="inner",
        validate="m:1",
    )
    full_current_policy_reference = current_policy_reference_ledger(
        full_champion_predictions,
        execution=evaluation_price_features,
        threshold=current_process["confidence_threshold"],
        minimum_entry_second=current_process["minimum_entry_second"],
        maximum_entry_second=current_process["maximum_entry_second"],
        minimum_share_price=current_process["minimum_share_price"],
        maximum_share_price=current_process["maximum_share_price"],
        maximum_depth_participation=current_process[
            "maximum_depth_participation"
        ],
        quantity=config.quantity,
    )
    paired_current_policy_reference = current_policy_reference_ledger(
        paired_champion_predictions,
        execution=paired_execution,
        threshold=current_process["confidence_threshold"],
        minimum_entry_second=current_process["minimum_entry_second"],
        maximum_entry_second=current_process["maximum_entry_second"],
        minimum_share_price=current_process["minimum_share_price"],
        maximum_share_price=current_process["maximum_share_price"],
        maximum_depth_participation=current_process[
            "maximum_depth_participation"
        ],
        quantity=config.quantity,
    )
    full_current_policy_metrics = ledger_metrics(full_current_policy_reference)
    full_current_policy_metrics["utc_day_block_bootstrap"] = _bootstrap_reference(
        full_current_policy_reference,
        config,
    )
    full_current_policy_metrics["terminal_crossing_diagnostics"] = (
        _current_reference_crossing_diagnostics(
            full_champion_predictions,
            full_current_policy_reference,
            current_process,
        )
    )
    paired_current_policy_metrics = ledger_metrics(
        paired_current_policy_reference
    )
    paired_current_policy_metrics["utc_day_block_bootstrap"] = (
        _bootstrap_reference(paired_current_policy_reference, config)
    )
    paired_current_policy_metrics["terminal_crossing_diagnostics"] = (
        _current_reference_crossing_diagnostics(
            paired_champion_predictions,
            paired_current_policy_reference,
            current_process,
        )
    )
    frozen_champion_thresholds = _frozen_champion_threshold_controls(
        full_champion_predictions,
        evaluation_price_features,
        current_process,
        config,
    )
    champion_executable_predictions = _champion_probability_frame(
        config,
        evaluation_price_features,
        include_execution=True,
    )
    guarded_champion_reference = confidence_reference_ledger(
        champion_executable_predictions,
        threshold=0.89,
        minimum_edge_per_share=(
            config.confidence_control_minimum_edge_per_share
        ),
        quantity=config.quantity,
    )
    guarded_champion_metrics = ledger_metrics(guarded_champion_reference)
    guarded_champion_metrics["utc_day_block_bootstrap"] = (
        _bootstrap_reference(guarded_champion_reference, config)
    )
    comparison = _risk_shape_comparison(
        selected_metrics,
        paired_current_policy_metrics,
    )
    accuracy_prices = accuracy_price_by_second(selected_evaluation_scored)
    evaluation_joint_surface = joint_accuracy_value_surface(
        selected_evaluation_scored
    )
    evaluation_both_side_surface = side_accuracy_value_surface(
        selected_evaluation_predictions
    )
    opportunity_calibration = opportunity_calibration_by_price_band(
        selected_evaluation_scored
    )
    exact_prices = exact_price_by_second(evaluation_prices)
    confidence_controls = _confidence_threshold_controls(
        policy_predictions,
        evaluation_predictions,
        config,
    )

    evaluation_predictions.write_parquet(
        run_dir / "evaluation-predictions.parquet",
        compression="zstd",
    )
    selected_evaluation_scored.write_parquet(
        run_dir / "evaluation-scored-opportunities.parquet",
        compression="zstd",
    )
    selected_ledger.write_parquet(run_dir / "selected-policy-ledger.parquet", compression="zstd")
    full_current_policy_reference.write_parquet(
        run_dir / "frozen-current-policy-full-reference-ledger.parquet",
        compression="zstd",
    )
    paired_current_policy_reference.write_parquet(
        run_dir / "frozen-current-policy-paired-reference-ledger.parquet",
        compression="zstd",
    )
    if matched_control_ledger is not None:
        matched_control_ledger.write_parquet(
            run_dir / "selected-matched-control-ledger.parquet",
            compression="zstd",
        )
    guarded_champion_reference.write_parquet(
        run_dir / "frozen-champion-edge-guarded-reference-ledger.parquet",
        compression="zstd",
    )
    price_bands = price_band_metrics(selected_ledger)
    pl.DataFrame(price_bands).write_csv(run_dir / "selected-price-band-economics.csv")
    pl.DataFrame(_policy_table(policy_metrics, selection["frontier"])).write_csv(
        run_dir / "candidate-policy-economics.csv"
    )
    pl.DataFrame(accuracy_prices).write_csv(
        run_dir / "accuracy-price-by-five-seconds.csv"
    )
    pl.DataFrame(exact_prices).write_csv(
        run_dir / "exact-price-by-observation-second.csv"
    )
    pl.DataFrame(opportunity_calibration).write_csv(
        run_dir / "opportunity-calibration-by-price-band.csv"
    )
    pl.DataFrame(confidence_controls["table"]).write_csv(
        run_dir / "core-control-confidence-threshold-economics.csv"
    )
    pl.DataFrame(frozen_champion_thresholds["table"]).write_csv(
        run_dir / "frozen-champion-confidence-threshold-economics.csv"
    )
    pl.DataFrame(
        [
            {"window": "policy", **row}
            for row in policy_joint_surface
        ]
        + [
            {"window": "evaluation", **row}
            for row in evaluation_joint_surface
        ]
    ).write_csv(run_dir / "model-second-side-price-band-surface.csv")
    pl.DataFrame(
        [{"window": "policy", **row} for row in policy_both_side_surface]
        + [
            {"window": "evaluation", **row}
            for row in evaluation_both_side_surface
        ]
    ).write_csv(run_dir / "model-second-yes-no-price-band-surface.csv")

    result: dict[str, Any] = {
        "schema_version": ASYMMETRIC_VALUE_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "objective": (
            "find fee- and reserve-adjusted lower-cost YES or NO claims whose calibrated "
            "win probability exceeds executable cost, while accepting lower hit rates"
        ),
        "paper_only": True,
        "live_capital_allowed": False,
        "runtime_changed": False,
        "trading_process_changed": False,
        "core_contract_changed": False,
        "model_contract": {
            "estimator_family": "champion-family histogram gradient boosting",
            "price_aware_candidates": True,
            "two_sided_value_selection": True,
            "underdog_selection_allowed": True,
            "accuracy_gate_used": False,
            "market_equal_row_weights": True,
            "candidate_specific_training_cohorts": True,
            "original_core_control_uses_universal_core": True,
            "opening_boundary_features_used_by_candidates": False,
            "causal_oracle_features": list(EARLY_CAUSAL_ORACLE_FEATURES),
            "feature_and_source_ablation_candidates": [
                CORE_PRICE,
                CORE_ORACLE_PRICE,
                CORE_L2_PRICE,
                CORE_CANDLES_PRICE,
                CORE_L2_CANDLES_PRICE,
                CORE_ORACLE_L2_CANDLES_PRICE,
            ],
            "matched_feature_controls": [
                ORACLE_MATCHED_CORE_PRICE_CONTROL,
                L2_CANDLES_MATCHED_CORE_PRICE_CONTROL,
                ORACLE_L2_CANDLES_MATCHED_CORE_ORACLE_PRICE_CONTROL,
            ],
            "l2_and_candle_single-source_arms_use_paired_common_cohort": True,
            "primary_maximum_admission_cost_per_share": max(
                policy.maximum_cost_per_share
                for policy in config.policies
                if policy.selection_eligible
            ),
            "diagnostic_maximum_admission_cost_per_share": (
                config.gates.maximum_mean_cost_per_share
            ),
            "primary_maximum_raw_share_price": 0.30,
            "execution_reserve_per_share": config.execution_reserve_per_share,
            "realized_execution_stress_per_share": 0.01,
            "selection_eligible_policies": [
                policy.name for policy in config.policies if policy.selection_eligible
            ],
        },
        "windows": {
            name: {"start": value.start.isoformat(), "end": value.end.isoformat()}
            for name, value in (
                ("fit", config.fit),
                ("calibration", config.calibration),
                ("policy", config.policy),
                ("evaluation", config.evaluation),
            )
        },
        "split_basis": (
            "existing frozen early-value benchmark boundaries; no boundary was moved "
            "using asymmetric-value outcome or PnL"
        ),
        "data": {
            "coverage": coverage,
            "price_manifests": {
                "development": development_price_manifest,
                "evaluation": evaluation_price_manifest,
            },
            "oracle_source_inventories": {
                "development": development_oracle_inventory,
                "evaluation": evaluation_oracle_inventory,
            },
            "price_source": (
                "immutable PMXT 250ms execution snapshots downsampled to exact "
                "1s/5s decision timestamps"
            ),
            "proxy_prices_used": False,
            "missing_prices_treated_as_losses": False,
            "missing_prices_treated_as_no_trade_in_resolved_market_denominator": True,
            "fill_model_limitation": (
                "decision-time exact VWAP5 is baseline execution evidence; the 1c/share "
                "stress is reported, but a post-decision fill lookup is not available"
            ),
            "historical_refprice_used": False,
            "historical_refprice_reason": (
                "local receipt/availability timestamps are not proven"
            ),
            "oracle_rounds_used": True,
            "oracle_round_minimum_propagation_seconds": (
                ORACLE_MINIMUM_PROPAGATION_SECONDS
            ),
            "oracle_round_maximum_age_seconds": ORACLE_MAXIMUM_AGE_SECONDS,
            "oracle_availability_limitation": (
                "block timestamp is not local receipt time; candidates impose a conservative "
                "two-second propagation delay before a round becomes eligible"
            ),
        },
        "training": training,
        "selection": {
            **selection,
            "seal_sha256": selection_seal_sha256,
            "evaluation_opened_after_seal": True,
        },
        "evaluation": {
            "status": evaluation_status,
            "qualified": evaluation_qualified,
            "checks": evaluation_checks,
            "economic_checks": evaluation_economic_checks,
            "evidence_checks": evaluation_evidence_checks,
            "economic_point_estimate_passed": all(
                check["passed"] for check in evaluation_economic_checks
            ),
            "evidence_sufficient": evaluation_evidence_sufficient,
            "selected_key": selected_key,
            "selected_metrics": selected_metrics,
            "selected_price_bands": price_bands,
            "accuracy_price_by_five_seconds": accuracy_prices,
            "joint_surface_artifact": "model-second-side-price-band-surface.csv",
            "both_side_surface_artifact": (
                "model-second-yes-no-price-band-surface.csv"
            ),
            "candidate_policy_economics_artifact": (
                "candidate-policy-economics.csv"
            ),
            "exact_price_by_observation_second": exact_prices,
            "opportunity_calibration_by_price_band": opportunity_calibration,
            "core_control_confidence_thresholds": confidence_controls,
            "frozen_champion_confidence_thresholds": (
                frozen_champion_thresholds
            ),
            "predeclared_evaluation_control_models": [
                CORE_CONTROL,
                *([selected_matched_control] if selected_matched_control else []),
            ],
            "non_predeclared_model_metrics_opened_on_evaluation": False,
            "policy_window_candidate_metrics": policy_metrics,
            "selected_feature_attribution_control": {
                "model": selected_matched_control,
                "metrics": matched_control_metrics,
                "same_evaluation_keys_as_selected": bool(
                    selected_matched_control is not None
                ),
            },
            "frozen_current_policy_full_exact_book_89": (
                full_current_policy_metrics
            ),
            "frozen_current_policy_market_paired_89": (
                paired_current_policy_metrics
            ),
            "frozen_current_policy_contract": current_process,
            "frozen_current_policy_reference_scope": (
                "frozen model on its five-second feature cadence, terminal first 89% "
                "confidence crossing before exact-book validation, 30-95c price bounds, "
                "five-share depth participation, and no added edge gate; full and "
                "selected-market-paired cohorts are reported separately"
            ),
            "frozen_current_policy_fidelity_limitations": (
                "historical PMXT evidence proves decision-time VWAP5 and depth at the model's "
                "five-second cadence, while the process loop runs every second; the replay "
                "does not reconstruct intermediate-loop quotes, marketable-limit price, "
                "quoted-size, or 150ms arrival fills"
            ),
            "frozen_champion_edge_guarded_diagnostic_89": guarded_champion_metrics,
            "risk_shape_comparison": comparison,
        },
        "hypothesis": {
            "alleviates_expensive_failure_risk": bool(
                evaluation_qualified
                and comparison["mean_share_price_reduced"]
                and comparison["mean_cost_reduced"]
                and comparison["loss_recovery_improved"]
                and comparison["capital_efficiency_improved"]
            ),
            "fresh_forward_shadow_required": True,
            "fresh_forward_stopping_rule": {
                "minimum_complete_source_days": 21,
                "minimum_strict_markets": 2000,
                "minimum_trades": 200,
                "maximum_calendar_days": 30,
                "stop_based_on_pnl": False,
            },
            "evaluation_is_consumed_development_evidence": True,
            "accuracy_is_reported_but_not_an_admission_gate": True,
            "lower_probability_side_may_be_selected": (
                "required to test low-priced underdog value such as 21% probability at 8c"
            ),
        },
        "lineage": {
            "benchmark_config_sha256": file_sha256(config.source_path),
            "core_config_sha256": file_sha256(config.core_config),
            "price_query_sha256": file_sha256(config.price_source_sql),
            "champion_model_sha256": file_sha256(config.champion_model),
            "champion_process_sha256": file_sha256(config.champion_process),
            "implementation_sha256": implementation_sha256,
            "dependency_versions": dependency_versions,
            "selection_seal_sha256": selection_seal_sha256,
            "development_core_content_sha256": (
                development_core_content_sha256
            ),
            "evaluation_core_content_sha256": evaluation_core_content_sha256,
            "policy_predictions_sha256": file_sha256(
                run_dir / "policy-predictions.parquet"
            ),
            "policy_selection_sha256": file_sha256(
                run_dir / "policy-selection.json"
            ),
            "development_external_features_sha256": file_sha256(
                config.feature_cache / "development-external.parquet"
            ),
            "evaluation_external_features_sha256": file_sha256(
                config.feature_cache / "evaluation-external.parquet"
            ),
            "development_oracle_features_sha256": file_sha256(
                config.feature_cache / DEVELOPMENT_ORACLE_CACHE
            ),
            "evaluation_oracle_features_sha256": file_sha256(
                config.feature_cache / EVALUATION_ORACLE_CACHE
            ),
        },
        "deployment": {
            "authorized": False,
            "runtime_exported": False,
            "trading_process_changed": False,
            "container_rebuilt": False,
            "reason": "offline training and economic benchmark only",
        },
    }
    write_json_atomic(run_dir / "benchmark.json", result)
    (run_dir / "benchmark-report.md").write_text(_markdown_report(result))
    print(
        "asymmetric-value: complete; selected="
        f"{selected_key}; qualified={evaluation_qualified}",
        flush=True,
    )
    return run_dir, result


def _candidate_frames(
    *,
    price: pl.DataFrame,
    oracle_price: pl.DataFrame,
    l2_candles_price: pl.DataFrame,
    oracle_l2_candles_price: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    return {
        PRICE_LOGISTIC: price,
        CORE_CONTROL: price,
        PAIRED_CORE_CONTROL: price,
        CORE_PRICE: price,
        ORACLE_MATCHED_CORE_PRICE_CONTROL: oracle_price,
        CORE_ORACLE_PRICE: oracle_price,
        CORE_L2_PRICE: l2_candles_price,
        CORE_CANDLES_PRICE: l2_candles_price,
        L2_CANDLES_MATCHED_CORE_PRICE_CONTROL: l2_candles_price,
        CORE_L2_CANDLES_PRICE: l2_candles_price,
        ORACLE_L2_CANDLES_MATCHED_CORE_ORACLE_PRICE_CONTROL: (
            oracle_l2_candles_price
        ),
        CORE_ORACLE_L2_CANDLES_PRICE: oracle_l2_candles_price,
    }


def _candidate_grid_summary(
    frame: pl.DataFrame,
    core: pl.DataFrame,
    config: AsymmetricValueConfig,
) -> dict[str, Any]:
    core_markets = core["market_id"].n_unique() if not core.is_empty() else 0
    expected_rows = core_markets * len(config.prediction_seconds)
    keys = frame.select("market_id", "seconds_elapsed").unique()
    observed_by_second = (
        keys.group_by("seconds_elapsed")
        .agg(pl.col("market_id").n_unique().alias("markets"))
    )
    by_second = (
        pl.DataFrame(
            {"seconds_elapsed": list(config.prediction_seconds)}
        )
        .join(
            observed_by_second,
            on="seconds_elapsed",
            how="left",
            validate="1:1",
        )
        .with_columns(pl.col("markets").fill_null(0))
        .with_columns(
            (pl.col("markets") / core_markets).alias("market_coverage")
            if core_markets
            else pl.lit(0.0).alias("market_coverage")
        )
        .to_dicts()
    )
    return {
        "core_markets": core_markets,
        "expected_rows": expected_rows,
        "candidate_rows": keys.height,
        "candidate_markets": keys["market_id"].n_unique() if keys.height else 0,
        "candidate_utc_days": (
            frame["window_start"].dt.date().n_unique() if not frame.is_empty() else 0
        ),
        "prediction_grid_coverage": (
            keys.height / expected_rows if expected_rows else 0.0
        ),
        "minimum_second_market_coverage": min(
            (float(row["market_coverage"]) for row in by_second),
            default=0.0,
        ),
        "by_second": by_second,
    }


def _join_early_oracle(
    frame: pl.DataFrame,
    oracle: pl.DataFrame,
) -> pl.DataFrame:
    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
    columns = [
        *keys,
        *EARLY_CAUSAL_ORACLE_FEATURES,
        "oracle_source_timestamp",
        "oracle_block_timestamp",
        "oracle_age_seconds",
        "early_oracle_eligible",
    ]
    return frame.join(
        oracle.select(*columns),
        on=keys,
        how="inner",
        validate="1:1",
    )


def _load_or_build_oracle_core(
    core: pl.DataFrame,
    config: AsymmetricValueConfig,
    *,
    destination: Path,
    source_inventory: dict[str, Any],
    core_content_sha256: str,
    force: bool,
) -> pl.DataFrame:
    metadata_path = destination.with_suffix(".metadata.json")
    identity = {
        "schema_version": "btc-asymmetric-value-early-oracle-v2",
        "core_key_sha256": _frame_key_digest(core),
        "core_content_sha256": core_content_sha256,
        "source_inventory_sha256": source_inventory["inventory_sha256"],
        "minimum_propagation_seconds": ORACLE_MINIMUM_PROPAGATION_SECONDS,
        "maximum_age_seconds": ORACLE_MAXIMUM_AGE_SECONDS,
        "features": list(EARLY_CAUSAL_ORACLE_FEATURES),
    }
    if destination.is_file() and not force:
        if not metadata_path.is_file():
            raise RuntimeError("oracle feature cache lacks a provenance manifest")
        metadata = json.loads(metadata_path.read_text())
        observed = {name: metadata.get(name) for name in identity}
        if observed != identity or metadata.get("sha256") != file_sha256(destination):
            raise RuntimeError("oracle feature cache provenance changed; rebuild intentionally")
        return pl.read_parquet(destination)
    frame = attach_early_causal_oracle_features(
        core,
        config.oracle_source,
        minimum_propagation_seconds=ORACLE_MINIMUM_PROPAGATION_SECONDS,
        maximum_age_seconds=ORACLE_MAXIMUM_AGE_SECONDS,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(destination, compression="zstd", statistics=True)
    write_json_atomic(
        metadata_path,
        {
            **identity,
            "rows": frame.height,
            "markets": frame["market_id"].n_unique(),
            "eligible_rows": int(frame["early_oracle_eligible"].sum()),
            "sha256": file_sha256(destination),
        },
    )
    return frame


def _oracle_source_inventory(
    core: pl.DataFrame,
    source: Path,
) -> dict[str, Any]:
    days = sorted(core["window_start"].dt.date().unique().to_list())
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for day in days:
        raw_path = source / f"{day.isoformat()}.parquet"
        oracle_path = source / f"oracle-{day.isoformat()}.parquet"
        if not raw_path.is_file() or not oracle_path.is_file():
            missing.append(day.isoformat())
            continue
        oracle = pl.read_parquet(oracle_path)
        causality_violations = oracle.filter(
            pl.col("oracle_source_timestamp") > pl.col("oracle_block_timestamp")
        ).height
        if causality_violations:
            raise RuntimeError(f"oracle source contains causal violations: {oracle_path}")
        records.append(
            {
                "date": day.isoformat(),
                "raw_path": raw_path.name,
                "raw_sha256": file_sha256(raw_path),
                "oracle_path": oracle_path.name,
                "oracle_sha256": file_sha256(oracle_path),
                "oracle_rows": oracle.height,
                "causality_violations": 0,
            }
        )
    payload = {
        "schema_version": "btc-asymmetric-value-oracle-source-inventory-v1",
        "expected_days": len(days),
        "available_days": len(records),
        "missing_days": missing,
        "records": records,
    }
    payload["inventory_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _load_or_build_external(
    core: pl.DataFrame,
    config: AsymmetricValueConfig,
    *,
    destination: Path,
    core_content_sha256: str,
    force: bool,
) -> pl.DataFrame:
    metadata_path = destination.with_suffix(".metadata.json")
    identity = {
        "schema_version": "btc-asymmetric-value-external-features-v3",
        "core_key_sha256": _frame_key_digest(core),
        "core_content_sha256": core_content_sha256,
        "core_rows": core.height,
        "core_markets": core["market_id"].n_unique(),
        "minimum_window_start": core["window_start"].min().isoformat(),
        "maximum_window_start": core["window_start"].max().isoformat(),
        "l2_metadata_sha256": _source_metadata_digest(config.l2_source),
        "candle_metadata_sha256": _source_metadata_digest(config.candle_source),
    }
    if destination.is_file() and not force:
        if not metadata_path.is_file():
            raise RuntimeError("external feature cache lacks a provenance manifest")
        metadata = json.loads(metadata_path.read_text())
        observed = {name: metadata.get(name) for name in identity}
        observed_sha256 = file_sha256(destination)
        if metadata.get("sha256") != observed_sha256:
            raise RuntimeError("external feature cache content hash changed")
        frame = pl.read_parquet(destination)
        if observed != identity:
            legacy_names = (
                "core_rows",
                "core_markets",
                "minimum_window_start",
                "maximum_window_start",
                "l2_metadata_sha256",
                "candle_metadata_sha256",
            )
            legacy_schema = metadata.get("schema_version")
            legacy_matches = legacy_schema in {
                "btc-asymmetric-value-external-features-v1",
                "btc-asymmetric-value-external-features-v2",
            } and all(metadata.get(name) == identity[name] for name in legacy_names)
            if not legacy_matches:
                raise RuntimeError(
                    "external feature cache provenance changed; rebuild intentionally"
                )
            _validate_external_core_keys(frame, core)
            write_json_atomic(
                metadata_path,
                {
                    **identity,
                    "rows": frame.height,
                    "markets": frame["market_id"].n_unique(),
                    "sha256": observed_sha256,
                    "upgraded_from_schema_version": metadata["schema_version"],
                },
            )
        return frame
    frame = build_partitioned_external_frame(core, config.l2_source, config.candle_source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(destination, compression="zstd", statistics=True)
    write_json_atomic(
        metadata_path,
        {
            **identity,
            "rows": frame.height,
            "markets": frame["market_id"].n_unique(),
            "sha256": file_sha256(destination),
        },
    )
    return frame


def _validate_external_core_keys(
    frame: pl.DataFrame,
    core: pl.DataFrame,
) -> None:
    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
    duplicates = frame.group_by(*keys).len().filter(pl.col("len") != 1)
    if duplicates.height:
        raise RuntimeError("external feature cache contains duplicate core keys")
    unmatched = frame.select(*keys).join(
        core.select(*keys),
        on=keys,
        how="anti",
    )
    if unmatched.height:
        raise RuntimeError("external feature cache contains keys outside the core cohort")
    missing_core_columns = sorted(set(core.columns) - set(frame.columns))
    if missing_core_columns:
        raise RuntimeError(
            "external feature cache lost core columns: "
            + ", ".join(missing_core_columns)
        )
    matching_core = core.join(
        frame.select(*keys),
        on=keys,
        how="semi",
    )
    if matching_core.height != frame.height:
        raise RuntimeError("external feature cache core-key cardinality changed")
    if _frame_content_digest(matching_core) != _frame_content_digest(
        frame.select(*core.columns)
    ):
        raise RuntimeError("external feature cache core values changed")


def _source_metadata_digest(source: Path) -> str:
    resolved = source.resolve()
    files = sorted(resolved.glob("*.parquet.json")) if resolved.is_dir() else []
    digest = hashlib.sha256()
    digest.update(str(resolved).encode())
    for path in files:
        digest.update(path.name.encode())
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def _frame_key_digest(frame: pl.DataFrame) -> str:
    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
    return _ordered_frame_digest(frame.select(*keys), keys)


def _frame_content_digest(frame: pl.DataFrame) -> str:
    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ValueError("frame content digest is missing keys: " + ", ".join(missing))
    return _ordered_frame_digest(frame, keys)


def _ordered_frame_digest(frame: pl.DataFrame, keys: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [(name, str(dtype)) for name, dtype in frame.schema.items()],
            separators=(",", ":"),
        ).encode()
    )
    keyed_hashes = (
        frame.select(*keys)
        .with_columns(frame.hash_rows(seed=20260804).alias("_row_hash"))
        .sort(*keys)
    )
    digest.update(keyed_hashes["_row_hash"].to_numpy().tobytes())
    return digest.hexdigest()


def _implementation_digest(config: AsymmetricValueConfig) -> str:
    source_root = Path(__file__).parent
    paths = [
        source_root / "asymmetric_value_benchmark.py",
        source_root / "asymmetric_value_config.py",
        source_root / "asymmetric_value_data.py",
        source_root / "asymmetric_value_evaluation.py",
        source_root / "asymmetric_value_training.py",
        source_root / "chainlink_oi_features.py",
        source_root / "core_config.py",
        source_root / "core_execution.py",
        source_root / "core_extract.py",
        source_root / "core_features.py",
        source_root / "core_training.py",
        source_root / "early_value_data.py",
        source_root / "early_value_training.py",
        source_root / "runtime_export.py",
        source_root / "spot_l2_chainlink_features.py",
        config.package_root / "pyproject.toml",
        config.price_source_sql,
        config.source_path,
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode())
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    return {
        package: version(package)
        for package in ("joblib", "numpy", "polars", "scikit-learn")
    }


def _current_process_contract(config: AsymmetricValueConfig) -> dict[str, Any]:
    process = json.loads(config.champion_process.read_text())
    model = json.loads(config.champion_model.read_text())
    raw = process["config"]["raw"]["btc_realtime_paper"]
    strategy = raw["strategy"]
    decision = strategy["decision_strategy"]
    paper = raw["paper"]
    runtime = raw["runtime"]
    prediction_policy = model["prediction_policy"]
    thresholds = {
        float(band["confidence_threshold"]) for band in model["time_bands"]
    }
    minimum_entry_second = int(strategy["min_seconds_after_open"])
    maximum_entry_second = 300 - int(strategy["min_seconds_before_close"])
    if (
        process.get("process_key")
        != "btc-5m-directional-model-paper-boundary-alignment"
        or decision.get("model_key") != model.get("model_key")
        or decision.get("artifact_sha256") != file_sha256(config.champion_model)
        or decision.get("feature_schema_sha256")
        != model["features"]["schema_sha256"]
        or paper.get("directional_model_entry_policy")
        != "execute_directional_prediction"
        or thresholds != {0.89}
        or float(strategy["target_size"]) != config.quantity
        or int(strategy["max_book_age_ms"])
        != config.book_freshness_seconds * 1_000
        or minimum_entry_second != 60
        or maximum_entry_second != 240
        or float(strategy["min_entry_price"]) != 0.30
        or float(strategy["max_entry_price"]) != 0.95
        or float(strategy["max_depth_participation"]) != 0.25
        or prediction_policy.get("type") != "first_confidence_crossing"
        or int(prediction_policy["cadence_seconds"]) != 5
        or int(prediction_policy["minimum_seconds_after_open"])
        != minimum_entry_second
        or int(prediction_policy["maximum_seconds_after_open"])
        != maximum_entry_second
        or int(runtime["strategy_interval_ms"]) != 1_000
    ):
        raise RuntimeError("frozen current-process reference contract changed")
    return {
        "process_key": process["process_key"],
        "model_key": decision["model_key"],
        "entry_policy": paper["directional_model_entry_policy"],
        "confidence_threshold": thresholds.pop(),
        "quantity": float(strategy["target_size"]),
        "minimum_entry_second": minimum_entry_second,
        "maximum_entry_second": maximum_entry_second,
        "minimum_share_price": float(strategy["min_entry_price"]),
        "maximum_share_price": float(strategy["max_entry_price"]),
        "maximum_depth_participation": float(
            strategy["max_depth_participation"]
        ),
        "maximum_book_age_seconds": int(strategy["max_book_age_ms"]) / 1_000,
        "model_prediction_cadence_seconds": int(
            prediction_policy["cadence_seconds"]
        ),
        "runtime_strategy_interval_ms": int(runtime["strategy_interval_ms"]),
        "terminal_first_crossing_before_execution_validation": True,
        "offline_emulation_adds_edge_gate": False,
    }


def _prediction_surface(
    frames: dict[str, pl.DataFrame],
    models: dict[str, Any],
) -> pl.DataFrame:
    if set(frames) != set(models):
        raise ValueError("prediction frames and model bundles must have identical keys")
    return pl.concat(
        [
            asymmetric_probability_frame(
                frames[name],
                bundle.probability(frames[name]),
                model=name,
            )
            for name, bundle in models.items()
        ],
        how="vertical_relaxed",
    )


def _champion_probability_frame(
    config: AsymmetricValueConfig,
    frame: pl.DataFrame,
    *,
    include_execution: bool = False,
) -> pl.DataFrame:
    model = json.loads(config.champion_model.read_text())
    eligible = frame.filter(pl.col("seconds_elapsed") >= 60)
    names = tuple(model["features"]["names"])
    if eligible.is_empty() or any(name not in eligible.columns for name in names):
        raise RuntimeError("frozen champion cannot be scored on asymmetric-value evaluation rows")
    matrix = eligible.select(*names).to_numpy()
    probability = np.array(
        [
            score_runtime_model(model, row.tolist(), seconds_elapsed=int(second))["probability_up"]
            for row, second in zip(matrix, eligible["seconds_elapsed"], strict=True)
        ],
        dtype=np.float64,
    )
    if include_execution:
        return asymmetric_probability_frame(
            eligible,
            probability,
            model=FROZEN_CHAMPION,
        )
    return eligible.select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
    ).with_columns(
        pl.lit(FROZEN_CHAMPION).alias("model"),
        pl.Series("probability_yes", probability),
    )


def _selected_matched_control(model: str) -> str | None:
    if model == CORE_ORACLE_PRICE:
        return ORACLE_MATCHED_CORE_PRICE_CONTROL
    if model in {CORE_L2_PRICE, CORE_CANDLES_PRICE, CORE_L2_CANDLES_PRICE}:
        return L2_CANDLES_MATCHED_CORE_PRICE_CONTROL
    if model == CORE_ORACLE_L2_CANDLES_PRICE:
        return ORACLE_L2_CANDLES_MATCHED_CORE_ORACLE_PRICE_CONTROL
    if model == CORE_PRICE:
        return PAIRED_CORE_CONTROL
    return None


def _current_reference_crossing_diagnostics(
    predictions: pl.DataFrame,
    ledger: pl.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    crossings = (
        predictions.with_columns(
            pl.max_horizontal(
                "probability_yes",
                1.0 - pl.col("probability_yes"),
            ).alias("confidence")
        )
        .filter(
            (pl.col("confidence") >= contract["confidence_threshold"])
            & pl.col("seconds_elapsed").is_between(
                contract["minimum_entry_second"],
                contract["maximum_entry_second"],
                closed="both",
            )
        )
        .group_by("model", "market_id")
        .first()
    )
    crossing_markets = crossings["market_id"].n_unique()
    executed_markets = ledger["market_id"].n_unique() if not ledger.is_empty() else 0
    return {
        "eligible_markets": predictions["market_id"].n_unique(),
        "terminal_confidence_crossings": crossing_markets,
        "executed_after_exact_book_validation": executed_markets,
        "rejected_or_missing_at_terminal_crossing": (
            crossing_markets - executed_markets
        ),
    }


def _coverage_summary(
    core: pl.DataFrame,
    external: pl.DataFrame,
    strict: pl.DataFrame,
    config: AsymmetricValueConfig,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, evidence in (
        ("fit", config.fit),
        ("calibration", config.calibration),
        ("policy", config.policy),
        ("evaluation", config.evaluation),
    ):
        core_rows = _window(core, evidence.start, evidence.end)
        external_rows = _window(external, evidence.start, evidence.end)
        strict_rows = _window(strict, evidence.start, evidence.end)
        core_markets = core_rows["market_id"].n_unique()
        external_markets = external_rows["market_id"].n_unique()
        strict_markets = strict_rows["market_id"].n_unique()
        output[name] = {
            "core_resolved_rows": core_rows.height,
            "core_resolved_markets": core_markets,
            "external_feature_qualified_rows": external_rows.height,
            "external_feature_qualified_markets": external_markets,
            "strict_rows": strict_rows.height,
            "strict_markets": strict_markets,
            "strict_executable_utc_days": (
                strict_rows["window_start"].dt.date().n_unique()
                if strict_rows.height
                else 0
            ),
            "strict_row_coverage": (
                strict_rows.height / core_rows.height if core_rows.height else 0.0
            ),
            "external_market_coverage": (
                external_markets / core_markets if core_markets else 0.0
            ),
            "strict_market_coverage": (
                strict_markets / core_markets if core_markets else 0.0
            ),
        }
    return output


def _risk_shape_comparison(
    candidate: dict[str, Any],
    champion: dict[str, Any],
) -> dict[str, Any]:
    candidate_cost = candidate.get("mean_admission_cost_per_share")
    champion_cost = champion.get("mean_admission_cost_per_share")
    candidate_recovery = candidate.get("loss_recovery_wins")
    champion_recovery = champion.get("loss_recovery_wins")
    candidate_efficiency = candidate.get("capital_efficiency")
    champion_efficiency = champion.get("capital_efficiency")
    candidate_share_price = candidate.get("mean_share_price")
    champion_share_price = champion.get("mean_share_price")
    candidate_stress = candidate.get("stress_1c_net_expectancy_per_trade")
    champion_stress = champion.get("stress_1c_net_expectancy_per_trade")
    return {
        "mean_cost_reduced": bool(
            candidate_cost is not None and champion_cost is not None and candidate_cost < champion_cost
        ),
        "loss_recovery_improved": bool(
            candidate_recovery is not None
            and champion_recovery is not None
            and candidate_recovery < champion_recovery
        ),
        "capital_efficiency_improved": bool(
            candidate_efficiency is not None
            and champion_efficiency is not None
            and candidate_efficiency > champion_efficiency
        ),
        "mean_share_price_reduced": bool(
            candidate_share_price is not None
            and champion_share_price is not None
            and candidate_share_price < champion_share_price
        ),
        "stress_expectancy_improved": bool(
            candidate_stress is not None
            and champion_stress is not None
            and candidate_stress > champion_stress
        ),
        "candidate_mean_cost_per_share": candidate_cost,
        "champion_mean_cost_per_share": champion_cost,
        "candidate_loss_recovery_wins": candidate_recovery,
        "champion_loss_recovery_wins": champion_recovery,
        "candidate_capital_efficiency": candidate_efficiency,
        "champion_capital_efficiency": champion_efficiency,
        "candidate_mean_share_price": candidate_share_price,
        "champion_mean_share_price": champion_share_price,
        "candidate_stress_1c_expectancy": candidate_stress,
        "champion_stress_1c_expectancy": champion_stress,
    }


def _add_opportunity_denominators(
    metrics: dict[str, dict[str, Any]],
    *,
    resolved_markets: int,
    strict_markets_by_model: dict[str, int],
) -> None:
    for key, values in metrics.items():
        model = key.split("::", maxsplit=1)[0]
        strict_markets = strict_markets_by_model[model]
        net_profit = float(values.get("net_profit", 0.0))
        trades = int(values.get("trades", 0))
        values.update(
            {
                "resolved_markets": resolved_markets,
                "strict_executable_markets": strict_markets,
                "strict_market_coverage": (
                    strict_markets / resolved_markets if resolved_markets else 0.0
                ),
                "trades_per_resolved_market": (
                    trades / resolved_markets if resolved_markets else 0.0
                ),
                "net_profit_per_resolved_market": (
                    net_profit / resolved_markets if resolved_markets else 0.0
                ),
            }
        )


def _confidence_threshold_controls(
    policy_predictions: pl.DataFrame,
    evaluation_predictions: pl.DataFrame,
    config: AsymmetricValueConfig,
) -> dict[str, Any]:
    """Benchmark the retrained champion family at conventional confidence gates."""

    policy_source = policy_predictions.filter(pl.col("model") == CORE_CONTROL)
    evaluation_source = evaluation_predictions.filter(pl.col("model") == CORE_CONTROL)
    policy_metrics: dict[str, dict[str, Any]] = {}
    evaluation_metrics: dict[str, dict[str, Any]] = {}
    table: list[dict[str, Any]] = []
    for offset, threshold in enumerate(config.confidence_thresholds):
        name = f"{threshold:.2f}"
        policy_ledger = confidence_control_ledger(
            policy_source,
            threshold=threshold,
            maximum_entry_second=240,
            maximum_cost_per_share=config.gates.maximum_mean_cost_per_share,
            minimum_edge_per_share=(
                config.confidence_control_minimum_edge_per_share
            ),
            quantity=config.quantity,
        )
        evaluation_ledger = confidence_control_ledger(
            evaluation_source,
            threshold=threshold,
            maximum_entry_second=240,
            maximum_cost_per_share=config.gates.maximum_mean_cost_per_share,
            minimum_edge_per_share=(
                config.confidence_control_minimum_edge_per_share
            ),
            quantity=config.quantity,
        )
        policy_values = ledger_metrics(policy_ledger)
        evaluation_values = ledger_metrics(evaluation_ledger)
        policy_values["utc_day_block_bootstrap"] = bootstrap_ledger_metrics(
            policy_ledger,
            resamples=config.bootstrap_resamples,
            seed=config.random_seed + 30_000 + offset,
        )
        evaluation_values["utc_day_block_bootstrap"] = bootstrap_ledger_metrics(
            evaluation_ledger,
            resamples=config.bootstrap_resamples,
            seed=config.random_seed + 40_000 + offset,
        )
        policy_metrics[name] = policy_values
        evaluation_metrics[name] = evaluation_values
        for role, values in (
            ("policy", policy_values),
            ("evaluation", evaluation_values),
        ):
            table.append(
                {
                    "model": CORE_CONTROL,
                    "window": role,
                    "confidence_threshold": threshold,
                    "maximum_cost_per_share": (
                        config.gates.maximum_mean_cost_per_share
                    ),
                    "trades": values.get("trades"),
                    "accuracy": values.get("accuracy"),
                    "mean_cost_per_share": values.get(
                        "mean_admission_cost_per_share"
                    ),
                    "net_profit": values.get("net_profit"),
                    "net_expectancy_per_trade": values.get(
                        "net_expectancy_per_trade"
                    ),
                    "capital_efficiency": values.get("capital_efficiency"),
                    "profit_factor": values.get("profit_factor"),
                    "loss_recovery_wins": values.get("loss_recovery_wins"),
                    "mean_entry_second": values.get("mean_entry_second"),
                }
            )
    return {
        "model": CORE_CONTROL,
        "description": (
            "diagnostic same-core-feature family retrained for 5-240s, with a 70c "
            "cost cap, 1.5c conservative edge guard, and conventional confidence "
            "thresholds; this is not a replay of the full live pipeline"
        ),
        "accuracy_thresholds_are_controls_not_selection_gates": True,
        "policy_window": policy_metrics,
        "evaluation_window": evaluation_metrics,
        "table": table,
    }


def _frozen_champion_threshold_controls(
    predictions: pl.DataFrame,
    execution: pl.DataFrame,
    contract: dict[str, Any],
    config: AsymmetricValueConfig,
) -> dict[str, Any]:
    """Vary only the frozen champion confidence threshold on untouched evidence."""

    metrics: dict[str, dict[str, Any]] = {}
    table: list[dict[str, Any]] = []
    for offset, threshold in enumerate(config.confidence_thresholds):
        ledger = current_policy_reference_ledger(
            predictions,
            execution=execution,
            threshold=threshold,
            minimum_entry_second=contract["minimum_entry_second"],
            maximum_entry_second=contract["maximum_entry_second"],
            minimum_share_price=contract["minimum_share_price"],
            maximum_share_price=contract["maximum_share_price"],
            maximum_depth_participation=contract[
                "maximum_depth_participation"
            ],
            quantity=config.quantity,
        )
        values = ledger_metrics(ledger)
        values["utc_day_block_bootstrap"] = bootstrap_ledger_metrics(
            ledger,
            resamples=config.bootstrap_resamples,
            seed=config.random_seed + 50_000 + offset,
        )
        values["terminal_crossing_diagnostics"] = (
            _current_reference_crossing_diagnostics(
                predictions,
                ledger,
                {**contract, "confidence_threshold": threshold},
            )
        )
        name = f"{threshold:.2f}"
        metrics[name] = values
        table.append(
            {
                "model": FROZEN_CHAMPION,
                "confidence_threshold": threshold,
                "minimum_entry_second": contract["minimum_entry_second"],
                "maximum_entry_second": contract["maximum_entry_second"],
                "minimum_share_price": contract["minimum_share_price"],
                "maximum_share_price": contract["maximum_share_price"],
                "trades": values.get("trades"),
                "accuracy": values.get("accuracy"),
                "mean_share_price": values.get("mean_share_price"),
                "mean_entry_second": values.get("mean_entry_second"),
                "net_profit": values.get("net_profit"),
                "net_expectancy_per_trade": values.get(
                    "net_expectancy_per_trade"
                ),
                "stress_1c_net_expectancy_per_trade": values.get(
                    "stress_1c_net_expectancy_per_trade"
                ),
                "capital_efficiency": values.get("capital_efficiency"),
                "profit_factor": values.get("profit_factor"),
                "average_loss": values.get("average_loss"),
                "loss_recovery_wins": values.get("loss_recovery_wins"),
            }
        )
    return {
        "model": FROZEN_CHAMPION,
        "description": (
            "hypothetical threshold-only variants of the frozen champion; all other "
            "30-95c, 60-240s, terminal-first-crossing, depth, and sizing rules remain fixed"
        ),
        "threshold_change_authorized_for_runtime": False,
        "evaluation_window": metrics,
        "table": table,
    }


def _bootstrap_reference(
    ledger: pl.DataFrame,
    config: AsymmetricValueConfig,
) -> dict[str, Any] | None:
    return bootstrap_ledger_metrics(
        ledger,
        resamples=config.bootstrap_resamples,
        seed=config.random_seed + 20_000,
    )


def _policy_table(
    metrics: dict[str, dict[str, Any]],
    frontier: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    qualification = {record["key"]: record for record in frontier}
    rows: list[dict[str, Any]] = []
    for key, values in sorted(metrics.items()):
        model, policy = key.split("::", maxsplit=1)
        bootstrap = values.get("utc_day_block_bootstrap") or {}
        rows.append(
            {
                "model": model,
                "policy": policy,
                "selection_eligible": qualification[key]["selection_eligible"],
                "qualified": qualification[key]["qualified"],
                "trades": values.get("trades"),
                "markets": values.get("markets"),
                "utc_days": values.get("utc_days"),
                "accuracy": values.get("accuracy"),
                "mean_share_price": values.get("mean_share_price"),
                "mean_admission_cost_per_share": values.get(
                    "mean_admission_cost_per_share"
                ),
                "net_profit": values.get("net_profit"),
                "net_expectancy_per_trade": values.get("net_expectancy_per_trade"),
                "stress_1c_net_expectancy_per_trade": values.get(
                    "stress_1c_net_expectancy_per_trade"
                ),
                "capital_efficiency": values.get("capital_efficiency"),
                "profit_factor": values.get("profit_factor"),
                "average_win": values.get("average_win"),
                "average_loss": values.get("average_loss"),
                "loss_recovery_wins": values.get("loss_recovery_wins"),
                "maximum_drawdown": values.get("maximum_drawdown"),
                "yes_net_profit": values.get("yes_net_profit"),
                "no_net_profit": values.get("no_net_profit"),
                "selected_calibration_bias": values.get(
                    "selected_calibration_bias"
                ),
                "underdog_trade_share": values.get("underdog_trade_share"),
                "pre60_trades": values.get("pre60_trades"),
                "twenty_to_thirty_cent_trades": values.get(
                    "twenty_to_thirty_cent_trades"
                ),
                "mean_entry_second": values.get("mean_entry_second"),
                "bootstrap_lower_95_expectancy_per_trade": (
                    (bootstrap.get("net_expectancy_per_trade") or {}).get(
                        "lower_95"
                    )
                ),
                "bootstrap_lower_95_capital_efficiency": (
                    (bootstrap.get("capital_efficiency") or {}).get("lower_95")
                ),
                "trades_per_resolved_market": values.get(
                    "trades_per_resolved_market"
                ),
                "net_profit_per_resolved_market": values.get(
                    "net_profit_per_resolved_market"
                ),
            }
        )
    return rows


def _markdown_report(result: dict[str, Any]) -> str:
    selected = result["evaluation"]["selected_metrics"]
    paired_champion = result["evaluation"][
        "frozen_current_policy_market_paired_89"
    ]
    full_champion = result["evaluation"][
        "frozen_current_policy_full_exact_book_89"
    ]
    matched_control = result["evaluation"][
        "selected_feature_attribution_control"
    ]
    comparison = result["evaluation"]["risk_shape_comparison"]
    evaluation = result["evaluation"]
    coverage = result["data"]["coverage"]["evaluation"]
    candidate_grid = coverage["candidate_prediction_grid"][
        result["selection"]["selected_model"]
    ]
    supported = result["hypothesis"]["alleviates_expensive_failure_risk"]
    selected_frontier = next(
        record
        for record in result["selection"]["frontier"]
        if record["key"] == evaluation["selected_key"]
    )
    selected_label = (
        "qualified hunter" if evaluation["qualified"] else "selected diagnostic candidate"
    )
    lines = [
        "# BTC asymmetric-value hunter benchmark",
        "",
        "This is consumed offline development evidence. It does not authorize live capital.",
        "",
        (
            "**Hypothesis supported by qualification evidence.**"
            if supported
            else "**Hypothesis not supported by qualification evidence in this run.**"
        ),
        "",
        f"{selected_label.capitalize()}: `{evaluation['selected_key']}`.",
        f"Policy-window qualification: `{result['selection']['qualified_on_policy_window']}`.",
        f"Frozen evaluation status: `{evaluation['status']}`.",
        "",
        "The primary search is restricted to raw share prices below 30 cents. Accuracy is a",
        "reported property, not an entry threshold; entry requires calibrated edge over all-in cost.",
        "",
        "## Policy selection versus frozen evaluation",
        "",
        "| Window | Trades | Accuracy | Raw share | EV/trade | +1c EV/trade | Profit factor | Calibration bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (
        ("Policy selection", selected_frontier["metrics"]),
        ("Frozen evaluation", selected),
    ):
        lines.append(
            f"| {name} | {metrics.get('trades')} | {_fmt(metrics.get('accuracy'))} | "
            f"{_fmt(metrics.get('mean_share_price'))} | "
            f"{_fmt(metrics.get('net_expectancy_per_trade'))} | "
            f"{_fmt(metrics.get('stress_1c_net_expectancy_per_trade'))} | "
            f"{_fmt(metrics.get('profit_factor'))} | "
            f"{_fmt(metrics.get('selected_calibration_bias'))} |"
        )

    lines.extend(
        [
            "",
            "## Selected candidate versus current 89% execute-directional policy shape",
            "",
            "The selected and market-paired current-policy rows share the selected model's market universe. The full row shows the wider exact-book universe. These are not promotion claims when qualification is false.",
            "",
            "| Strategy | Trades | Accuracy | Raw share | All-in admission | EV/trade | EV/trade +1c stress | Avg win | Avg loss | Wins/loss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in (
        ("Selected hunter", selected),
        ("Current 89% (market-paired)", paired_champion),
        ("Current 89% (full exact-book)", full_champion),
    ):
        lines.append(
            f"| {name} | {metrics.get('trades')} | {_fmt(metrics.get('accuracy'))} | "
            f"{_fmt(metrics.get('mean_share_price'))} | "
            f"{_fmt(metrics.get('mean_admission_cost_per_share'))} | "
            f"{_fmt(metrics.get('net_expectancy_per_trade'))} | "
            f"{_fmt(metrics.get('stress_1c_net_expectancy_per_trade'))} | "
            f"{_fmt(metrics.get('average_win'))} | {_fmt(metrics.get('average_loss'))} | "
            f"{_fmt(metrics.get('loss_recovery_wins'))} |"
        )

    lines.extend(
        [
            "",
            "## Frozen champion with threshold-only variants",
            "",
            "This isolates the user's threshold question: the trained frozen model is unchanged, and only confidence admission varies. Its original feature contract still cannot score before 60 seconds, and its 30-cent minimum price remains fixed.",
            "",
            "| Threshold | Trades | Accuracy | Mean entry second | Raw share | EV/trade | +1c EV/trade | Avg loss | Wins/loss |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["evaluation"]["frozen_champion_confidence_thresholds"][
        "table"
    ]:
        lines.append(
            f"| {_fmt(row['confidence_threshold'])} | {row['trades']} | "
            f"{_fmt(row['accuracy'])} | {_fmt(row['mean_entry_second'])} | "
            f"{_fmt(row['mean_share_price'])} | "
            f"{_fmt(row['net_expectancy_per_trade'])} | "
            f"{_fmt(row['stress_1c_net_expectancy_per_trade'])} | "
            f"{_fmt(row['average_loss'])} | "
            f"{_fmt(row['loss_recovery_wins'])} |"
        )

    if matched_control["model"] is not None:
        matched = matched_control["metrics"]
        lines.extend(
            [
                "",
                "## Untouched feature-attribution control",
                "",
                f"The selected enriched arm is compared on identical evaluation keys with `{matched_control['model']}`.",
                "",
                "| Model | Trades | Accuracy | Raw share | EV/trade | +1c EV/trade |",
                "|---|---:|---:|---:|---:|---:|",
                (
                    f"| Selected | {selected.get('trades')} | {_fmt(selected.get('accuracy'))} | "
                    f"{_fmt(selected.get('mean_share_price'))} | "
                    f"{_fmt(selected.get('net_expectancy_per_trade'))} | "
                    f"{_fmt(selected.get('stress_1c_net_expectancy_per_trade'))} |"
                ),
                (
                    f"| Matched control | {matched.get('trades')} | {_fmt(matched.get('accuracy'))} | "
                    f"{_fmt(matched.get('mean_share_price'))} | "
                    f"{_fmt(matched.get('net_expectancy_per_trade'))} | "
                    f"{_fmt(matched.get('stress_1c_net_expectancy_per_trade'))} |"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Evidence sufficiency",
            "",
            f"Selected-candidate markets: `{candidate_grid['candidate_markets']}`; selected-candidate UTC days: `{candidate_grid['candidate_utc_days']}`.",
            f"Selected-candidate prediction-grid coverage: `{_fmt(candidate_grid['prediction_grid_coverage'])}`; weakest decision-second market coverage: `{_fmt(candidate_grid['minimum_second_market_coverage'])}`.",
            f"Exact source-grid retained coverage: `{_fmt(coverage['source_grid']['retained_coverage'])}`; strict two-sided grid coverage: `{_fmt(coverage['source_grid']['strict_coverage'])}`.",
            f"Exact-price market coverage of all resolved core markets: `{_fmt(coverage['strict_market_coverage'])}`; L2/candle feature coverage: `{_fmt(coverage['external_market_coverage'])}`.",
            "Missing exact books are NoTrade and remain in the resolved-market denominator; they are not proxy prices or losses.",
            "",
            "## Raw share-price bands for selected trades",
            "",
            "| Band | Trades | Accuracy | Mean raw share | EV/trade |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in evaluation["selected_price_bands"]:
        lines.append(
            f"| {row['price_band']} | {row['trades']} | {_fmt(row['accuracy'])} | "
            f"{_fmt(row['mean_share_price'])} | "
            f"{_fmt(row['net_expectancy_per_trade'])} |"
        )

    lines.extend(
        [
            "",
            "## Accuracy and price at key decision seconds",
            "",
            "| Second | Markets | Argmax accuracy | Value-side accuracy | Mean raw share | Mean modeled edge |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    key_seconds = {5, 30, 55, 60, 120, 180, 240}
    for row in evaluation["accuracy_price_by_five_seconds"]:
        if row["seconds_elapsed"] not in key_seconds:
            continue
        lines.append(
            f"| {row['seconds_elapsed']} | {row['markets']} | "
            f"{_fmt(row['argmax_accuracy'])} | {_fmt(row['value_side_accuracy'])} | "
            f"{_fmt(row['mean_selected_share_price'])} | "
            f"{_fmt(row['mean_selected_edge_per_share'])} |"
        )

    lines.extend(
        [
            "",
            "## Core objective",
            "",
            "The following comparisons are descriptive point estimates; they do not override failed gates:",
            "",
            f"Mean raw share price reduced: `{comparison['mean_share_price_reduced']}`.  ",
            f"Mean all-in cost reduced: `{comparison['mean_cost_reduced']}`.  ",
            f"Loss recovery improved: `{comparison['loss_recovery_improved']}`.  ",
            f"Capital efficiency improved: `{comparison['capital_efficiency_improved']}`.",
            "",
            "The model may deliberately select the lower-probability side when price creates the",
            "larger calibrated edge; that is the intended 21%-probability-at-8c payoff pattern.",
            "The current-policy reference preserves terminal first-crossing semantics before",
            "execution validation. It uses the model's five-second feature cadence; the live",
            "process loop is one second, so intermediate-loop quote state is not reconstructed.",
            "Decision-time VWAP5 is not a guaranteed fill. Baseline and +1c/share stress economics",
            "are reported, and complete fresh forward evidence is required before any runtime change.",
            "",
            "See the CSV artifacts for every five-second accuracy point, every one-second price",
            "point before 60 seconds, the confidence-threshold controls, the selected-side",
            "surface, and the explicit YES/NO model × second × raw-price-band surface.",
            "",
        ]
    )
    return "\n".join(lines)


def _window(frame: pl.DataFrame, start: Any, end: Any) -> pl.DataFrame:
    return frame.filter(pl.col("window_start").is_between(start, end, closed="left"))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"
