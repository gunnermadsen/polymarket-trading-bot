"""End-to-end paper benchmark for lower-cost asymmetric-value opportunities."""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from .asymmetric_value_config import AsymmetricValueConfig, EvidenceWindow
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
    load_asymmetric_retained_execution_evidence,
    select_asymmetric_prediction_grid,
)
from .asymmetric_value_evaluation import (
    accuracy_price_by_second,
    add_bootstrap_metrics,
    bootstrap_ledger_metrics,
    candidate_policy_key,
    confidence_control_ledger,
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
    ASYMMETRIC_VALUE_CANDIDATES,
    ASYMMETRIC_VALUE_MODEL_MATRIX,
    CANDLE_MATCHED_CORE_PRICE_CONTROL,
    CORE_CANDLES_PRICE,
    CORE_L2_PRICE,
    CORE_ORACLE_L2_PRICE,
    CORE_ORACLE_PRICE,
    CORE_PRICE,
    EXPECTED_MODEL_FEATURE_COUNTS,
    L2_MATCHED_CORE_PRICE_CONTROL,
    MATCHED_ATTRIBUTION_CONTROLS,
    MODEL_SELECTION_ELIGIBLE,
    OFFLINE_ONLY_CANDIDATES,
    ORACLE_MATCHED_CORE_PRICE_CONTROL,
    PRICE_LOGISTIC,
    THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL,
    asymmetric_probability_frame,
    asymmetric_value_feature_sets,
    fit_asymmetric_value_models,
    target_calibration_gate_checks,
)
from .core_config import load_core_config
from .core_extract import extract_core_source, file_sha256, write_json_atomic
from .core_features import (
    build_core_features,
    feature_destination,
    validate_core_feature_cache,
)
from .early_value_data import (
    build_full_closed_candle_frame,
    build_partitioned_l2_frame,
)
from .runtime_export import score_runtime_model
from .spot_l2_chainlink_features import L2_FEATURES

ASYMMETRIC_VALUE_SCHEMA_VERSION = "btc-asymmetric-value-hunter-benchmark-v3"
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
    development = _load_asymmetric_core_grid(
        core_config,
        "pre_holdout",
        config,
    )
    development_core_content_sha256 = _frame_content_digest(development)

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
    development_executable_core = development_price_features.select(
        *development.columns
    )
    development_coverage = _base_coverage_summary(
        development,
        development_price_features,
        config,
    )
    development_price_features = _project_candidate_source(
        development_price_features,
        PRICE_LOGISTIC,
        CORE_PRICE,
    )
    development_oracle_inventory = _oracle_source_inventory(
        development_executable_core,
        config.oracle_source,
    )
    development_oracle = _load_or_build_oracle_core(
        development_executable_core,
        config,
        destination=config.feature_cache / DEVELOPMENT_ORACLE_CACHE,
        source_inventory=development_oracle_inventory,
        core_content_sha256=development_core_content_sha256,
        force=force,
    )
    development_oracle_price_features = _project_candidate_source(
        attach_asymmetric_value_features(
            development_oracle,
            development_prices,
            config,
        ).filter(pl.col("early_oracle_eligible")),
        ORACLE_MATCHED_CORE_PRICE_CONTROL,
        CORE_ORACLE_PRICE,
    )
    del development_oracle
    gc.collect()

    development_l2 = _load_or_build_source_features(
        development_executable_core,
        config,
        source_family="l2",
        destination=config.feature_cache / "development-l2.parquet",
        core_content_sha256=development_core_content_sha256,
        force=force,
    )
    _add_joint_source_coverage(
        development_coverage,
        development_l2,
        config,
        source_family="l2",
    )
    development_l2_price_features = _project_candidate_source(
        attach_asymmetric_value_features(
            development_l2,
            development_prices,
            config,
        ),
        L2_MATCHED_CORE_PRICE_CONTROL,
        CORE_L2_PRICE,
    )
    del development_l2
    gc.collect()
    development_three_source_price_features = _join_oracle_l2_candidate_features(
        development_oracle_price_features,
        development_l2_price_features,
    )
    _add_joint_source_coverage(
        development_coverage,
        development_three_source_price_features,
        config,
        source_family="oracle_l2",
    )
    development_three_source_price_features = _project_candidate_source(
        development_three_source_price_features,
        THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL,
        CORE_ORACLE_L2_PRICE,
    )

    development_candles = _load_or_build_source_features(
        development_executable_core,
        config,
        source_family="candles",
        destination=config.feature_cache / "development-candles.parquet",
        core_content_sha256=development_core_content_sha256,
        force=force,
    )
    _add_joint_source_coverage(
        development_coverage,
        development_candles,
        config,
        source_family="candle",
    )
    development_candle_price_features = _project_candidate_source(
        attach_asymmetric_value_features(
            development_candles,
            development_prices,
            config,
        ),
        CANDLE_MATCHED_CORE_PRICE_CONTROL,
        CORE_CANDLES_PRICE,
    )
    del development_candles
    policy_core = _window(
        development,
        config.policy.start,
        config.policy.end,
    ).select("market_id", "window_start", "seconds_elapsed")
    policy_resolved_markets = policy_core["market_id"].n_unique()
    policy_grid_coverage = execution_grid_coverage(
        config,
        scope="development",
        core=policy_core,
    )
    development_model_frames = _candidate_frames(
        price=development_price_features,
        oracle_price=development_oracle_price_features,
        l2_price=development_l2_price_features,
        candle_price=development_candle_price_features,
        three_source_price=development_three_source_price_features,
    )
    del (
        development,
        development_executable_core,
        development_prices,
    )
    gc.collect()

    print("asymmetric-value: fitting price-aware champion-family candidates", flush=True)
    models, training = fit_asymmetric_value_models(
        development_model_frames,
        config,
        core_config,
    )
    policy_frames = {
        name: _window(frame, config.policy.start, config.policy.end)
        for name, frame in development_model_frames.items()
    }
    policy_candidate_strict_markets = {
        name: frame["market_id"].n_unique()
        for name, frame in policy_frames.items()
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
        resolved_markets=policy_resolved_markets,
        strict_markets_by_model=policy_candidate_strict_markets,
    )
    policy_confidence_controls = _confidence_threshold_window(
        policy_predictions,
        config,
        window="policy",
        seed_offset=30_000,
        resolved_markets=policy_resolved_markets,
        strict_markets=policy_candidate_strict_markets[CORE_PRICE],
    )
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
    for name, profile in training["profiles"].items():
        policy_evidence_checks_by_model[name].extend(
            target_calibration_gate_checks(profile)
        )
    primary_policy = next(
        policy for policy in config.policies if policy.selection_eligible
    )
    policy_feature_attribution = _matched_feature_attribution(
        policy_metrics,
        policy_ledgers,
        policy_frames,
        policy_name=primary_policy.name,
        config=config,
        window=config.policy,
        seed_offset=26_000,
    )
    matched_control_checks, eligible_models = (
        _matched_control_noninferiority_checks(
            policy_metrics,
            policy_ledgers,
            primary_policy.name,
            config,
        )
    )
    for name, check in matched_control_checks.items():
        policy_evidence_checks_by_model[name].extend(check)
    selection = select_policy_candidate(
        policy_metrics,
        config,
        evidence_checks_by_model=policy_evidence_checks_by_model,
        eligible_models=eligible_models,
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
    write_json_atomic(
        run_dir / "policy-feature-attribution.json",
        policy_feature_attribution,
    )
    selection_seal = {
        "schema_version": "btc-asymmetric-value-selection-seal-v3",
        "sealed_at": datetime.now(UTC).isoformat(),
        "evaluation_opened": False,
        "selected_key": selection["selected_key"],
        "selected_model": selection["selected_model"],
        "selected_policy": selection["selected_policy"],
        "predeclared_evaluation_models": list(ASYMMETRIC_VALUE_CANDIDATES),
        "evaluation_policy_contract": {
            "policy": selection["selected_policy"],
            "quantity": config.quantity,
            "maximum_depth_participation": (
                config.maximum_depth_participation
            ),
            "same_policy_applied_to_every_trained_model": True,
            "qualification_limited_to_policy_window_selected_model": True,
        },
        "predeclared_frozen_evaluation_diagnostics": [
            "frozen_current_policy_full_exact_book_89",
            "frozen_current_policy_market_paired_89",
            "frozen_champion_low_price_value_60s_240s",
            "core_price_confidence_threshold_controls",
        ],
        "confidence_threshold_control_contract": {
            "model": CORE_PRICE,
            "thresholds": list(config.confidence_thresholds),
            "minimum_entry_second": 1,
            "maximum_entry_second": 240,
            "maximum_cost_per_share": (
                config.gates.maximum_mean_cost_per_share
            ),
            "minimum_edge_per_share": (
                config.confidence_control_minimum_edge_per_share
            ),
            "quantity": config.quantity,
            "maximum_depth_participation": (
                config.maximum_depth_participation
            ),
            "policy_window_metrics": policy_confidence_controls["metrics"],
        },
        "qualified_on_policy_window": selection["qualified_on_policy_window"],
        "policy_evidence_checks_by_model": policy_evidence_checks_by_model,
        "policy_source_grid": policy_grid_coverage,
        "policy_candidate_grid": policy_candidate_coverage,
        "matched_control_noninferiority": matched_control_checks,
        "policy_cohort_key_sha256": {
            name: _frame_key_digest(frame) for name, frame in policy_frames.items()
        },
        "policy_feature_attribution": policy_feature_attribution,
        "policy_feature_attribution_sha256": file_sha256(
            run_dir / "policy-feature-attribution.json"
        ),
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
        "development_l2_features_sha256": file_sha256(
            config.feature_cache / "development-l2.parquet"
        ),
        "development_l2_metadata_sha256": file_sha256(
            config.feature_cache / "development-l2.metadata.json"
        ),
        "development_candle_features_sha256": file_sha256(
            config.feature_cache / "development-candles.parquet"
        ),
        "development_candle_metadata_sha256": file_sha256(
            config.feature_cache / "development-candles.metadata.json"
        ),
        "development_oracle_features_sha256": file_sha256(
            config.feature_cache / DEVELOPMENT_ORACLE_CACHE
        ),
        "development_oracle_metadata_sha256": file_sha256(
            (config.feature_cache / DEVELOPMENT_ORACLE_CACHE).with_suffix(
                ".metadata.json"
            )
        ),
        "development_oracle_source_inventory": development_oracle_inventory,
        "model_artifact_sha256": model_hashes,
    }
    write_json_atomic(run_dir / "selection-seal.json", selection_seal)
    selection_seal_sha256 = file_sha256(run_dir / "selection-seal.json")
    del (
        development_price_features,
        development_oracle_price_features,
        development_l2_price_features,
        development_candle_price_features,
        development_three_source_price_features,
        development_model_frames,
        policy_frames,
        policy_predictions,
        policy_scored,
        policy_core,
    )
    gc.collect()

    print("asymmetric-value: selection sealed; opening frozen evaluation", flush=True)
    extract_core_source(core_config, "holdout", force=force)
    build_core_features(core_config, "holdout", force=force)
    evaluation = _load_asymmetric_core_grid(
        core_config,
        "holdout",
        config,
    )
    evaluation_core_content_sha256 = _frame_content_digest(evaluation)
    evaluation_price_manifest = extract_asymmetric_price_evidence(
        config,
        scope="evaluation",
        force=force,
    )
    evaluation_prices = load_asymmetric_price_evidence(
        config,
        scope="evaluation",
    )
    evaluation_reference_execution = (
        load_asymmetric_retained_execution_evidence(
            config,
            scope="evaluation",
        )
    )
    evaluation_price_features = attach_asymmetric_value_features(
        evaluation,
        evaluation_prices,
        config,
    )
    champion_executable_predictions = _champion_probability_frame(
        config,
        evaluation_price_features,
        include_execution=True,
    )
    evaluation_executable_core = evaluation_price_features.select(
        *evaluation.columns
    )
    evaluation_coverage = _base_coverage_summary(
        evaluation,
        evaluation_price_features,
        config,
    )
    evaluation_price_features = _project_candidate_source(
        evaluation_price_features,
        PRICE_LOGISTIC,
        CORE_PRICE,
    )
    evaluation_oracle_inventory = _oracle_source_inventory(
        evaluation_executable_core,
        config.oracle_source,
    )
    evaluation_oracle = _load_or_build_oracle_core(
        evaluation_executable_core,
        config,
        destination=config.feature_cache / EVALUATION_ORACLE_CACHE,
        source_inventory=evaluation_oracle_inventory,
        core_content_sha256=evaluation_core_content_sha256,
        force=force,
    )
    evaluation_oracle_price_features = _project_candidate_source(
        attach_asymmetric_value_features(
            evaluation_oracle,
            evaluation_prices,
            config,
        ).filter(pl.col("early_oracle_eligible")),
        ORACLE_MATCHED_CORE_PRICE_CONTROL,
        CORE_ORACLE_PRICE,
    )
    del evaluation_oracle
    gc.collect()

    evaluation_l2 = _load_or_build_source_features(
        evaluation_executable_core,
        config,
        source_family="l2",
        destination=config.feature_cache / "evaluation-l2.parquet",
        core_content_sha256=evaluation_core_content_sha256,
        force=force,
    )
    _add_joint_source_coverage(
        evaluation_coverage,
        evaluation_l2,
        config,
        source_family="l2",
    )
    evaluation_l2_price_features = _project_candidate_source(
        attach_asymmetric_value_features(
            evaluation_l2,
            evaluation_prices,
            config,
        ),
        L2_MATCHED_CORE_PRICE_CONTROL,
        CORE_L2_PRICE,
    )
    del evaluation_l2
    gc.collect()
    evaluation_three_source_price_features = _join_oracle_l2_candidate_features(
        evaluation_oracle_price_features,
        evaluation_l2_price_features,
    )
    _add_joint_source_coverage(
        evaluation_coverage,
        evaluation_three_source_price_features,
        config,
        source_family="oracle_l2",
    )
    evaluation_three_source_price_features = _project_candidate_source(
        evaluation_three_source_price_features,
        THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL,
        CORE_ORACLE_L2_PRICE,
    )

    evaluation_candles = _load_or_build_source_features(
        evaluation_executable_core,
        config,
        source_family="candles",
        destination=config.feature_cache / "evaluation-candles.parquet",
        core_content_sha256=evaluation_core_content_sha256,
        force=force,
    )
    _add_joint_source_coverage(
        evaluation_coverage,
        evaluation_candles,
        config,
        source_family="candle",
    )
    evaluation_candle_price_features = _project_candidate_source(
        attach_asymmetric_value_features(
            evaluation_candles,
            evaluation_prices,
            config,
        ),
        CANDLE_MATCHED_CORE_PRICE_CONTROL,
        CORE_CANDLES_PRICE,
    )
    del evaluation_candles
    evaluation_model_frames = _candidate_frames(
        price=evaluation_price_features,
        oracle_price=evaluation_oracle_price_features,
        l2_price=evaluation_l2_price_features,
        candle_price=evaluation_candle_price_features,
        three_source_price=evaluation_three_source_price_features,
    )
    selected_evaluation_frame = evaluation_model_frames[selection["selected_model"]]
    coverage = {
        "fit": development_coverage["fit"],
        "calibration": development_coverage["calibration"],
        "policy": development_coverage["policy"],
        "evaluation": evaluation_coverage["evaluation"],
    }
    del evaluation_executable_core
    gc.collect()
    coverage["policy"]["source_grid"] = policy_grid_coverage
    coverage["policy"]["candidate_strict_markets"] = {
        **policy_candidate_strict_markets
    }
    coverage["policy"]["candidate_prediction_grid"] = policy_candidate_coverage
    evaluation_grid_coverage = execution_grid_coverage(
        config,
        scope="evaluation",
        core=evaluation,
    )
    coverage["evaluation"]["source_grid"] = evaluation_grid_coverage
    evaluation_candidate_strict_markets = {
        name: frame["market_id"].n_unique()
        for name, frame in evaluation_model_frames.items()
    }
    coverage["evaluation"]["candidate_strict_markets"] = {
        **evaluation_candidate_strict_markets
    }
    evaluation_candidate_coverage = {
        name: _candidate_grid_summary(frame, evaluation, config)
        for name, frame in evaluation_model_frames.items()
    }
    coverage["evaluation"]["candidate_prediction_grid"] = (
        evaluation_candidate_coverage
    )

    evaluation_predictions = _prediction_surface(
        evaluation_model_frames,
        models,
    )
    evaluation_scored = score_two_sided_value(evaluation_predictions)
    selected_evaluation_scored = evaluation_scored.filter(
        pl.col("model") == selection["selected_model"]
    )
    selected_key = selection["selected_key"]
    selected_policy = next(
        policy
        for policy in config.policies
        if policy.name == selection["selected_policy"]
    )
    evaluation_ledgers: dict[str, pl.DataFrame] = {}
    evaluation_model_metrics: dict[str, dict[str, Any]] = {}
    for offset, name in enumerate(ASYMMETRIC_VALUE_CANDIDATES):
        scored = evaluation_scored.filter(pl.col("model") == name)
        ledger = policy_ledger(
            scored,
            selected_policy,
            quantity=config.quantity,
            maximum_depth_participation=config.maximum_depth_participation,
        )
        values = ledger_metrics(ledger)
        values["utc_day_block_bootstrap"] = bootstrap_ledger_metrics(
            ledger,
            resamples=config.bootstrap_resamples,
            seed=config.random_seed + 10_000 + offset,
        )
        evaluation_ledgers[name] = ledger
        evaluation_model_metrics[name] = values
    _add_opportunity_denominators(
        {
            f"{name}::{selected_policy.name}": values
            for name, values in evaluation_model_metrics.items()
        },
        resolved_markets=evaluation["market_id"].n_unique(),
        strict_markets_by_model=evaluation_candidate_strict_markets,
    )
    evaluation_feature_attribution = _matched_feature_attribution(
        {
            candidate_policy_key(name, selected_policy.name): values
            for name, values in evaluation_model_metrics.items()
        },
        {
            candidate_policy_key(name, selected_policy.name): ledger
            for name, ledger in evaluation_ledgers.items()
        },
        evaluation_model_frames,
        policy_name=selected_policy.name,
        config=config,
        window=config.evaluation,
        seed_offset=36_000,
    )
    evaluation_confidence_controls = _confidence_threshold_window(
        evaluation_predictions,
        config,
        window="evaluation",
        seed_offset=40_000,
        resolved_markets=evaluation["market_id"].n_unique(),
        strict_markets=evaluation_candidate_strict_markets[CORE_PRICE],
    )
    confidence_controls = {
        "model": CORE_PRICE,
        "description": evaluation_confidence_controls["description"],
        "accuracy_thresholds_are_diagnostics_not_selection_gates": True,
        "policy_window": policy_confidence_controls["metrics"],
        "evaluation_window": evaluation_confidence_controls["metrics"],
        "table": [
            *policy_confidence_controls["table"],
            *evaluation_confidence_controls["table"],
        ],
    }
    selected_ledger = evaluation_ledgers[selection["selected_model"]]
    selected_metrics = evaluation_model_metrics[selection["selected_model"]]
    matched_control_ledger = (
        evaluation_ledgers[selected_matched_control]
        if selected_matched_control is not None
        else None
    )
    matched_control_metrics = (
        evaluation_model_metrics[selected_matched_control]
        if selected_matched_control is not None
        else None
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
    paired_execution = evaluation_reference_execution.join(
        selected_market_ids,
        on="market_id",
        how="inner",
        validate="m:1",
    )
    full_current_policy_reference = current_policy_reference_ledger(
        full_champion_predictions,
        execution=evaluation_reference_execution,
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
    frozen_value_policy = replace(
        selected_policy,
        name="frozen_champion_low_price_value_60s_240s",
        maximum_entry_second=240,
    )
    frozen_champion_value_reference = policy_ledger(
        score_two_sided_value(champion_executable_predictions),
        frozen_value_policy,
        quantity=config.quantity,
        maximum_depth_participation=config.maximum_depth_participation,
    )
    frozen_champion_value_metrics = ledger_metrics(
        frozen_champion_value_reference
    )
    frozen_champion_value_metrics["utc_day_block_bootstrap"] = (
        _bootstrap_reference(frozen_champion_value_reference, config)
    )
    frozen_champion_value_metrics["supported_interval"] = [60, 240]
    frozen_champion_value_metrics["pre_60_supported"] = False
    frozen_champion_value_metrics["policy_contract"] = {
        "quantity": config.quantity,
        "maximum_depth_participation": (
            config.maximum_depth_participation
        ),
        "minimum_share_price": frozen_value_policy.minimum_share_price,
        "maximum_share_price": frozen_value_policy.maximum_share_price,
        "maximum_cost_per_share": frozen_value_policy.maximum_cost_per_share,
        "minimum_edge_per_share": frozen_value_policy.minimum_edge_per_share,
    }
    comparison = _risk_shape_comparison(
        selected_metrics,
        paired_current_policy_metrics,
    )
    accuracy_prices = accuracy_price_by_second(evaluation_scored)
    evaluation_joint_surface = joint_accuracy_value_surface(
        evaluation_scored
    )
    evaluation_both_side_surface = side_accuracy_value_surface(
        evaluation_predictions
    )
    opportunity_calibration = opportunity_calibration_by_price_band(
        evaluation_scored
    )
    exact_prices = exact_price_by_second(evaluation_prices)
    evaluation_economics_leaderboard = _evaluation_economics_table(
        evaluation_model_metrics,
        selected_model=selection["selected_model"],
        policy=selected_policy.name,
    )

    evaluation_predictions.write_parquet(
        run_dir / "evaluation-predictions.parquet",
        compression="zstd",
    )
    selected_evaluation_scored.write_parquet(
        run_dir / "selected-evaluation-scored-opportunities.parquet",
        compression="zstd",
    )
    evaluation_scored.write_parquet(
        run_dir / "evaluation-scored-opportunities.parquet",
        compression="zstd",
    )
    selected_ledger.write_parquet(run_dir / "selected-policy-ledger.parquet", compression="zstd")
    pl.concat(list(evaluation_ledgers.values()), how="vertical_relaxed").write_parquet(
        run_dir / "evaluation-model-policy-ledger.parquet",
        compression="zstd",
    )
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
    frozen_champion_value_reference.write_parquet(
        run_dir / "frozen-champion-low-price-value-reference-ledger.parquet",
        compression="zstd",
    )
    price_bands = price_band_metrics(selected_ledger)
    pl.DataFrame(price_bands).write_csv(run_dir / "selected-price-band-economics.csv")
    pl.DataFrame(_policy_table(policy_metrics, selection["frontier"])).write_csv(
        run_dir / "candidate-policy-economics.csv"
    )
    pl.DataFrame(accuracy_prices).write_csv(
        run_dir / "accuracy-price-by-observation-second.csv"
    )
    pl.DataFrame(exact_prices).write_csv(
        run_dir / "exact-price-by-observation-second.csv"
    )
    pl.DataFrame(opportunity_calibration).write_csv(
        run_dir / "opportunity-calibration-by-price-band.csv"
    )
    pl.DataFrame(evaluation_economics_leaderboard).write_csv(
        run_dir / "evaluation-model-economics.csv"
    )
    write_json_atomic(
        run_dir / "evaluation-feature-attribution.json",
        evaluation_feature_attribution,
    )
    pl.DataFrame(confidence_controls["table"]).write_csv(
        run_dir / "core-price-confidence-threshold-economics.csv"
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
    evaluation_artifact_names = [
        "evaluation-predictions.parquet",
        "selected-evaluation-scored-opportunities.parquet",
        "evaluation-scored-opportunities.parquet",
        "selected-policy-ledger.parquet",
        "evaluation-model-policy-ledger.parquet",
        "frozen-current-policy-full-reference-ledger.parquet",
        "frozen-current-policy-paired-reference-ledger.parquet",
        "frozen-champion-low-price-value-reference-ledger.parquet",
        "selected-price-band-economics.csv",
        "candidate-policy-economics.csv",
        "accuracy-price-by-observation-second.csv",
        "exact-price-by-observation-second.csv",
        "opportunity-calibration-by-price-band.csv",
        "evaluation-model-economics.csv",
        "evaluation-feature-attribution.json",
        "core-price-confidence-threshold-economics.csv",
        "model-second-side-price-band-surface.csv",
        "model-second-yes-no-price-band-surface.csv",
    ]
    if matched_control_ledger is not None:
        evaluation_artifact_names.append("selected-matched-control-ledger.parquet")

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
            "selected_estimator_family": training["profiles"][
                selection["selected_model"]
            ]["family"],
            "candidate_estimator_families": {
                name: values["family"]
                for name, values in training["profiles"].items()
            },
            "price_aware_candidates": True,
            "two_sided_value_selection": True,
            "underdog_selection_allowed": True,
            "accuracy_gate_used": False,
            "market_equal_row_weights": True,
            "candidate_specific_training_cohorts": True,
            "opening_boundary_features_used_by_candidates": False,
            "prediction_grid": "seconds 1-59 every second; seconds 60-240 every five seconds",
            "calibration": (
                "coherent joint probability by time band, raw 10c price band, "
                "and YES/NO side with identity parent fallback"
            ),
            "causal_oracle_features": list(EARLY_CAUSAL_ORACLE_FEATURES),
            "feature_and_source_ablation_candidates": [
                *ASYMMETRIC_VALUE_MODEL_MATRIX,
            ],
            "model_matrix_feature_counts": EXPECTED_MODEL_FEATURE_COUNTS,
            "selection_eligible_models": sorted(MODEL_SELECTION_ELIGIBLE),
            "offline_only_candidates": sorted(OFFLINE_ONLY_CANDIDATES),
            "combined_oracle_l2_candidate": {
                "model": CORE_ORACLE_L2_PRICE,
                "feature_count": EXPECTED_MODEL_FEATURE_COUNTS[
                    CORE_ORACLE_L2_PRICE
                ],
                "runtime_exportable": False,
                "selection_eligible": False,
                "deployment_blocker": (
                    "the current runtime has no combined Oracle plus spot-L2 "
                    "feature contract"
                ),
            },
            "matched_feature_controls": [
                ORACLE_MATCHED_CORE_PRICE_CONTROL,
                L2_MATCHED_CORE_PRICE_CONTROL,
                CANDLE_MATCHED_CORE_PRICE_CONTROL,
                THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL,
            ],
            "kitchen_sink_candidates_used": False,
            "matched_attribution_uses_identical_market_second_keys": True,
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
            "quantity": config.quantity,
            "maximum_depth_participation": (
                config.maximum_depth_participation
            ),
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
            "policy_feature_attribution": policy_feature_attribution,
            "policy_feature_attribution_artifact": (
                "policy-feature-attribution.json"
            ),
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
            "accuracy_price_by_observation_second": accuracy_prices,
            "accuracy_price_artifact": "accuracy-price-by-observation-second.csv",
            "joint_surface_artifact": "model-second-side-price-band-surface.csv",
            "both_side_surface_artifact": (
                "model-second-yes-no-price-band-surface.csv"
            ),
            "candidate_policy_economics_artifact": (
                "candidate-policy-economics.csv"
            ),
            "evaluation_model_economics_artifact": (
                "evaluation-model-economics.csv"
            ),
            "evaluation_model_economics": evaluation_model_metrics,
            "evaluation_model_economics_leaderboard": (
                evaluation_economics_leaderboard
            ),
            "feature_attribution": evaluation_feature_attribution,
            "feature_attribution_artifact": (
                "evaluation-feature-attribution.json"
            ),
            "core_price_confidence_threshold_controls": confidence_controls,
            "core_price_confidence_threshold_artifact": (
                "core-price-confidence-threshold-economics.csv"
            ),
            "exact_price_by_observation_second": exact_prices,
            "opportunity_calibration_by_price_band": opportunity_calibration,
            "predeclared_evaluation_models": list(ASYMMETRIC_VALUE_CANDIDATES),
            "all_predeclared_models_evaluated": True,
            "qualification_limited_to_selected_model": True,
            "policy_window_candidate_metrics": policy_metrics,
            "selected_feature_attribution_control": {
                "model": selected_matched_control,
                "metrics": matched_control_metrics,
                "same_evaluation_keys_as_selected": bool(
                    selected_matched_control is not None
                    and _frame_key_digest(
                        evaluation_model_frames[selection["selected_model"]]
                    )
                    == _frame_key_digest(
                        evaluation_model_frames[selected_matched_control]
                    )
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
                "five-share size with 25% maximum depth participation, and no added "
                "edge gate; full and "
                "selected-market-paired cohorts are reported separately"
            ),
            "frozen_current_policy_fidelity_limitations": (
                "historical PMXT evidence proves decision-time VWAP5 and depth at the model's "
                "five-second cadence, while the process loop runs every second; the replay "
                "does not reconstruct intermediate-loop quotes, marketable-limit price, "
                "quoted-size, or 150ms arrival fills"
            ),
            "frozen_champion_low_price_value_60s_240s": (
                frozen_champion_value_metrics
            ),
            "frozen_champion_pre60_value_diagnostic_supported": False,
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
            "policy_feature_attribution_sha256": file_sha256(
                run_dir / "policy-feature-attribution.json"
            ),
            "evaluation_feature_attribution_sha256": file_sha256(
                run_dir / "evaluation-feature-attribution.json"
            ),
            "development_price_manifest_sha256": file_sha256(
                config.price_cache / "development" / "manifest.json"
            ),
            "evaluation_price_manifest_sha256": file_sha256(
                config.price_cache / "evaluation" / "manifest.json"
            ),
            "development_l2_features_sha256": file_sha256(
                config.feature_cache / "development-l2.parquet"
            ),
            "evaluation_l2_features_sha256": file_sha256(
                config.feature_cache / "evaluation-l2.parquet"
            ),
            "development_l2_metadata_sha256": file_sha256(
                config.feature_cache / "development-l2.metadata.json"
            ),
            "evaluation_l2_metadata_sha256": file_sha256(
                config.feature_cache / "evaluation-l2.metadata.json"
            ),
            "development_candle_features_sha256": file_sha256(
                config.feature_cache / "development-candles.parquet"
            ),
            "evaluation_candle_features_sha256": file_sha256(
                config.feature_cache / "evaluation-candles.parquet"
            ),
            "development_candle_metadata_sha256": file_sha256(
                config.feature_cache / "development-candles.metadata.json"
            ),
            "evaluation_candle_metadata_sha256": file_sha256(
                config.feature_cache / "evaluation-candles.metadata.json"
            ),
            "development_oracle_features_sha256": file_sha256(
                config.feature_cache / DEVELOPMENT_ORACLE_CACHE
            ),
            "evaluation_oracle_features_sha256": file_sha256(
                config.feature_cache / EVALUATION_ORACLE_CACHE
            ),
            "development_oracle_metadata_sha256": file_sha256(
                (config.feature_cache / DEVELOPMENT_ORACLE_CACHE).with_suffix(
                    ".metadata.json"
                )
            ),
            "evaluation_oracle_metadata_sha256": file_sha256(
                (config.feature_cache / EVALUATION_ORACLE_CACHE).with_suffix(
                    ".metadata.json"
                )
            ),
            "evaluation_artifact_sha256": {
                name: file_sha256(run_dir / name)
                for name in evaluation_artifact_names
            },
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
    l2_price: pl.DataFrame,
    candle_price: pl.DataFrame,
    three_source_price: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    return {
        PRICE_LOGISTIC: price,
        CORE_PRICE: price,
        L2_MATCHED_CORE_PRICE_CONTROL: l2_price,
        CORE_L2_PRICE: l2_price,
        CANDLE_MATCHED_CORE_PRICE_CONTROL: candle_price,
        CORE_CANDLES_PRICE: candle_price,
        ORACLE_MATCHED_CORE_PRICE_CONTROL: oracle_price,
        CORE_ORACLE_PRICE: oracle_price,
        THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL: three_source_price,
        CORE_ORACLE_L2_PRICE: three_source_price,
    }


def _load_asymmetric_core_grid(
    core_config: Any,
    scope: str,
    config: AsymmetricValueConfig,
) -> pl.DataFrame:
    """Read only the frozen 96 decision rows from the larger 1-second cache."""

    validate_core_feature_cache(core_config, scope)
    frame = (
        pl.scan_parquet(feature_destination(core_config, scope))
        .filter(
            pl.col("seconds_elapsed").is_in(
                list(config.prediction_seconds)
            )
        )
        .collect()
    )
    return select_asymmetric_prediction_grid(frame, config)


def _project_candidate_source(
    frame: pl.DataFrame,
    *candidate_names: str,
) -> pl.DataFrame:
    feature_sets = asymmetric_value_feature_sets()
    columns = list(
        dict.fromkeys(
            (
                "market_id",
                "window_start",
                "observed_at",
                "seconds_elapsed",
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
                *(
                    feature
                    for name in candidate_names
                    for feature in feature_sets[name]
                ),
            )
        )
    )
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(
            "candidate source projection is missing columns: "
            + ", ".join(missing)
        )
    return frame.select(*columns)


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


def _join_oracle_l2_candidate_features(
    oracle_price: pl.DataFrame,
    l2_price: pl.DataFrame,
) -> pl.DataFrame:
    """Build the exact three-source cohort without filling either feed."""

    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
    required_oracle = {*keys, *EARLY_CAUSAL_ORACLE_FEATURES}
    required_l2 = {*keys, *L2_FEATURES}
    missing_oracle = sorted(required_oracle - set(oracle_price.columns))
    missing_l2 = sorted(required_l2 - set(l2_price.columns))
    if missing_oracle or missing_l2:
        raise RuntimeError(
            "three-source feature join is missing columns: "
            f"oracle={missing_oracle}; l2={missing_l2}"
        )
    joined = oracle_price.join(
        l2_price.select(*keys, *L2_FEATURES),
        on=keys,
        how="inner",
        validate="1:1",
    )
    if joined.is_empty():
        raise RuntimeError("causal Oracle and spot-L2 sources have no common decision rows")
    duplicate_keys = joined.group_by(*keys).len().filter(pl.col("len") != 1)
    if duplicate_keys.height:
        raise RuntimeError("three-source cohort contains duplicate market/second keys")
    return joined.sort(keys)


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


def _load_or_build_source_features(
    core: pl.DataFrame,
    config: AsymmetricValueConfig,
    *,
    source_family: str,
    destination: Path,
    core_content_sha256: str,
    force: bool,
) -> pl.DataFrame:
    if source_family == "l2":
        source = config.l2_source
        builder = build_partitioned_l2_frame
        require_full_identity = False
    elif source_family == "candles":
        source = config.candle_source
        builder = build_full_closed_candle_frame
        require_full_identity = True
    else:
        raise ValueError(f"unsupported asymmetric-value source family: {source_family}")
    metadata_path = destination.with_suffix(".metadata.json")
    identity = {
        "schema_version": "btc-asymmetric-value-source-features-v1",
        "source_family": source_family,
        "core_key_sha256": _frame_key_digest(core),
        "core_content_sha256": core_content_sha256,
        "core_rows": core.height,
        "core_markets": core["market_id"].n_unique(),
        "minimum_window_start": core["window_start"].min().isoformat(),
        "maximum_window_start": core["window_start"].max().isoformat(),
        "source_metadata_sha256": _source_metadata_digest(source),
        "require_full_core_key_identity": require_full_identity,
    }
    if destination.is_file() and not force:
        if not metadata_path.is_file():
            raise RuntimeError(f"{source_family} feature cache lacks a provenance manifest")
        metadata = json.loads(metadata_path.read_text())
        observed = {name: metadata.get(name) for name in identity}
        observed_sha256 = file_sha256(destination)
        if metadata.get("sha256") != observed_sha256:
            raise RuntimeError(f"{source_family} feature cache content hash changed")
        frame = pl.read_parquet(destination)
        if observed != identity:
            raise RuntimeError(
                f"{source_family} feature cache provenance changed; rebuild intentionally"
            )
        _validate_external_core_keys(frame, core)
        if require_full_identity and _frame_key_digest(frame) != _frame_key_digest(core):
            raise RuntimeError("closed-candle feature cache changed the core decision keys")
        return frame
    frame = builder(core, source)
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
        for package in ("joblib", "numpy", "polars", "scikit-learn", "scipy")
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
    return MATCHED_ATTRIBUTION_CONTROLS.get(model)


def _matched_feature_attribution(
    metrics: dict[str, dict[str, Any]],
    ledgers: dict[str, pl.DataFrame],
    frames: dict[str, pl.DataFrame],
    *,
    policy_name: str,
    config: AsymmetricValueConfig,
    window: EvidenceWindow,
    seed_offset: int,
) -> dict[str, Any]:
    """Attribute each optional source on identical eligible decision keys."""

    feature_sets = asymmetric_value_feature_sets()
    natural_metrics = {
        model: metrics[candidate_policy_key(model, policy_name)]
        for model in ASYMMETRIC_VALUE_MODEL_MATRIX
    }
    comparisons: dict[str, Any] = {}
    for offset, candidate in enumerate(ASYMMETRIC_VALUE_MODEL_MATRIX):
        control = MATCHED_ATTRIBUTION_CONTROLS.get(candidate)
        if control is None:
            continue
        candidate_frame = frames[candidate]
        control_frame = frames[control]
        candidate_digest = _frame_key_digest(candidate_frame)
        control_digest = _frame_key_digest(control_frame)
        if (
            candidate_frame.height != control_frame.height
            or candidate_digest != control_digest
        ):
            raise RuntimeError(
                f"{candidate} attribution control {control} does not share exact "
                "market/second keys"
            )
        candidate_key = candidate_policy_key(candidate, policy_name)
        control_key = candidate_policy_key(control, policy_name)
        candidate_metrics = metrics[candidate_key]
        control_metrics = metrics[control_key]
        differences = {
            name: _finite_difference(
                candidate_metrics.get(name),
                control_metrics.get(name),
            )
            for name in (
                "accuracy",
                "net_expectancy_per_trade",
                "stress_1c_net_expectancy_per_trade",
                "net_profit_per_resolved_market",
                "capital_efficiency",
                "profit_factor",
                "selected_calibration_bias",
                "trades_per_resolved_market",
            )
        }
        comparisons[candidate] = {
            "candidate": candidate,
            "matched_control": control,
            "candidate_feature_count": len(feature_sets[candidate]),
            "control_feature_count": len(feature_sets[control]),
            "added_features": sorted(
                set(feature_sets[candidate]) - set(feature_sets[control])
            ),
            "selection_eligible": candidate in MODEL_SELECTION_ELIGIBLE,
            "runtime_exportable": candidate not in OFFLINE_ONLY_CANDIDATES,
            "identical_market_second_keys_verified": True,
            "decision_cohort": {
                "rows": candidate_frame.height,
                "markets": candidate_frame["market_id"].n_unique(),
                "utc_days": candidate_frame["window_start"].dt.date().n_unique(),
                "key_sha256": candidate_digest,
            },
            "candidate_natural_cohort_metrics": candidate_metrics,
            "matched_control_metrics": control_metrics,
            "candidate_minus_control": differences,
            "paired_utc_day_net_profit": _paired_day_net_difference_bootstrap(
                ledgers[candidate_key],
                ledgers[control_key],
                config,
                seed=config.random_seed + seed_offset + offset,
                window_start=window.start,
                window_end=window.end,
            ),
        }
    return {
        "schema_version": "btc-asymmetric-value-feature-attribution-v1",
        "policy": policy_name,
        "window": {
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
        },
        "model_matrix": list(ASYMMETRIC_VALUE_MODEL_MATRIX),
        "natural_cohort_model_metrics": natural_metrics,
        "comparisons": comparisons,
        "comparison_rule": (
            "candidate and control are independently fit and scored on identical "
            "market_id/window_start/observed_at/seconds_elapsed keys"
        ),
    }


def _matched_control_noninferiority_checks(
    metrics: dict[str, dict[str, Any]],
    ledgers: dict[str, pl.DataFrame],
    policy_name: str,
    config: AsymmetricValueConfig,
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    checks: dict[str, list[dict[str, Any]]] = {}
    eligible = set(MODEL_SELECTION_ELIGIBLE)
    for offset, model in enumerate(sorted(MODEL_SELECTION_ELIGIBLE)):
        control = _selected_matched_control(model)
        if control is None:
            continue
        candidate_key = candidate_policy_key(model, policy_name)
        control_key = candidate_policy_key(control, policy_name)
        candidate = metrics[candidate_key]
        reference = metrics[control_key]
        expectancy_difference = _finite_difference(
            candidate.get("net_expectancy_per_trade"),
            reference.get("net_expectancy_per_trade"),
        )
        yield_difference = _finite_difference(
            candidate.get("net_profit_per_resolved_market"),
            reference.get("net_profit_per_resolved_market"),
        )
        paired = _paired_day_net_difference_bootstrap(
            ledgers[candidate_key],
            ledgers[control_key],
            config,
            seed=config.random_seed + 25_000 + offset,
        )
        checks[model] = [
            {
                "name": "matched_control_point_expectancy_noninferiority",
                "observed": expectancy_difference,
                "threshold": 0.0,
                "operator": ">=",
                "passed": bool(
                    expectancy_difference is not None
                    and expectancy_difference >= 0.0
                ),
                "matched_control": control,
            },
            {
                "name": "matched_control_opportunity_yield_noninferiority",
                "observed": yield_difference,
                "threshold": 0.0,
                "operator": ">=",
                "passed": bool(
                    yield_difference is not None and yield_difference >= 0.0
                ),
                "matched_control": control,
            },
            {
                "name": "matched_control_paired_day_net_lower_95_noninferiority",
                "observed": paired["lower_95"],
                "threshold": 0.0,
                "operator": ">=",
                "passed": bool(paired["lower_95"] >= 0.0),
                "matched_control": control,
                "paired_day_net_difference": paired,
            },
        ]
        if not all(check["passed"] for check in checks[model]):
            eligible.discard(model)
    return checks, eligible


def _finite_difference(candidate: Any, control: Any) -> float | None:
    if candidate is None or control is None:
        return None
    difference = float(candidate) - float(control)
    return difference if np.isfinite(difference) else None


def _paired_day_net_difference_bootstrap(
    candidate: pl.DataFrame,
    control: pl.DataFrame,
    config: AsymmetricValueConfig,
    *,
    seed: int,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> dict[str, Any]:
    days = []
    start = window_start or config.policy.start
    end = window_end or config.policy.end
    current = start.date()
    while current < end.date():
        days.append(current)
        current += timedelta(days=1)

    def daily_net(ledger: pl.DataFrame) -> dict[Any, float]:
        if ledger.is_empty():
            return {}
        return {
            row["utc_day"]: float(row["net_profit"])
            for row in (
                ledger.with_columns(
                    pl.col("window_start").dt.date().alias("utc_day")
                )
                .group_by("utc_day")
                .agg(pl.col("realized_net").sum().alias("net_profit"))
                .to_dicts()
            )
        }

    candidate_daily = daily_net(candidate)
    control_daily = daily_net(control)
    differences = np.asarray(
        [
            candidate_daily.get(day, 0.0) - control_daily.get(day, 0.0)
            for day in days
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = differences[
        rng.integers(0, len(differences), (config.bootstrap_resamples, len(differences)))
    ].mean(axis=1)
    return {
        "utc_days": len(days),
        "mean_daily_net_difference": float(differences.mean()),
        "lower_95": float(np.quantile(sampled, 0.025)),
        "upper_95": float(np.quantile(sampled, 0.975)),
    }


def _confidence_threshold_window(
    predictions: pl.DataFrame,
    config: AsymmetricValueConfig,
    *,
    window: str,
    seed_offset: int,
    resolved_markets: int,
    strict_markets: int,
) -> dict[str, Any]:
    source = predictions.filter(pl.col("model") == CORE_PRICE)
    metrics: dict[str, dict[str, Any]] = {}
    table: list[dict[str, Any]] = []
    for offset, threshold in enumerate(config.confidence_thresholds):
        ledger = confidence_control_ledger(
            source,
            threshold=threshold,
            maximum_entry_second=240,
            maximum_cost_per_share=config.gates.maximum_mean_cost_per_share,
            minimum_edge_per_share=(
                config.confidence_control_minimum_edge_per_share
            ),
            maximum_depth_participation=config.maximum_depth_participation,
            quantity=config.quantity,
        )
        values = ledger_metrics(ledger)
        values["utc_day_block_bootstrap"] = bootstrap_ledger_metrics(
            ledger,
            resamples=config.bootstrap_resamples,
            seed=config.random_seed + seed_offset + offset,
        )
        _add_opportunity_denominators(
            {CORE_PRICE: values},
            resolved_markets=resolved_markets,
            strict_markets_by_model={CORE_PRICE: strict_markets},
        )
        key = f"{threshold:.2f}"
        metrics[key] = values
        table.append(
            {
                "model": CORE_PRICE,
                "window": window,
                "confidence_threshold": threshold,
                "maximum_entry_second": 240,
                "maximum_cost_per_share": (
                    config.gates.maximum_mean_cost_per_share
                ),
                "minimum_edge_per_share": (
                    config.confidence_control_minimum_edge_per_share
                ),
                "maximum_depth_participation": (
                    config.maximum_depth_participation
                ),
                "trades": values.get("trades"),
                "accuracy": values.get("accuracy"),
                "mean_entry_second": values.get("mean_entry_second"),
                "mean_share_price": values.get("mean_share_price"),
                "mean_cost_per_share": values.get(
                    "mean_admission_cost_per_share"
                ),
                "net_profit": values.get("net_profit"),
                "net_profit_per_resolved_market": values.get(
                    "net_profit_per_resolved_market"
                ),
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
                "pre60_trades": values.get("pre60_trades"),
                "twenty_to_thirty_cent_trades": values.get(
                    "twenty_to_thirty_cent_trades"
                ),
                "strict_market_coverage": values.get(
                    "strict_market_coverage"
                ),
            }
        )
    return {
        "model": CORE_PRICE,
        "window": window,
        "description": (
            "same retrained core+price model with only the conventional confidence "
            "threshold varied; 1-240s timing, 70c all-in cost cap, 1.5c edge guard, "
            "five-share size, and 25% depth participation remain fixed"
        ),
        "accuracy_thresholds_are_diagnostics_not_selection_gates": True,
        "metrics": metrics,
        "table": table,
    }


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


def _base_coverage_summary(
    core: pl.DataFrame,
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
        strict_rows = _window(strict, evidence.start, evidence.end)
        core_markets = core_rows["market_id"].n_unique()
        strict_markets = strict_rows["market_id"].n_unique()
        output[name] = {
            "core_resolved_rows": core_rows.height,
            "core_resolved_markets": core_markets,
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
            "strict_market_coverage": (
                strict_markets / core_markets if core_markets else 0.0
            ),
        }
    return output


def _add_joint_source_coverage(
    coverage: dict[str, Any],
    source: pl.DataFrame,
    config: AsymmetricValueConfig,
    *,
    source_family: str,
) -> None:
    if source_family not in {"l2", "candle", "oracle_l2"}:
        raise ValueError(f"unsupported joint source coverage: {source_family}")
    for name, evidence in (
        ("fit", config.fit),
        ("calibration", config.calibration),
        ("policy", config.policy),
        ("evaluation", config.evaluation),
    ):
        rows = _window(source, evidence.start, evidence.end)
        markets = rows["market_id"].n_unique() if not rows.is_empty() else 0
        core_markets = int(coverage[name]["core_resolved_markets"])
        coverage[name][f"joint_pmxt_{source_family}_feature_rows"] = rows.height
        coverage[name][f"joint_pmxt_{source_family}_feature_markets"] = markets
        coverage[name][f"joint_pmxt_{source_family}_market_coverage"] = (
            markets / core_markets if core_markets else 0.0
        )


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


def _bootstrap_reference(
    ledger: pl.DataFrame,
    config: AsymmetricValueConfig,
) -> dict[str, Any] | None:
    return bootstrap_ledger_metrics(
        ledger,
        resamples=config.bootstrap_resamples,
        seed=config.random_seed + 20_000,
    )


def _evaluation_economics_table(
    metrics: dict[str, dict[str, Any]],
    *,
    selected_model: str,
    policy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ASYMMETRIC_VALUE_CANDIDATES:
        values = metrics[model]
        bootstrap = values.get("utc_day_block_bootstrap") or {}
        expectancy = bootstrap.get("net_expectancy_per_trade") or {}
        rows.append(
            {
                "model": model,
                "policy": policy,
                "selected_on_policy_window": model == selected_model,
                "selection_eligible": model in MODEL_SELECTION_ELIGIBLE,
                "resolved_markets": values.get("resolved_markets"),
                "strict_executable_markets": values.get(
                    "strict_executable_markets"
                ),
                "strict_market_coverage": values.get(
                    "strict_market_coverage"
                ),
                "trades": values.get("trades"),
                "trades_per_resolved_market": values.get(
                    "trades_per_resolved_market"
                ),
                "utc_days": values.get("utc_days"),
                "accuracy": values.get("accuracy"),
                "mean_share_price": values.get("mean_share_price"),
                "mean_admission_cost_per_share": values.get(
                    "mean_admission_cost_per_share"
                ),
                "mean_entry_second": values.get("mean_entry_second"),
                "pre60_trades": values.get("pre60_trades"),
                "twenty_to_thirty_cent_trades": values.get(
                    "twenty_to_thirty_cent_trades"
                ),
                "net_profit": values.get("net_profit"),
                "net_profit_per_resolved_market": values.get(
                    "net_profit_per_resolved_market"
                ),
                "net_expectancy_per_trade": values.get(
                    "net_expectancy_per_trade"
                ),
                "expectancy_lower_95": expectancy.get("lower_95"),
                "expectancy_upper_95": expectancy.get("upper_95"),
                "capital_efficiency": values.get("capital_efficiency"),
                "profit_factor": values.get("profit_factor"),
                "average_win": values.get("average_win"),
                "average_loss": values.get("average_loss"),
                "maximum_loss": values.get("maximum_loss"),
                "loss_recovery_wins": values.get("loss_recovery_wins"),
                "stress_1c_net_expectancy_per_trade": values.get(
                    "stress_1c_net_expectancy_per_trade"
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["net_profit_per_resolved_market"]
            if row["net_profit_per_resolved_market"] is not None
            else float("-inf")
        ),
        reverse=True,
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
    calibration = _calibration_report_summary(result["training"])
    selected_label = (
        "qualified hunter" if evaluation["qualified"] else "selected diagnostic candidate"
    )
    if supported:
        outcome_statement = "**Hypothesis supported by qualification evidence.**"
    elif evaluation["status"] == "insufficient_execution_evidence":
        outcome_statement = (
            "**Hypothesis remains inconclusive because exact execution evidence is "
            "insufficient; directional point estimates are reported below.**"
        )
    else:
        outcome_statement = (
            "**Hypothesis was not economically qualified on sufficient evidence in "
            "this run.**"
        )
    lines = [
        "# BTC asymmetric-value hunter benchmark",
        "",
        "This is consumed offline development evidence. It does not authorize live capital.",
        "",
        outcome_statement,
        "",
        f"{selected_label.capitalize()}: `{evaluation['selected_key']}`.",
        f"Policy-window qualification: `{result['selection']['qualified_on_policy_window']}`.",
        f"Frozen evaluation status: `{evaluation['status']}`.",
        "",
        "The primary search is restricted to raw share prices below 30 cents. Accuracy is a",
        "reported property, not an entry threshold; entry requires calibrated edge over all-in cost.",
        "",
        "## Calibration evidence",
        "",
        (
            f"Parent time-band Platt calibrators converged with positive slopes: "
            f"`{calibration['valid_parent_calibrators']}/{calibration['parent_calibrators']}` "
            f"(`{calibration['minimum_parent_rows']}`–`{calibration['maximum_parent_rows']}` rows "
            f"and `{calibration['minimum_parent_markets']}`–"
            f"`{calibration['maximum_parent_markets']}` markets per band)."
        ),
        (
            f"Specialized YES/NO × 10-cent-price × time cells fitted: "
            f"`{calibration['fitted_cells']}/{calibration['cells']}`; identity parent "
            f"fallbacks: `{calibration['fallback_cells']}`. Fallback cells contain at most "
            f"`{calibration['maximum_fallback_cell_utc_days']}` exact-book UTC days versus the "
            f"required `{calibration['minimum_cell_utc_days']}`."
        ),
        (
            "Fallback reasons (a cell may have more than one): "
            f"`{calibration['fallback_reason_text']}`."
        ),
        *(
            [
                (
                    "Policy-driving YES/NO × 20–30-cent × early-time cells genuinely "
                    f"fitted: `{calibration['target_fitted_cells']}/"
                    f"{calibration['target_required_cells']}` across "
                    f"`{calibration['target_qualified_models']}/"
                    f"{calibration['target_required_models']}` candidate profiles. "
                    f"Target qualification: `{calibration['target_qualified']}`."
                )
            ]
            if calibration["target_required_models"]
            else []
        ),
        (
            "Every prediction is parent-time-calibrated. A specialized side/price correction "
            "is applied only where its frozen evidence gate passes; identity fallback leaves "
            "the parent probability unchanged."
        ),
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
            "## Same low-price policy across every predeclared model",
            "",
            "Rows are ordered by net profit per resolved market so sparse source cohorts do not look artificially superior on EV/trade alone.",
            "",
            "| Model | Selected | Strict coverage | Trades | Accuracy | Entry second | Raw share | Net profit | Net/resolved market | EV/trade | EV lower 95% | Capital efficiency | Wins/loss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in evaluation["evaluation_model_economics_leaderboard"]:
        lines.append(
            f"| {row['model']} | {row['selected_on_policy_window']} | "
            f"{_fmt(row['strict_market_coverage'])} | {row['trades']} | "
            f"{_fmt(row['accuracy'])} | "
            f"{_fmt(row['mean_entry_second'])} | {_fmt(row['mean_share_price'])} | "
            f"{_fmt(row['net_profit'])} | "
            f"{_fmt(row['net_profit_per_resolved_market'])} | "
            f"{_fmt(row['net_expectancy_per_trade'])} | "
            f"{_fmt(row['expectancy_lower_95'])} | "
            f"{_fmt(row['capital_efficiency'])} | "
            f"{_fmt(row['loss_recovery_wins'])} |"
        )

    lines.extend(
        [
            "",
            "## Matched source attribution",
            "",
            (
                "Each candidate and control was independently fit and scored on the "
                "same market/second keys. The combined Oracle+L2 arm is offline-only "
                "and cannot be exported to the current runtime."
            ),
            "",
            "| Candidate | Matched control | Rows | Added features | Accuracy delta | EV/trade delta | Net/resolved delta | Paired-day net lower 95% | Exportable |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for attribution in evaluation["feature_attribution"]["comparisons"].values():
        delta = attribution["candidate_minus_control"]
        paired = attribution["paired_utc_day_net_profit"]
        lines.append(
            f"| {attribution['candidate']} | {attribution['matched_control']} | "
            f"{attribution['decision_cohort']['rows']} | "
            f"{len(attribution['added_features'])} | "
            f"{_fmt(delta['accuracy'])} | "
            f"{_fmt(delta['net_expectancy_per_trade'])} | "
            f"{_fmt(delta['net_profit_per_resolved_market'])} | "
            f"{_fmt(paired['lower_95'])} | "
            f"{attribution['runtime_exportable']} |"
        )

    lines.extend(
        [
            "",
            "## Same core+price model at lower confidence thresholds",
            "",
            "This diagnostic varies only the confidence threshold. The 1–240 second window, 70-cent all-in cap, 1.5-cent positive-edge guard, five-share size, and 25% depth participation remain fixed.",
            "",
            "| Threshold | Trades | Accuracy | Entry second | Raw share | Net/resolved market | EV/trade | +1c EV/trade | Avg loss | Wins/loss | Pre-60 | 20–30c |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in evaluation["core_price_confidence_threshold_controls"]["table"]:
        if row["window"] != "evaluation":
            continue
        lines.append(
            f"| {_fmt(row['confidence_threshold'])} | {row['trades']} | "
            f"{_fmt(row['accuracy'])} | {_fmt(row['mean_entry_second'])} | "
            f"{_fmt(row['mean_share_price'])} | "
            f"{_fmt(row['net_profit_per_resolved_market'])} | "
            f"{_fmt(row['net_expectancy_per_trade'])} | "
            f"{_fmt(row['stress_1c_net_expectancy_per_trade'])} | "
            f"{_fmt(row['average_loss'])} | "
            f"{_fmt(row['loss_recovery_wins'])} | {row['pre60_trades']} | "
            f"{row['twenty_to_thirty_cent_trades']} |"
        )

    frozen_value = evaluation["frozen_champion_low_price_value_60s_240s"]
    lines.extend(
        [
            "",
            "## Frozen champion under the low-price value policy",
            "",
            "The frozen champion is scored only on its supported 60–240 second contract; it cannot test the pre-60 hypothesis.",
            "",
            f"Trades: `{frozen_value.get('trades')}`; accuracy: `{_fmt(frozen_value.get('accuracy'))}`; raw share: `{_fmt(frozen_value.get('mean_share_price'))}`; EV/trade: `{_fmt(frozen_value.get('net_expectancy_per_trade'))}`.",
        ]
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
            f"Exact-price market coverage of all resolved core markets: `{_fmt(coverage['strict_market_coverage'])}`; joint PMXT+L2 candidate coverage: `{_fmt(coverage['joint_pmxt_l2_market_coverage'])}`; joint PMXT+closed-candle candidate coverage: `{_fmt(coverage['joint_pmxt_candle_market_coverage'])}`.",
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
            "| Model | Second | Markets | Argmax accuracy | Value-side accuracy | Mean raw share | Mean modeled edge |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    key_seconds = {1, 5, 15, 30, 55, 59, 60, 120, 180, 240}
    for row in evaluation["accuracy_price_by_observation_second"]:
        if row["seconds_elapsed"] not in key_seconds:
            continue
        lines.append(
            f"| {row['model']} | {row['seconds_elapsed']} | {row['markets']} | "
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
            "See the CSV artifacts for every one-second accuracy and price point before 60",
            "seconds, every five-second point afterward, per-model fixed-policy economics,",
            "the selected-side surface, and the explicit YES/NO model × second × raw-price-band surface.",
            "",
        ]
    )
    return "\n".join(lines)


def _calibration_report_summary(training: dict[str, Any]) -> dict[str, Any]:
    profiles = training["profiles"]
    parent_bands = [
        band
        for profile in profiles.values()
        for band in profile["calibration_bands"]
    ]
    calibrations = [
        profile["side_price_time_calibration"] for profile in profiles.values()
    ]
    cells = [cell for calibration in calibrations for cell in calibration["cells"]]
    fallback_cells = [cell for cell in cells if not cell["fitted"]]
    reason_counts: dict[str, int] = {}
    for cell in cells:
        if cell["fitted"]:
            continue
        for reason in (cell.get("fallback") or "unspecified").split("+"):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    reason_text = ", ".join(
        f"{reason}={count}" for reason, count in sorted(reason_counts.items())
    )
    target_contracts = [
        calibration.get("target_contract", {"required": False})
        for calibration in calibrations
    ]
    required_targets = [target for target in target_contracts if target["required"]]
    return {
        "parent_calibrators": len(parent_bands),
        "valid_parent_calibrators": sum(
            bool(band["converged"]) and float(band["slope"]) > 0.0
            for band in parent_bands
        ),
        "minimum_parent_rows": min(int(band["rows"]) for band in parent_bands),
        "maximum_parent_rows": max(int(band["rows"]) for band in parent_bands),
        "minimum_parent_markets": min(
            int(band["markets"]) for band in parent_bands
        ),
        "maximum_parent_markets": max(
            int(band["markets"]) for band in parent_bands
        ),
        "cells": len(cells),
        "fitted_cells": sum(bool(cell["fitted"]) for cell in cells),
        "fallback_cells": sum(not bool(cell["fitted"]) for cell in cells),
        "maximum_fallback_cell_utc_days": max(
            int(cell["utc_days"]) for cell in fallback_cells
        ) if fallback_cells else 0,
        "minimum_cell_utc_days": max(
            int(calibration["minimum_utc_days_per_cell"])
            for calibration in calibrations
        ),
        "fallback_reason_counts": reason_counts,
        "fallback_reason_text": reason_text or "none",
        "target_required_models": len(required_targets),
        "target_qualified_models": sum(
            bool(target["qualified"]) for target in required_targets
        ),
        "target_required_cells": sum(
            int(target["required_fitted_cells"]) for target in required_targets
        ),
        "target_fitted_cells": sum(
            int(target["fitted_cells"]) for target in required_targets
        ),
        "target_fallback_cells": sum(
            int(target["fallback_cells"]) for target in required_targets
        ),
        "target_qualified": bool(
            required_targets
            and all(bool(target["qualified"]) for target in required_targets)
        ),
    }


def _window(frame: pl.DataFrame, start: Any, end: Any) -> pl.DataFrame:
    return frame.filter(pl.col("window_start").is_between(start, end, closed="left"))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"
