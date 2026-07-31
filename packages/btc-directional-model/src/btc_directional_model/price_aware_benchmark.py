from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .continuous_context_features import (
    build_continuous_context_features,
    join_continuous_context_features,
)
from .core_benchmark import _execution_metrics
from .core_config import CORE_ORACLE_SOURCE_CONTRACT, CoreTrainingConfig, load_core_config
from .core_execution import (
    LEGACY_SNAPSHOT_SCHEMA_VERSION,
    ExecutionEvidenceConfig,
    extract_execution_evidence,
)
from .core_extract import file_sha256, write_json_atomic
from .core_features import build_core_features, feature_destination, validate_core_feature_cache
from .core_training import (
    FrozenTrainingBundle,
    chronological_inner_split,
    fit_probability_calibrator,
    range_frame,
    tune_and_fit_model,
)
from .offline_challengers import derive_strict_book_feature_frame
from .oracle_book_benchmark import _execution_economics_frame, _load_execution_evidence
from .price_aware_config import (
    PRICE_AWARE_CONTEXT_SECONDS,
    PRICE_AWARE_DECISION_SECONDS,
    PRICE_AWARE_PRICE_BANDS,
    PriceAwareBenchmarkConfig,
    PriceAwareGateConfig,
)
from .price_aware_training import (
    ACCURACY_ANCHORED_ADMISSION_CANDIDATE,
    ADMISSION_FEATURES,
    BOUNDARY_CONTROL_CANDIDATE,
    BOUNDARY_CONTROL_FEATURES,
    CONTEXT_OUTCOME_CANDIDATE,
    CONTEXT_OUTCOME_FEATURES,
    CORE_ORACLE_OUTCOME_CANDIDATE,
    CORE_ORACLE_OUTCOME_FEATURES,
    OUTCOME_DERIVED_FEATURES,
    OUTCOME_DIRECTION_CORRECT_COLUMN,
    OUTCOME_FEATURES,
    OUTCOME_SIGNAL_BLOCK_COLUMN,
    PRICE_AWARE_FEATURES,
    attach_five_share_economic_targets,
    attach_outcome_signals,
    fit_admission_bundle,
    mark_first_confident_action,
    mark_first_economic_action,
    price_aware_classifier_spec,
    score_admission_actions,
    score_probability_actions,
)
from .provenance import runtime_provenance
from .runtime_export import score_runtime_model

PRICE_AWARE_BENCHMARK_SCHEMA_VERSION = "btc-price-aware-economic-benchmark-v2"
DEPLOYED_PREDECESSOR_CANDIDATE = "deployed_boundary_alignment_control"
DEPLOYED_PREDECESSOR_MODEL_KEY = (
    "btc-5m-directional-boundary-alignment-20260421-20260720-paper-v1"
)
DEPLOYED_PREDECESSOR_MODEL_SHA256 = (
    "c0778189865ca97a748a9f76cbe72d13268fd6e76db683ea727ad142b0576bc4"
)
PRICE_AWARE_CANDIDATES = (
    BOUNDARY_CONTROL_CANDIDATE,
    ACCURACY_ANCHORED_ADMISSION_CANDIDATE,
)


def run_price_aware_benchmark(
    config: PriceAwareBenchmarkConfig,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    core_config = _validated_core_config(config)
    print("price-aware: building isolated 60-240 core/oracle features", flush=True)
    build_core_features(core_config, "pre_holdout", force=force)
    feature_metadata = validate_core_feature_cache(core_config, "pre_holdout")
    feature_path = feature_destination(core_config, "pre_holdout")

    print("price-aware: building narrow causal Binance 90/120-second context", flush=True)
    prewindow_metadata = build_continuous_context_features(
        core_config,
        config.paths.prewindow_features,
        force=force,
    )
    print("price-aware: extracting strict 55-240 execution evidence", flush=True)
    execution_config = _execution_config(config, core_config)
    execution_manifest = extract_execution_evidence(execution_config, force=force)

    universal, book, availability = _load_training_views(
        core_config,
        config,
        feature_path,
        execution_config,
        execution_manifest,
    )
    universal_blocks, book_blocks = _split_views(universal, book, config)
    availability["blocks"] = {
        name: {
            "all_core_oracle_rows": universal_blocks[name].height,
            "all_core_oracle_markets": universal_blocks[name]["market_id"].n_unique(),
            "book_qualified_rows": book_blocks[name].height,
            "book_qualified_markets": book_blocks[name]["market_id"].n_unique(),
            "book_qualified_market_coverage_of_all_core": (
                book_blocks[name]["market_id"].n_unique()
                / universal_blocks[name]["market_id"].n_unique()
            ),
        }
        for name in universal_blocks
    }

    print("price-aware: generating causal universal outcome OOF signals", flush=True)
    (
        outcome_training,
        outcome_book_blocks,
        outcome_universal_blocks,
    ) = _generate_outcome_oof(
        universal,
        universal_blocks,
        book_blocks,
        config,
        core_config,
    )
    print("price-aware: fitting blocked admission diagnostics at zero expected net", flush=True)
    admission_fold_training, scored_threshold_blocks = _fit_admission_policy_folds(
        outcome_book_blocks,
        config,
        core_config,
    )
    all_core_by_block = {
        name: universal_blocks[name]["market_id"].n_unique()
        for name in config.walk_forward.threshold_block_names
    }
    operating_point = select_walk_forward_operating_point(
        scored_threshold_blocks,
        thresholds=config.model.value_thresholds,
        gates=config.gates,
        all_core_markets_by_block=all_core_by_block,
    )
    selected_threshold = float(operating_point["threshold"])
    marked_threshold_blocks = {
        name: mark_first_economic_action(frame, threshold=selected_threshold)
        for name, frame in scored_threshold_blocks.items()
    }

    print("price-aware: fitting frozen confirmation admission and matched control", flush=True)
    final_training, evaluation_frames = _fit_final_candidates(
        outcome_book_blocks,
        config,
        core_config,
        selected_threshold,
    )
    evaluation_name = config.walk_forward.evaluation_block_name
    evaluation_book = outcome_book_blocks[evaluation_name]
    evaluation_universal = outcome_universal_blocks[evaluation_name]
    book_evaluation_markets = evaluation_book["market_id"].n_unique()
    all_core_evaluation_markets = universal_blocks[evaluation_name]["market_id"].n_unique()
    complete_market_ids = _complete_market_ids(evaluation_book)
    candidate_results = {
        candidate: _candidate_result(
            evaluation_frames[candidate],
            book_qualified_markets=book_evaluation_markets,
            all_core_markets=all_core_evaluation_markets,
            complete_market_ids=complete_market_ids,
        )
        for candidate in PRICE_AWARE_CANDIDATES
    }
    deployed_marked = _score_deployed_predecessor(evaluation_book, config)
    candidate_results[DEPLOYED_PREDECESSOR_CANDIDATE] = _candidate_result(
        deployed_marked,
        book_qualified_markets=book_evaluation_markets,
        all_core_markets=all_core_evaluation_markets,
        complete_market_ids=complete_market_ids,
    )
    evaluation_frames[DEPLOYED_PREDECESSOR_CANDIDATE] = deployed_marked
    selection = _selection_payload(
        candidate_results,
        config.gates,
        operating_point,
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    prediction_artifacts: dict[str, Any] = {}
    outcome_book_oof = pl.concat(
        [outcome_book_blocks[block.name] for block in config.walk_forward.blocks],
        how="vertical",
    )
    outcome_universal_oof = pl.concat(
        [outcome_universal_blocks[block.name] for block in config.walk_forward.blocks],
        how="vertical",
    )
    prediction_artifacts["outcome_universal_oof"] = _write_prediction_artifact(
        _outcome_oof_artifact_frame(outcome_universal_oof),
        run_dir / "accuracy-anchored-outcome-universal-oof.parquet",
    )
    prediction_artifacts["outcome_book_oof"] = _write_prediction_artifact(
        _outcome_oof_artifact_frame(outcome_book_oof),
        run_dir / "accuracy-anchored-outcome-book-oof.parquet",
    )
    policy_frame = pl.concat(
        [marked_threshold_blocks[name] for name in config.walk_forward.threshold_block_names],
        how="vertical",
    )
    prediction_artifacts["admission_threshold_oof"] = _write_prediction_artifact(
        _prediction_artifact_frame(policy_frame),
        run_dir / "accuracy-anchored-admission-threshold-oof.parquet",
    )
    for candidate, frame in evaluation_frames.items():
        prediction_artifacts[f"{candidate}_evaluation"] = _write_prediction_artifact(
            _prediction_artifact_frame(frame),
            run_dir / f"{candidate}-evaluation.parquet",
        )

    operating_points = {
        BOUNDARY_CONTROL_CANDIDATE: {
            "selection_kind": "fixed_confidence_on_retrained_feature_family",
            "threshold": config.model.boundary_confidence_threshold,
            "qualified_on_threshold_blocks": None,
            "frontier": [],
        },
        ACCURACY_ANCHORED_ADMISSION_CANDIDATE: operating_point,
    }
    payload: dict[str, Any] = {
        "schema_version": PRICE_AWARE_BENCHMARK_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": (
            "development_candidate_qualified"
            if selection["selected_candidate"]
            else "development_evaluation_complete"
        ),
        "objective": (
            "lock Up or Down to a causally calibrated outcome model, then learn "
            "Trade or NoTrade from expected net return at executable five-share cost"
        ),
        "evaluation": {
            "kind": "consumed_chronological_development",
            "block": evaluation_name,
            "independent": False,
            "note": config.evaluation_note,
            "july_29_onward_consumed": False,
        },
        "data_contract": {
            "range_start": core_config.data.range_start.isoformat(),
            "range_end": core_config.data.range_end.isoformat(),
            "outcome_history_start": config.walk_forward.history_start.isoformat(),
            "context_seconds": list(PRICE_AWARE_CONTEXT_SECONDS),
            "decision_seconds": list(PRICE_AWARE_DECISION_SECONDS),
            "quantity": config.model.quantity,
            "outcome_view": "universal core plus causal Chainlink oracle; no prewindow or book",
            "admission_view": (
                "strict causal two-sided VWAP5 book plus Binance prewindow and "
                "causal universal outcome signals"
            ),
            "fee_formula": "fee_rate * price * (1 - price)",
            "one_trade_per_market": True,
            "direction_flip_allowed": False,
            "complete_decision_window_filter_used_for_training": False,
            "quality_or_missingness_features": False,
            "book_imputation": False,
            "kraken_used": False,
        },
        "model_contract": {
            "control_kind": "retrained_boundary_feature_family",
            "control_features": list(BOUNDARY_CONTROL_FEATURES),
            "outcome_features": list(OUTCOME_FEATURES),
            "admission_features": list(ADMISSION_FEATURES),
            "candidate_names": list(PRICE_AWARE_CANDIDATES),
            "retired_candidates": ["direct_net_value", "return_on_capital_value"],
            "market_equal_row_weights": True,
            "recency_weighting": False,
            "outcome_calibration_fraction": (config.walk_forward.outcome_calibration_fraction),
        },
        "walk_forward": _walk_forward_payload(config),
        "availability": availability,
        "training": {
            "outcome_oof": outcome_training,
            "admission_threshold_folds": admission_fold_training,
            "final": final_training,
        },
        "operating_points": operating_points,
        "outcome_head_evaluation": _direction_metrics(evaluation_universal),
        "comparison_contract": {
            "book_qualified_evaluation_markets": book_evaluation_markets,
            "all_core_oracle_evaluation_markets": all_core_evaluation_markets,
            "action_accuracy_source": "locked economically selected Up or Down action",
            "admission_calibration_source": "P(locked outcome direction is correct)",
            "outcome_calibration_source": "outcome_probability_up",
            "price_bands_are_diagnostic_only": True,
            "deployed_predecessor_scored_as_control": True,
            "deployed_predecessor_identity_pinned": DEPLOYED_PREDECESSOR_MODEL_KEY,
            "deployed_predecessor_model_sha256": DEPLOYED_PREDECESSOR_MODEL_SHA256,
        },
        "candidates": candidate_results,
        "selection": selection,
        "prediction_artifacts": prediction_artifacts,
        "input_provenance": {
            "benchmark_config": str(config.source_path),
            "benchmark_config_sha256": file_sha256(config.source_path),
            "core_config": str(config.core_config),
            "core_config_sha256": file_sha256(config.core_config),
            "core_features": str(feature_path),
            "core_features_sha256": file_sha256(feature_path),
            "core_feature_metadata": feature_metadata,
            "continuous_context_features": str(config.paths.prewindow_features),
            "continuous_context_features_sha256": file_sha256(config.paths.prewindow_features),
            "continuous_context_metadata": prewindow_metadata,
            "execution_manifest": str(execution_config.output_dir / "manifest.json"),
            "execution_manifest_sha256": file_sha256(execution_config.output_dir / "manifest.json"),
            "execution_manifest_totals": execution_manifest["totals"],
            "outcome_universal_oof_key_sha256": _key_fingerprint(outcome_universal_oof),
            "outcome_book_oof_key_sha256": _key_fingerprint(outcome_book_oof),
        },
        "runtime_provenance": runtime_provenance(config.package_root),
        "deployment": {
            "authorized": False,
            "runtime_exported": False,
            "trading_process_created": False,
            "container_rebuilt": False,
            "reason": "training benchmark only; no deploy or export is authorized",
        },
    }
    write_json_atomic(run_dir / "benchmark.json", payload)
    _write_text_atomic(run_dir / "report.html", _render_report(payload))
    print(
        "price-aware: evaluation complete; selected development candidate="
        f"{selection['selected_candidate'] or 'none'}",
        flush=True,
    )
    return run_dir, payload


def select_walk_forward_operating_point(
    scored_by_block: dict[str, pl.DataFrame],
    *,
    thresholds: Iterable[float],
    gates: PriceAwareGateConfig,
    all_core_markets_by_block: dict[str, int],
) -> dict[str, Any]:
    if set(scored_by_block) != set(all_core_markets_by_block):
        raise ValueError("threshold score blocks and all-core denominators must match")
    frontier: list[dict[str, Any]] = []
    for threshold in thresholds:
        fold_records: dict[str, Any] = {}
        marked_frames: list[pl.DataFrame] = []
        for block_name, scored in scored_by_block.items():
            marked = mark_first_economic_action(scored, threshold=float(threshold))
            marked_frames.append(marked)
            selected = marked.filter(pl.col("policy_selected"))
            summary = _economic_summary(
                selected,
                book_qualified_markets=scored["market_id"].n_unique(),
                all_core_markets=all_core_markets_by_block[block_name],
            )
            checks = _fold_checks(selected, summary, gates)
            fold_records[block_name] = {
                "qualified": all(check["passed"] for check in checks),
                "checks": checks,
                "metrics": summary,
                "per_side": _per_side_metrics(selected),
                "admission_calibration": _admission_calibration(selected),
                "selected_net_calibration": _selected_net_calibration(selected),
            }
        combined = pl.concat(marked_frames, how="vertical")
        selected = combined.filter(pl.col("policy_selected"))
        aggregate = _economic_summary(
            selected,
            book_qualified_markets=sum(
                frame["market_id"].n_unique() for frame in scored_by_block.values()
            ),
            all_core_markets=sum(all_core_markets_by_block.values()),
        )
        aggregate_checks = _standalone_checks(
            aggregate,
            gates,
            minimum_trades=gates.minimum_evaluation_trades,
        )
        aggregate_checks.extend(_calibration_checks(selected, gates))
        qualified = all(check["passed"] for check in aggregate_checks) and all(
            record["qualified"] for record in fold_records.values()
        )
        minimum_fold_pnl = min(
            record["metrics"]["net_pnl_per_all_core_market"] for record in fold_records.values()
        )
        frontier.append(
            {
                "threshold": float(threshold),
                "qualified": bool(qualified),
                "aggregate_checks": aggregate_checks,
                "aggregate_metrics": aggregate,
                "aggregate_per_side": _per_side_metrics(selected),
                "aggregate_admission_calibration": _admission_calibration(selected),
                "aggregate_outcome_calibration": _outcome_calibration(selected),
                "aggregate_selected_net_calibration": _selected_net_calibration(selected),
                "folds": fold_records,
                "_minimum_fold_pnl": minimum_fold_pnl,
            }
        )
    qualified = [row for row in frontier if row["qualified"]]
    pool = qualified or frontier
    selected = max(
        pool,
        key=lambda row: (
            row["_minimum_fold_pnl"],
            row["aggregate_metrics"]["net_pnl_per_all_core_market"],
            row["aggregate_metrics"]["accuracy"],
            -_none_as_infinity(row["aggregate_metrics"]["median_selected_ask_vwap_5"]),
        ),
    )
    for row in frontier:
        row.pop("_minimum_fold_pnl")
    return {
        "selection_kind": "first_positive_locked_direction_expected_net",
        "threshold": selected["threshold"],
        "qualified_on_threshold_blocks": bool(selected["qualified"]),
        "selection_objective": (
            "evaluate the frozen first-positive expected-net rule on every "
            "blocked fold; no threshold search"
        ),
        "threshold_blocks": list(scored_by_block),
        "frontier": frontier,
    }


def select_economic_operating_point(
    scored: pl.DataFrame,
    *,
    thresholds: Iterable[float],
    gates: PriceAwareGateConfig,
    all_core_markets: int | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for a single policy block."""

    result = select_walk_forward_operating_point(
        {"policy": scored},
        thresholds=thresholds,
        gates=gates,
        all_core_markets_by_block={"policy": all_core_markets or scored["market_id"].n_unique()},
    )
    marked = mark_first_economic_action(
        scored,
        threshold=float(result["threshold"]),
    )
    selected = marked.filter(pl.col("policy_selected"))
    economics = _economic_summary(
        selected,
        book_qualified_markets=scored["market_id"].n_unique(),
        all_core_markets=all_core_markets or scored["market_id"].n_unique(),
    )
    legacy_checks = [
        _check("minimum policy trades", economics["trades"], ">=", 100),
        _check(
            "minimum decision accuracy",
            economics["accuracy"],
            ">=",
            gates.minimum_accuracy,
        ),
        _check(
            "minimum book-qualified coverage",
            economics["book_qualified_coverage"],
            ">=",
            gates.minimum_coverage,
        ),
        _check(
            "positive net PnL per all-core market",
            economics["net_pnl_per_all_core_market"],
            ">",
            0.0,
        ),
        _profit_factor_check(economics, gates),
    ]
    result["qualified_on_policy_data"] = all(check["passed"] for check in legacy_checks)
    return result


def _generate_outcome_oof(
    universal: pl.DataFrame,
    universal_blocks: dict[str, pl.DataFrame],
    book_blocks: dict[str, pl.DataFrame],
    config: PriceAwareBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> tuple[
    dict[str, Any],
    dict[str, pl.DataFrame],
    dict[str, pl.DataFrame],
]:
    specs = {
        BOUNDARY_CONTROL_CANDIDATE: tuple(BOUNDARY_CONTROL_FEATURES),
        CORE_ORACLE_OUTCOME_CANDIDATE: tuple(CORE_ORACLE_OUTCOME_FEATURES),
        CONTEXT_OUTCOME_CANDIDATE: tuple(CONTEXT_OUTCOME_FEATURES),
    }
    candidates: dict[str, Any] = {}
    attached_by_candidate: dict[str, dict[str, pl.DataFrame]] = {}
    universal_by_candidate: dict[str, dict[str, pl.DataFrame]] = {}
    for candidate_name, feature_names in specs.items():
        training, attached, scored = _generate_one_outcome_oof(
            universal,
            universal_blocks,
            book_blocks,
            config,
            core_config,
            candidate_name=candidate_name,
            feature_names=feature_names,
        )
        attached_by_candidate[candidate_name] = attached
        universal_by_candidate[candidate_name] = scored
        validation_names = [
            block.name
            for block in config.walk_forward.blocks[1:-1]
        ]
        validation = pl.concat(
            [attached[name] for name in validation_names],
            how="vertical",
        ).with_columns(
            pl.col(OUTCOME_DIRECTION_CORRECT_COLUMN).alias("correct")
        )
        metrics = _direction_metrics(validation)
        candidates[candidate_name] = {
            "feature_count": len(feature_names),
            "features": list(feature_names),
            "validation_blocks": validation_names,
            "common_book_row_metrics": metrics,
            "blocks": training,
        }
    selected = max(
        candidates,
        key=lambda name: (
            candidates[name]["common_book_row_metrics"]["accuracy"],
            -candidates[name]["common_book_row_metrics"]["calibration"]["brier_score"],
            -len(candidates[name]["features"]),
        ),
    )
    return (
        {
            "selection_rule": (
                "highest common-book-row validation accuracy, then lower Brier, "
                "then fewer features; confirmation excluded"
            ),
            "selected_candidate": selected,
            "candidates": candidates,
        },
        attached_by_candidate[selected],
        universal_by_candidate[selected],
    )


def _generate_one_outcome_oof(
    universal: pl.DataFrame,
    universal_blocks: dict[str, pl.DataFrame],
    book_blocks: dict[str, pl.DataFrame],
    config: PriceAwareBenchmarkConfig,
    core_config: CoreTrainingConfig,
    *,
    candidate_name: str,
    feature_names: tuple[str, ...],
) -> tuple[
    dict[str, Any],
    dict[str, pl.DataFrame],
    dict[str, pl.DataFrame],
]:
    spec = price_aware_classifier_spec(
        candidate_name,
        features=feature_names,
        recency_half_life_days=config.model.recency_half_life_days,
    )
    training: dict[str, Any] = {}
    attached_book: dict[str, pl.DataFrame] = {}
    scored_universal: dict[str, pl.DataFrame] = {}
    for block in config.walk_forward.blocks:
        history = range_frame(
            universal,
            config.walk_forward.history_start,
            block.start,
        ).sort(["window_start", "market_id", "seconds_elapsed"])
        fit, calibration = chronological_inner_split(
            history,
            validation_fraction=config.walk_forward.outcome_calibration_fraction,
        )
        _validate_binary_cohort(fit, f"{block.name} outcome fit")
        _validate_binary_cohort(calibration, f"{block.name} outcome calibration")
        model, tuning = tune_and_fit_model(fit, spec, core_config)
        calibrator = fit_probability_calibrator(
            model,
            calibration,
            core_config,
            spec,
        )
        bundle = FrozenTrainingBundle(model, calibrator, 0.5)
        block_frame = universal_blocks[block.name]
        probability = bundle.probability(block_frame)
        scored = (
            block_frame.select(
                "market_id",
                "window_start",
                "observed_at",
                "seconds_elapsed",
                "label_up",
            )
            .with_columns(
                pl.Series("outcome_probability_up", probability),
                (pl.Series("outcome_probability_up", probability) >= 0.5)
                .cast(pl.Int8)
                .alias("outcome_predicted_up"),
                pl.lit(block.name).alias("outcome_signal_block"),
                pl.lit(candidate_name).alias("candidate"),
            )
            .with_columns(
                (pl.col("outcome_predicted_up") == pl.col("label_up")).alias("correct"),
                pl.lit(block.name).alias("walk_forward_block"),
            )
        )
        scored_universal[block.name] = scored
        attached_book[block.name] = _attach_outcome_to_book(
            book_blocks[block.name],
            scored,
            block.name,
        )
        training[block.name] = {
            "fit": _cohort_lineage(fit),
            "calibration": _cohort_lineage(calibration),
            "score": _cohort_lineage(block_frame),
            "features": list(feature_names),
            "feature_count": len(feature_names),
            "tuning": tuning,
            "calibrator": asdict(calibrator),
            "score_metrics": _direction_metrics(scored),
            "causality": (
                f"fit and calibration end before {block.name} begins; "
                "the scored block contributes no outcome labels"
            ),
        }
    return training, attached_book, scored_universal


def _attach_outcome_to_book(
    book: pl.DataFrame,
    universal_signals: pl.DataFrame,
    block_name: str,
) -> pl.DataFrame:
    keys = [
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
    ]
    duplicate = universal_signals.group_by(keys).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise RuntimeError(f"{block_name} universal outcome keys are not unique")
    missing = book.select(keys).join(
        universal_signals.select(keys),
        on=keys,
        how="anti",
    )
    if missing.height:
        raise RuntimeError(
            f"{block_name} has {missing.height} strict-book keys without "
            "an exact universal outcome signal"
        )
    joined = book.join(
        universal_signals.select(*keys, "outcome_probability_up"),
        on=keys,
        how="left",
        validate="1:1",
    )
    if joined.height != book.height or joined["outcome_probability_up"].null_count():
        raise RuntimeError(f"{block_name} outcome-to-book lineage join was incomplete")
    probability = joined["outcome_probability_up"].to_numpy()
    attached = attach_outcome_signals(
        joined.drop("outcome_probability_up"),
        probability,
        block_name=block_name,
    ).with_columns(pl.lit(block_name).alias("walk_forward_block"))
    if set(attached["outcome_signal_block"].unique().to_list()) != {block_name}:
        raise RuntimeError(f"{block_name} outcome lineage was not preserved")
    return attached.sort(["window_start", "market_id", "seconds_elapsed"])


def _fit_admission_policy_folds(
    blocks: dict[str, pl.DataFrame],
    config: PriceAwareBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    ordered_names = [block.name for block in config.walk_forward.blocks]
    training: dict[str, Any] = {}
    scored: dict[str, pl.DataFrame] = {}
    for block_name in config.walk_forward.threshold_block_names:
        index = ordered_names.index(block_name)
        if index < 2:
            raise RuntimeError("admission threshold folds need fit and calibration history")
        calibration_name = ordered_names[index - 1]
        fit_names = ordered_names[: index - 1]
        fit = _concat_blocks(blocks, fit_names)
        calibration = blocks[calibration_name]
        _validate_admission_cohort(fit, f"{block_name} admission fit")
        _validate_admission_cohort(
            calibration,
            f"{block_name} admission calibration",
        )
        bundle, tuning = fit_admission_bundle(
            fit,
            calibration,
            core_config=core_config,
            recency_half_life_days=config.model.recency_half_life_days,
        )
        target = score_admission_actions(blocks[block_name], bundle).with_columns(
            pl.lit(block_name).alias("walk_forward_block")
        )
        _validate_locked_direction(target, block_name)
        scored[block_name] = target
        training[block_name] = {
            "fit_blocks": fit_names,
            "calibration_block": calibration_name,
            "score_block": block_name,
            "fit": _cohort_lineage(fit),
            "calibration": _cohort_lineage(calibration),
            "score": _cohort_lineage(target),
            "features": list(ADMISSION_FEATURES),
            "feature_count": len(ADMISSION_FEATURES),
            "tuning": tuning,
            "calibrator": asdict(bundle.calibrator),
        }
    return training, scored


def _fit_final_candidates(
    blocks: dict[str, pl.DataFrame],
    config: PriceAwareBenchmarkConfig,
    core_config: CoreTrainingConfig,
    admission_threshold: float,
) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    ordered_names = [block.name for block in config.walk_forward.blocks]
    evaluation_name = config.walk_forward.evaluation_block_name
    evaluation_index = ordered_names.index(evaluation_name)
    calibration_name = ordered_names[evaluation_index - 1]
    fit_names = ordered_names[: evaluation_index - 1]
    fit = _concat_blocks(blocks, fit_names)
    calibration = blocks[calibration_name]
    evaluation = blocks[evaluation_name]

    admission_bundle, admission_tuning = fit_admission_bundle(
        fit,
        calibration,
        core_config=core_config,
        recency_half_life_days=config.model.recency_half_life_days,
    )
    admission_scored = score_admission_actions(
        evaluation,
        admission_bundle,
    ).with_columns(pl.lit(evaluation_name).alias("walk_forward_block"))
    _validate_locked_direction(admission_scored, evaluation_name)
    admission_marked = mark_first_economic_action(
        admission_scored,
        threshold=admission_threshold,
    )

    control_spec = price_aware_classifier_spec(
        BOUNDARY_CONTROL_CANDIDATE,
        features=BOUNDARY_CONTROL_FEATURES,
        recency_half_life_days=config.model.recency_half_life_days,
    )
    control_model, control_tuning = tune_and_fit_model(
        fit,
        control_spec,
        core_config,
    )
    control_calibrator = fit_probability_calibrator(
        control_model,
        calibration,
        core_config,
        control_spec,
    )
    control_bundle = FrozenTrainingBundle(control_model, control_calibrator, 0.5)
    control_outcome_columns = tuple(
        column
        for column in (
            *OUTCOME_DERIVED_FEATURES,
            OUTCOME_DIRECTION_CORRECT_COLUMN,
            OUTCOME_SIGNAL_BLOCK_COLUMN,
        )
        if column in evaluation.columns
    )
    control_input = evaluation.drop(*control_outcome_columns)
    control_scored = score_probability_actions(
        control_input,
        control_bundle.probability(control_input),
        candidate=BOUNDARY_CONTROL_CANDIDATE,
        select_by_value=False,
    ).with_columns(pl.lit(evaluation_name).alias("walk_forward_block"))
    control_marked = mark_first_confident_action(
        control_scored,
        confidence_threshold=config.model.boundary_confidence_threshold,
    )
    lineage = {
        "fit_blocks": fit_names,
        "calibration_block": calibration_name,
        "evaluation_block": evaluation_name,
    }
    return (
        {
            ACCURACY_ANCHORED_ADMISSION_CANDIDATE: {
                **lineage,
                "family": "locked_outcome_direction_admission_classifier",
                "features": list(ADMISSION_FEATURES),
                "feature_count": len(ADMISSION_FEATURES),
                "tuning": admission_tuning,
                "calibrator": asdict(admission_bundle.calibrator),
                "frozen_edge_threshold": admission_threshold,
            },
            BOUNDARY_CONTROL_CANDIDATE: {
                **lineage,
                "family": "histogram_outcome_classifier",
                "features": list(BOUNDARY_CONTROL_FEATURES),
                "feature_count": len(BOUNDARY_CONTROL_FEATURES),
                "tuning": control_tuning,
                "calibrator": asdict(control_calibrator),
                "fixed_confidence_threshold": (config.model.boundary_confidence_threshold),
                "control_scope": (
                    "freshly retrained on identical strict-book chronological rows; "
                    "not the deployed predecessor artifact"
                ),
            },
        },
        {
            BOUNDARY_CONTROL_CANDIDATE: control_marked,
            ACCURACY_ANCHORED_ADMISSION_CANDIDATE: admission_marked,
        },
    )


def _concat_blocks(
    blocks: dict[str, pl.DataFrame],
    names: list[str],
) -> pl.DataFrame:
    if not names:
        raise RuntimeError("at least one fit block is required")
    return pl.concat([blocks[name] for name in names], how="vertical").sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )


def _score_deployed_predecessor(
    evaluation: pl.DataFrame,
    config: PriceAwareBenchmarkConfig,
) -> pl.DataFrame:
    model_path = (
        config.package_root
        / "runtime-models"
        / DEPLOYED_PREDECESSOR_MODEL_KEY
        / "model.json"
    )
    if file_sha256(model_path) != DEPLOYED_PREDECESSOR_MODEL_SHA256:
        raise RuntimeError("deployed predecessor model hash mismatch")
    model = json.loads(model_path.read_text())
    feature_names = tuple(model["features"]["names"])
    missing = sorted(set(feature_names) - set(evaluation.columns))
    if missing:
        raise RuntimeError(
            "deployed predecessor features are missing: " + ", ".join(missing)
        )
    rows = evaluation.select(*feature_names, "seconds_elapsed").to_dicts()
    predictions = [
        score_runtime_model(
            model,
            [row[name] for name in feature_names],
            seconds_elapsed=int(row["seconds_elapsed"]),
        )
        for row in rows
    ]
    probability = np.asarray(
        [float(row["probability_up"]) for row in predictions],
        dtype=np.float64,
    )
    scored = score_probability_actions(
        evaluation,
        probability,
        candidate=DEPLOYED_PREDECESSOR_CANDIDATE,
        select_by_value=False,
    ).with_columns(
        pl.Series(
            "runtime_trade_eligible",
            [row["action"] != "no_trade" for row in predictions],
        )
    )
    selected_keys = (
        scored.filter(pl.col("runtime_trade_eligible"))
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .group_by("market_id", maintain_order=True)
        .first()
        .select("market_id", "observed_at", "seconds_elapsed")
        .with_columns(pl.lit(True).alias("policy_selected"))
    )
    return scored.join(
        selected_keys,
        on=["market_id", "observed_at", "seconds_elapsed"],
        how="left",
        validate="1:1",
    ).with_columns(pl.col("policy_selected").fill_null(False))


def _load_training_views(
    core_config: CoreTrainingConfig,
    config: PriceAwareBenchmarkConfig,
    feature_path: Path,
    execution_config: ExecutionEvidenceConfig,
    execution_manifest: dict[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    universal = pl.read_parquet(feature_path).sort(["window_start", "market_id", "seconds_elapsed"])
    _validate_universal_training_frame(universal)
    prewindow = pl.read_parquet(config.paths.prewindow_features)
    with_prewindow = join_continuous_context_features(universal, prewindow).filter(
        pl.col("continuous_context_model_eligible")
    )
    _validate_feature_values(
        with_prewindow,
        OUTCOME_FEATURES,
        "universal outcome view with continuous context",
    )
    execution = _load_execution_evidence(execution_config, execution_manifest)
    strict = derive_strict_book_feature_frame(
        with_prewindow,
        execution,
        require_ten_share=False,
        include_deltas=True,
    ).filter(pl.col("seconds_elapsed").is_in(PRICE_AWARE_DECISION_SECONDS))
    economics = _execution_economics_frame(execution)
    book = attach_five_share_economic_targets(
        strict.join(
            economics,
            on=["market_id", "observed_at"],
            how="inner",
            validate="1:1",
        )
    ).sort(["window_start", "market_id", "seconds_elapsed"])
    _validate_book_training_frame(book)
    key_columns = [
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
    ]
    if (
        book.select(key_columns)
        .join(
            universal.select(key_columns),
            on=key_columns,
            how="anti",
        )
        .height
    ):
        raise RuntimeError("strict-book view is not an exact-key subset of universal view")
    universal = with_prewindow.sort(["window_start", "market_id", "seconds_elapsed"])
    universal_markets = universal["market_id"].n_unique()
    point_markets = book["market_id"].n_unique()
    complete_market_ids = _complete_market_ids(book)
    return (
        universal,
        book,
        {
            "universal_core_oracle_rows": universal.height,
            "universal_core_oracle_markets": universal_markets,
            "markets_with_execution_evidence": execution["market_id"].n_unique(),
            "point_qualified_rows": book.height,
            "point_qualified_markets": point_markets,
            "point_qualified_market_coverage_of_core": (
                point_markets / universal_markets if universal_markets else 0.0
            ),
            "complete_decision_window_markets_diagnostic_only": len(complete_market_ids),
            "point_qualified_rows_by_second": {
                str(second): book.filter(pl.col("seconds_elapsed") == second).height
                for second in PRICE_AWARE_DECISION_SECONDS
            },
            "universal_key_sha256": _key_fingerprint(universal),
            "strict_book_key_sha256": _key_fingerprint(book),
            "daily_coverage": _daily_availability(universal, execution, book),
        },
    )


def _split_views(
    universal: pl.DataFrame,
    book: pl.DataFrame,
    config: PriceAwareBenchmarkConfig,
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    universal_blocks: dict[str, pl.DataFrame] = {}
    book_blocks: dict[str, pl.DataFrame] = {}
    for block in config.walk_forward.blocks:
        universal_block = range_frame(universal, block.start, block.end).sort(
            ["window_start", "market_id", "seconds_elapsed"]
        )
        book_block = range_frame(book, block.start, block.end).sort(
            ["window_start", "market_id", "seconds_elapsed"]
        )
        _validate_binary_cohort(universal_block, f"{block.name} universal")
        _validate_binary_cohort(book_block, f"{block.name} strict book")
        if book_block["market_id"].n_unique() < 200:
            raise RuntimeError(f"{block.name} strict-book cohort has fewer than 200 markets")
        universal_blocks[block.name] = universal_block
        book_blocks[block.name] = book_block
    return universal_blocks, book_blocks


def _complete_market_ids(frame: pl.DataFrame) -> list[str]:
    expected = set(PRICE_AWARE_DECISION_SECONDS)
    return (
        frame.group_by("market_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("seconds_elapsed").n_unique().alias("seconds"),
            pl.col("seconds_elapsed").unique().alias("values"),
        )
        .filter(
            (pl.col("rows") == len(expected))
            & (pl.col("seconds") == len(expected))
            & pl.col("values").list.set_difference(list(expected)).list.len().eq(0)
        )["market_id"]
        .to_list()
    )


def _candidate_result(
    marked: pl.DataFrame,
    *,
    book_qualified_markets: int,
    all_core_markets: int,
    complete_market_ids: list[str],
) -> dict[str, Any]:
    selected = marked.filter(pl.col("policy_selected"))
    complete_selected = selected.filter(pl.col("market_id").is_in(complete_market_ids))
    return {
        "economics": _economic_summary(
            selected,
            book_qualified_markets=book_qualified_markets,
            all_core_markets=all_core_markets,
        ),
        "ten_share_vwap10_diagnostic": _execution_metrics(
            selected,
            quantity=10.0,
            vwap_depth=10,
        ),
        "per_side": _per_side_metrics(selected),
        "price_bands_diagnostic_only": _price_band_metrics(selected),
        "selected_net_calibration": _selected_net_calibration(selected),
        "admission_probability_calibration": _admission_calibration(selected),
        "outcome_probability_calibration": _outcome_calibration(selected),
        "complete_decision_window_sensitivity": {
            "complete_book_qualified_markets": len(complete_market_ids),
            "economics": _economic_summary(
                complete_selected,
                book_qualified_markets=len(complete_market_ids),
                all_core_markets=all_core_markets,
            ),
        },
    }


def _economic_summary(
    selected: pl.DataFrame,
    *,
    book_qualified_markets: int,
    all_core_markets: int,
) -> dict[str, Any]:
    execution = _execution_metrics(selected, quantity=5.0, vwap_depth=5)
    prices = selected["selected_ask_vwap_5"].to_numpy()
    fees = selected["selected_fee_per_share"].to_numpy()
    pnl_total = execution["realized_net_pnl_total"] or 0.0
    entry_notional = float(np.sum(prices) * 5.0) if len(prices) else 0.0
    entry_fees = float(np.sum(fees) * 5.0) if len(fees) else 0.0
    entry_debit = entry_notional + entry_fees
    correct = int(selected["correct"].sum()) if selected.height else 0
    accuracy = correct / selected.height if selected.height else 0.0
    wilson_lower, wilson_upper = _wilson_interval(correct, selected.height)
    true_up = selected.filter(pl.col("label_up") == 1)
    true_down = selected.filter(pl.col("label_up") == 0)
    up_recall = float(true_up["correct"].mean()) if true_up.height else None
    down_recall = float(true_down["correct"].mean()) if true_down.height else None
    balanced_accuracy = (
        (up_recall + down_recall) / 2.0
        if up_recall is not None and down_recall is not None
        else None
    )
    per_trade_pnl = (
        selected["realized_selected_net_per_share"].to_numpy() * 5.0
        if selected.height
        else np.asarray([], dtype=np.float64)
    )
    wins = per_trade_pnl[per_trade_pnl > 0.0]
    losses = per_trade_pnl[per_trade_pnl < 0.0]
    average_win = float(np.mean(wins)) if len(wins) else None
    average_loss = float(np.mean(losses)) if len(losses) else None
    return {
        "book_qualified_markets": book_qualified_markets,
        "all_core_oracle_markets": all_core_markets,
        "trades": selected.height,
        "book_qualified_coverage": (
            selected.height / book_qualified_markets if book_qualified_markets else 0.0
        ),
        "all_core_market_coverage": (
            selected.height / all_core_markets if all_core_markets else 0.0
        ),
        "book_qualified_no_trade_rate": (
            (book_qualified_markets - selected.height) / book_qualified_markets
            if book_qualified_markets
            else 0.0
        ),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "up_recall": up_recall,
        "down_recall": down_recall,
        "accuracy_wilson_lower_95": wilson_lower,
        "accuracy_wilson_upper_95": wilson_upper,
        "mean_selected_ask_vwap_5": float(np.mean(prices)) if len(prices) else None,
        "median_selected_ask_vwap_5": (float(np.median(prices)) if len(prices) else None),
        "p75_selected_ask_vwap_5": (float(np.quantile(prices, 0.75)) if len(prices) else None),
        "p90_selected_ask_vwap_5": (float(np.quantile(prices, 0.90)) if len(prices) else None),
        "median_seconds_elapsed": (
            float(selected["seconds_elapsed"].median()) if selected.height else None
        ),
        "p90_seconds_elapsed": (
            float(selected["seconds_elapsed"].quantile(0.90)) if selected.height else None
        ),
        "total_entry_notional": entry_notional,
        "total_entry_fees": entry_fees,
        "total_entry_debit": entry_debit,
        "capital_efficiency": pnl_total / entry_debit if entry_debit else None,
        "average_win": average_win,
        "average_loss": average_loss,
        "wins_to_recover_average_loss": (
            abs(average_loss) / average_win
            if average_loss is not None and average_win is not None and average_win > 0
            else None
        ),
        "direction_override_rate_vs_binance": (
            float((selected["predicted_up"] != selected["binance_sign_up"]).mean())
            if selected.height and "binance_sign_up" in selected.columns
            else None
        ),
        "net_pnl_per_book_qualified_market": (
            pnl_total / book_qualified_markets if book_qualified_markets else 0.0
        ),
        "net_pnl_per_all_core_market": (pnl_total / all_core_markets if all_core_markets else 0.0),
        "execution": execution,
    }


def _price_band_metrics(selected: pl.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, lower, upper in PRICE_AWARE_PRICE_BANDS:
        rows = selected.filter(
            pl.col("selected_ask_vwap_5").is_between(lower, upper, closed="left")
        )
        execution = _execution_metrics(rows, quantity=5.0, vwap_depth=5)
        output[name] = {
            "minimum_price_inclusive": lower,
            "maximum_price_exclusive": upper,
            "trades": rows.height,
            "accuracy": float(rows["correct"].mean()) if rows.height else None,
            "fee_inclusive_break_even_accuracy": (
                float(
                    (
                        rows["selected_ask_vwap_5"]
                        + rows["selected_fee_per_share"]
                    ).mean()
                )
                if rows.height
                else None
            ),
            "accuracy_minus_fee_inclusive_break_even": (
                float(rows["correct"].mean())
                - float(
                    (
                        rows["selected_ask_vwap_5"]
                        + rows["selected_fee_per_share"]
                    ).mean()
                )
                if rows.height
                else None
            ),
            "realized_net_pnl_total": execution["realized_net_pnl_total"],
            "realized_net_expectancy_per_trade": execution["realized_net_expectancy_per_trade"],
            "profit_factor": execution["profit_factor"],
        }
    return output


def _selected_net_calibration(selected: pl.DataFrame) -> dict[str, Any]:
    if selected.is_empty() or "predicted_net_per_share" not in selected.columns:
        return {
            "mean_predicted_net_per_share": None,
            "mean_realized_net_per_share": None,
            "signed_bias": None,
            "absolute_bias": None,
            "bins": [],
        }
    predicted = selected["predicted_net_per_share"].to_numpy()
    realized = selected["realized_selected_net_per_share"].to_numpy()
    signed_bias = float(np.mean(predicted) - np.mean(realized))
    ordered = selected.sort("predicted_net_per_share")
    bins: list[dict[str, Any]] = []
    for index, positions in enumerate(
        np.array_split(np.arange(ordered.height), min(10, ordered.height)),
        start=1,
    ):
        cohort = ordered[positions.tolist()]
        bins.append(
            {
                "bin": index,
                "rows": cohort.height,
                "mean_predicted_net_per_share": float(cohort["predicted_net_per_share"].mean()),
                "mean_realized_net_per_share": float(
                    cohort["realized_selected_net_per_share"].mean()
                ),
            }
        )
    return {
        "mean_predicted_net_per_share": float(np.mean(predicted)),
        "mean_realized_net_per_share": float(np.mean(realized)),
        "signed_bias": signed_bias,
        "absolute_bias": abs(signed_bias),
        "bins": bins,
    }


def _probability_calibration(
    frame: pl.DataFrame,
    *,
    probability_column: str,
    target_column: str,
) -> dict[str, Any]:
    if frame.is_empty() or probability_column not in frame.columns:
        return {"brier_score": None, "expected_calibration_error": None}
    probability = frame[probability_column].to_numpy()
    labels = frame[target_column].to_numpy()
    return {
        "brier_score": float(np.mean((probability - labels) ** 2)),
        "expected_calibration_error": _expected_calibration_error(
            probability,
            labels,
        ),
    }


def _admission_calibration(selected: pl.DataFrame) -> dict[str, Any]:
    return _probability_calibration(
        selected,
        probability_column="admission_probability_correct",
        target_column="correct",
    )


def _outcome_calibration(selected: pl.DataFrame) -> dict[str, Any]:
    return _probability_calibration(
        selected,
        probability_column="outcome_probability_up",
        target_column="label_up",
    )


def _per_side_metrics(selected: pl.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for side, predicted_up in (("up", 1), ("down", 0)):
        rows = selected.filter(pl.col("predicted_up") == predicted_up)
        execution = _execution_metrics(rows, quantity=5.0, vwap_depth=5)
        output[side] = {
            "trades": rows.height,
            "accuracy": float(rows["correct"].mean()) if rows.height else None,
            "median_selected_ask_vwap_5": (
                float(rows["selected_ask_vwap_5"].median()) if rows.height else None
            ),
            "realized_net_pnl_total": execution["realized_net_pnl_total"],
            "realized_net_expectancy_per_trade": execution["realized_net_expectancy_per_trade"],
            "profit_factor": execution["profit_factor"],
        }
    return output


def _direction_metrics(scored: pl.DataFrame) -> dict[str, Any]:
    if scored.is_empty():
        return {
            "unit": "decision_point",
            "rows": 0,
            "markets": 0,
            "accuracy": None,
            "accuracy_wilson_lower_95": None,
            "accuracy_wilson_upper_95": None,
            "calibration": {
                "brier_score": None,
                "expected_calibration_error": None,
            },
            "per_true_side": {},
        }
    correct = int(scored["correct"].sum())
    lower, upper = _wilson_interval(correct, scored.height)
    per_true_side: dict[str, Any] = {}
    recalls: list[float] = []
    for name, label in (("up", 1), ("down", 0)):
        rows = scored.filter(pl.col("label_up") == label)
        recall = float(rows["correct"].mean()) if rows.height else None
        if recall is not None:
            recalls.append(recall)
        per_true_side[name] = {
            "rows": rows.height,
            "markets": rows["market_id"].n_unique(),
            "accuracy": recall,
        }
    probability = np.clip(scored["outcome_probability_up"].to_numpy(), 1e-9, 1 - 1e-9)
    labels = scored["label_up"].to_numpy()
    log_loss_value = float(
        -np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability))
    )
    return {
        "unit": "decision_point",
        "rows": scored.height,
        "markets": scored["market_id"].n_unique(),
        "accuracy": correct / scored.height,
        "balanced_accuracy": float(np.mean(recalls)) if len(recalls) == 2 else None,
        "accuracy_wilson_lower_95": lower,
        "accuracy_wilson_upper_95": upper,
        "predicted_up_rate": float((scored["outcome_predicted_up"] == 1).mean()),
        "log_loss": log_loss_value,
        "calibration": _outcome_calibration(scored),
        "per_true_side": per_true_side,
    }


def _selection_payload(
    candidates: dict[str, Any],
    gates: PriceAwareGateConfig,
    operating_point: dict[str, Any],
) -> dict[str, Any]:
    control = candidates[BOUNDARY_CONTROL_CANDIDATE]["economics"]
    deployed = candidates[DEPLOYED_PREDECESSOR_CANDIDATE]["economics"]
    result = candidates[ACCURACY_ANCHORED_ADMISSION_CANDIDATE]
    economics = result["economics"]
    per_side = result["per_side"]
    checks = _standalone_checks(
        economics,
        gates,
        minimum_trades=gates.minimum_evaluation_trades,
    )
    checks.extend(
        [
            _check(
                "frozen zero-edge policy qualified before confirmation",
                int(operating_point["qualified_on_threshold_blocks"]),
                ">=",
                1,
            ),
            _check(
                "minimum confirmation Up trades",
                per_side["up"]["trades"],
                ">=",
                gates.minimum_fold_direction_trades,
            ),
            _check(
                "minimum confirmation Down trades",
                per_side["down"]["trades"],
                ">=",
                gates.minimum_fold_direction_trades,
            ),
            *_calibration_checks_from_result(result, gates),
            _check(
                "net PnL per all-core market improves matched control",
                economics["net_pnl_per_all_core_market"],
                ">",
                control["net_pnl_per_all_core_market"],
            ),
            _check(
                "net PnL per all-core market improves deployed predecessor",
                economics["net_pnl_per_all_core_market"],
                ">",
                deployed["net_pnl_per_all_core_market"],
            ),
            _check(
                "worst trade does not worsen matched control",
                economics["execution"]["worst_realized_net_pnl"],
                ">=",
                control["execution"]["worst_realized_net_pnl"],
            ),
            _check(
                "worst-one-percent mean does not worsen matched control",
                economics["execution"]["mean_worst_one_percent_realized_net_pnl"],
                ">=",
                control["execution"]["mean_worst_one_percent_realized_net_pnl"],
            ),
            _check(
                "worst trade does not worsen deployed predecessor",
                economics["execution"]["worst_realized_net_pnl"],
                ">=",
                deployed["execution"]["worst_realized_net_pnl"],
            ),
            _check(
                "worst-one-percent mean does not worsen deployed predecessor",
                economics["execution"]["mean_worst_one_percent_realized_net_pnl"],
                ">=",
                deployed["execution"]["mean_worst_one_percent_realized_net_pnl"],
            ),
        ]
    )
    passed = all(check["passed"] for check in checks)
    return {
        "selected_candidate": (ACCURACY_ANCHORED_ADMISSION_CANDIDATE if passed else None),
        "accuracy_target": gates.target_accuracy,
        "accuracy_floor": gates.minimum_accuracy,
        "ranking_objective": "net PnL per all-core market after absolute gates",
        "candidates": {
            ACCURACY_ANCHORED_ADMISSION_CANDIDATE: {
                "passed": passed,
                "target_accuracy_reached": (economics["accuracy"] >= gates.target_accuracy),
                "checks": checks,
            }
        },
    }


def _calibration_checks(
    selected: pl.DataFrame,
    gates: PriceAwareGateConfig,
) -> list[dict[str, Any]]:
    admission = _admission_calibration(selected)
    outcome = _outcome_calibration(selected)
    net = _selected_net_calibration(selected)
    return [
        _check(
            "admission ECE remains within standard",
            admission["expected_calibration_error"],
            "<=",
            gates.maximum_expected_calibration_error,
        ),
        _check(
            "outcome ECE remains within standard",
            outcome["expected_calibration_error"],
            "<=",
            gates.maximum_expected_calibration_error,
        ),
        _check(
            "selected-net absolute bias remains within standard",
            net["absolute_bias"],
            "<=",
            gates.maximum_selected_net_bias,
        ),
    ]


def _calibration_checks_from_result(
    result: dict[str, Any],
    gates: PriceAwareGateConfig,
) -> list[dict[str, Any]]:
    return [
        _check(
            "admission ECE remains within standard",
            result["admission_probability_calibration"]["expected_calibration_error"],
            "<=",
            gates.maximum_expected_calibration_error,
        ),
        _check(
            "outcome ECE remains within standard",
            result["outcome_probability_calibration"]["expected_calibration_error"],
            "<=",
            gates.maximum_expected_calibration_error,
        ),
        _check(
            "selected-net absolute bias remains within standard",
            result["selected_net_calibration"]["absolute_bias"],
            "<=",
            gates.maximum_selected_net_bias,
        ),
    ]


def _fold_checks(
    selected: pl.DataFrame,
    economics: dict[str, Any],
    gates: PriceAwareGateConfig,
) -> list[dict[str, Any]]:
    per_side = _per_side_metrics(selected)
    checks = _standalone_checks(
        economics,
        gates,
        minimum_trades=gates.minimum_fold_trades,
    )
    checks.extend(
        [
            _check(
                "minimum Up trades",
                per_side["up"]["trades"],
                ">=",
                gates.minimum_fold_direction_trades,
            ),
            _check(
                "minimum Down trades",
                per_side["down"]["trades"],
                ">=",
                gates.minimum_fold_direction_trades,
            ),
            *_calibration_checks(selected, gates),
        ]
    )
    return checks


def _standalone_checks(
    economics: dict[str, Any],
    gates: PriceAwareGateConfig,
    *,
    minimum_trades: int,
) -> list[dict[str, Any]]:
    return [
        _check("minimum trades", economics["trades"], ">=", minimum_trades),
        _check(
            "minimum decision accuracy",
            economics["accuracy"],
            ">=",
            gates.minimum_accuracy,
        ),
        _check(
            "minimum 95% Wilson accuracy lower bound",
            economics["accuracy_wilson_lower_95"],
            ">=",
            gates.minimum_wilson_lower,
        ),
        _check(
            "minimum book-qualified market coverage",
            economics["book_qualified_coverage"],
            ">=",
            gates.minimum_coverage,
        ),
        _check(
            "positive net PnL per all-core market",
            economics["net_pnl_per_all_core_market"],
            ">",
            0.0,
        ),
        _profit_factor_check(economics, gates),
    ]


def _profit_factor_check(
    economics: dict[str, Any],
    gates: PriceAwareGateConfig,
) -> dict[str, Any]:
    execution = economics["execution"]
    observed = execution["profit_factor"]
    no_loss_profit = (
        observed is None
        and (execution["gross_profit"] or 0.0) > 0.0
        and (execution["gross_loss"] or 0.0) == 0.0
    )
    return {
        "name": "minimum profit factor",
        "observed": observed,
        "operator": ">=",
        "required": gates.minimum_profit_factor,
        "passed": bool(
            no_loss_profit or (observed is not None and observed >= gates.minimum_profit_factor)
        ),
        "no_loss_positive_profit": no_loss_profit,
    }


def _check(name: str, observed: Any, operator: str, required: Any) -> dict[str, Any]:
    if observed is None or required is None:
        passed = False
    elif operator == ">=":
        passed = observed >= required
    elif operator == ">":
        passed = observed > required
    elif operator == "<=":
        passed = observed <= required
    else:
        raise ValueError(f"unsupported check operator: {operator}")
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "required": required,
        "passed": bool(passed),
    }


def _expected_calibration_error(
    probability: np.ndarray,
    labels: np.ndarray,
) -> float:
    total = len(probability)
    error = 0.0
    for lower in np.arange(0.0, 1.0, 0.1):
        upper = lower + 0.1
        mask = (probability >= lower) & (
            probability <= upper if upper >= 1.0 else probability < upper
        )
        if np.any(mask):
            error += float(np.sum(mask) / total) * abs(
                float(np.mean(probability[mask])) - float(np.mean(labels[mask]))
            )
    return error


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _validate_unique_point_keys(frame: pl.DataFrame, name: str) -> None:
    duplicate = (
        frame.group_by("market_id", "observed_at", "seconds_elapsed")
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise RuntimeError(f"{name} contains duplicate point keys")


def _validate_feature_values(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    name: str,
) -> None:
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} is missing features: " + ", ".join(missing))
    invalid = frame.filter(
        pl.any_horizontal(
            [pl.col(feature).is_null() | ~pl.col(feature).is_finite() for feature in features]
        )
    )
    if invalid.height:
        raise RuntimeError(f"{name} contains null or non-finite features")


def _validate_universal_training_frame(frame: pl.DataFrame) -> None:
    _validate_unique_point_keys(frame, "universal outcome view")
    _validate_feature_values(
        frame,
        CORE_ORACLE_OUTCOME_FEATURES,
        "universal core-oracle outcome view",
    )
    forbidden = {
        "official_outcome",
        "final_price",
        "correct",
        "realized_up_net_per_share",
        "realized_down_net_per_share",
    }
    leaked = sorted(forbidden & set(OUTCOME_FEATURES))
    if leaked:
        raise RuntimeError("outcome feature allowlist leaks targets: " + ", ".join(leaked))


def _validate_book_training_frame(frame: pl.DataFrame) -> None:
    _validate_unique_point_keys(frame, "strict-book admission view")
    _validate_feature_values(frame, PRICE_AWARE_FEATURES, "strict-book admission view")
    required = {
        "fee_rate",
        "up_ask_vwap_5",
        "down_ask_vwap_5",
        "realized_up_net_per_share",
        "realized_down_net_per_share",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError("strict-book admission view is missing: " + ", ".join(missing))


def _validate_binary_cohort(frame: pl.DataFrame, name: str) -> None:
    if frame["market_id"].n_unique() < 2:
        raise RuntimeError(f"{name} has fewer than two markets")
    if frame["label_up"].n_unique() != 2:
        raise RuntimeError(f"{name} does not contain both outcome labels")


def _validate_admission_cohort(frame: pl.DataFrame, name: str) -> None:
    _validate_feature_values(frame, ADMISSION_FEATURES, name)
    if frame["outcome_direction_correct"].n_unique() != 2:
        raise RuntimeError(f"{name} does not contain both admission target labels")


def _validate_locked_direction(frame: pl.DataFrame, block_name: str) -> None:
    flipped = frame.filter(pl.col("predicted_up") != pl.col("outcome_predicted_up"))
    if flipped.height:
        raise RuntimeError(
            f"{block_name} admission scoring flipped {flipped.height} outcome directions"
        )


def _cohort_lineage(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "window_start_min": frame["window_start"].min().isoformat(),
        "window_start_max": frame["window_start"].max().isoformat(),
        "key_sha256": _key_fingerprint(frame),
    }


def _key_fingerprint(frame: pl.DataFrame) -> str:
    keys = frame.select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
    ).sort(["window_start", "market_id", "seconds_elapsed"])
    digest = hashlib.sha256()
    digest.update(str(keys.height).encode())
    digest.update(keys.hash_rows(seed=0).to_numpy().tobytes())
    return digest.hexdigest()


def _validated_core_config(
    config: PriceAwareBenchmarkConfig,
) -> CoreTrainingConfig:
    core_config = load_core_config(config.core_config)
    if core_config.data.source_contract != CORE_ORACLE_SOURCE_CONTRACT:
        raise RuntimeError("price-aware training requires the causal oracle source")
    return core_config


def _execution_config(
    config: PriceAwareBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> ExecutionEvidenceConfig:
    return ExecutionEvidenceConfig(
        range_start=core_config.data.range_start,
        range_end=core_config.data.range_end,
        output_dir=config.paths.execution_evidence,
        sample_interval_seconds=5,
        min_seconds_after_open=PRICE_AWARE_CONTEXT_SECONDS[0],
        max_seconds_after_open=PRICE_AWARE_CONTEXT_SECONDS[-1],
        freshness_seconds=config.model.freshness_seconds,
        quantity=config.model.quantity,
        snapshot_schema_versions=(LEGACY_SNAPSHOT_SCHEMA_VERSION,),
        decision_min_seconds_after_open=PRICE_AWARE_DECISION_SECONDS[0],
    )


def _daily_availability(
    core: pl.DataFrame,
    execution: pl.DataFrame,
    point_qualified: pl.DataFrame,
) -> list[dict[str, Any]]:
    universal = (
        core.select("market_id", "window_start")
        .unique(subset=["market_id"])
        .with_columns(pl.col("window_start").dt.date().cast(pl.String).alias("date"))
        .group_by("date")
        .agg(pl.len().alias("universal_markets"))
    )
    evidence = (
        execution.select("market_id", "window_start")
        .unique(subset=["market_id"])
        .with_columns(pl.col("window_start").dt.date().cast(pl.String).alias("date"))
        .group_by("date")
        .agg(pl.len().alias("evidence_markets"))
    )
    qualified = (
        point_qualified.select("market_id", "window_start")
        .unique(subset=["market_id"])
        .with_columns(pl.col("window_start").dt.date().cast(pl.String).alias("date"))
        .group_by("date")
        .agg(pl.len().alias("point_qualified_markets"))
    )
    return (
        universal.join(evidence, on="date", how="left")
        .join(qualified, on="date", how="left")
        .with_columns(
            pl.col("evidence_markets").fill_null(0),
            pl.col("point_qualified_markets").fill_null(0),
        )
        .with_columns(
            pl.when(pl.col("evidence_markets") == 0)
            .then(pl.lit("no_retained_snapshots"))
            .when(pl.col("point_qualified_markets") == 0)
            .then(pl.lit("no_strict_point_qualified_book"))
            .when(pl.col("point_qualified_markets") / pl.col("universal_markets") < 0.25)
            .then(pl.lit("sparse"))
            .otherwise(pl.lit("healthy"))
            .alias("availability")
        )
        .sort("date")
        .to_dicts()
    )


def _walk_forward_payload(config: PriceAwareBenchmarkConfig) -> dict[str, Any]:
    return {
        "history_start": config.walk_forward.history_start.isoformat(),
        "outcome_calibration_fraction": (config.walk_forward.outcome_calibration_fraction),
        "threshold_block_names": list(config.walk_forward.threshold_block_names),
        "evaluation_block_name": config.walk_forward.evaluation_block_name,
        "blocks": [
            {
                "name": block.name,
                "start": block.start.isoformat(),
                "end": block.end.isoformat(),
            }
            for block in config.walk_forward.blocks
        ],
    }


def _none_as_infinity(value: float | None) -> float:
    return float(value) if value is not None else float("inf")


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(path)


def _write_prediction_artifact(
    frame: pl.DataFrame,
    destination: Path,
) -> dict[str, Any]:
    _write_parquet_atomic(frame, destination)
    return {
        "path": destination.name,
        "sha256": file_sha256(destination),
        "rows": frame.height,
        "columns": frame.columns,
        "key_sha256": _key_fingerprint(frame),
    }


def _outcome_oof_artifact_frame(frame: pl.DataFrame) -> pl.DataFrame:
    columns = (
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "walk_forward_block",
        "outcome_signal_block",
        "outcome_probability_up",
        "outcome_predicted_up",
        "correct",
        "outcome_probability_selected",
        "outcome_confidence_margin",
        "outcome_direction_correct",
    )
    return frame.select(*(column for column in columns if column in frame.columns))


def _prediction_artifact_frame(frame: pl.DataFrame) -> pl.DataFrame:
    columns = (
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "walk_forward_block",
        "outcome_signal_block",
        "candidate",
        "predicted_up",
        "probability_up",
        "confidence",
        "correct",
        "outcome_probability_up",
        "outcome_predicted_up",
        "outcome_probability_selected",
        "admission_probability_correct",
        "economic_score",
        "direct_net_edge_per_share",
        "predicted_net_per_share",
        "selected_ask_vwap_5",
        "selected_fee_per_share",
        "realized_selected_net_per_share",
        "fee_rate",
        "up_ask_vwap_5",
        "down_ask_vwap_5",
        "strict_both_side_eligible",
        "execution_evidence_available",
        "model_eligible",
        "policy_selected",
    )
    return frame.select(*(column for column in columns if column in frame.columns))


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(content)
    temporary.replace(path)


def _render_report(payload: dict[str, Any]) -> str:
    rows: list[str] = []
    for candidate in payload["model_contract"]["candidate_names"]:
        economics = payload["candidates"][candidate]["economics"]
        execution = economics["execution"]
        selected = payload["selection"]["selected_candidate"] == candidate
        rows.append(
            "<tr>"
            f"<td>{html.escape(candidate)}</td>"
            f"<td>{'yes' if selected else 'no'}</td>"
            f"<td>{economics['trades']:,}</td>"
            f"<td>{economics['book_qualified_coverage']:.1%}</td>"
            f"<td>{economics['all_core_market_coverage']:.1%}</td>"
            f"<td>{economics['accuracy']:.2%}</td>"
            f"<td>{economics['accuracy_wilson_lower_95']:.2%}</td>"
            f"<td>{_format(economics['median_selected_ask_vwap_5'])}</td>"
            f"<td>{_format(economics['p90_selected_ask_vwap_5'])}</td>"
            f"<td>{_format(economics['median_seconds_elapsed'])}</td>"
            f"<td>{_format(economics['net_pnl_per_book_qualified_market'])}</td>"
            f"<td>{_format(economics['net_pnl_per_all_core_market'])}</td>"
            f"<td>{_format(execution['profit_factor'])}</td>"
            f"<td>{_format(execution['maximum_drawdown'])}</td>"
            f"<td>{_format(execution['mean_worst_one_percent_realized_net_pnl'])}</td>"
            "</tr>"
        )
    operating_rows = []
    for candidate, operating_point in payload["operating_points"].items():
        qualified = operating_point["qualified_on_threshold_blocks"]
        operating_rows.append(
            "<tr>"
            f"<td>{html.escape(candidate)}</td>"
            f"<td>{_format(operating_point['threshold'])}</td>"
            f"<td>{'control' if qualified is None else ('yes' if qualified else 'no')}</td>"
            "</tr>"
        )
    check_rows = []
    for candidate, candidate_selection in payload["selection"]["candidates"].items():
        for check in candidate_selection["checks"]:
            check_rows.append(
                "<tr>"
                f"<td>{html.escape(candidate)}</td>"
                f"<td>{html.escape(check['name'])}</td>"
                f"<td>{_format(check['observed'])}</td>"
                f"<td>{html.escape(check['operator'])} {_format(check['required'])}</td>"
                f"<td>{'pass' if check['passed'] else 'fail'}</td>"
                "</tr>"
            )
    band_rows = []
    for candidate in payload["model_contract"]["candidate_names"]:
        for band, metrics in payload["candidates"][candidate][
            "price_bands_diagnostic_only"
        ].items():
            band_rows.append(
                "<tr>"
                f"<td>{html.escape(candidate)}</td>"
                f"<td>{html.escape(band)}</td>"
                f"<td>{metrics['trades']:,}</td>"
                f"<td>{_format(metrics['accuracy'])}</td>"
                f"<td>{_format(metrics['realized_net_pnl_total'])}</td>"
                f"<td>{_format(metrics['realized_net_expectancy_per_trade'])}</td>"
                f"<td>{_format(metrics['profit_factor'])}</td>"
                "</tr>"
            )
    calibration_rows = []
    for candidate in payload["model_contract"]["candidate_names"]:
        result = payload["candidates"][candidate]
        calibration_rows.append(
            "<tr>"
            f"<td>{html.escape(candidate)}</td>"
            f"<td>{_format(result['admission_probability_calibration']['expected_calibration_error'])}</td>"
            f"<td>{_format(result['outcome_probability_calibration']['expected_calibration_error'])}</td>"
            f"<td>{_format(result['selected_net_calibration']['absolute_bias'])}</td>"
            "</tr>"
        )
    admission_point = payload["operating_points"][ACCURACY_ANCHORED_ADMISSION_CANDIDATE]
    selected_frontier = next(
        row
        for row in admission_point["frontier"]
        if row["threshold"] == admission_point["threshold"]
    )
    threshold_rows = []
    for block_name, fold in selected_frontier["folds"].items():
        metrics = fold["metrics"]
        threshold_rows.append(
            "<tr>"
            f"<td>{html.escape(block_name)}</td>"
            f"<td>{'yes' if fold['qualified'] else 'no'}</td>"
            f"<td>{metrics['trades']:,}</td>"
            f"<td>{metrics['book_qualified_coverage']:.1%}</td>"
            f"<td>{metrics['accuracy']:.2%}</td>"
            f"<td>{metrics['accuracy_wilson_lower_95']:.2%}</td>"
            f"<td>{_format(metrics['median_selected_ask_vwap_5'])}</td>"
            f"<td>{_format(metrics['p90_selected_ask_vwap_5'])}</td>"
            f"<td>{_format(metrics['net_pnl_per_all_core_market'])}</td>"
            f"<td>{_format(metrics['execution']['profit_factor'])}</td>"
            f"<td>{_format(fold['admission_calibration']['expected_calibration_error'])}</td>"
            f"<td>{_format(fold['selected_net_calibration']['absolute_bias'])}</td>"
            "</tr>"
        )
    outcome = payload["outcome_head_evaluation"]
    selection = payload["selection"]["selected_candidate"] or "none"
    availability = payload["availability"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>BTC price-aware benchmark</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;line-height:1.45;color:#17202a}}
table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}th,td{{border:1px solid #ccd1d1;padding:.45rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.card{{background:#f5f7f9;border-radius:8px;padding:1rem;margin:1rem 0}}code{{background:#eef2f3;padding:.1rem .25rem}}
</style></head><body>
<h1>BTC five-minute price-aware economic benchmark</h1>
<div class="card"><strong>Objective:</strong> {html.escape(payload["objective"])}<br>
<strong>Selected development candidate:</strong> {html.escape(selection)}<br>
<strong>Deployment:</strong> disabled; this is consumed development evidence.</div>
<h2>Matched chronological evaluation</h2>
<p>The control is a freshly trained boundary-feature-family control on these same strict-book rows. It is not the currently deployed boundary-alignment artifact.</p>
<table><thead><tr><th>Candidate</th><th>Selected</th><th>Trades</th><th>Book-qualified coverage</th><th>All-core coverage</th><th>Action accuracy</th><th>Wilson lower</th><th>Median VWAP5</th><th>P90 VWAP5</th><th>Median second</th><th>Net PnL / book-qualified market</th><th>Net PnL / all-core market</th><th>Profit factor</th><th>Max drawdown</th><th>Worst 1% mean</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<h2>Universal outcome head</h2>
<p>{outcome["markets"]:,} confirmation markets and {outcome["rows"]:,} decision rows: accuracy {_format(outcome["accuracy"])}, Wilson lower {_format(outcome["accuracy_wilson_lower_95"])}, outcome ECE {_format(outcome["calibration"]["expected_calibration_error"])}, predicted-UP rate {_format(outcome["predicted_up_rate"])}.</p>
<h2>Frozen operating points</h2>
<table><thead><tr><th>Candidate</th><th>Threshold</th><th>Qualified before confirmation</th></tr></thead><tbody>{"".join(operating_rows)}</tbody></table>
<h2>Selected threshold by blocked fold</h2>
<table><thead><tr><th>Block</th><th>Qualified</th><th>Trades</th><th>Book coverage</th><th>Accuracy</th><th>Wilson lower</th><th>Median VWAP5</th><th>P90 VWAP5</th><th>Net PnL / all-core market</th><th>Profit factor</th><th>Admission ECE</th><th>Net bias</th></tr></thead><tbody>{"".join(threshold_rows)}</tbody></table>
<h2>Evaluation checks</h2>
<table><thead><tr><th>Candidate</th><th>Requirement</th><th>Observed</th><th>Required</th><th>Result</th></tr></thead><tbody>{"".join(check_rows)}</tbody></table>
<h2>Evaluation calibration</h2>
<table><thead><tr><th>Candidate</th><th>Admission ECE</th><th>Outcome ECE</th><th>Selected-net absolute bias</th></tr></thead><tbody>{"".join(calibration_rows)}</tbody></table>
<h2>Entry-price bands</h2>
<table><thead><tr><th>Candidate</th><th>VWAP5 band</th><th>Trades</th><th>Accuracy</th><th>Net PnL</th><th>Expectancy / trade</th><th>Profit factor</th></tr></thead><tbody>{"".join(band_rows)}</tbody></table>
<h2>Availability</h2>
<p>{availability["point_qualified_rows"]:,} point-qualified rows across {availability["point_qualified_markets"]:,} markets. The evaluation contains {availability["blocks"][payload["evaluation"]["block"]]["book_qualified_markets"]:,} book-qualified markets and {availability["blocks"][payload["evaluation"]["block"]]["all_core_oracle_markets"]:,} all-core markets; both coverage denominators are reported. Only {availability["complete_decision_window_markets_diagnostic_only"]:,} markets are complete throughout the decision window, so completeness was not used as a training filter.</p>
<h2>Contract</h2>
<p>Decisions are evaluated sequentially from 60 through 240 seconds at five-second cadence. The first positive fee-inclusive expected-net action ends that market. VWAP5 and positive stored fees are in both labels and evaluation economics; Chainlink, narrow 90/120-second Binance context, boundary, and strict causal book features are inputs. Kraken and missingness are excluded. Action accuracy measures the purchased side; calibration uses only the separate outcome probability, never a synthetic action-value score.</p>
</body></html>"""


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))
