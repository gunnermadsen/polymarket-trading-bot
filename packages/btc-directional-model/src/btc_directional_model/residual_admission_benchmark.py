from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .core_benchmark import (
    AdvancementCriteria,
    BenchmarkEvidence,
    CandidatePolicy,
    _execution_metrics,
    _with_execution_columns,
    benchmark_predictions,
)
from .core_config import load_core_config
from .core_evaluation import classification_metrics
from .core_execution import (
    ExecutionEvidenceConfig,
    load_execution_evidence_manifest,
)
from .core_extract import file_sha256, write_json_atomic
from .core_features import feature_destination, load_core_feature_frame
from .core_training import ProbabilityCalibrator
from .persistence_benchmark import (
    attach_execution_evidence,
    load_execution_evidence,
)
from .policy_benchmark import (
    load_probability_evidence,
    load_saved_probability_manifest,
)
from .provenance import runtime_provenance
from .residual_admission_config import (
    COMBINED_RESIDUAL_CANDIDATE,
    EARLY_RESIDUAL_CANDIDATE,
    RESCUE_RESIDUAL_CANDIDATE,
    ResidualAdmissionBenchmarkConfig,
    ResidualAdmissionGates,
    ResidualHeadConfig,
    residual_admission_config_to_dict,
)

RESIDUAL_ADMISSION_SCHEMA_VERSION = "btc-residual-admission-benchmark-v1"
RESIDUAL_WEIGHT_POLICY = "equal_total_per_market_balanced_proposal_direction"
RESIDUAL_JOIN_KEYS = (
    "market_id",
    "window_start",
    "observed_at",
    "seconds_elapsed",
)
RESIDUAL_CORE_FEATURES = (
    "btc_cross_venue_boundary_gap_bps",
    "btc_window_open_cross_venue_basis_bps",
    "btc_boundary_terminal_volatility_z",
    "btc_boundary_cross_count",
    "btc_seconds_since_boundary_cross",
    "btc_fraction_time_boundary_positive",
    "btc_fraction_time_boundary_negative",
    "btc_boundary_distance_velocity_5s_bps",
    "btc_boundary_momentum_alignment_5s",
)
RESIDUAL_FEATURES = (
    "residual_control_confidence",
    "residual_proposal_confidence",
    "residual_confidence_advantage",
    "residual_control_threshold_shortfall",
    "residual_proposal_probability_delta_5s_oriented",
    "residual_proposal_probability_delta_15s_oriented",
    "residual_proposal_confidence_delta_5s",
    "residual_proposal_confidence_mean_15s",
    "residual_agreement_persistence",
    "residual_seconds_elapsed_fraction",
    "residual_boundary_gap_oriented",
    "residual_window_open_basis_oriented",
    "residual_boundary_terminal_volatility_z_oriented",
    "residual_boundary_cross_count",
    "residual_seconds_since_boundary_cross",
    "residual_fraction_time_on_proposed_side",
    "residual_boundary_distance_velocity_oriented",
    "residual_boundary_momentum_alignment_oriented",
)


@dataclass
class FittedResidualSelector:
    head: str
    feature_names: tuple[str, ...]
    imputation_medians: np.ndarray
    standardization_means: np.ndarray
    standardization_scales: np.ndarray
    estimator: LogisticRegression
    regularization_c: float
    row_weight_policy: str = RESIDUAL_WEIGHT_POLICY

    def raw_logit(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = _feature_matrix(frame, self.feature_names)
        filled = np.where(np.isfinite(matrix), matrix, self.imputation_medians)
        transformed = (
            filled - self.standardization_means
        ) / self.standardization_scales
        return self.estimator.decision_function(transformed).astype(np.float64)


@dataclass(frozen=True)
class ResidualThresholdSelection:
    head: str
    threshold: float | None
    qualified: bool
    thresholds_evaluated: int
    qualifying_thresholds: int
    metrics: dict[str, Any]
    objective: dict[str, Any]
    checks: tuple[dict[str, Any], ...]
    threshold_diagnostics: tuple[dict[str, Any], ...]


def run_residual_admission_benchmark(
    config: ResidualAdmissionBenchmarkConfig,
) -> tuple[Path, dict[str, Any]]:
    """Train and evaluate two causal residual-admission heads.

    This runner consumes only saved out-of-fold direction probabilities and the
    isolated pre-holdout feature cache. It never exports a runtime artifact, reads
    an independent post-freeze cohort, or mutates a trading process.
    """

    manifest = load_saved_probability_manifest(config.probability_manifest)
    core_config = load_core_config(config.core_config)
    core_features = load_core_feature_frame(core_config, "pre_holdout")
    feature_path = feature_destination(core_config, "pre_holdout")
    control_folds = load_source_validation_folds(
        config,
        manifest,
        config.control_candidate,
    )
    proposal_folds = load_source_validation_folds(
        config,
        manifest,
        config.proposal_candidate,
    )
    prepared = {
        fold_index: prepare_residual_opportunities(
            control_folds[fold_index],
            proposal_folds[fold_index],
            core_features,
            base_confidence_threshold=config.selector.base_confidence_threshold,
            agreement_cadences=config.selector.agreement_cadences,
        )
        for fold_index in range(int(manifest["fold_count"]))
    }
    execution_config = _execution_config(config)
    execution_manifest = load_execution_evidence_manifest(execution_config)
    execution = load_execution_evidence(execution_config)

    selected_by_candidate: dict[str, list[pl.DataFrame]] = {
        name: [] for name in config.candidate_names
    }
    fold_results: list[dict[str, Any]] = []
    selector_summaries: list[dict[str, Any]] = []
    eligible_market_ids: list[str] = []
    for eval_fold in config.evaluation_folds:
        result = evaluate_residual_fold(
            config,
            eval_fold=eval_fold,
            prepared_folds=prepared,
            execution=execution,
        )
        fold_results.append(result["record"])
        selector_summaries.extend(result["selector_summaries"])
        eligible_market_ids.extend(result["eligible_market_ids"])
        for candidate_name, selected in result["selected"].items():
            selected_by_candidate[candidate_name].append(selected)

    selected_all = {
        candidate_name: pl.concat(frames, how="vertical_relaxed").sort(
            ["observed_at", "market_id", "fold_index"]
        )
        for candidate_name, frames in selected_by_candidate.items()
    }
    policies = {
        candidate_name: CandidatePolicy(
            confidence_threshold=None,
            deployment_compatible=False,
            selection_mode="explicit_preselected",
        )
        for candidate_name in config.candidate_names
    }
    diagnostics = benchmark_predictions(
        selected_all,
        policies=policies,
        control_candidate=config.control_candidate,
        evidence=BenchmarkEvidence(
            label=(
                "Five rolling residual-admission evaluation folds on consumed "
                "March 21-July 6 development evidence"
            ),
            kind="development",
            independent=False,
        ),
        eligible_market_ids=eligible_market_ids,
        minimum_samples=config.gates.minimum_selected_markets,
        minimum_executable_samples=config.gates.minimum_executable_markets,
        quantity=config.quantity,
        criteria=_benchmark_criteria(config.gates),
    )
    fold_by_index = {
        int(record["fold_index"]): record for record in fold_results
    }
    control_metrics = diagnostics["candidates"][config.control_candidate][
        "own_policy"
    ]
    candidate_results: dict[str, dict[str, Any]] = {}
    for candidate_name in config.candidate_names:
        own_policy = diagnostics["candidates"][candidate_name]["own_policy"]
        attribution = (
            None
            if candidate_name == config.control_candidate
            else paired_residual_attribution(
                selected_all[config.control_candidate],
                selected_all[candidate_name],
                eligible_market_ids,
            )
        )
        residual = residual_cohort_metrics(
            selected_all[candidate_name],
            eligible_markets=len(eligible_market_ids),
            quantity=config.quantity,
        )
        checkpoints = cumulative_decision_checkpoints(
            selected_all[candidate_name],
            eligible_markets=len(eligible_market_ids),
        )
        advance = residual_advancement_checks(
            candidate_name=candidate_name,
            control_candidate=config.control_candidate,
            metrics=own_policy,
            control_metrics=control_metrics,
            checkpoints=checkpoints,
            control_checkpoints=cumulative_decision_checkpoints(
                selected_all[config.control_candidate],
                eligible_markets=len(eligible_market_ids),
            ),
            attribution=attribution,
            residual=residual,
            folds=[
                fold_by_index[index]["candidates"][candidate_name]
                for index in config.evaluation_folds
            ],
            control_folds=[
                fold_by_index[index]["candidates"][config.control_candidate]
                for index in config.evaluation_folds
            ],
            gates=config.gates,
            quantity=config.quantity,
        )
        candidate_results[candidate_name] = {
            "candidate": candidate_name,
            "out_of_fold": own_policy,
            "timing": {
                "median_first_crossing_seconds": own_policy[
                    "median_seconds_elapsed"
                ],
                "p90_first_crossing_seconds": own_policy[
                    "p90_seconds_elapsed"
                ],
            },
            "no_trade_rate": own_policy["no_trade_rate"],
            "checkpoints": checkpoints,
            "residual_cohort": residual,
            "paired_attribution": attribution,
            "folds": [
                fold_by_index[index]["candidates"][candidate_name]
                for index in config.evaluation_folds
            ],
            "advance": advance,
        }

    early_passed = bool(
        candidate_results[EARLY_RESIDUAL_CANDIDATE]["advance"][
            "benchmark_passed"
        ]
    )
    rescue_passed = bool(
        candidate_results[RESCUE_RESIDUAL_CANDIDATE]["advance"][
            "benchmark_passed"
        ]
    )
    combined_advance = candidate_results[COMBINED_RESIDUAL_CANDIDATE][
        "advance"
    ]
    combined_component_check = _check(
        "both residual heads qualify independently",
        int(early_passed and rescue_passed),
        "==",
        1,
    )
    combined_advance["checks"].append(combined_component_check)
    combined_advance["benchmark_passed"] = bool(
        combined_advance["benchmark_passed"]
        and combined_component_check["passed"]
    )
    combined_advance["development_qualified"] = combined_advance[
        "benchmark_passed"
    ]

    passing = [
        name
        for name in (
            EARLY_RESIDUAL_CANDIDATE,
            RESCUE_RESIDUAL_CANDIDATE,
            COMBINED_RESIDUAL_CANDIDATE,
        )
        if candidate_results[name]["advance"]["benchmark_passed"]
    ]
    winner = (
        COMBINED_RESIDUAL_CANDIDATE
        if COMBINED_RESIDUAL_CANDIDATE in passing
        else EARLY_RESIDUAL_CANDIDATE
        if EARLY_RESIDUAL_CANDIDATE in passing
        else RESCUE_RESIDUAL_CANDIDATE
        if RESCUE_RESIDUAL_CANDIDATE in passing
        else None
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for candidate_name, selected in selected_all.items():
        _write_parquet_atomic(
            selected,
            run_dir / f"{candidate_name}-selected.parquet",
        )
    benchmark = {
        "schema_version": RESIDUAL_ADMISSION_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": residual_admission_config_to_dict(config),
        "evaluation_note": config.evaluation_note,
        "evaluation_is_independent": config.evaluation_is_independent,
        "runtime_provenance": runtime_provenance(config.package_root),
        "probability_evidence": {
            "manifest": str(config.probability_manifest),
            "manifest_sha256": file_sha256(config.probability_manifest),
            "source_profile": manifest["source_benchmark_profile"],
            "source_candidates": manifest["candidate_names"],
            "source_fold_count": manifest["fold_count"],
            "checksums_verified": True,
        },
        "core_features": {
            "path": str(feature_path),
            "sha256": file_sha256(feature_path),
            "rows": core_features.height,
            "markets": core_features["market_id"].n_unique(),
            "holdout_accessed": False,
        },
        "execution_evidence": {
            "path": str(config.execution_evidence),
            "manifest": execution_manifest,
            "role": "strict decision-time economics diagnostic only",
            "known_gap": "no strict cached rows on June 15-29",
        },
        "selector_training": {
            "feature_names": list(RESIDUAL_FEATURES),
            "row_weight_policy": RESIDUAL_WEIGHT_POLICY,
            "models": selector_summaries,
        },
        "control_candidate": config.control_candidate,
        "proposal_candidate": config.proposal_candidate,
        "candidates": candidate_results,
        "folds": fold_results,
        "benchmark_passed_candidates": passing,
        "winner": winner,
        "deployment": {
            "status": "blocked",
            "runtime_exported": False,
            "runtime_changed": False,
            "reasons": [
                "evaluation evidence is consumed development evidence",
                "five-fold strict execution evidence is incomplete",
                "residual selector and 106-feature proposal are not runtime contracts",
                "a new forward post-freeze cohort is required",
            ],
        },
    }
    write_json_atomic(run_dir / "benchmark.json", benchmark)
    from .residual_admission_report import generate_residual_admission_report

    generate_residual_admission_report(benchmark, run_dir / "report.html")
    return run_dir, benchmark


def load_source_validation_folds(
    config: ResidualAdmissionBenchmarkConfig,
    manifest: dict[str, Any],
    candidate_name: str,
) -> dict[int, pl.DataFrame]:
    candidate = manifest["candidates"].get(candidate_name)
    if candidate is None:
        raise ValueError(f"saved probability manifest is missing {candidate_name}")
    folds: dict[int, pl.DataFrame] = {}
    for record in candidate["folds"]:
        fold_index = int(record["fold_index"])
        folds[fold_index] = load_probability_evidence(
            config.probability_manifest,
            record["validation"],
            candidate_name,
            fold_index,
            "validation",
        )
    expected = tuple(range(int(manifest["fold_count"])))
    if tuple(sorted(folds)) != expected:
        raise RuntimeError(f"{candidate_name} validation fold sequence changed")
    return folds


def prepare_residual_opportunities(
    control_rows: pl.DataFrame,
    proposal_rows: pl.DataFrame,
    core_features: pl.DataFrame,
    *,
    base_confidence_threshold: float,
    agreement_cadences: int,
) -> pl.DataFrame:
    if agreement_cadences != 2:
        raise ValueError("residual agreement must remain two exact cadences")
    control = control_rows.select(
        *RESIDUAL_JOIN_KEYS,
        "label_up",
        "fold_index",
        pl.col("probability_up").alias("control_probability_up"),
        pl.col("confidence").alias("control_confidence"),
        pl.col("predicted_up").alias("control_predicted_up"),
        pl.col("correct").alias("control_correct"),
    )
    proposal = proposal_rows.select(
        *RESIDUAL_JOIN_KEYS,
        "label_up",
        "fold_index",
        pl.col("probability_up").alias("proposal_probability_up"),
        pl.col("confidence").alias("proposal_confidence"),
        pl.col("predicted_up").alias("proposal_predicted_up"),
        pl.col("correct").alias("proposal_correct"),
    )
    joined = control.join(
        proposal,
        on=list(RESIDUAL_JOIN_KEYS),
        how="inner",
        validate="1:1",
        suffix="_proposal",
    )
    if joined.height != control.height or joined.height != proposal.height:
        raise RuntimeError("control and proposal OOF rows do not share one universe")
    if joined.filter(
        (pl.col("label_up") != pl.col("label_up_proposal"))
        | (pl.col("fold_index") != pl.col("fold_index_proposal"))
    ).height:
        raise RuntimeError("control and proposal OOF labels or folds differ")
    core_join = core_features.select(
        *RESIDUAL_JOIN_KEYS,
        *RESIDUAL_CORE_FEATURES,
    ).with_columns(pl.lit(True).alias("_residual_core_joined"))
    joined = joined.join(
        core_join,
        on=list(RESIDUAL_JOIN_KEYS),
        how="left",
        validate="1:1",
    )
    if joined["_residual_core_joined"].null_count():
        raise RuntimeError("residual feature join missed saved probability rows")
    frame = joined.drop(
        "_residual_core_joined",
        "label_up_proposal",
        "fold_index_proposal",
    ).sort(["market_id", "seconds_elapsed", "observed_at"])
    with_lags = frame.with_columns(
        pl.col("seconds_elapsed").shift(1).over("market_id").alias("_seconds_lag_1"),
        pl.col("seconds_elapsed").shift(3).over("market_id").alias("_seconds_lag_3"),
        pl.col("proposal_probability_up")
        .shift(1)
        .over("market_id")
        .alias("_proposal_probability_lag_1"),
        pl.col("proposal_probability_up")
        .shift(3)
        .over("market_id")
        .alias("_proposal_probability_lag_3"),
        pl.col("proposal_confidence")
        .shift(1)
        .over("market_id")
        .alias("_proposal_confidence_lag_1"),
        pl.col("proposal_confidence")
        .shift(2)
        .over("market_id")
        .alias("_proposal_confidence_lag_2"),
        pl.col("proposal_confidence")
        .shift(3)
        .over("market_id")
        .alias("_proposal_confidence_lag_3"),
        pl.col("proposal_predicted_up")
        .shift(1)
        .over("market_id")
        .alias("_proposal_direction_lag_1"),
        (
            pl.col("control_predicted_up") == pl.col("proposal_predicted_up")
        ).alias("_agreement_now"),
    )
    with_lags = with_lags.with_columns(
        pl.col("_agreement_now")
        .shift(1)
        .over("market_id")
        .fill_null(False)
        .alias("_agreement_previous"),
        (
            pl.col("control_confidence") >= base_confidence_threshold
        ).alias("_control_crossed_now"),
    )
    with_lags = with_lags.with_columns(
        pl.col("_control_crossed_now")
        .cast(pl.Int32)
        .cum_sum()
        .over("market_id")
        .alias("_control_crossings_through_now")
    )
    exact_5s = (
        pl.col("seconds_elapsed") - pl.col("_seconds_lag_1")
    ) == 5
    exact_15s = (
        pl.col("seconds_elapsed") - pl.col("_seconds_lag_3")
    ) == 15
    direction_sign = (
        pl.col("proposal_predicted_up").cast(pl.Float64) * 2.0 - 1.0
    )
    proposal_side_fraction = (
        pl.when(pl.col("proposal_predicted_up") == 1)
        .then(pl.col("btc_fraction_time_boundary_positive"))
        .otherwise(pl.col("btc_fraction_time_boundary_negative"))
    )
    output = with_lags.with_columns(
        pl.col("control_confidence")
        .cast(pl.Float64)
        .alias("residual_control_confidence"),
        pl.col("proposal_confidence")
        .cast(pl.Float64)
        .alias("residual_proposal_confidence"),
        (
            pl.col("proposal_confidence") - pl.col("control_confidence")
        ).alias("residual_confidence_advantage"),
        (
            pl.lit(base_confidence_threshold) - pl.col("control_confidence")
        )
        .clip(lower_bound=0.0)
        .alias("residual_control_threshold_shortfall"),
        pl.when(exact_5s)
        .then(
            direction_sign
            * (
                pl.col("proposal_probability_up")
                - pl.col("_proposal_probability_lag_1")
            )
        )
        .otherwise(None)
        .alias("residual_proposal_probability_delta_5s_oriented"),
        pl.when(exact_15s)
        .then(
            direction_sign
            * (
                pl.col("proposal_probability_up")
                - pl.col("_proposal_probability_lag_3")
            )
        )
        .otherwise(None)
        .alias("residual_proposal_probability_delta_15s_oriented"),
        pl.when(exact_5s)
        .then(
            pl.col("proposal_confidence")
            - pl.col("_proposal_confidence_lag_1")
        )
        .otherwise(None)
        .alias("residual_proposal_confidence_delta_5s"),
        pl.when(exact_15s)
        .then(
            (
                pl.col("proposal_confidence")
                + pl.col("_proposal_confidence_lag_1")
                + pl.col("_proposal_confidence_lag_2")
                + pl.col("_proposal_confidence_lag_3")
            )
            / 4.0
        )
        .otherwise(None)
        .alias("residual_proposal_confidence_mean_15s"),
        ((
            pl.col("_agreement_now").cast(pl.Float64)
            + (
                exact_5s
                & pl.col("_agreement_previous")
                & (
                    pl.col("proposal_predicted_up")
                    == pl.col("_proposal_direction_lag_1")
                )
            ).cast(pl.Float64)
        )
        / 2.0).alias("residual_agreement_persistence"),
        (pl.col("seconds_elapsed").cast(pl.Float64) / 300.0).alias(
            "residual_seconds_elapsed_fraction"
        ),
        (
            direction_sign * pl.col("btc_cross_venue_boundary_gap_bps")
        ).alias("residual_boundary_gap_oriented"),
        (
            direction_sign
            * pl.col("btc_window_open_cross_venue_basis_bps")
        ).alias("residual_window_open_basis_oriented"),
        (
            direction_sign * pl.col("btc_boundary_terminal_volatility_z")
        ).alias("residual_boundary_terminal_volatility_z_oriented"),
        pl.col("btc_boundary_cross_count")
        .cast(pl.Float64)
        .alias("residual_boundary_cross_count"),
        pl.col("btc_seconds_since_boundary_cross")
        .cast(pl.Float64)
        .alias("residual_seconds_since_boundary_cross"),
        proposal_side_fraction.cast(pl.Float64).alias(
            "residual_fraction_time_on_proposed_side"
        ),
        (
            direction_sign
            * pl.col("btc_boundary_distance_velocity_5s_bps")
        ).alias("residual_boundary_distance_velocity_oriented"),
        (
            direction_sign
            * pl.col("btc_boundary_momentum_alignment_5s")
        ).alias("residual_boundary_momentum_alignment_oriented"),
        (
            (pl.col("_control_crossings_through_now") == 0)
            & pl.col("_agreement_now")
            & pl.col("_agreement_previous")
            & exact_5s
            & (
                pl.col("proposal_predicted_up")
                == pl.col("_proposal_direction_lag_1")
            )
        ).alias("residual_opportunity"),
    )
    temporary = [
        name
        for name in output.columns
        if name.startswith("_")
    ]
    return output.drop(temporary)


def evaluate_residual_fold(
    config: ResidualAdmissionBenchmarkConfig,
    *,
    eval_fold: int,
    prepared_folds: dict[int, pl.DataFrame],
    execution: pl.DataFrame,
) -> dict[str, Any]:
    prior_fold = eval_fold - 1
    fit_fold_indexes = tuple(range(prior_fold))
    if not fit_fold_indexes:
        raise RuntimeError("residual evaluation has no earlier fitting fold")
    prior_calibration, prior_policy, split = split_market_cohort(
        prepared_folds[prior_fold]
    )
    evaluation = prepared_folds[eval_fold]
    fit_source = pl.concat(
        [prepared_folds[index] for index in fit_fold_indexes],
        how="vertical_relaxed",
    )
    if fit_source["window_start"].max() >= prior_calibration["window_start"].min():
        raise RuntimeError(f"residual fold {eval_fold} fitting is not causal")
    if prior_calibration["window_start"].max() >= prior_policy["window_start"].min():
        raise RuntimeError(f"residual fold {eval_fold} calibration is not causal")
    if prior_policy["window_start"].max() >= evaluation["window_start"].min():
        raise RuntimeError(f"residual fold {eval_fold} policy is not causal")

    head_outputs: dict[str, dict[str, Any]] = {}
    selector_summaries: list[dict[str, Any]] = []
    for head in (config.early_head, config.rescue_head):
        fit_rows = head_opportunity_rows(fit_source, head)
        calibration_rows = head_opportunity_rows(prior_calibration, head)
        policy_rows = head_opportunity_rows(prior_policy, head)
        evaluation_rows = head_opportunity_rows(evaluation, head)
        model = fit_residual_selector(
            fit_rows,
            head=head.name,
            regularization_c=config.selector.regularization_c,
            random_seed=config.selector.random_seed,
        )
        calibrators, calibration_diagnostics = (
            fit_direction_platt_calibrators(
                model,
                calibration_rows,
                minimum_rows=config.selector.minimum_calibration_rows_per_direction,
                minimum_markets=(
                    config.selector.minimum_calibration_markets_per_direction
                ),
                random_seed=config.selector.random_seed,
            )
        )
        policy_scored = score_residual_q(
            model,
            calibrators,
            policy_rows,
            head=head.name,
        )
        selection = select_residual_threshold(
            prior_policy,
            policy_scored,
            head=head,
            config=config,
        )
        evaluation_scored = score_residual_q(
            model,
            calibrators,
            evaluation_rows,
            head=head.name,
        )
        head_outputs[head.name] = {
            "head": head,
            "selection": selection,
            "evaluation_scored": evaluation_scored,
        }
        selector_summaries.append(
            {
                "evaluation_fold": eval_fold,
                "head": head.name,
                "fit_folds": list(fit_fold_indexes),
                "fit_range": _frame_range(fit_rows),
                "calibration_range": _frame_range(calibration_rows),
                "policy_selection_range": _frame_range(policy_rows),
                "evaluation_range": _frame_range(evaluation_rows),
                "model": selector_model_summary(model),
                "calibrators": calibration_diagnostics,
                "threshold_selection": _selection_payload(selection),
            }
        )

    early_selection = head_outputs["early"]["selection"]
    rescue_selection = head_outputs["rescue"]["selection"]
    selected = {
        config.control_candidate: compose_control_priority_policy(
            evaluation,
            candidate_name=config.control_candidate,
            base_confidence_threshold=config.selector.base_confidence_threshold,
        ),
        config.early_head.candidate: compose_control_priority_policy(
            evaluation,
            candidate_name=config.early_head.candidate,
            base_confidence_threshold=config.selector.base_confidence_threshold,
            early=(
                head_outputs["early"]["evaluation_scored"],
                early_selection.threshold,
            ),
        ),
        config.rescue_head.candidate: compose_control_priority_policy(
            evaluation,
            candidate_name=config.rescue_head.candidate,
            base_confidence_threshold=config.selector.base_confidence_threshold,
            rescue=(
                head_outputs["rescue"]["evaluation_scored"],
                rescue_selection.threshold,
            ),
        ),
        config.combined_candidate: compose_control_priority_policy(
            evaluation,
            candidate_name=config.combined_candidate,
            base_confidence_threshold=config.selector.base_confidence_threshold,
            early=(
                head_outputs["early"]["evaluation_scored"],
                early_selection.threshold,
            ),
            rescue=(
                head_outputs["rescue"]["evaluation_scored"],
                rescue_selection.threshold,
            ),
        ),
    }
    eligible_ids = sorted(evaluation["market_id"].unique().to_list())
    selected_with_execution = {
        name: attach_execution_evidence(frame, execution)
        for name, frame in selected.items()
    }
    control = selected_with_execution[config.control_candidate]
    candidates: dict[str, Any] = {}
    for candidate_name, frame in selected_with_execution.items():
        metrics = policy_metrics(
            frame,
            eligible_markets=len(eligible_ids),
            quantity=config.quantity,
        )
        attribution = (
            None
            if candidate_name == config.control_candidate
            else paired_residual_attribution(
                control,
                frame,
                eligible_ids,
            )
        )
        candidates[candidate_name] = {
            "fold_index": eval_fold,
            "metrics": metrics,
            "checkpoints": cumulative_decision_checkpoints(
                frame,
                eligible_markets=len(eligible_ids),
            ),
            "residual_cohort": residual_cohort_metrics(
                frame,
                eligible_markets=len(eligible_ids),
                quantity=config.quantity,
            ),
            "paired_attribution": attribution,
            "selection_qualified": (
                True
                if candidate_name == config.control_candidate
                else early_selection.qualified
                if candidate_name == config.early_head.candidate
                else rescue_selection.qualified
                if candidate_name == config.rescue_head.candidate
                else early_selection.qualified and rescue_selection.qualified
            ),
        }
    return {
        "eligible_market_ids": eligible_ids,
        "selected": selected_with_execution,
        "selector_summaries": selector_summaries,
        "record": {
            "fold_index": eval_fold,
            "causal_order_verified": True,
            "selector_fit_folds": list(fit_fold_indexes),
            "prior_fold_index": prior_fold,
            "prior_fold_split": split,
            "thresholds": {
                "early": _selection_payload(early_selection),
                "rescue": _selection_payload(rescue_selection),
            },
            "candidates": candidates,
        },
    }


def split_market_cohort(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    markets = (
        frame.select("market_id", "window_start")
        .unique()
        .sort(["window_start", "market_id"])
    )
    if markets.height < 2:
        raise ValueError("prior fold needs at least two markets")
    split_index = markets.height // 2
    calibration_markets = markets.head(split_index)
    policy_markets = markets.slice(split_index)
    calibration = frame.join(
        calibration_markets.select("market_id"),
        on="market_id",
        how="semi",
    )
    policy = frame.join(
        policy_markets.select("market_id"),
        on="market_id",
        how="semi",
    )
    if calibration["window_start"].max() >= policy["window_start"].min():
        raise RuntimeError("residual prior-fold split overlaps")
    return (
        calibration,
        policy,
        {
            "method": "chronological_market_50_50",
            "source_markets": markets.height,
            "calibration_markets": calibration_markets.height,
            "policy_markets": policy_markets.height,
        },
    )


def head_opportunity_rows(
    frame: pl.DataFrame,
    head: ResidualHeadConfig,
) -> pl.DataFrame:
    rows = frame.filter(
        pl.col("residual_opportunity")
        & pl.col("seconds_elapsed").is_between(
            head.start_second,
            head.end_second_exclusive,
            closed="left",
        )
    )
    if rows.is_empty():
        raise RuntimeError(f"{head.name} residual head has no opportunity rows")
    return rows


def fit_residual_selector(
    frame: pl.DataFrame,
    *,
    head: str,
    regularization_c: float,
    random_seed: int,
) -> FittedResidualSelector:
    labels = frame["proposal_correct"].cast(pl.Int8).to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError(f"{head} residual fitting needs both correctness classes")
    matrix = _feature_matrix(frame, RESIDUAL_FEATURES)
    medians = _finite_medians(matrix)
    filled = np.where(np.isfinite(matrix), matrix, medians)
    weights = balanced_direction_market_weights(frame)
    means = np.average(filled, axis=0, weights=weights)
    variance = np.average((filled - means) ** 2, axis=0, weights=weights)
    scales = np.sqrt(np.maximum(variance, 0.0))
    scales = np.where(scales > 1e-12, scales, 1.0)
    transformed = (filled - means) / scales
    estimator = LogisticRegression(
        C=regularization_c,
        solver="lbfgs",
        max_iter=2_000,
        tol=1e-7,
        random_state=random_seed,
    )
    with threadpool_limits(limits=1):
        estimator.fit(transformed, labels, sample_weight=weights)
    if int(estimator.n_iter_[0]) >= estimator.max_iter:
        raise RuntimeError(f"{head} residual selector did not converge")
    return FittedResidualSelector(
        head=head,
        feature_names=RESIDUAL_FEATURES,
        imputation_medians=medians,
        standardization_means=means,
        standardization_scales=scales,
        estimator=estimator,
        regularization_c=regularization_c,
    )


def fit_direction_platt_calibrators(
    selector: FittedResidualSelector,
    calibration_rows: pl.DataFrame,
    *,
    minimum_rows: int,
    minimum_markets: int,
    random_seed: int,
) -> tuple[dict[int, ProbabilityCalibrator], list[dict[str, Any]]]:
    raw_logit = selector.raw_logit(calibration_rows)
    directions = calibration_rows["proposal_predicted_up"].to_numpy()
    correct = calibration_rows["proposal_correct"].cast(pl.Int8).to_numpy()
    calibrators: dict[int, ProbabilityCalibrator] = {}
    diagnostics: list[dict[str, Any]] = []
    for direction in (0, 1):
        mask = directions == direction
        rows = calibration_rows.filter(pl.Series(mask))
        if rows.height < minimum_rows:
            raise RuntimeError(
                f"{selector.head} direction {direction} calibration has too few rows"
            )
        if rows["market_id"].n_unique() < minimum_markets:
            raise RuntimeError(
                f"{selector.head} direction {direction} calibration has too few markets"
            )
        labels = correct[mask]
        if set(np.unique(labels)) != {0, 1}:
            raise RuntimeError(
                f"{selector.head} direction {direction} calibration needs both classes"
            )
        weights = equal_market_weights(rows)
        estimator = LogisticRegression(
            C=1_000_000,
            solver="lbfgs",
            max_iter=500,
            tol=1e-9,
            random_state=random_seed,
        )
        with threadpool_limits(limits=1):
            estimator.fit(
                raw_logit[mask].reshape(-1, 1),
                labels,
                sample_weight=weights,
            )
        calibrator = ProbabilityCalibrator(
            slope=float(estimator.coef_[0, 0]),
            intercept=float(estimator.intercept_[0]),
            converged=bool(estimator.n_iter_[0] < estimator.max_iter),
            iterations=int(estimator.n_iter_[0]),
        )
        if not calibrator.converged or calibrator.slope <= 0.0:
            raise RuntimeError(
                f"{selector.head} direction {direction} calibration is non-monotonic"
            )
        calibrators[direction] = calibrator
        diagnostics.append(
            {
                "proposed_direction": "UP" if direction else "DOWN",
                "rows": rows.height,
                "markets": rows["market_id"].n_unique(),
                **asdict(calibrator),
            }
        )
    return calibrators, diagnostics


def score_residual_q(
    selector: FittedResidualSelector,
    calibrators: dict[int, ProbabilityCalibrator],
    frame: pl.DataFrame,
    *,
    head: str,
) -> pl.DataFrame:
    if set(calibrators) != {0, 1}:
        raise ValueError("residual direction calibrators are incomplete")
    raw_logit = selector.raw_logit(frame)
    directions = frame["proposal_predicted_up"].to_numpy()
    q = np.full(frame.height, np.nan, dtype=np.float64)
    for direction in (0, 1):
        mask = directions == direction
        q[mask] = calibrators[direction].probability(raw_logit[mask])
    if not np.isfinite(q).all() or np.any((q < 0.0) | (q > 1.0)):
        raise RuntimeError(f"{head} residual calibration produced invalid q")
    return frame.with_columns(
        pl.Series("residual_q", q, dtype=pl.Float64),
        pl.lit(head).alias("residual_head"),
    )


def select_residual_threshold(
    full_policy_rows: pl.DataFrame,
    scored_opportunities: pl.DataFrame,
    *,
    head: ResidualHeadConfig,
    config: ResidualAdmissionBenchmarkConfig,
) -> ResidualThresholdSelection:
    eligible = full_policy_rows["market_id"].n_unique()
    control = compose_control_priority_policy(
        full_policy_rows,
        candidate_name=config.control_candidate,
        base_confidence_threshold=config.selector.base_confidence_threshold,
    )
    control_metrics = classification_metrics(control, eligible_markets=eligible)
    best: tuple[tuple[Any, ...], ResidualThresholdSelection] | None = None
    qualifying = 0
    threshold_diagnostics: list[dict[str, Any]] = []
    for threshold in head.threshold_candidates:
        candidate = compose_control_priority_policy(
            full_policy_rows,
            candidate_name=head.candidate,
            base_confidence_threshold=config.selector.base_confidence_threshold,
            **{head.name: (scored_opportunities, threshold)},
        )
        metrics = classification_metrics(candidate, eligible_markets=eligible)
        timing = decision_timing(candidate, eligible_markets=eligible)
        residual = candidate.filter(pl.col("decision_source") == head.name)
        residual_metrics = classification_metrics(
            residual,
            eligible_markets=eligible,
        )
        checkpoints = cumulative_decision_checkpoints(
            candidate,
            eligible_markets=eligible,
        )
        control_checkpoints = cumulative_decision_checkpoints(
            control,
            eligible_markets=eligible,
        )
        checks = tuple(
            _policy_selection_checks(
                metrics,
                control_metrics,
                residual_metrics,
                config.gates,
            )
        )
        qualified = all(check["passed"] for check in checks)
        qualifying += int(qualified)
        objective = {
            "coverage": metrics["coverage"],
            "decisions_by_120": checkpoints["120"],
            "decisions_by_120_uplift": (
                checkpoints["120"] - control_checkpoints["120"]
            ),
            "median_seconds_elapsed": timing["median_first_crossing_seconds"],
            "residual_markets": residual.height,
            "residual_accuracy": residual_metrics["accuracy"],
            "rescued_markets": paired_residual_attribution(
                control,
                candidate,
                full_policy_rows["market_id"].unique().to_list(),
            )["categories"]["rescued_control_no_trade"]["markets"],
        }
        selection = ResidualThresholdSelection(
            head=head.name,
            threshold=float(threshold),
            qualified=qualified,
            thresholds_evaluated=len(head.threshold_candidates),
            qualifying_thresholds=0,
            metrics=metrics,
            objective=objective,
            checks=checks,
            threshold_diagnostics=(),
        )
        threshold_diagnostics.append(
            {
                "threshold": float(threshold),
                "qualified": qualified,
                "metrics": metrics,
                "objective": objective,
                "checks": checks,
            }
        )
        if not qualified:
            continue
        if head.name == "early":
            rank = (
                objective["decisions_by_120"],
                -(objective["median_seconds_elapsed"] or float("inf")),
                objective["coverage"],
                objective["residual_accuracy"],
                threshold,
            )
        else:
            rank = (
                objective["rescued_markets"],
                objective["coverage"],
                objective["residual_accuracy"],
                threshold,
            )
        if best is None or rank > best[0]:
            best = (rank, selection)
    if best is None:
        return ResidualThresholdSelection(
            head=head.name,
            threshold=None,
            qualified=False,
            thresholds_evaluated=len(head.threshold_candidates),
            qualifying_thresholds=0,
            metrics=control_metrics,
            objective={
                "fallback": "control",
                "reason": "no threshold passed the frozen policy-selection gates",
            },
            checks=(),
            threshold_diagnostics=tuple(threshold_diagnostics),
        )
    selected = best[1]
    return ResidualThresholdSelection(
        **{
            **asdict(selected),
            "qualifying_thresholds": qualifying,
            "threshold_diagnostics": tuple(threshold_diagnostics),
        }
    )


def compose_control_priority_policy(
    full_rows: pl.DataFrame,
    *,
    candidate_name: str,
    base_confidence_threshold: float,
    early: tuple[pl.DataFrame, float | None] | None = None,
    rescue: tuple[pl.DataFrame, float | None] | None = None,
) -> pl.DataFrame:
    control = (
        full_rows.filter(
            pl.col("control_confidence") >= base_confidence_threshold
        )
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .group_by("market_id", maintain_order=True)
        .first()
    )
    decision_frames = [
        _control_decision_rows(control),
    ]
    for name, value in (("early", early), ("rescue", rescue)):
        if value is None:
            continue
        scored, threshold = value
        if threshold is None:
            continue
        residual = (
            scored.filter(pl.col("residual_q") >= threshold)
            .sort(["market_id", "seconds_elapsed", "observed_at"])
            .group_by("market_id", maintain_order=True)
            .first()
        )
        decision_frames.append(_residual_decision_rows(residual, name))
    decisions = pl.concat(decision_frames, how="diagonal_relaxed")
    decisions = (
        decisions.with_columns(
            pl.when(pl.col("decision_source") == "control")
            .then(0)
            .when(pl.col("decision_source") == "early")
            .then(1)
            .otherwise(2)
            .cast(pl.Int8)
            .alias("_decision_priority")
        )
        .sort(
            [
                "market_id",
                "seconds_elapsed",
                "_decision_priority",
                "observed_at",
            ]
        )
        .group_by("market_id", maintain_order=True)
        .first()
        .drop("_decision_priority")
        .with_columns(
            pl.lit(candidate_name).alias("candidate"),
            pl.lit(True).alias("model_eligible"),
            pl.lit(True).alias("policy_selected"),
        )
        .sort(["observed_at", "market_id"])
    )
    if decisions["market_id"].n_unique() != decisions.height:
        raise RuntimeError(f"{candidate_name} selected a market more than once")
    return decisions


def _control_decision_rows(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(
        *RESIDUAL_JOIN_KEYS,
        "label_up",
        "fold_index",
        pl.col("control_probability_up").alias("probability_up"),
        pl.col("control_confidence").alias("confidence"),
        pl.col("control_predicted_up").cast(pl.Int8).alias("predicted_up"),
        pl.col("control_correct").cast(pl.Boolean).alias("correct"),
        pl.lit("control").alias("decision_source"),
        pl.lit(None, dtype=pl.Float64).alias("residual_q"),
    )


def _residual_decision_rows(
    frame: pl.DataFrame,
    head: str,
) -> pl.DataFrame:
    q = pl.col("residual_q")
    return frame.select(
        *RESIDUAL_JOIN_KEYS,
        "label_up",
        "fold_index",
        pl.when(pl.col("proposal_predicted_up") == 1)
        .then(q)
        .otherwise(1.0 - q)
        .alias("probability_up"),
        q.alias("confidence"),
        pl.col("proposal_predicted_up").cast(pl.Int8).alias("predicted_up"),
        pl.col("proposal_correct").cast(pl.Boolean).alias("correct"),
        pl.lit(head).alias("decision_source"),
        "residual_q",
    )


def decision_timing(
    selected: pl.DataFrame,
    *,
    eligible_markets: int,
) -> dict[str, Any]:
    if selected.is_empty():
        return {
            "markets": 0,
            "eligible_markets": eligible_markets,
            "coverage": 0.0,
            "median_first_crossing_seconds": None,
            "p90_first_crossing_seconds": None,
        }
    elapsed = selected["seconds_elapsed"].to_numpy().astype(np.float64)
    return {
        "markets": selected.height,
        "eligible_markets": eligible_markets,
        "coverage": selected.height / eligible_markets if eligible_markets else 0.0,
        "median_first_crossing_seconds": float(np.median(elapsed)),
        "p90_first_crossing_seconds": float(np.quantile(elapsed, 0.9)),
    }


def policy_metrics(
    selected: pl.DataFrame,
    *,
    eligible_markets: int,
    quantity: float,
) -> dict[str, Any]:
    metrics = classification_metrics(
        selected,
        eligible_markets=eligible_markets,
    )
    timing = decision_timing(
        selected,
        eligible_markets=eligible_markets,
    )
    return {
        **metrics,
        "no_trade_markets": eligible_markets - selected.height,
        "no_trade_rate": (
            (eligible_markets - selected.height) / eligible_markets
            if eligible_markets
            else 0.0
        ),
        "median_seconds_elapsed": timing["median_first_crossing_seconds"],
        "p90_seconds_elapsed": timing["p90_first_crossing_seconds"],
        "execution": _execution_metrics(selected, quantity=quantity),
    }


def residual_cohort_metrics(
    selected: pl.DataFrame,
    *,
    eligible_markets: int,
    quantity: float,
) -> dict[str, Any]:
    residual = selected.filter(pl.col("decision_source") != "control")
    metrics = classification_metrics(
        residual,
        eligible_markets=eligible_markets,
    )
    metrics["execution"] = _execution_metrics(residual, quantity=quantity)
    metrics["hourly_net_bootstrap"] = hourly_net_bootstrap(
        residual,
        quantity=quantity,
    )
    return metrics


def hourly_net_bootstrap(
    rows: pl.DataFrame,
    *,
    quantity: float,
    resamples: int = 10_000,
    random_seed: int = 20260728,
) -> dict[str, Any]:
    economic = _with_execution_columns(rows, quantity=quantity).filter(
        pl.col("_execution_available")
        & pl.col("_realized_net_pnl").is_not_null()
        & pl.col("_realized_net_pnl").is_finite()
    )
    if economic.is_empty():
        return {
            "blocks": 0,
            "resamples": resamples,
            "observed": None,
            "lower_95": None,
            "upper_95": None,
        }
    blocks = (
        economic.with_columns(
            pl.col("window_start").dt.truncate("1h").alias("_hour")
        )
        .group_by("_hour")
        .agg(
            pl.col("_realized_net_pnl").sum().alias("net"),
            pl.len().alias("markets"),
        )
        .sort("_hour")
    )
    net = blocks["net"].to_numpy().astype(np.float64)
    markets = blocks["markets"].to_numpy().astype(np.float64)
    rng = np.random.default_rng(random_seed)
    indexes = rng.integers(0, len(net), size=(resamples, len(net)))
    distribution = net[indexes].sum(axis=1) / markets[indexes].sum(axis=1)
    return {
        "blocks": len(net),
        "resamples": resamples,
        "observed": float(net.sum() / markets.sum()),
        "lower_95": float(np.quantile(distribution, 0.025)),
        "upper_95": float(np.quantile(distribution, 0.975)),
    }


def cumulative_decision_checkpoints(
    selected: pl.DataFrame,
    *,
    eligible_markets: int,
) -> dict[str, float]:
    return {
        str(second): (
            selected.filter(pl.col("seconds_elapsed") <= second).height
            / eligible_markets
            if eligible_markets
            else 0.0
        )
        for second in (60, 90, 120, 180, 240)
    }


def paired_residual_attribution(
    control: pl.DataFrame,
    candidate: pl.DataFrame,
    eligible_market_ids: list[str],
) -> dict[str, Any]:
    labels = pl.DataFrame({"market_id": eligible_market_ids})
    control_rows = control.select(
        "market_id",
        pl.col("seconds_elapsed").alias("control_seconds"),
        pl.col("predicted_up").alias("control_predicted_up"),
        pl.col("correct").alias("control_correct"),
    )
    candidate_rows = candidate.select(
        "market_id",
        pl.col("seconds_elapsed").alias("candidate_seconds"),
        pl.col("predicted_up").alias("candidate_predicted_up"),
        pl.col("correct").alias("candidate_correct"),
        "decision_source",
    )
    joined = labels.join(control_rows, on="market_id", how="left").join(
        candidate_rows,
        on="market_id",
        how="left",
    )
    expressions = {
        "preserved_control": (
            pl.col("control_seconds").is_not_null()
            & pl.col("candidate_seconds").is_not_null()
            & (pl.col("candidate_seconds") == pl.col("control_seconds"))
            & (
                pl.col("candidate_predicted_up")
                == pl.col("control_predicted_up")
            )
        ),
        "earlier_same_direction": (
            pl.col("control_seconds").is_not_null()
            & pl.col("candidate_seconds").is_not_null()
            & (pl.col("candidate_seconds") < pl.col("control_seconds"))
            & (
                pl.col("candidate_predicted_up")
                == pl.col("control_predicted_up")
            )
        ),
        "earlier_preempted_direction_change": (
            pl.col("control_seconds").is_not_null()
            & pl.col("candidate_seconds").is_not_null()
            & (pl.col("candidate_seconds") < pl.col("control_seconds"))
            & (
                pl.col("candidate_predicted_up")
                != pl.col("control_predicted_up")
            )
        ),
        "rescued_control_no_trade": (
            pl.col("control_seconds").is_null()
            & pl.col("candidate_seconds").is_not_null()
        ),
        "lost_control": (
            pl.col("control_seconds").is_not_null()
            & pl.col("candidate_seconds").is_null()
        ),
        "still_no_trade": (
            pl.col("control_seconds").is_null()
            & pl.col("candidate_seconds").is_null()
        ),
    }
    categories: dict[str, Any] = {}
    assigned = np.zeros(joined.height, dtype=np.int8)
    for name, expression in expressions.items():
        mask = joined.select(expression.alias("mask"))["mask"].fill_null(False)
        assigned += mask.cast(pl.Int8).to_numpy()
        rows = joined.filter(mask)
        categories[name] = {
            "markets": rows.height,
            "accuracy": (
                float(rows["candidate_correct"].mean())
                if rows.height and rows["candidate_correct"].null_count() < rows.height
                else None
            ),
            "median_entry_difference_seconds": (
                float(
                    (
                        rows["control_seconds"] - rows["candidate_seconds"]
                    ).median()
                )
                if rows.height
                and rows["control_seconds"].null_count() == 0
                and rows["candidate_seconds"].null_count() == 0
                else None
            ),
            "added_wrong_trades": (
                rows.filter(~pl.col("candidate_correct")).height
                if name
                in {
                    "earlier_preempted_direction_change",
                    "rescued_control_no_trade",
                }
                else 0
            ),
        }
    if np.any(assigned != 1):
        raise RuntimeError("residual attribution is not disjoint and exhaustive")
    earlier = joined.filter(
        pl.col("control_seconds").is_not_null()
        & pl.col("candidate_seconds").is_not_null()
        & (pl.col("candidate_seconds") < pl.col("control_seconds"))
    )
    return {
        "eligible_markets": joined.height,
        "categories": categories,
        "advanced_markets": earlier.height,
        "median_advancement_seconds": (
            float(
                (
                    earlier["control_seconds"] - earlier["candidate_seconds"]
                ).median()
            )
            if earlier.height
            else None
        ),
        "lost_control_markets": categories["lost_control"]["markets"],
        "rescued_markets": categories["rescued_control_no_trade"]["markets"],
    }


def equal_market_weights(frame: pl.DataFrame) -> np.ndarray:
    market_ids = frame["market_id"].cast(pl.String).to_numpy()
    _, inverse, counts = np.unique(
        market_ids,
        return_inverse=True,
        return_counts=True,
    )
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights *= len(weights) / weights.sum()
    return weights


def balanced_direction_market_weights(frame: pl.DataFrame) -> np.ndarray:
    weights = equal_market_weights(frame)
    directions = frame["proposal_predicted_up"].cast(pl.Int8).to_numpy()
    direction_totals = {
        direction: weights[directions == direction].sum()
        for direction in (0, 1)
    }
    if any(total <= 0.0 for total in direction_totals.values()):
        raise RuntimeError("residual weights require both proposal directions")
    target = weights.sum() / 2.0
    for direction in (0, 1):
        weights[directions == direction] *= (
            target / direction_totals[direction]
        )
    weights *= len(weights) / weights.sum()
    return weights


def selector_model_summary(
    model: FittedResidualSelector,
) -> dict[str, Any]:
    return {
        "family": "logistic",
        "head": model.head,
        "target": "proposal direction equals official outcome",
        "feature_names": list(model.feature_names),
        "regularization_c": model.regularization_c,
        "row_weight_policy": model.row_weight_policy,
        "coefficients": {
            feature: float(value)
            for feature, value in zip(
                model.feature_names,
                model.estimator.coef_[0],
                strict=True,
            )
        },
        "intercept": float(model.estimator.intercept_[0]),
        "iterations": int(model.estimator.n_iter_[0]),
        "converged": int(model.estimator.n_iter_[0]) < model.estimator.max_iter,
    }


def residual_advancement_checks(
    *,
    candidate_name: str,
    control_candidate: str,
    metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    checkpoints: dict[str, float],
    control_checkpoints: dict[str, float],
    attribution: dict[str, Any] | None,
    residual: dict[str, Any],
    folds: list[dict[str, Any]],
    control_folds: list[dict[str, Any]],
    gates: ResidualAdmissionGates,
    quantity: float,
) -> dict[str, Any]:
    if candidate_name == control_candidate:
        return {
            "is_control": True,
            "checks": [],
            "benchmark_passed": None,
            "deployment_qualified": False,
        }
    checks = _quality_checks(metrics, control_metrics, gates)
    checks.extend(
        (
            _check(
                "minimum selected markets",
                metrics["markets"],
                ">=",
                gates.minimum_selected_markets,
            ),
            _check(
                "zero lost control decisions",
                attribution["lost_control_markets"] if attribution else None,
                "==",
                0,
            ),
            _check(
                "minimum residual accuracy",
                residual["accuracy"],
                ">=",
                gates.minimum_residual_accuracy,
            ),
            _check(
                "maximum residual ECE",
                residual["expected_calibration_error"],
                "<=",
                gates.maximum_expected_calibration_error,
            ),
        )
    )
    if candidate_name in {
        EARLY_RESIDUAL_CANDIDATE,
        COMBINED_RESIDUAL_CANDIDATE,
    }:
        checks.extend(
            (
                _check(
                    "median entry improves by at least five seconds",
                    _difference(
                        control_metrics["median_seconds_elapsed"],
                        metrics["median_seconds_elapsed"],
                    ),
                    ">=",
                    gates.minimum_median_entry_improvement_seconds,
                ),
                _check(
                    "maximum median entry second",
                    metrics["median_seconds_elapsed"],
                    "<=",
                    gates.maximum_median_entry_second,
                ),
                _check(
                    "decisions by second 120 improve",
                    checkpoints["120"] - control_checkpoints["120"],
                    ">=",
                    gates.minimum_decisions_by_120_uplift,
                ),
                _check(
                    "minimum advanced markets",
                    attribution["advanced_markets"] if attribution else None,
                    ">=",
                    gates.minimum_early_residual_markets,
                ),
                _check(
                    "minimum median advancement seconds",
                    (
                        attribution["median_advancement_seconds"]
                        if attribution
                        else None
                    ),
                    ">=",
                    gates.minimum_median_advancement_seconds,
                ),
            )
        )
    if candidate_name in {
        RESCUE_RESIDUAL_CANDIDATE,
        COMBINED_RESIDUAL_CANDIDATE,
    }:
        checks.extend(
            (
                _check(
                    "minimum NoTrade reduction",
                    control_metrics["no_trade_rate"] - metrics["no_trade_rate"],
                    ">=",
                    gates.minimum_no_trade_reduction,
                ),
                _check(
                    "minimum rescued control-NoTrade markets",
                    attribution["rescued_markets"] if attribution else None,
                    ">=",
                    gates.minimum_rescued_markets,
                ),
            )
        )
    execution = metrics["execution"]
    residual_execution = residual["execution"]
    realized_per_share = (
        execution["realized_net_expectancy_per_trade"] / quantity
        if execution["realized_net_expectancy_per_trade"] is not None
        else None
    )
    residual_realized_per_share = (
        residual_execution["realized_net_expectancy_per_trade"] / quantity
        if residual_execution["realized_net_expectancy_per_trade"] is not None
        else None
    )
    checks.extend(
        (
            _check(
                "minimum executable markets",
                execution["economic_markets"],
                ">=",
                gates.minimum_executable_markets,
            ),
            _check(
                "minimum residual executable markets",
                residual_execution["economic_markets"],
                ">=",
                gates.minimum_executable_markets,
            ),
            _check(
                "positive direct edge per share",
                execution["mean_direct_edge_per_share"],
                ">",
                gates.minimum_mean_direct_edge_per_share,
            ),
            _check(
                "positive realized net expectancy per share",
                realized_per_share,
                ">",
                gates.minimum_realized_net_per_share,
            ),
            _check(
                "positive residual direct edge per share",
                residual_execution["mean_direct_edge_per_share"],
                ">",
                gates.minimum_mean_direct_edge_per_share,
            ),
            _check(
                "positive residual realized net expectancy per share",
                residual_realized_per_share,
                ">",
                gates.minimum_realized_net_per_share,
            ),
            _check(
                "nonnegative residual hourly bootstrap lower bound",
                residual["hourly_net_bootstrap"]["lower_95"],
                ">=",
                0.0,
            ),
        )
    )
    per_fold_checks: list[dict[str, Any]] = []
    if gates.require_every_fold:
        checks.extend(
            (
                _check(
                    "required evaluation fold count",
                    len(folds),
                    "==",
                    gates.required_evaluation_folds,
                ),
                _check(
                    "matched control evaluation fold count",
                    len(control_folds),
                    "==",
                    gates.required_evaluation_folds,
                ),
            )
        )
        for fold, control_fold in zip(folds, control_folds, strict=True):
            per_fold_checks.append(
                {
                    "fold_index": fold.get("fold_index"),
                    "checks": _fold_advancement_checks(
                        candidate_name=candidate_name,
                        fold=fold,
                        control_fold=control_fold,
                        gates=gates,
                        quantity=quantity,
                    ),
                }
            )
        checks.append(
            _check(
                "all advancement gates pass every evaluation fold",
                sum(
                    all(check["passed"] for check in record["checks"])
                    for record in per_fold_checks
                ),
                "==",
                gates.required_evaluation_folds,
            )
        )
    benchmark_passed = all(check["passed"] for check in checks)
    return {
        "is_control": False,
        "checks": checks,
        "per_fold_checks": per_fold_checks,
        "benchmark_passed": benchmark_passed,
        "development_qualified": benchmark_passed,
        "deployment_qualified": False,
    }


def _fold_advancement_checks(
    *,
    candidate_name: str,
    fold: dict[str, Any],
    control_fold: dict[str, Any],
    gates: ResidualAdmissionGates,
    quantity: float,
) -> list[dict[str, Any]]:
    metrics = fold["metrics"]
    control_metrics = control_fold["metrics"]
    residual = fold["residual_cohort"]
    attribution = fold["paired_attribution"]
    checkpoints = fold["checkpoints"]
    control_checkpoints = control_fold["checkpoints"]
    execution = metrics["execution"]
    residual_execution = residual["execution"]
    realized_per_share = (
        execution["realized_net_expectancy_per_trade"] / quantity
        if execution["realized_net_expectancy_per_trade"] is not None
        else None
    )
    residual_realized_per_share = (
        residual_execution["realized_net_expectancy_per_trade"] / quantity
        if residual_execution["realized_net_expectancy_per_trade"] is not None
        else None
    )
    checks = _quality_checks(metrics, control_metrics, gates)
    checks.extend(
        (
            _check(
                "threshold selection qualified",
                int(bool(fold["selection_qualified"])),
                "==",
                1,
            ),
            _check(
                "minimum selected markets",
                metrics["markets"],
                ">=",
                gates.minimum_selected_markets,
            ),
            _check(
                "zero lost control decisions",
                attribution["lost_control_markets"],
                "==",
                0,
            ),
            _check(
                "minimum residual accuracy",
                residual["accuracy"],
                ">=",
                gates.minimum_residual_accuracy,
            ),
            _check(
                "maximum residual ECE",
                residual["expected_calibration_error"],
                "<=",
                gates.maximum_expected_calibration_error,
            ),
            _check(
                "minimum executable markets",
                execution["economic_markets"],
                ">=",
                gates.minimum_executable_markets,
            ),
            _check(
                "minimum residual executable markets",
                residual_execution["economic_markets"],
                ">=",
                gates.minimum_executable_markets,
            ),
            _check(
                "positive direct edge per share",
                execution["mean_direct_edge_per_share"],
                ">",
                gates.minimum_mean_direct_edge_per_share,
            ),
            _check(
                "positive realized net expectancy per share",
                realized_per_share,
                ">",
                gates.minimum_realized_net_per_share,
            ),
            _check(
                "positive residual direct edge per share",
                residual_execution["mean_direct_edge_per_share"],
                ">",
                gates.minimum_mean_direct_edge_per_share,
            ),
            _check(
                "positive residual realized net expectancy per share",
                residual_realized_per_share,
                ">",
                gates.minimum_realized_net_per_share,
            ),
            _check(
                "nonnegative residual hourly bootstrap lower bound",
                residual["hourly_net_bootstrap"]["lower_95"],
                ">=",
                0.0,
            ),
        )
    )
    if candidate_name in {
        EARLY_RESIDUAL_CANDIDATE,
        COMBINED_RESIDUAL_CANDIDATE,
    }:
        checks.extend(
            (
                _check(
                    "median entry improves by at least five seconds",
                    _difference(
                        control_metrics["median_seconds_elapsed"],
                        metrics["median_seconds_elapsed"],
                    ),
                    ">=",
                    gates.minimum_median_entry_improvement_seconds,
                ),
                _check(
                    "maximum median entry second",
                    metrics["median_seconds_elapsed"],
                    "<=",
                    gates.maximum_median_entry_second,
                ),
                _check(
                    "decisions by second 120 improve",
                    checkpoints["120"] - control_checkpoints["120"],
                    ">=",
                    gates.minimum_decisions_by_120_uplift,
                ),
                _check(
                    "minimum median advancement seconds",
                    attribution["median_advancement_seconds"],
                    ">=",
                    gates.minimum_median_advancement_seconds,
                ),
            )
        )
    if candidate_name in {
        RESCUE_RESIDUAL_CANDIDATE,
        COMBINED_RESIDUAL_CANDIDATE,
    }:
        checks.append(
            _check(
                "minimum NoTrade reduction",
                control_metrics["no_trade_rate"] - metrics["no_trade_rate"],
                ">=",
                gates.minimum_no_trade_reduction,
            )
        )
    return checks


def _policy_selection_checks(
    metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    residual_metrics: dict[str, Any],
    gates: ResidualAdmissionGates,
) -> list[dict[str, Any]]:
    return [
        _check("minimum accuracy", metrics["accuracy"], ">=", gates.minimum_accuracy),
        _check(
            "minimum balanced accuracy",
            metrics["balanced_accuracy"],
            ">=",
            gates.minimum_balanced_accuracy,
        ),
        _check(
            "minimum UP recall",
            metrics["up_recall"],
            ">=",
            gates.minimum_direction_recall,
        ),
        _check(
            "minimum DOWN recall",
            metrics["down_recall"],
            ">=",
            gates.minimum_direction_recall,
        ),
        _check(
            "maximum ECE",
            metrics["expected_calibration_error"],
            "<=",
            gates.maximum_expected_calibration_error,
        ),
        _check(
            "accuracy non-regression",
            metrics["accuracy"] - control_metrics["accuracy"],
            ">=",
            -gates.maximum_accuracy_regression,
        ),
        _check(
            "balanced accuracy non-regression",
            metrics["balanced_accuracy"] - control_metrics["balanced_accuracy"],
            ">=",
            -gates.maximum_balanced_accuracy_regression,
        ),
        _check(
            "UP recall non-regression",
            metrics["up_recall"] - control_metrics["up_recall"],
            ">=",
            -gates.maximum_direction_recall_regression,
        ),
        _check(
            "DOWN recall non-regression",
            metrics["down_recall"] - control_metrics["down_recall"],
            ">=",
            -gates.maximum_direction_recall_regression,
        ),
        _check(
            "residual accuracy floor",
            residual_metrics["accuracy"],
            ">=",
            gates.minimum_residual_accuracy,
        ),
        _check("minimum policy residual rows", residual_metrics["markets"], ">=", 100),
    ]


def _quality_checks(
    metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    gates: ResidualAdmissionGates,
) -> list[dict[str, Any]]:
    return [
        _check("minimum accuracy", metrics["accuracy"], ">=", gates.minimum_accuracy),
        _check(
            "minimum balanced accuracy",
            metrics["balanced_accuracy"],
            ">=",
            gates.minimum_balanced_accuracy,
        ),
        _check(
            "minimum UP recall",
            metrics["up_recall"],
            ">=",
            gates.minimum_direction_recall,
        ),
        _check(
            "minimum DOWN recall",
            metrics["down_recall"],
            ">=",
            gates.minimum_direction_recall,
        ),
        _check(
            "minimum Wilson lower bound",
            metrics["wilson_lower_95"],
            ">=",
            gates.minimum_wilson_lower_95,
        ),
        _check(
            "maximum expected calibration error",
            metrics["expected_calibration_error"],
            "<=",
            gates.maximum_expected_calibration_error,
        ),
        _check(
            "accuracy non-regression",
            metrics["accuracy"] - control_metrics["accuracy"],
            ">=",
            -gates.maximum_accuracy_regression,
        ),
        _check(
            "balanced accuracy non-regression",
            metrics["balanced_accuracy"] - control_metrics["balanced_accuracy"],
            ">=",
            -gates.maximum_balanced_accuracy_regression,
        ),
        _check(
            "UP recall non-regression",
            metrics["up_recall"] - control_metrics["up_recall"],
            ">=",
            -gates.maximum_direction_recall_regression,
        ),
        _check(
            "DOWN recall non-regression",
            metrics["down_recall"] - control_metrics["down_recall"],
            ">=",
            -gates.maximum_direction_recall_regression,
        ),
    ]


def _fold_quality_passed(
    metrics: dict[str, Any],
    control_metrics: dict[str, Any] | None,
    gates: ResidualAdmissionGates,
) -> bool:
    if control_metrics is None:
        return all(
            (
                metrics["accuracy"] >= gates.minimum_accuracy,
                metrics["balanced_accuracy"] >= gates.minimum_balanced_accuracy,
                metrics["up_recall"] >= gates.minimum_direction_recall,
                metrics["down_recall"] >= gates.minimum_direction_recall,
                metrics["wilson_lower_95"] >= gates.minimum_wilson_lower_95,
                metrics["expected_calibration_error"]
                <= gates.maximum_expected_calibration_error,
            )
        )
    return all(
        check["passed"]
        for check in _quality_checks(metrics, control_metrics, gates)
    )


def _benchmark_criteria(
    gates: ResidualAdmissionGates,
) -> AdvancementCriteria:
    return AdvancementCriteria(
        minimum_accuracy=gates.minimum_accuracy,
        minimum_balanced_accuracy=gates.minimum_balanced_accuracy,
        minimum_direction_recall=gates.minimum_direction_recall,
        minimum_wilson_lower_95=gates.minimum_wilson_lower_95,
        maximum_expected_calibration_error=gates.maximum_expected_calibration_error,
        minimum_coverage=0.0,
        minimum_coverage_uplift=0.0,
        maximum_accuracy_regression=gates.maximum_accuracy_regression,
        maximum_balanced_accuracy_regression=(
            gates.maximum_balanced_accuracy_regression
        ),
        maximum_direction_recall_regression=(
            gates.maximum_direction_recall_regression
        ),
        maximum_median_entry_seconds_regression=0.0,
        minimum_mean_direct_edge_per_share=(
            gates.minimum_mean_direct_edge_per_share
        ),
        minimum_realized_net_per_share=gates.minimum_realized_net_per_share,
        minimum_common_time_markets=1,
    )


def _feature_matrix(
    frame: pl.DataFrame,
    features: tuple[str, ...],
) -> np.ndarray:
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError("residual features are missing: " + ", ".join(missing))
    forbidden = {
        "label_up",
        "proposal_correct",
        "control_correct",
        "official_outcome",
        "final_price",
    }
    if forbidden.intersection(features):
        raise RuntimeError("residual feature allowlist contains outcome data")
    return frame.select(features).to_numpy().astype(np.float64)


def _finite_medians(matrix: np.ndarray) -> np.ndarray:
    medians = np.zeros(matrix.shape[1], dtype=np.float64)
    for index in range(matrix.shape[1]):
        values = matrix[np.isfinite(matrix[:, index]), index]
        medians[index] = float(np.median(values)) if values.size else 0.0
    return medians


def _selection_payload(
    selection: ResidualThresholdSelection,
) -> dict[str, Any]:
    return asdict(selection)


def _frame_range(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "start": frame["window_start"].min().isoformat(),
        "end": frame["window_start"].max().isoformat(),
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
    }


def _execution_config(
    config: ResidualAdmissionBenchmarkConfig,
) -> ExecutionEvidenceConfig:
    manifest = json.loads(
        (config.execution_evidence / "manifest.json").read_text()
    )
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


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


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
    elif operator == "<=":
        passed = observed <= required
    elif operator == "==":
        passed = observed == required
    else:
        raise ValueError(f"unsupported residual gate operator: {operator}")
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "required": required,
        "passed": bool(passed),
    }


def _write_parquet_atomic(
    frame: pl.DataFrame,
    destination: Path,
) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    frame.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )
    os.replace(temporary, destination)
