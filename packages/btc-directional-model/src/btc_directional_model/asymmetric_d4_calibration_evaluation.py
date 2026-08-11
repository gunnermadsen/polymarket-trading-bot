"""Evaluation contracts for frozen D4 side-aware calibration challengers.

Probability selection is deliberately isolated from economics.  The projected-PnL
report accepts only a completed probability-selection result and its closed seal,
then reports every predeclared arm.  Economics for nonselected arms are descriptive
diagnostics and can never promote a model.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

from .asymmetric_incumbent_evaluation import (
    DEFAULT_TIME_BANDS,
    PROBABILITY_KEY_COLUMNS,
    PROBABILITY_SELECTION_FORBIDDEN_COLUMNS,
    _shared_reference_probability_bootstrap,
    incumbent_probability_metrics,
    probability_only_first_crossings,
    target_opportunity_probability_cohort,
)
from .asymmetric_value_evaluation import EXECUTION_STRESS_PER_SHARE, ledger_metrics

D4_SIDE_CALIBRATION_EVALUATION_SCHEMA_VERSION = "btc-asymmetric-d4-side-calibration-evaluation-v1"
D4_PROJECTED_PNL_SCHEMA_VERSION = "btc-asymmetric-d4-projected-pnl-v1"
INCUMBENT_ID = "I0"
D4_BASE_ID = "D4-base"
POOLED_NO_CALIBRATION_ID = "N1"
TIME_LOCAL_NO_CALIBRATION_ID = "N2"
D4_MODEL_IDS = (
    INCUMBENT_ID,
    D4_BASE_ID,
    POOLED_NO_CALIBRATION_ID,
    TIME_LOCAL_NO_CALIBRATION_ID,
)
CONSUMED_EVIDENCE_SCOPE = "consumed_architecture_feedback"
FRESH_EVIDENCE_SCOPE = "fresh_untuned_validation"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class D4SideCalibrationSelectionThresholds:
    """Predeclared probability-only thresholds for N1/N2 selection."""

    noninferiority_margin: float = 0.005
    maximum_selected_bias: float = 0.03
    maximum_selected_yes_bias: float = 0.03
    maximum_selected_no_bias: float = 0.05
    minimum_no_bias_reduction: float = 0.02
    maximum_cell_overconfidence: float = 0.05
    maximum_cell_overconfidence_regression: float = 0.0
    minimum_joint_noninferior_days: int = 8
    required_validation_days: int = 10
    minimum_selected_yes: int = 20
    minimum_selected_no: int = 20

    def __post_init__(self) -> None:
        probabilities = (
            self.noninferiority_margin,
            self.maximum_selected_bias,
            self.maximum_selected_yes_bias,
            self.maximum_selected_no_bias,
            self.minimum_no_bias_reduction,
            self.maximum_cell_overconfidence,
            self.maximum_cell_overconfidence_regression,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("D4 probability thresholds must be finite and inside [0, 1]")
        if self.required_validation_days <= 0:
            raise ValueError("D4 validation day requirement must be positive")
        if not 1 <= self.minimum_joint_noninferior_days <= self.required_validation_days:
            raise ValueError("D4 joint-day requirement is outside the validation window")
        if self.minimum_selected_yes < 0 or self.minimum_selected_no < 0:
            raise ValueError("D4 selected-side support thresholds must be nonnegative")


@dataclass(frozen=True)
class D4ProjectedPnlThresholds:
    """Post-selection economic gates for the probability-selected arm only."""

    minimum_profit_factor: float = 1.05
    minimum_no_stressed_pnl_improvement: float = 0.0
    maximum_yes_stressed_pnl_regression_fraction: float = 0.10
    maximum_drawdown_regression_fraction: float = 0.10
    maximum_loss_recovery_regression_fraction: float = 0.10

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("D4 projected-PnL thresholds must be finite and nonnegative")


def select_d4_side_calibration_challenger(
    incumbent: pl.DataFrame,
    d4_base: pl.DataFrame,
    challengers: Mapping[str, pl.DataFrame],
    support_evidence: Mapping[str, Mapping[str, Any]],
    thresholds: D4SideCalibrationSelectionThresholds,
    *,
    resamples: int,
    seed: int,
    evidence_scope: str,
    qualification_eligible: bool,
) -> dict[str, Any]:
    """Select N1 or N2 using matched probability evidence and no economics."""

    if set(challengers) != {POOLED_NO_CALIBRATION_ID, TIME_LOCAL_NO_CALIBRATION_ID}:
        raise ValueError("D4 calibration selection requires exactly N1 and N2")
    if not isinstance(thresholds, D4SideCalibrationSelectionThresholds):
        raise TypeError("D4 selection thresholds must use the frozen threshold type")
    _validate_evidence_scope(evidence_scope, qualification_eligible)
    if resamples < 2 or seed < 0:
        raise ValueError("D4 probability bootstrap settings are invalid")
    _reject_economic_values(support_evidence)

    frames = {
        INCUMBENT_ID: _normalize_probability_frame(incumbent, INCUMBENT_ID),
        D4_BASE_ID: _normalize_probability_frame(d4_base, D4_BASE_ID),
        **{
            candidate_id: _normalize_probability_frame(frame, candidate_id)
            for candidate_id, frame in challengers.items()
        },
    }
    target = {
        candidate_id: target_opportunity_probability_cohort(frame)
        for candidate_id, frame in frames.items()
    }
    for candidate_id in D4_MODEL_IDS[1:]:
        _require_matched_keys_and_labels(target[INCUMBENT_ID], target[candidate_id], candidate_id)

    candidate_frames = {
        candidate_id: target[candidate_id]
        for candidate_id in (POOLED_NO_CALIBRATION_ID, TIME_LOCAL_NO_CALIBRATION_ID)
    }
    versus_incumbent = _shared_reference_probability_bootstrap(
        target[INCUMBENT_ID], candidate_frames, resamples=resamples, seed=seed
    )
    versus_d4 = _shared_reference_probability_bootstrap(
        target[D4_BASE_ID], candidate_frames, resamples=resamples, seed=seed
    )
    d4_to_incumbent = _shared_reference_probability_bootstrap(
        target[INCUMBENT_ID], {D4_BASE_ID: target[D4_BASE_ID]}, resamples=resamples, seed=seed
    )
    side_comparisons = {
        side: _shared_reference_probability_bootstrap(
            _target_side_frame(target[D4_BASE_ID], side),
            {
                candidate_id: _target_side_frame(frame, side)
                for candidate_id, frame in candidate_frames.items()
            },
            resamples=resamples,
            seed=seed,
        )
        for side in ("YES", "NO")
    }

    metrics = {
        candidate_id: incumbent_probability_metrics(frame) for candidate_id, frame in target.items()
    }
    selected = {
        candidate_id: probability_only_first_crossings(frame)
        for candidate_id, frame in target.items()
    }
    selected_metrics = {
        candidate_id: (incumbent_probability_metrics(frame) if not frame.is_empty() else None)
        for candidate_id, frame in selected.items()
    }
    d4_selected = selected_metrics[D4_BASE_ID]
    if d4_selected is None:
        raise RuntimeError("D4-base has no selected target opportunities")
    d4_no_bias = _absolute_metric(d4_selected, "sides", "NO", "bias")
    if d4_no_bias is None:
        raise RuntimeError("D4-base has no selected NO calibration evidence")

    records: list[dict[str, Any]] = []
    for candidate_id in (POOLED_NO_CALIBRATION_ID, TIME_LOCAL_NO_CALIBRATION_ID):
        candidate_selected = selected_metrics[candidate_id]
        comparison_i0 = versus_incumbent["comparisons"][candidate_id]
        comparison_d4 = versus_d4["comparisons"][candidate_id]
        yes_comparison = side_comparisons["YES"]["comparisons"][candidate_id]
        no_comparison = side_comparisons["NO"]["comparisons"][candidate_id]
        selected_bias = _absolute_metric(candidate_selected, "overall", "bias")
        selected_yes_bias = _absolute_metric(candidate_selected, "sides", "YES", "bias")
        selected_no_bias = _absolute_metric(candidate_selected, "sides", "NO", "bias")
        no_bias_reduction = d4_no_bias - selected_no_bias if selected_no_bias is not None else None
        daily = _joint_daily_noninferiority(
            comparison_i0["daily"],
            comparison_d4["daily"],
            thresholds.noninferiority_margin,
        )
        support = _normalize_support(support_evidence.get(candidate_id), candidate_id)
        gates = [
            _gate(
                "validation_day_count",
                versus_d4["utc_days"],
                thresholds.required_validation_days,
                "==",
            ),
            _gate("calibration_support", support["passed"], True, "=="),
            _gate(
                "brier_noninferior_to_i0",
                comparison_i0["brier_delta"]["simultaneous_upper_95"],
                thresholds.noninferiority_margin,
                "<=",
            ),
            _gate(
                "log_loss_noninferior_to_i0",
                comparison_i0["log_loss_delta"]["simultaneous_upper_95"],
                thresholds.noninferiority_margin,
                "<=",
            ),
            _gate(
                "brier_noninferior_to_d4_base",
                comparison_d4["brier_delta"]["simultaneous_upper_95"],
                thresholds.noninferiority_margin,
                "<=",
            ),
            _gate(
                "log_loss_noninferior_to_d4_base",
                comparison_d4["log_loss_delta"]["simultaneous_upper_95"],
                thresholds.noninferiority_margin,
                "<=",
            ),
            _gate(
                "proper_score_improvement_over_d4_base",
                _one_point_improves(comparison_d4),
                True,
                "==",
            ),
            _gate(
                "yes_brier_preserved",
                yes_comparison["brier_delta"]["simultaneous_upper_95"],
                thresholds.noninferiority_margin,
                "<=",
            ),
            _gate(
                "yes_log_loss_preserved",
                yes_comparison["log_loss_delta"]["simultaneous_upper_95"],
                thresholds.noninferiority_margin,
                "<=",
            ),
            _gate(
                "no_brier_noninferior",
                no_comparison["brier_delta"]["simultaneous_upper_95"],
                thresholds.noninferiority_margin,
                "<=",
            ),
            _gate(
                "no_log_loss_noninferior",
                no_comparison["log_loss_delta"]["simultaneous_upper_95"],
                thresholds.noninferiority_margin,
                "<=",
            ),
            _gate("selected_absolute_bias", selected_bias, thresholds.maximum_selected_bias, "<="),
            _gate(
                "selected_yes_absolute_bias",
                selected_yes_bias,
                thresholds.maximum_selected_yes_bias,
                "<=",
            ),
            _gate(
                "selected_no_absolute_bias",
                selected_no_bias,
                thresholds.maximum_selected_no_bias,
                "<=",
            ),
            _gate(
                "selected_no_bias_reduction",
                no_bias_reduction,
                thresholds.minimum_no_bias_reduction,
                ">=",
            ),
            _gate(
                "minimum_selected_yes",
                _side_rows(candidate_selected, "YES"),
                thresholds.minimum_selected_yes,
                ">=",
            ),
            _gate(
                "minimum_selected_no",
                _side_rows(candidate_selected, "NO"),
                thresholds.minimum_selected_no,
                ">=",
            ),
            _gate(
                "joint_noninferior_utc_days",
                daily["common"],
                thresholds.minimum_joint_noninferior_days,
                ">=",
            ),
            *_cell_overconfidence_gates(candidate_selected, d4_selected, thresholds),
        ]
        records.append(
            {
                "candidate_id": candidate_id,
                "selectable": True,
                "passed": all(gate["passed"] for gate in gates),
                "target_metrics": metrics[candidate_id],
                "selected_metrics": candidate_selected,
                "selected_opportunities": selected[candidate_id].height,
                "comparison_to_i0": comparison_i0,
                "comparison_to_d4_base": comparison_d4,
                "side_comparison_to_d4_base": {
                    "YES": yes_comparison,
                    "NO": no_comparison,
                },
                "selected_no_absolute_bias_reduction": no_bias_reduction,
                "noninferior_utc_days_to_i0": daily["incumbent"],
                "noninferior_utc_days_to_d4_base": daily["d4_base"],
                "joint_noninferior_utc_days": daily["common"],
                "support": support,
                "gates": gates,
            }
        )

    ranked = sorted(
        (record for record in records if record["passed"]),
        key=lambda record: (
            _absolute_metric(record["selected_metrics"], "sides", "NO", "bias"),
            record["target_metrics"]["overall"]["log_loss"],
            record["target_metrics"]["overall"]["brier"],
            0 if record["candidate_id"] == POOLED_NO_CALIBRATION_ID else 1,
        ),
    )
    selected_candidate_id = ranked[0]["candidate_id"] if ranked else None
    status = "selected" if selected_candidate_id else "blocked_no_quality_configuration"
    if not qualification_eligible:
        status = (
            "diagnostic_selected_consumed_evidence"
            if selected_candidate_id
            else "diagnostic_no_quality_configuration_consumed_evidence"
        )
    return {
        "schema_version": D4_SIDE_CALIBRATION_EVALUATION_SCHEMA_VERSION,
        "status": status,
        "selected_candidate_id": selected_candidate_id,
        "evidence_scope": evidence_scope,
        "qualification_eligible": qualification_eligible,
        "promotion_authorized": False,
        "target_cohort": {
            "definition": "seconds 1-55 and either YES or NO raw VWAP5 in [0.20,0.30)",
            "rows": target[INCUMBENT_ID].height,
            "markets": target[INCUMBENT_ID]["market_id"].n_unique(),
            "utc_days": target[INCUMBENT_ID]["window_start"].dt.date().n_unique(),
        },
        "thresholds": asdict(thresholds),
        "i0": {
            "target_metrics": metrics[INCUMBENT_ID],
            "selected_metrics": selected_metrics[INCUMBENT_ID],
            "selected_opportunities": selected[INCUMBENT_ID].height,
        },
        "d4_base": {
            "candidate_id": D4_BASE_ID,
            "selectable": False,
            "target_metrics": metrics[D4_BASE_ID],
            "selected_metrics": d4_selected,
            "selected_opportunities": selected[D4_BASE_ID].height,
            "comparison_to_i0": d4_to_incumbent["comparisons"][D4_BASE_ID],
        },
        "simultaneous_comparison_to_i0": versus_incumbent,
        "simultaneous_comparison_to_d4_base": versus_d4,
        "side_comparisons_to_d4_base": side_comparisons,
        "candidate_records": records,
        "rank_trace": [
            {
                "rank": index + 1,
                "candidate_id": record["candidate_id"],
                "selected_no_absolute_bias": _absolute_metric(
                    record["selected_metrics"], "sides", "NO", "bias"
                ),
                "log_loss": record["target_metrics"]["overall"]["log_loss"],
                "brier": record["target_metrics"]["overall"]["brier"],
            }
            for index, record in enumerate(ranked)
        ],
        "failure_trace": [
            {
                "candidate_id": record["candidate_id"],
                "failed_gates": [gate["name"] for gate in record["gates"] if not gate["passed"]],
            }
            for record in records
        ],
        "economics_used": False,
        "projected_pnl_available_only_after_selection_seal": True,
    }


def build_d4_projected_pnl_report(
    ledgers: Mapping[str, pl.DataFrame],
    eligible_markets: pl.DataFrame,
    probability_selection: Mapping[str, Any],
    selection_seal: Mapping[str, Any],
    *,
    selection_seal_sha256: str,
    resamples: int,
    seed: int,
    thresholds: D4ProjectedPnlThresholds | None = None,
) -> dict[str, Any]:
    """Report post-seal projected PnL for all arms without changing selection."""

    thresholds = thresholds or D4ProjectedPnlThresholds()
    _validate_closed_selection_seal(
        probability_selection,
        selection_seal,
        selection_seal_sha256,
    )
    if set(ledgers) != set(D4_MODEL_IDS):
        raise ValueError("projected PnL requires I0, D4-base, N1, and N2 ledgers")
    if resamples < 2 or seed < 0:
        raise ValueError("projected-PnL bootstrap settings are invalid")
    universe = _eligible_universe(eligible_markets)
    normalized_ledgers = {
        candidate_id: _validated_ledger(frame, candidate_id, universe)
        for candidate_id, frame in ledgers.items()
    }
    model_reports = {
        candidate_id: _model_pnl_report(
            candidate_id,
            normalized_ledgers[candidate_id],
            universe.height,
            probability_selection.get("selected_candidate_id"),
            bool(probability_selection.get("qualification_eligible")),
        )
        for candidate_id in D4_MODEL_IDS
    }
    comparisons = {
        candidate_id: {
            "to_i0": (
                None
                if candidate_id == INCUMBENT_ID
                else _paired_stressed_pnl_bootstrap(
                    normalized_ledgers[INCUMBENT_ID],
                    normalized_ledgers[candidate_id],
                    universe,
                    resamples=resamples,
                    seed=seed,
                )
            ),
            "to_d4_base": (
                None
                if candidate_id == D4_BASE_ID
                else _paired_stressed_pnl_bootstrap(
                    normalized_ledgers[D4_BASE_ID],
                    normalized_ledgers[candidate_id],
                    universe,
                    resamples=resamples,
                    seed=seed,
                )
            ),
        }
        for candidate_id in D4_MODEL_IDS
    }
    selected_candidate_id = probability_selection.get("selected_candidate_id")
    qualification = None
    if selected_candidate_id is not None:
        qualification = _selected_economic_qualification(
            selected_candidate_id,
            model_reports,
            comparisons[selected_candidate_id]["to_d4_base"],
            thresholds,
            bool(probability_selection.get("qualification_eligible")),
        )
    return {
        "schema_version": D4_PROJECTED_PNL_SCHEMA_VERSION,
        "status": (
            "post_selection_diagnostic_no_probability_winner"
            if selected_candidate_id is None
            else (qualification["status"] if qualification is not None else "diagnostic_only")
        ),
        "evidence_scope": probability_selection["evidence_scope"],
        "qualification_eligible": bool(probability_selection["qualification_eligible"]),
        "selection_seal_sha256": selection_seal_sha256,
        "probability_selected_candidate_id": selected_candidate_id,
        "probability_selection_used_pnl": False,
        "economics_opened_after_probability_selection_seal": True,
        "all_predeclared_models_reported": list(D4_MODEL_IDS),
        "projection_scope": "matched historical OOF replay under the frozen five-share policy",
        "projection_is_forward_guarantee": False,
        "eligible_markets": universe.height,
        "models": model_reports,
        "paired_stressed_pnl": comparisons,
        "selected_economic_qualification": qualification,
        "nonselected_pnl_may_qualify_model": False,
        "promotion_authorized": False,
    }


def _normalize_probability_frame(frame: pl.DataFrame, candidate_id: str) -> pl.DataFrame:
    renamed = frame
    if "incumbent_probability_yes" in renamed.columns and "probability_yes" not in renamed.columns:
        renamed = renamed.rename({"incumbent_probability_yes": "probability_yes"})
    if "candidate_name" in renamed.columns and "candidate_id" not in renamed.columns:
        renamed = renamed.rename({"candidate_name": "candidate_id"})
    if "candidate_id" in renamed.columns:
        identities = renamed["candidate_id"].drop_nulls().unique().to_list()
        if identities != [candidate_id]:
            raise ValueError(f"{candidate_id} probability-frame identity does not match")
    return renamed


def _require_matched_keys_and_labels(
    reference: pl.DataFrame,
    candidate: pl.DataFrame,
    candidate_id: str,
) -> None:
    reference = reference.sort(*PROBABILITY_KEY_COLUMNS)
    candidate = candidate.sort(*PROBABILITY_KEY_COLUMNS)
    if not candidate.select(*PROBABILITY_KEY_COLUMNS).equals(
        reference.select(*PROBABILITY_KEY_COLUMNS), null_equal=True
    ):
        raise ValueError(f"{candidate_id} keys do not match I0")
    if not candidate["label_up"].equals(reference["label_up"]):
        raise ValueError(f"{candidate_id} labels do not match I0")


def _target_side_frame(frame: pl.DataFrame, side: str) -> pl.DataFrame:
    column = "yes_target_eligible" if side == "YES" else "no_target_eligible"
    selected = frame.filter(pl.col(column))
    if selected.is_empty():
        raise RuntimeError(f"D4 target cohort has no {side} opportunities")
    return selected


def _one_point_improves(comparison: Mapping[str, Any]) -> bool:
    return bool(
        comparison["brier_delta"]["point"] < 0.0 or comparison["log_loss_delta"]["point"] < 0.0
    )


def _joint_daily_noninferiority(
    versus_i0: Mapping[str, Mapping[str, float]],
    versus_d4: Mapping[str, Mapping[str, float]],
    margin: float,
) -> dict[str, int]:
    if set(versus_i0) != set(versus_d4):
        raise RuntimeError("D4 daily comparisons do not share UTC days")
    i0 = d4 = common = 0
    for day in sorted(versus_i0):
        i0_passed = _daily_noninferior(versus_i0[day], margin)
        d4_passed = _daily_noninferior(versus_d4[day], margin)
        i0 += int(i0_passed)
        d4 += int(d4_passed)
        common += int(i0_passed and d4_passed)
    return {"incumbent": i0, "d4_base": d4, "common": common}


def _daily_noninferior(values: Mapping[str, float], margin: float) -> bool:
    return bool(values["brier_delta"] <= margin and values["log_loss_delta"] <= margin)


def _absolute_metric(metrics: Mapping[str, Any] | None, *keys: str) -> float | None:
    value: Any = metrics
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return abs(float(value)) if value is not None else None


def _side_rows(metrics: Mapping[str, Any] | None, side: str) -> int | None:
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get("sides", {}).get(side, {}).get("rows")
    return int(value) if value is not None else None


def _cell_overconfidence_gates(
    candidate: Mapping[str, Any] | None,
    d4_base: Mapping[str, Any],
    thresholds: D4SideCalibrationSelectionThresholds,
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for side in ("YES", "NO"):
        for lower, upper in DEFAULT_TIME_BANDS:
            name = f"{side}_{lower}_{upper}"
            candidate_cell = (candidate or {}).get("time_cells", {}).get(name, {})
            base_cell = d4_base["time_cells"].get(name, {})
            candidate_bias = candidate_cell.get("bias")
            base_bias = base_cell.get("bias")
            if candidate_cell.get("rows", 0) == 0:
                gates.append(
                    {
                        "name": f"selected_cell_overconfidence_{name.lower()}",
                        "observed": None,
                        "threshold": thresholds.maximum_cell_overconfidence,
                        "operator": "not_applicable_no_selected_entries",
                        "passed": True,
                    }
                )
                continue
            candidate_overconfidence = max(0.0, float(candidate_bias))
            base_overconfidence = max(0.0, float(base_bias or 0.0))
            limit = min(
                thresholds.maximum_cell_overconfidence,
                base_overconfidence + thresholds.maximum_cell_overconfidence_regression,
            )
            gates.append(
                _gate(
                    f"selected_cell_overconfidence_{name.lower()}",
                    candidate_overconfidence,
                    limit,
                    "<=",
                )
            )
    return gates


def _normalize_support(
    support: Mapping[str, Any] | None,
    candidate_id: str,
) -> dict[str, Any]:
    support = support if isinstance(support, Mapping) else {}
    explicit = support.get("support_passed", support.get("passed"))
    failures = support.get("support_failures", ())
    if not isinstance(failures, (list, tuple)):
        failures = ("invalid_support_failures",)
    no_markets = _integer(support.get("no_markets", support.get("markets")))
    no_days = _integer(support.get("no_utc_days", support.get("utc_days")))
    wins = _integer(support.get("no_winning_markets", support.get("winning_markets")))
    losses = _integer(support.get("no_losing_markets", support.get("losing_markets")))
    cells = support.get("time_cells", ())
    if isinstance(cells, Mapping):
        cell_values = list(cells.values())
    elif isinstance(cells, (list, tuple)):
        cell_values = list(cells)
    else:
        cell_values = []
    required_cells_passed = True
    if candidate_id == TIME_LOCAL_NO_CALIBRATION_ID:
        required_cells_passed = len(cell_values) == 4 and all(
            isinstance(cell, Mapping)
            and _integer(cell.get("winning_markets")) not in (None, 0)
            and _integer(cell.get("losing_markets")) not in (None, 0)
            for cell in cell_values
        )
    inferred = bool(
        no_markets is not None
        and no_markets >= 100
        and no_days is not None
        and no_days >= 14
        and wins is not None
        and wins >= 25
        and losses is not None
        and losses >= 25
        and required_cells_passed
    )
    return {
        "passed": (
            explicit is True and not failures if explicit is not None else inferred and not failures
        ),
        "explicit_pass": explicit,
        "support_failures": [str(value) for value in failures],
        "no_markets": no_markets,
        "no_utc_days": no_days,
        "no_winning_markets": wins,
        "no_losing_markets": losses,
        "required_time_cells_passed": required_cells_passed,
    }


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 and result == value else None


def _reject_economic_values(value: Any, path: str = "support_evidence") -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(
            str(key) for key in value if str(key).lower() in PROBABILITY_SELECTION_FORBIDDEN_COLUMNS
        )
        if forbidden:
            raise ValueError(
                f"probability selection cannot receive economic fields in {path}: "
                + ", ".join(forbidden)
            )
        for key, nested in value.items():
            _reject_economic_values(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_economic_values(nested, f"{path}[{index}]")


def _validate_evidence_scope(scope: str, qualification_eligible: bool) -> None:
    if scope not in {CONSUMED_EVIDENCE_SCOPE, FRESH_EVIDENCE_SCOPE}:
        raise ValueError("D4 evidence scope is not recognized")
    if qualification_eligible and scope != FRESH_EVIDENCE_SCOPE:
        raise ValueError("consumed D4 evidence cannot be qualification eligible")


def _validate_closed_selection_seal(
    selection: Mapping[str, Any],
    seal: Mapping[str, Any],
    seal_sha256: str,
) -> None:
    if selection.get("schema_version") != D4_SIDE_CALIBRATION_EVALUATION_SCHEMA_VERSION:
        raise ValueError("projected PnL requires a D4 probability-selection result")
    if selection.get("economics_used") is not False:
        raise ValueError("D4 probability selection was not economics-free")
    if selection.get("selected_candidate_id") not in {
        None,
        POOLED_NO_CALIBRATION_ID,
        TIME_LOCAL_NO_CALIBRATION_ID,
    }:
        raise ValueError("D4 probability selection contains an invalid winner")
    _validate_evidence_scope(
        str(selection.get("evidence_scope")),
        bool(selection.get("qualification_eligible")),
    )
    if not isinstance(seal_sha256, str) or not _SHA256_PATTERN.fullmatch(seal_sha256):
        raise ValueError("probability-selection seal SHA-256 is invalid")
    if (
        seal.get("selected_candidate_id") != selection.get("selected_candidate_id")
        or seal.get("economics_opened") is not False
        or seal.get("selection_uses_economics") is not False
        or not _SHA256_PATTERN.fullmatch(str(seal.get("probability_selection_sha256", "")))
    ):
        raise ValueError("probability-selection seal is incomplete or inconsistent")


def _eligible_universe(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"market_id", "window_start"}
    if not required.issubset(frame.columns) or frame.is_empty():
        raise ValueError("projected PnL requires a nonempty eligible-market universe")
    universe = frame.select("market_id", "window_start").unique()
    if universe.height != frame.select("market_id", "window_start").height:
        raise ValueError("eligible-market universe contains duplicate markets")
    return universe.sort("window_start", "market_id")


def _validated_ledger(
    frame: pl.DataFrame,
    candidate_id: str,
    universe: pl.DataFrame,
) -> pl.DataFrame:
    required = {
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "selected_yes",
        "won",
        "quantity",
        "selected_execution_cost_per_share",
        "selected_admission_cost_per_share",
        "selected_share_price",
        "selected_probability",
        "selected_underdog",
        "entry_debit",
        "realized_net",
        "selected_edge_per_share",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{candidate_id} projected-PnL ledger is missing: {', '.join(missing)}")
    if frame.select("market_id", "window_start").is_duplicated().any():
        raise ValueError(f"{candidate_id} projected-PnL ledger contains duplicate markets")
    if not frame.is_empty():
        if "candidate_id" in frame.columns:
            identities = frame["candidate_id"].drop_nulls().unique().to_list()
            if identities != [candidate_id]:
                raise ValueError(f"{candidate_id} projected-PnL ledger identity does not match")
        keys = set(frame.select("market_id", "window_start").iter_rows())
        if not keys.issubset(set(universe.iter_rows())):
            raise ValueError(f"{candidate_id} ledger contains ineligible markets")
        quantities = frame["quantity"].cast(pl.Float64).to_numpy()
        if not np.allclose(quantities, 5.0):
            raise ValueError(f"{candidate_id} projected PnL is fixed to five shares")
    return frame.sort("window_start", "observed_at")


def _model_pnl_report(
    candidate_id: str,
    ledger: pl.DataFrame,
    eligible_markets: int,
    selected_candidate_id: str | None,
    selection_qualification_eligible: bool,
) -> dict[str, Any]:
    metrics = ledger_metrics(ledger)
    yes_metrics = (
        ledger_metrics(ledger.filter(pl.col("selected_yes")))
        if not ledger.is_empty()
        else ledger_metrics(ledger)
    )
    no_metrics = (
        ledger_metrics(ledger.filter(~pl.col("selected_yes")))
        if not ledger.is_empty()
        else ledger_metrics(ledger)
    )
    selected = candidate_id == selected_candidate_id
    qualifies = bool(selected and selection_qualification_eligible)
    winning_trades = int(ledger["won"].sum()) if not ledger.is_empty() else 0
    losing_trades = metrics["trades"] - winning_trades
    return {
        "candidate_id": candidate_id,
        "role": (
            "incumbent_reference"
            if candidate_id == INCUMBENT_ID
            else "d4_architecture_control"
            if candidate_id == D4_BASE_ID
            else "probability_selected_challenger"
            if selected
            else "nonselected_challenger"
        ),
        "probability_selected": selected,
        "economic_qualification_eligible": qualifies,
        "diagnostic_only": not qualifies,
        "nonselected_pnl_can_change_probability_selection": False,
        "projected_pnl": {
            "trades": metrics["trades"],
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "accuracy": metrics["accuracy"],
            "net_profit": metrics["net_profit"],
            "stress_1c_net_profit": metrics.get("stress_1c_net_profit", 0.0),
            "net_expectancy_per_trade": metrics["net_expectancy_per_trade"],
            "stress_1c_net_expectancy_per_trade": metrics.get("stress_1c_net_expectancy_per_trade"),
            "profit_factor": metrics["profit_factor"],
            "maximum_drawdown": metrics.get("maximum_drawdown"),
        },
        "metrics": metrics,
        "coverage": {
            "eligible_markets": eligible_markets,
            "trades": metrics["trades"],
            "trades_per_eligible_market": metrics["trades"] / eligible_markets,
            "net_profit_per_eligible_market": metrics["net_profit"] / eligible_markets,
            "stress_1c_net_profit_per_eligible_market": metrics.get("stress_1c_net_profit", 0.0)
            / eligible_markets,
        },
        "sides": {"YES": yes_metrics, "NO": no_metrics},
    }


def _stressed_market_pnl(ledger: pl.DataFrame, name: str) -> pl.DataFrame:
    if ledger.is_empty():
        return ledger.select("market_id", "window_start").with_columns(
            pl.lit(0.0, dtype=pl.Float64).alias(name)
        )
    return ledger.select(
        "market_id",
        "window_start",
        (
            pl.col("quantity")
            * (
                pl.col("won").cast(pl.Float64)
                - (pl.col("selected_execution_cost_per_share") + EXECUTION_STRESS_PER_SHARE).clip(
                    upper_bound=1.0
                )
            )
        ).alias(name),
    )


def _paired_stressed_pnl_bootstrap(
    reference: pl.DataFrame,
    candidate: pl.DataFrame,
    universe: pl.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    daily = (
        universe.join(
            _stressed_market_pnl(reference, "reference_pnl"),
            on=["market_id", "window_start"],
            how="left",
            validate="1:1",
        )
        .join(
            _stressed_market_pnl(candidate, "candidate_pnl"),
            on=["market_id", "window_start"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("reference_pnl").fill_null(0.0),
            pl.col("candidate_pnl").fill_null(0.0),
            pl.col("window_start").dt.date().alias("utc_day"),
        )
        .group_by("utc_day")
        .agg(
            (pl.col("candidate_pnl") - pl.col("reference_pnl")).sum().alias("delta"),
            pl.len().alias("markets"),
        )
        .sort("utc_day")
    )
    delta = daily["delta"].to_numpy()
    markets = daily["markets"].to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, daily.height, daily.height)
        samples[index] = delta[chosen].sum() / markets[chosen].sum()
    point = float(delta.sum() / markets.sum())
    return {
        "block_unit": "utc_day",
        "utc_days": daily.height,
        "resamples": resamples,
        "seed": seed,
        "candidate_minus_reference_stress_1c_net_profit": float(delta.sum()),
        "candidate_minus_reference_stress_1c_net_profit_per_eligible_market": {
            "point": point,
            "lower_95": float(np.quantile(samples, 0.025)),
            "upper_95": float(np.quantile(samples, 0.975)),
        },
    }


def _selected_economic_qualification(
    candidate_id: str,
    reports: Mapping[str, Mapping[str, Any]],
    paired_to_d4: Mapping[str, Any],
    thresholds: D4ProjectedPnlThresholds,
    qualification_eligible: bool,
) -> dict[str, Any]:
    candidate = reports[candidate_id]
    d4 = reports[D4_BASE_ID]
    candidate_metrics = candidate["metrics"]
    d4_metrics = d4["metrics"]
    candidate_yes = candidate["sides"]["YES"]
    candidate_no = candidate["sides"]["NO"]
    d4_yes = d4["sides"]["YES"]
    d4_no = d4["sides"]["NO"]
    no_delta = candidate_no.get("stress_1c_net_profit", 0.0) - d4_no.get(
        "stress_1c_net_profit", 0.0
    )
    yes_floor = d4_yes.get("stress_1c_net_profit", 0.0) - (
        thresholds.maximum_yes_stressed_pnl_regression_fraction
        * abs(d4_yes.get("stress_1c_net_profit", 0.0))
    )
    d4_drawdown = abs(float(d4_metrics.get("maximum_drawdown") or 0.0))
    candidate_drawdown = abs(float(candidate_metrics.get("maximum_drawdown") or 0.0))
    d4_recovery = d4_metrics.get("loss_recovery_wins")
    candidate_recovery = candidate_metrics.get("loss_recovery_wins")
    profit_factor = candidate_metrics.get("profit_factor")
    profit_factor_observed = (
        profit_factor
        if profit_factor is not None
        else thresholds.minimum_profit_factor
        if candidate_metrics.get("profit_factor_no_losses")
        else None
    )
    gates = [
        _gate(
            "positive_stress_1c_expectancy",
            candidate_metrics.get("stress_1c_net_expectancy_per_trade"),
            0.0,
            ">",
        ),
        _gate(
            "minimum_profit_factor",
            profit_factor_observed,
            thresholds.minimum_profit_factor,
            ">=",
        ),
        _gate("positive_yes_stress_1c_pnl", candidate_yes.get("stress_1c_net_profit"), 0.0, ">"),
        _gate("positive_no_stress_1c_pnl", candidate_no.get("stress_1c_net_profit"), 0.0, ">"),
        _gate(
            "no_stress_1c_pnl_no_worse_than_d4_base",
            candidate_no.get("stress_1c_net_profit"),
            d4_no.get("stress_1c_net_profit"),
            ">=",
        ),
        _gate(
            "minimum_no_stress_1c_pnl_improvement",
            no_delta,
            thresholds.minimum_no_stressed_pnl_improvement,
            ">",
        ),
        _gate(
            "preserve_yes_stress_1c_pnl", candidate_yes.get("stress_1c_net_profit"), yes_floor, ">="
        ),
        _gate(
            "accuracy_no_worse_than_d4_base",
            candidate_metrics.get("accuracy"),
            d4_metrics.get("accuracy"),
            ">=",
        ),
        _gate(
            "maximum_drawdown_regression",
            candidate_drawdown,
            d4_drawdown * (1.0 + thresholds.maximum_drawdown_regression_fraction),
            "<=",
        ),
        _gate(
            "maximum_loss_recovery_regression",
            candidate_recovery,
            (
                float(d4_recovery) * (1.0 + thresholds.maximum_loss_recovery_regression_fraction)
                if d4_recovery is not None
                else None
            ),
            "<=",
        ),
        _gate(
            "positive_paired_stress_profit_per_eligible_market",
            paired_to_d4["candidate_minus_reference_stress_1c_net_profit_per_eligible_market"][
                "point"
            ],
            0.0,
            ">",
        ),
    ]
    gates_passed = all(gate["passed"] for gate in gates)
    return {
        "candidate_id": candidate_id,
        "economic_gates_passed": gates_passed,
        "qualification_eligible": qualification_eligible,
        "qualified": bool(gates_passed and qualification_eligible),
        "status": (
            "qualified"
            if gates_passed and qualification_eligible
            else "diagnostic_only_consumed_evidence"
            if not qualification_eligible
            else "not_qualified"
        ),
        "thresholds": asdict(thresholds),
        "gates": gates,
    }


def _gate(name: str, observed: Any, threshold: Any, operator: str) -> dict[str, Any]:
    passed = False
    if observed is not None and threshold is not None:
        if operator == "<=":
            passed = bool(observed <= threshold)
        elif operator == ">=":
            passed = bool(observed >= threshold)
        elif operator == ">":
            passed = bool(observed > threshold)
        elif operator == "==":
            passed = bool(observed == threshold)
        else:
            raise ValueError(f"unsupported D4 gate operator: {operator}")
    return {
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "operator": operator,
        "passed": passed,
    }
