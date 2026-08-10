"""Probability-only selection for asymmetric Polymarket-book residual challengers.

The selector deliberately operates before any economic reveal.  It evaluates a
frozen incumbent, one non-selectable static-book attribution control, and one to
four dynamic-book residual challengers on the same target opportunity keys.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import polars as pl

from .asymmetric_incumbent_evaluation import (
    PROBABILITY_KEY_COLUMNS,
    PROBABILITY_SELECTION_FORBIDDEN_COLUMNS,
    REQUIRED_CALIBRATION_CELLS,
    _shared_reference_probability_bootstrap,
    incumbent_probability_metrics,
    probability_only_first_crossings,
    target_opportunity_probability_cohort,
)

BOOK_ADMISSION_EVALUATION_SCHEMA_VERSION = "btc-asymmetric-book-admission-evaluation-v1"


@dataclass(frozen=True)
class BookAdmissionSelectionThresholds:
    """Predeclared probability gates for dynamic-book challenger selection."""

    noninferiority_margin: float = 0.005
    maximum_selected_bias: float = 0.03
    maximum_cell_bias: float = 0.05
    minimum_joint_noninferior_days: int = 8
    required_validation_days: int = 10
    minimum_feature_coverage: float = 0.90
    one_standard_error_multiplier: float = 1.0

    def __post_init__(self) -> None:
        for value, name in (
            (self.noninferiority_margin, "noninferiority margin"),
            (self.maximum_selected_bias, "maximum selected bias"),
            (self.maximum_cell_bias, "maximum cell bias"),
            (self.minimum_feature_coverage, "minimum feature coverage"),
            (self.one_standard_error_multiplier, "one-standard-error multiplier"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.maximum_selected_bias > 1.0 or self.maximum_cell_bias > 1.0:
            raise ValueError("probability-bias thresholds cannot exceed one")
        if self.minimum_feature_coverage > 1.0:
            raise ValueError("minimum feature coverage cannot exceed one")
        if self.required_validation_days <= 0:
            raise ValueError("required validation days must be positive")
        if not 1 <= self.minimum_joint_noninferior_days <= self.required_validation_days:
            raise ValueError("joint noninferior day requirement is outside the validation window")


def select_asymmetric_book_admission_challenger(
    incumbent: pl.DataFrame,
    static_control: pl.DataFrame,
    dynamic_challengers: Mapping[str, pl.DataFrame],
    support_evidence: Mapping[str, Mapping[str, Any]],
    thresholds: BookAdmissionSelectionThresholds,
    *,
    resamples: int,
    seed: int,
    static_candidate_id: str = "S0",
) -> dict[str, Any]:
    """Select at most one dynamic challenger without opening economics.

    ``support_evidence`` is keyed by the static and dynamic candidate IDs.  Each
    entry must establish independent support, feature coverage, and residual-cap
    compliance.  Dynamic entries additionally carry regularization strength and
    model complexity for the frozen deterministic ranking rule.
    """

    if not isinstance(thresholds, BookAdmissionSelectionThresholds):
        raise TypeError("book admission thresholds must use the frozen threshold type")
    if not isinstance(dynamic_challengers, Mapping):
        raise TypeError("dynamic challengers must be a candidate-to-frame mapping")
    if not isinstance(support_evidence, Mapping):
        raise TypeError("support evidence must be a candidate-to-evidence mapping")
    if not isinstance(static_candidate_id, str) or not static_candidate_id.strip():
        raise ValueError("static candidate ID must be a nonempty string")
    if not 1 <= len(dynamic_challengers) <= 4:
        raise ValueError("book admission selection requires one to four dynamic challengers")
    if static_candidate_id in dynamic_challengers:
        raise ValueError("static control cannot also be a dynamic challenger")
    if any(
        not isinstance(candidate_id, str) or not candidate_id.strip()
        for candidate_id in dynamic_challengers
    ):
        raise ValueError("dynamic candidate IDs must be nonempty strings")
    if resamples < 2 or seed < 0:
        raise ValueError("book admission bootstrap settings are invalid")
    _reject_economic_support(support_evidence)
    _validate_candidate_identity(static_control, static_candidate_id)
    for candidate_id, frame in dynamic_challengers.items():
        _validate_candidate_identity(frame, candidate_id)

    target_incumbent = target_opportunity_probability_cohort(incumbent)
    target_static = target_opportunity_probability_cohort(static_control)
    target_dynamic = {
        candidate_id: target_opportunity_probability_cohort(frame)
        for candidate_id, frame in dynamic_challengers.items()
    }
    _require_matched_keys_and_labels(target_incumbent, target_static, static_candidate_id)
    for candidate_id, frame in target_dynamic.items():
        _require_matched_keys_and_labels(target_incumbent, frame, candidate_id)

    incumbent_comparison = _shared_reference_probability_bootstrap(
        target_incumbent,
        target_dynamic,
        resamples=resamples,
        seed=seed,
    )
    static_comparison = _shared_reference_probability_bootstrap(
        target_static,
        target_dynamic,
        resamples=resamples,
        seed=seed,
    )
    static_to_incumbent = _shared_reference_probability_bootstrap(
        target_incumbent,
        {static_candidate_id: target_static},
        resamples=resamples,
        seed=seed,
    )

    incumbent_metrics = incumbent_probability_metrics(target_incumbent)
    static_metrics = incumbent_probability_metrics(target_static)
    incumbent_selected_metrics, incumbent_selected_count = _selected_metrics(target_incumbent)
    if incumbent_selected_metrics is None:
        raise RuntimeError("incumbent has no selected target opportunities")
    static_selected_metrics, static_selected_count = _selected_metrics(target_static)
    incumbent_selected_bias = abs(float(incumbent_selected_metrics["overall"]["bias"]))
    selected_bias_limit = min(thresholds.maximum_selected_bias, incumbent_selected_bias)

    static_support = _normalize_support(
        support_evidence.get(static_candidate_id),
        require_rank_metadata=False,
    )
    records: list[dict[str, Any]] = []
    for candidate_id in sorted(target_dynamic):
        frame = target_dynamic[candidate_id]
        metrics = incumbent_probability_metrics(frame)
        selected_metrics, selected_count = _selected_metrics(frame)
        support = _normalize_support(
            support_evidence.get(candidate_id),
            require_rank_metadata=True,
        )
        versus_incumbent = incumbent_comparison["comparisons"][candidate_id]
        versus_static = static_comparison["comparisons"][candidate_id]
        daily_counts = _joint_daily_noninferiority(
            versus_incumbent["daily"],
            versus_static["daily"],
            thresholds.noninferiority_margin,
        )
        selected_bias = (
            abs(float(selected_metrics["overall"]["bias"]))
            if selected_metrics is not None
            else None
        )
        cell_gates = _cell_bias_gates(metrics, thresholds.maximum_cell_bias)
        gates = [
            _gate(
                "validation_day_count",
                incumbent_comparison["utc_days"],
                thresholds.required_validation_days,
                "==",
            ),
            _gate("static_support_pass", static_support["support_passed"], True, "=="),
            _gate(
                "static_feature_coverage",
                static_support["coverage"],
                thresholds.minimum_feature_coverage,
                ">=",
            ),
            _gate(
                "static_residual_cap_pass",
                static_support["residual_cap_passed"],
                True,
                "==",
            ),
            _gate("candidate_support_pass", support["support_passed"], True, "=="),
            _gate(
                "candidate_feature_coverage",
                support["coverage"],
                thresholds.minimum_feature_coverage,
                ">=",
            ),
            _gate(
                "candidate_residual_cap_pass",
                support["residual_cap_passed"],
                True,
                "==",
            ),
            _gate(
                "ranking_metadata_complete",
                support["ranking_metadata_complete"],
                True,
                "==",
            ),
            _gate(
                "brier_point_no_worse_than_incumbent",
                versus_incumbent["brier_delta"]["point"],
                0.0,
                "<=",
            ),
            _gate(
                "log_loss_point_no_worse_than_incumbent",
                versus_incumbent["log_loss_delta"]["point"],
                0.0,
                "<=",
            ),
            _gate(
                "brier_simultaneous_noninferior_to_incumbent",
                versus_incumbent["brier_delta"]["simultaneous_upper_95"],
                thresholds.noninferiority_margin,
                "<=",
            ),
            _gate(
                "log_loss_simultaneous_noninferior_to_incumbent",
                versus_incumbent["log_loss_delta"]["simultaneous_upper_95"],
                thresholds.noninferiority_margin,
                "<=",
            ),
            _gate(
                "proper_score_improvement_over_incumbent",
                _one_point_improves(versus_incumbent),
                True,
                "==",
            ),
            _gate(
                "brier_point_no_worse_than_static",
                versus_static["brier_delta"]["point"],
                0.0,
                "<=",
            ),
            _gate(
                "log_loss_point_no_worse_than_static",
                versus_static["log_loss_delta"]["point"],
                0.0,
                "<=",
            ),
            _gate(
                "brier_simultaneous_noninferior_to_static",
                versus_static["brier_delta"]["simultaneous_upper_95"],
                thresholds.noninferiority_margin,
                "<=",
            ),
            _gate(
                "log_loss_simultaneous_noninferior_to_static",
                versus_static["log_loss_delta"]["simultaneous_upper_95"],
                thresholds.noninferiority_margin,
                "<=",
            ),
            _gate(
                "proper_score_improvement_over_static",
                _one_point_improves(versus_static),
                True,
                "==",
            ),
            _gate(
                "selected_side_absolute_bias",
                selected_bias,
                selected_bias_limit,
                "<=",
            ),
            _gate(
                "joint_noninferior_utc_days",
                daily_counts["common"],
                thresholds.minimum_joint_noninferior_days,
                ">=",
            ),
            *cell_gates,
        ]
        records.append(
            {
                "candidate_id": candidate_id,
                "selectable": True,
                "passed": all(gate["passed"] for gate in gates),
                "target_metrics": metrics,
                "selected_metrics": selected_metrics,
                "selected_opportunities": selected_count,
                "comparison_to_incumbent": versus_incumbent,
                "comparison_to_static": versus_static,
                "noninferior_utc_days_to_incumbent": daily_counts["incumbent"],
                "noninferior_utc_days_to_static": daily_counts["static"],
                "joint_noninferior_utc_days": daily_counts["common"],
                "support": support,
                "gates": gates,
            }
        )

    qualified = [record for record in records if record["passed"]]
    ranking = _rank_qualified(qualified, thresholds.one_standard_error_multiplier)
    selected_candidate_id = ranking[0]["candidate_id"] if ranking else None
    one_standard_error = _one_standard_error_trace(
        qualified,
        thresholds.one_standard_error_multiplier,
    )
    return {
        "schema_version": BOOK_ADMISSION_EVALUATION_SCHEMA_VERSION,
        "status": "selected" if selected_candidate_id else "blocked_no_quality_configuration",
        "selected_candidate_id": selected_candidate_id,
        "target_cohort": {
            "definition": "seconds 1-55 and either YES or NO raw VWAP5 in [0.20,0.30)",
            "rows": target_incumbent.height,
            "markets": target_incumbent["market_id"].n_unique(),
            "utc_days": target_incumbent["window_start"].dt.date().n_unique(),
        },
        "thresholds": asdict(thresholds),
        "selected_bias_limit": selected_bias_limit,
        "incumbent": {
            "target_metrics": incumbent_metrics,
            "selected_metrics": incumbent_selected_metrics,
            "selected_opportunities": incumbent_selected_count,
        },
        "static_control": {
            "candidate_id": static_candidate_id,
            "selectable": False,
            "target_metrics": static_metrics,
            "selected_metrics": static_selected_metrics,
            "selected_opportunities": static_selected_count,
            "comparison_to_incumbent": static_to_incumbent["comparisons"][static_candidate_id],
            "support": static_support,
        },
        "simultaneous_comparison_to_incumbent": incumbent_comparison,
        "simultaneous_comparison_to_static": static_comparison,
        "candidate_records": records,
        "one_standard_error": one_standard_error,
        "rank_trace": [
            {
                "rank": index + 1,
                "candidate_id": record["candidate_id"],
                "log_loss": record["target_metrics"]["overall"]["log_loss"],
                "brier": record["target_metrics"]["overall"]["brier"],
                "selected_absolute_bias": abs(record["selected_metrics"]["overall"]["bias"]),
                "regularization_strength": record["support"]["regularization_strength"],
                "model_complexity": record["support"]["model_complexity"],
            }
            for index, record in enumerate(ranking)
        ],
        "failure_trace": [
            {
                "candidate_id": record["candidate_id"],
                "failed_gates": [gate["name"] for gate in record["gates"] if not gate["passed"]],
            }
            for record in records
        ],
        "economics_used": False,
    }


def _selected_metrics(frame: pl.DataFrame) -> tuple[dict[str, Any] | None, int]:
    selected = probability_only_first_crossings(frame)
    if selected.is_empty():
        return None, 0
    return incumbent_probability_metrics(selected), selected.height


def _cell_bias_gates(metrics: Mapping[str, Any], limit: float) -> list[dict[str, Any]]:
    gates = []
    for required_name in REQUIRED_CALIBRATION_CELLS:
        evaluation_name = required_name.replace("_45_60", "_45_56")
        cell = metrics["time_cells"].get(evaluation_name)
        bias = abs(float(cell["bias"])) if cell and cell["bias"] is not None else None
        gates.append(_gate(f"target_cell_bias_{evaluation_name.lower()}", bias, limit, "<="))
    return gates


def _one_point_improves(comparison: Mapping[str, Any]) -> bool:
    return bool(
        comparison["brier_delta"]["point"] < 0.0 or comparison["log_loss_delta"]["point"] < 0.0
    )


def _joint_daily_noninferiority(
    versus_incumbent: Mapping[str, Mapping[str, float]],
    versus_static: Mapping[str, Mapping[str, float]],
    margin: float,
) -> dict[str, int]:
    if set(versus_incumbent) != set(versus_static):
        raise RuntimeError("daily incumbent and static comparisons do not share UTC days")
    incumbent_count = 0
    static_count = 0
    common_count = 0
    for utc_day in sorted(versus_incumbent):
        incumbent_passed = _daily_noninferior(versus_incumbent[utc_day], margin)
        static_passed = _daily_noninferior(versus_static[utc_day], margin)
        incumbent_count += int(incumbent_passed)
        static_count += int(static_passed)
        common_count += int(incumbent_passed and static_passed)
    return {
        "incumbent": incumbent_count,
        "static": static_count,
        "common": common_count,
    }


def _daily_noninferior(comparison: Mapping[str, float], margin: float) -> bool:
    return bool(comparison["brier_delta"] <= margin and comparison["log_loss_delta"] <= margin)


def _rank_qualified(
    qualified: list[dict[str, Any]],
    multiplier: float,
) -> list[dict[str, Any]]:
    trace = _one_standard_error_trace(qualified, multiplier)
    member_ids = set(trace["candidate_ids"])
    members = [record for record in qualified if record["candidate_id"] in member_ids]
    return sorted(
        members,
        key=lambda record: (
            record["target_metrics"]["overall"]["brier"],
            abs(record["selected_metrics"]["overall"]["bias"]),
            -record["support"]["regularization_strength"],
            record["support"]["model_complexity"],
            record["candidate_id"],
        ),
    )


def _one_standard_error_trace(
    qualified: list[dict[str, Any]],
    multiplier: float,
) -> dict[str, Any]:
    if not qualified:
        return {
            "best_log_loss_candidate_id": None,
            "best_log_loss": None,
            "standard_error": None,
            "upper_limit": None,
            "qualified_candidate_ids": [],
            "candidate_ids": [],
            "excluded_candidate_ids": [],
        }
    best = min(
        qualified,
        key=lambda record: (
            record["target_metrics"]["overall"]["log_loss"],
            record["candidate_id"],
        ),
    )
    standard_error = float(best["comparison_to_incumbent"]["log_loss_delta"]["standard_error"])
    best_log_loss = float(best["target_metrics"]["overall"]["log_loss"])
    upper_limit = best_log_loss + multiplier * standard_error
    qualified_candidate_ids = sorted(record["candidate_id"] for record in qualified)
    candidate_ids = sorted(
        record["candidate_id"]
        for record in qualified
        if record["target_metrics"]["overall"]["log_loss"] <= upper_limit + 1e-15
    )
    return {
        "best_log_loss_candidate_id": best["candidate_id"],
        "best_log_loss": best_log_loss,
        "standard_error": standard_error,
        "upper_limit": upper_limit,
        "qualified_candidate_ids": qualified_candidate_ids,
        "candidate_ids": candidate_ids,
        "excluded_candidate_ids": sorted(set(qualified_candidate_ids) - set(candidate_ids)),
    }


def _normalize_support(
    evidence: Mapping[str, Any] | None,
    *,
    require_rank_metadata: bool,
) -> dict[str, Any]:
    evidence = evidence if isinstance(evidence, Mapping) else {}
    support_passed = evidence.get("support_passed", evidence.get("passed")) is True
    residual_cap = evidence.get("residual_cap")
    residual_cap_passed = evidence.get("residual_cap_passed")
    if residual_cap_passed is None and isinstance(residual_cap, Mapping):
        residual_cap_passed = residual_cap.get("passed")
    coverage = _finite_number(
        evidence.get(
            "coverage",
            evidence.get("feature_coverage", evidence.get("dynamic_feature_coverage")),
        ),
        minimum=0.0,
        maximum=1.0,
    )
    regularization = _finite_number(
        evidence.get("regularization_strength", evidence.get("l2_strength")),
        minimum=0.0,
    )
    complexity = _nonnegative_integer(
        evidence.get(
            "model_complexity",
            evidence.get("complexity", evidence.get("complexity_rank")),
        )
    )
    ranking_complete = not require_rank_metadata or (
        regularization is not None and complexity is not None
    )
    return {
        "support_passed": support_passed,
        "coverage": coverage,
        "residual_cap_passed": residual_cap_passed is True,
        "ranking_metadata_complete": ranking_complete,
        "regularization_strength": regularization,
        "model_complexity": complexity,
    }


def _finite_number(
    value: Any,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < minimum:
        return None
    if maximum is not None and result > maximum:
        return None
    return result


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    if result < 0 or result != value:
        return None
    return result


def _validate_candidate_identity(frame: pl.DataFrame, candidate_id: str) -> None:
    if "candidate_id" not in frame.columns:
        return
    identities = frame["candidate_id"].drop_nulls().unique().to_list()
    if identities != [candidate_id]:
        raise ValueError(f"{candidate_id} frame candidate identity does not match its mapping key")


def _require_matched_keys_and_labels(
    reference: pl.DataFrame,
    challenger: pl.DataFrame,
    candidate_id: str,
) -> None:
    ordered_reference = reference.sort(*PROBABILITY_KEY_COLUMNS)
    ordered_challenger = challenger.sort(*PROBABILITY_KEY_COLUMNS)
    if not ordered_challenger.select(*PROBABILITY_KEY_COLUMNS).equals(
        ordered_reference.select(*PROBABILITY_KEY_COLUMNS),
        null_equal=True,
    ):
        raise ValueError(f"{candidate_id} keys do not match the incumbent")
    if not ordered_challenger["label_up"].equals(ordered_reference["label_up"]):
        raise ValueError(f"{candidate_id} labels do not match the incumbent")


def _reject_economic_support(value: Any, path: str = "support_evidence") -> None:
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
            _reject_economic_support(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_economic_support(nested, f"{path}[{index}]")


def _gate(name: str, observed: Any, threshold: Any, operator: str) -> dict[str, Any]:
    passed = False
    if observed is not None:
        if operator == "<=":
            passed = bool(observed <= threshold)
        elif operator == ">=":
            passed = bool(observed >= threshold)
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
