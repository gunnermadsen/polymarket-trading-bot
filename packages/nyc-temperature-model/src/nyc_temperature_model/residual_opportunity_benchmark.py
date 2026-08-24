from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from . import PROCESS_ID
from .asymmetric_benchmark import (
    AsymmetricPolicy,
    _candidate_coverage,
    _candidate_key,
    _compact_policy_metrics,
    _date_range,
    _json_default,
    _json_safe,
    _label_metrics,
    _policy_metrics,
    _price_cells,
)
from .config import Settings
from .database import connection
from .fees import taker_fee_per_share
from .residual_opportunity import (
    BOOTSTRAP_BLOCK_DAYS,
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    FEATURE_SCHEMA_VERSION,
    LOWER_QUANTILE,
    MAXIMUM_ALL_IN_COST,
    MINIMUM_ALL_IN_COST,
    PROBABILITY_CLIP,
    REFIT_INTERVAL_EVENT_DAYS,
    RIDGE_PENALTY,
    WARMUP_EVENT_DAYS,
    apply_rolling_market_offset,
    market_offset_probability,
)

BENCHMARK_METHOD = "rolling_market_offset_residual"
EVIDENCE_STATUS = "exploratory_post_holdout_redesign"
EXPECTED_DISCOVERY_START = date(2026, 4, 14)
EXPECTED_DISCOVERY_END = date(2026, 6, 30)
EXPECTED_EVALUATION_START = date(2026, 7, 1)
EXPECTED_EVALUATION_END = date(2026, 7, 30)
FIXED_POLICY = AsymmetricPolicy(
    name="balanced_asymmetry:midnight_then_noon",
    decision_mode="midnight_then_noon",
    minimum_all_in_cost=MINIMUM_ALL_IN_COST,
    maximum_all_in_cost=MAXIMUM_ALL_IN_COST,
    minimum_robust_edge=0.04,
    minimum_robust_roi=0.35,
    rank_by="robust_expected_roi",
)
NYC = ZoneInfo("America/New_York")
EXPECTED_QUANTITY = 5.0
EXPECTED_MODELED_SLIPPAGE_PER_SHARE = 0.01
ECONOMIC_ABSOLUTE_TOLERANCE = 5e-11


def _runtime_provenance(weather_model_image_id: str) -> dict[str, Any]:
    revision = os.environ.get("POLYMARKET_GIT_REVISION", "")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("POLYMARKET_GIT_REVISION must be a 40-character lowercase Git revision")
    if not weather_model_image_id.startswith("sha256:") or len(weather_model_image_id) != 71:
        raise ValueError("weather model image ID must be sha256:<64 lowercase hex characters>")
    if any(
        character not in "0123456789abcdef"
        for character in weather_model_image_id[7:]
    ):
        raise ValueError("weather model image ID contains non-hexadecimal characters")
    generated_at = datetime.now(UTC)
    prospective_start = generated_at.astimezone(NYC).date() + timedelta(days=1)
    return {
        "git_revision": revision,
        "runner_declared_weather_model_image_id": weather_model_image_id,
        "weather_model_image_id_source_contract": (
            "runner supplies .Id from docker image inspect for the exact launched image"
        ),
        "generated_at": generated_at,
        "earliest_prospective_event_date": prospective_start,
    }


def _source_candidate_digest(candidates: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        _json_safe(candidates),
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_policy_run(database_url: str, source_policy_run_id: str) -> dict[str, Any]:
    with connection(database_url) as conn:
        row = conn.execute(
            """
            SELECT policy_run_id::text,process_id::text,discovery_start,discovery_end,
                   evaluation_start,evaluation_end,quantity::double precision,
                   decision_models,policy,qualified,report_uri
            FROM weather.asymmetric_policy_runs
            WHERE process_id=%s AND policy_run_id=%s
            """,
            (PROCESS_ID, source_policy_run_id),
        ).fetchone()
    if not row:
        raise ValueError(f"unknown process-owned source policy run: {source_policy_run_id}")
    output = dict(row)
    expected_ranges = (
        EXPECTED_DISCOVERY_START,
        EXPECTED_DISCOVERY_END,
        EXPECTED_EVALUATION_START,
        EXPECTED_EVALUATION_END,
    )
    observed_ranges = (
        output["discovery_start"],
        output["discovery_end"],
        output["evaluation_start"],
        output["evaluation_end"],
    )
    if observed_ranges != expected_ranges:
        raise ValueError("source policy run does not match the frozen April-July evidence window")
    if not math.isclose(float(output["quantity"]), EXPECTED_QUANTITY, abs_tol=1e-12):
        raise ValueError("residual opportunity benchmark requires five-share source evidence")
    if output["policy"] != asdict(FIXED_POLICY):
        raise ValueError("source run does not contain the frozen balanced asymmetric policy")
    return output


def _validate_event_partition_label(row: dict[str, Any]) -> None:
    if row["canonical_event_date"] != row["event_date"]:
        raise ValueError(
            f"source market {row['market_id']} event date differs from its ledger row"
        )
    if (
        row["event_partition_min_date"] != row["event_date"]
        or row["event_partition_max_date"] != row["event_date"]
    ):
        raise ValueError(f"source event {row['event_id']} spans multiple event dates")
    if not row["event_partition_label_complete"] or row["label_available_at"] is None:
        raise ValueError(f"source event {row['event_id']} lacks a complete resolution label")
    if int(row["event_partition_bucket_count"]) < 1:
        raise ValueError(f"source event {row['event_id']} has no resolution buckets")
    if int(row["event_partition_winner_count"]) != 1:
        raise ValueError(f"source event {row['event_id']} must have exactly one winning bucket")
    canonical_side_resolution = (
        bool(row["canonical_resolved_yes"])
        if row["side"] == "YES"
        else not bool(row["canonical_resolved_yes"])
    )
    if bool(row["resolved_side"]) != canonical_side_resolution:
        raise ValueError(
            f"source candidate {row['market_id']} {row['side']} has a stale resolution"
        )


def _source_candidates(
    database_url: str, source_policy_run_id: str
) -> list[dict[str, Any]]:
    with connection(database_url) as conn:
        rows = conn.execute(
            """
            WITH scoped_ledger AS (
              SELECT l.*,m.event_id,m.event_date AS canonical_event_date,
                     m.resolved_yes AS canonical_resolved_yes
              FROM weather.asymmetric_candidate_ledger l
              JOIN weather.temperature_markets m ON m.market_id=l.market_id
              WHERE l.process_id=%s AND l.policy_run_id=%s
            ),
            scoped_events AS (
              SELECT DISTINCT event_id FROM scoped_ledger
            ),
            event_labels AS (
              SELECT m.event_id,
                     count(*)::int AS event_partition_bucket_count,
                     count(*) FILTER (WHERE m.resolved_yes IS TRUE)::int
                       AS event_partition_winner_count,
                     bool_and(m.resolved_yes IS NOT NULL AND m.resolved_at IS NOT NULL)
                       AS event_partition_label_complete,
                     max(m.resolved_at) AS label_available_at,
                     min(m.event_date) AS event_partition_min_date,
                     max(m.event_date) AS event_partition_max_date
              FROM weather.temperature_markets m
              JOIN scoped_events e ON e.event_id=m.event_id
              GROUP BY m.event_id
            )
            SELECT l.model_run_id::text,l.market_id,l.split,l.event_date,l.decision_time,
                   l.decision_hour_local,l.side,l.quantity::double precision,
                   l.probability::double precision,l.probability_lower::double precision,
                   l.market_probability_proxy::double precision,
                   l.ask_vwap::double precision,l.fees_enabled,
                   l.fee_rate::double precision,l.fee_exponent::double precision,
                   l.fee_taker_only,l.fee_per_share::double precision,
                   l.modeled_slippage_per_share::double precision,
                   l.all_in_cost_per_share::double precision,
                   l.break_even_probability::double precision,
                   l.model_edge_per_share::double precision,
                   l.robust_edge_per_share::double precision,
                   l.expected_roi::double precision,l.robust_expected_roi::double precision,
                   l.resolved_side,l.executable,l.realized_net_per_share::double precision,
                   l.source_timestamp,l.quote_age_seconds::double precision,
                   l.quality_flags,l.rejection_reasons,
                   l.event_id,l.canonical_event_date,l.canonical_resolved_yes,
                   labels.event_partition_bucket_count,
                   labels.event_partition_winner_count,
                   labels.event_partition_label_complete,
                   labels.label_available_at,labels.event_partition_min_date,
                   labels.event_partition_max_date
            FROM scoped_ledger l
            JOIN event_labels labels ON labels.event_id=l.event_id
            ORDER BY l.event_date,l.decision_time,l.market_id,l.side
            """,
            (PROCESS_ID, source_policy_run_id),
        ).fetchall()
    candidates = []
    for raw in rows:
        row = dict(raw)
        _validate_event_partition_label(row)
        probability = float(row["probability"])
        probability_lower = float(row["probability_lower"])
        row.update(
            {
                "process_id": PROCESS_ID,
                "weather_probability": probability,
                "weather_probability_lower": probability_lower,
                "probability_source": "weather_distribution",
                "feature_schema_version": None,
                "opportunity_fit_id": None,
                "market_probability_input": None,
                "weather_market_logit_residual": None,
                "eligible": False,
                "selected": False,
                # Executable rows had no execution-quality rejection before the old
                # economic policy was applied. Discard its stale probability gates.
                "rejection_reasons": (
                    [] if row["executable"] else list(row["rejection_reasons"] or [])
                ),
                "quality_flags": list(row["quality_flags"] or []),
            }
        )
        candidates.append(row)
    if not candidates:
        raise ValueError("source policy run has no candidate ledger rows")
    return candidates


def _validate_source_candidate_invariants(
    candidates: list[dict[str, Any]], source_run: dict[str, Any]
) -> None:
    raw_decision_models = source_run.get("decision_models")
    if not isinstance(raw_decision_models, dict):
        raise TypeError("source policy run decision_models must be an object")
    decision_models = {
        str(hour): str(model_id) for hour, model_id in raw_decision_models.items()
    }
    if set(decision_models) != {"0", "12"}:
        raise ValueError("source policy run must identify exactly midnight and noon models")
    expected_hour_models = {
        (int(hour), model_id) for hour, model_id in decision_models.items()
    }
    observed_hour_models: set[tuple[int, str]] = set()
    contract_sides: dict[tuple[str, str, datetime], set[str]] = defaultdict(set)
    for candidate in candidates:
        quantity = float(candidate["quantity"])
        if not math.isclose(quantity, EXPECTED_QUANTITY, abs_tol=1e-12):
            raise ValueError(
                f"source candidate {candidate['market_id']} has non-five-share quantity"
            )
        hour = int(candidate["decision_hour_local"])
        model_id = str(candidate["model_run_id"])
        observed_hour_models.add((hour, model_id))
        if decision_models.get(str(hour)) != model_id:
            raise ValueError(
                f"source candidate {candidate['market_id']} hour/model mapping differs from run"
            )
        contract_sides[(model_id, candidate["market_id"], candidate["decision_time"])].add(
            str(candidate["side"])
        )
        if not candidate["executable"]:
            continue
        slippage = candidate.get("modeled_slippage_per_share")
        if slippage is None or not math.isclose(
            float(slippage), EXPECTED_MODELED_SLIPPAGE_PER_SHARE, abs_tol=1e-12
        ):
            raise ValueError(
                f"source candidate {candidate['market_id']} does not use one-cent slippage"
            )
        required_fields = (
            "ask_vwap",
            "fee_per_share",
            "all_in_cost_per_share",
            "break_even_probability",
            "realized_net_per_share",
        )
        if any(candidate.get(field) is None for field in required_fields):
            raise ValueError(
                f"executable source candidate {candidate['market_id']} lacks economics"
            )
        execution_price = float(candidate["ask_vwap"]) + float(slippage)
        if not 0 < execution_price <= 1:
            raise ValueError(
                f"source candidate {candidate['market_id']} execution price is invalid"
            )
        expected_fee = taker_fee_per_share(
            execution_price,
            quantity=quantity,
            enabled=bool(candidate["fees_enabled"]),
            rate=float(candidate["fee_rate"]),
            exponent=float(candidate["fee_exponent"]),
        )
        if not math.isclose(
            float(candidate["fee_per_share"]),
            expected_fee,
            rel_tol=0.0,
            abs_tol=ECONOMIC_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(
                f"source candidate {candidate['market_id']} captured fee does not recompute"
            )
        expected_all_in = execution_price + expected_fee
        for field in ("all_in_cost_per_share", "break_even_probability"):
            if not math.isclose(
                float(candidate[field]),
                expected_all_in,
                rel_tol=0.0,
                abs_tol=ECONOMIC_ABSOLUTE_TOLERANCE,
            ):
                raise ValueError(
                    f"source candidate {candidate['market_id']} {field} does not recompute"
                )
        expected_realized = (
            (1.0 if bool(candidate["resolved_side"]) else 0.0) - expected_all_in
        )
        if not math.isclose(
            float(candidate["realized_net_per_share"]),
            expected_realized,
            rel_tol=0.0,
            abs_tol=ECONOMIC_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(
                f"source candidate {candidate['market_id']} realized PnL does not recompute"
            )
    if observed_hour_models != expected_hour_models:
        raise ValueError("source candidate hour/model coverage differs from source run")
    incomplete_contracts = [
        key for key, sides in contract_sides.items() if sides != {"YES", "NO"}
    ]
    if incomplete_contracts:
        raise ValueError("source candidate ledger must contain YES and NO for every contract")


def _canonicalize_fit_records(
    candidates: list[dict[str, Any]],
    fit_records: list[dict[str, Any]],
    *,
    source_policy_run_id: str,
    source_candidate_sha256: str,
) -> list[dict[str, Any]]:
    replacements: dict[str, str] = {}
    canonical_records = []
    ordered = sorted(fit_records, key=lambda row: (row["origin_time"], row["training_end"]))
    for record in ordered:
        old_fit_id = str(record["opportunity_fit_id"])
        if old_fit_id in replacements:
            raise ValueError("residual fit identifiers must be unique")
        fit_core = {
            key: value for key, value in record.items() if key != "opportunity_fit_id"
        }
        fit_payload = {
            "process_id": PROCESS_ID,
            "source_policy_run_id": source_policy_run_id,
            "source_candidate_ledger_sha256": source_candidate_sha256,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "fit": fit_core,
        }
        fit_sha256 = _canonical_sha256(fit_payload)
        canonical_fit_id = f"residual-fit-{fit_sha256[:32]}"
        replacements[old_fit_id] = canonical_fit_id
        canonical_records.append(
            {
                **fit_core,
                "process_id": PROCESS_ID,
                "source_policy_run_id": source_policy_run_id,
                "source_candidate_ledger_sha256": source_candidate_sha256,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "opportunity_fit_id": canonical_fit_id,
                "fit_record_sha256": fit_sha256,
            }
        )
    referenced_fit_ids = {
        str(candidate["opportunity_fit_id"])
        for candidate in candidates
        if candidate.get("opportunity_fit_id") is not None
    }
    if not referenced_fit_ids.issubset(replacements):
        raise ValueError("scored candidates reference an unidentified residual fit")
    for candidate in candidates:
        old_fit_id = candidate.get("opportunity_fit_id")
        if old_fit_id is not None:
            candidate["opportunity_fit_id"] = replacements[str(old_fit_id)]
    return canonical_records


def _finalize_scored_candidate_ledger(
    candidates: list[dict[str, Any]],
    *,
    selected: list[dict[str, Any]],
    rejection_reasons: dict[tuple[str, str, datetime, str], list[str]],
) -> None:
    selected_keys = {_candidate_key(candidate) for candidate in selected}
    for candidate in candidates:
        key = _candidate_key(candidate)
        if key not in rejection_reasons:
            raise ValueError("scored candidate is missing fixed-policy disposition")
        reasons = sorted(set(rejection_reasons[key]))
        candidate["rejection_reasons"] = reasons
        candidate["eligible"] = not reasons
        candidate["selected"] = key in selected_keys


def _retrospective_screen_supported(checks: dict[str, bool]) -> bool:
    required = (
        "evaluation_positive_total_net",
        "evaluation_positive_return_on_capital",
        "evaluation_positive_lower_90pct_daily_net",
        "evaluation_positive_without_best_trade",
        "evaluation_net_exceeds_weather_only",
    )
    return all(checks[key] for key in required)


def _weighted_probability_scores(
    rows: list[dict[str, Any]],
    weights: np.ndarray,
    outcomes: np.ndarray,
    probability_name: str,
) -> dict[str, float]:
    probabilities = np.clip(
        np.asarray([row[probability_name] for row in rows], dtype=np.float64),
        1e-12,
        1.0 - 1e-12,
    )
    weight_total = float(weights.sum())
    return {
        "weighted_binary_log_loss": -float(
            np.sum(
                weights
                * (
                    outcomes * np.log(probabilities)
                    + (1.0 - outcomes) * np.log(1.0 - probabilities)
                )
            )
            / weight_total
        ),
        "weighted_binary_brier_score": float(
            np.sum(weights * (probabilities - outcomes) ** 2) / weight_total
        ),
    }


def _probability_metrics(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    contracts: dict[tuple[str, str, datetime], dict[str, dict[str, Any]]] = defaultdict(dict)
    for candidate in candidates:
        contracts[
            (candidate["model_run_id"], candidate["market_id"], candidate["decision_time"])
        ][candidate["side"]] = candidate
    rows = []
    for sides in contracts.values():
        yes = sides.get("YES")
        no = sides.get("NO")
        if yes is None or no is None or yes.get("market_probability_proxy") is None:
            continue
        scored = yes if yes.get("probability_source") == "residual_market_offset" else no
        if scored.get("probability_source") != "residual_market_offset":
            continue
        residual_yes = (
            float(scored["probability"])
            if scored["side"] == "YES"
            else 1.0 - float(scored["probability"])
        )
        rows.append(
            {
                "event_date": yes["event_date"],
                "hour": int(yes["decision_hour_local"]),
                "outcome": float(yes["resolved_side"]),
                "weather": float(yes["weather_probability"]),
                "market": float(yes["market_probability_proxy"]),
                "residual": residual_yes,
            }
        )
    output: dict[str, Any] = {}
    for label, selected in (
        ("all", rows),
        ("midnight", [row for row in rows if row["hour"] == 0]),
        ("noon", [row for row in rows if row["hour"] == 12]),
    ):
        if not selected:
            output[label] = {"rows": 0, "event_days": 0}
            continue
        counts = defaultdict(int)
        for row in selected:
            counts[row["event_date"]] += 1
        weights = np.asarray([1.0 / counts[row["event_date"]] for row in selected])
        outcomes = np.asarray([row["outcome"] for row in selected], dtype=np.float64)

        output[label] = {
            "rows": len(selected),
            "event_days": len(counts),
            "weather": _weighted_probability_scores(
                selected, weights, outcomes, "weather"
            ),
            "market": _weighted_probability_scores(
                selected, weights, outcomes, "market"
            ),
            "residual": _weighted_probability_scores(
                selected, weights, outcomes, "residual"
            ),
        }
    return output


def _market_only_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = copy.deepcopy(candidates)
    for candidate in output:
        market = candidate.get("market_probability_proxy")
        cost = candidate.get("all_in_cost_per_share")
        if (
            not candidate.get("executable")
            or market is None
            or cost is None
            or not MINIMUM_ALL_IN_COST <= float(cost) <= MAXIMUM_ALL_IN_COST
        ):
            candidate["rejection_reasons"] = sorted(
                set(candidate["rejection_reasons"]) | {"market_comparator_unavailable"}
            )
            continue
        probability = float(market)
        candidate["probability"] = probability
        candidate["probability_lower"] = probability
        point_edge = probability - float(cost)
        candidate["model_edge_per_share"] = point_edge
        candidate["robust_edge_per_share"] = point_edge
        candidate["expected_roi"] = point_edge / float(cost)
        candidate["robust_expected_roi"] = point_edge / float(cost)
    return output


def _equal_blend_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = copy.deepcopy(candidates)
    for candidate in output:
        market = candidate.get("market_probability_proxy")
        cost = candidate.get("all_in_cost_per_share")
        if (
            not candidate.get("executable")
            or market is None
            or cost is None
            or not MINIMUM_ALL_IN_COST <= float(cost) <= MAXIMUM_ALL_IN_COST
        ):
            candidate["rejection_reasons"] = sorted(
                set(candidate["rejection_reasons"]) | {"blend_comparator_unavailable"}
            )
            continue
        probability = market_offset_probability(
            float(candidate["weather_probability"]), float(market), 0.5
        )
        conservative = market_offset_probability(
            float(candidate["weather_probability_lower"]), float(market), 0.5
        )
        candidate["probability"] = probability
        candidate["probability_lower"] = min(probability, conservative)
        point_edge = probability - float(cost)
        robust_edge = candidate["probability_lower"] - float(cost)
        candidate["model_edge_per_share"] = point_edge
        candidate["robust_edge_per_share"] = robust_edge
        candidate["expected_roi"] = point_edge / float(cost)
        candidate["robust_expected_roi"] = robust_edge / float(cost)
    return output


def _selected_trade_stress(
    selected: list[dict[str, Any]], slippage_per_share: float
) -> dict[str, Any]:
    pnl = []
    debit = []
    for trade in selected:
        execution_price = float(trade["ask_vwap"]) + slippage_per_share
        if not 0 < execution_price <= 1:
            raise ValueError("stress execution price falls outside contract payout")
        fee = taker_fee_per_share(
            execution_price,
            quantity=float(trade["quantity"]),
            enabled=bool(trade["fees_enabled"]),
            rate=float(trade["fee_rate"]),
            exponent=float(trade["fee_exponent"]),
        )
        all_in = execution_price + fee
        if all_in > 1:
            raise ValueError("stress all-in cost exceeds contract payout")
        quantity = float(trade["quantity"])
        debit.append(all_in * quantity)
        pnl.append(((1.0 if trade["resolved_side"] else 0.0) - all_in) * quantity)
    return {
        "policy_reoptimized": False,
        "selected_trade_set_frozen": True,
        "modeled_slippage_per_share": slippage_per_share,
        "trades": len(selected),
        "wins": sum(value > 0 for value in pnl),
        "total_entry_debit": float(sum(debit)),
        "total_net": float(sum(pnl)),
        "return_on_deployed_capital": (
            float(sum(pnl) / sum(debit)) if debit else None
        ),
        "worst_trade": min(pnl) if pnl else None,
    }


def _selected_trade_rows(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "event_date": row["event_date"],
            "decision_time": row["decision_time"],
            "market_id": row["market_id"],
            "side": row["side"],
            "ask_vwap": row["ask_vwap"],
            "all_in_cost_per_share": row["all_in_cost_per_share"],
            "weather_probability": row["weather_probability"],
            "market_probability": row["market_probability_input"],
            "residual_probability": row["probability"],
            "residual_probability_lower": row["probability_lower"],
            "opportunity_fit_id": row["opportunity_fit_id"],
            "resolved_side": row["resolved_side"],
            "realized_net": row["realized_net_per_share"] * row["quantity"],
        }
        for row in selected
    ]


def run_residual_opportunity_benchmark(
    settings: Settings,
    *,
    source_policy_run_id: str,
    weather_model_image_id: str,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    provenance = _runtime_provenance(weather_model_image_id)
    source_run = _source_policy_run(settings.database_url, source_policy_run_id)
    base_candidates = _source_candidates(settings.database_url, source_policy_run_id)
    _validate_source_candidate_invariants(base_candidates, source_run)
    source_candidate_sha256 = _source_candidate_digest(base_candidates)
    scored_candidates, fit_records = apply_rolling_market_offset(
        base_candidates, bootstrap_iterations=bootstrap_iterations
    )
    fit_records = _canonicalize_fit_records(
        scored_candidates,
        fit_records,
        source_policy_run_id=source_policy_run_id,
        source_candidate_sha256=source_candidate_sha256,
    )
    development_candidates = [
        row for row in scored_candidates if row["split"] == "discovery"
    ]
    evaluation_candidates = [
        row for row in scored_candidates if row["split"] == "evaluation"
    ]
    development_dates = _date_range(EXPECTED_DISCOVERY_START, EXPECTED_DISCOVERY_END)
    evaluation_dates = _date_range(EXPECTED_EVALUATION_START, EXPECTED_EVALUATION_END)
    development_metrics, development_trades, development_reasons = _policy_metrics(
        development_candidates, development_dates, FIXED_POLICY
    )
    evaluation_metrics, evaluation_trades, evaluation_reasons = _policy_metrics(
        evaluation_candidates, evaluation_dates, FIXED_POLICY
    )
    _finalize_scored_candidate_ledger(
        scored_candidates,
        selected=development_trades + evaluation_trades,
        rejection_reasons={**development_reasons, **evaluation_reasons},
    )
    scored_candidate_sha256 = _source_candidate_digest(scored_candidates)

    weather_development = [row for row in base_candidates if row["split"] == "discovery"]
    weather_evaluation = [row for row in base_candidates if row["split"] == "evaluation"]
    weather_development_metrics, _, _ = _policy_metrics(
        weather_development, development_dates, FIXED_POLICY
    )
    weather_evaluation_metrics, _, _ = _policy_metrics(
        weather_evaluation, evaluation_dates, FIXED_POLICY
    )
    market_candidates = _market_only_candidates(base_candidates)
    market_development_metrics, _, _ = _policy_metrics(
        [row for row in market_candidates if row["split"] == "discovery"],
        development_dates,
        FIXED_POLICY,
    )
    market_evaluation_metrics, _, _ = _policy_metrics(
        [row for row in market_candidates if row["split"] == "evaluation"],
        evaluation_dates,
        FIXED_POLICY,
    )
    blend_candidates = _equal_blend_candidates(base_candidates)
    blend_development_metrics, _, _ = _policy_metrics(
        [row for row in blend_candidates if row["split"] == "discovery"],
        development_dates,
        FIXED_POLICY,
    )
    blend_evaluation_metrics, _, _ = _policy_metrics(
        [row for row in blend_candidates if row["split"] == "evaluation"],
        evaluation_dates,
        FIXED_POLICY,
    )

    checks = {
        "retrospective_evidence_only": True,
        "evaluation_positive_total_net": evaluation_metrics["total_net"] > 0,
        "evaluation_positive_return_on_capital": (
            evaluation_metrics["return_on_deployed_capital"] > 0
        ),
        "evaluation_positive_lower_90pct_daily_net": (
            evaluation_metrics["lower_90pct_block_bootstrap_mean_daily_net"] > 0
        ),
        "evaluation_positive_without_best_trade": (
            evaluation_metrics["net_without_best_trade"] > 0
        ),
        "evaluation_net_exceeds_weather_only": (
            evaluation_metrics["total_net"] > weather_evaluation_metrics["total_net"]
        ),
        "at_least_50_prospective_trades": False,
        "at_least_10_prospective_wins_and_losses": False,
    }
    retrospective_screen_supported = _retrospective_screen_supported(checks)
    development_keys = {_candidate_key(row) for row in development_trades}
    evaluation_keys = {_candidate_key(row) for row in evaluation_trades}
    report_id = str(uuid.uuid4())
    settings.report_directory.mkdir(parents=True, exist_ok=True)
    report_path = settings.report_directory / f"residual-opportunity-{report_id}.json"
    report = {
        "schema_version": "nyc-temperature-residual-opportunity-v1",
        "benchmark_method": BENCHMARK_METHOD,
        "evidence_status": EVIDENCE_STATUS,
        "objective": "positive_net_expectancy_at_low_executable_cost_without_accuracy_gate",
        "process_id": PROCESS_ID,
        "provenance": provenance,
        "source_policy_run_id": source_policy_run_id,
        "source_candidate_ledger_rows": len(base_candidates),
        "source_candidate_ledger_sha256": source_candidate_sha256,
        "scored_candidate_ledger_rows": len(scored_candidates),
        "scored_candidate_ledger_sha256": scored_candidate_sha256,
        "scored_candidate_ledger_digest_contract": (
            "sha256(canonical JSON: UTF-8, sorted keys, compact separators, JSON-safe values)"
        ),
        "scored_candidate_ledger": scored_candidates,
        "source_policy_run": source_run,
        "model_contract": {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "formula": (
                "logit(q_side)=logit(market_side)+weather_residual_weight_horizon*"
                "(logit(weather_side)-logit(market_side))"
            ),
            "probability_clip": PROBABILITY_CLIP,
            "ridge_penalty": RIDGE_PENALTY,
            "coefficient_bounds": [0.0, 1.0],
            "training_side": "canonical_yes_only",
            "event_date_total_weight": 1.0,
            "warmup_event_days": WARMUP_EVENT_DAYS,
            "refit_interval_event_days": REFIT_INTERVAL_EVENT_DAYS,
            "label_availability": (
                "maximum_bucket_resolved_at_for_exact_event_id_partition_strictly_before_origin"
            ),
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_block_days": BOOTSTRAP_BLOCK_DAYS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "probability_lower_quantile": LOWER_QUANTILE,
            "hyperparameter_selection_using_pnl": False,
        },
        "execution_contract": {
            "quantity": EXPECTED_QUANTITY,
            "modeled_slippage_per_share": EXPECTED_MODELED_SLIPPAGE_PER_SHARE,
            "minimum_all_in_cost": MINIMUM_ALL_IN_COST,
            "maximum_all_in_cost": MAXIMUM_ALL_IN_COST,
            "minimum_robust_edge": FIXED_POLICY.minimum_robust_edge,
            "minimum_robust_roi": FIXED_POLICY.minimum_robust_roi,
            "decision_mode": FIXED_POLICY.decision_mode,
            "maximum_positions_per_event_day": 1,
            "dynamic_captured_fee_schedule": True,
        },
        "evidence_contract": {
            "april_july_designation": "retrospective_development_only",
            "reason": "July price-cell results motivated this model after the old holdout opened",
            "production_qualification_allowed": False,
            "prospective_start_rule": (
                "first midnight decision after git revision, image, and model config freeze"
            ),
            "earliest_prospective_event_date": provenance[
                "earliest_prospective_event_date"
            ],
            "minimum_prospective_trades": 50,
            "minimum_prospective_wins": 10,
            "minimum_prospective_losses": 10,
        },
        "fit_records": fit_records,
        "probability_metrics": {
            "development": _probability_metrics(development_candidates),
            "evaluation": _probability_metrics(evaluation_candidates),
        },
        "fixed_policy": asdict(FIXED_POLICY),
        "development": {
            "metrics": development_metrics,
            "candidate_coverage": _candidate_coverage(development_candidates),
            "price_cells": _price_cells(development_candidates, development_keys),
            "selected_trades": _selected_trade_rows(development_trades),
        },
        "evaluation": {
            "metrics": evaluation_metrics,
            "candidate_coverage": _candidate_coverage(evaluation_candidates),
            "price_cells": _price_cells(evaluation_candidates, evaluation_keys),
            "selected_trades": _selected_trade_rows(evaluation_trades),
        },
        "comparators": {
            "no_trade": {"development_total_net": 0.0, "evaluation_total_net": 0.0},
            "weather_only": {
                "development": _compact_policy_metrics(weather_development_metrics),
                "evaluation": _compact_policy_metrics(weather_evaluation_metrics),
            },
            "market_only": {
                "development": _compact_policy_metrics(market_development_metrics),
                "evaluation": _compact_policy_metrics(market_evaluation_metrics),
            },
            "equal_logit_blend": {
                "development": _compact_policy_metrics(blend_development_metrics),
                "evaluation": _compact_policy_metrics(blend_evaluation_metrics),
            },
        },
        "selected_trade_cost_stress": {
            f"{slippage:.3f}": {
                "development": _selected_trade_stress(development_trades, slippage),
                "evaluation": _selected_trade_stress(evaluation_trades, slippage),
            }
            for slippage in (0.0, 0.005, 0.01, 0.02)
        },
        "labels": _label_metrics(
            settings.database_url, EXPECTED_DISCOVERY_START, EXPECTED_EVALUATION_END
        ),
        "qualification_checks": checks,
        "retrospective_screen_supported": retrospective_screen_supported,
        "production_qualified": False,
        "report_id": report_id,
        "report_uri": str(report_path),
    }
    report = _json_safe(report)
    temporary = report_path.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n")
    temporary.replace(report_path)
    return report
