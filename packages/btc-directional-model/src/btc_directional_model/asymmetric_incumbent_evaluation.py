"""Matched evaluation for calibration-only asymmetric incumbent challengers.

This module deliberately separates the probability-selection boundary from the
post-selection economic comparison.  The probability helpers never emit realized
profit, debit, or execution-cost columns.  Economic helpers operate only on already
selected policy ledgers and compare them on one explicit eligible-market universe.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

from .asymmetric_value_evaluation import EXECUTION_STRESS_PER_SHARE, ledger_metrics

INCUMBENT_EVALUATION_SCHEMA_VERSION = "btc-asymmetric-incumbent-evaluation-v1"
INCUMBENT_CORRECTION_LEDGER_SCHEMA_VERSION = "btc-asymmetric-incumbent-correction-ledger-v1"
DEFAULT_TIME_BANDS = ((1, 15), (15, 30), (30, 45), (45, 56))
REQUIRED_CALIBRATION_TIME_BANDS = ((1, 15), (15, 30), (30, 45), (45, 60))
REQUIRED_CALIBRATION_CELLS = tuple(
    f"{side}_{lower}_{upper}"
    for side in ("YES", "NO")
    for lower, upper in REQUIRED_CALIBRATION_TIME_BANDS
)
PROBABILITY_KEY_COLUMNS = (
    "market_id",
    "window_start",
    "observed_at",
    "seconds_elapsed",
)
PROBABILITY_SELECTION_FORBIDDEN_COLUMNS = frozenset(
    {
        "realized_net",
        "entry_debit",
        "pnl",
        "net_profit",
        "profit_factor",
        "capital_efficiency",
        "maximum_drawdown",
        "average_loss",
        "maximum_loss",
        "stress_1c_net_profit",
        "stress_1c_net_expectancy_per_trade",
    }
)
PROBABILITY_SELECTION_OUTPUT_COLUMNS = (
    "market_id",
    "window_start",
    "observed_at",
    "seconds_elapsed",
    "label_up",
    "probability_yes",
    "selected_yes",
    "selected_probability",
    "selected_label",
    "time_cell",
)


def probability_only_first_crossings(
    scored: pl.DataFrame,
    *,
    minimum_share_price: float = 0.20,
    maximum_share_price: float = 0.30,
    minimum_edge_per_share: float = 0.03,
    maximum_seconds_elapsed: int = 55,
    quantity: float = 5.0,
    maximum_depth_participation: float = 0.25,
    maximum_admission_cost_per_share: float = 0.35,
) -> pl.DataFrame:
    """Return the first frozen-policy opportunity without economic outcomes.

    Edges are recomputed from each candidate's probability and the frozen
    admission costs.  This prevents a challenger probability from being paired
    with stale incumbent edge columns.  The returned frame contains labels for
    proper-score/calibration evaluation, but never ``won``, PnL, debit, fee,
    price, edge, or execution-cost columns.
    """

    _reject_preseal_economics(scored)
    required = {
        *PROBABILITY_KEY_COLUMNS,
        "label_up",
        "probability_yes",
        "yes_ask_vwap_5",
        "no_ask_vwap_5",
        "yes_ask_depth",
        "no_ask_depth",
        "yes_cost_per_share",
        "no_cost_per_share",
    }
    _require_columns(scored, required, "probability-only first crossing")
    _validate_policy_bounds(
        minimum_share_price=minimum_share_price,
        maximum_share_price=maximum_share_price,
        minimum_edge_per_share=minimum_edge_per_share,
        maximum_seconds_elapsed=maximum_seconds_elapsed,
        quantity=quantity,
        maximum_depth_participation=maximum_depth_participation,
        maximum_admission_cost_per_share=maximum_admission_cost_per_share,
    )
    _validate_probability_values(scored)
    identity_columns = _single_identity_columns(scored)
    duplicate_keys = [*identity_columns, *PROBABILITY_KEY_COLUMNS]
    if scored.select(*duplicate_keys).is_duplicated().any():
        raise ValueError("probability-only first crossing contains duplicate decision keys")

    within_time = pl.col("seconds_elapsed").is_between(
        1,
        maximum_seconds_elapsed,
        closed="both",
    )
    yes_edge = pl.col("probability_yes") - pl.col("yes_cost_per_share")
    no_edge = 1.0 - pl.col("probability_yes") - pl.col("no_cost_per_share")
    yes_eligible = (
        within_time
        & pl.col("yes_ask_vwap_5").is_between(
            minimum_share_price,
            maximum_share_price,
            closed="left",
        )
        & (pl.col("yes_cost_per_share") <= maximum_admission_cost_per_share)
        & (quantity <= pl.col("yes_ask_depth") * maximum_depth_participation)
        & (yes_edge >= minimum_edge_per_share)
    ).fill_null(False)
    no_eligible = (
        within_time
        & pl.col("no_ask_vwap_5").is_between(
            minimum_share_price,
            maximum_share_price,
            closed="left",
        )
        & (pl.col("no_cost_per_share") <= maximum_admission_cost_per_share)
        & (quantity <= pl.col("no_ask_depth") * maximum_depth_participation)
        & (no_edge >= minimum_edge_per_share)
    ).fill_null(False)
    staged = scored.with_columns(
        yes_edge.alias("_yes_edge"),
        no_edge.alias("_no_edge"),
        yes_eligible.alias("_yes_eligible"),
        no_eligible.alias("_no_eligible"),
    ).filter(pl.col("_yes_eligible") | pl.col("_no_eligible"))
    if staged.is_empty():
        return _empty_probability_selection(scored, identity_columns)
    staged = staged.with_columns(
        pl.when(pl.col("_yes_eligible") & pl.col("_no_eligible"))
        .then(pl.col("_yes_edge") >= pl.col("_no_edge"))
        .otherwise(pl.col("_yes_eligible"))
        .alias("selected_yes")
    ).with_columns(
        pl.when(pl.col("selected_yes"))
        .then(pl.col("probability_yes"))
        .otherwise(1.0 - pl.col("probability_yes"))
        .alias("selected_probability"),
        pl.when(pl.col("selected_yes"))
        .then(pl.col("label_up"))
        .otherwise(1 - pl.col("label_up"))
        .alias("selected_label"),
        _time_cell_expression(DEFAULT_TIME_BANDS).alias("time_cell"),
    )
    market_keys = [*identity_columns, "market_id", "window_start"]
    selected = (
        staged.sort(
            *identity_columns, "market_id", "window_start", "seconds_elapsed", "observed_at"
        )
        .group_by(*market_keys, maintain_order=True)
        .first()
        .select(*identity_columns, *PROBABILITY_SELECTION_OUTPUT_COLUMNS)
    )
    forbidden_output = PROBABILITY_SELECTION_FORBIDDEN_COLUMNS.intersection(selected.columns)
    if forbidden_output:
        raise RuntimeError("probability selection emitted economic columns")
    return selected


def incumbent_probability_metrics(
    frame: pl.DataFrame,
    *,
    time_bands: tuple[tuple[int, int], ...] = DEFAULT_TIME_BANDS,
) -> dict[str, Any]:
    """Calculate deterministic market-equal probability quality diagnostics."""

    required = {
        "market_id",
        "window_start",
        "seconds_elapsed",
        "label_up",
        "probability_yes",
    }
    _require_columns(frame, required, "incumbent probability metrics")
    _validate_time_bands(time_bands)
    _validate_probability_values(frame)
    if frame.is_empty():
        raise ValueError("incumbent probability metrics require rows")
    keys = [column for column in PROBABILITY_KEY_COLUMNS if column in frame.columns]
    if frame.select(*keys).is_duplicated().any():
        raise ValueError("incumbent probability metrics contain duplicate decision keys")

    probability = frame["probability_yes"].to_numpy().astype(np.float64)
    labels = frame["label_up"].to_numpy().astype(np.float64)
    market_ids = frame["market_id"].cast(pl.String).to_numpy()
    result: dict[str, Any] = {
        "overall": _probability_metrics(probability, labels, market_ids),
        "sides": {},
        "time_cells": {},
        "utc_days": {},
        "utc_day_time_cells": {},
    }
    dates = frame["window_start"].dt.date().to_list()
    for utc_day in sorted(set(dates)):
        mask = np.asarray([value == utc_day for value in dates], dtype=bool)
        result["utc_days"][utc_day.isoformat()] = _probability_metrics(
            probability[mask],
            labels[mask],
            market_ids[mask],
        )

    side_cohorts = _side_probability_cohorts(frame, probability, labels)
    seconds = frame["seconds_elapsed"].to_numpy()
    for side, (mask, side_probability, side_labels) in side_cohorts.items():
        result["sides"][side] = _probability_metrics(
            side_probability,
            side_labels,
            market_ids[mask],
            allow_empty=True,
        )
        for lower, upper in time_bands:
            cell_mask = mask & (seconds >= lower) & (seconds < upper)
            if side == "YES":
                cell_probability = probability[cell_mask]
                cell_labels = labels[cell_mask]
            else:
                cell_probability = 1.0 - probability[cell_mask]
                cell_labels = 1.0 - labels[cell_mask]
            result["time_cells"][f"{side}_{lower}_{upper}"] = _probability_metrics(
                cell_probability,
                cell_labels,
                market_ids[cell_mask],
                allow_empty=True,
            )
    for utc_day in sorted(set(dates)):
        day_mask = np.asarray([value == utc_day for value in dates], dtype=bool)
        day_cells: dict[str, Any] = {}
        for side, (side_mask, _, _) in side_cohorts.items():
            for lower, upper in time_bands:
                cell_mask = day_mask & side_mask & (seconds >= lower) & (seconds < upper)
                cell_probability = (
                    probability[cell_mask] if side == "YES" else 1.0 - probability[cell_mask]
                )
                cell_labels = labels[cell_mask] if side == "YES" else 1.0 - labels[cell_mask]
                day_cells[f"{side}_{lower}_{upper}"] = _probability_metrics(
                    cell_probability,
                    cell_labels,
                    market_ids[cell_mask],
                    allow_empty=True,
                )
        result["utc_day_time_cells"][utc_day.isoformat()] = day_cells
    return result


def target_opportunity_probability_cohort(
    frame: pl.DataFrame,
    *,
    minimum_share_price: float = 0.20,
    maximum_share_price: float = 0.30,
    maximum_seconds_elapsed: int = 55,
) -> pl.DataFrame:
    """Project the frozen matched target grid before probability comparison."""

    _reject_preseal_economics(frame)
    _require_columns(
        frame,
        {
            *PROBABILITY_KEY_COLUMNS,
            "label_up",
            "probability_yes",
            "yes_ask_vwap_5",
            "no_ask_vwap_5",
        },
        "target opportunity probability cohort",
    )
    if not (
        np.isfinite(minimum_share_price)
        and np.isfinite(maximum_share_price)
        and 0.0 <= minimum_share_price < maximum_share_price <= 1.0
    ):
        raise ValueError("target opportunity share-price bounds are invalid")
    if maximum_seconds_elapsed < 1:
        raise ValueError("target opportunity maximum second must be positive")
    _validate_matched_probability_frame(frame, "target opportunity")
    target = frame.filter(
        pl.col("seconds_elapsed").is_between(1, maximum_seconds_elapsed, closed="both")
        & (
            pl.col("yes_ask_vwap_5").is_between(
                minimum_share_price,
                maximum_share_price,
                closed="left",
            )
            | pl.col("no_ask_vwap_5").is_between(
                minimum_share_price,
                maximum_share_price,
                closed="left",
            )
        )
    ).sort(*PROBABILITY_KEY_COLUMNS)
    if target.is_empty():
        raise RuntimeError("target opportunity probability cohort is empty")
    return target


def simultaneous_paired_probability_bootstrap(
    incumbent: pl.DataFrame,
    challengers: Mapping[str, pl.DataFrame],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare three challengers with a shared UTC-day max-t bootstrap."""

    if len(challengers) != 3:
        raise ValueError("simultaneous probability comparison requires exactly three challengers")
    if resamples <= 0 or seed < 0:
        raise ValueError("simultaneous probability bootstrap settings are invalid")
    _validate_matched_probability_frame(incumbent, "incumbent")
    candidate_ids = sorted(challengers)
    reference = incumbent.sort(*PROBABILITY_KEY_COLUMNS)
    labels = reference["label_up"].to_numpy().astype(np.float64)
    incumbent_probability = reference["probability_yes"].to_numpy().astype(np.float64)
    dates = reference["window_start"].dt.date().to_list()
    markets = reference["market_id"].cast(pl.String).to_list()
    market_keys = [(day.isoformat(), market) for day, market in zip(dates, markets, strict=True)]
    unique_market_keys = sorted(set(market_keys))
    utc_days = sorted({key[0] for key in unique_market_keys})
    if len(utc_days) < 2:
        raise ValueError("UTC-day bootstrap requires at least two days")
    market_index = {key: index for index, key in enumerate(unique_market_keys)}
    day_index = {day: index for index, day in enumerate(utc_days)}
    market_day_index = np.asarray([day_index[key[0]] for key in unique_market_keys])
    daily_market_count = np.bincount(market_day_index, minlength=len(utc_days)).astype(np.float64)

    point = np.empty((len(candidate_ids), 2), dtype=np.float64)
    daily_sums = np.zeros((len(candidate_ids), 2, len(utc_days)), dtype=np.float64)
    daily: dict[str, dict[str, dict[str, float]]] = {}
    incumbent_brier = np.square(incumbent_probability - labels)
    incumbent_log_loss = _row_log_loss(labels, incumbent_probability)
    for candidate_index, candidate_id in enumerate(candidate_ids):
        candidate = challengers[candidate_id]
        _validate_matched_probability_frame(candidate, candidate_id)
        ordered = candidate.sort(*PROBABILITY_KEY_COLUMNS)
        if not ordered.select(*PROBABILITY_KEY_COLUMNS).equals(
            reference.select(*PROBABILITY_KEY_COLUMNS),
            null_equal=True,
        ):
            raise ValueError(f"{candidate_id} keys do not match the incumbent")
        if not ordered["label_up"].equals(reference["label_up"]):
            raise ValueError(f"{candidate_id} labels do not match the incumbent")
        candidate_probability = ordered["probability_yes"].to_numpy().astype(np.float64)
        deltas = np.column_stack(
            (
                np.square(candidate_probability - labels) - incumbent_brier,
                _row_log_loss(labels, candidate_probability) - incumbent_log_loss,
            )
        )
        market_sums = np.zeros((len(unique_market_keys), 2), dtype=np.float64)
        market_rows = np.zeros(len(unique_market_keys), dtype=np.int64)
        for row_index, key in enumerate(market_keys):
            index = market_index[key]
            market_sums[index] += deltas[row_index]
            market_rows[index] += 1
        market_means = market_sums / market_rows[:, None]
        point[candidate_index] = market_means.mean(axis=0)
        for market_position, utc_day_position in enumerate(market_day_index):
            daily_sums[candidate_index, :, utc_day_position] += market_means[market_position]
        daily[candidate_id] = {
            utc_day: {
                "brier_delta": float(
                    daily_sums[candidate_index, 0, day_index[utc_day]]
                    / daily_market_count[day_index[utc_day]]
                ),
                "log_loss_delta": float(
                    daily_sums[candidate_index, 1, day_index[utc_day]]
                    / daily_market_count[day_index[utc_day]]
                ),
            }
            for utc_day in utc_days
        }

    rng = np.random.default_rng(seed)
    sampled_days = rng.integers(0, len(utc_days), size=(resamples, len(utc_days)))
    samples = np.empty((resamples, len(candidate_ids), 2), dtype=np.float64)
    for sample_index, chosen in enumerate(sampled_days):
        denominator = max(float(daily_market_count[chosen].sum()), 1.0)
        samples[sample_index] = daily_sums[:, :, chosen].sum(axis=2) / denominator
    standard_errors = samples.std(axis=0, ddof=1)
    standardized = np.zeros_like(samples)
    nonzero = standard_errors > 0.0
    standardized[:, nonzero] = (samples[:, nonzero] - point[nonzero]) / standard_errors[nonzero]
    critical = float(np.quantile(standardized.max(axis=(1, 2)), 0.975))

    comparisons: dict[str, Any] = {}
    for candidate_index, candidate_id in enumerate(candidate_ids):
        metrics: dict[str, Any] = {}
        for metric_index, metric in enumerate(("brier_delta", "log_loss_delta")):
            values = samples[:, candidate_index, metric_index]
            metrics[metric] = {
                "point": float(point[candidate_index, metric_index]),
                "lower_95": float(np.quantile(values, 0.025)),
                "upper_95": float(np.quantile(values, 0.975)),
                "standard_error": float(standard_errors[candidate_index, metric_index]),
                "simultaneous_upper_95": float(
                    point[candidate_index, metric_index]
                    + critical * standard_errors[candidate_index, metric_index]
                ),
            }
        metrics["daily"] = daily[candidate_id]
        comparisons[candidate_id] = metrics
    return {
        "schema_version": INCUMBENT_EVALUATION_SCHEMA_VERSION,
        "block_unit": "utc_day",
        "utc_days": len(utc_days),
        "markets": len(unique_market_keys),
        "rows": incumbent.height,
        "challengers": candidate_ids,
        "resamples": resamples,
        "seed": seed,
        "max_t_critical_95": critical,
        "comparisons": comparisons,
    }


def select_incumbent_calibration_challenger(
    incumbent: pl.DataFrame,
    challengers: Mapping[str, pl.DataFrame],
    calibration_support: Mapping[str, Mapping[str, Any]],
    *,
    resamples: int,
    seed: int,
    incumbent_selected_bias: float = 0.02694,
    noninferiority_margin: float = 0.005,
    minimum_noninferior_days: int = 8,
    maximum_cell_bias: float = 0.05,
) -> dict[str, Any]:
    """Apply frozen probability gates and select one deterministic challenger."""

    for value, name in (
        (incumbent_selected_bias, "incumbent selected bias"),
        (noninferiority_margin, "noninferiority margin"),
        (maximum_cell_bias, "maximum cell bias"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if minimum_noninferior_days <= 0:
        raise ValueError("minimum noninferior days must be positive")
    target_incumbent = target_opportunity_probability_cohort(incumbent)
    target_challengers = {
        candidate_id: target_opportunity_probability_cohort(candidate)
        for candidate_id, candidate in challengers.items()
    }
    comparison = simultaneous_paired_probability_bootstrap(
        target_incumbent,
        target_challengers,
        resamples=resamples,
        seed=seed,
    )
    target_keys = target_incumbent.select(*PROBABILITY_KEY_COLUMNS)
    for candidate_id, candidate in target_challengers.items():
        if not candidate.select(*PROBABILITY_KEY_COLUMNS).equals(target_keys, null_equal=True):
            raise RuntimeError(
                f"{candidate_id} target cohort does not preserve incumbent decision keys"
            )
    incumbent_metrics = incumbent_probability_metrics(target_incumbent)
    incumbent_selected = probability_only_first_crossings(target_incumbent)
    if incumbent_selected.is_empty():
        raise RuntimeError("incumbent has no probability-only selected opportunities")
    incumbent_selected_metrics = incumbent_probability_metrics(incumbent_selected)

    records: list[dict[str, Any]] = []
    for candidate_id in sorted(target_challengers):
        candidate = target_challengers[candidate_id]
        metrics = incumbent_probability_metrics(candidate)
        selected = probability_only_first_crossings(candidate)
        selected_metrics = (
            incumbent_probability_metrics(selected) if not selected.is_empty() else None
        )
        delta = comparison["comparisons"][candidate_id]
        support = _calibration_support_evidence(
            calibration_support.get(candidate_id),
        )
        noninferior_days = sum(
            values["brier_delta"] <= noninferiority_margin
            and values["log_loss_delta"] <= noninferiority_margin
            for values in delta["daily"].values()
        )
        selected_bias = (
            abs(float(selected_metrics["overall"]["bias"]))
            if selected_metrics is not None
            else None
        )
        cell_bias_checks = []
        for cell_name in REQUIRED_CALIBRATION_CELLS:
            evaluation_cell_name = cell_name.replace("_45_60", "_45_56")
            cell = (selected_metrics or {}).get("time_cells", {}).get(evaluation_cell_name)
            observed = abs(float(cell["bias"])) if cell and cell["bias"] is not None else None
            cell_bias_checks.append(
                _gate(
                    f"selected_cell_bias_{evaluation_cell_name.lower()}",
                    observed,
                    maximum_cell_bias,
                    "<=",
                )
            )
        gates = [
            _gate("all_eight_cells_fitted", support["passed"], True, "=="),
            _gate("brier_point_no_worse", delta["brier_delta"]["point"], 0.0, "<="),
            _gate(
                "log_loss_point_no_worse",
                delta["log_loss_delta"]["point"],
                0.0,
                "<=",
            ),
            _gate(
                "brier_simultaneous_noninferiority",
                delta["brier_delta"]["simultaneous_upper_95"],
                noninferiority_margin,
                "<=",
            ),
            _gate(
                "log_loss_simultaneous_noninferiority",
                delta["log_loss_delta"]["simultaneous_upper_95"],
                noninferiority_margin,
                "<=",
            ),
            _gate(
                "at_least_one_proper_score_improves",
                bool(delta["brier_delta"]["point"] < 0.0 or delta["log_loss_delta"]["point"] < 0.0),
                True,
                "==",
            ),
            _gate(
                "selected_opportunity_absolute_bias",
                selected_bias,
                min(0.03, incumbent_selected_bias),
                "<=",
            ),
            _gate(
                "noninferior_utc_days",
                noninferior_days,
                minimum_noninferior_days,
                ">=",
            ),
            *cell_bias_checks,
        ]
        records.append(
            {
                "candidate_id": candidate_id,
                "passed": all(item["passed"] for item in gates),
                "metrics": metrics,
                "selected_opportunity_metrics": selected_metrics,
                "selected_opportunities": selected.height,
                "comparison_to_incumbent": delta,
                "calibration_support": support,
                "noninferior_utc_days": noninferior_days,
                "gates": gates,
            }
        )
    qualified = [record for record in records if record["passed"]]
    rank_trace = sorted(
        qualified,
        key=lambda record: (
            record["metrics"]["overall"]["log_loss"],
            record["metrics"]["overall"]["brier"],
            abs(record["selected_opportunity_metrics"]["overall"]["bias"]),
            record["candidate_id"],
        ),
    )
    selected_candidate_id = rank_trace[0]["candidate_id"] if rank_trace else None
    return {
        "schema_version": INCUMBENT_EVALUATION_SCHEMA_VERSION,
        "status": "selected" if selected_candidate_id else "blocked_no_quality_configuration",
        "selected_candidate_id": selected_candidate_id,
        "target_cohort": {
            "definition": "seconds 1-55 and either YES or NO raw VWAP5 in [0.20,0.30)",
            "rows": target_incumbent.height,
            "markets": target_incumbent["market_id"].n_unique(),
            "utc_days": target_incumbent["window_start"].dt.date().n_unique(),
        },
        "incumbent_metrics": incumbent_metrics,
        "incumbent_selected_opportunity_metrics": incumbent_selected_metrics,
        "simultaneous_comparison": comparison,
        "candidate_records": records,
        "rank_trace": [
            {
                "rank": index + 1,
                "candidate_id": record["candidate_id"],
                "log_loss": record["metrics"]["overall"]["log_loss"],
                "brier": record["metrics"]["overall"]["brier"],
                "selected_absolute_bias": abs(
                    record["selected_opportunity_metrics"]["overall"]["bias"]
                ),
            }
            for index, record in enumerate(rank_trace)
        ],
        "economics_used": False,
    }


def build_incumbent_correction_ledger(
    incumbent_ledger: pl.DataFrame,
    challenger_ledger: pl.DataFrame,
    *,
    quantity: float = 5.0,
    stress_per_share: float = EXECUTION_STRESS_PER_SHARE,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Reconcile every incumbent/challenger addition, removal, side, or timing change."""

    if not np.isfinite(quantity) or not math.isclose(quantity, 5.0):
        raise ValueError("correction ledger is fixed to exactly five shares")
    if not np.isfinite(stress_per_share) or not math.isclose(
        stress_per_share,
        EXECUTION_STRESS_PER_SHARE,
    ):
        raise ValueError("correction ledger is fixed to exactly +1c stress")
    required = {
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "selected_yes",
        "won",
        "quantity",
        "selected_execution_cost_per_share",
        "realized_net",
    }
    for name, frame in (("incumbent", incumbent_ledger), ("challenger", challenger_ledger)):
        _require_columns(frame, required, f"{name} correction ledger")
        _validate_policy_ledger(frame, name, quantity)

    keys = ["market_id", "window_start"]
    incumbent = _prefix_ledger(
        incumbent_ledger,
        "incumbent",
        keys,
        stress_per_share=stress_per_share,
    )
    challenger = _prefix_ledger(
        challenger_ledger,
        "challenger",
        keys,
        stress_per_share=stress_per_share,
    )
    joined = incumbent.join(
        challenger,
        on=keys,
        how="full",
        coalesce=True,
        validate="1:1",
    ).sort("window_start", "market_id")
    joined = joined.with_columns(
        pl.col("incumbent_present").fill_null(False),
        pl.col("challenger_present").fill_null(False),
        pl.col("incumbent_realized_net").fill_null(0.0),
        pl.col("challenger_realized_net").fill_null(0.0),
        pl.col("incumbent_stress_1c_net").fill_null(0.0),
        pl.col("challenger_stress_1c_net").fill_null(0.0),
    )
    both = pl.col("incumbent_present") & pl.col("challenger_present")
    side_changed = both & (pl.col("incumbent_selected_yes") != pl.col("challenger_selected_yes"))
    same_side = both & ~side_changed
    inconsistent_same_side = joined.filter(
        same_side & (pl.col("incumbent_won") != pl.col("challenger_won"))
    )
    inconsistent_side_change = joined.filter(
        side_changed & (pl.col("incumbent_won") == pl.col("challenger_won"))
    )
    if inconsistent_same_side.height or inconsistent_side_change.height:
        raise ValueError("incumbent and challenger outcomes do not reconcile by selected side")
    retimed = same_side & (
        (pl.col("incumbent_seconds_elapsed") != pl.col("challenger_seconds_elapsed"))
        | (pl.col("incumbent_observed_at") != pl.col("challenger_observed_at"))
    )
    joined = joined.with_columns(
        pl.when(pl.col("incumbent_present") & ~pl.col("challenger_present"))
        .then(
            pl.when(pl.col("incumbent_won"))
            .then(pl.lit("incumbent_only_win_suppressed"))
            .otherwise(pl.lit("incumbent_only_loss_avoided"))
        )
        .when(pl.col("challenger_present") & ~pl.col("incumbent_present"))
        .then(
            pl.when(pl.col("challenger_won"))
            .then(pl.lit("candidate_only_win"))
            .otherwise(pl.lit("candidate_only_loss"))
        )
        .when(side_changed)
        .then(
            pl.when(pl.col("challenger_won"))
            .then(pl.lit("correct_side_change"))
            .otherwise(pl.lit("incorrect_side_change"))
        )
        .when(retimed)
        .then(pl.lit("retimed_same_side"))
        .otherwise(pl.lit("unchanged_same_side"))
        .alias("correction_category"),
        (pl.col("challenger_realized_net") - pl.col("incumbent_realized_net")).alias(
            "incremental_pnl_exact"
        ),
        (pl.col("challenger_stress_1c_net") - pl.col("incumbent_stress_1c_net")).alias(
            "incremental_pnl_stress_1c"
        ),
    ).with_columns(
        pl.when(
            pl.col("correction_category").is_in(
                ["incumbent_only_loss_avoided", "correct_side_change", "candidate_only_win"]
            )
        )
        .then(pl.lit(1))
        .when(
            pl.col("correction_category").is_in(
                [
                    "incumbent_only_win_suppressed",
                    "incorrect_side_change",
                    "candidate_only_loss",
                ]
            )
        )
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("corrected_decision_score")
    )
    categories = {
        category: joined.filter(pl.col("correction_category") == category).height
        for category in (
            "incumbent_only_loss_avoided",
            "incumbent_only_win_suppressed",
            "correct_side_change",
            "incorrect_side_change",
            "candidate_only_win",
            "candidate_only_loss",
            "retimed_same_side",
            "unchanged_same_side",
        )
    }
    daily_corrections = (
        joined.with_columns(pl.col("window_start").dt.date().alias("utc_day"))
        .group_by("utc_day")
        .agg(pl.col("corrected_decision_score").sum().alias("net_corrected_decisions"))
    )
    category_pnl = {
        row[0]: {
            "exact": float(row[1]),
            "stress_1c": float(row[2]),
        }
        for row in joined.group_by("correction_category")
        .agg(
            pl.col("incremental_pnl_exact").sum(),
            pl.col("incremental_pnl_stress_1c").sum(),
        )
        .iter_rows()
    }
    summary = {
        "schema_version": INCUMBENT_CORRECTION_LEDGER_SCHEMA_VERSION,
        "markets": joined.height,
        "categories": categories,
        "avoided_losses": categories["incumbent_only_loss_avoided"],
        "suppressed_wins": categories["incumbent_only_win_suppressed"],
        "correct_side_changes": categories["correct_side_change"],
        "incorrect_side_changes": categories["incorrect_side_change"],
        "candidate_only_wins": categories["candidate_only_win"],
        "candidate_only_losses": categories["candidate_only_loss"],
        "incumbent_only_wins": categories["incumbent_only_win_suppressed"],
        "incumbent_only_losses": categories["incumbent_only_loss_avoided"],
        "retimed_same_side": categories["retimed_same_side"],
        "net_corrected_decisions": int(joined["corrected_decision_score"].sum()),
        "corrected_decision_improvement_utc_days": daily_corrections.filter(
            pl.col("net_corrected_decisions") > 0
        ).height,
        "incremental_pnl_exact": float(joined["incremental_pnl_exact"].sum()),
        "incremental_pnl_stress_1c": float(joined["incremental_pnl_stress_1c"].sum()),
        "incremental_pnl_by_category": category_pnl,
    }
    if sum(categories.values()) != joined.height:
        raise RuntimeError("correction categories do not reconcile to the market union")
    expected_score = (
        summary["avoided_losses"]
        + summary["correct_side_changes"]
        + summary["candidate_only_wins"]
        - summary["suppressed_wins"]
        - summary["incorrect_side_changes"]
        - summary["candidate_only_losses"]
    )
    if expected_score != summary["net_corrected_decisions"]:
        raise RuntimeError("net corrected decisions do not reconcile")
    expected_exact = float(challenger_ledger["realized_net"].sum()) - float(
        incumbent_ledger["realized_net"].sum()
    )
    if not math.isclose(summary["incremental_pnl_exact"], expected_exact, abs_tol=1e-9):
        raise RuntimeError("exact incremental PnL does not reconcile")
    return joined, summary


def paired_incumbent_economics(
    incumbent_ledger: pl.DataFrame,
    challenger_ledger: pl.DataFrame,
    *,
    eligible_markets: pl.DataFrame,
    resamples: int,
    seed: int,
    incumbent_correctness_margin: float = 0.026231,
) -> dict[str, Any]:
    """Apply the frozen post-seal economic and corrected-decision gates."""

    if resamples <= 0 or seed < 0:
        raise ValueError("paired economic bootstrap settings are invalid")
    if not np.isfinite(incumbent_correctness_margin):
        raise ValueError("incumbent correctness margin must be finite")
    universe = _eligible_market_universe(eligible_markets)
    correction, correction_summary = build_incumbent_correction_ledger(
        incumbent_ledger,
        challenger_ledger,
    )
    universe_keys = set(universe.iter_rows())
    correction_keys = set(correction.select("market_id", "window_start").iter_rows())
    if not correction_keys.issubset(universe_keys):
        raise ValueError("policy ledgers contain markets outside the eligible universe")
    incumbent_metrics = ledger_metrics(incumbent_ledger)
    challenger_metrics = ledger_metrics(challenger_ledger)
    eligible_count = universe.height
    incumbent_profit_per_market = incumbent_metrics["stress_1c_net_profit"] / eligible_count
    challenger_profit_per_market = challenger_metrics["stress_1c_net_profit"] / eligible_count
    paired_bootstrap = _paired_stressed_profit_bootstrap(
        correction,
        universe,
        resamples=resamples,
        seed=seed,
    )
    candidate_only = correction.filter(
        pl.col("correction_category").is_in(["candidate_only_win", "candidate_only_loss"])
    )
    candidate_only_stressed_pnl = float(candidate_only["challenger_stress_1c_net"].sum())
    candidate_only_positive = candidate_only.is_empty() or candidate_only_stressed_pnl > 0.0
    incumbent_avg_loss = abs(float(incumbent_metrics["average_loss"] or 0.0))
    candidate_avg_loss = abs(float(challenger_metrics["average_loss"] or 0.0))
    incumbent_max_loss = abs(float(incumbent_metrics["maximum_loss"] or 0.0))
    candidate_max_loss = abs(float(challenger_metrics["maximum_loss"] or 0.0))
    incumbent_drawdown = abs(float(incumbent_metrics["maximum_drawdown"] or 0.0))
    candidate_drawdown = abs(float(challenger_metrics["maximum_drawdown"] or 0.0))
    incumbent_margin = float(incumbent_metrics["selected_win_rate_advantage"])
    candidate_margin = (
        float(challenger_metrics["selected_win_rate_advantage"])
        if challenger_metrics["selected_win_rate_advantage"] is not None
        else None
    )
    incumbent_ev = float(incumbent_metrics["stress_1c_net_expectancy_per_trade"])
    candidate_ev = challenger_metrics["stress_1c_net_expectancy_per_trade"]
    ev_improves = candidate_ev is not None and candidate_ev > incumbent_ev
    profit_improves = challenger_profit_per_market > incumbent_profit_per_market
    profit_within_ten_percent = challenger_profit_per_market >= (
        incumbent_profit_per_market - 0.10 * abs(incumbent_profit_per_market)
    )
    ev_within_ten_percent = candidate_ev is not None and candidate_ev >= (
        incumbent_ev - 0.10 * abs(incumbent_ev)
    )
    gates = [
        _gate(
            "minimum_incumbent_frequency",
            challenger_metrics["trades"],
            math.ceil(0.80 * incumbent_metrics["trades"]),
            ">=",
        ),
        _gate("minimum_yes_entries", challenger_metrics.get("yes_trades", 0), 20, ">="),
        _gate("minimum_no_entries", challenger_metrics.get("no_trades", 0), 20, ">="),
        _gate("positive_stress_1c_expectancy", candidate_ev, 0.0, ">"),
        _profit_factor_gate(challenger_metrics, minimum=1.05),
        _gate(
            "maximum_mean_share_price",
            challenger_metrics["mean_share_price"],
            float(incumbent_metrics["mean_share_price"]) + 0.005,
            "<=",
        ),
        _gate("maximum_loss_recovery_burden", challenger_metrics["loss_recovery_wins"], 0.40, "<="),
        _gate(
            "maximum_average_loss_regression", candidate_avg_loss, 1.10 * incumbent_avg_loss, "<="
        ),
        _gate(
            "maximum_single_loss_regression", candidate_max_loss, 1.10 * incumbent_max_loss, "<="
        ),
        _gate("maximum_drawdown_regression", candidate_drawdown, 1.10 * incumbent_drawdown, "<="),
        _gate(
            "paired_stress_profit_per_eligible_market_lower_95",
            paired_bootstrap["challenger_minus_incumbent_stress_1c_profit_per_eligible_market"][
                "lower_95"
            ],
            0.0,
            ">=",
        ),
        _gate(
            "minimum_net_corrected_decisions",
            correction_summary["net_corrected_decisions"],
            2,
            ">=",
        ),
        _gate(
            "minimum_corrected_decision_utc_days",
            correction_summary["corrected_decision_improvement_utc_days"],
            3,
            ">=",
        ),
        _gate(
            "candidate_correctness_margin",
            candidate_margin,
            incumbent_correctness_margin,
            ">=",
        ),
        _gate(
            "candidate_minus_incumbent_correctness_margin",
            candidate_margin - incumbent_margin if candidate_margin is not None else None,
            0.0,
            ">=",
        ),
        _gate("candidate_only_positive_stressed_expectancy", candidate_only_positive, True, "=="),
        _gate(
            "expectancy_profit_tradeoff",
            bool(
                (ev_improves and profit_within_ten_percent)
                or (profit_improves and ev_within_ten_percent)
            ),
            True,
            "==",
        ),
    ]
    return {
        "schema_version": INCUMBENT_EVALUATION_SCHEMA_VERSION,
        "status": "qualified" if all(gate["passed"] for gate in gates) else "not_qualified",
        "eligible_markets": eligible_count,
        "incumbent": {
            "metrics": incumbent_metrics,
            "stress_1c_net_profit_per_eligible_market": incumbent_profit_per_market,
        },
        "challenger": {
            "metrics": challenger_metrics,
            "stress_1c_net_profit_per_eligible_market": challenger_profit_per_market,
        },
        "paired_bootstrap": paired_bootstrap,
        "correction_summary": correction_summary,
        "candidate_only_stress_1c_pnl": candidate_only_stressed_pnl,
        "gates": gates,
    }


def _reject_preseal_economics(frame: pl.DataFrame) -> None:
    forbidden = sorted(PROBABILITY_SELECTION_FORBIDDEN_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise ValueError(
            "probability selection cannot receive economic columns: " + ", ".join(forbidden)
        )


def _validate_policy_bounds(**values: float) -> None:
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("probability-only policy bounds must be finite")
    if not 0.0 <= values["minimum_share_price"] < values["maximum_share_price"] <= 1.0:
        raise ValueError("probability-only share-price bounds are invalid")
    if values["minimum_edge_per_share"] < 0.0:
        raise ValueError("probability-only minimum edge must be nonnegative")
    if values["maximum_seconds_elapsed"] < 1:
        raise ValueError("probability-only maximum second must be positive")
    if values["quantity"] <= 0.0 or values["maximum_depth_participation"] <= 0.0:
        raise ValueError("probability-only quantity and depth participation must be positive")
    if not 0.0 < values["maximum_admission_cost_per_share"] <= 1.0:
        raise ValueError("probability-only maximum admission cost is invalid")


def _single_identity_columns(frame: pl.DataFrame) -> list[str]:
    columns = [column for column in ("candidate_id", "model") if column in frame.columns]
    for column in columns:
        if frame[column].null_count() or frame[column].n_unique() != 1:
            raise ValueError(f"probability selection requires exactly one {column}")
    return columns


def _empty_probability_selection(
    source: pl.DataFrame,
    identity_columns: Sequence[str],
) -> pl.DataFrame:
    schema = {
        column: source.schema[column]
        for column in (*identity_columns, *PROBABILITY_SELECTION_OUTPUT_COLUMNS)
        if column in source.schema
    }
    schema.update(
        {
            "selected_yes": pl.Boolean,
            "selected_probability": pl.Float64,
            "selected_label": pl.Int64,
            "time_cell": pl.String,
        }
    )
    return pl.DataFrame(schema=schema).select(
        *identity_columns,
        *PROBABILITY_SELECTION_OUTPUT_COLUMNS,
    )


def _time_cell_expression(time_bands: Sequence[tuple[int, int]]) -> pl.Expr:
    expression: pl.Expr | None = None
    for lower, upper in time_bands:
        value = pl.when(pl.col("seconds_elapsed").is_between(lower, upper, closed="left")).then(
            pl.lit(f"{lower}_{upper}")
        )
        expression = (
            value
            if expression is None
            else expression.when(
                pl.col("seconds_elapsed").is_between(lower, upper, closed="left")
            ).then(pl.lit(f"{lower}_{upper}"))
        )
    if expression is None:
        raise ValueError("time cells must not be empty")
    return expression.otherwise(pl.lit(None, dtype=pl.String))


def _validate_time_bands(time_bands: Sequence[tuple[int, int]]) -> None:
    if not time_bands:
        raise ValueError("probability time bands must not be empty")
    prior_upper: int | None = None
    for lower, upper in time_bands:
        if lower >= upper or (prior_upper is not None and lower < prior_upper):
            raise ValueError("probability time bands must be ordered and non-overlapping")
        prior_upper = upper


def _validate_probability_values(frame: pl.DataFrame) -> None:
    if frame.is_empty():
        return
    labels = frame["label_up"].cast(pl.Float64, strict=False).to_numpy()
    probabilities = frame["probability_yes"].cast(pl.Float64, strict=False).to_numpy()
    seconds = frame["seconds_elapsed"].cast(pl.Float64, strict=False).to_numpy()
    if not np.all(np.isin(labels, (0.0, 1.0))):
        raise ValueError("probability labels must be binary")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("probabilities must be finite and inside [0, 1]")
    if not np.all(np.isfinite(seconds)) or np.any(seconds != np.floor(seconds)):
        raise ValueError("decision seconds must be finite integers")


def _side_probability_cohorts(
    frame: pl.DataFrame,
    probability: np.ndarray,
    labels: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if "selected_yes" in frame.columns:
        yes = frame["selected_yes"].to_numpy().astype(bool)
        return {
            "YES": (yes, probability[yes], labels[yes]),
            "NO": (~yes, 1.0 - probability[~yes], 1.0 - labels[~yes]),
        }
    if {"yes_target_eligible", "no_target_eligible"}.issubset(frame.columns):
        yes = frame["yes_target_eligible"].to_numpy().astype(bool)
        no = frame["no_target_eligible"].to_numpy().astype(bool)
        return {
            "YES": (yes, probability[yes], labels[yes]),
            "NO": (no, 1.0 - probability[no], 1.0 - labels[no]),
        }
    return {}


def _probability_metrics(
    probability: np.ndarray,
    labels: np.ndarray,
    market_ids: np.ndarray,
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if probability.size == 0:
        if not allow_empty:
            raise ValueError("probability metric cohort is empty")
        return {
            "rows": 0,
            "markets": 0,
            "brier": None,
            "log_loss": None,
            "bias": None,
            "ece": None,
            "mean_probability": None,
            "actual_rate": None,
        }
    _, inverse, counts = np.unique(market_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights /= weights.sum()
    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    brier = float(np.sum(weights * np.square(clipped - labels)))
    log_loss = float(np.sum(weights * _row_log_loss(labels, clipped)))
    bias = float(np.sum(weights * (clipped - labels)))
    bin_index = np.minimum((clipped * 10).astype(np.int16), 9)
    ece = 0.0
    for index in range(10):
        selected = bin_index == index
        if not selected.any():
            continue
        bin_weight = float(weights[selected].sum())
        bin_probability = float(np.sum(weights[selected] * clipped[selected]) / bin_weight)
        bin_rate = float(np.sum(weights[selected] * labels[selected]) / bin_weight)
        ece += bin_weight * abs(bin_probability - bin_rate)
    return {
        "rows": int(probability.size),
        "markets": int(np.unique(market_ids).size),
        "brier": brier,
        "log_loss": log_loss,
        "bias": bias,
        "ece": ece,
        "mean_probability": float(np.sum(weights * clipped)),
        "actual_rate": float(np.sum(weights * labels)),
    }


def _row_log_loss(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))


def _validate_matched_probability_frame(frame: pl.DataFrame, name: str) -> None:
    _require_columns(
        frame,
        {*PROBABILITY_KEY_COLUMNS, "label_up", "probability_yes"},
        f"{name} matched probability frame",
    )
    if frame.is_empty():
        raise ValueError(f"{name} matched probability frame is empty")
    _validate_probability_values(frame)
    if frame.select(*PROBABILITY_KEY_COLUMNS).is_duplicated().any():
        raise ValueError(f"{name} matched probability frame contains duplicate keys")


def _calibration_support_evidence(
    support: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cells = support.get("cells", support) if support is not None else {}
    cell_checks: list[dict[str, Any]] = []
    for name in REQUIRED_CALIBRATION_CELLS:
        cell = cells.get(name) if isinstance(cells, Mapping) else None
        fitted = bool(cell and cell.get("fitted") is True)
        converged = bool(cell and cell.get("converged", True) is True)
        no_fallback = bool(cell and cell.get("fallback", False) is False)
        negative, positive = _outcome_support(cell or {})
        cell_checks.append(
            {
                "cell": name,
                "fitted": fitted,
                "converged": converged,
                "fallback": not no_fallback,
                "outcome_0": negative,
                "outcome_1": positive,
                "both_outcomes": negative > 0 and positive > 0,
                "passed": bool(
                    fitted and converged and no_fallback and negative > 0 and positive > 0
                ),
            }
        )
    return {
        "required_cells": len(REQUIRED_CALIBRATION_CELLS),
        "cell_checks": cell_checks,
        "passed": all(item["passed"] for item in cell_checks),
    }


def _outcome_support(cell: Mapping[str, Any]) -> tuple[int, int]:
    counts = cell.get("outcome_counts") or cell.get("class_counts") or {}
    negative = cell.get("negative_outcomes", counts.get("0", counts.get(0, 0)))
    positive = cell.get("positive_outcomes", counts.get("1", counts.get(1, 0)))
    return int(negative or 0), int(positive or 0)


def _gate(name: str, observed: Any, threshold: Any, operator: str) -> dict[str, Any]:
    passed = False
    if observed is not None:
        if operator == "<=":
            passed = bool(observed <= threshold)
        elif operator == ">=":
            passed = bool(observed >= threshold)
        elif operator == ">":
            passed = bool(observed > threshold)
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


def _validate_policy_ledger(frame: pl.DataFrame, name: str, quantity: float) -> None:
    keys = ["market_id", "window_start"]
    if frame.select(*keys).is_duplicated().any():
        raise ValueError(f"{name} ledger contains duplicate markets")
    if frame.filter(pl.col("quantity") != quantity).height:
        raise ValueError(f"{name} ledger does not use exactly {quantity:g} shares")
    for column in ("selected_execution_cost_per_share", "realized_net"):
        values = frame[column].cast(pl.Float64, strict=False).to_numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} ledger contains non-finite {column}")
    if not set(frame["won"].unique().to_list()).issubset({False, True}):
        raise ValueError(f"{name} ledger contains non-binary outcomes")
    expected = frame["quantity"].to_numpy() * (
        frame["won"].cast(pl.Float64).to_numpy()
        - frame["selected_execution_cost_per_share"].to_numpy()
    )
    if not np.allclose(expected, frame["realized_net"].to_numpy(), atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} ledger realized PnL does not reconcile to execution")


def _prefix_ledger(
    frame: pl.DataFrame,
    prefix: str,
    keys: Sequence[str],
    *,
    stress_per_share: float,
) -> pl.DataFrame:
    selected = frame.select(
        *keys,
        "observed_at",
        "seconds_elapsed",
        "selected_yes",
        "won",
        "quantity",
        "selected_execution_cost_per_share",
        "realized_net",
    ).rename(
        {
            column: f"{prefix}_{column}"
            for column in (
                "observed_at",
                "seconds_elapsed",
                "selected_yes",
                "won",
                "quantity",
                "selected_execution_cost_per_share",
                "realized_net",
            )
        }
    )
    return selected.with_columns(
        pl.lit(True).alias(f"{prefix}_present"),
        (
            pl.col(f"{prefix}_quantity")
            * (
                pl.col(f"{prefix}_won").cast(pl.Float64)
                - (pl.col(f"{prefix}_selected_execution_cost_per_share") + stress_per_share).clip(
                    upper_bound=1.0
                )
            )
        ).alias(f"{prefix}_stress_1c_net"),
    )


def _eligible_market_universe(frame: pl.DataFrame) -> pl.DataFrame:
    _require_columns(frame, {"market_id", "window_start"}, "eligible market universe")
    universe = frame.select("market_id", "window_start").unique().sort("window_start", "market_id")
    if universe.is_empty():
        raise ValueError("eligible market universe is empty")
    return universe


def _paired_stressed_profit_bootstrap(
    correction: pl.DataFrame,
    universe: pl.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    daily_universe = (
        universe.with_columns(pl.col("window_start").dt.date().alias("utc_day"))
        .group_by("utc_day")
        .agg(pl.len().alias("eligible_markets"))
        .sort("utc_day")
    )
    if daily_universe.height < 2:
        raise ValueError("paired economic bootstrap requires at least two UTC days")
    daily_delta = (
        correction.with_columns(pl.col("window_start").dt.date().alias("utc_day"))
        .group_by("utc_day")
        .agg(pl.col("incremental_pnl_stress_1c").sum().alias("incremental_pnl"))
    )
    daily = daily_universe.join(daily_delta, on="utc_day", how="left").with_columns(
        pl.col("incremental_pnl").fill_null(0.0)
    )
    markets = daily["eligible_markets"].to_numpy()
    deltas = daily["incremental_pnl"].to_numpy()
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, daily.height, size=(resamples, daily.height))
    samples = deltas[chosen].sum(axis=1) / markets[chosen].sum(axis=1)
    return {
        "utc_day_blocks": daily.height,
        "point": float(deltas.sum() / markets.sum()),
        "challenger_minus_incumbent_stress_1c_profit_per_eligible_market": {
            "lower_95": float(np.quantile(samples, 0.025)),
            "median": float(np.quantile(samples, 0.5)),
            "upper_95": float(np.quantile(samples, 0.975)),
        },
    }


def _profit_factor_gate(
    metrics: Mapping[str, Any],
    *,
    minimum: float,
) -> dict[str, Any]:
    value = metrics.get("stress_1c_profit_factor")
    no_losses = bool(metrics.get("profit_factor_no_losses") and metrics.get("trades", 0))
    return {
        "name": "minimum_stress_1c_profit_factor",
        "observed": float(value) if value is not None else None,
        "no_losses": no_losses,
        "threshold": minimum,
        "operator": ">=",
        "passed": bool(no_losses or (value is not None and value >= minimum)),
    }


def _require_columns(frame: pl.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing columns: {', '.join(missing)}")
