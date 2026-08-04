"""Paired evaluation for the fixed spot-L2/Chainlink paper benchmark.

The functions in this module deliberately operate on market-level first
crossings.  A market without a first crossing is a real no-trade (zero PnL),
while a market with a first crossing but without qualified execution evidence
is unavailable and is excluded from a paired economic comparison.  Keeping
those states separate prevents missing books from being credited as avoided
losses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

from .core_evaluation import (
    FIRST_CROSSING_TIME_BANDS,
    classification_metrics,
)

POINT_KEYS = ("market_id", "window_start", "seconds_elapsed")
MARKET_KEYS = ("market_id", "window_start")
FIXED_BOOTSTRAP_RESAMPLES = 10_000
FIXED_EXECUTION_QUANTITY = 5.0
FIXED_CONFIDENCE_THRESHOLD = 0.89


@dataclass(frozen=True)
class AdvancementThresholds:
    """The immutable advancement contract from the benchmark specification."""

    minimum_net_expectancy_improvement: float = 0.0
    minimum_profit_factor_improvement: float = 0.0
    maximum_accuracy_regression: float = 0.01
    minimum_trade_coverage_ratio: float = 0.80
    minimum_win_retention_ratio: float = 0.80
    minimum_gross_loss_reduction: float = 0.10
    maximum_ece: float = 0.05
    minimum_consistent_days: int = 3

    @classmethod
    def fixed(cls, values: Mapping[str, float | int] | None = None) -> AdvancementThresholds:
        thresholds = cls()
        if values is None:
            return thresholds
        expected = asdict(thresholds)
        observed = {
            name: int(values[name]) if name == "minimum_consistent_days" else float(values[name])
            for name in expected
        }
        if observed != expected:
            raise ValueError("advancement thresholds are fixed by the benchmark contract")
        return thresholds


@dataclass(frozen=True)
class EconomicLedgerSpec:
    """Execution columns used to price one frozen paper scenario."""

    scenario_key: str
    quantity: float = FIXED_EXECUTION_QUANTITY
    up_price_column: str = "up_ask_vwap_10"
    down_price_column: str = "down_ask_vwap_10"
    eligibility_column: str = "strict_both_side_eligible_10"

    def __post_init__(self) -> None:
        if not self.scenario_key.strip():
            raise ValueError("scenario_key must be non-empty")
        if not math.isclose(self.quantity, FIXED_EXECUTION_QUANTITY):
            raise ValueError("the benchmark execution quantity is fixed at five shares")
        for name in (
            self.up_price_column,
            self.down_price_column,
            self.eligibility_column,
        ):
            if not name.strip():
                raise ValueError("execution column names must be non-empty")


def classification_summaries(
    rows: pl.DataFrame,
    *,
    eligible_rows: pl.DataFrame,
) -> dict[str, Any]:
    """Summarize scored decisions overall, by UTC day, and by policy band.

    ``eligible_rows`` is the matched cohort before confidence selection.  It
    may contain every five-second decision for each market.  Consequently the
    result reports decision-row coverage and unique-market coverage separately.
    """

    eligible = _eligible_decisions(eligible_rows)
    scored = _scored_rows(rows)
    outside = scored.select(*POINT_KEYS).join(
        eligible.select(*POINT_KEYS),
        on=list(POINT_KEYS),
        how="anti",
    )
    if outside.height:
        raise RuntimeError("scored predictions fall outside the matched eligible cohort")

    aggregate = _classification_bucket(scored, eligible)
    dates = (
        eligible.select(pl.col("window_start").dt.date().cast(pl.String).alias("date"))["date"]
        .unique()
        .sort()
        .to_list()
    )
    daily: list[dict[str, Any]] = []
    for date in dates:
        selected = scored.filter(pl.col("window_start").dt.date().cast(pl.String) == date)
        eligible_date = eligible.filter(pl.col("window_start").dt.date().cast(pl.String) == date)
        daily.append({"date": date, **_classification_bucket(selected, eligible_date)})

    time_bands = []
    for band, start, end in FIRST_CROSSING_TIME_BANDS:
        selected = scored.filter(
            (pl.col("seconds_elapsed") >= start) & (pl.col("seconds_elapsed") < end)
        )
        eligible_band = eligible.filter(
            (pl.col("seconds_elapsed") >= start) & (pl.col("seconds_elapsed") < end)
        )
        bucket = _classification_bucket(selected, eligible_band)
        bucket["market_coverage_of_full_cohort"] = (
            selected["market_id"].n_unique() / eligible["market_id"].n_unique()
            if eligible.height
            else 0.0
        )
        time_bands.append(
            {
                "band": band,
                "start_seconds": start,
                "end_seconds_exclusive": end,
                **bucket,
            }
        )
    return {
        "aggregate": aggregate,
        "daily": daily,
        "time_bands": time_bands,
    }


def build_economic_ledger(
    predictions: pl.DataFrame,
    execution: pl.DataFrame,
    *,
    eligible_markets: pl.DataFrame,
    profile: str,
    spec: EconomicLedgerSpec,
) -> pl.DataFrame:
    """Create one row per eligible market for one model and execution scenario.

    A first crossing is an actual signal.  Qualified execution requires the
    exact decision key, a causal/strict book row, a finite side-specific price,
    and a finite nonnegative fee rate.  Missing evidence on an actual signal
    leaves PnL null; absence of a signal is an intentional no-trade with PnL 0.
    """

    if not profile.strip():
        raise ValueError("profile must be non-empty")
    eligible = _eligible_market_rows(eligible_markets)
    signals = _first_crossing_signals(predictions)
    outside = signals.select(*MARKET_KEYS).join(
        eligible.select(*MARKET_KEYS),
        on=list(MARKET_KEYS),
        how="anti",
    )
    if outside.height:
        raise RuntimeError(f"{profile} signals fall outside the eligible market cohort")

    required_execution = (
        *POINT_KEYS,
        "scenario_key",
        "fee_rate",
        spec.up_price_column,
        spec.down_price_column,
        spec.eligibility_column,
    )
    _require_columns(execution, required_execution, "execution evidence")
    scenario = execution.filter(pl.col("scenario_key") == spec.scenario_key)
    if scenario.select(pl.struct(POINT_KEYS).n_unique()).item() != scenario.height:
        raise RuntimeError(f"execution scenario {spec.scenario_key} has duplicate point keys")
    metadata = _scenario_metadata(scenario, spec.scenario_key)
    optional_execution = tuple(
        name
        for name in (
            "decision_at",
            "snapshot_at",
            "configured_arrival_latency_ms",
            "realized_arrival_latency_ms",
            "visible_depth_fraction",
            "raw_quantity_required",
            "price_stress_method",
            "price_stress_exact",
        )
        if name in scenario.columns
    )
    scenario = scenario.select(
        *required_execution,
        *optional_execution,
    ).with_columns(pl.lit(True).alias("_execution_row_present"))

    signal_columns = (
        *POINT_KEYS,
        "observed_at",
        "label_up",
        "predicted_up",
        "probability_up",
        "confidence",
        "correct",
    )
    attached = signals.select(*signal_columns).join(
        scenario,
        on=list(POINT_KEYS),
        how="left",
        validate="1:1",
    )
    selected_price = (
        pl.when(pl.col("predicted_up").cast(pl.Boolean))
        .then(pl.col(spec.up_price_column))
        .otherwise(pl.col(spec.down_price_column))
    )
    evidence = (
        pl.col("_execution_row_present").fill_null(False)
        & pl.col(spec.eligibility_column).fill_null(False)
        & selected_price.is_not_null()
        & selected_price.is_finite()
        & selected_price.is_between(0.0, 1.0, closed="right")
        & pl.col("fee_rate").is_not_null()
        & pl.col("fee_rate").is_finite()
        & (pl.col("fee_rate") >= 0.0)
    )
    if "decision_at" in attached.columns:
        evidence = evidence & (pl.col("decision_at") == pl.col("observed_at"))
    attached = (
        attached.with_columns(
            selected_price.alias("execution_price"),
            evidence.alias("execution_evidence_available"),
        )
        .with_columns(
            (
                pl.col("fee_rate") * pl.col("execution_price") * (1.0 - pl.col("execution_price"))
            ).alias("fee_per_share"),
        )
        .with_columns(
            pl.when(pl.col("execution_evidence_available"))
            .then(
                spec.quantity
                * (
                    pl.col("correct").cast(pl.Float64)
                    - pl.col("execution_price")
                    - pl.col("fee_per_share")
                )
            )
            .otherwise(None)
            .alias("net_pnl")
        )
    )

    signal_payload = attached.select(
        *MARKET_KEYS,
        "seconds_elapsed",
        "observed_at",
        "label_up",
        "predicted_up",
        "probability_up",
        "confidence",
        "correct",
        "execution_price",
        "fee_rate",
        "fee_per_share",
        "execution_evidence_available",
        "net_pnl",
        *optional_execution,
    ).with_columns(pl.lit(True).alias("signal"))
    ledger = (
        eligible.join(
            signal_payload,
            on=list(MARKET_KEYS),
            how="left",
            validate="1:1",
            suffix="_signal",
        )
        .with_columns(
            pl.col("signal").fill_null(False),
            pl.col("execution_evidence_available").fill_null(False),
        )
        .with_columns(
            pl.when(~pl.col("signal"))
            .then(pl.lit(0.0))
            .otherwise(pl.col("net_pnl"))
            .alias("net_pnl"),
            (pl.col("signal") & pl.col("execution_evidence_available")).alias("trade"),
            (pl.col("signal") & ~pl.col("execution_evidence_available")).alias(
                "signal_missing_execution_evidence"
            ),
            pl.lit(profile).alias("profile"),
            pl.lit(spec.scenario_key).alias("scenario_key"),
            pl.lit(spec.quantity).alias("quantity"),
            pl.lit(metadata["price_stress_exact"]).alias("price_stress_exact"),
            pl.lit(metadata["price_stress_method"]).alias("price_stress_method"),
        )
    )
    return ledger.sort(["window_start", "market_id"])


def economic_ledger_metrics(
    ledger: pl.DataFrame,
    *,
    require_complete_signal_evidence: bool = True,
) -> dict[str, Any]:
    """Compute economic metrics without converting unavailable evidence to PnL 0."""

    _validate_ledger(ledger)
    excluded = ledger.filter(pl.col("signal_missing_execution_evidence"))
    if require_complete_signal_evidence:
        eligible = ledger.filter(~pl.col("signal_missing_execution_evidence"))
    else:
        eligible = ledger
    return _economic_metrics_from_columns(
        eligible,
        pnl_column="net_pnl",
        trade_column="trade",
        correct_column="correct",
        signal_column="signal",
        source_markets=ledger.height,
        missing_evidence_markets=excluded.height,
    )


def pair_economic_ledgers(
    candidate: pl.DataFrame,
    control: pl.DataFrame,
) -> pl.DataFrame:
    """Pair ledgers on eligible markets and exclude only unavailable actual signals."""

    _validate_ledger(candidate)
    _validate_ledger(control)
    candidate_scenarios = candidate["scenario_key"].unique().to_list()
    control_scenarios = control["scenario_key"].unique().to_list()
    if candidate_scenarios != control_scenarios or len(candidate_scenarios) != 1:
        raise RuntimeError("candidate and control ledgers must use one identical scenario")
    left_keys = candidate.select(*MARKET_KEYS).sort(list(MARKET_KEYS))
    right_keys = control.select(*MARKET_KEYS).sort(list(MARKET_KEYS))
    if not left_keys.equals(right_keys, null_equal=True):
        raise RuntimeError("candidate and control eligible market cohorts differ")

    selected = (
        "profile",
        "signal",
        "trade",
        "signal_missing_execution_evidence",
        "execution_evidence_available",
        "net_pnl",
        "correct",
        "predicted_up",
        "seconds_elapsed",
        "execution_price",
        "fee_rate",
        "price_stress_exact",
        "price_stress_method",
    )
    paired = candidate.select(*MARKET_KEYS, *selected).join(
        control.select(*MARKET_KEYS, *selected),
        on=list(MARKET_KEYS),
        how="inner",
        validate="1:1",
        suffix="_control",
    )
    rename = {
        name: f"candidate_{name}"
        for name in selected
        if name in paired.columns and f"{name}_control" in paired.columns
    }
    rename.update(
        {
            f"{name}_control": f"control_{name}"
            for name in selected
            if f"{name}_control" in paired.columns
        }
    )
    paired = paired.rename(rename)
    return (
        paired.with_columns(
            pl.lit(candidate_scenarios[0]).alias("scenario_key"),
            (
                ~pl.col("candidate_signal_missing_execution_evidence")
                & ~pl.col("control_signal_missing_execution_evidence")
            ).alias("paired_execution_evidence_eligible"),
        )
        .with_columns(
            pl.when(pl.col("paired_execution_evidence_eligible"))
            .then(pl.col("candidate_net_pnl") - pl.col("control_net_pnl"))
            .otherwise(None)
            .alias("net_pnl_delta")
        )
        .sort(["window_start", "market_id"])
    )


def paired_economic_metrics(paired: pl.DataFrame) -> dict[str, Any]:
    """Summarize candidate/control economics on the exact executable pair."""

    _validate_paired(paired)
    eligible = paired.filter(pl.col("paired_execution_evidence_eligible"))
    candidate = _economic_metrics_from_columns(
        eligible,
        pnl_column="candidate_net_pnl",
        trade_column="candidate_trade",
        correct_column="candidate_correct",
        signal_column="candidate_signal",
        source_markets=paired.height,
        missing_evidence_markets=(paired.height - eligible.height),
    )
    control = _economic_metrics_from_columns(
        eligible,
        pnl_column="control_net_pnl",
        trade_column="control_trade",
        correct_column="control_correct",
        signal_column="control_signal",
        source_markets=paired.height,
        missing_evidence_markets=(paired.height - eligible.height),
    )
    control_wins = eligible.filter(pl.col("control_trade") & (pl.col("control_net_pnl") > 0))
    retained_wins = control_wins.filter(
        pl.col("candidate_trade") & (pl.col("candidate_net_pnl") > 0)
    )
    control_losses = eligible.filter(pl.col("control_trade") & (pl.col("control_net_pnl") < 0))
    avoided_losses = control_losses.filter(pl.col("candidate_net_pnl") >= 0)
    daily = _paired_daily_metrics(eligible)

    profit_factor_improvement = _profit_factor_improvement(candidate, control)
    gross_loss_reduction = _relative_reduction(control["gross_loss"], candidate["gross_loss"])
    coverage_ratio = _safe_ratio(candidate["trades"], control["trades"])
    win_retention = _safe_ratio(retained_wins.height, control_wins.height)
    loss_avoidance = _safe_ratio(avoided_losses.height, control_losses.height)
    scenario_exact = _single_bool_pair(
        paired,
        "candidate_price_stress_exact",
        "control_price_stress_exact",
    )
    scenario_methods = sorted(
        set(paired["candidate_price_stress_method"].drop_nulls().to_list())
        | set(paired["control_price_stress_method"].drop_nulls().to_list())
    )
    signal_execution_evidence = {
        role: _actual_signal_execution_evidence(paired, role) for role in ("candidate", "control")
    }
    return {
        "source_eligible_markets": paired.height,
        "paired_execution_eligible_markets": eligible.height,
        "excluded_for_actual_signal_execution_evidence": paired.height - eligible.height,
        "pair_coverage": eligible.height / paired.height if paired.height else 0.0,
        "execution_scenario": {
            "scenario_key": _single_value(paired, "scenario_key", default=None),
            "price_stress_exact": scenario_exact,
            "price_stress_methods": scenario_methods,
            "exactly_reproducible": bool(scenario_exact),
        },
        "actual_signal_execution_evidence": signal_execution_evidence,
        "candidate": candidate,
        "control": control,
        "paired_outcomes": {
            "control_wins": control_wins.height,
            "challenger_wins_retained": retained_wins.height,
            "challenger_win_retention_ratio": win_retention,
            "control_losses": control_losses.height,
            "control_losses_avoided": avoided_losses.height,
            "control_loss_avoidance_ratio": loss_avoidance,
            "gross_control_loss_avoided": float((-avoided_losses["control_net_pnl"]).sum())
            if avoided_losses.height
            else 0.0,
        },
        "deltas": {
            "total_net_pnl": candidate["total_net_pnl"] - control["total_net_pnl"],
            "net_expectancy_per_eligible_market": (
                candidate["net_expectancy_per_eligible_market"]
                - control["net_expectancy_per_eligible_market"]
            ),
            "profit_factor_improvement": _json_number(profit_factor_improvement),
            "trade_coverage_ratio": coverage_ratio,
            "gross_loss_reduction": gross_loss_reduction,
            "worst_trade": _optional_delta(candidate["worst_trade"], control["worst_trade"]),
            "worst_one_percent_tail": _optional_delta(
                candidate["worst_one_percent_tail"],
                control["worst_one_percent_tail"],
            ),
            "consistent_nonnegative_days": sum(
                row["has_economic_signal"] and row["net_pnl_delta"] >= 0.0 for row in daily
            ),
            "negative_days": sum(
                row["has_economic_signal"] and row["net_pnl_delta"] < 0.0 for row in daily
            ),
        },
        "daily": daily,
    }


def paired_day_block_bootstrap(
    paired: pl.DataFrame,
    *,
    random_seed: int,
    resamples: int = FIXED_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Bootstrap paired market outcomes by resampling whole UTC days.

    Each sampled day contributes all of its markets.  The fixed count is
    enforced so callers cannot silently weaken or search the uncertainty
    analysis.
    """

    if resamples != FIXED_BOOTSTRAP_RESAMPLES:
        raise ValueError("the paired day-block bootstrap requires exactly 10000 resamples")
    _validate_paired(paired)
    eligible = paired.filter(pl.col("paired_execution_evidence_eligible"))
    if eligible.is_empty():
        return {
            "sampling_unit": "whole_utc_day",
            "resamples": resamples,
            "random_seed": random_seed,
            "days": 0,
            "markets": 0,
            "metrics": {},
        }
    blocks = _bootstrap_day_blocks(eligible)
    sample_count = blocks.height
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, sample_count, size=(resamples, sample_count))

    def sampled_sum(column: str) -> np.ndarray:
        values = blocks[column].to_numpy().astype(np.float64)
        return values[indices].sum(axis=1)

    markets = sampled_sum("markets")
    candidate_pnl = sampled_sum("candidate_pnl")
    control_pnl = sampled_sum("control_pnl")
    candidate_gross_profit = sampled_sum("candidate_gross_profit")
    candidate_gross_loss = sampled_sum("candidate_gross_loss")
    control_gross_profit = sampled_sum("control_gross_profit")
    control_gross_loss = sampled_sum("control_gross_loss")
    candidate_trades = sampled_sum("candidate_trades")
    control_trades = sampled_sum("control_trades")
    control_wins = sampled_sum("control_wins")
    retained_wins = sampled_sum("retained_wins")
    control_losses = sampled_sum("control_losses")
    avoided_losses = sampled_sum("avoided_losses")

    with np.errstate(divide="ignore", invalid="ignore"):
        expectancy_delta = (candidate_pnl - control_pnl) / markets
        candidate_pf = candidate_gross_profit / candidate_gross_loss
        control_pf = control_gross_profit / control_gross_loss
        profit_factor_delta = candidate_pf - control_pf
        coverage_ratio = candidate_trades / control_trades
        gross_loss_reduction = (control_gross_loss - candidate_gross_loss) / control_gross_loss
        win_retention = retained_wins / control_wins
        loss_avoidance = avoided_losses / control_losses

    observed = paired_economic_metrics(paired)
    return {
        "sampling_unit": "whole_utc_day",
        "resamples": resamples,
        "random_seed": random_seed,
        "days": blocks.height,
        "markets": eligible.height,
        "metrics": {
            "net_expectancy_improvement_per_eligible_market": _bootstrap_interval(
                expectancy_delta,
                observed["deltas"]["net_expectancy_per_eligible_market"],
            ),
            "profit_factor_improvement": _bootstrap_interval(
                profit_factor_delta,
                observed["deltas"]["profit_factor_improvement"],
            ),
            "trade_coverage_ratio": _bootstrap_interval(
                coverage_ratio,
                observed["deltas"]["trade_coverage_ratio"],
            ),
            "gross_loss_reduction": _bootstrap_interval(
                gross_loss_reduction,
                observed["deltas"]["gross_loss_reduction"],
            ),
            "win_retention_ratio": _bootstrap_interval(
                win_retention,
                observed["paired_outcomes"]["challenger_win_retention_ratio"],
            ),
            "control_loss_avoidance_ratio": _bootstrap_interval(
                loss_avoidance,
                observed["paired_outcomes"]["control_loss_avoidance_ratio"],
            ),
        },
    }


def evaluate_advancement_gates(
    *,
    candidate_classification: Mapping[str, Any],
    control_classification: Mapping[str, Any],
    paired_economics: Mapping[str, Any],
    stress_scenarios: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, float | int] | None = None,
) -> dict[str, Any]:
    """Evaluate every fixed advancement gate with explicit observed evidence."""

    fixed = AdvancementThresholds.fixed(thresholds)
    candidate_metrics = _aggregate_classification(candidate_classification)
    control_metrics = _aggregate_classification(control_classification)
    candidate_economic = paired_economics["candidate"]
    control_economic = paired_economics["control"]
    deltas = paired_economics["deltas"]
    outcomes = paired_economics["paired_outcomes"]

    checks = [
        _gate_check(
            "net expectancy improvement is nonnegative",
            deltas.get("net_expectancy_per_eligible_market"),
            fixed.minimum_net_expectancy_improvement,
            ">=",
        ),
        _gate_check(
            "profit-factor improvement is nonnegative",
            deltas.get("profit_factor_improvement"),
            fixed.minimum_profit_factor_improvement,
            ">=",
        ),
        _gate_check(
            "accuracy regression is at most one percentage point",
            candidate_metrics.get("accuracy"),
            _optional_number(control_metrics.get("accuracy"), -math.inf)
            - fixed.maximum_accuracy_regression,
            ">=",
        ),
        _gate_check(
            "matched-control trade coverage is retained",
            deltas.get("trade_coverage_ratio"),
            fixed.minimum_trade_coverage_ratio,
            ">=",
        ),
        _gate_check(
            "matched-control wins are retained",
            outcomes.get("challenger_win_retention_ratio"),
            fixed.minimum_win_retention_ratio,
            ">=",
        ),
        _gate_check(
            "gross losses are reduced",
            deltas.get("gross_loss_reduction"),
            fixed.minimum_gross_loss_reduction,
            ">=",
        ),
        _gate_check(
            "expected calibration error is acceptable",
            candidate_metrics.get("expected_calibration_error"),
            fixed.maximum_ece,
            "<=",
        ),
        _gate_check(
            "worst trade does not worsen",
            candidate_economic.get("worst_trade"),
            control_economic.get("worst_trade"),
            ">=",
        ),
        _gate_check(
            "worst one-percent tail does not worsen",
            candidate_economic.get("worst_one_percent_tail"),
            control_economic.get("worst_one_percent_tail"),
            ">=",
        ),
        _gate_check(
            "evidence is consistent across at least three UTC days",
            deltas.get("consistent_nonnegative_days"),
            fixed.minimum_consistent_days,
            ">=",
        ),
    ]

    stress_checks: list[dict[str, Any]] = [
        _gate_check(
            "all three prescribed execution scenarios are present",
            len(stress_scenarios),
            3,
            ">=",
            hard_failure=True,
        )
    ]
    for scenario_key in sorted(stress_scenarios):
        evidence = _stress_metrics(stress_scenarios[scenario_key])
        reproducible = evidence["execution_scenario"].get("exactly_reproducible")
        signal_execution = evidence.get("actual_signal_execution_evidence", {})
        for role in ("candidate", "control"):
            role_evidence = signal_execution.get(role, {})
            signal_markets = role_evidence.get("signal_markets")
            paired_executable = role_evidence.get("paired_executable_signal_markets")
            has_evidence = (
                signal_markets is not None
                and paired_executable is not None
                and (int(signal_markets) == 0 or int(paired_executable) > 0)
            )
            stress_checks.append(
                _gate_check(
                    f"{scenario_key} has paired execution evidence for {role} signals",
                    has_evidence,
                    True,
                    "is",
                    hard_failure=True,
                    evidence=role_evidence,
                )
            )
        stress_checks.append(
            _gate_check(
                f"{scenario_key} is exactly reproducible",
                reproducible,
                True,
                "is",
                hard_failure=True,
                evidence={
                    "price_stress_exact": evidence["execution_scenario"].get("price_stress_exact"),
                    "price_stress_methods": evidence["execution_scenario"].get(
                        "price_stress_methods"
                    ),
                },
            )
        )
        stress_checks.append(
            _gate_check(
                f"{scenario_key} challenger expectancy remains nonnegative",
                evidence["candidate"].get("net_expectancy_per_eligible_market"),
                0.0,
                ">=",
                evidence={"proxy_only": reproducible is not True},
            )
        )
    checks.extend(stress_checks)
    hard_failures = [
        check["name"] for check in checks if check.get("hard_failure") and not check["passed"]
    ]
    passed = all(check["passed"] for check in checks)
    return {
        "paper_only": True,
        "eligible_for_forward_paper_validation": passed,
        "passed": passed,
        "thresholds": asdict(fixed),
        "checks": checks,
        "hard_failures": hard_failures,
        "stress_gate_hard_failed": bool(hard_failures),
    }


def _actual_signal_execution_evidence(
    paired: pl.DataFrame,
    role: str,
) -> dict[str, Any]:
    signal_column = f"{role}_signal"
    trade_column = f"{role}_trade"
    missing_column = f"{role}_signal_missing_execution_evidence"
    _require_columns(
        paired,
        (signal_column, trade_column, missing_column),
        f"{role} paired execution evidence",
    )
    signals = paired.filter(pl.col(signal_column))
    paired_executable = signals.filter(pl.col("paired_execution_evidence_eligible"))
    return {
        "signal_markets": signals.height,
        "own_executable_signal_markets": paired.filter(pl.col(trade_column)).height,
        "paired_executable_signal_markets": paired_executable.height,
        "own_missing_execution_evidence_markets": paired.filter(pl.col(missing_column)).height,
        "paired_signal_execution_coverage": (
            paired_executable.height / signals.height if signals.height else None
        ),
    }


def _classification_bucket(
    rows: pl.DataFrame,
    eligible: pl.DataFrame,
) -> dict[str, Any]:
    eligible_markets = eligible["market_id"].n_unique() if eligible.height else 0
    predicted_markets = rows["market_id"].n_unique() if rows.height else 0
    metrics = classification_metrics(rows, eligible_markets=eligible_markets)
    market_coverage = predicted_markets / eligible_markets if eligible_markets else 0.0
    metrics["coverage"] = market_coverage
    return {
        "prediction_rows": rows.height,
        "eligible_prediction_rows": eligible.height,
        "prediction_coverage": rows.height / eligible.height if eligible.height else 0.0,
        "predicted_markets": predicted_markets,
        "eligible_markets": eligible_markets,
        "market_coverage": market_coverage,
        "metrics": metrics,
    }


def _scored_rows(rows: pl.DataFrame) -> pl.DataFrame:
    required = (
        *POINT_KEYS,
        "observed_at",
        "label_up",
        "predicted_up",
        "probability_up",
    )
    _require_columns(rows, required, "scored predictions")
    scored = rows
    if "correct" not in scored.columns:
        scored = scored.with_columns(
            (pl.col("predicted_up").cast(pl.Int8) == pl.col("label_up").cast(pl.Int8)).alias(
                "correct"
            )
        )
    if scored.select(pl.struct(POINT_KEYS).n_unique()).item() != scored.height:
        raise RuntimeError("scored predictions contain duplicate point keys")
    invalid = scored.filter(
        pl.any_horizontal(
            pl.col("market_id").is_null(),
            pl.col("window_start").is_null(),
            pl.col("seconds_elapsed").is_null(),
            pl.col("observed_at").is_null(),
            pl.col("label_up").is_null(),
            pl.col("predicted_up").is_null(),
            pl.col("probability_up").is_null(),
            pl.col("correct").is_null(),
        )
        | ~pl.col("label_up").cast(pl.Int8).is_in([0, 1])
        | ~pl.col("predicted_up").cast(pl.Int8).is_in([0, 1])
        | (pl.col("predicted_up").cast(pl.Int8) != (pl.col("probability_up") >= 0.5).cast(pl.Int8))
        | (
            pl.col("correct")
            != (pl.col("predicted_up").cast(pl.Int8) == pl.col("label_up").cast(pl.Int8))
        )
        | ~pl.col("probability_up").is_finite()
        | ~pl.col("probability_up").is_between(0.0, 1.0, closed="both")
        | ~pl.col("seconds_elapsed").is_between(60, 240, closed="both")
        | (pl.col("seconds_elapsed") % 5 != 0)
    )
    if invalid.height:
        raise RuntimeError("scored predictions contain invalid labels or probabilities")
    return scored.sort(["window_start", "market_id", "seconds_elapsed"])


def _eligible_decisions(rows: pl.DataFrame) -> pl.DataFrame:
    _require_columns(rows, (*POINT_KEYS, "observed_at"), "eligible decisions")
    eligible = rows.select(*POINT_KEYS, "observed_at").unique(maintain_order=False)
    if eligible.height != rows.select(*POINT_KEYS).unique().height:
        raise RuntimeError("eligible decisions map one key to multiple timestamps")
    return eligible.sort(["window_start", "market_id", "seconds_elapsed"])


def _eligible_market_rows(rows: pl.DataFrame) -> pl.DataFrame:
    _require_columns(rows, MARKET_KEYS, "eligible markets")
    columns = [*MARKET_KEYS]
    if "label_up" in rows.columns:
        columns.append("label_up")
    eligible = rows.select(columns).unique(maintain_order=False)
    if eligible["market_id"].n_unique() != eligible.height:
        raise RuntimeError("one market_id maps to multiple eligible-market records")
    return eligible.sort(["window_start", "market_id"])


def _first_crossing_signals(rows: pl.DataFrame) -> pl.DataFrame:
    scored = _scored_rows(rows)
    if scored["market_id"].n_unique() != scored.height:
        raise RuntimeError(
            "economic predictions must contain at most one first crossing per market"
        )
    if "confidence" not in scored.columns:
        scored = scored.with_columns(
            pl.max_horizontal("probability_up", 1.0 - pl.col("probability_up")).alias("confidence")
        )
    invalid = scored.filter(
        pl.col("confidence").is_null()
        | ~pl.col("confidence").is_finite()
        | (pl.col("confidence") < FIXED_CONFIDENCE_THRESHOLD)
        | (
            (
                pl.col("confidence")
                - pl.max_horizontal("probability_up", 1.0 - pl.col("probability_up"))
            ).abs()
            > 1e-12
        )
    )
    if invalid.height:
        raise RuntimeError("economic signals violate the fixed first-crossing threshold")
    return scored


def _scenario_metadata(frame: pl.DataFrame, scenario_key: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "price_stress_exact": False,
        "price_stress_method": "unavailable",
    }
    for name in ("price_stress_exact", "price_stress_method"):
        if name not in frame.columns or frame.is_empty():
            continue
        values = frame[name].drop_nulls().unique().to_list()
        if len(values) != 1:
            raise RuntimeError(f"execution scenario {scenario_key} has inconsistent {name}")
        metadata[name] = values[0]
    return metadata


def _economic_metrics_from_columns(
    rows: pl.DataFrame,
    *,
    pnl_column: str,
    trade_column: str,
    correct_column: str,
    signal_column: str,
    source_markets: int,
    missing_evidence_markets: int,
) -> dict[str, Any]:
    trades = rows.filter(pl.col(trade_column))
    pnl = trades[pnl_column].to_numpy().astype(np.float64) if trades.height else np.array([])
    if len(pnl) and not np.isfinite(pnl).all():
        raise RuntimeError("qualified economic trades contain non-finite PnL")
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    tail_count = max(1, math.ceil(len(pnl) * 0.01)) if len(pnl) else 0
    tail = np.sort(pnl)[:tail_count] if tail_count else np.array([])
    net_pnl = float(pnl.sum()) if len(pnl) else 0.0
    return {
        "source_eligible_markets": source_markets,
        "eligible_markets": rows.height,
        "missing_actual_signal_execution_evidence": missing_evidence_markets,
        "signals": int(rows[signal_column].sum()) if rows.height else 0,
        "trades": trades.height,
        "trade_coverage": trades.height / rows.height if rows.height else 0.0,
        "wins": len(wins),
        "losses": len(losses),
        "accuracy": float(trades[correct_column].mean()) if trades.height else None,
        "total_net_pnl": net_pnl,
        "net_expectancy_per_trade": float(pnl.mean()) if len(pnl) else None,
        "net_expectancy_per_eligible_market": net_pnl / rows.height if rows.height else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else None,
        "worst_trade": float(pnl.min()) if len(pnl) else None,
        "worst_one_percent_tail": float(tail.mean()) if len(tail) else None,
    }


def _paired_daily_metrics(rows: pl.DataFrame) -> list[dict[str, Any]]:
    if rows.is_empty():
        return []
    daily = (
        rows.with_columns(pl.col("window_start").dt.date().cast(pl.String).alias("date"))
        .group_by("date")
        .agg(
            pl.len().alias("eligible_markets"),
            pl.col("candidate_trade").sum().alias("candidate_trades"),
            pl.col("control_trade").sum().alias("control_trades"),
            pl.col("candidate_net_pnl").sum().alias("candidate_net_pnl"),
            pl.col("control_net_pnl").sum().alias("control_net_pnl"),
        )
        .with_columns(
            (pl.col("candidate_net_pnl") - pl.col("control_net_pnl")).alias("net_pnl_delta"),
            ((pl.col("candidate_trades") + pl.col("control_trades")) > 0).alias(
                "has_economic_signal"
            ),
        )
        .sort("date")
    )
    return daily.to_dicts()


def _bootstrap_day_blocks(rows: pl.DataFrame) -> pl.DataFrame:
    return (
        rows.with_columns(
            pl.col("window_start").dt.date().cast(pl.String).alias("date"),
            pl.when(pl.col("candidate_trade") & (pl.col("candidate_net_pnl") > 0))
            .then(pl.col("candidate_net_pnl"))
            .otherwise(0.0)
            .alias("candidate_profit"),
            pl.when(pl.col("candidate_trade") & (pl.col("candidate_net_pnl") < 0))
            .then(-pl.col("candidate_net_pnl"))
            .otherwise(0.0)
            .alias("candidate_loss"),
            pl.when(pl.col("control_trade") & (pl.col("control_net_pnl") > 0))
            .then(pl.col("control_net_pnl"))
            .otherwise(0.0)
            .alias("control_profit"),
            pl.when(pl.col("control_trade") & (pl.col("control_net_pnl") < 0))
            .then(-pl.col("control_net_pnl"))
            .otherwise(0.0)
            .alias("control_loss"),
            (pl.col("control_trade") & (pl.col("control_net_pnl") > 0)).alias("control_win"),
            (
                pl.col("control_trade")
                & (pl.col("control_net_pnl") > 0)
                & pl.col("candidate_trade")
                & (pl.col("candidate_net_pnl") > 0)
            ).alias("retained_win"),
            (pl.col("control_trade") & (pl.col("control_net_pnl") < 0)).alias("control_loss_trade"),
            (
                pl.col("control_trade")
                & (pl.col("control_net_pnl") < 0)
                & (pl.col("candidate_net_pnl") >= 0)
            ).alias("avoided_loss"),
        )
        .group_by("date")
        .agg(
            pl.len().alias("markets"),
            pl.col("candidate_net_pnl").sum().alias("candidate_pnl"),
            pl.col("control_net_pnl").sum().alias("control_pnl"),
            pl.col("candidate_profit").sum().alias("candidate_gross_profit"),
            pl.col("candidate_loss").sum().alias("candidate_gross_loss"),
            pl.col("control_profit").sum().alias("control_gross_profit"),
            pl.col("control_loss").sum().alias("control_gross_loss"),
            pl.col("candidate_trade").sum().alias("candidate_trades"),
            pl.col("control_trade").sum().alias("control_trades"),
            pl.col("control_win").sum().alias("control_wins"),
            pl.col("retained_win").sum().alias("retained_wins"),
            pl.col("control_loss_trade").sum().alias("control_losses"),
            pl.col("avoided_loss").sum().alias("avoided_losses"),
        )
        .sort("date")
    )


def _bootstrap_interval(distribution: np.ndarray, observed: Any) -> dict[str, Any]:
    finite = distribution[np.isfinite(distribution)]
    return {
        "observed": _json_number(observed),
        "finite_resamples": len(finite),
        "lower_95": float(np.quantile(finite, 0.025)) if len(finite) else None,
        "upper_95": float(np.quantile(finite, 0.975)) if len(finite) else None,
        "bootstrap_mean": float(finite.mean()) if len(finite) else None,
    }


def _validate_ledger(ledger: pl.DataFrame) -> None:
    required = (
        *MARKET_KEYS,
        "profile",
        "scenario_key",
        "signal",
        "trade",
        "signal_missing_execution_evidence",
        "execution_evidence_available",
        "net_pnl",
        "correct",
        "predicted_up",
        "seconds_elapsed",
        "execution_price",
        "fee_rate",
        "price_stress_exact",
        "price_stress_method",
    )
    _require_columns(ledger, required, "economic ledger")
    if ledger["market_id"].n_unique() != ledger.height:
        raise RuntimeError("economic ledger must contain exactly one row per market")
    invalid = ledger.filter(
        (~pl.col("signal") & (pl.col("net_pnl") != 0.0))
        | (pl.col("trade") & ~pl.col("execution_evidence_available"))
        | (pl.col("signal_missing_execution_evidence") & pl.col("net_pnl").is_not_null())
    )
    if invalid.height:
        raise RuntimeError("economic ledger conflates no-trade and unavailable evidence")


def _validate_paired(paired: pl.DataFrame) -> None:
    required = (
        *MARKET_KEYS,
        "paired_execution_evidence_eligible",
        "candidate_signal",
        "candidate_trade",
        "candidate_net_pnl",
        "candidate_correct",
        "candidate_price_stress_exact",
        "candidate_price_stress_method",
        "control_signal",
        "control_trade",
        "control_net_pnl",
        "control_correct",
        "control_price_stress_exact",
        "control_price_stress_method",
    )
    _require_columns(paired, required, "paired economic ledger")
    if paired["market_id"].n_unique() != paired.height:
        raise RuntimeError("paired economics must contain one row per market")


def _aggregate_classification(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    aggregate: Any = summary.get("aggregate", summary)
    if isinstance(aggregate, Mapping) and isinstance(aggregate.get("metrics"), Mapping):
        aggregate = aggregate["metrics"]
    if not isinstance(aggregate, Mapping):
        raise TypeError("classification summary does not contain aggregate metrics")
    return aggregate


def _stress_metrics(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics: Any = summary.get("metrics", summary)
    if not isinstance(metrics, Mapping):
        raise TypeError("stress scenario does not contain paired economic metrics")
    for key in ("execution_scenario", "candidate"):
        if key not in metrics:
            raise ValueError(f"stress scenario is missing {key}")
    return metrics


def _gate_check(
    name: str,
    observed: Any,
    required: Any,
    operator: str,
    *,
    hard_failure: bool = False,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    valid = observed is not None and required is not None
    if operator == "is":
        passed = valid and observed is required
    elif operator == ">=":
        passed = valid and float(observed) >= float(required)
    elif operator == "<=":
        passed = valid and float(observed) <= float(required)
    else:
        raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "name": name,
        "observed": _json_number(observed),
        "required": _json_number(required),
        "operator": operator,
        "passed": bool(passed),
        "hard_failure": hard_failure,
        "evidence": dict(evidence or {}),
    }


def _profit_factor_improvement(
    candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> float | None:
    candidate_pf = _comparable_profit_factor(candidate)
    control_pf = _comparable_profit_factor(control)
    if candidate_pf is None or control_pf is None:
        return None
    if math.isinf(candidate_pf) and math.isinf(control_pf):
        return 0.0
    return candidate_pf - control_pf


def _comparable_profit_factor(metrics: Mapping[str, Any]) -> float | None:
    value = metrics.get("profit_factor")
    if value is not None:
        return float(value)
    gross_profit = float(metrics.get("gross_profit", 0.0))
    gross_loss = float(metrics.get("gross_loss", 0.0))
    trades = int(metrics.get("trades", 0))
    if trades and gross_profit > 0.0 and gross_loss == 0.0:
        return math.inf
    return None


def _relative_reduction(control: Any, candidate: Any) -> float | None:
    if control is None or candidate is None or float(control) <= 0.0:
        return None
    return (float(control) - float(candidate)) / float(control)


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if denominator is None or float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _optional_delta(candidate: Any, control: Any) -> float | None:
    if candidate is None or control is None:
        return None
    return float(candidate) - float(control)


def _optional_number(value: Any, default: float) -> float:
    return float(value) if value is not None else default


def _single_bool_pair(frame: pl.DataFrame, left: str, right: str) -> bool:
    values = set(frame[left].drop_nulls().unique().to_list()) | set(
        frame[right].drop_nulls().unique().to_list()
    )
    return len(values) == 1 and values == {True}


def _single_value(frame: pl.DataFrame, column: str, *, default: Any) -> Any:
    if column not in frame.columns or frame.is_empty():
        return default
    values = frame[column].drop_nulls().unique().to_list()
    return values[0] if len(values) == 1 else default


def _json_number(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return None
        if math.isinf(float(value)):
            return "inf" if float(value) > 0 else "-inf"
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], role: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{role} is missing columns: {', '.join(missing)}")
