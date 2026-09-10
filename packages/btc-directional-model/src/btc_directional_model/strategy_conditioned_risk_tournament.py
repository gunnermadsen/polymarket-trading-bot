"""Evaluate universal candidate-risk models through six frozen trading strategies."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .candidate_loss_risk_tournament import (
    ECONOMIC_FEATURES,
    HISTORY_FEATURES,
    MARKET_FEATURES,
    _history,
    _models,
    _predictive,
    _sha256,
    _write_json,
    build_policy_panel,
    sequential_replay,
    trade_metrics,
)
from .runtime_export import reached_leaf_value

SCHEMA_VERSION = "strategy_conditioned_candidate_risk_tournament_v1"
BUCKETS = ("15_59", "60_89", "90_119", "120_149", "150_179", "180_240")
ALL_FEATURES = ECONOMIC_FEATURES + MARKET_FEATURES + HISTORY_FEATURES

STRATEGIES = {
    "btc-5m-specialist-distilled-fair-value-paper-20260823-v1": {
        "process_id": "eeb305cc-e385-462c-9719-af4ed81d21ac",
        "process_name": "BTC 5m specialist distilled fair value native admission paper",
        "artifact_sha256": "9ec96d8aaedd7ab656230fa12012122ea934f7a350995ada4d0bc2802464f799",
        "source_alias": None,
    },
    "btc-5m-bridge-aware-specialist-umr-20260902-confidence-080": {
        "process_id": "980a140e-3823-4548-b862-475c205d0e2f",
        "process_name": "BTC 5m bridge aware specialist UMR paper",
        "artifact_sha256": "06e5e92e560766ec31281ba3ceb1c43d738d8b326b677bea6ebaaf4dac6f7394",
        "source_alias": "bridge_aware_specialist",
    },
    "btc-5m-extended-specialist-official-umr-20260902-confidence-070": {
        "process_id": "4669169b-75b5-41e0-a08f-790d049a84da",
        "process_name": "BTC 5m extended specialist official UMR paper",
        "artifact_sha256": "457156bdcb08f6f2016781ac50292c32b7a2058bea77ba47542216a8afc3704a",
        "source_alias": "extended_specialist_official",
    },
    "btc-5m-official-high-precision-loss-veto-umr-20260902": {
        "process_id": "96c89496-da28-45fd-b18a-bad865352e9a",
        "process_name": "BTC 5m official high precision loss veto UMR paper",
        "artifact_sha256": "28b0a12d847e32e85dbbca4049f4d3b0f279d9627c7570115c331fc937abe6a4",
        "source_alias": "official_high_precision_loss_veto",
    },
    "btc-5m-official-temporal-consensus-umr-20260902-confidence-075": {
        "process_id": "c3d19b13-16dc-4d20-9afb-d67814f84d38",
        "process_name": "BTC 5m official temporal consensus UMR paper",
        "artifact_sha256": "014fae740a7c43de2baffa5f48023a34ee6b44f7e19b9be72f474ae0a59c94ec",
        "source_alias": "official_temporal_consensus",
    },
    "btc-5m-official-vwap-admission-umr-20260902": {
        "process_id": "4ec890d3-720e-49fb-9b8c-b579b2925091",
        "process_name": "BTC 5m official VWAP admission UMR paper",
        "artifact_sha256": "50cde2b43532409c0045bb426357ed80d175f8355364b4e13766ffffad49e40c",
        "source_alias": "official_vwap_admission",
    },
}


def _bucket_expr() -> pl.Expr:
    return (
        pl.when(pl.col("seconds_elapsed") < 60).then(pl.lit("15_59"))
        .when(pl.col("seconds_elapsed") < 90).then(pl.lit("60_89"))
        .when(pl.col("seconds_elapsed") < 120).then(pl.lit("90_119"))
        .when(pl.col("seconds_elapsed") < 150).then(pl.lit("120_149"))
        .when(pl.col("seconds_elapsed") < 180).then(pl.lit("150_179"))
        .otherwise(pl.lit("180_240"))
    )


def _fair_value_probability(model: dict[str, Any], frame: pl.DataFrame) -> np.ndarray:
    features = list(model["features"]["names"])
    work = frame
    if "early_oracle_eligible" not in work.columns:
        work = work.with_columns(
            pl.col("oracle_model_eligible").fill_null(False).alias("early_oracle_eligible")
        )
    missing = [name for name in features if name not in work.columns]
    if missing:
        raise RuntimeError(f"fair-value replay features unavailable: {missing}")
    outcome = model["payoff_model"]["outcome"]
    local_indices = tuple(int(index) for index in outcome["feature_indices"])
    predictions: list[float] = []
    for values in work.select(features).iter_rows():
        local = []
        for index in local_indices:
            value = values[index]
            numeric = float(value) if value is not None and not isinstance(value, bool) else math.nan
            local.append(numeric if math.isfinite(numeric) else math.nan)
        prediction = float(outcome["baseline"])
        for tree in outcome["trees"]:
            prediction += reached_leaf_value(tree["nodes"], local)
        predictions.append(float(np.clip(prediction, 1e-6, 1.0 - 1e-6)))
    return np.asarray(predictions, dtype=float)


def _fair_value_candidates(archive: Path, package_root: Path) -> pl.DataFrame:
    panel_path = archive / "data/btc-full-august-vwap-admission-20260321-20260901/vwap-admission-panel.parquet"
    model_path = package_root / "runtime-models/btc-5m-specialist-distilled-fair-value-paper-20260823-v1/model.json"
    model = json.loads(model_path.read_text())
    columns = tuple(dict.fromkeys((
        "market_id", "window_start", "observed_at", "seconds_elapsed", "label_up",
        "up_ask_vwap_5", "down_ask_vwap_5", "fee_rate", *MARKET_FEATURES,
        "pm_vwap5_overround", "pm_vwap50_overround", "pm_vwap200_overround",
        "pm_depth_imbalance", "pm_up_book_age_seconds", "pm_down_book_age_seconds",
        *model["features"]["names"], "oracle_model_eligible",
    )))
    scan = pl.scan_parquet(panel_path)
    schema = scan.collect_schema()
    present = [name for name in columns if name in schema]
    frame = (
        scan.filter(
            pl.col("window_start").is_between(
                datetime(2026, 8, 20, tzinfo=UTC), datetime(2026, 8, 26, tzinfo=UTC),
                closed="left",
            )
        )
        .select(present)
        .collect(engine="streaming")
        .sort(["window_start", "seconds_elapsed"])
    )
    probability = _fair_value_probability(model, frame)
    frame = frame.with_columns(pl.Series("probability", probability))
    payoff = model["payoff_model"]
    cells = payoff["policy"]["cells"]
    cell = (
        pl.when(pl.col("seconds_elapsed") < 90).then(pl.lit("early_15_89"))
        .when(pl.col("seconds_elapsed") < 120).then(pl.lit("middle_90_119"))
        .when(pl.col("seconds_elapsed") < 150).then(pl.lit("middle_120_149"))
        .when(pl.col("seconds_elapsed") < 180).then(pl.lit("middle_150_179"))
        .otherwise(pl.lit("late_180_240"))
    )
    frame = frame.with_columns(
        cell.alias("entry_cell"),
        pl.max_horizontal("probability", 1.0 - pl.col("probability")).alias("selected_probability"),
        pl.when(pl.col("probability") >= 0.5).then(pl.col("up_ask_vwap_5"))
        .otherwise(pl.col("down_ask_vwap_5")).alias("share_cost"),
    ).with_columns(
        pl.col("entry_cell").replace_strict(
            {name: float(value["confidence"]) for name, value in cells.items()},
            return_dtype=pl.Float64,
        ).alias("base_confidence_threshold"),
        pl.col("entry_cell").replace_strict(
            {name: float(value["stress_edge"]) for name, value in cells.items()},
            return_dtype=pl.Float64,
        ).alias("base_edge_threshold"),
        (pl.col("fee_rate") * pl.col("share_cost") * (1.0 - pl.col("share_cost"))).alias("fee_per_share"),
    ).with_columns(
        (pl.col("selected_probability") - pl.col("share_cost") - pl.col("fee_per_share")).alias("expected_edge"),
        ((pl.col("probability") >= 0.5) == (pl.col("label_up") == 1)).alias("direction_correct"),
    ).filter(
        (pl.col("selected_probability") >= pl.col("base_confidence_threshold"))
        & ((pl.col("selected_probability") - pl.col("share_cost") - 0.01) >= pl.col("base_edge_threshold"))
        & pl.col("share_cost").is_between(0.0, 1.0, closed="none")
    ).with_columns(
        (pl.when(pl.col("direction_correct")).then(1.0 - pl.col("share_cost") - pl.col("fee_per_share"))
         .otherwise(-pl.col("share_cost") - pl.col("fee_per_share")) * 5.0).alias("net_pnl")
    ).with_columns(
        (pl.col("net_pnl") - 0.05).alias("stress_net_pnl"),
        (pl.col("probability") - 0.5).abs().mul(2.0).alias("confidence"),
        pl.when(pl.col("probability") >= 0.5).then(pl.lit("UP")).otherwise(pl.lit("DOWN")).alias("side"),
        _bucket_expr().alias("time_bucket"),
        (pl.col("net_pnl") <= 0).cast(pl.Int8).alias("loss_label"),
        pl.lit(next(key for key, value in STRATEGIES.items() if value["source_alias"] is None)).alias("champion"),
    )
    return _history(frame)


def build_strategy_panel(archive: Path, package_root: Path) -> pl.DataFrame:
    legacy = build_policy_panel(archive)
    frames: list[pl.DataFrame] = []
    for strategy, identity in STRATEGIES.items():
        alias = identity["source_alias"]
        if alias is None:
            continue
        frames.append(
            legacy.filter(pl.col("champion") == alias).with_columns(pl.lit(strategy).alias("champion"))
        )
    frames.append(_fair_value_candidates(archive, package_root))
    all_columns = sorted(set().union(*(set(frame.columns) for frame in frames)))
    aligned = []
    for frame in frames:
        missing = [name for name in all_columns if name not in frame.columns]
        aligned.append(frame.with_columns(*(pl.lit(None).alias(name) for name in missing)).select(all_columns))
    return (
        pl.concat(aligned, how="diagonal_relaxed")
        .with_columns(_bucket_expr().alias("time_bucket"))
        .sort(["champion", "window_start", "seconds_elapsed"])
    )


class EconomicHarmModel:
    def __init__(self, seed: int):
        self.classifier = _models()["boosted_recent_history"]
        self.features = ALL_FEATURES
        self.loss = self._regressor(seed)
        self.win = self._regressor(seed + 1)

    @staticmethod
    def _regressor(seed: int) -> Pipeline:
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.06, max_iter=160, max_leaf_nodes=15,
                min_samples_leaf=100, l2_regularization=2.0, random_state=seed,
            )),
        ])

    def fit(self, fit: pl.DataFrame, calibration: pl.DataFrame) -> EconomicHarmModel:
        self.classifier.fit(fit, calibration)
        losses = fit.filter(pl.col("net_pnl") <= 0)
        wins = fit.filter(pl.col("net_pnl") > 0)
        self.loss.fit(losses.select(self.features).to_numpy(), (-losses["net_pnl"]).to_numpy())
        self.win.fit(wins.select(self.features).to_numpy(), wins["net_pnl"].to_numpy())
        return self

    def components(self, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = frame.select(self.features).to_numpy()
        probability = self.classifier.predict_proba(frame)
        loss = np.clip(self.loss.predict(matrix), 0.0, None)
        win = np.clip(self.win.predict(matrix), 0.0, None)
        return probability, loss, win


def _intervention_metrics(frame: pl.DataFrame, score: np.ndarray, threshold: float) -> dict[str, Any]:
    baseline = sequential_replay(frame, np.ones(frame.height, dtype=bool))
    risk = sequential_replay(frame, score <= threshold)
    selected_seconds = {
        (row["champion"], row["market_id"]): row["seconds_elapsed"]
        for row in risk.iter_rows(named=True)
    }
    blocked = baseline.with_columns(pl.Series(
        "blocked",
        [
            selected_seconds.get((row["champion"], row["market_id"]))
            != row["seconds_elapsed"]
            for row in baseline.iter_rows(named=True)
        ],
        dtype=pl.Boolean,
    )).filter("blocked")
    bad = int(blocked.filter(pl.col("net_pnl") <= 0).height)
    good = int(blocked.filter(pl.col("net_pnl") > 0).height)
    base_losses = int(baseline.filter(pl.col("net_pnl") <= 0).height)
    base_wins = int(baseline.filter(pl.col("net_pnl") > 0).height)
    loss_capture = bad / base_losses if base_losses else None
    opportunity_rejection = good / base_wins if base_wins else None
    alignment = (
        loss_capture / opportunity_rejection
        if loss_capture is not None and opportunity_rejection not in (None, 0.0) else None
    )
    avoided = float(-blocked.filter(pl.col("net_pnl") <= 0)["net_pnl"].sum())
    missed = float(blocked.filter(pl.col("net_pnl") > 0)["net_pnl"].sum())
    base_by_key = {(row["champion"], row["market_id"]): row for row in baseline.iter_rows(named=True)}
    later = sum(
        int(row["seconds_elapsed"] > base_by_key[(row["champion"], row["market_id"])]["seconds_elapsed"])
        for row in risk.iter_rows(named=True)
    )
    result = {
        "baseline": trade_metrics(baseline, baseline.height),
        "strategy_with_risk": trade_metrics(risk, baseline.height),
        "blocked_losses": bad,
        "blocked_winners": good,
        "loss_capture_rate": loss_capture,
        "opportunity_rejection_rate": opportunity_rejection,
        "risk_alignment_ratio": alignment,
        "block_precision": bad / (bad + good) if bad + good else None,
        "avoided_loss_dollars": avoided,
        "missed_profit_dollars": missed,
        "avoidance_efficiency": avoided / missed if missed else None,
        "deferred_to_later_entry": later,
        "fully_abstained_markets": baseline.height - risk.height,
    }
    result["net_risk_value"] = result["strategy_with_risk"]["net_pnl"] - result["baseline"]["net_pnl"]
    return result


def _threshold_key(frame: pl.DataFrame, score: np.ndarray, threshold: float) -> tuple[Any, ...]:
    rows = []
    for strategy in STRATEGIES:
        part = frame.filter(pl.col("champion") == strategy)
        local_score = score[(frame["champion"] == strategy).to_numpy()]
        rows.append(_intervention_metrics(part, local_score, threshold))
    positive = sum(row["net_risk_value"] > 0 for row in rows)
    normalized = [
        row["net_risk_value"] / max(1.0, abs(row["baseline"]["net_pnl"])) for row in rows
    ]
    aggregate = _intervention_metrics(frame, score, threshold)
    destructive = sum(row["net_risk_value"] < -10.0 for row in rows)
    return (
        -destructive,
        positive,
        float(np.median(normalized)),
        aggregate["strategy_with_risk"]["stress_net_pnl"],
        aggregate["avoidance_efficiency"] or 0.0,
        aggregate["strategy_with_risk"]["coverage_retained"],
        -(aggregate["opportunity_rejection_rate"] or 0.0),
    )


def _select_threshold(frame: pl.DataFrame, score: np.ndarray, floor: float) -> float:
    best: tuple[tuple[Any, ...], float] | None = None
    for threshold in np.unique(np.quantile(score, np.linspace(0.50, 0.98, 25))):
        metrics = _intervention_metrics(frame, score, float(threshold))
        if metrics["strategy_with_risk"]["coverage_retained"] < floor:
            continue
        candidate = (_threshold_key(frame, score, float(threshold)), float(threshold))
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("no threshold satisfies the policy coverage floor")
    return best[1]


def _slices(frame: pl.DataFrame, score: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    scored = frame.with_columns(pl.Series("risk_score", score))
    rows = []
    for strategy in STRATEGIES:
        for bucket in BUCKETS:
            for side in ("UP", "DOWN"):
                part = scored.filter(
                    (pl.col("champion") == strategy) & (pl.col("time_bucket") == bucket)
                    & (pl.col("side") == side)
                )
                local = part["risk_score"].to_numpy()
                evidence = _intervention_metrics(part.drop("risk_score"), local, threshold)
                rows.append({
                    "strategy_model": strategy, "time_bucket": bucket, "side": side,
                    "statistically_insufficient": evidence["baseline"]["trades"] < 30,
                    **evidence,
                })
    return rows


def _bucket_transitions(
    frame: pl.DataFrame, score: np.ndarray, threshold: float, risk_policy: str
) -> list[dict[str, Any]]:
    baseline = sequential_replay(frame, np.ones(frame.height, dtype=bool))
    risk = sequential_replay(frame, score <= threshold)
    selected = {
        (row["champion"], row["market_id"]): row
        for row in risk.iter_rows(named=True)
    }
    rows = []
    for base in baseline.iter_rows(named=True):
        result = selected.get((base["champion"], base["market_id"]))
        rows.append({
            "risk_policy": risk_policy,
            "strategy_model": base["champion"],
            "baseline_bucket": base["time_bucket"],
            "result_bucket": result["time_bucket"] if result is not None else "ABSTAIN",
            "baseline_net_pnl": float(base["net_pnl"]),
            "result_net_pnl": float(result["net_pnl"]) if result is not None else 0.0,
        })
    if not rows:
        return []
    return (
        pl.DataFrame(rows)
        .group_by("risk_policy", "strategy_model", "baseline_bucket", "result_bucket")
        .agg(
            pl.len().alias("markets"),
            pl.col("baseline_net_pnl").sum(),
            pl.col("result_net_pnl").sum(),
        )
        .with_columns(
            (pl.col("result_net_pnl") - pl.col("baseline_net_pnl")).alias("net_risk_value")
        )
        .sort("risk_policy", "strategy_model", "baseline_bucket", "result_bucket")
        .to_dicts()
    )


def _report(metrics: dict[str, Any]) -> str:
    def number(value: Any, digits: int = 3) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Strategy-Conditioned Candidate Risk Tournament", "",
        f"Run: `{metrics['run_id']}`", "",
        "Every PnL, PF, W/L, drawdown, and recovery value below belongs to the named strategy replay with the risk policy applied. Risk estimators themselves are measured by Brier/log loss/AUROC; interventions are measured by blocked-loss and blocked-winner behavior.", "",
        "## Universal risk-policy comparison", "",
        "| Risk policy | Threshold | Trades | W/L | Strategy PnL with risk | Delta PnL | Stress PnL | Coverage | PF | Recovery | Max DD | Bad blocked | Good blocked | Alignment | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics["test_results"].items():
        composite = row["aggregate"]["strategy_with_risk"]
        intervention = row["aggregate"]
        threshold = "—" if row["threshold"] is None else f"{row['threshold']:.6f}"
        lines.append(
            f"| {name} | {threshold} | {composite['trades']} | {composite['wins']}/{composite['losses']} | "
            f"${composite['net_pnl']:.2f} | ${intervention['net_risk_value']:.2f} | ${composite['stress_net_pnl']:.2f} | "
            f"{composite['coverage_retained']:.1%} | {number(composite['profit_factor'])} | {number(composite['recovery_wins_per_loss'])} | "
            f"${composite['max_drawdown']:.2f} | {number(100.0 * intervention['loss_capture_rate'] if intervention['loss_capture_rate'] is not None else None, 1)}% | "
            f"{number(100.0 * intervention['opportunity_rejection_rate'] if intervention['opportunity_rejection_rate'] is not None else None, 1)}% | {number(intervention['risk_alignment_ratio'])}x | "
            f"{number(row['predictive']['brier'], 4)} |"
        )
    winner = metrics["selection"]["universal_champion"]
    qualification = metrics["qualification"]
    lines += [
        "",
        f"Universal research champion selected without opening the test cohort: `{winner}`.",
        f"Untouched-test qualification: **{qualification['status']}**. "
        + ", ".join(f"{name}={'pass' if passed else 'fail'}" for name, passed in qualification["checks"].items())
        + ".",
        "",
        "## Strategy-by-risk-policy test matrix", "",
    ]
    for name, result in metrics["test_results"].items():
        lines += [f"### {name}", "", "| Strategy model | Base PnL | PnL with risk | Delta | Trades | W/L | Coverage | Avg entry | PF | Recovery | Max DD | Loss capture | Opportunity rejection | Alignment |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for strategy, row in result["by_strategy"].items():
            base, risk = row["baseline"], row["strategy_with_risk"]
            lines.append(
                f"| {strategy} | ${base['net_pnl']:.2f} | ${risk['net_pnl']:.2f} | ${row['net_risk_value']:.2f} | "
                f"{risk['trades']} | {risk['wins']}/{risk['losses']} | {risk['coverage_retained']:.1%} | "
                f"{number(risk['average_entry_seconds'], 1)}s | {number(risk['profit_factor'])} | {number(risk['recovery_wins_per_loss'])} | "
                f"${risk['max_drawdown']:.2f} | {number(100.0 * row['loss_capture_rate'] if row['loss_capture_rate'] is not None else None, 1)}% | "
                f"{number(100.0 * row['opportunity_rejection_rate'] if row['opportunity_rejection_rate'] is not None else None, 1)}% | {number(row['risk_alignment_ratio'])}x |"
            )
        lines.append("")
    lines += [
        "## Selected policy by natural entry bucket", "",
        "| Isolated candidate bucket | Trades with risk | W/L | PnL with risk | Delta PnL | Coverage | Avg entry | PF | Recovery | Loss capture | Opportunity rejection | Alignment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket, row in metrics["selected_bucket_summary"].items():
        risk = row["strategy_with_risk"]
        lines.append(
            f"| {bucket} | {risk['trades']} | {risk['wins']}/{risk['losses']} | ${risk['net_pnl']:.2f} | "
            f"${row['net_risk_value']:.2f} | {number(100.0 * risk['coverage_retained'] if risk['coverage_retained'] is not None else None, 1)}% | "
            f"{number(risk['average_entry_seconds'], 1)}s | {number(risk['profit_factor'])} | {number(risk['recovery_wins_per_loss'])} | "
            f"{number(100.0 * row['loss_capture_rate'] if row['loss_capture_rate'] is not None else None, 1)}% | "
            f"{number(100.0 * row['opportunity_rejection_rate'] if row['opportunity_rejection_rate'] is not None else None, 1)}% | {number(row['risk_alignment_ratio'])}x |"
        )
    lines += [
        "", "## Leave-one-strategy-out transfer", "",
        "The selected estimator was re-thresholded on five strategies in policy calibration and applied to the excluded sixth strategy in the untouched test cohort.", "",
        "| Held-out strategy | Threshold | Base PnL | PnL with risk | Delta | Coverage | Loss capture | Opportunity rejection | Alignment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["leave_one_strategy_out"][winner]:
        evidence = row["evidence"]
        lines.append(
            f"| {row['held_out_strategy']} | {row['threshold']:.6f} | ${evidence['baseline']['net_pnl']:.2f} | "
            f"${evidence['strategy_with_risk']['net_pnl']:.2f} | ${evidence['net_risk_value']:.2f} | "
            f"{evidence['strategy_with_risk']['coverage_retained']:.1%} | "
            f"{number(100.0 * evidence['loss_capture_rate'] if evidence['loss_capture_rate'] is not None else None, 1)}% | "
            f"{number(100.0 * evidence['opportunity_rejection_rate'] if evidence['opportunity_rejection_rate'] is not None else None, 1)}% | {number(evidence['risk_alignment_ratio'])}x |"
        )
    lines += ["", "## Time-bucket, side, and transition evidence", "", "The complete risk policy × strategy × natural-entry-bucket × side matrix is stored in `ledgers/slice-matrix.parquet` and `metrics.json`. Empty and sub-30-trade cells are retained and marked statistically insufficient. Baseline-bucket to resulting-bucket deferrals and abstentions are stored in `ledgers/bucket-transitions.parquet`.", "", "## Data and integrity", ""]
    lines.extend(f"- {item}" for item in metrics["limitations"])
    return "\n".join(lines) + "\n"


def train_tournament(config_path: Path, resume_run: str | None = None) -> Path:
    raw = tomllib.loads(config_path.read_text())
    package_root = config_path.resolve().parents[1]
    archive = Path(raw["data"]["archive_root"])
    pilot = package_root / raw["data"]["pilot_run"]
    output_root = package_root / raw["output"]["directory"]
    run_id = resume_run or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    construction_path = pilot / "checkpoints/construction-panel.parquet"
    if not construction_path.exists():
        raise FileNotFoundError(construction_path)
    construction = pl.read_parquet(construction_path)
    strategy_path = checkpoints / "strategy-panel.parquet"
    if strategy_path.exists():
        strategy_panel = pl.read_parquet(strategy_path)
    else:
        strategy_panel = build_strategy_panel(archive, package_root)
        strategy_panel.write_parquet(strategy_path, compression="zstd", statistics=True)

    parse = datetime.fromisoformat
    estimator_calibration_start = parse(raw["splits"]["estimator_calibration_start"])
    estimator_calibration_end = parse(raw["splits"]["estimator_calibration_end"])
    policy_calibration_start = parse(raw["splits"]["policy_calibration_start"])
    policy_test_start = parse(raw["splits"]["policy_test_start"])
    policy_test_end = parse(raw["splits"]["policy_test_end"])
    fit = construction.filter(pl.col("window_start") < estimator_calibration_start)
    estimator_calibration = construction.filter(
        pl.col("window_start").is_between(estimator_calibration_start, estimator_calibration_end, closed="left")
    )
    policy_calibration = strategy_panel.filter(
        pl.col("window_start").is_between(policy_calibration_start, policy_test_start, closed="left")
    )
    test = strategy_panel.filter(
        pl.col("window_start").is_between(policy_test_start, policy_test_end, closed="left")
    )
    sets = [set(frame["market_id"].unique()) for frame in (fit, estimator_calibration, policy_calibration, test)]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            if not sets[left].isdisjoint(sets[right]):
                raise RuntimeError(f"market leakage between chronological cohorts {left} and {right}")
    if any(policy_calibration.filter(pl.col("champion") == strategy).is_empty() for strategy in STRATEGIES):
        raise RuntimeError("policy calibration does not cover all six strategy models")
    if any(test.filter(pl.col("champion") == strategy).is_empty() for strategy in STRATEGIES):
        raise RuntimeError("test does not cover all six strategy models")

    model_checkpoint = checkpoints / "models.joblib"
    if model_checkpoint.exists():
        fitted = joblib.load(model_checkpoint)
    else:
        fitted = _models()
        for model in fitted.values():
            model.fit(fit, estimator_calibration)
        fitted["economic_harm"] = EconomicHarmModel(int(raw["training"]["random_seed"])).fit(fit, estimator_calibration)
        joblib.dump(fitted, model_checkpoint, compress=3)

    pilot_artifact = joblib.load(pilot / "tournament.joblib")
    pilot_model = pilot_artifact["models"]["boosted_recent_history"]
    from .candidate_loss_risk_tournament import CalibratedRiskModel
    frozen = CalibratedRiskModel(pilot_model["estimator"], tuple(pilot_model["features"]))
    frozen.calibrator = pilot_model["calibrator"]
    fitted["frozen_pilot_boosted_recent_history"] = frozen

    def score_models(frame: pl.DataFrame) -> dict[str, dict[str, np.ndarray]]:
        output = {}
        mean_loss = float(-fit.filter(pl.col("net_pnl") <= 0)["net_pnl"].mean())
        mean_win = float(fit.filter(pl.col("net_pnl") > 0)["net_pnl"].mean())
        deferral = float(raw["training"]["deferral_cost_per_candidate"])
        for name, model in fitted.items():
            if isinstance(model, EconomicHarmModel):
                probability, loss, win = model.components(frame)
            else:
                probability = model.predict_proba(frame)
                loss = np.full(frame.height, mean_loss)
                win = np.full(frame.height, mean_win)
            output[name] = {
                "probability": probability,
                "loss_magnitude": loss,
                "win_magnitude": win,
                "score": probability * loss - (1.0 - probability) * win - deferral,
            }
        return output

    calibration_scores = score_models(policy_calibration)
    test_scores = score_models(test)
    floor = float(raw["training"]["minimum_coverage"])
    thresholds = {
        name: _select_threshold(policy_calibration, values["score"], floor)
        for name, values in calibration_scores.items()
    }
    calibration_results = {
        name: _intervention_metrics(policy_calibration, values["score"], thresholds[name])
        for name, values in calibration_scores.items()
    }
    ordering = sorted(
        thresholds,
        key=lambda name: _threshold_key(policy_calibration, calibration_scores[name]["score"], thresholds[name]),
        reverse=True,
    )
    winner = ordering[0]
    test_results: dict[str, Any] = {}
    slice_rows = []
    transition_rows = []
    no_risk_score = np.zeros(test.height, dtype=float)
    no_risk = _intervention_metrics(test, no_risk_score, math.inf)
    no_risk_by_strategy = {}
    for strategy in STRATEGIES:
        mask = (test["champion"] == strategy).to_numpy()
        no_risk_by_strategy[strategy] = _intervention_metrics(
            test.filter(pl.Series(mask)), no_risk_score[mask], math.inf
        )
    test_results["no_risk"] = {
        "threshold": None, "aggregate": no_risk, "by_strategy": no_risk_by_strategy,
        "predictive": {"brier": None, "log_loss": None, "roc_auc": None},
    }
    for row in _slices(test, no_risk_score, math.inf):
        slice_rows.append({"risk_policy": "no_risk", **row})
    transition_rows.extend(_bucket_transitions(test, no_risk_score, math.inf, "no_risk"))
    for name, values in test_scores.items():
        threshold = thresholds[name]
        aggregate = _intervention_metrics(test, values["score"], threshold)
        by_strategy = {}
        for strategy in STRATEGIES:
            mask = (test["champion"] == strategy).to_numpy()
            by_strategy[strategy] = _intervention_metrics(test.filter(pl.Series(mask)), values["score"][mask], threshold)
        predictive = _predictive(test["loss_label"].to_numpy(), values["probability"])
        test_results[name] = {
            "threshold": threshold, "aggregate": aggregate,
            "by_strategy": by_strategy, "predictive": predictive,
        }
        for row in _slices(test, values["score"], threshold):
            slice_rows.append({"risk_policy": name, **row})
        transition_rows.extend(_bucket_transitions(test, values["score"], threshold, name))

    # Controls inherit the selected risk policy's calibration coverage, but their thresholds are frozen on calibration.
    target_coverage = calibration_results[winner]["strategy_with_risk"]["coverage_retained"]
    for control, column in (("matched_confidence", "confidence"), ("matched_edge", "expected_edge")):
        threshold = float(policy_calibration[column].quantile(max(0.0, 1.0 - target_coverage)))
        cal_score = -policy_calibration[column].to_numpy()
        test_score = -test[column].to_numpy()
        limit = -threshold
        calibration_results[control] = _intervention_metrics(policy_calibration, cal_score, limit)
        aggregate = _intervention_metrics(test, test_score, limit)
        by_strategy = {}
        for strategy in STRATEGIES:
            mask = (test["champion"] == strategy).to_numpy()
            by_strategy[strategy] = _intervention_metrics(test.filter(pl.Series(mask)), test_score[mask], limit)
        test_results[control] = {
            "threshold": threshold, "aggregate": aggregate, "by_strategy": by_strategy,
            "predictive": {"brier": None, "log_loss": None, "roc_auc": None},
        }
        for row in _slices(test, test_score, limit):
            slice_rows.append({"risk_policy": control, **row})
        transition_rows.extend(_bucket_transitions(test, test_score, limit, control))

    leave_one_out: dict[str, list[dict[str, Any]]] = {}
    for name, values in calibration_scores.items():
        leave_one_out[name] = []
        for held_out in STRATEGIES:
            calibration_mask = (policy_calibration["champion"] != held_out).to_numpy()
            threshold = _select_threshold(
                policy_calibration.filter(pl.Series(calibration_mask)),
                values["score"][calibration_mask],
                floor,
            )
            test_mask = (test["champion"] == held_out).to_numpy()
            evidence = _intervention_metrics(
                test.filter(pl.Series(test_mask)), test_scores[name]["score"][test_mask], threshold
            )
            leave_one_out[name].append({
                "held_out_strategy": held_out, "threshold": threshold, "evidence": evidence,
            })

    ledgers = run_dir / "ledgers"
    ledgers.mkdir(exist_ok=True)
    scored = test
    for name, values in test_scores.items():
        scored = scored.with_columns(
            pl.Series(f"loss_probability__{name}", values["probability"]),
            pl.Series(f"risk_score__{name}", values["score"]),
        )
    scored.write_parquet(ledgers / "test-candidates.parquet", compression="zstd", statistics=True)
    pl.DataFrame(slice_rows).write_parquet(ledgers / "slice-matrix.parquet", compression="zstd", statistics=True)
    pl.DataFrame(transition_rows).write_parquet(
        ledgers / "bucket-transitions.parquet", compression="zstd", statistics=True
    )

    producing_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package_root, text=True).strip()
    artifact = run_dir / "tournament.joblib"
    joblib.dump({
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "producing_commit": producing_commit,
        "models": fitted, "thresholds": thresholds, "universal_champion": winner,
        "deployment_status": "not_deployed",
    }, artifact, compress=3)
    artifact_sha = _sha256(artifact)
    (run_dir / "tournament.sha256").write_text(f"{artifact_sha}  tournament.joblib\n")
    selected_evidence = test_results[winner]["aggregate"]
    selected_risk = selected_evidence["strategy_with_risk"]
    selected_baseline = selected_evidence["baseline"]
    confidence_control = test_results["matched_confidence"]["aggregate"]
    edge_control = test_results["matched_edge"]["aggregate"]
    positive_strategies = sum(
        row["net_risk_value"] > 0 for row in test_results[winner]["by_strategy"].values()
    )
    qualification_checks = {
        "net_pnl_improved": selected_evidence["net_risk_value"] > 0,
        "stress_pnl_improved": selected_risk["stress_net_pnl"] > selected_baseline["stress_net_pnl"],
        "max_drawdown_reduced": selected_risk["max_drawdown"] < selected_baseline["max_drawdown"],
        "risk_alignment_above_one": (selected_evidence["risk_alignment_ratio"] or 0.0) > 1.0,
        "avoided_losses_exceed_missed_profit": selected_evidence["avoided_loss_dollars"] > selected_evidence["missed_profit_dollars"],
        "aggregate_coverage_floor": selected_risk["coverage_retained"] >= floor,
        "per_strategy_coverage_floor": all(
            row["strategy_with_risk"]["coverage_retained"] >= floor
            for row in test_results[winner]["by_strategy"].values()
        ),
        "positive_majority_of_strategies": positive_strategies >= 4,
        "no_materially_destructive_strategy": all(
            row["net_risk_value"] >= -10.0 for row in test_results[winner]["by_strategy"].values()
        ),
        "beats_matched_controls": selected_risk["net_pnl"] > max(
            confidence_control["strategy_with_risk"]["net_pnl"],
            edge_control["strategy_with_risk"]["net_pnl"],
        ),
    }
    selected_bucket_summary = {}
    winner_score = test_scores[winner]["score"]
    for bucket in BUCKETS:
        mask = (test["time_bucket"] == bucket).to_numpy()
        selected_bucket_summary[bucket] = _intervention_metrics(
            test.filter(pl.Series(mask)), winner_score[mask], thresholds[winner]
        )
    metrics = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(), "producing_commit": producing_commit,
        "artifact_sha256": artifact_sha, "qualification_status": "trained_evaluated_research_only",
        "deployment_status": "not_deployed", "strategy_identities": STRATEGIES,
        "splits": {
            "fit": [str(fit["window_start"].min()), str(fit["window_start"].max())],
            "estimator_calibration": [str(estimator_calibration["window_start"].min()), str(estimator_calibration["window_start"].max())],
            "policy_calibration": [str(policy_calibration["window_start"].min()), str(policy_calibration["window_start"].max())],
            "policy_test": [str(test["window_start"].min()), str(test["window_start"].max())],
        },
        "row_counts": {"fit": fit.height, "estimator_calibration": estimator_calibration.height, "policy_calibration": policy_calibration.height, "policy_test": test.height},
        "selection": {"universal_champion": winner, "ranking": ordering, "selected_on": "policy_calibration_only"},
        "thresholds": thresholds, "calibration_results": calibration_results,
        "test_results": test_results, "slice_matrix": slice_rows,
        "bucket_transitions": transition_rows,
        "selected_bucket_summary": selected_bucket_summary,
        "leave_one_strategy_out": leave_one_out,
        "qualification": {
            "status": "research_qualified" if all(qualification_checks.values()) else "research_not_qualified",
            "checks": qualification_checks,
        },
        "source_identity": {
            "construction_panel": {"path": str(construction_path), "sha256": _sha256(construction_path)},
            "strategy_panel": {"path": str(strategy_path), "sha256": _sha256(strategy_path)},
            "fair_value_model": {"path": str(package_root / "runtime-models/btc-5m-specialist-distilled-fair-value-paper-20260823-v1/model.json"), "sha256": STRATEGIES[next(iter(STRATEGIES))]["artifact_sha256"]},
        },
        "integrity": {"market_disjoint": True, "database_reads": False, "database_mutations": False, "new_tables": False, "new_schemas": False, "new_ingesters": False, "new_sources": False, "images_rebuilt": False},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "polars": pl.__version__, "sklearn": sklearn.__version__},
        "limitations": [
            "The risk-estimator construction pool uses causal OOF strategy candidates from the established March-August archive; strategy identity is deliberately excluded from predictive features.",
            "Exact admission candidate ledgers for all six currently running strategy models overlap only August 20-25, so threshold calibration uses August 20 and the untouched comparison uses August 21-25.",
            "Archived strategy candidate parity is unavailable for August 26-September 1 for five strategies; those dates were not synthesized or substituted.",
            "The archived common panel begins at second 60, so the 15-59 bucket is reported as unavailable rather than inferred.",
            "Fair-value strategy candidates are exact immutable-runtime-artifact replays; the other five are their frozen tournament admission ledgers.",
            "This is research-only pilot evidence. No model or policy was deployed and no trading process was changed.",
        ],
    }
    _write_json(run_dir / "selection-freeze.json", metrics["selection"])
    _write_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(_report(metrics))
    _write_json(run_dir / "completion.json", {"completed": True, "artifact_sha256": artifact_sha, "metrics_sha256": _sha256(run_dir / "metrics.json")})
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-run")
    args = parser.parse_args()
    print(train_tournament(args.config, args.resume_run))


if __name__ == "__main__":
    main()
