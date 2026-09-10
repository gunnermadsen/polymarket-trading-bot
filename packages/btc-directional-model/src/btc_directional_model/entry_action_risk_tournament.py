"""Train entry-action risk models and replay them through six frozen strategies."""

from __future__ import annotations

import argparse
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import __main__

from .candidate_loss_risk_tournament import _sha256, _write_json, trade_metrics
from .strategy_conditioned_risk_tournament import (
    BUCKETS,
    STRATEGIES,
    EconomicHarmModel,
    build_strategy_panel,
)

SCHEMA_VERSION = "strategy_conditioned_entry_action_risk_tournament_v1"
KEYS = ("market_id", "observed_at")
GROUP = ("champion", "market_id")
TARGET_COLUMNS = {
    "net_pnl", "stress_net_pnl", "loss_label", "direction_correct", "label_up",
    "official_outcome", "final_price", "future_best_net_pnl", "defer_value",
    "allow_advantage", "allow_is_optimal", "optimal_action",
}
IDENTITY_COLUMNS = {
    "champion", "market_id", "window_start", "window_end", "observed_at",
    "time_bucket", "side", "fold", "provider", "symbol", "market_slug",
}


def attach_action_targets(frame: pl.DataFrame, penalty: float) -> pl.DataFrame:
    """Attach oracle training targets using strictly later candidates in each market."""
    frame = frame.sort([*GROUP, "seconds_elapsed"])
    future = (
        pl.col("net_pnl").shift(-1).reverse().cum_max().reverse().over(GROUP)
        .fill_null(0.0)
    )
    frame = frame.with_columns(future.alias("future_best_net_pnl"))
    frame = frame.with_columns(
        pl.max_horizontal(pl.col("future_best_net_pnl") - penalty, 0.0).alias("defer_value")
    )
    return frame.with_columns(
        (pl.col("net_pnl") - pl.col("defer_value")).alias("allow_advantage"),
        (pl.col("net_pnl") >= pl.col("defer_value")).cast(pl.Int8).alias("allow_is_optimal"),
        pl.when(pl.col("net_pnl") >= pl.col("defer_value")).then(pl.lit("allow_now"))
        .when(pl.col("defer_value") > 0).then(pl.lit("defer"))
        .otherwise(pl.lit("abstain")).alias("optimal_action"),
    )


def attach_trajectory(frame: pl.DataFrame) -> pl.DataFrame:
    frame = frame.sort([*GROUP, "seconds_elapsed"])
    return frame.with_columns(
        pl.col("selected_probability").diff().over(GROUP).alias("probability_change"),
        pl.col("expected_edge").diff().over(GROUP).alias("edge_change"),
        pl.col("share_cost").diff().over(GROUP).alias("cost_change"),
        pl.col("seconds_elapsed").diff().over(GROUP).alias("seconds_since_prior_candidate"),
        (240.0 - pl.col("seconds_elapsed")).clip(0.0, 240.0).alias("seconds_remaining"),
        (pl.col("selected_probability") - pl.col("history_win_rate_25"))
        .alias("recent_calibration_gap"),
    )


def _feature_names(frame: pl.DataFrame) -> tuple[str, ...]:
    names = []
    for name, dtype in frame.schema.items():
        if name in TARGET_COLUMNS or name in IDENTITY_COLUMNS:
            continue
        if dtype.is_numeric() or dtype == pl.Boolean:
            names.append(name)
    return tuple(sorted(names))


def _join_rich_dimensions(base: pl.DataFrame, rich_path: Path) -> pl.DataFrame:
    schema = pl.scan_parquet(rich_path).collect_schema()
    usable = [
        name for name, dtype in schema.items()
        if name not in TARGET_COLUMNS | IDENTITY_COLUMNS
        and name not in base.columns
        and (dtype.is_numeric() or dtype == pl.Boolean)
    ]
    rich = (
        pl.scan_parquet(rich_path)
        .select(*KEYS, *usable)
        .unique(subset=list(KEYS), keep="first")
    )
    return base.lazy().join(rich, on=list(KEYS), how="left").collect(engine="streaming")


def build_training_panel(
    construction_path: Path, rich_path: Path, maximum_rows: int, penalty: float
) -> pl.DataFrame:
    panel = attach_trajectory(attach_action_targets(pl.read_parquet(construction_path), penalty))
    if panel.height > maximum_rows:
        stride = math.ceil(panel.height / maximum_rows)
        panel = panel.with_row_index().filter(pl.col("index") % stride == 0).drop("index")
    return _join_rich_dimensions(panel, rich_path)


class EntryActionModel:
    def __init__(self, name: str, features: tuple[str, ...], seed: int):
        self.name = name
        self.features = features
        self.seed = seed
        self.kind = "logistic" if name == "logistic_action_value" else "regression"
        if self.kind == "logistic":
            self.allow = Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.2, max_iter=500, class_weight="balanced")),
            ])
            self.defer = None
        else:
            loss = "quantile" if name == "downside_quantile_action" else "squared_error"
            quantile = 0.25 if loss == "quantile" else None
            self.allow = self._regressor(seed, loss, quantile)
            self.defer = self._regressor(seed + 1, "squared_error", None)

    @staticmethod
    def _regressor(seed: int, loss: str, quantile: float | None) -> Pipeline:
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", HistGradientBoostingRegressor(
                loss=loss, quantile=quantile, learning_rate=0.065, max_iter=90,
                max_leaf_nodes=15, min_samples_leaf=150, l2_regularization=2.0,
                random_state=seed,
            )),
        ])

    def fit(self, frame: pl.DataFrame) -> EntryActionModel:
        matrix = frame.select(self.features).to_numpy()
        if self.kind == "logistic":
            self.allow.fit(matrix, frame["allow_is_optimal"].to_numpy())
        else:
            self.allow.fit(matrix, frame["net_pnl"].to_numpy())
            assert self.defer is not None
            self.defer.fit(matrix, frame["defer_value"].to_numpy())
        return self

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = frame.select(self.features).to_numpy()
        if self.kind == "logistic":
            return self.allow.predict_proba(matrix)[:, 1]
        assert self.defer is not None
        return self.allow.predict(matrix) - self.defer.predict(matrix)


def _model_features(all_features: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    economic = tuple(name for name in all_features if name in {
        "seconds_elapsed", "seconds_remaining", "selected_probability", "confidence",
        "share_cost", "fee_per_share", "expected_edge", "probability_change", "edge_change",
        "cost_change", "seconds_since_prior_candidate", "recent_calibration_gap",
    } or name.startswith("history_"))
    regime = tuple(name for name in all_features if name.startswith((
        "btc_", "chainlink_", "binance_", "kraken_", "spot_l2_", "pm_", "oracle_"
    )))
    market_context = tuple(dict.fromkeys((*economic, *regime)))
    return {
        "logistic_action_value": economic,
        "boosted_enter_vs_defer": economic,
        "downside_quantile_action": market_context,
        "regime_conditioned_action": regime,
        "regime_strategy_health_action": all_features,
    }


def _selected(frame: pl.DataFrame, score: np.ndarray, threshold: float) -> pl.DataFrame:
    return (
        frame.with_columns(pl.Series("risk_allowed", score >= threshold))
        .filter("risk_allowed").sort(["champion", "window_start", "seconds_elapsed"])
        .group_by(GROUP, maintain_order=True).first()
    )


def intervention_metrics(frame: pl.DataFrame, score: np.ndarray, threshold: float) -> dict[str, Any]:
    baseline = _selected(frame, np.ones(frame.height), -math.inf)
    risk = _selected(frame, score, threshold)
    risk_by_key = {(r["champion"], r["market_id"]): r for r in risk.iter_rows(named=True)}
    oracle_by_key = {
        (r["champion"], r["market_id"]): max(0.0, float(r["oracle_net_pnl"]))
        for r in frame.group_by(GROUP).agg(pl.col("net_pnl").max().alias("oracle_net_pnl"))
        .iter_rows(named=True)
    }
    blocked_bad = blocked_good = deferred = improved = 0
    avoided = missed = regret = 0.0
    for row in baseline.iter_rows(named=True):
        key = (row["champion"], row["market_id"])
        chosen = risk_by_key.get(key)
        changed = chosen is None or chosen["observed_at"] != row["observed_at"]
        if changed and row["net_pnl"] <= 0:
            blocked_bad += 1
            avoided += -float(row["net_pnl"])
        elif changed:
            blocked_good += 1
            missed += float(row["net_pnl"])
        if chosen is not None and chosen["seconds_elapsed"] > row["seconds_elapsed"]:
            deferred += 1
            improved += int(chosen["net_pnl"] > row["net_pnl"])
        oracle = oracle_by_key[key]
        regret += oracle - (float(chosen["net_pnl"]) if chosen is not None else 0.0)
    losses = int(baseline.filter(pl.col("net_pnl") <= 0).height)
    wins = int(baseline.filter(pl.col("net_pnl") > 0).height)
    loss_capture = blocked_bad / losses if losses else None
    good_rejection = blocked_good / wins if wins else None
    alignment = (
        loss_capture / good_rejection
        if loss_capture is not None and good_rejection not in (None, 0.0)
        else None
    )
    base_metrics = trade_metrics(baseline, baseline.height)
    risk_metrics = trade_metrics(risk, baseline.height)
    return {
        "baseline": base_metrics, "strategy_with_risk": risk_metrics,
        "net_risk_value": risk_metrics["net_pnl"] - base_metrics["net_pnl"],
        "blocked_losses": blocked_bad, "blocked_winners": blocked_good,
        "loss_capture_rate": loss_capture, "opportunity_rejection_rate": good_rejection,
        "risk_alignment_ratio": alignment,
        "block_precision": blocked_bad / (blocked_bad + blocked_good) if blocked_bad + blocked_good else None,
        "avoided_loss_dollars": avoided, "missed_profit_dollars": missed,
        "deferred_markets": deferred, "beneficial_deferrals": improved,
        "deferral_improvement_rate": improved / deferred if deferred else None,
        "fully_abstained_markets": baseline.height - risk.height,
        "action_regret": regret,
    }


def threshold_for_coverage(frame: pl.DataFrame, score: np.ndarray, target: float) -> float:
    baseline_markets = frame.select(*GROUP).unique().height
    candidates = np.unique(np.quantile(score, np.linspace(0.0, 1.0, 51)))
    return min(
        (float(value) for value in candidates),
        key=lambda value: abs(
            (_selected(frame, score, value).height / baseline_markets) - target
        ),
    )


def _evaluate(
    frame: pl.DataFrame, score: np.ndarray, threshold: float, policy: str, coverage: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate = intervention_metrics(frame, score, threshold)
    by_strategy = {}
    slices = []
    for strategy in STRATEGIES:
        mask = (frame["champion"] == strategy).to_numpy()
        part = frame.filter(pl.Series(mask))
        local = score[mask]
        by_strategy[strategy] = intervention_metrics(part, local, threshold)
        for bucket in BUCKETS:
            for side in ("UP", "DOWN"):
                cell_mask = ((part["time_bucket"] == bucket) & (part["side"] == side)).to_numpy()
                cell = part.filter(pl.Series(cell_mask))
                evidence = intervention_metrics(cell, local[cell_mask], threshold)
                slices.append({
                    "risk_policy": policy, "coverage_target": coverage,
                    "strategy_model": strategy, "time_bucket": bucket, "side": side,
                    "statistically_insufficient": evidence["baseline"]["trades"] < 30,
                    **evidence,
                })
    return {"threshold": threshold, "aggregate": aggregate, "by_strategy": by_strategy}, slices


def _selection_key(folds: list[dict[str, Any]]) -> tuple[Any, ...]:
    aggregate = [item["aggregate"] for item in folds]
    return (
        sum(item["net_risk_value"] > 0 for item in aggregate),
        float(np.median([item["net_risk_value"] for item in aggregate])),
        float(np.median([(item["risk_alignment_ratio"] or 0.0) for item in aggregate])),
        -float(np.median([item["action_regret"] for item in aggregate])),
    )


def _report(metrics: dict[str, Any]) -> str:
    def n(value: Any, digits: int = 2) -> str:
        return "—" if value is None else f"{value:.{digits}f}"
    lines = [
        "# Strategy-Conditioned Entry-Action Risk Tournament", "",
        "PnL, PF, recovery, drawdown, W/L and entry time below are properties of the named strategy replay after the risk model is applied—not standalone risk-model returns.", "",
        f"Research selection: `{metrics['selection']['champion']}` at {metrics['selection']['primary_coverage']:.0%} target coverage.", "",
        "## Exact six-strategy compatibility comparison", "",
        "| Risk model | Coverage target | Trades | W/L | Strategy PnL | Delta PnL | Coverage | PF | Recovery | Max DD | Bad blocked | Good blocked | Alignment | Deferred | Better deferrals | Action regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, bands in metrics["compatibility_results"].items():
        for band, row in bands.items():
            e, risk = row["aggregate"], row["aggregate"]["strategy_with_risk"]
            lines.append(
                f"| {name} | {float(band):.0%} | {risk['trades']} | {risk['wins']}/{risk['losses']} | ${risk['net_pnl']:.2f} | ${e['net_risk_value']:.2f} | {risk['coverage_retained']:.1%} | {n(risk['profit_factor'],3)} | {n(risk['recovery_wins_per_loss'],3)} | ${risk['max_drawdown']:.2f} | {n(e['loss_capture_rate']*100 if e['loss_capture_rate'] is not None else None,1)}% | {n(e['opportunity_rejection_rate']*100 if e['opportunity_rejection_rate'] is not None else None,1)}% | {n(e['risk_alignment_ratio'],2)}x | {e['deferred_markets']} | {n(e['deferral_improvement_rate']*100 if e['deferral_improvement_rate'] is not None else None,1)}% | ${e['action_regret']:.2f} |"
            )
    lines += ["", "## Selected risk model through each trading strategy", ""]
    selected = metrics["selection"]["champion"]
    primary = str(metrics["selection"]["primary_coverage"])
    lines += ["| Trading process / strategy model | Base PnL | PnL with risk | Delta | Trades | W/L | Coverage | Avg entry | Avg cost | PF | Recovery | Max DD | Bad blocked | Good blocked | Alignment |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for strategy, e in metrics["compatibility_results"][selected][primary]["by_strategy"].items():
        b, r = e["baseline"], e["strategy_with_risk"]
        lines.append(f"| {STRATEGIES[strategy]['process_name']} | ${b['net_pnl']:.2f} | ${r['net_pnl']:.2f} | ${e['net_risk_value']:.2f} | {r['trades']} | {r['wins']}/{r['losses']} | {r['coverage_retained']:.1%} | {n(r['average_entry_seconds'],1)}s | ${n(r['average_cost'],3)} | {n(r['profit_factor'],3)} | {n(r['recovery_wins_per_loss'],3)} | ${r['max_drawdown']:.2f} | {n(e['loss_capture_rate']*100 if e['loss_capture_rate'] is not None else None,1)}% | {n(e['opportunity_rejection_rate']*100 if e['opportunity_rejection_rate'] is not None else None,1)}% | {n(e['risk_alignment_ratio'],2)}x |")
    lines += [
        "", "## Time buckets and sides", "",
        "The complete risk-model × coverage × strategy × natural time-bucket × side matrix is in `ledgers/slice-matrix.parquet` and `metrics.json`. Empty and sub-30-trade cells are retained.", "",
        "## Integrity and limitations", "",
    ]
    lines.extend(f"- {item}" for item in metrics["limitations"])
    return "\n".join(lines) + "\n"


def train_tournament(config_path: Path, resume_run: str | None = None) -> Path:
    raw = tomllib.loads(config_path.read_text())
    package_root = config_path.resolve().parents[1]
    archive = Path(raw["data"]["archive_root"])
    candidate_run = package_root / raw["data"]["candidate_risk_run"]
    strategy_run = package_root / raw["data"]["strategy_risk_run"]
    rich_path = archive / raw["data"]["rich_panel"]
    output_root = package_root / raw["output"]["directory"]
    run_id = resume_run or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    training_path = checkpoints / "training-panel.parquet"
    if training_path.exists():
        training = pl.read_parquet(training_path)
    else:
        training = build_training_panel(
            candidate_run / "checkpoints/construction-panel.parquet", rich_path,
            int(raw["training"]["maximum_rows"]), float(raw["training"]["deferral_penalty"]),
        )
        training.write_parquet(training_path, compression="zstd", statistics=True)
    compatibility_path = checkpoints / "strategy-panel.parquet"
    if compatibility_path.exists():
        compatibility = pl.read_parquet(compatibility_path)
    else:
        source = strategy_run / "checkpoints/strategy-panel.parquet"
        base = pl.read_parquet(source) if source.exists() else build_strategy_panel(archive, package_root)
        compatibility = _join_rich_dimensions(attach_trajectory(attach_action_targets(
            base, float(raw["training"]["deferral_penalty"])
        )), rich_path)
        compatibility.write_parquet(compatibility_path, compression="zstd", statistics=True)

    parse = datetime.fromisoformat
    fit_end = parse(raw["splits"]["selection_fit_end"])
    boundaries = [parse(value) for value in raw["splits"]["walk_forward_boundaries"]]
    selection_fit = training.filter(pl.col("window_start") < fit_end)
    all_features = _feature_names(training)
    feature_sets = _model_features(all_features)
    seed = int(raw["training"]["random_seed"])
    selection_checkpoint = checkpoints / "selection-models.joblib"
    if selection_checkpoint.exists():
        selection_models = joblib.load(selection_checkpoint)
    else:
        selection_models = {
            name: EntryActionModel(name, features, seed + index * 10).fit(selection_fit)
            for index, (name, features) in enumerate(feature_sets.items())
        }
        joblib.dump(selection_models, selection_checkpoint, compress=3)

    coverage_targets = tuple(float(value) for value in raw["training"]["coverage_targets"])
    primary = float(raw["training"]["primary_coverage"])
    walk_forward: dict[str, list[dict[str, Any]]] = {name: [] for name in selection_models}
    for model_name, model in selection_models.items():
        for index in range(len(boundaries) - 2):
            calibration = training.filter(pl.col("window_start").is_between(
                boundaries[index], boundaries[index + 1], closed="left"
            ))
            test = training.filter(pl.col("window_start").is_between(
                boundaries[index + 1], boundaries[index + 2], closed="left"
            ))
            cal_score, test_score = model.score(calibration), model.score(test)
            threshold = threshold_for_coverage(calibration, cal_score, primary)
            evidence, _ = _evaluate(test, test_score, threshold, model_name, primary)
            walk_forward[model_name].append({
                "calibration_range": [str(calibration["window_start"].min()), str(calibration["window_start"].max())],
                "test_range": [str(test["window_start"].min()), str(test["window_start"].max())],
                **evidence,
            })
    ranking = sorted(walk_forward, key=lambda name: _selection_key(walk_forward[name]), reverse=True)
    champion = ranking[0]
    _write_json(run_dir / "selection-freeze.json", {
        "champion": champion, "ranking": ranking, "primary_coverage": primary,
        "selected_on": "chronological historical walk-forward only",
    })

    final_checkpoint = checkpoints / "final-models.joblib"
    if final_checkpoint.exists():
        final_models = joblib.load(final_checkpoint)
    else:
        final_models = {
            name: EntryActionModel(name, features, seed + index * 10).fit(training)
            for index, (name, features) in enumerate(feature_sets.items())
        }
        joblib.dump(final_models, final_checkpoint, compress=3)

    cal_start = parse(raw["splits"]["compatibility_calibration_start"])
    test_start = parse(raw["splits"]["compatibility_test_start"])
    test_end = parse(raw["splits"]["compatibility_test_end"])
    cal = compatibility.filter(pl.col("window_start").is_between(cal_start, test_start, closed="left"))
    test = compatibility.filter(pl.col("window_start").is_between(test_start, test_end, closed="left"))
    cal_markets, test_markets = set(cal["market_id"]), set(test["market_id"])
    if not cal_markets.isdisjoint(test_markets):
        raise RuntimeError("market leakage between compatibility calibration and test")
    if any(test.filter(pl.col("champion") == strategy).is_empty() for strategy in STRATEGIES):
        raise RuntimeError("compatibility comparison does not cover all six strategies")

    results: dict[str, Any] = {}
    slice_rows: list[dict[str, Any]] = []
    scored = test
    thresholds: dict[str, dict[str, float]] = {}
    for name, model in final_models.items():
        cal_score, test_score = model.score(cal), model.score(test)
        scored = scored.with_columns(pl.Series(f"action_score__{name}", test_score))
        results[name], thresholds[name] = {}, {}
        for target in coverage_targets:
            threshold = threshold_for_coverage(cal, cal_score, target)
            thresholds[name][str(target)] = threshold
            evidence, slices = _evaluate(test, test_score, threshold, name, target)
            results[name][str(target)] = evidence
            slice_rows.extend(slices)

    # Frozen loss-risk and simple economics controls are comparison policies only. They do not
    # participate in research-model selection and are matched to the same coverage bands.
    # The preceding CLI-authored artifact recorded its local composite under __main__.
    # Bind that known class only for backwards-compatible deserialization of the frozen control.
    __main__.EconomicHarmModel = EconomicHarmModel
    prior_artifact = joblib.load(strategy_run / "tournament.joblib")
    prior_fit = pl.read_parquet(candidate_run / "checkpoints/construction-panel.parquet")
    mean_loss = float(-prior_fit.filter(pl.col("net_pnl") <= 0)["net_pnl"].mean())
    mean_win = float(prior_fit.filter(pl.col("net_pnl") > 0)["net_pnl"].mean())
    deferral_cost = float(raw["training"]["deferral_penalty"])
    control_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "matched_confidence": (
            cal["confidence"].to_numpy(), test["confidence"].to_numpy()
        ),
        "matched_edge": (
            cal["expected_edge"].to_numpy(), test["expected_edge"].to_numpy()
        ),
    }
    for control_name, prior_name in (
        ("frozen_logistic_loss_risk", "logistic_economics"),
        ("frozen_boosted_history_loss_risk", "boosted_recent_history"),
    ):
        prior_model = prior_artifact["models"][prior_name]
        def allow_score(frame: pl.DataFrame, model: Any = prior_model) -> np.ndarray:
            probability = model.predict_proba(frame)
            harm = probability * mean_loss - (1.0 - probability) * mean_win - deferral_cost
            return -harm
        control_scores[control_name] = (allow_score(cal), allow_score(test))
    for name, (cal_score, test_score) in control_scores.items():
        scored = scored.with_columns(pl.Series(f"action_score__{name}", test_score))
        results[name], thresholds[name] = {}, {}
        for target in coverage_targets:
            threshold = threshold_for_coverage(cal, cal_score, target)
            thresholds[name][str(target)] = threshold
            evidence, slices = _evaluate(test, test_score, threshold, name, target)
            results[name][str(target)] = evidence
            slice_rows.extend(slices)
    no_risk, no_risk_slices = _evaluate(
        test, np.ones(test.height), -math.inf, "no_risk", 1.0
    )
    results["no_risk"] = {"1.0": no_risk}
    thresholds["no_risk"] = {"1.0": -math.inf}
    slice_rows.extend(no_risk_slices)

    ledgers = run_dir / "ledgers"
    ledgers.mkdir(exist_ok=True)
    scored.write_parquet(ledgers / "compatibility-candidates.parquet", compression="zstd", statistics=True)
    pl.DataFrame(slice_rows).write_parquet(ledgers / "slice-matrix.parquet", compression="zstd", statistics=True)
    producing_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package_root, text=True).strip()
    artifact = run_dir / "tournament.joblib"
    joblib.dump({
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "producing_commit": producing_commit,
        "models": final_models, "thresholds": thresholds, "research_champion": champion,
        "primary_coverage": primary, "deployment_status": "not_deployed",
    }, artifact, compress=3)
    artifact_sha = _sha256(artifact)
    (run_dir / "tournament.sha256").write_text(f"{artifact_sha}  tournament.joblib\n")
    metrics = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(), "producing_commit": producing_commit,
        "artifact_sha256": artifact_sha, "qualification_status": "research_not_qualified_no_pristine_exact_strategy_cohort",
        "deployment_status": "not_deployed", "strategy_identities": STRATEGIES,
        "training_range": [str(training["window_start"].min()), str(training["window_start"].max())],
        "compatibility_calibration_range": [str(cal["window_start"].min()), str(cal["window_start"].max())],
        "compatibility_test_range": [str(test["window_start"].min()), str(test["window_start"].max())],
        "row_counts": {"training": training.height, "selection_fit": selection_fit.height, "compatibility_calibration": cal.height, "compatibility_test": test.height},
        "feature_usage": {"all_available_causal_dimensions": len(all_features), "by_model": {name: len(features) for name, features in feature_sets.items()}, "excluded_targets": sorted(TARGET_COLUMNS), "excluded_identity": sorted(IDENTITY_COLUMNS)},
        "selection": {"champion": champion, "ranking": ranking, "primary_coverage": primary, "selected_on": "chronological historical walk-forward only"},
        "walk_forward_results": walk_forward, "thresholds": thresholds,
        "compatibility_results": results, "slice_matrix": slice_rows,
        "source_identity": {
            "construction_panel": {"path": str(candidate_run / "checkpoints/construction-panel.parquet"), "sha256": _sha256(candidate_run / "checkpoints/construction-panel.parquet")},
            "rich_ssd_panel": {"path": str(rich_path), "sha256": _sha256(rich_path)},
            "six_strategy_panel": {"path": str(strategy_run / "checkpoints/strategy-panel.parquet"), "sha256": _sha256(strategy_run / "checkpoints/strategy-panel.parquet")},
        },
        "integrity": {"chronological_selection": True, "compatibility_market_disjoint": True, "database_reads": False, "database_mutations": False, "new_tables": False, "new_schemas": False, "new_ingesters": False, "new_sources": False, "images_rebuilt": False},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "polars": pl.__version__, "sklearn": sklearn.__version__},
        "limitations": [
            "The research winner was selected only from chronological historical walk-forward folds; the six-strategy compatibility results did not choose or replace it.",
            "The August 21-25 exact six-strategy cohort was viewed in the preceding tournament, so this run correctly reports it as compatibility evidence rather than a pristine sealed qualification cohort.",
            "Exact candidate ledgers for all six current strategies are unavailable after August 25; no strategy decisions were synthesized from generic execution data.",
            "The SSD rich panel's causal numeric and boolean dimensions were eligible after explicit outcome, target, timestamp, and identity exclusions; individual contestants use the feature subsets recorded in metrics.json.",
            "The archived common candidate construction begins at second 60, so 15-59 evidence is unavailable and retained as empty/insufficient slices.",
            "This is a research-only training run. No runtime model, trading process, database, ingester, data source, schema, or image was changed.",
        ],
    }
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
