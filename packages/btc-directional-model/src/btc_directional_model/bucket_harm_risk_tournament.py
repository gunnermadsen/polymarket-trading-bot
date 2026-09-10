"""Train bucket-conditioned, winner-preserving risk models for six strategy replays."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

import __main__

from .candidate_loss_risk_tournament import _sha256, _write_json
from .entry_action_risk_tournament import (
    EntryActionModel,
    _evaluate,
    _feature_names,
    intervention_metrics,
    threshold_for_coverage,
)
from .strategy_conditioned_risk_tournament import STRATEGIES, EconomicHarmModel

SCHEMA_VERSION = "strategy_conditioned_bucket_harm_risk_tournament_v1"
TRAINING_BUCKETS = ("60_89", "90_119", "120_149", "150_179", "180_240")
NEW_MODELS = (
    "enhanced_economic_harm",
    "bucket_conditioned_harm",
    "winner_preserving_loss",
    "distributional_downside",
    "bucket_regime_harm",
)


def _bucket_expr() -> pl.Expr:
    return (
        pl.when(pl.col("seconds_elapsed") < 90).then(pl.lit("60_89"))
        .when(pl.col("seconds_elapsed") < 120).then(pl.lit("90_119"))
        .when(pl.col("seconds_elapsed") < 150).then(pl.lit("120_149"))
        .when(pl.col("seconds_elapsed") < 180).then(pl.lit("150_179"))
        .otherwise(pl.lit("180_240"))
    )


def attach_bucket_targets(frame: pl.DataFrame, penalty: float) -> pl.DataFrame:
    """Build strictly later action labels without allowing deferral across a time bucket."""
    frame = frame.with_columns(_bucket_expr().alias("time_bucket")).sort(
        ["champion", "market_id", "time_bucket", "seconds_elapsed"]
    )
    group = ["champion", "market_id", "time_bucket"]
    future = pl.col("net_pnl").shift(-1).reverse().cum_max().reverse().over(group).fill_null(0.0)
    return frame.with_columns(future.alias("future_best_within_bucket_pnl")).with_columns(
        pl.max_horizontal(pl.col("future_best_within_bucket_pnl") - penalty, 0.0)
        .alias("within_bucket_defer_value"),
        (pl.col("net_pnl") <= 0).cast(pl.Int8).alias("loss_label"),
        pl.when(pl.col("net_pnl") > 0).then(pl.col("net_pnl")).otherwise(0.0)
        .alias("winner_value"),
        pl.when(pl.col("net_pnl") <= 0).then(-pl.col("net_pnl")).otherwise(0.0)
        .alias("loss_value"),
    )


def _matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return frame.select(pl.col(features).cast(pl.Float32)).to_numpy()


def _estimator(kind: str, seed: int, iterations: int) -> Any:
    common = {
        "learning_rate": 0.07,
        "max_iter": iterations,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 120,
        "l2_regularization": 2.0,
        "max_features": 0.65,
        "early_stopping": True,
        "random_state": seed,
    }
    if kind == "classifier":
        return HistGradientBoostingClassifier(class_weight="balanced", **common)
    if kind == "quantile":
        return HistGradientBoostingRegressor(loss="quantile", quantile=0.25, **common)
    return HistGradientBoostingRegressor(loss="squared_error", **common)


class HarmComponents:
    def __init__(self, features: tuple[str, ...], seed: int, iterations: int):
        self.features = features
        self.loss_probability = _estimator("classifier", seed, iterations)
        self.loss_magnitude = _estimator("regressor", seed + 1, iterations)
        self.win_magnitude = _estimator("regressor", seed + 2, iterations)

    def fit(self, frame: pl.DataFrame) -> HarmComponents:
        matrix = _matrix(frame, self.features)
        self.loss_probability.fit(matrix, frame["loss_label"].to_numpy())
        losses = frame["loss_label"].to_numpy() == 1
        wins = ~losses
        self.loss_magnitude.fit(matrix[losses], frame["loss_value"].to_numpy()[losses])
        self.win_magnitude.fit(matrix[wins], frame["winner_value"].to_numpy()[wins])
        return self

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = _matrix(frame, self.features)
        probability = self.loss_probability.predict_proba(matrix)[:, 1]
        loss = np.clip(self.loss_magnitude.predict(matrix), 0.0, None)
        win = np.clip(self.win_magnitude.predict(matrix), 0.0, None)
        return (1.0 - probability) * win - probability * loss


class BucketHarmModel:
    def __init__(self, features: tuple[str, ...], seed: int, iterations: int):
        self.features = features
        self.seed = seed
        self.iterations = iterations
        self.global_model = HarmComponents(features, seed, iterations)
        self.bucket_models: dict[str, HarmComponents] = {}

    def fit(self, frame: pl.DataFrame) -> BucketHarmModel:
        self.global_model.fit(frame)
        for index, bucket in enumerate(TRAINING_BUCKETS):
            part = frame.filter(pl.col("time_bucket") == bucket)
            if part.height >= 1_000 and part["loss_label"].n_unique() == 2:
                self.bucket_models[bucket] = HarmComponents(
                    self.features, self.seed + 10 * (index + 1), self.iterations
                ).fit(part)
        return self

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        output = self.global_model.score(frame)
        for bucket, model in self.bucket_models.items():
            mask = (frame["time_bucket"] == bucket).to_numpy()
            if mask.any():
                output[mask] = model.score(frame.filter(pl.Series(mask)))
        return output


class WinnerPreservingModel:
    def __init__(self, features: tuple[str, ...], seed: int, iterations: int):
        self.features = features
        self.estimator = _estimator("classifier", seed, iterations)

    def fit(self, frame: pl.DataFrame) -> WinnerPreservingModel:
        pnl = frame["net_pnl"].to_numpy()
        weights = np.where(pnl > 0, 1.5 * np.maximum(pnl, 0.10), np.maximum(-pnl, 0.10))
        self.estimator.fit(
            _matrix(frame, self.features), frame["loss_label"].to_numpy(), sample_weight=weights
        )
        return self

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        return 1.0 - self.estimator.predict_proba(_matrix(frame, self.features))[:, 1]


class DistributionalDownsideModel:
    def __init__(self, features: tuple[str, ...], seed: int, iterations: int):
        self.features = features
        self.estimator = _estimator("quantile", seed, iterations)

    def fit(self, frame: pl.DataFrame) -> DistributionalDownsideModel:
        self.estimator.fit(_matrix(frame, self.features), frame["net_pnl"].to_numpy())
        return self

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        return self.estimator.predict(_matrix(frame, self.features))


def _feature_sets(frame: pl.DataFrame) -> dict[str, tuple[str, ...]]:
    excluded = {
        "future_best_within_bucket_pnl", "within_bucket_defer_value", "winner_value", "loss_value"
    }
    all_features = tuple(
        name for name in _feature_names(frame)
        if name not in excluded and frame[name].drop_nulls().n_unique() >= 2
    )
    economic = tuple(name for name in all_features if name in {
        "seconds_elapsed", "seconds_remaining", "selected_probability", "confidence",
        "share_cost", "fee_per_share", "expected_edge", "probability_change", "edge_change",
        "cost_change", "seconds_since_prior_candidate", "recent_calibration_gap",
    } or name.startswith("history_"))
    regime = tuple(name for name in all_features if name.startswith((
        "btc_", "chainlink_", "binance_", "kraken_", "spot_l2_", "pm_", "oracle_"
    )))
    return {
        "all": all_features,
        "economic": economic,
        "regime": tuple(dict.fromkeys((*economic, *regime))),
    }


def _models(
    feature_sets: dict[str, tuple[str, ...]], seed: int, iterations: int
) -> dict[str, Any]:
    return {
        "enhanced_economic_harm": HarmComponents(feature_sets["all"], seed, iterations),
        "bucket_conditioned_harm": BucketHarmModel(
            feature_sets["economic"], seed + 100, iterations
        ),
        "winner_preserving_loss": WinnerPreservingModel(
            feature_sets["all"], seed + 200, iterations
        ),
        "distributional_downside": DistributionalDownsideModel(
            feature_sets["regime"], seed + 300, iterations
        ),
        "bucket_regime_harm": BucketHarmModel(
            feature_sets["regime"], seed + 400, iterations
        ),
    }


def bucket_thresholds(
    frame: pl.DataFrame, score: np.ndarray, target: float
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    global_threshold = threshold_for_coverage(frame, score, target)
    for bucket in TRAINING_BUCKETS:
        mask = (frame["time_bucket"] == bucket).to_numpy()
        thresholds[bucket] = (
            threshold_for_coverage(frame.filter(pl.Series(mask)), score[mask], target)
            if mask.sum() >= 100 else global_threshold
        )
    return thresholds


def safety_margin(
    frame: pl.DataFrame, score: np.ndarray, thresholds: dict[str, float]
) -> np.ndarray:
    return np.asarray([
        value - thresholds.get(bucket, min(thresholds.values()))
        for value, bucket in zip(score, frame["time_bucket"].to_list(), strict=True)
    ])


def _fold_key(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    return (
        sum(row["net_risk_value"] > 0 for row in rows),
        float(np.median([row["net_risk_value"] for row in rows])),
        float(np.median([(row["risk_alignment_ratio"] or 0.0) for row in rows])),
        -float(np.median([row["opportunity_rejection_rate"] or 0.0 for row in rows])),
    )


def _best_cells(slice_rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pl.DataFrame(slice_rows).filter(
        (pl.col("side") == "ALL") & (pl.col("time_bucket") != "15_59")
    )
    output = {}
    for bucket in TRAINING_BUCKETS:
        rows = frame.filter(pl.col("time_bucket") == bucket).to_dicts()
        supported = [row for row in rows if row["strategy_with_risk"]["trades"] >= 30]
        if not supported:
            output[bucket] = None
            continue
        pnl = max(supported, key=lambda row: row["strategy_with_risk"]["net_pnl"])
        pf = max(
            (row for row in supported if row["strategy_with_risk"]["profit_factor"] is not None),
            key=lambda row: row["strategy_with_risk"]["profit_factor"],
        )
        contribution = max(supported, key=lambda row: row["net_risk_value"])
        output[bucket] = {
            "pnl_champion": pnl, "profit_factor_champion": pf,
            "risk_contribution_champion": contribution,
        }
    return output


def _report(metrics: dict[str, Any]) -> str:
    def n(value: Any, digits: int = 2) -> str:
        return "—" if value is None else f"{value:.{digits}f}"
    lines = [
        "# Bucket-Conditioned Harm Risk Tournament", "",
        "Every PnL, PF, W/L, recovery and drawdown value is a strategy replay with the named risk model applied. The risk model itself does not own standalone trading returns.", "",
        "## Historical bucket selections", "",
    ]
    for bucket, champion in metrics["selection"]["bucket_champions"].items():
        lines.append(f"- {bucket}: `{champion}`")
    lines += [
        "", "## Exact six-strategy compatibility at primary coverage", "",
        "| Risk model | PnL with risk | Delta PnL | Trades | W/L | Coverage | PF | Recovery | Max DD | Bad blocked | Good blocked | Alignment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    primary = str(metrics["selection"]["primary_coverage"])
    for name, bands in metrics["compatibility_results"].items():
        band = "1.0" if name == "no_risk" else primary
        evidence = bands[band]["aggregate"]
        risk = evidence["strategy_with_risk"]
        lines.append(
            f"| {name} | ${risk['net_pnl']:.2f} | ${evidence['net_risk_value']:.2f} | "
            f"{risk['trades']} | {risk['wins']}/{risk['losses']} | {risk['coverage_retained']:.1%} | "
            f"{n(risk['profit_factor'],3)} | {n(risk['recovery_wins_per_loss'],3)} | "
            f"${risk['max_drawdown']:.2f} | {n((evidence['loss_capture_rate'] or 0)*100,1)}% | "
            f"{n((evidence['opportunity_rejection_rate'] or 0)*100,1)}% | "
            f"{n(evidence['risk_alignment_ratio'],2)}x |"
        )
    lines += ["", "## Strategy × risk model primary-band matrix", "", "| Risk model | Strategy model | PnL | Delta | Trades | W/L | Coverage | Avg entry | Avg cost | PF | Recovery | Max DD | Alignment |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, bands in metrics["compatibility_results"].items():
        band = "1.0" if name == "no_risk" else primary
        for strategy, evidence in bands[band]["by_strategy"].items():
            risk = evidence["strategy_with_risk"]
            lines.append(
                f"| {name} | {STRATEGIES[strategy]['process_name']} | ${risk['net_pnl']:.2f} | "
                f"${evidence['net_risk_value']:.2f} | {risk['trades']} | {risk['wins']}/{risk['losses']} | "
                f"{risk['coverage_retained']:.1%} | {n(risk['average_entry_seconds'],1)}s | "
                f"${n(risk['average_cost'],3)} | {n(risk['profit_factor'],3)} | "
                f"{n(risk['recovery_wins_per_loss'],3)} | ${risk['max_drawdown']:.2f} | "
                f"{n(evidence['risk_alignment_ratio'],2)}x |"
            )
    lines += [
        "", "## Granular champions", "",
        "Champion identity is risk model + strategy model + time bucket + side + coverage policy. The full supported and sparse cell inventory is in `ledgers/slice-matrix.parquet`.", "",
        "## Forward readiness", "",
        (
            f"September audit status: **{metrics['forward_audit']['status']}**. "
            f"The six processes produced "
            f"{sum(metrics['forward_audit']['process_decision_counts'])} decisions but only "
            f"{sum(metrics['forward_audit']['process_buy_counts'])} admitted buys; one process "
            "admitted zero. These rows were not repurposed into synthetic risk candidates."
        ), "",
        "## Integrity and limitations", "",
    ]
    lines.extend(f"- {item}" for item in metrics["limitations"])
    return "\n".join(lines) + "\n"


def train_tournament(config_path: Path, resume_run: str | None = None) -> Path:
    raw = tomllib.loads(config_path.read_text())
    package_root = config_path.resolve().parents[1]
    entry_run = package_root / raw["data"]["entry_action_run"]
    candidate_run = package_root / raw["data"]["candidate_risk_run"]
    strategy_run = package_root / raw["data"]["strategy_risk_run"]
    output_root = package_root / raw["output"]["directory"]
    run_id = resume_run or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    training_path = checkpoints / "training-panel.parquet"
    compatibility_path = checkpoints / "strategy-panel.parquet"
    penalty = float(raw["training"]["deferral_penalty"])
    if training_path.exists():
        training = pl.read_parquet(training_path)
    else:
        training = attach_bucket_targets(
            pl.read_parquet(entry_run / "checkpoints/training-panel.parquet"), penalty
        )
        training.write_parquet(training_path, compression="zstd", statistics=True)
    if compatibility_path.exists():
        compatibility = pl.read_parquet(compatibility_path)
    else:
        compatibility = attach_bucket_targets(
            pl.read_parquet(entry_run / "checkpoints/strategy-panel.parquet"), penalty
        )
        compatibility.write_parquet(compatibility_path, compression="zstd", statistics=True)

    parse = datetime.fromisoformat
    fit_end = parse(raw["splits"]["selection_fit_end"])
    boundaries = [parse(value) for value in raw["splits"]["walk_forward_boundaries"]]
    selection_fit = training.filter(pl.col("window_start") < fit_end)
    feature_sets = _feature_sets(selection_fit)
    seed = int(raw["training"]["random_seed"])
    iterations = int(raw["training"]["maximum_iterations"])
    selection_checkpoint = checkpoints / "selection-models.joblib"
    if selection_checkpoint.exists():
        selection_models = joblib.load(selection_checkpoint)
    else:
        selection_models = _models(feature_sets, seed, iterations)
        for name, model in selection_models.items():
            model.fit(selection_fit)
            joblib.dump(model, checkpoints / f"selection-{name}.joblib", compress=3)
        joblib.dump(selection_models, selection_checkpoint, compress=3)

    primary = float(raw["training"]["primary_coverage"])
    walk_forward: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {bucket: [] for bucket in TRAINING_BUCKETS} for name in NEW_MODELS
    }
    for name, model in selection_models.items():
        for index in range(len(boundaries) - 2):
            calibration = training.filter(pl.col("window_start").is_between(
                boundaries[index], boundaries[index + 1], closed="left"
            ))
            test = training.filter(pl.col("window_start").is_between(
                boundaries[index + 1], boundaries[index + 2], closed="left"
            ))
            cal_score, test_score = model.score(calibration), model.score(test)
            thresholds = bucket_thresholds(calibration, cal_score, primary)
            margin = safety_margin(test, test_score, thresholds)
            for bucket in TRAINING_BUCKETS:
                mask = (test["time_bucket"] == bucket).to_numpy()
                evidence = intervention_metrics(test.filter(pl.Series(mask)), margin[mask], 0.0)
                walk_forward[name][bucket].append(evidence)
    bucket_champions = {
        bucket: max(NEW_MODELS, key=lambda name: _fold_key(walk_forward[name][bucket]))
        for bucket in TRAINING_BUCKETS
    }
    selection = {
        "bucket_champions": bucket_champions,
        "primary_coverage": primary,
        "selected_on": "chronological historical walk-forward only",
    }
    _write_json(run_dir / "selection-freeze.json", selection)

    final_checkpoint = checkpoints / "final-models.joblib"
    if final_checkpoint.exists():
        final_models = joblib.load(final_checkpoint)
    else:
        final_models = _models(feature_sets, seed, iterations)
        for name, model in final_models.items():
            model.fit(training)
            joblib.dump(model, checkpoints / f"final-{name}.joblib", compress=3)
        joblib.dump(final_models, final_checkpoint, compress=3)

    cal_start = parse(raw["splits"]["compatibility_calibration_start"])
    test_start = parse(raw["splits"]["compatibility_test_start"])
    test_end = parse(raw["splits"]["compatibility_test_end"])
    calibration = compatibility.filter(pl.col("window_start").is_between(
        cal_start, test_start, closed="left"
    ))
    test = compatibility.filter(pl.col("window_start").is_between(
        test_start, test_end, closed="left"
    ))
    if not set(calibration["market_id"]).isdisjoint(set(test["market_id"])):
        raise RuntimeError("market leakage between compatibility cohorts")
    if any(test.filter(pl.col("champion") == strategy).is_empty() for strategy in STRATEGIES):
        raise RuntimeError("compatibility test does not cover all six strategies")

    coverage_targets = tuple(float(value) for value in raw["training"]["coverage_targets"])
    results: dict[str, Any] = {}
    threshold_manifest: dict[str, Any] = {}
    slice_rows: list[dict[str, Any]] = []
    scored = test
    for name, model in final_models.items():
        cal_score, test_score = model.score(calibration), model.score(test)
        scored = scored.with_columns(pl.Series(f"safety_score__{name}", test_score))
        results[name], threshold_manifest[name] = {}, {}
        for target in coverage_targets:
            thresholds = bucket_thresholds(calibration, cal_score, target)
            threshold_manifest[name][str(target)] = thresholds
            margin = safety_margin(test, test_score, thresholds)
            evidence, slices = _evaluate(test, margin, 0.0, name, target)
            results[name][str(target)] = evidence
            slice_rows.extend(slices)

    # Previous tournament models and simple economics remain non-selectable controls.
    if __name__ == "__main__":
        canonical = "btc_directional_model.entry_action_risk_tournament"
        sys.modules[canonical] = sys.modules[EntryActionModel.__module__]
    prior = joblib.load(entry_run / "tournament.joblib")
    control_models = {"previous_regime_conditioned_action": prior["models"]["regime_conditioned_action"]}
    __main__.EconomicHarmModel = EconomicHarmModel
    prior_loss = joblib.load(strategy_run / "tournament.joblib")
    construction = pl.read_parquet(candidate_run / "checkpoints/construction-panel.parquet")
    mean_loss = float(-construction.filter(pl.col("net_pnl") <= 0)["net_pnl"].mean())
    mean_win = float(construction.filter(pl.col("net_pnl") > 0)["net_pnl"].mean())
    control_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "matched_confidence": (calibration["confidence"].to_numpy(), test["confidence"].to_numpy()),
        "matched_edge": (calibration["expected_edge"].to_numpy(), test["expected_edge"].to_numpy()),
    }
    for name, model in control_models.items():
        control_scores[name] = (model.score(calibration), model.score(test))
    for name, source in (
        ("frozen_logistic_loss_risk", "logistic_economics"),
        ("frozen_boosted_history_loss_risk", "boosted_recent_history"),
    ):
        model = prior_loss["models"][source]
        def score(frame: pl.DataFrame, estimator: Any = model) -> np.ndarray:
            probability = estimator.predict_proba(frame)
            return -(
                probability * mean_loss - (1.0 - probability) * mean_win - penalty
            )
        control_scores[name] = (score(calibration), score(test))
    for name, (cal_score, test_score) in control_scores.items():
        results[name], threshold_manifest[name] = {}, {}
        for target in coverage_targets:
            thresholds = bucket_thresholds(calibration, cal_score, target)
            threshold_manifest[name][str(target)] = thresholds
            margin = safety_margin(test, test_score, thresholds)
            evidence, slices = _evaluate(test, margin, 0.0, name, target)
            results[name][str(target)] = evidence
            slice_rows.extend(slices)
    no_risk_limit = -float(np.finfo(np.float64).max)
    no_risk, no_risk_slices = _evaluate(
        test, np.ones(test.height), no_risk_limit, "no_risk", 1.0
    )
    results["no_risk"] = {"1.0": no_risk}
    threshold_manifest["no_risk"] = {"1.0": None}
    slice_rows.extend(no_risk_slices)

    ledgers = run_dir / "ledgers"
    ledgers.mkdir(exist_ok=True)
    scored.write_parquet(ledgers / "compatibility-candidates.parquet", compression="zstd")
    pl.DataFrame(slice_rows).write_parquet(ledgers / "slice-matrix.parquet", compression="zstd")
    producing_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package_root, text=True).strip()
    if __name__ == "__main__":
        canonical = "btc_directional_model.bucket_harm_risk_tournament"
        sys.modules[canonical] = sys.modules[__name__]
        for cls in (HarmComponents, BucketHarmModel, WinnerPreservingModel, DistributionalDownsideModel):
            cls.__module__ = canonical
    artifact = run_dir / "tournament.joblib"
    joblib.dump({
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "models": final_models,
        "thresholds": threshold_manifest, "bucket_champions": bucket_champions,
        "producing_commit": producing_commit, "deployment_status": "not_deployed",
    }, artifact, compress=3)
    artifact_sha = _sha256(artifact)
    (run_dir / "tournament.sha256").write_text(f"{artifact_sha}  tournament.joblib\n")
    forward = {
        **raw["forward_audit"],
        "status": "insufficient_for_six_strategy_post_admission_replay",
        "all_six_processes_observed": True,
        "all_six_have_admitted_candidates": False,
        "database_mutations": False,
    }
    metrics = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(), "producing_commit": producing_commit,
        "artifact_sha256": artifact_sha,
        "qualification_status": "research_not_qualified_no_pristine_complete_six_strategy_cohort",
        "deployment_status": "not_deployed", "strategy_identities": STRATEGIES,
        "selection": selection, "walk_forward_results": walk_forward,
        "compatibility_results": results, "thresholds": threshold_manifest,
        "slice_matrix": slice_rows, "granular_champions": _best_cells(slice_rows),
        "feature_usage": {
            "eligible_causal_dimensions": len(feature_sets["all"]),
            "by_family": {name: len(features) for name, features in feature_sets.items()},
            "strategy_identity_predictive_feature": False,
        },
        "row_counts": {
            "training": training.height, "selection_fit": selection_fit.height,
            "compatibility_calibration": calibration.height, "compatibility_test": test.height,
        },
        "ranges": {
            "training": [str(training["window_start"].min()), str(training["window_start"].max())],
            "compatibility_calibration": [str(calibration["window_start"].min()), str(calibration["window_start"].max())],
            "compatibility_test": [str(test["window_start"].min()), str(test["window_start"].max())],
        },
        "forward_audit": forward,
        "source_identity": {
            "training_panel": {"path": str(entry_run / "checkpoints/training-panel.parquet"), "sha256": _sha256(entry_run / "checkpoints/training-panel.parquet")},
            "six_strategy_panel": {"path": str(entry_run / "checkpoints/strategy-panel.parquet"), "sha256": _sha256(entry_run / "checkpoints/strategy-panel.parquet")},
        },
        "integrity": {
            "chronological_selection": True, "compatibility_market_disjoint": True,
            "within_bucket_deferral_labels": True, "database_reads": True,
            "database_mutations": False, "new_tables": False, "new_schemas": False,
            "new_ingesters": False, "new_sources": False, "images_rebuilt": False,
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "polars": pl.__version__, "sklearn": sklearn.__version__},
        "limitations": [
            "The August 21-25 exact six-strategy cohort was already viewed and is compatibility evidence, not a sealed qualification cohort.",
            "Five deployed UMR processes were created September 7; the September 7-9 audit has only 65 admitted buys across those five and zero for one process, so it cannot fairly qualify all six post-admission risk pairings.",
            "Rejected live strategy decisions were not converted into synthetic admitted candidates.",
            "The full established April-August causal OOF construction range was used; all available numeric and boolean SSD dimensions were eligible without strategy identity.",
            "The common historical construction begins at second 60, so 15-59 remains unavailable.",
            "This is research-only. No runtime, process, database, ingester, data source, schema, image or deployment was changed.",
        ],
    }
    _write_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(_report(metrics))
    _write_json(run_dir / "completion.json", {
        "completed": True, "artifact_sha256": artifact_sha,
        "metrics_sha256": _sha256(run_dir / "metrics.json"),
    })
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-run")
    args = parser.parse_args()
    print(train_tournament(args.config, args.resume_run))


if __name__ == "__main__":
    main()
