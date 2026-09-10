"""Train universal consensus and sequential action-value risk models."""

from __future__ import annotations

import argparse
import math
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

from .bucket_harm_risk_tournament import (
    TRAINING_BUCKETS,
    BucketHarmModel,
    _best_cells,
    _bucket_expr,
    _matrix,
    bucket_thresholds,
    safety_margin,
)
from .candidate_loss_risk_tournament import _sha256, _write_json
from .entry_action_risk_tournament import (
    _feature_names,
    _selected,
    intervention_metrics,
)
from .strategy_conditioned_risk_tournament import BUCKETS, STRATEGIES, EconomicHarmModel

SCHEMA_VERSION = "strategy_consensus_action_value_risk_tournament_v1"
NEW_MODELS = (
    "expected_action_value",
    "distributional_action_value",
    "consensus_disagreement_value",
    "pairwise_optimal_stopping",
    "conformal_action_value",
)
TARGETS = {
    "future_best_within_bucket_pnl",
    "wait_value",
    "enter_advantage",
    "enter_is_optimal",
    "optimal_action",
    "loss_value",
    "winner_value",
}
CONSENSUS_FEATURES = (
    "consensus_strategy_count",
    "consensus_up_share",
    "consensus_probability_mean",
    "consensus_probability_std",
    "consensus_probability_range",
    "consensus_confidence_mean",
    "consensus_confidence_std",
    "consensus_edge_mean",
    "consensus_edge_std",
    "strategy_probability_vs_consensus",
    "strategy_confidence_vs_consensus",
    "strategy_edge_vs_consensus",
    "strategy_side_consensus",
    "strategy_probability_outlier_z",
    "strategy_edge_outlier_z",
)


def attach_action_value_targets(frame: pl.DataFrame, penalty: float) -> pl.DataFrame:
    """Attach enter/wait/abstain values using only strictly later same-bucket rows."""
    frame = frame.with_columns(_bucket_expr().alias("time_bucket")).sort(
        ["champion", "market_id", "time_bucket", "seconds_elapsed"]
    )
    group = ["champion", "market_id", "time_bucket"]
    future = pl.col("net_pnl").shift(-1).reverse().cum_max().reverse().over(group).fill_null(0.0)
    frame = frame.with_columns(future.alias("future_best_within_bucket_pnl"))
    frame = frame.with_columns(
        pl.max_horizontal(pl.col("future_best_within_bucket_pnl") - penalty, 0.0).alias(
            "wait_value"
        )
    )
    return frame.with_columns(
        (pl.col("net_pnl") - pl.col("wait_value")).alias("enter_advantage"),
        (pl.col("net_pnl") >= pl.col("wait_value")).cast(pl.Int8).alias("enter_is_optimal"),
        pl.when(pl.col("net_pnl") >= pl.col("wait_value"))
        .then(pl.lit("enter_now"))
        .when(pl.col("wait_value") > 0)
        .then(pl.lit("wait"))
        .otherwise(pl.lit("abstain"))
        .alias("optimal_action"),
        pl.when(pl.col("net_pnl") <= 0).then(-pl.col("net_pnl")).otherwise(0.0).alias("loss_value"),
        pl.when(pl.col("net_pnl") > 0).then(pl.col("net_pnl")).otherwise(0.0).alias("winner_value"),
    )


def attach_consensus_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive causal same-timestamp strategy agreement without using strategy identity."""
    keys = ["market_id", "window_start", "seconds_elapsed"]
    base = frame.with_columns((pl.col("side") == "UP").cast(pl.Float64).alias("side_is_up"))
    consensus = base.group_by(keys).agg(
        pl.len().cast(pl.Float64).alias("consensus_strategy_count"),
        pl.col("side_is_up").mean().alias("consensus_up_share"),
        pl.col("selected_probability").mean().alias("consensus_probability_mean"),
        pl.col("selected_probability")
        .std(ddof=0)
        .fill_null(0.0)
        .alias("consensus_probability_std"),
        (pl.col("selected_probability").max() - pl.col("selected_probability").min()).alias(
            "consensus_probability_range"
        ),
        pl.col("confidence").mean().alias("consensus_confidence_mean"),
        pl.col("confidence").std(ddof=0).fill_null(0.0).alias("consensus_confidence_std"),
        pl.col("expected_edge").mean().alias("consensus_edge_mean"),
        pl.col("expected_edge").std(ddof=0).fill_null(0.0).alias("consensus_edge_std"),
    )
    return base.join(consensus, on=keys, how="left").with_columns(
        (pl.col("selected_probability") - pl.col("consensus_probability_mean")).alias(
            "strategy_probability_vs_consensus"
        ),
        (pl.col("confidence") - pl.col("consensus_confidence_mean")).alias(
            "strategy_confidence_vs_consensus"
        ),
        (pl.col("expected_edge") - pl.col("consensus_edge_mean")).alias(
            "strategy_edge_vs_consensus"
        ),
        pl.when(pl.col("side_is_up") > 0.5)
        .then(pl.col("consensus_up_share"))
        .otherwise(1.0 - pl.col("consensus_up_share"))
        .alias("strategy_side_consensus"),
        (
            (pl.col("selected_probability") - pl.col("consensus_probability_mean"))
            / pl.col("consensus_probability_std").clip(1e-6, None)
        ).alias("strategy_probability_outlier_z"),
        (
            (pl.col("expected_edge") - pl.col("consensus_edge_mean"))
            / pl.col("consensus_edge_std").clip(1e-6, None)
        ).alias("strategy_edge_outlier_z"),
    )


def build_panel(frame: pl.DataFrame, penalty: float) -> pl.DataFrame:
    return attach_consensus_features(attach_action_value_targets(frame, penalty))


def _regressor(seed: int, iterations: int, loss: str = "squared_error") -> Any:
    options: dict[str, Any] = {
        "loss": loss,
        "learning_rate": 0.065,
        "max_iter": iterations,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 120,
        "l2_regularization": 2.0,
        "max_features": 0.70,
        "early_stopping": True,
        "random_state": seed,
    }
    if loss == "quantile":
        options["quantile"] = 0.25
    return HistGradientBoostingRegressor(**options)


class ActionValueModel:
    def __init__(
        self,
        features: tuple[str, ...],
        seed: int,
        iterations: int,
        *,
        enter_loss: str = "squared_error",
    ):
        self.features = features
        self.enter = _regressor(seed, iterations, enter_loss)
        self.wait = _regressor(seed + 1, iterations)

    def fit(self, frame: pl.DataFrame) -> ActionValueModel:
        matrix = _matrix(frame, self.features)
        self.enter.fit(matrix, frame["net_pnl"].to_numpy())
        self.wait.fit(matrix, frame["wait_value"].to_numpy())
        return self

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = _matrix(frame, self.features)
        return self.enter.predict(matrix) - np.maximum(self.wait.predict(matrix), 0.0)


class PairwiseStoppingModel:
    def __init__(self, features: tuple[str, ...], seed: int, iterations: int):
        self.features = features
        self.estimator = HistGradientBoostingClassifier(
            learning_rate=0.065,
            max_iter=iterations,
            max_leaf_nodes=15,
            min_samples_leaf=120,
            l2_regularization=2.0,
            max_features=0.70,
            class_weight="balanced",
            early_stopping=True,
            random_state=seed,
        )

    def fit(self, frame: pl.DataFrame) -> PairwiseStoppingModel:
        advantage = frame["enter_advantage"].to_numpy()
        weights = np.maximum(np.abs(advantage), 0.05)
        self.estimator.fit(
            _matrix(frame, self.features),
            frame["enter_is_optimal"].to_numpy(),
            sample_weight=weights,
        )
        return self

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(_matrix(frame, self.features))[:, 1]


class ConformalActionValueModel:
    def __init__(
        self,
        features: tuple[str, ...],
        seed: int,
        iterations: int,
        calibration_fraction: float,
        error_quantile: float,
    ):
        self.features = features
        self.seed = seed
        self.iterations = iterations
        self.calibration_fraction = calibration_fraction
        self.error_quantile = error_quantile
        self.model = ActionValueModel(features, seed, iterations)
        self.bucket_error: dict[str, float] = {}
        self.global_error = 0.0

    def fit(self, frame: pl.DataFrame) -> ConformalActionValueModel:
        times = frame["window_start"].unique().sort()
        cut_index = max(1, min(len(times) - 1, int(len(times) * (1.0 - self.calibration_fraction))))
        cutoff = times[cut_index]
        fit = frame.filter(pl.col("window_start") < cutoff)
        calibration = frame.filter(pl.col("window_start") >= cutoff)
        self.model.fit(fit)
        residual = np.abs(calibration["enter_advantage"].to_numpy() - self.model.score(calibration))
        self.global_error = float(np.quantile(residual, self.error_quantile))
        for bucket in TRAINING_BUCKETS:
            mask = (calibration["time_bucket"] == bucket).to_numpy()
            self.bucket_error[bucket] = (
                float(np.quantile(residual[mask], self.error_quantile))
                if mask.sum() >= 100
                else self.global_error
            )
        return self

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        raw = self.model.score(frame)
        penalty = np.asarray(
            [
                self.bucket_error.get(bucket, self.global_error)
                for bucket in frame["time_bucket"].to_list()
            ]
        )
        return raw - penalty


def _feature_sets(frame: pl.DataFrame) -> dict[str, tuple[str, ...]]:
    all_features = tuple(
        name
        for name in _feature_names(frame)
        if name not in TARGETS and frame[name].drop_nulls().n_unique() >= 2
    )
    economic = tuple(
        name
        for name in all_features
        if name
        in {
            "seconds_elapsed",
            "seconds_remaining",
            "selected_probability",
            "confidence",
            "share_cost",
            "fee_per_share",
            "expected_edge",
            "probability_change",
            "edge_change",
            "cost_change",
            "seconds_since_prior_candidate",
            "recent_calibration_gap",
            "side_is_up",
        }
        or name.startswith("history_")
    )
    consensus = tuple(name for name in all_features if name in CONSENSUS_FEATURES)
    regime = tuple(
        name
        for name in all_features
        if name.startswith(
            ("btc_", "chainlink_", "binance_", "kraken_", "spot_l2_", "pm_", "oracle_")
        )
    )
    return {
        "all": all_features,
        "economic_consensus": tuple(dict.fromkeys((*economic, *consensus))),
        "consensus_regime": tuple(dict.fromkeys((*economic, *consensus, *regime))),
    }


def _models(feature_sets: dict[str, tuple[str, ...]], raw: dict[str, Any]) -> dict[str, Any]:
    seed = int(raw["training"]["random_seed"])
    iterations = int(raw["training"]["maximum_iterations"])
    compact = feature_sets["economic_consensus"]
    rich = feature_sets["consensus_regime"]
    return {
        "expected_action_value": ActionValueModel(feature_sets["all"], seed, iterations),
        "distributional_action_value": ActionValueModel(
            rich, seed + 100, iterations, enter_loss="quantile"
        ),
        "consensus_disagreement_value": ActionValueModel(compact, seed + 200, iterations),
        "pairwise_optimal_stopping": PairwiseStoppingModel(rich, seed + 300, iterations),
        "conformal_action_value": ConformalActionValueModel(
            rich,
            seed + 400,
            iterations,
            float(raw["training"]["conformal_calibration_fraction"]),
            float(raw["training"]["conformal_error_quantile"]),
        ),
    }


def _action_diagnostics(frame: pl.DataFrame, score: np.ndarray, threshold: float) -> dict[str, Any]:
    baseline = _selected(frame, np.ones(frame.height), -math.inf)
    risk = _selected(frame, score, threshold)
    selected = {(r["champion"], r["market_id"]): r for r in risk.iter_rows(named=True)}
    losses = baseline.filter(pl.col("net_pnl") <= 0)
    catastrophic = 0
    captured = 0
    if losses.height:
        cutoff = float(np.quantile(losses["net_pnl"].to_numpy(), 0.10))
        severe = losses.filter(pl.col("net_pnl") <= cutoff)
        catastrophic = severe.height
        for row in severe.iter_rows(named=True):
            chosen = selected.get((row["champion"], row["market_id"]))
            if chosen is None or chosen["observed_at"] != row["observed_at"]:
                captured += 1
    base = intervention_metrics(frame, score, threshold)
    coverage = base["strategy_with_risk"]["coverage_retained"]
    surrendered = 1.0 - coverage if coverage is not None else None
    advantage = frame["enter_advantage"].to_numpy()
    finite = np.isfinite(score) & np.isfinite(advantage)
    correlation = (
        float(np.corrcoef(score[finite], advantage[finite])[0, 1])
        if finite.sum() >= 3 and np.std(score[finite]) > 0 and np.std(advantage[finite]) > 0
        else None
    )
    return {
        "catastrophic_losses": catastrophic,
        "catastrophic_losses_blocked": captured,
        "catastrophic_loss_capture_rate": captured / catastrophic if catastrophic else None,
        "pnl_delta_per_coverage_point_surrendered": (
            base["net_risk_value"] / (100.0 * surrendered)
            if surrendered is not None and surrendered > 0
            else None
        ),
        "score_enter_advantage_correlation": correlation,
        "wait_capture_rate": base["deferral_improvement_rate"],
    }


def _evaluate_action(
    frame: pl.DataFrame,
    score: np.ndarray,
    threshold: float,
    policy: str,
    coverage: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate = intervention_metrics(frame, score, threshold)
    aggregate.update(_action_diagnostics(frame, score, threshold))
    by_strategy: dict[str, Any] = {}
    slices: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        mask = (frame["champion"] == strategy).to_numpy()
        part, local = frame.filter(pl.Series(mask)), score[mask]
        evidence = intervention_metrics(part, local, threshold)
        evidence.update(_action_diagnostics(part, local, threshold))
        by_strategy[strategy] = evidence
        for bucket in BUCKETS:
            for side in ("ALL", "UP", "DOWN"):
                cell_mask = (part["time_bucket"] == bucket).to_numpy()
                if side != "ALL":
                    cell_mask &= (part["side"] == side).to_numpy()
                cell, cell_score = part.filter(pl.Series(cell_mask)), local[cell_mask]
                cell_evidence = intervention_metrics(cell, cell_score, threshold)
                cell_evidence.update(_action_diagnostics(cell, cell_score, threshold))
                slices.append(
                    {
                        "risk_policy": policy,
                        "coverage_target": coverage,
                        "strategy_model": strategy,
                        "time_bucket": bucket,
                        "side": side,
                        "statistically_insufficient": cell_evidence["baseline"]["trades"] < 30,
                        **cell_evidence,
                    }
                )
        for side in ("UP", "DOWN"):
            cell_mask = (part["side"] == side).to_numpy()
            cell, cell_score = part.filter(pl.Series(cell_mask)), local[cell_mask]
            cell_evidence = intervention_metrics(cell, cell_score, threshold)
            cell_evidence.update(_action_diagnostics(cell, cell_score, threshold))
            slices.append(
                {
                    "risk_policy": policy,
                    "coverage_target": coverage,
                    "strategy_model": strategy,
                    "time_bucket": "ALL",
                    "side": side,
                    "statistically_insufficient": cell_evidence["baseline"]["trades"] < 30,
                    **cell_evidence,
                }
            )
    return {"threshold": threshold, "aggregate": aggregate, "by_strategy": by_strategy}, slices


def _fold_key(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    return (
        sum(row["net_risk_value"] > 0 for row in rows),
        float(np.median([row["net_risk_value"] for row in rows])),
        float(np.median([row["strategy_with_risk"]["net_pnl"] for row in rows])),
        -float(np.median([row["action_regret"] for row in rows])),
    )


def _report(metrics: dict[str, Any]) -> str:
    def n(value: Any, digits: int = 2) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Strategy-Consensus Action-Value Risk Tournament",
        "",
        "PnL and trading metrics describe each strategy replay after the named risk policy; the risk model has no standalone trading return.",
        "",
        "## Historical bucket selections",
        "",
    ]
    for bucket, champion in metrics["selection"]["bucket_champions"].items():
        lines.append(f"- {bucket}: `{champion}`")
    lines += [
        "",
        "## Exact six-strategy compatibility at primary coverage",
        "",
        "| Risk policy | Risk PnL | Delta | Stress PnL | Trades | W/L | Coverage | PF | Recovery | Max DD | Bad blocked | Good blocked | Alignment | Wait capture | Catastrophic capture | Portable strategies |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    primary = str(metrics["selection"]["primary_coverage"])
    for name, bands in metrics["compatibility_results"].items():
        band = "1.0" if name == "no_risk" else primary
        evidence = bands[band]["aggregate"]
        risk = evidence["strategy_with_risk"]
        portable = bands[band]["strategy_portability"]
        lines.append(
            f"| {name} | ${risk['net_pnl']:.2f} | ${evidence['net_risk_value']:.2f} | "
            f"${risk['stress_net_pnl']:.2f} | {risk['trades']} | {risk['wins']}/{risk['losses']} | "
            f"{risk['coverage_retained']:.1%} | {n(risk['profit_factor'], 3)} | "
            f"{n(risk['recovery_wins_per_loss'], 3)} | ${risk['max_drawdown']:.2f} | "
            f"{n((evidence['loss_capture_rate'] or 0) * 100, 1)}% | "
            f"{n((evidence['opportunity_rejection_rate'] or 0) * 100, 1)}% | "
            f"{n(evidence['risk_alignment_ratio'], 2)}x | "
            f"{n(evidence['wait_capture_rate'] * 100 if evidence['wait_capture_rate'] is not None else None, 1)}% | "
            f"{n(evidence['catastrophic_loss_capture_rate'] * 100 if evidence['catastrophic_loss_capture_rate'] is not None else None, 1)}% | "
            f"{portable}/6 |"
        )
    lines += [
        "",
        "## Granular champions",
        "",
        "Champion identity is risk policy + strategy model + time bucket + side + coverage policy. Complete supported and sparse cells are retained in `ledgers/slice-matrix.parquet` and `metrics.json`.",
        "",
        "## Data integrity and limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in metrics["limitations"])
    return "\n".join(lines) + "\n"


def train_tournament(config_path: Path, resume_run: str | None = None) -> Path:
    raw = tomllib.loads(config_path.read_text())
    package_root = config_path.resolve().parents[1]
    entry_run = package_root / raw["data"]["entry_action_run"]
    bucket_run = package_root / raw["data"]["bucket_harm_run"]
    candidate_run = package_root / raw["data"]["candidate_risk_run"]
    strategy_run = package_root / raw["data"]["strategy_risk_run"]
    run_id = resume_run or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = package_root / raw["output"]["directory"] / run_id
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    penalty = float(raw["training"]["deferral_penalty"])

    training_path = checkpoints / "training-panel.parquet"
    if training_path.exists():
        training = pl.read_parquet(training_path)
    else:
        training = build_panel(
            pl.read_parquet(entry_run / "checkpoints/training-panel.parquet"), penalty
        )
        training.write_parquet(training_path, compression="zstd", statistics=True)
    compatibility_path = checkpoints / "strategy-panel.parquet"
    if compatibility_path.exists():
        compatibility = pl.read_parquet(compatibility_path)
    else:
        compatibility = build_panel(
            pl.read_parquet(entry_run / "checkpoints/strategy-panel.parquet"), penalty
        )
        compatibility.write_parquet(compatibility_path, compression="zstd", statistics=True)

    parse = datetime.fromisoformat
    fit_end = parse(raw["splits"]["selection_fit_end"])
    boundaries = [parse(value) for value in raw["splits"]["walk_forward_boundaries"]]
    selection_fit = training.filter(pl.col("window_start") < fit_end)
    feature_sets = _feature_sets(selection_fit)
    selection_checkpoint = checkpoints / "selection-models.joblib"
    if selection_checkpoint.exists():
        selection_models = joblib.load(selection_checkpoint)
    else:
        selection_models = _models(feature_sets, raw)
        for name, model in selection_models.items():
            model.fit(selection_fit)
            joblib.dump(model, checkpoints / f"selection-{name}.joblib", compress=3)
        joblib.dump(selection_models, selection_checkpoint, compress=3)

    primary = float(raw["training"]["primary_coverage"])
    walk_forward = {name: {bucket: [] for bucket in TRAINING_BUCKETS} for name in NEW_MODELS}
    for name, model in selection_models.items():
        for index in range(len(boundaries) - 2):
            calibration = training.filter(
                pl.col("window_start").is_between(
                    boundaries[index], boundaries[index + 1], closed="left"
                )
            )
            test = training.filter(
                pl.col("window_start").is_between(
                    boundaries[index + 1], boundaries[index + 2], closed="left"
                )
            )
            cal_score, test_score = model.score(calibration), model.score(test)
            thresholds = bucket_thresholds(calibration, cal_score, primary)
            margin = safety_margin(test, test_score, thresholds)
            for bucket in TRAINING_BUCKETS:
                mask = (test["time_bucket"] == bucket).to_numpy()
                walk_forward[name][bucket].append(
                    intervention_metrics(test.filter(pl.Series(mask)), margin[mask], 0.0)
                )
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
        final_models = _models(feature_sets, raw)
        for name, model in final_models.items():
            model.fit(training)
            joblib.dump(model, checkpoints / f"final-{name}.joblib", compress=3)
        joblib.dump(final_models, final_checkpoint, compress=3)

    cal_start = parse(raw["splits"]["compatibility_calibration_start"])
    test_start = parse(raw["splits"]["compatibility_test_start"])
    test_end = parse(raw["splits"]["compatibility_test_end"])
    calibration = compatibility.filter(
        pl.col("window_start").is_between(cal_start, test_start, closed="left")
    )
    test = compatibility.filter(
        pl.col("window_start").is_between(test_start, test_end, closed="left")
    )
    if not set(calibration["market_id"]).isdisjoint(set(test["market_id"])):
        raise RuntimeError("market leakage between compatibility cohorts")
    if any(test.filter(pl.col("champion") == strategy).is_empty() for strategy in STRATEGIES):
        raise RuntimeError("compatibility test does not cover all six strategies")

    coverage_targets = tuple(float(value) for value in raw["training"]["coverage_targets"])
    results: dict[str, Any] = {}
    threshold_manifest: dict[str, Any] = {}
    slice_rows: list[dict[str, Any]] = []
    scored = test

    def evaluate_scores(name: str, cal_score: np.ndarray, test_score: np.ndarray) -> None:
        nonlocal scored
        scored = scored.with_columns(pl.Series(f"safety_score__{name}", test_score))
        results[name], threshold_manifest[name] = {}, {}
        for target in coverage_targets:
            thresholds = bucket_thresholds(calibration, cal_score, target)
            threshold_manifest[name][str(target)] = thresholds
            margin = safety_margin(test, test_score, thresholds)
            evidence, slices = _evaluate_action(test, margin, 0.0, name, target)
            evidence["strategy_portability"] = sum(
                row["net_risk_value"] > 0 for row in evidence["by_strategy"].values()
            )
            results[name][str(target)] = evidence
            slice_rows.extend(slices)

    for name, model in final_models.items():
        evaluate_scores(name, model.score(calibration), model.score(test))

    # Frozen prior policies and economic controls are comparisons only.
    if __name__ == "__main__":
        canonical = "btc_directional_model.bucket_harm_risk_tournament"
        sys.modules[canonical] = sys.modules[BucketHarmModel.__module__]
    prior_bucket = joblib.load(bucket_run / "tournament.joblib")
    component_cal = {
        name: model.score(calibration) for name, model in prior_bucket["models"].items()
    }
    component_test = {name: model.score(test) for name, model in prior_bucket["models"].items()}
    frozen_cal = np.asarray(
        [
            component_cal[prior_bucket["bucket_champions"][bucket]][index]
            for index, bucket in enumerate(calibration["time_bucket"].to_list())
        ]
    )
    frozen_test = np.asarray(
        [
            component_test[prior_bucket["bucket_champions"][bucket]][index]
            for index, bucket in enumerate(test["time_bucket"].to_list())
        ]
    )
    evaluate_scores("frozen_bucket_champion", frozen_cal, frozen_test)
    evaluate_scores(
        "frozen_winner_preserving_loss",
        prior_bucket["models"]["winner_preserving_loss"].score(calibration),
        prior_bucket["models"]["winner_preserving_loss"].score(test),
    )
    evaluate_scores(
        "matched_edge", calibration["expected_edge"].to_numpy(), test["expected_edge"].to_numpy()
    )

    __main__.EconomicHarmModel = EconomicHarmModel
    prior_loss = joblib.load(strategy_run / "tournament.joblib")
    construction = pl.read_parquet(candidate_run / "checkpoints/construction-panel.parquet")
    mean_loss = float(-construction.filter(pl.col("net_pnl") <= 0)["net_pnl"].mean())
    mean_win = float(construction.filter(pl.col("net_pnl") > 0)["net_pnl"].mean())
    for name, source in (
        ("frozen_logistic_loss_risk", "logistic_economics"),
        ("frozen_boosted_history_loss_risk", "boosted_recent_history"),
    ):
        model = prior_loss["models"][source]

        def score(frame: pl.DataFrame, estimator: Any = model) -> np.ndarray:
            probability = estimator.predict_proba(frame)
            return -(probability * mean_loss - (1.0 - probability) * mean_win - penalty)

        evaluate_scores(name, score(calibration), score(test))

    no_risk, no_risk_slices = _evaluate_action(
        test, np.ones(test.height), -float(np.finfo(np.float64).max), "no_risk", 1.0
    )
    no_risk["strategy_portability"] = 0
    results["no_risk"] = {"1.0": no_risk}
    threshold_manifest["no_risk"] = {"1.0": None}
    slice_rows.extend(no_risk_slices)

    ledgers = run_dir / "ledgers"
    ledgers.mkdir(exist_ok=True)
    scored.write_parquet(ledgers / "compatibility-candidates.parquet", compression="zstd")
    pl.DataFrame(slice_rows).write_parquet(ledgers / "slice-matrix.parquet", compression="zstd")
    producing_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=package_root, text=True
    ).strip()
    if __name__ == "__main__":
        canonical = "btc_directional_model.consensus_action_value_risk_tournament"
        sys.modules[canonical] = sys.modules[__name__]
        for cls in (ActionValueModel, PairwiseStoppingModel, ConformalActionValueModel):
            cls.__module__ = canonical
    artifact = run_dir / "tournament.joblib"
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "models": final_models,
            "thresholds": threshold_manifest,
            "bucket_champions": bucket_champions,
            "producing_commit": producing_commit,
            "deployment_status": "not_deployed",
        },
        artifact,
        compress=3,
    )
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
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "producing_commit": producing_commit,
        "artifact_sha256": artifact_sha,
        "qualification_status": "research_not_qualified_no_pristine_complete_six_strategy_cohort",
        "deployment_status": "not_deployed",
        "strategy_identities": STRATEGIES,
        "selection": selection,
        "walk_forward_results": walk_forward,
        "compatibility_results": results,
        "thresholds": threshold_manifest,
        "slice_matrix": slice_rows,
        "granular_champions": _best_cells(slice_rows),
        "feature_usage": {
            "eligible_causal_dimensions": len(feature_sets["all"]),
            "consensus_dimensions": list(CONSENSUS_FEATURES),
            "by_family": {name: len(features) for name, features in feature_sets.items()},
            "strategy_identity_predictive_feature": False,
        },
        "row_counts": {
            "training": training.height,
            "selection_fit": selection_fit.height,
            "compatibility_calibration": calibration.height,
            "compatibility_test": test.height,
        },
        "ranges": {
            "training": [str(training["window_start"].min()), str(training["window_start"].max())],
            "compatibility_calibration": [
                str(calibration["window_start"].min()),
                str(calibration["window_start"].max()),
            ],
            "compatibility_test": [
                str(test["window_start"].min()),
                str(test["window_start"].max()),
            ],
        },
        "forward_audit": forward,
        "source_identity": {
            "training_panel": {
                "path": str(entry_run / "checkpoints/training-panel.parquet"),
                "sha256": _sha256(entry_run / "checkpoints/training-panel.parquet"),
            },
            "six_strategy_panel": {
                "path": str(entry_run / "checkpoints/strategy-panel.parquet"),
                "sha256": _sha256(entry_run / "checkpoints/strategy-panel.parquet"),
            },
            "frozen_bucket_artifact": {
                "path": str(bucket_run / "tournament.joblib"),
                "sha256": _sha256(bucket_run / "tournament.joblib"),
            },
        },
        "integrity": {
            "chronological_selection": True,
            "compatibility_market_disjoint": True,
            "same_timestamp_consensus_only": True,
            "within_bucket_wait_labels": True,
            "strategy_identity_predictive_feature": False,
            "database_reads": False,
            "database_mutations": False,
            "new_tables": False,
            "new_schemas": False,
            "new_ingesters": False,
            "new_sources": False,
            "images_rebuilt": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "sklearn": sklearn.__version__,
        },
        "limitations": [
            "The August 21-25 exact six-strategy cohort was already viewed and remains compatibility evidence, not a sealed qualification cohort.",
            "Historical consensus features use the available frozen champion outputs at each timestamp; exact six-strategy identity is used only for compatibility evaluation and never as a predictive feature.",
            "September 7-9 remains audit-only because one process admitted zero candidates and the five newer processes admitted only 65 combined buys.",
            "Rejected live strategy decisions were not converted into synthetic admitted candidates.",
            "The full established April-August causal SSD-enriched panel was used; the common historical construction begins at second 60.",
            "This is research-only. No runtime, process, database, ingester, data source, schema, image or deployment was changed.",
        ],
    }
    _write_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(_report(metrics))
    _write_json(
        run_dir / "completion.json",
        {
            "completed": True,
            "artifact_sha256": artifact_sha,
            "metrics_sha256": _sha256(run_dir / "metrics.json"),
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-run")
    args = parser.parse_args()
    print(train_tournament(args.config, args.resume_run))


if __name__ == "__main__":
    main()
