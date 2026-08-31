"""Train and evaluate the five frozen middle-strategy challengers."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss

from .core_extract import file_sha256
from .middle_strategy_data import (
    bridge_cache,
    build_middle_panel,
    build_normalized_middle_panel,
    build_source_preserving_bridge_panel,
    extract_spot_l2,
    middle_cache,
    normalized_cache,
)
from .multivenue_early_entry_data import KEY_COLUMNS, load_data_config
from .multivenue_early_entry_tournament import (
    FORBIDDEN_INFERENCE_TOKENS,
    TreeModel,
    _fit_tree,
    _matrix,
    _neutralized_feature_indices,
    _predict_tree,
)

SCHEMA_VERSION = "btc-middle-strategy-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-middle-strategy-model-v1"
BASE_NAMES = (
    "middle_specialist_refit",
    "middle_q5_admission",
    "crossvenue_middle_specialist",
)
ALL_NAMES = (
    "middle_specialist_refit",
    "middle_q5_admission",
    "middle_agreement_ensemble",
    "crossvenue_middle_specialist",
    "price_time_calibrated_middle_ensemble",
)


@dataclass(frozen=True)
class BridgeTreeModel:
    features: tuple[str, ...]
    estimator: HistGradientBoostingRegressor
    calibrator: IsotonicRegression | None
    neutralized_columns: tuple[int, ...]


@dataclass(frozen=True)
class BridgeEnsembleCalibrator:
    estimator: Ridge

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        probability = np.clip(expit(self.estimator.predict(matrix)), 1e-6, 1 - 1e-6)
        return np.column_stack((1.0 - probability, probability))


def _uses_normalized_supervision(config: Any) -> bool:
    return "normalization" in config.raw


def _uses_bridge_supervision(config: Any) -> bool:
    return "settlement_bridge" in config.raw


def _tournament_cache(config: Any) -> Path:
    if _uses_bridge_supervision(config):
        return bridge_cache(config)
    return normalized_cache(config) if _uses_normalized_supervision(config) else middle_cache(config)


def _build_tournament_panel(config: Any, *, force: bool) -> tuple[pl.DataFrame, dict[str, Any]]:
    if _uses_bridge_supervision(config):
        return build_source_preserving_bridge_panel(config, force=force)
    if _uses_normalized_supervision(config):
        return build_normalized_middle_panel(config, force=force)
    return build_middle_panel(config, force=force)


def _fit_candidate_tree(
    frame: pl.DataFrame, features: tuple[str, ...], config: Any, seed: int
) -> TreeModel | BridgeTreeModel:
    """Fit the frozen tree family with the configured settlement supervision."""

    if "label_weight" not in frame.columns:
        return _fit_tree(frame, features, config, seed)
    markets = frame.select("market_id", "window_start").unique("market_id").sort("window_start")
    if markets.height < 250:
        raise RuntimeError(f"insufficient training markets: {markets.height}")
    fraction = float(config.raw["model"]["calibration_fraction"])
    boundary = markets["window_start"][max(1, int(markets.height * (1.0 - fraction)))]
    fit = frame.filter(pl.col("window_start") < boundary)
    calibration = frame.filter(pl.col("window_start") >= boundary)
    matrix = _matrix(fit, features)
    neutralized = _neutralized_feature_indices(matrix)
    if neutralized:
        matrix = np.delete(matrix, neutralized, axis=1)
    if not matrix.shape[1]:
        raise RuntimeError("all candidate features are constant or missing")
    spec = config.raw["model"]
    if "bridge_probability_target" in fit.columns:
        estimator = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(spec["learning_rate"]),
            max_iter=int(spec["max_iter"]),
            max_leaf_nodes=int(spec["max_leaf_nodes"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            l2_regularization=float(spec["l2_regularization"]),
            max_bins=int(spec["max_bins"]),
            early_stopping=False,
            random_state=seed,
        ).fit(
            matrix,
            fit["bridge_probability_target"].to_numpy(),
            sample_weight=fit["label_weight"].to_numpy(),
        )
        raw = np.clip(
            estimator.predict(_matrix(calibration, features, neutralized)),
            1e-6,
            1 - 1e-6,
        )
        calibrator: IsotonicRegression | None = None
        target = calibration["bridge_probability_target"].to_numpy()
        if np.ptp(raw) > 1e-9 and np.ptp(target) > 1e-9:
            calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            ).fit(raw, target, sample_weight=calibration["label_weight"].to_numpy())
        return BridgeTreeModel(features, estimator, calibrator, neutralized)
    estimator = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(spec["learning_rate"]),
        max_iter=int(spec["max_iter"]),
        max_leaf_nodes=int(spec["max_leaf_nodes"]),
        min_samples_leaf=int(spec["min_samples_leaf"]),
        l2_regularization=float(spec["l2_regularization"]),
        max_bins=int(spec["max_bins"]),
        early_stopping=False,
        random_state=seed,
    ).fit(matrix, fit["label_up"].to_numpy(), sample_weight=fit["label_weight"].to_numpy())
    raw = np.clip(
        estimator.predict_proba(_matrix(calibration, features, neutralized))[:, 1],
        1e-6,
        1 - 1e-6,
    )
    calibrator: LogisticRegression | None = None
    if calibration["label_up"].n_unique() == 2:
        calibrator = LogisticRegression(
            C=1.0, solver="lbfgs", random_state=seed, max_iter=500
        ).fit(
            logit(raw).reshape(-1, 1),
            calibration["label_up"].to_numpy(),
            sample_weight=calibration["label_weight"].to_numpy(),
        )
    return TreeModel(features, estimator, calibrator, neutralized)


def _predict_candidate_tree(
    model: TreeModel | BridgeTreeModel, frame: pl.DataFrame
) -> np.ndarray:
    if isinstance(model, BridgeTreeModel):
        raw = np.clip(
            model.estimator.predict(
                _matrix(frame, model.features, model.neutralized_columns)
            ),
            1e-6,
            1 - 1e-6,
        )
        if model.calibrator is None:
            return raw
        return np.clip(model.calibrator.predict(raw), 1e-6, 1 - 1e-6)
    return _predict_tree(model, frame)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _git_revision(package_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=package_root, text=True
    ).strip()


def _candidate_contract(config: Any, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups = {name: tuple(values) for name, values in manifest["feature_groups"].items()}
    output: dict[str, dict[str, Any]] = {}
    for row in config.raw["candidates"]:
        contract = dict(row)
        if row["kind"] == "base":
            features = tuple(
                dict.fromkeys(
                    feature
                    for group in row["feature_groups"]
                    for feature in groups[group]
                )
            )
            forbidden = [
                name
                for name in features
                if any(token in name.lower() for token in FORBIDDEN_INFERENCE_TOKENS)
            ]
            if forbidden:
                raise RuntimeError(f"forbidden inference features for {row['name']}: {forbidden}")
            contract["features"] = features
        output[row["name"]] = contract
    if tuple(output) != ALL_NAMES:
        raise RuntimeError("candidate roster or order differs from the frozen contract")
    return output


def _prediction_frame(
    frame: pl.DataFrame, probability: np.ndarray, fold: str, candidate: str
) -> pl.DataFrame:
    if len(probability) != frame.height or not np.isfinite(probability).all():
        raise RuntimeError(f"{candidate} produced invalid probabilities")
    columns = [*KEY_COLUMNS, "label_up"]
    for optional in ("label_weight", "bridge_probability_target"):
        if optional in frame.columns:
            columns.append(optional)
    return frame.select(*columns).with_columns(
        pl.lit(fold).alias("fold"),
        pl.lit(candidate).alias("candidate"),
        pl.Series("probability", probability),
        pl.lit(True).alias("eligible_signal"),
    )


def _base_oof(
    panel: pl.DataFrame,
    contracts: dict[str, dict[str, Any]],
    config: Any,
) -> pl.DataFrame:
    checkpoint = _tournament_cache(config) / "base-oof-predictions.parquet"
    pieces: list[pl.DataFrame] = []
    existing: set[tuple[str, str]] = set()
    if checkpoint.is_file():
        cached = pl.read_parquet(checkpoint)
        pieces.append(cached)
        existing = set(cached.select("candidate", "fold").unique().iter_rows())
    seed = int(config.raw["training"]["random_seed"])
    for fold_index, fold in enumerate(config.raw["folds"]):
        start = datetime.fromisoformat(fold["test_start"])
        end = datetime.fromisoformat(fold["test_end"])
        train = panel.filter(pl.col("window_start") < start)
        test = panel.filter(pl.col("window_start").is_between(start, end, closed="left"))
        if set(train["market_id"].unique()) & set(test["market_id"].unique()):
            raise RuntimeError(f"market contamination in {fold['name']}")
        for base_index, name in enumerate(BASE_NAMES):
            if (name, fold["name"]) in existing:
                continue
            model = _fit_candidate_tree(
                train,
                tuple(contracts[name]["features"]),
                config,
                seed + 100 * fold_index + base_index,
            )
            piece = _prediction_frame(
                test, _predict_candidate_tree(model, test), fold["name"], name
            )
            pieces.append(piece)
            pl.concat(pieces, how="vertical_relaxed").write_parquet(
                checkpoint, compression="zstd", statistics=True
            )
            print(
                f"middle tournament OOF: {fold['name']} {name} "
                f"{test['market_id'].n_unique():,} markets",
                flush=True,
            )
    return pl.concat(pieces, how="vertical_relaxed", rechunk=True).sort(
        ["fold", "candidate", "window_start", "market_id", "seconds_elapsed"]
    )


def _wide_base_predictions(base: pl.DataFrame) -> pl.DataFrame:
    optional = tuple(
        name
        for name in ("label_weight", "bridge_probability_target")
        if name in base.columns
    )
    keys = (*KEY_COLUMNS, "label_up", *optional, "fold")
    parts = []
    for name in BASE_NAMES:
        parts.append(
            base.filter(pl.col("candidate") == name)
            .select(*keys, pl.col("probability").alias(name))
        )
    output = parts[0]
    for part in parts[1:]:
        output = output.join(part, on=list(keys), how="inner", validate="1:1")
    return output


def _agreement_predictions(wide: pl.DataFrame) -> pl.DataFrame:
    first, second = BASE_NAMES[:2]
    agrees = (pl.col(first) >= 0.5) == (pl.col(second) >= 0.5)
    probability = ((pl.col(first) + pl.col(second)) / 2.0).alias("probability")
    optional = tuple(
        name
        for name in ("label_weight", "bridge_probability_target")
        if name in wide.columns
    )
    return wide.select(*KEY_COLUMNS, "label_up", *optional, "fold", probability, agrees.alias("eligible_signal")).with_columns(
        pl.lit("middle_agreement_ensemble").alias("candidate")
    ).select(
        *KEY_COLUMNS,
        "label_up",
        *optional,
        "fold",
        "candidate",
        "probability",
        "eligible_signal",
    )


def _ensemble_matrix(frame: pl.DataFrame) -> np.ndarray:
    probabilities = [
        np.clip(frame[name].to_numpy().astype(float), 1e-6, 1 - 1e-6)
        for name in BASE_NAMES
    ]
    seconds = frame["seconds_elapsed"].to_numpy().astype(float) / 300.0
    return np.column_stack((*[logit(values) for values in probabilities], seconds, seconds**2))


def _fit_ensemble_calibrator(
    frame: pl.DataFrame, seed: int
) -> LogisticRegression | BridgeEnsembleCalibrator:
    kwargs = {}
    if "label_weight" in frame.columns:
        kwargs["sample_weight"] = frame["label_weight"].to_numpy()
    if "bridge_probability_target" in frame.columns:
        target = np.clip(
            frame["bridge_probability_target"].to_numpy().astype(float),
            0.001,
            0.999,
        )
        estimator = Ridge(alpha=0.5, random_state=seed).fit(
            _ensemble_matrix(frame), logit(target), **kwargs
        )
        return BridgeEnsembleCalibrator(estimator)
    return LogisticRegression(C=0.5, solver="lbfgs", random_state=seed, max_iter=500).fit(
        _ensemble_matrix(frame), frame["label_up"].to_numpy(), **kwargs
    )


def _causal_calibrated_oof(wide: pl.DataFrame, config: Any) -> pl.DataFrame:
    pieces = []
    prior = []
    seed = int(config.raw["training"]["random_seed"])
    for fold_index, fold in enumerate(config.raw["folds"]):
        current = wide.filter(pl.col("fold") == fold["name"])
        if prior:
            fitting = pl.concat(prior, how="vertical_relaxed")
            calibrator = _fit_ensemble_calibrator(fitting, seed + fold_index)
            probability = calibrator.predict_proba(_ensemble_matrix(current))[:, 1]
        else:
            probability = current.select(pl.mean_horizontal(*BASE_NAMES)).to_series().to_numpy()
        pieces.append(
            _prediction_frame(
                current,
                probability,
                fold["name"],
                "price_time_calibrated_middle_ensemble",
            )
        )
        prior.append(current)
    return pl.concat(pieces, how="vertical_relaxed")


def _all_oof(base: pl.DataFrame, config: Any) -> tuple[pl.DataFrame, pl.DataFrame]:
    wide = _wide_base_predictions(base)
    agreement = _agreement_predictions(wide)
    calibrated = _causal_calibrated_oof(wide, config)
    return pl.concat((base, agreement, calibrated), how="vertical_relaxed"), wide


def _ece(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1]
            if index == bins - 1
            else probability < edges[index + 1]
        )
        if selected.any():
            score += selected.mean() * abs(probability[selected].mean() - labels[selected].mean())
    return float(score)


def predictive_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    labels = frame["label_up"].to_numpy().astype(float)
    probability = frame["probability"].to_numpy().astype(float)

    def summary(part: pl.DataFrame) -> dict[str, Any]:
        if part.is_empty():
            return {
                "rows": 0, "markets": 0, "brier_score": None, "log_loss": None,
                "accuracy": None, "ece_15": None,
            }
        y = part["label_up"].to_numpy().astype(float)
        p = part["probability"].to_numpy().astype(float)
        result = {
            "rows": part.height,
            "markets": part["market_id"].n_unique(),
            "brier_score": float(np.mean((p - y) ** 2)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "accuracy": float(np.mean((p >= 0.5) == y)),
            "ece_15": _ece(y, p),
        }
        if "bridge_probability_target" in part.columns:
            target = part["bridge_probability_target"].to_numpy().astype(float)
            result["bridge_target_brier_score"] = float(np.mean((p - target) ** 2))
            result["bridge_target_cross_entropy"] = float(
                np.mean(-target * np.log(p) - (1.0 - target) * np.log(1.0 - p))
            )
        return result

    bands = {}
    for name, start, end in (
        ("60_89", 60, 89),
        ("90_119", 90, 119),
        ("120_149", 120, 149),
        ("150_180", 150, 180),
    ):
        bands[name] = summary(
            frame.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        )
    result = summary(frame)
    result["mean_probability"] = float(probability.mean())
    result["positive_rate"] = float(labels.mean())
    result["by_timing_band"] = bands
    return result


def _opportunities(predictions: pl.DataFrame, panel: pl.DataFrame, config: Any) -> pl.DataFrame:
    quantity = int(config.raw["execution"]["quantity"])
    up, down = f"up_ask_vwap_{quantity}", f"down_ask_vwap_{quantity}"
    context = [up, down, "fee_rate", "pm_up_book_age_seconds", "pm_down_book_age_seconds"]
    joined = predictions.join(
        panel.select(*KEY_COLUMNS, *context), on=list(KEY_COLUMNS), how="left", validate="m:1"
    )
    reserve = float(config.raw["execution"]["execution_reserve_per_share"])
    return (
        joined.filter(
            pl.col("seconds_elapsed").is_between(
                int(config.raw["entry"]["economic_start_second"]),
                int(config.raw["entry"]["economic_end_second_inclusive"]),
                closed="both",
            )
        )
        .with_columns(
            pl.when(pl.col("probability") >= 0.5).then(pl.lit("up")).otherwise(pl.lit("down")).alias("side"),
            pl.max_horizontal("probability", 1.0 - pl.col("probability")).alias("selected_probability"),
            pl.when(pl.col("probability") >= 0.5).then(pl.col(up)).otherwise(pl.col(down)).alias("share_cost"),
        )
        .with_columns(
            (pl.col("fee_rate").fill_null(0.0) * pl.col("share_cost") * (1.0 - pl.col("share_cost"))).alias("fee_per_share")
        )
        .with_columns(
            (pl.col("selected_probability") - pl.col("share_cost") - pl.col("fee_per_share") - reserve).alias("expected_edge")
        )
    )


def _select_trades(opportunities: pl.DataFrame, policy: dict[str, float], config: Any) -> pl.DataFrame:
    quantity = int(config.raw["execution"]["quantity"])
    reserve = float(config.raw["execution"]["execution_reserve_per_share"])
    stress = float(config.raw["execution"]["stress_slippage_per_share"])
    eligible = (
        opportunities.filter(
            pl.col("eligible_signal")
            & pl.col("share_cost").is_not_null()
            & pl.col("share_cost").is_finite()
            & (pl.col("share_cost") > 0)
            & (pl.col("share_cost") <= policy["maximum_share_cost"])
            & (pl.col("selected_probability") >= policy["minimum_confidence"])
            & (pl.col("expected_edge") >= policy["minimum_edge"])
            & (pl.col("pm_up_book_age_seconds") > 0)
            & (pl.col("pm_up_book_age_seconds") <= 2)
            & (pl.col("pm_down_book_age_seconds") > 0)
            & (pl.col("pm_down_book_age_seconds") <= 2)
        )
        .sort(["market_id", "seconds_elapsed"])
        .group_by("market_id", maintain_order=True)
        .first()
    )
    won = ((pl.col("side") == "up") & (pl.col("label_up") == 1)) | (
        (pl.col("side") == "down") & (pl.col("label_up") == 0)
    )
    return eligible.with_columns(won.alias("won")).with_columns(
        pl.when(pl.col("won")).then(1.0 - pl.col("share_cost")).otherwise(-pl.col("share_cost")).sub(pl.col("fee_per_share") + reserve).mul(quantity).alias("net_pnl"),
        pl.when(pl.col("won")).then(1.0 - pl.col("share_cost") - stress).otherwise(-(pl.col("share_cost") + stress)).sub(pl.col("fee_per_share") + reserve).mul(quantity).alias("stress_net_pnl"),
    )


def economic_metrics(trades: pl.DataFrame, total_markets: int) -> dict[str, Any]:
    if trades.is_empty():
        return {
            "trades": 0, "winning_trades": 0, "losing_trades": 0, "win_loss_ratio": None,
            "win_rate": None, "net_pnl": 0.0, "expectancy_per_trade": None,
            "stress_net_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            "profit_factor": None, "loss_recovery_wins": None, "market_coverage": 0.0,
            "average_share_cost": None, "average_entry_second": None, "maximum_drawdown": 0.0,
            "active_days": 0, "profitable_day_ratio": None, "maximum_daily_pnl_concentration": None,
            "by_entry_cell": {}, "by_price_bucket": {},
        }
    pnl = trades["net_pnl"].to_numpy().astype(float)
    wins, losses = pnl[pnl > 0], -pnl[pnl < 0]
    cumulative = np.cumsum(pnl)
    drawdown = np.maximum.accumulate(np.r_[0.0, cumulative])[1:] - cumulative
    winning, losing = len(wins), len(losses)
    daily = trades.with_columns(pl.col("window_start").dt.date().alias("day")).group_by("day").agg(pl.col("net_pnl").sum()).sort("day")
    positive_daily = daily.filter(pl.col("net_pnl") > 0)["net_pnl"].sum()
    maximum_concentration = None
    if positive_daily and positive_daily > 0:
        maximum_concentration = float(daily["net_pnl"].max() / positive_daily)

    def grouped_metrics(column: str, breaks: tuple[tuple[str, float, float], ...]) -> dict[str, Any]:
        output = {}
        for name, start, end in breaks:
            part = trades.filter(pl.col(column).is_between(start, end, closed="both"))
            values = part["net_pnl"].to_numpy().astype(float)
            output[name] = {
                "trades": part.height,
                "wins": int((values > 0).sum()),
                "losses": int((values < 0).sum()),
                "net_pnl": float(values.sum()),
            }
        return output

    return {
        "trades": trades.height,
        "winning_trades": winning,
        "losing_trades": losing,
        "win_loss_ratio": winning / losing if losing else None,
        "win_rate": winning / trades.height,
        "net_pnl": float(pnl.sum()),
        "expectancy_per_trade": float(pnl.mean()),
        "stress_net_pnl": float(trades["stress_net_pnl"].sum()),
        "gross_profit": float(wins.sum()),
        "gross_loss": float(losses.sum()),
        "profit_factor": float(wins.sum() / losses.sum()) if losses.sum() else None,
        "loss_recovery_wins": float(losses.mean() / wins.mean()) if winning and losing else None,
        "market_coverage": trades["market_id"].n_unique() / max(total_markets, 1),
        "average_share_cost": float(trades["share_cost"].mean()),
        "average_entry_second": float(trades["seconds_elapsed"].mean()),
        "maximum_drawdown": float(drawdown.max(initial=0.0)),
        "active_days": daily.height,
        "profitable_day_ratio": float((daily["net_pnl"] > 0).mean()),
        "maximum_daily_pnl_concentration": maximum_concentration,
        "by_entry_cell": grouped_metrics(
            "seconds_elapsed",
            (("60_89", 60, 89), ("90_119", 90, 119),
             ("120_149", 120, 149), ("150_180", 150, 180)),
        ),
        "by_price_bucket": grouped_metrics("share_cost", (("0_0.65", 0.0, 0.65), ("0.65_0.80", 0.6500001, 0.80), ("0.80_0.95", 0.8000001, 0.95), ("0.95_1.0", 0.9500001, 1.0))),
    }


def _policy_score(trades: pl.DataFrame) -> float:
    if trades.is_empty():
        return -math.inf
    pnl = trades["net_pnl"].to_numpy().astype(float)
    if len(pnl) < 2:
        return -math.inf
    lower_expectancy = float(pnl.mean() - pnl.std(ddof=1) / math.sqrt(len(pnl)))
    folds = trades.group_by("fold").agg(pl.col("net_pnl").sum())["net_pnl"].to_numpy()
    return lower_expectancy + 0.01 * float(np.median(folds)) + 0.0005 * math.sqrt(len(pnl))


def _select_policy(opportunities: pl.DataFrame, config: Any) -> tuple[dict[str, float], dict[str, Any]]:
    execution = config.raw["execution"]
    best: tuple[float, dict[str, float], pl.DataFrame] | None = None
    for edge in execution["minimum_edges"]:
        for confidence in execution["minimum_confidences"]:
            for cost in execution["maximum_share_costs"]:
                policy = {"minimum_edge": float(edge), "minimum_confidence": float(confidence), "maximum_share_cost": float(cost)}
                trades = _select_trades(opportunities, policy, config)
                if trades.height < int(execution["minimum_policy_trades"]):
                    continue
                score = _policy_score(trades)
                if best is None or score > best[0]:
                    best = (score, policy, trades)
    if best is None:
        policy = {
            "minimum_edge": min(execution["minimum_edges"]),
            "minimum_confidence": min(execution["minimum_confidences"]),
            "maximum_share_cost": max(execution["maximum_share_costs"]),
        }
        return policy, {"selection_score": None, "development": economic_metrics(pl.DataFrame(), opportunities["market_id"].n_unique())}
    score, policy, trades = best
    return policy, {"selection_score": score, "development": economic_metrics(trades, opportunities["market_id"].n_unique())}


def _calibration_cells(config: Any) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (f"{int(start)}_{int(end)}", int(start), int(end))
        for start, end in config.raw["entry"]["calibration_cells"]
    )


def _select_cell_policies(opportunities: pl.DataFrame, config: Any) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    policies, evidence = {}, {}
    for name, start, end in _calibration_cells(config):
        policies[name], evidence[name] = _select_policy(
            opportunities.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both")), config
        )
    return policies, evidence


def _apply_cell_policies(opportunities: pl.DataFrame, policies: dict[str, dict[str, float]], config: Any) -> pl.DataFrame:
    pieces = []
    for name, start, end in _calibration_cells(config):
        pieces.append(_select_trades(opportunities.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both")), policies[name], config))
    trades = pl.concat(pieces, how="diagonal_relaxed")
    if trades.is_empty():
        return trades
    return trades.sort(["market_id", "seconds_elapsed"]).group_by("market_id", maintain_order=True).first()


def _apply_candidate_policy(
    opportunities: pl.DataFrame,
    policy: dict[str, Any],
    candidate: str,
    config: Any,
) -> pl.DataFrame:
    if candidate == "price_time_calibrated_middle_ensemble":
        return _apply_cell_policies(opportunities, policy, config)
    return _select_trades(opportunities, policy, config)


def _counterfactual_economics(
    opportunities: pl.DataFrame,
    policy: dict[str, Any],
    candidate: str,
    config: Any,
    total_markets: int,
) -> dict[str, Any]:
    late = _apply_candidate_policy(
        opportunities.filter(
            pl.col("seconds_elapsed").is_between(150, 180, closed="both")
        ),
        policy,
        candidate,
        config,
    )
    early_high_confidence = _apply_candidate_policy(
        opportunities.filter(
            ~pl.col("seconds_elapsed").is_between(60, 89, closed="both")
            | (pl.col("selected_probability") >= 0.80)
        ),
        policy,
        candidate,
        config,
    )
    return {
        "entries_150_180_only": economic_metrics(late, total_markets),
        "p80_required_at_60_89": economic_metrics(
            early_high_confidence, total_markets
        ),
    }


def _prior_run_comparison(config: Any, metrics: dict[str, Any]) -> dict[str, Any] | None:
    path_value = config.raw.get("comparison", {}).get("prior_metrics")
    if not path_value:
        return None
    path = config.package_root / path_value
    prior = json.loads(path.read_text())
    comparison: dict[str, Any] = {
        "prior_metrics_path": str(path.relative_to(config.package_root)),
        "prior_metrics_sha256": file_sha256(path),
        "prior_run_id": prior["run_id"],
        "same_sealed_market_contract": (
            prior["split_manifest"]["sealed_start_inclusive"]
            == metrics["split_manifest"]["sealed_start_inclusive"]
            and prior["split_manifest"]["sealed_end_exclusive"]
            == metrics["split_manifest"]["sealed_end_exclusive"]
        ),
        "candidates": {},
    }
    for name in ALL_NAMES:
        old_p = prior["sealed_predictive"][name]
        new_p = metrics["sealed_predictive"][name]
        old_e = prior["sealed_economic"][name]
        new_e = metrics["sealed_economic"][name]
        comparison["candidates"][name] = {
            "brier_score_delta": new_p["brier_score"] - old_p["brier_score"],
            "net_pnl_delta": new_e["net_pnl"] - old_e["net_pnl"],
            "stress_net_pnl_delta": (
                new_e["stress_net_pnl"] - old_e["stress_net_pnl"]
            ),
            "profit_factor_delta": (
                None
                if new_e["profit_factor"] is None or old_e["profit_factor"] is None
                else new_e["profit_factor"] - old_e["profit_factor"]
            ),
            "coverage_delta": new_e["market_coverage"] - old_e["market_coverage"],
            "trade_count_delta": new_e["trades"] - old_e["trades"],
        }
    return comparison


def _reference_manifests(config: Any) -> dict[str, Any]:
    output = {}
    for name, key in (("frozen_q5", "q5_manifest"), ("frozen_middle_specialist", "specialist_manifest")):
        path = config.package_root / config.raw["paths"][key]
        payload = json.loads(path.read_text())
        output[name] = {
            "manifest_path": str(path.relative_to(config.package_root)),
            "manifest_sha256": file_sha256(path),
            "model_artifact_sha256": payload.get("model_artifact", {}).get("sha256")
            or payload.get("artifact_sha256")
            or payload.get("model_sha256"),
            "scored_without_alteration": False,
            "reason": "Immutable runtime comparator retained for provenance; its input, target, and entry-policy contract is not identical to this tournament panel.",
        }
    return output


def _observed_ranges(config: Any) -> dict[str, Any]:
    panel = pl.scan_parquet(_tournament_cache(config) / "middle-panel.parquet")
    observed = panel.select(
        pl.col("window_start").min().alias("first_market"),
        pl.col("window_start").max().alias("last_market"),
        pl.col("market_id").n_unique().alias("markets"),
        pl.len().alias("rows"),
    ).collect().row(0, named=True)
    sealed = panel.filter(
        pl.col("window_start").is_between(
            config.sealed_start, config.sealed_end, closed="left"
        )
    ).select(
        pl.col("window_start").min().alias("first_market"),
        pl.col("window_start").max().alias("last_market"),
        pl.col("market_id").n_unique().alias("markets"),
    ).collect().row(0, named=True)
    return {
        "configured_start": config.source_start.isoformat(),
        "configured_end_exclusive": config.sealed_end.isoformat(),
        "first_observed_market": observed["first_market"].isoformat(),
        "last_observed_market": observed["last_market"].isoformat(),
        "rows": observed["rows"],
        "markets": observed["markets"],
        "sealed_first_observed_market": sealed["first_market"].isoformat(),
        "sealed_last_observed_market": sealed["last_market"].isoformat(),
        "sealed_markets": sealed["markets"],
    }


def _qualification_status(economic: dict[str, dict[str, Any]]) -> str:
    qualified = any(
        row["net_pnl"] > 0
        and row["stress_net_pnl"] > 0
        and row["profit_factor"] is not None
        and row["profit_factor"] > 1
        for row in economic.values()
    )
    return (
        "trained_evaluated_not_deployed"
        if qualified
        else "trained_evaluated_not_promoted_negative_expectancy"
    )


def _report(metrics: dict[str, Any]) -> str:
    def fmt(value: Any, digits: int = 3) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Middle-Strategy Tournament", "", f"Run: `{metrics['run_id']}`",
        f"Qualification: **{metrics['qualification_status']}**", "", "## Sealed high-level results", "",
        "| Candidate | PnL | Stress PnL | Coverage | Wins | Losses | W/L | Recovery wins/loss | Brier | PF | Avg cost | Avg entry |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ALL_NAMES:
        p, e = metrics["sealed_predictive"][name], metrics["sealed_economic"][name]
        lines.append(
            f"| {name} | {e['net_pnl']:.2f} | {e['stress_net_pnl']:.2f} | {e['market_coverage']:.2%} | {e['winning_trades']} | {e['losing_trades']} | {fmt(e['win_loss_ratio'])} | {fmt(e['loss_recovery_wins'])} | {p['brier_score']:.4f} | {fmt(e['profit_factor'])} | {fmt(e['average_share_cost'])} | {fmt(e['average_entry_second'], 1)} |"
        )
    if metrics.get("prior_run_comparison"):
        lines.extend((
            "", "## Paired change from the prior normalized tournament", "",
            "| Candidate | PnL delta | Stress PnL delta | Brier delta | PF delta | Coverage delta |",
            "|---|---:|---:|---:|---:|---:|",
        ))
        for name in ALL_NAMES:
            row = metrics["prior_run_comparison"]["candidates"][name]
            lines.append(
                f"| {name} | {row['net_pnl_delta']:.2f} | {row['stress_net_pnl_delta']:.2f} | {row['brier_score_delta']:.5f} | {fmt(row['profit_factor_delta'])} | {row['coverage_delta']:.2%} |"
            )
    if metrics.get("sealed_counterfactual_economic"):
        lines.extend((
            "", "## Prescribed sealed counterfactuals", "",
            "| Candidate | 150-180 PnL / PF / trades | p80 early PnL / PF / trades |",
            "|---|---:|---:|",
        ))
        for name in ALL_NAMES:
            row = metrics["sealed_counterfactual_economic"][name]
            late = row["entries_150_180_only"]
            p80 = row["p80_required_at_60_89"]
            lines.append(
                f"| {name} | {late['net_pnl']:.2f} / {fmt(late['profit_factor'])} / {late['trades']} | {p80['net_pnl']:.2f} / {fmt(p80['profit_factor'])} / {p80['trades']} |"
            )
    bridge = metrics["source_panel"].get("bridge_calibration")
    if bridge:
        ref = bridge["refprice_to_exact"]
        binance = bridge["binance_to_exact"]
        lines.extend((
            "", "## Settlement bridge", "",
            f"- RefPrice residual: {ref['paired_markets']} paired markets, location {ref['location_bps']:.4f} bps, scale {ref['scale_bps']:.4f} bps, MAE {ref['mae_bps']:.4f} bps, p99 {ref['p99_absolute_bps']:.4f} bps.",
            f"- Binance residual: {binance['paired_markets']} paired markets, location {binance['location_bps']:.4f} bps, scale {binance['scale_bps']:.4f} bps, MAE {binance['mae_bps']:.4f} bps, p99 {binance['p99_absolute_bps']:.4f} bps.",
            "- Bridge parameters were fitted before policy development and sealed evaluation; raw official, RefPrice, exact TWAP and Binance values remain lineage-distinct.",
        ))
    lines.extend((
        "", "## Integrity", "",
        f"- Configured source interval: {metrics['data_observed']['configured_start']} through {metrics['data_observed']['configured_end_exclusive']} exclusive.",
        f"- Actual retained markets: {metrics['data_observed']['first_observed_market']} through {metrics['data_observed']['last_observed_market']}; optional-source gaps removed no core markets.",
        "- Predictive training, policy development, and sealed testing are chronological and market-disjoint.",
        "- The replay dates were observed in earlier research, so they are computationally sealed here but are not claimed as epistemically untouched.",
        "- Settlement supervision is absent from inference; no row-selection lock was applied to the retained full-history panel.",
        "- Economic results require both recorded books to be no more than two seconds old.",
        "- No database writes, tables, ingesters, sources, runtime exports, deployments, or trading-process changes were made.",
    ))
    return "\n".join(lines) + "\n"


def _validate_artifact(config: Any, artifact: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(config.package_root / "src")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import joblib,sys; x=joblib.load(sys.argv[1]); assert len(x['base_models'])==3",
            str(artifact),
        ],
        check=True,
        cwd=config.package_root,
        env=environment,
    )
    return joblib.load(artifact)


def _predict_frozen_candidates(
    frame: pl.DataFrame,
    models: dict[str, TreeModel | BridgeTreeModel],
    calibrator: LogisticRegression | BridgeEnsembleCalibrator,
    fold: str,
) -> pl.DataFrame:
    base = pl.concat(
        [
            _prediction_frame(
                frame, _predict_candidate_tree(models[name], frame), fold, name
            )
            for name in BASE_NAMES
        ],
        how="vertical_relaxed",
    )
    wide = _wide_base_predictions(base)
    agreement = _agreement_predictions(wide)
    calibrated = _prediction_frame(
        wide,
        calibrator.predict_proba(_ensemble_matrix(wide))[:, 1],
        fold,
        "price_time_calibrated_middle_ensemble",
    )
    return pl.concat((base, agreement, calibrated), how="vertical_relaxed")


def finalize_run(config: Any, run_dir: Path) -> Path:
    """Finalize an already-trained checkpoint without fitting any model again."""

    artifact = run_dir / "tournament.joblib"
    payload = _validate_artifact(config, artifact)
    if payload["run_id"] != run_dir.name:
        raise RuntimeError("artifact run identity does not match the checkpoint directory")
    oof = pl.read_parquet(run_dir / "ledgers" / "candidate-oof-predictions.parquet")
    sealed = pl.read_parquet(run_dir / "ledgers" / "sealed-predictions.parquet")
    trades = pl.read_parquet(run_dir / "ledgers" / "sealed-trades.parquet")
    selection = json.loads((run_dir / "selection-freeze.json").read_text())
    split = json.loads((run_dir / "split-manifest.json").read_text())
    panel_manifest = json.loads((_tournament_cache(config) / "panel-manifest.json").read_text())
    panel = pl.read_parquet(_tournament_cache(config) / "middle-panel.parquet")
    sealed_panel = panel.filter(
        pl.col("window_start").is_between(
            config.sealed_start, config.sealed_end, closed="left"
        )
    )
    policies = selection["policies"]
    oof_predictive, sealed_predictive, sealed_economic = {}, {}, {}
    counterfactual_economic = {}
    policy_evidence = {}
    total_sealed_markets = sealed["market_id"].n_unique()
    for name in ALL_NAMES:
        candidate_oof = oof.filter(pl.col("candidate") == name)
        candidate_sealed = sealed.filter(pl.col("candidate") == name)
        candidate_trades = trades.filter(pl.col("candidate") == name)
        oof_predictive[name] = predictive_metrics(candidate_oof)
        sealed_predictive[name] = predictive_metrics(candidate_sealed)
        sealed_economic[name] = economic_metrics(candidate_trades, total_sealed_markets)
        counterfactual_economic[name] = _counterfactual_economics(
            _opportunities(candidate_sealed, sealed_panel, config),
            policies[name],
            name,
            config,
            total_sealed_markets,
        )
        policy_evidence[name] = {
            "selection_score": "recorded_before_seal",
            "policy": policies[name],
        }
    economic_ranking = sorted(
        ALL_NAMES,
        key=lambda name: (
            sealed_economic[name]["stress_net_pnl"],
            sealed_economic[name]["net_pnl"],
        ),
        reverse=True,
    )
    artifact_sha = file_sha256(artifact)
    (run_dir / "tournament.sha256").write_text(f"{artifact_sha}  tournament.joblib\n")
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": payload["run_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "producing_commit": payload["producing_commit"],
        "candidate_names": list(ALL_NAMES),
        "candidate_count_trained": 5,
        "learned_base_model_count": 3,
        "derived_ensemble_count": 2,
        "predictive_ranking_preseal": selection["predictive_ranking"],
        "economic_ranking_sealed": economic_ranking,
        "selection_frozen_at": selection["frozen_at"],
        "oof_predictive": oof_predictive,
        "selected_policies": policies,
        "policy_selection_evidence": policy_evidence,
        "sealed_predictive": sealed_predictive,
        "sealed_economic": sealed_economic,
        "sealed_counterfactual_economic": counterfactual_economic,
        "frozen_comparator_references": _reference_manifests(config),
        "source_panel": panel_manifest,
        "split_manifest": split,
        "artifact_sha256": artifact_sha,
        "qualification_status": _qualification_status(sealed_economic),
        "data_observed": _observed_ranges(config),
        "integrity": {
            "passed": True,
            "market_disjoint": True,
            "sealed_opened_after_selection": True,
            "artifact_round_trip_load": True,
            "finalized_from_existing_checkpoints_without_retraining": True,
            "full_history_retained": True,
            "optional_missingness_preserves_rows": True,
            "twap_inference_feature": False,
            "source_values_remain_distinct": _uses_bridge_supervision(config),
            "historical_official_label_preserved": _uses_bridge_supervision(config),
            "kraken_l2_included": False,
            "database_mutations": False,
            "new_tables": False,
            "new_ingesters": False,
            "new_sources": False,
            "runtime_exported": False,
            "deployed": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "sklearn": sklearn.__version__,
        },
        "limitations": [
            "The August 14-28 development/test period is row-disjoint but was observed during previous research and is not epistemically fresh.",
            "Kraken L2 is excluded because its historical backfill is incomplete.",
            "Qualified Binance spot L2 ends August 1; later rows remain usable with L2 missing.",
            "Projected PnL assumes recorded ask VWAP was fillable and does not model queue position.",
        ],
    }
    metrics["prior_run_comparison"] = _prior_run_comparison(config, metrics)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "candidate-contract.json", payload["candidate_contract"])
    _write_json(run_dir / "source-manifest.json", panel_manifest)
    (run_dir / "report.md").write_text(_report(metrics))
    _write_json(
        run_dir / "completion.json",
        {
            "run_id": payload["run_id"],
            "artifact_sha256": artifact_sha,
            "metrics_sha256": file_sha256(run_dir / "metrics.json"),
            "completed": True,
            "retrained_during_finalization": False,
        },
    )
    return run_dir


def train_tournament(config: Any, *, force: bool = False) -> Path:
    panel, panel_manifest = _build_tournament_panel(config, force=force)
    contracts = _candidate_contract(config, panel_manifest)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.results / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledgers = run_dir / "ledgers"
    ledgers.mkdir()
    entry = config.raw["entry"]
    entry_seconds = list(
        range(
            int(entry["start_second"]),
            int(entry["end_second_inclusive"]) + 1,
            int(entry["cadence_seconds"]),
        )
    )
    split = {
        "source_start": config.source_start.isoformat(),
        "fit_end_exclusive": config.fit_end.isoformat(),
        "development_start_inclusive": config.raw["windows"].get("development_start"),
        "development_end_exclusive": config.raw["windows"].get("development_end"),
        "sealed_start_inclusive": config.sealed_start.isoformat(),
        "sealed_end_exclusive": config.sealed_end.isoformat(),
        "folds": config.raw["folds"],
        "entry_seconds": entry_seconds,
        "economic_seconds": [
            int(entry["economic_start_second"]), int(entry["economic_end_second_inclusive"])
        ],
        "market_disjoint": True,
        "predictive_fit_excludes_development_and_sealed": True,
        "policy_development_excludes_sealed": True,
        "sealed_replay_epistemically_untouched": False,
    }
    _write_json(run_dir / "split-manifest.json", split)

    preseal = panel.filter(pl.col("window_start") < config.fit_end)
    base_oof = _base_oof(preseal, contracts, config)
    oof, wide_oof = _all_oof(base_oof, config)
    oof.write_parquet(ledgers / "candidate-oof-predictions.parquet", compression="zstd", statistics=True)
    oof_predictive, policies, policy_evidence = {}, {}, {}
    for name in ALL_NAMES:
        candidate = oof.filter(pl.col("candidate") == name)
        oof_predictive[name] = predictive_metrics(candidate)
    predictive_ranking = sorted(ALL_NAMES, key=lambda name: oof_predictive[name]["brier_score"])

    final_models: dict[str, TreeModel | BridgeTreeModel] = {}
    seed = int(config.raw["training"]["random_seed"])
    for index, name in enumerate(BASE_NAMES):
        final_models[name] = _fit_candidate_tree(
            preseal, tuple(contracts[name]["features"]), config, seed + 10_000 + index
        )
    final_calibrator = _fit_ensemble_calibrator(wide_oof, seed + 20_000)
    joblib.dump(
        {"base_models": final_models, "ensemble_calibrator": final_calibrator},
        run_dir / "predictive-model-checkpoint.joblib",
        compress=3,
    )

    if _uses_normalized_supervision(config) or _uses_bridge_supervision(config):
        development_start = datetime.fromisoformat(config.raw["windows"]["development_start"])
        development_end = datetime.fromisoformat(config.raw["windows"]["development_end"])
        development = panel.filter(
            pl.col("window_start").is_between(development_start, development_end, closed="left")
        )
        if set(preseal["market_id"].unique()) & set(development["market_id"].unique()):
            raise RuntimeError("policy-development markets entered predictive training")
        allowed_sources = (
            {"official_twap60_exact", "official_twap60_source_gap"}
            if _uses_bridge_supervision(config)
            else {"exact_chainlink_twap60", "official_twap60_capture_gap"}
        )
        if not set(development["label_source"].unique()) <= allowed_sources:
            raise RuntimeError("policy development contains non-canonical settlement labels")
        development_predictions = _predict_frozen_candidates(
            development, final_models, final_calibrator, "policy_development_20260814_20260820"
        )
        development_predictions.write_parquet(
            ledgers / "development-predictions.parquet", compression="zstd", statistics=True
        )
        policy_source = development_predictions
        policy_panel = development
    else:
        policy_source = oof
        policy_panel = preseal
    for name in ALL_NAMES:
        candidate = policy_source.filter(pl.col("candidate") == name)
        opportunities = _opportunities(candidate, policy_panel, config)
        if name == "price_time_calibrated_middle_ensemble":
            policies[name], policy_evidence[name] = _select_cell_policies(opportunities, config)
        else:
            policies[name], policy_evidence[name] = _select_policy(opportunities, config)
    selection_frozen_at = datetime.now(UTC).isoformat()
    _write_json(run_dir / "selection-freeze.json", {
        "frozen_at": selection_frozen_at, "predictive_ranking": predictive_ranking,
        "policies": policies, "sealed_metrics_accessed": False,
    })

    sealed = panel.filter(pl.col("window_start").is_between(config.sealed_start, config.sealed_end, closed="left"))
    if set(preseal["market_id"].unique()) & set(sealed["market_id"].unique()):
        raise RuntimeError("sealed markets entered training")
    if _uses_normalized_supervision(config) or _uses_bridge_supervision(config):
        allowed_sources = (
            {"official_twap60_exact", "official_twap60_source_gap"}
            if _uses_bridge_supervision(config)
            else {"exact_chainlink_twap60", "official_twap60_capture_gap"}
        )
        if not set(sealed["label_source"].unique()) <= allowed_sources:
            raise RuntimeError("sealed test contains non-canonical settlement labels")
        if set(development["market_id"].unique()) & set(sealed["market_id"].unique()):
            raise RuntimeError("sealed markets entered policy development")
    sealed_predictions = _predict_frozen_candidates(
        sealed, final_models, final_calibrator, "sealed_20260821_20260828"
    )
    sealed_predictions.write_parquet(ledgers / "sealed-predictions.parquet", compression="zstd", statistics=True)

    sealed_predictive, sealed_economic, counterfactual_economic, trade_pieces = {}, {}, {}, []
    total_sealed_markets = sealed["market_id"].n_unique()
    for name in ALL_NAMES:
        prediction = sealed_predictions.filter(pl.col("candidate") == name)
        sealed_predictive[name] = predictive_metrics(prediction)
        opportunities = _opportunities(prediction, sealed, config)
        trades = _apply_candidate_policy(opportunities, policies[name], name, config)
        if not trades.is_empty():
            trade_pieces.append(trades.with_columns(pl.lit(name).alias("candidate")))
        sealed_economic[name] = economic_metrics(trades, total_sealed_markets)
        counterfactual_economic[name] = _counterfactual_economics(
            opportunities,
            policies[name],
            name,
            config,
            total_sealed_markets,
        )
    trade_frame = pl.concat(trade_pieces, how="diagonal_relaxed") if trade_pieces else pl.DataFrame()
    trade_frame.write_parquet(ledgers / "sealed-trades.parquet", compression="zstd", statistics=True)

    producing_commit = _git_revision(config.package_root)
    artifact_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION, "model_family": config.raw["training"]["model_family"],
        "producing_commit": producing_commit, "run_id": run_id, "fit_end_exclusive": config.fit_end.isoformat(),
        "entry_seconds": tuple(entry_seconds),
        "economic_seconds": (
            int(entry["economic_start_second"]), int(entry["economic_end_second_inclusive"])
        ),
        "candidate_contract": contracts,
        "base_models": final_models, "ensemble_calibrator": final_calibrator, "selected_policies": policies,
        "runtime_exported": False, "deployment_status": "not_deployed",
    }
    artifact = run_dir / "tournament.joblib"
    joblib.dump(artifact_payload, artifact, compress=3)
    _validate_artifact(config, artifact)
    artifact_sha = file_sha256(artifact)
    (run_dir / "tournament.sha256").write_text(f"{artifact_sha}  tournament.joblib\n")
    economic_ranking = sorted(ALL_NAMES, key=lambda name: (sealed_economic[name]["stress_net_pnl"], sealed_economic[name]["net_pnl"]), reverse=True)
    metrics = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "created_at": datetime.now(UTC).isoformat(),
        "producing_commit": producing_commit, "candidate_names": list(ALL_NAMES), "candidate_count_trained": 5,
        "learned_base_model_count": 3, "derived_ensemble_count": 2,
        "predictive_ranking_preseal": predictive_ranking, "economic_ranking_sealed": economic_ranking,
        "selection_frozen_at": selection_frozen_at, "oof_predictive": oof_predictive,
        "selected_policies": policies, "policy_selection_evidence": policy_evidence,
        "sealed_predictive": sealed_predictive, "sealed_economic": sealed_economic,
        "sealed_counterfactual_economic": counterfactual_economic,
        "frozen_comparator_references": _reference_manifests(config), "source_panel": panel_manifest,
        "split_manifest": split, "artifact_sha256": artifact_sha,
        "qualification_status": _qualification_status(sealed_economic),
        "data_observed": _observed_ranges(config),
        "integrity": {
            "passed": True, "market_disjoint": True, "sealed_opened_after_selection": True,
            "artifact_round_trip_load": True, "full_history_retained": True,
            "optional_missingness_preserves_rows": True, "twap_inference_feature": False,
            "source_values_remain_distinct": _uses_bridge_supervision(config),
            "historical_official_label_preserved": _uses_bridge_supervision(config),
            "kraken_l2_included": False,
            "database_mutations": False, "new_tables": False, "new_ingesters": False,
            "new_sources": False, "runtime_exported": False, "deployed": False,
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "polars": pl.__version__, "sklearn": sklearn.__version__},
        "limitations": [
            "The August 14-28 development/test period is row-disjoint but was observed during previous research and is not epistemically fresh.",
            "Kraken L2 is excluded because its historical backfill is incomplete.",
            "Qualified Binance spot L2 ends August 1; later rows remain usable with L2 missing.",
            "Projected PnL assumes recorded ask VWAP was fillable and does not model queue position.",
        ],
    }
    metrics["prior_run_comparison"] = _prior_run_comparison(config, metrics)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "candidate-contract.json", contracts)
    _write_json(run_dir / "source-manifest.json", panel_manifest)
    (run_dir / "report.md").write_text(_report(metrics))
    _write_json(run_dir / "completion.json", {
        "run_id": run_id, "artifact_sha256": artifact_sha,
        "metrics_sha256": file_sha256(run_dir / "metrics.json"), "completed": True,
    })
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("command", choices=("extract", "prepare", "train", "finalize", "all"))
    arguments = parser.parse_args()
    config = load_data_config(arguments.config)
    if arguments.command in {"extract", "all"}:
        extract_spot_l2(config, force=arguments.force)
    if arguments.command in {"prepare", "all"}:
        _build_tournament_panel(config, force=arguments.force)
    if arguments.command in {"train", "all"}:
        print(train_tournament(config, force=arguments.force))
    if arguments.command == "finalize":
        if arguments.run_dir is None:
            parser.error("finalize requires --run-dir")
        print(finalize_run(config, arguments.run_dir.resolve()))


if __name__ == "__main__":
    main()
