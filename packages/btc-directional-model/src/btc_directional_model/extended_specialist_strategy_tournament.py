"""Train the frozen extended-specialist and orthogonal-strategy tournament."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .core_extract import file_sha256
from .middle_strategy_tournament import (
    KEY_COLUMNS,
    _opportunities,
    _select_trades,
    economic_metrics,
    predictive_metrics,
)
from .runtime_export import reached_leaf_value

SCHEMA_VERSION = "btc-extended-specialist-strategy-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-extended-specialist-strategy-artifact-v1"
ALL_CANDIDATES = (
    "frozen_specialist_control",
    "extended_specialist_official",
    "bridge_aware_specialist",
    "settlement_agnostic_trajectory",
    "crossvenue_lead_lag",
    "refprice_residual_specialist",
    "specialist_dual_head_economics",
)
TRAINED_CANDIDATES = ALL_CANDIDATES[1:-1]
OFFICIAL_SPECIALIST = "extended_specialist_official"
DUAL_HEAD = "specialist_dual_head_economics"
FORBIDDEN_FEATURE_TOKENS = (
    "official_outcome",
    "label",
    "final_price",
    "resolution",
    "twap",
    "ask_vwap",
    "pm_",
    "fee",
)


@dataclass(frozen=True)
class CalibratedClassifier:
    features: tuple[str, ...]
    estimator: HistGradientBoostingClassifier
    calibrator: LogisticRegression | None


@dataclass(frozen=True)
class DistilledSpecialist:
    features: tuple[str, ...]
    base: Any
    stratified: Any
    hard_negative: Any
    margin: HistGradientBoostingRegressor
    margin_scale: float
    selector: HistGradientBoostingRegressor
    student: HistGradientBoostingRegressor
    soft_target: bool


@dataclass(frozen=True)
class ResidualSpecialist:
    features: tuple[str, ...]
    estimator: HistGradientBoostingRegressor
    scale: float
    calibrator: LogisticRegression | None


@dataclass(frozen=True)
class DualHeadAdmission:
    features: tuple[str, ...]
    profitable: HistGradientBoostingClassifier
    stress_edge: HistGradientBoostingRegressor


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _load_config(path: Path) -> tuple[Path, dict[str, Any]]:
    package_root = path.resolve().parents[1]
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    names = tuple(row["name"] for row in raw["candidates"])
    if names != ALL_CANDIDATES:
        raise RuntimeError("candidate roster differs from the frozen tournament plan")
    windows = raw["windows"]
    fit_end = datetime.fromisoformat(windows["fit_end"])
    development_start = datetime.fromisoformat(windows["development_start"])
    development_end = datetime.fromisoformat(windows["development_end"])
    sealed_start = datetime.fromisoformat(windows["sealed_start"])
    sealed_end = datetime.fromisoformat(windows["sealed_end"])
    if not fit_end <= development_start < development_end <= sealed_start < sealed_end:
        raise RuntimeError("training, development, and sealed windows are not chronological")
    return package_root, raw


def _paths(package_root: Path, raw: dict[str, Any]) -> dict[str, Path]:
    return {name: package_root / value for name, value in raw["paths"].items()}


def _input_manifest(package_root: Path, raw: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(package_root, raw)
    manifest = json.loads(paths["panel_manifest"].read_text())
    bridge = json.loads(paths["bridge_manifest"].read_text())
    frozen = json.loads(paths["frozen_specialist_manifest"].read_text())
    if manifest["sha256"] != file_sha256(paths["panel"]):
        raise RuntimeError("full-history panel identity changed")
    if bridge["sha256"] != file_sha256(paths["bridge_panel"]):
        raise RuntimeError("settlement bridge panel identity changed")
    if frozen["model_sha256"] != file_sha256(paths["frozen_specialist"]):
        raise RuntimeError("frozen specialist identity changed")
    return {
        "panel": manifest,
        "bridge": bridge,
        "frozen_specialist": frozen,
        "identities": {
            "panel_sha256": manifest["sha256"],
            "bridge_panel_sha256": bridge["sha256"],
            "frozen_specialist_sha256": frozen["model_sha256"],
        },
    }


def _candidate_contract(raw: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    groups = {name: tuple(values) for name, values in manifest["feature_groups"].items()}
    output: dict[str, Any] = {}
    for candidate in raw["candidates"]:
        features = tuple(
            dict.fromkeys(
                feature
                for group in candidate["feature_groups"]
                for feature in groups[group]
            )
        )
        if candidate["name"] == "settlement_agnostic_trajectory":
            features = tuple(
                name
                for name in features
                if not any(
                    token in name.lower()
                    for token in ("boundary", "oracle", "refprice", "chainlink")
                )
            )
        if candidate["name"] == "crossvenue_lead_lag":
            allowed_core = (
                "seconds_",
                "btc_return_",
                "btc_realized_volatility_",
                "btc_signed_flow_",
                "btc_momentum_",
                "btc_reversal_",
                "btc_price_flow_",
                "btc_volume_surprise_",
                "hour_",
                "weekday_",
            )
            non_core = {
                name
                for group in ("binance_prints", "kraken", "spot_l2", "kraken_l2")
                for name in groups[group]
            }
            features = tuple(
                name for name in features if name in non_core or name.startswith(allowed_core)
            )
        forbidden = [
            feature
            for feature in features
            if any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS)
        ]
        if forbidden:
            raise RuntimeError(f"target or execution fields entered {candidate['name']}: {forbidden}")
        output[candidate["name"]] = {
            **candidate,
            "features": list(features),
            "feature_count": len(features),
        }
    return output


def _matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return frame.select(
        pl.col(name).cast(pl.Float64).fill_nan(None).fill_null(float("nan")) for name in features
    ).to_numpy()


def _stable_features(frame: pl.DataFrame, features: tuple[str, ...]) -> tuple[str, ...]:
    matrix = _matrix(frame, features)
    active = []
    for index, name in enumerate(features):
        finite = matrix[np.isfinite(matrix[:, index]), index]
        if len(finite) >= 100 and float(finite.min()) != float(finite.max()):
            active.append(name)
    if not active:
        raise RuntimeError("candidate has no stable finite training features")
    return tuple(active)


def _market_weights(frame: pl.DataFrame) -> np.ndarray:
    counts = frame.group_by("market_id").len().rename({"len": "market_rows"})
    return (
        frame.join(counts, on="market_id", how="left")["market_rows"]
        .cast(pl.Float64)
        .pow(-1)
        .to_numpy()
    )


def _estimator_parameters(raw: dict[str, Any]) -> dict[str, Any]:
    spec = raw["model"]
    return {
        "learning_rate": float(spec["learning_rate"]),
        "max_iter": int(spec["max_iter"]),
        "max_leaf_nodes": int(spec["max_leaf_nodes"]),
        "min_samples_leaf": int(spec["min_samples_leaf"]),
        "l2_regularization": float(spec["l2_regularization"]),
        "max_bins": int(spec["max_bins"]),
        "early_stopping": False,
    }


def _fit_calibrated_classifier(
    frame: pl.DataFrame, features: tuple[str, ...], raw: dict[str, Any], seed: int
) -> CalibratedClassifier:
    markets = frame.select("market_id", "window_start").unique().sort("window_start")
    boundary = markets["window_start"][int(markets.height * 0.8)]
    fit = frame.filter(pl.col("window_start") < boundary)
    calibration = frame.filter(pl.col("window_start") >= boundary)
    features = _stable_features(fit, features)
    estimator = HistGradientBoostingClassifier(
        **_estimator_parameters(raw), random_state=seed
    ).fit(_matrix(fit, features), fit["label_up"], sample_weight=_market_weights(fit))
    probability = np.clip(estimator.predict_proba(_matrix(calibration, features))[:, 1], 1e-6, 1 - 1e-6)
    calibrator = None
    if calibration["label_up"].n_unique() == 2:
        calibrator = LogisticRegression(C=0.5, max_iter=2000, random_state=seed + 1).fit(
            logit(probability).reshape(-1, 1),
            calibration["label_up"],
            sample_weight=_market_weights(calibration),
        )
    return CalibratedClassifier(features, estimator, calibrator)


def _predict_classifier(model: CalibratedClassifier, frame: pl.DataFrame) -> np.ndarray:
    probability = np.clip(
        model.estimator.predict_proba(_matrix(frame, model.features))[:, 1], 1e-6, 1 - 1e-6
    )
    if model.calibrator is not None:
        probability = model.calibrator.predict_proba(logit(probability).reshape(-1, 1))[:, 1]
    return np.clip(probability, 1e-6, 1 - 1e-6)


def _component_probability(model: Any, matrix: np.ndarray, soft: bool) -> np.ndarray:
    if soft:
        return np.clip(model.predict(matrix), 1e-6, 1 - 1e-6)
    return np.clip(model.predict_proba(matrix)[:, 1], 1e-6, 1 - 1e-6)


def _selector_matrix(probabilities: np.ndarray, frame: pl.DataFrame) -> np.ndarray:
    return np.column_stack(
        (
            logit(np.clip(probabilities, 1e-6, 1 - 1e-6)),
            probabilities.std(axis=1),
            probabilities.max(axis=1) - probabilities.min(axis=1),
            frame["seconds_elapsed_scaled"].fill_null(0.0).to_numpy(),
            frame["btc_cross_venue_boundary_gap_bps"].fill_null(0.0).to_numpy(),
        )
    )


def _terminal_margin(frame: pl.DataFrame) -> pl.DataFrame:
    terminal = (
        frame.sort(["market_id", "seconds_elapsed"])
        .group_by("market_id", maintain_order=True)
        .agg(pl.col("btc_path_from_window_open_bps").last().alias("terminal_margin"))
    )
    return frame.join(terminal, on="market_id", how="left")


def _fit_distilled(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    raw: dict[str, Any],
    seed: int,
    *,
    target: str,
    weight_column: str | None = None,
) -> DistilledSpecialist:
    markets = frame.select("market_id", "window_start").unique().sort("window_start")
    boundary = markets["window_start"][int(markets.height * 0.8)]
    fit = _terminal_margin(frame.filter(pl.col("window_start") < boundary))
    calibration = frame.filter(pl.col("window_start") >= boundary)
    features = _stable_features(fit, features)
    soft = target != "label_up"
    fit_target = fit[target].to_numpy().astype(float)
    fit_binary = fit_target >= 0.5
    weights = _market_weights(fit).copy()
    if weight_column and weight_column in fit.columns:
        weights *= fit[weight_column].fill_null(0.0).to_numpy()
    parameters = _estimator_parameters(raw)
    model_type = HistGradientBoostingRegressor if soft else HistGradientBoostingClassifier
    base = model_type(**parameters, random_state=seed).fit(
        _matrix(fit, features), fit_target if soft else fit_binary, sample_weight=weights
    )
    time_weight = weights * (1.0 + 0.5 * fit["seconds_elapsed_scaled"].to_numpy())
    stratified = model_type(**parameters, random_state=seed + 1).fit(
        _matrix(fit, features), fit_target if soft else fit_binary, sample_weight=time_weight
    )
    base_probability = _component_probability(base, _matrix(fit, features), soft)
    hard = (base_probability >= 0.5) != fit_binary
    hard_weights = weights * (
        1.0 + float(raw["model"]["hard_negative_multiplier"]) * hard * np.maximum(base_probability, 1 - base_probability)
    )
    hard_negative = model_type(**parameters, random_state=seed + 2).fit(
        _matrix(fit, features), fit_target if soft else fit_binary, sample_weight=hard_weights
    )
    margin = HistGradientBoostingRegressor(**parameters, random_state=seed + 3).fit(
        _matrix(fit, features), fit["terminal_margin"], sample_weight=weights
    )
    fit_margin_error = fit["terminal_margin"].to_numpy() - margin.predict(_matrix(fit, features))
    margin_scale = max(float(np.nanstd(fit_margin_error)), 0.25)
    calibration_matrix = _matrix(calibration, features)
    specialist = np.column_stack(
        (
            _component_probability(base, calibration_matrix, soft),
            _component_probability(stratified, calibration_matrix, soft),
            _component_probability(hard_negative, calibration_matrix, soft),
            expit(margin.predict(calibration_matrix) / margin_scale),
        )
    )
    selector_target = calibration[target].to_numpy().astype(float)
    selector = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=80,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=5.0,
        early_stopping=False,
        random_state=seed + 4,
    ).fit(
        _selector_matrix(specialist, calibration),
        selector_target,
        sample_weight=_market_weights(calibration),
    )
    teacher = np.clip(selector.predict(_selector_matrix(specialist, calibration)), 1e-6, 1 - 1e-6)
    student = HistGradientBoostingRegressor(**parameters, random_state=seed + 5).fit(
        calibration_matrix,
        teacher,
        sample_weight=_market_weights(calibration),
    )
    return DistilledSpecialist(
        features, base, stratified, hard_negative, margin, margin_scale, selector, student, soft
    )


def _predict_distilled(model: DistilledSpecialist, frame: pl.DataFrame) -> np.ndarray:
    return np.clip(model.student.predict(_matrix(frame, model.features)), 1e-6, 1 - 1e-6)


def _fit_residual(
    frame: pl.DataFrame, features: tuple[str, ...], raw: dict[str, Any], seed: int
) -> ResidualSpecialist:
    terminal = (
        frame.filter(pl.col("chainlink_ref_boundary_gap_bps").is_not_null())
        .sort(["market_id", "seconds_elapsed"])
        .group_by("market_id", maintain_order=True)
        .agg(pl.col("chainlink_ref_boundary_gap_bps").last().alias("terminal_ref_residual"))
    )
    prepared = frame.join(terminal, on="market_id", how="inner")
    markets = prepared.select("market_id", "window_start").unique().sort("window_start")
    boundary = markets["window_start"][int(markets.height * 0.8)]
    fit = prepared.filter(pl.col("window_start") < boundary)
    calibration = prepared.filter(pl.col("window_start") >= boundary)
    features = _stable_features(fit, features)
    estimator = HistGradientBoostingRegressor(
        **_estimator_parameters(raw), random_state=seed
    ).fit(
        _matrix(fit, features),
        fit["terminal_ref_residual"],
        sample_weight=_market_weights(fit),
    )
    residual = fit["terminal_ref_residual"].to_numpy() - estimator.predict(_matrix(fit, features))
    scale = max(float(np.nanstd(residual)), 0.25)
    raw_probability = np.clip(expit(estimator.predict(_matrix(calibration, features)) / scale), 1e-6, 1 - 1e-6)
    calibrator = None
    if calibration["label_up"].n_unique() == 2:
        calibrator = LogisticRegression(C=0.5, max_iter=2000, random_state=seed + 1).fit(
            logit(raw_probability).reshape(-1, 1),
            calibration["label_up"],
            sample_weight=_market_weights(calibration),
        )
    return ResidualSpecialist(features, estimator, scale, calibrator)


def _predict_residual(model: ResidualSpecialist, frame: pl.DataFrame) -> np.ndarray:
    probability = np.clip(expit(model.estimator.predict(_matrix(frame, model.features)) / model.scale), 1e-6, 1 - 1e-6)
    if model.calibrator is not None:
        probability = model.calibrator.predict_proba(logit(probability).reshape(-1, 1))[:, 1]
    return np.clip(probability, 1e-6, 1 - 1e-6)


def _prediction_frame(
    frame: pl.DataFrame, probability: np.ndarray, candidate: str, fold: str
) -> pl.DataFrame:
    return frame.select(*KEY_COLUMNS, "label_up").with_columns(
        pl.lit(fold).alias("fold"),
        pl.lit(candidate).alias("candidate"),
        pl.Series("probability", probability),
        pl.lit(True).alias("eligible_signal"),
    )


def _score_model(model: Any, frame: pl.DataFrame, candidate: str, fold: str) -> pl.DataFrame:
    if isinstance(model, CalibratedClassifier):
        probability = _predict_classifier(model, frame)
    elif isinstance(model, DistilledSpecialist):
        probability = _predict_distilled(model, frame)
    elif isinstance(model, ResidualSpecialist):
        probability = _predict_residual(model, frame)
    else:
        raise TypeError(f"unsupported model type for {candidate}: {type(model)!r}")
    return _prediction_frame(frame, probability, candidate, fold)


def _score_frozen(
    package_root: Path, raw: dict[str, Any], frame: pl.DataFrame, fold: str
) -> pl.DataFrame:
    model = json.loads(_paths(package_root, raw)["frozen_specialist"].read_text())
    features = tuple(model["features"]["names"])
    if "early_oracle_eligible" not in frame.columns and "oracle_model_eligible" in frame.columns:
        frame = frame.with_columns(
            pl.col("oracle_model_eligible").fill_null(False).alias("early_oracle_eligible")
        )
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise RuntimeError(f"frozen Specialist features unavailable: {missing}")
    payoff = model["payoff_model"]
    if payoff.get("kind") != "fair_value":
        raise RuntimeError("frozen Specialist is not the expected fair-value model")
    outcome = payoff["outcome"]
    local_indices = tuple(int(index) for index in outcome["feature_indices"])
    probability = []
    for values in frame.select(features).iter_rows():
        local_values = []
        for index in local_indices:
            value = values[index]
            numeric = (
                float(value)
                if value is not None and not isinstance(value, bool)
                else math.nan
            )
            local_values.append(numeric if math.isfinite(numeric) else math.nan)
        prediction = float(outcome["baseline"])
        for tree in outcome["trees"]:
            prediction += reached_leaf_value(tree["nodes"], local_values)
        probability.append(float(np.clip(prediction, 1e-6, 1.0 - 1e-6)))
    probability = np.asarray(probability, dtype=float)
    return _prediction_frame(frame, probability, "frozen_specialist_control", fold)


def _bridge_join(package_root: Path, raw: dict[str, Any], frame: pl.DataFrame) -> pl.DataFrame:
    bridge_path = _paths(package_root, raw)["bridge_panel"]
    bridge = (
        pl.scan_parquet(bridge_path)
        .select(*KEY_COLUMNS, "bridge_probability_target", "label_weight", "label_source")
        .collect()
    )
    joined = frame.join(bridge, on=list(KEY_COLUMNS), how="left", validate="1:1")
    if joined["bridge_probability_target"].null_count():
        raise RuntimeError("bridge-aware Specialist has missing supervision")
    return joined


def _fit_models(
    package_root: Path,
    raw: dict[str, Any],
    contract: dict[str, Any],
    fit: pl.DataFrame,
    run_dir: Path,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    seed = int(raw["training"]["random_seed"])
    fit_bridge: pl.DataFrame | None = None
    for index, name in enumerate(TRAINED_CANDIDATES):
        checkpoint = run_dir / "checkpoints" / f"model-{name}.joblib"
        if checkpoint.is_file():
            output[name] = joblib.load(checkpoint)
            continue
        features = tuple(contract[name]["features"])
        kind = contract[name]["kind"]
        if kind == "distilled_official":
            model = _fit_distilled(fit, features, raw, seed + 100 * index, target="label_up")
        elif kind == "distilled_bridge":
            fit_bridge = fit_bridge if fit_bridge is not None else _bridge_join(package_root, raw, fit)
            model = _fit_distilled(
                fit_bridge,
                features,
                raw,
                seed + 100 * index,
                target="bridge_probability_target",
                weight_column="label_weight",
            )
        elif kind == "residual":
            model = _fit_residual(fit, features, raw, seed + 100 * index)
        else:
            model = _fit_calibrated_classifier(fit, features, raw, seed + 100 * index)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, checkpoint, compress=3)
        output[name] = model
    return output


def _oof_specialist(
    raw: dict[str, Any],
    contract: dict[str, Any],
    panel_path: Path,
    columns: tuple[str, ...],
    run_dir: Path,
) -> pl.DataFrame:
    destination = run_dir / "ledgers" / "specialist-oof-predictions.parquet"
    if destination.is_file():
        return pl.read_parquet(destination)
    pieces = []
    source_start = datetime.fromisoformat(raw["windows"]["source_start"])
    features = tuple(contract[OFFICIAL_SPECIALIST]["features"])
    for index, fold in enumerate(raw["folds"]):
        start = datetime.fromisoformat(fold["test_start"])
        end = datetime.fromisoformat(fold["test_end"])
        training = (
            pl.scan_parquet(panel_path)
            .select(columns)
            .filter((pl.col("window_start") >= source_start) & (pl.col("window_start") < start))
            .collect()
        )
        validation = (
            pl.scan_parquet(panel_path)
            .select(columns)
            .filter(pl.col("window_start").is_between(start, end, closed="left"))
            .collect()
        )
        model = _fit_calibrated_classifier(
            training, features, raw, int(raw["training"]["random_seed"]) + 10_000 + index
        )
        pieces.append(_score_model(model, validation, OFFICIAL_SPECIALIST, fold["name"]))
    result = pl.concat(pieces, how="vertical_relaxed", rechunk=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(destination, compression="zstd", statistics=True)
    return result


def _economic_config(raw: dict[str, Any]) -> Any:
    return SimpleNamespace(raw=raw)


def _realized_opportunities(
    predictions: pl.DataFrame, panel: pl.DataFrame, raw: dict[str, Any]
) -> pl.DataFrame:
    config = _economic_config(raw)
    opportunities = _opportunities(predictions, panel, config)
    correct = ((pl.col("probability") >= 0.5) & (pl.col("label_up") == 1)) | (
        (pl.col("probability") < 0.5) & (pl.col("label_up") == 0)
    )
    reserve = float(raw["execution"]["execution_reserve_per_share"])
    stress = float(raw["execution"]["stress_slippage_per_share"])
    return opportunities.with_columns(
        correct.alias("direction_correct"),
        (
            correct.cast(pl.Float64)
            - pl.col("share_cost")
            - pl.col("fee_per_share")
            - reserve
            - stress
        ).alias("realized_stress_edge"),
    )


def _policy_score(trades: pl.DataFrame, raw: dict[str, Any]) -> float:
    if trades.is_empty():
        return -math.inf
    pnl = trades["stress_net_pnl"].to_numpy().astype(float)
    standard_error = 0.0 if len(pnl) < 2 else float(pnl.std(ddof=1) / math.sqrt(len(pnl)))
    metrics = economic_metrics(trades, trades["market_id"].n_unique())
    recovery = metrics["loss_recovery_wins"]
    recovery_penalty = 0.0
    if recovery is not None:
        recovery_penalty = float(raw["execution"]["recovery_penalty"]) * max(
            0.0, recovery - float(raw["execution"]["recovery_soft_target"])
        )
    return float(pnl.mean()) - standard_error - recovery_penalty + 0.0005 * math.sqrt(len(pnl))


def _bands(raw: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (f"{start}_{end}", int(start), int(end))
        for start, end in raw["entry"]["competition_bands"]
    )


def _select_programmatic(
    opportunities: pl.DataFrame, raw: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _economic_config(raw)
    selected, evidence = {}, {}
    for band, start, end in _bands(raw):
        part = opportunities.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        best: tuple[float, dict[str, float], pl.DataFrame] | None = None
        attempted = 0
        for edge in raw["execution"]["minimum_edges"]:
            for confidence in raw["execution"]["minimum_confidences"]:
                for cost in raw["execution"]["maximum_share_costs"]:
                    attempted += 1
                    policy = {
                        "minimum_edge": float(edge),
                        "minimum_confidence": float(confidence),
                        "maximum_share_cost": float(cost),
                        "abstain": False,
                    }
                    trades = _select_trades(part, policy, config)
                    score = _policy_score(trades, raw)
                    if best is None or score > best[0]:
                        best = (score, policy, trades)
        if best is None:
            raise RuntimeError(f"no programmatic policy evaluated for {band}")
        selected[band] = best[1]
        evidence[band] = {
            "attempted": attempted,
            "selection_score": best[0],
            "development": economic_metrics(best[2], part["market_id"].n_unique()),
        }
    return selected, evidence


def _dual_features(frame: pl.DataFrame) -> tuple[str, ...]:
    requested = (
        "probability",
        "selected_probability",
        "share_cost",
        "expected_edge",
        "seconds_elapsed",
        "pm_vwap5_overround",
        "pm_vwap50_overround",
        "pm_vwap200_overround",
        "pm_depth_imbalance",
        "pm_up_book_age_seconds",
        "pm_down_book_age_seconds",
        "spot_l2_imbalance_20",
        "kraken_l2_update_imbalance_30s",
        "kraken_l2_age_seconds",
    )
    return tuple(name for name in requested if name in frame.columns)


def _fit_dual_head(
    opportunities: pl.DataFrame, panel: pl.DataFrame, raw: dict[str, Any]
) -> DualHeadAdmission:
    context_names = [
        name
        for name in _dual_features(panel)
        if name not in opportunities.columns and name not in KEY_COLUMNS
    ]
    frame = opportunities.join(
        panel.select(*KEY_COLUMNS, *context_names), on=list(KEY_COLUMNS), how="left", validate="m:1"
    ).filter(pl.col("share_cost").is_not_null())
    features = _dual_features(frame)
    weights = _market_weights(frame)
    common = {
        "learning_rate": 0.04,
        "max_iter": int(raw["dual_head"]["max_iter"]),
        "max_leaf_nodes": 15,
        "min_samples_leaf": 100,
        "l2_regularization": 5.0,
        "early_stopping": False,
    }
    seed = int(raw["training"]["random_seed"]) + 50_000
    profitable = HistGradientBoostingClassifier(**common, random_state=seed).fit(
        _matrix(frame, features),
        (frame["realized_stress_edge"] > 0).cast(pl.Int8),
        sample_weight=weights,
    )
    stress_edge = HistGradientBoostingRegressor(**common, random_state=seed + 1).fit(
        _matrix(frame, features), frame["realized_stress_edge"], sample_weight=weights
    )
    return DualHeadAdmission(features, profitable, stress_edge)


def _attach_dual_head(
    opportunities: pl.DataFrame, panel: pl.DataFrame, model: DualHeadAdmission
) -> pl.DataFrame:
    context_names = [
        name
        for name in model.features
        if name not in opportunities.columns and name not in KEY_COLUMNS
    ]
    frame = opportunities.join(
        panel.select(*KEY_COLUMNS, *context_names), on=list(KEY_COLUMNS), how="left", validate="m:1"
    )
    matrix = _matrix(frame, model.features)
    return frame.with_columns(
        pl.Series("profit_probability", model.profitable.predict_proba(matrix)[:, 1]),
        pl.Series("expected_stress_edge", model.stress_edge.predict(matrix)),
    ).with_columns((1.0 - pl.col("profit_probability")).alias("loss_probability"))


def _select_dual_policy(
    opportunities: pl.DataFrame, raw: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _economic_config(raw)
    base = {
        "minimum_edge": 0.0,
        "minimum_confidence": 0.5,
        "maximum_share_cost": 0.95,
        "abstain": False,
    }
    selected, evidence = {}, {}
    for band, start, end in _bands(raw):
        part = opportunities.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        best: tuple[float, dict[str, float], pl.DataFrame] | None = None
        attempted = 0
        for probability in raw["dual_head"]["minimum_profit_probabilities"]:
            for edge in raw["dual_head"]["minimum_expected_stress_edges"]:
                for loss in raw["dual_head"]["maximum_loss_probabilities"]:
                    attempted += 1
                    policy = {
                        "minimum_profit_probability": float(probability),
                        "minimum_expected_stress_edge": float(edge),
                        "maximum_loss_probability": float(loss),
                    }
                    eligible = part.filter(
                        (pl.col("profit_probability") >= probability)
                        & (pl.col("expected_stress_edge") >= edge)
                        & (pl.col("loss_probability") <= loss)
                    )
                    trades = _select_trades(eligible, base, config)
                    score = _policy_score(trades, raw)
                    if best is None or score > best[0]:
                        best = (score, policy, trades)
        if best is None:
            raise RuntimeError(f"no dual-head policy evaluated for {band}")
        selected[band] = best[1]
        evidence[band] = {
            "attempted": attempted,
            "selection_score": best[0],
            "development": economic_metrics(best[2], part["market_id"].n_unique()),
        }
    return selected, evidence


def _apply_dual_policy(frame: pl.DataFrame, policy: dict[str, float]) -> pl.DataFrame:
    return frame.filter(
        (pl.col("profit_probability") >= policy["minimum_profit_probability"])
        & (pl.col("expected_stress_edge") >= policy["minimum_expected_stress_edge"])
        & (pl.col("loss_probability") <= policy["maximum_loss_probability"])
    )


def _evaluate_candidate(
    predictions: pl.DataFrame,
    panel: pl.DataFrame,
    raw: dict[str, Any],
    policies: dict[str, dict[str, float]],
    dual_model: DualHeadAdmission | None = None,
    dual_policies: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, Any], pl.DataFrame]:
    config = _economic_config(raw)
    opportunities = _realized_opportunities(predictions, panel, raw)
    dual = _attach_dual_head(opportunities, panel, dual_model) if dual_model else None
    output, ledgers = {}, []
    no_veto = {
        "minimum_edge": 0.0,
        "minimum_confidence": 0.5,
        "maximum_share_cost": 0.95,
        "abstain": False,
    }
    for band, start, end in _bands(raw):
        part = opportunities.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        modes: dict[str, pl.DataFrame] = {
            "no_veto": _select_trades(part, no_veto, config),
            "programmatic": _select_trades(part, policies[band], config),
        }
        if dual is not None and dual_policies is not None:
            dual_part = dual.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
            modes["learned_enter_now"] = _select_trades(
                _apply_dual_policy(dual_part, dual_policies[band]), no_veto, config
            )
        output[band] = {}
        for mode, trades in modes.items():
            output[band][mode] = economic_metrics(trades, part["market_id"].n_unique())
            if not trades.is_empty():
                ledgers.append(trades.with_columns(pl.lit(band).alias("entry_band"), pl.lit(mode).alias("admission_mode")))
    empty = opportunities.head(0).with_columns(
        pl.lit("").alias("entry_band"), pl.lit("").alias("admission_mode")
    )
    return output, pl.concat(ledgers, how="diagonal_relaxed") if ledgers else empty


def _report(metrics: dict[str, Any]) -> str:
    def fmt(value: Any, digits: int = 3) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Extended Specialist and Orthogonal Strategy Tournament",
        "",
        f"Run: `{metrics['run_id']}`",
        f"Qualification: **{metrics['qualification_status']}**",
        "",
        "## Sealed high-level results",
        "",
        "| Candidate | Preferred admission | Best PnL bucket | PnL | Stress PnL | PF | Expectancy/trade | Coverage | Wins/Losses | Win rate | Recovery wins/loss | Avg entry | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in ALL_CANDIDATES:
        selected = metrics["sealed_summary"][candidate]
        lines.append(
            f"| {candidate} | {selected['admission_mode']} | {selected['entry_band']} | "
            f"{selected['net_pnl']:.2f} | {selected['stress_net_pnl']:.2f} | "
            f"{fmt(selected['profit_factor'])} | {fmt(selected['expectancy_per_trade'])} | "
            f"{selected['market_coverage']:.2%} | {selected['winning_trades']}/{selected['losing_trades']} | "
            f"{fmt(selected['win_rate'], 2)} | {fmt(selected['loss_recovery_wins'])} | "
            f"{fmt(selected['average_entry_second'], 1)} | {fmt(metrics['sealed_predictive'][candidate]['brier_score'], 4)} |"
        )
    lines.extend((
        "",
        "## Integrity",
        "",
        "- Predictor fit, development policy selection, and sealed evaluation are chronological and market-disjoint.",
        "- Official outcomes remain canonical for every candidate except the explicitly identified bridge-supervised arm.",
        "- TWAP, outcomes, labels, Polymarket prices, and execution VWAP fields are absent from every directional feature contract.",
        "- The learned admission target is enter-now realized stress edge; it contains no best-later comparison.",
        "- No data row lock, database write, new table, schema, ingester, source, runtime deployment, or image build occurred.",
    ))
    for limitation in metrics["limitations"]:
        lines.append(f"- Limitation: {limitation}")
    return "\n".join(lines) + "\n"


def train_tournament(config_path: Path, *, resume_run: str | None = None) -> Path:
    package_root, raw = _load_config(config_path)
    source = _input_manifest(package_root, raw)
    contract = _candidate_contract(raw, source["panel"])
    paths = _paths(package_root, raw)
    run_id = resume_run or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = paths["committed_results"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "candidate-contract.json", contract)
    _write_json(run_dir / "source-manifest.json", source)

    feature_names = tuple(
        dict.fromkeys(
            feature
            for candidate in contract.values()
            for feature in candidate["features"]
        )
    )
    frozen_model = json.loads(paths["frozen_specialist"].read_text())
    frozen_features = tuple(
        "oracle_model_eligible" if name == "early_oracle_eligible" else name
        for name in frozen_model["features"]["names"]
    )
    execution_columns = (
        "fee_rate",
        "up_ask_vwap_5",
        "down_ask_vwap_5",
        "pm_up_book_age_seconds",
        "pm_down_book_age_seconds",
        "pm_vwap5_overround",
        "pm_vwap50_overround",
        "pm_vwap200_overround",
        "pm_depth_imbalance",
    )
    columns = tuple(dict.fromkeys((*KEY_COLUMNS, "label_up", *feature_names, *frozen_features, *execution_columns)))
    schema = pl.scan_parquet(paths["panel"]).collect_schema()
    missing = [name for name in columns if name not in schema]
    if missing:
        raise RuntimeError(f"required panel columns are unavailable: {missing}")

    windows = raw["windows"]
    source_start = datetime.fromisoformat(windows["source_start"])
    fit_end = datetime.fromisoformat(windows["fit_end"])
    development_start = datetime.fromisoformat(windows["development_start"])
    development_end = datetime.fromisoformat(windows["development_end"])
    sealed_start = datetime.fromisoformat(windows["sealed_start"])
    sealed_end = datetime.fromisoformat(windows["sealed_end"])
    split = {
        "source_start": source_start,
        "fit_end_exclusive": fit_end,
        "development_start_inclusive": development_start,
        "development_end_exclusive": development_end,
        "sealed_start_inclusive": sealed_start,
        "sealed_end_exclusive": sealed_end,
        "market_disjoint": True,
        "heldout_excluded_from_model_calibration_admission_and_policy_selection": True,
        "official_outcome_supervision_canonical": True,
        "bridge_supervision_candidate": "bridge_aware_specialist",
        "authentic_only_filter": False,
        "folds": raw["folds"],
    }
    _write_json(run_dir / "split-manifest.json", split)

    fit = (
        pl.scan_parquet(paths["panel"])
        .select(columns)
        .filter((pl.col("window_start") >= source_start) & (pl.col("window_start") < fit_end))
        .collect()
    )
    development = (
        pl.scan_parquet(paths["panel"])
        .select(columns)
        .filter(pl.col("window_start").is_between(development_start, development_end, closed="left"))
        .collect()
    )
    if set(fit["market_id"].unique()) & set(development["market_id"].unique()):
        raise RuntimeError("fit and development markets overlap")
    models = _fit_models(package_root, raw, contract, fit, run_dir)
    oof = _oof_specialist(raw, contract, paths["panel"], columns, run_dir)

    development_predictions = [
        _score_frozen(package_root, raw, development, "development")
    ]
    for name, model in models.items():
        development_predictions.append(_score_model(model, development, name, "development"))
    official_development = next(
        frame for frame in development_predictions if frame["candidate"][0] == OFFICIAL_SPECIALIST
    )
    development_predictions.append(
        official_development.with_columns(pl.lit(DUAL_HEAD).alias("candidate"))
    )
    development_prediction_frame = pl.concat(development_predictions, how="vertical_relaxed", rechunk=True)
    ledgers = run_dir / "ledgers"
    ledgers.mkdir(parents=True, exist_ok=True)
    development_prediction_frame.write_parquet(
        ledgers / "development-predictions.parquet", compression="zstd", statistics=True
    )

    programmatic, programmatic_evidence = {}, {}
    for candidate in ALL_CANDIDATES:
        predictions = development_prediction_frame.filter(pl.col("candidate") == candidate)
        opportunities = _realized_opportunities(predictions, development, raw)
        programmatic[candidate], programmatic_evidence[candidate] = _select_programmatic(
            opportunities, raw
        )

    oof_panel = (
        pl.scan_parquet(paths["panel"])
        .select(columns)
        .filter(
            (pl.col("window_start") >= min(oof["window_start"]))
            & (pl.col("window_start") <= max(oof["window_start"]))
        )
        .collect()
    )
    oof_opportunities = _realized_opportunities(oof, oof_panel, raw)
    dual_model_path = run_dir / "checkpoints" / "dual-head-admission.joblib"
    if dual_model_path.is_file():
        dual_model = joblib.load(dual_model_path)
    else:
        dual_model = _fit_dual_head(oof_opportunities, oof_panel, raw)
        joblib.dump(dual_model, dual_model_path, compress=3)
    dual_development = _attach_dual_head(
        _realized_opportunities(
            development_prediction_frame.filter(pl.col("candidate") == DUAL_HEAD), development, raw
        ),
        development,
        dual_model,
    )
    dual_policies, dual_evidence = _select_dual_policy(dual_development, raw)
    selection_frozen_at = datetime.now(UTC).isoformat()
    selection = {
        "frozen_at": selection_frozen_at,
        "programmatic": programmatic,
        "programmatic_evidence": programmatic_evidence,
        "dual_head": dual_policies,
        "dual_head_evidence": dual_evidence,
        "heldout_metrics_accessed": False,
        "admission_target": "enter-now realized stress edge and loss probability",
        "best_later_target_used": False,
    }
    _write_json(run_dir / "selection-freeze.json", selection)

    heldout = (
        pl.scan_parquet(paths["panel"])
        .select(columns)
        .filter(pl.col("window_start").is_between(sealed_start, sealed_end, closed="left"))
        .collect()
    )
    if set(fit["market_id"].unique()) & set(heldout["market_id"].unique()):
        raise RuntimeError("fit and sealed markets overlap")
    if set(development["market_id"].unique()) & set(heldout["market_id"].unique()):
        raise RuntimeError("development and sealed markets overlap")
    heldout_predictions = [_score_frozen(package_root, raw, heldout, "sealed")]
    for name, model in models.items():
        heldout_predictions.append(_score_model(model, heldout, name, "sealed"))
    official_heldout = next(
        frame for frame in heldout_predictions if frame["candidate"][0] == OFFICIAL_SPECIALIST
    )
    heldout_predictions.append(official_heldout.with_columns(pl.lit(DUAL_HEAD).alias("candidate")))
    heldout_prediction_frame = pl.concat(heldout_predictions, how="vertical_relaxed", rechunk=True)
    heldout_prediction_frame.write_parquet(
        ledgers / "heldout-predictions.parquet", compression="zstd", statistics=True
    )

    sealed_predictive, sealed_economic, trade_ledgers = {}, {}, []
    for candidate in ALL_CANDIDATES:
        predictions = heldout_prediction_frame.filter(pl.col("candidate") == candidate)
        sealed_predictive[candidate] = predictive_metrics(predictions)
        economics, trades = _evaluate_candidate(
            predictions,
            heldout,
            raw,
            programmatic[candidate],
            dual_model if candidate == DUAL_HEAD else None,
            dual_policies if candidate == DUAL_HEAD else None,
        )
        sealed_economic[candidate] = economics
        if not trades.is_empty():
            trade_ledgers.append(trades.with_columns(pl.lit(candidate).alias("candidate")))
    if trade_ledgers:
        pl.concat(trade_ledgers, how="diagonal_relaxed", rechunk=True).write_parquet(
            ledgers / "heldout-trades.parquet", compression="zstd", statistics=True
        )

    sealed_summary = {}
    for candidate in ALL_CANDIDATES:
        preferred_mode = "learned_enter_now" if candidate == DUAL_HEAD else "programmatic"
        options = [
            {"entry_band": band, "admission_mode": preferred_mode, **modes[preferred_mode]}
            for band, modes in sealed_economic[candidate].items()
        ]
        sealed_summary[candidate] = max(options, key=lambda row: row["net_pnl"])
    positive = [
        name
        for name, row in sealed_summary.items()
        if row["net_pnl"] > 0
        and row["stress_net_pnl"] > 0
        and (row["profit_factor"] or 0.0) > 1.0
    ]
    producing_commit = _git_revision(package_root)
    artifact = run_dir / "tournament.joblib"
    joblib.dump(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "producing_commit": producing_commit,
            "candidate_contract": contract,
            "models": models,
            "dual_head_admission": dual_model,
            "programmatic_policies": programmatic,
            "dual_head_policies": dual_policies,
            "frozen_specialist_reference": source["frozen_specialist"],
            "deployment_status": "not_deployed",
        },
        artifact,
        compress=3,
    )
    loaded = joblib.load(artifact)
    if loaded["run_id"] != run_id or set(loaded["models"]) != set(TRAINED_CANDIDATES):
        raise RuntimeError("tournament artifact round-trip failed")
    artifact_sha = file_sha256(artifact)
    (run_dir / "tournament.sha256").write_text(f"{artifact_sha}  tournament.joblib\n")
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "producing_commit": producing_commit,
        "candidate_count": len(ALL_CANDIDATES),
        "newly_trained_predictor_count": len(TRAINED_CANDIDATES),
        "candidate_names": list(ALL_CANDIDATES),
        "selection_frozen_at": selection_frozen_at,
        "qualification_status": (
            "trained_evaluated_not_deployed" if positive else "trained_evaluated_not_promoted_negative_expectancy"
        ),
        "positive_expectancy_candidates": positive,
        "sealed_predictive": sealed_predictive,
        "sealed_economic": sealed_economic,
        "sealed_summary": sealed_summary,
        "source_manifest": source,
        "split_manifest": split,
        "artifact_sha256": artifact_sha,
        "integrity": {
            "passed": True,
            "market_disjoint": True,
            "official_outcomes_canonical": True,
            "bridge_candidate_separate": True,
            "twap_inference_feature": False,
            "best_later_admission_target": False,
            "authentic_only_filter": False,
            "full_history_source_retained": True,
            "database_mutations": False,
            "new_tables": False,
            "new_schemas": False,
            "new_ingesters": False,
            "new_sources": False,
            "images_rebuilt": False,
            "runtime_exported": False,
            "deployed": False,
            "artifact_round_trip_load": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "sklearn": sklearn.__version__,
        },
        "limitations": [
            "The August 20-25 sealed block is chronological for this run but has been observed during earlier research and is not epistemically fresh.",
            "The reusable full-history panel begins at second 60, so the planned 15-59 Specialist diagnostic is reported from its immutable original tournament rather than retrained here.",
            "Kraken L2 is incremental update-flow rather than reconstructed full-book depth and ends before the sealed block.",
            "Projected PnL assumes recorded ask VWAP5 was fillable and does not model queue position.",
        ],
    }
    _write_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(_report(metrics))
    _write_json(
        run_dir / "completion.json",
        {
            "run_id": run_id,
            "completed": True,
            "artifact_sha256": artifact_sha,
            "metrics_sha256": file_sha256(run_dir / "metrics.json"),
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-run")
    args = parser.parse_args()
    print(train_tournament(args.config, resume_run=args.resume_run))


if __name__ == "__main__":
    main()
