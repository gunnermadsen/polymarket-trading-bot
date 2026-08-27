"""Bounded counterfactual TWAP-state training tournament."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import joblib
import numpy as np
import polars as pl
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from .continuous_edge_training import BOOK_RAW_FEATURES, VWAP_QUANTITIES
from .core_extract import file_sha256
from .counterfactual_twap_state_data import (
    BINANCE_DISAGREEMENT_FEATURES,
    DUAL_TWAP_FEATURES,
    REFPRICE_STATE_FEATURES,
    TWAP30_FEATURES,
    TWAP60_FEATURES,
    CounterfactualDataPaths,
    build_counterfactual_frame,
    extract_counterfactual_sources,
)
from .twap60_challenger_tournament import _decision_columns, _ece, _market_equal_weights
from .twap60_training_data import DataPaths

SCHEMA_VERSION = "btc-counterfactual-twap-state-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-counterfactual-twap-state-model-v1"

FEATURE_TREATMENTS = (
    "refprice_control",
    "twap30",
    "twap60",
    "dual_twap",
    "combined",
    "combined_disagreement",
)

HISTORY_ARMS = (
    "authentic_only",
    "chainlink_history",
    "binance_raw_extension",
    "binance_corrected_extension",
    "uncertainty_weighted_hybrid",
)

CANDIDATE_NAMES = (
    "refprice_state_control",
    "dual_twap_state",
    "refprice_dual_twap_state",
    "refprice_dual_twap_binance_extension",
    "refprice_dual_twap_margin_calibrated",
    "refprice_dual_twap_uncertainty_guard",
)

CHECKPOINT_SCHEMA_VERSION = "btc-counterfactual-twap-state-checkpoint-v1"
_T = TypeVar("_T")


@dataclass(frozen=True)
class ExecutionSettings:
    workers: int
    threads_per_fit: int


@dataclass(frozen=True)
class CheckpointStore:
    root: Path
    identity: str

    def load(self, stage: str) -> Any | None:
        path = self.root / f"{stage}.joblib"
        if not path.is_file():
            return None
        try:
            payload = joblib.load(path)
        except Exception as error:  # noqa: BLE001 - invalid caches are safely recomputed
            print(f"checkpoint: ignoring unreadable {stage}: {error}", flush=True)
            return None
        if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            return None
        if payload.get("identity") != self.identity or payload.get("stage") != stage:
            return None
        print(f"checkpoint: resumed {stage}", flush=True)
        return payload.get("value")

    def save(self, stage: str, value: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{stage}.joblib"
        temporary = self.root / f"{stage}.{os.getpid()}.partial"
        joblib.dump(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "identity": self.identity,
                "stage": stage,
                "value": value,
            },
            temporary,
            compress=3,
        )
        temporary.replace(path)
        print(f"checkpoint: saved {stage}", flush=True)


def execution_settings() -> ExecutionSettings:
    workers = max(1, int(os.environ.get("BTC_TWAP_TRAINING_WORKERS", "2")))
    if workers > 2:
        raise ValueError(
            "counterfactual TWAP training permits at most two concurrent fits"
        )
    default_threads = max(1, (os.cpu_count() or 2) // workers)
    threads = max(
        1,
        int(os.environ.get("BTC_TWAP_TRAINING_THREADS_PER_FIT", str(min(3, default_threads)))),
    )
    thread_budget = min(6, os.cpu_count() or 1)
    if workers * threads > thread_budget:
        raise ValueError(
            f"counterfactual TWAP training CPU budget is {thread_budget} total threads"
        )
    return ExecutionSettings(workers=workers, threads_per_fit=threads)


def _bounded_fit_map(
    function: Callable[[_T], Any], values: Iterable[_T], settings: ExecutionSettings
) -> list[Any]:
    ordered = list(values)
    if settings.workers == 1:
        with threadpool_limits(limits=settings.threads_per_fit):
            return [function(value) for value in ordered]
    with threadpool_limits(limits=settings.threads_per_fit):
        return Parallel(n_jobs=settings.workers, prefer="threads")(
            delayed(function)(value) for value in ordered
        )


def _record_stage_time(timings: dict[str, float], stage: str, started: float) -> None:
    elapsed = time.perf_counter() - started
    timings[stage] = elapsed
    print(f"timing: {stage} completed in {elapsed:.1f}s", flush=True)


@dataclass(frozen=True)
class Fold:
    name: str
    test_start: datetime
    test_end: datetime


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_iter: int
    calibration_c: float


@dataclass(frozen=True)
class TournamentConfig:
    source_path: Path
    package_root: Path
    raw: dict[str, Any]
    profile: str
    model_family: str
    random_seed: int
    start: datetime
    chainlink_start: datetime
    authentic_start: datetime
    current_start: datetime
    candidate_freeze: datetime
    end: datetime
    correction_fit_end: datetime
    correction_validation_end: datetime
    historical_folds: tuple[Fold, ...]
    development_folds: tuple[Fold, ...]
    paths: CounterfactualDataPaths
    runs: Path
    committed_results: Path


@dataclass
class ModelBundle:
    feature_names: tuple[str, ...]
    classifier: HistGradientBoostingClassifier
    lower_margin: HistGradientBoostingRegressor
    median_margin: HistGradientBoostingRegressor
    upper_margin: HistGradientBoostingRegressor
    calibrator: LogisticRegression
    hyperparameters: Hyperparameters
    treatment: str
    history_arm: str
    margin_calibrated: bool


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def load_config(path: Path) -> TournamentConfig:
    source = path.resolve()
    root = source.parent.parent
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    training = raw["training"]
    if not (
        training.get("paper_only") is True
        and training.get("training_only") is True
        and training.get("live_capital_allowed") is False
    ):
        raise ValueError("counterfactual TWAP-state workflow must remain training-only")
    regimes = raw["regimes"]
    paths = raw["paths"]
    base = DataPaths(
        package_root=root,
        cache=root / paths["cache"],
        core_features=root / paths["cache"] / "unused-precomputed-core.parquet",
        core_current_sql=root / paths["core_source_sql"],
        oracle_sql=root / paths["oracle_source_sql"],
        label_sql=root / paths["label_source_sql"],
        refprice_sql=root / paths["refprice_source_sql"],
        candle_sql=root / paths["candle_source_sql"],
        execution_sql=root / paths["execution_source_sql"],
    )
    config = TournamentConfig(
        source_path=source,
        package_root=root,
        raw=raw,
        profile=str(training["profile"]),
        model_family=str(training["model_family"]),
        random_seed=int(training["random_seed"]),
        start=_utc(regimes["binance_start"]),
        chainlink_start=_utc(regimes["chainlink_start"]),
        authentic_start=_utc(regimes["authentic_counterfactual_start"]),
        current_start=_utc(regimes["official_twap60_start"]),
        candidate_freeze=_utc(training["candidate_freeze"]),
        end=_utc(regimes["end"]),
        correction_fit_end=_utc(raw["correction"]["fit_end"]),
        correction_validation_end=_utc(raw["correction"]["validation_end"]),
        historical_folds=tuple(
            Fold(str(row["name"]), _utc(row["test_start"]), _utc(row["test_end"]))
            for row in raw["historical_folds"]
        ),
        development_folds=tuple(
            Fold(str(row["name"]), _utc(row["test_start"]), _utc(row["test_end"]))
            for row in raw["development_folds"]
        ),
        paths=CounterfactualDataPaths(base, root / paths["binance_source_sql"]),
        runs=root / paths["runs"],
        committed_results=root / paths["committed_results"],
    )
    _validate_config(config)
    return config


def _validate_config(config: TournamentConfig) -> None:
    if config.profile != "btc_5m_counterfactual_twap_state":
        raise ValueError("unexpected counterfactual TWAP-state profile")
    if config.model_family != "btc-5m-counterfactual-twap-state":
        raise ValueError("model family identity changed")
    boundaries = (
        config.start, config.chainlink_start, config.authentic_start,
        config.current_start, config.candidate_freeze, config.end,
    )
    if boundaries != tuple(sorted(boundaries)) or len(set(boundaries)) != len(boundaries):
        raise ValueError("data regimes must be strictly chronological")
    entry = config.raw["entry"]
    if (
        int(entry["start_second"]), int(entry["end_second_exclusive"]),
        int(entry["cadence_seconds"]), tuple(tuple(v) for v in entry["cells"]),
    ) != (60, 180, 5, ((60, 90), (90, 120), (120, 150), (150, 180))):
        raise ValueError("entry schedule changed from the training plan")
    if int(config.raw["model"]["hyperparameter_combinations"]) != 36:
        raise ValueError("hyperparameter search must remain bounded at 36 configurations")
    if tuple(config.raw["execution"]["quantities"]) != VWAP_QUANTITIES:
        raise ValueError("capacity curve must remain 5 through 200 shares")
    if len(config.development_folds) < 5:
        raise ValueError("at least five chronological development folds are required")
    required = (
        config.paths.base.core_current_sql, config.paths.base.oracle_sql,
        config.paths.base.label_sql, config.paths.base.refprice_sql,
        config.paths.base.candle_sql, config.paths.base.execution_sql,
        config.paths.binance_sql,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required read-only source queries missing: " + ", ".join(missing))


def predetermined_hyperparameters(config: TournamentConfig) -> tuple[Hyperparameters, ...]:
    rates = (0.02, 0.04, 0.06)
    leaves = (7, 15, 31)
    minimums = (80, 140, 240)
    regularization = (4.0, 8.0, 16.0)
    iterations = (120, 180, 240)
    cs = tuple(float(v) for v in config.raw["model"]["calibration_cs"])
    rows = tuple(
        Hyperparameters(
            rates[index % 3],
            leaves[(index // 3) % 3],
            minimums[(index // 9) % 3],
            regularization[(index * 2 + index // 3) % 3],
            iterations[(index + index // 4) % 3],
            cs[index % len(cs)],
        )
        for index in range(36)
    )
    if len(set(rows)) != 36:
        raise RuntimeError("bounded hyperparameter ledger is not unique")
    return rows


def feature_names(treatment: str) -> tuple[str, ...]:
    if treatment == "refprice_control":
        return REFPRICE_STATE_FEATURES
    if treatment == "twap30":
        return TWAP30_FEATURES
    if treatment == "twap60":
        return TWAP60_FEATURES
    if treatment == "dual_twap":
        return DUAL_TWAP_FEATURES
    if treatment == "combined":
        return tuple(dict.fromkeys((*REFPRICE_STATE_FEATURES, *DUAL_TWAP_FEATURES)))
    if treatment == "combined_disagreement":
        return tuple(
            dict.fromkeys((*REFPRICE_STATE_FEATURES, *DUAL_TWAP_FEATURES, *BINANCE_DISAGREEMENT_FEATURES))
        )
    raise ValueError(f"unknown feature treatment: {treatment}")


def _arm_frame(frame: pl.DataFrame, arm: str) -> pl.DataFrame:
    if arm == "authentic_only":
        return frame.filter(pl.col("label_source").str.starts_with("authentic_"))
    if arm == "chainlink_history":
        return frame.filter(pl.col("label_source") != "binance_synthetic_twap60")
    if arm == "binance_raw_extension":
        return frame.with_columns(
            pl.when(pl.col("label_source") == "binance_synthetic_twap60")
            .then((pl.col("binance_raw_margin_bps") >= 0).cast(pl.Int8))
            .otherwise(pl.col("label_up")).alias("label_up"),
            pl.when(pl.col("label_source") == "binance_synthetic_twap60")
            .then(pl.col("binance_raw_margin_bps")).otherwise(pl.col("target_margin_bps"))
            .alias("target_margin_bps"),
        )
    if arm == "binance_corrected_extension":
        return frame
    if arm == "uncertainty_weighted_hybrid":
        return frame.with_columns(
            (
                pl.col("base_label_weight")
                * (1.0 - pl.col("estimated_synthetic_label_error").fill_null(0.0))
            ).alias("base_label_weight")
        )
    raise ValueError(arm)


def _matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError("training frame missing features: " + ", ".join(missing))
    return frame.select(pl.col(name).cast(pl.Float64) for name in features).to_numpy()


def _weights(frame: pl.DataFrame) -> np.ndarray:
    return _market_equal_weights(frame) * frame["base_label_weight"].to_numpy()


def _split_fit_calibration(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    markets = frame.select("market_id", "window_start").unique().sort("window_start")
    if markets.height < 300:
        raise RuntimeError("insufficient markets for chronological fit/calibration")
    boundary = markets["window_start"][max(int(markets.height * 0.80), 1)]
    return frame.filter(pl.col("window_start") < boundary), frame.filter(pl.col("window_start") >= boundary)


def fit_model(
    frame: pl.DataFrame,
    *,
    treatment: str,
    history_arm: str,
    spec: Hyperparameters,
    seed: int,
    margin_calibrated: bool,
) -> ModelBundle:
    source = _arm_frame(frame, history_arm)
    features = feature_names(treatment)
    fit, calibration = _split_fit_calibration(source)
    fit_matrix = _matrix(fit, features)
    fit_labels = fit["label_up"].to_numpy()
    fit_margins = fit["target_margin_bps"].to_numpy()
    fit_weights = _weights(fit)
    calibration_matrix = _matrix(calibration, features)
    calibration_labels = calibration["label_up"].to_numpy()
    calibration_weights = _weights(calibration)
    classifier = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=spec.learning_rate, max_iter=spec.max_iter,
        max_leaf_nodes=spec.max_leaf_nodes, min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2_regularization, random_state=seed,
        early_stopping=False,
    )
    classifier.fit(fit_matrix, fit_labels, sample_weight=fit_weights)

    def margin_model(quantile: float, offset: int) -> HistGradientBoostingRegressor:
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile, learning_rate=spec.learning_rate,
            max_iter=spec.max_iter, max_leaf_nodes=spec.max_leaf_nodes,
            min_samples_leaf=spec.min_samples_leaf, l2_regularization=spec.l2_regularization,
            random_state=seed + offset, early_stopping=False,
        )
        model.fit(fit_matrix, fit_margins, sample_weight=fit_weights)
        return model

    lower = margin_model(0.05, 1)
    median = margin_model(0.50, 2)
    upper = margin_model(0.95, 3)
    raw = np.clip(classifier.predict_proba(calibration_matrix)[:, 1], 1e-6, 1 - 1e-6)
    median_prediction = median.predict(calibration_matrix)
    calibrator_x = np.column_stack((np.log(raw / (1 - raw)), median_prediction)) if margin_calibrated else np.log(raw / (1 - raw)).reshape(-1, 1)
    calibrator = LogisticRegression(C=spec.calibration_c, max_iter=2000, random_state=seed + 4)
    calibrator.fit(calibrator_x, calibration_labels, sample_weight=calibration_weights)
    return ModelBundle(features, classifier, lower, median, upper, calibrator, spec, treatment, history_arm, margin_calibrated)


def score_model(frame: pl.DataFrame, model: ModelBundle) -> pl.DataFrame:
    if frame.is_empty():
        return frame.with_columns(
            pl.Series("probability_up", [], dtype=pl.Float64),
            pl.Series("predicted_margin_lower_bps", [], dtype=pl.Float64),
            pl.Series("predicted_margin_bps", [], dtype=pl.Float64),
            pl.Series("predicted_margin_upper_bps", [], dtype=pl.Float64),
        )
    x = _matrix(frame, model.feature_names)
    raw = np.clip(model.classifier.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
    lower = model.lower_margin.predict(x)
    median = model.median_margin.predict(x)
    upper = model.upper_margin.predict(x)
    calibrator_x = np.column_stack((np.log(raw / (1 - raw)), median)) if model.margin_calibrated else np.log(raw / (1 - raw)).reshape(-1, 1)
    probability = model.calibrator.predict_proba(calibrator_x)[:, 1]
    return frame.with_columns(
        pl.Series("probability_up", probability),
        pl.Series("predicted_margin_lower_bps", np.minimum(lower, upper)),
        pl.Series("predicted_margin_bps", median),
        pl.Series("predicted_margin_upper_bps", np.maximum(lower, upper)),
    )


def probability_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    y = frame["label_up"].to_numpy()
    p = np.clip(frame["probability_up"].to_numpy(), 1e-9, 1 - 1e-9)
    weights = _market_equal_weights(frame)
    return {
        "markets": frame["market_id"].n_unique(),
        "brier": float(np.average((p - y) ** 2, weights=weights)),
        "log_loss": float(log_loss(y, p, sample_weight=weights, labels=[0, 1])),
        "expected_calibration_error": _ece(y, p, weights),
    }


def _bootstrap_means(values: np.ndarray, *, resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty(resamples)
    maximum_chunk_elements = 4_000_000
    chunk_size = max(1, min(64, maximum_chunk_elements // max(len(values), 1)))
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        output[start:stop] = rng.choice(
            values,
            size=(stop - start, len(values)),
            replace=True,
        ).mean(axis=1)
    return output


def _paired_bootstrap(candidate: pl.DataFrame, control: pl.DataFrame, seed: int, resamples: int) -> dict[str, float]:
    left = candidate.select("market_id", "label_up", "probability_up").unique("market_id")
    right = control.select("market_id", pl.col("probability_up").alias("control_probability")).unique("market_id")
    joined = left.join(right, on="market_id", how="inner")
    y = joined["label_up"].to_numpy()
    delta = (joined["probability_up"].to_numpy() - y) ** 2 - (joined["control_probability"].to_numpy() - y) ** 2
    samples = _bootstrap_means(delta, resamples=resamples, seed=seed)
    return {
        "candidate_minus_control_brier": float(delta.mean()),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
    }


def _candidate_contract(name: str) -> tuple[str, str, bool, bool]:
    return {
        "refprice_state_control": ("refprice_control", "chainlink_history", False, False),
        "dual_twap_state": ("dual_twap", "chainlink_history", False, False),
        "refprice_dual_twap_state": ("combined", "chainlink_history", False, False),
        "refprice_dual_twap_binance_extension": ("combined_disagreement", "binance_corrected_extension", False, False),
        "refprice_dual_twap_margin_calibrated": ("combined", "uncertainty_weighted_hybrid", True, False),
        "refprice_dual_twap_uncertainty_guard": ("combined_disagreement", "uncertainty_weighted_hybrid", True, True),
    }[name]


def _economic_frame(scored: pl.DataFrame, config: TournamentConfig) -> pl.DataFrame:
    eligible = scored.filter(
        (pl.col("window_start") >= config.current_start)
        & pl.all_horizontal(pl.col(name).is_not_null() & pl.col(name).is_finite() for name in BOOK_RAW_FEATURES)
    ).with_columns(pl.col("label_source").alias("label_regime"))
    return _decision_columns(eligible, config)


def _apply_admission(frame: pl.DataFrame, policy: dict[str, float], *, strict_guard: bool) -> pl.DataFrame:
    direction_bound = pl.when(pl.col("predicted_up"))
    if strict_guard:
        margin_ok = direction_bound.then(
            pl.col("predicted_margin_lower_bps") >= policy["minimum_margin_bound_bps"]
        ).otherwise(pl.col("predicted_margin_upper_bps") <= -policy["minimum_margin_bound_bps"])
    else:
        margin_ok = direction_bound.then(pl.col("predicted_margin_lower_bps") > 0).otherwise(
            pl.col("predicted_margin_upper_bps") < 0
        )
    admitted = frame.filter(
        (pl.col("probability_selected") - pl.col("selected_cost_5") >= policy["minimum_probability_edge"])
        & (pl.col("probability_selected") - pl.col("selected_cost_5") - 0.01 >= policy["minimum_stressed_edge"])
        & margin_ok
    )
    return admitted.sort(["window_start", "market_id", "seconds_elapsed"]).group_by(
        "market_id", maintain_order=True
    ).first().sort(["window_start", "market_id"])


def _pnl(frame: pl.DataFrame, quantity: int = 5) -> np.ndarray:
    selected_price = np.where(
        frame["predicted_up"].to_numpy(), frame[f"up_ask_vwap_{quantity}"].to_numpy(),
        frame[f"down_ask_vwap_{quantity}"].to_numpy(),
    )
    correct = frame["direction_correct"].to_numpy().astype(float)
    fee = frame["fee_rate"].to_numpy() * selected_price * (1 - selected_price)
    return quantity * (correct - selected_price - fee - 0.005 - 0.01)


def economic_metrics(ledger: pl.DataFrame, scheduled_markets: int, *, resamples: int, seed: int) -> dict[str, Any]:
    if ledger.is_empty():
        return {"trades": 0, "coverage": 0.0, "stressed_pnl": 0.0, "stressed_expectancy_per_trade": 0.0, "profit_factor": 0.0, "bootstrap_lower": -math.inf, "up_trades": 0, "down_trades": 0, "capacity_curve": {str(q): None for q in VWAP_QUANTITIES}}
    pnl = _pnl(ledger)
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    bootstrap = _bootstrap_means(pnl, resamples=resamples, seed=seed)
    capacity = {}
    for quantity in VWAP_QUANTITIES:
        if f"up_ask_vwap_{quantity}" not in ledger.columns:
            capacity[str(quantity)] = None
        else:
            values = _pnl(ledger, quantity)
            capacity[str(quantity)] = {"pnl": float(values.sum()), "expectancy": float(values.mean())}
    daily = ledger.with_columns(pl.Series("stressed_pnl_row", pnl), pl.col("window_start").dt.date().alias("date")).group_by("date").agg(pl.col("stressed_pnl_row").sum().alias("pnl"))
    positive_total = float(daily.filter(pl.col("pnl") > 0)["pnl"].sum() or 0.0)
    return {
        "trades": ledger.height,
        "coverage": ledger["market_id"].n_unique() / max(scheduled_markets, 1),
        "accuracy": float(ledger["direction_correct"].mean()),
        "stressed_pnl": float(pnl.sum()),
        "stressed_expectancy_per_trade": float(pnl.mean()),
        "profit_factor": float(wins / losses) if losses else math.inf,
        "bootstrap_lower": float(np.quantile(bootstrap, 0.025)),
        "up_trades": int(ledger["predicted_up"].sum()),
        "down_trades": int((~ledger["predicted_up"]).sum()),
        "maximum_day_profit_share": float(daily["pnl"].max() / positive_total) if positive_total > 0 else math.inf,
        "capacity_curve": capacity,
    }


def select_policy(frame: pl.DataFrame, config: TournamentConfig, *, strict_guard: bool, seed: int) -> tuple[dict[str, float], list[dict[str, Any]]]:
    attempts = []
    scheduled = frame["market_id"].n_unique()
    for probability_edge in config.raw["policy"]["minimum_probability_edges"]:
        for stressed_edge in config.raw["policy"]["minimum_stressed_edges"]:
            for margin_bound in config.raw["policy"]["minimum_margin_bounds_bps"]:
                policy = {
                    "minimum_probability_edge": float(probability_edge),
                    "minimum_stressed_edge": float(stressed_edge),
                    "minimum_margin_bound_bps": float(margin_bound),
                }
                ledger = _apply_admission(frame, policy, strict_guard=strict_guard)
                metrics = economic_metrics(ledger, scheduled, resamples=200, seed=seed + len(attempts))
                attempts.append({"policy": policy, "metrics": metrics})
    eligible = [row for row in attempts if row["metrics"]["coverage"] >= 0.10 and row["metrics"]["stressed_expectancy_per_trade"] > 0]
    winner = max(
        eligible or attempts,
        key=lambda row: (
            row["metrics"]["bootstrap_lower"], row["metrics"]["stressed_pnl"], row["metrics"]["coverage"]
        ),
    )
    return winner["policy"], attempts


def _development_fold_frames(
    frame: pl.DataFrame, config: TournamentConfig
) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    return {
        fold.name: (
            frame.filter(pl.col("window_start") < fold.test_start),
            frame.filter(
                pl.col("window_start").is_between(
                    fold.test_start, fold.test_end, closed="left"
                )
            ),
        )
        for fold in config.development_folds
    }


def _search(
    frame: pl.DataFrame,
    config: TournamentConfig,
    settings: ExecutionSettings | None = None,
) -> tuple[Hyperparameters, dict[str, Any]]:
    settings = settings or execution_settings()
    train = frame.filter(pl.col("window_start") < _utc("2026-08-07T00:00:00Z"))
    validation = frame.filter(pl.col("window_start").is_between(_utc("2026-08-07T00:00:00Z"), config.current_start, closed="left"))

    def evaluate(item: tuple[int, Hyperparameters]) -> dict[str, Any]:
        index, spec = item
        model = fit_model(
            train, treatment="combined", history_arm="uncertainty_weighted_hybrid",
            spec=spec, seed=config.random_seed + index, margin_calibrated=True,
        )
        metrics = probability_metrics(score_model(validation, model))
        return {"index": index, "hyperparameters": asdict(spec), **metrics}

    history = _bounded_fit_map(
        evaluate, enumerate(predetermined_hyperparameters(config)), settings
    )
    winner = min(history, key=lambda row: (row["brier"], row["log_loss"], row["expected_calibration_error"], row["index"]))
    return predetermined_hyperparameters(config)[winner["index"]], {"combinations": 36, "selected": winner, "ledger": history}


def _probability_bakeoff(
    frame: pl.DataFrame,
    config: TournamentConfig,
    spec: Hyperparameters,
    *,
    dimension: str,
    fold_frames: dict[str, tuple[pl.DataFrame, pl.DataFrame]] | None = None,
    settings: ExecutionSettings | None = None,
) -> dict[str, Any]:
    settings = settings or execution_settings()
    fold_frames = fold_frames or _development_fold_frames(frame, config)
    values = FEATURE_TREATMENTS if dimension == "features" else HISTORY_ARMS
    ledgers: dict[str, pl.DataFrame] = {}
    results: dict[str, Any] = {}

    def evaluate(
        item: tuple[int, str, int, Fold]
    ) -> tuple[str, int, pl.DataFrame, dict[str, Any]]:
        value_index, value, fold_index, fold = item
        treatment = value if dimension == "features" else "combined"
        arm = "uncertainty_weighted_hybrid" if dimension == "features" else value
        fit, test = fold_frames[fold.name]
        model = fit_model(
            fit,
            treatment=treatment,
            history_arm=arm,
            spec=spec,
            seed=config.random_seed + 1000 + value_index * 100 + fold_index,
            margin_calibrated=False,
        )
        scored = score_model(test, model).with_columns(pl.lit(fold.name).alias("fold"))
        return value, fold_index, scored, {"fold": fold.name, **probability_metrics(scored)}

    tasks = (
        (value_index, value, fold_index, fold)
        for value_index, value in enumerate(values)
        for fold_index, fold in enumerate(config.development_folds)
    )
    evaluated = _bounded_fit_map(evaluate, tasks, settings)
    for value_index, value in enumerate(values):
        rows = sorted(
            (row for row in evaluated if row[0] == value), key=lambda row: row[1]
        )
        pieces = [row[2] for row in rows]
        folds = [row[3] for row in rows]
        ledger = pl.concat(pieces, how="diagonal_relaxed")
        ledgers[value] = ledger
        weights = np.array([row["markets"] for row in folds], dtype=float)
        results[value] = {
            "folds": folds,
            **{key: float(np.average([row[key] for row in folds], weights=weights)) for key in ("brier", "log_loss", "expected_calibration_error")},
        }
    control_name = "refprice_control" if dimension == "features" else "authentic_only"
    control = ledgers[control_name]
    for index, value in enumerate(values):
        results[value]["paired_brier_bootstrap"] = _paired_bootstrap(
            ledgers[value], control, config.random_seed + 3000 + index,
            int(config.raw["gates"]["bootstrap_resamples"]),
        )
    if dimension == "features":
        eligible = [
            value for value in values
            if results[value]["brier"] <= results[control_name]["brier"]
            and results[value]["expected_calibration_error"] <= 0.03
        ]
        selected = min(eligible or values, key=lambda value: (results[value]["brier"], results[value]["log_loss"]))
    else:
        selected = min(values, key=lambda value: (results[value]["brier"], results[value]["log_loss"]))
    return {"dimension": dimension, "results": results, "selection": selected}


def _fidelity(labels: pl.DataFrame, config: TournamentConfig, convention: Any, frame_manifest: dict[str, Any]) -> dict[str, Any]:
    overlap = labels.filter(pl.col("authentic_label_up").is_not_null() & pl.col("proxy_label_up").is_not_null())
    outside = overlap.filter(pl.col("proxy_margin_bps").abs() >= 0.526)
    chainlink = {
        "markets": overlap.height,
        "overall_agreement": float((overlap["authentic_label_up"] == overlap["proxy_label_up"]).mean()) if overlap.height else None,
        "outside_uncertainty_agreement": float((outside["authentic_label_up"] == outside["proxy_label_up"]).mean()) if outside.height else None,
        "p99_price_error_bps": float(convention.calibration_p99_bps),
    }
    chainlink["passed"] = bool(
        (chainlink["overall_agreement"] or 0) >= float(config.raw["gates"]["chainlink_minimum_agreement"])
        and (chainlink["outside_uncertainty_agreement"] or 0) >= float(config.raw["gates"]["chainlink_outside_band_minimum_agreement"])
    )
    binance = dict(frame_manifest["binance_fidelity"])
    weekly_values = [float(row["agreement"]) for row in binance["weekly"] if row["markets"] >= 20]
    binance["passed"] = bool(
        (binance.get("authentic_agreement") or 0) >= float(config.raw["gates"]["binance_minimum_agreement"])
        and weekly_values and min(weekly_values) >= 0.98
    )
    return {"chainlink_reconstruction": chainlink, "binance_extension": binance}


def _frozen_comparators(config: TournamentConfig) -> dict[str, Any]:
    results = {}
    for row in config.raw["comparators"]:
        model = config.package_root / "runtime-models" / row["model_key"] / "model.json"
        digest = file_sha256(model)
        if digest != row["artifact_sha256"]:
            raise RuntimeError(f"frozen comparator digest changed: {row['model_key']}")
        results[row["model_key"]] = {
            "candidate": row["candidate"], "artifact_sha256": digest,
            "status": "frozen_external_comparator_not_retrained",
        }
    for key in ("q5_comparator_metrics", "fair_value_comparator_metrics", "previous_twap_metrics"):
        path = config.package_root / config.raw["paths"][key]
        results[key] = {
            "path": str(path.relative_to(config.package_root)),
            "sha256": file_sha256(path),
            "reported_selection": json.loads(path.read_text()).get("tournament", json.loads(path.read_text()).get("qualification", {})).get("selection", json.loads(path.read_text()).get("qualification", {})),
        }
    return results


def _qualification(
    prospective: dict[str, Any],
    development: dict[str, Any],
    fidelity: dict[str, Any],
    config: TournamentConfig,
) -> dict[str, Any]:
    gates = config.raw["gates"]
    folds = development["folds"]
    profitable = sum(row["economics"]["stressed_pnl"] > 0 for row in folds) / max(len(folds), 1)
    checks = {
        "chainlink_fidelity": fidelity["chainlink_reconstruction"]["passed"],
        "paired_predictive_improvement": development["paired_control"]["upper"] < 0,
        "ece": development["probability"]["expected_calibration_error"] <= float(gates["maximum_ece"]),
        "positive_stressed_pnl": prospective["economics"]["stressed_pnl"] > 0,
        "positive_stressed_expectancy": prospective["economics"]["stressed_expectancy_per_trade"] > 0,
        "profit_factor": prospective["economics"]["profit_factor"] >= float(gates["minimum_profit_factor"]),
        "positive_bootstrap_lower": prospective["economics"]["bootstrap_lower"] > 0,
        "profitable_temporal_folds": profitable >= float(gates["minimum_profitable_fold_ratio"]),
        "market_coverage": prospective["economics"]["coverage"] >= float(gates["minimum_market_coverage"]),
        "both_directions": min(prospective["economics"]["up_trades"], prospective["economics"]["down_trades"]) > 0,
        "no_single_day_majority": prospective["economics"].get("maximum_day_profit_share", math.inf) <= 0.50,
        "prospective_markets": prospective["probability"]["markets"] >= int(gates["minimum_prospective_markets"]),
        "prospective_days": prospective["days"] >= int(gates["minimum_prospective_days"]),
        "prospective_folds": prospective["folds"] >= int(gates["minimum_prospective_folds"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reasons": [name for name, passed in checks.items() if not passed],
        "deployment_status": "not_deployed_training_only",
    }


def _source_commit(config: TournamentConfig) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.package_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkpoint_store(
    config: TournamentConfig,
    source_manifest: dict[str, Any],
    source_commit: str,
    settings: ExecutionSettings,
) -> CheckpointStore:
    identity_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_commit": source_commit,
        "config_sha256": file_sha256(config.source_path),
        "source_manifest": source_manifest,
        "execution": asdict(settings),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return CheckpointStore(config.runs / "checkpoints" / identity, identity)


def _load_or_build_frame(
    config: TournamentConfig, checkpoints: CheckpointStore
) -> tuple[pl.DataFrame, pl.DataFrame, Any, dict[str, Any]]:
    frame_file = config.paths.base.cache / "tournament-frame.parquet"
    labels_file = config.paths.base.cache / "label-audit.parquet"
    cached = checkpoints.load("constructed-frame")
    if (
        isinstance(cached, dict)
        and frame_file.is_file()
        and labels_file.is_file()
        and file_sha256(frame_file) == cached.get("frame_sha256")
        and file_sha256(labels_file) == cached.get("label_audit_sha256")
    ):
        return (
            pl.read_parquet(frame_file),
            pl.read_parquet(labels_file),
            cached["convention"],
            cached["frame_manifest"],
        )

    frame, labels, convention, frame_manifest = build_counterfactual_frame(
        config.paths,
        chainlink_start=config.chainlink_start,
        authentic_start=config.authentic_start,
        official_start=config.current_start,
        correction_fit_end=config.correction_fit_end,
        correction_validation_end=config.correction_validation_end,
        seed=config.random_seed,
    )
    frame.write_parquet(frame_file, compression="zstd", statistics=True)
    labels.write_parquet(labels_file, compression="zstd", statistics=True)
    frame_manifest.update(
        {
            "frame_sha256": file_sha256(frame_file),
            "label_audit_sha256": file_sha256(labels_file),
        }
    )
    checkpoints.save(
        "constructed-frame",
        {
            "frame_sha256": frame_manifest["frame_sha256"],
            "label_audit_sha256": frame_manifest["label_audit_sha256"],
            "convention": convention,
            "frame_manifest": frame_manifest,
        },
    )
    return frame, labels, convention, frame_manifest


def run_tournament(config: TournamentConfig, *, force_extract: bool = False) -> tuple[Path, dict[str, Any]]:
    run_started = time.perf_counter()
    stage_timings: dict[str, float] = {}
    stage_started = time.perf_counter()
    source_manifest = extract_counterfactual_sources(
        config.paths, range_start=config.start, range_end=config.end, force=force_extract
    )
    _record_stage_time(stage_timings, "source-extraction", stage_started)
    source_commit = _source_commit(config)
    settings = execution_settings()
    checkpoints = _checkpoint_store(config, source_manifest, source_commit, settings)
    print(
        f"compute: {settings.workers} concurrent fit, "
        f"{settings.threads_per_fit} threads per fit",
        flush=True,
    )
    print("build: prioritized labels and causal TWAP-30/60 state", flush=True)
    stage_started = time.perf_counter()
    frame, labels, convention, frame_manifest = _load_or_build_frame(config, checkpoints)
    _record_stage_time(stage_timings, "constructed-frame", stage_started)
    fidelity = _fidelity(labels, config, convention, frame_manifest)
    if not fidelity["binance_extension"]["passed"]:
        frame = frame.filter(pl.col("label_source") != "binance_synthetic_twap60")
        frame_manifest["binance_source_tier_removed"] = True

    print("search: 36 predetermined outcome/margin configurations", flush=True)
    stage_started = time.perf_counter()
    search_checkpoint = checkpoints.load("hyperparameter-search")
    if search_checkpoint is None:
        selected_spec, hyperparameter_ledger = _search(frame, config, settings)
        checkpoints.save(
            "hyperparameter-search",
            {"selected_spec": selected_spec, "ledger": hyperparameter_ledger},
        )
    else:
        selected_spec = search_checkpoint["selected_spec"]
        hyperparameter_ledger = search_checkpoint["ledger"]
    _record_stage_time(stage_timings, "hyperparameter-search", stage_started)

    fold_frames = _development_fold_frames(frame, config)
    print("evaluate: fixed-feature and historical-source bakeoffs", flush=True)
    stage_started = time.perf_counter()
    feature_bakeoff = checkpoints.load("feature-treatment-bakeoff")
    if feature_bakeoff is None:
        feature_bakeoff = _probability_bakeoff(
            frame,
            config,
            selected_spec,
            dimension="features",
            fold_frames=fold_frames,
            settings=settings,
        )
        checkpoints.save("feature-treatment-bakeoff", feature_bakeoff)
    _record_stage_time(stage_timings, "feature-treatment-bakeoff", stage_started)
    stage_started = time.perf_counter()
    history_bakeoff = checkpoints.load("historical-source-bakeoff")
    if history_bakeoff is None:
        history_bakeoff = _probability_bakeoff(
            frame,
            config,
            selected_spec,
            dimension="history",
            fold_frames=fold_frames,
            settings=settings,
        )
        checkpoints.save("historical-source-bakeoff", history_bakeoff)
    _record_stage_time(stage_timings, "historical-source-bakeoff", stage_started)

    calibration_window = frame.filter(
        pl.col("window_start").is_between(config.current_start, config.candidate_freeze, closed="left")
    )
    candidate_results: dict[str, Any] = {}
    candidate_ledgers: dict[str, pl.DataFrame] = {}
    final_models: dict[str, ModelBundle] = {}
    control_scored_ledger: pl.DataFrame | None = None
    for candidate_index, name in enumerate(CANDIDATE_NAMES):
        stage_started = time.perf_counter()
        treatment, arm, margin_calibrated, strict_guard = _candidate_contract(name)
        stage = f"candidate-{candidate_index:02d}-{name}"
        candidate_checkpoint = checkpoints.load(stage)
        if candidate_checkpoint is not None:
            candidate_results[name] = candidate_checkpoint["result"]
            candidate_ledgers[name] = candidate_checkpoint["trade_ledger"]
            final_models[name] = candidate_checkpoint["final_model"]
            if name == "refprice_state_control":
                control_scored_ledger = candidate_checkpoint["scored_ledger"]
            _record_stage_time(stage_timings, stage, stage_started)
            continue

        def evaluate_fold(
            item: tuple[int, Fold],
            *,
            candidate_treatment: str = treatment,
            candidate_arm: str = arm,
            candidate_margin_calibrated: bool = margin_calibrated,
            candidate_strict_guard: bool = strict_guard,
            current_candidate_index: int = candidate_index,
        ) -> tuple[int, dict[str, Any], pl.DataFrame, pl.DataFrame]:
            fold_index, fold = item
            fit, test = fold_frames[fold.name]
            model = fit_model(
                fit,
                treatment=candidate_treatment,
                history_arm=candidate_arm,
                spec=selected_spec,
                seed=config.random_seed + 5000 + current_candidate_index * 100 + fold_index,
                margin_calibrated=candidate_margin_calibrated,
            )
            scored = score_model(test, model).with_columns(pl.lit(fold.name).alias("fold"))
            prior_economic = score_model(
                calibration_window.filter(pl.col("window_start") < fold.test_start), model
            )
            economic_prior = _economic_frame(prior_economic, config)
            if economic_prior.is_empty():
                policy = {"minimum_probability_edge": 0.03, "minimum_stressed_edge": 0.01, "minimum_margin_bound_bps": 0.5}
                policy_ledger = []
            else:
                policy, policy_ledger = select_policy(
                    economic_prior,
                    config,
                    strict_guard=candidate_strict_guard,
                    seed=(
                        config.random_seed
                        + 6000
                        + current_candidate_index * 100
                        + fold_index
                    ),
                )
            economics_frame = _economic_frame(scored, config)
            trades = _apply_admission(
                economics_frame, policy, strict_guard=candidate_strict_guard
            ).with_columns(pl.lit(fold.name).alias("fold"))
            fold_economics = economic_metrics(
                trades, test["market_id"].n_unique(), resamples=500,
                seed=(
                    config.random_seed
                    + 7000
                    + current_candidate_index * 100
                    + fold_index
                ),
            )
            fold_result = {
                "fold": fold.name,
                "probability": probability_metrics(scored),
                "economics": fold_economics,
                "policy": policy,
                "policy_attempts": len(policy_ledger),
            }
            return fold_index, fold_result, scored, trades

        evaluated = sorted(
            _bounded_fit_map(
                evaluate_fold, enumerate(config.development_folds), settings
            ),
            key=lambda row: row[0],
        )
        folds = [row[1] for row in evaluated]
        scored_parts = [row[2] for row in evaluated]
        trade_parts = [row[3] for row in evaluated]
        scored_ledger = pl.concat(scored_parts, how="diagonal_relaxed")
        trade_ledger = pl.concat(trade_parts, how="diagonal_relaxed") if trade_parts else pl.DataFrame()
        if name == "refprice_state_control":
            control_scored_ledger = scored_ledger
        if control_scored_ledger is None:
            raise RuntimeError("refprice control must be evaluated before challengers")
        probability = probability_metrics(scored_ledger)
        economics = economic_metrics(
            trade_ledger, scored_ledger["market_id"].n_unique(),
            resamples=int(config.raw["gates"]["bootstrap_resamples"]),
            seed=config.random_seed + 8000 + candidate_index,
        )
        result = {
            "contract": {"treatment": treatment, "history_arm": arm, "margin_calibrated": margin_calibrated, "strict_uncertainty_guard": strict_guard},
            "probability": probability,
            "economics": economics,
            "folds": folds,
            "paired_control": _paired_bootstrap(
                scored_ledger, control_scored_ledger,
                config.random_seed + 9000 + candidate_index,
                int(config.raw["gates"]["bootstrap_resamples"]),
            ),
        }
        with threadpool_limits(limits=settings.threads_per_fit):
            final_model = fit_model(
                frame.filter(pl.col("window_start") < config.candidate_freeze),
                treatment=treatment, history_arm=arm, spec=selected_spec,
                seed=config.random_seed + 10000 + candidate_index,
                margin_calibrated=margin_calibrated,
            )
        candidate_results[name] = result
        candidate_ledgers[name] = trade_ledger
        final_models[name] = final_model
        checkpoints.save(
            stage,
            {
                "result": result,
                "trade_ledger": trade_ledger,
                "scored_ledger": scored_ledger,
                "final_model": final_model,
            },
        )
        _record_stage_time(stage_timings, stage, stage_started)

    stage_started = time.perf_counter()
    provisional = min(
        CANDIDATE_NAMES[1:],
        key=lambda name: (
            candidate_results[name]["probability"]["brier"],
            -candidate_results[name]["economics"]["stressed_expectancy_per_trade"],
        ),
    )
    provisional_model = final_models[provisional]
    development_economic = _economic_frame(score_model(calibration_window, provisional_model), config)
    strict_guard = _candidate_contract(provisional)[3]
    frozen_policy, admission_ledger = select_policy(
        development_economic, config, strict_guard=strict_guard, seed=config.random_seed + 11000
    )
    prospective_frame = frame.filter(
        pl.col("window_start").is_between(config.candidate_freeze, config.end, closed="left")
        & (pl.col("label_source") == "authentic_official_twap60")
    )
    prospective_scored = score_model(prospective_frame, provisional_model)
    prospective_trades = _apply_admission(
        _economic_frame(prospective_scored, config), frozen_policy, strict_guard=strict_guard
    )
    prospective = {
        "start": config.candidate_freeze.isoformat(), "end": config.end.isoformat(),
        "probability": probability_metrics(prospective_scored) if not prospective_scored.is_empty() else {"markets": 0, "brier": None, "log_loss": None, "expected_calibration_error": None},
        "economics": economic_metrics(
            prospective_trades, prospective_frame["market_id"].n_unique(),
            resamples=int(config.raw["gates"]["bootstrap_resamples"]), seed=config.random_seed + 12000,
        ),
        "days": prospective_frame["window_start"].dt.date().n_unique() if not prospective_frame.is_empty() else 0,
        "folds": 1 if not prospective_frame.is_empty() else 0,
        "policy": frozen_policy,
        "post_freeze_tuning": False,
    }
    qualification = _qualification(
        prospective, candidate_results[provisional], fidelity, config
    )
    status = "deployable_challenger_qualified" if qualification["passed"] else "no_deployable_challenger_qualified"
    _record_stage_time(stage_timings, "final-selection-and-qualification", stage_started)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    temporary = config.runs / f"{run_id}.partial"
    final = config.committed_results / run_id
    if temporary.exists() or final.exists():
        raise FileExistsError(run_id)
    temporary.mkdir(parents=True)
    ledgers = temporary / "ledgers"
    ledgers.mkdir()
    for name, ledger in candidate_ledgers.items():
        ledger.write_parquet(ledgers / f"{name}.parquet", compression="zstd", statistics=True)
    prospective_trades.write_parquet(ledgers / "prospective-trades.parquet", compression="zstd", statistics=True)
    artifact_path = temporary / "tournament.joblib"
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_family": config.model_family,
        "data_watermark": config.candidate_freeze,
        "provisional_candidate": provisional,
        "final_models": final_models,
        "admission_policy": frozen_policy,
        "qualification_status": status,
        "deployment_status": "not_deployed_training_only",
    }
    joblib.dump(artifact, artifact_path, compress=3)
    artifact_sha = file_sha256(artifact_path)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "model_family": config.model_family,
        "paper_only": True, "training_only": True, "live_capital_allowed": False,
        "database_mutations": False, "new_tables": False, "new_ingesters": False,
        "new_data_sources": False, "trading_processes_changed": False,
        "source_commit": source_commit,
        "configuration": {"path": str(config.source_path.relative_to(config.package_root)), "sha256": file_sha256(config.source_path), "candidate_freeze": config.candidate_freeze.isoformat(), "data_watermark": config.end.isoformat()},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
            "parallel_fits": settings.workers,
            "threads_per_fit": settings.threads_per_fit,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_identity": checkpoints.identity,
            "stage_seconds": stage_timings,
            "elapsed_seconds_before_reporting": time.perf_counter() - run_started,
        },
        "source_manifest": source_manifest,
        "frame_manifest": frame_manifest,
        "fidelity": fidelity,
        "hyperparameter_ledger": hyperparameter_ledger,
        "feature_treatment_bakeoff": feature_bakeoff,
        "historical_source_bakeoff": history_bakeoff,
        "candidate_results": candidate_results,
        "frozen_comparators": _frozen_comparators(config),
        "selection": {"provisional_candidate": provisional, "status": status},
        "prospective_qualification": prospective,
        "qualification": qualification,
        "admission_search": {"attempts": admission_ledger},
        "model_artifact": {"path": "tournament.joblib", "sha256": artifact_sha},
        "limitations": [
            "Only one complete post-freeze UTC day is currently available; the 10-day/2,500-market prospective gate cannot pass.",
            "Frozen comparators remain external immutable artifacts and are not incorporated into this model family.",
            "Projected economics use recorded executable orderbook evidence and do not model queue position.",
            "No model was deployed and no trading process was changed.",
        ],
    }
    _write_json(temporary / "metrics.json", metrics)
    _write_json(temporary / "source-manifest.json", source_manifest)
    _write_json(temporary / "data-label-coverage.json", {"frame_manifest": frame_manifest, "fidelity": fidelity})
    _write_json(temporary / "feature-treatment-bakeoff.json", feature_bakeoff)
    _write_json(temporary / "historical-source-bakeoff.json", history_bakeoff)
    _write_json(temporary / "hyperparameter-ledger.json", hyperparameter_ledger)
    _write_json(temporary / "capacity-curves.json", {name: row["economics"]["capacity_curve"] for name, row in candidate_results.items()})
    _write_json(temporary / "frozen-comparators.json", metrics["frozen_comparators"])
    _write_json(temporary / "qualification.json", {"selection": metrics["selection"], "qualification": qualification, "prospective": prospective})
    provenance = {
        "schema_version": "btc-model-provenance-v1",
        "model_family": config.model_family,
        "model_artifact_sha256": artifact_sha,
        "artifact_path": "tournament.joblib",
        "producing_source_commit": source_commit,
        "training_run_id": run_id,
        "source_identity": frame_manifest["frame_sha256"],
        "input_manifest_sha256": hashlib.sha256(json.dumps(source_manifest, sort_keys=True, default=str).encode()).hexdigest(),
        "data_watermark": config.end.isoformat(),
        "candidate_freeze": config.candidate_freeze.isoformat(),
        "qualification_status": status,
        "deployment_status": "not_deployed_training_only",
    }
    _write_json(temporary / "model-provenance.json", provenance)
    (temporary / "tournament.sha256").write_text(artifact_sha + "\n")
    (temporary / "report.md").write_text(_render_report(metrics))
    config.committed_results.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    return final, metrics


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    raise TypeError(type(value).__name__)


def _render_report(metrics: dict[str, Any]) -> str:
    selection = metrics["selection"]
    qualification = metrics["qualification"]
    fidelity = metrics["fidelity"]
    rows = [
        "# BTC 5m Counterfactual TWAP-State Tournament",
        "",
        f"- Run: `{metrics['run_id']}`",
        f"- Model family: `{metrics['model_family']}`",
        f"- Provisional candidate: `{selection['provisional_candidate']}`",
        f"- Qualification: **{selection['status']}**",
        "- Deployment: **not deployed; training-only and paper-only**",
        "",
        "## Source fidelity",
        "",
        f"- Chainlink reconstruction passed: `{fidelity['chainlink_reconstruction']['passed']}`",
        f"- Binance extension passed: `{fidelity['binance_extension']['passed']}`",
        "",
        "## Required conclusion",
        "",
    ]
    if selection["status"] == "no_deployable_challenger_qualified":
        rows.append("No new deployable challenger qualified. The immutable candidate remains paper-only; prospective evidence is below the frozen minimum.")
    else:
        rows.append("Explicit TWAP state provided predictive and economic value beyond refprice under the frozen qualification gates.")
    rows.extend(["", "## Failed qualification gates", ""])
    rows.extend(f"- `{reason}`" for reason in qualification["reasons"])
    rows.extend(["", "## Scope controls", "", "No database mutation, migration, table, ingester, data source, runtime export, deployment, or trading-process change was performed.", ""])
    return "\n".join(rows)
