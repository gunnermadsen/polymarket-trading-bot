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
from datetime import UTC, date, datetime
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
    CAUSAL_BASIS_FEATURES,
    REFPRICE_STATE_FEATURES,
    RELATIVE_TWAP_FEATURES,
    SUPERVISION_ONLY_FIELDS,
    SUPERVISION_REGISTRY,
    CounterfactualDataPaths,
    build_counterfactual_frame,
    extract_counterfactual_sources,
    inference_feature_registry,
    validate_inference_features,
)
from .twap60_challenger_tournament import _decision_columns, _ece, _market_equal_weights
from .twap60_training_data import DataPaths

SCHEMA_VERSION = "btc-causal-twap-attribution-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-causal-twap-attribution-model-v1"

ATTRIBUTION_TREATMENTS = (
    "refprice_only",
    "refprice_non_twap_basis",
    "refprice_relative_twap",
    "refprice_absolute_twap",
)

ABSOLUTE_TWAP_FEATURES = (
    "opening_refprice",
    "opening_twap30",
    "current_twap30",
    "opening_twap60",
    "current_twap60",
    "opening_binance_twap30",
    "opening_binance_twap60",
)

CANDIDATE_NAMES = ATTRIBUTION_TREATMENTS

CHECKPOINT_SCHEMA_VERSION = "btc-causal-twap-attribution-checkpoint-v1"
CAUSAL_FEATURE_REGISTRY_SCHEMA_VERSION = "btc-causal-feature-registry-v1"
_T = TypeVar("_T")


def causal_feature_registry_payload() -> dict[str, Any]:
    registry = inference_feature_registry()
    candidate_features = {
        name: set(feature_names(_candidate_contract(name)[0])) for name in CANDIDATE_NAMES
    }
    attribution_features = {name: set(feature_names(name)) for name in ATTRIBUTION_TREATMENTS}
    for name, metadata in registry.items():
        metadata["candidate_membership"] = [
            candidate for candidate, features in candidate_features.items() if name in features
        ]
        metadata["treatment_membership"] = [
            treatment for treatment, features in attribution_features.items() if name in features
        ]
        metadata["diagnostic_membership"] = [
            treatment for treatment, features in attribution_features.items() if name in features
        ]
    return {
        "schema_version": CAUSAL_FEATURE_REGISTRY_SCHEMA_VERSION,
        "features": registry,
        "supervision_registry": SUPERVISION_REGISTRY,
        "supervision_only_fields": sorted(SUPERVISION_ONLY_FIELDS),
    }


def causal_feature_registry_sha256() -> str:
    return hashlib.sha256(
        json.dumps(causal_feature_registry_payload(), sort_keys=True).encode()
    ).hexdigest()


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
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        ):
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
        raise ValueError("counterfactual TWAP training permits at most two concurrent fits")
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
    development_folds: tuple[Fold, ...]
    paths: CounterfactualDataPaths
    runs: Path
    committed_results: Path


@dataclass
class ModelBundle:
    feature_names: tuple[str, ...]
    all_missing_feature_indices: tuple[int, ...]
    classifier: HistGradientBoostingClassifier
    lower_margin: HistGradientBoostingRegressor
    median_margin: HistGradientBoostingRegressor
    upper_margin: HistGradientBoostingRegressor
    calibrator: LogisticRegression
    hyperparameters: Hyperparameters
    treatment: str
    history_arm: str
    margin_calibrated: bool


@dataclass
class ProbabilityBundle:
    feature_names: tuple[str, ...]
    all_missing_feature_indices: tuple[int, ...]
    classifier: HistGradientBoostingClassifier
    calibrator: LogisticRegression
    hyperparameters: Hyperparameters
    treatment: str
    history_arm: str


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
    if not (
        config.start
        < config.chainlink_start
        < config.authentic_start
        < config.current_start
        < config.candidate_freeze
        <= config.end
    ):
        raise ValueError("data regimes must be strictly chronological")
    entry = config.raw["entry"]
    if (
        int(entry["start_second"]),
        int(entry["end_second_exclusive"]),
        int(entry["cadence_seconds"]),
        tuple(tuple(v) for v in entry["cells"]),
    ) != (60, 180, 5, ((60, 90), (90, 120), (120, 150), (150, 180))):
        raise ValueError("entry schedule changed from the training plan")
    if int(config.raw["model"]["hyperparameter_combinations"]) != 36:
        raise ValueError("hyperparameter search must remain bounded at 36 configurations")
    if tuple(config.raw["execution"]["quantities"]) != VWAP_QUANTITIES:
        raise ValueError("capacity curve must remain 5 through 200 shares")
    if _utc(config.raw["training"]["data_watermark"]) != config.end:
        raise ValueError("training data watermark must equal the evaluation range end")
    if len(config.development_folds) < 5:
        raise ValueError("at least five chronological development folds are required")
    if any(fold.test_end > config.candidate_freeze for fold in config.development_folds):
        raise ValueError("development folds must end at or before the candidate freeze")
    required = (
        config.paths.base.core_current_sql,
        config.paths.base.oracle_sql,
        config.paths.base.label_sql,
        config.paths.base.refprice_sql,
        config.paths.base.candle_sql,
        config.paths.base.execution_sql,
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
    if treatment == "refprice_only":
        return REFPRICE_STATE_FEATURES
    if treatment == "refprice_non_twap_basis":
        return tuple(dict.fromkeys((*REFPRICE_STATE_FEATURES, *CAUSAL_BASIS_FEATURES)))
    if treatment == "refprice_relative_twap":
        return tuple(
            dict.fromkeys(
                (
                    *REFPRICE_STATE_FEATURES,
                    *CAUSAL_BASIS_FEATURES,
                    *RELATIVE_TWAP_FEATURES,
                    *BINANCE_DISAGREEMENT_FEATURES,
                )
            )
        )
    if treatment == "refprice_absolute_twap":
        return tuple(
            dict.fromkeys(
                (
                    *feature_names("refprice_relative_twap"),
                    *ABSOLUTE_TWAP_FEATURES,
                )
            )
        )
    raise ValueError(f"unknown feature treatment: {treatment}")


def _arm_frame(frame: pl.DataFrame, arm: str) -> pl.DataFrame:
    if arm == "chainlink_history":
        return frame.filter(pl.col("label_source") != "binance_synthetic_twap60")
    raise ValueError(arm)


def _matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    validate_inference_features(features)
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError("training frame missing features: " + ", ".join(missing))
    return frame.select(pl.col(name).cast(pl.Float64) for name in features).to_numpy()


def _apply_all_missing_feature_mask(matrix: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    if not indices:
        return matrix
    normalized = matrix.copy()
    normalized[:, indices] = 0.0
    return normalized


def _neutralize_all_missing_fit_columns(
    matrix: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    indices = tuple(int(index) for index in np.flatnonzero(np.isnan(matrix).all(axis=0)))
    return _apply_all_missing_feature_mask(matrix, indices), indices


def _weights(frame: pl.DataFrame) -> np.ndarray:
    return _market_equal_weights(frame) * frame["base_label_weight"].to_numpy()


def _split_fit_calibration(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    markets = frame.select("market_id", "window_start").unique().sort("window_start")
    if markets.height < 300:
        raise RuntimeError("insufficient markets for chronological fit/calibration")
    boundary = markets["window_start"][max(int(markets.height * 0.80), 1)]
    return frame.filter(pl.col("window_start") < boundary), frame.filter(
        pl.col("window_start") >= boundary
    )


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
    fit_matrix, all_missing_feature_indices = _neutralize_all_missing_fit_columns(fit_matrix)
    fit_labels = fit["label_up"].to_numpy()
    fit_margins = fit["target_margin_bps"].to_numpy()
    fit_weights = _weights(fit)
    calibration_matrix = _apply_all_missing_feature_mask(
        _matrix(calibration, features), all_missing_feature_indices
    )
    calibration_labels = calibration["label_up"].to_numpy()
    calibration_weights = _weights(calibration)
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=spec.learning_rate,
        max_iter=spec.max_iter,
        max_leaf_nodes=spec.max_leaf_nodes,
        min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2_regularization,
        random_state=seed,
        early_stopping=False,
    )
    classifier.fit(fit_matrix, fit_labels, sample_weight=fit_weights)

    def margin_model(quantile: float, offset: int) -> HistGradientBoostingRegressor:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            learning_rate=spec.learning_rate,
            max_iter=spec.max_iter,
            max_leaf_nodes=spec.max_leaf_nodes,
            min_samples_leaf=spec.min_samples_leaf,
            l2_regularization=spec.l2_regularization,
            random_state=seed + offset,
            early_stopping=False,
        )
        model.fit(fit_matrix, fit_margins, sample_weight=fit_weights)
        return model

    lower = margin_model(0.05, 1)
    median = margin_model(0.50, 2)
    upper = margin_model(0.95, 3)
    raw = np.clip(classifier.predict_proba(calibration_matrix)[:, 1], 1e-6, 1 - 1e-6)
    median_prediction = median.predict(calibration_matrix)
    calibrator_x = (
        np.column_stack((np.log(raw / (1 - raw)), median_prediction))
        if margin_calibrated
        else np.log(raw / (1 - raw)).reshape(-1, 1)
    )
    calibrator = LogisticRegression(C=spec.calibration_c, max_iter=2000, random_state=seed + 4)
    calibrator.fit(calibrator_x, calibration_labels, sample_weight=calibration_weights)
    return ModelBundle(
        features,
        all_missing_feature_indices,
        classifier,
        lower,
        median,
        upper,
        calibrator,
        spec,
        treatment,
        history_arm,
        margin_calibrated,
    )


def fit_probability_model(
    frame: pl.DataFrame,
    *,
    treatment: str,
    history_arm: str,
    spec: Hyperparameters,
    seed: int,
) -> ProbabilityBundle:
    source = _arm_frame(frame, history_arm)
    features = feature_names(treatment)
    fit, calibration = _split_fit_calibration(source)
    fit_matrix, all_missing_feature_indices = _neutralize_all_missing_fit_columns(
        _matrix(fit, features)
    )
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=spec.learning_rate,
        max_iter=spec.max_iter,
        max_leaf_nodes=spec.max_leaf_nodes,
        min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2_regularization,
        random_state=seed,
        early_stopping=False,
    )
    classifier.fit(
        fit_matrix,
        fit["label_up"].to_numpy(),
        sample_weight=_weights(fit),
    )
    calibration_matrix = _apply_all_missing_feature_mask(
        _matrix(calibration, features), all_missing_feature_indices
    )
    raw = np.clip(classifier.predict_proba(calibration_matrix)[:, 1], 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(C=spec.calibration_c, max_iter=2000, random_state=seed + 1)
    calibrator.fit(
        np.log(raw / (1 - raw)).reshape(-1, 1),
        calibration["label_up"].to_numpy(),
        sample_weight=_weights(calibration),
    )
    return ProbabilityBundle(
        features,
        all_missing_feature_indices,
        classifier,
        calibrator,
        spec,
        treatment,
        history_arm,
    )


def score_probability_model(frame: pl.DataFrame, model: ProbabilityBundle) -> pl.DataFrame:
    if frame.is_empty():
        return frame.with_columns(pl.Series("probability_up", [], dtype=pl.Float64))
    matrix = _apply_all_missing_feature_mask(
        _matrix(frame, model.feature_names), model.all_missing_feature_indices
    )
    raw = np.clip(model.classifier.predict_proba(matrix)[:, 1], 1e-6, 1 - 1e-6)
    probability = model.calibrator.predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))[:, 1]
    return frame.with_columns(pl.Series("probability_up", probability))


def score_model(frame: pl.DataFrame, model: ModelBundle) -> pl.DataFrame:
    if frame.is_empty():
        return frame.with_columns(
            pl.Series("probability_up", [], dtype=pl.Float64),
            pl.Series("predicted_margin_lower_bps", [], dtype=pl.Float64),
            pl.Series("predicted_margin_bps", [], dtype=pl.Float64),
            pl.Series("predicted_margin_upper_bps", [], dtype=pl.Float64),
        )
    x = _apply_all_missing_feature_mask(
        _matrix(frame, model.feature_names), model.all_missing_feature_indices
    )
    raw = np.clip(model.classifier.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
    lower = model.lower_margin.predict(x)
    median = model.median_margin.predict(x)
    upper = model.upper_margin.predict(x)
    calibrator_x = (
        np.column_stack((np.log(raw / (1 - raw)), median))
        if model.margin_calibrated
        else np.log(raw / (1 - raw)).reshape(-1, 1)
    )
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


def _paired_loss_rows(candidate: pl.DataFrame, control: pl.DataFrame) -> pl.DataFrame:
    keys = ["market_id", "observed_at", "seconds_elapsed"]
    left = candidate.select(
        *keys,
        "window_start",
        "label_up",
        "target_margin_bps",
        "probability_up",
    )
    right = control.select(*keys, pl.col("probability_up").alias("control_probability"))
    joined = left.join(right, on=keys, how="inner", validate="1:1")
    if joined.height != left.height or joined.height != right.height:
        raise RuntimeError("paired attribution ledgers do not have identical observation keys")
    return joined.with_columns(
        (
            (pl.col("probability_up") - pl.col("label_up")) ** 2
            - (pl.col("control_probability") - pl.col("label_up")) ** 2
        ).alias("brier_delta")
    )


def _paired_bootstrap(
    candidate: pl.DataFrame, control: pl.DataFrame, seed: int, resamples: int
) -> dict[str, float]:
    market_delta = (
        _paired_loss_rows(candidate, control)
        .group_by("market_id")
        .agg(pl.col("brier_delta").mean())
    )
    delta = market_delta["brier_delta"].to_numpy()
    samples = _bootstrap_means(delta, resamples=resamples, seed=seed)
    return {
        "markets": market_delta.height,
        "candidate_minus_control_brier": float(delta.mean()),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
    }


def _paired_stability(
    candidate: pl.DataFrame,
    control: pl.DataFrame,
    config: TournamentConfig,
) -> dict[str, Any]:
    rows = _paired_loss_rows(candidate, control).with_columns(
        pl.col("window_start").dt.date().cast(pl.String).alias("date"),
        pl.when(pl.col("label_up") == 1)
        .then(pl.lit("up"))
        .otherwise(pl.lit("down"))
        .alias("direction"),
        pl.when(pl.col("seconds_elapsed") < 90)
        .then(pl.lit("60_89"))
        .when(pl.col("seconds_elapsed") < 120)
        .then(pl.lit("90_119"))
        .when(pl.col("seconds_elapsed") < 150)
        .then(pl.lit("120_149"))
        .otherwise(pl.lit("150_179"))
        .alias("entry_band"),
        pl.when(pl.col("target_margin_bps").abs() < 0.526)
        .then(pl.lit("under_0_526"))
        .when(pl.col("target_margin_bps").abs() < 1.578)
        .then(pl.lit("0_526_to_1_578"))
        .when(pl.col("target_margin_bps").abs() < 5.0)
        .then(pl.lit("1_578_to_5"))
        .otherwise(pl.lit("5_plus"))
        .alias("margin_band"),
    )
    maximum_degradation = float(config.raw["gates"]["maximum_slice_brier_degradation"])
    minimum_ratio = float(config.raw["gates"]["minimum_non_degrading_slice_ratio"])
    dimensions: dict[str, Any] = {}
    for dimension in ("date", "direction", "entry_band", "margin_band"):
        slices = (
            rows.group_by(dimension)
            .agg(
                pl.col("market_id").n_unique().alias("markets"),
                pl.col("brier_delta").mean().alias("brier_delta"),
            )
            .sort(dimension)
        )
        eligible = slices.filter(pl.col("markets") >= 50)
        non_degrading_ratio = (
            float((eligible["brier_delta"] <= 0).mean()) if not eligible.is_empty() else 0.0
        )
        worst = float(eligible["brier_delta"].max()) if not eligible.is_empty() else math.inf
        dimensions[dimension] = {
            "slices": slices.to_dicts(),
            "eligible_slices": eligible.height,
            "non_degrading_slice_ratio": non_degrading_ratio,
            "worst_brier_delta": worst,
            "passed": (
                eligible.height >= 2
                and non_degrading_ratio >= minimum_ratio
                and worst <= maximum_degradation
            ),
        }
    return {
        "maximum_slice_brier_degradation": maximum_degradation,
        "minimum_non_degrading_slice_ratio": minimum_ratio,
        "dimensions": dimensions,
        "passed": all(row["passed"] for row in dimensions.values()),
    }


def _candidate_contract(name: str) -> tuple[str, str, bool, bool]:
    if name not in ATTRIBUTION_TREATMENTS:
        raise ValueError(name)
    return name, "chainlink_history", False, False


def _economic_frame(
    scored: pl.DataFrame,
    config: TournamentConfig,
    *,
    minimum_start: datetime | None = None,
) -> pl.DataFrame:
    minimum_start = minimum_start or config.current_start
    eligible = scored.filter(
        (pl.col("window_start") >= minimum_start)
        & pl.all_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite() for name in BOOK_RAW_FEATURES
        )
    ).with_columns(pl.col("label_source").alias("label_regime"))
    return _decision_columns(eligible, config)


def _apply_admission(
    frame: pl.DataFrame, policy: dict[str, float], *, strict_guard: bool
) -> pl.DataFrame:
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
        (
            pl.col("probability_selected") - pl.col("selected_cost_5")
            >= policy["minimum_probability_edge"]
        )
        & (
            pl.col("probability_selected") - pl.col("selected_cost_5") - 0.01
            >= policy["minimum_stressed_edge"]
        )
        & margin_ok
    )
    return (
        admitted.sort(["window_start", "market_id", "seconds_elapsed"])
        .group_by("market_id", maintain_order=True)
        .first()
        .sort(["window_start", "market_id"])
    )


def _decision_ledger(
    scored: pl.DataFrame,
    eligible: pl.DataFrame,
    trades: pl.DataFrame,
    *,
    fold: str,
) -> pl.DataFrame:
    markets = scored.select("market_id", "window_start", "label_source", "label_up").unique(
        "market_id", maintain_order=True
    )
    economic = (
        eligible.select("market_id")
        .unique()
        .with_columns(pl.lit(True).alias("economic_evidence_eligible"))
    )
    admitted = trades.select(
        "market_id",
        "observed_at",
        "seconds_elapsed",
        "predicted_up",
        "probability_up",
        "predicted_margin_bps",
    ).unique("market_id", maintain_order=True)
    return (
        markets.join(economic, on="market_id", how="left", validate="1:1")
        .join(admitted, on="market_id", how="left", validate="1:1")
        .with_columns(
            pl.col("economic_evidence_eligible").fill_null(False),
            pl.col("observed_at").is_not_null().alias("admitted"),
            pl.when(pl.col("observed_at").is_not_null())
            .then(pl.lit("trade"))
            .when(pl.col("economic_evidence_eligible").fill_null(False))
            .then(pl.lit("policy_abstention"))
            .otherwise(pl.lit("no_authentic_executable_book"))
            .alias("decision"),
            pl.lit(fold).alias("fold"),
        )
    )


def _pnl(frame: pl.DataFrame, quantity: int = 5) -> np.ndarray:
    selected_price = np.where(
        frame["predicted_up"].to_numpy(),
        frame[f"up_ask_vwap_{quantity}"].to_numpy(),
        frame[f"down_ask_vwap_{quantity}"].to_numpy(),
    )
    correct = frame["direction_correct"].to_numpy().astype(float)
    fee = frame["fee_rate"].to_numpy() * selected_price * (1 - selected_price)
    return quantity * (correct - selected_price - fee - 0.005 - 0.01)


def economic_metrics(
    ledger: pl.DataFrame, scheduled_markets: int, *, resamples: int, seed: int
) -> dict[str, Any]:
    if ledger.is_empty():
        return {
            "trades": 0,
            "coverage": 0.0,
            "accuracy": None,
            "stressed_pnl": 0.0,
            "stressed_expectancy_per_trade": 0.0,
            "profit_factor": 0.0,
            "bootstrap_lower": -math.inf,
            "up_trades": 0,
            "down_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "gross_winning_pnl": 0.0,
            "losing_trade_pnl": 0.0,
            "average_winning_pnl": None,
            "average_losing_pnl": None,
            "winning_trades_to_recover_average_loss": None,
            "average_entry_second": None,
            "average_entry_price": None,
            "entry_time_distribution": [],
            "daily_pnl": [],
            "maximum_day_profit_share": math.inf,
            "maximum_drawdown": 0.0,
            "cvar_5pct": 0.0,
            "capacity_curve": {str(q): None for q in VWAP_QUANTITIES},
        }
    pnl = _pnl(ledger)
    selected_price = np.where(
        ledger["predicted_up"].to_numpy(),
        ledger["up_ask_vwap_5"].to_numpy(),
        ledger["down_ask_vwap_5"].to_numpy(),
    )
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    winning_count = int((pnl > 0).sum())
    losing_count = int((pnl < 0).sum())
    average_win = float(wins / winning_count) if winning_count else None
    average_loss = float(losses / losing_count) if losing_count else None
    bootstrap = _bootstrap_means(pnl, resamples=resamples, seed=seed)
    capacity = {}
    for quantity in VWAP_QUANTITIES:
        if f"up_ask_vwap_{quantity}" not in ledger.columns:
            capacity[str(quantity)] = None
        else:
            values = _pnl(ledger, quantity)
            capacity[str(quantity)] = {
                "pnl": float(values.sum()),
                "expectancy": float(values.mean()),
            }
    daily = (
        ledger.with_columns(
            pl.Series("stressed_pnl_row", pnl), pl.col("window_start").dt.date().alias("date")
        )
        .group_by("date")
        .agg(pl.col("stressed_pnl_row").sum().alias("pnl"))
    )
    entry_distribution = (
        ledger.group_by("seconds_elapsed").agg(pl.len().alias("trades")).sort("seconds_elapsed")
    )
    positive_total = float(daily.filter(pl.col("pnl") > 0)["pnl"].sum() or 0.0)
    cumulative = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
    maximum_drawdown = float(np.max(running_peak[1:] - cumulative, initial=0.0))
    tail_count = max(1, math.ceil(len(pnl) * 0.05))
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
        "winning_trades": winning_count,
        "losing_trades": losing_count,
        "gross_winning_pnl": float(wins),
        "losing_trade_pnl": float(pnl[pnl < 0].sum()),
        "average_winning_pnl": average_win,
        "average_losing_pnl": -average_loss if average_loss is not None else None,
        "winning_trades_to_recover_average_loss": (
            average_loss / average_win
            if average_loss is not None and average_win is not None and average_win > 0
            else None
        ),
        "average_entry_second": float(ledger["seconds_elapsed"].mean()),
        "average_entry_price": float(np.mean(selected_price)),
        "entry_time_distribution": entry_distribution.to_dicts(),
        "daily_pnl": daily.sort("date").to_dicts(),
        "maximum_day_profit_share": float(daily["pnl"].max() / positive_total)
        if positive_total > 0
        else math.inf,
        "maximum_drawdown": maximum_drawdown,
        "cvar_5pct": float(np.sort(pnl)[:tail_count].mean()),
        "capacity_curve": capacity,
    }


def select_policy(
    frame: pl.DataFrame, config: TournamentConfig, *, strict_guard: bool, seed: int
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if frame.is_empty():
        raise RuntimeError("authentic policy-calibration execution evidence is empty")
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
                metrics = economic_metrics(
                    ledger, scheduled, resamples=200, seed=seed + len(attempts)
                )
                attempts.append({"policy": policy, "metrics": metrics})
    eligible = [
        row
        for row in attempts
        if row["metrics"]["coverage"] >= 0.10
        and row["metrics"]["stressed_expectancy_per_trade"] > 0
    ]
    winner = max(
        eligible or attempts,
        key=lambda row: (
            row["metrics"]["bootstrap_lower"],
            row["metrics"]["stressed_pnl"],
            row["metrics"]["coverage"],
        ),
    )
    return winner["policy"], attempts


def _development_fold_frames(
    frame: pl.DataFrame, config: TournamentConfig
) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    folds = {
        fold.name: (
            frame.filter(pl.col("window_start") < fold.test_start),
            frame.filter(
                pl.col("window_start").is_between(fold.test_start, fold.test_end, closed="left")
            ),
        )
        for fold in config.development_folds
    }
    for name, (fit, test) in folds.items():
        overlap = set(fit["market_id"].unique()) & set(test["market_id"].unique())
        if overlap:
            raise RuntimeError(f"training/test market overlap in development fold {name}")
        if (
            not fit.is_empty()
            and not test.is_empty()
            and fit["window_start"].max() >= test["window_start"].min()
        ):
            raise RuntimeError(f"non-chronological development fold {name}")
    return folds


def _complete_utc_day_audit(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {"complete_dates": [], "days": []}
    days = (
        frame.select("market_id", "window_start")
        .unique("market_id")
        .with_columns(
            pl.col("window_start").dt.date().alias("date"),
            (
                pl.col("window_start").dt.hour().cast(pl.Int32) * 60
                + pl.col("window_start").dt.minute().cast(pl.Int32)
            ).alias("minute_of_day"),
        )
        .group_by("date")
        .agg(
            pl.len().alias("markets"),
            pl.col("minute_of_day").min().alias("first_minute"),
            pl.col("minute_of_day").max().alias("last_minute"),
        )
        .sort("date")
        .with_columns(
            (
                (pl.col("markets") >= 250)
                & (pl.col("first_minute") <= 5)
                & (pl.col("last_minute") >= 23 * 60 + 55)
            ).alias("complete")
        )
    )
    return {
        "complete_dates": days.filter(pl.col("complete"))["date"].to_list(),
        "days": days.to_dicts(),
    }


def _integrity_preflight(
    frame: pl.DataFrame,
    config: TournamentConfig,
) -> dict[str, Any]:
    contracts = ATTRIBUTION_TREATMENTS
    for treatment in contracts:
        validate_inference_features(feature_names(treatment))
    candidate_contracts = {
        name: {
            "treatment": _candidate_contract(name)[0],
            "features": list(feature_names(_candidate_contract(name)[0])),
        }
        for name in CANDIDATE_NAMES
    }
    used = {
        feature for contract in candidate_contracts.values() for feature in contract["features"]
    }
    contaminated = sorted(used & SUPERVISION_ONLY_FIELDS)
    if contaminated:
        raise RuntimeError(
            "supervision-only fields entered candidate inference: " + ", ".join(contaminated)
        )
    folds = _development_fold_frames(frame, config)
    prospective_ids = set(
        frame.filter(
            pl.col("window_start").is_between(config.candidate_freeze, config.end, closed="left")
        )["market_id"].unique()
    )
    fold_rows = []
    for fold in config.development_folds:
        fit, test = folds[fold.name]
        test_ids = set(test["market_id"].unique())
        if test_ids & prospective_ids:
            raise RuntimeError(f"development/prospective overlap in {fold.name}")
        fold_rows.append(
            {
                "fold": fold.name,
                "fit_markets": fit["market_id"].n_unique(),
                "test_markets": test["market_id"].n_unique(),
                "fit_test_overlap": 0,
                "prospective_overlap": 0,
            }
        )
    return {
        "passed": True,
        "candidate_roster": list(CANDIDATE_NAMES),
        "candidate_count": len(CANDIDATE_NAMES),
        "candidate_contracts": candidate_contracts,
        "label_source_predictive": False,
        "supervision_fields_in_inference": contaminated,
        "removed_field_substitutes": False,
        "fold_separation": fold_rows,
        "prospective_markets": len(prospective_ids),
        "availability": frame.select(
            pl.len().alias("rows"),
            pl.col("twap_feature_max_available_at").is_not_null().sum().alias("causal_twap_rows"),
        ).to_dicts()[0],
    }


def _artifact_parity_audit(
    artifact_path: Path,
    frame: pl.DataFrame,
    candidate: str,
) -> dict[str, Any]:
    loaded = joblib.load(artifact_path)
    model = loaded["final_models"][candidate]
    sample = frame.tail(min(frame.height, 16))
    batch = score_model(sample, model)
    single = pl.concat(
        [score_model(sample.slice(index, 1), model) for index in range(sample.height)],
        how="vertical",
    )
    columns = (
        "probability_up",
        "predicted_margin_lower_bps",
        "predicted_margin_bps",
        "predicted_margin_upper_bps",
    )
    maximum_error = max(
        float(np.max(np.abs(batch[name].to_numpy() - single[name].to_numpy()), initial=0.0))
        for name in columns
    )
    if maximum_error > 1e-12:
        raise RuntimeError(f"batch/single-row artifact scoring parity failed: {maximum_error}")
    perturbations = []
    for name in sorted(SUPERVISION_ONLY_FIELDS & set(sample.columns)):
        dtype = sample.schema[name]
        if dtype == pl.String:
            perturbations.append(pl.lit("perturbed").alias(name))
        elif dtype == pl.Boolean:
            perturbations.append(pl.col(name).not_().alias(name))
        elif dtype.is_integer():
            perturbations.append((1 - pl.col(name)).cast(dtype).alias(name))
        else:
            perturbations.append(pl.lit(987654.0).cast(dtype).alias(name))
    perturbed = sample.with_columns(*perturbations)
    perturbed_scored = score_model(perturbed, model)
    supervision_maximum_error = max(
        float(
            np.max(
                np.abs(batch[name].to_numpy() - perturbed_scored[name].to_numpy()),
                initial=0.0,
            )
        )
        for name in columns
    )
    if supervision_maximum_error > 1e-12:
        raise RuntimeError(
            "supervision-only perturbation changed artifact predictions: "
            f"{supervision_maximum_error}"
        )
    registry = set(inference_feature_registry())
    exported = {
        feature
        for candidate_model in loaded["final_models"].values()
        for feature in candidate_model.feature_names
    }
    if not exported <= registry:
        raise RuntimeError("exported artifact contains unregistered inference features")
    return {
        "serialization_reload": True,
        "batch_single_row_maximum_absolute_error": maximum_error,
        "supervision_perturbation_maximum_absolute_error": supervision_maximum_error,
        "exported_feature_names_registered": True,
        "exported_feature_names": sorted(exported),
        "rows": sample.height,
        "passed": True,
    }


def _search(
    frame: pl.DataFrame,
    config: TournamentConfig,
    settings: ExecutionSettings | None = None,
) -> tuple[Hyperparameters, dict[str, Any]]:
    settings = settings or execution_settings()
    train = frame.filter(pl.col("window_start") < _utc("2026-08-07T00:00:00Z"))
    validation = frame.filter(
        pl.col("window_start").is_between(
            _utc("2026-08-07T00:00:00Z"), config.current_start, closed="left"
        )
    )

    def evaluate(item: tuple[int, Hyperparameters]) -> dict[str, Any]:
        index, spec = item
        model = fit_probability_model(
            train,
            treatment="refprice_non_twap_basis",
            history_arm="chainlink_history",
            spec=spec,
            seed=config.random_seed + index,
        )
        metrics = probability_metrics(score_probability_model(validation, model))
        return {"index": index, "hyperparameters": asdict(spec), **metrics}

    history = _bounded_fit_map(evaluate, enumerate(predetermined_hyperparameters(config)), settings)
    winner = min(
        history,
        key=lambda row: (
            row["brier"],
            row["log_loss"],
            row["expected_calibration_error"],
            row["index"],
        ),
    )
    return predetermined_hyperparameters(config)[winner["index"]], {
        "combinations": 36,
        "selected": winner,
        "ledger": history,
    }


def _probability_bakeoff(
    frame: pl.DataFrame,
    config: TournamentConfig,
    spec: Hyperparameters,
    *,
    dimension: str,
    history_treatment: str = "combined",
    fold_frames: dict[str, tuple[pl.DataFrame, pl.DataFrame]] | None = None,
    settings: ExecutionSettings | None = None,
) -> dict[str, Any]:
    settings = settings or execution_settings()
    fold_frames = fold_frames or _development_fold_frames(frame, config)
    if dimension != "attribution":
        raise ValueError("the locked tournament permits only the attribution dimension")
    values = ATTRIBUTION_TREATMENTS
    ledgers: dict[str, pl.DataFrame] = {}
    results: dict[str, Any] = {}

    def evaluate(item: tuple[int, str, int, Fold]) -> tuple[str, int, pl.DataFrame, dict[str, Any]]:
        _value_index, value, fold_index, fold = item
        treatment = value
        arm = "chainlink_history"
        fit, test = fold_frames[fold.name]
        model = fit_probability_model(
            fit,
            treatment=treatment,
            history_arm=arm,
            spec=spec,
            seed=config.random_seed + 4000 + fold_index,
        )
        scored = score_probability_model(test, model).with_columns(pl.lit(fold.name).alias("fold"))
        return value, fold_index, scored, {"fold": fold.name, **probability_metrics(scored)}

    tasks = (
        (value_index, value, fold_index, fold)
        for value_index, value in enumerate(values)
        for fold_index, fold in enumerate(config.development_folds)
    )
    evaluated = _bounded_fit_map(evaluate, tasks, settings)
    for value_index, value in enumerate(values):
        rows = sorted((row for row in evaluated if row[0] == value), key=lambda row: row[1])
        pieces = [row[2] for row in rows]
        folds = [row[3] for row in rows]
        ledger = pl.concat(pieces, how="diagonal_relaxed")
        ledgers[value] = ledger
        weights = np.array([row["markets"] for row in folds], dtype=float)
        results[value] = {
            "folds": folds,
            **{
                key: float(np.average([row[key] for row in folds], weights=weights))
                for key in ("brier", "log_loss", "expected_calibration_error")
            },
        }
    control_name = "refprice_only"
    control = ledgers[control_name]
    for index, value in enumerate(values):
        results[value]["paired_brier_bootstrap"] = _paired_bootstrap(
            ledgers[value],
            control,
            config.random_seed + 3000 + index,
            int(config.raw["gates"]["bootstrap_resamples"]),
        )
    comparisons = {
        "basis_vs_refprice": _paired_bootstrap(
            ledgers["refprice_non_twap_basis"],
            ledgers["refprice_only"],
            config.random_seed + 3101,
            int(config.raw["gates"]["bootstrap_resamples"]),
        ),
        "relative_vs_basis": _paired_bootstrap(
            ledgers["refprice_relative_twap"],
            ledgers["refprice_non_twap_basis"],
            config.random_seed + 3102,
            int(config.raw["gates"]["bootstrap_resamples"]),
        ),
        "absolute_vs_relative": _paired_bootstrap(
            ledgers["refprice_absolute_twap"],
            ledgers["refprice_relative_twap"],
            config.random_seed + 3103,
            int(config.raw["gates"]["bootstrap_resamples"]),
        ),
    }
    best_twap = min(
        ("refprice_relative_twap", "refprice_absolute_twap"),
        key=lambda value: (results[value]["brier"], results[value]["log_loss"]),
    )
    comparisons["best_twap_vs_basis"] = _paired_bootstrap(
        ledgers[best_twap],
        ledgers["refprice_non_twap_basis"],
        config.random_seed + 3104,
        int(config.raw["gates"]["bootstrap_resamples"]),
    )
    stability = _paired_stability(
        ledgers[best_twap],
        ledgers["refprice_non_twap_basis"],
        config,
    )
    maximum_ece = float(config.raw["gates"]["maximum_ece"])
    twap_supported = (
        comparisons["best_twap_vs_basis"]["upper"] < 0
        and results[best_twap]["expected_calibration_error"] <= maximum_ece
        and stability["passed"]
    )
    if twap_supported:
        selected = best_twap
    elif (
        comparisons["basis_vs_refprice"]["upper"] < 0
        and results["refprice_non_twap_basis"]["expected_calibration_error"] <= maximum_ece
    ):
        selected = "refprice_non_twap_basis"
    else:
        selected = "refprice_only"
    return {
        "dimension": dimension,
        "results": results,
        "selection": selected,
        "paired_comparisons": comparisons,
        "best_twap_candidate": best_twap,
        "twap_hypothesis_supported": twap_supported,
        "best_twap_stability": stability,
        "decisive_comparison": "best_twap_vs_basis",
        "scored_ledgers": ledgers,
    }


def _fidelity(
    labels: pl.DataFrame, config: TournamentConfig, convention: Any, frame_manifest: dict[str, Any]
) -> dict[str, Any]:
    overlap = labels.filter(
        pl.col("authentic_label_up").is_not_null() & pl.col("proxy_label_up").is_not_null()
    )
    outside = overlap.filter(pl.col("proxy_margin_bps").abs() >= 0.526)
    chainlink = {
        "markets": overlap.height,
        "overall_agreement": float(
            (overlap["authentic_label_up"] == overlap["proxy_label_up"]).mean()
        )
        if overlap.height
        else None,
        "outside_uncertainty_agreement": float(
            (outside["authentic_label_up"] == outside["proxy_label_up"]).mean()
        )
        if outside.height
        else None,
        "p99_price_error_bps": float(convention.calibration_p99_bps),
    }
    chainlink["passed"] = bool(
        (chainlink["overall_agreement"] or 0)
        >= float(config.raw["gates"]["chainlink_minimum_agreement"])
        and (chainlink["outside_uncertainty_agreement"] or 0)
        >= float(config.raw["gates"]["chainlink_outside_band_minimum_agreement"])
    )
    binance = dict(frame_manifest["binance_fidelity"])
    weekly_values = [float(row["agreement"]) for row in binance["weekly"] if row["markets"] >= 20]
    binance["passed"] = bool(
        (binance.get("authentic_agreement") or 0)
        >= float(config.raw["gates"]["binance_minimum_agreement"])
        and weekly_values
        and min(weekly_values) >= 0.98
    )
    return {"chainlink_reconstruction": chainlink, "binance_extension": binance}


def _evidence_conclusions(
    attribution: dict[str, Any],
    status: str,
) -> list[str]:
    basis = attribution["results"]["refprice_non_twap_basis"]
    relative = attribution["results"]["refprice_relative_twap"]
    decisive = attribution["paired_comparisons"]["best_twap_vs_basis"]
    conclusions = [
        (
            "The synthetic TWAP hypothesis is supported: the best TWAP treatment "
            "improved paired Brier loss beyond the non-TWAP basis baseline with "
            "an upper 95% confidence bound below zero."
            if attribution["twap_hypothesis_supported"]
            else "The synthetic TWAP hypothesis is not supported by this frozen "
            "tournament: the best TWAP treatment did not improve paired Brier loss "
            "beyond the non-TWAP basis baseline with an upper 95% confidence bound "
            "below zero."
        )
    ]
    if relative["brier"] < basis["brier"]:
        conclusions.append(
            "Relational TWAP state had a lower point-estimate Brier score than the "
            "non-TWAP basis treatment."
        )
    else:
        conclusions.append(
            "Relational TWAP state did not have a lower point-estimate Brier score "
            "than the non-TWAP basis treatment."
        )
    conclusions.append(
        "Decisive best-TWAP minus basis Brier interval: "
        f"[{decisive['lower']:.6f}, {decisive['upper']:.6f}]."
    )
    if status == "prospective_evidence_pending":
        conclusions.append("Deployment qualification is pending untouched post-freeze evidence.")
    elif status == "no_deployable_challenger_qualified":
        conclusions.append("No new deployable challenger qualified.")
    return conclusions


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
        "twap_hypothesis_supported": development["twap_hypothesis_supported"],
        "paired_predictive_improvement": development["paired_control"]["upper"] < 0,
        "ece": development["probability"]["expected_calibration_error"]
        <= float(gates["maximum_ece"]),
        "positive_stressed_pnl": prospective["economics"]["stressed_pnl"] > 0,
        "positive_stressed_expectancy": prospective["economics"]["stressed_expectancy_per_trade"]
        > 0,
        "profit_factor": prospective["economics"]["profit_factor"]
        >= float(gates["minimum_profit_factor"]),
        "positive_bootstrap_lower": prospective["economics"]["bootstrap_lower"] > 0,
        "profitable_temporal_folds": profitable >= float(gates["minimum_profitable_fold_ratio"]),
        "market_coverage": prospective["economics"]["coverage"]
        >= float(gates["minimum_market_coverage"]),
        "both_directions": min(
            prospective["economics"]["up_trades"], prospective["economics"]["down_trades"]
        )
        > 0,
        "no_single_day_majority": prospective["economics"].get("maximum_day_profit_share", math.inf)
        <= 0.50,
        "prospective_markets": prospective["probability"]["markets"]
        >= int(gates["minimum_prospective_markets"]),
        "prospective_days": prospective["days"] >= int(gates["minimum_prospective_days"]),
        "prospective_folds": prospective["folds"] >= int(gates["minimum_prospective_folds"]),
        "drawdown_and_cvar_reported": (
            math.isfinite(prospective["economics"]["maximum_drawdown"])
            and math.isfinite(prospective["economics"]["cvar_5pct"])
        ),
        "base_capacity_positive": (
            prospective["economics"]["capacity_curve"].get("5") is not None
            and prospective["economics"]["capacity_curve"]["5"]["expectancy"] > 0
        ),
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
        "causal_feature_registry_sha256": causal_feature_registry_sha256(),
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


def run_tournament(
    config: TournamentConfig, *, force_extract: bool = False
) -> tuple[Path, dict[str, Any]]:
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
        f"compute: {settings.workers} concurrent fit, {settings.threads_per_fit} threads per fit",
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
    stage_started = time.perf_counter()
    integrity_preflight = _integrity_preflight(frame, config)
    checkpoints.save("integrity-preflight", integrity_preflight)
    _record_stage_time(stage_timings, "integrity-preflight", stage_started)

    print("search: 36 predetermined probability-model configurations", flush=True)
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
    print("evaluate: four locked causal attribution treatments", flush=True)
    stage_started = time.perf_counter()
    attribution_payload = checkpoints.load("causal-attribution-bakeoff")
    if attribution_payload is None:
        attribution_payload = _probability_bakeoff(
            frame,
            config,
            selected_spec,
            dimension="attribution",
            fold_frames=fold_frames,
            settings=settings,
        )
        checkpoints.save("causal-attribution-bakeoff", attribution_payload)
    scored_ledgers = attribution_payload["scored_ledgers"]
    attribution_results = {
        key: value for key, value in attribution_payload.items() if key != "scored_ledgers"
    }
    provisional = str(attribution_results["selection"])
    _record_stage_time(stage_timings, "causal-attribution-bakeoff", stage_started)

    print(
        f"economics: one fixed policy for predictive winner {provisional}",
        flush=True,
    )
    stage_started = time.perf_counter()
    economics_checkpoint = checkpoints.load("selected-treatment-economics")
    if economics_checkpoint is None:
        policy_fit = frame.filter(pl.col("window_start") < config.authentic_start)
        policy_validation = frame.filter(
            pl.col("window_start").is_between(
                config.authentic_start, config.current_start, closed="left"
            )
        )
        with threadpool_limits(limits=settings.threads_per_fit):
            policy_model = fit_model(
                policy_fit,
                treatment=provisional,
                history_arm="chainlink_history",
                spec=selected_spec,
                seed=config.random_seed + 6000,
                margin_calibrated=False,
            )
        policy_scored = score_model(policy_validation, policy_model)
        policy_economic = _economic_frame(
            policy_scored, config, minimum_start=config.authentic_start
        )
        frozen_policy, admission_ledger = select_policy(
            policy_economic,
            config,
            strict_guard=False,
            seed=config.random_seed + 6100,
        )

        def evaluate_economic_fold(
            item: tuple[int, Fold],
        ) -> tuple[int, dict[str, Any], pl.DataFrame, pl.DataFrame]:
            fold_index, fold = item
            fit, test = fold_frames[fold.name]
            model = fit_model(
                fit,
                treatment=provisional,
                history_arm="chainlink_history",
                spec=selected_spec,
                seed=config.random_seed + 6200 + fold_index,
                margin_calibrated=False,
            )
            scored = score_model(test, model).with_columns(pl.lit(fold.name).alias("fold"))
            eligible = _economic_frame(scored, config)
            trades = _apply_admission(eligible, frozen_policy, strict_guard=False).with_columns(
                pl.lit(fold.name).alias("fold")
            )
            result = {
                "fold": fold.name,
                "probability": probability_metrics(scored),
                "economics": economic_metrics(
                    trades,
                    test["market_id"].n_unique(),
                    resamples=500,
                    seed=config.random_seed + 6300 + fold_index,
                ),
            }
            return (
                fold_index,
                result,
                trades,
                _decision_ledger(scored, eligible, trades, fold=fold.name),
            )

        evaluated = sorted(
            _bounded_fit_map(
                evaluate_economic_fold,
                enumerate(config.development_folds),
                settings,
            ),
            key=lambda row: row[0],
        )
        selected_folds = [row[1] for row in evaluated]
        selected_trade_ledger = pl.concat([row[2] for row in evaluated], how="diagonal_relaxed")
        selected_decision_ledger = pl.concat([row[3] for row in evaluated], how="diagonal_relaxed")
        selected_economics = economic_metrics(
            selected_trade_ledger,
            scored_ledgers[provisional]["market_id"].n_unique(),
            resamples=int(config.raw["gates"]["bootstrap_resamples"]),
            seed=config.random_seed + 6400,
        )
        economics_checkpoint = {
            "policy": frozen_policy,
            "policy_search": admission_ledger,
            "folds": selected_folds,
            "trade_ledger": selected_trade_ledger,
            "decision_ledger": selected_decision_ledger,
            "economics": selected_economics,
        }
        checkpoints.save("selected-treatment-economics", economics_checkpoint)
    frozen_policy = economics_checkpoint["policy"]
    admission_ledger = economics_checkpoint["policy_search"]
    selected_trade_ledger = economics_checkpoint["trade_ledger"]
    selected_decision_ledger = economics_checkpoint["decision_ledger"]
    selected_development = {
        "probability": attribution_results["results"][provisional],
        "economics": economics_checkpoint["economics"],
        "folds": economics_checkpoint["folds"],
        "paired_control": (
            attribution_results["paired_comparisons"]["best_twap_vs_basis"]
            if provisional in ("refprice_relative_twap", "refprice_absolute_twap")
            else attribution_results["paired_comparisons"]["basis_vs_refprice"]
        ),
        "twap_hypothesis_supported": attribution_results["twap_hypothesis_supported"],
    }
    candidate_results = {
        name: {
            "contract": {
                "treatment": name,
                "history_arm": "chainlink_history",
                "feature_treatment_only_difference": True,
            },
            "probability": attribution_results["results"][name],
            "economics": (economics_checkpoint["economics"] if name == provisional else None),
        }
        for name in CANDIDATE_NAMES
    }
    _record_stage_time(stage_timings, "selected-treatment-economics", stage_started)

    stage_started = time.perf_counter()
    provisional_model = checkpoints.load("selected-final-model")
    if provisional_model is None:
        with threadpool_limits(limits=settings.threads_per_fit):
            provisional_model = fit_model(
                frame.filter(pl.col("window_start") < config.candidate_freeze),
                treatment=provisional,
                history_arm="chainlink_history",
                spec=selected_spec,
                seed=config.random_seed + 10000,
                margin_calibrated=False,
            )
        checkpoints.save("selected-final-model", provisional_model)
    final_models = {provisional: provisional_model}
    prospective_available = frame.filter(
        pl.col("window_start").is_between(config.candidate_freeze, config.end, closed="left")
        & (pl.col("label_source") == "authentic_official_twap60")
    )
    prospective_day_audit = _complete_utc_day_audit(prospective_available)
    prospective_frame = (
        prospective_available.filter(
            pl.col("window_start").dt.date().is_in(prospective_day_audit["complete_dates"])
        )
        if prospective_day_audit["complete_dates"]
        else prospective_available.head(0)
    )
    prospective_scored = score_model(prospective_frame, provisional_model)
    prospective_economic = _economic_frame(prospective_scored, config)
    prospective_trades = _apply_admission(prospective_economic, frozen_policy, strict_guard=False)
    prospective_decisions = _decision_ledger(
        prospective_scored,
        prospective_economic,
        prospective_trades,
        fold="untouched_prospective",
    )
    prospective = {
        "start": config.candidate_freeze.isoformat(),
        "end": config.end.isoformat(),
        "probability": probability_metrics(prospective_scored)
        if not prospective_scored.is_empty()
        else {"markets": 0, "brier": None, "log_loss": None, "expected_calibration_error": None},
        "economics": economic_metrics(
            prospective_trades,
            prospective_frame["market_id"].n_unique(),
            resamples=int(config.raw["gates"]["bootstrap_resamples"]),
            seed=config.random_seed + 12000,
        ),
        "days": prospective_frame["window_start"].dt.date().n_unique()
        if not prospective_frame.is_empty()
        else 0,
        "folds": (
            int(prospective_frame["window_start"].dt.date().n_unique()) // 2
            if not prospective_frame.is_empty()
            else 0
        ),
        "policy": frozen_policy,
        "post_freeze_tuning": False,
        "complete_day_audit": prospective_day_audit,
    }
    qualification = _qualification(prospective, selected_development, fidelity, config)
    if prospective["probability"]["markets"] == 0:
        status = "prospective_evidence_pending"
    else:
        status = (
            "deployable_challenger_qualified"
            if qualification["passed"]
            else "no_deployable_challenger_qualified"
        )
    _record_stage_time(stage_timings, "final-selection-and-qualification", stage_started)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    temporary = config.runs / f"{run_id}.partial"
    final = config.committed_results / run_id
    if temporary.exists() or final.exists():
        raise FileExistsError(run_id)
    temporary.mkdir(parents=True)
    ledgers = temporary / "ledgers"
    ledgers.mkdir()
    for name, ledger in scored_ledgers.items():
        ledger.write_parquet(
            ledgers / f"{name}-predictions.parquet",
            compression="zstd",
            statistics=True,
        )
    selected_trade_ledger.write_parquet(
        ledgers / f"{provisional}-trades.parquet",
        compression="zstd",
        statistics=True,
    )
    selected_decision_ledger.write_parquet(
        ledgers / f"{provisional}-decisions.parquet",
        compression="zstd",
        statistics=True,
    )
    prospective_trades.write_parquet(
        ledgers / "prospective-trades.parquet", compression="zstd", statistics=True
    )
    prospective_decisions.write_parquet(
        ledgers / "prospective-decisions.parquet", compression="zstd", statistics=True
    )
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
        "causal_feature_registry_sha256": causal_feature_registry_sha256(),
    }
    joblib.dump(artifact, artifact_path, compress=3)
    artifact_sha = file_sha256(artifact_path)
    artifact_parity = _artifact_parity_audit(artifact_path, frame, provisional)
    conclusions = _evidence_conclusions(attribution_results, status)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "model_family": config.model_family,
        "paper_only": True,
        "training_only": True,
        "live_capital_allowed": False,
        "database_mutations": False,
        "new_tables": False,
        "new_ingesters": False,
        "new_data_sources": False,
        "trading_processes_changed": False,
        "source_commit": source_commit,
        "causal_feature_registry": causal_feature_registry_payload(),
        "causal_feature_registry_sha256": causal_feature_registry_sha256(),
        "configuration": {
            "path": str(config.source_path.relative_to(config.package_root)),
            "sha256": file_sha256(config.source_path),
            "candidate_freeze": config.candidate_freeze.isoformat(),
            "data_watermark": config.end.isoformat(),
        },
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
        "integrity_preflight": integrity_preflight,
        "artifact_parity_audit": artifact_parity,
        "fidelity": fidelity,
        "hyperparameter_ledger": hyperparameter_ledger,
        "twap_attribution": attribution_results,
        "candidate_results": candidate_results,
        "evidence_protocol": {
            "binance_march21_to_june6": "source-fidelity-diagnostic-only",
            "primary_training_history": {
                "start": config.chainlink_start.isoformat(),
                "end": config.candidate_freeze.isoformat(),
                "history_arm": "chainlink_history",
            },
            "authentic_counterfactual_development": {
                "start": config.authentic_start.isoformat(),
                "end": config.current_start.isoformat(),
            },
            "official_current_contract_development": {
                "start": config.current_start.isoformat(),
                "end": config.candidate_freeze.isoformat(),
            },
            "untouched_prospective_qualification": {
                "start": config.candidate_freeze.isoformat(),
                "end": config.end.isoformat(),
                "post_freeze_tuning": False,
            },
        },
        "selection": {
            "provisional_candidate": provisional,
            "status": status,
            "twap_hypothesis_supported": attribution_results["twap_hypothesis_supported"],
        },
        "prospective_qualification": prospective,
        "qualification": qualification,
        "conclusions": conclusions,
        "admission_search": {"attempts": admission_ledger},
        "model_artifact": {"path": "tournament.joblib", "sha256": artifact_sha},
        "limitations": [
            f"Only {prospective['days']} complete post-freeze UTC day(s) and {prospective['probability']['markets']} market(s) are available; the frozen prospective evidence gates remain authoritative.",
            "Binance history before the Chainlink reconstruction boundary is diagnostic only and cannot affect primary model selection.",
            "Projected economics use recorded executable orderbook evidence and do not model queue position.",
            "No model was deployed and no trading process was changed.",
        ],
    }
    _write_json(temporary / "metrics.json", metrics)
    _write_json(temporary / "source-manifest.json", source_manifest)
    _write_json(temporary / "causal-feature-registry.json", causal_feature_registry_payload())
    _write_json(temporary / "supervision-registry.json", SUPERVISION_REGISTRY)
    _write_json(temporary / "integrity-preflight.json", integrity_preflight)
    _write_json(
        temporary / "data-label-coverage.json",
        {"frame_manifest": frame_manifest, "fidelity": fidelity},
    )
    _write_json(temporary / "twap-attribution.json", attribution_results)
    _write_json(temporary / "hyperparameter-ledger.json", hyperparameter_ledger)
    _write_json(
        temporary / "capacity-curves.json",
        {provisional: economics_checkpoint["economics"]["capacity_curve"]},
    )
    _write_json(
        temporary / "qualification.json",
        {
            "selection": metrics["selection"],
            "qualification": qualification,
            "prospective": prospective,
        },
    )
    provenance = {
        "schema_version": "btc-model-provenance-v1",
        "model_family": config.model_family,
        "model_artifact_sha256": artifact_sha,
        "artifact_path": "tournament.joblib",
        "producing_source_commit": source_commit,
        "training_run_id": run_id,
        "source_identity": frame_manifest["frame_sha256"],
        "input_manifest_sha256": hashlib.sha256(
            json.dumps(source_manifest, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "causal_feature_registry_sha256": causal_feature_registry_sha256(),
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
    if isinstance(value, (date, datetime, Path)):
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
        "# BTC 5m Causal TWAP Attribution Tournament",
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
        "## Causal integrity",
        "",
        f"- Feature registry: `{metrics['causal_feature_registry_sha256']}`",
        f"- Point-in-time availability audit passed: `{metrics['frame_manifest']['causal_availability_audit']['passed']}`",
        f"- Serialization and batch/single-row parity passed: `{metrics['artifact_parity_audit']['passed']}`",
        f"- Supervision perturbation parity error: `{metrics['artifact_parity_audit']['supervision_perturbation_maximum_absolute_error']}`",
        "",
        "## Predictive attribution",
        "",
        "| Layer | Brier | Log loss | ECE |",
        "|---|---:|---:|---:|",
    ]
    for name, result in metrics["twap_attribution"]["results"].items():
        rows.append(
            f"| `{name}` | {result['brier']:.5f} | {result['log_loss']:.5f} | "
            f"{result['expected_calibration_error']:.5f} |"
        )
    decisive = metrics["twap_attribution"]["paired_comparisons"]["best_twap_vs_basis"]
    rows.extend(
        [
            "",
            f"- Best TWAP candidate: `{metrics['twap_attribution']['best_twap_candidate']}`",
            (
                f"- Best TWAP minus basis paired Brier: "
                f"`{decisive['candidate_minus_control_brier']:.6f}` "
                f"(95% CI `[{decisive['lower']:.6f}, {decisive['upper']:.6f}]`)"
            ),
            (
                "- Date/direction/entry-band/margin-band stability passed: "
                f"`{metrics['twap_attribution']['best_twap_stability']['passed']}`"
            ),
            (
                "- TWAP hypothesis supported: "
                f"`{metrics['twap_attribution']['twap_hypothesis_supported']}`"
            ),
        ]
    )
    rows.extend(
        [
            "",
            "## Candidate development results",
            "",
            "| Candidate | Brier | ECE | Economic policy | Trades | Coverage | Accuracy | Stressed PnL | Profit factor | Max drawdown | CVaR 5% |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in metrics["candidate_results"].items():
        probability = result["probability"]
        economics = result["economics"]
        if economics is None:
            rows.append(
                f"| `{name}` | {probability['brier']:.5f} | "
                f"{probability['expected_calibration_error']:.5f} | not assigned | "
                "— | — | — | — | — | — | — |"
            )
        else:
            rows.append(
                f"| `{name}` | {probability['brier']:.5f} | "
                f"{probability['expected_calibration_error']:.5f} | fixed | "
                f"{economics['trades']} | {economics['coverage']:.2%} | "
                f"{economics['accuracy']:.2%} | "
                f"${economics['stressed_pnl']:,.2f} | "
                f"{economics['profit_factor']:.2f} | "
                f"${economics['maximum_drawdown']:,.2f} | "
                f"${economics['cvar_5pct']:,.2f} |"
            )
    rows.extend(
        [
            "",
            "## Prospective qualification",
            "",
            f"- Window: `{metrics['prospective_qualification']['start']}` to `{metrics['prospective_qualification']['end']}`",
            f"- Authentic markets: `{metrics['prospective_qualification']['probability']['markets']}`",
            f"- Trades: `{metrics['prospective_qualification']['economics']['trades']}`",
            f"- Stressed PnL: `${metrics['prospective_qualification']['economics']['stressed_pnl']:,.2f}`",
            f"- Maximum drawdown: `${metrics['prospective_qualification']['economics']['maximum_drawdown']:,.2f}`",
            "- Post-freeze tuning: `false`",
            "",
            "## Required conclusion",
            "",
        ]
    )
    rows.extend(f"- {conclusion}" for conclusion in metrics["conclusions"])
    rows.extend(["", "## Failed qualification gates", ""])
    rows.extend(f"- `{reason}`" for reason in qualification["reasons"])
    rows.extend(
        [
            "",
            "## Scope controls",
            "",
            "No database mutation, migration, table, ingester, data source, runtime export, deployment, or trading-process change was performed.",
            "",
        ]
    )
    return "\n".join(rows)
