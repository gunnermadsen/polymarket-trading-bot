from __future__ import annotations

import math
import os
import resource
import subprocess
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from joblib import Parallel, delayed, parallel_config
from scipy.stats import ttest_1samp
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .config import ExpectancyConfig, load_expectancy_config
from .dataset import _sha256, prepare_snapshot
from .evaluation import circular_block_expectancy_ci, trade_ledger
from .features import REGRESSION_FEATURE_SETS, prepare_feature_snapshot
from .models import feature_matrix
from .regression_evaluation import pooled_ledger_economics, regression_metrics
from .regression_models import NonnegativeAffineCalibrator
from .regression_training import _assert_complete_funding, _scan_features
from .reporting import write_json_artifact, write_text_artifact
from .splits import development_slices, frozen_training_slices

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
HORIZON_LABELS = {2: "30m", 4: "1h", 8: "2h", 16: "4h", 32: "8h", 48: "12h"}
STATUS_ORDER = {"promising": 0, "predictive_only": 1, "failed": 2}


@dataclass(frozen=True)
class TournamentCandidate:
    candidate_id: str
    generation: int
    model: str
    horizon_bars: int
    feature_set: str
    target_variant: str = "gross"
    parent_id: str | None = None
    parameter_variant: int = 0
    seed_offset: int = 0


@dataclass
class FittedSignedRegressor:
    candidate: TournamentCandidate
    feature_names: tuple[str, ...]
    estimator: Any
    calibrator: NonnegativeAffineCalibrator
    hyperparameters: dict[str, Any]
    convergence_warnings: tuple[str, ...]

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            raw = np.asarray(
                self.estimator.predict(feature_matrix(frame, self.feature_names)),
                dtype=np.float64,
            )
        calibrated = self.calibrator.predict(raw)
        if self.candidate.target_variant == "vol_scaled":
            calibrated = calibrated * _volatility_scale(frame)
        if not np.isfinite(calibrated).all():
            raise RuntimeError("signed regressor emitted non-finite predictions")
        return calibrated


def _git_revision() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PACKAGE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=PACKAGE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def _volatility_scale(frame: pl.DataFrame) -> np.ndarray:
    return np.clip(frame["volatility_16_bps"].to_numpy().astype(np.float64), 1.0, None)


def _target(frame: pl.DataFrame, variant: str) -> np.ndarray:
    values = frame["gross_forward_bps"].to_numpy().astype(np.float64)
    if variant == "vol_scaled":
        values = values / _volatility_scale(frame)
    elif variant != "gross":
        raise ValueError(f"unsupported target variant {variant}")
    if not np.isfinite(values).all():
        raise RuntimeError("training target contains non-finite values")
    return values


def _model_parameters(candidate: TournamentCandidate, *, seed: int) -> dict[str, Any]:
    variant = candidate.parameter_variant
    if candidate.model == "ridge":
        return {"alpha": (1.0, 10.0, 100.0, 0.1)[variant % 4], "solver": "lsqr"}
    if candidate.model == "elastic_net":
        return {
            "alpha": (0.1, 0.3, 1.0, 3.0)[variant % 4],
            "l1_ratio": (0.1, 0.35, 0.65, 0.9)[variant % 4],
            "max_iter": 20_000,
            "tol": 1e-3,
            "random_state": seed,
        }
    if candidate.model == "histogram":
        return {
            "loss": "squared_error" if variant % 2 == 0 else "absolute_error",
            "learning_rate": (0.04, 0.07, 0.03, 0.08)[variant % 4],
            "max_iter": (220, 180, 300, 160)[variant % 4],
            "max_leaf_nodes": (15, 31, 15, 63)[variant % 4],
            "min_samples_leaf": (100, 200, 300, 150)[variant % 4],
            "l2_regularization": (10.0, 30.0, 100.0, 50.0)[variant % 4],
            "early_stopping": False,
            "random_state": seed,
        }
    if candidate.model == "extra_trees":
        return {
            "n_estimators": (240, 320, 240, 400)[variant % 4],
            "max_features": (0.4, 0.6, 0.8, 0.5)[variant % 4],
            "min_samples_leaf": (25, 50, 100, 150)[variant % 4],
            "max_depth": (14, 18, 12, 20)[variant % 4],
            "bootstrap": False,
            "n_jobs": 1,
            "random_state": seed,
        }
    if candidate.model == "lightgbm":
        return {
            "objective": "regression_l1" if variant % 2 else "regression",
            "n_estimators": (240, 320, 200, 400)[variant % 4],
            "learning_rate": (0.04, 0.025, 0.06, 0.02)[variant % 4],
            "num_leaves": (15, 31, 15, 63)[variant % 4],
            "min_child_samples": (100, 200, 300, 150)[variant % 4],
            "feature_fraction": (0.6, 0.8, 1.0, 0.7)[variant % 4],
            "reg_alpha": (0.0, 1.0, 5.0, 10.0)[variant % 4],
            "reg_lambda": (10.0, 30.0, 100.0, 50.0)[variant % 4],
            "verbosity": -1,
            "n_jobs": 1,
            "random_state": seed,
        }
    raise ValueError(f"unsupported tournament model {candidate.model}")


def _build_estimator(candidate: TournamentCandidate, *, seed: int) -> tuple[Any, dict[str, Any]]:
    parameters = _model_parameters(candidate, seed=seed)
    if candidate.model == "ridge":
        regressor: Any = Ridge(**parameters)
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    ("regressor", regressor),
                ]
            ),
            parameters,
        )
    if candidate.model == "elastic_net":
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    ("regressor", ElasticNet(**parameters)),
                ]
            ),
            parameters,
        )
    if candidate.model == "histogram":
        return HistGradientBoostingRegressor(**parameters), parameters
    if candidate.model == "extra_trees":
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("regressor", ExtraTreesRegressor(**parameters)),
                ]
            ),
            parameters,
        )
    if candidate.model == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**parameters), parameters
    raise AssertionError(candidate.model)


def _fit_model(
    candidate: TournamentCandidate,
    fit: pl.DataFrame,
    calibration: pl.DataFrame,
    *,
    seed: int,
) -> FittedSignedRegressor:
    features = tuple(REGRESSION_FEATURE_SETS[candidate.feature_set])
    estimator, parameters = _build_estimator(candidate, seed=seed)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(feature_matrix(fit, features), _target(fit, candidate.target_variant))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            raw_calibration = np.asarray(
                estimator.predict(feature_matrix(calibration, features)), dtype=np.float64
            )
    convergence_warnings = tuple(
        str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
    )
    calibrator = NonnegativeAffineCalibrator.fit(
        raw_calibration,
        _target(calibration, candidate.target_variant),
    )
    return FittedSignedRegressor(
        candidate,
        features,
        estimator,
        calibrator,
        parameters,
        convergence_warnings,
    )


def _actions(predictions: np.ndarray, hurdle_bps: float, *, no_trade: bool = False) -> np.ndarray:
    if no_trade:
        return np.zeros(predictions.size, dtype=np.int8)
    result = np.zeros(predictions.size, dtype=np.int8)
    result[predictions >= hurdle_bps] = 1
    result[predictions <= -hurdle_bps] = -1
    return result


def _choose_policy(
    frame: pl.DataFrame,
    predictions: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid: list[dict[str, Any]] = []
    for hurdle in (0.0, 3.0, 6.0, 10.0, 15.0):
        ledger = trade_ledger(frame, _actions(predictions, hurdle))
        lower, upper = circular_block_expectancy_ci(
            ledger,
            start=frame["bucket_start"].min().date(),
            end=frame["bucket_start"].max().date(),
            repetitions=repetitions,
            seed=seed,
            confidence=0.80,
            block_days=7,
        )
        expectancy = float(np.mean([row["net_bps"] for row in ledger])) if ledger else None
        grid.append(
            {
                "hurdle_bps": hurdle,
                "trades": len(ledger),
                "net_expectancy_bps": expectancy,
                "bootstrap_80_lower_bps": lower,
                "bootstrap_80_upper_bps": upper,
            }
        )
    eligible = [
        row
        for row in grid
        if row["trades"] >= 50
        and row["bootstrap_80_lower_bps"] is not None
        and row["bootstrap_80_lower_bps"] > 0.0
    ]
    if not eligible:
        return {"hurdle_bps": 0.0, "no_trade": True}, grid
    selected = max(
        eligible,
        key=lambda row: (
            row["bootstrap_80_lower_bps"],
            row["net_expectancy_bps"],
            -row["trades"],
        ),
    )
    return {"hurdle_bps": selected["hurdle_bps"], "no_trade": False}, grid


def _economics(
    frame: pl.DataFrame,
    actions: np.ndarray,
    *,
    multiplier: float,
    repetitions: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ledger = trade_ledger(frame, actions, execution_cost_multiplier=multiplier)
    metrics = pooled_ledger_economics(
        ledger,
        evaluation_rows=frame.height,
        start=frame["bucket_start"].min().date(),
        end=frame["bucket_start"].max().date(),
        bootstrap_repetitions=repetitions,
        seed=seed,
    )
    return metrics, ledger


def _directional_accuracy(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.sign(observed).astype(np.int8)
    estimate = np.sign(predicted).astype(np.int8)
    return {
        "accuracy": float(np.mean(actual == estimate)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, estimate)),
    }


def _fold_task(
    config_path: str,
    feature_path: str,
    candidate: TournamentCandidate,
    fold_index: int,
) -> dict[str, Any]:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    config = load_expectancy_config(config_path)
    fold = config.validation.folds[fold_index]
    frame = _scan_features(Path(feature_path), end=fold.end)
    slices = development_slices(frame, config, fold)
    seed = config.compute.random_seed + candidate.seed_offset + fold_index
    fitted = _fit_model(candidate, slices.fit, slices.calibration, seed=seed)
    threshold_prediction = fitted.predict(slices.threshold)
    policy, grid = _choose_policy(
        slices.threshold,
        threshold_prediction,
        seed=seed + 1_000,
        repetitions=min(300, config.compute.bootstrap_resamples),
    )
    prediction = fitted.predict(slices.evaluation)
    observed = slices.evaluation["gross_forward_bps"].to_numpy().astype(np.float64)
    action = _actions(prediction, policy["hurdle_bps"], no_trade=policy["no_trade"])
    nominal, ledger = _economics(
        slices.evaluation,
        action,
        multiplier=1.0,
        repetitions=config.compute.bootstrap_resamples,
        seed=seed + 2_000,
    )
    stress_15, ledger_15 = _economics(
        slices.evaluation,
        action,
        multiplier=1.5,
        repetitions=config.compute.bootstrap_resamples,
        seed=seed + 3_000,
    )
    stress_20, ledger_20 = _economics(
        slices.evaluation,
        action,
        multiplier=2.0,
        repetitions=config.compute.bootstrap_resamples,
        seed=seed + 4_000,
    )
    delayed = np.zeros_like(action)
    delayed[1:] = action[:-1]
    delayed_metrics, delayed_ledger = _economics(
        slices.evaluation,
        delayed,
        multiplier=1.0,
        repetitions=config.compute.bootstrap_resamples,
        seed=seed + 5_000,
    )
    fixed_zero, _ = _economics(
        slices.evaluation,
        _actions(prediction, 0.0),
        multiplier=1.0,
        repetitions=min(300, config.compute.bootstrap_resamples),
        seed=seed + 6_000,
    )
    return {
        "candidate_id": candidate.candidate_id,
        "fold": fold.name,
        "fold_index": fold_index,
        "seed": seed,
        "rows": {
            "fit": slices.fit.height,
            "calibration": slices.calibration.height,
            "threshold": slices.threshold.height,
            "evaluation": slices.evaluation.height,
        },
        "policy": policy,
        "policy_grid": grid,
        "hyperparameters": fitted.hyperparameters,
        "calibration": asdict(fitted.calibrator),
        "convergence_warnings": list(fitted.convergence_warnings),
        "predictive": {
            **regression_metrics(observed, prediction),
            **_directional_accuracy(observed, prediction),
        },
        "economics": nominal,
        "cost_stress_1_5x": stress_15,
        "cost_stress_2_0x": stress_20,
        "delayed_entry": delayed_metrics,
        "fixed_zero_hurdle": fixed_zero,
        "wall_seconds": time.perf_counter() - wall_started,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
        "_observed": observed,
        "_predicted": prediction,
        "_ledger": ledger,
        "_ledger_15": ledger_15,
        "_ledger_20": ledger_20,
        "_delayed_ledger": delayed_ledger,
    }


def _pooled(
    ledgers: list[dict[str, Any]],
    *,
    rows: int,
    config: ExpectancyConfig,
    seed: int,
) -> dict[str, Any]:
    return pooled_ledger_economics(
        ledgers,
        evaluation_rows=rows,
        start=config.validation.folds[0].start.date(),
        end=(config.validation.folds[-1].end - timedelta(microseconds=1)).date(),
        bootstrap_repetitions=config.compute.bootstrap_resamples,
        seed=seed,
    )


def _spearman_ci(value: float | None, observations: int) -> tuple[float | None, float | None]:
    if value is None or observations <= 3 or abs(value) >= 1.0:
        return None, None
    z = np.arctanh(value)
    delta = 1.96 / math.sqrt(observations - 3)
    return float(np.tanh(z - delta)), float(np.tanh(z + delta))


def _positive_concentration(folds: list[dict[str, Any]]) -> float:
    positive = [
        row["economics"]["total_net_bps"] for row in folds if row["economics"]["total_net_bps"] > 0
    ]
    return max(positive) / sum(positive) if positive else 1.0


def _daily_p_value(ledger: list[dict[str, Any]]) -> float:
    if not ledger:
        return 1.0
    by_day: dict[Any, float] = {}
    for trade in ledger:
        day = trade["decision_time"].date()
        by_day[day] = by_day.get(day, 0.0) + float(trade["net_bps"])
    values = np.asarray(list(by_day.values()), dtype=np.float64)
    if values.size < 2 or np.std(values) == 0.0:
        return 1.0
    result = ttest_1samp(values, popmean=0.0, alternative="greater")
    return float(result.pvalue) if np.isfinite(result.pvalue) else 1.0


def _aggregate(
    candidate: TournamentCandidate,
    folds: list[dict[str, Any]],
    config: ExpectancyConfig,
) -> dict[str, Any]:
    folds = sorted(folds, key=lambda row: row["fold_index"])
    rows = sum(row["rows"]["evaluation"] for row in folds)
    ledger = [trade for row in folds for trade in row["_ledger"]]
    nominal = _pooled(ledger, rows=rows, config=config, seed=config.compute.random_seed + 10_000)
    stress_15 = _pooled(
        [trade for row in folds for trade in row["_ledger_15"]],
        rows=rows,
        config=config,
        seed=config.compute.random_seed + 11_000,
    )
    stress_20 = _pooled(
        [trade for row in folds for trade in row["_ledger_20"]],
        rows=rows,
        config=config,
        seed=config.compute.random_seed + 12_000,
    )
    delayed = _pooled(
        [trade for row in folds for trade in row["_delayed_ledger"]],
        rows=rows,
        config=config,
        seed=config.compute.random_seed + 13_000,
    )
    observed = np.concatenate([row["_observed"] for row in folds])
    predicted = np.concatenate([row["_predicted"] for row in folds])
    predictive = {
        **regression_metrics(observed, predicted),
        **_directional_accuracy(observed, predicted),
    }
    lower, upper = _spearman_ci(predictive["spearman"], predictive["observations"])
    predictive["spearman_95_lower"] = lower
    predictive["spearman_95_upper"] = upper
    positive_folds = sum(
        row["economics"]["net_expectancy_bps"] is not None
        and row["economics"]["net_expectancy_bps"] > 0
        for row in folds
    )
    profit_factor = nominal["profit_factor"] or 0.0
    converged = not any(row["convergence_warnings"] for row in folds)
    if (
        converged
        and nominal["net_expectancy_bps"] is not None
        and nominal["net_expectancy_bps"] > 0
        and positive_folds >= 4
        and profit_factor > 1.0
        and predictive["spearman"] is not None
        and predictive["spearman"] > 0
    ):
        status = "promising"
    elif converged and lower is not None and lower > 0:
        status = "predictive_only"
    else:
        status = "failed"
    compact_folds = []
    for row in folds:
        compact_folds.append({key: value for key, value in row.items() if not key.startswith("_")})
    return {
        **asdict(candidate),
        "horizon": HORIZON_LABELS[candidate.horizon_bars],
        "feature_count": len(REGRESSION_FEATURE_SETS[candidate.feature_set]),
        "status": status,
        "converged": converged,
        "rows": rows,
        "positive_folds": positive_folds,
        "positive_stress_folds": sum(
            row["cost_stress_1_5x"]["net_expectancy_bps"] is not None
            and row["cost_stress_1_5x"]["net_expectancy_bps"] > 0
            for row in folds
        ),
        "positive_fold_pnl_fraction": _positive_concentration(folds),
        "predictive": predictive,
        "economics": nominal,
        "cost_stress_1_5x": stress_15,
        "cost_stress_2_0x": stress_20,
        "delayed_entry": delayed,
        "raw_positive_p_value": _daily_p_value(ledger),
        "holm_adjusted_p_value": None,
        "wall_seconds": sum(row["wall_seconds"] for row in folds),
        "cpu_seconds": sum(row["cpu_seconds"] for row in folds),
        "peak_rss_mb": max(row["peak_rss_mb"] for row in folds),
        "folds": compact_folds,
    }


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    economy = row["economics"]["net_expectancy_bps"]
    spearman = row["predictive"]["spearman"]
    return (
        STATUS_ORDER[row["status"]],
        -(economy if economy is not None else -1e9),
        -(spearman if spearman is not None else -1e9),
        row["candidate_id"],
    )


def _run_candidates(
    candidates: list[TournamentCandidate],
    feature_paths: dict[int, Path],
    config: ExpectancyConfig,
) -> list[dict[str, Any]]:
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("tournament candidate ids must be unique within a comparison")
    jobs = [
        (candidate, fold_index)
        for candidate in candidates
        for fold_index in range(len(config.validation.folds))
    ]
    processes = min(config.compute.available_cores, len(jobs))
    with parallel_config(backend="loky", n_jobs=processes, inner_max_num_threads=1):
        fold_results = Parallel(n_jobs=processes, pre_dispatch=processes)(
            delayed(_fold_task)(
                str(config.source_path),
                str(feature_paths[candidate.horizon_bars]),
                candidate,
                fold_index,
            )
            for candidate, fold_index in jobs
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    for row in fold_results:
        grouped.setdefault(row["candidate_id"], []).append(row)
    for candidate_id, rows in grouped.items():
        if len(rows) != len(config.validation.folds):
            raise RuntimeError(
                f"candidate {candidate_id} produced {len(rows)} folds; "
                f"expected {len(config.validation.folds)}"
            )
    return sorted(
        [_aggregate(by_id[candidate_id], rows, config) for candidate_id, rows in grouped.items()],
        key=_rank_key,
    )


def _generation_one() -> list[TournamentCandidate]:
    horizons = {
        "ridge": (2, 4, 8, 16, 32),
        "elastic_net": (2, 4, 8, 16, 32),
        "histogram": (2, 4, 8, 16),
        "extra_trees": (4, 8, 16, 32),
        "lightgbm": (2, 4, 8, 16, 32),
    }
    candidates = [
        TournamentCandidate(
            f"g1_{model}_{HORIZON_LABELS[horizon]}_positioning",
            1,
            model,
            horizon,
            "positioning",
        )
        for model, model_horizons in horizons.items()
        for horizon in model_horizons
    ]
    candidates.extend(
        TournamentCandidate(
            f"g1_ridge_{HORIZON_LABELS[horizon]}_price", 1, "ridge", horizon, "price"
        )
        for horizon in (2, 4, 8, 16, 32)
    )
    return candidates


def _feature_ablation(parents: list[dict[str, Any]]) -> list[TournamentCandidate]:
    candidates: list[TournamentCandidate] = []
    seen: set[tuple[str, int, str]] = set()
    parent_index = 0
    for parent in parents:
        lineage = (parent["model"], parent["horizon_bars"], parent["feature_set"])
        if lineage in seen:
            continue
        seen.add(lineage)
        parent_index += 1
        for feature_set in ("price", "flow", "microstructure"):
            candidates.append(
                TournamentCandidate(
                    f"g1_ablate_{parent_index}_{parent['model']}_{parent['horizon']}_{feature_set}",
                    1,
                    parent["model"],
                    parent["horizon_bars"],
                    feature_set,
                    parent_id=parent["candidate_id"],
                )
            )
        if parent_index == 3:
            break
    return candidates


def _unique_lineages(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        key = (row["model"], row["horizon_bars"], row["feature_set"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) == limit:
            break
    return unique


def _generation_two(generation_one: list[dict[str, Any]]) -> tuple[list[TournamentCandidate], str]:
    eligible = _unique_lineages(
        [row for row in generation_one if row["status"] != "failed"], limit=2
    )
    if eligible:
        candidates = [
            TournamentCandidate(
                f"g2_{parent_index}_{parent['model']}_{parent['horizon']}_"
                f"{parent['feature_set']}_v{variant}",
                2,
                parent["model"],
                parent["horizon_bars"],
                parent["feature_set"],
                "gross" if parent["status"] == "promising" else "vol_scaled",
                parent["candidate_id"],
                variant,
            )
            for parent_index, parent in enumerate(eligible, start=1)
            for variant in range(4)
        ]
        return candidates, "improve eligible Generation 1 lineages with bounded local tuning"
    candidates = [
        TournamentCandidate(
            f"g2_{model}_{HORIZON_LABELS[horizon]}_vol_scaled",
            2,
            model,
            horizon,
            "positioning",
            "vol_scaled",
            None,
            1,
        )
        for model in ("elastic_net", "histogram", "lightgbm")
        for horizon in (16, 32, 48)
    ]
    return candidates, "pivot to longer-horizon volatility-scaled targets after no eligible lineage"


def _generation_three(
    generation_two: list[dict[str, Any]],
) -> tuple[list[TournamentCandidate], str]:
    eligible = _unique_lineages(
        [row for row in generation_two if row["status"] != "failed"], limit=2
    )
    if eligible:
        return (
            [
                TournamentCandidate(
                    f"g3_{parent_index}_{parent['model']}_{parent['horizon']}_{feature_set}_s{seed_index}",
                    3,
                    parent["model"],
                    parent["horizon_bars"],
                    feature_set,
                    parent["target_variant"],
                    parent["candidate_id"],
                    parent["parameter_variant"],
                    seed_index * 10_000,
                )
                for parent_index, parent in enumerate(eligible, start=1)
                for feature_set in (parent["feature_set"], "flow")
                for seed_index in (1, 2)
            ],
            "harden eligible Generation 2 lineages across features and deterministic seeds",
        )
    parent = generation_two[0]
    return (
        [
            TournamentCandidate(
                f"g3_{model}_{parent['horizon']}_{feature_set}_falsification",
                3,
                model,
                parent["horizon_bars"],
                feature_set,
                parent["target_variant"],
                parent["candidate_id"],
                parent["parameter_variant"] if model == parent["model"] else 0,
            )
            for model in (parent["model"], "ridge")
            for feature_set in ("positioning", "flow")
        ],
        "final longer-horizon falsification after Generation 2 produced no stable signal",
    )


def _holm(all_results: list[dict[str, Any]]) -> None:
    ordered = sorted(all_results, key=lambda row: row["raw_positive_p_value"])
    count = len(ordered)
    running = 0.0
    for index, row in enumerate(ordered):
        adjusted = min(1.0, (count - index) * row["raw_positive_p_value"])
        running = max(running, adjusted)
        row["holm_adjusted_p_value"] = running


def _qualification(row: dict[str, Any]) -> dict[str, Any]:
    economics = row["economics"]
    checks = {
        "positive_folds": row["positive_folds"] >= 5,
        "net_expectancy_bps": economics["net_expectancy_bps"] is not None
        and economics["net_expectancy_bps"] >= 3.0,
        "profit_factor": economics["profit_factor"] is not None
        and economics["profit_factor"] >= 1.15,
        "bootstrap_lower": economics["bootstrap_95_lower_bps"] is not None
        and economics["bootstrap_95_lower_bps"] > 0,
        "stress_folds": row["positive_stress_folds"] >= 5,
        "stress_2x": row["cost_stress_2_0x"]["net_expectancy_bps"] is not None
        and row["cost_stress_2_0x"]["net_expectancy_bps"] > 0,
        "positive_months": economics["positive_month_fraction"] >= 0.60,
        "fold_concentration": row["positive_fold_pnl_fraction"] <= 0.40,
        "effective_trades": economics["trades"] >= 300,
        "spearman": row["predictive"]["spearman"] is not None
        and row["predictive"]["spearman"] >= 0.02
        and row["predictive"]["spearman_95_lower"] is not None
        and row["predictive"]["spearman_95_lower"] > 0,
        "delayed_entry": row["delayed_entry"]["net_expectancy_bps"] is not None
        and row["delayed_entry"]["net_expectancy_bps"] > 0,
        "holm_significance": row["holm_adjusted_p_value"] is not None
        and row["holm_adjusted_p_value"] < 0.05,
        "converged": row["converged"],
    }
    return {"pass": all(checks.values()), "checks": checks}


def _consensus_policy(row: dict[str, Any]) -> dict[str, Any]:
    choices = Counter(
        (fold["policy"]["hurdle_bps"], fold["policy"]["no_trade"]) for fold in row["folds"]
    )
    hurdle, no_trade = min(
        choices,
        key=lambda item: (-choices[item], item[1], item[0]),
    )
    return {"hurdle_bps": hurdle, "no_trade": no_trade}


def _retain_models(
    generation: int,
    results: list[dict[str, Any]],
    feature_paths: dict[int, Path],
    config: ExpectancyConfig,
    run_directory: Path,
    count: int,
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for row in results[:count]:
        candidate = TournamentCandidate(
            row["candidate_id"],
            generation,
            row["model"],
            row["horizon_bars"],
            row["feature_set"],
            row["target_variant"],
            row["parent_id"],
            row["parameter_variant"],
            row["seed_offset"],
        )
        frame = _scan_features(
            feature_paths[candidate.horizon_bars], end=config.validation.holdout_start
        )
        slices = frozen_training_slices(frame, config)
        fitted = _fit_model(
            candidate,
            slices.fit,
            slices.calibration,
            seed=config.compute.random_seed + candidate.seed_offset + 90_000,
        )
        model_path = (
            run_directory
            / f"generation-{generation:02d}"
            / "models"
            / f"{candidate.candidate_id}.joblib"
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_suffix(".tmp")
        joblib.dump({"model": fitted, "policy": _consensus_policy(row)}, temporary, compress=3)
        os.replace(temporary, model_path)
        retained.append(
            {
                "candidate_id": candidate.candidate_id,
                "model": candidate.model,
                "horizon": HORIZON_LABELS[candidate.horizon_bars],
                "artifact_path": str(model_path),
                "artifact_sha256": _sha256(model_path),
                "artifact_bytes": model_path.stat().st_size,
                "policy": _consensus_policy(row),
                "qualification": row.get("qualification"),
            }
        )
    return retained


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Kraken Futures Tournament Generation {report['generation']}",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Decision: {report['decision']}",
        f"- Candidates: {len(report['candidates'])}",
        f"- Holdout rows used: **{report['holdout_rows_used']}**",
        "",
        "| Candidate | Model | Horizon | Features | Status | Spearman | "
        "Net bps/trade | Trades | PF | + folds | 1.5x folds | Delayed bps |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["candidates"]:
        economics = row["economics"]
        lines.append(
            (
                "| {candidate_id} | {model} | {horizon} | {feature_set} | {status} | "
                "{spearman} | {net} | {trades} | {pf} | {positive_folds}/6 | "
                "{positive_stress_folds}/6 | {delayed} |"
            ).format(
                candidate_id=row["candidate_id"],
                model=row["model"],
                horizon=row["horizon"],
                feature_set=row["feature_set"],
                status=row["status"],
                spearman=_fmt(row["predictive"]["spearman"]),
                net=_fmt(economics["net_expectancy_bps"]),
                trades=economics["trades"],
                pf=_fmt(economics["profit_factor"]),
                positive_folds=row["positive_folds"],
                positive_stress_folds=row["positive_stress_folds"],
                delayed=_fmt(row["delayed_entry"]["net_expectancy_bps"]),
            )
        )
    lines.extend(["", "## Retained artifacts", ""])
    for item in report["retained"]:
        lines.append(
            f"- `{item['candidate_id']}` — `{item['artifact_sha256']}` "
            f"({item['artifact_bytes']} bytes)"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _write_generation(
    report: dict[str, Any],
    run_directory: Path,
    report_directory: Path,
) -> None:
    name = f"tournament-generation-{report['generation']:02d}"
    generation_directory = run_directory / f"generation-{report['generation']:02d}"
    write_json_artifact(generation_directory / f"{name}.json", report)
    write_text_artifact(generation_directory / f"{name}.md", _report_markdown(report))
    write_json_artifact(report_directory / f"{name}.json", report)
    write_text_artifact(report_directory / f"{name}.md", _report_markdown(report))


def run_tournament(config: ExpectancyConfig, *, refresh: bool = False) -> dict[str, Any]:
    revision, dirty = _git_revision()
    if dirty:
        raise RuntimeError("tournament must run from a committed kraken-ml package")
    raw = prepare_snapshot(config, refresh=refresh)
    _assert_complete_funding(raw, config)
    horizons = (2, 4, 8, 16, 32, 48)
    snapshots = {
        horizon: prepare_feature_snapshot(config, raw, horizon_bars=horizon, refresh=refresh)
        for horizon in horizons
    }
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{revision[:8]}-{raw.sha256[:8]}"
    run_directory = config.artifacts.root / "tournament" / run_id
    if run_directory.exists():
        raise RuntimeError(f"tournament run already exists: {run_directory}")
    run_directory.mkdir(parents=True)
    report_directory = PACKAGE_ROOT / "reports" / "latest"

    generation_one_screen = _run_candidates(
        _generation_one(),
        {key: value.path for key, value in snapshots.items()},
        config,
    )
    ablation = _run_candidates(
        _feature_ablation(generation_one_screen),
        {key: value.path for key, value in snapshots.items()},
        config,
    )
    generation_one = sorted(generation_one_screen + ablation, key=_rank_key)
    generation_two_candidates, generation_two_decision = _generation_two(generation_one)
    generation_two = _run_candidates(
        generation_two_candidates,
        {key: value.path for key, value in snapshots.items()},
        config,
    )
    generation_three_candidates, generation_three_decision = _generation_three(generation_two)
    generation_three = _run_candidates(
        generation_three_candidates,
        {key: value.path for key, value in snapshots.items()},
        config,
    )
    all_results = generation_one + generation_two + generation_three
    _holm(all_results)
    for row in all_results:
        row["qualification"] = _qualification(row)
    retained_counts = (3, 2, 1)
    reports: list[dict[str, Any]] = []
    decisions = (
        "broad causal model/horizon screen and paired feature ablation",
        generation_two_decision,
        generation_three_decision,
    )
    for generation, results, count, decision in zip(
        (1, 2, 3),
        (generation_one, generation_two, generation_three),
        retained_counts,
        decisions,
        strict=True,
    ):
        retained = _retain_models(
            generation,
            results,
            {key: value.path for key, value in snapshots.items()},
            config,
            run_directory,
            count,
        )
        report = {
            "schema_version": 1,
            "run_id": run_id,
            "generation": generation,
            "decision": decision,
            "git_revision": revision,
            "git_dirty": False,
            "config_fingerprint": config.fingerprint,
            "raw_snapshot_sha256": raw.sha256,
            "feature_snapshots": {
                HORIZON_LABELS[key]: value.sha256 for key, value in snapshots.items()
            },
            "holdout_rows_used": 0,
            "candidates": results,
            "retained": retained,
        }
        _write_generation(report, run_directory, report_directory)
        reports.append(report)

    selected = generation_three[0]
    qualified = bool(selected["qualification"]["pass"])
    final = {
        "schema_version": 1,
        "run_id": run_id,
        "git_revision": revision,
        "raw_snapshot_sha256": raw.sha256,
        "generations": [
            {
                "generation": report["generation"],
                "decision": report["decision"],
                "candidate_count": len(report["candidates"]),
                "retained": report["retained"],
            }
            for report in reports
        ],
        "selected": selected,
        "qualified_for_confirmation": qualified,
        "confirmation": {
            "status": "sealed_ready" if qualified else "not_run",
            "reason": (
                "Generation 3 development gates passed; confirmation requires "
                "explicit one-time evaluation"
                if qualified
                else "no Generation 3 candidate passed every development gate"
            ),
        },
        "holdout": {
            "status": "sealed_not_opened",
            "rows_used": 0,
            "reason": "confirmation was not run"
            if not qualified
            else "confirmation remains sealed",
        },
        "verdict": "qualified_for_confirmation" if qualified else "no_edge_qualified",
    }
    write_json_artifact(run_directory / "tournament-final.json", final)
    write_json_artifact(report_directory / "tournament-final.json", final)
    final_markdown = _final_markdown(final, reports)
    write_text_artifact(run_directory / "tournament-final.md", final_markdown)
    write_text_artifact(report_directory / "tournament-final.md", final_markdown)
    return final


def _final_markdown(final: dict[str, Any], reports: list[dict[str, Any]]) -> str:
    selected = final["selected"]
    lines = [
        "# Kraken Futures Model Tournament",
        "",
        f"**Verdict: {final['verdict'].upper()}**",
        "",
        f"- Run: `{final['run_id']}`",
        f"- Git revision: `{final['git_revision']}`",
        f"- Raw snapshot: `{final['raw_snapshot_sha256']}`",
        f"- Confirmation: {final['confirmation']['status']}",
        f"- Holdout: {final['holdout']['status']} ({final['holdout']['rows_used']} rows used)",
        "",
        "## Generation summaries",
        "",
        "| Generation | Candidates | Best candidate | Status | Spearman | "
        "Net bps/trade | Trades | Qualified |",
        "|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for report in reports:
        row = report["candidates"][0]
        net = _fmt(row["economics"]["net_expectancy_bps"])
        spearman = _fmt(row["predictive"]["spearman"])
        lines.append(
            f"| {report['generation']} | {len(report['candidates'])} | "
            f"{row['candidate_id']} | {row['status']} | {spearman} | {net} | "
            f"{row['economics']['trades']} | {row['qualification']['pass']} |"
        )
    lines.extend(
        [
            "",
            "## Final selection",
            "",
            f"`{selected['candidate_id']}` was the strongest Generation 3 candidate. "
            f"Qualification: **{selected['qualification']['pass']}**.",
            "",
            "The confirmation and global holdout were not used for recursive selection.",
        ]
    )
    return "\n".join(lines) + "\n"
