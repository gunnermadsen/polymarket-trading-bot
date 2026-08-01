from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_config
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_info, threadpool_limits

from .config import BenchmarkConfig, load_config
from .dataset import Snapshot, _atomic_json, _sha256, prepare_snapshot
from .evaluation import (
    Policy,
    baseline_metrics,
    choose_policy,
    classification_metrics,
    economic_metrics,
    policy_actions,
    predicted_classes,
)
from .features import (
    FEATURE_SETS,
    FeatureSnapshot,
    prepare_feature_snapshot,
)
from .models import (
    CLASS_LABELS,
    MODEL_NAMES,
    FittedModel,
    feature_matrix,
    fit_calibrated_model,
    target_vector,
)
from .reporting import (
    write_development_report,
    write_holdout_report,
    write_json_artifact,
    write_text_artifact,
)
from .splits import development_slices, frozen_training_slices

RUN_ID_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}-[0-9a-f]{8}$")
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def prepare_benchmark(
    config: BenchmarkConfig, *, refresh: bool = False
) -> tuple[Snapshot, FeatureSnapshot]:
    raw = prepare_snapshot(config, refresh=refresh)
    features = prepare_feature_snapshot(config, raw, refresh=refresh)
    return raw, features


def _scan_features(
    path: Path,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    query = pl.scan_parquet(path)
    if start is not None:
        query = query.filter(pl.col("bucket_start") >= start)
    if end is not None:
        query = query.filter(pl.col("bucket_start") < end)
    return query.collect()


def _best_baseline_predictions(
    fit_labels: np.ndarray, frame: pl.DataFrame
) -> tuple[str, np.ndarray, float]:
    truth = target_vector(frame)
    counts = np.asarray([(fit_labels == label).sum() for label in CLASS_LABELS])
    majority = int(CLASS_LABELS[np.argmax(counts)])
    candidates = {
        "always_flat": np.zeros_like(truth),
        "training_majority": np.full_like(truth, majority),
        "momentum": frame["momentum_label"].to_numpy().astype(np.int8),
        "contrarian": -frame["momentum_label"].to_numpy().astype(np.int8),
    }
    scored = [
        (
            float(balanced_accuracy_score(truth, predictions)),
            name,
            predictions,
        )
        for name, predictions in candidates.items()
    ]
    score, name, predictions = max(scored, key=lambda row: (row[0], row[1]))
    return name, predictions, score


def _paired_daily_rows(
    frame: pl.DataFrame,
    model_probabilities: np.ndarray,
    baseline_predictions: np.ndarray,
) -> list[dict[str, Any]]:
    truth = target_vector(frame)
    differences = (predicted_classes(model_probabilities) == truth).astype(np.int8) - (
        baseline_predictions == truth
    ).astype(np.int8)
    daily: dict[date, list[int]] = {}
    for timestamp, difference in zip(frame["bucket_start"].to_list(), differences, strict=True):
        day = timestamp.date()
        values = daily.setdefault(day, [0, 0])
        values[0] += int(difference)
        values[1] += 1
    return [
        {"date": day.isoformat(), "correct_difference": values[0], "rows": values[1]}
        for day, values in sorted(daily.items())
    ]


def _fold_task(
    *,
    feature_path: str,
    config_path: str,
    model_name: str,
    feature_set: str,
    fold_index: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    fold = config.validation.folds[fold_index]
    frame = _scan_features(Path(feature_path), end=fold.end)
    slices = development_slices(frame, config, fold)
    seed = config.compute.random_seed + fold_index
    with threadpool_limits(limits=config.compute.comparison_estimator_threads):
        fitted = fit_calibrated_model(
            name=model_name,
            feature_set=feature_set,
            feature_names=FEATURE_SETS[feature_set],
            fit_frame=slices.fit,
            calibration_frame=slices.calibration,
            seed=seed,
            threads=config.compute.comparison_estimator_threads,
        )
        threshold_probabilities = fitted.predict_proba(slices.threshold)
        policy, policy_grid = choose_policy(
            slices.threshold,
            threshold_probabilities,
            config,
            seed_offset=10_000 + fold_index,
        )
        probabilities = fitted.predict_proba(slices.evaluation)

    classification = classification_metrics(target_vector(slices.evaluation), probabilities)
    baselines = baseline_metrics(
        fit_labels=target_vector(slices.fit),
        evaluation_frame=slices.evaluation,
    )
    baseline_name, baseline_predictions, baseline_balanced_accuracy = _best_baseline_predictions(
        target_vector(slices.fit), slices.evaluation
    )
    actions = policy_actions(probabilities, policy)
    economics = economic_metrics(
        slices.evaluation,
        actions,
        execution_cost_multiplier=1.0,
        bootstrap_repetitions=config.compute.bootstrap_resamples,
        seed=seed + 20_000,
    )
    cost_stress = economic_metrics(
        slices.evaluation,
        actions,
        execution_cost_multiplier=config.gates.execution_cost_stress_multiplier,
        bootstrap_repetitions=config.compute.bootstrap_resamples,
        seed=seed + 30_000,
    )
    paired_daily = _paired_daily_rows(slices.evaluation, probabilities, baseline_predictions)
    return {
        "fold": fold.name,
        "model": model_name,
        "feature_set": feature_set,
        "rows": {
            "fit": slices.fit.height,
            "calibration": slices.calibration.height,
            "threshold": slices.threshold.height,
            "evaluation": slices.evaluation.height,
        },
        "ranges": {
            "fit_start": slices.fit["bucket_start"].min(),
            "fit_end": slices.fit["bucket_start"].max(),
            "calibration_start": slices.calibration["bucket_start"].min(),
            "calibration_end": slices.calibration["bucket_start"].max(),
            "threshold_start": slices.threshold["bucket_start"].min(),
            "threshold_end": slices.threshold["bucket_start"].max(),
            "evaluation_start": slices.evaluation["bucket_start"].min(),
            "evaluation_end": slices.evaluation["bucket_start"].max(),
        },
        "classification": classification,
        "baselines": baselines,
        "best_baseline": {
            "name": baseline_name,
            "balanced_accuracy": baseline_balanced_accuracy,
        },
        "balanced_accuracy_uplift": (
            classification["balanced_accuracy"] - baseline_balanced_accuracy
        ),
        "log_loss_uplift": (
            baselines["class_prior_probability"]["log_loss"] - classification["log_loss"]
        ),
        "brier_uplift": (baselines["class_prior_probability"]["brier"] - classification["brier"]),
        "policy": {
            "probability_threshold": policy.probability_threshold,
            "directional_margin": policy.directional_margin,
            "no_trade": policy.no_trade,
        },
        "policy_grid": policy_grid,
        "economics": economics,
        "cost_stress": cost_stress,
        "paired_daily": paired_daily,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _run_fold_tasks(
    *,
    feature_path: Path,
    config: BenchmarkConfig,
    model_names: Iterable[str],
    feature_set: str,
) -> list[dict[str, Any]]:
    tasks = [
        (model_name, fold_index)
        for model_name in model_names
        for fold_index in range(len(config.validation.folds))
    ]
    jobs = min(config.compute.available_cores, len(tasks))
    with parallel_config(
        backend="loky",
        n_jobs=jobs,
        inner_max_num_threads=config.compute.comparison_estimator_threads,
    ):
        results = Parallel(n_jobs=jobs, pre_dispatch=jobs)(
            delayed(_fold_task)(
                feature_path=str(feature_path),
                config_path=str(config.source_path),
                model_name=model_name,
                feature_set=feature_set,
                fold_index=fold_index,
            )
            for model_name, fold_index in tasks
        )
    return sorted(results, key=lambda row: (row["model"], row["fold"]))


def _paired_daily_ci(
    fold_results: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
    block_days: int = 7,
) -> tuple[float, float, float]:
    rows = sorted(
        (row for result in fold_results for row in result["paired_daily"]),
        key=lambda row: row["date"],
    )
    daily_sum = np.asarray([row["correct_difference"] for row in rows], dtype=float)
    daily_count = np.asarray([row["rows"] for row in rows], dtype=float)
    point = float(daily_sum.sum() / daily_count.sum())
    sample_count = len(rows)
    blocks = int(np.ceil(sample_count / block_days))
    offsets = np.arange(block_days)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        starts = rng.integers(0, sample_count, size=blocks)
        indices = ((starts[:, None] + offsets) % sample_count).ravel()[:sample_count]
        estimates[repetition] = daily_sum[indices].sum() / daily_count[indices].sum()
    return (
        point,
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def _mean_metric(results: list[dict[str, Any]], section: str, metric: str) -> float | None:
    values = [
        result[section][metric] for result in results if result[section].get(metric) is not None
    ]
    return float(np.mean(values)) if values else None


def aggregate_fold_results(
    results: list[dict[str, Any]], config: BenchmarkConfig
) -> dict[str, Any]:
    if len(results) != len(config.validation.folds):
        raise RuntimeError("candidate does not contain every development fold")
    paired_point, paired_lower, paired_upper = _paired_daily_ci(
        results,
        repetitions=config.compute.bootstrap_resamples,
        seed=config.compute.random_seed + 40_000,
    )
    mean_balanced_uplift = float(np.mean([row["balanced_accuracy_uplift"] for row in results]))
    summary = {
        "model": results[0]["model"],
        "feature_set": results[0]["feature_set"],
        "fold_count": len(results),
        "mean_accuracy": _mean_metric(results, "classification", "accuracy"),
        "mean_balanced_accuracy": _mean_metric(results, "classification", "balanced_accuracy"),
        "mean_macro_f1": _mean_metric(results, "classification", "macro_f1"),
        "mean_log_loss": _mean_metric(results, "classification", "log_loss"),
        "mean_brier": _mean_metric(results, "classification", "brier"),
        "mean_ece": _mean_metric(results, "classification", "macro_ece"),
        "mean_balanced_accuracy_uplift": mean_balanced_uplift,
        "balanced_accuracy_winning_folds": sum(
            row["balanced_accuracy_uplift"] > 0 for row in results
        ),
        "macro_f1_winning_folds": sum(
            row["classification"]["macro_f1"] > row["baselines"]["best_macro_f1"] for row in results
        ),
        "log_loss_winning_folds": sum(row["log_loss_uplift"] > 0 for row in results),
        "brier_winning_folds": sum(row["brier_uplift"] > 0 for row in results),
        "positive_economic_folds": sum(
            (row["economics"]["net_expectancy_bps"] or float("-inf")) > 0 for row in results
        ),
        "positive_cost_stress_folds": sum(
            (row["cost_stress"]["net_expectancy_bps"] or float("-inf")) > 0 for row in results
        ),
        "no_trade_folds": sum(row["policy"]["no_trade"] for row in results),
        "mean_net_expectancy_bps": _mean_metric(results, "economics", "net_expectancy_bps"),
        "mean_cost_stress_expectancy_bps": _mean_metric(
            results, "cost_stress", "net_expectancy_bps"
        ),
        "total_trades": sum(row["economics"]["trades"] for row in results),
        "paired_accuracy_uplift": paired_point,
        "paired_accuracy_uplift_95_lower": paired_lower,
        "paired_accuracy_uplift_95_upper": paired_upper,
    }
    return summary


def development_gates(aggregate: dict[str, Any], config: BenchmarkConfig) -> dict[str, Any]:
    required_wins = config.gates.minimum_development_positive_folds
    checks = {
        "balanced_accuracy_uplift": {
            "pass": (
                aggregate["mean_balanced_accuracy_uplift"]
                >= config.gates.minimum_balanced_accuracy_uplift
            ),
            "actual": aggregate["mean_balanced_accuracy_uplift"],
            "required": config.gates.minimum_balanced_accuracy_uplift,
        },
        "balanced_accuracy_fold_stability": {
            "pass": aggregate["balanced_accuracy_winning_folds"] >= required_wins,
            "actual": aggregate["balanced_accuracy_winning_folds"],
            "required": required_wins,
        },
        "macro_f1_fold_stability": {
            "pass": aggregate["macro_f1_winning_folds"] >= required_wins,
            "actual": aggregate["macro_f1_winning_folds"],
            "required": required_wins,
        },
        "proper_scoring_rule_stability": {
            "pass": (
                aggregate["log_loss_winning_folds"] >= required_wins
                and aggregate["brier_winning_folds"] >= required_wins
            ),
            "actual": {
                "log_loss_wins": aggregate["log_loss_winning_folds"],
                "brier_wins": aggregate["brier_winning_folds"],
            },
            "required": required_wins,
        },
        "paired_accuracy_confidence": {
            "pass": aggregate["paired_accuracy_uplift_95_lower"] > 0,
            "actual": aggregate["paired_accuracy_uplift_95_lower"],
            "required": "> 0",
        },
        "positive_net_expectancy_stability": {
            "pass": aggregate["positive_economic_folds"] >= required_wins,
            "actual": aggregate["positive_economic_folds"],
            "required": required_wins,
        },
    }
    return {"pass": all(row["pass"] for row in checks.values()), "checks": checks}


def _runtime_metadata(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "polars": pl.__version__,
        "joblib": joblib.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "detected_cores": os.cpu_count() or 1,
        "reserved_cores": config.compute.reserve_cores,
        "parallel_fits": config.compute.available_cores,
        "comparison_estimator_threads": config.compute.comparison_estimator_threads,
        "polars_threads": int(
            os.environ.get(
                "POLARS_MAX_THREADS",
                config.compute.comparison_estimator_threads,
            )
        ),
        "final_refit_threads": min(
            config.compute.final_refit_threads,
            config.compute.available_cores,
        ),
        "threadpools": threadpool_info(),
    }


def _package_lineage() -> dict[str, Any]:
    source_files = sorted(
        {
            PACKAGE_ROOT / "pyproject.toml",
            PACKAGE_ROOT / "requirements.lock",
            *PACKAGE_ROOT.glob("configs/**/*.toml"),
            *PACKAGE_ROOT.glob("src/**/*.py"),
        }
    )
    digest = hashlib.sha256()
    for path in source_files:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    def git_output(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PACKAGE_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git_output("status", "--porcelain", "--", str(PACKAGE_ROOT))
    requirements_path = PACKAGE_ROOT / "requirements.lock"
    return {
        "package_source_sha256": digest.hexdigest(),
        "source_file_count": len(source_files),
        "requirements_lock_sha256": _sha256(requirements_path),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty_for_package": bool(status) if status is not None else None,
    }


def _run_id(config: BenchmarkConfig, features: FeatureSnapshot) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{features.sha256[:8]}-{config.fingerprint[:8]}"


def _resolved_report_directory(config: BenchmarkConfig) -> Path:
    path = config.artifacts.curated_report_directory
    if path.is_absolute():
        return path
    return config.source_path.parent.parent / path


def _class_counts(frame: pl.DataFrame) -> dict[str, int]:
    return {
        str(row["label"]): int(row["len"])
        for row in frame.group_by("label").len().sort("label").iter_rows(named=True)
    }


def _compact_fold_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items() if key not in {"paired_daily", "policy_grid"}
    }


def _development_notes(raw_snapshot: Snapshot) -> list[str]:
    notes = [
        (
            "Funding is excluded from every feature set and is applied only "
            "to realized target and ledger P&L."
        ),
        (
            "Normalized PostgreSQL analytics rows are excluded because the "
            "source audit found an unsafe child index and a slippage gap."
        ),
        (
            "Gross one-hour targets use linear-contract arithmetic returns "
            "relative to entry notional."
        ),
        (
            "Execution costs use the last order-book analytics observation in "
            "the completed 15-minute bucket at each simulated fill; exact "
            "boundary fills would require finer 1-minute or L2 data."
        ),
    ]
    manifest = json.loads(raw_snapshot.manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("contract_metadata", {})
    if contract:
        notes.append(
            "Contract metadata verifies "
            f"{contract.get('type')} {contract.get('base')}/{contract.get('quote')}, "
            f"tick size {contract.get('tick_size')}, and the configured "
            f"{contract.get('base_taker_fee_bps')} bps "
            f"{contract.get('fee_schedule_name')} base taker fee."
        )
    coverage = manifest.get("funding_coverage", {})
    if coverage.get("non_null_rows", 0) < raw_snapshot.row_count:
        notes.append(
            "Funding archive coverage begins at "
            f"{coverage.get('first_timestamp')}; earlier missing rates are treated "
            "as zero and this limitation is not used as a model feature."
        )
    return notes


def _feature_bounds(feature_snapshot: FeatureSnapshot) -> tuple[datetime, datetime, int]:
    row = (
        pl.scan_parquet(feature_snapshot.path)
        .select(
            pl.col("bucket_start").min().alias("start"),
            pl.col("bucket_start").max().alias("end"),
            pl.len().alias("rows"),
        )
        .collect()
        .row(0, named=True)
    )
    return row["start"], row["end"], int(row["rows"])


def _holdout_identity(config: BenchmarkConfig) -> str:
    payload = "|".join(
        (
            "kraken_futures",
            config.dataset.symbol,
            str(config.dataset.interval_seconds),
            config.validation.holdout_start.isoformat(),
            config.validation.holdout_end.isoformat(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_joblib(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    joblib.dump(value, temporary, compress=3)
    digest = _sha256(temporary)
    os.replace(temporary, path)
    return digest


def _freeze_candidate(
    *,
    config: BenchmarkConfig,
    feature_snapshot: FeatureSnapshot,
    raw_snapshot: Snapshot,
    run_directory: Path,
    selected_model: str,
    selected_feature_set: str,
    development: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    frame = _scan_features(
        feature_snapshot.path,
        end=config.validation.holdout_start,
    )
    slices = frozen_training_slices(frame, config)
    threads = min(
        config.compute.final_refit_threads,
        config.compute.available_cores,
    )
    fitted = fit_calibrated_model(
        name=selected_model,
        feature_set=selected_feature_set,
        feature_names=FEATURE_SETS[selected_feature_set],
        fit_frame=slices.fit,
        calibration_frame=slices.calibration,
        seed=config.compute.random_seed,
        threads=threads,
    )
    threshold_probabilities = fitted.predict_proba(slices.threshold)
    policy, policy_grid = choose_policy(
        slices.threshold,
        threshold_probabilities,
        config,
        seed_offset=50_000,
        bootstrap_repetitions=config.compute.bootstrap_resamples,
    )
    actions = policy_actions(threshold_probabilities, policy)
    threshold_classification = classification_metrics(
        target_vector(slices.threshold), threshold_probabilities
    )
    threshold_economics = economic_metrics(
        slices.threshold,
        actions,
        execution_cost_multiplier=1.0,
        bootstrap_repetitions=config.compute.bootstrap_resamples,
        seed=config.compute.random_seed + 60_000,
    )
    threshold_stress = economic_metrics(
        slices.threshold,
        actions,
        execution_cost_multiplier=config.gates.execution_cost_stress_multiplier,
        bootstrap_repetitions=config.compute.bootstrap_resamples,
        seed=config.compute.random_seed + 70_000,
    )
    threshold_checks = {
        "tradeable_policy": {
            "pass": not policy.no_trade,
            "actual": not policy.no_trade,
            "required": True,
        },
        "minimum_policy_trades": {
            "pass": (threshold_economics["trades"] >= config.selection.minimum_calibration_trades),
            "actual": threshold_economics["trades"],
            "required": config.selection.minimum_calibration_trades,
        },
        "positive_95_percent_lower_bound": {
            "pass": (
                threshold_economics["bootstrap_95_lower_bps"] is not None
                and threshold_economics["bootstrap_95_lower_bps"] > 0
            ),
            "actual": threshold_economics["bootstrap_95_lower_bps"],
            "required": "> 0 bps/trade",
        },
        "positive_cost_stress_expectancy": {
            "pass": (
                threshold_stress["net_expectancy_bps"] is not None
                and threshold_stress["net_expectancy_bps"] > 0
            ),
            "actual": threshold_stress["net_expectancy_bps"],
            "required": "> 0 bps/trade",
        },
    }
    threshold_gates = {
        "pass": all(row["pass"] for row in threshold_checks.values()),
        "checks": threshold_checks,
    }
    freeze_summary = {
        "policy": {
            "probability_threshold": policy.probability_threshold,
            "directional_margin": policy.directional_margin,
            "no_trade": policy.no_trade,
        },
        "threshold_classification": threshold_classification,
        "threshold_economics": threshold_economics,
        "threshold_cost_stress": threshold_stress,
        "threshold_gates": threshold_gates,
    }
    if not threshold_gates["pass"]:
        return freeze_summary, False

    freeze_directory = run_directory / "freeze"
    freeze_directory.mkdir(parents=True, exist_ok=False)
    raw_manifest_path = write_text_artifact(
        freeze_directory / "raw-manifest.json",
        raw_snapshot.manifest_path.read_text(encoding="utf-8"),
    )
    feature_manifest_path = write_text_artifact(
        freeze_directory / "feature-manifest.json",
        feature_snapshot.manifest_path.read_text(encoding="utf-8"),
    )
    model_path = freeze_directory / "model.joblib"
    model_sha256 = _atomic_joblib(model_path, fitted)
    policy_path = write_json_artifact(
        freeze_directory / "policy.json",
        {
            "schema_version": 1,
            "policy": freeze_summary["policy"],
        },
    )
    policy_sha256 = _sha256(policy_path)
    lock_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC),
        "config_fingerprint": config.fingerprint,
        "config": config.canonical_payload(),
        "raw_snapshot": {
            "path": raw_snapshot.path,
            "sha256": raw_snapshot.sha256,
            "source_manifest_path": raw_snapshot.manifest_path,
            "manifest_path": raw_manifest_path,
            "manifest_sha256": _sha256(raw_manifest_path),
        },
        "feature_snapshot": {
            "path": feature_snapshot.path,
            "sha256": feature_snapshot.sha256,
            "source_manifest_path": feature_snapshot.manifest_path,
            "manifest_path": feature_manifest_path,
            "manifest_sha256": _sha256(feature_manifest_path),
        },
        "model": {
            "name": fitted.name,
            "feature_set": fitted.feature_set,
            "feature_names": list(fitted.feature_names),
            "hyperparameters": fitted.hyperparameters,
            "path": model_path,
            "sha256": model_sha256,
        },
        "policy": freeze_summary["policy"],
        "policy_artifact": {
            "path": policy_path,
            "sha256": policy_sha256,
        },
        "training": {
            "fit_rows": slices.fit.height,
            "calibration_rows": slices.calibration.height,
            "threshold_rows": slices.threshold.height,
            "fit_class_counts": _class_counts(slices.fit),
            "last_threshold_label_exit": slices.threshold["label_exit_at"].max(),
        },
        "development_gates": development["gates"],
        "code_lineage": development["code_lineage"],
        "runtime_versions": {
            key: development["runtime"][key]
            for key in ("python", "numpy", "polars", "joblib", "scipy", "scikit_learn")
        },
        **freeze_summary,
        "holdout": {
            "identity": _holdout_identity(config),
            "start": config.validation.holdout_start,
            "end": config.validation.holdout_end,
            "status": "sealed",
        },
    }
    lock_path = freeze_directory / "lock.json"
    _atomic_json(lock_path, lock_manifest)
    lock_sha256 = _sha256(lock_path)
    lock_checksum_path = write_text_artifact(
        freeze_directory / "lock.sha256",
        f"{lock_sha256}\n",
    )
    for artifact in (
        raw_manifest_path,
        feature_manifest_path,
        model_path,
        policy_path,
        lock_path,
        lock_checksum_path,
    ):
        artifact.chmod(0o444)
    freeze_summary.update(
        {
            "freeze_directory": str(freeze_directory),
            "lock_manifest": str(lock_path),
            "lock_sha256": lock_sha256,
            "model_sha256": model_sha256,
            "policy_sha256": policy_sha256,
            "holdout_identity": _holdout_identity(config),
        }
    )
    return freeze_summary, True


def run_development(config: BenchmarkConfig, *, refresh: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    raw_snapshot, feature_snapshot = prepare_benchmark(config, refresh=refresh)
    feature_start, feature_end, feature_rows = _feature_bounds(feature_snapshot)
    if feature_rows != feature_snapshot.row_count:
        raise RuntimeError("feature snapshot row count changed after preparation")
    run_id = _run_id(config, feature_snapshot)
    run_directory = config.artifacts.root / "runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    model_results = _run_fold_tasks(
        feature_path=feature_snapshot.path,
        config=config,
        model_names=MODEL_NAMES,
        feature_set="full",
    )
    model_groups = {
        model_name: [result for result in model_results if result["model"] == model_name]
        for model_name in MODEL_NAMES
    }
    model_comparison = {
        model_name: aggregate_fold_results(results, config)
        for model_name, results in model_groups.items()
    }
    selected_model = min(
        MODEL_NAMES,
        key=lambda name: (
            model_comparison[name]["mean_log_loss"],
            -model_comparison[name]["mean_balanced_accuracy"],
            name,
        ),
    )

    ablation_results = [result for result in model_groups[selected_model]]
    for feature_set in ("price", "flow"):
        ablation_results.extend(
            _run_fold_tasks(
                feature_path=feature_snapshot.path,
                config=config,
                model_names=(selected_model,),
                feature_set=feature_set,
            )
        )
    feature_groups = {
        feature_set: [result for result in ablation_results if result["feature_set"] == feature_set]
        for feature_set in FEATURE_SETS
    }
    feature_comparison = {
        feature_set: aggregate_fold_results(results, config)
        for feature_set, results in feature_groups.items()
    }
    selected_feature_set = min(
        FEATURE_SETS,
        key=lambda name: (
            feature_comparison[name]["mean_log_loss"],
            -feature_comparison[name]["mean_balanced_accuracy"],
            name,
        ),
    )
    selected_results = feature_groups[selected_feature_set]
    selected_aggregate = feature_comparison[selected_feature_set]
    for model_name, summary in model_comparison.items():
        summary["selected"] = model_name == selected_model
    for feature_set, summary in feature_comparison.items():
        summary["selected"] = feature_set == selected_feature_set
    gates = development_gates(selected_aggregate, config)
    runtime = _runtime_metadata(config)
    code_lineage = _package_lineage()
    development: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC),
        "objective": (
            "Qualify a reproducible PF_XBTUSD one-hour net edge using classical machine learning."
        ),
        "provenance": {
            "config_path": config.source_path,
            "config_fingerprint": config.fingerprint,
            "canonical_source": "content_addressed_parquet_lake",
            "raw_snapshot_path": raw_snapshot.path,
            "raw_snapshot_sha256": raw_snapshot.sha256,
            "raw_manifest_path": raw_snapshot.manifest_path,
            "raw_manifest_sha256": _sha256(raw_snapshot.manifest_path),
            "feature_snapshot_path": feature_snapshot.path,
            "feature_snapshot_sha256": feature_snapshot.sha256,
            "feature_manifest_path": feature_snapshot.manifest_path,
            "feature_manifest_sha256": _sha256(feature_snapshot.manifest_path),
            "feature_rows": feature_snapshot.row_count,
            "normalized_postgres_training_source": False,
        },
        "code_lineage": code_lineage,
        "hashes": {
            "config_sha256": config.fingerprint,
            "dataset_sha256": raw_snapshot.sha256,
            "feature_sha256": feature_snapshot.sha256,
            "raw_manifest_sha256": _sha256(raw_snapshot.manifest_path),
            "feature_manifest_sha256": _sha256(feature_snapshot.manifest_path),
            "package_source_sha256": code_lineage["package_source_sha256"],
            "requirements_lock_sha256": code_lineage["requirements_lock_sha256"],
        },
        "scope": {
            "symbol": config.dataset.symbol,
            "interval_seconds": config.dataset.interval_seconds,
            "horizon_bars": config.dataset.horizon_bars,
            "start": feature_start,
            "end": feature_end,
            "rows": feature_rows,
            "configured_source_start": config.dataset.start,
            "configured_source_end": config.dataset.end,
            "canonical_source": "content_addressed_parquet_lake",
        },
        "runtime": runtime,
        "selection_protocol": {
            "model": "lowest mean walk-forward calibrated multiclass log loss",
            "feature_set": "lowest mean walk-forward calibrated multiclass log loss",
            "policy": (
                "chronological prior-month threshold search with a positive "
                "80% circular-block bootstrap lower bound"
            ),
            "holdout_evaluated": False,
            "holdout_rows_used_for_fit_calibration_policy_selection": 0,
            "full_range_features_materialized": True,
        },
        "selection_parameters": {
            "probability_thresholds": config.selection.probability_thresholds,
            "directional_margins": config.selection.directional_margins,
            "minimum_calibration_trades": (config.selection.minimum_calibration_trades),
        },
        "model_comparison": model_comparison,
        "model_fold_results": [_compact_fold_result(result) for result in model_results],
        "feature_comparison": feature_comparison,
        "feature_fold_results": [_compact_fold_result(result) for result in ablation_results],
        "selected_model": selected_model,
        "selected_feature_set": selected_feature_set,
        "selected": {
            "model": selected_model,
            "feature_set": selected_feature_set,
        },
        "selected_aggregate": selected_aggregate,
        "selected_fold_results": selected_results,
        "gates": gates,
        "qualified_for_freeze": False,
        "notes": _development_notes(raw_snapshot),
    }
    development["notes"].append(
        "Full-range deterministic features are materialized once, but holdout rows "
        "are excluded from model fitting, calibration, policy selection, and "
        "development evaluation."
    )
    if selected_aggregate["no_trade_folds"] == len(config.validation.folds):
        development["notes"].append(
            "Every selected-fold calibration policy resolved to no-trade because "
            "no candidate with at least "
            f"{config.selection.minimum_calibration_trades} trades had a positive "
            "80% block-bootstrap lower bound."
        )
    freeze_summary: dict[str, Any] | None = None
    if gates["pass"]:
        freeze_summary, frozen = _freeze_candidate(
            config=config,
            feature_snapshot=feature_snapshot,
            raw_snapshot=raw_snapshot,
            run_directory=run_directory,
            selected_model=selected_model,
            selected_feature_set=selected_feature_set,
            development=development,
        )
        development["freeze"] = freeze_summary
        development["qualified_for_freeze"] = frozen
    else:
        development["freeze"] = None
    development["elapsed_seconds"] = time.perf_counter() - started
    development["timings"] = {"total_development_seconds": development["elapsed_seconds"]}
    development["holdout_status"] = (
        "sealed_ready" if development["qualified_for_freeze"] else "sealed_not_qualified"
    )
    development["holdout_state"] = {
        "status": development["holdout_status"],
        "opened": False,
        "start": config.validation.holdout_start,
        "end": config.validation.holdout_end,
        "reason": (
            "candidate passed all pre-holdout gates"
            if development["qualified_for_freeze"]
            else (
                "candidate failed the positive net expectancy stability gate"
                if not gates["checks"]["positive_net_expectancy_stability"]["pass"]
                else "candidate failed at least one pre-holdout gate"
            )
        ),
        "identity": _holdout_identity(config),
    }

    _atomic_json(run_directory / "development.json", development)
    write_development_report(run_directory / "reports", development)
    write_development_report(_resolved_report_directory(config), development)
    return development


def _verify_frozen_environment(config: BenchmarkConfig, lock: dict[str, Any]) -> None:
    current_lineage = _package_lineage()
    for key in ("package_source_sha256", "requirements_lock_sha256"):
        if lock["code_lineage"][key] != current_lineage[key]:
            raise RuntimeError(f"frozen candidate {key} mismatch")
    current_runtime = _runtime_metadata(config)
    for key, expected in lock["runtime_versions"].items():
        if current_runtime.get(key) != expected:
            raise RuntimeError(
                f"frozen candidate runtime version changed for {key}: "
                f"{expected} -> {current_runtime.get(key)}"
            )


def _load_freeze(config: BenchmarkConfig, run_id: str) -> tuple[Path, dict[str, Any], FittedModel]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"invalid run id: {run_id}")
    run_directory = config.artifacts.root / "runs" / run_id
    lock_path = run_directory / "freeze" / "lock.json"
    if not lock_path.exists():
        raise RuntimeError(f"run has no qualified frozen candidate: {run_id}")
    lock_checksum_path = run_directory / "freeze" / "lock.sha256"
    if not lock_checksum_path.exists():
        raise RuntimeError("frozen candidate lock checksum is missing")
    expected_lock_sha256 = lock_checksum_path.read_text(encoding="utf-8").strip()
    if _sha256(lock_path) != expected_lock_sha256:
        raise RuntimeError("frozen candidate lock checksum mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["config_fingerprint"] != config.fingerprint:
        raise RuntimeError("configuration changed after the candidate was frozen")
    _verify_frozen_environment(config, lock)
    if lock["holdout"]["identity"] != _holdout_identity(config):
        raise RuntimeError("frozen holdout identity mismatch")
    raw_path = Path(lock["raw_snapshot"]["path"])
    if _sha256(raw_path) != lock["raw_snapshot"]["sha256"]:
        raise RuntimeError("frozen raw snapshot checksum mismatch")
    raw_manifest_path = Path(lock["raw_snapshot"]["manifest_path"])
    if _sha256(raw_manifest_path) != lock["raw_snapshot"]["manifest_sha256"]:
        raise RuntimeError("frozen raw manifest checksum mismatch")
    feature_path = Path(lock["feature_snapshot"]["path"])
    if _sha256(feature_path) != lock["feature_snapshot"]["sha256"]:
        raise RuntimeError("frozen feature snapshot checksum mismatch")
    feature_manifest_path = Path(lock["feature_snapshot"]["manifest_path"])
    if _sha256(feature_manifest_path) != lock["feature_snapshot"]["manifest_sha256"]:
        raise RuntimeError("frozen feature manifest checksum mismatch")
    model_path = Path(lock["model"]["path"])
    if _sha256(model_path) != lock["model"]["sha256"]:
        raise RuntimeError("frozen model checksum mismatch")
    policy_path = Path(lock["policy_artifact"]["path"])
    if _sha256(policy_path) != lock["policy_artifact"]["sha256"]:
        raise RuntimeError("frozen policy checksum mismatch")
    policy_artifact = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy_artifact.get("policy") != lock["policy"]:
        raise RuntimeError("frozen policy does not match the candidate lock")
    fitted = joblib.load(model_path)
    return run_directory, lock, fitted


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _claim_holdout(
    config: BenchmarkConfig,
    run_directory: Path,
    run_id: str,
) -> tuple[Path, Path]:
    identity = _holdout_identity(config)
    global_marker = config.artifacts.root / "holdouts" / f"{identity}.json"
    run_marker = run_directory / "holdout-access.json"
    payload = {
        "run_id": run_id,
        "holdout_identity": identity,
        "holdout_start": config.validation.holdout_start,
        "holdout_end": config.validation.holdout_end,
        "status": "opened",
        "opened_at": datetime.now(UTC).isoformat(),
    }
    try:
        _exclusive_json(global_marker, payload)
    except FileExistsError as error:
        raise RuntimeError(
            "this market/time-window holdout was already opened by another run"
        ) from error
    try:
        _exclusive_json(run_marker, payload)
    except Exception as error:
        _atomic_json(
            global_marker,
            {
                **payload,
                "status": "failed_and_sealed",
                "failed_at": datetime.now(UTC),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    return run_marker, global_marker


def _fit_labels_from_counts(counts: dict[str, int]) -> np.ndarray:
    return np.concatenate(
        [np.full(int(counts[str(label)]), label, dtype=np.int8) for label in CLASS_LABELS]
    )


def _holdout_importance(
    fitted: FittedModel,
    frame: pl.DataFrame,
    config: BenchmarkConfig,
) -> list[dict[str, Any]]:
    matrix = feature_matrix(frame, fitted.feature_names)
    truth = target_vector(frame)
    max_samples = min(1.0, 10_000 / frame.height)
    with threadpool_limits(limits=config.compute.available_cores):
        result = permutation_importance(
            fitted.calibrator,
            matrix,
            truth,
            scoring="neg_log_loss",
            n_repeats=5,
            # The frozen final estimator already owns its configured thread
            # budget. Serial permutations avoid nested process/thread fan-out.
            n_jobs=1,
            random_state=config.compute.random_seed,
            max_samples=max_samples,
        )
    rows = [
        {
            "feature": name,
            "mean_log_loss_degradation": float(result.importances_mean[index]),
            "standard_deviation": float(result.importances_std[index]),
        }
        for index, name in enumerate(fitted.feature_names)
    ]
    return sorted(
        rows,
        key=lambda row: row["mean_log_loss_degradation"],
        reverse=True,
    )


def _write_holdout_predictions(
    path: Path,
    frame: pl.DataFrame,
    probabilities: np.ndarray,
    actions: np.ndarray,
) -> str:
    output = frame.select(
        [
            "bucket_start",
            "feature_available_at",
            "entry_at",
            "label_exit_at",
            "label",
            "gross_forward_bps",
            "market_execution_cost_bps",
            "long_market_execution_cost_bps",
            "short_market_execution_cost_bps",
            "fee_cost_bps",
            "funding_horizon_bps",
        ]
    ).with_columns(
        [
            pl.Series("probability_short", probabilities[:, 0]),
            pl.Series("probability_flat", probabilities[:, 1]),
            pl.Series("probability_long", probabilities[:, 2]),
            pl.Series("model_class", predicted_classes(probabilities)),
            pl.Series("policy_action", actions),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    output.write_parquet(
        temporary,
        compression="zstd",
        compression_level=9,
        statistics=True,
    )
    digest = _sha256(temporary)
    os.replace(temporary, path)
    return digest


def holdout_gates(
    economics: dict[str, Any],
    cost_stress: dict[str, Any],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    checks = {
        "minimum_trades": {
            "pass": economics["trades"] >= config.gates.minimum_holdout_trades,
            "actual": economics["trades"],
            "required": config.gates.minimum_holdout_trades,
        },
        "minimum_net_expectancy": {
            "pass": (
                economics["net_expectancy_bps"] is not None
                and economics["net_expectancy_bps"] >= config.gates.minimum_net_expectancy_bps
            ),
            "actual": economics["net_expectancy_bps"],
            "required": config.gates.minimum_net_expectancy_bps,
        },
        "positive_95_percent_lower_bound": {
            "pass": (
                economics["bootstrap_95_lower_bps"] is not None
                and economics["bootstrap_95_lower_bps"] > 0
            ),
            "actual": economics["bootstrap_95_lower_bps"],
            "required": "> 0 bps/trade",
        },
        "minimum_profit_factor": {
            "pass": (
                (
                    economics["profit_factor"] is not None
                    and economics["profit_factor"] >= config.gates.minimum_profit_factor
                )
                or (
                    economics["profit_factor"] is None
                    and economics["trades"] > 0
                    and economics["win_rate"] == 1.0
                )
            ),
            "actual": (
                economics["profit_factor"]
                if economics["profit_factor"] is not None
                else "infinite"
                if economics["trades"] > 0 and economics["win_rate"] == 1.0
                else None
            ),
            "required": config.gates.minimum_profit_factor,
        },
        "positive_month_fraction": {
            "pass": (
                economics["positive_month_fraction"] >= config.gates.minimum_positive_month_fraction
            ),
            "actual": economics["positive_month_fraction"],
            "required": config.gates.minimum_positive_month_fraction,
        },
        "positive_cost_stress_expectancy": {
            "pass": (
                cost_stress["net_expectancy_bps"] is not None
                and cost_stress["net_expectancy_bps"] > 0
            ),
            "actual": cost_stress["net_expectancy_bps"],
            "required": (
                f"> 0 bps/trade at "
                f"{config.gates.execution_cost_stress_multiplier:.2f}x execution cost"
            ),
        },
    }
    return {"pass": all(row["pass"] for row in checks.values()), "checks": checks}


def evaluate_holdout(config: BenchmarkConfig, *, run_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    run_directory, lock, fitted = _load_freeze(config, run_id)
    result_path = run_directory / "holdout.json"
    if result_path.exists():
        raise RuntimeError(f"holdout was already evaluated for {run_id}")
    marker, global_marker = _claim_holdout(config, run_directory, run_id)
    try:
        frame = _scan_features(
            Path(lock["feature_snapshot"]["path"]),
            start=config.validation.holdout_start,
            end=config.validation.holdout_end,
        )
        if frame.is_empty():
            raise RuntimeError("locked holdout contains no observations")
        if frame["bucket_start"].min() < config.validation.holdout_start:
            raise RuntimeError("holdout filter included a pre-holdout observation")
        probabilities = fitted.predict_proba(frame)
        policy = Policy(**lock["policy"])
        actions = policy_actions(probabilities, policy)
        classification = classification_metrics(target_vector(frame), probabilities)
        fit_labels = _fit_labels_from_counts(lock["training"]["fit_class_counts"])
        baselines = baseline_metrics(
            fit_labels=fit_labels,
            evaluation_frame=frame,
        )
        baseline_name, baseline_predictions, baseline_score = _best_baseline_predictions(
            fit_labels, frame
        )
        economics = economic_metrics(
            frame,
            actions,
            execution_cost_multiplier=1.0,
            bootstrap_repetitions=config.compute.holdout_bootstrap_resamples,
            seed=config.compute.random_seed + 80_000,
        )
        cost_stress = economic_metrics(
            frame,
            actions,
            execution_cost_multiplier=config.gates.execution_cost_stress_multiplier,
            bootstrap_repetitions=config.compute.holdout_bootstrap_resamples,
            seed=config.compute.random_seed + 90_000,
        )
        double_cost_stress = economic_metrics(
            frame,
            actions,
            execution_cost_multiplier=2.0,
            bootstrap_repetitions=config.compute.holdout_bootstrap_resamples,
            seed=config.compute.random_seed + 100_000,
        )
        gates = holdout_gates(economics, cost_stress, config)
        importance = _holdout_importance(fitted, frame, config)
        prediction_path = run_directory / "holdout" / "predictions.parquet"
        prediction_sha256 = _write_holdout_predictions(
            prediction_path, frame, probabilities, actions
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "evaluated_at": datetime.now(UTC),
            "verdict": ("qualified_positive_net_edge" if gates["pass"] else "no_qualified_edge"),
            "model": {
                "name": fitted.name,
                "feature_set": fitted.feature_set,
                "feature_count": len(fitted.feature_names),
                "model_sha256": lock["model"]["sha256"],
                "policy": lock["policy"],
            },
            "selected": {
                "model": fitted.name,
                "feature_set": fitted.feature_set,
                **lock["policy"],
                "fit_rows": lock["training"]["fit_rows"],
                "calibration_rows": lock["training"]["calibration_rows"],
            },
            "scope": {
                "symbol": config.dataset.symbol,
                "interval_seconds": config.dataset.interval_seconds,
                "horizon_bars": config.dataset.horizon_bars,
                "start": config.validation.holdout_start,
                "end": config.validation.holdout_end,
                "rows": frame.height,
                "canonical_source": "content_addressed_parquet_lake",
            },
            "runtime": _runtime_metadata(config),
            "provenance": {
                "config_fingerprint": config.fingerprint,
                "raw_snapshot_sha256": lock["raw_snapshot"]["sha256"],
                "feature_snapshot_sha256": lock["feature_snapshot"]["sha256"],
                "holdout_start": config.validation.holdout_start,
                "holdout_end": config.validation.holdout_end,
                "holdout_rows": frame.height,
                "prediction_path": prediction_path,
                "prediction_sha256": prediction_sha256,
            },
            "classification": classification,
            "baselines": baselines,
            "best_baseline": {
                "name": baseline_name,
                "balanced_accuracy": baseline_score,
                "model_uplift": (classification["balanced_accuracy"] - baseline_score),
            },
            "economics": economics,
            "economic": {
                "1.0x": economics,
                f"{config.gates.execution_cost_stress_multiplier:.1f}x": cost_stress,
                "2.0x": double_cost_stress,
            },
            "cost_stress": cost_stress,
            "double_cost_stress": double_cost_stress,
            "gates": gates,
            "feature_importance": importance,
            "elapsed_seconds": time.perf_counter() - started,
            "timings": {
                "total_holdout_seconds": time.perf_counter() - started,
            },
            "holdout_access_marker": marker,
            "global_holdout_access_marker": global_marker,
            "notes": [
                "The holdout was evaluated once using the frozen model and policy.",
                "Funding affects realized P&L but is excluded from model features.",
            ],
        }
        _atomic_json(result_path, report)
        _atomic_json(
            marker,
            {
                "run_id": run_id,
                "status": "consumed",
                "opened_at": json.loads(marker.read_text(encoding="utf-8"))["opened_at"],
                "completed_at": datetime.now(UTC),
                "result_path": result_path,
            },
        )
        _atomic_json(
            global_marker,
            {
                "run_id": run_id,
                "holdout_identity": _holdout_identity(config),
                "status": "consumed",
                "opened_at": json.loads(global_marker.read_text(encoding="utf-8"))["opened_at"],
                "completed_at": datetime.now(UTC),
                "result_path": result_path,
            },
        )
        write_holdout_report(run_directory / "reports", report)
        write_holdout_report(_resolved_report_directory(config), report)
        return report
    except Exception as error:
        opened_at = json.loads(marker.read_text(encoding="utf-8"))["opened_at"]
        _atomic_json(
            marker,
            {
                "run_id": run_id,
                "status": "failed_and_sealed",
                "opened_at": opened_at,
                "failed_at": datetime.now(UTC),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        _atomic_json(
            global_marker,
            {
                "run_id": run_id,
                "holdout_identity": _holdout_identity(config),
                "status": "failed_and_sealed",
                "opened_at": json.loads(global_marker.read_text(encoding="utf-8"))["opened_at"],
                "failed_at": datetime.now(UTC),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
