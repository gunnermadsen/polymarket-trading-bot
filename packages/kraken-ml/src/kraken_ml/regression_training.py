from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from .config import CandidateConfig, ExpectancyConfig, load_expectancy_config
from .dataset import (
    Snapshot,
    _sha256,
    prepare_snapshot,
    validate_funding_provenance_binding,
)
from .evaluation import trade_ledger
from .features import (
    REGRESSION_FEATURE_SETS,
    FeatureSnapshot,
    prepare_feature_snapshot,
)
from .regression_evaluation import (
    FEE_COUNTERFACTUALS_BPS,
    RegressionPolicy,
    choose_regression_policy,
    economic_metrics_for_actions,
    fee_counterfactuals,
    pooled_ledger_economics,
    regression_diagnostics,
    regression_policy_actions,
)
from .regression_models import FittedNetRegressors, fit_net_regressors
from .regression_reporting import (
    write_regression_development_report,
    write_regression_holdout_report,
)
from .reporting import write_json_artifact, write_text_artifact
from .splits import development_slices, frozen_training_slices
from .training import (
    _claim_holdout,
    _holdout_identity,
    _package_lineage,
    _resolved_report_directory,
    _runtime_metadata,
)

RUN_ID_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}-[0-9a-f]{8}$")
SIMPLICITY_ORDER = {"ridge": 0, "histogram": 1, "extra_trees": 2}
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


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


def _assert_complete_funding(
    raw_snapshot: Snapshot,
    config: ExpectancyConfig,
) -> dict[str, Any]:
    manifest = json.loads(raw_snapshot.manifest_path.read_text(encoding="utf-8"))
    coverage = manifest.get("funding_coverage")
    if not isinstance(coverage, dict):
        raise RuntimeError("raw snapshot manifest has no funding coverage evidence")
    non_null_rows = int(coverage.get("non_null_rows", -1))
    missing_rows = int(coverage.get("missing_rows", raw_snapshot.row_count - non_null_rows))
    explicit_complete = coverage.get("complete")
    if explicit_complete is False or non_null_rows != raw_snapshot.row_count or missing_rows != 0:
        raise RuntimeError(
            "expectancy training requires complete funding coverage: "
            f"{non_null_rows}/{raw_snapshot.row_count} rows populated, "
            f"{missing_rows} missing"
        )
    first = coverage.get("first_timestamp")
    last = coverage.get("last_timestamp")
    if first is None or last is None:
        raise RuntimeError("funding coverage manifest is missing timestamp bounds")
    first_timestamp = datetime.fromisoformat(str(first).replace("Z", "+00:00"))
    last_timestamp = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
    if (
        first_timestamp > raw_snapshot.first_timestamp
        or last_timestamp < raw_snapshot.last_timestamp
    ):
        raise RuntimeError("funding coverage does not span the complete source snapshot")
    if raw_snapshot.first_timestamp < config.dataset.start:
        raise RuntimeError("raw snapshot begins before the configured source range")
    validate_funding_provenance_binding(manifest, config)
    return coverage


def prepare_expectancy_benchmark(
    config: ExpectancyConfig,
    *,
    refresh: bool = False,
) -> tuple[Snapshot, dict[int, FeatureSnapshot]]:
    raw_snapshot = prepare_snapshot(config, refresh=refresh)
    _assert_complete_funding(raw_snapshot, config)
    horizons = sorted({candidate.horizon_bars for candidate in config.candidates})
    feature_snapshots = {
        horizon: prepare_feature_snapshot(
            config,
            raw_snapshot,
            horizon_bars=horizon,
            refresh=refresh,
        )
        for horizon in horizons
    }
    return raw_snapshot, feature_snapshots


def _fold_task(
    *,
    config_path: str,
    feature_path: str,
    candidate_index: int,
    fold_index: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_expectancy_config(config_path)
    candidate = config.candidates[candidate_index]
    fold = config.validation.folds[fold_index]
    frame = _scan_features(Path(feature_path), end=fold.end)
    slices = development_slices(frame, config, fold)
    seed = config.compute.random_seed + candidate_index * 1_000 + fold_index
    with threadpool_limits(limits=1):
        fitted = fit_net_regressors(
            name=candidate.model,
            feature_set=candidate.feature_set,
            feature_names=REGRESSION_FEATURE_SETS[candidate.feature_set],
            fit_frame=slices.fit,
            calibration_frame=slices.calibration,
            seed=seed,
            threads=1,
        )
        threshold_predictions = fitted.predict(slices.threshold)
        policy, policy_grid = choose_regression_policy(
            slices.threshold,
            threshold_predictions,
            seed=seed + 10_000,
            bootstrap_repetitions=min(500, config.compute.bootstrap_resamples),
            minimum_trades=config.selection.minimum_threshold_trades,
        )
        predictions = fitted.predict(slices.evaluation)
        observed = slices.evaluation.select(
            "long_net_bps",
            "short_net_bps",
        ).to_numpy()

    actions = regression_policy_actions(predictions, policy)
    economics = economic_metrics_for_actions(
        slices.evaluation,
        actions,
        execution_cost_multiplier=1.0,
        bootstrap_repetitions=config.compute.bootstrap_resamples,
        seed=seed + 20_000,
    )
    cost_stress = economic_metrics_for_actions(
        slices.evaluation,
        actions,
        execution_cost_multiplier=config.gates.execution_cost_stress_multiplier,
        bootstrap_repetitions=config.compute.bootstrap_resamples,
        seed=seed + 30_000,
    )
    ledger = trade_ledger(slices.evaluation, actions)
    stress_ledger = trade_ledger(
        slices.evaluation,
        actions,
        execution_cost_multiplier=config.gates.execution_cost_stress_multiplier,
    )
    return {
        "candidate_id": candidate.candidate_id,
        "horizon_bars": candidate.horizon_bars,
        "model": candidate.model,
        "feature_set": candidate.feature_set,
        "fold": fold.name,
        "fold_index": fold_index,
        "seed": seed,
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
        "policy": {
            "hurdle_bps": policy.hurdle_bps,
            "advantage_bps": policy.advantage_bps,
            "expected_return_hurdle_bps": policy.hurdle_bps,
            "directional_advantage_bps": policy.advantage_bps,
            "no_trade": policy.no_trade,
        },
        "policy_grid": policy_grid,
        "regression": regression_diagnostics(slices.evaluation, predictions),
        "_regression_observed": observed,
        "_regression_predictions": predictions,
        "economics": economics,
        "cost_stress": cost_stress,
        "_ledger": ledger,
        "_stress_ledger": stress_ledger,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _fold_jobs(
    config: ExpectancyConfig,
    feature_snapshots: dict[int, FeatureSnapshot],
) -> list[tuple[int, int, str]]:
    jobs = [
        (
            candidate_index,
            fold_index,
            str(feature_snapshots[candidate.horizon_bars].path),
        )
        for candidate_index, candidate in enumerate(config.candidates)
        for fold_index in range(len(config.validation.folds))
    ]
    expected = len(config.candidates) * len(config.validation.folds)
    if len(jobs) != expected or expected != 48:
        raise RuntimeError(f"expected exactly 48 candidate/fold jobs, received {len(jobs)}")
    return jobs


def _run_fold_tasks(
    config: ExpectancyConfig,
    feature_snapshots: dict[int, FeatureSnapshot],
) -> list[dict[str, Any]]:
    tasks = _fold_jobs(config, feature_snapshots)
    process_count = min(config.compute.available_cores, len(tasks))
    if process_count < 1:
        raise RuntimeError("no CPU processes are available for the benchmark")
    with parallel_config(
        backend="loky",
        n_jobs=process_count,
        inner_max_num_threads=1,
    ):
        results = Parallel(n_jobs=process_count, pre_dispatch=process_count)(
            delayed(_fold_task)(
                config_path=str(config.source_path),
                feature_path=feature_path,
                candidate_index=candidate_index,
                fold_index=fold_index,
            )
            for candidate_index, fold_index, feature_path in tasks
        )
    return sorted(results, key=lambda row: (row["candidate_id"], row["fold_index"]))


def _profit_factor_pass(economics: dict[str, Any], required: float) -> bool:
    value = economics.get("profit_factor")
    if value is not None:
        return value >= required
    return economics.get("trades", 0) > 0 and economics.get("win_rate") == 1.0


def _positive_fold_pnl_concentration(results: list[dict[str, Any]]) -> float:
    positive = [
        float(result["economics"]["total_net_bps"])
        for result in results
        if result["economics"]["total_net_bps"] > 0.0
    ]
    if not positive:
        return 1.0
    return max(positive) / sum(positive)


def _pooled_fee_counterfactuals(
    results: list[dict[str, Any]],
    config: ExpectancyConfig,
) -> dict[str, dict[str, Any]]:
    source_ledger = [trade for result in results for trade in result["_ledger"]]
    evaluation_rows = sum(result["rows"]["evaluation"] for result in results)
    start = config.validation.folds[0].start.date()
    end = (config.validation.folds[-1].end - timedelta(microseconds=1)).date()
    counterfactuals: dict[str, dict[str, Any]] = {}
    for index, (name, fee_bps) in enumerate(FEE_COUNTERFACTUALS_BPS.items()):
        ledger = [
            {
                **trade,
                "fee_bps": fee_bps,
                "net_bps": trade["net_bps"] + trade["fee_bps"] - fee_bps,
            }
            for trade in source_ledger
        ]
        counterfactuals[name] = pooled_ledger_economics(
            ledger,
            evaluation_rows=evaluation_rows,
            start=start,
            end=end,
            bootstrap_repetitions=config.compute.bootstrap_resamples,
            seed=config.compute.random_seed + 55_000 + index,
        )
        counterfactuals[name]["round_trip_fee_bps"] = fee_bps
        counterfactuals[name]["fixed_actions_no_reselection"] = True
    return counterfactuals


def development_gates(
    aggregate: dict[str, Any],
    config: ExpectancyConfig,
) -> dict[str, Any]:
    required_folds = config.gates.minimum_development_positive_folds
    economics = aggregate["pooled_economics"]
    checks = {
        "nominal_positive_fold_stability": {
            "pass": aggregate["positive_nominal_folds"] >= required_folds,
            "actual": aggregate["positive_nominal_folds"],
            "required": required_folds,
        },
        "minimum_pooled_net_expectancy": {
            "pass": (
                economics["net_expectancy_bps"] is not None
                and economics["net_expectancy_bps"]
                >= config.gates.minimum_net_expectancy_bps
            ),
            "actual": economics["net_expectancy_bps"],
            "required": config.gates.minimum_net_expectancy_bps,
        },
        "cost_stress_positive_fold_stability": {
            "pass": aggregate["positive_stress_folds"] >= required_folds,
            "actual": aggregate["positive_stress_folds"],
            "required": required_folds,
        },
        "positive_daily_block_bootstrap_lower": {
            "pass": (
                economics["bootstrap_95_lower_bps"] is not None
                and economics["bootstrap_95_lower_bps"] > 0.0
            ),
            "actual": economics["bootstrap_95_lower_bps"],
            "required": "> 0 bps/trade",
        },
        "minimum_profit_factor": {
            "pass": _profit_factor_pass(economics, config.gates.minimum_profit_factor),
            "actual": economics["profit_factor"],
            "required": config.gates.minimum_profit_factor,
        },
        "positive_calendar_month_fraction": {
            "pass": (
                economics["positive_month_fraction"]
                >= config.gates.minimum_positive_month_fraction
            ),
            "actual": economics["positive_month_fraction"],
            "required": config.gates.minimum_positive_month_fraction,
        },
        "positive_fold_pnl_concentration": {
            "pass": (
                aggregate["positive_fold_pnl_fraction"]
                <= config.gates.maximum_positive_fold_pnl_fraction
            ),
            "actual": aggregate["positive_fold_pnl_fraction"],
            "required": f"<= {config.gates.maximum_positive_fold_pnl_fraction}",
        },
    }
    return {"pass": all(check["pass"] for check in checks.values()), "checks": checks}


def _pooled_regression_diagnostics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    observed = np.concatenate(
        [result["_regression_observed"] for result in results],
        axis=0,
    )
    predictions = np.concatenate(
        [result["_regression_predictions"] for result in results],
        axis=0,
    )
    return regression_diagnostics(
        pl.DataFrame(
            {
                "long_net_bps": observed[:, 0],
                "short_net_bps": observed[:, 1],
            }
        ),
        predictions,
    )


def aggregate_candidate_results(
    results: list[dict[str, Any]],
    config: ExpectancyConfig,
) -> dict[str, Any]:
    if len(results) != len(config.validation.folds):
        raise RuntimeError("candidate result does not contain every development fold")
    results = sorted(results, key=lambda row: row["fold_index"])
    ledgers = [trade for result in results for trade in result["_ledger"]]
    stress_ledgers = [trade for result in results for trade in result["_stress_ledger"]]
    start = config.validation.folds[0].start.date()
    end = (config.validation.folds[-1].end - timedelta(microseconds=1)).date()
    evaluation_rows = sum(result["rows"]["evaluation"] for result in results)
    economics = pooled_ledger_economics(
        ledgers,
        evaluation_rows=evaluation_rows,
        start=start,
        end=end,
        bootstrap_repetitions=config.compute.bootstrap_resamples,
        seed=config.compute.random_seed + 40_000,
    )
    stress = pooled_ledger_economics(
        stress_ledgers,
        evaluation_rows=evaluation_rows,
        start=start,
        end=end,
        bootstrap_repetitions=config.compute.bootstrap_resamples,
        seed=config.compute.random_seed + 50_000,
    )
    fold_stress = [
        (
            result["cost_stress"]["net_expectancy_bps"]
            if result["cost_stress"]["net_expectancy_bps"] is not None
            else 0.0
        )
        for result in results
    ]
    aggregate = {
        "candidate_id": results[0]["candidate_id"],
        "horizon_bars": results[0]["horizon_bars"],
        "model": results[0]["model"],
        "feature_set": results[0]["feature_set"],
        "fold_count": len(results),
        "positive_nominal_folds": sum(
            result["economics"]["net_expectancy_bps"] is not None
            and result["economics"]["net_expectancy_bps"] > 0.0
            for result in results
        ),
        "positive_stress_folds": sum(
            result["cost_stress"]["net_expectancy_bps"] is not None
            and result["cost_stress"]["net_expectancy_bps"] > 0.0
            for result in results
        ),
        "no_trade_folds": sum(result["policy"]["no_trade"] for result in results),
        "median_fold_stress_expectancy_bps": float(np.median(fold_stress)),
        "turnover": economics["action_coverage"],
        "positive_fold_pnl_fraction": _positive_fold_pnl_concentration(results),
        "pooled_economics": economics,
        "pooled_stress": stress,
        "pooled_regression": _pooled_regression_diagnostics(results),
        "fee_counterfactuals": _pooled_fee_counterfactuals(results, config),
        "oi_qualification": None,
    }
    aggregate["gates"] = development_gates(aggregate, config)
    aggregate["qualified"] = aggregate["gates"]["pass"]
    return aggregate


def apply_oi_qualification(
    aggregates: list[dict[str, Any]],
    fold_results: list[dict[str, Any]],
    config: ExpectancyConfig,
) -> None:
    baseline_id = "h4_extra_trees_price"
    baseline = next(
        aggregate for aggregate in aggregates if aggregate["candidate_id"] == baseline_id
    )
    baseline_folds = {
        result["fold"]: result
        for result in fold_results
        if result["candidate_id"] == baseline_id
    }
    for aggregate in aggregates:
        if aggregate["feature_set"] not in {"oi", "price_oi"}:
            continue
        candidate_folds = [
            result
            for result in fold_results
            if result["candidate_id"] == aggregate["candidate_id"]
        ]
        wins = sum(
            result["economics"]["net_expectancy_bps"] is not None
            and baseline_folds[result["fold"]]["economics"]["net_expectancy_bps"] is not None
            and result["economics"]["net_expectancy_bps"]
            > baseline_folds[result["fold"]]["economics"]["net_expectancy_bps"]
            for result in candidate_folds
        )
        candidate_stress = aggregate["pooled_stress"]["net_expectancy_bps"]
        baseline_stress = baseline["pooled_stress"]["net_expectancy_bps"]
        no_lower_stress = (
            candidate_stress is not None
            and baseline_stress is not None
            and candidate_stress >= baseline_stress
        )
        oi_checks = {
            "paired_nominal_fold_wins": {
                "pass": wins >= config.gates.minimum_oi_paired_wins,
                "actual": wins,
                "required": config.gates.minimum_oi_paired_wins,
            },
            "no_lower_aggregate_stressed_expectancy": {
                "pass": no_lower_stress,
                "actual": candidate_stress,
                "required": f">= price baseline {baseline_stress}",
            },
        }
        aggregate["oi_qualification"] = {
            "baseline_candidate_id": baseline_id,
            "pass": all(check["pass"] for check in oi_checks.values()),
            "checks": oi_checks,
        }
        aggregate["gates"]["checks"].update(
            {f"oi_{name}": check for name, check in oi_checks.items()}
        )
        aggregate["gates"]["pass"] = all(
            check["pass"] for check in aggregate["gates"]["checks"].values()
        )
        aggregate["qualified"] = aggregate["gates"]["pass"]


def select_candidate(
    aggregates: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    if not aggregates:
        raise ValueError("candidate selection requires aggregates")
    qualified = [aggregate for aggregate in aggregates if aggregate["qualified"]]
    diagnostic_only = not qualified
    pool = qualified or aggregates
    selected = min(
        pool,
        key=lambda aggregate: (
            -(
                aggregate["median_fold_stress_expectancy_bps"]
                if aggregate["median_fold_stress_expectancy_bps"] is not None
                else float("-inf")
            ),
            aggregate["turnover"],
            SIMPLICITY_ORDER[aggregate["model"]],
            aggregate["candidate_id"],
        ),
    )
    return selected, diagnostic_only


def _confirmation_gates(
    economics: dict[str, Any],
    cost_stress: dict[str, Any],
    config: ExpectancyConfig,
) -> dict[str, Any]:
    checks = {
        "minimum_confirmation_trades": {
            "pass": economics["trades"] >= config.selection.minimum_threshold_trades,
            "actual": economics["trades"],
            "required": config.selection.minimum_threshold_trades,
        },
        "minimum_net_expectancy": {
            "pass": (
                economics["net_expectancy_bps"] is not None
                and economics["net_expectancy_bps"]
                >= config.gates.minimum_net_expectancy_bps
            ),
            "actual": economics["net_expectancy_bps"],
            "required": config.gates.minimum_net_expectancy_bps,
        },
        "positive_daily_block_bootstrap_lower": {
            "pass": (
                economics["bootstrap_95_lower_bps"] is not None
                and economics["bootstrap_95_lower_bps"] > 0.0
            ),
            "actual": economics["bootstrap_95_lower_bps"],
            "required": "> 0 bps/trade",
        },
        "minimum_profit_factor": {
            "pass": _profit_factor_pass(economics, config.gates.minimum_profit_factor),
            "actual": economics["profit_factor"],
            "required": config.gates.minimum_profit_factor,
        },
        "positive_calendar_month_fraction": {
            "pass": (
                economics["positive_month_fraction"]
                >= config.gates.minimum_positive_month_fraction
            ),
            "actual": economics["positive_month_fraction"],
            "required": config.gates.minimum_positive_month_fraction,
        },
        "positive_cost_stress_expectancy": {
            "pass": (
                cost_stress["net_expectancy_bps"] is not None
                and cost_stress["net_expectancy_bps"] > 0.0
            ),
            "actual": cost_stress["net_expectancy_bps"],
            "required": (
                f"> 0 bps/trade at "
                f"{config.gates.execution_cost_stress_multiplier:.2f}x execution cost"
            ),
        },
    }
    return {"pass": all(check["pass"] for check in checks.values()), "checks": checks}


def _development_policy_counts(
    fold_results: list[dict[str, Any]],
) -> Counter[tuple[float, float]]:
    policies: Counter[tuple[float, float]] = Counter()
    for result in fold_results:
        policy = result["policy"]
        if policy["no_trade"]:
            continue
        policies[
            (
                float(policy["hurdle_bps"]),
                float(policy["advantage_bps"]),
            )
        ] += 1
    return policies


def _consensus_policy(
    fold_results: list[dict[str, Any]],
) -> RegressionPolicy:
    policies = _development_policy_counts(fold_results)
    if not policies:
        return RegressionPolicy(hurdle_bps=0.0, advantage_bps=0.0, no_trade=True)
    hurdle_bps, advantage_bps = min(
        policies,
        key=lambda policy: (
            -policies[policy],
            -policy[0],
            -policy[1],
        ),
    )
    return RegressionPolicy(
        hurdle_bps=hurdle_bps,
        advantage_bps=advantage_bps,
    )


def _final_confirmation(
    config: ExpectancyConfig,
    candidate: CandidateConfig,
    feature_snapshot: FeatureSnapshot,
    policy: RegressionPolicy,
    policy_preregistration: dict[str, Any],
) -> tuple[dict[str, Any], FittedNetRegressors, RegressionPolicy]:
    frame = _scan_features(feature_snapshot.path, end=config.validation.holdout_start)
    slices = frozen_training_slices(frame, config)
    threads = min(config.compute.final_refit_threads, config.compute.available_cores)
    fitted = fit_net_regressors(
        name=candidate.model,
        feature_set=candidate.feature_set,
        feature_names=REGRESSION_FEATURE_SETS[candidate.feature_set],
        fit_frame=slices.fit,
        calibration_frame=slices.calibration,
        seed=config.compute.random_seed,
        threads=threads,
    )
    predictions = fitted.predict(slices.threshold)
    actions = regression_policy_actions(predictions, policy)
    economics = economic_metrics_for_actions(
        slices.threshold,
        actions,
        execution_cost_multiplier=1.0,
        bootstrap_repetitions=config.compute.bootstrap_resamples,
        seed=config.compute.random_seed + 70_000,
    )
    cost_stress = economic_metrics_for_actions(
        slices.threshold,
        actions,
        execution_cost_multiplier=config.gates.execution_cost_stress_multiplier,
        bootstrap_repetitions=config.compute.bootstrap_resamples,
        seed=config.compute.random_seed + 80_000,
    )
    gates = _confirmation_gates(economics, cost_stress, config)
    confirmation = {
        "status": "passed" if gates["pass"] else "failed",
        "rows": {
            "fit": slices.fit.height,
            "calibration": slices.calibration.height,
            "confirmation": slices.threshold.height,
        },
        "range": {
            "start": slices.threshold["bucket_start"].min(),
            "end": slices.threshold["bucket_start"].max(),
            "last_label_exit": slices.threshold["label_exit_at"].max(),
        },
        "policy": {
            "hurdle_bps": policy.hurdle_bps,
            "advantage_bps": policy.advantage_bps,
            "expected_return_hurdle_bps": policy.hurdle_bps,
            "directional_advantage_bps": policy.advantage_bps,
            "no_trade": policy.no_trade,
        },
        "policy_source": (
            "modal non-no-trade policy from the selected candidate's six "
            "development folds; ties choose the higher hurdle, then advantage"
        ),
        "policy_preregistration": policy_preregistration,
        "regression": regression_diagnostics(slices.threshold, predictions),
        "economics": economics,
        "cost_stress": cost_stress,
        "fee_counterfactuals": fee_counterfactuals(
            slices.threshold,
            actions,
            execution_cost_multiplier=1.0,
            bootstrap_repetitions=config.compute.bootstrap_resamples,
            seed=config.compute.random_seed + 100_000,
        ),
        "gates": gates,
    }
    return confirmation, fitted, policy


def _runtime_fingerprint(runtime: dict[str, Any]) -> str:
    stable = {
        key: runtime.get(key)
        for key in (
            "platform",
            "machine",
            "python",
            "numpy",
            "polars",
            "joblib",
            "scipy",
            "scikit_learn",
        )
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_joblib(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    joblib.dump(value, temporary, compress=3)
    digest = _sha256(temporary)
    os.replace(temporary, path)
    return digest


def _regression_policy_payload(policy: RegressionPolicy) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "hurdle_bps": policy.hurdle_bps,
        "advantage_bps": policy.advantage_bps,
        "no_trade": policy.no_trade,
    }


def _freeze_candidate(
    *,
    config: ExpectancyConfig,
    raw_snapshot: Snapshot,
    feature_snapshot: FeatureSnapshot,
    run_directory: Path,
    candidate: CandidateConfig,
    fitted: FittedNetRegressors,
    policy: RegressionPolicy,
    confirmation: dict[str, Any],
    development_gates_payload: dict[str, Any],
    code_lineage: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    if not confirmation["gates"]["pass"] or not development_gates_payload["pass"]:
        raise RuntimeError("only a fully qualified candidate can be frozen")
    confirmation_policy = confirmation["policy_preregistration"]
    confirmation_policy_path = Path(confirmation_policy["path"])
    if _sha256(confirmation_policy_path) != confirmation_policy["sha256"]:
        raise RuntimeError("final confirmation policy preregistration checksum mismatch")
    freeze_directory = run_directory / "regression-freeze"
    freeze_directory.mkdir(parents=True, exist_ok=False)
    raw_manifest = write_text_artifact(
        freeze_directory / "raw-manifest.json",
        raw_snapshot.manifest_path.read_text(encoding="utf-8"),
    )
    feature_manifest = write_text_artifact(
        freeze_directory / "feature-manifest.json",
        feature_snapshot.manifest_path.read_text(encoding="utf-8"),
    )
    model_path = freeze_directory / "net-regressors.joblib"
    model_sha256 = _atomic_joblib(model_path, fitted)
    policy_payload = _regression_policy_payload(policy)
    policy_path = write_json_artifact(freeze_directory / "policy.json", policy_payload)
    policy_sha256 = _sha256(policy_path)
    lock = {
        "schema_version": 1,
        "created_at": datetime.now(UTC),
        "config_fingerprint": config.fingerprint,
        "config": config.canonical_payload(),
        "candidate": {
            "candidate_id": candidate.candidate_id,
            "horizon_bars": candidate.horizon_bars,
            "model": candidate.model,
            "feature_set": candidate.feature_set,
            "feature_names": REGRESSION_FEATURE_SETS[candidate.feature_set],
        },
        "raw_snapshot": {
            "path": raw_snapshot.path,
            "sha256": raw_snapshot.sha256,
            "manifest_path": raw_manifest,
            "manifest_sha256": _sha256(raw_manifest),
        },
        "feature_snapshot": {
            "path": feature_snapshot.path,
            "sha256": feature_snapshot.sha256,
            "manifest_path": feature_manifest,
            "manifest_sha256": _sha256(feature_manifest),
        },
        "model": {
            "path": model_path,
            "sha256": model_sha256,
            "hyperparameters": fitted.hyperparameters,
            "side_targets": ["long_net_bps", "short_net_bps"],
            "calibrators": [
                {
                    "slope": calibrator.slope,
                    "intercept": calibrator.intercept,
                }
                for calibrator in fitted.calibrators
            ],
        },
        "policy": policy_payload,
        "policy_artifact": {
            "path": policy_path,
            "sha256": policy_sha256,
        },
        "confirmation_policy_artifact": {
            "path": confirmation_policy_path,
            "sha256": confirmation_policy["sha256"],
        },
        "development_gates": development_gates_payload,
        "final_confirmation": confirmation,
        "code_lineage": code_lineage,
        "runtime": runtime,
        "runtime_sha256": _runtime_fingerprint(runtime),
        "hashes": {
            "source_sha256": raw_snapshot.sha256,
            "feature_sha256": feature_snapshot.sha256,
            "config_sha256": config.fingerprint,
            "code_sha256": code_lineage["package_source_sha256"],
            "runtime_sha256": _runtime_fingerprint(runtime),
            "model_sha256": model_sha256,
            "policy_sha256": policy_sha256,
            "confirmation_policy_sha256": confirmation_policy["sha256"],
        },
        "holdout": {
            "identity": _holdout_identity(config),
            "start": config.validation.holdout_start,
            "end": config.validation.holdout_end,
            "status": "sealed",
        },
    }
    lock_path = freeze_directory / "lock.json"
    write_json_artifact(lock_path, lock)
    lock_sha256 = _sha256(lock_path)
    checksum_path = write_text_artifact(
        freeze_directory / "lock.sha256",
        f"{lock_sha256}\n",
    )
    for artifact in (
        raw_manifest,
        feature_manifest,
        model_path,
        policy_path,
        lock_path,
        checksum_path,
    ):
        artifact.chmod(0o444)
    return {
        "directory": freeze_directory,
        "lock_path": lock_path,
        "lock_sha256": lock_sha256,
        "model_sha256": model_sha256,
        "policy_sha256": policy_sha256,
        "holdout_identity": _holdout_identity(config),
    }


def _run_id(config: ExpectancyConfig, raw_snapshot: Snapshot) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{raw_snapshot.sha256[:8]}-{config.fingerprint[:8]}"


def _compact_fold_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key
        not in {
            "_ledger",
            "_stress_ledger",
            "_regression_observed",
            "_regression_predictions",
        }
    }


def _historical_classifier_control() -> dict[str, Any]:
    path = PACKAGE_ROOT / "reports" / "latest" / "development-benchmark.json"
    if not path.exists():
        raise RuntimeError("committed historical classifier report is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "role": "historical_control_only",
        "eligible_for_regression_candidate_selection": False,
        "path": path,
        "sha256": _sha256(path),
        "run_id": payload.get("run_id"),
        "selected": payload.get("selected"),
        "selected_aggregate": payload.get("selected_aggregate"),
        "gates": payload.get("gates"),
        "holdout_status": payload.get("holdout_status"),
        "scope": payload.get("scope"),
    }


def run_regression_development(
    config: ExpectancyConfig,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw_snapshot, feature_snapshots = prepare_expectancy_benchmark(
        config,
        refresh=refresh,
    )
    raw_manifest = json.loads(raw_snapshot.manifest_path.read_text(encoding="utf-8"))
    funding_coverage = raw_manifest["funding_coverage"]
    funding_provenance = raw_manifest["funding_provenance"]
    run_id = _run_id(config, raw_snapshot)
    run_directory = config.artifacts.root / "runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    fold_results = _run_fold_tasks(config, feature_snapshots)
    grouped = {
        candidate.candidate_id: [
            result
            for result in fold_results
            if result["candidate_id"] == candidate.candidate_id
        ]
        for candidate in config.candidates
    }
    aggregates = [
        aggregate_candidate_results(grouped[candidate.candidate_id], config)
        for candidate in config.candidates
    ]
    apply_oi_qualification(aggregates, fold_results, config)
    selected_aggregate, diagnostic_only = select_candidate(aggregates)
    selected_candidate = next(
        candidate
        for candidate in config.candidates
        if candidate.candidate_id == selected_aggregate["candidate_id"]
    )
    runtime = _runtime_metadata(config)
    code_lineage = _package_lineage()

    confirmation: dict[str, Any] = {"status": "not_run"}
    freeze: dict[str, Any] | None = None
    if not diagnostic_only:
        candidate_fold_results = grouped[selected_candidate.candidate_id]
        policy = _consensus_policy(candidate_fold_results)
        development_policy_counts = _development_policy_counts(candidate_fold_results)
        policy_preregistration_path = (
            run_directory / "final-confirmation-policy.json"
        )
        policy_preregistration = {
            "schema_version": 1,
            "candidate_id": selected_candidate.candidate_id,
            "source": (
                "modal non-no-trade policy from the selected candidate's six "
                "development folds"
            ),
            "tie_breakers": [
                "higher hurdle bps",
                "higher directional advantage bps",
            ],
            "policy": _regression_policy_payload(policy),
            "development_policy_counts": [
                {
                    "hurdle_bps": hurdle_bps,
                    "advantage_bps": advantage_bps,
                    "folds": count,
                }
                for (hurdle_bps, advantage_bps), count in sorted(
                    development_policy_counts.items(),
                    key=lambda item: (
                        -item[1],
                        -item[0][0],
                        -item[0][1],
                    ),
                )
            ],
        }
        write_json_artifact(
            policy_preregistration_path,
            policy_preregistration,
        )
        policy_preregistration_path.chmod(0o444)
        policy_preregistration = {
            **policy_preregistration,
            "path": policy_preregistration_path,
            "sha256": _sha256(policy_preregistration_path),
        }
        confirmation, fitted, policy = _final_confirmation(
            config,
            selected_candidate,
            feature_snapshots[selected_candidate.horizon_bars],
            policy,
            policy_preregistration,
        )
        if confirmation["gates"]["pass"]:
            freeze = _freeze_candidate(
                config=config,
                raw_snapshot=raw_snapshot,
                feature_snapshot=feature_snapshots[selected_candidate.horizon_bars],
                run_directory=run_directory,
                candidate=selected_candidate,
                fitted=fitted,
                policy=policy,
                confirmation=confirmation,
                development_gates_payload=selected_aggregate["gates"],
                code_lineage=code_lineage,
                runtime=runtime,
            )

    if freeze is not None:
        verdict = "qualified_for_holdout"
        holdout_status = "sealed_ready"
        holdout_reason = "development and final confirmation gates passed; candidate frozen"
    elif diagnostic_only:
        verdict = "no_development_candidate_qualified"
        holdout_status = "sealed_not_qualified"
        holdout_reason = "no candidate passed every development gate"
    else:
        verdict = "final_confirmation_failed"
        holdout_status = "sealed_not_qualified"
        holdout_reason = "selected development candidate failed final pre-holdout confirmation"

    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "direct_net_expectancy_regression",
        "run_id": run_id,
        "generated_at": datetime.now(UTC),
        "verdict": verdict,
        "objective": (
            "Qualify a reproducible PF_XBTUSD net-expectancy edge using fixed "
            "classical regression candidates."
        ),
        "scope": {
            "symbol": config.dataset.symbol,
            "interval_seconds": config.dataset.interval_seconds,
            "source_start": config.dataset.start,
            "source_end": config.dataset.end,
            "development_start": config.validation.folds[0].start,
            "development_end": config.validation.folds[-1].end,
            "holdout_start": config.validation.holdout_start,
            "holdout_end": config.validation.holdout_end,
        },
        "candidate_registry": [
            {
                "candidate_id": candidate.candidate_id,
                "horizon_bars": candidate.horizon_bars,
                "model": candidate.model,
                "feature_set": candidate.feature_set,
                "feature_names": REGRESSION_FEATURE_SETS[candidate.feature_set],
                "side_targets": ["long_net_bps", "short_net_bps"],
            }
            for candidate in config.candidates
        ],
        "funding_readiness": {
            "pass": True,
            "required": "complete first-party funding coverage for every source bucket",
            "coverage": funding_coverage,
            "provenance_verified": True,
            "pinned_import_id": config.funding_provenance.import_id,
            "provenance": funding_provenance,
            "funding_used_as_feature": False,
            "funding_used_in_targets_and_ledgers": True,
        },
        "execution_realism": {
            "status": "preliminary_research_assumption",
            "feature_available_at_equals_entry_at": True,
            "inference_and_routing_latency_seconds": 0,
            "entry_price": "next completed 15-minute candle open",
            "deployable_edge_claim_allowed": False,
            "required_follow_up": (
                "delayed-entry sensitivity using 1-minute or L2 data before any "
                "live-capital interpretation"
            ),
        },
        "fee_assumptions": {
            "qualifying_schedule": "taker_10_bps",
            "taker_bps_per_side": config.fees.taker_bps_per_side,
            "taker_round_trip_bps": config.fees.taker_bps_per_side * 2.0,
            "maker_bps_per_side": config.fees.maker_bps_per_side,
            "diagnostic_round_trip_fees_bps": FEE_COUNTERFACTUALS_BPS,
            "counterfactual_actions_reselected": False,
            "execution_cost_stress_multiplier": (
                config.gates.execution_cost_stress_multiplier
            ),
        },
        "historical_classifier_control": _historical_classifier_control(),
        "hashes": {
            "config_sha256": config.fingerprint,
            "source_sha256": raw_snapshot.sha256,
            "source_manifest_sha256": _sha256(raw_snapshot.manifest_path),
            "feature_sha256": {
                str(horizon): snapshot.sha256
                for horizon, snapshot in feature_snapshots.items()
            },
            "code_sha256": code_lineage["package_source_sha256"],
            "requirements_sha256": code_lineage["requirements_lock_sha256"],
            "runtime_sha256": _runtime_fingerprint(runtime),
            "funding_provenance_binding_sha256": funding_provenance[
                "binding_sha256"
            ],
        },
        "source_snapshot": {
            "path": raw_snapshot.path,
            "rows": raw_snapshot.row_count,
            "first_timestamp": raw_snapshot.first_timestamp,
            "last_timestamp": raw_snapshot.last_timestamp,
            "manifest_path": raw_snapshot.manifest_path,
        },
        "feature_snapshots": {
            str(horizon): {
                "path": snapshot.path,
                "sha256": snapshot.sha256,
                "rows": snapshot.row_count,
                "manifest_path": snapshot.manifest_path,
            }
            for horizon, snapshot in feature_snapshots.items()
        },
        "runtime": runtime,
        "code_lineage": code_lineage,
        "execution": {
            "candidate_count": len(config.candidates),
            "fold_count": len(config.validation.folds),
            "candidate_fold_jobs": len(fold_results),
            "parallel_processes": min(config.compute.available_cores, len(fold_results)),
            "inner_threads_per_job": 1,
            "side_models_fit_sequentially": True,
        },
        "selection_protocol": {
            "primary": "highest median fold stressed expectancy",
            "tie_breakers": [
                "lower turnover",
                "simpler model: Ridge, HGB, ExtraTrees",
                "candidate id",
            ],
            "development_fold_policy": (
                "fixed 12-cell expected-return hurdle and directional-advantage grid "
                "on the chronological threshold slice"
            ),
            "final_confirmation_policy": (
                "modal non-no-trade policy across the selected candidate's six "
                "development folds, with ties resolved toward the higher hurdle "
                "then higher directional advantage; frozen before confirmation"
            ),
            "diagnostic_only_when_no_qualifier": True,
            "holdout_rows_used": 0,
        },
        "candidate_aggregates": aggregates,
        "fold_results": [_compact_fold_result(result) for result in fold_results],
        "selected": {
            **selected_aggregate,
            "diagnostic_only": diagnostic_only,
        },
        "final_confirmation": confirmation,
        "freeze": freeze,
        "holdout": {
            "status": holdout_status,
            "opened": False,
            "identity": _holdout_identity(config),
            "start": config.validation.holdout_start,
            "end": config.validation.holdout_end,
            "reason": holdout_reason,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "notes": [
            "Long and short net returns are modeled separately and calibrated on "
            "a later chronological slice.",
            "Funding is excluded from model features and included in every realized "
            "target and economic ledger.",
            "Only the taker-fee result qualifies an edge; hybrid, maker, and zero-fee "
            "results are diagnostics.",
            "The locked holdout remains unopened unless both development and final "
            "confirmation gates pass.",
            "This preliminary benchmark assumes zero inference/routing latency at "
            "the next 15-minute open; a pass is not a deployable-edge claim until "
            "delayed-entry sensitivity is evaluated with 1-minute or L2 data.",
        ],
    }
    write_json_artifact(run_directory / "regression-development.json", report)
    write_regression_development_report(run_directory / "reports", report)
    write_regression_development_report(_resolved_report_directory(config), report)
    return report


def _load_freeze(
    config: ExpectancyConfig,
    run_id: str,
) -> tuple[Path, dict[str, Any], FittedNetRegressors]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"invalid run id: {run_id}")
    run_directory = config.artifacts.root / "runs" / run_id
    freeze_directory = run_directory / "regression-freeze"
    lock_path = freeze_directory / "lock.json"
    checksum_path = freeze_directory / "lock.sha256"
    if not lock_path.exists() or not checksum_path.exists():
        raise RuntimeError(f"run has no qualified regression freeze: {run_id}")
    expected_lock_sha256 = checksum_path.read_text(encoding="utf-8").strip()
    if _sha256(lock_path) != expected_lock_sha256:
        raise RuntimeError("frozen regression lock checksum mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["config_fingerprint"] != config.fingerprint:
        raise RuntimeError("configuration changed after the regressors were frozen")
    if lock["holdout"]["identity"] != _holdout_identity(config):
        raise RuntimeError("frozen regression holdout identity mismatch")
    current_lineage = _package_lineage()
    for key in ("package_source_sha256", "requirements_lock_sha256"):
        if current_lineage[key] != lock["code_lineage"][key]:
            raise RuntimeError(f"frozen regression {key} mismatch")
    runtime = _runtime_metadata(config)
    if _runtime_fingerprint(runtime) != lock["runtime_sha256"]:
        raise RuntimeError("frozen regression runtime fingerprint mismatch")
    for artifact_name in ("raw_snapshot", "feature_snapshot"):
        artifact = lock[artifact_name]
        if _sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise RuntimeError(f"frozen regression {artifact_name} checksum mismatch")
        if _sha256(Path(artifact["manifest_path"])) != artifact["manifest_sha256"]:
            raise RuntimeError(f"frozen regression {artifact_name} manifest checksum mismatch")
    if _sha256(Path(lock["model"]["path"])) != lock["model"]["sha256"]:
        raise RuntimeError("frozen regression model checksum mismatch")
    if _sha256(Path(lock["policy_artifact"]["path"])) != lock["policy_artifact"]["sha256"]:
        raise RuntimeError("frozen regression policy checksum mismatch")
    confirmation_policy_artifact = lock["confirmation_policy_artifact"]
    if (
        _sha256(Path(confirmation_policy_artifact["path"]))
        != confirmation_policy_artifact["sha256"]
    ):
        raise RuntimeError(
            "frozen final confirmation policy preregistration checksum mismatch"
        )
    policy_payload = json.loads(Path(lock["policy_artifact"]["path"]).read_text(encoding="utf-8"))
    if policy_payload != lock["policy"]:
        raise RuntimeError("frozen regression policy differs from its lock")
    preregistered_policy = json.loads(
        Path(confirmation_policy_artifact["path"]).read_text(encoding="utf-8")
    )["policy"]
    if preregistered_policy != lock["policy"]:
        raise RuntimeError("frozen policy differs from its preregistration")
    fitted = joblib.load(Path(lock["model"]["path"]))
    if not isinstance(fitted, FittedNetRegressors):
        raise RuntimeError("frozen regression model has an unexpected type")
    if fitted.name != lock["candidate"]["model"]:
        raise RuntimeError("frozen regression model name differs from its lock")
    if list(fitted.feature_names) != lock["candidate"]["feature_names"]:
        raise RuntimeError("frozen regression feature names differ from its lock")
    return run_directory, lock, fitted


def _holdout_gates(
    economics: dict[str, Any],
    cost_stress: dict[str, Any],
    config: ExpectancyConfig,
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
                and economics["net_expectancy_bps"]
                >= config.gates.minimum_net_expectancy_bps
            ),
            "actual": economics["net_expectancy_bps"],
            "required": config.gates.minimum_net_expectancy_bps,
        },
        "positive_daily_block_bootstrap_lower": {
            "pass": (
                economics["bootstrap_95_lower_bps"] is not None
                and economics["bootstrap_95_lower_bps"] > 0.0
            ),
            "actual": economics["bootstrap_95_lower_bps"],
            "required": "> 0 bps/trade",
        },
        "minimum_profit_factor": {
            "pass": _profit_factor_pass(economics, config.gates.minimum_profit_factor),
            "actual": economics["profit_factor"],
            "required": config.gates.minimum_profit_factor,
        },
        "positive_calendar_month_fraction": {
            "pass": (
                economics["positive_month_fraction"]
                >= config.gates.minimum_positive_month_fraction
            ),
            "actual": economics["positive_month_fraction"],
            "required": config.gates.minimum_positive_month_fraction,
        },
        "positive_cost_stress_expectancy": {
            "pass": (
                cost_stress["net_expectancy_bps"] is not None
                and cost_stress["net_expectancy_bps"] > 0.0
            ),
            "actual": cost_stress["net_expectancy_bps"],
            "required": (
                f"> 0 bps/trade at "
                f"{config.gates.execution_cost_stress_multiplier:.2f}x execution cost"
            ),
        },
    }
    return {"pass": all(check["pass"] for check in checks.values()), "checks": checks}


def _write_holdout_predictions(
    path: Path,
    frame: pl.DataFrame,
    predictions: np.ndarray,
    actions: np.ndarray,
) -> str:
    output = frame.select(
        [
            "bucket_start",
            "feature_available_at",
            "entry_at",
            "label_exit_at",
            "long_net_bps",
            "short_net_bps",
            "gross_forward_bps",
            "long_market_execution_cost_bps",
            "short_market_execution_cost_bps",
            "fee_cost_bps",
            "funding_horizon_bps",
        ]
    ).with_columns(
        [
            pl.Series("predicted_long_net_bps", predictions[:, 0]),
            pl.Series("predicted_short_net_bps", predictions[:, 1]),
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


def evaluate_regression_holdout(
    config: ExpectancyConfig,
    *,
    run_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_directory, lock, fitted = _load_freeze(config, run_id)
    result_path = run_directory / "regression-holdout.json"
    if result_path.exists():
        raise RuntimeError(f"regression holdout was already evaluated for {run_id}")
    marker, global_marker = _claim_holdout(config, run_directory, run_id)
    try:
        frame = _scan_features(
            Path(lock["feature_snapshot"]["path"]),
            start=config.validation.holdout_start,
            end=config.validation.holdout_end,
        ).filter(pl.col("label_exit_at") < config.validation.holdout_end)
        if frame.is_empty():
            raise RuntimeError("locked regression holdout contains no observations")
        if frame["bucket_start"].min() < config.validation.holdout_start:
            raise RuntimeError("regression holdout filter included a pre-holdout row")
        predictions = fitted.predict(frame)
        policy = RegressionPolicy(
            hurdle_bps=float(lock["policy"]["hurdle_bps"]),
            advantage_bps=float(lock["policy"]["advantage_bps"]),
            no_trade=bool(lock["policy"]["no_trade"]),
        )
        actions = regression_policy_actions(predictions, policy)
        economics = economic_metrics_for_actions(
            frame,
            actions,
            execution_cost_multiplier=1.0,
            bootstrap_repetitions=config.compute.holdout_bootstrap_resamples,
            seed=config.compute.random_seed + 110_000,
        )
        cost_stress = economic_metrics_for_actions(
            frame,
            actions,
            execution_cost_multiplier=config.gates.execution_cost_stress_multiplier,
            bootstrap_repetitions=config.compute.holdout_bootstrap_resamples,
            seed=config.compute.random_seed + 120_000,
        )
        gates = _holdout_gates(economics, cost_stress, config)
        prediction_path = run_directory / "regression-holdout" / "predictions.parquet"
        prediction_sha256 = _write_holdout_predictions(
            prediction_path,
            frame,
            predictions,
            actions,
        )
        report = {
            "schema_version": 1,
            "benchmark": "direct_net_expectancy_regression",
            "run_id": run_id,
            "generated_at": datetime.now(UTC),
            "verdict": (
                "qualified_positive_net_edge"
                if gates["pass"]
                else "no_qualified_net_edge"
            ),
            "hashes": {
                **lock["hashes"],
                "lock_sha256": _sha256(
                    run_directory / "regression-freeze" / "lock.json"
                ),
                "predictions_sha256": prediction_sha256,
            },
            "selected": {
                **lock["candidate"],
                "policy": lock["policy"],
            },
            "scope": {
                "symbol": config.dataset.symbol,
                "interval_seconds": config.dataset.interval_seconds,
                "horizon_bars": lock["candidate"]["horizon_bars"],
                "start": config.validation.holdout_start,
                "end": config.validation.holdout_end,
                "rows": frame.height,
            },
            "regression": regression_diagnostics(frame, predictions),
            "economics": economics,
            "cost_stress": cost_stress,
            "fee_counterfactuals": fee_counterfactuals(
                frame,
                actions,
                execution_cost_multiplier=1.0,
                bootstrap_repetitions=config.compute.holdout_bootstrap_resamples,
                seed=config.compute.random_seed + 130_000,
            ),
            "gates": gates,
            "holdout_access_marker": marker,
            "global_holdout_access_marker": global_marker,
            "elapsed_seconds": time.perf_counter() - started,
            "notes": [
                "The globally sealed market/time-window holdout was evaluated once.",
                "The model, both side calibrators, feature list, and policy were "
                "loaded from the checksum-validated freeze.",
                "Only taker-fee economics determine the verdict.",
            ],
        }
        write_json_artifact(result_path, report)
        opened_at = json.loads(marker.read_text(encoding="utf-8"))["opened_at"]
        completion = {
            "run_id": run_id,
            "holdout_identity": _holdout_identity(config),
            "status": "consumed",
            "opened_at": opened_at,
            "completed_at": datetime.now(UTC),
            "result_path": result_path,
        }
        write_json_artifact(marker, completion)
        write_json_artifact(global_marker, completion)
        write_regression_holdout_report(run_directory / "reports", report)
        write_regression_holdout_report(_resolved_report_directory(config), report)
        return report
    except Exception as error:
        opened_at = json.loads(marker.read_text(encoding="utf-8"))["opened_at"]
        failure = {
            "run_id": run_id,
            "holdout_identity": _holdout_identity(config),
            "status": "failed_and_sealed",
            "opened_at": opened_at,
            "failed_at": datetime.now(UTC),
            "error": f"{type(error).__name__}: {error}",
        }
        write_json_artifact(marker, failure)
        write_json_artifact(global_marker, failure)
        raise
