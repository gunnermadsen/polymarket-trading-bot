from __future__ import annotations

import itertools
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .core_evaluation import classification_metrics, first_crossing_timing
from .core_extract import file_sha256, write_json_atomic
from .persistence_benchmark import SAVED_POLICY_PROBABILITY_SCHEMA_VERSION
from .policy_config import (
    PolicyAdvancementGates,
    PolicyThresholdBand,
    SavedPolicyBenchmarkConfig,
    policy_config_to_dict,
)
from .provenance import runtime_provenance

SAVED_POLICY_BENCHMARK_SCHEMA_VERSION = "btc-saved-policy-benchmark-v1"


@dataclass(frozen=True)
class TimeBandPolicySelection:
    thresholds: tuple[tuple[str, float], ...]
    qualified: bool
    combinations_evaluated: int
    qualifying_combinations: int
    metrics: dict[str, Any]
    timing: dict[str, Any]
    checks: tuple[dict[str, Any], ...]

    def threshold_map(self) -> dict[str, float]:
        return dict(self.thresholds)


def run_saved_policy_benchmark(
    config: SavedPolicyBenchmarkConfig,
) -> tuple[Path, dict[str, Any]]:
    manifest = load_saved_probability_manifest(config.probability_manifest)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    candidate_results: dict[str, dict[str, Any]] = {}
    selected_frames: dict[str, pl.DataFrame] = {}
    for candidate_name in config.candidate_names:
        result, selected = evaluate_saved_policy_candidate(
            config,
            manifest,
            candidate_name,
        )
        candidate_results[candidate_name] = result
        selected_frames[candidate_name] = selected

    control = candidate_results[config.control_candidate]
    for candidate_name in config.candidate_names:
        result = candidate_results[candidate_name]
        result["advance"] = policy_advancement_checks(
            result,
            control,
            config.gates,
            is_control=candidate_name == config.control_candidate,
        )
        selected_path = run_dir / f"{candidate_name}-validation-policy.parquet"
        _write_parquet_atomic(selected_frames[candidate_name], selected_path)
        result["selected_validation_evidence"] = {
            "path": selected_path.name,
            "sha256": file_sha256(selected_path),
            "rows": selected_frames[candidate_name].height,
            "markets": selected_frames[candidate_name]["market_id"].n_unique(),
        }

    passing = [
        name
        for name in config.candidate_names
        if name != config.control_candidate
        and candidate_results[name]["advance"]["benchmark_passed"]
    ]
    winner = (
        max(
            passing,
            key=lambda name: _candidate_rank(candidate_results[name]),
        )
        if passing
        else None
    )
    benchmark = {
        "schema_version": SAVED_POLICY_BENCHMARK_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": policy_config_to_dict(config),
        "evaluation_note": config.evaluation_note,
        "evaluation_is_independent": config.evaluation_is_independent,
        "probability_evidence": {
            "manifest": str(config.probability_manifest),
            "manifest_sha256": file_sha256(config.probability_manifest),
            "schema_version": manifest["schema_version"],
            "created_at": manifest.get("created_at"),
            "source_benchmark_profile": manifest.get("source_benchmark_profile"),
            "source_config": manifest.get("source_config"),
            "source_config_sha256": manifest.get("source_config_sha256"),
            "control_candidate": manifest.get("control_candidate"),
            "candidate_names": manifest.get("candidate_names"),
            "fold_count": manifest["fold_count"],
            "causal_contract": manifest.get("causal_contract"),
            "checksums_verified": True,
            "causal_contract_verified": True,
        },
        "runtime_provenance": runtime_provenance(config.package_root),
        "control_candidate": config.control_candidate,
        "candidates": candidate_results,
        "benchmark_passed_candidates": passing,
        "winner": winner,
        "deployment": {
            "status": "not_authorized",
            "runtime_exported": False,
            "runtime_changed": False,
            "scope": "offline training and saved-probability policy evidence only",
        },
    }
    write_json_atomic(run_dir / "benchmark.json", benchmark)
    from .policy_report import generate_saved_policy_report

    generate_saved_policy_report(benchmark, run_dir / "report.html")
    return run_dir, benchmark


def load_saved_probability_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SAVED_POLICY_PROBABILITY_SCHEMA_VERSION:
        raise ValueError("saved probability manifest schema is unsupported")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("saved probability manifest has no candidates")
    if int(manifest.get("fold_count", 0)) <= 0:
        raise ValueError("saved probability manifest has no folds")
    return manifest


def evaluate_saved_policy_candidate(
    config: SavedPolicyBenchmarkConfig,
    manifest: dict[str, Any],
    candidate_name: str,
) -> tuple[dict[str, Any], pl.DataFrame]:
    candidate = manifest["candidates"].get(candidate_name)
    if candidate is None:
        raise ValueError(f"saved probability manifest is missing {candidate_name}")
    fold_results: list[dict[str, Any]] = []
    selected_frames = []
    for fold_record in candidate["folds"]:
        fold_index = int(fold_record["fold_index"])
        policy_rows = load_probability_evidence(
            config.probability_manifest,
            fold_record["policy_selection"],
            candidate_name,
            fold_index,
            "policy_selection",
        )
        validation_rows = load_probability_evidence(
            config.probability_manifest,
            fold_record["validation"],
            candidate_name,
            fold_index,
            "validation",
        )
        policy_end = policy_rows["window_start"].max()
        validation_start = validation_rows["window_start"].min()
        if policy_end >= validation_start:
            raise RuntimeError(f"{candidate_name} fold {fold_index} violates causal ordering")
        selection = select_causal_time_band_thresholds(policy_rows, config)
        validation_scored = apply_time_band_policy(
            validation_rows,
            config.bands,
            selection.threshold_map(),
        )
        selected = validation_scored.filter(pl.col("policy_selected"))
        eligible_markets = validation_rows["market_id"].n_unique()
        metrics = classification_metrics(
            selected,
            eligible_markets=eligible_markets,
        )
        timing = first_crossing_timing(
            selected,
            eligible_markets=eligible_markets,
        )
        validation_checks = absolute_policy_checks(
            metrics,
            timing,
            config.gates,
        )
        validation_qualified = all(check["passed"] for check in validation_checks)
        selected_frames.append(
            selected.with_columns(pl.lit(fold_index).cast(pl.Int32).alias("fold_index"))
        )
        fold_results.append(
            {
                "fold_index": fold_index,
                "causal_order_verified": True,
                "policy_selection_range": {
                    "start": policy_rows["window_start"].min().isoformat(),
                    "end": policy_end.isoformat(),
                },
                "validation_range": {
                    "start": validation_start.isoformat(),
                    "end": validation_rows["window_start"].max().isoformat(),
                },
                "policy_selection": {
                    **asdict(selection),
                    "thresholds": dict(selection.thresholds),
                },
                "validation": {
                    "metrics": metrics,
                    "timing": timing,
                    "checks": validation_checks,
                    "qualified": validation_qualified,
                    "thresholds_frozen_before_access": True,
                    "threshold_search_performed": False,
                },
            }
        )
    selected_all = pl.concat(
        selected_frames,
        how="vertical_relaxed",
    ).sort(["observed_at", "market_id", "fold_index"])
    eligible_total = sum(
        int(fold["validation"]["metrics"]["eligible_markets"]) for fold in fold_results
    )
    metrics = classification_metrics(
        selected_all,
        eligible_markets=eligible_total,
    )
    timing = first_crossing_timing(
        selected_all,
        eligible_markets=eligible_total,
    )
    policy_qualified_folds = sum(
        bool(fold["policy_selection"]["qualified"]) for fold in fold_results
    )
    validation_qualified_folds = sum(bool(fold["validation"]["qualified"]) for fold in fold_results)
    result = {
        "candidate": candidate_name,
        "target_kind": candidate["target_kind"],
        "feature_kind": candidate["feature_kind"],
        "calibration_kind": candidate["calibration_kind"],
        "row_weight_schedule": candidate["row_weight_schedule"],
        "folds": fold_results,
        "fold_count": len(fold_results),
        "policy_qualified_folds": policy_qualified_folds,
        "validation_qualified_folds": validation_qualified_folds,
        "out_of_fold": metrics,
        "timing": timing,
        "no_trade_rate": 1.0 - metrics["coverage"],
        "validation_threshold_searches": 0,
        "validation_score_passes": len(fold_results),
    }
    return result, selected_all


def load_probability_evidence(
    manifest_path: Path,
    record: dict[str, Any],
    candidate_name: str,
    fold_index: int,
    role: str,
) -> pl.DataFrame:
    path = manifest_path.parent / str(record["path"])
    if not path.is_file():
        raise RuntimeError(f"{role} probability evidence is missing: {path}")
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"{role} probability evidence checksum changed")
    frame = pl.read_parquet(path)
    if frame.height != int(record["rows"]):
        raise RuntimeError(f"{role} probability evidence row count changed")
    if frame["market_id"].n_unique() != int(record["markets"]):
        raise RuntimeError(f"{role} probability evidence market count changed")
    if set(frame["candidate"].unique().to_list()) != {candidate_name}:
        raise RuntimeError(f"{role} probability candidate identity changed")
    if set(frame["fold_index"].unique().to_list()) != {fold_index}:
        raise RuntimeError(f"{role} probability fold identity changed")
    required = {
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "probability_up",
        "predicted_up",
        "confidence",
        "correct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{role} probability evidence is missing columns: " + ", ".join(missing))
    probability = frame["probability_up"].to_numpy()
    if not np.isfinite(probability).all() or ((probability < 0.0) | (probability > 1.0)).any():
        raise RuntimeError(f"{role} probability evidence is invalid")
    if (
        frame.select(
            "market_id",
            "observed_at",
            "seconds_elapsed",
        )
        .is_duplicated()
        .any()
    ):
        raise RuntimeError(f"{role} probability evidence contains duplicates")
    return frame


def select_causal_time_band_thresholds(
    policy_rows: pl.DataFrame,
    config: SavedPolicyBenchmarkConfig,
    *,
    include_timing_gates: bool = True,
) -> TimeBandPolicySelection:
    eligible_markets = policy_rows["market_id"].n_unique()
    cache = _build_selection_cache(
        policy_rows,
        config.bands,
        config.threshold_candidates,
    )
    best: tuple[tuple[Any, ...], TimeBandPolicySelection] | None = None
    qualifying_combinations = 0
    combinations_evaluated = 0
    for values in itertools.product(
        config.threshold_candidates,
        repeat=len(config.bands),
    ):
        combinations_evaluated += 1
        selected = cache.selected_rows(values)
        metrics = classification_metrics(
            selected,
            eligible_markets=eligible_markets,
        )
        timing = first_crossing_timing(
            selected,
            eligible_markets=eligible_markets,
        )
        checks = absolute_policy_checks(
            metrics,
            timing,
            config.gates,
            include_timing=include_timing_gates,
        )
        qualified = all(check["passed"] for check in checks)
        qualifying_combinations += int(qualified)
        selection = TimeBandPolicySelection(
            thresholds=tuple(
                (band.name, float(value)) for band, value in zip(config.bands, values, strict=True)
            ),
            qualified=qualified,
            combinations_evaluated=0,
            qualifying_combinations=0,
            metrics=metrics,
            timing=timing,
            checks=tuple(checks),
        )
        rank = _policy_selection_rank(
            selection,
            include_timing=include_timing_gates,
        )
        if best is None or rank > best[0]:
            best = (rank, selection)
    if best is None:
        raise RuntimeError("policy threshold matrix is empty")
    selected = best[1]
    return TimeBandPolicySelection(
        thresholds=selected.thresholds,
        qualified=selected.qualified,
        combinations_evaluated=combinations_evaluated,
        qualifying_combinations=qualifying_combinations,
        metrics=selected.metrics,
        timing=selected.timing,
        checks=selected.checks,
    )


def apply_time_band_policy(
    probability_rows: pl.DataFrame,
    bands: tuple[PolicyThresholdBand, ...],
    thresholds: dict[str, float],
) -> pl.DataFrame:
    if set(thresholds) != {band.name for band in bands}:
        raise ValueError("policy thresholds do not match the causal bands")
    threshold_expression: pl.Expr | None = None
    band_expression: pl.Expr | None = None
    for band in bands:
        in_band = pl.col("seconds_elapsed").is_between(
            band.start_second,
            band.end_second_exclusive,
            closed="left",
        )
        threshold_expression = (
            pl.when(in_band).then(pl.lit(float(thresholds[band.name])))
            if threshold_expression is None
            else threshold_expression.when(in_band).then(pl.lit(float(thresholds[band.name])))
        )
        band_expression = (
            pl.when(in_band).then(pl.lit(band.name))
            if band_expression is None
            else band_expression.when(in_band).then(pl.lit(band.name))
        )
    if threshold_expression is None or band_expression is None:
        raise ValueError("at least one policy band is required")
    scored = probability_rows.with_columns(
        threshold_expression.otherwise(None).alias("selected_confidence_threshold"),
        band_expression.otherwise(None).alias("policy_threshold_band"),
    )
    if scored["selected_confidence_threshold"].null_count():
        raise RuntimeError("policy bands do not cover every probability row")
    keys = [
        "candidate",
        "fold_index",
        "market_id",
        "observed_at",
        "seconds_elapsed",
    ]
    selected_keys = (
        scored.filter(pl.col("confidence") >= pl.col("selected_confidence_threshold"))
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .group_by("market_id", maintain_order=True)
        .first()
        .select(keys)
        .with_columns(pl.lit(True).alias("policy_selected"))
    )
    return (
        scored.join(
            selected_keys,
            on=keys,
            how="left",
            validate="1:1",
        )
        .with_columns(pl.col("policy_selected").fill_null(False))
        .sort(["market_id", "seconds_elapsed", "observed_at"])
    )


class _SelectionCache:
    def __init__(
        self,
        frame: pl.DataFrame,
        bands: tuple[PolicyThresholdBand, ...],
        thresholds: tuple[float, ...],
        indexes: dict[tuple[str, float], np.ndarray],
    ) -> None:
        self.frame = frame
        self.bands = bands
        self.thresholds = thresholds
        self.indexes = indexes

    def selected_rows(self, values: tuple[float, ...]) -> pl.DataFrame:
        selected = np.full(
            self.frame["market_id"].n_unique(),
            -1,
            dtype=np.int64,
        )
        for band, threshold in zip(self.bands, values, strict=True):
            indexes = self.indexes[(band.name, float(threshold))]
            available = (selected < 0) & (indexes >= 0)
            selected[available] = indexes[available]
        row_indexes = selected[selected >= 0]
        if row_indexes.size == 0:
            return self.frame.head(0).drop("_policy_row_index")
        return self.frame[row_indexes.tolist()].drop("_policy_row_index")


def _build_selection_cache(
    rows: pl.DataFrame,
    bands: tuple[PolicyThresholdBand, ...],
    thresholds: tuple[float, ...],
) -> _SelectionCache:
    frame = rows.sort(["market_id", "seconds_elapsed", "observed_at"]).with_row_index(
        "_policy_row_index"
    )
    markets = frame["market_id"].unique(maintain_order=True).to_list()
    market_indexes = {market_id: index for index, market_id in enumerate(markets)}
    indexes: dict[tuple[str, float], np.ndarray] = {}
    for band in bands:
        band_rows = frame.filter(
            pl.col("seconds_elapsed").is_between(
                band.start_second,
                band.end_second_exclusive,
                closed="left",
            )
        )
        if band_rows.is_empty():
            raise RuntimeError(f"policy band {band.name} has no rows")
        for threshold in thresholds:
            first = (
                band_rows.filter(pl.col("confidence") >= threshold)
                .group_by("market_id", maintain_order=True)
                .first()
                .select("market_id", "_policy_row_index")
            )
            selected = np.full(len(markets), -1, dtype=np.int64)
            for market_id, row_index in first.iter_rows():
                selected[market_indexes[market_id]] = int(row_index)
            indexes[(band.name, float(threshold))] = selected
    return _SelectionCache(frame, bands, thresholds, indexes)


def absolute_policy_checks(
    metrics: dict[str, Any],
    timing: dict[str, Any],
    gates: PolicyAdvancementGates,
    *,
    include_timing: bool = True,
) -> list[dict[str, Any]]:
    ece = metrics["expected_calibration_error"]
    checks = [
        _check(
            "minimum selected markets", metrics["markets"], ">=", gates.minimum_selected_markets
        ),
        _check("minimum accuracy", metrics["accuracy"], ">=", gates.minimum_accuracy),
        _check(
            "minimum balanced accuracy",
            metrics["balanced_accuracy"],
            ">=",
            gates.minimum_balanced_accuracy,
        ),
        _check("minimum UP recall", metrics["up_recall"], ">=", gates.minimum_direction_recall),
        _check(
            "minimum DOWN recall",
            metrics["down_recall"],
            ">=",
            gates.minimum_direction_recall,
        ),
        _check(
            "minimum Wilson lower bound",
            metrics["wilson_lower_95"],
            ">=",
            gates.minimum_wilson_lower_95,
        ),
        _check(
            "maximum expected calibration error",
            ece,
            "<=",
            gates.maximum_expected_calibration_error,
        ),
        _check("minimum coverage", metrics["coverage"], ">=", gates.minimum_coverage),
    ]
    if include_timing:
        checks.append(
            _check(
                "maximum median entry second",
                timing["median_first_crossing_seconds"],
                "<=",
                gates.maximum_median_entry_second,
            )
        )
    return checks


def policy_advancement_checks(
    candidate: dict[str, Any],
    control: dict[str, Any],
    gates: PolicyAdvancementGates,
    *,
    is_control: bool,
) -> dict[str, Any]:
    metrics = candidate["out_of_fold"]
    timing = candidate["timing"]
    checks = absolute_policy_checks(metrics, timing, gates)
    checks.extend(
        (
            _check(
                "policy threshold qualified in every fold",
                candidate["policy_qualified_folds"],
                "==",
                candidate["fold_count"],
            ),
            _check(
                "validation qualified in every fold",
                candidate["validation_qualified_folds"],
                "==",
                candidate["fold_count"],
            ),
        )
    )
    if not is_control:
        control_metrics = control["out_of_fold"]
        control_median = control["timing"]["median_first_crossing_seconds"]
        candidate_median = timing["median_first_crossing_seconds"]
        checks.extend(
            (
                _check(
                    "coverage improves over control",
                    metrics["coverage"] - control_metrics["coverage"],
                    ">=",
                    gates.minimum_coverage_uplift,
                ),
                _check(
                    "accuracy does not regress from control",
                    metrics["accuracy"] - control_metrics["accuracy"],
                    ">=",
                    gates.maximum_accuracy_regression,
                ),
                _check(
                    "balanced accuracy does not regress from control",
                    (metrics["balanced_accuracy"] - control_metrics["balanced_accuracy"]),
                    ">=",
                    gates.maximum_balanced_accuracy_regression,
                ),
                _check(
                    "UP recall does not regress from control",
                    metrics["up_recall"] - control_metrics["up_recall"],
                    ">=",
                    gates.maximum_direction_recall_regression,
                ),
                _check(
                    "DOWN recall does not regress from control",
                    metrics["down_recall"] - control_metrics["down_recall"],
                    ">=",
                    gates.maximum_direction_recall_regression,
                ),
                _check(
                    "median entry improves over control",
                    (
                        control_median - candidate_median
                        if control_median is not None and candidate_median is not None
                        else None
                    ),
                    ">=",
                    gates.minimum_median_entry_improvement_seconds,
                ),
            )
        )
    return {
        "checks": checks,
        "benchmark_passed": (not is_control and all(check["passed"] for check in checks)),
        "deployment_qualified": False,
    }


def _policy_selection_rank(
    selection: TimeBandPolicySelection,
    *,
    include_timing: bool = True,
) -> tuple[Any, ...]:
    metrics = selection.metrics
    timing = selection.timing
    median = timing["median_first_crossing_seconds"]
    ece = metrics["expected_calibration_error"]
    rank = (
        selection.qualified,
        sum(check["passed"] for check in selection.checks),
        metrics["coverage"],
        metrics["wilson_lower_95"],
        metrics["balanced_accuracy"],
        min(metrics["up_recall"], metrics["down_recall"]),
        metrics["accuracy"],
        -(ece if ece is not None else float("inf")),
    )
    if include_timing:
        rank = (
            rank[0],
            rank[1],
            rank[2],
            -(median if median is not None else float("inf")),
            *rank[3:],
        )
    return (*rank, tuple(value for _, value in selection.thresholds))


def _candidate_rank(result: dict[str, Any]) -> tuple[Any, ...]:
    metrics = result["out_of_fold"]
    timing = result["timing"]
    median = timing["median_first_crossing_seconds"]
    return (
        metrics["coverage"],
        -(median if median is not None else float("inf")),
        metrics["wilson_lower_95"],
        metrics["balanced_accuracy"],
        metrics["accuracy"],
    )


def _check(
    name: str,
    observed: float | None,
    operator: str,
    required: float,
) -> dict[str, Any]:
    if observed is None:
        passed = False
    elif operator == ">=":
        passed = observed >= required
    elif operator == "<=":
        passed = observed <= required
    elif operator == "==":
        passed = observed == required
    else:
        raise ValueError(f"unsupported policy gate operator: {operator}")
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "required": required,
        "passed": bool(passed),
    }


def _write_parquet_atomic(frame: pl.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    frame.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )
    os.replace(temporary, destination)
