"""Fixed paper benchmark for spot-L2 and closed Chainlink candle features."""

from __future__ import annotations

import gc
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .core_evaluation import FIRST_CROSSING_TIME_BANDS
from .core_extract import file_sha256, write_json_exclusive
from .runtime_export import score_runtime_model
from .spot_l2_chainlink_cache import FeatureCache, load_or_build_feature_cache
from .spot_l2_chainlink_config import (
    CANDLES,
    COMBINED,
    CONTROL,
    SPOT_L2,
    ExecutionScenario,
    SpotL2ChainlinkConfig,
    fixed_datetime,
)
from .spot_l2_chainlink_evaluation import (
    EconomicLedgerSpec,
    build_economic_ledger,
    classification_summaries,
    economic_ledger_metrics,
    evaluate_advancement_gates,
    pair_economic_ledgers,
    paired_day_block_bootstrap,
    paired_economic_metrics,
)
from .spot_l2_chainlink_extract import SourceCache, load_or_extract_source_cache
from .spot_l2_chainlink_features import L2Normalizer

SCHEMA_VERSION = "btc-spot-l2-chainlink-candles-benchmark-v2"
COVERAGE_SCHEMA_VERSION = "btc-spot-l2-chainlink-coverage-v2"
RUN_MANIFEST_SCHEMA_VERSION = "btc-spot-l2-chainlink-run-manifest-v1"
POINT_KEYS = ("market_id", "window_start", "seconds_elapsed")
MARKET_KEYS = ("market_id", "window_start")
BASE_EXECUTION_SCENARIO = "arrival_150ms_depth_80pct"
EXPECTED_EXECUTION_SCENARIOS = (
    BASE_EXECUTION_SCENARIO,
    "latency_300ms_depth_65pct",
    "latency_600ms_depth_50pct",
)
ADVANCEMENT_AUTHORITIES = frozenset(
    {
        ("primary_l2", SPOT_L2),
        ("strict_combined", CANDLES),
        ("strict_combined", COMBINED),
    }
)


@dataclass(frozen=True)
class PlattBand:
    name: str
    start_second: int
    end_second_exclusive: int
    slope: float
    intercept: float
    converged: bool
    iterations: int


@dataclass
class PaperModel:
    """Training-only deterministic model artifact; never a runtime export."""

    profile: str
    features: tuple[str, ...]
    estimator: HistGradientBoostingClassifier
    calibrators: tuple[PlattBand, ...]
    normalizer: L2Normalizer | None

    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        scored = self.normalizer.transform(frame) if self.normalizer else frame
        matrix = scored.select(*self.features).to_numpy().astype(np.float64, copy=False)
        raw = np.clip(self.estimator.predict_proba(matrix)[:, 1], 1e-9, 1.0 - 1e-9)
        logits = np.log(raw / (1.0 - raw))
        elapsed = scored["seconds_elapsed"].to_numpy()
        probability = np.full(scored.height, np.nan, dtype=np.float64)
        for band in self.calibrators:
            mask = (elapsed >= band.start_second) & (elapsed < band.end_second_exclusive)
            probability[mask] = _sigmoid(logits[mask] * band.slope + band.intercept)
        if not np.isfinite(probability).all():
            raise RuntimeError(f"{self.profile} calibration does not cover every decision")
        return probability


@dataclass
class ProfileRun:
    evidence: dict[str, Any]
    policy_points: pl.DataFrame
    policy_first: pl.DataFrame
    evaluation_points: pl.DataFrame
    evaluation_first: pl.DataFrame


def run_spot_l2_chainlink_benchmark(
    config: SpotL2ChainlinkConfig,
) -> tuple[Path, dict[str, Any]]:
    """Extract, train, evaluate, and report the single fixed paper benchmark."""

    print("spot-l2 benchmark: validating clean implementation provenance", flush=True)
    git = _git_provenance(config)
    print("spot-l2 benchmark: loading immutable bounded source cache", flush=True)
    source_cache, source_manifest = load_or_extract_source_cache(config)
    print("spot-l2 benchmark: deriving causal matched feature cohorts", flush=True)
    feature_cache, feature_manifest = load_or_build_feature_cache(
        config,
        source_cache,
        source_manifest,
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    execution = _load_execution_evidence(source_cache)
    coverage = _coverage_manifest(
        config,
        source_manifest=source_manifest,
        feature_manifest=feature_manifest,
        feature_cache=feature_cache,
        execution=execution,
        artifact_root=run_dir,
    )
    coverage_path = run_dir / "coverage-manifest.json"
    write_json_exclusive(coverage_path, _json_safe(coverage))
    print(
        "spot-l2 benchmark: coverage fixed before training; "
        f"primary rows={coverage['overall']['l2']['decision_rows']:,}; "
        f"strict rows={coverage['overall']['strict']['decision_rows']:,}",
        flush=True,
    )

    champion_model = json.loads(config.champion_model.read_text())
    cohorts = (
        (
            "primary_l2",
            feature_cache.primary_l2_files,
            (CONTROL, SPOT_L2),
        ),
        (
            "strict_combined",
            feature_cache.strict_l2_candle_files,
            (CONTROL, SPOT_L2, CANDLES, COMBINED),
        ),
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "paper_only": True,
        "live_capital_allowed": False,
        "runtime_changed": False,
        "deployment_changed": False,
        "process_configuration_changed": False,
        "locked_evaluation_globally_pristine": False,
        "maximum_advancement": "forward_paper_validation",
        "implementation_git": git,
        "configuration": {
            "path": str(config.source_path),
            "sha256": file_sha256(config.source_path),
            "source_schema_revision": config.source_schema_revision,
            "windows": {
                name: fixed_datetime(name).isoformat()
                for name in (
                    "source_start",
                    "source_end",
                    "fit_start",
                    "fit_end",
                    "calibration_start",
                    "calibration_end",
                    "policy_start",
                    "policy_end",
                    "evaluation_start",
                    "evaluation_end",
                )
            },
            "confidence_threshold": config.confidence_threshold,
            "decision_seconds": list(config.decision_seconds),
            "model": {
                "learning_rate": config.learning_rate,
                "max_iter": config.max_iter,
                "max_leaf_nodes": config.max_leaf_nodes,
                "min_samples_leaf": config.min_samples_leaf,
                "l2_regularization": config.l2_regularization,
                "random_seed": config.random_seed,
                "threads_per_fit": config.threads_per_fit,
            },
            "bootstrap_resamples": config.bootstrap_resamples,
        },
        "lineage": {
            "source_manifest": str(config.cache / "source" / "manifest.json"),
            "source_manifest_sha256": file_sha256(config.cache / "source" / "manifest.json"),
            "feature_manifest": str(config.cache / "features" / "manifest.json"),
            "feature_manifest_sha256": file_sha256(config.cache / "features" / "manifest.json"),
            "coverage_manifest": coverage_path.name,
            "coverage_manifest_sha256": file_sha256(coverage_path),
            "frozen_champion_model": str(config.champion_model),
            "frozen_champion_model_sha256": file_sha256(config.champion_model),
            "frozen_champion_manifest_sha256": file_sha256(config.champion_manifest),
            "frozen_paper_process_sha256": file_sha256(config.champion_process),
        },
        "coverage": coverage,
        "cohorts": {},
        "advancement": {"selected_candidate": None, "comparisons": {}},
    }

    qualifying: list[str] = []
    for cohort_name, files, profiles in cohorts:
        print(f"spot-l2 benchmark: loading {cohort_name} cohort", flush=True)
        cohort_result, cohort_qualifying = _run_cohort(
            config,
            cohort_name=cohort_name,
            files=files,
            profiles=profiles,
            execution=execution,
            champion_model=champion_model,
            run_dir=run_dir,
        )
        result["cohorts"][cohort_name] = cohort_result
        for profile in cohort_qualifying:
            key = f"{cohort_name}:{profile}"
            qualifying.append(key)
            result["advancement"]["comparisons"][key] = cohort_result["comparisons"][profile][
                "advancement_gates"
            ]
        for profile, comparison in cohort_result["comparisons"].items():
            key = f"{cohort_name}:{profile}"
            result["advancement"]["comparisons"].setdefault(
                key,
                comparison["advancement_gates"],
            )
        gc.collect()

    stress_evidence_blocked = any(
        comparison["stress_gate_hard_failed"]
        for comparison in result["advancement"]["comparisons"].values()
    )
    result["advancement"]["execution_evidence_blocked"] = stress_evidence_blocked

    # Multiple qualifying challengers would require a ranking rule that this fixed
    # benchmark deliberately does not define. The prescribed stress exactness gate
    # currently prevents that ambiguity, but keep the outcome explicit.
    if len(qualifying) == 1:
        result["advancement"]["selected_candidate"] = qualifying[0]
        result["status"] = "pilot_challenger_eligible_for_forward_paper_validation"
    elif qualifying:
        result["status"] = "multiple_challengers_qualified_without_ranking_rule"
        result["advancement"]["selection_blocker"] = (
            "fixed benchmark defines advancement gates but no tie-breaking search"
        )
    elif stress_evidence_blocked:
        result["status"] = "champion_retained_execution_evidence_blocked"
    else:
        result["status"] = "champion_retained_no_challenger_cleared_fixed_gates"
    result["advancement"]["qualifying_comparisons"] = qualifying
    result["champion_unchanged"] = True

    evidence_path = run_dir / "benchmark-evidence.json"
    write_json_exclusive(evidence_path, _json_safe(result))
    report_path = run_dir / "benchmark-report.md"
    report_path.write_text(_markdown_report(result), encoding="utf-8")
    run_manifest = _write_run_manifest(config, run_dir, result)
    result["run_manifest"] = run_manifest
    print(
        f"spot-l2 benchmark: complete; status={result['status']}; report={report_path}",
        flush=True,
    )
    return run_dir, result


def _run_cohort(
    config: SpotL2ChainlinkConfig,
    *,
    cohort_name: str,
    files: tuple[Path, ...],
    profiles: tuple[str, ...],
    execution: pl.DataFrame,
    champion_model: dict[str, Any],
    run_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    splits = {
        name: (
            pl.scan_parquet(list(files))
            .filter(
                (pl.col("window_start") >= fixed_datetime(start))
                & (pl.col("window_start") < fixed_datetime(end))
            )
            .collect()
            .sort(list(POINT_KEYS))
        )
        for name, start, end in (
            ("fit", "fit_start", "fit_end"),
            ("calibration", "calibration_start", "calibration_end"),
            ("policy", "policy_start", "policy_end"),
            ("evaluation", "evaluation_start", "evaluation_end"),
        )
    }
    for name, split in splits.items():
        if split.is_empty():
            raise RuntimeError(f"{cohort_name} {name} split is empty")
        _validate_cohort_frame(split, config, f"{cohort_name} {name}")
    _assert_disjoint_markets(*splits.values())
    key_path = run_dir / f"{cohort_name}-decision-keys.parquet"
    all_keys = pl.concat(
        [split.select(*POINT_KEYS) for split in splits.values()],
        how="vertical",
    ).sort(list(POINT_KEYS))
    all_keys.write_parquet(key_path, compression="zstd")
    cohort: dict[str, Any] = {
        "profiles": {},
        "frozen_champion_external_reference": {},
        "comparisons": {},
        "identical_decision_keys_verified": True,
        "decision_keys_artifact": {
            "path": key_path.name,
            "sha256": file_sha256(key_path),
            "rows": all_keys.height,
            "markets": all_keys["market_id"].n_unique(),
        },
        "splits": {
            name: {"rows": part.height, "markets": part["market_id"].n_unique()}
            for name, part in splits.items()
        },
    }
    internal: dict[str, ProfileRun] = {}
    expected_policy_keys = splits["policy"].select(*POINT_KEYS).sort(list(POINT_KEYS))
    expected_evaluation_keys = splits["evaluation"].select(*POINT_KEYS).sort(list(POINT_KEYS))
    for profile in profiles:
        print(f"spot-l2 benchmark: fitting {cohort_name}/{profile}", flush=True)
        features = config.feature_sets[profile]
        model, training = _fit(
            config,
            profile,
            features,
            splits["fit"],
            splits["calibration"],
        )
        policy_probability = model.probability(splits["policy"])
        evaluation_probability = model.probability(splits["evaluation"])
        policy_points = _point_predictions(splits["policy"], policy_probability)
        evaluation_points = _point_predictions(splits["evaluation"], evaluation_probability)
        _assert_prediction_keys(policy_points, expected_policy_keys, profile, "policy")
        _assert_prediction_keys(
            evaluation_points,
            expected_evaluation_keys,
            profile,
            "evaluation",
        )
        policy_first = _first_crossings(splits["policy"], policy_probability)
        evaluation_first = _first_crossings(splits["evaluation"], evaluation_probability)

        prefix = f"{cohort_name}-{profile}"
        model_path = run_dir / f"{prefix}-training-model.joblib"
        joblib.dump(model, model_path, compress=3)
        _verify_model_artifact(model_path, model, splits["evaluation"])
        golden_path = run_dir / f"{prefix}-golden-vectors.json"
        write_json_exclusive(
            golden_path,
            _golden_vectors(splits["evaluation"], model, features),
        )
        artifacts = _write_prediction_artifacts(
            run_dir,
            prefix,
            policy_points=policy_points,
            policy_first=policy_first,
            evaluation_points=evaluation_points,
            evaluation_first=evaluation_first,
        )
        evidence = {
            "feature_count": len(features),
            "feature_names": list(features),
            "training": training,
            "policy_confirmation": {
                "decision_quality": classification_summaries(
                    policy_points,
                    eligible_rows=splits["policy"],
                ),
                "first_crossing_quality": classification_summaries(
                    policy_first,
                    eligible_rows=splits["policy"],
                ),
            },
            "locked_evaluation": {
                "decision_quality": classification_summaries(
                    evaluation_points,
                    eligible_rows=splits["evaluation"],
                ),
                "first_crossing_quality": classification_summaries(
                    evaluation_first,
                    eligible_rows=splits["evaluation"],
                ),
            },
            "artifact": {
                "path": model_path.name,
                "sha256": file_sha256(model_path),
                "paper_only": True,
                "runtime_exported": False,
                "golden_vectors": golden_path.name,
                "golden_vectors_sha256": file_sha256(golden_path),
                "predictions": artifacts,
            },
        }
        cohort["profiles"][profile] = evidence
        internal[profile] = ProfileRun(
            evidence=evidence,
            policy_points=policy_points,
            policy_first=policy_first,
            evaluation_points=evaluation_points,
            evaluation_first=evaluation_first,
        )

    print(f"spot-l2 benchmark: scoring frozen champion on {cohort_name}", flush=True)
    champion = _score_frozen_champion(
        config,
        cohort_name=cohort_name,
        champion_model=champion_model,
        policy=splits["policy"],
        evaluation=splits["evaluation"],
        execution=execution,
        run_dir=run_dir,
    )
    cohort["frozen_champion_external_reference"] = champion

    qualifying: list[str] = []
    for profile in profiles:
        if profile == CONTROL:
            continue
        print(f"spot-l2 benchmark: paired evaluation {cohort_name}/{profile}", flush=True)
        comparison = _paired_comparison(
            config,
            cohort_name=cohort_name,
            profile=profile,
            candidate=internal[profile],
            control=internal[CONTROL],
            eligible_rows=splits["evaluation"],
            execution=execution,
            run_dir=run_dir,
        )
        authority = _is_advancement_authority(cohort_name, profile)
        comparison["advancement_authority"] = authority
        cohort["comparisons"][profile] = comparison
        if authority and comparison["advancement_gates"]["eligible_for_forward_paper_validation"]:
            qualifying.append(profile)
    return cohort, qualifying


def _is_advancement_authority(cohort_name: str, profile: str) -> bool:
    return (cohort_name, profile) in ADVANCEMENT_AUTHORITIES


def _fit(
    config: SpotL2ChainlinkConfig,
    profile: str,
    features: tuple[str, ...],
    fit: pl.DataFrame,
    calibration: pl.DataFrame,
) -> tuple[PaperModel, dict[str, Any]]:
    _require_finite(fit, (*features, "label_up"), f"{profile} fit")
    _require_finite(
        calibration,
        (*features, "label_up"),
        f"{profile} calibration",
    )
    normalizer = L2Normalizer.fit(fit) if profile in (SPOT_L2, COMBINED) else None
    fit_ready = normalizer.transform(fit) if normalizer else fit
    calibration_ready = normalizer.transform(calibration) if normalizer else calibration
    weights = _market_equal_weights(fit_ready)
    parameters = {
        "learning_rate": config.learning_rate,
        "max_iter": config.max_iter,
        "max_leaf_nodes": config.max_leaf_nodes,
        "min_samples_leaf": config.min_samples_leaf,
        "l2_regularization": config.l2_regularization,
        "early_stopping": False,
        "random_state": config.random_seed,
    }
    estimator = HistGradientBoostingClassifier(**parameters)
    with threadpool_limits(limits=config.threads_per_fit):
        estimator.fit(
            fit_ready.select(*features).to_numpy(),
            fit_ready["label_up"].to_numpy(),
            sample_weight=weights,
        )
    raw = np.clip(
        estimator.predict_proba(calibration_ready.select(*features).to_numpy())[:, 1],
        1e-9,
        1.0 - 1e-9,
    )
    logits = np.log(raw / (1.0 - raw))
    elapsed = calibration_ready["seconds_elapsed"].to_numpy()
    labels = calibration_ready["label_up"].to_numpy()
    calibrators: list[PlattBand] = []
    calibration_evidence: list[dict[str, Any]] = []
    for name, start, end in FIRST_CROSSING_TIME_BANDS:
        mask = (elapsed >= start) & (elapsed < end)
        band = calibration_ready.filter(
            (pl.col("seconds_elapsed") >= start) & (pl.col("seconds_elapsed") < end)
        )
        band_labels = labels[mask]
        if band.height < 100 or set(np.unique(band_labels).tolist()) != {0, 1}:
            raise RuntimeError(f"{profile} calibration band {name} is unusable")
        band_weights = _market_equal_weights(band)
        calibrator = LogisticRegression(
            C=1_000_000,
            solver="lbfgs",
            max_iter=500,
            tol=1e-9,
            random_state=config.random_seed,
        )
        with threadpool_limits(limits=config.threads_per_fit):
            calibrator.fit(
                logits[mask].reshape(-1, 1),
                band_labels,
                sample_weight=band_weights,
            )
        fitted = PlattBand(
            name=name,
            start_second=start,
            end_second_exclusive=end,
            slope=float(calibrator.coef_[0, 0]),
            intercept=float(calibrator.intercept_[0]),
            converged=bool(calibrator.n_iter_[0] < calibrator.max_iter),
            iterations=int(calibrator.n_iter_[0]),
        )
        if not fitted.converged or fitted.slope <= 0.0:
            raise RuntimeError(f"{profile} calibration band {name} did not converge monotonically")
        calibrators.append(fitted)
        calibration_evidence.append(
            {
                **asdict(fitted),
                "rows": band.height,
                "markets": band["market_id"].n_unique(),
                "positive_rate": float(band["label_up"].mean()),
                "market_weight_total_min": _market_weight_totals(band, band_weights)[0],
                "market_weight_total_max": _market_weight_totals(band, band_weights)[1],
            }
        )
    model = PaperModel(
        profile=profile,
        features=features,
        estimator=estimator,
        calibrators=tuple(calibrators),
        normalizer=normalizer,
    )
    fit_weight_min, fit_weight_max = _market_weight_totals(fit_ready, weights)
    return model, {
        "fit": {"rows": fit.height, "markets": fit["market_id"].n_unique()},
        "calibration": {
            "rows": calibration.height,
            "markets": calibration["market_id"].n_unique(),
        },
        "hyperparameters": parameters,
        "random_seed": config.random_seed,
        "threads_per_fit": config.threads_per_fit,
        "market_equal_row_weighting": True,
        "fit_market_weight_total_min": fit_weight_min,
        "fit_market_weight_total_max": fit_weight_max,
        "l2_normalization": (
            {
                "fit_interval_only": True,
                "means": normalizer.means,
                "scales": normalizer.scales,
            }
            if normalizer
            else None
        ),
        "calibration_bands": calibration_evidence,
    }


def _paired_comparison(
    config: SpotL2ChainlinkConfig,
    *,
    cohort_name: str,
    profile: str,
    candidate: ProfileRun,
    control: ProfileRun,
    eligible_rows: pl.DataFrame,
    execution: pl.DataFrame,
    run_dir: Path,
) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    expected = tuple(scenario.key for scenario in config.execution_scenarios)
    if expected != EXPECTED_EXECUTION_SCENARIOS:
        raise RuntimeError("configured execution scenarios changed")
    for scenario in config.execution_scenarios:
        spec = _economic_spec(scenario)
        candidate_ledger = build_economic_ledger(
            candidate.evaluation_first,
            execution,
            eligible_markets=eligible_rows,
            profile=profile,
            spec=spec,
        )
        control_ledger = build_economic_ledger(
            control.evaluation_first,
            execution,
            eligible_markets=eligible_rows,
            profile=CONTROL,
            spec=spec,
        )
        paired = pair_economic_ledgers(candidate_ledger, control_ledger)
        metrics = paired_economic_metrics(paired)
        bootstrap = paired_day_block_bootstrap(
            paired,
            random_seed=config.random_seed,
            resamples=config.bootstrap_resamples,
        )
        path = run_dir / f"{cohort_name}-{profile}-{scenario.key}-paired-ledger.parquet"
        paired.write_parquet(path, compression="zstd")
        scenarios[scenario.key] = {
            "configuration": asdict(scenario),
            "metrics": metrics,
            "bootstrap": bootstrap,
            "paired_ledger": path.name,
            "paired_ledger_sha256": file_sha256(path),
        }
    base = scenarios[BASE_EXECUTION_SCENARIO]["metrics"]
    candidate_quality = candidate.evidence["locked_evaluation"]["first_crossing_quality"]
    control_quality = control.evidence["locked_evaluation"]["first_crossing_quality"]
    gates = evaluate_advancement_gates(
        candidate_classification=candidate_quality,
        control_classification=control_quality,
        paired_economics=base,
        stress_scenarios=scenarios,
        thresholds=config.advancement,
    )
    return {
        "matched_control": CONTROL,
        "candidate": profile,
        "identical_decision_keys_verified": True,
        "base_execution_scenario": BASE_EXECUTION_SCENARIO,
        "execution_scenarios": scenarios,
        "advancement_gates": gates,
    }


def _score_frozen_champion(
    config: SpotL2ChainlinkConfig,
    *,
    cohort_name: str,
    champion_model: dict[str, Any],
    policy: pl.DataFrame,
    evaluation: pl.DataFrame,
    execution: pl.DataFrame,
    run_dir: Path,
) -> dict[str, Any]:
    features = tuple(champion_model["features"]["names"])
    if features != config.base_features:
        raise RuntimeError("frozen champion feature schema no longer matches control")
    policy_probability = _runtime_probabilities(champion_model, policy, features)
    evaluation_probability = _runtime_probabilities(champion_model, evaluation, features)
    policy_points = _point_predictions(policy, policy_probability)
    evaluation_points = _point_predictions(evaluation, evaluation_probability)
    policy_first = _first_crossings(policy, policy_probability)
    evaluation_first = _first_crossings(evaluation, evaluation_probability)
    prefix = f"{cohort_name}-frozen-champion-external-reference"
    artifacts = _write_prediction_artifacts(
        run_dir,
        prefix,
        policy_points=policy_points,
        policy_first=policy_first,
        evaluation_points=evaluation_points,
        evaluation_first=evaluation_first,
    )
    economics: dict[str, Any] = {}
    for scenario in config.execution_scenarios:
        ledger = build_economic_ledger(
            evaluation_first,
            execution,
            eligible_markets=evaluation,
            profile="frozen_champion_external_reference",
            spec=_economic_spec(scenario),
        )
        path = run_dir / f"{prefix}-{scenario.key}-ledger.parquet"
        ledger.write_parquet(path, compression="zstd")
        economics[scenario.key] = {
            "metrics": economic_ledger_metrics(ledger),
            "ledger": path.name,
            "ledger_sha256": file_sha256(path),
        }
    return {
        "external_reference_only": True,
        "replaces_matched_control": False,
        "trained_on_matched_cohort": False,
        "model_key": champion_model["model_key"],
        "model_sha256": file_sha256(config.champion_model),
        "policy_confirmation": classification_summaries(
            policy_first,
            eligible_rows=policy,
        ),
        "locked_evaluation": classification_summaries(
            evaluation_first,
            eligible_rows=evaluation,
        ),
        "economics": economics,
        "prediction_artifacts": artifacts,
    }


def _economic_spec(scenario: ExecutionScenario) -> EconomicLedgerSpec:
    return EconomicLedgerSpec(
        scenario_key=scenario.key,
        quantity=5.0,
        up_price_column="up_ask_vwap_10",
        down_price_column="down_ask_vwap_10",
        eligibility_column="strict_both_side_eligible_10",
    )


def _coverage_manifest(
    config: SpotL2ChainlinkConfig,
    *,
    source_manifest: dict[str, Any],
    feature_manifest: dict[str, Any],
    feature_cache: FeatureCache,
    execution: pl.DataFrame,
    artifact_root: Path,
) -> dict[str, Any]:
    core = (
        pl.scan_parquet(list(feature_cache.core_files))
        .select(
            *POINT_KEYS,
            "btc_realized_volatility_60s_bps",
        )
        .collect()
    )
    l2 = pl.scan_parquet(list(feature_cache.primary_l2_files)).select(*POINT_KEYS).collect()
    candles = (
        pl.scan_parquet(list(feature_cache.candle_qualified_key_files))
        .select(*POINT_KEYS)
        .collect()
    )
    strict = (
        pl.scan_parquet(list(feature_cache.strict_l2_candle_files)).select(*POINT_KEYS).collect()
    )
    for label, candidate in (("l2", l2), ("candles", candles), ("strict", strict)):
        _assert_key_subset(candidate, core, label)
    joined = core
    for label, candidate in (("l2", l2), ("candles", candles), ("strict", strict)):
        joined = joined.join(
            candidate.with_columns(pl.lit(True).alias(label)),
            on=list(POINT_KEYS),
            how="left",
            validate="1:1",
        ).with_columns(pl.col(label).fill_null(False))
    fit_volatility = joined.filter(
        (pl.col("window_start") >= fixed_datetime("fit_start"))
        & (pl.col("window_start") < fixed_datetime("fit_end"))
    )["btc_realized_volatility_60s_bps"]
    low = float(fit_volatility.quantile(1.0 / 3.0, interpolation="linear"))
    high = float(fit_volatility.quantile(2.0 / 3.0, interpolation="linear"))
    grouped = joined.with_columns(
        pl.col("window_start").dt.date().cast(pl.String).alias("utc_day"),
        _split_expression().alias("split"),
        _band_expression().alias("decision_time_band"),
        pl.when(pl.col("btc_realized_volatility_60s_bps") <= low)
        .then(pl.lit("low"))
        .when(pl.col("btc_realized_volatility_60s_bps") <= high)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("high"))
        .alias("volatility_regime"),
    )
    overall = _coverage_bucket(grouped)
    by_day = _coverage_groups(grouped, "utc_day")
    zero_intervals = [
        {
            "start": f"{row['utc_day']}T00:00:00+00:00",
            "end": (date.fromisoformat(row["utc_day"]) + timedelta(days=1)).isoformat()
            + "T00:00:00+00:00",
            "reason": "no_qualified_l2_decision_rows",
        }
        for row in by_day
        if row["l2"]["decision_rows"] == 0
    ]
    excluded_decision_intervals = _excluded_decision_intervals(joined)
    excluded_decision_intervals_path = artifact_root / "excluded-decision-intervals.parquet"
    excluded_decision_intervals.write_parquet(
        excluded_decision_intervals_path,
        compression="zstd",
    )
    qualified_seconds = int(source_manifest["sources"]["l2"]["totals"]["qualified_seconds"])
    expected_seconds = int(
        (fixed_datetime("source_end") - fixed_datetime("source_start")).total_seconds()
    )
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "created_before_training": True,
        "source_interval": {
            "start": fixed_datetime("source_start").isoformat(),
            "end": fixed_datetime("source_end").isoformat(),
            "semantics": "half_open_utc",
        },
        "qualified_l2_seconds": {
            "observed": qualified_seconds,
            "possible": expected_seconds,
            "coverage": qualified_seconds / expected_seconds,
            "not_training_coverage": True,
        },
        "overall": overall,
        "by_utc_day": by_day,
        "by_model_split": _coverage_groups(grouped, "split"),
        "by_decision_time_band": _coverage_groups(grouped, "decision_time_band"),
        "by_fit_defined_volatility_regime": _coverage_groups(
            grouped,
            "volatility_regime",
        ),
        "volatility_regime_fit_only_cut_points_bps": {"low": low, "high": high},
        "excluded_intervals": zero_intervals,
        "excluded_decision_intervals": {
            "path": excluded_decision_intervals_path.name,
            "sha256": file_sha256(excluded_decision_intervals_path),
            "rows": excluded_decision_intervals.height,
            "semantics": (
                "consecutive unavailable points collapsed on the fixed five-second "
                "decision grid; end_at_exclusive is the next grid boundary"
            ),
        },
        "excluded_key_artifacts": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": pl.scan_parquet(path).select(pl.len()).collect().item(),
            }
            for path in feature_cache.excluded_key_files
        ],
        "execution_evidence": _execution_coverage(execution),
        "feature_cache_manifest": feature_manifest,
        "causality": {
            "l2_available_at_strictly_before_decision": True,
            "l2_source_event_timestamp_strictly_before_decision": True,
            "l2_maximum_age_seconds": 2,
            "l2_maximum_age_applies_to_source_event_timestamp": True,
            "l2_forward_fill": False,
            "l2_interpolation": False,
            "chainlink_fully_closed_strictly_before_decision": True,
            "model_receives_missingness_provider_lineage_or_quality_flags": False,
        },
    }


def _excluded_decision_intervals(joined: pl.DataFrame) -> pl.DataFrame:
    keys = list(POINT_KEYS)
    l2 = (
        joined.filter(~pl.col("l2"))
        .select(*keys)
        .with_columns(
            pl.lit("primary_l2").alias("excluded_from"),
            pl.lit("spot_l2_unavailable_or_older_than_2s").alias("reason"),
        )
    )
    candles = (
        joined.filter(~pl.col("candles"))
        .select(*keys)
        .with_columns(
            pl.lit("candle_qualified_keys").alias("excluded_from"),
            pl.lit("chainlink_candle_unavailable_or_incomplete").alias("reason"),
        )
    )
    strict = (
        joined.filter(~pl.col("strict"))
        .select(*keys, "l2", "candles")
        .with_columns(
            pl.lit("strict_l2_candle").alias("excluded_from"),
            pl.when(~pl.col("l2") & ~pl.col("candles"))
            .then(pl.lit("spot_l2_and_chainlink_unavailable"))
            .when(~pl.col("l2"))
            .then(pl.lit("spot_l2_unavailable_or_older_than_2s"))
            .otherwise(pl.lit("chainlink_candle_unavailable_or_incomplete"))
            .alias("reason"),
        )
        .drop("l2", "candles")
    )
    group_keys = ["market_id", "window_start", "excluded_from", "reason"]
    return (
        pl.concat((l2, candles, strict), how="vertical_relaxed")
        .sort(*group_keys, "seconds_elapsed")
        .with_columns(
            (pl.col("seconds_elapsed").diff().over(group_keys) != 5)
            .fill_null(True)
            .alias("_interval_start")
        )
        .with_columns(
            pl.col("_interval_start").cast(pl.Int64).cum_sum().over(group_keys).alias("_interval")
        )
        .group_by(*group_keys, "_interval", maintain_order=True)
        .agg(
            pl.col("seconds_elapsed").min().alias("start_second"),
            pl.col("seconds_elapsed").max().alias("end_second_inclusive"),
            pl.len().alias("decision_rows"),
        )
        .with_columns(
            (pl.col("window_start") + pl.duration(seconds=pl.col("start_second"))).alias(
                "start_at"
            ),
            (
                pl.col("window_start") + pl.duration(seconds=pl.col("end_second_inclusive") + 5)
            ).alias("end_at_exclusive"),
        )
        .drop("_interval")
        .sort("window_start", "market_id", "excluded_from", "start_second")
    )


def _execution_coverage(execution: pl.DataFrame) -> dict[str, Any]:
    def bucket(frame: pl.DataFrame) -> dict[str, Any]:
        snapshots = frame.filter(pl.col("snapshot_at").is_not_null())
        eligible = frame.filter(pl.col("strict_both_side_eligible_10"))
        return {
            "scenario_rows": frame.height,
            "decision_keys": frame.select(pl.struct(POINT_KEYS).n_unique()).item(),
            "snapshot_rows": snapshots.height,
            "snapshot_utc_days": (
                snapshots["window_start"].dt.date().n_unique() if snapshots.height else 0
            ),
            "strict_10_share_eligible_rows": eligible.height,
            "strict_10_share_eligible_utc_days": (
                eligible["window_start"].dt.date().n_unique() if eligible.height else 0
            ),
            "price_stress_methods": sorted(
                frame["price_stress_method"].drop_nulls().unique().to_list()
            ),
            "price_stress_exact_values": sorted(
                frame["price_stress_exact"].drop_nulls().unique().to_list()
            ),
        }

    snapshots = execution.filter(pl.col("snapshot_at").is_not_null())
    eligible = execution.filter(pl.col("strict_both_side_eligible_10"))
    return {
        "scenario_rows": execution.height,
        "decision_keys": execution.select(pl.struct(POINT_KEYS).n_unique()).item(),
        "snapshot_rows": snapshots.height,
        "snapshot_utc_days": (
            snapshots["window_start"].dt.date().n_unique() if snapshots.height else 0
        ),
        "strict_10_share_eligible_rows": eligible.height,
        "strict_10_share_eligible_utc_days": (
            eligible["window_start"].dt.date().n_unique() if eligible.height else 0
        ),
        "by_scenario": {
            key: bucket(execution.filter(pl.col("scenario_key") == key))
            for key in EXPECTED_EXECUTION_SCENARIOS
        },
    }


def _coverage_bucket(frame: pl.DataFrame) -> dict[str, Any]:
    total_rows = frame.height
    total_markets = frame["market_id"].n_unique() if total_rows else 0
    output: dict[str, Any] = {"core": {"decision_rows": total_rows, "markets": total_markets}}
    for name in ("l2", "candles", "strict"):
        rows = frame.filter(pl.col(name))
        decision_rows = rows.height
        markets = rows["market_id"].n_unique() if decision_rows else 0
        output[name] = {
            "decision_rows": decision_rows,
            "markets": markets,
            "decision_row_coverage": decision_rows / total_rows if total_rows else 0.0,
            "market_coverage": markets / total_markets if total_markets else 0.0,
        }
    return output


def _coverage_groups(frame: pl.DataFrame, column: str) -> list[dict[str, Any]]:
    groups = sorted(frame[column].unique().drop_nulls().to_list())
    return [
        {column: group, **_coverage_bucket(frame.filter(pl.col(column) == group))}
        for group in groups
    ]


def _split_expression() -> pl.Expr:
    return (
        pl.when(pl.col("window_start") < fixed_datetime("fit_end"))
        .then(pl.lit("fit"))
        .when(pl.col("window_start") < fixed_datetime("calibration_end"))
        .then(pl.lit("calibration"))
        .when(pl.col("window_start") < fixed_datetime("policy_end"))
        .then(pl.lit("policy"))
        .otherwise(pl.lit("locked_evaluation"))
    )


def _band_expression() -> pl.Expr:
    expression = pl.lit(None, dtype=pl.String)
    for name, start, end in reversed(FIRST_CROSSING_TIME_BANDS):
        expression = (
            pl.when((pl.col("seconds_elapsed") >= start) & (pl.col("seconds_elapsed") < end))
            .then(pl.lit(name))
            .otherwise(expression)
        )
    return expression


def _load_execution_evidence(source: SourceCache) -> pl.DataFrame:
    execution = (
        pl.scan_parquet(list(source.execution_files))
        .filter(
            (pl.col("window_start") >= fixed_datetime("evaluation_start"))
            & (pl.col("window_start") < fixed_datetime("evaluation_end"))
        )
        .collect()
    )
    required = (
        *POINT_KEYS,
        "decision_at",
        "scenario_key",
        "snapshot_at",
        "fee_rate",
        "up_ask_vwap_10",
        "down_ask_vwap_10",
        "strict_both_side_eligible_10",
        "price_stress_method",
        "price_stress_exact",
    )
    missing = sorted(set(required) - set(execution.columns))
    if missing:
        raise RuntimeError("execution evidence is missing columns: " + ", ".join(missing))
    observed = tuple(sorted(execution["scenario_key"].drop_nulls().unique().to_list()))
    if observed != tuple(sorted(EXPECTED_EXECUTION_SCENARIOS)):
        raise RuntimeError("execution evidence does not contain the three fixed scenarios")
    if (
        execution.select(pl.struct((*POINT_KEYS, "scenario_key")).n_unique()).item()
        != execution.height
    ):
        raise RuntimeError("execution evidence contains duplicate scenario decision keys")
    return execution


def _point_predictions(frame: pl.DataFrame, probability: np.ndarray) -> pl.DataFrame:
    if frame.height != len(probability) or not np.isfinite(probability).all():
        raise RuntimeError("model probability output is incomplete or non-finite")
    return (
        frame.select(*POINT_KEYS, "observed_at", "label_up")
        .with_columns(pl.Series("probability_up", probability))
        .with_columns(
            (pl.col("probability_up") >= 0.5).cast(pl.Int8).alias("predicted_up"),
            pl.max_horizontal("probability_up", 1.0 - pl.col("probability_up")).alias("confidence"),
        )
        .with_columns(
            (pl.col("predicted_up") == pl.col("label_up")).alias("correct"),
            (pl.col("confidence") >= 0.89).alias("policy_selected_at_point"),
        )
        .sort(list(POINT_KEYS))
    )


def _first_crossings(
    frame: pl.DataFrame,
    probability: np.ndarray,
    threshold: float = 0.89,
) -> pl.DataFrame:
    if threshold != 0.89:
        raise ValueError("benchmark confidence threshold is fixed at 0.89")
    return (
        _point_predictions(frame, probability)
        .filter(pl.col("confidence") >= threshold)
        .sort(["market_id", "seconds_elapsed"])
        .group_by("market_id", maintain_order=True)
        .first()
        .sort(["window_start", "market_id"])
    )


def _market_equal_weights(frame: pl.DataFrame) -> np.ndarray:
    if frame.is_empty() or "market_id" not in frame.columns:
        raise RuntimeError("market-equal weights require nonempty market rows")
    return (
        frame.select((1.0 / pl.len().over("market_id")).alias("weight"))["weight"]
        .to_numpy()
        .astype(np.float64)
    )


def _market_weight_totals(
    frame: pl.DataFrame,
    weights: np.ndarray,
) -> tuple[float, float]:
    totals = (
        frame.select("market_id")
        .with_columns(pl.Series("weight", weights))
        .group_by("market_id")
        .agg(pl.col("weight").sum())
    )["weight"]
    return float(totals.min()), float(totals.max())


def _runtime_probabilities(
    model: dict[str, Any],
    frame: pl.DataFrame,
    features: tuple[str, ...],
) -> np.ndarray:
    _require_finite(frame, features, "frozen champion scoring frame")
    matrix = frame.select(*features).to_numpy()
    elapsed = frame["seconds_elapsed"].to_numpy()
    output = np.empty(frame.height, dtype=np.float64)
    for index in range(frame.height):
        scored = score_runtime_model(
            model,
            matrix[index],
            seconds_elapsed=int(elapsed[index]),
        )
        output[index] = float(scored["probability_up"])
    if not np.isfinite(output).all():
        raise RuntimeError("frozen champion produced non-finite probabilities")
    return output


def _golden_vectors(
    frame: pl.DataFrame,
    model: PaperModel,
    features: tuple[str, ...],
) -> dict[str, Any]:
    selected = pl.concat(
        [
            frame.filter((pl.col("seconds_elapsed") >= start) & (pl.col("seconds_elapsed") < end))
            .sort(list(POINT_KEYS))
            .head(3)
            for _, start, end in FIRST_CROSSING_TIME_BANDS
        ],
        how="vertical",
    )
    probability = model.probability(selected)
    vectors = []
    for index, row in enumerate(selected.to_dicts()):
        vectors.append(
            {
                "key": {name: _json_scalar(row[name]) for name in POINT_KEYS},
                "features": {name: float(row[name]) for name in features},
                "expected_probability_up": float(probability[index]),
            }
        )
    return {
        "schema_version": "btc-spot-l2-chainlink-golden-vectors-v1",
        "profile": model.profile,
        "paper_only": True,
        "feature_names": list(features),
        "vectors": vectors,
    }


def _verify_model_artifact(
    path: Path,
    expected: PaperModel,
    evaluation: pl.DataFrame,
) -> None:
    loaded = joblib.load(path)
    if not isinstance(loaded, PaperModel) or loaded.profile != expected.profile:
        raise RuntimeError("paper model artifact failed type/profile verification")
    sample = evaluation.sort(list(POINT_KEYS)).head(100)
    np.testing.assert_array_equal(loaded.probability(sample), expected.probability(sample))


def _write_prediction_artifacts(
    run_dir: Path,
    prefix: str,
    *,
    policy_points: pl.DataFrame,
    policy_first: pl.DataFrame,
    evaluation_points: pl.DataFrame,
    evaluation_first: pl.DataFrame,
) -> dict[str, Any]:
    frames = {
        "policy_points": policy_points,
        "policy_first_crossings": policy_first,
        "evaluation_points": evaluation_points,
        "evaluation_first_crossings": evaluation_first,
    }
    output: dict[str, Any] = {}
    for role, frame in frames.items():
        path = run_dir / f"{prefix}-{role.replace('_', '-')}.parquet"
        frame.write_parquet(path, compression="zstd")
        output[role] = {
            "path": path.name,
            "sha256": file_sha256(path),
            "rows": frame.height,
            "markets": frame["market_id"].n_unique() if frame.height else 0,
        }
    return output


def _write_run_manifest(
    config: SpotL2ChainlinkConfig,
    run_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    files = []
    for path in sorted(run_dir.iterdir()):
        if path.name == "run-manifest.json" or not path.is_file():
            continue
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": result["run_id"],
        "paper_only": True,
        "live_capital_allowed": False,
        "runtime_exported": False,
        "champion_changed": False,
        "process_configuration_changed": False,
        "configuration_sha256": file_sha256(config.source_path),
        "implementation_git_revision": result["implementation_git"]["revision"],
        "files": files,
    }
    path = run_dir / "run-manifest.json"
    write_json_exclusive(path, manifest)
    return {**manifest, "path": path.name, "sha256": file_sha256(path)}


def _markdown_report(result: dict[str, Any]) -> str:
    coverage = result["coverage"]
    lines = [
        "# BTC Spot-L2 + Chainlink Candle Paper Benchmark",
        "",
        f"**Outcome:** `{result['status']}`.",
        "",
        (
            "This is a retrospective paper-only benchmark. It did not change the "
            "champion, live-capital authorization, runtime model, trading process, "
            "database schema, ingesters, workers, or container stack."
        ),
        "",
        "## Coverage fixed before training",
        "",
        (
            f"Qualified spot-L2 seconds: {coverage['qualified_l2_seconds']['observed']:,} "
            f"of {coverage['qualified_l2_seconds']['possible']:,} "
            f"({coverage['qualified_l2_seconds']['coverage']:.4%}). This source-level "
            "rate is not treated as training coverage."
        ),
        "",
        "| Cohort source | Decision rows | Markets | Row coverage | Market coverage |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("l2", "Qualified L2"),
        ("candles", "Closed candles"),
        ("strict", "Strict intersection"),
    ):
        row = coverage["overall"][key]
        lines.append(
            f"| {label} | {row['decision_rows']:,} | {row['markets']:,} | "
            f"{row['decision_row_coverage']:.2%} | {row['market_coverage']:.2%} |"
        )
    lines.extend(
        [
            "",
            (
                f"Zero-L2 intervals explicitly excluded: "
                f"{len(coverage['excluded_intervals'])}. Exact excluded decision "
                "keys, collapsed partial intervals, and checksums are in the "
                "coverage manifest."
            ),
            "",
            "## Locked evaluation",
            "",
        ]
    )
    for cohort_name, cohort in result["cohorts"].items():
        lines.extend(
            [
                f"### `{cohort_name}`",
                "",
                "| Profile | Features | First crossings | Accuracy | Balanced accuracy | ECE |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for profile, evidence in cohort["profiles"].items():
            aggregate = evidence["locked_evaluation"]["first_crossing_quality"]["aggregate"]
            metrics = aggregate["metrics"]
            lines.append(
                f"| `{profile}` | {evidence['feature_count']} | "
                f"{aggregate['predicted_markets']:,} | "
                f"{_format_number(metrics['accuracy'])} | "
                f"{_format_number(metrics['balanced_accuracy'])} | "
                f"{_format_number(metrics['expected_calibration_error'])} |"
            )
        lines.append("")
        for profile, comparison in cohort["comparisons"].items():
            base = comparison["execution_scenarios"][BASE_EXECUTION_SCENARIO]["metrics"]
            deltas = base["deltas"]
            gates = comparison["advancement_gates"]
            lines.extend(
                [
                    (
                        f"- `{profile}` versus matched control: expectancy delta "
                        f"{_format_number(deltas['net_expectancy_per_eligible_market'])}, "
                        f"profit-factor delta "
                        f"{_format_number(deltas['profit_factor_improvement'])}, "
                        f"gross-loss reduction "
                        f"{_format_percent(deltas['gross_loss_reduction'])}; "
                        f"fixed gates passed: **{gates['passed']}**."
                    ),
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Advancement decision",
            "",
            (
                "No challenger is selected; the champion remains unchanged."
                if result["advancement"]["selected_candidate"] is None
                else f"Pilot challenger: `{result['advancement']['selected_candidate']}`."
            ),
            "",
            (
                "The 150 ms / 80% and 300 ms / 65% depth scenarios use the stored "
                "10-share VWAP as a conservative proxy for 6.25 and 7.6923 raw "
                "shares. Only the 600 ms / 50% scenario maps exactly to stored "
                "10-share VWAP. Non-exact stress evidence is a hard advancement "
                "failure, not a silent pass."
            ),
            "",
            (
                "Execution snapshots in the locked interval are available on "
                f"{coverage['execution_evidence']['snapshot_utc_days']} UTC days; "
                "strict 10-share execution evidence is available on "
                f"{coverage['execution_evidence']['strict_10_share_eligible_utc_days']} "
                "UTC days. The locked interval is also not globally pristine because "
                "portions of July were examined previously. Any future qualifying "
                "challenger still requires forward paper validation."
            ),
            "",
            "## Reproducibility",
            "",
            f"Implementation revision: `{result['implementation_git']['revision']}`.",
            (
                "All six matched-cohort models, prediction files, paired ledgers, "
                "golden vectors, source/feature manifests, and SHA-256 hashes are "
                "recorded in `run-manifest.json` and `benchmark-evidence.json`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _git_provenance(config: SpotL2ChainlinkConfig) -> dict[str, Any]:
    repository = config.package_root.parent.parent
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "benchmark execution requires a clean committed worktree; commit the "
            "implementation before producing immutable artifacts"
        )
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"revision": revision, "branch": branch, "clean": True}


def _validate_cohort_frame(
    frame: pl.DataFrame,
    config: SpotL2ChainlinkConfig,
    label: str,
) -> None:
    required = (
        *POINT_KEYS,
        "observed_at",
        "label_up",
        *config.base_features,
    )
    _require_finite(frame, required, label)
    if frame.select(pl.struct(POINT_KEYS).n_unique()).item() != frame.height:
        raise RuntimeError(f"{label} contains duplicate decision keys")
    if frame.filter(~pl.col("seconds_elapsed").is_in(config.decision_seconds)).height:
        raise RuntimeError(f"{label} contains decisions outside the fixed cadence")
    if frame.filter(
        (pl.col("window_start") < fixed_datetime("source_start"))
        | (pl.col("window_start") >= fixed_datetime("source_end"))
    ).height:
        raise RuntimeError(f"{label} contains rows outside the fixed source interval")


def _assert_prediction_keys(
    predictions: pl.DataFrame,
    expected: pl.DataFrame,
    profile: str,
    split: str,
) -> None:
    observed = predictions.select(*POINT_KEYS).sort(list(POINT_KEYS))
    if not observed.equals(expected, null_equal=True):
        raise RuntimeError(f"{profile} {split} prediction keys differ from its cohort")


def _assert_key_subset(left: pl.DataFrame, right: pl.DataFrame, label: str) -> None:
    outside = left.join(
        right.select(*POINT_KEYS),
        on=list(POINT_KEYS),
        how="anti",
    )
    if outside.height:
        raise RuntimeError(f"{label} coverage contains keys outside core decisions")


def _assert_disjoint_markets(*frames: pl.DataFrame) -> None:
    seen: set[str] = set()
    for frame in frames:
        markets = set(frame["market_id"].unique().to_list())
        if seen & markets:
            raise RuntimeError("a five-minute market crosses benchmark splits")
        seen |= markets


def _range(frame: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    result = frame.filter(
        (pl.col("window_start") >= fixed_datetime(start))
        & (pl.col("window_start") < fixed_datetime(end))
    )
    if result.is_empty():
        raise RuntimeError(f"benchmark split is empty: {start} to {end}")
    return result


def _require_finite(
    frame: pl.DataFrame,
    names: tuple[str, ...],
    label: str,
) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing columns: {', '.join(missing)}")
    numeric = [name for name in names if name not in {"market_id", "window_start", "observed_at"}]
    if frame.select(
        pl.any_horizontal(
            pl.col(name).is_null() | ~pl.col(name).cast(pl.Float64).is_finite() for name in numeric
        ).any()
    ).item():
        raise RuntimeError(f"{label} contains null or non-finite model values")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    return value


def _json_scalar(value: Any) -> Any:
    return _json_safe(value)


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{float(value):.6f}"


def _format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{float(value):.2%}"
