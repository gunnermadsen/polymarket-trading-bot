"""Offline full-window payoff challenger tournament.

The tournament reuses the existing continuous-edge feature, model, calibration,
and execution contracts.  It changes only offline admission and selection
policies and never exports runtime artifacts or mutates trading processes.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import tomllib
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn

from .continuous_edge_training import (
    CHAINLINK_FEATURES,
    PRIMARY_FEATURES,
    TrainingConfig,
    _block,
    _finite,
    _source_identity,
    attach_admission_probability,
    attach_calibration_guard,
    attach_price_time_calibration,
    attach_probability_stability,
    choose_candidate,
    coverage_summary,
    extract_capacity_evidence,
    fit_admission_model,
    fit_calibration_guard,
    fit_candidate,
    fit_family_proxies,
    fit_price_time_calibration,
    load_training_frame,
    policy_metrics,
    policy_metrics_by_price_bucket,
    rolling_policy_metrics,
    score_frame,
)
from .continuous_edge_training import (
    load_config as load_training_config,
)
from .core_extract import file_sha256

SCHEMA_VERSION = "btc-full-window-payoff-challenger-tournament-v1"
MODEL_SCHEMA_VERSION = "btc-full-window-payoff-challenger-model-v1"
CANDIDATE_NAMES = (
    "continuous_payoff_baseline",
    "time_price_stratified_payoff",
    "chronological_loss_tail_guard",
    "stability_selected_regime",
    "oof_expert_distilled_admission",
)
ENTRY_CELLS = (
    ("early_15_89", 15, 90),
    ("middle_90_119", 90, 120),
    ("middle_120_149", 120, 150),
    ("middle_150_179", 150, 180),
    ("late_180_240", 180, 241),
)


@dataclass(frozen=True)
class ChallengerPolicyConfig:
    names: tuple[str, ...]
    loss_probability_thresholds: tuple[float, ...]
    expected_shortfall_thresholds: tuple[float, ...]
    family_spread_thresholds: tuple[float, ...]
    coverage_targets: tuple[float, ...]
    minimum_nonempty_fold_ratio: float


@dataclass(frozen=True)
class SelectionCellConfig:
    name: str
    start_second: int
    end_second_exclusive: int
    minimum_validation_accuracy: float
    minimum_validation_trades: int


@dataclass(frozen=True)
class GateConfig:
    minimum_trades: int
    minimum_market_coverage: float
    minimum_accuracy: float
    minimum_wilson_lower: float
    minimum_profit_factor: float
    minimum_stress_expectancy_per_trade: float
    minimum_payoff_ratio: float
    minimum_profitable_fold_ratio: float
    minimum_active_days: int
    maximum_daily_pnl_concentration: float
    maximum_average_entry_second: float
    maximum_drawdown_to_net_pnl: float
    minimum_direction_trades: int
    minimum_price_bucket_trades: int
    bootstrap_resamples: int
    bootstrap_confidence: float


@dataclass(frozen=True)
class TournamentConfig:
    source_path: Path
    evidence_training: TrainingConfig
    training: TrainingConfig
    selection_cells: tuple[SelectionCellConfig, ...]
    challengers: ChallengerPolicyConfig
    gates: GateConfig


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    use_conservative_probability: bool
    use_price_floor: bool
    use_admission: bool
    use_loss_tail: bool
    use_consensus: bool
    stability_first: bool


SPECS = {
    "continuous_payoff_baseline": CandidateSpec(
        "continuous_payoff_baseline", False, False, False, False, False, False
    ),
    "time_price_stratified_payoff": CandidateSpec(
        "time_price_stratified_payoff", True, True, True, False, False, False
    ),
    "chronological_loss_tail_guard": CandidateSpec(
        "chronological_loss_tail_guard", True, True, True, True, False, False
    ),
    "stability_selected_regime": CandidateSpec(
        "stability_selected_regime", True, True, True, False, False, True
    ),
    "oof_expert_distilled_admission": CandidateSpec(
        "oof_expert_distilled_admission", True, True, True, False, True, False
    ),
}


def load_config(path: Path) -> TournamentConfig:
    source = path.resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    evidence_training = load_training_config(source)
    challenger_windows = raw["challenger_windows"]
    training = replace(
        evidence_training,
        windows=replace(
            evidence_training.windows,
            **{name: _utc_timestamp(value) for name, value in challenger_windows.items()},
        ),
    )
    challenger = raw["challengers"]
    config = TournamentConfig(
        source_path=source,
        evidence_training=evidence_training,
        training=training,
        selection_cells=tuple(SelectionCellConfig(**values) for values in raw["selection_cells"]),
        challengers=ChallengerPolicyConfig(
            names=tuple(str(value) for value in challenger["names"]),
            loss_probability_thresholds=tuple(
                float(value) for value in challenger["loss_probability_thresholds"]
            ),
            expected_shortfall_thresholds=tuple(
                float(value) for value in challenger["expected_shortfall_thresholds"]
            ),
            family_spread_thresholds=tuple(
                float(value) for value in challenger["family_spread_thresholds"]
            ),
            coverage_targets=tuple(float(value) for value in challenger["coverage_targets"]),
            minimum_nonempty_fold_ratio=float(challenger["minimum_nonempty_fold_ratio"]),
        ),
        gates=GateConfig(**raw["gates"]),
    )
    _validate_config(config)
    return config


def _validate_config(config: TournamentConfig) -> None:
    if config.training.profile != "btc_5m_full_window_payoff_challenger":
        raise ValueError("unexpected full-window challenger profile")
    if config.challengers.names != CANDIDATE_NAMES:
        raise ValueError("challenger matrix changed")
    if config.training.fresh_holdout:
        raise ValueError("previously consumed August evidence cannot be marked fresh")
    if (
        tuple(
            (cell.name, cell.start_second, cell.end_second_exclusive)
            for cell in config.selection_cells
        )
        != ENTRY_CELLS
    ):
        raise ValueError("selection cells changed from the five-cell challenger contract")
    if not 0 < config.challengers.minimum_nonempty_fold_ratio <= 1:
        raise ValueError("nonempty fold ratio must be in (0, 1]")
    if config.gates.bootstrap_resamples < 500:
        raise ValueError("bootstrap requires at least 500 resamples")


def run_tournament(config: TournamentConfig) -> tuple[Path, dict[str, Any]]:
    base = config.training
    print("load: immutable full-window capacity evidence", flush=True)
    evidence_manifest = extract_capacity_evidence(config.evidence_training)
    frame = load_training_frame(base, evidence_manifest)
    coverage = coverage_summary(frame, base)
    print(
        f"joined: {frame.height:,} strict rows, {frame['market_id'].n_unique():,} markets",
        flush=True,
    )

    probability_specs = {
        "core_oracle_vwap_curve": PRIMARY_FEATURES,
        "core_oracle_vwap_curve_chainlink": (*PRIMARY_FEATURES, *CHAINLINK_FEATURES),
    }
    probability_candidates: dict[str, Any] = {}
    probability_metrics: dict[str, Any] = {}
    for name, features in probability_specs.items():
        print(f"train probability: {name}", flush=True)
        experts, result = fit_candidate(base, frame, name, tuple(features))
        probability_candidates[name] = experts
        probability_metrics[name] = result
    selected_probability, probability_decision = choose_candidate(
        base,
        frame,
        probability_candidates,
        probability_metrics,
    )
    experts = probability_candidates[selected_probability]

    print("train: chronological family proxies", flush=True)
    family_proxies, family_proxy_metrics = fit_family_proxies(base, frame)
    calibration_raw = score_frame(
        _block(frame, base.windows.outcome_fit_end, base.windows.calibration_end),
        experts,
        family_proxies,
        base,
    )
    price_time_calibration = fit_price_time_calibration(calibration_raw, base)
    calibrated = attach_price_time_calibration(calibration_raw, price_time_calibration, base)
    calibration_guard = fit_calibration_guard(calibrated, base)

    admission_raw = _prepare_scored(
        _block(frame, base.windows.calibration_end, base.windows.admission_end),
        experts,
        family_proxies,
        price_time_calibration,
        calibration_guard,
        base,
    )
    print("train: chronological payoff and loss-tail admission", flush=True)
    admission_model = fit_admission_model(admission_raw, base)
    validation = attach_admission_probability(
        _prepare_scored(
            _block(frame, base.windows.validation_start, base.windows.validation_end),
            experts,
            family_proxies,
            price_time_calibration,
            calibration_guard,
            base,
        ),
        admission_model,
    )

    policies: dict[str, Any] = {}
    validation_results: dict[str, Any] = {}
    for index, name in enumerate(config.challengers.names):
        print(f"select validation policy: {name}", flush=True)
        spec = SPECS[name]
        policy, history = _select_policy(validation, spec, config)
        selected = _apply_policy(validation, spec, policy)
        evaluation = _evaluate(
            selected,
            validation,
            config,
            start=base.windows.validation_start,
            end=base.windows.validation_end,
            seed=base.random_seed + 1000 + index,
        )
        policies[name] = policy
        validation_results[name] = {
            "selection": history,
            "submitted_policy": policy,
            "submitted": evaluation,
            "coverage_profiles": _coverage_profiles(validation, spec, policy, config),
        }

    preselected_champion = max(
        config.challengers.names,
        key=lambda name: _candidate_rank(
            validation_results[name]["submitted"], SPECS[name].stability_first
        ),
    )

    print(
        "evaluation: opening reused development comparison block "
        f"{base.windows.validation_end.isoformat()} to {base.windows.test_end.isoformat()}",
        flush=True,
    )
    test = attach_admission_probability(
        _prepare_scored(
            _block(frame, base.windows.validation_end, base.windows.test_end),
            experts,
            family_proxies,
            price_time_calibration,
            calibration_guard,
            base,
        ),
        admission_model,
    )
    strict_markets = test["market_id"].n_unique()
    scheduled_markets = int(
        pl.scan_parquet(base.paths.capacity_evidence / "test.parquet")
        .select(pl.col("market_id").n_unique())
        .collect()
        .item()
    )
    development_results: dict[str, Any] = {}
    ledgers: dict[str, pl.DataFrame] = {}
    for index, name in enumerate(config.challengers.names):
        spec = SPECS[name]
        selected = _apply_policy(test, spec, policies[name])
        result = _evaluate(
            selected,
            test,
            config,
            start=base.windows.validation_end,
            end=base.windows.test_end,
            seed=base.random_seed + 2000 + index,
        )
        result["metrics"]["strict_data_coverage"] = (
            strict_markets / scheduled_markets if scheduled_markets else 0.0
        )
        result["metrics"]["end_to_end_market_coverage"] = (
            selected["market_id"].n_unique() / scheduled_markets if scheduled_markets else 0.0
        )
        result["metrics"]["strict_markets"] = strict_markets
        result["metrics"]["scheduled_markets"] = scheduled_markets
        development_results[name] = {
            "validation": validation_results[name],
            "test": result,
            "capacity": {
                str(quantity): policy_metrics(selected, base, quantity=quantity)
                for quantity in base.execution.quantities
            },
            "time_bands": {
                band.name: policy_metrics(
                    selected.filter(pl.col("time_band") == band.name), base, quantity=5
                )
                for band in base.bands
            },
            "entry_cells": _entry_cell_metrics(selected, base),
            "directions": _direction_metrics(selected, base),
            "price_buckets": policy_metrics_by_price_bucket(selected, base, quantity=5),
            "loss_tail": _loss_tail_metrics(selected, base),
        }
        ledgers[name] = selected

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    temporary = base.paths.runs / f"{run_id}.partial"
    final = base.paths.committed_results / run_id
    if temporary.exists() or final.exists():
        raise FileExistsError(run_id)
    temporary.mkdir(parents=True)
    artifact_path = temporary / "tournament.joblib"
    joblib.dump(
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "profile": base.profile,
            "selected_probability_candidate": selected_probability,
            "probability_experts": experts,
            "family_proxies": family_proxies,
            "price_time_calibration": price_time_calibration,
            "calibration_guard": calibration_guard,
            "admission_model": admission_model,
            "challenger_specs": SPECS,
            "policies": policies,
            "preselected_champion": preselected_champion,
            "runtime_exported": False,
            "production_qualified": False,
        },
        artifact_path,
        compress=3,
    )
    ledger_dir = temporary / "development-ledgers"
    ledger_dir.mkdir()
    for name, ledger in ledgers.items():
        ledger.write_parquet(ledger_dir / f"{name}.parquet", compression="zstd")

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": base.profile,
        "paper_only": True,
        "fresh_holdout": False,
        "runtime_exported": False,
        "production_qualified": False,
        "trading_processes_changed": False,
        "source_commit": _git_revision(base.package_root),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "configuration": {
            "path": str(config.source_path),
            "sha256": file_sha256(config.source_path),
            "windows": {name: value.isoformat() for name, value in asdict(base.windows).items()},
            "evidence_windows": {
                name: value.isoformat()
                for name, value in asdict(config.evidence_training.windows).items()
            },
            "bands": [asdict(band) for band in base.bands],
            "selection_cells": [asdict(cell) for cell in config.selection_cells],
            "execution": asdict(base.execution),
            "policy": asdict(base.policy),
            "challengers": asdict(config.challengers),
            "gates": asdict(config.gates),
        },
        "data": {
            "rows": frame.height,
            "markets": frame["market_id"].n_unique(),
            "capacity_manifest": evidence_manifest,
            "coverage": coverage,
            "oracle_features": _source_identity(base.paths.oracle_features),
            "chainlink_features": _source_identity(base.paths.chainlink_features),
            "directional_family_contract": _source_identity(base.paths.directional_family_model),
            "asymmetric_family_contract": _source_identity(base.paths.asymmetric_family_model),
            "open_interest": {
                "used": False,
                "reason": "causal OI does not cover the full outcome-fit window",
            },
            "l2_and_trade_prints": {
                "used": False,
                "reason": "causal full-window coverage remains insufficient",
            },
            "twap": {
                "used": False,
                "reason": "no persisted settlement-aligned historical source is available",
            },
        },
        "probability_candidates": probability_metrics,
        "probability_decision": probability_decision,
        "selected_probability_candidate": selected_probability,
        "family_proxy_metrics": family_proxy_metrics,
        "payoff_lower_bound": admission_model.oof_diagnostics,
        "preselected_champion": preselected_champion,
        "development_results": development_results,
        "model_artifact": {
            "path": artifact_path.name,
            "sha256": file_sha256(artifact_path),
        },
        "decision": {
            "status": "development_comparison_only",
            "champion_selected_before_test": preselected_champion,
            "paper_soak_remains_independent": True,
            "production_selection_allowed": False,
        },
        "limitations": [
            "The August 2 comparison block was consumed by prior work and is not an independent holdout.",
            "The active paper soak is intentionally excluded from training and selection.",
            "Projected PnL assumes recorded ask VWAP was fillable and does not model queue position.",
            "Open interest, Binance L2, trade prints, and TWAP were not admitted as mandatory full-window features.",
            "The OOF expert candidate distills chronological family-proxy agreement through the existing admission feature contract; it does not run an ensemble at inference.",
        ],
    }
    _write_json(temporary / "metrics.json", metrics)
    (temporary / "report.md").write_text(_render_report(metrics))
    (temporary / "tournament.sha256").write_text(file_sha256(artifact_path) + "\n")
    base.paths.committed_results.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    return final, metrics


def _prepare_scored(
    frame: pl.DataFrame,
    experts: dict[str, Any],
    family_proxies: dict[str, Any],
    price_time_calibration: Any,
    calibration_guard: Any,
    config: TrainingConfig,
) -> pl.DataFrame:
    scored = score_frame(frame, experts, family_proxies, config)
    scored = attach_price_time_calibration(scored, price_time_calibration, config)
    scored = attach_calibration_guard(scored, calibration_guard, config)
    return attach_probability_stability(scored)


def _select_policy(
    frame: pl.DataFrame,
    spec: CandidateSpec,
    config: TournamentConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    history: dict[str, Any] = {}
    base = config.training
    for cell in config.selection_cells:
        subset = frame.filter(
            pl.col("seconds_elapsed").is_between(
                cell.start_second,
                cell.end_second_exclusive - 1,
            )
        ).sort(["market_id", "seconds_elapsed", "observed_at"])
        records = []
        for values in _parameter_grid(spec, config):
            selected = _select_rows(subset, spec, values)
            metrics = policy_metrics(selected, base, quantity=5)
            folds = rolling_policy_metrics(
                selected,
                base,
                base.windows.validation_start,
                base.windows.validation_end,
                quantity=5,
            )
            fold_summary = _nonempty_fold_summary(folds)
            checks = _band_checks(metrics, fold_summary, cell, config)
            records.append(
                {
                    "values": values,
                    "metrics": metrics,
                    "rolling_folds": folds,
                    "nonempty_folds": fold_summary,
                    "checks": checks,
                    "qualified": all(checks.values()),
                }
            )
        qualified = [record for record in records if record["qualified"]]
        pool = qualified or records
        chosen = max(
            pool,
            key=lambda row: _band_rank(
                row,
                spec.stability_first,
                minimum_trades=cell.minimum_validation_trades,
            ),
        )
        policy = {**chosen["values"], "enabled": chosen["metrics"]["trades"] > 0}
        policies[cell.name] = policy
        history[cell.name] = {
            "attempts": len(records),
            "qualified_attempts": len(qualified),
            "strictly_qualified": bool(qualified),
            "selected": policy,
            "validation_metrics": chosen["metrics"],
            "rolling_folds": chosen["rolling_folds"],
            "nonempty_folds": chosen["nonempty_folds"],
            "qualification_checks": chosen["checks"],
            "risk_coverage_frontier": _risk_coverage_frontier(records),
        }
    return policies, history


def _parameter_grid(
    spec: CandidateSpec,
    config: TournamentConfig,
) -> list[dict[str, Any]]:
    base = config.training.policy
    admissions = base.admission_thresholds if spec.use_admission else (-math.inf,)
    lower_bounds = base.payoff_lower_bound_thresholds if spec.use_admission else (-math.inf,)
    loss_probabilities = (
        config.challengers.loss_probability_thresholds if spec.use_loss_tail else (math.inf,)
    )
    expected_shortfalls = (
        config.challengers.expected_shortfall_thresholds if spec.use_loss_tail else (math.inf,)
    )
    family_spreads = (
        config.challengers.family_spread_thresholds if spec.use_consensus else (math.inf,)
    )
    return [
        {
            "confidence": confidence,
            "edge": edge,
            "admission": admission,
            "payoff_lower_bound": lower_bound,
            "maximum_loss_probability": loss_probability,
            "maximum_expected_shortfall": expected_shortfall,
            "maximum_family_spread": family_spread,
            "require_expert_agreement": spec.use_consensus,
        }
        for (
            confidence,
            edge,
            admission,
            lower_bound,
            loss_probability,
            expected_shortfall,
            family_spread,
        ) in product(
            base.confidence_thresholds,
            base.edge_thresholds,
            admissions,
            lower_bounds,
            loss_probabilities,
            expected_shortfalls,
            family_spreads,
        )
    ]


def _select_rows(
    frame: pl.DataFrame,
    spec: CandidateSpec,
    values: dict[str, Any],
) -> pl.DataFrame:
    mask = _policy_mask(frame, spec, values)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return frame.head(0)
    markets = frame["market_id"].to_numpy()
    _, first = np.unique(markets[indices], return_index=True)
    return frame[indices[np.sort(first)].tolist()]


def _policy_mask(
    frame: pl.DataFrame,
    spec: CandidateSpec,
    values: dict[str, Any],
) -> np.ndarray:
    probability_name = (
        "conservative_probability_selected"
        if spec.use_conservative_probability
        else "probability_selected"
    )
    edge_name = "conservative_edge_5" if spec.use_conservative_probability else "selected_edge_5"
    edge = frame[edge_name].to_numpy()
    mask = (frame[probability_name].to_numpy() >= values["confidence"]) & (edge >= values["edge"])
    if spec.use_price_floor:
        mask &= edge >= frame["price_bucket_minimum_edge"].to_numpy()
    if spec.use_admission:
        mask &= (frame["admission_probability"].to_numpy() >= values["admission"]) & (
            frame["payoff_stress_edge_lower_bound"].to_numpy() >= values["payoff_lower_bound"]
        )
    if spec.use_loss_tail:
        mask &= (
            frame["payoff_loss_probability"].to_numpy() <= values["maximum_loss_probability"]
        ) & (frame["payoff_expected_shortfall"].to_numpy() <= values["maximum_expected_shortfall"])
    if spec.use_consensus:
        predicted = frame["predicted_up"].to_numpy().astype(bool)
        directional = frame["directional_family_probability_up"].to_numpy() >= 0.5
        asymmetric = frame["asymmetric_family_probability_up"].to_numpy() >= 0.5
        mask &= (
            (directional == predicted)
            & (asymmetric == predicted)
            & (frame["family_probability_spread"].to_numpy() <= values["maximum_family_spread"])
        )
    return mask


def _apply_policy(
    frame: pl.DataFrame,
    spec: CandidateSpec,
    policy: dict[str, dict[str, Any]],
) -> pl.DataFrame:
    eligible = np.zeros(frame.height, dtype=bool)
    seconds = frame["seconds_elapsed"].to_numpy()
    for name, values in policy.items():
        if not values.get("enabled", False):
            continue
        cell = next(cell for cell in ENTRY_CELLS if cell[0] == name)
        eligible |= (seconds >= cell[1]) & (seconds < cell[2]) & _policy_mask(frame, spec, values)
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return frame.head(0)
    markets = frame["market_id"].to_numpy()
    _, first = np.unique(markets[indices], return_index=True)
    return frame[indices[np.sort(first)].tolist()]


def _band_checks(
    metrics: dict[str, Any],
    folds: dict[str, Any],
    band: Any,
    config: TournamentConfig,
) -> dict[str, bool]:
    return {
        "minimum_trades": metrics["trades"] >= band.minimum_validation_trades,
        "minimum_accuracy": metrics["accuracy"] >= band.minimum_validation_accuracy,
        "positive_stress_expectancy": metrics["stress_expectancy_per_trade"]
        >= config.training.policy.minimum_stress_expectancy_per_trade,
        "minimum_profit_factor": (metrics["profit_factor"] or 0.0) >= 1.10,
        "minimum_payoff_ratio": metrics["payoff_ratio"]
        >= config.training.policy.minimum_payoff_ratio,
        "minimum_active_days": metrics["active_days"] >= 4,
        "minimum_nonempty_fold_ratio": folds["profitable_fold_ratio"]
        >= config.challengers.minimum_nonempty_fold_ratio,
    }


def _band_rank(
    record: dict[str, Any],
    stability_first: bool,
    *,
    minimum_trades: int,
) -> tuple[Any, ...]:
    metrics = record["metrics"]
    folds = record["nonempty_folds"]
    viable = metrics["trades"] >= minimum_trades and metrics["stress_expectancy_per_trade"] > 0
    if stability_first:
        return (
            record["qualified"],
            viable,
            folds["worst_stress_expectancy"],
            folds["median_stress_expectancy"],
            metrics["stress_expectancy_per_trade"],
            metrics["trades"],
        )
    return (
        record["qualified"],
        viable,
        folds["profitable_fold_ratio"],
        folds["median_stress_expectancy"],
        metrics["stress_net_pnl"],
        metrics["trades"],
    )


def _nonempty_fold_summary(folds: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in folds["folds"] if row["trades"] > 0]
    values = [row["stress_expectancy_per_trade"] for row in rows]
    profitable = sum(row["stress_net_pnl"] > 0 for row in rows)
    return {
        "folds": len(rows),
        "profitable_folds": profitable,
        "profitable_fold_ratio": profitable / len(rows) if rows else 0.0,
        "median_stress_expectancy": float(np.median(values)) if values else 0.0,
        "worst_stress_expectancy": min(values, default=0.0),
    }


def _risk_coverage_frontier(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "policy": row["values"],
            "trades": row["metrics"]["trades"],
            "accuracy": row["metrics"]["accuracy"],
            "stress_expectancy": row["metrics"]["stress_expectancy_per_trade"],
            "stress_pnl": row["metrics"]["stress_net_pnl"],
        }
        for row in records
    ]
    frontier = []
    for candidate in rows:
        if any(
            other["trades"] >= candidate["trades"]
            and other["stress_expectancy"] >= candidate["stress_expectancy"]
            and (
                other["trades"] > candidate["trades"]
                or other["stress_expectancy"] > candidate["stress_expectancy"]
            )
            for other in rows
        ):
            continue
        frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda row: (row["trades"], row["stress_expectancy"]),
        reverse=True,
    )[:25]


def _coverage_profiles(
    frame: pl.DataFrame,
    spec: CandidateSpec,
    policy: dict[str, dict[str, Any]],
    config: TournamentConfig,
) -> dict[str, Any]:
    candidates = []
    for delta in (-0.10, -0.05, 0.0, 0.05, 0.10):
        adjusted = {
            band: {
                **values,
                "confidence": min(max(values["confidence"] + delta, 0.50), 0.95),
            }
            for band, values in policy.items()
        }
        selected = _apply_policy(frame, spec, adjusted)
        metrics = policy_metrics(selected, config.training, quantity=5)
        markets = frame["market_id"].n_unique()
        metrics["market_coverage"] = selected["market_id"].n_unique() / markets if markets else 0.0
        candidates.append({"confidence_delta": delta, "policy": adjusted, "metrics": metrics})
    return {
        f"coverage_{int(target * 100)}": min(
            candidates,
            key=lambda row: (
                abs(row["metrics"]["market_coverage"] - target),
                -row["metrics"]["stress_net_pnl"],
            ),
        )
        for target in config.challengers.coverage_targets
    }


def _evaluate(
    selected: pl.DataFrame,
    eligible: pl.DataFrame,
    config: TournamentConfig,
    *,
    start: datetime,
    end: datetime,
    seed: int,
) -> dict[str, Any]:
    base = config.training
    metrics = policy_metrics(selected, base, quantity=5)
    markets = eligible["market_id"].n_unique()
    metrics["market_coverage"] = selected["market_id"].n_unique() / markets if markets else 0.0
    metrics["wilson_accuracy_lower"] = _wilson_lower(
        round(metrics["accuracy"] * metrics["trades"]), metrics["trades"]
    )
    metrics["bootstrap_stress_expectancy_lower"] = _day_bootstrap_lower(
        selected, base, config.gates, seed=seed
    )
    q10 = policy_metrics(selected, base, quantity=10)
    q20 = policy_metrics(selected, base, quantity=20)
    metrics["q10_stress_net_pnl"] = q10["stress_net_pnl"]
    metrics["q20_stress_net_pnl"] = q20["stress_net_pnl"]
    folds = rolling_policy_metrics(selected, base, start, end, quantity=5)
    nonempty_folds = _nonempty_fold_summary(folds)
    metrics["profitable_fold_ratio"] = nonempty_folds["profitable_fold_ratio"]
    metrics["worst_fold_stress_expectancy"] = nonempty_folds["worst_stress_expectancy"]
    metrics["median_fold_stress_expectancy"] = nonempty_folds["median_stress_expectancy"]
    directions = _direction_metrics(selected, base)
    buckets = policy_metrics_by_price_bucket(selected, base, quantity=5)
    metrics["enabled_directions_positive"] = all(
        row["stress_net_pnl"] > 0
        for row in directions.values()
        if row["trades"] >= config.gates.minimum_direction_trades
    )
    metrics["populated_price_buckets_nonnegative"] = all(
        row["stress_net_pnl"] >= 0
        for row in buckets.values()
        if row["trades"] >= config.gates.minimum_price_bucket_trades
    )
    metrics["drawdown_to_net_pnl"] = (
        metrics["maximum_drawdown"] / metrics["net_pnl"] if metrics["net_pnl"] > 0 else math.inf
    )
    research, strict = _qualification(metrics, config)
    return {
        "metrics": metrics,
        "rolling_folds": folds,
        "nonempty_folds": nonempty_folds,
        "research_qualification": research,
        "strict_qualification": strict,
    }


def _qualification(
    metrics: dict[str, Any],
    config: TournamentConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gates = config.gates
    research_checks = {
        "minimum_trades": metrics["trades"] >= gates.minimum_trades,
        "minimum_market_coverage": metrics["market_coverage"] >= gates.minimum_market_coverage,
        "minimum_accuracy": metrics["accuracy"] >= gates.minimum_accuracy,
        "positive_stress_expectancy": metrics["stress_expectancy_per_trade"]
        >= gates.minimum_stress_expectancy_per_trade,
        "minimum_profit_factor": (metrics["profit_factor"] or 0.0) >= gates.minimum_profit_factor,
        "minimum_payoff_ratio": metrics["payoff_ratio"] >= gates.minimum_payoff_ratio,
        "positive_q10_stress": metrics["q10_stress_net_pnl"] > 0,
        "positive_q20_stress": metrics["q20_stress_net_pnl"] > 0,
        "minimum_profitable_fold_ratio": metrics["profitable_fold_ratio"]
        >= gates.minimum_profitable_fold_ratio,
        "minimum_active_days": metrics["active_days"] >= gates.minimum_active_days,
        "daily_concentration": metrics["daily_pnl_concentration"]
        <= gates.maximum_daily_pnl_concentration,
        "average_entry_time": metrics["average_entry_second"] is not None
        and metrics["average_entry_second"] <= gates.maximum_average_entry_second,
        "maximum_drawdown_ratio": metrics["drawdown_to_net_pnl"]
        <= gates.maximum_drawdown_to_net_pnl,
        "directions_positive": metrics["enabled_directions_positive"],
        "price_buckets_nonnegative": metrics["populated_price_buckets_nonnegative"],
    }
    strict_checks = {
        **research_checks,
        "minimum_wilson_lower": metrics["wilson_accuracy_lower"] >= gates.minimum_wilson_lower,
        "bootstrap_lower_positive": metrics["bootstrap_stress_expectancy_lower"] > 0,
        "fresh_untouched_holdout": False,
    }
    return (
        {"passed": all(research_checks.values()), "checks": research_checks},
        {"passed": all(strict_checks.values()), "checks": strict_checks},
    )


def _candidate_rank(result: dict[str, Any], stability_first: bool) -> tuple[Any, ...]:
    metrics = result["metrics"]
    if stability_first:
        return (
            result["research_qualification"]["passed"],
            metrics["worst_fold_stress_expectancy"],
            metrics["median_fold_stress_expectancy"],
            metrics["stress_expectancy_per_trade"],
            metrics["market_coverage"],
        )
    return (
        result["research_qualification"]["passed"],
        metrics["profitable_fold_ratio"],
        metrics["stress_expectancy_per_trade"],
        metrics["market_coverage"],
        -metrics["maximum_drawdown"],
    )


def _direction_metrics(frame: pl.DataFrame, config: TrainingConfig) -> dict[str, Any]:
    return {
        name: policy_metrics(frame.filter(pl.col("predicted_up") == value), config, quantity=5)
        for name, value in (("up", True), ("down", False))
    }


def _entry_cell_metrics(frame: pl.DataFrame, config: TrainingConfig) -> dict[str, Any]:
    return {
        name: policy_metrics(
            frame.filter(pl.col("seconds_elapsed").is_between(start, end - 1)),
            config,
            quantity=5,
        )
        for name, start, end in ENTRY_CELLS
    }


def _loss_tail_metrics(frame: pl.DataFrame, config: TrainingConfig) -> dict[str, Any]:
    if frame.is_empty():
        return {
            "loss_trade_ratio": 0.0,
            "worst_trade_stress_pnl": 0.0,
            "worst_five_percent_mean_stress_pnl": 0.0,
        }
    predicted = frame["predicted_up"].to_numpy().astype(bool)
    correct = predicted == frame["label_up"].to_numpy().astype(bool)
    price = np.where(
        predicted,
        frame["up_ask_vwap_5"].to_numpy(),
        frame["down_ask_vwap_5"].to_numpy(),
    )
    fee = frame["fee_rate"].to_numpy() * price * (1.0 - price)
    pnl = (
        correct.astype(float)
        - price
        - fee
        - config.execution.execution_reserve_per_share
        - config.execution.stress_slippage_per_share
    ) * 5
    count = max(1, math.ceil(len(pnl) * 0.05))
    return {
        "loss_trade_ratio": float((pnl < 0).mean()),
        "worst_trade_stress_pnl": float(pnl.min()),
        "worst_five_percent_mean_stress_pnl": float(np.sort(pnl)[:count].mean()),
    }


def _day_bootstrap_lower(
    frame: pl.DataFrame,
    config: TrainingConfig,
    gates: GateConfig,
    *,
    seed: int,
) -> float:
    if frame.is_empty():
        return 0.0
    predicted = frame["predicted_up"].to_numpy().astype(bool)
    correct = predicted == frame["label_up"].to_numpy().astype(bool)
    price = np.where(
        predicted,
        frame["up_ask_vwap_5"].to_numpy(),
        frame["down_ask_vwap_5"].to_numpy(),
    )
    fee = frame["fee_rate"].to_numpy() * price * (1.0 - price)
    stress = (
        correct.astype(float)
        - price
        - fee
        - config.execution.execution_reserve_per_share
        - config.execution.stress_slippage_per_share
    ) * 5
    days = frame["window_start"].dt.date().to_numpy()
    unique_days = np.unique(days)
    daily = [stress[days == day] for day in unique_days]
    rng = np.random.default_rng(seed)
    means = np.empty(gates.bootstrap_resamples)
    for index in range(gates.bootstrap_resamples):
        sampled = rng.choice(len(daily), size=len(daily), replace=True)
        values = np.concatenate([daily[position] for position in sampled])
        means[index] = values.mean()
    return float(np.quantile(means, 1.0 - gates.bootstrap_confidence))


def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = probability + z * z / (2.0 * total)
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    )
    return (center - radius) / denominator


def _render_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# BTC Five-Minute Full-Window Payoff Challenger Tournament",
        "",
        "Status: **development comparison only; active paper soak remains independent**",
        "",
        f"Validation-preselected champion: `{metrics['preselected_champion']}`",
        "",
        "## Reused development comparison",
        "",
        "| Candidate | Research gate | Trades | Coverage | Accuracy | Net PnL | Stress PnL | PF | Avg entry | Worst fold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in metrics["development_results"].items():
        test = result["test"]
        row = test["metrics"]
        lines.append(
            f"| {name} | {'yes' if test['research_qualification']['passed'] else 'no'} | "
            f"{row['trades']} | {row['market_coverage']:.2%} | {row['accuracy']:.2%} | "
            f"{row['net_pnl']:.2f} | {row['stress_net_pnl']:.2f} | {_fmt(row['profit_factor'])} | "
            f"{_fmt(row['average_entry_second'])} | {row['worst_fold_stress_expectancy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Fixed-entry VWAP5 comparison",
            "",
            "PnL includes fees and the frozen execution reserve; stress PnL adds $0.01 per share.",
            "",
            "## Limitations",
            "",
            *(f"- {value}" for value in metrics["limitations"]),
        ]
    )
    return "\n".join(lines) + "\n"


def _git_revision(package_root: Path) -> str:
    git_file = package_root.parents[1] / ".git"
    if not git_file.is_file():
        return "unknown"
    content = git_file.read_text().strip()
    git_dir = Path(content.split(":", 1)[1].strip())
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head
    common = git_dir
    if (git_dir / "commondir").is_file():
        common = (git_dir / (git_dir / "commondir").read_text().strip()).resolve()
    return (common / head.split(" ", 1)[1]).read_text().strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_finite(payload), indent=2, sort_keys=True) + "\n")


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.3f}"


def _utc_timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run_dir, metrics = run_tournament(load_config(args.config))
    print(f"run: {run_dir}")
    print(json.dumps(metrics["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
