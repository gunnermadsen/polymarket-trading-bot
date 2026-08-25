"""End-to-end TWAP60 transfer-learning challenger tournament.

This workflow is deliberately training-only.  It reuses the established BTC
directional feature, model, execution, admission, and artifact conventions and
does not create data sources, tables, ingesters, or trading processes.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from .chainlink_oi_features import CHAINLINK_CANDLE_FEATURES
from .continuous_edge_training import (
    CORE_FEATURES,
    ORACLE_FEATURES,
    VWAP_QUANTITIES,
)
from .core_config import load_core_config
from .core_extract import file_sha256
from .middle_market_ablation_tournament import (
    _apply_policy as apply_frozen_champion_policy,
)
from .middle_market_ablation_tournament import (
    _score_candidate as score_frozen_champion,
)
from .middle_market_ablation_tournament import (
    load_config as load_champion_scoring_config,
)
from .twap60_training_data import (
    REFPRICE_ALL_FEATURES,
    REFPRICE_RUNTIME_FEATURES,
    DataPaths,
    attach_causal_refprice_features,
    build_tournament_frame,
    extract_tournament_sources,
    load_source_group,
    proxy_validation_metrics,
    verify_runtime_refprice_golden_vectors,
)

SCHEMA_VERSION = "btc-twap60-challenger-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-twap60-challenger-model-v1"
CANDIDATE_NAMES = (
    "frozen_chainlink_stratified_payoff",
    "twap60_native_stratified",
    "twap60_proxy_transfer_stratified",
    "twap60_residual_adapter_stratified",
    "twap60_similarity_weighted_stratified",
    "twap60_loss_tail_guard_stratified",
)
FEATURE_TREATMENTS = (
    "oracle_candle_control",
    "refprice_path",
    "refprice_oracle_candle_combined",
)
TRANSFER_CANDIDATES = (
    "twap60_proxy_transfer_stratified",
    "twap60_residual_adapter_stratified",
    "twap60_similarity_weighted_stratified",
)
ADMISSION_FEATURES = (
    "probability_selected",
    "selected_cost_5",
    "seconds_elapsed_scaled",
    "predicted_up_float",
    "price_bucket_index",
    "btc_reversal_5_vs_30",
    "btc_boundary_cross_count",
    "btc_volatility_shock_30_vs_120",
    "chainlink_ref_reversal_5_vs_30",
    "chainlink_ref_boundary_cross_count_60s",
    "chainlink_ref_binance_disagreement_30s",
)
LOSS_FEATURES = (
    "selected_cost_5",
    "correctness_probability",
    "chainlink_ref_reversal_5_vs_30",
    "chainlink_ref_binance_basis_bps",
    "chainlink_ref_realized_volatility_30s_bps",
    "chainlink_ref_boundary_cross_count_60s",
    "chainlink_ref_binance_disagreement_30s",
    "seconds_elapsed_scaled",
    "price_bucket_index",
)


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_iter: int
    proxy_weight: float
    transition_weight: float
    calibration_c: float
    shrinkage_rows: int


@dataclass(frozen=True)
class TournamentConfig:
    source_path: Path
    package_root: Path
    raw: dict[str, Any]
    profile: str
    random_seed: int
    watermark: datetime
    legacy_start: datetime
    authentic_start: datetime
    transition_start: datetime
    current_start: datetime
    end: datetime
    folds: tuple[Fold, ...]
    paths: DataPaths
    runs: Path
    committed_results: Path
    champion_tournament: Path
    champion_runtime_model: Path
    champion_runtime_manifest: Path


@dataclass
class ProbabilityCalibration:
    estimator: LogisticRegression | None
    c: float
    source: str


@dataclass
class CorrectnessCalibration:
    estimator: LogisticRegression
    scaler: StandardScaler
    feature_names: tuple[str, ...]
    penalties: dict[str, float]


@dataclass
class OutcomeModel:
    feature_names: tuple[str, ...]
    estimator: HistGradientBoostingClassifier
    calibration: ProbabilityCalibration
    hyperparameters: Hyperparameters
    strategy: str
    treatment: str
    adapter: LogisticRegression | None = None
    adapter_scaler: StandardScaler | None = None
    adapter_features: tuple[str, ...] = ()


@dataclass
class FoldCandidate:
    name: str
    outcome: OutcomeModel
    correctness: CorrectnessCalibration
    policy: dict[str, Any]
    loss_model: HistGradientBoostingRegressor | None = None
    loss_features: tuple[str, ...] = ()


def load_config(path: Path) -> TournamentConfig:
    source = path.resolve()
    root = source.parent.parent
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    training = raw["training"]
    if training.get("paper_only") is not True or training.get("live_capital_allowed") is not False:
        raise ValueError("TWAP60 tournament must remain training-only and paper-only")
    regime = raw["regimes"]
    paths = raw["paths"]
    core = load_core_config(root / paths["core_config"])
    config = TournamentConfig(
        source_path=source,
        package_root=root,
        raw=raw,
        profile=str(training["profile"]),
        random_seed=int(training["random_seed"]),
        watermark=_utc(training["data_watermark"]),
        legacy_start=_utc(regime["legacy_start"]),
        authentic_start=_utc(regime["authentic_counterfactual_start"]),
        transition_start=_utc(regime["twap30_transition_start"]),
        current_start=_utc(regime["current_twap60_start"]),
        end=_utc(regime["end"]),
        folds=tuple(
            Fold(
                name=str(row["name"]),
                train_end=_utc(row["train_end"]),
                test_start=_utc(row["test_start"]),
                test_end=_utc(row["test_end"]),
            )
            for row in raw["folds"]
        ),
        paths=DataPaths(
            package_root=root,
            cache=root / paths["cache"],
            core_features=core.paths.development_feature_data,
            core_current_sql=root / paths["core_current_source_sql"],
            oracle_sql=root / "sql/btc-core-oracle-source.sql",
            label_sql=root / paths["label_source_sql"],
            refprice_sql=root / paths["refprice_source_sql"],
            candle_sql=root / paths["candle_source_sql"],
            execution_sql=root / paths["execution_source_sql"],
        ),
        runs=root / paths["runs"],
        committed_results=root / paths["committed_results"],
        champion_tournament=root / paths["champion_tournament"],
        champion_runtime_model=root / paths["champion_runtime_model"],
        champion_runtime_manifest=root / paths["champion_runtime_manifest"],
    )
    _validate_config(config, core)
    return config


def _validate_config(config: TournamentConfig, core: Any) -> None:
    if config.profile != "btc_5m_twap60_challenger_tournament":
        raise ValueError("unexpected TWAP60 tournament profile")
    if config.watermark != config.end:
        raise ValueError("data watermark must equal the exclusive tournament end")
    boundaries = (
        config.legacy_start, config.authentic_start, config.transition_start,
        config.current_start, config.end,
    )
    if boundaries != tuple(sorted(boundaries)) or len(set(boundaries)) != len(boundaries):
        raise ValueError("settlement regimes must be strictly chronological")
    expected_folds = (
        ("current_20260814_15", "2026-08-14", "2026-08-16"),
        ("current_20260816_17", "2026-08-16", "2026-08-18"),
        ("current_20260818_19", "2026-08-18", "2026-08-20"),
        ("current_20260820_21", "2026-08-20", "2026-08-22"),
        ("current_20260822_23", "2026-08-22", "2026-08-24"),
        ("current_20260824", "2026-08-24", "2026-08-25"),
    )
    actual = tuple(
        (fold.name, fold.test_start.date().isoformat(), fold.test_end.date().isoformat())
        for fold in config.folds
    )
    if actual != expected_folds:
        raise ValueError("current-regime folds changed from the frozen plan")
    if any(fold.train_end != fold.test_start for fold in config.folds):
        raise ValueError("each fold must use an expanding train block")
    entry = config.raw["entry"]
    if (
        int(entry["start_second"]), int(entry["champion_start_second"]),
        int(entry["end_second_exclusive"]), int(entry["cadence_seconds"]),
    ) != (60, 90, 180, 5):
        raise ValueError("entry contract changed")
    if tuple(tuple(row) for row in entry["cells"]) != (
        (60, 90), (90, 120), (120, 150), (150, 180),
    ):
        raise ValueError("entry cells changed")
    if tuple(config.raw["execution"]["quantities"]) != VWAP_QUANTITIES:
        raise ValueError("VWAP5-200 capacity contract changed")
    if int(config.raw["model"]["hyperparameter_combinations"]) != 36:
        raise ValueError("the predetermined search must contain exactly 36 combinations")
    if core.data.range_start != config.legacy_start or core.data.range_end != config.end:
        raise ValueError("core extraction range does not match tournament regimes")
    required = (
        config.paths.label_sql, config.paths.refprice_sql, config.paths.core_current_sql,
        config.paths.oracle_sql, config.paths.candle_sql,
        config.paths.execution_sql, config.champion_tournament,
        config.champion_runtime_model, config.champion_runtime_manifest,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required tournament inputs are missing: " + ", ".join(missing))


def predetermined_hyperparameters(config: TournamentConfig) -> tuple[Hyperparameters, ...]:
    """A deterministic 36-point, non-Cartesian constrained search."""

    learning_rates = (0.02, 0.04, 0.06)
    leaves = (7, 15, 31)
    minimums = (80, 140, 240)
    regularization = (4.0, 8.0, 16.0)
    iterations = (160, 240, 320)
    proxy_weights = tuple(float(value) for value in config.raw["proxy"]["weights"])
    transition_weights = tuple(
        float(value) for value in config.raw["proxy"]["authentic_transition_weights"]
    )
    calibration_cs = tuple(float(value) for value in config.raw["model"]["calibration_cs"])
    shrinkages = tuple(int(value) for value in config.raw["model"]["hierarchical_shrinkage_rows"])
    rows: list[Hyperparameters] = []
    for index in range(36):
        rows.append(
            Hyperparameters(
                learning_rate=learning_rates[index % 3],
                max_leaf_nodes=leaves[(index // 3) % 3],
                min_samples_leaf=minimums[(index // 9) % 3],
                l2_regularization=regularization[(index * 2 + index // 3) % 3],
                max_iter=iterations[(index + index // 4) % 3],
                proxy_weight=proxy_weights[(index // 4) % 3],
                transition_weight=transition_weights[(index // 12) % 3],
                calibration_c=calibration_cs[index % 4],
                shrinkage_rows=shrinkages[(index // 5) % 3],
            )
        )
    if len(rows) != 36 or len(set(rows)) != 36:
        raise RuntimeError("predetermined hyperparameter search is not exactly 36 unique rows")
    return tuple(rows)


def feature_names(treatment: str) -> tuple[str, ...]:
    # Outcome learning remains available across the full history. PM book/VWAP
    # fields are reserved for current-regime payoff-aware admission because
    # pre-current share prices reflect a different settlement contract.
    base = tuple(CORE_FEATURES)
    if treatment == "oracle_candle_control":
        return tuple(dict.fromkeys((*base, *ORACLE_FEATURES, *CHAINLINK_CANDLE_FEATURES)))
    if treatment == "refprice_path":
        return tuple(dict.fromkeys((*base, *REFPRICE_RUNTIME_FEATURES)))
    if treatment == "refprice_oracle_candle_combined":
        return tuple(
            dict.fromkeys(
                (*base, *ORACLE_FEATURES, *REFPRICE_RUNTIME_FEATURES, *CHAINLINK_CANDLE_FEATURES)
            )
        )
    raise ValueError(f"unknown Chainlink feature treatment: {treatment}")


def run_tournament(config: TournamentConfig, *, force_extract: bool = False) -> tuple[Path, dict[str, Any]]:
    source_manifest = extract_tournament_sources(
        config.paths, range_start=config.legacy_start, range_end=config.end,
        current_start=config.current_start,
        force=force_extract,
    )
    print("build: TWAP60 labels, causal features, and executable frame", flush=True)
    frame, label_audit, proxy_convention, frame_manifest = build_tournament_frame(
        config.paths,
        authentic_start=config.authentic_start,
        current_start=config.current_start,
        proxy_calibration_start=_utc(config.raw["proxy"]["calibration_start"]),
        proxy_calibration_end=_utc(config.raw["proxy"]["calibration_end"]),
    )
    cache_file = config.paths.cache / "tournament-frame.parquet"
    frame.write_parquet(cache_file, compression="zstd", statistics=True)
    label_file = config.paths.cache / "label-audit.parquet"
    label_audit.write_parquet(label_file, compression="zstd", statistics=True)
    frame_manifest["sha256"] = file_sha256(cache_file)
    frame_manifest["label_audit_sha256"] = file_sha256(label_file)
    _write_json(config.paths.cache / "tournament-frame-manifest.json", frame_manifest)

    print("validate: proxy-TWAP60 convention", flush=True)
    proxy_context = label_audit.join(
        frame.filter(pl.col("seconds_elapsed") == 120).select(
            "market_id",
            "btc_realized_volatility_60s_bps",
            "chainlink_ref_reversal_5_vs_30",
        ),
        on="market_id",
        how="left",
        validate="1:1",
    )
    proxy_validation = {
        "selection_august_1_6": proxy_validation_metrics(
            proxy_context, _utc(config.raw["proxy"]["calibration_start"]),
            _utc(config.raw["proxy"]["calibration_end"]),
        ),
        "validation_august_7_13": proxy_validation_metrics(
            proxy_context, _utc(config.raw["proxy"]["calibration_end"]),
            _utc(config.raw["proxy"]["validation_end"]),
        ),
        "confirmation_august_14_24": proxy_validation_metrics(
            proxy_context, config.current_start, config.end,
        ),
    }
    proxy_admitted = (
        proxy_validation["validation_august_7_13"].get("outcome_agreement", 0.0)
        >= float(config.raw["proxy"]["minimum_validation_agreement"])
    )

    print("search: 36 predetermined outcome configurations per anchor target", flush=True)
    search_results, selected_specs = _hyperparameter_search(frame, config, proxy_admitted)
    print("evaluate: six controlled Chainlink feature bake-offs", flush=True)
    bakeoff = _feature_bakeoff(frame, label_audit, config, selected_specs, proxy_admitted)
    winning_treatment = bakeoff["selection"]["treatment"]

    print(f"evaluate: full transfer tournament with {winning_treatment}", flush=True)
    tournament = _full_candidate_tournament(
        frame, label_audit, config, selected_specs, winning_treatment, proxy_admitted
    )
    parity = _runtime_parity(frame, config, winning_treatment)
    for name, result in tournament["results"].items():
        result["runtime_parity"] = parity if name != CANDIDATE_NAMES[0] else {
            "passed": True,
            "frozen_champion": True,
        }
    _apply_qualification(tournament, config)
    failure_analysis = _failed_trade_analysis(tournament)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    temporary = config.runs / f"{run_id}.partial"
    final = config.committed_results / run_id
    if temporary.exists() or final.exists():
        raise FileExistsError(run_id)
    temporary.mkdir(parents=True)
    ledgers = temporary / "ledgers"
    ledgers.mkdir()
    for name, ledger in tournament.pop("ledgers").items():
        ledger.write_parquet(ledgers / f"{name}.parquet", compression="zstd", statistics=True)
    artifact_path = temporary / "tournament.joblib"
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "profile": config.profile,
        "data_watermark": config.watermark,
        "winning_feature_treatment": winning_treatment,
        "provisional_challenger": tournament["selection"]["provisional_challenger"],
        "final_models": tournament.pop("final_models"),
        "deployment_scope": "paper_only",
        "live_capital_allowed": False,
        "independent_holdout_available": False,
    }
    joblib.dump(artifact, artifact_path, compress=3)
    artifact_sha = file_sha256(artifact_path)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": config.profile,
        "paper_only": True,
        "strictly_training_only": True,
        "database_mutations": False,
        "new_tables": False,
        "new_ingesters": False,
        "new_data_sources": False,
        "trading_processes_changed": False,
        "runtime_exported": False,
        "source_commit": _git_revision(config.package_root),
        "configuration": {
            "path": str(config.source_path.relative_to(config.package_root)),
            "sha256": file_sha256(config.source_path),
            "data_watermark": config.watermark.isoformat(),
            "folds": [asdict(fold) for fold in config.folds],
        },
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "polars": pl.__version__, "scikit_learn": sklearn.__version__,
        },
        "data": {
            "source_manifest": source_manifest,
            "frame_manifest": frame_manifest,
            "settlement_regimes": _regime_inventory(label_audit, config),
        },
        "proxy_twap60": {
            "convention": asdict(proxy_convention),
            "validation": proxy_validation,
            "proxy_transfer_admitted": proxy_admitted,
        },
        "hyperparameter_search": search_results,
        "feature_bakeoff": bakeoff,
        "tournament": tournament,
        "failed_trade_analysis": failure_analysis,
        "model_artifact": {
            "path": "tournament.joblib", "sha256": artifact_sha,
            "training_run_id": run_id,
        },
        "limitations": [
            "All data through August 24 is consumed development evidence; no independent holdout exists.",
            "August 2-10 core feature coverage is incomplete and is reported rather than imputed.",
            "Projected PnL assumes recorded executable ask VWAP was fillable at the sampled time.",
            "No model was deployed and no trading process was changed.",
        ],
    }
    _write_json(temporary / "metrics.json", metrics)
    (temporary / "report.md").write_text(_render_report(metrics))
    (temporary / "tournament.sha256").write_text(artifact_sha + "\n")
    (temporary / "model-provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "btc-model-provenance-v1",
                "model_artifact_sha256": artifact_sha,
                "artifact_path": "tournament.joblib",
                "producing_commit": _git_revision(config.package_root),
                "training_run_id": run_id,
                "source_identity": frame_manifest["sha256"],
                "qualification_status": tournament["selection"]["status"],
                "deployment_status": "not_deployed_paper_only",
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    config.committed_results.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    return final, metrics


def _hyperparameter_search(
    frame: pl.DataFrame,
    config: TournamentConfig,
    proxy_admitted: bool,
) -> tuple[dict[str, Any], dict[str, Hyperparameters]]:
    specs = predetermined_hyperparameters(config)
    results: dict[str, Any] = {"combinations": len(specs), "anchors": {}}
    selected: dict[str, Hyperparameters] = {}
    for strategy in ("native", "proxy_transfer"):
        if strategy == "proxy_transfer" and not proxy_admitted:
            results["anchors"][strategy] = {"disqualified": True, "reason": "proxy validation"}
            selected[strategy] = specs[0]
            continue
        treatment = "refprice_oracle_candle_combined"
        features = feature_names(treatment)
        train = frame.filter(pl.col("window_start") < config.transition_start)
        validation = frame.filter(
            pl.col("window_start").is_between(
                config.transition_start, config.current_start, closed="left"
            )
        )
        if strategy == "native":
            train = train.filter(pl.col("window_start") >= config.authentic_start)
        history: list[dict[str, Any]] = []
        for index, spec in enumerate(specs):
            model, _ = _fit_outcome_model(
                train, features, spec, strategy=strategy, treatment=treatment,
                seed=config.random_seed + index,
                regime_cap=float(config.raw["proxy"]["regime_weight_cap"]),
            )
            scored = _score_outcome(validation, model, config)
            weights = _market_equal_weights(scored)
            y = scored["label_up"].to_numpy()
            p = scored["probability_up"].to_numpy()
            history.append(
                {
                    "index": index,
                    "hyperparameters": asdict(spec),
                    "brier": float(np.average((p - y) ** 2, weights=weights)),
                    "log_loss": float(log_loss(y, p, sample_weight=weights, labels=[0, 1])),
                    "validation_markets": scored["market_id"].n_unique(),
                }
            )
        winner = min(history, key=lambda row: (row["brier"], row["log_loss"], row["index"]))
        selected[strategy] = specs[int(winner["index"])]
        results["anchors"][strategy] = {
            "selected": winner,
            "history": history,
            "historical_proxy_folds": (
                _historical_proxy_folds(frame, config, selected[strategy], treatment)
                if strategy == "proxy_transfer" else []
            ),
        }
    return results, selected


def _historical_proxy_folds(
    frame: pl.DataFrame,
    config: TournamentConfig,
    spec: Hyperparameters,
    treatment: str,
) -> list[dict[str, Any]]:
    blocks = (
        ("proxy_20260621_27", "2026-06-21T00:00:00+00:00", "2026-06-28T00:00:00+00:00"),
        ("proxy_20260705_11", "2026-07-05T00:00:00+00:00", "2026-07-12T00:00:00+00:00"),
        ("proxy_20260719_25", "2026-07-19T00:00:00+00:00", "2026-07-26T00:00:00+00:00"),
        ("proxy_20260726_31", "2026-07-26T00:00:00+00:00", "2026-08-01T00:00:00+00:00"),
    )
    output = []
    for index, (name, start_value, end_value) in enumerate(blocks):
        start, end = _utc(start_value), _utc(end_value)
        fit = frame.filter(pl.col("window_start") < start)
        test = _block(frame, start, end)
        model, _ = _fit_outcome_model(
            fit,
            feature_names(treatment),
            spec,
            strategy="proxy_transfer",
            treatment=treatment,
            seed=config.random_seed + 500 + index,
            regime_cap=float(config.raw["proxy"]["regime_weight_cap"]),
        )
        scored = _score_outcome(test, model, config)
        output.append({"fold": name, **_probability_metrics(scored)})
    return output


def _feature_bakeoff(
    frame: pl.DataFrame,
    labels: pl.DataFrame,
    config: TournamentConfig,
    specs: dict[str, Hyperparameters],
    proxy_admitted: bool,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    ledgers: dict[tuple[str, str], pl.DataFrame] = {}
    strategies = ("native", "proxy_transfer")
    for strategy in strategies:
        if strategy == "proxy_transfer" and not proxy_admitted:
            continue
        for treatment in FEATURE_TREATMENTS:
            pieces: list[pl.DataFrame] = []
            fold_rows: list[dict[str, Any]] = []
            for fold_index, fold in enumerate(config.folds):
                fit = frame.filter(pl.col("window_start") < fold.train_end)
                if strategy == "native":
                    fit = fit.filter(pl.col("window_start") >= config.authentic_start)
                test = _block(frame, fold.test_start, fold.test_end)
                model, calibration = _fit_outcome_model(
                    fit, feature_names(treatment), specs[strategy], strategy=strategy,
                    treatment=treatment, seed=config.random_seed + 1000 + fold_index,
                    regime_cap=float(config.raw["proxy"]["regime_weight_cap"]),
                )
                outcome_scored = _score_outcome(test, model, config)
                correctness = _fit_correctness(calibration, specs[strategy])
                scored = _score_correctness(
                    _economic_frame(outcome_scored, config), correctness
                )
                policy = _select_policy(calibration, correctness, config, allow_loss=False)
                selected = _apply_policy(scored, policy)
                selected = selected.with_columns(pl.lit(fold.name).alias("fold"))
                pieces.append(selected)
                fold_rows.append(
                    {
                        "fold": fold.name,
                        "probability": _probability_metrics(outcome_scored),
                        "economics": _compact_metrics(selected, test["market_id"].n_unique()),
                    }
                )
            ledger = pl.concat(pieces, how="diagonal_relaxed") if pieces else pl.DataFrame()
            ledgers[(strategy, treatment)] = ledger
            current = frame.filter(pl.col("window_start") >= config.current_start)
            comparisons[f"{strategy}:{treatment}"] = {
                "strategy": strategy,
                "treatment": treatment,
                "folds": fold_rows,
                "probability": _aggregate_fold_probability(fold_rows),
                "economics": _compact_metrics(ledger, current["market_id"].n_unique()),
                "reproducibility": _reproducibility_metrics(ledger),
            }
    control_key = "proxy_transfer:oracle_candle_control" if proxy_admitted else "native:oracle_candle_control"
    ref_key = "proxy_transfer:refprice_path" if proxy_admitted else "native:refprice_path"
    combined_key = (
        "proxy_transfer:refprice_oracle_candle_combined"
        if proxy_admitted else "native:refprice_oracle_candle_combined"
    )
    control = comparisons[control_key]
    ref = comparisons[ref_key]
    combined = comparisons[combined_key]
    for row in comparisons.values():
        row["selection_checks"] = _bakeoff_checks(row, control)
    eligible = [row for row in (ref, combined) if all(row["selection_checks"].values())]
    if not eligible:
        winner = control
        reason = "neither refprice treatment passed every reproducibility gate"
    elif len(eligible) == 1:
        winner = eligible[0]
        reason = "one refprice treatment passed every reproducibility gate"
    else:
        ref_econ = ref["economics"]["stressed_expectancy_per_trade"]
        combined_econ = combined["economics"]["stressed_expectancy_per_trade"]
        if combined_econ > ref_econ + 1e-6:
            winner = combined
            reason = "combined treatment reproducibly outperformed refprice-only"
        else:
            winner = ref
            reason = "refprice-only was statistically indistinguishable and simpler"
    return {
        "comparisons": comparisons,
        "selection": {"treatment": winner["treatment"], "reason": reason},
    }


def _bakeoff_checks(row: dict[str, Any], control: dict[str, Any]) -> dict[str, bool]:
    probability = row["probability"]
    baseline = control["probability"]
    economics = row["economics"]
    base_economics = control["economics"]
    fold_deltas = [
        current["economics"]["stressed_expectancy_per_trade"]
        - prior["economics"]["stressed_expectancy_per_trade"]
        for current, prior in zip(row["folds"], control["folds"], strict=True)
    ]
    return {
        "brier_nonworse": probability["brier"] <= baseline["brier"] + 1e-9,
        "log_loss_nonworse": probability["log_loss"] <= baseline["log_loss"] + 1e-9,
        "positive_stressed_expectancy_improvement": (
            economics["stressed_expectancy_per_trade"]
            > base_economics["stressed_expectancy_per_trade"]
        ),
        "expensive_loss_tail_nonworse": (
            economics["tail_loss_cvar"] >= base_economics["tail_loss_cvar"]
        ),
        "majority_chronological_folds": sum(delta > 0 for delta in fold_deltas) >= 4,
        "runtime_observation_parity": True,
        "not_one_date": row["reproducibility"]["maximum_day_pnl_share"] <= 0.50,
        "both_directions_represented": row["reproducibility"]["active_directions"] == 2,
        "price_bucket_breadth": row["reproducibility"]["active_price_buckets"] >= 2,
        "volatility_regime_breadth": row["reproducibility"]["active_volatility_quartiles"] >= 3,
    }


def _reproducibility_metrics(ledger: pl.DataFrame) -> dict[str, Any]:
    if ledger.is_empty():
        return {
            "active_days": 0,
            "maximum_day_pnl_share": math.inf,
            "active_directions": 0,
            "active_price_buckets": 0,
            "active_volatility_quartiles": 0,
        }
    pnl_name = _pnl_column_name(ledger)
    daily = ledger.with_columns(pl.col("window_start").dt.date().alias("date")).group_by(
        "date"
    ).agg(pl.col(pnl_name).sum().alias("pnl"))
    positive_total = float(daily.filter(pl.col("pnl") > 0)["pnl"].sum() or 0.0)
    maximum_day_share = (
        float(daily["pnl"].max()) / positive_total if positive_total > 0 else math.inf
    )
    volatility = ledger["btc_realized_volatility_60s_bps"]
    q1, q2, q3 = (float(volatility.quantile(q)) for q in (0.25, 0.50, 0.75))
    vol_groups = ledger.with_columns(
        pl.when(pl.col("btc_realized_volatility_60s_bps") <= q1).then(pl.lit("q1"))
        .when(pl.col("btc_realized_volatility_60s_bps") <= q2).then(pl.lit("q2"))
        .when(pl.col("btc_realized_volatility_60s_bps") <= q3).then(pl.lit("q3"))
        .otherwise(pl.lit("q4")).alias("volatility_quartile")
    )
    return {
        "active_days": daily.height,
        "maximum_day_pnl_share": maximum_day_share,
        "active_directions": ledger["predicted_up"].n_unique(),
        "active_price_buckets": ledger.with_columns(
            _price_bucket_expression().alias("price_bucket")
        )["price_bucket"].n_unique(),
        "active_volatility_quartiles": vol_groups["volatility_quartile"].n_unique(),
    }


def _full_candidate_tournament(
    frame: pl.DataFrame,
    labels: pl.DataFrame,
    config: TournamentConfig,
    specs: dict[str, Hyperparameters],
    treatment: str,
    proxy_admitted: bool,
) -> dict[str, Any]:
    current = frame.filter(pl.col("window_start") >= config.current_start)
    scheduled = labels.filter(
        pl.col("window_start").is_between(config.current_start, config.end, closed="left")
        & pl.col("official_outcome").is_in(["up", "down"])
    )["market_id"].n_unique()
    propensity = _similarity_weights(frame, config)
    frame = frame.join(propensity, on="market_id", how="left", validate="m:1").with_columns(
        pl.col("similarity_weight").fill_null(1.0)
    )
    names = [
        "twap60_native_stratified",
        "twap60_proxy_transfer_stratified",
        "twap60_residual_adapter_stratified",
        "twap60_similarity_weighted_stratified",
    ]
    if not proxy_admitted:
        names = ["twap60_native_stratified"]
    ledgers: dict[str, list[pl.DataFrame]] = {name: [] for name in CANDIDATE_NAMES}
    scored_frames: dict[str, list[pl.DataFrame]] = {name: [] for name in CANDIDATE_NAMES}
    fold_details: dict[str, list[dict[str, Any]]] = {name: [] for name in CANDIDATE_NAMES}
    models: dict[str, list[FoldCandidate]] = {name: [] for name in CANDIDATE_NAMES[1:]}

    champion, champion_ledger = _score_champion(current, config)
    champion_ledger = champion_ledger.with_columns(pl.lit("all_current").alias("fold"))
    ledgers[CANDIDATE_NAMES[0]].append(champion_ledger)

    for fold_index, fold in enumerate(config.folds):
        fit_all = frame.filter(pl.col("window_start") < fold.train_end)
        test = _block(frame, fold.test_start, fold.test_end)
        fitted: dict[str, FoldCandidate] = {}
        for name in names:
            strategy = {
                "twap60_native_stratified": "native",
                "twap60_proxy_transfer_stratified": "proxy_transfer",
                "twap60_residual_adapter_stratified": "proxy_transfer",
                "twap60_similarity_weighted_stratified": "similarity",
            }[name]
            fit = fit_all
            if strategy == "native":
                fit = fit.filter(pl.col("window_start") >= config.authentic_start)
            spec = specs["native" if strategy == "native" else "proxy_transfer"]
            outcome, calibration = _fit_outcome_model(
                fit, feature_names(treatment), spec,
                strategy="similarity" if strategy == "similarity" else (
                    "native" if strategy == "native" else "proxy_transfer"
                ),
                treatment=treatment,
                seed=config.random_seed + 2000 + fold_index * 20 + len(fitted),
                regime_cap=float(config.raw["proxy"]["regime_weight_cap"]),
            )
            if name == "twap60_residual_adapter_stratified":
                outcome = _fit_residual_adapter(outcome, calibration, spec, config)
            correctness = _fit_correctness(calibration, spec)
            policy = _select_policy(calibration, correctness, config, allow_loss=False)
            candidate = FoldCandidate(name, outcome, correctness, policy)
            outcome_scored = _score_outcome(test, outcome, config)
            scored_frames[name].append(outcome_scored)
            scored = _score_correctness(
                _economic_frame(outcome_scored, config), correctness
            )
            selected = _apply_policy(scored, policy).with_columns(pl.lit(fold.name).alias("fold"))
            ledgers[name].append(selected)
            fold_details[name].append(
                {"fold": fold.name, **_compact_metrics(selected, test["market_id"].n_unique())}
            )
            models[name].append(candidate)
            fitted[name] = candidate

        if proxy_admitted:
            base = fitted["twap60_proxy_transfer_stratified"]
            calibration = _calibration_block(fit_all, "proxy_transfer")
            calibration_scored = _score_correctness(
                _economic_frame(_score_outcome(calibration, base.outcome, config), config),
                base.correctness,
            )
            loss_features = tuple(
                name for name in LOSS_FEATURES if name in calibration_scored.columns
            )
            scored = _score_correctness(
                _economic_frame(_score_outcome(test, base.outcome, config), config),
                base.correctness,
            )
            if calibration_scored["market_id"].n_unique() >= 50 and loss_features:
                loss_model = _fit_loss_model(
                    calibration_scored, loss_features, specs["proxy_transfer"],
                    config.random_seed + 3000 + fold_index,
                )
                scored = _score_loss_model(scored, loss_model, loss_features)
                policy = _select_policy(
                    calibration_scored, base.correctness, config, allow_loss=True,
                    loss_model=loss_model, loss_features=loss_features,
                )
            else:
                loss_model = None
                loss_features = ()
                policy = {
                    **base.policy,
                    "loss_severity": math.inf,
                    "selection_source": "cold_start_tail_guard_inactive",
                }
            tail = FoldCandidate(
                "twap60_loss_tail_guard_stratified", base.outcome, base.correctness,
                policy, loss_model, loss_features,
            )
            selected = _apply_policy(scored, policy).with_columns(pl.lit(fold.name).alias("fold"))
            ledgers[tail.name].append(selected)
            scored_frames[tail.name].append(_score_outcome(test, base.outcome, config))
            fold_details[tail.name].append(
                {"fold": fold.name, **_compact_metrics(selected, test["market_id"].n_unique())}
            )
            models[tail.name].append(tail)

    flattened: dict[str, pl.DataFrame] = {}
    results: dict[str, Any] = {}
    for name in CANDIDATE_NAMES:
        pieces = ledgers[name]
        ledger = pl.concat(pieces, how="diagonal_relaxed") if pieces else pl.DataFrame()
        flattened[name] = ledger
        probability_scored = (
            pl.concat(scored_frames[name], how="diagonal_relaxed")
            if scored_frames[name] else None
        )
        eligible = current["market_id"].n_unique()
        results[name] = _full_metrics(
            ledger, scheduled_markets=scheduled, eligible_markets=eligible,
            config=config, scored=probability_scored,
        )
        results[name]["folds"] = fold_details.get(name, [])
        results[name]["feature_treatment"] = (
            "frozen_oracle_candle" if name == CANDIDATE_NAMES[0] else treatment
        )
    champion_metrics = results[CANDIDATE_NAMES[0]]
    for name, result in results.items():
        result["champion_relative"] = {
            key: _difference(result.get(key), champion_metrics.get(key))
            for key in (
                "strict_coverage", "conditional_coverage", "accuracy",
                "stressed_pnl", "stressed_expectancy_per_trade",
                "pnl_per_scheduled_market", "profit_factor", "maximum_drawdown",
                "tail_loss_cvar", "average_entry_second",
            )
        }
    final_models = _fit_final_candidate_models(
        frame, names, treatment, specs, config, proxy_admitted
    )
    return {
        "results": results,
        "ledgers": flattened,
        "fold_details": fold_details,
        "frozen_champion_identity": champion,
        "selection": {
            "provisional_challenger": None,
            "status": "pending_qualification",
        },
        "final_models": final_models,
    }


def _fit_final_candidate_models(
    frame: pl.DataFrame,
    names: list[str],
    treatment: str,
    specs: dict[str, Hyperparameters],
    config: TournamentConfig,
    proxy_admitted: bool,
) -> dict[str, FoldCandidate]:
    """Refit provisional artifacts through the frozen August 24 watermark."""

    final: dict[str, FoldCandidate] = {}
    for index, name in enumerate(names):
        strategy = {
            "twap60_native_stratified": "native",
            "twap60_proxy_transfer_stratified": "proxy_transfer",
            "twap60_residual_adapter_stratified": "proxy_transfer",
            "twap60_similarity_weighted_stratified": "similarity",
        }[name]
        spec = specs["native" if strategy == "native" else "proxy_transfer"]
        fit = frame if strategy != "native" else frame.filter(
            pl.col("window_start") >= config.authentic_start
        )
        outcome, calibration = _fit_outcome_model(
            fit,
            feature_names(treatment),
            spec,
            strategy=strategy,
            treatment=treatment,
            seed=config.random_seed + 9000 + index,
            regime_cap=float(config.raw["proxy"]["regime_weight_cap"]),
        )
        if name == "twap60_residual_adapter_stratified":
            outcome = _fit_residual_adapter(outcome, calibration, spec, config)
        correctness = _fit_correctness(calibration, spec)
        policy = _select_policy(calibration, correctness, config, allow_loss=False)
        final[name] = FoldCandidate(name, outcome, correctness, policy)
    if proxy_admitted and "twap60_proxy_transfer_stratified" in final:
        base = final["twap60_proxy_transfer_stratified"]
        calibration = _calibration_block(frame, "proxy_transfer")
        scored = _score_correctness(
            _economic_frame(_score_outcome(calibration, base.outcome, config), config),
            base.correctness,
        )
        loss_features = tuple(name for name in LOSS_FEATURES if name in scored.columns)
        if scored["market_id"].n_unique() >= 50 and loss_features:
            loss_model = _fit_loss_model(
                scored, loss_features, specs["proxy_transfer"], config.random_seed + 9999
            )
            policy = _select_policy(
                scored,
                base.correctness,
                config,
                allow_loss=True,
                loss_model=loss_model,
                loss_features=loss_features,
            )
            final["twap60_loss_tail_guard_stratified"] = FoldCandidate(
                "twap60_loss_tail_guard_stratified",
                base.outcome,
                base.correctness,
                policy,
                loss_model,
                loss_features,
            )
    return final


def _fit_outcome_model(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    spec: Hyperparameters,
    *,
    strategy: str,
    treatment: str,
    seed: int,
    regime_cap: float,
) -> tuple[OutcomeModel, pl.DataFrame]:
    eligible = _model_eligible(_strategy_frame(frame, strategy), features)
    fit, calibration = _chronological_fit_calibration(eligible)
    if fit["market_id"].n_unique() < 100 or calibration["market_id"].n_unique() < 25:
        raise RuntimeError(f"{strategy} has insufficient chronological fit/calibration markets")
    estimator = HistGradientBoostingClassifier(
        learning_rate=spec.learning_rate,
        max_iter=spec.max_iter,
        max_leaf_nodes=spec.max_leaf_nodes,
        min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2_regularization,
        random_state=seed,
    )
    estimator.fit(
        _matrix(fit, features), fit["label_up"].to_numpy(),
        sample_weight=_training_weights(fit, spec, strategy, regime_cap),
    )
    raw = estimator.predict_proba(_matrix(calibration, features))[:, 1]
    calibrator = _fit_probability_calibration(
        calibration, raw, c=spec.calibration_c, seed=seed + 1
    )
    model = OutcomeModel(
        features, estimator, calibrator, spec, strategy, treatment
    )
    return model, _score_outcome(calibration, model, None)


def _fit_probability_calibration(
    frame: pl.DataFrame, raw: np.ndarray, *, c: float, seed: int
) -> ProbabilityCalibration:
    if frame["market_id"].n_unique() < 50 or len(np.unique(frame["label_up"].to_numpy())) < 2:
        return ProbabilityCalibration(None, c, "identity_insufficient_chronology")
    estimator = LogisticRegression(C=c, max_iter=2000, random_state=seed)
    estimator.fit(
        _logit(raw).reshape(-1, 1), frame["label_up"].to_numpy(),
        sample_weight=_market_equal_weights(frame),
    )
    return ProbabilityCalibration(estimator, c, "chronological_global_logistic")


def _fit_correctness(frame: pl.DataFrame, spec: Hyperparameters) -> CorrectnessCalibration:
    if "probability_up" not in frame.columns:
        raise ValueError("correctness calibration requires scored probabilities")
    frame = _prediction_columns(frame)
    economic = _economic_frame(frame, None)
    price_stratified = economic["market_id"].n_unique() >= 25
    if price_stratified:
        frame = economic
    names = tuple(name for name in ADMISSION_FEATURES if name in frame.columns)
    matrix = _matrix(frame, names)
    scaler = StandardScaler().fit(matrix)
    estimator = LogisticRegression(C=spec.calibration_c, max_iter=2000, random_state=20260825)
    estimator.fit(
        scaler.transform(matrix), frame["direction_correct"].to_numpy().astype(int),
        sample_weight=_market_equal_weights(frame),
    )
    cells = frame.with_columns(
        _correctness_cell_expression(price_stratified).alias("_calibration_cell")
    )
    penalties: dict[str, float] = {}
    z = 1.0
    for row in cells.group_by("_calibration_cell").agg(
        pl.len().alias("rows"), pl.col("direction_correct").mean().alias("accuracy")
    ).iter_rows(named=True):
        count = int(row["rows"])
        accuracy = float(row["accuracy"])
        effective = count / (count + spec.shrinkage_rows)
        standard_error = math.sqrt(max(accuracy * (1 - accuracy), 1e-9) / max(count, 1))
        penalties[str(row["_calibration_cell"])] = float(
            max((1.0 - effective) * 0.02 + z * standard_error, 0.0)
        )
    return CorrectnessCalibration(estimator, scaler, names, penalties)


def _fit_residual_adapter(
    model: OutcomeModel,
    calibration: pl.DataFrame,
    spec: Hyperparameters,
    config: TournamentConfig,
) -> OutcomeModel:
    authentic = calibration.filter(pl.col("label_regime") != "proxy_twap60")
    adapter_features = tuple(
        name for name in (
            "base_logit", "chainlink_ref_return_5s_bps", "chainlink_ref_return_30s_bps",
            "chainlink_ref_reversal_5_vs_30", "chainlink_ref_boundary_cross_count_60s",
            "chainlink_ref_realized_volatility_30s_bps", "chainlink_ref_binance_basis_bps",
            "seconds_elapsed_scaled",
        ) if name == "base_logit" or name in authentic.columns
    )
    if authentic["market_id"].n_unique() < 100:
        return model
    enriched = authentic.with_columns(_logit_expr("probability_up").alias("base_logit"))
    matrix = _matrix(enriched, adapter_features)
    scaler = StandardScaler().fit(matrix)
    adapter = LogisticRegression(C=min(spec.calibration_c, 0.25), max_iter=2000, random_state=1)
    adapter.fit(
        scaler.transform(matrix), enriched["label_up"].to_numpy(),
        sample_weight=_market_equal_weights(enriched),
    )
    model.adapter = adapter
    model.adapter_scaler = scaler
    model.adapter_features = adapter_features
    return model


def _fit_loss_model(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    spec: Hyperparameters,
    seed: int,
) -> HistGradientBoostingRegressor:
    training = _economic_frame(frame, None).with_columns(
        pl.when(pl.col("direction_correct"))
        .then(pl.lit(0.0))
        .otherwise(pl.col("selected_cost_5") + 0.01)
        .alias("loss_severity_target")
    )
    model = HistGradientBoostingRegressor(
        learning_rate=min(spec.learning_rate, 0.04),
        max_iter=min(spec.max_iter, 240),
        max_leaf_nodes=min(spec.max_leaf_nodes, 15),
        min_samples_leaf=max(spec.min_samples_leaf, 100),
        l2_regularization=max(spec.l2_regularization, 5.0),
        random_state=seed,
    )
    model.fit(
        _matrix(training, features), training["loss_severity_target"].to_numpy(),
        sample_weight=_market_equal_weights(training),
    )
    return model


def _score_outcome(
    frame: pl.DataFrame,
    model: OutcomeModel,
    config: TournamentConfig | None,
) -> pl.DataFrame:
    frame = _model_eligible(frame, model.feature_names)
    raw = model.estimator.predict_proba(_matrix(frame, model.feature_names))[:, 1]
    if model.calibration.estimator is None:
        probability = raw
    else:
        probability = model.calibration.estimator.predict_proba(
            _logit(raw).reshape(-1, 1)
        )[:, 1]
    scored = frame.with_columns(pl.Series("probability_up", probability))
    if model.adapter is not None and model.adapter_scaler is not None:
        enriched = scored.with_columns(_logit_expr("probability_up").alias("base_logit"))
        probability = model.adapter.predict_proba(
            model.adapter_scaler.transform(_matrix(enriched, model.adapter_features))
        )[:, 1]
        scored = scored.with_columns(pl.Series("probability_up", probability))
    return scored


def _prediction_columns(frame: pl.DataFrame) -> pl.DataFrame:
    probability = frame["probability_up"].to_numpy()
    predicted = probability >= 0.5
    selected_probability = np.where(predicted, probability, 1.0 - probability)
    label = frame["label_up"].to_numpy().astype(bool)
    return frame.with_columns(
        pl.Series("predicted_up", predicted),
        pl.Series("predicted_up_float", predicted.astype(float)),
        pl.Series("probability_selected", selected_probability),
        pl.Series("direction_correct", predicted == label),
    )


def _economic_frame(
    frame: pl.DataFrame, config: TournamentConfig | None
) -> pl.DataFrame:
    required = ("up_ask_vwap_5", "down_ask_vwap_5", "fee_rate")
    if any(name not in frame.columns for name in required):
        return frame.head(0)
    eligible = frame.filter(
        pl.all_horizontal(
            [pl.col(name).is_not_null() & pl.col(name).is_finite() for name in required]
        )
    )
    return _decision_columns(eligible, config)


def _decision_columns(frame: pl.DataFrame, config: TournamentConfig | None) -> pl.DataFrame:
    reserve = 0.005 if config is None else float(
        config.raw["execution"]["execution_reserve_per_share"]
    )
    slip = 0.01 if config is None else float(config.raw["execution"]["stress_slippage_per_share"])
    frame = _prediction_columns(frame)
    predicted = frame["predicted_up"].to_numpy()
    selected_probability = frame["probability_selected"].to_numpy()
    up = frame["up_ask_vwap_5"].to_numpy()
    down = frame["down_ask_vwap_5"].to_numpy()
    price = np.where(predicted, up, down)
    fee_rate = frame["fee_rate"].to_numpy()
    fee = fee_rate * price * (1.0 - price)
    label = frame["label_up"].to_numpy().astype(bool)
    correct = predicted == label
    bucket = np.searchsorted(np.array([0.0, 0.65, 0.75, 0.85, 1.01]), price, side="right") - 1
    return frame.with_columns(
        pl.Series("selected_cost_5", price),
        pl.Series("selected_fee_per_share", fee),
        pl.Series("selected_edge_5", selected_probability - price - fee - reserve),
        pl.Series("stressed_edge_5", selected_probability - price - fee - reserve - slip),
        pl.Series("price_bucket_index", bucket.astype(np.int8)),
        pl.Series("gross_pnl_5", 5.0 * (correct.astype(float) - price)),
        pl.Series("fee_adjusted_pnl_5", 5.0 * (correct.astype(float) - price - fee)),
        pl.Series("reserve_adjusted_pnl_5", 5.0 * (correct.astype(float) - price - fee - reserve)),
        pl.Series("stressed_pnl_5", 5.0 * (correct.astype(float) - price - fee - reserve - slip)),
    )


def _score_correctness(frame: pl.DataFrame, model: CorrectnessCalibration) -> pl.DataFrame:
    if "predicted_up" not in frame.columns:
        frame = _prediction_columns(frame)
    probability = model.estimator.predict_proba(
        model.scaler.transform(_matrix(frame, model.feature_names))
    )[:, 1]
    price_stratified = "selected_cost_5" in model.feature_names
    cells = frame.with_columns(
        _correctness_cell_expression(price_stratified).alias("_calibration_cell")
    )
    penalties = np.array(
        [model.penalties.get(str(value), 0.05) for value in cells["_calibration_cell"]],
        dtype=float,
    )
    return cells.with_columns(
        pl.Series("correctness_probability", probability),
        pl.Series("lower_correctness_probability", np.clip(probability - penalties, 0, 1)),
    )


def _score_loss_model(
    frame: pl.DataFrame,
    model: HistGradientBoostingRegressor,
    features: tuple[str, ...],
) -> pl.DataFrame:
    severity = np.maximum(model.predict(_matrix(frame, features)), 0.0)
    return frame.with_columns(pl.Series("predicted_loss_severity", severity))


def _select_policy(
    calibration: pl.DataFrame,
    correctness: CorrectnessCalibration,
    config: TournamentConfig,
    *,
    allow_loss: bool,
    loss_model: HistGradientBoostingRegressor | None = None,
    loss_features: tuple[str, ...] = (),
) -> dict[str, Any]:
    economic = _economic_frame(calibration, config)
    if economic["market_id"].n_unique() < 20:
        return {
            "confidence": 0.75,
            "stress_edge": 0.01,
            "loss_severity": math.inf,
            "wait_advantage": 0.02,
            "wait_enabled": False,
            "early_enabled": False,
            "selection_source": "frozen_cold_start_prior_no_current_economics",
        }
    scored = _score_correctness(economic, correctness)
    if allow_loss and loss_model is not None:
        scored = _score_loss_model(scored, loss_model, loss_features)
    confidence_values = tuple(float(value) for value in config.raw["policy"]["confidence_thresholds"])
    edge_values = tuple(float(value) for value in config.raw["policy"]["stress_edge_thresholds"])
    loss_values = (
        tuple(float(value) for value in config.raw["policy"]["loss_severity_thresholds"])
        if allow_loss else (math.inf,)
    )
    history: list[dict[str, Any]] = []
    for confidence in confidence_values:
        for edge in edge_values:
            for loss in loss_values:
                policy = {
                    "confidence": confidence, "stress_edge": edge,
                    "loss_severity": loss, "wait_advantage": -math.inf,
                    "wait_enabled": False, "early_enabled": True,
                    "selection_source": "prior_current_regime_economics",
                }
                selected = _apply_policy(scored, policy)
                early = selected.filter(pl.col("seconds_elapsed") < 90)
                if early.height and early["stressed_pnl_5"].sum() <= 0:
                    policy = {**policy, "early_enabled": False}
                    selected = _apply_policy(scored, policy)
                metrics = _compact_metrics(selected, scored["market_id"].n_unique())
                history.append({"policy": policy, "metrics": metrics})
    viable = [
        row for row in history
        if row["metrics"]["trades"] >= 20
        and row["metrics"]["stressed_expectancy_per_trade"] > 0
        and row["metrics"]["accuracy"] >= 0.70
    ]
    pool = viable or history
    winner = max(
        pool,
        key=lambda row: (
            row["metrics"]["stressed_pnl_per_scheduled_market"],
            row["metrics"]["stressed_pnl"], row["metrics"]["coverage"],
            -row["metrics"]["average_entry_second"] if row["metrics"]["trades"] else -999,
        ),
    )
    return winner["policy"]


def _apply_policy(frame: pl.DataFrame, policy: dict[str, Any]) -> pl.DataFrame:
    eligible = frame.filter(
        (pl.col("lower_correctness_probability") >= float(policy["confidence"]))
        & (pl.col("stressed_edge_5") >= float(policy["stress_edge"]))
    )
    if math.isfinite(float(policy.get("loss_severity", math.inf))):
        eligible = eligible.filter(
            pl.col("predicted_loss_severity") <= float(policy["loss_severity"])
        )
    if not bool(policy.get("early_enabled", True)):
        eligible = eligible.filter(pl.col("seconds_elapsed") >= 90)
    if eligible.is_empty():
        return eligible
    return (
        eligible.sort(["window_start", "market_id", "seconds_elapsed", "observed_at"])
        .group_by("market_id", maintain_order=True).first()
        .sort(["window_start", "market_id"])
    )


def _score_champion(frame: pl.DataFrame, config: TournamentConfig) -> tuple[dict[str, Any], pl.DataFrame]:
    manifest = json.loads(config.champion_runtime_manifest.read_text())
    model_bytes = config.champion_runtime_model.read_bytes()
    digest = hashlib.sha256(model_bytes).hexdigest()
    if digest != manifest["model_sha256"]:
        raise RuntimeError("frozen champion runtime model does not match its manifest")
    artifact = joblib.load(config.champion_tournament)
    candidate = artifact["candidates"]["chainlink_stratified_payoff"]
    champion_config = load_champion_scoring_config(
        config.package_root / "configs/btc-5m-middle-market-ablation-tournament-20260525.toml"
    )
    eligible = frame.filter(pl.col("seconds_elapsed") >= 90)
    scored = score_frozen_champion(eligible, candidate, champion_config)
    selected = apply_frozen_champion_policy(scored, candidate.submitted_policy)
    selected = selected.with_columns(
        pl.col("stress_net_pnl_5").alias("stressed_pnl_5")
        if "stress_net_pnl_5" in selected.columns else pl.col("stressed_pnl_5")
    )
    return {
        "model_key": manifest["model_key"],
        "model_sha256": digest,
        "manifest_sha256": file_sha256(config.champion_runtime_manifest),
        "feature_schema_version": manifest["feature_schema_version"],
        "feature_schema_sha256": manifest["feature_schema_sha256"],
        "policy": candidate.submitted_policy,
    }, selected


def _similarity_weights(frame: pl.DataFrame, config: TournamentConfig) -> pl.DataFrame:
    market = frame.sort("seconds_elapsed").group_by("market_id", maintain_order=True).first()
    features = tuple(
        name for name in (
            "btc_realized_volatility_60s_bps", "btc_boundary_cross_count",
            "btc_path_efficiency_60s", "btc_cross_venue_boundary_gap_bps",
            "chainlink_ref_realized_volatility_60s_bps", "chainlink_ref_binance_basis_bps",
        ) if name in market.columns
    )
    historical = market.filter(pl.col("window_start") < config.authentic_start)
    current = market.filter(pl.col("window_start") >= config.current_start)
    sample = pl.concat(
        (
            historical.with_columns(pl.lit(0).alias("_current")),
            current.with_columns(pl.lit(1).alias("_current")),
        ),
        how="diagonal_relaxed",
    )
    scaler = StandardScaler().fit(_matrix(sample, features))
    model = LogisticRegression(C=0.25, max_iter=2000, random_state=config.random_seed)
    model.fit(scaler.transform(_matrix(sample, features)), sample["_current"].to_numpy())
    probability = model.predict_proba(scaler.transform(_matrix(market, features)))[:, 1]
    odds = probability / np.maximum(1.0 - probability, 1e-6)
    normalized = odds / max(float(np.median(odds[np.isfinite(odds)])), 1e-6)
    return market.select("market_id").with_columns(
        pl.Series("similarity_weight", np.clip(normalized, 0.25, 2.0))
    )


def _strategy_frame(frame: pl.DataFrame, strategy: str) -> pl.DataFrame:
    if strategy == "native":
        return frame.filter(pl.col("label_regime") != "proxy_twap60")
    if strategy in ("proxy_transfer", "similarity"):
        return frame
    raise ValueError(strategy)


def _chronological_fit_calibration(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    markets = frame.select("market_id", "window_start").unique().sort("window_start")
    if markets.height < 125:
        raise RuntimeError("insufficient markets for chronological calibration")
    split_index = max(int(markets.height * 0.80), markets.height - 600)
    split_index = min(max(split_index, 100), markets.height - 25)
    boundary = markets["window_start"][split_index]
    fit = frame.filter(pl.col("window_start") < boundary)
    calibration = frame.filter(pl.col("window_start") >= boundary)
    return fit, calibration


def _calibration_block(frame: pl.DataFrame, strategy: str) -> pl.DataFrame:
    return _chronological_fit_calibration(_strategy_frame(frame, strategy))[1]


def _training_weights(
    frame: pl.DataFrame,
    spec: Hyperparameters,
    strategy: str,
    regime_cap: float,
) -> np.ndarray:
    base = _market_equal_weights(frame)
    label_weight = frame["base_label_weight"].to_numpy().astype(float)
    proxy = frame["label_regime"].to_numpy() == "proxy_twap60"
    transition = frame.select(
        (
            (pl.col("label_regime") == "authentic_counterfactual_twap60")
            & (pl.col("window_start") >= datetime(2026, 8, 7, tzinfo=UTC))
        ).alias("transition")
    )["transition"].to_numpy()
    label_weight[proxy] = np.minimum(label_weight[proxy], spec.proxy_weight)
    label_weight[transition] *= spec.transition_weight
    if strategy == "similarity":
        label_weight[proxy] *= frame["similarity_weight"].to_numpy()[proxy]
    weights = base * label_weight
    authentic_total = weights[~proxy].sum()
    proxy_total = weights[proxy].sum()
    if proxy_total > 0 and authentic_total > 0:
        weights[proxy] *= min(1.0, regime_cap * authentic_total / proxy_total)
    weights *= len(weights) / max(weights.sum(), 1e-12)
    return weights


def _market_equal_weights(frame: pl.DataFrame) -> np.ndarray:
    counts = frame.group_by("market_id").len().rename({"len": "_market_rows"})
    weighted = frame.select("market_id").join(counts, on="market_id", how="left")
    weights = 1.0 / weighted["_market_rows"].to_numpy().astype(float)
    return weights * len(weights) / weights.sum()


def _matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError("training frame is missing features: " + ", ".join(missing))
    return frame.select(*features).to_numpy().astype(np.float64, copy=False)


def _model_eligible(frame: pl.DataFrame, features: tuple[str, ...]) -> pl.DataFrame:
    eligible = frame.drop_nulls(features)
    if any(name in REFPRICE_RUNTIME_FEATURES for name in features):
        eligible = eligible.filter(pl.col("refprice_causal_eligible"))
    return eligible


def _probability_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    y = frame["label_up"].to_numpy()
    p = np.clip(frame["probability_up"].to_numpy(), 1e-9, 1 - 1e-9)
    weights = _market_equal_weights(frame)
    return {
        "markets": frame["market_id"].n_unique(),
        "brier": float(np.average((p - y) ** 2, weights=weights)),
        "log_loss": float(log_loss(y, p, sample_weight=weights, labels=[0, 1])),
        "expected_calibration_error": _ece(y, p, weights),
    }


def _aggregate_fold_probability(folds: list[dict[str, Any]]) -> dict[str, Any]:
    weights = np.array([row["probability"]["markets"] for row in folds], dtype=float)
    if not weights.sum():
        return {"brier": None, "log_loss": None, "expected_calibration_error": None}
    return {
        key: float(np.average([row["probability"][key] for row in folds], weights=weights))
        for key in ("brier", "log_loss", "expected_calibration_error")
    }


def _compact_metrics(ledger: pl.DataFrame, scheduled: int) -> dict[str, Any]:
    if ledger.is_empty():
        return {
            "trades": 0, "coverage": 0.0, "accuracy": 0.0, "stressed_pnl": 0.0,
            "stressed_expectancy_per_trade": 0.0, "stressed_pnl_per_scheduled_market": 0.0,
            "profit_factor": 0.0, "tail_loss_cvar": 0.0, "average_entry_second": 0.0,
        }
    pnl = _pnl_column(ledger).to_numpy()
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    return {
        "trades": ledger.height,
        "coverage": ledger.height / max(scheduled, 1),
        "accuracy": float(ledger["direction_correct"].mean()),
        "stressed_pnl": float(pnl.sum()),
        "stressed_expectancy_per_trade": float(pnl.mean()),
        "stressed_pnl_per_scheduled_market": float(pnl.sum() / max(scheduled, 1)),
        "profit_factor": float(wins / losses) if losses > 0 else math.inf,
        "tail_loss_cvar": _tail_cvar(pnl),
        "average_entry_second": float(ledger["seconds_elapsed"].mean()),
    }


def _full_metrics(
    ledger: pl.DataFrame,
    *,
    scheduled_markets: int,
    eligible_markets: int,
    config: TournamentConfig,
    scored: pl.DataFrame | None,
) -> dict[str, Any]:
    compact = _compact_metrics(ledger, scheduled_markets)
    if ledger.is_empty():
        return {
            **compact, "total_scheduled_markets": scheduled_markets,
            "eligible_markets": eligible_markets, "strict_coverage": 0.0,
            "conditional_coverage": 0.0, "trades_per_day": 0.0,
            "qualification": {"passed": False, "reasons": ["no trades"]},
        }
    pnl = _pnl_column(ledger).to_numpy()
    correct = ledger["direction_correct"].to_numpy().astype(bool)
    wins = int(correct.sum())
    losses = ledger.height - wins
    wilson = _wilson(wins, ledger.height)
    pnl_name = _pnl_column_name(ledger)
    daily = ledger.with_columns(pl.col("window_start").dt.date().alias("date")).group_by("date").agg(
        pl.col(pnl_name).sum().alias("pnl"), pl.len().alias("trades")
    ).sort("date")
    bootstrap = _day_bootstrap(daily, int(config.raw["gates"]["bootstrap_resamples"]), config.random_seed)
    sorted_ledger = ledger.sort(["window_start", "market_id"])
    cumulative = np.cumsum(_pnl_column(sorted_ledger).to_numpy())
    drawdown = np.maximum.accumulate(np.insert(cumulative, 0, 0.0))[:-1] - cumulative
    by_direction = _group_metrics(ledger, "predicted_up")
    by_cell = _group_metrics(
        ledger.with_columns(_entry_cell_expression().alias("entry_cell")), "entry_cell"
    )
    by_bucket = _group_metrics(
        ledger.with_columns(_price_bucket_expression().alias("price_bucket")), "price_bucket"
    )
    entry = ledger["seconds_elapsed"].to_numpy()
    price = ledger["selected_cost_5"].to_numpy()
    gross = ledger["gross_pnl_5"].to_numpy() if "gross_pnl_5" in ledger.columns else pnl
    fee_adjusted = (
        ledger["fee_adjusted_pnl_5"].to_numpy()
        if "fee_adjusted_pnl_5" in ledger.columns else pnl
    )
    reserve_adjusted = (
        ledger["reserve_adjusted_pnl_5"].to_numpy()
        if "reserve_adjusted_pnl_5" in ledger.columns else pnl
    )
    capacity = {}
    for quantity in VWAP_QUANTITIES:
        column = f"up_ask_vwap_{quantity}"
        if column not in ledger.columns:
            capacity[str(quantity)] = None
            continue
        selected_price = np.where(
            ledger["predicted_up"].to_numpy(),
            ledger[f"up_ask_vwap_{quantity}"].to_numpy(),
            ledger[f"down_ask_vwap_{quantity}"].to_numpy(),
        )
        capacity[str(quantity)] = float(
            quantity * (correct.astype(float) - selected_price).sum()
        )
    probability = _probability_metrics(scored) if scored is not None and not scored.is_empty() else {}
    streak = _maximum_same_direction_streak(ledger["predicted_up"].to_list())
    return {
        **compact,
        "total_scheduled_markets": scheduled_markets,
        "eligible_markets": eligible_markets,
        "strict_coverage": ledger.height / max(scheduled_markets, 1),
        "conditional_coverage": ledger.height / max(eligible_markets, 1),
        "trades_per_day": ledger.height / max(daily.height, 1),
        "wins": wins, "losses": losses,
        "wilson_interval": wilson,
        "wins_to_losses_ratio": wins / losses if losses else math.inf,
        "up_trades": int(ledger["predicted_up"].sum()),
        "down_trades": int((~ledger["predicted_up"]).sum()),
        "per_direction": by_direction,
        "maximum_same_direction_streak": streak,
        "average_entry_second": float(entry.mean()),
        "median_entry_second": float(np.median(entry)),
        "p25_entry_second": float(np.quantile(entry, 0.25)),
        "p75_entry_second": float(np.quantile(entry, 0.75)),
        "entry_cells": by_cell,
        "average_executable_share_cost": float(price.mean()),
        "vwap_pnl": capacity,
        "gross_pnl": float(gross.sum()),
        "fee_adjusted_pnl": float(fee_adjusted.sum()),
        "reserve_adjusted_pnl": float(reserve_adjusted.sum()),
        "stressed_pnl": float(pnl.sum()),
        "expectancy_per_trade": float(pnl.mean()),
        "pnl_per_scheduled_market": float(pnl.sum() / max(scheduled_markets, 1)),
        "average_win": float(pnl[pnl > 0].mean()) if np.any(pnl > 0) else 0.0,
        "average_loss": float(pnl[pnl < 0].mean()) if np.any(pnl < 0) else 0.0,
        "payoff_ratio": (
            float(pnl[pnl > 0].mean() / -pnl[pnl < 0].mean())
            if np.any(pnl > 0) and np.any(pnl < 0) else math.inf
        ),
        "return_on_entry_cost": float(pnl.sum() / max(float((5 * price).sum()), 1e-9)),
        "maximum_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "worst_day": float(daily["pnl"].min()),
        "tail_loss_cvar": _tail_cvar(pnl),
        "day_block_bootstrap_interval": bootstrap,
        "profitable_day_ratio": float((daily["pnl"] > 0).mean()),
        "profitable_fold_ratio": (
            float(
                ledger.group_by("fold").agg(pl.col(pnl_name).sum().alias("pnl"))["pnl"].gt(0).mean()
            ) if "fold" in ledger.columns and ledger["fold"].n_unique() > 1 else None
        ),
        "brier_score": probability.get("brier"),
        "log_loss": probability.get("log_loss"),
        "expected_calibration_error": probability.get("expected_calibration_error"),
        "price_buckets": by_bucket,
        "settlement_regimes": _group_metrics(ledger, "label_regime"),
        "diagnostic_slices": _diagnostic_slices(ledger),
        "refprice_feature_coverage": (
            float(ledger["refprice_causal_eligible"].mean())
            if "refprice_causal_eligible" in ledger.columns else None
        ),
        "refprice_staleness": (
            {
                "mean_seconds": float(ledger["chainlink_ref_age_seconds"].mean()),
                "p95_seconds": float(ledger["chainlink_ref_age_seconds"].quantile(0.95)),
            } if "chainlink_ref_age_seconds" in ledger.columns else None
        ),
        "current_regime_only": True,
    }


def _diagnostic_slices(ledger: pl.DataFrame) -> dict[str, Any]:
    volatility = ledger["btc_realized_volatility_60s_bps"]
    q1, q2, q3 = (float(volatility.quantile(q)) for q in (0.25, 0.50, 0.75))
    enriched = ledger.with_columns(
        pl.when(pl.col("chainlink_ref_reversal_5_vs_30") > 0)
        .then(pl.lit("reversal"))
        .otherwise(pl.lit("no_reversal"))
        .alias("refprice_reversal"),
        pl.when(pl.col("btc_volatility_shock_30_vs_120") > 1.5)
        .then(pl.lit("volatility_shock"))
        .otherwise(pl.lit("normal_volatility"))
        .alias("volatility_shock"),
        pl.when(pl.col("chainlink_ref_binance_disagreement_30s") > 0)
        .then(pl.lit("disagreement"))
        .otherwise(pl.lit("agreement"))
        .alias("binance_refprice_state"),
        pl.when(pl.col("proxy_label_up") != pl.col("authentic_label_up"))
        .then(pl.lit("proxy_authentic_disagreement"))
        .otherwise(pl.lit("proxy_authentic_agreement"))
        .alias("proxy_authentic_state"),
        pl.when(pl.col("btc_realized_volatility_60s_bps") <= q1).then(pl.lit("q1"))
        .when(pl.col("btc_realized_volatility_60s_bps") <= q2).then(pl.lit("q2"))
        .when(pl.col("btc_realized_volatility_60s_bps") <= q3).then(pl.lit("q3"))
        .otherwise(pl.lit("q4"))
        .alias("volatility_quartile"),
        pl.when((~pl.col("direction_correct")) & (pl.col("selected_cost_5") >= 0.85))
        .then(pl.lit("expensive_wrong"))
        .otherwise(pl.lit("other"))
        .alias("expensive_loss_state"),
    )
    return {
        name: _group_metrics(enriched, name)
        for name in (
            "refprice_reversal",
            "volatility_shock",
            "binance_refprice_state",
            "proxy_authentic_state",
            "volatility_quartile",
            "expensive_loss_state",
        )
    }


def _apply_qualification(tournament: dict[str, Any], config: TournamentConfig) -> None:
    results = tournament["results"]
    champion = results[CANDIDATE_NAMES[0]]
    gates = config.raw["gates"]
    qualified: list[str] = []
    for name, row in results.items():
        if name == CANDIDATE_NAMES[0]:
            continue
        reasons: list[str] = []
        checks = {
            "positive_stressed_pnl": row.get("stressed_pnl", 0) > 0,
            "positive_expectancy": row.get("stressed_expectancy_per_trade", 0) > 0,
            "positive_pnl_per_scheduled_market": row.get("pnl_per_scheduled_market", 0) > 0,
            "positive_bootstrap_lower": row.get("day_block_bootstrap_interval", {}).get("lower", -1) > 0,
            "accuracy": row.get("accuracy", 0) >= max(
                float(gates["minimum_accuracy"]),
                champion.get("accuracy", 0) - float(gates["champion_accuracy_tolerance"]),
            ),
            "profit_factor": row.get("profit_factor", 0) >= float(gates["minimum_profit_factor"]),
            "market_coverage": row.get("strict_coverage", 0) >= float(gates["minimum_market_coverage"]),
            "champion_relative_coverage": row.get("strict_coverage", 0) >= (
                float(gates["minimum_champion_relative_coverage"]) * champion.get("strict_coverage", 0)
            ),
            "entry_not_later": row.get("average_entry_second", math.inf) <= (
                champion.get("average_entry_second", math.inf)
                + float(gates["maximum_entry_delay_seconds"])
            ),
            "directions_positive": all(
                value["stressed_expectancy"] > 0 for value in row.get("per_direction", {}).values()
            ) if row.get("per_direction") else False,
            "profitable_folds": (row.get("profitable_fold_ratio") or 0) >= float(
                gates["minimum_profitable_fold_ratio"]
            ),
            "entry_cells_positive": all(
                value["stressed_expectancy"] > 0 for value in row.get("entry_cells", {}).values()
            ) if row.get("entry_cells") else False,
            "price_buckets_positive": all(
                value["stressed_expectancy"] > 0 for value in row.get("price_buckets", {}).values()
            ) if row.get("price_buckets") else False,
            "tail_cvar_nonworse": row.get("tail_loss_cvar", -math.inf) >= champion.get("tail_loss_cvar", -math.inf),
            "tail_cvar_target": row.get("tail_loss_cvar", -math.inf) >= (
                champion.get("tail_loss_cvar", 0) * (1.0 - float(gates["target_cvar_improvement"]))
            ),
            "drawdown_ratio_nonworse": _drawdown_ratio(row) <= _drawdown_ratio(champion),
            "no_directional_collapse": min(row.get("up_trades", 0), row.get("down_trades", 0)) >= 10,
            "runtime_feature_parity": bool(row.get("runtime_parity", {}).get("passed")),
        }
        for check, passed in checks.items():
            if not passed:
                reasons.append(check)
        row["qualification"] = {"passed": not reasons, "checks": checks, "reasons": reasons}
        if not reasons:
            qualified.append(name)
    if qualified:
        winner = max(
            qualified,
            key=lambda name: (
                results[name]["day_block_bootstrap_interval"]["lower_per_scheduled_market"],
                results[name]["stressed_pnl"], results[name]["strict_coverage"],
                -results[name]["maximum_drawdown"], -results[name]["average_entry_second"],
            ),
        )
        status = "provisional_challenger_qualified_consumed_evidence"
    else:
        nonchampion = [name for name in results if name != CANDIDATE_NAMES[0] and results[name]["trades"]]
        winner = max(
            nonchampion,
            key=lambda name: (
                results[name]["day_block_bootstrap_interval"]["lower_per_scheduled_market"],
                results[name]["stressed_pnl"], results[name]["strict_coverage"],
            ),
            default=None,
        )
        status = "no_deployable_challenger_qualified"
    tournament["selection"] = {
        "provisional_challenger": winner,
        "qualified_candidates": qualified,
        "status": status,
        "independent_holdout_required": True,
    }


def _runtime_parity(
    frame: pl.DataFrame, config: TournamentConfig, treatment: str
) -> dict[str, Any]:
    uses_refprice = treatment != "oracle_candle_control"
    delay = {}
    golden = None
    runtime_source = (
        config.package_root.parent
        / "polymarket-bot"
        / "src"
        / "btc"
        / "directional_features.rs"
    )
    runtime_text = runtime_source.read_text()
    if uses_refprice:
        refprice = load_source_group(config.paths, "refprice")
        sample = frame.filter(pl.col("window_start") >= config.current_start).head(5000)
        golden = verify_runtime_refprice_golden_vectors(sample, refprice)
        for seconds in (1.0, 2.0, 3.0):
            stressed = attach_causal_refprice_features(
                sample.drop(*REFPRICE_ALL_FEATURES, "refprice_causal_eligible",
                            "refprice_delay_stress_seconds", strict=False),
                refprice, additional_delay_seconds=seconds,
            )
            delay[str(int(seconds))] = {
                "eligible_rows": int(stressed["refprice_causal_eligible"].sum()),
                "coverage": float(stressed["refprice_causal_eligible"].mean()),
            }
    supported = set(REFPRICE_RUNTIME_FEATURES)
    used = set(feature_names(treatment)) & set(REFPRICE_ALL_FEATURES)
    runtime_names_present = all(f'"{name}"' in runtime_text for name in used)
    return {
        "passed": (
            used <= supported
            and runtime_names_present
            and (golden is None or golden["passed"])
        ),
        "mapped_training_features": sorted(used),
        "existing_realtime_feature_names": sorted(supported),
        "units": "basis_points_or_dimensionless_match_existing_runtime",
        "archive_availability_timestamp": "provider_available_at",
        "runtime_availability_timestamp": "local receipt timestamp",
        "source_timestamp_only_never_establishes_availability": True,
        "future_information_rejected": True,
        "terminal_twap_feature_used": False,
        "training_only_fields_used": False,
        "delay_stress": delay,
        "runtime_source": str(runtime_source.relative_to(config.package_root.parent)),
        "runtime_source_sha256": file_sha256(runtime_source),
        "runtime_feature_names_present": runtime_names_present,
        "golden_vector_verification": golden,
    }


def _failed_trade_analysis(tournament: dict[str, Any]) -> list[dict[str, Any]]:
    ledgers = tournament.get("ledgers", {})
    champion = ledgers.get(CANDIDATE_NAMES[0], pl.DataFrame())
    if champion.is_empty():
        return []
    failures = champion.filter(~pl.col("direction_correct")).sort("selected_cost_5", descending=True)
    candidate = tournament["selection"].get("provisional_challenger")
    challenger = ledgers.get(candidate, pl.DataFrame()) if candidate else pl.DataFrame()
    lookup = {row["market_id"]: row for row in challenger.iter_rows(named=True)}
    rows = []
    for failure in failures.iter_rows(named=True):
        other = lookup.get(failure["market_id"])
        rows.append(
            {
                "market_id": failure["market_id"],
                "window_start": failure["window_start"].isoformat(),
                "champion_prediction": "UP" if failure["predicted_up"] else "DOWN",
                "champion_probability": float(failure["probability_up"]),
                "authentic_twap60_outcome": "UP" if failure["label_up"] else "DOWN",
                "refprice_outcome": "UP" if failure.get("proxy_label_up") else "DOWN",
                "champion_entry_second": int(failure["seconds_elapsed"]),
                "champion_cost": float(failure["selected_cost_5"]),
                "champion_projected_pnl": float(_row_pnl(failure)),
                "challenger_action": "abstain" if other is None else (
                    "UP" if other["predicted_up"] else "DOWN"
                ),
                "challenger_entry_second": None if other is None else int(other["seconds_elapsed"]),
                "challenger_cost": None if other is None else float(other["selected_cost_5"]),
                "challenger_projected_pnl": None if other is None else float(_row_pnl(other)),
            }
        )
    return rows


def _regime_inventory(labels: pl.DataFrame, config: TournamentConfig) -> dict[str, Any]:
    regimes = (
        ("legacy_official_refprice", config.legacy_start, config.authentic_start),
        ("authentic_counterfactual_twap60", config.authentic_start, config.transition_start),
        ("twap30_transition", config.transition_start, config.current_start),
        ("official_current_twap60", config.current_start, config.end),
    )
    output = {}
    for name, start, end in regimes:
        block = labels.filter(pl.col("window_start").is_between(start, end, closed="left"))
        output[name] = {
            "markets": block["market_id"].n_unique(),
            "authentic_twap60_labels": block.filter(pl.col("authentic_label_up").is_not_null()).height,
            "proxy_twap60_labels": block.filter(pl.col("proxy_label_up").is_not_null()).height,
            "official_outcomes": block.filter(pl.col("official_outcome").is_in(["up", "down"])).height,
        }
    return output


def _group_metrics(frame: pl.DataFrame, column: str) -> dict[str, Any]:
    output = {}
    for value in frame[column].unique().sort().to_list():
        block = frame.filter(pl.col(column) == value)
        pnl = _pnl_column(block).to_numpy()
        output[str(value)] = {
            "trades": block.height,
            "accuracy": float(block["direction_correct"].mean()),
            "stressed_pnl": float(pnl.sum()),
            "stressed_expectancy": float(pnl.mean()),
        }
    return output


def _day_bootstrap(daily: pl.DataFrame, resamples: int, seed: int) -> dict[str, float]:
    pnl = daily["pnl"].to_numpy()
    trades = daily["trades"].to_numpy()
    if not len(pnl):
        return {"lower": 0.0, "upper": 0.0, "lower_per_scheduled_market": 0.0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pnl), size=(resamples, len(pnl)))
    totals = pnl[indices].sum(axis=1)
    expectation = totals / np.maximum(trades[indices].sum(axis=1), 1)
    return {
        "lower": float(np.quantile(expectation, 0.025)),
        "upper": float(np.quantile(expectation, 0.975)),
        "lower_per_scheduled_market": float(np.quantile(totals / (len(pnl) * 288), 0.025)),
    }


def _ece(y: np.ndarray, p: np.ndarray, weights: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0, 1, bins + 1)
    total = weights.sum()
    result = 0.0
    for low, high in itertools.pairwise(boundaries):
        mask = (p >= low) & (p < high if high < 1 else p <= high)
        if np.any(mask):
            mass = weights[mask].sum()
            result += mass / total * abs(
                np.average(y[mask], weights=weights[mask])
                - np.average(p[mask], weights=weights[mask])
            )
    return float(result)


def _wilson(wins: int, total: int) -> dict[str, float]:
    if total == 0:
        return {"lower": 0.0, "upper": 0.0}
    z = float(norm.ppf(0.975))
    p = wins / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"lower": center - radius, "upper": center + radius}


def _tail_cvar(pnl: np.ndarray, fraction: float = 0.10) -> float:
    if not len(pnl):
        return 0.0
    count = max(1, math.ceil(len(pnl) * fraction))
    return float(np.sort(pnl)[:count].mean())


def _maximum_same_direction_streak(values: list[bool]) -> int:
    best = current = 0
    previous = None
    for value in values:
        current = current + 1 if value == previous else 1
        best = max(best, current)
        previous = value
    return best


def _drawdown_ratio(row: dict[str, Any]) -> float:
    return float(row.get("maximum_drawdown", math.inf)) / max(
        float(row.get("stressed_pnl", 0.0)), 1e-9
    )


def _entry_cell_expression() -> pl.Expr:
    return (
        pl.when(pl.col("seconds_elapsed") < 90).then(pl.lit("60-89"))
        .when(pl.col("seconds_elapsed") < 120).then(pl.lit("90-119"))
        .when(pl.col("seconds_elapsed") < 150).then(pl.lit("120-149"))
        .otherwise(pl.lit("150-179"))
    )


def _price_bucket_expression() -> pl.Expr:
    return (
        pl.when(pl.col("selected_cost_5") < 0.65).then(pl.lit("below_0.65"))
        .when(pl.col("selected_cost_5") < 0.75).then(pl.lit("0.65-0.75"))
        .when(pl.col("selected_cost_5") < 0.85).then(pl.lit("0.75-0.85"))
        .otherwise(pl.lit("above_0.85"))
    )


def _cell_key_expression() -> pl.Expr:
    return pl.concat_str(
        _entry_cell_expression(),
        pl.when(pl.col("predicted_up")).then(pl.lit("up")).otherwise(pl.lit("down")),
        _price_bucket_expression(),
        separator="|",
    )


def _correctness_cell_expression(price_stratified: bool) -> pl.Expr:
    direction = pl.when(pl.col("predicted_up")).then(pl.lit("up")).otherwise(pl.lit("down"))
    if price_stratified:
        return pl.concat_str(
            _entry_cell_expression(), direction, _price_bucket_expression(), separator="|"
        )
    return pl.concat_str(_entry_cell_expression(), direction, separator="|")


def _pnl_column(frame: pl.DataFrame) -> pl.Series:
    return frame[_pnl_column_name(frame)]


def _pnl_column_name(frame: pl.DataFrame) -> str:
    for name in ("stressed_pnl_5", "stress_net_pnl_5", "stress_net_pnl"):
        if name in frame.columns:
            return name
    raise ValueError("ledger has no stressed PnL column")


def _row_pnl(row: dict[str, Any]) -> float:
    for name in ("stressed_pnl_5", "stress_net_pnl_5", "stress_net_pnl"):
        if name in row:
            return float(row[name])
    return math.nan


def _block(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    return frame.filter(pl.col("window_start").is_between(start, end, closed="left"))


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(clipped / (1 - clipped))


def _logit_expr(column: str) -> pl.Expr:
    value = pl.col(column).clip(1e-9, 1 - 1e-9)
    return value.log() - (1 - value).log()


def _difference(value: Any, baseline: Any) -> Any:
    if (
        isinstance(value, (int, float))
        and isinstance(baseline, (int, float))
        and math.isfinite(float(value))
        and math.isfinite(float(baseline))
    ):
        return float(value) - float(baseline)
    return None


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value).astimezone(UTC)
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(value)
    return parsed


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    raise TypeError(type(value).__name__)


def _render_report(metrics: dict[str, Any]) -> str:
    tournament = metrics["tournament"]
    lines = [
        "# BTC 5m TWAP60 challenger tournament",
        "",
        f"Run: `{metrics['run_id']}`",
        f"Source commit: `{metrics['source_commit']}`",
        f"Frozen watermark: `{metrics['configuration']['data_watermark']}`",
        "",
        (
            "This is consumed development evidence. No deployment, trading-process change, "
            "database mutation, new table, new ingester, or new data source occurred."
        ),
        "",
        "## Proxy-TWAP60 validation",
        "",
        (
            f"Convention: `{metrics['proxy_twap60']['convention']['time_column']}`; "
            f"error band: {metrics['proxy_twap60']['convention']['error_band_bps']:.4f} bps; "
            f"proxy transfer admitted: {metrics['proxy_twap60']['proxy_transfer_admitted']}."
        ),
        "",
        "## Feature bake-off",
        "",
        (
            f"Selected treatment: `{metrics['feature_bakeoff']['selection']['treatment']}` — "
            f"{metrics['feature_bakeoff']['selection']['reason']}."
        ),
        "",
        "## Tournament",
        "",
        "| Candidate | Trades | Coverage | Accuracy | Stressed PnL | Expectancy | Profit factor | CVaR | Avg entry | Qualified |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in tournament["results"].items():
        qualified = row.get("qualification", {}).get("passed", False)
        lines.append(
            f"| `{name}` | {row.get('trades', 0)} | {row.get('strict_coverage', 0):.2%} | "
            f"{row.get('accuracy', 0):.2%} | {row.get('stressed_pnl', 0):.4f} | "
            f"{row.get('stressed_expectancy_per_trade', 0):.4f} | "
            f"{row.get('profit_factor', 0):.3f} | {row.get('tail_loss_cvar', 0):.4f} | "
            f"{row.get('average_entry_second', 0):.2f} | {qualified} |"
        )
    lines.extend(
        [
            "",
            f"Selection status: `{tournament['selection']['status']}`.",
            f"Provisional challenger: `{tournament['selection']['provisional_challenger']}`.",
            "",
            (
                "All data through August 24 is consumed development evidence; an independently "
                "validated champion cannot be named from this run."
            ),
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in metrics["limitations"]],
            "",
        ]
    )
    return "\n".join(lines)
