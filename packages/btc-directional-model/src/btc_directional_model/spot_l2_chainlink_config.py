"""Immutable configuration contract for the fixed spot-L2/candle benchmark."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .chainlink_oi_features import CHAINLINK_CANDLE_FEATURES
from .core_evaluation import FIRST_CROSSING_TIME_BANDS
from .spot_l2_chainlink_features import L2_FEATURES

CONTROL = "boundary_matched_control"
SPOT_L2 = "boundary_spot_l2"
CANDLES = "boundary_chainlink_candles"
COMBINED = "boundary_spot_l2_chainlink_candles"
PROFILES = (CONTROL, SPOT_L2, CANDLES, COMBINED)

FIXED_WINDOWS = {
    "source_start": "2026-04-14T00:00:00Z",
    "source_end": "2026-08-02T00:00:00Z",
    "fit_start": "2026-04-14T00:00:00Z",
    "fit_end": "2026-07-06T00:00:00Z",
    "calibration_start": "2026-07-06T00:00:00Z",
    "calibration_end": "2026-07-13T00:00:00Z",
    "policy_start": "2026-07-13T00:00:00Z",
    "policy_end": "2026-07-20T00:00:00Z",
    "evaluation_start": "2026-07-20T00:00:00Z",
    "evaluation_end": "2026-08-02T00:00:00Z",
}
FIXED_MODEL = {
    "learning_rate": 0.05,
    "max_iter": 160,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "l2_regularization": 0.10,
    "random_seed": 20260726,
}
FIXED_PROFILE_COUNTS = {CONTROL: 68, SPOT_L2: 108, CANDLES: 76, COMBINED: 116}
FIXED_ADVANCEMENT = {
    "minimum_net_expectancy_improvement": 0.0,
    "minimum_profit_factor_improvement": 0.0,
    "maximum_accuracy_regression": 0.01,
    "minimum_trade_coverage_ratio": 0.80,
    "minimum_win_retention_ratio": 0.80,
    "minimum_gross_loss_reduction": 0.10,
    "maximum_ece": 0.05,
    "minimum_consistent_days": 3,
}


@dataclass(frozen=True)
class ExecutionScenario:
    key: str
    arrival_latency_ms: int
    visible_depth_haircut: float


@dataclass(frozen=True)
class SpotL2ChainlinkConfig:
    source_path: Path
    package_root: Path
    core_source_sql: Path
    l2_source_sql: Path
    candles_source_sql: Path
    champion_model: Path
    champion_manifest: Path
    champion_process: Path
    cache: Path
    runs: Path
    source_schema_revision: str
    l2_view: str
    l2_schema_version: str
    l2_materialization_contracts: tuple[str, ...]
    candle_symbol: str
    base_features: tuple[str, ...]
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    random_seed: int
    confidence_threshold: float
    decision_seconds: tuple[int, ...]
    quantity: float
    execution_freshness_seconds: int
    execution_scenarios: tuple[ExecutionScenario, ...]
    bootstrap_resamples: int
    threads_per_fit: int
    advancement: dict[str, float | int]

    @property
    def feature_sets(self) -> dict[str, tuple[str, ...]]:
        return {
            CONTROL: self.base_features,
            SPOT_L2: (*self.base_features, *L2_FEATURES),
            CANDLES: (*self.base_features, *CHAINLINK_CANDLE_FEATURES),
            COMBINED: (*self.base_features, *L2_FEATURES, *CHAINLINK_CANDLE_FEATURES),
        }


def load_spot_l2_chainlink_config(path: Path) -> SpotL2ChainlinkConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    benchmark = _section(raw, "benchmark")
    windows = _section(raw, "windows")
    profiles = _section(raw, "profiles")
    model = _section(raw, "model")
    execution = _section(raw, "execution")
    advancement = _section(raw, "advancement")
    sources = _section(raw, "sources")
    paths = _section(raw, "paths")

    if benchmark.get("profile") != "spot_l2_chainlink_candles_fixed_ablation":
        raise ValueError("benchmark profile identity is fixed")
    if benchmark.get("paper_only") is not True:
        raise ValueError("spot-L2 benchmark must remain paper-only")
    if benchmark.get("live_capital_allowed") is not False:
        raise ValueError("spot-L2 benchmark cannot authorize live capital")
    confidence_threshold = float(benchmark.get("confidence_threshold", -1.0))
    if confidence_threshold != 0.89:
        raise ValueError("confidence threshold is fixed at 0.89")
    decision_seconds = tuple(int(value) for value in benchmark.get("decision_seconds", ()))
    if decision_seconds != tuple(range(60, 241, 5)):
        raise ValueError("decision cadence must be five seconds from 60 through 240")
    source_schema_revision = str(benchmark.get("source_schema_revision", ""))
    if len(source_schema_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_schema_revision
    ):
        raise ValueError("source_schema_revision must pin a full Git commit")
    for name, expected in FIXED_WINDOWS.items():
        if windows.get(name) != expected:
            raise ValueError(f"{name} is fixed at {expected}")

    champion_model = package_root / str(paths["champion_model"])
    champion_manifest = package_root / str(paths["champion_manifest"])
    champion_process = package_root / str(paths["champion_process"])
    if (
        not champion_model.is_file()
        or not champion_manifest.is_file()
        or not champion_process.is_file()
    ):
        raise FileNotFoundError("frozen champion model, manifest, and paper process are required")
    champion = json.loads(champion_model.read_text())
    champion_lineage = json.loads(champion_manifest.read_text())
    if champion_lineage.get("model_sha256") != _sha256(champion_model):
        raise RuntimeError("frozen champion model hash does not match its manifest")
    if champion_lineage.get("deployment_scope") != "paper_only":
        raise RuntimeError("frozen champion reference has an unexpected deployment scope")
    if champion_lineage.get("live_capital_allowed") is not False:
        raise RuntimeError("frozen champion reference unexpectedly allows live capital")
    base_features = tuple(str(name) for name in profiles.get(CONTROL, ()))
    champion_features = tuple(champion.get("features", {}).get("names", ()))
    if base_features != champion_features or len(set(base_features)) != 68:
        raise ValueError("matched control must exactly equal the frozen 68-feature champion schema")
    if champion.get("features", {}).get("schema_version") != (
        "btc-5m-directional-boundary-features-v1"
    ):
        raise RuntimeError("frozen champion feature schema version changed")
    if champion.get("features", {}).get("schema_sha256") != champion_lineage.get(
        "feature_schema_sha256"
    ):
        raise RuntimeError("frozen champion feature schema hash changed")
    for name, count in FIXED_PROFILE_COUNTS.items():
        if int(profiles.get(f"{name}_count", -1)) != count:
            raise ValueError(f"{name} feature count must be fixed at {count}")
    if len(L2_FEATURES) != 40 or len(CHAINLINK_CANDLE_FEATURES) != 8:
        raise RuntimeError("fixed external feature schema dimensions changed")

    if model.get("estimator") != "histogram_gradient_boosting":
        raise ValueError("only histogram gradient boosting is permitted")
    observed_model: dict[str, float | int] = {
        "learning_rate": float(model["learning_rate"]),
        "max_iter": int(model["max_iter"]),
        "max_leaf_nodes": int(model["max_leaf_nodes"]),
        "min_samples_leaf": int(model["min_samples_leaf"]),
        "l2_regularization": float(model["l2_regularization"]),
        "random_seed": int(model["random_seed"]),
    }
    if observed_model != FIXED_MODEL:
        raise ValueError("model parameters and random seed must exactly match the champion")
    if champion.get("provenance", {}).get("training_hyperparameters") != {
        name: FIXED_MODEL[name]
        for name in (
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
        )
    }:
        raise RuntimeError("configured estimator parameters diverge from the frozen champion")
    if int(champion.get("provenance", {}).get("random_seed", -1)) != FIXED_MODEL["random_seed"]:
        raise RuntimeError("configured random seed diverges from the frozen champion")
    _validate_champion_policy(champion, decision_seconds, confidence_threshold)

    quantity = float(execution.get("quantity", -1.0))
    execution_freshness_seconds = int(execution.get("freshness_seconds", -1))
    if (quantity, execution_freshness_seconds) != (5.0, 2):
        raise ValueError("execution quantity and freshness assumptions are fixed")
    execution_scenarios = _load_execution_scenarios(
        champion_process,
        champion_model=champion,
        champion_model_sha256=champion_lineage["model_sha256"],
    )
    bootstrap_resamples = int(model.get("bootstrap_resamples", 0))
    if bootstrap_resamples != 10_000:
        raise ValueError("bootstrap resamples are fixed at 10000")
    threads_per_fit = int(model.get("threads_per_fit", 0))
    if threads_per_fit != 1:
        raise ValueError("deterministic benchmark fitting is fixed to one thread per fit")
    observed_advancement: dict[str, float | int] = {
        name: int(advancement[name])
        if name == "minimum_consistent_days"
        else float(advancement[name])
        for name in FIXED_ADVANCEMENT
    }
    if observed_advancement != FIXED_ADVANCEMENT:
        raise ValueError("advancement gates are fixed by the benchmark contract")

    expected_sources = {
        "l2_view": "polymarket.binance_spot_btcusdt_l2_training_features",
        "l2_schema_version": "binance-spot-btcusdt-l2-one-second-features-v1",
        "candle_symbol": "BTCUSD",
    }
    for name, expected in expected_sources.items():
        if sources.get(name) != expected:
            raise ValueError(f"{name} is fixed at {expected}")
    contracts = tuple(str(value) for value in sources.get("l2_materialization_contracts", ()))
    expected_contracts = (
        "cryptohft-binance-spot-btcusdt-l2-features-v1",
        "coinapi-binance-spot-btcusdt-l2-snapshots-v1",
        "huggingface-goooddy-binance-spot-btcusdt-l2-features-v1",
    )
    if contracts != expected_contracts:
        raise ValueError("spot-L2 materialization contracts are fixed")

    config = SpotL2ChainlinkConfig(
        source_path=source_path,
        package_root=package_root,
        core_source_sql=package_root / str(paths["core_source_sql"]),
        l2_source_sql=package_root / str(paths["l2_source_sql"]),
        candles_source_sql=package_root / str(paths["candles_source_sql"]),
        champion_model=champion_model,
        champion_manifest=champion_manifest,
        champion_process=champion_process,
        cache=package_root / str(paths["cache"]),
        runs=package_root / str(paths["runs"]),
        source_schema_revision=source_schema_revision,
        l2_view=str(sources["l2_view"]),
        l2_schema_version=str(sources["l2_schema_version"]),
        l2_materialization_contracts=contracts,
        candle_symbol=str(sources["candle_symbol"]),
        base_features=base_features,
        learning_rate=float(observed_model["learning_rate"]),
        max_iter=int(observed_model["max_iter"]),
        max_leaf_nodes=int(observed_model["max_leaf_nodes"]),
        min_samples_leaf=int(observed_model["min_samples_leaf"]),
        l2_regularization=float(observed_model["l2_regularization"]),
        random_seed=int(observed_model["random_seed"]),
        confidence_threshold=confidence_threshold,
        decision_seconds=decision_seconds,
        quantity=quantity,
        execution_freshness_seconds=execution_freshness_seconds,
        execution_scenarios=execution_scenarios,
        bootstrap_resamples=bootstrap_resamples,
        threads_per_fit=threads_per_fit,
        advancement=observed_advancement,
    )
    missing_paths = [
        path
        for path in (config.core_source_sql, config.l2_source_sql, config.candles_source_sql)
        if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError("benchmark SQL is missing: " + ", ".join(map(str, missing_paths)))
    return config


def fixed_datetime(name: str) -> datetime:
    try:
        value = FIXED_WINDOWS[name]
    except KeyError as error:
        raise ValueError(f"unknown fixed benchmark timestamp: {name}") from error
    return datetime.fromisoformat(value).astimezone(UTC)


def _validate_champion_policy(
    champion: dict[str, Any],
    decision_seconds: tuple[int, ...],
    confidence_threshold: float,
) -> None:
    policy = champion.get("prediction_policy", {})
    if (
        policy.get("type") != "first_confidence_crossing"
        or int(policy.get("minimum_seconds_after_open", -1)) != decision_seconds[0]
        or int(policy.get("maximum_seconds_after_open", -1)) != decision_seconds[-1]
        or int(policy.get("cadence_seconds", -1)) != 5
    ):
        raise RuntimeError("frozen champion prediction policy changed")
    observed_bands = tuple(
        (
            str(band.get("name")),
            int(band.get("start_seconds", -1)),
            int(band.get("end_seconds_exclusive", -1)),
        )
        for band in champion.get("time_bands", ())
    )
    if observed_bands != FIRST_CROSSING_TIME_BANDS:
        raise RuntimeError("frozen champion calibration bands changed")
    if any(
        float(band.get("confidence_threshold", -1.0)) != confidence_threshold
        for band in champion.get("time_bands", ())
    ):
        raise RuntimeError("frozen champion confidence threshold changed")


def _load_execution_scenarios(
    path: Path,
    *,
    champion_model: dict[str, Any],
    champion_model_sha256: str,
) -> tuple[ExecutionScenario, ...]:
    process = json.loads(path.read_text())
    paper_root = process.get("config", {}).get("raw", {}).get("btc_realtime_paper", {})
    strategy = paper_root.get("strategy", {})
    decision = strategy.get("decision_strategy", {})
    if (
        decision.get("type") != "btc_directional_model"
        or decision.get("model_key") != champion_model.get("model_key")
        or decision.get("artifact_sha256") != champion_model_sha256
        or float(strategy.get("target_size", -1.0)) != 5.0
        or int(strategy.get("max_book_age_ms", -1)) != 2_000
    ):
        raise RuntimeError("frozen champion paper process no longer matches the model")
    paper = paper_root.get("paper", {})
    scenarios = (
        ExecutionScenario(
            key="arrival_150ms_depth_80pct",
            arrival_latency_ms=int(paper.get("arrival_latency_ms", -1)),
            visible_depth_haircut=float(paper.get("visible_depth_haircut", -1.0)),
        ),
        *(
            ExecutionScenario(
                key=str(row.get("scenario_key", "")),
                arrival_latency_ms=int(row.get("arrival_latency_ms", -1)),
                visible_depth_haircut=float(row.get("visible_depth_haircut", -1.0)),
            )
            for row in paper.get("stress_previews", ())
        ),
    )
    expected = (
        ExecutionScenario("arrival_150ms_depth_80pct", 150, 0.80),
        ExecutionScenario("latency_300ms_depth_65pct", 300, 0.65),
        ExecutionScenario("latency_600ms_depth_50pct", 600, 0.50),
    )
    if scenarios != expected:
        raise RuntimeError("frozen champion latency/depth stress scenarios changed")
    return scenarios


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise TypeError(f"configuration section is required: {name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
