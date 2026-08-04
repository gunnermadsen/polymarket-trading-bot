"""End-to-end offline benchmark for early prediction and asymmetric payouts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from .core_config import load_core_config
from .core_extract import extract_core_source, file_sha256, write_json_atomic
from .core_features import build_core_features, load_core_feature_frame
from .early_value_config import EarlyValueConfig
from .early_value_data import (
    build_partitioned_external_frame,
    extract_price_evidence,
    load_price_evidence,
)
from .early_value_evaluation import (
    attach_execution_value,
    bootstrap_net_expectancy,
    calibration_by_time_and_cost,
    ledger_metrics,
    policy_ledgers,
    price_by_second,
)
from .early_value_training import (
    accuracy_by_second,
    fit_early_models,
    prediction_frame,
)
from .runtime_export import score_runtime_model


def run_early_value_benchmark(
    config: EarlyValueConfig,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    core_config = load_core_config(config.core_config)
    _prepare_core_features(core_config, force=force)
    development = load_core_feature_frame(core_config, "pre_holdout")
    evaluation = load_core_feature_frame(core_config, "holdout")
    development_external = _load_or_build_external(
        development,
        config,
        destination=config.package_root / "data/btc-early-price-value-20260414-20260802/features/development-external.parquet",
        force=force,
    )
    evaluation_external = _load_or_build_external(
        evaluation,
        config,
        destination=config.package_root / "data/btc-early-price-value-20260414-20260802/features/evaluation-external.parquet",
        force=force,
    )

    models, training = fit_early_models(development_external, config, core_config)
    predictions = pl.concat(
        [
            prediction_frame(evaluation_external, bundle.probability(evaluation_external), model=name)
            for name, bundle in models.items()
        ],
        how="vertical_relaxed",
    )
    champion = _champion_reference(config, evaluation_external)
    if champion.height:
        predictions = pl.concat((predictions, champion), how="vertical_relaxed")

    extract_price_evidence(config, force=force)
    prices = load_price_evidence(config)
    scored = attach_execution_value(predictions, prices)
    ledgers = policy_ledgers(scored, config)
    policy_results: dict[str, Any] = {}
    for name, ledger in ledgers.items():
        metrics = ledger_metrics(ledger)
        metrics["utc_day_block_bootstrap_net_expectancy"] = bootstrap_net_expectancy(
            ledger,
            resamples=config.bootstrap_resamples,
            seed=config.random_seed,
        )
        policy_results[name] = metrics

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts = run_dir / "models"
    artifacts.mkdir()
    for name, bundle in models.items():
        joblib.dump(bundle, artifacts / f"{name}.joblib", compress=3)
    predictions.write_parquet(run_dir / "predictions.parquet", compression="zstd")
    prices.write_parquet(run_dir / "price-evidence.parquet", compression="zstd")
    scored.write_parquet(run_dir / "scored-execution-points.parquet", compression="zstd")
    for name, ledger in ledgers.items():
        ledger.write_parquet(run_dir / f"ledger-{name}.parquet", compression="zstd")

    accuracy_rows = accuracy_by_second(predictions)
    price_rows = price_by_second(prices)
    calibration_rows = calibration_by_time_and_cost(scored)
    pl.DataFrame(accuracy_rows).write_csv(run_dir / "accuracy-by-second.csv")
    pl.DataFrame(price_rows).write_csv(run_dir / "price-by-second.csv")
    pl.DataFrame(calibration_rows).write_csv(run_dir / "calibration-by-time-and-cost.csv")
    result: dict[str, Any] = {
        "schema_version": "btc-early-price-value-benchmark-v1",
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "paper_only": True,
        "live_capital_allowed": False,
        "runtime_changed": False,
        "trading_process_changed": False,
        "core_contract_changed": False,
        "prediction_timing": {
            "seconds": list(config.prediction_seconds),
            "threshold_applied_to_accuracy_surface": False,
        },
        "price_timing": {"seconds": list(config.price_seconds)},
        "windows": {
            name: {"start": value.start.isoformat(), "end": value.end.isoformat()}
            for name, value in (
                ("fit", config.fit),
                ("calibration", config.calibration),
                ("policy", config.policy),
                ("evaluation", config.evaluation),
            )
        },
        "training": training,
        "evaluation": {
            "prediction_rows": predictions.height,
            "prediction_markets": predictions["market_id"].n_unique(),
            "price_rows": prices.height,
            "priced_prediction_rows": scored.height,
            "accuracy_by_second": accuracy_rows,
            "price_by_second": price_rows,
            "calibration_by_time_and_cost": calibration_rows,
            "policies": policy_results,
        },
        "interpretation": {
            "break_even_rule": "predicted win probability must exceed all-in executable cost per share",
            "example_25pct_at_30c": "negative expectancy before sampling error because 0.25 < 0.30 plus fees",
            "refprice_training_eligible": False,
            "refprice_reason": "historical availability/receipt timestamps are not proven",
            "fresh_forward_shadow_required": True,
        },
        "lineage": {
            "benchmark_config_sha256": file_sha256(config.source_path),
            "core_config_sha256": file_sha256(config.core_config),
            "price_query_sha256": file_sha256(config.price_source_sql),
            "champion_model_sha256": file_sha256(config.champion_model),
        },
    }
    write_json_atomic(run_dir / "benchmark.json", result)
    (run_dir / "benchmark-report.md").write_text(_markdown_report(result))
    return run_dir, result


def _prepare_core_features(core_config: Any, *, force: bool) -> None:
    for scope in ("pre_holdout", "holdout"):
        extract_core_source(core_config, scope, force=force)
        build_core_features(core_config, scope, force=force)


def _load_or_build_external(
    core: pl.DataFrame,
    config: EarlyValueConfig,
    *,
    destination: Path,
    force: bool,
) -> pl.DataFrame:
    if destination.is_file() and not force:
        return pl.read_parquet(destination)
    frame = build_partitioned_external_frame(core, config.l2_source, config.candle_source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(destination, compression="zstd", statistics=True)
    return frame


def _champion_reference(config: EarlyValueConfig, frame: pl.DataFrame) -> pl.DataFrame:
    model = json.loads(config.champion_model.read_text())
    eligible = frame.filter(pl.col("seconds_elapsed") >= 60)
    names = tuple(model["features"]["names"])
    if eligible.is_empty() or any(name not in eligible.columns for name in names):
        return pl.DataFrame()
    matrix = eligible.select(*names).to_numpy()
    probability = np.array(
        [
            score_runtime_model(model, row.tolist(), seconds_elapsed=int(second))["probability_up"]
            for row, second in zip(matrix, eligible["seconds_elapsed"], strict=True)
        ],
        dtype=np.float64,
    )
    return prediction_frame(eligible, probability, model="frozen_champion_reference_60s_plus")


def _markdown_report(result: dict[str, Any]) -> str:
    selected = result["training"]["selected_profile"]
    policies = result["evaluation"]["policies"]
    lines = [
        "# BTC early-price value benchmark",
        "",
        "This is offline development evidence. It does not authorize deployment or live capital.",
        "",
        f"Selected probability model by policy-window log loss: `{selected}`.",
        "Accuracy is reported for every prediction point without an 89% admission filter.",
        "The frozen champion is a reference only from second 60 because its contract does not cover earlier decisions.",
        "",
        "## Economic policies",
        "",
        "| Policy | Trades | Accuracy | Mean cost/share | Net expectancy/trade | Profit factor |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in policies.items():
        lines.append(
            f"| {name} | {metrics['trades']} | {_fmt(metrics['accuracy'])} | "
            f"{_fmt(metrics['mean_entry_cost_per_share'])} | "
            f"{_fmt(metrics['net_expectancy_per_trade'])} | {_fmt(metrics['profit_factor'])} |"
        )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "A trade has positive modeled value only when predicted win probability exceeds fee-adjusted executable cost. Therefore 25% accuracy at a true $0.30 all-in cost is negative expectancy; the relevant comparison is probability versus cost, not probability versus 50% or 89%.",
            "",
            "See `accuracy-by-second.csv`, `price-by-second.csv`, and `calibration-by-time-and-cost.csv` for the complete surfaces.",
        )
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
