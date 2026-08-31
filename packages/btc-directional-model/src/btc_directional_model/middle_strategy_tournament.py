"""Train and evaluate the five frozen middle-strategy challengers."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .core_extract import file_sha256
from .middle_strategy_data import build_middle_panel, extract_spot_l2, middle_cache
from .multivenue_early_entry_data import ENTRY_SECONDS, KEY_COLUMNS, load_data_config
from .multivenue_early_entry_tournament import (
    FORBIDDEN_INFERENCE_TOKENS,
    TreeModel,
    _fit_tree,
    _predict_tree,
)

SCHEMA_VERSION = "btc-middle-strategy-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-middle-strategy-model-v1"
BASE_NAMES = (
    "middle_specialist_refit",
    "middle_q5_admission",
    "crossvenue_middle_specialist",
)
ALL_NAMES = (
    "middle_specialist_refit",
    "middle_q5_admission",
    "middle_agreement_ensemble",
    "crossvenue_middle_specialist",
    "price_time_calibrated_middle_ensemble",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _git_revision(package_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=package_root, text=True
    ).strip()


def _candidate_contract(config: Any, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups = {name: tuple(values) for name, values in manifest["feature_groups"].items()}
    output: dict[str, dict[str, Any]] = {}
    for row in config.raw["candidates"]:
        contract = dict(row)
        if row["kind"] == "base":
            features = tuple(
                dict.fromkeys(
                    feature
                    for group in row["feature_groups"]
                    for feature in groups[group]
                )
            )
            forbidden = [
                name
                for name in features
                if any(token in name.lower() for token in FORBIDDEN_INFERENCE_TOKENS)
            ]
            if forbidden:
                raise RuntimeError(f"forbidden inference features for {row['name']}: {forbidden}")
            contract["features"] = features
        output[row["name"]] = contract
    if tuple(output) != ALL_NAMES:
        raise RuntimeError("candidate roster or order differs from the frozen contract")
    return output


def _prediction_frame(
    frame: pl.DataFrame, probability: np.ndarray, fold: str, candidate: str
) -> pl.DataFrame:
    if len(probability) != frame.height or not np.isfinite(probability).all():
        raise RuntimeError(f"{candidate} produced invalid probabilities")
    return frame.select(*KEY_COLUMNS, "label_up").with_columns(
        pl.lit(fold).alias("fold"),
        pl.lit(candidate).alias("candidate"),
        pl.Series("probability", probability),
        pl.lit(True).alias("eligible_signal"),
    )


def _base_oof(
    panel: pl.DataFrame,
    contracts: dict[str, dict[str, Any]],
    config: Any,
) -> pl.DataFrame:
    checkpoint = middle_cache(config) / "base-oof-predictions.parquet"
    pieces: list[pl.DataFrame] = []
    existing: set[tuple[str, str]] = set()
    if checkpoint.is_file():
        cached = pl.read_parquet(checkpoint)
        pieces.append(cached)
        existing = set(cached.select("candidate", "fold").unique().iter_rows())
    seed = int(config.raw["training"]["random_seed"])
    for fold_index, fold in enumerate(config.raw["folds"]):
        start = datetime.fromisoformat(fold["test_start"])
        end = datetime.fromisoformat(fold["test_end"])
        train = panel.filter(pl.col("window_start") < start)
        test = panel.filter(pl.col("window_start").is_between(start, end, closed="left"))
        if set(train["market_id"].unique()) & set(test["market_id"].unique()):
            raise RuntimeError(f"market contamination in {fold['name']}")
        for base_index, name in enumerate(BASE_NAMES):
            if (name, fold["name"]) in existing:
                continue
            model = _fit_tree(
                train,
                tuple(contracts[name]["features"]),
                config,
                seed + 100 * fold_index + base_index,
            )
            piece = _prediction_frame(
                test, _predict_tree(model, test), fold["name"], name
            )
            pieces.append(piece)
            pl.concat(pieces, how="vertical_relaxed").write_parquet(
                checkpoint, compression="zstd", statistics=True
            )
            print(
                f"middle tournament OOF: {fold['name']} {name} "
                f"{test['market_id'].n_unique():,} markets",
                flush=True,
            )
    return pl.concat(pieces, how="vertical_relaxed", rechunk=True).sort(
        ["fold", "candidate", "window_start", "market_id", "seconds_elapsed"]
    )


def _wide_base_predictions(base: pl.DataFrame) -> pl.DataFrame:
    keys = (*KEY_COLUMNS, "label_up", "fold")
    parts = []
    for name in BASE_NAMES:
        parts.append(
            base.filter(pl.col("candidate") == name)
            .select(*keys, pl.col("probability").alias(name))
        )
    output = parts[0]
    for part in parts[1:]:
        output = output.join(part, on=list(keys), how="inner", validate="1:1")
    return output


def _agreement_predictions(wide: pl.DataFrame) -> pl.DataFrame:
    first, second = BASE_NAMES[:2]
    agrees = (pl.col(first) >= 0.5) == (pl.col(second) >= 0.5)
    probability = ((pl.col(first) + pl.col(second)) / 2.0).alias("probability")
    return wide.select(*KEY_COLUMNS, "label_up", "fold", probability, agrees.alias("eligible_signal")).with_columns(
        pl.lit("middle_agreement_ensemble").alias("candidate")
    )


def _ensemble_matrix(frame: pl.DataFrame) -> np.ndarray:
    probabilities = [
        np.clip(frame[name].to_numpy().astype(float), 1e-6, 1 - 1e-6)
        for name in BASE_NAMES
    ]
    seconds = frame["seconds_elapsed"].to_numpy().astype(float) / 300.0
    return np.column_stack((*[logit(values) for values in probabilities], seconds, seconds**2))


def _fit_ensemble_calibrator(frame: pl.DataFrame, seed: int) -> LogisticRegression:
    return LogisticRegression(C=0.5, solver="lbfgs", random_state=seed, max_iter=500).fit(
        _ensemble_matrix(frame), frame["label_up"].to_numpy()
    )


def _causal_calibrated_oof(wide: pl.DataFrame, config: Any) -> pl.DataFrame:
    pieces = []
    prior = []
    seed = int(config.raw["training"]["random_seed"])
    for fold_index, fold in enumerate(config.raw["folds"]):
        current = wide.filter(pl.col("fold") == fold["name"])
        if prior:
            fitting = pl.concat(prior, how="vertical_relaxed")
            calibrator = _fit_ensemble_calibrator(fitting, seed + fold_index)
            probability = calibrator.predict_proba(_ensemble_matrix(current))[:, 1]
        else:
            probability = current.select(pl.mean_horizontal(*BASE_NAMES)).to_series().to_numpy()
        pieces.append(
            _prediction_frame(
                current,
                probability,
                fold["name"],
                "price_time_calibrated_middle_ensemble",
            )
        )
        prior.append(current)
    return pl.concat(pieces, how="vertical_relaxed")


def _all_oof(base: pl.DataFrame, config: Any) -> tuple[pl.DataFrame, pl.DataFrame]:
    wide = _wide_base_predictions(base)
    agreement = _agreement_predictions(wide)
    calibrated = _causal_calibrated_oof(wide, config)
    return pl.concat((base, agreement, calibrated), how="vertical_relaxed"), wide


def _ece(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1]
            if index == bins - 1
            else probability < edges[index + 1]
        )
        if selected.any():
            score += selected.mean() * abs(probability[selected].mean() - labels[selected].mean())
    return float(score)


def predictive_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    labels = frame["label_up"].to_numpy().astype(float)
    probability = frame["probability"].to_numpy().astype(float)

    def summary(part: pl.DataFrame) -> dict[str, Any]:
        y = part["label_up"].to_numpy().astype(float)
        p = part["probability"].to_numpy().astype(float)
        return {
            "rows": part.height,
            "markets": part["market_id"].n_unique(),
            "brier_score": float(np.mean((p - y) ** 2)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "accuracy": float(np.mean((p >= 0.5) == y)),
            "ece_15": _ece(y, p),
        }

    bands = {}
    for name, start, end in (
        ("60_89", 60, 89),
        ("90_179", 90, 179),
        ("180_240", 180, 240),
        ("90_119", 90, 119),
        ("120_149", 120, 149),
        ("150_179", 150, 179),
    ):
        bands[name] = summary(
            frame.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        )
    result = summary(frame)
    result["mean_probability"] = float(probability.mean())
    result["positive_rate"] = float(labels.mean())
    result["by_timing_band"] = bands
    return result


def _opportunities(predictions: pl.DataFrame, panel: pl.DataFrame, config: Any) -> pl.DataFrame:
    quantity = int(config.raw["execution"]["quantity"])
    up, down = f"up_ask_vwap_{quantity}", f"down_ask_vwap_{quantity}"
    context = [up, down, "fee_rate", "pm_up_book_age_seconds", "pm_down_book_age_seconds"]
    joined = predictions.join(
        panel.select(*KEY_COLUMNS, *context), on=list(KEY_COLUMNS), how="left", validate="m:1"
    )
    reserve = float(config.raw["execution"]["execution_reserve_per_share"])
    return (
        joined.filter(pl.col("seconds_elapsed").is_between(90, 179, closed="both"))
        .with_columns(
            pl.when(pl.col("probability") >= 0.5).then(pl.lit("up")).otherwise(pl.lit("down")).alias("side"),
            pl.max_horizontal("probability", 1.0 - pl.col("probability")).alias("selected_probability"),
            pl.when(pl.col("probability") >= 0.5).then(pl.col(up)).otherwise(pl.col(down)).alias("share_cost"),
        )
        .with_columns(
            (pl.col("fee_rate").fill_null(0.0) * pl.col("share_cost") * (1.0 - pl.col("share_cost"))).alias("fee_per_share")
        )
        .with_columns(
            (pl.col("selected_probability") - pl.col("share_cost") - pl.col("fee_per_share") - reserve).alias("expected_edge")
        )
    )


def _select_trades(opportunities: pl.DataFrame, policy: dict[str, float], config: Any) -> pl.DataFrame:
    quantity = int(config.raw["execution"]["quantity"])
    reserve = float(config.raw["execution"]["execution_reserve_per_share"])
    stress = float(config.raw["execution"]["stress_slippage_per_share"])
    eligible = (
        opportunities.filter(
            pl.col("eligible_signal")
            & pl.col("share_cost").is_not_null()
            & pl.col("share_cost").is_finite()
            & (pl.col("share_cost") > 0)
            & (pl.col("share_cost") <= policy["maximum_share_cost"])
            & (pl.col("selected_probability") >= policy["minimum_confidence"])
            & (pl.col("expected_edge") >= policy["minimum_edge"])
            & (pl.col("pm_up_book_age_seconds") > 0)
            & (pl.col("pm_up_book_age_seconds") <= 2)
            & (pl.col("pm_down_book_age_seconds") > 0)
            & (pl.col("pm_down_book_age_seconds") <= 2)
        )
        .sort(["market_id", "seconds_elapsed"])
        .group_by("market_id", maintain_order=True)
        .first()
    )
    won = ((pl.col("side") == "up") & (pl.col("label_up") == 1)) | (
        (pl.col("side") == "down") & (pl.col("label_up") == 0)
    )
    return eligible.with_columns(won.alias("won")).with_columns(
        pl.when(pl.col("won")).then(1.0 - pl.col("share_cost")).otherwise(-pl.col("share_cost")).sub(pl.col("fee_per_share") + reserve).mul(quantity).alias("net_pnl"),
        pl.when(pl.col("won")).then(1.0 - pl.col("share_cost") - stress).otherwise(-(pl.col("share_cost") + stress)).sub(pl.col("fee_per_share") + reserve).mul(quantity).alias("stress_net_pnl"),
    )


def economic_metrics(trades: pl.DataFrame, total_markets: int) -> dict[str, Any]:
    if trades.is_empty():
        return {
            "trades": 0, "winning_trades": 0, "losing_trades": 0, "win_loss_ratio": None,
            "win_rate": None, "net_pnl": 0.0, "expectancy_per_trade": None,
            "stress_net_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            "profit_factor": None, "loss_recovery_wins": None, "market_coverage": 0.0,
            "average_share_cost": None, "average_entry_second": None, "maximum_drawdown": 0.0,
            "active_days": 0, "profitable_day_ratio": None, "maximum_daily_pnl_concentration": None,
            "by_entry_cell": {}, "by_price_bucket": {},
        }
    pnl = trades["net_pnl"].to_numpy().astype(float)
    wins, losses = pnl[pnl > 0], -pnl[pnl < 0]
    cumulative = np.cumsum(pnl)
    drawdown = np.maximum.accumulate(np.r_[0.0, cumulative])[1:] - cumulative
    winning, losing = len(wins), len(losses)
    daily = trades.with_columns(pl.col("window_start").dt.date().alias("day")).group_by("day").agg(pl.col("net_pnl").sum()).sort("day")
    positive_daily = daily.filter(pl.col("net_pnl") > 0)["net_pnl"].sum()
    maximum_concentration = None
    if positive_daily and positive_daily > 0:
        maximum_concentration = float(daily["net_pnl"].max() / positive_daily)

    def grouped_metrics(column: str, breaks: tuple[tuple[str, float, float], ...]) -> dict[str, Any]:
        output = {}
        for name, start, end in breaks:
            part = trades.filter(pl.col(column).is_between(start, end, closed="both"))
            values = part["net_pnl"].to_numpy().astype(float)
            output[name] = {
                "trades": part.height,
                "wins": int((values > 0).sum()),
                "losses": int((values < 0).sum()),
                "net_pnl": float(values.sum()),
            }
        return output

    return {
        "trades": trades.height,
        "winning_trades": winning,
        "losing_trades": losing,
        "win_loss_ratio": winning / losing if losing else None,
        "win_rate": winning / trades.height,
        "net_pnl": float(pnl.sum()),
        "expectancy_per_trade": float(pnl.mean()),
        "stress_net_pnl": float(trades["stress_net_pnl"].sum()),
        "gross_profit": float(wins.sum()),
        "gross_loss": float(losses.sum()),
        "profit_factor": float(wins.sum() / losses.sum()) if losses.sum() else None,
        "loss_recovery_wins": float(losses.mean() / wins.mean()) if winning and losing else None,
        "market_coverage": trades["market_id"].n_unique() / max(total_markets, 1),
        "average_share_cost": float(trades["share_cost"].mean()),
        "average_entry_second": float(trades["seconds_elapsed"].mean()),
        "maximum_drawdown": float(drawdown.max(initial=0.0)),
        "active_days": daily.height,
        "profitable_day_ratio": float((daily["net_pnl"] > 0).mean()),
        "maximum_daily_pnl_concentration": maximum_concentration,
        "by_entry_cell": grouped_metrics("seconds_elapsed", (("90_119", 90, 119), ("120_149", 120, 149), ("150_179", 150, 179))),
        "by_price_bucket": grouped_metrics("share_cost", (("0_0.65", 0.0, 0.65), ("0.65_0.80", 0.6500001, 0.80), ("0.80_0.95", 0.8000001, 0.95), ("0.95_1.0", 0.9500001, 1.0))),
    }


def _policy_score(trades: pl.DataFrame) -> float:
    if trades.is_empty():
        return -math.inf
    pnl = trades["net_pnl"].to_numpy().astype(float)
    if len(pnl) < 2:
        return -math.inf
    lower_expectancy = float(pnl.mean() - pnl.std(ddof=1) / math.sqrt(len(pnl)))
    folds = trades.group_by("fold").agg(pl.col("net_pnl").sum())["net_pnl"].to_numpy()
    return lower_expectancy + 0.01 * float(np.median(folds)) + 0.0005 * math.sqrt(len(pnl))


def _select_policy(opportunities: pl.DataFrame, config: Any) -> tuple[dict[str, float], dict[str, Any]]:
    execution = config.raw["execution"]
    best: tuple[float, dict[str, float], pl.DataFrame] | None = None
    for edge in execution["minimum_edges"]:
        for confidence in execution["minimum_confidences"]:
            for cost in execution["maximum_share_costs"]:
                policy = {"minimum_edge": float(edge), "minimum_confidence": float(confidence), "maximum_share_cost": float(cost)}
                trades = _select_trades(opportunities, policy, config)
                if trades.height < int(execution["minimum_policy_trades"]):
                    continue
                score = _policy_score(trades)
                if best is None or score > best[0]:
                    best = (score, policy, trades)
    if best is None:
        policy = {
            "minimum_edge": min(execution["minimum_edges"]),
            "minimum_confidence": min(execution["minimum_confidences"]),
            "maximum_share_cost": max(execution["maximum_share_costs"]),
        }
        return policy, {"selection_score": None, "development": economic_metrics(pl.DataFrame(), opportunities["market_id"].n_unique())}
    score, policy, trades = best
    return policy, {"selection_score": score, "development": economic_metrics(trades, opportunities["market_id"].n_unique())}


def _select_cell_policies(opportunities: pl.DataFrame, config: Any) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    policies, evidence = {}, {}
    for name, start, end in (("90_119", 90, 119), ("120_149", 120, 149), ("150_179", 150, 179)):
        policies[name], evidence[name] = _select_policy(
            opportunities.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both")), config
        )
    return policies, evidence


def _apply_cell_policies(opportunities: pl.DataFrame, policies: dict[str, dict[str, float]], config: Any) -> pl.DataFrame:
    pieces = []
    for name, start, end in (("90_119", 90, 119), ("120_149", 120, 149), ("150_179", 150, 179)):
        pieces.append(_select_trades(opportunities.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both")), policies[name], config))
    trades = pl.concat(pieces, how="diagonal_relaxed")
    if trades.is_empty():
        return trades
    return trades.sort(["market_id", "seconds_elapsed"]).group_by("market_id", maintain_order=True).first()


def _reference_manifests(config: Any) -> dict[str, Any]:
    output = {}
    for name, key in (("frozen_q5", "q5_manifest"), ("frozen_middle_specialist", "specialist_manifest")):
        path = config.package_root / config.raw["paths"][key]
        payload = json.loads(path.read_text())
        output[name] = {
            "manifest_path": str(path.relative_to(config.package_root)),
            "manifest_sha256": file_sha256(path),
            "model_artifact_sha256": payload.get("model_artifact", {}).get("sha256")
            or payload.get("artifact_sha256")
            or payload.get("model_sha256"),
            "scored_without_alteration": False,
            "reason": "Immutable runtime comparator retained for provenance; its input and entry-policy contract is not identical to the 60-240 tournament panel.",
        }
    return output


def _report(metrics: dict[str, Any]) -> str:
    def fmt(value: Any, digits: int = 3) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Middle-Strategy Tournament", "", f"Run: `{metrics['run_id']}`  ",
        f"Qualification: **{metrics['qualification_status']}**", "", "## Sealed high-level results", "",
        "| Candidate | PnL | Stress PnL | Coverage | Wins | Losses | W/L | Recovery wins/loss | Brier | PF | Avg cost | Avg entry |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ALL_NAMES:
        p, e = metrics["sealed_predictive"][name], metrics["sealed_economic"][name]
        lines.append(
            f"| {name} | {e['net_pnl']:.2f} | {e['stress_net_pnl']:.2f} | {e['market_coverage']:.2%} | {e['winning_trades']} | {e['losing_trades']} | {fmt(e['win_loss_ratio'])} | {fmt(e['loss_recovery_wins'])} | {p['brier_score']:.4f} | {fmt(e['profit_factor'])} | {fmt(e['average_share_cost'])} | {fmt(e['average_entry_second'], 1)} |"
        )
    lines.extend((
        "", "## Integrity", "",
        "- Full retained source range: March 21 through August 28; optional-source gaps never remove a core market.",
        "- Training is chronological and market-disjoint. The August 14–28 replay is opened only after models and policies are frozen.",
        "- The replay dates were observed in earlier research, so they are computationally sealed here but are not claimed as epistemically untouched.",
        "- Official settlement outcomes are labels. TWAP and `authentic_only` are absent from inference and selection.",
        "- Economic results require both recorded books to be no more than two seconds old.",
        "- No database writes, tables, ingesters, sources, runtime exports, deployments, or trading-process changes were made.",
    ))
    return "\n".join(lines) + "\n"


def train_tournament(config: Any, *, force: bool = False) -> Path:
    panel, panel_manifest = build_middle_panel(config, force=force)
    contracts = _candidate_contract(config, panel_manifest)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.results / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledgers = run_dir / "ledgers"
    ledgers.mkdir()
    split = {
        "source_start": config.source_start.isoformat(), "fit_end_exclusive": config.fit_end.isoformat(),
        "sealed_start_inclusive": config.sealed_start.isoformat(), "sealed_end_exclusive": config.sealed_end.isoformat(),
        "folds": config.raw["folds"], "entry_seconds": list(ENTRY_SECONDS), "economic_seconds": [90, 179],
        "market_disjoint": True, "sealed_replay_epistemically_untouched": False,
    }
    _write_json(run_dir / "split-manifest.json", split)

    preseal = panel.filter(pl.col("window_start") < config.fit_end)
    base_oof = _base_oof(preseal, contracts, config)
    oof, wide_oof = _all_oof(base_oof, config)
    oof.write_parquet(ledgers / "candidate-oof-predictions.parquet", compression="zstd", statistics=True)
    oof_predictive, policies, policy_evidence = {}, {}, {}
    for name in ALL_NAMES:
        candidate = oof.filter(pl.col("candidate") == name)
        oof_predictive[name] = predictive_metrics(candidate)
        opportunities = _opportunities(candidate, preseal, config)
        if name == "price_time_calibrated_middle_ensemble":
            policies[name], policy_evidence[name] = _select_cell_policies(opportunities, config)
        else:
            policies[name], policy_evidence[name] = _select_policy(opportunities, config)
    predictive_ranking = sorted(ALL_NAMES, key=lambda name: oof_predictive[name]["brier_score"])
    selection_frozen_at = datetime.now(UTC).isoformat()
    _write_json(run_dir / "selection-freeze.json", {
        "frozen_at": selection_frozen_at, "predictive_ranking": predictive_ranking,
        "policies": policies, "sealed_metrics_accessed": False,
    })

    sealed = panel.filter(pl.col("window_start").is_between(config.sealed_start, config.sealed_end, closed="left"))
    if set(preseal["market_id"].unique()) & set(sealed["market_id"].unique()):
        raise RuntimeError("sealed markets entered training")
    final_models: dict[str, TreeModel] = {}
    final_base_predictions = []
    seed = int(config.raw["training"]["random_seed"])
    for index, name in enumerate(BASE_NAMES):
        model = _fit_tree(preseal, tuple(contracts[name]["features"]), config, seed + 10_000 + index)
        final_models[name] = model
        final_base_predictions.append(_prediction_frame(sealed, _predict_tree(model, sealed), "sealed_20260814_20260828", name))
    sealed_base = pl.concat(final_base_predictions, how="vertical_relaxed")
    sealed_wide = _wide_base_predictions(sealed_base)
    final_calibrator = _fit_ensemble_calibrator(wide_oof, seed + 20_000)
    sealed_agreement = _agreement_predictions(sealed_wide)
    sealed_calibrated = _prediction_frame(
        sealed_wide, final_calibrator.predict_proba(_ensemble_matrix(sealed_wide))[:, 1],
        "sealed_20260814_20260828", "price_time_calibrated_middle_ensemble",
    )
    sealed_predictions = pl.concat((sealed_base, sealed_agreement, sealed_calibrated), how="vertical_relaxed")
    sealed_predictions.write_parquet(ledgers / "sealed-predictions.parquet", compression="zstd", statistics=True)

    sealed_predictive, sealed_economic, trade_pieces = {}, {}, []
    for name in ALL_NAMES:
        prediction = sealed_predictions.filter(pl.col("candidate") == name)
        sealed_predictive[name] = predictive_metrics(prediction)
        opportunities = _opportunities(prediction, sealed, config)
        trades = _apply_cell_policies(opportunities, policies[name], config) if name == "price_time_calibrated_middle_ensemble" else _select_trades(opportunities, policies[name], config)
        if not trades.is_empty():
            trade_pieces.append(trades.with_columns(pl.lit(name).alias("candidate")))
        sealed_economic[name] = economic_metrics(trades, sealed["market_id"].n_unique())
    trade_frame = pl.concat(trade_pieces, how="diagonal_relaxed") if trade_pieces else pl.DataFrame()
    trade_frame.write_parquet(ledgers / "sealed-trades.parquet", compression="zstd", statistics=True)

    producing_commit = _git_revision(config.package_root)
    artifact_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION, "model_family": config.raw["training"]["model_family"],
        "producing_commit": producing_commit, "run_id": run_id, "fit_end_exclusive": config.fit_end.isoformat(),
        "entry_seconds": ENTRY_SECONDS, "economic_seconds": (90, 179), "candidate_contract": contracts,
        "base_models": final_models, "ensemble_calibrator": final_calibrator, "selected_policies": policies,
        "runtime_exported": False, "deployment_status": "not_deployed",
    }
    artifact = run_dir / "tournament.joblib"
    joblib.dump(artifact_payload, artifact, compress=3)
    subprocess.run(
        [sys.executable, "-c", "import joblib,sys; x=joblib.load(sys.argv[1]); assert len(x['base_models'])==3", str(artifact)],
        check=True, cwd=config.package_root,
    )
    artifact_sha = file_sha256(artifact)
    (run_dir / "tournament.sha256").write_text(f"{artifact_sha}  tournament.joblib\n")
    economic_ranking = sorted(ALL_NAMES, key=lambda name: (sealed_economic[name]["stress_net_pnl"], sealed_economic[name]["net_pnl"]), reverse=True)
    metrics = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "created_at": datetime.now(UTC).isoformat(),
        "producing_commit": producing_commit, "candidate_names": list(ALL_NAMES), "candidate_count_trained": 5,
        "learned_base_model_count": 3, "derived_ensemble_count": 2,
        "predictive_ranking_preseal": predictive_ranking, "economic_ranking_sealed": economic_ranking,
        "selection_frozen_at": selection_frozen_at, "oof_predictive": oof_predictive,
        "selected_policies": policies, "policy_selection_evidence": policy_evidence,
        "sealed_predictive": sealed_predictive, "sealed_economic": sealed_economic,
        "frozen_comparator_references": _reference_manifests(config), "source_panel": panel_manifest,
        "split_manifest": split, "artifact_sha256": artifact_sha,
        "qualification_status": "trained_evaluated_not_deployed",
        "integrity": {
            "passed": True, "market_disjoint": True, "sealed_opened_after_selection": True,
            "artifact_round_trip_load": True, "full_history_retained": True,
            "optional_missingness_preserves_rows": True, "twap_inference_feature": False,
            "authentic_only_filter": False, "kraken_l2_included": False,
            "database_mutations": False, "new_tables": False, "new_ingesters": False,
            "new_sources": False, "runtime_exported": False, "deployed": False,
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "polars": pl.__version__, "sklearn": sklearn.__version__},
        "limitations": [
            "The August 14-28 replay is row-disjoint but was observed during previous research and is not epistemically fresh.",
            "Kraken L2 is excluded because its historical backfill is incomplete.",
            "Qualified Binance spot L2 ends August 1; later rows remain usable with L2 missing.",
            "Projected PnL assumes recorded ask VWAP was fillable and does not model queue position.",
        ],
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "candidate-contract.json", contracts)
    _write_json(run_dir / "source-manifest.json", panel_manifest)
    (run_dir / "report.md").write_text(_report(metrics))
    _write_json(run_dir / "completion.json", {
        "run_id": run_id, "artifact_sha256": artifact_sha,
        "metrics_sha256": file_sha256(run_dir / "metrics.json"), "completed": True,
    })
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("command", choices=("extract", "prepare", "train", "all"))
    arguments = parser.parse_args()
    config = load_data_config(arguments.config)
    if arguments.command in {"extract", "all"}:
        extract_spot_l2(config, force=arguments.force)
    if arguments.command in {"prepare", "all"}:
        build_middle_panel(config, force=arguments.force)
    if arguments.command in {"train", "all"}:
        print(train_tournament(config, force=arguments.force))


if __name__ == "__main__":
    main()
