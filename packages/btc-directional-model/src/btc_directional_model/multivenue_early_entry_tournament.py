"""Train and evaluate the frozen multi-venue early-entry tournament."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .core_extract import file_sha256
from .multivenue_early_entry_data import (
    ENTRY_SECONDS,
    KEY_COLUMNS,
    TournamentDataConfig,
    build_panel,
    extract_sources,
    load_data_config,
)

SCHEMA_VERSION = "btc-multivenue-early-entry-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-multivenue-early-entry-model-v1"
PREDICTION_COLUMNS = (*KEY_COLUMNS, "label_up", "fold", "candidate", "probability")
FORBIDDEN_INFERENCE_TOKENS = ("twap", "official_outcome", "label_up", "final_price", "resolution")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _git_revision(package_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=package_root, text=True
    ).strip()


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class TreeModel:
    features: tuple[str, ...]
    estimator: HistGradientBoostingClassifier
    calibrator: LogisticRegression | None
    neutralized_columns: tuple[int, ...]


@dataclass(frozen=True)
class CandidateModel:
    name: str
    feature_groups: tuple[str, ...]
    models: dict[str, TreeModel]
    timing_bands: dict[str, tuple[int, int]]
    fit_end: str


@dataclass(frozen=True)
class ExecutionPolicy:
    minimum_edge: float
    minimum_confidence: float
    maximum_share_cost: float


def _candidate_contract(
    config: TournamentDataConfig, panel_manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    groups = {name: tuple(values) for name, values in panel_manifest["feature_groups"].items()}
    candidates: dict[str, dict[str, Any]] = {}
    for row in config.raw["candidates"]:
        feature_groups = tuple(row["feature_groups"])
        features = tuple(dict.fromkeys(name for group in feature_groups for name in groups[group]))
        forbidden = [
            name
            for name in features
            if any(token in name.lower() for token in FORBIDDEN_INFERENCE_TOKENS)
        ]
        if forbidden:
            raise RuntimeError(f"forbidden inference features for {row['name']}: {forbidden}")
        candidates[row["name"]] = {
            "name": row["name"],
            "feature_groups": feature_groups,
            "features": features,
            "time_specialist": bool(row.get("time_specialist", False)),
            "description": row["description"],
        }
    expected = {
        "full_history_price_control",
        "full_history_refprice_residual",
        "binance_flow",
        "kraken_crossvenue",
        "settlement_aligned_oracle",
        "dual_venue_flow_agreement",
        "multivenue_consensus",
        "time_specialist_ensemble",
    }
    if set(candidates) != expected:
        raise RuntimeError("candidate roster differs from the frozen tournament contract")
    return candidates


def _matrix(
    frame: pl.DataFrame, features: tuple[str, ...], neutralized: tuple[int, ...] = ()
) -> np.ndarray:
    matrix = frame.select(pl.col(name).cast(pl.Float64) for name in features).to_numpy()
    if neutralized:
        matrix[:, neutralized] = 0.0
    return matrix


def _fit_tree(
    frame: pl.DataFrame, features: tuple[str, ...], config: TournamentDataConfig, seed: int
) -> TreeModel:
    markets = frame.select("market_id", "window_start").unique("market_id").sort("window_start")
    if markets.height < 250:
        raise RuntimeError(f"insufficient training markets: {markets.height}")
    fraction = float(config.raw["model"]["calibration_fraction"])
    boundary = markets["window_start"][max(1, int(markets.height * (1.0 - fraction)))]
    fit = frame.filter(pl.col("window_start") < boundary)
    calibration = frame.filter(pl.col("window_start") >= boundary)
    matrix = _matrix(fit, features)
    neutralized = tuple(int(index) for index in np.flatnonzero(np.isnan(matrix).all(axis=0)))
    if neutralized:
        matrix[:, neutralized] = 0.0
    spec = config.raw["model"]
    estimator = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(spec["learning_rate"]),
        max_iter=int(spec["max_iter"]),
        max_leaf_nodes=int(spec["max_leaf_nodes"]),
        min_samples_leaf=int(spec["min_samples_leaf"]),
        l2_regularization=float(spec["l2_regularization"]),
        max_bins=int(spec["max_bins"]),
        early_stopping=False,
        random_state=seed,
    ).fit(matrix, fit["label_up"].to_numpy())
    raw = np.clip(
        estimator.predict_proba(_matrix(calibration, features, neutralized))[:, 1], 1e-6, 1 - 1e-6
    )
    calibrator: LogisticRegression | None = None
    if calibration["label_up"].n_unique() == 2:
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=seed).fit(
            logit(raw).reshape(-1, 1), calibration["label_up"].to_numpy()
        )
    return TreeModel(features, estimator, calibrator, neutralized)


def _predict_tree(model: TreeModel, frame: pl.DataFrame) -> np.ndarray:
    raw = np.clip(
        model.estimator.predict_proba(_matrix(frame, model.features, model.neutralized_columns))[
            :, 1
        ],
        1e-6,
        1 - 1e-6,
    )
    if model.calibrator is None:
        return raw
    return model.calibrator.predict_proba(logit(raw).reshape(-1, 1))[:, 1]


def _timing_bands(config: TournamentDataConfig) -> dict[str, tuple[int, int]]:
    names = ("early", "middle", "late")
    return {
        name: tuple(int(value) for value in band)
        for name, band in zip(names, config.raw["entry"]["timing_bands"], strict=True)
    }


def _fit_candidate(
    name: str,
    contract: dict[str, Any],
    frame: pl.DataFrame,
    config: TournamentDataConfig,
    seed: int,
    fit_end: datetime,
) -> CandidateModel:
    bands = _timing_bands(config) if contract["time_specialist"] else {"all": (60, 240)}
    models = {}
    for offset, (band_name, (start, end)) in enumerate(bands.items()):
        subset = frame.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        models[band_name] = _fit_tree(subset, contract["features"], config, seed + offset)
    return CandidateModel(name, contract["feature_groups"], models, bands, fit_end.isoformat())


def _predict_candidate(model: CandidateModel, frame: pl.DataFrame) -> np.ndarray:
    probability = np.full(frame.height, np.nan)
    seconds = frame["seconds_elapsed"].to_numpy()
    for band_name, (start, end) in model.timing_bands.items():
        positions = np.flatnonzero((seconds >= start) & (seconds <= end))
        if len(positions):
            probability[positions] = _predict_tree(model.models[band_name], frame[positions])
    if not np.isfinite(probability).all():
        raise RuntimeError(f"{model.name} produced nonfinite probabilities")
    return probability


def _ece(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if selected.any():
            score += selected.mean() * abs(probability[selected].mean() - labels[selected].mean())
    return float(score)


def predictive_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    labels = frame["label_up"].to_numpy().astype(float)
    probability = frame["probability"].to_numpy().astype(float)
    by_fold = []
    for key, part in frame.partition_by("fold", as_dict=True).items():
        fold = key[0] if isinstance(key, tuple) else key
        y = part["label_up"].to_numpy().astype(float)
        p = part["probability"].to_numpy().astype(float)
        by_fold.append(
            {
                "fold": fold,
                "rows": part.height,
                "markets": part["market_id"].n_unique(),
                "brier_score": float(np.mean((p - y) ** 2)),
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "accuracy": float(np.mean((p >= 0.5) == y)),
            }
        )
    timing = []
    for name, start, end in (("60_120", 60, 120), ("125_180", 125, 180), ("185_240", 185, 240)):
        part = frame.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        y = part["label_up"].to_numpy().astype(float)
        p = part["probability"].to_numpy().astype(float)
        timing.append(
            {
                "band": name,
                "rows": part.height,
                "markets": part["market_id"].n_unique(),
                "brier_score": float(np.mean((p - y) ** 2)),
                "accuracy": float(np.mean((p >= 0.5) == y)),
            }
        )
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "brier_score": float(np.mean((probability - labels) ** 2)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "accuracy": float(np.mean((probability >= 0.5) == labels)),
        "ece_15": _ece(labels, probability),
        "mean_probability": float(probability.mean()),
        "positive_rate": float(labels.mean()),
        "by_fold": by_fold,
        "by_timing_band": timing,
    }


def _oof_predictions(
    panel: pl.DataFrame,
    candidates: dict[str, dict[str, Any]],
    config: TournamentDataConfig,
    checkpoint: Path,
) -> pl.DataFrame:
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
        test = panel.filter((pl.col("window_start") >= start) & (pl.col("window_start") < end))
        train_markets = set(train["market_id"].unique().to_list())
        test_markets = set(test["market_id"].unique().to_list())
        if train_markets & test_markets:
            raise RuntimeError(f"market contamination in fold {fold['name']}")
        for candidate_index, (name, contract) in enumerate(candidates.items()):
            if (name, fold["name"]) in existing:
                continue
            model = _fit_candidate(
                name, contract, train, config, seed + 100 * fold_index + candidate_index, start
            )
            probability = _predict_candidate(model, test)
            piece = test.select(*KEY_COLUMNS, "label_up").with_columns(
                pl.lit(fold["name"]).alias("fold"),
                pl.lit(name).alias("candidate"),
                pl.Series("probability", probability),
            )
            pieces.append(piece)
            combined = pl.concat(pieces, how="vertical_relaxed", rechunk=True)
            combined.write_parquet(checkpoint, compression="zstd", statistics=True)
            test_markets_count = test["market_id"].n_unique()
            print(
                f"tournament OOF: {fold['name']} {name} {test_markets_count:,} markets",
                flush=True,
            )
    return pl.concat(pieces, how="vertical_relaxed", rechunk=True).sort(
        ["candidate", "window_start", "market_id", "seconds_elapsed"]
    )


def _opportunities(
    predictions: pl.DataFrame, panel: pl.DataFrame, config: TournamentDataConfig
) -> pl.DataFrame:
    quantity = int(config.raw["execution"]["quantity"])
    up = f"up_ask_vwap_{quantity}"
    down = f"down_ask_vwap_{quantity}"
    joined = predictions.join(
        panel.select(*KEY_COLUMNS, up, down, "fee_rate"),
        on=list(KEY_COLUMNS),
        how="left",
        validate="m:1",
    )
    reserve = float(config.raw["execution"]["execution_reserve_per_share"])
    return (
        joined.with_columns(
            pl.when(pl.col("probability") >= 0.5)
            .then(pl.lit("up"))
            .otherwise(pl.lit("down"))
            .alias("side"),
            pl.when(pl.col("probability") >= 0.5)
            .then(pl.col("probability"))
            .otherwise(1.0 - pl.col("probability"))
            .alias("selected_probability"),
            pl.when(pl.col("probability") >= 0.5)
            .then(pl.col(up))
            .otherwise(pl.col(down))
            .alias("share_cost"),
        )
        .with_columns(
            (
                pl.col("fee_rate").fill_null(0.0)
                * pl.col("share_cost")
                * (1.0 - pl.col("share_cost"))
            ).alias("fee_per_share")
        )
        .with_columns(
            (
                pl.col("selected_probability")
                - pl.col("share_cost")
                - pl.col("fee_per_share")
                - reserve
            ).alias("expected_edge")
        )
    )


def _select_trades(
    opportunities: pl.DataFrame, policy: ExecutionPolicy, config: TournamentDataConfig
) -> pl.DataFrame:
    quantity = int(config.raw["execution"]["quantity"])
    reserve = float(config.raw["execution"]["execution_reserve_per_share"])
    stress = float(config.raw["execution"]["stress_slippage_per_share"])
    eligible = (
        opportunities.filter(
            pl.col("share_cost").is_not_null()
            & pl.col("share_cost").is_finite()
            & (pl.col("share_cost") > 0)
            & (pl.col("share_cost") <= policy.maximum_share_cost)
            & (pl.col("selected_probability") >= policy.minimum_confidence)
            & (pl.col("expected_edge") >= policy.minimum_edge)
        )
        .sort(["market_id", "seconds_elapsed"])
        .group_by("market_id", maintain_order=True)
        .first()
    )
    won = ((pl.col("side") == "up") & (pl.col("label_up") == 1)) | (
        (pl.col("side") == "down") & (pl.col("label_up") == 0)
    )
    return eligible.with_columns(won.alias("won")).with_columns(
        pl.when(pl.col("won"))
        .then(1.0 - pl.col("share_cost"))
        .otherwise(-pl.col("share_cost"))
        .sub(pl.col("fee_per_share") + reserve)
        .mul(quantity)
        .alias("net_pnl"),
        pl.when(pl.col("won"))
        .then(1.0 - pl.col("share_cost") - stress)
        .otherwise(-(pl.col("share_cost") + stress))
        .sub(pl.col("fee_per_share") + reserve)
        .mul(quantity)
        .alias("stress_net_pnl"),
    )


def economic_metrics(trades: pl.DataFrame, total_markets: int) -> dict[str, Any]:
    if trades.is_empty():
        return {
            "trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": None,
            "net_pnl": 0.0,
            "stress_net_pnl": 0.0,
            "profit_factor": None,
            "loss_recovery_wins": None,
            "market_coverage": 0.0,
            "average_share_cost": None,
            "average_entry_second": None,
            "maximum_drawdown": 0.0,
        }
    pnl = trades["net_pnl"].to_numpy().astype(float)
    wins = pnl[pnl > 0]
    losses = -pnl[pnl < 0]
    cumulative = np.cumsum(pnl)
    drawdown = np.maximum.accumulate(np.r_[0.0, cumulative])[1:] - cumulative
    winning = int((pnl > 0).sum())
    losing = int((pnl < 0).sum())
    return {
        "trades": trades.height,
        "winning_trades": winning,
        "losing_trades": losing,
        "win_rate": winning / trades.height,
        "net_pnl": float(pnl.sum()),
        "stress_net_pnl": float(trades["stress_net_pnl"].sum()),
        "gross_profit": float(wins.sum()),
        "gross_loss": float(losses.sum()),
        "profit_factor": float(wins.sum() / losses.sum()) if losses.sum() > 0 else None,
        "loss_recovery_wins": float(losses.mean() / wins.mean())
        if len(wins) and len(losses)
        else None,
        "market_coverage": trades["market_id"].n_unique() / max(total_markets, 1),
        "average_share_cost": float(trades["share_cost"].mean()),
        "average_entry_second": float(trades["seconds_elapsed"].mean()),
        "maximum_drawdown": float(drawdown.max(initial=0.0)),
    }


def _policy_score(trades: pl.DataFrame) -> float:
    if trades.is_empty():
        return -math.inf
    pnl = trades["net_pnl"].to_numpy().astype(float)
    mean = float(pnl.mean())
    standard_error = float(pnl.std(ddof=1) / math.sqrt(len(pnl))) if len(pnl) > 1 else abs(mean)
    fold_pnl = trades.group_by("fold").agg(pl.col("net_pnl").sum()).get_column("net_pnl").to_numpy()
    stability = float(np.median(fold_pnl)) if len(fold_pnl) else 0.0
    return mean - standard_error + 0.01 * stability + 0.001 * math.sqrt(len(pnl))


def _select_policy(
    opportunities: pl.DataFrame, config: TournamentDataConfig
) -> tuple[ExecutionPolicy, dict[str, Any]]:
    best: tuple[float, ExecutionPolicy, pl.DataFrame] | None = None
    execution = config.raw["execution"]
    for edge in execution["minimum_edges"]:
        for confidence in execution["minimum_confidences"]:
            for cost in execution["maximum_share_costs"]:
                policy = ExecutionPolicy(float(edge), float(confidence), float(cost))
                trades = _select_trades(opportunities, policy, config)
                score = _policy_score(trades)
                candidate = (score, policy, trades)
                if best is None or score > best[0]:
                    best = candidate
    if best is None or not math.isfinite(best[0]):
        fallback = ExecutionPolicy(
            min(float(value) for value in execution["minimum_edges"]),
            min(float(value) for value in execution["minimum_confidences"]),
            max(float(value) for value in execution["maximum_share_costs"]),
        )
        return fallback, {
            "selection_score": None,
            "development": economic_metrics(pl.DataFrame(), opportunities["market_id"].n_unique()),
        }
    score, policy, trades = best
    return policy, {
        "selection_score": score,
        "development": economic_metrics(trades, opportunities["market_id"].n_unique()),
    }


def _prior_reference(config: TournamentDataConfig) -> dict[str, Any]:
    metrics_path = config.package_root / config.raw["paths"]["prior_champion_metrics"]
    artifact_path = config.package_root / config.raw["paths"]["prior_champion_artifact"]
    metrics = json.loads(metrics_path.read_text())
    name = "frozen_bridge_control"
    return {
        "name": "prior_frozen_champion_reference",
        "source_candidate": name,
        "artifact_path": str(artifact_path.relative_to(config.package_root)),
        "artifact_sha256": file_sha256(artifact_path),
        "observation_seconds": metrics.get("observation_seconds"),
        "predictive": metrics.get("predictive", {}).get(name),
        "prospective": metrics.get("prospective", {}).get(name),
        "directly_comparable_60_240": False,
        "reason": "The immutable prior model has a 60..180 input contract and is retained as a reference, not refit or extrapolated to 240 seconds.",
    }


def _report_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# Multi-Venue Early-Entry Tournament",
        "",
        f"Run: `{metrics['run_id']}`  ",
        f"Qualification: **{metrics['qualification_status']}**  ",
        f"Nominated predictive model: **{metrics['nominated_predictive_model']}**",
        "",
        "## Sealed high-level results",
        "",
        "| Candidate | PnL | Coverage | Wins | Losses | Win rate | Loss recovery wins | Brier | Profit factor | Avg cost | Avg entry |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in metrics["candidate_names"]:
        predictive = metrics["sealed_predictive"][name]
        economic = metrics["sealed_economic"][name]

        def fmt(value: Any, digits: int = 4) -> str:
            return "—" if value is None else f"{value:.{digits}f}"

        lines.append(
            f"| {name} | {economic['net_pnl']:.2f} | {economic['market_coverage']:.2%} | {economic['winning_trades']} | {economic['losing_trades']} | {fmt(economic['win_rate'], 3)} | {fmt(economic['loss_recovery_wins'], 3)} | {predictive['brier_score']:.4f} | {fmt(economic['profit_factor'], 3)} | {fmt(economic['average_share_cost'], 3)} | {fmt(economic['average_entry_second'], 1)} |"
        )
    lines.extend(
        (
            "",
            "## Integrity",
            "",
            "- Training and calibration use only markets ending before the sealed boundary.",
            "- The August 14–28 seal was evaluated after candidate and policy selection.",
            "- Optional-source absence does not remove otherwise usable core rows or markets.",
            "- Kraken candles and prints are available only after their one-second bucket closes; Kraken L2 is excluded.",
            "- No database writes, ingesters, source additions, tables, runtime exports, deployments, or trading-process changes were made.",
            "- The prior immutable 60–180 champion is reported as a provenance reference and is not presented as directly comparable at 181–240 seconds.",
        )
    )
    return "\n".join(lines) + "\n"


def train_tournament(config: TournamentDataConfig, *, force: bool = False) -> Path:
    panel, panel_manifest = build_panel(config, force=force)
    candidates = _candidate_contract(config, panel_manifest)
    run_id = _run_id()
    run_dir = config.results / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledgers = run_dir / "ledgers"
    ledgers.mkdir()
    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "fit_end_exclusive": config.fit_end.isoformat(),
        "sealed_start_inclusive": config.sealed_start.isoformat(),
        "sealed_end_exclusive": config.sealed_end.isoformat(),
        "folds": config.raw["folds"],
        "entry_seconds": list(ENTRY_SECONDS),
        "market_disjoint": True,
    }
    _write_json(run_dir / "split-manifest.json", split_manifest)

    preseal = panel.filter(pl.col("window_start") < config.fit_end)
    sealed = panel.filter(
        (pl.col("window_start") >= config.sealed_start)
        & (pl.col("window_start") < config.sealed_end)
    )
    if set(preseal["market_id"].unique().to_list()) & set(sealed["market_id"].unique().to_list()):
        raise RuntimeError("sealed markets entered pre-seal training")
    checkpoint = config.cache / "oof-predictions.parquet"
    oof = _oof_predictions(preseal, candidates, config, checkpoint)
    oof.write_parquet(
        ledgers / "candidate-oof-predictions.parquet", compression="zstd", statistics=True
    )

    oof_metrics = {}
    selected_policies: dict[str, ExecutionPolicy] = {}
    policy_evidence = {}
    for name in candidates:
        candidate_oof = oof.filter(pl.col("candidate") == name)
        oof_metrics[name] = predictive_metrics(candidate_oof)
        policy, evidence = _select_policy(_opportunities(candidate_oof, preseal, config), config)
        selected_policies[name] = policy
        policy_evidence[name] = evidence
    ranking = sorted(
        candidates,
        key=lambda name: (
            oof_metrics[name]["brier_score"],
            oof_metrics[name]["log_loss"],
            oof_metrics[name]["ece_15"],
        ),
    )
    nominated = ranking[0]

    final_models = {}
    sealed_predictions = []
    seed = int(config.raw["training"]["random_seed"])
    for index, (name, contract) in enumerate(candidates.items()):
        model = _fit_candidate(
            name, contract, preseal, config, seed + 10_000 + index, config.fit_end
        )
        final_models[name] = model
        sealed_predictions.append(
            sealed.select(*KEY_COLUMNS, "label_up").with_columns(
                pl.lit("sealed_20260814_20260828").alias("fold"),
                pl.lit(name).alias("candidate"),
                pl.Series("probability", _predict_candidate(model, sealed)),
            )
        )
    sealed_prediction_frame = pl.concat(sealed_predictions, how="vertical_relaxed", rechunk=True)
    sealed_prediction_frame.write_parquet(
        ledgers / "sealed-predictions.parquet", compression="zstd", statistics=True
    )
    sealed_predictive = {}
    sealed_economic = {}
    sealed_trades = []
    for name in candidates:
        prediction = sealed_prediction_frame.filter(pl.col("candidate") == name)
        sealed_predictive[name] = predictive_metrics(prediction)
        trades = _select_trades(
            _opportunities(prediction, sealed, config), selected_policies[name], config
        )
        if not trades.is_empty():
            sealed_trades.append(trades.with_columns(pl.lit(name).alias("candidate")))
        sealed_economic[name] = economic_metrics(trades, sealed["market_id"].n_unique())
    trade_frame = (
        pl.concat(sealed_trades, how="diagonal_relaxed", rechunk=True)
        if sealed_trades
        else pl.DataFrame()
    )
    trade_frame.write_parquet(
        ledgers / "sealed-trades.parquet", compression="zstd", statistics=True
    )

    model_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_family": config.raw["training"]["model_family"],
        "producing_commit": _git_revision(config.package_root),
        "run_id": run_id,
        "fit_end_exclusive": config.fit_end.isoformat(),
        "entry_seconds": ENTRY_SECONDS,
        "candidate_contract": candidates,
        "models": final_models,
        "selected_policies": selected_policies,
        "nominated_predictive_model": nominated,
        "runtime_exported": False,
        "deployment_status": "not_deployed",
    }
    artifact = run_dir / "tournament.joblib"
    joblib.dump(model_payload, artifact, compress=3)
    artifact_sha = file_sha256(artifact)
    (run_dir / "tournament.sha256").write_text(f"{artifact_sha}  tournament.joblib\n")
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "producing_commit": model_payload["producing_commit"],
        "model_family": model_payload["model_family"],
        "candidate_names": list(candidates),
        "candidate_count_trained": len(candidates),
        "prior_champion_reference": _prior_reference(config),
        "nominated_predictive_model": nominated,
        "predictive_ranking_preseal": ranking,
        "oof_predictive": oof_metrics,
        "selected_policies": {name: asdict(policy) for name, policy in selected_policies.items()},
        "policy_selection_evidence": policy_evidence,
        "sealed_predictive": sealed_predictive,
        "sealed_economic": sealed_economic,
        "source_panel": panel_manifest,
        "split_manifest": split_manifest,
        "artifact_sha256": artifact_sha,
        "qualification_status": "trained_and_sealed_not_deployed",
        "integrity": {
            "passed": True,
            "market_disjoint": True,
            "sealed_opened_after_selection": True,
            "optional_missingness_preserves_rows": True,
            "entry_schedule_exact": list(ENTRY_SECONDS),
            "forbidden_inference_tokens": list(FORBIDDEN_INFERENCE_TOKENS),
            "kraken_l2_included": False,
            "database_mutations": False,
            "new_tables": False,
            "new_ingesters": False,
            "new_sources": False,
            "runtime_exported": False,
            "deployed": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "candidate-contract.json", candidates)
    _write_json(
        run_dir / "source-manifest.json",
        json.loads((config.cache / "complete-source-manifest.json").read_text()),
    )
    (run_dir / "report.md").write_text(_report_markdown(metrics))
    _write_json(
        run_dir / "completion.json",
        {
            "run_id": run_id,
            "artifact_sha256": artifact_sha,
            "metrics_sha256": file_sha256(run_dir / "metrics.json"),
            "completed": True,
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("command", choices=("extract", "prepare", "train", "all"))
    arguments = parser.parse_args()
    config = load_data_config(arguments.config)
    if arguments.command in {"extract", "all"}:
        extract_sources(config, force=arguments.force)
    if arguments.command in {"prepare", "all"}:
        build_panel(config, force=arguments.force)
    if arguments.command in {"train", "all"}:
        print(train_tournament(config, force=arguments.force))


if __name__ == "__main__":
    main()
