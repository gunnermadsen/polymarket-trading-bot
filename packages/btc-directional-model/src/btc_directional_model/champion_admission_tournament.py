"""Train admission-only challengers around two immutable directional champions."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .continuous_edge_training import VWAP_QUANTITIES
from .core_extract import file_sha256
from .extended_specialist_strategy_tournament import (
    _bridge_join,
    _fit_distilled,
    _realized_opportunities,
    _score_model,
)
from .middle_strategy_tournament import (
    KEY_COLUMNS,
    _select_trades,
    economic_metrics,
    predictive_metrics,
)

SCHEMA_VERSION = "btc-champion-admission-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-champion-admission-artifact-v1"
MODULE = "btc_directional_model.champion_admission_tournament"
if __name__ == "__main__":
    sys.modules[MODULE] = sys.modules[__name__]

ALL_CANDIDATES = (
    "official_specialist_control",
    "bridge_aware_control",
    "official_vwap_admission",
    "bridge_vwap_admission",
    "official_loss_severity_veto",
    "bridge_capacity_curve_veto",
    "specialist_consensus_admission",
)
LEARNED_CANDIDATES = ALL_CANDIDATES[2:]


@dataclass(frozen=True)
class AdmissionModel:
    features: tuple[str, ...]
    profitable: HistGradientBoostingClassifier
    stress_edge: HistGradientBoostingRegressor
    loss_severity: HistGradientBoostingRegressor
    base: str
    kind: str


AdmissionModel.__module__ = MODULE


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _write_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary, compress=3)
    temporary.replace(path)


def _load_config(path: Path) -> tuple[Path, dict[str, Any]]:
    package_root = path.resolve().parents[1]
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if tuple(row["name"] for row in raw["candidates"]) != ALL_CANDIDATES:
        raise RuntimeError("frozen champion-admission roster changed")
    if any("authentic_only" in key.lower() for key in raw):
        raise RuntimeError("authentic-only filtering is prohibited")
    return package_root, raw


def _paths(root: Path, raw: dict[str, Any]) -> dict[str, Path]:
    return {key: root / value for key, value in raw["paths"].items()}


def _bands(raw: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (f"{start}_{end}", int(start), int(end)) for start, end in raw["entry"]["competition_bands"]
    )


def _source_manifest(root: Path, raw: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(root, raw)
    for name in (
        "panel",
        "panel_manifest",
        "bridge_panel",
        "bridge_manifest",
        "champion_artifact",
        "champion_metrics",
        "official_oof",
        "development_predictions",
    ):
        if not paths[name].is_file():
            raise RuntimeError(f"required immutable input is missing: {name}")
    panel_manifest = json.loads(paths["panel_manifest"].read_text())
    champion_metrics = json.loads(paths["champion_metrics"].read_text())
    identities = {
        "panel": file_sha256(paths["panel"]),
        "panel_manifest": file_sha256(paths["panel_manifest"]),
        "bridge_panel": file_sha256(paths["bridge_panel"]),
        "bridge_manifest": file_sha256(paths["bridge_manifest"]),
        "champion_artifact": file_sha256(paths["champion_artifact"]),
        "champion_metrics": file_sha256(paths["champion_metrics"]),
        "official_oof": file_sha256(paths["official_oof"]),
        "development_predictions": file_sha256(paths["development_predictions"]),
    }
    if identities["panel"] != panel_manifest["sha256"]:
        raise RuntimeError("full-history panel identity changed")
    if identities["champion_artifact"] != champion_metrics["artifact_sha256"]:
        raise RuntimeError("champion artifact identity changed")
    return {
        "identities": identities,
        "panel": panel_manifest,
        "champion_run": champion_metrics["run_id"],
        "champion_producing_commit": champion_metrics["producing_commit"],
        "existing_champion_collection_unchanged": True,
        "database_mutations": False,
        "new_sources": False,
        "new_tables": False,
        "new_schemas": False,
        "new_ingesters": False,
    }


def _matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return np.column_stack(
        [frame[name].cast(pl.Float64).fill_nan(None).fill_null(0.0).to_numpy() for name in features]
    )


def _market_weights(frame: pl.DataFrame) -> np.ndarray:
    counts = frame.group_by("market_id").len().rename({"len": "market_rows"})
    return (
        frame.join(counts, on="market_id", how="left")["market_rows"]
        .cast(pl.Float64)
        .pow(-1)
        .to_numpy()
        .copy()
    )


def _context_columns(schema: pl.Schema) -> tuple[str, ...]:
    requested = [
        *(f"up_ask_vwap_{quantity}" for quantity in VWAP_QUANTITIES),
        *(f"down_ask_vwap_{quantity}" for quantity in VWAP_QUANTITIES),
        "pm_vwap5_overround",
        "pm_vwap50_overround",
        "pm_vwap200_overround",
        "pm_up_depth_log",
        "pm_down_depth_log",
        "pm_depth_imbalance",
        "spot_l2_imbalance_20",
        "spot_l2_spread_bps",
        "kraken_l2_update_imbalance_30s",
        "kraken_l2_quantity_imbalance_30s",
        "kraken_l2_age_seconds",
        "has_kraken_l2",
    ]
    return tuple(name for name in requested if name in schema)


def _load_panel(
    paths: dict[str, Path], start: datetime, end: datetime, extra: tuple[str, ...] = ()
) -> pl.DataFrame:
    schema = pl.scan_parquet(paths["panel"]).collect_schema()
    columns = tuple(
        dict.fromkeys(
            (
                *KEY_COLUMNS,
                "label_up",
                "fee_rate",
                "up_ask_vwap_5",
                "down_ask_vwap_5",
                "pm_up_book_age_seconds",
                "pm_down_book_age_seconds",
                *_context_columns(schema),
                *extra,
            )
        )
    )
    return (
        pl.scan_parquet(paths["panel"])
        .select(columns)
        .filter((pl.col("window_start") >= start) & (pl.col("window_start") < end))
        .collect()
    )


def _bridge_oof(
    root: Path, raw: dict[str, Any], artifact: dict[str, Any], run_dir: Path
) -> pl.DataFrame:
    ledger = run_dir / "ledgers" / "bridge-oof-predictions.parquet"
    if ledger.is_file():
        return pl.read_parquet(ledger)
    paths = _paths(root, raw)
    features = tuple(artifact["candidate_contract"]["bridge_aware_specialist"]["features"])
    columns = tuple(dict.fromkeys((*KEY_COLUMNS, "label_up", *features)))
    source_start = datetime.fromisoformat(raw["windows"]["source_start"])
    fit_end = datetime.fromisoformat(raw["windows"]["fit_end"])
    full = (
        pl.scan_parquet(paths["panel"])
        .select(columns)
        .filter((pl.col("window_start") >= source_start) & (pl.col("window_start") < fit_end))
        .collect()
    )
    pieces = []
    for index, fold in enumerate(raw["folds"]):
        start, end = (
            datetime.fromisoformat(fold["test_start"]),
            datetime.fromisoformat(fold["test_end"]),
        )
        training = full.filter(pl.col("window_start") < start)
        validation = full.filter((pl.col("window_start") >= start) & (pl.col("window_start") < end))
        if training.is_empty() or validation.is_empty():
            continue
        checkpoint = run_dir / "checkpoints" / f"bridge-oof-{fold['name']}.joblib"
        if checkpoint.is_file():
            model = joblib.load(checkpoint)
        else:
            model = _fit_distilled(
                _bridge_join(root, raw, training),
                features,
                raw,
                int(raw["training"]["random_seed"]) + 100 + index,
                target="bridge_probability_target",
                weight_column="label_weight",
            )
            _write_joblib(checkpoint, model)
        pieces.append(_score_model(model, validation, "bridge_aware_specialist", fold["name"]))
    output = pl.concat(pieces, how="vertical_relaxed", rechunk=True)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    output.write_parquet(ledger, compression="zstd", statistics=True)
    return output


def _consensus(
    official: pl.DataFrame, bridge: pl.DataFrame, candidate: str = "consensus"
) -> pl.DataFrame:
    other = bridge.select(*KEY_COLUMNS, pl.col("probability").alias("bridge_probability"))
    return (
        official.join(other, on=list(KEY_COLUMNS), how="inner", validate="1:1")
        .with_columns(
            ((pl.col("probability") + pl.col("bridge_probability")) / 2).alias("probability"),
            (pl.col("probability") - pl.col("bridge_probability"))
            .abs()
            .alias("prediction_disagreement"),
            pl.lit(candidate).alias("candidate"),
        )
        .drop("bridge_probability")
    )


def _admission_frame(
    predictions: pl.DataFrame, counterpart: pl.DataFrame, panel: pl.DataFrame, raw: dict[str, Any]
) -> pl.DataFrame:
    opportunities = _realized_opportunities(predictions, panel, raw)
    context_names = [
        name for name in _context_columns(panel.schema) if name not in opportunities.columns
    ]
    output = opportunities.join(
        panel.select(*KEY_COLUMNS, *context_names), on=list(KEY_COLUMNS), how="left", validate="m:1"
    )
    other = counterpart.select(*KEY_COLUMNS, pl.col("probability").alias("counterpart_probability"))
    output = output.join(other, on=list(KEY_COLUMNS), how="left", validate="m:1")
    disagreement = (
        pl.col("prediction_disagreement")
        if "prediction_disagreement" in output.columns
        else (pl.col("probability") - pl.col("counterpart_probability")).abs().fill_null(0.0)
    )
    selected = []
    for quantity in VWAP_QUANTITIES:
        selected.append(
            pl.when(pl.col("side") == "up")
            .then(pl.col(f"up_ask_vwap_{quantity}"))
            .otherwise(pl.col(f"down_ask_vwap_{quantity}"))
            .alias(f"selected_vwap_{quantity}")
        )
    return (
        output.sort(["market_id", "seconds_elapsed"])
        .with_columns(
            *selected,
            pl.max_horizontal("probability", 1 - pl.col("probability")).alias("confidence"),
            disagreement.alias("prediction_disagreement"),
            (pl.col("probability") - pl.col("probability").shift(1).over("market_id"))
            .fill_null(0.0)
            .alias("probability_change_5s"),
            (pl.col("probability") - pl.col("probability").shift(3).over("market_id"))
            .fill_null(0.0)
            .alias("probability_change_15s"),
            pl.col("share_cost")
            .cut([0.65, 0.80, 0.95], labels=["0", "1", "2", "3"])
            .cast(pl.Int8)
            .alias("price_bucket_index"),
            (-pl.col("realized_stress_edge")).clip(0.0, 2.0).alias("realized_loss_severity"),
        )
        .with_columns(
            (pl.col("selected_vwap_25") - pl.col("selected_vwap_5")).alias(
                "selected_slippage_5_25"
            ),
            (pl.col("selected_vwap_100") - pl.col("selected_vwap_5")).alias(
                "selected_slippage_5_100"
            ),
            (pl.col("selected_vwap_200") - pl.col("selected_vwap_5")).alias(
                "selected_slippage_5_200"
            ),
            pl.col("selected_vwap_200")
            .is_not_null()
            .cast(pl.Float64)
            .alias("vwap_curve_available"),
        )
    )


def _feature_names(frame: pl.DataFrame, kind: str) -> tuple[str, ...]:
    common = [
        "probability",
        "confidence",
        "seconds_elapsed",
        "share_cost",
        "expected_edge",
        "price_bucket_index",
        "probability_change_5s",
        "probability_change_15s",
        "pm_vwap5_overround",
        "pm_up_book_age_seconds",
        "pm_down_book_age_seconds",
    ]
    loss = [
        "pm_vwap50_overround",
        "pm_vwap200_overround",
        "pm_depth_imbalance",
        "spot_l2_imbalance_20",
        "spot_l2_spread_bps",
        "kraken_l2_update_imbalance_30s",
        "kraken_l2_quantity_imbalance_30s",
        "kraken_l2_age_seconds",
        "has_kraken_l2",
    ]
    capacity = [
        *(f"selected_vwap_{q}" for q in VWAP_QUANTITIES),
        "selected_slippage_5_25",
        "selected_slippage_5_100",
        "selected_slippage_5_200",
        "vwap_curve_available",
    ]
    names = (
        common
        + (loss if kind in {"loss_severity", "capacity_curve", "consensus"} else [])
        + (capacity if kind in {"capacity_curve", "consensus"} else [])
        + (["prediction_disagreement"] if kind == "consensus" else [])
    )
    return tuple(
        name
        for name in dict.fromkeys(names)
        if name in frame.columns and frame[name].drop_nulls().n_unique() > 1
    )


def _fit_admission(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    raw: dict[str, Any],
    seed: int,
    base: str,
    kind: str,
) -> AdmissionModel:
    valid = frame.filter(
        pl.col("share_cost").is_not_null() & pl.col("realized_stress_edge").is_finite()
    )
    weights = _market_weights(valid)
    spec = raw["admission"]
    params = {
        "learning_rate": float(spec["learning_rate"]),
        "max_iter": int(spec["max_iter"]),
        "max_leaf_nodes": int(spec["max_leaf_nodes"]),
        "min_samples_leaf": int(spec["min_samples_leaf"]),
        "l2_regularization": float(spec["l2_regularization"]),
        "early_stopping": False,
    }
    matrix = _matrix(valid, features)
    profitable = HistGradientBoostingClassifier(**params, random_state=seed).fit(
        matrix, (valid["realized_stress_edge"] > 0).cast(pl.Int8), sample_weight=weights
    )
    stress = HistGradientBoostingRegressor(**params, random_state=seed + 1).fit(
        matrix, valid["realized_stress_edge"], sample_weight=weights
    )
    loss = HistGradientBoostingRegressor(**params, random_state=seed + 2).fit(
        matrix, valid["realized_loss_severity"], sample_weight=weights
    )
    return AdmissionModel(features, profitable, stress, loss, base, kind)


def _attach_model(frame: pl.DataFrame, model: AdmissionModel) -> pl.DataFrame:
    matrix = _matrix(frame, model.features)
    return frame.with_columns(
        pl.Series("admission_probability", model.profitable.predict_proba(matrix)[:, 1]),
        pl.Series("predicted_stress_edge", model.stress_edge.predict(matrix)),
        pl.Series(
            "predicted_loss_severity", np.clip(model.loss_severity.predict(matrix), 0.0, None)
        ),
    )


def _policy_filter(frame: pl.DataFrame, policy: dict[str, float]) -> pl.DataFrame:
    return frame.filter(
        (pl.col("admission_probability") >= policy["minimum_admission_probability"])
        & (pl.col("predicted_stress_edge") >= policy["minimum_predicted_stress_edge"])
        & (pl.col("predicted_loss_severity") <= policy["maximum_predicted_loss"])
        & (pl.col("share_cost") <= policy["maximum_share_cost"])
    )


def _no_veto() -> dict[str, Any]:
    return {
        "minimum_edge": -1.0,
        "minimum_confidence": 0.5,
        "maximum_share_cost": 1.0,
        "abstain": False,
    }


def _score_selection(trades: pl.DataFrame, raw: dict[str, Any]) -> float:
    if trades.is_empty():
        return -math.inf
    pnl = trades["stress_net_pnl"].to_numpy().astype(float)
    standard_error = 0.0 if len(pnl) < 2 else float(pnl.std(ddof=1) / math.sqrt(len(pnl)))
    metrics = economic_metrics(trades, trades["market_id"].n_unique())
    recovery = metrics["loss_recovery_wins"] or 0.0
    penalty = float(raw["execution"]["recovery_penalty"]) * max(
        0.0, recovery - float(raw["execution"]["recovery_soft_target"])
    )
    return float(pnl.mean()) - standard_error - penalty + 0.0005 * math.sqrt(len(pnl))


def _select_policies(
    frame: pl.DataFrame, raw: dict[str, Any], hard_cost_cap: float | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec, policies, evidence = raw["admission"], {}, {}
    for band, start, end in _bands(raw):
        part = frame.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        best = None
        for probability in spec["probability_thresholds"]:
            for edge in spec["stress_edge_thresholds"]:
                for loss in spec["maximum_loss_thresholds"]:
                    for cost in spec["maximum_share_costs"]:
                        if hard_cost_cap is not None and cost > hard_cost_cap:
                            continue
                        policy = {
                            "minimum_admission_probability": float(probability),
                            "minimum_predicted_stress_edge": float(edge),
                            "maximum_predicted_loss": float(loss),
                            "maximum_share_cost": float(cost),
                        }
                        trades = _select_trades(
                            _policy_filter(part, policy), _no_veto(), _config_namespace(raw)
                        )
                        if trades.height < int(spec["minimum_development_trades"]):
                            continue
                        score = _score_selection(trades, raw)
                        if best is None or score > best[0]:
                            best = (score, policy, trades)
        if best is None:
            policy = {
                "minimum_admission_probability": 1.0,
                "minimum_predicted_stress_edge": 1.0,
                "maximum_predicted_loss": 0.0,
                "maximum_share_cost": hard_cost_cap or 0.95,
            }
            policies[band], evidence[band] = (
                policy,
                {
                    "selection_score": None,
                    "development": economic_metrics(part.head(0), part["market_id"].n_unique()),
                },
            )
        else:
            score, policy, trades = best
            policies[band], evidence[band] = (
                policy,
                {
                    "selection_score": score,
                    "development": economic_metrics(trades, part["market_id"].n_unique()),
                },
            )
    return policies, evidence


def _config_namespace(raw: dict[str, Any]) -> Any:
    return type("Config", (), {"raw": raw})()


def _eligible_by_band(
    frame: pl.DataFrame, policies: dict[str, Any], raw: dict[str, Any], learned: bool
) -> pl.DataFrame:
    pieces = []
    for band, start, end in _bands(raw):
        part = frame.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        policy = policies[band]
        if learned:
            part = _policy_filter(part, policy)
        else:
            part = part.filter(
                (pl.col("share_cost") <= policy["maximum_share_cost"])
                & (pl.col("selected_probability") >= policy["minimum_confidence"])
                & (pl.col("expected_edge") >= policy["minimum_edge"])
            )
        pieces.append(part)
    return pl.concat(pieces, how="vertical_relaxed")


def _evaluate(
    frame: pl.DataFrame, policies: dict[str, Any], raw: dict[str, Any], learned: bool
) -> tuple[dict[str, Any], pl.DataFrame]:
    config = _config_namespace(raw)
    total = frame["market_id"].n_unique()
    metrics = {}
    for band, start, end in _bands(raw):
        part = frame.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        eligible = (
            _policy_filter(part, policies[band])
            if learned
            else _eligible_by_band(
                part,
                {band: policies[band]},
                {**raw, "entry": {**raw["entry"], "competition_bands": [[start, end]]}},
                False,
            )
        )
        metrics[band] = economic_metrics(_select_trades(eligible, _no_veto(), config), total)
    combined = _select_trades(_eligible_by_band(frame, policies, raw, learned), _no_veto(), config)
    metrics["combined_60_180"] = economic_metrics(combined, total)
    return metrics, combined


def _capacity_metrics(trades: pl.DataFrame, raw: dict[str, Any]) -> dict[str, Any]:
    reserve = float(raw["execution"]["execution_reserve_per_share"])
    stress = float(raw["execution"]["stress_slippage_per_share"])
    output = {}
    for quantity in VWAP_QUANTITIES:
        cost = (
            pl.when(pl.col("side") == "up")
            .then(pl.col(f"up_ask_vwap_{quantity}"))
            .otherwise(pl.col(f"down_ask_vwap_{quantity}"))
        )
        part = trades.with_columns(cost.alias("capacity_cost")).filter(
            pl.col("capacity_cost").is_not_null()
        )
        part = part.with_columns(
            (
                pl.col("fee_rate").fill_null(0.0)
                * pl.col("capacity_cost")
                * (1 - pl.col("capacity_cost"))
            ).alias("capacity_fee_per_share")
        ).with_columns(
            (
                pl.when(pl.col("won"))
                .then(1 - pl.col("capacity_cost"))
                .otherwise(-pl.col("capacity_cost"))
                - pl.col("capacity_fee_per_share")
                - reserve
            )
            .mul(quantity)
            .alias("capacity_pnl"),
            (
                pl.when(pl.col("won"))
                .then(1 - pl.col("capacity_cost") - stress)
                .otherwise(-(pl.col("capacity_cost") + stress))
                - pl.col("capacity_fee_per_share")
                - reserve
            )
            .mul(quantity)
            .alias("capacity_stress_pnl"),
        )
        values = part["capacity_pnl"].to_numpy().astype(float)
        wins, losses = values[values > 0], -values[values < 0]
        output[str(quantity)] = {
            "trades": part.height,
            "net_pnl": float(values.sum()),
            "stress_net_pnl": float(part["capacity_stress_pnl"].sum()),
            "profit_factor": float(wins.sum() / losses.sum()) if losses.sum() else None,
            "expectancy_per_trade": float(values.mean()) if len(values) else None,
        }
    return output


def _report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Champion Admission Tournament",
        "",
        f"Run: `{metrics['run_id']}`",
        f"Qualification: **{metrics['qualification_status']}**",
        "",
        "## Fixed-holdout high-level results",
        "",
        "| Candidate | Admission | PnL | Stress PnL | PF | Expectancy | Coverage | W/L | Win rate | Recovery wins/loss | Avg entry | Avg cost | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in ALL_CANDIDATES:
        row = metrics["sealed_economic"][candidate]["combined_60_180"]
        pred = metrics["sealed_predictive"][candidate]
        fmt = lambda value, digits=3: "—" if value is None else f"{value:.{digits}f}"
        lines.append(
            f"| {candidate} | {metrics['candidate_contract'][candidate]['admission']} | {row['net_pnl']:.2f} | {row['stress_net_pnl']:.2f} | {fmt(row['profit_factor'])} | {fmt(row['expectancy_per_trade'])} | {row['market_coverage']:.2%} | {row['winning_trades']}/{row['losing_trades']} | {fmt(row['win_rate'], 2)} | {fmt(row['loss_recovery_wins'])} | {fmt(row['average_entry_second'], 1)} | {fmt(row['average_share_cost'])} | {fmt(pred['brier_score'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            "- The two directional champions and the existing seven-model champion collection were not modified.",
            "- Admission models were trained separately from chronological OOF predictor outputs and packaged with immutable champion references.",
            "- VWAP and Polymarket execution fields were admission-only and never entered the directional predictors.",
            "- No best-later target, database write, table, schema, ingester, source, deployment, or image build was used.",
        ]
    )
    for limitation in metrics["limitations"]:
        lines.append(f"- Limitation: {limitation}")
    return "\n".join(lines) + "\n"


def train_tournament(config_path: Path, resume_run: str | None = None) -> Path:
    root, raw = _load_config(config_path)
    paths = _paths(root, raw)
    source = _source_manifest(root, raw)
    run_id = resume_run or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = paths["committed_results"] / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "ledgers").mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "source-manifest.json", source)
    artifact = joblib.load(paths["champion_artifact"])
    official_model = artifact["models"]["extended_specialist_official"]
    bridge_model = artifact["models"]["bridge_aware_specialist"]
    official_oof = pl.read_parquet(paths["official_oof"])
    bridge_oof = _bridge_oof(root, raw, artifact, run_dir)
    source_start, fit_end = (
        datetime.fromisoformat(raw["windows"]["source_start"]),
        datetime.fromisoformat(raw["windows"]["fit_end"]),
    )
    development_start, development_end = (
        datetime.fromisoformat(raw["windows"]["development_start"]),
        datetime.fromisoformat(raw["windows"]["development_end"]),
    )
    sealed_start, sealed_end = (
        datetime.fromisoformat(raw["windows"]["sealed_start"]),
        datetime.fromisoformat(raw["windows"]["sealed_end"]),
    )
    split = {
        "source_start": source_start,
        "fit_end_exclusive": fit_end,
        "development_start": development_start,
        "development_end_exclusive": development_end,
        "sealed_start": sealed_start,
        "sealed_end_exclusive": sealed_end,
        "market_disjoint": True,
        "heldout_excluded_from_admission_fit_and_policy_selection": True,
        "authentic_only_filter": False,
    }
    _write_json(run_dir / "split-manifest.json", split)
    fit_panel = _load_panel(paths, source_start, fit_end)
    development_panel = _load_panel(paths, development_start, development_end)
    development_predictions = pl.read_parquet(paths["development_predictions"])
    official_dev = development_predictions.filter(
        pl.col("candidate") == "extended_specialist_official"
    )
    bridge_dev = development_predictions.filter(pl.col("candidate") == "bridge_aware_specialist")
    frames = {
        "official": (
            _admission_frame(official_oof, bridge_oof, fit_panel, raw),
            _admission_frame(official_dev, bridge_dev, development_panel, raw),
        ),
        "bridge": (
            _admission_frame(bridge_oof, official_oof, fit_panel, raw),
            _admission_frame(bridge_dev, official_dev, development_panel, raw),
        ),
        "consensus": (
            _admission_frame(
                _consensus(official_oof, bridge_oof),
                _consensus(bridge_oof, official_oof),
                fit_panel,
                raw,
            ),
            _admission_frame(
                _consensus(official_dev, bridge_dev),
                _consensus(bridge_dev, official_dev),
                development_panel,
                raw,
            ),
        ),
    }
    candidate_contract = {
        row["name"]: {"base": row["base"], "admission": row["admission"]}
        for row in raw["candidates"]
    }
    models, policies, policy_evidence = {}, {}, {}
    for index, candidate in enumerate(LEARNED_CANDIDATES):
        spec = candidate_contract[candidate]
        base_key = (
            "official"
            if spec["base"] == "extended_specialist_official"
            else "bridge"
            if spec["base"] == "bridge_aware_specialist"
            else "consensus"
        )
        train_frame, dev_frame = frames[base_key]
        checkpoint = run_dir / "checkpoints" / f"admission-{candidate}.joblib"
        if checkpoint.is_file():
            model = joblib.load(checkpoint)
        else:
            features = _feature_names(train_frame, spec["admission"].replace("_below_080", ""))
            model = _fit_admission(
                train_frame,
                features,
                raw,
                int(raw["training"]["random_seed"]) + 1000 * index,
                spec["base"],
                spec["admission"],
            )
            _write_joblib(checkpoint, model)
        models[candidate] = model
        scored_dev = _attach_model(dev_frame, model)
        policies[candidate], policy_evidence[candidate] = _select_policies(
            scored_dev, raw, 0.80 if candidate == "bridge_vwap_admission" else None
        )
    policies["official_specialist_control"] = artifact["programmatic_policies"][
        "extended_specialist_official"
    ]
    policies["bridge_aware_control"] = artifact["programmatic_policies"]["bridge_aware_specialist"]
    selection_time = datetime.now(UTC).isoformat()
    _write_json(
        run_dir / "selection-freeze.json",
        {
            "frozen_at": selection_time,
            "policies": policies,
            "development_evidence": policy_evidence,
            "sealed_loaded_after_freeze": True,
        },
    )
    sealed_features = tuple(dict.fromkeys((*official_model.features, *bridge_model.features)))
    sealed_panel = _load_panel(paths, sealed_start, sealed_end, sealed_features)
    if set(fit_panel["market_id"].unique()) & set(sealed_panel["market_id"].unique()):
        raise RuntimeError("fit and sealed markets overlap")
    official_sealed = _score_model(
        official_model, sealed_panel, "extended_specialist_official", "sealed"
    )
    bridge_sealed = _score_model(bridge_model, sealed_panel, "bridge_aware_specialist", "sealed")
    sealed_frames = {
        "official": _admission_frame(official_sealed, bridge_sealed, sealed_panel, raw),
        "bridge": _admission_frame(bridge_sealed, official_sealed, sealed_panel, raw),
        "consensus": _admission_frame(
            _consensus(official_sealed, bridge_sealed),
            _consensus(bridge_sealed, official_sealed),
            sealed_panel,
            raw,
        ),
    }
    sealed_economic, sealed_predictive, ledgers = {}, {}, []
    base_for = {
        "official_specialist_control": "official",
        "bridge_aware_control": "bridge",
        **{
            name: (
                "official"
                if candidate_contract[name]["base"] == "extended_specialist_official"
                else "bridge"
                if candidate_contract[name]["base"] == "bridge_aware_specialist"
                else "consensus"
            )
            for name in LEARNED_CANDIDATES
        },
    }
    for candidate in ALL_CANDIDATES:
        base_key = base_for[candidate]
        frame = sealed_frames[base_key]
        learned = candidate in LEARNED_CANDIDATES
        scored = _attach_model(frame, models[candidate]) if learned else frame
        economics, trades = _evaluate(scored, policies[candidate], raw, learned)
        economics["capacity_curve"] = _capacity_metrics(trades, raw)
        economics["subperiods"] = {}
        for name, start, end in (
            ("august_20_25", sealed_start, datetime(2026, 8, 26, tzinfo=UTC)),
            ("august_26_31", datetime(2026, 8, 26, tzinfo=UTC), sealed_end),
        ):
            sub = scored.filter((pl.col("window_start") >= start) & (pl.col("window_start") < end))
            economics["subperiods"][name] = _evaluate(sub, policies[candidate], raw, learned)[0][
                "combined_60_180"
            ]
        sealed_economic[candidate] = economics
        prediction = (
            official_sealed
            if base_key == "official"
            else bridge_sealed
            if base_key == "bridge"
            else _consensus(official_sealed, bridge_sealed)
        )
        sealed_predictive[candidate] = predictive_metrics(prediction)
        if not trades.is_empty():
            ledgers.append(trades.with_columns(pl.lit(candidate).alias("strategy_candidate")))
    pl.concat(ledgers, how="diagonal_relaxed", rechunk=True).write_parquet(
        run_dir / "ledgers" / "sealed-trades.parquet", compression="zstd", statistics=True
    )
    _write_json(run_dir / "candidate-contract.json", candidate_contract)
    producing_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    model_artifact = run_dir / "tournament.joblib"
    _write_joblib(
        model_artifact,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "producing_commit": producing_commit,
            "directional_champion_reference": source["identities"]["champion_artifact"],
            "admission_models": models,
            "policies": policies,
            "deployment_status": "not_deployed",
        },
    )
    loaded = joblib.load(model_artifact)
    if set(loaded["admission_models"]) != set(LEARNED_CANDIDATES):
        raise RuntimeError("admission artifact round-trip failed")
    artifact_sha = file_sha256(model_artifact)
    (run_dir / "tournament.sha256").write_text(f"{artifact_sha}  tournament.joblib\n")
    positive = [
        name
        for name in ALL_CANDIDATES
        if sealed_economic[name]["combined_60_180"]["net_pnl"] > 0
        and sealed_economic[name]["combined_60_180"]["stress_net_pnl"] > 0
    ]
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "producing_commit": producing_commit,
        "candidate_names": list(ALL_CANDIDATES),
        "candidate_contract": candidate_contract,
        "positive_expectancy_candidates": positive,
        "qualification_status": "trained_evaluated_not_deployed",
        "sealed_predictive": sealed_predictive,
        "sealed_economic": sealed_economic,
        "source_manifest": source,
        "split_manifest": split,
        "selection_frozen_at": selection_time,
        "artifact_sha256": artifact_sha,
        "integrity": {
            "passed": True,
            "directional_champions_unchanged": True,
            "existing_champion_collection_unchanged": True,
            "admission_trained_from_oof_predictions": True,
            "best_later_target": False,
            "market_disjoint": True,
            "vwap_directional_feature": False,
            "database_mutations": False,
            "new_tables": False,
            "new_schemas": False,
            "new_ingesters": False,
            "new_sources": False,
            "deployed": False,
            "images_rebuilt": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "sklearn": sklearn.__version__,
        },
        "limitations": [
            "Executable VWAP5-200 evidence ends after August 25; August 26-31 contributes predictive metrics but zero economic trades.",
            "The August holdout has been observed in prior research and is chronological rather than epistemically fresh.",
            "Projected PnL assumes recorded ask VWAP was fillable and does not model queue position.",
        ],
    }
    _write_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(_report(metrics))
    _write_json(
        run_dir / "completion.json",
        {
            "completed": True,
            "run_id": run_id,
            "artifact_sha256": artifact_sha,
            "metrics_sha256": file_sha256(run_dir / "metrics.json"),
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-run")
    args = parser.parse_args()
    print(train_tournament(args.config, args.resume_run))


if __name__ == "__main__":
    main()
