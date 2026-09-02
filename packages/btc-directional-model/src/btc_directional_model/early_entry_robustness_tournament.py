"""Train the frozen early-entry robustness tournament without runtime changes."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from .champion_admission_tournament import (
    AdmissionModel,
    _admission_frame,
    _attach_model,
    _capacity_metrics,
    _fit_admission,
    _market_weights,
    _no_veto,
    _policy_filter,
)
from .continuous_edge_training import VWAP_QUANTITIES
from .core_extract import file_sha256
from .extended_specialist_strategy_tournament import _fit_distilled, _score_model
from .kraken_l2_training_data import (
    KRAKEN_L2_FEATURES,
    attach_kraken_l2,
    build_kraken_l2_features,
)
from .middle_strategy_tournament import (
    KEY_COLUMNS,
    _select_trades,
    economic_metrics,
    predictive_metrics,
)

SCHEMA_VERSION = "btc-early-entry-robustness-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-early-entry-robustness-artifact-v1"
MODULE = "btc_directional_model.early_entry_robustness_tournament"
if __name__ == "__main__":
    sys.modules[MODULE] = sys.modules[__name__]

ALL_CANDIDATES = (
    "official_early_control",
    "bridge_early_control",
    "official_dualvenue_l2_residual",
    "official_high_precision_loss_veto",
    "official_capacity_aware_admission",
    "official_temporal_consensus",
    "official_dualvenue_l2_veto",
)
LEARNED_ADMISSION = (
    "official_high_precision_loss_veto",
    "official_capacity_aware_admission",
)
SPOT_L2_FEATURES = (
    "spot_l2_microprice_to_midpoint_bps",
    "spot_l2_spread_bps",
    "spot_l2_imbalance_5",
    "spot_l2_imbalance_10",
    "spot_l2_imbalance_20",
    "spot_l2_bid_depth_slope_20",
    "spot_l2_ask_depth_slope_20",
    "spot_l2_bid_quote_replenishment_1s_log",
    "spot_l2_ask_quote_replenishment_1s_log",
    "spot_l2_bid_quote_churn_1s_log",
    "spot_l2_ask_quote_churn_1s_log",
    "spot_l2_midpoint_change_5s_bps",
    "spot_l2_imbalance_20_change_5s",
    "spot_l2_midpoint_change_15s_bps",
    "spot_l2_imbalance_20_change_15s",
    "spot_l2_midpoint_change_30s_bps",
    "spot_l2_imbalance_20_change_30s",
    "has_spot_l2",
)
L2_SIGNAL_FEATURES = tuple(name for name in SPOT_L2_FEATURES if name != "has_spot_l2") + (
    *KRAKEN_L2_FEATURES,
)


@dataclass(frozen=True)
class L2ResidualModel:
    features: tuple[str, ...]
    estimator: HistGradientBoostingRegressor
    maximum_absolute_correction: float


L2ResidualModel.__module__ = MODULE


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


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_config(path: Path) -> tuple[Path, dict[str, Any]]:
    root = path.resolve().parents[1]
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if tuple(row["name"] for row in raw["candidates"]) != ALL_CANDIDATES:
        raise RuntimeError("frozen early-entry robustness roster changed")
    return root, raw


def _paths(root: Path, raw: dict[str, Any]) -> dict[str, Path]:
    return {name: _resolve(root, value) for name, value in raw["paths"].items()}


def _bands(raw: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (f"{int(start)}_{int(end)}", int(start), int(end))
        for start, end in raw["entry"]["competition_bands"]
    )


def _frame_span(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "first_window": frame["window_start"].min(),
        "last_window": frame["window_start"].max(),
    }


def _config_namespace(raw: dict[str, Any]) -> Any:
    return type("Config", (), {"raw": raw})()


def _source_manifest(root: Path, raw: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(root, raw)
    required = (
        "panel",
        "panel_manifest",
        "bridge_panel",
        "bridge_manifest",
        "prior_kraken_l2_manifest",
        "directional_artifact",
        "directional_metrics",
        "official_oof",
        "development_predictions",
        "bridge_oof",
        "prior_admission_artifact",
        "prior_admission_metrics",
    )
    for name in required:
        if not paths[name].is_file():
            raise RuntimeError(f"required immutable input is missing: {name}")
    if not paths["kraken_l2_raw_root"].is_dir():
        raise RuntimeError("existing Kraken L2 archive is unavailable")
    panel_manifest = json.loads(paths["panel_manifest"].read_text())
    bridge_manifest = json.loads(paths["bridge_manifest"].read_text())
    legacy_lock_key = "authentic_only_filter"
    panel_manifest.pop(legacy_lock_key, None)
    bridge_manifest.pop(legacy_lock_key, None)
    directional_metrics = json.loads(paths["directional_metrics"].read_text())
    prior_metrics = json.loads(paths["prior_admission_metrics"].read_text())
    identities = {name: file_sha256(paths[name]) for name in required}
    if identities["panel"] != panel_manifest["sha256"]:
        raise RuntimeError("full-history panel identity changed")
    if identities["bridge_panel"] != bridge_manifest["sha256"]:
        raise RuntimeError("settlement bridge identity changed")
    if identities["directional_artifact"] != directional_metrics["artifact_sha256"]:
        raise RuntimeError("directional control artifact identity changed")
    if identities["prior_admission_artifact"] != prior_metrics["artifact_sha256"]:
        raise RuntimeError("prior admission artifact identity changed")
    return {
        "identities": identities,
        "panel": panel_manifest,
        "bridge": bridge_manifest,
        "directional_run": directional_metrics["run_id"],
        "prior_admission_run": prior_metrics["run_id"],
        "existing_kraken_l2_archive": str(paths["kraken_l2_raw_root"]),
        "existing_champion_collection_unchanged": True,
        "database_mutations": False,
        "new_sources": False,
        "new_tables": False,
        "new_schemas": False,
        "new_ingesters": False,
    }


def _tail_manifest(paths: dict[str, Path], raw: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    start = datetime.fromisoformat(raw["windows"]["kraken_l2_tail_start"])
    end = datetime.fromisoformat(raw["windows"]["kraken_l2_tail_end"])
    try:
        return build_kraken_l2_features(
            raw_root=paths["kraken_l2_raw_root"],
            cache=paths["kraken_l2_tail_cache"],
            start=start,
            end=end,
        )
    except RuntimeError as error:
        if str(error) != "Kraken L2 archive contains no files in the configured interval":
            raise
        manifest = {
            "schema_version": "btc-kraken-l2-update-flow-v1",
            "range_start": start.isoformat(),
            "range_end_exclusive": end.isoformat(),
            "days_present": 0,
            "partitions": [],
            "read_only_source": True,
            "database_mutations": False,
            "status": "no_usable_parquet_partitions",
            "unavailable_markers": len(
                list(paths["kraken_l2_raw_root"].glob("????-??-??/??/*.unavailable.json"))
            ),
        }
        destination = paths["kraken_l2_tail_cache"] / "kraken-l2-manifest.json"
        _write_json(destination, manifest)
        return destination, manifest


def _panel_columns(schema: pl.Schema, model_features: tuple[str, ...]) -> tuple[str, ...]:
    requested = (
        *KEY_COLUMNS,
        "label_up",
        "btc_close",
        "fee_rate",
        "pm_up_book_age_seconds",
        "pm_down_book_age_seconds",
        *(f"up_ask_vwap_{quantity}" for quantity in VWAP_QUANTITIES),
        *(f"down_ask_vwap_{quantity}" for quantity in VWAP_QUANTITIES),
        "pm_vwap5_overround",
        "pm_vwap50_overround",
        "pm_vwap200_overround",
        "pm_up_depth_log",
        "pm_down_depth_log",
        "pm_depth_imbalance",
        *SPOT_L2_FEATURES,
        *KRAKEN_L2_FEATURES,
        "has_kraken_l2",
        *model_features,
    )
    return tuple(name for name in dict.fromkeys(requested) if name in schema)


def _load_panel(
    paths: dict[str, Path],
    start: datetime,
    end: datetime,
    model_features: tuple[str, ...],
    tail: dict[str, Any] | None = None,
) -> pl.DataFrame:
    scan = pl.scan_parquet(paths["panel"])
    columns = _panel_columns(scan.collect_schema(), model_features)
    panel = (
        scan.select(columns)
        .filter((pl.col("window_start") >= start) & (pl.col("window_start") < end))
        .collect()
    )
    if tail is None:
        return panel
    tail_start = datetime.fromisoformat(tail["range_start"])
    before = panel.filter(pl.col("observed_at") < tail_start)
    after = panel.filter(pl.col("observed_at") >= tail_start)
    if after.is_empty():
        return panel
    removable = [name for name in (*KRAKEN_L2_FEATURES, "has_kraken_l2") if name in after.columns]
    refreshed = attach_kraken_l2(after.drop(removable), tail)
    if before.is_empty():
        return refreshed
    return pl.concat((before, refreshed), how="diagonal_relaxed", rechunk=True).sort(
        ["window_start", "observed_at"]
    )


def _extra_l2(frame: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    names = [
        name
        for name in (*SPOT_L2_FEATURES, *KRAKEN_L2_FEATURES, "has_kraken_l2")
        if name in panel.columns and name not in frame.columns
    ]
    return (
        frame.join(
            panel.select(*KEY_COLUMNS, *names), on=list(KEY_COLUMNS), how="left", validate="m:1"
        )
        if names
        else frame
    )


def _admission_context(
    predictions: pl.DataFrame,
    counterpart: pl.DataFrame,
    panel: pl.DataFrame,
    raw: dict[str, Any],
) -> pl.DataFrame:
    return _extra_l2(_admission_frame(predictions, counterpart, panel, raw), panel)


def _matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return np.column_stack(
        [frame[name].cast(pl.Float64).fill_nan(None).fill_null(0.0).to_numpy() for name in features]
    )


def _residual_features(frame: pl.DataFrame) -> tuple[str, ...]:
    requested = (
        "official_probability",
        "seconds_elapsed",
        *SPOT_L2_FEATURES,
        *KRAKEN_L2_FEATURES,
        "has_kraken_l2",
    )
    return tuple(
        name
        for name in dict.fromkeys(requested)
        if name in frame.columns and frame[name].drop_nulls().n_unique() > 1
    )


def _fit_residual(
    official_oof: pl.DataFrame,
    panel: pl.DataFrame,
    paths: dict[str, Path],
    raw: dict[str, Any],
) -> L2ResidualModel:
    bridge = (
        pl.scan_parquet(paths["bridge_panel"])
        .select(*KEY_COLUMNS, "bridge_probability_target", "label_weight")
        .collect()
    )
    context = panel.select(
        *KEY_COLUMNS,
        *[
            name
            for name in (*SPOT_L2_FEATURES, *KRAKEN_L2_FEATURES, "has_kraken_l2")
            if name in panel.columns
        ],
    )
    frame = (
        official_oof.filter(pl.col("seconds_elapsed").is_between(60, 89, closed="both"))
        .rename({"probability": "official_probability"})
        .join(bridge, on=list(KEY_COLUMNS), how="inner", validate="1:1")
        .join(context, on=list(KEY_COLUMNS), how="left", validate="1:1")
        .filter(
            pl.any_horizontal(
                pl.col(name).is_not_null() & pl.col(name).is_finite()
                for name in L2_SIGNAL_FEATURES
                if name in context.columns
            )
        )
    )
    features = _residual_features(frame)
    target = (
        frame["bridge_probability_target"].to_numpy() - frame["official_probability"].to_numpy()
    )
    weights = _market_weights(frame) * frame["label_weight"].fill_null(0.0).to_numpy()
    spec = raw["residual"]
    estimator = HistGradientBoostingRegressor(
        learning_rate=float(spec["learning_rate"]),
        max_iter=int(spec["max_iter"]),
        max_leaf_nodes=int(spec["max_leaf_nodes"]),
        min_samples_leaf=int(spec["min_samples_leaf"]),
        l2_regularization=float(spec["l2_regularization"]),
        early_stopping=False,
        random_state=int(raw["training"]["random_seed"]),
    ).fit(_matrix(frame, features), target, sample_weight=weights)
    return L2ResidualModel(features, estimator, float(spec["maximum_absolute_correction"]))


def _score_residual(
    model: L2ResidualModel, predictions: pl.DataFrame, panel: pl.DataFrame
) -> pl.DataFrame:
    context_names = [
        name
        for name in (*SPOT_L2_FEATURES, *KRAKEN_L2_FEATURES, "has_kraken_l2")
        if name in panel.columns
    ]
    frame = predictions.rename({"probability": "official_probability"}).join(
        panel.select(*KEY_COLUMNS, *context_names),
        on=list(KEY_COLUMNS),
        how="left",
        validate="1:1",
    )
    has_l2 = pl.any_horizontal(
        pl.col(name).is_not_null() & pl.col(name).is_finite()
        for name in L2_SIGNAL_FEATURES
        if name in frame.columns
    )
    predicted = np.clip(
        model.estimator.predict(_matrix(frame, model.features)),
        -model.maximum_absolute_correction,
        model.maximum_absolute_correction,
    )
    frame = frame.with_columns(
        has_l2.alias("has_any_l2"),
        pl.Series("l2_probability_correction", predicted),
    ).with_columns(
        pl.when(pl.col("has_any_l2"))
        .then(
            (pl.col("official_probability") + pl.col("l2_probability_correction")).clip(
                1e-6, 1 - 1e-6
            )
        )
        .otherwise(pl.col("official_probability"))
        .alias("probability")
    )
    return frame.drop(context_names)


def _temporal_models(
    panel: pl.DataFrame,
    features: tuple[str, ...],
    raw: dict[str, Any],
    run_dir: Path,
) -> list[Any]:
    models = []
    source_start = datetime.fromisoformat(raw["windows"]["source_start"])
    for index, value in enumerate(raw["windows"]["temporal_fit_ends"]):
        end = datetime.fromisoformat(value)
        checkpoint = run_dir / "checkpoints" / f"temporal-{end.date().isoformat()}.joblib"
        if checkpoint.is_file():
            model = joblib.load(checkpoint)
        else:
            fit = panel.filter(
                (pl.col("window_start") >= source_start) & (pl.col("window_start") < end)
            )
            model = _fit_distilled(
                fit,
                features,
                raw,
                int(raw["training"]["random_seed"]) + 100 + index,
                target="label_up",
            )
            _write_joblib(checkpoint, model)
        models.append(model)
    return models


def _attach_temporal(
    predictions: pl.DataFrame, panel: pl.DataFrame, models: list[Any]
) -> pl.DataFrame:
    probabilities = [predictions["probability"].to_numpy()]
    for index, model in enumerate(models):
        scored = _score_model(model, panel, f"temporal_{index}", "context")
        aligned = predictions.select(*KEY_COLUMNS).join(
            scored.select(*KEY_COLUMNS, "probability"),
            on=list(KEY_COLUMNS),
            how="left",
            validate="1:1",
        )
        if aligned["probability"].null_count():
            raise RuntimeError("temporal prediction alignment failed")
        probabilities.append(aligned["probability"].to_numpy())
    values = np.column_stack(probabilities)
    direction = values >= 0.5
    return predictions.with_columns(
        pl.Series("temporal_probability_std", values.std(axis=1)),
        pl.Series("temporal_probability_range", values.max(axis=1) - values.min(axis=1)),
        pl.Series("temporal_direction_agreement", (direction == direction[:, [0]]).mean(axis=1)),
    )


def _attach_l2_confirmation(frame: pl.DataFrame) -> pl.DataFrame:
    names = [
        name
        for name in (
            "spot_l2_imbalance_20",
            "kraken_l2_update_imbalance_30s",
            "kraken_l2_quantity_imbalance_30s",
            "kraken_l2_cancel_imbalance_30s",
        )
        if name in frame.columns
    ]
    raw_confirmation = pl.mean_horizontal(*(pl.col(name) for name in names))
    return frame.with_columns(
        pl.any_horizontal(pl.col(name).is_not_null() for name in names).alias("has_any_l2"),
        (pl.when(pl.col("side") == "up").then(1.0).otherwise(-1.0) * raw_confirmation).alias(
            "l2_directional_confirmation"
        ),
    )


def _admission_features(frame: pl.DataFrame, capacity: bool) -> tuple[str, ...]:
    common = (
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
        *SPOT_L2_FEATURES,
        *KRAKEN_L2_FEATURES,
        "has_kraken_l2",
    )
    curve = (
        *(f"selected_vwap_{quantity}" for quantity in VWAP_QUANTITIES),
        "selected_slippage_5_25",
        "selected_slippage_5_100",
        "selected_slippage_5_200",
        "vwap_curve_available",
        "pm_vwap50_overround",
        "pm_vwap200_overround",
        "pm_depth_imbalance",
    )
    return tuple(
        name
        for name in dict.fromkeys((*common, *(curve if capacity else ())))
        if name in frame.columns and frame[name].drop_nulls().n_unique() > 1
    )


def _base_policies(raw: dict[str, Any]) -> Iterable[dict[str, float]]:
    for edge, confidence, cost in itertools.product(
        raw["execution"]["minimum_edges"],
        raw["execution"]["minimum_confidences"],
        raw["execution"]["maximum_share_costs"],
    ):
        yield {
            "minimum_edge": float(edge),
            "minimum_confidence": float(confidence),
            "maximum_share_cost": float(cost),
            "abstain": False,
        }


def _programmatic_filter(frame: pl.DataFrame, policy: dict[str, Any]) -> pl.DataFrame:
    output = frame.filter(
        (pl.col("share_cost") <= policy["maximum_share_cost"])
        & (pl.col("selected_probability") >= policy["minimum_confidence"])
        & (pl.col("expected_edge") >= policy["minimum_edge"])
    )
    if "maximum_probability_std" in policy:
        output = output.filter(
            (pl.col("temporal_probability_std") <= policy["maximum_probability_std"])
            & (pl.col("temporal_direction_agreement") >= policy["minimum_direction_agreement"])
        )
    if "minimum_l2_confirmation" in policy:
        output = output.filter(
            pl.col("has_any_l2")
            & (pl.col("l2_directional_confirmation") >= policy["minimum_l2_confirmation"])
        )
    return output


def _selection_score(trades: pl.DataFrame, development: pl.DataFrame, raw: dict[str, Any]) -> float:
    if trades.is_empty():
        return -math.inf
    shrinkage = float(raw["execution"]["development_shrinkage_trades"])
    days = development.select(pl.col("window_start").dt.date().alias("day")).unique()["day"]
    daily = {
        row["day"]: (row["stress_pnl"], row["trades"])
        for row in trades.with_columns(pl.col("window_start").dt.date().alias("day"))
        .group_by("day")
        .agg(
            pl.col("stress_net_pnl").sum().alias("stress_pnl"),
            pl.len().alias("trades"),
        )
        .to_dicts()
    }
    smoothed = np.asarray(
        [daily.get(day, (0.0, 0))[0] / (daily.get(day, (0.0, 0))[1] + shrinkage) for day in days],
        dtype=float,
    )
    aggregate = float(trades["stress_net_pnl"].sum()) / (trades.height + shrinkage)
    metrics = economic_metrics(trades, development["market_id"].n_unique())
    recovery = metrics["loss_recovery_wins"] or 0.0
    recovery_penalty = float(raw["execution"]["recovery_penalty"]) * max(
        0.0, recovery - float(raw["execution"]["recovery_soft_target"])
    )
    stability = float(raw["execution"]["development_stability_penalty"])
    return aggregate + float(smoothed.mean()) - stability * float(smoothed.std()) - recovery_penalty


def _best_base_policy(
    part: pl.DataFrame, raw: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _config_namespace(raw)
    best: tuple[float, dict[str, Any], pl.DataFrame] | None = None
    for policy in _base_policies(raw):
        trades = _select_trades(_programmatic_filter(part, policy), _no_veto(), config)
        score = _selection_score(trades, part, raw)
        if best is None or score > best[0]:
            best = (score, policy, trades)
    if best is None:
        raise RuntimeError("no programmatic policy evaluated")
    return best[1], {
        "selection_score": best[0],
        "development": economic_metrics(best[2], part["market_id"].n_unique()),
    }


def _select_programmatic(
    frame: pl.DataFrame, raw: dict[str, Any], gate: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _config_namespace(raw)
    policies: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for band, start, end in _bands(raw):
        part = frame.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        base, base_evidence = _best_base_policy(part, raw)
        variants: list[dict[str, Any]] = [base]
        if gate == "temporal":
            variants = [
                {
                    **base,
                    "maximum_probability_std": float(std),
                    "minimum_direction_agreement": float(agreement),
                }
                for std, agreement in itertools.product(
                    raw["temporal"]["maximum_probability_std"],
                    raw["temporal"]["minimum_direction_agreement"],
                )
            ]
        elif gate == "l2":
            variants = [
                {**base, "minimum_l2_confirmation": float(threshold)}
                for threshold in raw["l2_veto"]["minimum_confirmation"]
            ]
        best: tuple[float, dict[str, Any], pl.DataFrame] | None = None
        for policy in variants:
            trades = _select_trades(_programmatic_filter(part, policy), _no_veto(), config)
            score = _selection_score(trades, part, raw)
            if best is None or score > best[0]:
                best = (score, policy, trades)
        if best is None:
            raise RuntimeError(f"no {gate or 'base'} policy evaluated for {band}")
        policies[band] = best[1]
        evidence[band] = {
            "base_selection": base_evidence,
            "selection_score": best[0],
            "development": economic_metrics(best[2], part["market_id"].n_unique()),
        }
    return policies, evidence


def _select_admission(
    frame: pl.DataFrame, raw: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _config_namespace(raw)
    policies: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    spec = raw["admission"]
    for band, start, end in _bands(raw):
        part = frame.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        best: tuple[float, dict[str, Any], pl.DataFrame] | None = None
        for probability, edge, loss, cost in itertools.product(
            spec["probability_thresholds"],
            spec["stress_edge_thresholds"],
            spec["maximum_loss_thresholds"],
            spec["maximum_share_costs"],
        ):
            policy = {
                "minimum_admission_probability": float(probability),
                "minimum_predicted_stress_edge": float(edge),
                "maximum_predicted_loss": float(loss),
                "maximum_share_cost": float(cost),
            }
            trades = _select_trades(_policy_filter(part, policy), _no_veto(), config)
            score = _selection_score(trades, part, raw)
            if best is None or score > best[0]:
                best = (score, policy, trades)
        if best is None:
            raise RuntimeError(f"no admission policy evaluated for {band}")
        policies[band] = best[1]
        evidence[band] = {
            "selection_score": best[0],
            "development": economic_metrics(best[2], part["market_id"].n_unique()),
        }
    return policies, evidence


def _eligible(
    frame: pl.DataFrame,
    policy: dict[str, Any],
    kind: str,
) -> pl.DataFrame:
    return (
        _policy_filter(frame, policy)
        if kind.startswith("learned_")
        else _programmatic_filter(frame, policy)
    )


def _side_metrics(trades: pl.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for side in ("up", "down"):
        part = trades.filter(pl.col("side") == side)
        output[side] = {
            "trades": part.height,
            "selection_share": part.height / trades.height if trades.height else None,
            "wins": int(part["won"].sum()) if part.height else 0,
            "losses": int((~part["won"]).sum()) if part.height else 0,
            "win_rate": float(part["won"].mean()) if part.height else None,
            "net_pnl": float(part["net_pnl"].sum()) if part.height else 0.0,
            "stress_net_pnl": float(part["stress_net_pnl"].sum()) if part.height else 0.0,
            "average_share_cost": float(part["share_cost"].mean()) if part.height else None,
        }
    return output


def _evaluate(
    frame: pl.DataFrame,
    policies: dict[str, Any],
    kind: str,
    raw: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    total = frame["market_id"].n_unique()
    config = _config_namespace(raw)
    metrics: dict[str, Any] = {}
    ledgers: dict[str, pl.DataFrame] = {}
    eligible_parts = []
    for band, start, end in _bands(raw):
        part = frame.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        eligible = _eligible(part, policies[band], kind)
        trades = _select_trades(eligible, _no_veto(), config)
        summary = economic_metrics(trades, total)
        summary["by_side"] = _side_metrics(trades)
        metrics[band], ledgers[band] = summary, trades
        eligible_parts.append(eligible)
    combined = _select_trades(
        pl.concat(eligible_parts, how="diagonal_relaxed", rechunk=True),
        _no_veto(),
        config,
    )
    combined_summary = economic_metrics(combined, total)
    combined_summary["by_side"] = _side_metrics(combined)
    metrics["combined_60_180"] = combined_summary
    ledgers["combined_60_180"] = combined
    return metrics, ledgers


def _candidate_frames(
    official: pl.DataFrame,
    bridge: pl.DataFrame,
    panel: pl.DataFrame,
    raw: dict[str, Any],
    residual: L2ResidualModel,
    temporal_models: list[Any],
    admissions: dict[str, AdmissionModel],
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    official_frame = _admission_context(official, bridge, panel, raw)
    bridge_frame = _admission_context(bridge, official, panel, raw)
    residual_predictions = _score_residual(residual, official, panel)
    residual_frame = _admission_context(residual_predictions, bridge, panel, raw)
    temporal_predictions = _attach_temporal(official, panel, temporal_models)
    temporal_frame = _admission_context(temporal_predictions, bridge, panel, raw)
    l2_frame = _attach_l2_confirmation(official_frame)
    frames = {
        "official_early_control": official_frame,
        "bridge_early_control": bridge_frame,
        "official_dualvenue_l2_residual": residual_frame,
        "official_high_precision_loss_veto": _attach_model(
            official_frame, admissions["official_high_precision_loss_veto"]
        ),
        "official_capacity_aware_admission": _attach_model(
            official_frame, admissions["official_capacity_aware_admission"]
        ),
        "official_temporal_consensus": temporal_frame,
        "official_dualvenue_l2_veto": l2_frame,
    }
    predictions = {
        "official_early_control": official,
        "bridge_early_control": bridge,
        "official_dualvenue_l2_residual": residual_predictions,
        "official_high_precision_loss_veto": official,
        "official_capacity_aware_admission": official,
        "official_temporal_consensus": official,
        "official_dualvenue_l2_veto": official,
    }
    return frames, predictions


def _report(metrics: dict[str, Any]) -> str:
    def fmt(value: Any, digits: int = 3) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Early-Entry Robustness Tournament",
        "",
        f"Run: `{metrics['run_id']}`",
        f"Qualification: **{metrics['qualification_status']}**",
        "",
        "## Primary 60–89-second results",
        "",
        "| Candidate | PnL | Stress PnL | PF | Expectancy | Coverage | W/L | Win rate | Recovery | Avg entry | Avg cost | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in ALL_CANDIDATES:
        row = metrics["sealed_economic"][candidate]["60_89"]
        predictive = metrics["sealed_predictive"][candidate]
        lines.append(
            f"| {candidate} | {row['net_pnl']:.2f} | {row['stress_net_pnl']:.2f} | {fmt(row['profit_factor'])} | {fmt(row['expectancy_per_trade'])} | {row['market_coverage']:.2%} | {row['winning_trades']}/{row['losing_trades']} | {fmt(row['win_rate'], 2)} | {fmt(row['loss_recovery_wins'])} | {fmt(row['average_entry_second'], 1)} | {fmt(row['average_share_cost'])} | {fmt(predictive['brier_score'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## PnL by independent entry bucket",
            "",
            "| Candidate | 60–89 | 90–119 | 120–149 | 150–180 | Combined 60–180 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in ALL_CANDIDATES:
        rows = metrics["sealed_economic"][candidate]
        lines.append(
            f"| {candidate} | {rows['60_89']['net_pnl']:.2f} | {rows['90_119']['net_pnl']:.2f} | {rows['120_149']['net_pnl']:.2f} | {rows['150_180']['net_pnl']:.2f} | {rows['combined_60_180']['net_pnl']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## UP/DOWN behavior in the primary bucket",
            "",
            "| Candidate | UP trades | DOWN trades | UP PnL | DOWN PnL | UP stress | DOWN stress |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in ALL_CANDIDATES:
        side = metrics["sealed_economic"][candidate]["60_89"]["by_side"]
        lines.append(
            f"| {candidate} | {side['up']['trades']} ({fmt(side['up']['selection_share'], 1)}) | {side['down']['trades']} ({fmt(side['down']['selection_share'], 1)}) | {side['up']['net_pnl']:.2f} | {side['down']['net_pnl']:.2f} | {side['up']['stress_net_pnl']:.2f} | {side['down']['stress_net_pnl']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Primary-bucket VWAP capacity",
            "",
            "| Candidate | Quantity | Trades | PnL | Stress PnL | PF | Expectancy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in ALL_CANDIDATES:
        for quantity in VWAP_QUANTITIES:
            row = metrics["sealed_capacity"][candidate][str(quantity)]
            lines.append(
                f"| {candidate} | {quantity} | {row['trades']} | {row['net_pnl']:.2f} | {row['stress_net_pnl']:.2f} | {fmt(row['profit_factor'])} | {fmt(row['expectancy_per_trade'])} |"
            )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            "- Frozen controls and the existing champion collection were not modified.",
            "- All learned components used chronological OOF inputs ending before development and holdout markets.",
            "- Policies were frozen before the sealed panel and Kraken L2 tail were loaded.",
            "- Settlement sources remained distinct; TWAP and future settlement values were not inference features.",
            "- No database write, table, schema, ingester, source, deployment, or image build occurred.",
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

    directional = joblib.load(paths["directional_artifact"])
    prior_admission = joblib.load(paths["prior_admission_artifact"])
    official_model = directional["models"]["extended_specialist_official"]
    bridge_model = directional["models"]["bridge_aware_specialist"]
    model_features = tuple(dict.fromkeys((*official_model.features, *bridge_model.features)))
    source_start = datetime.fromisoformat(raw["windows"]["source_start"])
    fit_end = datetime.fromisoformat(raw["windows"]["fit_end"])
    development_start = datetime.fromisoformat(raw["windows"]["development_start"])
    development_end = datetime.fromisoformat(raw["windows"]["development_end"])
    sealed_start = datetime.fromisoformat(raw["windows"]["sealed_start"])
    sealed_end = datetime.fromisoformat(raw["windows"]["sealed_end"])
    split = {
        "source_start": source_start,
        "fit_end_exclusive": fit_end,
        "development_start": development_start,
        "development_end_exclusive": development_end,
        "sealed_start": sealed_start,
        "sealed_end_exclusive": sealed_end,
        "prospective_start": raw["windows"]["prospective_start"],
        "prospective_end_exclusive": raw["windows"]["prospective_end"],
        "market_disjoint": True,
        "sealed_loaded_after_policy_freeze": True,
    }
    _write_json(run_dir / "split-manifest.json", split)

    fit_panel = _load_panel(paths, source_start, fit_end, model_features)
    development_panel = _load_panel(paths, development_start, development_end, model_features)
    official_oof = pl.read_parquet(paths["official_oof"])
    bridge_oof = pl.read_parquet(paths["bridge_oof"])
    development_predictions = pl.read_parquet(paths["development_predictions"])
    official_development = development_predictions.filter(
        pl.col("candidate") == "extended_specialist_official"
    )
    bridge_development = development_predictions.filter(
        pl.col("candidate") == "bridge_aware_specialist"
    )
    split["actual"] = {
        "fit_panel": _frame_span(fit_panel),
        "official_oof": _frame_span(official_oof),
        "bridge_oof": _frame_span(bridge_oof),
        "development_panel": _frame_span(development_panel),
    }
    _write_json(run_dir / "split-manifest.json", split)

    residual_checkpoint = run_dir / "checkpoints" / "l2-residual.joblib"
    if residual_checkpoint.is_file():
        residual = joblib.load(residual_checkpoint)
    else:
        residual = _fit_residual(official_oof, fit_panel, paths, raw)
        _write_joblib(residual_checkpoint, residual)

    admissions: dict[str, AdmissionModel] = {}
    official_fit_frame = _admission_context(official_oof, bridge_oof, fit_panel, raw)
    for index, candidate in enumerate(LEARNED_ADMISSION):
        checkpoint = run_dir / "checkpoints" / f"admission-{candidate}.joblib"
        if checkpoint.is_file():
            model = joblib.load(checkpoint)
        else:
            capacity = candidate == "official_capacity_aware_admission"
            features = _admission_features(official_fit_frame, capacity)
            model = _fit_admission(
                official_fit_frame,
                features,
                raw,
                int(raw["training"]["random_seed"]) + 1000 + index,
                "extended_specialist_official",
                "capacity" if capacity else "loss",
            )
            _write_joblib(checkpoint, model)
        admissions[candidate] = model

    temporal = _temporal_models(fit_panel, official_model.features, raw, run_dir)
    development_frames, development_candidate_predictions = _candidate_frames(
        official_development,
        bridge_development,
        development_panel,
        raw,
        residual,
        temporal,
        admissions,
    )
    candidate_contract = {
        row["name"]: {"kind": row["kind"], "base": row["base"]} for row in raw["candidates"]
    }
    selection_path = run_dir / "selection-freeze.json"
    if resume_run and selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        policies = selection["policies"]
        policy_evidence = selection["development_evidence"]
        selection_time = selection["frozen_at"]
    else:
        policies = {
            "official_early_control": prior_admission["policies"]["official_specialist_control"],
            "bridge_early_control": prior_admission["policies"]["bridge_aware_control"],
        }
        policy_evidence: dict[str, Any] = {}
        (
            policies["official_dualvenue_l2_residual"],
            policy_evidence["official_dualvenue_l2_residual"],
        ) = _select_programmatic(development_frames["official_dualvenue_l2_residual"], raw)
        for candidate in LEARNED_ADMISSION:
            policies[candidate], policy_evidence[candidate] = _select_admission(
                development_frames[candidate], raw
            )
        (
            policies["official_temporal_consensus"],
            policy_evidence["official_temporal_consensus"],
        ) = _select_programmatic(
            development_frames["official_temporal_consensus"], raw, gate="temporal"
        )
        (
            policies["official_dualvenue_l2_veto"],
            policy_evidence["official_dualvenue_l2_veto"],
        ) = _select_programmatic(development_frames["official_dualvenue_l2_veto"], raw, gate="l2")
        selection_time = datetime.now(UTC).isoformat()
        _write_json(
            selection_path,
            {
                "frozen_at": selection_time,
                "policies": policies,
                "development_evidence": policy_evidence,
                "sealed_loaded_after_freeze": True,
            },
        )
    development_economic: dict[str, Any] = {}
    development_predictive: dict[str, Any] = {}
    for candidate in ALL_CANDIDATES:
        kind = candidate_contract[candidate]["kind"]
        development_economic[candidate], _ = _evaluate(
            development_frames[candidate], policies[candidate], kind, raw
        )
        development_predictive[candidate] = predictive_metrics(
            development_candidate_predictions[candidate]
        )
    tail_path, tail = _tail_manifest(paths, raw)
    source["kraken_l2_tail"] = {
        "manifest_path": str(tail_path),
        "manifest_sha256": file_sha256(tail_path),
        "days_present": tail["days_present"],
        "range_start": tail["range_start"],
        "range_end_exclusive": tail["range_end_exclusive"],
        "read_only_source": tail["read_only_source"],
    }
    _write_json(run_dir / "source-manifest.json", source)
    sealed_panel = _load_panel(paths, sealed_start, sealed_end, model_features, tail)
    if set(fit_panel["market_id"].unique()) & set(sealed_panel["market_id"].unique()):
        raise RuntimeError("fit and sealed markets overlap")
    if set(development_panel["market_id"].unique()) & set(sealed_panel["market_id"].unique()):
        raise RuntimeError("development and sealed markets overlap")
    split["actual"]["sealed_panel"] = _frame_span(sealed_panel)
    _write_json(run_dir / "split-manifest.json", split)
    official_sealed = _score_model(
        official_model, sealed_panel, "extended_specialist_official", "sealed"
    )
    bridge_sealed = _score_model(bridge_model, sealed_panel, "bridge_aware_specialist", "sealed")
    sealed_frames, sealed_predictions = _candidate_frames(
        official_sealed,
        bridge_sealed,
        sealed_panel,
        raw,
        residual,
        temporal,
        admissions,
    )

    sealed_economic: dict[str, Any] = {}
    sealed_predictive: dict[str, Any] = {}
    sealed_capacity: dict[str, Any] = {}
    ledgers = []
    for candidate in ALL_CANDIDATES:
        kind = candidate_contract[candidate]["kind"]
        economics, candidate_ledgers = _evaluate(
            sealed_frames[candidate], policies[candidate], kind, raw
        )
        primary = candidate_ledgers["60_89"]
        sealed_economic[candidate] = economics
        sealed_predictive[candidate] = predictive_metrics(sealed_predictions[candidate])
        sealed_capacity[candidate] = _capacity_metrics(primary, raw)
        for bucket, trades in candidate_ledgers.items():
            if not trades.is_empty():
                ledgers.append(
                    trades.with_columns(
                        pl.lit(candidate).alias("strategy_candidate"),
                        pl.lit(bucket).alias("evaluation_bucket"),
                    )
                )
    pl.concat(ledgers, how="diagonal_relaxed", rechunk=True).write_parquet(
        run_dir / "ledgers" / "sealed-trades.parquet",
        compression="zstd",
        statistics=True,
    )
    _write_json(run_dir / "candidate-contract.json", candidate_contract)

    producing_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    artifact_path = run_dir / "tournament.joblib"
    _write_joblib(
        artifact_path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "producing_commit": producing_commit,
            "directional_artifact_sha256": source["identities"]["directional_artifact"],
            "residual_model": residual,
            "admission_models": admissions,
            "temporal_models": temporal,
            "policies": policies,
            "deployment_status": "not_deployed",
        },
    )
    loaded = joblib.load(artifact_path)
    if set(loaded["admission_models"]) != set(LEARNED_ADMISSION):
        raise RuntimeError("artifact round-trip failed")
    artifact_sha = file_sha256(artifact_path)
    (run_dir / "tournament.sha256").write_text(f"{artifact_sha}  tournament.joblib\n")
    profitable = [
        candidate
        for candidate in ALL_CANDIDATES
        if sealed_economic[candidate]["60_89"]["net_pnl"] > 0
        and sealed_economic[candidate]["60_89"]["stress_net_pnl"] > 0
    ]
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "producing_commit": producing_commit,
        "candidate_names": list(ALL_CANDIDATES),
        "candidate_contract": candidate_contract,
        "positive_primary_candidates": profitable,
        "qualification_status": "trained_evaluated_not_deployed",
        "sealed_economic": sealed_economic,
        "sealed_predictive": sealed_predictive,
        "sealed_capacity": sealed_capacity,
        "development_economic": development_economic,
        "development_predictive": development_predictive,
        "pre_cutover_oof_predictive": {
            "official": predictive_metrics(official_oof),
            "bridge": predictive_metrics(bridge_oof),
        },
        "source_manifest": source,
        "split_manifest": split,
        "selection_frozen_at": selection_time,
        "artifact_sha256": artifact_sha,
        "settlement_evaluation": {
            "fit_regime": "historical official RefPrice plus source-preserving bridge supervision",
            "development_regime": "official TWAP after 2026-08-14 cutover",
            "sealed_regime": "official TWAP",
            "sources_remain_distinct": True,
            "twap_inference_feature": False,
        },
        "l2_coverage": {
            "fit_kraken_rows": int(fit_panel["has_kraken_l2"].sum()),
            "development_kraken_rows": int(development_panel["has_kraken_l2"].sum()),
            "sealed_kraken_rows": int(sealed_panel["has_kraken_l2"].sum()),
            "fit_binance_rows": int(fit_panel["has_spot_l2"].sum()),
            "development_binance_rows": int(development_panel["has_spot_l2"].sum()),
            "sealed_binance_rows": int(sealed_panel["has_spot_l2"].sum()),
        },
        "prospective": {
            "range_start": raw["windows"]["prospective_start"],
            "range_end_exclusive": raw["windows"]["prospective_end"],
            "available_rows": 0,
            "status": "not_present_in_current_immutable_panel",
        },
        "integrity": {
            "passed": True,
            "frozen_controls_unchanged": True,
            "existing_champion_collection_unchanged": True,
            "market_disjoint": True,
            "selection_frozen_before_sealed_load": True,
            "best_later_target": False,
            "twap_inference_feature": False,
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
            "Executable VWAP evidence ends after August 25; August 26-31 contributes predictive metrics but no economic fills.",
            "Binance spot L2 ends after August 1; the sealed L2 evaluation therefore uses Kraken L2 while preserving Binance L2 missingness.",
            "The August holdout has been observed in prior research and is chronological rather than epistemically fresh.",
            "September 1 data is not present in the current immutable panel and is reported as unavailable rather than fabricated.",
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
