"""Frozen seven-candidate tournament with full-August holdout and VWAP-curve admission."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from . import hybrid_admission_tournament as hat
from . import middle_strategy_tournament as mst
from .continuous_edge_training import BOOK_RAW_FEATURES, VWAP_QUANTITIES
from .core_extract import file_sha256
from .kraken_l2_training_data import KRAKEN_L2_FEATURES
from .middle_strategy_data import build_middle_panel
from .multivenue_early_entry_data import (
    KEY_COLUMNS,
    TournamentDataConfig,
    extract_sources,
    load_data_config,
)
from .twap60_training_data import DataPaths

SCHEMA_VERSION = "btc-full-august-vwap-admission-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-full-august-vwap-admission-model-v1"
CANONICAL_MODULE = "btc_directional_model.vwap_curve_admission_tournament"
if __name__ == "__main__":
    sys.modules[CANONICAL_MODULE] = sys.modules[__name__]

ADMISSION_MODES = (
    "hybrid_vwap5",
    "hybrid_full_vwap_no_l2",
    "hybrid_full_vwap_dual_l2",
)
TARGET_ONLY_COLUMNS = (
    "official_label_up",
    "bridge_probability_target",
    "settlement_bridge_z",
    "bridge_uncertainty_bps",
    "target_margin_bps",
    "label_weight",
    "label_source",
)


@dataclass(frozen=True)
class ProbabilityBandCalibrator:
    estimators: dict[str, LogisticRegression | None]


@dataclass(frozen=True)
class VwapAdmissionModel:
    feature_names: tuple[str, ...]
    profitable_classifier: HistGradientBoostingClassifier
    stress_edge_regressor: HistGradientBoostingRegressor
    enter_now_regressor: HistGradientBoostingRegressor
    lower_bound_penalties: dict[str, float]
    global_lower_bound_penalty: float
    stress_slippage_per_share: float
    mode: str


ProbabilityBandCalibrator.__module__ = CANONICAL_MODULE
VwapAdmissionModel.__module__ = CANONICAL_MODULE


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _git_revision(package_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=package_root, text=True
    ).strip()


def _cache(config: TournamentDataConfig) -> Path:
    return config.package_root / config.raw["paths"]["cache"]


def _tail_config(config: TournamentDataConfig) -> TournamentDataConfig:
    tail_start = datetime.fromisoformat(config.raw["windows"]["tail_start"])
    relative = config.raw["paths"]["tail_cache"]
    tail_cache = config.package_root / relative
    raw = copy.deepcopy(config.raw)
    raw["windows"]["source_start"] = tail_start.isoformat()
    raw["windows"]["kraken_start"] = tail_start.isoformat()
    raw["windows"]["open_interest_start"] = tail_start.isoformat()
    raw["windows"]["binance_print_start"] = tail_start.isoformat()
    raw["paths"]["cache"] = relative
    raw["paths"]["middle_cache"] = relative
    raw["paths"]["hybrid_cache"] = relative
    standard = DataPaths(
        package_root=config.package_root,
        cache=tail_cache,
        core_features=tail_cache / "unused-core-features.parquet",
        core_current_sql=config.standard.core_current_sql,
        oracle_sql=config.standard.oracle_sql,
        label_sql=config.standard.label_sql,
        refprice_sql=config.standard.refprice_sql,
        candle_sql=config.standard.candle_sql,
        execution_sql=config.standard.execution_sql,
    )
    return replace(
        config,
        raw=raw,
        source_start=tail_start,
        kraken_start=tail_start,
        open_interest_start=tail_start,
        binance_print_start=tail_start,
        cache=tail_cache,
        standard=standard,
    )


def build_vwap_panel(
    config: TournamentDataConfig, *, force_tail: bool = False
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Reuse the immutable causal panel through August 26 and checkpoint only its tail."""

    cache = _cache(config)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "vwap-admission-panel.parquet"
    manifest_path = cache / "vwap-admission-panel-manifest.json"
    if destination.is_file() and manifest_path.is_file() and not force_tail:
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != file_sha256(destination):
            raise RuntimeError("VWAP admission panel changed after checkpoint")
        return pl.read_parquet(destination), manifest

    input_cache = config.package_root / config.raw["paths"]["input_hybrid_cache"]
    base_path = input_cache / "middle-panel.parquet"
    base_manifest_path = input_cache / "panel-manifest.json"
    if not base_path.is_file() or not base_manifest_path.is_file():
        raise RuntimeError("immutable prior tournament panel is unavailable")
    base_manifest = json.loads(base_manifest_path.read_text())
    if base_manifest["sha256"] != file_sha256(base_path):
        raise RuntimeError("immutable prior tournament panel identity changed")

    tail_config = _tail_config(config)
    extract_sources(tail_config, force=force_tail)
    tail, tail_manifest = build_middle_panel(tail_config, force=force_tail)
    tail = tail.with_columns(
        *(pl.lit(None, dtype=pl.Float64).alias(name) for name in KRAKEN_L2_FEATURES),
        pl.lit(False).alias("has_kraken_l2"),
    )

    tail_start = datetime.fromisoformat(config.raw["windows"]["tail_start"])
    base = pl.read_parquet(base_path).filter(pl.col("window_start") < tail_start)
    removable = [name for name in TARGET_ONLY_COLUMNS if name in base.columns]
    base = base.drop(removable)
    tail = tail.filter(pl.col("window_start") >= tail_start)
    panel = pl.concat((base, tail), how="diagonal_relaxed", rechunk=True).sort(
        list(KEY_COLUMNS)
    )
    duplicate_keys = panel.select(*KEY_COLUMNS).is_duplicated().sum()
    if duplicate_keys:
        raise RuntimeError(f"combined panel contains {duplicate_keys} duplicate decision rows")
    if any(name in panel.columns for name in TARGET_ONLY_COLUMNS):
        raise RuntimeError("settlement bridge supervision leaked into the tournament panel")
    fit = panel.filter(pl.col("window_start") < config.fit_end)
    heldout = panel.filter(
        pl.col("window_start").is_between(
            config.sealed_start, config.sealed_end, closed="left"
        )
    )
    if set(fit["market_id"].unique()) & set(heldout["market_id"].unique()):
        raise RuntimeError("predictive fit and full-August holdout markets overlap")

    panel.write_parquet(destination, compression="zstd", statistics=True)
    groups = dict(base_manifest["feature_groups"])
    groups["execution"] = [
        name
        for name in panel.columns
        if name.startswith(("up_ask_vwap_", "down_ask_vwap_", "pm_"))
    ]
    coverage = {}
    for name, feature_names in groups.items():
        available = [column for column in feature_names if column in panel.columns]
        if available:
            present = pl.any_horizontal(
                pl.col(column).is_not_null() for column in available
            )
            selected = panel.filter(present)
            coverage[name] = {
                "rows": selected.height,
                "markets": selected["market_id"].n_unique(),
            }
    curve = panel.filter(pl.col("up_ask_vwap_200").is_not_null())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rows": panel.height,
        "markets": panel["market_id"].n_unique(),
        "fit_markets": fit["market_id"].n_unique(),
        "full_august_markets": heldout["market_id"].n_unique(),
        "range_start": panel["window_start"].min(),
        "range_end": config.sealed_end,
        "feature_groups": groups,
        "coverage": coverage,
        "vwap_curve_coverage": {
            "quantities": list(VWAP_QUANTITIES),
            "rows": curve.height,
            "markets": curve["market_id"].n_unique(),
            "last_observed_at": curve["observed_at"].max(),
        },
        "immutable_base": {
            "path": str(base_path),
            "sha256": base_manifest["sha256"],
            "manifest_sha256": file_sha256(base_manifest_path),
            "used_before": tail_start,
        },
        "tail": {
            "cache": str(tail_config.cache),
            "source_manifest_sha256": file_sha256(tail_config.cache / "source-manifest.json"),
            "panel_sha256": tail_manifest["sha256"],
        },
        "supervision": "official resolved up/down outcome only",
        "bridge_supervision_columns_removed": list(TARGET_ONLY_COLUMNS),
        "optional_missingness_preserves_rows": True,
        "authentic_only_filter": False,
        "database_mutations": False,
        "new_tables": False,
        "new_schemas": False,
        "new_ingesters": False,
        "new_sources": False,
        "sha256": file_sha256(destination),
    }
    _write_json(manifest_path, manifest)
    return panel, manifest


def _time_band_expression(config: TournamentDataConfig) -> pl.Expr:
    expression: pl.Expr | None = None
    for name, start, end in mst._calibration_cells(config):
        condition = pl.col("seconds_elapsed").is_between(start, end, closed="both")
        expression = (
            pl.when(condition).then(pl.lit(name))
            if expression is None
            else expression.when(condition).then(pl.lit(name))
        )
    assert expression is not None
    return expression.otherwise(pl.lit("outside")).alias("time_band")


def _fit_probability_calibrator(frame: pl.DataFrame) -> LogisticRegression | None:
    if frame.height < 500 or frame["label_up"].n_unique() < 2:
        return None
    probability = np.clip(frame["probability"].to_numpy(), 1e-6, 1 - 1e-6)
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=500).fit(
        logit(probability).reshape(-1, 1), frame["label_up"].to_numpy()
    )


def _apply_probability_calibrator(
    frame: pl.DataFrame, model: ProbabilityBandCalibrator
) -> pl.DataFrame:
    pieces = []
    for band, estimator in model.estimators.items():
        part = frame.filter(pl.col("time_band") == band)
        raw = np.clip(part["probability"].to_numpy(), 1e-6, 1 - 1e-6)
        probability = (
            raw
            if estimator is None
            else estimator.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
        )
        pieces.append(
            part.with_columns(
                pl.col("probability").alias("raw_probability"),
                pl.Series("probability", probability),
            )
        )
    return pl.concat(pieces, how="vertical_relaxed").sort(
        ["candidate", "window_start", "market_id", "seconds_elapsed"]
    )


def calibrate_oof_by_time_band(
    predictions: pl.DataFrame, config: TournamentDataConfig
) -> tuple[pl.DataFrame, dict[str, ProbabilityBandCalibrator]]:
    tagged = predictions.with_columns(_time_band_expression(config))
    folds = [row["name"] for row in config.raw["folds"]]
    calibrated, final = [], {}
    for candidate in mst.ALL_NAMES:
        source = tagged.filter(pl.col("candidate") == candidate)
        pieces = []
        for index, fold in enumerate(folds):
            current = source.filter(pl.col("fold") == fold)
            prior = source.filter(pl.col("fold").is_in(folds[:index]))
            estimators = {
                band: _fit_probability_calibrator(
                    prior.filter(pl.col("time_band") == band)
                )
                for band, _, _ in mst._calibration_cells(config)
            }
            pieces.append(
                _apply_probability_calibrator(
                    current, ProbabilityBandCalibrator(estimators)
                )
            )
        calibrated.append(pl.concat(pieces, how="vertical_relaxed"))
        final[candidate] = ProbabilityBandCalibrator(
            {
                band: _fit_probability_calibrator(
                    source.filter(pl.col("time_band") == band)
                )
                for band, _, _ in mst._calibration_cells(config)
            }
        )
    return pl.concat(calibrated, how="vertical_relaxed"), final


def apply_band_calibrators(
    predictions: pl.DataFrame,
    calibrators: dict[str, ProbabilityBandCalibrator],
    config: TournamentDataConfig,
) -> pl.DataFrame:
    tagged = predictions.with_columns(_time_band_expression(config))
    return pl.concat(
        [
            _apply_probability_calibrator(
                tagged.filter(pl.col("candidate") == candidate), calibrators[candidate]
            )
            for candidate in mst.ALL_NAMES
        ],
        how="vertical_relaxed",
    )


def _prediction_context(predictions: pl.DataFrame) -> pl.DataFrame:
    return predictions.group_by(list(KEY_COLUMNS)).agg(
        pl.col("probability").std().fill_null(0.0).alias("candidate_probability_std"),
        (pl.col("probability").max() - pl.col("probability").min()).alias(
            "candidate_probability_range"
        ),
        pl.max_horizontal(
            (pl.col("probability") >= 0.5).mean(),
            (pl.col("probability") < 0.5).mean(),
        ).alias("candidate_direction_agreement"),
    )


def _curve_context_columns(panel: pl.DataFrame) -> tuple[str, ...]:
    requested = (
        *BOOK_RAW_FEATURES,
        "up_ask_depth",
        "down_ask_depth",
        "pm_vwap5_overround",
        "pm_vwap50_overround",
        "pm_vwap200_overround",
        "pm_up_slope_5_25",
        "pm_up_slope_25_100",
        "pm_up_slope_100_200",
        "pm_down_slope_5_25",
        "pm_down_slope_25_100",
        "pm_down_slope_100_200",
        "pm_up_depth_log",
        "pm_down_depth_log",
        "pm_depth_imbalance",
    )
    return tuple(name for name in requested if name in panel.columns)


def _attach_curve_features(frame: pl.DataFrame) -> pl.DataFrame:
    selected = []
    for quantity in VWAP_QUANTITIES:
        selected.extend(
            (
                pl.when(pl.col("side") == "up")
                .then(pl.col(f"up_ask_vwap_{quantity}"))
                .otherwise(pl.col(f"down_ask_vwap_{quantity}"))
                .alias(f"selected_vwap_{quantity}"),
                pl.when(pl.col("side") == "up")
                .then(pl.col(f"down_ask_vwap_{quantity}"))
                .otherwise(pl.col(f"up_ask_vwap_{quantity}"))
                .alias(f"opposite_vwap_{quantity}"),
                (
                    pl.col(f"up_ask_vwap_{quantity}")
                    + pl.col(f"down_ask_vwap_{quantity}")
                    - 1.0
                ).alias(f"vwap_overround_{quantity}"),
            )
        )
    output = frame.with_columns(*selected).with_columns(
        pl.when(pl.col("side") == "up")
        .then(pl.col("pm_up_depth_log"))
        .otherwise(pl.col("pm_down_depth_log"))
        .alias("selected_depth_log"),
        pl.when(pl.col("side") == "up")
        .then(pl.col("pm_down_depth_log"))
        .otherwise(pl.col("pm_up_depth_log"))
        .alias("opposite_depth_log"),
        pl.when(pl.col("side") == "up")
        .then(pl.col("pm_depth_imbalance"))
        .otherwise(-pl.col("pm_depth_imbalance"))
        .alias("selected_depth_imbalance"),
        pl.col("selected_vwap_200").is_not_null().cast(pl.Float64).alias(
            "vwap_curve_available"
        ),
        pl.col("pm_up_book_age_seconds").is_null().cast(pl.Float64).alias(
            "up_book_age_missing"
        ),
        pl.col("pm_down_book_age_seconds").is_null().cast(pl.Float64).alias(
            "down_book_age_missing"
        ),
    )
    return output.with_columns(
        ((pl.col("selected_vwap_25") - pl.col("selected_vwap_5")) / 20.0).alias(
            "selected_slope_5_25"
        ),
        ((pl.col("selected_vwap_100") - pl.col("selected_vwap_25")) / 75.0).alias(
            "selected_slope_25_100"
        ),
        ((pl.col("selected_vwap_200") - pl.col("selected_vwap_100")) / 100.0).alias(
            "selected_slope_100_200"
        ),
        ((pl.col("opposite_vwap_25") - pl.col("opposite_vwap_5")) / 20.0).alias(
            "opposite_slope_5_25"
        ),
        ((pl.col("opposite_vwap_100") - pl.col("opposite_vwap_25")) / 75.0).alias(
            "opposite_slope_25_100"
        ),
        ((pl.col("opposite_vwap_200") - pl.col("opposite_vwap_100")) / 100.0).alias(
            "opposite_slope_100_200"
        ),
        (pl.col("selected_vwap_100") - pl.col("selected_vwap_5")).alias(
            "selected_slippage_5_100"
        ),
        (pl.col("selected_vwap_200") - pl.col("selected_vwap_5")).alias(
            "selected_slippage_5_200"
        ),
        (pl.col("opposite_vwap_200") - pl.col("opposite_vwap_5")).alias(
            "opposite_slippage_5_200"
        ),
    ).with_columns(
        (pl.col("selected_slope_100_200") - pl.col("selected_slope_25_100")).alias(
            "selected_curve_convexity"
        ),
        (pl.col("opposite_slope_100_200") - pl.col("opposite_slope_25_100")).alias(
            "opposite_curve_convexity"
        ),
        (pl.col("selected_slippage_5_200") - pl.col("opposite_slippage_5_200")).alias(
            "capacity_slippage_imbalance"
        ),
    )


def _attach_enter_now_target(frame: pl.DataFrame) -> pl.DataFrame:
    ordered = frame.sort(["market_id", "seconds_elapsed"])
    markets = ordered["market_id"].to_numpy()
    edge = ordered["realized_stress_edge"].to_numpy().astype(float)
    advantage = np.full(ordered.height, np.nan)
    starts = np.r_[0, np.flatnonzero(markets[1:] != markets[:-1]) + 1]
    ends = np.r_[starts[1:], ordered.height]
    for start, end in zip(starts, ends, strict=True):
        future_best = 0.0
        for index in range(int(end) - 1, int(start) - 1, -1):
            if np.isfinite(edge[index]):
                advantage[index] = edge[index] - future_best
                future_best = max(future_best, edge[index])
    return ordered.with_columns(pl.Series("realized_enter_now_advantage", advantage))


def _admission_frame(
    predictions: pl.DataFrame,
    all_predictions: pl.DataFrame,
    panel: pl.DataFrame,
    config: TournamentDataConfig,
) -> pl.DataFrame:
    opportunities = mst._opportunities(predictions, panel, config)
    contextual = [
        name
        for name in (*hat.NON_L2_ADMISSION_FEATURES[12:], *hat.manifest_l2_features(panel))
        if name in panel.columns and name not in opportunities.columns
    ]
    curve = [
        name
        for name in _curve_context_columns(panel)
        if name not in opportunities.columns and name not in contextual
    ]
    context = panel.select(*KEY_COLUMNS, *contextual, *curve)
    frame = opportunities.join(context, on=list(KEY_COLUMNS), how="left", validate="m:1")
    frame = frame.join(
        _prediction_context(all_predictions), on=list(KEY_COLUMNS), how="left", validate="m:1"
    )
    reserve = float(config.raw["execution"]["execution_reserve_per_share"])
    stress = float(config.raw["execution"]["stress_slippage_per_share"])
    correct = ((pl.col("side") == "up") & (pl.col("label_up") == 1)) | (
        (pl.col("side") == "down") & (pl.col("label_up") == 0)
    )
    frame = (
        frame.sort(["market_id", "seconds_elapsed"])
        .with_columns(
            (pl.col("probability") - pl.col("probability").shift(1).over("market_id"))
            .fill_null(0.0)
            .alias("probability_change_5s"),
            (pl.col("probability") - pl.col("probability").shift(3).over("market_id"))
            .fill_null(0.0)
            .alias("probability_change_15s"),
            pl.col("probability")
            .rolling_std(7)
            .over("market_id")
            .fill_null(0.0)
            .alias("probability_instability_30s"),
            _time_band_expression(config),
            pl.col("share_cost")
            .cut([0.65, 0.80, 0.95], labels=["0", "1", "2", "3"])
            .cast(pl.Int8)
            .alias("price_bucket_index"),
            correct.alias("direction_correct"),
        )
        .with_columns(
            (
                pl.col("direction_correct").cast(pl.Float64)
                - pl.col("share_cost")
                - pl.col("fee_per_share")
                - reserve
                - stress
            ).alias("realized_stress_edge")
        )
    )
    return _attach_enter_now_target(_attach_curve_features(frame))


def _matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return np.column_stack(
        [
            frame[name].cast(pl.Float64).fill_nan(None).fill_null(0.0).to_numpy()
            for name in features
        ]
    )


def _market_weights(frame: pl.DataFrame) -> np.ndarray:
    counts = frame.group_by("market_id").len().rename({"len": "market_rows"})
    return (
        frame.join(counts, on="market_id", how="left")["market_rows"]
        .cast(pl.Float64)
        .pow(-1)
        .to_numpy()
    )


def _full_curve_feature_names(frame: pl.DataFrame) -> tuple[str, ...]:
    names = []
    prefixes = (
        "selected_vwap_",
        "opposite_vwap_",
        "vwap_overround_",
        "selected_slope_",
        "opposite_slope_",
    )
    explicit = {
        "selected_depth_log",
        "opposite_depth_log",
        "selected_depth_imbalance",
        "vwap_curve_available",
        "up_book_age_missing",
        "down_book_age_missing",
        "selected_slippage_5_100",
        "selected_slippage_5_200",
        "opposite_slippage_5_200",
        "selected_curve_convexity",
        "opposite_curve_convexity",
        "capacity_slippage_imbalance",
    }
    for name in frame.columns:
        if name.startswith(prefixes) or name in explicit:
            names.append(name)
    return tuple(names)


def _features(frame: pl.DataFrame, mode: str) -> tuple[str, ...]:
    names = [name for name in hat.NON_L2_ADMISSION_FEATURES if name in frame.columns]
    if mode != "hybrid_vwap5":
        names.extend(name for name in _full_curve_feature_names(frame) if name not in names)
    if mode == "hybrid_full_vwap_dual_l2":
        names.extend(
            name for name in hat.manifest_l2_features(frame) if name not in names
        )
    forbidden = [
        name
        for name in names
        if name in {"realized_stress_edge", "realized_enter_now_advantage"}
        or any(token in name.lower() for token in ("label", "settlement", "twap"))
    ]
    if forbidden:
        raise RuntimeError(f"target fields entered admission feature contract: {forbidden}")
    return tuple(name for name in names if frame[name].drop_nulls().n_unique() > 1)


def _fit_admission_estimators(
    frame: pl.DataFrame, features: tuple[str, ...], config: TournamentDataConfig, seed: int
) -> tuple[Any, Any, Any]:
    valid = frame.filter(
        pl.col("share_cost").is_not_null()
        & pl.col("realized_stress_edge").is_finite()
        & pl.col("realized_enter_now_advantage").is_finite()
    )
    if valid["market_id"].n_unique() < 250:
        raise RuntimeError("VWAP admission model lacks chronological executable markets")
    weights = _market_weights(valid)
    spec = config.raw["hybrid_admission"]
    common = {
        "learning_rate": 0.04,
        "max_iter": int(spec.get("max_iter", 120)),
        "max_leaf_nodes": 15,
        "min_samples_leaf": 100,
        "l2_regularization": 5.0,
        "early_stopping": False,
    }
    matrix = _matrix(valid, features)
    classifier = HistGradientBoostingClassifier(
        **common, random_state=seed
    ).fit(
        matrix,
        (valid["realized_stress_edge"] > 0).cast(pl.Int8),
        sample_weight=weights,
    )
    stress = HistGradientBoostingRegressor(
        **common, random_state=seed + 1
    ).fit(matrix, valid["realized_stress_edge"], sample_weight=weights)
    enter_now = HistGradientBoostingRegressor(
        **common, random_state=seed + 2
    ).fit(matrix, valid["realized_enter_now_advantage"], sample_weight=weights)
    return classifier, stress, enter_now


def _attach_admission(frame: pl.DataFrame, model: VwapAdmissionModel) -> pl.DataFrame:
    matrix = _matrix(frame, model.feature_names)
    probability = model.profitable_classifier.predict_proba(matrix)[:, 1]
    expected = model.stress_edge_regressor.predict(matrix)
    enter_now = model.enter_now_regressor.predict(matrix)
    penalties = np.asarray(
        [
            model.lower_bound_penalties.get(
                f"{band}:{bucket}", model.global_lower_bound_penalty
            )
            for band, bucket in zip(
                frame["time_band"].to_list(),
                frame["price_bucket_index"].to_list(),
                strict=True,
            )
        ]
    )
    conditional_loss = (
        frame["share_cost"].fill_null(1.0).to_numpy()
        + model.stress_slippage_per_share
    )
    return frame.with_columns(
        pl.Series("admission_probability", probability),
        pl.Series("payoff_expected_stress_edge", expected),
        pl.Series("payoff_stress_edge_lower_bound", expected - penalties),
        pl.Series("payoff_expected_shortfall", (1.0 - probability) * conditional_loss),
        pl.Series("enter_now_expected_advantage", enter_now),
    )


def fit_admission_oof(
    frame: pl.DataFrame,
    config: TournamentDataConfig,
    *,
    mode: str,
    seed: int,
) -> tuple[VwapAdmissionModel, pl.DataFrame, dict[str, Any]]:
    features = _features(frame, mode)
    fold_names = [row["name"] for row in config.raw["folds"]]
    pieces: list[pl.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for index, fold in enumerate(fold_names[1:], start=1):
        fit = frame.filter(pl.col("fold").is_in(fold_names[:index]))
        validation = frame.filter(pl.col("fold") == fold)
        fit_markets = fit.filter(pl.col("share_cost").is_not_null())["market_id"].n_unique()
        validation_markets = validation.filter(pl.col("share_cost").is_not_null())[
            "market_id"
        ].n_unique()
        if fit_markets < 250 or validation_markets < 40:
            diagnostics.append(
                {
                    "fold": fold,
                    "skipped": True,
                    "training_executable_markets": fit_markets,
                    "validation_executable_markets": validation_markets,
                }
            )
            continue
        classifier, stress, enter_now = _fit_admission_estimators(
            fit, features, config, seed + 100 * index
        )
        temporary = VwapAdmissionModel(
            features,
            classifier,
            stress,
            enter_now,
            {},
            0.0,
            float(config.raw["execution"]["stress_slippage_per_share"]),
            mode,
        )
        pieces.append(_attach_admission(validation, temporary))
        diagnostics.append(
            {
                "fold": fold,
                "training_markets": fit["market_id"].n_unique(),
                "validation_markets": validation["market_id"].n_unique(),
                "training_executable_markets": fit_markets,
                "validation_executable_markets": validation_markets,
            }
        )
    if not pieces:
        raise RuntimeError(f"{mode} admission OOF produced no chronological folds")
    oof = pl.concat(pieces, how="vertical_relaxed", rechunk=True)
    residual = (
        oof["payoff_expected_stress_edge"].to_numpy()
        - oof["realized_stress_edge"].to_numpy()
    )
    quantile = float(config.raw["hybrid_admission"].get("lower_bound_quantile", 0.90))
    global_penalty = float(np.quantile(residual[np.isfinite(residual)], quantile))
    penalties: dict[str, float] = {}
    for band, _, _ in mst._calibration_cells(config):
        for bucket in range(4):
            cell = oof.filter(
                (pl.col("time_band") == band)
                & (pl.col("price_bucket_index") == bucket)
            )
            values = (
                cell["payoff_expected_stress_edge"].to_numpy()
                - cell["realized_stress_edge"].to_numpy()
            )
            values = values[np.isfinite(values)]
            if not len(values):
                penalties[f"{band}:{bucket}"] = global_penalty
                continue
            cell_penalty = float(np.quantile(values, quantile))
            markets = cell["market_id"].n_unique()
            shrink = markets / (markets + 200.0)
            penalties[f"{band}:{bucket}"] = (
                shrink * cell_penalty + (1.0 - shrink) * global_penalty
            )
    classifier, stress, enter_now = _fit_admission_estimators(
        frame, features, config, seed + 10_000
    )
    model = VwapAdmissionModel(
        features,
        classifier,
        stress,
        enter_now,
        penalties,
        global_penalty,
        float(config.raw["execution"]["stress_slippage_per_share"]),
        mode,
    )
    oof_penalties = np.asarray(
        [
            penalties.get(f"{band}:{bucket}", global_penalty)
            for band, bucket in zip(
                oof["time_band"].to_list(),
                oof["price_bucket_index"].to_list(),
                strict=True,
            )
        ]
    )
    oof = oof.with_columns(
        pl.Series(
            "payoff_stress_edge_lower_bound",
            oof["payoff_expected_stress_edge"].to_numpy() - oof_penalties,
        )
    )
    return model, oof, {
        "mode": mode,
        "features": list(features),
        "folds": diagnostics,
        "oof_markets": oof["market_id"].n_unique(),
        "global_lower_bound_penalty": global_penalty,
        "empirical_lower_bound_coverage": float(
            (
                oof["realized_stress_edge"]
                >= oof["payoff_stress_edge_lower_bound"]
            ).mean()
        ),
        "enter_now_target": "current realized stress edge minus best later executable stress edge",
    }


def _basic_policy(config: TournamentDataConfig) -> dict[str, float]:
    execution = config.raw["execution"]
    return {
        "minimum_edge": float(min(execution["minimum_edges"])),
        "minimum_confidence": float(min(execution["minimum_confidences"])),
        "maximum_share_cost": float(max(execution["maximum_share_costs"])),
        "abstain": False,
    }


def _hybrid_trades(
    frame: pl.DataFrame,
    base_policy: dict[str, float],
    policy: dict[str, float],
    config: TournamentDataConfig,
) -> pl.DataFrame:
    if policy.get("abstain", False):
        return mst._select_trades(frame.head(0), base_policy, config)
    eligible = frame.filter(
        (pl.col("admission_probability") >= policy["minimum_admission_probability"])
        & (
            pl.col("payoff_stress_edge_lower_bound")
            >= policy["minimum_stress_edge_lower_bound"]
        )
        & (pl.col("payoff_expected_shortfall") <= policy["maximum_expected_shortfall"])
        & (
            pl.col("enter_now_expected_advantage")
            >= policy["minimum_enter_now_advantage"]
        )
    )
    return mst._select_trades(eligible, base_policy, config)


def select_hybrid_policies(
    frame: pl.DataFrame, config: TournamentDataConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = _basic_policy(config)
    selected: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    spec = config.raw["hybrid_admission"]
    for band, start, end in mst._calibration_cells(config):
        part = frame.filter(
            pl.col("seconds_elapsed").is_between(start, end, closed="both")
        )
        best: tuple[float, dict[str, float], pl.DataFrame] | None = None
        attempted = 0
        for probability in spec["probability_thresholds"]:
            for lower_bound in spec["lower_bound_thresholds"]:
                for shortfall in spec["maximum_expected_shortfalls"]:
                    for now_advantage in spec["minimum_now_advantages"]:
                        attempted += 1
                        policy = {
                            "minimum_admission_probability": float(probability),
                            "minimum_stress_edge_lower_bound": float(lower_bound),
                            "maximum_expected_shortfall": float(shortfall),
                            "minimum_enter_now_advantage": float(now_advantage),
                            "abstain": False,
                        }
                        trades = _hybrid_trades(part, base, policy, config)
                        if trades.is_empty():
                            continue
                        fold_pnl = trades.group_by("fold").agg(
                            pl.col("net_pnl").sum()
                        )
                        stable = float((fold_pnl["net_pnl"] > 0).mean()) >= float(
                            config.raw["execution"].get(
                                "minimum_positive_fold_ratio", 0.0
                            )
                        )
                        score = mst._policy_score(trades)
                        metrics = mst.economic_metrics(
                            trades, part["market_id"].n_unique()
                        )
                        stable = (
                            stable
                            and metrics["stress_net_pnl"] > 0
                            and score > 0
                        )
                        if stable and (best is None or score > best[0]):
                            best = (score, policy, trades)
        if best is None:
            selected[band] = {**base, "abstain": True}
            evidence[band] = {"attempted": attempted, "abstained": True}
        else:
            score, policy, trades = best
            selected[band] = policy
            evidence[band] = {
                "attempted": attempted,
                "abstained": False,
                "selection_score": score,
                "rolling_oof": mst.economic_metrics(
                    trades, part["market_id"].n_unique()
                ),
            }
    return selected, evidence


def _combined_hybrid_trades(
    frame: pl.DataFrame,
    policies: dict[str, Any],
    config: TournamentDataConfig,
) -> pl.DataFrame:
    pieces = []
    base = _basic_policy(config)
    for band, start, end in mst._calibration_cells(config):
        part = frame.filter(
            pl.col("seconds_elapsed").is_between(start, end, closed="both")
        )
        pieces.append(_hybrid_trades(part, base, policies[band], config))
    output = pl.concat(pieces, how="diagonal_relaxed")
    if output.is_empty():
        return output
    return (
        output.sort(["market_id", "seconds_elapsed"])
        .group_by("market_id", maintain_order=True)
        .first()
    )


def _mode_replays(
    frame: pl.DataFrame,
    *,
    mode: str,
    programmatic: dict[str, Any],
    hybrid: dict[str, Any] | None,
    config: TournamentDataConfig,
) -> dict[str, pl.DataFrame]:
    output: dict[str, pl.DataFrame] = {}
    for band, start, end in mst._calibration_cells(config):
        part = frame.filter(
            pl.col("seconds_elapsed").is_between(start, end, closed="both")
        )
        if mode == "veto_disabled":
            output[band] = mst._select_trades(part, _basic_policy(config), config)
        elif mode == "programmatic":
            output[band] = mst._select_trades(part, programmatic[band], config)
        else:
            assert hybrid is not None
            output[band] = _hybrid_trades(
                part, _basic_policy(config), hybrid[band], config
            )
    if mode == "veto_disabled":
        output["combined_60_180"] = mst._select_trades(
            frame, _basic_policy(config), config
        )
    elif mode == "programmatic":
        output["combined_60_180"] = mst._apply_cell_policies(
            frame, programmatic, config
        )
    else:
        assert hybrid is not None
        output["combined_60_180"] = _combined_hybrid_trades(
            frame, hybrid, config
        )
    return output


def _periods(config: TournamentDataConfig) -> dict[str, tuple[datetime, datetime]]:
    cutover = datetime.fromisoformat(config.raw["windows"]["cutover"])
    return {
        "full_august": (config.sealed_start, config.sealed_end),
        "pre_cutover_august_1_13": (config.sealed_start, cutover),
        "post_cutover_august_14_31": (cutover, config.sealed_end),
    }


def _evaluate(
    *,
    predictions: pl.DataFrame,
    panel: pl.DataFrame,
    programmatic: dict[str, Any],
    admission_models: dict[str, dict[str, VwapAdmissionModel]],
    hybrid_policies: dict[str, dict[str, Any]],
    config: TournamentDataConfig,
) -> tuple[dict[str, Any], dict[str, Any], pl.DataFrame]:
    economic: dict[str, Any] = {}
    predictive: dict[str, Any] = {}
    ledgers: list[pl.DataFrame] = []
    periods = _periods(config)
    for candidate in mst.ALL_NAMES:
        candidate_predictions = predictions.filter(pl.col("candidate") == candidate)
        predictive[candidate] = {}
        for period, (start, end) in periods.items():
            predictive_row = mst.predictive_metrics(
                candidate_predictions.filter(
                    pl.col("window_start").is_between(start, end, closed="left")
                )
            )
            scheduled_markets = int((end - start).total_seconds() // 300)
            predictive_row["scheduled_markets"] = scheduled_markets
            predictive_row["data_coverage"] = (
                predictive_row["markets"] / scheduled_markets
                if scheduled_markets
                else 0.0
            )
            predictive[candidate][period] = predictive_row
        admission = _admission_frame(
            candidate_predictions, predictions, panel, config
        )
        mode_frames = {
            "veto_disabled": admission,
            "programmatic": admission,
            **{
                mode: _attach_admission(admission, admission_models[candidate][mode])
                for mode in ADMISSION_MODES
            },
        }
        economic[candidate] = {}
        for mode, scored in mode_frames.items():
            economic[candidate][mode] = {}
            for period, (start, end) in periods.items():
                part = scored.filter(
                    pl.col("window_start").is_between(start, end, closed="left")
                )
                total_markets = panel.filter(
                    pl.col("window_start").is_between(start, end, closed="left")
                )["market_id"].n_unique()
                scheduled_markets = int((end - start).total_seconds() // 300)
                replays = _mode_replays(
                    part,
                    mode=mode,
                    programmatic=programmatic[candidate],
                    hybrid=(
                        hybrid_policies[candidate].get(mode)
                        if mode in ADMISSION_MODES
                        else None
                    ),
                    config=config,
                )
                replay_metrics = {}
                for replay, trades in replays.items():
                    row = mst.economic_metrics(trades, total_markets)
                    row["scheduled_markets"] = scheduled_markets
                    row["data_covered_markets"] = total_markets
                    row["data_coverage"] = (
                        total_markets / scheduled_markets if scheduled_markets else 0.0
                    )
                    row["end_to_end_market_coverage"] = (
                        trades["market_id"].n_unique() / scheduled_markets
                        if scheduled_markets
                        else 0.0
                    )
                    replay_metrics[replay] = row
                economic[candidate][mode][period] = replay_metrics
                for replay, trades in replays.items():
                    if not trades.is_empty():
                        ledgers.append(
                            trades.with_columns(
                                pl.lit(candidate).alias("candidate"),
                                pl.lit(mode).alias("admission_mode"),
                                pl.lit(period).alias("evaluation_period"),
                                pl.lit(replay).alias("entry_replay"),
                            )
                        )
    return (
        predictive,
        economic,
        pl.concat(ledgers, how="diagonal_relaxed") if ledgers else pl.DataFrame(),
    )


def _preferred_modes(config: TournamentDataConfig) -> dict[str, str]:
    return {row["name"]: row["preferred_admission"] for row in config.raw["candidates"]}


def _report(metrics: dict[str, Any]) -> str:
    def fmt(value: Any, digits: int = 3) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Full-August VWAP-Curve Admission Tournament",
        "",
        f"Run: `{metrics['run_id']}`",
        f"Qualification: **{metrics['qualification_status']}**",
        "",
        "## High-level preferred-mode results: full August, combined 60–180 replay",
        "",
        "| Candidate | Admission | PnL | Stress PnL | PF | Coverage | W | L | W/L | Recovery wins/loss | Avg entry | Avg cost | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in metrics["candidate_names"]:
        mode = metrics["preferred_admission_modes"][candidate]
        row = metrics["heldout_economic"][candidate][mode]["full_august"][
            "combined_60_180"
        ]
        brier = metrics["heldout_predictive"][candidate]["full_august"]["brier_score"]
        lines.append(
            f"| {candidate} | {mode} | {row['net_pnl']:.2f} | {row['stress_net_pnl']:.2f} | {fmt(row['profit_factor'])} | {row['end_to_end_market_coverage']:.2%} | {row['winning_trades']} | {row['losing_trades']} | {fmt(row['win_loss_ratio'])} | {fmt(row['loss_recovery_wins'])} | {fmt(row['average_entry_second'], 1)} | {fmt(row['average_share_cost'])} | {fmt(brier, 4)} |"
        )
    lines.extend(
        (
            "",
            "## Evaluation contract",
            "",
            "- Predictive fitting, probability calibration, admission fitting, and policy selection use only markets before August 1.",
            "- August 1–31 is opened once after selection is frozen and is reported as full month, pre-cutover, and post-cutover.",
            "- The 60–89, 90–119, 120–149, and 150–180 results are independent entry replays; combined 60–180 resets and selects the first crossing across the full range.",
            "- Official resolved outcomes are the only predictive supervision. TWAP, RefPrice settlement normalization, and bridge targets are absent.",
            "- All fourteen VWAP sizes are admission-only; directional candidate contracts remain unchanged.",
            "- Optional source gaps preserve rows and are encoded as missingness. No authentic-only filter is present.",
            "- No database writes, tables, schemas, ingesters, sources, runtime exports, deployments, or image rebuilds occurred.",
        )
    )
    return "\n".join(lines) + "\n"


def _validate_artifact(config: TournamentDataConfig, artifact: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(config.package_root / "src")
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import joblib,sys; x=joblib.load(sys.argv[1]); "
                "assert len(x['base_models'])==5; "
                "assert len(x['band_calibrators'])==7; "
                "assert len(x['admission_models'])==21"
            ),
            str(artifact),
        ],
        check=True,
        cwd=config.package_root,
        env=environment,
    )


def train_tournament(
    config: TournamentDataConfig,
    *,
    force_tail: bool = False,
    resume_run: str | None = None,
) -> Path:
    mst._configure_roster(config)
    panel, panel_manifest = build_vwap_panel(config, force_tail=force_tail)
    contracts = mst._candidate_contract(config, panel_manifest)
    run_id = resume_run or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.results / run_id
    ledgers = run_dir / "ledgers"
    if resume_run:
        if not run_dir.is_dir() or (run_dir / "completion.json").exists():
            raise RuntimeError("resume run must be an existing incomplete checkpoint")
        ledgers.mkdir(parents=True, exist_ok=True)
    else:
        ledgers.mkdir(parents=True, exist_ok=False)

    fit = panel.filter(pl.col("window_start") < config.fit_end)
    heldout = panel.filter(
        pl.col("window_start").is_between(
            config.sealed_start, config.sealed_end, closed="left"
        )
    )
    folds = config.raw["folds"]
    if any(datetime.fromisoformat(row["test_end"]) > config.fit_end for row in folds):
        raise RuntimeError("an OOF fold extends into the full-August holdout")
    fit_markets = set(fit["market_id"].unique())
    heldout_markets = set(heldout["market_id"].unique())
    if fit_markets & heldout_markets:
        raise RuntimeError("predictive fit and full-August holdout markets overlap")
    split = {
        "source_start": config.source_start.isoformat(),
        "fit_end_exclusive": config.fit_end.isoformat(),
        "heldout_start_inclusive": config.sealed_start.isoformat(),
        "heldout_end_exclusive": config.sealed_end.isoformat(),
        "fit_markets": len(fit_markets),
        "heldout_markets": len(heldout_markets),
        "folds": folds,
        "official_outcome_supervision_only": True,
        "time_band_calibration_pre_august_only": True,
        "predictive_oof_only_for_admission": True,
        "nested_chronological_admission_oof": True,
        "heldout_excluded_from_model_calibration_admission_and_policy_selection": True,
        "market_disjoint": True,
    }
    split_path = run_dir / "split-manifest.json"
    if split_path.exists():
        if json.loads(split_path.read_text()) != split:
            raise RuntimeError("resume split differs from its frozen checkpoint")
    else:
        _write_json(split_path, split)

    oof_path = ledgers / "candidate-oof-predictions.parquet"
    if oof_path.exists():
        calibrated_oof = pl.read_parquet(oof_path)
        raw_oof = calibrated_oof.with_columns(
            pl.col("raw_probability").alias("probability")
        )
        wide_oof = mst._wide_base_predictions(
            raw_oof.filter(pl.col("candidate").is_in(mst.BASE_NAMES))
        )
    else:
        base_oof = mst._base_oof(fit, contracts, config)
        raw_oof, wide_oof = mst._all_oof(base_oof, config)
        calibrated_oof, band_calibrators = calibrate_oof_by_time_band(raw_oof, config)
        calibrated_oof.write_parquet(oof_path, compression="zstd", statistics=True)
        joblib.dump(
            band_calibrators,
            run_dir / "band-calibrator-checkpoint.joblib",
            compress=3,
        )
    band_calibrators = joblib.load(run_dir / "band-calibrator-checkpoint.joblib")
    oof_predictive = {
        name: mst.predictive_metrics(
            calibrated_oof.filter(pl.col("candidate") == name)
        )
        for name in mst.ALL_NAMES
    }

    seed = int(config.raw["training"]["random_seed"])
    predictive_checkpoint = run_dir / "predictive-model-checkpoint.joblib"
    if predictive_checkpoint.exists():
        predictive_payload = joblib.load(predictive_checkpoint)
        final_models = predictive_payload["base_models"]
        ensemble_calibrator = predictive_payload["ensemble_calibrator"]
    else:
        final_models = {
            name: mst._fit_candidate_tree(
                fit,
                tuple(contracts[name]["features"]),
                config,
                seed + 10_000 + index,
                target_kind=contracts[name]["kind"],
            )
            for index, name in enumerate(mst.BASE_NAMES)
        }
        ensemble_calibrator = mst._fit_ensemble_calibrator(wide_oof, seed + 20_000)
        joblib.dump(
            {
                "base_models": final_models,
                "ensemble_calibrator": ensemble_calibrator,
            },
            predictive_checkpoint,
            compress=3,
        )

    programmatic: dict[str, Any] = {}
    programmatic_evidence: dict[str, Any] = {}
    admission_models: dict[str, dict[str, VwapAdmissionModel]] = {}
    hybrid_policies: dict[str, dict[str, Any]] = {}
    admission_evidence: dict[str, Any] = {}
    for candidate_index, candidate in enumerate(mst.ALL_NAMES):
        candidate_oof = calibrated_oof.filter(pl.col("candidate") == candidate)
        frame = _admission_frame(candidate_oof, calibrated_oof, fit, config)
        programmatic[candidate], programmatic_evidence[candidate] = mst._select_cell_policies(
            frame, config
        )
        admission_models[candidate] = {}
        hybrid_policies[candidate] = {}
        admission_evidence[candidate] = {}
        for mode_index, mode in enumerate(ADMISSION_MODES):
            checkpoint = run_dir / f"admission-{candidate}-{mode}.joblib"
            if checkpoint.exists():
                payload = joblib.load(checkpoint)
                model = payload["model"]
                policies = payload["policies"]
                evidence = payload["evidence"]
            else:
                model, admission_oof, model_evidence = fit_admission_oof(
                    frame,
                    config,
                    mode=mode,
                    seed=seed + 30_000 + 300 * candidate_index + 10 * mode_index,
                )
                policies, policy_evidence = select_hybrid_policies(
                    admission_oof, config
                )
                evidence = {**model_evidence, "policy": policy_evidence}
                joblib.dump(
                    {"model": model, "policies": policies, "evidence": evidence},
                    checkpoint,
                    compress=3,
                )
            admission_models[candidate][mode] = model
            hybrid_policies[candidate][mode] = policies
            admission_evidence[candidate][mode] = evidence
            print(f"admission checkpoint: {candidate} {mode}", flush=True)

    selection_frozen_at = datetime.now(UTC).isoformat()
    selection_payload = {
        "frozen_at": selection_frozen_at,
        "programmatic": programmatic,
        "programmatic_evidence": programmatic_evidence,
        "hybrid": hybrid_policies,
        "admission_evidence": admission_evidence,
        "preferred_modes": _preferred_modes(config),
        "heldout_metrics_accessed": False,
    }
    _write_json(run_dir / "selection-freeze.json", selection_payload)

    heldout_predictions = mst._predict_frozen_candidates(
        heldout, final_models, ensemble_calibrator, "heldout_20260801_20260901"
    )
    heldout_predictions = apply_band_calibrators(
        heldout_predictions, band_calibrators, config
    )
    heldout_predictions.write_parquet(
        ledgers / "heldout-predictions.parquet", compression="zstd", statistics=True
    )
    heldout_predictive, heldout_economic, heldout_trades = _evaluate(
        predictions=heldout_predictions,
        panel=heldout,
        programmatic=programmatic,
        admission_models=admission_models,
        hybrid_policies=hybrid_policies,
        config=config,
    )
    heldout_trades.write_parquet(
        ledgers / "heldout-trades.parquet", compression="zstd", statistics=True
    )

    producing_commit = _git_revision(config.package_root)
    artifact = run_dir / "tournament.joblib"
    flattened_admission = {
        f"{candidate}:{mode}": model
        for candidate, modes in admission_models.items()
        for mode, model in modes.items()
    }
    joblib.dump(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "producing_commit": producing_commit,
            "candidate_contract": contracts,
            "base_models": final_models,
            "ensemble_calibrator": ensemble_calibrator,
            "band_calibrators": band_calibrators,
            "admission_models": flattened_admission,
            "programmatic_policies": programmatic,
            "hybrid_policies": hybrid_policies,
            "preferred_admission_modes": _preferred_modes(config),
            "deployment_status": "not_deployed",
        },
        artifact,
        compress=3,
    )
    _validate_artifact(config, artifact)
    artifact_sha = file_sha256(artifact)
    (run_dir / "tournament.sha256").write_text(f"{artifact_sha}  tournament.joblib\n")

    preferred = _preferred_modes(config)
    positive = []
    for candidate, mode in preferred.items():
        row = heldout_economic[candidate][mode]["full_august"]["combined_60_180"]
        if (
            row["net_pnl"] > 0
            and row["stress_net_pnl"] > 0
            and (row["profit_factor"] or 0) > 1
        ):
            positive.append({"candidate": candidate, "mode": mode})
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "producing_commit": producing_commit,
        "candidate_names": list(mst.ALL_NAMES),
        "candidate_count_trained": 7,
        "learned_base_model_count": 5,
        "derived_ensemble_count": 2,
        "probability_band_calibrator_count": 28,
        "admission_model_count": 21,
        "selection_frozen_at": selection_frozen_at,
        "preferred_admission_modes": preferred,
        "oof_predictive": oof_predictive,
        "heldout_predictive": heldout_predictive,
        "heldout_economic": heldout_economic,
        "positive_expectancy_preferred_modes": positive,
        "qualification_status": (
            "trained_evaluated_not_deployed"
            if positive
            else "trained_evaluated_not_promoted_negative_expectancy"
        ),
        "source_panel": panel_manifest,
        "split_manifest": split,
        "artifact_sha256": artifact_sha,
        "frozen_comparator_references": mst._reference_manifests(config),
        "integrity": {
            "passed": True,
            "market_disjoint": True,
            "full_august_heldout": True,
            "official_outcome_supervision_only": True,
            "settlement_bridge_fields_absent": True,
            "time_band_calibration_pre_august_only": True,
            "nested_chronological_admission_oof": True,
            "independent_entry_replays": True,
            "artifact_round_trip_load": True,
            "full_history_retained": True,
            "optional_missingness_preserves_rows": True,
            "authentic_only_filter": False,
            "twap_inference_feature": False,
            "vwap_curve_directional_feature": False,
            "database_mutations": False,
            "new_tables": False,
            "new_schemas": False,
            "new_ingesters": False,
            "new_sources": False,
            "images_rebuilt": False,
            "runtime_exported": False,
            "deployed": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "sklearn": sklearn.__version__,
        },
        "limitations": [
            "The August holdout is chronologically clean for this run but was observed during earlier research, so it is not epistemically fresh.",
            "Kraken L2 incremental-update coverage ends during August 19; later rows preserve explicit L2 missingness.",
            "Kraken candles and prints end during August 30; later rows preserve explicit Kraken missingness.",
            "The latest capacity artifacts extend through August 31 20:59 UTC, but their August 27–31 snapshots contain missing-book flags and null curves; usable complete VWAP curves end August 25 23:59 UTC and later rows preserve explicit execution missingness.",
            "Projected PnL assumes recorded Polymarket ask VWAP5 was fillable and does not model queue position.",
        ],
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "candidate-contract.json", contracts)
    _write_json(run_dir / "source-manifest.json", panel_manifest)
    (run_dir / "report.md").write_text(_report(metrics))
    _write_json(
        run_dir / "completion.json",
        {
            "run_id": run_id,
            "completed": True,
            "artifact_sha256": artifact_sha,
            "metrics_sha256": file_sha256(run_dir / "metrics.json"),
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force-tail", action="store_true")
    parser.add_argument("--resume-run")
    args = parser.parse_args()
    result = train_tournament(
        load_data_config(args.config),
        force_tail=args.force_tail,
        resume_run=args.resume_run,
    )
    print(result)


if __name__ == "__main__":
    main()
