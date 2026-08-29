"""Causal data construction for the counterfactual TWAP-state model family.

Only existing immutable database evidence is read. Database writes, new tables,
new sources, and new ingesters are deliberately outside this module's contract.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from .core_extract import file_sha256, write_json_atomic
from .core_features import (
    attach_causal_oracle_rounds,
    derive_core_point_in_time_features,
    derive_oracle_point_in_time_features,
    prepare_causal_oracle_rounds,
)
from .twap60_training_data import (
    CORE_ORACLE_ROUND_SCHEMA,
    DataPaths,
    ProxyConvention,
    _isolated_query_frame,
    attach_candle_context,
    attach_causal_refprice_features,
    attach_execution,
    authentic_labels,
    canonical_refprice_path,
    construct_proxy_labels,
    ensure_oracle_eligibility_compatibility,
    extract_tournament_sources,
    load_source_group,
    piecewise_average,
    select_proxy_convention,
)

CHAINLINK_UNCERTAINTY_BPS = 0.526

REFPRICE_STATE_FEATURES = (
    "chainlink_ref_return_1s_bps",
    "chainlink_ref_return_5s_bps",
    "chainlink_ref_return_15s_bps",
    "chainlink_ref_return_30s_bps",
    "chainlink_ref_return_60s_bps",
    "chainlink_ref_boundary_gap_bps",
    "chainlink_ref_realized_volatility_30s_bps",
    "chainlink_ref_realized_volatility_60s_bps",
    "chainlink_ref_path_efficiency_60s",
    "chainlink_ref_boundary_cross_count_60s",
    "chainlink_ref_age_seconds",
    "chainlink_ref_max_gap_60s",
)

TWAP30_FEATURES = (
    "opening_twap30",
    "current_twap30",
    "twap30_change_from_open_bps",
    "twap30_gap_to_opening_twap60_bps",
    "twap30_slope_5s_bps",
    "twap30_slope_15s_bps",
    "twap30_slope_30s_bps",
    "twap30_slope_60s_bps",
)

TWAP60_FEATURES = (
    "opening_twap60",
    "current_twap60",
    "twap60_change_from_open_bps",
    "twap60_gap_to_opening_twap60_bps",
    "twap60_slope_5s_bps",
    "twap60_slope_15s_bps",
    "twap60_slope_30s_bps",
    "twap60_slope_60s_bps",
)

TWAP_CROSS_FEATURES = (
    "opening_refprice",
    "opening_binance_twap30",
    "opening_binance_twap60",
    "opening_binance_chainlink_basis_bps",
    "current_refprice_gap_to_opening_twap60_bps",
    "twap30_minus_twap60_bps",
    "refprice_minus_twap30_bps",
    "refprice_minus_twap60_bps",
    "twap_directional_agreement",
    "refprice_twap_directional_agreement",
    "twap_acceleration_bps",
    "twap_convergence_velocity_bps",
    "refprice_pressure_bps",
    "twap_boundary_cross_count",
    "twap_seconds_since_boundary_cross",
    "twap_path_efficiency",
    "twap_reversal_state",
    "twap_realized_volatility_bps",
    "settlement_margin_volatility_z",
    "twap_completed_prints_30",
    "twap_completed_prints_60",
    "twap_max_timestamp_gap_60s",
    "twap_source_age_seconds",
)

BINANCE_DISAGREEMENT_FEATURES = (
    "binance_twap30_minus_chainlink_twap30_bps",
    "binance_twap60_minus_chainlink_twap60_bps",
    "binance_chainlink_basis_bps",
    "binance_chainlink_basis_velocity_5s_bps",
)

DUAL_TWAP_FEATURES = (*TWAP30_FEATURES, *TWAP60_FEATURES, *TWAP_CROSS_FEATURES)

CAUSAL_BASIS_FEATURES = (
    "chainlink_ref_binance_basis_bps",
    "chainlink_ref_spread_bps",
    "chainlink_ref_spread_change_30s_bps",
    "chainlink_ref_binance_direction_agreement_30s",
    "chainlink_ref_binance_basis_velocity_5s_bps",
    "chainlink_ref_binance_disagreement_30s",
    "chainlink_ref_source_skew_seconds",
)

RELATIVE_TWAP_FEATURES = (
    "twap30_change_from_open_bps",
    "twap60_change_from_open_bps",
    "twap30_gap_to_opening_twap60_bps",
    "twap60_gap_to_opening_twap60_bps",
    "twap30_minus_twap60_bps",
    "refprice_minus_twap30_bps",
    "refprice_minus_twap60_bps",
    "twap_directional_agreement",
    "refprice_twap_directional_agreement",
    "twap30_slope_5s_bps",
    "twap30_slope_15s_bps",
    "twap30_slope_30s_bps",
    "twap30_slope_60s_bps",
    "twap60_slope_5s_bps",
    "twap60_slope_15s_bps",
    "twap60_slope_30s_bps",
    "twap60_slope_60s_bps",
    "twap_acceleration_bps",
    "twap_convergence_velocity_bps",
    "refprice_pressure_bps",
    "twap_boundary_cross_count",
    "twap_seconds_since_boundary_cross",
    "twap_path_efficiency",
    "twap_reversal_state",
    "twap_realized_volatility_bps",
    "settlement_margin_volatility_z",
    "twap_completed_prints_30",
    "twap_completed_prints_60",
    "twap_max_timestamp_gap_60s",
    "twap_source_age_seconds",
)

SUPERVISION_ONLY_FIELDS = frozenset(
    {
        "label_up",
        "label_source",
        "target_margin_bps",
        "authentic_label_up",
        "authentic_margin_bps",
        "proxy_label_up",
        "proxy_margin_bps",
        "binance_raw_margin_bps",
        "binance_corrected_margin_bps",
        "binance_margin_correction_bps",
        "estimated_synthetic_label_error",
        "raw_corrected_synthetic_margin_disagreement_bps",
        "official_outcome",
    }
)

SUPERVISION_REGISTRY: dict[str, dict[str, Any]] = {
    "label_up": {"role": "target", "source": "priority terminal label"},
    "target_margin_bps": {"role": "target", "source": "priority terminal margin"},
    "base_label_weight": {"role": "weight", "source": "frozen label-source confidence"},
    "estimated_synthetic_label_error": {
        "role": "weight",
        "source": "offline Binance synthetic-label reliability",
        "permitted_use": "Binance synthetic sample weighting only",
    },
    "raw_corrected_synthetic_margin_disagreement_bps": {
        "role": "diagnostic",
        "source": "completed Binance terminal margins",
        "permitted_use": "offline leakage-impact reporting only",
    },
    "label_source": {"role": "diagnostic", "source": "priority label assignment"},
    "authentic_label_up": {"role": "target", "source": "authentic completed TWAP-60"},
    "authentic_margin_bps": {"role": "target", "source": "authentic completed TWAP-60"},
    "proxy_label_up": {"role": "target", "source": "completed Chainlink reconstruction"},
    "proxy_margin_bps": {"role": "target", "source": "completed Chainlink reconstruction"},
    "binance_raw_margin_bps": {"role": "target", "source": "completed Binance TWAP-60"},
    "binance_corrected_margin_bps": {
        "role": "target",
        "source": "offline corrected completed Binance TWAP-60",
    },
    "binance_margin_correction_bps": {
        "role": "diagnostic",
        "source": "offline Binance terminal-margin correction",
    },
    "official_outcome": {"role": "diagnostic", "source": "completed official market outcome"},
}

_REFPRICE_INFERENCE_FEATURES = frozenset((*REFPRICE_STATE_FEATURES, *CAUSAL_BASIS_FEATURES))
_TWAP_INFERENCE_FEATURES = frozenset(
    (*TWAP30_FEATURES, *TWAP60_FEATURES, *TWAP_CROSS_FEATURES, *RELATIVE_TWAP_FEATURES)
)
_BINANCE_TWAP_INFERENCE_FEATURES = frozenset(BINANCE_DISAGREEMENT_FEATURES)

INFERENCE_FEATURE_REGISTRY: dict[str, dict[str, Any]] = {
    **{
        name: {
            "role": "inference",
            "source": "chainlink_refprice_and_current_binance",
            "source_event_timestamp": "source_timestamp and observed_at",
            "source_availability_timestamp": [
                "chainlink_twap_max_available_at",
                "core_source_max_available_at",
            ]
            if name in CAUSAL_BASIS_FEATURES
            else ["chainlink_twap_max_available_at"],
            "lookback_interval": "causal trailing window ending at feature_as_of",
            "feature_as_of_timestamp": "observed_at",
            "live_computable": True,
        }
        for name in sorted(_REFPRICE_INFERENCE_FEATURES)
    },
    **{
        name: {
            "role": "inference",
            "source": "causal_rolling_twap_state",
            "source_event_timestamp": "source_timestamp or open_timestamp",
            "source_availability_timestamp": ["twap_feature_max_available_at"],
            "lookback_interval": "[T-W,T) or causal trailing window ending at feature_as_of",
            "feature_as_of_timestamp": "observed_at",
            "live_computable": True,
        }
        for name in sorted(_TWAP_INFERENCE_FEATURES)
    },
    **{
        name: {
            "role": "inference",
            "source": "current_binance_chainlink_twap",
            "source_event_timestamp": "source_timestamp and open_timestamp",
            "source_availability_timestamp": [
                "disagreement_opening_feature_max_available_at"
                if name == "binance_chainlink_basis_velocity_5s_bps"
                else "disagreement_feature_max_available_at"
            ],
            "lookback_interval": "[T-W,T) and causal trailing window ending at feature_as_of",
            "feature_as_of_timestamp": "observed_at",
            "live_computable": True,
        }
        for name in sorted(_BINANCE_TWAP_INFERENCE_FEATURES)
    },
}


def inference_feature_registry() -> dict[str, dict[str, Any]]:
    """Return the immutable causal allowlist used to construct every model matrix."""

    return {name: dict(metadata) for name, metadata in INFERENCE_FEATURE_REGISTRY.items()}


def validate_inference_features(features: tuple[str, ...]) -> None:
    """Fail closed when supervision or unregistered fields enter a model contract."""

    forbidden = sorted(set(features) & SUPERVISION_ONLY_FIELDS)
    unregistered = sorted(set(features) - set(INFERENCE_FEATURE_REGISTRY))
    if forbidden or unregistered:
        problems = []
        if forbidden:
            problems.append("supervision-only: " + ", ".join(forbidden))
        if unregistered:
            problems.append("unregistered: " + ", ".join(unregistered))
        raise ValueError("invalid inference feature contract: " + "; ".join(problems))


CAUSAL_AVAILABILITY_COLUMNS = (
    "chainlink_twap_max_available_at",
    "binance_twap_max_available_at",
    "twap_feature_max_available_at",
    "disagreement_feature_max_available_at",
    "disagreement_opening_feature_max_available_at",
    "core_source_max_available_at",
)


def causal_availability_audit(frame: pl.DataFrame) -> dict[str, Any]:
    """Verify that point-in-time sources never exceed the decision timestamp."""

    missing = sorted(set(CAUSAL_AVAILABILITY_COLUMNS) - set(frame.columns))
    if missing:
        raise RuntimeError("causal availability audit columns missing: " + ", ".join(missing))
    violations: dict[str, int] = {}
    for name in CAUSAL_AVAILABILITY_COLUMNS:
        count = frame.filter(
            pl.col(name).is_not_null() & (pl.col(name) > pl.col("observed_at"))
        ).height
        violations[name] = count
    if any(violations.values()):
        raise RuntimeError(f"future source availability entered inference rows: {violations}")
    feature_violations: dict[str, int] = {}
    for feature, metadata in INFERENCE_FEATURE_REGISTRY.items():
        if feature not in frame.columns:
            continue
        populated = pl.col(feature).is_not_null()
        if frame.schema[feature].is_numeric():
            populated &= pl.col(feature).is_finite()
        availability = metadata["source_availability_timestamp"]
        count = frame.filter(
            populated
            & pl.any_horizontal(
                pl.col(column).is_null() | (pl.col(column) > pl.col("observed_at"))
                for column in availability
            )
        ).height
        feature_violations[feature] = count
    if any(feature_violations.values()):
        raise RuntimeError(
            f"populated inference values failed availability invariant: {feature_violations}"
        )
    return {
        "feature_as_of": "observed_at",
        "rows": frame.height,
        "availability_columns": list(CAUSAL_AVAILABILITY_COLUMNS),
        "violations": violations,
        "feature_violations": feature_violations,
        "features_audited": len(feature_violations),
        "passed": True,
    }


@dataclass(frozen=True)
class CounterfactualDataPaths:
    base: DataPaths
    binance_sql: Path


def extract_counterfactual_sources(
    paths: CounterfactualDataPaths,
    *,
    range_start: datetime,
    range_end: datetime,
    force: bool = False,
) -> dict[str, Any]:
    """Run bounded daily SELECT-only extraction with resumable local caches."""

    manifest = extract_tournament_sources(
        paths.base,
        range_start=range_start,
        range_end=range_end,
        current_start=range_start,
        force=force,
    )
    sql_sha = file_sha256(paths.binance_sql)
    contract = {
        "schema_version": "btc-counterfactual-binance-source-v1",
        "range_start": range_start.isoformat(),
        "range_end": range_end.isoformat(),
        "query_sha256": sql_sha,
        "read_only": True,
        "database_mutations": False,
    }
    checkpoint = paths.base.cache / "binance-manifest.partial.json"
    final = paths.base.cache / "binance-manifest.json"
    partitions: list[dict[str, Any]] = []
    if final.is_file() and not force and not checkpoint.exists():
        payload = json.loads(final.read_text())
        if payload["contract"] != contract:
            raise RuntimeError("existing Binance source contract changed")
        partitions = payload["partitions"]
    else:
        if force:
            checkpoint.unlink(missing_ok=True)
        if checkpoint.is_file():
            payload = json.loads(checkpoint.read_text())
            if payload["contract"] != contract:
                raise RuntimeError("partial Binance source contract changed")
            partitions = payload["partitions"]
        completed = {Path(row["path"]).stem for row in partitions}
        query = paths.binance_sql.read_text()
        day = range_start
        while day < range_end:
            if day.date().isoformat() in completed:
                day = min(day + timedelta(days=1), range_end)
                continue
            end = min(day + timedelta(days=1), range_end)
            frame = _isolated_query_frame(
                query,
                {"batch_start": day, "batch_end": end},
                cursor_name=f"btc_counterfactual_binance_{day:%Y%m%d}",
            )
            directory = paths.base.cache / "binance"
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"{day.date().isoformat()}.parquet"
            frame.write_parquet(destination, compression="zstd", statistics=True)
            partitions.append(
                {
                    "path": str(destination.relative_to(paths.base.cache)),
                    "rows": frame.height,
                    "sha256": file_sha256(destination),
                }
            )
            write_json_atomic(checkpoint, {"contract": contract, "partitions": partitions})
            print(f"counterfactual extract: binance {day.date()} {frame.height:,} rows", flush=True)
            day = end
        write_json_atomic(final, {"contract": contract, "partitions": partitions})
        checkpoint.unlink(missing_ok=True)
    manifest = dict(manifest)
    manifest["counterfactual_binance"] = {"contract": contract, "partitions": partitions}
    return manifest


def load_binance(paths: CounterfactualDataPaths) -> pl.DataFrame:
    payload = json.loads((paths.base.cache / "binance-manifest.json").read_text())
    frames = [pl.read_parquet(paths.base.cache / row["path"]) for row in payload["partitions"]]
    return pl.concat(frames, how="diagonal_relaxed", rechunk=True)


def construct_binance_labels(labels: pl.DataFrame, binance: pl.DataFrame) -> pl.DataFrame:
    """Construct completed-close [T-W,T) TWAP labels, one row per market."""

    samples = binance.filter(pl.col("available_at") <= pl.col("window_end"))
    aggregated = samples.group_by("market_id").agg(
        pl.col("close_price")
        .filter(
            (pl.col("open_timestamp") >= pl.col("window_start") - pl.duration(seconds=30))
            & (pl.col("open_timestamp") < pl.col("window_start"))
        )
        .mean()
        .alias("binance_open_twap30"),
        pl.col("close_price")
        .filter(
            (pl.col("open_timestamp") >= pl.col("window_start") - pl.duration(seconds=60))
            & (pl.col("open_timestamp") < pl.col("window_start"))
        )
        .mean()
        .alias("binance_open_twap60"),
        pl.col("close_price")
        .filter(
            (pl.col("open_timestamp") >= pl.col("window_end") - pl.duration(seconds=60))
            & (pl.col("open_timestamp") < pl.col("window_end"))
        )
        .mean()
        .alias("binance_close_twap60"),
        pl.col("open_timestamp")
        .filter(
            (pl.col("open_timestamp") >= pl.col("window_start") - pl.duration(seconds=60))
            & (pl.col("open_timestamp") < pl.col("window_start"))
        )
        .n_unique()
        .alias("binance_open_prints"),
        pl.col("available_at")
        .filter(
            (pl.col("open_timestamp") >= pl.col("window_start") - pl.duration(seconds=60))
            & (pl.col("open_timestamp") < pl.col("window_start"))
        )
        .max()
        .alias("opening_binance_max_available_at"),
        pl.col("open_timestamp")
        .filter(
            (pl.col("open_timestamp") >= pl.col("window_end") - pl.duration(seconds=60))
            & (pl.col("open_timestamp") < pl.col("window_end"))
        )
        .n_unique()
        .alias("binance_close_prints"),
    )
    return (
        labels.join(aggregated, on="market_id", how="left", validate="1:1")
        .with_columns(
            (pl.col("binance_close_twap60") / pl.col("binance_open_twap60"))
            .log()
            .mul(10_000.0)
            .alias("binance_raw_margin_bps")
        )
        .with_columns(
            (pl.col("binance_raw_margin_bps") >= 0).alias("binance_raw_label_up"),
            (
                (pl.col("binance_open_prints") == 60)
                & (pl.col("binance_close_prints") == 60)
                & pl.col("binance_raw_margin_bps").is_finite()
            ).alias("binance_label_complete"),
        )
    )


def fit_binance_margin_correction(
    labels: pl.DataFrame,
    *,
    overlap_start: datetime,
    fit_end: datetime,
    validation_end: datetime,
    authentic_end: datetime,
    seed: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Fit only the five-minute margin residual, in chronological blocks."""

    available = labels.filter(
        pl.col("binance_label_complete")
        & pl.col("proxy_margin_bps").is_finite()
        & (pl.col("window_start") >= overlap_start)
    ).with_columns(
        (pl.col("proxy_margin_bps") - pl.col("binance_raw_margin_bps")).alias("margin_residual")
    )
    fit = available.filter(pl.col("window_start") < fit_end)
    features = ("binance_raw_margin_bps", "binance_open_twap60")
    if fit.height < 500:
        raise RuntimeError("insufficient chronological overlap for Binance margin correction")
    model = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=8.0,
        random_state=seed,
        early_stopping=False,
    )
    x_fit = fit.select(features).to_numpy()
    model.fit(x_fit, fit["margin_residual"].to_numpy())
    finite = labels.filter(
        pl.col("binance_label_complete")
        & pl.col("binance_raw_margin_bps").is_finite()
        & pl.col("binance_open_twap60").is_finite()
    )
    correction = model.predict(finite.select(features).to_numpy())
    corrected = finite.with_columns(
        pl.Series("binance_margin_correction_bps", correction),
        (pl.col("binance_raw_margin_bps") + pl.Series(correction)).alias(
            "binance_corrected_margin_bps"
        ),
    )
    residual = fit["margin_residual"].to_numpy() - model.predict(x_fit)
    residual_scale = max(float(np.quantile(np.abs(residual), 0.90)), 0.25)
    corrected = corrected.with_columns(
        (pl.col("binance_corrected_margin_bps") >= 0).alias("binance_corrected_label_up"),
        (pl.col("binance_corrected_margin_bps") - pl.col("binance_raw_margin_bps"))
        .abs()
        .alias("raw_corrected_synthetic_margin_disagreement_bps"),
        (
            pl.lit(residual_scale)
            / pl.max_horizontal(pl.col("binance_corrected_margin_bps").abs(), pl.lit(0.01))
        )
        .clip(0.0, 0.5)
        .alias("estimated_synthetic_label_error"),
    )

    def block_metrics(start: datetime, end: datetime, target: str) -> dict[str, Any]:
        block = corrected.filter(
            pl.col("window_start").is_between(start, end, closed="left")
        ).filter(pl.col(target).is_not_null())
        admitted = block.filter(pl.col("binance_corrected_margin_bps").abs() >= 1.0)
        if admitted.is_empty():
            return {"markets": 0, "agreement": None}
        return {
            "markets": admitted.height,
            "agreement": float((admitted["binance_corrected_label_up"] == admitted[target]).mean()),
            "margin_mae_bps": float(
                (admitted["binance_corrected_margin_bps"] - admitted["proxy_margin_bps"])
                .abs()
                .mean()
            )
            if target == "proxy_label_up"
            else None,
        }

    report = {
        "target": "five_minute_twap_margin_residual_bps",
        "features": list(features),
        "fit": {
            "start": overlap_start.isoformat(),
            "end": fit_end.isoformat(),
            "markets": fit.height,
        },
        "validation": block_metrics(fit_end, validation_end, "proxy_label_up"),
        "authentic_confirmation": block_metrics(
            validation_end, authentic_end, "authentic_label_up"
        ),
        "residual_p90_bps": residual_scale,
    }
    return corrected, report


def assign_priority_labels(
    labels: pl.DataFrame,
    corrected: pl.DataFrame,
    *,
    chainlink_start: datetime,
    authentic_start: datetime,
    official_start: datetime,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    joined = labels.join(
        corrected.select(
            "market_id",
            "binance_margin_correction_bps",
            "binance_corrected_margin_bps",
            "binance_corrected_label_up",
            "estimated_synthetic_label_error",
            "raw_corrected_synthetic_margin_disagreement_bps",
        ),
        on="market_id",
        how="left",
        validate="1:1",
    )
    abs_proxy = pl.col("proxy_margin_bps").abs()
    proxy_weight = (
        pl.when(abs_proxy < CHAINLINK_UNCERTAINTY_BPS)
        .then(0.0)
        .when(abs_proxy < 2 * CHAINLINK_UNCERTAINTY_BPS)
        .then(0.25)
        .when(abs_proxy < 3 * CHAINLINK_UNCERTAINTY_BPS)
        .then(0.50)
        .otherwise(0.75)
    )
    abs_binance = pl.col("binance_corrected_margin_bps").abs()
    binance_weight = (
        pl.when(abs_binance < 1.0).then(0.0).when(abs_binance < 2.0).then(0.25).otherwise(0.75)
    )
    valid_authentic = pl.col("authentic_label_up").is_not_null()
    valid_proxy = (
        (pl.col("window_start") >= chainlink_start)
        & pl.col("proxy_label_up").is_not_null()
        & (proxy_weight > 0)
    )
    valid_binance = pl.col("binance_corrected_label_up").is_not_null() & (binance_weight > 0)
    assigned = joined.with_columns(
        pl.when(valid_authentic)
        .then(pl.col("authentic_label_up"))
        .when(valid_proxy)
        .then(pl.col("proxy_label_up"))
        .when(valid_binance)
        .then(pl.col("binance_corrected_label_up"))
        .otherwise(None)
        .cast(pl.Int8)
        .alias("label_up"),
        pl.when(valid_authentic)
        .then(1.0)
        .when(valid_proxy)
        .then(proxy_weight)
        .when(valid_binance)
        .then(binance_weight)
        .otherwise(0.0)
        .alias("base_label_weight"),
        pl.when(valid_authentic & (pl.col("window_start") >= official_start))
        .then(pl.lit("authentic_official_twap60"))
        .when(valid_authentic & (pl.col("window_start") >= authentic_start))
        .then(pl.lit("authentic_counterfactual_twap60"))
        .when(valid_proxy)
        .then(pl.lit("chainlink_reconstructed_twap60"))
        .when(valid_binance)
        .then(pl.lit("binance_synthetic_twap60"))
        .otherwise(pl.lit("excluded"))
        .alias("label_source"),
        pl.when(valid_authentic)
        .then(pl.col("authentic_margin_bps"))
        .when(valid_proxy)
        .then(pl.col("proxy_margin_bps"))
        .when(valid_binance)
        .then(pl.col("binance_corrected_margin_bps"))
        .otherwise(None)
        .alias("target_margin_bps"),
        pl.when(valid_authentic)
        .then(pl.col("twap_open_price"))
        .when(valid_proxy)
        .then(pl.col("proxy_open_price"))
        .when(valid_binance)
        .then(pl.col("binance_open_twap60"))
        .otherwise(None)
        .alias("opening_twap60"),
    )
    eligible_binance = assigned.filter(pl.col("label_source") == "binance_synthetic_twap60")
    agreement = joined.filter(
        pl.col("authentic_label_up").is_not_null()
        & pl.col("binance_corrected_label_up").is_not_null()
        & (pl.col("binance_corrected_margin_bps").abs() >= 1.0)
    )
    weekly = (
        agreement.with_columns(pl.col("window_start").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(
            pl.len().alias("markets"),
            (pl.col("authentic_label_up") == pl.col("binance_corrected_label_up"))
            .mean()
            .alias("agreement"),
        )
        .sort("week")
    )
    fidelity = {
        "eligible_markets": eligible_binance.height,
        "authentic_overlap_markets": agreement.height,
        "authentic_agreement": float(
            (agreement["authentic_label_up"] == agreement["binance_corrected_label_up"]).mean()
        )
        if agreement.height
        else None,
        "weekly": weekly.to_dicts(),
    }
    return assigned, fidelity


def _build_core_features(
    raw: pl.DataFrame,
    boundaries: pl.DataFrame,
    oracle: pl.DataFrame,
) -> pl.DataFrame:
    frame = (
        raw.drop("opening_boundary")
        .join(boundaries.select("market_id", "opening_twap60"), on="market_id", how="inner")
        .rename({"opening_twap60": "opening_boundary"})
        .sort(["market_id", "seconds_elapsed"])
    )
    complete = (
        frame.group_by("market_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("seconds_elapsed").n_unique().alias("seconds"),
            pl.col("seconds_elapsed").min().alias("minimum"),
            pl.col("seconds_elapsed").max().alias("maximum"),
        )
        .filter(
            (pl.col("rows") == 300)
            & (pl.col("seconds") == 300)
            & (pl.col("minimum") == 0)
            & (pl.col("maximum") == 299)
        )
        .select("market_id")
    )
    frame = frame.join(complete, on="market_id", how="inner")
    rounds = prepare_causal_oracle_rounds(oracle.select(CORE_ORACLE_ROUND_SCHEMA.names))
    frame = derive_oracle_point_in_time_features(
        derive_core_point_in_time_features(attach_causal_oracle_rounds(frame, rounds))
    )
    return frame.filter(
        pl.col("seconds_elapsed").is_between(60, 179, closed="both")
        & ((pl.col("seconds_elapsed") - 60) % 5 == 0)
        & pl.col("oracle_model_eligible")
    ).drop(
        "official_outcome",
        "final_price",
        "btc_path_positive",
        "btc_path_crossed",
        "btc_last_path_cross_second",
        "btc_boundary_positive",
        "btc_boundary_crossed",
        "btc_last_boundary_cross_second",
        "oracle_price",
        "oracle_source_timestamp",
        "oracle_block_timestamp",
        "oracle_phase_id",
        "oracle_round_id",
        "oracle_block_number",
        "oracle_log_index",
        "oracle_window_open_price",
        "oracle_round_changed",
        strict=False,
    )


def _rolling_mean_at(
    times: np.ndarray, values: np.ndarray, points: np.ndarray, seconds: int
) -> np.ndarray:
    if len(times) == 0:
        return np.full(len(points), np.nan)
    return piecewise_average(times, values, points, window_seconds=float(seconds))


def _opening_chainlink_availability(
    refprice: pl.DataFrame,
    boundaries: np.ndarray,
    *,
    time_column: str,
) -> np.ndarray:
    """Return the last availability needed by each completed opening TWAP window."""

    path = (
        canonical_refprice_path(refprice)
        .sort(time_column)
        .unique(subset=[time_column], keep="last")
        .sort(time_column)
    )
    event_times = path[time_column].to_numpy().astype("datetime64[us]").astype(np.int64)
    available = path["provider_available_at"].to_numpy().astype("datetime64[us]").astype(np.int64)
    boundary_us = boundaries.astype("datetime64[us]").astype(np.int64)
    output = np.full(len(boundary_us), np.datetime64("NaT", "us"))
    for row, boundary in enumerate(boundary_us):
        first = max(0, int(np.searchsorted(event_times, boundary - 60_000_000, side="right")) - 1)
        stop = int(np.searchsorted(event_times, boundary, side="left"))
        if stop > first:
            output[row] = np.datetime64(int(available[first:stop].max()), "us")
    return output


def _maximum_available_at(*arrays: np.ndarray) -> np.ndarray:
    values = [array.astype("datetime64[us]").astype(np.int64) for array in arrays]
    matrix = np.vstack(values)
    missing = matrix == np.iinfo(np.int64).min
    maximum = matrix.max(axis=0)
    maximum[missing.any(axis=0)] = np.iinfo(np.int64).min
    return maximum.astype("datetime64[us]")


def _utc_datetime_series(name: str, values: np.ndarray) -> pl.Series:
    return pl.Series(name, values).dt.replace_time_zone("UTC")


def attach_causal_twap_features(
    frame: pl.DataFrame,
    labels: pl.DataFrame,
    refprice: pl.DataFrame,
    binance: pl.DataFrame,
) -> pl.DataFrame:
    """Attach completed-information rolling TWAP state to scheduled observations."""

    ordered = frame.sort(["window_start", "market_id", "observed_at"])
    points = ordered["observed_at"].to_numpy()
    chain = (
        canonical_refprice_path(refprice)
        .sort("provider_available_at")
        .filter(pl.col("source_timestamp") == pl.col("source_timestamp").cum_max())
    )
    chain_times = chain["provider_available_at"].to_numpy()
    chain_values = chain["price"].to_numpy()
    chain_indices = np.searchsorted(chain_times, points, side="right") - 1
    chain_latest = np.full(ordered.height, np.datetime64("NaT", "us"))
    chain_available = chain_indices >= 0
    chain_latest[chain_available] = chain_times[chain_indices[chain_available]].astype(
        "datetime64[us]"
    )
    chain30 = _rolling_mean_at(chain_times, chain_values, points, 30)
    chain60 = _rolling_mean_at(chain_times, chain_values, points, 60)
    chain_slopes: dict[tuple[int, int], np.ndarray] = {}
    for window in (30, 60):
        for horizon in (5, 15, 30, 60):
            prior = _rolling_mean_at(
                chain_times, chain_values, points - np.timedelta64(horizon, "s"), window
            )
            current = chain30 if window == 30 else chain60
            chain_slopes[(window, horizon)] = np.log(current / prior) * 10_000.0

    binance_by_market = {
        key[0]: part for key, part in binance.partition_by("market_id", as_dict=True).items()
    }
    b30 = np.full(ordered.height, np.nan)
    b60 = np.full(ordered.height, np.nan)
    b_slopes = {(w, h): np.full(ordered.height, np.nan) for w in (30, 60) for h in (5, 15, 30, 60)}
    completed30 = np.zeros(ordered.height)
    completed60 = np.zeros(ordered.height)
    max_gap = np.full(ordered.height, np.nan)
    age = np.full(ordered.height, np.nan)
    binance_latest = np.full(ordered.height, np.datetime64("NaT", "us"))
    for key, indices in ordered.with_row_index("_row").group_by("market_id", maintain_order=True):
        market_id = key[0] if isinstance(key, tuple) else key
        part = binance_by_market.get(market_id)
        if part is None or part.is_empty():
            continue
        part = (
            part.sort(["open_timestamp", "available_at", "artifact_id"])
            .unique(subset=["open_timestamp"], keep="last", maintain_order=True)
            .sort("available_at")
        )
        t = part["available_at"].to_numpy()
        v = part["close_price"].to_numpy()
        rows = indices["_row"].to_numpy()
        q = points[rows]
        b30[rows] = _rolling_mean_at(t, v, q, 30)
        b60[rows] = _rolling_mean_at(t, v, q, 60)
        for w in (30, 60):
            current = b30[rows] if w == 30 else b60[rows]
            for h in (5, 15, 30, 60):
                prior = _rolling_mean_at(t, v, q - np.timedelta64(h, "s"), w)
                b_slopes[(w, h)][rows] = np.log(current / prior) * 10_000.0
        ti = t.astype("datetime64[us]").astype(np.int64)
        qi = q.astype("datetime64[us]").astype(np.int64)
        ends = np.searchsorted(ti, qi, side="right")
        for local, end in enumerate(ends):
            start30 = np.searchsorted(ti, qi[local] - 30_000_000, side="right")
            start60 = np.searchsorted(ti, qi[local] - 60_000_000, side="right")
            completed30[rows[local]] = end - start30
            completed60[rows[local]] = end - start60
            sample = ti[start60:end]
            if len(sample):
                age[rows[local]] = (qi[local] - sample[-1]) / 1_000_000.0
                max_gap[rows[local]] = (
                    float(np.diff(sample).max() / 1_000_000.0) if len(sample) > 1 else math.inf
                )
            if end:
                binance_latest[rows[local]] = t[end - 1].astype("datetime64[us]")

    label_map = labels.select(
        "market_id",
        "opening_twap60",
        "proxy_open_price",
        "proxy_open_twap30",
        "opening_refprice",
        "binance_open_twap60",
        "binance_open_twap30",
        "opening_chainlink_max_available_at",
        "opening_binance_max_available_at",
    )
    result = ordered.join(label_map, on="market_id", how="left", validate="m:1")
    use_chain = (
        np.isfinite(chain30) & np.isfinite(chain60) & ordered["refprice_causal_eligible"].to_numpy()
    )
    twap30 = np.where(use_chain, chain30, b30)
    twap60 = np.where(use_chain, chain60, b60)
    chain_open_available = result["opening_chainlink_max_available_at"].to_numpy()
    binance_open_available = result["opening_binance_max_available_at"].to_numpy()
    chain_open_ready = (
        ~np.isnat(chain_open_available)
        & (chain_open_available <= points)
        & np.isfinite(result["proxy_open_twap30"].to_numpy())
        & np.isfinite(result["proxy_open_price"].to_numpy())
    )
    binance_open_ready = (
        ~np.isnat(binance_open_available)
        & (binance_open_available <= points)
        & np.isfinite(result["binance_open_twap30"].to_numpy())
        & np.isfinite(result["binance_open_twap60"].to_numpy())
    )
    use_chain_open = use_chain & chain_open_ready
    opening_twap30 = np.where(
        use_chain_open,
        result["proxy_open_twap30"].to_numpy(),
        np.where(binance_open_ready, result["binance_open_twap30"].to_numpy(), np.nan),
    )
    opening_twap60 = np.where(
        use_chain_open,
        result["proxy_open_price"].to_numpy(),
        np.where(binance_open_ready, result["binance_open_twap60"].to_numpy(), np.nan),
    )
    opening_twap_available = np.where(
        use_chain_open,
        chain_open_available,
        np.where(binance_open_ready, binance_open_available, np.datetime64("NaT", "us")),
    )
    current_twap_available = np.where(use_chain, chain_latest, binance_latest)
    twap_feature_available = _maximum_available_at(opening_twap_available, current_twap_available)
    disagreement_available = _maximum_available_at(chain_latest, binance_latest)
    disagreement_opening_available = _maximum_available_at(
        disagreement_available, chain_open_available, binance_open_available
    )
    disagreement_opening_available = np.where(
        disagreement_opening_available <= points,
        disagreement_opening_available,
        np.datetime64("NaT", "us"),
    )
    for horizon in (5, 15, 30, 60):
        result = result.with_columns(
            pl.Series(
                f"twap30_slope_{horizon}s_bps",
                np.where(use_chain, chain_slopes[(30, horizon)], b_slopes[(30, horizon)]),
            ),
            pl.Series(
                f"twap60_slope_{horizon}s_bps",
                np.where(use_chain, chain_slopes[(60, horizon)], b_slopes[(60, horizon)]),
            ),
        )
    result = (
        result.with_columns(
            pl.Series("chainlink_twap30", chain30),
            pl.Series("chainlink_twap60", chain60),
            pl.Series("binance_twap30", b30),
            pl.Series("binance_twap60", b60),
            pl.Series("current_twap30", twap30),
            pl.Series("current_twap60", twap60),
            pl.Series("twap_completed_prints_30", completed30),
            pl.Series("twap_completed_prints_60", completed60),
            pl.Series("twap_max_timestamp_gap_60s", max_gap),
            pl.Series("twap_source_age_seconds", age),
            _utc_datetime_series("chainlink_twap_max_available_at", chain_latest),
            _utc_datetime_series("binance_twap_max_available_at", binance_latest),
            _utc_datetime_series("twap_feature_max_available_at", twap_feature_available),
            _utc_datetime_series("disagreement_feature_max_available_at", disagreement_available),
            _utc_datetime_series(
                "disagreement_opening_feature_max_available_at",
                disagreement_opening_available,
            ),
            _utc_datetime_series("core_source_max_available_at", points),
            pl.Series("opening_twap30", opening_twap30),
            pl.Series("opening_twap60", opening_twap60),
        )
        .with_columns(
            pl.when(pl.col("opening_binance_max_available_at") <= pl.col("observed_at"))
            .then(pl.col("binance_open_twap30").fill_null(pl.col("binance_open_twap60")))
            .otherwise(None)
            .alias("opening_binance_twap30"),
            pl.when(pl.col("opening_binance_max_available_at") <= pl.col("observed_at"))
            .then(pl.col("binance_open_twap60"))
            .otherwise(None)
            .alias("opening_binance_twap60"),
            pl.when(
                (pl.col("opening_binance_max_available_at") <= pl.col("observed_at"))
                & (pl.col("opening_chainlink_max_available_at") <= pl.col("observed_at"))
            )
            .then(pl.col("binance_open_twap60") / pl.col("proxy_open_price"))
            .otherwise(None)
            .log()
            .mul(10_000)
            .alias("opening_binance_chainlink_basis_bps"),
            pl.when(pl.col("opening_chainlink_max_available_at") <= pl.col("observed_at"))
            .then(pl.col("opening_refprice"))
            .otherwise(None)
            .alias("opening_refprice"),
        )
        .with_columns(
            (pl.col("current_twap30") / pl.col("opening_twap30"))
            .log()
            .mul(10_000)
            .alias("twap30_change_from_open_bps"),
            (pl.col("current_twap60") / pl.col("opening_twap60"))
            .log()
            .mul(10_000)
            .alias("twap60_change_from_open_bps"),
            (pl.col("current_twap30") / pl.col("opening_twap60"))
            .log()
            .mul(10_000)
            .alias("twap30_gap_to_opening_twap60_bps"),
            (pl.col("current_twap60") / pl.col("opening_twap60"))
            .log()
            .mul(10_000)
            .alias("twap60_gap_to_opening_twap60_bps"),
            (pl.col("current_twap30") / pl.col("current_twap60"))
            .log()
            .mul(10_000)
            .alias("twap30_minus_twap60_bps"),
            (pl.col("chainlink_twap30") / pl.col("binance_twap30"))
            .log()
            .mul(-10_000)
            .alias("binance_twap30_minus_chainlink_twap30_bps"),
            (pl.col("chainlink_twap60") / pl.col("binance_twap60"))
            .log()
            .mul(-10_000)
            .alias("binance_twap60_minus_chainlink_twap60_bps"),
            (pl.col("chainlink_twap60") / pl.col("binance_twap60"))
            .log()
            .mul(10_000)
            .alias("binance_chainlink_basis_bps"),
        )
        .with_columns(
            pl.col("chainlink_ref_boundary_gap_bps").alias(
                "current_refprice_gap_to_opening_twap60_bps"
            ),
            (
                pl.col("chainlink_ref_boundary_gap_bps")
                - pl.col("twap30_gap_to_opening_twap60_bps")
            ).alias("refprice_minus_twap30_bps"),
            (
                pl.col("chainlink_ref_boundary_gap_bps")
                - pl.col("twap60_gap_to_opening_twap60_bps")
            ).alias("refprice_minus_twap60_bps"),
            (
                pl.col("twap30_change_from_open_bps").sign()
                * pl.col("twap60_change_from_open_bps").sign()
            ).alias("twap_directional_agreement"),
            (
                pl.col("chainlink_ref_boundary_gap_bps").sign()
                * pl.col("twap60_change_from_open_bps").sign()
            ).alias("refprice_twap_directional_agreement"),
            (pl.col("twap30_slope_5s_bps") - pl.col("twap30_slope_15s_bps") / 3.0).alias(
                "twap_acceleration_bps"
            ),
            (
                pl.col("twap30_minus_twap60_bps")
                - pl.col("twap30_slope_5s_bps")
                + pl.col("twap60_slope_5s_bps")
            ).alias("twap_convergence_velocity_bps"),
            (pl.col("chainlink_ref_return_5s_bps") - pl.col("twap30_slope_5s_bps")).alias(
                "refprice_pressure_bps"
            ),
            pl.col("btc_boundary_cross_count").cast(pl.Float64).alias("twap_boundary_cross_count"),
            pl.col("btc_seconds_since_boundary_cross")
            .cast(pl.Float64)
            .alias("twap_seconds_since_boundary_cross"),
            pl.col("btc_path_efficiency_60s").alias("twap_path_efficiency"),
            pl.col("btc_reversal_5_vs_30").cast(pl.Float64).alias("twap_reversal_state"),
            pl.col("chainlink_ref_realized_volatility_60s_bps").alias(
                "twap_realized_volatility_bps"
            ),
            (
                pl.col("twap60_gap_to_opening_twap60_bps")
                / pl.max_horizontal(
                    pl.col("chainlink_ref_realized_volatility_60s_bps"), pl.lit(0.01)
                )
            ).alias("settlement_margin_volatility_z"),
            (
                pl.col("binance_chainlink_basis_bps")
                - pl.col("opening_binance_chainlink_basis_bps")
            ).alias("binance_chainlink_basis_velocity_5s_bps"),
        )
    )
    result = result.with_columns(
        pl.when(pl.col("twap_feature_max_available_at").is_not_null())
        .then(pl.col(name))
        .otherwise(None)
        .alias(name)
        for name in sorted(_TWAP_INFERENCE_FEATURES)
        if name in result.columns
    ).with_columns(
        pl.when(
            pl.col(
                "disagreement_opening_feature_max_available_at"
                if name == "binance_chainlink_basis_velocity_5s_bps"
                else "disagreement_feature_max_available_at"
            ).is_not_null()
        )
        .then(pl.col(name))
        .otherwise(None)
        .alias(name)
        for name in BINANCE_DISAGREEMENT_FEATURES
    )
    causal_availability_audit(result)
    return result


def build_counterfactual_frame(
    paths: CounterfactualDataPaths,
    *,
    chainlink_start: datetime,
    authentic_start: datetime,
    official_start: datetime,
    correction_fit_end: datetime,
    correction_validation_end: datetime,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, ProxyConvention, dict[str, Any]]:
    labels_raw = load_source_group(paths.base, "labels")
    refprice = load_source_group(paths.base, "refprice")
    binance = load_binance(paths)
    convention, _ = select_proxy_convention(
        labels_raw.filter(pl.col("window_start") >= authentic_start),
        refprice,
        calibration_start=authentic_start,
        calibration_end=authentic_start + timedelta(days=6),
    )
    proxy = construct_proxy_labels(labels_raw, refprice, time_column=convention.time_column)
    canonical = (
        canonical_refprice_path(refprice)
        .sort(convention.time_column)
        .unique(subset=[convention.time_column], keep="last")
        .sort(convention.time_column)
    )
    canonical_times = canonical[convention.time_column].to_numpy()
    canonical_prices = canonical["price"].to_numpy()
    boundary_times = proxy["window_start"].to_numpy()
    opening_indices = np.searchsorted(canonical_times, boundary_times, side="left") - 1
    opening_refprice = np.full(proxy.height, np.nan)
    valid_opening = opening_indices >= 0
    opening_refprice[valid_opening] = canonical_prices[opening_indices[valid_opening]]
    proxy = proxy.with_columns(
        pl.Series("opening_refprice", opening_refprice),
        _utc_datetime_series(
            "opening_chainlink_max_available_at",
            _opening_chainlink_availability(
                refprice,
                boundary_times,
                time_column=convention.time_column,
            ),
        ),
        pl.Series(
            "proxy_open_twap30",
            piecewise_average(
                canonical_times, canonical_prices, boundary_times, window_seconds=30.0
            ),
        ),
    )
    authentic = authentic_labels(labels_raw).select(
        "market_id", "authentic_label_up", "authentic_margin_bps"
    )
    labels = construct_binance_labels(proxy, binance).join(
        authentic, on="market_id", how="left", validate="1:1"
    )
    corrected, correction_report = fit_binance_margin_correction(
        labels,
        overlap_start=chainlink_start,
        fit_end=correction_fit_end,
        validation_end=correction_validation_end,
        authentic_end=official_start,
        seed=seed,
    )
    labels, binance_fidelity = assign_priority_labels(
        labels,
        corrected,
        chainlink_start=chainlink_start,
        authentic_start=authentic_start,
        official_start=official_start,
    )
    boundaries = labels.filter(pl.col("label_up").is_not_null()).select(
        "market_id", "opening_twap60"
    )
    core = _build_core_features(
        load_source_group(paths.base, "core_current"),
        boundaries,
        load_source_group(paths.base, "oracle"),
    )
    selected = labels.filter(
        pl.col("label_up").is_not_null() & (pl.col("base_label_weight") > 0)
    ).select(
        "market_id",
        "label_up",
        "base_label_weight",
        "label_source",
        "target_margin_bps",
        "authentic_label_up",
        "authentic_margin_bps",
        "proxy_label_up",
        "proxy_margin_bps",
        "binance_raw_margin_bps",
        "binance_corrected_margin_bps",
        "binance_margin_correction_bps",
        "estimated_synthetic_label_error",
        "raw_corrected_synthetic_margin_disagreement_bps",
        "opening_twap60",
        "proxy_open_price",
        "binance_open_twap60",
        "official_outcome",
    )
    frame = core.drop("label_up", strict=False).join(
        selected, on="market_id", how="inner", validate="m:1"
    )
    frame = ensure_oracle_eligibility_compatibility(frame)
    execution = load_source_group(paths.base, "execution").filter(
        pl.col("window_start") >= authentic_start
    )
    frame = attach_execution(frame, execution)
    frame = attach_candle_context(
        frame,
        load_source_group(paths.base, "candles").unique(subset=["close_timestamp"], keep="last"),
    )
    frame = attach_causal_refprice_features(frame, refprice)
    refprice_columns = [name for name in frame.columns if name.startswith("chainlink_ref_")]
    frame = frame.with_columns(
        pl.when(pl.col("refprice_causal_eligible")).then(pl.col(name)).otherwise(None).alias(name)
        for name in refprice_columns
    )
    frame = attach_causal_twap_features(frame, labels, refprice, binance).sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    current = labels.filter(
        (pl.col("window_start") >= official_start) & pl.col("authentic_label_up").is_not_null()
    )
    official_disagreement = current.filter(
        pl.col("authentic_label_up") != (pl.col("official_outcome") == "up")
    )
    if official_disagreement.height:
        raise RuntimeError("authentic TWAP labels disagree with current official outcomes")
    manifest = {
        "schema_version": "btc-counterfactual-twap-state-frame-v2-causal",
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "label_markets": labels.filter(pl.col("label_up").is_not_null()).height,
        "label_coverage": labels.group_by("label_source").len().sort("label_source").to_dicts(),
        "label_coverage_by_date_and_source": labels.filter(pl.col("label_up").is_not_null())
        .with_columns(pl.col("window_start").dt.date().alias("date"))
        .group_by("date", "label_source")
        .len()
        .sort("date", "label_source")
        .to_dicts(),
        "binance_margin_correction": correction_report,
        "binance_fidelity": binance_fidelity,
        "chainlink_uncertainty_bps": CHAINLINK_UNCERTAINTY_BPS,
        "completed_information_semantics": "[T-W,T)",
        "causal_availability_audit": causal_availability_audit(frame),
        "database_mutations": False,
        "new_tables": False,
        "new_sources": False,
    }
    return frame, labels, convention, manifest
