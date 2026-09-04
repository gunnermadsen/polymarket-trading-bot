"""Causal Binance context features and residual adjustment for latent TWAP."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import norm

from .chainlink_oi_features import (
    BINANCE_OI_FEATURES,
    _derive_open_interest_source_features,
    _prepare_open_interest,
)
from .core_extract import (
    configure_read_only_connection,
    database_connection,
    file_sha256,
    write_json_exclusive,
)
from .latent_twap_state_space import ProbabilityCalibrator
from .spot_l2_chainlink_extract import (
    L2_SOURCE_SCHEMA,
    L2_SOURCE_SCHEMA_VERSION,
    _l2_summary,
    _load_or_extract_partition,
)
from .spot_l2_chainlink_features import (
    L2_CAUSAL_AUDIT_COLUMNS,
    L2_FEATURES,
    join_qualified_l2,
)

CONTEXT_CACHE_SCHEMA_VERSION = "btc-binance-latent-twap-context-source-cache-v1"
FEATURE_FRAME_SCHEMA_VERSION = "btc-binance-latent-twap-context-frame-v1"
RESIDUAL_MODEL_SCHEMA_VERSION = "btc-binance-latent-twap-residual-adjustment-v1"

KLINE_HORIZONS_SECONDS = (5, 15, 30)
KLINE_FEATURES = tuple(
    [f"binance_kline_return_{seconds}s_bps" for seconds in KLINE_HORIZONS_SECONDS]
    + [
        f"binance_kline_realized_volatility_{seconds}s_bps"
        for seconds in KLINE_HORIZONS_SECONDS
    ]
    + [f"binance_kline_signed_flow_{seconds}s" for seconds in KLINE_HORIZONS_SECONDS]
    + [f"binance_kline_log_quote_volume_{seconds}s" for seconds in KLINE_HORIZONS_SECONDS]
    + [f"binance_kline_range_{seconds}s_bps" for seconds in KLINE_HORIZONS_SECONDS]
    + ["binance_kline_path_from_window_open_bps"]
)

OI_FEATURES = tuple(BINANCE_OI_FEATURES)
RESIDUAL_BASE_FEATURES = (
    "expected_margin_bps",
    "margin_velocity_bps_per_step",
    "process_uncertainty_log1p",
    "reversal_probability",
    "stable_regime_probability",
    "trending_regime_probability",
    "reversal_regime_probability",
    "seconds_elapsed_scaled",
)

OPEN_INTEREST_SCHEMA = pa.schema(
    [
        ("source_timestamp", pa.timestamp("us", tz="UTC")),
        ("period_seconds", pa.int32()),
        ("sum_open_interest", pa.float64()),
        ("sum_open_interest_value", pa.float64()),
    ]
)
OPEN_INTEREST_SCHEMA_VERSION = "binance-btcusdt-five-minute-open-interest-v1"


@dataclass(frozen=True)
class ContextSourceCache:
    open_interest: Path
    l2: tuple[Path, ...]
    manifest: Path


@dataclass(frozen=True)
class ResidualAdjustmentParameters:
    schema_version: str
    candidate_name: str
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    residual_variance_bps2: float
    ridge_alpha: float
    fit_markets: int
    fit_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResidualAdjustmentParameters:
        values = dict(payload)
        for name in ("feature_names", "feature_means", "feature_scales", "coefficients"):
            values[name] = tuple(values[name])
        return cls(**values)


def load_or_extract_context_sources(config: Any) -> tuple[ContextSourceCache, dict[str, Any]]:
    """Build or verify resumable immutable OI and L2 evidence from existing SQL."""

    root = Path(config.context_cache) / "source"
    manifest_path = root / "manifest.json"
    oi_sql = Path(config.open_interest_source_sql).read_text()
    l2_sql = Path(config.l2_source_sql).read_text()
    contract = {
        "schema_version": CONTEXT_CACHE_SCHEMA_VERSION,
        "range_start": config.historical_start.isoformat(),
        "range_end": config.freeze_at.isoformat(),
        "l2_extract_end": config.l2_extract_end.isoformat(),
        "read_only": True,
        "database_mutations": False,
        "new_data_sources": False,
        "open_interest_query_sha256": hashlib.sha256(oi_sql.encode()).hexdigest(),
        "l2_query_sha256": hashlib.sha256(l2_sql.encode()).hexdigest(),
        "source_relations": {
            "open_interest": "market_data.binance_futures_btcusdt_open_interest",
            "l2": "polymarket.binance_spot_btcusdt_l2_training_features",
        },
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("contract") != contract:
            raise RuntimeError("Binance context source contract changed")
        cache = _context_cache_from_manifest(root, manifest)
        return cache, manifest

    root.mkdir(parents=True, exist_ok=True)
    connection = database_connection()
    configure_read_only_connection(connection)
    try:
        oi_start = config.historical_start - timedelta(
            minutes=int(config.raw["binance"]["open_interest_history_minutes"])
        )
        oi_record = _load_or_extract_partition(
            connection,
            source="open_interest",
            output_dir=root / "open-interest",
            destination_name="2026-06-07_2026-08-28.parquet",
            batch_start=oi_start,
            batch_end=config.freeze_at,
            query=oi_sql,
            parameters={
                "range_start": config.historical_start,
                "range_end": config.freeze_at,
                "history_minutes": int(
                    config.raw["binance"]["open_interest_history_minutes"]
                ),
                "open_interest_symbol": "BTCUSDT",
            },
            schema=OPEN_INTEREST_SCHEMA,
            schema_version=OPEN_INTEREST_SCHEMA_VERSION,
            summary=lambda path: _open_interest_summary(path, oi_start, config.freeze_at),
        )
        l2_records: list[dict[str, Any]] = []
        for start, end in _daily_ranges(config.historical_start, config.l2_extract_end):
            l2_records.append(
                _load_or_extract_partition(
                    connection,
                    source="l2",
                    output_dir=root / "l2",
                    destination_name=f"{start.date().isoformat()}.parquet",
                    batch_start=start,
                    batch_end=end,
                    query=l2_sql,
                    parameters={"batch_start": start, "batch_end": end},
                    schema=L2_SOURCE_SCHEMA,
                    schema_version=L2_SOURCE_SCHEMA_VERSION,
                    summary=lambda path, start=start, end=end: _l2_summary(path, start, end),
                )
            )
    finally:
        connection.close()

    manifest = {
        "schema_version": CONTEXT_CACHE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "immutable": True,
        "paper_only": True,
        "contract": contract,
        "sources": {
            "open_interest": {"partitions": [oi_record]},
            "l2": {"partitions": l2_records},
        },
    }
    write_json_exclusive(manifest_path, manifest)
    return _context_cache_from_manifest(root, manifest), manifest


def derive_kline_checkpoint_features(source: pl.DataFrame) -> pl.DataFrame:
    """Derive fixed causal Binance kline context across contiguous one-second rows."""

    required = {
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "btc_open",
        "btc_high",
        "btc_low",
        "btc_close",
        "btc_quote_volume",
        "btc_taker_buy_quote_volume",
    }
    _require_columns(source, required, "Binance one-second kline source")
    ordered = source.select(*sorted(required)).sort("observed_at")
    duplicate = ordered.group_by("observed_at").len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise RuntimeError("Binance kline source contains duplicate observation timestamps")
    ordered = ordered.with_columns(
        pl.col("btc_close").log().diff().alias("_log_return_1s"),
        (
            2.0 * pl.col("btc_taker_buy_quote_volume") - pl.col("btc_quote_volume")
        ).alias("_signed_quote_volume"),
        pl.first("btc_open").over("market_id").alias("_window_open"),
    )
    expressions: list[pl.Expr] = []
    for seconds in KLINE_HORIZONS_SECONDS:
        exact = (
            pl.col("observed_at") - pl.col("observed_at").shift(seconds)
            == pl.duration(seconds=seconds)
        )
        quote = pl.col("btc_quote_volume").rolling_sum(
            window_size=seconds, min_samples=seconds
        )
        signed = pl.col("_signed_quote_volume").rolling_sum(
            window_size=seconds, min_samples=seconds
        )
        expressions.extend(
            (
                pl.when(exact)
                .then(
                    (pl.col("btc_close") / pl.col("btc_close").shift(seconds))
                    .log()
                    .mul(10_000.0)
                )
                .alias(f"binance_kline_return_{seconds}s_bps"),
                pl.when(exact)
                .then(
                    pl.col("_log_return_1s")
                    .rolling_std(window_size=seconds, min_samples=seconds)
                    .mul(10_000.0)
                )
                .alias(f"binance_kline_realized_volatility_{seconds}s_bps"),
                pl.when(exact)
                .then(signed / (quote + 1e-9))
                .alias(f"binance_kline_signed_flow_{seconds}s"),
                pl.when(exact)
                .then(quote.log1p())
                .alias(f"binance_kline_log_quote_volume_{seconds}s"),
                pl.when(exact)
                .then(
                    (
                        pl.col("btc_high").rolling_max(
                            window_size=seconds, min_samples=seconds
                        )
                        / pl.col("btc_low").rolling_min(
                            window_size=seconds, min_samples=seconds
                        )
                    )
                    .log()
                    .mul(10_000.0)
                )
                .alias(f"binance_kline_range_{seconds}s_bps"),
            )
        )
    return (
        ordered.with_columns(
            *expressions,
            (pl.col("btc_close") / pl.col("_window_open"))
            .log()
            .mul(10_000.0)
            .alias("binance_kline_path_from_window_open_bps"),
            pl.col("observed_at").alias("binance_kline_available_at"),
        )
        .select(
            "market_id",
            "window_start",
            "observed_at",
            "seconds_elapsed",
            "btc_close",
            "binance_kline_available_at",
            *KLINE_FEATURES,
        )
        .filter(pl.col("seconds_elapsed").is_between(30, 150, closed="both"))
    )


def attach_open_interest_features(
    frame: pl.DataFrame,
    source: pl.DataFrame,
    *,
    maximum_age_seconds: int = 300,
) -> pl.DataFrame:
    """Attach strictly prior OI state while preserving missing rows and audit timing."""

    interest = _derive_open_interest_source_features(_prepare_open_interest(source))
    source_features = tuple(
        name for name in OI_FEATURES if not name.startswith("binance_oi_path_agreement_")
    )
    original = tuple(frame.columns)
    joined = (
        frame.with_row_index("_context_row")
        .sort("observed_at")
        .join_asof(
            interest.select("available_at", "source_timestamp", *source_features),
            left_on="observed_at",
            right_on="available_at",
            strategy="backward",
            allow_exact_matches=False,
        )
    )
    age_us = (pl.col("observed_at") - pl.col("source_timestamp")).dt.total_microseconds()
    eligible = (
        pl.col("available_at").is_not_null()
        & (pl.col("available_at") < pl.col("observed_at"))
        & pl.col("source_timestamp").is_not_null()
        & (pl.col("source_timestamp") < pl.col("observed_at"))
        & (age_us > 0)
        & (age_us <= maximum_age_seconds * 1_000_000)
    )
    joined = joined.with_columns(
        *(
            pl.when(eligible).then(pl.col(name)).otherwise(None).alias(name)
            for name in source_features
        ),
        pl.when(eligible).then(pl.col("available_at")).alias("binance_oi_available_at"),
        pl.when(eligible).then(pl.col("source_timestamp")).alias("binance_oi_source_timestamp"),
        pl.when(eligible).then(age_us / 1_000_000.0).alias("binance_oi_age_seconds"),
    ).with_columns(
        (
            pl.col("binance_oi_change_15m_bps").sign()
            * pl.col("binance_kline_path_from_window_open_bps").sign()
        ).alias("binance_oi_path_agreement_15m"),
        (
            pl.col("binance_oi_change_60m_bps").sign()
            * pl.col("binance_kline_path_from_window_open_bps").sign()
        ).alias("binance_oi_path_agreement_60m"),
    )
    return joined.sort("_context_row").select(
        *original,
        *OI_FEATURES,
        "binance_oi_available_at",
        "binance_oi_source_timestamp",
        "binance_oi_age_seconds",
    )


def attach_l2_features(frame: pl.DataFrame, l2_files: Iterable[Path]) -> pl.DataFrame:
    """Attach the repository-qualified L2 contract without filling absent checkpoints."""

    files = tuple(l2_files)
    if not files:
        return frame.with_columns(
            *[pl.lit(None, dtype=pl.Float64).alias(name) for name in L2_FEATURES],
            *[
                pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias(name)
                for name in ("spot_l2_source_event_timestamp", "spot_l2_available_at")
            ],
            *[
                pl.lit(None, dtype=pl.Float64).alias(name)
                for name in ("spot_l2_availability_age_seconds", "spot_l2_state_age_seconds")
            ],
        )
    source = pl.scan_parquet(files).collect()
    keys = ("market_id", "window_start", "observed_at", "seconds_elapsed")
    qualified = join_qualified_l2(frame.select(*keys, "btc_close"), source).select(
        *keys, *L2_FEATURES, *L2_CAUSAL_AUDIT_COLUMNS
    )
    return frame.join(qualified, on=list(keys), how="left", validate="1:1")


def fit_residual_adjustment(
    scored: pl.DataFrame,
    *,
    candidate_name: str,
    context_features: tuple[str, ...],
    ridge_alpha: float,
) -> ResidualAdjustmentParameters:
    """Fit a market-balanced ridge adjustment to the latent final-margin residual."""

    feature_names = (*RESIDUAL_BASE_FEATURES, *context_features)
    prepared = _residual_design_frame(scored, context_features)
    if prepared.is_empty():
        raise RuntimeError(f"no eligible residual fitting rows for {candidate_name}")
    matrix = prepared.select(feature_names).to_numpy().astype(float)
    target = (
        prepared["target_margin_bps"].to_numpy().astype(float)
        - prepared["expected_margin_bps"].to_numpy().astype(float)
    )
    weights = _market_balanced_row_weights(prepared)
    means = np.sum(matrix * weights[:, None], axis=0)
    centered = matrix - means
    variance = np.sum(centered * centered * weights[:, None], axis=0)
    scales = np.sqrt(np.maximum(variance, 1e-12))
    normalized = centered / scales
    target_mean = float(np.sum(target * weights))
    centered_target = target - target_mean
    root_weight = np.sqrt(weights)
    weighted_matrix = normalized * root_weight[:, None]
    weighted_target = centered_target * root_weight
    penalty = np.eye(weighted_matrix.shape[1]) * float(ridge_alpha)
    coefficients = np.linalg.solve(
        weighted_matrix.T @ weighted_matrix + penalty,
        weighted_matrix.T @ weighted_target,
    )
    prediction = target_mean + normalized @ coefficients
    residual = target - prediction
    residual_variance = max(float(np.sum(weights * residual * residual)), 1e-6)
    return ResidualAdjustmentParameters(
        schema_version=RESIDUAL_MODEL_SCHEMA_VERSION,
        candidate_name=candidate_name,
        feature_names=feature_names,
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients),
        intercept=target_mean,
        residual_variance_bps2=residual_variance,
        ridge_alpha=float(ridge_alpha),
        fit_markets=prepared["market_id"].n_unique(),
        fit_rows=prepared.height,
    )


def apply_residual_adjustment(
    scored: pl.DataFrame,
    parameters: ResidualAdjustmentParameters,
    *,
    context_features: tuple[str, ...],
    calibrator: ProbabilityCalibrator | None = None,
) -> pl.DataFrame:
    """Apply a causal residual correction and recompute distributional outputs."""

    if tuple(parameters.feature_names) != (*RESIDUAL_BASE_FEATURES, *context_features):
        raise RuntimeError("residual model feature contract changed")
    prepared = _residual_design_frame(scored, context_features)
    if prepared.is_empty():
        return prepared
    matrix = prepared.select(parameters.feature_names).to_numpy().astype(float)
    normalized = (
        matrix - np.asarray(parameters.feature_means)[None, :]
    ) / np.asarray(parameters.feature_scales)[None, :]
    adjustment = parameters.intercept + normalized @ np.asarray(parameters.coefficients)
    expected = prepared["expected_margin_bps"].to_numpy().astype(float) + adjustment
    variance = (
        prepared["process_uncertainty_bps2"].to_numpy().astype(float)
        + parameters.residual_variance_bps2
    )
    sigma = np.sqrt(np.maximum(variance, 1e-9))
    probability = norm.cdf(expected / sigma)
    if calibrator is not None:
        calibrated = calibrator.transform(probability)
    else:
        calibrated = probability
    return prepared.with_columns(
        pl.Series("residual_adjustment_bps", adjustment),
        pl.Series("residual_variance_bps2", np.full(prepared.height, parameters.residual_variance_bps2)),
        pl.Series("expected_margin_bps", expected),
        pl.Series("margin_p05_bps", expected + norm.ppf(0.05) * sigma),
        pl.Series("margin_p50_bps", expected),
        pl.Series("margin_p95_bps", expected + norm.ppf(0.95) * sigma),
        pl.Series("process_uncertainty_bps2", variance),
        pl.Series("raw_probability_up", probability),
        pl.Series("probability_up", calibrated),
    )


def context_coverage(frame: pl.DataFrame, config: Any) -> dict[str, Any]:
    families = {
        "kline": KLINE_FEATURES,
        "open_interest": OI_FEATURES,
        "kline_open_interest": (*KLINE_FEATURES, *OI_FEATURES),
        "l2": tuple(L2_FEATURES),
    }
    periods = {
        "historical_fit": (config.historical_start, config.historical_end),
        "calibration": (config.calibration_start, config.calibration_end),
        "official_development": (config.development_start, config.development_end),
        "prospective": (config.prospective_start, None),
    }
    output: dict[str, Any] = {}
    for family, features in families.items():
        output[family] = {}
        eligible = pl.all_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite() for name in features
        )
        for period_name, (start, end) in periods.items():
            block = frame.filter(pl.col("window_start") >= start)
            if end is not None:
                block = block.filter(pl.col("window_start") < end)
            qualified = block.filter(eligible)
            total_markets = block["market_id"].n_unique() if block.height else 0
            qualified_markets = qualified["market_id"].n_unique() if qualified.height else 0
            output[family][period_name] = {
                "rows": qualified.height,
                "markets": qualified_markets,
                "scheduled_markets": total_markets,
                "market_coverage": qualified_markets / max(total_markets, 1),
            }
    return output


def verify_context_causality(frame: pl.DataFrame) -> dict[str, Any]:
    checks = {
        "kline_at_or_before_checkpoint": frame.filter(
            pl.col("binance_kline_available_at").is_not_null()
            & (pl.col("binance_kline_available_at") > pl.col("observed_at"))
        ).is_empty(),
        "oi_strictly_before_checkpoint": frame.filter(
            pl.col("binance_oi_available_at").is_not_null()
            & (pl.col("binance_oi_available_at") >= pl.col("observed_at"))
        ).is_empty(),
        "oi_age_bounded": frame.filter(
            pl.col("binance_oi_age_seconds").is_not_null()
            & (
                (pl.col("binance_oi_age_seconds") <= 0)
                | (pl.col("binance_oi_age_seconds") > 300)
            )
        ).is_empty(),
        "l2_strictly_before_checkpoint": frame.filter(
            pl.col("spot_l2_available_at").is_not_null()
            & (pl.col("spot_l2_available_at") >= pl.col("observed_at"))
        ).is_empty(),
        "l2_age_bounded": frame.filter(
            pl.col("spot_l2_availability_age_seconds").is_not_null()
            & (
                (pl.col("spot_l2_availability_age_seconds") <= 0)
                | (pl.col("spot_l2_availability_age_seconds") > 2)
            )
        ).is_empty(),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _residual_design_frame(
    scored: pl.DataFrame, context_features: tuple[str, ...]
) -> pl.DataFrame:
    required = {
        "market_id",
        "label_up",
        "target_margin_bps",
        "seconds_elapsed",
        "expected_margin_bps",
        "margin_velocity_bps_per_step",
        "process_uncertainty_bps2",
        "reversal_probability",
        "stable_regime_probability",
        "trending_regime_probability",
        "reversal_regime_probability",
        *context_features,
    }
    _require_columns(scored, required, "latent residual frame")
    prepared = scored.with_columns(
        pl.col("process_uncertainty_bps2").clip(lower_bound=0.0).log1p().alias(
            "process_uncertainty_log1p"
        ),
        (pl.col("seconds_elapsed") / 150.0).alias("seconds_elapsed_scaled"),
    )
    names = (*RESIDUAL_BASE_FEATURES, *context_features)
    return prepared.filter(
        pl.all_horizontal(pl.col(name).is_not_null() & pl.col(name).is_finite() for name in names)
        & pl.col("target_margin_bps").is_not_null()
        & pl.col("target_margin_bps").is_finite()
    )


def _market_balanced_row_weights(frame: pl.DataFrame) -> np.ndarray:
    markets = frame.select("market_id", "label_up").unique(subset=["market_id"])
    class_counts = {
        int(row["label_up"]): count
        for row in markets.group_by("label_up").len().iter_rows(named=True)
        for count in (int(row["len"]),)
    }
    row_counts = {
        str(row["market_id"]): int(row["len"])
        for row in frame.group_by("market_id").len().iter_rows(named=True)
    }
    weights = np.asarray(
        [
            0.5
            / max(class_counts[int(label)], 1)
            / max(row_counts[str(market_id)], 1)
            for market_id, label in frame.select("market_id", "label_up").iter_rows()
        ],
        dtype=float,
    )
    return weights / weights.sum()


def _context_cache_from_manifest(root: Path, manifest: dict[str, Any]) -> ContextSourceCache:
    records = manifest["sources"]
    oi_record = records["open_interest"]["partitions"][0]
    oi_path = root / "open-interest" / oi_record["path"]
    if not oi_path.is_file() or file_sha256(oi_path) != oi_record["sha256"]:
        raise RuntimeError("immutable open-interest context partition changed")
    l2_paths: list[Path] = []
    for record in records["l2"]["partitions"]:
        path = root / "l2" / record["path"]
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"immutable L2 context partition changed: {record['path']}")
        l2_paths.append(path)
    return ContextSourceCache(oi_path, tuple(l2_paths), root / "manifest.json")


def _open_interest_summary(path: Path, start: datetime, end: datetime) -> dict[str, Any]:
    table = pq.read_table(path)
    if table.schema != OPEN_INTEREST_SCHEMA:
        raise RuntimeError("open-interest parquet schema changed")
    for name in ("sum_open_interest", "sum_open_interest_value"):
        values = table[name]
        if values.null_count or pc.sum(pc.invert(pc.is_finite(values))).as_py():
            raise RuntimeError(f"open-interest partition contains invalid {name}")
    rows = table.select(["source_timestamp", "period_seconds"]).to_pylist()
    if any(
        row["period_seconds"] != 300
        or not start <= row["source_timestamp"] < end
        for row in rows
    ):
        raise RuntimeError("open-interest partition violates its bounded source contract")
    timestamps = [row["source_timestamp"] for row in rows]
    return {
        "rows": table.num_rows,
        "minimum_source_timestamp": min(timestamps).isoformat() if timestamps else None,
        "maximum_source_timestamp": max(timestamps).isoformat() if timestamps else None,
    }


def _daily_ranges(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    cursor = start
    while cursor < end:
        following = min(cursor + timedelta(days=1), end)
        yield cursor, following
        cursor = following


def _require_columns(frame: pl.DataFrame, names: Iterable[str], label: str) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")
