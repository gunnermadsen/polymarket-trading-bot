"""Causal executable-price evidence for asymmetric-value training."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from .asymmetric_value_config import AsymmetricValueConfig
from .core_execution import (
    LEGACY_SNAPSHOT_SCHEMA_VERSION,
    ExecutionEvidenceConfig,
    extract_execution_evidence,
)
from .core_extract import file_sha256, write_json_atomic
from .core_features import (
    attach_causal_oracle_rounds,
    derive_core_point_in_time_features,
    derive_oracle_point_in_time_features,
)

EARLY_CAUSAL_ORACLE_FEATURES = (
    "oracle_return_from_window_open_bps",
    "oracle_round_age_seconds_scaled",
    "oracle_update_count_since_open_scaled",
    "binance_oracle_basis_bps",
)
ORACLE_MINIMUM_PROPAGATION_SECONDS = 2
ORACLE_MAXIMUM_AGE_SECONDS = 300

PRICE_COLUMNS = (
    "market_id",
    "window_start",
    "window_end",
    "label_up",
    "fee_rate",
    "seconds_elapsed",
    "observed_at",
    "yes_received_at",
    "yes_best_ask",
    "yes_ask_vwap_5",
    "yes_ask_depth",
    "no_received_at",
    "no_best_ask",
    "no_ask_vwap_5",
    "no_ask_depth",
    "source_artifact_id",
    "source_schema_version",
    "quality_flags",
    "source_book_regime",
)

POLYMARKET_VALUE_FEATURES = (
    "pm_yes_cost_per_share",
    "pm_no_cost_per_share",
    "pm_yes_cost_logit",
    "pm_no_cost_logit",
    "pm_cost_overround",
    "pm_yes_minus_no_cost",
    "pm_yes_vwap_slippage",
    "pm_no_vwap_slippage",
    "pm_yes_depth_log",
    "pm_no_depth_log",
    "pm_depth_imbalance",
    "pm_yes_book_age_seconds",
    "pm_no_book_age_seconds",
)


def extract_asymmetric_price_evidence(
    config: AsymmetricValueConfig,
    *,
    scope: Literal["development", "evaluation"],
    force: bool = False,
) -> dict[str, Any]:
    """Extract immutable PMXT books at 1s before 60s and 5s thereafter."""

    early_config, later_config = _execution_evidence_configs(config, scope=scope)
    early = extract_execution_evidence(early_config, force=force)
    later = extract_execution_evidence(later_config, force=force)
    coverage_by_day = _coverage_by_day(early, later)
    manifest = {
        **_manifest_identity(config, scope=scope),
        "created_at": datetime.now(UTC).isoformat(),
        "read_only_source": True,
        "source_contract": "btc_execution_evidence_v2",
        "snapshot_provider": "pmxt_v2_execution_snapshots",
        "snapshot_schema_versions": [LEGACY_SNAPSHOT_SCHEMA_VERSION],
        "price_seconds": list(config.price_seconds),
        "early_one_second": early,
        "later_five_second": later,
        "coverage_by_day": coverage_by_day,
        "coverage_totals": _coverage_totals(coverage_by_day),
        "missing_books_are_excluded": True,
        "proxy_prices_used": False,
        "refprice_training_eligible": False,
        "refprice_exclusion_reason": (
            "historical RefPrice reports do not carry a proven local receipt/availability "
            "timestamp"
        ),
    }
    write_json_atomic(config.price_cache / scope / "manifest.json", manifest)
    return manifest


def load_asymmetric_price_evidence(
    config: AsymmetricValueConfig,
    *,
    scope: Literal["development", "evaluation"],
) -> pl.DataFrame:
    manifest_path = config.price_cache / scope / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"asymmetric-value execution manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    observed_identity = {
        name: manifest.get(name) for name in _manifest_identity(config, scope=scope)
    }
    if observed_identity != _manifest_identity(config, scope=scope):
        raise RuntimeError(
            "asymmetric-value execution cache identity changed; rebuild intentionally"
        )

    early_config, later_config = _execution_evidence_configs(config, scope=scope)
    pieces = [
        _load_strict_execution_rows(early_config.output_dir),
        _load_strict_execution_rows(later_config.output_dir),
    ]
    frame = pl.concat(pieces, how="vertical_relaxed")
    if frame.is_empty():
        raise RuntimeError("asymmetric-value price evidence has no strict executable rows")

    regimes = pl.DataFrame(
        {
            "_utc_day": [date.fromisoformat(row["date"]) for row in manifest["coverage_by_day"]],
            "source_book_regime": [
                row["retained_book_regime"] for row in manifest["coverage_by_day"]
            ],
        }
    )
    frame = (
        frame.with_columns(pl.col("window_start").dt.date().alias("_utc_day"))
        .join(regimes, on="_utc_day", how="left", validate="m:1")
        .drop("_utc_day")
        .sort("window_start", "seconds_elapsed")
    )
    missing = sorted(set(PRICE_COLUMNS) - set(frame.columns))
    if missing:
        raise RuntimeError(
            "asymmetric-value price evidence is missing columns: " + ", ".join(missing)
        )
    duplicates = (
        frame.group_by("market_id", "seconds_elapsed")
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicates.height:
        raise RuntimeError("asymmetric-value execution evidence contains duplicate points")
    return frame


def attach_asymmetric_value_features(
    features: pl.DataFrame,
    prices: pl.DataFrame,
    config: AsymmetricValueConfig,
) -> pl.DataFrame:
    """Join strict causal books and construct fee/reserve-inclusive value features."""

    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
    joined = features.join(
        prices,
        on=keys,
        how="inner",
        validate="1:1",
        suffix="_price",
    )
    mismatched_labels = joined.filter(pl.col("label_up") != pl.col("label_up_price"))
    if mismatched_labels.height:
        raise RuntimeError("feature and execution labels disagree for the same market")
    joined = joined.drop("label_up_price")
    strict = joined.filter(
        pl.col("fee_rate").is_not_null()
        & pl.col("fee_rate").is_finite()
        & (pl.col("fee_rate") >= 0)
        & pl.col("yes_best_ask").is_between(0.0, 1.0, closed="right")
        & pl.col("no_best_ask").is_between(0.0, 1.0, closed="right")
        & pl.col("yes_ask_vwap_5").is_between(0.0, 1.0, closed="right")
        & pl.col("no_ask_vwap_5").is_between(0.0, 1.0, closed="right")
        & (pl.col("yes_ask_depth") >= config.quantity)
        & (pl.col("no_ask_depth") >= config.quantity)
        & (pl.col("yes_received_at") <= pl.col("observed_at"))
        & (pl.col("no_received_at") <= pl.col("observed_at"))
        & (
            pl.col("yes_received_at")
            >= pl.col("observed_at")
            - pl.duration(seconds=config.book_freshness_seconds)
        )
        & (
            pl.col("no_received_at")
            >= pl.col("observed_at")
            - pl.duration(seconds=config.book_freshness_seconds)
        )
    )
    if strict.is_empty():
        raise RuntimeError("asymmetric-value feature frame has no strict executable books")

    reserve = config.execution_reserve_per_share
    with_costs = strict.with_columns(
        (
            pl.col("fee_rate")
            * pl.col("yes_ask_vwap_5")
            * (1.0 - pl.col("yes_ask_vwap_5"))
        ).alias("yes_fee_per_share"),
        (
            pl.col("fee_rate")
            * pl.col("no_ask_vwap_5")
            * (1.0 - pl.col("no_ask_vwap_5"))
        ).alias("no_fee_per_share"),
    ).with_columns(
        (pl.col("yes_ask_vwap_5") + pl.col("yes_fee_per_share")).alias(
            "yes_execution_cost_per_share"
        ),
        (pl.col("no_ask_vwap_5") + pl.col("no_fee_per_share")).alias(
            "no_execution_cost_per_share"
        ),
    ).with_columns(
        (pl.col("yes_execution_cost_per_share") + reserve).alias(
            "yes_cost_per_share"
        ),
        (pl.col("no_execution_cost_per_share") + reserve).alias(
            "no_cost_per_share"
        ),
    )
    epsilon = 1e-6
    enriched = with_costs.with_columns(
        pl.col("yes_cost_per_share").alias("pm_yes_cost_per_share"),
        pl.col("no_cost_per_share").alias("pm_no_cost_per_share"),
        (
            pl.col("yes_cost_per_share").clip(epsilon, 1.0 - epsilon).log()
            - (1.0 - pl.col("yes_cost_per_share").clip(epsilon, 1.0 - epsilon)).log()
        ).alias("pm_yes_cost_logit"),
        (
            pl.col("no_cost_per_share").clip(epsilon, 1.0 - epsilon).log()
            - (1.0 - pl.col("no_cost_per_share").clip(epsilon, 1.0 - epsilon)).log()
        ).alias("pm_no_cost_logit"),
        (pl.col("yes_cost_per_share") + pl.col("no_cost_per_share") - 1.0).alias(
            "pm_cost_overround"
        ),
        (pl.col("yes_cost_per_share") - pl.col("no_cost_per_share")).alias(
            "pm_yes_minus_no_cost"
        ),
        (pl.col("yes_ask_vwap_5") - pl.col("yes_best_ask")).alias(
            "pm_yes_vwap_slippage"
        ),
        (pl.col("no_ask_vwap_5") - pl.col("no_best_ask")).alias(
            "pm_no_vwap_slippage"
        ),
        pl.col("yes_ask_depth").log1p().alias("pm_yes_depth_log"),
        pl.col("no_ask_depth").log1p().alias("pm_no_depth_log"),
        (
            (pl.col("yes_ask_depth") - pl.col("no_ask_depth"))
            / (pl.col("yes_ask_depth") + pl.col("no_ask_depth")).clip(
                lower_bound=1e-9
            )
        ).alias("pm_depth_imbalance"),
        (
            (pl.col("observed_at") - pl.col("yes_received_at"))
            .dt.total_milliseconds()
            .cast(pl.Float64)
            / 1_000.0
        ).alias("pm_yes_book_age_seconds"),
        (
            (pl.col("observed_at") - pl.col("no_received_at"))
            .dt.total_milliseconds()
            .cast(pl.Float64)
            / 1_000.0
        ).alias("pm_no_book_age_seconds"),
    )
    duplicates = (
        enriched.group_by("market_id", "seconds_elapsed")
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicates.height:
        raise RuntimeError("asymmetric-value features contain duplicate decision points")
    return enriched.sort("window_start", "seconds_elapsed")


def attach_early_causal_oracle_features(
    core: pl.DataFrame,
    source: Path,
    *,
    minimum_propagation_seconds: int = ORACLE_MINIMUM_PROPAGATION_SECONDS,
    maximum_age_seconds: int = ORACLE_MAXIMUM_AGE_SECONDS,
) -> pl.DataFrame:
    """Derive boundary-independent t=5 oracle fields on raw one-second rows."""

    if core.is_empty():
        return core
    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
    dated = core.with_columns(pl.col("window_start").dt.date().alias("_utc_day"))
    pieces: list[pl.DataFrame] = []
    for day in sorted(dated["_utc_day"].unique().to_list()):
        raw_path = source / f"{day.isoformat()}.parquet"
        oracle_path = source / f"oracle-{day.isoformat()}.parquet"
        if not raw_path.is_file() or not oracle_path.is_file():
            continue
        daily_core = dated.filter(pl.col("_utc_day") == day).drop("_utc_day")
        raw = pl.read_parquet(raw_path).join(
            daily_core.select("market_id").unique(),
            on="market_id",
            how="inner",
            validate="m:1",
        )
        raw_grid = raw.group_by("market_id").agg(
            pl.len().alias("rows"),
            pl.col("seconds_elapsed").n_unique().alias("unique_seconds"),
            pl.col("seconds_elapsed").min().alias("minimum_second"),
            pl.col("seconds_elapsed").max().alias("maximum_second"),
        )
        incomplete = raw_grid.filter(
            (pl.col("rows") != 300)
            | (pl.col("unique_seconds") != 300)
            | (pl.col("minimum_second") != 0)
            | (pl.col("maximum_second") != 299)
        )
        if incomplete.height:
            raise RuntimeError(
                "oracle feature derivation requires a complete 0-299 one-second grid "
                f"for every market: {raw_path}"
            )
        raw_features = derive_core_point_in_time_features(raw)
        oracle_join_frame = raw_features.with_columns(
            pl.col("observed_at").alias("_decision_observed_at"),
            (
                pl.col("observed_at")
                - pl.duration(seconds=minimum_propagation_seconds)
            ).alias("observed_at"),
        )
        with_rounds = attach_causal_oracle_rounds(
            oracle_join_frame,
            pl.read_parquet(oracle_path),
        ).with_columns(
            pl.col("_decision_observed_at").alias("observed_at")
        ).drop(
            "_decision_observed_at"
        )
        oracle = derive_oracle_point_in_time_features(with_rounds).filter(
            pl.col("seconds_elapsed").is_in(list(range(5, 241, 5)))
        )
        oracle = oracle.with_columns(
            (
                pl.col("observed_at") - pl.col("oracle_block_timestamp")
            )
            .dt.total_seconds()
            .alias("oracle_age_seconds")
        ).with_columns(
            (
                pl.col("oracle_source_timestamp").is_not_null()
                & (
                    pl.col("oracle_source_timestamp")
                    <= pl.col("oracle_block_timestamp")
                )
                & (pl.col("oracle_block_timestamp") <= pl.col("observed_at"))
                & pl.col("oracle_age_seconds").is_between(
                    minimum_propagation_seconds,
                    maximum_age_seconds,
                    closed="both",
                )
            )
            .fill_null(False)
            .alias("early_oracle_eligible")
        )
        oracle = oracle.with_columns(
            pl.when(pl.col("early_oracle_eligible"))
            .then(pl.col(feature))
            .otherwise(None)
            .alias(feature)
            for feature in EARLY_CAUSAL_ORACLE_FEATURES
        )
        pieces.append(
            daily_core.join(
                oracle.select(
                    *keys,
                    *EARLY_CAUSAL_ORACLE_FEATURES,
                    "oracle_source_timestamp",
                    "oracle_block_timestamp",
                    "oracle_age_seconds",
                    "early_oracle_eligible",
                ),
                on=keys,
                how="left",
                validate="1:1",
            )
        )
    if not pieces:
        raise RuntimeError("no causal one-second oracle feature partitions were available")
    enriched = pl.concat(pieces, how="vertical_relaxed").sort(
        "window_start", "seconds_elapsed"
    )
    duplicates = (
        enriched.group_by("market_id", "seconds_elapsed")
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicates.height:
        raise RuntimeError("oracle-enriched core contains duplicate decision points")
    violations = enriched.filter(
        pl.col("early_oracle_eligible")
        & (
            (pl.col("oracle_source_timestamp") > pl.col("oracle_block_timestamp"))
            | (pl.col("oracle_block_timestamp") > pl.col("observed_at"))
            | (pl.col("oracle_age_seconds") < minimum_propagation_seconds)
            | (pl.col("oracle_age_seconds") > maximum_age_seconds)
        )
    )
    if violations.height:
        raise RuntimeError("early oracle feature cache contains causal violations")
    return enriched


def exact_price_by_second(prices: pl.DataFrame) -> list[dict[str, Any]]:
    """Summarize exact five-share executable prices at the configured cadence."""

    if prices.is_empty():
        return []
    return (
        prices.with_columns(
            pl.min_horizontal("yes_ask_vwap_5", "no_ask_vwap_5").alias(
                "cheaper_side_vwap_5"
            )
        )
        .group_by("seconds_elapsed")
        .agg(
            pl.len().alias("rows"),
            pl.col("market_id").n_unique().alias("markets"),
            pl.col("yes_ask_vwap_5").mean().alias("mean_yes_vwap_5"),
            pl.col("no_ask_vwap_5").mean().alias("mean_no_vwap_5"),
            pl.col("cheaper_side_vwap_5").mean().alias("mean_cheaper_side_vwap_5"),
            pl.col("cheaper_side_vwap_5").median().alias(
                "median_cheaper_side_vwap_5"
            ),
            pl.col("cheaper_side_vwap_5")
            .is_between(0.20, 0.30, closed="left")
            .mean()
            .alias("share_with_cheaper_side_20_30c"),
        )
        .sort("seconds_elapsed")
        .to_dicts()
    )


def execution_grid_coverage(
    config: AsymmetricValueConfig,
    *,
    scope: Literal["development", "evaluation"],
    core: pl.DataFrame,
) -> dict[str, int | float]:
    """Measure retained and strict source keys against the exact core-market grid."""

    market_ids = core.select("market_id").unique()
    expected_rows = market_ids.height * len(config.price_seconds)
    if expected_rows == 0:
        return {
            "core_markets": 0,
            "expected_rows": 0,
            "retained_rows": 0,
            "strict_rows": 0,
            "retained_coverage": 0.0,
            "strict_coverage": 0.0,
        }
    evidence_configs = _execution_evidence_configs(config, scope=scope)
    scans = [
        pl.scan_parquet(sorted(evidence.output_dir.glob("*.parquet"))).select(
            "market_id",
            "seconds_elapsed",
            "strict_both_side_eligible",
        )
        for evidence in evidence_configs
    ]
    matching = (
        pl.concat(scans, how="vertical_relaxed")
        .join(market_ids.lazy(), on="market_id", how="inner", validate="m:1")
        .collect()
    )
    duplicates = (
        matching.group_by("market_id", "seconds_elapsed")
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicates.height:
        raise RuntimeError("execution source grid contains duplicate core-market keys")
    retained_rows = matching.height
    strict_rows = int(matching["strict_both_side_eligible"].sum())
    return {
        "core_markets": market_ids.height,
        "expected_rows": expected_rows,
        "retained_rows": retained_rows,
        "strict_rows": strict_rows,
        "retained_coverage": retained_rows / expected_rows,
        "strict_coverage": strict_rows / expected_rows,
    }


def _execution_evidence_configs(
    config: AsymmetricValueConfig,
    *,
    scope: Literal["development", "evaluation"],
) -> tuple[ExecutionEvidenceConfig, ExecutionEvidenceConfig]:
    range_start, range_end = _scope_window(config, scope)
    common = {
        "range_start": range_start,
        "range_end": range_end,
        "freshness_seconds": config.book_freshness_seconds,
        "quantity": config.quantity,
        "snapshot_schema_versions": (LEGACY_SNAPSHOT_SCHEMA_VERSION,),
    }
    return (
        ExecutionEvidenceConfig(
            **common,
            output_dir=config.price_cache / scope / "seconds-1-59-one-second",
            sample_interval_seconds=1,
            min_seconds_after_open=1,
            max_seconds_after_open=59,
        ),
        ExecutionEvidenceConfig(
            **common,
            output_dir=config.price_cache / scope / "seconds-60-240-five-second",
            sample_interval_seconds=5,
            min_seconds_after_open=60,
            max_seconds_after_open=240,
        ),
    )


def _load_strict_execution_rows(source: Path) -> pl.DataFrame:
    files = sorted(source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"execution evidence has no Parquet partitions: {source}")
    return (
        pl.scan_parquet(files)
        .filter(pl.col("strict_both_side_eligible"))
        .select(
            "market_id",
            "window_start",
            "window_end",
            "label_up",
            "fee_rate",
            "seconds_elapsed",
            "observed_at",
            pl.col("up_provider_received_at").alias("yes_received_at"),
            pl.col("up_best_ask").alias("yes_best_ask"),
            pl.col("up_ask_vwap_5").alias("yes_ask_vwap_5"),
            pl.col("up_ask_depth").alias("yes_ask_depth"),
            pl.col("down_provider_received_at").alias("no_received_at"),
            pl.col("down_best_ask").alias("no_best_ask"),
            pl.col("down_ask_vwap_5").alias("no_ask_vwap_5"),
            pl.col("down_ask_depth").alias("no_ask_depth"),
            pl.col("artifact_id").alias("source_artifact_id"),
            pl.col("schema_version").alias("source_schema_version"),
            "quality_flags",
        )
        .collect()
    )


def _coverage_by_day(
    early: dict[str, Any],
    later: dict[str, Any],
) -> list[dict[str, Any]]:
    by_date: dict[str, dict[str, int]] = {}
    for cadence, manifest in (("early", early), ("later", later)):
        for partition in manifest["partitions"]:
            day = partition["path"].removesuffix(".parquet")
            values = by_date.setdefault(
                day,
                {
                    "retained_rows": 0,
                    "strict_rows": 0,
                    "markets": 0,
                    "early_retained_rows": 0,
                    "early_strict_rows": 0,
                    "later_retained_rows": 0,
                    "later_strict_rows": 0,
                },
            )
            retained = int(partition["rows"])
            strict = int(partition["strict_both_side_eligible_rows"])
            values["retained_rows"] += retained
            values["strict_rows"] += strict
            values["markets"] = max(values["markets"], int(partition["markets"]))
            values[f"{cadence}_retained_rows"] += retained
            values[f"{cadence}_strict_rows"] += strict
    records: list[dict[str, Any]] = []
    for day, values in sorted(by_date.items()):
        retained = values["retained_rows"]
        strict = values["strict_rows"]
        ratio = strict / retained if retained else 0.0
        if retained == 0:
            regime = "unavailable"
        elif strict == 0:
            regime = "non_executable"
        elif ratio >= 0.80:
            regime = "high"
        elif ratio >= 0.20:
            regime = "moderate"
        else:
            regime = "sparse"
        records.append(
            {
                "date": day,
                **values,
                "strict_ratio_of_retained": ratio,
                "retained_book_regime": regime,
            }
        )
    return records


def _coverage_totals(records: list[dict[str, Any]]) -> dict[str, Any]:
    retained = sum(int(row["retained_rows"]) for row in records)
    strict = sum(int(row["strict_rows"]) for row in records)
    regimes: dict[str, int] = {}
    for row in records:
        regime = str(row["retained_book_regime"])
        regimes[regime] = regimes.get(regime, 0) + 1
    return {
        "days": len(records),
        "retained_rows": retained,
        "strict_rows": strict,
        "strict_ratio_of_retained": strict / retained if retained else 0.0,
        "days_by_retained_book_regime": regimes,
    }


def _scope_window(
    config: AsymmetricValueConfig,
    scope: Literal["development", "evaluation"],
) -> tuple[datetime, datetime]:
    if scope == "development":
        return config.fit.start, config.evaluation.start
    return config.evaluation.start, config.evaluation.end


def _manifest_identity(
    config: AsymmetricValueConfig,
    *,
    scope: Literal["development", "evaluation"],
) -> dict[str, Any]:
    range_start, range_end = _scope_window(config, scope)
    return {
        "schema_version": "btc-asymmetric-value-price-evidence-v2",
        "scope": scope,
        "range_start": range_start.isoformat(),
        "range_end": range_end.isoformat(),
        "quantity": config.quantity,
        "freshness_seconds": config.book_freshness_seconds,
        "price_query_sha256": file_sha256(config.price_source_sql),
    }
