"""Immutable daily feature cache for the spot-L2/Chainlink benchmark.

The source extractor intentionally returns paths rather than one large frame.
This module preserves that property: one UTC day is derived and qualified at a
time, every completed output has an exclusive checksum record, and a completed
manifest is only ever read and verified.  Source timing and lineage columns are
used for qualification but are not persisted as model inputs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from . import chainlink_oi_features as chainlink_feature_builder
from . import core_features as core_feature_builder
from . import spot_l2_chainlink_features as l2_feature_builder
from .chainlink_oi_features import CHAINLINK_CANDLE_FEATURES
from .core_extract import file_sha256, write_json_exclusive
from .core_features import (
    CORE_BOUNDARY_ENRICHED_FEATURES,
    CORE_BOUNDARY_FEATURE_SCHEMA_VERSION,
    CORE_FEATURE_SCHEMA_VERSION,
    audit_final_prices,
    derive_core_point_in_time_features,
)
from .spot_l2_chainlink_config import SpotL2ChainlinkConfig
from .spot_l2_chainlink_extract import (
    L2_FRESHNESS_SECONDS,
    SOURCE_RANGE_END,
    SOURCE_RANGE_START,
    SourceCache,
)
from .spot_l2_chainlink_features import (
    L2_FEATURES,
    join_closed_chainlink_candles,
    join_qualified_l2,
)

FEATURE_CACHE_SCHEMA_VERSION = "btc-spot-l2-chainlink-feature-cache-v1"
FEATURE_PARTITION_RECORD_SCHEMA_VERSION = "btc-feature-partition-record-v1"
FEATURE_DAY_RECORD_SCHEMA_VERSION = "btc-feature-day-record-v1"
POINT_KEYS = ("market_id", "window_start", "seconds_elapsed")
DECISION_SECONDS = tuple(range(60, 241, 5))
HISTORY_SECONDS = tuple(range(241))
EXPECTED_DECISIONS_PER_MARKET = len(DECISION_SECONDS)
CANDLE_HISTORY_MINUTES = 61

CORE_COLUMNS = (
    "market_id",
    "window_start",
    "window_end",
    "official_outcome",
    "label_up",
    "opening_boundary",
    "observed_at",
    "seconds_elapsed",
    "btc_close",
    *CORE_BOUNDARY_ENRICHED_FEATURES,
)

DATASET_DIRECTORIES = {
    "core": "core",
    "primary_l2": "primary-l2",
    "strict_l2_candle": "strict-l2-candle",
    "candle_qualified_keys": "candle-qualified-keys",
    "excluded_keys": "excluded-keys",
}

EXCLUSION_REASONS = (
    "final_price_label_mismatch",
    "incomplete_core_history_0_240",
    "non_finite_base_feature",
    "spot_l2_unavailable_or_older_than_2s",
    "chainlink_candle_unavailable_or_incomplete",
    "spot_l2_and_chainlink_unavailable",
)

_EXCLUDED_SCHEMA = {
    "market_id": pl.String,
    "window_start": pl.Datetime("us", "UTC"),
    "seconds_elapsed": pl.Int32,
    "excluded_from": pl.String,
    "reason": pl.String,
}


@dataclass(frozen=True)
class FeatureCache:
    """Daily immutable outputs consumed by the benchmark runner."""

    core_files: tuple[Path, ...]
    primary_l2_files: tuple[Path, ...]
    strict_l2_candle_files: tuple[Path, ...]
    candle_qualified_key_files: tuple[Path, ...]
    excluded_key_files: tuple[Path, ...]


def load_or_build_feature_cache(
    config: SpotL2ChainlinkConfig,
    source_cache: SourceCache,
    source_manifest: dict[str, Any],
) -> tuple[FeatureCache, dict[str, Any]]:
    """Build, resume, or verify the fixed benchmark feature cache.

    The source cache is verified against its immutable manifest before any
    feature work begins.  Completed day records let an interrupted build skip
    prior days without loading their source partitions again.
    """

    _validate_fixed_config(config)
    source_lineage = _validate_source_cache(config, source_cache, source_manifest)
    feature_root = Path(config.cache) / "features"
    contract = _feature_contract(config, source_lineage)
    contract_sha256 = _json_sha256(contract)
    manifest_path = feature_root / "manifest.json"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        return _validate_completed_cache(feature_root, contract, manifest), manifest

    _prepare_partial_root(feature_root)
    core_by_day = _daily_source_map(source_cache.core_source_files, "core")
    l2_by_day = _daily_source_map(source_cache.l2_files, "spot-L2")
    candle_by_day, candle_support = _candle_source_map(source_cache.candle_files)

    day_records: list[dict[str, Any]] = []
    for current_day in _benchmark_days():
        inputs = _day_source_inputs(
            current_day=current_day,
            core_by_day=core_by_day,
            l2_by_day=l2_by_day,
            candle_by_day=candle_by_day,
            candle_support=candle_support,
        )
        day_contract = {
            "schema_version": FEATURE_DAY_RECORD_SCHEMA_VERSION,
            "date": current_day.isoformat(),
            "feature_contract_sha256": contract_sha256,
            "source_inputs": inputs,
        }
        day_record_path = feature_root / "days" / f"{current_day.isoformat()}.json"
        if day_record_path.is_file():
            day_record = json.loads(day_record_path.read_text())
            _validate_day_record(feature_root, day_contract, day_record)
        else:
            day_record = _build_day(
                config=config,
                feature_root=feature_root,
                current_day=current_day,
                core_by_day=core_by_day,
                l2_by_day=l2_by_day,
                candle_by_day=candle_by_day,
                candle_support=candle_support,
                day_contract=day_contract,
            )
            write_json_exclusive(day_record_path, day_record)
        day_records.append(
            {
                **day_record,
                "day_record_path": f"days/{day_record_path.name}",
                "day_record_sha256": file_sha256(day_record_path),
            }
        )

    manifest = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "immutable": True,
        "paper_only": True,
        "interval_semantics": "half_open_utc",
        "contract": contract,
        "days": day_records,
        "datasets": {
            dataset: {
                "partitions": [day["partitions"][dataset] for day in day_records],
                "totals": _dataset_totals(day_records, dataset),
            }
            for dataset in DATASET_DIRECTORIES
        },
        "totals": _cache_totals(day_records),
    }
    write_json_exclusive(manifest_path, manifest)
    return _validate_completed_cache(feature_root, contract, manifest), manifest


def _validate_fixed_config(config: SpotL2ChainlinkConfig) -> None:
    if tuple(config.decision_seconds) != DECISION_SECONDS:
        raise RuntimeError("feature cache requires the fixed 60..240 five-second cadence")
    if tuple(config.base_features) != tuple(CORE_BOUNDARY_ENRICHED_FEATURES):
        raise RuntimeError("feature cache base schema is not the exact 68-feature champion")
    if len(config.base_features) != 68 or len(L2_FEATURES) != 40:
        raise RuntimeError("fixed benchmark feature dimensions changed")
    if len(CHAINLINK_CANDLE_FEATURES) != 8:
        raise RuntimeError("fixed Chainlink candle feature dimension changed")
    if int(config.execution_freshness_seconds) != L2_FRESHNESS_SECONDS:
        raise RuntimeError("spot-L2 feature freshness must remain two seconds")


def _validate_source_cache(
    config: SpotL2ChainlinkConfig,
    source_cache: SourceCache,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    source_root = Path(config.cache) / "source"
    manifest_path = source_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source cache manifest is missing: {manifest_path}")
    on_disk = json.loads(manifest_path.read_text())
    if on_disk != source_manifest:
        raise RuntimeError("passed source manifest differs from the immutable on-disk manifest")
    sources = source_manifest.get("sources")
    if not isinstance(sources, dict):
        raise TypeError("source cache manifest has no sources")

    cache_paths = {
        "core": source_cache.core_source_files,
        "l2": source_cache.l2_files,
        "candles": source_cache.candle_files,
        "execution": source_cache.execution_files,
    }
    for source, observed_paths in cache_paths.items():
        records = sources.get(source, {}).get("partitions")
        if not isinstance(records, list):
            raise TypeError(f"source cache manifest has no {source} partitions")
        expected_paths: list[Path] = []
        for record in records:
            path = _safe_path(source_root, record.get("path"))
            if not path.is_file() or file_sha256(path) != record.get("sha256"):
                raise RuntimeError(f"immutable source partition hash mismatch: {path.name}")
            expected_paths.append(path)
        if tuple(path.resolve() for path in observed_paths) != tuple(
            path.resolve() for path in expected_paths
        ):
            raise RuntimeError(f"SourceCache {source} paths differ from its manifest")

    return {
        "manifest_path": "source/manifest.json",
        "manifest_sha256": file_sha256(manifest_path),
        "canonical_manifest_sha256": _json_sha256(source_manifest),
        "source_cache_schema_version": source_manifest.get("schema_version"),
        "source_contract": source_manifest.get("contract"),
    }


def _feature_contract(
    config: SpotL2ChainlinkConfig, source_lineage: dict[str, Any]
) -> dict[str, Any]:
    builder_paths = {
        "feature_cache": Path(__file__),
        "core_features": Path(core_feature_builder.__file__),
        "spot_l2_features": Path(l2_feature_builder.__file__),
        "chainlink_candle_features": Path(chainlink_feature_builder.__file__),
    }
    return {
        "range_start": SOURCE_RANGE_START.isoformat(),
        "range_end": SOURCE_RANGE_END.isoformat(),
        "decision_seconds": list(DECISION_SECONDS),
        "required_history_seconds": [HISTORY_SECONDS[0], HISTORY_SECONDS[-1]],
        "required_history_row_count": len(HISTORY_SECONDS),
        "l2_maximum_age_seconds": L2_FRESHNESS_SECONDS,
        "candle_maximum_age_seconds": 60,
        "candle_history_minutes": CANDLE_HISTORY_MINUTES,
        "causality": {
            "spot_l2": (
                "available_at_strictly_before_observed_at_and_source_event_timestamp_age_at_most_2s"
            ),
            "spot_l2_fill_or_interpolation": False,
            "chainlink_candle": ("close_timestamp_and_available_at_strictly_before_observed_at"),
        },
        "feature_schemas": {
            "core_schema_version": CORE_FEATURE_SCHEMA_VERSION,
            "boundary_schema_version": CORE_BOUNDARY_FEATURE_SCHEMA_VERSION,
            "base_features": list(config.base_features),
            "base_feature_sha256": _sequence_sha256(config.base_features),
            "l2_features": list(L2_FEATURES),
            "l2_feature_sha256": _sequence_sha256(L2_FEATURES),
            "candle_features": list(CHAINLINK_CANDLE_FEATURES),
            "candle_feature_sha256": _sequence_sha256(CHAINLINK_CANDLE_FEATURES),
        },
        "config_lineage": {
            "path": Path(config.source_path).name,
            "sha256": file_sha256(Path(config.source_path)),
            "source_schema_revision": config.source_schema_revision,
        },
        "source_lineage": source_lineage,
        "feature_builder_lineage": {
            name: {"path": path.name, "sha256": file_sha256(path)}
            for name, path in builder_paths.items()
        },
        "excluded_key_schema": list(_EXCLUDED_SCHEMA),
        "exclusion_reasons": list(EXCLUSION_REASONS),
    }


def _prepare_partial_root(feature_root: Path) -> None:
    allowed = {*DATASET_DIRECTORIES.values(), "days"}
    if feature_root.exists():
        unexpected = sorted(
            path.name for path in feature_root.iterdir() if path.name not in allowed
        )
        if unexpected:
            raise RuntimeError(
                "feature cache contains unrecognized partial outputs: " + ", ".join(unexpected)
            )
    feature_root.mkdir(parents=True, exist_ok=True)
    (feature_root / "days").mkdir(exist_ok=True)
    for directory in DATASET_DIRECTORIES.values():
        (feature_root / directory).mkdir(exist_ok=True)


def _benchmark_days() -> tuple[date, ...]:
    days: list[date] = []
    current = SOURCE_RANGE_START.date()
    while current < SOURCE_RANGE_END.date():
        days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def _daily_source_map(paths: Sequence[Path], label: str) -> dict[date, Path]:
    mapped: dict[date, Path] = {}
    for path in paths:
        try:
            current_day = date.fromisoformat(path.stem)
        except ValueError as error:
            raise RuntimeError(f"unexpected {label} source partition: {path.name}") from error
        if current_day in mapped:
            raise RuntimeError(f"duplicate {label} source partition: {current_day}")
        mapped[current_day] = path
    expected = set(_benchmark_days())
    if set(mapped) != expected:
        raise RuntimeError(f"{label} source cache does not contain the exact daily range")
    return mapped


def _candle_source_map(paths: Sequence[Path]) -> tuple[dict[date, Path], Path]:
    support = [path for path in paths if path.name == "lookback-support.parquet"]
    daily = [path for path in paths if path.name != "lookback-support.parquet"]
    if len(support) != 1:
        raise RuntimeError("Chainlink source cache requires one lookback-support partition")
    return _daily_source_map(daily, "Chainlink candle"), support[0]


def _day_source_inputs(
    *,
    current_day: date,
    core_by_day: dict[date, Path],
    l2_by_day: dict[date, Path],
    candle_by_day: dict[date, Path],
    candle_support: Path,
) -> dict[str, list[dict[str, Any]]]:
    previous = current_day - timedelta(days=1)
    l2_paths = [l2_by_day[current_day]]
    if previous in l2_by_day:
        l2_paths.insert(0, l2_by_day[previous])
    candle_paths = [candle_by_day[current_day]]
    if previous in candle_by_day:
        candle_paths.insert(0, candle_by_day[previous])
    else:
        candle_paths.insert(0, candle_support)
    return {
        "core": [_source_input_record(core_by_day[current_day])],
        "l2": [_source_input_record(path) for path in l2_paths],
        "candles": [_source_input_record(path) for path in candle_paths],
    }


def _source_input_record(path: Path) -> dict[str, Any]:
    return {
        "path": f"{path.parent.name}/{path.name}",
        "sha256": file_sha256(path),
    }


def _build_day(
    *,
    config: SpotL2ChainlinkConfig,
    feature_root: Path,
    current_day: date,
    core_by_day: dict[date, Path],
    l2_by_day: dict[date, Path],
    candle_by_day: dict[date, Path],
    candle_support: Path,
    day_contract: dict[str, Any],
) -> dict[str, Any]:
    start = datetime.combine(current_day, datetime.min.time(), tzinfo=UTC)
    end = start + timedelta(days=1)
    raw_core = pl.read_parquet(core_by_day[current_day]).sort(["market_id", "seconds_elapsed"])
    _validate_raw_core(raw_core, start, end)
    raw_markets = (
        raw_core.select("market_id", "window_start").unique().sort(["window_start", "market_id"])
    )
    expected_keys = _expected_decision_keys(raw_markets)

    audit = audit_final_prices(raw_core)
    mismatch_ids = sorted(
        audit.filter(pl.col("has_final_price") & ~pl.col("final_price_matches_official"))[
            "market_id"
        ].to_list()
    )
    history = (
        raw_core.filter(pl.col("seconds_elapsed").is_between(0, 240, closed="both"))
        .group_by("market_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("seconds_elapsed").n_unique().alias("unique_seconds"),
            pl.col("seconds_elapsed").min().alias("minimum_second"),
            pl.col("seconds_elapsed").max().alias("maximum_second"),
        )
    )
    complete_ids = set(
        history.filter(
            (pl.col("rows") == len(HISTORY_SECONDS))
            & (pl.col("unique_seconds") == len(HISTORY_SECONDS))
            & (pl.col("minimum_second") == HISTORY_SECONDS[0])
            & (pl.col("maximum_second") == HISTORY_SECONDS[-1])
        )["market_id"].to_list()
    )
    all_ids = set(raw_markets["market_id"].to_list())
    mismatch_set = set(mismatch_ids)
    incomplete_ids = sorted(all_ids - complete_ids - mismatch_set)
    eligible_ids = sorted(complete_ids - mismatch_set)

    derived = derive_core_point_in_time_features(raw_core)
    candidates = (
        derived.filter(
            pl.col("market_id").is_in(eligible_ids)
            & pl.col("seconds_elapsed").is_in(DECISION_SECONDS)
        )
        .select(*CORE_COLUMNS)
        .sort(["window_start", "market_id", "seconds_elapsed"])
    )
    if candidates.height != len(eligible_ids) * EXPECTED_DECISIONS_PER_MARKET:
        raise RuntimeError(f"{current_day} complete core markets lack fixed decision keys")
    invalid_core = candidates.filter(_non_finite_expression(config.base_features))
    core = candidates.filter(~_non_finite_expression(config.base_features))
    _validate_point_frame(core, config.base_features, "core")

    previous = current_day - timedelta(days=1)
    l2_paths = [l2_by_day[current_day]]
    if previous in l2_by_day:
        l2_paths.insert(0, l2_by_day[previous])
    l2_source = _read_bounded_sources(
        l2_paths,
        "available_at",
        start - timedelta(seconds=L2_FRESHNESS_SECONDS),
        end,
    )
    primary_l2 = (
        join_qualified_l2(core, l2_source)
        if l2_source.height
        else _empty_extended(core, L2_FEATURES)
    )
    _validate_point_frame(primary_l2, (*config.base_features, *L2_FEATURES), "primary L2")

    candle_paths = [candle_by_day[current_day]]
    if previous in candle_by_day:
        candle_paths.insert(0, candle_by_day[previous])
    else:
        candle_paths.insert(0, candle_support)
    candle_source = _read_bounded_sources(
        candle_paths,
        "close_timestamp",
        start - timedelta(minutes=CANDLE_HISTORY_MINUTES),
        end,
    )
    candle_qualified = (
        join_closed_chainlink_candles(core, candle_source)
        if candle_source.height
        else _empty_extended(core, CHAINLINK_CANDLE_FEATURES)
    )
    strict = (
        join_closed_chainlink_candles(primary_l2, candle_source)
        if candle_source.height and primary_l2.height
        else _empty_extended(primary_l2, CHAINLINK_CANDLE_FEATURES)
    )
    _validate_point_frame(
        strict,
        (*config.base_features, *L2_FEATURES, *CHAINLINK_CANDLE_FEATURES),
        "strict L2+candle",
    )
    candle_keys = candle_qualified.select(*POINT_KEYS).sort(list(POINT_KEYS))
    _assert_strict_intersection(primary_l2, candle_keys, strict)

    excluded = _excluded_keys(
        expected_keys=expected_keys,
        core=core,
        primary_l2=primary_l2,
        candle_keys=candle_keys,
        strict=strict,
        mismatch_ids=mismatch_ids,
        incomplete_ids=incomplete_ids,
        invalid_core=invalid_core,
    )
    _validate_excluded_keys(expected_keys, core, primary_l2, candle_keys, strict, excluded)

    frames = {
        "core": core,
        "primary_l2": primary_l2,
        "strict_l2_candle": strict,
        "candle_qualified_keys": candle_keys,
        "excluded_keys": excluded,
    }
    partitions = {
        dataset: _load_or_write_partition(
            feature_root=feature_root,
            dataset=dataset,
            current_day=current_day,
            frame=frame,
            feature_contract_sha256=day_contract["feature_contract_sha256"],
        )
        for dataset, frame in frames.items()
    }
    return {
        **day_contract,
        "source": {
            "core_rows": raw_core.height,
            "core_markets": raw_markets.height,
            "l2_rows_loaded": l2_source.height,
            "candle_rows_loaded": candle_source.height,
        },
        "final_price_audit": {
            "markets_with_final_price": audit.filter(pl.col("has_final_price")).height,
            "matching_markets": audit.filter(pl.col("final_price_matches_official")).height,
            "mismatch_markets": len(mismatch_ids),
            "missing_final_price_markets": audit.filter(~pl.col("has_final_price")).height,
        },
        "history": {
            "complete_markets": len(complete_ids),
            "incomplete_markets": len(all_ids - complete_ids),
        },
        "rows": {dataset: frame.height for dataset, frame in frames.items()},
        "markets": {dataset: _market_count(frame) for dataset, frame in frames.items()},
        "exclusions": _exclusion_summary(excluded),
        "partitions": partitions,
    }


def _validate_raw_core(frame: pl.DataFrame, start: datetime, end: datetime) -> None:
    required = {
        "market_id",
        "window_start",
        "window_end",
        "official_outcome",
        "label_up",
        "opening_boundary",
        "final_price",
        "observed_at",
        "seconds_elapsed",
        "btc_close",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError("core source is missing columns: " + ", ".join(missing))
    if frame.is_empty():
        raise RuntimeError(f"core source day is empty: {start.date()}")
    if frame.select(pl.struct(["market_id", "seconds_elapsed"]).n_unique()).item() != frame.height:
        raise RuntimeError(f"core source has duplicate market-second keys: {start.date()}")
    stable_columns = (
        "window_start",
        "window_end",
        "official_outcome",
        "label_up",
        "opening_boundary",
        "final_price",
    )
    inconsistent = (
        frame.group_by("market_id")
        .agg(*(pl.col(column).n_unique().alias(column) for column in stable_columns))
        .filter(pl.any_horizontal(pl.col(column) > 1 for column in stable_columns))
    )
    if inconsistent.height:
        raise RuntimeError(f"core source market facts are inconsistent: {start.date()}")
    invalid = frame.filter(
        (pl.col("window_start") < start)
        | (pl.col("window_start") >= end)
        | ~pl.col("official_outcome").is_in(["up", "down"])
        | (pl.col("label_up") != (pl.col("official_outcome") == "up").cast(pl.Int32))
        | (
            pl.col("observed_at")
            != pl.col("window_start") + pl.duration(seconds=pl.col("seconds_elapsed"))
        )
        | pl.col("opening_boundary").is_null()
        | ~pl.col("opening_boundary").is_finite()
        | (pl.col("opening_boundary") <= 0)
    )
    if invalid.height:
        raise RuntimeError(f"core source contains invalid market or timing rows: {start.date()}")


def _expected_decision_keys(markets: pl.DataFrame) -> pl.DataFrame:
    seconds = pl.DataFrame({"seconds_elapsed": pl.Series(DECISION_SECONDS, dtype=pl.Int32)})
    result = markets.join(seconds, how="cross").select(*POINT_KEYS).sort(list(POINT_KEYS))
    if result.height != markets.height * EXPECTED_DECISIONS_PER_MARKET:
        raise RuntimeError("failed to construct the exact decision-key denominator")
    return result


def _read_bounded_sources(
    paths: Sequence[Path], timestamp: str, start: datetime, end: datetime
) -> pl.DataFrame:
    return (
        pl.scan_parquet(paths)
        .filter((pl.col(timestamp) >= start) & (pl.col(timestamp) < end))
        .collect()
        .sort(timestamp)
    )


def _empty_extended(frame: pl.DataFrame, features: Sequence[str]) -> pl.DataFrame:
    result = frame.head(0)
    return result.with_columns(
        *(pl.Series(name, [], dtype=pl.Float64) for name in features)
    ).select(*frame.columns, *features)


def _non_finite_expression(features: Sequence[str]) -> pl.Expr:
    return pl.any_horizontal(
        pl.col(feature).is_null() | ~pl.col(feature).cast(pl.Float64).is_finite()
        for feature in features
    )


def _validate_point_frame(frame: pl.DataFrame, features: Sequence[str], label: str) -> None:
    required = {*POINT_KEYS, "observed_at", *features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} frame is missing columns: {', '.join(missing)}")
    if frame.select(pl.struct(POINT_KEYS).n_unique()).item() != frame.height:
        raise RuntimeError(f"{label} frame contains duplicate decision keys")
    if frame.height and frame.filter(_non_finite_expression(features)).height:
        raise RuntimeError(f"{label} frame contains null or non-finite model values")
    if frame.filter(~pl.col("seconds_elapsed").is_in(DECISION_SECONDS)).height:
        raise RuntimeError(f"{label} frame contains a decision outside the fixed cadence")


def _assert_strict_intersection(
    primary_l2: pl.DataFrame, candle_keys: pl.DataFrame, strict: pl.DataFrame
) -> None:
    expected = (
        primary_l2.select(*POINT_KEYS)
        .join(candle_keys, on=list(POINT_KEYS), how="inner", validate="1:1")
        .sort(list(POINT_KEYS))
    )
    observed = strict.select(*POINT_KEYS).sort(list(POINT_KEYS))
    if not observed.equals(expected, null_equal=True):
        raise RuntimeError("strict cohort is not the exact L2/candle key intersection")


def _excluded_keys(
    *,
    expected_keys: pl.DataFrame,
    core: pl.DataFrame,
    primary_l2: pl.DataFrame,
    candle_keys: pl.DataFrame,
    strict: pl.DataFrame,
    mismatch_ids: Sequence[str],
    incomplete_ids: Sequence[str],
    invalid_core: pl.DataFrame,
) -> pl.DataFrame:
    records: list[pl.DataFrame] = []
    records.append(
        _annotate_exclusion(
            expected_keys.filter(pl.col("market_id").is_in(mismatch_ids)),
            "core",
            "final_price_label_mismatch",
        )
    )
    records.append(
        _annotate_exclusion(
            expected_keys.filter(pl.col("market_id").is_in(incomplete_ids)),
            "core",
            "incomplete_core_history_0_240",
        )
    )
    records.append(
        _annotate_exclusion(invalid_core.select(*POINT_KEYS), "core", "non_finite_base_feature")
    )
    records.append(
        _annotate_exclusion(
            _anti_keys(core, primary_l2),
            "primary_l2",
            "spot_l2_unavailable_or_older_than_2s",
        )
    )
    records.append(
        _annotate_exclusion(
            _anti_keys(core, candle_keys),
            "candle_qualified_keys",
            "chainlink_candle_unavailable_or_incomplete",
        )
    )

    l2_keys = primary_l2.select(*POINT_KEYS).with_columns(pl.lit(True).alias("_has_l2"))
    chainlink_keys = candle_keys.with_columns(pl.lit(True).alias("_has_candle"))
    strict_missing = (
        core.select(*POINT_KEYS)
        .join(l2_keys, on=list(POINT_KEYS), how="left", validate="1:1")
        .join(chainlink_keys, on=list(POINT_KEYS), how="left", validate="1:1")
        .filter(pl.col("_has_l2").is_null() | pl.col("_has_candle").is_null())
        .with_columns(
            pl.when(pl.col("_has_l2").is_null() & pl.col("_has_candle").is_null())
            .then(pl.lit("spot_l2_and_chainlink_unavailable"))
            .when(pl.col("_has_l2").is_null())
            .then(pl.lit("spot_l2_unavailable_or_older_than_2s"))
            .otherwise(pl.lit("chainlink_candle_unavailable_or_incomplete"))
            .alias("reason")
        )
        .select(*POINT_KEYS, "reason")
        .with_columns(pl.lit("strict_l2_candle").alias("excluded_from"))
        .select(*_EXCLUDED_SCHEMA)
    )
    records.append(strict_missing)
    nonempty = [frame for frame in records if frame.height]
    if not nonempty:
        return pl.DataFrame(schema=_EXCLUDED_SCHEMA)
    return pl.concat(nonempty, how="vertical").sort(
        ["excluded_from", "window_start", "market_id", "seconds_elapsed", "reason"]
    )


def _annotate_exclusion(keys: pl.DataFrame, excluded_from: str, reason: str) -> pl.DataFrame:
    if reason not in EXCLUSION_REASONS:
        raise RuntimeError(f"unregistered feature exclusion reason: {reason}")
    return (
        keys.select(*POINT_KEYS)
        .with_columns(
            pl.lit(excluded_from).alias("excluded_from"),
            pl.lit(reason).alias("reason"),
        )
        .select(*_EXCLUDED_SCHEMA)
    )


def _anti_keys(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    return left.select(*POINT_KEYS).join(right.select(*POINT_KEYS), on=list(POINT_KEYS), how="anti")


def _validate_excluded_keys(
    expected_keys: pl.DataFrame,
    core: pl.DataFrame,
    primary_l2: pl.DataFrame,
    candle_keys: pl.DataFrame,
    strict: pl.DataFrame,
    excluded: pl.DataFrame,
) -> None:
    pairs = {
        "core": (expected_keys, core),
        "primary_l2": (core, primary_l2),
        "candle_qualified_keys": (core, candle_keys),
        "strict_l2_candle": (core, strict),
    }
    for cohort, (denominator, included) in pairs.items():
        expected = _anti_keys(denominator, included).sort(list(POINT_KEYS))
        observed = (
            excluded.filter(pl.col("excluded_from") == cohort)
            .select(*POINT_KEYS)
            .unique()
            .sort(list(POINT_KEYS))
        )
        if not observed.equals(expected, null_equal=True):
            raise RuntimeError(f"excluded-key evidence is incomplete for {cohort}")
    if excluded.filter(~pl.col("reason").is_in(EXCLUSION_REASONS)).height:
        raise RuntimeError("excluded-key evidence contains an unknown reason")
    if (
        excluded.select(pl.struct([*POINT_KEYS, "excluded_from"]).n_unique()).item()
        != excluded.height
    ):
        raise RuntimeError("excluded-key evidence contains duplicate cohort keys")


def _load_or_write_partition(
    *,
    feature_root: Path,
    dataset: str,
    current_day: date,
    frame: pl.DataFrame,
    feature_contract_sha256: str,
) -> dict[str, Any]:
    directory = feature_root / DATASET_DIRECTORIES[dataset]
    destination = directory / f"{current_day.isoformat()}.parquet"
    partial = destination.with_suffix(destination.suffix + ".partial")
    record_path = destination.with_suffix(destination.suffix + ".json")
    relative_path = f"{directory.name}/{destination.name}"
    relative_record_path = f"{directory.name}/{record_path.name}"
    schema = {name: str(dtype) for name, dtype in frame.schema.items()}
    contract = {
        "record_schema_version": FEATURE_PARTITION_RECORD_SCHEMA_VERSION,
        "feature_contract_sha256": feature_contract_sha256,
        "dataset": dataset,
        "date": current_day.isoformat(),
        "path": relative_path,
        "schema": schema,
    }
    summary = _frame_summary(frame)

    if record_path.is_file():
        record = json.loads(record_path.read_text())
        _require_contract(record, contract, f"{dataset} {current_day}")
        candidate = destination if destination.is_file() else partial
        if not candidate.is_file():
            raise RuntimeError(f"recorded feature partition is missing: {relative_path}")
        if record.get("sha256") != file_sha256(candidate) or record.get("summary") != summary:
            raise RuntimeError(f"immutable feature partition changed: {relative_path}")
        if candidate == partial:
            partial.replace(destination)
        elif partial.exists():
            raise RuntimeError(f"unexpected partial beside immutable partition: {relative_path}")
        return record

    if destination.exists():
        raise RuntimeError(f"unrecorded feature partition cannot be reused: {relative_path}")
    partial.unlink(missing_ok=True)
    try:
        frame.write_parquet(partial, compression="zstd", statistics=True)
        record = {
            **contract,
            "sha256": file_sha256(partial),
            "summary": summary,
            "record_path": relative_record_path,
        }
        write_json_exclusive(record_path, record)
        partial.replace(destination)
        return record
    except BaseException:
        if not record_path.exists():
            partial.unlink(missing_ok=True)
        raise


def _frame_summary(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "rows": frame.height,
        "markets": _market_count(frame),
        "minimum_window_start": (frame["window_start"].min().isoformat() if frame.height else None),
        "maximum_window_start": (frame["window_start"].max().isoformat() if frame.height else None),
    }


def _market_count(frame: pl.DataFrame) -> int:
    return frame["market_id"].n_unique() if frame.height else 0


def _exclusion_summary(frame: pl.DataFrame) -> dict[str, Any]:
    records = frame.select("excluded_from", "reason").to_dicts()
    by_reason = Counter((row["excluded_from"], row["reason"]) for row in records)
    return {
        "records": frame.height,
        "unique_decision_keys": frame.select(pl.struct(POINT_KEYS).n_unique()).item(),
        "by_cohort_and_reason": [
            {"excluded_from": cohort, "reason": reason, "rows": rows}
            for (cohort, reason), rows in sorted(by_reason.items())
        ],
    }


def _validate_day_record(
    feature_root: Path, expected_contract: dict[str, Any], record: dict[str, Any]
) -> None:
    _require_contract(record, expected_contract, f"feature day {expected_contract['date']}")
    partitions = record.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != set(DATASET_DIRECTORIES):
        raise RuntimeError("feature day record has an invalid partition set")
    for dataset, partition in partitions.items():
        _validate_partition_record(feature_root, dataset, partition)


def _validate_partition_record(feature_root: Path, dataset: str, record: dict[str, Any]) -> Path:
    if record.get("dataset") != dataset:
        raise RuntimeError(f"feature partition dataset changed: {dataset}")
    path = _safe_path(feature_root, record.get("path"))
    expected_parent = DATASET_DIRECTORIES[dataset]
    if path.parent.name != expected_parent:
        raise RuntimeError(f"feature partition escaped its dataset directory: {path}")
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise RuntimeError(f"immutable feature partition hash mismatch: {path.name}")
    record_path = _safe_path(feature_root, record.get("record_path"))
    if not record_path.is_file() or json.loads(record_path.read_text()) != record:
        raise RuntimeError(f"immutable feature partition record mismatch: {record_path.name}")
    return path


def _validate_completed_cache(
    feature_root: Path, contract: dict[str, Any], manifest: dict[str, Any]
) -> FeatureCache:
    expected = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "immutable": True,
        "paper_only": True,
        "interval_semantics": "half_open_utc",
        "contract": contract,
    }
    _require_contract(manifest, expected, "feature cache")
    days = manifest.get("days")
    if not isinstance(days, list) or [row.get("date") for row in days] != [
        current.isoformat() for current in _benchmark_days()
    ]:
        raise RuntimeError("feature cache does not contain the exact daily range")
    contract_sha256 = _json_sha256(contract)
    for current_day, day in zip(_benchmark_days(), days, strict=True):
        if (
            day.get("schema_version") != FEATURE_DAY_RECORD_SCHEMA_VERSION
            or day.get("date") != current_day.isoformat()
            or day.get("feature_contract_sha256") != contract_sha256
        ):
            raise RuntimeError(f"feature day contract changed: {current_day}")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASET_DIRECTORIES):
        raise RuntimeError("feature cache dataset set changed")

    paths: dict[str, tuple[Path, ...]] = {}
    for dataset in DATASET_DIRECTORIES:
        partitions = datasets[dataset].get("partitions")
        if not isinstance(partitions, list) or len(partitions) != len(days):
            raise RuntimeError(f"feature cache has an invalid {dataset} partition list")
        dataset_paths = tuple(
            _validate_partition_record(feature_root, dataset, record) for record in partitions
        )
        embedded = [day.get("partitions", {}).get(dataset) for day in days]
        if partitions != embedded:
            raise RuntimeError(f"feature cache {dataset} partitions differ from day records")
        if datasets[dataset].get("totals") != _dataset_totals(days, dataset):
            raise RuntimeError(f"feature cache {dataset} totals changed")
        for current_day, record in zip(_benchmark_days(), partitions, strict=True):
            if (
                record.get("record_schema_version") != FEATURE_PARTITION_RECORD_SCHEMA_VERSION
                or record.get("feature_contract_sha256") != contract_sha256
                or record.get("date") != current_day.isoformat()
            ):
                raise RuntimeError(f"feature partition contract changed: {dataset} {current_day}")
        paths[dataset] = dataset_paths

    for day in days:
        day_path = _safe_path(feature_root, day.get("day_record_path"))
        if not day_path.is_file() or file_sha256(day_path) != day.get("day_record_sha256"):
            raise RuntimeError(f"immutable feature day record mismatch: {day_path.name}")
        stored = json.loads(day_path.read_text())
        without_lineage = {
            key: value
            for key, value in day.items()
            if key not in {"day_record_path", "day_record_sha256"}
        }
        if stored != without_lineage:
            raise RuntimeError(f"feature day manifest embedding changed: {day_path.name}")

    if manifest.get("totals") != _cache_totals(days):
        raise RuntimeError("feature cache totals changed")

    return FeatureCache(
        core_files=paths["core"],
        primary_l2_files=paths["primary_l2"],
        strict_l2_candle_files=paths["strict_l2_candle"],
        candle_qualified_key_files=paths["candle_qualified_keys"],
        excluded_key_files=paths["excluded_keys"],
    )


def _dataset_totals(days: Sequence[dict[str, Any]], dataset: str) -> dict[str, int]:
    return {
        "rows": sum(int(day["rows"][dataset]) for day in days),
        "market_days": sum(int(day["markets"][dataset]) for day in days),
        "partitions": len(days),
    }


def _cache_totals(days: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_rows": sum(int(day["source"]["core_rows"]) for day in days),
        "source_market_days": sum(int(day["source"]["core_markets"]) for day in days),
        "final_price_mismatch_market_days": sum(
            int(day["final_price_audit"]["mismatch_markets"]) for day in days
        ),
        "history_incomplete_market_days": sum(
            int(day["history"]["incomplete_markets"]) for day in days
        ),
        "excluded_records": sum(int(day["exclusions"]["records"]) for day in days),
    }


def _safe_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise TypeError("cache manifest path must be a non-empty string")
    root_resolved = root.resolve()
    path = (root_resolved / relative).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as error:
        raise RuntimeError(f"cache manifest path escapes its root: {relative}") from error
    return path


def _require_contract(observed: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    changed = [key for key, value in expected.items() if observed.get(key) != value]
    if changed:
        raise RuntimeError(f"immutable {label} contract changed: {', '.join(changed)}")


def _sequence_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
