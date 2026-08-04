"""Immutable bounded source cache for the spot-L2/Chainlink benchmark.

``load_or_extract_source_cache`` accepts a config-like object with these
attributes:

* ``package_root`` and ``cache`` (``Path`` values),
* ``core_source_sql``, ``l2_source_sql``, and ``candles_source_sql``
  (``Path`` values; package SQL defaults are used for compatibility), and
* ``champion_process`` (the frozen boundary-alignment paper-process JSON), and
* ``quantity`` and ``freshness_seconds`` (the benchmark config exposes the
  latter as ``execution_freshness_seconds``).

Quantity and freshness are frozen to five shares and two seconds.  Arrival
latency/depth scenarios come only from the hashed process JSON.  The source
interval is intentionally not configurable: this cache is only valid for the
half-open UTC interval ``[2026-04-14, 2026-08-02)``.

The returned ``SourceCache`` contains paths rather than eagerly concatenated
frames.  Callers can scan only the partitions needed for a split.  Source
lineage lives in JSON manifests and is never added to model rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

import psycopg
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .core_execution import (
    classify_side_eligibility,
    strict_both_side_eligible,
    strict_both_side_eligible_10,
)
from .core_extract import (
    CORE_SOURCE_SCHEMA,
    configure_read_only_connection,
    database_connection,
    file_sha256,
    partition_summary,
    write_json_exclusive,
)

SOURCE_CACHE_SCHEMA_VERSION = "btc-spot-l2-chainlink-source-cache-v1"
PARTITION_RECORD_SCHEMA_VERSION = "btc-source-partition-record-v1"
CORE_SOURCE_SCHEMA_VERSION = "btc-spot-l2-core-source-v1"
L2_SOURCE_SCHEMA_VERSION = "binance-spot-btcusdt-l2-one-second-features-v1"
CANDLE_SOURCE_SCHEMA_VERSION = "chainlink-btcusd-closed-one-minute-candles-v1"

SOURCE_RANGE_START = datetime(2026, 4, 14, tzinfo=UTC)
SOURCE_RANGE_END = datetime(2026, 8, 2, tzinfo=UTC)
DECISION_SECONDS = tuple(range(60, 241, 5))
EXECUTION_SECONDS = tuple(range(60, 246, 5))
SAMPLE_INTERVAL_SECONDS = 5
L2_FRESHNESS_SECONDS = 2
EXECUTION_QUANTITY = 5.0
CHAINLINK_HISTORY_MINUTES = 61
CHAINLINK_SYMBOL = "BTCUSD"
BINANCE_SYMBOL = "BTCUSDT"

L2_SOURCE_RELATION = "polymarket.binance_spot_btcusdt_l2_training_features"
L2_BASE_RELATION = "polymarket.binance_spot_btcusdt_l2_one_second_features"
L2_MATERIALIZATION_CONTRACTS = (
    "cryptohft-binance-spot-btcusdt-l2-features-v1",
    "coinapi-binance-spot-btcusdt-l2-snapshots-v1",
    "huggingface-goooddy-binance-spot-btcusdt-l2-features-v1",
)

# These are the actual forty numeric columns of the canonical training view.
# Timing, provider, or quality metadata is deliberately not counted as a model
# dimension and must never be substituted for one of these columns.
L2_INFORMATION_COLUMNS = (
    "midpoint",
    "microprice",
    "spread_bps",
    "bid_depth_5",
    "ask_depth_5",
    "imbalance_5",
    "bid_depth_10",
    "ask_depth_10",
    "imbalance_10",
    "bid_depth_20",
    "ask_depth_20",
    "imbalance_20",
    "bid_depth_slope_20",
    "ask_depth_slope_20",
    "bid_depth_concentration_20",
    "ask_depth_concentration_20",
    "bid_quote_replenishment_1s",
    "ask_quote_replenishment_1s",
    "bid_quote_churn_1s",
    "ask_quote_churn_1s",
    "midpoint_change_bps_1s",
    "spread_bps_delta_1s",
    "depth_20_change_bps_1s",
    "imbalance_20_delta_1s",
    "midpoint_change_bps_5s",
    "spread_bps_delta_5s",
    "depth_20_change_bps_5s",
    "imbalance_20_delta_5s",
    "midpoint_change_bps_15s",
    "spread_bps_delta_15s",
    "depth_20_change_bps_15s",
    "imbalance_20_delta_15s",
    "midpoint_change_bps_30s",
    "spread_bps_delta_30s",
    "depth_20_change_bps_30s",
    "imbalance_20_delta_30s",
    "midpoint_change_bps_60s",
    "spread_bps_delta_60s",
    "depth_20_change_bps_60s",
    "imbalance_20_delta_60s",
)
L2_IDENTITY_COLUMNS = (
    "symbol",
    "second_start",
    "source_event_timestamp",
    "provider_received_at",
    "available_at",
    "source_update_id",
)
L2_SOURCE_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("second_start", pa.timestamp("us", tz="UTC")),
        ("source_event_timestamp", pa.timestamp("us", tz="UTC")),
        ("provider_received_at", pa.timestamp("us", tz="UTC")),
        ("available_at", pa.timestamp("us", tz="UTC")),
        ("source_update_id", pa.int64()),
        *((column, pa.float64()) for column in L2_INFORMATION_COLUMNS),
    ]
)
CANDLE_SOURCE_SCHEMA = pa.schema(
    [
        ("open_timestamp", pa.timestamp("us", tz="UTC")),
        ("close_timestamp", pa.timestamp("us", tz="UTC")),
        ("available_at", pa.timestamp("us", tz="UTC")),
        ("open_price", pa.float64()),
        ("high_price", pa.float64()),
        ("low_price", pa.float64()),
        ("close_price", pa.float64()),
    ]
)
EXPECTED_L2_DATABASE_SCHEMA = (
    ("symbol", "text"),
    ("second_start", "timestamp with time zone"),
    ("source_event_timestamp", "timestamp with time zone"),
    ("provider_received_at", "timestamp with time zone"),
    ("available_at", "timestamp with time zone"),
    ("source_update_id", "bigint"),
    *((column, "numeric") for column in L2_INFORMATION_COLUMNS),
)
EXECUTION_STRESS_SOURCE_SCHEMA_VERSION = "btc-boundary-execution-stress-source-v1"
EXECUTION_SNAPSHOT_SCHEMA_VERSION = "btc5m-book-250ms-v1"
EXECUTION_STRESS_SCHEMA = pa.schema(
    [
        ("market_id", pa.string()),
        ("window_start", pa.timestamp("us", tz="UTC")),
        ("window_end", pa.timestamp("us", tz="UTC")),
        ("official_outcome", pa.string()),
        ("label_up", pa.int32()),
        ("min_tick_size", pa.float64()),
        ("min_order_size", pa.float64()),
        ("fee_rate", pa.float64()),
        ("decision_at", pa.timestamp("us", tz="UTC")),
        ("seconds_elapsed", pa.int32()),
        ("scenario_key", pa.string()),
        ("configured_arrival_latency_ms", pa.int32()),
        ("visible_depth_fraction", pa.float64()),
        ("raw_quantity_required", pa.float64()),
        ("price_stress_method", pa.string()),
        ("price_stress_vwap_quantity", pa.int32()),
        ("price_stress_exact", pa.bool_()),
        ("snapshot_at", pa.timestamp("us", tz="UTC")),
        ("realized_arrival_latency_ms", pa.int32()),
        ("artifact_id", pa.string()),
        ("schema_version", pa.string()),
        ("up_source_row_number", pa.int64()),
        ("up_source_timestamp", pa.timestamp("us", tz="UTC")),
        ("up_provider_received_at", pa.timestamp("us", tz="UTC")),
        ("up_best_bid", pa.float64()),
        ("up_best_ask", pa.float64()),
        ("up_best_bid_size", pa.float64()),
        ("up_best_ask_size", pa.float64()),
        ("up_bid_depth", pa.float64()),
        ("up_ask_depth", pa.float64()),
        ("up_ask_vwap_1", pa.float64()),
        ("up_ask_vwap_5", pa.float64()),
        ("up_ask_vwap_10", pa.float64()),
        ("up_imbalance", pa.float64()),
        ("down_source_row_number", pa.int64()),
        ("down_source_timestamp", pa.timestamp("us", tz="UTC")),
        ("down_provider_received_at", pa.timestamp("us", tz="UTC")),
        ("down_best_bid", pa.float64()),
        ("down_best_ask", pa.float64()),
        ("down_best_bid_size", pa.float64()),
        ("down_best_ask_size", pa.float64()),
        ("down_bid_depth", pa.float64()),
        ("down_ask_depth", pa.float64()),
        ("down_ask_vwap_1", pa.float64()),
        ("down_ask_vwap_5", pa.float64()),
        ("down_ask_vwap_10", pa.float64()),
        ("down_imbalance", pa.float64()),
        ("quality_flags", pa.int32()),
        ("up_provider_causal", pa.bool_()),
        ("down_provider_causal", pa.bool_()),
        ("up_fields_complete", pa.bool_()),
        ("down_fields_complete", pa.bool_()),
        ("up_side_valid", pa.bool_()),
        ("down_side_valid", pa.bool_()),
        ("up_side_fresh", pa.bool_()),
        ("down_side_fresh", pa.bool_()),
        ("strict_both_side_eligible_5", pa.bool_()),
        ("strict_both_side_eligible_10", pa.bool_()),
    ]
)


@dataclass(frozen=True)
class ExecutionScenario:
    scenario_key: str
    arrival_latency_ms: int
    visible_depth_fraction: float
    raw_quantity_required: float
    price_stress_method: str
    price_stress_exact: bool


class SourceExtractConfig(Protocol):
    """Structural type accepted by :func:`load_or_extract_source_cache`."""

    package_root: Path
    cache: Path
    core_source_sql: Path
    l2_source_sql: Path
    candles_source_sql: Path
    champion_process: Path
    quantity: float
    freshness_seconds: int
    source_schema_revision: str


@dataclass(frozen=True)
class SourceCache:
    core_source_files: tuple[Path, ...]
    l2_files: tuple[Path, ...]
    candle_files: tuple[Path, ...]
    execution_files: tuple[Path, ...]


@dataclass(frozen=True)
class _ExtractSettings:
    package_root: Path
    source_root: Path
    core_source_sql: Path
    l2_source_sql: Path
    candles_source_sql: Path
    lineage_sql: Path
    execution_source_sql: Path
    champion_process: Path
    execution_scenarios: tuple[ExecutionScenario, ...]
    source_schema_revision: str
    quantity: float
    freshness_seconds: int


def load_or_extract_source_cache(
    config: SourceExtractConfig,
) -> tuple[SourceCache, dict[str, Any]]:
    """Load or resume the benchmark's immutable daily source cache.

    A completed ``source/manifest.json`` is never rewritten.  Before it exists,
    each completed Parquet partition has an exclusive checksum sidecar, so a
    stopped extraction can safely resume without replacing completed data.
    Database sessions are read-only and every query is bounded to one UTC day
    (apart from the explicit 61-minute candle lookback support partition).
    """

    settings = _extract_settings(config)
    contract = _cache_contract(settings)
    manifest_path = settings.source_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        cache = _validate_completed_cache(settings, contract, manifest)
        return cache, manifest

    settings.source_root.mkdir(parents=True, exist_ok=True)
    for name in ("core", "l2", "candles", "execution"):
        (settings.source_root / name).mkdir(parents=True, exist_ok=True)

    connection: psycopg.Connection[Any] | None = None
    try:
        connection = database_connection()
        configure_read_only_connection(connection)
        l2_relation = _record_l2_relation_identity(connection, settings)

        core_records = _extract_core_partitions(connection, settings)
        l2_records = _extract_l2_partitions(connection, settings)
        candle_records = _extract_candle_partitions(connection, settings)
        execution_partition_records = _extract_execution_partitions(connection, settings)

        if _inspect_l2_relation(connection) != l2_relation:
            raise RuntimeError("canonical spot-L2 view changed during extraction")
    finally:
        if connection is not None:
            connection.close()

    manifest = {
        "schema_version": SOURCE_CACHE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "immutable": True,
        "paper_only": True,
        "contract": contract,
        "sources": {
            "core": {
                "partitions": _manifest_partition_records(settings, "core", core_records),
                "totals": _core_totals(core_records),
            },
            "l2": {
                "partitions": _manifest_partition_records(settings, "l2", l2_records),
                "totals": _l2_totals(l2_records),
                "relation_lineage": l2_relation,
                "materialization_artifacts": _aggregate_l2_artifacts(l2_records),
            },
            "candles": {
                "partitions": _manifest_partition_records(settings, "candles", candle_records),
                "totals": _candle_totals(candle_records),
            },
            "execution": {
                "partitions": _manifest_partition_records(
                    settings, "execution", execution_partition_records
                ),
                "totals": _execution_totals(execution_partition_records),
                "scenarios": [
                    _execution_scenario_record(scenario)
                    for scenario in settings.execution_scenarios
                ],
                "snapshot_artifacts": _execution_artifacts(execution_partition_records),
            },
        },
    }
    write_json_exclusive(manifest_path, manifest)
    cache = _validate_completed_cache(settings, contract, manifest)
    return cache, manifest


def _extract_settings(config: SourceExtractConfig) -> _ExtractSettings:
    package_root = Path(config.package_root)
    cache = Path(config.cache)
    sql_root = package_root / "sql"
    quantity = float(getattr(config, "quantity", EXECUTION_QUANTITY))
    freshness = int(
        _first_attribute(
            config,
            ("freshness_seconds", "l2_freshness_seconds", "execution_freshness_seconds"),
            L2_FRESHNESS_SECONDS,
        )
    )
    if quantity != EXECUTION_QUANTITY:
        raise ValueError("source extraction quantity is frozen at five shares")
    if freshness != L2_FRESHNESS_SECONDS:
        raise ValueError("source freshness is frozen at two seconds")
    if len(L2_INFORMATION_COLUMNS) != 40:
        raise RuntimeError("canonical spot-L2 information dimension count changed")

    champion_process = Path(
        getattr(
            config,
            "champion_process",
            package_root.parents[1]
            / "infra"
            / "processes"
            / "btc-5m-directional-model-paper-boundary-alignment.json",
        )
    )
    execution_scenarios = _load_execution_scenarios(
        champion_process, quantity=quantity, freshness_seconds=freshness
    )
    configured_scenarios = getattr(config, "execution_scenarios", None)
    if configured_scenarios is not None:
        observed_scenarios = tuple(
            (
                str(scenario.key),
                int(scenario.arrival_latency_ms),
                float(scenario.visible_depth_haircut),
            )
            for scenario in configured_scenarios
        )
        expected_scenarios = tuple(
            (
                scenario.scenario_key,
                scenario.arrival_latency_ms,
                scenario.visible_depth_fraction,
            )
            for scenario in execution_scenarios
        )
        if observed_scenarios != expected_scenarios:
            raise RuntimeError("configured execution scenarios diverge from the process")
    if getattr(config, "l2_view", L2_SOURCE_RELATION) != L2_SOURCE_RELATION:
        raise ValueError("canonical spot-L2 source view changed")
    if (
        tuple(getattr(config, "l2_materialization_contracts", L2_MATERIALIZATION_CONTRACTS))
        != L2_MATERIALIZATION_CONTRACTS
    ):
        raise ValueError("spot-L2 materialization contracts changed")
    if getattr(config, "l2_schema_version", L2_SOURCE_SCHEMA_VERSION) != (L2_SOURCE_SCHEMA_VERSION):
        raise ValueError("spot-L2 feature schema version changed")
    if getattr(config, "candle_symbol", CHAINLINK_SYMBOL) != CHAINLINK_SYMBOL:
        raise ValueError("Chainlink candle symbol changed")
    source_schema_revision = str(getattr(config, "source_schema_revision", "not-provided"))

    settings = _ExtractSettings(
        package_root=package_root,
        source_root=cache / "source",
        core_source_sql=Path(getattr(config, "core_source_sql", sql_root / "btc-core-source.sql")),
        l2_source_sql=Path(getattr(config, "l2_source_sql", sql_root / "btc-spot-l2-source.sql")),
        candles_source_sql=Path(
            getattr(
                config,
                "candles_source_sql",
                sql_root / "btc-chainlink-one-minute-candles-source.sql",
            )
        ),
        lineage_sql=sql_root / "btc-spot-l2-materialization-lineage.sql",
        execution_source_sql=sql_root / "btc-spot-l2-execution-stress-source.sql",
        champion_process=champion_process,
        execution_scenarios=execution_scenarios,
        source_schema_revision=source_schema_revision,
        quantity=quantity,
        freshness_seconds=freshness,
    )
    for path in (
        settings.core_source_sql,
        settings.l2_source_sql,
        settings.candles_source_sql,
        settings.lineage_sql,
        settings.execution_source_sql,
        settings.champion_process,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"source SQL is missing: {path}")
    return settings


def _first_attribute(config: object, names: Sequence[str], default: object) -> object:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _load_execution_scenarios(
    path: Path,
    *,
    quantity: float,
    freshness_seconds: int,
) -> tuple[ExecutionScenario, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"champion paper process is missing: {path}")
    process = json.loads(path.read_text())
    try:
        process_config = process["config"]
        paper_config = process_config["raw"]["btc_realtime_paper"]
        strategy = paper_config["strategy"]
        paper = paper_config["paper"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("champion paper process has an invalid structure") from error
    if (
        process.get("process_key") != "btc-5m-directional-model-paper-boundary-alignment"
        or process_config.get("execution", {}).get("mode") != "paper"
        or process_config.get("execution", {}).get("live_capital") is not False
        or Decimal(str(strategy.get("target_size"))) != Decimal(5)
        or int(strategy.get("min_seconds_after_open", -1)) != DECISION_SECONDS[0]
        or int(strategy.get("min_seconds_before_close", -1)) != 60
        or int(strategy.get("max_book_age_ms", -1)) != freshness_seconds * 1_000
    ):
        raise RuntimeError("champion paper execution contract changed")

    observed = (
        (
            "arrival_150ms_depth_80pct",
            int(paper.get("arrival_latency_ms", -1)),
            Decimal(str(paper.get("visible_depth_haircut"))),
        ),
        *tuple(
            (
                str(preview.get("scenario_key")),
                int(preview.get("arrival_latency_ms", -1)),
                Decimal(str(preview.get("visible_depth_haircut"))),
            )
            for preview in paper.get("stress_previews", [])
        ),
    )
    expected = (
        ("arrival_150ms_depth_80pct", 150, Decimal("0.80")),
        ("latency_300ms_depth_65pct", 300, Decimal("0.65")),
        ("latency_600ms_depth_50pct", 600, Decimal("0.50")),
    )
    if observed != expected:
        raise RuntimeError("champion paper latency/depth scenarios changed")

    scenarios: list[ExecutionScenario] = []
    for scenario_key, arrival_latency_ms, visible_depth_fraction in observed:
        exact = visible_depth_fraction == Decimal("0.50")
        scenarios.append(
            ExecutionScenario(
                scenario_key=scenario_key,
                arrival_latency_ms=arrival_latency_ms,
                visible_depth_fraction=float(visible_depth_fraction),
                raw_quantity_required=float(Decimal(str(quantity)) / visible_depth_fraction),
                price_stress_method=(
                    "vwap10_exact"
                    if exact
                    else "vwap10_conservative_proxy_for_unstored_raw_quantity"
                ),
                price_stress_exact=exact,
            )
        )
    return tuple(scenarios)


def _execution_scenario_record(scenario: ExecutionScenario) -> dict[str, Any]:
    return {
        "scenario_key": scenario.scenario_key,
        "arrival_latency_ms": scenario.arrival_latency_ms,
        "visible_depth_fraction": scenario.visible_depth_fraction,
        "target_quantity": EXECUTION_QUANTITY,
        "raw_quantity_required": scenario.raw_quantity_required,
        "available_vwap_quantities": [1, 5, 10],
        "price_stress_method": scenario.price_stress_method,
        "price_stress_vwap_quantity": 10,
        "price_stress_exact": scenario.price_stress_exact,
    }


def _cache_contract(settings: _ExtractSettings) -> dict[str, Any]:
    schemas = {
        "core": _schema_sha256(CORE_SOURCE_SCHEMA),
        "l2": _schema_sha256(L2_SOURCE_SCHEMA),
        "candles": _schema_sha256(CANDLE_SOURCE_SCHEMA),
        "execution": _schema_sha256(EXECUTION_STRESS_SCHEMA),
    }
    sql = {
        "core": file_sha256(settings.core_source_sql),
        "l2": file_sha256(settings.l2_source_sql),
        "l2_materialization_lineage": file_sha256(settings.lineage_sql),
        "candles": file_sha256(settings.candles_source_sql),
        "execution": file_sha256(settings.execution_source_sql),
    }
    return {
        "range_start": SOURCE_RANGE_START.isoformat(),
        "range_end": SOURCE_RANGE_END.isoformat(),
        "interval_semantics": "half_open_utc",
        "source_schema_revision": settings.source_schema_revision,
        "source_schema_versions": {
            "core": CORE_SOURCE_SCHEMA_VERSION,
            "l2": L2_SOURCE_SCHEMA_VERSION,
            "candles": CANDLE_SOURCE_SCHEMA_VERSION,
            "execution": EXECUTION_STRESS_SOURCE_SCHEMA_VERSION,
        },
        "source_schema_sha256": schemas,
        "sql_sha256": sql,
        "source_relations": {
            "core": [
                "polymarket.btc_interval_markets",
                "polymarket.btc_market_reference_facts",
                "polymarket.binance_one_second_klines",
                "polymarket.backfill_artifacts",
            ],
            "l2": [L2_SOURCE_RELATION],
            "candles": [
                "polymarket.chainlink_btcusd_one_minute_candles",
                "polymarket.backfill_artifacts",
            ],
            "execution": [
                "polymarket.btc_interval_markets",
                "polymarket.btc_market_execution_snapshots",
                "polymarket.backfill_artifacts",
            ],
        },
        "l2_information_columns": list(L2_INFORMATION_COLUMNS),
        "l2_information_dimension_count": 40,
        "l2_materialization_contracts": list(L2_MATERIALIZATION_CONTRACTS),
        "decision_seconds": list(DECISION_SECONDS),
        "execution_context_seconds": list(EXECUTION_SECONDS),
        "execution_snapshot_window": {
            "first_snapshot_at_or_after_configured_latency": True,
            "upper_bound": "strictly_before_next_five_second_decision",
        },
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "freshness_seconds": settings.freshness_seconds,
        "quantity": settings.quantity,
        "champion_process_sha256": file_sha256(settings.champion_process),
        "execution_snapshot_schema_version": EXECUTION_SNAPSHOT_SCHEMA_VERSION,
        "execution_scenarios": [
            _execution_scenario_record(scenario) for scenario in settings.execution_scenarios
        ],
        "execution_price_stress_limitation": {
            "stored_vwap_quantities": [1, 5, 10],
            "vwap10_proxy_scenarios": [
                "arrival_150ms_depth_80pct",
                "latency_300ms_depth_65pct",
            ],
            "exact_vwap10_scenarios": ["latency_600ms_depth_50pct"],
            "note": (
                "Raw 6.25- and 7.6923076923-share VWAPs are not stored; "
                "VWAP10 is retained only as a conservative price proxy."
            ),
        },
        "chainlink_history_minutes": CHAINLINK_HISTORY_MINUTES,
        "chainlink_candle_availability": {
            "field": "available_at",
            "semantics": "provider_interval_close_timestamp",
            "database_ingested_at_is_not_historical_availability": True,
        },
        "causality": {
            "l2_available_at": "strictly_before_decision",
            "l2_maximum_age_seconds": settings.freshness_seconds,
            "l2_forward_fill": False,
            "l2_interpolation": False,
            "chainlink_candle": ("fully_closed_and_available_strictly_before_decision"),
        },
    }


def _extract_core_partitions(
    connection: psycopg.Connection[Any], settings: _ExtractSettings
) -> list[dict[str, Any]]:
    query = settings.core_source_sql.read_text()
    records: list[dict[str, Any]] = []
    for batch_start, batch_end in _daily_ranges(SOURCE_RANGE_START, SOURCE_RANGE_END):
        records.append(
            _load_or_extract_partition(
                connection,
                source="core",
                output_dir=settings.source_root / "core",
                destination_name=f"{batch_start.date().isoformat()}.parquet",
                batch_start=batch_start,
                batch_end=batch_end,
                query=query,
                parameters={
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "strict_final_price_audit": False,
                },
                schema=CORE_SOURCE_SCHEMA,
                schema_version=CORE_SOURCE_SCHEMA_VERSION,
                summary=lambda path, start=batch_start, end=batch_end: _core_summary(
                    path, start, end
                ),
            )
        )
    return records


def _extract_l2_partitions(
    connection: psycopg.Connection[Any], settings: _ExtractSettings
) -> list[dict[str, Any]]:
    query = settings.l2_source_sql.read_text()
    lineage_query = settings.lineage_sql.read_text()
    records: list[dict[str, Any]] = []
    for batch_start, batch_end in _daily_ranges(SOURCE_RANGE_START, SOURCE_RANGE_END):
        records.append(
            _load_or_extract_partition(
                connection,
                source="l2",
                output_dir=settings.source_root / "l2",
                destination_name=f"{batch_start.date().isoformat()}.parquet",
                batch_start=batch_start,
                batch_end=batch_end,
                query=query,
                parameters={"batch_start": batch_start, "batch_end": batch_end},
                schema=L2_SOURCE_SCHEMA,
                schema_version=L2_SOURCE_SCHEMA_VERSION,
                summary=lambda path, start=batch_start, end=batch_end: _l2_summary(
                    path, start, end
                ),
                extra=lambda start=batch_start, end=batch_end: {
                    "materialization_artifacts": _l2_artifact_lineage(
                        connection,
                        lineage_query,
                        batch_start=start,
                        batch_end=end,
                    )
                },
                validate_extra=_validate_l2_partition_lineage,
            )
        )
    return records


def _extract_candle_partitions(
    connection: psycopg.Connection[Any], settings: _ExtractSettings
) -> list[dict[str, Any]]:
    query = settings.candles_source_sql.read_text()
    output_dir = settings.source_root / "candles"
    support_start = SOURCE_RANGE_START - timedelta(minutes=CHAINLINK_HISTORY_MINUTES)
    ranges = [
        ("lookback-support.parquet", support_start, SOURCE_RANGE_START, "lookback_support"),
        *[
            (
                f"{batch_start.date().isoformat()}.parquet",
                batch_start,
                batch_end,
                "benchmark_interval",
            )
            for batch_start, batch_end in _daily_ranges(SOURCE_RANGE_START, SOURCE_RANGE_END)
        ],
    ]
    records: list[dict[str, Any]] = []
    for destination_name, batch_start, batch_end, role in ranges:
        record = _load_or_extract_partition(
            connection,
            source="candles",
            output_dir=output_dir,
            destination_name=destination_name,
            batch_start=batch_start,
            batch_end=batch_end,
            query=query,
            parameters={
                "range_start": batch_start,
                "range_end": batch_end,
                "history_minutes": 0,
                "candle_symbol": CHAINLINK_SYMBOL,
            },
            schema=CANDLE_SOURCE_SCHEMA,
            schema_version=CANDLE_SOURCE_SCHEMA_VERSION,
            summary=lambda path, start=batch_start, end=batch_end: _candle_summary(
                path, start, end
            ),
            contract_extra={"role": role},
        )
        records.append(record)
    return records


def _extract_execution_partitions(
    connection: psycopg.Connection[Any], settings: _ExtractSettings
) -> list[dict[str, Any]]:
    query = settings.execution_source_sql.read_text()
    scenarios = settings.execution_scenarios
    parameters = {
        "minimum_decision_second": DECISION_SECONDS[0],
        "maximum_decision_second": DECISION_SECONDS[-1],
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "scenario_keys": [scenario.scenario_key for scenario in scenarios],
        "arrival_latency_milliseconds": [scenario.arrival_latency_ms for scenario in scenarios],
        "visible_depth_fractions": [scenario.visible_depth_fraction for scenario in scenarios],
        "raw_quantities_required": [scenario.raw_quantity_required for scenario in scenarios],
        "price_stress_methods": [scenario.price_stress_method for scenario in scenarios],
        "price_stress_exactness": [scenario.price_stress_exact for scenario in scenarios],
        "freshness_seconds": settings.freshness_seconds,
    }
    records: list[dict[str, Any]] = []
    for batch_start, batch_end in _daily_ranges(SOURCE_RANGE_START, SOURCE_RANGE_END):
        records.append(
            _load_or_extract_partition(
                connection,
                source="execution",
                output_dir=settings.source_root / "execution",
                destination_name=f"{batch_start.date().isoformat()}.parquet",
                batch_start=batch_start,
                batch_end=batch_end,
                query=query,
                parameters={
                    **parameters,
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                },
                schema=EXECUTION_STRESS_SCHEMA,
                schema_version=EXECUTION_STRESS_SOURCE_SCHEMA_VERSION,
                summary=lambda path, start=batch_start, end=batch_end: _execution_summary(
                    path, start, end, settings
                ),
                contract_extra={
                    "champion_process_sha256": file_sha256(settings.champion_process),
                    "snapshot_schema_version": EXECUTION_SNAPSHOT_SCHEMA_VERSION,
                    "scenarios": [
                        _execution_scenario_record(scenario)
                        for scenario in settings.execution_scenarios
                    ],
                },
            )
        )
    return records


def _load_or_extract_partition(
    connection: psycopg.Connection[Any],
    *,
    source: str,
    output_dir: Path,
    destination_name: str,
    batch_start: datetime,
    batch_end: datetime,
    query: str,
    parameters: dict[str, Any],
    schema: pa.Schema,
    schema_version: str,
    summary: Callable[[Path], dict[str, Any]],
    extra: Callable[[], dict[str, Any]] | None = None,
    validate_extra: Callable[[dict[str, Any]], None] | None = None,
    contract_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / destination_name
    partial = destination.with_suffix(destination.suffix + ".partial")
    record_path = destination.with_suffix(destination.suffix + ".json")
    partition_contract = {
        "record_schema_version": PARTITION_RECORD_SCHEMA_VERSION,
        "source": source,
        "range_start": batch_start.isoformat(),
        "range_end": batch_end.isoformat(),
        "path": destination.name,
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "source_schema_version": schema_version,
        "source_schema_sha256": _schema_sha256(schema),
        **(contract_extra or {}),
    }

    if record_path.is_file():
        record = json.loads(record_path.read_text())
        _require_contract(record, partition_contract, f"{source} {destination.name}")
        candidate = destination if destination.is_file() else partial
        if not candidate.is_file():
            raise RuntimeError(f"recorded {source} partition is missing: {destination.name}")
        observed_summary = summary(candidate)
        if (
            record.get("sha256") != file_sha256(candidate)
            or record.get("summary") != observed_summary
        ):
            raise RuntimeError(f"immutable {source} partition changed: {destination.name}")
        if validate_extra is not None:
            validate_extra(record)
        if candidate == partial:
            partial.replace(destination)
        elif partial.exists():
            raise RuntimeError(
                f"unexpected partial file beside immutable partition: {partial.name}"
            )
        return record

    if destination.exists():
        raise RuntimeError(f"unrecorded {source} partition cannot be reused: {destination.name}")
    partial.unlink(missing_ok=True)
    try:
        _stream_query_to_parquet(
            connection,
            query,
            parameters,
            partial,
            schema,
            cursor_name=f"btc_spot_cache_{source}_{batch_start:%Y%m%d%H%M}",
        )
        observed_summary = summary(partial)
        record = {
            **partition_contract,
            "sha256": file_sha256(partial),
            "summary": observed_summary,
            **(extra() if extra is not None else {}),
        }
        if validate_extra is not None:
            validate_extra(record)
        write_json_exclusive(record_path, record)
        partial.replace(destination)
        return record
    except BaseException:
        if not record_path.exists():
            partial.unlink(missing_ok=True)
        raise


def _stream_query_to_parquet(
    connection: psycopg.Connection[Any],
    query: str,
    parameters: dict[str, Any],
    destination: Path,
    schema: pa.Schema,
    *,
    cursor_name: str,
) -> int:
    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        with connection.transaction():
            connection.execute("SET TRANSACTION READ ONLY")
            with connection.cursor(name=cursor_name) as cursor:
                cursor.execute(query, parameters)
                observed_columns = tuple(column.name for column in cursor.description or ())
                if observed_columns != tuple(schema.names):
                    raise RuntimeError(
                        "source SQL columns do not match the frozen schema: "
                        + ", ".join(observed_columns)
                    )
                while rows := cursor.fetchmany(10_000):
                    records = [dict(zip(schema.names, row, strict=True)) for row in rows]
                    table = pa.Table.from_pylist(records, schema=schema)
                    if writer is None:
                        dictionary_columns = [
                            column
                            for column in ("market_id", "official_outcome", "symbol")
                            if column in schema.names
                        ]
                        writer = pq.ParquetWriter(
                            destination,
                            schema,
                            compression="zstd",
                            write_statistics=True,
                            use_dictionary=dictionary_columns,
                        )
                    writer.write_table(table)
                    row_count += len(rows)
    finally:
        if writer is not None:
            writer.close()
    if row_count == 0:
        pq.write_table(
            pa.Table.from_pylist([], schema=schema),
            destination,
            compression="zstd",
        )
    return row_count


def _core_summary(path: Path, batch_start: datetime, batch_end: datetime) -> dict[str, Any]:
    _require_parquet_schema(path, CORE_SOURCE_SCHEMA)
    table = pq.read_table(
        path,
        columns=[
            "market_id",
            "window_start",
            "official_outcome",
            "label_up",
            "opening_boundary",
            "seconds_elapsed",
            "final_price",
        ],
    )
    rows = table.to_pylist()
    keys = [(row["market_id"], row["seconds_elapsed"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{path.name} contains duplicate core decision-source keys")
    invalid = [
        row
        for row in rows
        if not batch_start <= row["window_start"] < batch_end
        or row["official_outcome"] not in {"up", "down"}
        or row["label_up"] != (1 if row["official_outcome"] == "up" else 0)
        or row["opening_boundary"] is None
        or not math.isfinite(row["opening_boundary"])
        or row["opening_boundary"] <= 0
    ]
    if invalid:
        raise RuntimeError(f"{path.name} contains invalid official market/core rows")
    base = partition_summary(path)
    base["up_rows"] = sum(row["label_up"] == 1 for row in rows)
    base["down_rows"] = sum(row["label_up"] == 0 for row in rows)
    return base


def _l2_summary(path: Path, batch_start: datetime, batch_end: datetime) -> dict[str, Any]:
    _require_parquet_schema(path, L2_SOURCE_SCHEMA)
    table = pq.read_table(path)
    for column in L2_INFORMATION_COLUMNS:
        values = table[column]
        if values.null_count or pc.sum(pc.invert(pc.is_finite(values))).as_py():
            raise RuntimeError(f"{path.name} contains invalid L2 values in {column}")
    rows = table.select(L2_IDENTITY_COLUMNS).to_pylist()
    for row in rows:
        if row["symbol"] != BINANCE_SYMBOL:
            raise RuntimeError(f"{path.name} contains a non-BTCUSDT L2 row")
        if not batch_start <= row["available_at"] < batch_end:
            raise RuntimeError(f"{path.name} contains an out-of-range L2 row")
        if not (
            row["source_event_timestamp"] <= row["available_at"]
            and row["provider_received_at"] <= row["available_at"]
            and row["second_start"] <= row["available_at"]
            and row["available_at"] < row["second_start"] + timedelta(seconds=1)
        ):
            raise RuntimeError(f"{path.name} contains a non-causal L2 source row")
    seconds = {row["second_start"] for row in rows}
    available = [row["available_at"] for row in rows]
    return {
        "rows": table.num_rows,
        "qualified_seconds": len(seconds),
        "minimum_second_start": min(seconds).isoformat() if seconds else None,
        "maximum_second_start": max(seconds).isoformat() if seconds else None,
        "minimum_available_at": min(available).isoformat() if available else None,
        "maximum_available_at": max(available).isoformat() if available else None,
    }


def _candle_summary(path: Path, batch_start: datetime, batch_end: datetime) -> dict[str, Any]:
    _require_parquet_schema(path, CANDLE_SOURCE_SCHEMA)
    table = pq.read_table(path)
    rows = table.to_pylist()
    closes: list[datetime] = []
    for row in rows:
        prices = [row[name] for name in ("open_price", "high_price", "low_price", "close_price")]
        if (
            row["close_timestamp"] != row["open_timestamp"] + timedelta(minutes=1)
            or row["available_at"] < row["close_timestamp"]
            or not batch_start <= row["close_timestamp"] < batch_end
            or any(not math.isfinite(value) or value <= 0 for value in prices)
            or row["high_price"] < max(prices)
            or row["low_price"] > min(prices)
        ):
            raise RuntimeError(f"{path.name} contains an invalid or unclosed candle")
        closes.append(row["close_timestamp"])
    if len(closes) != len(set(closes)):
        raise RuntimeError(f"{path.name} contains duplicate Chainlink candle closes")
    return {
        "rows": table.num_rows,
        "fully_closed_candles": table.num_rows,
        "minimum_close_timestamp": min(closes).isoformat() if closes else None,
        "maximum_close_timestamp": max(closes).isoformat() if closes else None,
        "availability_semantics": "provider_interval_close_timestamp",
    }


def _execution_summary(
    path: Path,
    batch_start: datetime,
    batch_end: datetime,
    settings: _ExtractSettings,
) -> dict[str, Any]:
    _require_parquet_schema(path, EXECUTION_STRESS_SCHEMA)
    rows = pq.read_table(path).to_pylist()
    scenario_by_key = {scenario.scenario_key: scenario for scenario in settings.execution_scenarios}
    keys: set[tuple[str, datetime, str]] = set()
    markets: set[str] = set()
    decision_keys: set[tuple[str, datetime]] = set()
    rows_by_market: dict[str, int] = defaultdict(int)
    artifact_rows: dict[tuple[str, str], int] = defaultdict(int)
    by_scenario: dict[str, dict[str, int]] = {
        scenario.scenario_key: {
            "rows": 0,
            "snapshot_rows": 0,
            "missing_snapshot_rows": 0,
            "strict_both_side_eligible_5_rows": 0,
            "strict_both_side_eligible_10_rows": 0,
        }
        for scenario in settings.execution_scenarios
    }

    for row in rows:
        scenario = scenario_by_key.get(row["scenario_key"])
        if scenario is None:
            raise RuntimeError(f"{path.name} contains an unknown execution scenario")
        key = (row["market_id"], row["decision_at"], row["scenario_key"])
        if key in keys:
            raise RuntimeError(f"{path.name} contains duplicate execution stress keys")
        keys.add(key)
        markets.add(row["market_id"])
        decision_keys.add((row["market_id"], row["decision_at"]))
        rows_by_market[row["market_id"]] += 1
        scenario_summary = by_scenario[scenario.scenario_key]
        scenario_summary["rows"] += 1

        expected_decision_at = row["window_start"] + timedelta(seconds=row["seconds_elapsed"])
        if (
            not batch_start <= row["window_start"] < batch_end
            or row["seconds_elapsed"] not in DECISION_SECONDS
            or row["decision_at"] != expected_decision_at
            or row["official_outcome"] not in {"up", "down"}
            or row["label_up"] != (1 if row["official_outcome"] == "up" else 0)
            or row["configured_arrival_latency_ms"] != scenario.arrival_latency_ms
            or not math.isclose(
                row["visible_depth_fraction"],
                scenario.visible_depth_fraction,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                row["raw_quantity_required"],
                scenario.raw_quantity_required,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or row["price_stress_method"] != scenario.price_stress_method
            or row["price_stress_vwap_quantity"] != 10
            or row["price_stress_exact"] is not scenario.price_stress_exact
        ):
            raise RuntimeError(f"{path.name} contains an invalid execution stress key")

        if row["snapshot_at"] is None:
            scenario_summary["missing_snapshot_rows"] += 1
            if any(
                row[name]
                for name in (
                    "up_provider_causal",
                    "down_provider_causal",
                    "up_fields_complete",
                    "down_fields_complete",
                    "up_side_valid",
                    "down_side_valid",
                    "up_side_fresh",
                    "down_side_fresh",
                    "strict_both_side_eligible_5",
                    "strict_both_side_eligible_10",
                )
            ):
                raise RuntimeError(f"{path.name} marks a missing execution snapshot as eligible")
            continue

        scenario_summary["snapshot_rows"] += 1
        minimum_snapshot_at = row["decision_at"] + timedelta(
            milliseconds=scenario.arrival_latency_ms
        )
        maximum_snapshot_at = row["decision_at"] + timedelta(seconds=SAMPLE_INTERVAL_SECONDS)
        realized_latency = round((row["snapshot_at"] - row["decision_at"]).total_seconds() * 1_000)
        if (
            not minimum_snapshot_at <= row["snapshot_at"] < maximum_snapshot_at
            or row["realized_arrival_latency_ms"] != realized_latency
            or row["schema_version"] != EXECUTION_SNAPSHOT_SCHEMA_VERSION
            or row["artifact_id"] is None
            or row["quality_flags"] is None
        ):
            raise RuntimeError(f"{path.name} contains an invalid post-latency snapshot")
        artifact_rows[(row["artifact_id"], row["schema_version"])] += 1

        quality_flags = int(row["quality_flags"])
        up = classify_side_eligibility(
            side="up",
            observed_at=row["snapshot_at"],
            provider_received_at=row["up_provider_received_at"],
            best_bid=row["up_best_bid"],
            best_ask=row["up_best_ask"],
            best_bid_size=row["up_best_bid_size"],
            best_ask_size=row["up_best_ask_size"],
            bid_depth=row["up_bid_depth"],
            ask_depth=row["up_ask_depth"],
            ask_vwap_5=row["up_ask_vwap_5"],
            ask_vwap_10=row["up_ask_vwap_10"],
            imbalance=row["up_imbalance"],
            quality_flags=quality_flags,
            freshness_seconds=settings.freshness_seconds,
        )
        down = classify_side_eligibility(
            side="down",
            observed_at=row["snapshot_at"],
            provider_received_at=row["down_provider_received_at"],
            best_bid=row["down_best_bid"],
            best_ask=row["down_best_ask"],
            best_bid_size=row["down_best_bid_size"],
            best_ask_size=row["down_best_ask_size"],
            bid_depth=row["down_bid_depth"],
            ask_depth=row["down_ask_depth"],
            ask_vwap_5=row["down_ask_vwap_5"],
            ask_vwap_10=row["down_ask_vwap_10"],
            imbalance=row["down_imbalance"],
            quality_flags=quality_flags,
            freshness_seconds=settings.freshness_seconds,
        )
        up_fields_complete = all(
            row[name] is not None
            for name in (
                "up_best_bid",
                "up_best_ask",
                "up_best_bid_size",
                "up_best_ask_size",
                "up_bid_depth",
                "up_ask_depth",
                "up_ask_vwap_1",
                "up_ask_vwap_5",
                "up_ask_vwap_10",
                "up_imbalance",
            )
        )
        down_fields_complete = all(
            row[name] is not None
            for name in (
                "down_best_bid",
                "down_best_ask",
                "down_best_bid_size",
                "down_best_ask_size",
                "down_bid_depth",
                "down_ask_depth",
                "down_ask_vwap_1",
                "down_ask_vwap_5",
                "down_ask_vwap_10",
                "down_imbalance",
            )
        )
        expected_flags = {
            "up_provider_causal": (
                row["up_provider_received_at"] is not None
                and row["up_provider_received_at"] <= row["snapshot_at"]
            ),
            "down_provider_causal": (
                row["down_provider_received_at"] is not None
                and row["down_provider_received_at"] <= row["snapshot_at"]
            ),
            "up_fields_complete": up_fields_complete,
            "down_fields_complete": down_fields_complete,
            "up_side_valid": up.valid,
            "down_side_valid": down.valid,
            "up_side_fresh": up.fresh,
            "down_side_fresh": down.fresh,
            "strict_both_side_eligible_5": strict_both_side_eligible(
                up=up, down=down, quality_flags=quality_flags
            ),
            "strict_both_side_eligible_10": strict_both_side_eligible_10(
                up=up, down=down, quality_flags=quality_flags
            )
            and row["up_ask_vwap_1"] is not None
            and row["down_ask_vwap_1"] is not None,
        }
        mismatches = [name for name, value in expected_flags.items() if row[name] is not value]
        if mismatches:
            raise RuntimeError(
                f"{path.name} has inconsistent execution quality flags: " + ", ".join(mismatches)
            )
        if row["strict_both_side_eligible_5"]:
            scenario_summary["strict_both_side_eligible_5_rows"] += 1
        if row["strict_both_side_eligible_10"]:
            scenario_summary["strict_both_side_eligible_10_rows"] += 1

    expected_rows_per_market = len(DECISION_SECONDS) * len(settings.execution_scenarios)
    if any(count != expected_rows_per_market for count in rows_by_market.values()):
        raise RuntimeError(f"{path.name} does not retain every execution stress key")
    return {
        "rows": len(rows),
        "markets": len(markets),
        "decision_keys": len(decision_keys),
        "snapshot_rows": sum(value["snapshot_rows"] for value in by_scenario.values()),
        "missing_snapshot_rows": sum(
            value["missing_snapshot_rows"] for value in by_scenario.values()
        ),
        "strict_both_side_eligible_5_rows": sum(
            value["strict_both_side_eligible_5_rows"] for value in by_scenario.values()
        ),
        "strict_both_side_eligible_10_rows": sum(
            value["strict_both_side_eligible_10_rows"] for value in by_scenario.values()
        ),
        "by_scenario": by_scenario,
        "snapshot_artifacts": [
            {
                "artifact_id": artifact_id,
                "schema_version": schema_version,
                "rows": count,
            }
            for (artifact_id, schema_version), count in sorted(artifact_rows.items())
        ],
    }


def _l2_artifact_lineage(
    connection: psycopg.Connection[Any],
    query: str,
    *,
    batch_start: datetime,
    batch_end: datetime,
) -> list[dict[str, Any]]:
    with connection.transaction():
        connection.execute("SET LOCAL statement_timeout = '60s'")
        connection.execute("SET LOCAL work_mem = '16MB'")
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                {"batch_start": batch_start, "batch_end": batch_end},
            )
            columns = [column.name for column in cursor.description or ()]
            rows = cursor.fetchall()
    return [
        {key: _json_value(value) for key, value in zip(columns, row, strict=True)} for row in rows
    ]


def _validate_l2_partition_lineage(record: dict[str, Any]) -> None:
    artifacts = record.get("materialization_artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("L2 partition has no materialization lineage")
    contracts = {artifact.get("materialization_contract") for artifact in artifacts}
    if not contracts.issubset(L2_MATERIALIZATION_CONTRACTS):
        raise RuntimeError("L2 partition contains an unqualified materialization contract")
    source_rows = sum(int(artifact["source_rows"]) for artifact in artifacts)
    if source_rows != int(record["summary"]["rows"]):
        raise RuntimeError("L2 materialization lineage row count does not match source")


def _record_l2_relation_identity(
    connection: psycopg.Connection[Any], settings: _ExtractSettings
) -> dict[str, Any]:
    identity = _inspect_l2_relation(connection)
    path = settings.source_root / "l2" / "relation-lineage.json"
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != identity:
            raise RuntimeError("canonical spot-L2 view identity changed in a partial cache")
        return existing
    write_json_exclusive(path, identity)
    return identity


def _inspect_l2_relation(connection: psycopg.Connection[Any]) -> dict[str, Any]:
    with connection.transaction():
        connection.execute("SET LOCAL statement_timeout = '5s'")
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'polymarket'
                  AND table_name = 'binance_spot_btcusdt_l2_training_features'
                ORDER BY ordinal_position
                """
            )
            columns = tuple((str(row[0]), str(row[1])) for row in cursor.fetchall())
            cursor.execute(
                """
                SELECT relation.relkind, pg_get_viewdef(relation.oid, true)
                FROM pg_class relation
                JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = 'polymarket'
                  AND relation.relname = 'binance_spot_btcusdt_l2_training_features'
                """
            )
            relation = cursor.fetchone()
    if columns != EXPECTED_L2_DATABASE_SCHEMA:
        raise RuntimeError("canonical spot-L2 view does not have the expected 40-column schema")
    if relation is None or relation[0] != "v":
        raise RuntimeError("canonical spot-L2 training relation must be a database view")
    view_definition = str(relation[1])
    if L2_BASE_RELATION not in view_definition:
        raise RuntimeError("canonical spot-L2 view has unexpected source lineage")
    if any(contract not in view_definition for contract in L2_MATERIALIZATION_CONTRACTS):
        raise RuntimeError("canonical spot-L2 view is missing a frozen materialization contract")
    schema_payload = [
        {"ordinal_position": index, "name": name, "data_type": data_type}
        for index, (name, data_type) in enumerate(columns, start=1)
    ]
    return {
        "relation": L2_SOURCE_RELATION,
        "relation_kind": "view",
        "columns": schema_payload,
        "column_count": len(columns),
        "information_dimension_count": len(L2_INFORMATION_COLUMNS),
        "schema_sha256": hashlib.sha256(
            json.dumps(schema_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "view_definition": view_definition,
        "view_definition_sha256": hashlib.sha256(view_definition.encode()).hexdigest(),
        "materialization_contracts": list(L2_MATERIALIZATION_CONTRACTS),
    }


def _execution_artifacts(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        for row in record["summary"]["snapshot_artifacts"]:
            counts[(row["artifact_id"], row["schema_version"])] += int(row["rows"])
    return [
        {"artifact_id": artifact_id, "schema_version": schema_version, "rows": rows}
        for (artifact_id, schema_version), rows in sorted(counts.items())
    ]


def _manifest_partition_records(
    settings: _ExtractSettings,
    source: str,
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        record_path = settings.source_root / source / f"{record['path']}.json"
        output.append(
            {
                "path": f"{source}/{record['path']}",
                "sha256": record["sha256"],
                "rows": int(record["summary"]["rows"]),
                "range_start": record["range_start"],
                "range_end": record["range_end"],
                **({"role": record["role"]} if "role" in record else {}),
                "record_path": f"{source}/{record_path.name}",
                "record_sha256": file_sha256(record_path),
            }
        )
    return output


def _aggregate_l2_artifacts(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for record in records:
        for row in record["materialization_artifacts"]:
            artifact_id = str(row["artifact_id"])
            if artifact_id not in artifacts:
                artifacts[artifact_id] = {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "source_rows",
                        "qualified_seconds",
                        "minimum_second_start",
                        "maximum_second_start",
                        "minimum_available_at",
                        "maximum_available_at",
                    }
                }
                artifacts[artifact_id].update({"source_rows": 0, "qualified_seconds": 0})
            aggregate = artifacts[artifact_id]
            stable = {
                key: value
                for key, value in row.items()
                if key in aggregate and key not in {"source_rows", "qualified_seconds"}
            }
            if any(aggregate[key] != value for key, value in stable.items()):
                raise RuntimeError(f"L2 artifact lineage changed: {artifact_id}")
            aggregate["source_rows"] += int(row["source_rows"])
            aggregate["qualified_seconds"] += int(row["qualified_seconds"])
    return [artifacts[key] for key in sorted(artifacts)]


def _core_totals(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "rows",
        "markets",
        "complete_300_row_markets",
        "incomplete_markets",
        "markets_with_final_price",
        "up_rows",
        "down_rows",
    )
    return {key: sum(int(record["summary"][key]) for record in records) for key in keys}


def _l2_totals(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        "rows": sum(int(record["summary"]["rows"]) for record in records),
        "qualified_seconds": sum(int(record["summary"]["qualified_seconds"]) for record in records),
        "partitions": len(records),
    }


def _candle_totals(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    support = [record for record in records if record.get("role") == "lookback_support"]
    benchmark = [record for record in records if record.get("role") == "benchmark_interval"]
    return {
        "rows": sum(int(record["summary"]["rows"]) for record in records),
        "fully_closed_candles": sum(
            int(record["summary"]["fully_closed_candles"]) for record in records
        ),
        "lookback_support_rows": sum(int(record["summary"]["rows"]) for record in support),
        "benchmark_interval_rows": sum(int(record["summary"]["rows"]) for record in benchmark),
    }


def _execution_totals(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "rows",
        "markets",
        "decision_keys",
        "snapshot_rows",
        "missing_snapshot_rows",
        "strict_both_side_eligible_5_rows",
        "strict_both_side_eligible_10_rows",
    )
    totals: dict[str, Any] = {
        key: sum(int(record["summary"][key]) for record in records) for key in keys
    }
    by_scenario: dict[str, dict[str, int]] = {}
    for record in records:
        for scenario_key, summary in record["summary"]["by_scenario"].items():
            aggregate = by_scenario.setdefault(scenario_key, {key: 0 for key in summary})
            for key, value in summary.items():
                aggregate[key] += int(value)
    totals["by_scenario"] = by_scenario
    return totals


def _validate_completed_cache(
    settings: _ExtractSettings,
    contract: dict[str, Any],
    manifest: dict[str, Any],
) -> SourceCache:
    expected = {
        "schema_version": SOURCE_CACHE_SCHEMA_VERSION,
        "immutable": True,
        "paper_only": True,
        "contract": contract,
    }
    _require_contract(manifest, expected, "source cache")
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise TypeError("source cache manifest has no sources")
    expected_names = {"core", "l2", "candles", "execution"}
    if set(sources) != expected_names:
        raise RuntimeError("source cache manifest source set changed")

    paths: dict[str, tuple[Path, ...]] = {}
    partition_records: dict[str, list[dict[str, Any]]] = {}
    source_contracts = {
        "core": (
            settings.core_source_sql,
            CORE_SOURCE_SCHEMA_VERSION,
            CORE_SOURCE_SCHEMA,
        ),
        "l2": (
            settings.l2_source_sql,
            L2_SOURCE_SCHEMA_VERSION,
            L2_SOURCE_SCHEMA,
        ),
        "candles": (
            settings.candles_source_sql,
            CANDLE_SOURCE_SCHEMA_VERSION,
            CANDLE_SOURCE_SCHEMA,
        ),
        "execution": (
            settings.execution_source_sql,
            EXECUTION_STRESS_SOURCE_SCHEMA_VERSION,
            EXECUTION_STRESS_SCHEMA,
        ),
    }
    for source in ("core", "l2", "candles", "execution"):
        partitions = sources[source].get("partitions")
        if not isinstance(partitions, list):
            raise TypeError(f"source cache has no {source} partitions")
        source_paths: list[Path] = []
        source_records: list[dict[str, Any]] = []
        query_path, schema_version, schema = source_contracts[source]
        source_dir = settings.source_root / source
        for record in partitions:
            if not isinstance(record, dict):
                raise TypeError(f"source cache has an invalid {source} partition")
            path = _safe_cache_path(settings.source_root, record.get("path"))
            if path.parent != source_dir:
                raise RuntimeError(f"source cache {source} partition is misplaced")
            observed_sha256 = file_sha256(path) if path.is_file() else None
            if observed_sha256 is None or observed_sha256 != record.get("sha256"):
                raise RuntimeError(f"immutable source partition hash mismatch: {path.name}")
            source_paths.append(path)
            sidecar = _safe_cache_path(settings.source_root, record.get("record_path"))
            if sidecar != source_dir / f"{path.name}.json":
                raise RuntimeError(f"source cache {source} sidecar is misplaced")
            if not sidecar.is_file() or file_sha256(sidecar) != record.get("record_sha256"):
                raise RuntimeError(f"immutable source partition record mismatch: {sidecar.name}")
            sidecar_record = json.loads(sidecar.read_text())
            if not isinstance(sidecar_record, dict):
                raise TypeError(f"source cache has an invalid {source} sidecar")
            sidecar_contract = {
                "record_schema_version": PARTITION_RECORD_SCHEMA_VERSION,
                "source": source,
                "range_start": record.get("range_start"),
                "range_end": record.get("range_end"),
                "path": path.name,
                "query_sha256": file_sha256(query_path),
                "source_schema_version": schema_version,
                "source_schema_sha256": _schema_sha256(schema),
            }
            if "role" in record:
                sidecar_contract["role"] = record["role"]
            if source == "execution":
                sidecar_contract.update(
                    {
                        "champion_process_sha256": file_sha256(settings.champion_process),
                        "snapshot_schema_version": EXECUTION_SNAPSHOT_SCHEMA_VERSION,
                        "scenarios": [
                            _execution_scenario_record(scenario)
                            for scenario in settings.execution_scenarios
                        ],
                    }
                )
            _require_contract(
                sidecar_record,
                sidecar_contract,
                f"{source} {path.name} sidecar",
            )
            summary = sidecar_record.get("summary")
            if (
                sidecar_record.get("sha256") != observed_sha256
                or not isinstance(summary, dict)
                or int(summary.get("rows", -1)) != record.get("rows")
            ):
                raise RuntimeError(f"{source} {path.name} sidecar content changed")
            if source == "l2":
                _validate_l2_partition_lineage(sidecar_record)
            source_records.append(sidecar_record)
        paths[source] = tuple(source_paths)
        partition_records[source] = source_records

    _validate_partition_names(paths)
    _validate_partition_ranges(partition_records)
    expected_totals = {
        "core": _core_totals(partition_records["core"]),
        "l2": _l2_totals(partition_records["l2"]),
        "candles": _candle_totals(partition_records["candles"]),
        "execution": _execution_totals(partition_records["execution"]),
    }
    for source, totals in expected_totals.items():
        if sources[source].get("totals") != totals:
            raise RuntimeError(f"source cache {source} totals changed")
    if sources["l2"].get("materialization_artifacts") != (
        _aggregate_l2_artifacts(partition_records["l2"])
    ):
        raise RuntimeError("source cache L2 artifact lineage changed")
    expected_scenarios = [
        _execution_scenario_record(scenario) for scenario in settings.execution_scenarios
    ]
    if sources["execution"].get("scenarios") != expected_scenarios:
        raise RuntimeError("source cache execution scenarios changed")
    if sources["execution"].get("snapshot_artifacts") != (
        _execution_artifacts(partition_records["execution"])
    ):
        raise RuntimeError("source cache execution artifact lineage changed")
    relation_path = settings.source_root / "l2" / "relation-lineage.json"
    if not relation_path.is_file() or json.loads(relation_path.read_text()) != sources["l2"].get(
        "relation_lineage"
    ):
        raise RuntimeError("spot-L2 relation lineage record mismatch")
    return SourceCache(
        core_source_files=paths["core"],
        l2_files=paths["l2"],
        candle_files=paths["candles"],
        execution_files=paths["execution"],
    )


def _validate_partition_names(paths: dict[str, tuple[Path, ...]]) -> None:
    daily_names = tuple(
        f"{batch_start.date().isoformat()}.parquet"
        for batch_start, _ in _daily_ranges(SOURCE_RANGE_START, SOURCE_RANGE_END)
    )
    for source in ("core", "l2", "execution"):
        if tuple(path.name for path in paths[source]) != daily_names:
            raise RuntimeError(f"{source} cache does not contain the exact daily range")
    candle_names = ("lookback-support.parquet", *daily_names)
    if tuple(path.name for path in paths["candles"]) != candle_names:
        raise RuntimeError("candle cache does not contain exact lookback and daily ranges")


def _validate_partition_ranges(
    records: dict[str, list[dict[str, Any]]],
) -> None:
    daily_ranges = [
        (batch_start.isoformat(), batch_end.isoformat())
        for batch_start, batch_end in _daily_ranges(SOURCE_RANGE_START, SOURCE_RANGE_END)
    ]
    for source in ("core", "l2", "execution"):
        observed = [
            (record.get("range_start"), record.get("range_end")) for record in records[source]
        ]
        if observed != daily_ranges:
            raise RuntimeError(f"{source} cache does not contain the exact daily ranges")
    support_start = SOURCE_RANGE_START - timedelta(minutes=CHAINLINK_HISTORY_MINUTES)
    expected_candles = [
        (
            support_start.isoformat(),
            SOURCE_RANGE_START.isoformat(),
            "lookback_support",
        ),
        *[
            (range_start, range_end, "benchmark_interval")
            for range_start, range_end in daily_ranges
        ],
    ]
    observed_candles = [
        (
            record.get("range_start"),
            record.get("range_end"),
            record.get("role"),
        )
        for record in records["candles"]
    ]
    if observed_candles != expected_candles:
        raise RuntimeError("candle cache does not contain the exact source ranges")


def _safe_cache_path(root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise TypeError("source cache contains an invalid path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("source cache path escapes its immutable root")
    return root / relative


def _require_contract(observed: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    mismatches = [key for key, value in expected.items() if observed.get(key) != value]
    if mismatches:
        raise RuntimeError(f"{label} contract mismatch: {', '.join(mismatches)}")


def _require_parquet_schema(path: Path, schema: pa.Schema) -> None:
    observed = pq.read_schema(path)
    if not observed.equals(schema, check_metadata=False):
        raise RuntimeError(f"{path.name} does not match its frozen Parquet schema")


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.to_string().encode()).hexdigest()


def _daily_ranges(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    cursor = start
    while cursor < end:
        batch_end = min(cursor + timedelta(days=1), end)
        yield cursor, batch_end
        cursor = batch_end


def _json_value(value: object) -> object:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value
