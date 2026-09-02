"""Fail-closed source readiness for the asymmetric-value training round."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .asymmetric_value_config import AsymmetricValueConfig
from .asymmetric_value_data import (
    EARLY_CAUSAL_ORACLE_FEATURES,
    ORACLE_MAXIMUM_AGE_SECONDS,
    ORACLE_MINIMUM_PROPAGATION_SECONDS,
    PRICE_MANIFEST_IDENTITY_EXCLUDES,
    price_manifest_identity_sha256,
)
from .core_config import (
    CORE_ORACLE_SOURCE_CONTRACT,
    CORE_SOURCE_CONTRACT,
    CoreTrainingConfig,
    load_core_config,
)
from .core_execution import (
    EXECUTION_EVIDENCE_CONTRACT,
    EXECUTION_EVIDENCE_SCHEMA_VERSION,
    LEGACY_SNAPSHOT_SCHEMA_VERSION,
    ExecutionEvidenceConfig,
    load_execution_evidence_manifest,
)
from .core_extract import (
    CORE_ORACLE_ROUND_SCHEMA_VERSION,
    CORE_ORACLE_SOURCE_SCHEMA_VERSION,
    CORE_SOURCE_SCHEMA_VERSION,
    POLYGON_CHAINLINK_BTCUSD_PROXY,
    configure_read_only_connection,
    core_source_schema_version,
    database_connection,
    file_sha256,
    load_core_manifest,
    write_json_atomic,
    write_json_exclusive,
)
from .spot_l2_chainlink_extract import (
    CANDLE_SOURCE_SCHEMA_VERSION,
    L2_INFORMATION_COLUMNS,
    L2_MATERIALIZATION_CONTRACTS,
    L2_SOURCE_SCHEMA_VERSION,
)

READINESS_SCHEMA_VERSION = "btc-asymmetric-training-readiness-v4"
READINESS_RANGE_START = datetime(2026, 4, 14, tzinfo=UTC)
READINESS_RANGE_END = datetime(2026, 8, 2, tzinfo=UTC)
EXPECTED_DAILY_MARKETS = 288
EXPECTED_DAILY_ONE_SECOND_ROWS = 86_400
EXPECTED_DAILY_CANDLES = 1_440
EXPECTED_DAILY_PMXT_ARTIFACT_ROWS = 345_600
MINIMUM_OPENING_BOUNDARY_COVERAGE = 0.95
MINIMUM_PMXT_HOURLY_COVERAGE = 0.98
MINIMUM_L2_SECOND_COVERAGE = 0.70
RECENT_SOURCE_START = datetime(2026, 7, 16, tzinfo=UTC)
MINIMUM_RECENT_L2_SECOND_COVERAGE = 0.99
PMXT_PROVIDER = "pmxt_v2_execution_snapshots"
PMXT_INGESTER = "polymarket_btc_five_minute_execution_snapshots"
L2_INGESTER = "binance_spot_btcusdt_l2_one_second_features"
ORACLE_CACHE_SCHEMA_VERSION = "btc-asymmetric-value-early-oracle-v3"
DEVELOPMENT_ORACLE_CACHE = "development-oracle-propagation-2s.parquet"
EVALUATION_ORACLE_CACHE = "evaluation-oracle-propagation-2s.parquet"

NEW_DAY_READINESS_SCHEMA_VERSION = "btc-asymmetric-new-day-readiness-v2"
NEW_DAY_READINESS_CONTRACT = "early-no-cross-day-source-readiness-v2"
PLANNED_VALIDATION_START = datetime(2026, 8, 10, tzinfo=UTC)
PLANNED_VALIDATION_END = datetime(2026, 8, 20, tzinfo=UTC)
QUARANTINE_START = datetime(2026, 8, 2, tzinfo=UTC)
QUARANTINE_END = PLANNED_VALIDATION_START
NEW_DAY_VALIDATION_DAYS = 10
ASYMMETRIC_PREDICTION_SECONDS = (
    *range(1, 60),
    *range(60, 241, 5),
)
EXPECTED_DAILY_PREDICTION_ROWS = (
    EXPECTED_DAILY_MARKETS * len(ASYMMETRIC_PREDICTION_SECONDS)
)
SPOT_L2_CURRENT_MATERIALIZATION_END = datetime(2026, 8, 2, tzinfo=UTC)
CRYPTOHFT_L2_MATERIALIZATION_CONTRACT = (
    "cryptohft-binance-spot-btcusdt-l2-features-v1"
)
COINAPI_L2_MATERIALIZATION_CONTRACT = (
    "coinapi-binance-spot-btcusdt-l2-snapshots-v1"
)
COINAPI_DIRECT_MATERIALIZER = "scripts/materialize-coinapi-binance-spot-l2.mjs"


@dataclass(frozen=True)
class NewDayReadinessContract:
    """Frozen source-only contract for untouched cross-day validation."""

    source_path: Path
    contract: str
    frozen_at: datetime
    quarantine_start: datetime
    quarantine_end: datetime
    validation_start: datetime
    validation_end: datetime
    expected_daily_markets: int
    minimum_opening_boundary_coverage: float
    minimum_source_grid_coverage: float
    minimum_l2_second_coverage: float
    minimum_primary_l2_provider_fraction: float
    minimum_candidate_grid_coverage: float
    external_archive_mount: Path
    pmxt_cache_root: Path
    spot_l2_archive_root: Path
    spot_l2_sentinel: str
    spot_l2_sentinel_content: str
    coinapi_archive_root: Path


_NEW_DAY_INTEGER_FIELDS = (
    "labeled_markets",
    "up_markets",
    "down_markets",
    "opening_boundary_markets",
    "final_price_markets",
    "reference_causality_violations",
    "binance_one_second_rows",
    "binance_qualified_seconds",
    "binance_causality_violations",
    "pmxt_completed_hours",
    "pmxt_artifact_rows",
    "pmxt_exact_grid_rows",
    "pmxt_exact_grid_keys",
    "pmxt_strict_grid_rows",
    "pmxt_causality_violations",
    "l2_rows",
    "l2_qualified_seconds",
    "l2_causality_violations",
    "l2_cryptohft_rows",
    "l2_coinapi_rows",
    "l2_huggingface_rows",
    "oracle_rounds",
    "oracle_causality_violations",
    "chainlink_candle_rows",
    "chainlink_candle_causality_violations",
)

_MATERIALIZATION_SOURCES = frozenset(
    {
        "official_outcome",
        "reference_facts",
        "binance_one_second",
        "binance_spot_l2",
        "polymarket_execution",
    }
)

CANONICAL_SOURCE_CONTRACT: dict[str, Any] = {
    "target": {
        "relation": "polymarket.btc_interval_markets",
        "field": "official_outcome",
        "allowed_values": ["up", "down"],
        "core_source_contract": CORE_SOURCE_CONTRACT,
        "core_source_schema_version": CORE_SOURCE_SCHEMA_VERSION,
        "oracle_arm_source_contract": CORE_ORACLE_SOURCE_CONTRACT,
        "oracle_arm_source_schema_version": CORE_ORACLE_SOURCE_SCHEMA_VERSION,
        "source_cache_policy": "distinct base-Core and Core-plus-Oracle caches",
    },
    "reference": {
        "relation": "polymarket.btc_market_reference_facts",
        "opening_boundary_role": "cohort_eligibility_only",
        "final_price_role": "audit_only",
    },
    "binance_one_second": {
        "relation": "polymarket.binance_one_second_klines",
        "symbol": "BTCUSDT",
        "provider_policy": "completed backfill artifact recorded daily",
    },
    "polymarket_execution": {
        "relation": "polymarket.btc_market_execution_snapshots",
        "artifact_relation": "polymarket.backfill_artifacts",
        "provider": PMXT_PROVIDER,
        "ingester_key": PMXT_INGESTER,
        "snapshot_schema_versions": [LEGACY_SNAPSHOT_SCHEMA_VERSION],
        "execution_schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
    },
    "polygon_chainlink_oracle": {
        "relation": "polymarket.polygon_chainlink_btcusd_oracle_rounds",
        "feed_proxy_address": POLYGON_CHAINLINK_BTCUSD_PROXY,
        "source_schema_version": CORE_ORACLE_ROUND_SCHEMA_VERSION,
        "provider_policy": "completed backfill artifact recorded daily",
    },
    "binance_spot_l2": {
        "training_relation": "polymarket.binance_spot_btcusdt_l2_training_features",
        "base_relation": "polymarket.binance_spot_btcusdt_l2_one_second_features",
        "ingester_key": L2_INGESTER,
        "source_schema_version": L2_SOURCE_SCHEMA_VERSION,
        "materialization_contracts": list(L2_MATERIALIZATION_CONTRACTS),
        "primary_materialization_contract": (
            CRYPTOHFT_L2_MATERIALIZATION_CONTRACT
        ),
        "minimum_primary_provider_fraction": 0.95,
        "coinapi_role": (
            "accepted canonical supplement; primary validation regime requires "
            "separate source-equivalence qualification"
        ),
        "information_dimension_count": len(L2_INFORMATION_COLUMNS),
        "information_columns": list(L2_INFORMATION_COLUMNS),
    },
    "chainlink_candles": {
        "relation": "polymarket.chainlink_btcusd_one_minute_candles",
        "symbol": "BTCUSD",
        "source_schema_version": CANDLE_SOURCE_SCHEMA_VERSION,
        "provider_policy": "completed backfill artifact recorded daily",
    },
    "intentionally_excluded": {
        "polymarket.chainlink_btcusd_archive_ticks": (
            "historical local receipt availability is not proven"
        ),
        "polymarket.binance_btcusdt_five_minute_open_interest": (
            "not included in the approved asymmetric-value candidate matrix"
        ),
        "market_data.binance_spot_btcusdt_aggregate_trades": (
            "insufficient history; one-second Core already includes taker flow"
        ),
        "polymarket.binance_btcusdt_l2_training_features": (
            "futures L2 must not substitute for the canonical spot L2 view"
        ),
    },
}

_SQL_CONTRACTS = {
    "btc-core-source.sql": (
        "polymarket.btc_interval_markets",
        "polymarket.btc_market_reference_facts",
        "polymarket.binance_one_second_klines",
        "polymarket.backfill_artifacts",
    ),
    "btc-core-oracle-source.sql": (
        "polymarket.polygon_chainlink_btcusd_oracle_rounds",
        "round.source_timestamp <= round.block_timestamp",
        "polymarket.backfill_artifacts",
    ),
    "btc-execution-evidence.sql": (
        "polymarket.btc_market_execution_snapshots",
        "polymarket.btc_interval_markets",
        "polymarket.backfill_artifacts",
        PMXT_INGESTER,
        "snapshot.schema_version = ANY",
    ),
    "btc-spot-l2-source.sql": (
        "polymarket.binance_spot_btcusdt_l2_training_features",
        "feature.available_at >= %(batch_start)s",
        "feature.available_at < %(batch_end)s",
    ),
    "btc-chainlink-one-minute-candles-source.sql": (
        "polymarket.chainlink_btcusd_one_minute_candles",
        "candle.close_timestamp = candle.open_timestamp + interval '1 minute'",
        "polymarket.backfill_artifacts",
    ),
    "btc-asymmetric-training-readiness.sql": (
        "polymarket.btc_interval_markets",
        "polymarket.btc_market_reference_facts",
        "polymarket.binance_one_second_klines",
        "polymarket.polygon_chainlink_btcusd_oracle_rounds",
        "polymarket.binance_spot_btcusdt_l2_training_features",
        "polymarket.chainlink_btcusd_one_minute_candles",
        PMXT_PROVIDER,
        LEGACY_SNAPSHOT_SCHEMA_VERSION,
    ),
}

_NEW_DAY_SQL_CONTRACT = (
    "polymarket.btc_interval_markets",
    "polymarket.btc_market_reference_facts",
    "polymarket.binance_one_second_klines",
    "polymarket.binance_spot_btcusdt_l2_training_features",
    "polymarket.binance_spot_btcusdt_l2_one_second_features",
    "polymarket.btc_market_execution_snapshots",
    "polymarket.backfill_artifacts",
    "polymarket.polygon_chainlink_btcusd_oracle_rounds",
    "polymarket.chainlink_btcusd_one_minute_candles",
    PMXT_PROVIDER,
    LEGACY_SNAPSHOT_SCHEMA_VERSION,
    CRYPTOHFT_L2_MATERIALIZATION_CONTRACT,
    COINAPI_L2_MATERIALIZATION_CONTRACT,
)


def load_new_day_readiness_contract(path: Path) -> NewDayReadinessContract:
    """Load the frozen, source-only cross-day readiness contract."""

    source_path = path.resolve()
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    values = raw["readiness"]
    archives = raw["external_archive"]
    contract = NewDayReadinessContract(
        source_path=source_path,
        contract=str(values["contract"]),
        frozen_at=_parse_utc_datetime(values["frozen_at"]),
        quarantine_start=_parse_utc_datetime(values["quarantine_start"]),
        quarantine_end=_parse_utc_datetime(values["quarantine_end"]),
        validation_start=_parse_utc_datetime(values["validation_start"]),
        validation_end=_parse_utc_datetime(values["validation_end"]),
        expected_daily_markets=int(values["expected_daily_markets"]),
        minimum_opening_boundary_coverage=float(
            values["minimum_opening_boundary_coverage"]
        ),
        minimum_source_grid_coverage=float(
            values["minimum_source_grid_coverage"]
        ),
        minimum_l2_second_coverage=float(
            values["minimum_l2_second_coverage"]
        ),
        minimum_primary_l2_provider_fraction=float(
            values["minimum_primary_l2_provider_fraction"]
        ),
        minimum_candidate_grid_coverage=float(
            values["minimum_candidate_grid_coverage"]
        ),
        external_archive_mount=Path(str(archives["mount_root"])),
        pmxt_cache_root=Path(str(archives["pmxt_cache_root"])),
        spot_l2_archive_root=Path(str(archives["spot_l2_archive_root"])),
        spot_l2_sentinel=str(archives["spot_l2_sentinel"]),
        spot_l2_sentinel_content=str(archives["spot_l2_sentinel_content"]),
        coinapi_archive_root=Path(str(archives["coinapi_archive_root"])),
    )
    validate_new_day_readiness_contract(contract)
    return contract


def validate_new_day_readiness_contract(
    contract: NewDayReadinessContract,
) -> None:
    """Reject a mutable, contaminated, or underspecified validation window."""

    if contract.contract != NEW_DAY_READINESS_CONTRACT:
        raise ValueError("new-day readiness contract identity changed")
    for name, value in (
        ("quarantine_start", contract.quarantine_start),
        ("quarantine_end", contract.quarantine_end),
        ("validation_start", contract.validation_start),
        ("validation_end", contract.validation_end),
    ):
        if value.utcoffset() != timedelta(0) or value.time() != datetime.min.time():
            raise ValueError(f"{name} must be aligned to a UTC day")
    if contract.frozen_at.utcoffset() != timedelta(0):
        raise ValueError("frozen_at must be UTC")
    if (
        contract.quarantine_start != QUARANTINE_START
        or contract.quarantine_end != QUARANTINE_END
    ):
        raise ValueError("new-day readiness quarantine must remain [2026-08-02, 2026-08-10)")
    if contract.validation_end - contract.validation_start != timedelta(
        days=NEW_DAY_VALIDATION_DAYS
    ):
        raise ValueError("new-day validation must contain exactly ten full UTC days")
    if contract.validation_start < PLANNED_VALIDATION_START:
        raise ValueError("new-day validation cannot precede 2026-08-10")
    if contract.frozen_at >= contract.validation_start:
        raise ValueError(
            "validation was not frozen before its first UTC day; create a shifted frozen config"
        )
    if (
        contract.validation_start == PLANNED_VALIDATION_START
        and contract.validation_end != PLANNED_VALIDATION_END
    ):
        raise ValueError("planned validation must remain [2026-08-10, 2026-08-20)")
    if contract.expected_daily_markets != EXPECTED_DAILY_MARKETS:
        raise ValueError("new-day readiness requires 288 five-minute markets per UTC day")
    for name, value in (
        (
            "minimum_opening_boundary_coverage",
            contract.minimum_opening_boundary_coverage,
        ),
        ("minimum_source_grid_coverage", contract.minimum_source_grid_coverage),
        ("minimum_l2_second_coverage", contract.minimum_l2_second_coverage),
        (
            "minimum_primary_l2_provider_fraction",
            contract.minimum_primary_l2_provider_fraction,
        ),
        (
            "minimum_candidate_grid_coverage",
            contract.minimum_candidate_grid_coverage,
        ),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be inside (0, 1]")
    if contract.minimum_source_grid_coverage != 0.90:
        raise ValueError("new-day source-grid coverage must remain 90%")
    if contract.minimum_l2_second_coverage != 0.95:
        raise ValueError("new-day spot-L2 daily coverage must remain 95%")
    if contract.minimum_primary_l2_provider_fraction != 0.95:
        raise ValueError("new-day primary spot-L2 provider share must remain 95%")
    if contract.minimum_candidate_grid_coverage != 0.70:
        raise ValueError("new-day strict candidate-grid coverage must remain 70%")
    for name, value in (
        ("external archive mount", contract.external_archive_mount),
        ("PMXT cache root", contract.pmxt_cache_root),
        ("spot-L2 archive root", contract.spot_l2_archive_root),
        ("CoinAPI archive root", contract.coinapi_archive_root),
    ):
        if not value.is_absolute():
            raise ValueError(f"{name} must be absolute")
    if not contract.spot_l2_sentinel.strip() or Path(
        contract.spot_l2_sentinel
    ).name != contract.spot_l2_sentinel:
        raise ValueError("spot-L2 sentinel must be one relative file name")
    if contract.spot_l2_sentinel_content != "cryptohft-btcusdt-l2-archive-v1\n":
        raise ValueError("spot-L2 sentinel content contract changed")
    if contract.coinapi_archive_root != contract.spot_l2_archive_root:
        raise ValueError("CoinAPI and primary spot-L2 archives must share the durable root")


def collect_new_day_database_inventory(
    package_root: Path,
    contract: NewDayReadinessContract,
    *,
    connection: Any | None = None,
) -> list[dict[str, Any]]:
    """Collect one bounded row per requested UTC day from canonical relations."""

    validate_new_day_readiness_contract(contract)
    query_path = package_root / "sql" / "btc-asymmetric-new-day-readiness.sql"
    query = query_path.read_text()
    owned_connection = connection is None
    active = connection if connection is not None else database_connection()
    try:
        if owned_connection:
            configure_read_only_connection(active)
        with active.cursor() as cursor:
            cursor.execute(
                query,
                {
                    "range_start": contract.validation_start,
                    "range_end": contract.validation_end,
                    "oracle_feed_proxy_address": POLYGON_CHAINLINK_BTCUSD_PROXY,
                },
            )
            names = [
                column.name if hasattr(column, "name") else column[0]
                for column in cursor.description
            ]
            rows = [
                _json_ready(dict(zip(names, row, strict=True)))
                for row in cursor.fetchall()
            ]
    finally:
        if owned_connection:
            active.close()
    return rows


def _new_day_metrics(
    source_row: dict[str, Any],
) -> dict[str, Any]:
    row = dict(source_row)
    labeled = int(row["labeled_markets"])
    opening_coverage = (
        int(row["opening_boundary_markets"]) / labeled if labeled else 0.0
    )
    binance_coverage = (
        int(row["binance_qualified_seconds"]) / EXPECTED_DAILY_ONE_SECOND_ROWS
    )
    l2_coverage = (
        int(row["l2_qualified_seconds"]) / EXPECTED_DAILY_ONE_SECOND_ROWS
    )
    l2_total_provider_rows = sum(
        int(row[field])
        for field in (
            "l2_cryptohft_rows",
            "l2_coinapi_rows",
            "l2_huggingface_rows",
        )
    )
    primary_l2_provider_fraction = (
        int(row["l2_cryptohft_rows"]) / l2_total_provider_rows
        if l2_total_provider_rows
        else 0.0
    )
    pmxt_coverage = int(row["pmxt_exact_grid_keys"]) / (
        EXPECTED_DAILY_PREDICTION_ROWS
    )
    strict_coverage = int(row["pmxt_strict_grid_rows"]) / (
        EXPECTED_DAILY_PREDICTION_ROWS
    )
    row.update(
        {
            "expected_prediction_rows": EXPECTED_DAILY_PREDICTION_ROWS,
            "opening_boundary_coverage": opening_coverage,
            "binance_second_coverage": binance_coverage,
            "l2_second_coverage": l2_coverage,
            "l2_total_provider_rows": l2_total_provider_rows,
            "primary_l2_provider_fraction": primary_l2_provider_fraction,
            "pmxt_exact_grid_coverage": pmxt_coverage,
            "strict_candidate_grid_coverage": strict_coverage,
            "joint_source_grid_coverage": min(
                binance_coverage,
                l2_coverage,
                pmxt_coverage,
            ),
        }
    )
    return row


def _new_day_nonblocking_inventory(
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "date": str(row["date"]),
            "source": source,
            "blocking": False,
            "rows": int(row[count_name]),
            "causality_violations": int(row[violations_name]),
        }
        for source, count_name, violations_name in (
            (
                "polygon_chainlink_oracle",
                "oracle_rounds",
                "oracle_causality_violations",
            ),
            (
                "chainlink_candles",
                "chainlink_candle_rows",
                "chainlink_candle_causality_violations",
            ),
        )
    ]


def _new_day_closed_blockers(
    row: dict[str, Any],
    contract: NewDayReadinessContract,
) -> list[dict[str, Any]]:
    day = str(row["date"])
    blockers: list[dict[str, Any]] = []

    def expect(
        passed: bool,
        source: str,
        code: str,
        observed: Any,
        required: Any,
    ) -> None:
        if not passed:
            blockers.append(
                {
                    "date": day,
                    "source": source,
                    "code": code,
                    "observed": observed,
                    "required": required,
                }
            )

    labeled = int(row["labeled_markets"])
    outcomes = {"up": int(row["up_markets"]), "down": int(row["down_markets"])}
    expect(
        labeled == contract.expected_daily_markets,
        "official_outcome",
        "incomplete_daily_labels",
        labeled,
        contract.expected_daily_markets,
    )
    expect(
        all(outcomes.values()),
        "official_outcome",
        "missing_outcome_side",
        outcomes,
        "both sides",
    )
    expect(
        float(row["opening_boundary_coverage"])
        >= contract.minimum_opening_boundary_coverage,
        "reference_facts",
        "opening_boundary_coverage",
        row["opening_boundary_coverage"],
        contract.minimum_opening_boundary_coverage,
    )
    for source, prefix in (
        ("reference_facts", "reference"),
        ("binance_one_second", "binance"),
        ("binance_spot_l2", "l2"),
        ("polymarket_execution", "pmxt"),
    ):
        value = int(row[f"{prefix}_causality_violations"])
        expect(value == 0, source, f"{prefix}_causality_violation", value, 0)
    expect(
        int(row["binance_one_second_rows"])
        == int(row["binance_qualified_seconds"]),
        "binance_one_second",
        "duplicate_one_second_rows",
        int(row["binance_one_second_rows"]),
        int(row["binance_qualified_seconds"]),
    )
    expect(
        int(row["binance_qualified_seconds"]) == EXPECTED_DAILY_ONE_SECOND_ROWS,
        "binance_one_second",
        "one_second_coverage",
        int(row["binance_qualified_seconds"]),
        EXPECTED_DAILY_ONE_SECOND_ROWS,
    )
    expect(
        int(row["l2_rows"]) == int(row["l2_qualified_seconds"]),
        "binance_spot_l2",
        "duplicate_spot_l2_seconds",
        int(row["l2_rows"]),
        int(row["l2_qualified_seconds"]),
    )
    l2_contracts = set(row.get("l2_materialization_contracts", []))
    allowed_l2 = set(L2_MATERIALIZATION_CONTRACTS)
    expect(
        not int(row["l2_rows"])
        or bool(l2_contracts)
        and l2_contracts.issubset(allowed_l2),
        "binance_spot_l2",
        "unexpected_spot_l2_lineage",
        sorted(l2_contracts),
        sorted(allowed_l2),
    )
    expect(
        float(row["l2_second_coverage"])
        >= contract.minimum_l2_second_coverage,
        "binance_spot_l2",
        "spot_l2_coverage",
        row["l2_second_coverage"],
        contract.minimum_l2_second_coverage,
    )
    expect(
        int(row["l2_total_provider_rows"]) == int(row["l2_rows"]),
        "binance_spot_l2",
        "spot_l2_provider_row_reconciliation",
        int(row["l2_total_provider_rows"]),
        int(row["l2_rows"]),
    )
    expect(
        float(row["primary_l2_provider_fraction"])
        >= contract.minimum_primary_l2_provider_fraction,
        "binance_spot_l2",
        "spot_l2_primary_provider_fraction",
        row["primary_l2_provider_fraction"],
        contract.minimum_primary_l2_provider_fraction,
    )
    for field, code, required in (
        ("pmxt_completed_hours", "pmxt_completed_hours", 24),
        (
            "pmxt_artifact_rows",
            "pmxt_artifact_rows",
            EXPECTED_DAILY_PMXT_ARTIFACT_ROWS,
        ),
    ):
        value = int(row[field])
        expect(value == required, "polymarket_execution", code, value, required)
    for field, code, required in (
        ("pmxt_providers", "unexpected_pmxt_provider", [PMXT_PROVIDER]),
        (
            "pmxt_schema_versions",
            "unexpected_pmxt_schema",
            [LEGACY_SNAPSHOT_SCHEMA_VERSION],
        ),
    ):
        observed = list(row.get(field, []))
        expect(
            observed in ([], required),
            "polymarket_execution",
            code,
            observed,
            required,
        )
    expect(
        int(row["pmxt_exact_grid_rows"]) == int(row["pmxt_exact_grid_keys"]),
        "polymarket_execution",
        "duplicate_pmxt_grid_keys",
        int(row["pmxt_exact_grid_rows"]),
        int(row["pmxt_exact_grid_keys"]),
    )
    for field, code, required in (
        (
            "pmxt_exact_grid_coverage",
            "pmxt_exact_grid_coverage",
            contract.minimum_source_grid_coverage,
        ),
        (
            "strict_candidate_grid_coverage",
            "pmxt_strict_candidate_grid_coverage",
            contract.minimum_candidate_grid_coverage,
        ),
    ):
        observed = float(row[field])
        expect(
            observed >= required,
            "polymarket_execution",
            code,
            observed,
            required,
        )
    return blockers


def assess_new_day_database_inventory(
    rows: Sequence[dict[str, Any]],
    contract: NewDayReadinessContract,
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Assess mandatory source grids without making Oracle/candles blocking."""

    validate_new_day_readiness_contract(contract)
    assessment_time = observed_at or datetime.now(UTC)
    if assessment_time.utcoffset() != timedelta(0):
        raise ValueError("new-day readiness assessment time must be UTC")
    expected_dates = _date_strings(
        contract.validation_start,
        contract.validation_end,
    )
    if [str(row.get("date")) for row in rows] != expected_dates:
        raise RuntimeError(
            "new-day inventory does not contain the exact requested UTC days"
        )
    blockers: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []

    for source_row in rows:
        missing = [
            name for name in _NEW_DAY_INTEGER_FIELDS if name not in source_row
        ]
        if missing:
            raise RuntimeError(
                "new-day readiness row is missing fields: " + ", ".join(missing)
            )
        row = _new_day_metrics(dict(source_row))
        day = str(row["date"])
        diagnostics.extend(_new_day_nonblocking_inventory(row))

        day_end = datetime.fromisoformat(day).replace(tzinfo=UTC) + timedelta(
            days=1
        )
        if assessment_time < day_end:
            blockers.append(
                {
                    "date": day,
                    "source": "daily_cohort",
                    "code": "not_yet_available",
                    "observed": assessment_time.isoformat(),
                    "required": day_end.isoformat(),
                }
            )
            row["availability_status"] = "not_yet_available"
            row["daily_inventory_sha256"] = _canonical_sha256(row)
            daily.append(row)
            continue
        row["availability_status"] = "closed"
        blockers.extend(_new_day_closed_blockers(row, contract))
        row["daily_inventory_sha256"] = _canonical_sha256(row)
        daily.append(row)

    materialization_sources = sorted(
        {
            str(item["source"])
            for item in blockers
            if item["source"] in _MATERIALIZATION_SOURCES
        }
    )
    return {
        "ready": not blockers,
        "days": len(daily),
        "mandatory_blockers": blockers,
        "required_materialization_sources": materialization_sources,
        "nonblocking_source_inventory": diagnostics,
        "daily": daily,
        "daily_inventory_sha256": _canonical_sha256(daily),
        "checks": {
            "exact_ten_days": len(daily) == NEW_DAY_VALIDATION_DAYS,
            "assessment_time": assessment_time.isoformat(),
            "not_yet_available_days": sum(
                row["availability_status"] == "not_yet_available"
                for row in daily
            ),
            "oracle_blocking": False,
            "chainlink_candles_blocking": False,
            "polymarket_proxy_prices_used": False,
            "canonical_spot_l2_only": True,
        },
    }


def inspect_external_archive_status(
    contract: NewDayReadinessContract,
    *,
    required_materialization_sources: Sequence[str],
) -> dict[str, Any]:
    """Expose an unmounted SSD without creating fallback directories."""

    required = set(required_materialization_sources)
    needs_pmxt = "polymarket_execution" in required
    needs_l2 = "binance_spot_l2" in required
    mount_directory_present = contract.external_archive_mount.is_dir()
    mount_present = contract.external_archive_mount.is_mount()
    pmxt_present = contract.pmxt_cache_root.is_dir()
    l2_present = contract.spot_l2_archive_root.is_dir()
    coinapi_present = contract.coinapi_archive_root.is_dir()
    sentinel_path = contract.spot_l2_archive_root / contract.spot_l2_sentinel
    sentinel_present = sentinel_path.is_file()
    sentinel_content_valid = False
    if sentinel_present:
        try:
            sentinel_content_valid = (
                sentinel_path.read_text() == contract.spot_l2_sentinel_content
            )
        except OSError:
            sentinel_content_valid = False
    if not (needs_pmxt or needs_l2):
        status = "not_required_database_sources_ready"
    elif not mount_present:
        status = "blocked_archive_unmounted"
    elif needs_l2 and (
        not l2_present or not sentinel_present or not sentinel_content_valid
    ):
        status = "blocked_primary_spot_l2_archive_contract"
    elif needs_pmxt and not pmxt_present:
        status = "blocked_pmxt_cache_path_missing"
    else:
        status = "available_for_materialization"
    return {
        "status": status,
        "required_for": sorted(required & {"polymarket_execution", "binance_spot_l2"}),
        "mount_root": str(contract.external_archive_mount),
        "mount_directory_present": mount_directory_present,
        "mount_present": mount_present,
        "pmxt_cache_root": str(contract.pmxt_cache_root),
        "pmxt_cache_present": pmxt_present,
        "spot_l2_archive_root": str(contract.spot_l2_archive_root),
        "spot_l2_archive_present": l2_present,
        "spot_l2_sentinel": str(sentinel_path),
        "spot_l2_sentinel_present": sentinel_present,
        "spot_l2_sentinel_content_valid": sentinel_content_valid,
        "coinapi_archive_root": str(contract.coinapi_archive_root),
        "coinapi_archive_present": coinapi_present,
        "coinapi_direct_materializer": COINAPI_DIRECT_MATERIALIZER,
        "coinapi_materialization_contract": COINAPI_L2_MATERIALIZATION_CONTRACT,
        "coinapi_provider_regime_status": (
            "unqualified_for_primary_validation_provider"
        ),
        "fallback_directory_created": False,
    }


def prepare_new_day_training_readiness(
    contract: NewDayReadinessContract,
    *,
    package_root: Path,
    output_dir: Path,
    connection: Any | None = None,
    observed_at: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write a fail-closed preflight for the frozen new-day source cohort."""

    validate_new_day_readiness_contract(contract)
    sql_contract = _validate_new_day_sql_contract(package_root)
    rows = collect_new_day_database_inventory(
        package_root,
        contract,
        connection=connection,
    )
    database = assess_new_day_database_inventory(
        rows,
        contract,
        observed_at=observed_at,
    )
    l2_ready = bool(database["daily"]) and all(
        float(row["l2_second_coverage"])
        >= contract.minimum_l2_second_coverage
        and float(row["primary_l2_provider_fraction"])
        >= contract.minimum_primary_l2_provider_fraction
        for row in database["daily"]
    )
    spot_l2_upstream = {
        "status": (
            "materialization_path_available_requires_provider_regime_qualification"
            if not l2_ready
            and contract.validation_start >= SPOT_L2_CURRENT_MATERIALIZATION_END
            else "satisfied_by_canonical_database_rows"
        ),
        "current_range_end_exclusive": (
            SPOT_L2_CURRENT_MATERIALIZATION_END.isoformat()
        ),
        "required_range_end_exclusive": contract.validation_end.isoformat(),
        "request_validation_source": (
            "packages/polymarket-bot/src/ingestion/job.rs"
        ),
        "planner_source": (
            "packages/polymarket-bot/src/bin/binance-spot-l2-backfill-plan.rs"
        ),
        "primary_materialization_contract": (
            CRYPTOHFT_L2_MATERIALIZATION_CONTRACT
        ),
        "primary_path_status": (
            "blocked_by_current_rust_range_end"
            if not l2_ready
            else "satisfied_by_canonical_database_rows"
        ),
        "existing_no_rust_alternative": {
            "materializer": COINAPI_DIRECT_MATERIALIZER,
            "materialization_contract": COINAPI_L2_MATERIALIZATION_CONTRACT,
            "status": "available_but_provider_regime_unqualified",
            "maximum_role_without_separate_qualification": (
                "supplement_below_five_percent_of_daily_rows"
            ),
        },
        "minimum_primary_provider_fraction": (
            contract.minimum_primary_l2_provider_fraction
        ),
        "rust_change_in_this_scope": False,
    }
    archive_sources = set(database["required_materialization_sources"])
    if not l2_ready:
        archive_sources.add("binance_spot_l2")
    archive = inspect_external_archive_status(
        contract,
        required_materialization_sources=sorted(archive_sources),
    )
    not_yet_available = any(
        item["code"] == "not_yet_available"
        for item in database["mandatory_blockers"]
    )
    statuses = [
        archive["status"],
        spot_l2_upstream["status"],
    ]
    if database["ready"]:
        status = "ready"
    elif database["required_materialization_sources"] and (
        "blocked_archive_unmounted" in statuses
    ):
        status = "blocked_archive_unmounted"
    elif not_yet_available:
        status = "not_yet_available"
    elif (
        "materialization_path_available_requires_provider_regime_qualification"
        in statuses
    ):
        status = (
            "materialization_path_available_requires_provider_regime_qualification"
        )
    else:
        status = "blocked_source_materialization"
    payload: dict[str, Any] = {
        "schema_version": NEW_DAY_READINESS_SCHEMA_VERSION,
        "status": status,
        "blocking_statuses": sorted(
            {
                status,
                archive["status"],
                spot_l2_upstream["status"],
            }
            - {
                "available_for_materialization",
                "not_required_database_sources_ready",
                "satisfied_by_canonical_database_rows",
            }
        ),
        "ready": database["ready"],
        "contract": contract.contract,
        "contract_frozen_at": contract.frozen_at.isoformat(),
        "config_path": str(contract.source_path),
        "config_sha256": file_sha256(contract.source_path),
        "quarantine": {
            "range_start": contract.quarantine_start.isoformat(),
            "range_end": contract.quarantine_end.isoformat(),
            "usage": "research_feedback_only_never_unseen_validation_or_forward_proof",
        },
        "validation": {
            "range_start": contract.validation_start.isoformat(),
            "range_end": contract.validation_end.isoformat(),
            "range_semantics": "half_open_full_utc_days",
            "days": NEW_DAY_VALIDATION_DAYS,
            "planned_window_used": (
                contract.validation_start == PLANNED_VALIDATION_START
                and contract.validation_end == PLANNED_VALIDATION_END
            ),
        },
        "blocking_sources": [
            "official_outcome",
            "reference_facts",
            "binance_one_second",
            "binance_spot_l2",
            "polymarket_execution",
        ],
        "nonblocking_sources": [
            "polygon_chainlink_oracle",
            "chainlink_candles",
        ],
        "canonical_source_contract": CANONICAL_SOURCE_CONTRACT,
        "sql_contract": sql_contract,
        "database": database,
        "external_archive": archive,
        "upstream_materialization": {"binance_spot_l2": spot_l2_upstream},
        "checks": {
            "no_proxy_polymarket_prices": True,
            "oracle_and_candles_inventory_only": True,
            "quarantine_excluded_from_validation": True,
            "validation_frozen_before_first_day": (
                contract.frozen_at < contract.validation_start
            ),
            "no_rust_or_trading_pipeline_change": True,
        },
    }
    payload["readiness_identity_sha256"] = _readiness_identity_sha256(payload)
    payload["created_at"] = datetime.now(UTC).isoformat()
    payload["payload_sha256"] = _payload_sha256(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "asymmetric-new-day-readiness.json"
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing.get("ready") is True:
            existing_identity = _readiness_identity_sha256(existing)
            if (
                existing.get("schema_version") != NEW_DAY_READINESS_SCHEMA_VERSION
                or existing.get("payload_sha256") != _payload_sha256(existing)
                or existing.get("readiness_identity_sha256") != existing_identity
                or existing_identity != payload["readiness_identity_sha256"]
            ):
                raise RuntimeError(
                    "existing immutable new-day readiness seal changed"
                )
            return destination, existing
    write_json_atomic(destination, payload)
    return destination, payload


def prepare_asymmetric_training_readiness(
    config: AsymmetricValueConfig,
    *,
    output_dir: Path,
    connection: Any | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Validate every approved source and create or reuse its immutable readiness seal."""

    core_config = load_core_config(config.core_config)
    _validate_round_contract(config, core_config)
    sql_contract = _validate_sql_contracts(config.package_root)
    daily = collect_database_inventory(
        config.package_root,
        connection=connection,
    )
    database_summary = validate_database_inventory(daily)
    core_oracle = _validate_core_oracle_source(config, core_config)
    oracle_caches = _validate_oracle_feature_caches(config, core_config)
    external = _validate_external_source_cache(config)
    prices = _validate_price_manifests(config)

    payload: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "ready": True,
        "range_start": READINESS_RANGE_START.isoformat(),
        "range_end": READINESS_RANGE_END.isoformat(),
        "range_semantics": "half_open_utc",
        "canonical_source_contract": CANONICAL_SOURCE_CONTRACT,
        "sql_contract": sql_contract,
        "database_daily_inventory": daily,
        "database_summary": database_summary,
        "core_oracle_source": core_oracle,
        "oracle_feature_caches": oracle_caches,
        "external_source_cache": external,
        "execution_price_cache": prices,
        "external_ssd_required": False,
        "checks": {
            "all_110_days_present": True,
            "all_causality_checks_passed": True,
            "oracle_july_31_and_august_1_present": True,
            "oracle_cache_matches_current_source_inventory": True,
            "no_proxy_polymarket_prices": True,
        },
    }
    payload["readiness_identity_sha256"] = _readiness_identity_sha256(payload)
    payload["created_at"] = datetime.now(UTC).isoformat()
    payload["payload_sha256"] = _payload_sha256(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "asymmetric-training-readiness.json"
    if destination.exists():
        return destination, _reuse_identical_readiness_manifest(destination, payload)
    try:
        write_json_exclusive(destination, payload)
    except FileExistsError:
        return destination, _reuse_identical_readiness_manifest(destination, payload)
    return destination, payload


def collect_database_inventory(
    package_root: Path,
    *,
    connection: Any | None = None,
) -> list[dict[str, Any]]:
    """Read a bounded daily inventory from only the canonical relations."""

    query_path = package_root / "sql" / "btc-asymmetric-training-readiness.sql"
    query = query_path.read_text()
    owned_connection = connection is None
    active = connection if connection is not None else database_connection()
    try:
        if owned_connection:
            configure_read_only_connection(active)
        with active.cursor() as cursor:
            cursor.execute(
                query,
                {
                    "range_start": READINESS_RANGE_START,
                    "range_end": READINESS_RANGE_END,
                    "oracle_feed_proxy_address": POLYGON_CHAINLINK_BTCUSD_PROXY,
                },
            )
            names = [
                column.name if hasattr(column, "name") else column[0]
                for column in cursor.description
            ]
            rows = [_json_ready(dict(zip(names, row, strict=True))) for row in cursor.fetchall()]
    finally:
        if owned_connection:
            active.close()
    return rows


def validate_database_inventory(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fail closed on missing days, wrong lineage, or causal violations."""

    expected_dates = _date_strings(READINESS_RANGE_START, READINESS_RANGE_END)
    observed_dates = [str(row.get("date")) for row in rows]
    if observed_dates != expected_dates:
        raise RuntimeError("database readiness inventory does not contain the exact 110 UTC days")

    integer_fields = (
        "labeled_markets",
        "opening_boundary_markets",
        "final_price_markets",
        "binance_one_second_rows",
        "binance_causality_violations",
        "oracle_rounds",
        "oracle_causality_violations",
        "pmxt_completed_hours",
        "pmxt_artifact_rows",
        "l2_rows",
        "l2_qualified_seconds",
        "l2_causality_violations",
        "chainlink_candle_rows",
        "chainlink_candle_causality_violations",
    )
    for row in rows:
        missing = [name for name in integer_fields if name not in row]
        if missing:
            raise RuntimeError("database readiness row is missing fields: " + ", ".join(missing))
        if int(row["labeled_markets"]) != EXPECTED_DAILY_MARKETS:
            raise RuntimeError(f"official outcome coverage is incomplete on {row['date']}")
        if int(row["binance_one_second_rows"]) != EXPECTED_DAILY_ONE_SECOND_ROWS:
            raise RuntimeError(f"Binance one-second coverage is incomplete on {row['date']}")
        if int(row["oracle_rounds"]) <= 0:
            raise RuntimeError(f"Polygon Chainlink oracle coverage is empty on {row['date']}")
        if int(row["chainlink_candle_rows"]) != EXPECTED_DAILY_CANDLES:
            raise RuntimeError(f"Chainlink candle coverage is incomplete on {row['date']}")
        if int(row["l2_rows"]) != int(row["l2_qualified_seconds"]):
            raise RuntimeError(f"spot-L2 training view contains duplicate seconds on {row['date']}")
        violations = sum(
            int(row[name]) for name in integer_fields if name.endswith("causality_violations")
        )
        if violations:
            raise RuntimeError(f"causal source violations exist on {row['date']}")
        if int(row["pmxt_completed_hours"]):
            if list(row.get("pmxt_providers", [])) != [PMXT_PROVIDER]:
                raise RuntimeError(f"unexpected PMXT provider on {row['date']}")
            if list(row.get("pmxt_schema_versions", [])) != [LEGACY_SNAPSHOT_SCHEMA_VERSION]:
                raise RuntimeError(f"unexpected PMXT snapshot schema on {row['date']}")
        contracts = set(row.get("l2_materialization_contracts", []))
        if int(row["l2_rows"]) and (
            not contracts or not contracts.issubset(set(L2_MATERIALIZATION_CONTRACTS))
        ):
            raise RuntimeError(f"unexpected spot-L2 materialization lineage on {row['date']}")

    days = len(rows)
    labeled = sum(int(row["labeled_markets"]) for row in rows)
    opening = sum(int(row["opening_boundary_markets"]) for row in rows)
    pmxt_hours = sum(int(row["pmxt_completed_hours"]) for row in rows)
    l2_seconds = sum(int(row["l2_qualified_seconds"]) for row in rows)
    opening_rate = opening / labeled
    pmxt_rate = pmxt_hours / (days * 24)
    l2_rate = l2_seconds / (days * EXPECTED_DAILY_ONE_SECOND_ROWS)
    if opening_rate < MINIMUM_OPENING_BOUNDARY_COVERAGE:
        raise RuntimeError("opening-boundary cohort coverage is below 95%")
    if pmxt_rate < MINIMUM_PMXT_HOURLY_COVERAGE:
        raise RuntimeError("PMXT completed-hour coverage is below 98%")
    if l2_rate < MINIMUM_L2_SECOND_COVERAGE:
        raise RuntimeError("spot-L2 qualified-second coverage is below 70%")

    recent = [row for row in rows if str(row["date"]) >= RECENT_SOURCE_START.date().isoformat()]
    for row in recent:
        if int(row["pmxt_completed_hours"]) != 24:
            raise RuntimeError(f"recent PMXT coverage is not continuous on {row['date']}")
        if (
            int(row["l2_qualified_seconds"]) / EXPECTED_DAILY_ONE_SECOND_ROWS
            < MINIMUM_RECENT_L2_SECOND_COVERAGE
        ):
            raise RuntimeError(f"recent spot-L2 coverage is below 99% on {row['date']}")

    return {
        "days": days,
        "labeled_markets": labeled,
        "opening_boundary_markets": opening,
        "opening_boundary_coverage": opening_rate,
        "binance_one_second_rows": sum(int(row["binance_one_second_rows"]) for row in rows),
        "oracle_rounds": sum(int(row["oracle_rounds"]) for row in rows),
        "pmxt_completed_hours": pmxt_hours,
        "pmxt_completed_hour_coverage": pmxt_rate,
        "l2_qualified_seconds": l2_seconds,
        "l2_qualified_second_coverage": l2_rate,
        "chainlink_candle_rows": sum(int(row["chainlink_candle_rows"]) for row in rows),
    }


def oracle_source_inventory(
    source: Path,
    days: Iterable[date],
) -> dict[str, Any]:
    """Hash the exact daily Core+Oracle inputs used by Oracle feature fitting."""

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    expected = tuple(sorted(set(days)))
    for day in expected:
        raw_path = source / f"{day.isoformat()}.parquet"
        oracle_path = source / f"oracle-{day.isoformat()}.parquet"
        if not raw_path.is_file() or not oracle_path.is_file():
            missing.append(day.isoformat())
            continue
        oracle = pl.read_parquet(oracle_path)
        violations = oracle.filter(
            pl.col("oracle_source_timestamp") > pl.col("oracle_block_timestamp")
        ).height
        if violations:
            raise RuntimeError(f"oracle source contains causal violations: {oracle_path}")
        records.append(
            {
                "date": day.isoformat(),
                "raw_path": raw_path.name,
                "raw_sha256": file_sha256(raw_path),
                "oracle_path": oracle_path.name,
                "oracle_sha256": file_sha256(oracle_path),
                "oracle_rows": oracle.height,
                "causality_violations": 0,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "btc-asymmetric-value-oracle-source-inventory-v1",
        "expected_days": len(expected),
        "available_days": len(records),
        "missing_days": missing,
        "records": records,
    }
    payload["inventory_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def validate_oracle_cache_identity(
    metadata: dict[str, Any],
    source_inventory: dict[str, Any],
    *,
    cache_sha256: str,
) -> None:
    """Reject an Oracle feature cache built from any earlier source inventory."""

    expected = {
        "schema_version": ORACLE_CACHE_SCHEMA_VERSION,
        "source_inventory_sha256": source_inventory["inventory_sha256"],
        "minimum_propagation_seconds": ORACLE_MINIMUM_PROPAGATION_SECONDS,
        "maximum_age_seconds": ORACLE_MAXIMUM_AGE_SECONDS,
        "features": list(EARLY_CAUSAL_ORACLE_FEATURES),
        "sha256": cache_sha256,
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise RuntimeError(
            "Oracle feature cache is stale "
            f"({', '.join(mismatches)}); rebuild it from the current source inventory"
        )


def _validate_round_contract(
    config: AsymmetricValueConfig,
    core_config: CoreTrainingConfig,
) -> None:
    evaluation = config.evaluation
    round_end = config.policy.end if evaluation is None else evaluation.end
    if config.fit.start != READINESS_RANGE_START or round_end != READINESS_RANGE_END:
        raise ValueError("readiness requires exact [2026-04-14, 2026-08-02)")
    if core_config.data.range_start != READINESS_RANGE_START:
        raise ValueError("Core source range does not begin at the readiness boundary")
    if core_config.data.range_end != READINESS_RANGE_END:
        raise ValueError("Core source range does not end at the readiness boundary")
    if evaluation is None:
        if core_config.data.source_contract != CORE_SOURCE_CONTRACT:
            raise ValueError("target readiness requires the btc_core_v1 base source")
        if core_config.paths.source_data.resolve() == config.oracle_source.resolve():
            raise ValueError("target readiness requires distinct Core and Oracle caches")
    elif core_config.data.source_contract != CORE_ORACLE_SOURCE_CONTRACT:
        raise ValueError("legacy readiness requires the btc_core_oracle_v1 source cache")
    elif core_config.paths.source_data.resolve() != config.oracle_source.resolve():
        raise ValueError("legacy Core and Oracle must use the same source directory")


def _validate_sql_contracts(package_root: Path) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for filename, fragments in _SQL_CONTRACTS.items():
        path = package_root / "sql" / filename
        text = path.read_text()
        missing = [fragment for fragment in fragments if fragment not in text]
        if missing:
            raise RuntimeError(f"{filename} no longer matches its canonical relation contract")
        hashes[filename] = file_sha256(path)
    l2_text = (package_root / "sql" / "btc-spot-l2-source.sql").read_text()
    if "polymarket.binance_btcusdt_l2_training_features" in l2_text:
        raise RuntimeError("futures L2 relation cannot substitute for spot L2")
    return {"query_sha256": hashes, "relations": CANONICAL_SOURCE_CONTRACT}


def _validate_new_day_sql_contract(package_root: Path) -> dict[str, Any]:
    contract = _validate_sql_contracts(package_root)
    path = package_root / "sql" / "btc-asymmetric-new-day-readiness.sql"
    text = path.read_text()
    missing = [fragment for fragment in _NEW_DAY_SQL_CONTRACT if fragment not in text]
    if missing:
        raise RuntimeError(
            "btc-asymmetric-new-day-readiness.sql no longer matches its "
            "canonical relation contract"
        )
    return {
        **contract,
        "new_day_query_sha256": file_sha256(path),
    }


def _validate_core_oracle_source(
    config: AsymmetricValueConfig,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    scopes = ("pre_holdout",) if config.evaluation is None else ("pre_holdout", "holdout")
    core_manifests = {scope: load_core_manifest(core_config, scope) for scope in scopes}
    _validate_source_manifest_ranges(core_manifests, label="Core")
    if config.evaluation is None:
        oracle_config = _oracle_source_config(core_config, config.oracle_source)
        oracle_manifests = {scope: load_core_manifest(oracle_config, scope) for scope in scopes}
        _validate_source_manifest_ranges(oracle_manifests, label="Oracle arm")
    else:
        oracle_manifests = core_manifests
    expected_core = {
        f"{value}.parquet" for value in _date_strings(READINESS_RANGE_START, READINESS_RANGE_END)
    }
    expected_oracle = {
        f"oracle-{value}.parquet"
        for value in _date_strings(READINESS_RANGE_START, READINESS_RANGE_END)
    }
    core_records = [
        record for manifest in core_manifests.values() for record in manifest["partitions"]
    ]
    oracle_raw_records = [
        record for manifest in oracle_manifests.values() for record in manifest["partitions"]
    ]
    oracle_records = [
        record for manifest in oracle_manifests.values() for record in manifest["oracle_partitions"]
    ]
    observed_core = {record["path"] for record in core_records}
    observed_oracle_raw = {record["path"] for record in oracle_raw_records}
    observed_oracle = {record["path"] for record in oracle_records}
    if observed_core != expected_core or len(core_records) != 110:
        raise RuntimeError("Core source manifests do not contain exactly 110 daily partitions")
    if observed_oracle_raw != expected_core or len(oracle_raw_records) != 110:
        raise RuntimeError(
            "Oracle-arm source manifests do not contain exactly 110 raw Core partitions"
        )
    if observed_oracle != expected_oracle or len(oracle_records) != 110:
        raise RuntimeError("Oracle source manifests do not contain exactly 110 daily partitions")
    for record in (*core_records, *oracle_raw_records):
        if int(record["incomplete_markets"]) or int(record["rows"]) != 300 * int(record["markets"]):
            raise RuntimeError(f"incomplete raw Core source partition: {record['path']}")
    for record in oracle_records:
        if int(record["rows"]) <= 0 or int(record["causality_violations"]):
            raise RuntimeError(f"invalid Oracle source partition: {record['path']}")
    core_paths = {
        scope: core_config.paths.source_data / f"manifest-{scope}.json" for scope in core_manifests
    }
    oracle_paths = {
        scope: config.oracle_source / f"manifest-{scope}.json" for scope in oracle_manifests
    }
    return {
        "source_contract": core_config.data.source_contract,
        "core_source_schema_version": core_source_schema_version(core_config.data.source_contract),
        "oracle_source_contract": CORE_ORACLE_SOURCE_CONTRACT,
        "oracle_source_schema_version": CORE_ORACLE_SOURCE_SCHEMA_VERSION,
        "oracle_round_partition_schema_version": (
            CORE_ORACLE_ROUND_SCHEMA_VERSION
        ),
        "source_directory": str(core_config.paths.source_data.resolve()),
        "oracle_source_directory": str(config.oracle_source.resolve()),
        "source_scopes": list(scopes),
        "daily_core_partitions": len(core_records),
        "daily_oracle_raw_partitions": len(oracle_raw_records),
        "daily_oracle_partitions": len(oracle_records),
        "manifest_ranges": {
            scope: {
                "range_start": manifest["range_start"],
                "range_end": manifest["range_end"],
            }
            for scope, manifest in core_manifests.items()
        },
        "oracle_manifest_ranges": {
            scope: {
                "range_start": manifest["range_start"],
                "range_end": manifest["range_end"],
            }
            for scope, manifest in oracle_manifests.items()
        },
        "manifest_sha256": {scope: file_sha256(path) for scope, path in core_paths.items()},
        "oracle_manifest_sha256": {
            scope: file_sha256(path) for scope, path in oracle_paths.items()
        },
        "july_31_present": "oracle-2026-07-31.parquet" in observed_oracle,
        "august_1_present": "oracle-2026-08-01.parquet" in observed_oracle,
    }


def _oracle_source_config(
    core_config: CoreTrainingConfig,
    oracle_source: Path,
) -> CoreTrainingConfig:
    return replace(
        core_config,
        data=replace(
            core_config.data,
            source_contract=CORE_ORACLE_SOURCE_CONTRACT,
        ),
        paths=replace(core_config.paths, source_data=oracle_source),
    )


def _validate_source_manifest_ranges(
    manifests: dict[str, dict[str, Any]],
    *,
    label: str,
) -> None:
    if set(manifests) == {"pre_holdout"}:
        exact = (
            manifests["pre_holdout"].get("range_start") == READINESS_RANGE_START.isoformat()
            and manifests["pre_holdout"].get("range_end") == READINESS_RANGE_END.isoformat()
        )
    else:
        exact = (
            set(manifests) == {"pre_holdout", "holdout"}
            and manifests["pre_holdout"].get("range_start") == READINESS_RANGE_START.isoformat()
            and manifests["pre_holdout"].get("range_end") == manifests["holdout"].get("range_start")
            and manifests["holdout"].get("range_end") == READINESS_RANGE_END.isoformat()
        )
    if not exact:
        raise RuntimeError(f"{label} manifests do not cover the exact contiguous range")


def _validate_oracle_feature_caches(
    config: AsymmetricValueConfig,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    windows = _oracle_cache_windows(config)
    output: dict[str, Any] = {}
    inventories: dict[str, dict[str, Any]] = {}
    for scope, (start, end, filename) in windows.items():
        inventory = oracle_source_inventory(
            config.oracle_source,
            _dates(start, end),
        )
        inventories[scope] = inventory
        if inventory["missing_days"]:
            raise RuntimeError(
                f"{scope} Oracle source is missing: " + ", ".join(inventory["missing_days"])
            )
        cache_path = config.feature_cache / filename
        metadata_path = cache_path.with_suffix(".metadata.json")
        if not cache_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"{scope} Oracle feature cache is missing")
        metadata = json.loads(metadata_path.read_text())
        cache_sha256 = file_sha256(cache_path)
        try:
            validate_oracle_cache_identity(
                metadata,
                inventory,
                cache_sha256=cache_sha256,
            )
        except RuntimeError as error:
            raise RuntimeError(f"{scope} {error}") from error
        scan = pl.scan_parquet(cache_path)
        observed_dates = (
            scan.select(pl.col("window_start").dt.date().unique().sort())
            .collect()
            .get_column("window_start")
            .to_list()
        )
        if observed_dates != list(_dates(start, end)):
            raise RuntimeError(f"{scope} Oracle feature cache does not span its exact daily range")
        summary = scan.select(
            pl.len().alias("rows"),
            pl.col("market_id").n_unique().alias("markets"),
        ).collect()
        if (
            int(metadata.get("rows", -1)) != int(summary["rows"].item())
            or int(metadata.get("markets", -1)) != int(summary["markets"].item())
            or not _is_sha256(metadata.get("core_key_sha256"))
            or not _is_sha256(metadata.get("core_content_sha256"))
        ):
            raise RuntimeError(f"{scope} Oracle feature cache provenance is incomplete")
        violations = (
            scan.filter(
                (pl.col("oracle_source_timestamp") > pl.col("oracle_block_timestamp"))
                | (
                    pl.col("early_oracle_eligible")
                    & (
                        (
                            pl.col("oracle_block_timestamp")
                            > pl.col("observed_at")
                            - pl.duration(seconds=ORACLE_MINIMUM_PROPAGATION_SECONDS)
                        )
                        | (pl.col("oracle_age_seconds") < ORACLE_MINIMUM_PROPAGATION_SECONDS)
                        | (pl.col("oracle_age_seconds") > ORACLE_MAXIMUM_AGE_SECONDS)
                        | pl.any_horizontal(
                            [
                                pl.col(feature).is_null()
                                | ~pl.col(feature).cast(pl.Float64).is_finite()
                                for feature in EARLY_CAUSAL_ORACLE_FEATURES
                            ]
                        )
                    )
                )
            )
            .select(pl.len())
            .collect()
            .item()
        )
        if violations:
            raise RuntimeError(f"{scope} Oracle feature cache contains causal violations")
        output[scope] = {
            "range_start": start.isoformat(),
            "range_end": end.isoformat(),
            "source_inventory_sha256": inventory["inventory_sha256"],
            "source_days": inventory["available_days"],
            "cache_sha256": cache_sha256,
            "metadata_sha256": file_sha256(metadata_path),
        }
    required_dates = {"2026-07-31", "2026-08-01"}
    required_scope = "development" if config.evaluation is None else "evaluation"
    required_scope_dates = {record["date"] for record in inventories[required_scope]["records"]}
    if not required_dates.issubset(required_scope_dates):
        raise RuntimeError(f"{required_scope} Oracle cache omits July 31 or August 1")
    return output


def _validate_external_source_cache(config: AsymmetricValueConfig) -> dict[str, Any]:
    expected_dates = _date_strings(READINESS_RANGE_START, READINESS_RANGE_END)
    contracts = {
        "l2": (
            config.l2_source,
            L2_SOURCE_SCHEMA_VERSION,
            config.package_root / "sql" / "btc-spot-l2-source.sql",
        ),
        "candles": (
            config.candle_source,
            CANDLE_SOURCE_SCHEMA_VERSION,
            config.package_root / "sql" / "btc-chainlink-one-minute-candles-source.sql",
        ),
    }
    output: dict[str, Any] = {}
    for family, (source, schema_version, query_path) in contracts.items():
        records: list[dict[str, Any]] = []
        for day in expected_dates:
            data_path = source / f"{day}.parquet"
            record_path = source / f"{day}.parquet.json"
            if not data_path.is_file() or not record_path.is_file():
                raise FileNotFoundError(f"{family} source partition is missing: {day}")
            record = json.loads(record_path.read_text())
            expected = {
                "source": family,
                "source_schema_version": schema_version,
                "query_sha256": file_sha256(query_path),
                "sha256": file_sha256(data_path),
            }
            mismatches = [key for key, value in expected.items() if record.get(key) != value]
            if mismatches:
                raise RuntimeError(f"{family} source lineage changed on {day}")
            if family == "l2":
                observed_contracts = {
                    item.get("materialization_contract")
                    for item in record.get("materialization_artifacts", [])
                }
                qualified_seconds = int(record["summary"]["qualified_seconds"])
                if qualified_seconds and (
                    not observed_contracts
                    or not observed_contracts.issubset(set(L2_MATERIALIZATION_CONTRACTS))
                ):
                    raise RuntimeError(f"L2 source lineage is not allowed on {day}")
                if not qualified_seconds and observed_contracts:
                    raise RuntimeError(f"empty L2 partition has unexpected lineage on {day}")
            elif int(record["summary"]["fully_closed_candles"]) != EXPECTED_DAILY_CANDLES:
                raise RuntimeError(f"closed Chainlink candle coverage is incomplete on {day}")
            records.append(record)
        output[family] = {
            "days": len(records),
            "source_schema_version": schema_version,
            "inventory_sha256": hashlib.sha256(
                json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    return output


def _validate_price_manifests(config: AsymmetricValueConfig) -> dict[str, Any]:
    windows = _price_windows(config)
    output: dict[str, Any] = {}
    for scope, (start, end) in windows.items():
        path = config.price_cache / scope / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"{scope} PMXT price manifest is missing")
        manifest = json.loads(path.read_text())
        expected = {
            "schema_version": "btc-asymmetric-value-price-evidence-v2",
            "scope": scope,
            "range_start": start.isoformat(),
            "range_end": end.isoformat(),
            "source_contract": EXECUTION_EVIDENCE_CONTRACT,
            "snapshot_provider": PMXT_PROVIDER,
            "snapshot_schema_versions": [LEGACY_SNAPSHOT_SCHEMA_VERSION],
            "quantity": 5.0,
            "freshness_seconds": 2,
            "proxy_prices_used": False,
        }
        mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
        if mismatches:
            raise RuntimeError(f"{scope} PMXT price contract changed ({', '.join(mismatches)})")
        dates = [str(row["date"]) for row in manifest.get("coverage_by_day", [])]
        if dates != _date_strings(start, end):
            raise RuntimeError(f"{scope} PMXT manifest does not contain every UTC day")
        child_manifests: dict[str, str] = {}
        for cadence, execution_config in (
            (
                "seconds_1_59_one_second",
                ExecutionEvidenceConfig(
                    range_start=start,
                    range_end=end,
                    output_dir=(config.price_cache / scope / "seconds-1-59-one-second"),
                    sample_interval_seconds=1,
                    min_seconds_after_open=1,
                    max_seconds_after_open=59,
                    freshness_seconds=config.book_freshness_seconds,
                    quantity=config.quantity,
                    snapshot_schema_versions=(LEGACY_SNAPSHOT_SCHEMA_VERSION,),
                ),
            ),
            (
                "seconds_60_240_five_second",
                ExecutionEvidenceConfig(
                    range_start=start,
                    range_end=end,
                    output_dir=(config.price_cache / scope / "seconds-60-240-five-second"),
                    sample_interval_seconds=5,
                    min_seconds_after_open=60,
                    max_seconds_after_open=240,
                    freshness_seconds=config.book_freshness_seconds,
                    quantity=config.quantity,
                    snapshot_schema_versions=(LEGACY_SNAPSHOT_SCHEMA_VERSION,),
                ),
            ),
        ):
            child = load_execution_evidence_manifest(execution_config)
            if child.get("source_contract") != EXECUTION_EVIDENCE_CONTRACT:
                raise RuntimeError(f"{scope} PMXT child manifest uses a legacy contract")
            if child.get("snapshot_schema_versions") != [LEGACY_SNAPSHOT_SCHEMA_VERSION]:
                raise RuntimeError(f"{scope} PMXT child snapshot schema changed")
            child_manifests[cadence] = file_sha256(execution_config.output_dir / "manifest.json")
        output[scope] = {
            "days": len(dates),
            "manifest_identity_sha256": price_manifest_identity_sha256(manifest),
            "manifest_identity_excludes": list(PRICE_MANIFEST_IDENTITY_EXCLUDES),
            "child_manifest_sha256": child_manifests,
            "retained_rows": manifest["coverage_totals"]["retained_rows"],
            "strict_rows": manifest["coverage_totals"]["strict_rows"],
        }
    return output


def _oracle_cache_windows(
    config: AsymmetricValueConfig,
) -> dict[str, tuple[datetime, datetime, str]]:
    evaluation = config.evaluation
    if evaluation is None:
        return {
            "development": (
                config.fit.start,
                config.policy.end,
                DEVELOPMENT_ORACLE_CACHE,
            )
        }
    return {
        "development": (config.fit.start, evaluation.start, DEVELOPMENT_ORACLE_CACHE),
        "evaluation": (evaluation.start, evaluation.end, EVALUATION_ORACLE_CACHE),
    }


def _price_windows(
    config: AsymmetricValueConfig,
) -> dict[str, tuple[datetime, datetime]]:
    evaluation = config.evaluation
    if evaluation is None:
        return {"development": (config.fit.start, config.policy.end)}
    return {
        "development": (config.fit.start, evaluation.start),
        "evaluation": (evaluation.start, evaluation.end),
    }


def _readiness_identity_sha256(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "created_at",
            "payload_sha256",
            "readiness_identity_sha256",
        }
    }
    canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _payload_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "payload_sha256"},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _reuse_identical_readiness_manifest(
    destination: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    existing = json.loads(destination.read_text())
    if existing.get("schema_version") != READINESS_SCHEMA_VERSION:
        raise RuntimeError("existing readiness manifest uses a different schema")
    if existing.get("ready") is not True:
        raise RuntimeError("existing readiness manifest is not ready")
    if existing.get("payload_sha256") != _payload_sha256(existing):
        raise RuntimeError("existing readiness manifest payload hash is invalid")
    existing_identity = _readiness_identity_sha256(existing)
    if existing.get("readiness_identity_sha256") != existing_identity:
        raise RuntimeError("existing readiness manifest identity hash is invalid")
    if existing_identity != expected["readiness_identity_sha256"]:
        raise RuntimeError("existing immutable readiness manifest does not match current evidence")
    return existing


def _dates(start: datetime, end: datetime) -> tuple[date, ...]:
    values: list[date] = []
    current = start
    while current < end:
        values.append(current.date())
        current += timedelta(days=1)
    return tuple(values)


def _date_strings(start: datetime, end: datetime) -> list[str]:
    return [value.isoformat() for value in _dates(start, end)]


def _parse_utc_datetime(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("new-day readiness timestamps must include UTC")
    return parsed.astimezone(UTC)


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value
