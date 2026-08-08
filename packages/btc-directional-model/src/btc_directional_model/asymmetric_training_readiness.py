"""Fail-closed source readiness for the asymmetric-value training round."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .asymmetric_value_config import AsymmetricValueConfig
from .asymmetric_value_data import (
    EARLY_CAUSAL_ORACLE_FEATURES,
    ORACLE_MAXIMUM_AGE_SECONDS,
    ORACLE_MINIMUM_PROPAGATION_SECONDS,
)
from .core_config import CORE_ORACLE_SOURCE_CONTRACT, CoreTrainingConfig, load_core_config
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
    POLYGON_CHAINLINK_BTCUSD_PROXY,
    configure_read_only_connection,
    database_connection,
    file_sha256,
    load_core_manifest,
    write_json_exclusive,
)
from .spot_l2_chainlink_extract import (
    CANDLE_SOURCE_SCHEMA_VERSION,
    L2_INFORMATION_COLUMNS,
    L2_MATERIALIZATION_CONTRACTS,
    L2_SOURCE_SCHEMA_VERSION,
)

READINESS_SCHEMA_VERSION = "btc-asymmetric-training-readiness-v1"
READINESS_RANGE_START = datetime(2026, 4, 14, tzinfo=UTC)
READINESS_RANGE_END = datetime(2026, 8, 2, tzinfo=UTC)
EXPECTED_DAILY_MARKETS = 288
EXPECTED_DAILY_ONE_SECOND_ROWS = 86_400
EXPECTED_DAILY_CANDLES = 1_440
MINIMUM_OPENING_BOUNDARY_COVERAGE = 0.95
MINIMUM_PMXT_HOURLY_COVERAGE = 0.98
MINIMUM_L2_SECOND_COVERAGE = 0.70
RECENT_SOURCE_START = datetime(2026, 7, 16, tzinfo=UTC)
MINIMUM_RECENT_L2_SECOND_COVERAGE = 0.99
PMXT_PROVIDER = "pmxt_v2_execution_snapshots"
PMXT_INGESTER = "polymarket_btc_five_minute_execution_snapshots"
L2_INGESTER = "binance_spot_btcusdt_l2_one_second_features"
ORACLE_CACHE_SCHEMA_VERSION = "btc-asymmetric-value-early-oracle-v2"
DEVELOPMENT_ORACLE_CACHE = "development-oracle-propagation-2s.parquet"
EVALUATION_ORACLE_CACHE = "evaluation-oracle-propagation-2s.parquet"

CANONICAL_SOURCE_CONTRACT: dict[str, Any] = {
    "target": {
        "relation": "polymarket.btc_interval_markets",
        "field": "official_outcome",
        "allowed_values": ["up", "down"],
        "core_source_schema_version": CORE_ORACLE_SOURCE_SCHEMA_VERSION,
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
        "polymarket.binance_aggregate_trades": (
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


def prepare_asymmetric_training_readiness(
    config: AsymmetricValueConfig,
    *,
    output_dir: Path,
    connection: Any | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Validate every approved source and create a no-overwrite readiness seal."""

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
        "created_at": datetime.now(UTC).isoformat(),
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
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["payload_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "asymmetric-training-readiness.json"
    write_json_exclusive(destination, payload)
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
    if config.fit.start != READINESS_RANGE_START or config.evaluation.end != READINESS_RANGE_END:
        raise ValueError("readiness requires exact [2026-04-14, 2026-08-02)")
    if core_config.data.range_start != READINESS_RANGE_START:
        raise ValueError("Core source range does not begin at the readiness boundary")
    if core_config.data.range_end != READINESS_RANGE_END:
        raise ValueError("Core source range does not end at the readiness boundary")
    if core_config.data.source_contract != CORE_ORACLE_SOURCE_CONTRACT:
        raise ValueError("readiness requires a single btc_core_oracle_v1 source cache")
    if core_config.paths.source_data.resolve() != config.oracle_source.resolve():
        raise ValueError("Core and Oracle must use the same canonical source directory")


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


def _validate_core_oracle_source(
    config: AsymmetricValueConfig,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    manifests = {
        scope: load_core_manifest(core_config, scope) for scope in ("pre_holdout", "holdout")
    }
    if (
        manifests["pre_holdout"].get("range_start") != READINESS_RANGE_START.isoformat()
        or manifests["pre_holdout"].get("range_end") != manifests["holdout"].get("range_start")
        or manifests["holdout"].get("range_end") != READINESS_RANGE_END.isoformat()
    ):
        raise RuntimeError("Core+Oracle manifests do not cover the exact contiguous range")
    expected_core = {
        f"{value}.parquet" for value in _date_strings(READINESS_RANGE_START, READINESS_RANGE_END)
    }
    expected_oracle = {
        f"oracle-{value}.parquet"
        for value in _date_strings(READINESS_RANGE_START, READINESS_RANGE_END)
    }
    core_records = [record for manifest in manifests.values() for record in manifest["partitions"]]
    oracle_records = [
        record for manifest in manifests.values() for record in manifest["oracle_partitions"]
    ]
    observed_core = {record["path"] for record in core_records}
    observed_oracle = {record["path"] for record in oracle_records}
    if observed_core != expected_core or len(core_records) != 110:
        raise RuntimeError("Core source manifests do not contain exactly 110 daily partitions")
    if observed_oracle != expected_oracle or len(oracle_records) != 110:
        raise RuntimeError("Oracle source manifests do not contain exactly 110 daily partitions")
    for record in core_records:
        if int(record["incomplete_markets"]) or int(record["rows"]) != 300 * int(record["markets"]):
            raise RuntimeError(f"incomplete Core source partition: {record['path']}")
    for record in oracle_records:
        if int(record["rows"]) <= 0 or int(record["causality_violations"]):
            raise RuntimeError(f"invalid Oracle source partition: {record['path']}")
    paths = {scope: core_config.paths.source_data / f"manifest-{scope}.json" for scope in manifests}
    return {
        "source_contract": CORE_ORACLE_SOURCE_CONTRACT,
        "core_source_schema_version": CORE_ORACLE_SOURCE_SCHEMA_VERSION,
        "oracle_source_schema_version": CORE_ORACLE_ROUND_SCHEMA_VERSION,
        "source_directory": str(core_config.paths.source_data.resolve()),
        "daily_core_partitions": len(core_records),
        "daily_oracle_partitions": len(oracle_records),
        "manifest_ranges": {
            scope: {
                "range_start": manifest["range_start"],
                "range_end": manifest["range_end"],
            }
            for scope, manifest in manifests.items()
        },
        "manifest_sha256": {scope: file_sha256(path) for scope, path in paths.items()},
        "july_31_present": "oracle-2026-07-31.parquet" in observed_oracle,
        "august_1_present": "oracle-2026-08-01.parquet" in observed_oracle,
    }


def _validate_oracle_feature_caches(
    config: AsymmetricValueConfig,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    windows = {
        "development": (config.fit.start, config.evaluation.start, DEVELOPMENT_ORACLE_CACHE),
        "evaluation": (config.evaluation.start, config.evaluation.end, EVALUATION_ORACLE_CACHE),
    }
    output: dict[str, Any] = {}
    for scope, (start, end, filename) in windows.items():
        inventory = oracle_source_inventory(
            core_config.paths.source_data,
            _dates(start, end),
        )
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
    evaluation_dates = {
        record["date"]
        for record in oracle_source_inventory(
            core_config.paths.source_data,
            _dates(config.evaluation.start, config.evaluation.end),
        )["records"]
    }
    if not required_dates.issubset(evaluation_dates):
        raise RuntimeError("evaluation Oracle cache omits July 31 or August 1")
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
    windows = {
        "development": (config.fit.start, config.evaluation.start),
        "evaluation": (config.evaluation.start, config.evaluation.end),
    }
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
            "manifest_sha256": file_sha256(path),
            "child_manifest_sha256": child_manifests,
            "retained_rows": manifest["coverage_totals"]["retained_rows"],
            "strict_rows": manifest["coverage_totals"]["strict_rows"],
        }
    return output


def _dates(start: datetime, end: datetime) -> tuple[date, ...]:
    values: list[date] = []
    current = start
    while current < end:
        values.append(current.date())
        current += timedelta(days=1)
    return tuple(values)


def _date_strings(start: datetime, end: datetime) -> list[str]:
    return [value.isoformat() for value in _dates(start, end)]


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
