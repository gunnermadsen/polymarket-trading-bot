from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Self

import pytest

import btc_directional_model.asymmetric_training_readiness as readiness
from btc_directional_model.asymmetric_training_readiness import (
    ASYMMETRIC_PREDICTION_SECONDS,
    EXPECTED_DAILY_MARKETS,
    EXPECTED_DAILY_ONE_SECOND_ROWS,
    L2_MATERIALIZATION_CONTRACTS,
    LEGACY_SNAPSHOT_SCHEMA_VERSION,
    PLANNED_VALIDATION_END,
    PLANNED_VALIDATION_START,
    PMXT_PROVIDER,
    assess_new_day_database_inventory,
    collect_new_day_database_inventory,
    inspect_external_archive_status,
    load_new_day_readiness_contract,
    prepare_new_day_training_readiness,
    validate_new_day_readiness_contract,
)


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def config_path() -> Path:
    return (
        package_root()
        / "configs"
        / "btc-5m-directional-asymmetric-new-day-readiness-20260810-20260820.toml"
    )


def completed_assessment_time() -> datetime:
    return PLANNED_VALIDATION_END + timedelta(seconds=1)


def complete_inventory() -> list[dict[str, object]]:
    expected_grid = EXPECTED_DAILY_MARKETS * len(ASYMMETRIC_PREDICTION_SECONDS)
    rows: list[dict[str, object]] = []
    current = PLANNED_VALIDATION_START
    while current < PLANNED_VALIDATION_END:
        rows.append(
            {
                "date": current.date().isoformat(),
                "labeled_markets": EXPECTED_DAILY_MARKETS,
                "up_markets": 140,
                "down_markets": 148,
                "label_identity_md5": "a" * 32,
                "opening_boundary_markets": EXPECTED_DAILY_MARKETS,
                "final_price_markets": EXPECTED_DAILY_MARKETS,
                "reference_causality_violations": 0,
                "reference_source_identity_md5": "b" * 32,
                "binance_one_second_rows": EXPECTED_DAILY_ONE_SECOND_ROWS,
                "binance_qualified_seconds": EXPECTED_DAILY_ONE_SECOND_ROWS,
                "binance_causality_violations": 0,
                "binance_providers": ["binance-vision"],
                "binance_source_identity_md5": "c" * 32,
                "pmxt_completed_hours": 24,
                "pmxt_artifact_rows": 345_600,
                "pmxt_providers": [PMXT_PROVIDER],
                "pmxt_schema_versions": [LEGACY_SNAPSHOT_SCHEMA_VERSION],
                "pmxt_source_identity_md5": "d" * 32,
                "pmxt_exact_grid_rows": expected_grid,
                "pmxt_exact_grid_keys": expected_grid,
                "pmxt_strict_grid_rows": expected_grid,
                "pmxt_causality_violations": 0,
                "l2_rows": EXPECTED_DAILY_ONE_SECOND_ROWS,
                "l2_qualified_seconds": EXPECTED_DAILY_ONE_SECOND_ROWS,
                "l2_causality_violations": 0,
                "l2_providers": ["cryptohftdata"],
                "l2_materialization_contracts": [
                    L2_MATERIALIZATION_CONTRACTS[0]
                ],
                "l2_source_identity_md5": "e" * 32,
                "oracle_rounds": 0,
                "oracle_causality_violations": 0,
                "oracle_providers": [],
                "oracle_source_identity_md5": "",
                "chainlink_candle_rows": 0,
                "chainlink_candle_causality_violations": 0,
                "chainlink_candle_providers": [],
                "chainlink_candle_source_identity_md5": "",
            }
        )
        current += timedelta(days=1)
    return rows


class FakeColumn:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeCursor:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.description = [FakeColumn(name) for name in rows[0]]
        self.query = ""
        self.parameters: dict[str, object] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, parameters: dict[str, object]) -> None:
        self.query = query
        self.parameters = parameters

    def fetchall(self) -> list[tuple[object, ...]]:
        names = [column.name for column in self.description]
        return [tuple(row[name] for name in names) for row in self.rows]


class FakeConnection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.cursor_instance = FakeCursor(rows)

    def cursor(self) -> FakeCursor:
        return self.cursor_instance


def test_frozen_contract_preserves_quarantine_and_planned_validation() -> None:
    contract = load_new_day_readiness_contract(config_path())

    assert contract.frozen_at < PLANNED_VALIDATION_START
    assert contract.validation_start == PLANNED_VALIDATION_START
    assert contract.validation_end == PLANNED_VALIDATION_END
    assert contract.quarantine_start == datetime(2026, 8, 2, tzinfo=UTC)
    assert contract.quarantine_end == PLANNED_VALIDATION_START
    assert contract.minimum_source_grid_coverage == 0.90
    assert contract.minimum_l2_second_coverage == 0.95

    with pytest.raises(ValueError, match="shifted frozen config"):
        validate_new_day_readiness_contract(
            replace(contract, frozen_at=PLANNED_VALIDATION_START)
        )


def test_new_day_query_is_bounded_to_canonical_relations() -> None:
    contract = load_new_day_readiness_contract(config_path())
    connection = FakeConnection(complete_inventory())

    rows = collect_new_day_database_inventory(
        package_root(),
        contract,
        connection=connection,
    )

    assert len(rows) == 10
    assert connection.cursor_instance.parameters["range_start"] == (
        PLANNED_VALIDATION_START
    )
    assert connection.cursor_instance.parameters["range_end"] == (
        PLANNED_VALIDATION_END
    )
    query = connection.cursor_instance.query
    assert "polymarket.btc_market_execution_snapshots" in query
    assert "polymarket.binance_spot_btcusdt_l2_training_features" in query
    assert "polymarket.binance_btcusdt_l2_training_features" not in query
    assert "pmxt_v2_execution_snapshots" in query


def test_oracle_and_candles_are_inventory_only() -> None:
    contract = load_new_day_readiness_contract(config_path())

    result = assess_new_day_database_inventory(
        complete_inventory(),
        contract,
        observed_at=completed_assessment_time(),
    )

    assert result["ready"] is True
    assert result["mandatory_blockers"] == []
    assert len(result["nonblocking_source_inventory"]) == 20
    assert all(
        item["blocking"] is False
        for item in result["nonblocking_source_inventory"]
    )
    assert len(result["daily_inventory_sha256"]) == 64


def test_missing_spot_l2_and_pmxt_fail_with_material_reasons() -> None:
    contract = load_new_day_readiness_contract(config_path())
    rows = complete_inventory()
    rows[0].update(
        {
            "l2_rows": 0,
            "l2_qualified_seconds": 0,
            "l2_materialization_contracts": [],
            "pmxt_completed_hours": 0,
            "pmxt_artifact_rows": 0,
            "pmxt_providers": [],
            "pmxt_schema_versions": [],
            "pmxt_exact_grid_rows": 0,
            "pmxt_exact_grid_keys": 0,
            "pmxt_strict_grid_rows": 0,
        }
    )

    result = assess_new_day_database_inventory(
        rows,
        contract,
        observed_at=completed_assessment_time(),
    )
    codes = {item["code"] for item in result["mandatory_blockers"]}

    assert result["ready"] is False
    assert result["required_materialization_sources"] == [
        "binance_spot_l2",
        "polymarket_execution",
    ]
    assert "spot_l2_coverage" in codes
    assert "pmxt_completed_hours" in codes
    assert "pmxt_artifact_rows" in codes
    assert "pmxt_exact_grid_coverage" in codes
    assert "pmxt_strict_candidate_grid_coverage" in codes
    assert all(
        {"date", "source", "code", "observed", "required"}.issubset(item)
        for item in result["mandatory_blockers"]
    )


def test_binance_must_be_exact_and_spot_l2_must_retain_95_percent() -> None:
    contract = load_new_day_readiness_contract(config_path())
    rows = complete_inventory()
    rows[0].update(
        {
            "binance_one_second_rows": 86_399,
            "binance_qualified_seconds": 86_399,
            "l2_rows": 82_079,
            "l2_qualified_seconds": 82_079,
        }
    )

    result = assess_new_day_database_inventory(
        rows,
        contract,
        observed_at=completed_assessment_time(),
    )
    codes = {item["code"] for item in result["mandatory_blockers"]}

    assert "one_second_coverage" in codes
    assert "spot_l2_coverage" in codes


def test_open_validation_days_are_not_reported_as_missing_data() -> None:
    contract = load_new_day_readiness_contract(config_path())
    rows = complete_inventory()
    for row in rows:
        for key, value in tuple(row.items()):
            if isinstance(value, int):
                row[key] = 0

    result = assess_new_day_database_inventory(
        rows,
        contract,
        observed_at=contract.frozen_at,
    )

    assert {item["code"] for item in result["mandatory_blockers"]} == {
        "not_yet_available"
    }
    assert result["required_materialization_sources"] == []
    assert result["checks"]["not_yet_available_days"] == 10


def test_unmounted_archive_and_upstream_l2_contract_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "readiness.toml"
    source.write_text(config_path().read_text())
    contract = replace(
        load_new_day_readiness_contract(source),
        external_archive_mount=tmp_path / "unmounted",
        pmxt_cache_root=tmp_path / "unmounted" / "pmxt",
        spot_l2_archive_root=tmp_path / "unmounted" / "spot-l2",
    )
    rows = complete_inventory()
    for row in rows:
        row.update(
            {
                "l2_rows": 0,
                "l2_qualified_seconds": 0,
                "l2_materialization_contracts": [],
            }
        )
    monkeypatch.setattr(
        readiness,
        "collect_new_day_database_inventory",
        lambda *_args, **_kwargs: rows,
    )

    archive = inspect_external_archive_status(
        contract,
        required_materialization_sources=["binance_spot_l2"],
    )
    destination, payload = prepare_new_day_training_readiness(
        contract,
        package_root=package_root(),
        output_dir=tmp_path / "output",
        connection=object(),
        observed_at=completed_assessment_time(),
    )

    assert archive["status"] == "blocked_archive_unmounted"
    assert archive["fallback_directory_created"] is False
    assert payload["status"] == "blocked_archive_unmounted"
    assert payload["ready"] is False
    assert payload["upstream_materialization"]["binance_spot_l2"]["status"] == (
        "blocked_upstream_materialization_contract"
    )
    assert payload["checks"]["oracle_and_candles_inventory_only"] is True
    assert destination.is_file()
