from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import polars as pl
import pytest

from btc_directional_model.asymmetric_training_readiness import (
    DEVELOPMENT_ORACLE_CACHE,
    EXPECTED_DAILY_CANDLES,
    EXPECTED_DAILY_MARKETS,
    EXPECTED_DAILY_ONE_SECOND_ROWS,
    L2_MATERIALIZATION_CONTRACTS,
    LEGACY_SNAPSHOT_SCHEMA_VERSION,
    ORACLE_CACHE_SCHEMA_VERSION,
    ORACLE_MAXIMUM_AGE_SECONDS,
    ORACLE_MINIMUM_PROPAGATION_SECONDS,
    PMXT_PROVIDER,
    READINESS_RANGE_END,
    READINESS_RANGE_START,
    _validate_core_oracle_source,
    _validate_oracle_feature_caches,
    _validate_price_manifests,
    _validate_round_contract,
    _validate_sql_contracts,
    collect_database_inventory,
    oracle_source_inventory,
    prepare_asymmetric_training_readiness,
    validate_database_inventory,
    validate_oracle_cache_identity,
)
from btc_directional_model.asymmetric_value_config import load_asymmetric_value_config
from btc_directional_model.asymmetric_value_data import EARLY_CAUSAL_ORACLE_FEATURES
from btc_directional_model.core_config import (
    CORE_ORACLE_SOURCE_CONTRACT,
    CORE_SOURCE_CONTRACT,
    load_core_config,
)
from btc_directional_model.core_execution import EXECUTION_EVIDENCE_CONTRACT
from btc_directional_model.core_extract import (
    CORE_ORACLE_ROUND_SCHEMA_VERSION,
    CORE_ORACLE_SOURCE_SCHEMA_VERSION,
    file_sha256,
)


def config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "btc-5m-directional-asymmetric-value-one-second-20260414-20260802.toml"
    )


def target_core_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "btc-5m-directional-core-asymmetric-calibrated-20260414-20260802.toml"
    )


def target_profile_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "fit": SimpleNamespace(start=READINESS_RANGE_START),
        "policy": SimpleNamespace(end=READINESS_RANGE_END),
        "evaluation": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def complete_core_oracle_partitions() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    core: list[dict[str, object]] = []
    oracle: list[dict[str, object]] = []
    current = READINESS_RANGE_START
    while current < READINESS_RANGE_END:
        day = current.date().isoformat()
        core.append(
            {
                "path": f"{day}.parquet",
                "rows": 86_400,
                "markets": 288,
                "incomplete_markets": 0,
            }
        )
        oracle.append(
            {
                "path": f"oracle-{day}.parquet",
                "rows": 2_500,
                "causality_violations": 0,
            }
        )
        current += timedelta(days=1)
    return core, oracle


def complete_daily_inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current = READINESS_RANGE_START
    while current < READINESS_RANGE_END:
        rows.append(
            {
                "date": current.date().isoformat(),
                "labeled_markets": EXPECTED_DAILY_MARKETS,
                "opening_boundary_markets": 282,
                "final_price_markets": 280,
                "binance_one_second_rows": EXPECTED_DAILY_ONE_SECOND_ROWS,
                "binance_causality_violations": 0,
                "binance_providers": ["binance-vision"],
                "oracle_rounds": 2_500,
                "oracle_causality_violations": 0,
                "oracle_providers": ["polygon-rpc"],
                "pmxt_completed_hours": 24,
                "pmxt_artifact_rows": 1_000,
                "pmxt_providers": [PMXT_PROVIDER],
                "pmxt_schema_versions": [LEGACY_SNAPSHOT_SCHEMA_VERSION],
                "l2_rows": EXPECTED_DAILY_ONE_SECOND_ROWS,
                "l2_qualified_seconds": EXPECTED_DAILY_ONE_SECOND_ROWS,
                "l2_causality_violations": 0,
                "l2_providers": ["cryptohftdata"],
                "l2_materialization_contracts": [L2_MATERIALIZATION_CONTRACTS[0]],
                "chainlink_candle_rows": EXPECTED_DAILY_CANDLES,
                "chainlink_candle_causality_violations": 0,
                "chainlink_candle_providers": ["coinapi"],
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


def test_collect_database_inventory_uses_bounded_canonical_query() -> None:
    connection = FakeConnection(complete_daily_inventory())
    package_root = Path(__file__).resolve().parents[1]

    rows = collect_database_inventory(package_root, connection=connection)

    assert len(rows) == 110
    assert connection.cursor_instance.parameters == {
        "range_start": READINESS_RANGE_START,
        "range_end": READINESS_RANGE_END,
        "oracle_feed_proxy_address": ("0xc907e116054ad103354f2d350fd2514433d57f6f"),
    }
    query = connection.cursor_instance.query
    assert "polymarket.btc_interval_markets" in query
    assert "polymarket.btc_market_execution_snapshots" not in query
    assert "pmxt_v2_execution_snapshots" in query
    assert "polymarket.binance_spot_btcusdt_l2_training_features" in query
    assert "polymarket.polygon_chainlink_btcusd_oracle_rounds" in query


def test_database_inventory_accepts_complete_contract() -> None:
    summary = validate_database_inventory(complete_daily_inventory())

    assert summary["days"] == 110
    assert summary["labeled_markets"] == 31_680
    assert summary["binance_one_second_rows"] == 9_504_000
    assert summary["pmxt_completed_hour_coverage"] == 1.0
    assert summary["l2_qualified_second_coverage"] == 1.0


def test_sql_contract_names_the_canonical_spot_relations() -> None:
    contract = _validate_sql_contracts(Path(__file__).resolve().parents[1])

    assert len(contract["query_sha256"]) == 6
    sources = contract["relations"]
    assert sources["target"]["core_source_contract"] == CORE_SOURCE_CONTRACT
    assert sources["target"]["oracle_arm_source_contract"] == (CORE_ORACLE_SOURCE_CONTRACT)
    assert sources["polymarket_execution"]["provider"] == PMXT_PROVIDER
    assert sources["binance_spot_l2"]["information_dimension_count"] == 40
    assert (
        sources["binance_spot_l2"]["training_relation"]
        == "polymarket.binance_spot_btcusdt_l2_training_features"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "exact 110 UTC days"),
        (
            lambda rows: rows[0].__setitem__("binance_causality_violations", 1),
            "causal source violations",
        ),
        (
            lambda rows: rows[-1].__setitem__("pmxt_providers", ["wrong"]),
            "unexpected PMXT provider",
        ),
        (
            lambda rows: rows[-1].update({"l2_rows": 80_000, "l2_qualified_seconds": 80_000}),
            "recent spot-L2 coverage",
        ),
    ],
)
def test_database_inventory_fails_closed(mutation, message: str) -> None:
    rows = complete_daily_inventory()
    mutation(rows)

    with pytest.raises(RuntimeError, match=message):
        validate_database_inventory(rows)


def test_oracle_inventory_exposes_missing_july_31_and_august_1(
    tmp_path: Path,
) -> None:
    days = (
        READINESS_RANGE_END.date() - timedelta(days=2),
        READINESS_RANGE_END.date() - timedelta(days=1),
    )

    inventory = oracle_source_inventory(tmp_path, days)

    assert inventory["available_days"] == 0
    assert inventory["missing_days"] == ["2026-07-31", "2026-08-01"]


def test_oracle_cache_identity_requires_current_inventory() -> None:
    inventory = {"inventory_sha256": "a" * 64}
    metadata = {
        "schema_version": ORACLE_CACHE_SCHEMA_VERSION,
        "source_inventory_sha256": "b" * 64,
        "minimum_propagation_seconds": ORACLE_MINIMUM_PROPAGATION_SECONDS,
        "maximum_age_seconds": ORACLE_MAXIMUM_AGE_SECONDS,
        "features": list(EARLY_CAUSAL_ORACLE_FEATURES),
        "sha256": "c" * 64,
    }

    with pytest.raises(RuntimeError, match="stale"):
        validate_oracle_cache_identity(
            metadata,
            inventory,
            cache_sha256="c" * 64,
        )

    metadata["source_inventory_sha256"] = inventory["inventory_sha256"]
    validate_oracle_cache_identity(
        metadata,
        inventory,
        cache_sha256="c" * 64,
    )


def test_core_oracle_manifests_require_all_110_daily_partitions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import btc_directional_model.asymmetric_training_readiness as readiness

    core, oracle = complete_core_oracle_partitions()
    manifests = {
        "pre_holdout": {
            "range_start": READINESS_RANGE_START.isoformat(),
            "range_end": (READINESS_RANGE_START + timedelta(days=100)).isoformat(),
            "partitions": core[:100],
            "oracle_partitions": oracle[:100],
        },
        "holdout": {
            "range_start": (READINESS_RANGE_START + timedelta(days=100)).isoformat(),
            "range_end": READINESS_RANGE_END.isoformat(),
            "partitions": core[100:],
            "oracle_partitions": oracle[100:],
        },
    }
    for scope in manifests:
        (tmp_path / f"manifest-{scope}.json").write_text("{}")
    monkeypatch.setattr(
        readiness,
        "load_core_manifest",
        lambda _config, scope: manifests[scope],
    )
    core_config = SimpleNamespace(
        data=SimpleNamespace(source_contract=CORE_ORACLE_SOURCE_CONTRACT),
        paths=SimpleNamespace(source_data=tmp_path),
    )

    legacy_config = SimpleNamespace(
        evaluation=SimpleNamespace(),
        oracle_source=tmp_path,
    )
    result = _validate_core_oracle_source(legacy_config, core_config)

    assert result["daily_core_partitions"] == 110
    assert result["daily_oracle_partitions"] == 110
    assert (
        result["oracle_source_schema_version"]
        == CORE_ORACLE_SOURCE_SCHEMA_VERSION
    )
    assert (
        result["oracle_round_partition_schema_version"]
        == CORE_ORACLE_ROUND_SCHEMA_VERSION
    )
    assert result["july_31_present"] is True
    assert result["august_1_present"] is True

    manifests["holdout"]["oracle_partitions"].pop()
    with pytest.raises(RuntimeError, match="exactly 110 daily partitions"):
        _validate_core_oracle_source(legacy_config, core_config)


def test_target_profile_uses_one_110_day_core_oracle_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import btc_directional_model.asymmetric_training_readiness as readiness

    core, oracle = complete_core_oracle_partitions()
    core_manifest = {
        "range_start": READINESS_RANGE_START.isoformat(),
        "range_end": READINESS_RANGE_END.isoformat(),
        "partitions": core,
    }
    oracle_manifest = {
        **core_manifest,
        "oracle_partitions": oracle,
    }
    core_source = tmp_path / "core-market-source"
    oracle_source = tmp_path / "core-oracle-source"
    core_source.mkdir()
    oracle_source.mkdir()
    (core_source / "manifest-pre_holdout.json").write_text("{}")
    (oracle_source / "manifest-pre_holdout.json").write_text("{}")
    requested_sources: list[tuple[str, str]] = []

    def load_manifest(source_config, scope: str):
        requested_sources.append((source_config.data.source_contract, scope))
        if source_config.data.source_contract == CORE_SOURCE_CONTRACT:
            return core_manifest
        return oracle_manifest

    monkeypatch.setattr(readiness, "load_core_manifest", load_manifest)
    loaded = load_core_config(target_core_config_path())
    core_config = replace(
        loaded,
        paths=replace(loaded.paths, source_data=core_source),
    )
    config = target_profile_config(oracle_source=oracle_source)

    result = _validate_core_oracle_source(config, core_config)

    assert requested_sources == [
        (CORE_SOURCE_CONTRACT, "pre_holdout"),
        (CORE_ORACLE_SOURCE_CONTRACT, "pre_holdout"),
    ]
    assert result["source_scopes"] == ["pre_holdout"]
    assert result["source_contract"] == CORE_SOURCE_CONTRACT
    assert result["daily_oracle_raw_partitions"] == 110
    assert result["daily_core_partitions"] == 110
    assert result["daily_oracle_partitions"] == 110


def test_target_profile_round_contract_ends_at_policy_window(tmp_path: Path) -> None:
    source = tmp_path / "core-market-source"
    oracle_source = tmp_path / "core-oracle-source"
    config = target_profile_config(oracle_source=oracle_source)
    core_config = SimpleNamespace(
        data=SimpleNamespace(
            range_start=READINESS_RANGE_START,
            range_end=READINESS_RANGE_END,
            source_contract=CORE_SOURCE_CONTRACT,
        ),
        paths=SimpleNamespace(source_data=source),
    )

    _validate_round_contract(config, core_config)

    core_config.data.source_contract = CORE_ORACLE_SOURCE_CONTRACT
    with pytest.raises(ValueError, match="btc_core_v1"):
        _validate_round_contract(config, core_config)


def test_target_profile_validates_one_110_day_development_oracle_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import btc_directional_model.asymmetric_training_readiness as readiness

    cache_path = tmp_path / DEVELOPMENT_ORACLE_CACHE
    window_starts = [READINESS_RANGE_START + timedelta(days=offset) for offset in range(110)]
    observed_at = [value + timedelta(seconds=10) for value in window_starts]
    cache = pl.DataFrame(
        {
            "market_id": [f"market-{offset}" for offset in range(110)],
            "window_start": window_starts,
            "oracle_source_timestamp": [value - timedelta(seconds=5) for value in observed_at],
            "oracle_block_timestamp": [value - timedelta(seconds=5) for value in observed_at],
            "observed_at": observed_at,
            "early_oracle_eligible": [True] * 110,
            "oracle_age_seconds": [5.0] * 110,
            **{
                feature: [float(index + 1)] * 110
                for index, feature in enumerate(EARLY_CAUSAL_ORACLE_FEATURES)
            },
        }
    )
    cache.write_parquet(cache_path)
    inventory = {
        "inventory_sha256": "a" * 64,
        "missing_days": [],
        "available_days": 110,
        "records": [
            {"date": (READINESS_RANGE_START + timedelta(days=offset)).date().isoformat()}
            for offset in range(110)
        ],
    }
    inventory_sources: list[Path] = []

    def source_inventory(source: Path, _days) -> dict[str, object]:
        inventory_sources.append(source)
        return inventory

    monkeypatch.setattr(readiness, "oracle_source_inventory", source_inventory)
    cache_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "schema_version": ORACLE_CACHE_SCHEMA_VERSION,
                "source_inventory_sha256": inventory["inventory_sha256"],
                "minimum_propagation_seconds": ORACLE_MINIMUM_PROPAGATION_SECONDS,
                "maximum_age_seconds": ORACLE_MAXIMUM_AGE_SECONDS,
                "features": list(EARLY_CAUSAL_ORACLE_FEATURES),
                "core_key_sha256": "b" * 64,
                "core_content_sha256": "c" * 64,
                "rows": 110,
                "markets": 110,
                "sha256": file_sha256(cache_path),
            }
        )
    )
    config = target_profile_config(
        feature_cache=tmp_path,
        oracle_source=tmp_path / "oracle-source",
    )
    core_config = SimpleNamespace(paths=SimpleNamespace(source_data=tmp_path))

    result = _validate_oracle_feature_caches(config, core_config)

    assert list(result) == ["development"]
    assert result["development"]["range_start"] == READINESS_RANGE_START.isoformat()
    assert result["development"]["range_end"] == READINESS_RANGE_END.isoformat()
    assert result["development"]["source_days"] == 110
    assert inventory_sources == [config.oracle_source]

    cache.with_columns(
        pl.when(pl.col("market_id") == "market-0")
        .then(None)
        .otherwise(pl.col("oracle_return_from_window_open_bps"))
        .alias("oracle_return_from_window_open_bps")
    ).write_parquet(cache_path)
    metadata_path = cache_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["sha256"] = file_sha256(cache_path)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="causal violations"):
        _validate_oracle_feature_caches(config, core_config)


def test_target_profile_validates_one_110_day_development_pmxt_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import btc_directional_model.asymmetric_training_readiness as readiness

    scope_root = tmp_path / "development"
    scope_root.mkdir()
    manifest = {
        "schema_version": "btc-asymmetric-value-price-evidence-v2",
        "scope": "development",
        "range_start": READINESS_RANGE_START.isoformat(),
        "range_end": READINESS_RANGE_END.isoformat(),
        "source_contract": EXECUTION_EVIDENCE_CONTRACT,
        "snapshot_provider": PMXT_PROVIDER,
        "snapshot_schema_versions": [LEGACY_SNAPSHOT_SCHEMA_VERSION],
        "quantity": 5.0,
        "freshness_seconds": 2,
        "proxy_prices_used": False,
        "created_at": "2026-08-08T00:00:00+00:00",
        "coverage_by_day": [
            {"date": (READINESS_RANGE_START + timedelta(days=offset)).date().isoformat()}
            for offset in range(110)
        ],
        "coverage_totals": {"retained_rows": 1_000, "strict_rows": 900},
    }
    (scope_root / "manifest.json").write_text(json.dumps(manifest))
    child_ranges: list[tuple[datetime, datetime]] = []

    def load_child(config):
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "manifest.json").write_text("{}")
        child_ranges.append((config.range_start, config.range_end))
        return {
            "source_contract": EXECUTION_EVIDENCE_CONTRACT,
            "snapshot_schema_versions": [LEGACY_SNAPSHOT_SCHEMA_VERSION],
        }

    monkeypatch.setattr(readiness, "load_execution_evidence_manifest", load_child)
    config = target_profile_config(
        price_cache=tmp_path,
        book_freshness_seconds=2,
        quantity=5.0,
    )

    result = _validate_price_manifests(config)

    manifest["created_at"] = "2026-08-08T01:00:00+00:00"
    (scope_root / "manifest.json").write_text(json.dumps(manifest))
    repeated = _validate_price_manifests(config)

    assert list(result) == ["development"]
    assert result["development"]["days"] == 110
    assert result["development"]["manifest_identity_sha256"] == (
        repeated["development"]["manifest_identity_sha256"]
    )
    assert result["development"]["manifest_identity_excludes"] == [
        "created_at"
    ]
    assert child_ranges == [
        (READINESS_RANGE_START, READINESS_RANGE_END),
        (READINESS_RANGE_START, READINESS_RANGE_END),
    ] * 2


def test_readiness_manifest_reuses_identical_immutable_seal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import btc_directional_model.asymmetric_training_readiness as readiness

    config = load_asymmetric_value_config(config_path())
    monkeypatch.setattr(readiness, "load_core_config", lambda _path: SimpleNamespace())
    monkeypatch.setattr(readiness, "_validate_round_contract", lambda *_args: None)
    monkeypatch.setattr(
        readiness,
        "_validate_sql_contracts",
        lambda _root: {"query_sha256": {}},
    )
    monkeypatch.setattr(
        readiness,
        "collect_database_inventory",
        lambda *_args, **_kwargs: complete_daily_inventory(),
    )
    monkeypatch.setattr(
        readiness,
        "_validate_core_oracle_source",
        lambda *_args: {"daily_core_partitions": 110, "daily_oracle_partitions": 110},
    )
    monkeypatch.setattr(
        readiness,
        "_validate_oracle_feature_caches",
        lambda *_args: {"development": {}, "evaluation": {}},
    )
    monkeypatch.setattr(
        readiness,
        "_validate_external_source_cache",
        lambda *_args: {"l2": {"days": 110}, "candles": {"days": 110}},
    )
    monkeypatch.setattr(
        readiness,
        "_validate_price_manifests",
        lambda *_args: {"development": {}, "evaluation": {}},
    )

    destination, payload = prepare_asymmetric_training_readiness(
        config,
        output_dir=tmp_path,
        connection=object(),
    )

    assert payload["ready"] is True
    assert len(payload["payload_sha256"]) == 64
    assert json.loads(destination.read_text())["payload_sha256"] == payload["payload_sha256"]
    repeated_destination, repeated_payload = prepare_asymmetric_training_readiness(
        config,
        output_dir=tmp_path,
        connection=object(),
    )
    assert repeated_destination == destination
    assert repeated_payload == payload
    assert repeated_payload["readiness_identity_sha256"] == payload["readiness_identity_sha256"]
    assert DEVELOPMENT_ORACLE_CACHE.endswith(".parquet")
