from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from btc_directional_model.config import (
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    PathConfig,
    SplitConfig,
    TrainingConfig,
)
from btc_directional_model.extract import load_existing_manifest
from btc_directional_model.features import (
    feature_build_contract,
    load_source_manifest,
    validate_feature_cache,
    verified_source_files,
)


def minimal_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        source_path=tmp_path / "config.toml",
        package_root=tmp_path,
        data=DataConfig(
            datetime(2026, 4, 21, tzinfo=UTC),
            datetime(2026, 4, 22, tzinfo=UTC),
            5,
            60,
            60,
            False,
        ),
        split=SplitConfig(0.6, 0.2, 0.2),
        model=ModelConfig((0.1,), 0.5, 0.9, 0.01, 0.65, 0.6, 1, 1, 0.05, 1),
        evaluation=EvaluationConfig(True, "fresh chronological holdout"),
        paths=PathConfig(
            tmp_path / "source",
            tmp_path / "features.parquet",
            tmp_path / "runs",
            tmp_path / "artifacts",
        ),
    )


def test_source_cache_rejects_changed_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest_path = source / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "range_start": "2026-04-21T00:00:00+00:00",
                "range_end": "2026-04-22T00:00:00+00:00",
                "strict_final_price_audit": False,
                "query_sha256": "old",
                "partitions": [],
            }
        )
    )

    with pytest.raises(RuntimeError, match="query_sha256"):
        load_existing_manifest(
            manifest_path,
            {
                "range_start": "2026-04-21T00:00:00+00:00",
                "range_end": "2026-04-22T00:00:00+00:00",
                "strict_final_price_audit": False,
                "query_sha256": "new",
            },
            force=False,
        )


def test_source_cache_allows_date_extension_with_same_query(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest_path = source / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "range_start": "2026-04-21T00:00:00+00:00",
                "range_end": "2026-04-22T00:00:00+00:00",
                "strict_final_price_audit": False,
                "query_sha256": "same",
                "partitions": [{"path": "2026-04-21.parquet", "rows": 1, "sha256": "hash"}],
            }
        )
    )

    existing = load_existing_manifest(
        manifest_path,
        {
            "range_start": "2026-04-21T00:00:00+00:00",
            "range_end": "2026-04-23T00:00:00+00:00",
            "strict_final_price_audit": False,
            "query_sha256": "same",
        },
        force=False,
    )

    assert existing is not None
    assert existing["range_end"] == "2026-04-22T00:00:00+00:00"


def test_feature_source_files_are_manifest_scoped_and_hashed(tmp_path: Path) -> None:
    config = minimal_config(tmp_path)
    config.paths.source_data.mkdir()
    included = config.paths.source_data / "included.parquet"
    excluded = config.paths.source_data / "excluded.parquet"
    included.write_bytes(b"included")
    excluded.write_bytes(b"excluded")
    manifest = {
        "range_start": config.data.range_start.isoformat(),
        "range_end": config.data.range_end.isoformat(),
        "strict_final_price_audit": False,
        "partitions": [
            {
                "path": included.name,
                "rows": 1,
                "sha256": hashlib.sha256(included.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest_path = config.paths.source_data / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    loaded = load_source_manifest(manifest_path, config)
    paths = verified_source_files(config.paths.source_data, loaded)

    assert paths == [included]
    included.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verified_source_files(config.paths.source_data, loaded)


def test_training_feature_cache_validation_rejects_modified_parquet(tmp_path: Path) -> None:
    config = minimal_config(tmp_path)
    config.paths.source_data.mkdir()
    source_manifest_path = config.paths.source_data / "manifest.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "range_start": config.data.range_start.isoformat(),
                "range_end": config.data.range_end.isoformat(),
                "strict_final_price_audit": False,
                "partitions": [{"path": "source.parquet", "rows": 1, "sha256": "recorded"}],
            }
        )
    )
    config.paths.feature_data.write_bytes(b"feature-data")
    metadata_path = config.paths.feature_data.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "build_contract": feature_build_contract(config, source_manifest_path),
                "feature_file_sha256": hashlib.sha256(b"feature-data").hexdigest(),
            }
        )
    )

    assert validate_feature_cache(config)["feature_file_sha256"]
    config.paths.feature_data.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="does not match"):
        validate_feature_cache(config)


def test_sql_uses_only_confirmed_sources_and_prior_completed_second() -> None:
    sql_path = Path(__file__).parents[1] / "sql" / "directional-source.sql"
    sql = sql_path.read_text()

    for table in (
        "polymarket.btc_interval_markets",
        "polymarket.btc_market_reference_facts",
        "polymarket.binance_one_second_klines",
        "polymarket.btc_market_decision_execution_snapshots",
    ):
        assert table in sql
    assert "snapshot.sampled_at - interval '1 second'" in sql
    assert "snapshot_artifact.status = 'completed'" in sql
    assert "kline_artifact.status = 'completed'" in sql
    assert "chainlink" not in sql.lower()
    assert "btc_orderbook_archive_events" not in sql
