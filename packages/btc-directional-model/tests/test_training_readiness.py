from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from btc_directional_model import core_execution
from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_execution import (
    EXECUTION_CONTEXT_SECONDS,
    EXECUTION_EVIDENCE_CONTRACT,
    EXECUTION_EVIDENCE_SCHEMA,
    EXECUTION_EVIDENCE_SCHEMA_VERSION,
    ExecutionEvidenceConfig,
    execution_partition_summary,
)
from btc_directional_model.core_extract import (
    CORE_ORACLE_ROUND_SCHEMA,
    CORE_ORACLE_ROUND_SCHEMA_VERSION,
    CORE_ORACLE_SOURCE_SCHEMA_VERSION,
    CORE_SOURCE_SCHEMA,
    ORACLE_MAX_PUBLICATION_DELAY_SECONDS,
    POLYGON_CHAINLINK_BTCUSD_PROXY,
    aggregate_oracle_partition_summaries,
    aggregate_partition_summaries,
    file_sha256,
    oracle_partition_summary,
    partition_summary,
    write_json_atomic,
)
from btc_directional_model.core_features import (
    CORE_FEATURE_SCHEMA_VERSION,
    CORE_MATURE_REVERSAL_ORACLE_FEATURE_SCHEMA_VERSION,
    feature_build_contract,
)
from btc_directional_model.training_readiness import (
    _validate_execution_rows,
    generate_training_readiness,
)


def oracle_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "btc-5m-directional-core-oracle-early-entry-20260321-20260729.toml"
    )


def one_day_config(tmp_path: Path):
    base = load_core_config(oracle_config_path())
    start = datetime(2026, 7, 1, tzinfo=UTC)
    end = start + timedelta(days=1)
    return replace(
        base,
        data=replace(
            base.data,
            range_start=start,
            range_end=end,
        ),
        split=replace(
            base.split,
            development_start=start,
            development_end=start,
            probability_calibration_start=start,
            probability_calibration_end=start,
            policy_selection_start=start,
            policy_selection_end=end,
            holdout_start=end,
            holdout_end=end,
            validation_windows=(),
        ),
        paths=replace(
            base.paths,
            source_data=tmp_path / "source",
            development_feature_data=tmp_path / "features.parquet",
            holdout_feature_data=tmp_path / "holdout.parquet",
            runs=tmp_path / "runs",
            artifacts=tmp_path / "artifacts",
        ),
    )


def write_readiness_inputs(
    tmp_path: Path,
    *,
    missing_book_seconds: set[int] | None = None,
    empty_book: bool = False,
):
    missing_book_seconds = missing_book_seconds or set()
    config = one_day_config(tmp_path)
    start = config.data.range_start
    end = config.data.range_end
    config.paths.source_data.mkdir(parents=True)
    core_path = config.paths.source_data / f"{start.date()}.parquet"
    core_rows = []
    for second in range(300):
        price = 100_000.0 + second
        core_rows.append(
            {
                "market_id": "market-a",
                "window_start": start,
                "window_end": start + timedelta(minutes=5),
                "official_outcome": "up",
                "label_up": 1,
                "opening_boundary": 100_000.0,
                "final_price": 100_001.0,
                "observed_at": start + timedelta(seconds=second),
                "seconds_elapsed": second,
                "btc_open": price,
                "btc_high": price + 1,
                "btc_low": price - 1,
                "btc_close": price,
                "btc_base_volume": 1.0,
                "btc_quote_volume": price,
                "trade_count": 10,
                "btc_taker_buy_base_volume": 0.5,
                "btc_taker_buy_quote_volume": price / 2,
            }
        )
    pq.write_table(
        pa.Table.from_pylist(core_rows, schema=CORE_SOURCE_SCHEMA),
        core_path,
    )
    core_partition = {
        "path": core_path.name,
        "sha256": file_sha256(core_path),
        **partition_summary(core_path),
    }
    oracle_path = (
        config.paths.source_data / f"oracle-{start.date()}.parquet"
    )
    oracle_row = {
        "oracle_price": 100_000.0,
        "oracle_source_timestamp": start - timedelta(seconds=1),
        "oracle_block_timestamp": start - timedelta(seconds=1),
        "oracle_phase_id": 3,
        "oracle_round_id": 1,
        "oracle_block_number": 50_000_000,
        "oracle_log_index": 0,
    }
    pq.write_table(
        pa.Table.from_pylist(
            [oracle_row],
            schema=CORE_ORACLE_ROUND_SCHEMA,
        ),
        oracle_path,
    )
    oracle_partition = {
        "path": oracle_path.name,
        "sha256": file_sha256(oracle_path),
        **oracle_partition_summary(oracle_path),
    }
    package_root = config.package_root
    core_query = (package_root / "sql" / "btc-core-source.sql").read_text()
    oracle_query = (
        package_root / "sql" / "btc-core-oracle-source.sql"
    ).read_text()
    source_manifest = {
        "source_contract": config.data.source_contract,
        "source_schema_version": CORE_ORACLE_SOURCE_SCHEMA_VERSION,
        "source_schema_sha256": hashlib.sha256(
            CORE_SOURCE_SCHEMA.to_string().encode()
        ).hexdigest(),
        "scope": "pre_holdout",
        "range_start": start.isoformat(),
        "range_end": end.isoformat(),
        "strict_final_price_audit": False,
        "query_sha256": hashlib.sha256(core_query.encode()).hexdigest(),
        "oracle_feed_proxy_address": POLYGON_CHAINLINK_BTCUSD_PROXY,
        "oracle_max_publication_delay_seconds": (
            ORACLE_MAX_PUBLICATION_DELAY_SECONDS
        ),
        "oracle_source_schema_version": (
            CORE_ORACLE_ROUND_SCHEMA_VERSION
        ),
        "oracle_source_schema_sha256": hashlib.sha256(
            CORE_ORACLE_ROUND_SCHEMA.to_string().encode()
        ).hexdigest(),
        "oracle_query_sha256": hashlib.sha256(
            oracle_query.encode()
        ).hexdigest(),
        "partitions": [core_partition],
        "oracle_partitions": [oracle_partition],
        "totals": aggregate_partition_summaries([core_partition]),
        "oracle_totals": aggregate_oracle_partition_summaries(
            [oracle_partition]
        ),
    }
    source_manifest_path = (
        config.paths.source_data / "manifest-pre_holdout.json"
    )
    write_json_atomic(source_manifest_path, source_manifest)

    feature_rows = [
        {
            "market_id": "market-a",
            "window_start": start,
            "observed_at": start + timedelta(seconds=second),
            "seconds_elapsed": second,
            "oracle_model_eligible": True,
        }
        for second in range(120, 141, 5)
    ]
    pl.DataFrame(feature_rows).write_parquet(
        config.paths.development_feature_data
    )
    feature_metadata = {
        "build_contract": feature_build_contract(
            config,
            "pre_holdout",
            source_manifest_path,
        ),
        "feature_schema_version": CORE_FEATURE_SCHEMA_VERSION,
        "candidate_feature_schema_versions": {
            "histogram_mature_reversal_oracle": (
                CORE_MATURE_REVERSAL_ORACLE_FEATURE_SCHEMA_VERSION
            )
        },
        "daily_core_counts": [
            {
                "date": start.date().isoformat(),
                "markets": 1,
                "up_markets": 1,
                "down_markets": 0,
            }
        ],
        "oracle": {
            "daily_complete_counts": [
                {
                    "date": start.date().isoformat(),
                    "markets": 1,
                    "up_markets": 1,
                    "down_markets": 0,
                }
            ]
        },
        "feature_file_sha256": file_sha256(
            config.paths.development_feature_data
        ),
    }
    write_json_atomic(
        config.paths.development_feature_data.with_suffix(
            ".metadata.json"
        ),
        feature_metadata,
    )

    execution_dir = tmp_path / "execution"
    execution_dir.mkdir()
    execution_path = execution_dir / f"{start.date()}.parquet"
    execution_rows = []
    if not empty_book:
        for second in EXECUTION_CONTEXT_SECONDS:
            row: dict[str, object] = {
                name: None for name in EXECUTION_EVIDENCE_SCHEMA.names
            }
            strict_ten = second not in missing_book_seconds
            row.update(
                {
                    "market_id": "market-a",
                    "window_start": start,
                    "window_end": start + timedelta(minutes=5),
                    "official_outcome": "up",
                    "label_up": 1,
                    "observed_at": start + timedelta(seconds=second),
                    "seconds_elapsed": second,
                    "up_ask_vwap_5": 0.45,
                    "down_ask_vwap_5": 0.55,
                    "up_ask_vwap_10": 0.46,
                    "down_ask_vwap_10": 0.56,
                    "up_side_valid": True,
                    "down_side_valid": True,
                    "up_side_fresh": True,
                    "down_side_fresh": True,
                    "up_stale_initialized": False,
                    "down_stale_initialized": False,
                    "strict_both_side_eligible": True,
                    "strict_both_side_eligible_10": strict_ten,
                }
            )
            execution_rows.append(row)
    pq.write_table(
        pa.Table.from_pylist(
            execution_rows,
            schema=EXECUTION_EVIDENCE_SCHEMA,
        ),
        execution_path,
    )
    execution_partition = {
        "path": execution_path.name,
        "sha256": file_sha256(execution_path),
        **execution_partition_summary(execution_path),
    }
    execution_config = ExecutionEvidenceConfig(
        range_start=start,
        range_end=end,
        output_dir=execution_dir,
    )
    execution_manifest = {
        "source_contract": EXECUTION_EVIDENCE_CONTRACT,
        "source_schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "range_start": start.isoformat(),
        "range_end": end.isoformat(),
        "sample_interval_seconds": 5,
        "min_seconds_after_open": 90,
        "max_seconds_after_open": 140,
        "freshness_seconds": execution_config.freshness_seconds,
        "quantity": execution_config.quantity,
        "primary_key": ["market_id", "observed_at"],
        "partitions": [execution_partition],
        "totals": core_execution._aggregate_partition_summaries(
            [execution_partition]
        ),
    }
    write_json_atomic(
        execution_dir / "manifest.json",
        execution_manifest,
    )
    return config, execution_config


def test_readiness_counts_complete_oracle_and_exact_book_cohorts(
    tmp_path: Path,
) -> None:
    config, execution_config = write_readiness_inputs(tmp_path)

    json_path, markdown_path, payload = generate_training_readiness(
        config,
        execution_config=execution_config,
        output_dir=tmp_path / "readiness",
    )

    daily = payload["daily"][0]
    assert daily["expected_markets"] == 288
    assert daily["core_complete_markets"] == 1
    assert daily["core_oracle_complete_markets"] == 1
    assert daily["book_complete_11_point_markets"] == 1
    assert daily[
        "book_point_qualified_markets_by_second"
    ]["120"] == 1
    assert daily[
        "common_oracle_book_point_qualified_markets_by_second"
    ]["140"] == 1
    assert payload["cohorts"]["policy_selection"]["totals"][
        "core_oracle_complete_markets"
    ] == 1
    assert json.loads(json_path.read_text())["schema_version"].endswith(
        "-v1"
    )
    assert "Daily readiness" in markdown_path.read_text()


def test_readiness_requires_t_minus_five_book_context(
    tmp_path: Path,
) -> None:
    config, execution_config = write_readiness_inputs(
        tmp_path,
        missing_book_seconds={115},
    )

    _, _, payload = generate_training_readiness(
        config,
        execution_config=execution_config,
        output_dir=tmp_path / "readiness",
    )

    daily = payload["daily"][0]
    assert daily[
        "book_raw_strict_ten_share_markets_by_second"
    ]["120"] == 1
    assert daily[
        "book_point_qualified_markets_by_second"
    ]["120"] == 0
    assert daily[
        "common_oracle_book_point_qualified_markets_by_second"
    ]["120"] == 0


def test_readiness_preserves_zero_event_book_day(
    tmp_path: Path,
) -> None:
    config, execution_config = write_readiness_inputs(
        tmp_path,
        empty_book=True,
    )

    _, _, payload = generate_training_readiness(
        config,
        execution_config=execution_config,
        output_dir=tmp_path / "readiness",
    )

    assert len(payload["daily"]) == 1
    assert payload["daily"][0]["book_complete_11_point_markets"] == 0
    assert payload["totals"][
        "common_oracle_book_point_qualified_markets_by_second"
    ]["125"] == 0


def test_readiness_fails_on_feature_hash_mismatch(
    tmp_path: Path,
) -> None:
    config, execution_config = write_readiness_inputs(tmp_path)
    config.paths.development_feature_data.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="feature file hash mismatch"):
        generate_training_readiness(
            config,
            execution_config=execution_config,
            output_dir=tmp_path / "readiness",
        )


def test_readiness_rejects_duplicate_and_unexpected_execution_rows() -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    base = {
        "market_id": "market-a",
        "window_start": start,
        "observed_at": start + timedelta(seconds=120),
        "seconds_elapsed": 120,
        "strict_both_side_eligible": True,
        "strict_both_side_eligible_10": True,
    }
    duplicate = pl.DataFrame([base, base])
    with pytest.raises(RuntimeError, match="duplicate"):
        _validate_execution_rows(
            duplicate,
            start,
            start + timedelta(days=1),
        )

    unexpected = pl.DataFrame(
        [
            {
                **base,
                "window_start": start + timedelta(days=1),
                "observed_at": start
                + timedelta(days=1, seconds=120),
            }
        ]
    )
    with pytest.raises(RuntimeError, match="outside"):
        _validate_execution_rows(
            unexpected,
            start,
            start + timedelta(days=1),
        )
