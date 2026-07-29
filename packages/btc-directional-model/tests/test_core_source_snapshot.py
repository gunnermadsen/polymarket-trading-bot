from __future__ import annotations

import errno
import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from btc_directional_model import core_extract
from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_extract import (
    CORE_SOURCE_SCHEMA,
    CORE_SOURCE_SCHEMA_VERSION,
    IMMUTABLE_SOURCE_SNAPSHOT_KEY,
    aggregate_partition_summaries,
    extract_core_source,
    file_sha256,
    load_core_manifest,
    partition_summary,
    snapshot_residual_admission_source,
)


def residual_core_config() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "btc-5m-directional-core-residual-admission-20260321-20260721.toml"
    )


@pytest.fixture(scope="module")
def expanded_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    package_root = Path(__file__).resolve().parents[1]
    source = tmp_path_factory.mktemp("expanded-core-source")
    query = (package_root / "sql" / "btc-core-source.sql").read_text()
    partitions = []
    cursor = datetime(2026, 3, 21, tzinfo=UTC)
    end = datetime(2026, 7, 29, tzinfo=UTC)
    while cursor < end:
        destination = source / f"{cursor.date().isoformat()}.parquet"
        row = {
            "market_id": f"market-{cursor.date().isoformat()}",
            "window_start": cursor,
            "window_end": cursor + timedelta(minutes=5),
            "official_outcome": "up",
            "label_up": 1,
            "opening_boundary": 100.0,
            "final_price": 101.0,
            "observed_at": cursor + timedelta(seconds=1),
            "seconds_elapsed": 1,
            "btc_open": 100.0,
            "btc_high": 101.0,
            "btc_low": 99.0,
            "btc_close": 100.5,
            "btc_base_volume": 1.0,
            "btc_quote_volume": 100.5,
            "trade_count": 1,
            "btc_taker_buy_base_volume": 0.5,
            "btc_taker_buy_quote_volume": 50.25,
        }
        pq.write_table(
            pa.Table.from_pylist([row], schema=CORE_SOURCE_SCHEMA),
            destination,
            compression="zstd",
        )
        partitions.append(
            {
                "path": destination.name,
                "sha256": file_sha256(destination),
                **partition_summary(destination),
            }
        )
        cursor += timedelta(days=1)
    manifest = {
        "source_contract": "btc_core_v1",
        "source_schema_version": CORE_SOURCE_SCHEMA_VERSION,
        "source_schema_sha256": hashlib.sha256(
            CORE_SOURCE_SCHEMA.to_string().encode()
        ).hexdigest(),
        "scope": "pre_holdout",
        "range_start": "2026-03-21T00:00:00+00:00",
        "range_end": "2026-07-29T00:00:00+00:00",
        "strict_final_price_audit": False,
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "partitions": partitions,
        "totals": aggregate_partition_summaries(partitions),
    }
    (source / "manifest-pre_holdout.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return source


def snapshot_config(destination: Path):
    config = load_core_config(residual_core_config())
    return replace(
        config,
        paths=replace(config.paths, source_data=destination),
    )


def test_snapshot_creates_exact_checksum_verified_hard_links(
    tmp_path: Path,
    expanded_source: Path,
) -> None:
    destination = tmp_path / "snapshot"
    config = snapshot_config(destination)

    manifest = snapshot_residual_admission_source(config, expanded_source)

    assert len(manifest["partitions"]) == 122
    assert manifest["range_start"] == "2026-03-21T00:00:00+00:00"
    assert manifest["range_end"] == "2026-07-21T00:00:00+00:00"
    provenance = manifest[IMMUTABLE_SOURCE_SNAPSHOT_KEY]
    assert provenance["hard_linked_partitions"] == 122
    assert provenance["copied_partitions"] == 0
    assert provenance["partition_checksums_verified"] is True
    source_partition = expanded_source / "2026-03-21.parquet"
    snapshot_partition = destination / "2026-03-21.parquet"
    assert os.stat(source_partition).st_ino == os.stat(snapshot_partition).st_ino
    assert load_core_manifest(config, "pre_holdout") == manifest

    with pytest.raises(
        RuntimeError,
        match="immutable source snapshot cannot be rebuilt or overwritten",
    ):
        extract_core_source(config, "pre_holdout", force=True)
    with pytest.raises(FileExistsError, match="already exists"):
        snapshot_residual_admission_source(config, expanded_source)


def test_snapshot_uses_exclusive_copy_when_hard_links_are_unavailable(
    tmp_path: Path,
    expanded_source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "copied-snapshot"
    config = snapshot_config(destination)

    def unavailable(_source: Path, _destination: Path) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(core_extract, "_hard_link_partition", unavailable)
    manifest = snapshot_residual_admission_source(config, expanded_source)

    provenance = manifest[IMMUTABLE_SOURCE_SNAPSHOT_KEY]
    assert provenance["hard_linked_partitions"] == 0
    assert provenance["copied_partitions"] == 122
    source_partition = expanded_source / "2026-03-21.parquet"
    snapshot_partition = destination / "2026-03-21.parquet"
    assert os.stat(source_partition).st_ino != os.stat(snapshot_partition).st_ino
    assert file_sha256(source_partition) == file_sha256(snapshot_partition)


def test_snapshot_fails_before_destination_creation_on_checksum_mismatch(
    tmp_path: Path,
    expanded_source: Path,
) -> None:
    tampered_source = tmp_path / "tampered-source"
    tampered_source.mkdir()
    for source_path in expanded_source.iterdir():
        if source_path.name == "manifest-pre_holdout.json":
            continue
        os.link(source_path, tampered_source / source_path.name)
    manifest = json.loads(
        (expanded_source / "manifest-pre_holdout.json").read_text()
    )
    manifest["partitions"][0]["sha256"] = "0" * 64
    (tampered_source / "manifest-pre_holdout.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    destination = tmp_path / "rejected-snapshot"
    config = snapshot_config(destination)

    with pytest.raises(RuntimeError, match="partition checksum mismatch"):
        snapshot_residual_admission_source(config, tampered_source)

    assert not destination.exists()
