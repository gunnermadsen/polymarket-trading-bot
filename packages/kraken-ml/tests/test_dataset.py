from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

import kraken_ml.dataset as dataset_module
from kraken_ml.config import load_config, load_expectancy_config
from kraken_ml.dataset import (
    _canonical_sha256,
    _immutable_json,
    _json_text,
    _prepare_lake_snapshot,
    _sha256,
    _verified_funding_provenance,
    validate_funding_provenance_binding,
    validate_snapshot_frame,
)


def _frame_config(config_path: Path, frame: pl.DataFrame):
    config = load_config(config_path)
    interval = timedelta(seconds=config.dataset.interval_seconds)
    return replace(
        config,
        dataset=replace(
            config.dataset,
            start=frame.item(0, "bucket_start"),
            end=frame.item(frame.height - 1, "bucket_start") + interval,
        ),
    )


def _write_funding_provenance(config) -> tuple[dict, Path, Path]:
    lake_root = config.dataset.lake_root
    provenance_root = lake_root / "_provenance" / "funding_rates"
    sources_root = provenance_root / "sources"
    sources_root.mkdir(parents=True)
    source_values = {
        "official_csv_archive": b"official archive bytes",
        "official_historical_funding_rates_api": b'{"result":"success"}',
    }
    source_entries = []
    source_hashes = {}
    source_paths = []
    for kind, content in source_values.items():
        sha256 = hashlib.sha256(content).hexdigest()
        suffix = ".zip" if kind == "official_csv_archive" else ".json"
        source_path = sources_root / f"{sha256}{suffix}"
        source_path.write_bytes(content)
        source_paths.append(source_path)
        source_hashes[kind] = sha256
        source_entries.append(
            {
                "kind": kind,
                "origin": f"https://example.test/{source_path.name}",
                "sha256": sha256,
                "byte_size": len(content),
                "immutable_relative_path": f"sources/{source_path.name}",
                "rows": 2,
                "first_timestamp": config.dataset.start.isoformat(),
                "last_timestamp": config.dataset.end.isoformat(),
            }
        )

    object_path = (
        lake_root
        / "dataset=funding_rates"
        / f"symbol={config.dataset.symbol}"
        / f"interval_seconds={config.dataset.interval_seconds}"
        / "year=2024"
        / "month=01"
        / "funding.parquet"
    )
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(b"normalized funding parquet bytes")
    object_sha256 = _sha256(object_path)
    expected_rows = int(
        (config.dataset.end - config.dataset.start).total_seconds()
        // config.dataset.interval_seconds
    )
    import_id = _canonical_sha256(
        {
            "policy": "continuous-hourly-rate-locf-15m-v1",
            "archive_sha256": source_hashes["official_csv_archive"],
            "recent_sha256": source_hashes[
                "official_historical_funding_rates_api"
            ],
            "provider": "kraken_futures",
            "dataset": "funding_rates",
            "symbol": config.dataset.symbol,
            "interval_seconds": config.dataset.interval_seconds,
            "start": config.dataset.start.isoformat(),
            "end": config.dataset.end.isoformat(),
        }
    )
    manifest = {
        "schema_version": 1,
        "import_id": import_id,
        "provider": "kraken_futures",
        "dataset": "funding_rates",
        "symbol": config.dataset.symbol,
        "interval_seconds": config.dataset.interval_seconds,
        "configured_range": {
            "start": config.dataset.start.isoformat(),
            "end_exclusive": config.dataset.end.isoformat(),
            "expected_rows": expected_rows,
            "first_timestamp": config.dataset.start.isoformat(),
            "last_timestamp": (
                config.dataset.end
                - timedelta(seconds=config.dataset.interval_seconds)
            ).isoformat(),
        },
        "sources": source_entries,
        "normalization": {
            "policy": "continuous-hourly-rate-locf-15m-v1",
        },
        "existing_lake_reconciliation": {
            "relative_rate_match": "exact_decimal_equality",
        },
        "lake_mutation": {
            "preexisting_rows": 0,
            "published_rows": expected_rows,
            "published_objects": [
                {
                    "relative_path": str(object_path.relative_to(lake_root)),
                    "sha256": object_sha256,
                    "row_count": expected_rows,
                    "first_timestamp": config.dataset.start.isoformat(),
                    "last_timestamp": (
                        config.dataset.end
                        - timedelta(seconds=config.dataset.interval_seconds)
                    ).isoformat(),
                }
            ],
        },
        "coverage_validation": {
            "complete": True,
            "validated_rows": expected_rows,
            "missing_rows": 0,
            "conflicting_rows": 0,
        },
    }
    manifest_path = provenance_root / "manifests" / f"{import_id}.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, manifest_path, source_paths[0]


def test_snapshot_validation_requires_exact_configured_time_range(
    config_path: Path, raw_market_frame: pl.DataFrame
) -> None:
    config = _frame_config(config_path, raw_market_frame)
    validate_snapshot_frame(raw_market_frame, config)

    with pytest.raises(RuntimeError, match="row count"):
        validate_snapshot_frame(raw_market_frame.slice(1), config)


def test_snapshot_validation_requires_two_sided_slippage(
    config_path: Path, raw_market_frame: pl.DataFrame
) -> None:
    config = _frame_config(config_path, raw_market_frame)
    bid = raw_market_frame["bid_slippage_1k"].to_list()
    bid[10] = None
    incomplete = raw_market_frame.with_columns(pl.Series("bid_slippage_1k", bid))

    with pytest.raises(RuntimeError, match="two-sided coverage"):
        validate_snapshot_frame(incomplete, config)


def test_content_addressed_manifest_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    _immutable_json(path, {"schema_version": 1, "value": "fixed"})
    _immutable_json(path, {"schema_version": 1, "value": "fixed"})

    with pytest.raises(RuntimeError, match="immutable manifest content changed"):
        _immutable_json(path, {"schema_version": 1, "value": "changed"})


def test_json_artifacts_are_strict_and_normalize_nonfinite_values() -> None:
    payload = _json_text({"finite": 1.0, "nonfinite": float("nan")})

    assert '"finite": 1.0' in payload
    assert '"nonfinite": null' in payload
    assert "NaN" not in payload


def test_expectancy_snapshot_rejects_missing_funding(
    raw_market_frame: pl.DataFrame,
) -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "pf_xbtusd_15m_expectancy.toml"
    config = load_expectancy_config(path)
    interval = timedelta(seconds=config.dataset.interval_seconds)
    config = replace(
        config,
        dataset=replace(
            config.dataset,
            start=raw_market_frame.item(0, "bucket_start"),
            end=raw_market_frame.item(raw_market_frame.height - 1, "bucket_start") + interval,
        ),
    )
    rates = raw_market_frame["relative_funding_rate"].to_list()
    rates[10] = None
    incomplete = raw_market_frame.with_columns(pl.Series("relative_funding_rate", rates))

    with pytest.raises(RuntimeError, match="complete first-party funding"):
        validate_snapshot_frame(incomplete, config)


def test_first_party_funding_provenance_binds_all_immutable_objects(
    raw_market_frame: pl.DataFrame,
    tmp_path: Path,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "pf_xbtusd_15m_expectancy.toml"
    )
    config = load_expectancy_config(config_path)
    interval = timedelta(seconds=config.dataset.interval_seconds)
    config = replace(
        config,
        dataset=replace(
            config.dataset,
            lake_root=tmp_path / "provider=kraken_futures",
            start=raw_market_frame.item(0, "bucket_start"),
            end=raw_market_frame.item(raw_market_frame.height - 1, "bucket_start")
            + interval,
        ),
    )
    manifest, manifest_path, source_path = _write_funding_provenance(config)
    config = replace(
        config,
        funding_provenance=replace(
            config.funding_provenance,
            import_id=manifest["import_id"],
        ),
    )
    unrelated_manifest = manifest_path.parent / f"{'f' * 64}.json"
    unrelated_manifest.write_text("not valid JSON", encoding="utf-8")

    binding = _verified_funding_provenance(config)

    assert binding["manifest"]["sha256"] == _sha256(manifest_path)
    assert binding["binding_sha256"] == _canonical_sha256(
        {key: value for key, value in binding.items() if key != "binding_sha256"}
    )
    assert {source["kind"] for source in binding["sources"]} == {
        "official_csv_archive",
        "official_historical_funding_rates_api",
    }
    assert len(binding["published_objects"]) == 1

    source_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="source checksum mismatch"):
        _verified_funding_provenance(config)


def test_expectancy_raw_snapshot_embeds_verified_funding_binding(
    raw_market_frame: pl.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "pf_xbtusd_15m_expectancy.toml"
    )
    config = load_expectancy_config(config_path)
    interval = timedelta(seconds=config.dataset.interval_seconds)
    config = replace(
        config,
        dataset=replace(
            config.dataset,
            lake_root=tmp_path / "provider=kraken_futures",
            start=raw_market_frame.item(0, "bucket_start"),
            end=raw_market_frame.item(raw_market_frame.height - 1, "bucket_start")
            + interval,
        ),
        artifacts=replace(config.artifacts, root=tmp_path / "artifacts"),
    )
    manifest_payload, _, _ = _write_funding_provenance(config)
    config = replace(
        config,
        funding_provenance=replace(
            config.funding_provenance,
            import_id=manifest_payload["import_id"],
        ),
    )
    funding_object = next(
        (config.dataset.lake_root / "dataset=funding_rates").rglob("*.parquet")
    )
    monkeypatch.setattr(
        dataset_module,
        "_verified_lake_files",
        lambda unused_config: ([funding_object], "a" * 64),
    )
    monkeypatch.setattr(
        dataset_module,
        "_verified_contract_metadata",
        lambda unused_config: {"symbol": config.dataset.symbol},
    )
    monkeypatch.setattr(
        dataset_module,
        "_lake_source_frame",
        lambda unused_config: raw_market_frame,
    )

    snapshot = _prepare_lake_snapshot(config, refresh=True)
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    verified = _verified_funding_provenance(config)

    assert manifest["schema_version"] == 2
    assert manifest["funding_provenance"] == verified
    assert manifest["funding_coverage"]["complete"]
    assert manifest["funding_coverage"]["missing_rows"] == 0
    assert validate_funding_provenance_binding(manifest, config) == verified

    manifest["funding_provenance"]["binding_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="binding checksum is invalid"):
        validate_funding_provenance_binding(manifest, config)
