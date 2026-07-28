from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from kraken_ml.config import load_config
from kraken_ml.funding_backfill import (
    FundingRate,
    backfill_funding,
    merge_sources,
    normalize_to_buckets,
    parse_archive,
    parse_recent_json,
)


def _archive_bytes(rows: list[tuple[str, str, str]]) -> bytes:
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(["timestamp", "tradeable", "absolute_rate", "relative_rate"])
    for timestamp, absolute, relative in rows:
        writer.writerow([timestamp, "PF_XBTUSD", absolute, relative])
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("exports/PF_XBTUSD.csv", csv_buffer.getvalue())
    return archive_buffer.getvalue()


def _recent_bytes(rows: list[tuple[str, str, str]]) -> bytes:
    encoded_rows = [
        (
            '{"timestamp":'
            f"{json.dumps(timestamp)},"
            f'"fundingRate":{absolute},'
            f'"relativeFundingRate":{relative}'
            "}"
        )
        for timestamp, absolute, relative in rows
    ]
    return (
        '{"result":"success","serverTime":"2024-01-01T05:00:00.000Z","rates":['
        + ",".join(encoded_rows)
        + "]}"
    ).encode()


def _config(config_path: Path, tmp_path: Path):
    config = load_config(config_path)
    return replace(
        config,
        dataset=replace(
            config.dataset,
            lake_root=tmp_path / "provider=kraken_futures",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, 5, tzinfo=UTC),
        ),
    )


def _write_source_files(
    tmp_path: Path,
    *,
    recent_relative: str = "0.000002000000000003",
) -> tuple[Path, Path]:
    archive_path = tmp_path / "funding.zip"
    archive_path.write_bytes(
        _archive_bytes(
            [
                ("2023-12-31 20:00:00", "0.100000000000000001", "0.000001000000000001"),
                ("2024-01-01 02:00:00", "0.200000000000000003", recent_relative),
                ("2024-01-01 04:00:00", "-0.300000000000000007", "-0.000003000000000007"),
            ]
        )
    )
    recent_path = tmp_path / "recent.json"
    recent_path.write_bytes(
        _recent_bytes(
            [
                (
                    "2024-01-01T02:00:00.000Z",
                    "0.200000000000000003",
                    "0.000002000000000003",
                ),
                (
                    "2024-01-01T04:00:00.000Z",
                    "-0.300000000000000007",
                    "-0.000003000000000007",
                ),
            ]
        )
    )
    return archive_path, recent_path


def _write_existing(
    config,
    rows: list[tuple[datetime, str, str]],
) -> Path:
    path = (
        config.dataset.lake_root
        / "dataset=funding_rates"
        / "symbol=PF_XBTUSD"
        / "interval_seconds=900"
        / "year=2024"
        / "month=01"
        / ("existing_" + "0" * 64 + ".parquet")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "observed_at_ms": [int(timestamp.timestamp() * 1000) for timestamp, _, _ in rows],
            "dataset": ["funding_rates"] * len(rows),
            "symbol": ["PF_XBTUSD"] * len(rows),
            "interval_seconds": [900] * len(rows),
            "payload_json": [
                json.dumps(
                    {
                        "funding_rate": absolute,
                        "relative_funding_rate": relative,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for _, absolute, relative in rows
            ],
        },
        schema={
            "observed_at_ms": pl.Int64,
            "dataset": pl.String,
            "symbol": pl.String,
            "interval_seconds": pl.Int32,
            "payload_json": pl.String,
        },
    ).write_parquet(path)
    return path


def test_sources_parse_exact_decimals_and_require_exact_overlap() -> None:
    archive = parse_archive(
        _archive_bytes(
            [
                ("2022-03-22 16:00:00", "0.858943191939791995", "0.000020164637711864"),
                ("2022-03-22 20:00:00", "-0.084584032794196903", "-0.000001999193855932"),
            ]
        ),
        symbol="PF_XBTUSD",
    )
    recent = parse_recent_json(
        _recent_bytes(
            [
                (
                    "2022-03-22T20:00:00.000Z",
                    "-0.084584032794196903",
                    "-0.000001999193855932",
                )
            ]
        ),
        symbol="PF_XBTUSD",
    )

    merged, overlap = merge_sources(archive.rates, recent.rates)

    assert overlap == 1
    assert merged[datetime(2022, 3, 22, 16, tzinfo=UTC)].absolute == Decimal(
        "0.858943191939791995"
    )
    assert merged[datetime(2022, 3, 22, 20, tzinfo=UTC)].relative == Decimal(
        "-0.000001999193855932"
    )


def test_source_overlap_rejects_any_decimal_difference() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    archive = {timestamp: FundingRate(Decimal("1.0"), Decimal("0.000001"))}
    recent = {timestamp: FundingRate(Decimal("1.0"), Decimal("0.000001000000000001"))}

    with pytest.raises(RuntimeError, match="sources conflict"):
        merge_sources(archive, recent)


def test_normalization_carries_active_rate_to_every_15m_bucket() -> None:
    first = FundingRate(Decimal("1"), Decimal("0.000001"))
    second = FundingRate(Decimal("2"), Decimal("0.000002"))
    start = datetime(2024, 1, 1, tzinfo=UTC)

    normalized = normalize_to_buckets(
        {
            start - timedelta(hours=4): first,
            start + timedelta(hours=2): second,
        },
        start=start,
        end=start + timedelta(hours=3),
        interval_seconds=900,
    )

    assert len(normalized) == 12
    assert normalized[start + timedelta(hours=1, minutes=45)] == first
    assert normalized[start + timedelta(hours=2)] == second
    assert normalized[start + timedelta(hours=2, minutes=45)] == second


def test_backfill_publishes_only_missing_rows_and_is_idempotent(
    config_path: Path, tmp_path: Path
) -> None:
    config = _config(config_path, tmp_path)
    archive_path, recent_path = _write_source_files(tmp_path)
    existing_rows = [
        (
            datetime(2024, 1, 1, 4, minute, tzinfo=UTC),
            "-0.3000000000000000",
            "-0.000003000000000007",
        )
        for minute in (0, 15, 30, 45)
    ]
    existing_path = _write_existing(config, existing_rows)
    existing_bytes = existing_path.read_bytes()

    first = backfill_funding(
        config,
        archive_path=archive_path,
        recent_json_path=recent_path,
    )
    parquet_paths = sorted(
        (
            config.dataset.lake_root
            / "dataset=funding_rates"
            / "symbol=PF_XBTUSD"
            / "interval_seconds=900"
        ).rglob("*.parquet")
    )
    second = backfill_funding(
        config,
        archive_path=archive_path,
        recent_json_path=recent_path,
    )

    assert first.expected_rows == 20
    assert first.preexisting_rows == 4
    assert first.published_rows == 16
    assert first.published_objects == 1
    assert first.overlap_rows == 2
    assert not first.idempotent_replay
    assert second.import_id == first.import_id
    assert second.idempotent_replay
    assert sorted(
        (
            config.dataset.lake_root
            / "dataset=funding_rates"
            / "symbol=PF_XBTUSD"
            / "interval_seconds=900"
        ).rglob("*.parquet")
    ) == parquet_paths
    assert existing_path.read_bytes() == existing_bytes
    assert first.manifest_path.read_text() == second.manifest_path.read_text()

    frame = pl.read_parquet([str(path) for path in parquet_paths])
    assert frame["observed_at_ms"].n_unique() == 20
    assert frame.height == 20
    assert set(frame.columns) == {
        "observed_at_ms",
        "dataset",
        "symbol",
        "interval_seconds",
        "payload_json",
    }
    provenance = config.dataset.lake_root / "_provenance" / "funding_rates"
    assert len(list((provenance / "sources").iterdir())) == 2
    assert len(list((provenance / "manifests").glob("*.json"))) == 1
    manifest = json.loads(first.manifest_path.read_text())
    assert manifest["existing_lake_reconciliation"] == {
        "absolute_rate_max_difference": "0.000000000001",
        "reason": (
            "Kraken's 15-minute charts source truncates absolute funding "
            "rates while preserving relative funding rates exactly."
        ),
        "relative_rate_match": "exact_decimal_equality",
    }


def test_backfill_rejects_existing_conflict_before_publication(
    config_path: Path, tmp_path: Path
) -> None:
    config = _config(config_path, tmp_path)
    archive_path, recent_path = _write_source_files(tmp_path)
    _write_existing(
        config,
        [
            (
                datetime(2024, 1, 1, tzinfo=UTC),
                "999",
                "0.000001000000000001",
            )
        ],
    )

    with pytest.raises(RuntimeError, match="conflicts with 1 existing lake rows"):
        backfill_funding(
            config,
            archive_path=archive_path,
            recent_json_path=recent_path,
        )

    objects = list(
        (
            config.dataset.lake_root
            / "dataset=funding_rates"
            / "symbol=PF_XBTUSD"
            / "interval_seconds=900"
        ).rglob("*.parquet")
    )
    assert len(objects) == 1
    assert not (
        config.dataset.lake_root / "_provenance" / "funding_rates" / "manifests"
    ).exists()


@pytest.mark.parametrize(
    ("absolute", "relative"),
    [
        ("0.100000000001000002", "0.000001000000000001"),
        ("0.100000000000000001", "0.000001000000000002"),
    ],
)
def test_backfill_rejects_existing_rate_outside_canonical_tolerance(
    config_path: Path,
    tmp_path: Path,
    absolute: str,
    relative: str,
) -> None:
    config = _config(config_path, tmp_path)
    archive_path, recent_path = _write_source_files(tmp_path)
    _write_existing(
        config,
        [
            (
                datetime(2024, 1, 1, tzinfo=UTC),
                absolute,
                relative,
            )
        ],
    )

    with pytest.raises(RuntimeError, match="conflicts with 1 existing lake rows"):
        backfill_funding(
            config,
            archive_path=archive_path,
            recent_json_path=recent_path,
        )


def test_backfill_requires_source_at_configured_range_start(
    config_path: Path, tmp_path: Path
) -> None:
    config = _config(config_path, tmp_path)
    archive_path = tmp_path / "funding.zip"
    archive_path.write_bytes(
        _archive_bytes(
            [("2024-01-01 02:00:00", "0.2", "0.000002")]
        )
    )
    recent_path = tmp_path / "recent.json"
    recent_path.write_bytes(
        _recent_bytes(
            [("2024-01-01T02:00:00.000Z", "0.2", "0.000002")]
        )
    )

    with pytest.raises(RuntimeError, match="no active rate at configured range start"):
        backfill_funding(
            config,
            archive_path=archive_path,
            recent_json_path=recent_path,
        )
