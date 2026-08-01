from __future__ import annotations

import bisect
import csv
import fcntl
import hashlib
import io
import json
import os
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import polars as pl

from .config import BenchmarkConfig
from .dataset import _immutable_json

ARCHIVE_URL = (
    "https://assets-cms.kraken.com/files/51n36hrp/facade/"
    "4b70936c1227e4ae5514cba8bf41a5561cf13bd7.zip?dl="
)
RECENT_URL_TEMPLATE = (
    "https://futures.kraken.com/derivatives/api/v3/"
    "historical-funding-rates?symbol={symbol}"
)
FUNDING_DATASET = "funding_rates"
PROVIDER = "kraken_futures"
IMPORT_POLICY = "continuous-hourly-rate-locf-15m-v1"
CANONICAL_COLUMNS = (
    "observed_at_ms",
    "dataset",
    "symbol",
    "interval_seconds",
    "payload_json",
)
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
CANONICAL_ABSOLUTE_RATE_TOLERANCE = Decimal("0.000000000001")


@dataclass(frozen=True)
class FundingRate:
    absolute: Decimal
    relative: Decimal


@dataclass(frozen=True)
class ParsedSource:
    rates: dict[datetime, FundingRate]
    source_rows: int
    first_timestamp: datetime
    last_timestamp: datetime


@dataclass(frozen=True)
class SourceObject:
    kind: str
    origin: str
    sha256: str
    byte_size: int
    relative_path: str
    parsed: ParsedSource


@dataclass(frozen=True)
class FundingBackfillResult:
    import_id: str
    manifest_path: Path
    expected_rows: int
    preexisting_rows: int
    published_rows: int
    published_objects: int
    overlap_rows: int
    first_timestamp: datetime
    last_timestamp: datetime
    idempotent_replay: bool

    def summary(self) -> dict[str, Any]:
        return {
            "import_id": self.import_id,
            "manifest_path": str(self.manifest_path),
            "expected_rows": self.expected_rows,
            "preexisting_rows": self.preexisting_rows,
            "published_rows": self.published_rows,
            "published_objects": self.published_objects,
            "overlap_rows": self.overlap_rows,
            "first_timestamp": self.first_timestamp.isoformat(),
            "last_timestamp": self.last_timestamp.isoformat(),
            "idempotent_replay": self.idempotent_replay,
        }


def backfill_funding(
    config: BenchmarkConfig,
    *,
    archive_path: Path | None = None,
    recent_json_path: Path | None = None,
) -> FundingBackfillResult:
    """Import first-party Kraken funding history into missing canonical lake buckets."""
    if config.dataset.interval_seconds != 900:
        raise ValueError("funding backfill requires the canonical 900-second lake interval")
    if config.dataset.symbol != "PF_XBTUSD":
        raise ValueError("the official archive importer is intentionally scoped to PF_XBTUSD")

    archive_bytes, archive_origin = _source_bytes(archive_path, ARCHIVE_URL)
    recent_url = RECENT_URL_TEMPLATE.format(symbol=config.dataset.symbol)
    recent_bytes, recent_origin = _source_bytes(recent_json_path, recent_url)
    archive_parsed = parse_archive(archive_bytes, symbol=config.dataset.symbol)
    recent_parsed = parse_recent_json(recent_bytes, symbol=config.dataset.symbol)
    merged, overlap_rows = merge_sources(archive_parsed.rates, recent_parsed.rates)
    desired = normalize_to_buckets(
        merged,
        start=config.dataset.start,
        end=config.dataset.end,
        interval_seconds=config.dataset.interval_seconds,
    )

    provenance_root = config.dataset.lake_root / "_provenance" / FUNDING_DATASET
    lock_path = provenance_root / ".import.lock"
    with _exclusive_lock(lock_path):
        archive_object = _publish_source(
            provenance_root,
            kind="official_csv_archive",
            origin=archive_origin,
            suffix=".zip",
            content=archive_bytes,
            parsed=archive_parsed,
        )
        recent_object = _publish_source(
            provenance_root,
            kind="official_historical_funding_rates_api",
            origin=recent_origin,
            suffix=".json",
            content=recent_bytes,
            parsed=recent_parsed,
        )
        import_id = _import_identity(config, archive_object.sha256, recent_object.sha256)
        manifest_path = provenance_root / "manifests" / f"{import_id}.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _validate_replay_manifest(
                manifest,
                config=config,
                import_id=import_id,
                desired=desired,
                overlap_rows=overlap_rows,
            )
            return _result_from_manifest(manifest_path, manifest, idempotent_replay=True)

        existing = _load_existing_rates(config)
        missing = _missing_without_conflicts(desired, existing)
        published = _publish_missing(config, missing)
        final_rates = _load_existing_rates(config)
        _validate_complete_coverage(config, desired, final_rates)

        first_timestamp = min(desired)
        last_timestamp = max(desired)
        manifest = {
            "schema_version": 1,
            "import_id": import_id,
            "importer": "kraken_ml.funding_backfill",
            "provider": PROVIDER,
            "dataset": FUNDING_DATASET,
            "symbol": config.dataset.symbol,
            "interval_seconds": config.dataset.interval_seconds,
            "configured_range": {
                "start": config.dataset.start.isoformat(),
                "end_exclusive": config.dataset.end.isoformat(),
                "expected_rows": len(desired),
                "first_timestamp": first_timestamp.isoformat(),
                "last_timestamp": last_timestamp.isoformat(),
            },
            "sources": [
                _source_manifest(archive_object),
                _source_manifest(recent_object),
            ],
            "source_reconciliation": {
                "overlap_rows": overlap_rows,
                "absolute_rate_match": "exact_decimal_equality",
                "relative_rate_match": "exact_decimal_equality",
            },
            "existing_lake_reconciliation": {
                "relative_rate_match": "exact_decimal_equality",
                "absolute_rate_max_difference": format(
                    CANONICAL_ABSOLUTE_RATE_TOLERANCE, "f"
                ),
                "reason": (
                    "Kraken's 15-minute charts source truncates absolute funding "
                    "rates while preserving relative funding rates exactly."
                ),
            },
            "normalization": {
                "policy": IMPORT_POLICY,
                "description": (
                    "Each first-party observation is the active continuously accrued "
                    "per-hour rate until the next observation; missing canonical "
                    "15-minute buckets use last-observation carried forward."
                ),
                "float_conversion": False,
            },
            "lake_mutation": {
                "preexisting_rows": len(desired) - len(missing),
                "published_rows": len(missing),
                "published_objects": published,
                "overwrite_policy": "never",
                "conflict_policy": (
                    "fail before Parquet publication on any relative-rate difference "
                    "or absolute-rate difference above the documented tolerance"
                ),
            },
            "coverage_validation": {
                "complete": True,
                "validated_rows": len(desired),
                "missing_rows": 0,
                "conflicting_rows": 0,
            },
        }
        _immutable_json(manifest_path, manifest)
        return _result_from_manifest(manifest_path, manifest, idempotent_replay=False)


def parse_archive(content: bytes, *, symbol: str) -> ParsedSource:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            expected = f"exports/{symbol}.csv"
            matches = [
                name
                for name in archive.namelist()
                if name.removeprefix("./") == expected
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"official funding ZIP must contain exactly one {expected}; "
                    f"found {len(matches)}"
                )
            csv_bytes = archive.read(matches[0])
    except zipfile.BadZipFile as error:
        raise ValueError("official funding archive is not a valid ZIP") from error

    try:
        text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("official funding CSV is not UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    expected_headers = {"timestamp", "tradeable", "absolute_rate", "relative_rate"}
    if reader.fieldnames is None or set(reader.fieldnames) != expected_headers:
        raise ValueError(
            "official funding CSV headers changed: "
            f"expected {sorted(expected_headers)}, got {reader.fieldnames}"
        )

    rates: dict[datetime, FundingRate] = {}
    for row_number, row in enumerate(reader, start=2):
        if row["tradeable"].strip().upper() != symbol:
            raise ValueError(
                f"official funding CSV row {row_number} has unexpected tradeable "
                f"{row['tradeable']!r}"
            )
        timestamp = _parse_timestamp(
            row["timestamp"], source=f"official funding CSV row {row_number}", assume_utc=True
        )
        rate = FundingRate(
            absolute=_parse_decimal(
                row["absolute_rate"], source=f"official funding CSV row {row_number} absolute_rate"
            ),
            relative=_parse_decimal(
                row["relative_rate"], source=f"official funding CSV row {row_number} relative_rate"
            ),
        )
        _insert_exact(rates, timestamp, rate, source=f"official funding CSV row {row_number}")
    return _parsed_source(rates, source="official funding CSV")


def parse_recent_json(content: bytes, *, symbol: str) -> ParsedSource:
    del symbol  # The endpoint is symbol-filtered and the response does not repeat the symbol.

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        payload = json.loads(
            content.decode("utf-8"),
            parse_float=Decimal,
            parse_int=Decimal,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("historical-funding-rates response is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict) or payload.get("result") != "success":
        raise ValueError("historical-funding-rates response did not report success")
    rows = payload.get("rates")
    if not isinstance(rows, list):
        raise ValueError("historical-funding-rates response omitted rates")

    rates: dict[datetime, FundingRate] = {}
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"historical-funding-rates row {row_number} is not an object")
        missing = {"timestamp", "fundingRate", "relativeFundingRate"} - row.keys()
        if missing:
            raise ValueError(
                f"historical-funding-rates row {row_number} omitted {sorted(missing)}"
            )
        timestamp = _parse_timestamp(
            row["timestamp"], source=f"historical-funding-rates row {row_number}"
        )
        rate = FundingRate(
            absolute=_parse_decimal(
                row["fundingRate"],
                source=f"historical-funding-rates row {row_number} fundingRate",
            ),
            relative=_parse_decimal(
                row["relativeFundingRate"],
                source=f"historical-funding-rates row {row_number} relativeFundingRate",
            ),
        )
        _insert_exact(
            rates,
            timestamp,
            rate,
            source=f"historical-funding-rates row {row_number}",
        )
    return _parsed_source(rates, source="historical-funding-rates response")


def merge_sources(
    archive: dict[datetime, FundingRate],
    recent: dict[datetime, FundingRate],
) -> tuple[dict[datetime, FundingRate], int]:
    overlap = sorted(archive.keys() & recent.keys())
    if not overlap:
        raise RuntimeError(
            "official funding archive and historical-funding-rates response do not overlap"
        )
    for timestamp in overlap:
        if archive[timestamp] != recent[timestamp]:
            raise RuntimeError(
                "official funding sources conflict at "
                f"{timestamp.isoformat()}: {archive[timestamp]} != {recent[timestamp]}"
            )
    merged = dict(archive)
    merged.update(recent)
    return merged, len(overlap)


def normalize_to_buckets(
    rates: dict[datetime, FundingRate],
    *,
    start: datetime,
    end: datetime,
    interval_seconds: int,
) -> dict[datetime, FundingRate]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("configured funding range must be timezone-aware")
    if start >= end:
        raise ValueError("configured funding range is empty")
    if interval_seconds <= 0:
        raise ValueError("funding interval must be positive")
    source_times = sorted(rates)
    if not source_times:
        raise RuntimeError("official funding sources contain no rates")

    interval = timedelta(seconds=interval_seconds)
    expected_rows, remainder = divmod(
        int((end - start).total_seconds()), interval_seconds
    )
    if remainder:
        raise ValueError("configured funding range is not interval-aligned")
    desired: dict[datetime, FundingRate] = {}
    source_index = bisect.bisect_right(source_times, start) - 1
    if source_index < 0:
        raise RuntimeError(
            "official funding history has no active rate at configured range start "
            f"{start.isoformat()}"
        )
    for offset in range(expected_rows):
        bucket = start + offset * interval
        while (
            source_index + 1 < len(source_times)
            and source_times[source_index + 1] <= bucket
        ):
            source_index += 1
        desired[bucket] = rates[source_times[source_index]]
    return desired


def _source_bytes(path: Path | None, url: str) -> tuple[bytes, str]:
    if path is not None:
        resolved = path.expanduser().resolve()
        try:
            content = resolved.read_bytes()
        except OSError as error:
            raise RuntimeError(f"failed to read funding source {resolved}: {error}") from error
        if len(content) > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(f"funding source exceeds {MAX_DOWNLOAD_BYTES} bytes: {resolved}")
        return content, str(resolved)
    request = urllib.request.Request(url, headers={"User-Agent": "kraken-ml/0.2"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            content = response.read(MAX_DOWNLOAD_BYTES + 1)
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(
            f"failed to download first-party funding source {url}: {error}"
        ) from error
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise RuntimeError(f"first-party funding source exceeds {MAX_DOWNLOAD_BYTES} bytes")
    return content, url


def _parse_timestamp(value: Any, *, source: str, assume_utc: bool = False) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{source} timestamp must be a string")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{source} has invalid timestamp {value!r}") from error
    if parsed.tzinfo is None:
        if not assume_utc:
            raise ValueError(f"{source} timestamp must include a timezone")
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    if int(parsed.timestamp()) % 900 or parsed.microsecond:
        raise ValueError(f"{source} timestamp is not aligned to a 15-minute UTC bucket")
    return parsed


def _parse_decimal(value: Any, *, source: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError(f"{source} must be an exact decimal string or JSON number")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{source} is not a valid decimal: {value!r}") from error
    if not parsed.is_finite():
        raise ValueError(f"{source} must be finite")
    return parsed


def _insert_exact(
    rates: dict[datetime, FundingRate],
    timestamp: datetime,
    rate: FundingRate,
    *,
    source: str,
) -> None:
    existing = rates.get(timestamp)
    if existing is not None and existing != rate:
        raise ValueError(f"{source} conflicts with a duplicate timestamp")
    rates[timestamp] = rate


def _parsed_source(rates: dict[datetime, FundingRate], *, source: str) -> ParsedSource:
    if not rates:
        raise ValueError(f"{source} contains no funding rates")
    timestamps = sorted(rates)
    return ParsedSource(
        rates=rates,
        source_rows=len(rates),
        first_timestamp=timestamps[0],
        last_timestamp=timestamps[-1],
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_source(
    provenance_root: Path,
    *,
    kind: str,
    origin: str,
    suffix: str,
    content: bytes,
    parsed: ParsedSource,
) -> SourceObject:
    sha256 = _sha256_bytes(content)
    relative_path = Path("sources") / f"{sha256}{suffix}"
    path = provenance_root / relative_path
    _immutable_bytes(path, content)
    return SourceObject(
        kind=kind,
        origin=origin,
        sha256=sha256,
        byte_size=len(content),
        relative_path=str(relative_path),
        parsed=parsed,
    )


def _immutable_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError as error:
        if _sha256_file(path) != _sha256_bytes(content):
            raise RuntimeError(f"immutable source object content changed: {path}") from error
        return
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _source_manifest(source: SourceObject) -> dict[str, Any]:
    return {
        "kind": source.kind,
        "origin": source.origin,
        "sha256": source.sha256,
        "byte_size": source.byte_size,
        "immutable_relative_path": source.relative_path,
        "rows": source.parsed.source_rows,
        "first_timestamp": source.parsed.first_timestamp.isoformat(),
        "last_timestamp": source.parsed.last_timestamp.isoformat(),
    }


def _import_identity(
    config: BenchmarkConfig,
    archive_sha256: str,
    recent_sha256: str,
) -> str:
    payload = {
        "policy": IMPORT_POLICY,
        "archive_sha256": archive_sha256,
        "recent_sha256": recent_sha256,
        "provider": PROVIDER,
        "dataset": FUNDING_DATASET,
        "symbol": config.dataset.symbol,
        "interval_seconds": config.dataset.interval_seconds,
        "start": config.dataset.start.isoformat(),
        "end": config.dataset.end.isoformat(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dataset_paths(config: BenchmarkConfig) -> list[Path]:
    root = (
        config.dataset.lake_root
        / f"dataset={FUNDING_DATASET}"
        / f"symbol={config.dataset.symbol}"
        / f"interval_seconds={config.dataset.interval_seconds}"
    )
    return sorted(root.rglob("*.parquet"))


def _load_existing_rates(config: BenchmarkConfig) -> dict[datetime, FundingRate]:
    paths = _dataset_paths(config)
    if not paths:
        return {}
    try:
        frame = pl.read_parquet(
            [str(path) for path in paths],
            columns=list(CANONICAL_COLUMNS),
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise RuntimeError(f"failed to read canonical funding lake: {error}") from error
    frame = frame.filter(
        (pl.col("dataset") == FUNDING_DATASET)
        & (pl.col("symbol") == config.dataset.symbol)
        & (pl.col("interval_seconds") == config.dataset.interval_seconds)
        & (
            pl.col("observed_at_ms")
            >= int(config.dataset.start.timestamp() * 1000)
        )
        & (pl.col("observed_at_ms") < int(config.dataset.end.timestamp() * 1000))
    )
    rates: dict[datetime, FundingRate] = {}
    for row in frame.iter_rows(named=True):
        timestamp = datetime.fromtimestamp(row["observed_at_ms"] / 1000, tz=UTC)
        try:
            payload = json.loads(
                row["payload_json"],
                parse_float=Decimal,
                parse_int=Decimal,
            )
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"canonical funding payload is invalid at {timestamp.isoformat()}"
            ) from error
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"canonical funding payload is not an object at {timestamp.isoformat()}"
            )
        try:
            rate = FundingRate(
                absolute=_parse_decimal(
                    payload["funding_rate"],
                    source=f"canonical funding rate at {timestamp.isoformat()}",
                ),
                relative=_parse_decimal(
                    payload["relative_funding_rate"],
                    source=f"canonical relative funding rate at {timestamp.isoformat()}",
                ),
            )
        except KeyError as error:
            raise RuntimeError(
                f"canonical funding payload omitted {error.args[0]} at {timestamp.isoformat()}"
            ) from error
        existing = rates.get(timestamp)
        if existing is not None and not _canonical_rate_matches(existing, rate):
            raise RuntimeError(
                f"canonical funding lake has conflicting rows at {timestamp.isoformat()}"
            )
        rates[timestamp] = rate
    return rates


def _missing_without_conflicts(
    desired: dict[datetime, FundingRate],
    existing: dict[datetime, FundingRate],
) -> dict[datetime, FundingRate]:
    missing: dict[datetime, FundingRate] = {}
    conflicts: list[datetime] = []
    for timestamp, rate in desired.items():
        current = existing.get(timestamp)
        if current is None:
            missing[timestamp] = rate
        elif not _canonical_rate_matches(current, rate):
            conflicts.append(timestamp)
    if conflicts:
        first = min(conflicts)
        raise RuntimeError(
            f"official funding rate conflicts with {len(conflicts)} existing lake rows; "
            f"first conflict {first.isoformat()}"
        )
    return missing


def _publish_missing(
    config: BenchmarkConfig,
    missing: dict[datetime, FundingRate],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[tuple[datetime, FundingRate]]] = {}
    for timestamp, rate in sorted(missing.items()):
        groups.setdefault((timestamp.year, timestamp.month), []).append((timestamp, rate))
    published: list[dict[str, Any]] = []
    for object_index, ((year, month), rows) in enumerate(sorted(groups.items())):
        object_info = _publish_month(config, year, month, rows, object_index=object_index)
        published.append(object_info)
    return published


def _publish_month(
    config: BenchmarkConfig,
    year: int,
    month: int,
    rows: list[tuple[datetime, FundingRate]],
    *,
    object_index: int,
) -> dict[str, Any]:
    payloads = [
        json.dumps(
            {
                "funding_rate": format(rate.absolute, "f"),
                "relative_funding_rate": format(rate.relative, "f"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for _, rate in rows
    ]
    frame = pl.DataFrame(
        {
            "observed_at_ms": [int(timestamp.timestamp() * 1000) for timestamp, _ in rows],
            "dataset": [FUNDING_DATASET] * len(rows),
            "symbol": [config.dataset.symbol] * len(rows),
            "interval_seconds": [config.dataset.interval_seconds] * len(rows),
            "payload_json": payloads,
        },
        schema={
            "observed_at_ms": pl.Int64,
            "dataset": pl.String,
            "symbol": pl.String,
            "interval_seconds": pl.Int32,
            "payload_json": pl.String,
        },
    )
    staging = config.dataset.lake_root / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / f"funding-{os.getpid()}-{object_index}.parquet.tmp"
    try:
        frame.write_parquet(
            temporary,
            compression="zstd",
            compression_level=6,
            statistics=True,
        )
        sha256 = _sha256_file(temporary)
        first = rows[0][0]
        end = rows[-1][0] + timedelta(seconds=config.dataset.interval_seconds)
        partition = (
            config.dataset.lake_root
            / f"dataset={FUNDING_DATASET}"
            / f"symbol={config.dataset.symbol}"
            / f"interval_seconds={config.dataset.interval_seconds}"
            / f"year={year}"
            / f"month={month:02d}"
        )
        partition.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{_compact_timestamp(first)}_{_compact_timestamp(end)}_{sha256}.parquet"
        )
        final_path = partition / filename
        if final_path.exists():
            if _sha256_file(final_path) != sha256:
                raise RuntimeError(
                    f"existing funding object does not match its content address: {final_path}"
                )
            temporary.unlink()
        else:
            os.replace(temporary, final_path)
        return {
            "relative_path": str(final_path.relative_to(config.dataset.lake_root)),
            "sha256": sha256,
            "row_count": len(rows),
            "first_timestamp": first.isoformat(),
            "last_timestamp": rows[-1][0].isoformat(),
        }
    finally:
        temporary.unlink(missing_ok=True)


def _compact_timestamp(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _validate_complete_coverage(
    config: BenchmarkConfig,
    desired: dict[datetime, FundingRate],
    final_rates: dict[datetime, FundingRate],
) -> None:
    missing = sorted(desired.keys() - final_rates.keys())
    conflicts = sorted(
        timestamp
        for timestamp, rate in desired.items()
        if timestamp in final_rates
        and not _canonical_rate_matches(final_rates[timestamp], rate)
    )
    if missing or conflicts:
        details = []
        if missing:
            details.append(f"{len(missing)} missing (first {missing[0].isoformat()})")
        if conflicts:
            details.append(
                f"{len(conflicts)} conflicting (first {conflicts[0].isoformat()})"
            )
        raise RuntimeError(
            "funding lake failed complete configured-range validation: " + ", ".join(details)
        )


def _canonical_rate_matches(left: FundingRate, right: FundingRate) -> bool:
    return (
        left.relative == right.relative
        and abs(left.absolute - right.absolute) <= CANONICAL_ABSOLUTE_RATE_TOLERANCE
    )


def _validate_replay_manifest(
    manifest: dict[str, Any],
    *,
    config: BenchmarkConfig,
    import_id: str,
    desired: dict[datetime, FundingRate],
    overlap_rows: int,
) -> None:
    if manifest.get("import_id") != import_id:
        raise RuntimeError("funding import manifest identity changed")
    if manifest.get("source_reconciliation", {}).get("overlap_rows") != overlap_rows:
        raise RuntimeError("funding import overlap count changed")
    for object_info in manifest.get("lake_mutation", {}).get("published_objects", []):
        path = config.dataset.lake_root / object_info["relative_path"]
        if not path.exists() or _sha256_file(path) != object_info["sha256"]:
            raise RuntimeError(f"funding import object is missing or corrupt: {path}")
    _validate_complete_coverage(config, desired, _load_existing_rates(config))


def _result_from_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    idempotent_replay: bool,
) -> FundingBackfillResult:
    configured = manifest["configured_range"]
    mutation = manifest["lake_mutation"]
    reconciliation = manifest["source_reconciliation"]
    return FundingBackfillResult(
        import_id=manifest["import_id"],
        manifest_path=manifest_path,
        expected_rows=int(configured["expected_rows"]),
        preexisting_rows=int(mutation["preexisting_rows"]),
        published_rows=int(mutation["published_rows"]),
        published_objects=len(mutation["published_objects"]),
        overlap_rows=int(reconciliation["overlap_rows"]),
        first_timestamp=datetime.fromisoformat(configured["first_timestamp"]),
        last_timestamp=datetime.fromisoformat(configured["last_timestamp"]),
        idempotent_replay=idempotent_replay,
    )
