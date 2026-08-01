from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

import psycopg

from .config import Settings
from .database import connection, insert_artifact
from .jobs import Job, update_progress
from .sources import atomic_write, source_client


def _persist_observations(
    settings: Settings,
    job: Job,
    *,
    response,
    provider: str,
    logical_key: str,
    cache_name: str,
    valid_column: str,
    metadata: dict,
) -> dict:
    start = job.range_start.astimezone(UTC)
    end = job.range_end.astimezone(UTC)
    path = settings.cache_directory / provider / cache_name
    digest, size = atomic_write(path, response.content)
    reader = csv.DictReader(io.StringIO(response.content.decode("utf-8-sig")))
    rows = []
    missing = 0
    for row in reader:
        valid = (row.get(valid_column) or "").strip()
        if not valid:
            continue
        observed_at = datetime.strptime(valid, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        if observed_at < start or observed_at >= end:
            continue
        raw_temperature = (row.get("tmpf") or "").strip()
        try:
            temperature = Decimal(raw_temperature) if raw_temperature else None
        except InvalidOperation:
            temperature = None
        if temperature is None:
            missing += 1
        try:
            report_type = int(row["report_type"]) if row.get("report_type") else None
        except ValueError:
            report_type = None
        rows.append(("KLGA", observed_at, temperature, report_type))
    with connection(settings.database_url) as conn, conn.transaction():
        artifact_id = insert_artifact(
            conn,
            provider=provider,
            logical_key=logical_key,
            source_uri=str(response.request.url),
            sha256=digest,
            compressed_bytes=size,
            record_count=len(rows),
            metadata={**metadata, "missing_temperature_rows": missing},
            source_start=start,
            source_end=end,
        )
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO weather.station_observations (
                  station_id, observed_at, temperature_f, report_type, provider,
                  source_artifact_id, quality_flags
                ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (station_id, observed_at, provider) DO UPDATE SET
                  temperature_f=EXCLUDED.temperature_f,
                  report_type=EXCLUDED.report_type,
                  source_artifact_id=EXCLUDED.source_artifact_id,
                  quality_flags=EXCLUDED.quality_flags
                """,
                [
                    (
                        station,
                        observed,
                        temperature,
                        report_type,
                        provider,
                        artifact_id,
                        psycopg.types.json.Jsonb(
                            [] if temperature is not None else ["missing_temperature"]
                        ),
                    )
                    for station, observed, temperature, report_type in rows
                ],
            )
    progress = {"observations": len(rows), "missing_temperature": missing, "provider": provider}
    update_progress(settings, job, progress)
    return progress


def ingest_asos_resolution(settings: Settings, job: Job) -> dict:
    start = job.range_start.astimezone(UTC)
    end = job.range_end.astimezone(UTC)
    params = [
        ("station", "LGA"),
        ("data", "tmpf"),
        ("year1", start.year),
        ("month1", start.month),
        ("day1", start.day),
        ("year2", end.year),
        ("month2", end.month),
        ("day2", end.day),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("elev", "no"),
        ("missing", "empty"),
        ("trace", "empty"),
        ("direct", "no"),
        ("report_type", "3"),
        ("report_type", "4"),
    ]
    with source_client() as client:
        response = client.get(settings.iem_asos_metar_url, params=params)
        response.raise_for_status()
    return _persist_observations(
        settings,
        job,
        response=response,
        provider="iem_asos_metar",
        logical_key=f"iem:asos-metars:LGA:{start.date()}:{end.date()}",
        cache_name=f"LGA-{start.date()}-{end.date()}.csv",
        valid_column="valid",
        metadata={
            "station": "LGA",
            "source": "IEM ASOS/METAR archive",
            "report_types": [3, 4],
            "canonical_resolution_proxy": True,
        },
    )


def ingest_asos_one_minute(settings: Settings, job: Job) -> dict:
    start = job.range_start.astimezone(UTC)
    end = job.range_end.astimezone(UTC)
    params = [
        ("station", "LGA"),
        ("vars", "tmpf"),
        ("sts", start.strftime("%Y-%m-%dT%H:%MZ")),
        ("ets", end.strftime("%Y-%m-%dT%H:%MZ")),
        ("sample", "1min"),
        ("what", "download"),
        ("tz", "UTC"),
        ("delim", "comma"),
        ("gis", "no"),
    ]
    with source_client() as client:
        response = client.get(settings.iem_asos_one_minute_url, params=params)
        response.raise_for_status()
    return _persist_observations(
        settings,
        job,
        response=response,
        provider="iem_ncei_asos_one_minute",
        logical_key=f"iem:ncei-asos-one-minute:LGA:{start.date()}:{end.date()}",
        cache_name=f"LGA-{start.date()}-{end.date()}.csv",
        valid_column="valid(UTC)",
        metadata={
            "station": "LGA",
            "source": "NCEI ASOS one-minute archive processed by IEM",
            "sample": "1min",
            "canonical_resolution_proxy": False,
        },
    )


def ingest_asos(settings: Settings, job: Job) -> dict:
    """Backward-compatible handler for already queued one-minute jobs."""
    return ingest_asos_one_minute(settings, job)
