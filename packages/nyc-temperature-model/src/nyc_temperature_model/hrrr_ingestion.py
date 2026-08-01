from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar
from zoneinfo import ZoneInfo

import numpy as np

from . import STATION_ID, STATION_LATITUDE, STATION_LONGITUDE
from .config import Settings
from .database import connection, insert_artifact
from .jobs import Job, update_progress
from .sources import file_sha256

NYC = ZoneInfo("America/New_York")
HRRR_ARCHIVE_START = datetime(2014, 7, 30, tzinfo=UTC)
T = TypeVar("T")


class HrrrDownloadExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class HrrrField:
    temperature_f: float
    latitude: float
    longitude: float
    local_path: Path
    source_uri: str
    archive_source: str


def _model_run_for_decision(decision_time: datetime, availability_lag_minutes: int) -> datetime:
    available = decision_time.astimezone(UTC) - timedelta(minutes=availability_lag_minutes)
    synoptic_hour = available.hour - available.hour % 6
    return available.replace(hour=synoptic_hour, minute=0, second=0, microsecond=0)


def _decision_times(start: datetime, end: datetime) -> list[datetime]:
    first_date = start.astimezone(NYC).date()
    final_date = (end.astimezone(NYC) - timedelta(microseconds=1)).date()
    output = []
    current = first_date
    while current <= final_date:
        for hour in (0, 12):
            decision = datetime.combine(current, datetime.min.time(), NYC).replace(hour=hour)
            decision_utc = decision.astimezone(UTC)
            if start <= decision_utc < end:
                output.append(decision_utc)
        current += timedelta(days=1)
    return output


def _temperature_variable(dataset):
    preferred = ("t2m", "t", "TMP")
    for name in preferred:
        if name in dataset.data_vars:
            return dataset[name]
    for variable in dataset.data_vars.values():
        attrs = {str(key).lower(): str(value).lower() for key, value in variable.attrs.items()}
        if attrs.get("grib_shortname") in {"2t", "t"} or (
            attrs.get("grib_name") == "temperature"
            and attrs.get("grib_typeoflevel") == "heightaboveground"
        ):
            return variable
    raise ValueError("HRRR subset did not contain two-metre temperature")


def _point_temperature_f(dataset) -> tuple[float, float, float]:
    variable = _temperature_variable(dataset).squeeze(drop=True)
    lat = np.asarray(dataset["latitude"].values).squeeze()
    lon = np.asarray(dataset["longitude"].values).squeeze()
    target_lon = STATION_LONGITUDE % 360 if float(np.nanmax(lon)) > 180 else STATION_LONGITUDE
    distance = (lat - STATION_LATITUDE) ** 2 + (
        (lon - target_lon) * math.cos(math.radians(STATION_LATITUDE))
    ) ** 2
    flat_index = int(np.nanargmin(distance))
    kelvin = float(np.asarray(variable.values).reshape(-1)[flat_index])
    point_lat = float(lat.reshape(-1)[flat_index])
    point_lon = float(lon.reshape(-1)[flat_index])
    if point_lon > 180:
        point_lon -= 360
    return (kelvin - 273.15) * 9 / 5 + 32, point_lat, point_lon


def _source_order(sources: tuple[str, ...], attempt: int) -> tuple[str, ...]:
    offset = (attempt - 1) % len(sources)
    return sources[offset:] + sources[:offset]


def _is_transient_hrrr_error(error: BaseException) -> bool:
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "connection aborted",
            "connection broken",
            "connection reset",
            "httpsconnectionpool",
            "incompleteread",
            "max retries exceeded",
            "remote end closed",
            "ssl",
            "timed out",
            "unexpected_eof",
            "unexpected eof",
        )
    )


def _run_with_retry(
    operation: Callable[[tuple[str, ...], bool], T],
    *,
    attempts: int,
    sources: tuple[str, ...],
    retry_base_ms: int,
    retry_max_ms: int,
    sleep: Callable[[float], Any] = time.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
) -> T:
    last_error: BaseException | None = None
    only_not_found = True
    not_found_attempts = 0
    not_found_limit = min(attempts, len(sources))
    for attempt in range(1, attempts + 1):
        try:
            return operation(_source_order(sources, attempt), attempt > 1)
        except FileNotFoundError as error:
            last_error = error
            not_found_attempts += 1
            if not_found_attempts >= not_found_limit:
                raise
        except Exception as error:
            only_not_found = False
            if not _is_transient_hrrr_error(error):
                raise
            last_error = error
        if attempt < attempts:
            base_delay = min(retry_max_ms, retry_base_ms * 2 ** (attempt - 1)) / 1000
            sleep(base_delay + jitter(0, base_delay * 0.25))
    if only_not_found and isinstance(last_error, FileNotFoundError):
        raise last_error
    raise HrrrDownloadExhausted(
        f"HRRR field exhausted {attempts} attempts across {','.join(sources)}: {last_error}"
    ) from last_error


def _download_field(
    settings: Settings,
    herbie_factory,
    model_run: datetime,
    lead_hours: int,
    search: str,
) -> HrrrField:
    def operation(priority: tuple[str, ...], overwrite: bool) -> HrrrField:
        herbie = herbie_factory(
            model_run.replace(tzinfo=None),
            model="hrrr",
            product="sfc",
            fxx=lead_hours,
            priority=list(priority),
            save_dir=settings.cache_directory / "hrrr",
            overwrite=overwrite,
            verbose=False,
        )
        if getattr(herbie, "grib", None) is None:
            raise FileNotFoundError(
                f"HRRR archive field is unavailable for "
                f"{model_run:%Y-%m-%dT%H:%MZ} f{lead_hours:02d}"
            )
        local_path = Path(herbie.download(search, verbose=False, errors="raise"))
        dataset = herbie.xarray(search, remove_grib=False, verbose=False)
        try:
            temperature_f, point_lat, point_lon = _point_temperature_f(dataset)
        finally:
            dataset.close()
        return HrrrField(
            temperature_f=temperature_f,
            latitude=point_lat,
            longitude=point_lon,
            local_path=local_path,
            source_uri=str(getattr(herbie, "grib", "")),
            archive_source=str(getattr(herbie, "grib_source", priority[0])).lower(),
        )

    field = _run_with_retry(
        operation,
        attempts=settings.hrrr_download_attempts,
        sources=settings.hrrr_source_priority,
        retry_base_ms=settings.hrrr_retry_base_ms,
        retry_max_ms=settings.hrrr_retry_max_ms,
    )
    if settings.hrrr_request_interval_ms:
        time.sleep(settings.hrrr_request_interval_ms / 1000)
    return field


def _valid_times(decision_time: datetime, model_run: datetime) -> list[datetime]:
    local_date = decision_time.astimezone(NYC).date()
    local_end = datetime.combine(
        local_date + timedelta(days=1), datetime.min.time(), NYC
    ).astimezone(UTC)
    first_valid = max(model_run, decision_time)
    valid_at = first_valid.replace(minute=0, second=0, microsecond=0)
    if valid_at < first_valid:
        valid_at += timedelta(hours=1)
    output = []
    while valid_at < local_end:
        output.append(valid_at)
        valid_at += timedelta(hours=1)
    return output


def _existing_valid_times(
    settings: Settings, decision_time: datetime, model_run: datetime
) -> set[datetime]:
    with connection(settings.database_url) as conn:
        return {
            row["valid_at"]
            for row in conn.execute(
                """
                SELECT valid_at
                FROM weather.hrrr_point_forecasts
                WHERE station_id=%s AND decision_time=%s AND model_run=%s
                """,
                (STATION_ID, decision_time, model_run),
            )
        }


def ingest_hrrr(settings: Settings, job: Job) -> dict:
    if job.range_start < HRRR_ARCHIVE_START:
        raise ValueError("HRRR ingestion cannot begin before 2014-07-30")
    availability_lag = int(job.request.get("availability_lag_minutes", 75))
    if not 60 <= availability_lag <= 180:
        raise ValueError("availability_lag_minutes must be between 60 and 180")
    search = str(job.request.get("search", ":TMP:2 m above ground"))
    try:
        from herbie import Herbie
    except ImportError as error:
        raise RuntimeError("herbie-data is required for HRRR ingestion") from error

    decisions = _decision_times(job.range_start, job.range_end)
    downloaded_rows = 0
    reused_rows = 0
    missing = 0
    exhausted_fields: list[str] = []
    for decision_index, decision_time in enumerate(decisions, start=1):
        model_run = _model_run_for_decision(decision_time, availability_lag)
        expected_valid_times = _valid_times(decision_time, model_run)
        existing = _existing_valid_times(settings, decision_time, model_run)
        reused_rows += len(existing.intersection(expected_valid_times))
        decision_rows = len(existing.intersection(expected_valid_times))
        for valid_at in expected_valid_times:
            if valid_at in existing:
                continue
            lead_hours = int((valid_at - model_run).total_seconds() // 3600)
            try:
                field = _download_field(settings, Herbie, model_run, lead_hours, search)
                digest, size = file_sha256(field.local_path)
                logical_key = (
                    f"noaa:hrrr:sfc:{model_run:%Y%m%dT%H}:f{lead_hours:02d}:tmp2m"
                )
                with connection(settings.database_url) as conn, conn.transaction():
                    artifact_id = insert_artifact(
                        conn,
                        provider="noaa_hrrr_open_data",
                        logical_key=logical_key,
                        source_uri=field.source_uri,
                        sha256=digest,
                        compressed_bytes=size,
                        record_count=1,
                        metadata={
                            "model": "hrrr",
                            "product": "sfc",
                            "search": search,
                            "lead_hours": lead_hours,
                            "availability_lag_minutes": availability_lag,
                            "archive_source": field.archive_source,
                        },
                        source_start=valid_at,
                        source_end=valid_at + timedelta(hours=1),
                    )
                    conn.execute(
                        """
                        INSERT INTO weather.hrrr_point_forecasts (
                          station_id,decision_time,model_run,valid_at,lead_hours,
                          temperature_f,latitude,longitude,source_artifact_id
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (station_id,decision_time,model_run,valid_at) DO UPDATE SET
                          temperature_f=EXCLUDED.temperature_f,
                          latitude=EXCLUDED.latitude, longitude=EXCLUDED.longitude,
                          source_artifact_id=EXCLUDED.source_artifact_id
                        """,
                        (
                            STATION_ID,
                            decision_time,
                            model_run,
                            valid_at,
                            lead_hours,
                            field.temperature_f,
                            field.latitude,
                            field.longitude,
                            artifact_id,
                        ),
                    )
                decision_rows += 1
                downloaded_rows += 1
            except FileNotFoundError:
                missing += 1
            except HrrrDownloadExhausted as error:
                exhausted_fields.append(
                    f"{model_run:%Y-%m-%dT%H}:f{lead_hours:02d}:{error}"
                )
        update_progress(
            settings,
            job,
            {
                "decisions_completed": decision_index,
                "decisions_total": len(decisions),
                "downloaded_rows": downloaded_rows,
                "reused_rows": reused_rows,
                "forecast_rows_available": downloaded_rows + reused_rows,
                "missing_fields": missing,
                "exhausted_fields": len(exhausted_fields),
                "last_decision_time": decision_time.isoformat(),
                "last_decision_forecasts": decision_rows,
            },
        )
    if exhausted_fields:
        raise HrrrDownloadExhausted(
            f"{len(exhausted_fields)} fields remain after decision-level retries; "
            f"first={exhausted_fields[0]}"
        )
    return {
        "decisions": len(decisions),
        "downloaded_rows": downloaded_rows,
        "reused_rows": reused_rows,
        "forecast_rows_available": downloaded_rows + reused_rows,
        "missing_fields": missing,
        "availability_lag_minutes": availability_lag,
    }
