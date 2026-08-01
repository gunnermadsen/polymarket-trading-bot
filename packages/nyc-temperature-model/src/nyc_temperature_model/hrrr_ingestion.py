from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from . import STATION_ID, STATION_LATITUDE, STATION_LONGITUDE
from .config import Settings
from .database import connection, insert_artifact
from .jobs import Job, update_progress
from .sources import file_sha256

NYC = ZoneInfo("America/New_York")
HRRR_ARCHIVE_START = datetime(2014, 7, 30, tzinfo=UTC)


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
    forecast_rows = 0
    missing = 0
    for decision_index, decision_time in enumerate(decisions, start=1):
        model_run = _model_run_for_decision(decision_time, availability_lag)
        local_date = decision_time.astimezone(NYC).date()
        local_end = datetime.combine(
            local_date + timedelta(days=1), datetime.min.time(), NYC
        ).astimezone(UTC)
        first_valid = max(model_run, decision_time)
        valid_at = first_valid.replace(minute=0, second=0, microsecond=0)
        if valid_at < first_valid:
            valid_at += timedelta(hours=1)
        rows = []
        while valid_at < local_end:
            lead_hours = int((valid_at - model_run).total_seconds() // 3600)
            try:
                herbie = Herbie(
                    model_run.replace(tzinfo=None),
                    model="hrrr",
                    product="sfc",
                    fxx=lead_hours,
                    save_dir=settings.cache_directory / "hrrr",
                    overwrite=False,
                    verbose=False,
                )
                local_path = Path(herbie.download(search, verbose=False))
                dataset = herbie.xarray(search, remove_grib=False, verbose=False)
                temperature_f, point_lat, point_lon = _point_temperature_f(dataset)
                dataset.close()
                digest, size = file_sha256(local_path)
                logical_key = (
                    f"noaa:hrrr:sfc:{model_run:%Y%m%dT%H}:f{lead_hours:02d}:tmp2m"
                )
                with connection(settings.database_url) as conn, conn.transaction():
                    artifact_id = insert_artifact(
                        conn,
                        provider="noaa_hrrr_aws",
                        logical_key=logical_key,
                        source_uri=str(getattr(herbie, "grib", "")),
                        sha256=digest,
                        compressed_bytes=size,
                        record_count=1,
                        metadata={
                            "model": "hrrr",
                            "product": "sfc",
                            "search": search,
                            "lead_hours": lead_hours,
                            "availability_lag_minutes": availability_lag,
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
                            temperature_f,
                            point_lat,
                            point_lon,
                            artifact_id,
                        ),
                    )
                rows.append(valid_at)
                forecast_rows += 1
            except FileNotFoundError:
                missing += 1
            valid_at += timedelta(hours=1)
        update_progress(
            settings,
            job,
            {
                "decisions_completed": decision_index,
                "decisions_total": len(decisions),
                "forecast_rows": forecast_rows,
                "missing_fields": missing,
                "last_decision_time": decision_time.isoformat(),
                "last_decision_forecasts": len(rows),
            },
        )
    return {
        "decisions": len(decisions),
        "forecast_rows": forecast_rows,
        "missing_fields": missing,
        "availability_lag_minutes": availability_lag,
    }
