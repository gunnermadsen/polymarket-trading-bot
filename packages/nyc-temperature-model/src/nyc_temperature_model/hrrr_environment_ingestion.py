from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import xarray as xr

from . import PROCESS_ID, STATION_ID, STATION_LATITUDE, STATION_LONGITUDE
from .config import Settings
from .database import connection, insert_unified_backfill_artifact
from .hrrr_ingestion import (
    HRRR_ARCHIVE_START,
    HrrrDownloadExhausted,
    _decision_times,
    _model_run_for_decision,
    _run_with_retry,
    _valid_times,
)
from .backfill_contract import Job, update_progress
from .sources import file_sha256
from .spatial_features import RADII_KM, SECTORS, numeric_summary, spatial_mask

FEATURE_SCHEMA_VERSION = "hrrr-environment-klga-v2"
AVAILABILITY_LAG_MINUTES = 75
MINIMUM_VALID_PIXEL_FRACTION = 0.5
FIELD_SEARCH = (
    ":(?:TMP:2 m above ground|DPT:2 m above ground|UGRD:10 m above ground|"
    "VGRD:10 m above ground|TCDC:entire atmosphere|DSWRF:surface|HPBL:surface|"
    "APCP:surface|REFC:entire atmosphere)"
)

FIELD_ALIASES = {
    "temperature_2m": ("t2m", "2t", "TMP"),
    "dew_point_2m": ("d2m", "2d", "DPT"),
    "wind_u_10m": ("u10", "10u", "UGRD"),
    "wind_v_10m": ("v10", "10v", "VGRD"),
    "total_cloud_cover": ("tcc", "TCDC"),
    "downward_shortwave_radiation": ("sdswrf", "dswrf", "DSWRF"),
    "boundary_layer_height": ("blh", "HPBL"),
    "accumulated_precipitation": ("tp", "APCP"),
    "composite_reflectivity": ("refc", "REFC"),
}


@dataclass(frozen=True)
class EnvironmentSubset:
    local_path: Path
    source_uri: str
    archive_source: str


def _download_subset(
    settings: Settings, herbie_factory, model_run: datetime, lead_hours: int
) -> EnvironmentSubset:
    def operation(priority: tuple[str, ...], overwrite: bool) -> EnvironmentSubset:
        herbie = herbie_factory(
            model_run.replace(tzinfo=None),
            model="hrrr",
            product="sfc",
            fxx=lead_hours,
            priority=list(priority),
            save_dir=settings.cache_directory / "hrrr-environment",
            overwrite=overwrite,
            verbose=False,
        )
        if getattr(herbie, "grib", None) is None:
            raise FileNotFoundError(
                f"HRRR environment source unavailable for {model_run:%Y-%m-%dT%H:%MZ} f{lead_hours:02d}"
            )
        local_path = Path(herbie.download(FIELD_SEARCH, verbose=False, errors="raise"))
        if not local_path.is_file():
            raise FileNotFoundError(
                f"HRRR environment subset unavailable for {model_run:%Y-%m-%dT%H:%MZ} f{lead_hours:02d}"
            )
        return EnvironmentSubset(
            local_path=local_path,
            source_uri=str(herbie.grib),
            archive_source=str(getattr(herbie, "grib_source", priority[0])).lower(),
        )

    return _run_with_retry(
        operation,
        attempts=settings.hrrr_download_attempts,
        sources=settings.hrrr_source_priority,
        retry_base_ms=settings.hrrr_retry_base_ms,
        retry_max_ms=settings.hrrr_retry_max_ms,
    )


def _field_name(variable: xr.DataArray) -> str | None:
    candidates = {
        str(variable.name or "").lower(),
        str(variable.attrs.get("GRIB_shortName", "")).lower(),
        str(variable.attrs.get("GRIB_name", "")).lower(),
    }
    for canonical, aliases in FIELD_ALIASES.items():
        if candidates.intersection(alias.lower() for alias in aliases):
            return canonical
    return None


def _load_grib_fields(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
    try:
        import cfgrib
    except ImportError as error:
        raise RuntimeError("cfgrib is required for HRRR environmental ingestion") from error
    datasets = cfgrib.open_datasets(path)
    fields: dict[str, np.ndarray] = {}
    latitude = longitude = None
    units: dict[str, str | None] = {}
    try:
        for dataset in datasets:
            if latitude is None and "latitude" in dataset and "longitude" in dataset:
                latitude = np.asarray(dataset["latitude"].values, dtype=float).squeeze()
                longitude = np.asarray(dataset["longitude"].values, dtype=float).squeeze()
                longitude = np.where(longitude > 180, longitude - 360, longitude)
            for variable in dataset.data_vars.values():
                canonical = _field_name(variable)
                if canonical is None:
                    continue
                values = np.asarray(variable.squeeze(drop=True).values, dtype=float)
                if values.ndim == 2:
                    fields[canonical] = values
                    units[canonical] = variable.attrs.get("units")
    finally:
        for dataset in datasets:
            dataset.close()
    if latitude is None or longitude is None:
        raise ValueError("HRRR subset lacks latitude/longitude")
    if "total_cloud_cover" in fields and np.nanmax(fields["total_cloud_cover"]) > 1.5:
        fields["total_cloud_cover"] = fields["total_cloud_cover"] / 100.0
    return fields, latitude, longitude, units


def _extract_environment_patch(
    source_path: Path, patch_path: Path
) -> tuple[dict[tuple[int, str], dict[str, dict[str, float | None]]], list[str], dict]:
    fields, latitude, longitude, units = _load_grib_fields(source_path)
    outer = spatial_mask(latitude, longitude, 100, "all")
    indexes = np.argwhere(outer)
    if not indexes.size:
        raise ValueError("KLGA is outside the HRRR grid")
    y0, x0 = indexes.min(axis=0)
    y1, x1 = indexes.max(axis=0) + 1
    patch_latitude = latitude[y0:y1, x0:x1]
    patch_longitude = longitude[y0:y1, x0:x1]
    patch_fields = {name: values[y0:y1, x0:x1] for name, values in fields.items()}
    patch = xr.Dataset(
        data_vars={
            **{name: (("y", "x"), values) for name, values in patch_fields.items()},
            "latitude": (("y", "x"), patch_latitude),
            "longitude": (("y", "x"), patch_longitude),
        },
        attrs={"station_id": STATION_ID, "feature_schema_version": FEATURE_SCHEMA_VERSION},
    )
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    partial = patch_path.with_name(f".{patch_path.name}.partial")
    patch.to_netcdf(
        partial,
        engine="h5netcdf",
        encoding={name: {"compression": "gzip", "compression_opts": 4} for name in patch.data_vars},
    )
    os.replace(partial, patch_path)

    summaries: dict[tuple[int, str], dict[str, dict[str, float | None]]] = {}
    nearest = int(
        np.nanargmin(
            (patch_latitude - STATION_LATITUDE) ** 2
            + ((patch_longitude - STATION_LONGITUDE) * np.cos(np.deg2rad(STATION_LATITUDE))) ** 2
        )
    )
    point_mask = np.zeros(patch_latitude.shape, dtype=bool)
    point_mask.reshape(-1)[nearest] = True
    summaries[(0, "all")] = {
        name: numeric_summary(values, point_mask) for name, values in patch_fields.items()
    }
    for radius in RADII_KM:
        for sector in SECTORS:
            mask = spatial_mask(patch_latitude, patch_longitude, radius, sector)
            summaries[(radius, sector)] = {
                name: numeric_summary(values, mask) for name, values in patch_fields.items()
            }
    return summaries, sorted(fields), units


def _coverage_is_terminal(
    settings: Settings,
    decision_time: datetime,
    valid_at: datetime,
    retry_statuses: set[str],
) -> bool:
    with connection(settings.database_url) as conn:
        row = conn.execute(
            """
            SELECT status
            FROM weather.hrrr_environment_window_coverage
            WHERE process_id=%s AND station_id=%s AND decision_time=%s AND valid_at=%s
              AND feature_schema_version=%s
            """,
            (PROCESS_ID, STATION_ID, decision_time, valid_at, FEATURE_SCHEMA_VERSION),
        ).fetchone()
    return row is not None and row["status"] not in retry_statuses


def _insert_coverage(conn, values: tuple) -> None:
    conn.execute(
        """
        INSERT INTO weather.hrrr_environment_window_coverage (
          process_id,station_id,decision_time,model_run,valid_at,feature_schema_version,
          status,fields_present,fields_missing,source_artifact_id,cropped_artifact_path,
          cropped_artifact_sha256,valid_pixel_fraction,quality_flags,source_metadata
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (process_id,station_id,decision_time,valid_at,feature_schema_version)
        DO UPDATE SET status=EXCLUDED.status, fields_present=EXCLUDED.fields_present,
          fields_missing=EXCLUDED.fields_missing, source_artifact_id=EXCLUDED.source_artifact_id,
          cropped_artifact_path=EXCLUDED.cropped_artifact_path,
          cropped_artifact_sha256=EXCLUDED.cropped_artifact_sha256,
          valid_pixel_fraction=EXCLUDED.valid_pixel_fraction,
          quality_flags=EXCLUDED.quality_flags,source_metadata=EXCLUDED.source_metadata,
          checked_at=now()
        WHERE weather.hrrr_environment_window_coverage.status IN (
          'missing_source','download_failure','processing_failure'
        )
        """,
        values,
    )


def _feature_values(
    decision_time: datetime,
    model_run: datetime,
    valid_at: datetime,
    radius: int,
    sector: str,
    summaries: dict[str, dict[str, float | None]],
    gradients: dict[str, float | None],
    metadata: dict,
) -> tuple:
    field = lambda name: summaries.get(name, {})
    u = field("wind_u_10m").get("mean")
    v = field("wind_v_10m").get("mean")
    wind_speed = float(np.hypot(u, v)) if u is not None and v is not None else None
    fractions = [value.get("valid_pixel_fraction", 0.0) for value in summaries.values()]
    valid_fraction = min(fractions) if fractions else 0.0
    return (
        PROCESS_ID, STATION_ID, decision_time, model_run, valid_at,
        int((valid_at - model_run).total_seconds() // 3600), radius, sector,
        FEATURE_SCHEMA_VERSION,
        field("temperature_2m").get("mean"), field("temperature_2m").get("stddev"),
        field("dew_point_2m").get("mean"), field("dew_point_2m").get("stddev"),
        field("total_cloud_cover").get("mean"), field("total_cloud_cover").get("stddev"),
        field("downward_shortwave_radiation").get("mean"), u, v, wind_speed,
        field("boundary_layer_height").get("mean"),
        field("accumulated_precipitation").get("mean"),
        field("composite_reflectivity").get("mean"),
        field("composite_reflectivity").get("max"), valid_fraction,
        gradients.get("temperature_north_south"), gradients.get("temperature_east_west"),
        psycopg.types.json.Jsonb({}), psycopg.types.json.Jsonb(metadata),
    )


def _insert_feature(conn, values: tuple) -> None:
    conn.execute(
        """
        INSERT INTO weather.hrrr_environment_features (
          process_id,station_id,decision_time,model_run,valid_at,lead_hours,
          spatial_radius_km,sector,feature_schema_version,temperature_2m_mean_k,
          temperature_2m_stddev_k,dew_point_2m_mean_k,dew_point_2m_stddev_k,
          total_cloud_cover_mean_fraction,total_cloud_cover_stddev_fraction,
          downward_shortwave_radiation_mean_w_m2,wind_u_10m_mean_m_s,
          wind_v_10m_mean_m_s,wind_speed_10m_mean_m_s,boundary_layer_height_mean_m,
          accumulated_precipitation_mean_mm,composite_reflectivity_mean_dbz,
          composite_reflectivity_max_dbz,valid_pixel_fraction,
          temperature_2m_north_south_gradient_k,temperature_2m_east_west_gradient_k,
          quality_flags,source_metadata
        ) VALUES (
          %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
          %s,%s,%s,%s,%s,%s,%s,%s
        ) ON CONFLICT DO NOTHING
        """,
        values,
    )


def ingest_hrrr_environment(settings: Settings, job: Job) -> dict[str, Any]:
    if job.range_start < HRRR_ARCHIVE_START:
        raise ValueError("HRRR environmental ingestion cannot begin before 2014-07-30")
    version = str(job.request.get("feature_schema_version", FEATURE_SCHEMA_VERSION))
    lag = int(job.request.get("availability_lag_minutes", AVAILABILITY_LAG_MINUTES))
    if version != FEATURE_SCHEMA_VERSION or lag != AVAILABILITY_LAG_MINUTES:
        raise ValueError("HRRR environment v2 requires its frozen schema and 75-minute allowance")
    try:
        from herbie import Herbie
    except ImportError as error:
        raise RuntimeError("herbie-data is required for HRRR environmental ingestion") from error
    expected_fields = set(FIELD_ALIASES)
    retry_statuses = set(
        job.request.get("retry_statuses", ("download_failure", "processing_failure"))
    )
    decisions = _decision_times(job.range_start, job.range_end)
    counters = {"completed": 0, "reused": 0, "missing": 0, "failed": 0}
    exhausted: list[str] = []
    for decision_index, decision_time in enumerate(decisions, start=1):
        model_run = _model_run_for_decision(decision_time, lag)
        for valid_at in _valid_times(decision_time, model_run):
            if _coverage_is_terminal(settings, decision_time, valid_at, retry_statuses):
                counters["reused"] += 1
                continue
            lead_hours = int((valid_at - model_run).total_seconds() // 3600)
            source_path: Path | None = None
            try:
                subset = _download_subset(settings, Herbie, model_run, lead_hours)
                source_path = subset.local_path
                source_sha, source_size = file_sha256(source_path)
                patch_path = (
                    settings.hrrr_environment_directory
                    / f"{valid_at:%Y/%m/%d}"
                    / f"hrrr-{model_run:%Y%m%dT%H%MZ}-f{lead_hours:02d}.nc"
                )
                summaries, fields_present, units = _extract_environment_patch(source_path, patch_path)
                patch_sha, patch_size = file_sha256(patch_path)
                fields_missing = sorted(expected_fields - set(fields_present))
                valid_fraction = min(
                    (
                        summary.get("valid_pixel_fraction", 0.0)
                        for summary in summaries[(100, "all")].values()
                    ),
                    default=0.0,
                )
                clear_dry = all(
                    summaries[(25, "all")].get(name, {}).get("mean") == 0
                    for name in (
                        "total_cloud_cover",
                        "accumulated_precipitation",
                        "composite_reflectivity",
                    )
                    if name in summaries[(25, "all")]
                )
                status = (
                    "insufficient_valid_pixels"
                    if valid_fraction < MINIMUM_VALID_PIXEL_FRACTION
                    else "processing_failure"
                    if fields_missing
                    else "valid_zero"
                    if clear_dry
                    else "complete"
                )
                with connection(settings.database_url) as conn, conn.transaction():
                    artifact_id = insert_unified_backfill_artifact(
                        conn,
                        job_id=job.job_id,
                        strategy_key=job.ingester_key,
                        provider="noaa_hrrr_open_data",
                        logical_key=(
                            f"hrrr-environment:{model_run:%Y%m%dT%H}:f{lead_hours:02d}:"
                            f"{FEATURE_SCHEMA_VERSION}"
                        ),
                        source_uri=subset.source_uri,
                        sha256=source_sha,
                        compressed_bytes=source_size,
                        record_count=len(fields_present),
                        metadata={
                            "archive_source": subset.archive_source,
                            "search": FIELD_SEARCH,
                            "fields": fields_present,
                            "units": units,
                            "cropped_artifact_path": str(patch_path),
                            "cropped_artifact_sha256": patch_sha,
                            "cropped_bytes": patch_size,
                        },
                        source_start=valid_at,
                        source_end=valid_at + timedelta(hours=1),
                    )
                    metadata = {"source_artifact_id": artifact_id, "units": units}
                    _insert_coverage(
                        conn,
                        (
                            PROCESS_ID, STATION_ID, decision_time, model_run, valid_at,
                            FEATURE_SCHEMA_VERSION, status, fields_present, fields_missing,
                            artifact_id, str(patch_path), patch_sha, valid_fraction,
                            psycopg.types.json.Jsonb({}), psycopg.types.json.Jsonb(metadata),
                        ),
                    )
                    for (radius, sector), field_summaries in summaries.items():
                        if radius:
                            north = summaries[(radius, "north")].get("temperature_2m", {}).get("mean")
                            south = summaries[(radius, "south")].get("temperature_2m", {}).get("mean")
                            east = summaries[(radius, "east")].get("temperature_2m", {}).get("mean")
                            west = summaries[(radius, "west")].get("temperature_2m", {}).get("mean")
                        else:
                            north = south = east = west = None
                        gradients = {
                            "temperature_north_south": (
                                north - south if north is not None and south is not None else None
                            ),
                            "temperature_east_west": (
                                east - west if east is not None and west is not None else None
                            ),
                        }
                        _insert_feature(
                            conn,
                            _feature_values(
                                decision_time, model_run, valid_at, radius, sector,
                                field_summaries, gradients, metadata,
                            ),
                        )
                counters["completed"] += status in ("complete", "valid_zero")
                counters["failed"] += status not in ("complete", "valid_zero")
                source_path.unlink(missing_ok=True)
            except FileNotFoundError:
                counters["missing"] += 1
                with connection(settings.database_url) as conn, conn.transaction():
                    _insert_coverage(
                        conn,
                        (
                            PROCESS_ID, STATION_ID, decision_time, model_run, valid_at,
                            FEATURE_SCHEMA_VERSION, "missing_source", [], sorted(expected_fields),
                            None, None, None, None, psycopg.types.json.Jsonb({}),
                            psycopg.types.json.Jsonb({}),
                        ),
                    )
            except HrrrDownloadExhausted as error:
                exhausted.append(f"{model_run:%Y%m%dT%H}:f{lead_hours:02d}:{error}")
            except Exception as error:  # noqa: BLE001 - persist field-level failure evidence.
                counters["failed"] += 1
                with connection(settings.database_url) as conn, conn.transaction():
                    _insert_coverage(
                        conn,
                        (
                            PROCESS_ID, STATION_ID, decision_time, model_run, valid_at,
                            FEATURE_SCHEMA_VERSION, "processing_failure", [], sorted(expected_fields),
                            None, None, None, None,
                            psycopg.types.json.Jsonb({"error": str(error)[:1000]}),
                            psycopg.types.json.Jsonb({}),
                        ),
                    )
            finally:
                # A transaction failure intentionally leaves the source for resumable recovery.
                pass
        update_progress(
            settings,
            job,
            {
                **counters,
                "decisions_completed": decision_index,
                "decisions_total": len(decisions),
                "exhausted_fields": len(exhausted),
                "last_decision_time": decision_time.isoformat(),
            },
        )
    if exhausted:
        raise HrrrDownloadExhausted(
            f"{len(exhausted)} HRRR environment subsets exhausted retries; first={exhausted[0]}"
        )
    return {**counters, "decisions": len(decisions), "feature_schema_version": FEATURE_SCHEMA_VERSION}
