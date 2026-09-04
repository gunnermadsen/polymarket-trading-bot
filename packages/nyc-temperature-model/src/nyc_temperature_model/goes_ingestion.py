from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import xarray as xr

from . import PROCESS_ID, STATION_ID
from .config import Settings
from .database import connection, insert_unified_backfill_artifact
from .hrrr_ingestion import NYC, _decision_times
from .backfill_contract import Job, update_progress
from .noaa_transport import download_resumable, list_s3_keys
from .sources import file_sha256
from .spatial_features import (
    RADII_KM,
    SECTORS,
    first_data_variable,
    numeric_summary,
    quality_flag_summary,
    spatial_mask,
)

FEATURE_SCHEMA_VERSION = "goes-abi-klga-v2"
GOES_TRANSITION = datetime(2025, 4, 7, 15, 10, tzinfo=UTC)
SCAN_OFFSETS_MINUTES = (15, 60, 180)
MINIMUM_VALID_PIXEL_FRACTION = 0.5
SCAN_PATTERN = re.compile(r"_s(?P<start>\d{13,14})_e(?P<end>\d{13,14})_c")


@dataclass(frozen=True)
class Product:
    key: str
    archive_name: str
    channel: int | None
    variables: tuple[str, ...]
    required: bool
    noon_only: bool = False


PRODUCTS = (
    Product("infrared_c13", "ABI-L2-CMIPC", 13, ("CMI", "Rad"), True),
    Product("clear_sky_mask", "ABI-L2-ACMC", None, ("BCM", "ACM"), True),
    Product("cloud_top_temperature", "ABI-L2-ACHTF", None, ("TEMP",), True),
    Product("cloud_top_height", "ABI-L2-ACHAC", None, ("HT",), True),
    Product("visible_c02", "ABI-L2-CMIPC", 2, ("CMI", "Rad"), True, noon_only=True),
    Product("cloud_optical_depth", "ABI-L2-CODC", None, ("COD",), False),
    Product("water_vapor_c08", "ABI-L2-CMIPC", 8, ("CMI", "Rad"), False),
)


def operational_satellite(instant: datetime) -> str:
    return "G16" if instant.astimezone(UTC) < GOES_TRANSITION else "G19"


def _parse_scan_times(key: str) -> tuple[datetime, datetime]:
    match = SCAN_PATTERN.search(key)
    if not match:
        raise ValueError(f"GOES key lacks scan timestamps: {key}")

    def parse(value: str) -> datetime:
        core = value[:13]
        return datetime.strptime(core, "%Y%j%H%M%S").replace(tzinfo=UTC)

    return parse(match.group("start")), parse(match.group("end"))


def _archive_keys(satellite: str, product: Product, target: datetime) -> list[str]:
    bucket = f"noaa-goes{satellite[1:]}"
    keys: list[str] = []
    for hour in (target - timedelta(hours=1), target, target + timedelta(hours=1)):
        prefix = f"{product.archive_name}/{hour:%Y}/{hour:%j}/{hour:%H}/"
        keys.extend(list_s3_keys(bucket, prefix))
    if product.channel is not None:
        marker = f"C{product.channel:02d}_G{satellite[1:]}_"
        keys = [key for key in keys if marker in key]
    return keys


def select_causal_scan(
    keys: list[str], *, target: datetime, decision_time: datetime
) -> tuple[str, datetime, datetime] | None:
    cutoff = decision_time - timedelta(minutes=15)
    candidates = []
    for key in keys:
        try:
            scan_start, scan_end = _parse_scan_times(key)
        except ValueError:
            continue
        if scan_end <= cutoff and abs(scan_end - target) <= timedelta(minutes=30):
            candidates.append((abs(scan_end - target), key, scan_start, scan_end))
    if not candidates:
        return None
    _, key, scan_start, scan_end = min(candidates, key=lambda item: (item[0], -item[3].timestamp()))
    return key, scan_start, scan_end


def _goes_lat_lon(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    projection = dataset.get("goes_imager_projection")
    if projection is None:
        raise ValueError("GOES dataset is missing goes_imager_projection")
    attrs = projection.attrs
    required = (
        "perspective_point_height",
        "longitude_of_projection_origin",
        "semi_major_axis",
        "semi_minor_axis",
    )
    if any(name not in attrs for name in required):
        raise ValueError("GOES projection metadata is incomplete")
    x, y = np.meshgrid(np.asarray(dataset["x"]), np.asarray(dataset["y"]))
    height = float(attrs["perspective_point_height"])
    longitude_origin = np.deg2rad(float(attrs["longitude_of_projection_origin"]))
    semi_major = float(attrs["semi_major_axis"])
    semi_minor = float(attrs["semi_minor_axis"])
    satellite_height = height + semi_major
    sin_x, cos_x = np.sin(x), np.cos(x)
    sin_y, cos_y = np.sin(y), np.cos(y)
    a = sin_x**2 + cos_x**2 * (cos_y**2 + (semi_major**2 / semi_minor**2) * sin_y**2)
    b = -2 * satellite_height * cos_x * cos_y
    c = satellite_height**2 - semi_major**2
    discriminant = b**2 - 4 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        distance = (-b - np.sqrt(discriminant)) / (2 * a)
        sx = distance * cos_x * cos_y
        sy = -distance * sin_x
        sz = distance * cos_x * sin_y
        latitude = np.rad2deg(
            np.arctan((semi_major**2 / semi_minor**2) * sz / np.hypot(satellite_height - sx, sy))
        )
        longitude = np.rad2deg(longitude_origin - np.arctan2(sy, satellite_height - sx))
    latitude[discriminant < 0] = np.nan
    longitude[discriminant < 0] = np.nan
    return latitude, longitude


def _extract_patch(
    source_path: Path, product: Product, patch_path: Path
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    with xr.open_dataset(source_path, engine="h5netcdf") as dataset:
        if "goes_imager_projection" not in dataset:
            raise ValueError("source is not an ABI fixed-grid product")
        variable = first_data_variable(dataset, product.variables).squeeze(drop=True)
        if variable.ndim != 2:
            raise ValueError(f"{product.key} variable is not two-dimensional")
        latitude, longitude = _goes_lat_lon(dataset)
        outer = spatial_mask(latitude, longitude, 100, "all")
        indexes = np.argwhere(outer)
        if not indexes.size:
            raise ValueError("KLGA is outside the GOES CONUS grid")
        y0, x0 = indexes.min(axis=0)
        y1, x1 = indexes.max(axis=0) + 1
        values = np.asarray(variable.values, dtype=float)[y0:y1, x0:x1]
        patch_latitude = latitude[y0:y1, x0:x1]
        patch_longitude = longitude[y0:y1, x0:x1]
        dqf_values = None
        if "DQF" in dataset:
            dqf_values = np.asarray(dataset["DQF"].squeeze(drop=True).values)[y0:y1, x0:x1]
        patch = xr.Dataset(
            data_vars={
                product.key: (("y", "x"), values),
                "latitude": (("y", "x"), patch_latitude),
                "longitude": (("y", "x"), patch_longitude),
            },
            attrs={
                "source_product": product.archive_name,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "station_id": STATION_ID,
            },
        )
        if dqf_values is not None:
            patch["DQF"] = (("y", "x"), dqf_values)
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        partial = patch_path.with_name(f".{patch_path.name}.partial")
        patch.to_netcdf(
            partial,
            engine="h5netcdf",
            encoding={name: {"compression": "gzip", "compression_opts": 4} for name in patch.data_vars},
        )
        os.replace(partial, patch_path)

    summaries: dict[tuple[int, str], dict[str, Any]] = {}
    for radius in RADII_KM:
        for sector in SECTORS:
            mask = spatial_mask(patch_latitude, patch_longitude, radius, sector)
            summary = numeric_summary(values, mask)
            summary["quality"] = quality_flag_summary(dqf_values, mask) if dqf_values is not None else {}
            if product.key == "clear_sky_mask":
                selected = values[mask]
                selected = selected[np.isfinite(selected)]
                summary["clear_fraction"] = float(np.mean(selected >= 2)) if selected.size else None
                summary["cloudy_fraction"] = float(np.mean(selected < 2)) if selected.size else None
            summaries[(radius, sector)] = summary
    return summaries, {"variable": variable.name, "units": variable.attrs.get("units")}


def _existing_coverage(settings: Settings, decision_time: datetime) -> dict[tuple[int, str], str]:
    with connection(settings.database_url) as conn:
        rows = conn.execute(
            """
            SELECT requested_offset_minutes, product, status
            FROM weather.goes_abi_window_coverage
            WHERE process_id=%s AND station_id=%s AND decision_time=%s
              AND feature_schema_version=%s
            """,
            (PROCESS_ID, STATION_ID, decision_time, FEATURE_SCHEMA_VERSION),
        ).fetchall()
    return {
        (row["requested_offset_minutes"], row["product"]): row["status"] for row in rows
    }


def _existing_infrared_summaries(
    settings: Settings, decision_time: datetime
) -> dict[int, dict[tuple[int, str], dict[str, dict[str, float | None]]]]:
    output: dict[int, dict[tuple[int, str], dict[str, dict[str, float | None]]]] = {}
    with connection(settings.database_url) as conn:
        rows = conn.execute(
            """
            SELECT requested_offset_minutes,spatial_radius_km,sector,
                   infrared_brightness_temperature_mean_k
            FROM weather.goes_abi_features
            WHERE process_id=%s AND station_id=%s AND decision_time=%s
              AND feature_schema_version=%s
            """,
            (PROCESS_ID, STATION_ID, decision_time, FEATURE_SCHEMA_VERSION),
        ).fetchall()
    for row in rows:
        output.setdefault(row["requested_offset_minutes"], {})[
            (row["spatial_radius_km"], row["sector"])
        ] = {"infrared_c13": {"mean": row["infrared_brightness_temperature_mean_k"]}}
    return output


def _coverage_values(
    *,
    decision_time: datetime,
    offset: int,
    product: Product,
    satellite: str,
    status: str,
    scan_start: datetime | None = None,
    scan_end: datetime | None = None,
    valid_fraction: float | None = None,
    artifact_id: str | None = None,
    patch_path: Path | None = None,
    patch_sha: str | None = None,
    quality: dict | None = None,
    metadata: dict | None = None,
) -> tuple:
    return (
        PROCESS_ID,
        STATION_ID,
        decision_time,
        offset,
        product.key,
        FEATURE_SCHEMA_VERSION,
        status,
        satellite,
        scan_start,
        scan_end,
        valid_fraction,
        artifact_id,
        str(patch_path) if patch_path else None,
        patch_sha,
        psycopg.types.json.Jsonb(quality or {}),
        psycopg.types.json.Jsonb(metadata or {}),
    )


def _insert_coverage(conn, values: tuple) -> None:
    conn.execute(
        """
        INSERT INTO weather.goes_abi_window_coverage (
          process_id,station_id,decision_time,requested_offset_minutes,product,
          feature_schema_version,status,satellite,scan_start,scan_end,valid_pixel_fraction,
          source_artifact_id,cropped_artifact_path,cropped_artifact_sha256,
          quality_flags,source_metadata
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (
          process_id,station_id,decision_time,requested_offset_minutes,product,feature_schema_version
        ) DO UPDATE SET
          status=EXCLUDED.status, satellite=EXCLUDED.satellite,
          scan_start=EXCLUDED.scan_start, scan_end=EXCLUDED.scan_end,
          valid_pixel_fraction=EXCLUDED.valid_pixel_fraction,
          source_artifact_id=EXCLUDED.source_artifact_id,
          cropped_artifact_path=EXCLUDED.cropped_artifact_path,
          cropped_artifact_sha256=EXCLUDED.cropped_artifact_sha256,
          quality_flags=EXCLUDED.quality_flags, source_metadata=EXCLUDED.source_metadata,
          checked_at=now()
        WHERE weather.goes_abi_window_coverage.status IN (
          'missing_source','satellite_transition_issue','download_failure','processing_failure'
        )
        """,
        values,
    )


def _feature_values(
    decision_time: datetime,
    offset: int,
    radius: int,
    sector: str,
    satellite: str,
    scan_end: datetime,
    summaries: dict[str, dict[str, Any]],
    changes: dict[str, float | None],
    gradients: dict[str, float | None],
    source_metadata: dict[str, Any],
) -> tuple:
    infrared = summaries.get("infrared_c13", {})
    clear = summaries.get("clear_sky_mask", {})
    cloud_temperature = summaries.get("cloud_top_temperature", {})
    cloud_height = summaries.get("cloud_top_height", {})
    visible = summaries.get("visible_c02", {})
    optical_depth = summaries.get("cloud_optical_depth", {})
    water_vapor = summaries.get("water_vapor_c08", {})
    valid_fraction = float(infrared.get("valid_pixel_fraction", 0.0))
    quality = {key: value.get("quality", {}) for key, value in summaries.items()}
    return (
        PROCESS_ID, STATION_ID, decision_time, offset, radius, sector,
        FEATURE_SCHEMA_VERSION, satellite, scan_end,
        clear.get("clear_fraction"), clear.get("cloudy_fraction"),
        infrared.get("mean"), infrared.get("stddev"), infrared.get("p10"),
        infrared.get("p50"), infrared.get("p90"),
        cloud_temperature.get("mean"), cloud_temperature.get("p10"),
        cloud_temperature.get("p50"), cloud_temperature.get("p90"),
        cloud_height.get("mean"), cloud_height.get("p10"), cloud_height.get("p50"),
        cloud_height.get("p90"), visible.get("mean"), visible.get("stddev"),
        visible.get("p10"), visible.get("p50"), visible.get("p90"),
        optical_depth.get("mean"), optical_depth.get("p50"), optical_depth.get("p90"),
        water_vapor.get("mean"), changes.get("change_45m"), changes.get("change_165m"),
        gradients.get("infrared_north_south"), gradients.get("infrared_east_west"),
        gradients.get("cloudy_north_south"), gradients.get("cloudy_east_west"),
        valid_fraction, psycopg.types.json.Jsonb(quality),
        psycopg.types.json.Jsonb(source_metadata),
    )


def _insert_feature(conn, values: tuple) -> None:
    conn.execute(
        """
        INSERT INTO weather.goes_abi_features (
          process_id,station_id,decision_time,requested_offset_minutes,spatial_radius_km,
          sector,feature_schema_version,satellite,scan_end,clear_pixel_fraction,
          cloudy_pixel_fraction,infrared_brightness_temperature_mean_k,
          infrared_brightness_temperature_stddev_k,infrared_brightness_temperature_p10_k,
          infrared_brightness_temperature_p50_k,infrared_brightness_temperature_p90_k,
          cloud_top_temperature_mean_k,cloud_top_temperature_p10_k,
          cloud_top_temperature_p50_k,cloud_top_temperature_p90_k,
          cloud_top_height_mean_m,cloud_top_height_p10_m,cloud_top_height_p50_m,
          cloud_top_height_p90_m,visible_reflectance_mean,visible_reflectance_stddev,
          visible_reflectance_p10,visible_reflectance_p50,visible_reflectance_p90,
          cloud_optical_depth_mean,cloud_optical_depth_p50,cloud_optical_depth_p90,
          water_vapor_brightness_temperature_mean_k,infrared_change_45m_k,
          infrared_change_165m_k,infrared_north_south_gradient_k,
          infrared_east_west_gradient_k,cloudy_north_south_gradient,
          cloudy_east_west_gradient,valid_pixel_fraction,quality_flags,source_metadata
        ) VALUES (
          %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
          %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
          %s,%s,%s,%s
        ) ON CONFLICT DO NOTHING
        """,
        values,
    )


def _spatial_difference(
    spatial: dict[tuple[int, str], dict[str, dict[str, Any]]],
    radius: int,
    product_key: str,
    first: str,
    second: str,
) -> float | None:
    statistic = "cloudy_fraction" if product_key == "clear_sky_mask" else "mean"
    first_value = spatial[(radius, first)].get(product_key, {}).get(statistic)
    second_value = spatial[(radius, second)].get(product_key, {}).get(statistic)
    if first_value is None or second_value is None:
        return None
    return first_value - second_value


def ingest_goes(settings: Settings, job: Job) -> dict[str, Any]:
    feature_version = str(job.request.get("feature_schema_version", FEATURE_SCHEMA_VERSION))
    if feature_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"unsupported GOES feature schema: {feature_version}")
    offsets = tuple(int(value) for value in job.request.get("scan_offsets", SCAN_OFFSETS_MINUTES))
    radii = tuple(int(value) for value in job.request.get("radii_km", RADII_KM))
    if offsets != SCAN_OFFSETS_MINUTES or radii != RADII_KM:
        raise ValueError("GOES v2 requires scan_offsets=[15,60,180] and radii_km=[25,50,100]")
    decisions = _decision_times(job.range_start, job.range_end)
    retry_statuses = set(
        job.request.get("retry_statuses", ("download_failure", "processing_failure"))
    )
    counters = {"completed": 0, "reused": 0, "missing": 0, "failed": 0}
    for decision_index, decision_time in enumerate(decisions, start=1):
        existing = _existing_coverage(settings, decision_time)
        satellite = operational_satellite(decision_time)
        bucket = f"noaa-goes{satellite[1:]}"
        per_offset: dict[int, dict[tuple[int, str], dict[str, dict[str, Any]]]] = (
            _existing_infrared_summaries(settings, decision_time)
        )
        offset_scan_ends: dict[int, datetime] = {}
        for offset in reversed(offsets):
            target = decision_time - timedelta(minutes=offset)
            applicable = [
                product
                for product in PRODUCTS
                if not product.noon_only or decision_time.astimezone(NYC).hour == 12
            ]
            retrying_offset = any(
                existing.get((offset, product.key)) in retry_statuses for product in applicable
            )
            pending = [
                product
                for product in applicable
                if retrying_offset
                or (offset, product.key) not in existing
                or existing[(offset, product.key)] in retry_statuses
            ]
            counters["reused"] += len(applicable) - len(pending)
            product_results: list[dict[str, Any]] = []
            spatial: dict[tuple[int, str], dict[str, dict[str, Any]]] = {
                (radius, sector): {} for radius in radii for sector in SECTORS
            }
            temp_sources: list[Path] = []
            for product in pending:
                try:
                    selection = select_causal_scan(
                        _archive_keys(satellite, product, target),
                        target=target,
                        decision_time=decision_time,
                    )
                    if selection is None:
                        missing_status = (
                            "satellite_transition_issue"
                            if abs(decision_time - GOES_TRANSITION) < timedelta(days=1)
                            else "missing_source"
                        )
                        product_results.append(
                            {"product": product, "status": missing_status, "metadata": {}}
                        )
                        counters["missing"] += 1
                        continue
                    key, scan_start, scan_end = selection
                    source_uri = f"https://{bucket}.s3.amazonaws.com/{key}"
                    source_path = settings.cache_directory / "goes" / Path(key).name
                    download_resumable(
                        source_uri,
                        source_path,
                        attempts=settings.noaa_download_attempts,
                        retry_base_seconds=settings.noaa_retry_base_seconds,
                    )
                    temp_sources.append(source_path)
                    source_sha, source_size = file_sha256(source_path)
                    stamp = decision_time.strftime("%Y%m%dT%H%M%SZ")
                    patch_path = (
                        settings.goes_directory
                        / satellite.lower()
                        / f"{decision_time:%Y/%m/%d}"
                        / stamp
                        / f"offset-{offset:03d}-{product.key}.nc"
                    )
                    summaries, variable_metadata = _extract_patch(source_path, product, patch_path)
                    patch_sha, patch_size = file_sha256(patch_path)
                    for key_tuple, summary in summaries.items():
                        spatial[key_tuple][product.key] = summary
                    valid_fraction = max(
                        (summary["valid_pixel_fraction"] for summary in summaries.values()),
                        default=0.0,
                    )
                    status = (
                        "insufficient_valid_pixels"
                        if valid_fraction < MINIMUM_VALID_PIXEL_FRACTION
                        else "valid_zero"
                        if all(
                            summary["mean"] == 0
                            for summary in summaries.values()
                            if summary["mean"] is not None
                        )
                        else "complete"
                    )
                    product_results.append(
                        {
                            "product": product,
                            "status": status,
                            "source_uri": source_uri,
                            "source_path": source_path,
                            "source_sha": source_sha,
                            "source_size": source_size,
                            "scan_start": scan_start,
                            "scan_end": scan_end,
                            "patch_path": patch_path,
                            "patch_sha": patch_sha,
                            "patch_size": patch_size,
                            "valid_fraction": valid_fraction,
                            "raw_valid_fraction_100": summaries[(100, "all")][
                                "valid_pixel_fraction"
                            ],
                            "quality": {str(key): value["quality"] for key, value in summaries.items()},
                            "metadata": variable_metadata,
                        }
                    )
                    offset_scan_ends[offset] = max(offset_scan_ends.get(offset, scan_end), scan_end)
                except Exception as error:  # noqa: BLE001 - persist product-level failure evidence.
                    status = "download_failure" if "download" in str(error).lower() else "processing_failure"
                    product_results.append(
                        {"product": product, "status": status, "metadata": {"error": str(error)[:1000]}}
                    )
                    counters["failed"] += 1
            clear_summary = spatial[(100, "all")].get("clear_sky_mask", {})
            cloudy_fraction = clear_summary.get("cloudy_fraction")
            for result in product_results:
                if result["product"].key not in (
                    "cloud_top_temperature",
                    "cloud_top_height",
                ) or result["status"] not in (
                    "complete",
                    "valid_zero",
                    "insufficient_valid_pixels",
                ):
                    continue
                if cloudy_fraction == 0:
                    result["status"] = "valid_zero"
                    result["valid_fraction"] = 1.0
                elif cloudy_fraction:
                    cloud_valid_fraction = min(
                        1.0, result["raw_valid_fraction_100"] / cloudy_fraction
                    )
                    result["valid_fraction"] = cloud_valid_fraction
                    result["status"] = (
                        "complete"
                        if cloud_valid_fraction >= MINIMUM_VALID_PIXEL_FRACTION
                        else "insufficient_valid_pixels"
                    )
            with connection(settings.database_url) as conn, conn.transaction():
                source_metadata: dict[str, Any] = {}
                for result in product_results:
                    artifact_id = None
                    if result.get("source_path"):
                        logical_key = f"{satellite.lower()}:{result['source_uri'].rsplit('/', 1)[-1]}"
                        artifact_id = insert_unified_backfill_artifact(
                            conn,
                            job_id=job.job_id,
                            strategy_key=job.ingester_key,
                            provider="noaa_goes_open_data",
                            logical_key=logical_key,
                            source_uri=result["source_uri"],
                            sha256=result["source_sha"],
                            compressed_bytes=result["source_size"],
                            record_count=1,
                            metadata={
                                **result["metadata"],
                                "product": result["product"].key,
                                "cropped_artifact_path": str(result["patch_path"]),
                                "cropped_artifact_sha256": result["patch_sha"],
                                "cropped_bytes": result["patch_size"],
                            },
                            source_start=result["scan_start"],
                            source_end=result["scan_end"],
                        )
                        source_metadata[result["product"].key] = {
                            "source_artifact_id": artifact_id,
                            "scan_end": result["scan_end"].isoformat(),
                            "cropped_artifact_sha256": result["patch_sha"],
                        }
                    _insert_coverage(
                        conn,
                        _coverage_values(
                            decision_time=decision_time,
                            offset=offset,
                            product=result["product"],
                            satellite=satellite,
                            status=result["status"],
                            scan_start=result.get("scan_start"),
                            scan_end=result.get("scan_end"),
                            valid_fraction=result.get("valid_fraction"),
                            artifact_id=artifact_id,
                            patch_path=result.get("patch_path"),
                            patch_sha=result.get("patch_sha"),
                            quality=result.get("quality"),
                            metadata=result.get("metadata"),
                        ),
                    )
                    if result["status"] in ("complete", "valid_zero"):
                        counters["completed"] += 1
                transient_failure = any(
                    result["status"] in ("download_failure", "processing_failure")
                    for result in product_results
                )
                if not transient_failure and "infrared_c13" in next(iter(spatial.values()), {}):
                    per_offset[offset] = spatial
                    for radius in radii:
                        for sector in SECTORS:
                            current = spatial[(radius, sector)]
                            current_ir = current["infrared_c13"].get("mean")
                            changes: dict[str, float | None] = {
                                "change_45m": None,
                                "change_165m": None,
                            }
                            for prior_offset, name in ((60, "change_45m"), (180, "change_165m")):
                                prior = per_offset.get(prior_offset, {}).get((radius, sector), {})
                                prior_ir = prior.get("infrared_c13", {}).get("mean")
                                if current_ir is not None and prior_ir is not None and offset == 15:
                                    changes[name] = current_ir - prior_ir
                            gradients = {
                                "infrared_north_south": _spatial_difference(
                                    spatial, radius, "infrared_c13", "north", "south"
                                ),
                                "infrared_east_west": _spatial_difference(
                                    spatial, radius, "infrared_c13", "east", "west"
                                ),
                                "cloudy_north_south": _spatial_difference(
                                    spatial, radius, "clear_sky_mask", "north", "south"
                                ),
                                "cloudy_east_west": _spatial_difference(
                                    spatial, radius, "clear_sky_mask", "east", "west"
                                ),
                            }
                            _insert_feature(
                                conn,
                                _feature_values(
                                    decision_time, offset, radius, sector, satellite,
                                    offset_scan_ends.get(offset, decision_time - timedelta(minutes=15)),
                                    current, changes, gradients, source_metadata,
                                ),
                            )
            for source_path in temp_sources:
                source_path.unlink(missing_ok=True)
        update_progress(
            settings,
            job,
            {
                **counters,
                "decisions_completed": decision_index,
                "decisions_total": len(decisions),
                "last_decision_time": decision_time.isoformat(),
            },
        )
    return {**counters, "decisions": len(decisions), "feature_schema_version": FEATURE_SCHEMA_VERSION}
