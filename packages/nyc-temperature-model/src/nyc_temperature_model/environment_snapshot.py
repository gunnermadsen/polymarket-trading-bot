from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from . import PROCESS_ID, STATION_ID
from .config import Settings
from .database import connection
from .goes_ingestion import FEATURE_SCHEMA_VERSION as GOES_VERSION
from .hrrr_environment_ingestion import FEATURE_SCHEMA_VERSION as HRRR_VERSION
from .sources import file_sha256


def _write_query_parquet(conn, query: str, parameters: tuple, destination: Path) -> int:
    cursor = conn.cursor(name=f"snapshot_{destination.stem}")
    cursor.execute(query, parameters)
    writer = None
    rows_written = 0
    try:
        while rows := cursor.fetchmany(10_000):
            table = pa.Table.from_pylist([dict(row) for row in rows])
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            rows_written += len(rows)
        if writer is None:
            raise ValueError(f"snapshot query produced no rows for {destination.name}")
    finally:
        if writer is not None:
            writer.close()
        cursor.close()
    return rows_written


def _manifest_query() -> str:
    return """
      WITH coverage AS (
        SELECT 'goes'::text AS source_family, decision_time, product AS dimension,
               status, source_artifact_id, cropped_artifact_path, cropped_artifact_sha256
        FROM weather.goes_abi_window_coverage
        WHERE process_id=%s AND station_id=%s AND decision_time >= %s AND decision_time < %s
          AND feature_schema_version=%s
        UNION ALL
        SELECT 'hrrr_environment', decision_time, valid_at::text, status,
               source_artifact_id, cropped_artifact_path, cropped_artifact_sha256
        FROM weather.hrrr_environment_window_coverage
        WHERE process_id=%s AND station_id=%s AND decision_time >= %s AND decision_time < %s
          AND feature_schema_version=%s
      )
      SELECT c.source_family,c.decision_time,c.dimension,c.status,
             c.cropped_artifact_path,c.cropped_artifact_sha256,
             a.provider,a.logical_key,a.source_uri,a.sha256 AS source_sha256,
             a.compressed_bytes,a.source_start,a.source_end,a.metadata::text AS source_metadata
      FROM coverage c
      LEFT JOIN weather.source_artifacts a ON a.artifact_id=c.source_artifact_id
      ORDER BY c.source_family,c.decision_time,c.dimension
    """


def _verify_cropped_artifacts(rows: list[dict[str, Any]]) -> dict[str, int]:
    verified = missing = mismatched = 0
    seen: set[tuple[str, str]] = set()
    for row in rows:
        path_value = row.get("cropped_artifact_path")
        expected = row.get("cropped_artifact_sha256")
        if not path_value or not expected or (path_value, expected) in seen:
            continue
        seen.add((path_value, expected))
        path = Path(path_value)
        if not path.is_file():
            missing += 1
            continue
        actual, _ = file_sha256(path)
        if actual != expected:
            mismatched += 1
        else:
            verified += 1
    if missing or mismatched:
        raise RuntimeError(
            f"cropped artifact verification failed: missing={missing} mismatched={mismatched}"
        )
    return {"verified": verified, "missing": missing, "mismatched": mismatched}


def export_environment_snapshot(
    settings: Settings, *, start: date, end: date, snapshot_id: str
) -> dict[str, Any]:
    if not start < end:
        raise ValueError("snapshot end must be after start")
    if not snapshot_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in snapshot_id):
        raise ValueError("snapshot_id must contain lowercase letters, digits, hyphens, or underscores")
    destination = settings.training_snapshot_directory / snapshot_id
    if destination.exists():
        raise FileExistsError(f"immutable snapshot already exists: {destination}")
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    partial.mkdir(parents=True, exist_ok=False)
    start_at = datetime.combine(start, datetime.min.time(), UTC)
    end_at = datetime.combine(end, datetime.min.time(), UTC)
    try:
        with connection(settings.database_url) as conn:
            satellite_rows = _write_query_parquet(
                conn,
                """
                SELECT g.*,l.station_daily_max_f,l.station_rounded_max_f,l.winner_matches_station
                FROM weather.goes_abi_features g
                LEFT JOIN weather.label_reconciliation l
                  ON l.process_id=g.process_id
                 AND l.event_date=(g.decision_time AT TIME ZONE 'America/New_York')::date
                WHERE g.process_id=%s AND g.station_id=%s
                  AND g.decision_time >= %s AND g.decision_time < %s
                  AND g.feature_schema_version=%s
                ORDER BY g.decision_time,g.requested_offset_minutes,g.spatial_radius_km,g.sector
                """,
                (PROCESS_ID, STATION_ID, start_at, end_at, GOES_VERSION),
                partial / "satellite_training_matrix.parquet",
            )
            hrrr_rows = _write_query_parquet(
                conn,
                """
                SELECT h.*,l.station_daily_max_f,l.station_rounded_max_f,l.winner_matches_station
                FROM weather.hrrr_environment_features h
                LEFT JOIN weather.label_reconciliation l
                  ON l.process_id=h.process_id
                 AND l.event_date=(h.decision_time AT TIME ZONE 'America/New_York')::date
                WHERE h.process_id=%s AND h.station_id=%s
                  AND h.decision_time >= %s AND h.decision_time < %s
                  AND h.feature_schema_version=%s
                ORDER BY h.decision_time,h.valid_at,h.spatial_radius_km,h.sector
                """,
                (PROCESS_ID, STATION_ID, start_at, end_at, HRRR_VERSION),
                partial / "hrrr_environment_training_matrix.parquet",
            )
            manifest_rows = list(
                conn.execute(
                    _manifest_query(),
                    (
                        PROCESS_ID, STATION_ID, start_at, end_at, GOES_VERSION,
                        PROCESS_ID, STATION_ID, start_at, end_at, HRRR_VERSION,
                    ),
                ).fetchall()
            )
        artifact_verification = _verify_cropped_artifacts(manifest_rows)
        manifest_table = pa.Table.from_pylist([dict(row) for row in manifest_rows])
        pq.write_table(manifest_table, partial / "artifact_manifest.parquet", compression="zstd")
        file_manifest = {}
        for path in sorted(partial.glob("*.parquet")):
            digest, size = file_sha256(path)
            file_manifest[path.name] = {"sha256": digest, "bytes": size}
        manifest = {
            "snapshot_id": snapshot_id,
            "created_at": datetime.now(UTC).isoformat(),
            "process_id": PROCESS_ID,
            "station_id": STATION_ID,
            "range": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
            "feature_schema_versions": {"goes": GOES_VERSION, "hrrr_environment": HRRR_VERSION},
            "rows": {
                "satellite_training_matrix": satellite_rows,
                "hrrr_environment_training_matrix": hrrr_rows,
                "artifact_manifest": len(manifest_rows),
            },
            "cropped_artifact_verification": artifact_verification,
            "files": file_manifest,
            "training_performed": False,
        }
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode()
        (partial / "snapshot-manifest.json").write_bytes(manifest_bytes)
        (partial / "snapshot.sha256").write_text(
            hashlib.sha256(manifest_bytes).hexdigest() + "  snapshot-manifest.json\n"
        )
        os.replace(partial, destination)
        return {**manifest, "path": str(destination)}
    except Exception:  # noqa: TRY203 - preserve a diagnostic partial snapshot.
        # Keep a failed partial snapshot for diagnosis; it can never be mistaken for immutable output.
        raise


def audit_environment_coverage(settings: Settings, *, start: date, end: date) -> dict[str, Any]:
    start_at = datetime.combine(start, datetime.min.time(), UTC)
    end_at = datetime.combine(end, datetime.min.time(), UTC)
    with connection(settings.database_url) as conn:
        satellite_coverage = list(
            conn.execute(
                """
                SELECT EXTRACT(HOUR FROM decision_time AT TIME ZONE 'America/New_York')::int AS decision_hour,
                       product,count(*)::int AS expected,
                       count(*) FILTER (WHERE status IN ('complete','valid_zero'))::int AS available
                FROM weather.goes_abi_window_coverage
                WHERE process_id=%s AND station_id=%s AND decision_time >= %s AND decision_time < %s
                  AND feature_schema_version=%s
                  AND product IN ('infrared_c13','clear_sky_mask','cloud_top_temperature',
                                  'cloud_top_height','visible_c02')
                GROUP BY 1,2 ORDER BY 1,2
                """,
                (PROCESS_ID, STATION_ID, start_at, end_at, GOES_VERSION),
            ).fetchall()
        )
        checks = conn.execute(
            """
            SELECT
              (SELECT count(*)::int FROM weather.goes_abi_features
                WHERE process_id=%s AND decision_time >= %s AND decision_time < %s
                  AND scan_end > decision_time - interval '15 minutes') AS causal_violations,
              (SELECT count(*)::int FROM weather.goes_abi_features
                WHERE process_id=%s AND decision_time >= %s AND decision_time < %s
                  AND (valid_pixel_fraction NOT BETWEEN 0 AND 1
                    OR infrared_brightness_temperature_mean_k NOT BETWEEN 100 AND 400))
                AS satellite_physical_bound_violations,
              (SELECT count(*)::int FROM weather.hrrr_environment_features
                WHERE process_id=%s AND decision_time >= %s AND decision_time < %s
                  AND (valid_pixel_fraction NOT BETWEEN 0 AND 1
                    OR temperature_2m_mean_k NOT BETWEEN 180 AND 340
                    OR total_cloud_cover_mean_fraction NOT BETWEEN 0 AND 1))
                AS hrrr_physical_bound_violations
            """,
            (
                PROCESS_ID, start_at, end_at,
                PROCESS_ID, start_at, end_at,
                PROCESS_ID, start_at, end_at,
            ),
        ).fetchone()
        transition = list(
            conn.execute(
                """
                SELECT satellite,
                       EXTRACT(HOUR FROM decision_time AT TIME ZONE 'America/New_York')::int AS decision_hour,
                       count(*)::int AS samples,
                       avg(infrared_brightness_temperature_mean_k) AS infrared_mean_k,
                       stddev_samp(infrared_brightness_temperature_mean_k) AS infrared_stddev_k
                FROM weather.goes_abi_features
                WHERE process_id=%s AND station_id=%s AND spatial_radius_km=100 AND sector='all'
                  AND requested_offset_minutes=15
                  AND decision_time >= timestamptz '2025-03-08 15:10:00+00'
                  AND decision_time < timestamptz '2025-05-08 15:10:00+00'
                  AND feature_schema_version=%s
                GROUP BY 1,2 ORDER BY 1,2
                """,
                (PROCESS_ID, STATION_ID, GOES_VERSION),
            ).fetchall()
        )
    coverage = []
    for row in satellite_coverage:
        item = dict(row)
        item["coverage_fraction"] = item["available"] / item["expected"] if item["expected"] else 0
        coverage.append(item)
    return {
        "range": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "satellite_required_coverage": coverage,
        "minimum_required_coverage_met": bool(coverage) and all(
            row["coverage_fraction"] >= 0.95 for row in coverage
        ),
        "checks": dict(checks),
        "transition_distribution_audit": [dict(row) for row in transition],
    }
