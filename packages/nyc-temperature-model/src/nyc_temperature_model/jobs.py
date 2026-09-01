from __future__ import annotations

import logging
import os
import signal
import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg
import httpx

from .config import SUPPORTED_INGESTERS, Settings
from .database import connection


@dataclass(frozen=True)
class Job:
    job_id: str
    ingester_key: str
    range_start: datetime
    range_end: datetime
    request: dict[str, Any]
    attempt: int
    lease_token: str
    unified: bool = False
    worker_id: str | None = None


def enqueue(
    database_url: str,
    *,
    ingester_key: str,
    range_start: datetime,
    range_end: datetime,
    parameters: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    depends_on_job_id: str | None = None,
) -> str:
    if ingester_key not in SUPPORTED_INGESTERS:
        raise ValueError(f"unsupported ingester: {ingester_key}")
    if range_start.tzinfo is None or range_end.tzinfo is None or range_end <= range_start:
        raise ValueError("the ingestion range must be an increasing timezone-aware interval")
    idempotency_key = idempotency_key or (
        f"{ingester_key}:{range_start.isoformat()}:{range_end.isoformat()}:v1"
    )
    with connection(database_url) as conn, conn.transaction():
        row = conn.execute(
            """
            INSERT INTO weather.ingestion_jobs (
              depends_on_job_id, ingester_key, idempotency_key, range_start, range_end, request
            ) VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ingester_key, idempotency_key) DO UPDATE SET
              updated_at = weather.ingestion_jobs.updated_at
            RETURNING job_id::text
            """,
            (
                depends_on_job_id,
                ingester_key,
                idempotency_key,
                range_start,
                range_end,
                psycopg.types.json.Jsonb(parameters or {}),
            ),
        ).fetchone()
    return row["job_id"]


def claim(settings: Settings) -> Job | None:
    lease_token = str(uuid.uuid4())
    with connection(settings.database_url) as conn, conn.transaction():
        conn.execute(
            """
            UPDATE weather.ingestion_jobs
            SET status = CASE WHEN attempt >= max_attempts THEN 'failed' ELSE 'queued' END,
                worker_id = NULL, lease_token = NULL,
                lease_expires_at = NULL, heartbeat_at = NULL,
                next_attempt_at = now(), updated_at = now(),
                completed_at = CASE WHEN attempt >= max_attempts THEN now() ELSE completed_at END,
                error = COALESCE(error, 'worker lease expired')
            WHERE status = 'running' AND lease_expires_at < now()
            """
        )
        conn.execute(
            """
            UPDATE weather.ingestion_jobs
            SET status = 'cancelled', completed_at = now(), lease_expires_at = NULL,
                heartbeat_at = now(), updated_at = now()
            WHERE status = 'cancel_requested' AND lease_expires_at < now()
            """
        )
        row = conn.execute(
            """
            WITH candidate AS (
              SELECT job_id
              FROM weather.ingestion_jobs
              WHERE status = 'queued' AND next_attempt_at <= now() AND attempt < max_attempts
                AND ingester_key = ANY(%s::text[])
                AND (
                  depends_on_job_id IS NULL
                  OR EXISTS (
                    SELECT 1 FROM weather.ingestion_jobs dependency
                    WHERE dependency.job_id=weather.ingestion_jobs.depends_on_job_id
                      AND dependency.status='completed'
                  )
                )
              ORDER BY requested_at, next_attempt_at, job_id
              FOR UPDATE SKIP LOCKED
              LIMIT 1
            )
            UPDATE weather.ingestion_jobs AS job
            SET status = 'running', attempt = attempt + 1,
                worker_id = %s, lease_token = %s,
                lease_expires_at = now() + make_interval(secs => %s),
                heartbeat_at = now(), started_at = COALESCE(started_at, now()),
                updated_at = now(), error = NULL
            FROM candidate
            WHERE job.job_id = candidate.job_id
            RETURNING job.job_id::text, job.ingester_key, job.range_start,
                      job.range_end, job.request, job.attempt, job.lease_token::text
            """,
            (
                list(settings.worker_ingesters),
                settings.worker_id,
                lease_token,
                settings.lease_seconds,
            ),
        ).fetchone()
    return Job(**row) if row else None


class Heartbeat:
    def __init__(self, settings: Settings, job: Job):
        self.settings = settings
        self.job = job
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _run(self) -> None:
        interval = max(5, min(60, self.settings.lease_seconds // 4))
        while not self.stop_event.wait(interval):
            try:
                with connection(self.settings.database_url, autocommit=True) as conn:
                    result = conn.execute(
                        """
                        UPDATE weather.ingestion_jobs
                        SET heartbeat_at = now(),
                            lease_expires_at = now() + make_interval(secs => %s),
                            updated_at = now()
                        WHERE job_id = %s AND lease_token = %s AND status = 'running'
                        """,
                        (self.settings.lease_seconds, self.job.job_id, self.job.lease_token),
                    )
                    if result.rowcount != 1:
                        self.stop_event.set()
            except psycopg.Error as error:
                logging.getLogger(__name__).warning("weather worker heartbeat failed: %s", error)


def update_progress(settings: Settings, job: Job, progress: dict[str, Any]) -> None:
    if job.unified:
        master_url = os.environ["INGESTER_MASTER_URL"].rstrip("/")
        admin_token = os.environ.get("INGESTER_ADMIN_TOKEN") or os.environ[
            "MARKET_DATA_INGESTER_ADMIN_TOKEN"
        ]
        response = httpx.post(
            f"{master_url}/internal/workers/{job.worker_id}/jobs/{job.job_id}/heartbeat",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"lease_token": job.lease_token, "progress": progress, "checkpoint": progress},
            timeout=30,
        )
        if response.status_code == 409:
            raise RuntimeError("ingestion job lease was lost")
        response.raise_for_status()
        return
    with connection(settings.database_url, autocommit=True) as conn:
        result = conn.execute(
            """
            UPDATE weather.ingestion_jobs
            SET progress = %s, updated_at = now()
            WHERE job_id = %s AND lease_token = %s AND status = 'running'
            """,
            (psycopg.types.json.Jsonb(progress), job.job_id, job.lease_token),
        )
        if result.rowcount != 1:
            raise RuntimeError("ingestion job lease was lost")


def finish(settings: Settings, job: Job, summary: dict[str, Any]) -> None:
    with connection(settings.database_url, autocommit=True) as conn:
        result = conn.execute(
            """
            UPDATE weather.ingestion_jobs
            SET status = 'completed', summary = %s, completed_at = now(),
                lease_expires_at = NULL, heartbeat_at = now(), updated_at = now()
            WHERE job_id = %s AND lease_token = %s AND status = 'running'
            """,
            (psycopg.types.json.Jsonb(summary), job.job_id, job.lease_token),
        )
        if result.rowcount != 1:
            raise RuntimeError("ingestion job lease was lost before completion")


def fail(settings: Settings, job: Job, error: BaseException) -> None:
    message = "".join(traceback.format_exception_only(type(error), error)).strip()[:8000]
    with connection(settings.database_url, autocommit=True) as conn:
        conn.execute(
            """
            UPDATE weather.ingestion_jobs
            SET status = CASE WHEN attempt >= max_attempts THEN 'failed' ELSE 'queued' END,
                next_attempt_at = now() + make_interval(secs => LEAST(900, 15 * (2 ^ attempt))),
                error = %s, lease_expires_at = NULL, heartbeat_at = now(), updated_at = now()
            WHERE job_id = %s AND lease_token = %s AND status = 'running'
            """,
            (message, job.job_id, job.lease_token),
        )


def run_worker(settings: Settings, handlers: dict[str, Callable]) -> None:
    settings.prepare_directories()
    stopping = threading.Event()

    def stop(*_):
        stopping.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping.is_set():
        job = claim(settings)
        if job is None:
            stopping.wait(settings.poll_seconds)
            continue
        try:
            with Heartbeat(settings, job):
                summary = handlers[job.ingester_key](settings, job)
            finish(settings, job, summary)
        except Exception as error:
            logging.getLogger(__name__).exception(
                "weather ingestion job failed: job_id=%s ingester=%s attempt=%s",
                job.job_id,
                job.ingester_key,
                job.attempt,
            )
            fail(settings, job, error)


def job_rows(database_url: str, limit: int = 50) -> list[dict[str, Any]]:
    with connection(database_url) as conn:
        return list(
            conn.execute(
                """
                SELECT job_id::text, ingester_key, status, attempt, range_start, range_end,
                       progress, summary, error, requested_at, completed_at
                FROM weather.ingestion_jobs
                ORDER BY requested_at DESC, job_id DESC
                LIMIT %s
                """,
                (max(1, min(limit, 500)),),
            ).fetchall()
        )


def cancel_job(database_url: str, job_id: str) -> str:
    with connection(database_url) as conn, conn.transaction():
        row = conn.execute(
            """
            UPDATE weather.ingestion_jobs
            SET status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE 'cancel_requested' END,
                completed_at = CASE WHEN status = 'queued' THEN now() ELSE completed_at END,
                updated_at = now()
            WHERE job_id=%s AND status IN ('queued','running')
            RETURNING status
            """,
            (job_id,),
        ).fetchone()
    if not row:
        raise ValueError("job is not queued or running")
    return row["status"]


def json_default(value):
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)
