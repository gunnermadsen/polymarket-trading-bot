from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx


@dataclass(frozen=True)
class Job:
    job_id: str
    ingester_key: str
    range_start: datetime
    range_end: datetime
    request: dict[str, Any]
    attempt: int
    lease_token: str
    worker_id: str


def update_progress(settings, job: Job, progress: dict[str, Any]) -> None:
    del settings
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
