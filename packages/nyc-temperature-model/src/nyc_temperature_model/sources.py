from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import httpx


def source_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(120, connect=20),
        follow_redirects=True,
        headers={"User-Agent": "capitonic-nyc-temperature-model/0.1"},
    )


def atomic_write(path: Path, content: bytes) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    return digest, len(content)


def file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _partial_path(path: Path) -> Path:
    partial = path.with_name(f".{path.name}.partial")
    if partial.exists():
        return partial
    legacy = sorted(
        path.parent.glob(f".{path.name}.*.partial"),
        key=lambda candidate: candidate.stat().st_size,
        reverse=True,
    )
    if legacy:
        legacy[0].replace(partial)
    return partial


def _content_range_total(value: str | None) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
    if not match:
        return None
    return int(match.group(3))


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in {408, 425, 429, 500, 502, 503, 504}
    return False


def _download_once(client: httpx.Client, uri: str, path: Path, maximum_bytes: int) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _partial_path(path)
    existing = temporary.stat().st_size if temporary.exists() else 0
    if existing > maximum_bytes:
        raise ValueError(f"partial source object exceeds {maximum_bytes} bytes: {uri}")
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with client.stream("GET", uri, headers=headers) as response:
        if response.status_code == 404:
            raise FileNotFoundError(uri)
        if response.status_code == 416 and existing:
            expected = _content_range_total(response.headers.get("content-range"))
            if expected == existing:
                digest, size = file_sha256(temporary)
                temporary.replace(path)
                return digest, size
        response.raise_for_status()
        if existing and response.status_code == 200:
            existing = 0
        elif existing and response.status_code != 206:
            raise RuntimeError(f"source did not honor archive range request: {uri}")
        expected = _content_range_total(response.headers.get("content-range"))
        length = response.headers.get("content-length")
        expected_size = expected if expected is not None else existing + (int(length) if length else 0)
        if expected_size > maximum_bytes:
            raise ValueError(f"source object exceeds {maximum_bytes} bytes: {uri}")
        with temporary.open("ab" if existing else "wb") as output:
            for chunk in response.iter_bytes(8 * 1024 * 1024):
                if output.tell() + len(chunk) > maximum_bytes:
                    raise ValueError(f"source object exceeds {maximum_bytes} bytes: {uri}")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    digest, size = file_sha256(temporary)
    if expected_size and size != expected_size:
        raise RuntimeError(f"incomplete source object after download: {uri}")
    temporary.replace(path)
    return digest, size


def download_atomic(
    client: httpx.Client,
    uri: str,
    path: Path,
    maximum_bytes: int,
    *,
    attempts: int = 1,
    retry_base_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, int]:
    if attempts <= 0:
        raise ValueError("download attempts must be positive")
    if retry_base_seconds < 0:
        raise ValueError("download retry base seconds must be non-negative")
    for attempt in range(attempts):
        try:
            return _download_once(client, uri, path, maximum_bytes)
        except Exception as error:
            if not _is_retryable(error) or attempt + 1 == attempts:
                raise
            sleep(min(60.0, retry_base_seconds * (2**attempt)))
    raise AssertionError("unreachable")
