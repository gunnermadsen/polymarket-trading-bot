from __future__ import annotations

import hashlib
import os
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


def download_atomic(client: httpx.Client, uri: str, path: Path, maximum_bytes: int) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    digest = hashlib.sha256()
    total = 0
    with client.stream("GET", uri) as response:
        if response.status_code == 404:
            raise FileNotFoundError(uri)
        response.raise_for_status()
        length = response.headers.get("content-length")
        if length and int(length) > maximum_bytes:
            raise ValueError(f"source object exceeds {maximum_bytes} bytes: {uri}")
        with temporary.open("wb") as output:
            for chunk in response.iter_bytes(8 * 1024 * 1024):
                total += len(chunk)
                if total > maximum_bytes:
                    raise ValueError(f"source object exceeds {maximum_bytes} bytes: {uri}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    temporary.replace(path)
    return digest.hexdigest(), total
