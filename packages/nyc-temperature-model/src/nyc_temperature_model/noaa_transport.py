from __future__ import annotations

import os
import time
from pathlib import Path
from xml.etree import ElementTree

import httpx


def list_s3_keys(bucket: str, prefix: str, *, timeout_seconds: float = 30) -> list[str]:
    keys: list[str] = []
    continuation: str | None = None
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        while True:
            params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if continuation:
                params["continuation-token"] = continuation
            response = client.get(f"https://{bucket}.s3.amazonaws.com/", params=params)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            keys.extend(node.text or "" for node in root.findall("s3:Contents/s3:Key", namespace))
            truncated = root.findtext("s3:IsTruncated", default="false", namespaces=namespace)
            if truncated.lower() != "true":
                return keys
            continuation = root.findtext("s3:NextContinuationToken", namespaces=namespace)
            if not continuation:
                raise RuntimeError("NOAA S3 listing was truncated without a continuation token")


def download_resumable(
    source_uri: str,
    destination: Path,
    *,
    attempts: int,
    retry_base_seconds: float,
    expected_size: int | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with httpx.stream(
                "GET", source_uri, headers=headers, timeout=120, follow_redirects=True
            ) as response:
                if offset and response.status_code == 200:
                    partial.unlink()
                    offset = 0
                response.raise_for_status()
                mode = "ab" if offset and response.status_code == 206 else "wb"
                with partial.open(mode) as output:
                    for chunk in response.iter_bytes(1024 * 1024):
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise OSError(
                    f"downloaded size {partial.stat().st_size} did not match {expected_size}"
                )
            os.replace(partial, destination)
            return destination
        except (httpx.HTTPError, OSError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(retry_base_seconds * 2**attempt)
    raise RuntimeError(f"download failed after {attempts} attempts: {source_uri}") from last_error
