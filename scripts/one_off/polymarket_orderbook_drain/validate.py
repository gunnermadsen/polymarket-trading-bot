from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from .contract import CONTRACT_VERSION, schema


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Polymarket orderbook Parquet drain")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifests = sorted(args.root.glob("date=*/hour=*.manifest.json"))
    if not manifests:
        raise RuntimeError("no partition manifests found")
    totals = {"source_total": 0, "duplicate_count": 0, "output_count": 0}
    expected_schema = schema()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        parquet = Path(manifest["file"])
        if manifest["contract_version"] != CONTRACT_VERSION:
            raise RuntimeError(f"contract mismatch: {parquet}")
        if sha256(parquet) != manifest["file_sha256"]:
            raise RuntimeError(f"checksum mismatch: {parquet}")
        metadata = pq.read_metadata(parquet)
        if metadata.num_rows != manifest["output_count"]:
            raise RuntimeError(f"row count mismatch: {parquet}")
        if pq.read_schema(parquet) != expected_schema:
            raise RuntimeError(f"schema mismatch: {parquet}")
        if manifest["source_total"] != manifest["output_count"] + manifest["duplicate_count"]:
            raise RuntimeError(f"accounting mismatch: {parquet}")
        for key in totals:
            totals[key] += manifest[key]
    print(json.dumps({"partition_count": len(manifests), **totals}, sort_keys=True))


if __name__ == "__main__":
    main()
