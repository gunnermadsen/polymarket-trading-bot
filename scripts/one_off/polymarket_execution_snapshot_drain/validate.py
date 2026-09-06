from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from .contract import CONTRACT_VERSION, schema
from .drain import sha256


def validate(root: Path) -> dict[str, int]:
    expected = schema()
    manifests = sorted(root.glob("date=*/hour=*.manifest.json"))
    if not manifests:
        raise RuntimeError("no execution-snapshot partitions")
    totals = {"source_total": 0, "output_count": 0, "duplicate_count": 0, "conflict_count": 0}
    previous_end = None
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        parquet = Path(manifest["file"])
        if manifest["contract_version"] != CONTRACT_VERSION:
            raise RuntimeError(f"contract mismatch: {parquet}")
        if pq.read_schema(parquet) != expected or sha256(parquet) != manifest["file_sha256"]:
            raise RuntimeError(f"invalid partition: {parquet}")
        if pq.read_metadata(parquet).num_rows != manifest["output_count"]:
            raise RuntimeError(f"row count mismatch: {parquet}")
        if manifest["source_total"] != manifest["output_count"] + manifest["duplicate_count"]:
            raise RuntimeError(f"accounting mismatch: {parquet}")
        if manifest["conflict_count"]:
            raise RuntimeError(f"conflicts present: {parquet}")
        if previous_end is not None and manifest["window_start"] != previous_end:
            raise RuntimeError(f"partition gap before: {parquet}")
        previous_end = manifest["window_end"]
        for key in totals:
            totals[key] += manifest[key]
    result = {"partition_count": len(manifests), **totals}
    global_manifest = json.loads((root / "manifest.json").read_text())
    for key, value in result.items():
        if global_manifest[key] != value:
            raise RuntimeError(f"global manifest mismatch for {key}")
    if sum(item["row_count"] for item in global_manifest["source_watermarks"].values()) != totals["source_total"]:
        raise RuntimeError("source watermark counts do not match partition accounting")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
