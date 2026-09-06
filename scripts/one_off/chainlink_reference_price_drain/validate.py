from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from .contract import DIRECT_CONTRACT_VERSION, PMDATA_CONTRACT_VERSION, schema
from .drain import sha256


def validate_product(root: Path, product: str, contract_version: str) -> dict[str, int]:
    manifests = sorted((root / product).glob("date=*/hour=*.manifest.json"))
    if not manifests:
        raise RuntimeError(f"no {product} manifests")
    totals = {"source_total": 0, "duplicate_count": 0, "conflict_count": 0, "output_count": 0}
    expected = schema(contract_version)
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        parquet = Path(manifest["file"])
        if manifest["contract_version"] != contract_version:
            raise RuntimeError(f"contract mismatch: {parquet}")
        if sha256(parquet) != manifest["file_sha256"]:
            raise RuntimeError(f"checksum mismatch: {parquet}")
        if pq.read_schema(parquet) != expected:
            raise RuntimeError(f"schema mismatch: {parquet}")
        if pq.read_metadata(parquet).num_rows != manifest["output_count"]:
            raise RuntimeError(f"row count mismatch: {parquet}")
        if manifest["source_total"] != manifest["output_count"] + manifest["duplicate_count"]:
            raise RuntimeError(f"accounting mismatch: {parquet}")
        if manifest["conflict_count"]:
            raise RuntimeError(f"conflicts present: {parquet}")
        for key in totals:
            totals[key] += manifest[key]
    return {"partition_count": len(manifests), **totals}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical Chainlink reference-price Parquet")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = {
        "direct": validate_product(args.root, "direct", DIRECT_CONTRACT_VERSION),
        "pmdata": validate_product(args.root, "pmdata", PMDATA_CONTRACT_VERSION),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
