from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from .contract import PRODUCTS, schema
from .drain import sha256


def validate(root: Path, product: str) -> dict[str, int]:
    expected = schema(PRODUCTS[product]["contract_version"])
    manifests = sorted((root / product).glob("date=*/hour=*.manifest.json"))
    if not manifests:
        raise RuntimeError(f"no {product} partitions")
    totals = {"source_total": 0, "output_count": 0, "duplicate_count": 0, "conflict_count": 0}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        parquet = Path(manifest["file"])
        if pq.read_schema(parquet) != expected or sha256(parquet) != manifest["file_sha256"]:
            raise RuntimeError(f"invalid partition: {parquet}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("product", choices=PRODUCTS)
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.product), sort_keys=True))


if __name__ == "__main__":
    main()
