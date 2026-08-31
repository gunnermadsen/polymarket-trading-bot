"""Read-only, checkpointed optional-data attachment for the middle tournament."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .core_extract import file_sha256
from .multivenue_early_entry_data import KEY_COLUMNS, TournamentDataConfig, build_panel
from .spot_l2_chainlink_features import L2_FEATURES, join_qualified_l2
from .twap60_training_data import _isolated_query_frame

SCHEMA_VERSION = "btc-middle-strategy-data-v1"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def middle_cache(config: TournamentDataConfig) -> Path:
    return config.package_root / config.raw["paths"]["middle_cache"]


def extract_spot_l2(config: TournamentDataConfig, *, force: bool = False) -> dict[str, Any]:
    """Extract only the existing qualified L2 states needed at decision timestamps."""

    cache = middle_cache(config)
    partitions = cache / "spot-l2"
    partitions.mkdir(parents=True, exist_ok=True)
    query_path = config.package_root / config.raw["paths"]["spot_l2_source_sql"]
    contract = {
        "schema_version": SCHEMA_VERSION,
        "range_start": config.source_start.isoformat(),
        "range_end": config.sealed_end.isoformat(),
        "query_sha256": file_sha256(query_path),
        "source_relation": config.raw["sources"]["binance_spot_l2_relation"],
        "read_only": True,
        "database_mutations": False,
        "new_table": False,
        "maximum_causal_age_seconds": 2,
    }
    manifest_path = cache / "spot-l2-manifest.json"
    if manifest_path.is_file() and not force:
        payload = json.loads(manifest_path.read_text())
        if payload["contract"] != contract:
            raise RuntimeError("spot-L2 extraction contract changed")
        for row in payload["partitions"]:
            path = cache / row["path"]
            if not path.is_file() or file_sha256(path) != row["sha256"]:
                raise RuntimeError(f"spot-L2 checkpoint changed: {path}")
        return payload

    records: list[dict[str, Any]] = []
    cursor = config.source_start
    while cursor < config.sealed_end:
        end = min(cursor + timedelta(days=1), config.sealed_end)
        destination = partitions / f"{cursor.date().isoformat()}.parquet"
        if force or not destination.is_file():
            frame = _isolated_query_frame(
                query_path.read_text(),
                {"batch_start": cursor, "batch_end": end},
                cursor_name=f"middle_strategy_l2_{cursor:%Y%m%d}",
            )
            frame.write_parquet(destination, compression="zstd", statistics=True)
            print(f"middle data: L2 {cursor.date()} {frame.height:,} rows", flush=True)
        record = {
            "path": str(destination.relative_to(cache)),
            "rows": pl.scan_parquet(destination).select(pl.len()).collect().item(),
            "sha256": file_sha256(destination),
        }
        records.append(record)
        _write_json(
            cache / "spot-l2-manifest.partial.json",
            {"contract": contract, "partitions": records},
        )
        cursor = end
    payload = {
        "contract": contract,
        "partitions": records,
        "rows": sum(int(row["rows"]) for row in records),
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_json(manifest_path, payload)
    (cache / "spot-l2-manifest.partial.json").unlink(missing_ok=True)
    return payload


def build_middle_panel(
    config: TournamentDataConfig, *, force: bool = False
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Preserve the complete base panel and causally attach optional spot L2."""

    cache = middle_cache(config)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "middle-panel.parquet"
    manifest_path = cache / "panel-manifest.json"
    if destination.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != file_sha256(destination):
            raise RuntimeError("middle panel changed after checkpoint")
        return pl.read_parquet(destination), manifest

    base, base_manifest = build_panel(config, force=False)
    l2_manifest = extract_spot_l2(config, force=force)
    records = [row for row in l2_manifest["partitions"] if row["rows"]]
    if records:
        pieces = []
        for row in records:
            path = cache / row["path"]
            day = datetime.fromisoformat(path.stem).replace(tzinfo=UTC)
            core = base.filter(
                pl.col("observed_at").is_between(day, day + timedelta(days=1), closed="left")
            ).select(*KEY_COLUMNS, "btc_close")
            if core.is_empty():
                continue
            source = pl.read_parquet(path).unique(subset=["second_start"], keep="last").sort(
                "available_at"
            )
            pieces.append(
                join_qualified_l2(core, source).select(*KEY_COLUMNS, *L2_FEATURES)
            )
        qualified = pl.concat(pieces, how="vertical_relaxed", rechunk=True)
        panel = base.join(qualified, on=list(KEY_COLUMNS), how="left", validate="1:1")
    else:
        panel = base.with_columns(
            *(pl.lit(None, dtype=pl.Float64).alias(name) for name in L2_FEATURES)
        )
    panel = panel.with_columns(
        pl.any_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite() for name in L2_FEATURES
        ).alias("has_spot_l2")
    )
    if panel.height != base.height or panel["market_id"].n_unique() != base["market_id"].n_unique():
        raise RuntimeError("optional L2 attachment changed base market coverage")
    panel.write_parquet(destination, compression="zstd", statistics=True)
    feature_groups = dict(base_manifest["feature_groups"])
    feature_groups["spot_l2"] = list(L2_FEATURES)
    execution = [
        name
        for name in panel.columns
        if name.startswith(("up_ask_vwap_", "down_ask_vwap_", "pm_"))
    ]
    feature_groups["execution"] = execution
    coverage = dict(base_manifest["coverage"])
    coverage["spot_l2"] = {
        "rows": panel.filter("has_spot_l2").height,
        "markets": panel.filter("has_spot_l2")["market_id"].n_unique(),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rows": panel.height,
        "markets": panel["market_id"].n_unique(),
        "range_start": config.source_start.isoformat(),
        "range_end": config.sealed_end.isoformat(),
        "feature_groups": feature_groups,
        "coverage": coverage,
        "base_panel": base_manifest,
        "spot_l2_source": l2_manifest,
        "optional_missingness_preserves_rows": True,
        "kraken_l2_included": False,
        "twap_inference_feature": False,
        "authentic_only_filter": False,
        "database_mutations": False,
        "new_tables": False,
        "new_ingesters": False,
        "new_sources": False,
        "sha256": file_sha256(destination),
    }
    _write_json(manifest_path, manifest)
    return panel, manifest
