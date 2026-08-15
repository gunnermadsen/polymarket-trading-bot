"""Exact-size retraining of frozen BTC model lineages at VWAP 10/15/20."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .asymmetric_incumbent_replay import (
    FROZEN_ASYMMETRIC_INCUMBENT_POLICY,
    load_frozen_asymmetric_incumbent,
    score_asymmetric_runtime_row,
)
from .capacity_training_config import CapacityLineage, CapacityTrainingConfig
from .champion_vwap_benchmark import fit_conditioned_calibrator
from .core_extract import (
    configure_read_only_connection,
    database_connection,
    file_sha256,
    write_json_atomic,
)
from .runtime_export import score_runtime_model

SCHEMA_VERSION = "btc-vwap-capacity-training-v1"
EVIDENCE_SCHEMA_VERSION = "btc-vwap-capacity-evidence-v1"

EVIDENCE_SCHEMA = pa.schema(
    [
        pa.field("market_id", pa.string(), nullable=False),
        pa.field("window_start", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("window_end", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("label_up", pa.int8(), nullable=False),
        pa.field("fee_rate", pa.float64(), nullable=False),
        pa.field("observed_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("seconds_elapsed", pa.int32(), nullable=False),
        pa.field("artifact_id", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("up_provider_received_at", pa.timestamp("us", tz="UTC")),
        pa.field("up_best_ask", pa.float64()),
        pa.field("up_ask_depth", pa.float64()),
        pa.field("up_ask_vwap_5", pa.float64()),
        pa.field("up_ask_vwap_10", pa.float64()),
        pa.field("up_ask_vwap_15", pa.float64()),
        pa.field("up_ask_vwap_20", pa.float64()),
        pa.field("down_provider_received_at", pa.timestamp("us", tz="UTC")),
        pa.field("down_best_ask", pa.float64()),
        pa.field("down_ask_depth", pa.float64()),
        pa.field("down_ask_vwap_5", pa.float64()),
        pa.field("down_ask_vwap_10", pa.float64()),
        pa.field("down_ask_vwap_15", pa.float64()),
        pa.field("down_ask_vwap_20", pa.float64()),
        pa.field("quality_flags", pa.int32(), nullable=False),
        pa.field("strict_both_side_eligible_10", pa.bool_(), nullable=False),
        pa.field("strict_both_side_eligible_15", pa.bool_(), nullable=False),
        pa.field("strict_both_side_eligible_20", pa.bool_(), nullable=False),
    ]
)


def extract_capacity_evidence(
    config: CapacityTrainingConfig, *, force: bool = False
) -> dict[str, Any]:
    """Stream immutable, completed PMXT capacity facts into hashed daily Parquet."""

    query_path = config.package_root / "sql" / "btc-capacity-execution-evidence.sql"
    query = query_path.read_text()
    query_sha256 = file_sha256(query_path)
    config.evidence.mkdir(parents=True, exist_ok=True)
    manifest_path = config.evidence / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    contract = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "range_start": config.windows.development_start.isoformat(),
        "range_end": config.windows.freeze_at.isoformat(),
        "freshness_seconds": config.execution.freshness_seconds,
        "maximum_depth_participation": config.execution.maximum_depth_participation,
        "query_sha256": query_sha256,
    }
    if existing and any(existing.get(key) != value for key, value in contract.items()):
        raise RuntimeError("capacity evidence contract changed; use a new evidence path")
    old = {item["path"]: item for item in (existing or {}).get("partitions", [])}
    partitions: list[dict[str, Any]] = []
    connection = database_connection()
    configure_read_only_connection(connection)
    try:
        _require_complete_capacity_coverage(connection, config)
        start = config.windows.development_start
        while start < config.windows.freeze_at:
            end = min(start + timedelta(days=1), config.windows.freeze_at)
            destination = config.evidence / f"{start.date().isoformat()}.parquet"
            expected = old.get(destination.name)
            if destination.exists() and not force:
                sha256 = file_sha256(destination)
                rows = pq.ParquetFile(destination).metadata.num_rows
                if expected and (expected["sha256"] != sha256 or expected["rows"] != rows):
                    raise RuntimeError(f"capacity partition changed: {destination.name}")
            else:
                rows = _extract_partition(
                    connection,
                    query,
                    destination,
                    batch_start=start,
                    batch_end=end,
                    freshness_seconds=config.execution.freshness_seconds,
                )
                sha256 = file_sha256(destination)
            partitions.append(
                {"path": destination.name, "rows": rows, "sha256": sha256}
            )
            start = end
    finally:
        connection.close()
    manifest = {
        **contract,
        "created_at": datetime.now(UTC).isoformat(),
        "source_table": "polymarket.btc_market_capacity_execution_snapshots",
        "source_provider": "pmxt_v2_capacity_execution_snapshots",
        "immutable_completed_artifacts_only": True,
        "partitions": partitions,
        "rows": sum(item["rows"] for item in partitions),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def _require_complete_capacity_coverage(
    connection: Any, config: CapacityTrainingConfig
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            WITH expected AS MATERIALIZED (
              SELECT hour
              FROM generate_series(
                %(range_start)s::timestamptz,
                %(range_end)s::timestamptz - interval '1 hour',
                interval '1 hour'
              ) AS hour
            ), completed AS MATERIALIZED (
              SELECT minimum_source_timestamp AS hour
              FROM polymarket.backfill_artifacts
              WHERE provider = 'pmxt_v2_capacity_execution_snapshots'
                AND status = 'completed'
                AND minimum_source_timestamp >= %(range_start)s
                AND minimum_source_timestamp < %(range_end)s
                AND record_count = 1152
            )
            SELECT count(*)::bigint, min(expected.hour)
            FROM expected
            LEFT JOIN completed USING (hour)
            WHERE completed.hour IS NULL
            """,
            {
                "range_start": config.windows.development_start,
                "range_end": config.windows.freeze_at,
            },
        )
        missing, first_missing = cursor.fetchone()
    if missing:
        raise RuntimeError(
            "capacity evidence is not materialized for "
            f"{missing} hourly partition(s); first missing hour: {first_missing.isoformat()}"
        )


def run_capacity_training(
    config: CapacityTrainingConfig, *, force: bool = False
) -> tuple[Path, dict[str, Any]]:
    """Extract exact execution evidence and fit 12 size-aware child calibrators."""

    evidence_manifest = extract_capacity_evidence(config, force=force)
    if evidence_manifest["rows"] == 0:
        raise RuntimeError("capacity evidence is empty; complete the PMXT backfill first")
    evidence = _load_evidence(config, evidence_manifest)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    for lineage in config.lineages:
        missing_features = [path for path in lineage.features if not path.is_file()]
        if missing_features:
            raise RuntimeError(
                f"{lineage.name} feature cache is missing: {missing_features[0]}"
            )
        model, identity = _load_lineage(lineage)
        feature_names = tuple(model["features"]["names"])
        feature_schema = pl.scan_parquet(lineage.features).collect_schema()
        selected_feature_names = tuple(
            name for name in feature_names if name in feature_schema
        )
        missing_model_features = set(feature_names) - set(selected_feature_names)
        if lineage.hypothesis == "directional" and missing_model_features:
            raise RuntimeError(
                f"{lineage.name} feature cache does not satisfy its runtime schema"
            )
        features = (
            pl.read_parquet(lineage.features)
            .filter(
                (pl.col("window_start") >= config.windows.development_start)
                & (pl.col("window_start") < config.windows.freeze_at)
            )
            .select(
                "market_id",
                "window_start",
                "observed_at",
                "seconds_elapsed",
                "label_up",
                *selected_feature_names,
            )
            .sort(["market_id", "seconds_elapsed", "observed_at"])
        )
        for quantity in config.execution.quantities:
            if lineage.hypothesis == "directional":
                result, artifact = _train_directional(
                    config, lineage, model, identity, features, evidence, quantity
                )
            else:
                result, artifact = _train_asymmetric(
                    config, lineage, model, identity, features, evidence, quantity
                )
            artifact_path = run_dir / f"{lineage.name}-vwap-{quantity}.json"
            write_json_atomic(artifact_path, artifact)
            result["artifact"] = {
                "path": str(artifact_path),
                "sha256": file_sha256(artifact_path),
            }
            results.append(result)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "forward_holdout_pending",
        "production_qualified": False,
        "runtime_exported": False,
        "trading_processes_changed": False,
        "configuration_sha256": file_sha256(config.source_path),
        "evidence_manifest_sha256": file_sha256(config.evidence / "manifest.json"),
        "windows": {key: value.isoformat() for key, value in asdict(config.windows).items()},
        "execution": asdict(config.execution),
        "results": results,
        "required_next_evidence": "independent chronological observations after freeze_at",
    }
    write_json_atomic(run_dir / "training.json", payload)
    (run_dir / "training-report.md").write_text(_report(payload))
    return run_dir, payload


def _extract_partition(
    connection: Any,
    query: str,
    destination: Path,
    *,
    batch_start: datetime,
    batch_end: datetime,
    freshness_seconds: int,
) -> int:
    temporary = destination.with_suffix(".parquet.partial")
    writer: pq.ParquetWriter | None = None
    count = 0
    try:
        with connection.transaction(), connection.cursor(
            name=f"btc_capacity_{batch_start:%Y%m%d}"
        ) as cursor:
            cursor.execute(
                query,
                {
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "freshness_seconds": freshness_seconds,
                },
            )
            while rows := cursor.fetchmany(10_000):
                records = [dict(zip(EVIDENCE_SCHEMA.names, row, strict=True)) for row in rows]
                table = pa.Table.from_pylist(records, schema=EVIDENCE_SCHEMA)
                writer = writer or pq.ParquetWriter(
                    temporary, EVIDENCE_SCHEMA, compression="zstd"
                )
                writer.write_table(table)
                count += len(rows)
    finally:
        if writer:
            writer.close()
    if count == 0:
        pq.write_table(pa.Table.from_pylist([], schema=EVIDENCE_SCHEMA), temporary)
    temporary.replace(destination)
    return count


def _load_evidence(
    config: CapacityTrainingConfig, manifest: dict[str, Any]
) -> pl.DataFrame:
    paths = []
    for item in manifest["partitions"]:
        path = config.evidence / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise RuntimeError(f"capacity evidence hash mismatch: {path.name}")
        paths.append(path)
    frame = pl.read_parquet(paths)
    duplicates = frame.group_by("market_id", "observed_at").len().filter(pl.col("len") != 1)
    if duplicates.height:
        raise RuntimeError("capacity evidence contains duplicate decision points")
    return frame


def _load_lineage(lineage: CapacityLineage) -> tuple[dict[str, Any], dict[str, Any]]:
    model = json.loads(lineage.model.read_text())
    manifest = json.loads(lineage.manifest.read_text())
    model_hash = file_sha256(lineage.model)
    if manifest.get("model_key") != model.get("model_key"):
        raise RuntimeError(f"{lineage.name} model identity mismatch")
    if manifest.get("model_sha256") != model_hash:
        raise RuntimeError(f"{lineage.name} model hash mismatch")
    if manifest.get("feature_schema_sha256") != model["features"]["schema_sha256"]:
        raise RuntimeError(f"{lineage.name} feature schema mismatch")
    return model, {
        "model_key": model["model_key"],
        "model_sha256": model_hash,
        "manifest_sha256": file_sha256(lineage.manifest),
        "feature_schema_sha256": model["features"]["schema_sha256"],
        "feature_files": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in lineage.features
        ],
        "source_process_id": str(lineage.process_id),
    }


def _train_directional(
    config: CapacityTrainingConfig,
    lineage: CapacityLineage,
    model: dict[str, Any],
    identity: dict[str, Any],
    features: pl.DataFrame,
    evidence: pl.DataFrame,
    quantity: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    decisions = _directional_first_crossings(features, model)
    frame = _join_execution(decisions, evidence, quantity)
    training = _range(frame, config.windows.development_start, config.windows.policy_start)
    policy = _range(frame, config.windows.policy_start, config.windows.freeze_at)
    if training.height < config.gates.minimum_training_rows:
        raise RuntimeError(f"{lineage.name} VWAP{quantity} training rows below gate")
    calibrator = fit_conditioned_calibrator(
        training,
        candidate=f"vwap_{quantity}_conditioned",
        feature_names=(
            "selected_raw_logit",
            "selected_ask_vwap",
            "vwap_size_minus_vwap5",
        ),
        c=1.0,
        maximum_iterations=500,
    )
    candidate = policy.with_columns(
        pl.Series("selection_probability", calibrator.probability(policy)),
    ).with_columns(
        (pl.col("selection_probability") >= config.execution.confidence_threshold).alias(
            "policy_selected"
        )
    )
    control = policy.with_columns(pl.lit(True).alias("policy_selected"))
    folds = _directional_folds(config, frame, quantity)
    result = _result(
        config, lineage, identity, quantity, training, control, candidate, folds
    )
    artifact = {
        "schema_version": "btc-vwap-capacity-calibrator-v1",
        **identity,
        "hypothesis": lineage.hypothesis,
        "quantity": quantity,
        "execution_price": f"exact_vwap_{quantity}",
        "calibrator": asdict(calibrator),
        "direction_and_timestamp_locked_to_parent": True,
        "runtime_exported": False,
        "production_qualified": False,
    }
    return result, artifact


def _train_asymmetric(
    config: CapacityTrainingConfig,
    lineage: CapacityLineage,
    model_payload: dict[str, Any],
    identity: dict[str, Any],
    features: pl.DataFrame,
    evidence: pl.DataFrame,
    quantity: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = load_frozen_asymmetric_incumbent(lineage.model)
    frame = _join_asymmetric_rows(
        features,
        evidence,
        frozen,
        quantity,
        reserve=config.execution.execution_reserve_per_share,
    )
    training = _range(frame, config.windows.development_start, config.windows.policy_start)
    policy = _range(frame, config.windows.policy_start, config.windows.freeze_at)
    if training.height < config.gates.minimum_training_rows:
        raise RuntimeError(f"{lineage.name} VWAP{quantity} training rows below gate")
    fit_frame = training.with_columns(pl.col("label_up").cast(pl.Boolean).alias("correct"))
    calibrator = fit_conditioned_calibrator(
        fit_frame,
        candidate=f"asymmetric_vwap_{quantity}",
        feature_names=(
            "raw_logit",
            "up_ask_vwap",
            "down_ask_vwap",
            "up_vwap_size_minus_vwap5",
            "down_vwap_size_minus_vwap5",
        ),
        c=1.0,
        maximum_iterations=500,
    )
    probability = calibrator.probability(policy)
    candidate = _select_asymmetric(policy, probability, quantity)
    control = _select_asymmetric(policy, policy["parent_probability_up"].to_numpy(), quantity)
    folds = _asymmetric_folds(config, frame, quantity)
    result = _result(
        config, lineage, identity, quantity, training, control, candidate, folds
    )
    artifact = {
        "schema_version": "btc-vwap-capacity-calibrator-v1",
        **identity,
        "hypothesis": lineage.hypothesis,
        "quantity": quantity,
        "execution_price": f"exact_vwap_{quantity}",
        "calibrator": asdict(calibrator),
        "frozen_value_policy": asdict(FROZEN_ASYMMETRIC_INCUMBENT_POLICY),
        "hypothesis_locked_to_parent": True,
        "runtime_exported": False,
        "production_qualified": False,
    }
    return result, artifact


def _directional_first_crossings(frame: pl.DataFrame, model: dict[str, Any]) -> pl.DataFrame:
    names = tuple(model["features"]["names"])
    records = []
    for market in frame.partition_by("market_id", maintain_order=True):
        for row in market.iter_rows(named=True):
            prediction = score_runtime_model(
                model,
                [row[name] for name in names],
                seconds_elapsed=int(row["seconds_elapsed"]),
            )
            if prediction["action"] == "no_trade":
                continue
            up = prediction["action"] == "up"
            raw_logit = float(prediction["raw_logit"])
            records.append(
                {
                    "market_id": row["market_id"],
                    "window_start": row["window_start"],
                    "observed_at": row["observed_at"],
                    "seconds_elapsed": row["seconds_elapsed"],
                    "label_up": row["label_up"],
                    "selected_up": up,
                    "selected_raw_logit": raw_logit if up else -raw_logit,
                }
            )
            break
    return pl.DataFrame(records)


def _join_execution(
    decisions: pl.DataFrame, evidence: pl.DataFrame, quantity: int
) -> pl.DataFrame:
    eligible = f"strict_both_side_eligible_{quantity}"
    joined = decisions.join(
        evidence.filter(pl.col(eligible)),
        on=["market_id", "window_start", "observed_at", "seconds_elapsed", "label_up"],
        how="inner",
        validate="1:1",
    )
    return joined.with_columns(
        pl.when(pl.col("selected_up"))
        .then(pl.col(f"up_ask_vwap_{quantity}"))
        .otherwise(pl.col(f"down_ask_vwap_{quantity}"))
        .alias("selected_ask_vwap"),
        pl.when(pl.col("selected_up"))
        .then(pl.col("up_ask_vwap_5"))
        .otherwise(pl.col("down_ask_vwap_5"))
        .alias("selected_ask_vwap_5"),
        (pl.col("selected_up") == pl.col("label_up").cast(pl.Boolean)).alias("correct"),
    ).with_columns(
        (pl.col("selected_ask_vwap") - pl.col("selected_ask_vwap_5")).alias(
            "vwap_size_minus_vwap5"
        )
    )


def _join_asymmetric_rows(
    features: pl.DataFrame,
    evidence: pl.DataFrame,
    frozen: Any,
    quantity: int,
    *,
    reserve: float,
) -> pl.DataFrame:
    eligible = f"strict_both_side_eligible_{quantity}"
    joined = features.join(
        evidence.filter(pl.col(eligible)),
        on=["market_id", "window_start", "observed_at", "seconds_elapsed", "label_up"],
        how="inner",
        validate="1:1",
    ).filter(pl.col("seconds_elapsed") <= FROZEN_ASYMMETRIC_INCUMBENT_POLICY.maximum_entry_second)
    joined = _attach_asymmetric_book_features(joined, quantity, reserve)
    missing = sorted(set(frozen.feature_names) - set(joined.columns))
    if missing:
        raise RuntimeError(
            "asymmetric capacity feature frame is missing: " + ", ".join(missing)
        )
    predictions = []
    raw_logits = []
    matrix = joined.select(*frozen.feature_names).to_numpy()
    for row, second, up_price, down_price in zip(
        matrix,
        joined["seconds_elapsed"],
        joined[f"up_ask_vwap_{quantity}"],
        joined[f"down_ask_vwap_{quantity}"],
        strict=True,
    ):
        scored = score_asymmetric_runtime_row(
            frozen,
            row.tolist(),
            seconds_elapsed=int(second),
            yes_ask_vwap=float(up_price),
            no_ask_vwap=float(down_price),
        )
        predictions.append(scored["probability_up"])
        raw_logits.append(scored["raw_logit"])
    return joined.with_columns(
        pl.Series("parent_probability_up", predictions),
        pl.Series("raw_logit", raw_logits),
        pl.col(f"up_ask_vwap_{quantity}").alias("up_ask_vwap"),
        pl.col(f"down_ask_vwap_{quantity}").alias("down_ask_vwap"),
    ).with_columns(
        (pl.col("up_ask_vwap") - pl.col("up_ask_vwap_5")).alias(
            "up_vwap_size_minus_vwap5"
        ),
        (pl.col("down_ask_vwap") - pl.col("down_ask_vwap_5")).alias(
            "down_vwap_size_minus_vwap5"
        ),
    )


def _attach_asymmetric_book_features(
    frame: pl.DataFrame, quantity: int, reserve: float
) -> pl.DataFrame:
    up_vwap = f"up_ask_vwap_{quantity}"
    down_vwap = f"down_ask_vwap_{quantity}"
    epsilon = 1e-6
    enriched = frame.with_columns(
        (
            pl.col(up_vwap)
            + pl.col("fee_rate") * pl.col(up_vwap) * (1 - pl.col(up_vwap))
            + reserve
        ).alias("pm_yes_cost_per_share"),
        (
            pl.col(down_vwap)
            + pl.col("fee_rate") * pl.col(down_vwap) * (1 - pl.col(down_vwap))
            + reserve
        ).alias("pm_no_cost_per_share"),
    ).with_columns(
        (
            pl.col("pm_yes_cost_per_share").clip(epsilon, 1 - epsilon).log()
            - (1 - pl.col("pm_yes_cost_per_share").clip(epsilon, 1 - epsilon)).log()
        ).alias("pm_yes_cost_logit"),
        (
            pl.col("pm_no_cost_per_share").clip(epsilon, 1 - epsilon).log()
            - (1 - pl.col("pm_no_cost_per_share").clip(epsilon, 1 - epsilon)).log()
        ).alias("pm_no_cost_logit"),
        (pl.col("pm_yes_cost_per_share") + pl.col("pm_no_cost_per_share") - 1).alias(
            "pm_cost_overround"
        ),
        (pl.col("pm_yes_cost_per_share") - pl.col("pm_no_cost_per_share")).alias(
            "pm_yes_minus_no_cost"
        ),
        (pl.col(up_vwap) - pl.col("up_best_ask")).alias("pm_yes_vwap_slippage"),
        (pl.col(down_vwap) - pl.col("down_best_ask")).alias("pm_no_vwap_slippage"),
        pl.col("up_ask_depth").log1p().alias("pm_yes_depth_log"),
        pl.col("down_ask_depth").log1p().alias("pm_no_depth_log"),
        (
            (pl.col("up_ask_depth") - pl.col("down_ask_depth"))
            / (pl.col("up_ask_depth") + pl.col("down_ask_depth")).clip(1e-9)
        ).alias("pm_depth_imbalance"),
        (
            (pl.col("observed_at") - pl.col("up_provider_received_at"))
            .dt.total_milliseconds()
            / 1_000
        ).alias("pm_yes_book_age_seconds"),
        (
            (pl.col("observed_at") - pl.col("down_provider_received_at"))
            .dt.total_milliseconds()
            / 1_000
        ).alias("pm_no_book_age_seconds"),
    )
    return enriched


def _select_asymmetric(
    frame: pl.DataFrame, probability_up: np.ndarray, quantity: int
) -> pl.DataFrame:
    policy = FROZEN_ASYMMETRIC_INCUMBENT_POLICY
    candidates = frame.with_columns(pl.Series("probability_up", probability_up)).with_columns(
        (pl.col("probability_up") - pl.col("up_ask_vwap")).alias("up_edge"),
        ((1.0 - pl.col("probability_up")) - pl.col("down_ask_vwap")).alias("down_edge"),
    ).with_columns(
        (pl.col("up_edge") >= pl.col("down_edge")).alias("selected_up"),
    ).with_columns(
        pl.when(pl.col("selected_up")).then(pl.col("up_edge")).otherwise(pl.col("down_edge")).alias("selected_edge"),
        pl.when(pl.col("selected_up")).then(pl.col("up_ask_vwap")).otherwise(pl.col("down_ask_vwap")).alias("selected_ask_vwap"),
        (pl.col("selected_up") == pl.col("label_up").cast(pl.Boolean)).alias("correct"),
    ).filter(
        pl.col("selected_ask_vwap").is_between(
            policy.minimum_share_price, policy.maximum_share_price, closed="both"
        )
        & (pl.col("selected_ask_vwap") <= policy.maximum_cost_per_share)
        & (pl.col("selected_edge") >= policy.minimum_edge_per_share)
    ).sort(["market_id", "seconds_elapsed", "observed_at"])
    selected = candidates.group_by("market_id", maintain_order=True).first()
    return selected.with_columns(pl.lit(True).alias("policy_selected"))


def _result(
    config: CapacityTrainingConfig,
    lineage: CapacityLineage,
    identity: dict[str, Any],
    quantity: int,
    training: pl.DataFrame,
    control: pl.DataFrame,
    candidate: pl.DataFrame,
    folds: list[dict[str, Any]],
) -> dict[str, Any]:
    control_metrics = _metrics(control, quantity, config.execution.execution_reserve_per_share)
    candidate_metrics = _metrics(candidate, quantity, config.execution.execution_reserve_per_share)
    checks = {
        "minimum_policy_trades": candidate_metrics["trades"] >= config.gates.minimum_policy_trades,
        "positive_net_expectancy": candidate_metrics["expectancy_per_trade"] > 0,
        "profit_factor": candidate_metrics["profit_factor"] >= config.gates.minimum_profit_factor,
        "stress_expectancy": candidate_metrics["stress_plus_one_cent"]["expectancy_per_trade"]
        >= config.gates.minimum_stress_expectancy_per_trade,
        "net_pnl_improves_control": candidate_metrics["net_pnl"] > control_metrics["net_pnl"],
        "multiple_chronological_folds_improve": sum(
            fold["candidate_improves_control"] for fold in folds
        )
        >= config.gates.minimum_improving_folds,
    }
    return {
        "lineage": lineage.name,
        "hypothesis": lineage.hypothesis,
        "source_process_id": str(lineage.process_id),
        "model_key": identity["model_key"],
        "quantity": quantity,
        "training_rows": training.height,
        "policy_window_rows": control.height,
        "control": control_metrics,
        "candidate": candidate_metrics,
        "development_gates": checks,
        "chronological_folds": folds,
        "development_qualified": all(checks.values()),
        "production_qualified": False,
        "status": "forward_holdout_pending",
    }


def _directional_folds(
    config: CapacityTrainingConfig, frame: pl.DataFrame, quantity: int
) -> list[dict[str, Any]]:
    output = []
    for index, start, end in _fold_windows(config):
        fit = _range(frame, config.windows.development_start, start)
        evaluation = _range(frame, start, end)
        if fit.height < config.gates.minimum_training_rows or evaluation.is_empty():
            raise RuntimeError(f"VWAP{quantity} chronological fold {index} lacks evidence")
        calibrator = fit_conditioned_calibrator(
            fit,
            candidate=f"vwap_{quantity}_fold_{index}",
            feature_names=(
                "selected_raw_logit",
                "selected_ask_vwap",
                "vwap_size_minus_vwap5",
            ),
            c=1.0,
            maximum_iterations=500,
        )
        candidate = evaluation.with_columns(
            pl.Series("selection_probability", calibrator.probability(evaluation))
        ).with_columns(
            (pl.col("selection_probability") >= config.execution.confidence_threshold).alias(
                "policy_selected"
            )
        )
        control = evaluation.with_columns(pl.lit(True).alias("policy_selected"))
        output.append(_fold_result(index, start, end, fit, control, candidate, quantity, config))
    return output


def _asymmetric_folds(
    config: CapacityTrainingConfig, frame: pl.DataFrame, quantity: int
) -> list[dict[str, Any]]:
    output = []
    for index, start, end in _fold_windows(config):
        fit = _range(frame, config.windows.development_start, start)
        evaluation = _range(frame, start, end)
        if fit.height < config.gates.minimum_training_rows or evaluation.is_empty():
            raise RuntimeError(f"VWAP{quantity} chronological fold {index} lacks evidence")
        fit_frame = fit.with_columns(pl.col("label_up").cast(pl.Boolean).alias("correct"))
        calibrator = fit_conditioned_calibrator(
            fit_frame,
            candidate=f"asymmetric_vwap_{quantity}_fold_{index}",
            feature_names=(
                "raw_logit",
                "up_ask_vwap",
                "down_ask_vwap",
                "up_vwap_size_minus_vwap5",
                "down_vwap_size_minus_vwap5",
            ),
            c=1.0,
            maximum_iterations=500,
        )
        candidate = _select_asymmetric(
            evaluation, calibrator.probability(evaluation), quantity
        )
        control = _select_asymmetric(
            evaluation, evaluation["parent_probability_up"].to_numpy(), quantity
        )
        output.append(_fold_result(index, start, end, fit, control, candidate, quantity, config))
    return output


def _fold_windows(
    config: CapacityTrainingConfig,
) -> list[tuple[int, datetime, datetime]]:
    span = config.windows.policy_start - config.windows.calibration_start
    return [
        (
            index + 1,
            config.windows.calibration_start + span * index / 5,
            config.windows.calibration_start + span * (index + 1) / 5,
        )
        for index in range(5)
    ]


def _fold_result(
    index: int,
    start: datetime,
    end: datetime,
    fit: pl.DataFrame,
    control: pl.DataFrame,
    candidate: pl.DataFrame,
    quantity: int,
    config: CapacityTrainingConfig,
) -> dict[str, Any]:
    control_metrics = _metrics(
        control, quantity, config.execution.execution_reserve_per_share
    )
    candidate_metrics = _metrics(
        candidate, quantity, config.execution.execution_reserve_per_share
    )
    return {
        "fold": index,
        "fit_rows": fit.height,
        "evaluation_start": start.isoformat(),
        "evaluation_end": end.isoformat(),
        "control": control_metrics,
        "candidate": candidate_metrics,
        "candidate_improves_control": candidate_metrics["net_pnl"]
        > control_metrics["net_pnl"],
    }


def _metrics(frame: pl.DataFrame, quantity: int, reserve: float) -> dict[str, Any]:
    selected = frame.filter(pl.col("policy_selected"))
    if selected.is_empty():
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "expectancy_per_trade": 0.0,
            "profit_factor": 0.0,
            "maximum_drawdown": 0.0,
            "worst_one_percent_mean": 0.0,
            "stress_plus_one_cent": {"net_pnl": 0.0, "expectancy_per_trade": 0.0},
        }
    pnl = (
        selected["correct"].cast(pl.Float64)
        - selected["selected_ask_vwap"]
        - selected["fee_rate"] * selected["selected_ask_vwap"] * (1 - selected["selected_ask_vwap"])
        - reserve
    ) * quantity
    stress = pnl - 0.01 * quantity
    values = pnl.to_numpy()
    cumulative = np.cumsum(values)
    drawdown = np.maximum.accumulate(np.concatenate(([0.0], cumulative))) - np.concatenate(([0.0], cumulative))
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    tail_count = max(1, math.ceil(len(values) * 0.01))
    return {
        "trades": len(values),
        "net_pnl": float(values.sum()),
        "expectancy_per_trade": float(values.mean()),
        "profit_factor": gains / losses if losses else (math.inf if gains else 0.0),
        "maximum_drawdown": float(drawdown.max()),
        "worst_one_percent_mean": float(np.sort(values)[:tail_count].mean()),
        "stress_plus_one_cent": {
            "net_pnl": float(stress.sum()),
            "expectancy_per_trade": float(stress.mean()),
        },
    }


def _range(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    return frame.filter((pl.col("window_start") >= start) & (pl.col("window_start") < end))


def _report(payload: dict[str, Any]) -> str:
    lines = [
        "# BTC VWAP Capacity Training",
        "",
        f"Status: **{payload['status']}**",
        "",
        "No runtime model or trading process was changed. Results are development-only until independent post-freeze evidence exists.",
        "",
        "| Lineage | Shares | Candidate trades | Net PnL | Expectancy | PF | Development gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in payload["results"]:
        metric = result["candidate"]
        lines.append(
            f"| {result['lineage']} | {result['quantity']} | {metric['trades']} | "
            f"{metric['net_pnl']:.4f} | {metric['expectancy_per_trade']:.4f} | "
            f"{metric['profit_factor']:.3f} | {result['development_qualified']} |"
        )
    return "\n".join(lines) + "\n"
