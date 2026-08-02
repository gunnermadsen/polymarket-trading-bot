from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .chainlink_oi_benchmark import (
    _candidate_feature_sets,
    _cohort,
    _direction_metrics,
)
from .chainlink_oi_config import (
    CHAINLINK_FULL_CANDIDATE,
    CHAINLINK_FULL_OI_CANDIDATE,
    LONG_HISTORY_CANDLE_CANDIDATE,
    ChainlinkOiBenchmarkConfig,
)
from .chainlink_oi_features import (
    POINT_KEY_COLUMNS,
    ExternalSourceFrames,
    _query_frame,
    derive_chainlink_candle_feature_frame,
    derive_chainlink_oi_feature_frames,
    extract_external_source_frames,
)
from .chainlink_oi_paper_export import PAPER_EXPORT_SPECS
from .core_extract import (
    ORACLE_MAX_PUBLICATION_DELAY_SECONDS,
    configure_read_only_connection,
    database_connection,
    file_sha256,
    write_json_atomic,
)
from .core_features import (
    attach_causal_oracle_rounds,
    audit_final_prices,
    derive_core_point_in_time_features,
    derive_oracle_point_in_time_features,
    prepare_causal_oracle_rounds,
)
from .runtime_export import score_runtime_model

SCHEMA_VERSION = "btc-chainlink-oi-untouched-forward-score-v1"
MAX_FORWARD_RANGE_DAYS = 31
FORWARD_REPORT_FILENAME = "forward-score.json"


@dataclass(frozen=True)
class ForwardRuntimeModel:
    candidate: str
    directory: Path
    manifest: dict[str, Any]
    model: dict[str, Any]


@dataclass(frozen=True)
class ForwardCoreFrames:
    raw: pl.DataFrame
    complete: pl.DataFrame
    oracle_complete: pl.DataFrame
    final_price_mismatch_markets: int


def run_chainlink_oi_forward_score(
    config: ChainlinkOiBenchmarkConfig,
    *,
    range_start: datetime,
    range_end: datetime,
    output_root: Path,
    runtime_model_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Score immutable candidates on rows excluded from every benchmark role.

    The function performs bounded, read-only source queries and never fits or
    mutates a model. Missing source cohorts are an evidence result: a durable
    zero-row report is written rather than substituting or imputing values.
    """

    _validate_forward_range(config, range_start, range_end)
    runtime_root = (runtime_model_root or config.package_root / "runtime-models").resolve()
    models = _load_forward_runtime_models(config, runtime_root)

    with database_connection() as connection:
        configure_read_only_connection(connection)
        inventory = _query_forward_inventory(
            connection,
            config,
            range_start=range_start,
            range_end=range_end,
        )
        raw_core = _query_frame(
            connection,
            (config.package_root / "sql" / "btc-core-source.sql").read_text(),
            {
                "batch_start": range_start,
                "batch_end": range_end,
                "strict_final_price_audit": False,
            },
            cursor_name="btc_chainlink_oi_forward_core",
        )
        oracle_rounds = _query_frame(
            connection,
            (config.package_root / "sql" / "btc-core-oracle-source.sql").read_text(),
            {
                "batch_start": range_start,
                "batch_end": range_end,
                "oracle_feed_proxy_address": config.sources.polygon_oracle_proxy,
                "oracle_max_publication_delay_seconds": (ORACLE_MAX_PUBLICATION_DELAY_SECONDS),
            },
            cursor_name="btc_chainlink_oi_forward_oracle",
        )
        external = extract_external_source_frames(
            connection,
            config.package_root,
            range_start=range_start,
            range_end=range_end,
            refprice_feed_id=config.sources.refprice_feed_id,
            candle_symbol=config.sources.candle_symbol,
            open_interest_symbol=config.sources.open_interest_symbol,
        )

    result = build_chainlink_oi_forward_result(
        config,
        range_start=range_start,
        range_end=range_end,
        inventory=inventory,
        raw_core=raw_core,
        oracle_rounds=oracle_rounds,
        external=external,
        runtime_models=models,
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(run_dir / FORWARD_REPORT_FILENAME, result)
    (run_dir / "report.html").write_text(_report_html(result), encoding="utf-8")
    return run_dir, result


def build_chainlink_oi_forward_result(
    config: ChainlinkOiBenchmarkConfig,
    *,
    range_start: datetime,
    range_end: datetime,
    inventory: pl.DataFrame,
    raw_core: pl.DataFrame,
    oracle_rounds: pl.DataFrame,
    external: ExternalSourceFrames,
    runtime_models: dict[str, ForwardRuntimeModel],
) -> dict[str, Any]:
    """Build a deterministic result from already-extracted read-only frames."""

    _validate_forward_range(config, range_start, range_end)
    core = derive_forward_core_frames(raw_core, oracle_rounds, config)
    candidates = _derive_forward_candidate_frames(core.oracle_complete, external, config)
    model_metrics: dict[str, dict[str, Any]] = {}
    for candidate, model in runtime_models.items():
        frame = candidates[candidate]
        model_metrics[candidate] = score_forward_candidate(frame, model.model)

    inventory_payload = _inventory_payload(inventory)
    source_eligibility = {
        "inventory_by_utc_date": inventory_payload["days"],
        "inventory_totals": inventory_payload["totals"],
        "raw_core": _cohort(core.raw),
        "complete_core": _cohort(core.complete),
        "oracle_complete_core": _cohort(core.oracle_complete),
        "final_price_mismatch_markets_excluded": core.final_price_mismatch_markets,
        "raw_external_sources": {
            "polygon_oracle": _timestamped_source_summary(oracle_rounds, "oracle_block_timestamp"),
            "chainlink_refprice": _timestamped_source_summary(
                external.refprice, "source_timestamp"
            ),
            "chainlink_one_minute_candles": _timestamped_source_summary(
                external.candles, "close_timestamp"
            ),
            "binance_five_minute_open_interest": _timestamped_source_summary(
                external.open_interest, "source_timestamp"
            ),
        },
        "strict_model_cohorts": {
            candidate: _cohort(frame) for candidate, frame in candidates.items()
        },
        "strict_candidate_keys_identical": _candidate_keys_identical(candidates),
    }
    blockers = _forward_blockers(source_eligibility, model_metrics)
    models_with_rows = sum(metrics["eligible_markets"] > 0 for metrics in model_metrics.values())
    if models_with_rows == 0:
        status = "blocked_zero_rows"
    elif models_with_rows < len(model_metrics):
        status = "partially_scored"
    else:
        status = "scored_with_source_gaps" if blockers else "scored"

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "evidence_classification": (
            "untouched post-benchmark-window chronological evidence; excluded from fit, "
            "calibration, threshold selection, and model selection"
        ),
        "range": {
            "start": range_start.isoformat(),
            "end_exclusive": range_end.isoformat(),
            "days": int((range_end - range_start).total_seconds() // 86_400),
            "benchmark_window_end": config.windows.source_range_end.isoformat(),
        },
        "training_performed": False,
        "model_parameters_changed": False,
        "processes_changed": False,
        "database_mutated": False,
        "causal_contract": {
            "decision_clock": "observed_at",
            "binance_kline_available_by_decision": True,
            "oracle_source_and_block_at_or_before_decision": True,
            "refprice_strictly_before_decision": True,
            "refprice_max_age_seconds": config.staleness.refprice_seconds,
            "closed_candle_at_or_before_decision": True,
            "candle_max_age_seconds": config.staleness.candles_seconds,
            "open_interest_strictly_before_decision": True,
            "open_interest_max_age_seconds": config.staleness.open_interest_seconds,
            "imputation": False,
            "missingness_features": False,
            "short_candidate_rows_oi_matched": True,
        },
        "source_eligibility": source_eligibility,
        "models": model_metrics,
        "runtime_lineage": {
            candidate: {
                "model_key": runtime.model["model_key"],
                "directory": str(runtime.directory),
                "model_sha256": runtime.manifest["model_sha256"],
                "feature_schema_version": runtime.manifest["feature_schema_version"],
                "feature_schema_sha256": runtime.manifest["feature_schema_sha256"],
                "deployment_scope": runtime.manifest.get("deployment_scope"),
                "production_qualified": runtime.manifest.get("production_qualified"),
            }
            for candidate, runtime in runtime_models.items()
        },
        "execution_economics": {
            "status": "not_scored",
            "execution_snapshot_rows": source_eligibility["inventory_totals"][
                "execution_snapshot_rows"
            ],
            "reason": (
                "zero retained forward execution-book rows exist, so VWAP, fees, PnL, "
                "expectancy, and drawdown are unavailable"
            ),
        },
        "blockers": blockers,
    }


def derive_forward_core_frames(
    raw_core: pl.DataFrame,
    oracle_rounds: pl.DataFrame,
    config: ChainlinkOiBenchmarkConfig,
) -> ForwardCoreFrames:
    if raw_core.is_empty():
        return ForwardCoreFrames(raw_core, raw_core, raw_core, 0)
    ordered = raw_core.sort(["market_id", "seconds_elapsed"])
    base = derive_core_point_in_time_features(ordered)
    final_audit = audit_final_prices(base)
    mismatch_ids = final_audit.filter(
        pl.col("has_final_price") & ~pl.col("final_price_matches_official")
    )["market_id"]
    if len(mismatch_ids):
        base = base.filter(~pl.col("market_id").is_in(mismatch_ids.implode()))
    complete = _complete_candidate_markets(base, config)
    if oracle_rounds.is_empty():
        return ForwardCoreFrames(base, complete, complete.head(0), len(mismatch_ids))
    joined = attach_causal_oracle_rounds(base, prepare_causal_oracle_rounds(oracle_rounds))
    with_oracle = derive_oracle_point_in_time_features(joined)
    oracle_complete = _complete_candidate_markets(
        with_oracle,
        config,
        eligibility_column="oracle_model_eligible",
    )
    return ForwardCoreFrames(base, complete, oracle_complete, len(mismatch_ids))


def score_forward_candidate(frame: pl.DataFrame, model: dict[str, Any]) -> dict[str, Any]:
    feature_names = tuple(str(name) for name in model["features"]["names"])
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing and frame.height:
        raise RuntimeError("forward feature frame is missing: " + ", ".join(missing))
    if frame.is_empty():
        return _empty_forward_metrics()
    finite = frame.select(
        pl.all_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite() for name in feature_names
        ).all()
    ).item()
    if not finite:
        raise RuntimeError(
            "forward features contain missing/non-finite values; imputation is disabled"
        )

    matrix = frame.select(*feature_names).to_numpy()
    elapsed = frame["seconds_elapsed"].to_numpy()
    scored = [
        score_runtime_model(model, row, seconds_elapsed=int(second))
        for row, second in zip(matrix, elapsed, strict=True)
    ]
    score_frame = (
        frame.with_columns(
            pl.Series("probability_up", [row["probability_up"] for row in scored]),
            pl.Series("confidence", [row["confidence"] for row in scored]),
            pl.Series("action", [row["action"] for row in scored]),
        )
        .with_columns(
            (pl.col("action") == "up").alias("predicted_up"),
            (pl.col("action") != "no_trade").alias("policy_selected"),
        )
        .with_columns((pl.col("predicted_up").cast(pl.Int8) == pl.col("label_up")).alias("correct"))
    )
    selected_points = score_frame.filter(pl.col("policy_selected"))
    first = (
        selected_points.sort(["window_start", "market_id", "seconds_elapsed"])
        .group_by("market_id", maintain_order=True)
        .first()
    )
    eligible_markets = frame["market_id"].n_unique()
    metrics = _direction_metrics(first, eligible_markets)
    point_accuracy = float(selected_points["correct"].mean()) if selected_points.height else None
    return {
        "eligible_rows": frame.height,
        "eligible_markets": eligible_markets,
        "accepted_point_rows": selected_points.height,
        "accepted_point_accuracy": point_accuracy,
        "no_trade_point_rows": frame.height - selected_points.height,
        "first_crossing": metrics,
        "first_crossing_ece": _expected_calibration_error(first),
        "daily_first_crossing": _daily_first_crossing_metrics(first, frame),
    }


def _derive_forward_candidate_frames(
    oracle_core: pl.DataFrame,
    external: ExternalSourceFrames,
    config: ChainlinkOiBenchmarkConfig,
) -> dict[str, pl.DataFrame]:
    empty = oracle_core.head(0)
    candle = empty
    if not oracle_core.is_empty() and not external.candles.is_empty():
        candle = derive_chainlink_candle_feature_frame(
            oracle_core,
            external.candles,
            candle_max_age_seconds=config.staleness.candles_seconds,
        )
    chainlink = empty
    chainlink_oi = empty
    if (
        not oracle_core.is_empty()
        and not external.refprice.is_empty()
        and not external.candles.is_empty()
        and not external.open_interest.is_empty()
    ):
        chainlink, chainlink_oi = derive_chainlink_oi_feature_frames(
            oracle_core,
            external.refprice,
            external.candles,
            external.open_interest,
            refprice_max_age_seconds=config.staleness.refprice_seconds,
            candle_max_age_seconds=config.staleness.candles_seconds,
            open_interest_max_age_seconds=config.staleness.open_interest_seconds,
        )
    return {
        CHAINLINK_FULL_CANDIDATE: chainlink,
        CHAINLINK_FULL_OI_CANDIDATE: chainlink_oi,
        LONG_HISTORY_CANDLE_CANDIDATE: candle,
    }


def _complete_candidate_markets(
    frame: pl.DataFrame,
    config: ChainlinkOiBenchmarkConfig,
    *,
    eligibility_column: str | None = None,
) -> pl.DataFrame:
    minimum = config.model.minimum_seconds_after_open
    maximum = config.model.maximum_seconds_after_open
    cadence = config.model.cadence_seconds
    history = (
        frame.filter(pl.col("seconds_elapsed").is_between(0, maximum, closed="both"))
        .group_by("market_id")
        .agg(
            pl.len().alias("history_rows"),
            pl.col("seconds_elapsed").n_unique().alias("history_unique_seconds"),
            pl.col("seconds_elapsed").min().alias("history_min_second"),
            pl.col("seconds_elapsed").max().alias("history_max_second"),
        )
    )
    complete_history = history.filter(
        (pl.col("history_rows") == maximum + 1)
        & (pl.col("history_unique_seconds") == maximum + 1)
        & (pl.col("history_min_second") == 0)
        & (pl.col("history_max_second") == maximum)
    ).select("market_id")
    candidates = frame.filter(
        pl.col("seconds_elapsed").is_between(minimum, maximum, closed="both")
        & ((pl.col("seconds_elapsed") - minimum) % cadence == 0)
    ).join(complete_history, on="market_id", how="inner")
    if eligibility_column is not None:
        candidates = candidates.filter(pl.col(eligibility_column))
    expected = ((maximum - minimum) // cadence) + 1
    complete_candidates = (
        candidates.group_by("market_id")
        .agg(
            pl.len().alias("candidate_rows"),
            pl.col("seconds_elapsed").n_unique().alias("candidate_unique_seconds"),
        )
        .filter(
            (pl.col("candidate_rows") == expected)
            & (pl.col("candidate_unique_seconds") == expected)
        )
        .select("market_id")
    )
    return candidates.join(complete_candidates, on="market_id", how="inner").sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )


def _load_forward_runtime_models(
    config: ChainlinkOiBenchmarkConfig,
    runtime_root: Path,
) -> dict[str, ForwardRuntimeModel]:
    expected_features = _candidate_feature_sets()
    loaded: dict[str, ForwardRuntimeModel] = {}
    for spec in PAPER_EXPORT_SPECS:
        directory = runtime_root / spec.model_key
        manifest_path = directory / "manifest.json"
        model_path = directory / "model.json"
        if not manifest_path.is_file() or not model_path.is_file():
            raise RuntimeError(f"frozen runtime candidate is missing: {spec.model_key}")
        manifest = _read_json(manifest_path)
        model = _read_json(model_path)
        if manifest.get("model_key") != spec.model_key or model.get("model_key") != spec.model_key:
            raise RuntimeError(f"runtime model key mismatch: {spec.model_key}")
        if manifest.get("model_sha256") != file_sha256(model_path):
            raise RuntimeError(f"runtime model checksum mismatch: {spec.model_key}")
        if (
            manifest.get("feature_schema_version") != spec.feature_schema_version
            or model.get("features", {}).get("schema_version") != spec.feature_schema_version
        ):
            raise RuntimeError(f"runtime feature schema mismatch: {spec.model_key}")
        if tuple(model.get("features", {}).get("names", ())) != expected_features[spec.candidate]:
            raise RuntimeError(f"runtime feature order mismatch: {spec.model_key}")
        if (
            manifest.get("deployment_scope") != "paper_only"
            or manifest.get("production_qualified") is not False
            or manifest.get("live_capital_allowed") is not False
        ):
            raise RuntimeError(f"runtime candidate is not paper-only: {spec.model_key}")
        loaded[spec.candidate] = ForwardRuntimeModel(
            candidate=spec.candidate,
            directory=directory,
            manifest=manifest,
            model=model,
        )
    return loaded


def _query_forward_inventory(
    connection: Any,
    config: ChainlinkOiBenchmarkConfig,
    *,
    range_start: datetime,
    range_end: datetime,
) -> pl.DataFrame:
    return _query_frame(
        connection,
        (config.package_root / "sql" / "btc-chainlink-oi-forward-coverage.sql").read_text(),
        {"range_start": range_start, "range_end": range_end},
        cursor_name="btc_chainlink_oi_forward_inventory",
    )


def _validate_forward_range(
    config: ChainlinkOiBenchmarkConfig,
    range_start: datetime,
    range_end: datetime,
) -> None:
    for name, value in (("range_start", range_start), ("range_end", range_end)):
        if value.tzinfo is None:
            raise ValueError(f"{name} must be timezone aware")
        if value.utcoffset() != timedelta(0) or any(
            (value.hour, value.minute, value.second, value.microsecond)
        ):
            raise ValueError(f"{name} must be an exact UTC day boundary")
    if range_start >= range_end:
        raise ValueError("forward range_start must precede range_end")
    if range_start < config.windows.source_range_end:
        raise ValueError("forward range overlaps consumed development evidence")
    if range_end - range_start > timedelta(days=MAX_FORWARD_RANGE_DAYS):
        raise ValueError(f"forward range cannot exceed {MAX_FORWARD_RANGE_DAYS} days")


def _inventory_payload(frame: pl.DataFrame) -> dict[str, Any]:
    columns = (
        "labeled_markets",
        "opening_reference_markets",
        "core_fact_markets",
        "complete_core_history_markets",
        "execution_snapshot_rows",
    )
    if frame.is_empty():
        return {"days": [], "totals": {name: 0 for name in columns}}
    days = []
    for row in frame.to_dicts():
        days.append(
            {
                "utc_date": row["utc_date"].isoformat(),
                **{name: int(row[name]) for name in columns},
            }
        )
    return {
        "days": days,
        "totals": {name: int(frame[name].sum()) for name in columns},
    }


def _timestamped_source_summary(frame: pl.DataFrame, timestamp: str) -> dict[str, Any]:
    if frame.is_empty():
        return {"rows": 0, "start": None, "end": None}
    return {
        "rows": frame.height,
        "start": frame[timestamp].min().isoformat(),
        "end": frame[timestamp].max().isoformat(),
    }


def _empty_forward_metrics() -> dict[str, Any]:
    return {
        "eligible_rows": 0,
        "eligible_markets": 0,
        "accepted_point_rows": 0,
        "accepted_point_accuracy": None,
        "no_trade_point_rows": 0,
        "first_crossing": _direction_metrics(pl.DataFrame(), 0),
        "first_crossing_ece": None,
        "daily_first_crossing": [],
    }


def _expected_calibration_error(frame: pl.DataFrame) -> float | None:
    if frame.is_empty():
        return None
    probability = frame["probability_up"].to_numpy()
    labels = frame["label_up"].to_numpy()
    order = np.argsort(probability)
    error = 0.0
    for indices in np.array_split(order, min(10, len(order))):
        if len(indices):
            error += (
                len(indices)
                / len(order)
                * abs(float(probability[indices].mean()) - float(labels[indices].mean()))
            )
    return float(error)


def _daily_first_crossing_metrics(
    first: pl.DataFrame,
    eligible: pl.DataFrame,
) -> list[dict[str, Any]]:
    eligible_days = (
        eligible.select("market_id", "window_start")
        .unique("market_id")
        .with_columns(pl.col("window_start").dt.date().alias("utc_date"))
        .group_by("utc_date")
        .agg(pl.len().alias("eligible_markets"))
    )
    if first.is_empty():
        return [
            {
                "utc_date": row["utc_date"].isoformat(),
                "eligible_markets": int(row["eligible_markets"]),
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "accuracy": None,
                "coverage": 0.0,
            }
            for row in eligible_days.sort("utc_date").to_dicts()
        ]
    trades = (
        first.with_columns(pl.col("window_start").dt.date().alias("utc_date"))
        .group_by("utc_date")
        .agg(
            pl.len().alias("trades"),
            pl.col("correct").sum().alias("wins"),
            pl.col("correct").mean().alias("accuracy"),
        )
    )
    return (
        eligible_days.join(trades, on="utc_date", how="left")
        .with_columns(
            pl.col("trades").fill_null(0),
            pl.col("wins").fill_null(0),
        )
        .with_columns(
            (pl.col("trades") - pl.col("wins")).alias("losses"),
            (pl.col("trades") / pl.col("eligible_markets")).alias("coverage"),
        )
        .sort("utc_date")
        .with_columns(pl.col("utc_date").cast(pl.String))
        .to_dicts()
    )


def _forward_blockers(
    eligibility: dict[str, Any],
    models: dict[str, dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    totals = eligibility["inventory_totals"]
    if totals["labeled_markets"] == 0:
        blockers.append("no valid official Up/Down labels exist in the forward range")
    if totals["opening_reference_markets"] < totals["labeled_markets"]:
        missing = totals["labeled_markets"] - totals["opening_reference_markets"]
        blockers.append(f"{missing} labeled markets lack exactly one completed opening reference")
    if totals["complete_core_history_markets"] < totals["core_fact_markets"]:
        missing = totals["core_fact_markets"] - totals["complete_core_history_markets"]
        blockers.append(f"{missing} fact-qualified markets lack complete causal Binance history")
    raw_external = eligibility["raw_external_sources"]
    for source, label in (
        ("polygon_oracle", "Polygon oracle"),
        ("chainlink_refprice", "Chainlink RefPrice"),
        ("chainlink_one_minute_candles", "Chainlink one-minute candles"),
        ("binance_five_minute_open_interest", "Binance five-minute open interest"),
    ):
        if raw_external[source]["rows"] == 0:
            blockers.append(f"{label} has zero bounded source rows")
    if not eligibility["strict_candidate_keys_identical"]:
        blockers.append("candidate strict causal point keys are not identical")
    for candidate, metrics in models.items():
        if metrics["eligible_markets"] == 0:
            blockers.append(f"{candidate} has zero strict causal labeled markets")
    return blockers


def _candidate_keys_identical(frames: dict[str, pl.DataFrame]) -> bool:
    values = list(frames.values())
    if not values:
        return True
    if all(frame.is_empty() for frame in values):
        return True
    if any(frame.is_empty() for frame in values):
        return False
    anchor = values[0].select(*POINT_KEY_COLUMNS)
    return all(
        anchor.equals(frame.select(*POINT_KEY_COLUMNS), null_equal=True)
        for frame in values[1:]
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact is not an object: {path}")
    return value


def _report_html(result: dict[str, Any]) -> str:
    model_rows = []
    for name, metrics in result["models"].items():
        crossing = metrics["first_crossing"]
        accuracy = crossing.get("accuracy")
        model_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{metrics['eligible_markets']:,}</td>"
            f"<td>{crossing.get('trades', 0):,}</td>"
            f"<td>{'n/a' if accuracy is None else f'{accuracy:.2%}'}</td>"
            f"<td>{crossing.get('coverage', 0.0):.2%}</td>"
            f"<td>{crossing.get('losses', 0):,}</td>"
            "</tr>"
        )
    blockers = "".join(f"<li>{html.escape(item)}</li>" for item in result["blockers"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Untouched forward score</title>
<style>body{{font:15px system-ui;max-width:1000px;margin:40px auto;padding:0 20px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border:1px solid #ddd;text-align:right}}
th:first-child,td:first-child{{text-align:left}}code{{background:#f3f3f3;padding:2px 4px}}</style></head>
<body><h1>Chainlink/OI untouched forward score</h1>
<p>Status: <code>{html.escape(result["status"])}</code>. Range:
{html.escape(result["range"]["start"])} to {html.escape(result["range"]["end_exclusive"])}.</p>
<table><thead><tr><th>Model</th><th>Eligible markets</th><th>Trades</th>
<th>Accuracy</th><th>Coverage</th><th>Losses</th></tr></thead>
<tbody>{"".join(model_rows)}</tbody></table>
<h2>Blockers</h2><ul>{blockers or "<li>None</li>"}</ul>
<p>No retraining, process changes, or database mutations were performed.</p></body></html>"""
