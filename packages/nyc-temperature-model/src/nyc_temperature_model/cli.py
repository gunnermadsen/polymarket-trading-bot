from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from . import PROCESS_ID
from .asos_ingestion import ingest_asos, ingest_asos_one_minute, ingest_asos_resolution
from .asymmetric_benchmark import run_asymmetric_benchmark
from .benchmark import run_benchmark
from .config import Settings
from .database import connection
from .execution_ingestion import ingest_pmxt_execution
from .hrrr_ingestion import ingest_hrrr
from .jobs import (
    SUPPORTED_INGESTERS,
    cancel_job,
    enqueue,
    job_rows,
    json_default,
    run_worker,
)
from .market_ingestion import ingest_markets, ingest_price_history
from .modeling import reconcile_labels, train_model
from .residual_opportunity_benchmark import run_residual_opportunity_benchmark

HANDLERS = {
    "polymarket_temperature_markets": ingest_markets,
    "polymarket_temperature_price_history": ingest_price_history,
    "asos_station_observations": ingest_asos,
    "asos_resolution_observations": ingest_asos_resolution,
    "asos_one_minute_observations": ingest_asos_one_minute,
    "hrrr_point_forecasts": ingest_hrrr,
    "pmxt_temperature_execution": ingest_pmxt_execution,
}


def _date(value: str) -> date:
    return date.fromisoformat(value)


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _sha256_image_id(value: str) -> str:
    if len(value) != 71 or not value.startswith("sha256:"):
        raise argparse.ArgumentTypeError("image ID must use sha256:<64 lowercase hex characters>")
    if any(character not in "0123456789abcdef" for character in value[7:]):
        raise argparse.ArgumentTypeError("image ID must use lowercase hexadecimal characters")
    return value


def _utc_day(value: date) -> datetime:
    return datetime.combine(value, datetime.min.time(), UTC)


def _month_ranges(start: date, end: date):
    current = start.replace(day=1)
    while current < end:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield max(start, current), min(end, next_month)
        current = next_month


def _year_ranges(start: date, end: date):
    current = start
    while current < end:
        next_year = date(current.year + 1, 1, 1)
        yield current, min(end, next_year)
        current = next_year


def _model_path(settings: Settings, model_run_id: str) -> Path:
    with connection(settings.database_url) as conn:
        row = conn.execute(
            """
            SELECT model_uri
            FROM weather.model_runs
            WHERE process_id=%s AND model_run_id=%s
            """,
            (PROCESS_ID, model_run_id),
        ).fetchone()
    if not row:
        raise ValueError(f"unknown model_run_id: {model_run_id}")
    return Path(row["model_uri"])


def _readiness(settings: Settings) -> dict:
    with connection(settings.database_url) as conn:
        return conn.execute(
            """
            SELECT
              (SELECT count(*)::int FROM weather.temperature_markets) AS markets,
              (SELECT count(DISTINCT event_date)::int FROM weather.temperature_markets) AS market_days,
              (SELECT count(*)::int FROM weather.temperature_markets WHERE resolved_yes IS NOT NULL)
                AS resolved_markets,
              (SELECT count(*)::bigint FROM weather.station_observations
                WHERE provider='iem_asos_metar' AND temperature_f IS NOT NULL)
                AS resolution_observations,
              (SELECT count(*)::bigint FROM weather.station_observations
                WHERE provider='iem_ncei_asos_one_minute' AND temperature_f IS NOT NULL)
                AS one_minute_observations,
              (SELECT count(DISTINCT decision_time)::int FROM weather.hrrr_point_forecasts)
                AS hrrr_decisions,
              (SELECT count(*)::bigint FROM weather.price_history) AS indicative_prices,
              (SELECT count(*)::int FROM weather.execution_snapshots WHERE yes_ask_vwap IS NOT NULL)
                AS executable_yes_snapshots,
              (SELECT count(*)::int FROM weather.label_reconciliation
                WHERE winner_matches_station IS NOT NULL) AS reconciled_days,
              (SELECT count(*)::int FROM weather.label_reconciliation
                WHERE winner_matches_station) AS matching_days,
              (SELECT count(*)::int FROM weather.model_runs) AS model_runs,
              (SELECT count(*)::int FROM weather.benchmark_runs) AS benchmark_runs,
              (SELECT count(*)::int FROM weather.asymmetric_policy_runs)
                AS asymmetric_policy_runs
            """
        ).fetchone()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nyc-temperature-model")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("worker")
    jobs = subparsers.add_parser("jobs")
    jobs.add_argument("--limit", type=int, default=50)
    cancel = subparsers.add_parser("cancel-job")
    cancel.add_argument("job_id")
    enqueue_parser = subparsers.add_parser("enqueue")
    enqueue_parser.add_argument("ingester", choices=SUPPORTED_INGESTERS)
    enqueue_parser.add_argument("--start", required=True, type=_instant)
    enqueue_parser.add_argument("--end", required=True, type=_instant)
    enqueue_parser.add_argument("--parameters", default="{}")
    enqueue_parser.add_argument("--idempotency-key")
    enqueue_parser.add_argument("--depends-on-job-id")
    pilot = subparsers.add_parser("enqueue-pilot")
    pilot.add_argument("--weather-start", type=_date, default=date(2019, 1, 1))
    pilot.add_argument("--market-start", type=_date, default=date(2025, 9, 1))
    pilot.add_argument("--end", type=_date, default=datetime.now(UTC).date() + timedelta(days=1))
    reconcile = subparsers.add_parser("reconcile-labels")
    reconcile.add_argument("--start", required=True, type=_date)
    reconcile.add_argument("--end", required=True, type=_date)
    train = subparsers.add_parser("train")
    train.add_argument(
        "--candidate",
        required=True,
        choices=("raw_hrrr", "linear_bias", "histogram_residual"),
    )
    train.add_argument("--decision-hour", required=True, type=int, choices=(0, 12))
    train.add_argument("--training-start", required=True, type=_date)
    train.add_argument("--training-end", required=True, type=_date)
    train.add_argument("--calibration-start", required=True, type=_date)
    train.add_argument("--calibration-end", required=True, type=_date)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--model-run-id", required=True)
    benchmark.add_argument("--evaluation-start", required=True, type=_date)
    benchmark.add_argument("--evaluation-end", required=True, type=_date)
    benchmark.add_argument("--quantity", type=float, default=5.0, choices=(1.0, 5.0, 10.0))
    benchmark.add_argument(
        "--evidence-tier", choices=("indicative", "executable_taker"), default="executable_taker"
    )
    benchmark.add_argument("--safety-buffer", type=float, default=0.02)
    asymmetric = subparsers.add_parser("asymmetric-benchmark")
    asymmetric.add_argument("--midnight-model-run-id", required=True)
    asymmetric.add_argument("--noon-model-run-id", required=True)
    asymmetric.add_argument("--discovery-start", required=True, type=_date)
    asymmetric.add_argument("--discovery-end", required=True, type=_date)
    asymmetric.add_argument("--evaluation-start", required=True, type=_date)
    asymmetric.add_argument("--evaluation-end", required=True, type=_date)
    asymmetric.add_argument("--quantity", type=float, default=5.0, choices=(1.0, 5.0, 10.0))
    asymmetric.add_argument("--modeled-slippage", type=float, default=0.01)
    asymmetric.add_argument("--probability-bootstrap-iterations", type=int, default=2000)
    residual = subparsers.add_parser("residual-opportunity-benchmark")
    residual.add_argument("--source-policy-run-id", required=True)
    residual.add_argument("--weather-model-image-id", required=True, type=_sha256_image_id)
    residual.add_argument("--bootstrap-iterations", type=int, default=1000)
    subparsers.add_parser("readiness")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    settings.prepare_directories()
    if args.command == "worker":
        run_worker(settings, HANDLERS)
        return
    if args.command == "jobs":
        result = job_rows(settings.database_url, args.limit)
    elif args.command == "cancel-job":
        result = {"job_id": args.job_id, "status": cancel_job(settings.database_url, args.job_id)}
    elif args.command == "enqueue":
        result = {
            "job_id": enqueue(
                settings.database_url,
                ingester_key=args.ingester,
                range_start=args.start,
                range_end=args.end,
                parameters=json.loads(args.parameters),
                idempotency_key=args.idempotency_key,
                depends_on_job_id=args.depends_on_job_id,
            )
        }
    elif args.command == "enqueue-pilot":
        if not args.weather_start < args.end or not args.market_start < args.end:
            raise ValueError("pilot start dates must be before end")
        jobs = []
        market_job = enqueue(
            settings.database_url,
            ingester_key="polymarket_temperature_markets",
            range_start=_utc_day(args.market_start),
            range_end=_utc_day(args.end),
            idempotency_key=f"nyc-temperature-markets:{args.market_start}:{args.end}:v1",
        )
        jobs.append({"ingester": "polymarket_temperature_markets", "job_id": market_job})
        for start, end in _year_ranges(args.weather_start, args.end):
            resolution_job_id = enqueue(
                settings.database_url,
                ingester_key="asos_resolution_observations",
                range_start=_utc_day(start),
                range_end=_utc_day(end),
                idempotency_key=f"klga-asos-metars:{start}:{end}:v1",
            )
            jobs.append(
                {"ingester": "asos_resolution_observations", "job_id": resolution_job_id}
            )
            one_minute_job_id = enqueue(
                settings.database_url,
                ingester_key="asos_one_minute_observations",
                range_start=_utc_day(start),
                range_end=_utc_day(end),
                idempotency_key=f"klga-asos-one-minute:{start}:{end}:v1",
            )
            jobs.append(
                {"ingester": "asos_one_minute_observations", "job_id": one_minute_job_id}
            )
        for start, end in _month_ranges(args.weather_start, args.end):
            job_id = enqueue(
                settings.database_url,
                ingester_key="hrrr_point_forecasts",
                range_start=_utc_day(start),
                range_end=_utc_day(end),
                parameters={"availability_lag_minutes": 75},
                idempotency_key=f"klga-hrrr:{start}:{end}:lag75:v1",
            )
            jobs.append({"ingester": "hrrr_point_forecasts", "job_id": job_id})
        for start, end in _month_ranges(args.market_start, args.end):
            price_job = enqueue(
                settings.database_url,
                ingester_key="polymarket_temperature_price_history",
                range_start=_utc_day(start),
                range_end=_utc_day(end),
                depends_on_job_id=market_job,
                idempotency_key=f"nyc-temperature-prices:{start}:{end}:v1",
            )
            jobs.append({"ingester": "polymarket_temperature_price_history", "job_id": price_job})
            pmxt_start = max(start, date(2026, 4, 14))
            if pmxt_start < end:
                pmxt_job = enqueue(
                    settings.database_url,
                    ingester_key="pmxt_temperature_execution",
                    range_start=_utc_day(pmxt_start),
                    range_end=_utc_day(end),
                    depends_on_job_id=market_job,
                    idempotency_key=f"nyc-temperature-pmxt:{pmxt_start}:{end}:v1",
                )
                jobs.append({"ingester": "pmxt_temperature_execution", "job_id": pmxt_job})
        result = {"jobs": jobs, "count": len(jobs)}
    elif args.command == "reconcile-labels":
        result = reconcile_labels(settings, args.start, args.end)
    elif args.command == "train":
        result = train_model(
            settings,
            candidate=args.candidate,
            decision_hour=args.decision_hour,
            training_start=args.training_start,
            training_end=args.training_end,
            calibration_start=args.calibration_start,
            calibration_end=args.calibration_end,
        )
    elif args.command == "benchmark":
        result = run_benchmark(
            settings,
            model_run_id=args.model_run_id,
            model_path=_model_path(settings, args.model_run_id),
            evaluation_start=args.evaluation_start,
            evaluation_end=args.evaluation_end,
            quantity=args.quantity,
            evidence_tier=args.evidence_tier,
            safety_buffer=args.safety_buffer,
        )
    elif args.command == "asymmetric-benchmark":
        result = run_asymmetric_benchmark(
            settings,
            midnight_model_run_id=args.midnight_model_run_id,
            noon_model_run_id=args.noon_model_run_id,
            discovery_start=args.discovery_start,
            discovery_end=args.discovery_end,
            evaluation_start=args.evaluation_start,
            evaluation_end=args.evaluation_end,
            quantity=args.quantity,
            modeled_slippage_per_share=args.modeled_slippage,
            probability_bootstrap_iterations=args.probability_bootstrap_iterations,
        )
    elif args.command == "residual-opportunity-benchmark":
        result = run_residual_opportunity_benchmark(
            settings,
            source_policy_run_id=args.source_policy_run_id,
            weather_model_image_id=args.weather_model_image_id,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    elif args.command == "readiness":
        result = _readiness(settings)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
