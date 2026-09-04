from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

from . import PROCESS_ID
from .asos_ingestion import ingest_asos, ingest_asos_one_minute, ingest_asos_resolution
from .asymmetric_benchmark import run_asymmetric_benchmark
from .benchmark import run_benchmark
from .challenger_tournament import run_challenger_tournament
from .config import Settings
from .database import connection
from .environment_snapshot import audit_environment_coverage, export_environment_snapshot
from .execution_ingestion import ingest_pmxt_execution
from .goes_ingestion import ingest_goes
from .hrrr_environment_ingestion import ingest_hrrr_environment
from .hrrr_ingestion import ingest_hrrr
from .backfill_contract import Job
from .market_ingestion import ingest_markets, ingest_price_history
from .modeling import reconcile_labels, train_model
from .pmxt_full_market_tournament import run_pmxt_full_market_tournament
from .residual_opportunity_benchmark import run_residual_opportunity_benchmark
from .tail_calibration_tournament import run_tail_calibration_tournament

HANDLERS = {
    "polymarket_temperature_markets": ingest_markets,
    "polymarket_temperature_price_history": ingest_price_history,
    "asos_station_observations": ingest_asos,
    "asos_resolution_observations": ingest_asos_resolution,
    "asos_one_minute_observations": ingest_asos_one_minute,
    "hrrr_point_forecasts": ingest_hrrr,
    "goes_abi_klga_features": ingest_goes,
    "hrrr_environment_features": ingest_hrrr_environment,
    "pmxt_temperature_execution": ingest_pmxt_execution,
}


def json_default(value):
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)


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


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("SHA-256 must use 64 lowercase hexadecimal characters")
    return value


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
    unified = subparsers.add_parser("execute-unified-backfill")
    unified.add_argument("--strategy-key", required=True, choices=(
        "goes_abi_klga_features", "hrrr_environment_features"
    ))
    unified.add_argument("--job-id", required=True)
    unified.add_argument("--lease-token", required=True)
    unified.add_argument("--worker-id", required=True)
    unified.add_argument("--start", required=True, type=_instant)
    unified.add_argument("--end", required=True, type=_instant)
    unified.add_argument("--parameters", required=True)
    environment_snapshot = subparsers.add_parser("export-environment-snapshot")
    environment_snapshot.add_argument("--start", type=_date, default=date(2019, 1, 1))
    environment_snapshot.add_argument("--end", type=_date, default=date(2026, 8, 10))
    environment_snapshot.add_argument("--snapshot-id", required=True)
    environment_audit = subparsers.add_parser("audit-environment")
    environment_audit.add_argument("--start", type=_date, default=date(2019, 1, 1))
    environment_audit.add_argument("--end", type=_date, default=date(2026, 8, 10))
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
    tournament = subparsers.add_parser("challenger-tournament")
    tournament.add_argument("--training-start", required=True, type=_date)
    tournament.add_argument("--training-end", required=True, type=_date)
    tournament.add_argument("--calibration-start", required=True, type=_date)
    tournament.add_argument("--calibration-end", required=True, type=_date)
    tournament.add_argument("--economic-start", required=True, type=_date)
    tournament.add_argument("--economic-end", required=True, type=_date)
    tournament.add_argument("--sealed-start", required=True, type=_date)
    tournament.add_argument(
        "--weather-model-image-id", required=True, type=_sha256_image_id
    )
    tail_tournament = subparsers.add_parser("tail-calibration-tournament")
    tail_tournament.add_argument("--training-start", required=True, type=_date)
    tail_tournament.add_argument("--training-end", required=True, type=_date)
    tail_tournament.add_argument("--calibration-start", required=True, type=_date)
    tail_tournament.add_argument("--calibration-end", required=True, type=_date)
    tail_tournament.add_argument("--economic-start", required=True, type=_date)
    tail_tournament.add_argument("--economic-end", required=True, type=_date)
    tail_tournament.add_argument("--sealed-start", required=True, type=_date)
    tail_tournament.add_argument("--git-revision", required=True)
    tail_tournament.add_argument("--runner-image-id", required=True, type=_sha256_image_id)
    pmxt_tournament = subparsers.add_parser("pmxt-full-market-tournament")
    pmxt_tournament.add_argument("--source-weather-artifact", required=True, type=Path)
    pmxt_tournament.add_argument("--source-weather-artifact-sha256", required=True, type=_sha256)
    pmxt_tournament.add_argument("--development-start", required=True, type=_date)
    pmxt_tournament.add_argument("--development-end", required=True, type=_date)
    pmxt_tournament.add_argument("--sealed-start", required=True, type=_date)
    pmxt_tournament.add_argument("--sealed-end", required=True, type=_date)
    pmxt_tournament.add_argument("--git-revision", required=True)
    pmxt_tournament.add_argument("--runner-image-id", required=True, type=_sha256_image_id)
    pmxt_tournament.add_argument("--output-directory", type=Path)
    pmxt_tournament.add_argument("--bootstrap-iterations", type=int, default=200)
    subparsers.add_parser("readiness")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    settings.prepare_directories()
    if args.command == "execute-unified-backfill":
        request = json.loads(args.parameters)
        job = Job(
            job_id=args.job_id,
            ingester_key=args.strategy_key,
            range_start=args.start,
            range_end=args.end,
            request=request,
            attempt=1,
            lease_token=args.lease_token,
            worker_id=args.worker_id,
        )
        summary = HANDLERS[args.strategy_key](settings, job)
        counters = summary.get("counters", summary)
        records_verified = sum(
            int(counters.get(key, 0)) for key in ("completed", "reused")
        )
        result = {
            "records_verified": records_verified,
            "verified_coverage": {
                "requested_start": args.start,
                "requested_end": args.end,
                "strategy_key": args.strategy_key,
                "records_verified": records_verified,
            },
            "summary": summary,
        }
    elif args.command == "export-environment-snapshot":
        result = export_environment_snapshot(
            settings,
            start=args.start,
            end=args.end,
            snapshot_id=args.snapshot_id,
        )
    elif args.command == "audit-environment":
        result = audit_environment_coverage(settings, start=args.start, end=args.end)
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
    elif args.command == "challenger-tournament":
        result = run_challenger_tournament(
            settings,
            training_start=args.training_start,
            training_end=args.training_end,
            calibration_start=args.calibration_start,
            calibration_end=args.calibration_end,
            economic_start=args.economic_start,
            economic_end=args.economic_end,
            sealed_start=args.sealed_start,
            weather_model_image_id=args.weather_model_image_id,
        )
    elif args.command == "tail-calibration-tournament":
        result = run_tail_calibration_tournament(
            settings,
            training_start=args.training_start,
            training_end=args.training_end,
            calibration_start=args.calibration_start,
            calibration_end=args.calibration_end,
            economic_start=args.economic_start,
            economic_end=args.economic_end,
            sealed_start=args.sealed_start,
            git_revision=args.git_revision,
            runner_image_id=args.runner_image_id,
        )
    elif args.command == "pmxt-full-market-tournament":
        result = run_pmxt_full_market_tournament(
            settings,
            source_weather_artifact_path=args.source_weather_artifact,
            source_weather_artifact_sha256=args.source_weather_artifact_sha256,
            development_start=args.development_start,
            development_end=args.development_end,
            sealed_start=args.sealed_start,
            sealed_end=args.sealed_end,
            git_revision=args.git_revision,
            runner_image_id=args.runner_image_id,
            output_directory=args.output_directory,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    elif args.command == "readiness":
        result = _readiness(settings)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
