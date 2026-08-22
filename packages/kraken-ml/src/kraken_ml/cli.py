from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import (
    BenchmarkConfig,
    ExpectancyConfig,
    load_config,
    load_expectancy_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kraken-ml",
        description="Run the frozen PF_XBTUSD classical-ML edge benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare",
        "develop",
        "prepare-expectancy",
        "develop-expectancy",
        "tournament",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument(
            "--refresh",
            action="store_true",
            help="Rebuild content-addressed snapshots after verifying all source objects.",
        )
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--run-id", required=True)
    evaluate_expectancy = subparsers.add_parser("evaluate-expectancy")
    evaluate_expectancy.add_argument("--config", type=Path, required=True)
    evaluate_expectancy.add_argument("--run-id", required=True)
    funding = subparsers.add_parser(
        "backfill-funding",
        help="Import first-party Kraken funding history into missing lake buckets.",
    )
    funding.add_argument("--config", type=Path, required=True)
    funding.add_argument(
        "--archive-path",
        type=Path,
        help="Use a local Kraken funding export ZIP instead of downloading it.",
    )
    funding.add_argument(
        "--recent-json",
        type=Path,
        help="Use a local historical-funding-rates response instead of downloading it.",
    )
    return parser


def _configure_cpu_environment(
    config: BenchmarkConfig | ExpectancyConfig,
    *,
    command: str,
) -> None:
    threads = min(
        config.compute.final_refit_threads,
        config.compute.available_cores,
    )
    value = str(threads)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = value
    os.environ["LOKY_MAX_CPU_COUNT"] = str(config.compute.available_cores)
    os.environ["POLARS_MAX_THREADS"] = (
        str(config.compute.comparison_estimator_threads)
        if command in {"develop", "develop-expectancy"}
        else value
    )


def _print_summary(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command in {
        "prepare-expectancy",
        "develop-expectancy",
        "evaluate-expectancy",
        "tournament",
    }:
        config = load_expectancy_config(arguments.config)
    elif arguments.command == "backfill-funding":
        try:
            config = load_expectancy_config(arguments.config)
        except KeyError:
            config = load_config(arguments.config)
    else:
        config = load_config(arguments.config)
    _configure_cpu_environment(config, command=arguments.command)

    if arguments.command == "prepare":
        from .training import prepare_benchmark

        raw, features = prepare_benchmark(config, refresh=arguments.refresh)
        _print_summary(
            {
                "raw_snapshot": str(raw.path),
                "raw_sha256": raw.sha256,
                "raw_rows": raw.row_count,
                "feature_snapshot": str(features.path),
                "feature_sha256": features.sha256,
                "feature_rows": features.row_count,
                "parallel_fits": config.compute.available_cores,
                "reserved_cores": config.compute.reserve_cores,
            }
        )
        return 0

    if arguments.command == "develop":
        from .training import run_development

        result = run_development(config, refresh=arguments.refresh)
        _print_summary(
            {
                "run_id": result["run_id"],
                "selected_model": result["selected_model"],
                "selected_feature_set": result["selected_feature_set"],
                "development_gates_passed": result["gates"]["pass"],
                "qualified_for_freeze": result["qualified_for_freeze"],
                "holdout_status": result["holdout_status"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
        return 0

    if arguments.command == "evaluate":
        from .training import evaluate_holdout

        result = evaluate_holdout(config, run_id=arguments.run_id)
        _print_summary(
            {
                "run_id": result["run_id"],
                "verdict": result["verdict"],
                "holdout_gates_passed": result["gates"]["pass"],
                "trades": result["economics"]["trades"],
                "net_expectancy_bps": result["economics"]["net_expectancy_bps"],
                "profit_factor": result["economics"]["profit_factor"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
        return 0

    if arguments.command == "prepare-expectancy":
        from .regression_training import prepare_expectancy_benchmark

        raw, features = prepare_expectancy_benchmark(
            config,
            refresh=arguments.refresh,
        )
        _print_summary(
            {
                "raw_snapshot": str(raw.path),
                "raw_sha256": raw.sha256,
                "raw_rows": raw.row_count,
                "feature_snapshots": {
                    str(horizon): {
                        "path": str(snapshot.path),
                        "sha256": snapshot.sha256,
                        "rows": snapshot.row_count,
                    }
                    for horizon, snapshot in features.items()
                },
                "parallel_fits": config.compute.available_cores,
                "reserved_cores": config.compute.reserve_cores,
            }
        )
        return 0

    if arguments.command == "develop-expectancy":
        from .regression_training import run_regression_development

        result = run_regression_development(
            config,
            refresh=arguments.refresh,
        )
        selected = result["selected"]
        _print_summary(
            {
                "run_id": result["run_id"],
                "verdict": result["verdict"],
                "candidate_id": selected["candidate_id"],
                "diagnostic_only": selected["diagnostic_only"],
                "development_gates_passed": selected["gates"]["pass"],
                "final_confirmation": result["final_confirmation"]["status"],
                "holdout_status": result["holdout"]["status"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
        return 0

    if arguments.command == "evaluate-expectancy":
        from .regression_training import evaluate_regression_holdout

        result = evaluate_regression_holdout(config, run_id=arguments.run_id)
        _print_summary(
            {
                "run_id": result["run_id"],
                "verdict": result["verdict"],
                "holdout_gates_passed": result["gates"]["pass"],
                "trades": result["economics"]["trades"],
                "net_expectancy_bps": result["economics"]["net_expectancy_bps"],
                "profit_factor": result["economics"]["profit_factor"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
        return 0

    if arguments.command == "tournament":
        from .tournament import run_tournament

        result = run_tournament(config, refresh=arguments.refresh)
        _print_summary(
            {
                "run_id": result["run_id"],
                "verdict": result["verdict"],
                "selected_candidate": result["selected"]["candidate_id"],
                "qualified_for_confirmation": result["qualified_for_confirmation"],
                "confirmation_status": result["confirmation"]["status"],
                "holdout_status": result["holdout"]["status"],
            }
        )
        return 0

    if arguments.command == "backfill-funding":
        from .funding_backfill import backfill_funding

        result = backfill_funding(
            config,
            archive_path=arguments.archive_path,
            recent_json_path=arguments.recent_json,
        )
        _print_summary(result.summary())
        return 0

    raise AssertionError(f"unhandled command: {arguments.command}")


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except (RuntimeError, ValueError) as error:
        print(f"kraken-ml: error: {error}", file=sys.stderr)
        return 2
