from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .admission_benchmark import run_admission_benchmark
from .admission_config import load_admission_benchmark_config
from .benchmark_config import load_entry_benchmark_config
from .champion_vwap_benchmark import run_champion_vwap_benchmark
from .champion_vwap_config import load_champion_vwap_config
from .config import load_config
from .core_config import load_core_config
from .core_extract import extract_core_source, snapshot_residual_admission_source
from .core_features import build_core_features
from .core_report import generate_core_report
from .core_training import develop_core_models, evaluate_core_holdout
from .entry_benchmark import run_entry_benchmark
from .extract import extract_source
from .features import build_features
from .fixed_time_benchmark import (
    FIXED_TIME_PAPER_AUTHORIZATION,
    freeze_and_export_fixed_time_paper_candidate,
    run_fixed_time_accuracy_benchmark,
)
from .fixed_time_config import load_fixed_time_accuracy_config
from .fixed_time_reversal_benchmark import (
    FIXED_TIME_REVERSAL_PAPER_AUTHORIZATION,
    freeze_and_export_fixed_time_reversal_paper_candidate,
    run_fixed_time_reversal_benchmark,
)
from .fixed_time_reversal_config import load_fixed_time_reversal_config
from .fixed_time_selective_benchmark import (
    FIXED_TIME_SELECTIVE_PAPER_AUTHORIZATION,
    freeze_and_export_fixed_time_selective_paper_candidate,
    run_fixed_time_selective_benchmark,
)
from .fixed_time_selective_config import load_fixed_time_selective_config
from .frequency_policy_benchmark import run_frequency_policy_benchmark
from .frequency_policy_config import load_frequency_policy_benchmark_config
from .loss_tail_benchmark import run_loss_tail_benchmark
from .loss_tail_config import load_loss_tail_benchmark_config
from .oracle_book_benchmark import (
    load_oracle_book_benchmark_config,
    run_oracle_book_benchmark,
)
from .paper_candidate import (
    PAPER_ONLY_AUTHORIZATION,
    run_paper_candidate_export,
)
from .persistence_benchmark import run_persistence_benchmark
from .persistence_config import load_persistence_benchmark_config
from .policy_benchmark import run_saved_policy_benchmark
from .policy_config import load_saved_policy_benchmark_config
from .price_aware_benchmark import run_price_aware_benchmark
from .price_aware_config import load_price_aware_benchmark_config
from .report import generate_report
from .residual_admission_benchmark import run_residual_admission_benchmark
from .residual_admission_config import load_residual_admission_config
from .runtime_export import export_runtime_model
from .train import train_models
from .training_readiness import prepare_training_readiness


def main() -> None:
    parser = argparse.ArgumentParser(prog="btc-directional-model")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "features", "train", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--force", action="store_true")
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run", type=Path, required=True)
    core_report_parser = subparsers.add_parser("core-report")
    core_report_parser.add_argument("--run", type=Path, required=True)
    for name in ("core-extract", "core-features"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument(
            "--scope",
            choices=("pre_holdout", "holdout"),
            required=True,
        )
        command.add_argument("--force", action="store_true")
    core_snapshot = subparsers.add_parser("core-snapshot-residual-source")
    core_snapshot.add_argument("--config", type=Path, required=True)
    core_snapshot.add_argument("--source-dir", type=Path, required=True)
    core_develop = subparsers.add_parser("core-develop")
    core_develop.add_argument("--config", type=Path, required=True)
    core_evaluate = subparsers.add_parser("core-evaluate-holdout")
    core_evaluate.add_argument("--config", type=Path, required=True)
    core_evaluate.add_argument("--freeze", type=Path, required=True)
    core_export = subparsers.add_parser("core-export-runtime")
    core_export.add_argument("--freeze", type=Path, required=True)
    core_export.add_argument("--golden-features", type=Path, required=True)
    core_export.add_argument("--output-root", type=Path, required=True)
    core_export.add_argument("--model-key", required=True)
    core_run = subparsers.add_parser("core-run")
    core_run.add_argument("--config", type=Path, required=True)
    core_run.add_argument("--force", action="store_true")
    core_readiness = subparsers.add_parser("core-training-readiness")
    core_readiness.add_argument("--config", type=Path, required=True)
    core_readiness.add_argument(
        "--execution-output",
        type=Path,
        required=True,
    )
    core_readiness.add_argument("--output-dir", type=Path, required=True)
    core_readiness.add_argument("--force", action="store_true")
    entry_run = subparsers.add_parser("entry-benchmark-run")
    entry_run.add_argument("--config", type=Path, required=True)
    entry_run.add_argument("--force", action="store_true")
    oracle_book_run = subparsers.add_parser("oracle-book-benchmark-run")
    oracle_book_run.add_argument("--config", type=Path, required=True)
    price_aware_run = subparsers.add_parser("price-aware-benchmark-run")
    price_aware_run.add_argument("--config", type=Path, required=True)
    price_aware_run.add_argument("--force", action="store_true")
    loss_tail_run = subparsers.add_parser("loss-tail-benchmark-run")
    loss_tail_run.add_argument("--config", type=Path, required=True)
    loss_tail_run.add_argument("--force", action="store_true")
    champion_vwap_run = subparsers.add_parser("champion-vwap-calibration-run")
    champion_vwap_run.add_argument("--config", type=Path, required=True)
    fixed_time_run = subparsers.add_parser("fixed-120-benchmark-run")
    fixed_time_run.add_argument("--config", type=Path, required=True)
    fixed_time_export = subparsers.add_parser("fixed-120-paper-candidate-export")
    fixed_time_export.add_argument("--config", type=Path, required=True)
    fixed_time_export.add_argument(
        "--benchmark-run",
        type=Path,
        required=True,
    )
    fixed_time_export.add_argument("--model-key", required=True)
    fixed_time_export.add_argument(
        "--authorize-paper-only",
        action="store_true",
        help="authorize an exact-120-second model only for paper evaluation",
    )
    fixed_time_selective_run = subparsers.add_parser(
        "fixed-120-selective-benchmark-run"
    )
    fixed_time_selective_run.add_argument("--config", type=Path, required=True)
    fixed_time_selective_export = subparsers.add_parser(
        "fixed-120-selective-paper-candidate-export"
    )
    fixed_time_selective_export.add_argument("--config", type=Path, required=True)
    fixed_time_selective_export.add_argument(
        "--benchmark-run",
        type=Path,
        required=True,
    )
    fixed_time_selective_export.add_argument("--model-key", required=True)
    fixed_time_selective_export.add_argument(
        "--authorize-paper-only",
        action="store_true",
        help="authorize a selective exact-120 model only for paper evaluation",
    )
    fixed_time_reversal_run = subparsers.add_parser(
        "fixed-120-reversal-benchmark-run"
    )
    fixed_time_reversal_run.add_argument("--config", type=Path, required=True)
    fixed_time_reversal_export = subparsers.add_parser(
        "fixed-120-reversal-paper-candidate-export"
    )
    fixed_time_reversal_export.add_argument("--config", type=Path, required=True)
    fixed_time_reversal_export.add_argument(
        "--benchmark-run",
        type=Path,
        required=True,
    )
    fixed_time_reversal_export.add_argument("--model-key", required=True)
    fixed_time_reversal_export.add_argument(
        "--authorize-paper-only",
        action="store_true",
        help="authorize an exact-120 reversal model only for paper evaluation",
    )
    persistence_run = subparsers.add_parser("persistence-benchmark-run")
    persistence_run.add_argument("--config", type=Path, required=True)
    persistence_run.add_argument("--force", action="store_true")
    persistence_paper_export = subparsers.add_parser(
        "persistence-paper-candidate-export"
    )
    persistence_paper_export.add_argument("--config", type=Path, required=True)
    persistence_paper_export.add_argument(
        "--benchmark-run",
        type=Path,
        required=True,
    )
    persistence_paper_export.add_argument(
        "--candidate",
        help=(
            "explicit time-banded paper candidate; omit for the "
            "regime-robust recency candidate"
        ),
    )
    persistence_paper_export.add_argument(
        "--policy-benchmark-run",
        type=Path,
        help="causal frozen-policy evidence required by the frequency candidate",
    )
    persistence_paper_export.add_argument(
        "--freeze-root",
        type=Path,
        required=True,
    )
    persistence_paper_export.add_argument(
        "--runtime-output-root",
        type=Path,
        required=True,
    )
    persistence_paper_export.add_argument("--model-key", required=True)
    persistence_paper_export.add_argument(
        "--authorize-paper-only",
        action="store_true",
        help="authorize a non-production artifact exclusively for paper evaluation",
    )
    persistence_policy_run = subparsers.add_parser(
        "persistence-policy-benchmark-run"
    )
    persistence_policy_run.add_argument("--config", type=Path, required=True)
    frequency_policy_run = subparsers.add_parser(
        "frequency-policy-benchmark-run"
    )
    frequency_policy_run.add_argument("--config", type=Path, required=True)
    admission_run = subparsers.add_parser("admission-benchmark-run")
    admission_run.add_argument("--config", type=Path, required=True)
    residual_admission_run = subparsers.add_parser(
        "residual-admission-benchmark-run"
    )
    residual_admission_run.add_argument("--config", type=Path, required=True)
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--run", type=Path, required=True)
    serve_parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.command == "serve":
        serve(args.run.resolve(), args.port)
        return
    if args.command == "report":
        destination = generate_report(args.run.resolve())
        print(destination)
        return
    if args.command == "core-report":
        destination = generate_core_report(args.run.resolve())
        print(destination)
        return
    if args.command == "core-export-runtime":
        destination = export_runtime_model(
            freeze_dir=args.freeze,
            golden_features=args.golden_features,
            output_root=args.output_root,
            model_key=args.model_key,
        )
        print(f"runtime model: {destination}")
        return
    if args.command == "entry-benchmark-run":
        config = load_entry_benchmark_config(args.config)
        run_dir, benchmark = run_entry_benchmark(
            config,
            force=args.force,
        )
        print(f"report: {run_dir / 'report.html'}")
        print(
            "deployment-qualified: "
            + (
                ", ".join(benchmark["deployment_qualified_candidates"])
                if benchmark["deployment_qualified_candidates"]
                else "none"
            )
        )
        return
    if args.command == "oracle-book-benchmark-run":
        config = load_oracle_book_benchmark_config(args.config)
        run_dir, benchmark = run_oracle_book_benchmark(config)
        print(f"report: {run_dir / 'report.html'}")
        print(
            "status: "
            f"{benchmark['status']}; runtime export/deployment: disabled"
        )
        return
    if args.command == "price-aware-benchmark-run":
        config = load_price_aware_benchmark_config(args.config)
        run_dir, benchmark = run_price_aware_benchmark(
            config,
            force=args.force,
        )
        print(f"report: {run_dir / 'report.html'}")
        print(
            "selected development candidate: "
            f"{benchmark['selection']['selected_candidate'] or 'none'}"
        )
        print("runtime export/deployment: disabled")
        return
    if args.command == "loss-tail-benchmark-run":
        config = load_loss_tail_benchmark_config(args.config)
        run_dir, benchmark = run_loss_tail_benchmark(
            config,
            force=args.force,
        )
        print(f"report: {run_dir / 'report.html'}")
        print(
            "selected development candidate: "
            f"{benchmark['selection']['selected_candidate'] or 'none'}"
        )
        print("runtime export/deployment: disabled")
        return
    if args.command == "champion-vwap-calibration-run":
        config = load_champion_vwap_config(args.config)
        run_dir, benchmark = run_champion_vwap_benchmark(config)
        print(f"report: {run_dir / 'report.html'}")
        print(
            "promotion result: "
            f"{benchmark['promotion']['selected_candidate'] or 'champion retained'}"
        )
        print("runtime changed: false")
        return
    if args.command == "fixed-120-benchmark-run":
        config = load_fixed_time_accuracy_config(args.config)
        run_dir, benchmark = run_fixed_time_accuracy_benchmark(config)
        print(f"report: {run_dir / 'report.html'}")
        print(
            "paper-freeze qualified: "
            + str(benchmark["selection"]["paper_candidate_qualified"]).lower()
        )
        print("decision: exact 120 seconds; live capital: not authorized")
        return
    if args.command == "fixed-120-paper-candidate-export":
        if not args.authorize_paper_only:
            parser.error(
                "fixed-120-paper-candidate-export requires "
                "--authorize-paper-only"
            )
        config = load_fixed_time_accuracy_config(args.config)
        freeze_dir, runtime_dir, manifest = (
            freeze_and_export_fixed_time_paper_candidate(
                config=config,
                benchmark_run=args.benchmark_run,
                model_key=args.model_key,
                authorization=FIXED_TIME_PAPER_AUTHORIZATION,
            )
        )
        print(f"freeze: {freeze_dir}")
        print(f"runtime model: {runtime_dir}")
        print(
            "scope: paper_only; production-qualified: "
            f"{str(manifest['production_qualified']).lower()}"
        )
        return
    if args.command == "fixed-120-selective-benchmark-run":
        config = load_fixed_time_selective_config(args.config)
        run_dir, benchmark = run_fixed_time_selective_benchmark(config)
        print(f"report: {run_dir / 'report.html'}")
        print(
            "development candidate: "
            f"{benchmark['selection']['selected_candidate'] or 'none'}"
        )
        print("evidence: consumed development; July 29 onward excluded")
        print("live capital: not authorized")
        return
    if args.command == "fixed-120-selective-paper-candidate-export":
        if not args.authorize_paper_only:
            parser.error(
                "fixed-120-selective-paper-candidate-export requires "
                "--authorize-paper-only"
            )
        config = load_fixed_time_selective_config(args.config)
        freeze_dir, runtime_dir, manifest = (
            freeze_and_export_fixed_time_selective_paper_candidate(
                config=config,
                benchmark_run=args.benchmark_run,
                model_key=args.model_key,
                authorization=FIXED_TIME_SELECTIVE_PAPER_AUTHORIZATION,
            )
        )
        print(f"freeze: {freeze_dir}")
        print(f"runtime model: {runtime_dir}")
        print(
            "scope: paper_only; production-qualified: "
            f"{str(manifest['production_qualified']).lower()}"
        )
        print("fresh forward evidence: required from July 29, 2026")
        return
    if args.command == "fixed-120-reversal-benchmark-run":
        config = load_fixed_time_reversal_config(args.config)
        run_dir, benchmark = run_fixed_time_reversal_benchmark(config)
        print(f"report: {run_dir / 'report.html'}")
        print(
            "development candidate: "
            f"{benchmark['selection']['selected_candidate'] or 'none'}"
        )
        print("decision: exact 120 seconds; evidence ends July 28, 2026")
        print("live capital: not authorized")
        return
    if args.command == "fixed-120-reversal-paper-candidate-export":
        if not args.authorize_paper_only:
            parser.error(
                "fixed-120-reversal-paper-candidate-export requires "
                "--authorize-paper-only"
            )
        config = load_fixed_time_reversal_config(args.config)
        freeze_dir, runtime_dir, manifest = (
            freeze_and_export_fixed_time_reversal_paper_candidate(
                config=config,
                benchmark_run=args.benchmark_run,
                model_key=args.model_key,
                authorization=FIXED_TIME_REVERSAL_PAPER_AUTHORIZATION,
            )
        )
        print(f"freeze: {freeze_dir}")
        print(f"runtime model: {runtime_dir}")
        print(
            "scope: paper_only; production-qualified: "
            f"{str(manifest['production_qualified']).lower()}"
        )
        print("fresh forward evidence: required from July 29, 2026")
        return
    if args.command == "persistence-benchmark-run":
        config = load_persistence_benchmark_config(args.config)
        run_dir, benchmark = run_persistence_benchmark(
            config,
            force=args.force,
        )
        print(f"report: {run_dir / 'report.html'}")
        print(
            "training finalist: "
            + str(benchmark["training_selection"]["finalist"] or "none")
        )
        print("holdout labels/features: not accessed")
        print("holdout book quality: only if separately recorded in run evidence")
        print("runtime: unchanged")
        return
    if args.command == "persistence-paper-candidate-export":
        if not args.authorize_paper_only:
            parser.error(
                "persistence-paper-candidate-export requires "
                "--authorize-paper-only"
            )
        freeze_dir, runtime_dir, manifest = run_paper_candidate_export(
            config_path=args.config,
            benchmark_run=args.benchmark_run,
            candidate=args.candidate,
            policy_benchmark_run=args.policy_benchmark_run,
            freeze_root=args.freeze_root,
            runtime_output_root=args.runtime_output_root,
            model_key=args.model_key,
            authorization=PAPER_ONLY_AUTHORIZATION,
        )
        print(f"freeze: {freeze_dir}")
        print(f"runtime model: {runtime_dir}")
        print(
            "scope: paper_only; production-qualified: "
            f"{str(manifest['production_qualified']).lower()}"
        )
        return
    if args.command == "persistence-policy-benchmark-run":
        config = load_saved_policy_benchmark_config(args.config)
        run_dir, benchmark = run_saved_policy_benchmark(config)
        print(f"report: {run_dir / 'report.html'}")
        print(
            "benchmark passed: "
            + (
                ", ".join(benchmark["benchmark_passed_candidates"])
                if benchmark["benchmark_passed_candidates"]
                else "none"
            )
        )
        print(f"winner: {benchmark['winner'] or 'none'}")
        print("evidence: non-independent development validation")
        print("runtime: unchanged")
        return
    if args.command == "frequency-policy-benchmark-run":
        config = load_frequency_policy_benchmark_config(args.config)
        run_dir, benchmark = run_frequency_policy_benchmark(config)
        print(f"report: {run_dir / 'report.html'}")
        print(
            "frequency-qualified: "
            + (
                ", ".join(benchmark["benchmark_passed_candidates"])
                if benchmark["benchmark_passed_candidates"]
                else "none"
            )
        )
        print(f"winner: {benchmark['winner'] or 'none'}")
        print("policy: one anchor-fold threshold vector applied to all validation folds")
        print("timing: diagnostic only")
        print("evidence: non-independent development validation")
        print("runtime: unchanged")
        return
    if args.command == "admission-benchmark-run":
        config = load_admission_benchmark_config(args.config)
        run_dir, benchmark = run_admission_benchmark(config)
        print(f"report: {run_dir / 'report.html'}")
        print(
            "development benchmark passed: "
            + (
                ", ".join(benchmark["benchmark_passed_candidates"])
                if benchmark["benchmark_passed_candidates"]
                else "none"
            )
        )
        print(f"winner: {benchmark['winner'] or 'none'}")
        print("selector evaluation: rolling folds 2, 3, 4")
        print("deployment: not qualified; only three selector validation folds")
        print("holdout/database/runtime: untouched")
        return
    if args.command == "residual-admission-benchmark-run":
        config = load_residual_admission_config(args.config)
        run_dir, benchmark = run_residual_admission_benchmark(config)
        print(f"report: {run_dir / 'report.html'}")
        print(
            "development benchmark passed: "
            + (
                ", ".join(benchmark["benchmark_passed_candidates"])
                if benchmark["benchmark_passed_candidates"]
                else "none"
            )
        )
        print(f"winner: {benchmark['winner'] or 'none'}")
        print("selector evaluation: rolling folds 2 through 6")
        print("holdout/runtime: unchanged; post-freeze evidence required")
        return
    if args.command.startswith("core-"):
        run_core_command(args)
        return

    config = load_config(args.config)
    if args.command == "extract":
        print(json.dumps(extract_source(config, force=args.force), indent=2))
    elif args.command == "features":
        print(json.dumps(build_features(config, force=args.force), indent=2))
    elif args.command == "train":
        run_dir, metrics = train_models(config)
        destination = generate_report(run_dir, metrics)
        print(f"report: {destination}")
    elif args.command == "run":
        extract_source(config, force=args.force)
        build_features(config, force=args.force)
        run_dir, metrics = train_models(config)
        destination = generate_report(run_dir, metrics)
        print(f"report: {destination}")


def run_core_command(args: argparse.Namespace) -> None:
    config = load_core_config(args.config)
    if args.command == "core-snapshot-residual-source":
        manifest = snapshot_residual_admission_source(
            config,
            args.source_dir,
        )
        print(json.dumps(manifest, indent=2))
    elif args.command == "core-extract":
        manifest = extract_core_source(config, args.scope, force=args.force)
        print(json.dumps(manifest, indent=2))
    elif args.command == "core-features":
        metadata = build_core_features(config, args.scope, force=args.force)
        print(json.dumps(metadata, indent=2))
    elif args.command == "core-develop":
        run_dir, freeze_dir, metrics = develop_core_models(config)
        destination = generate_core_report(run_dir, metrics)
        print(f"report: {destination}")
        print(f"freeze: {freeze_dir if freeze_dir is not None else 'blocked'}")
    elif args.command == "core-evaluate-holdout":
        run_dir, metrics = evaluate_core_holdout(config, args.freeze.resolve())
        destination = generate_core_report(run_dir, metrics)
        print(f"report: {destination}")
    elif args.command == "core-run":
        extract_core_source(config, "pre_holdout", force=args.force)
        build_core_features(config, "pre_holdout", force=args.force)
        run_dir, freeze_dir, metrics = develop_core_models(config)
        generate_core_report(run_dir, metrics)
        if freeze_dir is None:
            print(f"report: {run_dir / 'report.html'}")
            print("holdout: not accessed because pre-holdout gates did not pass")
            return
        extract_core_source(config, "holdout", force=args.force)
        build_core_features(config, "holdout", force=args.force)
        run_dir, metrics = evaluate_core_holdout(config, freeze_dir)
        destination = generate_core_report(run_dir, metrics)
        print(f"report: {destination}")
    elif args.command == "core-training-readiness":
        json_path, markdown_path, payload = prepare_training_readiness(
            config,
            execution_output_dir=args.execution_output.resolve(),
            output_dir=args.output_dir.resolve(),
            force=args.force,
        )
        print(f"readiness JSON: {json_path}")
        print(f"readiness report: {markdown_path}")
        print(
            "core + oracle complete markets: "
            f"{payload['totals']['core_oracle_complete_markets']:,}"
        )
        print(
            "book-complete 11-point markets: "
            f"{payload['totals']['book_complete_11_point_markets']:,}"
        )


def serve(directory: Path, port: int) -> None:
    if not (directory / "report.html").exists():
        raise FileNotFoundError(f"{directory} does not contain report.html")
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"serving {directory / 'report.html'} at http://127.0.0.1:{port}/report.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
