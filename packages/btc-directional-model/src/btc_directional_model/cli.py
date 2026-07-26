from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import load_config
from .core_config import load_core_config
from .core_extract import extract_core_source
from .core_features import build_core_features
from .core_report import generate_core_report
from .core_training import develop_core_models, evaluate_core_holdout
from .extract import extract_source
from .features import build_features
from .report import generate_report
from .train import train_models


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
    core_develop = subparsers.add_parser("core-develop")
    core_develop.add_argument("--config", type=Path, required=True)
    core_evaluate = subparsers.add_parser("core-evaluate-holdout")
    core_evaluate.add_argument("--config", type=Path, required=True)
    core_evaluate.add_argument("--freeze", type=Path, required=True)
    core_run = subparsers.add_parser("core-run")
    core_run.add_argument("--config", type=Path, required=True)
    core_run.add_argument("--force", action="store_true")
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
    if args.command == "core-extract":
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
