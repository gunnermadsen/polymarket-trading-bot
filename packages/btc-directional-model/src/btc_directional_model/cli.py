from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import load_config
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
