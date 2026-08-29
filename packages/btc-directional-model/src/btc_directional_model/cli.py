from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .admission_benchmark import run_admission_benchmark
from .admission_config import load_admission_benchmark_config
from .asymmetric_book_admission_benchmark import (
    load_and_run_book_admission_benchmark,
)
from .asymmetric_d4_side_benchmark import (
    load_and_run_d4_side_calibration_benchmark,
)
from .asymmetric_incumbent_benchmark import (
    load_and_run_incumbent_calibration_benchmark,
)
from .asymmetric_training_readiness import (
    load_new_day_readiness_contract,
    prepare_asymmetric_training_readiness,
    prepare_new_day_training_readiness,
)
from .asymmetric_value_benchmark import run_asymmetric_value_benchmark
from .asymmetric_value_config import load_asymmetric_value_config
from .benchmark_config import load_entry_benchmark_config
from .capacity_training import extract_capacity_evidence, run_capacity_training
from .capacity_training_config import load_capacity_training_config
from .chainlink_oi_benchmark import run_chainlink_oi_benchmark
from .chainlink_oi_config import load_chainlink_oi_benchmark_config
from .chainlink_oi_forward_score import run_chainlink_oi_forward_score
from .chainlink_oi_paper_export import (
    PAPER_ONLY_AUTHORIZATION as CHAINLINK_OI_PAPER_AUTHORIZATION,
)
from .chainlink_oi_paper_export import export_chainlink_oi_paper_candidates
from .champion_vwap_benchmark import run_champion_vwap_benchmark
from .champion_vwap_config import load_champion_vwap_config
from .config import load_config
from .core_config import load_core_config
from .core_extract import extract_core_source, snapshot_residual_admission_source
from .core_features import build_core_features
from .core_report import generate_core_report
from .core_training import develop_core_models, evaluate_core_holdout
from .counterfactual_twap_state_tournament import (
    load_config as load_counterfactual_twap_state_config,
)
from .counterfactual_twap_state_tournament import (
    run_tournament as run_counterfactual_twap_state_tournament,
)
from .causal_twap_attribution_tournament import (
    load_config as load_causal_twap_attribution_config,
)
from .causal_twap_attribution_tournament import (
    run_tournament as run_causal_twap_attribution_tournament,
)
from .early_entry_settlement_consensus_tournament import (
    load_config as load_early_entry_consensus_config,
)
from .early_entry_settlement_consensus_tournament import (
    run_tournament as run_early_entry_consensus_tournament,
)
from .early_value_benchmark import run_early_value_benchmark
from .early_value_config import load_early_value_config
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
from .runtime_export import (
    export_runtime_model,
    promote_runtime_model_for_live_pilot,
)
from .settlement_bridge_residual_tournament import (
    load_config as load_settlement_bridge_config,
)
from .settlement_bridge_residual_tournament import (
    run_tournament as run_settlement_bridge_tournament,
)
from .spot_l2_chainlink_benchmark import run_spot_l2_chainlink_benchmark
from .spot_l2_chainlink_config import load_spot_l2_chainlink_config
from .train import train_models
from .training_readiness import prepare_training_readiness
from .twap60_challenger_tournament import (
    load_config as load_twap60_tournament_config,
)
from .twap60_challenger_tournament import (
    run_tournament as run_twap60_tournament,
)


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
    live_pilot_export = subparsers.add_parser("core-promote-runtime-live-pilot")
    live_pilot_export.add_argument("--source-runtime", type=Path, required=True)
    live_pilot_export.add_argument("--output-root", type=Path, required=True)
    live_pilot_export.add_argument("--model-key", required=True)
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
    early_value_run = subparsers.add_parser("early-value-benchmark-run")
    early_value_run.add_argument("--config", type=Path, required=True)
    early_value_run.add_argument("--force", action="store_true")
    asymmetric_value_run = subparsers.add_parser(
        "asymmetric-value-benchmark-run"
    )
    asymmetric_value_run.add_argument("--config", type=Path, required=True)
    asymmetric_value_run.add_argument("--force", action="store_true")
    incumbent_calibration_run = subparsers.add_parser(
        "asymmetric-incumbent-calibration-run"
    )
    incumbent_calibration_run.add_argument("--config", type=Path, required=True)
    incumbent_calibration_run.add_argument("--force", action="store_true")
    book_admission_run = subparsers.add_parser(
        "asymmetric-book-admission-run"
    )
    book_admission_run.add_argument("--config", type=Path, required=True)
    book_admission_run.add_argument("--force", action="store_true")
    d4_side_calibration_run = subparsers.add_parser(
        "asymmetric-d4-side-calibration-run"
    )
    d4_side_calibration_run.add_argument("--config", type=Path, required=True)
    d4_side_calibration_run.add_argument("--force", action="store_true")
    asymmetric_readiness = subparsers.add_parser(
        "asymmetric-training-readiness"
    )
    asymmetric_readiness.add_argument("--config", type=Path, required=True)
    asymmetric_readiness.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    new_day_readiness = subparsers.add_parser(
        "asymmetric-new-day-readiness"
    )
    new_day_readiness.add_argument("--config", type=Path, required=True)
    new_day_readiness.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
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
    for name in ("capacity-evidence-extract", "capacity-training-run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--force", action="store_true")
    twap60_run = subparsers.add_parser("twap60-challenger-tournament-run")
    twap60_run.add_argument("--config", type=Path, required=True)
    twap60_run.add_argument("--force", action="store_true")
    settlement_bridge_run = subparsers.add_parser(
        "settlement-bridge-residual-tournament-run"
    )
    settlement_bridge_run.add_argument("--config", type=Path, required=True)
    settlement_bridge_run.add_argument("--force", action="store_true")
    causal_twap_attribution = subparsers.add_parser(
        "causal-twap-attribution-tournament-run"
    )
    causal_twap_attribution.add_argument("--config", type=Path, required=True)
    causal_twap_attribution.add_argument("--force", action="store_true")
    counterfactual_twap_state = subparsers.add_parser(
        "counterfactual-twap-state-tournament-run"
    )
    counterfactual_twap_state.add_argument("--config", type=Path, required=True)
    counterfactual_twap_state.add_argument("--force", action="store_true")
    early_entry_consensus_run = subparsers.add_parser(
        "early-entry-settlement-consensus-tournament-run"
    )
    early_entry_consensus_run.add_argument("--config", type=Path, required=True)
    chainlink_oi_run = subparsers.add_parser(
        "chainlink-oi-champion-benchmark-run"
    )
    chainlink_oi_run.add_argument("--config", type=Path, required=True)
    chainlink_oi_export = subparsers.add_parser(
        "chainlink-oi-paper-candidates-export"
    )
    chainlink_oi_export.add_argument("--config", type=Path, required=True)
    chainlink_oi_export.add_argument(
        "--benchmark-run",
        type=Path,
        required=True,
    )
    chainlink_oi_export.add_argument(
        "--freeze-root",
        type=Path,
        required=True,
    )
    chainlink_oi_export.add_argument(
        "--runtime-output-root",
        type=Path,
        required=True,
    )
    chainlink_oi_export.add_argument(
        "--authorize-paper-only",
        action="store_true",
        help="authorize all three challengers exclusively for paper evaluation",
    )
    chainlink_oi_forward = subparsers.add_parser(
        "chainlink-oi-forward-score"
    )
    chainlink_oi_forward.add_argument("--config", type=Path, required=True)
    chainlink_oi_forward.add_argument("--start", type=_parse_utc_day, required=True)
    chainlink_oi_forward.add_argument("--end", type=_parse_utc_day, required=True)
    chainlink_oi_forward.add_argument("--output-root", type=Path, required=True)
    chainlink_oi_forward.add_argument("--runtime-model-root", type=Path)
    spot_l2_chainlink_run = subparsers.add_parser(
        "spot-l2-chainlink-candles-benchmark-run"
    )
    spot_l2_chainlink_run.add_argument("--config", type=Path, required=True)
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
    if args.command == "core-promote-runtime-live-pilot":
        destination = promote_runtime_model_for_live_pilot(
            source_runtime=args.source_runtime,
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
    if args.command == "early-value-benchmark-run":
        config = load_early_value_config(args.config)
        run_dir, result = run_early_value_benchmark(config, force=args.force)
        print(f"report: {run_dir / 'benchmark-report.md'}")
        print(f"selected probability model: {result['training']['selected_profile']}")
        print("runtime/trading pipeline changes: none")
        return
    if args.command == "asymmetric-value-benchmark-run":
        config = load_asymmetric_value_config(args.config)
        run_dir, result = run_asymmetric_value_benchmark(
            config,
            force=args.force,
        )
        print(f"report: {run_dir / 'benchmark-report.md'}")
        print(f"selected value hunter: {result['selection']['selected_key']}")
        print(f"evaluation status: {result['evaluation']['status']}")
        print("runtime/trading pipeline changes: none")
        return
    if args.command == "asymmetric-incumbent-calibration-run":
        run_dir, result = load_and_run_incumbent_calibration_benchmark(
            args.config,
            force=args.force,
        )
        print(f"report: {run_dir / 'benchmark-report.md'}")
        print(f"status: {result['status']}")
        print(
            "selected challenger: "
            f"{result.get('selected_candidate_id') or 'incumbent retained'}"
        )
        print(f"paper artifact: {result.get('paper_artifact') or 'none'}")
        print("source process changed: false; live capital: not authorized")
        return
    if args.command == "asymmetric-book-admission-run":
        run_dir, result = load_and_run_book_admission_benchmark(
            args.config,
            force=args.force,
        )
        print(f"report: {run_dir / 'benchmark-report.md'}")
        print(f"status: {result['status']}")
        print(
            "selected challenger: "
            f"{result.get('selected_candidate_id') or 'incumbent retained'}"
        )
        print(
            "batch-forward artifact: "
            f"{result.get('batch_forward_artifact') or 'none'}"
        )
        print("runtime deployable: false; source process changed: false")
        return
    if args.command == "asymmetric-d4-side-calibration-run":
        run_dir, result = load_and_run_d4_side_calibration_benchmark(
            args.config,
            force=args.force,
        )
        print(f"report: {run_dir / 'benchmark-report.md'}")
        print(f"projected PnL: {run_dir / 'projected-pnl.md'}")
        print(f"status: {result['status']}")
        print(
            "probability-selected challenger: "
            f"{result.get('probability_selected_candidate_id') or 'none'}"
        )
        print("qualification eligible: false; source process changed: false")
        return
    if args.command == "asymmetric-training-readiness":
        config = load_asymmetric_value_config(args.config)
        destination, payload = prepare_asymmetric_training_readiness(
            config,
            output_dir=args.output_dir,
        )
        print(f"readiness manifest: {destination}")
        print(f"ready: {str(payload['ready']).lower()}")
        print("external SSD required: false")
        return
    if args.command == "asymmetric-new-day-readiness":
        contract = load_new_day_readiness_contract(args.config)
        destination, payload = prepare_new_day_training_readiness(
            contract,
            package_root=Path(__file__).resolve().parents[2],
            output_dir=args.output_dir,
        )
        print(f"readiness manifest: {destination}")
        print(f"status: {payload['status']}")
        print(f"ready: {str(payload['ready']).lower()}")
        print(
            "external archive status: "
            f"{payload['external_archive']['status']}"
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
    if args.command == "capacity-evidence-extract":
        config = load_capacity_training_config(args.config)
        manifest = extract_capacity_evidence(config, force=args.force)
        print(f"evidence manifest: {config.evidence / 'manifest.json'}")
        print(f"rows: {manifest['rows']}")
        print("trading processes changed: false")
        return
    if args.command == "capacity-training-run":
        config = load_capacity_training_config(args.config)
        run_dir, result = run_capacity_training(config, force=args.force)
        print(f"report: {run_dir / 'training-report.md'}")
        print(f"status: {result['status']}")
        print("runtime/trading processes changed: false")
        return
    if args.command == "twap60-challenger-tournament-run":
        config = load_twap60_tournament_config(args.config)
        run_dir, result = run_twap60_tournament(config, force_extract=args.force)
        print(f"report: {run_dir / 'report.md'}")
        print(f"status: {result['tournament']['selection']['status']}")
        print(
            "provisional challenger: "
            f"{result['tournament']['selection']['provisional_challenger'] or 'none'}"
        )
        print("runtime/database/trading processes changed: false")
        return
    if args.command == "settlement-bridge-residual-tournament-run":
        config = load_settlement_bridge_config(args.config)
        run_dir, result = run_settlement_bridge_tournament(
            config, force_data=args.force
        )
        print(f"report: {run_dir / 'report.md'}")
        print(f"conclusion: {result['conclusion']}")
        print(f"deployment status: {result['deployment_status']}")
        print("runtime/database/trading processes changed: false")
        return
    if args.command == "causal-twap-attribution-tournament-run":
        config = load_causal_twap_attribution_config(args.config)
        run_dir, result = run_causal_twap_attribution_tournament(
            config, force_extract=args.force
        )
        print(f"report: {run_dir / 'report.md'}")
        print(f"status: {result['selection']['status']}")
        print(
            "provisional candidate: "
            f"{result['selection']['provisional_candidate']}"
        )
        print("runtime/database/trading processes changed: false")
        return
    if args.command == "counterfactual-twap-state-tournament-run":
        config = load_counterfactual_twap_state_config(args.config)
        run_dir, result = run_counterfactual_twap_state_tournament(
            config, force_extract=args.force
        )
        print(f"report: {run_dir / 'report.md'}")
        print(f"status: {result['selection']['status']}")
        print(
            "provisional candidate: "
            f"{result['selection']['provisional_candidate']}"
        )
        print("runtime/database/trading processes changed: false")
        return
    if args.command == "early-entry-settlement-consensus-tournament-run":
        config = load_early_entry_consensus_config(args.config)
        run_dir, result = run_early_entry_consensus_tournament(config)
        print(f"report: {run_dir / 'report.md'}")
        print(f"qualification status: {result['qualification_status']}")
        print(f"top candidate: {result['ranking'][0]}")
        print("runtime/database/trading processes changed: false")
        return
    if args.command == "chainlink-oi-champion-benchmark-run":
        config = load_chainlink_oi_benchmark_config(args.config)
        run_dir, benchmark = run_chainlink_oi_benchmark(config)
        print(f"report: {run_dir / 'report.html'}")
        print(f"status: {benchmark['status']}")
        print(
            "selected candidate: "
            f"{benchmark.get('selected_candidate') or 'none'}"
        )
        print("runtime/deployment: unchanged")
        return
    if args.command == "chainlink-oi-forward-score":
        config = load_chainlink_oi_benchmark_config(args.config)
        run_dir, result = run_chainlink_oi_forward_score(
            config,
            range_start=args.start,
            range_end=args.end,
            output_root=args.output_root,
            runtime_model_root=args.runtime_model_root,
        )
        print(f"report: {run_dir / 'report.html'}")
        print(f"status: {result['status']}")
        print(f"blockers: {len(result['blockers'])}")
        print("training/process/database changes: none")
        return
    if args.command == "chainlink-oi-paper-candidates-export":
        if not args.authorize_paper_only:
            parser.error(
                "chainlink-oi-paper-candidates-export requires "
                "--authorize-paper-only"
            )
        config = load_chainlink_oi_benchmark_config(args.config)
        results = export_chainlink_oi_paper_candidates(
            config=config,
            benchmark_run=args.benchmark_run,
            freeze_root=args.freeze_root,
            runtime_output_root=args.runtime_output_root,
            authorization=CHAINLINK_OI_PAPER_AUTHORIZATION,
        )
        for result in results:
            print(f"candidate: {result.candidate}")
            print(f"freeze: {result.freeze_dir}")
            print(f"runtime model: {result.runtime_dir}")
            print(f"model SHA-256: {result.model_sha256}")
            print(f"feature schema SHA-256: {result.feature_schema_sha256}")
        print("scope: paper_only; production-qualified: false")
        return
    if args.command == "spot-l2-chainlink-candles-benchmark-run":
        config = load_spot_l2_chainlink_config(args.config)
        run_dir, _report = run_spot_l2_chainlink_benchmark(config)
        print(f"report: {run_dir / 'benchmark-report.md'}")
        print("advancement: paper-only forward validation at most")
        print("runtime/deployment: unchanged")
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


def _parse_utc_day(value: str) -> datetime:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected UTC date in YYYY-MM-DD form") from error
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("expected UTC date in YYYY-MM-DD form")
    return datetime.combine(parsed, time.min, UTC)


if __name__ == "__main__":
    main()
