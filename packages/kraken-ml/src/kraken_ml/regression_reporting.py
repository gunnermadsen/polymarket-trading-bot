from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .reporting import write_json_artifact, write_text_artifact


def _escape(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.4f}"
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_No rows._"
    rendered = [
        "| " + " | ".join(_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    rendered.extend("| " + " | ".join(_escape(value) for value in row) + " |" for row in rows)
    return "\n".join(rendered)


def _gate_rows(gates: Mapping[str, Any] | None) -> list[tuple[Any, ...]]:
    if not gates:
        return []
    checks = gates.get("checks", {})
    return [
        (
            name.replace("_", " ").capitalize(),
            check.get("pass"),
            check.get("actual"),
            check.get("required"),
        )
        for name, check in checks.items()
    ]


def _candidate_rows(report: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    candidates = report.get("candidate_aggregates", [])
    return [
        (
            candidate.get("candidate_id"),
            candidate.get("horizon_bars"),
            candidate.get("model"),
            candidate.get("feature_set"),
            candidate.get("pooled_regression", {}).get("pooled", {}).get("mae_bps"),
            candidate.get("pooled_regression", {}).get("pooled", {}).get("rmse_bps"),
            candidate.get("pooled_regression", {}).get("pooled", {}).get("r2"),
            candidate.get("pooled_regression", {}).get("pooled", {}).get("spearman"),
            candidate.get("positive_nominal_folds"),
            candidate.get("pooled_economics", {}).get("net_expectancy_bps"),
            candidate.get("pooled_stress", {}).get("net_expectancy_bps"),
            candidate.get("median_fold_stress_expectancy_bps"),
            candidate.get("pooled_economics", {}).get("trades"),
            candidate.get("qualified"),
        )
        for candidate in candidates
    ]


def _fold_rows(report: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    selected = report.get("selected") or {}
    candidate_id = selected.get("candidate_id")
    return [
        (
            result.get("fold"),
            result.get("policy", {}).get("expected_return_hurdle_bps"),
            result.get("policy", {}).get("directional_advantage_bps"),
            result.get("economics", {}).get("trades"),
            result.get("economics", {}).get("net_expectancy_bps"),
            result.get("cost_stress", {}).get("net_expectancy_bps"),
        )
        for result in report.get("fold_results", [])
        if result.get("candidate_id") == candidate_id
    ]


def _regression_rows(report: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    selected = report.get("selected") or {}
    candidate_id = selected.get("candidate_id")
    rows: list[tuple[Any, ...]] = []
    for result in report.get("fold_results", []):
        if result.get("candidate_id") != candidate_id:
            continue
        diagnostics = result.get("regression", {})
        long = diagnostics.get("long", {})
        short = diagnostics.get("short", {})
        rows.append(
            (
                result.get("fold"),
                long.get("mae_bps"),
                long.get("rmse_bps"),
                long.get("r2"),
                long.get("spearman"),
                short.get("mae_bps"),
                short.get("rmse_bps"),
                short.get("r2"),
                short.get("spearman"),
            )
        )
    return rows


def _candidate_fee_rows(report: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for candidate in report.get("candidate_aggregates", []):
        candidate_id = candidate.get("candidate_id")
        scenarios = candidate.get("fee_counterfactuals", {})
        for scenario in _fee_scenario_rows(scenarios):
            if not scenario[2]:
                continue
            rows.append((candidate_id, *scenario))
    return rows


def _fee_scenario_rows(scenarios: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            name,
            metrics.get("round_trip_fee_bps"),
            metrics.get("trades"),
            metrics.get("net_expectancy_bps"),
            metrics.get("bootstrap_95_lower_bps"),
            metrics.get("profit_factor"),
        )
        for name, metrics in scenarios.items()
    ]


def _oi_rows(report: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for candidate in report.get("candidate_aggregates", []):
        qualification = candidate.get("oi_qualification")
        if not qualification:
            continue
        checks = qualification.get("checks", {})
        paired = checks.get("paired_nominal_fold_wins", {})
        stress = checks.get("no_lower_aggregate_stressed_expectancy", {})
        rows.append(
            (
                candidate.get("candidate_id"),
                paired.get("actual"),
                paired.get("required"),
                stress.get("actual"),
                qualification.get("pass"),
            )
        )
    return rows


def _identity_lines(report: Mapping[str, Any]) -> list[str]:
    hashes = report.get("hashes", {})
    lines = [
        f"- Run: `{_escape(report.get('run_id'))}`",
        f"- Generated: {_escape(report.get('generated_at'))}",
    ]
    for label, key in (
        ("Configuration", "config_sha256"),
        ("Source snapshot", "source_sha256"),
        ("Code", "code_sha256"),
    ):
        if hashes.get(key):
            lines.append(f"- {label} SHA-256: `{_escape(hashes[key])}`")
    return lines


def render_regression_development_report(report: Mapping[str, Any]) -> str:
    selected = report.get("selected") or {}
    gates = selected.get("gates") or {}
    confirmation = report.get("final_confirmation") or {}
    holdout = report.get("holdout", {})
    funding = report.get("funding_readiness", {})
    coverage = funding.get("coverage", {})
    provenance = funding.get("provenance", {})
    fees = report.get("fee_assumptions", {})
    execution_realism = report.get("execution_realism", {})
    classifier = report.get("historical_classifier_control", {})
    sections = [
        "# Kraken Futures Net-Expectancy Development Benchmark",
        (
            "## Verdict\n\n"
            f"**{_escape(report.get('verdict', 'not_assessed')).upper()}**"
        ),
        "## Reproducibility\n\n" + "\n".join(_identity_lines(report)),
        (
            "## Training prerequisites\n\n"
            + _table(
                ("Item", "Value"),
                [
                    ("Funding complete", funding.get("pass")),
                    ("Funding rows", coverage.get("non_null_rows")),
                    ("Funding first", coverage.get("first_timestamp")),
                    ("Funding last", coverage.get("last_timestamp")),
                    ("Funding provenance verified", funding.get("provenance_verified")),
                    ("Funding import id", funding.get("pinned_import_id")),
                    ("Funding binding SHA-256", provenance.get("binding_sha256")),
                    (
                        "Funding manifest SHA-256",
                        provenance.get("manifest", {}).get("sha256"),
                    ),
                    ("Immutable source objects", len(provenance.get("sources", []))),
                    (
                        "Verified published objects",
                        len(provenance.get("published_objects", [])),
                    ),
                    ("Funding used as feature", funding.get("funding_used_as_feature")),
                    ("Taker fee per side bps", fees.get("taker_bps_per_side")),
                    ("Maker fee per side bps", fees.get("maker_bps_per_side")),
                ],
            )
        ),
        (
            "## Execution realism\n\n"
            + _table(
                ("Item", "Value"),
                [
                    ("Status", execution_realism.get("status")),
                    (
                        "Feature available at entry",
                        execution_realism.get("feature_available_at_equals_entry_at"),
                    ),
                    (
                        "Inference/routing latency seconds",
                        execution_realism.get("inference_and_routing_latency_seconds"),
                    ),
                    ("Entry price", execution_realism.get("entry_price")),
                    (
                        "Deployable-edge claim allowed",
                        execution_realism.get("deployable_edge_claim_allowed"),
                    ),
                    ("Required follow-up", execution_realism.get("required_follow_up")),
                ],
            )
        ),
        (
            "## Candidate comparison\n\n"
            + _table(
                (
                    "Candidate",
                    "Bars",
                    "Model",
                    "Features",
                    "OOF MAE bps",
                    "OOF RMSE bps",
                    "OOF R²",
                    "OOF Spearman",
                    "Positive folds",
                    "Net bps/trade",
                    "Stress bps/trade",
                    "Median fold stress",
                    "Trades",
                    "Qualified",
                ),
                _candidate_rows(report),
            )
        ),
        (
            "## Selected candidate\n\n"
            + _table(
                ("Item", "Value"),
                [
                    ("Candidate", selected.get("candidate_id")),
                    ("Horizon bars", selected.get("horizon_bars")),
                    ("Model", selected.get("model")),
                    ("Feature set", selected.get("feature_set")),
                    ("Diagnostic only", selected.get("diagnostic_only")),
                ],
            )
        ),
        (
            "## Selected candidate folds\n\n"
            + _table(
                (
                    "Fold",
                    "Hurdle bps",
                    "Advantage bps",
                    "Trades",
                    "Net bps/trade",
                    "Stress bps/trade",
                ),
                _fold_rows(report),
            )
        ),
        (
            "## Selected candidate regression diagnostics\n\n"
            + _table(
                (
                    "Fold",
                    "Long MAE",
                    "Long RMSE",
                    "Long R²",
                    "Long Spearman",
                    "Short MAE",
                    "Short RMSE",
                    "Short R²",
                    "Short Spearman",
                ),
                _regression_rows(report),
            )
        ),
        (
            "## Active-candidate fixed-action fee counterfactuals\n\n"
            + _table(
                (
                    "Candidate",
                    "Scenario",
                    "Round-trip fee bps",
                    "Trades",
                    "Net bps/trade",
                    "95% lower",
                    "Profit factor",
                ),
                _candidate_fee_rows(report),
            )
        ),
        (
            "## Open-interest qualification\n\n"
            + _table(
                (
                    "Candidate",
                    "Paired wins",
                    "Required",
                    "Stressed bps/trade",
                    "Qualified",
                ),
                _oi_rows(report),
            )
        ),
        (
            "## Development gates\n\n"
            + _table(("Gate", "Pass", "Actual", "Required"), _gate_rows(gates))
        ),
        (
            "## Final pre-holdout confirmation\n\n"
            + _table(
                ("Item", "Value"),
                [
                    ("Status", confirmation.get("status", "not_run")),
                    ("Passed", confirmation.get("gates", {}).get("pass")),
                    (
                        "Net bps/trade",
                        confirmation.get("economics", {}).get("net_expectancy_bps"),
                    ),
                    (
                        "Stress bps/trade",
                        confirmation.get("cost_stress", {}).get("net_expectancy_bps"),
                    ),
                ],
            )
        ),
        (
            "## Final confirmation gates\n\n"
            + _table(
                ("Gate", "Pass", "Actual", "Required"),
                _gate_rows(confirmation.get("gates")),
            )
        ),
        (
            "## Historical classifier control\n\n"
            + _table(
                ("Item", "Value"),
                [
                    ("Role", classifier.get("role")),
                    ("Run", classifier.get("run_id")),
                    (
                        "Eligible for selection",
                        classifier.get("eligible_for_regression_candidate_selection"),
                    ),
                    ("Development gates passed", classifier.get("gates", {}).get("pass")),
                    ("Holdout status", classifier.get("holdout_status")),
                ],
            )
        ),
        (
            "## Holdout state\n\n"
            + _table(
                ("Item", "Value"),
                [
                    ("Status", holdout.get("status")),
                    ("Opened", holdout.get("opened")),
                    ("Identity", holdout.get("identity")),
                    ("Reason", holdout.get("reason")),
                ],
            )
        ),
    ]
    notes = report.get("notes", [])
    if notes:
        sections.append("## Notes\n\n" + "\n".join(f"- {_escape(note)}" for note in notes))
    return "\n\n".join(section for section in sections if section) + "\n"


def render_regression_holdout_report(report: Mapping[str, Any]) -> str:
    selected = report.get("selected", {})
    economics = report.get("economics", {})
    stress = report.get("cost_stress", {})
    sections = [
        "# Kraken Futures Net-Expectancy Locked-Holdout Benchmark",
        (
            "## Verdict\n\n"
            f"**{_escape(report.get('verdict', 'not_assessed')).upper()}**"
        ),
        "## Reproducibility\n\n" + "\n".join(_identity_lines(report)),
        (
            "## Frozen candidate\n\n"
            + _table(
                ("Item", "Value"),
                [
                    ("Candidate", selected.get("candidate_id")),
                    ("Horizon bars", selected.get("horizon_bars")),
                    ("Model", selected.get("model")),
                    ("Feature set", selected.get("feature_set")),
                    (
                        "Expected-return hurdle bps",
                        selected.get("policy", {}).get("hurdle_bps"),
                    ),
                    (
                        "Directional advantage bps",
                        selected.get("policy", {}).get("advantage_bps"),
                    ),
                ],
            )
        ),
        (
            "## Economic result\n\n"
            + _table(
                ("Metric", "Nominal", "Cost stress"),
                [
                    (
                        "Trades",
                        economics.get("trades"),
                        stress.get("trades"),
                    ),
                    (
                        "Net expectancy bps/trade",
                        economics.get("net_expectancy_bps"),
                        stress.get("net_expectancy_bps"),
                    ),
                    (
                        "95% bootstrap lower bps/trade",
                        economics.get("bootstrap_95_lower_bps"),
                        stress.get("bootstrap_95_lower_bps"),
                    ),
                    (
                        "Profit factor",
                        economics.get("profit_factor"),
                        stress.get("profit_factor"),
                    ),
                    (
                        "Positive-month fraction",
                        economics.get("positive_month_fraction"),
                        stress.get("positive_month_fraction"),
                    ),
                ],
            )
        ),
        (
            "## Holdout gates\n\n"
            + _table(
                ("Gate", "Pass", "Actual", "Required"),
                _gate_rows(report.get("gates")),
            )
        ),
        (
            "## Fixed-action fee counterfactuals\n\n"
            + _table(
                (
                    "Scenario",
                    "Round-trip fee bps",
                    "Trades",
                    "Net bps/trade",
                    "95% lower",
                    "Profit factor",
                ),
                _fee_scenario_rows(report.get("fee_counterfactuals", {})),
            )
        ),
    ]
    notes = report.get("notes", [])
    if notes:
        sections.append("## Notes\n\n" + "\n".join(f"- {_escape(note)}" for note in notes))
    return "\n\n".join(section for section in sections if section) + "\n"


def write_regression_development_report(
    directory: str | Path,
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    root = Path(directory)
    return (
        write_json_artifact(root / "regression-development-benchmark.json", report),
        write_text_artifact(
            root / "regression-development-benchmark.md",
            render_regression_development_report(report),
        ),
    )


def write_regression_holdout_report(
    directory: str | Path,
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    root = Path(directory)
    return (
        write_json_artifact(root / "regression-holdout-benchmark.json", report),
        write_text_artifact(
            root / "regression-holdout-benchmark.md",
            render_regression_holdout_report(report),
        ),
    )
