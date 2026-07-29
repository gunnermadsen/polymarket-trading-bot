from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .policy_benchmark import SAVED_POLICY_BENCHMARK_SCHEMA_VERSION

FREQUENCY_POLICY_BENCHMARK_SCHEMA_VERSION = "btc-frequency-policy-benchmark-v1"


def generate_saved_policy_report(
    benchmark: dict[str, Any],
    destination: Path,
) -> Path:
    document = render_saved_policy_report(benchmark)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.partial")
    temporary.write_text(document)
    temporary.replace(destination)
    return destination


def render_saved_policy_report(benchmark: dict[str, Any]) -> str:
    if benchmark.get("schema_version") not in {
        SAVED_POLICY_BENCHMARK_SCHEMA_VERSION,
        FREQUENCY_POLICY_BENCHMARK_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported saved-policy benchmark schema")
    independent = bool(benchmark["evaluation_is_independent"])
    evidence_label = (
        "Independent evaluation"
        if independent
        else "Development-only · non-independent validation"
    )
    candidates = benchmark["candidates"]
    control_name = benchmark["control_candidate"]
    winner = benchmark.get("winner")
    passed = benchmark.get("benchmark_passed_candidates", [])
    evidence = benchmark["probability_evidence"]
    objective = str(benchmark.get("qualification_objective", "accuracy_timing"))
    selection_mode = str(
        benchmark.get("policy_selection_mode", "per_fold")
    )
    frequency = objective == "frequency"
    cards = "".join(
        (
            _card("Evidence", evidence_label, "pass" if independent else "warning"),
            _card("Objective", objective),
            _card("Policy", selection_mode),
            _card("Candidates", str(len(candidates))),
            _card("Control", control_name),
            _card("Passing challengers", str(len(passed)), "pass" if passed else "blocked"),
            _card("Winner", str(winner or "none"), "pass" if winner else "blocked"),
            _card("Runtime", "unchanged", "pass"),
        )
    )
    warning = (
        "These results reuse saved model probabilities from consumed chronological "
        "development folds. They are useful for selecting and rejecting threshold "
        "policies, but they are not an independent holdout and cannot authorize "
        "deployment."
        if not independent
        else (
            "This report uses independent evidence. Deployment still requires every "
            "pre-registered gate and an explicitly authorized export."
        )
    )
    deployment = benchmark.get("deployment", {})
    execution = benchmark.get("execution_evidence")
    title = (
        "BTC frequency-policy qualification"
        if frequency
        else "BTC saved-prediction policy benchmark"
    )
    subtitle = (
        "One anchor-fold threshold vector · unchanged across five later folds · "
        "timing diagnostic only · no model retraining or runtime export"
        if frequency
        else (
            "Causal time-band threshold selection · first crossing per market · "
            "real saved model probabilities · no model retraining or runtime export"
        )
    )
    execution_panel = (
        '<section class="panel"><h2>Five-share execution evidence</h2>'
        f"{_execution_provenance_table(execution)}</section>"
        if isinstance(execution, dict)
        else ""
    )
    details_json = html.escape(
        json.dumps(benchmark, indent=2, sort_keys=True, allow_nan=False)
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Capitonic {html.escape(title)}</title>
<style>
:root {{ color-scheme:dark;--bg:#0a1020;--panel:#141d31;--line:#2a3652;
  --text:#eaf0ff;--muted:#9eabc8;--accent:#70d7ff;--pass:#62e6aa;
  --blocked:#ff7d90;--warning:#ffd166 }}
* {{ box-sizing:border-box }} body {{ margin:0;background:var(--bg);color:var(--text);
  font:14px/1.5 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }}
main {{ max-width:1600px;margin:auto;padding:24px }} h1 {{ margin:0;font-size:28px }}
h2 {{ margin:0 0 12px;font-size:18px }} h3 {{ margin:18px 0 8px;font-size:15px }}
.subtitle {{ color:var(--muted);margin:6px 0 20px }} .cards {{ display:grid;
  grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:14px }}
.card,.panel {{ background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:16px }} .label {{ color:var(--muted);font-size:12px;
  text-transform:uppercase;letter-spacing:.06em }} .value {{ font-size:18px;
  font-weight:700;margin-top:5px;overflow-wrap:anywhere }} .pass {{ color:var(--pass) }}
.blocked {{ color:var(--blocked) }} .warning {{ color:var(--warning) }}
.notice {{ border-color:var(--warning);color:#fff4c6;margin-bottom:14px }}
.grid {{ display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;
  margin-top:14px }} .table-wrap {{ overflow-x:auto }} table {{
  width:100%;border-collapse:collapse;white-space:nowrap }} th,td {{
  padding:8px;border-bottom:1px solid var(--line);text-align:right }} th:first-child,
td:first-child {{ text-align:left }} th {{ color:var(--muted);font-weight:600 }}
td.name {{ max-width:360px;white-space:normal;overflow-wrap:anywhere }} code {{
  color:var(--accent);overflow-wrap:anywhere }} .status {{ font-weight:700 }}
p {{ margin:8px 0 }} details {{ margin-top:14px }} pre {{ white-space:pre-wrap;
  overflow-wrap:anywhere;color:var(--muted) }}
@media(max-width:900px) {{ .grid {{ grid-template-columns:1fr }} }}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<div class="subtitle">{html.escape(subtitle)}</div>
<div class="cards">{cards}</div>
<section class="panel notice"><strong>{html.escape(evidence_label)}.</strong>
{html.escape(warning)}</section>
<section class="panel"><h2>Candidate outcomes</h2>
<div class="table-wrap">{_candidate_table(benchmark)}</div></section>
<div class="grid">
<section class="panel"><h2>Saved-probability provenance</h2>
{_provenance_table(evidence)}</section>
{execution_panel}
<section class="panel"><h2>Deployment boundary</h2>
<p>Status: <span class="status blocked">{html.escape(str(deployment.get("status")))}</span></p>
<p>{html.escape(str(deployment.get("scope", "")))}</p>
<p>Runtime exported: <strong>{_yes_no(deployment.get("runtime_exported"))}</strong></p>
<p>Runtime changed: <strong>{_yes_no(deployment.get("runtime_changed"))}</strong></p>
</section></div>
<section class="panel" style="margin-top:14px"><h2>Frozen fold policies</h2>
<div class="table-wrap">{_fold_policy_table(benchmark)}</div></section>
<section class="panel" style="margin-top:14px"><h2>Advancement gates</h2>
{_gate_tables(benchmark)}</section>
<details class="panel"><summary>Deterministic benchmark record</summary>
<pre>{details_json}</pre></details>
</main></body></html>"""


def _candidate_table(benchmark: dict[str, Any]) -> str:
    rows = []
    control_name = benchmark["control_candidate"]
    for name, candidate in benchmark["candidates"].items():
        metrics = candidate["out_of_fold"]
        timing = candidate["timing"]
        passed = bool(candidate["advance"]["benchmark_passed"])
        status = (
            "control"
            if name == control_name
            else ("passed" if passed else "blocked")
        )
        rows.append(
            (
                name,
                status,
                _integer(metrics["markets"]),
                _percent(metrics["coverage"]),
                _percent(1.0 - metrics["coverage"]),
                _percent(metrics["accuracy"]),
                _percent(metrics["balanced_accuracy"]),
                _percent(metrics["up_recall"]),
                _percent(metrics["down_recall"]),
                _percent(metrics["wilson_lower_95"]),
                _percent(metrics["expected_calibration_error"]),
                _number(timing["median_first_crossing_seconds"], 1),
                f"{candidate['policy_qualified_folds']}/{candidate['fold_count']}",
                f"{candidate['validation_qualified_folds']}/{candidate['fold_count']}",
            )
        )
    policy_heading = (
        "Frozen policy applied"
        if benchmark.get("policy_selection_mode") == "single_frozen"
        else "Policy folds"
    )
    return _table(
        (
            "Candidate",
            "Result",
            "Selected",
            "Coverage",
            "NoTrade",
            "Accuracy",
            "Balanced",
            "UP recall",
            "DOWN recall",
            "Wilson lower",
            "ECE",
            "Median sec",
            policy_heading,
            "Validation folds",
        ),
        rows,
        status_column=1,
    )


def _fold_policy_table(benchmark: dict[str, Any]) -> str:
    rows = []
    for name, candidate in benchmark["candidates"].items():
        for fold in candidate["folds"]:
            selection = fold["policy_selection"]
            validation = fold["validation"]
            thresholds = selection["thresholds"]
            threshold_text = " · ".join(
                f"{band}: {_number(value, 3)}"
                for band, value in thresholds.items()
            )
            rows.append(
                (
                    name,
                    str(fold["fold_index"]),
                    str(selection.get("source_fold_index", fold["fold_index"])),
                    threshold_text,
                    "passed" if selection["qualified"] else "blocked",
                    _percent(selection["metrics"]["coverage"]),
                    _percent(selection["metrics"]["accuracy"]),
                    _number(
                        selection["timing"]["median_first_crossing_seconds"],
                        1,
                    ),
                    "passed" if validation["qualified"] else "blocked",
                    _percent(validation["metrics"]["coverage"]),
                    _percent(validation["metrics"]["accuracy"]),
                    _number(
                        validation["timing"]["median_first_crossing_seconds"],
                        1,
                    ),
                    _range_label(fold["policy_selection_range"]),
                    _range_label(fold["validation_range"]),
                )
            )
    return _table(
        (
            "Candidate",
            "Fold",
            "Policy source fold",
            "Frozen thresholds",
            "Policy",
            "Policy coverage",
            "Policy accuracy",
            "Policy median",
            "Validation",
            "Validation coverage",
            "Validation accuracy",
            "Validation median",
            "Policy range",
            "Validation range",
        ),
        rows,
        status_columns={4, 8},
    )


def _gate_tables(benchmark: dict[str, Any]) -> str:
    sections = []
    for name, candidate in benchmark["candidates"].items():
        advance = candidate["advance"]
        rows = [
            (
                check["name"],
                _observed(check["observed"]),
                check["operator"],
                _observed(check["required"]),
                "passed" if check["passed"] else "blocked",
            )
            for check in advance["checks"]
        ]
        outcome = "passed" if advance["benchmark_passed"] else "blocked"
        sections.append(
            f"<h3>{html.escape(name)} · "
            f'<span class="status {outcome}">{outcome}</span></h3>'
            f'<div class="table-wrap">{_table(("Gate", "Observed", "Rule", "Required", "Result"), rows, status_column=4)}</div>'
        )
    return "".join(sections)


def _provenance_table(evidence: dict[str, Any]) -> str:
    rows = (
        ("Manifest", evidence.get("manifest")),
        ("Manifest SHA-256", evidence.get("manifest_sha256")),
        ("Schema", evidence.get("schema_version")),
        ("Manifest created", evidence.get("created_at")),
        ("Source benchmark profile", evidence.get("source_benchmark_profile")),
        ("Source config", evidence.get("source_config")),
        ("Source config SHA-256", evidence.get("source_config_sha256")),
        ("Fold count", evidence.get("fold_count")),
        ("Checksums verified", _yes_no(evidence.get("checksums_verified"))),
        ("Causal contract verified", _yes_no(evidence.get("causal_contract_verified"))),
        ("Causal contract", evidence.get("causal_contract")),
    )
    return _table(("Field", "Value"), rows)


def _execution_provenance_table(evidence: dict[str, Any]) -> str:
    rows = (
        ("Manifest", evidence.get("manifest")),
        ("Manifest SHA-256", evidence.get("manifest_sha256")),
        ("Source contract", evidence.get("source_contract")),
        ("Source schema", evidence.get("source_schema_version")),
        ("Range start", evidence.get("range_start")),
        ("Range end", evidence.get("range_end")),
        ("Quantity", evidence.get("quantity")),
        ("Checksums verified", _yes_no(evidence.get("checksums_verified"))),
    )
    return _table(("Field", "Value"), rows)


def _table(
    headings: tuple[str, ...],
    rows: list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...],
    *,
    status_column: int | None = None,
    status_columns: set[int] | None = None,
) -> str:
    marked = status_columns or ({status_column} if status_column is not None else set())
    header = "".join(f"<th>{html.escape(value)}</th>" for value in headings)
    rendered_rows = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            text = "" if value is None else str(value)
            css = ""
            if index == 0:
                css = ' class="name"'
            elif index in marked and text in {"passed", "blocked"}:
                css = f' class="status {text}"'
            cells.append(f"<td{css}>{html.escape(text)}</td>")
        rendered_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rendered_rows)
        + "</tbody></table>"
    )


def _card(label: str, value: str, css: str = "") -> str:
    class_name = f" {css}" if css else ""
    return (
        '<div class="card"><div class="label">'
        + html.escape(label)
        + f'</div><div class="value{class_name}">'
        + html.escape(value)
        + "</div></div>"
    )


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _number(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _integer(value: float | None) -> str:
    return "n/a" if value is None else f"{int(value):,}"


def _observed(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return _yes_no(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _yes_no(value: Any) -> str:
    return "yes" if value is True else "no"


def _range_label(value: dict[str, Any]) -> str:
    return f"{value.get('start', '?')} → {value.get('end', '?')}"
