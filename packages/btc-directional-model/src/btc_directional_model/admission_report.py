from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .admission_benchmark import ADMISSION_BENCHMARK_SCHEMA_VERSION


def generate_admission_report(
    benchmark: dict[str, Any],
    destination: Path,
) -> Path:
    document = render_admission_report(benchmark)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.partial")
    temporary.write_text(document)
    temporary.replace(destination)
    return destination


def render_admission_report(benchmark: dict[str, Any]) -> str:
    if benchmark.get("schema_version") != ADMISSION_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("unsupported correctness-admission benchmark schema")
    candidates = benchmark["candidates"]
    selector_name = benchmark["selector_candidate"]
    selector = candidates[selector_name]
    deployment = benchmark["deployment"]
    cards = "".join(
        (
            _card("Eligible markets", _integer(benchmark["eligible_markets"])),
            _card("Evaluation folds", str(len(benchmark["evaluation_folds"]))),
            _card(
                "Selector coverage",
                _percent(selector["out_of_fold"]["coverage"]),
            ),
            _card(
                "Selector accuracy",
                _percent(selector["out_of_fold"]["accuracy"]),
            ),
            _card(
                "Selector median entry",
                _seconds(selector["out_of_fold"]["median_seconds_elapsed"]),
            ),
            _card(
                "Selector NoTrade",
                _percent(selector["out_of_fold"]["no_trade_rate"]),
            ),
            _card(
                "Development gates",
                "passed" if selector["advance"]["benchmark_passed"] else "blocked",
                "pass" if selector["advance"]["benchmark_passed"] else "blocked",
            ),
            _card("Deployment", "not qualified", "warning"),
        )
    )
    details = html.escape(
        json.dumps(benchmark, indent=2, sort_keys=True, allow_nan=False)
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Capitonic correctness-admission benchmark</title>
<style>
:root {{ color-scheme:dark;--bg:#09101e;--panel:#121d30;--line:#293954;
  --text:#edf3ff;--muted:#9baac4;--pass:#61e5aa;--blocked:#ff7c91;
  --warning:#ffd166;--accent:#6fd5ff }}
* {{ box-sizing:border-box }} body {{ margin:0;background:var(--bg);color:var(--text);
  font:14px/1.5 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }}
main {{ max-width:1500px;margin:auto;padding:24px }} h1 {{ margin:0;font-size:28px }}
h2 {{ margin:0 0 12px;font-size:18px }} .subtitle {{ color:var(--muted);
  margin:6px 0 20px }} .cards {{ display:grid;
  grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:14px }}
.card,.panel {{ background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:16px }} .label {{ color:var(--muted);font-size:12px;
  text-transform:uppercase;letter-spacing:.06em }} .value {{ font-size:18px;
  font-weight:700;margin-top:5px;overflow-wrap:anywhere }} .pass {{ color:var(--pass) }}
.blocked {{ color:var(--blocked) }} .warning {{ color:var(--warning) }}
.notice {{ border-color:var(--warning);color:#fff2c0;margin-bottom:14px }}
.grid {{ display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;
  margin-top:14px }} .table-wrap {{ overflow-x:auto }} table {{
  width:100%;border-collapse:collapse;white-space:nowrap }} th,td {{
  padding:8px;border-bottom:1px solid var(--line);text-align:right }} th:first-child,
td:first-child {{ text-align:left }} th {{ color:var(--muted);font-weight:600 }}
td.name {{ max-width:430px;white-space:normal;overflow-wrap:anywhere }}
code {{ color:var(--accent);overflow-wrap:anywhere }} details {{ margin-top:14px }}
pre {{ color:var(--muted);white-space:pre-wrap;overflow-wrap:anywhere }}
@media(max-width:900px) {{ .grid {{ grid-template-columns:1fr }} }}
</style></head><body><main>
<h1>BTC correctness-admission benchmark</h1>
<div class="subtitle">Core-only causal selector · four-band Platt calibration ·
rolling development folds · five-share execution economics</div>
<div class="cards">{cards}</div>
<section class="panel notice"><strong>Development evidence only.</strong>
The selector has {len(benchmark["evaluation_folds"])} rolling validation folds,
below the deployment minimum of {deployment["required_validation_folds"]}. This
benchmark can reject or prioritize the selector, but it cannot authorize live
capital or a runtime export.</section>
<section class="panel"><h2>Candidate outcomes</h2>
<div class="table-wrap">{_candidate_table(benchmark)}</div></section>
<div class="grid">
<section class="panel"><h2>Rolling selector folds</h2>
<div class="table-wrap">{_selector_fold_table(selector)}</div></section>
<section class="panel"><h2>Execution economics</h2>
<div class="table-wrap">{_economics_table(benchmark)}</div></section>
</div>
<section class="panel" style="margin-top:14px"><h2>Selector advancement gates</h2>
<div class="table-wrap">{_gate_table(selector["advance"])}</div></section>
<section class="panel" style="margin-top:14px"><h2>Causal evidence boundary</h2>
<p>Saved probabilities: <code>{html.escape(benchmark["probability_evidence"]["manifest"])}</code></p>
<p>Core feature scope: <strong>{html.escape(benchmark["core_feature_evidence"]["scope"])}</strong>;
holdout accessed: <strong>no</strong>.</p>
<p>Database accessed: <strong>no</strong>. Runtime exported or changed:
<strong>no</strong>.</p>
<p>{html.escape(deployment["reason"])}</p></section>
<details class="panel"><summary>Deterministic benchmark record</summary>
<pre>{details}</pre></details>
</main></body></html>"""


def _candidate_table(benchmark: dict[str, Any]) -> str:
    rows = []
    control_name = benchmark["control_candidate"]
    for name in benchmark["candidate_order"]:
        result = benchmark["candidates"][name]
        metrics = result["out_of_fold"]
        advance = result["advance"]
        status = (
            "control"
            if name == control_name
            else "passed"
            if advance["benchmark_passed"]
            else "blocked"
        )
        rows.append(
            (
                name,
                status,
                _integer(metrics["markets"]),
                _percent(metrics["coverage"]),
                _percent(metrics["no_trade_rate"]),
                _percent(metrics["accuracy"]),
                _percent(metrics["balanced_accuracy"]),
                _percent(metrics["up_recall"]),
                _percent(metrics["down_recall"]),
                _percent(metrics["wilson_lower_95"]),
                _percent(metrics["expected_calibration_error"]),
                _seconds(metrics["median_seconds_elapsed"]),
            )
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
            "Median entry",
        ),
        rows,
        status_column=1,
    )


def _selector_fold_table(selector: dict[str, Any]) -> str:
    rows = []
    for fold in selector["folds"]:
        validation = fold["validation"]
        metrics = validation["metrics"]
        timing = validation["timing"]
        thresholds = fold["policy_selection"]["thresholds"]
        rows.append(
            (
                str(fold["fold_index"]),
                ",".join(str(value) for value in fold["selector_fit_folds"]),
                str(fold["prior_fold_index"]),
                " · ".join(
                    f"{name}={value:.3f}" for name, value in thresholds.items()
                ),
                _integer(metrics["markets"]),
                _percent(metrics["coverage"]),
                _percent(metrics["accuracy"]),
                _seconds(timing["median_first_crossing_seconds"]),
                _integer(validation["fully_abstained_markets"]),
                "passed" if validation["qualified"] else "blocked",
            )
        )
    return _table(
        (
            "Eval fold",
            "Fit folds",
            "Cal/policy fold",
            "q thresholds",
            "Selected",
            "Coverage",
            "Accuracy",
            "Median",
            "Fully abstained",
            "Absolute gates",
        ),
        rows,
        status_column=9,
    )


def _economics_table(benchmark: dict[str, Any]) -> str:
    rows = []
    for name in benchmark["candidate_order"]:
        execution = benchmark["candidates"][name]["execution"]
        realized_per_share = (
            execution["realized_net_expectancy_per_trade"]
            / benchmark["configuration"]["quantity"]
            if execution["realized_net_expectancy_per_trade"] is not None
            else None
        )
        rows.append(
            (
                name,
                _integer(execution["economic_markets"]),
                _number(execution["mean_direct_edge_per_share"], 5),
                _number(realized_per_share, 5),
                _percent(execution["positive_direct_edge_rate"]),
                _number(execution["maximum_drawdown"], 3),
            )
        )
    return _table(
        (
            "Candidate",
            "Economic markets",
            "Mean direct edge/share",
            "Realized net/share",
            "Positive edge",
            "Max drawdown",
        ),
        rows,
    )


def _gate_table(advance: dict[str, Any]) -> str:
    rows = [
        (
            check["name"],
            _number(check["observed"], 6),
            check["operator"],
            _number(check["required"], 6),
            "passed" if check["passed"] else "blocked",
        )
        for check in (*advance["checks"], *advance["deployment_checks"])
    ]
    return _table(
        ("Gate", "Observed", "Operator", "Required", "Result"),
        rows,
        status_column=4,
    )


def _table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    *,
    status_column: int | None = None,
) -> str:
    header = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            class_name = "name" if index == 0 else ""
            if status_column == index:
                class_name = f"{class_name} {value}".strip()
            cells.append(
                f'<td class="{html.escape(class_name)}">'
                f"{html.escape(str(value))}</td>"
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _card(label: str, value: str, status: str = "") -> str:
    return (
        '<div class="card"><div class="label">'
        + html.escape(label)
        + '</div><div class="value '
        + html.escape(status)
        + '">'
        + html.escape(value)
        + "</div></div>"
    )


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}s"


def _integer(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,}"


def _number(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"
