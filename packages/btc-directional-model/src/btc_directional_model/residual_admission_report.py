from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .residual_admission_benchmark import RESIDUAL_ADMISSION_SCHEMA_VERSION

_CHECKPOINTS = ("60", "90", "120", "180", "240")
_ATTRIBUTION_LABELS = {
    "preserved_control": "Preserved control",
    "earlier_same_direction": "Earlier, same direction",
    "earlier_preempted_direction_change": "Earlier, preempted direction change",
    "rescued_control_no_trade": "Rescued control NoTrade",
    "lost_control": "Lost control",
    "still_no_trade": "Still NoTrade",
}


def generate_residual_admission_report(
    benchmark: dict[str, Any],
    destination: Path,
) -> Path:
    document = render_residual_admission_report(benchmark)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.partial")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(destination)
    return destination


def render_residual_admission_report(benchmark: dict[str, Any]) -> str:
    if benchmark.get("schema_version") != RESIDUAL_ADMISSION_SCHEMA_VERSION:
        raise ValueError("unsupported residual-admission benchmark schema")

    candidates = benchmark["candidates"]
    order = _candidate_order(benchmark)
    control_name = benchmark["control_candidate"]
    control = candidates[control_name]
    passing = benchmark.get("benchmark_passed_candidates", [])
    winner = benchmark.get("winner")
    deployment = benchmark["deployment"]
    evidence_label = (
        "Independent evidence"
        if benchmark["evaluation_is_independent"]
        else "Consumed development evidence"
    )
    eligible = control["out_of_fold"]["eligible_markets"]
    cards = "".join(
        (
            _card(
                "Evidence",
                evidence_label,
                "pass" if benchmark["evaluation_is_independent"] else "warning",
            ),
            _card("Eligible markets", _integer(eligible)),
            _card("Evaluation folds", str(len(benchmark["folds"]))),
            _card("Frozen control", control_name),
            _card(
                "Passing residuals",
                str(len(passing)),
                "pass" if passing else "blocked",
            ),
            _card(
                "Winner",
                str(winner or "none"),
                "pass" if winner else "blocked",
            ),
            _card("Deployment", str(deployment["status"]), "blocked"),
            _card("Runtime", "unchanged", "pass"),
        )
    )
    details = html.escape(
        json.dumps(benchmark, indent=2, sort_keys=True, allow_nan=False)
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Capitonic BTC residual-admission benchmark</title>
<style>
:root {{ color-scheme:dark;--bg:#09101d;--panel:#121d30;--line:#293a56;
  --text:#edf3ff;--muted:#9cabc5;--accent:#70d8ff;--pass:#61e5aa;
  --blocked:#ff7d91;--warning:#ffd166 }}
* {{ box-sizing:border-box }} body {{ margin:0;background:var(--bg);color:var(--text);
  font:14px/1.5 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }}
main {{ max-width:1680px;margin:auto;padding:24px }} h1 {{ margin:0;font-size:28px }}
h2 {{ margin:0 0 12px;font-size:18px }} h3 {{ margin:18px 0 8px;font-size:15px }}
p {{ margin:8px 0 }} ul {{ margin:8px 0;padding-left:22px }}
.subtitle {{ color:var(--muted);margin:6px 0 20px }} .cards {{ display:grid;
  grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:14px }}
.card,.panel {{ background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:16px }} .label {{ color:var(--muted);font-size:12px;
  text-transform:uppercase;letter-spacing:.06em }} .value {{ font-size:18px;
  font-weight:700;margin-top:5px;overflow-wrap:anywhere }} .pass {{ color:var(--pass) }}
.blocked {{ color:var(--blocked) }} .warning {{ color:var(--warning) }}
.notice {{ border-color:var(--warning);color:#fff3c2;margin-bottom:14px }}
.danger {{ border-color:var(--blocked) }} .grid {{ display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:14px }}
.table-wrap {{ overflow-x:auto }} table {{ width:100%;border-collapse:collapse;
  white-space:nowrap }} th,td {{ padding:8px;border-bottom:1px solid var(--line);
  text-align:right }} th:first-child,td:first-child {{ text-align:left }}
th {{ color:var(--muted);font-weight:600 }} td.name {{ max-width:410px;
  white-space:normal;overflow-wrap:anywhere }} code {{ color:var(--accent);
  overflow-wrap:anywhere }} .status {{ font-weight:700 }} details {{ margin-top:14px }}
pre {{ color:var(--muted);white-space:pre-wrap;overflow-wrap:anywhere }}
@media(max-width:940px) {{ .grid {{ grid-template-columns:1fr }} }}
</style></head><body><main>
<h1>BTC residual-admission benchmark</h1>
<div class="subtitle">Frozen control priority · independent early and rescue heads ·
five causal evaluation folds · fixed five-share execution diagnostics</div>
<div class="cards">{cards}</div>
<section class="panel notice"><strong>Development evidence only.</strong>
These rolling folds reuse consumed evidence. They can reject or prioritize an
admission policy, but they cannot authorize runtime export, live deployment, or
capital allocation.</section>
<section class="panel"><h2>Control vs residual outcomes</h2>
<div class="table-wrap">{_candidate_table(benchmark, order)}</div></section>
<section class="panel" style="margin-top:14px"><h2>Cumulative decision coverage</h2>
<p>Each checkpoint is the share of all eligible markets with a decision by that
second. Deltas are paired against the frozen control on the same market universe.</p>
<div class="table-wrap">{_checkpoint_table(benchmark, order)}</div></section>
<div class="grid">
<section class="panel"><h2>Paired control attribution</h2>
<div class="table-wrap">{_attribution_summary_table(benchmark, order)}</div></section>
<section class="panel"><h2>Attribution categories</h2>
<div class="table-wrap">{_attribution_category_table(benchmark, order)}</div></section>
</div>
<section class="panel" style="margin-top:14px"><h2>Per-fold frozen thresholds</h2>
<div class="table-wrap">{_threshold_table(benchmark)}</div></section>
<section class="panel" style="margin-top:14px"><h2>Threshold grid diagnostics</h2>
<p>Every frozen q candidate is shown before threshold selection. Policy quality
and paired residual outcomes use the same calibration/policy fold.</p>
<div class="table-wrap">{_threshold_diagnostics_table(benchmark)}</div></section>
<section class="panel" style="margin-top:14px"><h2>Per-fold candidate quality</h2>
<div class="table-wrap">{_fold_quality_table(benchmark, order)}</div></section>
<section class="panel" style="margin-top:14px"><h2>Five-share execution economics</h2>
<div class="table-wrap">{_economics_table(benchmark, order)}</div></section>
{_economics_limitation_panel(benchmark)}
<section class="panel" style="margin-top:14px"><h2>Frozen advancement gates</h2>
{_gate_tables(benchmark, order)}</section>
<div class="grid">
{_provenance_panel(benchmark)}
{_deployment_panel(benchmark)}
</div>
<details class="panel"><summary>Deterministic benchmark record</summary>
<pre>{details}</pre></details>
</main></body></html>"""


def _candidate_order(benchmark: dict[str, Any]) -> list[str]:
    candidates = benchmark["candidates"]
    configured = benchmark.get("configuration", {}).get("candidate_names", [])
    order = [name for name in configured if name in candidates]
    order.extend(name for name in candidates if name not in order)
    return order


def _candidate_table(
    benchmark: dict[str, Any],
    order: list[str],
) -> str:
    rows = []
    control_name = benchmark["control_candidate"]
    for name in order:
        candidate = benchmark["candidates"][name]
        metrics = candidate["out_of_fold"]
        residual = candidate["residual_cohort"]
        advance = candidate["advance"]
        status = (
            "control"
            if name == control_name
            else "passed"
            if advance["benchmark_passed"]
            else "blocked"
        )
        rows.append(
            (
                _candidate_label(benchmark, name),
                status,
                _integer(metrics["markets"]),
                _percent(metrics["coverage"]),
                _percent(candidate["no_trade_rate"]),
                _percent(metrics["accuracy"]),
                _percent(metrics["balanced_accuracy"]),
                _percent(metrics["up_recall"]),
                _percent(metrics["down_recall"]),
                _percent(metrics["wilson_lower_95"]),
                _percent(metrics["expected_calibration_error"]),
                _seconds(candidate["timing"]["median_first_crossing_seconds"]),
                _seconds(candidate["timing"]["p90_first_crossing_seconds"]),
                _integer(residual["markets"]),
                _percent(residual["accuracy"]),
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
            "Median",
            "P90",
            "Residual",
            "Residual accuracy",
        ),
        rows,
        status_column=1,
    )


def _checkpoint_table(
    benchmark: dict[str, Any],
    order: list[str],
) -> str:
    control_name = benchmark["control_candidate"]
    control = benchmark["candidates"][control_name]["checkpoints"]
    rows = []
    for name in order:
        checkpoints = benchmark["candidates"][name]["checkpoints"]
        rows.append(
            (
                _candidate_label(benchmark, name),
                *(_percent(checkpoints[second]) for second in _CHECKPOINTS),
                _signed_points(checkpoints["120"] - control["120"]),
                _signed_points(checkpoints["240"] - control["240"]),
            )
        )
    return _table(
        (
            "Candidate",
            "60 sec",
            "90 sec",
            "120 sec",
            "180 sec",
            "240 sec",
            "Δ120 vs control",
            "Δ240 vs control",
        ),
        rows,
    )


def _attribution_summary_table(
    benchmark: dict[str, Any],
    order: list[str],
) -> str:
    rows = []
    control_name = benchmark["control_candidate"]
    for name in order:
        if name == control_name:
            continue
        attribution = benchmark["candidates"][name]["paired_attribution"]
        rows.append(
            (
                _candidate_label(benchmark, name),
                _integer(attribution["advanced_markets"]),
                _seconds(attribution["median_advancement_seconds"]),
                _integer(attribution["rescued_markets"]),
                _integer(attribution["lost_control_markets"]),
                _integer(
                    attribution["categories"]["earlier_preempted_direction_change"][
                        "markets"
                    ]
                ),
                _integer(
                    sum(
                        category["added_wrong_trades"]
                        for category in attribution["categories"].values()
                    )
                ),
            )
        )
    return _table(
        (
            "Candidate",
            "Advanced",
            "Median advancement",
            "NoTrade rescued",
            "Control lost",
            "Direction preempted",
            "Added wrong trades",
        ),
        rows,
    )


def _attribution_category_table(
    benchmark: dict[str, Any],
    order: list[str],
) -> str:
    rows = []
    control_name = benchmark["control_candidate"]
    for name in order:
        if name == control_name:
            continue
        categories = benchmark["candidates"][name]["paired_attribution"]["categories"]
        for category_name, label in _ATTRIBUTION_LABELS.items():
            category = categories[category_name]
            rows.append(
                (
                    _candidate_label(benchmark, name),
                    label,
                    _integer(category["markets"]),
                    _percent(category["accuracy"]),
                    _seconds(category["median_entry_difference_seconds"]),
                    _integer(category["added_wrong_trades"]),
                )
            )
    return _table(
        (
            "Candidate",
            "Paired category",
            "Markets",
            "Candidate accuracy",
            "Entry improvement",
            "Wrong trades",
        ),
        rows,
    )


def _threshold_table(benchmark: dict[str, Any]) -> str:
    rows = []
    for fold in benchmark["folds"]:
        for head_name in ("early", "rescue"):
            selection = fold["thresholds"][head_name]
            checks = selection.get("checks", [])
            objective = selection.get("objective", {})
            rows.append(
                (
                    str(fold["fold_index"]),
                    head_name,
                    ",".join(str(index) for index in fold["selector_fit_folds"]),
                    str(fold["prior_fold_index"]),
                    _number(selection["threshold"], 3),
                    "passed" if selection["qualified"] else "blocked",
                    _integer(selection["thresholds_evaluated"]),
                    _integer(selection["qualifying_thresholds"]),
                    _integer(objective.get("residual_markets")),
                    _percent(objective.get("residual_accuracy")),
                    _integer(objective.get("rescued_markets")),
                    _gate_count(checks),
                )
            )
    return _table(
        (
            "Eval fold",
            "Head",
            "Fit folds",
            "Cal/policy fold",
            "Frozen q",
            "Selection",
            "Tried",
            "Qualified",
            "Residual markets",
            "Residual accuracy",
            "NoTrade rescued",
            "Policy gates",
        ),
        rows,
        status_column=5,
    )


def _threshold_diagnostics_table(benchmark: dict[str, Any]) -> str:
    rows = []
    for fold in benchmark["folds"]:
        for head_name in ("early", "rescue"):
            selection = fold["thresholds"][head_name]
            selected_threshold = selection.get("threshold")
            for diagnostic in selection.get("threshold_diagnostics", []):
                threshold = diagnostic["threshold"]
                metrics = diagnostic["metrics"]
                objective = diagnostic["objective"]
                checks = diagnostic.get("checks", [])
                failed_gates = [
                    str(check["name"]) for check in checks if not check["passed"]
                ]
                rows.append(
                    (
                        str(fold["fold_index"]),
                        head_name,
                        _number(threshold, 3),
                        _yes_no(
                            diagnostic["qualified"]
                            and selected_threshold is not None
                            and threshold == selected_threshold
                        ),
                        "passed" if diagnostic["qualified"] else "blocked",
                        _percent(metrics["accuracy"]),
                        _percent(metrics["up_recall"]),
                        _percent(metrics["down_recall"]),
                        _percent(metrics["expected_calibration_error"]),
                        _percent(metrics["coverage"]),
                        _integer(objective.get("residual_markets")),
                        _percent(objective.get("residual_accuracy")),
                        _signed_points(objective.get("decisions_by_120_uplift")),
                        _integer(objective.get("rescued_markets")),
                        "; ".join(failed_gates) if failed_gates else "none",
                    )
                )
    return _table(
        (
            "Eval fold",
            "Head",
            "q",
            "Selected",
            "Policy gates",
            "Policy accuracy",
            "UP recall",
            "DOWN recall",
            "ECE",
            "Coverage",
            "Residual markets",
            "Residual accuracy",
            "Δ120 vs control",
            "NoTrade rescued",
            "Failed gates",
        ),
        rows,
        status_column=4,
    )


def _fold_quality_table(
    benchmark: dict[str, Any],
    order: list[str],
) -> str:
    rows = []
    gates = benchmark.get("configuration", {}).get("gates", {})
    control_name = benchmark["control_candidate"]
    for fold in benchmark["folds"]:
        for name in order:
            candidate = fold["candidates"][name]
            metrics = candidate["metrics"]
            residual = candidate["residual_cohort"]
            attribution = candidate["paired_attribution"]
            quality = (
                "control"
                if name == control_name
                else _fold_quality_status(metrics, gates)
            )
            rows.append(
                (
                    str(fold["fold_index"]),
                    _candidate_label(benchmark, name),
                    quality,
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
                    _integer(residual["markets"]),
                    _integer(
                        attribution["rescued_markets"]
                        if attribution is not None
                        else None
                    ),
                )
            )
    return _table(
        (
            "Fold",
            "Candidate",
            "Absolute quality",
            "Selected",
            "Coverage",
            "NoTrade",
            "Accuracy",
            "Balanced",
            "UP recall",
            "DOWN recall",
            "Wilson",
            "ECE",
            "Median",
            "Residual",
            "Rescued",
        ),
        rows,
        status_column=2,
    )


def _economics_table(
    benchmark: dict[str, Any],
    order: list[str],
) -> str:
    rows = []
    quantity = float(benchmark["configuration"]["quantity"])
    control_name = benchmark["control_candidate"]
    for name in order:
        candidate = benchmark["candidates"][name]
        cohorts = [("All selected", candidate["out_of_fold"]["execution"], None)]
        if name != control_name:
            residual = candidate["residual_cohort"]
            cohorts.append(
                (
                    "Residual additions",
                    residual["execution"],
                    residual["hourly_net_bootstrap"],
                )
            )
        for cohort_name, execution, bootstrap in cohorts:
            net_per_share = (
                execution["realized_net_expectancy_per_trade"] / quantity
                if execution["realized_net_expectancy_per_trade"] is not None
                else None
            )
            rows.append(
                (
                    _candidate_label(benchmark, name),
                    cohort_name,
                    _integer(execution["selected_markets"]),
                    _integer(execution["execution_evidence_markets"]),
                    _integer(execution["executable_markets"]),
                    _integer(execution["economic_markets"]),
                    _percent(execution["execution_evidence_coverage"]),
                    _percent(execution["executable_coverage_all_selected"]),
                    _number(execution["mean_direct_edge_per_share"], 5),
                    _number(net_per_share, 5),
                    _percent(execution["positive_direct_edge_rate"]),
                    _number(execution["maximum_drawdown"], 3),
                    _number(
                        bootstrap.get("lower_95") if bootstrap is not None else None,
                        5,
                    ),
                )
            )
    return _table(
        (
            "Candidate",
            "Cohort",
            "Selected",
            "Evidence",
            "Executable",
            "Economic",
            "Evidence coverage",
            "Executable coverage",
            "Direct edge/share",
            "Realized net/share",
            "Positive edge",
            "Max drawdown",
            "Hourly lower 95%",
        ),
        rows,
    )


def _economics_limitation_panel(benchmark: dict[str, Any]) -> str:
    evidence = benchmark["execution_evidence"]
    return (
        '<section class="panel notice" style="margin-top:14px">'
        "<h2>Economics limitations</h2>"
        f"<p><strong>Role:</strong> {html.escape(str(evidence['role']))}.</p>"
        f"<p><strong>Known gap:</strong> {html.escape(str(evidence['known_gap']))}.</p>"
        "<p>Execution results cover only strict decision-time rows that can be joined "
        "to the cached evidence. Missing rows are not inferred as executable or "
        "profitable. The incomplete five-fold execution cohort blocks deployment "
        "regardless of aggregate model accuracy or coverage.</p></section>"
    )


def _gate_tables(
    benchmark: dict[str, Any],
    order: list[str],
) -> str:
    sections = []
    control_name = benchmark["control_candidate"]
    for name in order:
        if name == control_name:
            continue
        advance = benchmark["candidates"][name]["advance"]
        rows = [
            (
                check["name"],
                _number(check["observed"], 6),
                check["operator"],
                _number(check["required"], 6),
                "passed" if check["passed"] else "blocked",
            )
            for check in advance["checks"]
        ]
        status = "passed" if advance["benchmark_passed"] else "blocked"
        sections.append(
            f"<h3>{html.escape(_candidate_label(benchmark, name))} — "
            f'<span class="status {status}">{status}</span></h3>'
            + _table(
                ("Gate", "Observed", "Operator", "Required", "Result"),
                rows,
                status_column=4,
            )
        )
    return "".join(sections)


def _provenance_panel(benchmark: dict[str, Any]) -> str:
    probabilities = benchmark["probability_evidence"]
    features = benchmark["core_features"]
    rows = [
        ("Saved probability manifest", probabilities["manifest"]),
        ("Manifest SHA-256", probabilities["manifest_sha256"]),
        ("Probability checksums verified", _yes_no(probabilities["checksums_verified"])),
        ("Source profile", probabilities["source_profile"]),
        ("Core feature path", features["path"]),
        ("Core feature SHA-256", features["sha256"]),
        ("Core rows", _integer(features["rows"])),
        ("Core markets", _integer(features["markets"])),
        ("Sealed holdout accessed", _yes_no(features["holdout_accessed"])),
    ]
    return (
        '<section class="panel"><h2>Evidence provenance</h2>'
        + _table(("Item", "Value"), rows)
        + "</section>"
    )


def _deployment_panel(benchmark: dict[str, Any]) -> str:
    deployment = benchmark["deployment"]
    reasons = "".join(
        f"<li>{html.escape(str(reason))}</li>"
        for reason in deployment.get("reasons", [])
    )
    return (
        '<section class="panel danger"><h2>Deployment remains blocked</h2>'
        f'<p>Status: <span class="status blocked">'
        f"{html.escape(str(deployment['status']))}</span>.</p>"
        f"<p>Runtime exported: <strong>{_yes_no(deployment['runtime_exported'])}"
        f"</strong>; runtime changed: <strong>{_yes_no(deployment['runtime_changed'])}"
        "</strong>.</p>"
        f"<ul>{reasons}</ul></section>"
    )


def _candidate_label(benchmark: dict[str, Any], name: str) -> str:
    configuration = benchmark.get("configuration", {})
    if name == benchmark["control_candidate"]:
        role = "Control"
    elif name == configuration.get("early_head", {}).get("candidate"):
        role = "Early"
    elif name == configuration.get("rescue_head", {}).get("candidate"):
        role = "Rescue"
    elif name == configuration.get("combined_candidate"):
        role = "Combined"
    else:
        role = "Candidate"
    return f"{role} · {name}"


def _fold_quality_status(
    metrics: dict[str, Any],
    gates: dict[str, Any],
) -> str:
    required = (
        "minimum_accuracy",
        "minimum_balanced_accuracy",
        "minimum_direction_recall",
        "minimum_wilson_lower_95",
        "maximum_expected_calibration_error",
    )
    if not all(name in gates for name in required):
        return "unknown"
    passed = all(
        (
            metrics["accuracy"] >= gates["minimum_accuracy"],
            metrics["balanced_accuracy"] >= gates["minimum_balanced_accuracy"],
            metrics["up_recall"] >= gates["minimum_direction_recall"],
            metrics["down_recall"] >= gates["minimum_direction_recall"],
            metrics["wilson_lower_95"] >= gates["minimum_wilson_lower_95"],
            metrics["expected_calibration_error"]
            <= gates["maximum_expected_calibration_error"],
        )
    )
    return "passed" if passed else "blocked"


def _gate_count(checks: list[dict[str, Any]]) -> str:
    if not checks:
        return "0/0"
    return f"{sum(bool(check['passed']) for check in checks)}/{len(checks)}"


def _table(
    headers: tuple[str, ...],
    rows: list[tuple[Any, ...]],
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


def _signed_points(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.2f} pp"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}s"


def _integer(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,}"


def _number(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"
