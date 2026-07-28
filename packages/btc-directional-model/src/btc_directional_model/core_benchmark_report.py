from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - Plotly is an optional report enhancement.
    go = None

from .core_benchmark import BENCHMARK_SCHEMA_VERSION


def generate_benchmark_report(benchmark: dict[str, Any], destination: Path) -> Path:
    document = render_benchmark_report(benchmark)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.partial")
    temporary.write_text(document)
    temporary.replace(destination)
    return destination


def render_benchmark_report(benchmark: dict[str, Any]) -> str:
    if benchmark.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("unsupported benchmark schema")
    evidence = benchmark["evaluation"]
    control_name = benchmark["control_candidate"]
    candidate_order = benchmark["candidate_order"]
    evidence_status = _evidence_status(evidence)
    cards = "".join(
        (
            _card("Evidence", evidence_status, _evidence_css(evidence)),
            _card("Control", control_name),
            _card("Eligible markets", f"{benchmark['eligible_markets']:,}"),
            _card("Quantity", f"{benchmark['quantity']:.0f} shares"),
            _card(
                "Benchmark pass",
                str(len(benchmark["benchmark_passed_candidates"])),
                (
                    "pass"
                    if benchmark["benchmark_passed_candidates"]
                    else "blocked"
                ),
            ),
            _card(
                "Deployment-qualified",
                str(len(benchmark["deployment_qualified_candidates"])),
                (
                    "pass"
                    if benchmark["deployment_qualified_candidates"]
                    else "blocked"
                ),
            ),
        )
    )
    warning = (
        "This is non-independent development evidence. It can compare candidates, "
        "but it cannot qualify a model for deployment."
        if evidence["development_only"]
        else (
            "This report is based on an independent holdout. Qualification still "
            "requires every pre-registered benchmark gate to pass."
        )
    )
    figures = _benchmark_figures(benchmark)
    plot_html = []
    for index, figure in enumerate(figures):
        plot_html.append(
            figure.to_html(
                full_html=False,
                include_plotlyjs=index == 0,
                config={"displaylogo": False, "responsive": True},
                div_id=f"btc-benchmark-chart-{index + 1}",
            )
        )
    plots = "".join(
        f'<section class="panel plot">{figure}</section>' for figure in plot_html
    )
    details_json = html.escape(
        json.dumps(benchmark, indent=2, sort_keys=True, allow_nan=False)
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Capitonic BTC Model Benchmark</title>
<style>
:root {{ color-scheme:dark;--bg:#0b1020;--panel:#141c30;--line:#27324c;
  --text:#e9eefc;--muted:#9ca9c7;--accent:#67d5ff;--pass:#5ee6a8;
  --blocked:#ff7b8d;--warn:#ffd166 }}
* {{ box-sizing:border-box }} body {{ margin:0;background:var(--bg);color:var(--text);
  font:14px/1.5 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }}
main {{ max-width:1600px;margin:auto;padding:24px }} h1 {{ margin:0;font-size:28px }}
h2 {{ margin:0 0 12px;font-size:18px }} h3 {{ margin:18px 0 8px;font-size:15px }}
.subtitle {{ color:var(--muted);margin:6px 0 20px }} .cards {{ display:grid;
  grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:14px }}
.card,.panel {{ background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:16px }} .label {{ color:var(--muted);font-size:12px;
  text-transform:uppercase;letter-spacing:.06em }} .value {{ font-size:20px;
  font-weight:700;margin-top:5px;overflow-wrap:anywhere }} .pass {{ color:var(--pass) }}
.blocked {{ color:var(--blocked) }} .warning {{ border-color:var(--warn);
  color:#fff4c4;margin-bottom:14px }} .grid {{ display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));gap:14px }}
.plot {{ min-height:390px }} .table-wrap {{ overflow-x:auto }} table {{
  width:100%;border-collapse:collapse;white-space:nowrap }} th,td {{
  padding:8px;border-bottom:1px solid var(--line);text-align:right }} th:first-child,
td:first-child {{ text-align:left }} th {{ color:var(--muted);font-weight:600 }}
code {{ color:var(--accent) }} details {{ margin-top:14px }} pre {{ white-space:pre-wrap;
  overflow-wrap:anywhere;color:var(--muted) }}
@media(max-width:900px) {{ .grid {{ grid-template-columns:1fr }} }}
</style></head><body><main>
<h1>BTC five-minute real-model benchmark</h1>
<div class="subtitle">{html.escape(evidence['label'])} · own confidence policies ·
same-market, exact-timestamp checkpoint comparisons · fixed five-share economics</div>
<div class="cards">{cards}</div>
<section class="panel warning">{html.escape(warning)}</section>
<section class="panel"><h2>Own-policy outcomes</h2><div class="table-wrap">
{_candidate_table(benchmark, candidate_order)}</div></section>
{_training_evidence_panel(benchmark)}
{_strict_book_chronology_panel(benchmark)}
{_book_residual_panel(benchmark)}
{_persistence_analysis_panel(benchmark)}
{_training_selection_panel(benchmark)}
{_data_evidence_panel(benchmark)}
<div class="grid" style="margin-top:14px">{plots}</div>
<section class="panel" style="margin-top:14px"><h2>Advance gates</h2>
{_advance_tables(benchmark, candidate_order)}</section>
<section class="panel" style="margin-top:14px"><h2>Accepted prediction time bands</h2>
<div class="table-wrap">{_time_band_table(benchmark, candidate_order)}</div></section>
<section class="panel" style="margin-top:14px"><h2>Common exact-timestamp comparisons
(unfiltered predictions)</h2>
<div class="table-wrap">{_common_comparison_table(benchmark)}</div></section>
{_deployment_panel(benchmark)}
<details class="panel"><summary>Deterministic benchmark record</summary>
<pre>{details_json}</pre></details>
</main></body></html>"""


def _strict_book_chronology_panel(benchmark: dict[str, Any]) -> str:
    selection = benchmark.get("strict_book_selection")
    evidence = benchmark.get("training_evidence", {}).get(
        "strict_book_chronology"
    )
    if not isinstance(selection, dict) or not isinstance(evidence, dict):
        return ""
    candidate_rows = []
    for name, record in evidence.get("candidates", {}).items():
        training = record["training"]
        policy = record["policy_diagnostic"]
        recent = record["later_vintage_evaluation"]
        candidate_rows.append(
            (
                name,
                _number(training["confidence_threshold"], 2),
                "PASS" if training["threshold_qualified"] else "BLOCKED",
                f"{policy['metrics']['markets']:,}",
                _percent(policy["metrics"]["accuracy"]),
                _number(
                    policy["timing"]["median_first_crossing_seconds"],
                    0,
                ),
                f"{recent['metrics']['markets']:,}",
                _percent(recent["metrics"]["coverage"]),
                _percent(recent["metrics"]["accuracy"]),
                _percent(recent["metrics"]["balanced_accuracy"]),
                _percent(recent["metrics"]["up_recall"]),
                _percent(recent["metrics"]["down_recall"]),
                _percent(recent["metrics"]["wilson_lower_95"]),
                _number(
                    recent["timing"]["median_first_crossing_seconds"],
                    0,
                ),
            )
        )
    failed = [
        f"{check['cohort']}: {check['name']}"
        for check in selection.get("statistical_checks", [])
        if not check["passed"]
    ]
    failed_label = (
        "; ".join(failed)
        if failed
        else "all frozen offline statistical gates passed"
    )
    winner = selection.get("winner") or "none"
    return (
        '<section class="panel" style="margin-top:14px">'
        "<h2>Strict-book chronological challenge</h2>"
        "<p>BTC-only and BTC-plus-book models use identical strict-valid rows. "
        "Book quality and provider age route rows but never predict direction. "
        "May–June selects the model policy; July 16–19 is a later, consumed "
        "development challenge. Ten-share book validity is required while "
        "economics remain fixed at five shares.</p>"
        + _table(
            (
                "Candidate",
                "Threshold",
                "Threshold gate",
                "June accepted",
                "June accuracy",
                "June median sec",
                "July accepted",
                "July coverage",
                "July accuracy",
                "July balanced",
                "July UP recall",
                "July DOWN recall",
                "July Wilson",
                "July median sec",
            ),
            candidate_rows,
        )
        + f"<p>Offline selection: <strong>{html.escape(str(selection['status']))}"
        f"</strong>; winner: <strong>{html.escape(str(winner))}</strong>. "
        f"{html.escape(failed_label)}. Runtime evidence is deferred and no "
        "deployment artifact was exported.</p></section>"
    )


def _book_residual_panel(benchmark: dict[str, Any]) -> str:
    selection = benchmark.get("book_residual_selection")
    evidence = benchmark.get("training_evidence", {}).get("book_residual")
    data = benchmark.get("data_evidence", {})
    if not isinstance(selection, dict) or not isinstance(evidence, dict):
        return ""

    final_residual = evidence["final_residual"]
    model = final_residual["model"]
    coefficients = [model["gamma"], *model["beta"]]
    scales: list[float | None] = [None, *model["feature_scales"]]
    coefficient_rows = [
        (
            feature,
            _number(coefficient, 6),
            "unscaled" if scale is None else _number(scale, 6),
        )
        for feature, coefficient, scale in zip(
            model["feature_names"],
            coefficients,
            scales,
            strict=True,
        )
    ]
    cohort_rows = []
    for name, cohort in data.get("cohorts", {}).items():
        if "universal_rows" in cohort:
            cohort_rows.append(
                (
                    name,
                    f"{cohort['universal_rows']:,}",
                    f"{cohort['universal_markets']:,}",
                    f"{cohort['strict_rows']:,}",
                    f"{cohort['strict_markets']:,}",
                    _percent(cohort["strict_market_coverage"]),
                )
            )
        else:
            cohort_rows.append(
                (
                    name,
                    "OOF strict only",
                    "OOF strict only",
                    f"{cohort['rows']:,}",
                    f"{cohort['markets']:,}",
                    "100.00%",
                )
            )

    calibration_rows = []
    for candidate, diagnostic in evidence[
        "direction_time_calibration"
    ].items():
        for cell in diagnostic["cells"]:
            calibration_rows.append(
                (
                    candidate,
                    cell["band"],
                    cell["raw_direction"],
                    f"{cell['markets']:,}",
                    f"{cell['positives']:,}",
                    _number(cell["slope"], 4),
                    _number(cell["intercept"], 4),
                    "PASS" if cell["converged"] else "BLOCKED",
                )
            )

    threshold_rows = [
        (
            candidate,
            _number(record["threshold"], 2),
            "PASS" if record["qualified"] else "BLOCKED",
            f"{max(row['markets'] for row in record['history']):,}",
        )
        for candidate, record in evidence["threshold_selection"].items()
    ]
    holdout = data["sealed_holdout"]
    winner = selection.get("winner") or "none"
    return (
        '<section class="panel" style="margin-top:14px">'
        "<h2>Compact orderbook residual challenge</h2>"
        "<p>The challenger adds one small L2-regularized book correction to "
        "the universal BTC-core logit. It routes only on simultaneous strict "
        "10-share books with an exact prior five-second row; every other row "
        "is the unchanged BTC-core fallback. Quality flags and provider age "
        "never predict direction.</p>"
        "<h3>Chronological cohorts</h3>"
        + _table(
            (
                "Cohort",
                "Universal rows",
                "Universal markets",
                "Strict rows",
                "Strict markets",
                "Strict coverage",
            ),
            cohort_rows,
        )
        + "<h3>Frozen residual coefficients</h3>"
        + _table(("Feature", "Coefficient", "RMS scale"), coefficient_rows)
        + "<h3>Direction/time calibration cells</h3>"
        + _table(
            (
                "Candidate",
                "Band",
                "Raw direction",
                "Markets",
                "Positive rows",
                "Slope",
                "Intercept",
                "Converged",
            ),
            calibration_rows,
        )
        + "<h3>Frozen confidence policies</h3>"
        + _table(
            ("Candidate", "Threshold", "Qualified", "Max accepted"),
            threshold_rows,
        )
        + f"<p>Development selection: <strong>{html.escape(str(selection['status']))}"
        f"</strong>; winner: <strong>{html.escape(str(winner))}</strong>. "
        f"Sealed holdout {html.escape(str(holdout['range_start']))} to "
        f"{html.escape(str(holdout['range_end']))}: "
        f"<strong>{html.escape(str(holdout['status']))}</strong>; "
        f"{html.escape(str(holdout['reason']))}. No runtime artifact was "
        "exported.</p></section>"
    )


def _persistence_analysis_panel(benchmark: dict[str, Any]) -> str:
    analysis = benchmark.get("persistence_analysis")
    if not isinstance(analysis, dict):
        return ""
    rows = []
    for name in benchmark.get("candidate_order", []):
        candidate = analysis.get(name)
        if not isinstance(candidate, dict):
            continue
        followed = candidate["path_behavior"]["followed"]
        reversed_rows = candidate["path_behavior"]["reversed"]
        early = candidate["early"]
        rows.append(
            (
                name,
                candidate["target_kind"],
                candidate["feature_kind"],
                candidate["calibration_kind"],
                f"{early['markets']:,}",
                _percent(early["coverage"]),
                _percent(early["accuracy"]),
                f"{followed['markets']:,}",
                _percent(followed["accuracy"]),
                f"{reversed_rows['markets']:,}",
                _percent(reversed_rows["accuracy"]),
            )
        )
    if not rows:
        return ""
    return (
        '<section class="panel" style="margin-top:14px">'
        "<h2>Path-persistence behavior</h2>"
        "<p>Every probability below is converted back into outcome-space "
        "<code>P(UP)</code> before direction, accuracy, confidence, or economics "
        "are evaluated. A reversal is a real model decision opposite the current "
        "Binance path, not a second strategy.</p>"
        '<div class="table-wrap">'
        + _table(
            (
                "Candidate",
                "Target",
                "Features",
                "Calibration",
                "≤120 sec",
                "Early coverage",
                "Early accuracy",
                "Follow",
                "Follow accuracy",
                "Reverse",
                "Reverse accuracy",
            ),
            rows,
        )
        + "</div></section>"
    )


def _training_evidence_panel(benchmark: dict[str, Any]) -> str:
    evidence = benchmark.get("training_evidence")
    if not isinstance(evidence, dict):
        return ""
    rows = []
    for name, candidate in evidence.get("core_candidates", {}).items():
        rows.append(
            _training_evidence_row(
                name,
                "five-fold walk-forward",
                candidate["out_of_fold"],
                candidate["timing"],
                candidate.get("passed_development"),
            )
        )
    prior_diagnostics = evidence.get("prior_diagnostics")
    reused_prior = (
        isinstance(prior_diagnostics, dict)
        and prior_diagnostics.get("reused_without_retraining") is True
    )
    preopen = evidence.get("preopen_candidate")
    if isinstance(preopen, dict):
        rows.append(
            _training_evidence_row(
                str(preopen["candidate"]),
                (
                    "prior diagnostic — reused, not retrained, not eligible"
                    if reused_prior
                    else "five-fold walk-forward / pre-open BTC"
                ),
                preopen["out_of_fold"],
                preopen["timing"],
                preopen.get("passed_development"),
            )
        )
    book = evidence.get("strict_book_candidate")
    if isinstance(book, dict):
        rows.append(
            _training_evidence_row(
                str(book["candidate"]),
                (
                    "prior diagnostic — reused, not retrained, not eligible"
                    if reused_prior
                    else "clean-book chronological policy"
                ),
                book["metrics"],
                book["timing"],
                bool(book.get("threshold_qualified")),
            )
        )
    if not rows:
        return ""
    headings = (
        "Candidate",
        "Cohort",
        "Accepted",
        "Coverage",
        "Accuracy",
        "Balanced",
        "UP recall",
        "DOWN recall",
        "Wilson lower",
        "Median sec",
        "Early coverage",
        "Core statistical gate",
    )
    return (
        '<section class="panel" style="margin-top:14px">'
        "<h2>Chronological training evidence</h2>"
        '<div class="table-wrap">'
        + _table(headings, rows)
        + "</div></section>"
    )


def _training_selection_panel(benchmark: dict[str, Any]) -> str:
    selection = benchmark.get("training_selection")
    if not isinstance(selection, dict):
        return ""
    rows = []
    for name, candidate in selection.get("candidates", {}).items():
        failed = [
            check["name"]
            for check in candidate.get("checks", [])
            if not check["passed"]
        ]
        rows.append(
            (
                name,
                "PASS" if candidate.get("passed") else "BLOCKED",
                ", ".join(failed) if failed else "all frozen training gates",
                "deferred; no runtime freeze created",
            )
        )
    finalist = selection.get("finalist") or "none"
    return (
        '<section class="panel" style="margin-top:14px">'
        "<h2>Frozen training selection</h2>"
        f"<p>Finalist: <strong>{html.escape(str(finalist))}</strong>. "
        "This selection excludes deferred native runtime evidence and does not "
        "create a deployment artifact.</p>"
        + _table(
            ("Candidate", "Training gate", "Failed checks", "Runtime"),
            rows,
        )
        + "</section>"
    )


def _training_evidence_row(
    name: str,
    cohort: str,
    metrics: dict[str, Any],
    timing: dict[str, Any],
    passed: bool | None,
) -> tuple[str, ...]:
    return (
        name,
        cohort,
        f"{metrics['markets']:,}",
        _percent(metrics["coverage"]),
        _percent(metrics["accuracy"]),
        _percent(metrics["balanced_accuracy"]),
        _percent(metrics["up_recall"]),
        _percent(metrics["down_recall"]),
        _percent(metrics["wilson_lower_95"]),
        _number(timing["median_first_crossing_seconds"], 0),
        _percent(timing["early_entry_coverage"]),
        "PASS" if passed else "BLOCKED",
    )


def _data_evidence_panel(benchmark: dict[str, Any]) -> str:
    evidence = benchmark.get("data_evidence")
    if not isinstance(evidence, dict):
        return ""
    execution = evidence.get("execution", {}).get("totals", {})
    preopen = evidence.get("preopen", {})
    if not execution:
        return ""
    rows = [
        (
            "Exact training range",
            _range_label(evidence.get("training_range")),
        ),
        (
            "Execution-economics cohort",
            _range_label(evidence.get("execution_cohort")),
        ),
        ("Execution-evidence rows", f"{execution['rows']:,}"),
        ("Execution-evidence markets", f"{execution['markets']:,}"),
        (
            "Strict fresh two-sided rows",
            f"{execution['strict_both_side_eligible_rows']:,}",
        ),
        ("Fresh UP rows", f"{execution['up_side_fresh_rows']:,}"),
        ("Fresh DOWN rows", f"{execution['down_side_fresh_rows']:,}"),
        (
            "Complete pre-open markets",
            f"{preopen.get('complete_feature_markets', 0):,}",
        ),
        (
            "Pre-open/book diagnostics",
            (
                "prior run reused by pinned SHA; not retrained or selectable"
                if evidence.get("prior_diagnostics", {}).get(
                    "reused_without_retraining"
                )
                else "trained in this benchmark run"
            ),
        ),
        (
            "Book quality role",
            "eligibility/routing only; never a directional feature",
        ),
        ("Raw PMXT archive", "not read; compact execution snapshots only"),
    ]
    recent = evidence.get("recent_backfill_book_quality")
    if isinstance(recent, dict):
        for cohort in recent.get("cohorts", []):
            rows.append(
                (
                    f"Recent compact book {cohort['date']}",
                    (
                        f"{cohort['markets']:,} markets; "
                        f"{_percent(cohort['strict_valid_rate'])} strict-valid; "
                        f"{cohort['completion']}"
                    ),
                )
            )
        rows.extend(
            (
                (
                    "Recent clean-book total",
                    (
                        f"{recent.get('combined_markets', 0):,} noncontiguous markets; "
                        f"required {recent.get('minimum_holdout_markets', 0):,}"
                    ),
                ),
                (
                    "Recent book decision",
                    str(recent.get("decision", "diagnostic only")),
                ),
            )
        )
    return (
        '<section class="panel" style="margin-top:14px">'
        "<h2>Data-quality and execution evidence</h2>"
        + _table(("Evidence", "Observed"), rows)
        + "</section>"
    )


def _deployment_panel(benchmark: dict[str, Any]) -> str:
    deployment = benchmark.get("deployment")
    if not isinstance(deployment, dict):
        return ""
    reasons = "".join(
        f"<li>{html.escape(str(reason))}</li>"
        for reason in deployment.get("reasons", [])
    )
    return (
        '<section class="panel warning" style="margin-top:14px">'
        "<h2>Deployment decision</h2>"
        f"<p><strong>{html.escape(str(deployment['status']))}</strong> — "
        f"{html.escape(str(deployment['action']))}</p>"
        f"<ul>{reasons}</ul>"
        "</section>"
    )


def _candidate_table(
    benchmark: dict[str, Any],
    candidate_order: list[str],
) -> str:
    headings = (
        "Candidate",
        "Threshold",
        "Accepted",
        "Coverage",
        "Total NoTrade",
        "Confidence NoTrade",
        "Unavailable",
        "Accuracy",
        "Balanced",
        "UP recall",
        "DOWN recall",
        "Wilson lower",
        "ECE",
        "Median sec",
        "P90 sec",
        "Evidence / selected",
        "Executable / evidence",
        "Executable / all selected",
        "Median VWAP",
        "Fee/share",
        "Direct edge",
        "Net expectancy",
        "Loss streak",
        "Drawdown",
        "Gate",
    )
    rows = []
    for name in candidate_order:
        candidate = benchmark["candidates"][name]
        metrics = candidate["own_policy"]
        execution = metrics["execution"]
        advance = candidate["advance"]
        gate = (
            "CONTROL"
            if advance["is_control"]
            else ("PASS" if advance["benchmark_passed"] else "BLOCKED")
        )
        rows.append(
            (
                name,
                _policy_threshold_label(candidate["policy"]),
                f"{metrics['markets']:,}",
                _percent(metrics["coverage"]),
                f"{metrics['no_trade_markets']:,}",
                f"{metrics['confidence_no_trade_markets']:,}",
                f"{metrics['data_unavailable_markets']:,}",
                _percent(metrics["accuracy"]),
                _percent(metrics["balanced_accuracy"]),
                _percent(metrics["up_recall"]),
                _percent(metrics["down_recall"]),
                _percent(metrics["wilson_lower_95"]),
                _percent(metrics["expected_calibration_error"]),
                _number(metrics["median_seconds_elapsed"], 0),
                _number(metrics["p90_seconds_elapsed"], 0),
                _percent(execution.get("execution_evidence_coverage")),
                _percent(execution.get("executable_coverage_within_evidence")),
                _percent(
                    execution.get(
                        "executable_coverage_all_selected",
                        execution.get("executable_coverage"),
                    )
                ),
                _currency(execution["median_selected_ask_vwap_5"], 4),
                _currency(execution["mean_fee_per_share"], 5),
                _signed_currency(execution["mean_direct_edge_per_share"], 5),
                _signed_currency(
                    execution["realized_net_expectancy_per_trade"],
                    4,
                ),
                _number(execution["maximum_net_loss_streak"], 0),
                _currency(execution["maximum_drawdown"], 4),
                gate,
            )
        )
    return _table(headings, rows)


def _advance_tables(
    benchmark: dict[str, Any],
    candidate_order: list[str],
) -> str:
    sections = []
    for name in candidate_order:
        advance = benchmark["candidates"][name]["advance"]
        if advance["is_control"]:
            continue
        status = "PASS" if advance["benchmark_passed"] else "BLOCKED"
        deployment = (
            "deployment-qualified"
            if advance["deployment_qualified"]
            else "not deployment-qualified"
        )
        rows = [
            (
                check["name"],
                _format_value(check["observed"]),
                f"{check['operator']} {_format_value(check['required'])}",
                "PASS" if check["passed"] else "BLOCKED",
            )
            for check in advance["checks"]
        ]
        sections.append(
            f"<h3>{html.escape(name)} · {status} · {deployment}</h3>"
            + '<div class="table-wrap">'
            + _table(("Gate", "Observed", "Required", "Status"), rows)
            + "</div>"
        )
    return "".join(sections) or "<p>No challenger candidates were supplied.</p>"


def _time_band_table(
    benchmark: dict[str, Any],
    candidate_order: list[str],
) -> str:
    rows = []
    for name in candidate_order:
        for band in benchmark["candidates"][name]["time_bands"]:
            rows.append(
                (
                    name,
                    band["band"],
                    f"{band['markets']:,}",
                    _percent(band["coverage"]),
                    _percent(band["accuracy"]),
                    _percent(band["balanced_accuracy"]),
                    _percent(band["up_recall"]),
                    _percent(band["down_recall"]),
                    _percent(band["wilson_lower_95"]),
                )
            )
    return _table(
        (
            "Candidate",
            "Seconds",
            "Accepted",
            "Eligible coverage",
            "Accuracy",
            "Balanced",
            "UP recall",
            "DOWN recall",
            "Wilson lower",
        ),
        rows,
    )


def _common_comparison_table(benchmark: dict[str, Any]) -> str:
    rows = []
    for name in sorted(benchmark["common_comparisons"]):
        comparison = benchmark["common_comparisons"][name]
        for checkpoint in comparison["checkpoints"]:
            rows.append(
                (
                    name,
                    str(checkpoint["seconds_elapsed"]),
                    f"{checkpoint['common_markets']:,}",
                    _percent(checkpoint["control"]["accuracy"]),
                    _percent(checkpoint["candidate"]["accuracy"]),
                    _signed_percent_points(checkpoint["accuracy_delta"]),
                    _signed_percent_points(checkpoint["balanced_accuracy_delta"]),
                    _signed_percent_points(checkpoint["up_recall_delta"]),
                    _signed_percent_points(checkpoint["down_recall_delta"]),
                )
            )
    return _table(
        (
            "Candidate",
            "Checkpoint",
            "Common markets",
            "Control accuracy",
            "Candidate accuracy",
            "Accuracy delta",
            "Balanced delta",
            "UP recall delta",
            "DOWN recall delta",
        ),
        rows,
    )


def _benchmark_figures(benchmark: dict[str, Any]) -> list[Any]:
    if go is None:
        return []
    names = benchmark["candidate_order"]
    candidates = benchmark["candidates"]
    accuracy = go.Figure()
    accuracy.add_bar(
        name="Accuracy",
        x=names,
        y=[candidates[name]["own_policy"]["accuracy"] for name in names],
    )
    accuracy.add_bar(
        name="Coverage",
        x=names,
        y=[candidates[name]["own_policy"]["coverage"] for name in names],
    )
    _style(accuracy, "Own-policy accuracy and eligible-market coverage", "Rate")

    timing = go.Figure()
    timing.add_bar(
        name="Median",
        x=names,
        y=[candidates[name]["own_policy"]["median_seconds_elapsed"] for name in names],
    )
    timing.add_bar(
        name="P90",
        x=names,
        y=[candidates[name]["own_policy"]["p90_seconds_elapsed"] for name in names],
    )
    _style(timing, "First accepted prediction timing", "Seconds elapsed")

    economics = go.Figure()
    economics.add_bar(
        name="Net expectancy / executable trade",
        x=names,
        y=[
            candidates[name]["own_policy"]["execution"][
                "realized_net_expectancy_per_trade"
            ]
            for name in names
        ],
    )
    economics.add_bar(
        name="Mean direct edge / share",
        x=names,
        y=[
            candidates[name]["own_policy"]["execution"][
                "mean_direct_edge_per_share"
            ]
            for name in names
        ],
    )
    economics.add_hline(y=0, line_color="#9ca9c7")
    _style(economics, "Five-share execution economics", "USD")

    checkpoints = go.Figure()
    for name in sorted(benchmark["common_comparisons"]):
        rows = benchmark["common_comparisons"][name]["checkpoints"]
        checkpoints.add_scatter(
            x=[row["seconds_elapsed"] for row in rows],
            y=[row["accuracy_delta"] for row in rows],
            name=name,
            mode="lines+markers",
        )
    checkpoints.add_hline(y=0, line_color="#9ca9c7")
    _style(
        checkpoints,
        "Accuracy delta on common exact-timestamp cohorts",
        "Candidate minus control",
    )
    return [accuracy, timing, economics, checkpoints]


def _style(figure: Any, title: str, y_title: str) -> None:
    figure.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#141c30",
        plot_bgcolor="#141c30",
        font={"color": "#e9eefc"},
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        legend={"orientation": "h"},
    )
    figure.update_yaxes(title=y_title)


def _table(headings: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    header = "".join(f"<th>{html.escape(heading)}</th>" for heading in headings)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _card(label: str, value: str, css_class: str = "") -> str:
    return (
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value {css_class}">{html.escape(value)}</div></div>'
    )


def _evidence_status(evidence: dict[str, Any]) -> str:
    if evidence["kind"] == "holdout" and evidence["independent"]:
        return "independent holdout"
    if evidence["kind"] == "holdout":
        return "non-independent holdout"
    return "development only"


def _evidence_css(evidence: dict[str, Any]) -> str:
    return (
        "pass"
        if evidence["kind"] == "holdout" and evidence["independent"]
        else "blocked"
    )


def _policy_threshold_label(policy: dict[str, Any]) -> str:
    if policy.get("selection_mode") == "chronological_preselected":
        minimum = policy.get("confidence_threshold_min")
        maximum = policy.get("confidence_threshold_max")
        if minimum is None or maximum is None:
            return "chronological"
        return f"chronological {minimum:.2f}–{maximum:.2f}"
    return _decimal(policy.get("confidence_threshold"), 2)


def _range_label(value: Any) -> str:
    if not isinstance(value, dict):
        return "N/A"
    start = value.get("start")
    end = value.get("end_exclusive")
    if start is None or end is None:
        return "N/A"
    days = value.get("calendar_days")
    suffix = f" ({days} calendar days)" if days is not None else ""
    return f"[{start}, {end}){suffix}"


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _signed_percent_points(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:+.2f} pp"


def _decimal(value: float | None, places: int) -> str:
    return "N/A" if value is None else f"{value:.{places}f}"


def _number(value: float | None, places: int) -> str:
    return "N/A" if value is None else f"{value:.{places}f}"


def _currency(value: float | None, places: int) -> str:
    return "N/A" if value is None else f"${value:.{places}f}"


def _signed_currency(value: float | None, places: int) -> str:
    return "N/A" if value is None else f"${value:+.{places}f}"


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
