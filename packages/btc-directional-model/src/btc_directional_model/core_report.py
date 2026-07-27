from __future__ import annotations

import html
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .core_extract import write_json_atomic


def generate_core_report(
    run_dir: Path,
    metrics: dict[str, Any] | None = None,
) -> Path:
    if metrics is None:
        final_path = run_dir / "metrics.json"
        development_path = run_dir / "development-metrics.json"
        metrics = json.loads(
            (final_path if final_path.exists() else development_path).read_text()
        )
    selected_name = metrics["selected_candidate"]
    selected = metrics["candidates"][selected_name]
    holdout = metrics.get("holdout")
    status = metrics["status"]
    figures = [
        candidate_comparison(metrics),
        fold_uplift_figure(selected),
        policy_threshold_figure(metrics),
    ]
    if holdout is not None:
        figures.extend(
            [
                confusion_figure(holdout),
                daily_figure(holdout),
                reliability_figure(holdout),
                time_figure(holdout),
            ]
        )
    plot_html = []
    for index, figure in enumerate(figures):
        plot_html.append(
            figure.to_html(
                full_html=False,
                include_plotlyjs=index == 0,
                config={"displaylogo": False, "responsive": True},
            )
        )
    document = render_document(metrics, plot_html)
    destination = run_dir / "report.html"
    destination.write_text(document)
    write_json_atomic(
        run_dir / "progress.json",
        {
            "stage": "report_complete",
            "completion": 1.0,
            "details": {"report": destination.name, "status": status},
        },
    )
    return destination


def render_document(metrics: dict[str, Any], plots: list[str]) -> str:
    selected_name = metrics["selected_candidate"]
    selected = metrics["candidates"][selected_name]
    holdout = metrics.get("holdout")
    status = metrics["status"]
    status_class = (
        "pass"
        if status in {"candidate_ready_for_freeze", "prediction_qualified"}
        else "fail"
    )
    if holdout is None:
        primary = selected["out_of_fold"]
        paired = selected["paired"]
        evaluation_label = "Walk-forward development"
        qualification_rows = pre_holdout_checks(metrics)
    else:
        primary = holdout["metrics"]
        paired = holdout["paired"]
        evaluation_label = holdout_evaluation_label(metrics)
        qualification_rows = "".join(
            gate_row(check) for check in holdout["qualification_checks"]
        )
    cards = "".join(
        [
            card("Status", status.replace("_", " "), status_class),
            card("Prediction role", "selective path persistence"),
            card("Selected model", selected_name),
            card(f"{evaluation_label} accuracy", percent(primary["accuracy"])),
            card("Balanced accuracy", percent(primary["balanced_accuracy"])),
            card("UP recall", percent(primary["up_recall"])),
            card("DOWN recall", percent(primary["down_recall"])),
            card("Accepted coverage", percent(primary["coverage"])),
            card("Accepted markets", f"{primary['markets']:,}"),
            card("Binance path-sign uplift", signed_percent(paired["accuracy_uplift"])),
            card("Maximum loss streak", str(primary["maximum_consecutive_losses"])),
        ]
    )
    data = metrics["data"]
    blocking = metrics.get("blocking_reasons", [])
    blocking_panel = ""
    if blocking:
        blocking_panel = (
            '<section class="panel"><h2>Blocking reasons</h2><ul>'
            + "".join(f"<li>{html.escape(reason)}</li>" for reason in blocking)
            + "</ul></section>"
        )
    plots_html = "".join(f'<section class="panel plot">{plot}</section>' for plot in plots)
    holdout_note = (
        "The frozen candidate was evaluated exactly once on the isolated holdout."
        if holdout is not None
        else "The holdout was not opened because pre-holdout qualification did not pass."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Capitonic BTC Core Training</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1020; --panel:#141c30; --line:#27324c;
  --text:#e9eefc; --muted:#9ca9c7; --accent:#67d5ff; --pass:#5ee6a8; --fail:#ff7b8d; }}
* {{ box-sizing:border-box }} body {{ margin:0;background:var(--bg);color:var(--text);
  font:14px/1.5 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }}
main {{ max-width:1500px;margin:auto;padding:24px }} h1 {{ margin:0;font-size:28px }}
h2 {{ margin:0 0 12px;font-size:17px }} .subtitle {{ color:var(--muted);margin:6px 0 20px }}
.cards {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:14px }}
.card,.panel {{ background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px }}
.label {{ color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em }}
.value {{ font-size:21px;font-weight:700;margin-top:5px }} .pass {{ color:var(--pass) }}
.fail {{ color:var(--fail) }} .grid {{ display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px }}
.plot {{ min-height:390px }} table {{ width:100%;border-collapse:collapse }}
td,th {{ padding:8px;border-bottom:1px solid var(--line);text-align:left }} th {{ color:var(--muted) }}
code {{ color:var(--accent);word-break:break-all }} ul {{ margin:0;padding-left:20px }}
@media(max-width:900px) {{ .grid {{ grid-template-columns:1fr }} }}
</style></head><body><main>
<h1>BTC five-minute universal core</h1>
<div class="subtitle">{html.escape(evaluation_label)} · first confidence crossing ·
no orderbook features · trading deployment remains blocked pending execution economics</div>
<div class="cards">{cards}</div>
<section class="panel" style="margin-bottom:14px"><h2>Qualification contract</h2>
<table><thead><tr><th>Gate</th><th>Observed</th><th>Required</th><th>Status</th></tr></thead>
<tbody>{qualification_rows}</tbody></table></section>
{blocking_panel}
<section class="panel" style="margin:14px 0"><h2>Data and isolation evidence</h2>
<table><tbody>
<tr><td>Source range</td><td>{html.escape(data["range_start"])} – {html.escape(data["range_end"])}</td></tr>
<tr><td>Candidate markets</td><td>{data["candidate_markets"]:,}</td></tr>
<tr><td>Candidate rows</td><td>{data["candidate_rows"]:,}</td></tr>
<tr><td>Dense-history markets</td><td>{data["history_complete_markets"]:,}</td></tr>
<tr><td>Incomplete histories excluded</td><td>{data["history_incomplete_markets"]:,}</td></tr>
<tr><td>Final-price mismatches quarantined</td><td>{data["final_price_mismatch_markets"]:,}</td></tr>
<tr><td>Feature schema</td><td><code>{html.escape(data["feature_schema_version"])}</code></td></tr>
<tr><td>Feature file SHA-256</td><td><code>{html.escape(data["feature_file_sha256"])}</code></td></tr>
<tr><td>Holdout handling</td><td>{html.escape(holdout_note)}</td></tr>
</tbody></table></section>
<div class="grid">{plots_html}</div>
<section class="panel" style="margin-top:14px"><h2>Runtime provenance</h2>
{runtime_table(metrics)}</section>
</main></body></html>"""


def holdout_evaluation_label(metrics: dict[str, Any]) -> str:
    holdout_range = metrics.get("freeze", {}).get("holdout_range")
    if not isinstance(holdout_range, dict):
        return "Untouched holdout"
    try:
        start = datetime.fromisoformat(holdout_range["start"]).date()
        end = datetime.fromisoformat(holdout_range["end"]).date() - timedelta(days=1)
    except (KeyError, TypeError, ValueError):
        return "Untouched holdout"
    if end < start:
        return "Untouched holdout"
    if start.year == end.year and start.month == end.month:
        date_span = f"{start:%B} {start.day}–{end.day}"
    elif start.year == end.year:
        date_span = f"{start:%B} {start.day}–{end:%B} {end.day}"
    else:
        date_span = (
            f"{start:%B} {start.day}, {start.year}–"
            f"{end:%B} {end.day}, {end.year}"
        )
    return f"Untouched {date_span} holdout"


def candidate_comparison(metrics: dict[str, Any]) -> go.Figure:
    names = list(metrics["candidates"])
    figure = go.Figure()
    figure.add_bar(
        name="Model",
        x=names,
        y=[
            metrics["candidates"][name]["out_of_fold"]["accuracy"] for name in names
        ],
    )
    figure.add_bar(
        name="Binance path sign",
        x=names,
        y=[
            metrics["candidates"][name]["baseline"]["accuracy"] for name in names
        ],
    )
    return styled(
        figure,
        "Walk-forward accuracy on matched accepted cohorts",
        "Accuracy",
    )


def fold_uplift_figure(selected: dict[str, Any]) -> go.Figure:
    folds = selected["folds"]
    figure = go.Figure(
        go.Bar(
            x=[f"Fold {row['fold_index'] + 1}" for row in folds],
            y=[row["paired"]["accuracy_uplift"] for row in folds],
            marker_color=[
                "#5ee6a8" if row["paired"]["accuracy_uplift"] >= 0 else "#ff7b8d"
                for row in folds
            ],
        )
    )
    figure.add_hline(y=0, line_color="#9ca9c7")
    return styled(figure, "Selected-model paired uplift by fold", "Accuracy uplift")


def policy_threshold_figure(metrics: dict[str, Any]) -> go.Figure:
    rows = metrics["policy"]["threshold_history"]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_scatter(
        x=[row["threshold"] for row in rows],
        y=[row["accuracy"] for row in rows],
        name="Accuracy",
        mode="lines+markers",
        secondary_y=False,
    )
    figure.add_scatter(
        x=[row["threshold"] for row in rows],
        y=[row["coverage"] for row in rows],
        name="Coverage",
        mode="lines",
        secondary_y=True,
    )
    figure.add_vline(
        x=metrics["policy"]["confidence_threshold"],
        line_dash="dash",
        line_color="#67d5ff",
    )
    figure.update_yaxes(title_text="Accuracy", secondary_y=False)
    figure.update_yaxes(title_text="Coverage", secondary_y=True)
    return styled(figure, "Frozen confidence policy", "")


def confusion_figure(holdout: dict[str, Any]) -> go.Figure:
    matrix = holdout["metrics"]["confusion_matrix"]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predicted DOWN", "Predicted UP"],
            y=["Actual DOWN", "Actual UP"],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Viridis",
        )
    )
    return styled(figure, "Untouched-holdout confusion matrix", "")


def daily_figure(holdout: dict[str, Any]) -> go.Figure:
    rows = holdout["daily_accuracy"]
    figure = go.Figure()
    figure.add_scatter(
        x=[row["date"] for row in rows],
        y=[row["accuracy"] for row in rows],
        name="Model",
        mode="lines+markers",
    )
    figure.add_scatter(
        x=[row["date"] for row in rows],
        y=[row["baseline_accuracy"] for row in rows],
        name="Binance path sign",
        mode="lines+markers",
    )
    return styled(figure, "Daily holdout accuracy", "Accuracy")


def reliability_figure(holdout: dict[str, Any]) -> go.Figure:
    rows = holdout["reliability"]
    figure = go.Figure()
    figure.add_scatter(
        x=[row["mean_probability_up"] for row in rows],
        y=[row["observed_up_rate"] for row in rows],
        mode="lines+markers",
        name="Observed",
        text=[row["markets"] for row in rows],
    )
    figure.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Ideal",
        line={"dash": "dash"},
    )
    return styled(figure, "Holdout probability reliability", "Observed UP rate")


def time_figure(holdout: dict[str, Any]) -> go.Figure:
    rows = holdout["time_accuracy"]
    figure = go.Figure()
    figure.add_scatter(
        x=[row["seconds_elapsed"] for row in rows],
        y=[row["accuracy"] for row in rows],
        mode="lines+markers",
        name="Model",
        text=[row["markets"] for row in rows],
    )
    figure.add_scatter(
        x=[row["seconds_elapsed"] for row in rows],
        y=[row["baseline_accuracy"] for row in rows],
        mode="lines",
        name="Binance path sign",
    )
    return styled(figure, "Accuracy by first accepted prediction time", "Accuracy")


def styled(figure: go.Figure, title: str, y_title: str) -> go.Figure:
    figure.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#141c30",
        plot_bgcolor="#141c30",
        font={"color": "#e9eefc"},
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        legend={"orientation": "h"},
    )
    if y_title:
        figure.update_yaxes(title=y_title)
    return figure


def pre_holdout_checks(metrics: dict[str, Any]) -> str:
    selected = metrics["candidates"][metrics["selected_candidate"]]
    required_folds = metrics["configuration"]["gates"][
        "minimum_nonnegative_uplift_folds"
    ]
    checks = [
        {
            "name": "Nonnegative same-time path-uplift folds",
            "observed": selected["nonnegative_uplift_folds"],
            "operator": ">=",
            "target": required_folds,
            "passed": selected["nonnegative_uplift_folds"] >= required_folds,
        },
        {
            "name": "Walk-forward development contract",
            "observed": selected["passed_development"],
            "operator": "=",
            "target": True,
            "passed": selected["passed_development"],
        },
        {
            "name": "Policy-selection contract",
            "observed": metrics["policy"]["passed"],
            "operator": "=",
            "target": True,
            "passed": metrics["policy"]["passed"],
        },
    ]
    return "".join(gate_row(check) for check in checks)


def gate_row(check: dict[str, Any]) -> str:
    css = "pass" if check["passed"] else "fail"
    status = "PASS" if check["passed"] else "BLOCKED"
    return (
        f"<tr><td>{html.escape(str(check['name']))}</td>"
        f"<td>{format_value(check['observed'])}</td>"
        f"<td>{html.escape(str(check['operator']))} {format_value(check['target'])}</td>"
        f'<td class="{css}">{status}</td></tr>'
    )


def runtime_table(metrics: dict[str, Any]) -> str:
    provenance = metrics["runtime_provenance"]
    git = provenance["git"]
    dependencies = ", ".join(
        f"{name} {version}" for name, version in provenance["dependencies"].items()
    )
    return (
        "<table><tbody>"
        f"<tr><td>Python</td><td>{html.escape(provenance['python'])}</td></tr>"
        f"<tr><td>Platform</td><td>{html.escape(provenance['platform'])}</td></tr>"
        f"<tr><td>Dependencies</td><td>{html.escape(dependencies)}</td></tr>"
        f"<tr><td>Git commit</td><td><code>{html.escape(str(git['commit']))}</code></td></tr>"
        f"<tr><td>Git branch</td><td>{html.escape(str(git['branch']))}</td></tr>"
        f"<tr><td>Git dirty</td><td>{html.escape(str(git['dirty']))}</td></tr>"
        f"<tr><td>Source tree SHA-256</td><td><code>{html.escape(provenance['source_tree_sha256'])}</code></td></tr>"
        "</tbody></table>"
    )


def card(label: str, value: str, css_class: str = "") -> str:
    return (
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value {css_class}">{html.escape(value)}</div></div>'
    )


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def signed_percent(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))
