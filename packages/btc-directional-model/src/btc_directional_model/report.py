from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .train import update_progress


def generate_report(run_dir: Path, metrics: dict[str, Any] | None = None) -> Path:
    if metrics is None:
        metrics = json.loads((run_dir / "metrics.json").read_text())
    selected_name = metrics["selected_feature_group"]
    selected = metrics["groups"][selected_name]
    qualification = metrics["qualification"]

    figures = [
        group_comparison(metrics),
        confusion_figure(selected),
        threshold_figure(selected),
        confidence_figure(selected),
        time_figure(selected),
        daily_figure(selected),
        coefficient_figure(selected),
        tuning_figure(metrics),
    ]
    figure_html = []
    for index, figure in enumerate(figures):
        figure_html.append(
            figure.to_html(
                full_html=False,
                include_plotlyjs=index == 0,
                config={"displaylogo": False, "responsive": True},
            )
        )

    test = selected["test"]
    first_executable = selected.get("first_executable_test", selected.get("executable_test"))
    selected_executable = selected.get("selected_prediction_executable")
    baselines = selected["test_baselines"]
    holdout_is_independent = qualification.get("holdout_is_independent", True)
    if not holdout_is_independent:
        status = "EXPLORATORY — HOLDOUT CONSUMED"
    else:
        status = "QUALIFIED" if qualification["passed"] else "NOT QUALIFIED"
    status_class = "pass" if qualification["passed"] else "fail"
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>BTC directional model — {html.escape(metrics["run_id"])}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b1020; --panel:#151c31; --line:#2a3554;
      --text:#e8edf8; --muted:#9aa8c7; --accent:#56d4b3; --danger:#ff6b7a; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.45 Inter,system-ui,sans-serif; }}
    main {{ max-width:1440px; margin:auto; padding:28px; }}
    h1,h2 {{ margin:0 0 12px; }} h1 {{ font-size:30px; }} h2 {{ font-size:19px; }}
    .sub {{ color:var(--muted); margin-bottom:22px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(185px,1fr)); gap:12px; margin:20px 0; }}
    .card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }}
    .value {{ font-size:28px; font-weight:700; margin-top:5px; }} .label {{ color:var(--muted); }}
    .pass {{ color:var(--accent); }} .fail {{ color:var(--danger); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(560px,1fr)); gap:14px; }}
    .plot {{ min-height:410px; }} table {{ width:100%; border-collapse:collapse; }}
    th,td {{ text-align:left; padding:8px; border-bottom:1px solid var(--line); }} th {{ color:var(--muted); }}
    code {{ color:var(--accent); }}
  </style>
</head>
<body><main>
  <h1>BTC five-minute directional model</h1>
  <div class="sub">Run <code>{html.escape(metrics["run_id"])}</code> · selected feature group
    <code>{html.escape(selected_name)}</code> · chronological holdout evidence</div>
  <div class="cards">
    {card("Qualification", status, status_class)}
    {card("Independent holdout", "yes" if holdout_is_independent else "no", "pass" if holdout_is_independent else "fail")}
    {card("Accepted prediction accuracy", percent(test["accuracy"]))}
    {card("95% lower confidence bound", percent(test["wilson_lower_95"]))}
    {card("Accepted test markets", f"{test['markets']:,}")}
    {card("Confidence threshold", f"{selected['confidence_threshold']:.2f}")}
    {card("First-executable policy accuracy", percent(first_executable["accuracy"]))}
    {card("First-executable holdout markets", f"{first_executable['markets']:,}")}
    {card("Executable at original decision", executable_card(selected_executable))}
    {card("Maximum test loss streak", str(test["maximum_consecutive_losses"]))}
  </div>
  <div class="panel" style="margin-bottom:14px">
    <h2>Acceptance contract</h2>
    <table><tbody>
      {criterion("Accuracy", test["accuracy"], qualification["target_accuracy"])}
      {criterion("Wilson lower bound", test["wilson_lower_95"], qualification["target_wilson_lower"])}
      {boolean_criterion("Independent holdout", holdout_is_independent)}
      {boolean_criterion("Calibration threshold gate", qualification.get("calibration_threshold_passed", selected.get("threshold_qualified_on_calibration", False)))}
      {boolean_criterion("Model optimizer convergence", qualification.get("optimizer_converged", selected.get("converged", False)))}
      {boolean_criterion("Probability calibrator convergence", qualification.get("calibrator_converged", selected.get("calibrator_converged", False)))}
      <tr><td>Accepted test markets</td><td>{test["markets"]:,}</td>
        <td>≥ {qualification["minimum_test_markets"]:,}</td></tr>
      <tr><td>Train/test accuracy gap</td><td>{percent(abs(qualification["observed_train_test_accuracy_gap"]))}</td>
        <td>≤ {percent(qualification["maximum_train_test_accuracy_gap"])}</td></tr>
    </tbody></table>
  </div>
  <div class="panel" style="margin-bottom:14px">
    <h2>Holdout status</h2>
    <p>{html.escape(qualification.get("holdout_note", "No holdout note was recorded."))}</p>
  </div>
  <div class="panel" style="margin-bottom:14px">
    <h2>Execution-gate diagnostic</h2>
    <p>The first-executable policy waits for the first threshold-crossing prediction whose selected
      side can be bought from the recorded book. It is a distinct timing policy, not a subset of the
      original prediction timestamps, and it does not override the overall qualification result.</p>
    <table><tbody>
      <tr><td>First-executable accuracy</td><td>{percent(first_executable["accuracy"])}</td></tr>
      <tr><td>First-executable markets</td><td>{first_executable["markets"]:,}</td></tr>
      {selected_executable_row(selected_executable)}
    </tbody></table>
  </div>
  <div class="panel" style="margin-bottom:14px">
    <h2>Same-market baseline comparison</h2>
    <table><thead><tr><th>Predictor</th><th>Accuracy</th><th>Markets</th><th>95% lower bound</th></tr></thead>
    <tbody>
      {baseline_row("Selected model", test)}
      {baseline_row("Binance sign from opening", baselines["binance_sign"])}
      {baseline_row("Polymarket favorite", baselines["polymarket_favorite"])}
      {baseline_row("Training-set majority", baselines["majority_up"])}
    </tbody></table>
  </div>
  <div class="grid">{"".join(f'<section class="panel plot">{item}</section>' for item in figure_html)}</div>
  <div class="panel" style="margin-top:14px">
    <h2>Data and runtime evidence</h2>
    <table><tbody>
      <tr><td>Candidate markets</td><td>{metrics["data"]["candidate_markets"]:,}</td></tr>
      <tr><td>Candidate rows</td><td>{metrics["data"]["candidate_rows"]:,}</td></tr>
      <tr><td>Dense-history markets</td><td>{metrics["data"].get("history_complete_markets", "not recorded")}</td></tr>
      <tr><td>Final-price audit coverage</td><td>{metrics["data"]["markets_with_final_price"]:,}</td></tr>
      <tr><td>Official labels matching available final-price audit</td><td>{metrics["data"]["matching_final_price_labels"]:,}</td></tr>
      <tr><td>Up / Down class balance</td><td>{metrics["data"]["class_up_markets"]:,} / {metrics["data"]["class_down_markets"]:,}</td></tr>
      <tr><td>Train range</td><td>{html.escape(split_range(metrics, "train"))}</td></tr>
      <tr><td>Calibration range</td><td>{html.escape(split_range(metrics, "calibration"))}</td></tr>
      {calibration_subsplit_rows(selected)}
      <tr><td>Holdout range</td><td>{html.escape(split_range(metrics, "test"))}</td></tr>
      <tr><td>Threshold selection objective</td><td>{html.escape(selected.get("threshold_selection_objective", "legacy lowest qualifying threshold"))}</td></tr>
      <tr><td>Training backend</td><td>{html.escape(metrics["training_backend"]["name"])} ({metrics["training_backend"]["device"]})</td></tr>
      <tr><td>Backend rationale</td><td>{html.escape(metrics["training_backend"]["reason"])}</td></tr>
      <tr><td>Selected fit converged</td><td>{selected.get("converged", "not recorded")}</td></tr>
      {runtime_rows(metrics)}
    </tbody></table>
  </div>
</main></body></html>"""
    destination = run_dir / "report.html"
    destination.write_text(document)
    update_progress(run_dir, "report_complete", 1.0, {"report": destination.name})
    return destination


def card(label: str, value: str, css_class: str = "") -> str:
    return f'<div class="card"><div class="label">{label}</div><div class="value {css_class}">{value}</div></div>'


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def criterion(name: str, observed: float, target: float) -> str:
    return f"<tr><td>{name}</td><td>{percent(observed)}</td><td>≥ {percent(target)}</td></tr>"


def boolean_criterion(name: str, passed: bool) -> str:
    observed = "passed" if passed else "failed"
    return f"<tr><td>{name}</td><td>{observed}</td><td>must pass before holdout</td></tr>"


def executable_card(metrics: dict[str, Any] | None) -> str:
    if metrics is None:
        return "not recorded"
    return f"{metrics['markets']:,} markets"


def selected_executable_row(metrics: dict[str, Any] | None) -> str:
    if metrics is None:
        return ""
    return (
        "<tr><td>Original predictions executable at their selected timestamp</td>"
        f"<td>{metrics['markets']:,} ({percent(metrics['accuracy'])} accuracy)</td></tr>"
    )


def split_range(metrics: dict[str, Any], name: str) -> str:
    split = metrics["split"][name]
    return f"{split['range_start']} – {split['range_end']} ({split['markets']:,} markets)"


def calibration_subsplit_rows(selected: dict[str, Any]) -> str:
    splits = selected.get("calibration_subsplit")
    if splits is None:
        return ""
    calibration = splits["probability_calibration"]
    policy = splits["policy_selection"]
    return (
        "<tr><td>Probability-calibration subrange</td>"
        f"<td>{html.escape(calibration['range_start'])} – "
        f"{html.escape(calibration['range_end'])} ({calibration['markets']:,} markets)</td></tr>"
        "<tr><td>Threshold/group-selection subrange</td>"
        f"<td>{html.escape(policy['range_start'])} – "
        f"{html.escape(policy['range_end'])} ({policy['markets']:,} markets)</td></tr>"
    )


def runtime_rows(metrics: dict[str, Any]) -> str:
    provenance = metrics.get("runtime_provenance")
    if provenance is None:
        return ""
    dependencies = ", ".join(
        f"{name} {value}" for name, value in provenance["dependencies"].items()
    )
    git = provenance["git"]
    return (
        f"<tr><td>Python</td><td>{html.escape(provenance['python'])}</td></tr>"
        f"<tr><td>Dependencies</td><td>{html.escape(dependencies)}</td></tr>"
        f"<tr><td>Source tree SHA-256</td><td><code>{html.escape(provenance['source_tree_sha256'])}</code></td></tr>"
        f"<tr><td>Git provenance</td><td>{html.escape(str(git))}</td></tr>"
    )


def baseline_row(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"<tr><td>{name}</td><td>{percent(metrics['accuracy'])}</td>"
        f"<td>{metrics['markets']:,}</td><td>{percent(metrics['wilson_lower_95'])}</td></tr>"
    )


def group_comparison(metrics: dict[str, Any]) -> go.Figure:
    names = list(metrics["groups"])
    figure = go.Figure()
    figure.add_bar(
        name="Policy selection",
        x=names,
        y=[metrics["groups"][name]["calibration"]["accuracy"] for name in names],
    )
    figure.add_bar(
        name="Chronological holdout",
        x=names,
        y=[metrics["groups"][name]["test"]["accuracy"] for name in names],
    )
    return styled(figure, "Feature-group directional accuracy", "Accuracy")


def confusion_figure(selected: dict[str, Any]) -> go.Figure:
    matrix = selected["test"]["confusion_matrix"]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predicted Down", "Predicted Up"],
            y=["Actual Down", "Actual Up"],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Viridis",
        )
    )
    return styled(figure, "Chronological-holdout confusion matrix", "")


def threshold_figure(selected: dict[str, Any]) -> go.Figure:
    rows = selected["threshold_history"]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_scatter(
        x=[row["threshold"] for row in rows],
        y=[row["accuracy"] for row in rows],
        name="Policy-selection accuracy",
        mode="lines+markers",
        secondary_y=False,
    )
    figure.add_scatter(
        x=[row["threshold"] for row in rows],
        y=[row["markets"] for row in rows],
        name="Accepted markets",
        mode="lines",
        secondary_y=True,
    )
    figure.add_vline(x=selected["confidence_threshold"], line_dash="dash", line_color="#ffcc66")
    figure.update_yaxes(title_text="Accuracy", secondary_y=False)
    figure.update_yaxes(title_text="Markets", secondary_y=True)
    return styled(figure, "Confidence threshold selected before test", "")


def confidence_figure(selected: dict[str, Any]) -> go.Figure:
    rows = selected["confidence_buckets"]
    figure = go.Figure(
        go.Bar(
            x=[f"{row['bucket']:.2f}" for row in rows],
            y=[row["accuracy"] for row in rows],
            text=[row["markets"] for row in rows],
            name="Accuracy",
        )
    )
    return styled(figure, "Test accuracy by model confidence", "Accuracy")


def time_figure(selected: dict[str, Any]) -> go.Figure:
    rows = selected["time_buckets"]
    figure = go.Figure(
        go.Scatter(
            x=[row["seconds_elapsed"] for row in rows],
            y=[row["accuracy"] for row in rows],
            mode="lines+markers",
            text=[row["markets"] for row in rows],
        )
    )
    return styled(figure, "Accuracy by first accepted prediction time", "Accuracy")


def daily_figure(selected: dict[str, Any]) -> go.Figure:
    rows = selected["daily_accuracy"]
    figure = go.Figure(
        go.Scatter(
            x=[row["date"] for row in rows],
            y=[row["accuracy"] for row in rows],
            mode="lines+markers",
            text=[row["markets"] for row in rows],
        )
    )
    return styled(figure, "Chronological test accuracy", "Accuracy")


def coefficient_figure(selected: dict[str, Any]) -> go.Figure:
    rows = selected["coefficient_ranking"][:20][::-1]
    figure = go.Figure(
        go.Bar(
            x=[row["coefficient"] for row in rows],
            y=[row["feature"] for row in rows],
            orientation="h",
        )
    )
    return styled(figure, "Largest standardized coefficients", "Coefficient")


def tuning_figure(metrics: dict[str, Any]) -> go.Figure:
    figure = go.Figure()
    for name, group in metrics["groups"].items():
        history = group["tuning_history"]
        figure.add_scatter(
            x=[row["c"] for row in history],
            y=[row["validation_log_loss"] for row in history],
            mode="lines+markers",
            name=name,
        )
    figure.update_xaxes(type="log", title="Regularization C")
    return styled(figure, "Training/tuning progress", "Inner validation log loss")


def styled(figure: go.Figure, title: str, y_title: str) -> go.Figure:
    figure.update_layout(
        title=title,
        paper_bgcolor="#151c31",
        plot_bgcolor="#151c31",
        font={"color": "#e8edf8"},
        margin={"l": 55, "r": 30, "t": 55, "b": 55},
        legend={"orientation": "h", "y": 1.1},
    )
    figure.update_xaxes(gridcolor="#2a3554", zerolinecolor="#2a3554")
    figure.update_yaxes(gridcolor="#2a3554", zerolinecolor="#2a3554", title=y_title)
    return figure
