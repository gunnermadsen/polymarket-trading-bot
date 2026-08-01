from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID


def json_safe(value: Any) -> Any:
    """Return a deterministic, strict-JSON-compatible representation."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (Path, UUID)):
        return str(value)
    if isinstance(value, Enum):
        return json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in normalized:
                raise ValueError(f"duplicate JSON key after normalization: {normalized_key}")
            normalized[normalized_key] = json_safe(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (set, frozenset)):
        normalized_items = [json_safe(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar = item_method()
        except (TypeError, ValueError):
            pass
        else:
            if scalar is not value:
                return json_safe(scalar)

    list_method = getattr(value, "tolist", None)
    if callable(list_method):
        return json_safe(list_method())

    dictionary_method = getattr(value, "to_dict", None)
    if callable(dictionary_method):
        return json_safe(dictionary_method())

    raise TypeError(f"unsupported artifact value: {type(value).__qualname__}")


def write_json_artifact(path: str | Path, payload: Any) -> Path:
    """Atomically write strict, sorted JSON with a stable trailing newline."""
    serialized = json.dumps(
        json_safe(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    return write_text_artifact(path, f"{serialized}\n")


def write_text_artifact(path: str | Path, content: str) -> Path:
    """Atomically write a UTF-8 text artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def render_development_report(report: Mapping[str, Any]) -> str:
    """Render a concise pre-holdout benchmark report from dictionary data."""
    sections = [
        "# Kraken Futures ML Development Benchmark",
        _render_verdict(report, holdout=False),
        _render_identity(report),
        _render_scope(report),
        _render_selection(report, title="Selected development candidate"),
        _render_compute(report),
        _render_candidate_table(
            "Model comparison",
            _first_value(report, ("model_comparison",), ("models",)),
        ),
        _render_candidate_table(
            "Feature-set comparison",
            _first_value(report, ("feature_comparison",), ("feature_ablation",)),
        ),
        _render_policy_calibration(report),
        _render_fold_table(report),
        _render_classification_section(report),
        _render_economic_section(report),
        _render_gates(report),
        _render_holdout_state(report),
        _render_notes(report),
    ]
    return _join_sections(sections)


def render_holdout_report(report: Mapping[str, Any]) -> str:
    """Render a concise locked-holdout benchmark report from dictionary data."""
    sections = [
        "# Kraken Futures ML Locked-Holdout Benchmark",
        _render_verdict(report, holdout=True),
        _render_identity(report),
        _render_scope(report),
        _render_selection(report, title="Frozen candidate"),
        _render_compute(report),
        _render_classification_section(report),
        _render_baselines(report),
        _render_economic_section(report),
        _render_importance(report),
        _render_gates(report),
        _render_notes(report),
    ]
    return _join_sections(sections)


def write_development_report(
    directory: str | Path,
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write development-benchmark.json and .md, returning both paths."""
    root = Path(directory)
    json_path = write_json_artifact(root / "development-benchmark.json", report)
    markdown_path = write_text_artifact(
        root / "development-benchmark.md",
        render_development_report(report),
    )
    return json_path, markdown_path


def write_holdout_report(
    directory: str | Path,
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write holdout-benchmark.json and .md, returning both paths."""
    root = Path(directory)
    json_path = write_json_artifact(root / "holdout-benchmark.json", report)
    markdown_path = write_text_artifact(
        root / "holdout-benchmark.md",
        render_holdout_report(report),
    )
    return json_path, markdown_path


def _join_sections(sections: Sequence[str]) -> str:
    return "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"


def _first_value(root: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = root
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            if current is not None:
                return current
    return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _markdown(value: Any) -> str:
    if value is None:
        return "—"
    return (
        str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    )


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _markdown(value)
    if not math.isfinite(number):
        return "—"
    if number == 0:
        return "0"
    if abs(number) >= 1000:
        return f"{number:,.{digits}f}"
    rendered = f"{number:.{digits}f}"
    return rendered.rstrip("0").rstrip(".") if digits > 0 else rendered


def _format_percent(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _markdown(value)
    if not math.isfinite(number):
        return "—"
    return f"{number * 100:.2f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return ""
    escaped_headers = [_markdown(header) for header in headers]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join("---" for _ in escaped_headers) + " |",
    ]
    lines.extend("| " + " | ".join(_markdown(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}" if body else ""


def _render_verdict(report: Mapping[str, Any], *, holdout: bool) -> str:
    verdict = _first_value(
        report,
        ("edge_verdict",),
        ("verdict",),
        ("result", "edge_verdict"),
        ("summary", "edge_verdict"),
    )
    if isinstance(verdict, Mapping):
        status = _first_value(
            verdict,
            ("status",),
            ("verdict",),
            ("qualified",),
            ("passed",),
        )
        reason = _first_value(verdict, ("reason",), ("detail",), ("message",))
        rendered = _status_text(status)
        if reason:
            rendered = f"{rendered} — {_markdown(reason)}"
    elif verdict is not None:
        rendered = _status_text(verdict)
    else:
        qualification = _first_value(
            report,
            ("candidate_qualified",),
            ("qualified",),
            ("gates_passed",),
        )
        if qualification is not None:
            rendered = _status_text(qualification)
        else:
            gate_results = _gate_outcomes(report)
            if gate_results and all(gate_results):
                suffix = (
                    "locked-holdout evidence meets every configured edge gate"
                    if holdout
                    else "development evidence meets every configured pre-holdout gate"
                )
                rendered = f"PASS — {suffix}."
            elif gate_results:
                suffix = (
                    "locked-holdout evidence does not meet every configured edge gate"
                    if holdout
                    else "development evidence does not meet every pre-holdout gate"
                )
                rendered = f"FAIL — {suffix}."
            else:
                rendered = "NOT ASSESSED — no explicit verdict or gate outcomes supplied."
    return f"## Edge verdict\n\n**{rendered}**"


def _status_text(value: Any) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return _markdown(value).upper()


def _render_identity(report: Mapping[str, Any]) -> str:
    rows: list[tuple[str, Any]] = []
    run_id = _first_value(report, ("run_id",), ("identity", "run_id"))
    if run_id is not None:
        rows.append(("Run", run_id))
    generated_at = _first_value(
        report,
        ("generated_at",),
        ("created_at",),
        ("evaluated_at",),
        ("completed_at",),
        ("identity", "generated_at"),
    )
    if generated_at is not None:
        rows.append(("Generated", json_safe(generated_at)))

    aliases = (
        (
            "Configuration SHA-256",
            (
                ("hashes", "config_sha256"),
                ("hashes", "config"),
                ("config_sha256",),
                ("config_hash",),
                ("config_fingerprint",),
                ("provenance", "config_fingerprint"),
            ),
        ),
        (
            "Dataset SHA-256",
            (
                ("hashes", "dataset_sha256"),
                ("hashes", "data_sha256"),
                ("hashes", "dataset"),
                ("dataset_sha256",),
                ("data_hash",),
                ("provenance", "raw_snapshot_sha256"),
            ),
        ),
        (
            "Feature SHA-256",
            (
                ("hashes", "feature_sha256"),
                ("hashes", "features"),
                ("feature_sha256",),
                ("feature_hash",),
                ("provenance", "feature_snapshot_sha256"),
            ),
        ),
        (
            "Model SHA-256",
            (
                ("hashes", "model_sha256"),
                ("hashes", "model"),
                ("model_sha256",),
                ("model_hash",),
                ("model", "model_sha256"),
            ),
        ),
        (
            "Policy SHA-256",
            (
                ("hashes", "policy_sha256"),
                ("hashes", "policy"),
                ("policy_sha256",),
            ),
        ),
    )
    for label, paths in aliases:
        value = _first_value(report, *paths)
        if value is not None:
            rows.append((label, f"`{_markdown(value)}`"))
    return _section("Reproducibility", _table(("Item", "Value"), rows))


def _render_scope(report: Mapping[str, Any]) -> str:
    scope = _as_mapping(_first_value(report, ("scope",), ("dataset",), ("data",), ("holdout",)))
    rows: list[tuple[str, Any]] = []
    keys = (
        ("Market", ("symbol", "market", "instrument")),
        ("Bar interval", ("interval", "bar_interval", "interval_seconds")),
        ("Forecast horizon", ("horizon", "horizon_bars")),
        ("Start", ("start", "start_at")),
        ("End", ("end", "end_at")),
        ("Rows", ("rows", "row_count", "samples")),
        ("Trades", ("trades", "trade_count")),
        ("Canonical source", ("source", "canonical_source")),
    )
    for label, aliases in keys:
        value = next((scope[key] for key in aliases if key in scope), None)
        if value is not None:
            rows.append((label, json_safe(value)))
    return _section("Data scope", _table(("Item", "Value"), rows))


def _render_selection(report: Mapping[str, Any], *, title: str) -> str:
    selected = _as_mapping(
        _first_value(
            report,
            ("selected",),
            ("selection",),
            ("candidate",),
            ("model",),
        )
    )
    rows: list[tuple[str, Any]] = []
    for label, aliases in (
        ("Model", ("model", "model_name", "name")),
        ("Feature set", ("feature_set", "features")),
        ("Probability threshold", ("probability_threshold", "threshold")),
        ("Directional margin", ("directional_margin", "margin")),
        ("No-trade policy", ("no_trade",)),
        ("Training rows", ("training_rows", "fit_rows")),
        ("Calibration rows", ("calibration_rows",)),
    ):
        value = next((selected[key] for key in aliases if key in selected), None)
        if value is not None:
            rows.append((label, json_safe(value)))
    return _section(title, _table(("Item", "Value"), rows))


def _render_compute(report: Mapping[str, Any]) -> str:
    cpu = _as_mapping(
        _first_value(
            report,
            ("cpu",),
            ("compute",),
            ("runtime", "cpu"),
            ("runtime",),
        )
    )
    rows: list[tuple[str, Any]] = []
    labels = (
        ("Host", ("host", "processor", "chip")),
        ("Detected cores", ("detected_cores", "cpu_count", "logical_cores")),
        ("Reserved cores", ("reserve_cores", "reserved_cores")),
        ("Parallel fits", ("parallel_fits", "available_cores", "workers", "n_jobs")),
        (
            "Comparison threads / estimator",
            ("comparison_estimator_threads", "estimator_threads", "inner_threads"),
        ),
        ("Polars threads / process", ("polars_threads",)),
        ("Final refit threads", ("final_refit_threads",)),
        ("Parallel backend", ("backend", "parallel_backend")),
        ("Random seed", ("random_seed", "seed")),
    )
    for label, aliases in labels:
        value = next((cpu[key] for key in aliases if key in cpu), None)
        if value is not None:
            rows.append((label, json_safe(value)))

    timing = _as_mapping(_first_value(report, ("timings",), ("runtime", "timings"), ("durations",)))
    for name in sorted(timing):
        value = timing[name]
        if isinstance(value, (int, float)):
            rendered: Any = f"{_format_number(value, 2)} s"
        elif isinstance(value, Mapping):
            seconds = _first_value(value, ("seconds",), ("elapsed_seconds",))
            rendered = (
                f"{_format_number(seconds, 2)} s"
                if seconds is not None
                else json.dumps(json_safe(value), sort_keys=True)
            )
        else:
            rendered = json_safe(value)
        rows.append((f"Timing: {name.replace('_', ' ')}", rendered))
    return _section("CPU and runtime", _table(("Parameter", "Value"), rows))


def _render_candidate_table(title: str, candidates: Any) -> str:
    rows: list[tuple[Any, ...]] = []
    if isinstance(candidates, Mapping):
        iterable = [
            {"name": name, **dict(value)}
            if isinstance(value, Mapping)
            else {"name": name, "value": value}
            for name, value in sorted(candidates.items())
        ]
    elif isinstance(candidates, Sequence) and not isinstance(candidates, str):
        iterable = list(candidates)
    else:
        iterable = []
    for raw in iterable:
        candidate = _as_mapping(raw)
        aggregate = _as_mapping(_first_value(candidate, ("aggregate",), ("metrics",), ("mean",)))
        if not aggregate:
            aggregate = candidate
        classification = _as_mapping(
            _first_value(
                aggregate,
                ("classification",),
                ("classification_metrics",),
            )
        )
        if not classification:
            classification = aggregate
        economic = _base_economic(_first_value(aggregate, ("economic",), ("economics",)))
        if not economic:
            economic = _base_economic(_first_value(candidate, ("economic",), ("economics",)))
        if not economic:
            economic = aggregate
        rows.append(
            (
                _first_value(
                    candidate,
                    ("model",),
                    ("model_name",),
                    ("name",),
                ),
                _first_value(candidate, ("feature_set",), ("features",)),
                _format_percent(
                    _metric(classification, "balanced_accuracy", "mean_balanced_accuracy")
                ),
                _format_percent(_metric(classification, "macro_f1", "mean_macro_f1")),
                _format_number(_metric(classification, "log_loss", "mean_log_loss")),
                _format_number(_metric(economic, "net_expectancy_bps", "mean_net_expectancy_bps")),
                _first_value(
                    aggregate,
                    ("positive_folds",),
                    ("development_positive_folds",),
                    ("positive_economic_folds",),
                ),
                (
                    "yes"
                    if _first_value(candidate, ("selected",), ("qualified",)) is True
                    else "no"
                    if _first_value(candidate, ("selected",), ("qualified",)) is False
                    else "—"
                ),
            )
        )
    table = _table(
        (
            "Model",
            "Features",
            "Balanced acc.",
            "Macro F1",
            "Log loss",
            "Net bps/trade",
            "Positive folds",
            "Selected",
        ),
        rows,
    )
    return _section(title, table)


def _render_fold_table(report: Mapping[str, Any]) -> str:
    folds = _first_value(
        report,
        ("folds",),
        ("fold_results",),
        ("selected_fold_results",),
        ("development_folds",),
        ("results", "folds"),
    )
    if not isinstance(folds, Sequence) or isinstance(folds, str):
        return ""
    rows: list[tuple[Any, ...]] = []
    ordered_folds = sorted(
        folds,
        key=lambda raw: str(
            _first_value(
                _as_mapping(raw),
                ("ranges", "evaluation_start"),
                ("evaluation_start",),
                ("fold",),
            )
        ),
    )
    for raw in ordered_folds:
        fold = _as_mapping(raw)
        classification = _as_mapping(
            _first_value(
                fold,
                ("classification",),
                ("classification_metrics",),
                ("metrics", "classification"),
            )
        )
        economic = _base_economic(
            _first_value(
                fold,
                ("economic",),
                ("economics",),
                ("economic_metrics",),
                ("metrics", "economic"),
            )
        )
        policy = _as_mapping(_first_value(fold, ("policy",), ("selected_policy",)))
        baseline = _as_mapping(_first_value(fold, ("baseline",), ("baselines",)))
        uplift = _first_value(
            fold,
            ("balanced_accuracy_uplift",),
            ("uplift", "balanced_accuracy"),
        )
        if uplift is None:
            model_accuracy = _metric(classification, "balanced_accuracy")
            baseline_accuracy = _metric(baseline, "best_balanced_accuracy")
            if model_accuracy is not None and baseline_accuracy is not None:
                uplift = float(model_accuracy) - float(baseline_accuracy)
        rows.append(
            (
                _first_value(fold, ("fold",), ("name",), ("fold_name",)),
                _first_value(fold, ("model",), ("model_name",)),
                _first_value(fold, ("feature_set",), ("features",)),
                _format_percent(_metric(classification, "balanced_accuracy")),
                _format_percent(uplift),
                _format_percent(_metric(classification, "macro_f1")),
                _format_number(_metric(classification, "log_loss")),
                _format_number(_metric(economic, "trades"), 0),
                _format_number(_metric(economic, "net_expectancy_bps")),
                _format_number(_metric(economic, "bootstrap_95_lower_bps")),
                _format_number(_metric(economic, "profit_factor")),
                _policy_text(policy),
            )
        )
    return _section(
        "Walk-forward folds",
        _table(
            (
                "Fold",
                "Model",
                "Features",
                "Balanced acc.",
                "Uplift",
                "Macro F1",
                "Log loss",
                "Trades",
                "Net bps/trade",
                "95% CI lower",
                "Profit factor",
                "Policy",
            ),
            rows,
        ),
    )


def _render_policy_calibration(report: Mapping[str, Any]) -> str:
    folds = _first_value(
        report,
        ("selected_fold_results",),
        ("fold_results",),
        ("development_folds",),
    )
    if not isinstance(folds, Sequence) or isinstance(folds, str):
        return ""
    minimum_trades = _first_value(
        report,
        ("selection_parameters", "minimum_calibration_trades"),
    )
    try:
        minimum_trades = int(minimum_trades)
    except (TypeError, ValueError):
        minimum_trades = 50
    rows: list[tuple[Any, ...]] = []
    ordered = sorted(
        folds,
        key=lambda raw: str(
            _first_value(
                _as_mapping(raw),
                ("ranges", "evaluation_start"),
                ("evaluation_start",),
                ("fold",),
            )
        ),
    )
    for raw in ordered:
        fold = _as_mapping(raw)
        grid = fold.get("policy_grid")
        if not isinstance(grid, Sequence) or isinstance(grid, str):
            continue
        eligible = [
            _as_mapping(candidate)
            for candidate in grid
            if isinstance(candidate, Mapping)
            and (candidate.get("trades") or 0) >= minimum_trades
            and candidate.get("net_expectancy_bps") is not None
        ]
        if not eligible:
            continue
        best = max(eligible, key=lambda candidate: candidate["net_expectancy_bps"])
        rows.append(
            (
                _first_value(fold, ("fold",), ("name",)),
                _format_number(best.get("trades"), 0),
                _format_number(best.get("net_expectancy_bps")),
                _format_number(best.get("bootstrap_80_lower_bps")),
                _format_number(best.get("profit_factor")),
                _format_number(best.get("probability_threshold"), 2),
                _format_number(best.get("directional_margin"), 2),
            )
        )
    body = _table(
        (
            "Fold",
            "Trades",
            "Best net bps/trade",
            "80% CI lower",
            "Profit factor",
            "Probability",
            "Margin",
        ),
        rows,
    )
    if body:
        body += (
            "\n\nA policy qualifies only when its 80% circular-block "
            f"bootstrap lower bound is positive with at least {minimum_trades} trades."
        )
    return _section("Threshold-window policy diagnostics", body)


def _render_classification_section(report: Mapping[str, Any]) -> str:
    metrics = _as_mapping(
        _first_value(
            report,
            ("classification",),
            ("classification_metrics",),
            ("aggregate", "classification"),
            ("metrics", "classification"),
            ("holdout", "classification"),
        )
    )
    rows: list[tuple[str, Any]] = []
    for label, key, percent in (
        ("Accuracy", "accuracy", True),
        ("Balanced accuracy", "balanced_accuracy", True),
        ("Macro F1", "macro_f1", True),
        ("Log loss", "log_loss", False),
        ("Multiclass Brier", "brier", False),
        ("Macro ECE", "macro_ece", False),
    ):
        if key in metrics:
            formatter = _format_percent if percent else _format_number
            rows.append((label, formatter(metrics[key])))
    matrix = metrics.get("confusion_matrix")
    body = _table(("Metric", "Value"), rows)
    if matrix is not None:
        encoded_matrix = json.dumps(json_safe(matrix))
        body = f"{body}\n\nConfusion matrix (`short`, `flat`, `long`): `{encoded_matrix}`"
    return _section("Classification", body)


def _render_baselines(report: Mapping[str, Any]) -> str:
    baselines = _as_mapping(
        _first_value(report, ("baselines",), ("baseline",), ("metrics", "baselines"))
    )
    rows: list[tuple[Any, ...]] = []
    for name in sorted(baselines):
        metrics = baselines[name]
        if not isinstance(metrics, Mapping):
            continue
        rows.append(
            (
                name,
                _format_percent(_metric(metrics, "accuracy")),
                _format_percent(_metric(metrics, "balanced_accuracy")),
                _format_percent(_metric(metrics, "macro_f1")),
                _format_number(_metric(metrics, "log_loss")),
                _format_number(_metric(metrics, "brier")),
            )
        )
    return _section(
        "Naive baselines",
        _table(
            ("Baseline", "Accuracy", "Balanced acc.", "Macro F1", "Log loss", "Brier"),
            rows,
        ),
    )


def _render_economic_section(report: Mapping[str, Any]) -> str:
    economics = _first_value(
        report,
        ("economic",),
        ("economics",),
        ("economic_metrics",),
        ("aggregate", "economic"),
        ("metrics", "economic"),
        ("holdout", "economic"),
    )
    if not isinstance(economics, Mapping):
        return ""
    if any(key in economics for key in ("trades", "net_expectancy_bps")):
        scenarios: list[tuple[str, Mapping[str, Any]]] = [("1.0×", economics)]
    else:
        scenarios = [
            (str(name), metrics)
            for name, metrics in sorted(economics.items())
            if isinstance(metrics, Mapping)
        ]
    rows = [
        (
            scenario,
            _format_number(_metric(metrics, "trades"), 0),
            _format_percent(_metric(metrics, "action_coverage")),
            _format_number(_metric(metrics, "net_expectancy_bps")),
            _format_number(_metric(metrics, "bootstrap_95_lower_bps")),
            _format_number(_metric(metrics, "bootstrap_95_upper_bps")),
            _format_number(_metric(metrics, "profit_factor")),
            _format_percent(_metric(metrics, "win_rate")),
            _format_number(_metric(metrics, "total_net_bps")),
            _format_number(_metric(metrics, "max_drawdown_bps")),
            _format_percent(_metric(metrics, "positive_month_fraction")),
        )
        for scenario, metrics in scenarios
    ]
    return _section(
        "Net expectancy after costs",
        _table(
            (
                "Execution cost",
                "Trades",
                "Coverage",
                "Net bps/trade",
                "95% CI lower",
                "95% CI upper",
                "Profit factor",
                "Win rate",
                "Total net bps",
                "Max drawdown bps",
                "Positive months",
            ),
            rows,
        ),
    )


def _base_economic(value: Any) -> Mapping[str, Any]:
    economics = _as_mapping(value)
    if any(key in economics for key in ("trades", "net_expectancy_bps")):
        return economics
    for key in ("1.0x", "1.0×", "1.0", "base", "nominal"):
        if isinstance(economics.get(key), Mapping):
            return economics[key]
    return {}


def _metric(metrics: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def _policy_text(policy: Mapping[str, Any]) -> str:
    if not policy:
        return "—"
    if policy.get("no_trade"):
        return "no trade"
    threshold = _first_value(policy, ("probability_threshold",), ("threshold",))
    margin = _first_value(policy, ("directional_margin",), ("margin",))
    return f"p≥{_format_number(threshold, 2)}, Δ≥{_format_number(margin, 2)}"


def _gate_entries(report: Mapping[str, Any]) -> list[tuple[str, Any]]:
    gates = _first_value(
        report,
        ("gates",),
        ("gate_results",),
        ("validation", "gates"),
        ("result", "gates"),
    )
    if isinstance(gates, Mapping):
        checks = gates.get("checks")
        if isinstance(checks, Mapping):
            return [(str(name), checks[name]) for name in sorted(checks)]
        return [(str(name), gates[name]) for name in sorted(gates)]
    if isinstance(gates, Sequence) and not isinstance(gates, str):
        entries = []
        for index, gate in enumerate(gates):
            gate_mapping = _as_mapping(gate)
            name = _first_value(gate_mapping, ("name",), ("gate",), ("id",))
            entries.append((str(name or f"gate_{index + 1}"), gate))
        return entries
    return []


def _gate_passed(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        status = _first_value(
            value,
            ("passed",),
            ("pass",),
            ("qualified",),
            ("status",),
        )
        if isinstance(status, bool):
            return status
        if isinstance(status, str):
            normalized = status.casefold()
            if normalized in {"pass", "passed", "qualified", "true"}:
                return True
            if normalized in {"fail", "failed", "disqualified", "false"}:
                return False
    return None


def _gate_outcomes(report: Mapping[str, Any]) -> list[bool]:
    return [
        outcome
        for _, value in _gate_entries(report)
        if (outcome := _gate_passed(value)) is not None
    ]


def _render_gates(report: Mapping[str, Any]) -> str:
    rows: list[tuple[Any, ...]] = []
    for name, raw in _gate_entries(report):
        value = _as_mapping(raw)
        outcome = _gate_passed(raw)
        observed = _first_value(value, ("observed",), ("value",), ("actual",))
        requirement = _first_value(
            value,
            ("requirement",),
            ("threshold",),
            ("minimum",),
            ("expected",),
            ("required",),
        )
        detail = _first_value(value, ("reason",), ("detail",), ("message",))
        if not value and not isinstance(raw, bool):
            observed = raw
        rows.append(
            (
                name.replace("_", " "),
                "PASS" if outcome is True else "FAIL" if outcome is False else "—",
                _format_number(observed) if isinstance(observed, (int, float)) else observed,
                _format_number(requirement)
                if isinstance(requirement, (int, float))
                else requirement,
                detail,
            )
        )
    return _section(
        "Decision gates",
        _table(("Gate", "Result", "Observed", "Requirement", "Detail"), rows),
    )


def _render_holdout_state(report: Mapping[str, Any]) -> str:
    state = _as_mapping(
        _first_value(report, ("holdout_state",), ("holdout",), ("freeze", "holdout"))
    )
    rows: list[tuple[str, Any]] = []
    for label, aliases in (
        ("Status", ("status", "state")),
        ("Holdout opened", ("opened", "evaluated", "consumed")),
        ("Start", ("start", "start_at")),
        ("End", ("end", "end_at")),
        ("Reason", ("reason", "detail")),
    ):
        value = next((state[key] for key in aliases if key in state), None)
        if value is not None:
            rows.append((label, json_safe(value)))
    return _section("Locked holdout", _table(("Item", "Value"), rows))


def _render_importance(report: Mapping[str, Any]) -> str:
    importance = _first_value(
        report,
        ("permutation_importance",),
        ("feature_importance",),
        ("diagnostics", "permutation_importance"),
    )
    ranked_rows: list[tuple[float | None, tuple[Any, ...]]] = []
    if isinstance(importance, Mapping):
        iterable = [
            {"feature": feature, "importance": value}
            if not isinstance(value, Mapping)
            else {"feature": feature, **dict(value)}
            for feature, value in importance.items()
        ]
    elif isinstance(importance, Sequence) and not isinstance(importance, str):
        iterable = importance
    else:
        iterable = []
    for raw in iterable:
        row = _as_mapping(raw)
        raw_mean = _first_value(
            row,
            ("importance",),
            ("mean",),
            ("importance_mean",),
            ("mean_log_loss_degradation",),
        )
        try:
            rank_value = float(raw_mean) if raw_mean is not None else None
        except (TypeError, ValueError):
            rank_value = None
        ranked_rows.append(
            (
                rank_value,
                (
                    _first_value(row, ("feature",), ("name",)),
                    _format_number(raw_mean),
                    _format_number(
                        _first_value(
                            row,
                            ("std",),
                            ("importance_std",),
                            ("standard_deviation",),
                        )
                    ),
                ),
            )
        )
    ranked_rows.sort(
        key=lambda item: (
            -item[0] if item[0] is not None else float("inf"),
            str(item[1][0]),
        )
    )
    rows = [row for _, row in ranked_rows]
    return _section(
        "Permutation importance",
        _table(("Feature", "Mean importance", "Std. dev."), rows[:20]),
    )


def _render_notes(report: Mapping[str, Any]) -> str:
    entries: list[str] = []
    for key, prefix in (("warnings", "Warning"), ("notes", "Note")):
        value = report.get(key)
        if value is None:
            continue
        items = value if isinstance(value, Sequence) and not isinstance(value, str) else [value]
        entries.extend(f"- **{prefix}:** {_markdown(item)}" for item in items)
    return _section("Notes", "\n".join(entries))
