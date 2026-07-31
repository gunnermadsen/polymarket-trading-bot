from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from btc_directional_model import cli
from btc_directional_model.fixed_time_selective_benchmark import (
    FIXED_TIME_SELECTIVE_PAPER_AUTHORIZATION,
    _assert_identical_scored_universes,
    _control_predictions_on_selected_rows,
    _fold_summary,
    _same_row_control_comparison,
    _select_candidate,
)
from btc_directional_model.fixed_time_selective_config import (
    FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE,
)


def scored_rows() -> pl.DataFrame:
    start = datetime(2026, 7, 21, tzinfo=UTC)
    return pl.DataFrame(
        {
            "fold_index": [6, 6, 6],
            "market_id": ["market-a", "market-b", "market-c"],
            "observed_at": [
                start + timedelta(minutes=5 * index, seconds=120)
                for index in range(3)
            ],
            "seconds_elapsed": [120, 120, 120],
            "label_up": [1, 0, 0],
            "predicted_up": [1, 0, 1],
            "correct": [True, True, False],
        }
    )


def candidate_payload(
    name: str,
    *,
    minimum_accuracy: float,
    aggregate_accuracy: float,
    qualified: bool = True,
) -> dict[str, object]:
    return {
        "candidate": name,
        "advancement_qualified": qualified,
        "feature_count": 71,
        "fold_summary": {
            "minimum_accuracy": minimum_accuracy,
            "minimum_up_recall": 0.91,
            "minimum_down_recall": 0.90,
        },
        "primary": {
            "metrics": {
                "accuracy": aggregate_accuracy,
                "balanced_accuracy": aggregate_accuracy - 0.01,
            },
            "predicted_side_hard_confident_errors": {
                "all": {"hard_confident_error_exposure_rate": 0.002}
            },
        },
    }


def test_scored_universe_requires_identical_unique_fixed_120_rows() -> None:
    control = scored_rows()
    _assert_identical_scored_universes(
        {
            FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE: control,
            "challenger": control.clone(),
        }
    )

    changed = control.with_columns(
        pl.when(pl.col("market_id") == "market-c")
        .then(pl.lit("market-d"))
        .otherwise(pl.col("market_id"))
        .alias("market_id")
    )
    with pytest.raises(RuntimeError, match="scored universe differs"):
        _assert_identical_scored_universes(
            {
                FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE: control,
                "challenger": changed,
            }
        )

    with pytest.raises(RuntimeError, match="duplicate identities"):
        _assert_identical_scored_universes(
            {
                FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE: pl.concat(
                    [control, control[:1]]
                )
            }
        )


def test_same_row_control_comparison_uses_candidate_selected_rows() -> None:
    candidate = scored_rows()
    control = candidate.with_columns(
        pl.Series("predicted_up", [1, 1, 0]),
        pl.Series("correct", [True, False, False]),
    )

    comparison = _same_row_control_comparison(candidate, control)

    assert comparison == {
        "markets": 3,
        "candidate_accuracy": pytest.approx(2 / 3),
        "control_accuracy": pytest.approx(1 / 3),
        "accuracy_delta": pytest.approx(1 / 3),
        "candidate_only_correct": 1,
        "control_only_correct": 0,
        "direction_disagreements": 2,
    }


def test_hard_tail_control_predictions_use_candidate_selected_keys() -> None:
    candidate = scored_rows().filter(pl.col("market_id") != "market-b")
    control = scored_rows().with_columns(
        pl.Series("probability_up", [0.99, 0.10, 0.97]),
        pl.Series("confidence", [0.99, 0.90, 0.97]),
    )

    same_rows = _control_predictions_on_selected_rows(
        candidate,
        control,
        keys=("fold_index", "market_id", "observed_at"),
    )

    assert same_rows["market_id"].to_list() == ["market-a", "market-c"]
    assert same_rows["confidence"].to_list() == [0.99, 0.97]
    assert same_rows["correct"].to_list() == [True, False]


def test_fold_summary_and_selection_prioritize_adverse_fold_accuracy() -> None:
    folds = [
        {
            "primary": {
                "metrics": {
                    "accuracy": 0.94,
                    "balanced_accuracy": 0.93,
                    "up_recall": 0.92,
                    "down_recall": 0.91,
                    "expected_calibration_error": 0.03,
                    "coverage": 0.15,
                }
            }
        },
        {
            "primary": {
                "metrics": {
                    "accuracy": 0.91,
                    "balanced_accuracy": 0.90,
                    "up_recall": 0.90,
                    "down_recall": 0.89,
                    "expected_calibration_error": 0.05,
                    "coverage": 0.14,
                }
            }
        },
    ]
    assert _fold_summary(folds) == {
        "fold_count": 2,
        "minimum_accuracy": 0.91,
        "maximum_accuracy": 0.94,
        "minimum_balanced_accuracy": 0.90,
        "minimum_up_recall": 0.90,
        "minimum_down_recall": 0.89,
        "maximum_expected_calibration_error": 0.05,
        "minimum_coverage": 0.14,
        "maximum_coverage": 0.15,
    }

    config = SimpleNamespace(
        split=SimpleNamespace(
            policy_selection_end=datetime(2026, 7, 29, tzinfo=UTC)
        )
    )
    selection = _select_candidate(
        config,
        {
            FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE: candidate_payload(
                FIXED_TIME_SELECTIVE_CONTROL_CANDIDATE,
                minimum_accuracy=0.99,
                aggregate_accuracy=0.99,
                qualified=False,
            ),
            "higher-average": candidate_payload(
                "higher-average",
                minimum_accuracy=0.91,
                aggregate_accuracy=0.95,
            ),
            "stronger-adverse-fold": candidate_payload(
                "stronger-adverse-fold",
                minimum_accuracy=0.92,
                aggregate_accuracy=0.93,
            ),
        },
    )
    assert selection["selected_candidate"] == "stronger-adverse-fold"
    assert selection["qualified_candidates"] == [
        "stronger-adverse-fold",
        "higher-average",
    ]
    assert selection["independently_qualified"] is False
    assert selection["fresh_forward_evidence_start"] == (
        "2026-07-29T00:00:00+00:00"
    )


def test_selective_benchmark_cli_dispatches_with_development_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    config = object()
    monkeypatch.setattr(cli, "load_fixed_time_selective_config", lambda _: config)
    monkeypatch.setattr(
        cli,
        "run_fixed_time_selective_benchmark",
        lambda loaded: (
            run_dir,
            {"selection": {"selected_candidate": "selective-candidate"}},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "btc-directional-model",
            "fixed-120-selective-benchmark-run",
            "--config",
            str(tmp_path / "benchmark.toml"),
        ],
    )

    cli.main()

    output = capsys.readouterr().out
    assert f"report: {run_dir / 'report.html'}" in output
    assert "development candidate: selective-candidate" in output
    assert "July 29 onward excluded" in output
    assert "live capital: not authorized" in output


def test_selective_export_cli_supplies_explicit_paper_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = object()
    captured: dict[str, object] = {}

    def export(**kwargs: object) -> tuple[Path, Path, dict[str, bool]]:
        captured.update(kwargs)
        return tmp_path / "freeze", tmp_path / "runtime", {
            "production_qualified": False
        }

    monkeypatch.setattr(cli, "load_fixed_time_selective_config", lambda _: config)
    monkeypatch.setattr(
        cli,
        "freeze_and_export_fixed_time_selective_paper_candidate",
        export,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "btc-directional-model",
            "fixed-120-selective-paper-candidate-export",
            "--config",
            str(tmp_path / "benchmark.toml"),
            "--benchmark-run",
            str(tmp_path / "run"),
            "--model-key",
            "btc-fixed-120-selective-paper",
            "--authorize-paper-only",
        ],
    )

    cli.main()

    assert captured == {
        "config": config,
        "benchmark_run": tmp_path / "run",
        "model_key": "btc-fixed-120-selective-paper",
        "authorization": FIXED_TIME_SELECTIVE_PAPER_AUTHORIZATION,
    }
    output = capsys.readouterr().out
    assert "scope: paper_only; production-qualified: false" in output
    assert "fresh forward evidence: required from July 29, 2026" in output
