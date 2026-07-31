from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import btc_directional_model.fixed_time_reversal_benchmark as benchmark_module
from btc_directional_model import cli
from btc_directional_model.fixed_time_reversal_benchmark import (
    FIXED_TIME_REVERSAL_PAPER_AUTHORIZATION,
    _assert_identical_scored_universes,
    _candidate_rank,
    _fold_summary,
    _same_row_control_comparison,
    _score_candidate_rows,
    _select_candidate,
    _validate_eligible_cohort,
    hard_false_up_metrics,
    sign_override_metrics,
)
from btc_directional_model.fixed_time_reversal_config import (
    FIXED_TIME_REVERSAL_CONTROL_CANDIDATE,
)


def scored_rows() -> pl.DataFrame:
    start = datetime(2026, 7, 21, tzinfo=UTC)
    label = [0, 0, 1, 1, 0, 1]
    sign = [1, 0, 0, 1, 1, 0]
    predicted = [1, 1, 1, 1, 0, 0]
    probability = [0.98, 0.97, 0.97, 0.99, 0.20, 0.03]
    return pl.DataFrame(
        {
            "fold_index": [6] * 6,
            "market_id": [f"market-{index}" for index in range(6)],
            "window_start": [start + timedelta(minutes=5 * index) for index in range(6)],
            "observed_at": [
                start + timedelta(minutes=5 * index, seconds=120) for index in range(6)
            ],
            "seconds_elapsed": [120] * 6,
            "label_up": label,
            "binance_sign_up": sign,
            "probability_up": probability,
            "predicted_up": predicted,
            "confidence": [max(value, 1.0 - value) for value in probability],
            "correct": [prediction == actual for prediction, actual in zip(predicted, label)],
            "baseline_correct": [baseline == actual for baseline, actual in zip(sign, label)],
        }
    )


def test_sign_override_metrics_prove_uplift_identity_and_false_up_source() -> None:
    metrics = sign_override_metrics(scored_rows(), eligible_markets=10)

    assert metrics["actual_reversals"] == 4
    assert metrics["overrides"] == 3
    assert metrics["correct_overrides"] == 2
    assert metrics["incorrect_overrides"] == 1
    assert metrics["override_precision"] == pytest.approx(2 / 3)
    assert metrics["reversal_recall"] == pytest.approx(1 / 2)
    assert metrics["missed_reversals"] == 2
    assert metrics["sign_baseline_accuracy_uplift"] == pytest.approx(1 / 6)
    assert metrics["selected_conditional_accuracy_uplift"] == pytest.approx(1 / 6)
    assert metrics["eligible_exposure_accuracy_uplift"] == pytest.approx(1 / 10)
    assert metrics["uplift_identity"] == {
        "correct_overrides_minus_incorrect_overrides": 1,
        "model_correct_minus_sign_baseline_correct": 1,
        "verified": True,
    }
    assert metrics["selected_conditional_uplift_identity"] == {
        "denominator_markets": 6,
        "override_net_correct": 1,
        "override_net_accuracy_uplift": pytest.approx(1 / 6),
        "model_minus_sign_baseline_accuracy": pytest.approx(1 / 6),
        "verified": True,
    }
    assert metrics["eligible_exposure_uplift_identity"] == {
        "denominator_markets": 10,
        "override_net_correct": 1,
        "override_net_accuracy_uplift": pytest.approx(1 / 10),
        "model_minus_sign_baseline_correct_exposure": pytest.approx(1 / 10),
        "selected_coverage_times_conditional_uplift": pytest.approx(1 / 10),
        "verified": True,
    }
    assert metrics["false_up_markets"] == 2
    assert metrics["false_up_followed_sign"] == 1
    assert metrics["false_up_bad_override"] == 1


def test_hard_false_up_metrics_and_same_row_control_are_candidate_scoped() -> None:
    candidate = scored_rows()
    hard = hard_false_up_metrics(
        candidate,
        eligible_markets=10,
        confidence_floor=0.95,
    )
    assert hard["hard_error_markets"] == 3
    assert hard["hard_false_up_markets"] == 2
    assert hard["hard_false_up_followed_sign"] == 1
    assert hard["hard_false_up_bad_override"] == 1

    control = candidate.with_columns(
        pl.col("binance_sign_up").alias("predicted_up"),
        pl.when(pl.col("binance_sign_up") == 1)
        .then(pl.lit(0.99))
        .otherwise(pl.lit(0.01))
        .alias("probability_up"),
        pl.lit(0.99).alias("confidence"),
        pl.col("baseline_correct").alias("correct"),
    )
    comparison = _same_row_control_comparison(
        candidate.filter(pl.col("market_id") != "market-3"),
        control,
        hard_confidence_floor=0.95,
    )

    assert comparison["markets"] == 5
    assert comparison["candidate_hard"]["hard_error_markets"] == 3
    assert comparison["control_hard"]["hard_error_markets"] == 4
    assert comparison["candidate_hard"]["hard_false_up_markets"] == 2
    assert comparison["control_hard"]["hard_false_up_markets"] == 2


def test_scored_universe_requires_identical_unique_rows_and_official_labels() -> None:
    control = scored_rows()
    _assert_identical_scored_universes(
        {
            FIXED_TIME_REVERSAL_CONTROL_CANDIDATE: control,
            "reversal": control.clone(),
        }
    )

    changed = control.with_columns(
        pl.when(pl.col("market_id") == "market-5")
        .then(pl.lit(0))
        .otherwise(pl.col("label_up"))
        .alias("label_up")
    )
    with pytest.raises(RuntimeError, match="universe differs"):
        _assert_identical_scored_universes(
            {
                FIXED_TIME_REVERSAL_CONTROL_CANDIDATE: control,
                "reversal": changed,
            }
        )

    with pytest.raises(RuntimeError, match="contains duplicates"):
        _assert_identical_scored_universes(
            {FIXED_TIME_REVERSAL_CONTROL_CANDIDATE: pl.concat([control, control[:1]])}
        )

    non_exact = control.with_columns(pl.lit(125).alias("seconds_elapsed"))
    with pytest.raises(RuntimeError, match="not an exact-120 universe"):
        _assert_identical_scored_universes(
            {
                FIXED_TIME_REVERSAL_CONTROL_CANDIDATE: non_exact,
                "reversal": non_exact.clone(),
            }
        )


def test_probability_reversal_tie_uses_executable_override_polarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = scored_rows()[:1].select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "binance_sign_up",
    )
    monkeypatch.setattr(
        benchmark_module,
        "reversal_probability_semantics",
        lambda *_: {
            "p_target": np.array([0.5]),
            "p_persistence": np.array([0.5]),
            "p_reversal": np.array([0.5]),
            "probability_up": np.array([0.4]),
        },
    )
    model = SimpleNamespace(raw_logit=lambda _: np.array([0.0]))
    calibrator = SimpleNamespace(probability=lambda _: np.array([0.5]))
    candidate = SimpleNamespace(name="reversal", target_kind="path_persistence")

    scored = _score_candidate_rows(
        row,
        candidate=candidate,
        model=model,
        calibrator=calibrator,
        fold_index=0,
    )

    assert scored["p_reversal"].to_list() == [0.5]
    assert scored["path_overridden"].to_list() == [True]
    assert scored["predicted_reversal"].to_list() == [True]


def test_eligible_cohort_keeps_exact_120_without_future_path_conditioning() -> None:
    start = datetime(2026, 3, 21, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "market_id": ["market-a", "market-a", "market-b", "market-c"],
            "window_start": [
                start,
                start,
                start + timedelta(minutes=5),
                start + timedelta(minutes=10),
            ],
            "observed_at": [
                start + timedelta(seconds=120),
                start + timedelta(seconds=125),
                start + timedelta(minutes=5, seconds=120),
                start + timedelta(minutes=10, seconds=125),
            ],
            "seconds_elapsed": [120, 125, 120, 125],
            "label_up": [1, 1, 0, 1],
            "binance_sign_up": [1, 1, 0, 1],
            "btc_path_from_window_open_bps": [1.0, 2.0, -1.0, 1.0],
        }
    )
    config = SimpleNamespace(
        model=SimpleNamespace(
            expected_eligible_markets=3,
            expected_eligible_estimator_rows=4,
            expected_exact_120_eligible_markets=2,
            estimator_training_seconds=(120, 125),
            decision_second=120,
            path_zero_epsilon_bps=1e-12,
        ),
        split=SimpleNamespace(
            development_start=start,
            policy_selection_end=datetime(2026, 7, 29, tzinfo=UTC),
        ),
    )

    _validate_eligible_cohort(frame, config)


def test_fold_summary_reports_all_raw_uplifts_without_gating_them() -> None:
    folds = []
    for index, uplift in enumerate((0.01, -0.02, 0.03)):
        folds.append(
            {
                "fold_index": index,
                "raw": {"sign_override_metrics": {"sign_baseline_accuracy_uplift": uplift}},
                "primary": {
                    "metrics": {
                        "accuracy": 0.93 - index * 0.01,
                        "balanced_accuracy": 0.92 - index * 0.01,
                        "up_recall": 0.91,
                        "down_recall": 0.90,
                        "expected_calibration_error": 0.04 + index * 0.01,
                        "coverage": 0.10,
                    }
                },
            }
        )

    summary = _fold_summary(folds)

    assert summary["minimum_raw_sign_baseline_accuracy_uplift"] == -0.02
    assert summary["nonnegative_raw_uplift_folds"] == 2
    assert summary["all_fold_raw_uplift_is_ranking_evidence_only"] is True
    assert [item["accuracy_uplift"] for item in summary["raw_sign_baseline_uplift_by_fold"]] == [
        0.01,
        -0.02,
        0.03,
    ]


def selection_payload(
    name: str,
    *,
    qualified: bool,
    minimum_raw_uplift: float,
) -> dict[str, object]:
    return {
        "candidate": name,
        "advancement_qualified": qualified,
        "feature_count": 71,
        "fold_summary": {
            "minimum_raw_sign_baseline_accuracy_uplift": minimum_raw_uplift,
            "minimum_primary_accuracy": 0.91,
        },
        "raw": {"sign_override_metrics": {"sign_baseline_accuracy_uplift": 0.01}},
        "primary": {
            "metrics": {"accuracy": 0.93},
            "sign_override_metrics": {"override_precision": 0.60},
            "hard_false_up_metrics": {"hard_false_up_rate_selected": 0.01},
        },
    }


def test_selection_never_advances_control_and_ranks_every_fold_raw_uplift() -> None:
    config = SimpleNamespace(
        split=SimpleNamespace(policy_selection_end=datetime(2026, 7, 29, tzinfo=UTC))
    )
    control = selection_payload(
        FIXED_TIME_REVERSAL_CONTROL_CANDIDATE,
        qualified=False,
        minimum_raw_uplift=0.50,
    )
    weaker = selection_payload(
        "reversal-weaker",
        qualified=True,
        minimum_raw_uplift=-0.01,
    )
    stronger = selection_payload(
        "reversal-stronger",
        qualified=True,
        minimum_raw_uplift=0.0,
    )

    assert _candidate_rank(stronger) > _candidate_rank(weaker)
    selection = _select_candidate(
        config,
        {
            FIXED_TIME_REVERSAL_CONTROL_CANDIDATE: control,
            "reversal-weaker": weaker,
            "reversal-stronger": stronger,
        },
    )
    assert selection["selected_candidate"] == "reversal-stronger"
    assert selection["qualified_candidates"] == [
        "reversal-stronger",
        "reversal-weaker",
    ]
    assert selection["trading_process_creation_authorized"] is False
    assert selection["live_capital_authorized"] is False
    assert selection["fresh_forward_evidence_start"] == ("2026-07-29T00:00:00+00:00")


def test_reversal_benchmark_cli_dispatches_with_fixed_time_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    config = object()
    monkeypatch.setattr(cli, "load_fixed_time_reversal_config", lambda _: config)
    monkeypatch.setattr(
        cli,
        "run_fixed_time_reversal_benchmark",
        lambda loaded: (
            run_dir,
            {"selection": {"selected_candidate": "reversal-candidate"}},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "btc-directional-model",
            "fixed-120-reversal-benchmark-run",
            "--config",
            str(tmp_path / "benchmark.toml"),
        ],
    )

    cli.main()

    output = capsys.readouterr().out
    assert f"report: {run_dir / 'report.html'}" in output
    assert "development candidate: reversal-candidate" in output
    assert "decision: exact 120 seconds; evidence ends July 28, 2026" in output
    assert "live capital: not authorized" in output


def test_reversal_export_cli_supplies_explicit_paper_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = object()
    captured: dict[str, object] = {}

    def export(**kwargs: object) -> tuple[Path, Path, dict[str, bool]]:
        captured.update(kwargs)
        return tmp_path / "freeze", tmp_path / "runtime", {"production_qualified": False}

    monkeypatch.setattr(cli, "load_fixed_time_reversal_config", lambda _: config)
    monkeypatch.setattr(
        cli,
        "freeze_and_export_fixed_time_reversal_paper_candidate",
        export,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "btc-directional-model",
            "fixed-120-reversal-paper-candidate-export",
            "--config",
            str(tmp_path / "benchmark.toml"),
            "--benchmark-run",
            str(tmp_path / "run"),
            "--model-key",
            "btc-fixed-120-reversal-paper",
            "--authorize-paper-only",
        ],
    )

    cli.main()

    assert captured == {
        "config": config,
        "benchmark_run": tmp_path / "run",
        "model_key": "btc-fixed-120-reversal-paper",
        "authorization": FIXED_TIME_REVERSAL_PAPER_AUTHORIZATION,
    }
    output = capsys.readouterr().out
    assert "scope: paper_only; production-qualified: false" in output
    assert "fresh forward evidence: required from July 29, 2026" in output
