from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from btc_directional_model import cli
from btc_directional_model.chainlink_oi_config import (
    CHAINLINK_FULL_CANDIDATE,
    CHAINLINK_FULL_OI_CANDIDATE,
    LONG_HISTORY_CANDLE_CANDIDATE,
    load_chainlink_oi_benchmark_config,
)
from btc_directional_model.chainlink_oi_features import ExternalSourceFrames
from btc_directional_model.chainlink_oi_forward_score import (
    ForwardRuntimeModel,
    _validate_forward_range,
    build_chainlink_oi_forward_result,
    score_forward_candidate,
)

PACKAGE_ROOT = Path(__file__).parents[1]
CONFIG_PATH = (
    PACKAGE_ROOT / "configs" / "btc-5m-directional-chainlink-oi-champion-20260321-20260729.toml"
)


def test_forward_range_is_strictly_after_benchmark_window_and_bounded() -> None:
    config = load_chainlink_oi_benchmark_config(CONFIG_PATH)
    _validate_forward_range(
        config,
        datetime(2026, 7, 29, tzinfo=UTC),
        datetime(2026, 8, 2, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="overlaps consumed"):
        _validate_forward_range(
            config,
            datetime(2026, 7, 28, tzinfo=UTC),
            datetime(2026, 7, 30, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="exact UTC day boundary"):
        _validate_forward_range(
            config,
            datetime(2026, 7, 29, 1, tzinfo=UTC),
            datetime(2026, 7, 30, tzinfo=UTC),
        )


def test_runtime_forward_score_uses_first_native_confidence_crossing() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["m1", "m1", "m2", "m2"],
            "window_start": [
                datetime(2026, 7, 29, 0, 0, tzinfo=UTC),
                datetime(2026, 7, 29, 0, 0, tzinfo=UTC),
                datetime(2026, 7, 29, 0, 5, tzinfo=UTC),
                datetime(2026, 7, 29, 0, 5, tzinfo=UTC),
            ],
            "observed_at": [
                datetime(2026, 7, 29, 0, 1, tzinfo=UTC),
                datetime(2026, 7, 29, 0, 1, 5, tzinfo=UTC),
                datetime(2026, 7, 29, 0, 6, tzinfo=UTC),
                datetime(2026, 7, 29, 0, 6, 5, tzinfo=UTC),
            ],
            "seconds_elapsed": [60, 65, 60, 65],
            "label_up": [1, 1, 0, 0],
            "signal": [1.0, 1.0, 1.0, 1.0],
        }
    )
    metrics = score_forward_candidate(frame, _constant_up_runtime_model())

    assert metrics["eligible_markets"] == 2
    assert metrics["accepted_point_rows"] == 4
    assert metrics["first_crossing"]["markets"] == 2
    assert metrics["first_crossing"]["wins"] == 1
    assert metrics["first_crossing"]["losses"] == 1
    assert metrics["first_crossing"]["accuracy"] == pytest.approx(0.5)
    assert metrics["first_crossing"]["median_entry_second"] == pytest.approx(60.0)


def test_zero_core_rows_produce_explicit_source_blocker_report() -> None:
    config = load_chainlink_oi_benchmark_config(CONFIG_PATH)
    candidates = (
        CHAINLINK_FULL_CANDIDATE,
        CHAINLINK_FULL_OI_CANDIDATE,
        LONG_HISTORY_CANDLE_CANDIDATE,
    )
    runtimes = {
        candidate: ForwardRuntimeModel(
            candidate=candidate,
            directory=Path(f"/{candidate}"),
            manifest={
                "model_sha256": "a" * 64,
                "feature_schema_version": "schema",
                "feature_schema_sha256": "b" * 64,
                "deployment_scope": "paper_only",
                "production_qualified": False,
            },
            model={**_constant_up_runtime_model(), "model_key": candidate},
        )
        for candidate in candidates
    }
    inventory = pl.DataFrame(
        {
            "utc_date": [date(2026, 7, 29)],
            "labeled_markets": [288],
            "opening_reference_markets": [0],
            "core_fact_markets": [0],
            "complete_core_history_markets": [0],
            "execution_snapshot_rows": [0],
        }
    )
    empty = pl.DataFrame()
    result = build_chainlink_oi_forward_result(
        config,
        range_start=datetime(2026, 7, 29, tzinfo=UTC),
        range_end=datetime(2026, 7, 30, tzinfo=UTC),
        inventory=inventory,
        raw_core=empty,
        oracle_rounds=empty,
        external=ExternalSourceFrames(empty, empty, empty),
        runtime_models=runtimes,
    )

    assert result["status"] == "blocked_zero_rows"
    assert result["training_performed"] is False
    assert result["database_mutated"] is False
    assert "post-benchmark-window" in result["evidence_classification"]
    assert result["execution_economics"]["execution_snapshot_rows"] == 0
    assert "zero retained forward execution-book rows" in result["execution_economics"]["reason"]
    assert result["source_eligibility"]["inventory_totals"]["labeled_markets"] == 288
    assert any("288 labeled markets lack" in blocker for blocker in result["blockers"])
    assert result["source_eligibility"]["strict_candidate_keys_identical"] is True
    assert all(metrics["eligible_rows"] == 0 for metrics in result["models"].values())


def test_forward_cli_dispatches_exact_utc_range(monkeypatch, capsys, tmp_path: Path) -> None:
    config = SimpleNamespace()
    calls: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_chainlink_oi_benchmark_config", lambda _: config)

    def fake_run(candidate_config, **kwargs):
        calls["config"] = candidate_config
        calls.update(kwargs)
        run_dir = tmp_path / "run"
        return run_dir, {"status": "blocked_zero_rows", "blockers": ["missing"]}

    monkeypatch.setattr(cli, "run_chainlink_oi_forward_score", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "btc-directional-model",
            "chainlink-oi-forward-score",
            "--config",
            str(CONFIG_PATH),
            "--start",
            "2026-07-29",
            "--end",
            "2026-08-02",
            "--output-root",
            str(tmp_path),
        ],
    )

    cli.main()

    assert calls["config"] is config
    assert calls["range_start"] == datetime(2026, 7, 29, tzinfo=UTC)
    assert calls["range_end"] == datetime(2026, 8, 2, tzinfo=UTC)
    assert calls["output_root"] == tmp_path
    assert calls["runtime_model_root"] is None
    assert "status: blocked_zero_rows" in capsys.readouterr().out


def _constant_up_runtime_model() -> dict[str, object]:
    return {
        "schema_version": "capitonic-btc-directional-runtime-model-v2",
        "model_key": "constant-up",
        "features": {"names": ["signal"], "imputation_medians": [0.0]},
        "estimator": {"baseline_logit": 3.0, "trees": []},
        "target": {"type": "outcome_up"},
        "time_bands": [
            {
                "name": "60-240",
                "start_seconds": 60,
                "end_seconds_exclusive": 241,
                "confidence_threshold": 0.89,
                "calibration": {
                    "type": "platt_logit",
                    "slope": 1.0,
                    "intercept": 0.0,
                    "input_probability_clip": {"minimum": 1e-9, "maximum": 1 - 1e-9},
                    "output_logit_clip": {"minimum": -40.0, "maximum": 40.0},
                },
            }
        ],
        "decision": {
            "below_confidence_action": "no_trade",
            "probability_up_threshold": 0.5,
            "up_action": "up",
            "down_action": "down",
        },
    }
