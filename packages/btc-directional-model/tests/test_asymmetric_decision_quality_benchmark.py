from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import btc_directional_model.asymmetric_decision_quality_benchmark as benchmark
from btc_directional_model.asymmetric_decision_quality import (
    MATCHED_CORE_CONTROL_CANDIDATE_ID,
    OOF_SELECTION_COLUMNS,
    decision_quality_oof_key_digest,
)
from btc_directional_model.asymmetric_value_config import load_asymmetric_value_config
from btc_directional_model.asymmetric_value_training import (
    CORE_CANDLES_PRICE,
    CORE_L2_PRICE,
    CORE_ORACLE_L2_PRICE,
    CORE_ORACLE_PRICE,
)
from btc_directional_model.core_extract import file_sha256, write_json_atomic


def _config(tmp_path: Path):
    source = (
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-decision-quality-20260414-20260802.toml"
    )
    return replace(
        load_asymmetric_value_config(source),
        feature_cache=tmp_path / "features",
        runs=tmp_path / "runs",
    )


def _oof(candidate_id: str, *, when: datetime | None = None) -> pl.DataFrame:
    window_start = when or datetime(2026, 6, 11, tzinfo=UTC)
    return pl.DataFrame(
        {
            "candidate_id": [candidate_id],
            "base_candidate": [candidate_id.split("__", maxsplit=1)[0]],
            "fold": ["jun11_jun18"],
            "parent_source": ["alltime"],
            "identity_l2": [1.0],
            "market_id": ["m1"],
            "window_start": [window_start],
            "observed_at": [window_start + timedelta(seconds=5)],
            "seconds_elapsed": [5],
            "label_up": [1],
            "probability_yes": [0.30],
            "yes_target_eligible": [True],
            "no_target_eligible": [False],
            "target_time_band": ["1_15"],
        }
    ).select(*OOF_SELECTION_COLUMNS)


def _seal_payload(run_dir: Path, *, run_id: str) -> dict[str, object]:
    selection = {
        "status": "selected",
        "selected_candidate_id": "candidate",
        "selected_base_candidate": "base",
        "economics_used": False,
    }
    selection_path = run_dir / "decision-quality-selection.json"
    write_json_atomic(selection_path, {"selection": selection})
    oof_path = run_dir / "decision-quality-oof-predictions.parquet"
    oof = _oof("candidate")
    oof.write_parquet(oof_path)
    matched_path = run_dir / "matched-core-oof-predictions.parquet"
    _oof(MATCHED_CORE_CONTROL_CANDIDATE_ID).write_parquet(matched_path)
    matched_probability = {
        "brier_delta": {"point": 0.0, "lower_95": 0.0, "upper_95": 0.0},
        "log_loss_delta": {"point": 0.0, "lower_95": 0.0, "upper_95": 0.0},
    }
    return {
        "run_id": run_id,
        "selection": selection,
        "selection_artifact": selection_path.name,
        "oof_artifact": oof_path.name,
        "matched_core_oof_artifact": matched_path.name,
        "matched_core_probability": matched_probability,
        "oof": {
            "rows": oof.height,
            "markets": 1,
            "key_sha256": decision_quality_oof_key_digest(oof),
            "content_sha256": file_sha256(oof_path),
        },
        "artifact_sha256": {
            selection_path.name: file_sha256(selection_path),
            oof_path.name: file_sha256(oof_path),
            matched_path.name: file_sha256(matched_path),
        },
        "selection_uses_economics": False,
    }


def test_selection_seal_is_canonical_across_runs_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path, first = benchmark.write_decision_selection_seal(
        first_dir,
        _seal_payload(first_dir, run_id="first"),
    )
    _, second = benchmark.write_decision_selection_seal(
        second_dir,
        _seal_payload(second_dir, run_id="second"),
    )

    assert first["selection_identity_sha256"] == second["selection_identity_sha256"]
    tampered = json.loads(first_path.read_text())
    tampered["selection"]["selected_candidate_id"] = "other"
    tampered["selection_identity_sha256"] = benchmark._selection_identity(tampered)
    first_path.write_text(json.dumps(tampered, sort_keys=True))
    with pytest.raises(RuntimeError, match="does not match its selection evidence"):
        benchmark.load_verified_decision_selection_seal(first_path)


def test_selection_seal_rejects_economic_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = _seal_payload(run_dir, run_id="run")
    payload["selection"] = {**payload["selection"], "net_profit": 10.0}
    with pytest.raises(RuntimeError, match="economic field entered"):
        benchmark.write_decision_selection_seal(run_dir, payload)


def _execution_source(*windows: datetime) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "market_id": [f"m{index}" for index in range(len(windows))],
            "window_start": list(windows),
            "observed_at": [value + timedelta(seconds=5) for value in windows],
            "seconds_elapsed": [5] * len(windows),
            "label_up": [1] * len(windows),
            "fee_rate": [0.07] * len(windows),
            "yes_best_ask": [0.25] * len(windows),
            "yes_ask_vwap_5": [0.25] * len(windows),
            "yes_ask_depth": [100.0] * len(windows),
            "no_best_ask": [0.75] * len(windows),
            "no_ask_vwap_5": [0.75] * len(windows),
            "no_ask_depth": [100.0] * len(windows),
            "yes_cost_per_share": [0.27] * len(windows),
            "no_cost_per_share": [0.77] * len(windows),
            "yes_execution_cost_per_share": [0.26] * len(windows),
            "no_execution_cost_per_share": [0.76] * len(windows),
        }
    )


def test_oof_execution_join_excludes_final_calibration(tmp_path: Path) -> None:
    config = _config(tmp_path)
    validation = datetime(2026, 6, 11, tzinfo=UTC)
    final_calibration = datetime(2026, 7, 23, tzinfo=UTC)
    selected = _oof("candidate", when=validation)
    source = _execution_source(final_calibration, validation)

    joined = benchmark._join_oof_probability_to_execution(
        selected,
        source,
        model="candidate",
        config=config,
    )

    assert joined.height == 1
    assert joined["window_start"].to_list() == [validation]
    with pytest.raises(RuntimeError, match="final calibration rows"):
        benchmark._join_oof_probability_to_execution(
            _oof("candidate", when=final_calibration),
            source,
            model="candidate",
            config=config,
        )


def test_economic_reveal_consumes_selected_probability_from_sealed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "sealed"
    run_dir.mkdir()
    seal_path, sealed = benchmark.write_decision_selection_seal(
        run_dir,
        _seal_payload(run_dir, run_id="sealed"),
    )
    captured: list[tuple[str, float]] = []

    monkeypatch.setattr(
        benchmark,
        "paired_probability_delta",
        lambda *args, **kwargs: sealed["matched_core_probability"],
    )

    def stop_after_sealed_read(frame: pl.DataFrame, *args, **kwargs):
        captured.append((frame["candidate_id"][0], frame["probability_yes"][0]))
        raise RuntimeError("sealed bytes observed")

    monkeypatch.setattr(
        benchmark,
        "_join_oof_probability_to_execution",
        stop_after_sealed_read,
    )
    with pytest.raises(RuntimeError, match="sealed bytes observed"):
        benchmark.reveal_decision_quality_economics(
            seal_path=seal_path,
            attribution_manifest_path=tmp_path / "unused-attribution.json",
            development_model_frames={},
            l2_frame=pl.DataFrame(),
            oracle_frame=pl.DataFrame(),
            oof_core=pl.DataFrame(),
            oof_grid_coverage={},
            config=config,
        )

    assert captured == [("candidate", 0.30)]


def test_development_and_fresh_forward_evidence_thresholds_remain_distinct(
    tmp_path: Path,
) -> None:
    scope = benchmark.DEVELOPMENT_OOF_GATE_SCOPE
    assert scope == {
        "name": "consumed_oof_development",
        "policy_window_gates": True,
        "minimum_trades": 100,
        "minimum_trade_utc_days": 5,
        "minimum_strict_markets": 1_000,
        "minimum_strict_grid_coverage": 0.40,
        "minimum_candidate_grid_coverage": 0.40,
        "fresh_forward_contract_is_separate": True,
    }
    config = _config(tmp_path)
    result = benchmark._benchmark_result(
        config=config,
        run_id="run",
        selection={
            "status": "no_qualified_configuration",
            "selected_candidate_id": None,
            "selected_base_candidate": None,
            "candidate_records": [],
            "rank_trace": [],
            "economics_used": False,
        },
        selection_seal={"selection_identity_sha256": "a" * 64},
        selection_seal_sha256="b" * 64,
        development_coverage={},
        economic_evidence=None,
        artifact_hashes={},
    )
    assert result["forward_requirements"]["minimum_selected_trades"] == 200
    assert result["forward_requirements"]["minimum_trade_utc_days"] == 10
    assert result["forward_requirements"]["minimum_strict_markets"] == 2_000
    assert result["forward_requirements"]["minimum_strict_grid_coverage"] == 0.70


def test_post_selection_attribution_manifest_fails_on_artifact_mutation(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "evidence.json"
    artifact.write_text("sealed")
    payload = {
        "schema_version": benchmark.POST_SELECTION_ATTRIBUTION_RUN_SCHEMA_VERSION,
        "created_at": "2026-08-08T00:00:00+00:00",
        "selection_identity_sha256": "a" * 64,
        "selection_uses_economics": False,
        "economics_opened": False,
        "pairs": {name: {} for name in benchmark.POST_SELECTION_ATTRIBUTION_PAIRS},
        "artifact_sha256": {artifact.name: file_sha256(artifact)},
    }
    payload["attribution_identity_sha256"] = benchmark._post_selection_attribution_identity(
        payload
    )
    manifest = tmp_path / "post-selection-attribution.json"
    write_json_atomic(manifest, payload)

    benchmark._load_verified_post_selection_attribution(
        manifest,
        selection_identity_sha256="a" * 64,
    )
    artifact.write_text("mutated")
    with pytest.raises(RuntimeError, match="artifact changed"):
        benchmark._load_verified_post_selection_attribution(
            manifest,
            selection_identity_sha256="a" * 64,
        )


def test_quality_summary_uses_probability_metric_schema_and_zero_trade_report() -> None:
    selection = {
        "status": "selected",
        "selected_candidate_id": "candidate",
        "selected_base_candidate": "base",
        "economics_used": False,
        "rank_trace": [{"candidate_id": "candidate"}],
        "candidate_records": [
            {
                "candidate_id": "candidate",
                "base_candidate": "base",
                "target_weight": 0.25,
                "histogram_profile": "h1",
                "parent_source": "alltime",
                "identity_l2": 1.0,
                "metrics": {
                    "overall": {
                        "rows": 10,
                        "log_loss": 0.61,
                        "brier": 0.21,
                        "bias": 0.01,
                        "ece": 0.02,
                    }
                },
                "gates": [{"name": "quality", "passed": True}],
                "qualified": True,
            }
        ],
    }
    summary = benchmark._quality_selection_summary(selection)
    rows = benchmark._quality_candidate_rows(selection)
    assert summary["selected"]["overall"]["brier"] == 0.21
    assert rows[0]["rank"] == 1
    result = {
        "selection": {"status": "selected", "selected_candidate_id": "candidate"},
        "decision_quality": summary,
        "development": {
            "economic_reveal_status": "blocked",
            "evidence": {
                "qualified": False,
                "selected_metrics": {
                    "trades": 0,
                    "losing_trades": 0,
                    "accuracy": None,
                    "yes_trades": 0,
                    "no_trades": 0,
                    "net_profit": 0.0,
                    "net_expectancy_per_trade": None,
                    "profit_factor": None,
                    "mean_share_price": None,
                    "mean_entry_second": None,
                    "utc_day_block_bootstrap": {},
                },
                "checks": [],
            },
        },
        "evaluation": {"status": "blocked_development_economics"},
    }
    assert "Wins / losses: 0 / 0 (0.00% accuracy)" in benchmark._decision_quality_markdown_report(
        result
    )


def _readiness(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    path = tmp_path / "readiness.json"
    payload: dict[str, object] = {
        "ready": True,
        "readiness_identity_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "range_start": "2026-04-14T00:00:00+00:00",
        "range_end": "2026-08-02T00:00:00+00:00",
        "external_ssd_required": False,
    }
    write_json_atomic(path, payload)
    return path, payload


def test_no_quality_winner_never_opens_economics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    events: list[str] = []
    oof = _oof("control")
    training = {
        "selection": {
            "status": "no_qualified_configuration",
            "selected_candidate_id": None,
            "selected_base_candidate": None,
            "economics_used": False,
        }
    }
    monkeypatch.setattr(
        benchmark,
        "prepare_asymmetric_training_readiness",
        lambda *args, **kwargs: events.append("readiness") or _readiness(tmp_path),
    )
    monkeypatch.setattr(
        benchmark,
        "fit_decision_quality_walk_forward",
        lambda *args, **kwargs: events.append("fit") or (oof, training),
    )
    monkeypatch.setattr(
        benchmark,
        "reveal_decision_quality_economics",
        lambda **kwargs: pytest.fail("economics opened without a quality winner"),
    )

    run_dir, result = benchmark.run_decision_quality_benchmark(
        config=config,
        core_config=object(),
        development_model_frames={
            CORE_L2_PRICE: pl.DataFrame(),
            CORE_ORACLE_PRICE: pl.DataFrame(),
            CORE_CANDLES_PRICE: pl.DataFrame(),
            CORE_ORACLE_L2_PRICE: pl.DataFrame(),
        },
        oof_core=pl.DataFrame(),
        oof_grid_coverage={},
        development_coverage={},
        development_price_manifest={"created_at": "ignored"},
        implementation_sha256="c" * 64,
        dependency_versions={},
        current_process={},
    )

    assert events == ["readiness", "fit"]
    assert result["selection"]["selected_key"] is None
    assert result["evaluation"]["status"] == "blocked_no_quality_configuration"
    assert not (run_dir / "development-economic-reveal.json").exists()


def test_verified_seal_precedes_losing_winner_economic_reveal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    selected_id = "hybrid_25_h3__alltime_l2_1_0"
    selected_oof = _oof(selected_id)
    events: list[str] = []
    selection = {
        "status": "selected",
        "selected_candidate_id": selected_id,
        "selected_base_candidate": "hybrid_25_h3",
        "economics_used": False,
    }
    monkeypatch.setattr(
        benchmark,
        "prepare_asymmetric_training_readiness",
        lambda *args, **kwargs: events.append("readiness") or _readiness(tmp_path),
    )
    monkeypatch.setattr(
        benchmark,
        "fit_decision_quality_walk_forward",
        lambda *args, **kwargs: (
            events.append("walk_forward") or (selected_oof, {"selection": selection})
        ),
    )
    matched = _oof(MATCHED_CORE_CONTROL_CANDIDATE_ID)
    monkeypatch.setattr(
        benchmark,
        "fit_selected_matched_core_walk_forward",
        lambda *args, **kwargs: events.append("matched_oof") or (matched, {}),
    )
    monkeypatch.setattr(
        benchmark,
        "fit_final_decision_quality_model",
        lambda *args, **kwargs: events.append("final") or ({"model": "winner"}, {}),
    )
    monkeypatch.setattr(
        benchmark,
        "fit_final_matched_core_model",
        lambda *args, **kwargs: events.append("final_core") or ({"model": "core"}, {}),
    )
    monkeypatch.setattr(
        benchmark,
        "paired_probability_delta",
        lambda *args, **kwargs: {
            "brier_delta": {"point": -0.01, "lower_95": -0.02, "upper_95": 0.0},
            "log_loss_delta": {"point": -0.01, "lower_95": -0.02, "upper_95": 0.0},
        },
    )
    original_load = benchmark.load_verified_decision_selection_seal

    def verified(path: Path):
        events.append("seal_load")
        return original_load(path)

    monkeypatch.setattr(benchmark, "load_verified_decision_selection_seal", verified)
    attribution_path = tmp_path / "attribution.json"
    attribution_path.write_text("{}")
    monkeypatch.setattr(
        benchmark,
        "_fit_post_selection_attribution",
        lambda **kwargs: events.append("attribution") or attribution_path,
    )
    monkeypatch.setattr(
        benchmark,
        "_load_verified_post_selection_attribution",
        lambda *args, **kwargs: {"artifact_sha256": {}},
    )

    empty = pl.DataFrame({"placeholder": []})

    def reveal(**kwargs):
        events.append("economics")
        assert (
            benchmark.load_verified_decision_selection_seal(kwargs["seal_path"])["selection"][
                "selected_candidate_id"
            ]
            == selected_id
        )
        metrics = {
            "trades": 1,
            "accuracy": 0.0,
            "yes_trades": 1,
            "no_trades": 0,
            "net_profit": -1.0,
            "net_expectancy_per_trade": -1.0,
            "profit_factor": 0.0,
            "mean_share_price": 0.25,
            "mean_entry_second": 5.0,
        }
        return {
            "schema_version": "test",
            "selected_scored": empty,
            "selected_ledger": empty,
            "matched_core_scored": empty,
            "matched_core_ledger": empty,
            "temporal_ledgers": {},
            "vwap10_ledger": empty,
            "incumbent_ledger": empty,
            "common_frequency_ledger": empty,
            "evidence": {
                "status": "blocked",
                "qualified": False,
                "selected_metrics": metrics,
                "checks": [],
                "accuracy_price_by_second": [],
            },
        }

    monkeypatch.setattr(benchmark, "reveal_decision_quality_economics", reveal)
    monkeypatch.setattr(benchmark, "accuracy_price_by_five_second_interval", lambda frame: [])
    monkeypatch.setattr(benchmark, "side_time_price_strata_economics", lambda *args, **kwargs: [])

    _, result = benchmark.run_decision_quality_benchmark(
        config=config,
        core_config=object(),
        development_model_frames={
            CORE_L2_PRICE: pl.DataFrame(),
            CORE_ORACLE_PRICE: pl.DataFrame(),
            CORE_CANDLES_PRICE: pl.DataFrame(),
            CORE_ORACLE_L2_PRICE: pl.DataFrame(),
        },
        oof_core=pl.DataFrame(),
        oof_grid_coverage={},
        development_coverage={},
        development_price_manifest={"created_at": "ignored"},
        implementation_sha256="c" * 64,
        dependency_versions={},
        current_process={},
    )

    assert events.index("readiness") < events.index("walk_forward")
    assert events.index("seal_load") < events.index("economics")
    assert events.index("seal_load") < events.index("attribution") < events.index("economics")
    assert result["selection"]["selected_key"] == selected_id
    assert result["development"]["economically_qualified"] is False
    assert result["deployment"]["authorized"] is False
