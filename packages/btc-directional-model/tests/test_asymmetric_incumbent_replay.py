from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from btc_directional_model.asymmetric_incumbent_replay import (
    DEFAULT_FROZEN_ASYMMETRIC_INCUMBENT_MODEL,
    FROZEN_ASYMMETRIC_INCUMBENT_KEY,
    FROZEN_ASYMMETRIC_INCUMBENT_MODEL_SHA256,
    load_frozen_asymmetric_incumbent,
    replay_frozen_asymmetric_incumbent,
    score_asymmetric_runtime_row,
    validate_asymmetric_runtime_model,
)
from btc_directional_model.asymmetric_value_training import (
    CORE_ORACLE_PRICE,
    CORE_PRICE,
    asymmetric_value_feature_sets,
)
from btc_directional_model.runtime_export import feature_schema_sha256


def test_frozen_incumbent_identity_and_feature_contract_are_pinned() -> None:
    model = load_frozen_asymmetric_incumbent()

    assert model.path == DEFAULT_FROZEN_ASYMMETRIC_INCUMBENT_MODEL
    assert model.payload["model_key"] == FROZEN_ASYMMETRIC_INCUMBENT_KEY
    assert model.model_sha256 == FROZEN_ASYMMETRIC_INCUMBENT_MODEL_SHA256
    assert model.feature_contract == CORE_ORACLE_PRICE
    assert len(model.feature_names) == 75
    assert len(model.manifest_sha256) == 64


def test_frozen_incumbent_reproduces_exported_golden_probability() -> None:
    model = load_frozen_asymmetric_incumbent()
    golden = json.loads(model.path.with_name("golden-vectors.json").read_text())[
        "vectors"
    ][0]

    scored = score_asymmetric_runtime_row(
        model,
        golden["feature_values"],
        seconds_elapsed=golden["seconds_elapsed"],
        yes_ask_vwap=golden["yes_ask_vwap"],
        no_ask_vwap=golden["no_ask_vwap"],
    )

    assert scored["raw_logit"] == pytest.approx(
        golden["expected"]["raw_logit"],
        abs=2e-15,
    )
    assert scored["probability_up"] == golden["expected"]["probability_up"]
    assert scored["confidence"] == golden["expected"]["confidence"]
    assert scored["action"] == golden["expected"]["action"]
    assert scored["accepted"] is True

    with_missing = list(golden["feature_values"])
    with_missing[0] = None
    missing_scored = score_asymmetric_runtime_row(
        model,
        with_missing,
        seconds_elapsed=golden["seconds_elapsed"],
        yes_ask_vwap=golden["yes_ask_vwap"],
        no_ask_vwap=golden["no_ask_vwap"],
    )
    assert missing_scored == scored


def test_runtime_validation_accepts_only_frozen_71_or_75_feature_order() -> None:
    model = load_frozen_asymmetric_incumbent()
    assert validate_asymmetric_runtime_model(model.payload) == CORE_ORACLE_PRICE

    core_payload = copy.deepcopy(model.payload)
    core_names = list(asymmetric_value_feature_sets()[CORE_PRICE])
    medians_by_name = dict(
        zip(
            model.payload["features"]["names"],
            model.payload["features"]["imputation_medians"],
            strict=True,
        )
    )
    core_payload["features"]["names"] = core_names
    core_payload["features"]["imputation_medians"] = [
        medians_by_name[name] for name in core_names
    ]
    core_payload["features"]["schema_sha256"] = feature_schema_sha256(
        core_payload["features"]["schema_version"],
        core_names,
    )
    core_payload["estimator"]["trees"] = [
        {"nodes": [{"kind": "leaf", "value": 0.0}]}
    ]
    core_payload["provenance"]["candidate"] = CORE_PRICE
    assert validate_asymmetric_runtime_model(core_payload) == CORE_PRICE

    reordered = copy.deepcopy(model.payload)
    reordered["features"]["names"][0:2] = reversed(
        reordered["features"]["names"][0:2]
    )
    reordered["features"]["schema_sha256"] = feature_schema_sha256(
        reordered["features"]["schema_version"],
        reordered["features"]["names"],
    )
    with pytest.raises(ValueError, match="71/75 contract"):
        validate_asymmetric_runtime_model(reordered)


def test_runtime_scoring_enforces_schedule_and_closed_final_price_cell() -> None:
    model = load_frozen_asymmetric_incumbent()
    medians = model.payload["features"]["imputation_medians"]

    final_cell = score_asymmetric_runtime_row(
        model,
        medians,
        seconds_elapsed=60,
        yes_ask_vwap=1.0,
        no_ask_vwap=0.0,
    )
    assert 0.0 < final_cell["probability_up"] < 1.0

    with pytest.raises(ValueError, match="prediction schedule"):
        score_asymmetric_runtime_row(
            model,
            medians,
            seconds_elapsed=61,
            yes_ask_vwap=0.25,
            no_ask_vwap=0.75,
        )


def test_frozen_replay_applies_lower_price_policy_and_reports_frequency() -> None:
    replay = replay_frozen_asymmetric_incumbent(_policy_frame())

    assert replay.model_key == FROZEN_ASYMMETRIC_INCUMBENT_KEY
    assert replay.feature_contract == CORE_ORACLE_PRICE
    assert replay.eligible_resolved_markets == 3
    assert replay.selected_trades.height == 2
    assert replay.trades_per_eligible_resolved_market == pytest.approx(2 / 3)
    assert replay.selected_trades["market_id"].to_list() == ["market-a", "market-b"]
    assert replay.selected_trades["seconds_elapsed"].to_list() == [30, 30]
    assert replay.selected_trades["selected_yes"].to_list() == [True, False]
    assert replay.selected_trades["quantity"].to_list() == [5.0, 5.0]
    assert replay.audit_hashes["model_artifact_sha256"] == (
        FROZEN_ASYMMETRIC_INCUMBENT_MODEL_SHA256
    )
    assert all(len(value) == 64 for value in replay.audit_hashes.values())


def test_frozen_replay_fails_closed_on_duplicate_or_unresolved_rows() -> None:
    frame = _policy_frame()
    duplicate = pl.concat((frame, frame.head(1)), how="vertical")
    with pytest.raises(ValueError, match="duplicate market decision points"):
        replay_frozen_asymmetric_incumbent(duplicate)

    unresolved = frame.with_columns(
        pl.when(pl.col("market_id") == "market-c")
        .then(None)
        .otherwise(pl.col("label_up"))
        .alias("label_up")
    )
    with pytest.raises(ValueError, match="resolved binary outcomes"):
        replay_frozen_asymmetric_incumbent(unresolved)


def test_frozen_replay_rejects_modified_model_bytes(tmp_path: Path) -> None:
    model_dir = tmp_path / FROZEN_ASYMMETRIC_INCUMBENT_KEY
    model_dir.mkdir()
    source = DEFAULT_FROZEN_ASYMMETRIC_INCUMBENT_MODEL
    (model_dir / "model.json").write_text(source.read_text() + "\n")
    (model_dir / "manifest.json").write_text(source.with_name("manifest.json").read_text())

    with pytest.raises(RuntimeError, match="model SHA-256 changed"):
        load_frozen_asymmetric_incumbent(model_dir / "model.json")


def _policy_frame() -> pl.DataFrame:
    model = load_frozen_asymmetric_incumbent()
    feature_columns = {
        name: [float(median)] * 4
        for name, median in zip(
            model.feature_names,
            model.payload["features"]["imputation_medians"],
            strict=True,
        )
    }
    start = datetime(2026, 7, 23, tzinfo=UTC)
    window_starts = [start, start, start + timedelta(minutes=5), start + timedelta(minutes=10)]
    seconds = [30, 31, 30, 30]
    yes_prices = [0.25, 0.25, 0.75, 0.30]
    no_prices = [0.75, 0.75, 0.25, 0.70]
    return pl.DataFrame(
        {
            **feature_columns,
            "market_id": ["market-a", "market-a", "market-b", "market-c"],
            "window_start": window_starts,
            "observed_at": [
                window_start + timedelta(seconds=second)
                for window_start, second in zip(window_starts, seconds, strict=True)
            ],
            "seconds_elapsed": seconds,
            "label_up": [1, 1, 0, 1],
            "fee_rate": [0.0] * 4,
            "yes_best_ask": yes_prices,
            "yes_ask_vwap_5": yes_prices,
            "yes_ask_depth": [100.0] * 4,
            "no_best_ask": no_prices,
            "no_ask_vwap_5": no_prices,
            "no_ask_depth": [100.0] * 4,
            "yes_cost_per_share": [0.26, 0.26, 0.76, 0.31],
            "no_cost_per_share": [0.76, 0.76, 0.26, 0.71],
            "yes_execution_cost_per_share": yes_prices,
            "no_execution_cost_per_share": no_prices,
        }
    )
