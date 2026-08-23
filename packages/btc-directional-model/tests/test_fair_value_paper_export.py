from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from btc_directional_model.core_extract import file_sha256
from btc_directional_model.fair_value_paper_export import (
    CANDIDATE,
    ENTRY_CELLS,
    FAIR_VALUE_FEATURE_SCHEMA,
    MODEL_KEY,
    SOURCE_FOLD,
    export,
)


def test_specialist_export_reuses_payoff_bundle_and_covers_runtime_contract(
    tmp_path: Path,
) -> None:
    estimator = HistGradientBoostingRegressor(
        max_iter=8,
        max_leaf_nodes=5,
        min_samples_leaf=2,
        learning_rate=0.2,
        random_state=7,
        early_stopping=False,
    )
    signal = np.linspace(-4.0, 4.0, 40)
    values = np.column_stack((signal, np.zeros_like(signal)))
    estimator.fit(values, np.clip(0.5 + values[:, 0] / 10.0, 0.05, 0.95))
    tournament = {
        "source_commit": "a" * 40,
        "diagnostic_leader": CANDIDATE,
        "runtime_exported": False,
        "fold_models": {
            SOURCE_FOLD: {
                "fair_value": {
                    "feature_names": (
                        "signal",
                        "chainlink_candle_return_5m_bps",
                    ),
                    "distilled": estimator,
                }
            }
        },
    }
    tournament_path = tmp_path / "tournament.joblib"
    joblib.dump(tournament, tournament_path)
    policy = {
        "type": "cell_first_crossing",
        "cells": {
            name: {"enabled": True, "confidence": 0.5, "edge": 0.0, "rank": None}
            for name, _, _ in ENTRY_CELLS
        },
    }
    metrics = {
        "run_id": "test-run",
        "model_artifact": {"sha256": file_sha256(tournament_path)},
        "fold_results": {CANDIDATE: {SOURCE_FOLD: {"policy": policy}}},
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))
    rows = []
    observed_at = datetime(2026, 8, 1, tzinfo=UTC)
    for index, (cell, start, _) in enumerate(ENTRY_CELLS):
        for predicted_up, signal in ((False, -4.0), (True, 4.0)):
            rows.append(
                {
                    "market_id": f"{cell}-{predicted_up}",
                    "observed_at": observed_at + timedelta(seconds=index),
                    "seconds_elapsed": start,
                    "entry_cell": cell,
                    "predicted_up": predicted_up,
                    "outer_fold": SOURCE_FOLD,
                    "signal": signal,
                    "chainlink_candle_return_5m_bps": signal,
                    "up_cost_5": 0.1,
                    "down_cost_5": 0.1,
                }
            )
    ledger_path = tmp_path / "ledger.parquet"
    pl.DataFrame(rows).write_parquet(ledger_path)

    destination = export(
        tournament_path=tournament_path,
        metrics_path=metrics_path,
        ledger_path=ledger_path,
        output_root=tmp_path / "runtime-models",
    )

    model = json.loads((destination / "model.json").read_text())
    manifest = json.loads((destination / "manifest.json").read_text())
    vectors = json.loads((destination / "golden-vectors.json").read_text())
    assert destination.name == MODEL_KEY
    assert model["features"]["schema_version"] == FAIR_VALUE_FEATURE_SCHEMA
    assert model["payoff_model"]["kind"] == "fair_value"
    assert model["payoff_model"]["outcome"]["output"] == "regression"
    assert model["prediction_policy"] == {
        "type": "first_confidence_crossing",
        "minimum_seconds_after_open": 15,
        "maximum_seconds_after_open": 240,
        "cadence_seconds": 5,
        "early_end_second": 89,
        "early_cadence_seconds": 1,
    }
    assert manifest["deployment_scope"] == "paper_only"
    assert manifest["production_qualified"] is False
    assert manifest["live_capital_allowed"] is False
    assert manifest["model_sha256"] == file_sha256(destination / "model.json")
    assert len(vectors["vectors"]) == 12
    assert {row["expected"]["action"] for row in vectors["vectors"]} == {
        "up",
        "down",
        "no_trade",
    }
