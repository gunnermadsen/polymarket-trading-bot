from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model.settlement_bridge_residual_tournament import (
    BASE_FEATURES,
    CANDIDATES,
    INFERENCE_FEATURES,
    RELATIVE_TWAP_FEATURES,
    SUPERVISION_FIELDS,
    _attach_capacity,
    _candidate_metrics,
    _causal_piecewise_average,
    _market_equal_weights,
    _predictive_selection,
    _prepare_economic_rows,
    _render_report,
    base_specs,
    causal_feature_registry,
    load_config,
    residual_specs,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/btc-5m-settlement-bridge-residual-20260321-20260828.toml"


def test_contract_freezes_exact_candidates_windows_and_searches() -> None:
    config = load_config(CONFIG)

    assert CANDIDATES == (
        "refprice_bridge_baseline",
        "non_twap_settlement_correction",
        "relative_twap_settlement_correction",
        "twap_margin_residual_bridge",
    )
    assert config.candidate_freeze == datetime(2026, 8, 28, tzinfo=UTC)
    assert config.windows.reconstruction_start == datetime(2026, 6, 7, tzinfo=UTC)
    assert config.windows.paired_training_end == datetime(2026, 8, 1, tzinfo=UTC)
    assert len(base_specs(config)) == 12
    assert len(base_specs(config)) <= 36
    assert len(residual_specs(config)) == 4
    assert len(residual_specs(config)) <= 12


def test_inference_contract_excludes_supervision_and_date_shortcuts() -> None:
    assert not set(INFERENCE_FEATURES) & set(SUPERVISION_FIELDS)
    assert not set(INFERENCE_FEATURES) & {"window_start", "market_date", "label_source", "regime"}
    assert "hour_sin" not in BASE_FEATURES
    assert "weekday_sin" not in BASE_FEATURES


def test_every_inference_feature_has_causal_registry_metadata() -> None:
    registry = causal_feature_registry()
    by_name = {row["feature"]: row for row in registry}

    assert set(by_name) == set(INFERENCE_FEATURES)
    for row in registry:
        assert row["source_event_timestamp"]
        assert row["source_availability_timestamp"]
        assert row["lookback"]
        assert row["feature_as_of"]
        assert row["live_computable"] is True


def test_causal_twap_uses_only_reports_available_before_target() -> None:
    start = np.datetime64("2026-06-07T00:00:00", "us").astype(np.int64)
    source = start + np.array([0, 30, 60, 90], dtype=np.int64) * 1_000_000
    available = source + 1_000_000
    # The 90-second report is deliberately unavailable until after the target.
    available[-1] = start + 200 * 1_000_000
    prices = np.array([100.0, 110.0, 120.0, 1_000_000.0])
    target = np.array([start + 120 * 1_000_000], dtype=np.int64)

    result = _causal_piecewise_average(source, available, prices, target, 60)

    assert np.allclose(result, [120.0])


def test_market_equal_weights_do_not_overweight_observation_count() -> None:
    frame = pl.DataFrame({"market_id": ["a", "a", "b"]})

    weights = _market_equal_weights(frame)

    assert np.isclose(weights[:2].sum(), weights[2])


def test_read_only_queries_are_bounded_and_have_no_mutations() -> None:
    sql = (ROOT / "sql/btc-settlement-bridge-binance-label-diagnostic.sql").read_text().lower()
    assert "batch_start" in sql and "batch_end" in sql
    assert not any(token in sql for token in (
        "insert ", "update ", "delete ", "create table", "alter table", "drop table"
    ))


def test_relative_twap_roster_is_complete_and_contains_no_absolute_price() -> None:
    assert len(RELATIVE_TWAP_FEATURES) == 13
    assert not any(name in {"price", "btc_price", "twap30", "twap60"} for name in RELATIVE_TWAP_FEATURES)


def test_empty_filtered_capacity_preserves_prediction_rows_as_ineligible() -> None:
    instant = datetime(2026, 8, 26, tzinfo=UTC)
    frame = pl.DataFrame({"market_id": ["m"], "observed_at": [instant]})
    capacity = pl.DataFrame(
        {
            "market_id": ["m"],
            "observed_at": [instant],
            "seconds_elapsed": [60],
            "quality_flags": [1],
            "up_provider_received_at": [instant],
            "down_provider_received_at": [instant],
        }
    )

    result = _attach_capacity(frame, capacity, 10)

    assert result.height == 1
    assert result["up_ask_vwap_5"].null_count() == 1


def test_report_serializes_numpy_scalar_metrics() -> None:
    result = {
        "run_id": "test-run",
        "model_family": "btc-5m-settlement-bridge-residual",
        "source_commit": "deadbeef",
        "deployment_status": "not_deployed",
        "conclusion": "no settlement bridge is justified",
        "predictive_selection": {"passed": np.bool_(True)},
    }

    report = _render_report(result)

    assert '"passed": true' in report


def test_predictive_selection_reports_every_required_pair_after_failed_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG)
    probabilities = {
        CANDIDATES[0]: (0.1, 0.9),
        CANDIDATES[1]: (0.4, 0.6),
        CANDIDATES[2]: (0.0, 1.0),
        CANDIDATES[3]: (0.0, 1.0),
    }
    rows = []
    for candidate, values in probabilities.items():
        for index, probability in enumerate(values):
            rows.append(
                {
                    "market_id": f"m{index}",
                    "observed_at": datetime(2026, 8, 14, 0, index, tzinfo=UTC),
                    "twap_label_up": index,
                    "probability_up": probability,
                    "candidate": candidate,
                    "fold": "f1",
                    "seconds_elapsed": 60,
                }
            )
    ledger = pl.DataFrame(rows)
    monkeypatch.setattr(
        "btc_directional_model.settlement_bridge_residual_tournament._candidate_metrics",
        lambda _frame: {"ece": 0.01},
    )
    monkeypatch.setattr(
        "btc_directional_model.settlement_bridge_residual_tournament._bootstrap_mean",
        lambda values, _resamples, _seed: {
            "lower": float(values.mean() - 0.001),
            "mean": float(values.mean()),
            "upper": float(values.mean() + 0.001),
        },
    )
    monkeypatch.setattr(
        "btc_directional_model.settlement_bridge_residual_tournament._fold_improvement_count",
        lambda *_args: 2,
    )
    monkeypatch.setattr(
        "btc_directional_model.settlement_bridge_residual_tournament._candidate_coherence",
        lambda _frame: True,
    )

    result = _predictive_selection(ledger, [], config)

    assert set(result["comparisons"]) == {
        "settlement_bridge_is_useful",
        "non_twap_correction_is_useful",
        "relative_twap_adds_value",
        "margin_residual_is_superior",
    }
    assert result["comparisons"]["relative_twap_adds_value"]["passed"] is True
    assert result["winner"] == CANDIDATES[0]


def test_candidate_metrics_derive_price_buckets_from_executable_side() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["up", "down"],
            "window_start": [
                datetime(2026, 8, 14, tzinfo=UTC),
                datetime(2026, 8, 14, tzinfo=UTC),
            ],
            "seconds_elapsed": [60, 90],
            "probability_up": [0.8, 0.2],
            "twap_label_up": [1, 0],
            "up_ask_vwap_5": [0.7, 0.9],
            "down_ask_vwap_5": [0.4, 0.6],
        }
    )

    metrics = _candidate_metrics(frame)

    assert set(metrics["by_price_bucket"]) == {"<0.65", "0.65-0.75"}


def test_economic_rows_fill_null_classification_intervals_from_base() -> None:
    config = load_config(CONFIG)
    frame = pl.DataFrame(
        {
            "market_id": ["m"],
            "up_ask_vwap_5": [0.4],
            "down_ask_vwap_5": [0.7],
            "probability_up": [0.8],
            "fee_rate": [0.0],
            "twap_label_up": [1],
            "base_margin_lower": [1.0],
            "base_margin_median": [2.0],
            "base_margin_upper": [3.0],
            "adjusted_margin_lower": [None],
            "adjusted_margin_median": [None],
            "adjusted_margin_upper": [None],
        },
        schema_overrides={
            "adjusted_margin_lower": pl.Float64,
            "adjusted_margin_median": pl.Float64,
            "adjusted_margin_upper": pl.Float64,
        },
    )

    result = _prepare_economic_rows(frame, config)

    assert result.select(
        "adjusted_margin_lower", "adjusted_margin_median", "adjusted_margin_upper"
    ).row(0) == (1.0, 2.0, 3.0)
