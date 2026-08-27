from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest

from btc_directional_model.continuous_edge_training import BOOK_RAW_FEATURES
from btc_directional_model.counterfactual_twap_state_data import (
    CHAINLINK_UNCERTAINTY_BPS,
    _rolling_mean_at,
    construct_binance_labels,
)
from btc_directional_model.counterfactual_twap_state_tournament import (
    CANDIDATE_NAMES,
    CHECKPOINT_SCHEMA_VERSION,
    FEATURE_TREATMENTS,
    HISTORY_ARMS,
    CheckpointStore,
    _bootstrap_means,
    _economic_frame,
    execution_settings,
    feature_names,
    load_config,
    predetermined_hyperparameters,
    score_model,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_ROOT / "configs/btc-5m-counterfactual-twap-state-20260321-20260826.toml"


def test_contract_is_new_training_only_family_with_exact_schedule() -> None:
    config = load_config(CONFIG)

    assert config.model_family == "btc-5m-counterfactual-twap-state"
    assert config.start == datetime(2026, 3, 21, tzinfo=UTC)
    assert config.chainlink_start == datetime(2026, 6, 7, tzinfo=UTC)
    assert config.authentic_start == datetime(2026, 8, 1, tzinfo=UTC)
    assert config.current_start == datetime(2026, 8, 14, tzinfo=UTC)
    assert config.candidate_freeze == datetime(2026, 8, 25, tzinfo=UTC)
    assert config.raw["training"]["training_only"] is True
    assert config.raw["training"]["live_capital_allowed"] is False
    assert tuple(tuple(value) for value in config.raw["entry"]["cells"]) == (
        (60, 90), (90, 120), (120, 150), (150, 180)
    )


def test_tournament_dimensions_and_search_are_frozen() -> None:
    config = load_config(CONFIG)

    assert FEATURE_TREATMENTS == (
        "refprice_control", "twap30", "twap60", "dual_twap", "combined",
        "combined_disagreement",
    )
    assert len(HISTORY_ARMS) == 5
    assert len(CANDIDATE_NAMES) == 6
    assert len(predetermined_hyperparameters(config)) == 36
    assert len(set(predetermined_hyperparameters(config))) == 36
    assert not set(feature_names("combined_disagreement")) & {
        "label_source", "label_up", "target_margin_bps", "window_start"
    }


def test_binance_twap_uses_completed_left_closed_right_open_closes() -> None:
    start = datetime(2026, 3, 21, tzinfo=UTC)
    times = [start - timedelta(seconds=60) + timedelta(seconds=i) for i in range(360)]
    binance = pl.DataFrame(
        {
            "market_id": ["m"] * 360,
            "window_start": [start] * 360,
            "window_end": [start + timedelta(minutes=5)] * 360,
            "open_timestamp": times,
            "available_at": [value + timedelta(seconds=1) for value in times],
            "close_price": np.arange(100.0, 460.0),
        }
    )
    labels = pl.DataFrame(
        {
            "market_id": ["m"], "window_start": [start],
            "window_end": [start + timedelta(minutes=5)],
        }
    )

    result = construct_binance_labels(labels, binance)

    assert result["binance_open_prints"].item() == 60
    assert result["binance_close_prints"].item() == 60
    assert result["binance_open_twap60"].item() == np.mean(np.arange(100.0, 160.0))
    assert result["binance_close_twap60"].item() == np.mean(np.arange(400.0, 460.0))


def test_source_queries_are_bounded_select_only_and_add_no_schema() -> None:
    names = (
        "btc-twap60-label-source.sql", "btc-twap60-refprice-source.sql",
        "btc-twap60-core-current-source.sql", "btc-core-oracle-source.sql",
        "btc-twap60-candle-source.sql", "btc-capacity-execution-evidence.sql",
        "btc-counterfactual-twap-binance-source.sql",
    )
    forbidden = ("insert ", "update ", "delete ", "create ", "alter ", "drop ")
    for name in names:
        sql = (PACKAGE_ROOT / "sql" / name).read_text().lower()
        assert "select" in sql
        assert not any(token in sql for token in forbidden)
        assert "batch_start" in sql or "history_start" in sql
        assert "batch_end" in sql or "range_end" in sql


def test_chainlink_uncertainty_exclusion_boundary_is_not_lowered() -> None:
    assert CHAINLINK_UNCERTAINTY_BPS == 0.526


def test_rolling_twap_rejects_unsorted_evidence_before_modeling() -> None:
    start = np.datetime64("2026-03-21T00:00:00", "us")
    times = start + np.array([0, 2, 1], dtype="timedelta64[s]")
    points = start + np.array([30], dtype="timedelta64[s]")

    try:
        _rolling_mean_at(times, np.array([100.0, 102.0, 101.0]), points, 30)
    except ValueError as error:
        assert "unique increasing timestamps" in str(error)
    else:
        raise AssertionError("unsorted completed evidence must fail closed")


def test_scoring_empty_chronological_window_preserves_scored_schema() -> None:
    result = score_model(
        pl.DataFrame(
            schema={
                "market_id": pl.String,
                "label_up": pl.Int8,
            }
        ),
        object(),  # type: ignore[arg-type]
    )

    assert result.is_empty()
    assert result.schema == {
        "market_id": pl.String,
        "label_up": pl.Int8,
        "probability_up": pl.Float64,
        "predicted_margin_lower_bps": pl.Float64,
        "predicted_margin_bps": pl.Float64,
        "predicted_margin_upper_bps": pl.Float64,
    }


def test_economic_frame_applies_current_regime_datetime_filter() -> None:
    config = load_config(CONFIG)
    row = {
        "window_start": [config.current_start],
        "label_source": ["authentic_official_twap60"],
        "label_up": [1],
        "probability_up": [0.6],
        "fee_rate": [0.0],
    }
    row.update({name: [0.5] for name in BOOK_RAW_FEATURES})

    result = _economic_frame(pl.DataFrame(row), config)

    assert result.height == 1
    assert result["label_regime"].item() == "authentic_official_twap60"


def test_vectorized_bootstrap_preserves_seeded_legacy_draws() -> None:
    values = np.linspace(-1.5, 2.5, 1001)
    legacy_rng = np.random.default_rng(20260826)
    legacy = np.array(
        [legacy_rng.choice(values, len(values), replace=True).mean() for _ in range(137)]
    )

    optimized = _bootstrap_means(values, resamples=137, seed=20260826)

    np.testing.assert_array_equal(optimized, legacy)


def test_execution_settings_use_bounded_fit_parallelism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BTC_TWAP_TRAINING_WORKERS", "2")
    monkeypatch.setenv("BTC_TWAP_TRAINING_THREADS_PER_FIT", "3")

    settings = execution_settings()

    assert settings.workers == 2
    assert settings.threads_per_fit == 3
    assert len(predetermined_hyperparameters(load_config(CONFIG))) == 36


def test_execution_settings_reject_unbounded_parallel_histogram_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BTC_TWAP_TRAINING_WORKERS", "3")

    with pytest.raises(ValueError, match="at most two"):
        execution_settings()

    monkeypatch.setenv("BTC_TWAP_TRAINING_WORKERS", "2")
    monkeypatch.setenv("BTC_TWAP_TRAINING_THREADS_PER_FIT", "4")
    with pytest.raises(ValueError, match="CPU budget"):
        execution_settings()


def test_checkpoint_resume_requires_exact_training_identity(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "exact-source-config-runtime-identity")
    value = {"selected_index": 7, "combinations": 36}

    store.save("hyperparameter-search", value)

    assert store.load("hyperparameter-search") == value
    assert CheckpointStore(tmp_path, "different-identity").load(
        "hyperparameter-search"
    ) is None
    payload = joblib.load(tmp_path / "hyperparameter-search.joblib")
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
