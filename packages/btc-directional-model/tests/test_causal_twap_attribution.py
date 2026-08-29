from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest

from btc_directional_model.causal_twap_attribution_data import (
    CAUSAL_AVAILABILITY_COLUMNS,
    CHAINLINK_UNCERTAINTY_BPS,
    SUPERVISION_ONLY_FIELDS,
    _rolling_mean_at,
    causal_availability_audit,
    construct_binance_labels,
    inference_feature_registry,
    validate_inference_features,
)
from btc_directional_model.causal_twap_attribution_tournament import (
    ATTRIBUTION_TREATMENTS,
    CANDIDATE_NAMES,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointStore,
    _apply_all_missing_feature_mask,
    _bootstrap_means,
    _complete_utc_day_audit,
    _development_fold_frames,
    _economic_frame,
    _json_default,
    _matrix,
    _neutralize_all_missing_fit_columns,
    _paired_bootstrap,
    execution_settings,
    feature_names,
    load_config,
    predetermined_hyperparameters,
    score_model,
    select_policy,
)
from btc_directional_model.continuous_edge_training import BOOK_RAW_FEATURES

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_ROOT / "configs/btc-5m-causal-twap-attribution-20260607-20260828.toml"


def test_contract_is_new_training_only_family_with_exact_schedule() -> None:
    config = load_config(CONFIG)

    assert config.model_family == "btc-5m-counterfactual-twap-state"
    assert config.start == datetime(2026, 3, 21, tzinfo=UTC)
    assert config.chainlink_start == datetime(2026, 6, 7, tzinfo=UTC)
    assert config.authentic_start == datetime(2026, 8, 1, tzinfo=UTC)
    assert config.current_start == datetime(2026, 8, 14, tzinfo=UTC)
    assert config.candidate_freeze == datetime(2026, 8, 26, tzinfo=UTC)
    assert config.end == datetime(2026, 8, 28, tzinfo=UTC)
    assert config.raw["training"]["training_only"] is True
    assert config.raw["training"]["live_capital_allowed"] is False
    assert tuple(tuple(value) for value in config.raw["entry"]["cells"]) == (
        (60, 90),
        (90, 120),
        (120, 150),
        (150, 180),
    )


def test_tournament_dimensions_and_search_are_frozen() -> None:
    config = load_config(CONFIG)

    assert ATTRIBUTION_TREATMENTS == (
        "refprice_only",
        "refprice_non_twap_basis",
        "refprice_relative_twap",
        "refprice_absolute_twap",
    )
    assert CANDIDATE_NAMES == ATTRIBUTION_TREATMENTS
    assert len(predetermined_hyperparameters(config)) == 36
    assert len(set(predetermined_hyperparameters(config))) == 36
    assert not set(feature_names("refprice_absolute_twap")) & {
        "label_source",
        "label_up",
        "target_margin_bps",
        "window_start",
    }


def test_every_tournament_feature_is_causal_registered_and_supervision_free() -> None:
    registry = inference_feature_registry()
    for treatment in ATTRIBUTION_TREATMENTS:
        features = feature_names(treatment)
        validate_inference_features(features)
        assert set(features) <= set(registry)
        assert not set(features) & SUPERVISION_ONLY_FIELDS
    for metadata in registry.values():
        assert set(metadata) >= {
            "role",
            "source",
            "source_event_timestamp",
            "source_availability_timestamp",
            "lookback_interval",
            "feature_as_of_timestamp",
            "live_computable",
        }


def test_inference_matrix_rejects_completed_market_supervision() -> None:
    frame = pl.DataFrame(
        {
            "refprice_return_from_open_bps": [1.0],
            "estimated_synthetic_label_error": [0.01],
        }
    )

    with pytest.raises(ValueError, match="supervision-only"):
        _matrix(frame, ("estimated_synthetic_label_error",))


def test_future_binance_prices_cannot_change_earlier_rolling_twap() -> None:
    start = np.datetime64("2026-08-20T12:00:00", "us")
    times = start + np.arange(300).astype("timedelta64[s]")
    point = start + np.array([60], dtype="timedelta64[s]")
    original = np.linspace(100.0, 110.0, 300)
    perturbed = original.copy()
    perturbed[61:] = 1_000_000.0

    before = _rolling_mean_at(times, original, point, 60)
    after = _rolling_mean_at(times, perturbed, point, 60)

    np.testing.assert_array_equal(before, after)


def test_causal_availability_audit_fails_on_future_source_timestamp() -> None:
    observed = datetime(2026, 8, 20, 12, 1, tzinfo=UTC)
    valid = pl.DataFrame(
        {
            "observed_at": [observed],
            **{name: [observed - timedelta(seconds=1)] for name in CAUSAL_AVAILABILITY_COLUMNS},
        }
    )
    assert causal_availability_audit(valid)["passed"] is True

    invalid = valid.with_columns(
        pl.lit(observed + timedelta(seconds=1)).alias(CAUSAL_AVAILABILITY_COLUMNS[1])
    )
    with pytest.raises(RuntimeError, match="future source availability"):
        causal_availability_audit(invalid)


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
            "market_id": ["m"],
            "window_start": [start],
            "window_end": [start + timedelta(minutes=5)],
        }
    )

    result = construct_binance_labels(labels, binance)

    assert result["binance_open_prints"].item() == 60
    assert result["binance_close_prints"].item() == 60
    assert result["binance_open_twap60"].item() == np.mean(np.arange(100.0, 160.0))
    assert result["binance_close_twap60"].item() == np.mean(np.arange(400.0, 460.0))
    assert result["opening_binance_max_available_at"].item() == start


def test_source_queries_are_bounded_select_only_and_add_no_schema() -> None:
    names = (
        "btc-twap60-label-source.sql",
        "btc-twap60-refprice-source.sql",
        "btc-twap60-core-current-source.sql",
        "btc-core-oracle-source.sql",
        "btc-twap60-candle-source.sql",
        "btc-capacity-execution-evidence.sql",
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


def test_all_missing_fit_feature_is_neutralized_for_fit_and_scoring() -> None:
    fit = np.array([[np.nan, 1.0], [np.nan, 2.0], [np.nan, 3.0]])
    scoring = np.array([[17.0, 4.0], [np.nan, 5.0]])

    normalized_fit, indices = _neutralize_all_missing_fit_columns(fit)
    normalized_scoring = _apply_all_missing_feature_mask(scoring, indices)

    assert indices == (0,)
    np.testing.assert_array_equal(normalized_fit[:, 0], np.zeros(3))
    np.testing.assert_array_equal(normalized_scoring[:, 0], np.zeros(2))
    np.testing.assert_array_equal(normalized_scoring[:, 1], scoring[:, 1])


def test_development_folds_are_market_disjoint_and_chronological() -> None:
    config = load_config(CONFIG)
    frame = pl.DataFrame(
        {
            "market_id": ["before", "test"],
            "window_start": [
                datetime(2026, 8, 13, tzinfo=UTC),
                datetime(2026, 8, 14, tzinfo=UTC),
            ],
        }
    )

    folds = _development_fold_frames(frame, config)
    fit, test = folds["official_20260814_15"]

    assert fit["market_id"].to_list() == ["before"]
    assert test["market_id"].to_list() == ["test"]


def test_complete_day_audit_rejects_partial_post_freeze_days() -> None:
    complete = datetime(2026, 8, 25, tzinfo=UTC)
    partial = datetime(2026, 8, 26, tzinfo=UTC)
    rows = [
        {
            "market_id": f"complete-{index}",
            "window_start": complete + timedelta(minutes=5 * index),
        }
        for index in range(288)
    ]
    rows.extend(
        {
            "market_id": f"partial-{index}",
            "window_start": partial + timedelta(minutes=5 * index),
        }
        for index in range(239)
    )

    audit = _complete_utc_day_audit(pl.DataFrame(rows))

    assert audit["complete_dates"] == [complete.date()]
    assert audit["days"][1]["complete"] is False


def test_report_serializer_handles_complete_day_dates() -> None:
    assert _json_default(datetime(2026, 8, 25, tzinfo=UTC).date()) == "2026-08-25"


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


def test_policy_selection_requires_isolated_authentic_execution_evidence() -> None:
    with pytest.raises(RuntimeError, match="policy-calibration"):
        select_policy(
            pl.DataFrame(),
            load_config(CONFIG),
            strict_guard=False,
            seed=20260828,
        )


def test_vectorized_bootstrap_preserves_seeded_legacy_draws() -> None:
    values = np.linspace(-1.5, 2.5, 1001)
    legacy_rng = np.random.default_rng(20260826)
    legacy = np.array(
        [legacy_rng.choice(values, len(values), replace=True).mean() for _ in range(137)]
    )

    optimized = _bootstrap_means(values, resamples=137, seed=20260826)

    np.testing.assert_array_equal(optimized, legacy)


def test_paired_bootstrap_uses_every_observation_then_resamples_markets() -> None:
    start = datetime(2026, 8, 14, tzinfo=UTC)
    base = {
        "market_id": ["a", "a", "b", "b"],
        "observed_at": [start + timedelta(seconds=value) for value in (60, 65, 60, 65)],
        "seconds_elapsed": [60, 65, 60, 65],
        "window_start": [start, start, start + timedelta(minutes=5), start + timedelta(minutes=5)],
        "label_up": [1, 1, 0, 0],
        "target_margin_bps": [2.0, 2.0, -2.0, -2.0],
    }
    candidate = pl.DataFrame({**base, "probability_up": [0.9, 0.8, 0.1, 0.2]})
    control = pl.DataFrame({**base, "probability_up": [0.7, 0.7, 0.3, 0.3]})

    result = _paired_bootstrap(candidate, control, seed=7, resamples=100)

    expected = np.mean(
        (candidate["probability_up"].to_numpy() - np.array(base["label_up"])) ** 2
        - (control["probability_up"].to_numpy() - np.array(base["label_up"])) ** 2
    )
    assert result["markets"] == 2
    assert result["candidate_minus_control_brier"] == pytest.approx(expected)


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
    assert CheckpointStore(tmp_path, "different-identity").load("hyperparameter-search") is None
    payload = joblib.load(tmp_path / "hyperparameter-search.joblib")
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
