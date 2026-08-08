from __future__ import annotations

import io
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest

import btc_directional_model.asymmetric_value_training as asymmetric_training
from btc_directional_model.asymmetric_value_config import load_asymmetric_value_config
from btc_directional_model.asymmetric_value_data import (
    EARLY_CAUSAL_ORACLE_FEATURES,
    POLYMARKET_VALUE_FEATURES,
)
from btc_directional_model.asymmetric_value_training import (
    ASYMMETRIC_VALUE_CANDIDATES,
    ASYMMETRIC_VALUE_MODEL_MATRIX,
    CANDLE_MATCHED_CORE_PRICE_CONTROL,
    CORE_CANDLES_PRICE,
    CORE_L2_PRICE,
    CORE_ORACLE_L2_PRICE,
    CORE_ORACLE_PRICE,
    CORE_PRICE,
    EXPECTED_MODEL_FEATURE_COUNTS,
    L2_MATCHED_CORE_PRICE_CONTROL,
    MATCHED_ATTRIBUTION_CONTROLS,
    MODEL_SELECTION_ELIGIBLE,
    OFFLINE_ONLY_CANDIDATES,
    ORACLE_MATCHED_CORE_PRICE_CONTROL,
    PRICE_LOGISTIC,
    THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL,
    AsymmetricCalibrationCell,
    AsymmetricValueModel,
    _coherent_calibration_objective,
    _coherent_probability_from_parameters,
    _price_band_indices,
    _target_fit_key_digest,
    asymmetric_value_feature_sets,
    fit_side_price_time_calibrators,
    select_target_fit_cohort,
    target_calibration_evidence,
    target_calibration_gate_checks,
    target_fit_cohort_contract,
)
from btc_directional_model.chainlink_oi_features import CHAINLINK_CANDLE_FEATURES
from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_training import ProbabilityCalibrator, market_equal_weights
from btc_directional_model.early_value_training import TimeBandCalibrator
from btc_directional_model.spot_l2_chainlink_features import L2_FEATURES


class _ZeroLogitModel:
    candidate_name = "zero_logit"

    def raw_logit(self, frame: pl.DataFrame) -> np.ndarray:
        return np.zeros(frame.height, dtype=np.float64)


class _ConstantLogitModel:
    candidate_name = "constant_logit"

    def raw_logit(self, frame: pl.DataFrame) -> np.ndarray:
        return np.full(frame.height, np.log(0.7 / 0.3), dtype=np.float64)


def _calibrated_bundle() -> AsymmetricValueModel:
    band = TimeBandCalibrator(
        start_second=1,
        end_second_exclusive=241,
        calibrator=ProbabilityCalibrator(
            slope=1.0,
            intercept=0.0,
            converged=True,
            iterations=1,
        ),
        rows=100,
        markets=100,
    )
    cells = []
    for price_index in range(10):
        for side in ("YES", "NO"):
            cells.append(
                AsymmetricCalibrationCell(
                    start_second=1,
                    end_second_exclusive=241,
                    minimum_price=price_index / 10,
                    maximum_price=(price_index + 1) / 10,
                    side=side,
                    slope=1.0,
                    intercept=0.4 if side == "YES" and price_index == 2 else 0.0,
                    fitted=side == "YES" and price_index == 2,
                    fallback=None if side == "YES" and price_index == 2 else "test_parent",
                    rows=100,
                    markets=100,
                    utc_days=5,
                    positives=50,
                    negatives=50,
                    identity_l2_strength=1.0,
                    converged=side == "YES" and price_index == 2,
                    iterations=1,
                    objective=0.5,
                    weighted_log_loss=0.5,
                )
            )
    return AsymmetricValueModel(
        name="test_asymmetric",
        model=_ZeroLogitModel(),  # type: ignore[arg-type]
        time_calibrators=(band,),
        cells=tuple(cells),
    )


def test_exact_model_matrix_and_attribution_control_contract() -> None:
    feature_sets = asymmetric_value_feature_sets()

    assert ASYMMETRIC_VALUE_CANDIDATES == (
        PRICE_LOGISTIC,
        CORE_PRICE,
        L2_MATCHED_CORE_PRICE_CONTROL,
        CORE_L2_PRICE,
        CANDLE_MATCHED_CORE_PRICE_CONTROL,
        CORE_CANDLES_PRICE,
        ORACLE_MATCHED_CORE_PRICE_CONTROL,
        CORE_ORACLE_PRICE,
        THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL,
        CORE_ORACLE_L2_PRICE,
    )
    assert set(feature_sets) == set(ASYMMETRIC_VALUE_CANDIDATES)
    assert ASYMMETRIC_VALUE_MODEL_MATRIX == (
        CORE_PRICE,
        CORE_ORACLE_PRICE,
        CORE_L2_PRICE,
        CORE_CANDLES_PRICE,
        CORE_ORACLE_L2_PRICE,
    )
    assert {
        name: len(feature_sets[name]) for name in ASYMMETRIC_VALUE_MODEL_MATRIX
    } == EXPECTED_MODEL_FEATURE_COUNTS == {
        CORE_PRICE: 71,
        CORE_ORACLE_PRICE: 75,
        CORE_L2_PRICE: 111,
        CORE_CANDLES_PRICE: 79,
        CORE_ORACLE_L2_PRICE: 115,
    }
    assert MODEL_SELECTION_ELIGIBLE == frozenset(
        {CORE_PRICE, CORE_ORACLE_PRICE, CORE_L2_PRICE}
    )
    assert CORE_ORACLE_L2_PRICE in OFFLINE_ONLY_CANDIDATES
    assert CORE_ORACLE_L2_PRICE not in MODEL_SELECTION_ELIGIBLE
    assert set(POLYMARKET_VALUE_FEATURES).issubset(feature_sets[CORE_PRICE])
    for name, features in feature_sets.items():
        optional_sources = sum(
            bool(set(source).intersection(features))
            for source in (
                L2_FEATURES,
                CHAINLINK_CANDLE_FEATURES,
                EARLY_CAUSAL_ORACLE_FEATURES,
            )
        )
        expected_sources = 2 if name == CORE_ORACLE_L2_PRICE else 1
        assert optional_sources <= expected_sources
    assert set(L2_FEATURES).issubset(feature_sets[CORE_ORACLE_L2_PRICE])
    assert set(EARLY_CAUSAL_ORACLE_FEATURES).issubset(
        feature_sets[CORE_ORACLE_L2_PRICE]
    )


def test_early_oracle_contract_excludes_unproven_boundary_features() -> None:
    feature_sets = asymmetric_value_feature_sets()

    assert "oracle_gap_to_opening_boundary_bps" not in EARLY_CAUSAL_ORACLE_FEATURES
    assert "oracle_boundary_binance_path_agreement" not in (
        EARLY_CAUSAL_ORACLE_FEATURES
    )
    assert set(EARLY_CAUSAL_ORACLE_FEATURES).issubset(
        feature_sets[CORE_ORACLE_PRICE]
    )


def test_oracle_ablation_has_a_same_cohort_core_price_control() -> None:
    feature_sets = asymmetric_value_feature_sets()

    assert ORACLE_MATCHED_CORE_PRICE_CONTROL in ASYMMETRIC_VALUE_CANDIDATES
    assert ORACLE_MATCHED_CORE_PRICE_CONTROL not in MODEL_SELECTION_ELIGIBLE
    assert feature_sets[ORACLE_MATCHED_CORE_PRICE_CONTROL] == feature_sets[CORE_PRICE]
    assert set(EARLY_CAUSAL_ORACLE_FEATURES).isdisjoint(
        feature_sets[ORACLE_MATCHED_CORE_PRICE_CONTROL]
    )


def test_l2_ablation_has_a_same_cohort_core_price_control() -> None:
    feature_sets = asymmetric_value_feature_sets()

    assert L2_MATCHED_CORE_PRICE_CONTROL in ASYMMETRIC_VALUE_CANDIDATES
    assert L2_MATCHED_CORE_PRICE_CONTROL not in MODEL_SELECTION_ELIGIBLE
    assert feature_sets[L2_MATCHED_CORE_PRICE_CONTROL] == feature_sets[CORE_PRICE]
    assert set(L2_FEATURES).isdisjoint(feature_sets[L2_MATCHED_CORE_PRICE_CONTROL])


def test_raw_price_band_boundaries_are_left_closed() -> None:
    observed = _price_band_indices(
        np.asarray([0.199999, 0.20, 0.299999, 0.30, 1.0])
    )

    assert observed.tolist() == [1, 2, 2, 3, 9]


def test_target_fit_cohort_uses_exact_time_and_half_open_either_side_boundaries() -> None:
    config = load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-calibrated-20260414-20260802.toml"
    )
    start = config.fit.start
    rows = [
        ("accept_yes_min", 1, 0.20, 0.80, 0),
        ("accept_no_below_max", 55, 0.70, 0.299999, 1),
        ("reject_second_0", 0, 0.25, 0.75, 0),
        ("reject_second_56", 56, 0.25, 0.75, 0),
        ("reject_below_min", 5, 0.199999, 0.80, 1),
        ("reject_exact_max", 5, 0.30, 0.70, 0),
        ("reject_neither_side", 5, 0.50, 0.50, 1),
        ("reject_nonfinite", 5, float("nan"), 0.70, 0),
    ]
    frame = pl.DataFrame(
        {
            "market_id": [row[0] for row in rows],
            "window_start": [start for _ in rows],
            "observed_at": [start + timedelta(seconds=row[1]) for row in rows],
            "seconds_elapsed": [row[1] for row in rows],
            "label_up": [row[4] for row in rows],
            "yes_ask_vwap_5": [row[2] for row in rows],
            "no_ask_vwap_5": [row[3] for row in rows],
        }
    )

    selected = select_target_fit_cohort(frame, config, model="boundary_test")

    assert set(selected["market_id"].to_list()) == {
        "accept_yes_min",
        "accept_no_below_max",
    }
    assert _target_fit_key_digest(selected) == _target_fit_key_digest(
        selected.reverse()
    )
    assert target_fit_cohort_contract(config) == {
        "policy": "raw20_30_by55_edge_3c",
        "fit_window_start": "2026-04-14T00:00:00+00:00",
        "fit_window_end_exclusive": "2026-07-16T00:00:00+00:00",
        "minimum_entry_second": 1,
        "maximum_entry_second": 55,
        "entry_second_interval": "closed",
        "minimum_raw_share_price": 0.20,
        "maximum_raw_share_price": 0.30,
        "raw_share_price_interval": "left_closed_right_open",
        "side_eligibility": "either_yes_or_no_raw_vwap_5",
        "price_columns": ["yes_ask_vwap_5", "no_ask_vwap_5"],
        "label_column": "label_up",
        "required_labels": [0, 1],
    }


def test_target_fit_cohort_fails_closed_without_both_outcomes() -> None:
    config = load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-calibrated-20260414-20260802.toml"
    )
    start = config.fit.start
    frame = pl.DataFrame(
        {
            "market_id": ["m1", "m2"],
            "window_start": [start, start],
            "observed_at": [
                start + timedelta(seconds=1),
                start + timedelta(seconds=2),
            ],
            "seconds_elapsed": [1, 2],
            "label_up": [1, 1],
            "yes_ask_vwap_5": [0.25, 0.25],
            "no_ask_vwap_5": [0.75, 0.75],
        }
    )

    with pytest.raises(RuntimeError, match="requires both outcomes"):
        select_target_fit_cohort(frame, config, model="single_class")


def test_target_fit_cohort_preserves_equal_total_weight_per_market() -> None:
    config = load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-calibrated-20260414-20260802.toml"
    )
    start = config.fit.start
    frame = pl.DataFrame(
        {
            "market_id": ["m1", "m1", "m1", "m2"],
            "window_start": [start] * 4,
            "observed_at": [start + timedelta(seconds=value) for value in (1, 2, 56, 1)],
            "seconds_elapsed": [1, 2, 56, 1],
            "label_up": [0, 1, 0, 1],
            "yes_ask_vwap_5": [0.25, 0.25, 0.25, 0.25],
            "no_ask_vwap_5": [0.75, 0.75, 0.75, 0.75],
        }
    )
    selected = select_target_fit_cohort(frame, config, model="weight_test")
    weights = market_equal_weights(selected)
    totals = (
        selected.with_columns(pl.Series("weight", weights))
        .group_by("market_id")
        .agg(pl.col("weight").sum())
        .sort("market_id")["weight"]
        .to_numpy()
    )

    np.testing.assert_allclose(totals, np.repeat(totals[0], len(totals)))


def test_model_wiring_fits_every_candidate_on_target_rows_and_seals_matched_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-calibrated-20260414-20260802.toml"
    )
    core_config = load_core_config(config.core_config)
    rows = [
        ("fit_yes", config.fit.start, 1, 0.20, 0.80, 0),
        ("fit_no", config.fit.start + timedelta(minutes=5), 55, 0.75, 0.25, 1),
        ("fit_late", config.fit.start + timedelta(minutes=10), 56, 0.25, 0.75, 0),
        ("fit_mid", config.fit.start + timedelta(minutes=15), 5, 0.50, 0.50, 1),
        ("calibration", config.calibration.start, 1, 0.25, 0.75, 0),
        ("policy", config.policy.start, 1, 0.25, 0.75, 1),
    ]
    columns: dict[str, list[object]] = {
        "market_id": [row[0] for row in rows],
        "window_start": [row[1] for row in rows],
        "observed_at": [row[1] + timedelta(seconds=row[2]) for row in rows],
        "seconds_elapsed": [row[2] for row in rows],
        "label_up": [row[5] for row in rows],
        "yes_ask_vwap_5": [row[3] for row in rows],
        "no_ask_vwap_5": [row[4] for row in rows],
    }
    all_features = set().union(*asymmetric_value_feature_sets().values())
    for feature in all_features - set(columns):
        columns[feature] = [0.0] * len(rows)
    source = pl.DataFrame(columns)
    frames = {name: source for name in ASYMMETRIC_VALUE_CANDIDATES}
    observed_fit_frames: dict[str, pl.DataFrame] = {}

    def fake_fit_model(
        frame: pl.DataFrame,
        spec: asymmetric_training.CandidateSpec,
        *_: object,
    ) -> object:
        observed_fit_frames[spec.name] = frame
        return object()

    class FakeBundle:
        def __init__(self, **_: object) -> None:
            pass

        def probability(self, frame: pl.DataFrame) -> np.ndarray:
            return np.full(frame.height, 0.5, dtype=np.float64)

    monkeypatch.setattr(asymmetric_training, "fit_model", fake_fit_model)
    monkeypatch.setattr(
        asymmetric_training,
        "fit_asymmetric_time_band_calibrators",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        asymmetric_training,
        "fit_side_price_time_calibrators",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        asymmetric_training,
        "target_calibration_evidence",
        lambda *_args, **_kwargs: {"required": False, "qualified": True},
    )
    monkeypatch.setattr(
        asymmetric_training,
        "_calibration_coverage",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(asymmetric_training, "AsymmetricValueModel", FakeBundle)

    _, summary = asymmetric_training.fit_asymmetric_value_models(
        frames,
        config,
        core_config,
    )

    assert set(observed_fit_frames) == set(ASYMMETRIC_VALUE_CANDIDATES)
    for name, fit_frame in observed_fit_frames.items():
        assert set(fit_frame["market_id"].to_list()) == {"fit_yes", "fit_no"}
        profile = summary["profiles"][name]
        assert profile["source_fit_rows"] == 4
        assert profile["source_fit_markets"] == 4
        assert profile["fit_rows"] == profile["target_fit_rows"] == 2
        assert profile["fit_markets"] == profile["target_fit_markets"] == 2
        assert profile["target_fit_contract"] == summary["target_fit_cohort"]["contract"]
        assert profile["target_fit_key_sha256"] == summary["target_fit_cohort"][
            "candidate_evidence"
        ][name]["key_sha256"]
        assert len(profile["target_fit_key_sha256"]) == 64
    for candidate, control in MATCHED_ATTRIBUTION_CONTROLS.items():
        assert (
            summary["profiles"][candidate]["target_fit_key_sha256"]
            == summary["profiles"][control]["target_fit_key_sha256"]
        )
        assert summary["target_fit_cohort"]["matched_control_key_checks"][candidate][
            "matched"
        ]


def test_side_corrections_renormalize_and_ignore_evaluation_labels() -> None:
    bundle = _calibrated_bundle()
    frame = pl.DataFrame(
        {
            "seconds_elapsed": [1, 5],
            "yes_ask_vwap_5": [0.25, 0.25],
            "no_ask_vwap_5": [0.75, 0.75],
            "label_up": [0, 1],
        }
    )

    first = bundle.probability(frame)
    changed_labels = bundle.probability(
        frame.with_columns((1 - pl.col("label_up")).alias("label_up"))
    )

    assert np.all((first > 0.0) & (first < 1.0))
    assert np.all(first > 0.5)
    np.testing.assert_allclose(first, changed_labels)


def test_identity_side_cells_preserve_parent_probability() -> None:
    source = _calibrated_bundle()
    bundle = AsymmetricValueModel(
        name="identity",
        model=_ConstantLogitModel(),  # type: ignore[arg-type]
        time_calibrators=source.time_calibrators,
        cells=tuple(
            replace(
                cell,
                slope=1.0,
                intercept=0.0,
                fitted=False,
                fallback="identity",
            )
            for cell in source.cells
        ),
    )
    frame = pl.DataFrame(
        {
            "seconds_elapsed": [1, 15, 60],
            "yes_ask_vwap_5": [0.25, 0.45, 0.75],
            "no_ask_vwap_5": [0.75, 0.55, 0.25],
        }
    )

    np.testing.assert_allclose(bundle.probability(frame), 0.7)


def test_coherent_joint_calibration_gradient_matches_finite_difference() -> None:
    parameters = np.asarray([1.1, 0.1, 0.9, -0.1], dtype=np.float64)
    parent_logit = np.asarray([-0.4, 0.2, 0.8, -0.1], dtype=np.float64)
    labels = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    weights = np.full(4, 0.25, dtype=np.float64)
    yes_prices = np.asarray([2, 2, 3, 2], dtype=np.int16)
    no_prices = np.asarray([7, 6, 7, 7], dtype=np.int16)
    active_keys = ((2, 0), (7, 1))
    penalty_weights = np.asarray([0.75, 0.75], dtype=np.float64)
    objective, gradient = _coherent_calibration_objective(
        parameters,
        parent_logit,
        labels,
        weights,
        yes_prices,
        no_prices,
        active_keys,
        penalty_weights,
        1.0,
    )
    epsilon = 1e-6
    numerical = np.empty_like(parameters)
    for index in range(len(parameters)):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_value = _coherent_calibration_objective(
            plus,
            parent_logit,
            labels,
            weights,
            yes_prices,
            no_prices,
            active_keys,
            penalty_weights,
            1.0,
        )[0]
        minus_value = _coherent_calibration_objective(
            minus,
            parent_logit,
            labels,
            weights,
            yes_prices,
            no_prices,
            active_keys,
            penalty_weights,
            1.0,
        )[0]
        numerical[index] = (plus_value - minus_value) / (2.0 * epsilon)

    assert np.isfinite(objective)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-6)


def test_coherent_calibration_objective_matches_runtime_probability() -> None:
    parameters = np.asarray([1.2, 0.3, 0.8, -0.2], dtype=np.float64)
    parent_logit = np.asarray([-0.7, 0.1, 0.9, -0.2], dtype=np.float64)
    labels = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    weights = np.asarray([0.10, 0.20, 0.30, 0.40], dtype=np.float64)
    yes_prices = np.asarray([2, 2, 3, 2], dtype=np.int16)
    no_prices = np.asarray([7, 6, 7, 7], dtype=np.int16)
    active_keys = ((2, 0), (7, 1))
    penalty_weights = np.asarray([0.7, 0.8], dtype=np.float64)
    identity_l2 = 0.5
    objective, _ = _coherent_calibration_objective(
        parameters,
        parent_logit,
        labels,
        weights,
        yes_prices,
        no_prices,
        active_keys,
        penalty_weights,
        identity_l2,
    )
    slopes = np.ones((10, 2), dtype=np.float64)
    intercepts = np.zeros_like(slopes)
    for offset, key in enumerate(active_keys):
        slopes[key] = parameters[2 * offset]
        intercepts[key] = parameters[2 * offset + 1]
    probability = _coherent_probability_from_parameters(
        parent_logit,
        yes_prices,
        no_prices,
        slopes,
        intercepts,
    )
    expected_log_loss = -np.sum(
        weights
        * (
            labels * np.log(probability)
            + (1.0 - labels) * np.log(1.0 - probability)
        )
    )
    delta = parameters.copy()
    delta[0::2] -= 1.0
    expected_penalty = 0.5 * identity_l2 * float(
        (np.repeat(penalty_weights, 2) * delta) @ delta
    )

    np.testing.assert_allclose(
        objective,
        expected_log_loss + expected_penalty,
        rtol=1e-12,
        atol=1e-12,
    )


def test_asymmetric_calibration_joblib_round_trip_preserves_routing() -> None:
    bundle = _calibrated_bundle()
    frame = pl.DataFrame(
        {
            "seconds_elapsed": [1],
            "yes_ask_vwap_5": [0.25],
            "no_ask_vwap_5": [0.75],
        }
    )
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)

    np.testing.assert_allclose(bundle.probability(frame), restored.probability(frame))


def test_sparse_side_price_cells_fall_back_to_parent_time_calibration() -> None:
    config = load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-one-second-20260414-20260802.toml"
    )
    calibrators = tuple(
        TimeBandCalibrator(
            start_second=start,
            end_second_exclusive=end,
            calibrator=ProbabilityCalibrator(1.0, 0.0, True, 1),
            rows=1,
            markets=1,
        )
        for start, end in config.calibration_bands
    )
    start = datetime(2026, 7, 6, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "market_id": ["m1"],
            "window_start": [start],
            "seconds_elapsed": [1],
            "yes_ask_vwap_5": [0.25],
            "no_ask_vwap_5": [0.75],
            "label_up": [1],
        }
    )

    cells = fit_side_price_time_calibrators(
        _ZeroLogitModel(),  # type: ignore[arg-type]
        calibrators,
        frame,
        config,
    )

    assert len(cells) == len(config.calibration_bands) * 20
    assert not any(cell.fitted for cell in cells)
    yes_twenty_to_thirty = next(
        cell
        for cell in cells
        if cell.start_second == 1
        and cell.side == "YES"
        and cell.minimum_price == 0.2
    )
    assert "insufficient_markets" in (yes_twenty_to_thirty.fallback or "")
    assert yes_twenty_to_thirty.slope == 1.0
    assert yes_twenty_to_thirty.intercept == 0.0


def test_target_calibration_fallback_is_explicitly_disqualifying() -> None:
    config = load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-calibrated-20260414-20260802.toml"
    )
    calibrators = tuple(
        TimeBandCalibrator(
            start_second=start,
            end_second_exclusive=end,
            calibrator=ProbabilityCalibrator(1.0, 0.0, True, 1),
            rows=1,
            markets=1,
        )
        for start, end in config.calibration_bands
    )
    frame = pl.DataFrame(
        {
            "market_id": ["m1"],
            "window_start": [datetime(2026, 7, 16, tzinfo=UTC)],
            "seconds_elapsed": [1],
            "yes_ask_vwap_5": [0.25],
            "no_ask_vwap_5": [0.75],
            "label_up": [1],
        }
    )

    cells = fit_side_price_time_calibrators(
        _ZeroLogitModel(),  # type: ignore[arg-type]
        calibrators,
        frame,
        config,
    )
    evidence = target_calibration_evidence(cells, config)
    checks = target_calibration_gate_checks(
        {"side_price_time_calibration": {"target_contract": evidence}}
    )

    assert len(cells) == 160
    assert len(evidence["cells"]) == 8
    assert evidence["fitted_cells"] == 0
    assert evidence["fallback_cells"] == 8
    assert evidence["qualified"] is False
    assert checks[0]["name"] == "target_calibration_cells_genuinely_fitted"
    assert checks[0]["passed"] is False
    assert all(not check["passed"] for check in checks[1:])
    assert any(
        "parent_fallback" in cell["failure_reasons"]
        for cell in evidence["cells"]
    )


def test_all_eight_supported_target_cells_pass_qualification() -> None:
    config = load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-calibrated-20260414-20260802.toml"
    )
    target = config.target_calibration
    assert target is not None
    cells: list[AsymmetricCalibrationCell] = []
    for start, end in config.calibration_bands:
        for price_index in range(10):
            minimum_price = price_index / 10
            maximum_price = (price_index + 1) / 10
            for side in ("YES", "NO"):
                targeted = (
                    (start, end) in target.time_bands
                    and np.isclose(minimum_price, target.minimum_price)
                    and np.isclose(maximum_price, target.maximum_price)
                )
                cells.append(
                    AsymmetricCalibrationCell(
                        start_second=start,
                        end_second_exclusive=end,
                        minimum_price=minimum_price,
                        maximum_price=maximum_price,
                        side=side,
                        slope=1.0,
                        intercept=0.0,
                        fitted=targeted,
                        fallback=None if targeted else "parent_time",
                        rows=100 if targeted else 0,
                        markets=50 if targeted else 0,
                        utc_days=5 if targeted else 0,
                        positives=25 if targeted else 0,
                        negatives=25 if targeted else 0,
                        identity_l2_strength=1.0,
                        converged=targeted,
                        iterations=3 if targeted else 0,
                        objective=0.5 if targeted else None,
                        weighted_log_loss=0.5 if targeted else None,
                    )
                )

    evidence = target_calibration_evidence(tuple(cells), config)
    checks = target_calibration_gate_checks(
        {"side_price_time_calibration": {"target_contract": evidence}}
    )

    assert len(cells) == 160
    assert evidence["fitted_cells"] == 8
    assert evidence["fallback_cells"] == 0
    assert evidence["qualified"] is True
    assert len(checks) == 9
    assert all(check["passed"] for check in checks)
