from __future__ import annotations

import io
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl

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
    MODEL_SELECTION_ELIGIBLE,
    OFFLINE_ONLY_CANDIDATES,
    ORACLE_MATCHED_CORE_PRICE_CONTROL,
    PRICE_LOGISTIC,
    THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL,
    AsymmetricCalibrationCell,
    AsymmetricValueModel,
    _coherent_calibration_objective,
    _price_band_indices,
    asymmetric_value_feature_sets,
    fit_side_price_time_calibrators,
)
from btc_directional_model.chainlink_oi_features import CHAINLINK_CANDLE_FEATURES
from btc_directional_model.core_training import ProbabilityCalibrator
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
