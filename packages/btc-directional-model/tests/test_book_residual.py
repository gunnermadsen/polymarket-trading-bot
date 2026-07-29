from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from btc_directional_model.book_residual import (
    BOOK_RESIDUAL_FEATURES,
    CALIBRATION_BANDS,
    RAW_DIRECTIONS,
    STANDARDIZED_BOOK_RESIDUAL_FEATURES,
    STRICT_BOOK_RESIDUAL_INPUT_COLUMNS,
    CalibrationCell,
    DirectionTimeCalibrator,
    ResidualBookModel,
    calibration_route_keys,
    derive_strict_book_residual_features,
    fit_direction_time_calibrator,
    fit_residual_book_model,
    route_book_residual_probability,
)


def strict_frame(rows: int = 4) -> pl.DataFrame:
    observed = [datetime(2026, 7, 16, tzinfo=UTC) + timedelta(seconds=5 * i) for i in range(rows)]
    return pl.DataFrame(
        {
            "market_id": [f"market-{i // 2}" for i in range(rows)],
            "observed_at": observed,
            "model_eligible": [True] * rows,
            "book_up_mid": [0.55 + 0.01 * i for i in range(rows)],
            "book_down_mid": [0.45 - 0.01 * i for i in range(rows)],
            "book_up_ask_vwap_10": [0.57 + 0.01 * i for i in range(rows)],
            "book_down_ask_vwap_10": [0.47 - 0.01 * i for i in range(rows)],
            "book_up_imbalance": [0.20 + 0.02 * i for i in range(rows)],
            "book_down_imbalance": [-0.10 - 0.01 * i for i in range(rows)],
            "book_up_log_bid_depth": [2.0 + 0.1 * i for i in range(rows)],
            "book_down_log_bid_depth": [1.7 - 0.05 * i for i in range(rows)],
            "book_up_log_ask_depth": [1.8 + 0.03 * i for i in range(rows)],
            "book_down_log_ask_depth": [2.1 - 0.02 * i for i in range(rows)],
            "book_up_spread": [0.02 + 0.001 * i for i in range(rows)],
            "book_down_spread": [0.03 - 0.001 * i for i in range(rows)],
            "book_up_mid_delta_5s": [0.01] * rows,
            "book_down_mid_delta_5s": [-0.01] * rows,
            "book_up_imbalance_delta_5s": [0.02] * rows,
            "book_down_imbalance_delta_5s": [-0.01] * rows,
        }
    )


def swapped_frame(frame: pl.DataFrame) -> pl.DataFrame:
    pairs = (
        ("book_up_mid", "book_down_mid"),
        ("book_up_ask_vwap_10", "book_down_ask_vwap_10"),
        ("book_up_imbalance", "book_down_imbalance"),
        ("book_up_log_bid_depth", "book_down_log_bid_depth"),
        ("book_up_log_ask_depth", "book_down_log_ask_depth"),
        ("book_up_spread", "book_down_spread"),
        ("book_up_mid_delta_5s", "book_down_mid_delta_5s"),
        ("book_up_imbalance_delta_5s", "book_down_imbalance_delta_5s"),
    )
    expressions: list[pl.Expr] = []
    paired_names = {name for pair in pairs for name in pair}
    expressions.extend(pl.col(name) for name in frame.columns if name not in paired_names)
    for up, down in pairs:
        expressions.extend((pl.col(down).alias(up), pl.col(up).alias(down)))
    return frame.select(expressions)


def eight_calibration_cells(
    *,
    slope: float = 1.0,
    intercept: float = 0.0,
) -> tuple[CalibrationCell, ...]:
    return tuple(
        CalibrationCell(
            band=band.name,
            raw_direction=direction,
            slope=slope,
            intercept=intercept,
            converged=True,
            optimizer_status=0,
            optimizer_message="test",
            iterations=1,
            function_evaluations=1,
            rows=10,
            markets=10,
            positives=5,
            identity_l2_strength=1.0,
            objective=0.5,
            weighted_log_loss=0.5,
            penalty=0.0,
        )
        for band in CALIBRATION_BANDS
        for direction in RAW_DIRECTIONS
    )


def test_identity_model_is_exactly_the_unmodified_core() -> None:
    core = np.asarray((-2.0, -0.5, 0.0, 1.0, 3.0))
    features = np.arange(40, dtype=np.float64).reshape(5, 8) / 10
    identity = ResidualBookModel.identity()

    np.testing.assert_array_equal(identity.raw_logit(core, features), core)
    np.testing.assert_array_equal(
        identity.probability(core, features),
        1.0 / (1.0 + np.exp(-core)),
    )


def test_router_preserves_core_probability_bit_for_bit_on_fallback_rows() -> None:
    core = np.asarray((-1.5, -0.4, 0.6, 1.7))
    features = np.full((4, 8), np.nan)
    features[[1, 3]] = 0.2
    model = ResidualBookModel(
        gamma=0.5,
        beta=(0.1,) * 7,
        feature_scales=(1.0,) * 7,
        l2_strength=1.0,
    )
    eligible = np.asarray((False, True, False, True))

    probabilities, diagnostics = route_book_residual_probability(
        model,
        core,
        features,
        eligible,
    )

    expected_core = 1.0 / (1.0 + np.exp(-core))
    np.testing.assert_array_equal(probabilities[~eligible], expected_core[~eligible])
    assert diagnostics.residual_rows == 2
    assert diagnostics.core_fallback_rows == 2


def test_all_eight_derived_features_are_antisymmetric_under_side_swap() -> None:
    frame = strict_frame()
    core = np.asarray((0.4, -0.3, 0.7, -0.6))

    features = derive_strict_book_residual_features(frame, core)
    swapped = derive_strict_book_residual_features(swapped_frame(frame), -core)

    assert features.shape == (4, 8)
    np.testing.assert_allclose(swapped, -features, rtol=1e-12, atol=1e-12)


def test_regularized_fit_is_deterministic_and_l2_shrinks_correction() -> None:
    rng = np.random.default_rng(41)
    markets = np.asarray([f"market-{i // 6}" for i in range(600)])
    core = rng.normal(0.0, 0.7, len(markets))
    features = rng.normal(0.0, 1.0, (len(markets), 8))
    outcome_logit = core + 0.8 * features[:, 0] + 0.5 * features[:, 1]
    labels = (rng.random(len(markets)) < 1.0 / (1.0 + np.exp(-outcome_logit))).astype(int)

    loose, loose_diagnostics = fit_residual_book_model(
        core,
        features,
        labels,
        markets,
        l2_strength=0.01,
    )
    repeat, repeat_diagnostics = fit_residual_book_model(
        core,
        features,
        labels,
        markets,
        l2_strength=0.01,
    )
    tight, _ = fit_residual_book_model(
        core,
        features,
        labels,
        markets,
        l2_strength=100.0,
    )

    assert loose == repeat
    assert loose_diagnostics == repeat_diagnostics
    assert loose_diagnostics.converged
    assert 0.0 <= loose.gamma <= 1.0
    loose_norm = np.linalg.norm((loose.gamma, *loose.beta))
    tight_norm = np.linalg.norm((tight.gamma, *tight.beta))
    assert tight_norm < loose_norm


def test_fitted_and_calibrated_probabilities_remain_finite() -> None:
    rng = np.random.default_rng(7)
    rows_per_cell = 24
    raw = np.concatenate(
        [
            np.linspace(-2.0, -0.1, rows_per_cell)
            if direction == "DOWN"
            else np.linspace(0.1, 2.0, rows_per_cell)
            for _band in CALIBRATION_BANDS
            for direction in RAW_DIRECTIONS
        ]
    )
    seconds = np.concatenate(
        [
            np.full(rows_per_cell, band.start_second)
            for band in CALIBRATION_BANDS
            for _direction in RAW_DIRECTIONS
        ]
    )
    labels = (rng.random(len(raw)) < 1.0 / (1.0 + np.exp(-raw))).astype(int)
    for index in range(0, len(labels), rows_per_cell):
        labels[index : index + 2] = (0, 1)
    markets = np.asarray([f"market-{i}" for i in range(len(raw))])

    calibrator, diagnostics = fit_direction_time_calibrator(
        raw,
        seconds,
        labels,
        markets,
        identity_l2_strength=2.0,
        minimum_rows_per_cell=20,
        minimum_markets_per_cell=20,
    )
    probabilities = calibrator.probability(raw, seconds)

    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0.0) & (probabilities < 1.0)).all()
    assert all(cell.slope >= 0.0 and cell.converged for cell in calibrator.cells)
    assert np.isfinite(diagnostics.raw_to_calibrated_direction_flip_rate)
    assert DirectionTimeCalibrator.from_dict(calibrator.to_dict()) == calibrator


def test_calibration_routes_on_raw_direction_and_exact_time_band() -> None:
    raw = np.asarray((-0.2, 0.0, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2))
    seconds = np.asarray((60, 89, 90, 119, 120, 179, 180, 240))
    keys = calibration_route_keys(raw, seconds)

    assert keys.tolist() == [
        "60-89:DOWN",
        "60-89:UP",
        "90-119:DOWN",
        "90-119:UP",
        "120-179:DOWN",
        "120-179:UP",
        "180-240:DOWN",
        "180-240:UP",
    ]

    cells = list(eight_calibration_cells())
    target_index = next(index for index, cell in enumerate(cells) if cell.key == "60-89:DOWN")
    cells[target_index] = replace(cells[target_index], slope=0.0, intercept=1.0)
    calibrator = DirectionTimeCalibrator(tuple(cells))
    calibrated = calibrator.calibrated_logit(raw, seconds)

    assert calibrated[0] == 1.0
    np.testing.assert_array_equal(calibrated[1:], raw[1:])


def test_frozen_feature_contract_excludes_quality_and_provider_artifacts() -> None:
    assert len(BOOK_RESIDUAL_FEATURES) == 8
    assert len(STANDARDIZED_BOOK_RESIDUAL_FEATURES) == 7
    forbidden = ("quality", "provider_age", "label", "outcome", "crossed", "stale")
    contract = (*BOOK_RESIDUAL_FEATURES, *STRICT_BOOK_RESIDUAL_INPUT_COLUMNS)
    assert not any(token in name for name in contract for token in forbidden)
