from __future__ import annotations

import numpy as np

from btc_directional_model.latent_twap_state_space import (
    CANDIDATES,
    ProbabilityCalibrator,
    StateSpaceParameters,
    filter_sequence,
    fit_chronological_mle,
    predetermined_configurations,
)


def _training_sequences(sensor_count: int) -> tuple[list[np.ndarray], np.ndarray]:
    targets = np.array([-12.0, -8.0, -4.0, 4.0, 8.0, 12.0])
    sequences = []
    for target in targets:
        path = np.linspace(0.15 * target, 0.85 * target, 8)
        sensors = [path + offset for offset in np.linspace(-0.2, 0.2, sensor_count)]
        sequences.append(np.column_stack(sensors))
    return sequences, targets


def test_search_budget_is_exactly_twelve_predetermined_rows() -> None:
    rows = predetermined_configurations()

    assert len(rows) == 12
    assert len({row.identifier for row in rows}) == 12


def test_filter_is_causal_and_does_not_read_future_observations() -> None:
    candidate = CANDIDATES[0]
    sequences, targets = _training_sequences(len(candidate.sensors))
    parameters = fit_chronological_mle(
        sequences, targets, candidate, predetermined_configurations()[0]
    )
    original = sequences[-1].copy()
    changed_future = original.copy()
    changed_future[5:] *= -10.0

    left = filter_sequence(original, candidate, parameters)
    right = filter_sequence(changed_future, candidate, parameters)

    np.testing.assert_allclose(left.expected_margin_bps[:5], right.expected_margin_bps[:5])
    np.testing.assert_allclose(left.probability_up[:5], right.probability_up[:5])


def test_regime_probabilities_are_normalized_and_serialization_round_trips() -> None:
    candidate = CANDIDATES[3]
    sequences, targets = _training_sequences(len(candidate.sensors))
    fitted = fit_chronological_mle(
        sequences, targets, candidate, predetermined_configurations()[3]
    )

    restored = StateSpaceParameters.from_dict(fitted.to_dict())
    result = filter_sequence(sequences[-1], candidate, restored)

    np.testing.assert_allclose(result.regime_probabilities.sum(axis=1), 1.0)
    assert np.all(result.regime_probabilities >= 0.0)
    assert np.all(result.regime_probabilities <= 1.0)


def test_probability_calibration_is_direction_symmetric() -> None:
    calibrator = ProbabilityCalibrator(temperature_slope=1.7, support_markets=100, support_rows=2500)
    values = np.array([0.1, 0.25, 0.6, 0.9])

    np.testing.assert_allclose(
        calibrator.transform(1.0 - values),
        1.0 - calibrator.transform(values),
        atol=1e-12,
    )
