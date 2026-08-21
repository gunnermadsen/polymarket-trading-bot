import numpy as np

from nyc_temperature_model.challenger_tournament import (
    ENSEMBLE_COMPONENTS,
    MODEL_CANDIDATES,
    SUPPORT,
    _bucket_probability,
    _fit_ensemble_weights,
    _probability_lower,
    _rounded_samples_probability,
    _select_champion,
)


def test_sample_distribution_is_normalized_and_bucket_partition_is_complete():
    probabilities = _rounded_samples_probability(np.asarray([69.4, 69.6, 70.2, 71.0]))

    assert probabilities.shape == SUPPORT.shape
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.isclose(
        _bucket_probability(probabilities, None, 69)
        + _bucket_probability(probabilities, 70, 70)
        + _bucket_probability(probabilities, 71, None),
        1.0,
    )


def test_probability_lower_is_conservative_for_both_sides():
    for probability in (0.08, 0.24, 0.50, 0.91):
        yes_lower = _probability_lower(probability)
        no_lower = _probability_lower(1.0 - probability)

        assert 0.0 <= yes_lower < probability
        assert 0.0 <= no_lower < 1.0 - probability


def test_convex_ensemble_weights_are_bounded_normalized_and_repeatable():
    row_count = 30
    target = np.resize(np.arange(55, 85), row_count).astype(float)
    target_indices = target.astype(int) - int(SUPPORT[0])
    probability_matrices = {}
    for component_index, name in enumerate(ENSEMBLE_COMPONENTS):
        matrix = np.full((row_count, len(SUPPORT)), 1e-5)
        offset = 0 if component_index == 0 else component_index + 2
        matrix[np.arange(row_count), np.clip(target_indices + offset, 0, len(SUPPORT) - 1)] = 1.0
        matrix /= matrix.sum(axis=1, keepdims=True)
        probability_matrices[name] = matrix

    first = _fit_ensemble_weights(probability_matrices, target)
    second = _fit_ensemble_weights(probability_matrices, target)

    assert np.all(first >= 0.0)
    assert np.all(first <= 1.0)
    assert np.isclose(first.sum(), 1.0)
    assert np.allclose(first, second)
    assert first[0] > 0.99


def test_champion_selection_uses_forecast_loss_not_pnl():
    row_count = 100
    metrics = {}
    losses = {}
    for hour in (0, 12):
        metrics[hour] = {}
        losses[hour] = {}
        for candidate in MODEL_CANDIDATES:
            candidate_loss = 0.8 if candidate == "analog_ensemble" else 1.0
            metrics[hour][candidate] = {
                "ranked_probability_score": candidate_loss,
                "rounded_temperature_log_loss": candidate_loss,
            }
            losses[hour][candidate] = np.full(row_count, candidate_loss)

    champion, decision = _select_champion(metrics, losses)

    assert champion == "analog_ensemble"
    assert decision["selected_champion"] == "analog_ensemble"
