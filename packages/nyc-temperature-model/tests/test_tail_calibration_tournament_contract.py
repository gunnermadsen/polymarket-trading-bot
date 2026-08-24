from datetime import date

import numpy as np

from nyc_temperature_model.challenger_tournament import SUPPORT
from nyc_temperature_model.tail_calibration_tournament import (
    _gaussian_probabilities,
    _policy_candidates,
    _temperature_transform,
    _validate_image_id,
    _validate_revision,
)


def test_gaussian_distribution_is_normalized_and_moves_with_point():
    probabilities = _gaussian_probabilities(np.asarray([60.0, 80.0]), np.asarray([1.5, 2.0]))

    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert SUPPORT[np.argmax(probabilities[0])] == 60
    assert SUPPORT[np.argmax(probabilities[1])] == 80


def test_temperature_transform_preserves_normalization_and_order():
    source = np.asarray([[0.1, 0.2, 0.7]])
    transformed = _temperature_transform(source, 1.5)

    assert np.allclose(transformed.sum(axis=1), 1.0)
    assert transformed[0, 2] > source[0, 2]
    assert np.array_equal(np.argsort(source), np.argsort(transformed))


def test_selection_adjustment_is_more_conservative_for_larger_family():
    base = {
        "event_date": date(2026, 6, 1),
        "decision_hour_local": 0,
        "probability": 0.24,
        "probability_lower": 0.20,
        "all_in_cost_per_share": 0.12,
        "executable": True,
        "rejection_reasons": [],
    }
    adjusted = _policy_candidates([dict(base) for _ in range(8)], "selection_adjusted")

    assert all(row["selection_family_size"] == 8 for row in adjusted)
    assert all(row["probability_lower"] < base["probability_lower"] for row in adjusted)
    assert all(row["robust_edge_per_share"] < 0.08 for row in adjusted)


def test_git_revision_must_be_exact_lowercase_sha1():
    revision = "a" * 40
    assert _validate_revision(revision) == revision
    assert _validate_image_id("sha256:" + "b" * 64) == "sha256:" + "b" * 64
