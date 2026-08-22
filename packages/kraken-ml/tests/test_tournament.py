from __future__ import annotations

import numpy as np

from kraken_ml.tournament import (
    TournamentCandidate,
    _actions,
    _generation_one,
    _generation_three,
    _generation_two,
    _holm,
    _model_parameters,
)


def test_generation_one_has_unique_model_horizon_screen() -> None:
    candidates = _generation_one()
    assert len(candidates) == 28
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert {candidate.horizon_bars for candidate in candidates} == {2, 4, 8, 16, 32}
    assert {candidate.model for candidate in candidates} == {
        "ridge",
        "elastic_net",
        "histogram",
        "extra_trees",
        "lightgbm",
    }


def test_signed_actions_respect_hurdle_and_no_trade() -> None:
    predictions = np.asarray([-8.0, -2.0, 0.0, 3.0, 9.0])
    assert _actions(predictions, 3.0).tolist() == [-1, 0, 0, 1, 1]
    assert _actions(predictions, 0.0, no_trade=True).tolist() == [0, 0, 0, 0, 0]


def test_parameter_variants_are_deterministic() -> None:
    candidate = TournamentCandidate("x", 2, "lightgbm", 4, "positioning", parameter_variant=2)
    assert _model_parameters(candidate, seed=11) == _model_parameters(candidate, seed=11)
    assert _model_parameters(candidate, seed=11)["random_state"] == 11


def test_failed_generation_pivots_then_falsifies() -> None:
    failed = {
        "candidate_id": "failed",
        "model": "lightgbm",
        "horizon": "8h",
        "horizon_bars": 32,
        "feature_set": "positioning",
        "target_variant": "vol_scaled",
        "parameter_variant": 1,
        "status": "failed",
    }
    generation_two, decision_two = _generation_two([failed])
    assert len(generation_two) == 9
    assert "pivot" in decision_two
    generation_three, decision_three = _generation_three([failed])
    assert len(generation_three) == 4
    assert "falsification" in decision_three


def test_generation_three_deduplicates_parent_flow_feature() -> None:
    parent = {
        "candidate_id": "flow-parent",
        "model": "ridge",
        "horizon": "4h",
        "horizon_bars": 16,
        "feature_set": "flow",
        "target_variant": "vol_scaled",
        "parameter_variant": 1,
        "status": "predictive_only",
    }
    candidates, decision = _generation_three([parent])
    assert len(candidates) == 2
    assert len({candidate.candidate_id for candidate in candidates}) == 2
    assert "harden" in decision


def test_holm_adjustment_is_monotonic() -> None:
    rows = [
        {"raw_positive_p_value": 0.01},
        {"raw_positive_p_value": 0.02},
        {"raw_positive_p_value": 0.20},
    ]
    _holm(rows)
    adjusted = [row["holm_adjusted_p_value"] for row in rows]
    assert adjusted == [0.03, 0.04, 0.20]
