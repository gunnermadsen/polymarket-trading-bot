from __future__ import annotations

import pytest

from btc_directional_model.entry_benchmark import (
    _require_exact_policy_threshold,
)


def test_exact_fold_threshold_evidence_accepts_matching_policy() -> None:
    _require_exact_policy_threshold(
        "candidate",
        {
            "folds": [
                {"confidence_threshold": 0.89},
                {"confidence_threshold": 0.89},
            ]
        },
        0.89,
    )


def test_exact_fold_threshold_evidence_rejects_different_policy() -> None:
    with pytest.raises(RuntimeError, match="full five-second scores are required"):
        _require_exact_policy_threshold(
            "candidate",
            {
                "folds": [
                    {"confidence_threshold": 0.89},
                    {"confidence_threshold": 0.90},
                ]
            },
            0.89,
        )
