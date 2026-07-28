from __future__ import annotations

import inspect
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from btc_directional_model.benchmark_config import (
    PriorDiagnosticsConfig,
    load_entry_benchmark_config,
)
from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_extract import file_sha256
from btc_directional_model.core_training import develop_core_models
from btc_directional_model.entry_benchmark import (
    EARLY_ENTRY_CORE_CANDIDATES,
    _chronological_threshold_frame,
    _load_prior_diagnostics,
    _require_exact_policy_threshold,
    _training_finalist_rank,
    _training_selection,
    _validate_core_only_contract,
)


def early_entry_benchmark_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-early-entry-benchmark-20260421-20260720.toml"
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


def test_core_only_contract_and_threshold_lineage_are_chronological() -> None:
    config = load_entry_benchmark_config(early_entry_benchmark_config())
    core_config = load_core_config(config.benchmark.core_config)
    folds = []
    for fold_index, (validation_start, validation_end) in enumerate(
        core_config.split.validation_windows
    ):
        folds.append(
            {
                "fold_index": fold_index,
                "confidence_threshold": 0.87 + 0.01 * fold_index,
                "policy_range_end": (
                    validation_start - timedelta(minutes=5)
                ).isoformat(),
                "validation_range_start": validation_start.isoformat(),
                "validation_range_end": validation_end.isoformat(),
            }
        )
    metrics = {
        "candidates": {
            candidate: {"folds": folds}
            for candidate in EARLY_ENTRY_CORE_CANDIDATES
        }
    }

    _validate_core_only_contract(config, core_config)
    thresholds = _chronological_threshold_frame(
        metrics,
        core_config,
        config.benchmark.candidate_names,
    )

    assert thresholds.height == 20
    assert set(thresholds["selected_confidence_threshold"]) == {
        0.87,
        0.88,
        0.89,
        0.90,
        0.91,
    }


def test_prior_diagnostics_fail_closed_on_sha_and_load_when_pinned(
    tmp_path: Path,
) -> None:
    config = load_entry_benchmark_config(early_entry_benchmark_config())
    record_path = tmp_path / "benchmark.json"
    record_path.write_text(
        json.dumps(
            {
                "run_id": "20260727T223611Z",
                "training_evidence": {
                    "preopen_candidate": {"candidate": "preopen"},
                    "strict_book_candidate": {"candidate": "book"},
                },
            },
            sort_keys=True,
        )
    )
    pinned = replace(
        config,
        prior_diagnostics=PriorDiagnosticsConfig(
            record=record_path,
            sha256=file_sha256(record_path),
            run_id="20260727T223611Z",
        ),
    )

    assert _load_prior_diagnostics(pinned)["run_id"] == "20260727T223611Z"

    invalid = replace(
        pinned,
        prior_diagnostics=replace(
            pinned.prior_diagnostics,
            sha256="0" * 64,
        ),
    )
    with pytest.raises(RuntimeError, match="sha256"):
        _load_prior_diagnostics(invalid)


def test_execution_benchmark_can_disable_legacy_auto_freeze() -> None:
    parameters = inspect.signature(develop_core_models).parameters

    assert parameters["freeze_if_ready"].default is True
    assert parameters["freeze_if_ready"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["fit_final_candidate"].default is True
    assert parameters["fit_final_candidate"].kind is inspect.Parameter.KEYWORD_ONLY


def test_training_selection_enforces_five_folds_and_bootstrap() -> None:
    config = load_entry_benchmark_config(early_entry_benchmark_config())
    core_config = load_core_config(config.benchmark.core_config)
    benchmark_candidates = {}
    core_candidates = {}
    for index, name in enumerate(config.benchmark.candidate_names[1:]):
        benchmark_candidates[name] = {
            "advance": {
                "checks": [
                    {"name": "minimum accuracy", "passed": True},
                    {
                        "name": "native inference p99 is within budget",
                        "passed": False,
                    },
                    {
                        "name": "runtime model size is within budget",
                        "passed": False,
                    },
                ]
            },
            "own_policy": {
                "wilson_lower_95": 0.88 + index * 0.001,
                "accuracy": 0.89,
                "balanced_accuracy": 0.89,
                "coverage": 0.60,
                "median_seconds_elapsed": 100 - index * 5,
                "execution": {
                    "realized_net_expectancy_per_trade": 0.25,
                    "mean_direct_edge_per_share": 0.05,
                },
            },
        }
        core_candidates[name] = {
            "passed_development": True,
            "nonnegative_uplift_folds": 5,
            "bootstrap": {"lower_95": 0.0},
        }
    core_candidates["histogram_early_weighted_moderate"][
        "nonnegative_uplift_folds"
    ] = 4
    core_candidates["histogram_early_90_120"]["bootstrap"]["lower_95"] = -0.001

    selection = _training_selection(
        {"candidates": benchmark_candidates},
        {"candidates": core_candidates},
        config,
        core_config,
    )

    assert selection["finalist"] == "histogram_early_weighted"
    assert selection["passing_candidates"] == ["histogram_early_weighted"]
    assert not selection["candidates"][
        "histogram_early_weighted_moderate"
    ]["passed"]
    assert not selection["candidates"]["histogram_early_90_120"]["passed"]
    assert selection["runtime_freeze_created"] is False


def test_training_finalist_rank_is_coverage_then_timing_then_economics() -> None:
    def metrics(
        *,
        coverage: float,
        median: float,
        net: float,
        direct_edge: float,
        accuracy: float,
    ) -> dict:
        return {
            "coverage": coverage,
            "median_seconds_elapsed": median,
            "wilson_lower_95": accuracy - 0.01,
            "accuracy": accuracy,
            "balanced_accuracy": accuracy,
            "execution": {
                "realized_net_expectancy_per_trade": net,
                "mean_direct_edge_per_share": direct_edge,
            },
        }

    high_coverage = metrics(
        coverage=0.61,
        median=120,
        net=0.10,
        direct_edge=0.01,
        accuracy=0.88,
    )
    lower_coverage_high_quality = metrics(
        coverage=0.60,
        median=80,
        net=0.50,
        direct_edge=0.10,
        accuracy=0.95,
    )
    earlier = metrics(
        coverage=0.61,
        median=115,
        net=0.05,
        direct_edge=0.01,
        accuracy=0.88,
    )
    better_economics = metrics(
        coverage=0.61,
        median=115,
        net=0.20,
        direct_edge=0.02,
        accuracy=0.87,
    )

    assert _training_finalist_rank(high_coverage) > _training_finalist_rank(
        lower_coverage_high_quality
    )
    assert _training_finalist_rank(earlier) > _training_finalist_rank(
        high_coverage
    )
    assert _training_finalist_rank(better_economics) > _training_finalist_rank(
        earlier
    )
