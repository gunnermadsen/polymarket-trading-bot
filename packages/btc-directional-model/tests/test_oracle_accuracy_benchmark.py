from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from btc_directional_model.core_benchmark import (
    BenchmarkEvidence,
    CandidatePolicy,
    benchmark_predictions,
)
from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_features import (
    CORE_MATURE_REVERSAL_ENRICHED_FEATURES,
    CORE_MATURE_REVERSAL_ORACLE_FEATURES,
    CORE_ORACLE_FEATURES,
)
from btc_directional_model.persistence_benchmark import (
    CANDIDATE_PROFILES,
    _assert_matched_oracle_candidate_rows,
    _candidate_eligible_frame,
    _candidate_spec,
)
from btc_directional_model.persistence_config import (
    MATURE_REVERSAL_ORACLE_ACCURACY_CANDIDATES,
    MATURE_REVERSAL_ORACLE_ACCURACY_PROFILE,
    MATURE_REVERSAL_ORACLE_CANDIDATE,
    MATURE_REVERSAL_ORACLE_CONTROL_CANDIDATE,
    ORACLE_EARLY_ENTRY_FIXED_EVALUATION_SECONDS,
    _parse_fixed_evaluation_seconds,
    _validate_oracle_accuracy_core_contract,
    load_persistence_benchmark_config,
    validate_persistence_benchmark_config,
)


def oracle_accuracy_config_path() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-mature-reversal-oracle-accuracy-20260321-20260729.toml"
    )


def test_oracle_accuracy_config_freezes_recency_matched_contract() -> None:
    config = load_persistence_benchmark_config(
        oracle_accuracy_config_path()
    )

    assert config.profile == MATURE_REVERSAL_ORACLE_ACCURACY_PROFILE
    assert config.control_candidate == (
        MATURE_REVERSAL_ORACLE_CONTROL_CANDIDATE
    )
    assert config.candidate_names == (
        MATURE_REVERSAL_ORACLE_ACCURACY_CANDIDATES
    )
    assert config.fixed_evaluation_seconds == (
        ORACLE_EARLY_ENTRY_FIXED_EVALUATION_SECONDS
    )
    assert config.quantity == 5.0
    assert config.early_cutoff_second == 120
    assert config.hard_confidence_floor == 0.95
    assert config.minimum_hard_confident_error_count_reduction == 1
    assert config.minimum_accuracy_uplift == 0.001
    assert config.minimum_balanced_accuracy_uplift == 0.001
    assert config.minimum_wilson_lower_uplift == 0.001
    assert config.row_weight_schedules == ()
    assert config.execution_evidence.name == "execution-evidence"
    assert config.runs.name == (
        "btc-mature-reversal-oracle-accuracy-20260321-20260729"
    )


def test_oracle_accuracy_candidate_specs_differ_only_by_oracle_features() -> None:
    config = load_persistence_benchmark_config(
        oracle_accuracy_config_path()
    )
    control_profile = CANDIDATE_PROFILES[
        MATURE_REVERSAL_ORACLE_CONTROL_CANDIDATE
    ]
    challenger_profile = CANDIDATE_PROFILES[
        MATURE_REVERSAL_ORACLE_CANDIDATE
    ]
    control = _candidate_spec(control_profile, config)
    challenger = _candidate_spec(challenger_profile, config)

    assert control.feature_names == tuple(
        CORE_MATURE_REVERSAL_ENRICHED_FEATURES
    )
    assert challenger.feature_names == tuple(
        CORE_MATURE_REVERSAL_ORACLE_FEATURES
    )
    assert len(control.feature_names) == 71
    assert len(challenger.feature_names) == 82
    assert challenger.feature_names[: len(control.feature_names)] == (
        control.feature_names
    )
    assert challenger.feature_names[len(control.feature_names) :] == tuple(
        CORE_ORACLE_FEATURES
    )
    assert control.recency_half_life_days == 28.0
    assert challenger.recency_half_life_days == 28.0
    assert control.row_weight_schedule == challenger.row_weight_schedule
    assert control_profile.eligibility_kind == "oracle"
    assert challenger_profile.eligibility_kind == "oracle"


def test_oracle_accuracy_eligibility_is_matched_and_not_path_conditioned() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["oracle-ready-zero-path", "oracle-missing"],
            "oracle_model_eligible": [True, False],
            "btc_path_from_window_open_bps": [0.0, 10.0],
        }
    )

    for candidate in MATURE_REVERSAL_ORACLE_ACCURACY_CANDIDATES:
        eligible = _candidate_eligible_frame(
            frame,
            CANDIDATE_PROFILES[candidate],
        )
        assert eligible["market_id"].to_list() == [
            "oracle-ready-zero-path"
        ]


def test_oracle_accuracy_benchmark_uses_all_five_exact_checkpoints() -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    seconds = ORACLE_EARLY_ENTRY_FIXED_EVALUATION_SECONDS
    market_ids = [f"market-{second}" for second in seconds]
    base = pl.DataFrame(
        {
            "candidate": [
                MATURE_REVERSAL_ORACLE_CONTROL_CANDIDATE
            ]
            * len(seconds),
            "market_id": market_ids,
            "window_start": [start] * len(seconds),
            "observed_at": [
                start + timedelta(seconds=second) for second in seconds
            ],
            "seconds_elapsed": seconds,
            "label_up": [1, 0, 1, 0, 1],
            "predicted_up": [1, 0, 1, 0, 1],
            "probability_up": [0.9, 0.1, 0.9, 0.1, 0.9],
            "confidence": [0.9] * len(seconds),
            "correct": [True] * len(seconds),
        }
    )
    challenger = base.with_columns(
        pl.lit(MATURE_REVERSAL_ORACLE_CANDIDATE).alias("candidate")
    )
    policy = CandidatePolicy(
        confidence_threshold=0.5,
        deployment_compatible=False,
    )

    benchmark = benchmark_predictions(
        {
            MATURE_REVERSAL_ORACLE_CONTROL_CANDIDATE: base,
            MATURE_REVERSAL_ORACLE_CANDIDATE: challenger,
        },
        policies={
            MATURE_REVERSAL_ORACLE_CONTROL_CANDIDATE: policy,
            MATURE_REVERSAL_ORACLE_CANDIDATE: policy,
        },
        control_candidate=MATURE_REVERSAL_ORACLE_CONTROL_CANDIDATE,
        evidence=BenchmarkEvidence(
            label="focused oracle checkpoint test",
            kind="development",
            independent=False,
        ),
        minimum_samples=1,
        minimum_executable_samples=1,
        fixed_checkpoints=seconds,
    )

    assert benchmark["fixed_checkpoints"] == list(seconds)
    comparison = benchmark["common_comparisons"][
        MATURE_REVERSAL_ORACLE_CANDIDATE
    ]["checkpoints"]
    assert [row["seconds_elapsed"] for row in comparison] == list(seconds)
    assert [row["common_markets"] for row in comparison] == [1] * len(
        seconds
    )


def test_oracle_accuracy_config_rejects_legacy_fixed_checkpoints(
) -> None:
    config = load_persistence_benchmark_config(
        oracle_accuracy_config_path()
    )

    with pytest.raises(
        ValueError,
        match="fixed evaluation checkpoints",
    ):
        validate_persistence_benchmark_config(
            replace(
                config,
                fixed_evaluation_seconds=(60, 90, 120, 180, 240),
            )
        )


def test_oracle_accuracy_rejects_fractional_fixed_checkpoints() -> None:
    with pytest.raises(ValueError, match="only integers"):
        _parse_fixed_evaluation_seconds([120.9, 125, 130, 135, 140])


def test_oracle_accuracy_freezes_core_gates_and_histogram_search() -> None:
    config = load_persistence_benchmark_config(oracle_accuracy_config_path())
    core = load_core_config(config.core_config)

    with pytest.raises(ValueError, match="absolute accuracy gates"):
        _validate_oracle_accuracy_core_contract(
            replace(
                core,
                gates=replace(core.gates, target_accuracy=0.80),
            )
        )
    with pytest.raises(ValueError, match="histogram search"):
        _validate_oracle_accuracy_core_contract(
            replace(
                core,
                model=replace(
                    core.model,
                    histogram_candidates=core.model.histogram_candidates[:-1]
                    + (
                        replace(
                            core.model.histogram_candidates[-1],
                            max_iter=221,
                        ),
                    ),
                ),
            )
        )


def test_oracle_accuracy_requires_exact_matched_out_of_fold_rows() -> None:
    config = load_persistence_benchmark_config(oracle_accuracy_config_path())
    start = datetime(2026, 7, 1, tzinfo=UTC)
    rows = pl.DataFrame(
        {
            "fold_index": [0, 0],
            "market_id": ["market-a", "market-b"],
            "window_start": [start, start + timedelta(minutes=5)],
            "observed_at": [
                start + timedelta(seconds=120),
                start + timedelta(minutes=7),
            ],
            "seconds_elapsed": [120, 120],
            "label_up": [1, 0],
        }
    )
    results = {
        candidate: {"scored_rows": rows}
        for candidate in MATURE_REVERSAL_ORACLE_ACCURACY_CANDIDATES
    }
    _assert_matched_oracle_candidate_rows(results, config)

    results[MATURE_REVERSAL_ORACLE_CANDIDATE] = {
        "scored_rows": rows.with_columns(
            pl.when(pl.col("market_id") == "market-b")
            .then(pl.lit("market-c"))
            .otherwise(pl.col("market_id"))
            .alias("market_id")
        )
    }
    with pytest.raises(RuntimeError, match="exact oracle control"):
        _assert_matched_oracle_candidate_rows(results, config)
