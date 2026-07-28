from pathlib import Path

import pytest

from btc_directional_model.benchmark_config import (
    CORE_ONLY_REUSE_DIAGNOSTICS_MODE,
    load_entry_benchmark_config,
    validate_entry_benchmark_config,
)


def repository_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-entry-benchmark-20260421-20260720.toml"
    )


def early_entry_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-early-entry-benchmark-20260421-20260720.toml"
    )


def test_repository_benchmark_contract_is_frozen_and_non_independent() -> None:
    config = load_entry_benchmark_config(repository_config())

    assert config.benchmark.fixed_evaluation_seconds == (60, 90, 120, 180, 240)
    assert config.benchmark.quantity == 5.0
    assert not config.benchmark.evaluation_is_independent
    assert config.execution.range_start == config.book_split.fit_start
    assert config.execution.range_end == config.book_split.policy_end
    assert config.gates.maximum_median_entry_seconds_regression == -1.0


def test_consumed_range_cannot_be_mislabeled_independent() -> None:
    config = load_entry_benchmark_config(repository_config())
    object.__setattr__(config.benchmark, "evaluation_is_independent", True)

    with pytest.raises(ValueError, match="must remain marked non-independent"):
        validate_entry_benchmark_config(config)


def test_core_only_benchmark_freezes_candidates_paths_and_gates() -> None:
    config = load_entry_benchmark_config(early_entry_config())

    assert config.benchmark.mode == CORE_ONLY_REUSE_DIAGNOSTICS_MODE
    assert config.benchmark.candidate_names == (
        "histogram_enriched",
        "histogram_early_weighted",
        "histogram_early_weighted_moderate",
        "histogram_early_90_120",
    )
    assert config.gates.maximum_accuracy_regression == 0.0
    assert config.gates.maximum_balanced_accuracy_regression == 0.0
    assert config.gates.maximum_direction_recall_regression == 0.0
    assert config.gates.maximum_median_entry_seconds_regression == -5.0
    assert config.gates.minimum_executable_markets == 500
    assert config.gates.minimum_common_time_markets == 500
    assert config.execution.output_dir.name == (
        "btc-execution-evidence-20260527-20260611"
    )
    assert config.paths.runs.name == (
        "btc-core-early-entry-benchmark-20260421-20260720"
    )
    assert config.prior_diagnostics is not None
    assert config.prior_diagnostics.run_id == "20260727T223611Z"
    assert config.prior_diagnostics.sha256 == (
        "c0132d0985c96d7c9c3c9a86c2daba3b75e2e2bce11c27cdccad848277975ba6"
    )
