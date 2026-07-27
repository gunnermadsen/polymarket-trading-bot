from pathlib import Path

import pytest

from btc_directional_model.benchmark_config import (
    load_entry_benchmark_config,
    validate_entry_benchmark_config,
)


def repository_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-entry-benchmark-20260421-20260720.toml"
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
