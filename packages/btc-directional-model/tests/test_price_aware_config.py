from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from btc_directional_model.price_aware_config import (
    load_price_aware_benchmark_config,
    validate_price_aware_benchmark_config,
)

CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "btc-5m-directional-price-aware-economic-20260321-20260729.toml"
)


def test_price_aware_config_freezes_walk_forward_contract() -> None:
    config = load_price_aware_benchmark_config(CONFIG_PATH)
    walk_forward = config.walk_forward

    assert walk_forward.history_start == datetime(2026, 3, 21, tzinfo=UTC)
    assert walk_forward.outcome_calibration_fraction == pytest.approx(0.20)
    assert walk_forward.threshold_block_names == (
        "validation_jun02",
        "validation_jun09",
        "validation_jul03",
    )
    assert walk_forward.evaluation_block_name == "confirmation_jul14"
    assert tuple((block.name, block.start, block.end) for block in walk_forward.blocks) == (
        (
            "initial_book_history",
            datetime(2026, 4, 13, tzinfo=UTC),
            datetime(2026, 5, 26, tzinfo=UTC),
        ),
        (
            "validation_may26",
            datetime(2026, 5, 26, tzinfo=UTC),
            datetime(2026, 6, 2, tzinfo=UTC),
        ),
        (
            "validation_jun02",
            datetime(2026, 6, 2, tzinfo=UTC),
            datetime(2026, 6, 9, tzinfo=UTC),
        ),
        (
            "validation_jun09",
            datetime(2026, 6, 9, tzinfo=UTC),
            datetime(2026, 7, 3, tzinfo=UTC),
        ),
        (
            "validation_jul03",
            datetime(2026, 7, 3, tzinfo=UTC),
            datetime(2026, 7, 14, tzinfo=UTC),
        ),
        (
            "confirmation_jul14",
            datetime(2026, 7, 14, tzinfo=UTC),
            datetime(2026, 7, 29, tzinfo=UTC),
        ),
    )


def test_price_aware_config_keeps_only_value_thresholds_and_new_gates() -> None:
    config = load_price_aware_benchmark_config(CONFIG_PATH)

    assert config.model.value_thresholds == (0.0,)
    assert config.model.recency_half_life_days is None
    assert not hasattr(config.model, "return_thresholds")
    assert config.gates.target_accuracy == pytest.approx(0.92)
    assert config.gates.minimum_accuracy == pytest.approx(0.89)
    assert config.gates.minimum_wilson_lower == pytest.approx(0.87)
    assert config.gates.minimum_coverage == pytest.approx(0.20)
    assert config.gates.minimum_evaluation_trades == 200
    assert config.gates.minimum_fold_trades == 30
    assert config.gates.minimum_fold_direction_trades == 10
    assert not hasattr(config.gates, "maximum_median_vwap_5")
    assert not hasattr(config.gates, "maximum_p90_vwap_5")
    assert config.gates.minimum_profit_factor == pytest.approx(1.05)
    assert config.gates.maximum_expected_calibration_error == pytest.approx(0.08)
    assert config.gates.maximum_selected_net_bias == pytest.approx(0.03)


def test_price_aware_config_rejects_walk_forward_range_drift() -> None:
    config = load_price_aware_benchmark_config(CONFIG_PATH)
    blocks = config.walk_forward.blocks
    drifted_blocks = (
        blocks[0],
        replace(blocks[1], start=blocks[1].start + timedelta(days=1)),
        *blocks[2:],
    )
    drifted = replace(
        config,
        walk_forward=replace(config.walk_forward, blocks=drifted_blocks),
    )

    with pytest.raises(ValueError, match="chronological and contiguous"):
        validate_price_aware_benchmark_config(drifted)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("history_start", datetime(2026, 3, 22, tzinfo=UTC), "March 21"),
        ("outcome_calibration_fraction", 0.25, "must remain 0.20"),
        ("threshold_block_names", ("a", "b", "c"), "frozen dates"),
        ("evaluation_block_name", "other", "frozen confirmation"),
    ),
)
def test_price_aware_config_rejects_walk_forward_role_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_price_aware_benchmark_config(CONFIG_PATH)
    drifted = replace(
        config,
        walk_forward=replace(config.walk_forward, **{field: value}),
    )

    with pytest.raises(ValueError, match=message):
        validate_price_aware_benchmark_config(drifted)


def test_price_aware_config_rejects_undersized_direction_fold_gate() -> None:
    config = load_price_aware_benchmark_config(CONFIG_PATH)
    drifted = replace(
        config,
        gates=replace(
            config.gates,
            minimum_fold_trades=19,
            minimum_fold_direction_trades=10,
        ),
    )

    with pytest.raises(ValueError, match="accommodate both direction"):
        validate_price_aware_benchmark_config(drifted)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("recency_half_life_days", 21.0, "without recency weighting"),
        (
            "value_thresholds",
            (0.0, 0.01),
            "frozen economic grid",
        ),
    ),
)
def test_price_aware_config_rejects_model_policy_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_price_aware_benchmark_config(CONFIG_PATH)
    drifted = replace(
        config,
        model=replace(config.model, **{field: value}),
    )

    with pytest.raises(ValueError, match=message):
        validate_price_aware_benchmark_config(drifted)


def test_price_aware_config_rejects_nonfinite_profit_factor() -> None:
    config = load_price_aware_benchmark_config(CONFIG_PATH)
    drifted = replace(
        config,
        gates=replace(config.gates, minimum_profit_factor=float("nan")),
    )

    with pytest.raises(ValueError, match="finite and exceed one"):
        validate_price_aware_benchmark_config(drifted)
