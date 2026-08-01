from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from btc_directional_model.loss_tail_config import (
    BOUNDARY_RESIDUAL_ECONOMIC_HGB_CANDIDATE,
    LOSS_TAIL_CANDIDATE_NAMES,
    LOSS_TAIL_CONTEXT_SECONDS,
    LOSS_TAIL_DECISION_SECONDS,
    load_loss_tail_benchmark_config,
    validate_loss_tail_benchmark_config,
)

CONFIG_PATH = (
    Path(__file__).parents[1] / "configs" / "btc-5m-directional-loss-tail-20260413-20260729.toml"
)


def test_loss_tail_config_freezes_data_and_walk_forward_contract() -> None:
    config = load_loss_tail_benchmark_config(CONFIG_PATH)

    assert config.walk_forward.history_start == datetime(2026, 3, 21, tzinfo=UTC)
    assert config.walk_forward.blocks[0].start == datetime(2026, 4, 13, tzinfo=UTC)
    assert config.walk_forward.blocks[-1].end == datetime(2026, 7, 29, tzinfo=UTC)
    assert tuple(
        (
            fold.fit_block_names,
            fold.calibration_block_name,
            fold.evaluation_block_name,
        )
        for fold in config.walk_forward.folds
    ) == (
        (
            ("book_history_apr13", "calibration_may26"),
            "calibration_jun02",
            "evaluation_jun09",
        ),
        (
            ("book_history_apr13", "calibration_may26", "calibration_jun02"),
            "evaluation_jun09",
            "evaluation_jul03",
        ),
        (
            (
                "book_history_apr13",
                "calibration_may26",
                "calibration_jun02",
                "evaluation_jun09",
            ),
            "evaluation_jul03",
            "confirmation_jul14",
        ),
    )
    assert config.data.decision_seconds == LOSS_TAIL_DECISION_SECONDS
    assert config.data.context_seconds == LOSS_TAIL_CONTEXT_SECONDS
    assert config.data.quantity == pytest.approx(5.0)
    assert config.data.strict_book_required
    assert not config.data.allow_book_imputation
    assert not config.data.allow_missingness_features
    assert not config.data.require_complete_paths_for_strategy
    assert config.data.allow_point_qualified_row_metrics


def test_loss_tail_config_freezes_candidate_and_resource_contract() -> None:
    config = load_loss_tail_benchmark_config(CONFIG_PATH)

    assert config.candidate_names == LOSS_TAIL_CANDIDATE_NAMES
    assert config.candidate_names == (BOUNDARY_RESIDUAL_ECONOMIC_HGB_CANDIDATE,)
    assert config.features.direct_feature_count == 124
    assert tuple(candidate.estimator for candidate in config.candidates) == (
        "histogram_gradient_boosting",
    )
    correctness = config.candidates[0]
    assert correctness.direction_policy == "boundary_locked_economic_action"
    assert correctness.requires_causal_oof_sources
    assert all(candidate.calibration_weighting == "unweighted" for candidate in config.candidates)
    assert config.weighting.normalization == "one_proposal_equal_base"
    assert config.weighting.minimum_multiplier == pytest.approx(1.0)
    assert config.weighting.maximum_multiplier == pytest.approx(10.0)
    assert config.weighting.denominator_floor == pytest.approx(0.05)
    assert config.weighting.fee_inclusive_debit
    assert (config.resources.workers, config.resources.threads_per_worker) == (1, 3)
    assert config.resources.memory_limit_gib == 8


def test_loss_tail_config_freezes_promotion_gates() -> None:
    gates = load_loss_tail_benchmark_config(CONFIG_PATH).gates

    assert gates.minimum_selected_accuracy == pytest.approx(0.89)
    assert gates.minimum_wilson_lower == pytest.approx(0.87)
    assert gates.maximum_control_accuracy_regression == pytest.approx(0.005)
    assert gates.maximum_expected_calibration_error == pytest.approx(0.05)
    assert gates.maximum_wins_per_average_loss == pytest.approx(6.34)
    assert gates.maximum_gross_loss_profit_ratio == pytest.approx(0.712)
    assert gates.minimum_profit_factor == pytest.approx(1.265)
    assert gates.minimum_pnl_per_all_core_market == pytest.approx(0.02044)
    assert gates.minimum_total_pnl == pytest.approx(283.15)
    assert gates.maximum_mean_selected_price == pytest.approx(0.8697)
    assert gates.minimum_book_qualified_coverage == pytest.approx(0.371)
    assert gates.minimum_pooled_trades == 500
    assert gates.minimum_confirmation_trades == 200
    assert gates.required_nonnegative_expectancy_folds == 3
    assert gates.minimum_loss_improvement_folds == 2
    assert gates.minimum_gross_loss_reduction == pytest.approx(0.30)
    assert gates.minimum_gross_profit_retention == pytest.approx(0.70)
    assert gates.minimum_high_debit_loss_recall == pytest.approx(0.30)
    assert not gates.allow_fallback_winner


def test_loss_tail_config_rejects_walk_forward_drift() -> None:
    config = load_loss_tail_benchmark_config(CONFIG_PATH)
    blocks = config.walk_forward.blocks
    drifted = replace(
        config,
        walk_forward=replace(
            config.walk_forward,
            blocks=(
                blocks[0],
                replace(blocks[1], start=blocks[1].start + timedelta(days=1)),
                *blocks[2:],
            ),
        ),
    )

    with pytest.raises(ValueError, match="chronological and contiguous"):
        validate_loss_tail_benchmark_config(drifted)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("decision_start_seconds", 65, "60-240"),
        ("context_start_seconds", 60, "55-240"),
        ("quantity", 10.0, "five-share"),
        ("allow_book_imputation", True, "remain strict"),
        ("allow_missingness_features", True, "remain strict"),
    ),
)
def test_loss_tail_config_rejects_data_contract_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_loss_tail_benchmark_config(CONFIG_PATH)
    drifted = replace(config, data=replace(config.data, **{field: value}))

    with pytest.raises(ValueError, match=message):
        validate_loss_tail_benchmark_config(drifted)


def test_loss_tail_config_rejects_candidate_drift() -> None:
    config = load_loss_tail_benchmark_config(CONFIG_PATH)
    candidates = config.candidates
    drifted = replace(
        config,
        candidates=(replace(candidates[0], estimator="logistic_regression"), *candidates[1:]),
    )

    with pytest.raises(ValueError, match="candidate contract"):
        validate_loss_tail_benchmark_config(drifted)


def test_loss_tail_config_rejects_resource_drift() -> None:
    config = load_loss_tail_benchmark_config(CONFIG_PATH)
    drifted = replace(
        config,
        resources=replace(config.resources, threads_per_worker=4),
    )

    with pytest.raises(ValueError, match="one worker with three threads and 8 GiB"):
        validate_loss_tail_benchmark_config(drifted)


def test_loss_tail_config_rejects_fallback_winner() -> None:
    config = load_loss_tail_benchmark_config(CONFIG_PATH)
    drifted = replace(config, gates=replace(config.gates, allow_fallback_winner=True))

    with pytest.raises(ValueError, match="no-fallback"):
        validate_loss_tail_benchmark_config(drifted)
