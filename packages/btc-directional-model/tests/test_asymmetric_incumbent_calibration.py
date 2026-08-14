from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from btc_directional_model.asymmetric_incumbent_calibration import (
    COMPRESSED_ARM,
    DAY_BALANCED_ARM,
    INCUMBENT_ARM,
    INCUMBENT_CALIBRATION_ARM_NAMES,
    SUPPORTED_ARM,
    _calibration_weights,
    _cell_exposure_weights,
    _independent_objective,
    _prepare_target_cohort,
    _shared_compression_objective,
    clone_incumbent_with_calibration,
    fit_incumbent_calibration_arm,
    frozen_parent_probabilities,
    load_incumbent_calibration_config,
    score_incumbent_calibration_payload,
    validate_incumbent_calibration_parity,
)
from btc_directional_model.asymmetric_incumbent_replay import (
    load_frozen_asymmetric_incumbent,
    score_asymmetric_runtime_row,
)


def test_gen2_config_freezes_incumbent_policy_chronology_and_contingency() -> None:
    config = load_incumbent_calibration_config()

    assert config.process_id == "81f82de7-002b-4ac7-814b-236c6742d81c"
    assert config.feature_count == 75
    assert config.evidence_scope == "consumed_cross_day_development"
    assert config.calibration_fit.start == datetime(2026, 7, 16, tzinfo=UTC)
    assert config.calibration_fit.end == config.matched_comparison.start
    assert config.matched_comparison.end == datetime(2026, 8, 2, tzinfo=UTC)
    assert config.final_refit.start == config.calibration_fit.start
    assert config.final_refit.end == config.matched_comparison.end
    assert config.target_price_band == (0.20, 0.30)
    assert config.target_time_bands == ((1, 15), (15, 30), (30, 45), (45, 60))
    assert tuple(arm.name for arm in config.arms) == INCUMBENT_CALIBRATION_ARM_NAMES
    assert config.policy.quantity == config.policy.vwap_quantity == 5.0
    assert config.policy.maximum_depth_participation == 0.25
    assert config.policy.execution_reserve_per_share == 0.01
    assert config.policy.minimum_edge_per_share == 0.03
    assert config.policy.maximum_entry_second == 55
    assert config.incumbent_model == config.incumbent_runtime_dir / "model.json"
    assert config.conditional_estimator.maximum_total_challengers == 5
    assert config.conditional_estimator.histogram.l2_regularization == 10.0
    assert [candidate.name for candidate in config.conditional_estimator.candidates] == [
        "E1_hybrid50_h3",
        "E2_hybrid50_h3_boundary_weighted",
    ]


def test_frozen_incumbent_has_exact_target_and_non_target_cell_contract() -> None:
    config = load_incumbent_calibration_config()
    model = load_frozen_asymmetric_incumbent(config.incumbent_model)
    cells = model.payload["asymmetric_value_calibration"]["side_price_cells"]
    target = [cell for cell in cells if _is_target(cell)]

    assert len(cells) == 160
    assert len(target) == 8
    assert len(cells) - len(target) == 152
    assert all(cell["fitted"] is False for cell in target)
    assert all(cell["slope"] == 1.0 and cell["intercept"] == 0.0 for cell in target)


def test_vectorized_payload_scoring_matches_native_row_runtime() -> None:
    config = load_incumbent_calibration_config()
    model = load_frozen_asymmetric_incumbent(config.incumbent_model)
    frame = _support_frame(markets_per_day=1).head(8)
    parents = frozen_parent_probabilities(model, frame)
    actual = score_incumbent_calibration_payload(
        model,
        model.payload,
        frame,
        parent_probabilities=parents,
    )
    matrix = frame.select(*model.feature_names).to_numpy()
    expected = np.asarray(
        [
            score_asymmetric_runtime_row(
                model,
                row.tolist(),
                seconds_elapsed=int(second),
                yes_ask_vwap=float(yes_price),
                no_ask_vwap=float(no_price),
            )["probability_up"]
            for row, second, yes_price, no_price in zip(
                matrix,
                frame["seconds_elapsed"],
                frame["yes_ask_vwap_5"],
                frame["no_ask_vwap_5"],
                strict=True,
            )
        ]
    )

    assert actual == pytest.approx(expected, abs=2e-15)


def test_i0_is_exact_clone_and_supported_arm_changes_only_eight_cells() -> None:
    config = load_incumbent_calibration_config()
    model = load_frozen_asymmetric_incumbent(config.incumbent_model)
    frame = _support_frame()
    parents = _deliberately_overconfident_parents(frame)

    incumbent = fit_incumbent_calibration_arm(
        INCUMBENT_ARM,
        model,
        frame,
        config,
        parent_probabilities=parents,
    )
    supported = fit_incumbent_calibration_arm(
        SUPPORTED_ARM,
        model,
        frame,
        config,
        parent_probabilities=parents,
    )

    assert incumbent.payload == model.payload
    assert incumbent.iterations == 0
    assert all(not cell.fitted for cell in incumbent.cells)
    assert all(cell.fitted and cell.fallback is None for cell in supported.cells)
    assert all(cell.markets == 56 and cell.utc_days == 7 for cell in supported.cells)
    assert all(cell.positives == cell.negatives == 28 for cell in supported.cells)
    assert _changed_cell_count(model.payload, supported.payload) == 8
    validate_incumbent_calibration_parity(model.payload, supported.payload, config)

    candidate = clone_incumbent_with_calibration(
        model,
        supported,
        model_key="btc-5m-asymmetric-core-oracle-gen2-paper-test",
    )
    assert candidate["model_key"] == "btc-5m-asymmetric-core-oracle-gen2-paper-test"
    validate_incumbent_calibration_parity(model.payload, candidate, config)


def test_compressed_arm_uses_one_bounded_shared_slope() -> None:
    config = load_incumbent_calibration_config()
    model = load_frozen_asymmetric_incumbent(config.incumbent_model)
    frame = _support_frame()
    fit = fit_incumbent_calibration_arm(
        COMPRESSED_ARM,
        model,
        frame,
        config,
        parent_probabilities=_deliberately_overconfident_parents(frame),
    )

    slopes = {cell.slope for cell in fit.cells}
    assert len(slopes) == 1
    assert 0.50 <= next(iter(slopes)) <= 1.00
    assert _changed_cell_count(model.payload, fit.payload) == 8


def test_day_balanced_weights_and_optimizer_are_deterministic() -> None:
    config = load_incumbent_calibration_config()
    model = load_frozen_asymmetric_incumbent(config.incumbent_model)
    frame = _support_frame()
    parents = _deliberately_overconfident_parents(frame)
    first = fit_incumbent_calibration_arm(
        DAY_BALANCED_ARM,
        model,
        frame,
        config,
        parent_probabilities=parents,
    )
    second = fit_incumbent_calibration_arm(
        DAY_BALANCED_ARM,
        model,
        frame,
        config,
        parent_probabilities=parents,
    )
    reversed_fit = fit_incumbent_calibration_arm(
        DAY_BALANCED_ARM,
        model,
        frame.reverse(),
        config,
        parent_probabilities=parents[::-1],
    )

    assert first.payload_sha256 == second.payload_sha256
    assert first.payload_sha256 == reversed_fit.payload_sha256
    assert first.cells == second.cells
    ids = np.asarray(["a", "a", "b", "c", "c", "c"])
    days = np.asarray(["d1", "d1", "d1", "d2", "d2", "d2"])
    weights = _calibration_weights(ids, days, weighting="day_market_row_equal")
    assert weights == pytest.approx([0.125, 0.125, 0.25, 1 / 6, 1 / 6, 1 / 6])
    assert weights.sum() == pytest.approx(1.0)


def test_fit_fails_closed_without_required_market_or_class_support() -> None:
    config = load_incumbent_calibration_config()
    model = load_frozen_asymmetric_incumbent(config.incumbent_model)
    sparse = _support_frame(markets_per_day=7)
    with pytest.raises(RuntimeError, match="support failed"):
        fit_incumbent_calibration_arm(
            SUPPORTED_ARM,
            model,
            sparse,
            config,
            parent_probabilities=_deliberately_overconfident_parents(sparse),
        )

    one_class = _support_frame().with_columns(pl.lit(1).cast(pl.Int8).alias("label_up"))
    with pytest.raises(RuntimeError, match="single_class"):
        fit_incumbent_calibration_arm(
            SUPPORTED_ARM,
            model,
            one_class,
            config,
            parent_probabilities=_deliberately_overconfident_parents(one_class),
        )


def test_optimizer_gradients_match_centered_finite_differences() -> None:
    config = load_incumbent_calibration_config()
    frame = _support_frame()
    parents = _deliberately_overconfident_parents(frame)
    prepared = _prepare_target_cohort(frame, parents, config)
    weights = _calibration_weights(
        prepared["market_ids"],
        prepared["utc_days"],
        weighting="market_equal",
    )
    penalties = _cell_exposure_weights(prepared, weights)
    independent = np.tile(np.asarray((0.85, 0.07)), 8)
    shared = np.concatenate((np.asarray((0.85,)), np.linspace(-0.08, 0.08, 8)))

    independent_gradient = _independent_objective(
        independent,
        prepared,
        weights,
        penalties,
        config.identity_l2,
    )[1]
    shared_gradient = _shared_compression_objective(
        shared,
        prepared,
        weights,
        penalties,
        config.identity_l2,
    )[1]

    assert independent_gradient == pytest.approx(
        _centered_gradient(
            lambda values: _independent_objective(
                values,
                prepared,
                weights,
                penalties,
                config.identity_l2,
            )[0],
            independent,
        ),
        abs=2e-8,
    )
    assert shared_gradient == pytest.approx(
        _centered_gradient(
            lambda values: _shared_compression_objective(
                values,
                prepared,
                weights,
                penalties,
                config.identity_l2,
            )[0],
            shared,
        ),
        abs=2e-8,
    )


def test_parity_rejects_estimator_or_non_target_cell_change() -> None:
    config = load_incumbent_calibration_config()
    model = load_frozen_asymmetric_incumbent(config.incumbent_model)
    estimator_changed = copy.deepcopy(model.payload)
    estimator_changed["estimator"]["baseline_logit"] += 0.01
    with pytest.raises(RuntimeError, match="estimator/runtime"):
        validate_incumbent_calibration_parity(model.payload, estimator_changed, config)

    non_target_changed = copy.deepcopy(model.payload)
    non_target_changed["asymmetric_value_calibration"]["side_price_cells"][0]["intercept"] = 0.1
    with pytest.raises(RuntimeError, match="non-target"):
        validate_incumbent_calibration_parity(model.payload, non_target_changed, config)


def _support_frame(*, markets_per_day: int = 8) -> pl.DataFrame:
    model = load_frozen_asymmetric_incumbent()
    start = datetime(2026, 7, 16, tzinfo=UTC)
    seconds_by_band = ((1, 2), (15, 16), (30, 31), (45, 46))
    records: list[dict[str, object]] = []
    market_number = 0
    for day_offset in range(7):
        for _ in range(markets_per_day):
            window_start = start + timedelta(days=day_offset, minutes=5 * market_number)
            label = market_number % 2
            for yes_second, no_second in seconds_by_band:
                records.append(
                    _row(
                        model,
                        market_number,
                        window_start,
                        yes_second,
                        label,
                        yes_price=0.25,
                        no_price=0.75,
                    )
                )
                records.append(
                    _row(
                        model,
                        market_number,
                        window_start,
                        no_second,
                        label,
                        yes_price=0.75,
                        no_price=0.25,
                    )
                )
            market_number += 1
    return pl.DataFrame(records).with_columns(pl.col("label_up").cast(pl.Int8))


def _row(
    model: object,
    market_number: int,
    window_start: datetime,
    second: int,
    label: int,
    *,
    yes_price: float,
    no_price: float,
) -> dict[str, object]:
    feature_names = model.feature_names
    medians = model.payload["features"]["imputation_medians"]
    return {
        **dict(zip(feature_names, medians, strict=True)),
        "market_id": f"market-{market_number:04d}",
        "window_start": window_start,
        "seconds_elapsed": second,
        "label_up": label,
        "yes_ask_vwap_5": yes_price,
        "no_ask_vwap_5": no_price,
    }


def _deliberately_overconfident_parents(frame: pl.DataFrame) -> np.ndarray:
    return np.where(frame["seconds_elapsed"].to_numpy() % 2 == 1, 0.90, 0.10)


def _is_target(cell: dict[str, object]) -> bool:
    return (
        (cell["start_seconds"], cell["end_seconds_exclusive"])
        in {(1, 15), (15, 30), (30, 45), (45, 60)}
        and float(cell["minimum_price"]) == pytest.approx(0.20)
        and float(cell["maximum_price"]) == pytest.approx(0.30)
    )


def _changed_cell_count(before: dict[str, object], after: dict[str, object]) -> int:
    before_cells = before["asymmetric_value_calibration"]["side_price_cells"]
    after_cells = after["asymmetric_value_calibration"]["side_price_cells"]
    return sum(left != right for left, right in zip(before_cells, after_cells, strict=True))


def _centered_gradient(function: object, values: np.ndarray) -> np.ndarray:
    step = 1e-6
    gradient = np.empty_like(values)
    for index in range(values.size):
        upper = values.copy()
        lower = values.copy()
        upper[index] += step
        lower[index] -= step
        gradient[index] = (function(upper) - function(lower)) / (2.0 * step)
    return gradient
