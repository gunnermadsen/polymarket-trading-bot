from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from btc_directional_model.asymmetric_book_admission_evaluation import (
    BookAdmissionSelectionThresholds,
    select_asymmetric_book_admission_challenger,
)
from btc_directional_model.asymmetric_incumbent_evaluation import (
    PROBABILITY_SELECTION_FORBIDDEN_COLUMNS,
    simultaneous_paired_probability_bootstrap,
)

SelectedProbability = float | Callable[[int, str, int, int], float]


def _probability_frame(
    candidate_id: str, selected_probability: SelectedProbability
) -> pl.DataFrame:
    start = datetime(2026, 7, 21, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    market_index = 0
    for utc_day in range(10):
        for side in ("YES", "NO"):
            for second in (5, 20, 35, 50):
                for outcome_index, selected_label in enumerate((1, 1, 1, 0)):
                    probability = (
                        selected_probability(utc_day, side, second, outcome_index)
                        if callable(selected_probability)
                        else selected_probability
                    )
                    window_start = start + timedelta(
                        days=utc_day,
                        minutes=market_index % 240,
                    )
                    yes_price = 0.25 if side == "YES" else 0.75
                    no_price = 0.25 if side == "NO" else 0.75
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "market_id": f"market-{market_index}",
                            "window_start": window_start,
                            "observed_at": window_start + timedelta(seconds=second),
                            "seconds_elapsed": second,
                            "label_up": selected_label if side == "YES" else 1 - selected_label,
                            "probability_yes": (
                                probability if side == "YES" else 1.0 - probability
                            ),
                            "yes_ask_vwap_5": yes_price,
                            "no_ask_vwap_5": no_price,
                            "yes_ask_depth": 100.0,
                            "no_ask_depth": 100.0,
                            "yes_cost_per_share": yes_price + 0.01,
                            "no_cost_per_share": no_price + 0.01,
                        }
                    )
                    market_index += 1
    return pl.DataFrame(rows)


def _support(
    candidate_ids: tuple[str, ...],
    *,
    regularization: Mapping[str, float] | None = None,
) -> dict[str, dict[str, object]]:
    evidence: dict[str, dict[str, object]] = {
        "S0": {
            "support_passed": True,
            "coverage": 0.98,
            "residual_cap_passed": True,
        }
    }
    for index, candidate_id in enumerate(candidate_ids):
        evidence[candidate_id] = {
            "support_passed": True,
            "coverage": 0.97,
            "residual_cap_passed": True,
            "regularization_strength": (
                regularization[candidate_id] if regularization is not None else float(index + 1)
            ),
            "model_complexity": 8,
        }
    return evidence


def _select(
    incumbent_probability: SelectedProbability,
    static_probability: SelectedProbability,
    dynamics: Mapping[str, SelectedProbability],
    *,
    support: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    dynamic_frames = {
        candidate_id: _probability_frame(candidate_id, probability)
        for candidate_id, probability in dynamics.items()
    }
    return select_asymmetric_book_admission_challenger(
        _probability_frame("I0", incumbent_probability),
        _probability_frame("S0", static_probability),
        dynamic_frames,
        support or _support(tuple(dynamics)),
        BookAdmissionSelectionThresholds(),
        resamples=200,
        seed=43,
    )


def _record(result: Mapping[str, object], candidate_id: str) -> dict[str, object]:
    records = result["candidate_records"]
    assert isinstance(records, list)
    return next(record for record in records if record["candidate_id"] == candidate_id)


def _gates(record: Mapping[str, object]) -> dict[str, dict[str, object]]:
    gates = record["gates"]
    assert isinstance(gates, list)
    return {gate["name"]: gate for gate in gates}


def test_four_dynamic_arms_use_shared_comparisons_and_one_se_rank() -> None:
    dynamics = {
        "D1": _probability_frame("D1", 0.75),
        "D2": _probability_frame("D2", 0.75),
        "D3": _probability_frame("D3", 0.74),
        "D4": _probability_frame("D4", 0.73),
    }
    incumbent = _probability_frame("I0", 0.70)
    static = _probability_frame("S0", 0.72)
    support = _support(
        tuple(dynamics),
        regularization={"D1": 5.0, "D2": 10.0, "D3": 20.0, "D4": 30.0},
    )

    comparison = simultaneous_paired_probability_bootstrap(
        incumbent,
        dynamics,
        resamples=200,
        seed=43,
    )
    result = select_asymmetric_book_admission_challenger(
        incumbent,
        static,
        dynamics,
        support,
        BookAdmissionSelectionThresholds(),
        resamples=200,
        seed=43,
    )

    assert comparison["challengers"] == ["D1", "D2", "D3", "D4"]
    assert result["status"] == "selected"
    assert result["selected_candidate_id"] == "D2"
    assert result["static_control"]["selectable"] is False
    assert result["one_standard_error"]["candidate_ids"] == ["D1", "D2"]
    assert result["one_standard_error"]["excluded_candidate_ids"] == ["D3", "D4"]
    assert result["rank_trace"][0]["candidate_id"] == "D2"
    assert set(result["simultaneous_comparison_to_static"]["challengers"]) == set(dynamics)
    assert "target_cell_bias_yes_45_56" in _gates(_record(result, "D2"))
    assert "target_cell_bias_yes_45_60" not in _gates(_record(result, "D2"))
    assert result["economics_used"] is False
    json.dumps(result, allow_nan=False)
    for forbidden in PROBABILITY_SELECTION_FORBIDDEN_COLUMNS:
        assert f"'{forbidden}'" not in repr(result)


def test_selection_rejects_mismatched_keys_and_labels() -> None:
    incumbent = _probability_frame("I0", 0.70)
    static = _probability_frame("S0", 0.72)
    dynamic = _probability_frame("D1", 0.75)
    first_market = dynamic["market_id"][0]
    mismatched_key = dynamic.with_columns(
        pl.when(pl.col("market_id") == first_market)
        .then(pl.lit("different-market"))
        .otherwise(pl.col("market_id"))
        .alias("market_id")
    )
    with pytest.raises(ValueError, match="keys do not match"):
        select_asymmetric_book_admission_challenger(
            incumbent,
            static,
            {"D1": mismatched_key},
            _support(("D1",)),
            BookAdmissionSelectionThresholds(),
            resamples=50,
            seed=1,
        )

    mismatched_label = dynamic.with_columns(
        pl.when(pl.col("market_id") == first_market)
        .then(1 - pl.col("label_up"))
        .otherwise(pl.col("label_up"))
        .alias("label_up")
    )
    with pytest.raises(ValueError, match="labels do not match"):
        select_asymmetric_book_admission_challenger(
            incumbent,
            static,
            {"D1": mismatched_label},
            _support(("D1",)),
            BookAdmissionSelectionThresholds(),
            resamples=50,
            seed=1,
        )


def test_selection_rejects_pnl_mutation_before_scoring() -> None:
    dynamic = _probability_frame("D1", 0.75).with_columns(pl.lit(99.0).alias("pnl"))

    with pytest.raises(ValueError, match="cannot receive economic columns: pnl"):
        select_asymmetric_book_admission_challenger(
            _probability_frame("I0", 0.70),
            _probability_frame("S0", 0.72),
            {"D1": dynamic},
            _support(("D1",)),
            BookAdmissionSelectionThresholds(),
            resamples=50,
            seed=1,
        )


@pytest.mark.parametrize(
    ("field", "value", "failed_gate"),
    [
        ("support_passed", False, "candidate_support_pass"),
        ("coverage", 0.89, "candidate_feature_coverage"),
        ("residual_cap_passed", False, "candidate_residual_cap_pass"),
    ],
)
def test_support_coverage_and_residual_cap_fail_closed(
    field: str,
    value: object,
    failed_gate: str,
) -> None:
    support = _support(("D1",))
    support["D1"][field] = value

    result = _select(0.70, 0.72, {"D1": 0.75}, support=support)

    record = _record(result, "D1")
    assert record["passed"] is False
    assert _gates(record)[failed_gate]["passed"] is False


def test_static_attribution_gate_blocks_dynamic_that_only_beats_incumbent() -> None:
    result = _select(0.65, 0.75, {"D1": 0.74})
    record = _record(result, "D1")
    gates = _gates(record)

    assert gates["proper_score_improvement_over_incumbent"]["passed"] is True
    assert gates["proper_score_improvement_over_static"]["passed"] is False
    assert record["passed"] is False
    assert "proper_score_improvement_over_static" in result["failure_trace"][0]["failed_gates"]


def test_selected_bias_is_measured_in_selected_side_probability_space() -> None:
    result = _select(0.70, 0.76, {"D1": 0.79})
    record = _record(result, "D1")
    gates = _gates(record)

    assert record["target_metrics"]["overall"]["bias"] == pytest.approx(0.0)
    assert record["selected_metrics"]["overall"]["bias"] == pytest.approx(0.04)
    assert gates["selected_side_absolute_bias"]["passed"] is False
    assert gates["selected_side_absolute_bias"]["threshold"] == pytest.approx(0.03)


def test_joint_daily_gate_requires_eight_of_the_ten_days() -> None:
    def dynamic_probability(utc_day: int, _side: str, _second: int, _outcome: int) -> float:
        return 0.75 if utc_day < 7 else 0.50

    result = _select(0.55, 0.55, {"D1": dynamic_probability})
    record = _record(result, "D1")
    daily_gate = _gates(record)["joint_noninferior_utc_days"]

    assert record["joint_noninferior_utc_days"] == 7
    assert daily_gate == {
        "name": "joint_noninferior_utc_days",
        "observed": 7,
        "threshold": 8,
        "operator": ">=",
        "passed": False,
    }


def test_joint_daily_gate_uses_common_days_across_both_references() -> None:
    def incumbent_probability(utc_day: int, _side: str, _second: int, _outcome: int) -> float:
        if utc_day < 2:
            return 0.75
        if utc_day < 4:
            return 0.50
        return 0.65

    def static_probability(utc_day: int, _side: str, _second: int, _outcome: int) -> float:
        if utc_day < 2:
            return 0.50
        if utc_day < 4:
            return 0.75
        return 0.65

    result = _select(incumbent_probability, static_probability, {"D1": 0.65})
    record = _record(result, "D1")

    assert record["noninferior_utc_days_to_incumbent"] == 8
    assert record["noninferior_utc_days_to_static"] == 8
    assert record["joint_noninferior_utc_days"] == 6
    assert _gates(record)["joint_noninferior_utc_days"]["passed"] is False
