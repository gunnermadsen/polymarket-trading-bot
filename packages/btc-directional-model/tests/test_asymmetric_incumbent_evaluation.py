from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from btc_directional_model.asymmetric_incumbent_evaluation import (
    PROBABILITY_SELECTION_FORBIDDEN_COLUMNS,
    REQUIRED_CALIBRATION_CELLS,
    build_incumbent_correction_ledger,
    incumbent_probability_metrics,
    paired_incumbent_economics,
    probability_only_first_crossings,
    select_incumbent_calibration_challenger,
    simultaneous_paired_probability_bootstrap,
    target_opportunity_probability_cohort,
)


def _scored_row(
    *,
    candidate_id: str,
    market_id: str,
    window_start: datetime,
    seconds_elapsed: int,
    label_up: int,
    probability_yes: float,
    cheap_side: str,
) -> dict[str, object]:
    yes_price = 0.25 if cheap_side == "YES" else 0.75
    no_price = 0.25 if cheap_side == "NO" else 0.75
    return {
        "candidate_id": candidate_id,
        "market_id": market_id,
        "window_start": window_start,
        "observed_at": window_start + timedelta(seconds=seconds_elapsed),
        "seconds_elapsed": seconds_elapsed,
        "label_up": label_up,
        "probability_yes": probability_yes,
        "yes_ask_vwap_5": yes_price,
        "no_ask_vwap_5": no_price,
        "yes_ask_depth": 100.0,
        "no_ask_depth": 100.0,
        "yes_cost_per_share": yes_price + 0.01,
        "no_cost_per_share": no_price + 0.01,
    }


def test_probability_only_first_crossing_recomputes_edges_and_strips_economics() -> None:
    start = datetime(2026, 7, 23, tzinfo=UTC)
    frame = pl.DataFrame(
        [
            _scored_row(
                candidate_id="C1",
                market_id="m1",
                window_start=start,
                seconds_elapsed=1,
                label_up=1,
                probability_yes=0.27,
                cheap_side="YES",
            ),
            _scored_row(
                candidate_id="C1",
                market_id="m1",
                window_start=start,
                seconds_elapsed=2,
                label_up=1,
                probability_yes=0.70,
                cheap_side="YES",
            ),
            _scored_row(
                candidate_id="C1",
                market_id="m2",
                window_start=start + timedelta(minutes=5),
                seconds_elapsed=16,
                label_up=0,
                probability_yes=0.25,
                cheap_side="NO",
            ),
        ]
    )

    selected = probability_only_first_crossings(frame)

    assert selected.height == 2
    assert selected.sort("market_id")["seconds_elapsed"].to_list() == [2, 16]
    assert selected.sort("market_id")["selected_yes"].to_list() == [True, False]
    assert selected.sort("market_id")["selected_label"].to_list() == [1, 1]
    assert selected.sort("market_id")["time_cell"].to_list() == ["1_15", "15_30"]
    assert not PROBABILITY_SELECTION_FORBIDDEN_COLUMNS.intersection(selected.columns)
    assert not {
        "yes_cost_per_share",
        "selected_share_price",
        "selected_edge_per_share",
        "won",
    }.intersection(selected.columns)

    selected_metrics = incumbent_probability_metrics(selected)
    assert selected_metrics["overall"]["actual_rate"] == pytest.approx(1.0)
    assert selected_metrics["overall"]["mean_probability"] == pytest.approx(
        (0.70 + 0.75) / 2.0
    )

    with pytest.raises(ValueError, match="cannot receive economic columns"):
        probability_only_first_crossings(frame.with_columns(pl.lit(1.0).alias("realized_net")))


def test_probability_metrics_are_market_equal_and_report_day_side_time() -> None:
    start = datetime(2026, 7, 23, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "market_id": ["many", "many", "single"],
            "window_start": [start, start, start + timedelta(days=1)],
            "observed_at": [
                start + timedelta(seconds=1),
                start + timedelta(seconds=2),
                start + timedelta(days=1, seconds=16),
            ],
            "seconds_elapsed": [1, 2, 16],
            "label_up": [1, 1, 1],
            "probability_yes": [0.9, 0.9, 0.1],
            "selected_yes": [True, True, True],
        }
    )

    metrics = incumbent_probability_metrics(frame)

    assert metrics["overall"]["brier"] == pytest.approx((0.01 + 0.81) / 2.0)
    assert metrics["overall"]["markets"] == 2
    assert set(metrics["utc_days"]) == {"2026-07-23", "2026-07-24"}
    assert metrics["sides"]["YES"]["rows"] == 3
    assert metrics["time_cells"]["YES_1_15"]["rows"] == 2
    assert metrics["time_cells"]["YES_15_30"]["rows"] == 1
    assert metrics["time_cells"]["NO_1_15"]["rows"] == 0
    assert "YES_45_60" in REQUIRED_CALIBRATION_CELLS
    assert "YES_45_56" not in REQUIRED_CALIBRATION_CELLS


def _matched_probability_frames() -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    start = datetime(2026, 7, 23, tzinfo=UTC)
    rows = []
    candidate_probabilities: dict[str, list[float]] = {"C1": [], "C2": [], "C3": []}
    for index in range(10):
        label = index % 2
        rows.append(
            {
                "market_id": f"m{index}",
                "window_start": start + timedelta(days=index),
                "observed_at": start + timedelta(days=index, seconds=5),
                "seconds_elapsed": 5,
                "label_up": label,
                "probability_yes": 0.6 if label else 0.4,
            }
        )
        candidate_probabilities["C1"].append(0.7 if label else 0.3)
        candidate_probabilities["C2"].append(0.6 if label else 0.4)
        candidate_probabilities["C3"].append(0.4 if label else 0.6)
    incumbent = pl.DataFrame(rows)
    challengers = {
        name: incumbent.with_columns(pl.Series("probability_yes", probabilities))
        for name, probabilities in candidate_probabilities.items()
    }
    return incumbent, challengers


def test_simultaneous_bootstrap_is_paired_shared_and_deterministic() -> None:
    incumbent, challengers = _matched_probability_frames()

    first = simultaneous_paired_probability_bootstrap(
        incumbent,
        challengers,
        resamples=500,
        seed=17,
    )
    second = simultaneous_paired_probability_bootstrap(
        incumbent,
        challengers,
        resamples=500,
        seed=17,
    )

    assert first == second
    assert first["challengers"] == ["C1", "C2", "C3"]
    assert first["comparisons"]["C1"]["brier_delta"]["point"] < 0.0
    assert first["comparisons"]["C2"]["brier_delta"]["point"] == pytest.approx(0.0)
    assert first["comparisons"]["C3"]["brier_delta"]["point"] > 0.0
    assert first["comparisons"]["C1"]["brier_delta"]["simultaneous_upper_95"] <= 0.0

    mismatched = dict(challengers)
    mismatched["C1"] = mismatched["C1"].filter(pl.col("market_id") != "m0")
    with pytest.raises(ValueError, match="keys do not match"):
        simultaneous_paired_probability_bootstrap(
            incumbent,
            mismatched,
            resamples=100,
            seed=17,
        )


def test_simultaneous_bootstrap_supports_two_predeclared_estimator_arms() -> None:
    incumbent, calibration_challengers = _matched_probability_frames()
    challengers = {
        "E1": calibration_challengers["C1"],
        "E2": calibration_challengers["C3"],
    }

    first = simultaneous_paired_probability_bootstrap(
        incumbent,
        challengers,
        resamples=500,
        seed=19,
    )
    second = simultaneous_paired_probability_bootstrap(
        incumbent,
        challengers,
        resamples=500,
        seed=19,
    )

    assert first == second
    assert first["challengers"] == ["E1", "E2"]
    assert set(first["comparisons"]) == {"E1", "E2"}
    assert first["comparisons"]["E1"]["brier_delta"]["point"] < 0.0
    assert first["comparisons"]["E2"]["brier_delta"]["point"] > 0.0


@pytest.mark.parametrize("challenger_count", [0, 1, 5])
def test_probability_comparison_rejects_challenger_counts_outside_two_to_four(
    challenger_count: int,
) -> None:
    incumbent, calibration_challengers = _matched_probability_frames()
    source = tuple(calibration_challengers.values())
    challengers = {
        f"candidate-{index}": source[index % len(source)] for index in range(challenger_count)
    }

    with pytest.raises(ValueError, match="requires two to four challengers"):
        simultaneous_paired_probability_bootstrap(
            incumbent,
            challengers,
            resamples=100,
            seed=17,
        )


def _selection_frames() -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    start = datetime(2026, 7, 23, tzinfo=UTC)
    seconds = (5, 20, 35, 50)
    incumbent_rows = []
    candidate_rows: dict[str, list[dict[str, object]]] = {
        "C1": [],
        "C2": [],
        "C3": [],
    }
    market_index = 0
    for utc_day in range(10):
        for side in ("YES", "NO"):
            for second in seconds:
                for selected_label in (1, 1, 1, 0):
                    raw_label = selected_label if side == "YES" else 1 - selected_label
                    window = start + timedelta(days=utc_day, minutes=market_index % 200)
                    common = {
                        "market_id": f"selection-{market_index}",
                        "window_start": window,
                        "seconds_elapsed": second,
                        "label_up": raw_label,
                        "cheap_side": side,
                    }
                    incumbent_rows.append(
                        _scored_row(
                            candidate_id="I0",
                            probability_yes=0.70 if side == "YES" else 0.30,
                            **common,
                        )
                    )
                    for candidate_id, selected_probability in (
                        ("C1", 0.75),
                        ("C2", 0.74),
                        ("C3", 0.60),
                    ):
                        candidate_rows[candidate_id].append(
                            _scored_row(
                                candidate_id=candidate_id,
                                probability_yes=(
                                    selected_probability
                                    if side == "YES"
                                    else 1.0 - selected_probability
                                ),
                                **common,
                            )
                        )
                    market_index += 1
    return pl.DataFrame(incumbent_rows), {
        candidate_id: pl.DataFrame(rows) for candidate_id, rows in candidate_rows.items()
    }


def _complete_calibration_support() -> dict[str, object]:
    return {
        "cells": {
            name: {
                "fitted": True,
                "converged": True,
                "fallback": False,
                "outcome_counts": {"0": 25, "1": 75},
            }
            for name in REQUIRED_CALIBRATION_CELLS
        }
    }


def test_selection_uses_probability_only_gates_and_deterministic_rank() -> None:
    incumbent, challengers = _selection_frames()
    out_of_scope_start = datetime(2026, 7, 23, 23, tzinfo=UTC)
    incumbent = pl.concat(
        [
            incumbent,
            pl.DataFrame(
                [
                    _scored_row(
                        candidate_id="I0",
                        market_id="after-policy",
                        window_start=out_of_scope_start,
                        seconds_elapsed=56,
                        label_up=1,
                        probability_yes=0.01,
                        cheap_side="YES",
                    )
                ]
            ),
        ],
        how="vertical_relaxed",
    )
    challengers = {
        candidate_id: pl.concat(
            [
                frame,
                pl.DataFrame(
                    [
                        _scored_row(
                            candidate_id=candidate_id,
                            market_id="after-policy",
                            window_start=out_of_scope_start,
                            seconds_elapsed=56,
                            label_up=1,
                            probability_yes=0.99,
                            cheap_side="YES",
                        )
                    ]
                ),
            ],
            how="vertical_relaxed",
        )
        for candidate_id, frame in challengers.items()
    }
    support = {candidate_id: _complete_calibration_support() for candidate_id in challengers}

    selection = select_incumbent_calibration_challenger(
        incumbent,
        challengers,
        support,
        resamples=500,
        seed=31,
    )

    assert selection["status"] == "selected"
    assert selection["selected_candidate_id"] == "C1"
    assert selection["economics_used"] is False
    assert selection["target_cohort"] == {
        "definition": "seconds 1-55 and either YES or NO raw VWAP5 in [0.20,0.30)",
        "rows": 320,
        "markets": 320,
        "utc_days": 10,
    }
    assert selection["rank_trace"][0]["candidate_id"] == "C1"
    records = {record["candidate_id"]: record for record in selection["candidate_records"]}
    assert records["C1"]["passed"] is True
    assert records["C3"]["passed"] is False
    assert all(item["passed"] for item in records["C1"]["calibration_support"]["cell_checks"])
    json.dumps(selection, allow_nan=False)
    rendered = repr(selection)
    for forbidden in PROBABILITY_SELECTION_FORBIDDEN_COLUMNS:
        assert f"'{forbidden}'" not in rendered


def test_cell_bias_gate_uses_all_target_rows_not_only_first_entries() -> None:
    incumbent, challengers = _selection_frames()

    def append_late_rows(frame: pl.DataFrame, *, overconfident: bool) -> pl.DataFrame:
        rows = []
        for row in frame.filter(pl.col("seconds_elapsed") == 5).iter_rows(named=True):
            row["seconds_elapsed"] = 6
            row["observed_at"] = row["window_start"] + timedelta(seconds=6)
            if overconfident:
                row["probability_yes"] = 0.99 if row["yes_ask_vwap_5"] < 0.30 else 0.01
            rows.append(row)
        return pl.concat([frame, pl.DataFrame(rows)], how="vertical_relaxed")

    incumbent = append_late_rows(incumbent, overconfident=False)
    challengers = {
        candidate_id: append_late_rows(
            frame,
            overconfident=candidate_id == "C1",
        )
        for candidate_id, frame in challengers.items()
    }
    support = {candidate_id: _complete_calibration_support() for candidate_id in challengers}

    selection = select_incumbent_calibration_challenger(
        incumbent,
        challengers,
        support,
        resamples=200,
        seed=41,
    )

    c1 = next(record for record in selection["candidate_records"] if record["candidate_id"] == "C1")
    gates = {gate["name"]: gate for gate in c1["gates"]}
    assert abs(c1["selected_opportunity_metrics"]["time_cells"]["YES_1_15"]["bias"]) <= 0.05
    assert abs(c1["metrics"]["time_cells"]["YES_1_15"]["bias"]) > 0.05
    assert gates["target_cell_bias_yes_1_15"]["passed"] is False


def test_selection_supports_only_two_predeclared_estimator_arms_deterministically() -> None:
    incumbent, calibration_challengers = _selection_frames()
    challengers = {
        "E1": calibration_challengers["C1"].with_columns(pl.lit("E1").alias("candidate_id")),
        "E2": calibration_challengers["C2"].with_columns(pl.lit("E2").alias("candidate_id")),
    }
    support = {candidate_id: _complete_calibration_support() for candidate_id in challengers}

    first = select_incumbent_calibration_challenger(
        incumbent,
        challengers,
        support,
        resamples=500,
        seed=37,
    )
    second = select_incumbent_calibration_challenger(
        incumbent,
        challengers,
        support,
        resamples=500,
        seed=37,
    )

    assert first == second
    assert first["status"] == "selected"
    assert first["selected_candidate_id"] == "E1"
    assert first["simultaneous_comparison"]["challengers"] == ["E1", "E2"]
    assert {record["candidate_id"] for record in first["candidate_records"]} == {"E1", "E2"}
    assert "D0" not in repr(first)


@pytest.mark.parametrize("challenger_count", [1, 5])
def test_selection_rejects_challenger_counts_outside_two_to_four(
    challenger_count: int,
) -> None:
    incumbent, calibration_challengers = _selection_frames()
    source = tuple(calibration_challengers.values())
    challengers = {
        f"candidate-{index}": source[index % len(source)] for index in range(challenger_count)
    }
    support = {candidate_id: _complete_calibration_support() for candidate_id in challengers}

    with pytest.raises(ValueError, match="requires two to four challengers"):
        select_incumbent_calibration_challenger(
            incumbent,
            challengers,
            support,
            resamples=100,
            seed=31,
        )


def test_target_opportunity_cohort_requires_prices_and_filters_before_scoring() -> None:
    incumbent, _ = _selection_frames()
    target = target_opportunity_probability_cohort(incumbent)

    assert target.height == 320
    assert target["seconds_elapsed"].max() == 50
    with pytest.raises(ValueError, match="missing columns: no_ask_vwap_5"):
        target_opportunity_probability_cohort(incumbent.drop("no_ask_vwap_5"))


def _minimal_ledger_row(
    *,
    market_id: str,
    window_start: datetime,
    second: int,
    selected_yes: bool,
    won: bool,
) -> dict[str, object]:
    execution_cost = 0.25
    quantity = 5.0
    return {
        "market_id": market_id,
        "window_start": window_start,
        "observed_at": window_start + timedelta(seconds=second),
        "seconds_elapsed": second,
        "selected_yes": selected_yes,
        "won": won,
        "quantity": quantity,
        "selected_execution_cost_per_share": execution_cost,
        "realized_net": quantity * (float(won) - execution_cost),
    }


def test_correction_ledger_categories_and_pnl_reconcile() -> None:
    start = datetime(2026, 7, 23, tzinfo=UTC)
    incumbent = pl.DataFrame(
        [
            _minimal_ledger_row(
                market_id="avoided", window_start=start, second=5, selected_yes=True, won=False
            ),
            _minimal_ledger_row(
                market_id="suppressed",
                window_start=start + timedelta(minutes=5),
                second=5,
                selected_yes=True,
                won=True,
            ),
            _minimal_ledger_row(
                market_id="correct",
                window_start=start + timedelta(minutes=10),
                second=5,
                selected_yes=True,
                won=False,
            ),
            _minimal_ledger_row(
                market_id="incorrect",
                window_start=start + timedelta(minutes=15),
                second=5,
                selected_yes=True,
                won=True,
            ),
            _minimal_ledger_row(
                market_id="retimed",
                window_start=start + timedelta(minutes=20),
                second=5,
                selected_yes=True,
                won=True,
            ),
            _minimal_ledger_row(
                market_id="unchanged",
                window_start=start + timedelta(minutes=25),
                second=5,
                selected_yes=False,
                won=False,
            ),
        ]
    )
    challenger = pl.DataFrame(
        [
            _minimal_ledger_row(
                market_id="correct",
                window_start=start + timedelta(minutes=10),
                second=5,
                selected_yes=False,
                won=True,
            ),
            _minimal_ledger_row(
                market_id="incorrect",
                window_start=start + timedelta(minutes=15),
                second=5,
                selected_yes=False,
                won=False,
            ),
            _minimal_ledger_row(
                market_id="candidate-win",
                window_start=start + timedelta(minutes=30),
                second=5,
                selected_yes=True,
                won=True,
            ),
            _minimal_ledger_row(
                market_id="candidate-loss",
                window_start=start + timedelta(minutes=35),
                second=5,
                selected_yes=False,
                won=False,
            ),
            _minimal_ledger_row(
                market_id="retimed",
                window_start=start + timedelta(minutes=20),
                second=6,
                selected_yes=True,
                won=True,
            ),
            _minimal_ledger_row(
                market_id="unchanged",
                window_start=start + timedelta(minutes=25),
                second=5,
                selected_yes=False,
                won=False,
            ),
        ]
    )

    correction, summary = build_incumbent_correction_ledger(incumbent, challenger)

    assert correction.height == 8
    assert all(value == 1 for value in summary["categories"].values())
    assert summary["net_corrected_decisions"] == 0
    assert summary["incremental_pnl_exact"] == pytest.approx(
        challenger["realized_net"].sum() - incumbent["realized_net"].sum()
    )
    assert summary["incremental_pnl_stress_1c"] == pytest.approx(
        correction["incremental_pnl_stress_1c"].sum()
    )


def _economic_ledger_row(
    *,
    market_id: str,
    window_start: datetime,
    selected_yes: bool,
    won: bool,
) -> dict[str, object]:
    row = _minimal_ledger_row(
        market_id=market_id,
        window_start=window_start,
        second=5,
        selected_yes=selected_yes,
        won=won,
    )
    row.update(
        {
            "entry_debit": 1.25,
            "selected_probability": 0.35,
            "selected_admission_cost_per_share": 0.26,
            "selected_share_price": 0.25,
            "selected_underdog": True,
            "selected_edge_per_share": 0.09,
        }
    )
    return row


def test_paired_economics_qualifies_reconciled_common_universe_improvement() -> None:
    start = datetime(2026, 7, 23, tzinfo=UTC)
    incumbent_rows = []
    challenger_rows = []
    universe_rows = []
    for index in range(60):
        window = start + timedelta(days=index % 10, minutes=index * 5)
        market_id = f"economic-{index}"
        universe_rows.append({"market_id": market_id, "window_start": window})
        if index >= 50:
            continue
        incumbent_won = index >= 15
        incumbent_yes = index % 2 == 0
        incumbent_rows.append(
            _economic_ledger_row(
                market_id=market_id,
                window_start=window,
                selected_yes=incumbent_yes,
                won=incumbent_won,
            )
        )
        corrected = index < 3
        challenger_rows.append(
            _economic_ledger_row(
                market_id=market_id,
                window_start=window,
                selected_yes=(not incumbent_yes) if corrected else incumbent_yes,
                won=True if corrected else incumbent_won,
            )
        )

    result = paired_incumbent_economics(
        pl.DataFrame(incumbent_rows),
        pl.DataFrame(challenger_rows),
        eligible_markets=pl.DataFrame(universe_rows),
        resamples=2_000,
        seed=41,
    )

    assert result["status"] == "qualified"
    assert result["eligible_markets"] == 60
    assert result["correction_summary"]["correct_side_changes"] == 3
    assert result["correction_summary"]["net_corrected_decisions"] == 3
    assert result["correction_summary"]["corrected_decision_improvement_utc_days"] == 3
    assert (
        result["paired_bootstrap"][
            "challenger_minus_incumbent_stress_1c_profit_per_eligible_market"
        ]["lower_95"]
        >= 0.0
    )
    assert all(gate["passed"] for gate in result["gates"])
    json.dumps(result, allow_nan=False)
