from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from btc_directional_model.asymmetric_d4_calibration_evaluation import (
    CONSUMED_EVIDENCE_SCOPE,
    D4_BASE_ID,
    D4_MODEL_IDS,
    INCUMBENT_ID,
    POOLED_NO_CALIBRATION_ID,
    TIME_LOCAL_NO_CALIBRATION_ID,
    D4SideCalibrationSelectionThresholds,
    build_d4_projected_pnl_report,
    select_d4_side_calibration_challenger,
)


def _probability_frame(
    candidate_id: str, yes_probability: float, no_probability: float
) -> pl.DataFrame:
    start = datetime(2026, 7, 21, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    market_index = 0
    for day in range(10):
        for side in ("YES", "NO"):
            for second in (5, 20, 35, 50):
                for selected_label in (1, 1, 1, 0):
                    window_start = start + timedelta(days=day, minutes=market_index)
                    selected_probability = yes_probability if side == "YES" else no_probability
                    rows.append(
                        {
                            "candidate_name": candidate_id,
                            "market_id": f"market-{market_index}",
                            "window_start": window_start,
                            "observed_at": window_start + timedelta(seconds=second),
                            "seconds_elapsed": second,
                            "label_up": selected_label if side == "YES" else 1 - selected_label,
                            "incumbent_probability_yes": (
                                selected_probability
                                if side == "YES"
                                else 1.0 - selected_probability
                            ),
                            "yes_ask_vwap_5": 0.25 if side == "YES" else 0.75,
                            "no_ask_vwap_5": 0.25 if side == "NO" else 0.75,
                            "yes_ask_depth": 100.0,
                            "no_ask_depth": 100.0,
                            "yes_cost_per_share": 0.26 if side == "YES" else 0.76,
                            "no_cost_per_share": 0.26 if side == "NO" else 0.76,
                        }
                    )
                    market_index += 1
    return pl.DataFrame(rows)


def _support() -> dict[str, dict[str, object]]:
    return {
        POOLED_NO_CALIBRATION_ID: {
            "support_passed": True,
            "support_failures": (),
        },
        TIME_LOCAL_NO_CALIBRATION_ID: {
            "support_passed": True,
            "support_failures": (),
        },
    }


def _selection(*, support: dict[str, dict[str, object]] | None = None) -> dict[str, object]:
    return select_d4_side_calibration_challenger(
        _probability_frame(INCUMBENT_ID, 0.70, 0.70),
        _probability_frame(D4_BASE_ID, 0.75, 0.82),
        {
            POOLED_NO_CALIBRATION_ID: _probability_frame(POOLED_NO_CALIBRATION_ID, 0.75, 0.76),
            TIME_LOCAL_NO_CALIBRATION_ID: _probability_frame(
                TIME_LOCAL_NO_CALIBRATION_ID, 0.75, 0.75
            ),
        },
        support or _support(),
        D4SideCalibrationSelectionThresholds(),
        resamples=200,
        seed=17,
        evidence_scope=CONSUMED_EVIDENCE_SCOPE,
        qualification_eligible=False,
    )


def _record(selection: dict[str, object], candidate_id: str) -> dict[str, object]:
    records = selection["candidate_records"]
    assert isinstance(records, list)
    return next(record for record in records if record["candidate_id"] == candidate_id)


def _ledger(candidate_id: str, no_wins_per_four: int) -> pl.DataFrame:
    start = datetime(2026, 7, 21, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for index in range(80):
        selected_yes = index % 2 == 0
        group_position = (index // 2) % 4
        won = group_position < (3 if selected_yes else no_wins_per_four)
        cost = 0.26
        window_start = start + timedelta(days=index // 8, minutes=index)
        rows.append(
            {
                "candidate_id": candidate_id,
                "market_id": f"economic-market-{index}",
                "window_start": window_start,
                "observed_at": window_start + timedelta(seconds=20),
                "seconds_elapsed": 20,
                "selected_yes": selected_yes,
                "won": won,
                "quantity": 5.0,
                "selected_execution_cost_per_share": cost,
                "selected_admission_cost_per_share": cost,
                "selected_share_price": 0.25,
                "selected_probability": 0.75 if won else 0.30,
                "selected_underdog": True,
                "entry_debit": cost * 5.0,
                "realized_net": (float(won) - cost) * 5.0,
                "selected_edge_per_share": 0.04,
            }
        )
    return pl.DataFrame(rows)


def _pnl_inputs() -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    ledgers = {
        INCUMBENT_ID: _ledger(INCUMBENT_ID, 3),
        D4_BASE_ID: _ledger(D4_BASE_ID, 2),
        POOLED_NO_CALIBRATION_ID: _ledger(POOLED_NO_CALIBRATION_ID, 2),
        TIME_LOCAL_NO_CALIBRATION_ID: _ledger(TIME_LOCAL_NO_CALIBRATION_ID, 3),
    }
    eligible = ledgers[INCUMBENT_ID].select("market_id", "window_start")
    return ledgers, eligible


def _seal(selection: dict[str, object]) -> dict[str, object]:
    return {
        "selected_candidate_id": selection["selected_candidate_id"],
        "probability_selection_sha256": "a" * 64,
        "economics_opened": False,
        "selection_uses_economics": False,
    }


def test_probability_selection_is_matched_side_aware_and_diagnostic_only() -> None:
    selection = _selection()

    assert selection["status"] == "diagnostic_selected_consumed_evidence"
    assert selection["selected_candidate_id"] == TIME_LOCAL_NO_CALIBRATION_ID
    assert selection["qualification_eligible"] is False
    assert selection["economics_used"] is False
    assert selection["promotion_authorized"] is False
    assert selection["target_cohort"] == {
        "definition": "seconds 1-55 and either YES or NO raw VWAP5 in [0.20,0.30)",
        "rows": 320,
        "markets": 320,
        "utc_days": 10,
    }
    n2 = _record(selection, TIME_LOCAL_NO_CALIBRATION_ID)
    assert n2["selected_no_absolute_bias_reduction"] == pytest.approx(0.07)
    assert n2["selected_metrics"]["sides"]["YES"]["bias"] == pytest.approx(0.0)
    assert n2["joint_noninferior_utc_days"] == 10
    assert selection["rank_trace"][0]["candidate_id"] == TIME_LOCAL_NO_CALIBRATION_ID
    json.dumps(selection, allow_nan=False)


def test_failed_calibration_support_is_never_selectable() -> None:
    support = _support()
    support[TIME_LOCAL_NO_CALIBRATION_ID]["support_passed"] = False
    selection = _selection(support=support)

    n2 = _record(selection, TIME_LOCAL_NO_CALIBRATION_ID)
    assert n2["passed"] is False
    assert "calibration_support" in selection["failure_trace"][1]["failed_gates"]
    assert selection["selected_candidate_id"] == POOLED_NO_CALIBRATION_ID


def test_probability_selector_rejects_economics_and_mismatched_candidates() -> None:
    support = _support()
    support[POOLED_NO_CALIBRATION_ID]["pnl"] = 1.0
    with pytest.raises(ValueError, match="cannot receive economic fields"):
        _selection(support=support)

    with pytest.raises(ValueError, match="exactly N1 and N2"):
        select_d4_side_calibration_challenger(
            _probability_frame(INCUMBENT_ID, 0.70, 0.70),
            _probability_frame(D4_BASE_ID, 0.75, 0.82),
            {POOLED_NO_CALIBRATION_ID: _probability_frame(POOLED_NO_CALIBRATION_ID, 0.75, 0.76)},
            _support(),
            D4SideCalibrationSelectionThresholds(),
            resamples=10,
            seed=1,
            evidence_scope=CONSUMED_EVIDENCE_SCOPE,
            qualification_eligible=False,
        )


def test_post_seal_projected_pnl_reports_every_arm_without_promoting_consumed_evidence() -> None:
    selection = _selection()
    ledgers, eligible = _pnl_inputs()

    report = build_d4_projected_pnl_report(
        ledgers,
        eligible,
        selection,
        _seal(selection),
        selection_seal_sha256="b" * 64,
        resamples=200,
        seed=23,
    )

    assert report["all_predeclared_models_reported"] == list(D4_MODEL_IDS)
    assert set(report["models"]) == set(D4_MODEL_IDS)
    assert report["probability_selection_used_pnl"] is False
    assert report["economics_opened_after_probability_selection_seal"] is True
    assert report["nonselected_pnl_may_qualify_model"] is False
    assert report["promotion_authorized"] is False
    for candidate_id, model in report["models"].items():
        assert model["metrics"]["net_profit"] is not None
        assert model["metrics"]["stress_1c_net_profit"] is not None
        assert model["projected_pnl"]["winning_trades"] > 0
        assert model["projected_pnl"]["net_profit"] == model["metrics"]["net_profit"]
        assert model["coverage"]["trades_per_eligible_market"] == pytest.approx(1.0)
        assert set(model["sides"]) == {"YES", "NO"}
        if candidate_id != TIME_LOCAL_NO_CALIBRATION_ID:
            assert model["diagnostic_only"] is True
            assert model["economic_qualification_eligible"] is False
    qualification = report["selected_economic_qualification"]
    assert qualification["candidate_id"] == TIME_LOCAL_NO_CALIBRATION_ID
    assert qualification["qualification_eligible"] is False
    assert qualification["qualified"] is False
    assert qualification["status"] == "diagnostic_only_consumed_evidence"
    assert (
        report["paired_stressed_pnl"][TIME_LOCAL_NO_CALIBRATION_ID]["to_d4_base"][
            "candidate_minus_reference_stress_1c_net_profit"
        ]
        > 0.0
    )
    json.dumps(report, allow_nan=False)


def test_post_seal_projected_pnl_still_reports_all_arms_without_a_winner() -> None:
    selection = _selection()
    selection["selected_candidate_id"] = None
    selection["status"] = "diagnostic_no_quality_configuration_consumed_evidence"
    ledgers, eligible = _pnl_inputs()

    report = build_d4_projected_pnl_report(
        ledgers,
        eligible,
        selection,
        _seal(selection),
        selection_seal_sha256="b" * 64,
        resamples=50,
        seed=3,
    )

    assert report["status"] == "post_selection_diagnostic_no_probability_winner"
    assert report["selected_economic_qualification"] is None
    assert set(report["models"]) == set(D4_MODEL_IDS)
    assert all(model["diagnostic_only"] for model in report["models"].values())


def test_projected_pnl_fails_closed_before_or_after_an_invalid_seal() -> None:
    selection = _selection()
    ledgers, eligible = _pnl_inputs()
    opened = {**_seal(selection), "economics_opened": True}

    with pytest.raises(ValueError, match="seal is incomplete"):
        build_d4_projected_pnl_report(
            ledgers,
            eligible,
            selection,
            opened,
            selection_seal_sha256="b" * 64,
            resamples=20,
            seed=1,
        )

    with pytest.raises(ValueError, match="seal SHA-256 is invalid"):
        build_d4_projected_pnl_report(
            ledgers,
            eligible,
            selection,
            _seal(selection),
            selection_seal_sha256="not-a-hash",
            resamples=20,
            seed=1,
        )
