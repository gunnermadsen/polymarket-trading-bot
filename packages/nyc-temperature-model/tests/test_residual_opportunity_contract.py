from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from nyc_temperature_model.residual_opportunity import (
    MarketOffsetFit,
    ResidualTrainingRow,
    _event_date_weights,
    apply_rolling_market_offset,
    contract_training_rows,
    fit_market_offset,
    market_offset_probability,
)


def _training_row(index: int, *, outcome: bool | None = None) -> ResidualTrainingRow:
    event_date = date(2026, 4, 1) + timedelta(days=index)
    decision_time = datetime.combine(event_date, datetime.min.time(), UTC)
    market = 0.12 if index % 2 == 0 else 0.22
    weather = 0.45 if index % 3 == 0 else 0.08
    return ResidualTrainingRow(
        event_date=event_date,
        decision_time=decision_time,
        source_timestamp=decision_time - timedelta(seconds=1),
        decision_hour_local=0 if index % 2 == 0 else 12,
        market_id=f"market-{index}",
        weather_probability_yes=weather,
        market_probability_yes=market,
        resolved_yes=(index % 7 == 0) if outcome is None else outcome,
        label_available_at=decision_time + timedelta(days=1, hours=2),
    )


def _candidate(
    index: int,
    side: str,
    *,
    label_available_at: datetime,
    outcome: bool | None = None,
) -> dict:
    event_date = date(2026, 4, 1) + timedelta(days=index)
    decision_time = datetime.combine(event_date, datetime.min.time(), UTC)
    yes_outcome = (index % 7 == 0) if outcome is None else outcome
    yes_weather = 0.30
    yes_market = 0.15
    yes_side = side == "YES"
    probability = yes_weather if yes_side else 1.0 - yes_weather
    market = yes_market if yes_side else 1.0 - yes_market
    cost = 0.12 if yes_side else 0.88
    resolved = yes_outcome if yes_side else not yes_outcome
    return {
        "process_id": "process",
        "model_run_id": "model-0",
        "market_id": f"market-{index}",
        "split": "discovery",
        "event_date": event_date,
        "decision_time": decision_time,
        "decision_hour_local": 0,
        "side": side,
        "quantity": 5.0,
        "probability": probability,
        "probability_lower": max(0.0, probability - 0.03),
        "weather_probability": probability,
        "weather_probability_lower": max(0.0, probability - 0.03),
        "market_probability_proxy": market,
        "ask_vwap": cost - 0.01,
        "fees_enabled": False,
        "fee_rate": 0.0,
        "fee_exponent": 1.0,
        "fee_taker_only": True,
        "fee_per_share": 0.0,
        "modeled_slippage_per_share": 0.01,
        "all_in_cost_per_share": cost,
        "break_even_probability": cost,
        "model_edge_per_share": probability - cost,
        "robust_edge_per_share": probability - 0.03 - cost,
        "expected_roi": (probability - cost) / cost,
        "robust_expected_roi": (probability - 0.03 - cost) / cost,
        "resolved_side": resolved,
        "executable": True,
        "eligible": False,
        "selected": False,
        "realized_net_per_share": (1.0 if resolved else 0.0) - cost,
        "source_timestamp": decision_time - timedelta(seconds=1),
        "quote_age_seconds": 1.0,
        "quality_flags": [],
        "rejection_reasons": [],
        "label_available_at": label_available_at,
    }


def test_market_offset_preserves_yes_no_complementarity():
    yes = market_offset_probability(0.31, 0.18, 0.37)
    no = market_offset_probability(0.69, 0.82, 0.37)

    assert np.isclose(yes + no, 1.0)


def test_training_weights_total_one_per_event_date():
    rows = [_training_row(0), _training_row(0), _training_row(1)]
    weights = _event_date_weights(rows)

    assert np.isclose(weights[:2].sum(), 1.0)
    assert np.isclose(weights[2], 1.0)


def test_contract_training_uses_one_yes_label_and_rejects_crossed_market_input():
    origin = datetime(2026, 6, 1, tzinfo=UTC)
    candidates = [
        _candidate(0, side, label_available_at=origin - timedelta(days=1))
        for side in ("YES", "NO")
    ]

    assert len(contract_training_rows(candidates)) == 1

    for candidate in candidates:
        candidate["quality_flags"] = ["crossed_yes_book"]
    assert contract_training_rows(candidates) == []


def test_contract_training_requires_a_causal_non_null_source_timestamp():
    origin = datetime(2026, 6, 1, tzinfo=UTC)
    candidates = [
        _candidate(0, side, label_available_at=origin - timedelta(days=1))
        for side in ("YES", "NO")
    ]

    candidates[0]["source_timestamp"] = None
    assert contract_training_rows(candidates) == []

    candidates[0]["source_timestamp"] = candidates[0][
        "decision_time"
    ] + timedelta(microseconds=1)
    assert contract_training_rows(candidates) == []


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("decision_time", "decision time"),
        ("label_available_at", "label availability time"),
    ),
)
def test_fit_rejects_rows_not_strictly_before_origin(field: str, message: str):
    rows = [_training_row(index) for index in range(30)]
    origin = datetime(2026, 7, 1, tzinfo=UTC)
    rows[0] = replace(rows[0], **{field: origin})

    with pytest.raises(ValueError, match=message):
        fit_market_offset(rows, origin_time=origin, bootstrap_iterations=100)


@pytest.mark.parametrize("source_timestamp", (None, datetime(2026, 7, 2, tzinfo=UTC)))
def test_fit_rejects_missing_or_future_source_time(source_timestamp: datetime | None):
    rows = [_training_row(index) for index in range(30)]
    rows[0] = replace(rows[0], source_timestamp=source_timestamp)

    with pytest.raises(ValueError, match="source time"):
        fit_market_offset(
            rows,
            origin_time=datetime(2026, 7, 1, tzinfo=UTC),
            bootstrap_iterations=100,
        )


def test_fit_rejects_unsupported_decision_hour():
    rows = [_training_row(index) for index in range(30)]
    rows[0] = replace(rows[0], decision_hour_local=6)

    with pytest.raises(ValueError, match="midnight and noon"):
        fit_market_offset(
            rows,
            origin_time=datetime(2026, 7, 1, tzinfo=UTC),
            bootstrap_iterations=100,
        )


def test_block_bootstrap_fit_is_deterministic_and_bounded():
    rows = [_training_row(index) for index in range(35)]
    origin = datetime(2026, 7, 1, tzinfo=UTC)

    first = fit_market_offset(rows, origin_time=origin, bootstrap_iterations=100)
    second = fit_market_offset(rows, origin_time=origin, bootstrap_iterations=100)

    assert first.converged
    assert all(0 <= value <= 1 for value in first.coefficients)
    assert np.allclose(first.coefficients, second.coefficients)
    assert np.allclose(first.bootstrap_coefficients, second.bootstrap_coefficients)


def test_robust_bound_combines_weather_and_coefficient_uncertainty(monkeypatch):
    current_index = 30
    current_origin = datetime.combine(
        date(2026, 4, 1) + timedelta(days=current_index), datetime.min.time(), UTC
    )
    candidates = []
    for index in range(current_index + 1):
        label_available_at = (
            current_origin - timedelta(seconds=1)
            if index < current_index
            else current_origin + timedelta(days=1)
        )
        for side in ("YES", "NO"):
            candidate = _candidate(
                index, side, label_available_at=label_available_at
            )
            if index == current_index and side == "NO":
                candidate["all_in_cost_per_share"] = 0.12
            candidates.append(candidate)

    def fixed_fit(
        rows: list[ResidualTrainingRow],
        *,
        origin_time: datetime,
        bootstrap_iterations: int,
    ) -> MarketOffsetFit:
        return MarketOffsetFit(
            opportunity_fit_id="fixed-fit",
            origin_time=origin_time,
            latest_label_available_at=max(row.label_available_at for row in rows),
            training_start=min(row.event_date for row in rows),
            training_end=max(row.event_date for row in rows),
            training_event_days=len({row.event_date for row in rows}),
            training_rows=len(rows),
            coefficients=(0.5, 0.5),
            bootstrap_coefficients=np.full((bootstrap_iterations, 2), 0.5),
            converged=True,
            fit_metrics={},
        )

    monkeypatch.setattr(
        "nyc_temperature_model.residual_opportunity.fit_market_offset", fixed_fit
    )
    scored, _ = apply_rolling_market_offset(candidates, bootstrap_iterations=100)

    for side, weather, weather_lower, market in (
        ("YES", 0.30, 0.27, 0.15),
        ("NO", 0.70, 0.67, 0.85),
    ):
        current = next(
            row
            for row in scored
            if row["event_date"]
            == date(2026, 4, 1) + timedelta(days=current_index)
            and row["side"] == side
        )
        expected_point = market_offset_probability(weather, market, 0.5)
        expected_lower = market_offset_probability(weather_lower, market, 0.5)

        assert np.isclose(current["probability"], expected_point)
        assert np.isclose(current["probability_lower"], expected_lower)
        assert current["probability_lower"] < current["probability"]


def test_current_outcome_cannot_change_rolling_prediction():
    current_index = 31
    current_origin = datetime.combine(
        date(2026, 4, 1) + timedelta(days=current_index), datetime.min.time(), UTC
    )
    candidates = []
    for index in range(current_index + 1):
        label_available_at = (
            current_origin - timedelta(seconds=1)
            if index < current_index
            else current_origin + timedelta(days=1)
        )
        for side in ("YES", "NO"):
            candidates.append(
                _candidate(index, side, label_available_at=label_available_at)
            )
    altered = [dict(candidate) for candidate in candidates]
    for candidate in altered:
        if candidate["event_date"] == date(2026, 4, 1) + timedelta(days=current_index):
            candidate["resolved_side"] = not candidate["resolved_side"]

    scored, fits = apply_rolling_market_offset(candidates, bootstrap_iterations=100)
    altered_scored, altered_fits = apply_rolling_market_offset(
        altered, bootstrap_iterations=100
    )
    current = next(
        row
        for row in scored
        if row["event_date"] == date(2026, 4, 1) + timedelta(days=current_index)
        and row["side"] == "YES"
    )
    altered_current = next(
        row
        for row in altered_scored
        if row["event_date"] == current["event_date"] and row["side"] == "YES"
    )

    assert len(fits) == len(altered_fits) == 1
    assert current["probability_source"] == "residual_market_offset"
    assert np.isclose(current["probability"], altered_current["probability"])
    assert np.isclose(current["probability_lower"], altered_current["probability_lower"])


def test_label_at_origin_is_not_available_for_warmup():
    current_index = 30
    current_origin = datetime.combine(
        date(2026, 4, 1) + timedelta(days=current_index), datetime.min.time(), UTC
    )
    candidates = []
    for index in range(current_index + 1):
        label_available_at = (
            current_origin
            if index == current_index - 1
            else (
                current_origin - timedelta(seconds=1)
                if index < current_index
                else current_origin + timedelta(days=1)
            )
        )
        for side in ("YES", "NO"):
            candidates.append(
                _candidate(index, side, label_available_at=label_available_at)
            )

    scored, fits = apply_rolling_market_offset(candidates, bootstrap_iterations=100)
    current = next(
        row
        for row in scored
        if row["event_date"] == date(2026, 4, 1) + timedelta(days=current_index)
        and row["side"] == "YES"
    )

    assert fits == []
    assert current["probability_source"] == "residual_model_warmup"
    assert "residual_model_warmup" in current["rejection_reasons"]
