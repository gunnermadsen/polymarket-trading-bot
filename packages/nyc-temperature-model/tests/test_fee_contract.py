import pytest

from nyc_temperature_model.fees import fee_schedule_from_market, taker_fee_per_share


def test_weather_fee_schedule_and_order_rounding_are_exact():
    schedule = fee_schedule_from_market(
        {
            "feesEnabled": True,
            "feeSchedule": {"rate": 0.05, "exponent": 1, "takerOnly": True},
        }
    )

    assert schedule.enabled
    assert schedule.rate == 0.05
    assert schedule.exponent == 1
    assert schedule.taker_only
    assert taker_fee_per_share(
        0.12,
        quantity=5,
        enabled=schedule.enabled,
        rate=schedule.rate,
        exponent=schedule.exponent,
    ) == pytest.approx(0.00528)


def test_fee_enabled_market_without_a_rate_fails_closed():
    with pytest.raises(ValueError, match="positive fee rate"):
        fee_schedule_from_market({"feesEnabled": True, "feeSchedule": {}})


def test_fee_free_market_has_zero_taker_fee():
    schedule = fee_schedule_from_market({"feesEnabled": False})
    assert taker_fee_per_share(
        0.50,
        quantity=5,
        enabled=schedule.enabled,
        rate=schedule.rate,
        exponent=schedule.exponent,
    ) == 0


def test_unknown_fee_exponent_fails_closed():
    with pytest.raises(ValueError, match="unsupported fee exponent"):
        fee_schedule_from_market(
            {
                "feesEnabled": True,
                "feeSchedule": {"rate": 0.05, "exponent": 2},
            }
        )
