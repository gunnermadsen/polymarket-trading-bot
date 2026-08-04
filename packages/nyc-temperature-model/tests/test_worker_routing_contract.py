import pytest

from nyc_temperature_model.config import SUPPORTED_INGESTERS, Settings


def test_worker_ingesters_default_to_every_supported_ingester(monkeypatch):
    monkeypatch.delenv("WEATHER_WORKER_INGESTERS", raising=False)

    assert Settings.from_env().worker_ingesters == SUPPORTED_INGESTERS


def test_worker_ingesters_are_deduplicated_and_ordered(monkeypatch):
    monkeypatch.setenv(
        "WEATHER_WORKER_INGESTERS",
        "hrrr_point_forecasts, polymarket_temperature_price_history,hrrr_point_forecasts",
    )

    assert Settings.from_env().worker_ingesters == (
        "hrrr_point_forecasts",
        "polymarket_temperature_price_history",
    )


@pytest.mark.parametrize("value", ["", "unknown_ingester"])
def test_worker_ingesters_reject_empty_or_unsupported_routes(monkeypatch, value):
    monkeypatch.setenv("WEATHER_WORKER_INGESTERS", value)

    with pytest.raises(ValueError, match="WEATHER_WORKER_INGESTERS"):
        Settings.from_env()
