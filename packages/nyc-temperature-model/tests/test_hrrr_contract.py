from datetime import UTC, datetime

from nyc_temperature_model.hrrr_ingestion import _model_run_for_decision


def test_hrrr_decisions_use_available_extended_synoptic_cycles():
    assert _model_run_for_decision(datetime(2026, 7, 4, 4, tzinfo=UTC), 75) == datetime(
        2026, 7, 4, 0, tzinfo=UTC
    )
    assert _model_run_for_decision(datetime(2026, 7, 4, 16, tzinfo=UTC), 75) == datetime(
        2026, 7, 4, 12, tzinfo=UTC
    )
