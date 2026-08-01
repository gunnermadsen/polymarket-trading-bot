from datetime import UTC, datetime

import pytest

from nyc_temperature_model.hrrr_ingestion import (
    _model_run_for_decision,
    _run_with_retry,
    _valid_times,
)


def test_hrrr_decisions_use_available_extended_synoptic_cycles():
    assert _model_run_for_decision(datetime(2026, 7, 4, 4, tzinfo=UTC), 75) == datetime(
        2026, 7, 4, 0, tzinfo=UTC
    )
    assert _model_run_for_decision(datetime(2026, 7, 4, 16, tzinfo=UTC), 75) == datetime(
        2026, 7, 4, 12, tzinfo=UTC
    )


def test_hrrr_valid_times_cover_the_remaining_local_day():
    midnight = datetime(2026, 7, 4, 4, tzinfo=UTC)
    noon = datetime(2026, 7, 4, 16, tzinfo=UTC)
    assert len(_valid_times(midnight, datetime(2026, 7, 4, 0, tzinfo=UTC))) == 24
    assert len(_valid_times(noon, datetime(2026, 7, 4, 12, tzinfo=UTC))) == 12


def test_hrrr_field_retry_rotates_sources_and_uses_backoff():
    calls = []
    sleeps = []

    def operation(priority, overwrite):
        calls.append((priority, overwrite))
        if len(calls) < 3:
            raise RuntimeError("HTTPSConnectionPool connection broken: IncompleteRead")
        return "downloaded"

    result = _run_with_retry(
        operation,
        attempts=4,
        sources=("google", "aws", "nomads"),
        retry_base_ms=1000,
        retry_max_ms=10000,
        sleep=sleeps.append,
        jitter=lambda _start, _end: 0,
    )

    assert result == "downloaded"
    assert calls == [
        (("google", "aws", "nomads"), False),
        (("aws", "nomads", "google"), True),
        (("nomads", "google", "aws"), True),
    ]
    assert sleeps == [1, 2]


def test_hrrr_field_retry_does_not_hide_deterministic_errors():
    with pytest.raises(ValueError, match="invalid field"):
        _run_with_retry(
            lambda _priority, _overwrite: (_ for _ in ()).throw(ValueError("invalid field")),
            attempts=8,
            sources=("google", "aws"),
            retry_base_ms=1000,
            retry_max_ms=30000,
            sleep=lambda _delay: None,
        )
