from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from nyc_temperature_model.hrrr_ingestion import (
    _download_field,
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


def test_hrrr_field_retry_bounds_verified_archive_misses_by_source_count():
    calls = []
    sleeps = []

    def operation(priority, overwrite):
        calls.append((priority, overwrite))
        raise FileNotFoundError("archive field is unavailable")

    with pytest.raises(FileNotFoundError, match="archive field is unavailable"):
        _run_with_retry(
            operation,
            attempts=8,
            sources=("google", "aws", "nomads"),
            retry_base_ms=1000,
            retry_max_ms=30000,
            sleep=sleeps.append,
            jitter=lambda _start, _end: 0,
        )

    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_hrrr_download_classifies_a_missing_grib_as_an_archive_gap(tmp_path):
    calls = []

    def herbie_factory(*_args, **kwargs):
        calls.append(kwargs["priority"])
        return SimpleNamespace(grib=None)

    settings = SimpleNamespace(
        cache_directory=tmp_path,
        hrrr_download_attempts=8,
        hrrr_source_priority=("google", "aws", "nomads"),
        hrrr_retry_base_ms=0,
        hrrr_retry_max_ms=0,
        hrrr_request_interval_ms=0,
    )

    with pytest.raises(FileNotFoundError, match="2019-03-11T12:00Z f08"):
        _download_field(
            settings,
            herbie_factory,
            datetime(2019, 3, 11, 12, tzinfo=UTC),
            8,
            ":TMP:2 m above ground",
        )

    assert calls == [
        ["google", "aws", "nomads"],
        ["aws", "nomads", "google"],
        ["nomads", "google", "aws"],
    ]
