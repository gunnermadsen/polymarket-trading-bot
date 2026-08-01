import pytest

from nyc_temperature_model.market_ingestion import (
    _eligible_resolution_source,
    parse_bucket,
    validate_bucket_partition,
)


def test_temperature_bucket_parser_handles_all_shapes():
    assert parse_bucket("Will the highest temperature be 66-67°F?") == (66, 67)
    assert parse_bucket("Will it be 65°F or below?") == (None, 65)
    assert parse_bucket("Will it be 74°F or higher?") == (74, None)
    assert parse_bucket("Will the temperature be 70°F?") == (70, 70)


def test_temperature_buckets_must_form_one_contiguous_partition():
    valid = [
        {"bucket_lower_f": None, "bucket_upper_f": 68},
        {"bucket_lower_f": 69, "bucket_upper_f": 71},
        {"bucket_lower_f": 72, "bucket_upper_f": None},
    ]
    validate_bucket_partition(valid)
    invalid = [valid[0], {"bucket_lower_f": 70, "bucket_upper_f": 71}, valid[2]]
    with pytest.raises(ValueError, match="contiguous"):
        validate_bucket_partition(invalid)


def test_historical_event_without_canonical_source_is_ineligible():
    event = {"resolutionSource": ""}
    assert not _eligible_resolution_source(event, {"resolutionSource": ""})
    assert _eligible_resolution_source(
        event,
        {
            "resolutionSource": (
                "https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA"
            )
        },
    )
