from datetime import date

import pytest

from nyc_temperature_model.contracts import canonical_market_rows
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


def test_non_archived_partition_wins_only_when_duplicate_exists():
    rows = [
        {
            "event_date": date(2026, 5, 18),
            "event_id": "arch-only",
            "event_slug": "arch-highest-temperature-may-18",
            "market_id": "1",
        },
        {
            "event_date": date(2026, 5, 19),
            "event_id": "arch-duplicate",
            "event_slug": "arch-highest-temperature-may-19",
            "market_id": "2",
        },
        {
            "event_date": date(2026, 5, 19),
            "event_id": "canonical",
            "event_slug": "highest-temperature-may-19",
            "market_id": "3",
        },
    ]

    selected = canonical_market_rows(rows)
    assert [row["market_id"] for row in selected] == ["1", "3"]
