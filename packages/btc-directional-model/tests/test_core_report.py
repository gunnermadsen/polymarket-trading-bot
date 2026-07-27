from btc_directional_model.core_report import (
    holdout_evaluation_label,
    holdout_handling_note,
)


def test_holdout_label_uses_frozen_range() -> None:
    metrics = {
        "freeze": {
            "holdout_range": {
                "start": "2026-07-14T00:00:00+00:00",
                "end": "2026-07-21T00:00:00+00:00",
            }
        }
    }

    assert holdout_evaluation_label(metrics) == "Untouched July 14–20 holdout"


def test_holdout_label_handles_month_boundary() -> None:
    metrics = {
        "freeze": {
            "holdout_range": {
                "start": "2026-06-29T00:00:00+00:00",
                "end": "2026-07-06T00:00:00+00:00",
            }
        }
    }

    assert holdout_evaluation_label(metrics) == "Untouched June 29–July 5 holdout"


def test_holdout_note_distinguishes_qualified_unopened_candidate() -> None:
    assert (
        holdout_handling_note({"holdout": None, "ready_for_holdout": True})
        == "The candidate passed every pre-holdout gate; the isolated holdout "
        "was deliberately not opened during this development-only run."
    )


def test_holdout_note_reports_failed_prequalification() -> None:
    assert (
        holdout_handling_note({"holdout": None, "ready_for_holdout": False})
        == "The holdout was not opened because pre-holdout qualification did not pass."
    )


def test_holdout_note_reports_single_evaluation() -> None:
    assert (
        holdout_handling_note({"holdout": {"metrics": {}}})
        == "The frozen candidate was evaluated exactly once on the isolated holdout."
    )
