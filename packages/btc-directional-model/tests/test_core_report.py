from btc_directional_model.core_report import holdout_evaluation_label


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
