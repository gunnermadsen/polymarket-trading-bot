from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any


def canonical_market_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[row["event_date"]].append(row)

    selected: list[dict[str, Any]] = []
    for event_date, day_rows in sorted(by_date.items()):
        partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in day_rows:
            partitions[str(row["event_id"])].append(row)
        if len(partitions) == 1:
            selected.extend(next(iter(partitions.values())))
            continue

        canonical = [
            partition
            for partition in partitions.values()
            if not str(partition[0]["event_slug"]).startswith("arch-")
        ]
        if len(canonical) != 1:
            slugs = sorted(str(partition[0]["event_slug"]) for partition in partitions.values())
            raise ValueError(
                f"event date {event_date} has no unique canonical market partition: {slugs}"
            )
        selected.extend(canonical[0])
    return sorted(selected, key=lambda row: (row["event_date"], row["market_id"]))
