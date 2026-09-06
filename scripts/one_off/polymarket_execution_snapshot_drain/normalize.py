from __future__ import annotations

from typing import Any

from .contract import (
    CONTRACT_COLUMNS, EXPANDED_VWAP_COLUMNS, MEASUREMENT_COLUMNS,
    SHARED_FACT_COLUMNS,
)


def normalize(row: dict[str, Any], has_expanded_vwap: bool) -> dict[str, Any]:
    record = {
        column: row.get(column) if has_expanded_vwap or column not in EXPANDED_VWAP_COLUMNS else None
        for column in CONTRACT_COLUMNS
    }
    record["artifact_id"] = str(record["artifact_id"])
    return record


def identity(record: dict[str, Any]) -> tuple[Any, Any]:
    return record["market_id"], record["sampled_at"]


def shared_fact(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[column] for column in SHARED_FACT_COLUMNS)


def expanded_fact(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[column] for column in EXPANDED_VWAP_COLUMNS)


def compatible_measurements(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left[column] is None
        or right[column] is None
        or left[column] == right[column]
        for column in MEASUREMENT_COLUMNS
    )
