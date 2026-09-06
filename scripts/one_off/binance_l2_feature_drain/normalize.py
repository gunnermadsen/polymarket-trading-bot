from __future__ import annotations

from typing import Any

from .contract import CONTRACT_COLUMNS, FACT_COLUMNS


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    record = {column: row[column] for column in CONTRACT_COLUMNS}
    record["artifact_id"] = str(record["artifact_id"])
    return record


def identity(record: dict[str, Any]) -> tuple[Any, Any]:
    return record["symbol"], record["second_start"]


def factual_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[column] for column in FACT_COLUMNS)
