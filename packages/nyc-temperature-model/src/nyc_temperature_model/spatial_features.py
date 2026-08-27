from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from . import STATION_LATITUDE, STATION_LONGITUDE

RADII_KM = (25, 50, 100)
SECTORS = ("all", "north", "south", "east", "west")


def distance_components_km(
    latitudes: np.ndarray, longitudes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    north = (np.asarray(latitudes, dtype=float) - STATION_LATITUDE) * 111.195
    east = (
        (np.asarray(longitudes, dtype=float) - STATION_LONGITUDE)
        * 111.195
        * math.cos(math.radians(STATION_LATITUDE))
    )
    return north, east, np.hypot(north, east)


def spatial_mask(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    radius_km: int,
    sector: str,
) -> np.ndarray:
    north, east, distance = distance_components_km(latitudes, longitudes)
    mask = distance <= radius_km
    if sector == "north":
        mask &= north >= np.abs(east)
    elif sector == "south":
        mask &= -north > np.abs(east)
    elif sector == "east":
        mask &= east >= np.abs(north)
    elif sector == "west":
        mask &= -east > np.abs(north)
    elif sector != "all":
        raise ValueError(f"unsupported sector: {sector}")
    return mask


def numeric_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    selected = np.asarray(values, dtype=float)[mask]
    valid = selected[np.isfinite(selected)]
    total = int(selected.size)
    if not valid.size:
        return {
            "mean": None,
            "stddev": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "max": None,
            "valid_pixel_fraction": 0.0,
        }
    return {
        "mean": float(np.mean(valid)),
        "stddev": float(np.std(valid)),
        "p10": float(np.quantile(valid, 0.1)),
        "p50": float(np.quantile(valid, 0.5)),
        "p90": float(np.quantile(valid, 0.9)),
        "max": float(np.max(valid)),
        "valid_pixel_fraction": float(valid.size / total) if total else 0.0,
    }


def quality_flag_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(values)[mask]
    selected = selected[np.isfinite(selected)]
    if not selected.size:
        return {"valid": 0, "counts": {}}
    unique, counts = np.unique(selected.astype(int), return_counts=True)
    return {
        "valid": int(selected.size),
        "counts": {str(int(key)): int(value) for key, value in zip(unique, counts)},
    }


def first_data_variable(dataset: Any, names: Iterable[str]):
    for name in names:
        if name in dataset.data_vars:
            return dataset[name]
    normalized = {name.lower(): name for name in dataset.data_vars}
    for name in names:
        if name.lower() in normalized:
            return dataset[normalized[name.lower()]]
    raise ValueError(f"none of the expected variables are present: {', '.join(names)}")
