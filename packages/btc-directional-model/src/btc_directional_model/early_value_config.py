"""Configuration for the early-price prediction value benchmark."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path

from .core_config import parse_utc_day


@dataclass(frozen=True)
class EvidenceWindow:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class EarlyValueConfig:
    source_path: Path
    package_root: Path
    core_config: Path
    l2_source: Path
    candle_source: Path
    refprice_source: Path
    price_source_sql: Path
    price_cache: Path
    runs: Path
    champion_model: Path
    fit: EvidenceWindow
    calibration: EvidenceWindow
    policy: EvidenceWindow
    evaluation: EvidenceWindow
    prediction_seconds: tuple[int, ...]
    price_seconds: tuple[int, ...]
    calibration_bands: tuple[tuple[int, int], ...]
    quantity: float
    book_freshness_seconds: int
    minimum_edge_per_share: float
    cheap_price_min: float
    cheap_price_max: float
    confidence_reference: float
    bootstrap_resamples: int
    random_seed: int


def load_early_value_config(path: Path) -> EarlyValueConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    windows = raw["windows"]
    timing = raw["timing"]
    economics = raw["economics"]
    model = raw["model"]
    paths = raw["paths"]
    benchmark = raw["benchmark"]
    if benchmark.get("profile") != "btc_early_price_value_hypothesis_v1":
        raise ValueError("early-value benchmark profile identity changed")
    if benchmark.get("paper_only") is not True or benchmark.get("live_capital_allowed") is not False:
        raise ValueError("early-value benchmark must remain offline and paper-only")

    def window(name: str) -> EvidenceWindow:
        section = windows[name]
        return EvidenceWindow(parse_utc_day(section["start"]), parse_utc_day(section["end"]))

    minimum_prediction = int(timing["prediction_start_second"])
    maximum_prediction = int(timing["prediction_end_second"])
    prediction_cadence = int(timing["prediction_cadence_seconds"])
    prediction_seconds = tuple(range(minimum_prediction, maximum_prediction + 1, prediction_cadence))
    early_price_end = int(timing["early_price_end_second"])
    later_price_cadence = int(timing["later_price_cadence_seconds"])
    price_seconds = (
        *range(1, early_price_end + 1),
        *range(early_price_end + 1, maximum_prediction + 1, later_price_cadence),
    )
    calibration_bands = tuple(
        (int(band["start_second"]), int(band["end_second_exclusive"]))
        for band in model["calibration_bands"]
    )
    config = EarlyValueConfig(
        source_path=source_path,
        package_root=package_root,
        core_config=package_root / str(paths["core_config"]),
        l2_source=package_root / str(paths["l2_source"]),
        candle_source=package_root / str(paths["candle_source"]),
        refprice_source=package_root / str(paths["refprice_source"]),
        price_source_sql=package_root / str(paths["price_source_sql"]),
        price_cache=package_root / str(paths["price_cache"]),
        runs=package_root / str(paths["runs"]),
        champion_model=package_root / str(paths["champion_model"]),
        fit=window("fit"),
        calibration=window("calibration"),
        policy=window("policy"),
        evaluation=window("evaluation"),
        prediction_seconds=prediction_seconds,
        price_seconds=tuple(int(value) for value in price_seconds),
        calibration_bands=calibration_bands,
        quantity=float(economics["quantity"]),
        book_freshness_seconds=int(economics["book_freshness_seconds"]),
        minimum_edge_per_share=float(economics["minimum_edge_per_share"]),
        cheap_price_min=float(economics["cheap_price_min"]),
        cheap_price_max=float(economics["cheap_price_max"]),
        confidence_reference=float(economics["confidence_reference"]),
        bootstrap_resamples=int(model["bootstrap_resamples"]),
        random_seed=int(model["random_seed"]),
    )
    validate_early_value_config(config)
    return config


def validate_early_value_config(config: EarlyValueConfig) -> None:
    windows = (config.fit, config.calibration, config.policy, config.evaluation)
    if any(item.start >= item.end for item in windows):
        raise ValueError("every evidence window must have positive duration")
    if any(left.end != right.start for left, right in pairwise(windows)):
        raise ValueError("fit, calibration, policy, and evaluation windows must be contiguous")
    if config.prediction_seconds != tuple(range(5, 241, 5)):
        raise ValueError("predictions must cover 5 through 240 seconds every five seconds")
    expected_prices = (*range(1, 60), *range(60, 241, 5))
    if config.price_seconds != expected_prices:
        raise ValueError("prices must cover seconds 1-59 and then 60-240 every five seconds")
    if config.calibration_bands != (
        (5, 15), (15, 30), (30, 45), (45, 60),
        (60, 90), (90, 120), (120, 180), (180, 241),
    ):
        raise ValueError("time calibration bands changed")
    if config.quantity != 5.0 or config.book_freshness_seconds != 2:
        raise ValueError("economic evidence is fixed to five shares and two-second freshness")
    if not 0.0 <= config.minimum_edge_per_share < 1.0:
        raise ValueError("minimum edge per share must be in [0, 1)")
    if not 0.0 < config.cheap_price_min < config.cheap_price_max < 1.0:
        raise ValueError("cheap-price interval must be strictly inside (0, 1)")
    if config.confidence_reference != 0.89:
        raise ValueError("the existing 89 percent rule is a fixed reference policy")
    if config.bootstrap_resamples < 1000:
        raise ValueError("economic bootstrap requires at least 1000 resamples")
    required = (
        config.core_config,
        config.l2_source,
        config.candle_source,
        config.price_source_sql,
        config.champion_model,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required early-value evidence is missing: " + ", ".join(missing))
