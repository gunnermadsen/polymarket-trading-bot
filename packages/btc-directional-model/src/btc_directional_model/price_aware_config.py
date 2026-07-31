from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from .core_config import load_core_config, parse_utc_day
from .core_extract import file_sha256

PRICE_AWARE_DECISION_SECONDS = tuple(range(60, 241, 5))
PRICE_AWARE_CONTEXT_SECONDS = tuple(range(55, 241, 5))
PRICE_AWARE_PRICE_BANDS = (
    ("under_0_50", 0.00, 0.50),
    ("0_50_to_0_60", 0.50, 0.60),
    ("0_60_to_0_70", 0.60, 0.70),
    ("0_70_to_0_80", 0.70, 0.80),
    ("0_80_to_0_90", 0.80, 0.90),
    ("0_90_and_over", 0.90, 1.01),
)


@dataclass(frozen=True)
class PriceAwareBlockConfig:
    name: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class PriceAwareWalkForwardConfig:
    history_start: datetime
    outcome_calibration_fraction: float
    threshold_block_names: tuple[str, ...]
    evaluation_block_name: str
    blocks: tuple[PriceAwareBlockConfig, ...]


@dataclass(frozen=True)
class PriceAwareModelConfig:
    quantity: float
    freshness_seconds: int
    recency_half_life_days: float | None
    boundary_confidence_threshold: float
    value_thresholds: tuple[float, ...]


@dataclass(frozen=True)
class PriceAwareGateConfig:
    target_accuracy: float
    minimum_accuracy: float
    minimum_wilson_lower: float
    minimum_coverage: float
    minimum_evaluation_trades: int
    minimum_fold_trades: int
    minimum_fold_direction_trades: int
    minimum_profit_factor: float
    maximum_expected_calibration_error: float
    maximum_selected_net_bias: float


@dataclass(frozen=True)
class PriceAwarePathConfig:
    execution_evidence: Path
    prewindow_features: Path
    runs: Path


@dataclass(frozen=True)
class PriceAwareBenchmarkConfig:
    source_path: Path
    package_root: Path
    core_config: Path
    core_config_sha256: str
    evaluation_note: str
    walk_forward: PriceAwareWalkForwardConfig
    model: PriceAwareModelConfig
    gates: PriceAwareGateConfig
    paths: PriceAwarePathConfig


_EXPECTED_WALK_FORWARD_BLOCKS = (
    ("initial_book_history", datetime(2026, 4, 13, tzinfo=UTC), datetime(2026, 5, 26, tzinfo=UTC)),
    ("validation_may26", datetime(2026, 5, 26, tzinfo=UTC), datetime(2026, 6, 2, tzinfo=UTC)),
    ("validation_jun02", datetime(2026, 6, 2, tzinfo=UTC), datetime(2026, 6, 9, tzinfo=UTC)),
    ("validation_jun09", datetime(2026, 6, 9, tzinfo=UTC), datetime(2026, 7, 3, tzinfo=UTC)),
    ("validation_jul03", datetime(2026, 7, 3, tzinfo=UTC), datetime(2026, 7, 14, tzinfo=UTC)),
    ("confirmation_jul14", datetime(2026, 7, 14, tzinfo=UTC), datetime(2026, 7, 29, tzinfo=UTC)),
)
_EXPECTED_THRESHOLD_BLOCK_NAMES = (
    "validation_jun02",
    "validation_jun09",
    "validation_jul03",
)
_EXPECTED_EVALUATION_BLOCK_NAME = "confirmation_jul14"
_EXPECTED_HISTORY_START = datetime(2026, 3, 21, tzinfo=UTC)
_EXPECTED_OUTCOME_CALIBRATION_FRACTION = 0.20
_EXPECTED_VALUE_THRESHOLDS = (0.0,)


def load_price_aware_benchmark_config(path: Path) -> PriceAwareBenchmarkConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    benchmark = raw["benchmark"]
    walk_forward = raw["walk_forward"]
    model = raw["model"]
    gates = raw["gates"]
    paths = raw["paths"]
    config = PriceAwareBenchmarkConfig(
        source_path=source_path,
        package_root=package_root,
        core_config=package_root / str(benchmark["core_config"]),
        core_config_sha256=str(benchmark["core_config_sha256"]).lower(),
        evaluation_note=str(benchmark["evaluation_note"]).strip(),
        walk_forward=PriceAwareWalkForwardConfig(
            history_start=parse_utc_day(walk_forward["history_start"]),
            outcome_calibration_fraction=float(walk_forward["outcome_calibration_fraction"]),
            threshold_block_names=tuple(
                str(value) for value in walk_forward["threshold_block_names"]
            ),
            evaluation_block_name=str(walk_forward["evaluation_block_name"]),
            blocks=tuple(
                PriceAwareBlockConfig(
                    name=str(block["name"]),
                    start=parse_utc_day(block["start"]),
                    end=parse_utc_day(block["end"]),
                )
                for block in walk_forward["blocks"]
            ),
        ),
        model=PriceAwareModelConfig(
            quantity=float(model["quantity"]),
            freshness_seconds=int(model["freshness_seconds"]),
            recency_half_life_days=None,
            boundary_confidence_threshold=float(model["boundary_confidence_threshold"]),
            value_thresholds=tuple(float(value) for value in model["value_thresholds"]),
        ),
        gates=PriceAwareGateConfig(
            target_accuracy=float(gates["target_accuracy"]),
            minimum_accuracy=float(gates["minimum_accuracy"]),
            minimum_wilson_lower=float(gates["minimum_wilson_lower"]),
            minimum_coverage=float(gates["minimum_coverage"]),
            minimum_evaluation_trades=int(gates["minimum_evaluation_trades"]),
            minimum_fold_trades=int(gates["minimum_fold_trades"]),
            minimum_fold_direction_trades=int(gates["minimum_fold_direction_trades"]),
            minimum_profit_factor=float(gates["minimum_profit_factor"]),
            maximum_expected_calibration_error=float(gates["maximum_expected_calibration_error"]),
            maximum_selected_net_bias=float(gates["maximum_selected_net_bias"]),
        ),
        paths=PriceAwarePathConfig(
            execution_evidence=package_root / str(paths["execution_evidence"]),
            prewindow_features=package_root / str(paths["prewindow_features"]),
            runs=package_root / str(paths["runs"]),
        ),
    )
    validate_price_aware_benchmark_config(config)
    return config


def validate_price_aware_benchmark_config(
    config: PriceAwareBenchmarkConfig,
) -> None:
    if not config.core_config.is_file():
        raise ValueError(f"core config is missing: {config.core_config}")
    if len(config.core_config_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in config.core_config_sha256
    ):
        raise ValueError("core_config_sha256 must be 64 lowercase hexadecimal digits")
    if file_sha256(config.core_config) != config.core_config_sha256:
        raise ValueError("pinned core training config hash mismatch")
    if not config.evaluation_note:
        raise ValueError("evaluation_note must be non-empty")

    core_config = load_core_config(config.core_config)
    expected_core_timing = (
        PRICE_AWARE_DECISION_SECONDS[0],
        PRICE_AWARE_DECISION_SECONDS[-1],
        5,
    )
    observed_core_timing = (
        core_config.data.min_seconds_after_open,
        300 - core_config.data.min_seconds_before_close,
        core_config.data.sample_interval_seconds,
    )
    if observed_core_timing != expected_core_timing:
        raise ValueError("price-aware core features must cover 60-240 seconds at 5s")

    walk_forward = config.walk_forward
    blocks = walk_forward.blocks
    if not blocks:
        raise ValueError("price-aware walk-forward blocks must be non-empty")
    if any(block.start >= block.end for block in blocks):
        raise ValueError("every price-aware walk-forward block must have a positive range")
    if any(previous.end != current.start for previous, current in pairwise(blocks)):
        raise ValueError("price-aware walk-forward blocks must be chronological and contiguous")
    names = tuple(block.name for block in blocks)
    if len(set(names)) != len(names):
        raise ValueError("price-aware walk-forward block names must be unique")
    observed_blocks = tuple((block.name, block.start, block.end) for block in blocks)
    if observed_blocks != _EXPECTED_WALK_FORWARD_BLOCKS:
        raise ValueError("price-aware walk-forward block names and ranges must remain frozen")
    if walk_forward.threshold_block_names != _EXPECTED_THRESHOLD_BLOCK_NAMES:
        raise ValueError("price-aware policy blocks must remain on the frozen dates")
    if walk_forward.evaluation_block_name != _EXPECTED_EVALUATION_BLOCK_NAME:
        raise ValueError("price-aware evaluation block must remain the frozen confirmation range")
    if walk_forward.evaluation_block_name in walk_forward.threshold_block_names:
        raise ValueError("the evaluation block cannot select an operating threshold")
    if walk_forward.history_start != _EXPECTED_HISTORY_START:
        raise ValueError("price-aware outcome history must begin March 21, 2026")
    if not math.isclose(
        walk_forward.outcome_calibration_fraction,
        _EXPECTED_OUTCOME_CALIBRATION_FRACTION,
    ):
        raise ValueError("outcome_calibration_fraction must remain 0.20")
    if (
        walk_forward.history_start != core_config.data.range_start
        or blocks[0].start < core_config.data.range_start
        or blocks[-1].end != core_config.data.range_end
    ):
        raise ValueError("price-aware history and evaluation must preserve the core data range")

    model = config.model
    if not math.isclose(model.quantity, 5.0):
        raise ValueError("price-aware training is fixed to five-share execution")
    if model.freshness_seconds != 2:
        raise ValueError("price-aware training requires two-second book freshness")
    if model.recency_half_life_days is not None:
        raise ValueError("initial price-aware training must use market-equal weights without recency weighting")
    if not 0.5 <= model.boundary_confidence_threshold <= 1.0:
        raise ValueError("boundary confidence threshold must be in [0.5, 1]")
    _validate_thresholds("value_thresholds", model.value_thresholds)
    if model.value_thresholds != _EXPECTED_VALUE_THRESHOLDS:
        raise ValueError("value_thresholds must remain on the frozen economic grid")

    gates = config.gates
    probabilities = (
        gates.target_accuracy,
        gates.minimum_accuracy,
        gates.minimum_wilson_lower,
        gates.minimum_coverage,
        gates.maximum_expected_calibration_error,
        gates.maximum_selected_net_bias,
    )
    if any(not 0 <= value <= 1 for value in probabilities):
        raise ValueError("accuracy, coverage, and calibration gates must be in [0, 1]")
    if gates.target_accuracy < gates.minimum_accuracy:
        raise ValueError("target accuracy cannot be below the minimum accuracy")
    if gates.minimum_wilson_lower > gates.minimum_accuracy:
        raise ValueError("minimum Wilson lower bound cannot exceed minimum accuracy")
    if gates.minimum_evaluation_trades <= 0:
        raise ValueError("minimum_evaluation_trades must be positive")
    if gates.minimum_fold_trades <= 0:
        raise ValueError("minimum_fold_trades must be positive")
    if gates.minimum_fold_direction_trades <= 0:
        raise ValueError("minimum_fold_direction_trades must be positive")
    if 2 * gates.minimum_fold_direction_trades > gates.minimum_fold_trades:
        raise ValueError("minimum_fold_trades must accommodate both direction trade minimums")
    if not math.isfinite(gates.minimum_profit_factor) or gates.minimum_profit_factor <= 1:
        raise ValueError("minimum_profit_factor must be finite and exceed one")
    generated_paths = (
        core_config.paths.development_feature_data,
        config.paths.execution_evidence,
        config.paths.prewindow_features,
        config.paths.runs,
    )
    if len(set(generated_paths)) != len(generated_paths):
        raise ValueError("price-aware generated paths must be isolated")


def _validate_thresholds(name: str, values: tuple[float, ...]) -> None:
    if (
        not values
        or values != tuple(sorted(set(values)))
        or any(not math.isfinite(value) or value < 0 for value in values)
    ):
        raise ValueError(f"{name} must be unique, increasing, finite, and nonnegative")
