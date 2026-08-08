"""Configuration for the BTC asymmetric-value hunter benchmark."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path

from .core_config import load_core_config, parse_utc_day

LEGACY_TRAINING_CONTRACT = "legacy_four_window"
TARGET_CALIBRATED_TRAINING_CONTRACT = "early_price_target_calibrated"
SUPPORTED_TRAINING_CONTRACTS = frozenset(
    {LEGACY_TRAINING_CONTRACT, TARGET_CALIBRATED_TRAINING_CONTRACT}
)


@dataclass(frozen=True)
class EvidenceWindow:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ValuePolicy:
    name: str
    selection_eligible: bool
    maximum_entry_second: int
    minimum_share_price: float
    maximum_share_price: float
    maximum_cost_per_share: float
    minimum_edge_per_share: float


@dataclass(frozen=True)
class ValueGates:
    minimum_calibration_markets_per_band: int
    minimum_calibration_markets_per_cell: int
    minimum_calibration_days_per_cell: int
    minimum_policy_strict_markets: int
    minimum_policy_executable_days: int
    minimum_policy_source_grid_coverage: float
    minimum_policy_strict_grid_coverage: float
    minimum_policy_candidate_grid_coverage: float
    minimum_evaluation_strict_markets: int
    minimum_evaluation_executable_days: int
    minimum_evaluation_source_grid_coverage: float
    minimum_evaluation_strict_grid_coverage: float
    minimum_evaluation_candidate_grid_coverage: float
    minimum_policy_trades: int
    minimum_evaluation_trades: int
    minimum_profit_factor: float
    minimum_net_expectancy_per_trade: float
    minimum_capital_efficiency: float
    minimum_stress_expectancy_per_trade: float
    maximum_mean_cost_per_share: float
    maximum_mean_share_price: float
    maximum_selected_calibration_bias: float
    maximum_loss_recovery_wins: float
    maximum_average_loss: float
    maximum_single_loss: float
    minimum_side_trades: int
    minimum_pre60_trades: int
    minimum_20_30c_trades: int


@dataclass(frozen=True)
class TargetCalibrationContract:
    minimum_price: float
    maximum_price: float
    sides: tuple[str, ...]
    time_bands: tuple[tuple[int, int], ...]
    required_fitted_cells: int


@dataclass(frozen=True)
class AsymmetricValueConfig:
    source_path: Path
    package_root: Path
    training_contract: str
    fit: EvidenceWindow
    calibration: EvidenceWindow
    policy: EvidenceWindow
    evaluation: EvidenceWindow | None
    prediction_seconds: tuple[int, ...]
    price_seconds: tuple[int, ...]
    calibration_bands: tuple[tuple[int, int], ...]
    quantity: float
    maximum_depth_participation: float
    book_freshness_seconds: int
    execution_reserve_per_share: float
    confidence_control_minimum_edge_per_share: float
    confidence_thresholds: tuple[float, ...]
    policies: tuple[ValuePolicy, ...]
    gates: ValueGates
    random_seed: int
    bootstrap_resamples: int
    calibration_identity_l2: float
    target_calibration: TargetCalibrationContract | None
    core_config: Path
    oracle_source: Path
    l2_source: Path
    candle_source: Path
    price_source_sql: Path
    price_cache: Path
    feature_cache: Path
    runs: Path
    champion_model: Path
    champion_process: Path


def load_asymmetric_value_config(path: Path) -> AsymmetricValueConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    benchmark = raw["benchmark"]
    if benchmark.get("profile") != "btc_asymmetric_value_hunter":
        raise ValueError("asymmetric-value benchmark profile identity changed")
    if benchmark.get("paper_only") is not True or benchmark.get("live_capital_allowed") is not False:
        raise ValueError("asymmetric-value benchmark must remain offline and paper-only")
    training_contract = str(
        benchmark.get("training_contract", LEGACY_TRAINING_CONTRACT)
    )

    def window(name: str) -> EvidenceWindow:
        values = raw["windows"][name]
        return EvidenceWindow(
            start=parse_utc_day(values["start"]),
            end=parse_utc_day(values["end"]),
        )

    timing = raw["timing"]
    prediction_start = int(timing["prediction_start_second"])
    prediction_end = int(timing["prediction_end_second"])
    early_end = int(timing["early_end_second"])
    early_cadence = int(timing["early_cadence_seconds"])
    later_cadence = int(timing["later_cadence_seconds"])
    prediction_seconds = (
        *range(prediction_start, early_end + 1, early_cadence),
        *range(early_end + 1, prediction_end + 1, later_cadence),
    )
    price_seconds = prediction_seconds

    economics = raw["economics"]
    policies = tuple(
        ValuePolicy(
            name=str(values["name"]),
            selection_eligible=bool(values["selection_eligible"]),
            maximum_entry_second=int(values["maximum_entry_second"]),
            minimum_share_price=float(values["minimum_share_price"]),
            maximum_share_price=float(values["maximum_share_price"]),
            maximum_cost_per_share=float(values["maximum_cost_per_share"]),
            minimum_edge_per_share=float(values["minimum_edge_per_share"]),
        )
        for values in raw["policies"]
    )
    gate_values = raw["gates"]
    model = raw["model"]
    target_raw = model.get("target_calibration")
    target_calibration = (
        TargetCalibrationContract(
            minimum_price=float(target_raw["minimum_price"]),
            maximum_price=float(target_raw["maximum_price"]),
            sides=tuple(str(value) for value in target_raw["sides"]),
            time_bands=tuple(
                (int(values["start_second"]), int(values["end_second_exclusive"]))
                for values in target_raw["time_bands"]
            ),
            required_fitted_cells=int(target_raw["required_fitted_cells"]),
        )
        if target_raw is not None
        else None
    )
    paths = raw["paths"]
    config = AsymmetricValueConfig(
        source_path=source_path,
        package_root=package_root,
        training_contract=training_contract,
        fit=window("fit"),
        calibration=window("calibration"),
        policy=window("policy"),
        evaluation=(window("evaluation") if "evaluation" in raw["windows"] else None),
        prediction_seconds=tuple(int(value) for value in prediction_seconds),
        price_seconds=tuple(int(value) for value in price_seconds),
        calibration_bands=tuple(
            (int(values["start_second"]), int(values["end_second_exclusive"]))
            for values in model["calibration_bands"]
        ),
        quantity=float(economics["quantity"]),
        maximum_depth_participation=float(
            economics["maximum_depth_participation"]
        ),
        book_freshness_seconds=int(economics["book_freshness_seconds"]),
        execution_reserve_per_share=float(economics["execution_reserve_per_share"]),
        confidence_control_minimum_edge_per_share=float(
            economics["confidence_control_minimum_edge_per_share"]
        ),
        confidence_thresholds=tuple(
            float(value) for value in economics["confidence_thresholds"]
        ),
        policies=policies,
        gates=ValueGates(
            minimum_calibration_markets_per_band=int(
                gate_values["minimum_calibration_markets_per_band"]
            ),
            minimum_calibration_markets_per_cell=int(
                gate_values["minimum_calibration_markets_per_cell"]
            ),
            minimum_calibration_days_per_cell=int(
                gate_values["minimum_calibration_days_per_cell"]
            ),
            minimum_policy_strict_markets=int(
                gate_values["minimum_policy_strict_markets"]
            ),
            minimum_policy_executable_days=int(
                gate_values["minimum_policy_executable_days"]
            ),
            minimum_policy_source_grid_coverage=float(
                gate_values["minimum_policy_source_grid_coverage"]
            ),
            minimum_policy_strict_grid_coverage=float(
                gate_values["minimum_policy_strict_grid_coverage"]
            ),
            minimum_policy_candidate_grid_coverage=float(
                gate_values["minimum_policy_candidate_grid_coverage"]
            ),
            minimum_evaluation_strict_markets=int(
                gate_values["minimum_evaluation_strict_markets"]
            ),
            minimum_evaluation_executable_days=int(
                gate_values["minimum_evaluation_executable_days"]
            ),
            minimum_evaluation_source_grid_coverage=float(
                gate_values["minimum_evaluation_source_grid_coverage"]
            ),
            minimum_evaluation_strict_grid_coverage=float(
                gate_values["minimum_evaluation_strict_grid_coverage"]
            ),
            minimum_evaluation_candidate_grid_coverage=float(
                gate_values["minimum_evaluation_candidate_grid_coverage"]
            ),
            minimum_policy_trades=int(gate_values["minimum_policy_trades"]),
            minimum_evaluation_trades=int(gate_values["minimum_evaluation_trades"]),
            minimum_profit_factor=float(gate_values["minimum_profit_factor"]),
            minimum_net_expectancy_per_trade=float(
                gate_values["minimum_net_expectancy_per_trade"]
            ),
            minimum_capital_efficiency=float(gate_values["minimum_capital_efficiency"]),
            minimum_stress_expectancy_per_trade=float(
                gate_values["minimum_stress_expectancy_per_trade"]
            ),
            maximum_mean_cost_per_share=float(gate_values["maximum_mean_cost_per_share"]),
            maximum_mean_share_price=float(gate_values["maximum_mean_share_price"]),
            maximum_selected_calibration_bias=float(
                gate_values["maximum_selected_calibration_bias"]
            ),
            maximum_loss_recovery_wins=float(gate_values["maximum_loss_recovery_wins"]),
            maximum_average_loss=float(gate_values["maximum_average_loss"]),
            maximum_single_loss=float(gate_values["maximum_single_loss"]),
            minimum_side_trades=int(gate_values["minimum_side_trades"]),
            minimum_pre60_trades=int(gate_values["minimum_pre60_trades"]),
            minimum_20_30c_trades=int(gate_values["minimum_20_30c_trades"]),
        ),
        random_seed=int(model["random_seed"]),
        bootstrap_resamples=int(model["bootstrap_resamples"]),
        calibration_identity_l2=float(model["calibration_identity_l2"]),
        target_calibration=target_calibration,
        core_config=package_root / str(paths["core_config"]),
        oracle_source=package_root / str(paths["oracle_source"]),
        l2_source=package_root / str(paths["l2_source"]),
        candle_source=package_root / str(paths["candle_source"]),
        price_source_sql=package_root / str(paths["price_source_sql"]),
        price_cache=package_root / str(paths["price_cache"]),
        feature_cache=package_root / str(paths["feature_cache"]),
        runs=package_root / str(paths["runs"]),
        champion_model=package_root / str(paths["champion_model"]),
        champion_process=package_root / str(paths["champion_process"]),
    )
    validate_asymmetric_value_config(config)
    return config


def validate_asymmetric_value_config(config: AsymmetricValueConfig) -> None:
    if config.training_contract not in SUPPORTED_TRAINING_CONTRACTS:
        raise ValueError("asymmetric-value training contract is unsupported")
    windows = (config.fit, config.calibration, config.policy)
    if any(item.start >= item.end for item in windows):
        raise ValueError("asymmetric-value evidence windows must have positive ranges")
    if any(left.end != right.start for left, right in pairwise(windows)):
        raise ValueError("fit, calibration, and policy windows must be contiguous")
    if config.training_contract == LEGACY_TRAINING_CONTRACT:
        if config.evaluation is None or config.evaluation.start >= config.evaluation.end:
            raise ValueError("legacy asymmetric-value evaluation must have a positive range")
        if config.policy.end != config.evaluation.start:
            raise ValueError(
                "legacy asymmetric-value policy and evaluation windows must be contiguous"
            )
        if config.target_calibration is not None:
            raise ValueError("legacy asymmetric-value training cannot require target cells")
    else:
        expected_windows = (
            (config.fit.start.isoformat(), config.fit.end.isoformat()),
            (config.calibration.start.isoformat(), config.calibration.end.isoformat()),
            (config.policy.start.isoformat(), config.policy.end.isoformat()),
        )
        required_windows = (
            ("2026-04-14T00:00:00+00:00", "2026-07-16T00:00:00+00:00"),
            ("2026-07-16T00:00:00+00:00", "2026-07-23T00:00:00+00:00"),
            ("2026-07-23T00:00:00+00:00", "2026-08-02T00:00:00+00:00"),
        )
        if expected_windows != required_windows:
            raise ValueError(
                "target-calibrated asymmetric-value windows must preserve the frozen contract"
            )
        if config.evaluation is not None:
            raise ValueError(
                "target-calibrated asymmetric-value training requires fresh forward evaluation"
            )

    expected_predictions = (*range(1, 60), *range(60, 241, 5))
    expected_prices = expected_predictions
    if config.prediction_seconds != expected_predictions:
        raise ValueError(
            "asymmetric-value predictions must cover seconds 1-59 every second "
            "and seconds 60-240 every five seconds"
        )
    if config.price_seconds != expected_prices:
        raise ValueError("asymmetric-value prices must cover seconds 1-59 and 60-240 every five seconds")

    expected_bands = (
        (1, 15),
        (15, 30),
        (30, 45),
        (45, 60),
        (60, 90),
        (90, 120),
        (120, 180),
        (180, 241),
    )
    if config.calibration_bands != expected_bands:
        raise ValueError("asymmetric-value calibration bands must preserve the causal time contract")
    if config.training_contract == TARGET_CALIBRATED_TRAINING_CONTRACT:
        target = config.target_calibration
        if target is None:
            raise ValueError("target-calibrated training requires a target-cell contract")
        required_target_bands = expected_bands[:4]
        if (
            not math.isclose(target.minimum_price, 0.20)
            or not math.isclose(target.maximum_price, 0.30)
            or target.sides != ("YES", "NO")
            or target.time_bands != required_target_bands
            or target.required_fitted_cells != 8
        ):
            raise ValueError(
                "target calibration must require eight YES/NO 20-30c early-time cells"
            )
    if not math.isclose(config.quantity, 5.0):
        raise ValueError("asymmetric-value economics are fixed to five-share execution")
    if not math.isclose(config.maximum_depth_participation, 0.25):
        raise ValueError(
            "asymmetric-value execution must preserve 25% maximum depth participation"
        )
    if config.book_freshness_seconds != 2:
        raise ValueError("asymmetric-value books must be no more than two seconds old")
    if not 0.0 <= config.execution_reserve_per_share <= 0.05:
        raise ValueError("asymmetric-value execution reserve must stay inside [0, 5c]")
    if not 0.0 <= config.confidence_control_minimum_edge_per_share <= 0.05:
        raise ValueError("asymmetric-value confidence control edge is invalid")
    expected_thresholds = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.89)
    if config.confidence_thresholds != expected_thresholds:
        raise ValueError(
            "asymmetric-value confidence controls must cover 50%-89%"
        )
    if config.random_seed < 0 or config.bootstrap_resamples < 1_000:
        raise ValueError("asymmetric-value randomness and bootstrap settings are invalid")
    if (
        not math.isfinite(config.calibration_identity_l2)
        or config.calibration_identity_l2 <= 0.0
    ):
        raise ValueError("asymmetric-value calibration identity L2 must be positive")

    if not config.policies:
        raise ValueError("asymmetric-value policies must be non-empty")
    if sum(policy.selection_eligible for policy in config.policies) != 1:
        raise ValueError("exactly one asymmetric-value policy must be selection eligible")
    names = tuple(policy.name for policy in config.policies)
    if len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise ValueError("asymmetric-value policy names must be unique and non-empty")
    for policy in config.policies:
        if policy.maximum_entry_second not in config.prediction_seconds:
            raise ValueError("policy maximum entry seconds must be model decision points")
        if not (
            0.0 < policy.minimum_share_price < policy.maximum_share_price < 0.89
        ):
            raise ValueError("policy raw share-price ranges must be lower priced")
        if policy.maximum_share_price > policy.maximum_cost_per_share:
            raise ValueError("policy all-in cost cap cannot be below its raw share-price cap")
        if not 0.0 < policy.maximum_cost_per_share <= config.gates.maximum_mean_cost_per_share:
            raise ValueError("policy costs must remain inside the lower-price loss ceiling")
        if not math.isfinite(policy.minimum_edge_per_share) or policy.minimum_edge_per_share <= 0:
            raise ValueError("policy minimum edge must be finite and positive")
    if config.training_contract == TARGET_CALIBRATED_TRAINING_CONTRACT:
        primary = next(policy for policy in config.policies if policy.selection_eligible)
        if (
            primary.maximum_entry_second != 55
            or not math.isclose(primary.minimum_share_price, 0.20)
            or not math.isclose(primary.maximum_share_price, 0.30)
            or not math.isclose(primary.maximum_cost_per_share, 0.35)
            or not math.isclose(primary.minimum_edge_per_share, 0.03)
        ):
            raise ValueError(
                "target-calibrated primary policy must preserve 20-30c by55 economics"
            )

    gates = config.gates
    if gates.minimum_policy_trades <= 0 or gates.minimum_evaluation_trades <= 0:
        raise ValueError("asymmetric-value trade gates must be positive")
    evidence_gates = (
        gates.minimum_calibration_markets_per_band,
        gates.minimum_calibration_markets_per_cell,
        gates.minimum_calibration_days_per_cell,
        gates.minimum_policy_strict_markets,
        gates.minimum_policy_executable_days,
        gates.minimum_evaluation_strict_markets,
        gates.minimum_evaluation_executable_days,
    )
    if any(value <= 0 for value in evidence_gates):
        raise ValueError("asymmetric-value evidence sufficiency gates must be positive")
    if (
        gates.minimum_calibration_markets_per_cell
        > gates.minimum_calibration_markets_per_band
    ):
        raise ValueError(
            "asymmetric-value calibration-cell market support cannot exceed its band"
        )
    calibration_days = (config.calibration.end - config.calibration.start).days
    if gates.minimum_calibration_days_per_cell > calibration_days:
        raise ValueError(
            "asymmetric-value calibration-cell day support exceeds the calibration window"
        )
    if config.training_contract == TARGET_CALIBRATED_TRAINING_CONTRACT and (
        gates.minimum_calibration_markets_per_cell < 50
        or gates.minimum_calibration_days_per_cell < 5
    ):
        raise ValueError(
            "target calibration requires at least 50 markets and five UTC days per cell"
        )
    source_coverage_gates = (
        gates.minimum_policy_source_grid_coverage,
        gates.minimum_policy_strict_grid_coverage,
        gates.minimum_policy_candidate_grid_coverage,
        gates.minimum_evaluation_source_grid_coverage,
        gates.minimum_evaluation_strict_grid_coverage,
        gates.minimum_evaluation_candidate_grid_coverage,
    )
    if any(not 0.0 < value <= 1.0 for value in source_coverage_gates):
        raise ValueError("asymmetric-value source coverage gates must be inside (0, 1]")
    if gates.minimum_side_trades <= 0 or 2 * gates.minimum_side_trades > gates.minimum_policy_trades:
        raise ValueError("asymmetric-value side coverage gate is invalid")
    if gates.minimum_pre60_trades <= 0 or gates.minimum_20_30c_trades <= 0:
        raise ValueError("asymmetric-value lower-price/time trade gates must be positive")
    if not math.isfinite(gates.minimum_profit_factor) or gates.minimum_profit_factor <= 1.0:
        raise ValueError("asymmetric-value profit factor must exceed one")
    if (
        gates.minimum_net_expectancy_per_trade < 0
        or gates.minimum_capital_efficiency < 0
        or gates.minimum_stress_expectancy_per_trade < 0
    ):
        raise ValueError("asymmetric-value expectancy gates must be nonnegative")
    if not 0.0 < gates.maximum_mean_cost_per_share < 0.89:
        raise ValueError("asymmetric-value cost must stay below the champion confidence regime")
    if not 0.0 < gates.maximum_mean_share_price <= gates.maximum_mean_cost_per_share:
        raise ValueError("asymmetric-value mean share-price gate is invalid")
    if not 0.0 < gates.maximum_selected_calibration_bias < 0.25:
        raise ValueError("asymmetric-value selected calibration-bias gate is invalid")
    if (
        gates.maximum_loss_recovery_wins <= 0
        or gates.maximum_average_loss <= 0
        or gates.maximum_single_loss <= 0
    ):
        raise ValueError("asymmetric-value loss-severity gates must be positive")

    required_inputs = [
        config.core_config,
        config.l2_source,
        config.candle_source,
        config.price_source_sql,
        config.champion_model,
        config.champion_process,
    ]
    if config.training_contract == LEGACY_TRAINING_CONTRACT:
        required_inputs.append(config.oracle_source)
    missing = [str(path) for path in required_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("required asymmetric-value evidence is missing: " + ", ".join(missing))
    if config.training_contract == TARGET_CALIBRATED_TRAINING_CONTRACT:
        core = load_core_config(config.core_config)
        if core.data.source_contract != "btc_core_oracle_v1":
            raise ValueError(
                "target-calibrated training must extract the causal Core + Oracle source"
            )
        if core.paths.source_data.resolve() != config.oracle_source.resolve():
            raise ValueError(
                "target-calibrated Core and Oracle sources must share one exact-range cache"
            )
