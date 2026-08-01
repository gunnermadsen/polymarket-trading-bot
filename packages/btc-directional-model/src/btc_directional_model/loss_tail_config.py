from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from .core_config import load_core_config, parse_utc_day

LOSS_TAIL_DECISION_SECONDS = tuple(range(60, 241, 5))
LOSS_TAIL_CONTEXT_SECONDS = tuple(range(55, 241, 5))

BOUNDARY_RESIDUAL_ECONOMIC_HGB_CANDIDATE = "boundary-alignment-residual-economic-hgb-v1"
LOSS_TAIL_CANDIDATE_NAMES = (BOUNDARY_RESIDUAL_ECONOMIC_HGB_CANDIDATE,)


@dataclass(frozen=True)
class LossTailBlockConfig:
    name: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class LossTailFoldConfig:
    name: str
    fit_block_names: tuple[str, ...]
    calibration_block_name: str
    evaluation_block_name: str


@dataclass(frozen=True)
class LossTailWalkForwardConfig:
    history_start: datetime
    blocks: tuple[LossTailBlockConfig, ...]
    folds: tuple[LossTailFoldConfig, ...]


@dataclass(frozen=True)
class LossTailDataConfig:
    decision_start_seconds: int
    decision_end_seconds: int
    decision_interval_seconds: int
    context_start_seconds: int
    quantity: float
    book_freshness_seconds: int
    strict_book_required: bool
    allow_book_imputation: bool
    allow_missingness_features: bool
    require_complete_paths_for_strategy: bool
    allow_point_qualified_row_metrics: bool

    @property
    def decision_seconds(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.decision_start_seconds,
                self.decision_end_seconds + 1,
                self.decision_interval_seconds,
            )
        )

    @property
    def context_seconds(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.context_start_seconds,
                self.decision_end_seconds + 1,
                self.decision_interval_seconds,
            )
        )


@dataclass(frozen=True)
class LossTailFeatureConfig:
    direct_profile: str
    correctness_profile: str
    boundary_feature_count: int
    mature_reversal_feature_count: int
    oracle_feature_count: int
    strict_book_feature_count: int

    @property
    def direct_feature_count(self) -> int:
        return (
            self.boundary_feature_count
            + self.mature_reversal_feature_count
            + self.oracle_feature_count
            + self.strict_book_feature_count
        )


@dataclass(frozen=True)
class LossTailCandidateConfig:
    name: str
    estimator: str
    target: str
    feature_profile: str
    row_weighting: str
    probability_calibration: str
    calibration_weighting: str
    direction_policy: str
    requires_causal_oof_sources: bool
    random_seed: int


@dataclass(frozen=True)
class LossTailResourceConfig:
    workers: int
    threads_per_worker: int
    memory_limit_gib: int


@dataclass(frozen=True)
class LossTailWeightingConfig:
    normalization: str
    minimum_multiplier: float
    maximum_multiplier: float
    denominator_floor: float
    fee_inclusive_debit: bool


@dataclass(frozen=True)
class LossTailGateConfig:
    minimum_selected_accuracy: float
    minimum_wilson_lower: float
    maximum_control_accuracy_regression: float
    maximum_expected_calibration_error: float
    maximum_wins_per_average_loss: float
    maximum_gross_loss_profit_ratio: float
    minimum_profit_factor: float
    minimum_pnl_per_all_core_market: float
    minimum_total_pnl: float
    maximum_mean_selected_price: float
    minimum_book_qualified_coverage: float
    minimum_pooled_trades: int
    minimum_confirmation_trades: int
    minimum_fold_direction_trades: int
    minimum_worst_trade: float
    minimum_mean_worst_one_percent: float
    maximum_drawdown: float
    required_nonnegative_expectancy_folds: int
    minimum_loss_improvement_folds: int
    minimum_gross_loss_reduction: float
    minimum_gross_profit_retention: float
    high_debit_threshold: float
    minimum_high_debit_loss_recall: float
    require_highest_price_band_positive: bool
    allow_fallback_winner: bool


@dataclass(frozen=True)
class LossTailPathConfig:
    core_features: Path
    execution_evidence: Path
    shared_cache: Path
    runs: Path


@dataclass(frozen=True)
class LossTailBenchmarkConfig:
    source_path: Path
    package_root: Path
    core_config: Path
    core_config_sha256: str
    evaluation_note: str
    walk_forward: LossTailWalkForwardConfig
    data: LossTailDataConfig
    features: LossTailFeatureConfig
    candidates: tuple[LossTailCandidateConfig, ...]
    weighting: LossTailWeightingConfig
    resources: LossTailResourceConfig
    gates: LossTailGateConfig
    paths: LossTailPathConfig

    @property
    def candidate_names(self) -> tuple[str, ...]:
        return tuple(candidate.name for candidate in self.candidates)


_EXPECTED_BLOCKS = (
    ("book_history_apr13", datetime(2026, 4, 13, tzinfo=UTC), datetime(2026, 5, 26, tzinfo=UTC)),
    ("calibration_may26", datetime(2026, 5, 26, tzinfo=UTC), datetime(2026, 6, 2, tzinfo=UTC)),
    ("calibration_jun02", datetime(2026, 6, 2, tzinfo=UTC), datetime(2026, 6, 9, tzinfo=UTC)),
    ("evaluation_jun09", datetime(2026, 6, 9, tzinfo=UTC), datetime(2026, 7, 3, tzinfo=UTC)),
    ("evaluation_jul03", datetime(2026, 7, 3, tzinfo=UTC), datetime(2026, 7, 14, tzinfo=UTC)),
    ("confirmation_jul14", datetime(2026, 7, 14, tzinfo=UTC), datetime(2026, 7, 29, tzinfo=UTC)),
)
_EXPECTED_FOLDS = (
    (
        "walk_forward_jun09",
        ("book_history_apr13", "calibration_may26"),
        "calibration_jun02",
        "evaluation_jun09",
    ),
    (
        "walk_forward_jul03",
        ("book_history_apr13", "calibration_may26", "calibration_jun02"),
        "evaluation_jun09",
        "evaluation_jul03",
    ),
    (
        "walk_forward_jul14",
        (
            "book_history_apr13",
            "calibration_may26",
            "calibration_jun02",
            "evaluation_jun09",
        ),
        "evaluation_jul03",
        "confirmation_jul14",
    ),
)
_EXPECTED_CANDIDATES = (
    (
        BOUNDARY_RESIDUAL_ECONOMIC_HGB_CANDIDATE,
        "histogram_gradient_boosting",
        "boundary_direction_correct",
        "boundary_disagreement_correctness",
        "incorrect_proposal_debit_odds_capped",
        "platt",
        "unweighted",
        "boundary_locked_economic_action",
        True,
    ),
)
_EXPECTED_RANDOM_SEED = 20260731


def load_loss_tail_benchmark_config(path: Path) -> LossTailBenchmarkConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    benchmark = raw["benchmark"]
    walk_forward = raw["walk_forward"]
    data = raw["data"]
    features = raw["features"]
    weighting = raw["weighting"]
    resources = raw["resources"]
    gates = raw["gates"]
    paths = raw["paths"]
    config = LossTailBenchmarkConfig(
        source_path=source_path,
        package_root=package_root,
        core_config=package_root / str(benchmark["core_config"]),
        core_config_sha256=str(benchmark["core_config_sha256"]).lower(),
        evaluation_note=str(benchmark["evaluation_note"]).strip(),
        walk_forward=LossTailWalkForwardConfig(
            history_start=parse_utc_day(walk_forward["history_start"]),
            blocks=tuple(
                LossTailBlockConfig(
                    name=str(block["name"]),
                    start=parse_utc_day(block["start"]),
                    end=parse_utc_day(block["end"]),
                )
                for block in walk_forward["blocks"]
            ),
            folds=tuple(
                LossTailFoldConfig(
                    name=str(fold["name"]),
                    fit_block_names=tuple(str(value) for value in fold["fit_block_names"]),
                    calibration_block_name=str(fold["calibration_block_name"]),
                    evaluation_block_name=str(fold["evaluation_block_name"]),
                )
                for fold in walk_forward["folds"]
            ),
        ),
        data=LossTailDataConfig(
            decision_start_seconds=int(data["decision_start_seconds"]),
            decision_end_seconds=int(data["decision_end_seconds"]),
            decision_interval_seconds=int(data["decision_interval_seconds"]),
            context_start_seconds=int(data["context_start_seconds"]),
            quantity=float(data["quantity"]),
            book_freshness_seconds=int(data["book_freshness_seconds"]),
            strict_book_required=bool(data["strict_book_required"]),
            allow_book_imputation=bool(data["allow_book_imputation"]),
            allow_missingness_features=bool(data["allow_missingness_features"]),
            require_complete_paths_for_strategy=bool(data["require_complete_paths_for_strategy"]),
            allow_point_qualified_row_metrics=bool(data["allow_point_qualified_row_metrics"]),
        ),
        features=LossTailFeatureConfig(
            direct_profile=str(features["direct_profile"]),
            correctness_profile=str(features["correctness_profile"]),
            boundary_feature_count=int(features["boundary_feature_count"]),
            mature_reversal_feature_count=int(features["mature_reversal_feature_count"]),
            oracle_feature_count=int(features["oracle_feature_count"]),
            strict_book_feature_count=int(features["strict_book_feature_count"]),
        ),
        candidates=tuple(
            LossTailCandidateConfig(
                name=str(candidate["name"]),
                estimator=str(candidate["estimator"]),
                target=str(candidate["target"]),
                feature_profile=str(candidate["feature_profile"]),
                row_weighting=str(candidate["row_weighting"]),
                probability_calibration=str(candidate["probability_calibration"]),
                calibration_weighting=str(candidate["calibration_weighting"]),
                direction_policy=str(candidate["direction_policy"]),
                requires_causal_oof_sources=bool(candidate["requires_causal_oof_sources"]),
                random_seed=int(candidate["random_seed"]),
            )
            for candidate in raw["candidates"]
        ),
        weighting=LossTailWeightingConfig(
            normalization=str(weighting["normalization"]),
            minimum_multiplier=float(weighting["minimum_multiplier"]),
            maximum_multiplier=float(weighting["maximum_multiplier"]),
            denominator_floor=float(weighting["denominator_floor"]),
            fee_inclusive_debit=bool(weighting["fee_inclusive_debit"]),
        ),
        resources=LossTailResourceConfig(
            workers=int(resources["workers"]),
            threads_per_worker=int(resources["threads_per_worker"]),
            memory_limit_gib=int(resources["memory_limit_gib"]),
        ),
        gates=LossTailGateConfig(
            minimum_selected_accuracy=float(gates["minimum_selected_accuracy"]),
            minimum_wilson_lower=float(gates["minimum_wilson_lower"]),
            maximum_control_accuracy_regression=float(gates["maximum_control_accuracy_regression"]),
            maximum_expected_calibration_error=float(gates["maximum_expected_calibration_error"]),
            maximum_wins_per_average_loss=float(gates["maximum_wins_per_average_loss"]),
            maximum_gross_loss_profit_ratio=float(gates["maximum_gross_loss_profit_ratio"]),
            minimum_profit_factor=float(gates["minimum_profit_factor"]),
            minimum_pnl_per_all_core_market=float(gates["minimum_pnl_per_all_core_market"]),
            minimum_total_pnl=float(gates["minimum_total_pnl"]),
            maximum_mean_selected_price=float(gates["maximum_mean_selected_price"]),
            minimum_book_qualified_coverage=float(gates["minimum_book_qualified_coverage"]),
            minimum_pooled_trades=int(gates["minimum_pooled_trades"]),
            minimum_confirmation_trades=int(gates["minimum_confirmation_trades"]),
            minimum_fold_direction_trades=int(gates["minimum_fold_direction_trades"]),
            minimum_worst_trade=float(gates["minimum_worst_trade"]),
            minimum_mean_worst_one_percent=float(gates["minimum_mean_worst_one_percent"]),
            maximum_drawdown=float(gates["maximum_drawdown"]),
            required_nonnegative_expectancy_folds=int(
                gates["required_nonnegative_expectancy_folds"]
            ),
            minimum_loss_improvement_folds=int(gates["minimum_loss_improvement_folds"]),
            minimum_gross_loss_reduction=float(gates["minimum_gross_loss_reduction"]),
            minimum_gross_profit_retention=float(gates["minimum_gross_profit_retention"]),
            high_debit_threshold=float(gates["high_debit_threshold"]),
            minimum_high_debit_loss_recall=float(gates["minimum_high_debit_loss_recall"]),
            require_highest_price_band_positive=bool(gates["require_highest_price_band_positive"]),
            allow_fallback_winner=bool(gates["allow_fallback_winner"]),
        ),
        paths=LossTailPathConfig(
            core_features=package_root / str(paths["core_features"]),
            execution_evidence=package_root / str(paths["execution_evidence"]),
            shared_cache=package_root / str(paths["shared_cache"]),
            runs=package_root / str(paths["runs"]),
        ),
    )
    validate_loss_tail_benchmark_config(config)
    return config


def validate_loss_tail_benchmark_config(config: LossTailBenchmarkConfig) -> None:
    _validate_source_contract(config)
    _validate_walk_forward_contract(config.walk_forward)
    _validate_data_contract(config.data)
    _validate_feature_contract(config.features)
    _validate_candidate_contract(config.candidates)
    _validate_weighting_contract(config.weighting)
    _validate_resource_contract(config.resources)
    _validate_gate_contract(config.gates, len(config.walk_forward.folds))
    _validate_path_contract(config)


def _validate_source_contract(config: LossTailBenchmarkConfig) -> None:
    if not config.core_config.is_file():
        raise ValueError(f"core config is missing: {config.core_config}")
    if len(config.core_config_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in config.core_config_sha256
    ):
        raise ValueError("core_config_sha256 must be 64 lowercase hexadecimal digits")
    if _file_sha256(config.core_config) != config.core_config_sha256:
        raise ValueError("pinned core training config hash mismatch")
    if not config.evaluation_note:
        raise ValueError("evaluation_note must be non-empty")

    core_config = load_core_config(config.core_config)
    observed_timing = (
        core_config.data.min_seconds_after_open,
        300 - core_config.data.min_seconds_before_close,
        core_config.data.sample_interval_seconds,
    )
    if observed_timing != (60, 240, 5):
        raise ValueError("loss-tail core features must cover 60-240 seconds at 5s")
    if core_config.data.range_start != datetime(
        2026, 3, 21, tzinfo=UTC
    ) or core_config.data.range_end != datetime(2026, 7, 29, tzinfo=UTC):
        raise ValueError("loss-tail core history must remain March 21 through July 28")


def _validate_walk_forward_contract(config: LossTailWalkForwardConfig) -> None:
    if config.history_start != datetime(2026, 3, 21, tzinfo=UTC):
        raise ValueError("loss-tail history must begin March 21, 2026")
    if any(block.start >= block.end for block in config.blocks):
        raise ValueError("every loss-tail block must have a positive range")
    if any(previous.end != current.start for previous, current in pairwise(config.blocks)):
        raise ValueError("loss-tail blocks must be chronological and contiguous")
    observed_blocks = tuple((block.name, block.start, block.end) for block in config.blocks)
    if observed_blocks != _EXPECTED_BLOCKS:
        raise ValueError("loss-tail block identities and ranges must remain frozen")
    observed_folds = tuple(
        (
            fold.name,
            fold.fit_block_names,
            fold.calibration_block_name,
            fold.evaluation_block_name,
        )
        for fold in config.folds
    )
    if observed_folds != _EXPECTED_FOLDS:
        raise ValueError("loss-tail fit, calibration, and evaluation folds must remain frozen")


def _validate_data_contract(config: LossTailDataConfig) -> None:
    if config.decision_seconds != LOSS_TAIL_DECISION_SECONDS:
        raise ValueError("loss-tail decisions must cover 60-240 seconds at 5s")
    if config.context_seconds != LOSS_TAIL_CONTEXT_SECONDS:
        raise ValueError("loss-tail causal context must cover 55-240 seconds at 5s")
    if not math.isclose(config.quantity, 5.0):
        raise ValueError("loss-tail economics require five-share executable VWAP")
    if config.book_freshness_seconds != 2:
        raise ValueError("loss-tail training requires two-second book freshness")
    strict_values = (
        config.strict_book_required,
        not config.allow_book_imputation,
        not config.allow_missingness_features,
        not config.require_complete_paths_for_strategy,
        config.allow_point_qualified_row_metrics,
    )
    if not all(strict_values):
        raise ValueError("loss-tail book qualification and evaluation policy must remain strict")


def _validate_feature_contract(config: LossTailFeatureConfig) -> None:
    if (
        config.direct_profile != "loss_tail_direct_124"
        or config.correctness_profile != "boundary_disagreement_correctness"
        or config.boundary_feature_count != 68
        or config.mature_reversal_feature_count != 13
        or config.oracle_feature_count != 11
        or config.strict_book_feature_count != 32
        or config.direct_feature_count != 124
    ):
        raise ValueError("loss-tail feature profiles and the 124-feature direct matrix are frozen")


def _validate_candidate_contract(
    candidates: tuple[LossTailCandidateConfig, ...],
) -> None:
    observed = tuple(
        (
            candidate.name,
            candidate.estimator,
            candidate.target,
            candidate.feature_profile,
            candidate.row_weighting,
            candidate.probability_calibration,
            candidate.calibration_weighting,
            candidate.direction_policy,
            candidate.requires_causal_oof_sources,
        )
        for candidate in candidates
    )
    if observed != _EXPECTED_CANDIDATES:
        raise ValueError("the residual economic HGB candidate contract must remain frozen")
    if any(candidate.random_seed != _EXPECTED_RANDOM_SEED for candidate in candidates):
        raise ValueError("loss-tail candidate random seeds must remain frozen")


def _validate_weighting_contract(config: LossTailWeightingConfig) -> None:
    if (
        config.normalization != "one_proposal_equal_base"
        or not math.isclose(config.minimum_multiplier, 1.0)
        or not math.isclose(config.maximum_multiplier, 10.0)
        or not math.isclose(config.denominator_floor, 0.05)
        or not config.fee_inclusive_debit
    ):
        raise ValueError("loss-tail wrong-side debit severity weighting is frozen")


def _validate_resource_contract(config: LossTailResourceConfig) -> None:
    if (config.workers, config.threads_per_worker, config.memory_limit_gib) != (1, 3, 8):
        raise ValueError("loss-tail training requires one worker with three threads and 8 GiB")


def _validate_gate_contract(config: LossTailGateConfig, fold_count: int) -> None:
    exact_probabilities = (
        (config.minimum_selected_accuracy, 0.89),
        (config.minimum_wilson_lower, 0.87),
        (config.maximum_control_accuracy_regression, 0.005),
        (config.maximum_expected_calibration_error, 0.05),
        (config.maximum_wins_per_average_loss, 6.34),
        (config.maximum_gross_loss_profit_ratio, 0.712),
        (config.minimum_profit_factor, 1.265),
        (config.minimum_pnl_per_all_core_market, 0.02044),
        (config.maximum_mean_selected_price, 0.8697),
        (config.minimum_book_qualified_coverage, 0.371),
        (config.minimum_gross_loss_reduction, 0.30),
        (config.minimum_gross_profit_retention, 0.70),
        (config.high_debit_threshold, 0.90),
        (config.minimum_high_debit_loss_recall, 0.30),
    )
    if any(not math.isclose(observed, expected) for observed, expected in exact_probabilities):
        raise ValueError(
            "loss-tail accuracy, tail-loss, calibration, and coverage gates are frozen"
        )
    if config.minimum_wilson_lower > config.minimum_selected_accuracy:
        raise ValueError("minimum Wilson lower bound cannot exceed selected accuracy")
    if (
        config.minimum_pooled_trades != 500
        or config.minimum_confirmation_trades != 200
        or config.minimum_fold_direction_trades != 10
    ):
        raise ValueError("loss-tail trade-count gates are frozen")
    if (
        config.required_nonnegative_expectancy_folds != fold_count
        or config.minimum_loss_improvement_folds != 2
    ):
        raise ValueError("loss-tail fold robustness gates are frozen")
    exact_economics = (
        (config.minimum_total_pnl, 283.15),
        (config.minimum_worst_trade, -4.90686),
        (config.minimum_mean_worst_one_percent, -4.79185),
        (config.maximum_drawdown, 38.69),
    )
    if any(not math.isclose(observed, expected) for observed, expected in exact_economics):
        raise ValueError("loss-tail absolute economic gates are frozen")
    required_flags = (
        config.require_highest_price_band_positive,
        not config.allow_fallback_winner,
    )
    if not all(required_flags):
        raise ValueError("loss-tail comparative gates and no-fallback policy are mandatory")


def _validate_path_contract(config: LossTailBenchmarkConfig) -> None:
    core_config = load_core_config(config.core_config)
    if config.paths.core_features != core_config.paths.development_feature_data:
        raise ValueError("loss-tail core feature path must match the pinned core config")
    generated_paths = (
        config.paths.execution_evidence,
        config.paths.shared_cache,
        config.paths.runs,
    )
    if len(set(generated_paths)) != len(generated_paths):
        raise ValueError("loss-tail evidence, cache, and run paths must be isolated")
    if any(path == config.paths.core_features for path in generated_paths):
        raise ValueError("loss-tail outputs cannot overwrite the immutable core feature cache")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
