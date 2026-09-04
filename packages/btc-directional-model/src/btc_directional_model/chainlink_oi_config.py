from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

CHAINLINK_FULL_CANDIDATE = "chainlink_full"
CHAINLINK_FULL_OI_CANDIDATE = "chainlink_full_oi"
LONG_HISTORY_CANDLE_CANDIDATE = "long_history_candle"
CHAINLINK_OI_CANDIDATE_NAMES = (
    CHAINLINK_FULL_CANDIDATE,
    CHAINLINK_FULL_OI_CANDIDATE,
    LONG_HISTORY_CANDLE_CANDIDATE,
)

_SOURCE_SCHEMA_REVISION = "f1ff0094753967b91e6dca90e38ce29359af3f3b"
_CHAMPION_MODEL_KEY = "btc-5m-directional-boundary-alignment-20260421-20260720-paper-v1"
_CHAMPION_MODEL_SHA256 = "c0778189865ca97a748a9f76cbe72d13268fd6e76db683ea727ad142b0576bc4"
_CHAMPION_MANIFEST_SHA256 = "f4c2a5fa95e856105d732b5e9578df72e16033c5b2b9ae33f81874b214f8a8f7"
_CHAMPION_FEATURE_SCHEMA_SHA256 = "d2cadbff9ae97e20d2cbeb562310e27af80279f9ecbdbec85221af82ef4b9eea"
_REFPRICE_FEED_ID = "0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8"
_POLYGON_ORACLE_PROXY = "0xc907e116054ad103354f2d350fd2514433d57f6f"


@dataclass(frozen=True)
class ChainlinkOiChampionConfig:
    model_key: str
    model_sha256: str
    manifest_sha256: str
    feature_schema_sha256: str


@dataclass(frozen=True)
class ChainlinkOiWindowConfig:
    source_range_start: datetime
    source_range_end: datetime
    short_fit_start: datetime
    short_fit_end: datetime
    calibration_start: datetime
    calibration_end: datetime
    economic_confirmation_start: datetime
    economic_confirmation_end: datetime
    directional_stress_start: datetime
    directional_stress_end: datetime


@dataclass(frozen=True)
class ChainlinkOiSourceConfig:
    market_table: str
    reference_facts_table: str
    binance_klines_table: str
    refprice_table: str
    oracle_table: str
    candles_table: str
    open_interest_table: str
    execution_snapshots_table: str
    refprice_feed_id: str
    polygon_chain_id: int
    polygon_oracle_proxy: str
    candle_symbol: str
    open_interest_symbol: str


@dataclass(frozen=True)
class ChainlinkOiFeatureConfig:
    core_profile: str
    refprice_return_seconds: tuple[int, ...]
    candle_return_minutes: tuple[int, ...]
    open_interest_return_minutes: tuple[int, ...]


@dataclass(frozen=True)
class ChainlinkOiCandidateConfig:
    name: str
    fit_window: str
    feature_profile: str
    include_refprice: bool
    include_oracle: bool
    include_candles: bool
    include_open_interest: bool


@dataclass(frozen=True)
class ChainlinkOiModelConfig:
    estimator: str
    target: str
    confidence_threshold: float
    minimum_seconds_after_open: int
    maximum_seconds_after_open: int
    cadence_seconds: int
    quantity: float
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    probability_calibration: str
    random_seed: int

    @property
    def decision_seconds(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.minimum_seconds_after_open,
                self.maximum_seconds_after_open + 1,
                self.cadence_seconds,
            )
        )


@dataclass(frozen=True)
class ChainlinkOiCausalityConfig:
    decision_clock: str
    split_unit: str
    require_source_at_or_before_decision: bool
    require_closed_candles: bool
    require_completed_open_interest_bucket: bool
    strict_common_rows_for_short_candidates: bool
    allow_imputation: bool
    allow_missingness_features: bool
    allow_market_rows_across_splits: bool


@dataclass(frozen=True)
class ChainlinkOiStalenessConfig:
    binance_klines_seconds: int
    refprice_seconds: int
    oracle_seconds: int
    candles_seconds: int
    open_interest_seconds: int
    execution_book_seconds: int


@dataclass(frozen=True)
class ChainlinkOiPromotionConfig:
    maximum_accuracy_regression: float
    minimum_champion_coverage_ratio: float
    minimum_loss_capture_rate: float
    minimum_win_retention_rate: float
    minimum_gross_loss_reduction: float
    minimum_saved_loss_to_forgone_win_ratio: float
    minimum_net_expectancy_delta: float
    minimum_profit_factor_delta: float
    maximum_expected_calibration_error: float
    minimum_economic_confirmation_trades: int
    minimum_directional_stress_markets: int
    minimum_confirmation_days: int
    require_nonworse_worst_trade: bool
    require_nonworse_worst_one_percent: bool
    paper_only: bool
    allow_fallback_winner: bool


@dataclass(frozen=True)
class ChainlinkOiPathConfig:
    core_config: Path
    refprice_source_sql: Path
    candles_source_sql: Path
    open_interest_source_sql: Path
    shared_cache: Path
    execution_evidence: Path
    runs: Path
    champion_model: Path
    champion_manifest: Path


@dataclass(frozen=True)
class ChainlinkOiBenchmarkConfig:
    source_path: Path
    package_root: Path
    profile: str
    source_schema_revision: str
    control_candidate: str
    evaluation_note: str
    champion: ChainlinkOiChampionConfig
    windows: ChainlinkOiWindowConfig
    sources: ChainlinkOiSourceConfig
    features: ChainlinkOiFeatureConfig
    candidates: tuple[ChainlinkOiCandidateConfig, ...]
    model: ChainlinkOiModelConfig
    causality: ChainlinkOiCausalityConfig
    staleness: ChainlinkOiStalenessConfig
    promotion: ChainlinkOiPromotionConfig
    paths: ChainlinkOiPathConfig

    @property
    def candidate_names(self) -> tuple[str, ...]:
        return tuple(candidate.name for candidate in self.candidates)


def load_chainlink_oi_benchmark_config(path: Path) -> ChainlinkOiBenchmarkConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    benchmark = raw["benchmark"]
    champion = raw["champion"]
    windows = raw["windows"]
    sources = raw["sources"]
    features = raw["features"]
    model = raw["model"]
    causality = raw["causality"]
    staleness = raw["staleness"]
    promotion = raw["promotion"]
    paths = raw["paths"]

    config = ChainlinkOiBenchmarkConfig(
        source_path=source_path,
        package_root=package_root,
        profile=str(benchmark["profile"]),
        source_schema_revision=str(benchmark["source_schema_revision"]).lower(),
        control_candidate=str(benchmark["control_candidate"]),
        evaluation_note=str(benchmark["evaluation_note"]).strip(),
        champion=ChainlinkOiChampionConfig(
            model_key=str(champion["model_key"]),
            model_sha256=str(champion["model_sha256"]).lower(),
            manifest_sha256=str(champion["manifest_sha256"]).lower(),
            feature_schema_sha256=str(champion["feature_schema_sha256"]).lower(),
        ),
        windows=ChainlinkOiWindowConfig(
            source_range_start=_parse_utc(windows["source_range_start"]),
            source_range_end=_parse_utc(windows["source_range_end"]),
            short_fit_start=_parse_utc(windows["short_fit_start"]),
            short_fit_end=_parse_utc(windows["short_fit_end"]),
            calibration_start=_parse_utc(windows["calibration_start"]),
            calibration_end=_parse_utc(windows["calibration_end"]),
            economic_confirmation_start=_parse_utc(windows["economic_confirmation_start"]),
            economic_confirmation_end=_parse_utc(windows["economic_confirmation_end"]),
            directional_stress_start=_parse_utc(windows["directional_stress_start"]),
            directional_stress_end=_parse_utc(windows["directional_stress_end"]),
        ),
        sources=ChainlinkOiSourceConfig(
            market_table=str(sources["market_table"]),
            reference_facts_table=str(sources["reference_facts_table"]),
            binance_klines_table=str(sources["binance_klines_table"]),
            refprice_table=str(sources["refprice_table"]),
            oracle_table=str(sources["oracle_table"]),
            candles_table=str(sources["candles_table"]),
            open_interest_table=str(sources["open_interest_table"]),
            execution_snapshots_table=str(sources["execution_snapshots_table"]),
            refprice_feed_id=str(sources["refprice_feed_id"]).lower(),
            polygon_chain_id=int(sources["polygon_chain_id"]),
            polygon_oracle_proxy=str(sources["polygon_oracle_proxy"]).lower(),
            candle_symbol=str(sources["candle_symbol"]),
            open_interest_symbol=str(sources["open_interest_symbol"]),
        ),
        features=ChainlinkOiFeatureConfig(
            core_profile=str(features["core_profile"]),
            refprice_return_seconds=tuple(
                int(value) for value in features["refprice_return_seconds"]
            ),
            candle_return_minutes=tuple(int(value) for value in features["candle_return_minutes"]),
            open_interest_return_minutes=tuple(
                int(value) for value in features["open_interest_return_minutes"]
            ),
        ),
        candidates=tuple(
            ChainlinkOiCandidateConfig(
                name=str(candidate["name"]),
                fit_window=str(candidate["fit_window"]),
                feature_profile=str(candidate["feature_profile"]),
                include_refprice=bool(candidate["include_refprice"]),
                include_oracle=bool(candidate["include_oracle"]),
                include_candles=bool(candidate["include_candles"]),
                include_open_interest=bool(candidate["include_open_interest"]),
            )
            for candidate in raw["candidates"]
        ),
        model=ChainlinkOiModelConfig(
            estimator=str(model["estimator"]),
            target=str(model["target"]),
            confidence_threshold=float(model["confidence_threshold"]),
            minimum_seconds_after_open=int(model["minimum_seconds_after_open"]),
            maximum_seconds_after_open=int(model["maximum_seconds_after_open"]),
            cadence_seconds=int(model["cadence_seconds"]),
            quantity=float(model["quantity"]),
            learning_rate=float(model["learning_rate"]),
            max_iter=int(model["max_iter"]),
            max_leaf_nodes=int(model["max_leaf_nodes"]),
            min_samples_leaf=int(model["min_samples_leaf"]),
            l2_regularization=float(model["l2_regularization"]),
            probability_calibration=str(model["probability_calibration"]),
            random_seed=int(model["random_seed"]),
        ),
        causality=ChainlinkOiCausalityConfig(
            decision_clock=str(causality["decision_clock"]),
            split_unit=str(causality["split_unit"]),
            require_source_at_or_before_decision=bool(
                causality["require_source_at_or_before_decision"]
            ),
            require_closed_candles=bool(causality["require_closed_candles"]),
            require_completed_open_interest_bucket=bool(
                causality["require_completed_open_interest_bucket"]
            ),
            strict_common_rows_for_short_candidates=bool(
                causality["strict_common_rows_for_short_candidates"]
            ),
            allow_imputation=bool(causality["allow_imputation"]),
            allow_missingness_features=bool(causality["allow_missingness_features"]),
            allow_market_rows_across_splits=bool(causality["allow_market_rows_across_splits"]),
        ),
        staleness=ChainlinkOiStalenessConfig(
            binance_klines_seconds=int(staleness["binance_klines_seconds"]),
            refprice_seconds=int(staleness["refprice_seconds"]),
            oracle_seconds=int(staleness["oracle_seconds"]),
            candles_seconds=int(staleness["candles_seconds"]),
            open_interest_seconds=int(staleness["open_interest_seconds"]),
            execution_book_seconds=int(staleness["execution_book_seconds"]),
        ),
        promotion=ChainlinkOiPromotionConfig(
            maximum_accuracy_regression=float(promotion["maximum_accuracy_regression"]),
            minimum_champion_coverage_ratio=float(promotion["minimum_champion_coverage_ratio"]),
            minimum_loss_capture_rate=float(promotion["minimum_loss_capture_rate"]),
            minimum_win_retention_rate=float(promotion["minimum_win_retention_rate"]),
            minimum_gross_loss_reduction=float(promotion["minimum_gross_loss_reduction"]),
            minimum_saved_loss_to_forgone_win_ratio=float(
                promotion["minimum_saved_loss_to_forgone_win_ratio"]
            ),
            minimum_net_expectancy_delta=float(promotion["minimum_net_expectancy_delta"]),
            minimum_profit_factor_delta=float(promotion["minimum_profit_factor_delta"]),
            maximum_expected_calibration_error=float(
                promotion["maximum_expected_calibration_error"]
            ),
            minimum_economic_confirmation_trades=int(
                promotion["minimum_economic_confirmation_trades"]
            ),
            minimum_directional_stress_markets=int(promotion["minimum_directional_stress_markets"]),
            minimum_confirmation_days=int(promotion["minimum_confirmation_days"]),
            require_nonworse_worst_trade=bool(promotion["require_nonworse_worst_trade"]),
            require_nonworse_worst_one_percent=bool(
                promotion["require_nonworse_worst_one_percent"]
            ),
            paper_only=bool(promotion["paper_only"]),
            allow_fallback_winner=bool(promotion["allow_fallback_winner"]),
        ),
        paths=ChainlinkOiPathConfig(
            core_config=package_root / str(paths["core_config"]),
            refprice_source_sql=package_root / str(paths["refprice_source_sql"]),
            candles_source_sql=package_root / str(paths["candles_source_sql"]),
            open_interest_source_sql=package_root / str(paths["open_interest_source_sql"]),
            shared_cache=package_root / str(paths["shared_cache"]),
            execution_evidence=package_root / str(paths["execution_evidence"]),
            runs=package_root / str(paths["runs"]),
            champion_model=package_root / str(paths["champion_model"]),
            champion_manifest=package_root / str(paths["champion_manifest"]),
        ),
    )
    validate_chainlink_oi_benchmark_config(config)
    return config


def validate_chainlink_oi_benchmark_config(config: ChainlinkOiBenchmarkConfig) -> None:
    if not config.profile or not config.evaluation_note:
        raise ValueError("benchmark profile and evaluation note must be non-empty")
    if config.source_schema_revision != _SOURCE_SCHEMA_REVISION:
        raise ValueError("source schema revision must remain pinned")
    if config.control_candidate != "frozen_boundary_alignment_champion":
        raise ValueError("the boundary-alignment champion must remain the control")

    _validate_champion(config)
    _validate_windows(config.windows)
    _validate_sources(config.sources)
    _validate_features(config.features)
    _validate_candidates(config.candidates)
    _validate_model(config.model)
    _validate_causality(config.causality)
    _validate_staleness(config.staleness)
    _validate_promotion(config.promotion)

    source_files = (
        config.paths.refprice_source_sql,
        config.paths.candles_source_sql,
        config.paths.open_interest_source_sql,
    )
    if any(path.suffix != ".sql" for path in source_files):
        raise ValueError("source SQL paths must resolve to SQL files")
    required_files = (config.paths.core_config, *source_files)
    if any(not path.is_file() for path in required_files):
        raise ValueError("core configuration and source SQL files must exist")
    for name, path in (
        ("core_config", config.paths.core_config),
        ("refprice_source_sql", config.paths.refprice_source_sql),
        ("candles_source_sql", config.paths.candles_source_sql),
        ("open_interest_source_sql", config.paths.open_interest_source_sql),
        ("shared_cache", config.paths.shared_cache),
        ("execution_evidence", config.paths.execution_evidence),
        ("runs", config.paths.runs),
        ("champion_model", config.paths.champion_model),
        ("champion_manifest", config.paths.champion_manifest),
    ):
        try:
            path.relative_to(config.package_root)
        except ValueError as error:
            raise ValueError(f"{name} must remain inside the package root") from error


def _validate_champion(config: ChainlinkOiBenchmarkConfig) -> None:
    champion = config.champion
    expected = (
        _CHAMPION_MODEL_KEY,
        _CHAMPION_MODEL_SHA256,
        _CHAMPION_MANIFEST_SHA256,
        _CHAMPION_FEATURE_SCHEMA_SHA256,
    )
    actual = (
        champion.model_key,
        champion.model_sha256,
        champion.manifest_sha256,
        champion.feature_schema_sha256,
    )
    if actual != expected:
        raise ValueError("frozen champion identity or hashes changed")
    for name, value in (
        ("model_sha256", champion.model_sha256),
        ("manifest_sha256", champion.manifest_sha256),
        ("feature_schema_sha256", champion.feature_schema_sha256),
    ):
        _validate_sha256(name, value)
    if not config.paths.champion_model.is_file():
        raise ValueError("frozen champion model is missing")
    if not config.paths.champion_manifest.is_file():
        raise ValueError("frozen champion manifest is missing")
    if _file_sha256(config.paths.champion_model) != champion.model_sha256:
        raise ValueError("frozen champion model hash mismatch")
    if _file_sha256(config.paths.champion_manifest) != champion.manifest_sha256:
        raise ValueError("frozen champion manifest hash mismatch")
    with config.paths.champion_manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        manifest.get("model_key") != champion.model_key
        or manifest.get("model_sha256") != champion.model_sha256
        or manifest.get("feature_schema_sha256") != champion.feature_schema_sha256
    ):
        raise ValueError("frozen champion manifest identity mismatch")


def _validate_windows(windows: ChainlinkOiWindowConfig) -> None:
    expected = (
        datetime(2026, 3, 21, tzinfo=UTC),
        datetime(2026, 7, 29, tzinfo=UTC),
        datetime(2026, 7, 3, tzinfo=UTC),
        datetime(2026, 7, 14, tzinfo=UTC),
        datetime(2026, 7, 14, tzinfo=UTC),
        datetime(2026, 7, 16, tzinfo=UTC),
        datetime(2026, 7, 16, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 7, 29, tzinfo=UTC),
    )
    actual = (
        windows.source_range_start,
        windows.source_range_end,
        windows.short_fit_start,
        windows.short_fit_end,
        windows.calibration_start,
        windows.calibration_end,
        windows.economic_confirmation_start,
        windows.economic_confirmation_end,
        windows.directional_stress_start,
        windows.directional_stress_end,
    )
    if actual != expected:
        raise ValueError("training and evaluation windows must remain frozen")
    if not (
        windows.source_range_start
        < windows.short_fit_start
        < windows.short_fit_end
        == windows.calibration_start
        < windows.calibration_end
        == windows.economic_confirmation_start
        < windows.economic_confirmation_end
        == windows.directional_stress_start
        < windows.directional_stress_end
        == windows.source_range_end
    ):
        raise ValueError("training and evaluation windows must be chronological")


def _validate_sources(sources: ChainlinkOiSourceConfig) -> None:
    expected_tables = (
        "polymarket.btc_interval_markets",
        "polymarket.btc_market_reference_facts",
        "market_data.binance_spot_btcusdt_one_second_ohlcv",
        "polymarket.chainlink_btcusd_archive_ticks",
        "market_data.polygon_chainlink_btcusd_oracle_rounds",
        "market_data.chainlink_btcusd_one_minute_candles",
        "market_data.binance_futures_btcusdt_open_interest",
        "polymarket.btc_market_execution_snapshots",
    )
    actual_tables = (
        sources.market_table,
        sources.reference_facts_table,
        sources.binance_klines_table,
        sources.refprice_table,
        sources.oracle_table,
        sources.candles_table,
        sources.open_interest_table,
        sources.execution_snapshots_table,
    )
    if actual_tables != expected_tables:
        raise ValueError("source table contract changed")
    if (
        sources.refprice_feed_id != _REFPRICE_FEED_ID
        or sources.polygon_chain_id != 137
        or sources.polygon_oracle_proxy != _POLYGON_ORACLE_PROXY
        or sources.candle_symbol != "BTCUSD"
        or sources.open_interest_symbol != "BTCUSDT"
    ):
        raise ValueError("source identity contract changed")


def _validate_features(features: ChainlinkOiFeatureConfig) -> None:
    if features.core_profile != "boundary_alignment_v1":
        raise ValueError("candidate core features must remain boundary-aligned")
    if features.refprice_return_seconds != (1, 5, 15, 30, 60):
        raise ValueError("RefPrice horizons must remain fixed")
    if features.candle_return_minutes != (5, 15, 30, 60):
        raise ValueError("candle horizons must remain fixed")
    if features.open_interest_return_minutes != (5, 15, 30, 60):
        raise ValueError("open-interest horizons must remain fixed")


def _validate_candidates(candidates: tuple[ChainlinkOiCandidateConfig, ...]) -> None:
    expected = (
        (
            CHAINLINK_FULL_CANDIDATE,
            "short_common",
            "boundary_plus_chainlink_full",
            True,
            True,
            True,
            False,
        ),
        (
            CHAINLINK_FULL_OI_CANDIDATE,
            "short_common",
            "boundary_plus_chainlink_full_oi",
            True,
            True,
            True,
            True,
        ),
        (
            LONG_HISTORY_CANDLE_CANDIDATE,
            "long_history",
            "boundary_plus_oracle_candles",
            False,
            True,
            True,
            False,
        ),
    )
    actual = tuple(
        (
            candidate.name,
            candidate.fit_window,
            candidate.feature_profile,
            candidate.include_refprice,
            candidate.include_oracle,
            candidate.include_candles,
            candidate.include_open_interest,
        )
        for candidate in candidates
    )
    if actual != expected:
        raise ValueError("candidate feature-ablation contract changed")


def _validate_model(model: ChainlinkOiModelConfig) -> None:
    if (
        model.estimator != "histogram_gradient_boosting"
        or model.target != "official_up_outcome"
        or model.probability_calibration != "four_time_band_platt"
        or model.random_seed != 20260801
    ):
        raise ValueError("estimator contract must remain fixed to the champion family")
    if (
        not math.isclose(model.confidence_threshold, 0.89)
        or model.minimum_seconds_after_open != 60
        or model.maximum_seconds_after_open != 240
        or model.cadence_seconds != 5
        or not math.isclose(model.quantity, 5.0)
    ):
        raise ValueError("champion decision contract must remain 0.89, 60-240/5, five-share")
    parameters = (
        model.learning_rate,
        model.max_iter,
        model.max_leaf_nodes,
        model.min_samples_leaf,
        model.l2_regularization,
    )
    if parameters != (0.05, 160, 15, 100, 0.1):
        raise ValueError("HGB parameters must remain frozen")


def _validate_causality(causality: ChainlinkOiCausalityConfig) -> None:
    if causality.decision_clock != "observed_at" or causality.split_unit != "utc_day_market":
        raise ValueError("causal clock and chronological split unit must remain fixed")
    if not (
        causality.require_source_at_or_before_decision
        and causality.require_closed_candles
        and causality.require_completed_open_interest_bucket
        and causality.strict_common_rows_for_short_candidates
    ):
        raise ValueError("all candidate inputs must remain causal and common-row qualified")
    if (
        causality.allow_imputation
        or causality.allow_missingness_features
        or causality.allow_market_rows_across_splits
    ):
        raise ValueError("imputation, missingness features, and split leakage are disabled")


def _validate_staleness(staleness: ChainlinkOiStalenessConfig) -> None:
    if (
        staleness.binance_klines_seconds,
        staleness.refprice_seconds,
        staleness.oracle_seconds,
        staleness.candles_seconds,
        staleness.open_interest_seconds,
        staleness.execution_book_seconds,
    ) != (2, 5, 3600, 120, 600, 2):
        raise ValueError("source staleness limits must remain fixed")


def _validate_promotion(promotion: ChainlinkOiPromotionConfig) -> None:
    bounded = (
        promotion.maximum_accuracy_regression,
        promotion.minimum_champion_coverage_ratio,
        promotion.minimum_loss_capture_rate,
        promotion.minimum_win_retention_rate,
        promotion.minimum_gross_loss_reduction,
        promotion.maximum_expected_calibration_error,
    )
    if any(not 0 <= value <= 1 for value in bounded):
        raise ValueError("bounded promotion criteria must remain inside [0, 1]")
    if (
        promotion.maximum_accuracy_regression != 0.01
        or promotion.minimum_champion_coverage_ratio != 0.80
        or promotion.minimum_loss_capture_rate != 0.20
        or promotion.minimum_win_retention_rate != 0.80
        or promotion.minimum_gross_loss_reduction != 0.10
        or promotion.minimum_saved_loss_to_forgone_win_ratio != 1.0
        or promotion.minimum_net_expectancy_delta != 0.0
        or promotion.minimum_profit_factor_delta != 0.0
        or promotion.maximum_expected_calibration_error != 0.05
        or promotion.minimum_economic_confirmation_trades != 50
        or promotion.minimum_directional_stress_markets != 1_000
        or promotion.minimum_confirmation_days != 3
    ):
        raise ValueError("economic promotion criteria must remain frozen")
    if not (
        promotion.require_nonworse_worst_trade
        and promotion.require_nonworse_worst_one_percent
        and promotion.paper_only
    ):
        raise ValueError("tail-loss checks and paper-only scope are mandatory")
    if promotion.allow_fallback_winner:
        raise ValueError("the benchmark has a strict no-fallback promotion policy")


def _parse_utc(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal digits")
