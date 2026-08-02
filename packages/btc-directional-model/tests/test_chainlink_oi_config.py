from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from btc_directional_model.chainlink_oi_config import (
    CHAINLINK_FULL_CANDIDATE,
    CHAINLINK_FULL_OI_CANDIDATE,
    CHAINLINK_OI_CANDIDATE_NAMES,
    LONG_HISTORY_CANDLE_CANDIDATE,
    load_chainlink_oi_benchmark_config,
    validate_chainlink_oi_benchmark_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PACKAGE_ROOT / "configs" / "btc-5m-directional-chainlink-oi-champion-20260321-20260729.toml"
)


def test_config_pins_champion_source_contract_and_paths() -> None:
    config = load_chainlink_oi_benchmark_config(CONFIG_PATH)

    assert config.source_schema_revision == "f1ff0094753967b91e6dca90e38ce29359af3f3b"
    assert config.control_candidate == "frozen_boundary_alignment_champion"
    assert config.champion.model_key.endswith("20260421-20260720-paper-v1")
    assert config.champion.model_sha256 == (
        "c0778189865ca97a748a9f76cbe72d13268fd6e76db683ea727ad142b0576bc4"
    )
    assert config.paths.champion_model.is_file()
    assert config.paths.champion_manifest.is_file()
    assert config.paths.core_config == (
        PACKAGE_ROOT / "configs" / "btc-5m-directional-core-chainlink-oi-20260321-20260729.toml"
    )
    assert config.paths.refprice_source_sql == (
        PACKAGE_ROOT / "sql" / "btc-chainlink-refprice-source.sql"
    )
    assert config.paths.candles_source_sql == (
        PACKAGE_ROOT / "sql" / "btc-chainlink-one-minute-candles-source.sql"
    )
    assert config.paths.open_interest_source_sql == (
        PACKAGE_ROOT / "sql" / "btc-binance-five-minute-open-interest-source.sql"
    )
    assert config.paths.execution_evidence == (
        PACKAGE_ROOT / "data" / "btc-chainlink-oi-champion-benchmark" / "execution-evidence"
    )
    assert config.paths.shared_cache.is_relative_to(PACKAGE_ROOT)
    assert config.paths.runs.is_relative_to(PACKAGE_ROOT)
    assert config.sources.refprice_feed_id.endswith("95ed75b8")
    assert config.sources.polygon_oracle_proxy.endswith("d57f6f")


def test_config_freezes_training_and_evaluation_windows() -> None:
    windows = load_chainlink_oi_benchmark_config(CONFIG_PATH).windows

    assert (windows.source_range_start, windows.source_range_end) == (
        datetime(2026, 3, 21, tzinfo=UTC),
        datetime(2026, 7, 29, tzinfo=UTC),
    )
    assert (windows.short_fit_start, windows.short_fit_end) == (
        datetime(2026, 7, 3, tzinfo=UTC),
        datetime(2026, 7, 14, tzinfo=UTC),
    )
    assert (windows.calibration_start, windows.calibration_end) == (
        datetime(2026, 7, 14, tzinfo=UTC),
        datetime(2026, 7, 16, tzinfo=UTC),
    )
    assert (
        windows.economic_confirmation_start,
        windows.economic_confirmation_end,
    ) == (
        datetime(2026, 7, 16, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
    )
    assert (windows.directional_stress_start, windows.directional_stress_end) == (
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 7, 29, tzinfo=UTC),
    )


def test_config_freezes_candidate_ablation_and_hgb_contract() -> None:
    config = load_chainlink_oi_benchmark_config(CONFIG_PATH)

    assert config.candidate_names == CHAINLINK_OI_CANDIDATE_NAMES
    assert config.candidate_names == (
        CHAINLINK_FULL_CANDIDATE,
        CHAINLINK_FULL_OI_CANDIDATE,
        LONG_HISTORY_CANDLE_CANDIDATE,
    )
    no_oi, with_oi, long_history = config.candidates
    assert (
        no_oi.include_refprice,
        no_oi.include_oracle,
        no_oi.include_candles,
        no_oi.include_open_interest,
    ) == (True, True, True, False)
    assert (
        with_oi.include_refprice,
        with_oi.include_oracle,
        with_oi.include_candles,
        with_oi.include_open_interest,
    ) == (True, True, True, True)
    assert long_history.fit_window == "long_history"
    assert not long_history.include_refprice
    assert not long_history.include_open_interest
    assert config.model.decision_seconds == tuple(range(60, 241, 5))
    assert config.model.confidence_threshold == pytest.approx(0.89)
    assert (
        config.model.learning_rate,
        config.model.max_iter,
        config.model.max_leaf_nodes,
        config.model.min_samples_leaf,
        config.model.l2_regularization,
    ) == (0.05, 160, 15, 100, 0.1)


def test_config_disables_leakage_imputation_and_fallback_promotion() -> None:
    config = load_chainlink_oi_benchmark_config(CONFIG_PATH)

    assert config.causality.require_source_at_or_before_decision
    assert config.causality.require_closed_candles
    assert config.causality.require_completed_open_interest_bucket
    assert config.causality.strict_common_rows_for_short_candidates
    assert not config.causality.allow_imputation
    assert not config.causality.allow_missingness_features
    assert not config.causality.allow_market_rows_across_splits
    assert config.staleness.refprice_seconds == 5
    assert config.staleness.open_interest_seconds == 600
    assert config.promotion.minimum_loss_capture_rate == pytest.approx(0.20)
    assert config.promotion.minimum_win_retention_rate == pytest.approx(0.80)
    assert config.promotion.minimum_gross_loss_reduction == pytest.approx(0.10)
    assert config.promotion.paper_only
    assert not config.promotion.allow_fallback_winner


def test_config_rejects_candidate_or_causality_drift() -> None:
    config = load_chainlink_oi_benchmark_config(CONFIG_PATH)
    candidate_drift = replace(
        config,
        candidates=(
            replace(config.candidates[0], include_open_interest=True),
            *config.candidates[1:],
        ),
    )
    with pytest.raises(ValueError, match="feature-ablation"):
        validate_chainlink_oi_benchmark_config(candidate_drift)

    causality_drift = replace(
        config,
        causality=replace(config.causality, allow_imputation=True),
    )
    with pytest.raises(ValueError, match="imputation"):
        validate_chainlink_oi_benchmark_config(causality_drift)


def test_config_rejects_champion_or_promotion_drift() -> None:
    config = load_chainlink_oi_benchmark_config(CONFIG_PATH)
    champion_drift = replace(
        config,
        champion=replace(config.champion, model_sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="champion identity"):
        validate_chainlink_oi_benchmark_config(champion_drift)

    fallback_drift = replace(
        config,
        promotion=replace(config.promotion, allow_fallback_winner=True),
    )
    with pytest.raises(ValueError, match="no-fallback"):
        validate_chainlink_oi_benchmark_config(fallback_drift)
