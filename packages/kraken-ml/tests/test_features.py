from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

from kraken_ml.config import load_config
from kraken_ml.features import (
    FEATURE_SETS,
    FULL_FEATURES,
    TARGET_COLUMNS,
    build_feature_frame,
)


def _frame_config(config_path: Path, raw_market_frame: pl.DataFrame):
    config = load_config(config_path)
    interval = timedelta(seconds=config.dataset.interval_seconds)
    return replace(
        config,
        dataset=replace(
            config.dataset,
            start=raw_market_frame.item(0, "bucket_start"),
            end=raw_market_frame.item(raw_market_frame.height - 1, "bucket_start") + interval,
        ),
    )


def test_next_open_label_offsets_and_cost_accounting(
    config_path: Path, raw_market_frame: pl.DataFrame
) -> None:
    config = _frame_config(config_path, raw_market_frame)
    features = build_feature_frame(raw_market_frame, config)
    row = features.row(0, named=True)
    raw_index = 96
    entry_index = raw_index + 1
    exit_index = raw_index + config.dataset.horizon_bars + 1

    decision_at = raw_market_frame.item(raw_index, "bucket_start")
    expected_entry = raw_market_frame.item(entry_index, "bucket_start")
    expected_exit = raw_market_frame.item(exit_index, "bucket_start")
    entry_price = raw_market_frame.item(entry_index, "trade_open")
    exit_price = raw_market_frame.item(exit_index, "trade_open")
    expected_gross = 10_000.0 * (exit_price / entry_price - 1.0)

    assert row["bucket_start"] == decision_at
    assert row["feature_available_at"] == decision_at + timedelta(minutes=15)
    assert row["entry_at"] == expected_entry
    assert row["label_exit_at"] == expected_exit
    assert row["label_exit_at"] - row["entry_at"] == timedelta(hours=1)
    assert row["gross_forward_bps"] == pytest.approx(expected_gross)
    # Entry uses the latest completed order-book analytics bucket; exit uses
    # the bucket completed at the modeled exit, avoiding post-fill data.
    assert row["market_execution_cost_bps"] == pytest.approx(2.0)
    assert row["fee_cost_bps"] == pytest.approx(10.0)
    assert row["execution_cost_bps"] == pytest.approx(12.0)
    assert row["long_net_bps"] == pytest.approx(expected_gross - 12.0)
    assert row["short_net_bps"] == pytest.approx(-expected_gross - 12.0)


def test_future_funding_changes_target_but_never_feature_vector(
    config_path: Path, raw_market_frame: pl.DataFrame
) -> None:
    config = _frame_config(config_path, raw_market_frame)
    baseline = build_feature_frame(raw_market_frame, config)
    decision_at = baseline.item(0, "bucket_start")

    funding = raw_market_frame["relative_funding_rate"].to_list()
    # The first emitted feature row corresponds to raw row 96, and its target
    # includes funding observations at offsets +1 through +4.
    for index in range(97, 101):
        funding[index] = 0.001
    mutated_raw = raw_market_frame.with_columns(pl.Series("relative_funding_rate", funding))
    mutated = build_feature_frame(mutated_raw, config)

    baseline_row = baseline.filter(pl.col("bucket_start") == decision_at)
    mutated_row = mutated.filter(pl.col("bucket_start") == decision_at)
    assert baseline_row.select(FULL_FEATURES).equals(mutated_row.select(FULL_FEATURES))
    # Kraken reports a rate standardized per hour. Four 15-minute exposures
    # therefore realize one hourly rate, not four hourly rates.
    assert mutated_row.item(0, "funding_horizon_bps") == pytest.approx(10.0)
    assert baseline_row.item(0, "funding_horizon_bps") == pytest.approx(0.0)
    assert mutated_row.item(0, "long_net_bps") == pytest.approx(
        baseline_row.item(0, "long_net_bps") - 10.0
    )
    assert mutated_row.item(0, "short_net_bps") == pytest.approx(
        baseline_row.item(0, "short_net_bps") + 10.0
    )


def test_funding_and_target_columns_are_excluded_from_every_feature_set() -> None:
    forbidden = {"relative_funding_rate", *TARGET_COLUMNS}

    assert set(FEATURE_SETS) == {"price", "flow", "full"}
    for names in FEATURE_SETS.values():
        assert not forbidden.intersection(names)
        assert len(names) == len(set(names))


def test_cvd_features_derive_from_bar_flow_not_resetting_archive_level(
    config_path: Path, raw_market_frame: pl.DataFrame
) -> None:
    config = _frame_config(config_path, raw_market_frame)
    reset_cvd = [float(index % 17) for index in range(raw_market_frame.height)]
    mutated = raw_market_frame.with_columns(pl.Series("cvd", reset_cvd))

    features = build_feature_frame(mutated, config)

    assert features.item(0, "cvd_delta_1") == pytest.approx(20.0)
    assert features.item(0, "cvd_delta_4") == pytest.approx(80.0)
