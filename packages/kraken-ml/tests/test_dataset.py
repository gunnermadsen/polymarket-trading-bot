from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

from kraken_ml.config import load_config
from kraken_ml.dataset import _immutable_json, validate_snapshot_frame


def _frame_config(config_path: Path, frame: pl.DataFrame):
    config = load_config(config_path)
    interval = timedelta(seconds=config.dataset.interval_seconds)
    return replace(
        config,
        dataset=replace(
            config.dataset,
            start=frame.item(0, "bucket_start"),
            end=frame.item(frame.height - 1, "bucket_start") + interval,
        ),
    )


def test_snapshot_validation_requires_exact_configured_time_range(
    config_path: Path, raw_market_frame: pl.DataFrame
) -> None:
    config = _frame_config(config_path, raw_market_frame)
    validate_snapshot_frame(raw_market_frame, config)

    with pytest.raises(RuntimeError, match="row count"):
        validate_snapshot_frame(raw_market_frame.slice(1), config)


def test_snapshot_validation_requires_two_sided_slippage(
    config_path: Path, raw_market_frame: pl.DataFrame
) -> None:
    config = _frame_config(config_path, raw_market_frame)
    bid = raw_market_frame["bid_slippage_1k"].to_list()
    bid[10] = None
    incomplete = raw_market_frame.with_columns(pl.Series("bid_slippage_1k", bid))

    with pytest.raises(RuntimeError, match="two-sided coverage"):
        validate_snapshot_frame(incomplete, config)


def test_content_addressed_manifest_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    _immutable_json(path, {"schema_version": 1, "value": "fixed"})
    _immutable_json(path, {"schema_version": 1, "value": "fixed"})

    with pytest.raises(RuntimeError, match="immutable manifest content changed"):
        _immutable_json(path, {"schema_version": 1, "value": "changed"})
