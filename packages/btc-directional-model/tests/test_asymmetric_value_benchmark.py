from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from btc_directional_model.asymmetric_value_benchmark import (
    _candidate_grid_summary,
    _frame_content_digest,
    _selected_matched_control,
    _validate_external_core_keys,
)
from btc_directional_model.asymmetric_value_config import (
    load_asymmetric_value_config,
)
from btc_directional_model.asymmetric_value_training import (
    CORE_L2_CANDLES_PRICE,
    CORE_ORACLE_PRICE,
    L2_CANDLES_MATCHED_CORE_PRICE_CONTROL,
    ORACLE_MATCHED_CORE_PRICE_CONTROL,
)


def _core_frame() -> pl.DataFrame:
    start = datetime(2026, 7, 16, tzinfo=UTC)
    return pl.DataFrame(
        {
            "market_id": ["a", "a"],
            "window_start": [start, start],
            "observed_at": [
                start + timedelta(seconds=5),
                start + timedelta(seconds=10),
            ],
            "seconds_elapsed": [5, 10],
            "label_up": [1, 1],
            "btc_close": [100_001.0, 100_002.0],
        }
    )


def test_core_content_digest_is_order_invariant_and_value_sensitive() -> None:
    core = _core_frame()
    reversed_core = core.reverse()
    changed = core.with_columns(
        pl.when(pl.col("seconds_elapsed") == 10)
        .then(999.0)
        .otherwise(pl.col("btc_close"))
        .alias("btc_close")
    )

    assert _frame_content_digest(core) == _frame_content_digest(reversed_core)
    assert _frame_content_digest(core) != _frame_content_digest(changed)


def test_external_cache_validation_binds_inherited_core_values() -> None:
    core = _core_frame()
    external = core.with_columns(pl.lit(1.0).alias("external_feature"))
    changed = external.with_columns(pl.lit(999.0).alias("btc_close"))

    _validate_external_core_keys(external, core)
    with pytest.raises(RuntimeError, match="core values changed"):
        _validate_external_core_keys(changed, core)


def test_selected_enriched_models_have_predeclared_matched_controls() -> None:
    assert _selected_matched_control(CORE_ORACLE_PRICE) == (
        ORACLE_MATCHED_CORE_PRICE_CONTROL
    )
    assert _selected_matched_control(CORE_L2_CANDLES_PRICE) == (
        L2_CANDLES_MATCHED_CORE_PRICE_CONTROL
    )


def test_candidate_grid_materializes_missing_prediction_seconds() -> None:
    config = load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-hunter-20260414-20260802.toml"
    )
    core = _core_frame()

    summary = _candidate_grid_summary(core, core, config)

    by_second = {
        row["seconds_elapsed"]: row for row in summary["by_second"]
    }
    assert by_second[5]["markets"] == 1
    assert by_second[15]["markets"] == 0
    assert summary["minimum_second_market_coverage"] == 0.0
