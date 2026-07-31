from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from btc_directional_model.continuous_context_features import (
    CONTINUOUS_CONTEXT_MODEL_FEATURES,
    derive_continuous_context_features,
    join_continuous_context_features,
)


def test_continuous_context_crosses_market_boundary_causally() -> None:
    start = datetime(2026, 4, 13, tzinfo=UTC)
    count = 181
    observed = [start + timedelta(seconds=index) for index in range(count)]
    rows = pl.DataFrame(
        {
            "market_id": ["a" if index < 60 else "b" for index in range(count)],
            "window_start": [
                start if index < 60 else start + timedelta(seconds=60)
                for index in range(count)
            ],
            "observed_at": observed,
            "seconds_elapsed": [
                index if index < 60 else index - 60 for index in range(count)
            ],
            "btc_close": np.exp(np.arange(count) * 0.0001),
            "btc_quote_volume": [10.0] * count,
            "btc_taker_buy_quote_volume": [6.0] * count,
        }
    )

    derived = derive_continuous_context_features(rows)
    point = derived.row(120, named=True)

    assert point["continuous_return_90s_bps"] == pytest.approx(90.0)
    assert point["continuous_return_120s_bps"] == pytest.approx(120.0)
    assert point["continuous_signed_flow_90s"] == pytest.approx(0.2)
    assert point["continuous_signed_flow_120s"] == pytest.approx(0.2)
    assert point["continuous_realized_volatility_90s_bps"] == pytest.approx(
        0.0, abs=1e-10
    )
    assert point["continuous_realized_volatility_120s_bps"] == pytest.approx(
        0.0, abs=1e-10
    )


def test_context_join_requires_finite_exact_key_features() -> None:
    observed_at = datetime(2026, 4, 13, tzinfo=UTC)
    core = pl.DataFrame(
        {
            "market_id": ["m"],
            "window_start": [observed_at],
            "observed_at": [observed_at],
        }
    )
    context = core.with_columns(
        *[
            pl.lit(1.0).alias(feature)
            for feature in CONTINUOUS_CONTEXT_MODEL_FEATURES
        ]
    )

    joined = join_continuous_context_features(core, context)

    assert joined["continuous_context_model_eligible"].to_list() == [True]
