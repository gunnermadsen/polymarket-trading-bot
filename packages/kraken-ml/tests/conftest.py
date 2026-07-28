from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


@pytest.fixture(scope="session")
def config_path() -> Path:
    return PACKAGE_ROOT / "configs" / "pf_xbtusd_15m_1h.toml"


@pytest.fixture
def raw_market_frame() -> pl.DataFrame:
    """A continuous 15-minute frame with enough history for 96-bar features."""
    row_count = 160
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=15 * index) for index in range(row_count)]

    # Alternating trend regimes produce long, flat, and short net labels after costs.
    prices: list[float] = []
    price = 40_000.0
    moves_bps = [22.0] * 12 + [0.0] * 12 + [-22.0] * 12
    for index in range(row_count):
        price *= math.exp(moves_bps[index % len(moves_bps)] / 10_000.0)
        prices.append(price)

    bid_best = [price * (1.0 - 0.00005) for price in prices]
    ask_best = [price * (1.0 + 0.00005) for price in prices]
    bid_slippage_1k = [price * (1.0 - 0.00010) for price in prices]
    ask_slippage_1k = [price * (1.0 + 0.00010) for price in prices]
    bid_slippage_10k = [price * (1.0 - 0.00025) for price in prices]
    ask_slippage_10k = [price * (1.0 + 0.00025) for price in prices]

    values: dict[str, object] = {
        "bucket_start": timestamps,
        "trade_open": prices,
        "trade_high": [price * 1.0005 for price in prices],
        "trade_low": [price * 0.9995 for price in prices],
        "trade_close": [
            price * math.exp(((-1) ** index) * 2.0 / 10_000.0) for index, price in enumerate(prices)
        ],
        "trade_candle_volume": [100.0 + index for index in range(row_count)],
        "mark_close": [price * 1.0001 for price in prices],
        "spot_close": [price * 0.9999 for price in prices],
        "oi_close": [10_000.0 + 3.0 * index for index in range(row_count)],
        "future_basis": [0.001 + index * 0.000001 for index in range(row_count)],
        "aggressor_differential": [math.sin(index / 4.0) * 10.0 for index in range(row_count)],
        "trade_volume": [200.0 + index for index in range(row_count)],
        "trade_count": [40.0 + index % 7 for index in range(row_count)],
        "cvd": [index * 2.0 + math.sin(index) for index in range(row_count)],
        "buy_volume": [110.0 + index for index in range(row_count)],
        "sell_volume": [90.0 + index for index in range(row_count)],
        "liquidation_volume": [float(index % 11) for index in range(row_count)],
        "bid_best": bid_best,
        "ask_best": ask_best,
        "bid_slippage_1k": bid_slippage_1k,
        "ask_slippage_1k": ask_slippage_1k,
        "bid_slippage_10k": bid_slippage_10k,
        "ask_slippage_10k": ask_slippage_10k,
        "relative_funding_rate": [0.0] * row_count,
    }
    for depth_index, depth in enumerate(("005", "01", "025", "05", "10"), start=1):
        values[f"bid_liquidity_{depth}"] = [
            1_000.0 * depth_index + index for index in range(row_count)
        ]
        values[f"ask_liquidity_{depth}"] = [
            900.0 * depth_index + index for index in range(row_count)
        ]
    return pl.DataFrame(values)
