from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


@dataclass(frozen=True)
class FeeSchedule:
    enabled: bool
    rate: float
    exponent: float
    taker_only: bool


def fee_schedule_from_market(market: dict[str, Any]) -> FeeSchedule:
    schedule = market.get("fee_schedule") or market.get("feeSchedule") or {}
    if not isinstance(schedule, dict):
        schedule = {}

    raw_rate = schedule.get("rate")
    if raw_rate is None:
        raw_bps = (
            schedule.get("fee_rate_bps")
            or schedule.get("feeRateBps")
            or market.get("feeRateBps")
        )
        raw_rate = float(raw_bps) / 10_000 if raw_bps is not None else 0.0
    rate = float(raw_rate)
    exponent = float(schedule.get("exponent", 1.0))
    taker_only = bool(schedule.get("takerOnly", schedule.get("taker_only", True)))
    explicit_enabled = market.get("feesEnabled")
    if explicit_enabled is None:
        explicit_enabled = market.get("fees_enabled")
    enabled = bool(explicit_enabled) if explicit_enabled is not None else rate > 0

    if not math.isfinite(rate) or rate < 0:
        raise ValueError("fee rate must be finite and non-negative")
    if not math.isfinite(exponent) or exponent <= 0:
        raise ValueError("fee exponent must be finite and positive")
    if enabled and exponent != 1.0:
        raise ValueError(f"unsupported fee exponent: {exponent}")
    if enabled and rate <= 0:
        raise ValueError("fee-enabled market is missing a positive fee rate")
    return FeeSchedule(
        enabled=enabled,
        rate=rate if enabled else 0.0,
        exponent=exponent,
        taker_only=taker_only,
    )


def taker_fee_per_share(
    price: float,
    *,
    quantity: float,
    enabled: bool,
    rate: float,
    exponent: float = 1.0,
) -> float:
    if not math.isfinite(price) or not 0 <= price <= 1:
        raise ValueError("price must be finite and between zero and one")
    if not math.isfinite(quantity) or quantity <= 0:
        raise ValueError("quantity must be finite and positive")
    if not enabled:
        return 0.0
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("enabled taker fee requires a positive finite rate")
    if not math.isfinite(exponent) or exponent <= 0:
        raise ValueError("fee exponent must be finite and positive")
    if exponent != 1.0:
        raise ValueError(f"unsupported fee exponent: {exponent}")
    total = Decimal(str(quantity)) * Decimal(str(rate)) * Decimal(str(price)) * (
        Decimal(1) - Decimal(str(price))
    )
    rounded_total = total.quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)
    return float(rounded_total / Decimal(str(quantity)))
