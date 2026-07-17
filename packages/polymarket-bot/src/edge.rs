use chrono::Duration;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

#[derive(Debug, Clone)]
pub struct ThresholdState {
    pub bootstrap_threshold: Decimal,
    pub realized_slippage: Vec<Decimal>,
    pub tick_size: Decimal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reserves {
    pub threshold: Decimal,
    pub carry_reserve: Decimal,
    pub leg_risk_reserve: Decimal,
    pub conversion_latency_reserve: Decimal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeBreakdown {
    pub gross: Decimal,
    pub fees: Decimal,
    pub reserves: Decimal,
    pub net: Decimal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorstCaseLoss {
    pub amount: Decimal,
    pub cap: Decimal,
}

impl WorstCaseLoss {
    pub fn passes(self) -> bool {
        self.amount <= self.cap
    }
}

pub fn dynamic_threshold(state: &ThresholdState) -> Decimal {
    if state.realized_slippage.len() < 30 {
        return state.bootstrap_threshold;
    }
    let mean = state.realized_slippage.iter().copied().sum::<Decimal>()
        / Decimal::from(state.realized_slippage.len());
    let variance = state
        .realized_slippage
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<Decimal>()
        / Decimal::from(state.realized_slippage.len());
    let std = decimal_sqrt(variance);
    (mean + dec!(2) * std)
        .max(dec!(2) * state.tick_size)
        .max(dec!(0.005))
}

pub fn carry_reserve(expected_resolution_days: Decimal, size: Decimal) -> Decimal {
    let extra_days = (expected_resolution_days - dec!(2)).max(Decimal::ZERO);
    extra_days * dec!(0.005) * size
}

pub fn leg_risk_reserve(size: Decimal, stage_8_to_9_failure_rate: Decimal) -> Decimal {
    let mut reserve = dec!(0.005) * size;
    if stage_8_to_9_failure_rate > dec!(0.20) {
        reserve += dec!(0.005) * size;
    }
    reserve
}

pub fn conversion_latency_reserve(
    p95_latency: Duration,
    observed_price_velocity_per_second: Decimal,
    yes_leg_count: usize,
    size: Decimal,
) -> Decimal {
    let seconds = Decimal::from(p95_latency.num_milliseconds().max(0)) / Decimal::from(1000);
    seconds * observed_price_velocity_per_second * Decimal::from(yes_leg_count) * size
}

pub fn cheap_basket_edge(
    size: Decimal,
    buy_cost_all_yes: Decimal,
    taker_fees: Decimal,
    reserves: Reserves,
) -> EdgeBreakdown {
    let gross = Decimal::ONE * size - buy_cost_all_yes;
    let reserve_total =
        reserves.threshold * size + reserves.carry_reserve + reserves.leg_risk_reserve;
    EdgeBreakdown {
        gross,
        fees: taker_fees,
        reserves: reserve_total,
        net: gross - taker_fees - reserve_total,
    }
}

pub fn expensive_basket_inventory_edge(
    sell_revenue_all_yes: Decimal,
    inventory_cost_basis: Decimal,
    taker_fees: Decimal,
    size: Decimal,
    reserves: Reserves,
) -> EdgeBreakdown {
    let gross = sell_revenue_all_yes - inventory_cost_basis;
    let reserve_total =
        reserves.threshold * size + reserves.carry_reserve + reserves.leg_risk_reserve;
    EdgeBreakdown {
        gross,
        fees: taker_fees,
        reserves: reserve_total,
        net: gross - taker_fees - reserve_total,
    }
}

pub fn expensive_basket_ctf_split_edge(
    sell_revenue_all_yes: Decimal,
    split_cost_usdc: Decimal,
    no_token_unwind_cost: Decimal,
    taker_fees: Decimal,
    size: Decimal,
    reserves: Reserves,
) -> EdgeBreakdown {
    let gross = sell_revenue_all_yes - split_cost_usdc - no_token_unwind_cost;
    let reserve_total =
        reserves.threshold * size + reserves.carry_reserve + reserves.leg_risk_reserve;
    EdgeBreakdown {
        gross,
        fees: taker_fees,
        reserves: reserve_total,
        net: gross - taker_fees - reserve_total,
    }
}

pub fn conversion_edge(
    yes_sell_revenue: Decimal,
    no_buy_cost: Decimal,
    taker_fees: Decimal,
    conversion_cost: Decimal,
    size: Decimal,
    reserves: Reserves,
) -> EdgeBreakdown {
    let gross = yes_sell_revenue - no_buy_cost;
    let reserve_total = reserves.threshold * size + reserves.conversion_latency_reserve;
    EdgeBreakdown {
        gross,
        fees: taker_fees,
        reserves: reserve_total + conversion_cost,
        net: gross - taker_fees - conversion_cost - reserve_total,
    }
}

pub fn cheap_basket_worst_case_loss(
    filled_leg_cost: Decimal,
    unwind_fees: Decimal,
    immediate_unwind_revenue: Decimal,
    cap: Decimal,
) -> WorstCaseLoss {
    WorstCaseLoss {
        amount: filled_leg_cost + unwind_fees - immediate_unwind_revenue,
        cap,
    }
}

pub fn expensive_basket_worst_case_loss(
    inventory_cost_basis: Decimal,
    failed_sell_unwind_revenue: Decimal,
    fees: Decimal,
    residual_unwind_cost: Decimal,
    cap: Decimal,
) -> WorstCaseLoss {
    WorstCaseLoss {
        amount: inventory_cost_basis - failed_sell_unwind_revenue + fees + residual_unwind_cost,
        cap,
    }
}

pub fn conversion_worst_case_loss(
    no_a_cost: Decimal,
    conversion_gas: Decimal,
    no_a_exit_bid: Decimal,
    cap: Decimal,
) -> WorstCaseLoss {
    WorstCaseLoss {
        amount: no_a_cost + conversion_gas - no_a_exit_bid,
        cap,
    }
}

fn decimal_sqrt(value: Decimal) -> Decimal {
    if value <= Decimal::ZERO {
        return Decimal::ZERO;
    }
    Decimal::from_f64_retain(value.to_f64().unwrap_or(0.0).sqrt()).unwrap_or(Decimal::ZERO)
}

#[cfg(test)]
mod tests {
    use chrono::Duration;
    use rust_decimal_macros::dec;

    use super::*;

    #[test]
    fn carry_reserve_only_applies_after_two_days() {
        assert_eq!(carry_reserve(dec!(1.5), dec!(100)), dec!(0.000));
        assert_eq!(carry_reserve(dec!(4), dec!(100)), dec!(1.000));
    }

    #[test]
    fn conversion_latency_reserve_scales_with_legs_and_size() {
        let reserve = conversion_latency_reserve(Duration::seconds(10), dec!(0.001), 3, dec!(100));
        assert_eq!(reserve, dec!(3.000));
    }

    #[test]
    fn conversion_worst_case_uses_no_exit_bid_only() {
        let loss = conversion_worst_case_loss(dec!(50), dec!(1), dec!(45), dec!(10));
        assert_eq!(loss.amount, dec!(6));
        assert!(loss.passes());
    }
}
