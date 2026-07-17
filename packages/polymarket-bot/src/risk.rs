use rust_decimal::Decimal;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RiskDecision {
    Allow,
    Reject(String),
}

#[derive(Debug, Clone)]
pub struct LiveTradingCaps {
    pub max_order_notional_usd: Decimal,
    pub max_open_notional_usd: Decimal,
    pub max_daily_loss_usd: Decimal,
    pub max_open_positions: usize,
}

#[derive(Debug, Clone)]
pub struct LiveTradingState {
    pub current_open_notional_usd: Decimal,
    pub open_positions: usize,
    pub realized_pnl_today: Decimal,
    pub user_ws_healthy: bool,
    pub rest_reconcile_fresh: bool,
    pub idempotency_clean: bool,
    pub unresolved_live_orders: usize,
    pub exit_book_available: bool,
    pub wallet_balance_verified: bool,
}

pub fn evaluate_live_entry(
    caps: &LiveTradingCaps,
    state: &LiveTradingState,
    order_notional: Decimal,
) -> RiskDecision {
    if order_notional <= Decimal::ZERO {
        return RiskDecision::Reject("live_order_notional_not_positive".to_string());
    }
    if order_notional > caps.max_order_notional_usd {
        return RiskDecision::Reject("live_order_notional_cap_exceeded".to_string());
    }
    if state.current_open_notional_usd + order_notional > caps.max_open_notional_usd {
        return RiskDecision::Reject("live_open_notional_cap_exceeded".to_string());
    }
    if state.open_positions >= caps.max_open_positions {
        return RiskDecision::Reject("live_open_position_cap_exceeded".to_string());
    }
    if state.realized_pnl_today <= -caps.max_daily_loss_usd {
        return RiskDecision::Reject("live_daily_loss_cap_reached".to_string());
    }
    if !state.user_ws_healthy {
        return RiskDecision::Reject("live_user_ws_unhealthy".to_string());
    }
    if !state.rest_reconcile_fresh {
        return RiskDecision::Reject("live_rest_reconcile_stale".to_string());
    }
    if !state.idempotency_clean {
        return RiskDecision::Reject("live_idempotency_repair_not_clean".to_string());
    }
    if state.unresolved_live_orders > 0 {
        return RiskDecision::Reject("live_unresolved_orders_present".to_string());
    }
    if !state.exit_book_available {
        return RiskDecision::Reject("live_exit_book_unavailable".to_string());
    }
    if !state.wallet_balance_verified {
        return RiskDecision::Reject("live_wallet_balance_unverified".to_string());
    }
    RiskDecision::Allow
}

pub fn normalize_underlying_key(
    manual_override: Option<&str>,
    category: Option<&str>,
    title: &str,
) -> String {
    if let Some(value) = manual_override.filter(|value| !value.trim().is_empty()) {
        return value.trim().to_ascii_lowercase();
    }
    let category = category.unwrap_or("uncategorized").to_ascii_lowercase();
    let mut words: Vec<String> = title
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|word| word.len() > 2)
        .take(8)
        .map(|word| word.to_ascii_lowercase())
        .collect();
    words.sort();
    words.dedup();
    format!("{}:{}", category, words.join("-"))
}

#[cfg(test)]
mod tests {
    use rust_decimal_macros::dec;

    use super::*;

    #[test]
    fn correlation_key_uses_manual_override_first() {
        assert_eq!(
            normalize_underlying_key(Some("Election:US-2028"), Some("Politics"), "Will A win?"),
            "election:us-2028"
        );
    }

    #[test]
    fn live_entry_rejects_above_two_dollar_order_cap() {
        let caps = LiveTradingCaps {
            max_order_notional_usd: dec!(2),
            max_open_notional_usd: dec!(30),
            max_daily_loss_usd: dec!(10),
            max_open_positions: 6,
        };
        let state = LiveTradingState {
            current_open_notional_usd: dec!(0),
            open_positions: 0,
            realized_pnl_today: dec!(0),
            user_ws_healthy: true,
            rest_reconcile_fresh: true,
            idempotency_clean: true,
            unresolved_live_orders: 0,
            exit_book_available: true,
            wallet_balance_verified: true,
        };

        assert_eq!(
            evaluate_live_entry(&caps, &state, dec!(2.01)),
            RiskDecision::Reject("live_order_notional_cap_exceeded".to_string())
        );
    }

    #[test]
    fn live_entry_allows_clean_two_dollar_canary_order() {
        let caps = LiveTradingCaps {
            max_order_notional_usd: dec!(2),
            max_open_notional_usd: dec!(30),
            max_daily_loss_usd: dec!(10),
            max_open_positions: 6,
        };
        let state = LiveTradingState {
            current_open_notional_usd: dec!(25),
            open_positions: 5,
            realized_pnl_today: dec!(0),
            user_ws_healthy: true,
            rest_reconcile_fresh: true,
            idempotency_clean: true,
            unresolved_live_orders: 0,
            exit_book_available: true,
            wallet_balance_verified: true,
        };

        assert_eq!(
            evaluate_live_entry(&caps, &state, dec!(2)),
            RiskDecision::Allow
        );
    }
}
