use std::collections::{HashMap, HashSet};

use chrono::{DateTime, TimeZone, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use uuid::Uuid;

use crate::models::{
    DataApiClosedPosition, DataApiPosition, DataApiTrade, WalletPerformance, WalletScore,
    WhaleTrade,
};

pub const SCORE_VERSION: &str = "whale_score_v1";
pub const PERFORMANCE_SCORE_VERSION: &str = "whale_performance_v1";

#[derive(Debug, Clone, Default)]
pub struct WalletScoreInput {
    pub proxy_wallet: String,
    pub trades: Vec<WhaleTrade>,
    pub realized_pnl: Decimal,
    pub total_bought: Decimal,
    pub closed_position_count: i32,
    pub winning_closed_position_count: i32,
}

pub fn score_wallets(trades: &[WhaleTrade]) -> Vec<WalletScore> {
    let mut by_wallet: HashMap<String, Vec<WhaleTrade>> = HashMap::new();
    for trade in trades {
        by_wallet
            .entry(trade.proxy_wallet.clone())
            .or_default()
            .push(trade.clone());
    }

    by_wallet
        .into_iter()
        .map(|(proxy_wallet, trades)| {
            score_wallet(WalletScoreInput {
                proxy_wallet,
                trades,
                ..WalletScoreInput::default()
            })
        })
        .collect()
}

pub fn score_wallet(input: WalletScoreInput) -> WalletScore {
    let total_trades = input.trades.len() as i32;
    let total_volume: Decimal = input.trades.iter().map(|trade| trade.cash_value).sum();
    let avg_trade_size = if total_trades > 0 {
        total_volume / Decimal::from(total_trades)
    } else {
        Decimal::ZERO
    };
    let markets: HashSet<String> = input
        .trades
        .iter()
        .filter_map(|trade| trade.condition_id.clone().or_else(|| trade.slug.clone()))
        .collect();
    let resolved_markets = markets.len() as i32;
    let capital_basis = input.total_bought.max(total_volume);
    let roi = if capital_basis > Decimal::ZERO {
        input.realized_pnl / capital_basis
    } else {
        Decimal::ZERO
    };
    let win_rate = if input.closed_position_count > 0 {
        Decimal::from(input.winning_closed_position_count)
            / Decimal::from(input.closed_position_count)
    } else {
        Decimal::ZERO
    };

    let volume_score = bounded(total_volume / dec!(100000), dec!(0), dec!(1)) * dec!(20);
    let trade_count_score =
        bounded(Decimal::from(total_trades) / dec!(100), dec!(0), dec!(1)) * dec!(15);
    let market_count_score =
        bounded(Decimal::from(resolved_markets) / dec!(25), dec!(0), dec!(1)) * dec!(15);
    let avg_size_score = bounded(avg_trade_size / dec!(5000), dec!(0), dec!(1)) * dec!(15);
    let pnl_score = normalize_signed(input.realized_pnl, dec!(1000)) * dec!(20);
    let roi_score = normalize_signed(roi, dec!(1)) * dec!(10);
    let win_rate_score = bounded(win_rate, Decimal::ZERO, Decimal::ONE) * dec!(5);
    let concentration_penalty = if resolved_markets <= 1 && total_trades >= 10 {
        dec!(10)
    } else {
        Decimal::ZERO
    };
    let score = bounded(
        volume_score
            + trade_count_score
            + market_count_score
            + avg_size_score
            + pnl_score
            + roi_score
            + win_rate_score
            - concentration_penalty,
        Decimal::ZERO,
        dec!(100),
    );

    WalletScore {
        proxy_wallet: input.proxy_wallet,
        score_version: SCORE_VERSION.to_string(),
        resolved_markets,
        total_trades,
        total_volume,
        realized_pnl: input.realized_pnl,
        roi,
        win_rate,
        avg_trade_size,
        max_drawdown: Decimal::ZERO,
        score: score.round_dp(4),
        metadata: serde_json::json!({
            "score_basis": "trade_activity_and_pnl_v1",
            "closed_position_count": input.closed_position_count
        }),
    }
}

pub fn rank_wallet_scores(mut scores: Vec<WalletScore>) -> Vec<WalletScore> {
    scores.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.proxy_wallet.cmp(&right.proxy_wallet))
    });
    scores
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WalletPerformanceScore {
    pub proxy_wallet: String,
    pub realized_pnl_usd: Decimal,
    pub total_bought_usd: Decimal,
    pub roi: Decimal,
    pub closed_positions: i32,
    pub winning_positions: i32,
    pub win_rate: Decimal,
    pub rank_score: Decimal,
}

impl WalletPerformanceScore {
    pub fn into_wallet_performance(
        self,
        sample_updated_at: DateTime<Utc>,
        raw_payload: serde_json::Value,
    ) -> WalletPerformance {
        WalletPerformance {
            proxy_wallet: self.proxy_wallet,
            sample_updated_at,
            realized_pnl_usd: self.realized_pnl_usd,
            total_bought_usd: self.total_bought_usd,
            roi: self.roi,
            closed_positions: self.closed_positions,
            winning_positions: self.winning_positions,
            win_rate: self.win_rate,
            rank_score: self.rank_score,
            raw_payload,
            metadata: serde_json::json!({
                "score_basis": "closed_positions_performance_v1",
                "rank_formula": "realized_pnl_usd * roi"
            }),
        }
    }

    pub fn into_wallet_score(self) -> WalletScore {
        let score = bounded(self.rank_score, Decimal::ZERO, dec!(100)).round_dp(4);
        WalletScore {
            proxy_wallet: self.proxy_wallet,
            score_version: PERFORMANCE_SCORE_VERSION.to_string(),
            resolved_markets: self.closed_positions,
            total_trades: 0,
            total_volume: self.total_bought_usd,
            realized_pnl: self.realized_pnl_usd,
            roi: self.roi,
            win_rate: self.win_rate,
            avg_trade_size: Decimal::ZERO,
            max_drawdown: Decimal::ZERO,
            score,
            metadata: serde_json::json!({
                "score_basis": "closed_positions_performance_v1",
                "realized_pnl_usd": self.realized_pnl_usd,
                "total_bought_usd": self.total_bought_usd,
                "closed_positions": self.closed_positions,
                "winning_positions": self.winning_positions,
                "rank_score": self.rank_score
            }),
        }
    }
}

pub fn score_closed_position_performance(
    proxy_wallet: impl Into<String>,
    closed_positions: &[DataApiClosedPosition],
) -> WalletPerformanceScore {
    let proxy_wallet = proxy_wallet.into().to_ascii_lowercase();
    let mut score = WalletPerformanceScore {
        proxy_wallet: proxy_wallet.clone(),
        ..WalletPerformanceScore::default()
    };

    for position in closed_positions
        .iter()
        .filter(|position| wallet_matches(&proxy_wallet, position.proxy_wallet.as_deref()))
    {
        let realized_pnl = position.realized_pnl.unwrap_or(Decimal::ZERO);
        score.realized_pnl_usd += realized_pnl;
        score.total_bought_usd += position.total_bought.unwrap_or(Decimal::ZERO);
        score.closed_positions = score.closed_positions.saturating_add(1);
        if realized_pnl > Decimal::ZERO {
            score.winning_positions = score.winning_positions.saturating_add(1);
        }
    }

    score.roi = if score.total_bought_usd > Decimal::ZERO {
        score.realized_pnl_usd / score.total_bought_usd
    } else {
        Decimal::ZERO
    };
    score.win_rate = if score.closed_positions > 0 {
        Decimal::from(score.winning_positions) / Decimal::from(score.closed_positions)
    } else {
        Decimal::ZERO
    };
    score.rank_score = score.realized_pnl_usd * score.roi;
    score
}

pub fn score_closed_position_wallets(
    closed_positions_by_wallet: &[(String, Vec<DataApiClosedPosition>)],
) -> Vec<WalletScore> {
    rank_wallet_scores(
        closed_positions_by_wallet
            .iter()
            .map(|(wallet, closed_positions)| {
                score_closed_position_performance(wallet, closed_positions).into_wallet_score()
            })
            .collect(),
    )
}

pub fn score_input_from_data_api(
    proxy_wallet: impl Into<String>,
    positions: &[DataApiPosition],
    closed_positions: &[DataApiClosedPosition],
    trades: &[DataApiTrade],
) -> WalletScoreInput {
    let proxy_wallet = proxy_wallet.into().to_ascii_lowercase();
    let mut input = WalletScoreInput {
        proxy_wallet: proxy_wallet.clone(),
        ..WalletScoreInput::default()
    };

    for position in positions
        .iter()
        .filter(|position| wallet_matches(&proxy_wallet, position.proxy_wallet.as_deref()))
    {
        input.realized_pnl += position.realized_pnl.unwrap_or(Decimal::ZERO);
        input.realized_pnl += position.cash_pnl.unwrap_or(Decimal::ZERO);
        input.total_bought += position.total_bought.unwrap_or(Decimal::ZERO);
    }

    for position in closed_positions
        .iter()
        .filter(|position| wallet_matches(&proxy_wallet, position.proxy_wallet.as_deref()))
    {
        let realized_pnl = position.realized_pnl.unwrap_or(Decimal::ZERO);
        input.realized_pnl += realized_pnl;
        input.total_bought += position.total_bought.unwrap_or(Decimal::ZERO);
        input.closed_position_count = input.closed_position_count.saturating_add(1);
        if realized_pnl > Decimal::ZERO {
            input.winning_closed_position_count =
                input.winning_closed_position_count.saturating_add(1);
        }
    }

    input.trades = trades
        .iter()
        .filter(|trade| wallet_matches(&proxy_wallet, trade.proxy_wallet.as_deref()))
        .filter_map(api_trade_to_whale_trade)
        .collect();

    input
}

fn bounded(value: Decimal, min: Decimal, max: Decimal) -> Decimal {
    value.max(min).min(max)
}

fn normalize_signed(value: Decimal, saturation: Decimal) -> Decimal {
    if saturation <= Decimal::ZERO {
        return dec!(0.5);
    }
    let clamped = bounded(value, -saturation, saturation);
    (clamped + saturation) / (saturation * dec!(2))
}

fn wallet_matches(expected_lowercase: &str, actual: Option<&str>) -> bool {
    actual
        .map(|actual| actual.eq_ignore_ascii_case(expected_lowercase))
        .unwrap_or(false)
}

fn api_trade_to_whale_trade(trade: &DataApiTrade) -> Option<WhaleTrade> {
    let price = trade.price?;
    let size = trade.size?;
    let timestamp_utc = trade
        .timestamp
        .and_then(|timestamp| {
            let seconds = if timestamp > 10_000_000_000 {
                timestamp / 1000
            } else {
                timestamp
            };
            Utc.timestamp_opt(seconds, 0).single()
        })
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).unwrap());
    Some(WhaleTrade {
        trade_id: Uuid::nil(),
        proxy_wallet: trade.proxy_wallet.clone()?.to_ascii_lowercase(),
        asset: trade.asset.clone().unwrap_or_default(),
        condition_id: trade.condition_id.clone(),
        market_id: None,
        side: trade
            .side
            .clone()
            .unwrap_or_else(|| "unknown".to_string())
            .to_ascii_uppercase(),
        outcome: trade.outcome.clone(),
        price,
        size,
        cash_value: price * size,
        timestamp_utc,
        title: trade.title.clone(),
        slug: trade.slug.clone(),
        event_slug: trade.event_slug.clone(),
        transaction_hash: trade.transaction_hash.clone(),
        raw_payload: serde_json::to_value(trade).unwrap_or_else(|_| serde_json::json!({})),
    })
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    use super::*;

    fn trade(wallet: &str, condition: &str, cash: Decimal) -> WhaleTrade {
        WhaleTrade {
            trade_id: Uuid::new_v4(),
            proxy_wallet: wallet.to_string(),
            asset: "asset".to_string(),
            condition_id: Some(condition.to_string()),
            market_id: None,
            side: "BUY".to_string(),
            outcome: Some("Yes".to_string()),
            price: dec!(0.5),
            size: cash / dec!(0.5),
            cash_value: cash,
            timestamp_utc: Utc::now(),
            title: None,
            slug: None,
            event_slug: None,
            transaction_hash: None,
            raw_payload: serde_json::json!({}),
        }
    }

    #[test]
    fn scoring_rewards_volume_and_market_count() {
        let trades = vec![
            trade("0x1", "c1", dec!(50000)),
            trade("0x1", "c2", dec!(50000)),
            trade("0x1", "c3", dec!(50000)),
        ];
        let score = score_wallet(WalletScoreInput {
            proxy_wallet: "0x1".to_string(),
            trades,
            realized_pnl: Decimal::ZERO,
            total_bought: Decimal::ZERO,
            closed_position_count: 0,
            winning_closed_position_count: 0,
        });
        assert!(score.score > dec!(40));
        assert_eq!(score.resolved_markets, 3);
    }

    #[test]
    fn scoring_rewards_profit_and_is_deterministic() {
        let trades = vec![
            trade("0x1", "c1", dec!(1000)),
            trade("0x1", "c2", dec!(1000)),
        ];
        let input = WalletScoreInput {
            proxy_wallet: "0x1".to_string(),
            trades,
            realized_pnl: dec!(250),
            total_bought: dec!(1000),
            closed_position_count: 4,
            winning_closed_position_count: 3,
        };

        let first = score_wallet(input.clone());
        let second = score_wallet(input);

        assert_eq!(first.score, second.score);
        assert_eq!(first.roi, dec!(0.125));
        assert_eq!(first.win_rate, dec!(0.75));
        assert!(first.score > dec!(25));
    }

    #[test]
    fn ranks_equal_scores_by_wallet_address() {
        let scores = vec![
            score_wallet(WalletScoreInput {
                proxy_wallet: "0xbbb".to_string(),
                ..WalletScoreInput::default()
            }),
            score_wallet(WalletScoreInput {
                proxy_wallet: "0xaaa".to_string(),
                ..WalletScoreInput::default()
            }),
        ];

        let ranked = rank_wallet_scores(scores);

        assert_eq!(ranked[0].proxy_wallet, "0xaaa");
        assert_eq!(ranked[1].proxy_wallet, "0xbbb");
    }

    #[test]
    fn aggregates_data_api_wallet_inputs_case_insensitively() {
        let positions: Vec<DataApiPosition> = serde_json::from_value(serde_json::json!([
            {
                "proxyWallet": "0xABC",
                "cashPnl": "10",
                "realizedPnl": "5",
                "totalBought": "80"
            },
            {
                "proxyWallet": "0xdef",
                "cashPnl": "999"
            }
        ]))
        .unwrap();
        let closed_positions: Vec<DataApiClosedPosition> =
            serde_json::from_value(serde_json::json!([
                {
                    "proxyWallet": "0xabc",
                    "realizedPnl": "20",
                    "totalBought": "40"
                },
                {
                    "proxyWallet": "0xabc",
                    "realizedPnl": "-3",
                    "totalBought": "10"
                }
            ]))
            .unwrap();
        let trades: Vec<DataApiTrade> = serde_json::from_value(serde_json::json!([
            { "proxyWallet": "0xabc", "asset": "a", "price": "0.5", "size": "10" },
            { "proxyWallet": "0xABC", "asset": "b", "price": "0.25", "size": "20" },
            { "proxyWallet": "0xdef", "asset": "c", "price": "1", "size": "999" }
        ]))
        .unwrap();

        let input = score_input_from_data_api("0xabc", &positions, &closed_positions, &trades);

        assert_eq!(input.proxy_wallet, "0xabc");
        assert_eq!(input.realized_pnl, dec!(32));
        assert_eq!(input.total_bought, dec!(130));
        assert_eq!(input.closed_position_count, 2);
        assert_eq!(input.winning_closed_position_count, 1);
        assert_eq!(input.trades.len(), 2);
    }

    #[test]
    fn closed_position_performance_computes_requested_fields() {
        let closed_positions: Vec<DataApiClosedPosition> =
            serde_json::from_value(serde_json::json!([
                { "proxyWallet": "0xABC", "realizedPnl": "25", "totalBought": "100" },
                { "proxyWallet": "0xabc", "realizedPnl": "-5", "totalBought": "50" },
                { "proxyWallet": "0xdef", "realizedPnl": "999", "totalBought": "1" }
            ]))
            .unwrap();

        let score = score_closed_position_performance("0xabc", &closed_positions);

        assert_eq!(score.proxy_wallet, "0xabc");
        assert_eq!(score.realized_pnl_usd, dec!(20));
        assert_eq!(score.total_bought_usd, dec!(150));
        assert_eq!(score.roi, dec!(0.1333333333333333333333333333));
        assert_eq!(score.closed_positions, 2);
        assert_eq!(score.winning_positions, 1);
        assert_eq!(score.win_rate, dec!(0.5));
        assert_eq!(score.rank_score, dec!(2.666666666666666666666666666));
    }

    #[test]
    fn closed_position_wallet_scores_rank_by_rank_score() {
        let high: Vec<DataApiClosedPosition> = serde_json::from_value(serde_json::json!([
            { "proxyWallet": "0xhigh", "realizedPnl": "50", "totalBought": "100" }
        ]))
        .unwrap();
        let low: Vec<DataApiClosedPosition> = serde_json::from_value(serde_json::json!([
            { "proxyWallet": "0xlow", "realizedPnl": "20", "totalBought": "100" }
        ]))
        .unwrap();

        let scores = score_closed_position_wallets(&[
            ("0xlow".to_string(), low),
            ("0xhigh".to_string(), high),
        ]);

        assert_eq!(scores[0].proxy_wallet, "0xhigh");
        assert_eq!(scores[0].score_version, PERFORMANCE_SCORE_VERSION);
        assert_eq!(scores[0].score, dec!(25.0));
        assert_eq!(scores[1].score, dec!(4.0));
    }
}
