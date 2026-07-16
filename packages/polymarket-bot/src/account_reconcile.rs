use anyhow::Result;
use chrono::{TimeZone, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    data_api::{ActivityQuery, DataApiClient, PositionsQuery},
    execution::live::LiveVenueEvent,
    models::{DataApiActivity, DataApiPosition},
    store::{AccountPositionSnapshot, AccountTrade, Store},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountReconcileRequest {
    pub account_address: Option<String>,
    pub lookback_hours: Option<i64>,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub token_id: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountPositionMismatch {
    pub token_id: String,
    pub db_open_size: Decimal,
    pub account_size: Decimal,
    pub delta_size: Decimal,
    pub mismatch_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountReconcileReport {
    pub account_address: String,
    pub source: String,
    pub dry_run: bool,
    pub token_id: Option<String>,
    pub lookback_hours: i64,
    pub activities_fetched: usize,
    pub account_trades_detected: usize,
    pub account_trades_inserted: u64,
    pub position_snapshots_detected: usize,
    pub position_snapshots_inserted: u64,
    pub exits_detected: u64,
    pub exits_applied: u64,
    pub exit_size_applied: Decimal,
    pub position_adjustments_detected: u64,
    pub position_adjustments_applied: u64,
    pub position_adjustment_size_applied: Decimal,
    pub mismatches: Vec<AccountPositionMismatch>,
    pub unmatched_trades: u64,
}

pub async fn reconcile_account_positions(
    store: &Store,
    data_api: &DataApiClient,
    request: AccountReconcileRequest,
) -> Result<AccountReconcileReport> {
    let account_address = request
        .account_address
        .as_deref()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let lookback_hours = request.lookback_hours.unwrap_or(24).clamp(1, 24 * 30);
    let source = request
        .source
        .as_deref()
        .unwrap_or(if request.dry_run { "admin" } else { "poll" })
        .to_string();
    let trade_source = match source.as_str() {
        "manual_backfill" => "manual_backfill",
        "poll" => "poll",
        _ => "data_api",
    };
    let start = (Utc::now() - chrono::Duration::hours(lookback_hours)).timestamp();
    let mut activity_query = ActivityQuery::for_user(account_address.clone());
    activity_query.limit = Some(500);
    activity_query.start = Some(start);
    activity_query.sort_by = Some("timestamp".to_string());
    activity_query.sort_direction = Some("desc".to_string());
    let activities = data_api.fetch_activity(&activity_query).await?;
    let mut trades = activities
        .iter()
        .filter_map(|activity| {
            account_trade_from_activity(&account_address, activity, trade_source)
        })
        .filter(|trade| {
            request
                .token_id
                .as_ref()
                .map(|token| token == &trade.token_id)
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    trades.sort_by_key(|trade| trade.timestamp_utc);

    let mut positions_query = PositionsQuery::for_user(account_address.clone());
    positions_query.limit = Some(500);
    positions_query.size_threshold = Some(Decimal::ZERO);
    let positions = data_api.fetch_positions(&positions_query).await?;
    let snapshots = positions
        .iter()
        .filter_map(|position| {
            account_position_snapshot_from_data_api(&account_address, position, trade_source)
        })
        .filter(|snapshot| {
            request
                .token_id
                .as_ref()
                .map(|token| token == &snapshot.token_id)
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    let mut inserted_trades = 0;
    let mut inserted_snapshots = 0;
    if !request.dry_run {
        for trade in &trades {
            if store.upsert_account_trade(trade).await? {
                inserted_trades += 1;
            }
        }
        for snapshot in &snapshots {
            if store.insert_account_position_snapshot(snapshot).await? {
                inserted_snapshots += 1;
            }
        }
    }

    let report = AccountReconcileReport {
        account_address,
        source: source.clone(),
        dry_run: request.dry_run,
        token_id: request.token_id,
        lookback_hours,
        activities_fetched: activities.len(),
        account_trades_detected: trades.len(),
        account_trades_inserted: inserted_trades,
        position_snapshots_detected: snapshots.len(),
        position_snapshots_inserted: inserted_snapshots,
        exits_detected: 0,
        exits_applied: 0,
        exit_size_applied: Decimal::ZERO,
        position_adjustments_detected: 0,
        position_adjustments_applied: 0,
        position_adjustment_size_applied: Decimal::ZERO,
        mismatches: Vec::new(),
        unmatched_trades: 0,
    };
    store.insert_account_reconciliation_run(&report).await?;
    Ok(report)
}

pub fn account_trade_from_live_event(
    account_address: &str,
    event: &LiveVenueEvent,
) -> Option<AccountTrade> {
    if event.event_type != "trade" {
        return None;
    }
    let token_id = json_str(&event.raw_payload, "asset_id")
        .or_else(|| json_str(&event.raw_payload, "asset"))
        .or_else(|| maker_order_str(&event.raw_payload, "asset_id"))?
        .to_string();
    let side = normalize_side(
        json_str(&event.raw_payload, "side")
            .or_else(|| maker_order_str(&event.raw_payload, "side"))
            .unwrap_or(""),
    )?;
    let price = json_decimal(&event.raw_payload, "price")
        .or_else(|| maker_order_decimal(&event.raw_payload, "price"))?;
    let size = json_decimal(&event.raw_payload, "size")
        .or_else(|| json_decimal(&event.raw_payload, "matched_amount"))
        .or_else(|| maker_order_decimal(&event.raw_payload, "matched_amount"))?;
    let timestamp = json_i64(&event.raw_payload, "match_time")
        .or_else(|| json_i64(&event.raw_payload, "timestamp"))
        .and_then(timestamp_from_raw)
        .unwrap_or_else(Utc::now);
    let venue_trade_id = event
        .venue_trade_id
        .clone()
        .or_else(|| json_str(&event.raw_payload, "trade_id").map(str::to_string));
    Some(AccountTrade::new(
        account_address,
        &token_id,
        json_str(&event.raw_payload, "market").map(str::to_string),
        side,
        price,
        size,
        timestamp,
        json_str(&event.raw_payload, "transaction_hash").map(str::to_string),
        event.venue_order_id.clone(),
        venue_trade_id,
        "user_ws",
        event.raw_payload.clone(),
    ))
}

fn account_trade_from_activity(
    account_address: &str,
    activity: &DataApiActivity,
    source: &str,
) -> Option<AccountTrade> {
    let token_id = activity.asset.clone()?;
    let side = normalize_side(activity.side.as_deref()?)?;
    let price = activity.price?;
    let size = activity.size?;
    let timestamp = timestamp_from_raw(activity.timestamp?)?;
    let mut raw_payload = serde_json::to_value(activity).ok()?;
    if let serde_json::Value::Object(object) = &mut raw_payload {
        object.insert("normalizer_source".to_string(), serde_json::json!(source));
    }
    Some(AccountTrade::new(
        account_address,
        &token_id,
        activity.condition_id.clone(),
        side,
        price,
        size,
        timestamp,
        activity.transaction_hash.clone(),
        None,
        None,
        source,
        raw_payload,
    ))
}

fn account_position_snapshot_from_data_api(
    account_address: &str,
    position: &DataApiPosition,
    source: &str,
) -> Option<AccountPositionSnapshot> {
    let token_id = position.asset.clone()?;
    let raw_payload = serde_json::to_value(position).ok()?;
    Some(AccountPositionSnapshot {
        snapshot_id: Uuid::new_v4(),
        account_address: account_address.to_ascii_lowercase(),
        token_id,
        market_id: position.condition_id.clone(),
        size: position.size.unwrap_or(Decimal::ZERO).max(Decimal::ZERO),
        avg_price: position.avg_price,
        current_price: position.cur_price,
        current_value: position.current_value,
        cash_pnl: position.cash_pnl,
        percent_pnl: position.percent_pnl,
        snapshot_at: Utc::now(),
        source: source.to_string(),
        raw_payload,
    })
}

fn normalize_side(value: &str) -> Option<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "buy" | "bought" => Some("buy"),
        "sell" | "sold" => Some("sell"),
        _ => None,
    }
}

fn timestamp_from_raw(raw: i64) -> Option<chrono::DateTime<Utc>> {
    let seconds = if raw > 10_000_000_000 {
        raw / 1000
    } else {
        raw
    };
    Utc.timestamp_opt(seconds, 0).single()
}

fn json_str<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(|value| value.as_str())
}

fn json_i64(value: &serde_json::Value, key: &str) -> Option<i64> {
    match value.get(key)? {
        serde_json::Value::Number(number) => number.as_i64(),
        serde_json::Value::String(raw) => raw.parse().ok(),
        _ => None,
    }
}

fn json_decimal(value: &serde_json::Value, key: &str) -> Option<Decimal> {
    decimal_from_json(value.get(key)?)
}

fn maker_order_str<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value
        .get("maker_orders")
        .and_then(|orders| orders.as_array())
        .and_then(|orders| orders.first())
        .and_then(|order| order.get(key))
        .and_then(|value| value.as_str())
}

fn maker_order_decimal(value: &serde_json::Value, key: &str) -> Option<Decimal> {
    value
        .get("maker_orders")
        .and_then(|orders| orders.as_array())
        .and_then(|orders| orders.first())
        .and_then(|order| order.get(key))
        .and_then(decimal_from_json)
}

fn decimal_from_json(value: &serde_json::Value) -> Option<Decimal> {
    match value {
        serde_json::Value::String(raw) => raw.parse().ok(),
        serde_json::Value::Number(number) => number.to_string().parse().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use rust_decimal_macros::dec;

    use super::*;

    #[test]
    fn websocket_trade_normalizes_to_account_trade() {
        let event = LiveVenueEvent {
            source: "user_ws".to_string(),
            event_type: "trade".to_string(),
            venue_event_id: Some("evt-1".to_string()),
            venue_order_id: Some("order-1".to_string()),
            venue_trade_id: Some("trade-1".to_string()),
            event_status: Some("MATCHED".to_string()),
            raw_payload: serde_json::json!({
                "asset_id": "token-1",
                "side": "SELL",
                "price": "0.42",
                "size": "3.5",
                "match_time": 1710000000,
                "transaction_hash": "0xabc"
            }),
        };

        let trade = account_trade_from_live_event("0xABC", &event).unwrap();

        assert_eq!(trade.account_address, "0xabc");
        assert_eq!(trade.token_id, "token-1");
        assert_eq!(trade.side, "sell");
        assert_eq!(trade.price, dec!(0.42));
        assert_eq!(trade.size, dec!(3.5));
        assert_eq!(trade.notional, dec!(1.470));
        assert_eq!(trade.source, "user_ws");
    }

    #[test]
    fn websocket_maker_order_fields_are_supported() {
        let event = LiveVenueEvent {
            source: "user_ws".to_string(),
            event_type: "trade".to_string(),
            venue_event_id: None,
            venue_order_id: None,
            venue_trade_id: Some("trade-2".to_string()),
            event_status: Some("MATCHED".to_string()),
            raw_payload: serde_json::json!({
                "maker_orders": [{
                    "asset_id": "token-2",
                    "side": "BUY",
                    "price": "0.55",
                    "matched_amount": "2"
                }],
                "timestamp": 1710000000000i64
            }),
        };

        let trade = account_trade_from_live_event("0xabc", &event).unwrap();

        assert_eq!(trade.token_id, "token-2");
        assert_eq!(trade.side, "buy");
        assert_eq!(trade.price, dec!(0.55));
        assert_eq!(trade.size, dec!(2));
    }
}
