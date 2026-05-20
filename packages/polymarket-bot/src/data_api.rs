use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::{Client, Url};
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::models::{
    DataApiActivity, DataApiClosedPosition, DataApiPosition, DataApiValue, WhaleTrade,
};

#[derive(Debug, Clone)]
pub struct DataApiClient {
    http: Client,
    base_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradesQuery {
    pub limit: usize,
    pub offset: usize,
    pub min_trade_usd: Decimal,
    pub user: Option<String>,
    pub market: Option<String>,
    pub event_id: Option<i64>,
    pub side: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PositionsQuery {
    pub user: String,
    #[serde(default)]
    pub markets: Vec<String>,
    #[serde(default)]
    pub event_ids: Vec<i64>,
    pub size_threshold: Option<Decimal>,
    pub redeemable: Option<bool>,
    pub mergeable: Option<bool>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub sort_by: Option<String>,
    pub sort_direction: Option<String>,
    pub title: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClosedPositionsQuery {
    pub user: String,
    #[serde(default)]
    pub markets: Vec<String>,
    pub title: Option<String>,
    #[serde(default)]
    pub event_ids: Vec<i64>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub sort_by: Option<String>,
    pub sort_direction: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActivityQuery {
    pub user: String,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    #[serde(default)]
    pub markets: Vec<String>,
    #[serde(default)]
    pub event_ids: Vec<i64>,
    #[serde(default)]
    pub activity_types: Vec<String>,
    pub start: Option<i64>,
    pub end: Option<i64>,
    pub sort_by: Option<String>,
    pub sort_direction: Option<String>,
    pub side: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValueQuery {
    pub user: String,
    #[serde(default)]
    pub markets: Vec<String>,
}

impl TradesQuery {
    pub fn whale_page(limit: usize, offset: usize, min_trade_usd: Decimal) -> Self {
        Self {
            limit,
            offset,
            min_trade_usd,
            user: None,
            market: None,
            event_id: None,
            side: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiTrade {
    pub proxy_wallet: Option<String>,
    pub asset: Option<String>,
    pub condition_id: Option<String>,
    pub size: Option<serde_json::Value>,
    pub price: Option<serde_json::Value>,
    pub timestamp: Option<serde_json::Value>,
    pub title: Option<String>,
    pub slug: Option<String>,
    pub event_slug: Option<String>,
    pub outcome: Option<String>,
    pub side: Option<String>,
    pub transaction_hash: Option<String>,
}

impl DataApiClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }

    pub fn with_http_client(base_url: impl Into<String>, http: Client) -> Self {
        Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }

    pub fn polymarket() -> Self {
        Self::new("https://data-api.polymarket.com")
    }

    pub async fn fetch_trades(&self, query: &TradesQuery) -> Result<Vec<ApiTrade>> {
        let mut url = Url::parse(&format!("{}/trades", self.base_url))
            .context("invalid Polymarket Data API trades URL")?;
        {
            let mut pairs = url.query_pairs_mut();
            pairs.append_pair("limit", &query.limit.to_string());
            pairs.append_pair("offset", &query.offset.to_string());
            pairs.append_pair("takerOnly", "true");
            pairs.append_pair("filterType", "CASH");
            pairs.append_pair("filterAmount", &query.min_trade_usd.to_string());
            if let Some(user) = &query.user {
                pairs.append_pair("user", user);
            }
            if let Some(market) = &query.market {
                pairs.append_pair("market", market);
            }
            if let Some(event_id) = query.event_id {
                pairs.append_pair("eventId", &event_id.to_string());
            }
            if let Some(side) = &query.side {
                pairs.append_pair("side", side);
            }
        }

        self.http
            .get(url)
            .send()
            .await
            .context("failed to request Polymarket trades")?
            .error_for_status()
            .context("Polymarket trades response was not successful")?
            .json::<Vec<ApiTrade>>()
            .await
            .context("failed to decode Polymarket trades")
    }

    pub async fn fetch_positions(&self, query: &PositionsQuery) -> Result<Vec<DataApiPosition>> {
        self.fetch_endpoint("positions", positions_params(query))
            .await
    }

    pub async fn fetch_closed_positions(
        &self,
        query: &ClosedPositionsQuery,
    ) -> Result<Vec<DataApiClosedPosition>> {
        self.fetch_endpoint("closed-positions", closed_positions_params(query))
            .await
    }

    pub async fn fetch_activity(&self, query: &ActivityQuery) -> Result<Vec<DataApiActivity>> {
        self.fetch_endpoint("activity", activity_params(query))
            .await
    }

    pub async fn fetch_value(&self, query: &ValueQuery) -> Result<Vec<DataApiValue>> {
        self.fetch_endpoint("value", value_params(query)).await
    }

    async fn fetch_endpoint<T>(&self, path: &str, params: Vec<(String, String)>) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let url = Url::parse(&format!("{}/{}", self.base_url, path))
            .with_context(|| format!("invalid Polymarket Data API {path} URL"))?;
        self.http
            .get(url)
            .query(&params)
            .send()
            .await
            .with_context(|| format!("failed to request Polymarket {path}"))?
            .error_for_status()
            .with_context(|| format!("Polymarket {path} response was not successful"))?
            .json::<T>()
            .await
            .with_context(|| format!("failed to decode Polymarket {path}"))
    }
}

impl PositionsQuery {
    pub fn for_user(user: impl Into<String>) -> Self {
        Self {
            user: user.into(),
            ..Self::default()
        }
    }
}

impl ClosedPositionsQuery {
    pub fn for_user(user: impl Into<String>) -> Self {
        Self {
            user: user.into(),
            ..Self::default()
        }
    }
}

impl ActivityQuery {
    pub fn for_user(user: impl Into<String>) -> Self {
        Self {
            user: user.into(),
            ..Self::default()
        }
    }
}

impl ValueQuery {
    pub fn for_user(user: impl Into<String>) -> Self {
        Self {
            user: user.into(),
            ..Self::default()
        }
    }
}

impl ApiTrade {
    pub fn into_whale_trade(self, min_trade_usd: Decimal) -> Option<WhaleTrade> {
        let raw_payload = serde_json::to_value(&self).unwrap_or_else(|_| serde_json::json!({}));
        let proxy_wallet = self.proxy_wallet?.to_ascii_lowercase();
        let asset = self.asset.unwrap_or_default();
        let price = decimal_from_value(self.price.as_ref())?;
        let size = decimal_from_value(self.size.as_ref())?;
        let cash_value = price * size;
        if cash_value < min_trade_usd {
            return None;
        }
        let timestamp_utc = timestamp_from_value(self.timestamp.as_ref())?;
        let side = self
            .side
            .unwrap_or_else(|| "unknown".to_string())
            .to_ascii_uppercase();
        let transaction_hash = self.transaction_hash;
        let trade_id = stable_trade_id(
            transaction_hash.as_deref(),
            &proxy_wallet,
            &asset,
            &side,
            price,
            size,
            timestamp_utc,
        );
        Some(WhaleTrade {
            trade_id,
            proxy_wallet,
            asset,
            condition_id: self.condition_id,
            market_id: None,
            side,
            outcome: self.outcome,
            price,
            size,
            cash_value,
            timestamp_utc,
            title: self.title,
            slug: self.slug,
            event_slug: self.event_slug,
            transaction_hash,
            raw_payload,
        })
    }
}

fn stable_trade_id(
    transaction_hash: Option<&str>,
    proxy_wallet: &str,
    asset: &str,
    side: &str,
    price: Decimal,
    size: Decimal,
    timestamp_utc: DateTime<Utc>,
) -> Uuid {
    let identity = format!(
        "{}|{}|{}|{}|{}|{}|{}",
        transaction_hash.unwrap_or(""),
        proxy_wallet,
        asset,
        side,
        price.normalize(),
        size.normalize(),
        timestamp_utc.timestamp()
    );
    Uuid::new_v5(&Uuid::NAMESPACE_URL, identity.as_bytes())
}

fn decimal_from_value(value: Option<&serde_json::Value>) -> Option<Decimal> {
    match value? {
        serde_json::Value::String(raw) => raw.parse().ok(),
        serde_json::Value::Number(number) => number.as_f64().and_then(Decimal::from_f64),
        _ => None,
    }
}

fn timestamp_from_value(value: Option<&serde_json::Value>) -> Option<DateTime<Utc>> {
    match value? {
        serde_json::Value::Number(number) => {
            let raw = number.as_i64()?;
            let seconds = if raw > 10_000_000_000 {
                raw / 1000
            } else {
                raw
            };
            Utc.timestamp_opt(seconds, 0).single()
        }
        serde_json::Value::String(raw) => raw
            .parse::<i64>()
            .ok()
            .and_then(|ts| {
                Utc.timestamp_opt(if ts > 10_000_000_000 { ts / 1000 } else { ts }, 0)
                    .single()
            })
            .or_else(|| {
                DateTime::parse_from_rfc3339(raw)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            }),
        _ => None,
    }
}

fn positions_params(query: &PositionsQuery) -> Vec<(String, String)> {
    let mut params = vec![("user".to_string(), query.user.clone())];
    push_csv(&mut params, "market", &query.markets);
    push_csv(&mut params, "eventId", &query.event_ids);
    push_decimal(&mut params, "sizeThreshold", query.size_threshold);
    push_bool(&mut params, "redeemable", query.redeemable);
    push_bool(&mut params, "mergeable", query.mergeable);
    push_usize(&mut params, "limit", query.limit);
    push_usize(&mut params, "offset", query.offset);
    push_string(&mut params, "sortBy", query.sort_by.as_deref());
    push_string(
        &mut params,
        "sortDirection",
        query.sort_direction.as_deref(),
    );
    push_string(&mut params, "title", query.title.as_deref());
    params
}

fn closed_positions_params(query: &ClosedPositionsQuery) -> Vec<(String, String)> {
    let mut params = vec![("user".to_string(), query.user.clone())];
    push_csv(&mut params, "market", &query.markets);
    push_string(&mut params, "title", query.title.as_deref());
    push_csv(&mut params, "eventId", &query.event_ids);
    push_usize(&mut params, "limit", query.limit);
    push_usize(&mut params, "offset", query.offset);
    push_string(&mut params, "sortBy", query.sort_by.as_deref());
    push_string(
        &mut params,
        "sortDirection",
        query.sort_direction.as_deref(),
    );
    params
}

fn activity_params(query: &ActivityQuery) -> Vec<(String, String)> {
    let mut params = vec![("user".to_string(), query.user.clone())];
    push_usize(&mut params, "limit", query.limit);
    push_usize(&mut params, "offset", query.offset);
    push_csv(&mut params, "market", &query.markets);
    push_csv(&mut params, "eventId", &query.event_ids);
    push_csv(&mut params, "type", &query.activity_types);
    push_i64(&mut params, "start", query.start);
    push_i64(&mut params, "end", query.end);
    push_string(&mut params, "sortBy", query.sort_by.as_deref());
    push_string(
        &mut params,
        "sortDirection",
        query.sort_direction.as_deref(),
    );
    push_string(&mut params, "side", query.side.as_deref());
    params
}

fn value_params(query: &ValueQuery) -> Vec<(String, String)> {
    let mut params = vec![("user".to_string(), query.user.clone())];
    push_csv(&mut params, "market", &query.markets);
    params
}

fn push_string(params: &mut Vec<(String, String)>, key: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|value| !value.trim().is_empty()) {
        params.push((key.to_string(), value.to_string()));
    }
}

fn push_usize(params: &mut Vec<(String, String)>, key: &str, value: Option<usize>) {
    if let Some(value) = value {
        params.push((key.to_string(), value.to_string()));
    }
}

fn push_i64(params: &mut Vec<(String, String)>, key: &str, value: Option<i64>) {
    if let Some(value) = value {
        params.push((key.to_string(), value.to_string()));
    }
}

fn push_bool(params: &mut Vec<(String, String)>, key: &str, value: Option<bool>) {
    if let Some(value) = value {
        params.push((key.to_string(), value.to_string()));
    }
}

fn push_decimal(params: &mut Vec<(String, String)>, key: &str, value: Option<Decimal>) {
    if let Some(value) = value {
        params.push((key.to_string(), value.normalize().to_string()));
    }
}

fn push_csv<T>(params: &mut Vec<(String, String)>, key: &str, values: &[T])
where
    T: ToString,
{
    if values.is_empty() {
        return;
    }
    params.push((
        key.to_string(),
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(","),
    ));
}

#[cfg(test)]
mod tests {
    use rust_decimal_macros::dec;

    use super::*;

    #[test]
    fn converts_trade_to_whale_trade_when_cash_threshold_passes() {
        let trade = ApiTrade {
            proxy_wallet: Some("0xABCDEFabcdefABCDEFabcdefABCDEFabcdefabcd".to_string()),
            asset: Some("token".to_string()),
            condition_id: Some("0x01".to_string()),
            size: Some(serde_json::json!("2000")),
            price: Some(serde_json::json!("0.55")),
            timestamp: Some(serde_json::json!(1_700_000_000_i64)),
            title: Some("Question".to_string()),
            slug: None,
            event_slug: None,
            outcome: Some("Yes".to_string()),
            side: Some("BUY".to_string()),
            transaction_hash: Some("0xhash".to_string()),
        };
        let converted = trade.into_whale_trade(dec!(1000)).unwrap();
        assert_eq!(
            converted.proxy_wallet,
            "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
        );
        assert_eq!(converted.cash_value, dec!(1100.00));
    }

    #[test]
    fn positions_params_include_only_set_values_and_csv_lists() {
        let mut query = PositionsQuery::for_user("0xabc");
        query.markets = vec!["0xmarket1".to_string(), "0xmarket2".to_string()];
        query.event_ids = vec![10, 20];
        query.limit = Some(50);
        query.size_threshold = Some(dec!(0.25));

        assert_eq!(
            positions_params(&query),
            vec![
                ("user".to_string(), "0xabc".to_string()),
                ("market".to_string(), "0xmarket1,0xmarket2".to_string()),
                ("eventId".to_string(), "10,20".to_string()),
                ("sizeThreshold".to_string(), "0.25".to_string()),
                ("limit".to_string(), "50".to_string()),
            ]
        );
    }

    #[test]
    fn position_deserializes_decimal_numbers_strings_and_extra_fields() {
        let position: DataApiPosition = serde_json::from_value(serde_json::json!({
            "proxyWallet": "0xabc",
            "size": "12.50",
            "currentValue": 4.2,
            "unexpectedField": "kept"
        }))
        .unwrap();

        assert_eq!(position.size, Some(dec!(12.50)));
        assert_eq!(position.current_value, Some(dec!(4.2)));
        assert_eq!(
            position.extra.get("unexpectedField"),
            Some(&serde_json::Value::String("kept".to_string()))
        );
    }
}
