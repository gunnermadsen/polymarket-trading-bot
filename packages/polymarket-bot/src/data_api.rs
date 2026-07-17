use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::{Client, Url};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::models::{DataApiActivity, DataApiPosition, DataApiValue};

#[derive(Debug, Clone)]
pub struct DataApiClient {
    http: Client,
    base_url: String,
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

    pub async fn fetch_positions(&self, query: &PositionsQuery) -> Result<Vec<DataApiPosition>> {
        self.fetch_endpoint("positions", positions_params(query))
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
