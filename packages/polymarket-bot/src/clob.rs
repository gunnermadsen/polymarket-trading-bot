use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::Client;
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use serde::Deserialize;

use crate::orderbook::{BookSide, LocalOrderBook};

#[derive(Debug, Clone)]
pub struct ClobClient {
    http: Client,
    base_url: String,
}

#[derive(Debug, Deserialize)]
struct ApiBook {
    #[serde(default)]
    bids: Vec<ApiLevel>,
    #[serde(default)]
    asks: Vec<ApiLevel>,
}

#[derive(Debug, Deserialize)]
struct ApiLevel {
    price: String,
    size: String,
}

impl ClobClient {
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

    pub async fn fetch_orderbook(&self, token_id: &str) -> Result<LocalOrderBook> {
        let url = format!("{}/book?token_id={}", self.base_url, token_id);
        let api_book: ApiBook = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("failed to request CLOB book for token {token_id}"))?
            .error_for_status()
            .context("CLOB book response was not successful")?
            .json()
            .await
            .context("failed to decode CLOB book")?;

        let now = chrono::Utc::now();
        let mut book = LocalOrderBook::default();
        for level in api_book.bids {
            if let (Ok(price), Ok(size)) = (
                level.price.parse::<Decimal>(),
                level.size.parse::<Decimal>(),
            ) {
                book.upsert_level(BookSide::Bid, price, size, now);
            }
        }
        for level in api_book.asks {
            if let (Ok(price), Ok(size)) = (
                level.price.parse::<Decimal>(),
                level.size.parse::<Decimal>(),
            ) {
                book.upsert_level(BookSide::Ask, price, size, now);
            }
        }
        Ok(book)
    }

    pub async fn fetch_fee_rate(&self) -> Result<Decimal> {
        // Polymarket fees are externally fetched in production. Keep a safe zero default for sim v1
        // if the endpoint shape changes, rather than blocking startup.
        let url = format!("{}/fees", self.base_url);
        let value = self.http.get(&url).send().await;
        match value {
            Ok(response) if response.status().is_success() => {
                let json: serde_json::Value = response.json().await.unwrap_or_default();
                let bps = json
                    .get("feeRateBps")
                    .or_else(|| json.get("fee_rate_bps"))
                    .and_then(decimal_from_json)
                    .map(|bps| bps / Decimal::from(10_000))
                    .unwrap_or(Decimal::ZERO);
                Ok(bps)
            }
            _ => Ok(Decimal::ZERO),
        }
    }
}

fn decimal_from_json(value: &serde_json::Value) -> Option<Decimal> {
    if let Some(raw) = value.as_str() {
        return raw.parse().ok();
    }
    value.as_f64().and_then(Decimal::from_f64)
}
