use anyhow::{bail, Context, Result};
use chrono::{NaiveDate, TimeZone, Utc};
use rust_decimal::Decimal;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{binance_archive::ArchiveCancellation, job::ChainlinkBtcusdOneMinuteCandle};

pub const CHAINLINK_CANDLESTICK_PROVIDER: &str = "chainlink_candlestick";
pub const DEFAULT_CHAINLINK_CANDLESTICK_BASE_URL: &str = "https://priceapi.dataengine.chain.link";
pub const DEFAULT_CHAINLINK_CANDLESTICK_SYMBOL: &str = "BTCUSD";
const CHAINLINK_PRICE_SCALE: i128 = 1_000_000_000_000_000_000;

#[derive(Debug, Clone)]
pub struct ChainlinkCandlestickCredentials {
    pub login: String,
    pub api_key: String,
}

#[derive(Debug, Clone)]
pub struct ChainlinkCandlestickConfig {
    pub base_url: String,
    pub symbol: String,
    pub credentials: Option<ChainlinkCandlestickCredentials>,
}

#[derive(Debug)]
pub struct ChainlinkCandlestickDay {
    pub records: Vec<ChainlinkBtcusdOneMinuteCandle>,
    pub sha256: String,
    pub response_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct AuthorizationResponse {
    s: String,
    d: AuthorizationData,
}

#[derive(Debug, Deserialize)]
struct AuthorizationData {
    access_token: String,
}

#[derive(Debug, Deserialize)]
struct CandleResponse {
    s: String,
    candles: Vec<Vec<Value>>,
}

impl ChainlinkCandlestickConfig {
    pub fn validate(&self) -> Result<()> {
        if self.base_url.trim().is_empty() {
            bail!("POLYMARKET_CHAINLINK_CANDLESTICK_BASE_URL must not be empty");
        }
        if self.symbol != DEFAULT_CHAINLINK_CANDLESTICK_SYMBOL {
            bail!("Chainlink candlestick symbol must be BTCUSD");
        }
        if let Some(credentials) = &self.credentials {
            if credentials.login.trim().is_empty() || credentials.api_key.trim().is_empty() {
                bail!("Chainlink Candlestick credentials must both be non-empty");
            }
        }
        Ok(())
    }

    pub fn logical_key(&self, date: NaiveDate) -> String {
        format!("chainlink-candlestick:{}:1m:{date}", self.symbol)
    }

    pub fn source_uri(&self, date: NaiveDate) -> String {
        let start = date
            .and_hms_opt(0, 0, 0)
            .expect("UTC midnight is representable")
            .and_utc()
            .timestamp();
        format!(
            "{}/api/v1/history/rows?symbol={}&resolution=1m&from={}&to={}",
            self.base_url.trim_end_matches('/'),
            self.symbol,
            start,
            start + 86_399
        )
    }

    pub async fn fetch_day(
        &self,
        client: &reqwest::Client,
        date: NaiveDate,
        cancellation: &ArchiveCancellation,
    ) -> Result<ChainlinkCandlestickDay> {
        self.validate()?;
        if cancellation.is_cancelled() {
            bail!("archive operation was cancelled");
        }
        let credentials = self
            .credentials
            .as_ref()
            .context("Chainlink Candlestick credentials are not configured for this worker")?;
        let authorization = client
            .post(format!(
                "{}/api/v1/authorize",
                self.base_url.trim_end_matches('/')
            ))
            .form(&[
                ("login", credentials.login.trim()),
                ("password", credentials.api_key.trim()),
            ])
            .send()
            .await
            .context("failed to authorize Chainlink Candlestick request")?
            .error_for_status()
            .context("Chainlink Candlestick authorization was rejected")?
            .json::<AuthorizationResponse>()
            .await
            .context("invalid Chainlink Candlestick authorization response")?;
        if authorization.s != "ok" || authorization.d.access_token.trim().is_empty() {
            bail!("invalid Chainlink Candlestick authorization response");
        }
        if cancellation.is_cancelled() {
            bail!("archive operation was cancelled");
        }

        let day_start = date
            .and_hms_opt(0, 0, 0)
            .context("invalid Chainlink candlestick date")?
            .and_utc();
        let response = client
            .get(format!(
                "{}/api/v1/history/rows",
                self.base_url.trim_end_matches('/')
            ))
            .bearer_auth(authorization.d.access_token)
            .query(&[
                ("symbol", self.symbol.as_str()),
                ("resolution", "1m"),
                ("from", &day_start.timestamp().to_string()),
                ("to", &(day_start.timestamp() + 86_399).to_string()),
            ])
            .send()
            .await
            .context("failed to request Chainlink Candlestick history")?
            .error_for_status()
            .context("Chainlink Candlestick history request was rejected")?;
        let body = response
            .bytes()
            .await
            .context("failed to read Chainlink Candlestick history")?;
        let response_bytes = u64::try_from(body.len()).context("candlestick response overflow")?;
        let payload: CandleResponse =
            serde_json::from_slice(&body).context("invalid Chainlink Candlestick history JSON")?;
        if payload.s != "ok" {
            bail!("Chainlink Candlestick history returned a non-ok status");
        }

        let mut records = Vec::with_capacity(payload.candles.len());
        for candle in payload.candles {
            if candle.len() != 6 {
                bail!("Chainlink Candlestick row did not contain six values");
            }
            let timestamp = json_i64(&candle[0], "timestamp")?;
            let open_timestamp = Utc
                .timestamp_opt(timestamp, 0)
                .single()
                .context("invalid Chainlink Candlestick timestamp")?;
            if open_timestamp < day_start
                || open_timestamp >= day_start + chrono::Duration::days(1)
                || timestamp.rem_euclid(60) != 0
            {
                bail!("Chainlink Candlestick row escaped its UTC day or minute alignment");
            }
            let open_price = scaled_price(&candle[1], "open")?;
            let high_price = scaled_price(&candle[2], "high")?;
            let low_price = scaled_price(&candle[3], "low")?;
            let close_price = scaled_price(&candle[4], "close")?;
            if decimal_value(&candle[5], "volume")? != Decimal::ZERO {
                bail!("Chainlink Candlestick unexpectedly returned nonzero unsupported volume");
            }
            if open_price <= Decimal::ZERO
                || high_price < open_price
                || high_price < close_price
                || high_price < low_price
                || low_price > open_price
                || low_price > close_price
            {
                bail!("Chainlink Candlestick row contained invalid OHLC prices");
            }
            records.push(ChainlinkBtcusdOneMinuteCandle {
                symbol: self.symbol.clone(),
                open_timestamp,
                close_timestamp: open_timestamp + chrono::Duration::minutes(1),
                open_price,
                high_price,
                low_price,
                close_price,
                volume: None,
                volume_supported: false,
            });
        }
        records.sort_unstable_by_key(|record| record.open_timestamp);
        if records
            .windows(2)
            .any(|rows| rows[0].open_timestamp >= rows[1].open_timestamp)
        {
            bail!("Chainlink Candlestick history contained duplicate timestamps");
        }
        Ok(ChainlinkCandlestickDay {
            records,
            sha256: format!("{:x}", Sha256::digest(&body)),
            response_bytes,
        })
    }
}

fn json_i64(value: &Value, field: &str) -> Result<i64> {
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
        .with_context(|| format!("Chainlink Candlestick {field} was not an integer"))
}

fn scaled_price(value: &Value, field: &str) -> Result<Decimal> {
    decimal_value(value, field)?
        .checked_div(Decimal::from_i128_with_scale(CHAINLINK_PRICE_SCALE, 0))
        .with_context(|| format!("Chainlink Candlestick {field} scale overflow"))
}

fn decimal_value(value: &Value, field: &str) -> Result<Decimal> {
    let encoded = match value {
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        _ => bail!("Chainlink Candlestick {field} was not numeric"),
    };
    encoded
        .parse::<Decimal>()
        .or_else(|_| Decimal::from_scientific(&encoded))
        .with_context(|| format!("Chainlink Candlestick {field} exceeded decimal capacity"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn scientific_prices_are_scaled_from_eighteen_decimals() {
        let value = serde_json::json!(6.123456789e22);
        assert_eq!(scaled_price(&value, "price").unwrap(), dec!(61234.56789));
    }

    #[test]
    fn config_requires_the_training_symbol() {
        let mut config = ChainlinkCandlestickConfig {
            base_url: DEFAULT_CHAINLINK_CANDLESTICK_BASE_URL.to_string(),
            symbol: DEFAULT_CHAINLINK_CANDLESTICK_SYMBOL.to_string(),
            credentials: None,
        };
        assert!(config.validate().is_ok());
        config.symbol = "ETHUSD".to_string();
        assert!(config.validate().is_err());
    }
}
