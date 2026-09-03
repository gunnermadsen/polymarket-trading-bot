use anyhow::{bail, Context, Result};
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use rust_decimal::Decimal;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{binance_archive::ArchiveCancellation, job::BinanceBtcusdtOpenInterestRecord};

pub const BINANCE_OPEN_INTEREST_PROVIDER: &str = "binance_futures";
pub const DEFAULT_BINANCE_FUTURES_DATA_BASE_URL: &str = "https://fapi.binance.com";
pub const DEFAULT_BINANCE_OPEN_INTEREST_SYMBOL: &str = "BTCUSDT";
pub const BINANCE_OPEN_INTEREST_PERIOD_SECONDS: i32 = 300;

#[derive(Debug, Clone)]
pub struct BinanceOpenInterestConfig {
    pub base_url: String,
    pub symbol: String,
}

#[derive(Debug)]
pub struct BinanceOpenInterestDay {
    pub records: Vec<BinanceBtcusdtOpenInterestRecord>,
    pub sha256: String,
    pub response_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawOpenInterestRecord {
    symbol: String,
    sum_open_interest: String,
    sum_open_interest_value: String,
    #[serde(rename = "CMCCirculatingSupply")]
    cmc_circulating_supply: Option<String>,
    timestamp: i64,
}

impl BinanceOpenInterestConfig {
    pub fn validate(&self) -> Result<()> {
        if self.base_url.trim().is_empty() {
            bail!("POLYMARKET_BINANCE_FUTURES_DATA_BASE_URL must not be empty");
        }
        if self.symbol != DEFAULT_BINANCE_OPEN_INTEREST_SYMBOL {
            bail!("Binance open-interest symbol must be BTCUSDT");
        }
        Ok(())
    }

    pub fn logical_key(&self, date: NaiveDate) -> String {
        format!("binance-open-interest:{}:5m:{date}", self.symbol)
    }

    pub fn source_uri(&self, date: NaiveDate) -> String {
        let start_millis = date
            .and_hms_opt(0, 0, 0)
            .expect("UTC midnight is representable")
            .and_utc()
            .timestamp_millis();
        format!(
            "{}/futures/data/openInterestHist?symbol={}&period=5m&startTime={}&endTime={}&limit=500",
            self.base_url.trim_end_matches('/'),
            self.symbol,
            start_millis,
            start_millis + 86_399_999
        )
    }

    pub async fn fetch_day(
        &self,
        client: &reqwest::Client,
        date: NaiveDate,
        cancellation: &ArchiveCancellation,
    ) -> Result<BinanceOpenInterestDay> {
        self.validate()?;
        if cancellation.is_cancelled() {
            bail!("archive operation was cancelled");
        }
        let day_start = date
            .and_hms_opt(0, 0, 0)
            .context("invalid Binance open-interest date")?
            .and_utc();
        let response = client
            .get(format!(
                "{}/futures/data/openInterestHist",
                self.base_url.trim_end_matches('/')
            ))
            .query(&[
                ("symbol", self.symbol.as_str()),
                ("period", "5m"),
                ("startTime", &day_start.timestamp_millis().to_string()),
                (
                    "endTime",
                    &(day_start.timestamp_millis() + 86_399_999).to_string(),
                ),
                ("limit", "500"),
            ])
            .send()
            .await
            .context("failed to request Binance open-interest history")?
            .error_for_status()
            .context("Binance open-interest history request was rejected")?;
        let body = response
            .bytes()
            .await
            .context("failed to read Binance open-interest history")?;
        let response_bytes =
            u64::try_from(body.len()).context("open-interest response overflow")?;
        let payload: Vec<RawOpenInterestRecord> =
            serde_json::from_slice(&body).context("invalid Binance open-interest history JSON")?;
        let records = decode_open_interest_records(payload, &self.symbol, Some(day_start), false)?;
        Ok(BinanceOpenInterestDay {
            records,
            sha256: format!("{:x}", Sha256::digest(&body)),
            response_bytes,
        })
    }
}

fn decode_open_interest_records(
    payload: Vec<RawOpenInterestRecord>,
    expected_symbol: &str,
    utc_day_start: Option<DateTime<Utc>>,
    require_positive: bool,
) -> Result<Vec<BinanceBtcusdtOpenInterestRecord>> {
    let mut records = Vec::with_capacity(payload.len());
    for row in payload {
        let source_timestamp = Utc
            .timestamp_millis_opt(row.timestamp)
            .single()
            .context("invalid Binance open-interest timestamp")?;
        if row.timestamp.rem_euclid(300_000) != 0
            || utc_day_start.is_some_and(|day_start| {
                source_timestamp < day_start
                    || source_timestamp >= day_start + chrono::Duration::days(1)
            })
        {
            bail!("Binance open-interest row escaped its requested range or five-minute alignment");
        }
        let sum_open_interest = decimal(&row.sum_open_interest, "sumOpenInterest")?;
        let sum_open_interest_value =
            decimal(&row.sum_open_interest_value, "sumOpenInterestValue")?;
        let cmc_circulating_supply = row
            .cmc_circulating_supply
            .as_deref()
            .map(|value| decimal(value, "CMCCirculatingSupply"))
            .transpose()?;
        let non_positive = require_positive
            && (sum_open_interest <= Decimal::ZERO || sum_open_interest_value <= Decimal::ZERO);
        if row.symbol != expected_symbol
            || non_positive
            || sum_open_interest < Decimal::ZERO
            || sum_open_interest_value < Decimal::ZERO
            || cmc_circulating_supply.is_some_and(|value| value < Decimal::ZERO)
        {
            bail!("Binance open-interest row contained invalid values");
        }
        records.push(BinanceBtcusdtOpenInterestRecord {
            symbol: row.symbol,
            source_timestamp,
            period_seconds: BINANCE_OPEN_INTEREST_PERIOD_SECONDS,
            sum_open_interest,
            sum_open_interest_value,
            cmc_circulating_supply,
        });
    }
    records.sort_unstable_by_key(|record| record.source_timestamp);
    if records
        .windows(2)
        .any(|rows| rows[0].source_timestamp >= rows[1].source_timestamp)
    {
        bail!("Binance open-interest history contained duplicate timestamps");
    }
    Ok(records)
}

fn decimal(value: &str, field: &str) -> Result<Decimal> {
    value
        .parse::<Decimal>()
        .with_context(|| format!("Binance {field} exceeded decimal capacity"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_an_unrelated_contract() {
        let mut config = BinanceOpenInterestConfig {
            base_url: DEFAULT_BINANCE_FUTURES_DATA_BASE_URL.to_string(),
            symbol: DEFAULT_BINANCE_OPEN_INTEREST_SYMBOL.to_string(),
        };
        assert!(config.validate().is_ok());
        config.symbol = "ETHUSDT".to_string();
        assert!(config.validate().is_err());
    }
}
