use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use reqwest::{Client, Url};
use rust_decimal::Decimal;
use serde_json::{Map, Value};

use super::types::{
    AnalyticsRow, CandleRow, FeeScheduleRow, FundingRateRow, InstrumentRow, KrakenBackfillJob,
    KrakenDataset, NormalizedRows,
};

pub const DEFAULT_FUTURES_BASE_URL: &str = "https://futures.kraken.com";

#[derive(Debug, Clone)]
pub struct KrakenArchiveClient {
    client: Client,
    base_url: String,
}

#[derive(Debug, Clone)]
pub struct FetchedArchive {
    pub source_url: String,
    pub rows: NormalizedRows,
}

impl KrakenArchiveClient {
    pub fn new(client: Client, base_url: String) -> Result<Self> {
        let parsed = Url::parse(&base_url).context("invalid Kraken Futures base URL")?;
        if parsed.scheme() != "https" && parsed.scheme() != "http" {
            bail!("Kraken Futures base URL must use HTTP or HTTPS");
        }
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    pub async fn fetch(&self, job: &KrakenBackfillJob) -> Result<FetchedArchive> {
        let dataset = job.dataset()?;
        match dataset {
            KrakenDataset::Instruments => self.fetch_instruments(job).await,
            KrakenDataset::FeeSchedules => self.fetch_fee_schedules().await,
            KrakenDataset::TradeCandles
            | KrakenDataset::MarkCandles
            | KrakenDataset::SpotCandles => self.fetch_candles(job, dataset).await,
            KrakenDataset::FundingRates => self.fetch_funding_analytics(job).await,
            _ => self.fetch_analytics(job, dataset).await,
        }
    }

    async fn fetch_fee_schedules(&self) -> Result<FetchedArchive> {
        let url = format!("{}/derivatives/api/v3/feeschedules", self.base_url);
        let payload = self.get_json(&url, &[]).await?;
        let schedules = payload
            .get("feeSchedules")
            .and_then(Value::as_array)
            .context("Kraken fee schedule response omitted feeSchedules")?;
        let observed_at = payload
            .get("serverTime")
            .and_then(Value::as_str)
            .map(DateTime::parse_from_rfc3339)
            .transpose()
            .context("invalid Kraken fee schedule serverTime")?
            .map(|value| value.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);
        let rows = schedules
            .iter()
            .map(|schedule| parse_fee_schedule(schedule, observed_at))
            .collect::<Result<Vec<_>>>()?;
        Ok(FetchedArchive {
            source_url: url,
            rows: NormalizedRows::FeeSchedules(rows),
        })
    }

    async fn fetch_instruments(&self, job: &KrakenBackfillJob) -> Result<FetchedArchive> {
        let url = format!("{}/derivatives/api/v3/instruments", self.base_url);
        let payload = self.get_json(&url, &[]).await?;
        let instruments = payload
            .get("instruments")
            .and_then(Value::as_array)
            .context("Kraken instruments response omitted instruments")?;
        let observed_at = Utc::now();
        let rows = instruments
            .iter()
            .filter(|instrument| {
                instrument
                    .get("symbol")
                    .and_then(Value::as_str)
                    .is_some_and(|symbol| symbol.eq_ignore_ascii_case(&job.symbol))
            })
            .map(|instrument| parse_instrument(instrument, observed_at))
            .collect::<Result<Vec<_>>>()?;
        if rows.is_empty() {
            bail!("Kraken instrument {} was not returned", job.symbol);
        }
        Ok(FetchedArchive {
            source_url: url,
            rows: NormalizedRows::Instruments(rows),
        })
    }

    async fn fetch_candles(
        &self,
        job: &KrakenBackfillJob,
        dataset: KrakenDataset,
    ) -> Result<FetchedArchive> {
        let kind = dataset
            .candle_kind()
            .context("candle dataset had no kind")?;
        let resolution = resolution_name(job.interval_seconds)?;
        let url = format!(
            "{}/api/charts/v1/{kind}/{}/{resolution}",
            self.base_url, job.symbol
        );
        let from = job.range_start.timestamp().to_string();
        let to = (job.range_end.timestamp() - 1).to_string();
        let payload = self.get_json(&url, &[("from", &from), ("to", &to)]).await?;
        if payload
            .get("more_candles")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            bail!("Kraken candle job range exceeded one API page");
        }
        let candles = payload
            .get("candles")
            .and_then(Value::as_array)
            .context("Kraken candle response omitted candles")?;
        let rows = candles
            .iter()
            .map(parse_candle)
            .collect::<Result<Vec<_>>>()?;
        Ok(FetchedArchive {
            source_url: request_url(&url, &[("from", &from), ("to", &to)])?,
            rows: NormalizedRows::Candles(rows),
        })
    }

    async fn fetch_analytics(
        &self,
        job: &KrakenBackfillJob,
        dataset: KrakenDataset,
    ) -> Result<FetchedArchive> {
        let slug = dataset
            .analytics_slug()
            .context("analytics dataset had no endpoint slug")?;
        let url = format!(
            "{}/api/charts/v1/analytics/{}/{}",
            self.base_url, job.symbol, slug
        );
        let since = job.range_start.timestamp().to_string();
        let to = (job.range_end.timestamp() - 1).to_string();
        let interval = job.interval_seconds.to_string();
        let query = [
            ("since", since.as_str()),
            ("to", to.as_str()),
            ("interval", interval.as_str()),
        ];
        let payload = self.get_json(&url, &query).await?;
        let result = payload
            .get("result")
            .and_then(Value::as_object)
            .context("Kraken analytics response omitted result")?;
        if result.get("more").and_then(Value::as_bool).unwrap_or(false) {
            bail!("Kraken analytics job range exceeded one API page");
        }
        let timestamps = result
            .get("timestamp")
            .and_then(Value::as_array)
            .context("Kraken analytics result omitted timestamp")?;
        let data = result
            .get("data")
            .context("Kraken analytics result omitted data")?;
        let rows = timestamps
            .iter()
            .enumerate()
            .map(|(index, timestamp)| {
                let timestamp = timestamp
                    .as_i64()
                    .context("Kraken analytics timestamp was not an integer")?;
                let bucket_start = analytics_timestamp(timestamp)?;
                Ok(AnalyticsRow {
                    bucket_start,
                    values: sample_data(data, index, timestamps.len())?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(FetchedArchive {
            source_url: request_url(&url, &query)?,
            rows: NormalizedRows::Analytics(rows),
        })
    }

    async fn fetch_funding_analytics(&self, job: &KrakenBackfillJob) -> Result<FetchedArchive> {
        let url = format!(
            "{}/api/charts/v1/analytics/{}/funding",
            self.base_url, job.symbol
        );
        let since = job.range_start.timestamp().to_string();
        let to = (job.range_end.timestamp() - 1).to_string();
        let interval = job.interval_seconds.to_string();
        let query = [
            ("since", since.as_str()),
            ("to", to.as_str()),
            ("interval", interval.as_str()),
        ];
        let payload = self.get_json(&url, &query).await?;
        let result = payload
            .get("result")
            .and_then(Value::as_object)
            .context("Kraken funding analytics response omitted result")?;
        if result.get("more").and_then(Value::as_bool).unwrap_or(false) {
            bail!("Kraken funding job range exceeded one API page");
        }
        let timestamps = result
            .get("timestamp")
            .and_then(Value::as_array)
            .context("Kraken funding analytics omitted timestamps")?;
        let data = result
            .get("data")
            .and_then(Value::as_object)
            .context("Kraken funding analytics omitted data")?;
        let rates = data
            .get("rate")
            .and_then(Value::as_array)
            .context("Kraken funding analytics omitted rate")?;
        let relative_rates = data
            .get("relativeRate")
            .and_then(Value::as_array)
            .context("Kraken funding analytics omitted relativeRate")?;
        if rates.len() != timestamps.len() || relative_rates.len() != timestamps.len() {
            bail!("Kraken funding analytics arrays had inconsistent lengths");
        }
        let rows = timestamps
            .iter()
            .enumerate()
            .map(|(index, timestamp)| {
                let timestamp = timestamp
                    .as_i64()
                    .context("Kraken funding timestamp was not an integer")?;
                Ok(FundingRateRow {
                    funding_time: analytics_timestamp(timestamp)?,
                    funding_rate: analytics_close(&rates[index])?,
                    relative_funding_rate: analytics_close(&relative_rates[index])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(FetchedArchive {
            source_url: request_url(&url, &query)?,
            rows: NormalizedRows::FundingRates(rows),
        })
    }

    async fn get_json(&self, url: &str, query: &[(&str, &str)]) -> Result<Value> {
        let response = self
            .client
            .get(url)
            .query(query)
            .send()
            .await
            .with_context(|| format!("failed to request {url}"))?;
        let status = response.status();
        let body = response
            .bytes()
            .await
            .with_context(|| format!("failed to read response from {url}"))?;
        if !status.is_success() {
            let message = String::from_utf8_lossy(&body);
            bail!("Kraken request {url} returned {status}: {message}");
        }
        serde_json::from_slice(&body).with_context(|| format!("invalid JSON from {url}"))
    }
}

fn parse_instrument(value: &Value, observed_at: DateTime<Utc>) -> Result<InstrumentRow> {
    let symbol = required_string(value, "symbol")?.to_ascii_uppercase();
    let instrument_type = required_string(value, "type")?.to_string();
    let tradeable = value
        .get("tradeable")
        .and_then(Value::as_bool)
        .context("Kraken instrument tradeable was not a boolean")?;
    Ok(InstrumentRow {
        symbol,
        instrument_type,
        tradeable,
        tick_size: optional_decimal(value.get("tickSize"))?,
        contract_size: optional_decimal(value.get("contractSize"))?,
        base_currency: optional_string(value, "base"),
        quote_currency: optional_string(value, "quote"),
        pair: optional_string(value, "pair"),
        contract_value_trade_precision: value
            .get("contractValueTradePrecision")
            .and_then(Value::as_i64)
            .map(i32::try_from)
            .transpose()
            .context("Kraken contract precision exceeded i32")?,
        max_position_size: optional_decimal(value.get("maxPositionSize"))?,
        funding_rate_coefficient: optional_decimal(value.get("fundingRateCoefficient"))?,
        max_relative_funding_rate: optional_decimal(value.get("maxRelativeFundingRate"))?,
        fee_schedule_uid: value
            .get("feeScheduleUid")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        margin_levels: value
            .get("marginLevels")
            .cloned()
            .unwrap_or_else(|| Value::Array(Vec::new())),
        retail_margin_levels: value
            .get("retailMarginLevels")
            .cloned()
            .unwrap_or_else(|| Value::Array(Vec::new())),
        margin_schedules: value
            .get("marginSchedules")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new())),
        raw_payload: value.clone(),
        source_observed_at: observed_at,
    })
}

fn parse_fee_schedule(value: &Value, observed_at: DateTime<Utc>) -> Result<FeeScheduleRow> {
    Ok(FeeScheduleRow {
        fee_schedule_uid: required_string(value, "uid")?.to_string(),
        name: required_string(value, "name")?.to_string(),
        tiers: value
            .get("tiers")
            .cloned()
            .context("Kraken fee schedule omitted tiers")?,
        raw_payload: value.clone(),
        source_observed_at: observed_at,
    })
}

fn parse_candle(value: &Value) -> Result<CandleRow> {
    let time = value
        .get("time")
        .and_then(Value::as_i64)
        .context("Kraken candle time was not an integer")?;
    let seconds = time.div_euclid(1_000);
    Ok(CandleRow {
        bucket_start: DateTime::from_timestamp(seconds, 0)
            .context("Kraken candle time was outside UTC range")?,
        open: required_decimal(value, "open")?,
        high: required_decimal(value, "high")?,
        low: required_decimal(value, "low")?,
        close: required_decimal(value, "close")?,
        volume: required_decimal(value, "volume")?,
    })
}

fn analytics_close(value: &Value) -> Result<Decimal> {
    match value {
        Value::Array(values) => values
            .last()
            .context("Kraken analytics OHLC value was empty")
            .and_then(decimal),
        scalar => decimal(scalar),
    }
}

fn analytics_timestamp(timestamp: i64) -> Result<DateTime<Utc>> {
    if timestamp.unsigned_abs() >= 100_000_000_000 {
        DateTime::from_timestamp_millis(timestamp)
            .context("Kraken analytics millisecond timestamp was outside UTC range")
    } else {
        DateTime::from_timestamp(timestamp, 0)
            .context("Kraken analytics second timestamp was outside UTC range")
    }
}

fn decimal(value: &Value) -> Result<Decimal> {
    match value {
        Value::String(value) => value
            .parse()
            .with_context(|| format!("invalid Kraken decimal {value}")),
        Value::Number(value) => value
            .to_string()
            .parse()
            .with_context(|| format!("invalid Kraken decimal {value}")),
        other => bail!("expected Kraken numeric value, received {other}"),
    }
}

fn sample_data(value: &Value, index: usize, observation_count: usize) -> Result<Value> {
    match value {
        Value::Array(values) if values.len() == observation_count => values
            .get(index)
            .cloned()
            .context("Kraken analytics array was shorter than timestamps"),
        Value::Array(values) => values
            .iter()
            .map(|entry| sample_data(entry, index, observation_count))
            .collect::<Result<Vec<_>>>()
            .map(Value::Array),
        Value::Object(values) => {
            let sampled = values
                .iter()
                .map(|(key, entry)| {
                    Ok((key.clone(), sample_data(entry, index, observation_count)?))
                })
                .collect::<Result<Map<String, Value>>>()?;
            Ok(Value::Object(sampled))
        }
        scalar => Ok(scalar.clone()),
    }
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("Kraken field {field} was not a string"))
}

fn optional_string(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn required_decimal(value: &Value, field: &str) -> Result<Decimal> {
    optional_decimal(value.get(field))?.with_context(|| format!("Kraken field {field} was missing"))
}

fn optional_decimal(value: Option<&Value>) -> Result<Option<Decimal>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => decimal(value).map(Some),
    }
}

fn resolution_name(interval_seconds: i32) -> Result<&'static str> {
    match interval_seconds {
        60 => Ok("1m"),
        300 => Ok("5m"),
        900 => Ok("15m"),
        1_800 => Ok("30m"),
        3_600 => Ok("1h"),
        14_400 => Ok("4h"),
        43_200 => Ok("12h"),
        86_400 => Ok("1d"),
        604_800 => Ok("1w"),
        _ => bail!("unsupported Kraken candle interval {interval_seconds}"),
    }
}

fn request_url(base: &str, query: &[(&str, &str)]) -> Result<String> {
    let mut url = Url::parse(base).context("invalid Kraken request URL")?;
    url.query_pairs_mut().extend_pairs(query.iter().copied());
    Ok(url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_flat_and_nested_analytics_arrays() {
        let flat = serde_json::json!([["1", "2", "3", "4"], ["5", "6", "7", "8"]]);
        assert_eq!(
            sample_data(&flat, 1, 2).unwrap(),
            serde_json::json!(["5", "6", "7", "8"])
        );

        let nested = serde_json::json!({
            "bid": {"best_price": ["100", "101"]},
            "ask": {"best_price": ["102", "103"]}
        });
        assert_eq!(
            sample_data(&nested, 1, 2).unwrap(),
            serde_json::json!({
                "bid": {"best_price": "101"},
                "ask": {"best_price": "103"}
            })
        );
    }

    #[test]
    fn supported_interval_names_are_stable() {
        assert_eq!(resolution_name(900).unwrap(), "15m");
        assert!(resolution_name(42).is_err());
    }

    #[test]
    fn funding_analytics_uses_ohlc_close() {
        assert_eq!(
            analytics_close(&serde_json::json!(["1", "2", "3", "4"])).unwrap(),
            Decimal::from(4)
        );
    }

    #[test]
    fn analytics_timestamps_accept_endpoint_specific_units() {
        assert_eq!(
            analytics_timestamp(1_678_188_600).unwrap(),
            analytics_timestamp(1_678_188_600_000).unwrap()
        );
    }
}
