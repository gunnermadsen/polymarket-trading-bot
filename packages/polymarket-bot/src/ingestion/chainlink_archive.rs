use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use chainlink_data_streams_report::report::{decode_full_report, v3::ReportDataV3, Report};
use chrono::{NaiveDate, TimeZone, Utc};
use hmac::{Hmac, Mac};
use rust_decimal::Decimal;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{binance_archive::ArchiveCancellation, job::ChainlinkBtcusdArchiveTick};

pub const CHAINLINK_ARCHIVE_PROVIDER: &str = "chainlink_data_streams";
pub const DEFAULT_CHAINLINK_REST_URL: &str = "https://api.dataengine.chain.link";
pub const DEFAULT_CHAINLINK_BTCUSD_FEED_ID: &str =
    "0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8";

#[derive(Debug, Clone)]
pub struct ChainlinkCredentials {
    pub api_key: String,
    pub api_secret: String,
}

#[derive(Debug, Clone)]
pub struct ChainlinkArchiveConfig {
    pub rest_url: String,
    pub feed_id: String,
    pub page_limit: usize,
    pub credentials: Option<ChainlinkCredentials>,
}

#[derive(Debug)]
pub struct ChainlinkDay {
    pub records: Vec<ChainlinkBtcusdArchiveTick>,
    pub sha256: String,
    pub response_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct ReportsPage {
    reports: Vec<Report>,
}

impl ChainlinkArchiveConfig {
    pub fn validate(&self) -> Result<()> {
        if self.rest_url.trim().is_empty() {
            bail!("POLYMARKET_CHAINLINK_DATA_STREAMS_REST_URL must not be empty");
        }
        validate_feed_id(&self.feed_id)?;
        if !(1..=10_000).contains(&self.page_limit) {
            bail!("POLYMARKET_CHAINLINK_DATA_STREAMS_PAGE_LIMIT must be between 1 and 10000");
        }
        if let Some(credentials) = &self.credentials {
            if credentials.api_key.trim().is_empty() || credentials.api_secret.trim().is_empty() {
                bail!("Chainlink Data Streams credentials must both be non-empty");
            }
        }
        Ok(())
    }

    pub fn logical_key(&self, date: NaiveDate) -> String {
        format!("chainlink:{}:{}", self.feed_id.to_ascii_lowercase(), date)
    }

    pub fn source_uri(&self, date: NaiveDate) -> String {
        format!(
            "{}/api/v1/reports/page?feedID={}&startTimestamp={}",
            self.rest_url.trim_end_matches('/'),
            self.feed_id,
            date.and_hms_opt(0, 0, 0)
                .expect("UTC midnight is representable")
                .and_utc()
                .timestamp()
        )
    }

    pub async fn fetch_day(
        &self,
        client: &reqwest::Client,
        date: NaiveDate,
        cancellation: &ArchiveCancellation,
    ) -> Result<ChainlinkDay> {
        self.validate()?;
        let credentials = self
            .credentials
            .as_ref()
            .context("Chainlink Data Streams credentials are not configured for this worker")?;
        let day_start = date
            .and_hms_opt(0, 0, 0)
            .context("invalid Chainlink archive date")?
            .and_utc();
        let day_end = day_start + chrono::Duration::days(1);
        let mut next_timestamp = day_start.timestamp();
        let mut records = Vec::with_capacity(86_400);
        let mut response_bytes = 0u64;
        let mut digest = Sha256::new();

        while next_timestamp < day_end.timestamp() {
            if cancellation.is_cancelled() {
                bail!("archive operation was cancelled");
            }
            let path = format!(
                "/api/v1/reports/page?feedID={}&startTimestamp={}&limit={}",
                self.feed_id, next_timestamp, self.page_limit
            );
            let timestamp_ms = current_timestamp_millis()?;
            let signature = sign_request(credentials, "GET", &path, timestamp_ms)?;
            let response = client
                .get(format!("{}{}", self.rest_url.trim_end_matches('/'), path))
                .header("Authorization", credentials.api_key.trim())
                .header("X-Authorization-Timestamp", timestamp_ms.to_string())
                .header("X-Authorization-Signature-SHA256", signature)
                .send()
                .await
                .context("failed to request Chainlink Data Streams reports")?
                .error_for_status()
                .context("Chainlink Data Streams rejected reports page")?;
            let body = response
                .bytes()
                .await
                .context("failed to read Chainlink Data Streams reports page")?;
            response_bytes = response_bytes.saturating_add(
                u64::try_from(body.len()).context("Chainlink response size overflow")?,
            );
            let page: ReportsPage =
                serde_json::from_slice(&body).context("invalid Chainlink reports page JSON")?;
            if page.reports.is_empty() {
                break;
            }

            let mut page_advanced = false;
            for envelope in page.reports {
                let source_seconds = i64::try_from(envelope.observations_timestamp)
                    .context("Chainlink observation timestamp overflow")?;
                if source_seconds >= day_end.timestamp() {
                    return Ok(ChainlinkDay {
                        records,
                        sha256: format!("{:x}", digest.finalize()),
                        response_bytes,
                    });
                }
                if source_seconds < day_start.timestamp() || source_seconds < next_timestamp {
                    bail!("Chainlink reports page was not strictly ordered");
                }
                let record = decode_report(&envelope, &self.feed_id)?;
                digest.update(envelope.full_report.as_bytes());
                records.push(record);
                next_timestamp = source_seconds.saturating_add(1);
                page_advanced = true;
            }
            if !page_advanced {
                bail!("Chainlink reports pagination did not advance");
            }
        }

        Ok(ChainlinkDay {
            records,
            sha256: format!("{:x}", digest.finalize()),
            response_bytes,
        })
    }
}

pub fn sign_request(
    credentials: &ChainlinkCredentials,
    method: &str,
    full_path: &str,
    timestamp_ms: u64,
) -> Result<String> {
    type HmacSha256 = Hmac<Sha256>;
    let body_hash = format!("{:x}", Sha256::digest([]));
    let message = format!(
        "{} {} {} {} {}",
        method.to_ascii_uppercase(),
        full_path,
        body_hash,
        credentials.api_key.trim(),
        timestamp_ms
    );
    let mut mac = HmacSha256::new_from_slice(credentials.api_secret.trim().as_bytes())
        .context("invalid Chainlink HMAC secret")?;
    mac.update(message.as_bytes());
    Ok(hex::encode(mac.finalize().into_bytes()))
}

pub(crate) fn decode_report(
    envelope: &Report,
    expected_feed_id: &str,
) -> Result<ChainlinkBtcusdArchiveTick> {
    let bytes = hex::decode(envelope.full_report.trim_start_matches("0x"))
        .context("Chainlink fullReport was not valid hex")?;
    let (_, blob) = decode_full_report(&bytes).context("failed to decode Chainlink full report")?;
    let report = ReportDataV3::decode(&blob).context("failed to decode Chainlink v3 report")?;
    let feed_id = report.feed_id.to_string().to_ascii_lowercase();
    if feed_id != expected_feed_id.to_ascii_lowercase()
        || envelope.feed_id.to_string().to_ascii_lowercase() != feed_id
    {
        bail!("Chainlink report feed ID did not match configured BTC/USD feed");
    }
    if usize::try_from(report.observations_timestamp).ok() != Some(envelope.observations_timestamp)
        || usize::try_from(report.valid_from_timestamp).ok() != Some(envelope.valid_from_timestamp)
    {
        bail!("Chainlink report envelope timestamps did not match its signed payload");
    }
    let source_timestamp = Utc
        .timestamp_opt(i64::from(report.observations_timestamp), 0)
        .single()
        .context("invalid Chainlink observation timestamp")?;
    let valid_from_timestamp = Utc
        .timestamp_opt(i64::from(report.valid_from_timestamp), 0)
        .single()
        .context("invalid Chainlink valid-from timestamp")?;
    let price = scaled_decimal(&report.benchmark_price.to_string(), 18)?;
    let bid = scaled_decimal(&report.bid.to_string(), 18)?;
    let ask = scaled_decimal(&report.ask.to_string(), 18)?;
    if price <= Decimal::ZERO || bid > price || price > ask {
        bail!("Chainlink BTC/USD report had invalid bid/price/ask ordering");
    }
    Ok(ChainlinkBtcusdArchiveTick {
        feed_id,
        source_timestamp,
        valid_from_timestamp,
        price,
        bid,
        ask,
        report_sha256: format!("{:x}", Sha256::digest(&bytes)),
    })
}

fn scaled_decimal(unscaled: &str, scale: u32) -> Result<Decimal> {
    let value = unscaled
        .parse::<i128>()
        .with_context(|| format!("Chainlink price {unscaled} exceeded decimal capacity"))?;
    Ok(Decimal::from_i128_with_scale(value, scale))
}

fn validate_feed_id(feed_id: &str) -> Result<()> {
    if feed_id.len() != 66
        || !feed_id.starts_with("0x")
        || !feed_id[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("Chainlink feed ID must be a 32-byte 0x-prefixed hexadecimal value");
    }
    Ok(())
}

fn current_timestamp_millis() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time predates Unix epoch")?
        .as_millis()
        .try_into()
        .context("system timestamp overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_is_stable_and_uses_empty_body_hash() {
        let credentials = ChainlinkCredentials {
            api_key: "test-key".to_string(),
            api_secret: "test-secret".to_string(),
        };
        assert_eq!(
            sign_request(
                &credentials,
                "GET",
                "/api/v1/reports/page?feedID=0x123&startTimestamp=1&limit=10",
                1_716_211_845_123,
            )
            .unwrap(),
            "7a1c55e02d4b5d43bd45bfbdd6c95e519260289007ca9a1b4d785fb856102cfc"
        );
    }

    #[test]
    fn validates_feed_identity_and_decimal_scaling() {
        assert!(validate_feed_id(DEFAULT_CHAINLINK_BTCUSD_FEED_ID).is_ok());
        assert!(validate_feed_id("0x123").is_err());
        assert_eq!(
            scaled_decimal("67123456789000000000000", 18).unwrap(),
            Decimal::new(67_123_456_789, 6)
        );
    }
}
