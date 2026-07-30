use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Context, Result};
use chrono::{NaiveDate, TimeZone, Utc};
use futures_util::{stream, StreamExt, TryStreamExt};
use rust_decimal::Decimal;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sha3::{Digest as Sha3Digest, Keccak256};

use super::{binance_archive::ArchiveCancellation, job::PolygonChainlinkBtcusdOracleRound};

pub const POLYGON_CHAINLINK_ORACLE_PROVIDER: &str = "chainlink_polygon_data_feed";
pub const DEFAULT_POLYGON_RPC_URL: &str = "https://polygon.drpc.org";
pub const DEFAULT_POLYGON_CHAINLINK_BTCUSD_PROXY: &str =
    "0xc907e116054ad103354f2d350fd2514433d57f6f";
pub const POLYGON_CHAIN_ID: i64 = 137;

const MAX_SUPPORTED_DECIMALS: u32 = 18;
const BLOCK_TIME_BOUNDARY_PADDING_SECONDS: i64 = 300;
const BLOCK_FETCH_CONCURRENCY: usize = 4;

#[derive(Debug, Clone)]
pub struct PolygonChainlinkOracleConfig {
    pub rpc_url: String,
    pub feed_proxy_address: String,
    pub maximum_block_range: u64,
}

#[derive(Debug)]
pub struct PolygonChainlinkOracleDay {
    pub records: Vec<PolygonChainlinkBtcusdOracleRound>,
    pub sha256: String,
    pub response_bytes: u64,
    pub start_block: u64,
    pub end_block: u64,
    pub phase_count: u16,
}

#[derive(Debug, Deserialize)]
struct RpcEnvelope {
    result: Option<Value>,
    error: Option<RpcError>,
}

#[derive(Debug, Deserialize)]
struct RpcError {
    code: i64,
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RpcBlock {
    number: Option<String>,
    timestamp: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RpcLog {
    address: String,
    topics: Vec<String>,
    data: String,
    block_number: String,
    block_hash: String,
    transaction_hash: String,
    log_index: String,
    #[serde(default)]
    removed: bool,
}

impl PolygonChainlinkOracleConfig {
    pub fn validate(&self) -> Result<()> {
        if self.rpc_url.trim().is_empty() {
            bail!("POLYMARKET_POLYGON_RPC_URL must not be empty");
        }
        validate_address(&self.feed_proxy_address)?;
        if !(100..=20_000).contains(&self.maximum_block_range) {
            bail!("POLYMARKET_POLYGON_RPC_MAX_BLOCK_RANGE must be between 100 and 20000");
        }
        Ok(())
    }

    pub fn logical_key(&self, date: NaiveDate) -> String {
        format!(
            "eip155:{POLYGON_CHAIN_ID}:{}:{date}",
            self.feed_proxy_address.to_ascii_lowercase()
        )
    }

    pub fn source_uri(&self, date: NaiveDate) -> String {
        format!(
            "eip155:{POLYGON_CHAIN_ID}/{}/answer-updated/{date}",
            self.feed_proxy_address.to_ascii_lowercase()
        )
    }

    pub async fn fetch_day(
        &self,
        client: &reqwest::Client,
        date: NaiveDate,
        cancellation: &ArchiveCancellation,
    ) -> Result<PolygonChainlinkOracleDay> {
        self.validate()?;
        let day_start = date
            .and_hms_opt(0, 0, 0)
            .context("invalid Polygon Chainlink archive date")?
            .and_utc();
        let day_end = day_start + chrono::Duration::days(1);
        let mut response_bytes = 0u64;

        let (decimals_value, bytes) = self
            .eth_call(client, &abi_calldata("decimals()", &[]), cancellation)
            .await?;
        response_bytes = response_bytes.saturating_add(bytes);
        let decimals = u32::try_from(parse_abi_u64(&decimals_value)?)
            .context("Polygon Chainlink decimals overflow")?;
        if decimals > MAX_SUPPORTED_DECIMALS {
            bail!("Polygon Chainlink feed decimals {decimals} exceed supported precision");
        }

        let (phase_value, bytes) = self
            .eth_call(client, &abi_calldata("phaseId()", &[]), cancellation)
            .await?;
        response_bytes = response_bytes.saturating_add(bytes);
        let phase_count = u16::try_from(parse_abi_u64(&phase_value)?)
            .context("Polygon Chainlink phase ID overflow")?;
        if phase_count == 0 {
            bail!("Polygon Chainlink feed reported no aggregator phases");
        }

        let mut aggregators = Vec::with_capacity(usize::from(phase_count));
        for phase_id in 1..=phase_count {
            ensure_not_cancelled(cancellation)?;
            let argument = encode_u16_word(phase_id);
            let (value, bytes) = self
                .eth_call(
                    client,
                    &abi_calldata("phaseAggregators(uint16)", &[argument]),
                    cancellation,
                )
                .await?;
            response_bytes = response_bytes.saturating_add(bytes);
            let address = parse_abi_address(&value)?;
            if address != "0x0000000000000000000000000000000000000000" {
                aggregators.push((phase_id, address));
            }
        }
        if aggregators.is_empty() {
            bail!("Polygon Chainlink feed had no usable aggregator addresses");
        }

        let latest_block = self.block_number(client, cancellation).await?;
        response_bytes = response_bytes.saturating_add(latest_block.1);
        let start_target = day_start.timestamp() - BLOCK_TIME_BOUNDARY_PADDING_SECONDS;
        let end_target = day_end.timestamp() + BLOCK_TIME_BOUNDARY_PADDING_SECONDS;
        let start = self
            .first_block_at_or_after(client, latest_block.0, start_target, cancellation)
            .await?;
        response_bytes = response_bytes.saturating_add(start.1);
        let end = self
            .first_block_at_or_after(client, latest_block.0, end_target, cancellation)
            .await?;
        response_bytes = response_bytes.saturating_add(end.1);
        let start_block = start.0;
        let end_block = end.0;

        let topic = event_topic("AnswerUpdated(int256,uint256,uint256)");
        let mut decoded = Vec::new();
        for (phase_id, aggregator_address) in aggregators {
            let mut from_block = start_block;
            while from_block <= end_block {
                ensure_not_cancelled(cancellation)?;
                let to_block = from_block
                    .saturating_add(self.maximum_block_range.saturating_sub(1))
                    .min(end_block);
                let (logs, bytes) = self
                    .logs(
                        client,
                        &aggregator_address,
                        &topic,
                        from_block,
                        to_block,
                        cancellation,
                    )
                    .await?;
                response_bytes = response_bytes.saturating_add(bytes);
                for log in logs {
                    if let Some(record) = decode_answer_updated(
                        log,
                        &self.feed_proxy_address,
                        &aggregator_address,
                        phase_id,
                        decimals,
                        day_start.timestamp(),
                        day_end.timestamp(),
                    )? {
                        decoded.push(record);
                    }
                }
                if to_block == u64::MAX {
                    break;
                }
                from_block = to_block + 1;
            }
        }

        decoded.sort_by_key(|record| {
            (
                record.source_timestamp,
                record.block_number,
                record.log_index,
            )
        });
        let mut identities = BTreeSet::new();
        for record in &decoded {
            if !identities.insert((record.transaction_hash.clone(), record.log_index)) {
                bail!("Polygon Chainlink RPC returned a duplicate event identity");
            }
        }

        let block_numbers = decoded
            .iter()
            .map(|record| {
                u64::try_from(record.block_number).context("Polygon block number was negative")
            })
            .collect::<Result<BTreeSet<_>>>()?;
        let config = self.clone();
        let client = client.clone();
        let cancellation = cancellation.clone();
        let block_results = stream::iter(block_numbers.into_iter().map(|number| {
            let config = config.clone();
            let client = client.clone();
            let cancellation = cancellation.clone();
            async move {
                let result = config
                    .block_by_number(&client, number, &cancellation)
                    .await?;
                Ok::<_, anyhow::Error>((number, result.0, result.1))
            }
        }))
        .buffer_unordered(BLOCK_FETCH_CONCURRENCY)
        .try_collect::<Vec<_>>()
        .await?;
        let mut block_timestamps = BTreeMap::new();
        for (number, timestamp, bytes) in block_results {
            response_bytes = response_bytes.saturating_add(bytes);
            block_timestamps.insert(number, timestamp);
        }
        for record in &mut decoded {
            let block_number =
                u64::try_from(record.block_number).context("Polygon block number was negative")?;
            record.block_timestamp = *block_timestamps
                .get(&block_number)
                .context("missing Polygon block timestamp for oracle update")?;
        }

        let mut digest = Sha256::new();
        for record in &decoded {
            digest.update(
                format!(
                    "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n",
                    record.feed_proxy_address,
                    record.aggregator_address,
                    record.phase_id,
                    record.aggregator_round_id,
                    record.source_timestamp.to_rfc3339(),
                    record.block_timestamp.to_rfc3339(),
                    record.answer_raw,
                    record.price,
                    record.decimals,
                    record.block_number,
                    record.block_hash,
                    record.transaction_hash,
                    record.log_index,
                )
                .as_bytes(),
            );
        }

        Ok(PolygonChainlinkOracleDay {
            records: decoded,
            sha256: format!("{:x}", digest.finalize()),
            response_bytes,
            start_block,
            end_block,
            phase_count,
        })
    }

    async fn block_number(
        &self,
        client: &reqwest::Client,
        cancellation: &ArchiveCancellation,
    ) -> Result<(u64, u64)> {
        let (value, bytes) = self
            .rpc(client, "eth_blockNumber", json!([]), cancellation)
            .await?;
        let encoded = value
            .as_str()
            .context("Polygon eth_blockNumber returned a non-string result")?;
        Ok((parse_quantity_u64(encoded)?, bytes))
    }

    async fn first_block_at_or_after(
        &self,
        client: &reqwest::Client,
        latest: u64,
        target_timestamp: i64,
        cancellation: &ArchiveCancellation,
    ) -> Result<(u64, u64)> {
        let mut low = 0u64;
        let mut high = latest;
        let mut response_bytes = 0u64;
        while low < high {
            ensure_not_cancelled(cancellation)?;
            let middle = low + (high - low) / 2;
            let (timestamp, bytes) = self.block_by_number(client, middle, cancellation).await?;
            response_bytes = response_bytes.saturating_add(bytes);
            if timestamp.timestamp() < target_timestamp {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        Ok((low, response_bytes))
    }

    async fn block_by_number(
        &self,
        client: &reqwest::Client,
        block_number: u64,
        cancellation: &ArchiveCancellation,
    ) -> Result<(chrono::DateTime<Utc>, u64)> {
        let (value, bytes) = self
            .rpc(
                client,
                "eth_getBlockByNumber",
                json!([format_quantity(block_number), false]),
                cancellation,
            )
            .await?;
        let block: RpcBlock =
            serde_json::from_value(value).context("invalid Polygon block response")?;
        let returned_number = block
            .number
            .as_deref()
            .context("Polygon RPC returned a pending block for a historical query")
            .and_then(parse_quantity_u64)?;
        if returned_number != block_number {
            bail!("Polygon RPC returned block {returned_number} when {block_number} was requested");
        }
        let timestamp = parse_quantity_u64(&block.timestamp)?;
        let timestamp = i64::try_from(timestamp).context("Polygon block timestamp overflow")?;
        let timestamp = Utc
            .timestamp_opt(timestamp, 0)
            .single()
            .context("invalid Polygon block timestamp")?;
        Ok((timestamp, bytes))
    }

    async fn eth_call(
        &self,
        client: &reqwest::Client,
        data: &str,
        cancellation: &ArchiveCancellation,
    ) -> Result<(String, u64)> {
        let (value, bytes) = self
            .rpc(
                client,
                "eth_call",
                json!([
                    {
                        "to": self.feed_proxy_address.to_ascii_lowercase(),
                        "data": data,
                    },
                    "latest"
                ]),
                cancellation,
            )
            .await?;
        let value = value
            .as_str()
            .context("Polygon eth_call returned a non-string result")?
            .to_string();
        Ok((value, bytes))
    }

    async fn logs(
        &self,
        client: &reqwest::Client,
        address: &str,
        topic: &str,
        from_block: u64,
        to_block: u64,
        cancellation: &ArchiveCancellation,
    ) -> Result<(Vec<RpcLog>, u64)> {
        let (value, bytes) = self
            .rpc(
                client,
                "eth_getLogs",
                json!([{
                    "address": address,
                    "fromBlock": format_quantity(from_block),
                    "toBlock": format_quantity(to_block),
                    "topics": [topic],
                }]),
                cancellation,
            )
            .await?;
        let logs = serde_json::from_value(value).context("invalid Polygon log response")?;
        Ok((logs, bytes))
    }

    async fn rpc(
        &self,
        client: &reqwest::Client,
        method: &str,
        params: Value,
        cancellation: &ArchiveCancellation,
    ) -> Result<(Value, u64)> {
        ensure_not_cancelled(cancellation)?;
        let response = client
            .post(self.rpc_url.trim())
            .timeout(std::time::Duration::from_secs(30))
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params,
            }))
            .send()
            .await
            .with_context(|| format!("failed to call Polygon RPC method {method}"))?
            .error_for_status()
            .with_context(|| format!("Polygon RPC rejected method {method}"))?;
        let body = response
            .bytes()
            .await
            .with_context(|| format!("failed to read Polygon RPC method {method} response"))?;
        let response_bytes =
            u64::try_from(body.len()).context("Polygon RPC response size overflow")?;
        let envelope: RpcEnvelope =
            serde_json::from_slice(&body).context("invalid Polygon JSON-RPC response")?;
        if let Some(error) = envelope.error {
            bail!(
                "Polygon JSON-RPC method {method} failed with {}: {}",
                error.code,
                error.message
            );
        }
        let result = envelope
            .result
            .with_context(|| format!("Polygon JSON-RPC method {method} omitted its result"))?;
        Ok((result, response_bytes))
    }
}

fn decode_answer_updated(
    log: RpcLog,
    feed_proxy_address: &str,
    expected_aggregator_address: &str,
    phase_id: u16,
    decimals: u32,
    day_start: i64,
    day_end: i64,
) -> Result<Option<PolygonChainlinkBtcusdOracleRound>> {
    if log.removed {
        return Ok(None);
    }
    if log.address.to_ascii_lowercase() != expected_aggregator_address.to_ascii_lowercase() {
        bail!("Polygon Chainlink log address did not match the requested aggregator");
    }
    if log.topics.len() != 3
        || log.topics[0].to_ascii_lowercase()
            != event_topic("AnswerUpdated(int256,uint256,uint256)")
    {
        bail!("Polygon Chainlink AnswerUpdated log had an unexpected topic layout");
    }
    let answer = parse_positive_i128_word(&log.topics[1])?;
    let aggregator_round_id = i64::try_from(parse_abi_u64(&log.topics[2])?)
        .context("Polygon Chainlink aggregator round ID overflow")?;
    let source_seconds =
        i64::try_from(parse_first_abi_u64(&log.data)?).context("oracle timestamp overflow")?;
    if source_seconds < day_start || source_seconds >= day_end {
        return Ok(None);
    }
    let source_timestamp = Utc
        .timestamp_opt(source_seconds, 0)
        .single()
        .context("invalid Polygon Chainlink source timestamp")?;
    let block_number = i64::try_from(parse_quantity_u64(&log.block_number)?)
        .context("Polygon block number overflow")?;
    let log_index =
        i32::try_from(parse_quantity_u64(&log.log_index)?).context("Polygon log index overflow")?;
    validate_hash(&log.block_hash)?;
    validate_hash(&log.transaction_hash)?;
    let answer_raw = Decimal::from_i128_with_scale(answer, 0);
    let price = Decimal::from_i128_with_scale(answer, decimals);
    if price <= Decimal::ZERO {
        bail!("Polygon Chainlink BTC/USD update had a non-positive answer");
    }
    Ok(Some(PolygonChainlinkBtcusdOracleRound {
        chain_id: POLYGON_CHAIN_ID,
        feed_proxy_address: feed_proxy_address.to_ascii_lowercase(),
        aggregator_address: expected_aggregator_address.to_ascii_lowercase(),
        phase_id: i32::from(phase_id),
        aggregator_round_id,
        source_timestamp,
        block_timestamp: source_timestamp,
        answer_raw,
        price,
        decimals: i32::try_from(decimals).context("oracle decimals overflow")?,
        block_number,
        block_hash: log.block_hash.to_ascii_lowercase(),
        transaction_hash: log.transaction_hash.to_ascii_lowercase(),
        log_index,
    }))
}

fn abi_calldata(signature: &str, arguments: &[String]) -> String {
    let digest = Keccak256::digest(signature.as_bytes());
    let mut encoded = format!("0x{}", hex::encode(&digest[..4]));
    for argument in arguments {
        encoded.push_str(argument);
    }
    encoded
}

fn event_topic(signature: &str) -> String {
    format!("0x{}", hex::encode(Keccak256::digest(signature.as_bytes())))
}

fn encode_u16_word(value: u16) -> String {
    format!("{value:064x}")
}

fn parse_abi_address(value: &str) -> Result<String> {
    let word = normalized_word(value)?;
    if !word[..24].bytes().all(|byte| byte == b'0') {
        bail!("ABI address had non-zero high bytes");
    }
    let address = format!("0x{}", &word[24..]);
    validate_address(&address)?;
    Ok(address)
}

fn parse_first_abi_u64(value: &str) -> Result<u64> {
    let encoded = value.strip_prefix("0x").unwrap_or(value);
    if encoded.len() < 64 {
        bail!("ABI data did not contain a complete word");
    }
    parse_abi_u64(&encoded[..64])
}

fn parse_abi_u64(value: &str) -> Result<u64> {
    let word = normalized_word(value)?;
    if !word[..48].bytes().all(|byte| byte == b'0') {
        bail!("ABI unsigned integer exceeded 64 bits");
    }
    u64::from_str_radix(&word[48..], 16).context("invalid ABI unsigned integer")
}

fn parse_positive_i128_word(value: &str) -> Result<i128> {
    let word = normalized_word(value)?;
    if !word[..32].bytes().all(|byte| byte == b'0') {
        bail!("oracle answer was negative or exceeded 128 bits");
    }
    let raw = u128::from_str_radix(&word[32..], 16).context("invalid oracle answer")?;
    i128::try_from(raw).context("oracle answer exceeded signed 128-bit capacity")
}

fn normalized_word(value: &str) -> Result<String> {
    let word = value.strip_prefix("0x").unwrap_or(value);
    if word.len() != 64 || !word.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("ABI value was not one 32-byte hexadecimal word");
    }
    Ok(word.to_ascii_lowercase())
}

fn parse_quantity_u64(value: &str) -> Result<u64> {
    let encoded = value
        .strip_prefix("0x")
        .context("JSON-RPC quantity lacked its 0x prefix")?;
    if encoded.is_empty() || !encoded.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("invalid JSON-RPC quantity");
    }
    u64::from_str_radix(encoded, 16).context("JSON-RPC quantity exceeded 64 bits")
}

fn format_quantity(value: u64) -> String {
    format!("0x{value:x}")
}

fn validate_address(value: &str) -> Result<()> {
    if value.len() != 42
        || !value.starts_with("0x")
        || !value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("EVM address must be a 20-byte 0x-prefixed hexadecimal value");
    }
    Ok(())
}

fn validate_hash(value: &str) -> Result<()> {
    if value.len() != 66
        || !value.starts_with("0x")
        || !value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("EVM hash must be a 32-byte 0x-prefixed hexadecimal value");
    }
    Ok(())
}

fn ensure_not_cancelled(cancellation: &ArchiveCancellation) -> Result<()> {
    if cancellation.is_cancelled() {
        bail!("archive operation was cancelled");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_canonical_chainlink_selectors_and_topic() {
        assert_eq!(abi_calldata("decimals()", &[]), "0x313ce567");
        assert_eq!(abi_calldata("phaseId()", &[]), "0x58303b10");
        assert_eq!(
            event_topic("AnswerUpdated(int256,uint256,uint256)"),
            "0x0559884fd3a460db3073b7fc896cc77986f16e378210ded43186175bf646fc5f"
        );
    }

    #[test]
    fn decodes_answer_updated_without_losing_precision() {
        let source_seconds = Utc
            .with_ymd_and_hms(2026, 3, 21, 12, 0, 1)
            .unwrap()
            .timestamp();
        let log = RpcLog {
            address: "0x1111111111111111111111111111111111111111".to_string(),
            topics: vec![
                event_topic("AnswerUpdated(int256,uint256,uint256)"),
                format!("0x{:064x}", 8_412_345_678_901i128),
                format!("0x{:064x}", 42u64),
            ],
            data: format!("0x{:064x}", source_seconds),
            block_number: "0x64".to_string(),
            block_hash: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                .to_string(),
            transaction_hash: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                .to_string(),
            log_index: "0x2".to_string(),
            removed: false,
        };
        let record = decode_answer_updated(
            log,
            DEFAULT_POLYGON_CHAINLINK_BTCUSD_PROXY,
            "0x1111111111111111111111111111111111111111",
            3,
            8,
            Utc.with_ymd_and_hms(2026, 3, 21, 0, 0, 0)
                .unwrap()
                .timestamp(),
            Utc.with_ymd_and_hms(2026, 3, 22, 0, 0, 0)
                .unwrap()
                .timestamp(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(record.aggregator_round_id, 42);
        assert_eq!(record.answer_raw, Decimal::new(8_412_345_678_901, 0));
        assert_eq!(record.price, Decimal::new(8_412_345_678_901, 8));
        assert_eq!(record.block_number, 100);
        assert_eq!(record.log_index, 2);
    }

    #[test]
    fn rejects_out_of_range_and_malformed_oracle_events() {
        assert!(parse_positive_i128_word(
            "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        )
        .is_err());
        assert!(parse_abi_address(
            "0x0000000000000000000000001111111111111111111111111111111111111111"
        )
        .is_ok());
        assert!(parse_abi_u64(
            "0x0000000000000001000000000000000000000000000000000000000000000000"
        )
        .is_err());
    }
}
