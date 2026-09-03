use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    env, fmt,
    time::Duration as StdDuration,
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::Deserialize;
use serde_json::{json, Value};
use sha3::{Digest, Keccak256};
use tokio::{
    sync::{watch, RwLock},
    time::{interval, MissedTickBehavior},
};
use tracing::warn;

use crate::ingestion::{
    binance_open_interest::{
        BinanceOpenInterestConfig, DEFAULT_BINANCE_FUTURES_DATA_BASE_URL,
        DEFAULT_BINANCE_OPEN_INTEREST_SYMBOL,
    },
    polygon_chainlink_oracle::{DEFAULT_POLYGON_CHAINLINK_BTCUSD_PROXY, DEFAULT_POLYGON_RPC_URL},
};

use super::directional_features::BTC_DIRECTIONAL_ORACLE_MAX_AGE_SECONDS;
use super::types::{RealtimeState, ReferencePriceSource, ReferencePriceTick};

const ORACLE_HISTORY_SECONDS: i64 = 600;
const RTDS_MID_CAPACITY: usize = 4_096;
const ORACLE_CAPACITY: usize = 512;
const OPEN_INTEREST_CAPACITY: usize = 32;
const OPEN_INTEREST_LATEST_LIMIT: usize = 24;
const MAX_SOURCE_ERROR_BYTES: usize = 256;
const HTTP_TIMEOUT: StdDuration = StdDuration::from_secs(10);
const POLYGON_RPC_TIMEOUT: StdDuration = StdDuration::from_secs(3);
const POLYGON_ORACLE_FUTURE_TOLERANCE: chrono::Duration = chrono::Duration::seconds(2);
const POLYGON_CHAIN_ID: u128 = 137;

#[derive(Clone)]
pub struct DirectionalExternalRuntimeConfig {
    pub enabled: bool,
    pub polygon_rpc_url: String,
    pub polygon_rpc_fallback_url: Option<String>,
    pub polygon_proxy_address: String,
    pub binance_futures_base_url: String,
    pub oracle_poll_interval: StdDuration,
    pub open_interest_poll_interval: StdDuration,
}

impl fmt::Debug for DirectionalExternalRuntimeConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DirectionalExternalRuntimeConfig")
            .field("enabled", &self.enabled)
            .field("polygon_rpc_url", &self.polygon_rpc_url)
            .field(
                "polygon_rpc_fallback_url",
                &self
                    .polygon_rpc_fallback_url
                    .as_ref()
                    .map(|_| "[configured]"),
            )
            .field("polygon_proxy_address", &self.polygon_proxy_address)
            .field("binance_futures_base_url", &self.binance_futures_base_url)
            .field("oracle_poll_interval", &self.oracle_poll_interval)
            .field(
                "open_interest_poll_interval",
                &self.open_interest_poll_interval,
            )
            .finish()
    }
}

impl Default for DirectionalExternalRuntimeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            polygon_rpc_url: DEFAULT_POLYGON_RPC_URL.to_string(),
            polygon_rpc_fallback_url: None,
            polygon_proxy_address: DEFAULT_POLYGON_CHAINLINK_BTCUSD_PROXY.to_string(),
            binance_futures_base_url: DEFAULT_BINANCE_FUTURES_DATA_BASE_URL.to_string(),
            oracle_poll_interval: StdDuration::from_secs(2),
            open_interest_poll_interval: StdDuration::from_secs(60),
        }
    }
}

impl DirectionalExternalRuntimeConfig {
    pub fn from_env() -> Result<Self> {
        let enabled = env_bool("POLYMARKET_BTC_DIRECTIONAL_EXTERNAL_ENABLED", false)?;
        let defaults = Self::default();
        let config = Self {
            enabled,
            polygon_rpc_url: env_string("POLYMARKET_POLYGON_RPC_URL", &defaults.polygon_rpc_url),
            polygon_rpc_fallback_url: nonempty_env("POLYMARKET_POLYGON_RPC_FALLBACK_URL"),
            polygon_proxy_address: env_string(
                "POLYMARKET_POLYGON_CHAINLINK_BTCUSD_PROXY",
                &defaults.polygon_proxy_address,
            )
            .to_ascii_lowercase(),
            binance_futures_base_url: env_string(
                "POLYMARKET_BINANCE_FUTURES_DATA_BASE_URL",
                &defaults.binance_futures_base_url,
            ),
            oracle_poll_interval: env_duration_millis(
                "POLYMARKET_BTC_POLYGON_ORACLE_POLL_MS",
                defaults.oracle_poll_interval,
                500,
                10_000,
            )?,
            open_interest_poll_interval: env_duration_millis(
                "POLYMARKET_BTC_BINANCE_OI_POLL_MS",
                defaults.open_interest_poll_interval,
                5_000,
                300_000,
            )?,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        if !self.polygon_rpc_url.starts_with("https://") {
            bail!("directional external Polygon endpoint must use HTTPS");
        }
        if self
            .polygon_rpc_fallback_url
            .as_deref()
            .is_some_and(|url| !url.starts_with("https://"))
        {
            bail!("directional external Polygon fallback endpoint must use HTTPS");
        }
        if self.polygon_rpc_fallback_url.as_deref() == Some(self.polygon_rpc_url.as_str()) {
            bail!("directional external Polygon fallback endpoint must differ from the primary");
        }
        if !self.binance_futures_base_url.starts_with("https://") {
            bail!("directional external Binance endpoint must use HTTPS");
        }
        validate_hex(&self.polygon_proxy_address, 20, "Polygon proxy address")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChainlinkMidPoint {
    pub source_timestamp: DateTime<Utc>,
    pub available_at: DateTime<Utc>,
    pub price: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChainlinkRefPricePoint {
    pub source_timestamp: DateTime<Utc>,
    pub valid_from_timestamp: DateTime<Utc>,
    pub available_at: DateTime<Utc>,
    pub price: Decimal,
    pub bid: Decimal,
    pub ask: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolygonOraclePoint {
    pub phase_id: u16,
    pub round_id: u64,
    pub source_timestamp: DateTime<Utc>,
    pub block_timestamp: DateTime<Utc>,
    pub available_at: DateTime<Utc>,
    pub price: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinanceOpenInterestPoint {
    pub source_timestamp: DateTime<Utc>,
    pub available_at: DateTime<Utc>,
    pub sum_open_interest: Decimal,
    pub sum_open_interest_value: Decimal,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DirectionalExternalSourceStatus {
    pub last_success_at: Option<DateTime<Utc>>,
    pub last_error_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DirectionalExternalState {
    pub chainlink_mid: VecDeque<ChainlinkMidPoint>,
    pub refprice: VecDeque<ChainlinkRefPricePoint>,
    pub oracle: VecDeque<PolygonOraclePoint>,
    pub open_interest: VecDeque<BinanceOpenInterestPoint>,
    pub source_status: BTreeMap<&'static str, DirectionalExternalSourceStatus>,
}

impl DirectionalExternalState {
    pub fn chainlink_candle_complete_minutes(&self, at: DateTime<Utc>) -> usize {
        const REQUIRED_MINUTES: i64 = 61;
        let latest_close_seconds = at.timestamp().div_euclid(60) * 60;
        let Some(latest_close) = DateTime::from_timestamp(latest_close_seconds, 0) else {
            return 0;
        };
        let earliest_open = latest_close - chrono::Duration::minutes(REQUIRED_MINUTES);
        self.chainlink_mid
            .iter()
            .filter(|point| {
                point.available_at <= at
                    && point.source_timestamp >= earliest_open
                    && point.source_timestamp < latest_close
            })
            .map(|point| point.source_timestamp.timestamp().div_euclid(60))
            .collect::<HashSet<_>>()
            .len()
            .min(REQUIRED_MINUTES as usize)
    }

    pub fn polygon_oracle_age_seconds(&self, at: DateTime<Utc>) -> Option<i64> {
        self.oracle
            .iter()
            .filter(|point| point.available_at <= at && point.block_timestamp <= at)
            .max_by_key(|point| (point.block_timestamp, point.phase_id, point.round_id))
            .map(|point| (at - point.block_timestamp).num_seconds().max(0))
    }

    pub fn polygon_oracle_ready(&self, at: DateTime<Utc>) -> bool {
        self.polygon_oracle_age_seconds(at)
            .is_some_and(|age| age <= BTC_DIRECTIONAL_ORACLE_MAX_AGE_SECONDS)
    }

    pub fn observe_rtds_chainlink(&mut self, tick: &ReferencePriceTick) -> Result<()> {
        if tick.source != ReferencePriceSource::RtdsChainlink {
            return Ok(());
        }
        let price = tick.price;
        if price <= Decimal::ZERO {
            bail!("RTDS Chainlink price was invalid for directional features");
        }
        insert_bounded_first_seen(
            &mut self.chainlink_mid,
            ChainlinkMidPoint {
                source_timestamp: tick.source_timestamp,
                available_at: tick.received_at,
                price,
            },
            RTDS_MID_CAPACITY,
            |point| point.source_timestamp,
        );
        self.record_success("chainlink_mid", tick.received_at);
        Ok(())
    }

    pub(crate) fn merge_oracle(&mut self, points: Vec<PolygonOraclePoint>, at: DateTime<Utc>) {
        for point in points {
            let identity = (point.phase_id, point.round_id);
            if self
                .oracle
                .iter()
                .any(|row| (row.phase_id, row.round_id) == identity)
            {
                continue;
            }
            self.oracle.push_back(point);
        }
        self.oracle
            .make_contiguous()
            .sort_by_key(|point| (point.block_timestamp, point.phase_id, point.round_id));
        while self.oracle.len() > ORACLE_CAPACITY {
            self.oracle.pop_front();
        }
        self.record_success("oracle", at);
    }

    pub(crate) fn merge_open_interest(
        &mut self,
        points: Vec<BinanceOpenInterestPoint>,
        at: DateTime<Utc>,
    ) {
        for point in points {
            if self
                .open_interest
                .iter()
                .any(|existing| existing.source_timestamp == point.source_timestamp)
            {
                continue;
            }
            self.open_interest.push_back(point);
        }
        self.open_interest
            .make_contiguous()
            .sort_by_key(|point| point.source_timestamp);
        while self.open_interest.len() > OPEN_INTEREST_CAPACITY {
            self.open_interest.pop_front();
        }
        self.record_success("open_interest", at);
    }

    fn record_success(&mut self, source: &'static str, at: DateTime<Utc>) {
        let status = self.source_status.entry(source).or_default();
        status.last_success_at = Some(at);
        status.last_error = None;
    }

    fn record_error(&mut self, source: &'static str, at: DateTime<Utc>, error: &anyhow::Error) {
        let status = self.source_status.entry(source).or_default();
        status.last_error_at = Some(at);
        let message = error.to_string();
        status.last_error = Some(message.chars().take(MAX_SOURCE_ERROR_BYTES).collect());
    }
}

pub(crate) async fn run_directional_external_supervisor(
    config: DirectionalExternalRuntimeConfig,
    state: std::sync::Arc<RwLock<RealtimeState>>,
    mut shutdown: watch::Receiver<bool>,
) {
    if !config.enabled {
        let _ = shutdown.changed().await;
        return;
    }
    let client = match Client::builder().timeout(HTTP_TIMEOUT).build() {
        Ok(client) => client,
        Err(error) => {
            state.write().await.directional_external.record_error(
                "configuration",
                Utc::now(),
                &error.into(),
            );
            let _ = shutdown.changed().await;
            return;
        }
    };
    let mut oracle = PolygonOraclePoller::new(&config);
    let open_interest = BinanceOpenInterestConfig {
        base_url: config.binance_futures_base_url.clone(),
        symbol: DEFAULT_BINANCE_OPEN_INTEREST_SYMBOL.to_string(),
    };
    let mut oracle_tick = interval(config.oracle_poll_interval);
    let mut open_interest_tick = interval(config.open_interest_poll_interval);
    for ticker in [&mut oracle_tick, &mut open_interest_tick] {
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    }

    loop {
        tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    break;
                }
            }
            _ = oracle_tick.tick() => {
                let at = Utc::now();
                match oracle.poll(&client, at).await {
                    Ok(points) => state.write().await.directional_external.merge_oracle(points, at),
                    Err(error) => {
                        warn!(error = %error, "directional Polygon oracle refresh failed closed");
                        state.write().await.directional_external.record_error("oracle", at, &error);
                    }
                }
            }
            _ = open_interest_tick.tick() => {
                let at = Utc::now();
                match open_interest.fetch_latest(&client, OPEN_INTEREST_LATEST_LIMIT).await {
                    Ok(records) => {
                        let points = records.into_iter().filter_map(|record| {
                            let sum_open_interest = record.sum_open_interest;
                            let sum_open_interest_value = record.sum_open_interest_value;
                            (sum_open_interest > Decimal::ZERO
                                && sum_open_interest_value > Decimal::ZERO)
                                .then_some(BinanceOpenInterestPoint {
                                    source_timestamp: record.source_timestamp,
                                    available_at: at,
                                    sum_open_interest,
                                    sum_open_interest_value,
                                })
                        }).collect();
                        state.write().await.directional_external.merge_open_interest(points, at);
                    }
                    Err(error) => {
                        warn!(error = %error, "directional Binance OI refresh failed closed");
                        state.write().await.directional_external.record_error("open_interest", at, &error);
                    }
                }
            }
        }
    }
}

struct PolygonOraclePoller {
    rpc_urls: Vec<String>,
    proxy_address: String,
    decimals: Option<u32>,
    latest_round: Option<u128>,
    validated_endpoints: HashSet<usize>,
}

impl PolygonOraclePoller {
    fn new(config: &DirectionalExternalRuntimeConfig) -> Self {
        let mut rpc_urls = vec![config.polygon_rpc_url.clone()];
        if let Some(fallback) = &config.polygon_rpc_fallback_url {
            rpc_urls.push(fallback.clone());
        }
        Self {
            rpc_urls,
            proxy_address: config.polygon_proxy_address.clone(),
            decimals: None,
            latest_round: None,
            validated_endpoints: HashSet::new(),
        }
    }

    async fn poll(
        &mut self,
        client: &Client,
        at: DateTime<Utc>,
    ) -> Result<Vec<PolygonOraclePoint>> {
        let mut failures = Vec::new();
        for endpoint_index in 0..self.rpc_urls.len() {
            match self.poll_endpoint(client, at, endpoint_index).await {
                Ok(points) => {
                    if endpoint_index > 0 {
                        tracing::info!(
                            endpoint_role = "fallback",
                            "directional Polygon oracle fallback succeeded"
                        );
                    }
                    return Ok(points);
                }
                Err(error) => failures.push(format!(
                    "{}: {error:#}",
                    if endpoint_index == 0 {
                        "primary"
                    } else {
                        "fallback"
                    }
                )),
            }
        }
        bail!("Polygon oracle endpoints failed: {}", failures.join("; "))
    }

    async fn poll_endpoint(
        &mut self,
        client: &Client,
        at: DateTime<Utc>,
        endpoint_index: usize,
    ) -> Result<Vec<PolygonOraclePoint>> {
        let rpc_url = self.rpc_urls[endpoint_index].clone();
        if !self.validated_endpoints.contains(&endpoint_index) {
            let chain_id = self
                .rpc_request(client, &rpc_url, "eth_chainId", json!([]))
                .await?;
            let chain_id = chain_id
                .as_str()
                .context("Polygon eth_chainId omitted a hexadecimal result")?;
            let chain_id = u128::from_str_radix(chain_id.trim_start_matches("0x"), 16)
                .context("Polygon eth_chainId was invalid")?;
            if chain_id != POLYGON_CHAIN_ID {
                bail!("Polygon RPC returned unexpected chain ID {chain_id}");
            }
            self.validated_endpoints.insert(endpoint_index);
        }
        let decimals = match self.decimals {
            Some(decimals) => decimals,
            None => {
                let value = self
                    .eth_call(client, &rpc_url, &abi_calldata("decimals()", None))
                    .await?;
                let decimals = u32::try_from(parse_u128_word(&value)?)
                    .context("Polygon oracle decimals overflow")?;
                if decimals > 18 {
                    bail!("Polygon oracle decimals exceed supported precision");
                }
                self.decimals = Some(decimals);
                decimals
            }
        };
        let latest = self.fetch_round(client, &rpc_url, None, decimals).await?;
        if self.latest_round == Some(composite_round_id(&latest)) {
            return Ok(Vec::new());
        }

        let mut points = vec![latest.clone()];
        if self.latest_round.is_none() {
            let mut composite = composite_round_id(&latest);
            while points.len() < ORACLE_CAPACITY {
                let phase = composite >> 64;
                let round = composite & u128::from(u64::MAX);
                if round <= 1 {
                    break;
                }
                composite = (phase << 64) | (round - 1);
                let point = match self
                    .fetch_round(client, &rpc_url, Some(composite), decimals)
                    .await
                {
                    Ok(point) => point,
                    Err(_) => break,
                };
                let old_enough = point.source_timestamp
                    <= at - chrono::Duration::seconds(ORACLE_HISTORY_SECONDS);
                points.push(point);
                if old_enough {
                    break;
                }
            }
        }
        self.latest_round = Some(composite_round_id(&latest));
        Ok(points)
    }

    async fn fetch_round(
        &self,
        client: &Client,
        rpc_url: &str,
        round_id: Option<u128>,
        decimals: u32,
    ) -> Result<PolygonOraclePoint> {
        let (signature, argument) = match round_id {
            Some(round_id) => ("getRoundData(uint80)", Some(format!("{round_id:064x}"))),
            None => ("latestRoundData()", None),
        };
        let value = self
            .eth_call(
                client,
                rpc_url,
                &abi_calldata(signature, argument.as_deref()),
            )
            .await?;
        // A round can be published while the RPC request is in flight. Use the
        // completed response time for causality validation instead of the poll
        // start time so a newly published, already observable round is not
        // rejected as future data.
        let received_at = Utc::now();
        decode_polygon_oracle_round(&value, decimals, received_at)
    }

    async fn eth_call(&self, client: &Client, rpc_url: &str, data: &str) -> Result<String> {
        self.rpc_request(
            client,
            rpc_url,
            "eth_call",
            json!([{"to": self.proxy_address, "data": data}, "latest"]),
        )
        .await?
        .as_str()
        .map(str::to_string)
        .context("Polygon oracle RPC omitted a hexadecimal result")
    }

    async fn rpc_request(
        &self,
        client: &Client,
        rpc_url: &str,
        method: &str,
        params: Value,
    ) -> Result<Value> {
        let response = client
            .post(rpc_url.trim())
            .timeout(POLYGON_RPC_TIMEOUT)
            .json(&json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}))
            .send()
            .await
            .context("failed to call Polygon oracle RPC")?
            .error_for_status()
            .context("Polygon oracle RPC request was rejected")?;
        let envelope = response
            .json::<RpcEnvelope>()
            .await
            .context("invalid Polygon oracle JSON-RPC response")?;
        if let Some(error) = envelope.error {
            bail!(
                "Polygon oracle RPC failed with {}: {}",
                error.code,
                error.message
            );
        }
        envelope
            .result
            .context("Polygon oracle RPC omitted a result")
    }
}

fn decode_polygon_oracle_round(
    value: &str,
    decimals: u32,
    received_at: DateTime<Utc>,
) -> Result<PolygonOraclePoint> {
    let words = abi_words(value, 5)?;
    let round_id = parse_u128_word(words[0])?;
    let answer = parse_positive_i128_word(words[1])?;
    let updated_at = parse_u128_word(words[3])?;
    let answered_in_round = parse_u128_word(words[4])?;
    if updated_at == 0 || answered_in_round < round_id {
        bail!("Polygon oracle returned an incomplete round");
    }
    let phase_id = u16::try_from(round_id >> 64).context("Polygon phase ID overflow")?;
    let round =
        u64::try_from(round_id & u128::from(u64::MAX)).context("Polygon round ID overflow")?;
    let source_timestamp = Utc
        .timestamp_opt(
            i64::try_from(updated_at).context("oracle timestamp overflow")?,
            0,
        )
        .single()
        .context("invalid Polygon oracle timestamp")?;
    let future_skew = source_timestamp.signed_duration_since(received_at);
    if future_skew > POLYGON_ORACLE_FUTURE_TOLERANCE {
        bail!(
            "Polygon oracle round timestamp exceeded the bounded future tolerance: skew_ms={}",
            future_skew.num_milliseconds(),
        );
    }
    let price = Decimal::from_i128_with_scale(answer, decimals);
    if price <= Decimal::ZERO {
        bail!("Polygon oracle returned an invalid price");
    }
    Ok(PolygonOraclePoint {
        phase_id,
        round_id: round,
        source_timestamp,
        // The Polygon provider's AnswerUpdated logs report the same timestamp for the
        // source update and its containing block. This equality was verified across the
        // full July archive and is preserved by latestRoundData.updatedAt.
        block_timestamp: source_timestamp,
        available_at: received_at.max(source_timestamp),
        price,
    })
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

fn composite_round_id(point: &PolygonOraclePoint) -> u128 {
    (u128::from(point.phase_id) << 64) | u128::from(point.round_id)
}

fn abi_calldata(signature: &str, argument: Option<&str>) -> String {
    let digest = Keccak256::digest(signature.as_bytes());
    let mut encoded = format!("0x{}", hex::encode(&digest[..4]));
    if let Some(argument) = argument {
        encoded.push_str(argument);
    }
    encoded
}

fn abi_words(value: &str, count: usize) -> Result<Vec<&str>> {
    let encoded = value.strip_prefix("0x").unwrap_or(value);
    if encoded.len() != count * 64 || !encoded.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("ABI response had an unexpected word count");
    }
    Ok((0..count)
        .map(|index| &encoded[index * 64..(index + 1) * 64])
        .collect())
}

fn parse_u128_word(value: &str) -> Result<u128> {
    let word = value.strip_prefix("0x").unwrap_or(value);
    if word.len() != 64
        || !word[..32].bytes().all(|byte| byte == b'0')
        || !word[32..].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("ABI unsigned value exceeded 128 bits");
    }
    u128::from_str_radix(&word[32..], 16).context("invalid ABI unsigned value")
}

fn parse_positive_i128_word(value: &str) -> Result<i128> {
    let raw = parse_u128_word(value)?;
    i128::try_from(raw).context("ABI answer was negative or exceeded i128")
}

fn insert_bounded_first_seen<T, F>(rows: &mut VecDeque<T>, value: T, capacity: usize, timestamp: F)
where
    F: Fn(&T) -> DateTime<Utc>,
{
    let key = timestamp(&value);
    if rows.back().is_none_or(|last| timestamp(last) < key) {
        rows.push_back(value);
    } else {
        let insert_at = {
            let contiguous = rows.make_contiguous();
            match contiguous.binary_search_by_key(&key, &timestamp) {
                Ok(_) => return,
                Err(index) => index,
            }
        };
        rows.insert(insert_at, value);
    }
    while rows.len() > capacity {
        rows.pop_front();
    }
}

fn env_bool(key: &str, default: bool) -> Result<bool> {
    match env::var(key) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => bail!("{key} must be a boolean"),
        },
        Err(_) => Ok(default),
    }
}

fn env_duration_millis(
    key: &str,
    default: StdDuration,
    minimum: u64,
    maximum: u64,
) -> Result<StdDuration> {
    let milliseconds = env::var(key)
        .ok()
        .map(|value| value.parse::<u64>())
        .transpose()
        .with_context(|| format!("{key} must be an integer number of milliseconds"))?
        .unwrap_or_else(|| u64::try_from(default.as_millis()).expect("default duration fits u64"));
    if !(minimum..=maximum).contains(&milliseconds) {
        bail!("{key} must be between {minimum} and {maximum} milliseconds");
    }
    Ok(StdDuration::from_millis(milliseconds))
}

fn env_string(key: &str, default: &str) -> String {
    nonempty_env(key).unwrap_or_else(|| default.to_string())
}

fn nonempty_env(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn validate_hex(value: &str, bytes: usize, role: &str) -> Result<()> {
    if value.len() != 2 + bytes * 2
        || !value.starts_with("0x")
        || !value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("{role} must be a {bytes}-byte hexadecimal value");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enabled_runtime_validates_without_chainlink_data_streams_credentials() {
        let mut config = DirectionalExternalRuntimeConfig::default();
        config.enabled = true;

        config.validate().unwrap();
    }

    #[test]
    fn directional_health_reports_complete_candle_window_and_fresh_oracle() {
        let at = DateTime::from_timestamp(1_800_000_000, 0).unwrap();
        let mut state = DirectionalExternalState::default();
        let earliest = at - chrono::Duration::minutes(61);
        for minute in 0..61 {
            let source_timestamp = earliest + chrono::Duration::minutes(minute);
            state.chainlink_mid.push_back(ChainlinkMidPoint {
                source_timestamp,
                available_at: source_timestamp,
                price: Decimal::new(60_000, 0),
            });
        }
        state.oracle.push_back(PolygonOraclePoint {
            phase_id: 1,
            round_id: 42,
            source_timestamp: at - chrono::Duration::seconds(5),
            block_timestamp: at - chrono::Duration::seconds(5),
            available_at: at - chrono::Duration::seconds(4),
            price: Decimal::new(60_000, 0),
        });

        assert_eq!(state.chainlink_candle_complete_minutes(at), 61);
        assert!(state.polygon_oracle_ready(at));
        assert_eq!(state.polygon_oracle_age_seconds(at), Some(5));
    }

    #[test]
    fn directional_health_fails_closed_for_incomplete_candles_and_stale_oracle() {
        let at = DateTime::from_timestamp(1_800_000_000, 0).unwrap();
        let mut state = DirectionalExternalState::default();
        state.chainlink_mid.push_back(ChainlinkMidPoint {
            source_timestamp: at - chrono::Duration::minutes(1),
            available_at: at - chrono::Duration::minutes(1),
            price: Decimal::new(60_000, 0),
        });
        state.oracle.push_back(PolygonOraclePoint {
            phase_id: 1,
            round_id: 42,
            source_timestamp: at
                - chrono::Duration::seconds(BTC_DIRECTIONAL_ORACLE_MAX_AGE_SECONDS + 1),
            block_timestamp: at
                - chrono::Duration::seconds(BTC_DIRECTIONAL_ORACLE_MAX_AGE_SECONDS + 1),
            available_at: at
                - chrono::Duration::seconds(BTC_DIRECTIONAL_ORACLE_MAX_AGE_SECONDS + 1),
            price: Decimal::new(60_000, 0),
        });

        assert_eq!(state.chainlink_candle_complete_minutes(at), 1);
        assert!(!state.polygon_oracle_ready(at));
    }

    #[test]
    fn abi_round_data_decoding_preserves_phase_and_round_identity() {
        let composite = (u128::from(3_u16) << 64) | 42;
        let encoded = format!(
            "0x{composite:064x}{:064x}{:064x}{:064x}{composite:064x}",
            6_123_456_789_000_i128, 1_700_000_000_u64, 1_700_000_001_u64,
        );
        let words = abi_words(&encoded, 5).unwrap();
        assert_eq!(parse_u128_word(words[0]).unwrap(), composite);
        assert_eq!(
            parse_positive_i128_word(words[1]).unwrap(),
            6_123_456_789_000
        );
        assert_eq!(parse_u128_word(words[3]).unwrap(), 1_700_000_001);
    }

    #[test]
    fn oracle_round_published_during_request_uses_response_receipt_time() {
        let composite = (u128::from(3_u16) << 64) | 42;
        let source_timestamp = 1_700_000_001_u64;
        let encoded = format!(
            "0x{composite:064x}{:064x}{:064x}{source_timestamp:064x}{composite:064x}",
            6_123_456_789_000_i128, 1_700_000_000_u64,
        );
        let received_at = DateTime::from_timestamp(1_700_000_002, 0).unwrap();

        let point = decode_polygon_oracle_round(&encoded, 8, received_at).unwrap();

        assert_eq!(point.source_timestamp.timestamp(), 1_700_000_001);
        assert_eq!(point.available_at, received_at);
    }

    #[test]
    fn oracle_round_with_bounded_clock_skew_preserves_causal_availability() {
        let composite = (u128::from(3_u16) << 64) | 42;
        let source_timestamp = 1_700_000_003_u64;
        let encoded = format!(
            "0x{composite:064x}{:064x}{:064x}{source_timestamp:064x}{composite:064x}",
            6_123_456_789_000_i128, 1_700_000_000_u64,
        );
        let received_at = DateTime::from_timestamp(1_700_000_002, 0).unwrap();

        let point = decode_polygon_oracle_round(&encoded, 8, received_at).unwrap();
        assert_eq!(point.source_timestamp.timestamp(), 1_700_000_003);
        assert_eq!(point.available_at, point.source_timestamp);
    }

    #[test]
    fn oracle_round_beyond_bounded_clock_skew_remains_rejected() {
        let composite = (u128::from(3_u16) << 64) | 42;
        let source_timestamp = 1_700_000_005_u64;
        let encoded = format!(
            "0x{composite:064x}{:064x}{:064x}{source_timestamp:064x}{composite:064x}",
            6_123_456_789_000_i128, 1_700_000_000_u64,
        );
        let received_at = DateTime::from_timestamp(1_700_000_002, 0).unwrap();

        assert!(decode_polygon_oracle_round(&encoded, 8, received_at).is_err());
    }

    #[test]
    fn bounded_source_state_preserves_first_seen_duplicate_timestamp() {
        let at = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let mut state = DirectionalExternalState::default();
        let tick = ReferencePriceTick {
            tick_id: uuid::Uuid::new_v4(),
            dedup_key: "a".to_string(),
            source: ReferencePriceSource::RtdsChainlink,
            symbol: "BTCUSD".to_string(),
            price: rust_decimal::Decimal::new(60_000, 0),
            source_timestamp: at,
            envelope_timestamp: None,
            received_at: at,
            connection_id: uuid::Uuid::new_v4(),
            ingest_sequence: 1,
            source_event_id: None,
            raw_payload: json!({}),
        };
        state.observe_rtds_chainlink(&tick).unwrap();
        let mut duplicate = tick;
        duplicate.price = Decimal::new(61_000, 0);
        duplicate.received_at += chrono::Duration::seconds(1);
        state.observe_rtds_chainlink(&duplicate).unwrap();
        assert_eq!(state.chainlink_mid.len(), 1);
        assert_eq!(state.chainlink_mid[0].price, Decimal::new(60_000, 0));
        assert_eq!(state.chainlink_mid[0].available_at, at);
    }

    #[test]
    fn bounded_source_insert_keeps_rare_out_of_order_rows_sorted() {
        let at = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let mut rows = VecDeque::new();
        for offset in [2_i64, 0, 1] {
            insert_bounded_first_seen(
                &mut rows,
                ChainlinkMidPoint {
                    source_timestamp: at + chrono::Duration::seconds(offset),
                    available_at: at + chrono::Duration::seconds(offset),
                    price: Decimal::new(60_000 + offset, 0),
                },
                3,
                |point| point.source_timestamp,
            );
        }
        assert_eq!(
            rows.iter()
                .map(|point| (point.source_timestamp - at).num_seconds())
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn open_interest_refresh_keeps_first_causal_observation() {
        let at = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let source_timestamp = at - chrono::Duration::minutes(5);
        let mut state = DirectionalExternalState::default();
        state.merge_open_interest(
            vec![BinanceOpenInterestPoint {
                source_timestamp,
                available_at: at,
                sum_open_interest: Decimal::new(100_000, 0),
                sum_open_interest_value: Decimal::new(1_000_000, 0),
            }],
            at,
        );
        state.merge_open_interest(
            vec![BinanceOpenInterestPoint {
                source_timestamp,
                available_at: at + chrono::Duration::minutes(1),
                sum_open_interest: Decimal::new(200_000, 0),
                sum_open_interest_value: Decimal::new(2_000_000, 0),
            }],
            at + chrono::Duration::minutes(1),
        );

        assert_eq!(state.open_interest.len(), 1);
        assert_eq!(state.open_interest[0].available_at, at);
        assert_eq!(
            state.open_interest[0].sum_open_interest,
            Decimal::new(100_000, 0)
        );
    }
}
