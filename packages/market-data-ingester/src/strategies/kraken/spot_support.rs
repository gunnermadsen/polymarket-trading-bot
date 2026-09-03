use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use arrow_array::{BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use chrono::{DateTime, Datelike, Utc};
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use reqwest::Client;
use rust_decimal::{prelude::ToPrimitive, Decimal};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::{task, time::sleep};

use crate::strategies::binance::archive_support::ArchiveCancellation;

pub const KRAKEN_SPOT_PROVIDER: &str = "kraken_spot";
pub const KRAKEN_SPOT_SYMBOL: &str = "XBTUSD";
pub const KRAKEN_SPOT_PAIR_KEY: &str = "XXBTZUSD";
pub const KRAKEN_SPOT_SCHEMA_VERSION: &str = "kraken-spot-trades-ohlcv-1s-v1";
const PAGE_SIZE: usize = 1_000;
const MAX_PAGES_PER_DAY: usize = 20_000;

#[derive(Debug, Clone)]
pub struct KrakenSpotTradeConfig {
    pub base_url: String,
    pub lake_root: PathBuf,
    pub request_delay: Duration,
}

impl KrakenSpotTradeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.base_url.trim().is_empty() {
            bail!("Kraken Spot base URL must not be empty");
        }
        if !self.lake_root.is_absolute() {
            bail!("Kraken Spot data lake root must be absolute");
        }
        if self.request_delay < Duration::from_millis(50) {
            bail!("Kraken Spot request delay must be at least 50ms");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KrakenSpotTrade {
    pub trade_id: i64,
    pub timestamp_ns: i64,
    pub price: Decimal,
    pub base_volume: Decimal,
    pub side: String,
    pub order_type: String,
}

#[derive(Debug, Clone)]
pub struct PublishedKrakenSpotDay {
    pub trade_path: String,
    pub candle_path: String,
    pub trade_sha256: String,
    pub candle_sha256: String,
    pub combined_sha256: String,
    pub trade_bytes: u64,
    pub candle_bytes: u64,
    pub trade_count: u64,
    pub candle_count: u64,
    pub minimum_timestamp: DateTime<Utc>,
    pub maximum_timestamp: DateTime<Utc>,
    pub pages: u64,
}

#[derive(Debug, Clone)]
struct OneSecondCandle {
    second_start_ns: i64,
    open: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    base_volume: Decimal,
    quote_volume: Decimal,
    trade_count: i64,
    first_trade_id: i64,
    last_trade_id: i64,
}

impl OneSecondCandle {
    fn new(trade: &KrakenSpotTrade) -> Self {
        Self {
            second_start_ns: trade.timestamp_ns.div_euclid(1_000_000_000) * 1_000_000_000,
            open: trade.price,
            high: trade.price,
            low: trade.price,
            close: trade.price,
            base_volume: trade.base_volume,
            quote_volume: trade.price * trade.base_volume,
            trade_count: 1,
            first_trade_id: trade.trade_id,
            last_trade_id: trade.trade_id,
        }
    }

    fn observe(&mut self, trade: &KrakenSpotTrade) {
        self.high = self.high.max(trade.price);
        self.low = self.low.min(trade.price);
        self.close = trade.price;
        self.base_volume += trade.base_volume;
        self.quote_volume += trade.price * trade.base_volume;
        self.trade_count += 1;
        self.last_trade_id = trade.trade_id;
    }
}

pub async fn fetch_and_publish_day(
    client: &Client,
    config: &KrakenSpotTradeConfig,
    range_start: DateTime<Utc>,
    range_end: DateTime<Utc>,
    cancellation: &ArchiveCancellation,
) -> Result<PublishedKrakenSpotDay> {
    config.validate()?;
    if range_end <= range_start || (range_end - range_start).num_seconds() > 86_400 {
        bail!("Kraken Spot shard must be positive and no longer than one day");
    }
    let (trades, pages) =
        fetch_trades(client, config, range_start, range_end, cancellation).await?;
    if trades.is_empty() {
        bail!("Kraken Spot returned no BTC/USD trades for the requested shard");
    }
    let root = config.lake_root.clone();
    task::spawn_blocking(move || publish_day(&root, range_start, range_end, trades, pages))
        .await
        .context("Kraken Spot Parquet publication task failed")?
}

async fn fetch_trades(
    client: &Client,
    config: &KrakenSpotTradeConfig,
    range_start: DateTime<Utc>,
    range_end: DateTime<Utc>,
    cancellation: &ArchiveCancellation,
) -> Result<(Vec<KrakenSpotTrade>, u64)> {
    let mut cursor = range_start
        .timestamp_nanos_opt()
        .context("Kraken Spot range start is outside nanosecond range")?;
    let end_ns = range_end
        .timestamp_nanos_opt()
        .context("Kraken Spot range end is outside nanosecond range")?;
    let mut records = Vec::new();
    let mut seen = HashSet::new();
    let mut pages = 0u64;
    let mut consecutive_rate_limits = 0u32;

    while cursor < end_ns {
        if cancellation.is_cancelled() {
            bail!("Kraken Spot trade download cancelled");
        }
        if usize::try_from(pages).unwrap_or(usize::MAX) >= MAX_PAGES_PER_DAY {
            bail!("Kraken Spot daily trade pagination exceeded safety limit");
        }
        let response = client
            .get(format!(
                "{}/0/public/Trades",
                config.base_url.trim_end_matches('/')
            ))
            .query(&[
                ("pair", KRAKEN_SPOT_SYMBOL.to_string()),
                ("since", cursor.to_string()),
                ("count", PAGE_SIZE.to_string()),
            ])
            .send()
            .await
            .context("Kraken Spot Trades request failed")?;
        let status = response.status();
        let payload: Value = response
            .json()
            .await
            .context("failed to decode Kraken Spot Trades response")?;
        if !status.is_success() {
            bail!("Kraken Spot Trades returned HTTP {status}: {payload}");
        }
        let errors = payload
            .get("error")
            .and_then(Value::as_array)
            .context("Kraken Spot Trades response omitted error array")?;
        if !errors.is_empty() {
            let message = errors
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            if message.contains("Rate limit") {
                consecutive_rate_limits = consecutive_rate_limits.saturating_add(1);
                if consecutive_rate_limits > 12 {
                    bail!("Kraken Spot rate limit remained active after retries");
                }
                sleep(Duration::from_secs(u64::from(
                    consecutive_rate_limits.min(10),
                )))
                .await;
                continue;
            }
            bail!("Kraken Spot Trades returned {message}");
        }
        consecutive_rate_limits = 0;
        let result = payload
            .get("result")
            .and_then(Value::as_object)
            .context("Kraken Spot Trades response omitted result")?;
        let rows = result
            .get(KRAKEN_SPOT_PAIR_KEY)
            .and_then(Value::as_array)
            .context("Kraken Spot Trades response omitted BTC/USD rows")?;
        let next_cursor = result
            .get("last")
            .and_then(Value::as_str)
            .context("Kraken Spot Trades response omitted pagination cursor")?
            .parse::<i64>()
            .context("Kraken Spot pagination cursor was invalid")?;
        let mut reached_end = false;
        for row in rows {
            let trade = parse_trade(row)?;
            if trade.timestamp_ns >= end_ns {
                reached_end = true;
                break;
            }
            if trade.timestamp_ns >= cursor && seen.insert(trade.trade_id) {
                records.push(trade);
            }
        }
        pages = pages.saturating_add(1);
        if reached_end {
            break;
        }
        if next_cursor <= cursor {
            bail!("Kraken Spot pagination cursor did not advance");
        }
        cursor = next_cursor;
        sleep(config.request_delay).await;
    }
    records.sort_by_key(|trade| (trade.timestamp_ns, trade.trade_id));
    Ok((records, pages))
}

fn parse_trade(value: &Value) -> Result<KrakenSpotTrade> {
    let row = value
        .as_array()
        .context("Kraken Spot trade row was not an array")?;
    if row.len() < 7 {
        bail!("Kraken Spot trade row was shorter than seven fields");
    }
    let timestamp_seconds = Decimal::from_str(&row[2].to_string())
        .context("Kraken Spot trade timestamp was not numeric")?;
    let timestamp_ns = (timestamp_seconds * Decimal::from(1_000_000_000_i64))
        .round()
        .to_i64()
        .context("Kraken Spot trade timestamp was outside the supported range")?;
    Ok(KrakenSpotTrade {
        trade_id: row[6]
            .as_i64()
            .context("Kraken Spot trade ID was not an integer")?,
        timestamp_ns,
        price: Decimal::from_str(
            row[0]
                .as_str()
                .context("Kraken Spot trade price was not a string")?,
        )
        .context("Kraken Spot trade price was invalid")?,
        base_volume: Decimal::from_str(
            row[1]
                .as_str()
                .context("Kraken Spot trade volume was not a string")?,
        )
        .context("Kraken Spot trade volume was invalid")?,
        side: row[3]
            .as_str()
            .context("Kraken Spot trade side was not a string")?
            .to_string(),
        order_type: row[4]
            .as_str()
            .context("Kraken Spot trade order type was not a string")?
            .to_string(),
    })
}

fn publish_day(
    root: &Path,
    range_start: DateTime<Utc>,
    range_end: DateTime<Utc>,
    trades: Vec<KrakenSpotTrade>,
    pages: u64,
) -> Result<PublishedKrakenSpotDay> {
    let candles = aggregate_candles(&trades);
    let partition = format!(
        "symbol=BTCUSD/year={:04}/month={:02}/day={:02}",
        range_start.year(),
        range_start.month(),
        range_start.day()
    );
    let trade_relative = PathBuf::from("dataset=trade_prints")
        .join(&partition)
        .join(format!(
            "{}-{}.parquet",
            range_start.format("%Y%m%dT%H%M%SZ"),
            range_end.format("%Y%m%dT%H%M%SZ")
        ));
    let candle_relative = PathBuf::from("dataset=ohlcv_1s")
        .join(&partition)
        .join(format!(
            "{}-{}.parquet",
            range_start.format("%Y%m%dT%H%M%SZ"),
            range_end.format("%Y%m%dT%H%M%SZ")
        ));
    let staging = root.join(".staging");
    fs::create_dir_all(&staging).context("failed to create Kraken Spot staging directory")?;
    let trade_tmp = staging.join(format!("{}-trades.parquet.tmp", range_start.timestamp()));
    let candle_tmp = staging.join(format!("{}-ohlcv.parquet.tmp", range_start.timestamp()));
    write_trades(&trade_tmp, &trades)?;
    write_candles(&candle_tmp, &candles)?;
    let (trade_sha256, trade_bytes) = hash_file(&trade_tmp)?;
    let (candle_sha256, candle_bytes) = hash_file(&candle_tmp)?;
    publish_atomic(&trade_tmp, &root.join(&trade_relative), &trade_sha256)?;
    publish_atomic(&candle_tmp, &root.join(&candle_relative), &candle_sha256)?;
    let minimum_timestamp = DateTime::from_timestamp_nanos(trades.first().unwrap().timestamp_ns);
    let maximum_timestamp = DateTime::from_timestamp_nanos(trades.last().unwrap().timestamp_ns);
    let combined_sha256 = format!(
        "{:x}",
        Sha256::digest(format!("{trade_sha256}:{candle_sha256}").as_bytes())
    );
    Ok(PublishedKrakenSpotDay {
        trade_path: trade_relative.to_string_lossy().to_string(),
        candle_path: candle_relative.to_string_lossy().to_string(),
        trade_sha256,
        candle_sha256,
        combined_sha256,
        trade_bytes,
        candle_bytes,
        trade_count: trades.len() as u64,
        candle_count: candles.len() as u64,
        minimum_timestamp,
        maximum_timestamp,
        pages,
    })
}

fn aggregate_candles(trades: &[KrakenSpotTrade]) -> Vec<OneSecondCandle> {
    let mut candles = BTreeMap::<i64, OneSecondCandle>::new();
    for trade in trades {
        let second = trade.timestamp_ns.div_euclid(1_000_000_000);
        match candles.get_mut(&second) {
            Some(candle) => candle.observe(trade),
            None => {
                candles.insert(second, OneSecondCandle::new(trade));
            }
        }
    }
    candles.into_values().collect()
}

fn parquet_properties() -> Result<WriterProperties> {
    Ok(WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(6).context("invalid Kraken Spot Zstandard level")?,
        ))
        .build())
}

fn write_trades(path: &Path, trades: &[KrakenSpotTrade]) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("exchange", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("trade_id", DataType::Int64, false),
        Field::new("event_timestamp_ns", DataType::Int64, false),
        Field::new("price", DataType::Utf8, false),
        Field::new("base_volume", DataType::Utf8, false),
        Field::new("side", DataType::Utf8, false),
        Field::new("order_type", DataType::Utf8, false),
        Field::new("schema_version", DataType::Utf8, false),
    ]));
    let prices = trades
        .iter()
        .map(|v| v.price.to_string())
        .collect::<Vec<_>>();
    let volumes = trades
        .iter()
        .map(|v| v.base_volume.to_string())
        .collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![KRAKEN_SPOT_PROVIDER; trades.len()])),
            Arc::new(StringArray::from(vec!["BTCUSD"; trades.len()])),
            Arc::new(Int64Array::from(
                trades.iter().map(|v| v.trade_id).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                trades.iter().map(|v| v.timestamp_ns).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                prices.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                volumes.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                trades.iter().map(|v| v.side.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                trades
                    .iter()
                    .map(|v| v.order_type.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(vec![
                KRAKEN_SPOT_SCHEMA_VERSION;
                trades.len()
            ])),
        ],
    )?;
    write_batch(path, schema, batch)
}

fn write_candles(path: &Path, candles: &[OneSecondCandle]) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("exchange", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("bucket_start_ns", DataType::Int64, false),
        Field::new("open", DataType::Utf8, false),
        Field::new("high", DataType::Utf8, false),
        Field::new("low", DataType::Utf8, false),
        Field::new("close", DataType::Utf8, false),
        Field::new("base_volume", DataType::Utf8, false),
        Field::new("quote_volume", DataType::Utf8, false),
        Field::new("trade_count", DataType::Int64, false),
        Field::new("first_trade_id", DataType::Int64, false),
        Field::new("last_trade_id", DataType::Int64, false),
        Field::new("has_trades", DataType::Boolean, false),
        Field::new("schema_version", DataType::Utf8, false),
    ]));
    let strings = |f: fn(&OneSecondCandle) -> Decimal| {
        candles.iter().map(|v| f(v).to_string()).collect::<Vec<_>>()
    };
    let open = strings(|v| v.open);
    let high = strings(|v| v.high);
    let low = strings(|v| v.low);
    let close = strings(|v| v.close);
    let base = strings(|v| v.base_volume);
    let quote = strings(|v| v.quote_volume);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![KRAKEN_SPOT_PROVIDER; candles.len()])),
            Arc::new(StringArray::from(vec!["BTCUSD"; candles.len()])),
            Arc::new(Int64Array::from(
                candles
                    .iter()
                    .map(|v| v.second_start_ns)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                open.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                high.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                low.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                close.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                base.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                quote.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                candles.iter().map(|v| v.trade_count).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                candles.iter().map(|v| v.first_trade_id).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                candles.iter().map(|v| v.last_trade_id).collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(vec![true; candles.len()])),
            Arc::new(StringArray::from(vec![
                KRAKEN_SPOT_SCHEMA_VERSION;
                candles.len()
            ])),
        ],
    )?;
    write_batch(path, schema, batch)
}

fn write_batch(path: &Path, schema: Arc<Schema>, batch: RecordBatch) -> Result<()> {
    let file = File::create(path).context("failed to create Kraken Spot Parquet staging file")?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(parquet_properties()?))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn publish_atomic(staging: &Path, final_path: &Path, expected_hash: &str) -> Result<()> {
    let parent = final_path
        .parent()
        .context("Kraken Spot output path has no parent")?;
    fs::create_dir_all(parent).context("failed to create Kraken Spot partition directory")?;
    if final_path.exists() {
        let (actual, _) = hash_file(final_path)?;
        if actual != expected_hash {
            bail!("existing Kraken Spot Parquet object failed content verification");
        }
        fs::remove_file(staging).context("failed to remove duplicate Kraken Spot staging file")?;
    } else {
        fs::rename(staging, final_path)
            .context("failed to publish Kraken Spot Parquet atomically")?;
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<(String, u64)> {
    let file = File::open(path).context("failed to open Kraken Spot Parquet for hashing")?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    let mut bytes = 0u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        bytes = bytes.saturating_add(read as u64);
    }
    Ok((format!("{:x}", digest.finalize()), bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trade(id: i64, ns: i64, price: i64, volume: i64) -> KrakenSpotTrade {
        KrakenSpotTrade {
            trade_id: id,
            timestamp_ns: ns,
            price: Decimal::from(price),
            base_volume: Decimal::from(volume),
            side: "b".to_string(),
            order_type: "l".to_string(),
        }
    }

    #[test]
    fn aggregates_trade_prints_by_exchange_second() {
        let rows = vec![
            trade(1, 1_100_000_000, 100, 2),
            trade(2, 1_900_000_000, 105, 3),
            trade(3, 2_000_000_000, 99, 1),
        ];
        let candles = aggregate_candles(&rows);
        assert_eq!(candles.len(), 2);
        assert_eq!(candles[0].open, Decimal::from(100));
        assert_eq!(candles[0].high, Decimal::from(105));
        assert_eq!(candles[0].low, Decimal::from(100));
        assert_eq!(candles[0].close, Decimal::from(105));
        assert_eq!(candles[0].base_volume, Decimal::from(5));
        assert_eq!(candles[0].trade_count, 2);
        assert_eq!(candles[0].first_trade_id, 1);
        assert_eq!(candles[0].last_trade_id, 2);
    }
}
