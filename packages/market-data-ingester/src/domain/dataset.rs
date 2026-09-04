use std::fmt;

use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq)]
pub struct BinanceAggregateTradeRecord {
    pub symbol: String,
    pub aggregate_trade_id: i64,
    pub price: Decimal,
    pub quantity: Decimal,
    pub first_trade_id: i64,
    pub last_trade_id: i64,
    pub trade_timestamp: DateTime<Utc>,
    pub buyer_maker: bool,
    pub best_match: bool,
    pub payload_sha256: String,
}

impl BinanceAggregateTradeRecord {
    pub fn canonical_payload_sha256(&self) -> String {
        let payload = format!(
            "v1|source=binance_spot|symbol={}|aggregate_trade_id={}|trade_timestamp_ms={}|price={}|quantity={}|first_trade_id={}|last_trade_id={}|buyer_maker={}|best_match={}",
            self.symbol,
            self.aggregate_trade_id,
            self.trade_timestamp.timestamp_millis(),
            self.price.normalize(),
            self.quantity.normalize(),
            self.first_trade_id,
            self.last_trade_id,
            self.buyer_maker,
            self.best_match,
        );
        format!("{:x}", Sha256::digest(payload.as_bytes()))
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct BinanceOneSecondKlineRecord {
    pub symbol: String,
    pub open_timestamp: DateTime<Utc>,
    pub close_timestamp: DateTime<Utc>,
    pub open_price: Decimal,
    pub high_price: Decimal,
    pub low_price: Decimal,
    pub close_price: Decimal,
    pub base_volume: Decimal,
    pub quote_volume: Decimal,
    pub trade_count: i64,
    pub taker_buy_base_volume: Decimal,
    pub taker_buy_quote_volume: Decimal,
}

impl BinanceOneSecondKlineRecord {
    pub fn canonical_payload_sha256(&self) -> String {
        let canonical_close = self.open_timestamp + chrono::Duration::milliseconds(999);
        let payload = format!(
            "v1|source=binance_spot|symbol={}|open_timestamp_ms={}|close_timestamp_ms={}|open_price={}|high_price={}|low_price={}|close_price={}|base_volume={}|quote_volume={}|trade_count={}|taker_buy_base_volume={}|taker_buy_quote_volume={}",
            self.symbol,
            self.open_timestamp.timestamp_millis(),
            canonical_close.timestamp_millis(),
            self.open_price.normalize(),
            self.high_price.normalize(),
            self.low_price.normalize(),
            self.close_price.normalize(),
            self.base_volume.normalize(),
            self.quote_volume.normalize(),
            self.trade_count,
            self.taker_buy_base_volume.normalize(),
            self.taker_buy_quote_volume.normalize(),
        );
        format!("{:x}", Sha256::digest(payload.as_bytes()))
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct BinanceBtcusdtOpenInterestRecord {
    pub symbol: String,
    pub source_timestamp: DateTime<Utc>,
    pub period_seconds: i32,
    pub sum_open_interest: Decimal,
    pub sum_open_interest_value: Decimal,
    pub cmc_circulating_supply: Option<Decimal>,
}

impl BinanceBtcusdtOpenInterestRecord {
    pub fn canonical_source_payload(&self) -> serde_json::Value {
        serde_json::json!({
            "CMCCirculatingSupply": self
                .cmc_circulating_supply
                .map(|value| value.normalize().to_string()),
            "sumOpenInterest": self.sum_open_interest.normalize().to_string(),
            "sumOpenInterestValue": self.sum_open_interest_value.normalize().to_string(),
            "symbol": self.symbol,
            "timestamp": self.source_timestamp.timestamp_millis(),
        })
    }

    pub fn canonical_payload_sha256(&self) -> String {
        let canonical = serde_json::json!({
            "cmc_circulating_supply": self
                .cmc_circulating_supply
                .map(|value| value.normalize().to_string()),
            "sum_open_interest": self.sum_open_interest.normalize().to_string(),
            "sum_open_interest_value": self.sum_open_interest_value.normalize().to_string(),
            "symbol": self.symbol,
            "timestamp": self.source_timestamp.timestamp_millis(),
        });
        format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(&canonical).expect("canonical open-interest JSON serializes")
            )
        )
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct ChainlinkBtcusdArchiveTick {
    pub feed_id: String,
    pub source_timestamp: DateTime<Utc>,
    pub valid_from_timestamp: DateTime<Utc>,
    pub price: Decimal,
    pub bid: Decimal,
    pub ask: Decimal,
    pub report_sha256: String,
}
#[derive(Debug, Clone, PartialEq)]
pub struct ChainlinkBtcusdOneMinuteCandle {
    pub symbol: String,
    pub open_timestamp: DateTime<Utc>,
    pub close_timestamp: DateTime<Utc>,
    pub open_price: Decimal,
    pub high_price: Decimal,
    pub low_price: Decimal,
    pub close_price: Decimal,
    pub volume: Option<Decimal>,
    pub volume_supported: bool,
}

impl ChainlinkBtcusdOneMinuteCandle {
    pub fn canonical_payload_sha256(&self) -> String {
        let payload = format!(
            "v1|source=chainlink_candlestick|symbol={}|open_timestamp={}|close_timestamp={}|open_price={}|high_price={}|low_price={}|close_price={}|volume=unsupported",
            self.symbol,
            self.open_timestamp.timestamp(),
            self.close_timestamp.timestamp(),
            self.open_price.normalize(),
            self.high_price.normalize(),
            self.low_price.normalize(),
            self.close_price.normalize(),
        );
        format!("{:x}", Sha256::digest(payload.as_bytes()))
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct PolygonChainlinkBtcusdOracleRound {
    pub chain_id: i64,
    pub feed_proxy_address: String,
    pub aggregator_address: String,
    pub phase_id: i32,
    pub aggregator_round_id: i64,
    pub source_timestamp: DateTime<Utc>,
    pub block_timestamp: DateTime<Utc>,
    pub answer_raw: Decimal,
    pub price: Decimal,
    pub decimals: i32,
    pub block_number: i64,
    pub block_hash: String,
    pub transaction_hash: String,
    pub log_index: i32,
}

impl PolygonChainlinkBtcusdOracleRound {
    pub fn canonical_source_payload(&self) -> serde_json::Value {
        serde_json::json!({
            "aggregatorAddress": self.aggregator_address,
            "aggregatorRoundId": self.aggregator_round_id,
            "answerRaw": self.answer_raw.normalize().to_string(),
            "blockHash": self.block_hash,
            "blockNumber": self.block_number,
            "blockTimestamp": self.block_timestamp.timestamp(),
            "chainId": self.chain_id,
            "decimals": self.decimals,
            "feedProxyAddress": self.feed_proxy_address,
            "logIndex": self.log_index,
            "phaseId": self.phase_id,
            "sourceTimestamp": self.source_timestamp.timestamp(),
            "transactionHash": self.transaction_hash,
        })
    }

    pub fn canonical_payload_sha256(&self) -> String {
        format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(&self.canonical_source_payload())
                    .expect("canonical oracle JSON serializes")
            )
        )
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct PmdataChainlinkBtcusdTwapRecord {
    pub source_timestamp: DateTime<Utc>,
    pub provider_received_at: DateTime<Utc>,
    pub valid_from_timestamp: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub window_seconds: i16,
    pub twap_price: Decimal,
    pub full_accuracy_value: String,
    pub report_version: String,
    pub source_date: NaiveDate,
    pub archive_row_number: i64,
}
#[derive(Debug, Clone, PartialEq)]
pub struct PmdataChainlinkBtcusdRefpriceRecord {
    pub source_timestamp: DateTime<Utc>,
    pub provider_received_at: DateTime<Utc>,
    pub valid_from_timestamp: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub price: Decimal,
    pub bid: Option<Decimal>,
    pub ask: Option<Decimal>,
    pub report_version: Option<String>,
    pub canonical_row_sha256: String,
    pub source_date: NaiveDate,
    pub archive_row_number: i64,
}

/// Stable identity of a logical data product. Collection mode and physical
/// storage are deliberately excluded from this identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DatasetKey {
    BinanceSpotAggregateTrades,
    BinanceSpotOneSecondOhlcv,
    BinanceFuturesOpenInterest,
    BinanceSpotL2Snapshots,
    BinanceSpotL2OneSecondFeatures,
    BinanceFuturesL2OneSecondFeatures,
    ChainlinkBtcusdReferencePrices,
    ChainlinkBtcusdOneMinuteCandles,
    PolygonChainlinkBtcusdOracleRounds,
    PmdataChainlinkBtcusdTwap,
    PolymarketChainlinkBtcusdTwap,
    PolymarketBtcFiveMinuteOrderbookSnapshots,
}

impl DatasetKey {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BinanceSpotAggregateTrades => "binance_spot_btcusdt_aggregate_trades",
            Self::BinanceSpotOneSecondOhlcv => "binance_spot_btcusdt_one_second_ohlcv",
            Self::BinanceFuturesOpenInterest => "binance_futures_btcusdt_open_interest",
            Self::BinanceSpotL2Snapshots => "binance_spot_btcusdt_l2_snapshots",
            Self::BinanceSpotL2OneSecondFeatures => "binance_spot_btcusdt_l2_one_second_features",
            Self::BinanceFuturesL2OneSecondFeatures => {
                "binance_futures_btcusdt_l2_one_second_features"
            }
            Self::ChainlinkBtcusdReferencePrices => "chainlink_btcusd_reference_prices",
            Self::ChainlinkBtcusdOneMinuteCandles => "chainlink_btcusd_one_minute_candles",
            Self::PolygonChainlinkBtcusdOracleRounds => "polygon_chainlink_btcusd_oracle_rounds",
            Self::PmdataChainlinkBtcusdTwap => "pmdata_chainlink_btcusd_twap",
            Self::PolymarketChainlinkBtcusdTwap => "polymarket_chainlink_btcusd_twap",
            Self::PolymarketBtcFiveMinuteOrderbookSnapshots => {
                "polymarket_btc_five_minute_orderbook_snapshots"
            }
        }
    }
}

impl fmt::Display for DatasetKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetContract {
    pub key: DatasetKey,
    pub version: i32,
    pub canonical_table: &'static str,
    pub natural_key: &'static [&'static str],
    pub fields: &'static [&'static str],
}

const fn contract(
    key: DatasetKey,
    canonical_table: &'static str,
    natural_key: &'static [&'static str],
    fields: &'static [&'static str],
) -> DatasetContract {
    DatasetContract {
        key,
        version: 1,
        canonical_table,
        natural_key,
        fields,
    }
}

pub const CONTRACTS: &[DatasetContract] = &[
    contract(
        DatasetKey::BinanceSpotAggregateTrades,
        "market_data.binance_spot_btcusdt_aggregate_trades",
        &["source", "symbol", "aggregate_trade_id"],
        &[
            "source",
            "symbol",
            "aggregate_trade_id",
            "trade_timestamp",
            "provider_available_at",
            "received_at",
            "price",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "buyer_maker",
            "best_match",
            "payload_sha256",
            "strategy_key",
            "capture_artifact_id",
            "ingested_at",
        ],
    ),
    contract(
        DatasetKey::BinanceSpotOneSecondOhlcv,
        "market_data.binance_spot_btcusdt_one_second_ohlcv",
        &["source", "symbol", "open_timestamp"],
        &[
            "source",
            "symbol",
            "open_timestamp",
            "close_timestamp",
            "provider_available_at",
            "received_at",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "base_volume",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "payload_sha256",
            "strategy_key",
            "capture_artifact_id",
            "ingested_at",
        ],
    ),
    contract(
        DatasetKey::BinanceFuturesOpenInterest,
        "market_data.binance_futures_btcusdt_open_interest",
        &["source", "symbol", "source_timestamp", "period_seconds"],
        &[
            "source",
            "source_timestamp",
            "symbol",
            "period_seconds",
            "sum_open_interest",
            "sum_open_interest_value",
            "cmc_circulating_supply",
            "provider_available_at",
            "received_at",
            "source_payload",
            "payload_sha256",
            "strategy_key",
            "capture_artifact_id",
            "ingested_at",
        ],
    ),
    contract(
        DatasetKey::BinanceSpotL2Snapshots,
        "market_data.binance_spot_btcusdt_l2_snapshots",
        &["source", "symbol", "connection_epoch", "source_update_id"],
        &[
            "source_timestamp",
            "received_at",
            "source",
            "symbol",
            "source_update_id",
            "connection_epoch",
            "sample_depth",
            "bids",
            "asks",
            "book_sha256",
            "sampling_policy",
            "sampling_policy_sha256",
            "payload_sha256",
            "strategy_key",
            "capture_artifact_id",
            "ingested_at",
        ],
    ),
    contract(
        DatasetKey::BinanceSpotL2OneSecondFeatures,
        "market_data.binance_spot_btcusdt_l2_one_second_features",
        &["symbol", "second_start"],
        &[
            "symbol",
            "second_start",
            "source_event_timestamp",
            "provider_received_at",
            "available_at",
            "source_update_id",
            "feature_schema_version",
            "quality_status",
            "artifact_id",
            "midpoint",
            "microprice",
            "spread_bps",
            "ingested_at",
        ],
    ),
    contract(
        DatasetKey::BinanceFuturesL2OneSecondFeatures,
        "market_data.binance_futures_btcusdt_l2_one_second_features",
        &["symbol", "second_start"],
        &[
            "symbol",
            "second_start",
            "source_event_timestamp",
            "provider_received_at",
            "available_at",
            "source_update_id",
            "feature_schema_version",
            "quality_status",
            "artifact_id",
            "midpoint",
            "microprice",
            "spread_bps",
            "ingested_at",
        ],
    ),
    contract(
        DatasetKey::ChainlinkBtcusdReferencePrices,
        "market_data.chainlink_btcusd_reference_prices",
        &["source", "feed_id", "source_timestamp", "report_sha256"],
        &[
            "source",
            "feed_id",
            "source_timestamp",
            "valid_from_timestamp",
            "provider_available_at",
            "received_at",
            "price",
            "bid",
            "ask",
            "report_sha256",
            "payload_sha256",
            "strategy_key",
            "capture_artifact_id",
            "ingested_at",
            "expires_at",
            "report_version",
            "source_date",
            "archive_row_number",
            "backfill_artifact_id",
            "report_hash_kind",
        ],
    ),
    contract(
        DatasetKey::ChainlinkBtcusdOneMinuteCandles,
        "market_data.chainlink_btcusd_one_minute_candles",
        &["source", "symbol", "open_timestamp"],
        &[
            "source",
            "symbol",
            "open_timestamp",
            "close_timestamp",
            "provider_available_at",
            "received_at",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
            "volume_supported",
            "payload_sha256",
            "strategy_key",
            "capture_artifact_id",
            "ingested_at",
        ],
    ),
    contract(
        DatasetKey::PolygonChainlinkBtcusdOracleRounds,
        "market_data.polygon_chainlink_btcusd_oracle_rounds",
        &[
            "chain_id",
            "aggregator_address",
            "phase_id",
            "aggregator_round_id",
        ],
        &[
            "source",
            "chain_id",
            "feed_proxy_address",
            "aggregator_address",
            "phase_id",
            "aggregator_round_id",
            "source_timestamp",
            "block_timestamp",
            "answer_raw",
            "price",
            "decimals",
            "block_number",
            "block_hash",
            "transaction_hash",
            "log_index",
            "provider_available_at",
            "received_at",
            "source_payload",
            "payload_sha256",
            "strategy_key",
            "capture_artifact_id",
            "ingested_at",
        ],
    ),
    contract(
        DatasetKey::PmdataChainlinkBtcusdTwap,
        "market_data.pmdata_chainlink_btcusd_twap",
        &["source_date", "archive_row_number"],
        &[
            "source_timestamp",
            "provider_received_at",
            "valid_from_timestamp",
            "expires_at",
            "symbol",
            "window_seconds",
            "twap_price",
            "full_accuracy_value",
            "report_version",
            "source_date",
            "archive_row_number",
            "artifact_id",
            "ingested_at",
        ],
    ),
    contract(
        DatasetKey::PolymarketChainlinkBtcusdTwap,
        "market_data.polymarket_chainlink_btcusd_twap",
        &["source", "symbol", "source_timestamp", "window_seconds"],
        &[
            "source_timestamp",
            "published_at",
            "received_at",
            "source",
            "symbol",
            "window_seconds",
            "twap_price",
            "full_accuracy_value",
            "source_payload",
            "payload_sha256",
            "strategy_key",
            "capture_artifact_id",
            "ingested_at",
        ],
    ),
    contract(
        DatasetKey::PolymarketBtcFiveMinuteOrderbookSnapshots,
        "polymarket.btc_five_minute_orderbook_snapshots",
        &["source", "token_id", "sampled_at"],
        &[
            "sampled_at",
            "source_timestamp",
            "provider_available_at",
            "received_at",
            "source",
            "market_id",
            "condition_id",
            "event_slug",
            "window_start",
            "window_end",
            "token_id",
            "outcome",
            "connection_epoch",
            "ingest_sequence",
            "tick_size",
            "best_bid",
            "best_ask",
            "bid_depth",
            "ask_depth",
            "bids",
            "asks",
            "source_hash",
            "book_sha256",
            "sampling_policy",
            "sampling_policy_sha256",
            "payload_sha256",
            "strategy_key",
            "capture_artifact_id",
            "ingested_at",
        ],
    ),
];

pub fn contract_for(key: DatasetKey) -> Option<&'static DatasetContract> {
    CONTRACTS.iter().find(|item| item.key == key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use std::collections::BTreeSet;

    #[test]
    fn contracts_have_unique_keys_and_canonical_tables() {
        let keys = CONTRACTS
            .iter()
            .map(|item| item.key)
            .collect::<BTreeSet<_>>();
        let tables = CONTRACTS
            .iter()
            .map(|item| item.canonical_table)
            .collect::<BTreeSet<_>>();
        assert_eq!(keys.len(), CONTRACTS.len());
        assert_eq!(tables.len(), CONTRACTS.len());
        assert!(CONTRACTS
            .iter()
            .all(|item| item.canonical_table.starts_with("market_data.")));
    }

    #[test]
    fn natural_key_columns_are_contract_fields() {
        for contract in CONTRACTS {
            for key in contract.natural_key {
                assert!(
                    contract.fields.contains(key),
                    "{key} missing from {}",
                    contract.key
                );
            }
        }
    }

    #[test]
    fn open_interest_payload_hash_is_stable_across_realtime_and_backfill() {
        let record = BinanceBtcusdtOpenInterestRecord {
            symbol: "BTCUSDT".to_owned(),
            source_timestamp: Utc.timestamp_millis_opt(1_783_036_800_000).unwrap(),
            period_seconds: 300,
            sum_open_interest: "106938.477".parse().unwrap(),
            sum_open_interest_value: "6581058037.6662".parse().unwrap(),
            cmc_circulating_supply: Some("20050843".parse().unwrap()),
        };
        assert_eq!(
            record.canonical_payload_sha256(),
            "858426ba35c3dcef2b7538b76cf1eaa3940bd239c649cffeaf0a25a3ff8c8b3d"
        );
    }
}
