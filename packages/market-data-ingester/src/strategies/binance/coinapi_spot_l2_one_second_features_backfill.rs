use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    str::FromStr,
    time::Duration,
};

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use reqwest::{Client, Url};
use rust_decimal::Decimal;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::fs;

use super::{backfill_support, l2_backfill_support, types::BinanceL2OneSecondFeature};
use crate::domain::{
    BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
    BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
};

pub const STRATEGY_KEY: &str = "coinapi_binance_spot_btcusdt_l2_one_second_features_backfill";
const ARTIFACT_KEY: &str = "binance_spot_btcusdt_l2_one_second_features";
const TARGET: &str = "polymarket.binance_spot_btcusdt_l2_one_second_features";
const PROVIDER: &str = "coinapi";
const SYMBOL: &str = "BTCUSDT";
const COINAPI_SYMBOL: &str = "BINANCE_SPOT_BTC_USDT";
const FEATURE_SCHEMA: &str = "binance-spot-btcusdt-l2-one-second-features-v1";
const MATERIALIZATION_CONTRACT: &str = "coinapi-binance-spot-btcusdt-l2-snapshots-v1";
const MAX_SNAPSHOTS: usize = 100_000;
const HORIZONS: [i64; 5] = [1, 5, 15, 30, 60];

pub struct CoinapiBinanceSpotL2OneSecondFeaturesBackfill {
    descriptor: StrategyDescriptor,
}

impl CoinapiBinanceSpotL2OneSecondFeaturesBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        Ok(Self {
            descriptor: backfill_support::descriptor(
                STRATEGY_KEY,
                "CoinAPI Binance Spot BTCUSDT L2 one-second features backfill",
                "Materializes Binance Spot BTCUSDT L2 one-second features from CoinAPI order-book history",
            )?,
        })
    }
}

#[async_trait]
impl BackfillWorkerStrategy for CoinapiBinanceSpotL2OneSecondFeaturesBackfill {
    fn descriptor(&self) -> &StrategyDescriptor {
        &self.descriptor
    }

    fn validate_request(
        &self,
        request: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
        backfill_support::validate_empty_request(&self.descriptor, request)
    }

    fn plan_shards(
        &self,
        request: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        backfill_support::hourly_shards(request, self.descriptor.maximum_shards)
    }

    async fn execute_backfill(
        &self,
        context: BackfillContext,
        shard: BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        execute(context, shard).await
    }
}

#[derive(Debug, Deserialize)]
struct Snapshot {
    symbol_id: String,
    time_exchange: String,
    time_coinapi: String,
    bids: Vec<Level>,
    asks: Vec<Level>,
}

#[derive(Debug, Deserialize)]
struct Level {
    #[serde(deserialize_with = "number")]
    price: f64,
    #[serde(deserialize_with = "number")]
    size: f64,
}

fn number<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Number(value) => value
            .as_f64()
            .ok_or_else(|| serde::de::Error::custom("number outside f64 range")),
        serde_json::Value::String(value) => value.parse().map_err(serde::de::Error::custom),
        _ => Err(serde::de::Error::custom("expected numeric level")),
    }
}

#[derive(Clone)]
struct State {
    second: DateTime<Utc>,
    source: DateTime<Utc>,
    received: DateTime<Utc>,
    update_id: i64,
    bids: BTreeMap<String, (f64, f64)>,
    asks: BTreeMap<String, (f64, f64)>,
    midpoint: f64,
    microprice: f64,
    spread_bps: f64,
    bid5: f64,
    ask5: f64,
    bid10: f64,
    ask10: f64,
    bid20: f64,
    ask20: f64,
    imbalance5: f64,
    imbalance10: f64,
    imbalance20: f64,
    bid_slope20: f64,
    ask_slope20: f64,
    bid_concentration20: f64,
    ask_concentration20: f64,
}

async fn execute(
    context: BackfillContext,
    shard: BackfillShard,
) -> Result<BackfillOutcome, BackfillExecutionError> {
    let logical_key = format!(
        "coinapi:binance-spot:{SYMBOL}:l2-snapshots-v1:{}:{}",
        shard.range_start.to_rfc3339(),
        shard.range_end.to_rfc3339()
    );
    if let Some(outcome) =
        backfill_support::completed_outcome(&context, STRATEGY_KEY, &logical_key).await?
    {
        return Ok(outcome);
    }
    let api_key = std::env::var("COIN_API_KEY")
        .map_err(|_| backfill_support::source_error("COIN_API_KEY is required"))?;
    let mut url = Url::parse(
        &std::env::var("COINAPI_API_ORIGIN")
            .unwrap_or_else(|_| "https://api-ncsa.coinapi.io".into()),
    )
    .map_err(backfill_support::source_error)?;
    url.set_path(&format!("/v1/orderbooks/{COINAPI_SYMBOL}/history"));
    url.query_pairs_mut()
        .append_pair(
            "time_start",
            &(shard.range_start - chrono::Duration::seconds(61)).to_rfc3339(),
        )
        .append_pair("time_end", &shard.range_end.to_rfc3339())
        .append_pair("limit", &MAX_SNAPSHOTS.to_string())
        .append_pair("limit_levels", "20");

    let root = std::env::var("INGESTER_COINAPI_ARCHIVE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/binance-l2/coinapi"));
    let directory = root
        .join("binance-spot")
        .join(SYMBOL)
        .join(shard.range_start.format("%Y-%m-%d").to_string());
    fs::create_dir_all(&directory)
        .await
        .map_err(backfill_support::source_error)?;
    let archive = directory.join(format!(
        "orderbook-{}-{}.json",
        shard.range_start.format("%Y%m%dT%H%M%SZ"),
        shard.range_end.format("%Y%m%dT%H%M%SZ")
    ));
    let payload = load_payload(&context, &url, &api_key, &archive).await?;
    let checksum = format!("{:x}", Sha256::digest(&payload));
    let snapshots: Vec<Snapshot> =
        serde_json::from_slice(&payload).map_err(backfill_support::source_error)?;
    if snapshots.len() >= MAX_SNAPSHOTS {
        return Err(backfill_support::integrity(
            "CoinAPI response reached the snapshot cap; use smaller shards",
        ));
    }
    let source_snapshot_count = snapshots.len();
    let (states, rejected) = select_states(snapshots)?;
    let existing: BTreeSet<DateTime<Utc>> = sqlx::query_scalar(
        "SELECT second_start FROM polymarket.binance_spot_btcusdt_l2_one_second_features WHERE symbol=$1 AND second_start >= $2 AND second_start < $3",
    )
    .bind(SYMBOL)
    .bind(shard.range_start)
    .bind(shard.range_end)
    .fetch_all(&context.pool)
    .await
    .map_err(backfill_support::database_error)?
    .into_iter()
    .collect();
    let features = materialize_features(&states, &existing, &shard)?;
    let artifact_id = backfill_support::create_artifact(
        &context,
        STRATEGY_KEY,
        ARTIFACT_KEY,
        &logical_key,
        PROVIDER,
        url.as_str(),
        TARGET,
    )
    .await?;
    l2_backfill_support::persist(
        &context,
        artifact_id,
        TARGET,
        &features,
        &checksum,
        payload.len() as u64,
        json!({
            "materialization_contract": MATERIALIZATION_CONTRACT,
            "source_market": "binance-spot",
            "symbol": SYMBOL,
            "snapshot_limit_levels": 20,
            "availability_basis": "time_coinapi",
            "maximum_source_staleness_ms": 1000,
            "rolling_features_require_contiguous_coinapi_seconds": true,
            "flow_basis": "adjacent_snapshot_top_20_quote_delta_proxy",
            "source_snapshot_count": source_snapshot_count,
            "selected_one_second_states": states.len(),
            "rejected_snapshots": rejected,
            "existing_seconds_skipped": existing.len(),
            "candidate_feature_rows": features.len(),
            "archive_path": archive,
            "query_range_start": shard.range_start,
            "query_range_end": shard.range_end,
        }),
    )
    .await?;
    Ok(backfill_support::outcome(
        features.len() as i64,
        &shard,
        json!({
            "provider": PROVIDER,
            "records_verified": features.len(),
            "feature_schema_version": FEATURE_SCHEMA,
        }),
    ))
}

async fn load_payload(
    context: &BackfillContext,
    url: &Url,
    api_key: &str,
    archive: &PathBuf,
) -> Result<Vec<u8>, BackfillExecutionError> {
    if fs::try_exists(archive)
        .await
        .map_err(backfill_support::source_error)?
    {
        let bytes = fs::read(archive)
            .await
            .map_err(backfill_support::source_error)?;
        serde_json::from_slice::<serde_json::Value>(&bytes)
            .map_err(backfill_support::source_error)?;
        return Ok(bytes);
    }
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(900))
        .user_agent("capitonic-ingester-worker/1")
        .build()
        .map_err(backfill_support::source_error)?;
    let bytes = client
        .get(url.clone())
        .header("X-CoinAPI-Key", api_key)
        .send()
        .await
        .map_err(backfill_support::source_error)?
        .error_for_status()
        .map_err(backfill_support::source_error)?
        .bytes()
        .await
        .map_err(backfill_support::source_error)?
        .to_vec();
    serde_json::from_slice::<serde_json::Value>(&bytes).map_err(backfill_support::source_error)?;
    let temporary = archive.with_extension(format!("json.{}.part", context.job_id));
    fs::write(&temporary, &bytes)
        .await
        .map_err(backfill_support::source_error)?;
    fs::rename(&temporary, archive)
        .await
        .map_err(backfill_support::source_error)?;
    Ok(bytes)
}

fn select_states(
    snapshots: Vec<Snapshot>,
) -> Result<(BTreeMap<DateTime<Utc>, State>, usize), BackfillExecutionError> {
    let mut selected = BTreeMap::new();
    let mut rejected = 0;
    let mut previous_source = None;
    for (index, snapshot) in snapshots.into_iter().enumerate() {
        let state = match state(snapshot, index as i64) {
            Ok(value) => value,
            Err(_) => {
                rejected += 1;
                continue;
            }
        };
        if previous_source.is_some_and(|prior| state.source < prior) {
            return Err(backfill_support::integrity(
                "CoinAPI exchange timestamps were not monotonic",
            ));
        }
        previous_source = Some(state.source);
        if selected
            .get(&state.second)
            .is_none_or(|current: &State| state.received >= current.received)
        {
            selected.insert(state.second, state);
        }
    }
    Ok((selected, rejected))
}

fn state(snapshot: Snapshot, update_id: i64) -> Result<State, String> {
    if snapshot.symbol_id != COINAPI_SYMBOL {
        return Err("unexpected symbol".into());
    }
    let source = DateTime::parse_from_rfc3339(&snapshot.time_exchange)
        .map_err(|error| error.to_string())?
        .with_timezone(&Utc);
    let received = DateTime::parse_from_rfc3339(&snapshot.time_coinapi)
        .map_err(|error| error.to_string())?
        .with_timezone(&Utc);
    if !(0..=1000).contains(&(received - source).num_milliseconds()) {
        return Err("stale snapshot".into());
    }
    let bids = valid_levels(snapshot.bids, true)?;
    let asks = valid_levels(snapshot.asks, false)?;
    let best_bid = bids[0];
    let best_ask = asks[0];
    if best_bid.0 >= best_ask.0 {
        return Err("crossed or locked book".into());
    }
    let midpoint = (best_bid.0 + best_ask.0) / 2.0;
    let bid5 = depth(&bids, 5);
    let ask5 = depth(&asks, 5);
    let bid10 = depth(&bids, 10);
    let ask10 = depth(&asks, 10);
    let bid20 = depth(&bids, 20);
    let ask20 = depth(&asks, 20);
    Ok(State {
        second: Utc
            .timestamp_millis_opt(received.timestamp_millis().div_euclid(1000) * 1000)
            .single()
            .ok_or("invalid second")?,
        source,
        received,
        update_id,
        bids: level_map(&bids),
        asks: level_map(&asks),
        midpoint,
        microprice: (best_ask.0 * best_bid.1 + best_bid.0 * best_ask.1) / (best_bid.1 + best_ask.1),
        spread_bps: relative(best_ask.0, best_bid.0),
        bid5,
        ask5,
        bid10,
        ask10,
        bid20,
        ask20,
        imbalance5: imbalance(bid5, ask5),
        imbalance10: imbalance(bid10, ask10),
        imbalance20: imbalance(bid20, ask20),
        bid_slope20: relative(best_bid.0, bids[19].0) / bid20,
        ask_slope20: relative(asks[19].0, best_ask.0) / ask20,
        bid_concentration20: bid5 / bid20,
        ask_concentration20: ask5 / ask20,
    })
}

fn valid_levels(levels: Vec<Level>, bids: bool) -> Result<Vec<(f64, f64)>, String> {
    if levels.len() < 20 {
        return Err("insufficient depth".into());
    }
    let levels: Vec<_> = levels
        .into_iter()
        .take(20)
        .map(|value| (value.price, value.size))
        .collect();
    for (index, level) in levels.iter().enumerate() {
        if !level.0.is_finite() || !level.1.is_finite() || level.0 <= 0.0 || level.1 <= 0.0 {
            return Err("invalid level".into());
        }
        if index > 0
            && if bids {
                levels[index - 1].0 <= level.0
            } else {
                levels[index - 1].0 >= level.0
            }
        {
            return Err("unordered levels".into());
        }
    }
    Ok(levels)
}

fn depth(levels: &[(f64, f64)], count: usize) -> f64 {
    levels.iter().take(count).map(|value| value.1).sum()
}
fn imbalance(bid: f64, ask: f64) -> f64 {
    (bid - ask) / (bid + ask)
}
fn relative(current: f64, previous: f64) -> f64 {
    (current - previous) * 10_000.0 / previous
}
fn level_map(levels: &[(f64, f64)]) -> BTreeMap<String, (f64, f64)> {
    levels
        .iter()
        .map(|&(price, size)| (format!("{price:.10}"), (price, size)))
        .collect()
}
fn decimal(value: f64) -> Result<Decimal, BackfillExecutionError> {
    Decimal::from_str(&format!("{value:.10}")).map_err(backfill_support::source_error)
}
fn flow(
    current: &BTreeMap<String, (f64, f64)>,
    prior: &BTreeMap<String, (f64, f64)>,
) -> (f64, f64) {
    let keys: BTreeSet<_> = current.keys().chain(prior.keys()).collect();
    keys.into_iter()
        .fold((0.0, 0.0), |(replenishment, churn), key| {
            let (price, _) = current
                .get(key)
                .copied()
                .or_else(|| prior.get(key).copied())
                .expect("flow key came from one side");
            let delta = current.get(key).map_or(0.0, |value| value.1)
                - prior.get(key).map_or(0.0, |value| value.1);
            if delta > 0.0 {
                (replenishment + price * delta, churn)
            } else {
                (replenishment, churn + price * -delta)
            }
        })
}

fn materialize_features(
    states: &BTreeMap<DateTime<Utc>, State>,
    existing: &BTreeSet<DateTime<Utc>>,
    shard: &BackfillShard,
) -> Result<Vec<BinanceL2OneSecondFeature>, BackfillExecutionError> {
    let mut rows = Vec::new();
    for current in states.values().filter(|value| {
        value.second >= shard.range_start
            && value.second < shard.range_end
            && !existing.contains(&value.second)
    }) {
        let priors: Option<Vec<_>> = HORIZONS
            .iter()
            .map(|horizon| states.get(&(current.second - chrono::Duration::seconds(*horizon))))
            .collect();
        let Some(priors) = priors else { continue };
        let (bid_replenishment, bid_churn) = flow(&current.bids, &priors[0].bids);
        let (ask_replenishment, ask_churn) = flow(&current.asks, &priors[0].asks);
        let changes: Result<Vec<_>, BackfillExecutionError> = priors
            .iter()
            .map(|prior| {
                Ok((
                    decimal(relative(current.midpoint, prior.midpoint))?,
                    decimal(current.spread_bps - prior.spread_bps)?,
                    decimal(relative(
                        current.bid20 + current.ask20,
                        prior.bid20 + prior.ask20,
                    ))?,
                    decimal(current.imbalance20 - prior.imbalance20)?,
                ))
            })
            .collect();
        let c = changes?;
        rows.push(BinanceL2OneSecondFeature {
            symbol: SYMBOL.into(),
            second_start: current.second,
            source_event_timestamp: current.source,
            provider_received_at: current.received,
            available_at: current.received,
            source_update_id: current.update_id,
            feature_schema_version: FEATURE_SCHEMA.into(),
            quality_status: "qualified".into(),
            midpoint: decimal(current.midpoint)?,
            microprice: decimal(current.microprice)?,
            spread_bps: decimal(current.spread_bps)?,
            bid_depth_5: decimal(current.bid5)?,
            ask_depth_5: decimal(current.ask5)?,
            imbalance_5: decimal(current.imbalance5)?,
            bid_depth_10: decimal(current.bid10)?,
            ask_depth_10: decimal(current.ask10)?,
            imbalance_10: decimal(current.imbalance10)?,
            bid_depth_20: decimal(current.bid20)?,
            ask_depth_20: decimal(current.ask20)?,
            imbalance_20: decimal(current.imbalance20)?,
            bid_depth_slope_20: decimal(current.bid_slope20)?,
            ask_depth_slope_20: decimal(current.ask_slope20)?,
            bid_depth_concentration_20: decimal(current.bid_concentration20)?,
            ask_depth_concentration_20: decimal(current.ask_concentration20)?,
            bid_quote_replenishment_1s: decimal(bid_replenishment)?,
            ask_quote_replenishment_1s: decimal(ask_replenishment)?,
            bid_quote_churn_1s: decimal(bid_churn)?,
            ask_quote_churn_1s: decimal(ask_churn)?,
            midpoint_change_bps_1s: c[0].0,
            spread_bps_delta_1s: c[0].1,
            depth_20_change_bps_1s: c[0].2,
            imbalance_20_delta_1s: c[0].3,
            midpoint_change_bps_5s: c[1].0,
            spread_bps_delta_5s: c[1].1,
            depth_20_change_bps_5s: c[1].2,
            imbalance_20_delta_5s: c[1].3,
            midpoint_change_bps_15s: c[2].0,
            spread_bps_delta_15s: c[2].1,
            depth_20_change_bps_15s: c[2].2,
            imbalance_20_delta_15s: c[2].3,
            midpoint_change_bps_30s: c[3].0,
            spread_bps_delta_30s: c[3].1,
            depth_20_change_bps_30s: c[3].2,
            imbalance_20_delta_30s: c[3].3,
            midpoint_change_bps_60s: c[4].0,
            spread_bps_delta_60s: c[4].1,
            depth_20_change_bps_60s: c[4].2,
            imbalance_20_delta_60s: c[4].3,
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{BackfillWorkerStrategy, StrategyCapability};
    use chrono::TimeZone;

    #[test]
    fn contract_is_backfill_only_and_hourly_sharded() {
        let strategy = CoinapiBinanceSpotL2OneSecondFeaturesBackfill::new().unwrap();
        assert_eq!(
            strategy.descriptor().capabilities,
            vec![StrategyCapability::Backfill]
        );
        let start = Utc.with_ymd_and_hms(2026, 7, 2, 5, 30, 0).unwrap();
        let request =
            backfill_support::request(STRATEGY_KEY, start, start + chrono::Duration::hours(2));
        let shards = strategy
            .plan_shards(&strategy.validate_request(&request).unwrap())
            .unwrap();
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].range_end, start + chrono::Duration::hours(1));
        assert_eq!(shards[1].range_end, start + chrono::Duration::hours(2));
    }

    #[test]
    fn materialization_requires_contiguous_sixty_second_context() {
        let start = Utc.with_ymd_and_hms(2026, 7, 2, 5, 0, 0).unwrap();
        let snapshots = (0..=60)
            .map(|offset| Snapshot {
                symbol_id: COINAPI_SYMBOL.into(),
                time_exchange: (start + chrono::Duration::seconds(offset)).to_rfc3339(),
                time_coinapi: (start + chrono::Duration::seconds(offset)).to_rfc3339(),
                bids: (0..20)
                    .map(|level| Level {
                        price: 100.0 - f64::from(level),
                        size: 1.0,
                    })
                    .collect(),
                asks: (0..20)
                    .map(|level| Level {
                        price: 101.0 + f64::from(level),
                        size: 1.0,
                    })
                    .collect(),
            })
            .collect();
        let (states, rejected) = select_states(snapshots).unwrap();
        assert_eq!(rejected, 0);
        let shard = BackfillShard {
            shard_key: "fixture".into(),
            range_start: start + chrono::Duration::seconds(60),
            range_end: start + chrono::Duration::seconds(61),
            parameters: json!({}),
        };
        let rows = materialize_features(&states, &BTreeSet::new(), &shard).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].second_start, shard.range_start);
        assert_eq!(rows[0].feature_schema_version, FEATURE_SCHEMA);
    }
}
