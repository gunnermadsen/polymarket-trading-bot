use std::collections::BTreeSet;

use chrono::{DateTime, Duration, Utc};
use serde_json::{json, Value};

use crate::domain::{BackfillRequest, BackfillWorkerStrategy, StrategyCapability};

use super::{
    support::{
        parse_gamma_btc_interval_event, EXECUTION_SNAPSHOTS_BACKFILL_KEY,
        MARKET_CONTRACTS_BACKFILL_KEY, ORDERBOOK_EVENTS_BACKFILL_KEY, RESOLUTIONS_BACKFILL_KEY,
    },
    PolymarketBtcExecutionSnapshotsBackfill, PolymarketBtcMarketContractsBackfill,
    PolymarketBtcOrderbookEventsBackfill, PolymarketBtcResolutionsBackfill,
};

fn request(key: &str, start: DateTime<Utc>, end: DateTime<Utc>) -> BackfillRequest {
    serde_json::from_value(json!({
        "strategy_key": key,
        "range": {"start": start, "end": end},
        "parameters": {},
        "execution": {},
    }))
    .unwrap()
}

#[test]
fn four_strategies_have_unique_backfill_only_contracts() {
    let strategies: Vec<Box<dyn BackfillWorkerStrategy>> = vec![
        Box::new(PolymarketBtcMarketContractsBackfill::new().unwrap()),
        Box::new(PolymarketBtcResolutionsBackfill::new().unwrap()),
        Box::new(PolymarketBtcOrderbookEventsBackfill::new().unwrap()),
        Box::new(PolymarketBtcExecutionSnapshotsBackfill::new().unwrap()),
    ];
    let keys = strategies
        .iter()
        .map(|strategy| strategy.descriptor().strategy_key.to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(keys.len(), 4);
    assert_eq!(
        keys,
        BTreeSet::from([
            MARKET_CONTRACTS_BACKFILL_KEY.to_owned(),
            RESOLUTIONS_BACKFILL_KEY.to_owned(),
            ORDERBOOK_EVENTS_BACKFILL_KEY.to_owned(),
            EXECUTION_SNAPSHOTS_BACKFILL_KEY.to_owned(),
        ])
    );
    assert!(strategies.iter().all(|strategy| {
        strategy.descriptor().capabilities == vec![StrategyCapability::Backfill]
    }));
}

#[test]
fn one_hour_requests_produce_exactly_one_deterministic_shard() {
    let start: DateTime<Utc> = "2026-07-22T00:00:00Z".parse().unwrap();
    for strategy in [
        Box::new(PolymarketBtcMarketContractsBackfill::new().unwrap())
            as Box<dyn BackfillWorkerStrategy>,
        Box::new(PolymarketBtcResolutionsBackfill::new().unwrap()),
        Box::new(PolymarketBtcOrderbookEventsBackfill::new().unwrap()),
        Box::new(PolymarketBtcExecutionSnapshotsBackfill::new().unwrap()),
    ] {
        let key = strategy.descriptor().strategy_key.as_ref();
        let validated = strategy
            .validate_request(&request(key, start, start + Duration::hours(1)))
            .unwrap();
        let first = strategy.plan_shards(&validated).unwrap();
        let second = strategy.plan_shards(&validated).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].range_start, start);
        assert_eq!(first[0].range_end, start + Duration::hours(1));
    }
}

#[test]
fn pmxt_rejects_partial_hour_ranges() {
    let start: DateTime<Utc> = "2026-07-22T00:05:00Z".parse().unwrap();
    let strategy = PolymarketBtcOrderbookEventsBackfill::new().unwrap();
    let error = strategy
        .validate_request(&request(
            ORDERBOOK_EVENTS_BACKFILL_KEY,
            start,
            start + Duration::hours(1),
        ))
        .unwrap_err();
    assert_eq!(error.code, "range_alignment_invalid");
}

#[test]
fn original_gamma_contract_fixture_parses_without_contract_drift() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/polymarket/gamma_btc_five_minute_event_v1.json"
    ))
    .unwrap();
    let start: DateTime<Utc> = "2026-07-13T00:30:00Z".parse().unwrap();
    let market = parse_gamma_btc_interval_event(&fixture, start).unwrap();
    assert_eq!(market.window_start, start);
    assert_eq!(market.window_end, start + Duration::minutes(5));
    assert_ne!(market.up_token_id, market.down_token_id);
}
