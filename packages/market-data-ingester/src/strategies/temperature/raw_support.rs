use crate::{
    domain::{
        BackfillExecutionError, BackfillRequest, BackfillShard, StrategyCapability,
        StrategyDescriptor, ValidatedBackfillRequest,
    },
    strategies::raw_archive::{self, RawObject},
};
use chrono::{Datelike, Utc};
use reqwest::Client;
use serde_json::{json, Value};
use std::{collections::HashSet, sync::Arc};
pub struct Support {
    descriptor: StrategyDescriptor,
    pub client: Client,
}
impl Support {
    pub fn new(
        key: &'static str,
        name: &'static str,
        description: &'static str,
        maximum_shards: usize,
    ) -> Result<Self, BackfillExecutionError> {
        let descriptor = StrategyDescriptor {
            strategy_key: Arc::from(key),
            name: Arc::from(name),
            description: Arc::from(description),
            capabilities: vec![StrategyCapability::Backfill],
            strategy_contract_version: 1,
            request_schema_version: Some(1),
            shardable: true,
            maximum_shards,
        };
        descriptor.validate()?;
        Ok(Self {
            descriptor,
            client: raw_archive::client()?,
        })
    }
    pub fn descriptor(&self) -> &StrategyDescriptor {
        &self.descriptor
    }
    pub fn validate(
        &self,
        r: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
        if r.strategy_key != self.descriptor.strategy_key.as_ref() {
            return Err(BackfillExecutionError::invalid(
                "strategy_key_mismatch",
                "request strategy key did not match raw temperature strategy",
            ));
        }
        if r.range.end <= r.range.start || r.range.end > Utc::now() {
            return Err(BackfillExecutionError::invalid(
                "range_invalid",
                "range must be increasing and may not end in the future",
            ));
        }
        if !r.parameters.as_object().is_some_and(|v| v.is_empty()) {
            return Err(BackfillExecutionError::invalid(
                "parameters_invalid",
                "raw temperature strategies accept no parameters",
            ));
        }
        r.execution.validate()?;
        Ok(ValidatedBackfillRequest {
            strategy_key: self.descriptor.strategy_key.clone(),
            strategy_contract_version: 1,
            request_schema_version: 1,
            range_start: r.range.start,
            range_end: r.range.end,
            parameters: json!({}),
            execution: r.execution.clone(),
        })
    }
    pub fn hourly(
        &self,
        r: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        shards(
            r,
            self.descriptor.maximum_shards,
            chrono::Duration::hours(1),
        )
    }
    pub fn daily(
        &self,
        r: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        shards(r, self.descriptor.maximum_shards, chrono::Duration::days(1))
    }
}
fn shards(
    r: &ValidatedBackfillRequest,
    max: usize,
    width: chrono::Duration,
) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
    let mut c = r.range_start;
    let mut v = vec![];
    while c < r.range_end {
        let e = (c + width).min(r.range_end);
        v.push(BackfillShard {
            shard_key: format!("{}-{}", c.timestamp(), e.timestamp()),
            range_start: c,
            range_end: e,
            parameters: json!({}),
        });
        if v.len() > max {
            return Err(BackfillExecutionError::invalid(
                "too_many_shards",
                "request exceeds strategy shard limit",
            ));
        }
        c = e;
    }
    Ok(v)
}
pub async fn market_objects(
    client: &Client,
    s: &BackfillShard,
) -> Result<Vec<RawObject>, BackfillExecutionError> {
    let base = std::env::var("POLYMARKET_GAMMA_BASE_URL")
        .unwrap_or_else(|_| "https://gamma-api.polymarket.com".into());
    let mut out = vec![];
    for closed in [true, false] {
        let mut offset = 0;
        loop {
            let response = client
                .get(format!("{}/events", base.trim_end_matches('/')))
                .query(&[
                    ("closed", closed.to_string()),
                    ("limit", "100".into()),
                    ("offset", offset.to_string()),
                    ("series_id", "10005".into()),
                    ("end_date_min", s.range_start.to_rfc3339()),
                    ("end_date_max", s.range_end.to_rfc3339()),
                ])
                .send()
                .await
                .map_err(source)?
                .error_for_status()
                .map_err(source)?;
            let url = response.url().to_string();
            let bytes = response.bytes().await.map_err(source)?;
            let payload: Value = serde_json::from_slice(&bytes).map_err(invalid)?;
            let count = payload
                .as_array()
                .ok_or_else(|| invalid("Gamma response was not an array"))?
                .len();
            let stamp = s.range_start.format("%Y%m%dT%H%M%SZ");
            let path = format!(
                "polymarket/temperature-markets/{}/{:02}/{:02}/{stamp}-{}-{offset}.json",
                s.range_start.year(),
                s.range_start.month(),
                s.range_start.day(),
                u8::from(closed)
            );
            out.push(RawObject {
                logical_key: format!(
                    "gamma:temperature-events:{stamp}:{}:{offset}",
                    u8::from(closed)
                ),
                provider: "polymarket_gamma",
                source_uri: url,
                relative_path: path.into(),
                media_type: "application/json",
                minimum: s.range_start,
                maximum: s.range_end,
            });
            if count < 100 {
                break;
            }
            offset += 100;
        }
    }
    Ok(out)
}
pub async fn price_objects(
    client: &Client,
    s: &BackfillShard,
) -> Result<Vec<RawObject>, BackfillExecutionError> {
    let gamma = std::env::var("POLYMARKET_GAMMA_BASE_URL")
        .unwrap_or_else(|_| "https://gamma-api.polymarket.com".into());
    let response = client
        .get(format!("{}/events", gamma.trim_end_matches('/')))
        .query(&[
            ("closed", "true".to_owned()),
            ("limit", "100".to_owned()),
            ("offset", "0".to_owned()),
            ("series_id", "10005".to_owned()),
            ("end_date_min", s.range_start.to_rfc3339()),
            ("end_date_max", s.range_end.to_rfc3339()),
        ])
        .send()
        .await
        .map_err(source)?
        .error_for_status()
        .map_err(source)?;
    let events: Value = response.json().await.map_err(source)?;
    let base = std::env::var("POLYMARKET_CLOB_BASE_URL")
        .unwrap_or_else(|_| "https://clob.polymarket.com".into());
    let mut out = vec![];
    let events = events
        .as_array()
        .ok_or_else(|| invalid("Gamma response was not an array"))?;
    for event in events {
        for market in event
            .get("markets")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let market_id = market
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("Gamma market lacked an id"))?;
            let tokens = json_string_array(market.get("clobTokenIds"))?;
            if tokens.len() != 2 {
                return Err(invalid(
                    "Gamma temperature market did not expose two CLOB tokens",
                ));
            }
            for token in tokens {
                let uri = reqwest::Url::parse_with_params(
                    &format!("{}/prices-history", base.trim_end_matches('/')),
                    &[
                        ("market", token.as_str()),
                        ("startTs", &s.range_start.timestamp().to_string()),
                        ("endTs", &s.range_end.timestamp().to_string()),
                        ("interval", "all"),
                        ("fidelity", "1"),
                    ],
                )
                .map_err(invalid)?;
                out.push(RawObject {
                    logical_key: format!(
                        "clob:temperature-price-history:{token}:{}:{}",
                        s.range_start.timestamp(),
                        s.range_end.timestamp()
                    ),
                    provider: "polymarket_clob_price_history",
                    source_uri: uri.to_string(),
                    relative_path: format!(
                        "polymarket/temperature-prices/{}/{:02}/{:02}/{market}/{}-{}-{token}.json",
                        s.range_start.year(),
                        s.range_start.month(),
                        s.range_start.day(),
                        s.range_start.timestamp(),
                        s.range_end.timestamp(),
                        market = market_id
                    )
                    .into(),
                    media_type: "application/json",
                    minimum: s.range_start,
                    maximum: s.range_end,
                });
            }
        }
    }
    Ok(out)
}

fn json_string_array(value: Option<&Value>) -> Result<Vec<String>, BackfillExecutionError> {
    let value = value.ok_or_else(|| invalid("Gamma market lacked CLOB token IDs"))?;
    let parsed = match value {
        Value::Array(values) => values.clone(),
        Value::String(encoded) => serde_json::from_str(encoded).map_err(invalid)?,
        _ => return Err(invalid("Gamma CLOB token IDs were not an array")),
    };
    parsed
        .into_iter()
        .map(|value| {
            value
                .as_str()
                .filter(|token| !token.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| invalid("Gamma CLOB token ID was not a non-empty string"))
        })
        .collect()
}
pub fn pmxt_objects(s: &BackfillShard) -> Vec<RawObject> {
    let hour = s.range_start;
    let name = format!(
        "polymarket_orderbook_{}.parquet",
        hour.format("%Y-%m-%dT%H")
    );
    let base = std::env::var("POLYMARKET_PMXT_ARCHIVE_BASE_URL")
        .unwrap_or_else(|_| "https://r2v2.pmxt.dev".into());
    vec![RawObject {
        logical_key: format!(
            "pmxt:v2:polymarket_orderbook:{}",
            hour.format("%Y-%m-%dT%H")
        ),
        provider: "pmxt_v2",
        source_uri: format!("{}/{name}", base.trim_end_matches('/')),
        relative_path: format!(
            "pmxt/polymarket-orderbook/{}/{:02}/{:02}/{name}",
            hour.year(),
            hour.month(),
            hour.day()
        )
        .into(),
        media_type: "application/vnd.apache.parquet",
        minimum: s.range_start,
        maximum: s.range_end,
    }]
}

pub struct PmxtMarketScope {
    pub condition_ids: HashSet<String>,
    pub token_ids: HashSet<String>,
    pub market_ids: Vec<String>,
    pub gamma_uris: Vec<String>,
}

pub async fn pmxt_market_scope(
    client: &Client,
    shard: &BackfillShard,
) -> Result<PmxtMarketScope, BackfillExecutionError> {
    let base = std::env::var("POLYMARKET_GAMMA_BASE_URL")
        .unwrap_or_else(|_| "https://gamma-api.polymarket.com".into());
    let day = shard.range_start.date_naive();
    let day_start = day
        .and_hms_opt(0, 0, 0)
        .ok_or_else(|| invalid("invalid UTC day"))?
        .and_utc();
    let day_end = day_start + chrono::Duration::days(1);
    let mut condition_ids = HashSet::new();
    let mut token_ids = HashSet::new();
    let mut market_ids = Vec::new();
    let mut gamma_uris = Vec::new();
    for closed in [true, false] {
        let response = client
            .get(format!("{}/events", base.trim_end_matches('/')))
            .query(&[
                ("closed", closed.to_string()),
                ("limit", "100".to_owned()),
                ("offset", "0".to_owned()),
                ("series_id", "10005".to_owned()),
                ("end_date_min", day_start.to_rfc3339()),
                ("end_date_max", day_end.to_rfc3339()),
            ])
            .send()
            .await
            .map_err(source)?
            .error_for_status()
            .map_err(source)?;
        gamma_uris.push(response.url().to_string());
        let events: Value = response.json().await.map_err(source)?;
        for event in events
            .as_array()
            .ok_or_else(|| invalid("Gamma response was not an array"))?
        {
            let title = event.get("title").and_then(Value::as_str).unwrap_or("");
            if !title
                .to_ascii_lowercase()
                .contains("highest temperature in nyc")
            {
                continue;
            }
            for market in event
                .get("markets")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let resolution = market
                    .get("resolutionSource")
                    .or_else(|| event.get("resolutionSource"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if !resolution.contains("wunderground.com/history/daily")
                    || !resolution.contains("klga")
                {
                    continue;
                }
                let market_id = required_text(market, "id")?;
                let condition_id = required_text(market, "conditionId")?;
                let tokens = json_string_array(market.get("clobTokenIds"))?;
                if tokens.len() != 2 {
                    return Err(invalid(
                        "Gamma temperature market did not expose two CLOB tokens",
                    ));
                }
                if condition_ids.insert(condition_id) {
                    market_ids.push(market_id);
                }
                token_ids.extend(tokens);
            }
        }
    }
    market_ids.sort();
    market_ids.dedup();
    gamma_uris.sort();
    gamma_uris.dedup();
    if condition_ids.is_empty() || token_ids.is_empty() {
        return Err(invalid(
            "no authoritative NYC temperature markets were found",
        ));
    }
    Ok(PmxtMarketScope {
        condition_ids,
        token_ids,
        market_ids,
        gamma_uris,
    })
}

fn required_text(value: &Value, field: &str) -> Result<String, BackfillExecutionError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| invalid(format!("Gamma market lacked {field}")))
}
fn source(e: impl std::fmt::Display) -> BackfillExecutionError {
    BackfillExecutionError::new(
        crate::domain::BackfillFailureKind::TransientSource,
        "temperature_source",
        e.to_string(),
    )
}
fn invalid(e: impl std::fmt::Display) -> BackfillExecutionError {
    BackfillExecutionError::invalid("temperature_source_invalid", e.to_string())
}
