use crate::{
    domain::{
        BackfillContext, BackfillExecutionError, BackfillRequest, BackfillShard,
        StrategyCapability, StrategyDescriptor, ValidatedBackfillRequest,
    },
    strategies::raw_archive::{self, RawObject},
};
use chrono::{Datelike, Utc};
use reqwest::Client;
use serde_json::{json, Value};
use sqlx::Row;
use std::sync::Arc;
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
    context: &BackfillContext,
    s: &BackfillShard,
) -> Result<Vec<RawObject>, BackfillExecutionError> {
    let rows=sqlx::query("SELECT market_id,yes_token_id,no_token_id FROM weather.temperature_markets WHERE event_date >= $1::date AND event_date < $2::date ORDER BY market_id LIMIT 1000").bind(s.range_start).bind(s.range_end).fetch_all(&context.pool).await.map_err(crate::strategies::backfill_support::database_error)?;
    let base = std::env::var("POLYMARKET_CLOB_BASE_URL")
        .unwrap_or_else(|_| "https://clob.polymarket.com".into());
    let mut out = vec![];
    for row in rows {
        let market: String = row
            .try_get("market_id")
            .map_err(crate::strategies::backfill_support::database_error)?;
        for token in [
            row.try_get::<String, _>("yes_token_id"),
            row.try_get::<String, _>("no_token_id"),
        ] {
            let token = token.map_err(crate::strategies::backfill_support::database_error)?;
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
                    s.range_end.timestamp()
                )
                .into(),
                media_type: "application/json",
                minimum: s.range_start,
                maximum: s.range_end,
            });
        }
    }
    Ok(out)
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
