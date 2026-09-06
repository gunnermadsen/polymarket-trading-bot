use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use sqlx::{Postgres, QueryBuilder, Transaction};
use uuid::Uuid;

pub const DIRECT_SOURCE: &str = "chainlink_data_streams";
pub const PMDATA_SOURCE: &str = "pmdata_chainlink_streams";

#[derive(Debug, Clone, Copy)]
pub enum ReferencePriceArtifact {
    Capture(Uuid),
    Backfill(Uuid),
}

pub struct ChainlinkReferencePriceWrite<'a> {
    pub feed_id: &'a str,
    pub source_timestamp: DateTime<Utc>,
    pub valid_from_timestamp: Option<DateTime<Utc>>,
    pub provider_available_at: Option<DateTime<Utc>>,
    pub received_at: DateTime<Utc>,
    pub price: Decimal,
    pub bid: Option<Decimal>,
    pub ask: Option<Decimal>,
    pub report_sha256: &'a str,
    pub payload_sha256: &'a str,
    pub artifact: ReferencePriceArtifact,
    pub expires_at: Option<DateTime<Utc>>,
    pub report_version: Option<&'a str>,
    pub source_date: Option<NaiveDate>,
    pub archive_row_number: Option<i64>,
    pub report_hash_kind: &'a str,
}

pub async fn insert_chainlink_reference_prices(
    transaction: &mut Transaction<'_, Postgres>,
    strategy_key: &str,
    writes: &[ChainlinkReferencePriceWrite<'_>],
) -> Result<Vec<(DateTime<Utc>, String)>, sqlx::Error> {
    insert(
        transaction,
        "market_data.chainlink_btcusd_reference_prices",
        DIRECT_SOURCE,
        strategy_key,
        writes,
    )
    .await
}

pub async fn insert_pmdata_chainlink_reference_prices(
    transaction: &mut Transaction<'_, Postgres>,
    strategy_key: &str,
    writes: &[ChainlinkReferencePriceWrite<'_>],
) -> Result<Vec<(DateTime<Utc>, String)>, sqlx::Error> {
    insert(
        transaction,
        "market_data.pmdata_chainlink_btcusd_reference_prices",
        PMDATA_SOURCE,
        strategy_key,
        writes,
    )
    .await
}

async fn insert(
    transaction: &mut Transaction<'_, Postgres>,
    table: &str,
    source: &str,
    strategy_key: &str,
    writes: &[ChainlinkReferencePriceWrite<'_>],
) -> Result<Vec<(DateTime<Utc>, String)>, sqlx::Error> {
    if writes.is_empty() {
        return Ok(Vec::new());
    }
    let mut query = QueryBuilder::<Postgres>::new(format!(
        "INSERT INTO {table} (source,feed_id,source_timestamp,valid_from_timestamp,provider_available_at,received_at,price,bid,ask,report_sha256,payload_sha256,strategy_key,capture_artifact_id,expires_at,report_version,source_date,archive_row_number,backfill_artifact_id,report_hash_kind) "
    ));
    query.push_values(writes, |mut row, write| {
        let (capture_artifact_id, backfill_artifact_id) = match write.artifact {
            ReferencePriceArtifact::Capture(id) => (Some(id), None),
            ReferencePriceArtifact::Backfill(id) => (None, Some(id)),
        };
        row.push_bind(source)
            .push_bind(write.feed_id)
            .push_bind(write.source_timestamp)
            .push_bind(write.valid_from_timestamp)
            .push_bind(write.provider_available_at)
            .push_bind(write.received_at)
            .push_bind(write.price)
            .push_bind(write.bid)
            .push_bind(write.ask)
            .push_bind(write.report_sha256)
            .push_bind(write.payload_sha256)
            .push_bind(strategy_key)
            .push_bind(capture_artifact_id)
            .push_bind(write.expires_at)
            .push_bind(write.report_version)
            .push_bind(write.source_date)
            .push_bind(write.archive_row_number)
            .push_bind(backfill_artifact_id)
            .push_bind(write.report_hash_kind);
    });
    query.push(
        " ON CONFLICT (feed_id,source_timestamp,report_sha256) DO NOTHING RETURNING source_timestamp,report_sha256::text",
    );
    query
        .build_query_as::<(DateTime<Utc>, String)>()
        .fetch_all(&mut **transaction)
        .await
}
