use std::{
    fs::{self, File},
    io::{Cursor, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use chrono::{DateTime, Datelike, Duration as ChronoDuration, NaiveDate, TimeZone, Utc};
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use reqwest::{Client, StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tokio::time::sleep;
use uuid::Uuid;
use zip::ZipArchive;

use crate::{
    domain::{
        BackfillContext, BackfillExecutionError, BackfillFailureKind, BackfillOutcome,
        BackfillRequest, BackfillShard, StrategyCapability, StrategyDescriptor,
        ValidatedBackfillRequest,
    },
    strategies::backfill_support as ledger,
};

use super::types::{CausalRow, EconomicDataset, FetchResult};

const CONTRACT_VERSION: i32 = 1;
const REQUEST_VERSION: i32 = 1;
const MAX_SHARDS: usize = 10_000;
const FRED_BASE_URL: &str = "https://api.stlouisfed.org/fred/series/observations";
const NY_FED_RATES_URL: &str = "https://markets.newyorkfed.org/api/rates/all/search.json";
const NY_FED_SOMA_URL: &str = "https://markets.newyorkfed.org/api/soma/summary.json";
const TREASURY_BASE_URL: &str = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service";

pub const FRED_SERIES: &[&str] = &[
    "DGS2",
    "DGS5",
    "DGS10",
    "DGS30",
    "DFII5",
    "DFII10",
    "DFII30",
    "T5YIE",
    "T10YIE",
    "DFF",
    "EFFR",
    "RRPONTSYD",
    "WTREGEN",
    "WRESBAL",
    "WALCL",
    "NFCI",
    "CPIAUCSL",
    "PAYEMS",
    "UNRATE",
];

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct EmptyParameters {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct FredParameters {
    pub series_ids: Vec<String>,
}

impl Default for FredParameters {
    fn default() -> Self {
        Self {
            series_ids: FRED_SERIES
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
        }
    }
}

pub fn descriptor(
    key: &'static str,
    name: &'static str,
    description: &'static str,
) -> Result<StrategyDescriptor, BackfillExecutionError> {
    let descriptor = StrategyDescriptor {
        strategy_key: Arc::from(key),
        name: Arc::from(name),
        description: Arc::from(description),
        capabilities: vec![StrategyCapability::Backfill],
        strategy_contract_version: CONTRACT_VERSION,
        request_schema_version: Some(REQUEST_VERSION),
        shardable: true,
        maximum_shards: MAX_SHARDS,
    };
    descriptor.validate()?;
    Ok(descriptor)
}

pub fn validate_request(
    descriptor: &StrategyDescriptor,
    request: &BackfillRequest,
    dataset: EconomicDataset,
) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
    if request.strategy_key != descriptor.strategy_key.as_ref() {
        return Err(BackfillExecutionError::invalid(
            "strategy_key_mismatch",
            "request strategy key does not match the economic-data strategy",
        ));
    }
    if request.range.end <= request.range.start || request.range.end > Utc::now() {
        return Err(BackfillExecutionError::invalid(
            "range_invalid",
            "range must be increasing and may not end in the future",
        ));
    }
    let parameters = match dataset {
        EconomicDataset::FredEconomicSeries => {
            let parameters: FredParameters = serde_json::from_value(request.parameters.clone())
                .map_err(|error| {
                    BackfillExecutionError::invalid(
                        "parameters_invalid",
                        format!("invalid FRED parameters: {error}"),
                    )
                })?;
            if parameters.series_ids.is_empty() {
                return Err(BackfillExecutionError::invalid(
                    "series_ids_empty",
                    "FRED series_ids must not be empty",
                ));
            }
            let mut unique = parameters.series_ids.clone();
            unique.sort();
            unique.dedup();
            if unique.len() != parameters.series_ids.len()
                || unique
                    .iter()
                    .any(|series| !FRED_SERIES.contains(&series.as_str()))
            {
                return Err(BackfillExecutionError::invalid(
                    "series_ids_invalid",
                    "FRED series_ids must be unique members of the canonical allowlist",
                ));
            }
            serde_json::to_value(parameters).map_err(|error| {
                BackfillExecutionError::invalid("parameters_invalid", error.to_string())
            })?
        }
        _ => {
            let parameters: EmptyParameters = serde_json::from_value(request.parameters.clone())
                .map_err(|error| {
                    BackfillExecutionError::invalid(
                        "parameters_invalid",
                        format!("strategy accepts no parameters: {error}"),
                    )
                })?;
            serde_json::to_value(parameters).map_err(|error| {
                BackfillExecutionError::invalid("parameters_invalid", error.to_string())
            })?
        }
    };
    request.execution.validate()?;
    Ok(ValidatedBackfillRequest {
        strategy_key: descriptor.strategy_key.clone(),
        strategy_contract_version: CONTRACT_VERSION,
        request_schema_version: REQUEST_VERSION,
        range_start: request.range.start,
        range_end: request.range.end,
        parameters,
        execution: request.execution.clone(),
    })
}

pub fn plan_year_shards(
    request: &ValidatedBackfillRequest,
    maximum: usize,
    dataset: EconomicDataset,
) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
    let series = match dataset {
        EconomicDataset::FredEconomicSeries => {
            serde_json::from_value::<FredParameters>(request.parameters.clone())
                .map_err(|error| {
                    BackfillExecutionError::invalid("parameters_invalid", error.to_string())
                })?
                .series_ids
        }
        _ => vec![String::new()],
    };
    let mut shards = Vec::new();
    for series_id in series {
        let mut cursor = request.range_start;
        while cursor < request.range_end {
            let next_year = Utc
                .with_ymd_and_hms(cursor.year() + 1, 1, 1, 0, 0, 0)
                .single()
                .ok_or_else(|| {
                    BackfillExecutionError::invalid(
                        "range_overflow",
                        "economic-data range overflowed",
                    )
                })?;
            let end = next_year.min(request.range_end);
            let shard_parameters = json!({"series_id": series_id});
            let prefix = if series_id.is_empty() {
                dataset.dataset().to_owned()
            } else {
                format!("{}-{series_id}", dataset.dataset())
            };
            shards.push(BackfillShard {
                shard_key: format!(
                    "{}-{}-{}",
                    prefix,
                    cursor.format("%Y%m%dT%H%M%SZ"),
                    end.format("%Y%m%dT%H%M%SZ")
                ),
                range_start: cursor,
                range_end: end,
                parameters: shard_parameters,
            });
            if shards.len() > maximum {
                return Err(BackfillExecutionError::invalid(
                    "too_many_shards",
                    format!("request exceeds {maximum} economic-data shards"),
                ));
            }
            cursor = end;
        }
    }
    Ok(shards)
}

pub async fn execute(
    context: BackfillContext,
    shard: BackfillShard,
    strategy_key: &str,
    dataset: EconomicDataset,
) -> Result<BackfillOutcome, BackfillExecutionError> {
    if context.shutdown.is_cancelled() {
        return Err(cancelled());
    }
    let series_id = shard
        .parameters
        .get("series_id")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            BackfillExecutionError::invalid("parameters_invalid", "shard omitted series_id")
        })?;
    let logical_key = format!(
        "{}:{}:{}:{}",
        dataset.dataset(),
        safe_component(series_id),
        shard.range_start,
        shard.range_end
    );
    if let Some(outcome) = ledger::completed_outcome(&context, strategy_key, &logical_key).await? {
        return Ok(outcome);
    }
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(120))
        .user_agent("capitonic-unified-ingester/1.0")
        .build()
        .map_err(source_error)?;
    let fetched = match dataset {
        EconomicDataset::FredEconomicSeries => fetch_fred(&client, &shard, series_id).await?,
        EconomicDataset::NewYorkFedReferenceRates | EconomicDataset::NewYorkFedSomaHoldings => {
            fetch_new_york_fed(&client, &shard, dataset).await?
        }
        EconomicDataset::CftcLegacyFutures | EconomicDataset::CftcTradersFinancialFutures => {
            fetch_cftc(&client, &shard, dataset).await?
        }
        EconomicDataset::TreasuryAuctions
        | EconomicDataset::TreasuryDebtToPenny
        | EconomicDataset::TreasuryDepositsWithdrawals
        | EconomicDataset::TreasuryOperatingCashBalance => {
            fetch_treasury(&client, &shard, dataset).await?
        }
    };
    if context.shutdown.is_cancelled() {
        return Err(cancelled());
    }
    if fetched.rows.is_empty() {
        return Err(BackfillExecutionError::new(
            BackfillFailureKind::Integrity,
            "source_rows_empty",
            "official source returned no rows for the requested shard",
        ));
    }
    let root = data_root()?;
    let published = publish(&root, context.job_id, dataset, series_id, &shard, &fetched)?;
    let artifact_id = ledger::create_artifact(
        &context,
        strategy_key,
        strategy_key,
        &logical_key,
        dataset.provider(),
        &fetched.source_url,
        &published.parquet_path,
    )
    .await?;
    let minimum = fetched.rows.iter().map(|row| row.event_at).min();
    let maximum = fetched.rows.iter().map(|row| row.event_at).max();
    let record_count = i64::try_from(fetched.rows.len())
        .map_err(|_| ledger::integrity("economic-data row count overflowed"))?;
    let mut tx = context.pool.begin().await.map_err(ledger::database_error)?;
    ledger::complete_artifact(
        &mut tx,
        &context,
        ledger::ArtifactCompletion {
            artifact_id,
            checksum: &published.parquet_sha256,
            byte_size: published.parquet_bytes,
            record_count,
            minimum,
            maximum,
            metadata: json!({
                "provider": dataset.provider(),
                "dataset": dataset.dataset(),
                "series_id": series_id,
                "source_url": fetched.source_url,
                "raw_path": published.raw_path,
                "raw_sha256": published.raw_sha256,
                "parquet_path": published.parquet_path,
                "records_verified": record_count,
            }),
        },
    )
    .await?;
    tx.commit().await.map_err(ledger::database_error)?;
    Ok(ledger::outcome(
        record_count,
        &shard,
        json!({
            "provider": dataset.provider(),
            "dataset": dataset.dataset(),
            "series_id": series_id,
            "records_verified": record_count,
            "durable_target": published.parquet_path,
        }),
    ))
}

async fn fetch_fred(
    client: &Client,
    shard: &BackfillShard,
    series_id: &str,
) -> Result<FetchResult, BackfillExecutionError> {
    let api_key = std::env::var("FRED_API_KEY")
        .ok()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            BackfillExecutionError::new(
                BackfillFailureKind::TransientSource,
                "fred_api_key_missing",
                "FRED_API_KEY is required for FRED backfills",
            )
        })?;
    let mut url = Url::parse(FRED_BASE_URL).map_err(source_error)?;
    let realtime_end = (shard.range_end + ChronoDuration::days(366)).min(Utc::now());
    url.query_pairs_mut()
        .append_pair("series_id", series_id)
        .append_pair("api_key", &api_key)
        .append_pair("file_type", "json")
        .append_pair("output_type", "4")
        .append_pair(
            "realtime_start",
            &shard.range_start.format("%Y-%m-%d").to_string(),
        )
        .append_pair("realtime_end", &realtime_end.format("%Y-%m-%d").to_string())
        .append_pair(
            "observation_start",
            &shard.range_start.format("%Y-%m-%d").to_string(),
        )
        .append_pair(
            "observation_end",
            &shard.range_end.format("%Y-%m-%d").to_string(),
        );
    sleep(Duration::from_millis(750)).await;
    let bytes = fetch_bytes(client, url.as_str(), 5).await?;
    let root: Value = serde_json::from_slice(&bytes).map_err(integrity_error)?;
    let observations = root
        .get("observations")
        .and_then(Value::as_array)
        .ok_or_else(|| integrity_error("FRED response omitted observations"))?;
    let mut rows = Vec::new();
    for value in observations {
        let event = parse_date_field(value, &["date"])?;
        let vintage = string_field(value, &["realtime_start"])
            .unwrap_or_else(|| event.format("%Y-%m-%d").to_string());
        let release = parse_date(&vintage)? + ChronoDuration::days(1);
        rows.push(CausalRow {
            event_at: event,
            released_at: release,
            available_at: release,
            revision: vintage,
            payload: value.clone(),
        });
    }
    Ok(FetchResult {
        source_url: redact_api_key(url),
        source_bytes: bytes,
        rows,
    })
}

async fn fetch_new_york_fed(
    client: &Client,
    shard: &BackfillShard,
    dataset: EconomicDataset,
) -> Result<FetchResult, BackfillExecutionError> {
    let base = match dataset {
        EconomicDataset::NewYorkFedReferenceRates => NY_FED_RATES_URL,
        EconomicDataset::NewYorkFedSomaHoldings => NY_FED_SOMA_URL,
        _ => return Err(integrity_error("invalid New York Fed dataset")),
    };
    let mut url = Url::parse(base).map_err(source_error)?;
    url.query_pairs_mut()
        .append_pair(
            "startDate",
            &shard.range_start.format("%m/%d/%Y").to_string(),
        )
        .append_pair("endDate", &shard.range_end.format("%m/%d/%Y").to_string());
    if matches!(dataset, EconomicDataset::NewYorkFedReferenceRates) {
        url.query_pairs_mut().append_pair("type", "rate");
    }
    let bytes = fetch_bytes(client, url.as_str(), 4).await?;
    let root: Value = serde_json::from_slice(&bytes).map_err(integrity_error)?;
    let values = first_object_array(&root)
        .ok_or_else(|| integrity_error("New York Fed response contained no record array"))?;
    let rows = values
        .iter()
        .filter_map(|value| {
            let event = parse_date_field(value, &["effectiveDate", "asOfDate", "date"]).ok()?;
            if event < shard.range_start || event >= shard.range_end {
                return None;
            }
            let available = if matches!(dataset, EconomicDataset::NewYorkFedSomaHoldings) {
                event + ChronoDuration::days(1) + ChronoDuration::hours(22)
            } else {
                event + ChronoDuration::days(1) + ChronoDuration::hours(13)
            };
            Some(CausalRow {
                event_at: event,
                released_at: available,
                available_at: available,
                revision: String::new(),
                payload: value.clone(),
            })
        })
        .collect();
    Ok(FetchResult {
        source_url: url.to_string(),
        source_bytes: bytes,
        rows,
    })
}

async fn fetch_treasury(
    client: &Client,
    shard: &BackfillShard,
    dataset: EconomicDataset,
) -> Result<FetchResult, BackfillExecutionError> {
    let endpoint = match dataset {
        EconomicDataset::TreasuryAuctions => "v1/accounting/od/auctions_query",
        EconomicDataset::TreasuryOperatingCashBalance => "v1/accounting/dts/operating_cash_balance",
        EconomicDataset::TreasuryDepositsWithdrawals => {
            "v1/accounting/dts/deposits_withdrawals_operating_cash"
        }
        EconomicDataset::TreasuryDebtToPenny => "v2/accounting/od/debt_to_penny",
        _ => return Err(integrity_error("invalid Treasury dataset")),
    };
    let base_url = Url::parse(&format!("{TREASURY_BASE_URL}/{endpoint}")).map_err(source_error)?;
    let filter = format!(
        "record_date:gte:{},record_date:lt:{}",
        shard.range_start.format("%Y-%m-%d"),
        shard.range_end.format("%Y-%m-%d")
    );
    let mut page = 1_u32;
    let mut pages = Vec::new();
    let mut values = Vec::new();
    loop {
        let mut url = base_url.clone();
        url.query_pairs_mut()
            .append_pair("filter", &filter)
            .append_pair("page[size]", "1000")
            .append_pair("page[number]", &page.to_string())
            .append_pair("sort", "record_date");
        sleep(Duration::from_millis(500)).await;
        let bytes = fetch_bytes(client, url.as_str(), 4).await?;
        let root: Value = serde_json::from_slice(&bytes).map_err(integrity_error)?;
        let data = root
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| integrity_error("Treasury response omitted data"))?;
        let count = data.len();
        values.extend(data.iter().cloned());
        pages.push(root);
        if count < 1000 {
            break;
        }
        page = page.checked_add(1).ok_or_else(|| {
            BackfillExecutionError::new(
                BackfillFailureKind::Integrity,
                "treasury_page_overflow",
                "Treasury pagination overflowed",
            )
        })?;
    }
    let source_bytes = serde_json::to_vec(&json!({"pages": pages})).map_err(integrity_error)?;
    let rows = values
        .into_iter()
        .filter_map(|payload| {
            let event = parse_date_field(&payload, &["record_date", "auction_date"]).ok()?;
            let available = event + ChronoDuration::days(1) + ChronoDuration::hours(22);
            Some(CausalRow {
                event_at: event,
                released_at: available,
                available_at: available,
                revision: string_field(&payload, &["record_date"]).unwrap_or_default(),
                payload,
            })
        })
        .collect();
    Ok(FetchResult {
        source_url: base_url.to_string(),
        source_bytes,
        rows,
    })
}

async fn fetch_cftc(
    client: &Client,
    shard: &BackfillShard,
    dataset: EconomicDataset,
) -> Result<FetchResult, BackfillExecutionError> {
    let (slug, separator) = match dataset {
        EconomicDataset::CftcLegacyFutures => ("deacot", ""),
        EconomicDataset::CftcTradersFinancialFutures => ("fut_fin_txt", "_"),
        _ => return Err(integrity_error("invalid CFTC dataset")),
    };
    let year = shard.range_start.year();
    let url = format!("https://www.cftc.gov/files/dea/history/{slug}{separator}{year}.zip");
    let bytes = fetch_bytes(client, &url, 4).await?;
    let mut archive = ZipArchive::new(Cursor::new(bytes.as_slice())).map_err(integrity_error)?;
    let mut csv_bytes = Vec::new();
    archive
        .by_index(0)
        .map_err(integrity_error)?
        .read_to_end(&mut csv_bytes)
        .map_err(integrity_error)?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(csv_bytes.as_slice());
    let headers = reader.headers().map_err(integrity_error)?.clone();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(integrity_error)?;
        let mut object = Map::new();
        for (header, value) in headers.iter().zip(record.iter()) {
            object.insert(
                header.trim().to_owned(),
                Value::String(value.trim().to_owned()),
            );
        }
        let payload = Value::Object(object);
        if !cftc_selected_contract(&payload) {
            continue;
        }
        let Ok(event) = parse_date_field(
            &payload,
            &[
                "Report_Date_as_YYYY-MM-DD",
                "As_of_Date_In_Form_YYMMDD",
                "As_of_Date_In_Form_YYYY-MM-DD",
                "As_of_Date_Form_YYYY-MM-DD",
            ],
        ) else {
            continue;
        };
        if event < shard.range_start || event >= shard.range_end {
            continue;
        }
        let available = event + ChronoDuration::days(3) + ChronoDuration::hours(22);
        rows.push(CausalRow {
            event_at: event,
            released_at: available,
            available_at: available,
            revision: String::new(),
            payload,
        });
    }
    Ok(FetchResult {
        source_url: url,
        source_bytes: bytes,
        rows,
    })
}

async fn fetch_bytes(
    client: &Client,
    url: &str,
    attempts: u32,
) -> Result<Vec<u8>, BackfillExecutionError> {
    let mut last_error = String::new();
    let mut last_status = None;
    for retry in 0..attempts {
        match client.get(url).send().await {
            Ok(response) if response.status().is_success() => {
                return response
                    .bytes()
                    .await
                    .map(|bytes| bytes.to_vec())
                    .map_err(source_error);
            }
            Ok(response) => {
                last_status = Some(response.status());
                last_error = format!("source returned HTTP {}", response.status());
            }
            Err(error) => last_error = format!("source request failed: {error}"),
        }
        sleep(Duration::from_secs(2_u64.pow(retry).min(30))).await;
    }
    let kind = if last_status == Some(StatusCode::TOO_MANY_REQUESTS) {
        BackfillFailureKind::RateLimited
    } else {
        BackfillFailureKind::TransientSource
    };
    Err(BackfillExecutionError::new(
        kind,
        "economic_source_request_failed",
        last_error,
    ))
}

struct PublishedArtifact {
    raw_path: String,
    raw_sha256: String,
    parquet_path: String,
    parquet_sha256: String,
    parquet_bytes: u64,
}

fn publish(
    root: &Path,
    job_id: Uuid,
    dataset: EconomicDataset,
    series_id: &str,
    shard: &BackfillShard,
    fetched: &FetchResult,
) -> Result<PublishedArtifact, BackfillExecutionError> {
    let raw_sha256 = sha256(&fetched.source_bytes);
    let raw_relative = PathBuf::from(format!(
        "raw/provider={}/dataset={}/series_id={}/year={}/{}.source",
        dataset.provider(),
        dataset.dataset(),
        safe_component(series_id),
        shard.range_start.year(),
        raw_sha256
    ));
    atomic_publish(root, &raw_relative, &fetched.source_bytes)?;
    let parquet_bytes = encode_parquet(dataset, series_id, &fetched.rows, root, job_id)?;
    let parquet_sha256 = sha256(&parquet_bytes);
    let parquet_relative = PathBuf::from(format!(
        "normalized/provider={}/dataset={}/series_id={}/year={}/{}_{}_{}.parquet",
        dataset.provider(),
        dataset.dataset(),
        safe_component(series_id),
        shard.range_start.year(),
        shard.range_start.format("%Y%m%d"),
        shard.range_end.format("%Y%m%d"),
        parquet_sha256
    ));
    atomic_publish(root, &parquet_relative, &parquet_bytes)?;
    Ok(PublishedArtifact {
        raw_path: path_string(&root.join(raw_relative))?,
        raw_sha256,
        parquet_path: path_string(&root.join(parquet_relative))?,
        parquet_sha256,
        parquet_bytes: u64::try_from(parquet_bytes.len())
            .map_err(|_| ledger::integrity("economic-data artifact size overflowed"))?,
    })
}

fn encode_parquet(
    dataset: EconomicDataset,
    series_id: &str,
    rows: &[CausalRow],
    root: &Path,
    job_id: Uuid,
) -> Result<Vec<u8>, BackfillExecutionError> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_at_ms", DataType::Int64, false),
        Field::new("released_at_ms", DataType::Int64, false),
        Field::new("available_at_ms", DataType::Int64, false),
        Field::new("provider", DataType::Utf8, false),
        Field::new("dataset", DataType::Utf8, false),
        Field::new("series_id", DataType::Utf8, false),
        Field::new("revision", DataType::Utf8, false),
        Field::new("payload_json", DataType::Utf8, false),
    ]));
    let payloads = rows
        .iter()
        .map(|row| serde_json::to_string(&row.payload))
        .collect::<Result<Vec<_>, _>>()
        .map_err(integrity_error)?;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(
                rows.iter()
                    .map(|row| row.event_at.timestamp_millis())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter()
                    .map(|row| row.released_at.timestamp_millis())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter()
                    .map(|row| row.available_at.timestamp_millis())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(vec![dataset.provider(); rows.len()])),
            Arc::new(StringArray::from(vec![dataset.dataset(); rows.len()])),
            Arc::new(StringArray::from(vec![series_id; rows.len()])),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.revision.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                payloads.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
        ],
    )
    .map_err(integrity_error)?;
    let staging = root.join(".staging");
    fs::create_dir_all(&staging).map_err(source_error)?;
    let temporary = staging.join(format!("{job_id}.parquet.tmp"));
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(6).map_err(integrity_error)?,
        ))
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(&temporary).map_err(source_error)?,
        schema,
        Some(properties),
    )
    .map_err(integrity_error)?;
    writer.write(&batch).map_err(integrity_error)?;
    writer.close().map_err(integrity_error)?;
    let bytes = fs::read(&temporary).map_err(source_error)?;
    fs::remove_file(&temporary).map_err(source_error)?;
    Ok(bytes)
}

fn atomic_publish(
    root: &Path,
    relative: &Path,
    bytes: &[u8],
) -> Result<(), BackfillExecutionError> {
    let final_path = root.join(relative);
    if final_path.exists() {
        let existing = fs::read(&final_path).map_err(source_error)?;
        if sha256(&existing) != sha256(bytes) {
            return Err(ledger::integrity(format!(
                "immutable lake object hash changed: {}",
                final_path.display()
            )));
        }
        return Ok(());
    }
    let parent = final_path
        .parent()
        .ok_or_else(|| ledger::integrity("lake object has no parent"))?;
    fs::create_dir_all(parent).map_err(source_error)?;
    let staging = parent.join(format!(
        ".{}.tmp-{}",
        final_path
            .file_name()
            .ok_or_else(|| ledger::integrity("lake object has no filename"))?
            .to_string_lossy(),
        Uuid::new_v4()
    ));
    let mut file = File::create(&staging).map_err(source_error)?;
    file.write_all(bytes).map_err(source_error)?;
    file.sync_all().map_err(source_error)?;
    fs::rename(staging, final_path).map_err(source_error)?;
    Ok(())
}

fn data_root() -> Result<PathBuf, BackfillExecutionError> {
    let root = PathBuf::from(
        std::env::var("INGESTER_FINANCIAL_DATA_ROOT")
            .unwrap_or_else(|_| "/var/lib/financial-data".to_owned()),
    );
    if !root.is_absolute() {
        return Err(BackfillExecutionError::invalid(
            "financial_data_root_invalid",
            "INGESTER_FINANCIAL_DATA_ROOT must be absolute",
        ));
    }
    Ok(root)
}

fn first_object_array(value: &Value) -> Option<&Vec<Value>> {
    match value {
        Value::Array(items) if items.iter().all(Value::is_object) => Some(items),
        Value::Object(map) => map.values().find_map(first_object_array),
        _ => None,
    }
}

fn cftc_selected_contract(value: &Value) -> bool {
    let name = string_field(value, &["Market_and_Exchange_Names"])
        .unwrap_or_default()
        .to_ascii_uppercase();
    [
        "BITCOIN",
        "U.S. TREASURY",
        "US TREASURY",
        "TREASURY BOND",
        "TREASURY NOTE",
        "FEDERAL FUNDS",
        "SOFR",
        "U.S. DOLLAR INDEX",
        "NASDAQ",
        "S&P 500",
        "GOLD",
    ]
    .iter()
    .any(|needle| name.contains(needle))
}

fn parse_date_field(
    value: &Value,
    names: &[&str],
) -> Result<DateTime<Utc>, BackfillExecutionError> {
    let text =
        string_field(value, names).ok_or_else(|| integrity_error("record omitted event date"))?;
    if let Ok(value) = DateTime::parse_from_rfc3339(&text) {
        return Ok(value.with_timezone(&Utc));
    }
    for format in ["%Y-%m-%d", "%y%m%d"] {
        if let Ok(date) = NaiveDate::parse_from_str(&text, format) {
            let midnight = date
                .and_hms_opt(0, 0, 0)
                .ok_or_else(|| integrity_error("event date could not form midnight"))?;
            return Ok(Utc.from_utc_datetime(&midnight));
        }
    }
    Err(integrity_error(format!("invalid event date {text}")))
}

fn string_field(value: &Value, names: &[&str]) -> Option<String> {
    let object = value.as_object()?;
    for name in names {
        if let Some(value) = object.get(*name).and_then(Value::as_str) {
            return Some(value.to_owned());
        }
        let normalized = normalized_field_name(name);
        if let Some(value) = object
            .iter()
            .find(|(key, _)| normalized_field_name(key) == normalized)
            .and_then(|(_, value)| value.as_str())
        {
            return Some(value.to_owned());
        }
    }
    None
}

fn normalized_field_name(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn parse_date(value: &str) -> Result<DateTime<Utc>, BackfillExecutionError> {
    let date = NaiveDate::parse_from_str(value, "%Y-%m-%d").map_err(integrity_error)?;
    let midnight = date
        .and_hms_opt(0, 0, 0)
        .ok_or_else(|| integrity_error("release date could not form midnight"))?;
    Ok(Utc.from_utc_datetime(&midnight))
}

fn redact_api_key(mut url: Url) -> String {
    let pairs = url
        .query_pairs()
        .filter(|(key, _)| key != "api_key")
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    url.set_query(None);
    url.query_pairs_mut().extend_pairs(pairs);
    url.to_string()
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn safe_component(value: &str) -> String {
    if value.is_empty() {
        return "all".to_owned();
    }
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn path_string(path: &Path) -> Result<String, BackfillExecutionError> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| ledger::integrity("artifact path was not UTF-8"))
}

fn source_error(error: impl std::fmt::Display) -> BackfillExecutionError {
    ledger::source_error(error)
}

fn integrity_error(error: impl std::fmt::Display) -> BackfillExecutionError {
    ledger::integrity(error.to_string())
}

fn cancelled() -> BackfillExecutionError {
    BackfillExecutionError::new(
        BackfillFailureKind::Cancelled,
        "backfill_cancelled",
        "economic-data backfill was cancelled",
    )
}

#[cfg(test)]
pub fn test_request(
    key: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    dataset: EconomicDataset,
) -> BackfillRequest {
    let parameters = if matches!(dataset, EconomicDataset::FredEconomicSeries) {
        json!({"series_ids": ["DGS10"]})
    } else {
        json!({})
    };
    BackfillRequest {
        strategy_key: key.to_owned(),
        range: crate::domain::BackfillRange { start, end },
        parameters,
        execution: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    #[test]
    fn year_shards_are_deterministic_and_clamped() {
        let start = Utc.with_ymd_and_hms(2024, 12, 15, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2025, 2, 1, 0, 0, 0).unwrap();
        let descriptor = descriptor("test_backfill", "test", "test").unwrap();
        let request = test_request(
            "test_backfill",
            start,
            end,
            EconomicDataset::TreasuryAuctions,
        );
        let validated =
            validate_request(&descriptor, &request, EconomicDataset::TreasuryAuctions).unwrap();
        let shards = plan_year_shards(
            &validated,
            descriptor.maximum_shards,
            EconomicDataset::TreasuryAuctions,
        )
        .unwrap();
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].range_start, start);
        assert_eq!(shards[1].range_end, end);
        assert_eq!(shards[0].range_end, shards[1].range_start);
    }

    #[test]
    fn fred_shards_each_allowed_series_and_rejects_unknown_series() {
        let start = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2025, 2, 1, 0, 0, 0).unwrap();
        let descriptor = descriptor("fred_test", "test", "test").unwrap();
        let mut request =
            test_request("fred_test", start, end, EconomicDataset::FredEconomicSeries);
        request.parameters = json!({"series_ids": ["DGS10", "DFF"]});
        let validated =
            validate_request(&descriptor, &request, EconomicDataset::FredEconomicSeries).unwrap();
        let shards = plan_year_shards(
            &validated,
            descriptor.maximum_shards,
            EconomicDataset::FredEconomicSeries,
        )
        .unwrap();
        assert_eq!(shards.len(), 2);
        request.parameters = json!({"series_ids": ["NOT_A_SERIES"]});
        assert!(
            validate_request(&descriptor, &request, EconomicDataset::FredEconomicSeries).is_err()
        );
    }

    #[test]
    fn dataset_parameters_reject_unknown_fields() {
        let start = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
        let descriptor = descriptor("test_backfill", "test", "test").unwrap();
        let mut request = test_request(
            "test_backfill",
            start,
            start + ChronoDuration::days(1),
            EconomicDataset::TreasuryAuctions,
        );
        request.parameters = json!({"dataset": "other"});
        assert!(
            validate_request(&descriptor, &request, EconomicDataset::TreasuryAuctions).is_err()
        );
    }

    #[test]
    fn date_fields_accept_cftc_formats() {
        assert_eq!(
            parse_date_field(
                &json!({"As_of_Date_In_Form_YYMMDD": "250801"}),
                &["As_of_Date_In_Form_YYMMDD"]
            )
            .unwrap(),
            Utc.with_ymd_and_hms(2025, 8, 1, 0, 0, 0).unwrap()
        );
    }

    #[test]
    fn api_keys_are_redacted_from_source_urls() {
        let url = Url::parse("https://example.test/data?series_id=DGS10&api_key=secret").unwrap();
        let redacted = redact_api_key(url);
        assert!(redacted.contains("series_id=DGS10"));
        assert!(!redacted.contains("secret"));
        assert!(!redacted.contains("api_key"));
    }
}
