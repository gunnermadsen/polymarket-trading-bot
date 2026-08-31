use std::{
    env,
    fs::{self, File},
    io::{Cursor, Read, Write},
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use chrono::{DateTime, Datelike, Duration as ChronoDuration, NaiveDate, TimeZone, Utc};
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use reqwest::{Client, Url};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use sqlx::{postgres::PgPoolOptions, FromRow, PgPool};
use tokio::time::sleep;
use tracing::{error, info, warn};
use uuid::Uuid;
use zip::ZipArchive;

use crate::config::AppConfig;

const DEFAULT_START: &str = "2022-01-01T00:00:00Z";
const DEFAULT_END: &str = "2026-07-28T03:00:00Z";
const FRED_BASE_URL: &str = "https://api.stlouisfed.org/fred/series/observations";
const NY_FED_RATES_URL: &str = "https://markets.newyorkfed.org/api/rates/all/search.json";
const NY_FED_SOMA_URL: &str = "https://markets.newyorkfed.org/api/soma/summary.json";
const TREASURY_BASE_URL: &str = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service";

const FRED_SERIES: &[&str] = &[
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
    "BAMLH0A0HYM2",
    "BAMLC0A0CM",
    "NFCI",
    "CPIAUCSL",
    "PAYEMS",
    "UNRATE",
];

const TREASURY_DATASETS: &[(&str, &str)] = &[
    ("auctions", "v1/accounting/od/auctions_query"),
    (
        "operating_cash_balance",
        "v1/accounting/dts/operating_cash_balance",
    ),
    (
        "deposits_withdrawals",
        "v1/accounting/dts/deposits_withdrawals_operating_cash",
    ),
    ("debt_to_penny", "v2/accounting/od/debt_to_penny"),
];

const CFTC_REPORTS: &[(&str, &str)] = &[
    ("traders_financial_futures", "fut_fin_txt"),
    ("legacy_futures", "deacot"),
];

#[derive(Debug, Clone, FromRow)]
struct Job {
    job_id: Uuid,
    provider: String,
    dataset: String,
    series_id: String,
    range_start: DateTime<Utc>,
    range_end: DateTime<Utc>,
    attempt: i32,
    max_attempts: i32,
    lease_token: Uuid,
}

#[derive(Debug, Clone)]
struct CausalRow {
    event_at: DateTime<Utc>,
    released_at: DateTime<Utc>,
    available_at: DateTime<Utc>,
    revision: String,
    payload: Value,
}

#[derive(Debug)]
struct FetchResult {
    url: String,
    bytes: Vec<u8>,
    rows: Vec<CausalRow>,
}

#[derive(Clone)]
pub struct WorkerConfig {
    worker_id: String,
    lake_root: PathBuf,
    fred_api_key: Option<String>,
    poll_interval: Duration,
    lease_duration: Duration,
}

impl WorkerConfig {
    fn from_env() -> Result<Self> {
        let lake_root = PathBuf::from(env_or(
            "FINANCIAL_DATA_LAKE_ROOT",
            "/var/lib/global-financial-data",
        ));
        if !lake_root.is_absolute() {
            bail!("FINANCIAL_DATA_LAKE_ROOT must be absolute");
        }
        Ok(Self {
            worker_id: env_or("FINANCIAL_DATA_WORKER_ID", "financial-data-worker-1"),
            lake_root,
            fred_api_key: env::var("FRED_API_KEY")
                .ok()
                .filter(|value| !value.is_empty()),
            poll_interval: Duration::from_millis(env_u64("FINANCIAL_DATA_POLL_INTERVAL_MS", 1000)?),
            lease_duration: Duration::from_secs(env_u64(
                "FINANCIAL_DATA_LEASE_DURATION_SECS",
                300,
            )?),
        })
    }
}

pub async fn run_worker() -> Result<()> {
    let config = WorkerConfig::from_env()?;
    let app = AppConfig::from_env()?;
    let pool = PgPoolOptions::new()
        .max_connections(2)
        .connect(&app.postgres.database_url())
        .await
        .context("failed to connect financial-data worker to PostgreSQL")?;
    let client = Client::builder()
        .user_agent("capitonic-global-financial-data/1.0")
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(120))
        .build()?;
    fs::create_dir_all(config.lake_root.join("raw"))?;
    fs::create_dir_all(config.lake_root.join("normalized"))?;
    info!(worker_id = %config.worker_id, root = %config.lake_root.display(), "financial-data worker started");

    loop {
        match claim_job(&pool, &config).await {
            Ok(Some(job)) => {
                if let Err(err) = execute_job(&pool, &client, &config, &job).await {
                    warn!(job_id = %job.job_id, error = %err, "financial-data job failed");
                    fail_or_retry(&pool, &job, &err.to_string()).await?;
                }
            }
            Ok(None) => sleep(config.poll_interval).await,
            Err(err) => {
                error!(error = %err, "financial-data job claim failed");
                sleep(config.poll_interval).await;
            }
        }
    }
}

pub async fn enqueue_plan() -> Result<u64> {
    let app = AppConfig::from_env()?;
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect(&app.postgres.database_url())
        .await?;
    let start = parse_env_time("FINANCIAL_DATA_BACKFILL_START", DEFAULT_START)?;
    let end = parse_env_time("FINANCIAL_DATA_BACKFILL_END", DEFAULT_END)?;
    if start >= end {
        bail!("financial-data backfill start must precede end");
    }
    let mut inserted = 0_u64;

    for year in start.year()..=end.year() {
        let year_start = Utc
            .with_ymd_and_hms(year, 1, 1, 0, 0, 0)
            .single()
            .unwrap()
            .max(start);
        let year_end = Utc
            .with_ymd_and_hms(year + 1, 1, 1, 0, 0, 0)
            .single()
            .unwrap()
            .min(end);
        if year_start >= year_end {
            continue;
        }
        for series in FRED_SERIES {
            inserted += enqueue(
                &pool,
                "fred",
                "economic_series",
                series,
                year_start,
                year_end,
            )
            .await? as u64;
        }
        inserted += enqueue(
            &pool,
            "new_york_fed",
            "reference_rates",
            "all",
            year_start,
            year_end,
        )
        .await? as u64;
        inserted += enqueue(
            &pool,
            "new_york_fed",
            "soma_holdings",
            "summary",
            year_start,
            year_end,
        )
        .await? as u64;
        for (dataset, _) in TREASURY_DATASETS {
            inserted +=
                enqueue(&pool, "us_treasury", dataset, "", year_start, year_end).await? as u64;
        }
        for (dataset, _) in CFTC_REPORTS {
            inserted += enqueue(
                &pool,
                "cftc",
                dataset,
                "selected_contracts",
                year_start,
                year_end,
            )
            .await? as u64;
        }
    }
    info!(inserted, start = %start, end = %end, "financial-data backfill plan enqueued");
    Ok(inserted)
}

async fn enqueue(
    pool: &PgPool,
    provider: &str,
    dataset: &str,
    series: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<bool> {
    Ok(sqlx::query_scalar::<_, bool>(
        r#"WITH inserted AS (
          INSERT INTO financial_data.backfill_jobs
            (provider, dataset, series_id, range_start, range_end)
          VALUES ($1,$2,$3,$4,$5)
          ON CONFLICT (provider,dataset,series_id,range_start,range_end) DO UPDATE SET
            status='queued',attempt=0,next_attempt_at=now(),error_message=NULL,
            completed_at=NULL,updated_at=now()
          WHERE financial_data.backfill_jobs.provider='fred'
            AND financial_data.backfill_jobs.error_message LIKE 'FRED_API_KEY%'
            AND financial_data.backfill_jobs.status IN ('queued','failed')
          RETURNING TRUE
        ) SELECT COALESCE(bool_or(TRUE), FALSE) FROM inserted"#,
    )
    .bind(provider)
    .bind(dataset)
    .bind(series)
    .bind(start)
    .bind(end)
    .fetch_one(pool)
    .await?)
}

async fn claim_job(pool: &PgPool, config: &WorkerConfig) -> Result<Option<Job>> {
    let token = Uuid::new_v4();
    let seconds = i64::try_from(config.lease_duration.as_secs())?;
    let job = sqlx::query_as::<_, Job>(
        r#"WITH candidate AS (
          SELECT job_id FROM financial_data.backfill_jobs
          WHERE (status='queued' AND next_attempt_at <= now())
             OR (status='running' AND lease_expires_at < now())
          ORDER BY requested_at, job_id FOR UPDATE SKIP LOCKED LIMIT 1
        )
        UPDATE financial_data.backfill_jobs j SET
          status='running', attempt=j.attempt+1, worker_id=$1, lease_token=$2,
          lease_expires_at=now()+make_interval(secs => $3),
          started_at=COALESCE(j.started_at,now()), updated_at=now(), error_message=NULL
        FROM candidate WHERE j.job_id=candidate.job_id
        RETURNING j.job_id,j.provider,j.dataset,j.series_id,j.range_start,j.range_end,
                  j.attempt,j.max_attempts,j.lease_token"#,
    )
    .bind(&config.worker_id)
    .bind(token)
    .bind(seconds)
    .fetch_optional(pool)
    .await?;
    if let Some(ref value) = job {
        sqlx::query(
            r#"INSERT INTO financial_data.worker_status(worker_id,state,current_job_id,heartbeat_at)
               VALUES($1,'running',$2,now()) ON CONFLICT(worker_id) DO UPDATE SET
               state='running',current_job_id=EXCLUDED.current_job_id,heartbeat_at=now()"#,
        )
        .bind(&config.worker_id)
        .bind(value.job_id)
        .execute(pool)
        .await?;
    }
    Ok(job)
}

async fn execute_job(
    pool: &PgPool,
    client: &Client,
    config: &WorkerConfig,
    job: &Job,
) -> Result<()> {
    let fetched = match job.provider.as_str() {
        "fred" => fetch_fred(client, config, job).await?,
        "new_york_fed" => fetch_new_york_fed(client, job).await?,
        "us_treasury" => fetch_treasury(client, job).await?,
        "cftc" => fetch_cftc(client, job).await?,
        other => bail!("unsupported financial-data provider {other}"),
    };
    if fetched.rows.is_empty() {
        bail!("official source returned no rows for requested range");
    }
    let raw = publish_raw(&config.lake_root, job, &fetched)?;
    let parquet = publish_parquet(&config.lake_root, job, &fetched.rows)?;
    let minimum = fetched.rows.iter().map(|row| row.event_at).min();
    let maximum = fetched.rows.iter().map(|row| row.event_at).max();
    let mut tx = pool.begin().await?;
    sqlx::query(
        r#"INSERT INTO financial_data.source_artifacts
           (job_id,source_url,relative_path,sha256,byte_size) VALUES($1,$2,$3,$4,$5)
           ON CONFLICT(relative_path) DO NOTHING"#,
    )
    .bind(job.job_id)
    .bind(&fetched.url)
    .bind(&raw.0)
    .bind(&raw.1)
    .bind(raw.2)
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        r#"INSERT INTO financial_data.parquet_objects
           (job_id,provider,dataset,series_id,relative_path,sha256,byte_size,row_count,minimum_event_at,maximum_event_at)
           VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) ON CONFLICT(relative_path) DO NOTHING"#,
    ).bind(job.job_id).bind(&job.provider).bind(&job.dataset).bind(&job.series_id)
     .bind(&parquet.0).bind(&parquet.1).bind(parquet.2).bind(i64::try_from(fetched.rows.len())?)
     .bind(minimum).bind(maximum).execute(&mut *tx).await?;
    sqlx::query(
        r#"UPDATE financial_data.backfill_jobs SET status='completed',rows_written=$3,
           completed_at=now(),updated_at=now(),lease_token=NULL,lease_expires_at=NULL
           WHERE job_id=$1 AND lease_token=$2"#,
    )
    .bind(job.job_id)
    .bind(job.lease_token)
    .bind(i64::try_from(fetched.rows.len())?)
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        r#"UPDATE financial_data.worker_status SET state='idle',current_job_id=NULL,
           jobs_completed=jobs_completed+1,rows_written=rows_written+$2,heartbeat_at=now()
           WHERE worker_id=$1"#,
    )
    .bind(&config.worker_id)
    .bind(i64::try_from(fetched.rows.len())?)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    info!(job_id=%job.job_id, provider=%job.provider, dataset=%job.dataset, rows=fetched.rows.len(), "financial-data job completed");
    Ok(())
}

async fn fail_or_retry(pool: &PgPool, job: &Job, message: &str) -> Result<()> {
    if message.contains("FRED_API_KEY") {
        sqlx::query(
            r#"UPDATE financial_data.backfill_jobs SET status='queued',attempt=GREATEST(0,attempt-1),
               error_message=$3,next_attempt_at=now()+INTERVAL '1 hour',lease_token=NULL,
               lease_expires_at=NULL,updated_at=now()
               WHERE job_id=$1 AND lease_token=$2"#,
        )
        .bind(job.job_id)
        .bind(job.lease_token)
        .bind(message)
        .execute(pool)
        .await?;
        return Ok(());
    }
    let terminal = job.attempt >= job.max_attempts;
    let status = if terminal { "failed" } else { "queued" };
    sqlx::query(
        r#"UPDATE financial_data.backfill_jobs SET status=$3,error_message=$4,
           next_attempt_at=now()+make_interval(secs => LEAST(3600, 5 * (1 << LEAST(attempt,9)))),
           lease_token=NULL,lease_expires_at=NULL,updated_at=now(),
           completed_at=CASE WHEN $3='failed' THEN now() ELSE NULL END
           WHERE job_id=$1 AND lease_token=$2"#,
    )
    .bind(job.job_id)
    .bind(job.lease_token)
    .bind(status)
    .bind(message)
    .execute(pool)
    .await?;
    Ok(())
}

async fn fetch_fred(client: &Client, config: &WorkerConfig, job: &Job) -> Result<FetchResult> {
    let key = config
        .fred_api_key
        .as_deref()
        .context("FRED_API_KEY is required for FRED jobs")?;
    let mut url = Url::parse(FRED_BASE_URL)?;
    url.query_pairs_mut()
        .append_pair("series_id", &job.series_id)
        .append_pair("api_key", key)
        .append_pair("file_type", "json")
        .append_pair("output_type", "4")
        .append_pair(
            "observation_start",
            &job.range_start.format("%Y-%m-%d").to_string(),
        )
        .append_pair(
            "observation_end",
            &job.range_end.format("%Y-%m-%d").to_string(),
        );
    let bytes = fetch_bytes(client, url.as_str()).await?;
    let root: Value = serde_json::from_slice(&bytes)?;
    let observations = root
        .get("observations")
        .and_then(Value::as_array)
        .context("FRED response omitted observations")?;
    let mut rows = Vec::new();
    for value in observations {
        let event = parse_date_field(value, &["date"])?;
        let vintage = string_field(value, &["realtime_start"])
            .unwrap_or_else(|| event.format("%Y-%m-%d").to_string());
        let release_date = parse_date(&vintage)?;
        rows.push(CausalRow {
            event_at: event,
            released_at: release_date + ChronoDuration::days(1),
            available_at: release_date + ChronoDuration::days(1),
            revision: vintage,
            payload: value.clone(),
        });
    }
    Ok(FetchResult {
        url: redact_api_key(url),
        bytes,
        rows,
    })
}

async fn fetch_new_york_fed(client: &Client, job: &Job) -> Result<FetchResult> {
    let base = if job.dataset == "reference_rates" {
        NY_FED_RATES_URL
    } else {
        NY_FED_SOMA_URL
    };
    let mut url = Url::parse(base)?;
    url.query_pairs_mut()
        .append_pair("startDate", &job.range_start.format("%m/%d/%Y").to_string())
        .append_pair("endDate", &job.range_end.format("%m/%d/%Y").to_string());
    if job.dataset == "reference_rates" {
        url.query_pairs_mut().append_pair("type", "rate");
    }
    let bytes = fetch_bytes(client, url.as_str()).await?;
    let root: Value = serde_json::from_slice(&bytes)?;
    let values =
        first_object_array(&root).context("New York Fed response contained no record array")?;
    let rows = values
        .iter()
        .filter_map(|value| {
            let event = parse_date_field(value, &["effectiveDate", "asOfDate", "date"]).ok()?;
            if event < job.range_start || event >= job.range_end {
                return None;
            }
            let available = if job.dataset == "soma_holdings" {
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
        url: url.to_string(),
        bytes,
        rows,
    })
}

async fn fetch_treasury(client: &Client, job: &Job) -> Result<FetchResult> {
    let endpoint = TREASURY_DATASETS
        .iter()
        .find(|(name, _)| *name == job.dataset)
        .map(|(_, path)| *path)
        .with_context(|| format!("unsupported Treasury dataset {}", job.dataset))?;
    let mut url = Url::parse(&format!("{TREASURY_BASE_URL}/{endpoint}"))?;
    url.query_pairs_mut()
        .append_pair(
            "filter",
            &format!(
                "record_date:gte:{},record_date:lt:{}",
                job.range_start.format("%Y-%m-%d"),
                job.range_end.format("%Y-%m-%d")
            ),
        )
        .append_pair("page[size]", "10000")
        .append_pair("sort", "record_date");
    let bytes = fetch_bytes(client, url.as_str()).await?;
    let root: Value = serde_json::from_slice(&bytes)?;
    let values = root
        .get("data")
        .and_then(Value::as_array)
        .context("Treasury response omitted data")?;
    let rows = values
        .iter()
        .filter_map(|value| {
            let event = parse_date_field(value, &["record_date", "auction_date"]).ok()?;
            let available = event + ChronoDuration::days(1) + ChronoDuration::hours(22);
            Some(CausalRow {
                event_at: event,
                released_at: available,
                available_at: available,
                revision: string_field(value, &["record_date"]).unwrap_or_default(),
                payload: value.clone(),
            })
        })
        .collect();
    Ok(FetchResult {
        url: url.to_string(),
        bytes,
        rows,
    })
}

async fn fetch_cftc(client: &Client, job: &Job) -> Result<FetchResult> {
    let slug = CFTC_REPORTS
        .iter()
        .find(|(name, _)| *name == job.dataset)
        .map(|(_, slug)| *slug)
        .with_context(|| format!("unsupported CFTC dataset {}", job.dataset))?;
    let year = job.range_start.year();
    let separator = if slug == "deacot" { "" } else { "_" };
    let url = format!("https://www.cftc.gov/files/dea/history/{slug}{separator}{year}.zip");
    let bytes = fetch_bytes(client, &url).await?;
    let mut archive = ZipArchive::new(Cursor::new(bytes.as_slice()))?;
    let mut csv_bytes = Vec::new();
    archive.by_index(0)?.read_to_end(&mut csv_bytes)?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(csv_bytes.as_slice());
    let headers = reader.headers()?.clone();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record?;
        let mut object = Map::new();
        for (header, value) in headers.iter().zip(record.iter()) {
            object.insert(
                header.trim().to_string(),
                Value::String(value.trim().to_string()),
            );
        }
        let payload = Value::Object(object);
        if !cftc_selected_contract(&payload) {
            continue;
        }
        let event = match parse_date_field(
            &payload,
            &[
                "Report_Date_as_YYYY-MM-DD",
                "As_of_Date_In_Form_YYMMDD",
                "As_of_Date_Form_YYYY-MM-DD",
            ],
        ) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if event < job.range_start || event >= job.range_end {
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
    Ok(FetchResult { url, bytes, rows })
}

async fn fetch_bytes(client: &Client, url: &str) -> Result<Vec<u8>> {
    let response = client.get(url).send().await?.error_for_status()?;
    Ok(response.bytes().await?.to_vec())
}

fn publish_raw(root: &Path, job: &Job, fetched: &FetchResult) -> Result<(String, String, i64)> {
    let hash = sha256(&fetched.bytes);
    let relative = PathBuf::from(format!(
        "raw/provider={}/dataset={}/series_id={}/year={}/{}.source",
        job.provider,
        job.dataset,
        safe_component(&job.series_id),
        job.range_start.year(),
        hash
    ));
    atomic_publish(root, &relative, &fetched.bytes)?;
    Ok((
        path_string(&relative)?,
        hash,
        i64::try_from(fetched.bytes.len())?,
    ))
}

fn publish_parquet(root: &Path, job: &Job, rows: &[CausalRow]) -> Result<(String, String, i64)> {
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
        .collect::<serde_json::Result<Vec<_>>>()?;
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
            Arc::new(StringArray::from(vec![job.provider.as_str(); rows.len()])),
            Arc::new(StringArray::from(vec![job.dataset.as_str(); rows.len()])),
            Arc::new(StringArray::from(vec![job.series_id.as_str(); rows.len()])),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.revision.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                payloads.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
        ],
    )?;
    let mut temporary = root.join(".staging");
    fs::create_dir_all(&temporary)?;
    temporary.push(format!("{}.parquet.tmp", job.job_id));
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(6)?))
        .build();
    let mut writer = ArrowWriter::try_new(File::create(&temporary)?, schema, Some(properties))?;
    writer.write(&batch)?;
    writer.close()?;
    let bytes = fs::read(&temporary)?;
    let hash = sha256(&bytes);
    let relative = PathBuf::from(format!(
        "normalized/provider={}/dataset={}/series_id={}/year={}/{}_{}_{}.parquet",
        job.provider,
        job.dataset,
        safe_component(&job.series_id),
        job.range_start.year(),
        job.range_start.format("%Y%m%d"),
        job.range_end.format("%Y%m%d"),
        hash
    ));
    atomic_publish(root, &relative, &bytes)?;
    fs::remove_file(temporary)?;
    Ok((path_string(&relative)?, hash, i64::try_from(bytes.len())?))
}

fn atomic_publish(root: &Path, relative: &Path, bytes: &[u8]) -> Result<()> {
    let final_path = root.join(relative);
    if final_path.exists() {
        if sha256(&fs::read(&final_path)?) != sha256(bytes) {
            bail!(
                "immutable lake object hash changed: {}",
                final_path.display()
            );
        }
        return Ok(());
    }
    let parent = final_path.parent().context("lake object has no parent")?;
    fs::create_dir_all(parent)?;
    let staging = parent.join(format!(
        ".{}.tmp-{}",
        final_path.file_name().unwrap().to_string_lossy(),
        Uuid::new_v4()
    ));
    let mut file = File::create(&staging)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(staging, final_path)?;
    Ok(())
}

fn first_object_array(value: &Value) -> Option<&Vec<Value>> {
    match value {
        Value::Array(items) if items.iter().all(Value::is_object) => Some(items),
        Value::Object(map) => map.values().find_map(first_object_array),
        _ => None,
    }
}

fn cftc_selected_contract(value: &Value) -> bool {
    let name = string_field(
        value,
        &["Market_and_Exchange_Names", "Market_and_Exchange_Names"],
    )
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

fn parse_date_field(value: &Value, names: &[&str]) -> Result<DateTime<Utc>> {
    let text = string_field(value, names).context("record omitted event date")?;
    if let Ok(value) = DateTime::parse_from_rfc3339(&text) {
        return Ok(value.with_timezone(&Utc));
    }
    if let Ok(date) = NaiveDate::parse_from_str(&text, "%Y-%m-%d") {
        return Ok(Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).unwrap()));
    }
    if let Ok(date) = NaiveDate::parse_from_str(&text, "%y%m%d") {
        return Ok(Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).unwrap()));
    }
    bail!("invalid event date {text}")
}

fn string_field(value: &Value, names: &[&str]) -> Option<String> {
    let object = value.as_object()?;
    for name in names {
        if let Some(value) = object.get(*name).and_then(Value::as_str) {
            return Some(value.to_string());
        }
        if let Some(value) = object
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .and_then(|(_, value)| value.as_str())
        {
            return Some(value.to_string());
        }
    }
    None
}

fn parse_date(value: &str) -> Result<DateTime<Utc>> {
    let date = NaiveDate::parse_from_str(value, "%Y-%m-%d")?;
    Ok(Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).unwrap()))
}

fn parse_env_time(key: &str, default: &str) -> Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(&env_or(key, default))?.with_timezone(&Utc))
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
        "all".to_string()
    } else {
        value
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }
}
fn path_string(path: &Path) -> Result<String> {
    Ok(path
        .to_str()
        .context("lake path was not UTF-8")?
        .to_string())
}
fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}
fn env_u64(key: &str, default: u64) -> Result<u64> {
    Ok(u64::from_str(&env_or(key, &default.to_string()))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cftc_release_is_after_report_date() {
        let event = parse_date("2026-08-25").unwrap();
        let available = event + ChronoDuration::days(3) + ChronoDuration::hours(22);
        assert!(available > event);
    }

    #[test]
    fn lake_components_are_path_safe() {
        assert_eq!(safe_component("10Y/real"), "10Y_real");
    }

    #[test]
    fn filters_cftc_contracts_to_training_scope() {
        let selected = serde_json::json!({"Market_and_Exchange_Names":"BITCOIN - CHICAGO MERCANTILE EXCHANGE"});
        let excluded =
            serde_json::json!({"Market_and_Exchange_Names":"CORN - CHICAGO BOARD OF TRADE"});
        assert!(cftc_selected_contract(&selected));
        assert!(!cftc_selected_contract(&excluded));
    }
}
