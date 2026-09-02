use crate::{
    domain::{
        BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
        BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
    },
    strategies::{
        backfill_support, raw_archive,
        temperature::pmxt_filter,
        temperature::raw_support::{self, Support},
    },
};
use async_trait::async_trait;
use chrono::Datelike;
use serde_json::json;
use std::path::PathBuf;
use tokio::fs;
pub const STRATEGY_KEY: &str = "pmxt_polymarket_orderbook_archives_backfill";
pub struct PmxtPolymarketOrderbookArchivesBackfill {
    support: Support,
}
impl PmxtPolymarketOrderbookArchivesBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        Ok(Self {
            support: Support::new(
                STRATEGY_KEY,
                "PMXT NYC temperature orderbook archives",
                "Stores filtered PMXT orderbook rows for authoritative NYC temperature markets",
                17520,
            )?,
        })
    }
}
#[async_trait]
impl BackfillWorkerStrategy for PmxtPolymarketOrderbookArchivesBackfill {
    fn descriptor(&self) -> &StrategyDescriptor {
        self.support.descriptor()
    }
    fn validate_request(
        &self,
        r: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
        self.support.validate(r)
    }
    fn plan_shards(
        &self,
        r: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        self.support.hourly(r)
    }
    async fn execute_backfill(
        &self,
        c: BackfillContext,
        s: BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        let object = raw_support::pmxt_objects(&s)
            .into_iter()
            .next()
            .ok_or_else(|| {
                BackfillExecutionError::invalid(
                    "pmxt_source_missing",
                    "PMXT source was not planned",
                )
            })?;
        let logical_key = format!(
            "pmxt:v2:nyc_temperature_orderbooks:v1:{}",
            s.range_start.format("%Y-%m-%dT%H")
        );
        if let Some(outcome) =
            backfill_support::completed_outcome(&c, STRATEGY_KEY, &logical_key).await?
        {
            return Ok(outcome);
        }
        let scope = raw_support::pmxt_market_scope(&self.support.client, &s).await?;
        let root = PathBuf::from(
            std::env::var("INGESTER_WEATHER_CURATED_ROOT")
                .unwrap_or_else(|_| "/var/lib/weather/curated".into()),
        );
        if !root.is_absolute() {
            return Err(BackfillExecutionError::invalid(
                "pmxt_output_root_invalid",
                "PMXT curated root must be absolute",
            ));
        }
        let directory = root.join(format!(
            "pmxt/nyc-temperature/{}/{:02}/{:02}",
            s.range_start.year(),
            s.range_start.month(),
            s.range_start.day()
        ));
        fs::create_dir_all(&directory).await.map_err(storage)?;
        let name = format!(
            "nyc_temperature_orderbooks_{}.parquet",
            s.range_start.format("%Y-%m-%dT%H")
        );
        let final_path = directory.join(&name);
        let output_partial = directory.join(format!(".{name}.{}.partial", c.lease_token));
        let source_directory = root.join(".tmp/pmxt");
        fs::create_dir_all(&source_directory)
            .await
            .map_err(storage)?;
        let source_path = source_directory.join(format!("{}.source.parquet", c.lease_token));
        let _ = fs::remove_file(&source_path).await;
        let _ = fs::remove_file(&output_partial).await;

        let (source_checksum, source_bytes) =
            raw_archive::download(&c, &self.support.client, &object, &source_path).await?;
        let filter_source = source_path.clone();
        let filter_output = output_partial.clone();
        let start = s.range_start;
        let end = s.range_end;
        let condition_ids = scope.condition_ids.clone();
        let token_ids = scope.token_ids.clone();
        let filtered = tokio::task::spawn_blocking(move || {
            pmxt_filter::filter_archive(
                &filter_source,
                &filter_output,
                condition_ids,
                token_ids,
                start,
                end,
            )
        })
        .await
        .map_err(storage)?;
        let _ = fs::remove_file(&source_path).await;
        let filtered = match filtered {
            Ok(value) => value,
            Err(error) => {
                let _ = fs::remove_file(&output_partial).await;
                return Err(error);
            }
        };
        if filtered.records == 0 {
            let _ = fs::remove_file(&output_partial).await;
            return Err(BackfillExecutionError::new(
                crate::domain::BackfillFailureKind::Integrity,
                "pmxt_temperature_rows_empty",
                "PMXT archive contained no rows for the authoritative NYC temperature markets",
            ));
        }
        fs::rename(&output_partial, &final_path)
            .await
            .map_err(storage)?;
        let artifact_id = backfill_support::create_artifact(
            &c,
            STRATEGY_KEY,
            STRATEGY_KEY,
            &logical_key,
            object.provider,
            &object.source_uri,
            final_path.to_string_lossy().as_ref(),
        )
        .await?;
        let mut tx = c
            .pool
            .begin()
            .await
            .map_err(backfill_support::database_error)?;
        backfill_support::complete_artifact(
            &mut tx,
            &c,
            backfill_support::ArtifactCompletion {
                artifact_id,
                checksum: &filtered.checksum,
                byte_size: filtered.bytes,
                record_count: filtered.records,
                minimum: filtered.minimum,
                maximum: filtered.maximum,
                metadata: json!({
                    "media_type":"application/vnd.apache.parquet",
                    "dataset":"nyc_temperature_orderbooks",
                    "dataset_version":1,
                    "source_sha256":source_checksum,
                    "source_byte_size":source_bytes,
                    "source_uri":object.source_uri,
                    "gamma_uris":scope.gamma_uris,
                    "market_ids":scope.market_ids,
                    "condition_ids":scope.condition_ids,
                    "token_ids":scope.token_ids,
                    "durable_target":final_path,
                }),
            },
        )
        .await?;
        tx.commit()
            .await
            .map_err(backfill_support::database_error)?;
        Ok(BackfillOutcome {
            records_verified: filtered.records,
            verified_coverage: json!({
                "minimum_source_timestamp":filtered.minimum,
                "maximum_source_timestamp":filtered.maximum,
                "records_verified":filtered.records,
            }),
            summary: json!({
                "records_verified":filtered.records,
                "byte_size":filtered.bytes,
                "sha256":filtered.checksum,
                "durable_target":final_path,
            }),
        })
    }
}

fn storage(error: impl std::fmt::Display) -> BackfillExecutionError {
    BackfillExecutionError::new(
        crate::domain::BackfillFailureKind::Integrity,
        "pmxt_temperature_storage",
        error.to_string(),
    )
}
