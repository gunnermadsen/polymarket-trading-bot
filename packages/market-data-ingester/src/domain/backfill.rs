use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::PgPool;
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyCapability {
    Realtime,
    Backfill,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StrategyDescriptor {
    pub strategy_key: Arc<str>,
    pub name: Arc<str>,
    pub description: Arc<str>,
    pub capabilities: Vec<StrategyCapability>,
    pub strategy_contract_version: i32,
    pub request_schema_version: Option<i32>,
    pub shardable: bool,
    pub maximum_shards: usize,
}

impl StrategyDescriptor {
    pub fn validate(&self) -> Result<(), BackfillExecutionError> {
        if self.strategy_key.trim().is_empty() {
            return Err(BackfillExecutionError::invalid(
                "strategy_key_empty",
                "strategy key is empty",
            ));
        }
        if self.strategy_contract_version <= 0 {
            return Err(BackfillExecutionError::invalid(
                "strategy_contract_version_invalid",
                "strategy contract version must be positive",
            ));
        }
        if self.capabilities.contains(&StrategyCapability::Backfill)
            && (self.request_schema_version.is_none() || self.maximum_shards == 0)
        {
            return Err(BackfillExecutionError::invalid(
                "backfill_descriptor_invalid",
                "backfill strategies require a request schema version and positive shard limit",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BackfillRequest {
    pub strategy_key: String,
    pub range: BackfillRange,
    #[serde(default = "empty_object")]
    pub parameters: Value,
    #[serde(default)]
    pub execution: ExecutionSelector,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BackfillRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ExecutionSelector {
    pub required_worker_id: Option<String>,
    pub required_deployment: Option<String>,
}

impl ExecutionSelector {
    pub fn validate(&self) -> Result<(), BackfillExecutionError> {
        if self.required_worker_id.is_some() && self.required_deployment.is_some() {
            return Err(BackfillExecutionError::invalid(
                "execution_selector_ambiguous",
                "required_worker_id and required_deployment are mutually exclusive",
            ));
        }
        for (name, value) in [
            ("required_worker_id", self.required_worker_id.as_deref()),
            ("required_deployment", self.required_deployment.as_deref()),
        ] {
            if value.is_some_and(|value| value.trim().is_empty() || value.len() > 255) {
                return Err(BackfillExecutionError::invalid(
                    "execution_selector_invalid",
                    format!("{name} must be non-empty and at most 255 bytes"),
                ));
            }
        }
        Ok(())
    }
}

fn empty_object() -> Value {
    Value::Object(Default::default())
}

#[derive(Debug, Clone)]
pub struct ValidatedBackfillRequest {
    pub strategy_key: Arc<str>,
    pub strategy_contract_version: i32,
    pub request_schema_version: i32,
    pub range_start: DateTime<Utc>,
    pub range_end: DateTime<Utc>,
    pub parameters: Value,
    pub execution: ExecutionSelector,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BackfillShard {
    pub shard_key: String,
    pub range_start: DateTime<Utc>,
    pub range_end: DateTime<Utc>,
    pub parameters: Value,
}

#[derive(Clone)]
pub struct BackfillContext {
    pub pool: PgPool,
    pub job_id: Uuid,
    pub lease_token: Uuid,
    pub worker_id: Arc<str>,
    pub shutdown: CancellationToken,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BackfillOutcome {
    pub records_verified: i64,
    pub verified_coverage: Value,
    pub summary: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackfillFailureKind {
    TransientSource,
    TransientDatabase,
    RateLimited,
    InvalidRequest,
    Integrity,
    LeaseLost,
    Cancelled,
}

impl BackfillFailureKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TransientSource => "transient_source",
            Self::TransientDatabase => "transient_database",
            Self::RateLimited => "rate_limited",
            Self::InvalidRequest => "invalid_request",
            Self::Integrity => "integrity",
            Self::LeaseLost => "lease_lost",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct BackfillExecutionError {
    pub kind: BackfillFailureKind,
    pub code: &'static str,
    pub message: String,
}

impl BackfillExecutionError {
    pub fn new(kind: BackfillFailureKind, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            code,
            message: message.into(),
        }
    }

    pub fn invalid(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(BackfillFailureKind::InvalidRequest, code, message)
    }
}

#[async_trait]
pub trait BackfillWorkerStrategy: Send + Sync {
    fn descriptor(&self) -> &StrategyDescriptor;

    fn validate_request(
        &self,
        request: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError>;

    fn plan_shards(
        &self,
        request: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError>;

    async fn execute_backfill(
        &self,
        context: BackfillContext,
        shard: BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError>;
}
