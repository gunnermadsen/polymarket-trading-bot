use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use super::IngesterStrategyKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesiredState {
    Running,
    Stopped,
}

impl DesiredState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Stopped => "stopped",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservedState {
    Starting,
    Running,
    Degraded,
    Restarting,
    Stopping,
    Stopped,
    Failed,
    Unsupported,
}

impl ObservedState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Degraded => "degraded",
            Self::Restarting => "restarting",
            Self::Stopping => "stopping",
            Self::Stopped => "stopped",
            Self::Failed => "failed",
            Self::Unsupported => "unsupported",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Unknown,
    Healthy,
    Degraded,
    Unhealthy,
}

impl HealthStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IngesterProfile {
    pub strategy_key: IngesterStrategyKey,
    pub config_schema_version: i32,
    pub config: Value,
    pub desired_state: DesiredState,
    pub desired_generation: i64,
    pub observed_state: ObservedState,
    pub health_status: HealthStatus,
    pub applied_generation: Option<i64>,
    pub checkpoint_schema_version: i32,
    pub checkpoint: Value,
    pub lease_owner: Option<String>,
    pub lease_token: Option<Uuid>,
    pub lease_expires_at: Option<DateTime<Utc>>,
    pub heartbeat_at: Option<DateTime<Utc>>,
    pub started_at: Option<DateTime<Utc>>,
    pub stopped_at: Option<DateTime<Utc>>,
    pub last_source_event_at: Option<DateTime<Utc>>,
    pub last_provider_available_at: Option<DateTime<Utc>>,
    pub last_persisted_at: Option<DateTime<Utc>>,
    pub source_watermark: Option<DateTime<Utc>>,
    pub availability_watermark: Option<DateTime<Utc>>,
    pub consecutive_failures: i32,
    pub restart_count: i64,
    pub last_error_code: Option<String>,
    pub last_error_message: Option<String>,
    pub last_error_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl IngesterProfile {
    pub fn lease_is_current(&self, now: DateTime<Utc>) -> bool {
        self.lease_token.is_some_and(|_| {
            self.lease_expires_at
                .is_some_and(|expires_at| expires_at > now)
        })
    }
}
