use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use super::IngesterStrategyKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactStatus {
    Open,
    Completed,
    Failed,
}

impl ArtifactStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }

    pub fn from_database(value: &str) -> Option<Self> {
        match value {
            "open" => Some(Self::Open),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }

    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed)
    }
}

/// Bounded lineage for a fixed realtime capture window.
///
/// Artifact identity records how facts were captured. Source-native fields in
/// each fact table remain the factual identity and never include `artifact_id`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CaptureArtifact {
    pub artifact_id: Uuid,
    pub strategy_key: IngesterStrategyKey,
    pub profile_generation: i64,
    pub config_schema_version: i32,
    pub config_sha256: String,
    pub config_snapshot: Value,
    pub capture_window_start: DateTime<Utc>,
    pub capture_window_end: DateTime<Utc>,
    pub minimum_source_timestamp: Option<DateTime<Utc>>,
    pub maximum_source_timestamp: Option<DateTime<Utc>>,
    pub minimum_received_at: Option<DateTime<Utc>>,
    pub maximum_received_at: Option<DateTime<Utc>>,
    pub start_cursor: Option<String>,
    pub end_cursor: Option<String>,
    pub record_count: i64,
    pub content_sha256: Option<String>,
    pub status: ArtifactStatus,
    pub failure_code: Option<String>,
    pub failure_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_statuses_round_trip_database_values() {
        for status in [
            ArtifactStatus::Open,
            ArtifactStatus::Completed,
            ArtifactStatus::Failed,
        ] {
            assert_eq!(ArtifactStatus::from_database(status.as_str()), Some(status));
        }
        assert_eq!(ArtifactStatus::from_database("sealing"), None);
    }

    #[test]
    fn only_completed_and_failed_are_terminal() {
        assert!(!ArtifactStatus::Open.is_terminal());
        assert!(ArtifactStatus::Completed.is_terminal());
        assert!(ArtifactStatus::Failed.is_terminal());
    }
}
