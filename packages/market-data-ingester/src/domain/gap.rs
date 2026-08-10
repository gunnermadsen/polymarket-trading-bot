use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::IngesterStrategyKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GapStatus {
    Open,
    Repairing,
    Repaired,
    Unrecoverable,
}

impl GapStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Repairing => "repairing",
            Self::Repaired => "repaired",
            Self::Unrecoverable => "unrecoverable",
        }
    }

    pub fn from_database(value: &str) -> Option<Self> {
        match value {
            "open" => Some(Self::Open),
            "repairing" => Some(Self::Repairing),
            "repaired" => Some(Self::Repaired),
            "unrecoverable" => Some(Self::Unrecoverable),
            _ => None,
        }
    }

    pub const fn is_resolved(self) -> bool {
        matches!(self, Self::Repaired | Self::Unrecoverable)
    }
}

/// An explicit source interval or cursor discontinuity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DataGap {
    pub gap_id: Uuid,
    pub gap_fingerprint: String,
    pub strategy_key: IngesterStrategyKey,
    pub detected_artifact_id: Option<Uuid>,
    pub repair_artifact_id: Option<Uuid>,
    pub gap_kind: String,
    pub reason_code: String,
    pub reason_message: Option<String>,
    pub source_time_start: Option<DateTime<Utc>>,
    pub source_time_end: Option<DateTime<Utc>>,
    pub start_cursor: Option<String>,
    pub end_cursor: Option<String>,
    pub status: GapStatus,
    pub repair_attempts: i32,
    pub detected_at: DateTime<Utc>,
    pub repair_started_at: Option<DateTime<Utc>>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub resolution_code: Option<String>,
    pub resolution_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gap_statuses_round_trip_database_values() {
        for status in [
            GapStatus::Open,
            GapStatus::Repairing,
            GapStatus::Repaired,
            GapStatus::Unrecoverable,
        ] {
            assert_eq!(GapStatus::from_database(status.as_str()), Some(status));
        }
        assert_eq!(GapStatus::from_database("ignored"), None);
    }

    #[test]
    fn only_repaired_and_unrecoverable_are_resolved() {
        assert!(!GapStatus::Open.is_resolved());
        assert!(!GapStatus::Repairing.is_resolved());
        assert!(GapStatus::Repaired.is_resolved());
        assert!(GapStatus::Unrecoverable.is_resolved());
    }
}
