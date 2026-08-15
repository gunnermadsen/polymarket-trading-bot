use chrono::{DateTime, Utc};

/// Causal timestamps retained with a source fact.
///
/// These fields intentionally have no universal cross-clock ordering check.
/// Provider clocks and the ingester host clock can be skewed, and some source
/// timestamps describe an interval rather than publication. Source adapters
/// may apply protocol-specific sanity windows without changing the raw values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FactualTimestamps {
    pub source_timestamp: DateTime<Utc>,
    pub provider_available_at: Option<DateTime<Utc>>,
    pub received_at: DateTime<Utc>,
}
