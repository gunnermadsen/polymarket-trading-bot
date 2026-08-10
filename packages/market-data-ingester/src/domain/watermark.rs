use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StrategyWatermarks {
    pub source: Option<DateTime<Utc>>,
    pub availability: Option<DateTime<Utc>>,
    pub last_received: Option<DateTime<Utc>>,
    pub last_persisted: Option<DateTime<Utc>>,
}
