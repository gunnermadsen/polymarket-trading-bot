use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEvent {
    pub event_type: String,
    pub emitted_at: DateTime<Utc>,
    pub payload: serde_json::Value,
}

impl ServiceEvent {
    pub fn new(event_type: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            event_type: event_type.into(),
            emitted_at: Utc::now(),
            payload,
        }
    }
}

// Kafka intentionally remains inactive in v1. Future publisher boundary:
//
// pub async fn publish_future_kafka_event(event: &ServiceEvent) -> anyhow::Result<()> {
//     // kafka.publish("polymarket.signals.v1", event).await?;
//     // kafka.publish("polymarket.orders.v1", event).await?;
//     // kafka.publish("polymarket.fills.v1", event).await?;
//     // kafka.publish("polymarket.risk_events.v1", event).await?;
//     Ok(())
// }
