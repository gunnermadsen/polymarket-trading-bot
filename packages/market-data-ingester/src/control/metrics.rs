use anyhow::Result;
use prometheus_client::{
    encoding::{text::encode, EncodeLabelSet},
    metrics::{family::Family, gauge::Gauge},
    registry::Registry,
};

use crate::domain::IngesterProfile;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct StrategyStateLabels {
    strategy: String,
    desired_state: String,
    observed_state: String,
    health_status: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct StrategyLabels {
    strategy: String,
}

pub fn render(profiles: &[IngesterProfile], ready: bool) -> Result<String> {
    let mut registry = Registry::with_prefix("market_data_ingester");
    let liveness = Gauge::<i64>::default();
    let readiness = Gauge::<i64>::default();
    let strategy_state = Family::<StrategyStateLabels, Gauge<i64>>::default();
    let last_persistence = Family::<StrategyLabels, Gauge<i64>>::default();

    registry.register(
        "liveness",
        "Whether the market-data-ingester HTTP process is live.",
        liveness.clone(),
    );
    registry.register(
        "readiness",
        "Whether the strategy supervisor has completed startup reconciliation.",
        readiness.clone(),
    );
    registry.register(
        "strategy_state",
        "Current desired, observed, and health state for each ingester strategy.",
        strategy_state.clone(),
    );
    registry.register(
        "strategy_last_persistence_timestamp_seconds",
        "Unix timestamp of the latest durable persistence for each ingester strategy, or zero before first persistence.",
        last_persistence.clone(),
    );

    liveness.set(1);
    readiness.set(i64::from(ready));
    for profile in profiles {
        let strategy = profile.strategy_key.as_str().to_owned();
        strategy_state
            .get_or_create(&StrategyStateLabels {
                strategy: strategy.clone(),
                desired_state: profile.desired_state.as_str().to_owned(),
                observed_state: profile.observed_state.as_str().to_owned(),
                health_status: profile.health_status.as_str().to_owned(),
            })
            .set(1);
        last_persistence
            .get_or_create(&StrategyLabels { strategy })
            .set(
                profile
                    .last_persisted_at
                    .map_or(0, |timestamp| timestamp.timestamp()),
            );
    }

    let mut body = String::new();
    encode(&mut body, &registry)?;
    Ok(body)
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};
    use serde_json::json;

    use crate::domain::{DesiredState, HealthStatus, IngesterStrategyKey, ObservedState};

    use super::*;

    #[test]
    fn renders_service_and_strategy_health_without_unbounded_labels() {
        let persisted_at = Utc
            .timestamp_opt(1_787_600_000, 0)
            .single()
            .expect("valid timestamp");
        let profile = IngesterProfile {
            strategy_key: IngesterStrategyKey::PolymarketChainlinkBtcusdTwap,
            config_schema_version: 1,
            config: json!({}),
            desired_state: DesiredState::Running,
            desired_generation: 3,
            observed_state: ObservedState::Degraded,
            health_status: HealthStatus::Degraded,
            applied_generation: Some(3),
            checkpoint_schema_version: 1,
            checkpoint: json!({}),
            lease_owner: None,
            lease_token: None,
            lease_expires_at: None,
            heartbeat_at: None,
            started_at: None,
            stopped_at: None,
            last_source_event_at: None,
            last_provider_available_at: None,
            last_persisted_at: Some(persisted_at),
            source_watermark: None,
            availability_watermark: None,
            consecutive_failures: 0,
            restart_count: 0,
            last_error_code: Some("not_exported_as_a_label".to_owned()),
            last_error_message: Some("also not exported".to_owned()),
            last_error_at: None,
            created_at: persisted_at,
            updated_at: persisted_at,
        };

        let rendered = render(&[profile], true).expect("metrics render");

        assert!(rendered.contains("market_data_ingester_liveness 1"));
        assert!(rendered.contains("market_data_ingester_readiness 1"));
        assert!(
            rendered.contains("market_data_ingester_strategy_state"),
            "{rendered}"
        );
        assert!(rendered.contains("strategy=\"polymarket_chainlink_btcusd_twap\""));
        assert!(rendered.contains("desired_state=\"running\""));
        assert!(rendered.contains("observed_state=\"degraded\""));
        assert!(rendered.contains("health_status=\"degraded\""));
        assert!(rendered.contains(
            "market_data_ingester_strategy_last_persistence_timestamp_seconds{strategy=\"polymarket_chainlink_btcusd_twap\"} 1787600000"
        ));
        assert!(!rendered.contains("not_exported_as_a_label"));
        assert!(!rendered.contains("also not exported"));
    }

    #[test]
    fn renders_exactly_one_bounded_state_and_persistence_series_for_every_strategy() {
        let timestamp = Utc.timestamp_opt(1_787_600_000, 0).single().unwrap();
        let profiles = IngesterStrategyKey::ALL
            .into_iter()
            .map(|strategy_key| IngesterProfile {
                strategy_key,
                config_schema_version: 1,
                config: json!({}),
                desired_state: DesiredState::Stopped,
                desired_generation: 1,
                observed_state: ObservedState::Stopped,
                health_status: HealthStatus::Unknown,
                applied_generation: None,
                checkpoint_schema_version: 1,
                checkpoint: json!({}),
                lease_owner: None,
                lease_token: None,
                lease_expires_at: None,
                heartbeat_at: None,
                started_at: None,
                stopped_at: Some(timestamp),
                last_source_event_at: None,
                last_provider_available_at: None,
                last_persisted_at: None,
                source_watermark: None,
                availability_watermark: None,
                consecutive_failures: 0,
                restart_count: 0,
                last_error_code: None,
                last_error_message: None,
                last_error_at: None,
                created_at: timestamp,
                updated_at: timestamp,
            })
            .collect::<Vec<_>>();

        let rendered = render(&profiles, true).expect("metrics render");
        assert_eq!(
            rendered
                .matches("market_data_ingester_strategy_state{")
                .count(),
            11
        );
        assert_eq!(
            rendered
                .matches("market_data_ingester_strategy_last_persistence_timestamp_seconds{")
                .count(),
            11
        );
        for key in IngesterStrategyKey::ALL {
            assert!(rendered.contains(&format!("strategy=\"{}\"", key.as_str())));
        }
        assert!(!rendered.contains("error_code="));
    }
}
