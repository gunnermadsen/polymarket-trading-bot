use std::{collections::BTreeSet, time::Duration};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};

use crate::{btc::BtcIntervalMarket, config::GrafanaLiveConfig};

pub const COUNTDOWN_CHANNEL: &str = "stream/polymarket/btc_market_countdown";
const COUNTDOWN_MEASUREMENT: &str = "btc_market_countdown";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountdownStatus {
    Active,
    NoActiveProcess,
    Discovering,
    MarketDisagreement,
}

impl CountdownStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::NoActiveProcess => "no_active_process",
            Self::Discovering => "discovering",
            Self::MarketDisagreement => "market_disagreement",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CountdownSnapshot {
    pub observed_at: DateTime<Utc>,
    pub status: CountdownStatus,
    pub seconds_remaining: i64,
    pub active_processes: usize,
    pub market_id: String,
    pub event_slug: String,
    pub window_end: Option<DateTime<Utc>>,
}

impl CountdownSnapshot {
    pub fn resolve(
        observed_at: DateTime<Utc>,
        active_processes: usize,
        markets: Vec<BtcIntervalMarket>,
    ) -> Self {
        if active_processes == 0 {
            return Self::unavailable(
                observed_at,
                CountdownStatus::NoActiveProcess,
                active_processes,
            );
        }

        let current_markets = markets
            .into_iter()
            .filter(|market| market.is_trade_window(observed_at))
            .collect::<Vec<_>>();
        if current_markets.is_empty() {
            return Self::unavailable(observed_at, CountdownStatus::Discovering, active_processes);
        }

        let identities = current_markets
            .iter()
            .map(|market| {
                (
                    market.market_id.clone(),
                    market.condition_id.clone(),
                    market.window_start,
                    market.window_end,
                )
            })
            .collect::<BTreeSet<_>>();
        if identities.len() != 1 || current_markets.len() != active_processes {
            return Self::unavailable(
                observed_at,
                CountdownStatus::MarketDisagreement,
                active_processes,
            );
        }

        let market = &current_markets[0];
        let remaining_millis = market
            .window_end
            .signed_duration_since(observed_at)
            .num_milliseconds()
            .max(0);
        Self {
            observed_at,
            status: CountdownStatus::Active,
            seconds_remaining: remaining_millis.saturating_add(999) / 1_000,
            active_processes,
            market_id: market.market_id.clone(),
            event_slug: market.event_slug.clone(),
            window_end: Some(market.window_end),
        }
    }

    fn unavailable(
        observed_at: DateTime<Utc>,
        status: CountdownStatus,
        active_processes: usize,
    ) -> Self {
        Self {
            observed_at,
            status,
            seconds_remaining: -1,
            active_processes,
            market_id: String::new(),
            event_slug: String::new(),
            window_end: None,
        }
    }

    pub fn influx_line(&self) -> String {
        let timestamp_nanos = self
            .observed_at
            .timestamp_nanos_opt()
            .expect("a current UTC timestamp is representable in nanoseconds");
        let window_end_epoch_seconds = self.window_end.map(|value| value.timestamp()).unwrap_or(0);
        format!(
            "{COUNTDOWN_MEASUREMENT},status={} seconds_remaining={}i,active_processes={}i,available={},market_id=\"{}\",event_slug=\"{}\",window_end_epoch_seconds={}i {}",
            self.status.as_str(),
            self.seconds_remaining,
            self.active_processes,
            self.status == CountdownStatus::Active,
            escape_influx_string(&self.market_id),
            escape_influx_string(&self.event_slug),
            window_end_epoch_seconds,
            timestamp_nanos,
        )
    }
}

fn escape_influx_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

pub struct GrafanaLivePublisher {
    client: reqwest::Client,
    config: GrafanaLiveConfig,
}

impl GrafanaLivePublisher {
    pub fn new(config: GrafanaLiveConfig) -> Result<Self> {
        if !config.enabled {
            bail!("Grafana Live publisher cannot start while disabled");
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .context("failed to build Grafana Live HTTP client")?;
        Ok(Self { client, config })
    }

    pub fn publish_interval(&self) -> Duration {
        self.config.publish_interval
    }

    pub async fn publish(&self, snapshot: &CountdownSnapshot) -> Result<()> {
        let request = self
            .client
            .post(&self.config.push_url)
            .header(reqwest::header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body(snapshot.influx_line());
        let request = if let Some(token) = &self.config.bearer_token {
            request.bearer_auth(token)
        } else {
            request.basic_auth(
                self.config
                    .basic_auth_username
                    .as_deref()
                    .unwrap_or_default(),
                self.config.basic_auth_password.as_deref(),
            )
        };
        request
            .send()
            .await
            .context("failed to send Grafana Live countdown measurement")?
            .error_for_status()
            .context("Grafana rejected Grafana Live countdown measurement")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use chrono::{Duration as ChronoDuration, TimeZone};
    use rust_decimal_macros::dec;

    use super::*;

    fn market(id: &str, now: DateTime<Utc>) -> BtcIntervalMarket {
        BtcIntervalMarket {
            event_id: format!("event-{id}"),
            event_slug: format!("btc-updown-5m-{}", now.timestamp()),
            series_slug: "btc-up-or-down-5m".to_string(),
            market_id: id.to_string(),
            condition_id: format!("condition-{id}"),
            window_start: now - ChronoDuration::seconds(100),
            window_end: now + ChronoDuration::milliseconds(146_001),
            up_token_id: format!("up-{id}"),
            down_token_id: format!("down-{id}"),
            tick_size: dec!(0.01),
            minimum_order_size: None,
            resolution_source: "Chainlink BTC/USD".to_string(),
            active: true,
            closed: false,
            accepting_orders: true,
            fees_enabled: false,
            fee_schedule: serde_json::json!({}),
            raw_payload: serde_json::json!({}),
        }
    }

    #[test]
    fn resolves_one_canonical_market_for_all_active_processes() {
        let now = Utc.timestamp_opt(1_800_000_000, 0).unwrap();
        let current = market("one", now);
        let snapshot = CountdownSnapshot::resolve(now, 2, vec![current.clone(), current]);

        assert_eq!(snapshot.status, CountdownStatus::Active);
        assert_eq!(snapshot.seconds_remaining, 147);
        assert_eq!(snapshot.active_processes, 2);
        assert_eq!(snapshot.market_id, "one");
    }

    #[test]
    fn fails_closed_when_active_processes_disagree() {
        let now = Utc.timestamp_opt(1_800_000_000, 0).unwrap();
        let snapshot =
            CountdownSnapshot::resolve(now, 2, vec![market("one", now), market("two", now)]);

        assert_eq!(snapshot.status, CountdownStatus::MarketDisagreement);
        assert_eq!(snapshot.seconds_remaining, -1);
    }

    #[test]
    fn reports_discovery_instead_of_reusing_an_expired_market() {
        let now = Utc.timestamp_opt(1_800_000_000, 0).unwrap();
        let mut expired = market("one", now);
        expired.window_end = now;
        let snapshot = CountdownSnapshot::resolve(now, 1, vec![expired]);

        assert_eq!(snapshot.status, CountdownStatus::Discovering);
        assert_eq!(snapshot.seconds_remaining, -1);
    }

    #[test]
    fn renders_a_stable_influx_measurement_contract() {
        let now = Utc.timestamp_opt(1_800_000_000, 0).unwrap();
        let snapshot = CountdownSnapshot::resolve(now, 1, vec![market("one", now)]);
        let line = snapshot.influx_line();

        assert!(line.starts_with("btc_market_countdown,status=active "));
        assert!(line.contains("seconds_remaining=147i"));
        assert!(line.contains("active_processes=1i"));
        assert!(line.contains("available=true"));
        assert!(line.contains("market_id=\"one\""));
    }
}
