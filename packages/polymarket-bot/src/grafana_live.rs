use std::{collections::BTreeSet, fmt::Write as _, time::Duration};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

use crate::{
    btc::{BtcIntervalMarket, ChainlinkMidPoint},
    config::GrafanaLiveConfig,
};

pub const COUNTDOWN_CHANNEL: &str = "stream/polymarket/btc_market_countdown";
pub const MARKET_PATH_CHANNEL: &str = "stream/polymarket/btc_market_path";
const COUNTDOWN_MEASUREMENT: &str = "btc_market_countdown";
const MARKET_PATH_MEASUREMENT: &str = "btc_market_path";

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
            "{COUNTDOWN_MEASUREMENT} seconds_remaining={}i,active_processes={}i,available={},status=\"{}\",market_id=\"{}\",event_slug=\"{}\",window_end_epoch_seconds={}i {}",
            self.seconds_remaining,
            self.active_processes,
            self.status == CountdownStatus::Active,
            self.status.as_str(),
            escape_influx_string(&self.market_id),
            escape_influx_string(&self.event_slug),
            window_end_epoch_seconds,
            timestamp_nanos,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarketPathPoint {
    pub observed_at: DateTime<Utc>,
    pub price: Decimal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarketPathSnapshot {
    pub observed_at: DateTime<Utc>,
    pub market_id: String,
    pub price_to_beat: Option<Decimal>,
    pub points: Vec<MarketPathPoint>,
}

impl MarketPathSnapshot {
    pub fn resolve(
        observed_at: DateTime<Utc>,
        market: &BtcIntervalMarket,
        opening_reference: Option<MarketPathPoint>,
        chainlink_history: impl IntoIterator<Item = ChainlinkMidPoint>,
    ) -> Self {
        let opening_reference = opening_reference.filter(|point| {
            point.observed_at >= market.window_start
                && point.observed_at <= market.window_end
                && point.observed_at <= observed_at
        });
        let price_to_beat = opening_reference.as_ref().map(|point| point.price);
        let mut points = Vec::new();
        if let Some(opening_reference) = opening_reference {
            points.push(opening_reference);
        }
        points.extend(
            chainlink_history
                .into_iter()
                .filter(|point| {
                    point.source_timestamp >= market.window_start
                        && point.source_timestamp <= market.window_end
                        && point.source_timestamp <= observed_at
                        && point.available_at <= observed_at
                })
                .map(|point| MarketPathPoint {
                    observed_at: point.source_timestamp,
                    price: point.price,
                }),
        );
        points.sort_by_key(|point| point.observed_at);
        points.dedup_by(|right, left| right.observed_at == left.observed_at);
        Self {
            observed_at,
            market_id: market.market_id.clone(),
            price_to_beat,
            points,
        }
    }

    pub fn reset(observed_at: DateTime<Utc>) -> Self {
        Self {
            observed_at,
            market_id: String::new(),
            price_to_beat: None,
            points: Vec::new(),
        }
    }

    pub fn influx_body(&self) -> String {
        let snapshot_epoch_nanos = self
            .observed_at
            .timestamp_nanos_opt()
            .expect("a current UTC timestamp is representable in nanoseconds");
        let snapshot_field = format!("snapshot_{snapshot_epoch_nanos}");
        if self.points.is_empty() {
            return format!("{MARKET_PATH_MEASUREMENT} {snapshot_field}=1i {snapshot_epoch_nanos}");
        }

        let mut body = String::with_capacity(self.points.len().saturating_mul(128));
        for point in &self.points {
            let point_epoch_nanos = point
                .observed_at
                .timestamp_nanos_opt()
                .expect("a current UTC timestamp is representable in nanoseconds");
            match self.price_to_beat {
                Some(price_to_beat) => writeln!(
                    body,
                    "{MARKET_PATH_MEASUREMENT} chainlink_price={},price_to_beat={price_to_beat},{snapshot_field}=1i {point_epoch_nanos}",
                    point.price,
                ),
                None => writeln!(
                    body,
                    "{MARKET_PATH_MEASUREMENT} chainlink_price={},{}=1i {point_epoch_nanos}",
                    point.price, snapshot_field,
                ),
            }
            .expect("writing an Influx line into a String cannot fail");
        }
        body.pop();
        body
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
        self.publish_body(snapshot.influx_line(), "countdown").await
    }

    pub async fn publish_market_path(&self, snapshot: &MarketPathSnapshot) -> Result<()> {
        self.publish_body(snapshot.influx_body(), "market path")
            .await
    }

    async fn publish_body(&self, body: String, measurement_name: &str) -> Result<()> {
        let request = self
            .client
            .post(&self.config.push_url)
            .header(reqwest::header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body(body);
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
            .with_context(|| format!("failed to send Grafana Live {measurement_name} measurement"))?
            .error_for_status()
            .with_context(|| {
                format!("Grafana rejected Grafana Live {measurement_name} measurement")
            })?;
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

    fn chainlink_point(source_timestamp: DateTime<Utc>, price: Decimal) -> ChainlinkMidPoint {
        ChainlinkMidPoint {
            source_timestamp,
            available_at: source_timestamp,
            price,
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

        assert!(line.starts_with("btc_market_countdown seconds_remaining=147i"));
        assert_eq!(line.split_once(' ').unwrap().0, "btc_market_countdown");
        assert!(line.contains("seconds_remaining=147i"));
        assert!(line.contains("active_processes=1i"));
        assert!(line.contains("available=true"));
        assert!(line.contains("status=\"active\""));
        assert!(line.contains("market_id=\"one\""));
    }

    #[test]
    fn market_path_contains_only_the_current_bounded_market_window() {
        let now = Utc.timestamp_opt(1_800_000_100, 0).unwrap();
        let market = market("one", now);
        let before = chainlink_point(market.window_start - ChronoDuration::seconds(1), dec!(99));
        let current = chainlink_point(market.window_start + ChronoDuration::seconds(1), dec!(100));
        let future = chainlink_point(now + ChronoDuration::seconds(1), dec!(101));
        let opening = MarketPathPoint {
            observed_at: market.window_start,
            price: dec!(100.5),
        };

        let snapshot = MarketPathSnapshot::resolve(
            now,
            &market,
            Some(opening),
            [before, current.clone(), future],
        );

        assert_eq!(snapshot.points.len(), 2);
        assert_eq!(snapshot.points[0].observed_at, market.window_start);
        assert_eq!(snapshot.points[0].price, dec!(100.5));
        assert_eq!(snapshot.points[1].observed_at, current.source_timestamp);
        assert_eq!(snapshot.points[1].price, dec!(100));
    }

    #[test]
    fn market_path_renders_price_target_and_snapshot_replacement_marker() {
        let now = Utc.timestamp_opt(1_800_000_100, 0).unwrap();
        let market = market("one", now);
        let opening = MarketPathPoint {
            observed_at: market.window_start,
            price: dec!(100.5),
        };
        let snapshot = MarketPathSnapshot::resolve(
            now,
            &market,
            Some(opening),
            [chainlink_point(
                market.window_start + ChronoDuration::seconds(1),
                dec!(100),
            )],
        );
        let body = snapshot.influx_body();

        assert!(body.starts_with("btc_market_path chainlink_price=100.5,price_to_beat=100.5"));
        assert!(body.contains("snapshot_1800000100000000000=1i"));
        assert_ne!(
            body,
            MarketPathSnapshot::reset(now + ChronoDuration::seconds(1)).influx_body()
        );
    }

    #[test]
    fn market_path_reset_emits_an_empty_replacement_frame() {
        let now = Utc.timestamp_opt(1_800_000_100, 0).unwrap();
        assert_eq!(
            MarketPathSnapshot::reset(now).influx_body(),
            "btc_market_path snapshot_1800000100000000000=1i 1800000100000000000"
        );
    }
}
