use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::Serialize;
use uuid::Uuid;

use crate::{
    btc::{BtcIntervalMarket, ChainlinkTwap60Point},
    config::GrafanaLiveConfig,
};

pub const COUNTDOWN_CHANNEL: &str = "stream/polymarket/btc_market_countdown";
pub const MARKET_PATH_CHANNEL: &str = "stream/polymarket/btc_market_path";
pub const ENTRY_STATUS_CHANNEL: &str = "stream/polymarket/btc_entry_status";
const COUNTDOWN_MEASUREMENT: &str = "btc_market_countdown";
const MARKET_PATH_MEASUREMENT: &str = "btc_market_path";
const ENTRY_STATUS_MEASUREMENT: &str = "btc_entry_status";
const MARKET_DISCOVERY_GRACE: chrono::Duration = chrono::Duration::seconds(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryPermissionState {
    Enabled,
    Blocked,
    DisabledByConfiguration,
    Stopped,
    Unknown,
}

impl EntryPermissionState {
    const fn alert_enabled(self) -> i64 {
        match self {
            Self::Enabled => 1,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EntryPermission {
    pub state: EntryPermissionState,
    pub display: String,
    pub reason: Option<String>,
    pub alert_enabled: i64,
}

impl EntryPermission {
    pub fn enabled() -> Self {
        Self::new(EntryPermissionState::Enabled, None)
    }

    pub fn blocked(reason: Option<String>) -> Self {
        Self::new(EntryPermissionState::Blocked, reason)
    }

    pub fn disabled() -> Self {
        Self::new(EntryPermissionState::DisabledByConfiguration, None)
    }

    pub fn stopped() -> Self {
        Self::new(EntryPermissionState::Stopped, None)
    }

    pub fn unknown(reason: Option<String>) -> Self {
        Self::new(EntryPermissionState::Unknown, reason)
    }

    fn new(state: EntryPermissionState, reason: Option<String>) -> Self {
        let reason = reason.filter(|value| !value.trim().is_empty());
        let display = match state {
            EntryPermissionState::Enabled => "Enabled".to_string(),
            EntryPermissionState::Blocked => format!(
                "Blocked — {}",
                reason.as_deref().unwrap_or("status unavailable")
            ),
            EntryPermissionState::DisabledByConfiguration => {
                "Blocked — disabled by configuration".to_string()
            }
            EntryPermissionState::Stopped => "Blocked — stopped".to_string(),
            EntryPermissionState::Unknown => "Blocked — status unavailable".to_string(),
        };
        Self {
            state,
            display,
            reason,
            alert_enabled: state.alert_enabled(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProcessEntryPermission {
    pub process_id: Uuid,
    #[serde(flatten)]
    pub permission: EntryPermission,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TradingEntryStatusSnapshot {
    pub observed_at: DateTime<Utc>,
    pub observed_at_epoch_seconds: i64,
    pub processes: Vec<ProcessEntryPermission>,
    pub aggregate: EntryPermission,
}

impl TradingEntryStatusSnapshot {
    pub fn new(observed_at: DateTime<Utc>, mut processes: Vec<ProcessEntryPermission>) -> Self {
        processes.sort_by_key(|process| process.process_id);
        let aggregate = aggregate_entry_permission(&processes);
        Self {
            observed_at,
            observed_at_epoch_seconds: observed_at.timestamp(),
            processes,
            aggregate,
        }
    }

    pub fn select(&self, scope: &str, process_id: Option<Uuid>) -> EntryStatusSelection {
        let permission = if scope.eq_ignore_ascii_case("all processes") || scope == "all" {
            self.aggregate.clone()
        } else {
            process_id
                .and_then(|process_id| {
                    self.processes
                        .iter()
                        .find(|process| process.process_id == process_id)
                })
                .map(|process| process.permission.clone())
                .unwrap_or_else(|| {
                    EntryPermission::unknown(Some("process_status_unavailable".to_string()))
                })
        };
        let state = if permission.state == EntryPermissionState::Enabled {
            EntryPermissionState::Enabled
        } else {
            EntryPermissionState::Blocked
        };
        EntryStatusSelection {
            observed_at: self.observed_at,
            observed_at_epoch_seconds: self.observed_at_epoch_seconds,
            state,
            display: permission.display,
            reason: permission.reason,
            alert_enabled: permission.alert_enabled,
        }
    }

    pub fn influx_line(&self) -> String {
        let timestamp_nanos = self
            .observed_at
            .timestamp_nanos_opt()
            .expect("a current UTC timestamp is representable in nanoseconds");
        let mut fields = Vec::with_capacity(self.processes.len() + 1);
        fields.push(format!(
            "aggregate=\"{}\"",
            escape_influx_string(&self.aggregate.display)
        ));
        for process in &self.processes {
            fields.push(format!(
                "process_{}=\"{}\"",
                process.process_id.simple(),
                escape_influx_string(&process.permission.display)
            ));
        }
        format!(
            "{ENTRY_STATUS_MEASUREMENT} {} {timestamp_nanos}",
            fields.join(",")
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EntryStatusSelection {
    pub observed_at: DateTime<Utc>,
    pub observed_at_epoch_seconds: i64,
    pub state: EntryPermissionState,
    pub display: String,
    pub reason: Option<String>,
    pub alert_enabled: i64,
}

fn aggregate_entry_permission(processes: &[ProcessEntryPermission]) -> EntryPermission {
    if processes.is_empty() {
        return EntryPermission::blocked(Some("status unavailable".to_string()));
    }
    let total = processes.len();
    let blocked = processes
        .iter()
        .filter(|process| process.permission.state != EntryPermissionState::Enabled)
        .count();
    if blocked > 0 {
        let mut permission = EntryPermission::blocked(Some(format!("{blocked}_of_{total}")));
        permission.display = format!("Blocked — {blocked} of {total}");
        return permission;
    }
    let mut permission = EntryPermission::enabled();
    permission.display = "All enabled".to_string();
    permission
}

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
            .filter(|market| market.is_interval_window(observed_at))
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

#[derive(Debug, Clone)]
struct RetainedMarketPath {
    market_id: String,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    price_to_beat: Option<Decimal>,
    points: BTreeMap<DateTime<Utc>, Decimal>,
}

#[derive(Debug, Default)]
pub struct MarketPathPublicationState {
    retained: Option<RetainedMarketPath>,
}

impl MarketPathPublicationState {
    pub fn observe(
        &mut self,
        observed_at: DateTime<Utc>,
        observation: Option<(BtcIntervalMarket, Vec<ChainlinkTwap60Point>)>,
    ) -> Option<MarketPathSnapshot> {
        if let Some((market, points)) = observation {
            let rollover = self
                .retained
                .as_ref()
                .is_none_or(|retained| retained.market_id != market.market_id);
            if rollover {
                self.retained = Some(RetainedMarketPath {
                    market_id: market.market_id.clone(),
                    window_start: market.window_start,
                    window_end: market.window_end,
                    price_to_beat: None,
                    points: BTreeMap::new(),
                });
            }
            if let Some(retained) = self.retained.as_mut() {
                for point in points {
                    if point.source_timestamp < retained.window_start
                        || point.source_timestamp > retained.window_end
                        || point.source_timestamp > observed_at
                        || point.available_at > observed_at
                    {
                        continue;
                    }
                    retained.points.insert(point.source_timestamp, point.price);
                }
                if retained.price_to_beat.is_none() {
                    retained.price_to_beat = retained
                        .points
                        .range(
                            retained.window_start
                                ..=retained.window_start + chrono::Duration::seconds(5),
                        )
                        .next()
                        .map(|(_, price)| *price);
                }
            }
        }

        let retained = self.retained.as_ref()?;
        if observed_at > retained.window_end + MARKET_DISCOVERY_GRACE {
            self.retained = None;
            return None;
        }
        if retained.points.is_empty() {
            return None;
        }
        Some(MarketPathSnapshot {
            observed_at,
            market_id: retained.market_id.clone(),
            price_to_beat: retained.price_to_beat,
            points: retained
                .points
                .iter()
                .map(|(observed_at, price)| MarketPathPoint {
                    observed_at: *observed_at,
                    price: *price,
                })
                .collect(),
        })
    }
}

impl MarketPathSnapshot {
    pub fn resolve(
        observed_at: DateTime<Utc>,
        market: &BtcIntervalMarket,
        twap_history: impl IntoIterator<Item = ChainlinkTwap60Point>,
    ) -> Self {
        let mut points = twap_history
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
            })
            .collect::<Vec<_>>();
        points.sort_by_key(|point| point.observed_at);
        points.dedup_by(|right, left| right.observed_at == left.observed_at);
        let price_to_beat = points
            .first()
            .filter(|point| point.observed_at <= market.window_start + chrono::Duration::seconds(5))
            .map(|point| point.price);
        Self {
            observed_at,
            market_id: market.market_id.clone(),
            price_to_beat,
            points,
        }
    }

    pub fn influx_body(&self) -> Option<String> {
        if self.points.is_empty() {
            return None;
        }
        let price_to_beat = self.price_to_beat?;

        let mut body = String::with_capacity(self.points.len().saturating_mul(128));
        for point in &self.points {
            let point_epoch_nanos = point
                .observed_at
                .timestamp_nanos_opt()
                .expect("a current UTC timestamp is representable in nanoseconds");
            writeln!(
                body,
                "{MARKET_PATH_MEASUREMENT} twap_price={},price_to_beat={price_to_beat} {point_epoch_nanos}",
                point.price,
            )
            .expect("writing an Influx line into a String cannot fail");
        }
        body.pop();
        Some(body)
    }
}

fn escape_influx_string(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
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
        let Some(body) = snapshot.influx_body() else {
            return Ok(());
        };
        self.publish_body(body, "market path").await
    }

    pub async fn publish_entry_status(&self, snapshot: &TradingEntryStatusSnapshot) -> Result<()> {
        self.publish_body(snapshot.influx_line(), "trading entry status")
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

    fn twap_point(source_timestamp: DateTime<Utc>, price: Decimal) -> ChainlinkTwap60Point {
        ChainlinkTwap60Point {
            source_timestamp,
            available_at: source_timestamp,
            price,
        }
    }

    fn process_permission(process_id: &str, permission: EntryPermission) -> ProcessEntryPermission {
        ProcessEntryPermission {
            process_id: Uuid::parse_str(process_id).unwrap(),
            permission,
        }
    }

    #[test]
    fn entry_status_selects_exactly_one_process_or_aggregate_value() {
        let now = Utc.timestamp_opt(1_800_000_000, 0).unwrap();
        let enabled_id = "11111111-1111-1111-1111-111111111111";
        let blocked_id = "22222222-2222-2222-2222-222222222222";
        let snapshot = TradingEntryStatusSnapshot::new(
            now,
            vec![
                process_permission(enabled_id, EntryPermission::enabled()),
                process_permission(
                    blocked_id,
                    EntryPermission::blocked(Some("live_rest_reconcile_stale".to_string())),
                ),
            ],
        );

        assert_eq!(
            snapshot
                .select(
                    "Selected process",
                    Some(Uuid::parse_str(enabled_id).unwrap())
                )
                .display,
            "Enabled"
        );
        assert_eq!(
            snapshot.select("All processes", None).display,
            "Blocked — 1 of 2"
        );
        assert_eq!(snapshot.select("All processes", None).alert_enabled, 0);
        let unavailable = snapshot.select("Selected process", None);
        assert_eq!(unavailable.state, EntryPermissionState::Blocked);
        assert_eq!(unavailable.display, "Blocked — status unavailable");
    }

    #[test]
    fn entry_status_aggregate_reports_every_non_enabled_process_as_blocked() {
        let now = Utc.timestamp_opt(1_800_000_000, 0).unwrap();
        let snapshot = TradingEntryStatusSnapshot::new(
            now,
            vec![
                process_permission(
                    "11111111-1111-1111-1111-111111111111",
                    EntryPermission::unknown(Some("runtime_not_attached".to_string())),
                ),
                process_permission(
                    "22222222-2222-2222-2222-222222222222",
                    EntryPermission::stopped(),
                ),
                process_permission(
                    "33333333-3333-3333-3333-333333333333",
                    EntryPermission::disabled(),
                ),
            ],
        );

        assert_eq!(snapshot.aggregate.display, "Blocked — 3 of 3");
        assert_eq!(snapshot.aggregate.state, EntryPermissionState::Blocked);
    }

    #[test]
    fn entry_status_live_measurement_has_one_field_per_selection() {
        let now = Utc.timestamp_opt(1_800_000_000, 0).unwrap();
        let snapshot = TradingEntryStatusSnapshot::new(
            now,
            vec![process_permission(
                "11111111-1111-1111-1111-111111111111",
                EntryPermission::enabled(),
            )],
        );
        let line = snapshot.influx_line();

        assert!(line.starts_with("btc_entry_status aggregate=\"All enabled\""));
        assert!(line.contains("process_11111111111111111111111111111111=\"Enabled\""));
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
    fn countdown_uses_time_window_without_weakening_trading_status() {
        let now = Utc.timestamp_opt(1_800_000_000, 0).unwrap();
        let mut closed = market("one", now);
        closed.closed = true;
        closed.accepting_orders = false;

        assert!(!closed.is_trade_window(now));
        let snapshot = CountdownSnapshot::resolve(now, 1, vec![closed]);
        assert_eq!(snapshot.status, CountdownStatus::Active);
        assert_eq!(snapshot.seconds_remaining, 147);
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
        let before = twap_point(market.window_start - ChronoDuration::seconds(1), dec!(99));
        let opening = twap_point(market.window_start, dec!(100.5));
        let current = twap_point(market.window_start + ChronoDuration::seconds(1), dec!(100));
        let future = twap_point(now + ChronoDuration::seconds(1), dec!(101));

        let snapshot =
            MarketPathSnapshot::resolve(now, &market, [before, opening, current.clone(), future]);

        assert_eq!(snapshot.points.len(), 2);
        assert_eq!(snapshot.points[0].observed_at, market.window_start);
        assert_eq!(snapshot.points[0].price, dec!(100.5));
        assert_eq!(snapshot.points[1].observed_at, current.source_timestamp);
        assert_eq!(snapshot.points[1].price, dec!(100));
    }

    #[test]
    fn market_path_renders_a_stable_measurement_schema() {
        let now = Utc.timestamp_opt(1_800_000_100, 0).unwrap();
        let market = market("one", now);
        let snapshot = MarketPathSnapshot::resolve(
            now,
            &market,
            [
                twap_point(market.window_start, dec!(100.5)),
                twap_point(market.window_start + ChronoDuration::seconds(1), dec!(100)),
            ],
        );
        let body = snapshot.influx_body().expect("market path has points");

        assert!(body.starts_with("btc_market_path twap_price=100.5,price_to_beat=100.5"));
        assert!(!body.contains("snapshot_"));
        assert!(body.lines().all(|line| {
            line.split_once(' ')
                .is_some_and(|(_, fields_and_timestamp)| {
                    fields_and_timestamp
                        .split_once(' ')
                        .is_some_and(|(fields, _)| {
                            fields.split(',').all(|field| {
                                field.starts_with("twap_price=")
                                    || field.starts_with("price_to_beat=")
                            })
                        })
                })
        }));
    }

    #[test]
    fn market_path_without_points_does_not_emit_an_invalid_numeric_frame() {
        let now = Utc.timestamp_opt(1_800_000_100, 0).unwrap();
        let market = market("one", now);
        let snapshot = MarketPathSnapshot::resolve(now, &market, []);

        assert_eq!(snapshot.influx_body(), None);
    }

    #[test]
    fn market_path_without_opening_target_does_not_publish_a_partial_schema() {
        let now = Utc.timestamp_opt(1_800_000_100, 0).unwrap();
        let market = market("one", now);
        let snapshot = MarketPathSnapshot::resolve(
            now,
            &market,
            [twap_point(
                market.window_start + ChronoDuration::seconds(10),
                dec!(100),
            )],
        );

        assert_eq!(snapshot.price_to_beat, None);
        assert_eq!(snapshot.influx_body(), None);
    }

    #[test]
    fn publication_state_retains_opening_target_after_more_than_six_hundred_updates() {
        let now = Utc.timestamp_opt(1_800_000_100, 0).unwrap();
        let market = market("one", now);
        let first_batch = (0..600)
            .map(|index| {
                twap_point(
                    market.window_start + ChronoDuration::milliseconds(index * 100),
                    dec!(100.5),
                )
            })
            .collect();
        let second_batch = (600..1_200)
            .map(|index| {
                twap_point(
                    market.window_start + ChronoDuration::milliseconds(index * 100),
                    dec!(101),
                )
            })
            .collect();
        let mut state = MarketPathPublicationState::default();
        state.observe(
            market.window_start + ChronoDuration::seconds(60),
            Some((market.clone(), first_batch)),
        );
        let snapshot = state
            .observe(
                market.window_start + ChronoDuration::seconds(120),
                Some((market.clone(), second_batch)),
            )
            .unwrap();

        assert_eq!(snapshot.points.len(), 1_200);
        assert_eq!(snapshot.points[0].observed_at, market.window_start);
        assert_eq!(snapshot.price_to_beat, Some(dec!(100.5)));
    }

    #[test]
    fn publication_state_freezes_target_and_survives_discovery_gap_through_close() {
        let now = Utc.timestamp_opt(1_800_000_100, 0).unwrap();
        let market = market("one", now);
        let mut state = MarketPathPublicationState::default();
        state.observe(
            market.window_start + ChronoDuration::seconds(1),
            Some((
                market.clone(),
                vec![twap_point(market.window_start, dec!(100.5))],
            )),
        );
        let changed_opening = ChainlinkTwap60Point {
            price: dec!(999),
            source_timestamp: market.window_start,
            available_at: market.window_start + ChronoDuration::seconds(2),
        };
        let updated = state
            .observe(
                market.window_start + ChronoDuration::seconds(2),
                Some((market.clone(), vec![changed_opening])),
            )
            .unwrap();
        let at_close = state.observe(market.window_end, None).unwrap();

        assert_eq!(updated.price_to_beat, Some(dec!(100.5)));
        assert_eq!(at_close.price_to_beat, Some(dec!(100.5)));
        assert_eq!(at_close.market_id, "one");
    }

    #[test]
    fn publication_state_resets_only_for_confirmed_successor_market() {
        let now = Utc.timestamp_opt(1_800_000_100, 0).unwrap();
        let first = market("one", now);
        let mut second = market("two", now + ChronoDuration::seconds(300));
        second.window_start = first.window_end;
        second.window_end = second.window_start + ChronoDuration::seconds(300);
        let mut state = MarketPathPublicationState::default();
        state.observe(
            first.window_start,
            Some((
                first.clone(),
                vec![twap_point(first.window_start, dec!(100.5))],
            )),
        );
        let successor = state
            .observe(
                second.window_start,
                Some((
                    second.clone(),
                    vec![twap_point(second.window_start, dec!(102))],
                )),
            )
            .unwrap();

        assert_eq!(successor.market_id, "two");
        assert_eq!(successor.points.len(), 1);
        assert_eq!(successor.price_to_beat, Some(dec!(102)));
    }
}
