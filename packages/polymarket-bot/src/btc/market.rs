use std::str::FromStr;

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;

use super::types::{BtcIntervalMarket, BtcOutcome, BTC_INTERVAL_SECONDS, BTC_INTERVAL_SLUG_PREFIX};

#[derive(Debug, Clone, PartialEq)]
pub struct ClobRestOfficialResolution {
    pub market_id: String,
    pub winning_token_id: String,
    pub winning_outcome: BtcOutcome,
    pub observed_at: DateTime<Utc>,
    pub raw_payload: serde_json::Value,
}

pub fn aligned_window_start(now: DateTime<Utc>) -> DateTime<Utc> {
    let epoch = now.timestamp().div_euclid(BTC_INTERVAL_SECONDS) * BTC_INTERVAL_SECONDS;
    DateTime::from_timestamp(epoch, 0).expect("an aligned UTC timestamp is representable")
}

pub fn slug_for_window(window_start: DateTime<Utc>) -> String {
    format!("{BTC_INTERVAL_SLUG_PREFIX}{}", window_start.timestamp())
}

pub fn discovery_windows(now: DateTime<Utc>) -> [DateTime<Utc>; 3] {
    let current = aligned_window_start(now);
    [
        current - Duration::seconds(BTC_INTERVAL_SECONDS),
        current,
        current + Duration::seconds(BTC_INTERVAL_SECONDS),
    ]
}

pub fn window_start_from_slug(slug: &str) -> Result<DateTime<Utc>> {
    let epoch = slug
        .strip_prefix(BTC_INTERVAL_SLUG_PREFIX)
        .context("event slug is not a BTC Up/Down 5m slug")?
        .parse::<i64>()
        .context("BTC Up/Down 5m slug has an invalid epoch suffix")?;
    if epoch.rem_euclid(BTC_INTERVAL_SECONDS) != 0 {
        bail!("BTC Up/Down 5m slug epoch is not aligned to a five-minute boundary");
    }
    DateTime::from_timestamp(epoch, 0).context("BTC Up/Down 5m slug epoch is out of range")
}

/// Parses and validates a Gamma GET /events/slug/{slug} response.
///
/// This deliberately rejects ambiguous metadata. A rotating five-minute strategy must not infer
/// token direction by array position or use a market whose time or resolution source differs from
/// the contract it expects.
pub fn parse_gamma_btc_interval_event(
    value: &serde_json::Value,
    expected_window_start: DateTime<Utc>,
) -> Result<BtcIntervalMarket> {
    let object = value
        .as_object()
        .context("Gamma event response must be an object")?;
    let event_slug = required_string(object, &["slug"])?;
    let slug_window_start = window_start_from_slug(&event_slug)?;
    if slug_window_start != expected_window_start {
        bail!(
            "Gamma event slug window {} does not match requested window {}",
            slug_window_start,
            expected_window_start
        );
    }

    let series_slug = string_field(object, &["seriesSlug", "series_slug"])
        .or_else(|| series_slug_from_relation(object))
        .context("Gamma event is missing a series slug")?;
    if series_slug != "btc-up-or-down-5m" {
        bail!("Gamma event belongs to unexpected series {series_slug}");
    }

    let window_start = datetime_field(object, &["eventStartTime", "startTime"])
        .context("Gamma event is missing eventStartTime")?;
    if window_start != slug_window_start {
        bail!("Gamma eventStartTime does not match its slug epoch");
    }

    let markets = object
        .get("markets")
        .and_then(serde_json::Value::as_array)
        .context("Gamma event is missing its markets array")?;
    if markets.len() != 1 {
        bail!("BTC Up/Down 5m event must contain exactly one market");
    }
    let market = markets[0]
        .as_object()
        .context("Gamma event market must be an object")?;

    let window_end = datetime_field(market, &["endDate", "endDateIso"])
        .or_else(|| datetime_field(object, &["endDate"]))
        .context("Gamma event is missing the market end date")?;
    if window_end - window_start != Duration::seconds(BTC_INTERVAL_SECONDS) {
        bail!("BTC Up/Down market does not have an exact five-minute window");
    }

    let resolution_source = string_field(market, &["resolutionSource", "resolution_source"])
        .or_else(|| string_field(object, &["resolutionSource", "resolution_source"]))
        .context("Gamma event is missing a resolution source")?;
    if !is_chainlink_btc_usd_source(&resolution_source) {
        bail!("BTC Up/Down market resolution source is not Chainlink BTC/USD");
    }

    let outcomes =
        string_array_field(market, &["outcomes"]).context("Gamma market is missing outcomes")?;
    let token_ids = string_array_field(
        market,
        &["clobTokenIds", "clob_token_ids", "tokenIds", "token_ids"],
    )
    .context("Gamma market is missing CLOB token IDs")?;
    if outcomes.len() != 2 || token_ids.len() != 2 || outcomes.len() != token_ids.len() {
        bail!("BTC Up/Down market must contain exactly two outcomes and token IDs");
    }

    let mut up_token_id = None;
    let mut down_token_id = None;
    for (outcome, token_id) in outcomes.iter().zip(&token_ids) {
        match normalize_outcome(outcome)? {
            BtcOutcome::Up if up_token_id.replace(token_id.clone()).is_some() => {
                bail!("BTC Up/Down market contains duplicate Up outcomes")
            }
            BtcOutcome::Down if down_token_id.replace(token_id.clone()).is_some() => {
                bail!("BTC Up/Down market contains duplicate Down outcomes")
            }
            _ => {}
        }
    }
    let up_token_id = up_token_id.context("BTC Up/Down market is missing its Up token")?;
    let down_token_id = down_token_id.context("BTC Up/Down market is missing its Down token")?;
    if up_token_id == down_token_id {
        bail!("BTC Up/Down market token IDs must be distinct");
    }

    let event_id = required_string(object, &["id"])?;
    let market_id = required_string(market, &["id"])?;
    let condition_id = required_string(market, &["conditionId", "condition_id"])?;
    let tick_size = decimal_field(
        market,
        &[
            "orderPriceMinTickSize",
            "minimumTickSize",
            "tickSize",
            "tick_size",
        ],
    )
    .context("Gamma market is missing its minimum tick size")?;
    if tick_size <= Decimal::ZERO {
        bail!("Gamma market minimum tick size must be positive");
    }

    Ok(BtcIntervalMarket {
        event_id,
        event_slug,
        series_slug,
        market_id,
        condition_id,
        window_start,
        window_end,
        up_token_id,
        down_token_id,
        tick_size,
        minimum_order_size: decimal_field(market, &["orderMinSize", "minimumOrderSize"]),
        resolution_source,
        active: bool_field(market, &["active"])
            .or_else(|| bool_field(object, &["active"]))
            .unwrap_or(false),
        closed: bool_field(market, &["closed"])
            .or_else(|| bool_field(object, &["closed"]))
            .unwrap_or(false),
        accepting_orders: bool_field(market, &["acceptingOrders", "accepting_orders"])
            .unwrap_or(false),
        fees_enabled: bool_field(market, &["feesEnabled", "fees_enabled"])
            .or_else(|| bool_field(object, &["feesEnabled", "fees_enabled"]))
            .unwrap_or(false),
        fee_schedule: market
            .get("feeSchedule")
            .or_else(|| object.get("feeSchedule"))
            .cloned()
            .unwrap_or_else(|| serde_json::json!({})),
        raw_payload: value.clone(),
    })
}

/// Validates the public CLOB `GET /markets/{condition_id}` response against the immutable
/// Gamma-discovered identity. An ended market is authoritative only when CLOB marks it closed,
/// exposes exactly the two expected tokens, and names exactly one binary winner at its terminal
/// price. Open markets return `None` and remain watched.
pub fn parse_clob_rest_official_resolution(
    value: &serde_json::Value,
    market: &BtcIntervalMarket,
    observed_at: DateTime<Utc>,
) -> Result<Option<ClobRestOfficialResolution>> {
    let object = value
        .as_object()
        .context("CLOB market response must be an object")?;
    let condition_id = required_string(object, &["condition_id", "conditionId"])?;
    if condition_id != market.condition_id {
        bail!(
            "CLOB condition {} does not match stored condition {}",
            condition_id,
            market.condition_id
        );
    }
    let closed = bool_field(object, &["closed"]).context("CLOB market is missing closed status")?;
    let tokens = object
        .get("tokens")
        .and_then(serde_json::Value::as_array)
        .context("CLOB market is missing its tokens array")?;
    if tokens.len() != 2 {
        bail!("CLOB BTC interval market must contain exactly two tokens");
    }

    let mut parsed = Vec::with_capacity(2);
    let mut saw_up = false;
    let mut saw_down = false;
    for token in tokens {
        let token = token
            .as_object()
            .context("CLOB market token must be an object")?;
        let token_id = required_string(token, &["token_id", "tokenId"])?;
        let outcome = normalize_outcome(&required_string(token, &["outcome"])?)?;
        let expected_token_id = market.token_id(outcome);
        if token_id != expected_token_id {
            bail!(
                "CLOB token {} does not match stored {:?} token {}",
                token_id,
                outcome,
                expected_token_id
            );
        }
        match outcome {
            BtcOutcome::Up if saw_up => bail!("CLOB market contains duplicate Up tokens"),
            BtcOutcome::Down if saw_down => bail!("CLOB market contains duplicate Down tokens"),
            BtcOutcome::Up => saw_up = true,
            BtcOutcome::Down => saw_down = true,
        }
        parsed.push((
            token_id,
            outcome,
            bool_field(token, &["winner"]),
            decimal_field(token, &["price"]),
        ));
    }
    if !saw_up || !saw_down {
        bail!("CLOB market does not contain one Up and one Down token");
    }
    if !closed {
        return Ok(None);
    }

    let winners = parsed
        .iter()
        .filter(|(_, _, winner, _)| *winner == Some(true))
        .collect::<Vec<_>>();
    if winners.len() != 1 {
        bail!("closed CLOB market must expose exactly one winner");
    }
    for (token_id, _, winner, price) in &parsed {
        let winner = winner.context("closed CLOB market token is missing winner status")?;
        let price = price.context("closed CLOB market token is missing terminal price")?;
        let expected_price = if winner { Decimal::ONE } else { Decimal::ZERO };
        if price != expected_price {
            bail!(
                "closed CLOB token {} has terminal price {}, expected {}",
                token_id,
                price,
                expected_price
            );
        }
    }
    let (winning_token_id, winning_outcome, _, _) = winners[0];
    Ok(Some(ClobRestOfficialResolution {
        market_id: market.market_id.clone(),
        winning_token_id: winning_token_id.clone(),
        winning_outcome: *winning_outcome,
        observed_at: DateTime::from_timestamp_micros(observed_at.timestamp_micros())
            .expect("a valid observation timestamp remains valid at microsecond precision"),
        raw_payload: value.clone(),
    }))
}

fn normalize_outcome(value: &str) -> Result<BtcOutcome> {
    match value.trim().to_ascii_lowercase().as_str() {
        "up" => Ok(BtcOutcome::Up),
        "down" => Ok(BtcOutcome::Down),
        other => bail!("unexpected BTC interval outcome {other}"),
    }
}

fn is_chainlink_btc_usd_source(source: &str) -> bool {
    let normalized = source.trim().to_ascii_lowercase();
    (normalized.contains("chainlink") || normalized.contains("chain.link"))
        && (normalized.contains("btc-usd")
            || normalized.contains("btc/usd")
            || (normalized.contains("btc") && normalized.contains("usd")))
}

fn series_slug_from_relation(
    object: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    object
        .get("series")?
        .as_array()?
        .iter()
        .filter_map(serde_json::Value::as_object)
        .find_map(|series| string_field(series, &["slug"]))
}

fn required_string(
    object: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Result<String> {
    string_field(object, keys).with_context(|| format!("missing required field {}", keys[0]))
}

fn string_field(
    object: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Option<String> {
    keys.iter().find_map(|key| match object.get(*key)? {
        serde_json::Value::String(value) => {
            let value = value.trim();
            (!value.is_empty()).then(|| value.to_string())
        }
        serde_json::Value::Number(value) => Some(value.to_string()),
        _ => None,
    })
}

fn string_array_field(
    object: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Option<Vec<String>> {
    let value = keys.iter().find_map(|key| object.get(*key))?;
    let array = match value {
        serde_json::Value::Array(values) => values.clone(),
        serde_json::Value::String(value) => serde_json::from_str(value).ok()?,
        _ => return None,
    };
    array
        .iter()
        .map(|value| match value {
            serde_json::Value::String(value) if !value.trim().is_empty() => {
                Some(value.trim().to_string())
            }
            serde_json::Value::Number(value) => Some(value.to_string()),
            _ => None,
        })
        .collect()
}

fn decimal_field(
    object: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Option<Decimal> {
    let raw = keys.iter().find_map(|key| object.get(*key))?;
    match raw {
        serde_json::Value::String(value) => Decimal::from_str(value).ok(),
        serde_json::Value::Number(value) => Decimal::from_str(&value.to_string()).ok(),
        _ => None,
    }
}

fn bool_field(object: &serde_json::Map<String, serde_json::Value>, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .find_map(|key| object.get(*key).and_then(serde_json::Value::as_bool))
}

fn datetime_field(
    object: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Option<DateTime<Utc>> {
    let value = string_field(object, keys)?;
    DateTime::parse_from_rfc3339(&value)
        .ok()
        .map(|value| value.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use rust_decimal_macros::dec;

    use super::*;

    fn start() -> DateTime<Utc> {
        Utc.timestamp_opt(1_783_902_600, 0).unwrap()
    }

    fn valid_event() -> serde_json::Value {
        serde_json::json!({
            "id": "event-1",
            "slug": "btc-updown-5m-1783902600",
            "seriesSlug": "btc-up-or-down-5m",
            "eventStartTime": "2026-07-13T00:30:00Z",
            "endDate": "2026-07-13T00:35:00Z",
            "active": true,
            "closed": false,
            "resolutionSource": "https://data.chain.link/streams/btc-usd",
            "feesEnabled": true,
            "feeSchedule": {"rate": "0.02", "exponent": "2"},
            "markets": [{
                "id": "market-1",
                "conditionId": "0xcondition",
                "active": true,
                "closed": false,
                "acceptingOrders": true,
                "endDate": "2026-07-13T00:35:00Z",
                "outcomes": "[\"Down\",\"Up\"]",
                "clobTokenIds": "[\"down-token\",\"up-token\"]",
                "orderPriceMinTickSize": "0.01",
                "orderMinSize": 5
            }]
        })
    }

    #[test]
    fn aligns_boundaries_using_utc_epoch() {
        let inside = Utc.timestamp_opt(1_783_902_731, 999_000_000).unwrap();
        assert_eq!(aligned_window_start(inside), start());
        assert_eq!(slug_for_window(start()), "btc-updown-5m-1783902600");
        assert_eq!(discovery_windows(inside)[1], start());
    }

    #[test]
    fn rejects_unaligned_slug_epochs() {
        assert!(window_start_from_slug("btc-updown-5m-1783902601").is_err());
        assert!(window_start_from_slug("eth-updown-5m-1783902600").is_err());
    }

    #[test]
    fn maps_up_and_down_by_label_not_array_position() {
        let market = parse_gamma_btc_interval_event(&valid_event(), start()).unwrap();
        assert_eq!(market.up_token_id, "up-token");
        assert_eq!(market.down_token_id, "down-token");
        assert_eq!(market.tick_size, dec!(0.01));
        assert_eq!(market.minimum_order_size, Some(dec!(5)));
        assert!(market.is_trade_window(start() + Duration::seconds(1)));
    }

    #[test]
    fn accepts_series_relation_and_json_arrays() {
        let mut event = valid_event();
        event.as_object_mut().unwrap().remove("seriesSlug");
        event["series"] = serde_json::json!([{"slug": "btc-up-or-down-5m"}]);
        event["markets"][0]["outcomes"] = serde_json::json!(["Up", "Down"]);
        event["markets"][0]["clobTokenIds"] = serde_json::json!(["up", "down"]);
        let market = parse_gamma_btc_interval_event(&event, start()).unwrap();
        assert_eq!(market.up_token_id, "up");
        assert_eq!(market.down_token_id, "down");
    }

    #[test]
    fn rejects_wrong_window_series_source_and_outcomes() {
        let mut wrong_window = valid_event();
        wrong_window["markets"][0]["endDate"] = serde_json::json!("2026-07-13T00:40:00Z");
        assert!(parse_gamma_btc_interval_event(&wrong_window, start()).is_err());

        let mut wrong_series = valid_event();
        wrong_series["seriesSlug"] = serde_json::json!("eth-up-or-down-5m");
        assert!(parse_gamma_btc_interval_event(&wrong_series, start()).is_err());

        let mut wrong_source = valid_event();
        wrong_source["resolutionSource"] = serde_json::json!("Binance BTC/USDT");
        assert!(parse_gamma_btc_interval_event(&wrong_source, start()).is_err());

        let mut wrong_outcomes = valid_event();
        wrong_outcomes["markets"][0]["outcomes"] = serde_json::json!(["Yes", "No"]);
        assert!(parse_gamma_btc_interval_event(&wrong_outcomes, start()).is_err());
    }

    #[test]
    fn rejects_requested_window_mismatch() {
        let next = start() + Duration::seconds(BTC_INTERVAL_SECONDS);
        assert!(parse_gamma_btc_interval_event(&valid_event(), next).is_err());
    }

    #[test]
    fn parses_closed_clob_resolution_by_token_identity() {
        let market = parse_gamma_btc_interval_event(&valid_event(), start()).unwrap();
        let value = serde_json::json!({
            "condition_id": "0xcondition",
            "closed": true,
            "tokens": [
                {"token_id": "up-token", "outcome": "Up", "price": "1", "winner": true},
                {"token_id": "down-token", "outcome": "Down", "price": "0", "winner": false}
            ]
        });
        let resolution = parse_clob_rest_official_resolution(
            &value,
            &market,
            market.window_end + Duration::seconds(30),
        )
        .unwrap()
        .unwrap();
        assert_eq!(resolution.market_id, market.market_id);
        assert_eq!(resolution.winning_token_id, "up-token");
        assert_eq!(resolution.winning_outcome, BtcOutcome::Up);
    }

    #[test]
    fn keeps_open_clob_market_pending() {
        let market = parse_gamma_btc_interval_event(&valid_event(), start()).unwrap();
        let value = serde_json::json!({
            "condition_id": "0xcondition",
            "closed": false,
            "tokens": [
                {"token_id": "down-token", "outcome": "Down"},
                {"token_id": "up-token", "outcome": "Up"}
            ]
        });
        assert!(
            parse_clob_rest_official_resolution(&value, &market, market.window_end)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn rejects_ambiguous_or_mismatched_clob_resolution() {
        let market = parse_gamma_btc_interval_event(&valid_event(), start()).unwrap();
        let cases = [
            serde_json::json!({
                "condition_id": "wrong",
                "closed": true,
                "tokens": [
                    {"token_id": "up-token", "outcome": "Up", "price": "1", "winner": true},
                    {"token_id": "down-token", "outcome": "Down", "price": "0", "winner": false}
                ]
            }),
            serde_json::json!({
                "condition_id": "0xcondition",
                "closed": true,
                "tokens": [
                    {"token_id": "up-token", "outcome": "Up", "price": "1", "winner": true},
                    {"token_id": "down-token", "outcome": "Down", "price": "1", "winner": true}
                ]
            }),
            serde_json::json!({
                "condition_id": "0xcondition",
                "closed": true,
                "tokens": [
                    {"token_id": "unknown", "outcome": "Up", "price": "1", "winner": true},
                    {"token_id": "down-token", "outcome": "Down", "price": "0", "winner": false}
                ]
            }),
        ];
        for value in cases {
            assert!(
                parse_clob_rest_official_resolution(&value, &market, market.window_end).is_err()
            );
        }
    }
}
