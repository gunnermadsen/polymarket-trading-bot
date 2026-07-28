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

#[derive(Debug, Clone, PartialEq)]
pub struct GammaRestOfficialResolution {
    pub market_id: String,
    pub winning_token_id: String,
    pub winning_outcome: BtcOutcome,
    pub source_timestamp: DateTime<Utc>,
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

/// Validates terminal Gamma evidence against the immutable market identity captured at discovery.
///
/// Gamma remains non-authoritative while both the event and market are open. Once either claims a
/// terminal state, every terminal marker must agree and the outcome prices must be an exact binary
/// payout. This prevents a stale tradable quote or partially updated Gamma response from becoming
/// an official settlement fact.
pub fn parse_gamma_rest_official_resolution(
    value: &serde_json::Value,
    market: &BtcIntervalMarket,
    observed_at: DateTime<Utc>,
) -> Result<Option<GammaRestOfficialResolution>> {
    let candidate = parse_gamma_btc_interval_event(value, market.window_start)?;
    validate_gamma_market_identity(&candidate, market)?;

    let event = value
        .as_object()
        .context("Gamma event response must be an object")?;
    let markets = event
        .get("markets")
        .and_then(serde_json::Value::as_array)
        .context("Gamma event is missing its markets array")?;
    let gamma_market = markets[0]
        .as_object()
        .context("Gamma event market must be an object")?;
    let event_closed =
        bool_field(event, &["closed"]).context("Gamma event is missing closed status")?;
    let market_closed =
        bool_field(gamma_market, &["closed"]).context("Gamma market is missing closed status")?;
    let resolution_status = string_field(
        gamma_market,
        &[
            "umaResolutionStatus",
            "uma_resolution_status",
            "resolutionStatus",
            "resolution_status",
        ],
    );
    let has_resolution_timestamp = ["umaEndDate", "uma_end_date", "closedTime", "closed_time"]
        .iter()
        .any(|key| gamma_market.contains_key(*key))
        || ["closedTime", "closed_time"]
            .iter()
            .any(|key| event.contains_key(*key));

    match (event_closed, market_closed) {
        (false, false)
            if resolution_status
                .as_deref()
                .is_none_or(|status| !status.eq_ignore_ascii_case("resolved"))
                && !has_resolution_timestamp =>
        {
            return Ok(None);
        }
        (false, false) => bail!("open Gamma market contains terminal resolution evidence"),
        (true, true) => {}
        _ => bail!("Gamma event and market closed states disagree"),
    }
    if bool_field(gamma_market, &["acceptingOrders", "accepting_orders"])
        .context("closed Gamma market is missing accepting-orders status")?
    {
        bail!("closed Gamma market is still accepting orders");
    }

    let resolution_status =
        resolution_status.context("closed Gamma market is missing resolution status")?;
    if !resolution_status.eq_ignore_ascii_case("resolved") {
        bail!("closed Gamma market has non-terminal resolution status {resolution_status}");
    }

    let resolution_times = [
        strict_optional_datetime_field(gamma_market, &["umaEndDate", "uma_end_date"])?,
        strict_optional_datetime_field(gamma_market, &["closedTime", "closed_time"])?,
        strict_optional_datetime_field(event, &["closedTime", "closed_time"])?,
    ];
    let source_timestamp = resolution_times
        .iter()
        .flatten()
        .next()
        .copied()
        .context("resolved Gamma market is missing its resolution timestamp")?;
    if resolution_times
        .iter()
        .flatten()
        .any(|timestamp| *timestamp != source_timestamp)
    {
        bail!("Gamma resolution timestamps disagree");
    }
    if source_timestamp < market.window_end || observed_at < market.window_end {
        bail!("Gamma official resolution predates market close");
    }

    let outcomes = string_array_field(gamma_market, &["outcomes"])
        .context("resolved Gamma market is missing outcomes")?;
    let token_ids = string_array_field(
        gamma_market,
        &["clobTokenIds", "clob_token_ids", "tokenIds", "token_ids"],
    )
    .context("resolved Gamma market is missing CLOB token IDs")?;
    let outcome_prices = string_array_field(gamma_market, &["outcomePrices", "outcome_prices"])
        .context("resolved Gamma market is missing outcome prices")?;
    if outcomes.len() != 2
        || token_ids.len() != 2
        || outcome_prices.len() != 2
        || outcomes.len() != token_ids.len()
        || outcomes.len() != outcome_prices.len()
    {
        bail!("resolved Gamma market must contain exactly two outcomes, token IDs, and prices");
    }

    let mut winner = None;
    for ((outcome, token_id), price) in outcomes.iter().zip(&token_ids).zip(&outcome_prices) {
        let outcome = normalize_outcome(outcome)?;
        let expected_token_id = market.token_id(outcome);
        if token_id != expected_token_id {
            bail!(
                "Gamma token {} does not match stored {:?} token {}",
                token_id,
                outcome,
                expected_token_id
            );
        }
        let price = Decimal::from_str(price)
            .with_context(|| format!("Gamma outcome {outcome:?} has an invalid terminal price"))?;
        if price == Decimal::ONE {
            if winner.replace((token_id.clone(), outcome)).is_some() {
                bail!("resolved Gamma market exposes multiple winning outcomes");
            }
        } else if price != Decimal::ZERO {
            bail!("resolved Gamma outcome {outcome:?} has non-binary terminal price {price}");
        }
    }
    let (winning_token_id, winning_outcome) =
        winner.context("resolved Gamma market does not expose a winning outcome")?;

    Ok(Some(GammaRestOfficialResolution {
        market_id: market.market_id.clone(),
        winning_token_id,
        winning_outcome,
        source_timestamp: microsecond_timestamp(source_timestamp),
        observed_at: microsecond_timestamp(observed_at),
        raw_payload: value.clone(),
    }))
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

fn validate_gamma_market_identity(
    candidate: &BtcIntervalMarket,
    stored: &BtcIntervalMarket,
) -> Result<()> {
    for (name, candidate, stored) in [
        ("event id", &candidate.event_id, &stored.event_id),
        ("event slug", &candidate.event_slug, &stored.event_slug),
        ("series slug", &candidate.series_slug, &stored.series_slug),
        ("market id", &candidate.market_id, &stored.market_id),
        (
            "condition id",
            &candidate.condition_id,
            &stored.condition_id,
        ),
        ("Up token id", &candidate.up_token_id, &stored.up_token_id),
        (
            "Down token id",
            &candidate.down_token_id,
            &stored.down_token_id,
        ),
        (
            "resolution source",
            &candidate.resolution_source,
            &stored.resolution_source,
        ),
    ] {
        if candidate != stored {
            bail!("Gamma {name} {candidate} does not match stored {name} {stored}");
        }
    }
    if candidate.window_start != stored.window_start || candidate.window_end != stored.window_end {
        bail!("Gamma market window does not match its stored window");
    }
    Ok(())
}

fn microsecond_timestamp(timestamp: DateTime<Utc>) -> DateTime<Utc> {
    DateTime::from_timestamp_micros(timestamp.timestamp_micros())
        .expect("a valid timestamp remains valid at microsecond precision")
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

fn strict_optional_datetime_field(
    object: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Result<Option<DateTime<Utc>>> {
    let Some(key) = keys.iter().find(|key| object.contains_key(**key)) else {
        return Ok(None);
    };
    let value = required_string(object, &[*key])?;
    let normalized = normalize_gamma_datetime_offset(&value);
    let timestamp = DateTime::parse_from_rfc3339(&normalized)
        .with_context(|| format!("Gamma field {key} is not a valid RFC3339 timestamp"))?
        .with_timezone(&Utc);
    Ok(Some(timestamp))
}

fn normalize_gamma_datetime_offset(value: &str) -> String {
    let bytes = value.as_bytes();
    if bytes.len() >= 3 {
        let offset = &bytes[bytes.len() - 3..];
        if matches!(offset[0], b'+' | b'-')
            && offset[1].is_ascii_digit()
            && offset[2].is_ascii_digit()
        {
            return format!("{value}:00");
        }
    }
    value.to_string()
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

    fn resolved_event(up_wins: bool) -> serde_json::Value {
        let mut event = valid_event();
        event["closed"] = serde_json::json!(true);
        event["closedTime"] = serde_json::json!("2026-07-13T00:35:19Z");
        event["markets"][0]["closed"] = serde_json::json!(true);
        event["markets"][0]["acceptingOrders"] = serde_json::json!(false);
        event["markets"][0]["umaResolutionStatus"] = serde_json::json!("resolved");
        event["markets"][0]["umaEndDate"] = serde_json::json!("2026-07-13T00:35:19Z");
        // Gamma's market-level timestamp uses a space and `+00`; the event and UMA fields use
        // the equivalent RFC3339 `T`/`Z` representation.
        event["markets"][0]["closedTime"] = serde_json::json!("2026-07-13 00:35:19+00");
        event["markets"][0]["outcomePrices"] = if up_wins {
            serde_json::json!("[\"0\", \"1\"]")
        } else {
            serde_json::json!("[\"1\", \"0\"]")
        };
        event
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
    fn parses_strict_gamma_resolution_by_outcome_and_token_identity() {
        let market = parse_gamma_btc_interval_event(&valid_event(), start()).unwrap();
        let observed_at = market.window_end + Duration::minutes(1);

        let up = parse_gamma_rest_official_resolution(&resolved_event(true), &market, observed_at)
            .unwrap()
            .unwrap();
        assert_eq!(up.market_id, market.market_id);
        assert_eq!(up.winning_token_id, "up-token");
        assert_eq!(up.winning_outcome, BtcOutcome::Up);
        assert_eq!(
            up.source_timestamp,
            market.window_end + Duration::seconds(19)
        );
        assert_eq!(up.observed_at, observed_at);

        let down =
            parse_gamma_rest_official_resolution(&resolved_event(false), &market, observed_at)
                .unwrap()
                .unwrap();
        assert_eq!(down.winning_token_id, "down-token");
        assert_eq!(down.winning_outcome, BtcOutcome::Down);
    }

    #[test]
    fn keeps_clearly_open_gamma_market_pending() {
        let market = parse_gamma_btc_interval_event(&valid_event(), start()).unwrap();
        let mut event = valid_event();
        event["markets"][0]["outcomePrices"] = serde_json::json!("[\"0.995\", \"0.005\"]");
        assert!(
            parse_gamma_rest_official_resolution(&event, &market, market.window_end)
                .unwrap()
                .is_none()
        );

        event["markets"][0]["acceptingOrders"] = serde_json::json!(false);
        assert!(
            parse_gamma_rest_official_resolution(&event, &market, market.window_end)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn rejects_gamma_resolution_identity_conflicts() {
        let market = parse_gamma_btc_interval_event(&valid_event(), start()).unwrap();
        let observed_at = market.window_end + Duration::minutes(1);
        let cases = [
            {
                let mut value = resolved_event(true);
                value["id"] = serde_json::json!("different-event");
                value
            },
            {
                let mut value = resolved_event(true);
                value["markets"][0]["id"] = serde_json::json!("different-market");
                value
            },
            {
                let mut value = resolved_event(true);
                value["markets"][0]["conditionId"] = serde_json::json!("different-condition");
                value
            },
            {
                let mut value = resolved_event(true);
                value["markets"][0]["clobTokenIds"] =
                    serde_json::json!("[\"down-token\", \"different-up-token\"]");
                value
            },
            {
                let mut value = resolved_event(true);
                value["markets"][0]["resolutionSource"] =
                    serde_json::json!("https://data.chain.link/streams/alternate-btc-usd");
                value
            },
        ];
        for value in cases {
            assert!(
                parse_gamma_rest_official_resolution(&value, &market, observed_at).is_err(),
                "identity conflict was accepted: {value}"
            );
        }
    }

    #[test]
    fn rejects_ambiguous_or_incomplete_gamma_terminal_evidence() {
        let market = parse_gamma_btc_interval_event(&valid_event(), start()).unwrap();
        let observed_at = market.window_end + Duration::minutes(1);
        let cases = [
            {
                let mut value = resolved_event(true);
                value["closed"] = serde_json::json!(false);
                value
            },
            {
                let mut value = resolved_event(true);
                value["markets"][0]["acceptingOrders"] = serde_json::json!(true);
                value
            },
            {
                let mut value = resolved_event(true);
                value["markets"][0]["umaResolutionStatus"] = serde_json::json!("proposed");
                value
            },
            {
                let mut value = resolved_event(true);
                value["markets"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("umaResolutionStatus");
                value
            },
            {
                let mut value = resolved_event(true);
                value["markets"][0]["outcomePrices"] = serde_json::json!("[\"0.50\", \"0.50\"]");
                value
            },
            {
                let mut value = resolved_event(true);
                value["markets"][0]["outcomePrices"] = serde_json::json!("[\"1\", \"1\"]");
                value
            },
            {
                let mut value = resolved_event(true);
                value["markets"][0]["umaEndDate"] = serde_json::json!("2026-07-13T00:34:59Z");
                value["markets"][0]["closedTime"] = serde_json::json!("2026-07-13T00:34:59Z");
                value["closedTime"] = serde_json::json!("2026-07-13T00:34:59Z");
                value
            },
            {
                let mut value = resolved_event(true);
                value.as_object_mut().unwrap().remove("closedTime");
                let gamma_market = value["markets"][0].as_object_mut().unwrap();
                gamma_market.remove("umaEndDate");
                gamma_market.remove("closedTime");
                value
            },
            {
                let mut value = resolved_event(true);
                value["closedTime"] = serde_json::json!("2026-07-13T00:35:20Z");
                value
            },
            {
                let mut value = resolved_event(true);
                value["closed"] = serde_json::json!(false);
                value["markets"][0]["closed"] = serde_json::json!(false);
                value
            },
        ];
        for value in cases {
            assert!(
                parse_gamma_rest_official_resolution(&value, &market, observed_at).is_err(),
                "ambiguous Gamma terminal evidence was accepted: {value}"
            );
        }
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
