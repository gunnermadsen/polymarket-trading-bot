use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;

use crate::{
    models::{GammaMarketMetadata, Market, OutcomeToken, TokenSide},
    risk::normalize_underlying_key,
    taxonomy::{metadata_from_gamma_event, metadata_from_gamma_market},
};

#[derive(Debug, Clone)]
pub struct GammaClient {
    http: Client,
    base_url: String,
}

#[derive(Debug, Deserialize)]
struct GammaEvent {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    slug: Option<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    active: Option<bool>,
    #[serde(default)]
    closed: Option<bool>,
    #[serde(default)]
    archived: Option<bool>,
    #[serde(default, rename = "negRisk")]
    neg_risk: Option<bool>,
    #[serde(default, rename = "negRiskAugmented")]
    neg_risk_augmented: Option<bool>,
    #[serde(default)]
    markets: Vec<GammaMarket>,
    #[serde(flatten)]
    raw_extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct GammaMarket {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    question: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    active: Option<bool>,
    #[serde(default)]
    closed: Option<bool>,
    #[serde(default)]
    archived: Option<bool>,
    #[serde(default, rename = "negRisk")]
    neg_risk: Option<bool>,
    #[serde(default, rename = "negRiskAugmented")]
    neg_risk_augmented: Option<bool>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default, rename = "endDate")]
    end_date: Option<DateTime<Utc>>,
    #[serde(flatten)]
    raw_extra: serde_json::Map<String, serde_json::Value>,
}

impl GammaClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }

    pub async fn fetch_active_events(&self, limit: usize) -> Result<Vec<Market>> {
        let url = format!(
            "{}/events?active=true&closed=false&archived=false&limit={}",
            self.base_url, limit
        );
        let events: Vec<GammaEvent> = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("failed to request Gamma events from {url}"))?
            .error_for_status()
            .context("Gamma events response was not successful")?
            .json()
            .await
            .context("failed to decode Gamma events")?;

        Ok(markets_from_gamma_events(events))
    }

    pub async fn fetch_event_taxonomy_by_slug(
        &self,
        event_slug: &str,
    ) -> Result<Option<GammaMarketMetadata>> {
        let slug = event_slug.trim();
        if slug.is_empty() {
            return Ok(None);
        }
        let mut url = reqwest::Url::parse(&format!("{}/events", self.base_url))
            .context("failed to build Gamma events URL")?;
        url.query_pairs_mut()
            .append_pair("slug", slug)
            .append_pair("limit", "1");
        let events: Vec<serde_json::Value> = self
            .http
            .get(url.clone())
            .send()
            .await
            .with_context(|| format!("failed to request Gamma event taxonomy from {url}"))?
            .error_for_status()
            .context("Gamma event taxonomy response was not successful")?
            .json()
            .await
            .context("failed to decode Gamma event taxonomy")?;
        Ok(events
            .into_iter()
            .next()
            .and_then(|event| metadata_from_gamma_event(slug, &event)))
    }

    pub async fn fetch_market_taxonomy_by_slug(
        &self,
        market_slug: &str,
    ) -> Result<Option<GammaMarketMetadata>> {
        let slug = market_slug.trim();
        if slug.is_empty() {
            return Ok(None);
        }
        let mut url = reqwest::Url::parse(&format!("{}/markets", self.base_url))
            .context("failed to build Gamma markets URL")?;
        url.query_pairs_mut()
            .append_pair("slug", slug)
            .append_pair("limit", "1");
        let markets: Vec<serde_json::Value> = self
            .http
            .get(url.clone())
            .send()
            .await
            .with_context(|| format!("failed to request Gamma market taxonomy from {url}"))?
            .error_for_status()
            .context("Gamma market taxonomy response was not successful")?
            .json()
            .await
            .context("failed to decode Gamma market taxonomy")?;
        Ok(markets
            .into_iter()
            .next()
            .and_then(|market| metadata_from_gamma_market(slug, &market)))
    }
}

fn markets_from_gamma_events(events: Vec<GammaEvent>) -> Vec<Market> {
    let mut markets = Vec::new();
    for event in events {
        let event_id = event
            .id
            .clone()
            .or(event.slug.clone())
            .unwrap_or_else(|| "unknown-event".to_string());
        let outcome_group_id = complete_event_group_id(&event);
        for market in &event.markets {
            let market_id = market
                .id
                .clone()
                .unwrap_or_else(|| format!("{}:unknown-market", event_id));
            let question = market
                .question
                .clone()
                .or(event.title.clone())
                .unwrap_or_else(|| market_id.clone());
            let category = market.category.clone().or(event.category.clone());
            let raw = serde_json::json!({
                "event": event.raw_extra,
                "market": market.raw_extra,
            });
            let neg_risk = market.neg_risk.or(event.neg_risk).unwrap_or(false);
            markets.push(Market {
                event_id: event_id.clone(),
                market_id: market_id.clone(),
                outcome_group_id: outcome_group_id.clone(),
                question: question.clone(),
                category: category.clone(),
                active: market.active.or(event.active).unwrap_or(false),
                closed: market.closed.or(event.closed).unwrap_or(false),
                archived: market.archived.or(event.archived).unwrap_or(false),
                neg_risk,
                neg_risk_augmented: market
                    .neg_risk_augmented
                    .or(event.neg_risk_augmented)
                    .unwrap_or(false),
                rules: market.description.clone(),
                end_date: market.end_date,
                underlying_key: normalize_underlying_key(None, category.as_deref(), &question),
                resolution_score: resolution_score(market.description.as_deref()),
                outcome_tokens: outcome_tokens_from_market(&market_id, market, neg_risk),
                raw,
            });
        }
    }
    markets
}

fn outcome_tokens_from_market(
    market_id: &str,
    market: &GammaMarket,
    neg_risk: bool,
) -> Vec<OutcomeToken> {
    let tick_size = decimal_field(
        &market.raw_extra,
        &[
            "orderPriceMinTickSize",
            "minimumTickSize",
            "tickSize",
            "tick_size",
        ],
    )
    .unwrap_or(dec!(0.01));
    let condition_id = string_field(
        &market.raw_extra,
        &[
            "conditionId",
            "condition_id",
            "conditionID",
            "questionID",
            "questionId",
        ],
    );

    let mut tokens = tokens_from_objects(
        market_id,
        market,
        tick_size,
        condition_id.as_deref(),
        neg_risk,
    );
    if tokens.is_empty() {
        let token_ids = string_vec_field(
            &market.raw_extra,
            &[
                "clobTokenIds",
                "clob_token_ids",
                "tokenIds",
                "token_ids",
                "tokens",
            ],
        );
        let outcomes = string_vec_field(&market.raw_extra, &["outcomes", "outcomeNames"]);
        tokens = token_ids
            .into_iter()
            .enumerate()
            .map(|(index, token_id)| {
                let outcome = outcomes
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| default_outcome(index));
                OutcomeToken {
                    market_id: market_id.to_string(),
                    token_id,
                    outcome: outcome.clone(),
                    side: token_side(&outcome, index),
                    condition_id: condition_id.clone(),
                    tick_size,
                    neg_risk,
                }
            })
            .collect();
    }

    tokens
}

fn tokens_from_objects(
    market_id: &str,
    market: &GammaMarket,
    default_tick_size: Decimal,
    default_condition_id: Option<&str>,
    neg_risk: bool,
) -> Vec<OutcomeToken> {
    let Some(value) = first_value(&market.raw_extra, &["tokens", "outcomeTokens"]) else {
        return Vec::new();
    };
    let Some(items) = array_value(value) else {
        return Vec::new();
    };

    items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            let object = item.as_object()?;
            let token_id = string_from_value(first_value(
                object,
                &["token_id", "tokenId", "clobTokenId", "clob_token_id", "id"],
            )?)?;
            let outcome = first_value(object, &["outcome", "name", "label"])
                .and_then(string_from_value)
                .unwrap_or_else(|| default_outcome(index));
            let condition_id = first_value(object, &["condition_id", "conditionId"])
                .and_then(string_from_value)
                .or_else(|| default_condition_id.map(ToString::to_string));
            let tick_size = first_value(object, &["tick_size", "tickSize"])
                .and_then(decimal_from_value)
                .unwrap_or(default_tick_size);

            Some(OutcomeToken {
                market_id: market_id.to_string(),
                token_id,
                outcome: outcome.clone(),
                side: token_side(&outcome, index),
                condition_id,
                tick_size,
                neg_risk,
            })
        })
        .collect()
}

fn complete_event_group_id(event: &GammaEvent) -> Option<String> {
    let event_id = string_field(&event.raw_extra, &["negRiskMarketID", "negRiskMarketId"])
        .or_else(|| event.id.clone())
        .or_else(|| event.slug.clone())?;
    let active_markets: Vec<&GammaMarket> = event
        .markets
        .iter()
        .filter(|market| {
            market.active.or(event.active).unwrap_or(false)
                && !market.closed.or(event.closed).unwrap_or(false)
                && !market.archived.or(event.archived).unwrap_or(false)
        })
        .collect();
    if active_markets.len() < 2 {
        return None;
    }

    active_markets
        .iter()
        .all(|market| outcome_tokens_from_market("group-check", market, false).len() >= 2)
        .then(|| event_id.to_string())
}

fn string_vec_field(
    object: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Vec<String> {
    first_value(object, keys)
        .and_then(array_value)
        .map(|items| items.iter().filter_map(string_from_value).collect())
        .unwrap_or_default()
}

fn string_field(
    object: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Option<String> {
    first_value(object, keys).and_then(string_from_value)
}

fn decimal_field(
    object: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Option<Decimal> {
    first_value(object, keys).and_then(decimal_from_value)
}

fn first_value<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Option<&'a serde_json::Value> {
    keys.iter().find_map(|key| object.get(*key))
}

fn array_value(value: &serde_json::Value) -> Option<Vec<serde_json::Value>> {
    match value {
        serde_json::Value::Array(items) => Some(items.clone()),
        serde_json::Value::String(text) => {
            let parsed: serde_json::Value = serde_json::from_str(text).ok()?;
            match parsed {
                serde_json::Value::Array(items) => Some(items),
                _ => None,
            }
        }
        _ => None,
    }
}

fn string_from_value(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => {
            let trimmed = text.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        serde_json::Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

fn decimal_from_value(value: &serde_json::Value) -> Option<Decimal> {
    match value {
        serde_json::Value::String(text) => text.parse().ok(),
        serde_json::Value::Number(number) => number.to_string().parse().ok(),
        _ => None,
    }
}

fn default_outcome(index: usize) -> String {
    match index {
        0 => "Yes".to_string(),
        1 => "No".to_string(),
        _ => format!("Outcome {index}"),
    }
}

fn token_side(outcome: &str, index: usize) -> TokenSide {
    match outcome.trim().to_ascii_lowercase().as_str() {
        "no" => TokenSide::No,
        "yes" => TokenSide::Yes,
        _ if index == 1 => TokenSide::No,
        _ => TokenSide::Yes,
    }
}

fn resolution_score(rules: Option<&str>) -> i32 {
    let text = rules.unwrap_or("").to_ascii_lowercase();
    let mut score = 0;
    if [
        "ap",
        "associated press",
        "official",
        "fifa",
        "sec",
        "court",
        "election",
    ]
    .iter()
    .any(|needle| text.contains(needle))
    {
        score += 1;
    }
    if !text.contains("generally accepted") && !text.contains("unclear") && !text.is_empty() {
        score += 1;
    }
    // UMA dispute and historical-clean checks need persisted external data; default to conservative partial credit.
    score += 1;
    score += 1;
    if !text.contains("indefinite") {
        score += 1;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_json_string_token_arrays() {
        let events: Vec<GammaEvent> = serde_json::from_str(
            r#"
            [{
              "id": "event-1",
              "title": "Election winner",
              "active": true,
              "closed": false,
              "archived": false,
              "negRisk": true,
              "markets": [{
                "id": "market-1",
                "question": "Will Alice win?",
                "active": true,
                "closed": false,
                "archived": false,
                "conditionId": "0xabc",
                "clobTokenIds": "[\"101\", \"102\"]",
                "outcomes": "[\"Yes\", \"No\"]",
                "tickSize": "0.001"
              }, {
                "id": "market-2",
                "question": "Will Bob win?",
                "active": true,
                "closed": false,
                "archived": false,
                "conditionId": "0xdef",
                "clobTokenIds": "[\"201\", \"202\"]",
                "outcomes": "[\"Yes\", \"No\"]",
                "tickSize": "0.01"
              }]
            }]
            "#,
        )
        .unwrap();

        let markets = markets_from_gamma_events(events);

        assert_eq!(markets.len(), 2);
        assert_eq!(markets[0].outcome_group_id.as_deref(), Some("event-1"));
        assert_eq!(markets[0].outcome_tokens.len(), 2);
        assert_eq!(markets[0].outcome_tokens[0].token_id, "101");
        assert_eq!(markets[0].outcome_tokens[0].outcome, "Yes");
        assert_eq!(markets[0].outcome_tokens[0].side, TokenSide::Yes);
        assert_eq!(
            markets[0].outcome_tokens[0].condition_id.as_deref(),
            Some("0xabc")
        );
        assert_eq!(markets[0].outcome_tokens[0].tick_size, dec!(0.001));
        assert_eq!(markets[0].outcome_tokens[1].side, TokenSide::No);
    }

    #[test]
    fn parses_object_token_arrays_and_defaults_tick_size() {
        let events: Vec<GammaEvent> = serde_json::from_str(
            r#"
            [{
              "slug": "event-2",
              "title": "Tournament winner",
              "active": true,
              "markets": [{
                "id": "market-3",
                "question": "Will Carol win?",
                "active": true,
                "tokens": [
                  {"token_id": "301", "outcome": "Carol", "condition_id": "0xaaa"},
                  {"tokenId": "302", "outcome": "No", "tickSize": 0.005}
                ]
              }, {
                "id": "market-4",
                "question": "Will Dan win?",
                "active": true,
                "clobTokenIds": ["401", "402"],
                "outcomes": ["Yes", "No"]
              }]
            }]
            "#,
        )
        .unwrap();

        let markets = markets_from_gamma_events(events);

        assert_eq!(markets.len(), 2);
        assert_eq!(markets[0].outcome_group_id.as_deref(), Some("event-2"));
        assert_eq!(markets[0].outcome_tokens[0].token_id, "301");
        assert_eq!(markets[0].outcome_tokens[0].outcome, "Carol");
        assert_eq!(markets[0].outcome_tokens[0].side, TokenSide::Yes);
        assert_eq!(
            markets[0].outcome_tokens[0].condition_id.as_deref(),
            Some("0xaaa")
        );
        assert_eq!(markets[0].outcome_tokens[0].tick_size, dec!(0.01));
        assert_eq!(markets[0].outcome_tokens[1].tick_size, dec!(0.005));
        assert_eq!(markets[1].outcome_tokens[0].tick_size, dec!(0.01));
    }
    #[test]
    fn parses_gamma_token_arrays_encoded_as_strings() {
        let market: GammaMarket = serde_json::from_value(serde_json::json!({
            "id": "m1",
            "outcomes": "[\"Yes\", \"No\"]",
            "clobTokenIds": "[\"111\", \"222\"]",
            "conditionId": "0xabc",
            "orderPriceMinTickSize": 0.01
        }))
        .unwrap();

        let tokens = outcome_tokens_from_market("m1", &market, true);
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].token_id, "111");
        assert_eq!(tokens[0].side, TokenSide::Yes);
        assert_eq!(tokens[1].side, TokenSide::No);
        assert_eq!(tokens[0].condition_id.as_deref(), Some("0xabc"));
    }
}
