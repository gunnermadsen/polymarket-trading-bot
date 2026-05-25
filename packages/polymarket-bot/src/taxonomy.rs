use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use crate::{
    models::{GammaMarketMetadata, WalletTradeTaxonomyCandidate, WalletTradeTaxonomyUpdate},
    segments::{classify_segment, SegmentText},
};

pub const GAMMA_TAXONOMY_VERSION: &str = "gamma_taxonomy_v1";

pub fn metadata_from_gamma_event(
    lookup_slug: &str,
    event: &serde_json::Value,
) -> Option<GammaMarketMetadata> {
    let event_slug = json_str(event, "slug").or_else(|| Some(lookup_slug.to_string()));
    let event_id = json_str(event, "id");
    let category = json_str(event, "category");
    let series_slug = first_slug(event.get("series"));
    let tag_slugs = slugs(event.get("tags"));
    let sport_key = sport_from_series_or_tags(series_slug.as_deref(), &tag_slugs);
    let (taxonomy_segment, taxonomy_confidence) = derive_taxonomy_segment(
        category.as_deref(),
        series_slug.as_deref(),
        sport_key.as_deref(),
        &tag_slugs,
    );
    Some(GammaMarketMetadata {
        cache_key: cache_key("event_slug", lookup_slug),
        lookup_type: "event_slug".to_string(),
        lookup_slug: lookup_slug.to_string(),
        event_slug,
        market_slug: None,
        gamma_event_id: event_id,
        gamma_market_id: None,
        category,
        series_slug,
        tag_slugs,
        sport_key,
        taxonomy_segment,
        taxonomy_source: "gamma".to_string(),
        taxonomy_confidence,
        taxonomy_version: GAMMA_TAXONOMY_VERSION.to_string(),
        raw_payload: event.clone(),
        fetched_at: Utc::now(),
    })
}

pub fn metadata_from_gamma_market(
    lookup_slug: &str,
    market: &serde_json::Value,
) -> Option<GammaMarketMetadata> {
    let market_slug = json_str(market, "slug").or_else(|| Some(lookup_slug.to_string()));
    let event_slug = json_str(market, "eventSlug")
        .or_else(|| json_str(market, "event_slug"))
        .or_else(|| {
            market
                .get("event")
                .and_then(|event| json_str(event, "slug"))
        });
    let category = json_str(market, "category").or_else(|| {
        market
            .get("event")
            .and_then(|event| json_str(event, "category"))
    });
    let series_slug = first_slug(market.get("series")).or_else(|| {
        market
            .get("event")
            .and_then(|event| first_slug(event.get("series")))
    });
    let mut tag_slugs = slugs(market.get("tags"));
    if tag_slugs.is_empty() {
        if let Some(event) = market.get("event") {
            tag_slugs = slugs(event.get("tags"));
        }
    }
    let sport_key = sport_from_series_or_tags(series_slug.as_deref(), &tag_slugs);
    let (taxonomy_segment, taxonomy_confidence) = derive_taxonomy_segment(
        category.as_deref(),
        series_slug.as_deref(),
        sport_key.as_deref(),
        &tag_slugs,
    );
    Some(GammaMarketMetadata {
        cache_key: cache_key("market_slug", lookup_slug),
        lookup_type: "market_slug".to_string(),
        lookup_slug: lookup_slug.to_string(),
        event_slug,
        market_slug,
        gamma_event_id: market
            .get("event")
            .and_then(|event| json_str(event, "id"))
            .or_else(|| json_str(market, "eventId")),
        gamma_market_id: json_str(market, "id"),
        category,
        series_slug,
        tag_slugs,
        sport_key,
        taxonomy_segment,
        taxonomy_source: "gamma".to_string(),
        taxonomy_confidence,
        taxonomy_version: GAMMA_TAXONOMY_VERSION.to_string(),
        raw_payload: market.clone(),
        fetched_at: Utc::now(),
    })
}

pub fn taxonomy_update_from_metadata(
    trade: &WalletTradeTaxonomyCandidate,
    metadata: &GammaMarketMetadata,
) -> Option<WalletTradeTaxonomyUpdate> {
    let segment = metadata.taxonomy_segment.as_ref()?.clone();
    Some(WalletTradeTaxonomyUpdate {
        trade_id: trade.trade_id,
        taxonomy_segment: segment,
        taxonomy_source: metadata.taxonomy_source.clone(),
        taxonomy_confidence: metadata.taxonomy_confidence,
        taxonomy_version: metadata.taxonomy_version.clone(),
        taxonomy_fetched_at: metadata.fetched_at,
        taxonomy_metadata: serde_json::json!({
            "source": metadata.taxonomy_source,
            "category": metadata.category,
            "series_slug": metadata.series_slug,
            "tag_slugs": metadata.tag_slugs,
            "sport_key": metadata.sport_key,
            "cache_key": metadata.cache_key,
            "lookup_type": metadata.lookup_type,
            "lookup_slug": metadata.lookup_slug,
            "gamma_event_id": metadata.gamma_event_id,
            "gamma_market_id": metadata.gamma_market_id
        }),
    })
}

pub fn fallback_taxonomy_update(trade: &WalletTradeTaxonomyCandidate) -> WalletTradeTaxonomyUpdate {
    let classification = classify_segment(SegmentText {
        title: trade.title.as_deref(),
        slug: trade.slug.as_deref(),
        event_slug: trade.event_slug.as_deref(),
        question: None,
    });
    WalletTradeTaxonomyUpdate {
        trade_id: trade.trade_id,
        taxonomy_segment: classification.segment_key,
        taxonomy_source: "keyword_fallback".to_string(),
        taxonomy_confidence: classification.confidence,
        taxonomy_version: GAMMA_TAXONOMY_VERSION.to_string(),
        taxonomy_fetched_at: Utc::now(),
        taxonomy_metadata: serde_json::json!({
            "source": "keyword_fallback",
            "classifier_version": classification.classifier_version,
            "matched_rule": classification.matched_rule,
            "matched_terms": classification.matched_terms,
            "source_fields": classification.source_fields
        }),
    }
}

pub fn cache_key(lookup_type: &str, slug: &str) -> String {
    format!("{}:{}", lookup_type, slug.trim().to_ascii_lowercase())
}

fn derive_taxonomy_segment(
    category: Option<&str>,
    series_slug: Option<&str>,
    sport_key: Option<&str>,
    tag_slugs: &[String],
) -> (Option<String>, Decimal) {
    let category = category.and_then(normalize_key);
    let series_slug = series_slug.and_then(normalize_key);
    let sport_key = sport_key.and_then(normalize_key);
    let priority_tag = tag_slugs.iter().find_map(|tag| priority_tag_segment(tag));

    if let Some(category) = category {
        if category == "sports" {
            if let Some(series_slug) = series_slug.or(sport_key) {
                return (Some(format!("sports.{series_slug}")), dec!(1.0));
            }
            return (Some("sports".to_string()), dec!(0.90));
        }
        if category == "crypto" {
            if let Some(tag) = priority_tag {
                return (Some(format!("crypto.{tag}")), dec!(0.95));
            }
            return (Some("crypto".to_string()), dec!(0.90));
        }
        return (Some(category), dec!(0.90));
    }

    if let Some(series_slug) = series_slug {
        return (Some(format!("series.{series_slug}")), dec!(0.75));
    }
    if let Some(tag) = priority_tag.or_else(|| tag_slugs.iter().find_map(|tag| normalize_key(tag)))
    {
        return (Some(format!("tag.{tag}")), dec!(0.70));
    }
    (None, Decimal::ZERO)
}

fn priority_tag_segment(tag: &str) -> Option<String> {
    let tag = normalize_key(tag)?;
    matches!(
        tag.as_str(),
        "bitcoin" | "btc" | "ethereum" | "eth" | "solana" | "sol" | "xrp" | "dogecoin" | "doge"
    )
    .then_some(tag)
}

fn sport_from_series_or_tags(series_slug: Option<&str>, tag_slugs: &[String]) -> Option<String> {
    series_slug
        .and_then(normalize_key)
        .or_else(|| tag_slugs.iter().find_map(|tag| normalize_key(tag)))
}

fn first_slug(value: Option<&serde_json::Value>) -> Option<String> {
    match value? {
        serde_json::Value::Array(items) => items.iter().find_map(|item| json_str(item, "slug")),
        serde_json::Value::Object(_) => json_str(value?, "slug"),
        _ => None,
    }
}

fn slugs(value: Option<&serde_json::Value>) -> Vec<String> {
    match value {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| json_str(item, "slug").and_then(|slug| normalize_key(&slug)))
            .filter(|slug| slug != "all")
            .collect(),
        _ => Vec::new(),
    }
}

fn json_str(value: &serde_json::Value, key: &str) -> Option<String> {
    value.get(key).and_then(|value| match value {
        serde_json::Value::String(text) => {
            let trimmed = text.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        serde_json::Value::Number(number) => Some(number.to_string()),
        _ => None,
    })
}

fn normalize_key(value: &str) -> Option<String> {
    let normalized = value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    (!normalized.is_empty()).then_some(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_sports_series_segment_from_gamma_event() {
        let metadata = metadata_from_gamma_event(
            "nba-event",
            &serde_json::json!({
                "id": "1",
                "slug": "nba-event",
                "category": "Sports",
                "series": [{"slug": "nba", "title": "NBA"}],
                "tags": [{"slug": "all"}]
            }),
        )
        .unwrap();

        assert_eq!(metadata.taxonomy_segment.as_deref(), Some("sports.nba"));
        assert_eq!(metadata.taxonomy_confidence, dec!(1.0));
    }

    #[test]
    fn derives_crypto_subsegment_from_priority_tag() {
        let metadata = metadata_from_gamma_market(
            "btc-market",
            &serde_json::json!({
                "id": "2",
                "slug": "btc-market",
                "category": "Crypto",
                "tags": [{"slug": "bitcoin"}, {"slug": "all"}]
            }),
        )
        .unwrap();

        assert_eq!(metadata.taxonomy_segment.as_deref(), Some("crypto.bitcoin"));
        assert_eq!(metadata.tag_slugs, vec!["bitcoin"]);
    }

    #[test]
    fn fallback_uses_existing_keyword_classifier() {
        let trade = WalletTradeTaxonomyCandidate {
            trade_id: uuid::Uuid::nil(),
            title: Some("Will Bitcoin hit 120k?".to_string()),
            slug: Some("bitcoin-120k".to_string()),
            event_slug: None,
            market_id: None,
            condition_id: None,
            asset: "token".to_string(),
            raw_payload: serde_json::json!({}),
        };

        let update = fallback_taxonomy_update(&trade);
        assert_eq!(update.taxonomy_segment, "crypto");
        assert_eq!(update.taxonomy_source, "keyword_fallback");
    }
}
