use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use crate::{
    models::{GammaMarketMetadata, WalletTradeTaxonomyCandidate, WalletTradeTaxonomyUpdate},
    segments::{
        classify_segment, SegmentText, GAMMA_SEGMENT_CLASSIFIER_VERSION,
        MRS_SEGMENT_V2_SCORE_VERSION,
    },
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
            "classifier_version": GAMMA_SEGMENT_CLASSIFIER_VERSION,
            "segment_key_schema": MRS_SEGMENT_V2_SCORE_VERSION,
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

pub fn normalize_gamma_taxonomy_label(label: &str) -> Option<String> {
    let (kind, slug) = label.split_once('.')?;
    let kind = normalize_key(kind)?;
    let slug = normalize_key(slug)?;

    match kind.as_str() {
        "series" => normalize_gamma_series_slug(&slug),
        "tag" => normalize_gamma_tag_slug(&slug),
        "category" => normalize_gamma_category(&slug),
        _ => None,
    }
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
    let normalized_series = series_slug
        .as_deref()
        .and_then(|slug| normalize_gamma_series_slug(slug));
    let normalized_sport = sport_key
        .as_deref()
        .and_then(|slug| normalize_gamma_series_slug(slug));
    let normalized_tag = tag_slugs
        .iter()
        .find_map(|tag| normalize_gamma_tag_slug(tag));
    let priority_tag = tag_slugs.iter().find_map(|tag| priority_tag_segment(tag));

    if let Some(category) = category.as_deref() {
        if category == "sports" {
            if let Some(segment) = normalized_series.or(normalized_sport) {
                return (Some(segment), dec!(1.0));
            }
            if let Some(series_slug) = series_slug.as_ref().or(sport_key.as_ref()) {
                return (Some(format!("sports.{series_slug}")), dec!(1.0));
            }
            return (Some("sports".to_string()), dec!(0.90));
        }
        if category == "crypto" {
            if let Some(segment) = normalized_series {
                return (Some(segment), dec!(1.0));
            }
            if let Some(segment) = normalized_tag.filter(|segment| segment.starts_with("crypto.")) {
                return (Some(segment), dec!(0.95));
            }
            if let Some(tag) = priority_tag {
                return (Some(format!("crypto.{tag}")), dec!(0.95));
            }
            return (Some("crypto".to_string()), dec!(0.90));
        }
        if category == "politics" {
            if let Some(segment) = normalized_tag.filter(|segment| segment.starts_with("politics."))
            {
                return (Some(segment), dec!(0.95));
            }
            return (Some("politics.general".to_string()), dec!(0.90));
        }
        if let Some(segment) = normalized_tag.or_else(|| normalize_gamma_category(category)) {
            return (Some(segment), dec!(0.90));
        }
        return (Some(category.to_string()), dec!(0.90));
    }

    if let Some(segment) = normalized_series {
        return (Some(segment), dec!(0.85));
    }
    if let Some(series_slug) = series_slug {
        return (Some(format!("series.{series_slug}")), dec!(0.75));
    }
    if let Some(segment) = normalized_tag {
        return (Some(segment), dec!(0.80));
    }
    if let Some(tag) = priority_tag.or_else(|| tag_slugs.iter().find_map(|tag| normalize_key(tag)))
    {
        return (Some(format!("tag.{tag}")), dec!(0.70));
    }
    (None, Decimal::ZERO)
}

fn normalize_gamma_series_slug(slug: &str) -> Option<String> {
    if slug.starts_with("btc-updown-5m")
        || slug.starts_with("btc-updown-15m")
        || slug.starts_with("btc-up-or-down-5m")
        || slug.starts_with("btc-up-or-down-15m")
    {
        return Some("crypto.bitcoin.short_interval".to_string());
    }
    if slug.starts_with("eth-updown-5m")
        || slug.starts_with("eth-updown-15m")
        || slug.starts_with("eth-up-or-down-5m")
        || slug.starts_with("eth-up-or-down-15m")
    {
        return Some("crypto.ethereum.short_interval".to_string());
    }

    match slug {
        "btc-up-or-down-5m" => Some("crypto.bitcoin.short_interval".to_string()),
        "btc-up-or-down-15m" => Some("crypto.bitcoin.short_interval".to_string()),
        "btc-up-or-down-hourly" => Some("crypto.bitcoin.hourly".to_string()),
        "btc-multi-strikes-weekly" | "bitcoin-hit-price-monthly" => {
            Some("crypto.bitcoin".to_string())
        }
        "eth-up-or-down-5m" | "eth-up-or-down-15m" => {
            Some("crypto.ethereum.short_interval".to_string())
        }
        "eth-up-or-down-hourly" => Some("crypto.ethereum.hourly".to_string()),
        "mlb" => Some("sports.mlb".to_string()),
        "nba" | "nba-2025" | "nba-2026" => Some("sports.nba".to_string()),
        "wnba" | "wnba-2025" | "wnba-2026" => Some("sports.wnba".to_string()),
        "nfl" | "nfl-2025" | "nfl-2026" => Some("sports.nfl".to_string()),
        "nhl" | "nhl-2025" | "nhl-2026" => Some("sports.nhl".to_string()),
        "atp" => Some("sports.tennis.atp".to_string()),
        "wta" => Some("sports.tennis.wta".to_string()),
        "league-of-legends" => Some("esports.league_of_legends".to_string()),
        "dota-2" => Some("esports.dota2".to_string()),
        "iran-regime" | "hormuz-traffic-returns-to-normal" => Some("geopolitics.iran".to_string()),
        "fomc" => Some("macro.fed".to_string()),
        "elon-tweets" => Some("culture.elon".to_string()),
        _ => None,
    }
}

fn normalize_gamma_tag_slug(slug: &str) -> Option<String> {
    match slug {
        "iran" => Some("geopolitics.iran".to_string()),
        "politics" => Some("politics.general".to_string()),
        "election" | "elections" => Some("politics.elections".to_string()),
        "bitcoin" | "btc" => Some("crypto.bitcoin".to_string()),
        "ethereum" | "eth" => Some("crypto.ethereum".to_string()),
        "solana" | "sol" => Some("crypto.solana".to_string()),
        "xrp" => Some("crypto.xrp".to_string()),
        "dogecoin" | "doge" => Some("crypto.dogecoin".to_string()),
        _ => normalize_gamma_series_slug(slug),
    }
}

fn normalize_gamma_category(category: &str) -> Option<String> {
    match category {
        "politics" => Some("politics.general".to_string()),
        _ => None,
    }
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
    fn normalizes_known_gamma_taxonomy_labels_to_segment_keys() {
        let cases = [
            ("series.btc-up-or-down-5m", "crypto.bitcoin.short_interval"),
            ("series.btc-up-or-down-hourly", "crypto.bitcoin.hourly"),
            ("series.mlb", "sports.mlb"),
            ("series.atp", "sports.tennis.atp"),
            ("series.wta", "sports.tennis.wta"),
            ("series.league-of-legends", "esports.league_of_legends"),
            ("series.dota-2", "esports.dota2"),
            ("tag.iran", "geopolitics.iran"),
            ("tag.politics", "politics.general"),
            ("tag.elections", "politics.elections"),
        ];

        for (label, expected) in cases {
            assert_eq!(
                normalize_gamma_taxonomy_label(label).as_deref(),
                Some(expected),
                "{label}"
            );
        }
    }

    #[test]
    fn leaves_unknown_gamma_taxonomy_labels_unmapped() {
        assert_eq!(normalize_gamma_taxonomy_label("series.some-new-show"), None);
        assert_eq!(normalize_gamma_taxonomy_label("topic.elections"), None);
        assert_eq!(normalize_gamma_taxonomy_label("elections"), None);
    }

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
    fn derives_crypto_interval_segment_from_gamma_series() {
        let metadata = metadata_from_gamma_event(
            "btc-up-or-down-5m-event",
            &serde_json::json!({
                "id": "btc-5m",
                "slug": "btc-up-or-down-5m-event",
                "category": "Crypto",
                "series": [{"slug": "btc-up-or-down-5m"}],
                "tags": [{"slug": "bitcoin"}]
            }),
        )
        .unwrap();

        assert_eq!(
            metadata.taxonomy_segment.as_deref(),
            Some("crypto.bitcoin.short_interval")
        );
        assert_eq!(metadata.taxonomy_confidence, dec!(1.0));
    }

    #[test]
    fn derives_politics_segment_from_gamma_tag() {
        let metadata = metadata_from_gamma_market(
            "election-market",
            &serde_json::json!({
                "id": "3",
                "slug": "election-market",
                "category": "Politics",
                "tags": [{"slug": "elections"}]
            }),
        )
        .unwrap();

        assert_eq!(
            metadata.taxonomy_segment.as_deref(),
            Some("politics.elections")
        );
        assert_eq!(metadata.taxonomy_confidence, dec!(0.95));
    }

    #[test]
    fn derives_geopolitics_segment_from_gamma_tag_without_category() {
        let metadata = metadata_from_gamma_market(
            "iran-market",
            &serde_json::json!({
                "id": "4",
                "slug": "iran-market",
                "tags": [{"slug": "iran"}]
            }),
        )
        .unwrap();

        assert_eq!(
            metadata.taxonomy_segment.as_deref(),
            Some("geopolitics.iran")
        );
        assert_eq!(metadata.taxonomy_confidence, dec!(0.80));
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
