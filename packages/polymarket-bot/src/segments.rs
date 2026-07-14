use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use crate::models::WalletSegmentPerformance;

pub const GAMMA_SEGMENT_CLASSIFIER_VERSION: &str = "gamma_taxonomy_v1";
pub const MRS_SEGMENT_SCORE_VERSION: &str = "mrs_segment_v1";
pub const MRS_SEGMENT_V2_SCORE_VERSION: &str = "mrs_segment_v2";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentClassification {
    pub segment_key: String,
    pub classifier_version: String,
    pub confidence: Decimal,
    pub matched_rule: String,
    pub matched_terms: Vec<String>,
    pub source_fields: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WalletSegmentPerformanceInput {
    pub proxy_wallet: String,
    pub segment_key: String,
    pub classifier_version: String,
    pub closed_positions: i32,
    pub winning_positions: i32,
    pub realized_pnl_usd: Decimal,
    pub total_bought_usd: Decimal,
    pub observed_trade_count: i32,
    pub observed_volume_usd: Decimal,
    pub sample_start: Option<DateTime<Utc>>,
    pub sample_end: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalletSegmentScore {
    pub proxy_wallet: String,
    pub segment_key: String,
    pub score: Decimal,
    pub confidence: Decimal,
    pub roi_score: Decimal,
    pub pnl_score: Decimal,
    pub win_rate_score: Decimal,
    pub sample_score: Decimal,
    pub activity_score: Decimal,
    pub input: WalletSegmentPerformanceInput,
}

pub fn classify_gamma_taxonomy_segment(
    taxonomy_segment: &str,
    taxonomy_confidence: Decimal,
    source_fields: serde_json::Value,
) -> Option<SegmentClassification> {
    let segment_key = normalize_gamma_segment_key(taxonomy_segment)?;
    Some(SegmentClassification {
        segment_key,
        classifier_version: GAMMA_SEGMENT_CLASSIFIER_VERSION.to_string(),
        confidence: taxonomy_confidence,
        matched_rule: "gamma_taxonomy".to_string(),
        matched_terms: vec![taxonomy_segment.to_string()],
        source_fields,
    })
}

pub fn normalize_gamma_segment_key(taxonomy_segment: &str) -> Option<String> {
    let raw = taxonomy_segment.trim().to_ascii_lowercase();
    if let Some(canonical) = normalize_canonical_gamma_segment_key(&raw) {
        return Some(canonical.to_string());
    }

    let segment = raw.replace('_', "-");
    let value = segment
        .strip_prefix("series.")
        .or_else(|| segment.strip_prefix("tag."))
        .or_else(|| segment.strip_prefix("category."))
        .unwrap_or(segment.as_str());

    if is_btc_short_interval_slug(value) {
        return Some("crypto.bitcoin.short_interval".to_string());
    }
    if is_eth_short_interval_slug(value) {
        return Some("crypto.ethereum.short_interval".to_string());
    }

    let normalized = match value {
        "btc-up-or-down-5m" | "btc-up-or-down-15m" => "crypto.bitcoin.short_interval",
        "btc-up-or-down-hourly" => "crypto.bitcoin.hourly",
        "btc-multi-strikes-weekly" | "bitcoin-hit-price-monthly" => "crypto.bitcoin",
        "eth-up-or-down-5m" | "eth-up-or-down-15m" => "crypto.ethereum.short_interval",
        "eth-up-or-down-hourly" => "crypto.ethereum.hourly",
        "crypto" => "crypto.general",
        "bitcoin" | "btc" => "crypto.bitcoin",
        "ethereum" | "eth" => "crypto.ethereum",
        "mlb" => "sports.mlb",
        "nba" | "nba-2025" | "nba-2026" => "sports.nba",
        "wnba" | "wnba-2025" | "wnba-2026" => "sports.wnba",
        "nhl" | "nhl-2025" | "nhl-2026" => "sports.nhl",
        "nfl" | "nfl-2025" | "nfl-2026" => "sports.nfl",
        "atp" => "sports.tennis.atp",
        "wta" => "sports.tennis.wta",
        "soccer" | "sports" => "sports.general",
        "league-of-legends" => "esports.league_of_legends",
        "dota-2" => "esports.dota2",
        "iran" | "iran-regime" | "hormuz-traffic-returns-to-normal" => "geopolitics.iran",
        "geopolitics" => "geopolitics.general",
        "world-elections" | "elections" => "politics.elections",
        "politics" => "politics.general",
        "fomc" => "macro.fed",
        "brazil" => "politics.brazil",
        "elon-tweets" => "culture.elon",
        _ => return None,
    };

    Some(normalized.to_string())
}

fn is_btc_short_interval_slug(value: &str) -> bool {
    value.starts_with("btc-updown-5m")
        || value.starts_with("btc-updown-15m")
        || value.starts_with("btc-up-or-down-5m")
        || value.starts_with("btc-up-or-down-15m")
}

fn is_eth_short_interval_slug(value: &str) -> bool {
    value.starts_with("eth-updown-5m")
        || value.starts_with("eth-updown-15m")
        || value.starts_with("eth-up-or-down-5m")
        || value.starts_with("eth-up-or-down-15m")
}

fn normalize_canonical_gamma_segment_key(segment_key: &str) -> Option<&'static str> {
    match segment_key {
        "crypto.general" => Some("crypto.general"),
        "crypto.bitcoin" => Some("crypto.bitcoin"),
        "crypto.bitcoin.short_interval" | "crypto.bitcoin.short-interval" => {
            Some("crypto.bitcoin.short_interval")
        }
        "crypto.bitcoin.hourly" => Some("crypto.bitcoin.hourly"),
        "crypto.ethereum" => Some("crypto.ethereum"),
        "crypto.ethereum.short_interval" | "crypto.ethereum.short-interval" => {
            Some("crypto.ethereum.short_interval")
        }
        "crypto.ethereum.hourly" => Some("crypto.ethereum.hourly"),
        "crypto.solana" => Some("crypto.solana"),
        "crypto.xrp" => Some("crypto.xrp"),
        "crypto.dogecoin" => Some("crypto.dogecoin"),
        "sports.general" => Some("sports.general"),
        "sports.mlb" => Some("sports.mlb"),
        "sports.nba" => Some("sports.nba"),
        "sports.wnba" => Some("sports.wnba"),
        "sports.nhl" => Some("sports.nhl"),
        "sports.nfl" => Some("sports.nfl"),
        "sports.tennis.atp" => Some("sports.tennis.atp"),
        "sports.tennis.wta" => Some("sports.tennis.wta"),
        "esports.league_of_legends" | "esports.league-of-legends" => {
            Some("esports.league_of_legends")
        }
        "esports.dota2" => Some("esports.dota2"),
        "geopolitics.general" => Some("geopolitics.general"),
        "geopolitics.iran" => Some("geopolitics.iran"),
        "politics.general" => Some("politics.general"),
        "politics.elections" => Some("politics.elections"),
        "politics.brazil" => Some("politics.brazil"),
        "macro.fed" => Some("macro.fed"),
        "culture.elon" => Some("culture.elon"),
        _ => None,
    }
}

pub fn score_wallet_segment(input: WalletSegmentPerformanceInput) -> WalletSegmentScore {
    let roi = if input.total_bought_usd > Decimal::ZERO {
        input.realized_pnl_usd / input.total_bought_usd
    } else {
        Decimal::ZERO
    };
    let win_rate = if input.closed_positions > 0 {
        Decimal::from(input.winning_positions) / Decimal::from(input.closed_positions)
    } else {
        Decimal::ZERO
    };
    let positive_roi = roi.max(Decimal::ZERO);
    let positive_pnl = input.realized_pnl_usd.max(Decimal::ZERO);
    let roi_score = bounded(positive_roi / dec!(0.50), Decimal::ZERO, Decimal::ONE) * dec!(100);
    let pnl_score = bounded(positive_pnl / dec!(1000), Decimal::ZERO, Decimal::ONE) * dec!(100);
    let win_rate_score = bounded(win_rate, Decimal::ZERO, Decimal::ONE) * dec!(100);
    let sample_score = bounded(
        Decimal::from(input.closed_positions.max(0)) / dec!(12),
        Decimal::ZERO,
        Decimal::ONE,
    ) * dec!(100);
    let volume_component = bounded(
        input.observed_volume_usd / dec!(25000),
        Decimal::ZERO,
        Decimal::ONE,
    );
    let trade_component = bounded(
        Decimal::from(input.observed_trade_count.max(0)) / dec!(30),
        Decimal::ZERO,
        Decimal::ONE,
    );
    let activity_score =
        ((volume_component * dec!(0.60)) + (trade_component * dec!(0.40))) * dec!(100);
    let confidence = bounded(
        (Decimal::from(input.closed_positions.max(0)) / dec!(10) * dec!(0.70))
            + (Decimal::from(input.observed_trade_count.max(0)) / dec!(30) * dec!(0.30)),
        Decimal::ZERO,
        Decimal::ONE,
    )
    .round_dp(4);
    let score = bounded(
        roi_score * dec!(0.30)
            + pnl_score * dec!(0.25)
            + win_rate_score * dec!(0.20)
            + sample_score * dec!(0.15)
            + activity_score * dec!(0.10),
        Decimal::ZERO,
        dec!(100),
    )
    .round_dp(4);

    WalletSegmentScore {
        proxy_wallet: input.proxy_wallet.clone(),
        segment_key: input.segment_key.clone(),
        score,
        confidence,
        roi_score: roi_score.round_dp(4),
        pnl_score: pnl_score.round_dp(4),
        win_rate_score: win_rate_score.round_dp(4),
        sample_score: sample_score.round_dp(4),
        activity_score: activity_score.round_dp(4),
        input,
    }
}

impl WalletSegmentScore {
    pub fn into_wallet_segment_performance(self) -> WalletSegmentPerformance {
        let classifier_version = self.input.classifier_version.clone();
        let losing_positions = self
            .input
            .closed_positions
            .saturating_sub(self.input.winning_positions);
        let roi = if self.input.total_bought_usd > Decimal::ZERO {
            self.input.realized_pnl_usd / self.input.total_bought_usd
        } else {
            Decimal::ZERO
        };
        let win_rate = if self.input.closed_positions > 0 {
            Decimal::from(self.input.winning_positions) / Decimal::from(self.input.closed_positions)
        } else {
            Decimal::ZERO
        };
        WalletSegmentPerformance {
            proxy_wallet: self.proxy_wallet,
            segment_key: self.segment_key,
            score_version: MRS_SEGMENT_SCORE_VERSION.to_string(),
            classifier_version,
            score: self.score,
            confidence: self.confidence,
            closed_positions: self.input.closed_positions,
            winning_positions: self.input.winning_positions,
            losing_positions,
            win_rate,
            realized_pnl_usd: self.input.realized_pnl_usd,
            total_bought_usd: self.input.total_bought_usd,
            roi,
            observed_trade_count: self.input.observed_trade_count,
            observed_volume_usd: self.input.observed_volume_usd,
            sample_start: self.input.sample_start,
            sample_end: self.input.sample_end,
            metadata: serde_json::json!({
                "score_basis": MRS_SEGMENT_SCORE_VERSION,
                "classifier_version": self.input.classifier_version,
                "score_range": "0_to_100",
                "components": {
                    "roi_score": self.roi_score,
                    "pnl_score": self.pnl_score,
                    "win_rate_score": self.win_rate_score,
                    "sample_score": self.sample_score,
                    "activity_score": self.activity_score
                },
                "weights": {
                    "roi": 0.30,
                    "realized_pnl": 0.25,
                    "win_rate": 0.20,
                    "sample_size": 0.15,
                    "activity": 0.10
                }
            }),
        }
    }
}

fn bounded(value: Decimal, min: Decimal, max: Decimal) -> Decimal {
    value.max(min).min(max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_taxonomy_normalizes_specific_segments() {
        assert_eq!(
            normalize_gamma_segment_key("series.btc-up-or-down-5m").as_deref(),
            Some("crypto.bitcoin.short_interval")
        );
        assert_eq!(
            normalize_gamma_segment_key("btc-updown-5m-1779894000").as_deref(),
            Some("crypto.bitcoin.short_interval")
        );
        assert_eq!(
            normalize_gamma_segment_key("crypto.bitcoin.short_interval").as_deref(),
            Some("crypto.bitcoin.short_interval")
        );
        assert_eq!(
            normalize_gamma_segment_key("series.mlb").as_deref(),
            Some("sports.mlb")
        );
        assert_eq!(
            normalize_gamma_segment_key("series.nba-2026").as_deref(),
            Some("sports.nba")
        );
        assert_eq!(
            normalize_gamma_segment_key("series.atp").as_deref(),
            Some("sports.tennis.atp")
        );
        assert_eq!(
            normalize_gamma_segment_key("esports.league_of_legends").as_deref(),
            Some("esports.league_of_legends")
        );
        assert_eq!(
            normalize_gamma_segment_key("series.league-of-legends").as_deref(),
            Some("esports.league_of_legends")
        );
        assert_eq!(
            normalize_gamma_segment_key("tag.iran").as_deref(),
            Some("geopolitics.iran")
        );
        assert_eq!(
            normalize_gamma_segment_key("tag.elections").as_deref(),
            Some("politics.elections")
        );
        assert_eq!(
            normalize_gamma_segment_key("tag.politics").as_deref(),
            Some("politics.general")
        );
        assert_eq!(normalize_gamma_segment_key("series.some-new-label"), None);
    }

    #[test]
    fn gamma_taxonomy_classification_uses_gamma_version() {
        let classification = classify_gamma_taxonomy_segment(
            "series.btc-up-or-down-hourly",
            dec!(0.75),
            serde_json::json!({"source": "test"}),
        )
        .expect("classification");
        assert_eq!(classification.segment_key, "crypto.bitcoin.hourly");
        assert_eq!(
            classification.classifier_version,
            GAMMA_SEGMENT_CLASSIFIER_VERSION
        );
        assert_eq!(classification.confidence, dec!(0.75));
    }

    #[test]
    fn segment_score_is_bounded() {
        let score = score_wallet_segment(WalletSegmentPerformanceInput {
            proxy_wallet: "0xabc".to_string(),
            segment_key: "crypto".to_string(),
            classifier_version: GAMMA_SEGMENT_CLASSIFIER_VERSION.to_string(),
            closed_positions: 50,
            winning_positions: 50,
            realized_pnl_usd: dec!(100000),
            total_bought_usd: dec!(1),
            observed_trade_count: 1000,
            observed_volume_usd: dec!(1000000),
            sample_start: None,
            sample_end: None,
        });
        assert_eq!(score.score, dec!(100.0000));
        assert_eq!(score.confidence, dec!(1.0000));
    }
}
