use std::{collections::HashMap, fmt};

use chrono::{DateTime, Datelike, Timelike, Utc};
use rust_decimal::{prelude::ToPrimitive, Decimal};

use super::types::{
    BinanceFiveMinuteSummary, BinanceOneSecondKline, BinanceOneSecondWindow, OrderbookCheckpoint,
    BINANCE_PREWINDOW_SUMMARY_CAPACITY,
};

pub const BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION: &str = "btc-5m-directional-core-features-v2";
pub const BTC_DIRECTIONAL_FEATURE_COUNT: usize = 58;
pub const BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION: &str =
    "btc-5m-directional-mature-reversal-features-v1";
pub const BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_COUNT: usize = 71;
pub const BTC_DIRECTIONAL_BOUNDARY_FEATURE_SCHEMA_VERSION: &str =
    "btc-5m-directional-boundary-features-v1";
pub const BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT: usize = 68;
pub const BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_SCHEMA_VERSION: &str =
    "btc-5m-directional-boundary-oracle-chainlink-candle-features-v1";
pub const BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_COUNT: usize = 87;
pub const BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_SCHEMA_VERSION: &str =
    "btc-5m-directional-boundary-oracle-chainlink-refprice-candle-features-v1";
pub const BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_COUNT: usize = 97;
pub const BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_SCHEMA_VERSION:
    &str = "btc-5m-directional-boundary-oracle-chainlink-refprice-candle-oi-features-v1";
pub const BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_COUNT: usize = 106;
pub const BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SCHEMA_VERSION: &str =
    "btc-5m-directional-path-persistence-prewindow-features-v1";
pub const BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_COUNT: usize = 100;
pub const BTC_DIRECTIONAL_FIRST_CANDIDATE_SECOND: i64 = 60;
pub const BTC_DIRECTIONAL_LAST_CANDIDATE_SECOND: i64 = 240;
pub const BTC_DIRECTIONAL_CANDIDATE_CADENCE_SECONDS: i64 = 5;

const MARKET_WINDOW_SECONDS: i64 = 300;
const EPSILON: f64 = 1e-9;
const BPS: f64 = 10_000.0;
pub const BTC_DIRECTIONAL_REFPRICE_MAX_AGE_SECONDS: i64 = 5;
pub const BTC_DIRECTIONAL_CHAINLINK_CANDLE_MAX_AGE_SECONDS: i64 = 120;
pub const BTC_DIRECTIONAL_OPEN_INTEREST_MAX_AGE_SECONDS: i64 = 600;
pub const BTC_DIRECTIONAL_ORACLE_MAX_AGE_SECONDS: i64 = 3_600;

/// Frozen feature order consumed by the deployed model artifact.
///
/// This order must remain byte-for-byte aligned with `CORE_ENRICHED_FEATURES` in the
/// Python training package. A subsequent model with a different feature contract must
/// use a new schema version rather than mutating this array.
pub const BTC_DIRECTIONAL_FEATURE_NAMES: [&str; BTC_DIRECTIONAL_FEATURE_COUNT] = [
    "seconds_elapsed_scaled",
    "seconds_remaining_scaled",
    "btc_path_from_window_open_bps",
    "btc_return_1s_bps",
    "btc_return_5s_bps",
    "btc_return_15s_bps",
    "btc_return_30s_bps",
    "btc_return_60s_bps",
    "btc_realized_volatility_5s_bps",
    "btc_realized_volatility_15s_bps",
    "btc_realized_volatility_30s_bps",
    "btc_realized_volatility_60s_bps",
    "btc_range_5s_bps",
    "btc_range_30s_bps",
    "btc_range_60s_bps",
    "btc_path_efficiency_30s",
    "btc_path_efficiency_60s",
    "btc_range_position_30s",
    "btc_volatility_expansion_5_to_30",
    "btc_log_quote_volume_5s",
    "btc_log_quote_volume_30s",
    "btc_log_quote_volume_60s",
    "btc_log_trade_count_5s",
    "btc_log_trade_count_30s",
    "btc_taker_buy_share_5s",
    "btc_taker_buy_share_30s",
    "btc_signed_flow_5s",
    "btc_signed_flow_30s",
    "btc_volume_surprise_5_to_60",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "btc_path_terminal_volatility_z",
    "btc_path_abs_terminal_volatility_z",
    "btc_path_cross_count",
    "btc_seconds_since_path_cross",
    "btc_fraction_time_path_positive",
    "btc_fraction_time_path_negative",
    "btc_momentum_agreement_5_15",
    "btc_momentum_agreement_15_30",
    "btc_momentum_multihorizon_score",
    "btc_momentum_acceleration_5_vs_30",
    "btc_momentum_acceleration_15_vs_60",
    "btc_reversal_5_vs_30",
    "btc_range_position_60s",
    "btc_distance_from_high_30s_bps",
    "btc_distance_from_low_30s_bps",
    "btc_distance_from_high_60s_bps",
    "btc_distance_from_low_60s_bps",
    "btc_volatility_regime_60_vs_elapsed",
    "btc_log_trade_count_60s",
    "btc_taker_buy_share_60s",
    "btc_signed_flow_60s",
    "btc_flow_persistence_5_30",
    "btc_flow_persistence_30_60",
    "btc_price_flow_agreement_30s",
    "btc_price_flow_divergence_30s",
];

pub const BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SUFFIX_NAMES: [&str;
    BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_COUNT - BTC_DIRECTIONAL_FEATURE_COUNT] = [
    "btc_path_max_favorable_excursion_bps",
    "btc_path_max_adverse_excursion_bps",
    "btc_path_pullback_from_favorable_extreme_bps",
    "btc_path_recovery_from_adverse_extreme_bps",
    "btc_seconds_since_path_high_scaled",
    "btc_seconds_since_path_low_scaled",
    "btc_path_sign_normalized_return_5s_bps",
    "btc_path_sign_normalized_return_15s_bps",
    "btc_path_sign_normalized_return_30s_bps",
    "btc_path_sign_normalized_return_60s_bps",
    "btc_path_sign_normalized_flow_5s",
    "btc_path_sign_normalized_flow_30s",
    "btc_path_sign_normalized_flow_60s",
];

const fn mature_reversal_feature_names(
) -> [&'static str; BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_COUNT] {
    let mut names = [""; BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_COUNT];
    let mut index = 0;
    while index < BTC_DIRECTIONAL_FEATURE_COUNT {
        names[index] = BTC_DIRECTIONAL_FEATURE_NAMES[index];
        index += 1;
    }
    let mut suffix_index = 0;
    while suffix_index < BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SUFFIX_NAMES.len() {
        names[index] = BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SUFFIX_NAMES[suffix_index];
        index += 1;
        suffix_index += 1;
    }
    names
}

pub const BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_NAMES: [&str;
    BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_COUNT] = mature_reversal_feature_names();

pub const BTC_DIRECTIONAL_BOUNDARY_FEATURE_SUFFIX_NAMES: [&str;
    BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT - BTC_DIRECTIONAL_FEATURE_COUNT] = [
    "btc_cross_venue_boundary_gap_bps",
    "btc_window_open_cross_venue_basis_bps",
    "btc_boundary_terminal_volatility_z",
    "btc_boundary_abs_terminal_volatility_z",
    "btc_boundary_cross_count",
    "btc_seconds_since_boundary_cross",
    "btc_fraction_time_boundary_positive",
    "btc_fraction_time_boundary_negative",
    "btc_boundary_distance_velocity_5s_bps",
    "btc_boundary_momentum_alignment_5s",
];

const fn boundary_feature_names() -> [&'static str; BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT] {
    let mut names = [""; BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT];
    let mut index = 0;
    while index < BTC_DIRECTIONAL_FEATURE_COUNT {
        names[index] = BTC_DIRECTIONAL_FEATURE_NAMES[index];
        index += 1;
    }
    let mut suffix_index = 0;
    while suffix_index < BTC_DIRECTIONAL_BOUNDARY_FEATURE_SUFFIX_NAMES.len() {
        names[index] = BTC_DIRECTIONAL_BOUNDARY_FEATURE_SUFFIX_NAMES[suffix_index];
        index += 1;
        suffix_index += 1;
    }
    names
}

pub const BTC_DIRECTIONAL_BOUNDARY_FEATURE_NAMES: [&str; BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT] =
    boundary_feature_names();

pub const BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES: [&str; 11] = [
    "oracle_gap_to_opening_boundary_bps",
    "oracle_return_from_window_open_bps",
    "oracle_return_30s_bps",
    "oracle_return_60s_bps",
    "oracle_round_age_seconds_scaled",
    "oracle_update_count_since_open_scaled",
    "binance_oracle_basis_bps",
    "binance_oracle_basis_change_30s_bps",
    "oracle_binance_direction_agreement_30s",
    "oracle_boundary_binance_path_agreement",
    "oracle_return_60s_binance_volatility_z",
];

pub const BTC_DIRECTIONAL_CHAINLINK_REFPRICE_FEATURE_NAMES: [&str; 10] = [
    "chainlink_ref_return_1s_bps",
    "chainlink_ref_return_5s_bps",
    "chainlink_ref_return_15s_bps",
    "chainlink_ref_return_30s_bps",
    "chainlink_ref_return_60s_bps",
    "chainlink_ref_binance_basis_bps",
    "chainlink_ref_boundary_gap_bps",
    "chainlink_ref_spread_bps",
    "chainlink_ref_spread_change_30s_bps",
    "chainlink_ref_binance_direction_agreement_30s",
];

pub const BTC_DIRECTIONAL_CHAINLINK_CANDLE_FEATURE_NAMES: [&str; 8] = [
    "chainlink_candle_return_5m_bps",
    "chainlink_candle_return_15m_bps",
    "chainlink_candle_return_30m_bps",
    "chainlink_candle_return_60m_bps",
    "chainlink_candle_realized_volatility_15m_bps",
    "chainlink_candle_realized_volatility_60m_bps",
    "chainlink_candle_range_15m_bps",
    "chainlink_candle_range_60m_bps",
];

pub const BTC_DIRECTIONAL_BINANCE_OPEN_INTEREST_FEATURE_NAMES: [&str; 9] = [
    "binance_oi_change_5m_bps",
    "binance_oi_change_15m_bps",
    "binance_oi_change_30m_bps",
    "binance_oi_change_60m_bps",
    "binance_oi_value_change_15m_bps",
    "binance_oi_value_change_60m_bps",
    "binance_oi_acceleration_5_vs_30_bps",
    "binance_oi_path_agreement_15m",
    "binance_oi_path_agreement_60m",
];

const fn append_feature_names<const OUTPUT: usize, const SUFFIX: usize>(
    mut names: [&'static str; OUTPUT],
    mut index: usize,
    suffix: [&'static str; SUFFIX],
) -> [&'static str; OUTPUT] {
    let mut suffix_index = 0;
    while suffix_index < SUFFIX {
        names[index] = suffix[suffix_index];
        index += 1;
        suffix_index += 1;
    }
    names
}

const fn boundary_oracle_chainlink_candle_feature_names(
) -> [&'static str; BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_COUNT] {
    let mut names = [""; BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_COUNT];
    let mut index = 0;
    while index < BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT {
        names[index] = BTC_DIRECTIONAL_BOUNDARY_FEATURE_NAMES[index];
        index += 1;
    }
    names = append_feature_names(names, index, BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES);
    index += BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.len();
    append_feature_names(names, index, BTC_DIRECTIONAL_CHAINLINK_CANDLE_FEATURE_NAMES)
}

const fn boundary_oracle_chainlink_refprice_candle_feature_names(
) -> [&'static str; BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_COUNT] {
    let mut names = [""; BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_COUNT];
    let mut index = 0;
    while index < BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT {
        names[index] = BTC_DIRECTIONAL_BOUNDARY_FEATURE_NAMES[index];
        index += 1;
    }
    names = append_feature_names(names, index, BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES);
    index += BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.len();
    names = append_feature_names(
        names,
        index,
        BTC_DIRECTIONAL_CHAINLINK_REFPRICE_FEATURE_NAMES,
    );
    index += BTC_DIRECTIONAL_CHAINLINK_REFPRICE_FEATURE_NAMES.len();
    append_feature_names(names, index, BTC_DIRECTIONAL_CHAINLINK_CANDLE_FEATURE_NAMES)
}

const fn boundary_oracle_chainlink_refprice_candle_oi_feature_names(
) -> [&'static str; BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_COUNT] {
    let mut names =
        [""; BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_COUNT];
    let mut index = 0;
    while index < BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT {
        names[index] = BTC_DIRECTIONAL_BOUNDARY_FEATURE_NAMES[index];
        index += 1;
    }
    names = append_feature_names(names, index, BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES);
    index += BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.len();
    names = append_feature_names(
        names,
        index,
        BTC_DIRECTIONAL_CHAINLINK_REFPRICE_FEATURE_NAMES,
    );
    index += BTC_DIRECTIONAL_CHAINLINK_REFPRICE_FEATURE_NAMES.len();
    names = append_feature_names(names, index, BTC_DIRECTIONAL_CHAINLINK_CANDLE_FEATURE_NAMES);
    index += BTC_DIRECTIONAL_CHAINLINK_CANDLE_FEATURE_NAMES.len();
    append_feature_names(
        names,
        index,
        BTC_DIRECTIONAL_BINANCE_OPEN_INTEREST_FEATURE_NAMES,
    )
}

pub const BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_NAMES: [&str;
    BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_COUNT] =
    boundary_oracle_chainlink_candle_feature_names();
pub const BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_NAMES: [&str;
    BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_COUNT] =
    boundary_oracle_chainlink_refprice_candle_feature_names();
pub const BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_NAMES: [&str;
    BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_COUNT] =
    boundary_oracle_chainlink_refprice_candle_oi_feature_names();

pub const BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SUFFIX_NAMES: [&str;
    BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_COUNT - BTC_DIRECTIONAL_FEATURE_COUNT] = [
    "prewindow_return_5m_bps",
    "prewindow_range_5m_bps",
    "prewindow_realized_volatility_5m_bps",
    "prewindow_log_quote_volume_5m",
    "prewindow_log_trade_count_5m",
    "prewindow_taker_buy_share_5m",
    "prewindow_signed_flow_5m",
    "prewindow_return_15m_bps",
    "prewindow_range_15m_bps",
    "prewindow_realized_volatility_15m_bps",
    "prewindow_log_quote_volume_15m",
    "prewindow_log_trade_count_15m",
    "prewindow_taker_buy_share_15m",
    "prewindow_signed_flow_15m",
    "prewindow_return_30m_bps",
    "prewindow_range_30m_bps",
    "prewindow_realized_volatility_30m_bps",
    "prewindow_log_quote_volume_30m",
    "prewindow_log_trade_count_30m",
    "prewindow_taker_buy_share_30m",
    "prewindow_signed_flow_30m",
    "prewindow_return_60m_bps",
    "prewindow_range_60m_bps",
    "prewindow_realized_volatility_60m_bps",
    "prewindow_log_quote_volume_60m",
    "prewindow_log_trade_count_60m",
    "prewindow_taker_buy_share_60m",
    "prewindow_signed_flow_60m",
    "prewindow_momentum_agreement_5_15",
    "prewindow_momentum_agreement_5_30",
    "prewindow_momentum_agreement_5_60",
    "prewindow_momentum_acceleration_5_vs_15",
    "prewindow_momentum_acceleration_5_vs_30",
    "prewindow_momentum_acceleration_5_vs_60",
    "btc_path_prewindow_5m_agreement",
    "btc_path_prewindow_5m_reversal",
    "btc_path_prewindow_15m_agreement",
    "btc_path_prewindow_15m_reversal",
    "btc_path_prewindow_30m_agreement",
    "btc_path_prewindow_30m_reversal",
    "btc_path_prewindow_60m_agreement",
    "btc_path_prewindow_60m_reversal",
];

const fn path_prewindow_feature_names(
) -> [&'static str; BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_COUNT] {
    let mut names = [""; BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_COUNT];
    let mut index = 0;
    while index < BTC_DIRECTIONAL_FEATURE_COUNT {
        names[index] = BTC_DIRECTIONAL_FEATURE_NAMES[index];
        index += 1;
    }
    let mut suffix_index = 0;
    while suffix_index < BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SUFFIX_NAMES.len() {
        names[index] = BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SUFFIX_NAMES[suffix_index];
        index += 1;
        suffix_index += 1;
    }
    names
}

pub const BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_NAMES: [&str;
    BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_COUNT] = path_prewindow_feature_names();

pub fn directional_feature_names(schema_version: &str) -> Option<&'static [&'static str]> {
    match schema_version {
        BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION => Some(&BTC_DIRECTIONAL_FEATURE_NAMES),
        BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION => {
            Some(&BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_NAMES)
        }
        BTC_DIRECTIONAL_BOUNDARY_FEATURE_SCHEMA_VERSION => {
            Some(&BTC_DIRECTIONAL_BOUNDARY_FEATURE_NAMES)
        }
        BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_SCHEMA_VERSION => {
            Some(&BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_NAMES)
        }
        BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_SCHEMA_VERSION => {
            Some(&BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_NAMES)
        }
        BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_SCHEMA_VERSION => {
            Some(&BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_NAMES)
        }
        BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SCHEMA_VERSION => {
            Some(&BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_NAMES)
        }
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DirectionalFeatureVector {
    schema_version: &'static str,
    pub feature_as_of: DateTime<Utc>,
    pub seconds_elapsed: u16,
    pub values: Vec<f64>,
}

impl DirectionalFeatureVector {
    pub fn schema_version(&self) -> &'static str {
        self.schema_version
    }

    pub fn names(&self) -> &'static [&'static str] {
        directional_feature_names(self.schema_version)
            .expect("directional feature vectors carry only supported schemas")
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        self.names()
            .iter()
            .position(|candidate| *candidate == name)
            .map(|index| self.values[index])
    }
}

/// One causally available Polygon Chainlink round. `available_at` is the local receipt clock;
/// callers must never substitute archive insertion time for live availability.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectionalOracleRound {
    pub phase_id: i32,
    pub aggregator_round_id: i64,
    pub source_timestamp: DateTime<Utc>,
    pub block_timestamp: DateTime<Utc>,
    /// Archive-only provenance. Runtime `latestRoundData` observations retain neither field.
    /// The pair must be either fully present or fully absent and is never a learned feature.
    pub block_number: Option<i64>,
    pub log_index: Option<i32>,
    pub price: Decimal,
    pub available_at: DateTime<Utc>,
}

/// One signed Chainlink RefPrice report decoded by the runtime source adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectionalChainlinkRefPrice {
    pub source_timestamp: DateTime<Utc>,
    pub valid_from_timestamp: DateTime<Utc>,
    pub price: Decimal,
    pub bid: Decimal,
    pub ask: Decimal,
    pub available_at: DateTime<Utc>,
}

/// One closed Chainlink one-minute candle. Its timestamp contract matches the training table.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectionalChainlinkCandle {
    pub open_timestamp: DateTime<Utc>,
    pub close_timestamp: DateTime<Utc>,
    pub open_price: Decimal,
    pub high_price: Decimal,
    pub low_price: Decimal,
    pub close_price: Decimal,
    pub available_at: DateTime<Utc>,
}

/// One completed Binance Futures five-minute open-interest observation.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectionalBinanceOpenInterest {
    pub source_timestamp: DateTime<Utc>,
    pub period_seconds: i32,
    pub sum_open_interest: Decimal,
    pub sum_open_interest_value: Decimal,
    pub available_at: DateTime<Utc>,
}

/// Bounded, ascending source histories captured at the same immutable decision boundary.
#[derive(Debug, Clone, Copy, Default)]
pub struct DirectionalExternalFeatureInputs<'a> {
    pub oracle_rounds: &'a [DirectionalOracleRound],
    pub refprice_reports: &'a [DirectionalChainlinkRefPrice],
    pub chainlink_candles: &'a [DirectionalChainlinkCandle],
    pub open_interest: &'a [DirectionalBinanceOpenInterest],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DirectionalExternalFeatureRequirements {
    pub oracle: bool,
    pub refprice: bool,
    pub chainlink_candles: bool,
    pub open_interest: bool,
}

pub fn directional_external_feature_requirements(
    schema_version: &str,
) -> DirectionalExternalFeatureRequirements {
    match schema_version {
        BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_SCHEMA_VERSION => {
            DirectionalExternalFeatureRequirements {
                oracle: true,
                chainlink_candles: true,
                ..DirectionalExternalFeatureRequirements::default()
            }
        }
        BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_SCHEMA_VERSION => {
            DirectionalExternalFeatureRequirements {
                oracle: true,
                refprice: true,
                chainlink_candles: true,
                open_interest: false,
            }
        }
        BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_SCHEMA_VERSION => {
            DirectionalExternalFeatureRequirements {
                oracle: true,
                refprice: true,
                chainlink_candles: true,
                open_interest: true,
            }
        }
        "btc-5m-payoff-aware-q5-features-v1" => DirectionalExternalFeatureRequirements {
            oracle: true,
            ..DirectionalExternalFeatureRequirements::default()
        },
        "btc-5m-payoff-aware-middle-features-v1" => DirectionalExternalFeatureRequirements {
            oracle: true,
            chainlink_candles: true,
            ..DirectionalExternalFeatureRequirements::default()
        },
        "btc-5m-payoff-aware-middle-oi-features-v1" => DirectionalExternalFeatureRequirements {
            oracle: true,
            chainlink_candles: true,
            open_interest: true,
            ..DirectionalExternalFeatureRequirements::default()
        },
        _ => DirectionalExternalFeatureRequirements::default(),
    }
}

pub fn directional_schema_requires_opening_boundary(schema_version: &str) -> bool {
    schema_version == BTC_DIRECTIONAL_BOUNDARY_FEATURE_SCHEMA_VERSION
        || schema_version.starts_with("btc-5m-payoff-aware-")
        || directional_external_feature_requirements(schema_version).oracle
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectionalFeatureTimingReason {
    NotWholeSecond,
    BeforeFirstCandidate,
    AfterLastCandidate,
    OffCadence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectionalFeatureError {
    InvalidTiming {
        feature_as_of: DateTime<Utc>,
        seconds_elapsed: i64,
        reason: DirectionalFeatureTimingReason,
    },
    MissingHistory {
        required_start: DateTime<Utc>,
        required_end: DateTime<Utc>,
        available_start: Option<DateTime<Utc>>,
        available_end: Option<DateTime<Utc>>,
    },
    GappedHistory {
        expected_open_timestamp: DateTime<Utc>,
        next_open_timestamp: DateTime<Utc>,
    },
    DuplicateOrOutOfOrderHistory {
        previous_open_timestamp: DateTime<Utc>,
        open_timestamp: DateTime<Utc>,
    },
    IncompleteCandle {
        open_timestamp: DateTime<Utc>,
        synthetic: bool,
    },
    InvalidCandle {
        open_timestamp: DateTime<Utc>,
        field: &'static str,
    },
    NonFiniteFeature {
        index: usize,
        name: &'static str,
    },
    UnsupportedFeatureSchema {
        schema_version: String,
    },
    MissingPrewindowHistory {
        required_start: DateTime<Utc>,
        required_end: DateTime<Utc>,
        available_start: Option<DateTime<Utc>>,
        available_end: Option<DateTime<Utc>>,
    },
    GappedPrewindowHistory {
        expected_window_start: DateTime<Utc>,
        actual_window_start: DateTime<Utc>,
    },
    IncompletePrewindowHistory {
        window_start: DateTime<Utc>,
    },
    MissingOpeningBoundary,
    InvalidOpeningBoundary,
    ExternalFeatureUnavailable {
        source: &'static str,
        reason: &'static str,
    },
}

impl fmt::Display for DirectionalFeatureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTiming {
                feature_as_of,
                seconds_elapsed,
                reason,
            } => write!(
                formatter,
                "directional feature time {feature_as_of} ({seconds_elapsed}s) is invalid: {reason:?}"
            ),
            Self::MissingHistory {
                required_start,
                required_end,
                available_start,
                available_end,
            } => write!(
                formatter,
                "directional features require complete history [{required_start}, {required_end}], \
                 available range is {available_start:?} through {available_end:?}"
            ),
            Self::GappedHistory {
                expected_open_timestamp,
                next_open_timestamp,
            } => write!(
                formatter,
                "directional feature history is gapped at {expected_open_timestamp}; \
                 next candle opens at {next_open_timestamp}"
            ),
            Self::DuplicateOrOutOfOrderHistory {
                previous_open_timestamp,
                open_timestamp,
            } => write!(
                formatter,
                "directional feature history is duplicate or out of order: \
                 {open_timestamp} follows {previous_open_timestamp}"
            ),
            Self::IncompleteCandle {
                open_timestamp,
                synthetic,
            } => write!(
                formatter,
                "directional feature candle {open_timestamp} is not source-complete \
                 (synthetic={synthetic})"
            ),
            Self::InvalidCandle {
                open_timestamp,
                field,
            } => write!(
                formatter,
                "directional feature candle {open_timestamp} has invalid {field}"
            ),
            Self::NonFiniteFeature { index, name } => write!(
                formatter,
                "directional feature {name} at index {index} is not finite"
            ),
            Self::UnsupportedFeatureSchema { schema_version } => {
                write!(
                    formatter,
                    "directional feature schema {schema_version} is not supported"
                )
            }
            Self::MissingPrewindowHistory {
                required_start,
                required_end,
                available_start,
                available_end,
            } => write!(
                formatter,
                "directional pre-window features require complete summaries [{required_start}, \
                 {required_end}], available range is {available_start:?} through {available_end:?}"
            ),
            Self::GappedPrewindowHistory {
                expected_window_start,
                actual_window_start,
            } => write!(
                formatter,
                "directional pre-window history expected {expected_window_start} but found \
                 {actual_window_start}"
            ),
            Self::IncompletePrewindowHistory { window_start } => write!(
                formatter,
                "directional pre-window history at {window_start} is incomplete"
            ),
            Self::MissingOpeningBoundary => {
                write!(formatter, "directional boundary features require an opening boundary")
            }
            Self::InvalidOpeningBoundary => {
                write!(formatter, "directional feature opening boundary must be positive")
            }
            Self::ExternalFeatureUnavailable { source, reason } => write!(
                formatter,
                "directional external feature source {source} is unavailable: {reason}"
            ),
        }
    }
}

impl DirectionalFeatureError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidTiming { .. } => "invalid_timing",
            Self::MissingHistory { .. } => "missing_history",
            Self::GappedHistory { .. } => "gapped_history",
            Self::DuplicateOrOutOfOrderHistory { .. } => "duplicate_or_out_of_order_history",
            Self::IncompleteCandle { .. } => "incomplete_candle",
            Self::InvalidCandle { .. } => "invalid_candle",
            Self::NonFiniteFeature { .. } => "non_finite_feature",
            Self::UnsupportedFeatureSchema { .. } => "unsupported_feature_schema",
            Self::MissingPrewindowHistory { .. } => "missing_prewindow_history",
            Self::GappedPrewindowHistory { .. } => "gapped_prewindow_history",
            Self::IncompletePrewindowHistory { .. } => "incomplete_prewindow_history",
            Self::MissingOpeningBoundary => "missing_opening_boundary",
            Self::InvalidOpeningBoundary => "invalid_opening_boundary",
            Self::ExternalFeatureUnavailable { .. } => "external_feature_unavailable",
        }
    }
}

impl std::error::Error for DirectionalFeatureError {}

/// Builds the frozen 58-value feature vector from closed Binance one-second candles.
///
/// `feature_as_of` uses the training contract's observation clock: the candle opening at
/// `window_start - 1s` is observed at `window_start` and has `seconds_elapsed = 0`.
/// Consequently, a candidate at second 60 consumes candles opening from
/// `window_start - 1s` through `window_start + 59s`, never the in-progress second-60
/// candle.
pub fn build_directional_features(
    window: &BinanceOneSecondWindow,
    window_start: DateTime<Utc>,
    feature_as_of: DateTime<Utc>,
) -> Result<DirectionalFeatureVector, DirectionalFeatureError> {
    build_directional_features_for_schema(
        window,
        window_start,
        feature_as_of,
        BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION,
    )
}

pub fn build_directional_features_for_schema(
    window: &BinanceOneSecondWindow,
    window_start: DateTime<Utc>,
    feature_as_of: DateTime<Utc>,
    schema_version: &str,
) -> Result<DirectionalFeatureVector, DirectionalFeatureError> {
    build_directional_features_for_schema_with_boundary(
        window,
        window_start,
        feature_as_of,
        schema_version,
        None,
    )
}

pub fn build_directional_features_for_schema_with_boundary(
    window: &BinanceOneSecondWindow,
    window_start: DateTime<Utc>,
    feature_as_of: DateTime<Utc>,
    schema_version: &str,
    opening_boundary: Option<Decimal>,
) -> Result<DirectionalFeatureVector, DirectionalFeatureError> {
    build_directional_features_for_schema_with_external(
        window,
        window_start,
        feature_as_of,
        schema_version,
        opening_boundary,
        None,
    )
}

pub fn build_directional_features_for_schema_with_external(
    window: &BinanceOneSecondWindow,
    window_start: DateTime<Utc>,
    feature_as_of: DateTime<Utc>,
    schema_version: &str,
    opening_boundary: Option<Decimal>,
    external: Option<&DirectionalExternalFeatureInputs<'_>>,
) -> Result<DirectionalFeatureVector, DirectionalFeatureError> {
    build_directional_features_for_schema_with_external_and_timing(
        window,
        window_start,
        feature_as_of,
        schema_version,
        opening_boundary,
        external,
        DirectionalFeatureTimingPolicy::LegacyDirectional,
        None,
    )
}

/// Builds the frozen core feature vector on the asymmetric value model's
/// manifest cadence without changing the legacy directional-model cadence.
pub fn build_directional_features_for_asymmetric_value(
    window: &BinanceOneSecondWindow,
    window_start: DateTime<Utc>,
    feature_as_of: DateTime<Utc>,
    imputation_medians: &[f64],
) -> Result<DirectionalFeatureVector, DirectionalFeatureError> {
    build_directional_features_for_schema_with_external_and_timing(
        window,
        window_start,
        feature_as_of,
        BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION,
        None,
        None,
        DirectionalFeatureTimingPolicy::AsymmetricValue,
        Some(imputation_medians),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectionalFeatureTimingPolicy {
    LegacyDirectional,
    AsymmetricValue,
}

#[allow(clippy::too_many_arguments)]
fn build_directional_features_for_schema_with_external_and_timing(
    window: &BinanceOneSecondWindow,
    window_start: DateTime<Utc>,
    feature_as_of: DateTime<Utc>,
    schema_version: &str,
    opening_boundary: Option<Decimal>,
    external: Option<&DirectionalExternalFeatureInputs<'_>>,
    timing_policy: DirectionalFeatureTimingPolicy,
    core_imputation_medians: Option<&[f64]>,
) -> Result<DirectionalFeatureVector, DirectionalFeatureError> {
    let schema_version = canonical_feature_schema_version(schema_version).ok_or_else(|| {
        DirectionalFeatureError::UnsupportedFeatureSchema {
            schema_version: schema_version.to_string(),
        }
    })?;
    let opening_boundary = if directional_schema_requires_opening_boundary(schema_version) {
        let boundary = opening_boundary.ok_or(DirectionalFeatureError::MissingOpeningBoundary)?;
        Some(
            boundary
                .to_f64()
                .filter(|value| value.is_finite() && *value > 0.0)
                .ok_or(DirectionalFeatureError::InvalidOpeningBoundary)?,
        )
    } else {
        None
    };
    let seconds_elapsed = validate_feature_time(window_start, feature_as_of, timing_policy)?;
    let prewindow_summaries =
        if schema_version == BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SCHEMA_VERSION {
            Some(collect_required_prewindow_summaries(window, window_start)?)
        } else {
            None
        };
    let required_start = window_start - chrono::Duration::seconds(1);
    let required_end = feature_as_of - chrono::Duration::seconds(1);
    let completed = collect_required_candles(window, required_start, required_end)?;
    let numeric = completed
        .into_iter()
        .map(NumericCandle::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let base_schema = if directional_external_feature_requirements(schema_version)
        != DirectionalExternalFeatureRequirements::default()
    {
        BTC_DIRECTIONAL_BOUNDARY_FEATURE_SCHEMA_VERSION
    } else {
        schema_version
    };
    let mut features = if timing_policy == DirectionalFeatureTimingPolicy::AsymmetricValue
        && seconds_elapsed < BTC_DIRECTIONAL_FIRST_CANDIDATE_SECOND
    {
        derive_asymmetric_early_core_features(
            &numeric,
            feature_as_of,
            seconds_elapsed,
            core_imputation_medians.ok_or(DirectionalFeatureError::ExternalFeatureUnavailable {
                source: "asymmetric_model",
                reason: "core imputation medians were not supplied",
            })?,
        )?
    } else {
        derive_directional_features(
            &numeric,
            feature_as_of,
            seconds_elapsed,
            base_schema,
            opening_boundary,
            prewindow_summaries.as_ref(),
        )?
    };
    let requirements = directional_external_feature_requirements(schema_version);
    if requirements != DirectionalExternalFeatureRequirements::default() {
        let external = external.ok_or(DirectionalFeatureError::ExternalFeatureUnavailable {
            source: "external",
            reason: "required source histories were not supplied",
        })?;
        append_external_features(
            &mut features.values,
            requirements,
            external,
            &numeric,
            window_start,
            feature_as_of,
            seconds_elapsed,
            opening_boundary.expect("external schemas require an opening boundary"),
        )?;
        features.schema_version = schema_version;
    }
    let names = directional_feature_names(schema_version)
        .expect("feature schema was canonicalized before derivation");
    if features.values.len() != names.len() {
        return Err(DirectionalFeatureError::ExternalFeatureUnavailable {
            source: "external",
            reason: "derived feature width does not match its frozen schema",
        });
    }
    if let Some((index, _)) = features
        .values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(DirectionalFeatureError::NonFiniteFeature {
            index,
            name: names[index],
        });
    }
    Ok(features)
}

fn canonical_feature_schema_version(schema_version: &str) -> Option<&'static str> {
    match schema_version {
        BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION => Some(BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION),
        BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION => {
            Some(BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION)
        }
        BTC_DIRECTIONAL_BOUNDARY_FEATURE_SCHEMA_VERSION => {
            Some(BTC_DIRECTIONAL_BOUNDARY_FEATURE_SCHEMA_VERSION)
        }
        BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_SCHEMA_VERSION => {
            Some(BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_SCHEMA_VERSION)
        }
        BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_SCHEMA_VERSION => {
            Some(BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_SCHEMA_VERSION)
        }
        BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_SCHEMA_VERSION => {
            Some(
                BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_SCHEMA_VERSION,
            )
        }
        BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SCHEMA_VERSION => {
            Some(BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SCHEMA_VERSION)
        }
        _ => None,
    }
}

fn validate_feature_time(
    window_start: DateTime<Utc>,
    feature_as_of: DateTime<Utc>,
    timing_policy: DirectionalFeatureTimingPolicy,
) -> Result<i64, DirectionalFeatureError> {
    let elapsed = feature_as_of.signed_duration_since(window_start);
    let seconds_elapsed = elapsed.num_seconds();
    if window_start + chrono::Duration::seconds(seconds_elapsed) != feature_as_of {
        return Err(DirectionalFeatureError::InvalidTiming {
            feature_as_of,
            seconds_elapsed,
            reason: DirectionalFeatureTimingReason::NotWholeSecond,
        });
    }
    let (first_candidate, last_candidate, on_cadence) = match timing_policy {
        DirectionalFeatureTimingPolicy::LegacyDirectional => (
            BTC_DIRECTIONAL_FIRST_CANDIDATE_SECOND,
            BTC_DIRECTIONAL_LAST_CANDIDATE_SECOND,
            (seconds_elapsed - BTC_DIRECTIONAL_FIRST_CANDIDATE_SECOND)
                % BTC_DIRECTIONAL_CANDIDATE_CADENCE_SECONDS
                == 0,
        ),
        DirectionalFeatureTimingPolicy::AsymmetricValue => {
            (1, 240, seconds_elapsed <= 59 || seconds_elapsed % 5 == 0)
        }
    };
    if seconds_elapsed < first_candidate {
        return Err(DirectionalFeatureError::InvalidTiming {
            feature_as_of,
            seconds_elapsed,
            reason: DirectionalFeatureTimingReason::BeforeFirstCandidate,
        });
    }
    if seconds_elapsed > last_candidate {
        return Err(DirectionalFeatureError::InvalidTiming {
            feature_as_of,
            seconds_elapsed,
            reason: DirectionalFeatureTimingReason::AfterLastCandidate,
        });
    }
    if !on_cadence {
        return Err(DirectionalFeatureError::InvalidTiming {
            feature_as_of,
            seconds_elapsed,
            reason: DirectionalFeatureTimingReason::OffCadence,
        });
    }
    Ok(seconds_elapsed)
}

fn collect_required_candles(
    window: &BinanceOneSecondWindow,
    required_start: DateTime<Utc>,
    required_end: DateTime<Utc>,
) -> Result<Vec<&BinanceOneSecondKline>, DirectionalFeatureError> {
    let available_start = window
        .completed()
        .front()
        .map(|candle| candle.open_timestamp);
    let available_end = window
        .completed()
        .back()
        .map(|candle| candle.open_timestamp);
    let mut candles = Vec::with_capacity(
        required_end
            .signed_duration_since(required_start)
            .num_seconds()
            .saturating_add(1) as usize,
    );
    let mut expected = required_start;
    let mut previous = None;

    for candle in window.completed() {
        if let Some(previous_open_timestamp) = previous {
            if candle.open_timestamp <= previous_open_timestamp {
                return Err(DirectionalFeatureError::DuplicateOrOutOfOrderHistory {
                    previous_open_timestamp,
                    open_timestamp: candle.open_timestamp,
                });
            }
        }
        previous = Some(candle.open_timestamp);

        if candle.open_timestamp < required_start {
            continue;
        }
        if candle.open_timestamp > required_end {
            break;
        }
        if candles.is_empty() && candle.open_timestamp != required_start {
            return Err(DirectionalFeatureError::MissingHistory {
                required_start,
                required_end,
                available_start,
                available_end,
            });
        }
        if candle.open_timestamp > expected {
            return Err(DirectionalFeatureError::GappedHistory {
                expected_open_timestamp: expected,
                next_open_timestamp: candle.open_timestamp,
            });
        }
        candles.push(candle);
        expected += chrono::Duration::seconds(1);
    }

    if candles.is_empty() || expected <= required_end {
        return Err(DirectionalFeatureError::MissingHistory {
            required_start,
            required_end,
            available_start,
            available_end,
        });
    }
    Ok(candles)
}

fn collect_required_prewindow_summaries(
    window: &BinanceOneSecondWindow,
    window_start: DateTime<Utc>,
) -> Result<[&BinanceFiveMinuteSummary; BINANCE_PREWINDOW_SUMMARY_CAPACITY], DirectionalFeatureError>
{
    let required_start = window_start - chrono::Duration::seconds(MARKET_WINDOW_SECONDS * 12);
    let required_end = window_start - chrono::Duration::seconds(MARKET_WINDOW_SECONDS);
    let available_start = window
        .completed_five_minute_summaries()
        .front()
        .map(|summary| summary.window_start);
    let available_end = window
        .completed_five_minute_summaries()
        .back()
        .map(|summary| summary.window_start);
    if available_start != Some(required_start) || available_end != Some(required_end) {
        return Err(DirectionalFeatureError::MissingPrewindowHistory {
            required_start,
            required_end,
            available_start,
            available_end,
        });
    }
    let mut selected = [None; BINANCE_PREWINDOW_SUMMARY_CAPACITY];
    let mut count = 0;
    for summary in window.completed_five_minute_summaries() {
        if summary.window_start < required_start {
            continue;
        }
        if summary.window_start > required_end {
            break;
        }
        if count == BINANCE_PREWINDOW_SUMMARY_CAPACITY {
            return Err(DirectionalFeatureError::MissingPrewindowHistory {
                required_start,
                required_end,
                available_start,
                available_end,
            });
        }
        let expected_window_start =
            required_start + chrono::Duration::seconds(MARKET_WINDOW_SECONDS * count as i64);
        if summary.window_start != expected_window_start {
            return Err(DirectionalFeatureError::GappedPrewindowHistory {
                expected_window_start,
                actual_window_start: summary.window_start,
            });
        }
        if !summary.source_complete {
            return Err(DirectionalFeatureError::IncompletePrewindowHistory {
                window_start: summary.window_start,
            });
        }
        selected[count] = Some(summary);
        count += 1;
    }
    if count != BINANCE_PREWINDOW_SUMMARY_CAPACITY {
        return Err(DirectionalFeatureError::MissingPrewindowHistory {
            required_start,
            required_end,
            available_start,
            available_end,
        });
    }
    Ok(selected.map(|summary| summary.expect("all pre-window summary slots were populated")))
}

#[derive(Debug, Clone, Copy)]
struct NumericCandle {
    high: f64,
    low: f64,
    close: f64,
    quote_volume: f64,
    trade_count: f64,
    taker_buy_quote_volume: f64,
}

impl TryFrom<&BinanceOneSecondKline> for NumericCandle {
    type Error = DirectionalFeatureError;

    fn try_from(candle: &BinanceOneSecondKline) -> Result<Self, Self::Error> {
        if !candle.source_complete {
            return Err(DirectionalFeatureError::IncompleteCandle {
                open_timestamp: candle.open_timestamp,
                synthetic: candle.synthetic,
            });
        }
        if candle.close_timestamp - candle.open_timestamp != chrono::Duration::seconds(1) {
            return Err(invalid_candle(candle, "timestamp interval"));
        }

        let open = decimal_to_f64(candle, "open_price", candle.open_price)?;
        let high = decimal_to_f64(candle, "high_price", candle.high_price)?;
        let low = decimal_to_f64(candle, "low_price", candle.low_price)?;
        let close = decimal_to_f64(candle, "close_price", candle.close_price)?;
        let base_volume = decimal_to_f64(candle, "base_volume", candle.base_volume)?;
        let quote_volume = decimal_to_f64(candle, "quote_volume", candle.quote_volume)?;
        let taker_buy_base_volume = decimal_to_f64(
            candle,
            "taker_buy_base_volume",
            candle.taker_buy_base_volume,
        )?;
        let taker_buy_quote_volume = decimal_to_f64(
            candle,
            "taker_buy_quote_volume",
            candle.taker_buy_quote_volume,
        )?;

        if open <= 0.0 {
            return Err(invalid_candle(candle, "open_price"));
        }
        if close <= 0.0 {
            return Err(invalid_candle(candle, "close_price"));
        }
        if low <= 0.0 || low > open.min(close) {
            return Err(invalid_candle(candle, "low_price"));
        }
        if high < open.max(close) || high < low {
            return Err(invalid_candle(candle, "high_price"));
        }
        if base_volume < 0.0 {
            return Err(invalid_candle(candle, "base_volume"));
        }
        if quote_volume < 0.0 {
            return Err(invalid_candle(candle, "quote_volume"));
        }
        if taker_buy_base_volume < 0.0 || taker_buy_base_volume > base_volume {
            return Err(invalid_candle(candle, "taker_buy_base_volume"));
        }
        if taker_buy_quote_volume < 0.0 || taker_buy_quote_volume > quote_volume {
            return Err(invalid_candle(candle, "taker_buy_quote_volume"));
        }
        Ok(Self {
            high,
            low,
            close,
            quote_volume,
            trade_count: candle.trade_count as f64,
            taker_buy_quote_volume,
        })
    }
}

fn decimal_to_f64(
    candle: &BinanceOneSecondKline,
    field: &'static str,
    value: rust_decimal::Decimal,
) -> Result<f64, DirectionalFeatureError> {
    value
        .to_f64()
        .filter(|converted| converted.is_finite())
        .ok_or_else(|| invalid_candle(candle, field))
}

fn invalid_candle(candle: &BinanceOneSecondKline, field: &'static str) -> DirectionalFeatureError {
    DirectionalFeatureError::InvalidCandle {
        open_timestamp: candle.open_timestamp,
        field,
    }
}

#[allow(clippy::too_many_arguments)]
fn append_external_features(
    values: &mut Vec<f64>,
    requirements: DirectionalExternalFeatureRequirements,
    external: &DirectionalExternalFeatureInputs<'_>,
    binance: &[NumericCandle],
    window_start: DateTime<Utc>,
    feature_as_of: DateTime<Utc>,
    seconds_elapsed: i64,
    opening_boundary: f64,
) -> Result<(), DirectionalFeatureError> {
    let btc_close = binance
        .last()
        .expect("validated directional history is non-empty")
        .close;
    let btc_close_30s = binance
        .get(binance.len().saturating_sub(31))
        .expect("directional candidates always contain 30 seconds of history")
        .close;
    let btc_return_30s_bps = (btc_close / btc_close_30s).ln() * BPS;
    let btc_path_from_window_open_bps = values[2];
    let btc_realized_volatility_60s_bps = values[11];

    values.extend_from_slice(&derive_oracle_features(
        external.oracle_rounds,
        window_start,
        feature_as_of,
        seconds_elapsed,
        opening_boundary,
        btc_close,
        btc_close_30s,
        btc_return_30s_bps,
        btc_path_from_window_open_bps,
        btc_realized_volatility_60s_bps,
    )?);
    if requirements.refprice {
        values.extend_from_slice(&derive_refprice_features(
            external.refprice_reports,
            feature_as_of,
            opening_boundary,
            btc_close,
            btc_return_30s_bps,
        )?);
    }
    values.extend_from_slice(&derive_chainlink_candle_features(
        external.chainlink_candles,
        feature_as_of,
    )?);
    if requirements.open_interest {
        values.extend_from_slice(&derive_open_interest_features(
            external.open_interest,
            feature_as_of,
            btc_path_from_window_open_bps,
        )?);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OracleIdentity {
    phase_id: i32,
    aggregator_round_id: i64,
}

#[allow(clippy::too_many_arguments)]
fn derive_oracle_features(
    rounds: &[DirectionalOracleRound],
    window_start: DateTime<Utc>,
    feature_as_of: DateTime<Utc>,
    seconds_elapsed: i64,
    opening_boundary: f64,
    btc_close: f64,
    btc_close_30s: f64,
    btc_return_30s_bps: f64,
    btc_path_from_window_open_bps: f64,
    btc_realized_volatility_60s_bps: f64,
) -> Result<[f64; 11], DirectionalFeatureError> {
    validate_oracle_rounds(rounds)?;
    let current = oracle_at(rounds, feature_as_of, feature_as_of).ok_or_else(|| {
        unavailable(
            "polygon_oracle",
            "no causal round exists at the decision time",
        )
    })?;
    let age_seconds = duration_seconds(current.block_timestamp, feature_as_of)?;
    if !(0.0..=BTC_DIRECTIONAL_ORACLE_MAX_AGE_SECONDS as f64).contains(&age_seconds) {
        return Err(unavailable(
            "polygon_oracle",
            "latest causal round exceeds the freshness bound",
        ));
    }
    let opening = oracle_at(rounds, window_start, feature_as_of).ok_or_else(|| {
        unavailable(
            "polygon_oracle",
            "no causal round exists at the market open",
        )
    })?;
    let anchor_30 = oracle_at(
        rounds,
        feature_as_of - chrono::Duration::seconds(30),
        feature_as_of,
    )
    .ok_or_else(|| unavailable("polygon_oracle", "the 30-second anchor is unavailable"))?;
    let anchor_60 = oracle_at(
        rounds,
        feature_as_of - chrono::Duration::seconds(60),
        feature_as_of,
    )
    .ok_or_else(|| unavailable("polygon_oracle", "the 60-second anchor is unavailable"))?;

    let current_price = positive_external_decimal(current.price, "polygon_oracle", "price")?;
    let opening_price = positive_external_decimal(opening.price, "polygon_oracle", "price")?;
    let price_30 = positive_external_decimal(anchor_30.price, "polygon_oracle", "price")?;
    let price_60 = positive_external_decimal(anchor_60.price, "polygon_oracle", "price")?;
    let oracle_return_30s_bps = (current_price / price_30).ln() * BPS;
    let oracle_return_60s_bps = (current_price / price_60).ln() * BPS;
    let binance_oracle_basis_bps = (btc_close / current_price).ln() * BPS;
    let prior_binance_oracle_basis_bps = (btc_close_30s / price_30).ln() * BPS;

    let mut update_count = 0_u32;
    let mut previous = oracle_identity(opening);
    for second in 1..=seconds_elapsed {
        let at = window_start + chrono::Duration::seconds(second);
        let selected = oracle_at(rounds, at, feature_as_of).ok_or_else(|| {
            unavailable(
                "polygon_oracle",
                "the market-window causal round path is incomplete",
            )
        })?;
        let identity = oracle_identity(selected);
        if identity != previous {
            update_count += 1;
            previous = identity;
        }
    }

    let oracle_gap_to_opening_boundary_bps = (current_price / opening_boundary).ln() * BPS;
    Ok([
        oracle_gap_to_opening_boundary_bps,
        (current_price / opening_price).ln() * BPS,
        oracle_return_30s_bps,
        oracle_return_60s_bps,
        age_seconds / MARKET_WINDOW_SECONDS as f64,
        update_count as f64 / (seconds_elapsed + 1) as f64,
        binance_oracle_basis_bps,
        binance_oracle_basis_bps - prior_binance_oracle_basis_bps,
        sign(oracle_return_30s_bps) * sign(btc_return_30s_bps),
        sign(oracle_gap_to_opening_boundary_bps) * sign(btc_path_from_window_open_bps),
        oracle_return_60s_bps / (btc_realized_volatility_60s_bps + EPSILON),
    ])
}

fn validate_oracle_rounds(
    rounds: &[DirectionalOracleRound],
) -> Result<(), DirectionalFeatureError> {
    if rounds.is_empty() {
        return Err(unavailable("polygon_oracle", "round history is empty"));
    }
    let mut previous = None;
    for round in rounds {
        let valid_provenance = match (round.block_number, round.log_index) {
            (Some(block_number), Some(log_index)) => block_number > 0 && log_index >= 0,
            (None, None) => true,
            _ => false,
        };
        if round.phase_id <= 0
            || round.aggregator_round_id <= 0
            || !valid_provenance
            || round.source_timestamp > round.block_timestamp
        {
            return Err(unavailable(
                "polygon_oracle",
                "round history contains an invalid or non-causal row",
            ));
        }
        positive_external_decimal(round.price, "polygon_oracle", "price")?;
        let order = (
            round.block_timestamp,
            round.source_timestamp,
            round.block_number,
            round.log_index,
            round.phase_id,
            round.aggregator_round_id,
        );
        if previous.is_some_and(|previous| previous >= order) {
            return Err(unavailable(
                "polygon_oracle",
                "round history is duplicate or out of order",
            ));
        }
        previous = Some(order);
    }
    Ok(())
}

fn oracle_at(
    rounds: &[DirectionalOracleRound],
    target: DateTime<Utc>,
    available_at: DateTime<Utc>,
) -> Option<&DirectionalOracleRound> {
    rounds
        .iter()
        .filter(|round| round.block_timestamp <= target && round.available_at <= available_at)
        .max_by_key(|round| {
            (
                round.block_timestamp,
                round.source_timestamp,
                round.block_number,
                round.log_index,
                round.phase_id,
                round.aggregator_round_id,
            )
        })
}

fn oracle_identity(round: &DirectionalOracleRound) -> OracleIdentity {
    OracleIdentity {
        phase_id: round.phase_id,
        aggregator_round_id: round.aggregator_round_id,
    }
}

fn derive_refprice_features(
    reports: &[DirectionalChainlinkRefPrice],
    feature_as_of: DateTime<Utc>,
    opening_boundary: f64,
    btc_close: f64,
    btc_return_30s_bps: f64,
) -> Result<[f64; 10], DirectionalFeatureError> {
    validate_refprice_reports(reports)?;
    let current = refprice_before(reports, feature_as_of, feature_as_of).ok_or_else(|| {
        unavailable(
            "chainlink_refprice",
            "no causal report exists before the decision",
        )
    })?;
    let current_age = duration_seconds(current.source_timestamp, feature_as_of)?;
    if current_age <= 0.0 || current_age > BTC_DIRECTIONAL_REFPRICE_MAX_AGE_SECONDS as f64 {
        return Err(unavailable(
            "chainlink_refprice",
            "latest causal report exceeds the freshness bound",
        ));
    }
    let mut anchors = [current; 5];
    for (index, seconds) in [1_i64, 5, 15, 30, 60].into_iter().enumerate() {
        let target = feature_as_of - chrono::Duration::seconds(seconds);
        let anchor = refprice_at_or_before(reports, target, feature_as_of).ok_or_else(|| {
            unavailable(
                "chainlink_refprice",
                "a required return anchor is unavailable",
            )
        })?;
        let age = duration_seconds(anchor.source_timestamp, target)?;
        if age < 0.0 || age > BTC_DIRECTIONAL_REFPRICE_MAX_AGE_SECONDS as f64 {
            return Err(unavailable(
                "chainlink_refprice",
                "a required return anchor exceeds the freshness bound",
            ));
        }
        anchors[index] = anchor;
    }
    let current_price = refprice_values(current)?.0;
    let mut returns = [0.0; 5];
    for (index, anchor) in anchors.into_iter().enumerate() {
        returns[index] = (current_price / refprice_values(anchor)?.0).ln() * BPS;
    }
    let current_spread = refprice_values(current)?.1;
    let spread_30s = refprice_values(anchors[3])?.1;
    Ok([
        returns[0],
        returns[1],
        returns[2],
        returns[3],
        returns[4],
        (current_price / btc_close).ln() * BPS,
        (current_price / opening_boundary).ln() * BPS,
        current_spread,
        current_spread - spread_30s,
        sign(btc_return_30s_bps) * sign(returns[3]),
    ])
}

fn validate_refprice_reports(
    reports: &[DirectionalChainlinkRefPrice],
) -> Result<(), DirectionalFeatureError> {
    if reports.is_empty() {
        return Err(unavailable("chainlink_refprice", "report history is empty"));
    }
    let mut previous = None;
    for report in reports {
        let (price, _) = refprice_values(report)?;
        let bid = positive_external_decimal(report.bid, "chainlink_refprice", "bid")?;
        let ask = positive_external_decimal(report.ask, "chainlink_refprice", "ask")?;
        if report.valid_from_timestamp > report.source_timestamp || bid > price || price > ask {
            return Err(unavailable(
                "chainlink_refprice",
                "report history contains an invalid signed price row",
            ));
        }
        if previous.is_some_and(|previous| previous >= report.source_timestamp) {
            return Err(unavailable(
                "chainlink_refprice",
                "report history is duplicate or out of order",
            ));
        }
        previous = Some(report.source_timestamp);
    }
    Ok(())
}

fn refprice_before(
    reports: &[DirectionalChainlinkRefPrice],
    target: DateTime<Utc>,
    available_at: DateTime<Utc>,
) -> Option<&DirectionalChainlinkRefPrice> {
    reports
        .iter()
        .rev()
        .find(|report| report.source_timestamp < target && report.available_at <= available_at)
}

fn refprice_at_or_before(
    reports: &[DirectionalChainlinkRefPrice],
    target: DateTime<Utc>,
    available_at: DateTime<Utc>,
) -> Option<&DirectionalChainlinkRefPrice> {
    reports
        .iter()
        .rev()
        .find(|report| report.source_timestamp <= target && report.available_at <= available_at)
}

fn refprice_values(
    report: &DirectionalChainlinkRefPrice,
) -> Result<(f64, f64), DirectionalFeatureError> {
    let price = positive_external_decimal(report.price, "chainlink_refprice", "price")?;
    let bid = positive_external_decimal(report.bid, "chainlink_refprice", "bid")?;
    let ask = positive_external_decimal(report.ask, "chainlink_refprice", "ask")?;
    Ok((price, (ask - bid) / price * BPS))
}

fn derive_chainlink_candle_features(
    candles: &[DirectionalChainlinkCandle],
    feature_as_of: DateTime<Utc>,
) -> Result<[f64; 8], DirectionalFeatureError> {
    validate_chainlink_candles(candles)?;
    let eligible = candles
        .iter()
        .filter(|candle| {
            candle.close_timestamp <= feature_as_of && candle.available_at <= feature_as_of
        })
        .collect::<Vec<_>>();
    let current = eligible
        .last()
        .copied()
        .ok_or_else(|| unavailable("chainlink_candles", "closed candle history is empty"))?;
    let age = duration_seconds(current.close_timestamp, feature_as_of)?;
    if !(0.0..BTC_DIRECTIONAL_CHAINLINK_CANDLE_MAX_AGE_SECONDS as f64).contains(&age) {
        return Err(unavailable(
            "chainlink_candles",
            "latest closed candle exceeds the freshness bound",
        ));
    }
    if eligible.len() < 61 {
        return Err(unavailable(
            "chainlink_candles",
            "61 contiguous closed candles are required",
        ));
    }
    let selected = &eligible[eligible.len() - 61..];
    if selected.windows(2).any(|pair| {
        pair[1].close_timestamp - pair[0].close_timestamp != chrono::Duration::minutes(1)
    }) {
        return Err(unavailable(
            "chainlink_candles",
            "required closed candle history is gapped",
        ));
    }
    let close = candle_price(current, "close_price", current.close_price)?;
    let mut returns = [0.0; 4];
    for (index, minutes) in [5_usize, 15, 30, 60].into_iter().enumerate() {
        let anchor = selected[selected.len() - 1 - minutes];
        returns[index] =
            (close / candle_price(anchor, "close_price", anchor.close_price)?).ln() * BPS;
    }
    let volatility_15 = candle_realized_volatility_bps(selected, 15)?;
    let volatility_60 = candle_realized_volatility_bps(selected, 60)?;
    let range_15 = candle_range_bps(selected, 15)?;
    let range_60 = candle_range_bps(selected, 60)?;
    Ok([
        returns[0],
        returns[1],
        returns[2],
        returns[3],
        volatility_15,
        volatility_60,
        range_15,
        range_60,
    ])
}

fn validate_chainlink_candles(
    candles: &[DirectionalChainlinkCandle],
) -> Result<(), DirectionalFeatureError> {
    let mut previous = None;
    for candle in candles {
        let open = candle_price(candle, "open_price", candle.open_price)?;
        let high = candle_price(candle, "high_price", candle.high_price)?;
        let low = candle_price(candle, "low_price", candle.low_price)?;
        let close = candle_price(candle, "close_price", candle.close_price)?;
        if candle.close_timestamp - candle.open_timestamp != chrono::Duration::minutes(1)
            || high < open.max(close)
            || high < low
            || low > open.min(close)
        {
            return Err(unavailable(
                "chainlink_candles",
                "candle history contains an invalid row",
            ));
        }
        if previous.is_some_and(|previous| previous >= candle.close_timestamp) {
            return Err(unavailable(
                "chainlink_candles",
                "candle history is duplicate or out of order",
            ));
        }
        previous = Some(candle.close_timestamp);
    }
    Ok(())
}

fn candle_price(
    _candle: &DirectionalChainlinkCandle,
    field: &'static str,
    value: Decimal,
) -> Result<f64, DirectionalFeatureError> {
    positive_external_decimal(value, "chainlink_candles", field)
}

fn candle_realized_volatility_bps(
    selected: &[&DirectionalChainlinkCandle],
    minutes: usize,
) -> Result<f64, DirectionalFeatureError> {
    let start = selected.len() - 1 - minutes;
    let mut returns = Vec::with_capacity(minutes);
    for pair in selected[start..].windows(2) {
        let prior = candle_price(pair[0], "close_price", pair[0].close_price)?;
        let current = candle_price(pair[1], "close_price", pair[1].close_price)?;
        returns.push((current / prior).ln());
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (returns.len() - 1) as f64;
    Ok(variance.sqrt() * BPS)
}

fn candle_range_bps(
    selected: &[&DirectionalChainlinkCandle],
    minutes: usize,
) -> Result<f64, DirectionalFeatureError> {
    let start = selected.len() - minutes;
    let mut high = f64::NEG_INFINITY;
    let mut low = f64::INFINITY;
    for candle in &selected[start..] {
        high = high.max(candle_price(candle, "high_price", candle.high_price)?);
        low = low.min(candle_price(candle, "low_price", candle.low_price)?);
    }
    Ok((high / low).ln() * BPS)
}

fn derive_open_interest_features(
    rows: &[DirectionalBinanceOpenInterest],
    feature_as_of: DateTime<Utc>,
    btc_path_from_window_open_bps: f64,
) -> Result<[f64; 9], DirectionalFeatureError> {
    validate_open_interest(rows)?;
    let eligible = rows
        .iter()
        .filter(|row| row.source_timestamp < feature_as_of && row.available_at <= feature_as_of)
        .collect::<Vec<_>>();
    let current = eligible
        .last()
        .copied()
        .ok_or_else(|| unavailable("binance_open_interest", "observation history is empty"))?;
    let age = duration_seconds(current.source_timestamp, feature_as_of)?;
    if age <= 0.0 || age > BTC_DIRECTIONAL_OPEN_INTEREST_MAX_AGE_SECONDS as f64 {
        return Err(unavailable(
            "binance_open_interest",
            "latest observation exceeds the freshness bound",
        ));
    }
    if eligible.len() < 13 {
        return Err(unavailable(
            "binance_open_interest",
            "13 contiguous observations are required",
        ));
    }
    let selected = &eligible[eligible.len() - 13..];
    if selected.windows(2).any(|pair| {
        pair[1].source_timestamp - pair[0].source_timestamp != chrono::Duration::minutes(5)
    }) {
        return Err(unavailable(
            "binance_open_interest",
            "required observation history is gapped",
        ));
    }
    let current_oi = positive_external_decimal(
        current.sum_open_interest,
        "binance_open_interest",
        "sum_open_interest",
    )?;
    let current_value = positive_external_decimal(
        current.sum_open_interest_value,
        "binance_open_interest",
        "sum_open_interest_value",
    )?;
    let mut changes = [0.0; 4];
    for (index, steps) in [1_usize, 3, 6, 12].into_iter().enumerate() {
        let anchor = selected[selected.len() - 1 - steps];
        let anchor_oi = positive_external_decimal(
            anchor.sum_open_interest,
            "binance_open_interest",
            "sum_open_interest",
        )?;
        changes[index] = (current_oi / anchor_oi).ln() * BPS;
    }
    let value_15 = positive_external_decimal(
        selected[selected.len() - 4].sum_open_interest_value,
        "binance_open_interest",
        "sum_open_interest_value",
    )?;
    let value_60 = positive_external_decimal(
        selected[0].sum_open_interest_value,
        "binance_open_interest",
        "sum_open_interest_value",
    )?;
    let value_change_15 = (current_value / value_15).ln() * BPS;
    let value_change_60 = (current_value / value_60).ln() * BPS;
    Ok([
        changes[0],
        changes[1],
        changes[2],
        changes[3],
        value_change_15,
        value_change_60,
        changes[0] - changes[2] / 6.0,
        sign(changes[1]) * sign(btc_path_from_window_open_bps),
        sign(changes[3]) * sign(btc_path_from_window_open_bps),
    ])
}

fn validate_open_interest(
    rows: &[DirectionalBinanceOpenInterest],
) -> Result<(), DirectionalFeatureError> {
    let mut previous = None;
    for row in rows {
        if row.period_seconds != 300 {
            return Err(unavailable(
                "binance_open_interest",
                "observation period is not five minutes",
            ));
        }
        positive_external_decimal(
            row.sum_open_interest,
            "binance_open_interest",
            "sum_open_interest",
        )?;
        positive_external_decimal(
            row.sum_open_interest_value,
            "binance_open_interest",
            "sum_open_interest_value",
        )?;
        if previous.is_some_and(|previous| previous >= row.source_timestamp) {
            return Err(unavailable(
                "binance_open_interest",
                "observation history is duplicate or out of order",
            ));
        }
        previous = Some(row.source_timestamp);
    }
    Ok(())
}

fn positive_external_decimal(
    value: Decimal,
    source: &'static str,
    _field: &'static str,
) -> Result<f64, DirectionalFeatureError> {
    value
        .to_f64()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| {
            unavailable(
                source,
                "source history contains a non-positive numeric value",
            )
        })
}

fn duration_seconds(
    earlier: DateTime<Utc>,
    later: DateTime<Utc>,
) -> Result<f64, DirectionalFeatureError> {
    later
        .signed_duration_since(earlier)
        .num_microseconds()
        .map(|microseconds| microseconds as f64 / 1_000_000.0)
        .ok_or_else(|| unavailable("external", "source timestamp distance overflowed"))
}

fn unavailable(source: &'static str, reason: &'static str) -> DirectionalFeatureError {
    DirectionalFeatureError::ExternalFeatureUnavailable { source, reason }
}

/// Reproduces the Python training frame before its frozen median imputer is applied.
/// Features whose lookback is not yet causally available are replaced with the exact
/// model-artifact median. The legacy directional builder never enters this path.
fn derive_asymmetric_early_core_features(
    candles: &[NumericCandle],
    feature_as_of: DateTime<Utc>,
    seconds_elapsed: i64,
    imputation_medians: &[f64],
) -> Result<DirectionalFeatureVector, DirectionalFeatureError> {
    debug_assert!(seconds_elapsed < BTC_DIRECTIONAL_FIRST_CANDIDATE_SECOND);
    debug_assert_eq!(candles.len(), seconds_elapsed as usize + 1);
    if imputation_medians.len() < BTC_DIRECTIONAL_FEATURE_COUNT {
        return Err(DirectionalFeatureError::ExternalFeatureUnavailable {
            source: "asymmetric_model",
            reason: "core imputation median width was smaller than the frozen core schema",
        });
    }

    let end = candles.len() - 1;
    let log_closes = candles
        .iter()
        .map(|candle| candle.close.ln())
        .collect::<Vec<_>>();
    let log_returns = (1..candles.len())
        .map(|index| log_closes[index] - log_closes[index - 1])
        .collect::<Vec<_>>();
    let path_from_open = (log_closes[end] - log_closes[0]) * BPS;

    let horizon = |seconds| early_horizon_return(&log_closes, end, seconds);
    let return_1 = horizon(1);
    let return_5 = horizon(5);
    let return_15 = horizon(15);
    let return_30 = horizon(30);
    let return_60 = horizon(60);
    let volatility = |window, minimum_samples| {
        rolling_volatility(&log_returns, end, window, minimum_samples).map(|value| value * BPS)
    };
    let volatility_5 = volatility(5, 2);
    let volatility_15 = volatility(15, 7);
    let volatility_30 = volatility(30, 15);
    let volatility_60 = volatility(60, 30);

    let high_low_5 = rolling_high_low_partial(candles, end, 5, 2);
    let high_low_30 = rolling_high_low_partial(candles, end, 30, 15);
    let high_low_60 = rolling_high_low_partial(candles, end, 60, 30);
    let range_bps = |high_low: Option<(f64, f64)>| {
        high_low.map(|(high, low)| (high - low) / candles[end].close * BPS)
    };
    let range_position = |high_low: Option<(f64, f64)>| {
        high_low.map(|(high, low)| (candles[end].close - low) / (high - low + EPSILON))
    };

    let quote_volume_5 = rolling_sum_partial(candles, end, 5, 2, |candle| candle.quote_volume);
    let quote_volume_30 = rolling_sum_partial(candles, end, 30, 15, |candle| candle.quote_volume);
    let quote_volume_60 = rolling_sum_partial(candles, end, 60, 30, |candle| candle.quote_volume);
    let trade_count_5 = rolling_sum_partial(candles, end, 5, 2, |candle| candle.trade_count);
    let trade_count_30 = rolling_sum_partial(candles, end, 30, 15, |candle| candle.trade_count);
    let trade_count_60 = rolling_sum_partial(candles, end, 60, 30, |candle| candle.trade_count);
    let taker_volume_5 =
        rolling_sum_partial(candles, end, 5, 2, |candle| candle.taker_buy_quote_volume);
    let taker_volume_30 =
        rolling_sum_partial(candles, end, 30, 15, |candle| candle.taker_buy_quote_volume);
    let taker_volume_60 =
        rolling_sum_partial(candles, end, 60, 30, |candle| candle.taker_buy_quote_volume);
    let share = |taker: Option<f64>, volume: Option<f64>| {
        taker
            .zip(volume)
            .map(|(taker, volume)| taker / (volume + EPSILON))
    };
    let flow = |taker: Option<f64>, volume: Option<f64>| {
        taker
            .zip(volume)
            .map(|(taker, volume)| (2.0 * taker - volume) / (volume + EPSILON))
    };
    let taker_share_5 = share(taker_volume_5, quote_volume_5);
    let taker_share_30 = share(taker_volume_30, quote_volume_30);
    let taker_share_60 = share(taker_volume_60, quote_volume_60);
    let signed_flow_5 = flow(taker_volume_5, quote_volume_5);
    let signed_flow_30 = flow(taker_volume_30, quote_volume_30);
    let signed_flow_60 = flow(taker_volume_60, quote_volume_60);

    let path_efficiency_30 = return_30
        .zip(rolling_absolute_return_partial(&log_returns, end, 30, 15))
        .map(|(ret, absolute)| ret.abs() / (absolute * BPS + EPSILON));
    let path_efficiency_60 = return_60
        .zip(rolling_absolute_return_partial(&log_returns, end, 60, 30))
        .map(|(ret, absolute)| ret.abs() / (absolute * BPS + EPSILON));
    let volatility_expansion = volatility_5
        .zip(volatility_30)
        .map(|(short, long)| short / (long + EPSILON));
    let volume_surprise = quote_volume_5
        .zip(quote_volume_60)
        .map(|(short, long)| short / (long / 12.0 + EPSILON));

    let path_stats = elapsed_path_stats(&log_closes);
    let (elapsed_volatility_sum, elapsed_volatility_count) = (0..=end)
        .filter_map(|row| rolling_volatility(&log_returns, row, 60, 30))
        .fold((0.0, 0_usize), |(sum, count), value| {
            (sum + value * BPS, count + 1)
        });
    let elapsed_volatility_mean = (elapsed_volatility_count > 0)
        .then_some(elapsed_volatility_sum / elapsed_volatility_count as f64);
    let terminal_denominator = volatility_60.map(|value| {
        value * ((MARKET_WINDOW_SECONDS - seconds_elapsed).max(1) as f64).sqrt() + EPSILON
    });
    let terminal_volatility_z =
        terminal_denominator.map(|denominator| path_from_open / denominator);
    let terminal_abs_volatility_z =
        terminal_denominator.map(|denominator| path_from_open.abs() / denominator);

    let signed = |value: Option<f64>| value.map(sign);
    let sign_5 = signed(return_5);
    let sign_15 = signed(return_15);
    let sign_30 = signed(return_30);
    let sign_60 = signed(return_60);
    let momentum_agreement_5_15 = sign_5.zip(sign_15).map(|(left, right)| left * right);
    let momentum_agreement_15_30 = sign_15.zip(sign_30).map(|(left, right)| left * right);
    let momentum_multihorizon = sign_5
        .zip(sign_15)
        .zip(sign_30)
        .zip(sign_60)
        .map(|(((a, b), c), d)| (a + b + c + d) / 4.0);
    let momentum_acceleration_5_30 = return_5
        .zip(return_30)
        .map(|(short, long)| short - long * (5.0 / 30.0));
    let momentum_acceleration_15_60 = return_15
        .zip(return_60)
        .map(|(short, long)| short - long * (15.0 / 60.0));
    let reversal_5_30 = return_5
        .zip(return_30)
        .map(|(short, long)| f64::from(short * long < 0.0));

    let close = candles[end].close;
    let distance_high =
        |high_low: Option<(f64, f64)>| high_low.map(|(high, _)| (high - close) / close * BPS);
    let distance_low =
        |high_low: Option<(f64, f64)>| high_low.map(|(_, low)| (close - low) / close * BPS);
    let volatility_regime = volatility_60
        .zip(elapsed_volatility_mean)
        .map(|(current, mean)| current / (mean + EPSILON));
    let flow_persistence_5_30 = signed_flow_5
        .zip(signed_flow_30)
        .map(|(short, long)| short * long);
    let flow_persistence_30_60 = signed_flow_30
        .zip(signed_flow_60)
        .map(|(short, long)| short * long);
    let price_flow_agreement_30 = sign_30
        .zip(signed_flow_30)
        .map(|(price, flow)| price * flow);
    let price_flow_divergence_30 = sign_30
        .zip(signed_flow_30)
        .map(|(price, flow)| price * -flow);

    let hour_angle = feature_as_of.hour() as f64 * std::f64::consts::TAU / 24.0;
    let weekday_angle =
        feature_as_of.weekday().number_from_monday() as f64 * std::f64::consts::TAU / 7.0;
    let raw = [
        Some(seconds_elapsed as f64 / MARKET_WINDOW_SECONDS as f64),
        Some((MARKET_WINDOW_SECONDS - seconds_elapsed) as f64 / MARKET_WINDOW_SECONDS as f64),
        Some(path_from_open),
        return_1,
        return_5,
        return_15,
        return_30,
        return_60,
        volatility_5,
        volatility_15,
        volatility_30,
        volatility_60,
        range_bps(high_low_5),
        range_bps(high_low_30),
        range_bps(high_low_60),
        path_efficiency_30,
        path_efficiency_60,
        range_position(high_low_30),
        volatility_expansion,
        quote_volume_5.map(f64::ln_1p),
        quote_volume_30.map(f64::ln_1p),
        quote_volume_60.map(f64::ln_1p),
        trade_count_5.map(f64::ln_1p),
        trade_count_30.map(f64::ln_1p),
        taker_share_5,
        taker_share_30,
        signed_flow_5,
        signed_flow_30,
        volume_surprise,
        Some(hour_angle.sin()),
        Some(hour_angle.cos()),
        Some(weekday_angle.sin()),
        Some(weekday_angle.cos()),
        terminal_volatility_z,
        terminal_abs_volatility_z,
        Some(path_stats.cross_count as f64),
        Some(path_stats.seconds_since_cross as f64),
        Some(path_stats.positive_fraction),
        Some(1.0 - path_stats.positive_fraction),
        momentum_agreement_5_15,
        momentum_agreement_15_30,
        momentum_multihorizon,
        momentum_acceleration_5_30,
        momentum_acceleration_15_60,
        reversal_5_30,
        range_position(high_low_60),
        distance_high(high_low_30),
        distance_low(high_low_30),
        distance_high(high_low_60),
        distance_low(high_low_60),
        volatility_regime,
        trade_count_60.map(f64::ln_1p),
        taker_share_60,
        signed_flow_60,
        flow_persistence_5_30,
        flow_persistence_30_60,
        price_flow_agreement_30,
        price_flow_divergence_30,
    ];
    let mut values = Vec::with_capacity(BTC_DIRECTIONAL_FEATURE_COUNT);
    for (index, candidate) in raw.into_iter().enumerate() {
        let value = candidate.unwrap_or(imputation_medians[index]);
        if !value.is_finite() {
            return Err(DirectionalFeatureError::NonFiniteFeature {
                index,
                name: BTC_DIRECTIONAL_FEATURE_NAMES[index],
            });
        }
        values.push(value);
    }
    Ok(DirectionalFeatureVector {
        schema_version: BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION,
        feature_as_of,
        seconds_elapsed: seconds_elapsed as u16,
        values,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn build_payoff_aware_feature_values(
    window: &BinanceOneSecondWindow,
    window_start: DateTime<Utc>,
    feature_as_of: DateTime<Utc>,
    opening_boundary: Decimal,
    external: &DirectionalExternalFeatureInputs<'_>,
    up_book: &OrderbookCheckpoint,
    down_book: &OrderbookCheckpoint,
    fee_rate: f64,
    feature_names: &[String],
) -> Result<Vec<f64>, DirectionalFeatureError> {
    let seconds_elapsed = (feature_as_of - window_start).num_seconds();
    if !(15..=240).contains(&seconds_elapsed) {
        return Err(DirectionalFeatureError::InvalidTiming {
            feature_as_of,
            seconds_elapsed,
            reason: if seconds_elapsed < 15 {
                DirectionalFeatureTimingReason::BeforeFirstCandidate
            } else {
                DirectionalFeatureTimingReason::AfterLastCandidate
            },
        });
    }
    let required_start = window_start - chrono::Duration::seconds(1);
    let required_end = feature_as_of - chrono::Duration::seconds(1);
    let completed = collect_required_candles(window, required_start, required_end)?;
    let candles = completed
        .into_iter()
        .map(NumericCandle::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let end = candles.len() - 1;
    let logs = candles
        .iter()
        .map(|candle| candle.close.ln())
        .collect::<Vec<_>>();
    let returns = (1..candles.len())
        .map(|index| logs[index] - logs[index - 1])
        .collect::<Vec<_>>();
    let boundary = opening_boundary
        .to_f64()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or(DirectionalFeatureError::InvalidOpeningBoundary)?;
    let close = candles[end].close;
    let path = (logs[end] - logs[0]) * BPS;
    let boundary_gap = (close / boundary).ln() * BPS;
    let window_basis = (candles[0].close / boundary).ln() * BPS;
    let mut values = HashMap::<String, f64>::new();
    let mut put = |name: &str, value: f64| {
        values.insert(name.to_string(), value);
    };
    put("seconds_elapsed_scaled", seconds_elapsed as f64 / 300.0);
    put(
        "seconds_remaining_scaled",
        (300 - seconds_elapsed) as f64 / 300.0,
    );
    put("btc_path_from_window_open_bps", path);
    put("btc_cross_venue_boundary_gap_bps", boundary_gap);
    put("btc_window_open_cross_venue_basis_bps", window_basis);
    let horizon = |seconds: usize| early_horizon_return(&logs, end, seconds).unwrap_or(f64::NAN);
    for seconds in [1, 5, 15, 30, 60, 90, 120, 180] {
        put(&format!("btc_return_{seconds}s_bps"), horizon(seconds));
    }
    let volatility = |seconds: usize| {
        let minimum = if seconds <= 60 {
            (seconds / 2).max(2)
        } else {
            seconds
        };
        rolling_volatility(&returns, end, seconds, minimum)
            .map(|value| value * BPS)
            .unwrap_or(f64::NAN)
    };
    for seconds in [5, 15, 30, 60, 90, 120, 180] {
        put(
            &format!("btc_realized_volatility_{seconds}s_bps"),
            volatility(seconds),
        );
    }
    let partial_range = |seconds: usize| {
        rolling_high_low_partial(&candles, end, seconds, (seconds / 2).max(2))
            .map(|(high, low)| (high - low) / close * BPS)
            .unwrap_or(f64::NAN)
    };
    for seconds in [5, 30, 60] {
        put(&format!("btc_range_{seconds}s_bps"), partial_range(seconds));
    }
    let return_5 = horizon(5);
    let return_15 = horizon(15);
    let return_30 = horizon(30);
    let return_60 = horizon(60);
    let absolute = |seconds: usize| {
        rolling_absolute_return_partial(&returns, end, seconds, (seconds / 2).max(2))
            .unwrap_or(f64::NAN)
            * BPS
    };
    put(
        "btc_path_efficiency_30s",
        return_30.abs() / (absolute(30) + EPSILON),
    );
    put(
        "btc_path_efficiency_60s",
        return_60.abs() / (absolute(60) + EPSILON),
    );
    let position = |seconds: usize| {
        rolling_high_low_partial(&candles, end, seconds, (seconds / 2).max(2))
            .map(|(high, low)| (close - low) / (high - low + EPSILON))
            .unwrap_or(f64::NAN)
    };
    put("btc_range_position_30s", position(30));
    put("btc_range_position_60s", position(60));
    let mut signed_flows = HashMap::new();
    let mut quote_volumes = HashMap::new();
    let mut trade_counts = HashMap::new();
    let mut buy_shares = HashMap::new();
    for seconds in [5, 30, 60, 90, 120, 180] {
        let minimum = if seconds <= 60 {
            (seconds / 2).max(1)
        } else {
            seconds
        };
        let quote = rolling_sum_partial(&candles, end, seconds, minimum, |candle| {
            candle.quote_volume
        })
        .unwrap_or(f64::NAN);
        let buy = rolling_sum_partial(&candles, end, seconds, minimum, |candle| {
            candle.taker_buy_quote_volume
        })
        .unwrap_or(f64::NAN);
        let trades =
            rolling_sum_partial(&candles, end, seconds, minimum, |candle| candle.trade_count)
                .unwrap_or(f64::NAN);
        let flow = (2.0 * buy - quote) / (quote + EPSILON);
        signed_flows.insert(seconds, flow);
        quote_volumes.insert(seconds, quote);
        trade_counts.insert(seconds, trades);
        buy_shares.insert(seconds, buy / (quote + EPSILON));
        put(&format!("btc_signed_flow_{seconds}s"), flow);
    }
    for seconds in [5, 30, 60] {
        put(
            &format!("btc_log_quote_volume_{seconds}s"),
            quote_volumes[&seconds].ln_1p(),
        );
        put(
            &format!("btc_log_trade_count_{seconds}s"),
            trade_counts[&seconds].ln_1p(),
        );
        put(
            &format!("btc_taker_buy_share_{seconds}s"),
            buy_shares[&seconds],
        );
    }
    let vol5 = volatility(5);
    let vol30 = volatility(30);
    let vol60 = volatility(60);
    let vol120 = volatility(120);
    let vol180 = volatility(180);
    put("btc_volatility_expansion_5_to_30", vol5 / (vol30 + EPSILON));
    put(
        "btc_volume_surprise_5_to_60",
        quote_volumes[&5] / (quote_volumes[&60] / 12.0 + EPSILON),
    );
    let path_stats = elapsed_path_stats(&logs);
    let boundary_stats = elapsed_anchor_stats(&logs, boundary.ln());
    put("btc_path_cross_count", path_stats.cross_count as f64);
    put(
        "btc_seconds_since_path_cross",
        path_stats.seconds_since_cross as f64,
    );
    put(
        "btc_fraction_time_path_positive",
        path_stats.positive_fraction,
    );
    put(
        "btc_fraction_time_path_negative",
        1.0 - path_stats.positive_fraction,
    );
    put(
        "btc_boundary_cross_count",
        boundary_stats.cross_count as f64,
    );
    put(
        "btc_seconds_since_boundary_cross",
        boundary_stats.seconds_since_cross as f64,
    );
    let terminal = vol60 * ((300 - seconds_elapsed).max(1) as f64).sqrt() + EPSILON;
    put("btc_path_terminal_volatility_z", path / terminal);
    put("btc_path_abs_terminal_volatility_z", path.abs() / terminal);
    put(
        "btc_boundary_terminal_volatility_z",
        boundary_gap / terminal,
    );
    let old_boundary = candles
        .get(end.saturating_sub(5))
        .map(|candle| (candle.close / boundary).ln() * BPS)
        .unwrap_or(boundary_gap);
    put(
        "btc_boundary_distance_velocity_5s_bps",
        boundary_gap.abs() - old_boundary.abs(),
    );
    put(
        "btc_boundary_momentum_alignment_5s",
        sign(boundary_gap) * sign(return_5),
    );
    put(
        "btc_momentum_agreement_5_15",
        sign(return_5) * sign(return_15),
    );
    put(
        "btc_momentum_agreement_15_30",
        sign(return_15) * sign(return_30),
    );
    put(
        "btc_momentum_multihorizon_score",
        (sign(return_5) + sign(return_15) + sign(return_30) + sign(return_60)) / 4.0,
    );
    put(
        "btc_momentum_acceleration_5_vs_30",
        return_5 - return_30 * 5.0 / 30.0,
    );
    put(
        "btc_momentum_acceleration_15_vs_60",
        return_15 - return_60 * 15.0 / 60.0,
    );
    put(
        "btc_reversal_5_vs_30",
        f64::from(return_5 * return_30 < 0.0),
    );
    let extremes = elapsed_path_extremes(&candles);
    let direction = if path >= 0.0 { 1.0 } else { -1.0 };
    let high_bps = (extremes.running_high / candles[0].close).ln() * BPS;
    let low_bps = (extremes.running_low / candles[0].close).ln() * BPS;
    let drawdown = (extremes.running_high / close).ln() * BPS;
    let rebound = (close / extremes.running_low).ln() * BPS;
    let (favorable, adverse, pullback, recovery) = if direction > 0.0 {
        (high_bps.max(0.0), (-low_bps).max(0.0), drawdown, rebound)
    } else {
        ((-low_bps).max(0.0), high_bps.max(0.0), rebound, drawdown)
    };
    put("btc_path_max_favorable_excursion_bps", favorable);
    put("btc_path_max_adverse_excursion_bps", adverse);
    put("btc_path_pullback_from_favorable_extreme_bps", pullback);
    put("btc_path_recovery_from_adverse_extreme_bps", recovery);
    put(
        "btc_seconds_since_path_high_scaled",
        extremes.seconds_since_high as f64 / 300.0,
    );
    put(
        "btc_seconds_since_path_low_scaled",
        extremes.seconds_since_low as f64 / 300.0,
    );
    put("btc_volatility_shock_30_vs_120", vol30 / (vol120 + EPSILON));
    put("btc_volatility_shock_60_vs_180", vol60 / (vol180 + EPSILON));
    for seconds in [5, 30, 60, 90, 120] {
        put(
            &format!("btc_path_sign_normalized_return_{seconds}s_bps"),
            direction * horizon(seconds),
        );
        put(
            &format!("btc_path_sign_normalized_flow_{seconds}s"),
            direction * signed_flows[&seconds],
        );
    }
    let (high30, low30) =
        rolling_high_low_partial(&candles, end, 30, 15).unwrap_or((f64::NAN, f64::NAN));
    let (high60, low60) =
        rolling_high_low_partial(&candles, end, 60, 30).unwrap_or((f64::NAN, f64::NAN));
    put(
        "btc_distance_from_high_30s_bps",
        (high30 - close) / close * BPS,
    );
    put(
        "btc_distance_from_low_30s_bps",
        (close - low30) / close * BPS,
    );
    put(
        "btc_distance_from_high_60s_bps",
        (high60 - close) / close * BPS,
    );
    put(
        "btc_distance_from_low_60s_bps",
        (close - low60) / close * BPS,
    );
    let elapsed_vols = (0..=end)
        .filter_map(|row| rolling_volatility(&returns, row, 60, 30))
        .map(|value| value * BPS)
        .collect::<Vec<_>>();
    let elapsed_mean = elapsed_vols.iter().sum::<f64>() / elapsed_vols.len().max(1) as f64;
    put(
        "btc_volatility_regime_60_vs_elapsed",
        vol60 / (elapsed_mean + EPSILON),
    );
    put(
        "btc_flow_persistence_5_30",
        signed_flows[&5] * signed_flows[&30],
    );
    put(
        "btc_flow_persistence_30_60",
        signed_flows[&30] * signed_flows[&60],
    );
    put(
        "btc_price_flow_agreement_30s",
        sign(return_30) * signed_flows[&30],
    );
    put(
        "btc_price_flow_divergence_30s",
        -sign(return_30) * signed_flows[&30],
    );
    let hour = feature_as_of.hour() as f64 * std::f64::consts::TAU / 24.0;
    let weekday = feature_as_of.weekday().number_from_monday() as f64 * std::f64::consts::TAU / 7.0;
    put("hour_sin", hour.sin());
    put("hour_cos", hour.cos());
    put("weekday_sin", weekday.sin());
    put("weekday_cos", weekday.cos());
    let close30 = candles
        .get(end.saturating_sub(30))
        .map(|candle| candle.close)
        .unwrap_or(close);
    let oracle = derive_oracle_features(
        external.oracle_rounds,
        window_start,
        feature_as_of,
        seconds_elapsed,
        boundary,
        close,
        close30,
        return_30,
        path,
        vol60,
    )?;
    for (name, value) in BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.iter().zip(oracle) {
        put(name, value);
    }
    put("early_oracle_eligible", 1.0);
    if feature_names
        .iter()
        .any(|name| name.starts_with("chainlink_candle_"))
    {
        let chainlink =
            derive_chainlink_candle_features(external.chainlink_candles, feature_as_of)?;
        for (name, value) in BTC_DIRECTIONAL_CHAINLINK_CANDLE_FEATURE_NAMES
            .iter()
            .zip(chainlink)
        {
            put(name, value);
        }
    }
    if feature_names
        .iter()
        .any(|name| name.starts_with("binance_oi_"))
    {
        let oi = derive_open_interest_features(external.open_interest, feature_as_of, path)?;
        for (name, value) in BTC_DIRECTIONAL_BINANCE_OPEN_INTEREST_FEATURE_NAMES
            .iter()
            .zip(oi)
        {
            put(name, value);
        }
    }
    append_payoff_book_features(&mut values, up_book, down_book, feature_as_of, fee_rate)?;
    feature_names
        .iter()
        .map(|name| {
            values.get(name).copied().ok_or_else(|| {
                unavailable("payoff_model", "required payoff feature is unavailable")
            })
        })
        .collect()
}

fn append_payoff_book_features(
    values: &mut HashMap<String, f64>,
    up: &OrderbookCheckpoint,
    down: &OrderbookCheckpoint,
    observed_at: DateTime<Utc>,
    fee_rate: f64,
) -> Result<(), DirectionalFeatureError> {
    let quantities = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200];
    let curve = |book: &OrderbookCheckpoint| -> Option<(Vec<f64>, f64)> {
        let mut asks = book.asks.iter().collect::<Vec<_>>();
        asks.sort_by_key(|level| level.price);
        let depth = asks
            .iter()
            .filter_map(|level| level.size.to_f64())
            .sum::<f64>();
        if depth < 800.0 {
            return None;
        }
        let mut output = Vec::new();
        for quantity in quantities {
            let mut remaining = quantity as f64;
            let mut notional = 0.0;
            for level in &asks {
                let size = level.size.to_f64()?;
                let price = level.price.to_f64()?;
                let take = remaining.min(size);
                notional += take * price;
                remaining -= take;
                if remaining <= 1e-12 {
                    break;
                }
            }
            if remaining > 1e-9 {
                return None;
            }
            output.push(notional / quantity as f64);
        }
        Some((output, depth))
    };
    let (up_curve, up_depth) = curve(up).ok_or_else(|| {
        unavailable(
            "polymarket_book",
            "UP book lacks the frozen full-ladder depth",
        )
    })?;
    let (down_curve, down_depth) = curve(down).ok_or_else(|| {
        unavailable(
            "polymarket_book",
            "DOWN book lacks the frozen full-ladder depth",
        )
    })?;
    for (index, quantity) in quantities.iter().enumerate() {
        values.insert(format!("up_ask_vwap_{quantity}"), up_curve[index]);
        values.insert(format!("down_ask_vwap_{quantity}"), down_curve[index]);
    }
    values.insert(
        "pm_vwap5_overround".into(),
        up_curve[0] + down_curve[0] - 1.0,
    );
    values.insert(
        "pm_vwap50_overround".into(),
        up_curve[7] + down_curve[7] - 1.0,
    );
    values.insert(
        "pm_vwap200_overround".into(),
        up_curve[13] + down_curve[13] - 1.0,
    );
    for (side, curve) in [("up", &up_curve), ("down", &down_curve)] {
        values.insert(format!("pm_{side}_slope_5_25"), curve[4] - curve[0]);
        values.insert(format!("pm_{side}_slope_25_100"), curve[9] - curve[4]);
        values.insert(format!("pm_{side}_slope_100_200"), curve[13] - curve[9]);
    }
    values.insert("pm_up_depth_log".into(), up_depth.ln_1p());
    values.insert("pm_down_depth_log".into(), down_depth.ln_1p());
    values.insert(
        "pm_depth_imbalance".into(),
        (up_depth - down_depth) / (up_depth + down_depth).max(EPSILON),
    );
    let up_age = duration_seconds(up.received_at, observed_at)?;
    let down_age = duration_seconds(down.received_at, observed_at)?;
    values.insert("pm_up_book_age_seconds".into(), up_age);
    values.insert("pm_down_book_age_seconds".into(), down_age);
    let reserve = 0.005;
    let up_cost = up_curve[0] + fee_rate * up_curve[0] * (1.0 - up_curve[0]) + reserve;
    let down_cost = down_curve[0] + fee_rate * down_curve[0] * (1.0 - down_curve[0]) + reserve;
    let up_best = up
        .best_ask
        .and_then(|value| value.to_f64())
        .ok_or_else(|| unavailable("polymarket_book", "UP best ask is missing"))?;
    let down_best = down
        .best_ask
        .and_then(|value| value.to_f64())
        .ok_or_else(|| unavailable("polymarket_book", "DOWN best ask is missing"))?;
    for (name, value) in [
        ("pm_yes_cost_per_share", up_cost),
        ("pm_no_cost_per_share", down_cost),
        ("pm_yes_vwap_slippage", up_curve[0] - up_best),
        ("pm_no_vwap_slippage", down_curve[0] - down_best),
        ("pm_yes_depth_log", up_depth.ln_1p()),
        ("pm_no_depth_log", down_depth.ln_1p()),
        ("pm_yes_book_age_seconds", up_age),
        ("pm_no_book_age_seconds", down_age),
        (
            "pm_yes_cost_logit",
            (up_cost.clamp(1e-6, 1.0 - 1e-6) / (1.0 - up_cost.clamp(1e-6, 1.0 - 1e-6))).ln(),
        ),
        (
            "pm_no_cost_logit",
            (down_cost.clamp(1e-6, 1.0 - 1e-6) / (1.0 - down_cost.clamp(1e-6, 1.0 - 1e-6))).ln(),
        ),
        ("pm_cost_overround", up_cost + down_cost - 1.0),
        ("pm_yes_minus_no_cost", up_cost - down_cost),
        ("fee_rate", fee_rate),
    ] {
        values.insert(name.into(), value);
    }
    Ok(())
}

fn derive_directional_features(
    candles: &[NumericCandle],
    feature_as_of: DateTime<Utc>,
    seconds_elapsed: i64,
    schema_version: &'static str,
    opening_boundary: Option<f64>,
    prewindow_summaries: Option<&[&BinanceFiveMinuteSummary; BINANCE_PREWINDOW_SUMMARY_CAPACITY]>,
) -> Result<DirectionalFeatureVector, DirectionalFeatureError> {
    debug_assert_eq!(candles.len(), seconds_elapsed as usize + 1);
    let end = candles.len() - 1;
    let log_closes = candles
        .iter()
        .map(|candle| candle.close.ln())
        .collect::<Vec<_>>();
    let log_returns = (1..candles.len())
        .map(|index| log_closes[index] - log_closes[index - 1])
        .collect::<Vec<_>>();

    let path_from_open = (log_closes[end] - log_closes[0]) * BPS;
    let return_1 = horizon_return(&log_closes, end, 1);
    let return_5 = horizon_return(&log_closes, end, 5);
    let return_15 = horizon_return(&log_closes, end, 15);
    let return_30 = horizon_return(&log_closes, end, 30);
    let return_60 = horizon_return(&log_closes, end, 60);

    let volatility_5 = rolling_volatility(&log_returns, end, 5, 2).unwrap() * BPS;
    let volatility_15 = rolling_volatility(&log_returns, end, 15, 7).unwrap() * BPS;
    let volatility_30 = rolling_volatility(&log_returns, end, 30, 15).unwrap() * BPS;
    let volatility_60 = rolling_volatility(&log_returns, end, 60, 30).unwrap() * BPS;

    let range_5 = rolling_range_bps(candles, end, 5);
    let range_30 = rolling_range_bps(candles, end, 30);
    let range_60 = rolling_range_bps(candles, end, 60);
    let path_efficiency_30 =
        return_30.abs() / (rolling_absolute_return(&log_returns, end, 30) * BPS + EPSILON);
    let path_efficiency_60 =
        return_60.abs() / (rolling_absolute_return(&log_returns, end, 60) * BPS + EPSILON);
    let range_position_30 = rolling_range_position(candles, end, 30);
    let range_position_60 = rolling_range_position(candles, end, 60);
    let volatility_expansion = volatility_5 / (volatility_30 + EPSILON);

    let quote_volume_5 = rolling_sum(candles, end, 5, |candle| candle.quote_volume);
    let quote_volume_30 = rolling_sum(candles, end, 30, |candle| candle.quote_volume);
    let quote_volume_60 = rolling_sum(candles, end, 60, |candle| candle.quote_volume);
    let trade_count_5 = rolling_sum(candles, end, 5, |candle| candle.trade_count);
    let trade_count_30 = rolling_sum(candles, end, 30, |candle| candle.trade_count);
    let trade_count_60 = rolling_sum(candles, end, 60, |candle| candle.trade_count);
    let taker_buy_quote_volume_5 =
        rolling_sum(candles, end, 5, |candle| candle.taker_buy_quote_volume);
    let taker_buy_quote_volume_30 =
        rolling_sum(candles, end, 30, |candle| candle.taker_buy_quote_volume);
    let taker_buy_quote_volume_60 =
        rolling_sum(candles, end, 60, |candle| candle.taker_buy_quote_volume);
    let taker_buy_share_5 = taker_buy_quote_volume_5 / (quote_volume_5 + EPSILON);
    let taker_buy_share_30 = taker_buy_quote_volume_30 / (quote_volume_30 + EPSILON);
    let taker_buy_share_60 = taker_buy_quote_volume_60 / (quote_volume_60 + EPSILON);
    let signed_flow_5 =
        (2.0 * taker_buy_quote_volume_5 - quote_volume_5) / (quote_volume_5 + EPSILON);
    let signed_flow_30 =
        (2.0 * taker_buy_quote_volume_30 - quote_volume_30) / (quote_volume_30 + EPSILON);
    let signed_flow_60 =
        (2.0 * taker_buy_quote_volume_60 - quote_volume_60) / (quote_volume_60 + EPSILON);
    let volume_surprise = quote_volume_5 / (quote_volume_60 / 12.0 + EPSILON);

    let path_stats = elapsed_path_stats(&log_closes);
    let (elapsed_volatility_sum, elapsed_volatility_count) = (0..=end)
        .filter_map(|row| rolling_volatility(&log_returns, row, 60, 30))
        .fold((0.0, 0_usize), |(sum, count), volatility| {
            (sum + volatility * BPS, count + 1)
        });
    let elapsed_volatility_mean = elapsed_volatility_sum / elapsed_volatility_count.max(1) as f64;
    let seconds_remaining = (MARKET_WINDOW_SECONDS - seconds_elapsed).max(1);
    let terminal_denominator = volatility_60 * (seconds_remaining as f64).sqrt() + EPSILON;
    let terminal_volatility_z = path_from_open / terminal_denominator;
    let path_abs_terminal_volatility_z = path_from_open.abs() / terminal_denominator;

    let sign_5 = sign(return_5);
    let sign_15 = sign(return_15);
    let sign_30 = sign(return_30);
    let sign_60 = sign(return_60);
    let momentum_agreement_5_15 = sign_5 * sign_15;
    let momentum_agreement_15_30 = sign_15 * sign_30;
    let momentum_multihorizon_score = (sign_5 + sign_15 + sign_30 + sign_60) / 4.0;
    let momentum_acceleration_5_vs_30 = return_5 - return_30 * (5.0 / 30.0);
    let momentum_acceleration_15_vs_60 = return_15 - return_60 * (15.0 / 60.0);
    let reversal_5_vs_30 = f64::from(return_5 * return_30 < 0.0);

    let (rolling_high_30, rolling_low_30) = rolling_high_low(candles, end, 30);
    let (rolling_high_60, rolling_low_60) = rolling_high_low(candles, end, 60);
    let close = candles[end].close;
    let distance_from_high_30 = (rolling_high_30 - close) / close * BPS;
    let distance_from_low_30 = (close - rolling_low_30) / close * BPS;
    let distance_from_high_60 = (rolling_high_60 - close) / close * BPS;
    let distance_from_low_60 = (close - rolling_low_60) / close * BPS;
    let volatility_regime_60_vs_elapsed = volatility_60 / (elapsed_volatility_mean + EPSILON);
    let flow_persistence_5_30 = signed_flow_5 * signed_flow_30;
    let flow_persistence_30_60 = signed_flow_30 * signed_flow_60;
    let price_flow_agreement_30 = sign_30 * signed_flow_30;
    let price_flow_divergence_30 = sign_30 * -signed_flow_30;

    let hour_angle = feature_as_of.hour() as f64 * std::f64::consts::TAU / 24.0;
    let weekday_angle =
        feature_as_of.weekday().number_from_monday() as f64 * std::f64::consts::TAU / 7.0;

    let mut values = vec![
        seconds_elapsed as f64 / MARKET_WINDOW_SECONDS as f64,
        (MARKET_WINDOW_SECONDS - seconds_elapsed) as f64 / MARKET_WINDOW_SECONDS as f64,
        path_from_open,
        return_1,
        return_5,
        return_15,
        return_30,
        return_60,
        volatility_5,
        volatility_15,
        volatility_30,
        volatility_60,
        range_5,
        range_30,
        range_60,
        path_efficiency_30,
        path_efficiency_60,
        range_position_30,
        volatility_expansion,
        quote_volume_5.ln_1p(),
        quote_volume_30.ln_1p(),
        quote_volume_60.ln_1p(),
        trade_count_5.ln_1p(),
        trade_count_30.ln_1p(),
        taker_buy_share_5,
        taker_buy_share_30,
        signed_flow_5,
        signed_flow_30,
        volume_surprise,
        hour_angle.sin(),
        hour_angle.cos(),
        weekday_angle.sin(),
        weekday_angle.cos(),
        terminal_volatility_z,
        path_abs_terminal_volatility_z,
        path_stats.cross_count as f64,
        path_stats.seconds_since_cross as f64,
        path_stats.positive_fraction,
        1.0 - path_stats.positive_fraction,
        momentum_agreement_5_15,
        momentum_agreement_15_30,
        momentum_multihorizon_score,
        momentum_acceleration_5_vs_30,
        momentum_acceleration_15_vs_60,
        reversal_5_vs_30,
        range_position_60,
        distance_from_high_30,
        distance_from_low_30,
        distance_from_high_60,
        distance_from_low_60,
        volatility_regime_60_vs_elapsed,
        trade_count_60.ln_1p(),
        taker_buy_share_60,
        signed_flow_60,
        flow_persistence_5_30,
        flow_persistence_30_60,
        price_flow_agreement_30,
        price_flow_divergence_30,
    ];
    if schema_version == BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION {
        let path_direction_sign = if path_from_open >= 0.0 { 1.0 } else { -1.0 };
        let path_extremes = elapsed_path_extremes(candles);
        let window_open_close = candles[0].close;
        let running_path_high_bps = (path_extremes.running_high / window_open_close).ln() * BPS;
        let running_path_low_bps = (path_extremes.running_low / window_open_close).ln() * BPS;
        let path_drawdown_from_high_bps = (path_extremes.running_high / close).ln() * BPS;
        let path_rebound_from_low_bps = (close / path_extremes.running_low).ln() * BPS;
        let (
            path_max_favorable_excursion_bps,
            path_max_adverse_excursion_bps,
            path_pullback_from_favorable_extreme_bps,
            path_recovery_from_adverse_extreme_bps,
        ) = if path_direction_sign > 0.0 {
            (
                running_path_high_bps.max(0.0),
                (-running_path_low_bps).max(0.0),
                path_drawdown_from_high_bps,
                path_rebound_from_low_bps,
            )
        } else {
            (
                (-running_path_low_bps).max(0.0),
                running_path_high_bps.max(0.0),
                path_rebound_from_low_bps,
                path_drawdown_from_high_bps,
            )
        };
        values.extend_from_slice(&[
            path_max_favorable_excursion_bps,
            path_max_adverse_excursion_bps,
            path_pullback_from_favorable_extreme_bps,
            path_recovery_from_adverse_extreme_bps,
            path_extremes.seconds_since_high as f64 / MARKET_WINDOW_SECONDS as f64,
            path_extremes.seconds_since_low as f64 / MARKET_WINDOW_SECONDS as f64,
            path_direction_sign * return_5,
            path_direction_sign * return_15,
            path_direction_sign * return_30,
            path_direction_sign * return_60,
            path_direction_sign * signed_flow_5,
            path_direction_sign * signed_flow_30,
            path_direction_sign * signed_flow_60,
        ]);
    } else if schema_version == BTC_DIRECTIONAL_BOUNDARY_FEATURE_SCHEMA_VERSION {
        let opening_boundary =
            opening_boundary.expect("boundary schema validates its opening boundary");
        let boundary_log_price = opening_boundary.ln();
        let boundary_gap = (log_closes[end] - boundary_log_price) * BPS;
        let window_open_basis = (log_closes[0] - boundary_log_price) * BPS;
        let boundary_stats = elapsed_anchor_stats(&log_closes, boundary_log_price);
        let boundary_gap_5 = (log_closes[end - 5] - boundary_log_price) * BPS;
        values.extend_from_slice(&[
            boundary_gap,
            window_open_basis,
            boundary_gap / terminal_denominator,
            boundary_gap.abs() / terminal_denominator,
            boundary_stats.cross_count as f64,
            boundary_stats.seconds_since_cross as f64,
            boundary_stats.positive_fraction,
            1.0 - boundary_stats.positive_fraction,
            boundary_gap.abs() - boundary_gap_5.abs(),
            sign(boundary_gap) * sign_5,
        ]);
    } else if schema_version == BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SCHEMA_VERSION {
        let summaries =
            prewindow_summaries.expect("path pre-window schema validates summary history");
        values.extend_from_slice(&derive_path_prewindow_features(
            summaries,
            candles[0].close,
            path_from_open,
        ));
    }
    let feature_names = directional_feature_names(schema_version)
        .expect("feature schema was canonicalized before derivation");
    debug_assert_eq!(values.len(), feature_names.len());
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(DirectionalFeatureError::NonFiniteFeature {
            index,
            name: feature_names[index],
        });
    }

    Ok(DirectionalFeatureVector {
        schema_version,
        feature_as_of,
        seconds_elapsed: seconds_elapsed as u16,
        values,
    })
}

fn derive_path_prewindow_features(
    summaries: &[&BinanceFiveMinuteSummary; BINANCE_PREWINDOW_SUMMARY_CAPACITY],
    current_open_available_close: f64,
    current_path_bps: f64,
) -> [f64; BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_COUNT - BTC_DIRECTIONAL_FEATURE_COUNT] {
    const HORIZON_WINDOWS: [usize; 4] = [1, 3, 6, 12];
    let mut values =
        [0.0; BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_COUNT - BTC_DIRECTIONAL_FEATURE_COUNT];
    let mut returns = [0.0; HORIZON_WINDOWS.len()];
    for (horizon_index, windows) in HORIZON_WINDOWS.into_iter().enumerate() {
        let selected = &summaries[BINANCE_PREWINDOW_SUMMARY_CAPACITY - windows..];
        let prewindow_return =
            (current_open_available_close / selected[0].open_available_close).ln() * BPS;
        let (window_high, window_low) = selected.iter().fold(
            (f64::NEG_INFINITY, f64::INFINITY),
            |(high, low), summary| (high.max(summary.window_high), low.min(summary.window_low)),
        );
        let quote_volume = selected
            .iter()
            .map(|summary| summary.window_quote_volume)
            .sum::<f64>();
        let trade_count = selected
            .iter()
            .map(|summary| summary.window_trade_count)
            .sum::<f64>();
        let taker_buy_quote_volume = selected
            .iter()
            .map(|summary| summary.window_taker_buy_quote_volume)
            .sum::<f64>();
        let volatility = (selected
            .iter()
            .map(|summary| summary.window_volatility_bps.powi(2))
            .sum::<f64>()
            / windows as f64)
            .sqrt();
        let base = horizon_index * 7;
        returns[horizon_index] = prewindow_return;
        values[base] = prewindow_return;
        values[base + 1] = (window_high - window_low) / current_open_available_close * BPS;
        values[base + 2] = volatility;
        values[base + 3] = quote_volume.ln_1p();
        values[base + 4] = trade_count.ln_1p();
        values[base + 5] = taker_buy_quote_volume / (quote_volume + EPSILON);
        values[base + 6] = (2.0 * taker_buy_quote_volume - quote_volume) / (quote_volume + EPSILON);
    }

    values[28] = sign(returns[0]) * sign(returns[1]);
    values[29] = sign(returns[0]) * sign(returns[2]);
    values[30] = sign(returns[0]) * sign(returns[3]);
    values[31] = returns[0] - returns[1] / 3.0;
    values[32] = returns[0] - returns[2] / 6.0;
    values[33] = returns[0] - returns[3] / 12.0;
    for (horizon_index, prewindow_return) in returns.into_iter().enumerate() {
        let base = 34 + horizon_index * 2;
        values[base] = sign(current_path_bps) * sign(prewindow_return);
        values[base + 1] = f64::from(current_path_bps * prewindow_return < 0.0);
    }
    values
}

fn horizon_return(log_closes: &[f64], end: usize, seconds: usize) -> f64 {
    (log_closes[end] - log_closes[end - seconds]) * BPS
}

fn early_horizon_return(log_closes: &[f64], end: usize, seconds: usize) -> Option<f64> {
    (end >= seconds).then(|| (log_closes[end] - log_closes[end - seconds]) * BPS)
}

fn rolling_volatility(
    log_returns: &[f64],
    row: usize,
    window: usize,
    minimum_samples: usize,
) -> Option<f64> {
    if row == 0 {
        return None;
    }
    // `log_returns[i - 1]` is the one-second return stored on candle row `i`.
    let first_row = (row + 1).saturating_sub(window).max(1);
    let samples = &log_returns[first_row - 1..row];
    if samples.len() < minimum_samples || samples.len() < 2 {
        return None;
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let squared_deviations = samples
        .iter()
        .map(|sample| {
            let deviation = sample - mean;
            deviation * deviation
        })
        .sum::<f64>();
    Some((squared_deviations / (samples.len() - 1) as f64).sqrt())
}

fn rolling_absolute_return(log_returns: &[f64], end: usize, window: usize) -> f64 {
    let first_row = end + 1 - window;
    log_returns[first_row - 1..end]
        .iter()
        .map(|value| value.abs())
        .sum()
}

fn rolling_absolute_return_partial(
    log_returns: &[f64],
    end: usize,
    window: usize,
    minimum_samples: usize,
) -> Option<f64> {
    if end == 0 {
        return None;
    }
    let first_row = (end + 1).saturating_sub(window).max(1);
    let samples = &log_returns[first_row - 1..end];
    (samples.len() >= minimum_samples).then(|| samples.iter().map(|value| value.abs()).sum())
}

fn rolling_sum(
    candles: &[NumericCandle],
    end: usize,
    window: usize,
    value: impl Fn(&NumericCandle) -> f64,
) -> f64 {
    candles[end + 1 - window..=end].iter().map(value).sum()
}

fn rolling_sum_partial(
    candles: &[NumericCandle],
    end: usize,
    window: usize,
    minimum_samples: usize,
    value: impl Fn(&NumericCandle) -> f64,
) -> Option<f64> {
    let first = (end + 1).saturating_sub(window);
    let samples = &candles[first..=end];
    (samples.len() >= minimum_samples).then(|| samples.iter().map(value).sum())
}

fn rolling_high_low(candles: &[NumericCandle], end: usize, window: usize) -> (f64, f64) {
    candles[end + 1 - window..=end]
        .iter()
        .fold((f64::NEG_INFINITY, f64::INFINITY), |(high, low), candle| {
            (high.max(candle.high), low.min(candle.low))
        })
}

fn rolling_high_low_partial(
    candles: &[NumericCandle],
    end: usize,
    window: usize,
    minimum_samples: usize,
) -> Option<(f64, f64)> {
    let first = (end + 1).saturating_sub(window);
    let samples = &candles[first..=end];
    (samples.len() >= minimum_samples).then(|| {
        samples
            .iter()
            .fold((f64::NEG_INFINITY, f64::INFINITY), |(high, low), candle| {
                (high.max(candle.high), low.min(candle.low))
            })
    })
}

fn rolling_range_bps(candles: &[NumericCandle], end: usize, window: usize) -> f64 {
    let (high, low) = rolling_high_low(candles, end, window);
    (high - low) / candles[end].close * BPS
}

fn rolling_range_position(candles: &[NumericCandle], end: usize, window: usize) -> f64 {
    let (high, low) = rolling_high_low(candles, end, window);
    (candles[end].close - low) / (high - low + EPSILON)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ElapsedPathStats {
    cross_count: usize,
    seconds_since_cross: usize,
    positive_fraction: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ElapsedPathExtremes {
    running_high: f64,
    running_low: f64,
    seconds_since_high: usize,
    seconds_since_low: usize,
}

fn elapsed_path_extremes(candles: &[NumericCandle]) -> ElapsedPathExtremes {
    let mut running_high = f64::NEG_INFINITY;
    let mut running_low = f64::INFINITY;
    let mut last_high_second = 0;
    let mut last_low_second = 0;
    for (second, candle) in candles.iter().enumerate() {
        running_high = running_high.max(candle.high);
        running_low = running_low.min(candle.low);
        // Python updates the most recent extreme on equality as well as on a new extreme.
        if candle.high >= running_high {
            last_high_second = second;
        }
        if candle.low <= running_low {
            last_low_second = second;
        }
    }
    let end = candles.len() - 1;
    ElapsedPathExtremes {
        running_high,
        running_low,
        seconds_since_high: end - last_high_second,
        seconds_since_low: end - last_low_second,
    }
}

fn elapsed_path_stats(log_closes: &[f64]) -> ElapsedPathStats {
    elapsed_anchor_stats(log_closes, log_closes[0])
}

fn elapsed_anchor_stats(log_closes: &[f64], anchor: f64) -> ElapsedPathStats {
    let mut previous_positive = log_closes[0] - anchor >= 0.0;
    let mut positive_count = 0;
    let mut cross_count = 0;
    let mut last_cross = None;

    for (second, close) in log_closes.iter().enumerate() {
        let positive = close - anchor >= 0.0;
        positive_count += usize::from(positive);
        if second > 0 && positive != previous_positive {
            cross_count += 1;
            last_cross = Some(second);
        }
        previous_positive = positive;
    }
    let end = log_closes.len() - 1;
    ElapsedPathStats {
        cross_count,
        // Python fills a never-crossed last-cross value with the current second, yielding zero.
        seconds_since_cross: end - last_cross.unwrap_or(end),
        positive_fraction: positive_count as f64 / log_closes.len() as f64,
    }
}

fn sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, TimeZone};
    use rust_decimal::Decimal;

    use super::*;

    const PYTHON_GOLDEN_SECOND_240: [f64; BTC_DIRECTIONAL_FEATURE_COUNT] = [
        0.8,
        0.2,
        2.1997580354885677,
        -9.46343781263792,
        -7.345684740727165,
        -2.0493390953291168,
        -4.098258297613455,
        1.799766035261996,
        4.468950041507519,
        2.580583822710948,
        2.519423501676631,
        2.1877390169194246,
        9.637879666474248,
        10.097778488732478,
        10.897602527443963,
        0.12227523143444172,
        0.03084757723254909,
        0.05148514851431291,
        1.7737986633687086,
        15.64006001130563,
        17.422420476346527,
        18.10326086697242,
        6.282266746896006,
        8.08794755464267,
        0.48999999999999994,
        0.49,
        -0.019999999999999997,
        -0.02,
        1.0219429277840733,
        1.2246467991473532e-16,
        -1.0,
        -2.4492935982947064e-16,
        1.0,
        0.12980869245587595,
        0.12980869245587595,
        18.0,
        79.0,
        0.8049792531120332,
        0.19502074688796678,
        1.0,
        1.0,
        -0.5,
        -6.662641691124922,
        -2.499280604144616,
        0.0,
        0.12110091743005494,
        9.577892863570305,
        0.5198856251621733,
        9.577892863570305,
        1.3197096638736567,
        1.0053854114897578,
        8.778479952508487,
        0.49,
        -0.02,
        0.00039999999999999996,
        0.00040000000000000002,
        0.02,
        -0.02,
    ];

    const PYTHON_MATURE_REVERSAL_SUFFIX_SECOND_240: [f64;
        BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SUFFIX_NAMES.len()] = [
        11.773067024164103,
        4.5210218279224943,
        9.5733089886757377,
        6.7207798634092528,
        0.0033333333333333335,
        0.79666666666666675,
        -7.3456847407271653,
        -2.0493390953291168,
        -4.0982582976134552,
        1.799766035261996,
        -0.019999999999999997,
        -0.02,
        -0.02,
    ];

    const PYTHON_BOUNDARY_SUFFIX_SECOND_240: [f64; BTC_DIRECTIONAL_BOUNDARY_FEATURE_SUFFIX_NAMES
        .len()] = [
        1.199808032155038,
        -0.9999500033332495,
        0.07080120146828416,
        0.07080120146828416,
        19.0,
        58.0,
        0.7178423236514523,
        0.2821576763485477,
        -7.3456847407145505,
        -1.0,
    ];

    const PYTHON_PATH_PREWINDOW_SUFFIX_SECOND_240: [f64;
        BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SUFFIX_NAMES.len()] = [
        2.296451991781443,
        8.635664796437913,
        1.194265333751639,
        19.817884137115634,
        10.385605012165374,
        0.49000301690828724,
        -0.01999396618342555,
        -0.49915891732780154,
        11.745502465906014,
        1.2177671557935095,
        20.89398787957333,
        11.484484837338378,
        0.4900030855845033,
        -0.019993828830993432,
        6.3914356940399495,
        16.17814428049467,
        1.2064276468049047,
        21.552392817022486,
        12.177621729356158,
        0.4900031946685695,
        -0.019993610662860997,
        20.106856894113257,
        25.158237326038776,
        1.2012517690489037,
        22.17221029686565,
        12.870756049089538,
        0.49000340720942975,
        -0.019993185581140546,
        -1.0,
        1.0,
        1.0,
        2.4628382975573766,
        1.2312127094414516,
        0.6208805839386717,
        1.0,
        0.0,
        -1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
    ];

    #[test]
    fn feature_names_match_frozen_python_order() {
        assert_eq!(BTC_DIRECTIONAL_FEATURE_NAMES.len(), 58);
        assert_eq!(BTC_DIRECTIONAL_FEATURE_NAMES[0], "seconds_elapsed_scaled");
        assert_eq!(BTC_DIRECTIONAL_FEATURE_NAMES[32], "weekday_cos");
        assert_eq!(
            BTC_DIRECTIONAL_FEATURE_NAMES[57],
            "btc_price_flow_divergence_30s"
        );
    }

    #[test]
    fn feature_vector_matches_python_polars_fixture() {
        let window_start = Utc.with_ymd_and_hms(2026, 6, 14, 12, 35, 0).unwrap();
        let window = BinanceOneSecondWindow::from_completed(
            (0..=240)
                .map(|second| fixture_candle(window_start, second))
                .collect(),
        )
        .unwrap();
        let features = build_directional_features(
            &window,
            window_start,
            window_start + Duration::seconds(240),
        )
        .unwrap();

        assert_eq!(features.seconds_elapsed, 240);
        assert_eq!(features.values.len(), BTC_DIRECTIONAL_FEATURE_COUNT);
        for (index, (actual, expected)) in features
            .values
            .iter()
            .zip(PYTHON_GOLDEN_SECOND_240)
            .enumerate()
        {
            let tolerance = 2e-10_f64.max(expected.abs() * 2e-11);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{} mismatch at {index}: actual={actual:.17}, expected={expected:.17}, \
                 tolerance={tolerance:.3e}",
                BTC_DIRECTIONAL_FEATURE_NAMES[index],
            );
        }
    }

    #[test]
    fn mature_reversal_schema_has_exact_order_and_matches_python_suffix() {
        assert_eq!(BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_NAMES.len(), 71);
        assert_eq!(
            &BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_NAMES[..BTC_DIRECTIONAL_FEATURE_COUNT],
            &BTC_DIRECTIONAL_FEATURE_NAMES
        );
        assert_eq!(
            &BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_NAMES[BTC_DIRECTIONAL_FEATURE_COUNT..],
            &BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SUFFIX_NAMES
        );

        let window_start = Utc.with_ymd_and_hms(2026, 6, 14, 12, 35, 0).unwrap();
        let window = BinanceOneSecondWindow::from_completed(
            (0..=240)
                .map(|second| fixture_candle(window_start, second))
                .collect(),
        )
        .unwrap();
        let core = build_directional_features(
            &window,
            window_start,
            window_start + Duration::seconds(240),
        )
        .unwrap();
        let mature = build_directional_features_for_schema(
            &window,
            window_start,
            window_start + Duration::seconds(240),
            BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
        )
        .unwrap();

        assert_eq!(
            mature.schema_version(),
            BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION
        );
        assert_eq!(
            mature.names(),
            &BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_NAMES
        );
        assert_eq!(
            &mature.values[..BTC_DIRECTIONAL_FEATURE_COUNT],
            core.values.as_slice()
        );
        for (index, (actual, expected)) in mature.values[BTC_DIRECTIONAL_FEATURE_COUNT..]
            .iter()
            .zip(PYTHON_MATURE_REVERSAL_SUFFIX_SECOND_240)
            .enumerate()
        {
            let tolerance = 2e-10_f64.max(expected.abs() * 2e-11);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{} mismatch: actual={actual:.17}, expected={expected:.17}, \
                 tolerance={tolerance:.3e}",
                BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SUFFIX_NAMES[index],
            );
        }
    }

    #[test]
    fn mature_reversal_zero_path_uses_positive_sign_and_latest_tied_extremes() {
        let window_start = Utc.with_ymd_and_hms(2026, 6, 14, 12, 35, 0).unwrap();
        let price = Decimal::new(100_000_000, 3);
        let candles = (0..=60)
            .map(|second| {
                let mut candle = fixture_candle(window_start, second);
                candle.open_price = price;
                candle.high_price = price;
                candle.low_price = price;
                candle.close_price = price;
                candle.base_volume = candle.quote_volume / price;
                candle.taker_buy_base_volume = candle.taker_buy_quote_volume / price;
                candle
            })
            .collect();
        let window = BinanceOneSecondWindow::from_completed(candles).unwrap();
        let features = build_directional_features_for_schema(
            &window,
            window_start,
            window_start + Duration::seconds(60),
            BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
        )
        .unwrap();
        let suffix = &features.values[BTC_DIRECTIONAL_FEATURE_COUNT..];

        assert_eq!(&suffix[..6], &[0.0; 6]);
        assert_eq!(suffix[6], 0.0);
        assert_eq!(suffix[7], 0.0);
        assert_eq!(suffix[8], 0.0);
        assert_eq!(suffix[9], 0.0);
        assert_eq!(suffix[10], features.values[26]);
        assert_eq!(suffix[11], features.values[27]);
        assert_eq!(suffix[12], features.values[53]);
    }

    #[test]
    fn boundary_schema_has_exact_python_order_and_values() {
        assert_eq!(BTC_DIRECTIONAL_BOUNDARY_FEATURE_NAMES.len(), 68);
        assert_eq!(
            &BTC_DIRECTIONAL_BOUNDARY_FEATURE_NAMES[..BTC_DIRECTIONAL_FEATURE_COUNT],
            &BTC_DIRECTIONAL_FEATURE_NAMES
        );
        assert_eq!(
            &BTC_DIRECTIONAL_BOUNDARY_FEATURE_NAMES[BTC_DIRECTIONAL_FEATURE_COUNT..],
            &BTC_DIRECTIONAL_BOUNDARY_FEATURE_SUFFIX_NAMES
        );

        let window_start = Utc.with_ymd_and_hms(2026, 6, 14, 12, 35, 0).unwrap();
        let window = BinanceOneSecondWindow::from_completed(
            (0..=240)
                .map(|second| fixture_candle(window_start, second))
                .collect(),
        )
        .unwrap();
        let core = build_directional_features(
            &window,
            window_start,
            window_start + Duration::seconds(240),
        )
        .unwrap();
        let boundary = build_directional_features_for_schema_with_boundary(
            &window,
            window_start,
            window_start + Duration::seconds(240),
            BTC_DIRECTIONAL_BOUNDARY_FEATURE_SCHEMA_VERSION,
            Some(Decimal::new(100_010, 0)),
        )
        .unwrap();

        assert_eq!(
            boundary.schema_version(),
            BTC_DIRECTIONAL_BOUNDARY_FEATURE_SCHEMA_VERSION
        );
        assert_eq!(boundary.names(), &BTC_DIRECTIONAL_BOUNDARY_FEATURE_NAMES);
        assert_eq!(
            &boundary.values[..BTC_DIRECTIONAL_FEATURE_COUNT],
            core.values.as_slice()
        );
        for (index, (actual, expected)) in boundary.values[BTC_DIRECTIONAL_FEATURE_COUNT..]
            .iter()
            .zip(PYTHON_BOUNDARY_SUFFIX_SECOND_240)
            .enumerate()
        {
            let tolerance = 2e-10_f64.max(expected.abs() * 2e-11);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{} mismatch: actual={actual:.17}, expected={expected:.17}, \
                 tolerance={tolerance:.3e}",
                BTC_DIRECTIONAL_BOUNDARY_FEATURE_SUFFIX_NAMES[index],
            );
        }
    }

    #[test]
    fn boundary_schema_requires_a_positive_opening_boundary() {
        let window_start = Utc.with_ymd_and_hms(2026, 6, 14, 12, 35, 0).unwrap();
        let as_of = window_start + Duration::seconds(60);
        let window = BinanceOneSecondWindow::from_completed(
            (0..=60)
                .map(|second| fixture_candle(window_start, second))
                .collect(),
        )
        .unwrap();

        assert_eq!(
            build_directional_features_for_schema(
                &window,
                window_start,
                as_of,
                BTC_DIRECTIONAL_BOUNDARY_FEATURE_SCHEMA_VERSION,
            ),
            Err(DirectionalFeatureError::MissingOpeningBoundary)
        );
        assert_eq!(
            build_directional_features_for_schema_with_boundary(
                &window,
                window_start,
                as_of,
                BTC_DIRECTIONAL_BOUNDARY_FEATURE_SCHEMA_VERSION,
                Some(Decimal::ZERO),
            ),
            Err(DirectionalFeatureError::InvalidOpeningBoundary)
        );
    }

    #[test]
    fn external_schemas_have_exact_frozen_orders_and_requirements() {
        assert_eq!(
            &BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_NAMES
                [..BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT],
            &BTC_DIRECTIONAL_BOUNDARY_FEATURE_NAMES
        );
        assert_eq!(
            &BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_NAMES
                [BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT
                    ..BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT
                        + BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.len()],
            &BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES
        );
        assert_eq!(
            &BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_NAMES
                [BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT
                    + BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.len()..],
            &BTC_DIRECTIONAL_CHAINLINK_CANDLE_FEATURE_NAMES
        );

        assert_eq!(
            &BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_NAMES
                [..BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT],
            &BTC_DIRECTIONAL_BOUNDARY_FEATURE_NAMES
        );
        assert_eq!(
            &BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_NAMES
                [BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT
                    ..BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT
                        + BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.len()],
            &BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES
        );
        assert_eq!(
            &BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_NAMES
                [BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT + BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.len()
                    ..BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT
                        + BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.len()
                        + BTC_DIRECTIONAL_CHAINLINK_REFPRICE_FEATURE_NAMES.len()],
            &BTC_DIRECTIONAL_CHAINLINK_REFPRICE_FEATURE_NAMES
        );
        assert_eq!(
            &BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_NAMES
                [BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT
                    + BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.len()
                    + BTC_DIRECTIONAL_CHAINLINK_REFPRICE_FEATURE_NAMES.len()..],
            &BTC_DIRECTIONAL_CHAINLINK_CANDLE_FEATURE_NAMES
        );

        assert_eq!(
            &BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_NAMES
                [..BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_COUNT],
            &BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_NAMES
        );
        assert_eq!(
            &BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_NAMES
                [BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_COUNT..],
            &BTC_DIRECTIONAL_BINANCE_OPEN_INTEREST_FEATURE_NAMES
        );

        assert_eq!(
            directional_external_feature_requirements(
                BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_SCHEMA_VERSION
            ),
            DirectionalExternalFeatureRequirements {
                oracle: true,
                refprice: false,
                chainlink_candles: true,
                open_interest: false,
            }
        );
        assert_eq!(
            directional_external_feature_requirements(
                BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_SCHEMA_VERSION
            ),
            DirectionalExternalFeatureRequirements {
                oracle: true,
                refprice: true,
                chainlink_candles: true,
                open_interest: false,
            }
        );
        assert_eq!(
            directional_external_feature_requirements(
                BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_SCHEMA_VERSION
            ),
            DirectionalExternalFeatureRequirements {
                oracle: true,
                refprice: true,
                chainlink_candles: true,
                open_interest: true,
            }
        );
    }

    #[test]
    fn external_schemas_derive_causal_exact_width_feature_vectors() {
        let window_start = Utc.with_ymd_and_hms(2026, 7, 10, 12, 35, 0).unwrap();
        let as_of = window_start + Duration::seconds(240);
        let opening_boundary = Decimal::new(100_010, 0);
        let window = BinanceOneSecondWindow::from_completed(
            (0..=240)
                .map(|second| fixture_candle(window_start, second))
                .collect(),
        )
        .unwrap();
        let (oracle, refprice, candles, open_interest) = external_fixture(window_start, as_of);
        let external = DirectionalExternalFeatureInputs {
            oracle_rounds: &oracle,
            refprice_reports: &refprice,
            chainlink_candles: &candles,
            open_interest: &open_interest,
        };
        let boundary = build_directional_features_for_schema_with_boundary(
            &window,
            window_start,
            as_of,
            BTC_DIRECTIONAL_BOUNDARY_FEATURE_SCHEMA_VERSION,
            Some(opening_boundary),
        )
        .unwrap();
        let long = build_directional_features_for_schema_with_external(
            &window,
            window_start,
            as_of,
            BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_SCHEMA_VERSION,
            Some(opening_boundary),
            Some(&external),
        )
        .unwrap();
        let full = build_directional_features_for_schema_with_external(
            &window,
            window_start,
            as_of,
            BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_SCHEMA_VERSION,
            Some(opening_boundary),
            Some(&external),
        )
        .unwrap();
        let full_oi = build_directional_features_for_schema_with_external(
            &window,
            window_start,
            as_of,
            BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_SCHEMA_VERSION,
            Some(opening_boundary),
            Some(&external),
        )
        .unwrap();

        assert_eq!(
            long.values.len(),
            BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_COUNT
        );
        assert_eq!(
            full.values.len(),
            BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_COUNT
        );
        assert_eq!(
            full_oi.values.len(),
            BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_COUNT
        );
        assert_eq!(
            &long.values[..BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT],
            boundary.values.as_slice()
        );
        assert_eq!(
            &full.values[..BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT],
            boundary.values.as_slice()
        );
        assert_eq!(
            &full_oi.values
                [..BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_COUNT],
            full.values.as_slice()
        );
        assert_eq!(
            &long.values[BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT
                ..BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT
                    + BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.len()],
            &full.values[BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT
                ..BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT
                    + BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.len()]
        );
        assert_eq!(
            &long.values[BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT
                + BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.len()..],
            &full.values[BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT
                + BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.len()
                + BTC_DIRECTIONAL_CHAINLINK_REFPRICE_FEATURE_NAMES.len()..]
        );
        assert!(full_oi.values.iter().all(|value| value.is_finite()));

        let oracle_offset = BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT;
        assert_close(
            long.values[oracle_offset],
            (99_836.0_f64 / 100_010.0).ln() * BPS,
        );
        assert_close(long.values[oracle_offset + 4], 0.0);
        assert_close(long.values[oracle_offset + 5], 24.0 / 241.0);

        let refprice_offset = oracle_offset + BTC_DIRECTIONAL_ORACLE_FEATURE_NAMES.len();
        assert_close(full.values[refprice_offset], 0.0);
        assert_close(
            full.values[refprice_offset + 1],
            (100_069.0_f64 / 100_065.0).ln() * BPS,
        );
        assert_close(full.values[refprice_offset + 7], 1.0 / 100_069.0 * BPS);

        let oi_offset = BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_FEATURE_COUNT;
        assert_close(
            full_oi.values[oi_offset],
            (1_013_000.0_f64 / 1_012_000.0).ln() * BPS,
        );
        assert_close(
            full_oi.values[oi_offset + 3],
            (1_013_000.0_f64 / 1_001_000.0).ln() * BPS,
        );
    }

    #[test]
    fn external_feature_builder_ignores_future_source_rows() {
        let window_start = Utc.with_ymd_and_hms(2026, 7, 10, 12, 35, 0).unwrap();
        let as_of = window_start + Duration::seconds(240);
        let window = BinanceOneSecondWindow::from_completed(
            (0..=240)
                .map(|second| fixture_candle(window_start, second))
                .collect(),
        )
        .unwrap();
        let (mut oracle, mut refprice, mut candles, mut open_interest) =
            external_fixture(window_start, as_of);
        let baseline_external = DirectionalExternalFeatureInputs {
            oracle_rounds: &oracle,
            refprice_reports: &refprice,
            chainlink_candles: &candles,
            open_interest: &open_interest,
        };
        let baseline = build_directional_features_for_schema_with_external(
            &window,
            window_start,
            as_of,
            BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_SCHEMA_VERSION,
            Some(Decimal::new(100_010, 0)),
            Some(&baseline_external),
        )
        .unwrap();

        oracle.last_mut().unwrap().price = Decimal::new(200_000, 0);
        for report in refprice.iter_mut().rev().take(2) {
            report.price = Decimal::new(200_000, 0);
            report.bid = Decimal::new(199_999, 0);
            report.ask = Decimal::new(200_001, 0);
        }
        let future_candle = candles.last_mut().unwrap();
        future_candle.open_price = Decimal::new(200_000, 0);
        future_candle.high_price = Decimal::new(200_001, 0);
        future_candle.low_price = Decimal::new(199_999, 0);
        future_candle.close_price = Decimal::new(200_000, 0);
        let future_oi = open_interest.last_mut().unwrap();
        future_oi.sum_open_interest = Decimal::new(2_000_000, 0);
        future_oi.sum_open_interest_value = Decimal::new(20_000_000, 0);
        let with_future_external = DirectionalExternalFeatureInputs {
            oracle_rounds: &oracle,
            refprice_reports: &refprice,
            chainlink_candles: &candles,
            open_interest: &open_interest,
        };
        let with_future = build_directional_features_for_schema_with_external(
            &window,
            window_start,
            as_of,
            BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_SCHEMA_VERSION,
            Some(Decimal::new(100_010, 0)),
            Some(&with_future_external),
        )
        .unwrap();

        assert_eq!(baseline, with_future);
    }

    #[test]
    fn required_external_sources_fail_closed_when_missing_stale_or_gapped() {
        let window_start = Utc.with_ymd_and_hms(2026, 7, 10, 12, 35, 0).unwrap();
        let as_of = window_start + Duration::seconds(240);
        let window = BinanceOneSecondWindow::from_completed(
            (0..=240)
                .map(|second| fixture_candle(window_start, second))
                .collect(),
        )
        .unwrap();
        assert!(matches!(
            build_directional_features_for_schema_with_external(
                &window,
                window_start,
                as_of,
                BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_SCHEMA_VERSION,
                Some(Decimal::new(100_010, 0)),
                None,
            ),
            Err(DirectionalFeatureError::ExternalFeatureUnavailable {
                source: "external",
                ..
            })
        ));

        let (oracle, refprice, mut candles, mut open_interest) =
            external_fixture(window_start, as_of);
        candles.remove(candles.len() - 20);
        let gapped = DirectionalExternalFeatureInputs {
            oracle_rounds: &oracle,
            refprice_reports: &refprice,
            chainlink_candles: &candles,
            open_interest: &open_interest,
        };
        assert!(matches!(
            build_directional_features_for_schema_with_external(
                &window,
                window_start,
                as_of,
                BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_CANDLE_FEATURE_SCHEMA_VERSION,
                Some(Decimal::new(100_010, 0)),
                Some(&gapped),
            ),
            Err(DirectionalFeatureError::ExternalFeatureUnavailable {
                source: "chainlink_candles",
                ..
            })
        ));

        open_interest.truncate(open_interest.len() - 3);
        let (_, _, complete_candles, _) = external_fixture(window_start, as_of);
        let stale = DirectionalExternalFeatureInputs {
            oracle_rounds: &oracle,
            refprice_reports: &refprice,
            chainlink_candles: &complete_candles,
            open_interest: &open_interest,
        };
        assert!(matches!(
            build_directional_features_for_schema_with_external(
                &window,
                window_start,
                as_of,
                BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_SCHEMA_VERSION,
                Some(Decimal::new(100_010, 0)),
                Some(&stale),
            ),
            Err(DirectionalFeatureError::ExternalFeatureUnavailable {
                source: "binance_open_interest",
                ..
            })
        ));
    }

    #[test]
    fn oracle_archive_provenance_is_paired_and_optional_for_runtime_rounds() {
        let window_start = Utc.with_ymd_and_hms(2026, 7, 10, 12, 35, 0).unwrap();
        let as_of = window_start + Duration::seconds(240);
        let (mut oracle, _, _, _) = external_fixture(window_start, as_of);
        for round in &mut oracle {
            round.block_number = None;
            round.log_index = None;
        }
        assert!(validate_oracle_rounds(&oracle).is_ok());

        oracle[0].block_number = Some(1);
        assert!(matches!(
            validate_oracle_rounds(&oracle),
            Err(DirectionalFeatureError::ExternalFeatureUnavailable {
                source: "polygon_oracle",
                ..
            })
        ));
    }

    #[test]
    fn path_prewindow_schema_is_compact_and_matches_python() {
        assert_eq!(BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_NAMES.len(), 100);
        assert_eq!(
            &BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_NAMES[..BTC_DIRECTIONAL_FEATURE_COUNT],
            &BTC_DIRECTIONAL_FEATURE_NAMES
        );
        assert_eq!(
            &BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_NAMES[BTC_DIRECTIONAL_FEATURE_COUNT..],
            &BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SUFFIX_NAMES
        );

        let window_start = Utc.with_ymd_and_hms(2026, 6, 14, 12, 35, 0).unwrap();
        let window = BinanceOneSecondWindow::from_completed(
            (0..3_841)
                .map(|index| prewindow_fixture_candle(window_start, index))
                .collect(),
        )
        .unwrap();
        let as_of = window_start + Duration::seconds(240);
        let core = build_directional_features(&window, window_start, as_of).unwrap();
        let path = build_directional_features_for_schema(
            &window,
            window_start,
            as_of,
            BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SCHEMA_VERSION,
        )
        .unwrap();

        assert_eq!(
            path.schema_version(),
            BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SCHEMA_VERSION
        );
        assert_eq!(path.names(), &BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_NAMES);
        assert_eq!(
            &path.values[..BTC_DIRECTIONAL_FEATURE_COUNT],
            core.values.as_slice()
        );
        assert_eq!(
            window.completed().len(),
            super::super::types::BINANCE_ONE_SECOND_WINDOW_CAPACITY
        );
        assert_eq!(
            window.completed_five_minute_summaries().len(),
            BINANCE_PREWINDOW_SUMMARY_CAPACITY
        );
        for (index, (actual, expected)) in path.values[BTC_DIRECTIONAL_FEATURE_COUNT..]
            .iter()
            .zip(PYTHON_PATH_PREWINDOW_SUFFIX_SECOND_240)
            .enumerate()
        {
            let tolerance = 2e-10_f64.max(expected.abs() * 2e-11);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{} mismatch: actual={actual:.17}, expected={expected:.17}, \
                 tolerance={tolerance:.3e}",
                BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SUFFIX_NAMES[index],
            );
        }
    }

    #[test]
    fn path_prewindow_schema_fails_closed_without_complete_prior_windows() {
        let window_start = Utc.with_ymd_and_hms(2026, 6, 14, 12, 35, 0).unwrap();
        let as_of = window_start + Duration::seconds(240);
        let only_eleven_windows = BinanceOneSecondWindow::from_completed(
            (300..3_841)
                .map(|index| prewindow_fixture_candle(window_start, index))
                .collect(),
        )
        .unwrap();
        assert!(matches!(
            build_directional_features_for_schema(
                &only_eleven_windows,
                window_start,
                as_of,
                BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SCHEMA_VERSION,
            ),
            Err(DirectionalFeatureError::MissingPrewindowHistory { .. })
        ));

        let mut incomplete_candles = (0..3_841)
            .map(|index| prewindow_fixture_candle(window_start, index))
            .collect::<Vec<_>>();
        incomplete_candles[100].source_complete = false;
        let incomplete = BinanceOneSecondWindow::from_completed(incomplete_candles).unwrap();
        assert!(matches!(
            build_directional_features_for_schema(
                &incomplete,
                window_start,
                as_of,
                BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SCHEMA_VERSION,
            ),
            Err(DirectionalFeatureError::IncompletePrewindowHistory { .. })
        ));
    }

    #[test]
    fn schema_selected_builder_ignores_future_candles() {
        let window_start = Utc.with_ymd_and_hms(2026, 6, 14, 12, 35, 0).unwrap();
        let as_of = window_start + Duration::seconds(120);
        let observed = BinanceOneSecondWindow::from_completed(
            (0..=120)
                .map(|second| fixture_candle(window_start, second))
                .collect(),
        )
        .unwrap();
        let with_future = BinanceOneSecondWindow::from_completed(
            (0..=240)
                .map(|second| fixture_candle(window_start, second))
                .collect(),
        )
        .unwrap();

        let observed = build_directional_features_for_schema(
            &observed,
            window_start,
            as_of,
            BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
        )
        .unwrap();
        let with_future = build_directional_features_for_schema(
            &with_future,
            window_start,
            as_of,
            BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
        )
        .unwrap();
        assert_eq!(observed, with_future);
    }

    #[test]
    fn only_exact_policy_candidates_are_accepted() {
        let window_start = Utc.with_ymd_and_hms(2026, 6, 14, 12, 35, 0).unwrap();
        let valid = [60, 65, 120, 235, 240];
        for second in valid {
            assert_eq!(
                validate_feature_time(
                    window_start,
                    window_start + Duration::seconds(second),
                    DirectionalFeatureTimingPolicy::LegacyDirectional,
                ),
                Ok(second)
            );
        }
        assert_timing_error(
            window_start,
            59,
            DirectionalFeatureTimingReason::BeforeFirstCandidate,
        );
        assert_timing_error(window_start, 61, DirectionalFeatureTimingReason::OffCadence);
        assert_timing_error(
            window_start,
            241,
            DirectionalFeatureTimingReason::AfterLastCandidate,
        );
        assert_eq!(
            validate_feature_time(
                window_start,
                window_start + Duration::seconds(60) + Duration::milliseconds(1),
                DirectionalFeatureTimingPolicy::LegacyDirectional,
            ),
            Err(DirectionalFeatureError::InvalidTiming {
                feature_as_of: window_start + Duration::seconds(60) + Duration::milliseconds(1),
                seconds_elapsed: 60,
                reason: DirectionalFeatureTimingReason::NotWholeSecond,
            })
        );
    }

    #[test]
    fn asymmetric_value_timing_accepts_early_seconds_without_relaxing_legacy_timing() {
        let window_start = Utc.with_ymd_and_hms(2026, 6, 14, 12, 35, 0).unwrap();
        let window = BinanceOneSecondWindow::from_completed(
            (0..=61)
                .map(|second| fixture_candle(window_start, second))
                .collect(),
        )
        .unwrap();

        for second in [1, 5, 30, 55, 59, 60] {
            let features = build_directional_features_for_asymmetric_value(
                &window,
                window_start,
                window_start + Duration::seconds(second),
                &[42.0; BTC_DIRECTIONAL_FEATURE_COUNT],
            )
            .unwrap();
            assert_eq!(features.values.len(), BTC_DIRECTIONAL_FEATURE_COUNT);
        }
        let first_second = build_directional_features_for_asymmetric_value(
            &window,
            window_start,
            window_start + Duration::seconds(1),
            &[42.0; BTC_DIRECTIONAL_FEATURE_COUNT],
        )
        .unwrap();
        assert_eq!(first_second.values[4], 42.0);
        assert_eq!(first_second.values[7], 42.0);
        assert_ne!(first_second.values[12], 42.0);
        assert_close(first_second.values[2], -4.4709993428150065);
        assert_close(first_second.values[12], 4.572043703535189);
        assert_close(first_second.values[24], 0.4600074943792154);
        assert_eq!(first_second.values[35], 1.0);
        assert_eq!(first_second.values[36], 0.0);

        let thirtieth_second = build_directional_features_for_asymmetric_value(
            &window,
            window_start,
            window_start + Duration::seconds(30),
            &[42.0; BTC_DIRECTIONAL_FEATURE_COUNT],
        )
        .unwrap();
        assert_eq!(thirtieth_second.values[7], 42.0);
        assert_eq!(thirtieth_second.values[16], 42.0);
        assert_eq!(thirtieth_second.values[41], 42.0);
        assert_close(thirtieth_second.values[6], 0.8999595024228313);
        assert_close(thirtieth_second.values[11], 2.013414078535966);
        assert_close(thirtieth_second.values[15], 0.03126886959448458);
        assert_close(thirtieth_second.values[33], 0.027202447687763365);
        assert_close(thirtieth_second.values[50], 0.9999999995033312);
        let asymmetric_at_60 = build_directional_features_for_asymmetric_value(
            &window,
            window_start,
            window_start + Duration::seconds(60),
            &[42.0; BTC_DIRECTIONAL_FEATURE_COUNT],
        )
        .unwrap();
        let legacy_at_60 =
            build_directional_features(&window, window_start, window_start + Duration::seconds(60))
                .unwrap();
        assert_eq!(asymmetric_at_60, legacy_at_60);
        assert!(matches!(
            build_directional_features_for_asymmetric_value(
                &window,
                window_start,
                window_start + Duration::seconds(61),
                &[42.0; BTC_DIRECTIONAL_FEATURE_COUNT],
            ),
            Err(DirectionalFeatureError::InvalidTiming {
                reason: DirectionalFeatureTimingReason::OffCadence,
                ..
            })
        ));
        assert!(matches!(
            build_directional_features(&window, window_start, window_start + Duration::seconds(55),),
            Err(DirectionalFeatureError::InvalidTiming {
                reason: DirectionalFeatureTimingReason::BeforeFirstCandidate,
                ..
            })
        ));
    }

    #[test]
    fn missing_gapped_and_incomplete_history_fail_closed() {
        let window_start = Utc.with_ymd_and_hms(2026, 6, 14, 12, 35, 0).unwrap();
        let as_of = window_start + Duration::seconds(60);

        let missing = BinanceOneSecondWindow::from_completed(
            (1..=60)
                .map(|second| fixture_candle(window_start, second))
                .collect(),
        )
        .unwrap();
        assert!(matches!(
            build_directional_features(&missing, window_start, as_of),
            Err(DirectionalFeatureError::MissingHistory { .. })
        ));

        let mut gapped = (0..=60)
            .map(|second| fixture_candle(window_start, second))
            .collect::<Vec<_>>();
        // `from_completed` correctly rejects this gap before feature derivation.
        gapped.remove(30);
        assert!(BinanceOneSecondWindow::from_completed(gapped).is_err());

        let mut incomplete_candles = (0..=60)
            .map(|second| fixture_candle(window_start, second))
            .collect::<Vec<_>>();
        incomplete_candles[30].source_complete = false;
        let incomplete = BinanceOneSecondWindow::from_completed(incomplete_candles).unwrap();
        assert!(matches!(
            build_directional_features(&incomplete, window_start, as_of),
            Err(DirectionalFeatureError::IncompleteCandle {
                synthetic: false,
                ..
            })
        ));

        let mut synthetic_candles = (0..=60)
            .map(|second| fixture_candle(window_start, second))
            .collect::<Vec<_>>();
        let carry_price = synthetic_candles[29].close_price;
        let synthetic = &mut synthetic_candles[30];
        synthetic.open_price = carry_price;
        synthetic.high_price = carry_price;
        synthetic.low_price = carry_price;
        synthetic.close_price = carry_price;
        synthetic.base_volume = Decimal::ZERO;
        synthetic.quote_volume = Decimal::ZERO;
        synthetic.trade_count = 0;
        synthetic.taker_buy_base_volume = Decimal::ZERO;
        synthetic.taker_buy_quote_volume = Decimal::ZERO;
        synthetic.source_complete = true;
        synthetic.synthetic = true;
        let synthetic_window = BinanceOneSecondWindow::from_completed(synthetic_candles).unwrap();
        assert!(build_directional_features(&synthetic_window, window_start, as_of).is_ok());
    }

    fn assert_timing_error(
        window_start: DateTime<Utc>,
        second: i64,
        reason: DirectionalFeatureTimingReason,
    ) {
        let feature_as_of = window_start + Duration::seconds(second);
        assert_eq!(
            validate_feature_time(
                window_start,
                feature_as_of,
                DirectionalFeatureTimingPolicy::LegacyDirectional,
            ),
            Err(DirectionalFeatureError::InvalidTiming {
                feature_as_of,
                seconds_elapsed: second,
                reason,
            })
        );
    }

    fn fixture_candle(window_start: DateTime<Utc>, second: i64) -> BinanceOneSecondKline {
        let close_milli = if second == 0 {
            100_000_000
        } else {
            100_000_000
                + second * 300
                + ((second % 20) - 10) * 5_000
                + if second % 11 == 0 { 1_200 } else { 0 }
        };
        let open_milli = close_milli + ((second % 3) - 1) * 200;
        let high_milli = open_milli.max(close_milli) + 500 + (second % 5) * 100;
        let low_milli = open_milli.min(close_milli) - 400 - (second % 4) * 100;
        let quote_cents = 100_000_000 + second * 100_000 + (second % 13) * 50_000;
        let taker_buy_quote_cents = quote_cents * (45 + (second % 5) * 2) / 100;
        let close_price = Decimal::new(close_milli, 3);
        let quote_volume = Decimal::new(quote_cents, 2);
        let taker_buy_quote_volume = Decimal::new(taker_buy_quote_cents, 2);
        let open_timestamp = window_start - Duration::seconds(1) + Duration::seconds(second);
        let trade_id = second as u64 + 1;

        BinanceOneSecondKline {
            open_timestamp,
            close_timestamp: open_timestamp + Duration::seconds(1),
            open_price: Decimal::new(open_milli, 3),
            high_price: Decimal::new(high_milli, 3),
            low_price: Decimal::new(low_milli, 3),
            close_price,
            base_volume: quote_volume / close_price,
            quote_volume,
            trade_count: (100 + second % 17) as u64,
            taker_buy_base_volume: taker_buy_quote_volume / close_price,
            taker_buy_quote_volume,
            first_aggregate_trade_id: trade_id,
            last_aggregate_trade_id: trade_id,
            first_source_timestamp: open_timestamp + Duration::milliseconds(100),
            last_source_timestamp: open_timestamp + Duration::milliseconds(900),
            max_received_at: open_timestamp + Duration::milliseconds(950),
            source_complete: true,
            synthetic: false,
        }
    }

    fn external_fixture(
        window_start: DateTime<Utc>,
        feature_as_of: DateTime<Utc>,
    ) -> (
        Vec<DirectionalOracleRound>,
        Vec<DirectionalChainlinkRefPrice>,
        Vec<DirectionalChainlinkCandle>,
        Vec<DirectionalBinanceOpenInterest>,
    ) {
        let oracle_start = window_start - Duration::seconds(120);
        let oracle = (0..=37)
            .map(|index| {
                let block_timestamp = oracle_start + Duration::seconds(index * 10);
                DirectionalOracleRound {
                    phase_id: 1,
                    aggregator_round_id: index + 1,
                    source_timestamp: block_timestamp - Duration::seconds(1),
                    block_timestamp,
                    block_number: Some(1_000 + index),
                    log_index: Some(0),
                    price: Decimal::new(99_800_000 + index * 1_000, 3),
                    available_at: block_timestamp,
                }
            })
            .collect();
        let refprice_start = feature_as_of - Duration::seconds(70);
        let refprice = (0..=71)
            .map(|index| {
                let source_timestamp = refprice_start + Duration::seconds(index);
                let price = Decimal::new(100_000_000 + index * 1_000, 3);
                DirectionalChainlinkRefPrice {
                    source_timestamp,
                    valid_from_timestamp: source_timestamp - Duration::milliseconds(100),
                    price,
                    bid: price - Decimal::new(500, 3),
                    ask: price + Decimal::new(500, 3),
                    available_at: source_timestamp,
                }
            })
            .collect();
        let candle_start = feature_as_of - Duration::minutes(70);
        let candles = (0..=71)
            .map(|index| {
                let close_timestamp = candle_start + Duration::minutes(index);
                let close_price = Decimal::new(99_000_000 + index * 10_000, 3);
                DirectionalChainlinkCandle {
                    open_timestamp: close_timestamp - Duration::minutes(1),
                    close_timestamp,
                    open_price: close_price,
                    high_price: close_price + Decimal::ONE,
                    low_price: close_price - Decimal::ONE,
                    close_price,
                    available_at: close_timestamp,
                }
            })
            .collect();
        let open_interest_start = feature_as_of - Duration::minutes(70);
        let open_interest = (0..=14)
            .map(|index| {
                let source_timestamp = open_interest_start + Duration::minutes(index * 5);
                DirectionalBinanceOpenInterest {
                    source_timestamp,
                    period_seconds: 300,
                    sum_open_interest: Decimal::new(1_000_000 + index * 1_000, 0),
                    sum_open_interest_value: Decimal::new(10_000_000 + index * 10_000, 0),
                    available_at: source_timestamp,
                }
            })
            .collect();
        (oracle, refprice, candles, open_interest)
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 2e-10_f64.max(expected.abs() * 2e-11);
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17}, expected={expected:.17}, tolerance={tolerance:.3e}"
        );
    }

    fn prewindow_fixture_candle(window_start: DateTime<Utc>, index: i64) -> BinanceOneSecondKline {
        let close_milli = 100_000_000
            + index * 50
            + ((index % 37) - 18) * 2_000
            + if index % 29 == 0 { 800 } else { 0 };
        let open_milli = close_milli + ((index % 3) - 1) * 200;
        let high_milli = open_milli.max(close_milli) + 500 + (index % 5) * 100;
        let low_milli = open_milli.min(close_milli) - 400 - (index % 4) * 100;
        let quote_cents = 100_000_000 + index * 10_000 + (index % 13) * 50_000;
        let taker_buy_quote_cents = quote_cents * (45 + (index % 5) * 2) / 100;
        let close_price = Decimal::new(close_milli, 3);
        let quote_volume = Decimal::new(quote_cents, 2);
        let taker_buy_quote_volume = Decimal::new(taker_buy_quote_cents, 2);
        let open_timestamp = window_start - Duration::seconds(3_601) + Duration::seconds(index);
        let trade_id = index as u64 + 1;

        BinanceOneSecondKline {
            open_timestamp,
            close_timestamp: open_timestamp + Duration::seconds(1),
            open_price: Decimal::new(open_milli, 3),
            high_price: Decimal::new(high_milli, 3),
            low_price: Decimal::new(low_milli, 3),
            close_price,
            base_volume: quote_volume / close_price,
            quote_volume,
            trade_count: (100 + index % 17) as u64,
            taker_buy_base_volume: taker_buy_quote_volume / close_price,
            taker_buy_quote_volume,
            first_aggregate_trade_id: trade_id,
            last_aggregate_trade_id: trade_id,
            first_source_timestamp: open_timestamp + Duration::milliseconds(100),
            last_source_timestamp: open_timestamp + Duration::milliseconds(900),
            max_received_at: open_timestamp + Duration::milliseconds(950),
            source_complete: true,
            synthetic: false,
        }
    }
}
