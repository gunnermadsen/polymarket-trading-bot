use std::fmt;

use chrono::{DateTime, Datelike, Timelike, Utc};
use rust_decimal::prelude::ToPrimitive;

use super::types::{BinanceOneSecondKline, BinanceOneSecondWindow};

pub const BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION: &str = "btc-5m-directional-core-features-v2";
pub const BTC_DIRECTIONAL_FEATURE_COUNT: usize = 58;
pub const BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION: &str =
    "btc-5m-directional-mature-reversal-features-v1";
pub const BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_COUNT: usize = 71;
pub const BTC_DIRECTIONAL_FIRST_CANDIDATE_SECOND: i64 = 60;
pub const BTC_DIRECTIONAL_LAST_CANDIDATE_SECOND: i64 = 240;
pub const BTC_DIRECTIONAL_CANDIDATE_CADENCE_SECONDS: i64 = 5;

const MARKET_WINDOW_SECONDS: i64 = 300;
const EPSILON: f64 = 1e-9;
const BPS: f64 = 10_000.0;

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

pub fn directional_feature_names(schema_version: &str) -> Option<&'static [&'static str]> {
    match schema_version {
        BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION => Some(&BTC_DIRECTIONAL_FEATURE_NAMES),
        BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION => {
            Some(&BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_NAMES)
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
    let schema_version = canonical_feature_schema_version(schema_version).ok_or_else(|| {
        DirectionalFeatureError::UnsupportedFeatureSchema {
            schema_version: schema_version.to_string(),
        }
    })?;
    let seconds_elapsed = validate_feature_time(window_start, feature_as_of)?;
    let required_start = window_start - chrono::Duration::seconds(1);
    let required_end = feature_as_of - chrono::Duration::seconds(1);
    let completed = collect_required_candles(window, required_start, required_end)?;
    let numeric = completed
        .into_iter()
        .map(NumericCandle::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    derive_directional_features(&numeric, feature_as_of, seconds_elapsed, schema_version)
}

fn canonical_feature_schema_version(schema_version: &str) -> Option<&'static str> {
    match schema_version {
        BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION => Some(BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION),
        BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION => {
            Some(BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION)
        }
        _ => None,
    }
}

fn validate_feature_time(
    window_start: DateTime<Utc>,
    feature_as_of: DateTime<Utc>,
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
    if seconds_elapsed < BTC_DIRECTIONAL_FIRST_CANDIDATE_SECOND {
        return Err(DirectionalFeatureError::InvalidTiming {
            feature_as_of,
            seconds_elapsed,
            reason: DirectionalFeatureTimingReason::BeforeFirstCandidate,
        });
    }
    if seconds_elapsed > BTC_DIRECTIONAL_LAST_CANDIDATE_SECOND {
        return Err(DirectionalFeatureError::InvalidTiming {
            feature_as_of,
            seconds_elapsed,
            reason: DirectionalFeatureTimingReason::AfterLastCandidate,
        });
    }
    if (seconds_elapsed - BTC_DIRECTIONAL_FIRST_CANDIDATE_SECOND)
        % BTC_DIRECTIONAL_CANDIDATE_CADENCE_SECONDS
        != 0
    {
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

fn derive_directional_features(
    candles: &[NumericCandle],
    feature_as_of: DateTime<Utc>,
    seconds_elapsed: i64,
    schema_version: &'static str,
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

fn horizon_return(log_closes: &[f64], end: usize, seconds: usize) -> f64 {
    (log_closes[end] - log_closes[end - seconds]) * BPS
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

fn rolling_sum(
    candles: &[NumericCandle],
    end: usize,
    window: usize,
    value: impl Fn(&NumericCandle) -> f64,
) -> f64 {
    candles[end + 1 - window..=end].iter().map(value).sum()
}

fn rolling_high_low(candles: &[NumericCandle], end: usize, window: usize) -> (f64, f64) {
    candles[end + 1 - window..=end]
        .iter()
        .fold((f64::NEG_INFINITY, f64::INFINITY), |(high, low), candle| {
            (high.max(candle.high), low.min(candle.low))
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
    let window_open = log_closes[0];
    let mut previous_positive = true;
    let mut positive_count = 0;
    let mut cross_count = 0;
    let mut last_cross = None;

    for (second, close) in log_closes.iter().enumerate() {
        let positive = close - window_open >= 0.0;
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
                validate_feature_time(window_start, window_start + Duration::seconds(second)),
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
                window_start + Duration::seconds(60) + Duration::milliseconds(1)
            ),
            Err(DirectionalFeatureError::InvalidTiming {
                feature_as_of: window_start + Duration::seconds(60) + Duration::milliseconds(1),
                seconds_elapsed: 60,
                reason: DirectionalFeatureTimingReason::NotWholeSecond,
            })
        );
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
            validate_feature_time(window_start, feature_as_of),
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
}
