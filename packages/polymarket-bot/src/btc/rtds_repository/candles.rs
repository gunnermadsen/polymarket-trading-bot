//! RTDS-derived OHLC; calculations and causal bounds preserved from the runtime consumer.
use crate::btc::directional_features::{DirectionalChainlinkCandle, DirectionalFeatureError};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

#[derive(Debug, Clone, Copy)]
struct ChainlinkCandleAccumulator {
    first_timestamp: DateTime<Utc>,
    last_timestamp: DateTime<Utc>,
    open: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    available_at: DateTime<Utc>,
}

impl ChainlinkCandleAccumulator {
    fn new(source_timestamp: DateTime<Utc>, available_at: DateTime<Utc>, price: Decimal) -> Self {
        Self {
            first_timestamp: source_timestamp,
            last_timestamp: source_timestamp,
            open: price,
            high: price,
            low: price,
            close: price,
            available_at,
        }
    }

    fn observe(
        &mut self,
        source_timestamp: DateTime<Utc>,
        available_at: DateTime<Utc>,
        price: Decimal,
    ) {
        if source_timestamp < self.first_timestamp {
            self.first_timestamp = source_timestamp;
            self.open = price;
        }
        if source_timestamp >= self.last_timestamp {
            self.last_timestamp = source_timestamp;
            self.close = price;
        }
        self.high = self.high.max(price);
        self.low = self.low.min(price);
        self.available_at = self.available_at.max(available_at);
    }
}

pub(super) fn closed_candles<'a>(
    points: impl Iterator<Item = &'a super::RtdsPoint>,
    feature_as_of: DateTime<Utc>,
) -> Result<Vec<DirectionalChainlinkCandle>, DirectionalFeatureError> {
    const REQUIRED_CANDLES: usize = 61;
    let latest_close = DateTime::from_timestamp(feature_as_of.timestamp().div_euclid(60) * 60, 0)
        .ok_or_else(|| {
        external_snapshot_error(
            "chainlink_candles",
            "decision timestamp could not be minute-aligned",
        )
    })?;
    let earliest_open = latest_close - chrono::Duration::minutes(REQUIRED_CANDLES as i64);
    let mut accumulators: Vec<Option<ChainlinkCandleAccumulator>> = vec![None; REQUIRED_CANDLES];

    for point in points {
        if point.available_at > feature_as_of
            || point.source_timestamp < earliest_open
            || point.source_timestamp >= latest_close
        {
            continue;
        }
        if point.price <= Decimal::ZERO {
            return Err(external_snapshot_error(
                "chainlink_candles",
                "runtime midpoint history contained an invalid price",
            ));
        }
        let bucket = (point.source_timestamp - earliest_open).num_seconds() / 60;
        let index = usize::try_from(bucket).map_err(|_| {
            external_snapshot_error(
                "chainlink_candles",
                "runtime midpoint fell outside the required candle window",
            )
        })?;
        let accumulator = accumulators.get_mut(index).ok_or_else(|| {
            external_snapshot_error(
                "chainlink_candles",
                "runtime midpoint fell outside the required candle window",
            )
        })?;
        match accumulator {
            Some(accumulator) => {
                accumulator.observe(point.source_timestamp, point.available_at, point.price)
            }
            slot @ None => {
                *slot = Some(ChainlinkCandleAccumulator::new(
                    point.source_timestamp,
                    point.available_at,
                    point.price,
                ));
            }
        }
    }

    accumulators
        .into_iter()
        .enumerate()
        .map(|(index, accumulator)| {
            let accumulator = accumulator.ok_or_else(|| {
                external_snapshot_error(
                    "chainlink_candles",
                    "61 contiguous closed RTDS midpoint candles are unavailable at the decision time",
                )
            })?;
            let open_timestamp = earliest_open
                + chrono::Duration::minutes(i64::try_from(index).expect("61 candles fit i64"));
            Ok(DirectionalChainlinkCandle {
                open_timestamp,
                close_timestamp: open_timestamp + chrono::Duration::minutes(1),
                open_price: external_decimal_value(
                    accumulator.open,
                    "chainlink_candles",
                    "derived open price was invalid",
                )?,
                high_price: external_decimal_value(
                    accumulator.high,
                    "chainlink_candles",
                    "derived high price was invalid",
                )?,
                low_price: external_decimal_value(
                    accumulator.low,
                    "chainlink_candles",
                    "derived low price was invalid",
                )?,
                close_price: external_decimal_value(
                    accumulator.close,
                    "chainlink_candles",
                    "derived close price was invalid",
                )?,
                available_at: accumulator.available_at,
            })
        })
        .collect()
}

fn external_decimal_value(
    value: Decimal,
    source: &'static str,
    reason: &'static str,
) -> Result<Decimal, DirectionalFeatureError> {
    if value <= Decimal::ZERO {
        return Err(external_snapshot_error(source, reason));
    }
    Ok(value)
}

fn external_snapshot_error(source: &'static str, reason: &'static str) -> DirectionalFeatureError {
    DirectionalFeatureError::ExternalFeatureUnavailable { source, reason }
}
