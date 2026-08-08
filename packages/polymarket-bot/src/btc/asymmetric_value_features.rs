use std::collections::HashSet;

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use rust_decimal::prelude::ToPrimitive;

use super::{
    binance_spot_l2::BinanceL2OneSecondFeature,
    directional_features::build_directional_features_for_asymmetric_value,
    directional_model::{
        asymmetric_value_model_input_sha256, BtcDirectionalModelFeatureSnapshot,
        RuntimeDirectionalModel, RuntimeModelSelection, ASYMMETRIC_L2_FEATURE_NAMES,
    },
    strategy::BtcFeatureSnapshot,
    types::{BtcIntervalMarket, RealtimeState},
};

const EXECUTION_RESERVE_PER_SHARE: f64 = 0.01;
const ORACLE_PROPAGATION_DELAY_SECONDS: i64 = 2;
const ORACLE_MAXIMUM_AGE_SECONDS: i64 = 300;

pub fn build_asymmetric_value_feature_snapshot(
    selection: &RuntimeModelSelection,
    model: &RuntimeDirectionalModel,
    state: &RealtimeState,
    market: &BtcIntervalMarket,
    feature_as_of: DateTime<Utc>,
    snapshot: &BtcFeatureSnapshot,
) -> Result<BtcDirectionalModelFeatureSnapshot> {
    if !model.is_asymmetric_value() {
        bail!("asymmetric feature construction requires an asymmetric model");
    }
    let core = build_directional_features_for_asymmetric_value(
        &state.binance_one_second_window,
        market.window_start,
        feature_as_of,
        model.imputation_medians(),
    )?;
    let mut values = Vec::with_capacity(model.feature_names().len());
    values.extend(core.values);

    if model
        .feature_names()
        .iter()
        .any(|name| name == "oracle_return_from_window_open_bps")
    {
        values.extend(oracle_features(state, market, feature_as_of)?);
    } else if model
        .feature_names()
        .iter()
        .any(|name| name == ASYMMETRIC_L2_FEATURE_NAMES[0])
    {
        let l2 = state
            .binance_spot_l2
            .latest_eligible(feature_as_of)
            .context("no strictly prior qualified Binance spot L2 feature was available")?;
        let btc_close = state
            .binance_one_second_window
            .completed()
            .iter()
            .rev()
            .find(|candle| candle.close_timestamp == feature_as_of)
            .map(|candle| decimal(candle.close_price, "Binance one-second close"))
            .transpose()?
            .context("no Binance one-second close matched the asymmetric decision")?;
        values.extend(l2_features(l2, btc_close)?);
    }
    let (polymarket, yes_ask_vwap, no_ask_vwap) = polymarket_features(snapshot)?;
    values.extend(polymarket);
    if values.len() != model.feature_names().len() || values.iter().any(|value| !value.is_finite())
    {
        bail!("asymmetric runtime feature vector did not match its frozen schema");
    }
    let seconds_elapsed = (feature_as_of - market.window_start).num_seconds();
    let input_sha256 = asymmetric_value_model_input_sha256(
        selection,
        model.feature_schema_version(),
        &market.market_id,
        market.window_start,
        feature_as_of,
        seconds_elapsed,
        &values,
        yes_ask_vwap,
        no_ask_vwap,
    )?;
    Ok(BtcDirectionalModelFeatureSnapshot {
        model_key: selection.model_key.clone(),
        model_artifact_sha256: selection.artifact_sha256.clone(),
        feature_schema_version: model.feature_schema_version().to_string(),
        feature_schema_sha256: selection.feature_schema_sha256.clone(),
        feature_as_of,
        seconds_elapsed,
        feature_values: values,
        input_sha256,
    })
}

fn oracle_features(
    state: &RealtimeState,
    market: &BtcIntervalMarket,
    feature_as_of: DateTime<Utc>,
) -> Result<[f64; 4]> {
    let decision_cutoff = feature_as_of - Duration::seconds(ORACLE_PROPAGATION_DELAY_SECONDS);
    let open_cutoff = market.window_start - Duration::seconds(ORACLE_PROPAGATION_DELAY_SECONDS);
    let eligible = |cutoff: DateTime<Utc>| {
        state
            .directional_external
            .oracle
            .iter()
            .filter(|round| {
                round.source_timestamp <= round.block_timestamp
                    && round.block_timestamp <= cutoff
                    && cutoff - round.block_timestamp
                        <= Duration::seconds(ORACLE_MAXIMUM_AGE_SECONDS)
            })
            .max_by_key(|round| (round.block_timestamp, round.phase_id, round.round_id))
    };
    let opening = eligible(open_cutoff).context("no causal oracle round covered window open")?;
    let current = eligible(decision_cutoff).context("no causal oracle round covered decision")?;
    let opening_price = decimal(opening.price, "opening oracle price")?;
    let current_price = decimal(current.price, "current oracle price")?;
    let btc_close = state
        .binance_one_second_window
        .completed()
        .iter()
        .rev()
        .find(|candle| candle.close_timestamp == feature_as_of)
        .map(|candle| decimal(candle.close_price, "Binance one-second close"))
        .transpose()?
        .context("no Binance one-second close matched oracle decision")?;
    let distinct_updates = state
        .directional_external
        .oracle
        .iter()
        .filter(|round| {
            round.source_timestamp <= round.block_timestamp
                && round.block_timestamp > opening.block_timestamp
                && round.block_timestamp <= decision_cutoff
        })
        .map(|round| (round.phase_id, round.round_id))
        .collect::<HashSet<_>>()
        .len();
    let seconds_elapsed = (feature_as_of - market.window_start).num_seconds();
    let age_seconds = (feature_as_of - current.block_timestamp).num_milliseconds() as f64 / 1_000.0;
    Ok([
        (current_price / opening_price).ln() * 10_000.0,
        age_seconds / 300.0,
        distinct_updates as f64 / (seconds_elapsed + 1) as f64,
        (btc_close / current_price).ln() * 10_000.0,
    ])
}

fn l2_features(feature: &BinanceL2OneSecondFeature, btc_close: f64) -> Result<[f64; 40]> {
    let midpoint = decimal(feature.midpoint, "L2 midpoint")?;
    let microprice = decimal(feature.microprice, "L2 microprice")?;
    let depth = |value, label| decimal(value, label);
    let values = [
        (midpoint / btc_close).ln() * 10_000.0,
        (microprice / midpoint).ln() * 10_000.0,
        decimal(feature.spread_bps, "L2 spread")?,
        depth(feature.bid_depth_5, "L2 bid depth 5")?.ln(),
        depth(feature.ask_depth_5, "L2 ask depth 5")?.ln(),
        decimal(feature.imbalance_5, "L2 imbalance 5")?,
        depth(feature.bid_depth_10, "L2 bid depth 10")?.ln(),
        depth(feature.ask_depth_10, "L2 ask depth 10")?.ln(),
        decimal(feature.imbalance_10, "L2 imbalance 10")?,
        depth(feature.bid_depth_20, "L2 bid depth 20")?.ln(),
        depth(feature.ask_depth_20, "L2 ask depth 20")?.ln(),
        decimal(feature.imbalance_20, "L2 imbalance 20")?,
        decimal(feature.bid_depth_slope_20, "L2 bid slope")?,
        decimal(feature.ask_depth_slope_20, "L2 ask slope")?,
        decimal(feature.bid_depth_concentration_20, "L2 bid concentration")?,
        decimal(feature.ask_depth_concentration_20, "L2 ask concentration")?,
        decimal(feature.bid_quote_replenishment_1s, "L2 bid replenishment")?.ln_1p(),
        decimal(feature.ask_quote_replenishment_1s, "L2 ask replenishment")?.ln_1p(),
        decimal(feature.bid_quote_churn_1s, "L2 bid churn")?.ln_1p(),
        decimal(feature.ask_quote_churn_1s, "L2 ask churn")?.ln_1p(),
        decimal(feature.midpoint_change_bps_1s, "L2 midpoint change 1s")?,
        decimal(feature.spread_bps_delta_1s, "L2 spread change 1s")?,
        decimal(feature.depth_20_change_bps_1s, "L2 depth change 1s")?,
        decimal(feature.imbalance_20_delta_1s, "L2 imbalance change 1s")?,
        decimal(feature.midpoint_change_bps_5s, "L2 midpoint change 5s")?,
        decimal(feature.spread_bps_delta_5s, "L2 spread change 5s")?,
        decimal(feature.depth_20_change_bps_5s, "L2 depth change 5s")?,
        decimal(feature.imbalance_20_delta_5s, "L2 imbalance change 5s")?,
        decimal(feature.midpoint_change_bps_15s, "L2 midpoint change 15s")?,
        decimal(feature.spread_bps_delta_15s, "L2 spread change 15s")?,
        decimal(feature.depth_20_change_bps_15s, "L2 depth change 15s")?,
        decimal(feature.imbalance_20_delta_15s, "L2 imbalance change 15s")?,
        decimal(feature.midpoint_change_bps_30s, "L2 midpoint change 30s")?,
        decimal(feature.spread_bps_delta_30s, "L2 spread change 30s")?,
        decimal(feature.depth_20_change_bps_30s, "L2 depth change 30s")?,
        decimal(feature.imbalance_20_delta_30s, "L2 imbalance change 30s")?,
        decimal(feature.midpoint_change_bps_60s, "L2 midpoint change 60s")?,
        decimal(feature.spread_bps_delta_60s, "L2 spread change 60s")?,
        decimal(feature.depth_20_change_bps_60s, "L2 depth change 60s")?,
        decimal(feature.imbalance_20_delta_60s, "L2 imbalance change 60s")?,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        bail!("Binance spot L2 transformation produced a non-finite value");
    }
    Ok(values)
}

fn polymarket_features(snapshot: &BtcFeatureSnapshot) -> Result<([f64; 13], f64, f64)> {
    let yes_vwap = decimal(
        snapshot
            .up_book
            .executable_ask_vwap
            .context("YES executable VWAP was unavailable")?,
        "YES executable VWAP",
    )?;
    let no_vwap = decimal(
        snapshot
            .down_book
            .executable_ask_vwap
            .context("NO executable VWAP was unavailable")?,
        "NO executable VWAP",
    )?;
    let yes_best = decimal(
        snapshot
            .up_book
            .best_ask
            .context("YES best ask was unavailable")?,
        "YES best ask",
    )?;
    let no_best = decimal(
        snapshot
            .down_book
            .best_ask
            .context("NO best ask was unavailable")?,
        "NO best ask",
    )?;
    let fee_rate = decimal(
        snapshot.fee_rate.context("fee rate was unavailable")?,
        "fee rate",
    )?;
    let yes_depth = decimal(snapshot.up_book.ask_depth, "YES ask depth")?;
    let no_depth = decimal(snapshot.down_book.ask_depth, "NO ask depth")?;
    let yes_age = snapshot
        .up_book
        .received_at
        .map(|received| (snapshot.observed_at - received).num_milliseconds() as f64 / 1_000.0)
        .context("YES book receipt time was unavailable")?;
    let no_age = snapshot
        .down_book
        .received_at
        .map(|received| (snapshot.observed_at - received).num_milliseconds() as f64 / 1_000.0)
        .context("NO book receipt time was unavailable")?;
    if !(0.0..=1.0).contains(&yes_vwap)
        || !(0.0..=1.0).contains(&no_vwap)
        || yes_depth <= 0.0
        || no_depth <= 0.0
        || yes_age < 0.0
        || no_age < 0.0
        || fee_rate < 0.0
    {
        bail!("Polymarket asymmetric feature evidence was invalid");
    }
    let yes_cost = yes_vwap + fee_rate * yes_vwap * (1.0 - yes_vwap) + EXECUTION_RESERVE_PER_SHARE;
    let no_cost = no_vwap + fee_rate * no_vwap * (1.0 - no_vwap) + EXECUTION_RESERVE_PER_SHARE;
    let logit = |value: f64| {
        let clipped = value.clamp(1e-6, 1.0 - 1e-6);
        (clipped / (1.0 - clipped)).ln()
    };
    Ok((
        [
            yes_cost,
            no_cost,
            logit(yes_cost),
            logit(no_cost),
            yes_cost + no_cost - 1.0,
            yes_cost - no_cost,
            yes_vwap - yes_best,
            no_vwap - no_best,
            yes_depth.ln_1p(),
            no_depth.ln_1p(),
            (yes_depth - no_depth) / (yes_depth + no_depth).max(1e-9),
            yes_age,
            no_age,
        ],
        yes_vwap,
        no_vwap,
    ))
}

fn decimal(value: rust_decimal::Decimal, label: &str) -> Result<f64> {
    let converted = value
        .to_f64()
        .with_context(|| format!("{label} could not be represented as f64"))?;
    if !converted.is_finite() {
        bail!("{label} was not a finite value");
    }
    Ok(converted)
}
