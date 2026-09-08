//! Frozen early-entry feature binding. Unsupported products remain explicitly absent;
//! RTDS midpoint candles are never substituted for the historical OHLC product.
use crate::btc::{
    directional_features::{
        build_payoff_feature_values_with_policy, DirectionalExternalFeatureInputs,
        DirectionalOracleRound,
    },
    types::{OrderbookCheckpoint, RealtimeState},
};
use anyhow::{ensure, Context, Result};
use chrono::{DateTime, Utc};
use rust_decimal::prelude::ToPrimitive;

#[allow(clippy::too_many_arguments)]
pub fn build(
    state: &RealtimeState,
    names: &[String],
    binding: &super::super::contract::ProcessBinding,
    window_start: DateTime<Utc>,
    at: DateTime<Utc>,
    up: &OrderbookCheckpoint,
    down: &OrderbookCheckpoint,
    fee_rate: f64,
) -> Result<Vec<f64>> {
    let anchor = state
        .binance_one_second_window
        .completed()
        .iter()
        .find(|c| c.open_timestamp == window_start - chrono::Duration::seconds(1))
        .context("UMR pre-window Binance opening boundary unavailable")?;
    let up = state
        .unified_book_history
        .at(&up.market_id, &up.token_id, up.connection_id, at)
        .with_context(|| {
            format!(
                "UMR causal UP book unavailable: expected_epoch={}; {}",
                up.connection_id,
                state
                    .unified_book_history
                    .availability_detail(&up.token_id, at)
            )
        })?;
    let down = state
        .unified_book_history
        .at(&down.market_id, &down.token_id, down.connection_id, at)
        .with_context(|| {
            format!(
                "UMR causal DOWN book unavailable: expected_epoch={}; {}",
                down.connection_id,
                state
                    .unified_book_history
                    .availability_detail(&down.token_id, at)
            )
        })?;
    for book in [up, down] {
        let age = (at - book.received_at).num_milliseconds();
        ensure!(
            (0..=2000).contains(&age),
            "UMR causal book is outside frozen two-second eligibility"
        );
    }
    let rounds = state
        .directional_external
        .oracle
        .iter()
        .filter(|v| v.available_at <= at && binding.sources.iter().any(|s| s.slot == "oracle"))
        .map(|v| DirectionalOracleRound {
            phase_id: i32::from(v.phase_id),
            aggregator_round_id: v.round_id as i64,
            source_timestamp: v.source_timestamp,
            block_timestamp: v.block_timestamp,
            block_number: None,
            log_index: None,
            price: v.price,
            available_at: v.available_at,
        })
        .collect::<Vec<_>>();
    let external = DirectionalExternalFeatureInputs {
        oracle_rounds: &rounds,
        ..Default::default()
    };
    let values = build_payoff_feature_values_with_policy(
        &state.binance_one_second_window,
        window_start,
        at,
        anchor.open_price,
        &external,
        up,
        down,
        fee_rate,
        names,
        true,
    )?;
    ensure!(
        anchor.open_price.to_f64().is_some_and(|v| v > 0.0),
        "UMR opening boundary is invalid"
    );
    Ok(values)
}
