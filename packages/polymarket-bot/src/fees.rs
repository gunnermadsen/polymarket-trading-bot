use std::str::FromStr;

use anyhow::{bail, Context, Result};
use rust_decimal::Decimal;

pub const DYNAMIC_FEE_RATE_METADATA_KEY: &str = "dynamic_fee_rate";

/// Polymarket's dynamic crypto taker fee: `contracts * rate * price * (1 - price)`.
pub fn dynamic_crypto_taker_fee(contracts: Decimal, fee_rate: Decimal, price: Decimal) -> Decimal {
    if contracts <= Decimal::ZERO
        || fee_rate <= Decimal::ZERO
        || price <= Decimal::ZERO
        || price >= Decimal::ONE
    {
        return Decimal::ZERO;
    }
    contracts * fee_rate * price * (Decimal::ONE - price)
}

/// Per-contract form used by floating-point feature calculations. Execution and persisted
/// accounting continue to use [`dynamic_crypto_taker_fee`] so financial values remain decimal.
pub fn dynamic_crypto_taker_fee_per_contract_f64(fee_rate: f64, price: f64) -> f64 {
    if !fee_rate.is_finite()
        || !price.is_finite()
        || fee_rate <= 0.0
        || price <= 0.0
        || price >= 1.0
    {
        return 0.0;
    }
    fee_rate * price * (1.0 - price)
}

/// Reads the immutable fee rate sealed into an order request. Legacy aliases remain accepted so
/// existing paper and live orders retain their established metadata contract.
pub fn dynamic_fee_rate_from_metadata(metadata: &serde_json::Value) -> Option<Decimal> {
    [
        metadata.get(DYNAMIC_FEE_RATE_METADATA_KEY),
        metadata.get("taker_fee_rate"),
        metadata
            .get("execution")
            .and_then(|execution| execution.get(DYNAMIC_FEE_RATE_METADATA_KEY)),
        metadata
            .get("execution")
            .and_then(|execution| execution.get("taker_fee_rate")),
    ]
    .into_iter()
    .flatten()
    .find_map(decimal_from_json)
}

pub fn sealed_dynamic_fee_rate(metadata: &serde_json::Value) -> Result<Decimal> {
    let fee_rate = dynamic_fee_rate_from_metadata(metadata)
        .context("order is missing sealed dynamic_fee_rate evidence")?;
    if fee_rate < Decimal::ZERO || fee_rate > Decimal::ONE {
        bail!("sealed dynamic_fee_rate must be between zero and one");
    }
    Ok(fee_rate)
}

pub(crate) fn decimal_from_json(value: &serde_json::Value) -> Option<Decimal> {
    value
        .as_str()
        .and_then(|text| Decimal::from_str(text).ok())
        .or_else(|| Decimal::from_str(&value.to_string()).ok())
}

#[cfg(test)]
mod tests {
    use rust_decimal_macros::dec;
    use serde_json::json;

    use super::*;

    #[test]
    fn dynamic_crypto_fee_matches_polymarket_formula() {
        assert_eq!(
            dynamic_crypto_taker_fee(dec!(10), dec!(0.25), dec!(0.40)),
            dec!(0.60)
        );
        assert_eq!(
            dynamic_crypto_taker_fee(dec!(2), dec!(0.25), dec!(0.49)),
            dec!(0.124950)
        );
    }

    #[test]
    fn invalid_economics_are_zero_and_do_not_overflow_callers() {
        assert_eq!(
            dynamic_crypto_taker_fee(Decimal::ZERO, dec!(0.25), dec!(0.40)),
            Decimal::ZERO
        );
        assert_eq!(
            dynamic_crypto_taker_fee(dec!(10), dec!(0.25), Decimal::ONE),
            Decimal::ZERO
        );
        assert_eq!(dynamic_crypto_taker_fee_per_contract_f64(0.25, 0.4), 0.06);
        assert_eq!(dynamic_crypto_taker_fee_per_contract_f64(0.25, 1.0), 0.0);
    }

    #[test]
    fn sealed_rate_parser_preserves_existing_metadata_shapes() {
        assert_eq!(
            sealed_dynamic_fee_rate(&json!({"dynamic_fee_rate": "0.25"})).unwrap(),
            dec!(0.25)
        );
        assert_eq!(
            sealed_dynamic_fee_rate(&json!({"execution": {"taker_fee_rate": 0.25}})).unwrap(),
            dec!(0.25)
        );
        assert!(sealed_dynamic_fee_rate(&json!({})).is_err());
        assert!(sealed_dynamic_fee_rate(&json!({"dynamic_fee_rate": "1.01"})).is_err());
    }
}
