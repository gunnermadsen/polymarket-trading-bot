use rust_decimal::Decimal;

use crate::orderbook::FillQuote;

pub fn compute_taker_fee(fills: &[FillQuote], fee_rate: Decimal) -> Decimal {
    fills
        .iter()
        .map(|fill| fill.size * fee_rate * fill.price * (Decimal::ONE - fill.price))
        .sum()
}

#[cfg(test)]
mod tests {
    use rust_decimal_macros::dec;

    use super::*;

    #[test]
    fn fee_is_computed_per_fill_level() {
        let fills = vec![
            FillQuote {
                price: dec!(0.40),
                size: dec!(10),
            },
            FillQuote {
                price: dec!(0.60),
                size: dec!(5),
            },
        ];
        assert_eq!(compute_taker_fee(&fills, dec!(0.02)), dec!(0.0720));
    }
}
