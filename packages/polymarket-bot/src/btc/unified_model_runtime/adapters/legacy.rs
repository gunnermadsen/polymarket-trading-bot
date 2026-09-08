//! Describes the existing schema-driven adapter without modifying its mathematics.
use super::super::contract::{InputContract, ModelContract, CONTRACT_VERSION};
use crate::btc::{
    directional_features::directional_external_feature_requirements,
    directional_model::RuntimeDirectionalModel,
};
pub fn contract(model: &RuntimeDirectionalModel) -> ModelContract {
    let requirements = directional_external_feature_requirements(model.feature_schema_version());
    let input = |slot: &str, product: &str, lookback_seconds, maximum_age_ms| InputContract {
        slot: slot.into(),
        product: product.into(),
        semantics: format!("legacy:{}", model.feature_schema_version()),
        required: true,
        lookback_seconds,
        maximum_age_ms,
    };
    let mut inputs = vec![
        input(
            "btc_seconds",
            "binance_spot_btcusdt_one_second_ohlcv",
            3900,
            5000,
        ),
        input(
            "execution_book",
            "polymarket_btc_five_minute_orderbooks",
            2,
            2000,
        ),
    ];
    if requirements.oracle {
        inputs.push(input(
            "oracle",
            "polygon_chainlink_btcusd_oracle",
            600,
            600000,
        ));
    }
    if requirements.chainlink_candles {
        inputs.push(input(
            "candles",
            "polymarket_rtds_chainlink_reference_price",
            3660,
            60000,
        ));
    }
    if requirements.refprice {
        inputs.push(input(
            "refprice",
            "chainlink_btcusd_reference_prices",
            300,
            60000,
        ));
    }
    if requirements.open_interest {
        inputs.push(input(
            "open_interest",
            "binance_futures_btcusdt_open_interest",
            3900,
            360000,
        ));
    }
    ModelContract {
        version: CONTRACT_VERSION.into(),
        adapter: "legacy_directional".into(),
        adapter_version: 1,
        inputs,
        probability_semantics: "probability_up".into(),
        feature_clock: "closed_binance_second_as_of".into(),
        missing_policy: "legacy_preserved".into(),
        qualified_trade_size: None,
    }
}
