use anyhow::{bail, Result};
use rust_decimal::Decimal;
use uuid::Uuid;

use crate::models::ConversionResult;

#[derive(Debug, Clone)]
pub struct WalletClient {
    live_enabled: bool,
}

impl WalletClient {
    pub fn new(live_enabled: bool) -> Self {
        Self { live_enabled }
    }

    pub async fn convert_negative_risk(
        &self,
        _market_id: &str,
        _no_token_id: &str,
        _size: Decimal,
    ) -> Result<ConversionResult> {
        if !self.live_enabled {
            return Ok(ConversionResult {
                conversion_id: Uuid::new_v4(),
                status: "sim_wallet_noop".to_string(),
                tx_hash: None,
                latency_ms: 0,
                gas_cost_usd: Decimal::ZERO,
            });
        }
        bail!("live wallet conversion is scaffolded but not implemented in v1")
    }
}
