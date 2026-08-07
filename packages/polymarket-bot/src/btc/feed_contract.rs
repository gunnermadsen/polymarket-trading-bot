//! Versioned, read-only model-feed requirements.
//!
//! Feed infrastructure is global runtime state. Trading processes may declare
//! requirements, but never own sockets, endpoints, or credentials.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

pub const BTC_MODEL_FEED_CONTRACT_VERSION: &str = "btc-model-feed-contract-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BtcModelFeedId {
    BinanceBtcusdtOneSecondV1,
    BinanceBtcusdtL2V1,
    ChainlinkBtcusdOracleV1,
    PolymarketBtc5mClobExecutionV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BtcModelFeedRequirement {
    pub feed: BtcModelFeedId,
    pub maximum_age_ms: i64,
    #[serde(default)]
    pub require_sequence_integrity: bool,
}

impl BtcModelFeedRequirement {
    pub fn validate(&self) -> Result<()> {
        if self.maximum_age_ms <= 0 {
            bail!("BTC model feed maximum age must be positive");
        }
        if self.require_sequence_integrity && self.feed != BtcModelFeedId::BinanceBtcusdtL2V1 {
            bail!("only Binance L2 feed requirements may require sequence integrity");
        }
        Ok(())
    }
}

pub fn validate_feed_requirements(requirements: &[BtcModelFeedRequirement]) -> Result<()> {
    if requirements.is_empty() {
        bail!("BTC model must declare at least one feed requirement");
    }
    for requirement in requirements {
        requirement.validate()?;
    }
    for (index, requirement) in requirements.iter().enumerate() {
        if requirements[index + 1..]
            .iter()
            .any(|other| other.feed == requirement.feed)
        {
            bail!("BTC model feed requirements contain a duplicate feed");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_or_invalid_feed_requirements() {
        assert!(validate_feed_requirements(&[]).is_err());
        assert!(validate_feed_requirements(&[
            BtcModelFeedRequirement {
                feed: BtcModelFeedId::BinanceBtcusdtL2V1,
                maximum_age_ms: 2_000,
                require_sequence_integrity: true,
            },
            BtcModelFeedRequirement {
                feed: BtcModelFeedId::BinanceBtcusdtL2V1,
                maximum_age_ms: 2_000,
                require_sequence_integrity: true,
            },
        ])
        .is_err());
    }
}
