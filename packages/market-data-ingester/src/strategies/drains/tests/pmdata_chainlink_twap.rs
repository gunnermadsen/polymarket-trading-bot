use super::PmdataChainlinkTwapDrain;
use crate::domain::DrainWorkerStrategy;

#[test]
fn exposes_archive_drain_identity() {
    let adapter = PmdataChainlinkTwapDrain::from_environment().unwrap();
    assert_eq!(
        adapter.descriptor().relation.as_ref(),
        "market_data.pmdata_chainlink_btcusd_twap"
    );
}
