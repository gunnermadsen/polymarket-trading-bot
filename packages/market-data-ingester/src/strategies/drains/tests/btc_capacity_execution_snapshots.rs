use super::BtcCapacityExecutionSnapshotsDrain;
use crate::domain::DrainWorkerStrategy;
#[test]
fn uses_existing_drain_contract() {
    let adapter = BtcCapacityExecutionSnapshotsDrain::from_environment().unwrap();
    assert_eq!(adapter.descriptor().contract_version, 1);
    assert_eq!(
        adapter.descriptor().relation.as_ref(),
        "polymarket.btc_market_capacity_execution_snapshots"
    );
}
