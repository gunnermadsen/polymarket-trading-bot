use super::PolygonChainlinkOracleRoundsDrain;
use crate::domain::{DrainRequest, DrainWorkerStrategy, ExecutionSelector};
use chrono::{Duration, Utc};
#[test]
fn preserves_thirty_day_retention() {
    let adapter = PolygonChainlinkOracleRoundsDrain::from_environment().unwrap();
    let request = DrainRequest {
        strategy_key: adapter.descriptor().strategy_key.to_string(),
        cutoff: Utc::now() - Duration::days(29),
        dry_run: true,
        mode: Default::default(),
        execution: ExecutionSelector::default(),
    };
    assert!(adapter.validate_request(&request).is_err());
}
