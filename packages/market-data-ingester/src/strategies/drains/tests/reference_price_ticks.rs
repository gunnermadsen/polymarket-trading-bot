use super::ReferencePriceTicksDrain;
use crate::domain::{DrainRequest, DrainWorkerStrategy, ExecutionSelector};
use chrono::{Duration, Utc};

#[test]
fn preserves_two_week_retention() {
    let adapter = ReferencePriceTicksDrain::from_environment().unwrap();
    let request = DrainRequest {
        strategy_key: adapter.descriptor().strategy_key.to_string(),
        cutoff: Utc::now() - Duration::days(13),
        dry_run: true,
        execution: ExecutionSelector::default(),
    };
    assert!(adapter.validate_request(&request).is_err());
}
