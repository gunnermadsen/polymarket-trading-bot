use chrono::{Duration, Utc};

use crate::domain::{DrainMode, DrainRequest, DrainWorkerStrategy, ExecutionSelector};

use super::{PolymarketOrderbooksDrain, KEY, RETENTION_DAYS};

fn request(cutoff: chrono::DateTime<Utc>) -> DrainRequest {
    DrainRequest {
        strategy_key: KEY.into(),
        cutoff,
        dry_run: true,
        mode: Default::default(),
        execution: ExecutionSelector::default(),
    }
}

#[test]
fn rejects_a_cutoff_inside_the_retained_window() {
    let adapter = PolymarketOrderbooksDrain::from_environment().unwrap();
    let error = adapter
        .validate_request(&request(Utc::now() - Duration::days(RETENTION_DAYS - 1)))
        .unwrap_err();
    assert_eq!(error.code, "drain_retention_violation");
}

#[test]
fn accepts_a_cutoff_older_than_the_retained_window() {
    let adapter = PolymarketOrderbooksDrain::from_environment().unwrap();
    adapter
        .validate_request(&request(Utc::now() - Duration::days(RETENTION_DAYS + 1)))
        .unwrap();
}

#[test]
fn accepts_copy_only_reconciliation_inside_the_retained_window() {
    let adapter = PolymarketOrderbooksDrain::from_environment().unwrap();
    let mut request = request(Utc::now());
    request.mode = DrainMode::Reconcile;
    adapter.validate_request(&request).unwrap();
}

#[test]
fn parquet_schema_contains_every_source_column() {
    let schema = crate::strategies::drains::orderbook_schema::schema();
    assert_eq!(schema.fields().len(), 29);
    assert_eq!(schema.field(0).name(), "sampled_at");
    assert_eq!(schema.field(28).name(), "ingested_at");
}
