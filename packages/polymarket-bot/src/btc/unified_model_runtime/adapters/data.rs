//! Bounded immutable book observations for causal model inputs. Execution keeps
//! using the existing current-book registry and its independent safety checks.
use crate::btc::types::{BookReadiness, FeedIntegrityStatus, OrderbookCheckpoint};
use chrono::{DateTime, Utc};
use std::{collections::VecDeque, sync::Arc};
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BookHistory {
    snapshots: VecDeque<Arc<OrderbookCheckpoint>>,
}
/// Bounded diagnostic vocabulary; timestamps stay in logs, never metric labels.
#[derive(Debug)]
pub struct BookInputError {
    pub reason: &'static str,
    pub source_timestamp: Option<DateTime<Utc>>,
    pub received_at: Option<DateTime<Utc>>,
}
impl BookHistory {
    /// Resolve the current book from the captured observation, not the live registry.
    /// This endpoint is the observation boundary; frozen model timestamps still use `at`.
    pub fn observation_book(
        &self,
        market: &str,
        token: &str,
        readiness: Option<&BookReadiness>,
        at: DateTime<Utc>,
        maximum_age: chrono::Duration,
    ) -> Result<&OrderbookCheckpoint, BookInputError> {
        let error = |reason| BookInputError {
            reason,
            source_timestamp: readiness.and_then(|v| v.source_timestamp),
            received_at: readiness.and_then(|v| v.received_at),
        };
        let readiness = readiness.ok_or_else(|| error("missing_snapshot"))?;
        if readiness.market_id != market || readiness.token_id != token {
            return Err(error("identity_mismatch"));
        }
        if !readiness.bootstrapped
            || readiness.source_timestamp.is_none()
            || readiness.received_at.is_none()
        {
            return Err(error("missing_snapshot"));
        }
        if readiness.integrity_status != FeedIntegrityStatus::Ok {
            return Err(error("invalid_integrity"));
        }
        let matching = || {
            self.snapshots
                .iter()
                .filter(|v| v.market_id == market && v.token_id == token)
        };
        if !matching().any(|v| v.connection_id == readiness.connection_id) {
            return Err(error(if matching().next().is_some() {
                "epoch_mismatch"
            } else {
                "missing_snapshot"
            }));
        }
        let book = matching()
            .filter(|v| {
                v.connection_id == readiness.connection_id
                    && v.source_timestamp <= at
                    && v.received_at <= at
            })
            .max_by_key(|v| (v.received_at, v.ingest_sequence))
            .ok_or_else(|| error("future_timestamp"))?;
        let book_error = |reason| BookInputError {
            reason,
            source_timestamp: Some(book.source_timestamp),
            received_at: Some(book.received_at),
        };
        if book.integrity_status != FeedIntegrityStatus::Ok {
            return Err(book_error("invalid_integrity"));
        }
        if at - book.source_timestamp > maximum_age {
            return Err(book_error("stale_source_timestamp"));
        }
        if at - book.received_at > maximum_age {
            return Err(book_error("stale_received_timestamp"));
        }
        Ok(book)
    }
}
impl BookHistory {
    pub fn availability_detail(&self, token: &str, at: DateTime<Utc>) -> String {
        let matching = self
            .snapshots
            .iter()
            .filter(|v| v.token_id == token)
            .collect::<Vec<_>>();
        format!(
            "retained={}, token_snapshots={}, candidate={}, oldest={:?}, newest={:?}",
            self.snapshots.len(),
            matching.len(),
            at,
            matching.first().map(|v| (v.received_at, v.connection_id)),
            matching.last().map(|v| (v.received_at, v.connection_id))
        )
    }
    pub fn observe(&mut self, book: OrderbookCheckpoint) {
        if self.snapshots.back().is_some_and(|v| {
            v.token_id == book.token_id
                && v.connection_id == book.connection_id
                && v.ingest_sequence == book.ingest_sequence
        }) {
            return;
        }
        let oldest = book.received_at - chrono::Duration::seconds(5);
        self.snapshots.retain(|v| {
            v.received_at >= oldest
                && (v.token_id != book.token_id || v.connection_id == book.connection_id)
        });
        // Frozen candidate timestamps are whole seconds. Preserve the latest
        // observation in each completed second instead of allowing a burst of
        // intra-second publications to evict the previous causal boundary.
        self.snapshots.retain(|v| {
            !(v.token_id == book.token_id
                && v.connection_id == book.connection_id
                && (v.received_at.timestamp_micros() - 1).div_euclid(1_000_000)
                    == (book.received_at.timestamp_micros() - 1).div_euclid(1_000_000)
                && v.received_at <= book.received_at)
        });
        self.snapshots.push_back(Arc::new(book));
        while self.snapshots.len() > 64 {
            self.snapshots.pop_front();
        }
    }
    pub fn at(
        &self,
        market: &str,
        token: &str,
        epoch: uuid::Uuid,
        at: DateTime<Utc>,
    ) -> Option<&OrderbookCheckpoint> {
        self.snapshots
            .iter()
            .filter(|v| {
                v.market_id == market
                    && v.token_id == token
                    && v.connection_id == epoch
                    && v.source_timestamp <= at
                    && v.received_at <= at
                    && at - v.received_at <= chrono::Duration::seconds(2)
            })
            .max_by_key(|v| (v.received_at, v.ingest_sequence))
            .map(AsRef::as_ref)
    }
}
