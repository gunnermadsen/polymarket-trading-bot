//! Bounded immutable book observations for causal model inputs. Execution keeps
//! using the existing current-book registry and its independent safety checks.
use crate::btc::types::OrderbookCheckpoint;
use chrono::{DateTime, Utc};
use std::{collections::VecDeque, sync::Arc};
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BookHistory {
    snapshots: VecDeque<Arc<OrderbookCheckpoint>>,
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
