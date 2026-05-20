use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};

use crate::orderbook::LocalOrderBook;

#[derive(Debug, Default)]
pub struct WsBookState {
    books: HashMap<String, LocalOrderBook>,
    tick_size_changed_at: HashMap<String, DateTime<Utc>>,
}

impl WsBookState {
    pub fn upsert_book(&mut self, token_id: impl Into<String>, book: LocalOrderBook) {
        self.books.insert(token_id.into(), book);
    }

    pub fn book(&self, token_id: &str) -> Option<&LocalOrderBook> {
        self.books.get(token_id)
    }

    pub fn mark_tick_size_change(&mut self, token_id: impl Into<String>, at: DateTime<Utc>) {
        self.tick_size_changed_at.insert(token_id.into(), at);
    }

    pub fn tick_size_pause_active(&self, token_id: &str, now: DateTime<Utc>) -> bool {
        self.tick_size_changed_at
            .get(token_id)
            .map(|changed_at| now - *changed_at < Duration::seconds(60))
            .unwrap_or(false)
    }
}

// v1 keeps the WebSocket state container separate from orchestration. The live stream task should
// feed decoded CLOB book, price_change, best_bid_ask, and tick_size_change events into WsBookState.
