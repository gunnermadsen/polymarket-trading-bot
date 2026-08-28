//! Administrative API and database-profile reconciliation.

mod api;
mod error;
mod metrics;

pub use api::{ControlApi, ControlReadiness};
