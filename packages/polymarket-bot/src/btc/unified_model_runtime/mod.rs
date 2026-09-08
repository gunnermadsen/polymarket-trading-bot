//! Stable model integration boundaries; see docs/unified-model-runtime/README.md.
//! Shared feeds and process execution remain owned by the existing BTC runtime.
pub mod adapters;
pub mod catalog;
pub mod contract;
pub mod telemetry;
#[cfg(test)]
mod tests;
