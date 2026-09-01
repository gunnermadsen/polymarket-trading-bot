//! Strategy registry, lifecycle supervision, leases, and health state.

mod backfill_worker;
mod registry;
mod supervisor;

pub(crate) use backfill_worker::BackfillWorkerRuntime;
pub use registry::{StrategyFactory, StrategyFactoryError, StrategyRegistry};
pub(crate) use supervisor::{StrategySupervisor, SupervisorSettings};
