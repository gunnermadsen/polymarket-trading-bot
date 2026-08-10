//! Strategy registry, lifecycle supervision, leases, and health state.

mod registry;
mod supervisor;

pub use registry::{StrategyFactory, StrategyFactoryError, StrategyRegistry};
pub(crate) use supervisor::{StrategySupervisor, SupervisorSettings};
