use super::IngesterStrategyKey;

pub const ALLOCATION_CONTRACT_VERSION: i32 = 1;
pub const DEFAULT_WORKER_CAPACITY_UNITS: i32 = 4;
pub const DEFAULT_REALTIME_SLOT_LIMIT: i32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationClass {
    LatencyCritical,
    Standard,
    Heavy,
    Exclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkloadProfile {
    pub capacity_units: i32,
    pub isolation: IsolationClass,
}

pub const fn realtime_profile(key: IngesterStrategyKey) -> WorkloadProfile {
    let isolation = match key {
        IngesterStrategyKey::PolymarketBtcFiveMinuteOrderbooks
        | IngesterStrategyKey::BinanceSpotBtcusdtL2Snapshots => IsolationClass::LatencyCritical,
        _ => IsolationClass::Standard,
    };
    WorkloadProfile {
        capacity_units: 2,
        isolation,
    }
}

pub fn backfill_profile(strategy_key: &str) -> WorkloadProfile {
    match strategy_key {
        "pmxt_polymarket_orderbook_archives" => WorkloadProfile {
            capacity_units: 4,
            isolation: IsolationClass::Exclusive,
        },
        "binance_spot_btcusdt_aggregate_trades" => WorkloadProfile {
            capacity_units: 3,
            isolation: IsolationClass::Heavy,
        },
        _ => WorkloadProfile {
            capacity_units: 2,
            isolation: IsolationClass::Standard,
        },
    }
}

pub fn admits_realtime(
    worker_capacity: i32,
    realtime_slot_limit: i32,
    active_realtime: i64,
    active_backfill_units: i64,
    candidate: WorkloadProfile,
) -> bool {
    active_realtime < i64::from(realtime_slot_limit)
        && active_backfill_units + i64::from(candidate.capacity_units) <= i64::from(worker_capacity)
        && !(candidate.isolation == IsolationClass::LatencyCritical && active_backfill_units > 0)
}

pub fn admits_backfill(
    worker_capacity: i32,
    active_realtime: Option<WorkloadProfile>,
    active_backfill_units: i64,
    candidate: WorkloadProfile,
) -> bool {
    if active_realtime.is_some_and(|profile| profile.isolation == IsolationClass::LatencyCritical) {
        return false;
    }
    if candidate.isolation == IsolationClass::Exclusive
        && (active_realtime.is_some() || active_backfill_units > 0)
    {
        return false;
    }
    let realtime_units = active_realtime.map_or(0, |profile| i64::from(profile.capacity_units));
    realtime_units + active_backfill_units + i64::from(candidate.capacity_units)
        <= i64::from(worker_capacity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realtime_slot_is_strict() {
        let candidate = realtime_profile(IngesterStrategyKey::BinanceSpotBtcusdtOneSecondOhlcv);
        assert!(admits_realtime(4, 1, 0, 0, candidate));
        assert!(!admits_realtime(4, 1, 1, 0, candidate));
    }

    #[test]
    fn latency_critical_realtime_is_exclusive_from_backfills() {
        let candidate = realtime_profile(IngesterStrategyKey::PolymarketBtcFiveMinuteOrderbooks);
        assert!(!admits_realtime(4, 1, 0, 2, candidate));
        assert!(!admits_backfill(
            4,
            Some(candidate),
            0,
            backfill_profile("ordinary")
        ));
    }

    #[test]
    fn standard_workloads_fit_the_capacity_cup() {
        let realtime = realtime_profile(IngesterStrategyKey::BinanceSpotBtcusdtOneSecondOhlcv);
        assert!(admits_backfill(
            4,
            Some(realtime),
            0,
            backfill_profile("ordinary")
        ));
        assert!(!admits_backfill(
            4,
            Some(realtime),
            2,
            backfill_profile("ordinary")
        ));
    }

    #[test]
    fn exclusive_backfill_requires_an_empty_worker() {
        let candidate = WorkloadProfile {
            capacity_units: 4,
            isolation: IsolationClass::Exclusive,
        };
        assert!(admits_backfill(4, None, 0, candidate));
        assert!(!admits_backfill(4, None, 2, candidate));
    }
}
