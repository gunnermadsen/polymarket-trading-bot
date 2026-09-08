use super::*;
use chrono::Duration;
use rust_decimal_macros::dec;
use std::sync::Arc;

fn point(at: DateTime<Utc>, price: Decimal) -> RtdsPoint {
    RtdsPoint {
        source_timestamp: at,
        available_at: at + Duration::seconds(1),
        price,
    }
}

#[test]
fn hydrated_candles_preserve_every_legacy_ohlc_and_availability_value() {
    let at = DateTime::from_timestamp(1788818400, 0).unwrap();
    let mut repo = RtdsRepository::default();
    // Live observation arrives before the overlapping historical seed.
    repo.insert(point(at - Duration::seconds(10), dec!(160)))
        .unwrap();
    for minute in 0..61 {
        let open = at - Duration::minutes(61 - minute);
        let base = dec!(100) + Decimal::from(minute);
        for (second, price) in [
            (5, base),
            (20, base + dec!(2)),
            (30, base - dec!(1)),
            (50, base),
        ] {
            repo.insert(point(open + Duration::seconds(second), price))
                .unwrap();
        }
    }
    assert_eq!(repo.complete_minutes(at), 61);
    let candles = repo.closed_candles(at).unwrap();
    for (index, candle) in candles.iter().enumerate() {
        let open = at - Duration::minutes(61 - index as i64);
        let base = dec!(100) + Decimal::from(index);
        assert_eq!(
            *candle,
            DirectionalChainlinkCandle {
                open_timestamp: open,
                close_timestamp: open + Duration::minutes(1),
                open_price: base,
                high_price: base + dec!(2),
                low_price: base - dec!(1),
                close_price: base,
                available_at: open + Duration::seconds(51),
            }
        );
    }
}

#[test]
fn duplicate_seed_preserves_first_seen_value_and_snapshot_isolation() {
    let at = DateTime::from_timestamp(1788818400, 0).unwrap();
    let mut owner = Arc::new(RtdsRepository::default());
    Arc::make_mut(&mut owner)
        .insert(point(at, dec!(100)))
        .unwrap();
    let consumer = owner.clone();
    assert!(Arc::ptr_eq(&owner, &consumer));
    Arc::make_mut(&mut owner)
        .insert(point(at, dec!(999)))
        .unwrap();
    Arc::make_mut(&mut owner)
        .insert(point(at + Duration::seconds(1), dec!(101)))
        .unwrap();
    assert_eq!(consumer.points_as_of(at + Duration::seconds(3)).count(), 1);
    let points = owner
        .points_as_of(at + Duration::seconds(3))
        .collect::<Vec<_>>();
    assert_eq!(points.len(), 2);
    assert_eq!(points[0].price, dec!(100));
    assert_eq!(owner.points_as_of(at).count(), 0);
}

#[test]
fn missing_and_future_history_fail_closed_with_existing_diagnostics() {
    let at = DateTime::from_timestamp(1788818400, 0).unwrap();
    let mut repo = RtdsRepository::default();
    for minute in 1..=61 {
        let mut p = point(at - Duration::minutes(minute), dec!(100));
        if minute == 20 {
            p.available_at = at + Duration::seconds(1);
        }
        repo.insert(p).unwrap();
    }
    assert_eq!(repo.complete_minutes(at), 60);
    assert!(repo
        .closed_candles(at)
        .unwrap_err()
        .to_string()
        .contains("61 contiguous closed RTDS"));
    assert_eq!(
        repo.closed_candles(at + Duration::seconds(1))
            .unwrap()
            .len(),
        61
    );
}

#[test]
fn out_of_order_hydration_cannot_evict_newer_ticks_or_exceed_capacity() {
    let at = DateTime::from_timestamp(1788818400, 0).unwrap();
    let mut repo = RtdsRepository::default();
    for second in (0..5000).rev() {
        repo.insert(point(at + Duration::seconds(second), dec!(100)))
            .unwrap();
    }
    assert_eq!(repo.points.len(), RTDS_MID_CAPACITY);
    assert_eq!(
        repo.points.back().unwrap().source_timestamp,
        at + Duration::seconds(4999)
    );
    assert_eq!(
        repo.points.front().unwrap().source_timestamp,
        at + Duration::seconds(904)
    );
}
