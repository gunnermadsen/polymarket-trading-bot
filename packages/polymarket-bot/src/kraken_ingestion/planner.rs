use anyhow::{Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, TimeZone, Utc};

use super::{
    job::{KrakenDataset, DEFAULT_INTERVAL_SECONDS},
    repository::KrakenRepository,
};

const JOB_DAYS: i64 = 20;

pub async fn enqueue_historical_plan(repository: &KrakenRepository, symbol: &str) -> Result<u64> {
    let end = floor_interval(Utc::now(), DEFAULT_INTERVAL_SECONDS)?;
    let mut inserted = 0_u64;
    for dataset in KrakenDataset::ALL {
        let archive_start = dataset_start(dataset)?;
        let start = repository
            .latest_completed_range_end(dataset, symbol, DEFAULT_INTERVAL_SECONDS)
            .await?
            .map_or(archive_start, |completed_end| {
                completed_end.max(archive_start)
            });
        if start >= end {
            continue;
        }
        if matches!(
            dataset,
            KrakenDataset::Instruments | KrakenDataset::FeeSchedules
        ) {
            inserted += repository
                .enqueue(
                    dataset,
                    symbol,
                    DEFAULT_INTERVAL_SECONDS,
                    start,
                    end,
                    expected_units(start, end),
                )
                .await? as u64;
            continue;
        }

        let mut range_start = start;
        while range_start < end {
            let range_end = (range_start + ChronoDuration::days(JOB_DAYS)).min(end);
            inserted += repository
                .enqueue(
                    dataset,
                    symbol,
                    DEFAULT_INTERVAL_SECONDS,
                    range_start,
                    range_end,
                    expected_units(range_start, range_end),
                )
                .await? as u64;
            range_start = range_end;
        }
    }
    Ok(inserted)
}

fn dataset_start(dataset: KrakenDataset) -> Result<DateTime<Utc>> {
    let date = match dataset {
        KrakenDataset::Instruments | KrakenDataset::FeeSchedules => (2022, 3, 22),
        KrakenDataset::TradeCandles
        | KrakenDataset::AggressorDifferential
        | KrakenDataset::TradeVolume
        | KrakenDataset::TradeCount
        | KrakenDataset::Cvd
        | KrakenDataset::LiquidationVolume => (2022, 3, 23),
        KrakenDataset::MarkCandles | KrakenDataset::SpotCandles | KrakenDataset::FutureBasis => {
            (2022, 3, 22)
        }
        KrakenDataset::OpenInterest => (2023, 3, 7),
        KrakenDataset::Spreads | KrakenDataset::Liquidity | KrakenDataset::Slippage => {
            (2023, 5, 31)
        }
        KrakenDataset::FundingRates => (2022, 3, 22),
    };
    Utc.with_ymd_and_hms(date.0, date.1, date.2, 0, 0, 0)
        .single()
        .context("invalid Kraken dataset start date")
}

fn floor_interval(value: DateTime<Utc>, interval_seconds: i32) -> Result<DateTime<Utc>> {
    let seconds = i64::from(interval_seconds);
    let timestamp = value.timestamp().div_euclid(seconds) * seconds;
    DateTime::from_timestamp(timestamp, 0).context("Kraken planner timestamp outside UTC range")
}

fn expected_units(start: DateTime<Utc>, end: DateTime<Utc>) -> i64 {
    ((end - start).num_seconds() / i64::from(DEFAULT_INTERVAL_SECONDS)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_ranges_fit_below_kraken_page_limit() {
        let buckets =
            ChronoDuration::days(JOB_DAYS).num_seconds() / i64::from(DEFAULT_INTERVAL_SECONDS);
        assert_eq!(buckets, 1_920);
        assert!(buckets < 1_954);
    }

    #[test]
    fn floor_interval_uses_last_complete_bucket_boundary() {
        let value = Utc.with_ymd_and_hms(2026, 8, 21, 12, 17, 42).unwrap();
        assert_eq!(
            floor_interval(value, DEFAULT_INTERVAL_SECONDS).unwrap(),
            Utc.with_ymd_and_hms(2026, 8, 21, 12, 15, 0).unwrap()
        );
    }
}
