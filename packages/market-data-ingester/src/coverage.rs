use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::{FromRow, PgPool};

const MAX_KNOWN_GAPS_PER_PRODUCT: i64 = 10_000;

#[derive(Debug, Clone)]
pub struct CoverageTarget {
    pub product_key: String,
    pub relation: Option<String>,
    pub backfill_strategy_keys: Vec<String>,
    pub drain_strategy_key: Option<String>,
    pub gap_strategy_keys: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CoverageReport {
    pub generated_at: DateTime<Utc>,
    pub products: Vec<ProductCoverage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProductCoverage {
    pub product_key: String,
    pub relation: Option<String>,
    pub backfill_strategy_key: Option<String>,
    pub backfill_strategy_keys: Vec<String>,
    pub database: DatabaseCoverage,
    pub ssd: SsdCoverage,
    pub combined: CombinedCoverage,
    pub gaps: Vec<CoverageGap>,
    pub known_gaps_truncated: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatabaseCoverage {
    pub from: Option<DateTime<Utc>>,
    pub through: Option<DateTime<Utc>>,
    pub closed_chunks: usize,
    pub open_chunks: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SsdCoverage {
    pub from: Option<DateTime<Utc>>,
    pub through: Option<DateTime<Utc>>,
    pub verified_objects: usize,
    pub rows: i64,
    pub bytes: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CombinedCoverage {
    pub from: Option<DateTime<Utc>>,
    pub through: Option<DateTime<Utc>>,
    pub interval_count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CoverageGap {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub kind: String,
    pub status: String,
    pub reason_code: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Interval {
    start: DateTime<Utc>,
    end: DateTime<Utc>,
}

#[derive(Debug, FromRow)]
struct ChunkRow {
    product_key: String,
    range_start: DateTime<Utc>,
    range_end: DateTime<Utc>,
    replicated: bool,
}

#[derive(Debug, FromRow)]
struct ObjectRow {
    product_key: String,
    source_start: DateTime<Utc>,
    source_end: DateTime<Utc>,
    row_count: i64,
    byte_size: i64,
}

#[derive(Debug, FromRow)]
struct ArtifactRow {
    product_key: String,
    source_start: DateTime<Utc>,
    source_end: DateTime<Utc>,
    row_count: i64,
    byte_size: i64,
    object_count: i64,
}

#[derive(Debug, FromRow)]
struct GapRow {
    product_key: String,
    source_time_start: DateTime<Utc>,
    source_time_end: DateTime<Utc>,
    status: String,
    reason_code: String,
    total_count: i64,
}

pub async fn detect(
    pool: &PgPool,
    targets: Vec<CoverageTarget>,
) -> Result<CoverageReport, sqlx::Error> {
    let generated_at = Utc::now();
    let gap_targets = targets
        .iter()
        .flat_map(|target| {
            target
                .gap_strategy_keys
                .iter()
                .map(|key| (target.product_key.clone(), key.clone()))
        })
        .collect::<Vec<_>>();
    let gap_product_keys = gap_targets
        .iter()
        .map(|(product_key, _)| product_key.clone())
        .collect::<Vec<_>>();
    let gap_strategy_keys = gap_targets
        .iter()
        .map(|(_, strategy_key)| strategy_key.clone())
        .collect::<Vec<_>>();
    let relation_targets = targets
        .iter()
        .filter_map(|target| {
            target.relation.as_deref().map(|relation| {
                let (schema, table) = relation_parts(relation);
                (
                    target.product_key.clone(),
                    schema.to_owned(),
                    table.to_owned(),
                    target.drain_strategy_key.clone().unwrap_or_default(),
                )
            })
        })
        .collect::<Vec<_>>();
    let relation_product_keys = relation_targets
        .iter()
        .map(|(product_key, _, _, _)| product_key.clone())
        .collect::<Vec<_>>();
    let schemas = relation_targets
        .iter()
        .map(|(_, schema, _, _)| schema.clone())
        .collect::<Vec<_>>();
    let tables = relation_targets
        .iter()
        .map(|(_, _, table, _)| table.clone())
        .collect::<Vec<_>>();
    let relation_drain_keys = relation_targets
        .iter()
        .map(|(_, _, _, drain_key)| drain_key.clone())
        .collect::<Vec<_>>();

    let chunks = sqlx::query_as::<_, ChunkRow>(
        "WITH targets AS (SELECT * FROM unnest($1::text[],$2::text[],$3::text[],$4::text[]) \
         AS target(product_key,schema_name,table_name,drain_strategy_key)) \
         SELECT target.product_key,chunk.range_start,chunk.range_end, \
         EXISTS (SELECT 1 FROM ingester.drain_objects object \
         WHERE object.strategy_key=NULLIF(target.drain_strategy_key,'') \
         AND object.source_chunk_schema=chunk.chunk_schema \
         AND object.source_chunk_name=chunk.chunk_name \
         AND object.status IN ('published','removed')) AS replicated \
         FROM targets target JOIN timescaledb_information.chunks chunk \
         ON chunk.hypertable_schema=target.schema_name \
         AND chunk.hypertable_name=target.table_name \
         ORDER BY target.product_key,chunk.range_start,chunk.range_end",
    )
    .bind(&relation_product_keys)
    .bind(&schemas)
    .bind(&tables)
    .bind(&relation_drain_keys)
    .fetch_all(pool)
    .await?;
    let drain_targets = targets
        .iter()
        .filter_map(|target| {
            target
                .drain_strategy_key
                .as_ref()
                .map(|key| (target.product_key.clone(), key.clone()))
        })
        .collect::<Vec<_>>();
    let drain_product_keys = drain_targets
        .iter()
        .map(|(product_key, _)| product_key.clone())
        .collect::<Vec<_>>();
    let drain_strategy_keys = drain_targets
        .iter()
        .map(|(_, strategy_key)| strategy_key.clone())
        .collect::<Vec<_>>();
    let objects = sqlx::query_as::<_, ObjectRow>(
        "WITH targets AS (SELECT * FROM unnest($1::text[],$2::text[]) \
         AS target(product_key,strategy_key)) \
         SELECT target.product_key,object.source_start,object.source_end, \
         object.row_count,object.byte_size FROM targets target \
         JOIN ingester.drain_objects object ON object.strategy_key=target.strategy_key \
         WHERE object.status IN ('published','removed') \
         ORDER BY target.product_key,object.source_start,object.source_end",
    )
    .bind(&drain_product_keys)
    .bind(&drain_strategy_keys)
    .fetch_all(pool)
    .await?;
    let artifact_targets = targets
        .iter()
        .flat_map(|target| {
            target
                .backfill_strategy_keys
                .iter()
                .map(|key| (target.product_key.clone(), key.clone()))
        })
        .collect::<Vec<_>>();
    let artifact_product_keys = artifact_targets
        .iter()
        .map(|(product_key, _)| product_key.clone())
        .collect::<Vec<_>>();
    let artifact_strategy_keys = artifact_targets
        .iter()
        .map(|(_, strategy_key)| strategy_key.clone())
        .collect::<Vec<_>>();
    let artifacts = sqlx::query_as::<_, ArtifactRow>(
        "WITH targets AS (SELECT * FROM unnest($1::text[],$2::text[]) \
         AS target(product_key,strategy_key)) \
         SELECT target.product_key,job.range_start AS source_start,job.range_end AS source_end, \
         COALESCE(SUM(artifact.record_count),0)::bigint AS row_count, \
         COALESCE(SUM(artifact.byte_size),0)::bigint AS byte_size, \
         COUNT(*)::bigint AS object_count \
         FROM targets target JOIN ingester.backfill_jobs job \
         ON job.strategy_key=target.strategy_key AND job.status='completed' \
         JOIN ingester.backfill_artifacts artifact ON artifact.job_id=job.job_id \
         AND artifact.strategy_key=target.strategy_key AND artifact.status='completed' \
         AND artifact.checksum ~ '^[0-9a-f]{64}$' AND artifact.durable_target IS NOT NULL \
         GROUP BY target.product_key,job.job_id,job.range_start,job.range_end \
         ORDER BY target.product_key,job.range_start,job.range_end",
    )
    .bind(&artifact_product_keys)
    .bind(&artifact_strategy_keys)
    .fetch_all(pool)
    .await?;
    let known_gaps = sqlx::query_as::<_, GapRow>(
        "WITH targets AS (SELECT * FROM unnest($1::text[],$2::text[]) \
         AS target(product_key,strategy_key)), \
         ranked AS (SELECT target.product_key,gap.source_time_start,gap.source_time_end, \
         gap.status,gap.reason_code,count(*) OVER (PARTITION BY target.product_key)::bigint AS total_count, \
         row_number() OVER (PARTITION BY target.product_key ORDER BY gap.source_time_start,gap.source_time_end,gap.gap_id) AS position \
         FROM targets target JOIN ingester.data_gaps gap ON gap.strategy_key=target.strategy_key \
         WHERE gap.source_time_start IS NOT NULL AND gap.source_time_end IS NOT NULL \
         AND gap.status IN ('open','repairing','unrecoverable')) \
         SELECT product_key,source_time_start,source_time_end,status,reason_code,total_count \
         FROM ranked WHERE position<=$3 ORDER BY product_key,source_time_start,source_time_end",
    )
    .bind(&gap_product_keys)
    .bind(&gap_strategy_keys)
    .bind(MAX_KNOWN_GAPS_PER_PRODUCT)
    .fetch_all(pool)
    .await?;

    let mut products = Vec::with_capacity(targets.len());
    for target in targets {
        let product_chunks = chunks
            .iter()
            .filter(|row| row.product_key == target.product_key)
            .collect::<Vec<_>>();
        let product_objects = objects
            .iter()
            .filter(|row| row.product_key == target.product_key)
            .collect::<Vec<_>>();
        let product_artifacts = artifacts
            .iter()
            .filter(|row| row.product_key == target.product_key)
            .collect::<Vec<_>>();
        let closed_chunks = product_chunks
            .iter()
            .filter(|row| row.range_end <= generated_at)
            .copied()
            .collect::<Vec<_>>();
        let database_intervals = product_chunks
            .iter()
            .map(|row| Interval {
                start: row.range_start,
                end: row.range_end,
            })
            .collect::<Vec<_>>();
        let ssd_intervals = product_objects
            .iter()
            .map(|row| Interval {
                start: row.source_start,
                end: row.source_end,
            })
            .chain(product_artifacts.iter().map(|row| Interval {
                start: row.source_start,
                end: row.source_end,
            }))
            .collect::<Vec<_>>();
        let combined = merge_intervals(
            database_intervals
                .iter()
                .chain(ssd_intervals.iter())
                .copied()
                .collect(),
        );
        let mut gaps = gaps_between(&combined);
        gaps.extend(
            closed_chunks
                .iter()
                .filter(|row| !row.replicated)
                .map(|row| CoverageGap {
                    start: row.range_start,
                    end: row.range_end,
                    kind: "ssd_replication_gap".to_owned(),
                    status: "actionable".to_owned(),
                    reason_code: "closed_database_chunk_not_archived".to_owned(),
                }),
        );
        gaps.extend(
            known_gaps
                .iter()
                .filter(|row| row.product_key == target.product_key)
                .map(|row| CoverageGap {
                    start: row.source_time_start,
                    end: row.source_time_end,
                    kind: "recorded_ingestion_gap".to_owned(),
                    status: row.status.clone(),
                    reason_code: row.reason_code.clone(),
                }),
        );
        gaps.sort_by_key(|gap| (gap.start, gap.end, gap.kind.clone()));
        let known_gap_count = known_gaps
            .iter()
            .find(|row| row.product_key == target.product_key)
            .map_or(0, |row| row.total_count);
        let backfill_strategy_key = target.backfill_strategy_keys.first().cloned();
        products.push(ProductCoverage {
            product_key: target.product_key,
            relation: target.relation,
            backfill_strategy_key,
            backfill_strategy_keys: target.backfill_strategy_keys,
            database: DatabaseCoverage {
                from: database_intervals
                    .iter()
                    .map(|interval| interval.start)
                    .min(),
                through: database_intervals.iter().map(|interval| interval.end).max(),
                closed_chunks: closed_chunks.len(),
                open_chunks: product_chunks.len() - closed_chunks.len(),
            },
            ssd: SsdCoverage {
                from: ssd_intervals.iter().map(|interval| interval.start).min(),
                through: ssd_intervals.iter().map(|interval| interval.end).max(),
                verified_objects: product_objects.len()
                    + product_artifacts
                        .iter()
                        .map(|row| usize::try_from(row.object_count).unwrap_or(usize::MAX))
                        .fold(0usize, usize::saturating_add),
                rows: product_objects.iter().map(|row| row.row_count).sum::<i64>()
                    + product_artifacts
                        .iter()
                        .map(|row| row.row_count)
                        .sum::<i64>(),
                bytes: product_objects.iter().map(|row| row.byte_size).sum::<i64>()
                    + product_artifacts
                        .iter()
                        .map(|row| row.byte_size)
                        .sum::<i64>(),
            },
            combined: CombinedCoverage {
                from: combined.first().map(|interval| interval.start),
                through: combined.last().map(|interval| interval.end),
                interval_count: combined.len(),
            },
            gaps,
            known_gaps_truncated: known_gap_count > MAX_KNOWN_GAPS_PER_PRODUCT,
        });
    }
    products.sort_by(|left, right| left.product_key.cmp(&right.product_key));
    Ok(CoverageReport {
        generated_at,
        products,
    })
}

fn relation_parts(relation: &str) -> (&str, &str) {
    relation
        .split_once('.')
        .expect("registered drain relation must be schema-qualified")
}

fn merge_intervals(mut intervals: Vec<Interval>) -> Vec<Interval> {
    intervals.sort_by_key(|interval| (interval.start, interval.end));
    let mut merged: Vec<Interval> = Vec::new();
    for interval in intervals {
        if let Some(previous) = merged.last_mut() {
            if interval.start <= previous.end {
                previous.end = previous.end.max(interval.end);
                continue;
            }
        }
        merged.push(interval);
    }
    merged
}

fn gaps_between(intervals: &[Interval]) -> Vec<CoverageGap> {
    intervals
        .windows(2)
        .map(|pair| CoverageGap {
            start: pair[0].end,
            end: pair[1].start,
            kind: "storage_coverage_gap".to_owned(),
            status: "observed".to_owned(),
            reason_code: "no_database_chunk_or_verified_ssd_object".to_owned(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::{gaps_between, merge_intervals, Interval};

    fn at(hour: u32) -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 1, hour, 0, 0).unwrap()
    }

    #[test]
    fn merges_overlapping_and_adjacent_database_and_ssd_intervals() {
        assert_eq!(
            merge_intervals(vec![
                Interval {
                    start: at(2),
                    end: at(3),
                },
                Interval {
                    start: at(0),
                    end: at(1),
                },
                Interval {
                    start: at(1),
                    end: at(2),
                },
            ]),
            vec![Interval {
                start: at(0),
                end: at(3)
            }]
        );
    }

    #[test]
    fn reports_only_internal_storage_gaps() {
        let gaps = gaps_between(&[
            Interval {
                start: at(0),
                end: at(1),
            },
            Interval {
                start: at(2),
                end: at(3),
            },
        ]);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start, at(1));
        assert_eq!(gaps[0].end, at(2));
    }
}
