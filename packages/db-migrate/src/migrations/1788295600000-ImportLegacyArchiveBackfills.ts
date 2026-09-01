import { MigrationInterface, QueryRunner } from 'typeorm';

const { Client } = require('pg');

type LegacySource = 'kraken' | 'financial_data';

export class ImportLegacyArchiveBackfills1788295600000
  implements MigrationInterface
{
  name = 'ImportLegacyArchiveBackfills1788295600000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    const archive = new Client({
      host: process.env.KRAKEN_ARCHIVE_POSTGRES_HOST || 'kraken-timescaledb',
      port: +(process.env.KRAKEN_ARCHIVE_POSTGRES_PORT || 5432),
      user: process.env.POSTGRES_USER || 'postgres',
      password: requiredEnv('POSTGRES_PASSWORD'),
      database: process.env.KRAKEN_ARCHIVE_POSTGRES_DB || 'kraken_archive',
      application_name: 'unified-ingester-history-import',
    });
    await archive.connect();
    try {
      await archive.query(`SET statement_timeout = '15s'`);
      await this.importJobs(queryRunner, archive, 'kraken');
      await this.importJobs(queryRunner, archive, 'financial_data');
      await this.importEvents(queryRunner, archive, 'kraken');
      await this.importEvents(queryRunner, archive, 'financial_data');
      await this.importKrakenArtifacts(queryRunner, archive);
    } finally {
      await archive.end();
    }
  }

  private async importJobs(
    queryRunner: QueryRunner,
    archive: any,
    source: LegacySource,
  ): Promise<void> {
    const result = await archive.query(
      source === 'kraken'
        ? `SELECT job_id,dataset,symbol,interval_seconds,range_start,range_end,status,
             attempt,max_attempts,next_attempt_at,worker_id,lease_token,lease_expires_at,
             expected_work_units,processed_work_units,rows_written,error_message,
             requested_at,started_at,completed_at,updated_at
           FROM kraken.backfill_jobs ORDER BY requested_at,job_id LIMIT 100000`
        : `SELECT job_id,provider,dataset,series_id,range_start,range_end,status,
             attempt,max_attempts,next_attempt_at,worker_id,lease_token,lease_expires_at,
             rows_written,error_message,requested_at,started_at,completed_at,updated_at
           FROM financial_data.backfill_jobs ORDER BY requested_at,job_id LIMIT 100000`,
    );
    for (const row of result.rows) {
      const legacySource = `${source}.backfill_jobs`;
      const identity = source === 'kraken'
        ? { dataset: row.dataset, symbol: row.symbol, interval_seconds: row.interval_seconds }
        : { provider: row.provider, dataset: row.dataset, series_id: row.series_id };
      const strategyKey = source === 'kraken'
        ? `kraken_${safeKey(row.dataset)}`
        : `${safeKey(row.provider)}_${safeKey(row.dataset)}`;
      const progress = source === 'kraken'
        ? {
            expected_work_units: row.expected_work_units,
            processed_work_units: row.processed_work_units,
            rows_written: row.rows_written,
          }
        : { rows_written: row.rows_written };
      await queryRunner.query(
        `INSERT INTO ingester.backfill_jobs (
           job_kind,strategy_key,strategy_contract_version,request_schema_version,
           canonical_request,request_hash,range_start,range_end,status,attempt,max_attempts,
           next_attempt_at,assigned_worker_id,lease_token,lease_expires_at,heartbeat_at,
           progress,summary,last_error_message,requested_at,started_at,completed_at,
           created_at,updated_at,legacy_source,legacy_job_id
         ) VALUES (
           'request',$1,1,1,$2,
           encode(digest($3,'sha256'),'hex'),$4,$5,$6,$7,$8,$9,$10,$11,$12,$12,
           $13,$14,$15,$16,$17,$18,$16,$19,$20,$21
         ) ON CONFLICT (legacy_source,legacy_job_id) DO NOTHING`,
        [
          strategyKey,
          JSON.stringify({
            strategy_key: strategyKey,
            request_schema_version: 1,
            range: { start: row.range_start, end: row.range_end },
            parameters: identity,
            legacy_source: legacySource,
          }),
          `${legacySource}:${row.job_id}`,
          row.range_start,
          row.range_end,
          normalizeStatus(row.status),
          row.attempt,
          row.max_attempts,
          row.next_attempt_at,
          row.worker_id,
          row.lease_token,
          row.lease_expires_at,
          JSON.stringify(progress),
          JSON.stringify({ legacy_identity: identity }),
          row.error_message,
          row.requested_at,
          row.started_at,
          row.completed_at,
          row.updated_at,
          legacySource,
          row.job_id,
        ],
      );
    }
  }

  private async importEvents(
    queryRunner: QueryRunner,
    archive: any,
    source: LegacySource,
  ): Promise<void> {
    const table = `${source}.backfill_job_events`;
    const result = await archive.query(
      `SELECT event_id,job_id,recorded_at,level,message,metadata
       FROM ${table} ORDER BY recorded_at,event_id LIMIT 200000`,
    );
    for (const row of result.rows) {
      await queryRunner.query(
        `INSERT INTO ingester.backfill_job_events
           (job_id,recorded_at,level,event_code,message,metadata)
         SELECT job_id,$1,$2,'legacy_event',$3,$4
         FROM ingester.backfill_jobs
         WHERE legacy_source=$5 AND legacy_job_id=$6
           AND NOT EXISTS (
             SELECT 1 FROM ingester.backfill_job_events event
             WHERE event.metadata->>'legacy_event_id'=$7
               AND event.metadata->>'legacy_source'=$8
           )`,
        [
          row.recorded_at,
          normalizeLevel(row.level),
          row.message,
          JSON.stringify({
            ...(row.metadata || {}),
            legacy_source: table,
            legacy_event_id: row.event_id,
          }),
          `${source}.backfill_jobs`,
          row.job_id,
          row.event_id,
          table,
        ],
      );
    }
  }

  private async importKrakenArtifacts(
    queryRunner: QueryRunner,
    archive: any,
  ): Promise<void> {
    const result = await archive.query(
      `SELECT artifact_id,job_id,provider,source_url,sha256,lake_relative_path,
         row_count,source_min_time,source_max_time,metadata,created_at
       FROM kraken.backfill_artifacts ORDER BY created_at,artifact_id LIMIT 100000`,
    );
    for (const row of result.rows) {
      await queryRunner.query(
        `INSERT INTO ingester.backfill_artifacts (
           job_id,strategy_key,logical_key,provider,source_uri,checksum,record_count,
           minimum_source_timestamp,maximum_source_timestamp,durable_target,status,
           metadata,created_at,completed_at,legacy_source,legacy_artifact_id
         ) SELECT job_id,strategy_key,$1,$2,$3,$4,$5,$6,$7,$8,'completed',$9,$10,$10,$11,$12
           FROM ingester.backfill_jobs
           WHERE legacy_source='kraken.backfill_jobs' AND legacy_job_id=$13
         ON CONFLICT DO NOTHING`,
        [
          row.lake_relative_path || row.source_url,
          row.provider,
          row.source_url,
          row.sha256,
          row.row_count,
          row.source_min_time,
          row.source_max_time,
          row.lake_relative_path,
          JSON.stringify({ ...(row.metadata || {}), legacy_source: 'kraken.backfill_artifacts' }),
          row.created_at,
          'kraken.backfill_artifacts',
          row.artifact_id,
          row.job_id,
        ],
      );
    }
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DELETE FROM ingester.backfill_artifacts
      WHERE legacy_source='kraken.backfill_artifacts';
      DELETE FROM ingester.backfill_job_events
      WHERE metadata->>'legacy_source' IN (
        'kraken.backfill_job_events','financial_data.backfill_job_events'
      );
      DELETE FROM ingester.backfill_jobs
      WHERE legacy_source IN ('kraken.backfill_jobs','financial_data.backfill_jobs');
    `);
  }
}

function requiredEnv(key: string): string {
  const value = process.env[key];
  if (!value || value.trim() === '') throw new Error(`${key} is required`);
  return value;
}

function safeKey(value: unknown): string {
  return String(value || 'unknown').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}

function normalizeStatus(value: unknown): string {
  const status = String(value || '').toLowerCase();
  if (['queued','running','cancel_requested','completed','failed','cancelled'].includes(status)) return status;
  if (['succeeded','complete'].includes(status)) return 'completed';
  if (['canceled'].includes(status)) return 'cancelled';
  return 'failed';
}

function normalizeLevel(value: unknown): string {
  const level = String(value || '').toLowerCase();
  if (['debug','info','warn','error'].includes(level)) return level;
  return level === 'warning' ? 'warn' : 'info';
}
