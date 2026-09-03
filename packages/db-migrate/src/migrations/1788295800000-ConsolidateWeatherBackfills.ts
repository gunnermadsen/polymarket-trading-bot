import { MigrationInterface, QueryRunner } from 'typeorm';
import { AddTemperatureEnvironmentalFeatures1787760000000 } from './weather/1787760000000-AddTemperatureEnvironmentalFeatures';
import { AddTemperatureSpatialGradients1787767200000 } from './weather/1787767200000-AddTemperatureSpatialGradients';

const { Client } = require('pg');

const ARTIFACT_JOB_IDS: Record<string, string> = {
  goes_abi_klga_features: '7f7d4f98-2c88-4f5e-9a00-000000000001',
  hrrr_environment_features: '7f7d4f98-2c88-4f5e-9a00-000000000002',
  weather_legacy_artifacts: '7f7d4f98-2c88-4f5e-9a00-000000000003',
};

const ACTIVE_PARENT_IDS: Record<string, string> = {
  goes_abi_klga_features: '7f7d4f98-2c88-4f5e-9a00-000000000011',
  hrrr_environment_features: '7f7d4f98-2c88-4f5e-9a00-000000000012',
};

const WEATHER_TABLES = [
  'goes_abi_features',
  'goes_abi_window_coverage',
  'hrrr_environment_features',
  'hrrr_environment_window_coverage',
] as const;

export class ConsolidateWeatherBackfills1788295800000
  implements MigrationInterface
{
  name = 'ConsolidateWeatherBackfills1788295800000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    const source = new Client({
      host: process.env.WEATHER_ARCHIVE_POSTGRES_HOST || 'temperature-postgres',
      port: +(process.env.WEATHER_ARCHIVE_POSTGRES_PORT || 5432),
      user: process.env.WEATHER_ARCHIVE_POSTGRES_USER || 'postgres',
      password:
        process.env.WEATHER_ARCHIVE_POSTGRES_PASSWORD ||
        process.env.POSTGRES_PASSWORD,
      database:
        process.env.WEATHER_ARCHIVE_POSTGRES_DB || 'temperature_expectancy',
      application_name: 'unified-ingester-weather-import',
    });
    await source.connect();
    try {
      await source.query(`SET statement_timeout = 0`);
      await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS weather`);
      await this.createArtifactStaging(queryRunner);
      await new AddTemperatureEnvironmentalFeatures1787760000000().up(
        queryRunner,
      );
      await new AddTemperatureSpatialGradients1787767200000().up(queryRunner);
      await queryRunner.query(`
        ALTER TABLE weather.goes_abi_features
          DROP CONSTRAINT goes_abi_features_process_id_fkey;
        ALTER TABLE weather.goes_abi_window_coverage
          DROP CONSTRAINT goes_abi_window_coverage_process_id_fkey;
        ALTER TABLE weather.hrrr_environment_features
          DROP CONSTRAINT hrrr_environment_features_process_id_fkey;
        ALTER TABLE weather.hrrr_environment_window_coverage
          DROP CONSTRAINT hrrr_environment_window_coverage_process_id_fkey;
      `);

      const target = (queryRunner as any).databaseConnection;
      await copyInBatches(
        source,
        target,
        'weather',
        'source_artifacts',
      );

      await this.createArtifactLedgerJobs(queryRunner);
      await queryRunner.query(`
        INSERT INTO ingester.backfill_artifacts (
          artifact_id,job_id,strategy_key,logical_key,provider,source_uri,
          checksum,byte_size,record_count,minimum_source_timestamp,
          maximum_source_timestamp,status,metadata,created_at,completed_at,
          legacy_source,legacy_artifact_id
        )
        SELECT
          artifact_id,
          CASE
            WHEN provider='noaa_goes_open_data' THEN $1::uuid
            WHEN provider='noaa_hrrr_open_data' THEN $2::uuid
            ELSE $3::uuid
          END,
          CASE
            WHEN provider='noaa_goes_open_data' THEN 'goes_abi_klga_features'
            WHEN provider='noaa_hrrr_open_data' THEN 'hrrr_environment_features'
            ELSE 'weather_legacy_artifacts'
          END,
          CASE
            WHEN provider IN ('noaa_goes_open_data','noaa_hrrr_open_data')
              THEN logical_key
            ELSE provider || ':' || logical_key
          END,
          provider,source_uri,sha256,compressed_bytes,record_count,
          source_start,source_end,'completed',
          metadata || jsonb_build_object('legacy_source','weather.source_artifacts'),
          ingested_at,ingested_at,'weather.source_artifacts',artifact_id
        FROM weather.source_artifacts
        ON CONFLICT DO NOTHING
      `, [
        ARTIFACT_JOB_IDS.goes_abi_klga_features,
        ARTIFACT_JOB_IDS.hrrr_environment_features,
        ARTIFACT_JOB_IDS.weather_legacy_artifacts,
      ]);

      for (const table of WEATHER_TABLES) {
        await copyInBatches(
          source,
          target,
          'weather',
          table,
        );
      }

      await this.importJobs(queryRunner, source);
      await this.reconcile(queryRunner, source);

      await queryRunner.query(`
        ALTER TABLE weather.goes_abi_window_coverage
          DROP CONSTRAINT goes_abi_window_coverage_source_artifact_id_fkey;
        ALTER TABLE weather.hrrr_environment_window_coverage
          DROP CONSTRAINT hrrr_environment_window_coverage_source_artifact_id_fkey;
        ALTER TABLE weather.goes_abi_window_coverage
          ADD CONSTRAINT goes_abi_window_coverage_source_artifact_id_fkey
          FOREIGN KEY (source_artifact_id)
          REFERENCES ingester.backfill_artifacts (artifact_id) ON DELETE RESTRICT;
        ALTER TABLE weather.hrrr_environment_window_coverage
          ADD CONSTRAINT hrrr_environment_window_coverage_source_artifact_id_fkey
          FOREIGN KEY (source_artifact_id)
          REFERENCES ingester.backfill_artifacts (artifact_id) ON DELETE RESTRICT;
        DROP TABLE weather.source_artifacts;
      `);
    } finally {
      await source.end();
    }
  }

  private async createArtifactStaging(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE weather.source_artifacts (
        artifact_id uuid PRIMARY KEY,
        provider text NOT NULL,
        logical_key text NOT NULL,
        source_uri text NOT NULL,
        source_start timestamptz,
        source_end timestamptz,
        sha256 text,
        compressed_bytes bigint,
        record_count bigint NOT NULL,
        metadata jsonb NOT NULL,
        ingested_at timestamptz NOT NULL,
        UNIQUE (provider,logical_key)
      );
    `);
  }

  private async createArtifactLedgerJobs(queryRunner: QueryRunner): Promise<void> {
    for (const [strategyKey, jobId] of Object.entries(ARTIFACT_JOB_IDS)) {
      await queryRunner.query(`
        INSERT INTO ingester.backfill_jobs (
          job_id,job_kind,strategy_key,strategy_contract_version,
          request_schema_version,canonical_request,request_hash,range_start,
          range_end,status,attempt,max_attempts,next_attempt_at,summary,
          requested_at,started_at,completed_at,created_at,updated_at,
          legacy_source,legacy_job_id
        ) VALUES (
          $1::uuid,'request',$2::text,1,1,
          jsonb_build_object('strategy_key',$2::text,'parameters',jsonb_build_object(),
            'range',jsonb_build_object('start','2014-01-01T00:00:00Z',
              'end','2026-09-01T00:00:00Z')),
          encode(digest('weather-artifacts:' || $2::text,'sha256'),'hex'),
          '2014-01-01T00:00:00Z','2026-09-01T00:00:00Z','completed',1,1,
          now(),jsonb_build_object('imported_legacy_artifact_ledger',true),
          now(),now(),now(),now(),now(),'weather.artifact_ledger',$1::uuid
        ) ON CONFLICT (legacy_source,legacy_job_id) DO NOTHING
      `, [jobId, strategyKey]);
    }
  }

  private async importJobs(queryRunner: QueryRunner, source: any): Promise<void> {
    const result = await source.query(`
      SELECT job_id,depends_on_job_id,ingester_key,idempotency_key,range_start,
        range_end,request,status,attempt,max_attempts,next_attempt_at,progress,
        summary,error,requested_at,started_at,completed_at,updated_at
      FROM weather.ingestion_jobs
      ORDER BY requested_at,job_id
      LIMIT 10000
    `);
    const active = result.rows.filter((row: any) =>
      ['queued', 'running', 'cancel_requested'].includes(row.status),
    );
    for (const strategyKey of Object.keys(ACTIVE_PARENT_IDS)) {
      const rows = active.filter((row: any) => row.ingester_key === strategyKey);
      if (rows.length === 0) continue;
      const start = rows.reduce(
        (value: Date, row: any) => (row.range_start < value ? row.range_start : value),
        rows[0].range_start,
      );
      const end = rows.reduce(
        (value: Date, row: any) => (row.range_end > value ? row.range_end : value),
        rows[0].range_end,
      );
      await queryRunner.query(`
        INSERT INTO ingester.backfill_jobs (
          job_id,job_kind,strategy_key,strategy_contract_version,
          request_schema_version,canonical_request,request_hash,range_start,
          range_end,status,attempt,max_attempts,next_attempt_at,progress,checkpoint,
          summary,requested_at,created_at,updated_at,legacy_source,legacy_job_id
        ) VALUES (
          $1::uuid,'request',$2::text,1,1,
          jsonb_build_object('strategy_key',$2::text,'parameters',jsonb_build_object(),
            'range',jsonb_build_object('start',$3::timestamptz,'end',$4::timestamptz)),
          encode(digest('weather-active-parent:' || $2::text,'sha256'),'hex'),
          $3,$4,'queued',0,1,now(),'{}','{}',
          jsonb_build_object('migrated_active_shards',$5::int),now(),now(),now(),
          'weather.active_parent',$1::uuid
        ) ON CONFLICT (legacy_source,legacy_job_id) DO NOTHING
      `, [ACTIVE_PARENT_IDS[strategyKey], strategyKey, start, end, rows.length]);
    }

    for (const row of result.rows) {
      const isActive = ['queued', 'running', 'cancel_requested'].includes(row.status);
      const parentId = ACTIVE_PARENT_IDS[row.ingester_key];
      if (isActive && !parentId) {
        throw new Error(`active weather strategy is not registered: ${row.ingester_key}`);
      }
      const status = isActive
        ? 'queued'
        : row.status === 'cancel_requested'
          ? 'cancelled'
          : row.status;
      await queryRunner.query(`
        INSERT INTO ingester.backfill_jobs (
          job_id,parent_job_id,job_kind,strategy_key,strategy_contract_version,
          request_schema_version,canonical_request,request_hash,range_start,
          range_end,shard_key,status,attempt,max_attempts,next_attempt_at,
          progress,checkpoint,summary,last_error_message,requested_at,started_at,
          completed_at,created_at,updated_at,legacy_source,legacy_job_id
        ) VALUES (
          $1::uuid,$2::uuid,$3::text,$4::text,1,1,
          jsonb_build_object('strategy_key',$4::text,'parameters',$5::jsonb,
            'range',jsonb_build_object('start',$6::timestamptz,'end',$7::timestamptz),
            'legacy_idempotency_key',$8::text),
          encode(digest('weather.ingestion_jobs:' || $1::text,'sha256'),'hex'),
          $6,$7,$9::text,$10::text,$11::int,$12::int,
          CASE WHEN $3::text='shard' THEN now() ELSE $13::timestamptz END,
          $14::jsonb,$14::jsonb,$15::jsonb,$16::text,$17,$18,$19,$17,$20,
          'weather.ingestion_jobs',$1::uuid
        ) ON CONFLICT (legacy_source,legacy_job_id) DO NOTHING
      `, [
        row.job_id,
        isActive ? parentId : null,
        isActive ? 'shard' : 'request',
        row.ingester_key,
        JSON.stringify(row.request || {}),
        row.range_start,
        row.range_end,
        row.idempotency_key,
        isActive ? row.job_id : null,
        status,
        row.attempt,
        row.max_attempts,
        row.next_attempt_at,
        JSON.stringify(row.progress || {}),
        JSON.stringify(row.summary || {}),
        row.error,
        row.requested_at,
        row.started_at,
        isActive ? null : row.completed_at,
        row.updated_at,
      ]);
    }
  }

  private async reconcile(queryRunner: QueryRunner, source: any): Promise<void> {
    const sourceCounts = await source.query(`
      SELECT 'source_artifacts' AS table_name,count(*)::bigint AS rows FROM weather.source_artifacts
      UNION ALL SELECT 'goes_abi_features',count(*)::bigint FROM weather.goes_abi_features
      UNION ALL SELECT 'goes_abi_window_coverage',count(*)::bigint FROM weather.goes_abi_window_coverage
      UNION ALL SELECT 'hrrr_environment_features',count(*)::bigint FROM weather.hrrr_environment_features
      UNION ALL SELECT 'hrrr_environment_window_coverage',count(*)::bigint FROM weather.hrrr_environment_window_coverage
      UNION ALL SELECT 'ingestion_jobs',count(*)::bigint FROM weather.ingestion_jobs
    `);
    const targetCounts = await queryRunner.query(`
      SELECT 'source_artifacts' AS table_name,count(*)::bigint AS rows
        FROM ingester.backfill_artifacts WHERE legacy_source='weather.source_artifacts'
      UNION ALL SELECT 'goes_abi_features',count(*)::bigint FROM weather.goes_abi_features
      UNION ALL SELECT 'goes_abi_window_coverage',count(*)::bigint FROM weather.goes_abi_window_coverage
      UNION ALL SELECT 'hrrr_environment_features',count(*)::bigint FROM weather.hrrr_environment_features
      UNION ALL SELECT 'hrrr_environment_window_coverage',count(*)::bigint FROM weather.hrrr_environment_window_coverage
      UNION ALL SELECT 'ingestion_jobs',count(*)::bigint FROM ingester.backfill_jobs
        WHERE legacy_source='weather.ingestion_jobs'
    `);
    const expected = new Map(
      sourceCounts.rows.map((row: any) => [row.table_name, String(row.rows)]),
    );
    for (const row of targetCounts) {
      if (expected.get(row.table_name) !== String(row.rows)) {
        throw new Error(
          `weather migration reconciliation failed for ${row.table_name}: ` +
            `${expected.get(row.table_name)} source rows, ${row.rows} target rows`,
        );
      }
    }
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DROP TABLE IF EXISTS weather.hrrr_environment_window_coverage;
      DROP TABLE IF EXISTS weather.hrrr_environment_features;
      DROP TABLE IF EXISTS weather.goes_abi_window_coverage;
      DROP TABLE IF EXISTS weather.goes_abi_features;
      DELETE FROM ingester.backfill_artifacts
        WHERE legacy_source='weather.source_artifacts';
      DELETE FROM ingester.backfill_jobs
        WHERE legacy_source IN (
          'weather.ingestion_jobs','weather.active_parent','weather.artifact_ledger'
        );
      DROP SCHEMA IF EXISTS weather;
    `);
  }
}

async function copyInBatches(
  source: any,
  target: any,
  schema: string,
  table: string,
): Promise<void> {
  const columnsResult = await source.query(
    `SELECT column_name FROM information_schema.columns
     WHERE table_schema=$1 AND table_name=$2 ORDER BY ordinal_position`,
    [schema, table],
  );
  const columns = columnsResult.rows.map((row: any) => row.column_name);
  if (columns.length === 0) throw new Error(`source table is missing: ${schema}.${table}`);
  const quotedColumns = columns.map(quoteIdentifier).join(',');
  let cursor = '(0,0)';
  const batchSize = 500;
  for (;;) {
    const batch = await source.query(
      `SELECT ctid::text AS __copy_cursor,${quotedColumns}
       FROM ${quoteIdentifier(schema)}.${quoteIdentifier(table)}
       WHERE ctid > $1::tid ORDER BY ctid LIMIT $2`,
      [cursor, batchSize],
    );
    if (batch.rows.length === 0) break;
    const values: unknown[] = [];
    const tuples = batch.rows.map((row: any) => {
      const placeholders = columns.map((column: string) => {
        values.push(row[column]);
        return `$${values.length}`;
      });
      return `(${placeholders.join(',')})`;
    });
    await target.query(
      `INSERT INTO ${quoteIdentifier(schema)}.${quoteIdentifier(table)}
       (${quotedColumns}) VALUES ${tuples.join(',')}`,
      values,
    );
    cursor = batch.rows[batch.rows.length - 1].__copy_cursor;
  }
}

function quoteIdentifier(value: string): string {
  return `"${value.replace(/"/g, '""')}"`;
}
