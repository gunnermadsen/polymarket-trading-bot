import { MigrationInterface, QueryRunner } from 'typeorm';

const { Client } = require('pg');

export class CompleteBackfillLedgerConsolidation1788295900000
  implements MigrationInterface
{
  name = 'CompleteBackfillLedgerConsolidation1788295900000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    const archive = new Client({
      host: process.env.KRAKEN_ARCHIVE_POSTGRES_HOST || 'kraken-timescaledb',
      port: +(process.env.KRAKEN_ARCHIVE_POSTGRES_PORT || 5432),
      user: process.env.POSTGRES_USER || 'postgres',
      password: requiredEnv('POSTGRES_PASSWORD'),
      database: process.env.KRAKEN_ARCHIVE_POSTGRES_DB || 'kraken_archive',
      application_name: 'unified-ingester-lineage-import',
    });
    await archive.connect();
    try {
      await this.importFinancialLineage(queryRunner, archive);
      await this.importKrakenParquetLineage(queryRunner, archive);
    } finally {
      await archive.end();
    }

    const legacy = await queryRunner.query(`
      SELECT
        (SELECT count(*)::bigint FROM polymarket.backfill_jobs) AS jobs,
        (SELECT count(*)::bigint FROM polymarket.backfill_job_events) AS events,
        (SELECT count(*)::bigint FROM polymarket.backfill_artifacts) AS artifacts,
        (SELECT count(*)::bigint FROM ingester.backfill_jobs
          WHERE legacy_source='polymarket.backfill_jobs') AS imported_jobs,
        (SELECT count(*)::bigint FROM ingester.backfill_job_events
          WHERE metadata->>'legacy_source'='polymarket.backfill_job_events') AS imported_events,
        (SELECT count(*)::bigint FROM ingester.backfill_artifacts
          WHERE legacy_source='polymarket.backfill_artifacts') AS imported_artifacts
    `);
    const counts = legacy[0];
    for (const kind of ['jobs', 'events', 'artifacts']) {
      if (String(counts[kind]) !== String(counts[`imported_${kind}`])) {
        throw new Error(
          `polymarket ${kind} ledger reconciliation failed: ` +
            `${counts[kind]} legacy, ${counts[`imported_${kind}`]} unified`,
        );
      }
    }
    await this.consolidateArtifactTableInPlace(queryRunner);
    await queryRunner.query(`
      DROP TABLE polymarket.backfill_job_events;
      DROP TABLE polymarket.backfill_jobs;
    `);
  }

  private async consolidateArtifactTableInPlace(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ingester.backfill_artifacts RENAME TO backfill_artifacts_imported;
      DROP TRIGGER trg_reject_completed_backfill_artifact_change
        ON polymarket.backfill_artifacts;
      ALTER TABLE polymarket.backfill_artifacts
        DROP CONSTRAINT backfill_artifacts_job_id_fkey,
        DROP CONSTRAINT chk_poly_backfill_artifacts_actual_checksum,
        DROP CONSTRAINT chk_poly_backfill_artifacts_completion,
        DROP CONSTRAINT chk_poly_backfill_artifacts_counts,
        DROP CONSTRAINT chk_poly_backfill_artifacts_expected_checksum,
        DROP CONSTRAINT chk_poly_backfill_artifacts_identity,
        DROP CONSTRAINT chk_poly_backfill_artifacts_source_range,
        DROP CONSTRAINT chk_poly_backfill_artifacts_times,
        DROP CONSTRAINT uq_poly_backfill_artifacts_logical_source,
        ADD COLUMN strategy_key text,
        ADD COLUMN checksum text,
        ADD COLUMN byte_size bigint,
        ADD COLUMN durable_target text,
        ADD COLUMN legacy_source text,
        ADD COLUMN legacy_artifact_id uuid;

      UPDATE polymarket.backfill_artifacts target SET
        strategy_key=source.strategy_key,
        checksum=source.checksum,
        byte_size=source.byte_size,
        durable_target=source.durable_target,
        legacy_source=source.legacy_source,
        legacy_artifact_id=source.legacy_artifact_id,
        metadata=source.metadata,
        completed_at=source.completed_at
      FROM ingester.backfill_artifacts_imported source
      WHERE source.artifact_id=target.artifact_id;

      INSERT INTO polymarket.backfill_artifacts (
        artifact_id,job_id,ingester_key,logical_key,provider,source_uri,
        checksum_algorithm,expected_checksum,actual_checksum,compressed_bytes,
        record_count,minimum_source_timestamp,maximum_source_timestamp,status,
        metadata,created_at,updated_at,completed_at,strategy_key,checksum,
        byte_size,durable_target,legacy_source,legacy_artifact_id
      )
      SELECT artifact_id,job_id,strategy_key,logical_key,provider,source_uri,
        checksum_algorithm,checksum,checksum,byte_size,record_count,
        minimum_source_timestamp,maximum_source_timestamp,status,metadata,
        created_at,COALESCE(completed_at,created_at),completed_at,strategy_key,
        checksum,byte_size,durable_target,legacy_source,legacy_artifact_id
      FROM ingester.backfill_artifacts_imported source
      WHERE NOT EXISTS (
        SELECT 1 FROM polymarket.backfill_artifacts target
        WHERE target.artifact_id=source.artifact_id
      );
    `);

    const constraints = await queryRunner.query(`
      SELECT namespace.nspname AS table_schema, relation.relname AS table_name,
        constraint_record.conname,
        array_to_json(ARRAY(
          SELECT attribute.attname
          FROM unnest(constraint_record.conkey) WITH ORDINALITY key(attnum,ordinality)
          JOIN pg_attribute attribute
            ON attribute.attrelid=constraint_record.conrelid
           AND attribute.attnum=key.attnum
          ORDER BY key.ordinality
        )) AS columns,
        constraint_record.confdeltype
      FROM pg_constraint constraint_record
      JOIN pg_class relation ON relation.oid=constraint_record.conrelid
      JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace
      WHERE constraint_record.contype='f'
        AND constraint_record.confrelid='ingester.backfill_artifacts_imported'::regclass
        AND namespace.nspname <> '_timescaledb_internal'
      ORDER BY namespace.nspname,relation.relname,constraint_record.conname
    `);
    for (const constraint of constraints) {
      const table = `${quoteIdentifier(constraint.table_schema)}.${quoteIdentifier(constraint.table_name)}`;
      const name = quoteIdentifier(constraint.conname);
      const columns = constraint.columns.map(quoteIdentifier).join(',');
      const deleteAction = deleteActionSql(constraint.confdeltype);
      await queryRunner.query(`
        ALTER TABLE ${table} DROP CONSTRAINT ${name};
        ALTER TABLE ${table} ADD CONSTRAINT ${name}
          FOREIGN KEY (${columns}) REFERENCES polymarket.backfill_artifacts (artifact_id)
          ON DELETE ${deleteAction} NOT VALID;
        ALTER TABLE ${table} VALIDATE CONSTRAINT ${name};
      `);
    }
    await queryRunner.query(`
      DROP TABLE ingester.backfill_artifacts_imported;
      ALTER TABLE polymarket.backfill_artifacts SET SCHEMA ingester;
      ALTER TABLE ingester.backfill_artifacts
        ALTER COLUMN ingester_key DROP NOT NULL,
        ALTER COLUMN strategy_key SET NOT NULL,
        ADD CONSTRAINT backfill_artifacts_job_id_fkey
          FOREIGN KEY (job_id) REFERENCES ingester.backfill_jobs (job_id) ON DELETE RESTRICT,
        ADD CONSTRAINT uq_ingester_backfill_artifact_logical
          UNIQUE (strategy_key,logical_key),
        ADD CONSTRAINT uq_ingester_backfill_artifact_legacy
          UNIQUE (legacy_source,legacy_artifact_id),
        ADD CONSTRAINT chk_ingester_backfill_artifact_checksum
          CHECK (checksum_algorithm='sha256' AND (checksum IS NULL OR checksum ~ '^[0-9a-f]{64}$')),
        ADD CONSTRAINT chk_ingester_backfill_artifact_counts
          CHECK ((byte_size IS NULL OR byte_size >= 0) AND (record_count IS NULL OR record_count >= 0));
      CREATE INDEX idx_ingester_backfill_artifacts_job
        ON ingester.backfill_artifacts (job_id,created_at,artifact_id);
      CREATE OR REPLACE FUNCTION ingester.reject_completed_backfill_artifact_change()
      RETURNS trigger LANGUAGE plpgsql AS $function$
      BEGIN
        IF TG_OP='DELETE' THEN
          IF OLD.status='completed' THEN
            RAISE EXCEPTION 'completed backfill artifact % is immutable', OLD.artifact_id
              USING ERRCODE='integrity_constraint_violation';
          END IF;
          RETURN OLD;
        END IF;
        IF OLD.status='completed' AND NEW IS DISTINCT FROM OLD THEN
          IF OLD.provider='pmxt_v2_capacity_execution_snapshots_v2'
            AND COALESCE((OLD.metadata->>'source_events_consumed')::bigint,0)=0
            AND NEW.record_count=0
            AND NEW.metadata->>'superseded_reason'='reconstructed_from_local_canonical_orderbooks'
            AND NEW.metadata->>'superseded_by_artifact_id' IS NOT NULL
            AND NEW.artifact_id=OLD.artifact_id
            AND NEW.job_id=OLD.job_id
            AND NEW.strategy_key=OLD.strategy_key
            AND NEW.logical_key=OLD.logical_key
            AND NEW.provider=OLD.provider
            AND NEW.source_uri=OLD.source_uri
            AND NEW.status=OLD.status
            AND NEW.checksum IS NOT DISTINCT FROM OLD.checksum
            AND NEW.minimum_source_timestamp IS NOT DISTINCT FROM OLD.minimum_source_timestamp
            AND NEW.maximum_source_timestamp IS NOT DISTINCT FROM OLD.maximum_source_timestamp
          THEN
            RETURN NEW;
          END IF;
          RAISE EXCEPTION 'completed backfill artifact % is immutable', OLD.artifact_id
            USING ERRCODE='integrity_constraint_violation';
        END IF;
        RETURN NEW;
      END;
      $function$;
      CREATE TRIGGER trg_reject_completed_backfill_artifact_change
        BEFORE DELETE OR UPDATE ON ingester.backfill_artifacts
        FOR EACH ROW EXECUTE FUNCTION ingester.reject_completed_backfill_artifact_change();
    `);
  }

  private async importFinancialLineage(queryRunner: QueryRunner, archive: any): Promise<void> {
    const sourceArtifacts = await archive.query(`
      SELECT artifact_id,job_id,source_url,relative_path,sha256,byte_size,fetched_at
      FROM financial_data.source_artifacts ORDER BY artifact_id LIMIT 10000
    `);
    for (const row of sourceArtifacts.rows) {
      await queryRunner.query(`
        INSERT INTO ingester.backfill_artifacts (
          artifact_id,job_id,strategy_key,logical_key,provider,source_uri,
          checksum,byte_size,status,metadata,created_at,completed_at,
          legacy_source,legacy_artifact_id
        )
        SELECT $1::uuid,job_id,strategy_key,$2::text,'financial_data',$3::text,
          $4::text,$5::bigint,
          'completed',jsonb_build_object('relative_path',$2::text,
            'legacy_source','financial_data.source_artifacts'),
          $6::timestamptz,$6::timestamptz,'financial_data.source_artifacts',$1::uuid
        FROM ingester.backfill_jobs
        WHERE legacy_source='financial_data.backfill_jobs' AND legacy_job_id=$7::uuid
        ON CONFLICT DO NOTHING
      `, [
        row.artifact_id,
        row.relative_path,
        row.source_url,
        row.sha256,
        row.byte_size,
        row.fetched_at,
        row.job_id,
      ]);
    }
    const parquet = await archive.query(`
      SELECT object_id,job_id,provider,dataset,series_id,relative_path,sha256,
        byte_size,row_count,minimum_event_at,maximum_event_at,created_at
      FROM financial_data.parquet_objects ORDER BY object_id LIMIT 10000
    `);
    for (const row of parquet.rows) {
      await queryRunner.query(`
        INSERT INTO ingester.backfill_artifacts (
          artifact_id,job_id,strategy_key,logical_key,provider,source_uri,
          checksum,byte_size,record_count,minimum_source_timestamp,
          maximum_source_timestamp,durable_target,status,metadata,created_at,
          completed_at,legacy_source,legacy_artifact_id
        )
        SELECT $1::uuid,job_id,strategy_key,$2::text,$3::text,$2::text,$4::text,
          $5::bigint,$6::bigint,$7::timestamptz,$8::timestamptz,$2::text,
          'completed',jsonb_build_object('dataset',$9::text,'series_id',$10::text,
            'legacy_source','financial_data.parquet_objects'),
          $11::timestamptz,$11::timestamptz,'financial_data.parquet_objects',$1::uuid
        FROM ingester.backfill_jobs
        WHERE legacy_source='financial_data.backfill_jobs' AND legacy_job_id=$12::uuid
        ON CONFLICT DO NOTHING
      `, [
        row.object_id,
        row.relative_path,
        row.provider,
        row.sha256,
        row.byte_size,
        row.row_count,
        row.minimum_event_at,
        row.maximum_event_at,
        row.dataset,
        row.series_id,
        row.created_at,
        row.job_id,
      ]);
    }
    await assertCount(queryRunner, archive, 'financial_data.source_artifacts');
    await assertCount(queryRunner, archive, 'financial_data.parquet_objects');
  }

  private async importKrakenParquetLineage(queryRunner: QueryRunner, archive: any): Promise<void> {
    const result = await archive.query(`
      SELECT object_id,artifact_id,dataset,symbol,interval_seconds,
        lake_relative_path,sha256,byte_size,row_count,schema_version,published_at
      FROM kraken.parquet_objects ORDER BY object_id LIMIT 10000
    `);
    for (const row of result.rows) {
      await queryRunner.query(`
        UPDATE ingester.backfill_artifacts SET
          checksum=$3::text,byte_size=$4::bigint,record_count=$5::bigint,
          durable_target=$2::text,
          metadata=metadata || jsonb_build_object('dataset',$6::text,'symbol',$7::text,
            'interval_seconds',$8::int,'schema_version',$9::int,
            'source_artifact_id',$10::text,
            'legacy_parquet_object_id',$1::text,
            'legacy_parquet_source','kraken.parquet_objects')
        WHERE legacy_source='kraken.backfill_artifacts'
          AND legacy_artifact_id=$10::uuid
      `, [
        row.object_id,
        row.lake_relative_path,
        row.sha256,
        row.byte_size,
        row.row_count,
        row.dataset,
        row.symbol,
        row.interval_seconds,
        row.schema_version,
        row.artifact_id,
      ]);
    }
    const target = await queryRunner.query(`
      SELECT count(*)::bigint AS rows FROM ingester.backfill_artifacts
      WHERE metadata->>'legacy_parquet_source'='kraken.parquet_objects'
    `);
    if (String(result.rowCount) !== String(target[0].rows)) {
      throw new Error(
        `kraken.parquet_objects reconciliation failed: ${result.rowCount} source rows, ` +
          `${target[0].rows} unified rows`,
      );
    }
  }

  public async down(): Promise<void> {
    throw new Error('legacy backfill ledgers cannot be recreated after verified consolidation');
  }
}

async function assertCount(
  queryRunner: QueryRunner,
  archive: any,
  sourceTable: string,
): Promise<void> {
  const source = await archive.query(`SELECT count(*)::bigint AS rows FROM ${sourceTable}`);
  const target = await queryRunner.query(
    `SELECT count(*)::bigint AS rows FROM ingester.backfill_artifacts WHERE legacy_source=$1`,
    [sourceTable],
  );
  if (String(source.rows[0].rows) !== String(target[0].rows)) {
    throw new Error(
      `${sourceTable} reconciliation failed: ${source.rows[0].rows} source rows, ` +
        `${target[0].rows} unified rows`,
    );
  }
}

function requiredEnv(key: string): string {
  const value = process.env[key];
  if (!value || value.trim() === '') throw new Error(`${key} is required`);
  return value;
}

function quoteIdentifier(value: string): string {
  return `"${value.replace(/"/g, '""')}"`;
}

function deleteActionSql(value: string): string {
  if (value === 'c') return 'CASCADE';
  if (value === 'n') return 'SET NULL';
  if (value === 'd') return 'SET DEFAULT';
  return 'RESTRICT';
}
