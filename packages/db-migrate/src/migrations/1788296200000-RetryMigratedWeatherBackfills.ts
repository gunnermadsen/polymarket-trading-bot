import { MigrationInterface, QueryRunner } from 'typeorm';

export class RetryMigratedWeatherBackfills1788296200000
  implements MigrationInterface
{
  name = 'RetryMigratedWeatherBackfills1788296200000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    const result = await queryRunner.query(`
      SELECT count(*)::int AS shards
      FROM ingester.backfill_jobs
      WHERE legacy_source='weather.ingestion_jobs'
        AND job_kind='shard'
        AND strategy_key IN (
          'goes_abi_klga_features',
          'hrrr_environment_features'
        )
    `);
    if (Number(result[0].shards) !== 96) {
      throw new Error(
        `expected 96 migrated weather shards, found ${result[0].shards}`,
      );
    }

    await queryRunner.query(`
      WITH reset AS (
        UPDATE ingester.backfill_jobs
        SET status='queued',
            attempt=0,
            next_attempt_at=now(),
            assigned_worker_id=NULL,
            lease_token=NULL,
            lease_expires_at=NULL,
            heartbeat_at=NULL,
            started_at=NULL,
            completed_at=NULL,
            cancel_requested_at=NULL,
            assigned_worker_image_digest=NULL,
            assigned_worker_source_revision=NULL,
            last_error_kind=NULL,
            last_error_code=NULL,
            last_error_message=NULL,
            updated_at=now()
        WHERE legacy_source='weather.ingestion_jobs'
          AND job_kind='shard'
          AND strategy_key IN (
            'goes_abi_klga_features',
            'hrrr_environment_features'
          )
        RETURNING job_id
      )
      INSERT INTO ingester.backfill_job_events (
        job_id,level,event_code,message,metadata
      )
      SELECT job_id,'info','migrated_weather_retry_scheduled',
        'Migrated weather shard rescheduled after shared runtime correction',
        jsonb_build_object('migration',
          '1788296200000-RetryMigratedWeatherBackfills')
      FROM reset;

      UPDATE ingester.backfill_jobs
      SET status='queued',
          attempt=0,
          next_attempt_at=now(),
          started_at=NULL,
          completed_at=NULL,
          last_error_kind=NULL,
          last_error_code=NULL,
          last_error_message=NULL,
          updated_at=now()
      WHERE legacy_source='weather.active_parent'
        AND strategy_key IN (
          'goes_abi_klga_features',
          'hrrr_environment_features'
        );
    `);
  }

  public async down(): Promise<void> {
    throw new Error('RetryMigratedWeatherBackfills is an irreversible queue repair');
  }
}
