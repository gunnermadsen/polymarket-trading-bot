import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddTemperaturePmxtTradeWindowCoverage1787659200000 implements MigrationInterface {
  name = 'AddTemperaturePmxtTradeWindowCoverage1787659200000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE weather.pmxt_trade_window_coverage (
        process_id uuid NOT NULL
          REFERENCES polymarket.trading_processes (process_id) ON DELETE RESTRICT,
        decision_time timestamptz NOT NULL,
        market_id text NOT NULL
          REFERENCES weather.temperature_markets (market_id) ON DELETE RESTRICT,
        yes_token_id text NOT NULL,
        no_token_id text NOT NULL,
        archive_manifest_sha256 text NOT NULL,
        expected_archive_hours smallint NOT NULL,
        available_archive_hours smallint NOT NULL,
        yes_trade_count integer NOT NULL,
        no_trade_count integer NOT NULL,
        quality_flags jsonb NOT NULL DEFAULT '[]'::jsonb,
        source_artifact_ids uuid[] NOT NULL,
        refreshed_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (process_id, decision_time, market_id),
        CONSTRAINT chk_weather_pmxt_trade_window_manifest CHECK (
          archive_manifest_sha256 ~ '^[0-9a-f]{64}$'
        ),
        CONSTRAINT chk_weather_pmxt_trade_window_archive_hours CHECK (
          expected_archive_hours > 0
          AND available_archive_hours >= 0
          AND available_archive_hours <= expected_archive_hours
        ),
        CONSTRAINT chk_weather_pmxt_trade_window_counts CHECK (
          yes_trade_count >= 0 AND no_trade_count >= 0
        ),
        CONSTRAINT chk_weather_pmxt_trade_window_flags CHECK (
          jsonb_typeof(quality_flags) = 'array'
        )
      );

      CREATE INDEX idx_weather_pmxt_trade_window_decision
        ON weather.pmxt_trade_window_coverage (process_id, decision_time);
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP TABLE weather.pmxt_trade_window_coverage;`);
  }
}
