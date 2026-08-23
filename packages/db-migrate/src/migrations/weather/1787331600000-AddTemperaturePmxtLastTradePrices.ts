import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddTemperaturePmxtLastTradePrices1787331600000 implements MigrationInterface {
  name = 'AddTemperaturePmxtLastTradePrices1787331600000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE weather.pmxt_last_trade_prices (
        process_id uuid NOT NULL
          REFERENCES polymarket.trading_processes (process_id) ON DELETE RESTRICT,
        event_id text NOT NULL,
        market_id text NOT NULL
          REFERENCES weather.temperature_markets (market_id) ON DELETE RESTRICT,
        token_id text NOT NULL,
        outcome text NOT NULL,
        source_timestamp timestamptz NOT NULL,
        provider_received_at timestamptz NOT NULL,
        price numeric(18,8) NOT NULL,
        size numeric(30,10),
        trade_side text,
        source_artifact_id uuid NOT NULL
          REFERENCES weather.source_artifacts (artifact_id) ON DELETE RESTRICT,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (process_id, event_id),
        CONSTRAINT chk_weather_pmxt_last_trade_outcome CHECK (outcome IN ('YES','NO')),
        CONSTRAINT chk_weather_pmxt_last_trade_price CHECK (price > 0 AND price < 1),
        CONSTRAINT chk_weather_pmxt_last_trade_size CHECK (size IS NULL OR size >= 0),
        CONSTRAINT chk_weather_pmxt_last_trade_side CHECK (
          trade_side IS NULL OR trade_side IN ('BUY','SELL')
        )
      );

      CREATE INDEX idx_weather_pmxt_last_trade_market_received
        ON weather.pmxt_last_trade_prices (
          process_id, market_id, provider_received_at DESC, source_timestamp DESC
        );
      CREATE INDEX idx_weather_pmxt_last_trade_artifact
        ON weather.pmxt_last_trade_prices (source_artifact_id);
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP TABLE weather.pmxt_last_trade_prices;`);
  }
}
