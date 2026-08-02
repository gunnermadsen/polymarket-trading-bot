import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddChainlinkCandlesAndBinanceOpenInterest1785612000000
  implements MigrationInterface
{
  name = 'AddChainlinkCandlesAndBinanceOpenInterest1785612000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE polymarket.chainlink_btcusd_one_minute_candles (
        symbol text NOT NULL,
        open_timestamp timestamptz NOT NULL,
        close_timestamp timestamptz NOT NULL,
        open_price numeric(38,18) NOT NULL,
        high_price numeric(38,18) NOT NULL,
        low_price numeric(38,18) NOT NULL,
        close_price numeric(38,18) NOT NULL,
        volume numeric(38,18),
        volume_supported boolean NOT NULL,
        artifact_id uuid NOT NULL
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        ingested_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_chainlink_btcusd_one_minute_candles
          PRIMARY KEY (symbol, open_timestamp),
        CONSTRAINT chk_chainlink_btcusd_one_minute_symbol
          CHECK (symbol = 'BTCUSD'),
        CONSTRAINT chk_chainlink_btcusd_one_minute_time CHECK (
          close_timestamp = open_timestamp + INTERVAL '1 minute'
          AND extract(epoch FROM open_timestamp)::bigint % 60 = 0
        ),
        CONSTRAINT chk_chainlink_btcusd_one_minute_prices CHECK (
          open_price > 0 AND high_price > 0 AND low_price > 0 AND close_price > 0
          AND high_price >= open_price AND high_price >= close_price AND high_price >= low_price
          AND low_price <= open_price AND low_price <= close_price
        ),
        CONSTRAINT chk_chainlink_btcusd_one_minute_volume CHECK (
          (volume_supported AND volume IS NOT NULL AND volume >= 0)
          OR (NOT volume_supported AND volume IS NULL)
        )
      );

      SELECT create_hypertable(
        'polymarket.chainlink_btcusd_one_minute_candles',
        'open_timestamp',
        chunk_time_interval => INTERVAL '1 day',
        if_not_exists => TRUE
      );

      ALTER TABLE polymarket.chainlink_btcusd_one_minute_candles SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'open_timestamp ASC',
        timescaledb.compress_segmentby = 'symbol, artifact_id'
      );

      CREATE INDEX idx_chainlink_btcusd_one_minute_candles_artifact
        ON polymarket.chainlink_btcusd_one_minute_candles (artifact_id, open_timestamp);

      CREATE TRIGGER trg_reject_chainlink_btcusd_one_minute_candle_change
        BEFORE UPDATE OR DELETE ON polymarket.chainlink_btcusd_one_minute_candles
        FOR EACH ROW
        EXECUTE FUNCTION polymarket.reject_historical_market_event_change();
    `);

    await queryRunner.query(`
      CREATE TABLE polymarket.binance_btcusdt_five_minute_open_interest (
        symbol text NOT NULL,
        source_timestamp timestamptz NOT NULL,
        period_seconds integer NOT NULL,
        sum_open_interest numeric(38,18) NOT NULL,
        sum_open_interest_value numeric(38,18) NOT NULL,
        cmc_circulating_supply numeric(38,18),
        artifact_id uuid NOT NULL
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        ingested_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_binance_btcusdt_five_minute_open_interest
          PRIMARY KEY (symbol, source_timestamp),
        CONSTRAINT chk_binance_btcusdt_five_minute_oi_symbol
          CHECK (symbol = 'BTCUSDT'),
        CONSTRAINT chk_binance_btcusdt_five_minute_oi_time CHECK (
          period_seconds = 300
          AND extract(epoch FROM source_timestamp)::bigint % 300 = 0
        ),
        CONSTRAINT chk_binance_btcusdt_five_minute_oi_values CHECK (
          sum_open_interest >= 0
          AND sum_open_interest_value >= 0
          AND (cmc_circulating_supply IS NULL OR cmc_circulating_supply >= 0)
        )
      );

      SELECT create_hypertable(
        'polymarket.binance_btcusdt_five_minute_open_interest',
        'source_timestamp',
        chunk_time_interval => INTERVAL '1 day',
        if_not_exists => TRUE
      );

      ALTER TABLE polymarket.binance_btcusdt_five_minute_open_interest SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'source_timestamp ASC',
        timescaledb.compress_segmentby = 'symbol, artifact_id'
      );

      CREATE INDEX idx_binance_btcusdt_five_minute_oi_artifact
        ON polymarket.binance_btcusdt_five_minute_open_interest (
          artifact_id, source_timestamp
        );

      CREATE TRIGGER trg_reject_binance_btcusdt_five_minute_oi_change
        BEFORE UPDATE OR DELETE ON polymarket.binance_btcusdt_five_minute_open_interest
        FOR EACH ROW
        EXECUTE FUNCTION polymarket.reject_historical_market_event_change();
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1 FROM polymarket.chainlink_btcusd_one_minute_candles LIMIT 1
        ) OR EXISTS (
          SELECT 1 FROM polymarket.binance_btcusdt_five_minute_open_interest LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'refusing to remove Chainlink candle or Binance open-interest data while rows exist';
        END IF;
      END $$;

      DROP TRIGGER trg_reject_binance_btcusdt_five_minute_oi_change
        ON polymarket.binance_btcusdt_five_minute_open_interest;
      DROP TABLE polymarket.binance_btcusdt_five_minute_open_interest;

      DROP TRIGGER trg_reject_chainlink_btcusd_one_minute_candle_change
        ON polymarket.chainlink_btcusd_one_minute_candles;
      DROP TABLE polymarket.chainlink_btcusd_one_minute_candles;
    `);
  }
}
