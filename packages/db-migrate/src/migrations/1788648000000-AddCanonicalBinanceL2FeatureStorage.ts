import { MigrationInterface, QueryRunner } from 'typeorm';

const products = [
  {
    source: 'polymarket.binance_spot_btcusdt_l2_one_second_features',
    target: 'market_data.binance_spot_btcusdt_l2_one_second_features',
    name: 'binance_spot_btcusdt_l2_one_second_features',
  },
  {
    source: 'polymarket.binance_btcusdt_l2_one_second_features',
    target: 'market_data.binance_futures_btcusdt_l2_one_second_features',
    name: 'binance_futures_btcusdt_l2_one_second_features',
  },
];

export class AddCanonicalBinanceL2FeatureStorage1788648000000
  implements MigrationInterface
{
  name = 'AddCanonicalBinanceL2FeatureStorage1788648000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    for (const product of products) {
      await queryRunner.query(`
        CREATE TABLE ${product.target} (
          LIKE ${product.source}
            INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING STORAGE
        );
        ALTER TABLE ${product.target}
          ADD CONSTRAINT pk_market_data_${product.name}
            PRIMARY KEY (symbol, second_start),
          ADD CONSTRAINT fk_market_data_${product.name}_artifact
            FOREIGN KEY (artifact_id)
            REFERENCES ingester.backfill_artifacts (artifact_id)
            ON DELETE RESTRICT;
        SELECT create_hypertable(
          '${product.target}', 'second_start',
          chunk_time_interval => INTERVAL '1 day',
          create_default_indexes => FALSE,
          if_not_exists => TRUE
        );
        ALTER TABLE ${product.target} SET (
          timescaledb.compress = true,
          timescaledb.compress_orderby = 'second_start ASC',
          timescaledb.compress_segmentby = 'symbol'
        );
        SELECT add_compression_policy(
          '${product.target}', INTERVAL '7 days', if_not_exists => TRUE
        );
        CREATE INDEX idx_market_data_${product.name}_available
          ON ${product.target} (symbol, available_at DESC, second_start DESC);
        CREATE INDEX idx_market_data_${product.name}_artifact
          ON ${product.target} (artifact_id, second_start);
        CREATE TRIGGER trg_reject_market_data_${product.name}_change
          BEFORE UPDATE OR DELETE ON ${product.target}
          FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();
      `);
    }
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    for (const product of [...products].reverse()) {
      await queryRunner.query(`
        DO $$
        BEGIN
          IF EXISTS (SELECT 1 FROM ${product.target} LIMIT 1) THEN
            RAISE EXCEPTION 'refusing to remove populated canonical Binance L2 feature table';
          END IF;
        END $$;
        SELECT remove_compression_policy('${product.target}', if_exists => TRUE);
        DROP TABLE ${product.target};
      `);
    }
  }
}
