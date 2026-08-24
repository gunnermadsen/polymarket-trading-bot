import { MigrationInterface, QueryRunner } from 'typeorm';

const TABLE_NAME = 'market_data.pmdata_chainlink_btcusd_twap';

export class AddPmdataChainlinkBtcusdTwap1787594400000
  implements MigrationInterface
{
  name = 'AddPmdataChainlinkBtcusdTwap1787594400000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE ${TABLE_NAME} (
        source_timestamp timestamptz NOT NULL,
        provider_received_at timestamptz NOT NULL,
        valid_from_timestamp timestamptz NOT NULL,
        expires_at timestamptz NOT NULL,
        symbol text NOT NULL DEFAULT 'BTCUSD',
        window_seconds smallint NOT NULL,
        twap_price numeric(38,18) NOT NULL,
        full_accuracy_value text NOT NULL,
        report_version text NOT NULL,
        source_date date NOT NULL,
        archive_row_number bigint NOT NULL,
        artifact_id uuid NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_market_data_pmdata_chainlink_btcusd_twap
          PRIMARY KEY (source_timestamp, window_seconds),
        CONSTRAINT uq_market_data_pmdata_chainlink_btcusd_twap_archive_row
          UNIQUE (artifact_id, source_timestamp, archive_row_number),
        CONSTRAINT fk_market_data_pmdata_chainlink_btcusd_twap_artifact
          FOREIGN KEY (artifact_id)
          REFERENCES polymarket.backfill_artifacts (artifact_id)
          ON DELETE RESTRICT,
        CONSTRAINT chk_market_data_pmdata_chainlink_btcusd_twap_identity CHECK (
          symbol = 'BTCUSD'
          AND window_seconds IN (30, 60)
          AND archive_row_number >= 0
        ),
        CONSTRAINT chk_market_data_pmdata_chainlink_btcusd_twap_value CHECK (
          full_accuracy_value ~ '^[0-9]{1,29}$'
          AND twap_price * 1000000000000000000::numeric = full_accuracy_value::numeric
          AND twap_price > 0
        ),
        CONSTRAINT chk_market_data_pmdata_chainlink_btcusd_twap_time CHECK (
          valid_from_timestamp <= source_timestamp
          AND expires_at > source_timestamp
          AND source_timestamp >= source_date::timestamptz
          AND source_timestamp < source_date::timestamptz + INTERVAL '1 day'
        ),
        CONSTRAINT chk_market_data_pmdata_chainlink_btcusd_twap_version CHECK (
          length(btrim(report_version)) > 0
        )
      );

      SELECT create_hypertable(
        '${TABLE_NAME}', 'source_timestamp',
        chunk_time_interval => INTERVAL '1 day',
        create_default_indexes => FALSE, if_not_exists => TRUE
      );

      CREATE INDEX idx_market_data_pmdata_chainlink_btcusd_twap_latest
        ON ${TABLE_NAME} (window_seconds, source_timestamp DESC);
      CREATE INDEX idx_market_data_pmdata_chainlink_btcusd_twap_artifact
        ON ${TABLE_NAME} (artifact_id, archive_row_number);

      CREATE TRIGGER trg_reject_md_pmdata_chainlink_btcusd_twap_change
        BEFORE UPDATE OR DELETE ON ${TABLE_NAME}
        FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (SELECT 1 FROM ${TABLE_NAME} LIMIT 1) THEN
          RAISE EXCEPTION
            'refusing to remove PMData Chainlink BTC/USD TWAP while facts exist';
        END IF;
      END $$;
      DROP TRIGGER trg_reject_md_pmdata_chainlink_btcusd_twap_change
        ON ${TABLE_NAME};
      DROP TABLE ${TABLE_NAME};
    `);
  }
}
