import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddBtcPhase6PaperCapital1777123000000 implements MigrationInterface {
  name = 'AddBtcPhase6PaperCapital1777123000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE polymarket.btc_interval_markets
        ADD CONSTRAINT uq_btc_interval_market_official_identity
        UNIQUE (
          market_id, official_outcome, official_winning_token_id,
          official_resolution_received_at, official_resolution_source
        );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.btc_paper_settlement_ledger (
        settlement_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        experiment_id uuid NOT NULL
          REFERENCES polymarket.btc_paper_experiments (experiment_id) ON DELETE RESTRICT,
        process_id uuid NOT NULL
          REFERENCES polymarket.trading_processes (process_id) ON DELETE RESTRICT,
        order_id text NOT NULL
          REFERENCES polymarket.orders (order_id) ON DELETE RESTRICT,
        market_id text NOT NULL
          REFERENCES polymarket.btc_interval_markets (market_id) ON DELETE RESTRICT,
        token_id text NOT NULL,
        fill_ids jsonb NOT NULL,
        official_outcome text NOT NULL,
        official_winning_token_id text NOT NULL,
        official_resolution_received_at timestamptz NOT NULL,
        official_resolution_source text NOT NULL,
        filled_size numeric(30,10) NOT NULL,
        entry_notional numeric(30,10) NOT NULL,
        entry_fees numeric(30,10) NOT NULL,
        payout numeric(30,10) NOT NULL,
        net_pnl numeric(30,10) NOT NULL,
        credit_status text NOT NULL DEFAULT 'pending',
        credited_at timestamptz,
        credit_attempts bigint NOT NULL DEFAULT 0,
        credit_evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT uq_btc_paper_settlement_experiment_order
          UNIQUE (experiment_id, order_id),
        CONSTRAINT fk_btc_paper_settlement_official_identity
          FOREIGN KEY (
            market_id, official_outcome, official_winning_token_id,
            official_resolution_received_at, official_resolution_source
          )
          REFERENCES polymarket.btc_interval_markets (
            market_id, official_outcome, official_winning_token_id,
            official_resolution_received_at, official_resolution_source
          ) ON DELETE RESTRICT,
        CONSTRAINT chk_btc_paper_settlement_outcome CHECK (
          official_outcome IN ('up', 'down')
        ),
        CONSTRAINT chk_btc_paper_settlement_fill_ids CHECK (
          jsonb_typeof(fill_ids) = 'array' AND jsonb_array_length(fill_ids) > 0
        ),
        CONSTRAINT chk_btc_paper_settlement_source CHECK (
          official_resolution_source IN ('clob_websocket', 'clob_rest_reconciliation')
        ),
        CONSTRAINT chk_btc_paper_settlement_amounts CHECK (
          filled_size > 0
          AND entry_notional >= 0
          AND entry_fees >= 0
          AND payout >= 0
          AND payout = CASE
            WHEN token_id = official_winning_token_id THEN filled_size
            ELSE 0
          END
          AND net_pnl = payout - entry_notional - entry_fees
        ),
        CONSTRAINT chk_btc_paper_settlement_credit_status CHECK (
          credit_status IN ('pending', 'credited')
        ),
        CONSTRAINT chk_btc_paper_settlement_credit_state CHECK (
          (credit_status = 'pending' AND credited_at IS NULL AND credit_attempts = 0)
          OR
          (credit_status = 'credited' AND credited_at IS NOT NULL AND credit_attempts >= 1)
        ),
        CONSTRAINT chk_btc_paper_settlement_evidence CHECK (
          jsonb_typeof(credit_evidence) = 'object'
        )
      );
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_btc_paper_settlement_pending
        ON polymarket.btc_paper_settlement_ledger (
          experiment_id, official_resolution_received_at, settlement_id
        )
        WHERE credit_status = 'pending';
      CREATE INDEX IF NOT EXISTS idx_btc_paper_settlement_experiment_credited
        ON polymarket.btc_paper_settlement_ledger (
          experiment_id, credited_at DESC, settlement_id
        )
        WHERE credit_status = 'credited';
      CREATE INDEX IF NOT EXISTS idx_btc_features_process_window_asof
        ON polymarket.btc_feature_snapshots (
          (features->>'process_id'), window_start, feature_as_of DESC
        );
      CREATE INDEX IF NOT EXISTS idx_btc_decisions_experiment_process_at
        ON polymarket.btc_strategy_decisions (
          experiment_id, process_id, decision_at DESC, snapshot_id
        );
      CREATE INDEX IF NOT EXISTS idx_book_checkpoints_token_received_source
        ON polymarket.orderbook_checkpoints (
          token_id, received_at DESC, source_timestamp DESC
        );
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        PERFORM remove_compression_policy(
          'polymarket.reference_price_ticks', if_exists => true
        );
        PERFORM remove_compression_policy(
          'polymarket.btc_feature_snapshots', if_exists => true
        );
        PERFORM remove_compression_policy(
          'polymarket.ml_feature_vectors', if_exists => true
        );
        PERFORM remove_compression_policy(
          'polymarket.ml_shadow_predictions', if_exists => true
        );

        PERFORM add_compression_policy(
          'polymarket.reference_price_ticks', INTERVAL '1 day', if_not_exists => true
        );
        PERFORM add_compression_policy(
          'polymarket.btc_feature_snapshots', INTERVAL '1 day', if_not_exists => true
        );
        PERFORM add_compression_policy(
          'polymarket.ml_feature_vectors', INTERVAL '1 day', if_not_exists => true
        );
        PERFORM add_compression_policy(
          'polymarket.ml_shadow_predictions', INTERVAL '1 day', if_not_exists => true
        );
      END $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        PERFORM remove_compression_policy(
          'polymarket.reference_price_ticks', if_exists => true
        );
        PERFORM remove_compression_policy(
          'polymarket.btc_feature_snapshots', if_exists => true
        );
        PERFORM remove_compression_policy(
          'polymarket.ml_feature_vectors', if_exists => true
        );
        PERFORM remove_compression_policy(
          'polymarket.ml_shadow_predictions', if_exists => true
        );

        PERFORM add_compression_policy(
          'polymarket.reference_price_ticks', INTERVAL '7 days', if_not_exists => true
        );
        PERFORM add_compression_policy(
          'polymarket.btc_feature_snapshots', INTERVAL '7 days', if_not_exists => true
        );
        PERFORM add_compression_policy(
          'polymarket.ml_feature_vectors', INTERVAL '7 days', if_not_exists => true
        );
        PERFORM add_compression_policy(
          'polymarket.ml_shadow_predictions', INTERVAL '7 days', if_not_exists => true
        );
      END $$;
    `);

    await queryRunner.query(`
      DROP INDEX IF EXISTS polymarket.idx_book_checkpoints_token_received_source;
      DROP INDEX IF EXISTS polymarket.idx_btc_decisions_experiment_process_at;
      DROP INDEX IF EXISTS polymarket.idx_btc_features_process_window_asof;
      DROP TABLE IF EXISTS polymarket.btc_paper_settlement_ledger;
      ALTER TABLE polymarket.btc_interval_markets
        DROP CONSTRAINT IF EXISTS uq_btc_interval_market_official_identity;
    `);
  }
}
