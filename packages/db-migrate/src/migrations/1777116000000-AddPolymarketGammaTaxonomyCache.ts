import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddPolymarketGammaTaxonomyCache1777116000000 implements MigrationInterface {
  name = 'AddPolymarketGammaTaxonomyCache1777116000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.gamma_market_metadata (
        cache_key text PRIMARY KEY,
        lookup_type text NOT NULL,
        lookup_slug text NOT NULL,
        event_slug text,
        market_slug text,
        gamma_event_id text,
        gamma_market_id text,
        category text,
        series_slug text,
        tag_slugs text[] NOT NULL DEFAULT '{}'::text[],
        sport_key text,
        taxonomy_segment text,
        taxonomy_source text NOT NULL DEFAULT 'gamma',
        taxonomy_confidence numeric(6,4) NOT NULL DEFAULT 0,
        taxonomy_version text NOT NULL,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        fetched_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_poly_gamma_lookup_type CHECK (lookup_type IN ('event_slug', 'market_slug')),
        CONSTRAINT chk_poly_gamma_taxonomy_confidence CHECK (taxonomy_confidence >= 0 AND taxonomy_confidence <= 1)
      );
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.wallet_trades
        ADD COLUMN IF NOT EXISTS taxonomy_segment text,
        ADD COLUMN IF NOT EXISTS taxonomy_source text,
        ADD COLUMN IF NOT EXISTS taxonomy_confidence numeric(6,4),
        ADD COLUMN IF NOT EXISTS taxonomy_version text,
        ADD COLUMN IF NOT EXISTS taxonomy_fetched_at timestamptz,
        ADD COLUMN IF NOT EXISTS taxonomy_metadata jsonb;
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_gamma_market_metadata_event_slug
      ON polymarket.gamma_market_metadata (event_slug)
      WHERE event_slug IS NOT NULL;
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_gamma_market_metadata_market_slug
      ON polymarket.gamma_market_metadata (market_slug)
      WHERE market_slug IS NOT NULL;
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_gamma_market_metadata_taxonomy_segment
      ON polymarket.gamma_market_metadata (taxonomy_segment)
      WHERE taxonomy_segment IS NOT NULL;
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_trades_taxonomy
      ON polymarket.wallet_trades (taxonomy_version, taxonomy_source, taxonomy_segment);
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_trades_taxonomy_unresolved
      ON polymarket.wallet_trades (timestamp_utc DESC)
      WHERE taxonomy_version IS NULL;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_wallet_trades_taxonomy_unresolved;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_wallet_trades_taxonomy;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_gamma_market_metadata_taxonomy_segment;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_gamma_market_metadata_market_slug;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_gamma_market_metadata_event_slug;`);
    await queryRunner.query(`
      ALTER TABLE polymarket.wallet_trades
        DROP COLUMN IF EXISTS taxonomy_metadata,
        DROP COLUMN IF EXISTS taxonomy_fetched_at,
        DROP COLUMN IF EXISTS taxonomy_version,
        DROP COLUMN IF EXISTS taxonomy_confidence,
        DROP COLUMN IF EXISTS taxonomy_source,
        DROP COLUMN IF EXISTS taxonomy_segment;
    `);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.gamma_market_metadata;`);
  }
}
