import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddBtcRealtimePaperAndMlShadow1777120000000 implements MigrationInterface {
  name = 'AddBtcRealtimePaperAndMlShadow1777120000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.btc_interval_markets (
        market_id text PRIMARY KEY,
        event_id text NOT NULL,
        event_slug text NOT NULL UNIQUE,
        question text NOT NULL,
        series_slug text NOT NULL,
        window_start timestamptz NOT NULL UNIQUE,
        window_end timestamptz NOT NULL,
        condition_id text NOT NULL,
        up_token_id text NOT NULL,
        down_token_id text NOT NULL,
        resolution_source text NOT NULL,
        accepting_orders boolean NOT NULL DEFAULT false,
        active boolean NOT NULL DEFAULT false,
        closed boolean NOT NULL DEFAULT false,
        min_tick_size numeric(18,8) NOT NULL,
        min_order_size numeric(30,10) NOT NULL,
        fee_rate numeric(18,8),
        fee_exponent integer,
        fee_taker_only boolean,
        validation_status text NOT NULL,
        validation_errors jsonb NOT NULL DEFAULT '[]'::jsonb,
        reference_price numeric(30,10),
        reference_source_timestamp timestamptz,
        resolution_price numeric(30,10),
        resolution_source_timestamp timestamptz,
        resolved_outcome text,
        discovered_at timestamptz NOT NULL,
        last_refreshed_at timestamptz NOT NULL,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_btc_interval_window CHECK (window_end = window_start + interval '5 minutes'),
        CONSTRAINT chk_btc_interval_tokens CHECK (up_token_id <> down_token_id),
        CONSTRAINT chk_btc_interval_validation CHECK (validation_status IN ('valid','invalid','ineligible')),
        CONSTRAINT chk_btc_interval_outcome CHECK (resolved_outcome IS NULL OR resolved_outcome IN ('up','down'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.reference_price_ticks (
        tick_id uuid NOT NULL,
        source_timestamp timestamptz NOT NULL,
        received_at timestamptz NOT NULL,
        persisted_at timestamptz NOT NULL DEFAULT now(),
        source text NOT NULL,
        symbol text NOT NULL,
        price numeric(30,10) NOT NULL,
        envelope_timestamp timestamptz,
        connection_id uuid NOT NULL,
        ingest_sequence bigint NOT NULL,
        source_event_id text,
        dedup_key text NOT NULL,
        clock_skew_ms bigint NOT NULL,
        integrity_status text NOT NULL,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT pk_reference_price_ticks PRIMARY KEY (tick_id, source_timestamp),
        CONSTRAINT chk_reference_price_source CHECK (source IN ('direct_binance','rtds_binance','rtds_chainlink')),
        CONSTRAINT chk_reference_price_positive CHECK (price > 0)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.market_feed_events (
        feed_event_id uuid NOT NULL,
        source_timestamp timestamptz NOT NULL,
        received_at timestamptz NOT NULL,
        persisted_at timestamptz NOT NULL DEFAULT now(),
        connection_id uuid NOT NULL,
        ingest_sequence bigint NOT NULL,
        market_id text,
        token_id text,
        event_type text NOT NULL,
        source_hash text,
        applied boolean NOT NULL DEFAULT false,
        integrity_status text NOT NULL,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT pk_market_feed_events PRIMARY KEY (feed_event_id, source_timestamp)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.orderbook_checkpoints (
        checkpoint_id uuid NOT NULL,
        source_timestamp timestamptz NOT NULL,
        received_at timestamptz NOT NULL,
        persisted_at timestamptz NOT NULL DEFAULT now(),
        connection_id uuid NOT NULL,
        ingest_sequence bigint NOT NULL,
        market_id text NOT NULL,
        token_id text NOT NULL,
        best_bid numeric(18,8),
        best_ask numeric(18,8),
        spread numeric(18,8),
        tick_size numeric(18,8) NOT NULL,
        depth_bid numeric(30,10) NOT NULL DEFAULT 0,
        depth_ask numeric(30,10) NOT NULL DEFAULT 0,
        book jsonb NOT NULL,
        source_hash text,
        bootstrap_source text NOT NULL,
        integrity_status text NOT NULL,
        CONSTRAINT pk_orderbook_checkpoints PRIMARY KEY (checkpoint_id, source_timestamp)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.feed_sessions (
        connection_id uuid PRIMARY KEY,
        feed_name text NOT NULL,
        endpoint text NOT NULL,
        reconnect_ordinal integer NOT NULL,
        started_at timestamptz NOT NULL,
        connected_at timestamptz,
        disconnected_at timestamptz,
        messages_received bigint NOT NULL DEFAULT 0,
        messages_persisted bigint NOT NULL DEFAULT 0,
        decode_errors bigint NOT NULL DEFAULT 0,
        integrity_gaps bigint NOT NULL DEFAULT 0,
        dropped_messages bigint NOT NULL DEFAULT 0,
        disconnect_reason text,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now()
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.btc_feature_snapshots (
        snapshot_id uuid NOT NULL,
        feature_as_of timestamptz NOT NULL,
        received_at timestamptz NOT NULL,
        market_id text NOT NULL,
        window_start timestamptz NOT NULL,
        window_end timestamptz NOT NULL,
        feature_schema_version text NOT NULL,
        feature_hash text NOT NULL,
        chainlink_price numeric(30,10) NOT NULL,
        chainlink_open_price numeric(30,10) NOT NULL,
        binance_price numeric(30,10) NOT NULL,
        seconds_to_close numeric(18,6) NOT NULL,
        chainlink_gap_bps numeric(30,10) NOT NULL,
        binance_return_1s_bps numeric(30,10) NOT NULL,
        binance_return_5s_bps numeric(30,10) NOT NULL,
        binance_return_30s_bps numeric(30,10) NOT NULL,
        realized_vol_30s_bps numeric(30,10) NOT NULL,
        basis_bps numeric(30,10) NOT NULL,
        up_best_bid numeric(18,8),
        up_best_ask numeric(18,8),
        down_best_bid numeric(18,8),
        down_best_ask numeric(18,8),
        up_depth_ask numeric(30,10) NOT NULL DEFAULT 0,
        down_depth_ask numeric(30,10) NOT NULL DEFAULT 0,
        up_imbalance numeric(18,8),
        down_imbalance numeric(18,8),
        chainlink_age_ms bigint NOT NULL,
        binance_age_ms bigint NOT NULL,
        book_age_ms bigint NOT NULL,
        source_skew_ms bigint NOT NULL,
        fair_up_probability numeric(18,10),
        fair_up_lower numeric(18,10),
        fair_up_upper numeric(18,10),
        deterministic_logit numeric(30,10),
        readiness_status text NOT NULL,
        quality_flags jsonb NOT NULL DEFAULT '[]'::jsonb,
        features jsonb NOT NULL,
        lineage jsonb NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_btc_feature_snapshots PRIMARY KEY (snapshot_id, feature_as_of)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.btc_strategy_decisions (
        decision_id uuid NOT NULL,
        decision_at timestamptz NOT NULL,
        experiment_id uuid,
        process_id uuid,
        market_id text NOT NULL,
        snapshot_id uuid NOT NULL,
        strategy_version text NOT NULL,
        config_hash text NOT NULL,
        action text NOT NULL,
        outcome text,
        token_id text,
        fair_probability numeric(18,10),
        executable_price numeric(18,8),
        gross_edge_per_share numeric(30,10),
        fee_per_share numeric(30,10),
        reserve_per_share numeric(30,10),
        net_edge_per_share numeric(30,10),
        size numeric(30,10) NOT NULL DEFAULT 0,
        status text NOT NULL,
        reject_reason text,
        order_plan_id uuid,
        execution_mode text NOT NULL,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_btc_strategy_decisions PRIMARY KEY (decision_id, decision_at),
        CONSTRAINT chk_btc_decision_action CHECK (action IN ('buy','no_trade')),
        CONSTRAINT chk_btc_decision_outcome CHECK (outcome IS NULL OR outcome IN ('up','down')),
        CONSTRAINT chk_btc_decision_mode CHECK (execution_mode IN ('sim','paper'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.btc_paper_experiments (
        experiment_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        name text NOT NULL UNIQUE,
        status text NOT NULL,
        strategy_version text NOT NULL,
        feature_schema_version text NOT NULL,
        config_hash text NOT NULL,
        config jsonb NOT NULL,
        process_id uuid,
        started_at timestamptz,
        stopped_at timestamptz,
        stop_reason text,
        markets_observed bigint NOT NULL DEFAULT 0,
        snapshots_recorded bigint NOT NULL DEFAULT 0,
        decisions_recorded bigint NOT NULL DEFAULT 0,
        trades_entered bigint NOT NULL DEFAULT 0,
        trades_resolved bigint NOT NULL DEFAULT 0,
        gross_pnl numeric(30,10) NOT NULL DEFAULT 0,
        fees_paid numeric(30,10) NOT NULL DEFAULT 0,
        net_pnl numeric(30,10) NOT NULL DEFAULT 0,
        summary jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_btc_experiment_status CHECK (status IN ('configured','running','stopped','failed','completed'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.btc_market_labels (
        market_id text PRIMARY KEY,
        window_start timestamptz NOT NULL,
        window_end timestamptz NOT NULL,
        open_price numeric(30,10) NOT NULL,
        close_price numeric(30,10) NOT NULL,
        outcome text NOT NULL,
        label_source text NOT NULL,
        label_version text NOT NULL,
        source_open_timestamp timestamptz NOT NULL,
        source_close_timestamp timestamptz NOT NULL,
        label_available_at timestamptz NOT NULL,
        official_outcome text,
        official_resolved_at timestamptz,
        evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_btc_label_outcome CHECK (outcome IN ('up','down')),
        CONSTRAINT chk_btc_official_outcome CHECK (official_outcome IS NULL OR official_outcome IN ('up','down'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.ml_feature_vectors (
        vector_id uuid NOT NULL,
        feature_as_of timestamptz NOT NULL,
        snapshot_id uuid NOT NULL,
        market_id text NOT NULL,
        task text NOT NULL,
        feature_schema_version text NOT NULL,
        feature_schema_sha256 text NOT NULL,
        feature_hash text NOT NULL,
        vector jsonb NOT NULL,
        lineage jsonb NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_ml_feature_vectors PRIMARY KEY (vector_id, feature_as_of),
        CONSTRAINT chk_ml_feature_task CHECK (task IN ('fair_value_residual','fill_probability','toxicity'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.ml_dataset_manifests (
        dataset_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        name text NOT NULL,
        task text NOT NULL,
        feature_schema_version text NOT NULL,
        feature_schema_sha256 text NOT NULL,
        label_version text NOT NULL,
        trained_from timestamptz,
        trained_through timestamptz,
        row_count bigint NOT NULL,
        market_count bigint NOT NULL,
        manifest jsonb NOT NULL,
        manifest_sha256 text NOT NULL UNIQUE,
        created_at timestamptz NOT NULL DEFAULT now()
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.ml_model_versions (
        model_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        model_version text NOT NULL UNIQUE,
        task text NOT NULL,
        state text NOT NULL,
        artifact_version text NOT NULL,
        feature_schema_version text NOT NULL,
        feature_schema_sha256 text NOT NULL,
        dataset_manifest_sha256 text NOT NULL,
        label_version text NOT NULL,
        artifact jsonb NOT NULL,
        artifact_sha256 text NOT NULL UNIQUE,
        training_commit text,
        created_at timestamptz NOT NULL DEFAULT now(),
        retired_at timestamptz,
        CONSTRAINT chk_ml_model_task CHECK (task IN ('fair_value_residual','fill_probability','toxicity')),
        CONSTRAINT chk_ml_model_state CHECK (state IN ('candidate','shadow','retired'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.ml_shadow_predictions (
        prediction_id uuid NOT NULL,
        predicted_at timestamptz NOT NULL,
        snapshot_id uuid NOT NULL,
        market_id text NOT NULL,
        model_version text NOT NULL,
        task text NOT NULL,
        prior numeric(18,10),
        prediction numeric(18,10),
        calibrated_prediction numeric(18,10),
        status text NOT NULL,
        reject_reason text,
        inference_latency_us bigint NOT NULL DEFAULT 0,
        feature_hash text NOT NULL,
        artifact_sha256 text NOT NULL,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_ml_shadow_predictions PRIMARY KEY (prediction_id, predicted_at),
        CONSTRAINT chk_ml_shadow_status CHECK (status IN ('predicted','abstained','error'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.ml_evaluation_runs (
        evaluation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        model_version text NOT NULL,
        dataset_manifest_sha256 text NOT NULL,
        status text NOT NULL,
        config jsonb NOT NULL,
        started_at timestamptz NOT NULL,
        completed_at timestamptz,
        summary jsonb NOT NULL DEFAULT '{}'::jsonb,
        error text,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_ml_evaluation_status CHECK (status IN ('running','completed','failed'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.ml_evaluation_metrics (
        evaluation_id uuid NOT NULL REFERENCES polymarket.ml_evaluation_runs (evaluation_id) ON DELETE CASCADE,
        fold_key text NOT NULL,
        metric_name text NOT NULL,
        metric_value numeric(30,10),
        metric_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (evaluation_id, fold_key, metric_name)
      );
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_btc_interval_active_window ON polymarket.btc_interval_markets (active, closed, window_start DESC);
      CREATE UNIQUE INDEX IF NOT EXISTS uq_reference_ticks_dedup ON polymarket.reference_price_ticks (source, dedup_key, source_timestamp);
      CREATE INDEX IF NOT EXISTS idx_reference_ticks_source_ts ON polymarket.reference_price_ticks (source, symbol, source_timestamp DESC);
      CREATE INDEX IF NOT EXISTS idx_feed_events_market_ts ON polymarket.market_feed_events (market_id, source_timestamp DESC);
      CREATE INDEX IF NOT EXISTS idx_feed_events_integrity_ts ON polymarket.market_feed_events (integrity_status, source_timestamp DESC);
      CREATE INDEX IF NOT EXISTS idx_book_checkpoints_token_ts ON polymarket.orderbook_checkpoints (token_id, source_timestamp DESC);
      CREATE INDEX IF NOT EXISTS idx_btc_features_market_ts ON polymarket.btc_feature_snapshots (market_id, feature_as_of DESC);
      CREATE UNIQUE INDEX IF NOT EXISTS uq_btc_features_market_hash_ts ON polymarket.btc_feature_snapshots (market_id, feature_hash, feature_as_of);
      CREATE INDEX IF NOT EXISTS idx_btc_decisions_market_ts ON polymarket.btc_strategy_decisions (market_id, decision_at DESC);
      CREATE UNIQUE INDEX IF NOT EXISTS uq_btc_decision_entry_per_experiment_market ON polymarket.btc_strategy_decisions (experiment_id, market_id) WHERE action = 'buy' AND status IN ('approved','submitted','filled');
      CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_feature_snapshot_schema_ts ON polymarket.ml_feature_vectors (snapshot_id, feature_schema_version, feature_as_of);
      CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_shadow_snapshot_model_ts ON polymarket.ml_shadow_predictions (snapshot_id, model_version, predicted_at);
      CREATE INDEX IF NOT EXISTS idx_ml_shadow_model_ts ON polymarket.ml_shadow_predictions (model_version, predicted_at DESC);
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        PERFORM create_hypertable('polymarket.reference_price_ticks', 'source_timestamp', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
        PERFORM create_hypertable('polymarket.market_feed_events', 'source_timestamp', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);
        PERFORM create_hypertable('polymarket.orderbook_checkpoints', 'source_timestamp', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);
        PERFORM create_hypertable('polymarket.btc_feature_snapshots', 'feature_as_of', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
        PERFORM create_hypertable('polymarket.ml_feature_vectors', 'feature_as_of', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
        PERFORM create_hypertable('polymarket.ml_shadow_predictions', 'predicted_at', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
      END $$;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.reference_price_ticks SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'source_timestamp DESC, tick_id',
        timescaledb.compress_segmentby = 'source,symbol'
      );
      ALTER TABLE polymarket.market_feed_events SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'source_timestamp DESC, feed_event_id',
        timescaledb.compress_segmentby = 'event_type'
      );
      ALTER TABLE polymarket.orderbook_checkpoints SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'source_timestamp DESC, checkpoint_id',
        timescaledb.compress_segmentby = 'token_id'
      );
      ALTER TABLE polymarket.btc_feature_snapshots SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'feature_as_of DESC, snapshot_id',
        timescaledb.compress_segmentby = 'market_id,feature_schema_version'
      );
      ALTER TABLE polymarket.ml_feature_vectors SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'feature_as_of DESC, vector_id',
        timescaledb.compress_segmentby = 'task,feature_schema_version'
      );
      ALTER TABLE polymarket.ml_shadow_predictions SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'predicted_at DESC, prediction_id',
        timescaledb.compress_segmentby = 'task,model_version'
      );
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        PERFORM add_retention_policy('polymarket.market_feed_events', INTERVAL '14 days', if_not_exists => true);
        PERFORM add_retention_policy('polymarket.orderbook_checkpoints', INTERVAL '90 days', if_not_exists => true);
        PERFORM add_retention_policy('polymarket.reference_price_ticks', INTERVAL '180 days', if_not_exists => true);
        PERFORM add_retention_policy('polymarket.btc_feature_snapshots', INTERVAL '365 days', if_not_exists => true);
        PERFORM add_retention_policy('polymarket.ml_feature_vectors', INTERVAL '365 days', if_not_exists => true);
        PERFORM add_retention_policy('polymarket.ml_shadow_predictions', INTERVAL '365 days', if_not_exists => true);

        PERFORM add_compression_policy('polymarket.market_feed_events', INTERVAL '1 day', if_not_exists => true);
        PERFORM add_compression_policy('polymarket.orderbook_checkpoints', INTERVAL '1 day', if_not_exists => true);
        PERFORM add_compression_policy('polymarket.reference_price_ticks', INTERVAL '7 days', if_not_exists => true);
        PERFORM add_compression_policy('polymarket.btc_feature_snapshots', INTERVAL '7 days', if_not_exists => true);
        PERFORM add_compression_policy('polymarket.ml_feature_vectors', INTERVAL '7 days', if_not_exists => true);
        PERFORM add_compression_policy('polymarket.ml_shadow_predictions', INTERVAL '7 days', if_not_exists => true);
      END $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.ml_evaluation_metrics;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.ml_evaluation_runs;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.ml_shadow_predictions;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.ml_model_versions;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.ml_dataset_manifests;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.ml_feature_vectors;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.btc_market_labels;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.btc_paper_experiments;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.btc_strategy_decisions;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.btc_feature_snapshots;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.feed_sessions;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.orderbook_checkpoints;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.market_feed_events;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.reference_price_ticks;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.btc_interval_markets;`);
  }
}
