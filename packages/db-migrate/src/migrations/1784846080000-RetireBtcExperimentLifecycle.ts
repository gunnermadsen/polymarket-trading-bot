import { MigrationInterface, QueryRunner } from 'typeorm';

export class RetireBtcExperimentLifecycle1784846080000
  implements MigrationInterface
{
  name = 'RetireBtcExperimentLifecycle1784846080000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    // This schema cutover is intentionally not compatible with the old binary.
    // Quiesce the bot before migration and start the validated new image immediately afterward.
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);

    await queryRunner.query(`
      DO $$
      DECLARE
        actual_column_count integer;
        expected_object_count integer;
        source_experiment_attnum smallint;
        source_name_attnum smallint;
        decision_experiment_attnum smallint;
        decision_market_attnum smallint;
        decision_process_attnum smallint;
        decision_at_attnum smallint;
        decision_snapshot_attnum smallint;
        settlement_experiment_attnum smallint;
        settlement_order_attnum smallint;
        settlement_resolution_received_attnum smallint;
        settlement_id_attnum smallint;
        settlement_credited_at_attnum smallint;
      BEGIN
        IF to_regclass('polymarket.btc_paper_experiments') IS NULL
          OR to_regclass('polymarket.btc_strategy_decisions') IS NULL
          OR to_regclass('polymarket.btc_paper_settlement_ledger') IS NULL
          OR to_regclass('polymarket.trading_processes') IS NULL
          OR to_regclass('polymarket.trading_process_events') IS NULL
        THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: required tables are missing';
        END IF;

        SELECT count(*)
        INTO actual_column_count
        FROM pg_attribute
        WHERE attrelid = 'polymarket.btc_paper_experiments'::regclass
          AND attnum > 0
          AND NOT attisdropped;

        IF actual_column_count <> 22 OR EXISTS (
          WITH expected(attnum, column_name, column_type, is_not_null) AS (
            VALUES
              (1, 'experiment_id', 'uuid', true),
              (2, 'name', 'text', true),
              (3, 'status', 'text', true),
              (4, 'strategy_version', 'text', true),
              (5, 'feature_schema_version', 'text', true),
              (6, 'config_hash', 'text', true),
              (7, 'config', 'jsonb', true),
              (8, 'process_id', 'uuid', false),
              (9, 'started_at', 'timestamp with time zone', false),
              (10, 'stopped_at', 'timestamp with time zone', false),
              (11, 'stop_reason', 'text', false),
              (12, 'markets_observed', 'bigint', true),
              (13, 'snapshots_recorded', 'bigint', true),
              (14, 'decisions_recorded', 'bigint', true),
              (15, 'trades_entered', 'bigint', true),
              (16, 'trades_resolved', 'bigint', true),
              (17, 'gross_pnl', 'numeric(30,10)', true),
              (18, 'fees_paid', 'numeric(30,10)', true),
              (19, 'net_pnl', 'numeric(30,10)', true),
              (20, 'summary', 'jsonb', true),
              (21, 'created_at', 'timestamp with time zone', true),
              (22, 'updated_at', 'timestamp with time zone', true)
          )
          SELECT 1
          FROM expected
          LEFT JOIN pg_attribute actual
            ON actual.attrelid =
                 'polymarket.btc_paper_experiments'::regclass
           AND actual.attnum = expected.attnum
           AND NOT actual.attisdropped
          WHERE actual.attname IS DISTINCT FROM expected.column_name
             OR pg_catalog.format_type(actual.atttypid, actual.atttypmod)
                  IS DISTINCT FROM expected.column_type
             OR actual.attnotnull IS DISTINCT FROM expected.is_not_null
        ) THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: source schema has drifted';
        END IF;

        SELECT count(*)
        INTO actual_column_count
        FROM pg_attribute
        WHERE attrelid = 'polymarket.trading_process_events'::regclass
          AND attnum > 0
          AND NOT attisdropped;

        IF actual_column_count <> 8 OR EXISTS (
          WITH expected(attnum, column_name, column_type, is_not_null) AS (
            VALUES
              (1, 'event_id', 'uuid', true),
              (2, 'process_id', 'uuid', true),
              (3, 'timestamp_utc', 'timestamp with time zone', true),
              (4, 'level', 'text', true),
              (5, 'event_type', 'text', true),
              (6, 'message', 'text', false),
              (7, 'metadata', 'jsonb', true),
              (8, 'created_at', 'timestamp with time zone', true)
          )
          SELECT 1
          FROM expected
          LEFT JOIN pg_attribute actual
            ON actual.attrelid =
                 'polymarket.trading_process_events'::regclass
           AND actual.attnum = expected.attnum
           AND NOT actual.attisdropped
          WHERE actual.attname IS DISTINCT FROM expected.column_name
             OR pg_catalog.format_type(actual.atttypid, actual.atttypmod)
                  IS DISTINCT FROM expected.column_type
             OR actual.attnotnull IS DISTINCT FROM expected.is_not_null
        ) THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: event schema has drifted';
        END IF;

        IF NOT EXISTS (
          SELECT 1
          FROM pg_attribute
          WHERE attrelid = 'polymarket.btc_strategy_decisions'::regclass
            AND attname = 'experiment_id'
            AND pg_catalog.format_type(atttypid, atttypmod) = 'uuid'
            AND NOT attnotnull
            AND NOT attisdropped
        ) OR NOT EXISTS (
          SELECT 1
          FROM pg_attribute
          WHERE attrelid =
                  'polymarket.btc_paper_settlement_ledger'::regclass
            AND attname = 'experiment_id'
            AND pg_catalog.format_type(atttypid, atttypmod) = 'uuid'
            AND attnotnull
            AND NOT attisdropped
        ) OR EXISTS (
          SELECT 1
          FROM pg_attribute
          WHERE attrelid IN (
            'polymarket.btc_strategy_decisions'::regclass,
            'polymarket.btc_paper_settlement_ledger'::regclass
          )
            AND attname = 'run_id'
            AND NOT attisdropped
        ) THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: child run columns have drifted';
        END IF;

        SELECT
          (
            SELECT attnum
            FROM pg_attribute
            WHERE attrelid =
                    'polymarket.btc_paper_experiments'::regclass
              AND attname = 'experiment_id'
              AND NOT attisdropped
          ),
          (
            SELECT attnum
            FROM pg_attribute
            WHERE attrelid =
                    'polymarket.btc_strategy_decisions'::regclass
              AND attname = 'experiment_id'
              AND NOT attisdropped
          ),
          (
            SELECT attnum
            FROM pg_attribute
            WHERE attrelid =
                    'polymarket.btc_paper_experiments'::regclass
              AND attname = 'name'
              AND NOT attisdropped
          ),
          (
            SELECT attnum
            FROM pg_attribute
            WHERE attrelid =
                    'polymarket.btc_strategy_decisions'::regclass
              AND attname = 'market_id'
              AND NOT attisdropped
          ),
          (
            SELECT attnum
            FROM pg_attribute
            WHERE attrelid =
                    'polymarket.btc_strategy_decisions'::regclass
              AND attname = 'process_id'
              AND NOT attisdropped
          ),
          (
            SELECT attnum
            FROM pg_attribute
            WHERE attrelid =
                    'polymarket.btc_strategy_decisions'::regclass
              AND attname = 'decision_at'
              AND NOT attisdropped
          ),
          (
            SELECT attnum
            FROM pg_attribute
            WHERE attrelid =
                    'polymarket.btc_strategy_decisions'::regclass
              AND attname = 'snapshot_id'
              AND NOT attisdropped
          ),
          (
            SELECT attnum
            FROM pg_attribute
            WHERE attrelid =
                    'polymarket.btc_paper_settlement_ledger'::regclass
              AND attname = 'experiment_id'
              AND NOT attisdropped
          ),
          (
            SELECT attnum
            FROM pg_attribute
            WHERE attrelid =
                    'polymarket.btc_paper_settlement_ledger'::regclass
              AND attname = 'order_id'
              AND NOT attisdropped
          ),
          (
            SELECT attnum
            FROM pg_attribute
            WHERE attrelid =
                    'polymarket.btc_paper_settlement_ledger'::regclass
              AND attname = 'official_resolution_received_at'
              AND NOT attisdropped
          ),
          (
            SELECT attnum
            FROM pg_attribute
            WHERE attrelid =
                    'polymarket.btc_paper_settlement_ledger'::regclass
              AND attname = 'settlement_id'
              AND NOT attisdropped
          ),
          (
            SELECT attnum
            FROM pg_attribute
            WHERE attrelid =
                    'polymarket.btc_paper_settlement_ledger'::regclass
              AND attname = 'credited_at'
              AND NOT attisdropped
          )
        INTO
          source_experiment_attnum,
          decision_experiment_attnum,
          source_name_attnum,
          decision_market_attnum,
          decision_process_attnum,
          decision_at_attnum,
          decision_snapshot_attnum,
          settlement_experiment_attnum,
          settlement_order_attnum,
          settlement_resolution_received_attnum,
          settlement_id_attnum,
          settlement_credited_at_attnum;

        SELECT count(*)
        INTO actual_column_count
        FROM pg_constraint
        WHERE conrelid = 'polymarket.btc_paper_experiments'::regclass;

        SELECT count(*)
        INTO expected_object_count
        FROM pg_constraint constraint_record
        LEFT JOIN pg_index backing_index
          ON backing_index.indexrelid = constraint_record.conindid
        WHERE constraint_record.conrelid =
                'polymarket.btc_paper_experiments'::regclass
          AND (
            (
              constraint_record.conname =
                'btc_paper_experiments_pkey'
              AND constraint_record.contype = 'p'
              AND constraint_record.conkey =
                    ARRAY[source_experiment_attnum]::smallint[]
              AND NOT constraint_record.condeferrable
              AND NOT constraint_record.condeferred
              AND constraint_record.convalidated
              AND backing_index.indrelid =
                    'polymarket.btc_paper_experiments'::regclass
              AND backing_index.indisprimary
              AND backing_index.indisunique
              AND backing_index.indisvalid
              AND backing_index.indisready
              AND backing_index.indislive
              AND NOT backing_index.indisexclusion
              AND backing_index.indnkeyatts = 1
              AND backing_index.indnatts = 1
              AND backing_index.indkey[0] =
                    source_experiment_attnum
              AND backing_index.indoption[0] = 0
              AND backing_index.indexprs IS NULL
              AND backing_index.indpred IS NULL
            )
            OR
            (
              constraint_record.conname =
                'btc_paper_experiments_name_key'
              AND constraint_record.contype = 'u'
              AND constraint_record.conkey =
                    ARRAY[source_name_attnum]::smallint[]
              AND NOT constraint_record.condeferrable
              AND NOT constraint_record.condeferred
              AND constraint_record.convalidated
              AND backing_index.indrelid =
                    'polymarket.btc_paper_experiments'::regclass
              AND NOT backing_index.indisprimary
              AND backing_index.indisunique
              AND backing_index.indisvalid
              AND backing_index.indisready
              AND backing_index.indislive
              AND NOT backing_index.indisexclusion
              AND backing_index.indnkeyatts = 1
              AND backing_index.indnatts = 1
              AND backing_index.indkey[0] = source_name_attnum
              AND backing_index.indoption[0] = 0
              AND backing_index.indexprs IS NULL
              AND backing_index.indpred IS NULL
            )
            OR
            (
              constraint_record.conname = 'chk_btc_experiment_status'
              AND constraint_record.contype = 'c'
            )
          );

        IF actual_column_count <> 3 OR expected_object_count <> 3 THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: source constraints have drifted';
        END IF;

        IF (
          SELECT count(*)
          FROM pg_constraint
          WHERE confrelid = 'polymarket.btc_paper_experiments'::regclass
        ) <> 1 OR NOT EXISTS (
          SELECT 1
          FROM pg_constraint constraint_record
          WHERE constraint_record.conname =
                  'btc_paper_settlement_ledger_experiment_id_fkey'
            AND constraint_record.conrelid =
                  'polymarket.btc_paper_settlement_ledger'::regclass
            AND constraint_record.confrelid =
                  'polymarket.btc_paper_experiments'::regclass
            AND constraint_record.contype = 'f'
            AND constraint_record.conkey =
                  ARRAY[settlement_experiment_attnum]::smallint[]
            AND constraint_record.confkey =
                  ARRAY[source_experiment_attnum]::smallint[]
            AND constraint_record.confupdtype = 'a'
            AND constraint_record.confdeltype = 'r'
            AND constraint_record.confmatchtype = 's'
            AND NOT constraint_record.condeferrable
            AND NOT constraint_record.condeferred
            AND constraint_record.convalidated
        ) OR NOT EXISTS (
          SELECT 1
          FROM pg_constraint constraint_record
          JOIN pg_index backing_index
            ON backing_index.indexrelid = constraint_record.conindid
          WHERE constraint_record.conname =
                  'uq_btc_paper_settlement_experiment_order'
            AND constraint_record.conrelid =
                  'polymarket.btc_paper_settlement_ledger'::regclass
            AND constraint_record.contype = 'u'
            AND constraint_record.conkey = ARRAY[
              settlement_experiment_attnum,
              settlement_order_attnum
            ]::smallint[]
            AND NOT constraint_record.condeferrable
            AND NOT constraint_record.condeferred
            AND constraint_record.convalidated
            AND backing_index.indrelid =
                  'polymarket.btc_paper_settlement_ledger'::regclass
            AND backing_index.indisunique
            AND backing_index.indisvalid
            AND backing_index.indisready
            AND backing_index.indislive
            AND NOT backing_index.indisprimary
            AND NOT backing_index.indisexclusion
            AND backing_index.indnkeyatts = 2
            AND backing_index.indnatts = 2
            AND backing_index.indkey[0] =
                  settlement_experiment_attnum
            AND backing_index.indkey[1] = settlement_order_attnum
            AND backing_index.indoption[0] = 0
            AND backing_index.indoption[1] = 0
            AND backing_index.indexprs IS NULL
            AND backing_index.indpred IS NULL
        ) THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: child constraints have drifted';
        END IF;

        SELECT count(*)
        INTO expected_object_count
        FROM pg_class index_record
        JOIN pg_namespace index_namespace
          ON index_namespace.oid = index_record.relnamespace
        JOIN pg_index index_definition
          ON index_definition.indexrelid = index_record.oid
        JOIN pg_am index_method
          ON index_method.oid = index_record.relam
        WHERE index_namespace.nspname = 'polymarket'
          AND index_record.relkind = 'i'
          AND index_method.amname = 'btree'
          AND (
            (
              index_record.relname =
                'uq_btc_decision_entry_per_experiment_market'
              AND index_definition.indrelid =
                'polymarket.btc_strategy_decisions'::regclass
              AND index_definition.indisunique
              AND index_definition.indisvalid
              AND index_definition.indisready
              AND index_definition.indislive
              AND NOT index_definition.indisprimary
              AND NOT index_definition.indisexclusion
              AND index_definition.indnkeyatts = 2
              AND index_definition.indnatts = 2
              AND index_definition.indkey[0] =
                    decision_experiment_attnum
              AND index_definition.indkey[1] = decision_market_attnum
              AND index_definition.indoption[0] = 0
              AND index_definition.indoption[1] = 0
              AND index_definition.indexprs IS NULL
              AND index_definition.indpred IS NOT NULL
              AND regexp_replace(
                pg_get_expr(
                  index_definition.indpred,
                  index_definition.indrelid
                ),
                '[[:space:]()]',
                '',
                'g'
              ) =
                'action=''buy''::textANDstatus=ANYARRAY[''approved''::text,''submitted''::text,''filled''::text]'
            )
            OR
            (
              index_record.relname =
                'idx_btc_decisions_experiment_process_at'
              AND index_definition.indrelid =
                'polymarket.btc_strategy_decisions'::regclass
              AND NOT index_definition.indisunique
              AND index_definition.indisvalid
              AND index_definition.indisready
              AND index_definition.indislive
              AND NOT index_definition.indisprimary
              AND NOT index_definition.indisexclusion
              AND index_definition.indnkeyatts = 4
              AND index_definition.indnatts = 4
              AND index_definition.indkey[0] =
                    decision_experiment_attnum
              AND index_definition.indkey[1] = decision_process_attnum
              AND index_definition.indkey[2] = decision_at_attnum
              AND index_definition.indkey[3] = decision_snapshot_attnum
              AND index_definition.indoption[0] = 0
              AND index_definition.indoption[1] = 0
              AND index_definition.indoption[2] = 3
              AND index_definition.indoption[3] = 0
              AND index_definition.indexprs IS NULL
              AND index_definition.indpred IS NULL
            )
            OR
            (
              index_record.relname =
                'idx_btc_paper_settlement_experiment_credited'
              AND index_definition.indrelid =
                'polymarket.btc_paper_settlement_ledger'::regclass
              AND NOT index_definition.indisunique
              AND index_definition.indisvalid
              AND index_definition.indisready
              AND index_definition.indislive
              AND NOT index_definition.indisprimary
              AND NOT index_definition.indisexclusion
              AND index_definition.indnkeyatts = 3
              AND index_definition.indnatts = 3
              AND index_definition.indkey[0] =
                    settlement_experiment_attnum
              AND index_definition.indkey[1] =
                    settlement_credited_at_attnum
              AND index_definition.indkey[2] = settlement_id_attnum
              AND index_definition.indoption[0] = 0
              AND index_definition.indoption[1] = 3
              AND index_definition.indoption[2] = 0
              AND index_definition.indexprs IS NULL
              AND index_definition.indpred IS NOT NULL
              AND regexp_replace(
                pg_get_expr(
                  index_definition.indpred,
                  index_definition.indrelid
                ),
                '[[:space:]()]',
                '',
                'g'
              ) = 'credit_status=''credited''::text'
            )
            OR
            (
              index_record.relname =
                'idx_btc_paper_settlement_pending'
              AND index_definition.indrelid =
                'polymarket.btc_paper_settlement_ledger'::regclass
              AND NOT index_definition.indisunique
              AND index_definition.indisvalid
              AND index_definition.indisready
              AND index_definition.indislive
              AND NOT index_definition.indisprimary
              AND NOT index_definition.indisexclusion
              AND index_definition.indnkeyatts = 3
              AND index_definition.indnatts = 3
              AND index_definition.indkey[0] =
                    settlement_experiment_attnum
              AND index_definition.indkey[1] =
                    settlement_resolution_received_attnum
              AND index_definition.indkey[2] = settlement_id_attnum
              AND index_definition.indoption[0] = 0
              AND index_definition.indoption[1] = 0
              AND index_definition.indoption[2] = 0
              AND index_definition.indexprs IS NULL
              AND index_definition.indpred IS NOT NULL
              AND regexp_replace(
                pg_get_expr(
                  index_definition.indpred,
                  index_definition.indrelid
                ),
                '[[:space:]()]',
                '',
                'g'
              ) = 'credit_status=''pending''::text'
            )
          );

        IF expected_object_count <> 4 THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: child indexes have drifted';
        END IF;

        IF EXISTS (
          SELECT 1
          FROM pg_constraint
          WHERE conname = 'uq_btc_paper_settlement_run_order'
            AND conrelid =
                  'polymarket.btc_paper_settlement_ledger'::regclass
        ) OR to_regclass(
          'polymarket.uq_btc_paper_settlement_run_order'
        ) IS NOT NULL OR to_regclass(
          'polymarket.uq_btc_decision_entry_per_run_market'
        ) IS NOT NULL OR to_regclass(
          'polymarket.idx_btc_decisions_run_process_at'
        ) IS NOT NULL OR to_regclass(
          'polymarket.idx_btc_paper_settlement_run_credited'
        ) IS NOT NULL THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: destination names already exist';
        END IF;

        IF to_regclass('timescaledb_information.hypertables') IS NULL
          OR NOT EXISTS (
            SELECT 1
            FROM timescaledb_information.hypertables
            WHERE hypertable_schema = 'polymarket'
              AND hypertable_name = 'trading_process_events'
          ) THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: event target is not a hypertable';
        END IF;

        IF to_regclass('timescaledb_information.jobs') IS NULL THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: retention policy catalog is unavailable';
        END IF;

        IF EXISTS (
          SELECT 1
          FROM timescaledb_information.jobs
          WHERE hypertable_schema = 'polymarket'
            AND hypertable_name = 'trading_process_events'
            AND proc_name = 'policy_retention'
        ) THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: run manifests would be subject to event retention';
        END IF;
      END $$;
    `);

    const validateRetirementData = `
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1
          FROM polymarket.btc_paper_experiments manifest
          LEFT JOIN polymarket.trading_processes process
            ON process.process_id = manifest.process_id
          WHERE manifest.process_id IS NULL
             OR process.process_id IS NULL
             OR process.process_type IS DISTINCT FROM 'btc_5m'
             OR process.process_scope IS DISTINCT FROM 'realtime_paper'
             OR jsonb_typeof(process.config) IS DISTINCT FROM 'object'
             OR length(btrim(manifest.name)) = 0
             OR manifest.status NOT IN (
               'configured',
               'running',
               'stopped',
               'failed',
               'completed'
             )
             OR manifest.config_hash !~ '^[0-9a-f]{64}$'
             OR jsonb_typeof(manifest.config) IS DISTINCT FROM 'object'
             OR (
               manifest.status = 'running'
               AND (
                 manifest.started_at IS NULL
                 OR manifest.stopped_at IS NOT NULL
                 OR manifest.name IS DISTINCT FROM
                      process.config #>>
                        '{raw,btc_realtime_paper,next_experiment_key}'
               )
             )
          LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: run ownership or frozen config is invalid';
        END IF;

        IF EXISTS (
          WITH lifecycle AS (
            SELECT
              process.process_id,
              (
                process.enabled
                AND process.status IN ('starting', 'running', 'stopping')
                AND process.stopped_at IS NULL
              ) AS is_active,
              count(manifest.experiment_id) FILTER (
                WHERE manifest.status = 'running'
              ) AS running_run_count
            FROM polymarket.trading_processes process
            LEFT JOIN polymarket.btc_paper_experiments manifest
              ON manifest.process_id = process.process_id
            WHERE process.process_type = 'btc_5m'
              AND process.process_scope = 'realtime_paper'
            GROUP BY
              process.process_id,
              process.enabled,
              process.status,
              process.stopped_at
          )
          SELECT 1
          FROM lifecycle
          WHERE (is_active AND running_run_count <> 1)
             OR (NOT is_active AND running_run_count <> 0)
          LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: process and run lifecycle states differ';
        END IF;

        IF EXISTS (
          SELECT 1
          FROM polymarket.btc_paper_experiments manifest
          JOIN polymarket.trading_process_events event
            ON event.event_id = manifest.experiment_id
          LIMIT 1
        ) OR EXISTS (
          SELECT 1
          FROM polymarket.trading_process_events
          WHERE event_type = 'btc_run_manifest'
          LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: run manifest event identity collides';
        END IF;
      END $$;
    `;

    const validateChildOwnershipData = `
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1
          FROM polymarket.btc_strategy_decisions decision
          LEFT JOIN polymarket.btc_paper_experiments manifest
            ON manifest.experiment_id = decision.experiment_id
          WHERE decision.experiment_id IS NULL
             OR decision.process_id IS NULL
             OR manifest.experiment_id IS NULL
             OR decision.process_id IS DISTINCT FROM manifest.process_id
          LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: decision process/run ownership is incomplete or invalid';
        END IF;

        IF EXISTS (
          SELECT 1
          FROM polymarket.btc_paper_settlement_ledger settlement
          JOIN polymarket.btc_paper_experiments manifest
            ON manifest.experiment_id = settlement.experiment_id
          WHERE settlement.process_id IS DISTINCT FROM manifest.process_id
          LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: settlement run ownership is invalid';
        END IF;
      END $$;
    `;

    // Validate without heavyweight table locks so drift fails before cutover.
    await queryRunner.query(validateRetirementData);
    await queryRunner.query(validateChildOwnershipData);

    await queryRunner.query(`
      SELECT pg_advisory_xact_lock(
        hashtextextended('polymarket.btc_run_manifest.start', 0)
      );
    `);

    await queryRunner.query(`
      LOCK TABLE
        polymarket.btc_paper_experiments,
        polymarket.btc_strategy_decisions,
        polymarket.btc_paper_settlement_ledger
      IN ACCESS EXCLUSIVE MODE;
      LOCK TABLE polymarket.trading_processes IN SHARE MODE;
    `);

    // The short lock window closes races with the legacy lifecycle writers.
    await queryRunner.query(validateRetirementData);
    await queryRunner.query(validateChildOwnershipData);

    await queryRunner.query(`
      DO $$
      DECLARE
        inserted_count bigint;
        source_count bigint;
        manifest_event_count bigint;
      BEGIN
        SELECT count(*)
        INTO source_count
        FROM polymarket.btc_paper_experiments;

        INSERT INTO polymarket.trading_process_events (
          event_id,
          process_id,
          timestamp_utc,
          level,
          event_type,
          message,
          metadata,
          created_at
        )
        SELECT
          manifest.experiment_id,
          manifest.process_id,
          COALESCE(manifest.started_at, manifest.created_at),
          'info',
          'btc_run_manifest',
          'Immutable BTC execution run manifest',
          jsonb_build_object(
            'run_id', manifest.experiment_id,
            'run_key', manifest.name,
            'config_hash', manifest.config_hash,
            'frozen_process_config', manifest.config
          ),
          manifest.created_at
        FROM polymarket.btc_paper_experiments manifest
        ORDER BY manifest.experiment_id;

        GET DIAGNOSTICS inserted_count = ROW_COUNT;

        SELECT count(*)
        INTO manifest_event_count
        FROM polymarket.trading_process_events
        WHERE event_type = 'btc_run_manifest';

        IF inserted_count <> source_count
          OR manifest_event_count <> source_count
        THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: run manifest row count differs';
        END IF;

        IF EXISTS (
          SELECT 1
          FROM polymarket.btc_paper_experiments manifest
          LEFT JOIN polymarket.trading_process_events event
            ON event.event_id = manifest.experiment_id
           AND event.timestamp_utc =
                 COALESCE(manifest.started_at, manifest.created_at)
          WHERE event.event_id IS NULL
             OR event.process_id IS DISTINCT FROM manifest.process_id
             OR event.level IS DISTINCT FROM 'info'
             OR event.event_type IS DISTINCT FROM 'btc_run_manifest'
             OR event.message IS DISTINCT FROM
                  'Immutable BTC execution run manifest'
             OR event.metadata IS DISTINCT FROM jsonb_build_object(
               'run_id', manifest.experiment_id,
               'run_key', manifest.name,
               'config_hash', manifest.config_hash,
               'frozen_process_config', manifest.config
             )
             OR event.created_at IS DISTINCT FROM manifest.created_at
          LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: source-to-event mapping differs';
        END IF;

        IF EXISTS (
          SELECT 1
          FROM polymarket.trading_process_events event
          LEFT JOIN polymarket.btc_paper_experiments manifest
            ON manifest.experiment_id = event.event_id
           AND COALESCE(manifest.started_at, manifest.created_at) =
                 event.timestamp_utc
          WHERE event.event_type = 'btc_run_manifest'
            AND (
              manifest.experiment_id IS NULL
              OR event.process_id IS DISTINCT FROM manifest.process_id
              OR event.metadata IS DISTINCT FROM jsonb_build_object(
                'run_id', manifest.experiment_id,
                'run_key', manifest.name,
                'config_hash', manifest.config_hash,
                'frozen_process_config', manifest.config
              )
            )
          LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'refusing to retire BTC experiment lifecycle: event-to-source mapping differs';
        END IF;
      END $$;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.btc_paper_settlement_ledger
        DROP CONSTRAINT btc_paper_settlement_ledger_experiment_id_fkey;

      ALTER TABLE polymarket.btc_strategy_decisions
        RENAME COLUMN experiment_id TO run_id;
      ALTER TABLE polymarket.btc_paper_settlement_ledger
        RENAME COLUMN experiment_id TO run_id;

      ALTER TABLE polymarket.btc_strategy_decisions
        ALTER COLUMN process_id SET NOT NULL,
        ALTER COLUMN run_id SET NOT NULL;

      ALTER TABLE polymarket.btc_paper_settlement_ledger
        RENAME CONSTRAINT uq_btc_paper_settlement_experiment_order
        TO uq_btc_paper_settlement_run_order;

      ALTER INDEX polymarket.uq_btc_decision_entry_per_experiment_market
        RENAME TO uq_btc_decision_entry_per_run_market;
      ALTER INDEX polymarket.idx_btc_decisions_experiment_process_at
        RENAME TO idx_btc_decisions_run_process_at;
      ALTER INDEX polymarket.idx_btc_paper_settlement_experiment_credited
        RENAME TO idx_btc_paper_settlement_run_credited;

      DROP TABLE polymarket.btc_paper_experiments;
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    throw new Error(
      'RetireBtcExperimentLifecycle1784846080000 is intentionally irreversible',
    );
  }
}
