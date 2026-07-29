import { createHash } from 'node:crypto';

import { MigrationInterface, QueryRunner } from 'typeorm';

const PROCESS_ID = '12b66dd9-5c9b-4ed3-bb88-1f12c4d61b82';
const PROCESS_KEY =
  'btc-5m-directional-model-paper-min-entry-030-recency-28d-share-size-10';
const RUN_KEY =
  'btc-5m-directional-model-20260729-recency-28d-paper-share-size-10-v1';
const RUN_ID = '38c7ca07-2379-5c51-9e43-e6435edc47a6';
const SOURCE_PROCESS_ID = 'bab4c090-68bd-415f-8cad-8788865bc058';
const SOURCE_MODEL_KEY =
  'btc-5m-directional-histogram-enriched-20260421-20260720-coverage-v2';
const SOURCE_MODEL_SHA256 =
  'b318c70cf5d0262cfe437dccdf9fc04a1329ee78e3eb184b02a74a27db798c29';
const MODEL_KEY =
  'btc-5m-directional-histogram-mature-reversal-recency-28d-20260321-20260728-paper-v1';
const MODEL_SHA256 =
  'd94d44fe307926e97b3880ca0e52ac1285007121801cbf483606cba7700b55a6';
const FEATURE_SCHEMA_VERSION =
  'btc-5m-directional-mature-reversal-features-v1';
const FEATURE_SCHEMA_SHA256 =
  'c4ab70d0df6e9fdd8e3274cfffe5a90d0be06b89845a44650f699b3ff5e9628b';
const PREREGISTRATION =
  '2026-07-29|btc-5m-directional-histogram-mature-reversal-recency-28d-20260321-20260728-paper-v1|recency-28d-paper-challenger-strategy-only-no-entry-admission|first-confidence-crossing-60-240-5|confidence-0.87|target-size-10|min-entry-0.30|paper-only|entry-policy-execute-directional-prediction';
const PREREGISTRATION_SHA256 =
  'e413f39119191ce45e3d5509919806a55db43e4b23b7869b902a6ccd43feef04';

const processConfig = {
  execution: {
    mode: 'paper',
    execute_signals: true,
    live_capital: false,
  },
  raw: {
    btc_realtime_paper: {
      schema_version: 'btc_realtime_paper_process_v3',
      next_experiment_key: RUN_KEY,
      preregistration_sha256: PREREGISTRATION_SHA256,
      strategy: {
        decision_strategy: {
          type: 'btc_directional_model',
          model_key: MODEL_KEY,
          artifact_sha256: MODEL_SHA256,
          feature_schema_sha256: FEATURE_SCHEMA_SHA256,
        },
        target_size: '10',
        min_seconds_after_open: 60,
        min_seconds_before_close: 60,
        max_reference_age_ms: 2000,
        max_chainlink_open_delay_ms: 5000,
        max_book_age_ms: 2000,
        max_source_skew_ms: 1000,
        max_fee_age_ms: 3600000,
        min_entry_price: '0.30',
        max_entry_price: '0.95',
        max_depth_participation: '0.25',
        volatility_floor_per_sqrt_second: '0.00005',
        probability_floor: '0.01',
        basis_lead_weight: '0.25',
        momentum_1s_weight: '0.05',
        momentum_5s_weight: '0.10',
        momentum_30s_weight: '0.10',
        max_lead_sigma_fraction: '0.25',
        base_probability_uncertainty: '0.015',
        basis_uncertainty_weight: '1',
        feed_age_uncertainty_per_second: '0.002',
        max_probability_uncertainty: '0.10',
        spread_reserve_fraction: '0.10',
        slippage_reserve_bps: '25',
        latency_reserve_per_share: '0.005',
        min_net_edge_per_share: '0.015',
        min_net_edge_usd: '0.02',
        max_fee_rate: '1',
      },
      runtime: {
        strategy_interval_ms: 1000,
        official_resolution_audit_grace_secs: 120,
        official_resolution_watch_retention_secs: 3600,
      },
      paper: {
        arrival_latency_ms: 150,
        visible_depth_haircut: '0.80',
        starting_collateral_usd: '1000',
        directional_model_entry_policy: 'execute_directional_prediction',
        stress_previews: [
          {
            scenario_key: 'latency_300ms_depth_65pct',
            arrival_latency_ms: 300,
            visible_depth_haircut: '0.65',
          },
          {
            scenario_key: 'latency_600ms_depth_50pct',
            arrival_latency_ms: 600,
            visible_depth_haircut: '0.50',
          },
        ],
      },
    },
  },
};

const processMetadata = {
  source_process_id: SOURCE_PROCESS_ID,
  source_model_key: SOURCE_MODEL_KEY,
  source_model_artifact_sha256: SOURCE_MODEL_SHA256,
  comparison_cohort: 'btc5m-directional-model-recency-28d-paper-v1-20260729',
  comparison_arm: 'recency_28d_accuracy_challenger_share_size_10',
  strategy_family: 'btc_5m_directional_model',
  model_key: MODEL_KEY,
  model_artifact_sha256: MODEL_SHA256,
  model_feature_schema_version: FEATURE_SCHEMA_VERSION,
  model_feature_schema_sha256: FEATURE_SCHEMA_SHA256,
  development_evidence_range_start_utc: '2026-03-21T00:00:00Z',
  development_evidence_range_end_exclusive_utc: '2026-07-29T00:00:00Z',
  estimator_recency_half_life_days: 28,
  confidence_threshold: '0.87',
  target_size: '10',
  entry_admission_policy: 'none',
  directional_model_entry_policy: 'execute_directional_prediction',
  paper_challenger: true,
  production_qualified: false,
  evidence_objective:
    'measure_recency_weighted_model_decision_accuracy_confidence_tail_entry_timing_trade_frequency_and_paper_pnl',
  preregistration: PREREGISTRATION,
};

function assertFrozenIdentity(): void {
  if (!/^[0-9a-f]{64}$/.test(MODEL_SHA256)) {
    throw new Error(
      'refusing to add directional recency paper process before the model artifact SHA-256 is frozen',
    );
  }
  if (!/^[0-9a-f]{64}$/.test(FEATURE_SCHEMA_SHA256)) {
    throw new Error(
      'refusing to add directional recency paper process with an invalid feature-schema SHA-256',
    );
  }
  const calculatedPreregistrationSha256 = createHash('sha256')
    .update(PREREGISTRATION)
    .digest('hex');
  if (calculatedPreregistrationSha256 !== PREREGISTRATION_SHA256) {
    throw new Error(
      'refusing to add directional recency paper process with mismatched preregistration evidence',
    );
  }
}

export class AddBtcDirectionalRecencyPaperProcess1785357000000
  implements MigrationInterface
{
  name = 'AddBtcDirectionalRecencyPaperProcess1785357000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    assertFrozenIdentity();
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);

    const [tables]: Array<{
      process_table: string | null;
      event_table: string | null;
    }> = await queryRunner.query(`
      SELECT
        to_regclass('polymarket.trading_processes')::text AS process_table,
        to_regclass('polymarket.trading_process_events')::text AS event_table;
    `);
    if (!tables?.process_table || !tables?.event_table) {
      throw new Error(
        'refusing to add directional recency paper process: required tables are missing',
      );
    }

    await queryRunner.query(`
      LOCK TABLE polymarket.trading_processes IN SHARE ROW EXCLUSIVE MODE;
      LOCK TABLE polymarket.trading_process_events IN SHARE MODE;
    `);

    const [source]: Array<{ count: string }> = await queryRunner.query(
      `
        SELECT count(*)::text AS count
        FROM polymarket.trading_processes
        WHERE process_id = $1::uuid
          AND process_type = 'btc_5m'
          AND process_scope = 'realtime_paper'
          AND config #>>
                '{raw,btc_realtime_paper,strategy,decision_strategy,type}'
                = 'btc_directional_model'
          AND config #>>
                '{raw,btc_realtime_paper,strategy,decision_strategy,model_key}'
                = $2
          AND config #>>
                '{raw,btc_realtime_paper,strategy,decision_strategy,artifact_sha256}'
                = $3;
      `,
      [SOURCE_PROCESS_ID, SOURCE_MODEL_KEY, SOURCE_MODEL_SHA256],
    );
    if (Number(source.count) !== 1) {
      throw new Error(
        'refusing to add directional recency paper process without its exact predecessor',
      );
    }

    const [collision]: Array<{ count: string }> = await queryRunner.query(
      `
        SELECT count(*)::text AS count
        FROM polymarket.trading_processes
        WHERE process_id = $1::uuid
           OR (
             process_type = 'btc_5m'
             AND process_scope = 'realtime_paper'
             AND process_key = $2
           );
      `,
      [PROCESS_ID, PROCESS_KEY],
    );
    if (Number(collision.count) !== 0) {
      throw new Error('refusing to reuse directional recency process identity');
    }

    const [runCollision]: Array<{ count: string }> = await queryRunner.query(
      `
        SELECT (
          (
            SELECT count(*)
            FROM polymarket.trading_process_events event
            WHERE event.event_type = 'btc_run_manifest'
              AND (
                event.event_id = $1::uuid
                OR event.metadata ->> 'run_id' = $1::text
                OR event.metadata ->> 'run_key' = $2
              )
          )
          +
          (
            SELECT count(*)
            FROM polymarket.trading_processes process
            WHERE process.config #>>
                  '{raw,btc_realtime_paper,next_experiment_key}'
                  = $2
          )
        )::text AS count;
      `,
      [RUN_ID, RUN_KEY],
    );
    if (Number(runCollision.count) !== 0) {
      throw new Error('refusing to reuse directional recency paper run identity');
    }

    await queryRunner.query(
      `
        INSERT INTO polymarket.trading_processes (
          process_id,
          name,
          process_type,
          process_scope,
          process_key,
          status,
          enabled,
          started_at,
          heartbeat_at,
          stopped_at,
          config,
          metadata
        )
        VALUES (
          $1::uuid,
          $2,
          'btc_5m',
          'realtime_paper',
          $3,
          'created',
          false,
          NULL,
          NULL,
          NULL,
          $4::jsonb,
          $5::jsonb
        );
      `,
      [
        PROCESS_ID,
        'BTC 5m directional model recency 28d paper min entry 0.30 10 share',
        PROCESS_KEY,
        JSON.stringify(processConfig),
        JSON.stringify(processMetadata),
      ],
    );

    const [inserted]: Array<{ count: string }> = await queryRunner.query(
      `
        SELECT count(*)::text AS count
        FROM polymarket.trading_processes process
        WHERE process.process_id = $1::uuid
          AND process.name = $2
          AND process.process_key = $3
          AND process.process_type = 'btc_5m'
          AND process.process_scope = 'realtime_paper'
          AND process.status = 'created'
          AND NOT process.enabled
          AND process.hostname IS NULL
          AND process.pid IS NULL
          AND process.version IS NULL
          AND process.started_at IS NULL
          AND process.heartbeat_at IS NULL
          AND process.stopped_at IS NULL
          AND process.stop_reason IS NULL
          AND process.last_error IS NULL
          AND process.config = $4::jsonb
          AND process.metadata = $5::jsonb
          AND process.config #>> '{execution,mode}' = 'paper'
          AND process.config #>> '{execution,execute_signals}' = 'true'
          AND process.config #>> '{execution,live_capital}' = 'false'
          AND process.config #>>
                '{raw,btc_realtime_paper,schema_version}'
                = 'btc_realtime_paper_process_v3'
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,decision_strategy,type}'
                = 'btc_directional_model'
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,decision_strategy,model_key}'
                = $6
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,decision_strategy,artifact_sha256}'
                = $7
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,decision_strategy,feature_schema_sha256}'
                = $8
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,target_size}'
                = '10'
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,min_seconds_after_open}'
                = '60'
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,min_seconds_before_close}'
                = '60'
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,min_entry_price}'
                = '0.30'
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,max_entry_price}'
                = '0.95'
          AND process.config #>
                '{raw,btc_realtime_paper,entry_admission}'
                IS NULL
          AND process.config #>>
                '{raw,btc_realtime_paper,paper,directional_model_entry_policy}'
                = 'execute_directional_prediction'
          AND process.config #>>
                '{raw,btc_realtime_paper,next_experiment_key}'
                = $9
          AND process.config #>>
                '{raw,btc_realtime_paper,preregistration_sha256}'
                = $10
          AND process.metadata ->> 'source_process_id' = $11
          AND process.metadata ->> 'source_model_key' = $12
          AND process.metadata ->> 'source_model_artifact_sha256' = $13
          AND process.metadata ->> 'model_key' = $6
          AND process.metadata ->> 'model_artifact_sha256' = $7
          AND process.metadata ->> 'model_feature_schema_version' = $14
          AND process.metadata ->> 'model_feature_schema_sha256' = $8
          AND process.metadata ->> 'entry_admission_policy' = 'none'
          AND process.metadata ->> 'directional_model_entry_policy'
                = 'execute_directional_prediction'
          AND process.metadata ->> 'paper_challenger' = 'true'
          AND process.metadata ->> 'production_qualified' = 'false'
          AND process.metadata ->> 'preregistration' = $15
          AND NOT EXISTS (
            SELECT 1
            FROM polymarket.trading_process_events event
            WHERE event.process_id = process.process_id
          );
      `,
      [
        PROCESS_ID,
        'BTC 5m directional model recency 28d paper min entry 0.30 10 share',
        PROCESS_KEY,
        JSON.stringify(processConfig),
        JSON.stringify(processMetadata),
        MODEL_KEY,
        MODEL_SHA256,
        FEATURE_SCHEMA_SHA256,
        RUN_KEY,
        PREREGISTRATION_SHA256,
        SOURCE_PROCESS_ID,
        SOURCE_MODEL_KEY,
        SOURCE_MODEL_SHA256,
        FEATURE_SCHEMA_VERSION,
        PREREGISTRATION,
      ],
    );
    if (Number(inserted.count) !== 1) {
      throw new Error(
        'directional recency paper process failed post-insert validation',
      );
    }
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    assertFrozenIdentity();
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);

    const [tables]: Array<{
      process_table: string | null;
      event_table: string | null;
    }> = await queryRunner.query(`
      SELECT
        to_regclass('polymarket.trading_processes')::text AS process_table,
        to_regclass('polymarket.trading_process_events')::text AS event_table;
    `);
    if (!tables?.process_table || !tables?.event_table) {
      throw new Error(
        'refusing to remove directional recency paper process: required tables are missing',
      );
    }

    await queryRunner.query(`
      LOCK TABLE polymarket.trading_processes IN SHARE ROW EXCLUSIVE MODE;
      LOCK TABLE polymarket.trading_process_events IN SHARE MODE;
    `);

    const [eligible]: Array<{ count: string }> = await queryRunner.query(
      `
        SELECT count(*)::text AS count
        FROM polymarket.trading_processes process
        WHERE process.process_id = $1::uuid
          AND process.name = $2
          AND process.process_key = $3
          AND process.process_type = 'btc_5m'
          AND process.process_scope = 'realtime_paper'
          AND process.status = 'created'
          AND NOT process.enabled
          AND process.hostname IS NULL
          AND process.pid IS NULL
          AND process.version IS NULL
          AND process.started_at IS NULL
          AND process.heartbeat_at IS NULL
          AND process.stopped_at IS NULL
          AND process.stop_reason IS NULL
          AND process.last_error IS NULL
          AND process.config = $4::jsonb
          AND process.metadata = $5::jsonb
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,decision_strategy,model_key}'
                = $6
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,decision_strategy,artifact_sha256}'
                = $7
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,decision_strategy,feature_schema_sha256}'
                = $8
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,target_size}'
                = '10'
          AND process.config #>>
                '{raw,btc_realtime_paper,next_experiment_key}'
                = $9
          AND process.config #>>
                '{raw,btc_realtime_paper,preregistration_sha256}'
                = $10
          AND process.metadata ->> 'source_process_id' = $11
          AND process.metadata ->> 'preregistration' = $12
          AND NOT EXISTS (
            SELECT 1
            FROM polymarket.trading_process_events event
            WHERE event.process_id = process.process_id
          );
      `,
      [
        PROCESS_ID,
        'BTC 5m directional model recency 28d paper min entry 0.30 10 share',
        PROCESS_KEY,
        JSON.stringify(processConfig),
        JSON.stringify(processMetadata),
        MODEL_KEY,
        MODEL_SHA256,
        FEATURE_SCHEMA_SHA256,
        RUN_KEY,
        PREREGISTRATION_SHA256,
        SOURCE_PROCESS_ID,
        PREREGISTRATION,
      ],
    );
    if (Number(eligible.count) !== 1) {
      throw new Error(
        'refusing to remove directional recency paper process after lifecycle, definition, or evidence mutation',
      );
    }

    const deleteResult: Array<{ process_id: string }> = await queryRunner.query(
      `
        DELETE FROM polymarket.trading_processes
        WHERE process_id = $1::uuid
          AND process_key = $2
        RETURNING process_id::text;
      `,
      [PROCESS_ID, PROCESS_KEY],
    );
    if (
      deleteResult.length !== 1 ||
      deleteResult[0].process_id !== PROCESS_ID
    ) {
      throw new Error(
        'directional recency paper process failed exact rollback deletion',
      );
    }

    const [remaining]: Array<{ count: string }> = await queryRunner.query(
      `
        SELECT count(*)::text AS count
        FROM polymarket.trading_processes
        WHERE process_id = $1::uuid
           OR (
             process_type = 'btc_5m'
             AND process_scope = 'realtime_paper'
             AND process_key = $2
           );
      `,
      [PROCESS_ID, PROCESS_KEY],
    );
    if (Number(remaining.count) !== 0) {
      throw new Error(
        'directional recency paper process identity remained after rollback',
      );
    }
  }
}
