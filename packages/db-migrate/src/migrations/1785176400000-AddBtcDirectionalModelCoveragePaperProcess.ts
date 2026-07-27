import { MigrationInterface, QueryRunner } from 'typeorm';

const PROCESS_ID = 'bab4c090-68bd-415f-8cad-8788865bc058';
const PROCESS_KEY = 'btc-5m-directional-model-paper-min-entry-030-coverage';
const RUN_KEY = 'btc-5m-directional-model-20260727-coverage-validation-v2';
const RUN_ID = '5644836a-f437-506a-bc73-6e60e99aed9f';
const MODEL_KEY =
  'btc-5m-directional-histogram-enriched-20260421-20260720-coverage-v2';
const MODEL_SHA256 =
  'b318c70cf5d0262cfe437dccdf9fc04a1329ee78e3eb184b02a74a27db798c29';
const FEATURE_SCHEMA_SHA256 =
  '392aac87ecbc6704929b0ab91aed5de90dff26daf40edfa4e22d2c78a9d03dfc';
const PREREGISTRATION_SHA256 =
  '289a6ff5b8588b29ca15f699130dd3619bc42982bc65cd3184564842faab86d4';

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
        target_size: '5',
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
  source_process_id: '88fc207a-39d0-407d-88a7-cc33ac3f795e',
  comparison_cohort: 'btc5m-directional-model-coverage-validation-v2-20260727',
  comparison_arm: 'coverage_challenger',
  strategy_family: 'btc_5m_directional_model',
  model_key: MODEL_KEY,
  model_artifact_sha256: MODEL_SHA256,
  entry_admission_policy: 'none',
  directional_model_entry_policy: 'execute_directional_prediction',
  evidence_objective:
    'compare_calibrated_extended_model_decision_coverage_accuracy_latency_and_paper_pnl',
  preregistration:
    '2026-07-27|btc5m-directional-model-histogram-enriched-20260421-20260720-coverage-v2|coverage-challenger-strategy-only-no-entry-admission|first-confidence-crossing-60-240-5|confidence-0.89|target-size-5|min-entry-0.30|paper-only|entry-policy-execute-directional-prediction',
};

export class AddBtcDirectionalModelCoveragePaperProcess1785176400000
  implements MigrationInterface
{
  name = 'AddBtcDirectionalModelCoveragePaperProcess1785176400000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);

    const [tables]: Array<{ process_table: string | null; event_table: string | null }> =
      await queryRunner.query(`
        SELECT
          to_regclass('polymarket.trading_processes')::text AS process_table,
          to_regclass('polymarket.trading_process_events')::text AS event_table;
      `);
    if (!tables?.process_table || !tables?.event_table) {
      throw new Error(
        'refusing to add directional-model coverage paper process: required tables are missing',
      );
    }

    await queryRunner.query(`
      LOCK TABLE polymarket.trading_processes IN SHARE ROW EXCLUSIVE MODE;
      LOCK TABLE polymarket.trading_process_events IN SHARE MODE;
    `);

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
      throw new Error(
        'refusing to reuse directional-model coverage process identity',
      );
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
      throw new Error('refusing to reuse directional-model coverage run key');
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
        'BTC 5m directional model paper min entry 0.30 coverage challenger',
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
          AND process.process_key = $2
          AND process.process_type = 'btc_5m'
          AND process.process_scope = 'realtime_paper'
          AND process.status = 'created'
          AND NOT process.enabled
          AND process.started_at IS NULL
          AND process.heartbeat_at IS NULL
          AND process.stopped_at IS NULL
          AND process.config = $3::jsonb
          AND process.metadata = $4::jsonb
          AND process.config #>> '{execution,mode}' = 'paper'
          AND process.config #>> '{execution,execute_signals}' = 'true'
          AND process.config #>> '{execution,live_capital}' = 'false'
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,decision_strategy,type}'
                = 'btc_directional_model'
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,decision_strategy,model_key}'
                = $5
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,decision_strategy,artifact_sha256}'
                = $6
          AND process.config #>>
                '{raw,btc_realtime_paper,strategy,decision_strategy,feature_schema_sha256}'
                = $7
          AND process.config #> '{raw,btc_realtime_paper,entry_admission}' IS NULL
          AND process.config #>>
                '{raw,btc_realtime_paper,paper,directional_model_entry_policy}'
                = 'execute_directional_prediction'
          AND process.config #>>
                '{raw,btc_realtime_paper,next_experiment_key}'
                = $8
          AND process.config #>>
                '{raw,btc_realtime_paper,preregistration_sha256}'
                = $9
          AND process.metadata ->> 'source_process_id'
                = '88fc207a-39d0-407d-88a7-cc33ac3f795e'
          AND process.metadata ->> 'comparison_arm' = 'coverage_challenger'
          AND NOT EXISTS (
            SELECT 1
            FROM polymarket.trading_process_events event
            WHERE event.process_id = process.process_id
          );
      `,
      [
        PROCESS_ID,
        PROCESS_KEY,
        JSON.stringify(processConfig),
        JSON.stringify(processMetadata),
        MODEL_KEY,
        MODEL_SHA256,
        FEATURE_SCHEMA_SHA256,
        RUN_KEY,
        PREREGISTRATION_SHA256,
      ],
    );
    if (Number(inserted.count) !== 1) {
      throw new Error(
        'directional-model coverage process failed post-insert validation',
      );
    }
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);
    await queryRunner.query(`
      LOCK TABLE polymarket.trading_processes IN SHARE ROW EXCLUSIVE MODE;
      LOCK TABLE polymarket.trading_process_events IN SHARE MODE;
    `);

    const [eligible]: Array<{ count: string }> = await queryRunner.query(
      `
        SELECT count(*)::text AS count
        FROM polymarket.trading_processes process
        WHERE process.process_id = $1::uuid
          AND process.process_key = $2
          AND process.process_type = 'btc_5m'
          AND process.process_scope = 'realtime_paper'
          AND process.status = 'created'
          AND NOT process.enabled
          AND process.started_at IS NULL
          AND process.heartbeat_at IS NULL
          AND process.stopped_at IS NULL
          AND process.config = $3::jsonb
          AND process.metadata = $4::jsonb
          AND NOT EXISTS (
            SELECT 1
            FROM polymarket.trading_process_events event
            WHERE event.process_id = process.process_id
          );
      `,
      [
        PROCESS_ID,
        PROCESS_KEY,
        JSON.stringify(processConfig),
        JSON.stringify(processMetadata),
      ],
    );
    if (Number(eligible.count) !== 1) {
      throw new Error(
        'refusing to remove directional-model coverage process after lifecycle or evidence mutation',
      );
    }

    await queryRunner.query(
      `
        DELETE FROM polymarket.trading_processes
        WHERE process_id = $1::uuid
          AND process_key = $2;
      `,
      [PROCESS_ID, PROCESS_KEY],
    );
  }
}
