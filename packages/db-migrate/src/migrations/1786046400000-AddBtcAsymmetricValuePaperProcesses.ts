import { createHash } from 'node:crypto';

import { MigrationInterface, QueryRunner } from 'typeorm';

type ModelFeed = {
  feed: string;
  maximum_age_ms: number;
  require_sequence_integrity?: boolean;
};

type ProcessDefinition = {
  processId: string;
  processKey: string;
  name: string;
  runId: string;
  runKey: string;
  arm: string;
  modelKey: string;
  modelSha256: string;
  featureSchemaVersion: string;
  featureSchemaSha256: string;
  featureCount: number;
  goldenVectorsSha256: string;
  sourceTrainingModelSha256: string;
  requiredFeeds: readonly ModelFeed[];
};

const COMPARISON_COHORT = 'btc5m-asymmetric-value-paper-v1-20260806';
const SOURCE_BENCHMARK_SHA256 =
  '37a47df0eb7e764b69fdbc539bf658f27e31126d78dff73cbf7908229dffaba6';

const commonFeeds: readonly ModelFeed[] = [
  { feed: 'binance_btcusdt_one_second_v1', maximum_age_ms: 1000 },
  { feed: 'polymarket_btc5m_clob_execution_v1', maximum_age_ms: 2000 },
];

const definitions: readonly ProcessDefinition[] = [
  {
    processId: '81f82de7-002b-4ac7-814b-236c6742d81c',
    processKey: 'btc-5m-asymmetric-core-oracle-paper-20260806-v1',
    name: 'BTC 5m asymmetric value core plus oracle paper',
    runId: '3343ae8a-515a-4aec-971e-b6aaea514d72',
    runKey: 'btc-5m-asymmetric-core-oracle-paper-run-20260806-v1',
    arm: 'core_oracle',
    modelKey: 'btc-5m-asymmetric-core-oracle-paper-20260805-v1',
    modelSha256:
      '2c91e894356f6fee7fe9514e24c39da6e11602ffcb7f961f64848bde72418db9',
    featureSchemaVersion:
      'btc-5m-asymmetric-core-oracle-paper-20260805-v1-features-v1',
    featureSchemaSha256:
      'fe2a5aaee3df1ef899d2553712555091aa29b7481b3fed7805ba140dc8aa5014',
    featureCount: 75,
    goldenVectorsSha256:
      '9295a49203b23fe87e48f5acd3cdd3e6e5369a66d841b3c317c9ab50445d2a3a',
    sourceTrainingModelSha256:
      'f17dfca17a69c229a19ad35ba593f7a58904ba65530f750760eb4ea8c26e6b46',
    requiredFeeds: [
      ...commonFeeds,
      { feed: 'chainlink_btcusd_oracle_v1', maximum_age_ms: 300000 },
    ],
  },
  {
    processId: '92298c8b-40e3-4c9f-ad01-61311de1b247',
    processKey: 'btc-5m-asymmetric-core-binance-l2-paper-20260806-v1',
    name: 'BTC 5m asymmetric value core plus Binance L2 paper',
    runId: '54eeeabc-b2b4-4acc-a3f8-ffad4b4dcd88',
    runKey: 'btc-5m-asymmetric-core-binance-l2-paper-run-20260806-v1',
    arm: 'core_binance_l2',
    modelKey: 'btc-5m-asymmetric-core-binance-l2-paper-20260805-v1',
    modelSha256:
      '885d1452bffed2625fb2bbeb8e65a3d1c0310e3db2c6662e8074719baffc4d4d',
    featureSchemaVersion:
      'btc-5m-asymmetric-core-binance-l2-paper-20260805-v1-features-v1',
    featureSchemaSha256:
      '77e556b0def726dba4e6573f8fd47aff1fff5b7dd2293bbf2152719320f9e1a3',
    featureCount: 111,
    goldenVectorsSha256:
      '5cf60463eda58b52dda74a060f5a69ae3ec1ccbb3b629c8dac0b7e4c6f3aebc6',
    sourceTrainingModelSha256:
      '6248e7a6c63ec3a4bcbb05a3a131462174be1cd5b1fdc92da69227a9cc098e36',
    requiredFeeds: [
      ...commonFeeds,
      {
        feed: 'binance_btcusdt_l2_v1',
        maximum_age_ms: 2000,
        require_sequence_integrity: true,
      },
    ],
  },
  {
    processId: '2136d1f0-3530-4480-870f-087147e08b50',
    processKey: 'btc-5m-asymmetric-core-paper-20260806-v1',
    name: 'BTC 5m asymmetric value core paper',
    runId: 'fed328ca-8d9f-47e3-b8c0-889a806639d7',
    runKey: 'btc-5m-asymmetric-core-paper-run-20260806-v1',
    arm: 'core',
    modelKey: 'btc-5m-asymmetric-core-paper-20260805-v1',
    modelSha256:
      '4379aee1ab04b382425b76f2c8f32e80c86299a149cd9994c2515b8138de9813',
    featureSchemaVersion: 'btc-5m-asymmetric-core-paper-20260805-v1-features-v1',
    featureSchemaSha256:
      '633033efb069dfb54a5f7834ab355ea380bd01c5774b800fddb528322e1e73dd',
    featureCount: 71,
    goldenVectorsSha256:
      '74302410801482a65a7295e080b76115b90fd2a547d99433d3888938c895f157',
    sourceTrainingModelSha256:
      'f8388b5d45625785b5bc636fb92de8f3da150b5d0a6e7291b825577f8338143f',
    requiredFeeds: commonFeeds,
  },
];

const sharedStrategy = {
  target_size: '5',
  min_seconds_after_open: 1,
  min_seconds_before_close: 244,
  max_directional_feature_age_ms: 1000,
  max_reference_age_ms: 2000,
  max_chainlink_open_delay_ms: 5000,
  max_book_age_ms: 2000,
  max_source_skew_ms: 1000,
  max_fee_age_ms: 3600000,
  min_entry_price: '0.20',
  max_entry_price: '0.30',
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
  spread_reserve_fraction: '0',
  slippage_reserve_bps: '0',
  latency_reserve_per_share: '0.01',
  min_net_edge_per_share: '0.03',
  min_net_edge_usd: '0.15',
  max_fee_rate: '1',
};

function preregistration(definition: ProcessDefinition): string {
  return [
    '2026-08-06',
    definition.modelKey,
    'paper-only',
    'asymmetric-value',
    'seconds-1-through-55',
    'share-price-0.20-through-0.30',
    'quantity-5',
    'depth-participation-0.25',
    'execution-reserve-0.01',
    'minimum-edge-per-share-0.03',
  ].join('|');
}

function preregistrationSha256(definition: ProcessDefinition): string {
  return createHash('sha256').update(preregistration(definition)).digest('hex');
}

function processConfig(definition: ProcessDefinition): object {
  return {
    execution: { mode: 'paper', execute_signals: true, live_capital: false },
    raw: {
      btc_realtime_paper: {
        schema_version: 'btc_realtime_paper_process_v3',
        next_experiment_key: definition.runKey,
        preregistration_sha256: preregistrationSha256(definition),
        strategy: {
          decision_strategy: {
            type: 'btc_asymmetric_value_model',
            model_key: definition.modelKey,
            artifact_sha256: definition.modelSha256,
            feature_schema_sha256: definition.featureSchemaSha256,
          },
          required_model_feeds: definition.requiredFeeds,
          ...sharedStrategy,
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
          directional_model_entry_policy: 'require_positive_direct_edge',
          stress_previews: [],
        },
      },
    },
  };
}

function processMetadata(definition: ProcessDefinition): object {
  return {
    comparison_cohort: COMPARISON_COHORT,
    comparison_arm: definition.arm,
    strategy_family: 'btc_5m_asymmetric_value_model',
    model_key: definition.modelKey,
    model_artifact_sha256: definition.modelSha256,
    model_feature_schema_version: definition.featureSchemaVersion,
    model_feature_schema_sha256: definition.featureSchemaSha256,
    model_feature_count: definition.featureCount,
    model_golden_vectors_sha256: definition.goldenVectorsSha256,
    model_source_benchmark_sha256: SOURCE_BENCHMARK_SHA256,
    source_training_model_sha256: definition.sourceTrainingModelSha256,
    planned_run_id: definition.runId,
    planned_run_key: definition.runKey,
    policy: 'raw20_30_by55_edge_3c',
    deployment_scope: 'paper_only',
    live_capital_allowed: false,
    production_qualified: false,
    preregistration: preregistration(definition),
  };
}

function assertDefinitions(): void {
  const processIds = new Set<string>();
  const processKeys = new Set<string>();
  for (const definition of definitions) {
    for (const digest of [
      definition.modelSha256,
      definition.featureSchemaSha256,
      definition.goldenVectorsSha256,
      definition.sourceTrainingModelSha256,
      SOURCE_BENCHMARK_SHA256,
    ]) {
      if (!/^[0-9a-f]{64}$/.test(digest)) {
        throw new Error(`invalid frozen SHA-256 for ${definition.modelKey}`);
      }
    }
    if (processIds.has(definition.processId) || processKeys.has(definition.processKey)) {
      throw new Error('duplicate asymmetric paper process identity');
    }
    processIds.add(definition.processId);
    processKeys.add(definition.processKey);
  }
}

export class AddBtcAsymmetricValuePaperProcesses1786046400000
  implements MigrationInterface
{
  name = 'AddBtcAsymmetricValuePaperProcesses1786046400000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    assertDefinitions();
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);
    await queryRunner.query(`LOCK TABLE polymarket.trading_processes IN SHARE ROW EXCLUSIVE MODE;`);

    for (const definition of definitions) {
      const [collision]: Array<{ count: string }> = await queryRunner.query(
        `SELECT count(*)::text AS count
           FROM polymarket.trading_processes
          WHERE process_id = $1::uuid
             OR (process_type = 'btc_5m' AND process_scope = 'realtime_paper' AND process_key = $2);`,
        [definition.processId, definition.processKey],
      );
      if (Number(collision.count) !== 0) {
        throw new Error(`refusing to reuse process identity ${definition.processKey}`);
      }
      await queryRunner.query(
        `INSERT INTO polymarket.trading_processes (
           process_id, name, process_type, process_scope, process_key,
           status, enabled, config, metadata
         ) VALUES ($1::uuid, $2, 'btc_5m', 'realtime_paper', $3,
                   'created', false, $4::jsonb, $5::jsonb);`,
        [
          definition.processId,
          definition.name,
          definition.processKey,
          JSON.stringify(processConfig(definition)),
          JSON.stringify(processMetadata(definition)),
        ],
      );
    }

    const [validated]: Array<{ count: string }> = await queryRunner.query(
      `SELECT count(*)::text AS count
         FROM polymarket.trading_processes
        WHERE process_id = ANY($1::uuid[])
          AND status = 'created'
          AND NOT enabled
          AND config #>> '{execution,mode}' = 'paper'
          AND config #>> '{execution,execute_signals}' = 'true'
          AND config #>> '{execution,live_capital}' = 'false'
          AND config #>> '{raw,btc_realtime_paper,strategy,decision_strategy,type}' = 'btc_asymmetric_value_model'
          AND config #>> '{raw,btc_realtime_paper,strategy,min_seconds_after_open}' = '1'
          AND config #>> '{raw,btc_realtime_paper,strategy,min_seconds_before_close}' = '244'
          AND config #>> '{raw,btc_realtime_paper,strategy,min_entry_price}' = '0.20'
          AND config #>> '{raw,btc_realtime_paper,strategy,max_entry_price}' = '0.30'
          AND config #>> '{raw,btc_realtime_paper,strategy,min_net_edge_per_share}' = '0.03'
          AND config #>> '{raw,btc_realtime_paper,paper,directional_model_entry_policy}' = 'require_positive_direct_edge'
          AND metadata ->> 'deployment_scope' = 'paper_only'
          AND metadata ->> 'live_capital_allowed' = 'false';`,
      [definitions.map((definition) => definition.processId)],
    );
    if (Number(validated.count) !== definitions.length) {
      throw new Error('asymmetric value paper process post-insert validation failed');
    }
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    assertDefinitions();
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);
    await queryRunner.query(`LOCK TABLE polymarket.trading_processes IN SHARE ROW EXCLUSIVE MODE;`);
    for (const definition of definitions) {
      const result = await queryRunner.query(
        `DELETE FROM polymarket.trading_processes process
          WHERE process.process_id = $1::uuid
            AND process.status = 'created'
            AND NOT process.enabled
            AND process.started_at IS NULL
            AND process.heartbeat_at IS NULL
            AND process.config = $2::jsonb
            AND process.metadata = $3::jsonb
            AND NOT EXISTS (
              SELECT 1 FROM polymarket.trading_process_events event
               WHERE event.process_id = process.process_id
            )
        RETURNING process_id;`,
        [
          definition.processId,
          JSON.stringify(processConfig(definition)),
          JSON.stringify(processMetadata(definition)),
        ],
      );
      if (result.length !== 1) {
        throw new Error(`refusing to remove mutated process ${definition.processKey}`);
      }
    }
  }
}
