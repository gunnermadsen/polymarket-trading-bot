import { createHash } from 'node:crypto';

import { MigrationInterface, QueryRunner } from 'typeorm';

type PaperProcessDefinition = {
  processId: string;
  processKey: string;
  name: string;
  runId: string;
  runKey: string;
  comparisonArm: string;
  modelKey: string;
  modelSha256: string;
  featureSchemaVersion: string;
  featureSchemaSha256: string;
  featureCount: number;
  goldenVectorsSha256: string;
  sourceFreezeManifestSha256: string;
  sourceTrainingModelSha256: string;
  trainingRangeStartUtc: string;
  preregistration: string;
  preregistrationSha256: string;
};

const COMPARISON_COHORT =
  'btc5m-chainlink-oi-paper-model-comparison-v1-20260801';
const TRAINING_RANGE_END_EXCLUSIVE_UTC = '2026-07-14T00:00:00Z';
const BENCHMARK_HOLDOUT_RANGE_START_UTC = '2026-07-21T00:00:00Z';
const BENCHMARK_HOLDOUT_RANGE_END_EXCLUSIVE_UTC = '2026-07-29T00:00:00Z';

const processDefinitions: readonly PaperProcessDefinition[] = [
  {
    processId: '8b780d47-373f-47f9-8dac-c5658a01f119',
    processKey: 'btc-5m-chainlink-full-directional-paper-5-share-v1',
    name: 'BTC 5m Chainlink full directional model paper 5 share',
    runId: 'ea4cec70-eff4-53f6-aa18-8adf1a79da49',
    runKey: 'btc-5m-chainlink-full-20260703-20260713-paper-5-share-v1',
    comparisonArm: 'chainlink_full',
    modelKey: 'btc-5m-directional-chainlink-full-20260703-20260713-paper-v1',
    modelSha256:
      'f7606fccd8104d169ba49154e642ed8b23891a54e716ad53605f95511fa64e39',
    featureSchemaVersion:
      'btc-5m-directional-boundary-oracle-chainlink-refprice-candle-features-v1',
    featureSchemaSha256:
      '928601d3c6b68cf4c3bac0b5bb7dad213ada96b7364739cfb3dffdf1ed595bcd',
    featureCount: 97,
    goldenVectorsSha256:
      '60149e514f575dc3d0c4748adfdd3147e27a633455e6b76fac44a8b32f87c509',
    sourceFreezeManifestSha256:
      'f82ba9419b2b0c1e926f85c3ce899885cb3b793243733f297ed87b0eb5958a91',
    sourceTrainingModelSha256:
      'aba7992b2a9617df2a9e4d64999e106844f57148855d206331968b6f824b2cc1',
    trainingRangeStartUtc: '2026-07-03T00:00:00Z',
    preregistration:
      '2026-08-01|btc-5m-directional-chainlink-full-20260703-20260713-paper-v1|paper-only-strategy-only-no-entry-admission|first-confidence-crossing-60-240-5|confidence-0.89|target-size-5|min-entry-0.30|max-entry-0.95|entry-policy-execute-directional-prediction',
    preregistrationSha256:
      '884bd4ce086672f72652cf3d8e2439fb534966ad134b9ec50407023fe4e4ec09',
  },
  {
    processId: '8b2f7bef-5365-4c4f-80d3-95caf6381906',
    processKey: 'btc-5m-chainlink-full-oi-directional-paper-5-share-v1',
    name: 'BTC 5m Chainlink full OI directional model paper 5 share',
    runId: '2196055b-b951-5551-aaa1-6f3cc076c830',
    runKey: 'btc-5m-chainlink-full-oi-20260703-20260713-paper-5-share-v1',
    comparisonArm: 'chainlink_full_oi',
    modelKey:
      'btc-5m-directional-chainlink-full-oi-20260703-20260713-paper-v1',
    modelSha256:
      '6463f7932820e4ca0214aa98a0a6d71044891a027ca62ef3321d59b9333f7d7a',
    featureSchemaVersion:
      'btc-5m-directional-boundary-oracle-chainlink-refprice-candle-oi-features-v1',
    featureSchemaSha256:
      '8c26567a31b1985d0fbc018d1e79166fae3ea3574e1ad9aea9ff3c9f857317b5',
    featureCount: 106,
    goldenVectorsSha256:
      'd6fabe954588749f406096793b709faedbe5e013f351172d51fa943bed2ce78b',
    sourceFreezeManifestSha256:
      '51a6a11605b68a48a0c4a8bbfc68bda20b15d7b224f48df0bf9c3f445d7fc6d1',
    sourceTrainingModelSha256:
      'd25fe9388ffbf622fafee44ba29069cb25db022abf50077e93fd1d87b72688a5',
    trainingRangeStartUtc: '2026-07-03T00:00:00Z',
    preregistration:
      '2026-08-01|btc-5m-directional-chainlink-full-oi-20260703-20260713-paper-v1|paper-only-strategy-only-no-entry-admission|first-confidence-crossing-60-240-5|confidence-0.89|target-size-5|min-entry-0.30|max-entry-0.95|entry-policy-execute-directional-prediction',
    preregistrationSha256:
      '369bec0fd5ba55ed655a8ebb4cec0839da1de03050e1b6cfcbd1e314feaa21ce',
  },
  {
    processId: 'e12c3281-9209-42c7-a3c0-78175ab079ce',
    processKey:
      'btc-5m-long-history-chainlink-candle-directional-paper-5-share-v1',
    name: 'BTC 5m long-history Chainlink candle directional model paper 5 share',
    runId: '03d5b015-b847-5d52-a73c-1b7ba29a75c6',
    runKey:
      'btc-5m-long-history-chainlink-candle-20260321-20260713-paper-5-share-v1',
    comparisonArm: 'long_history_candle',
    modelKey:
      'btc-5m-directional-long-history-candle-20260321-20260713-paper-v1',
    modelSha256:
      '6685c1995bb6a938fb7b99a8164e7586ada0ed256fbbf5c605c9bdea71abfed3',
    featureSchemaVersion:
      'btc-5m-directional-boundary-oracle-chainlink-candle-features-v1',
    featureSchemaSha256:
      'a69ee1fefc2bbe312d35f5704a49e2d1296453caa6aff0f5bc94a8a63955e4da',
    featureCount: 87,
    goldenVectorsSha256:
      'd1c55294038eef82a319abdf2a38f63f66e21f6175315b653d635f3c0af8a0e6',
    sourceFreezeManifestSha256:
      'a85c5da6e2c7cc6477cdd2f8a1ea66e73e488fdb1a17cd264ad2eafa2eb42cf0',
    sourceTrainingModelSha256:
      '413fdf1a02bd743f39ad3c74c27afaebd819cae48a833324270721dc59784aae',
    trainingRangeStartUtc: '2026-03-21T00:00:00Z',
    preregistration:
      '2026-08-01|btc-5m-directional-long-history-candle-20260321-20260713-paper-v1|paper-only-strategy-only-no-entry-admission|first-confidence-crossing-60-240-5|confidence-0.89|target-size-5|min-entry-0.30|max-entry-0.95|entry-policy-execute-directional-prediction',
    preregistrationSha256:
      '9b302f482dac614aff23fea42313c72455d5b5708475b6ffc989ceb16d3c0b50',
  },
];

const sharedStrategyParameters = {
  target_size: '5',
  min_seconds_after_open: 60,
  min_seconds_before_close: 60,
  max_directional_feature_age_ms: 5000,
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
};

const sharedRuntimeParameters = {
  strategy_interval_ms: 1000,
  official_resolution_audit_grace_secs: 120,
  official_resolution_watch_retention_secs: 3600,
};

const sharedPaperParameters = {
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
};

function buildProcessConfig(definition: PaperProcessDefinition): object {
  return {
    execution: {
      mode: 'paper',
      execute_signals: true,
      live_capital: false,
    },
    raw: {
      btc_realtime_paper: {
        schema_version: 'btc_realtime_paper_process_v3',
        next_experiment_key: definition.runKey,
        preregistration_sha256: definition.preregistrationSha256,
        strategy: {
          decision_strategy: {
            type: 'btc_directional_model',
            model_key: definition.modelKey,
            artifact_sha256: definition.modelSha256,
            feature_schema_sha256: definition.featureSchemaSha256,
          },
          ...sharedStrategyParameters,
        },
        runtime: sharedRuntimeParameters,
        paper: sharedPaperParameters,
      },
    },
  };
}

function buildProcessMetadata(definition: PaperProcessDefinition): object {
  return {
    comparison_cohort: COMPARISON_COHORT,
    comparison_arm: definition.comparisonArm,
    strategy_family: 'btc_5m_directional_model',
    model_key: definition.modelKey,
    model_artifact_sha256: definition.modelSha256,
    model_feature_schema_version: definition.featureSchemaVersion,
    model_feature_schema_sha256: definition.featureSchemaSha256,
    model_feature_count: definition.featureCount,
    model_golden_vectors_sha256: definition.goldenVectorsSha256,
    model_source_freeze_manifest_sha256:
      definition.sourceFreezeManifestSha256,
    source_training_model_sha256: definition.sourceTrainingModelSha256,
    training_range_start_utc: definition.trainingRangeStartUtc,
    training_range_end_exclusive_utc: TRAINING_RANGE_END_EXCLUSIVE_UTC,
    benchmark_holdout_range_start_utc: BENCHMARK_HOLDOUT_RANGE_START_UTC,
    benchmark_holdout_range_end_exclusive_utc:
      BENCHMARK_HOLDOUT_RANGE_END_EXCLUSIVE_UTC,
    planned_run_id: definition.runId,
    planned_run_key: definition.runKey,
    confidence_threshold: '0.89',
    target_size: '5',
    entry_admission_policy: 'none',
    directional_model_entry_policy: 'execute_directional_prediction',
    deployment_scope: 'paper_only',
    paper_challenger: true,
    live_capital_allowed: false,
    production_qualified: false,
    evidence_objective:
      'compare_frozen_chainlink_and_open_interest_directional_models_on_identical_five_share_paper_execution',
    preregistration: definition.preregistration,
  };
}

function assertFrozenIdentities(): void {
  const processIds = new Set<string>();
  const processKeys = new Set<string>();
  const runIds = new Set<string>();
  const runKeys = new Set<string>();
  const modelKeys = new Set<string>();

  for (const definition of processDefinitions) {
    for (const [label, hash] of [
      ['model artifact', definition.modelSha256],
      ['feature schema', definition.featureSchemaSha256],
      ['golden vectors', definition.goldenVectorsSha256],
      ['source freeze manifest', definition.sourceFreezeManifestSha256],
      ['source training model', definition.sourceTrainingModelSha256],
    ] as const) {
      if (!/^[0-9a-f]{64}$/.test(hash)) {
        throw new Error(
          `refusing to add Chainlink/OI paper processes with invalid ${label} SHA-256 for ${definition.modelKey}`,
        );
      }
    }

    const calculatedPreregistrationSha256 = createHash('sha256')
      .update(definition.preregistration)
      .digest('hex');
    if (
      calculatedPreregistrationSha256 !== definition.preregistrationSha256
    ) {
      throw new Error(
        `refusing to add Chainlink/OI paper process with mismatched preregistration for ${definition.modelKey}`,
      );
    }

    if (
      processIds.has(definition.processId) ||
      processKeys.has(definition.processKey) ||
      runIds.has(definition.runId) ||
      runKeys.has(definition.runKey) ||
      modelKeys.has(definition.modelKey)
    ) {
      throw new Error(
        'refusing to add Chainlink/OI paper processes with duplicate frozen identities',
      );
    }
    processIds.add(definition.processId);
    processKeys.add(definition.processKey);
    runIds.add(definition.runId);
    runKeys.add(definition.runKey);
    modelKeys.add(definition.modelKey);
  }
}

async function assertRequiredTables(queryRunner: QueryRunner): Promise<void> {
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
      'refusing to mutate Chainlink/OI paper processes: required tables are missing',
    );
  }
}

export class AddBtcChainlinkOiPaperProcesses1785638500000
  implements MigrationInterface
{
  name = 'AddBtcChainlinkOiPaperProcesses1785638500000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    assertFrozenIdentities();
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);
    await assertRequiredTables(queryRunner);

    await queryRunner.query(`
      LOCK TABLE polymarket.trading_processes IN SHARE ROW EXCLUSIVE MODE;
      LOCK TABLE polymarket.trading_process_events IN SHARE MODE;
    `);

    for (const definition of processDefinitions) {
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
        [definition.processId, definition.processKey],
      );
      if (Number(collision.count) !== 0) {
        throw new Error(
          `refusing to reuse Chainlink/OI paper process identity ${definition.processKey}`,
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
        [definition.runId, definition.runKey],
      );
      if (Number(runCollision.count) !== 0) {
        throw new Error(
          `refusing to reuse Chainlink/OI paper run identity ${definition.runKey}`,
        );
      }
    }

    for (const definition of processDefinitions) {
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
          definition.processId,
          definition.name,
          definition.processKey,
          JSON.stringify(buildProcessConfig(definition)),
          JSON.stringify(buildProcessMetadata(definition)),
        ],
      );
    }

    for (const definition of processDefinitions) {
      const processConfig = buildProcessConfig(definition);
      const processMetadata = buildProcessMetadata(definition);
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
                  = '5'
            AND process.config #>>
                  '{raw,btc_realtime_paper,strategy,min_seconds_after_open}'
                  = '60'
            AND process.config #>>
                  '{raw,btc_realtime_paper,strategy,min_seconds_before_close}'
                  = '60'
            AND process.config #>>
                  '{raw,btc_realtime_paper,strategy,max_directional_feature_age_ms}'
                  = '5000'
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
            AND process.metadata ->> 'comparison_cohort' = $11
            AND process.metadata ->> 'comparison_arm' = $12
            AND process.metadata ->> 'model_key' = $6
            AND process.metadata ->> 'model_artifact_sha256' = $7
            AND process.metadata ->> 'model_feature_schema_version' = $13
            AND process.metadata ->> 'model_feature_schema_sha256' = $8
            AND process.metadata ->> 'model_feature_count' = $14
            AND process.metadata ->> 'model_golden_vectors_sha256' = $15
            AND process.metadata ->> 'model_source_freeze_manifest_sha256' = $16
            AND process.metadata ->> 'source_training_model_sha256' = $17
            AND process.metadata ->> 'planned_run_id' = $18
            AND process.metadata ->> 'planned_run_key' = $9
            AND process.metadata ->> 'entry_admission_policy' = 'none'
            AND process.metadata ->> 'directional_model_entry_policy'
                  = 'execute_directional_prediction'
            AND process.metadata ->> 'deployment_scope' = 'paper_only'
            AND process.metadata ->> 'live_capital_allowed' = 'false'
            AND process.metadata ->> 'production_qualified' = 'false'
            AND process.metadata ->> 'preregistration' = $19
            AND NOT EXISTS (
              SELECT 1
              FROM polymarket.trading_process_events event
              WHERE event.process_id = process.process_id
            );
        `,
        [
          definition.processId,
          definition.name,
          definition.processKey,
          JSON.stringify(processConfig),
          JSON.stringify(processMetadata),
          definition.modelKey,
          definition.modelSha256,
          definition.featureSchemaSha256,
          definition.runKey,
          definition.preregistrationSha256,
          COMPARISON_COHORT,
          definition.comparisonArm,
          definition.featureSchemaVersion,
          String(definition.featureCount),
          definition.goldenVectorsSha256,
          definition.sourceFreezeManifestSha256,
          definition.sourceTrainingModelSha256,
          definition.runId,
          definition.preregistration,
        ],
      );
      if (Number(inserted.count) !== 1) {
        throw new Error(
          `Chainlink/OI paper process failed post-insert validation: ${definition.processKey}`,
        );
      }
    }

    const [sharedParameters]: Array<{
      process_count: string;
      strategy_parameter_variants: string;
      runtime_variants: string;
      paper_variants: string;
    }> = await queryRunner.query(
      `
        SELECT
          count(*)::text AS process_count,
          count(DISTINCT (
            process.config #> '{raw,btc_realtime_paper,strategy}'
            - 'decision_strategy'
          ))::text AS strategy_parameter_variants,
          count(DISTINCT process.config #>
            '{raw,btc_realtime_paper,runtime}'
          )::text AS runtime_variants,
          count(DISTINCT process.config #>
            '{raw,btc_realtime_paper,paper}'
          )::text AS paper_variants
        FROM polymarket.trading_processes process
        WHERE process.process_id = ANY($1::uuid[]);
      `,
      [processDefinitions.map((definition) => definition.processId)],
    );
    if (
      Number(sharedParameters.process_count) !== processDefinitions.length ||
      Number(sharedParameters.strategy_parameter_variants) !== 1 ||
      Number(sharedParameters.runtime_variants) !== 1 ||
      Number(sharedParameters.paper_variants) !== 1
    ) {
      throw new Error(
        'Chainlink/OI paper arms do not share identical strategy, runtime, and paper parameters',
      );
    }
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    assertFrozenIdentities();
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);
    await assertRequiredTables(queryRunner);

    await queryRunner.query(`
      LOCK TABLE polymarket.trading_processes IN SHARE ROW EXCLUSIVE MODE;
      LOCK TABLE polymarket.trading_process_events IN SHARE MODE;
    `);

    for (const definition of processDefinitions) {
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
                  '{raw,btc_realtime_paper,next_experiment_key}'
                  = $9
            AND process.metadata ->> 'planned_run_id' = $10
            AND process.metadata ->> 'preregistration' = $11
            AND NOT EXISTS (
              SELECT 1
              FROM polymarket.trading_process_events event
              WHERE event.process_id = process.process_id
            );
        `,
        [
          definition.processId,
          definition.name,
          definition.processKey,
          JSON.stringify(buildProcessConfig(definition)),
          JSON.stringify(buildProcessMetadata(definition)),
          definition.modelKey,
          definition.modelSha256,
          definition.featureSchemaSha256,
          definition.runKey,
          definition.runId,
          definition.preregistration,
        ],
      );
      if (Number(eligible.count) !== 1) {
        throw new Error(
          `refusing to remove Chainlink/OI paper process after lifecycle, definition, or evidence mutation: ${definition.processKey}`,
        );
      }
    }

    const deleted: Array<{ process_id: string }> = await queryRunner.query(
      `
        DELETE FROM polymarket.trading_processes process
        WHERE process.process_id = ANY($1::uuid[])
          AND process.process_key = ANY($2::text[])
        RETURNING process.process_id::text;
      `,
      [
        processDefinitions.map((definition) => definition.processId),
        processDefinitions.map((definition) => definition.processKey),
      ],
    );
    const deletedIds = new Set(deleted.map((row) => row.process_id));
    if (
      deleted.length !== processDefinitions.length ||
      processDefinitions.some(
        (definition) => !deletedIds.has(definition.processId),
      )
    ) {
      throw new Error(
        'Chainlink/OI paper processes failed exact rollback deletion',
      );
    }

    const [remaining]: Array<{ count: string }> = await queryRunner.query(
      `
        SELECT count(*)::text AS count
        FROM polymarket.trading_processes process
        WHERE process.process_id = ANY($1::uuid[])
           OR (
             process.process_type = 'btc_5m'
             AND process.process_scope = 'realtime_paper'
             AND process.process_key = ANY($2::text[])
           );
      `,
      [
        processDefinitions.map((definition) => definition.processId),
        processDefinitions.map((definition) => definition.processKey),
      ],
    );
    if (Number(remaining.count) !== 0) {
      throw new Error(
        'Chainlink/OI paper process identities remained after rollback',
      );
    }
  }
}
