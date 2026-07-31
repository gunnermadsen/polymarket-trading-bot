import { createHash } from 'node:crypto';

import { MigrationInterface, QueryRunner } from 'typeorm';

const SOURCE_PROCESS_ID: string = '78f614b9-97c9-4202-a9b1-cc8f01af455b';
const SOURCE_PROCESS_NAME =
  'BTC 5m directional model boundary alignment paper';
const SOURCE_PROCESS_KEY: string =
  'btc-5m-directional-model-paper-boundary-alignment';

const PROCESS_ID: string = 'effa3e5e-2f5a-4f18-98ba-06e4c0da74ef';
const PROCESS_NAME = 'BTC 5m directional model boundary alignment live pilot';
const PROCESS_KEY: string =
  'btc-5m-directional-model-boundary-alignment-live-pilot';
const ACCOUNT_REF = 'polymarket-primary';
const RUN_KEY =
  'btc-5m-directional-model-boundary-alignment-live-pilot-v1';
const RUN_ID = 'a8d51471-7cd0-5df5-b698-85248a57732c';
const PREREGISTRATION =
  '2026-07-31|btc-5m-directional-boundary-alignment-20260421-20260720-paper-v1|shared-runner-live-credential-validation-only|source-process-78f614b9-97c9-4202-a9b1-cc8f01af455b|account-ref-polymarket-primary|execution-disabled|live-capital-disabled';
const PREREGISTRATION_SHA256 =
  '830c15427659e53fb06d0a62efbc795c8839ba27e9f0e5c6dd15fa733f90c891';

const MODEL_KEY =
  'btc-5m-directional-boundary-alignment-20260421-20260720-paper-v1';
const MODEL_SHA256 =
  'c0778189865ca97a748a9f76cbe72d13268fd6e76db683ea727ad142b0576bc4';
const FEATURE_SCHEMA_VERSION = 'btc-5m-directional-boundary-features-v1';
const FEATURE_SCHEMA_SHA256 =
  'd2cadbff9ae97e20d2cbeb562310e27af80279f9ecbdbec85221af82ef4b9eea';
const SOURCE_PREREGISTRATION =
  '2026-07-29|btc-5m-directional-boundary-alignment-20260421-20260720-paper-v1|shared-runner-paper|target-size-5|min-after-open-60|min-before-close-60|min-entry-0.30|max-entry-0.95|entry-policy-execute-directional-prediction';
const SOURCE_PREREGISTRATION_SHA256 =
  '1f0f4023135da030b744c62212cd3a5b6965701ea04e3dd8b393d7eb3778e144';

const strategy = {
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
};

const runtime = {
  strategy_interval_ms: 1000,
  official_resolution_audit_grace_secs: 120,
  official_resolution_watch_retention_secs: 3600,
};

const paper = {
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

const sourceProcessConfig = {
  execution: {
    mode: 'paper',
    execute_signals: true,
    live_capital: false,
    taker_fee_rate: null,
  },
  raw: {
    btc_realtime_paper: {
      schema_version: 'btc_realtime_paper_process_v3',
      next_experiment_key:
        'btc-5m-directional-model-paper-boundary-alignment-run-v1',
      preregistration_sha256: SOURCE_PREREGISTRATION_SHA256,
      strategy,
      runtime,
      paper,
    },
  },
};

const sourceProcessMetadata = {
  strategy_family: 'btc_5m_directional_model',
  model_key: MODEL_KEY,
  model_artifact_sha256: MODEL_SHA256,
  model_feature_schema_version: FEATURE_SCHEMA_VERSION,
  model_feature_schema_sha256: FEATURE_SCHEMA_SHA256,
  deployment_scope: 'paper_only',
  production_qualified: false,
  directional_model_entry_policy: 'execute_directional_prediction',
  preregistration: SOURCE_PREREGISTRATION,
};

function uuidV5Url(name: string): string {
  const namespace = Buffer.from(
    '6ba7b8119dad11d180b400c04fd430c8',
    'hex',
  );
  const digest = createHash('sha1').update(namespace).update(name).digest();
  digest.writeUInt8((digest.readUInt8(6) & 0x0f) | 0x50, 6);
  digest.writeUInt8((digest.readUInt8(8) & 0x3f) | 0x80, 8);
  const hex = digest.subarray(0, 16).toString('hex');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

const processConfig = {
  execution: {
    mode: 'live',
    execute_signals: false,
    live_capital: false,
    account_ref: ACCOUNT_REF,
    taker_fee_rate: null,
  },
  raw: {
    btc_realtime_paper: {
      schema_version: 'btc_realtime_paper_process_v3',
      next_experiment_key: RUN_KEY,
      preregistration_sha256: PREREGISTRATION_SHA256,
      strategy,
      runtime,
      paper,
    },
  },
};

const processMetadata = {
  ...sourceProcessMetadata,
  source_process_id: SOURCE_PROCESS_ID,
  source_process_key: SOURCE_PROCESS_KEY,
  account_ref: ACCOUNT_REF,
  live_capital_allowed: false,
  credential_validation_only: true,
  live_execution_authorized: false,
  preregistration: PREREGISTRATION,
};

function assertFrozenIdentity(): void {
  for (const [name, digest] of [
    ['model', MODEL_SHA256],
    ['feature schema', FEATURE_SCHEMA_SHA256],
    ['source preregistration', SOURCE_PREREGISTRATION_SHA256],
    ['pilot preregistration', PREREGISTRATION_SHA256],
  ]) {
    if (!/^[0-9a-f]{64}$/.test(digest)) {
      throw new Error(`refusing to add live pilot with invalid ${name} SHA-256`);
    }
  }

  const calculatedPreregistrationSha256 = createHash('sha256')
    .update(PREREGISTRATION)
    .digest('hex');
  if (calculatedPreregistrationSha256 !== PREREGISTRATION_SHA256) {
    throw new Error(
      'refusing to add live pilot with mismatched preregistration evidence',
    );
  }

  if (uuidV5Url(`polymarket-bot/btc-live/${RUN_KEY}`) !== RUN_ID) {
    throw new Error('refusing to add live pilot with mismatched immutable run identity');
  }

  if (
    PROCESS_ID === SOURCE_PROCESS_ID ||
    PROCESS_KEY === SOURCE_PROCESS_KEY ||
    processConfig.execution.mode !== 'live' ||
    processConfig.execution.execute_signals ||
    processConfig.execution.live_capital ||
    processMetadata.deployment_scope !== 'paper_only' ||
    processMetadata.production_qualified ||
    processMetadata.live_capital_allowed ||
    !processMetadata.credential_validation_only ||
    processMetadata.live_execution_authorized
  ) {
    throw new Error('refusing to add an execution-authorized live pilot');
  }
}

export class AddBtcBoundaryAlignmentLivePilot1785531100000
  implements MigrationInterface
{
  name = 'AddBtcBoundaryAlignmentLivePilot1785531100000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    assertFrozenIdentity();
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);

    const [tables]: Array<{
      process_table: string | null;
      event_table: string | null;
      orders_table: string | null;
      fills_table: string | null;
      fill_identities_table: string | null;
      decisions_table: string | null;
      settlement_table: string | null;
      live_reconciliation_table: string | null;
      account_reconciliation_table: string | null;
    }> = await queryRunner.query(`
      SELECT
        to_regclass('polymarket.trading_processes')::text AS process_table,
        to_regclass('polymarket.trading_process_events')::text AS event_table,
        to_regclass('polymarket.orders')::text AS orders_table,
        to_regclass('polymarket.fills')::text AS fills_table,
        to_regclass('polymarket.fill_identities')::text AS fill_identities_table,
        to_regclass('polymarket.btc_strategy_decisions')::text AS decisions_table,
        to_regclass('polymarket.btc_paper_settlement_ledger')::text AS settlement_table,
        to_regclass('polymarket.live_reconciliation_runs')::text AS live_reconciliation_table,
        to_regclass('polymarket.account_reconciliation_runs')::text AS account_reconciliation_table;
    `);
    if (!tables || Object.values(tables).some((table) => !table)) {
      throw new Error(
        'refusing to add boundary-alignment live pilot: required tables are missing',
      );
    }

    // Serialize the clone identity and immutable run-key proof with process definition/event
    // writers. This mirrors the existing process-definition migrations and makes the collision
    // checks and insert one atomic critical section.
    await queryRunner.query(`
      LOCK TABLE polymarket.trading_processes IN SHARE ROW EXCLUSIVE MODE;
      LOCK TABLE polymarket.trading_process_events IN SHARE MODE;
    `);

    const candidates: Array<{ process_id: string }> = await queryRunner.query(
      `
        SELECT process_id::text
        FROM polymarket.trading_processes
        WHERE process_id = $1::uuid
           OR (
             process_type = 'btc_5m'
             AND process_scope = 'realtime_paper'
             AND process_key = $2
           )
        FOR KEY SHARE;
      `,
      [SOURCE_PROCESS_ID, SOURCE_PROCESS_KEY],
    );
    if (candidates.length === 0) {
      throw new Error(
        'refusing to add boundary-alignment live pilot: exact source process is missing',
      );
    }

    const [source]: Array<{ count: string }> = await queryRunner.query(
      `
        SELECT count(*)::text AS count
        FROM polymarket.trading_processes
        WHERE process_id = $1::uuid
          AND name = $2
          AND process_type = 'btc_5m'
          AND process_scope = 'realtime_paper'
          AND process_key = $3
          AND config = $4::jsonb
          AND metadata = $5::jsonb;
      `,
      [
        SOURCE_PROCESS_ID,
        SOURCE_PROCESS_NAME,
        SOURCE_PROCESS_KEY,
        JSON.stringify(sourceProcessConfig),
        JSON.stringify(sourceProcessMetadata),
      ],
    );
    if (Number(source.count) !== 1) {
      throw new Error(
        'refusing to add boundary-alignment live pilot: source definition has drifted',
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
      throw new Error('refusing to reuse boundary-alignment live pilot identity');
    }

    const [runCollision]: Array<{ count: string }> = await queryRunner.query(
      `
        SELECT (
          (
            SELECT count(*)
            FROM polymarket.trading_processes process
            WHERE process.config #>>
                  '{raw,btc_realtime_paper,next_experiment_key}' = $1
          ) + (
            SELECT count(*)
            FROM polymarket.trading_process_events event
            WHERE event.event_type = 'btc_run_manifest'
              AND event.metadata ->> 'run_key' = $1
          )
        )::text AS count;
      `,
      [RUN_KEY],
    );
    if (Number(runCollision.count) !== 0) {
      throw new Error('refusing to reuse boundary-alignment live pilot run identity');
    }

    const inserted: Array<{ process_id: string }> = await queryRunner.query(
      `
        INSERT INTO polymarket.trading_processes (
          process_id,
          name,
          process_type,
          process_scope,
          process_key,
          status,
          enabled,
          hostname,
          pid,
          version,
          started_at,
          heartbeat_at,
          stopped_at,
          stop_reason,
          last_error,
          config,
          metadata
        )
        SELECT
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
          NULL,
          NULL,
          NULL,
          NULL,
          NULL,
          $4::jsonb,
          $5::jsonb
        FROM polymarket.trading_processes source
        WHERE source.process_id = $6::uuid
          AND source.name = $7
          AND source.process_type = 'btc_5m'
          AND source.process_scope = 'realtime_paper'
          AND source.process_key = $8
          AND source.config = $9::jsonb
          AND source.metadata = $10::jsonb
        RETURNING process_id::text;
      `,
      [
        PROCESS_ID,
        PROCESS_NAME,
        PROCESS_KEY,
        JSON.stringify(processConfig),
        JSON.stringify(processMetadata),
        SOURCE_PROCESS_ID,
        SOURCE_PROCESS_NAME,
        SOURCE_PROCESS_KEY,
        JSON.stringify(sourceProcessConfig),
        JSON.stringify(sourceProcessMetadata),
      ],
    );
    if (inserted.length !== 1 || inserted[0].process_id !== PROCESS_ID) {
      throw new Error('boundary-alignment live pilot insert was not atomic');
    }

    const [validated]: Array<{ count: string }> = await queryRunner.query(
      `
        SELECT count(*)::text AS count
        FROM polymarket.trading_processes process
        JOIN polymarket.trading_processes source
          ON source.process_id = $6::uuid
        WHERE process.process_id = $1::uuid
          AND process.name = $2
          AND process.process_type = 'btc_5m'
          AND process.process_scope = 'realtime_paper'
          AND process.process_key = $3
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
          AND process.config #>> '{execution,mode}' = 'live'
          AND process.config #>> '{execution,execute_signals}' = 'false'
          AND process.config #>> '{execution,live_capital}' = 'false'
          AND process.config #>> '{execution,account_ref}' = $7
          AND process.metadata ->> 'deployment_scope' = 'paper_only'
          AND process.metadata ->> 'production_qualified' = 'false'
          AND process.metadata ->> 'live_capital_allowed' = 'false'
          AND process.metadata ->> 'credential_validation_only' = 'true'
          AND process.metadata ->> 'live_execution_authorized' = 'false'
          AND process.metadata ->> 'source_process_id' = $6::text
          AND process.config #> '{raw,btc_realtime_paper,strategy}'
              = source.config #> '{raw,btc_realtime_paper,strategy}'
          AND process.config #> '{raw,btc_realtime_paper,runtime}'
              = source.config #> '{raw,btc_realtime_paper,runtime}'
          AND process.config #> '{raw,btc_realtime_paper,paper}'
              = source.config #> '{raw,btc_realtime_paper,paper}'
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.trading_process_events evidence
            WHERE evidence.process_id = process.process_id
          )
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.orders evidence
            WHERE evidence.process_id = process.process_id
          )
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.fills evidence
            WHERE evidence.process_id = process.process_id
          )
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.fill_identities evidence
            WHERE evidence.process_id = process.process_id
          )
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.btc_strategy_decisions evidence
            WHERE evidence.run_id = $8::uuid
          )
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.btc_paper_settlement_ledger evidence
            WHERE evidence.process_id = process.process_id
          )
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.live_reconciliation_runs evidence
            WHERE evidence.process_id = process.process_id
          )
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.account_reconciliation_runs evidence
            WHERE evidence.process_id = process.process_id
          );
      `,
      [
        PROCESS_ID,
        PROCESS_NAME,
        PROCESS_KEY,
        JSON.stringify(processConfig),
        JSON.stringify(processMetadata),
        SOURCE_PROCESS_ID,
        ACCOUNT_REF,
        RUN_ID,
      ],
    );
    if (Number(validated.count) !== 1) {
      throw new Error(
        'boundary-alignment live pilot failed post-insert validation',
      );
    }
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    assertFrozenIdentity();
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);

    const candidates: Array<{ process_id: string }> = await queryRunner.query(
      `
        SELECT process_id::text
        FROM polymarket.trading_processes
        WHERE process_id = $1::uuid
           OR (
             process_type = 'btc_5m'
             AND process_scope = 'realtime_paper'
             AND process_key = $2
           )
        FOR UPDATE;
      `,
      [PROCESS_ID, PROCESS_KEY],
    );
    if (candidates.length === 0) {
      return;
    }

    const [eligible]: Array<{ count: string }> = await queryRunner.query(
      `
        SELECT count(*)::text AS count
        FROM polymarket.trading_processes process
        WHERE process.process_id = $1::uuid
          AND process.name = $2
          AND process.process_type = 'btc_5m'
          AND process.process_scope = 'realtime_paper'
          AND process.process_key = $3
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
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.trading_process_events evidence
            WHERE evidence.process_id = process.process_id
          )
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.orders evidence
            WHERE evidence.process_id = process.process_id
          )
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.fills evidence
            WHERE evidence.process_id = process.process_id
          )
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.fill_identities evidence
            WHERE evidence.process_id = process.process_id
          )
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.btc_strategy_decisions evidence
            WHERE evidence.run_id = $6::uuid
          )
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.btc_paper_settlement_ledger evidence
            WHERE evidence.process_id = process.process_id
          )
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.live_reconciliation_runs evidence
            WHERE evidence.process_id = process.process_id
          )
          AND NOT EXISTS (
            SELECT 1 FROM polymarket.account_reconciliation_runs evidence
            WHERE evidence.process_id = process.process_id
          );
      `,
      [
        PROCESS_ID,
        PROCESS_NAME,
        PROCESS_KEY,
        JSON.stringify(processConfig),
        JSON.stringify(processMetadata),
        RUN_ID,
      ],
    );
    if (Number(eligible.count) !== 1) {
      throw new Error(
        'refusing to remove boundary-alignment live pilot after lifecycle, definition, or evidence mutation',
      );
    }

    const deleted: Array<{ process_id: string }> = await queryRunner.query(
      `
        DELETE FROM polymarket.trading_processes
        WHERE process_id = $1::uuid
          AND process_key = $2
        RETURNING process_id::text;
      `,
      [PROCESS_ID, PROCESS_KEY],
    );
    if (deleted.length !== 1 || deleted[0].process_id !== PROCESS_ID) {
      throw new Error(
        'boundary-alignment live pilot failed exact rollback deletion',
      );
    }
  }
}
