#!/usr/bin/env node

import { createHash, randomUUID } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync, existsSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';

const SYMBOL = 'BTCUSDT';
const COINAPI_SYMBOL = 'BINANCE_SPOT_BTC_USDT';
const FEATURE_SCHEMA = 'binance-spot-btcusdt-l2-one-second-features-v1';
const MATERIALIZATION_CONTRACT = 'coinapi-binance-spot-btcusdt-l2-snapshots-v1';
const ROLLING_HORIZONS = [1, 5, 15, 30, 60];
const LOOKBACK_SECONDS = 61;
const MAX_STALE_MS = 1_000;
const MAX_RESPONSE_SNAPSHOTS = 100_000;

function usage(message) {
  console.error(message);
  console.error('usage: materialize-coinapi-binance-spot-l2.mjs --start <UTC ISO> --end <UTC ISO> --apply');
  process.exit(2);
}

function parseArguments(argv) {
  const args = new Map();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (!key.startsWith('--')) usage(`unexpected argument: ${key}`);
    if (key === '--apply') {
      args.set(key, true);
      continue;
    }
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) usage(`missing value for ${key}`);
    args.set(key, value);
    index += 1;
  }
  if (!args.has('--start') || !args.has('--end') || !args.has('--apply')) {
    usage('--start, --end, and --apply are required');
  }
  return { start: parseUtc(args.get('--start')), end: parseUtc(args.get('--end')) };
}

function parseUtc(value) {
  const milliseconds = Date.parse(value);
  if (!Number.isFinite(milliseconds) || !value.endsWith('Z')) {
    usage(`invalid UTC timestamp: ${value}`);
  }
  return milliseconds;
}

function iso(milliseconds) {
  return new Date(milliseconds).toISOString();
}

function secondIso(milliseconds) {
  return iso(Math.floor(milliseconds / 1_000) * 1_000);
}

function sqlLiteral(value) {
  return `'${String(value).replaceAll("'", "''")}'`;
}

function numeric(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`${label} was not finite`);
  return parsed;
}

function fixed(value, label) {
  if (!Number.isFinite(value)) throw new Error(`${label} was not finite`);
  return value.toFixed(10);
}

function psql(sql) {
  return execFileSync('psql', ['-X', '-v', 'ON_ERROR_STOP=1', '-At', '-F', '\t'], {
    input: sql,
    encoding: 'utf8',
    stdio: ['pipe', 'pipe', 'inherit'],
  });
}

function queryExistingSeconds(start, end) {
  const output = psql(`
    SELECT to_char(second_start AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
    FROM polymarket.binance_spot_btcusdt_l2_one_second_features
    WHERE symbol = '${SYMBOL}'
      AND second_start >= ${sqlLiteral(iso(start))}::timestamptz
      AND second_start < ${sqlLiteral(iso(end))}::timestamptz;
  `);
  return new Set(output.split('\n').filter(Boolean));
}

function existingArtifact(logicalKey) {
  const output = psql(`
    SELECT artifact_id, status, actual_checksum, record_count
    FROM polymarket.backfill_artifacts
    WHERE provider = 'coinapi' AND logical_key = ${sqlLiteral(logicalKey)};
  `).trim();
  if (!output) return null;
  const [artifactId, status, checksum, recordCount] = output.split('\t');
  return { artifactId, status, checksum, recordCount };
}

function validateLevels(levels, side) {
  if (!Array.isArray(levels) || levels.length < 20) {
    throw new Error(`${side} did not contain 20 levels`);
  }
  const result = levels.slice(0, 20).map((level, index) => ({
    price: numeric(level.price, `${side}[${index}].price`),
    size: numeric(level.size, `${side}[${index}].size`),
  }));
  for (let index = 0; index < result.length; index += 1) {
    if (result[index].price <= 0 || result[index].size <= 0) {
      throw new Error(`${side}[${index}] was not positive`);
    }
    if (index > 0) {
      const ordered = side === 'bids'
        ? result[index - 1].price > result[index].price
        : result[index - 1].price < result[index].price;
      if (!ordered) throw new Error(`${side} prices were not strictly ordered`);
    }
  }
  return result;
}

function levelsToMap(levels) {
  return new Map(levels.map(({ price, size }) => [price, size]));
}

function sumDepth(levels, count) {
  return levels.slice(0, count).reduce((sum, level) => sum + level.size, 0);
}

function relativeChangeBps(current, previous) {
  return ((current - previous) * 10_000) / previous;
}

function snapshotState(snapshot, sourceUpdateId) {
  if (snapshot.symbol_id !== COINAPI_SYMBOL) {
    throw new Error(`unexpected CoinAPI symbol: ${snapshot.symbol_id}`);
  }
  const sourceMillis = parseUtc(snapshot.time_exchange);
  const receivedMillis = parseUtc(snapshot.time_coinapi);
  if (receivedMillis < sourceMillis || receivedMillis - sourceMillis > MAX_STALE_MS) {
    throw new Error(`snapshot latency was outside [0, ${MAX_STALE_MS}]ms`);
  }
  const bids = validateLevels(snapshot.bids, 'bids');
  const asks = validateLevels(snapshot.asks, 'asks');
  const bestBid = bids[0];
  const bestAsk = asks[0];
  if (bestBid.price >= bestAsk.price) throw new Error('snapshot book was crossed or locked');
  const midpoint = (bestBid.price + bestAsk.price) / 2;
  const topSize = bestBid.size + bestAsk.size;
  const bidDepth5 = sumDepth(bids, 5);
  const bidDepth10 = sumDepth(bids, 10);
  const bidDepth20 = sumDepth(bids, 20);
  const askDepth5 = sumDepth(asks, 5);
  const askDepth10 = sumDepth(asks, 10);
  const askDepth20 = sumDepth(asks, 20);
  const imbalance = (bid, ask) => (bid - ask) / (bid + ask);
  return {
    secondStart: secondIso(receivedMillis),
    sourceEventTimestamp: snapshot.time_exchange,
    providerReceivedAt: snapshot.time_coinapi,
    availableAt: snapshot.time_coinapi,
    sourceUpdateId,
    receivedMillis,
    sourceMillis,
    bids: levelsToMap(bids),
    asks: levelsToMap(asks),
    midpoint,
    microprice: ((bestAsk.price * bestBid.size) + (bestBid.price * bestAsk.size)) / topSize,
    spreadBps: relativeChangeBps(bestAsk.price, bestBid.price),
    bidDepth5,
    askDepth5,
    imbalance5: imbalance(bidDepth5, askDepth5),
    bidDepth10,
    askDepth10,
    imbalance10: imbalance(bidDepth10, askDepth10),
    bidDepth20,
    askDepth20,
    imbalance20: imbalance(bidDepth20, askDepth20),
    bidDepthSlope20: relativeChangeBps(bestBid.price, bids[19].price) / bidDepth20,
    askDepthSlope20: relativeChangeBps(asks[19].price, bestAsk.price) / askDepth20,
    bidDepthConcentration20: bidDepth5 / bidDepth20,
    askDepthConcentration20: askDepth5 / askDepth20,
  };
}

function sideFlow(current, previous) {
  let replenishment = 0;
  let churn = 0;
  const prices = new Set([...current.keys(), ...previous.keys()]);
  for (const price of prices) {
    const delta = (current.get(price) ?? 0) - (previous.get(price) ?? 0);
    if (delta > 0) replenishment += price * delta;
    if (delta < 0) churn += price * -delta;
  }
  return { replenishment, churn };
}

function featureFromState(current, priorByHorizon) {
  const prior1 = priorByHorizon.get(1);
  const bidFlow = sideFlow(current.bids, prior1.bids);
  const askFlow = sideFlow(current.asks, prior1.asks);
  const values = [
    SYMBOL, current.secondStart, current.sourceEventTimestamp, current.providerReceivedAt,
    current.availableAt, String(current.sourceUpdateId), FEATURE_SCHEMA, 'qualified',
    fixed(current.midpoint, 'midpoint'), fixed(current.microprice, 'microprice'),
    fixed(current.spreadBps, 'spread_bps'), fixed(current.bidDepth5, 'bid_depth_5'),
    fixed(current.askDepth5, 'ask_depth_5'), fixed(current.imbalance5, 'imbalance_5'),
    fixed(current.bidDepth10, 'bid_depth_10'), fixed(current.askDepth10, 'ask_depth_10'),
    fixed(current.imbalance10, 'imbalance_10'), fixed(current.bidDepth20, 'bid_depth_20'),
    fixed(current.askDepth20, 'ask_depth_20'), fixed(current.imbalance20, 'imbalance_20'),
    fixed(current.bidDepthSlope20, 'bid_depth_slope_20'),
    fixed(current.askDepthSlope20, 'ask_depth_slope_20'),
    fixed(current.bidDepthConcentration20, 'bid_depth_concentration_20'),
    fixed(current.askDepthConcentration20, 'ask_depth_concentration_20'),
    fixed(bidFlow.replenishment, 'bid_quote_replenishment_1s'),
    fixed(askFlow.replenishment, 'ask_quote_replenishment_1s'),
    fixed(bidFlow.churn, 'bid_quote_churn_1s'), fixed(askFlow.churn, 'ask_quote_churn_1s'),
  ];
  for (const horizon of ROLLING_HORIZONS) {
    const prior = priorByHorizon.get(horizon);
    values.push(
      fixed(relativeChangeBps(current.midpoint, prior.midpoint), `midpoint_change_bps_${horizon}s`),
      fixed(current.spreadBps - prior.spreadBps, `spread_bps_delta_${horizon}s`),
      fixed(relativeChangeBps(current.bidDepth20 + current.askDepth20, prior.bidDepth20 + prior.askDepth20), `depth_20_change_bps_${horizon}s`),
      fixed(current.imbalance20 - prior.imbalance20, `imbalance_20_delta_${horizon}s`),
    );
  }
  return values;
}

function csvEscape(value) {
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function featureColumns() {
  return [
    'symbol', 'second_start', 'source_event_timestamp', 'provider_received_at', 'available_at',
    'source_update_id', 'feature_schema_version', 'quality_status', 'midpoint', 'microprice',
    'spread_bps', 'bid_depth_5', 'ask_depth_5', 'imbalance_5', 'bid_depth_10', 'ask_depth_10',
    'imbalance_10', 'bid_depth_20', 'ask_depth_20', 'imbalance_20', 'bid_depth_slope_20',
    'ask_depth_slope_20', 'bid_depth_concentration_20', 'ask_depth_concentration_20',
    'bid_quote_replenishment_1s', 'ask_quote_replenishment_1s', 'bid_quote_churn_1s',
    'ask_quote_churn_1s', 'midpoint_change_bps_1s', 'spread_bps_delta_1s',
    'depth_20_change_bps_1s', 'imbalance_20_delta_1s', 'midpoint_change_bps_5s',
    'spread_bps_delta_5s', 'depth_20_change_bps_5s', 'imbalance_20_delta_5s',
    'midpoint_change_bps_15s', 'spread_bps_delta_15s', 'depth_20_change_bps_15s',
    'imbalance_20_delta_15s', 'midpoint_change_bps_30s', 'spread_bps_delta_30s',
    'depth_20_change_bps_30s', 'imbalance_20_delta_30s', 'midpoint_change_bps_60s',
    'spread_bps_delta_60s', 'depth_20_change_bps_60s', 'imbalance_20_delta_60s', 'artifact_id',
  ];
}

async function loadPayload(url, archivePath) {
  if (existsSync(archivePath)) return readFileSync(archivePath);
  const temporaryPath = `${archivePath}.${process.pid}.part`;
  const curlConfigPath = join('/tmp', `coinapi-curl-${process.pid}-${randomUUID()}.conf`);
  try {
    writeFileSync(curlConfigPath, [
      'fail-with-body', 'silent', 'show-error', 'location',
      'connect-timeout = 10', 'max-time = 120',
      `header = "X-CoinAPI-Key: ${process.env.COIN_API_KEY}"`,
      `output = "${temporaryPath.replaceAll('"', '\\"')}"`,
      `url = "${url.toString().replaceAll('"', '\\"')}"`,
    ].join('\n'), { mode: 0o600, flag: 'wx' });
    execFileSync('curl', [
      '--config', curlConfigPath,
    ], { stdio: ['ignore', 'ignore', 'inherit'] });
    renameSync(temporaryPath, archivePath);
    return readFileSync(archivePath);
  } finally {
    if (existsSync(temporaryPath)) unlinkSync(temporaryPath);
    if (existsSync(curlConfigPath)) unlinkSync(curlConfigPath);
  }
}

function materializeSnapshots(payload, start, end, existingSeconds) {
  const snapshots = JSON.parse(payload.toString('utf8'));
  if (!Array.isArray(snapshots) || snapshots.length === 0) throw new Error('CoinAPI returned no snapshots');
  if (snapshots.length >= MAX_RESPONSE_SNAPSHOTS) {
    throw new Error(`CoinAPI response reached the ${MAX_RESPONSE_SNAPSHOTS} snapshot cap; split this interval`);
  }
  const selected = new Map();
  let previousSourceMillis = Number.NEGATIVE_INFINITY;
  snapshots.forEach((snapshot, index) => {
    const state = snapshotState(snapshot, index);
    if (state.sourceMillis < previousSourceMillis) throw new Error('CoinAPI exchange timestamps were not monotonic');
    previousSourceMillis = state.sourceMillis;
    const current = selected.get(state.secondStart);
    if (!current || state.receivedMillis >= current.receivedMillis) selected.set(state.secondStart, state);
  });
  const states = [...selected.values()].sort((left, right) => left.receivedMillis - right.receivedMillis);
  const stateBySecond = new Map(states.map((state) => [state.secondStart, state]));
  const features = [];
  for (const current of states) {
    const currentMillis = Date.parse(current.secondStart);
    if (currentMillis < start || currentMillis >= end || existingSeconds.has(current.secondStart)) continue;
    const priorByHorizon = new Map();
    let continuous = true;
    for (const horizon of ROLLING_HORIZONS) {
      const prior = stateBySecond.get(secondIso(currentMillis - horizon * 1_000));
      if (!prior) {
        continuous = false;
        break;
      }
      priorByHorizon.set(horizon, prior);
    }
    if (continuous) features.push(featureFromState(current, priorByHorizon));
  }
  return { snapshots, states, features };
}

function applyFeatures({ artifactId, jobId, logicalKey, checksum, archivePath, sourceUri, start, end, snapshots, states, features }) {
  const csvPath = join(dirname(archivePath), `.${basename(archivePath)}.${artifactId}.csv`);
  const copyRows = features.map((feature) => [...feature, artifactId].map(csvEscape).join(',')).join('\n');
  writeFileSync(csvPath, copyRows.length > 0 ? `${copyRows}\n` : '', { flag: 'wx' });
  const minimumSource = states.reduce((minimum, state) => Math.min(minimum, state.sourceMillis), Number.POSITIVE_INFINITY);
  const maximumSource = states.reduce((maximum, state) => Math.max(maximum, state.sourceMillis), Number.NEGATIVE_INFINITY);
  const metadata = JSON.stringify({
    materialization_contract: MATERIALIZATION_CONTRACT,
    source_market: 'binance-spot',
    symbol: SYMBOL,
    snapshot_limit_levels: 20,
    availability_basis: 'time_coinapi',
    maximum_source_staleness_ms: MAX_STALE_MS,
    rolling_features_require_contiguous_coinapi_seconds: true,
    flow_basis: 'adjacent_snapshot_top_20_quote_delta_proxy',
    source_snapshot_count: snapshots.length,
    candidate_feature_rows: features.length,
    archive_path: archivePath,
    query_range_start: iso(start),
    query_range_end: iso(end),
  });
  const columns = featureColumns().join(', ');
  try {
    psql(`
      BEGIN;
      INSERT INTO polymarket.backfill_jobs (
        job_id, ingester_key, status, requested_at, started_at, completed_at,
        range_start, range_end, idempotency_key, request, progress, checkpoint, summary
      ) VALUES (
        ${sqlLiteral(jobId)}::uuid,
        'binance_spot_btcusdt_l2_one_second_features',
        'completed', now(), now(), now(),
        ${sqlLiteral(iso(start))}::timestamptz, ${sqlLiteral(iso(end))}::timestamptz,
        ${sqlLiteral(`coinapi:${logicalKey}:${checksum}`)},
        ${sqlLiteral(JSON.stringify({ provider: 'coinapi', source_uri: sourceUri }))}::jsonb,
        '{}'::jsonb, '{}'::jsonb, ${sqlLiteral(JSON.stringify({ candidate_feature_rows: features.length }))}::jsonb
      );
      INSERT INTO polymarket.backfill_artifacts (
        artifact_id, job_id, ingester_key, logical_key, provider, source_uri, source_date,
        checksum_algorithm, expected_checksum, actual_checksum, compressed_bytes, record_count,
        minimum_source_timestamp, maximum_source_timestamp, status, metadata,
        created_at, updated_at, completed_at
      ) VALUES (
        ${sqlLiteral(artifactId)}::uuid, ${sqlLiteral(jobId)}::uuid,
        'binance_spot_btcusdt_l2_one_second_features', ${sqlLiteral(logicalKey)}, 'coinapi',
        ${sqlLiteral(sourceUri)}, ${sqlLiteral(iso(start).slice(0, 10))}::date,
        'sha256', ${sqlLiteral(checksum)}, ${sqlLiteral(checksum)},
        ${readFileSync(archivePath).length}, 0,
        ${sqlLiteral(iso(minimumSource))}::timestamptz, ${sqlLiteral(iso(maximumSource))}::timestamptz,
        'ingesting', ${sqlLiteral(metadata)}::jsonb, now(), now(), NULL
      );
      CREATE TEMP TABLE coinapi_feature_stage
        (LIKE polymarket.binance_spot_btcusdt_l2_one_second_features INCLUDING DEFAULTS INCLUDING CONSTRAINTS)
        ON COMMIT DROP;
      \\copy coinapi_feature_stage (${columns}) FROM ${sqlLiteral(csvPath)} WITH (FORMAT csv)
      LOCK TABLE polymarket.binance_spot_btcusdt_l2_one_second_features IN SHARE ROW EXCLUSIVE MODE;
      SELECT decompress_chunk(format('%I.%I', chunk_schema, chunk_name)::regclass, if_compressed => true)
      FROM timescaledb_information.chunks
      WHERE hypertable_schema = 'polymarket'
        AND hypertable_name = 'binance_spot_btcusdt_l2_one_second_features'
        AND is_compressed
        AND range_start < ${sqlLiteral(iso(end))}::timestamptz
        AND range_end > ${sqlLiteral(iso(start))}::timestamptz;
      INSERT INTO polymarket.binance_spot_btcusdt_l2_one_second_features (${columns})
      SELECT ${columns}
      FROM coinapi_feature_stage stage
      WHERE NOT EXISTS (
        SELECT 1
        FROM polymarket.binance_spot_btcusdt_l2_one_second_features existing
        WHERE existing.symbol = stage.symbol AND existing.second_start = stage.second_start
      )
      ORDER BY stage.second_start;
      UPDATE polymarket.backfill_artifacts artifact
      SET status = 'completed', record_count = (
            SELECT count(*)::bigint
            FROM polymarket.binance_spot_btcusdt_l2_one_second_features feature
            WHERE feature.artifact_id = artifact.artifact_id
          ),
          metadata = artifact.metadata || jsonb_build_object('inserted_feature_rows', (
            SELECT count(*)::bigint
            FROM polymarket.binance_spot_btcusdt_l2_one_second_features feature
            WHERE feature.artifact_id = artifact.artifact_id
          )),
          completed_at = now(), updated_at = now()
      WHERE artifact.artifact_id = ${sqlLiteral(artifactId)}::uuid;
      COMMIT;
    `);
  } finally {
    if (existsSync(csvPath)) unlinkSync(csvPath);
  }
}

async function main() {
  const { start, end } = parseArguments(process.argv.slice(2));
  if (end <= start || end - start > 24 * 60 * 60 * 1_000) usage('range must be positive and no longer than 24 hours');
  if (!process.env.COIN_API_KEY) throw new Error('COIN_API_KEY is required');
  const archiveRoot = process.env.POLYMARKET_COINAPI_ARCHIVE_ROOT;
  if (!archiveRoot) throw new Error('POLYMARKET_COINAPI_ARCHIVE_ROOT is required');
  const queryStart = start - LOOKBACK_SECONDS * 1_000;
  const url = new URL(`https://rest.coinapi.io/v1/orderbooks/${COINAPI_SYMBOL}/history`);
  url.searchParams.set('time_start', iso(queryStart));
  url.searchParams.set('time_end', iso(end));
  url.searchParams.set('limit', String(MAX_RESPONSE_SNAPSHOTS));
  url.searchParams.set('limit_levels', '20');
  const startLabel = iso(start).replaceAll(/[-:.]/g, '').replace('Z', 'Z');
  const endLabel = iso(end).replaceAll(/[-:.]/g, '').replace('Z', 'Z');
  const archiveDirectory = resolve(archiveRoot, 'coinapi', 'binance-spot', SYMBOL, iso(start).slice(0, 10));
  mkdirSync(archiveDirectory, { recursive: true });
  const archivePath = join(archiveDirectory, `orderbook-${startLabel}-${endLabel}.json`);
  const payload = await loadPayload(url, archivePath);
  const checksum = createHash('sha256').update(payload).digest('hex');
  const logicalKey = `coinapi:binance-spot:${SYMBOL}:l2-snapshots-v1:${iso(start)}:${iso(end)}`;
  const priorArtifact = existingArtifact(logicalKey);
  if (priorArtifact) {
    if (priorArtifact.status !== 'completed' || priorArtifact.checksum !== checksum) {
      throw new Error(`existing CoinAPI artifact ${priorArtifact.artifactId} conflicts with this immutable source object`);
    }
    console.log(JSON.stringify({ status: 'already_completed', artifact_id: priorArtifact.artifactId, record_count: Number(priorArtifact.recordCount) }));
    return;
  }
  const existingSeconds = queryExistingSeconds(start, end);
  const { snapshots, states, features } = materializeSnapshots(payload, start, end, existingSeconds);
  const artifactId = randomUUID();
  const jobId = randomUUID();
  applyFeatures({ artifactId, jobId, logicalKey, checksum, archivePath, sourceUri: url.toString(), start, end, snapshots, states, features });
  console.log(JSON.stringify({
    status: 'completed', artifact_id: artifactId, raw_archive: archivePath,
    source_snapshots: snapshots.length, selected_one_second_states: states.length,
    existing_seconds_skipped: existingSeconds.size, inserted_feature_rows: features.length,
  }));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
