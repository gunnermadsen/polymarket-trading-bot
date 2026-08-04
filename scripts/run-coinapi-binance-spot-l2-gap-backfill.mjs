#!/usr/bin/env node

import { execFileSync, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const SCRIPT_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const MATERIALIZER = join(SCRIPT_DIRECTORY, 'materialize-coinapi-binance-spot-l2.mjs');
const START_DATE = '2026-04-14';
const END_DATE = '2026-08-02';
const DEFAULT_PARALLELISM = 4;
const MAXIMUM_RETRY_DELAY_MS = 300_000;

function positiveIntegerEnvironment(name, fallback, maximum) {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1 || value > maximum) {
    throw new Error(`${name} must be an integer between 1 and ${maximum}`);
  }
  return value;
}

const PARALLELISM = positiveIntegerEnvironment('COINAPI_BACKFILL_PARALLELISM', DEFAULT_PARALLELISM, 16);

function requireEnvironment(name) {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function psql(sql) {
  return execFileSync('psql', ['-X', '-At'], {
    input: sql,
    encoding: 'utf8',
    stdio: ['pipe', 'pipe', 'inherit'],
  });
}

function incompleteDates() {
  return psql(`
    SELECT source_date
    FROM polymarket.backfill_artifacts
    WHERE provider = 'cryptohftdata'
      AND ingester_key = 'binance_spot_btcusdt_l2_one_second_features'
      AND status = 'completed'
      AND source_date >= DATE '${START_DATE}'
      AND source_date < DATE '${END_DATE}'
      AND record_count < 86400
    ORDER BY source_date;
  `).split('\n').filter(Boolean);
}

function shardsForDate(date) {
  const midnight = Date.parse(`${date}T00:00:00Z`);
  if (!Number.isFinite(midnight)) throw new Error(`invalid source date: ${date}`);
  return Array.from({ length: 24 }, (_, hour) => ({
    start: new Date(midnight + hour * 3_600_000).toISOString().replace('.000Z', 'Z'),
    end: new Date(midnight + (hour + 1) * 3_600_000).toISOString().replace('.000Z', 'Z'),
  }));
}

function completedCoinapiArtifactCovers(start, end) {
  return psql(`
    SELECT EXISTS (
      SELECT 1
      FROM polymarket.backfill_artifacts artifact
      JOIN polymarket.backfill_jobs job ON job.job_id = artifact.job_id
      WHERE artifact.provider = 'coinapi'
        AND artifact.ingester_key = 'binance_spot_btcusdt_l2_one_second_features'
        AND artifact.status = 'completed'
        AND job.range_start <= '${start}'::timestamptz
        AND job.range_end >= '${end}'::timestamptz
    );
  `).trim() === 't';
}

function narrowedMissingRange(shard) {
  if (completedCoinapiArtifactCovers(shard.start, shard.end)) {
    return { skipReason: 'coinapi_range_already_retrieved' };
  }
  const existingSeconds = Number(psql(`
    SELECT count(*)
    FROM polymarket.binance_spot_btcusdt_l2_one_second_features
    WHERE symbol = 'BTCUSDT'
      AND second_start >= '${shard.start}'::timestamptz
      AND second_start < '${shard.end}'::timestamptz;
  `).trim());
  if (existingSeconds >= 3_600) return { skipReason: 'complete_hour' };

  const output = psql(`
    WITH seconds AS (
      SELECT generate_series(
        '${shard.start}'::timestamptz,
        '${shard.end}'::timestamptz - interval '1 second',
        interval '1 second'
      ) AS second_start
    ), missing AS (
      SELECT seconds.second_start
      FROM seconds
      WHERE NOT EXISTS (
        SELECT 1
        FROM polymarket.binance_spot_btcusdt_l2_one_second_features feature
        WHERE feature.symbol = 'BTCUSDT'
          AND feature.second_start = seconds.second_start
      )
    )
    SELECT
      coalesce(to_char(min(second_start) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"'), ''),
      coalesce(to_char((max(second_start) + interval '1 second') AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"'), ''),
      count(*)
    FROM missing;
  `).trim();
  const [start, end, missingSecondsRaw] = output.split('|');
  const missingSeconds = Number(missingSecondsRaw);
  if (!start || !end || missingSeconds === 0) return { skipReason: 'complete_hour' };
  if (completedCoinapiArtifactCovers(start, end)) {
    return { skipReason: 'coinapi_range_already_retrieved' };
  }
  return { start, end, missingSeconds };
}

function runShardOnce(shard, environment) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [MATERIALIZER, '--start', shard.start, '--end', shard.end, '--apply'], {
      env: environment,
      stdio: 'inherit',
    });
    child.once('error', reject);
    child.once('exit', (code, signal) => {
      if (code === 0) resolve();
      else reject(new Error(`CoinAPI shard ${shard.start}–${shard.end} failed (code=${code}, signal=${signal})`));
    });
  });
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function runShard(shard, environment) {
  let attempt = 0;
  for (;;) {
    try {
      await runShardOnce(shard, environment);
      return;
    } catch (error) {
      attempt += 1;
      const retryDelayMs = Math.min(MAXIMUM_RETRY_DELAY_MS, 5_000 * (2 ** Math.min(attempt - 1, 6)));
      console.error(JSON.stringify({
        status: 'retrying', start: shard.start, end: shard.end, attempt,
        retry_delay_ms: retryDelayMs,
        error: error instanceof Error ? error.message : String(error),
      }));
      await delay(retryDelayMs);
    }
  }
}

async function main() {
  const apiKey = requireEnvironment('COIN_API_KEY');
  requireEnvironment('POLYMARKET_COINAPI_ARCHIVE_ROOT');
  const tasks = incompleteDates().flatMap(shardsForDate);
  let nextTask = 0;
  let completed = 0;
  const environment = { ...process.env, COIN_API_KEY: apiKey };

  console.log(JSON.stringify({ status: 'started', scheduled_shards: tasks.length, parallelism: PARALLELISM }));
  const worker = async () => {
    while (nextTask < tasks.length) {
      const shard = tasks[nextTask++];
      const missingRange = narrowedMissingRange(shard);
      if (!missingRange.skipReason) {
        console.log(JSON.stringify({
          status: 'scheduled_missing_range', hour_start: shard.start,
          start: missingRange.start, end: missingRange.end,
          missing_seconds: missingRange.missingSeconds,
        }));
        await runShard(missingRange, environment);
      } else {
        console.log(JSON.stringify({
          status: 'skipped_hour', reason: missingRange.skipReason,
          start: shard.start, end: shard.end,
        }));
      }
      completed += 1;
      console.log(JSON.stringify({ status: 'progress', completed_shards: completed, scheduled_shards: tasks.length, end: shard.end }));
    }
  };
  await Promise.all(Array.from({ length: Math.min(PARALLELISM, tasks.length) }, worker));
  console.log(JSON.stringify({ status: 'completed', completed_shards: completed, scheduled_shards: tasks.length }));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
