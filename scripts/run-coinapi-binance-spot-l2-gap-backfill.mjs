#!/usr/bin/env node

import { execFileSync, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const SCRIPT_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const MATERIALIZER = join(SCRIPT_DIRECTORY, 'materialize-coinapi-binance-spot-l2.mjs');
const START_DATE = '2026-04-14';
const END_DATE = '2026-08-02';
const PARALLELISM = 2;
const MAXIMUM_RETRY_DELAY_MS = 300_000;

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
      await runShard(shard, environment);
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
