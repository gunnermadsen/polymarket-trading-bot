import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

// Exercise the provisioned SQL against VALUES fixtures only; no database tables or writes.
const dashboard = JSON.parse(readFileSync(new URL('../common/configs/grafana/dashboards/trading-pnl-market-metrics.json', import.meta.url)));
const pnlStat = dashboard.panels.find(p => p.id === 4);
const selectedFields = new RegExp(pnlStat.options.reduceOptions.fields.slice(1, -1));
assert(selectedFields.test('Current Position Snapshot Unrealized PnL'));
assert(selectedFields.test('Current Market Entry Orders'));
assert(!selectedFields.test('Current Open Cost Basis'));
const stat = pnlStat.targets[0].rawSql;
const table = dashboard.panels.find(p => p.id === 2).targets[0].rawSql;
const markJoin = sql => sql.slice(sql.indexOf('  LEFT JOIN LATERAL ('), sql.indexOf('  ) mark ON true') + '  ) mark ON true'.length);
assert.equal(markJoin(stat), markJoin(table));
assert(stat.includes('polymarket.btc_five_minute_orderbook_snapshots'));
assert(!markJoin(stat).includes('account_reconciliation_runs'));

const scenarios = [
  { name: 'paper position uses market mark', live: false, expected: 1.9 },
  { name: 'live position uses the same market mark', live: true, expected: 1.9 },
  { name: 'other market', market: 'other', expected: null },
  { name: 'other token', token: 'other', expected: null },
  { name: 'stale sample', sampled: -3, expected: null },
  { name: 'stale source timestamp', source: -3, expected: null },
  { name: 'future received timestamp', received: 1, expected: null },
  { name: 'missing bid', bid: 'NULL::numeric', expected: null },
  { name: 'mixed priced and unpriced is incomplete', mixed: true, expected: null },
  { name: 'no current exposure is zero', empty: true, expected: 0 },
];

for (const s of scenarios) {
  const sql = `WITH clock AS (SELECT timestamptz '2026-09-05 12:00:00+00' AS at),
  current_entries AS (
    SELECT 1 AS process_id,'market'::text AS market_id,'token'::text AS token_id,
      ${s.live ?? false} AS is_live,at-interval '30 seconds' AS filled_at,
      10::numeric AS filled_size,4::numeric AS entry_cost,0.1::numeric AS entry_fees
    FROM clock WHERE ${!s.empty}
    ${s.mixed ? "UNION ALL SELECT 2,'market','unpriced',false,at-interval '30 seconds',10,4,0.1 FROM clock" : ''}
  ), fixture_marks AS (
    SELECT '${s.market ?? 'market'}'::text AS market_id,'${s.token ?? 'token'}'::text AS token_id,
      ${s.bid ?? '0.6::numeric'} AS best_bid,at+interval '${s.sampled ?? -1} seconds' AS sampled_at,
      at+interval '${s.source ?? -1} seconds' AS source_timestamp,
      at+interval '${s.received ?? -1} seconds' AS received_at,1::bigint AS ingest_sequence FROM clock
  ), marked AS (
    SELECT e.*,mark.mark_price FROM current_entries e CROSS JOIN clock c
    ${markJoin(stat).replaceAll('polymarket.btc_five_minute_orderbook_snapshots', 'fixture_marks')}
  ) ${stat.slice(stat.indexOf('SELECT CASE WHEN COUNT(*)=0'))}`;
  const result = spawnSync('docker', ['exec', '-i', '-e', 'PGOPTIONS=-c statement_timeout=3000 -c default_transaction_read_only=on',
    'timescaledb-0', 'psql', '-X', '-U', 'postgres', '-d', 'polymarket', '-v', 'ON_ERROR_STOP=1', '-At'],
  { input: `SELECT row_to_json(r) FROM (${sql.replace(/;\s*$/, '')}) r;`, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const row = JSON.parse(result.stdout);
  assert.equal(row['Current Position Snapshot Unrealized PnL'], s.expected, s.name);
  assert.equal(row['Current Open Cost Basis'], s.empty ? 0 : s.mixed ? 8.2 : 4.1, s.name);
  console.log(`PASS ${s.name}`);
}
