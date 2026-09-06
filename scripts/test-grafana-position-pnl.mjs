import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

// Exercise the provisioned SQL against VALUES fixtures only; no database tables or writes.
const dashboard = JSON.parse(readFileSync(new URL('../common/configs/grafana/dashboards/trading-pnl-market-metrics.json', import.meta.url)));
assert(!JSON.stringify(dashboard).includes('orderbook_snapshots'));
const pnlStat = dashboard.panels.find(p => p.id === 4);
const selectedFields = new RegExp(pnlStat.options.reduceOptions.fields.slice(1, -1));
assert(selectedFields.test('Current Position Snapshot Unrealized PnL'));
assert(selectedFields.test('Current Market Entry Orders'));
assert(!selectedFields.test('Current Open Cost Basis'));
const stat = dashboard.panels.find(p => p.id === 4).targets[0].rawSql;
const table = dashboard.panels.find(p => p.id === 2).targets[0].rawSql;
const joins = sql => sql.slice(sql.indexOf('  LEFT JOIN LATERAL ('), sql.indexOf('  ) mark ON true') + '  ) mark ON true'.length);
assert.equal(joins(stat), joins(table));
const scenarios = [
  { name: 'live price uses process quantity and fees', expected: 1.9 },
  { name: 'paper has no account valuation', live: false, expected: null },
  { name: 'stale reconciliation', start: -61, expected: null },
  { name: 'other process', owner: 2, expected: null },
  { name: 'other account', account: 'other', expected: null },
  { name: 'other token', token: 'other', expected: null },
  { name: 'other condition', condition: 'other', expected: null },
  { name: 'snapshot precedes fill', snapshot: -40, expected: null },
  { name: 'future snapshot', snapshot: 1, expected: null },
  { name: 'mismatch', mismatches: 1, expected: null },
  { name: 'failed reconciliation', status: 'error', expected: null },
  { name: 'dry run', dry: true, expected: null },
  { name: 'missing price', price: 'NULL::numeric', expected: null },
  { name: 'empty account position', size: 0, expected: null },
  { name: 'mixed priced and unpriced is incomplete', mixed: true, expected: null },
  { name: 'no current exposure is zero', empty: true, expected: 0 },
];
for (const s of scenarios) {
  const sql = `WITH clock AS (SELECT timestamptz '2026-09-05 12:00:00+00' AS at),
  current_entries AS (
    SELECT 1 AS process_id,'token'::text AS token_id,'condition'::text AS condition_id,
      ${s.live ?? true} AS is_live,at-interval '30 seconds' AS filled_at,
      10::numeric AS filled_size,4::numeric AS entry_cost,0.1::numeric AS entry_fees
    FROM clock WHERE ${!s.empty}
    ${s.mixed ? "UNION ALL SELECT 2,'other','condition',true,at-interval '30 seconds',10,4,0.1 FROM clock" : ''}
  ), fixture_runs AS (
    SELECT ${s.owner ?? 1} AS process_id,1 AS run_id,'account'::text AS account_address,
      at+interval '${s.start ?? -20} seconds' AS started_at,at-interval '5 seconds' AS completed_at,
      '${s.status ?? 'completed'}'::text AS status,${s.mismatches ?? 0} AS mismatches_found,${s.dry ?? false} AS dry_run FROM clock
  ), fixture_snapshots AS (
    SELECT 1 AS snapshot_id,'${s.account ?? 'account'}'::text AS account_address,
      '${s.token ?? 'token'}'::text AS token_id,'${s.condition ?? 'condition'}'::text AS market_id,
      ${s.price ?? '0.6::numeric'} AS current_price,${s.size ?? 100}::numeric AS size,
      at+interval '${s.snapshot ?? -10} seconds' AS snapshot_at,'poll'::text AS source FROM clock
  ), marked AS (
    SELECT e.*,mark.mark_price FROM current_entries e CROSS JOIN clock c
    ${joins(stat).replaceAll('polymarket.account_reconciliation_runs', 'fixture_runs').replaceAll('polymarket.account_position_snapshots', 'fixture_snapshots')}
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
