# BTC 5-minute paper v10 evidence

This directory freezes the final evidence for experiment
`ba68b0ff-6192-5648-8d8b-244e60282011` and process
`939b77d4-805e-458e-9b44-9a7e8e80e77f`.

- Experiment key: `btc-5m-chainlink-fair-value-paper-v10`
- Started: `2026-07-13 15:51:04.125326+00`
- Stopped: `2026-07-13 17:04:26.631904+00`
- Stop reason: `service_shutdown`
- Frozen config hash: `6c9cd59d639b671aa287c9c517f4d8ba8168d36c49bd317303e7bb79b437a900`
- Built image: `sha256:682e86f6fbc0df0a4a430b3c932adde1d1153ff13d0eb974e225b3714095935e`
- Final audit SHA-256: `37d2d85a10d80c005d35ae107944551d1afa49a7a982655155a86a4d1f454396`

## Exit result

The overall result is a qualified failure because the pre-registered official
resolution SLO failed for one of 13 complete windows. This result must not be
relabelled by lengthening the grace period after inspection.

- Realtime completeness: pass; 13/13 consecutive full windows, 300 snapshots
  and 300 decisions per window, with valid boundary labels.
- Safety/config: pass; paper enabled, live capital disabled, ML execution
  authority disabled, and zero live fills.
- Paper execution parity: pass; 15 order plans, 12 atomic paper fills, exact
  latency/depth/fee causality, and zero parity mismatches.
- Settlement coverage: 13/13 official outcomes eventually recovered and in
  agreement, but only 12/13 arrived within 120 seconds.
- ML shadow contract: pass; 4,370 fill-probability vectors/predictions, 4,370
  toxicity vectors/predictions, and 393 fair-value residual
  vectors/predictions, all at 100% key coverage with zero missing, duplicate,
  lineage, contract, authority, or queue/failure errors.
- Accounting: 12/12 fills officially resolved; gross P&L `4.550000`, fees
  `0.725935`, and independently recomputed net P&L `3.824065`, matching the
  stored experiment exactly.
- Profitability: not established. Twelve filled trades are far too few for a
  holdout, regime, drawdown, or confidence-interval claim.

Market `2896589` (16:35-16:40 UTC) remained unresolved upstream beyond the
120-second audit SLO. Durable REST reconciliation recovered the official Up
outcome at `16:46:34.005387+00`, 394.005 seconds after close. The durable watch
and final official provenance are correct; the late receipt remains an SLO
failure.

## Audit and database incident

The first post-stop audit used an unsafe planner shape over ML lineage and its
PostgreSQL backend was killed with signal 9. PostgreSQL recovered and the v10
rows remained intact. The audit was rewritten to prevent vector/prediction
cross-products and to scope the global diagnostic. The final complete audit
then passed under a session `work_mem` of 2 MB with a 60-second statement
timeout; no query timed out or restarted the database.

TimescaleDB now has a 2 GB cgroup ceiling with a matching conservative profile:
512 MB shared buffers, 1,536 MB effective cache, 256 MB maintenance memory,
64 MB autovacuum memory, approximately 10 MB work memory, 25 connections, and
bounded parallel/background workers. After the final audit and Grafana query
checks, cgroup peak memory was 712,851,456 bytes and all OOM counters were zero.
All 23 Timescale jobs reported `Success`.

Grafana was reprovisioned with a 15-second dashboard refresh and a five-open,
two-idle PostgreSQL pool. Representative current-stats, current-unrealized, and
24-hour realized-history panels returned datasource status 200. Current totals
correctly read zero because v10 is stopped; v10 remains visible in history.

## Files

- `readiness-audit-final.txt`: complete SELECT-only exit-gate audit.
- `readiness-running.json`: authenticated readiness snapshot captured before
  stop.
- `paper-experiment-running.json`: authenticated experiment snapshot captured
  before stop.
- `metrics-running.prom`: cumulative runtime metrics captured before stop.
- `health-running.json`: runtime health captured before stop.
