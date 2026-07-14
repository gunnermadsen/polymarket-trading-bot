-- Deterministic BTC five-minute realtime-paper experiment report.
--
-- This script is SELECT-only and is intended to be run with psql. It never creates
-- temporary objects or mutates experiment state. The checked-in preregistration at
-- docs/experiments/btc-5m-chainlink-fair-value-20260713-c/preregistration.json
-- owns the thresholds. scripts/btc-paper-experiment-audit.sh reads that manifest and supplies
-- the same values explicitly. Defaults below make direct diagnostic execution useful.
--
-- Required:
--   -v experiment_id=<immutable experiment uuid>
-- Optional:
--   -v audit_mode=daily|final

\set ON_ERROR_STOP on
\pset pager off

\if :{?experiment_id}
\else
  \echo 'experiment_id is required; refusing to select the newest experiment implicitly'
  \quit 3
\endif

\if :{?experiment_key}
\else
  \set experiment_key btc-5m-chainlink-fair-value-20260713-c
\endif
\if :{?process_key}
\else
  \set process_key btc-5m-chainlink-fair-value-paper
\endif
\if :{?audit_mode}
\else
  \set audit_mode daily
\endif
\if :{?preregistration_sha256}
\else
  \set preregistration_sha256 unpinned-direct-diagnostic-run
\endif
\if :{?expected_compiled_source_identity}
\else
  \set expected_compiled_source_identity unpinned-direct-diagnostic-run
\endif
\if :{?expected_experiment_config_contract}
\else
  \set expected_experiment_config_contract '{}'
\endif
\if :{?required_complete_windows}
\else
  \set required_complete_windows 2000
\endif
\if :{?minimum_coverage_pct}
\else
  \set minimum_coverage_pct 99.5
\endif
\if :{?minimum_official_slo_pct}
\else
  \set minimum_official_slo_pct 99.5
\endif
\if :{?minimum_snapshots_per_window}
\else
  \set minimum_snapshots_per_window 285
\endif
\if :{?maximum_snapshot_gap_seconds}
\else
  \set maximum_snapshot_gap_seconds 5
\endif
\if :{?official_resolution_grace_seconds}
\else
  \set official_resolution_grace_seconds 120
\endif
\if :{?resolution_watch_retention_seconds}
\else
  \set resolution_watch_retention_seconds 3600
\endif
\if :{?minimum_qualifying_fills}
\else
  \set minimum_qualifying_fills 100
\endif
\if :{?minimum_distinct_utc_days}
\else
  \set minimum_distinct_utc_days 7
\endif
\if :{?bootstrap_replicates}
\else
  \set bootstrap_replicates 10000
\endif
\if :{?bootstrap_seed}
\else
  \set bootstrap_seed 20260713
\endif
\if :{?bootstrap_lower_quantile}
\else
  \set bootstrap_lower_quantile 0.025
\endif
\if :{?bootstrap_upper_quantile}
\else
  \set bootstrap_upper_quantile 0.975
\endif
\if :{?fee_stress_multiplier}
\else
  \set fee_stress_multiplier 1.25
\endif
\if :{?missed_best_fill_fraction}
\else
  \set missed_best_fill_fraction 0.10
\endif
\if :{?probability_uncertainty_multiplier}
\else
  \set probability_uncertainty_multiplier 1.25
\endif
\if :{?maximum_drawdown_usd}
\else
  \set maximum_drawdown_usd 100
\endif
\if :{?minimum_worst_trade_pnl_usd}
\else
  \set minimum_worst_trade_pnl_usd -5.10
\endif
\if :{?maximum_concentration_fraction}
\else
  \set maximum_concentration_fraction 0.80
\endif
\if :{?pnl_reconciliation_tolerance_usd}
\else
  \set pnl_reconciliation_tolerance_usd 0.000001
\endif

BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL statement_timeout = '120s';
SET LOCAL lock_timeout = '5s';
SET LOCAL work_mem = '4MB';
SET LOCAL TIME ZONE 'UTC';

SELECT EXISTS (
  SELECT 1
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
) AS found \gset selected_

\if :selected_found
\else
  \echo 'selected experiment_id does not exist'
  ROLLBACK;
  \quit 4
\endif

\echo '== realtime-paper experiment preregistered parameters =='
SELECT :'audit_mode' AS audit_mode,
       :'experiment_key' AS preregistered_experiment_key,
       :'process_key' AS stable_process_key,
       :'preregistration_sha256' AS preregistration_sha256,
       :'required_complete_windows'::bigint AS required_complete_windows,
       :'minimum_coverage_pct'::numeric AS minimum_window_and_label_coverage_pct,
       :'minimum_official_slo_pct'::numeric AS minimum_official_resolution_slo_pct,
       :'minimum_qualifying_fills'::bigint AS minimum_qualifying_filled_orders,
       :'minimum_distinct_utc_days'::bigint AS minimum_distinct_utc_days,
       :'bootstrap_replicates'::integer AS bootstrap_replicates,
       :'fee_stress_multiplier'::numeric AS fee_stress_multiplier,
       :'missed_best_fill_fraction'::numeric AS missed_best_fill_fraction,
       :'probability_uncertainty_multiplier'::numeric AS probability_uncertainty_multiplier,
       :'maximum_drawdown_usd'::numeric AS maximum_drawdown_usd,
       :'maximum_concentration_fraction'::numeric AS maximum_concentration_fraction;

\echo '== Immutable experiment identity and safety contract =='
WITH selected AS (
  SELECT e.*, p.process_type, p.process_scope, p.process_key,
         p.status AS process_status, p.enabled AS process_enabled,
         p.heartbeat_at, e.config AS process_config
  FROM polymarket.btc_paper_experiments e
  JOIN polymarket.trading_processes p ON p.process_id = e.process_id
  WHERE e.experiment_id = NULLIF(:'experiment_id', '')::uuid
)
SELECT experiment_id, name, status AS experiment_status, process_id,
       process_status, process_enabled, heartbeat_at, started_at, stopped_at,
       stop_reason, strategy_version, feature_schema_version, config_hash,
       process_config #>> '{raw,build,compiled_source_identity}' AS compiled_source_identity,
       process_config #>> '{raw,preregistration_sha256}' AS recorded_preregistration_sha256,
       process_config #>> '{raw,preregistration_sha256}' AS raw_preregistration_sha256,
       process_config #>> '{raw,pipeline_version}' AS pipeline_version,
       process_config #>> '{execution,mode}' AS execution_mode,
       (process_config #>> '{execution,live_capital}')::boolean AS live_capital,
       (process_config #>> '{raw,paper,execution_enabled}')::boolean AS paper_enabled,
       (process_config #>> '{raw,ml_shadow,execution_authority}')::boolean
         AS ml_execution_authority,
       process_config #>> '{raw,paper,venue,starting_collateral_usd}'
         AS starting_collateral_usd,
       ((process_config #- '{raw,preregistration_sha256}')
         = :'expected_experiment_config_contract'::jsonb) AS exact_frozen_config_contract_match,
       (
         name = :'experiment_key'
         AND process_key = :'process_key'
         AND process_type = 'btc_5m'
         AND process_scope = 'realtime_paper'
         AND strategy_version = 'btc_5m_chainlink_fair_value_v1'
         AND feature_schema_version = 'btc_5m_features_v2'
         AND process_config #>> '{raw,pipeline_version}' = 'btc_realtime_paper_pipeline_v11'
         AND process_config #>> '{execution,mode}' = 'paper'
         AND NOT (process_config #>> '{execution,live_capital}')::boolean
         AND (process_config #>> '{execution,execute_signals}')::boolean
         AND (process_config #>> '{raw,paper,execution_enabled}')::boolean
         AND (process_config #>> '{raw,ml_shadow,ml_a_enabled}')::boolean
         AND (process_config #>> '{raw,ml_shadow,ml_b_enabled}')::boolean
         AND NOT (process_config #>> '{raw,ml_shadow,execution_authority}')::boolean
         AND (process_config #>> '{raw,paper,venue,starting_collateral_usd}')::numeric=1000
         AND (
           (process_config #>> '{raw,runtime,strategy_interval,secs}')::numeric*1000
           +(process_config #>> '{raw,runtime,strategy_interval,nanos}')::numeric/1000000
         )=1000
         AND (process_config #>> '{raw,strategy,target_size}')::numeric = 5
         AND (process_config #>> '{raw,paper,venue,visible_depth_haircut}')::numeric = 0.80
         AND (
           (process_config #>> '{raw,paper,venue,arrival_latency,secs}')::numeric * 1000
           + (process_config #>> '{raw,paper,venue,arrival_latency,nanos}')::numeric / 1000000
         ) = 150
         AND (process_config #>> '{raw,runtime,official_resolution_audit_grace,secs}')::bigint
               = :'official_resolution_grace_seconds'::bigint
         AND (process_config #>> '{raw,runtime,official_resolution_watch_retention,secs}')::bigint
               = :'resolution_watch_retention_seconds'::bigint
         AND process_config #>> '{raw,preregistration_sha256}'=:'preregistration_sha256'
         AND coalesce(process_config #>> '{raw,build,compiled_source_identity}','')<>''
         AND process_config #>> '{raw,build,compiled_source_identity}'
               =:'expected_compiled_source_identity'
         AND (process_config #- '{raw,preregistration_sha256}')
               = :'expected_experiment_config_contract'::jsonb
       ) AS identity_and_safety_gate_pass
FROM selected;

WITH selected AS (
  SELECT e.*, p.process_type, p.process_scope, p.process_key,
         e.config AS process_config
  FROM polymarket.btc_paper_experiments e
  JOIN polymarket.trading_processes p ON p.process_id = e.process_id
  WHERE e.experiment_id = NULLIF(:'experiment_id', '')::uuid
)
SELECT experiment_id,
       name AS experiment_name,
       status AS experiment_status,
       process_id,
       (
         name = :'experiment_key'
         AND process_key = :'process_key'
         AND process_type = 'btc_5m'
         AND process_scope = 'realtime_paper'
         AND strategy_version = 'btc_5m_chainlink_fair_value_v1'
         AND feature_schema_version = 'btc_5m_features_v2'
         AND process_config #>> '{raw,pipeline_version}' = 'btc_realtime_paper_pipeline_v11'
         AND process_config #>> '{execution,mode}' = 'paper'
         AND NOT (process_config #>> '{execution,live_capital}')::boolean
         AND (process_config #>> '{execution,execute_signals}')::boolean
         AND (process_config #>> '{raw,paper,execution_enabled}')::boolean
         AND (process_config #>> '{raw,ml_shadow,ml_a_enabled}')::boolean
         AND (process_config #>> '{raw,ml_shadow,ml_b_enabled}')::boolean
         AND NOT (process_config #>> '{raw,ml_shadow,execution_authority}')::boolean
         AND (process_config #>> '{raw,paper,venue,starting_collateral_usd}')::numeric=1000
         AND (
           (process_config #>> '{raw,runtime,strategy_interval,secs}')::numeric*1000
           +(process_config #>> '{raw,runtime,strategy_interval,nanos}')::numeric/1000000
         )=1000
         AND (process_config #>> '{raw,strategy,target_size}')::numeric = 5
         AND (process_config #>> '{raw,paper,venue,visible_depth_haircut}')::numeric = 0.80
         AND (
           (process_config #>> '{raw,paper,venue,arrival_latency,secs}')::numeric * 1000
           + (process_config #>> '{raw,paper,venue,arrival_latency,nanos}')::numeric / 1000000
         ) = 150
         AND (process_config #>> '{raw,runtime,official_resolution_audit_grace,secs}')::bigint
               = :'official_resolution_grace_seconds'::bigint
         AND (process_config #>> '{raw,runtime,official_resolution_watch_retention,secs}')::bigint
               = :'resolution_watch_retention_seconds'::bigint
         AND process_config #>> '{raw,preregistration_sha256}'=:'preregistration_sha256'
         AND coalesce(process_config #>> '{raw,build,compiled_source_identity}','')<>''
         AND process_config #>> '{raw,build,compiled_source_identity}'
               =:'expected_compiled_source_identity'
         AND (process_config #- '{raw,preregistration_sha256}')
               = :'expected_experiment_config_contract'::jsonb
       ) AS identity_gate_pass
FROM selected \gset identity_

\echo '== Expected-window completeness and official-resolution coverage =='
WITH selected AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
), bounds AS (
  SELECT s.*,
         to_timestamp(ceil(extract(epoch FROM started_at) / 300.0) * 300.0) AS aligned_start,
         least(coalesce(stopped_at, now()), now())
           - :'official_resolution_grace_seconds'::double precision * interval '1 second'
             AS eligible_cutoff
  FROM selected s
), expected_windows AS (
  SELECT b.experiment_id, b.process_id, g AS window_start,
         g + interval '5 minutes' AS window_end
  FROM bounds b
  CROSS JOIN LATERAL generate_series(
    b.aligned_start,
    to_timestamp(
      floor((extract(epoch FROM b.eligible_cutoff) - 300.0) / 300.0) * 300.0
    ),
    interval '5 minutes'
  ) g
  WHERE b.eligible_cutoff >= b.aligned_start + interval '5 minutes'
), snapshot_sequence AS (
  SELECT s.window_start, s.window_end, s.snapshot_id, s.feature_as_of,
         s.readiness_status,
         extract(epoch FROM (
           s.feature_as_of - lag(s.feature_as_of) OVER (
             PARTITION BY s.window_start ORDER BY s.feature_as_of, s.snapshot_id
           )
         )) AS gap_seconds
  FROM selected e
  JOIN polymarket.btc_feature_snapshots s
    ON s.features->>'process_id' = e.process_id::text
   AND s.feature_as_of >= e.started_at
   AND s.feature_as_of <= least(coalesce(e.stopped_at, now()), now())
), snapshot_stats AS (
  SELECT window_start, min(window_end) AS window_end,
         count(*) AS snapshots,
         count(*) FILTER (WHERE readiness_status = 'ready') AS ready_snapshots,
         min(feature_as_of) AS first_snapshot_at,
         max(feature_as_of) AS last_snapshot_at,
         coalesce(max(gap_seconds), 0) AS maximum_gap_seconds
  FROM snapshot_sequence
  GROUP BY window_start
), decision_stats AS (
  SELECT s.window_start, count(*) AS decisions
  FROM selected e
  JOIN polymarket.btc_strategy_decisions d ON d.experiment_id = e.experiment_id
  JOIN polymarket.btc_feature_snapshots s ON s.snapshot_id = d.snapshot_id
  GROUP BY s.window_start
), market_stats AS (
  SELECT w.window_start,
         count(m.market_id) AS matching_markets,
         count(m.market_id) FILTER (
           WHERE m.validation_status = 'valid'
         ) AS valid_markets,
         count(l.market_id) AS local_labels,
         count(m.market_id) FILTER (
           WHERE m.official_outcome IN ('up','down')
             AND m.official_winning_token_id IN (m.up_token_id, m.down_token_id)
             AND m.official_resolved_at >= m.window_end
             AND m.official_resolution_source IN ('clob_websocket','clob_rest_reconciliation')
             AND m.official_resolution_received_at >= m.window_end
             AND jsonb_typeof(m.official_resolution_payload) = 'object'
             AND rw.status IN ('resolved','resolved_late')
             AND rw.resolution_source = m.official_resolution_source
             AND rw.resolution_received_at = m.official_resolution_received_at
         ) AS official_labels,
         count(m.market_id) FILTER (
           WHERE m.official_outcome IN ('up','down')
             AND m.official_winning_token_id IN (m.up_token_id, m.down_token_id)
             AND m.official_resolved_at >= m.window_end
             AND m.official_resolution_source IN ('clob_websocket','clob_rest_reconciliation')
             AND m.official_resolution_received_at >= m.window_end
             AND m.official_resolution_received_at <=
                   m.window_end
                   + :'official_resolution_grace_seconds'::double precision * interval '1 second'
             AND jsonb_typeof(m.official_resolution_payload) = 'object'
             AND rw.status IN ('resolved','resolved_late')
             AND rw.resolution_source = m.official_resolution_source
             AND rw.resolution_received_at = m.official_resolution_received_at
         ) AS official_labels_within_slo
  FROM expected_windows w
  LEFT JOIN polymarket.btc_interval_markets m
    ON m.window_start = w.window_start AND m.window_end = w.window_end
  LEFT JOIN polymarket.btc_market_labels l ON l.market_id = m.market_id
  LEFT JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id = m.market_id
  GROUP BY w.window_start
), window_results AS (
  SELECT w.window_start, w.window_end,
         coalesce(s.snapshots, 0) AS snapshots,
         coalesce(d.decisions, 0) AS decisions,
         coalesce(s.ready_snapshots, 0) AS ready_snapshots,
         s.first_snapshot_at, s.last_snapshot_at,
         coalesce(s.maximum_gap_seconds, 0) AS maximum_gap_seconds,
         coalesce(m.valid_markets, 0) AS valid_markets,
         coalesce(m.local_labels, 0) AS local_labels,
         coalesce(m.official_labels, 0) AS official_labels,
         coalesce(m.official_labels_within_slo, 0) AS official_labels_within_slo,
         (
           coalesce(s.snapshots, 0) >= :'minimum_snapshots_per_window'::bigint
           AND coalesce(d.decisions, 0) = coalesce(s.snapshots, 0)
           AND coalesce(s.ready_snapshots, 0) > 0
           AND s.first_snapshot_at <= w.window_start + interval '5 seconds'
           AND s.last_snapshot_at >= w.window_end - interval '5 seconds'
           AND coalesce(s.maximum_gap_seconds, 0) <= :'maximum_snapshot_gap_seconds'::numeric
           AND coalesce(m.valid_markets, 0) = 1
           AND coalesce(m.local_labels, 0) = 1
         ) AS window_and_label_complete,
         (coalesce(m.official_labels_within_slo, 0) = 1) AS official_slo_complete
  FROM expected_windows w
  LEFT JOIN snapshot_stats s ON s.window_start = w.window_start
  LEFT JOIN decision_stats d ON d.window_start = w.window_start
  LEFT JOIN market_stats m ON m.window_start = w.window_start
)
SELECT count(*) AS expected_complete_windows,
       count(*) FILTER (WHERE window_and_label_complete) AS complete_windows,
       count(*) FILTER (WHERE official_slo_complete) AS official_windows_within_slo,
       round(100.0 * count(*) FILTER (WHERE window_and_label_complete)
             / nullif(count(*), 0), 4) AS window_and_label_coverage_pct,
       round(100.0 * count(*) FILTER (WHERE official_slo_complete)
             / nullif(count(*), 0), 4) AS official_resolution_slo_coverage_pct,
       min(window_start) AS first_expected_window,
       max(window_end) AS last_expected_window_end,
       (
         count(*) >= :'required_complete_windows'::bigint
         AND 100.0 * count(*) FILTER (WHERE window_and_label_complete)
               / nullif(count(*), 0) >= :'minimum_coverage_pct'::numeric
         AND 100.0 * count(*) FILTER (WHERE official_slo_complete)
               / nullif(count(*), 0) >= :'minimum_official_slo_pct'::numeric
       ) AS operational_window_gate_pass
FROM window_results;

WITH selected AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
), bounds AS (
  SELECT s.*,
         to_timestamp(ceil(extract(epoch FROM started_at) / 300.0) * 300.0) AS aligned_start,
         least(coalesce(stopped_at, now()), now())
           - :'official_resolution_grace_seconds'::double precision * interval '1 second'
             AS eligible_cutoff
  FROM selected s
), expected_windows AS (
  SELECT b.experiment_id, b.process_id, g AS window_start,
         g + interval '5 minutes' AS window_end
  FROM bounds b
  CROSS JOIN LATERAL generate_series(
    b.aligned_start,
    to_timestamp(floor((extract(epoch FROM b.eligible_cutoff) - 300.0) / 300.0) * 300.0),
    interval '5 minutes'
  ) g
  WHERE b.eligible_cutoff >= b.aligned_start + interval '5 minutes'
), snapshot_sequence AS (
  SELECT s.window_start, s.window_end, s.snapshot_id, s.feature_as_of,
         s.readiness_status,
         extract(epoch FROM (s.feature_as_of - lag(s.feature_as_of) OVER (
           PARTITION BY s.window_start ORDER BY s.feature_as_of, s.snapshot_id
         ))) AS gap_seconds
  FROM selected e
  JOIN polymarket.btc_feature_snapshots s
    ON s.features->>'process_id' = e.process_id::text
   AND s.feature_as_of >= e.started_at
   AND s.feature_as_of <= least(coalesce(e.stopped_at, now()), now())
), snapshot_stats AS (
  SELECT window_start, count(*) AS snapshots,
         count(*) FILTER (WHERE readiness_status = 'ready') AS ready_snapshots,
         min(feature_as_of) AS first_snapshot_at,
         max(feature_as_of) AS last_snapshot_at,
         coalesce(max(gap_seconds), 0) AS maximum_gap_seconds
  FROM snapshot_sequence GROUP BY window_start
), decision_stats AS (
  SELECT s.window_start, count(*) AS decisions
  FROM selected e
  JOIN polymarket.btc_strategy_decisions d ON d.experiment_id = e.experiment_id
  JOIN polymarket.btc_feature_snapshots s ON s.snapshot_id = d.snapshot_id
  GROUP BY s.window_start
), market_stats AS (
  SELECT w.window_start,
         count(m.market_id) FILTER (WHERE m.validation_status='valid') AS valid_markets,
         count(l.market_id) AS local_labels,
         count(m.market_id) FILTER (
           WHERE m.official_outcome IN ('up','down')
             AND m.official_winning_token_id IN (m.up_token_id,m.down_token_id)
             AND m.official_resolved_at>=m.window_end
             AND m.official_resolution_source IN ('clob_websocket','clob_rest_reconciliation')
             AND m.official_resolution_received_at>=m.window_end
             AND m.official_resolution_received_at <= m.window_end
                 + :'official_resolution_grace_seconds'::double precision * interval '1 second'
             AND jsonb_typeof(m.official_resolution_payload)='object'
             AND rw.status IN ('resolved','resolved_late')
             AND rw.resolution_source=m.official_resolution_source
             AND rw.resolution_received_at=m.official_resolution_received_at
         ) AS official_labels_within_slo
  FROM expected_windows w
  LEFT JOIN polymarket.btc_interval_markets m
    ON m.window_start=w.window_start AND m.window_end=w.window_end
  LEFT JOIN polymarket.btc_market_labels l ON l.market_id=m.market_id
  LEFT JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  GROUP BY w.window_start
), results AS (
  SELECT w.window_start, w.window_end,
         coalesce(s.snapshots,0) AS snapshots,
         coalesce(d.decisions,0) AS decisions,
         coalesce(s.ready_snapshots,0) AS ready_snapshots,
         s.first_snapshot_at, s.last_snapshot_at,
         coalesce(s.maximum_gap_seconds,0) AS maximum_gap_seconds,
         coalesce(m.valid_markets,0) AS valid_markets,
         coalesce(m.local_labels,0) AS local_labels,
         coalesce(m.official_labels_within_slo,0) AS official_labels_within_slo,
         (
           coalesce(s.snapshots,0) >= :'minimum_snapshots_per_window'::bigint
           AND coalesce(d.decisions,0)=coalesce(s.snapshots,0)
           AND coalesce(s.ready_snapshots,0)>0
           AND s.first_snapshot_at <= w.window_start + interval '5 seconds'
           AND s.last_snapshot_at >= w.window_end - interval '5 seconds'
           AND coalesce(s.maximum_gap_seconds,0) <= :'maximum_snapshot_gap_seconds'::numeric
           AND coalesce(m.valid_markets,0)=1
           AND coalesce(m.local_labels,0)=1
           AND coalesce(m.official_labels_within_slo,0)=1
         ) AS complete
  FROM expected_windows w
  LEFT JOIN snapshot_stats s USING(window_start)
  LEFT JOIN decision_stats d USING(window_start)
  LEFT JOIN market_stats m USING(window_start)
)
SELECT *
FROM results
WHERE NOT complete
ORDER BY window_start
LIMIT 200;

WITH selected AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
), bounds AS (
  SELECT s.*,
         to_timestamp(ceil(extract(epoch FROM started_at) / 300.0) * 300.0) AS aligned_start,
         least(coalesce(stopped_at, now()), now())
           - :'official_resolution_grace_seconds'::double precision * interval '1 second'
             AS eligible_cutoff
  FROM selected s
), expected_windows AS (
  SELECT b.experiment_id, b.process_id, g AS window_start,
         g + interval '5 minutes' AS window_end
  FROM bounds b
  CROSS JOIN LATERAL generate_series(
    b.aligned_start,
    to_timestamp(floor((extract(epoch FROM b.eligible_cutoff) - 300.0) / 300.0) * 300.0),
    interval '5 minutes'
  ) g
  WHERE b.eligible_cutoff >= b.aligned_start + interval '5 minutes'
), snapshot_sequence AS (
  SELECT s.window_start, s.snapshot_id, s.feature_as_of, s.readiness_status,
         extract(epoch FROM (s.feature_as_of - lag(s.feature_as_of) OVER (
           PARTITION BY s.window_start ORDER BY s.feature_as_of, s.snapshot_id
         ))) AS gap_seconds
  FROM selected e
  JOIN polymarket.btc_feature_snapshots s
    ON s.features->>'process_id'=e.process_id::text
   AND s.feature_as_of>=e.started_at
   AND s.feature_as_of<=least(coalesce(e.stopped_at,now()),now())
), snapshot_stats AS (
  SELECT window_start, count(*) AS snapshots,
         count(*) FILTER (WHERE readiness_status='ready') AS ready_snapshots,
         min(feature_as_of) AS first_snapshot_at, max(feature_as_of) AS last_snapshot_at,
         coalesce(max(gap_seconds),0) AS maximum_gap_seconds
  FROM snapshot_sequence GROUP BY window_start
), decision_stats AS (
  SELECT s.window_start, count(*) AS decisions
  FROM selected e
  JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=e.experiment_id
  JOIN polymarket.btc_feature_snapshots s ON s.snapshot_id=d.snapshot_id
  GROUP BY s.window_start
), market_stats AS (
  SELECT w.window_start,
         count(m.market_id) FILTER (WHERE m.validation_status='valid') AS valid_markets,
         count(l.market_id) AS local_labels,
         count(m.market_id) FILTER (
           WHERE m.official_outcome IN ('up','down')
             AND m.official_winning_token_id IN (m.up_token_id,m.down_token_id)
             AND m.official_resolved_at>=m.window_end
             AND m.official_resolution_source IN ('clob_websocket','clob_rest_reconciliation')
             AND m.official_resolution_received_at>=m.window_end
             AND m.official_resolution_received_at <= m.window_end
               + :'official_resolution_grace_seconds'::double precision * interval '1 second'
             AND jsonb_typeof(m.official_resolution_payload)='object'
             AND rw.status IN ('resolved','resolved_late')
             AND rw.resolution_source=m.official_resolution_source
             AND rw.resolution_received_at=m.official_resolution_received_at
         ) AS official_slo
  FROM expected_windows w
  LEFT JOIN polymarket.btc_interval_markets m
    ON m.window_start=w.window_start AND m.window_end=w.window_end
  LEFT JOIN polymarket.btc_market_labels l ON l.market_id=m.market_id
  LEFT JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  GROUP BY w.window_start
), result AS (
  SELECT count(*) AS expected_windows,
         count(*) FILTER (
           WHERE coalesce(s.snapshots,0) >= :'minimum_snapshots_per_window'::bigint
             AND coalesce(d.decisions,0)=coalesce(s.snapshots,0)
             AND coalesce(s.ready_snapshots,0)>0
             AND s.first_snapshot_at<=w.window_start+interval '5 seconds'
             AND s.last_snapshot_at>=w.window_end-interval '5 seconds'
             AND coalesce(s.maximum_gap_seconds,0)<=:'maximum_snapshot_gap_seconds'::numeric
             AND coalesce(m.valid_markets,0)=1
             AND coalesce(m.local_labels,0)=1
         ) AS complete_windows,
         count(*) FILTER (WHERE coalesce(m.official_slo,0)=1) AS official_slo_windows
  FROM expected_windows w
  LEFT JOIN snapshot_stats s USING(window_start)
  LEFT JOIN decision_stats d USING(window_start)
  LEFT JOIN market_stats m USING(window_start)
)
SELECT expected_windows,
       complete_windows,
       official_slo_windows,
       coalesce(round(100.0*complete_windows/nullif(expected_windows,0),4),0) AS coverage_pct,
       coalesce(round(100.0*official_slo_windows/nullif(expected_windows,0),4),0) AS official_slo_pct,
       (
         expected_windows >= :'required_complete_windows'::bigint
         AND coalesce(100.0*complete_windows/nullif(expected_windows,0),0)
               >= :'minimum_coverage_pct'::numeric
         AND coalesce(100.0*official_slo_windows/nullif(expected_windows,0),0)
               >= :'minimum_official_slo_pct'::numeric
       ) AS gate_pass
FROM result \gset operational_

\echo '== Safety, FOK atomicity, official-fill coverage, and accounting =='
WITH selected AS (
  SELECT e.*, e.config AS process_config
  FROM polymarket.btc_paper_experiments e
  JOIN polymarket.trading_processes p ON p.process_id=e.process_id
  WHERE e.experiment_id=NULLIF(:'experiment_id','')::uuid
), process_orders AS (
  SELECT o.*,
         o.raw_payload #>> '{request,metadata,experiment_id}' AS order_experiment_id,
         o.raw_payload #>> '{request,metadata,decision_id}' AS order_decision_id
  FROM selected e
  JOIN polymarket.orders o ON o.process_id=e.process_id
   AND o.created_at>=e.started_at
   AND o.created_at<=least(coalesce(e.stopped_at,now()),now())
), cohort_orders AS (
  SELECT o.*
  FROM selected e
  JOIN process_orders o ON o.order_experiment_id=e.experiment_id::text
), fill_agg AS (
  SELECT f.order_id, count(*) AS fill_rows, sum(f.size) AS filled_size,
         sum(f.price*f.size) AS entry_notional, sum(f.fee) AS fees,
         count(*) FILTER (WHERE f.source<>'paper') AS nonpaper_fill_rows,
         count(*) FILTER (WHERE f.process_id IS DISTINCT FROM o.process_id)
           AS wrong_process_fill_rows
  FROM polymarket.fills f
  JOIN cohort_orders o ON o.order_id=f.order_id
  GROUP BY f.order_id
), official_orders AS (
  SELECT o.order_id,
         (m.window_end <= now()
           - :'official_resolution_grace_seconds'::double precision * interval '1 second')
           AS resolution_eligible,
         (
           m.official_outcome IN ('up','down')
           AND m.official_winning_token_id IN (m.up_token_id,m.down_token_id)
           AND m.official_resolved_at>=m.window_end
           AND m.official_resolution_source IN ('clob_websocket','clob_rest_reconciliation')
           AND m.official_resolution_received_at>=m.window_end
           AND jsonb_typeof(m.official_resolution_payload)='object'
           AND rw.status IN ('resolved','resolved_late')
           AND rw.resolution_source=m.official_resolution_source
           AND rw.resolution_received_at=m.official_resolution_received_at
         ) AS officially_resolved
  FROM cohort_orders o
  JOIN polymarket.btc_interval_markets m ON m.market_id=o.market_id
  LEFT JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
), recomputed AS (
  SELECT coalesce(sum(
           CASE
             WHEN NOT coalesce(oo.officially_resolved,false) THEN 0
             WHEN f.token_id=m.official_winning_token_id
               THEN f.size-f.price*f.size-f.fee
             ELSE -f.price*f.size-f.fee
           END
         ),0)::numeric AS recomputed_net_pnl
  FROM cohort_orders o
  JOIN polymarket.fills f ON f.order_id=o.order_id AND f.source='paper'
  JOIN polymarket.btc_interval_markets m ON m.market_id=o.market_id
  JOIN official_orders oo ON oo.order_id=o.order_id
), decision_violations AS (
  SELECT
    count(*) FILTER (
      WHERE d.action='buy'
        AND d.status IN ('approved','submitted','filled')
        AND d.decision_at>=l.label_available_at
    ) AS post_label_entries
  FROM selected e
  JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=e.experiment_id
  LEFT JOIN polymarket.btc_market_labels l ON l.market_id=d.market_id
), duplicate_entries AS (
  SELECT count(*) AS duplicate_market_entry_groups
  FROM (
    SELECT d.market_id
    FROM selected e
    JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=e.experiment_id
    WHERE d.action='buy' AND d.status IN ('approved','submitted','filled')
    GROUP BY d.market_id
    HAVING count(*)>1
  ) duplicates
), ml_authority AS (
  SELECT count(p.prediction_id) FILTER (
           WHERE p.metadata->>'influenced_order_plan' IS DISTINCT FROM 'false'
              OR p.metadata->>'shadow_only' IS DISTINCT FROM 'true'
         ) AS prediction_authority_metadata_violations
  FROM selected e
  JOIN polymarket.btc_feature_snapshots s
    ON s.features->>'process_id'=e.process_id::text
   AND s.feature_as_of>=e.started_at
   AND s.feature_as_of<=least(coalesce(e.stopped_at,now()),now())
  LEFT JOIN polymarket.ml_shadow_predictions p ON p.snapshot_id=s.snapshot_id
), summary AS (
  SELECT e.experiment_id, e.net_pnl AS recorded_net_pnl,
         r.recomputed_net_pnl,
         count(po.order_id) FILTER (
           WHERE po.order_experiment_id IS DISTINCT FROM e.experiment_id::text
         ) AS process_orders_missing_or_wrong_experiment,
         count(o.order_id) AS planned_orders,
         count(o.order_id) FILTER (WHERE o.side<>'buy' OR o.order_type<>'fok')
           AS non_buy_or_non_fok_orders,
         count(o.order_id) FILTER (WHERE o.state='filled') AS filled_orders,
         count(o.order_id) FILTER (
           WHERE o.state='filled' AND coalesce(fa.fill_rows,0)=0
             AND (:'audit_mode'='final' OR o.updated_at<=now()-interval '10 seconds')
         ) AS filled_orders_without_fills,
         count(o.order_id) FILTER (
           WHERE o.state='filled'
             AND coalesce(fa.filled_size,0) IS DISTINCT FROM o.size
             AND (:'audit_mode'='final' OR o.updated_at<=now()-interval '10 seconds')
         ) AS partial_fok_orders,
         count(o.order_id) FILTER (
           WHERE o.state<>'filled' AND coalesce(fa.fill_rows,0)>0
             AND (:'audit_mode'='final' OR o.updated_at<=now()-interval '10 seconds')
         ) AS nonfilled_orders_with_fills,
         coalesce(sum(fa.nonpaper_fill_rows),0) AS nonpaper_fill_rows,
         coalesce(sum(fa.wrong_process_fill_rows),0) AS wrong_process_fill_rows,
         count(o.order_id) FILTER (
           WHERE o.state='filled' AND coalesce(oo.officially_resolved,false)
         ) AS officially_resolved_filled_orders,
         count(o.order_id) FILTER (
           WHERE o.state='filled' AND coalesce(oo.resolution_eligible,false)
             AND NOT coalesce(oo.officially_resolved,false)
         ) AS official_unverified_filled_orders,
         dv.post_label_entries, de.duplicate_market_entry_groups,
         ma.prediction_authority_metadata_violations,
         (e.process_config #>> '{raw,ml_shadow,execution_authority}')::boolean
           AS ml_execution_authority
  FROM selected e
  LEFT JOIN process_orders po ON true
  LEFT JOIN cohort_orders o ON o.order_id=po.order_id
  LEFT JOIN fill_agg fa ON fa.order_id=o.order_id
  LEFT JOIN official_orders oo ON oo.order_id=o.order_id
  CROSS JOIN recomputed r
  CROSS JOIN decision_violations dv
  CROSS JOIN duplicate_entries de
  CROSS JOIN ml_authority ma
  GROUP BY e.experiment_id,e.net_pnl,e.process_config,r.recomputed_net_pnl,
           dv.post_label_entries,de.duplicate_market_entry_groups,
           ma.prediction_authority_metadata_violations
)
SELECT *,
       recorded_net_pnl-recomputed_net_pnl AS recorded_minus_recomputed_net_pnl,
       round(100.0*filled_orders/nullif(planned_orders,0),4) AS simulated_fok_fill_rate_pct,
       round(100.0*officially_resolved_filled_orders/nullif(filled_orders,0),4)
         AS official_fill_coverage_pct,
       (
         process_orders_missing_or_wrong_experiment=0
         AND non_buy_or_non_fok_orders=0
         AND filled_orders_without_fills=0
         AND partial_fok_orders=0
         AND nonfilled_orders_with_fills=0
         AND nonpaper_fill_rows=0
         AND wrong_process_fill_rows=0
         AND official_unverified_filled_orders=0
         AND post_label_entries=0
         AND duplicate_market_entry_groups=0
         AND prediction_authority_metadata_violations=0
         AND NOT ml_execution_authority
         AND abs(recorded_net_pnl-recomputed_net_pnl)
               <= :'pnl_reconciliation_tolerance_usd'::numeric
       ) AS safety_execution_accounting_gate_pass
FROM summary;

WITH selected AS (
  SELECT e.*,e.config AS process_config
  FROM polymarket.btc_paper_experiments e
  JOIN polymarket.trading_processes p USING(process_id)
  WHERE e.experiment_id=NULLIF(:'experiment_id','')::uuid
), process_orders AS (
  SELECT o.*,o.raw_payload #>> '{request,metadata,experiment_id}' AS order_experiment_id
  FROM selected e JOIN polymarket.orders o ON o.process_id=e.process_id
   AND o.created_at>=e.started_at
   AND o.created_at<=least(coalesce(e.stopped_at,now()),now())
), cohort_orders AS (
  SELECT o.* FROM selected e JOIN process_orders o
    ON o.order_experiment_id=e.experiment_id::text
), fill_agg AS (
  SELECT f.order_id,count(*) AS fill_rows,sum(f.size) AS filled_size,
         count(*) FILTER(WHERE f.source<>'paper') AS nonpaper_fill_rows,
         count(*) FILTER(WHERE f.process_id IS DISTINCT FROM o.process_id)
           AS wrong_process_fill_rows
  FROM polymarket.fills f JOIN cohort_orders o ON o.order_id=f.order_id
  GROUP BY f.order_id
), official_orders AS (
  SELECT o.order_id,
    (m.window_end <= now()
      - :'official_resolution_grace_seconds'::double precision * interval '1 second')
      AS resolution_eligible,
    (
    m.official_outcome IN ('up','down')
    AND m.official_winning_token_id IN(m.up_token_id,m.down_token_id)
    AND m.official_resolved_at>=m.window_end
    AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
    AND m.official_resolution_received_at>=m.window_end
    AND jsonb_typeof(m.official_resolution_payload)='object'
    AND rw.status IN('resolved','resolved_late')
    AND rw.resolution_source=m.official_resolution_source
    AND rw.resolution_received_at=m.official_resolution_received_at
  ) AS officially_resolved
  FROM cohort_orders o JOIN polymarket.btc_interval_markets m ON m.market_id=o.market_id
  LEFT JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
), recomputed AS (
  SELECT coalesce(sum(CASE WHEN NOT coalesce(oo.officially_resolved,false) THEN 0
    WHEN f.token_id=m.official_winning_token_id THEN f.size-f.price*f.size-f.fee
    ELSE -f.price*f.size-f.fee END),0)::numeric AS net_pnl
  FROM cohort_orders o JOIN polymarket.fills f ON f.order_id=o.order_id AND f.source='paper'
  JOIN polymarket.btc_interval_markets m ON m.market_id=o.market_id
  JOIN official_orders oo ON oo.order_id=o.order_id
), post_label AS (
  SELECT count(*) FILTER(WHERE d.action='buy' AND d.status IN('approved','submitted','filled')
    AND d.decision_at>=l.label_available_at) AS violations
  FROM selected e JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=e.experiment_id
  LEFT JOIN polymarket.btc_market_labels l ON l.market_id=d.market_id
), duplicates AS (
  SELECT count(*) AS violations FROM (SELECT d.market_id FROM selected e
    JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=e.experiment_id
    WHERE d.action='buy' AND d.status IN('approved','submitted','filled')
    GROUP BY d.market_id HAVING count(*)>1) x
), ml_authority AS (
  SELECT count(p.prediction_id) FILTER(WHERE p.metadata->>'influenced_order_plan' IS DISTINCT FROM 'false'
    OR p.metadata->>'shadow_only' IS DISTINCT FROM 'true') AS violations
  FROM selected e JOIN polymarket.btc_feature_snapshots s
    ON s.features->>'process_id'=e.process_id::text
    AND s.feature_as_of>=e.started_at
    AND s.feature_as_of<=least(coalesce(e.stopped_at,now()),now())
  LEFT JOIN polymarket.ml_shadow_predictions p ON p.snapshot_id=s.snapshot_id
), summary AS (
  SELECT e.net_pnl,r.net_pnl AS recomputed_net_pnl,
    count(po.order_id) FILTER(WHERE po.order_experiment_id IS DISTINCT FROM e.experiment_id::text)
      AS wrong_process_orders,
    count(o.order_id) AS planned_orders,
    count(o.order_id) FILTER(WHERE o.state='filled') AS filled_orders,
    count(o.order_id) FILTER(WHERE o.side<>'buy' OR o.order_type<>'fok') AS wrong_order_type,
    count(o.order_id) FILTER(WHERE o.state='filled' AND coalesce(fa.fill_rows,0)=0
      AND (:'audit_mode'='final' OR o.updated_at<=now()-interval '10 seconds'))
      AS missing_fills,
    count(o.order_id) FILTER(WHERE o.state='filled'
      AND coalesce(fa.filled_size,0) IS DISTINCT FROM o.size
      AND (:'audit_mode'='final' OR o.updated_at<=now()-interval '10 seconds'))
      AS partial_fok,
    count(o.order_id) FILTER(WHERE o.state<>'filled' AND coalesce(fa.fill_rows,0)>0
      AND (:'audit_mode'='final' OR o.updated_at<=now()-interval '10 seconds'))
      AS fills_on_nonfilled,
    coalesce(sum(fa.nonpaper_fill_rows),0) AS nonpaper_fills,
    coalesce(sum(fa.wrong_process_fill_rows),0) AS wrong_process_fills,
    count(o.order_id) FILTER(WHERE o.state='filled' AND coalesce(oo.resolution_eligible,false)
      AND NOT coalesce(oo.officially_resolved,false))
      AS unverified_filled_orders,
    pl.violations AS post_label_entries,du.violations AS duplicate_entries,
    ma.violations AS ml_authority_violations,
    (e.process_config #>> '{raw,ml_shadow,execution_authority}')::boolean AS ml_authority
  FROM selected e LEFT JOIN process_orders po ON true
  LEFT JOIN cohort_orders o ON o.order_id=po.order_id
  LEFT JOIN fill_agg fa ON fa.order_id=o.order_id LEFT JOIN official_orders oo ON oo.order_id=o.order_id
  CROSS JOIN recomputed r CROSS JOIN post_label pl CROSS JOIN duplicates du CROSS JOIN ml_authority ma
  GROUP BY e.experiment_id,e.net_pnl,e.process_config,r.net_pnl,pl.violations,du.violations,ma.violations
)
SELECT planned_orders,filled_orders,
       coalesce(round(100.0*filled_orders/nullif(planned_orders,0),4),0) AS fok_fill_rate_pct,
       (
         wrong_process_orders=0 AND wrong_order_type=0 AND missing_fills=0 AND partial_fok=0
         AND fills_on_nonfilled=0 AND nonpaper_fills=0 AND wrong_process_fills=0
         AND unverified_filled_orders=0
         AND post_label_entries=0 AND duplicate_entries=0 AND ml_authority_violations=0
         AND NOT ml_authority
         AND abs(net_pnl-recomputed_net_pnl)<=:'pnl_reconciliation_tolerance_usd'::numeric
       ) AS gate_pass
FROM summary \gset integrity_

\echo '== Authoritative paper capital, settlement ledger, and process-fill integrity =='
WITH selected AS (
  SELECT e.*,e.config AS process_config
  FROM polymarket.btc_paper_experiments e
  JOIN polymarket.trading_processes p USING(process_id)
  WHERE e.experiment_id=NULLIF(:'experiment_id','')::uuid
), process_fill_audit AS (
  SELECT count(*) FILTER(WHERE
      (:'audit_mode'='final' OR f.created_at<=now()-interval '10 seconds')
      AND o.order_id IS NULL) AS orphan_process_fills,
    count(*) FILTER(WHERE
      (:'audit_mode'='final' OR f.created_at<=now()-interval '10 seconds')
      AND f.source<>'paper') AS wrong_source_process_fills,
    count(*) FILTER(WHERE
      (:'audit_mode'='final' OR f.created_at<=now()-interval '10 seconds')
      AND o.order_id IS NOT NULL
      AND (o.process_id IS DISTINCT FROM s.process_id
        OR o.raw_payload #>> '{request,metadata,experiment_id}'
             IS DISTINCT FROM s.experiment_id::text)) AS wrong_identity_process_fills
  FROM selected s
  LEFT JOIN polymarket.fills f ON f.process_id=s.process_id
   AND f.created_at>=s.started_at
   AND f.created_at<=least(coalesce(s.stopped_at,now()),now())
  LEFT JOIN polymarket.orders o ON o.order_id=f.order_id
  GROUP BY s.process_id,s.experiment_id
), paper_order_fills AS (
  SELECT o.order_id,o.market_id,o.token_id,
    jsonb_agg(to_jsonb(f.fill_id) ORDER BY f.timestamp_utc,f.fill_id) AS fill_ids,
    round(sum(f.size)::numeric,10) AS filled_size,
    round(sum(f.price*f.size)::numeric,10) AS entry_notional,
    round(sum(f.fee)::numeric,10) AS entry_fees
  FROM selected s
  JOIN polymarket.orders o ON o.process_id=s.process_id
   AND o.raw_payload #>> '{request,metadata,experiment_id}'=s.experiment_id::text
  JOIN polymarket.fills f ON f.order_id=o.order_id AND f.source='paper'
  GROUP BY o.order_id,o.market_id,o.token_id
), official_fills AS (
  SELECT pf.*,m.window_end,m.official_outcome,m.official_winning_token_id,
    m.official_resolution_received_at,m.official_resolution_source,
    CASE WHEN pf.token_id=m.official_winning_token_id THEN pf.filled_size ELSE 0::numeric END
      AS expected_payout,
    CASE WHEN pf.token_id=m.official_winning_token_id
      THEN pf.filled_size-pf.entry_notional-pf.entry_fees
      ELSE -pf.entry_notional-pf.entry_fees END AS expected_net_pnl,
    (m.window_end<=now()
      - :'official_resolution_grace_seconds'::double precision*interval '1 second')
      AS resolution_eligible
  FROM paper_order_fills pf
  JOIN polymarket.btc_interval_markets m ON m.market_id=pf.market_id
  JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE m.official_outcome IN('up','down')
    AND m.official_winning_token_id IN(m.up_token_id,m.down_token_id)
    AND m.official_resolved_at>=m.window_end
    AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
    AND m.official_resolution_received_at>=m.window_end
    AND jsonb_typeof(m.official_resolution_payload)='object'
    AND rw.status IN('resolved','resolved_late')
    AND rw.resolution_source=m.official_resolution_source
    AND rw.resolution_received_at=m.official_resolution_received_at
), ledger_validation AS (
  SELECT count(l.settlement_id) AS ledger_rows,
    count(l.settlement_id) FILTER(WHERE l.credit_status='credited') AS credited_rows,
    count(l.settlement_id) FILTER(WHERE l.credit_status='pending'
      AND m.window_end<=now()
        - :'official_resolution_grace_seconds'::double precision*interval '1 second')
      AS overdue_pending_settlements,
    count(l.settlement_id) FILTER(WHERE NOT coalesce(
      l.process_id=s.process_id
      AND o.process_id=s.process_id
      AND o.raw_payload #>> '{request,metadata,experiment_id}'=s.experiment_id::text
      AND l.order_id=o.order_id AND l.market_id=o.market_id AND l.token_id=o.token_id
      AND pf.order_id=l.order_id AND l.fill_ids=pf.fill_ids
      AND l.filled_size=pf.filled_size AND l.entry_notional=pf.entry_notional
      AND l.entry_fees=pf.entry_fees
      AND l.official_outcome=m.official_outcome
      AND l.official_winning_token_id=m.official_winning_token_id
      AND l.official_resolution_received_at=m.official_resolution_received_at
      AND l.official_resolution_source=m.official_resolution_source
      AND rw.status IN('resolved','resolved_late')
      AND rw.resolution_source=m.official_resolution_source
      AND rw.resolution_received_at=m.official_resolution_received_at
      AND l.payout=CASE WHEN l.token_id=l.official_winning_token_id
        THEN l.filled_size ELSE 0::numeric END
      AND l.net_pnl=l.payout-l.entry_notional-l.entry_fees
      AND (
        l.credit_status='pending'
        OR (
          l.credit_status='credited'
          AND l.credit_evidence->>'evidence_version'='btc_paper_capital_credit_v1'
          AND l.credit_evidence->>'settlement_id'=l.settlement_id::text
          AND l.credit_evidence->>'experiment_id'=l.experiment_id::text
          AND l.credit_evidence->>'process_id'=l.process_id::text
          AND l.credit_evidence->>'order_id'=l.order_id
          AND l.credit_evidence->>'market_id'=l.market_id
          AND l.credit_evidence->>'token_id'=l.token_id
          AND l.credit_evidence->>'credited_by_config_hash'=s.config_hash
          AND l.credit_evidence #>> '{venue_credit,settlement_id}'=l.settlement_id::text
          AND jsonb_typeof(l.credit_evidence #> '{venue_credit,newly_applied}')='boolean'
          AND nullif(l.credit_evidence #>> '{venue_credit,payout_usd}','')::numeric=l.payout
        )
      ),false)) AS ledger_identity_or_provenance_violations
  FROM selected s
  LEFT JOIN polymarket.btc_paper_settlement_ledger l ON l.experiment_id=s.experiment_id
  LEFT JOIN polymarket.orders o ON o.order_id=l.order_id
  LEFT JOIN paper_order_fills pf ON pf.order_id=l.order_id
  LEFT JOIN polymarket.btc_interval_markets m ON m.market_id=l.market_id
  LEFT JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  GROUP BY s.process_id,s.experiment_id,s.config_hash
), eligible_reconciliation AS (
  SELECT count(of.order_id) AS expected_eligible_settlements,
    count(l.settlement_id) FILTER(WHERE l.credit_status='credited') AS credited_eligible_settlements,
    coalesce(sum(of.expected_payout),0)::numeric AS expected_eligible_payout,
    coalesce(sum(l.payout) FILTER(WHERE l.credit_status='credited'),0)::numeric
      AS credited_eligible_payout,
    coalesce(sum(of.expected_net_pnl),0)::numeric AS expected_eligible_net_pnl,
    coalesce(sum(l.net_pnl) FILTER(WHERE l.credit_status='credited'),0)::numeric
      AS credited_eligible_net_pnl
  FROM official_fills of
  LEFT JOIN polymarket.btc_paper_settlement_ledger l ON l.order_id=of.order_id
    AND l.experiment_id=(SELECT experiment_id FROM selected)
  WHERE of.resolution_eligible
), fill_totals AS (
  SELECT count(*) AS filled_orders,
    coalesce(sum(entry_notional+entry_fees),0)::numeric AS total_entry_debits
  FROM paper_order_fills
), recent_capital_events AS (
  SELECT count(DISTINCT o.order_id) AS recent_orders_or_fills
  FROM selected s JOIN polymarket.orders o ON o.process_id=s.process_id
   AND o.raw_payload #>> '{request,metadata,experiment_id}'=s.experiment_id::text
  LEFT JOIN polymarket.fills f ON f.order_id=o.order_id
  WHERE o.updated_at>now()-interval '10 seconds'
     OR f.created_at>now()-interval '10 seconds'
), ledger_totals AS (
  SELECT count(*) FILTER(WHERE credit_status='credited') AS credited_settlements,
    coalesce(sum(payout) FILTER(WHERE credit_status='credited'),0)::numeric AS credited_payout,
    coalesce(sum(net_pnl) FILTER(WHERE credit_status='credited'),0)::numeric AS credited_net_pnl
  FROM polymarket.btc_paper_settlement_ledger l
  WHERE l.experiment_id=(SELECT experiment_id FROM selected)
), capital AS (
  SELECT nullif(s.summary #>> '{paper_capital,venue,starting_collateral_usd}','')::numeric
      AS starting_collateral,
    nullif(s.summary #>> '{paper_capital,venue,available_collateral_usd}','')::numeric
      AS available_collateral,
    nullif(s.summary #>> '{paper_capital,venue,entry_debits_usd}','')::numeric
      AS venue_entry_debits,
    nullif(s.summary #>> '{paper_capital,venue,settlement_credits_usd}','')::numeric
      AS venue_settlement_credits,
    nullif(s.summary #>> '{paper_capital,venue,settlements_applied}','')::bigint
      AS venue_settlements_applied,
    s.net_pnl::numeric AS experiment_net_pnl
  FROM selected s
), result AS (
  SELECT pfa.*,lv.ledger_rows,lv.credited_rows,lv.overdue_pending_settlements,
    lv.ledger_identity_or_provenance_violations,
    er.expected_eligible_settlements,er.credited_eligible_settlements,
    rce.recent_orders_or_fills,
    abs(er.expected_eligible_payout-er.credited_eligible_payout) AS eligible_payout_gap,
    abs(er.expected_eligible_net_pnl-er.credited_eligible_net_pnl) AS eligible_net_pnl_gap,
    ft.filled_orders,lt.credited_settlements,
    c.starting_collateral,c.available_collateral,c.venue_entry_debits,
    c.venue_settlement_credits,c.venue_settlements_applied,c.experiment_net_pnl,
    coalesce(abs(c.available_collateral-(c.starting_collateral-c.venue_entry_debits
      +c.venue_settlement_credits)),999999) AS collateral_reconciliation_gap,
    coalesce(abs(c.venue_entry_debits-ft.total_entry_debits),999999)
      AS entry_debit_reconciliation_gap,
    coalesce(abs(c.venue_settlement_credits-lt.credited_payout),999999)
      AS payout_reconciliation_gap,
    coalesce(abs(c.venue_settlements_applied-lt.credited_settlements),999999)
      AS settlement_count_gap,
    coalesce(abs(c.experiment_net_pnl-lt.credited_net_pnl),999999)
      AS experiment_ledger_net_pnl_gap
  FROM process_fill_audit pfa CROSS JOIN ledger_validation lv
  CROSS JOIN eligible_reconciliation er CROSS JOIN fill_totals ft
  CROSS JOIN recent_capital_events rce CROSS JOIN ledger_totals lt CROSS JOIN capital c
)
SELECT *,coalesce((
    orphan_process_fills=0 AND wrong_source_process_fills=0
    AND wrong_identity_process_fills=0
    AND overdue_pending_settlements=0
    AND ledger_identity_or_provenance_violations=0
    AND expected_eligible_settlements=credited_eligible_settlements
    AND eligible_payout_gap<=:'pnl_reconciliation_tolerance_usd'::numeric
    AND eligible_net_pnl_gap<=:'pnl_reconciliation_tolerance_usd'::numeric
    AND ((:'audit_mode'='daily' AND recent_orders_or_fills>0) OR (
      collateral_reconciliation_gap<=:'pnl_reconciliation_tolerance_usd'::numeric
      AND entry_debit_reconciliation_gap<=:'pnl_reconciliation_tolerance_usd'::numeric
      AND payout_reconciliation_gap<=:'pnl_reconciliation_tolerance_usd'::numeric
      AND settlement_count_gap=0
      AND experiment_ledger_net_pnl_gap<=:'pnl_reconciliation_tolerance_usd'::numeric
    ))
  ),false) AS gate_pass
FROM result \gset capital_

SELECT :'capital_orphan_process_fills'::bigint AS orphan_process_fills,
  :'capital_wrong_source_process_fills'::bigint AS wrong_source_process_fills,
  :'capital_wrong_identity_process_fills'::bigint AS wrong_identity_process_fills,
  :'capital_overdue_pending_settlements'::bigint AS overdue_pending_settlements,
  :'capital_ledger_identity_or_provenance_violations'::bigint
    AS ledger_identity_or_provenance_violations,
  :'capital_expected_eligible_settlements'::bigint AS expected_eligible_settlements,
  :'capital_credited_eligible_settlements'::bigint AS credited_eligible_settlements,
  :'capital_recent_orders_or_fills'::bigint AS recent_inflight_capital_events,
  :'capital_collateral_reconciliation_gap'::numeric AS collateral_reconciliation_gap,
  :'capital_entry_debit_reconciliation_gap'::numeric AS entry_debit_reconciliation_gap,
  :'capital_payout_reconciliation_gap'::numeric AS payout_reconciliation_gap,
  :'capital_settlement_count_gap'::bigint AS settlement_count_gap,
  :'capital_experiment_ledger_net_pnl_gap'::numeric AS experiment_ledger_net_pnl_gap,
  :'capital_gate_pass'::boolean AS paper_capital_settlement_gate_pass;

SELECT (:'integrity_gate_pass'::boolean AND :'capital_gate_pass'::boolean) AS gate_pass
  \gset integrity_

\echo '== Consolidated deterministic readiness and execution-parity gate =='
WITH selected AS (
  SELECT e.*,
    CASE WHEN e.status='failed' THEN e.stop_reason ELSE NULL END AS last_error,
    e.config AS process_config
  FROM polymarket.btc_paper_experiments e
  JOIN polymarket.trading_processes p USING(process_id)
  WHERE e.experiment_id=NULLIF(:'experiment_id','')::uuid
), expected_constraints(table_name,constraint_name) AS (
  VALUES
    ('btc_interval_markets'::text,'uq_btc_interval_market_official_identity'::text),
    ('btc_paper_settlement_ledger','btc_paper_settlement_ledger_pkey'),
    ('btc_paper_settlement_ledger','btc_paper_settlement_ledger_experiment_id_fkey'),
    ('btc_paper_settlement_ledger','btc_paper_settlement_ledger_process_id_fkey'),
    ('btc_paper_settlement_ledger','btc_paper_settlement_ledger_order_id_fkey'),
    ('btc_paper_settlement_ledger','btc_paper_settlement_ledger_market_id_fkey'),
    ('btc_paper_settlement_ledger','uq_btc_paper_settlement_experiment_order'),
    ('btc_paper_settlement_ledger','fk_btc_paper_settlement_official_identity'),
    ('btc_paper_settlement_ledger','chk_btc_paper_settlement_outcome'),
    ('btc_paper_settlement_ledger','chk_btc_paper_settlement_fill_ids'),
    ('btc_paper_settlement_ledger','chk_btc_paper_settlement_source'),
    ('btc_paper_settlement_ledger','chk_btc_paper_settlement_amounts'),
    ('btc_paper_settlement_ledger','chk_btc_paper_settlement_credit_status'),
    ('btc_paper_settlement_ledger','chk_btc_paper_settlement_credit_state'),
    ('btc_paper_settlement_ledger','chk_btc_paper_settlement_evidence')
), expected_indexes(index_name) AS (
  VALUES ('idx_btc_paper_settlement_pending'::text),
    ('idx_btc_paper_settlement_experiment_credited'),
    ('idx_btc_features_process_window_asof'),
    ('idx_btc_decisions_experiment_process_at'),
    ('idx_book_checkpoints_token_received_source')
), schema_state AS (
  SELECT
    (SELECT count(*) FROM public.migrations
      WHERE timestamp=1777123000000
        AND name='AddBtcPhase6PaperCapital1777123000000') AS paper_capital_migrations,
    (SELECT count(*) FROM public.migrations
      WHERE timestamp=1777124000000
        AND name='AllowUnstartedTradingProcesses1777124000000')
      AS unstarted_process_migrations,
    coalesce((SELECT is_nullable='YES'
      FROM information_schema.columns
      WHERE table_schema='polymarket' AND table_name='trading_processes'
        AND column_name='started_at'),false) AS trading_process_started_at_nullable,
    (SELECT count(*) FROM information_schema.tables
      WHERE table_schema='polymarket' AND table_name='btc_paper_settlement_ledger')
      AS ledger_tables,
    (SELECT count(*) FROM expected_constraints e JOIN pg_namespace n ON n.nspname='polymarket'
      JOIN pg_class t ON t.relnamespace=n.oid AND t.relname=e.table_name
      JOIN pg_constraint c ON c.conrelid=t.oid AND c.conname=e.constraint_name)
      AS paper_capital_constraints,
    (SELECT count(*) FROM expected_indexes e JOIN pg_indexes i
      ON i.schemaname='polymarket' AND i.indexname=e.index_name) AS paper_capital_indexes,
    (SELECT count(*) FROM timescaledb_information.jobs
      WHERE hypertable_schema='polymarket'
        AND hypertable_name IN('reference_price_ticks','btc_feature_snapshots',
          'ml_feature_vectors','ml_shadow_predictions')
        AND proc_name='policy_compression' AND scheduled
        AND config->>'compress_after'='1 day') AS one_day_compression_policies
), feed_state AS (
  SELECT count(f.connection_id) AS sessions,
    coalesce(sum(f.decode_errors),0) AS decode_errors,
    coalesce(sum(f.integrity_gaps),0) AS integrity_gaps,
    coalesce(sum(f.dropped_messages),0) AS dropped_messages,
    count(f.connection_id) FILTER(WHERE f.disconnect_reason IS NOT NULL
      AND f.disconnect_reason NOT IN('shutdown','market_watch_changed')) AS unexpected_disconnects
  FROM selected s LEFT JOIN polymarket.feed_sessions f
    ON f.started_at<=least(coalesce(s.stopped_at,now()),now())
   AND coalesce(f.disconnected_at,least(coalesce(s.stopped_at,now()),now()))>=s.started_at
), snapshots AS (
  SELECT fs.* FROM selected s JOIN polymarket.btc_feature_snapshots fs
    ON fs.features->>'process_id'=s.process_id::text
   AND fs.feature_as_of>=s.started_at
   AND fs.feature_as_of<=least(coalesce(s.stopped_at,now()),now())
   AND (:'audit_mode'='final' OR fs.created_at<=now()-interval '10 seconds')
), snapshot_state AS (
  SELECT count(*) AS snapshots,
    count(*) FILTER(WHERE fs.features->>'snapshot_id' IS DISTINCT FROM fs.snapshot_id::text
      OR fs.features->>'market_id' IS DISTINCT FROM fs.market_id
      OR fs.features->'lineage' IS DISTINCT FROM fs.lineage) AS identity_or_embedded_lineage_mismatches,
    count(*) FILTER(WHERE fs.readiness_status='ready'
      AND fs.window_start>=(
        SELECT to_timestamp(
          ceil(extract(epoch FROM s.started_at)/300.0)*300.0
        )
        FROM selected s
      )
      AND (
      fs.lineage->>'chainlink_open_tick_id' IS NULL OR fs.lineage->>'chainlink_tick_id' IS NULL
      OR fs.lineage->>'binance_tick_id' IS NULL
      OR fs.lineage->>'up_book_checkpoint_id' IS NULL
      OR fs.lineage->>'down_book_checkpoint_id' IS NULL)) AS ready_missing_core_lineage,
    count(*) FILTER(WHERE
      nullif(fs.lineage->>'chainlink_open_source_timestamp','')::timestamptz>fs.feature_as_of
      OR nullif(fs.lineage->>'chainlink_source_timestamp','')::timestamptz>fs.feature_as_of
      OR nullif(fs.lineage->>'binance_source_timestamp','')::timestamptz>fs.feature_as_of
      OR nullif(fs.features #>> '{up_book,source_timestamp}','')::timestamptz>fs.feature_as_of
      OR nullif(fs.features #>> '{down_book,source_timestamp}','')::timestamptz>fs.feature_as_of)
      AS future_source_snapshots,
    count(*) FILTER(WHERE
      nullif(fs.lineage->>'chainlink_open_received_at','')::timestamptz>fs.received_at
      OR nullif(fs.lineage->>'chainlink_received_at','')::timestamptz>fs.received_at
      OR nullif(fs.lineage->>'binance_received_at','')::timestamptz>fs.received_at
      OR nullif(fs.features #>> '{up_book,received_at}','')::timestamptz>fs.received_at
      OR nullif(fs.features #>> '{down_book,received_at}','')::timestamptz>fs.received_at)
      AS future_received_snapshots
  FROM snapshots fs
), decision_state AS (
  SELECT count(d.decision_id) AS decisions,
    count(d.decision_id) FILTER(WHERE d.process_id IS DISTINCT FROM s.process_id
      OR d.config_hash IS DISTINCT FROM s.config_hash
      OR fs.snapshot_id IS NULL) AS decision_identity_mismatches,
    count(d.decision_id) FILTER(WHERE d.order_plan_id IS NOT NULL AND (
      fs.readiness_status IS DISTINCT FROM 'ready' OR d.decision_at<fs.feature_as_of
      OR d.decision_at<fs.received_at)) AS approved_not_point_in_time_ready
  FROM selected s LEFT JOIN polymarket.btc_strategy_decisions d
    ON d.experiment_id=s.experiment_id
   AND (:'audit_mode'='final' OR d.created_at<=now()-interval '10 seconds')
  LEFT JOIN polymarket.btc_feature_snapshots fs ON fs.snapshot_id=d.snapshot_id
  GROUP BY s.process_id,s.config_hash
), scoped_orders AS (
  SELECT o.*,
    count(f.fill_id) AS fill_rows,coalesce(sum(f.size),0) AS filled_size,
    coalesce(sum(f.fee),0) AS recorded_fee,
    coalesce(sum(round(f.size
      *(o.raw_payload #>> '{request,metadata,paper_execution,dynamic_fee_rate}')::numeric
      *f.price*(1::numeric-f.price),10)),0) AS recomputed_fee,
    count(f.fill_id) FILTER(WHERE
      o.raw_payload #>> '{request,metadata,paper_execution,dynamic_fee_rate}' IS NULL)
      AS fills_without_dynamic_fee_rate
  FROM selected s JOIN polymarket.orders o ON o.process_id=s.process_id
   AND o.raw_payload #>> '{request,metadata,experiment_id}'=s.experiment_id::text
  LEFT JOIN polymarket.fills f ON f.order_id=o.order_id
  GROUP BY o.order_id
), execution_state AS (
  SELECT count(o.order_id) AS orders,
    count(o.order_id) FILTER(WHERE
      (:'audit_mode'='final' OR o.updated_at<=now()-interval '10 seconds')
      AND ((o.state='filled' AND o.filled_size<>o.size)
        OR (o.state<>'filled' AND o.filled_size<>0))) AS fok_atomicity_mismatches,
    count(o.order_id) FILTER(WHERE o.side<>'buy' OR o.order_type<>'fok')
      AS wrong_order_types,
    count(o.order_id) FILTER(WHERE
      (:'audit_mode'='final' OR o.updated_at<=now()-interval '10 seconds')
      AND (d.decision_id IS NULL OR d.order_plan_id IS NULL
        OR d.metadata #>> '{paper_order_plan,plan_id}' IS DISTINCT FROM d.order_plan_id::text))
      AS decision_plan_link_mismatches,
    count(o.order_id) FILTER(WHERE
      (:'audit_mode'='final' OR o.updated_at<=now()-interval '10 seconds')
      AND o.state='filled' AND (
        o.raw_payload #>> '{request,metadata,paper_execution,arrival_checkpoint_id}' IS NULL
        OR o.raw_payload #>> '{request,metadata,paper_execution,arrival_at}' IS NULL
        OR (o.raw_payload #>> '{request,metadata,paper_execution,arrival_checkpoint,received_at}')::timestamptz
             >(o.raw_payload #>> '{request,metadata,paper_execution,arrival_at}')::timestamptz
        OR o.raw_payload #>> '{request,metadata,paper_execution,arrival_checkpoint_id}'
             IS DISTINCT FROM o.raw_payload #>> '{request,metadata,paper_execution,arrival_checkpoint,checkpoint_id}'))
      AS arrival_causality_mismatches,
    count(o.order_id) FILTER(WHERE
      (o.raw_payload #>> '{request,metadata,paper_execution,configured_latency_ms}')::numeric
        IS DISTINCT FROM ((s.process_config #>> '{raw,paper,venue,arrival_latency,secs}')::numeric*1000
          +(s.process_config #>> '{raw,paper,venue,arrival_latency,nanos}')::numeric/1000000)
      OR (o.raw_payload #>> '{request,metadata,paper_execution,visible_depth_haircut}')::numeric
        IS DISTINCT FROM (s.process_config #>> '{raw,paper,venue,visible_depth_haircut}')::numeric)
      AS latency_or_haircut_mismatches,
    coalesce(sum(o.fills_without_dynamic_fee_rate),0) AS fills_without_dynamic_fee_rate,
    count(o.order_id) FILTER(WHERE o.recorded_fee IS DISTINCT FROM o.recomputed_fee)
      AS dynamic_fee_mismatches
  FROM selected s LEFT JOIN scoped_orders o ON true
  LEFT JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=s.experiment_id
   AND d.decision_id::text=o.raw_payload #>> '{request,metadata,decision_id}'
  GROUP BY s.experiment_id,s.process_config
), result AS (
  SELECT ss.paper_capital_migrations,ss.unstarted_process_migrations,
    ss.trading_process_started_at_nullable,ss.ledger_tables,
    ss.paper_capital_constraints,ss.paper_capital_indexes,
    ss.one_day_compression_policies,fs.sessions,fs.decode_errors,fs.integrity_gaps,
    fs.dropped_messages,fs.unexpected_disconnects,sn.*,ds.decisions,
    ds.decision_identity_mismatches,ds.approved_not_point_in_time_ready,es.*,
    (SELECT last_error FROM selected) AS process_last_error
  FROM schema_state ss CROSS JOIN feed_state fs CROSS JOIN snapshot_state sn
  CROSS JOIN decision_state ds CROSS JOIN execution_state es
)
SELECT *,coalesce((paper_capital_migrations=1 AND unstarted_process_migrations=1
    AND trading_process_started_at_nullable AND ledger_tables=1 AND paper_capital_constraints=15
    AND paper_capital_indexes=5 AND one_day_compression_policies=4
    AND process_last_error IS NULL
    AND decode_errors=0 AND integrity_gaps=0 AND dropped_messages=0
    AND identity_or_embedded_lineage_mismatches=0 AND ready_missing_core_lineage=0
    AND future_source_snapshots=0 AND future_received_snapshots=0
    AND decision_identity_mismatches=0 AND approved_not_point_in_time_ready=0
    AND fok_atomicity_mismatches=0 AND wrong_order_types=0
    AND decision_plan_link_mismatches=0 AND arrival_causality_mismatches=0
    AND latency_or_haircut_mismatches=0 AND fills_without_dynamic_fee_rate=0
    AND dynamic_fee_mismatches=0),false) AS gate_pass
FROM result \gset readiness_

SELECT :'readiness_paper_capital_migrations'::bigint AS paper_capital_migrations,
  :'readiness_unstarted_process_migrations'::bigint AS unstarted_process_migrations,
  :'readiness_trading_process_started_at_nullable'::boolean
    AS trading_process_started_at_nullable,
  :'readiness_paper_capital_constraints'::bigint AS paper_capital_constraints,
  :'readiness_paper_capital_indexes'::bigint AS paper_capital_indexes,
  :'readiness_one_day_compression_policies'::bigint AS one_day_compression_policies,
  :'readiness_decode_errors'::bigint AS feed_decode_errors,
  :'readiness_integrity_gaps'::bigint AS feed_integrity_gaps,
  :'readiness_dropped_messages'::bigint AS dropped_messages,
  :'readiness_unexpected_disconnects'::bigint AS unexpected_disconnects,
  :'readiness_approved_not_point_in_time_ready'::bigint AS approved_not_point_in_time_ready,
  :'readiness_arrival_causality_mismatches'::bigint AS arrival_causality_mismatches,
  :'readiness_dynamic_fee_mismatches'::bigint AS dynamic_fee_mismatches,
  :'readiness_gate_pass'::boolean AS consolidated_readiness_gate_pass;
SELECT (:'integrity_gate_pass'::boolean AND :'readiness_gate_pass'::boolean) AS gate_pass
  \gset integrity_

\echo '== ML shadow diagnostics (never a deterministic realtime-paper experiment profitability gate) =='
SELECT e.summary #>> '{ml_shadow_runtime,enqueued}' AS ml_enqueued,
       e.summary #>> '{ml_shadow_runtime,completed}' AS ml_completed,
       e.summary #>> '{ml_shadow_runtime,rejected_full}' AS ml_rejected_full,
       e.summary #>> '{ml_shadow_runtime,rejected_closed}' AS ml_rejected_closed,
       e.summary #>> '{ml_shadow_runtime,tasks_failed}' AS ml_tasks_failed,
       e.summary #>> '{ml_shadow_runtime,queue_depth}' AS ml_queue_depth,
       'diagnostic_only_unless_primary_snapshot_or_decision_loss_or_resource_instability'
         AS experiment_treatment
FROM polymarket.btc_paper_experiments e
WHERE e.experiment_id=NULLIF(:'experiment_id','')::uuid;

\echo '== Official-outcome fill-level aggregation =='
WITH selected AS (
  SELECT * FROM polymarket.btc_paper_experiments
  WHERE experiment_id=NULLIF(:'experiment_id','')::uuid
), fills AS (
  SELECT f.fill_id,f.order_id,o.market_id,m.window_end::date AS outcome_day_utc,
         f.token_id,f.size,f.price,f.fee,m.official_outcome,m.official_winning_token_id,
         m.official_resolution_source,m.official_resolution_received_at,
         (m.official_resolution_received_at-m.window_end) AS resolution_delay,
         (f.token_id=m.official_winning_token_id) AS won,
         CASE WHEN f.token_id=m.official_winning_token_id THEN f.size ELSE 0 END AS payout,
         f.price*f.size AS entry_notional,
         CASE WHEN f.token_id=m.official_winning_token_id
           THEN f.size-f.price*f.size-f.fee ELSE -f.price*f.size-f.fee END AS net_pnl
  FROM selected e JOIN polymarket.orders o ON o.process_id=e.process_id
    AND o.raw_payload #>> '{request,metadata,experiment_id}'=e.experiment_id::text
  JOIN polymarket.fills f ON f.order_id=o.order_id AND f.source='paper'
  JOIN polymarket.btc_interval_markets m ON m.market_id=o.market_id
  JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE m.official_outcome IN('up','down')
    AND m.official_winning_token_id IN(m.up_token_id,m.down_token_id)
    AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
    AND rw.status IN('resolved','resolved_late')
)
SELECT official_outcome,won,count(*) AS fill_level_rows,
       count(DISTINCT order_id) AS filled_orders,
       sum(size) AS shares,sum(entry_notional) AS entry_notional,
       sum(payout) AS payout,sum(fee) AS fees,sum(net_pnl) AS net_pnl,
       avg(net_pnl) AS mean_fill_level_net_pnl,
       min(net_pnl) AS worst_fill_level_net_pnl,max(net_pnl) AS best_fill_level_net_pnl
FROM fills
GROUP BY official_outcome,won
ORDER BY official_outcome,won;

\echo '== Decision and execution funnel =='
WITH selected AS (
  SELECT * FROM polymarket.btc_paper_experiments
  WHERE experiment_id=NULLIF(:'experiment_id','')::uuid
), snapshots AS (
  SELECT s.* FROM selected e JOIN polymarket.btc_feature_snapshots s
    ON s.features->>'process_id'=e.process_id::text
   AND s.feature_as_of>=e.started_at
   AND s.feature_as_of<=least(coalesce(e.stopped_at,now()),now())
), decisions AS (
  SELECT d.* FROM selected e JOIN polymarket.btc_strategy_decisions d
    ON d.experiment_id=e.experiment_id
), cohort_orders AS (
  SELECT o.* FROM selected e JOIN polymarket.orders o ON o.process_id=e.process_id
   AND o.raw_payload #>> '{request,metadata,experiment_id}'=e.experiment_id::text
), order_fills AS (
  SELECT DISTINCT f.order_id FROM polymarket.fills f JOIN cohort_orders o USING(order_id)
  WHERE f.source='paper'
)
SELECT step,events,distinct_markets
FROM (
  SELECT 1 AS ordinal,'feature_snapshots' AS step,count(*) AS events,count(DISTINCT market_id) AS distinct_markets FROM snapshots
  UNION ALL SELECT 2,'ready_feature_snapshots',count(*),count(DISTINCT market_id) FROM snapshots WHERE readiness_status='ready'
  UNION ALL SELECT 3,'decisions',count(*),count(DISTINCT market_id) FROM decisions
  UNION ALL SELECT 4,'decisions_with_fair_probability',count(*),count(DISTINCT market_id) FROM decisions WHERE fair_probability IS NOT NULL
  UNION ALL SELECT 5,'positive_gross_edge',count(*),count(DISTINCT market_id) FROM decisions WHERE gross_edge_per_share>0
  UNION ALL SELECT 6,'positive_after_recorded_fee',count(*),count(DISTINCT market_id) FROM decisions WHERE gross_edge_per_share-fee_per_share>0
  UNION ALL SELECT 7,'positive_after_all_reserves',count(*),count(DISTINCT market_id) FROM decisions WHERE net_edge_per_share>0
  UNION ALL SELECT 8,'approved_order_plans',count(*),count(DISTINCT market_id) FROM decisions WHERE order_plan_id IS NOT NULL
  UNION ALL SELECT 9,'paper_fok_orders',count(*),count(DISTINCT market_id) FROM cohort_orders WHERE side='buy' AND order_type='fok'
  UNION ALL SELECT 10,'paper_fok_filled_orders',count(*),count(DISTINCT market_id) FROM cohort_orders WHERE state='filled'
  UNION ALL SELECT 11,'paper_orders_with_fill_rows',count(*),count(DISTINCT o.market_id) FROM cohort_orders o JOIN order_fills f USING(order_id)
) funnel
ORDER BY ordinal;

SELECT coalesce(reject_reason,'none') AS reject_reason,status,action,
       count(*) AS decisions,count(DISTINCT market_id) AS markets
FROM polymarket.btc_strategy_decisions
WHERE experiment_id=NULLIF(:'experiment_id','')::uuid
GROUP BY coalesce(reject_reason,'none'),status,action
ORDER BY decisions DESC,reject_reason,status,action;

\echo '== Probability-uncertainty stress opportunity and fill survival =='
WITH selected AS (
  SELECT e.*,
         (e.config #>> '{raw,strategy,min_net_edge_per_share}')::numeric AS min_edge_per_share,
         (e.config #>> '{raw,strategy,min_net_edge_usd}')::numeric AS min_edge_usd
  FROM polymarket.btc_paper_experiments e
  JOIN polymarket.trading_processes p USING(process_id)
  WHERE e.experiment_id=NULLIF(:'experiment_id','')::uuid
), opportunities AS (
  SELECT d.decision_id,d.market_id,d.status,d.outcome,d.fair_probability,
         d.size,d.net_edge_per_share,
         CASE WHEN d.metadata #>> '{fair_value,probability_uncertainty}'
           ~ '^[+]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$'
           THEN (d.metadata #>> '{fair_value,probability_uncertainty}')::numeric END
           AS probability_uncertainty,
         d.fair_probability
           - (:'probability_uncertainty_multiplier'::numeric-1)
             * CASE WHEN d.metadata #>> '{fair_value,probability_uncertainty}'
                 ~ '^[+]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$'
               THEN (d.metadata #>> '{fair_value,probability_uncertainty}')::numeric END
               AS uncertainty_stressed_selected_probability,
         d.net_edge_per_share
           - (:'probability_uncertainty_multiplier'::numeric-1)
             * CASE WHEN d.metadata #>> '{fair_value,probability_uncertainty}'
                 ~ '^[+]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$'
               THEN (d.metadata #>> '{fair_value,probability_uncertainty}')::numeric END
               AS uncertainty_stressed_net_edge_per_share,
         s.min_edge_per_share,s.min_edge_usd
  FROM selected s
  JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=s.experiment_id
  WHERE d.order_plan_id IS NOT NULL
    AND (:'audit_mode'='final' OR d.decision_at<=now()-interval '10 seconds')
), survival AS (
  SELECT *,(
    probability_uncertainty BETWEEN 0 AND 1
    AND fair_probability BETWEEN 0 AND 1
    AND min_edge_per_share>=0 AND min_edge_usd>=0 AND size>0
    AND uncertainty_stressed_net_edge_per_share>=min_edge_per_share
    AND uncertainty_stressed_net_edge_per_share*size>=min_edge_usd
  ) AS survives
  FROM opportunities
)
SELECT count(*) AS approved_opportunities,
       count(*) FILTER(WHERE NOT coalesce(
         fair_probability BETWEEN 0 AND 1 AND probability_uncertainty BETWEEN 0 AND 1
         AND min_edge_per_share>=0 AND min_edge_usd>=0 AND size>0,false))
         AS invalid_or_missing_uncertainty_inputs,
       avg(fair_probability) AS mean_selected_conservative_probability,
       avg(uncertainty_stressed_selected_probability)
         AS mean_uncertainty_stressed_selected_probability,
       count(*) FILTER(WHERE survives) AS uncertainty_stressed_surviving_opportunities,
       round(100.0*count(*) FILTER(WHERE survives)/nullif(count(*),0),4)
         AS opportunity_survival_pct,
       count(*) FILTER(WHERE status='filled') AS originally_filled_opportunities,
       count(*) FILTER(WHERE status='filled' AND survives)
         AS uncertainty_stressed_surviving_filled_opportunities,
       round(100.0*count(*) FILTER(WHERE status='filled' AND survives)
             /nullif(count(*) FILTER(WHERE status='filled'),0),4)
         AS filled_opportunity_survival_pct,
       :'probability_uncertainty_multiplier'::numeric AS uncertainty_multiplier
FROM survival;

\echo '== Qualifying-fill probability-uncertainty input integrity =='
WITH selected AS (
  SELECT e.*,
    (e.config #>> '{raw,strategy,min_net_edge_per_share}')::numeric AS min_edge_per_share,
    (e.config #>> '{raw,strategy,min_net_edge_usd}')::numeric AS min_edge_usd
  FROM polymarket.btc_paper_experiments e
  JOIN polymarket.trading_processes p USING(process_id)
  WHERE e.experiment_id=NULLIF(:'experiment_id','')::uuid
), filled AS (
  SELECT o.order_id,d.decision_id,d.outcome,d.fair_probability,d.net_edge_per_share,d.size,
    CASE WHEN d.metadata #>> '{fair_value,probability_uncertainty}'
      ~ '^[+]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$'
      THEN (d.metadata #>> '{fair_value,probability_uncertainty}')::numeric END AS uncertainty,
    s.min_edge_per_share,s.min_edge_usd
  FROM selected s
  JOIN polymarket.orders o ON o.process_id=s.process_id
   AND o.raw_payload #>> '{request,metadata,experiment_id}'=s.experiment_id::text
   AND o.state='filled' AND o.side='buy' AND o.order_type='fok'
  LEFT JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=s.experiment_id
   AND d.decision_id::text=o.raw_payload #>> '{request,metadata,decision_id}'
)
SELECT count(*) AS qualifying_filled_orders,
  count(*) FILTER(WHERE NOT coalesce(
    decision_id IS NOT NULL AND outcome IN('up','down')
    AND fair_probability BETWEEN 0 AND 1
    AND fair_probability::text NOT IN('NaN','Infinity','-Infinity')
    AND net_edge_per_share IS NOT NULL
    AND net_edge_per_share::text NOT IN('NaN','Infinity','-Infinity')
    AND size>0 AND size::text NOT IN('NaN','Infinity','-Infinity')
    AND uncertainty IS NOT NULL AND uncertainty BETWEEN 0 AND 1
    AND uncertainty::text NOT IN('NaN','Infinity','-Infinity')
    AND min_edge_per_share>=0 AND min_edge_usd>=0
    AND min_edge_per_share::text NOT IN('NaN','Infinity','-Infinity')
    AND min_edge_usd::text NOT IN('NaN','Infinity','-Infinity'),false))
    AS input_violations
FROM filled \gset uncertainty_

SELECT :'uncertainty_qualifying_filled_orders'::bigint AS qualifying_filled_orders,
  :'uncertainty_input_violations'::bigint AS probability_uncertainty_input_violations;
SELECT (:'integrity_gate_pass'::boolean AND :'uncertainty_input_violations'::bigint=0)
  AS gate_pass \gset integrity_

\echo '== Non-mutating online latency/depth counterfactual previews =='
WITH selected AS (
  SELECT e.*,e.config AS process_config
  FROM polymarket.btc_paper_experiments e
  JOIN polymarket.trading_processes p USING(process_id)
  WHERE e.experiment_id=NULLIF(:'experiment_id','')::uuid
), expected_scenarios AS (
  SELECT item->>'scenario_key' AS scenario_key,
         (item #>> '{arrival_latency,secs}')::numeric*1000
           +(item #>> '{arrival_latency,nanos}')::numeric/1000000
             AS arrival_latency_ms,
         (item->>'visible_depth_haircut')::numeric AS visible_depth_haircut
  FROM selected s
  CROSS JOIN LATERAL jsonb_array_elements(CASE
    WHEN jsonb_typeof((:'expected_experiment_config_contract'::jsonb)
      #> '{raw,paper,stress_previews}')='array'
      THEN (:'expected_experiment_config_contract'::jsonb)
        #> '{raw,paper,stress_previews}'
    WHEN jsonb_typeof(s.process_config #> '{raw,paper,stress_previews}')='array'
      THEN s.process_config #> '{raw,paper,stress_previews}'
    ELSE '[]'::jsonb END
  ) a(item)
), eligible AS (
  SELECT d.*,m.window_end,m.official_winning_token_id,
         (
           m.official_outcome IN('up','down')
           AND m.official_winning_token_id IN(m.up_token_id,m.down_token_id)
           AND m.official_resolved_at>=m.window_end
           AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
           AND m.official_resolution_received_at>=m.window_end
           AND jsonb_typeof(m.official_resolution_payload)='object'
           AND rw.status IN('resolved','resolved_late')
           AND rw.resolution_source=m.official_resolution_source
           AND rw.resolution_received_at=m.official_resolution_received_at
         ) AS officially_resolved
  FROM selected e
  JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=e.experiment_id
  JOIN polymarket.btc_interval_markets m ON m.market_id=d.market_id
  LEFT JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE d.order_plan_id IS NOT NULL
), expected AS (
  SELECT e.*,s.scenario_key,s.arrival_latency_ms,s.visible_depth_haircut
  FROM eligible e CROSS JOIN expected_scenarios s
), telemetry AS (
  SELECT x.decision_id,x.market_id,x.window_end,x.token_id,x.size,x.official_winning_token_id,
         x.officially_resolved,x.scenario_key,x.arrival_latency_ms,x.visible_depth_haircut,
         x.metadata #>> '{paper_stress_previews,telemetry_only}' AS telemetry_only,
         x.metadata #>> '{paper_stress_previews,influenced_primary_execution}'
           AS influenced_primary_execution,
         x.metadata #>> '{paper_stress_previews,primary_config_hash}' AS primary_config_hash,
         count(p.item) AS telemetry_rows,
         count(p.item) FILTER(WHERE p.item->>'status'='observed') AS observed_rows,
         count(p.item) FILTER(WHERE p.item->>'status'='error') AS error_rows,
         max(p.item #>> '{result,state}') AS preview_state,
         max(nullif(p.item #>> '{result,filled_size}','')::numeric) AS filled_size,
         max(nullif(p.item #>> '{result,filled_notional}','')::numeric) AS filled_notional,
         max(nullif(p.item #>> '{result,fees}','')::numeric) AS fees,
         max(p.item #>> '{result,reject_reason}') AS reject_reason,
         max(nullif(p.item #>> '{result,execution_metadata,paper_execution,configured_latency_ms}','')::numeric)
           AS observed_configured_latency_ms,
         max(nullif(p.item #>> '{result,execution_metadata,paper_execution,visible_depth_haircut}','')::numeric)
           AS observed_visible_depth_haircut,
         bool_and(coalesce((p.item #>> '{result,execution_metadata,paper_execution,non_mutating_preview}')::boolean,false))
           FILTER(WHERE p.item IS NOT NULL) AS non_mutating_preview,
         bool_and(NOT coalesce((p.item #>> '{result,execution_metadata,paper_execution,collateral_enforced}')::boolean,true))
           FILTER(WHERE p.item IS NOT NULL) AS collateral_not_enforced,
         bool_and(p.item #>> '{result,execution_metadata,paper_execution,preview_scenario_key}'=x.scenario_key)
           FILTER(WHERE p.item IS NOT NULL) AS preview_scenario_matches
  FROM expected x
  LEFT JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(x.metadata #> '{paper_stress_previews,scenarios}')='array'
      THEN x.metadata #> '{paper_stress_previews,scenarios}' ELSE '[]'::jsonb END
  ) p(item) ON coalesce(p.item #>> '{result,scenario_key}',p.item->>'scenario_key')=x.scenario_key
  GROUP BY x.decision_id,x.market_id,x.window_end,x.token_id,x.size,
           x.official_winning_token_id,x.officially_resolved,x.scenario_key,
           x.arrival_latency_ms,x.visible_depth_haircut,x.metadata,x.config_hash
), scored AS (
  SELECT t.*,
         (
           telemetry_rows=1 AND observed_rows=1 AND error_rows=0
           AND telemetry_only='true' AND influenced_primary_execution='false'
           AND primary_config_hash IS NOT DISTINCT FROM (
             SELECT config_hash FROM selected LIMIT 1
           )
           AND observed_configured_latency_ms=arrival_latency_ms
           AND observed_visible_depth_haircut=visible_depth_haircut
           AND coalesce(non_mutating_preview,false)
           AND coalesce(collateral_not_enforced,false)
           AND coalesce(preview_scenario_matches,false)
           AND (
             (preview_state='filled' AND filled_size=size AND filled_size>0
               AND filled_notional IS NOT NULL AND filled_notional>=0
               AND filled_notional<=filled_size AND fees IS NOT NULL AND fees>=0)
             OR (preview_state='rejected' AND filled_size=0
               AND filled_notional=0 AND fees=0)
           )
         ) AS telemetry_valid,
         CASE
           WHEN NOT officially_resolved OR preview_state IS DISTINCT FROM 'filled' THEN 0::numeric
           WHEN token_id=official_winning_token_id
             THEN filled_size-filled_notional-fees
           ELSE -filled_notional-fees
         END AS official_counterfactual_net_pnl
  FROM telemetry t
)
SELECT scenario_key,
       count(*) AS primary_eligible_orders,
       count(*) FILTER(WHERE officially_resolved) AS officially_resolved_eligible_orders,
       count(*) FILTER(WHERE telemetry_valid) AS valid_preview_rows,
       count(*) FILTER(WHERE error_rows>0) AS preview_error_rows,
       count(*) FILTER(WHERE preview_state='filled') AS counterfactual_fok_fills,
       count(*) FILTER(WHERE preview_state='filled' AND officially_resolved)
         AS officially_resolved_counterfactual_fok_fills,
       round(100.0*count(*) FILTER(WHERE preview_state='filled')/nullif(count(*),0),4)
         AS counterfactual_fok_fill_rate_pct,
       sum(official_counterfactual_net_pnl) AS official_counterfactual_net_pnl,
       sum(official_counterfactual_net_pnl)/nullif(count(*),0)
         AS mean_net_pnl_per_primary_eligible_order,
       bool_and(telemetry_valid) AS scenario_telemetry_gate_pass
FROM scored
GROUP BY scenario_key
ORDER BY scenario_key;

WITH selected AS (
  SELECT e.*,e.config AS process_config
  FROM polymarket.btc_paper_experiments e JOIN polymarket.trading_processes p USING(process_id)
  WHERE e.experiment_id=NULLIF(:'experiment_id','')::uuid
), expected_scenarios AS (
  SELECT item->>'scenario_key' AS scenario_key,
         (item #>> '{arrival_latency,secs}')::numeric*1000
           +(item #>> '{arrival_latency,nanos}')::numeric/1000000
             AS arrival_latency_ms,
         (item->>'visible_depth_haircut')::numeric AS visible_depth_haircut
  FROM selected s
  CROSS JOIN LATERAL jsonb_array_elements(CASE
    WHEN jsonb_typeof((:'expected_experiment_config_contract'::jsonb)
      #> '{raw,paper,stress_previews}')='array'
      THEN (:'expected_experiment_config_contract'::jsonb)
        #> '{raw,paper,stress_previews}'
    WHEN jsonb_typeof(s.process_config #> '{raw,paper,stress_previews}')='array'
      THEN s.process_config #> '{raw,paper,stress_previews}'
    ELSE '[]'::jsonb END
  ) a(item)
), definition_sources AS (
  SELECT item
  FROM selected s
  CROSS JOIN LATERAL jsonb_array_elements(CASE
    WHEN jsonb_typeof(s.process_config #> '{raw,paper,stress_previews}')='array'
      THEN s.process_config #> '{raw,paper,stress_previews}' ELSE '[]'::jsonb END) a(item)
), definition_check AS (
  SELECT s.scenario_key,count(d.item) AS matching_definitions
  FROM expected_scenarios s
  LEFT JOIN definition_sources d ON d.item->>'scenario_key'=s.scenario_key
    AND (
      (d.item #>> '{arrival_latency,secs}')::numeric*1000
      +(d.item #>> '{arrival_latency,nanos}')::numeric/1000000
    )=s.arrival_latency_ms
    AND (d.item->>'visible_depth_haircut')::numeric=s.visible_depth_haircut
  GROUP BY s.scenario_key
), eligible AS (
  SELECT d.*,m.official_winning_token_id,
    (m.official_outcome IN('up','down')
      AND m.official_winning_token_id IN(m.up_token_id,m.down_token_id)
      AND m.official_resolved_at>=m.window_end
      AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
      AND m.official_resolution_received_at>=m.window_end
      AND jsonb_typeof(m.official_resolution_payload)='object'
      AND rw.status IN('resolved','resolved_late')
      AND rw.resolution_source=m.official_resolution_source
      AND rw.resolution_received_at=m.official_resolution_received_at) AS officially_resolved
  FROM selected e JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=e.experiment_id
  JOIN polymarket.btc_interval_markets m ON m.market_id=d.market_id
  LEFT JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE d.order_plan_id IS NOT NULL
    AND (:'audit_mode'='final' OR d.decision_at<=now()-interval '10 seconds')
), expected AS (
  SELECT e.*,s.scenario_key,s.arrival_latency_ms,s.visible_depth_haircut
  FROM eligible e CROSS JOIN expected_scenarios s
), telemetry AS (
  SELECT x.decision_id,x.token_id,x.size,x.config_hash,x.official_winning_token_id,
    x.officially_resolved,x.scenario_key,x.arrival_latency_ms,x.visible_depth_haircut,x.metadata,
    count(p.item) AS rows,count(p.item) FILTER(WHERE p.item->>'status'='observed') AS observed,
    count(p.item) FILTER(WHERE p.item->>'status'='error') AS errors,
    max(p.item #>> '{result,state}') AS state,
    max(nullif(p.item #>> '{result,filled_size}','')::numeric) AS filled_size,
    max(nullif(p.item #>> '{result,filled_notional}','')::numeric) AS notional,
    max(nullif(p.item #>> '{result,fees}','')::numeric) AS fees,
    max(nullif(p.item #>> '{result,execution_metadata,paper_execution,configured_latency_ms}','')::numeric) AS latency,
    max(nullif(p.item #>> '{result,execution_metadata,paper_execution,visible_depth_haircut}','')::numeric) AS haircut,
    bool_and(coalesce((p.item #>> '{result,execution_metadata,paper_execution,non_mutating_preview}')::boolean,false))
      FILTER(WHERE p.item IS NOT NULL) AS nonmutating,
    bool_and(NOT coalesce((p.item #>> '{result,execution_metadata,paper_execution,collateral_enforced}')::boolean,true))
      FILTER(WHERE p.item IS NOT NULL) AS no_collateral,
    bool_and(p.item #>> '{result,execution_metadata,paper_execution,preview_scenario_key}'=x.scenario_key)
      FILTER(WHERE p.item IS NOT NULL) AS scenario_matches
  FROM expected x LEFT JOIN LATERAL jsonb_array_elements(CASE
    WHEN jsonb_typeof(x.metadata #> '{paper_stress_previews,scenarios}')='array'
      THEN x.metadata #> '{paper_stress_previews,scenarios}' ELSE '[]'::jsonb END) p(item)
    ON coalesce(p.item #>> '{result,scenario_key}',p.item->>'scenario_key')=x.scenario_key
  GROUP BY x.decision_id,x.token_id,x.size,x.config_hash,x.official_winning_token_id,
    x.officially_resolved,x.scenario_key,x.arrival_latency_ms,x.visible_depth_haircut,x.metadata
), scored AS (
  SELECT t.*,
    (rows=1 AND observed=1 AND errors=0
      AND metadata #>> '{paper_stress_previews,telemetry_only}'='true'
      AND metadata #>> '{paper_stress_previews,influenced_primary_execution}'='false'
      AND metadata #>> '{paper_stress_previews,primary_config_hash}'=config_hash
      AND latency=arrival_latency_ms AND haircut=visible_depth_haircut
      AND coalesce(nonmutating,false) AND coalesce(no_collateral,false)
      AND coalesce(scenario_matches,false)
      AND ((state='filled' AND filled_size=size AND filled_size>0
              AND notional IS NOT NULL AND notional>=0 AND notional<=filled_size
              AND fees IS NOT NULL AND fees>=0)
        OR (state='rejected' AND filled_size=0 AND notional=0 AND fees=0)))
        AS valid,
    CASE WHEN NOT officially_resolved OR state IS DISTINCT FROM 'filled' THEN 0::numeric
      WHEN token_id=official_winning_token_id THEN filled_size-notional-fees
      ELSE -notional-fees END AS net_pnl
  FROM telemetry t
), scenario_summary AS (
  SELECT scenario_key,count(*) AS eligible,count(*) FILTER(WHERE officially_resolved) AS official,
    count(*) FILTER(WHERE state='filled') AS preview_fills,
    count(*) FILTER(WHERE state='filled' AND officially_resolved) AS official_preview_fills,
    count(*) FILTER(WHERE valid) AS valid,sum(net_pnl)/nullif(count(*),0) AS mean_net
  FROM scored GROUP BY scenario_key
), total AS (
  SELECT (SELECT count(*) FROM eligible) AS eligible_orders,
    count(*) AS scenario_count,
    coalesce(bool_and(eligible>0 AND valid=eligible),false) AS telemetry_pass,
    coalesce(bool_and(preview_fills=official_preview_fills),false) AS economic_coverage_pass,
    coalesce(min(mean_net),-999999) AS minimum_scenario_mean
  FROM scenario_summary
), definitions AS (
  SELECT coalesce(count(*)=(SELECT count(*) FROM expected_scenarios)
    AND (SELECT count(*) FROM definition_sources)=(SELECT count(*) FROM expected_scenarios)
    AND bool_and(matching_definitions=1),false) AS pass
  FROM definition_check
)
SELECT t.eligible_orders,t.scenario_count,t.minimum_scenario_mean,
  (d.pass AND (t.eligible_orders=0 OR (t.telemetry_pass AND t.scenario_count=2)))
    AS telemetry_gate_pass,
  (t.scenario_count=2 AND t.economic_coverage_pass AND t.minimum_scenario_mean>0)
    AS economic_gate_pass
FROM total t CROSS JOIN definitions d \gset preview_

\echo '== Order-level official-outcome economics and tail statistics =='
WITH selected AS (
  SELECT e.*,
         (e.config #>> '{raw,strategy,min_net_edge_per_share}')::numeric AS min_edge_per_share,
         (e.config #>> '{raw,strategy,min_net_edge_usd}')::numeric AS min_edge_usd
  FROM polymarket.btc_paper_experiments e
  JOIN polymarket.trading_processes p USING(process_id)
  WHERE e.experiment_id=NULLIF(:'experiment_id','')::uuid
), trades AS (
  SELECT o.order_id,o.market_id,m.window_end,m.window_end::date AS outcome_day_utc,
         d.decision_id,d.outcome,d.net_edge_per_share,d.size AS decision_size,
         CASE WHEN d.metadata #>> '{fair_value,probability_uncertainty}'
           ~ '^[+]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$'
           THEN (d.metadata #>> '{fair_value,probability_uncertainty}')::numeric END
           AS probability_uncertainty,
         s.min_edge_per_share,s.min_edge_usd,
         fs.seconds_to_close,fs.realized_vol_30s_bps,
         CASE WHEN d.outcome='up' THEN (fs.up_best_ask-fs.up_best_bid)*10000
              WHEN d.outcome='down' THEN (fs.down_best_ask-fs.down_best_bid)*10000 END
           AS quoted_spread_bps,
         (o.raw_payload #>> '{request,metadata,paper_execution,observed_submit_to_arrival_ms}')::numeric
           AS observed_arrival_latency_ms,
         sum(f.size) AS filled_size,sum(f.price*f.size) AS entry_notional,
         sum(f.fee) AS fees,
         sum(CASE WHEN f.token_id=m.official_winning_token_id THEN f.size ELSE 0 END) AS payout,
         sum(CASE WHEN f.token_id=m.official_winning_token_id
               THEN f.size-f.price*f.size ELSE -f.price*f.size END) AS gross_pnl,
         sum(CASE WHEN f.token_id=m.official_winning_token_id
               THEN f.size-f.price*f.size-f.fee ELSE -f.price*f.size-f.fee END) AS net_pnl,
         bool_or(f.token_id=m.official_winning_token_id) AS won
  FROM selected s
  JOIN polymarket.orders o ON o.process_id=s.process_id
   AND o.raw_payload #>> '{request,metadata,experiment_id}'=s.experiment_id::text
   AND o.state='filled' AND o.side='buy' AND o.order_type='fok'
  JOIN polymarket.fills f ON f.order_id=o.order_id AND f.source='paper'
  JOIN polymarket.btc_strategy_decisions d
    ON d.experiment_id=s.experiment_id
   AND d.decision_id::text=o.raw_payload #>> '{request,metadata,decision_id}'
  JOIN polymarket.btc_feature_snapshots fs ON fs.snapshot_id=d.snapshot_id
  JOIN polymarket.btc_interval_markets m ON m.market_id=o.market_id
  JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE m.official_outcome IN('up','down')
    AND m.official_winning_token_id IN(m.up_token_id,m.down_token_id)
    AND m.official_resolved_at>=m.window_end
    AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
    AND m.official_resolution_received_at>=m.window_end
    AND jsonb_typeof(m.official_resolution_payload)='object'
    AND rw.status IN('resolved','resolved_late')
    AND rw.resolution_source=m.official_resolution_source
    AND rw.resolution_received_at=m.official_resolution_received_at
  GROUP BY o.order_id,o.market_id,m.window_end,d.decision_id,d.outcome,d.net_edge_per_share,
           d.size,d.metadata,s.min_edge_per_share,s.min_edge_usd,fs.seconds_to_close,
           fs.realized_vol_30s_bps,fs.up_best_ask,fs.up_best_bid,fs.down_best_ask,
           fs.down_best_bid,o.raw_payload
), ordered AS (
  SELECT t.*,row_number() OVER(ORDER BY window_end,order_id) AS sequence,
         sum(net_pnl) OVER(ORDER BY window_end,order_id ROWS UNBOUNDED PRECEDING)
           AS cumulative_net_pnl
  FROM trades t
), curve AS (
  SELECT o.*,greatest(0::numeric,max(cumulative_net_pnl) OVER(
           ORDER BY sequence ROWS UNBOUNDED PRECEDING)) AS running_peak
  FROM ordered o
), ranked_tail AS (
  SELECT t.*,row_number() OVER(ORDER BY net_pnl,window_end,order_id) AS loss_rank,
         count(*) OVER() AS trade_count
  FROM trades t
)
SELECT count(*) AS qualifying_filled_orders,
       count(*) FILTER(WHERE won) AS winners,
       round(100.0*count(*) FILTER(WHERE won)/nullif(count(*),0),4) AS win_rate_pct,
       sum(gross_pnl) AS gross_pnl,sum(fees) AS fees,sum(net_pnl) AS net_pnl,
       avg(net_pnl) AS mean_net_pnl_per_filled_order,
       stddev_samp(net_pnl) AS sample_stddev_net_pnl,
       min(net_pnl) AS worst_filled_order_net_pnl,
       percentile_cont(0.05) WITHIN GROUP(ORDER BY net_pnl) AS p05_net_pnl,
       (SELECT avg(net_pnl) FROM ranked_tail
         WHERE loss_rank<=greatest(1,ceil(trade_count*0.05)::bigint)) AS lower_5pct_expected_shortfall,
       max(net_pnl) AS best_filled_order_net_pnl,
       -(SELECT min(cumulative_net_pnl-running_peak) FROM curve) AS maximum_drawdown_usd
FROM trades;

\echo '== Daily PnL and cumulative drawdown =='
WITH selected AS (
  SELECT * FROM polymarket.btc_paper_experiments
  WHERE experiment_id=NULLIF(:'experiment_id','')::uuid
), trades AS (
  SELECT o.order_id,m.window_end,m.window_end::date AS outcome_day_utc,
         sum(CASE WHEN f.token_id=m.official_winning_token_id THEN f.size ELSE 0 END) AS payout,
         sum(f.price*f.size) AS entry_notional,sum(f.fee) AS fees,
         sum(CASE WHEN f.token_id=m.official_winning_token_id
           THEN f.size-f.price*f.size-f.fee ELSE -f.price*f.size-f.fee END) AS net_pnl,
         bool_or(f.token_id=m.official_winning_token_id) AS won
  FROM selected s JOIN polymarket.orders o ON o.process_id=s.process_id
   AND o.raw_payload #>> '{request,metadata,experiment_id}'=s.experiment_id::text
   AND o.state='filled' AND o.side='buy' AND o.order_type='fok'
  JOIN polymarket.fills f ON f.order_id=o.order_id AND f.source='paper'
  JOIN polymarket.btc_interval_markets m ON m.market_id=o.market_id
  JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE m.official_outcome IN('up','down')
    AND m.official_winning_token_id IN(m.up_token_id,m.down_token_id)
    AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
    AND m.official_resolution_received_at>=m.window_end
    AND rw.status IN('resolved','resolved_late')
  GROUP BY o.order_id,m.window_end
), daily AS (
  SELECT outcome_day_utc,count(*) AS filled_orders,
         count(*) FILTER(WHERE won) AS winners,
         sum(entry_notional) AS entry_notional,sum(fees) AS fees,sum(net_pnl) AS net_pnl,
         avg(net_pnl) AS mean_net_pnl,min(net_pnl) AS worst_trade,max(net_pnl) AS best_trade
  FROM trades GROUP BY outcome_day_utc
), running AS (
  SELECT d.*,sum(net_pnl) OVER(ORDER BY outcome_day_utc) AS cumulative_net_pnl
  FROM daily d
), curve AS (
  SELECT r.*,greatest(0::numeric,max(cumulative_net_pnl) OVER(
    ORDER BY outcome_day_utc ROWS UNBOUNDED PRECEDING)) AS running_peak
  FROM running r
)
SELECT *,cumulative_net_pnl-running_peak AS drawdown_from_prior_peak
FROM curve ORDER BY outcome_day_utc;

WITH selected AS (
  SELECT e.*,
         (e.config #>> '{raw,strategy,min_net_edge_per_share}')::numeric AS min_edge_per_share,
         (e.config #>> '{raw,strategy,min_net_edge_usd}')::numeric AS min_edge_usd
  FROM polymarket.btc_paper_experiments e JOIN polymarket.trading_processes p USING(process_id)
  WHERE e.experiment_id=NULLIF(:'experiment_id','')::uuid
), trades AS (
  SELECT o.order_id,m.window_end,m.window_end::date AS outcome_day_utc,d.net_edge_per_share,d.size,
         CASE WHEN d.metadata #>> '{fair_value,probability_uncertainty}'
           ~ '^[+]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$'
           THEN (d.metadata #>> '{fair_value,probability_uncertainty}')::numeric END
           AS probability_uncertainty,
         s.min_edge_per_share,s.min_edge_usd,
         sum(f.price*f.size) AS entry_notional,sum(f.fee) AS fees,
         sum(CASE WHEN f.token_id=m.official_winning_token_id THEN f.size ELSE 0 END) AS payout,
         sum(CASE WHEN f.token_id=m.official_winning_token_id
           THEN f.size-f.price*f.size-f.fee ELSE -f.price*f.size-f.fee END) AS net_pnl
  FROM selected s JOIN polymarket.orders o ON o.process_id=s.process_id
   AND o.raw_payload #>> '{request,metadata,experiment_id}'=s.experiment_id::text
   AND o.state='filled' AND o.side='buy' AND o.order_type='fok'
  JOIN polymarket.fills f ON f.order_id=o.order_id AND f.source='paper'
  JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=s.experiment_id
   AND d.decision_id::text=o.raw_payload #>> '{request,metadata,decision_id}'
  JOIN polymarket.btc_interval_markets m ON m.market_id=o.market_id
  JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE m.official_outcome IN('up','down')
    AND m.official_winning_token_id IN(m.up_token_id,m.down_token_id)
    AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
    AND m.official_resolution_received_at>=m.window_end
    AND jsonb_typeof(m.official_resolution_payload)='object'
    AND rw.status IN('resolved','resolved_late')
  GROUP BY o.order_id,m.window_end,d.net_edge_per_share,d.size,d.metadata,
           s.min_edge_per_share,s.min_edge_usd
), ranked AS (
  SELECT t.*,row_number() OVER(ORDER BY net_pnl DESC,window_end,order_id) AS profit_rank,
         count(*) OVER() AS total_trades,
         sum(net_pnl) OVER(ORDER BY window_end,order_id ROWS UNBOUNDED PRECEDING) AS cumulative_net
  FROM trades t
), stressed AS (
  SELECT r.*,
         CASE WHEN profit_rank<=ceil(total_trades*:'missed_best_fill_fraction'::numeric)
           THEN 0::numeric
           ELSE payout-entry_notional-fees*:'fee_stress_multiplier'::numeric END
             AS fee_and_missed_fill_stressed_net_pnl,
         CASE WHEN NOT coalesce(
           probability_uncertainty BETWEEN 0 AND 1
           AND min_edge_per_share>=0 AND min_edge_usd>=0 AND size>0
           AND net_edge_per_share::text NOT IN('NaN','Infinity','-Infinity'),false)
           THEN net_pnl
         WHEN
           net_edge_per_share
             - (:'probability_uncertainty_multiplier'::numeric-1)*probability_uncertainty
               >= min_edge_per_share
           AND (
             net_edge_per_share
               - (:'probability_uncertainty_multiplier'::numeric-1)*probability_uncertainty
           )*size>=min_edge_usd
           THEN net_pnl ELSE 0::numeric END AS uncertainty_stressed_net_pnl
  FROM ranked r
), curve AS (
  SELECT s.*,greatest(0::numeric,max(cumulative_net) OVER(
    ORDER BY window_end,order_id ROWS UNBOUNDED PRECEDING)) AS running_peak
  FROM stressed s
), daily AS (
  SELECT outcome_day_utc,count(*) AS fills,sum(net_pnl) AS net_pnl,
         sum(fee_and_missed_fill_stressed_net_pnl) AS fee_missed_stressed_net_pnl,
         sum(uncertainty_stressed_net_pnl) AS uncertainty_stressed_net_pnl
  FROM stressed GROUP BY outcome_day_utc
), totals AS (
  SELECT count(*) AS fills,count(DISTINCT outcome_day_utc) AS utc_days,
         sum(net_pnl) AS net_pnl,avg(net_pnl) AS mean_net,
         sum(fee_and_missed_fill_stressed_net_pnl) AS fee_missed_stressed_net,
         avg(fee_and_missed_fill_stressed_net_pnl) AS fee_missed_stressed_mean,
         sum(uncertainty_stressed_net_pnl) AS uncertainty_stressed_net,
         avg(uncertainty_stressed_net_pnl) AS uncertainty_stressed_mean,
         min(net_pnl) AS worst_trade,
         -(SELECT min(cumulative_net-running_peak) FROM curve) AS max_drawdown
  FROM stressed
), concentration AS (
  SELECT coalesce(max(greatest(net_pnl,0))/nullif(sum(greatest(net_pnl,0)),0),1)
           AS maximum_positive_day_share
  FROM daily
), leave_one_out AS (
  SELECT min((t.net_pnl-d.net_pnl)/nullif(t.fills-d.fills,0)) AS minimum_loo_expectancy
  FROM daily d CROSS JOIN totals t
  WHERE t.fills>d.fills
)
SELECT t.*,c.maximum_positive_day_share,l.minimum_loo_expectancy,
       (
         t.fills>=:'minimum_qualifying_fills'::bigint
         AND t.utc_days>=:'minimum_distinct_utc_days'::bigint
         AND t.mean_net>0
         AND t.fee_missed_stressed_mean>0
         AND t.uncertainty_stressed_mean>0
         AND t.max_drawdown<=:'maximum_drawdown_usd'::numeric
         AND t.worst_trade>=:'minimum_worst_trade_pnl_usd'::numeric
         AND c.maximum_positive_day_share<=:'maximum_concentration_fraction'::numeric
         AND coalesce(l.minimum_loo_expectancy,-999999)>0
       ) AS economic_point_tail_concentration_gate_pass
FROM totals t CROSS JOIN concentration c CROSS JOIN leave_one_out l;

WITH selected AS (
  SELECT e.*,(e.config #>> '{raw,strategy,min_net_edge_per_share}')::numeric AS min_edge_per_share,
         (e.config #>> '{raw,strategy,min_net_edge_usd}')::numeric AS min_edge_usd
  FROM polymarket.btc_paper_experiments e JOIN polymarket.trading_processes p USING(process_id)
  WHERE e.experiment_id=NULLIF(:'experiment_id','')::uuid
), trades AS (
  SELECT o.order_id,m.window_end,m.window_end::date AS day,d.net_edge_per_share,d.size,
    CASE WHEN d.metadata #>> '{fair_value,probability_uncertainty}'
      ~ '^[+]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$'
      THEN (d.metadata #>> '{fair_value,probability_uncertainty}')::numeric END AS uncertainty,
    s.min_edge_per_share,s.min_edge_usd,sum(f.price*f.size) AS entry_notional,sum(f.fee) AS fees,
    sum(CASE WHEN f.token_id=m.official_winning_token_id THEN f.size ELSE 0 END) AS payout,
    sum(CASE WHEN f.token_id=m.official_winning_token_id THEN f.size-f.price*f.size-f.fee
      ELSE -f.price*f.size-f.fee END) AS net_pnl
  FROM selected s JOIN polymarket.orders o ON o.process_id=s.process_id
   AND o.raw_payload #>> '{request,metadata,experiment_id}'=s.experiment_id::text
   AND o.state='filled' AND o.side='buy' AND o.order_type='fok'
  JOIN polymarket.fills f ON f.order_id=o.order_id AND f.source='paper'
  JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=s.experiment_id
   AND d.decision_id::text=o.raw_payload #>> '{request,metadata,decision_id}'
  JOIN polymarket.btc_interval_markets m ON m.market_id=o.market_id
  JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE m.official_outcome IN('up','down') AND m.official_winning_token_id IN(m.up_token_id,m.down_token_id)
   AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
   AND m.official_resolution_received_at>=m.window_end AND rw.status IN('resolved','resolved_late')
  GROUP BY o.order_id,m.window_end,d.net_edge_per_share,d.size,d.metadata,
    s.min_edge_per_share,s.min_edge_usd
), ranked AS (
  SELECT t.*,row_number() OVER(ORDER BY net_pnl DESC,window_end,order_id) AS profit_rank,
    count(*) OVER() AS total_trades,
    sum(net_pnl) OVER(ORDER BY window_end,order_id ROWS UNBOUNDED PRECEDING) AS cumulative_net
  FROM trades t
), stressed AS (
  SELECT r.*,CASE WHEN profit_rank<=ceil(total_trades*:'missed_best_fill_fraction'::numeric)
      THEN 0::numeric ELSE payout-entry_notional-fees*:'fee_stress_multiplier'::numeric END AS fee_stress,
    CASE WHEN NOT coalesce(uncertainty BETWEEN 0 AND 1
         AND min_edge_per_share>=0 AND min_edge_usd>=0 AND size>0
         AND net_edge_per_share::text NOT IN('NaN','Infinity','-Infinity'),false)
      THEN net_pnl
    WHEN net_edge_per_share-(:'probability_uncertainty_multiplier'::numeric-1)*uncertainty
          >=min_edge_per_share
       AND (net_edge_per_share-(:'probability_uncertainty_multiplier'::numeric-1)*uncertainty)*size
          >=min_edge_usd THEN net_pnl ELSE 0::numeric END AS uncertainty_stress
  FROM ranked r
), curve AS (
  SELECT s.*,greatest(0::numeric,max(cumulative_net) OVER(
    ORDER BY window_end,order_id ROWS UNBOUNDED PRECEDING)) AS peak FROM stressed s
), daily AS (
  SELECT day,count(*) AS fills,sum(net_pnl) AS net,sum(fee_stress) AS fee_stress,
    sum(uncertainty_stress) AS uncertainty_stress FROM stressed GROUP BY day
), totals AS (
  SELECT count(*) AS fills,count(DISTINCT day) AS days,
    coalesce(avg(net_pnl),-999999) AS mean_net,
    coalesce(avg(fee_stress),-999999) AS fee_stress_mean,
    coalesce(avg(uncertainty_stress),-999999) AS uncertainty_stress_mean,
    coalesce(min(net_pnl),-999999) AS worst,
    coalesce(-(SELECT min(cumulative_net-peak) FROM curve),999999) AS drawdown,
    coalesce(sum(net_pnl),0) AS total_net FROM stressed
), concentration AS (
  SELECT coalesce(max(greatest(net,0))/nullif(sum(greatest(net,0)),0),1) AS max_day_share FROM daily
), loo AS (
  SELECT min((t.total_net-d.net)/nullif(t.fills-d.fills,0)) AS min_loo
  FROM daily d CROSS JOIN totals t WHERE t.fills>d.fills
)
SELECT t.fills,t.days,t.mean_net,t.fee_stress_mean,t.uncertainty_stress_mean,t.worst,t.drawdown,
  c.max_day_share,coalesce(l.min_loo,-999999) AS min_loo,
  (t.fills>=:'minimum_qualifying_fills'::bigint
   AND t.days>=:'minimum_distinct_utc_days'::bigint
   AND t.mean_net>0 AND t.fee_stress_mean>0 AND t.uncertainty_stress_mean>0
   AND t.worst>=:'minimum_worst_trade_pnl_usd'::numeric
   AND t.drawdown<=:'maximum_drawdown_usd'::numeric
   AND c.max_day_share<=:'maximum_concentration_fraction'::numeric
   AND coalesce(l.min_loo,-999999)>0) AS gate_pass
FROM totals t CROSS JOIN concentration c CROSS JOIN loo l \gset economics_

\echo '== Deterministic UTC-day cluster bootstrap (normal and preregistered stresses) =='
WITH selected AS (
  SELECT e.*,(e.config #>> '{raw,strategy,min_net_edge_per_share}')::numeric AS min_edge_per_share,
         (e.config #>> '{raw,strategy,min_net_edge_usd}')::numeric AS min_edge_usd
  FROM polymarket.btc_paper_experiments e JOIN polymarket.trading_processes p USING(process_id)
  WHERE e.experiment_id=NULLIF(:'experiment_id','')::uuid
), trades AS (
  SELECT o.order_id,m.window_end,m.window_end::date AS day,d.net_edge_per_share,d.size,
    CASE WHEN d.metadata #>> '{fair_value,probability_uncertainty}'
      ~ '^[+]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$'
      THEN (d.metadata #>> '{fair_value,probability_uncertainty}')::numeric END AS uncertainty,
    s.min_edge_per_share,s.min_edge_usd,sum(f.price*f.size) AS entry_notional,
    sum(f.fee) AS fees,
    sum(CASE WHEN f.token_id=m.official_winning_token_id THEN f.size ELSE 0 END) AS payout,
    sum(CASE WHEN f.token_id=m.official_winning_token_id THEN f.size-f.price*f.size-f.fee
      ELSE -f.price*f.size-f.fee END) AS net_pnl
  FROM selected s JOIN polymarket.orders o ON o.process_id=s.process_id
   AND o.raw_payload #>> '{request,metadata,experiment_id}'=s.experiment_id::text
   AND o.state='filled' AND o.side='buy' AND o.order_type='fok'
  JOIN polymarket.fills f ON f.order_id=o.order_id AND f.source='paper'
  JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=s.experiment_id
   AND d.decision_id::text=o.raw_payload #>> '{request,metadata,decision_id}'
  JOIN polymarket.btc_interval_markets m ON m.market_id=o.market_id
  JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE m.official_outcome IN('up','down')
    AND m.official_winning_token_id IN(m.up_token_id,m.down_token_id)
    AND m.official_resolved_at>=m.window_end
    AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
    AND m.official_resolution_received_at>=m.window_end
    AND jsonb_typeof(m.official_resolution_payload)='object'
    AND rw.status IN('resolved','resolved_late')
    AND rw.resolution_source=m.official_resolution_source
    AND rw.resolution_received_at=m.official_resolution_received_at
  GROUP BY o.order_id,m.window_end,d.net_edge_per_share,d.size,d.metadata,
    s.min_edge_per_share,s.min_edge_usd
), ranked AS (
  SELECT t.*,row_number() OVER(ORDER BY net_pnl DESC,window_end,order_id) AS profit_rank,
    count(*) OVER() AS total_trades
  FROM trades t
), stressed AS (
  SELECT r.*,
    CASE WHEN profit_rank<=ceil(total_trades*:'missed_best_fill_fraction'::numeric)
      THEN 0::numeric ELSE payout-entry_notional-fees*:'fee_stress_multiplier'::numeric END
        AS fee_missed_net_pnl,
    CASE WHEN NOT coalesce(uncertainty BETWEEN 0 AND 1
          AND min_edge_per_share>=0 AND min_edge_usd>=0 AND size>0
          AND net_edge_per_share::text NOT IN('NaN','Infinity','-Infinity'),false)
      THEN net_pnl
    WHEN net_edge_per_share-(:'probability_uncertainty_multiplier'::numeric-1)*uncertainty
          >=min_edge_per_share
      AND (net_edge_per_share-(:'probability_uncertainty_multiplier'::numeric-1)*uncertainty)*size
          >=min_edge_usd THEN net_pnl ELSE 0::numeric END AS uncertainty_net_pnl
  FROM ranked r
), daily AS (
  SELECT day,count(*)::bigint AS fills,sum(net_pnl) AS normal_pnl,
    sum(fee_missed_net_pnl) AS fee_missed_pnl,
    sum(uncertainty_net_pnl) AS uncertainty_pnl
  FROM stressed GROUP BY day
), numbered_days AS (
  SELECT d.*,row_number() OVER(ORDER BY day) AS day_number FROM daily d
), meta AS (
  SELECT count(*)::bigint AS day_count FROM numbered_days
), draws AS (
  SELECT replicate,slot,
    1+mod(mod(hashtextextended(
      concat_ws(':',:'experiment_id',replicate::text,slot::text),
      :'bootstrap_seed'::bigint
    ),nullif(m.day_count,0))+m.day_count,nullif(m.day_count,0)) AS selected_day_number
  FROM meta m
  CROSS JOIN generate_series(1,:'bootstrap_replicates'::integer) replicate
  CROSS JOIN LATERAL generate_series(1,m.day_count::integer) slot
  WHERE m.day_count>0
), replicate_stats AS (
  SELECT b.replicate,
    sum(d.normal_pnl)/nullif(sum(d.fills),0) AS normal_expectancy,
    sum(d.fee_missed_pnl)/nullif(sum(d.fills),0) AS fee_missed_expectancy,
    sum(d.uncertainty_pnl)/nullif(sum(d.fills),0) AS uncertainty_expectancy
  FROM draws b JOIN numbered_days d ON d.day_number=b.selected_day_number
  GROUP BY b.replicate
), intervals AS (
  SELECT count(*) AS completed_replicates,
    coalesce(percentile_cont(:'bootstrap_lower_quantile'::double precision)
      WITHIN GROUP(ORDER BY normal_expectancy),-999999) AS normal_lower,
    percentile_cont(:'bootstrap_upper_quantile'::double precision)
      WITHIN GROUP(ORDER BY normal_expectancy) AS normal_upper,
    coalesce(percentile_cont(:'bootstrap_lower_quantile'::double precision)
      WITHIN GROUP(ORDER BY fee_missed_expectancy),-999999) AS fee_stress_lower,
    percentile_cont(:'bootstrap_upper_quantile'::double precision)
      WITHIN GROUP(ORDER BY fee_missed_expectancy) AS fee_stress_upper,
    coalesce(percentile_cont(:'bootstrap_lower_quantile'::double precision)
      WITHIN GROUP(ORDER BY uncertainty_expectancy),-999999) AS uncertainty_stress_lower,
    percentile_cont(:'bootstrap_upper_quantile'::double precision)
      WITHIN GROUP(ORDER BY uncertainty_expectancy) AS uncertainty_stress_upper,
    avg((normal_expectancy>0)::integer)::numeric AS normal_probability_positive,
    avg((fee_missed_expectancy>0)::integer)::numeric AS fee_stress_probability_positive,
    avg((uncertainty_expectancy>0)::integer)::numeric AS uncertainty_stress_probability_positive
  FROM replicate_stats
)
SELECT m.day_count,i.*,
  (m.day_count>=:'minimum_distinct_utc_days'::bigint
    AND i.completed_replicates=:'bootstrap_replicates'::integer
    AND i.normal_lower>0 AND i.fee_stress_lower>0 AND i.uncertainty_stress_lower>0)
      AS gate_pass
FROM meta m CROSS JOIN intervals i;

WITH selected AS (
  SELECT e.*,(e.config #>> '{raw,strategy,min_net_edge_per_share}')::numeric AS min_edge_per_share,
    (e.config #>> '{raw,strategy,min_net_edge_usd}')::numeric AS min_edge_usd
  FROM polymarket.btc_paper_experiments e JOIN polymarket.trading_processes p USING(process_id)
  WHERE e.experiment_id=NULLIF(:'experiment_id','')::uuid
), trades AS (
  SELECT o.order_id,m.window_end,m.window_end::date AS day,d.net_edge_per_share,d.size,
    CASE WHEN d.metadata #>> '{fair_value,probability_uncertainty}'
      ~ '^[+]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$'
      THEN (d.metadata #>> '{fair_value,probability_uncertainty}')::numeric END AS uncertainty,
    s.min_edge_per_share,s.min_edge_usd,sum(f.price*f.size) AS entry_notional,sum(f.fee) AS fees,
    sum(CASE WHEN f.token_id=m.official_winning_token_id THEN f.size ELSE 0 END) AS payout,
    sum(CASE WHEN f.token_id=m.official_winning_token_id THEN f.size-f.price*f.size-f.fee
      ELSE -f.price*f.size-f.fee END) AS net_pnl
  FROM selected s JOIN polymarket.orders o ON o.process_id=s.process_id
   AND o.raw_payload #>> '{request,metadata,experiment_id}'=s.experiment_id::text
   AND o.state='filled' AND o.side='buy' AND o.order_type='fok'
  JOIN polymarket.fills f ON f.order_id=o.order_id AND f.source='paper'
  JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=s.experiment_id
   AND d.decision_id::text=o.raw_payload #>> '{request,metadata,decision_id}'
  JOIN polymarket.btc_interval_markets m ON m.market_id=o.market_id
  JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE m.official_outcome IN('up','down') AND m.official_winning_token_id IN(m.up_token_id,m.down_token_id)
    AND m.official_resolved_at>=m.window_end
    AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
    AND m.official_resolution_received_at>=m.window_end
    AND jsonb_typeof(m.official_resolution_payload)='object'
    AND rw.status IN('resolved','resolved_late')
    AND rw.resolution_source=m.official_resolution_source
    AND rw.resolution_received_at=m.official_resolution_received_at
  GROUP BY o.order_id,m.window_end,d.net_edge_per_share,d.size,d.metadata,
    s.min_edge_per_share,s.min_edge_usd
), ranked AS (
  SELECT t.*,row_number() OVER(ORDER BY net_pnl DESC,window_end,order_id) AS rank,
    count(*) OVER() AS total FROM trades t
), daily AS (
  SELECT day,count(*)::bigint AS fills,sum(net_pnl) AS normal,
    sum(CASE WHEN rank<=ceil(total*:'missed_best_fill_fraction'::numeric) THEN 0::numeric
      ELSE payout-entry_notional-fees*:'fee_stress_multiplier'::numeric END) AS fee_stress,
    sum(CASE WHEN NOT coalesce(uncertainty BETWEEN 0 AND 1
        AND min_edge_per_share>=0 AND min_edge_usd>=0 AND size>0
        AND net_edge_per_share::text NOT IN('NaN','Infinity','-Infinity'),false)
      THEN net_pnl
      WHEN net_edge_per_share-(:'probability_uncertainty_multiplier'::numeric-1)*uncertainty
          >=min_edge_per_share
      AND (net_edge_per_share-(:'probability_uncertainty_multiplier'::numeric-1)*uncertainty)*size
          >=min_edge_usd THEN net_pnl ELSE 0::numeric END) AS uncertainty_stress
  FROM ranked GROUP BY day
), numbered AS (
  SELECT d.*,row_number() OVER(ORDER BY day) AS day_number FROM daily d
), meta AS (SELECT count(*)::bigint AS days FROM numbered), draws AS (
  SELECT rep,slot,1+mod(mod(hashtextextended(
    concat_ws(':',:'experiment_id',rep::text,slot::text),:'bootstrap_seed'::bigint
  ),nullif(m.days,0))+m.days,nullif(m.days,0)) AS chosen
  FROM meta m CROSS JOIN generate_series(1,:'bootstrap_replicates'::integer) rep
  CROSS JOIN LATERAL generate_series(1,m.days::integer) slot WHERE m.days>0
), reps AS (
  SELECT rep,sum(normal)/nullif(sum(fills),0) AS normal,
    sum(fee_stress)/nullif(sum(fills),0) AS fee_stress,
    sum(uncertainty_stress)/nullif(sum(fills),0) AS uncertainty_stress
  FROM draws JOIN numbered ON day_number=chosen GROUP BY rep
), result AS (
  SELECT count(*) AS replicates,
    coalesce(percentile_cont(:'bootstrap_lower_quantile'::double precision)
      WITHIN GROUP(ORDER BY normal),-999999) AS normal_lower,
    coalesce(percentile_cont(:'bootstrap_lower_quantile'::double precision)
      WITHIN GROUP(ORDER BY fee_stress),-999999) AS fee_stress_lower,
    coalesce(percentile_cont(:'bootstrap_lower_quantile'::double precision)
      WITHIN GROUP(ORDER BY uncertainty_stress),-999999) AS uncertainty_stress_lower
  FROM reps
)
SELECT r.replicates,r.normal_lower,r.fee_stress_lower,r.uncertainty_stress_lower,
  (m.days>=:'minimum_distinct_utc_days'::bigint
   AND r.replicates=:'bootstrap_replicates'::integer
   AND r.normal_lower>0 AND r.fee_stress_lower>0 AND r.uncertainty_stress_lower>0) AS gate_pass
FROM result r CROSS JOIN meta m \gset bootstrap_

\echo '== Official-outcome forecast calibration and residuals =='
WITH selected AS (
  SELECT * FROM polymarket.btc_paper_experiments
  WHERE experiment_id=NULLIF(:'experiment_id','')::uuid
), forecasts AS (
  SELECT s.snapshot_id,s.market_id,s.feature_as_of,s.seconds_to_close,
    least(greatest(s.fair_up_probability,0.000000000001),0.999999999999) AS strategy_p,
    CASE WHEN s.up_best_bid IS NOT NULL AND s.up_best_ask IS NOT NULL
      THEN least(greatest((s.up_best_bid+s.up_best_ask)/2,0.000000000001),0.999999999999)
      ELSE NULL END AS market_p,
    CASE m.official_outcome WHEN 'up' THEN 1::numeric WHEN 'down' THEN 0::numeric END AS y
  FROM selected e JOIN polymarket.btc_feature_snapshots s
    ON s.features->>'process_id'=e.process_id::text
   AND s.feature_as_of>=e.started_at
   AND s.feature_as_of<=least(coalesce(e.stopped_at,now()),now())
  JOIN polymarket.btc_interval_markets m ON m.market_id=s.market_id
  JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE s.fair_up_probability IS NOT NULL
    AND m.official_outcome IN('up','down')
    AND m.official_winning_token_id IN(m.up_token_id,m.down_token_id)
    AND m.official_resolved_at>=m.window_end
    AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
    AND m.official_resolution_received_at>=m.window_end
    AND jsonb_typeof(m.official_resolution_payload)='object'
    AND rw.status IN('resolved','resolved_late')
    AND rw.resolution_source=m.official_resolution_source
    AND rw.resolution_received_at=m.official_resolution_received_at
)
SELECT 'deterministic_strategy' AS forecast_source,count(*) AS forecast_rows,
  count(DISTINCT market_id) AS resolved_markets,
  avg(power(strategy_p-y,2)) AS brier_score,
  avg(-(y*ln(strategy_p)+(1-y)*ln(1-strategy_p))) AS log_loss,
  avg(strategy_p-y) AS mean_probability_residual,
  avg(strategy_p) AS mean_forecast_probability,avg(y) AS observed_up_rate
FROM forecasts
UNION ALL
SELECT 'arrival_book_midpoint',count(market_p),count(DISTINCT market_id) FILTER(WHERE market_p IS NOT NULL),
  avg(power(market_p-y,2)) FILTER(WHERE market_p IS NOT NULL),
  avg(-(y*ln(market_p)+(1-y)*ln(1-market_p))) FILTER(WHERE market_p IS NOT NULL),
  avg(market_p-y) FILTER(WHERE market_p IS NOT NULL),avg(market_p),
  avg(y) FILTER(WHERE market_p IS NOT NULL)
FROM forecasts;

WITH selected AS (
  SELECT * FROM polymarket.btc_paper_experiments
  WHERE experiment_id=NULLIF(:'experiment_id','')::uuid
), forecasts AS (
  SELECT s.market_id,least(greatest(s.fair_up_probability,0::numeric),1::numeric) AS p,
    CASE m.official_outcome WHEN 'up' THEN 1::numeric WHEN 'down' THEN 0::numeric END AS y
  FROM selected e JOIN polymarket.btc_feature_snapshots s
    ON s.features->>'process_id'=e.process_id::text
   AND s.feature_as_of>=e.started_at
   AND s.feature_as_of<=least(coalesce(e.stopped_at,now()),now())
  JOIN polymarket.btc_interval_markets m ON m.market_id=s.market_id
  JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE s.fair_up_probability IS NOT NULL AND m.official_outcome IN('up','down')
    AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
    AND rw.status IN('resolved','resolved_late')
), bucketed AS (
  SELECT *,least(10,floor(p*10)::integer+1) AS probability_decile FROM forecasts
)
SELECT probability_decile,count(*) AS forecast_rows,count(DISTINCT market_id) AS resolved_markets,
  min(p) AS minimum_probability,max(p) AS maximum_probability,
  avg(p) AS mean_forecast_probability,avg(y) AS observed_up_rate,
  avg(p-y) AS mean_probability_residual
FROM bucketed GROUP BY probability_decile ORDER BY probability_decile;

WITH selected AS (
  SELECT * FROM polymarket.btc_paper_experiments
  WHERE experiment_id=NULLIF(:'experiment_id','')::uuid
), forecasts AS (
  SELECT s.market_id,s.seconds_to_close,
    least(greatest(s.fair_up_probability,0.000000000001),0.999999999999) AS p,
    CASE m.official_outcome WHEN 'up' THEN 1::numeric WHEN 'down' THEN 0::numeric END AS y
  FROM selected e JOIN polymarket.btc_feature_snapshots s
    ON s.features->>'process_id'=e.process_id::text
   AND s.feature_as_of>=e.started_at
   AND s.feature_as_of<=least(coalesce(e.stopped_at,now()),now())
  JOIN polymarket.btc_interval_markets m ON m.market_id=s.market_id
  JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE s.fair_up_probability IS NOT NULL AND m.official_outcome IN('up','down')
    AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
    AND rw.status IN('resolved','resolved_late')
), bucketed AS (
  SELECT *,CASE WHEN seconds_to_close<15 THEN '000-014s'
    WHEN seconds_to_close<30 THEN '015-029s'
    WHEN seconds_to_close<60 THEN '030-059s'
    WHEN seconds_to_close<120 THEN '060-119s'
    ELSE '120s+' END AS horizon_bucket
  FROM forecasts
)
SELECT horizon_bucket,count(*) AS forecast_rows,count(DISTINCT market_id) AS resolved_markets,
  avg(power(p-y,2)) AS brier_score,
  avg(-(y*ln(p)+(1-y)*ln(1-p))) AS log_loss,
  avg(p-y) AS mean_probability_residual
FROM bucketed GROUP BY horizon_bucket ORDER BY horizon_bucket;

\echo '== Filled-order regime slices and PnL concentration =='
WITH selected AS (
  SELECT * FROM polymarket.btc_paper_experiments
  WHERE experiment_id=NULLIF(:'experiment_id','')::uuid
), trades AS (
  SELECT o.order_id,m.window_end,d.outcome,fs.seconds_to_close,fs.realized_vol_30s_bps,
    CASE WHEN d.outcome='up' THEN (fs.up_best_ask-fs.up_best_bid)*10000
      WHEN d.outcome='down' THEN (fs.down_best_ask-fs.down_best_bid)*10000 END AS spread_bps,
    (o.raw_payload #>> '{request,metadata,paper_execution,observed_submit_to_arrival_ms}')::numeric
      AS latency_ms,
    sum(CASE WHEN f.token_id=m.official_winning_token_id THEN f.size-f.price*f.size-f.fee
      ELSE -f.price*f.size-f.fee END) AS net_pnl,
    bool_or(f.token_id=m.official_winning_token_id) AS won
  FROM selected e JOIN polymarket.orders o ON o.process_id=e.process_id
   AND o.raw_payload #>> '{request,metadata,experiment_id}'=e.experiment_id::text
   AND o.state='filled' AND o.side='buy' AND o.order_type='fok'
  JOIN polymarket.fills f ON f.order_id=o.order_id AND f.source='paper'
  JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=e.experiment_id
   AND d.decision_id::text=o.raw_payload #>> '{request,metadata,decision_id}'
  JOIN polymarket.btc_feature_snapshots fs ON fs.snapshot_id=d.snapshot_id
  JOIN polymarket.btc_interval_markets m ON m.market_id=o.market_id
  JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE m.official_outcome IN('up','down') AND m.official_winning_token_id IN(m.up_token_id,m.down_token_id)
    AND m.official_resolved_at>=m.window_end
    AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
    AND m.official_resolution_received_at>=m.window_end
    AND jsonb_typeof(m.official_resolution_payload)='object'
    AND rw.status IN('resolved','resolved_late')
    AND rw.resolution_source=m.official_resolution_source
    AND rw.resolution_received_at=m.official_resolution_received_at
  GROUP BY o.order_id,m.window_end,d.outcome,fs.seconds_to_close,fs.realized_vol_30s_bps,
    fs.up_best_ask,fs.up_best_bid,fs.down_best_ask,fs.down_best_bid,o.raw_payload
), ranked AS (
  SELECT t.*,ntile(4) OVER(ORDER BY realized_vol_30s_bps NULLS LAST,window_end,order_id)
    AS volatility_quartile FROM trades t
), dimensions AS (
  SELECT 'seconds_to_close_fixed_buckets'::text AS dimension,
    CASE WHEN seconds_to_close<15 THEN '000-014s' WHEN seconds_to_close<30 THEN '015-029s'
      WHEN seconds_to_close<60 THEN '030-059s' WHEN seconds_to_close<120 THEN '060-119s'
      ELSE '120s+' END AS bucket,net_pnl,won FROM ranked
  UNION ALL SELECT 'realized_volatility_cohort_quartiles',
    CASE WHEN realized_vol_30s_bps IS NULL THEN 'unknown'
      ELSE 'q'||volatility_quartile::text END,net_pnl,won FROM ranked
  UNION ALL SELECT 'quoted_spread_fixed_buckets',
    CASE WHEN spread_bps IS NULL THEN 'unknown' WHEN spread_bps<25 THEN '000-024bps'
      WHEN spread_bps<50 THEN '025-049bps' WHEN spread_bps<100 THEN '050-099bps'
      ELSE '100bps+' END,net_pnl,won FROM ranked
  UNION ALL SELECT 'observed_paper_arrival_latency_fixed_buckets',
    CASE WHEN latency_ms IS NULL THEN 'unknown' WHEN latency_ms<=175 THEN '000-175ms'
      WHEN latency_ms<=300 THEN '176-300ms' WHEN latency_ms<=600 THEN '301-600ms'
      ELSE '601ms+' END,net_pnl,won FROM ranked
), buckets AS (
  SELECT dimension,bucket,count(*) AS fills,count(*) FILTER(WHERE won) AS winners,
    sum(net_pnl) AS net_pnl,avg(net_pnl) AS mean_net_pnl,min(net_pnl) AS worst_trade
  FROM dimensions GROUP BY dimension,bucket
)
SELECT dimension,bucket,fills,winners,
  round(100.0*winners/nullif(fills,0),4) AS win_rate_pct,net_pnl,mean_net_pnl,worst_trade,
  fills>=20 AS interpretable_slice,
  greatest(net_pnl,0)/nullif(sum(greatest(net_pnl,0)) OVER(PARTITION BY dimension),0)
    AS positive_pnl_share_within_dimension
FROM buckets ORDER BY dimension,bucket;

WITH selected AS (
  SELECT * FROM polymarket.btc_paper_experiments
  WHERE experiment_id=NULLIF(:'experiment_id','')::uuid
), trades AS (
  SELECT o.order_id,m.window_end,d.outcome,fs.seconds_to_close,fs.realized_vol_30s_bps,
    CASE WHEN d.outcome='up' THEN (fs.up_best_ask-fs.up_best_bid)*10000
      WHEN d.outcome='down' THEN (fs.down_best_ask-fs.down_best_bid)*10000 END AS spread_bps,
    (o.raw_payload #>> '{request,metadata,paper_execution,observed_submit_to_arrival_ms}')::numeric AS latency_ms,
    sum(CASE WHEN f.token_id=m.official_winning_token_id THEN f.size-f.price*f.size-f.fee
      ELSE -f.price*f.size-f.fee END) AS net_pnl
  FROM selected e JOIN polymarket.orders o ON o.process_id=e.process_id
   AND o.raw_payload #>> '{request,metadata,experiment_id}'=e.experiment_id::text
   AND o.state='filled' AND o.side='buy' AND o.order_type='fok'
  JOIN polymarket.fills f ON f.order_id=o.order_id AND f.source='paper'
  JOIN polymarket.btc_strategy_decisions d ON d.experiment_id=e.experiment_id
   AND d.decision_id::text=o.raw_payload #>> '{request,metadata,decision_id}'
  JOIN polymarket.btc_feature_snapshots fs ON fs.snapshot_id=d.snapshot_id
  JOIN polymarket.btc_interval_markets m ON m.market_id=o.market_id
  JOIN polymarket.btc_official_resolution_watches rw ON rw.market_id=m.market_id
  WHERE m.official_outcome IN('up','down') AND m.official_winning_token_id IN(m.up_token_id,m.down_token_id)
    AND m.official_resolution_source IN('clob_websocket','clob_rest_reconciliation')
    AND rw.status IN('resolved','resolved_late')
  GROUP BY o.order_id,m.window_end,d.outcome,fs.seconds_to_close,fs.realized_vol_30s_bps,
    fs.up_best_ask,fs.up_best_bid,fs.down_best_ask,fs.down_best_bid,o.raw_payload
), ranked AS (
  SELECT t.*,ntile(4) OVER(ORDER BY realized_vol_30s_bps NULLS LAST,window_end,order_id) AS vol_q
  FROM trades t
), dimensions AS (
  SELECT 'seconds_to_close'::text AS dimension,
    CASE WHEN seconds_to_close<15 THEN 'a' WHEN seconds_to_close<30 THEN 'b'
      WHEN seconds_to_close<60 THEN 'c' WHEN seconds_to_close<120 THEN 'd' ELSE 'e' END AS bucket,net_pnl FROM ranked
  UNION ALL SELECT 'realized_volatility',coalesce('q'||vol_q::text,'unknown'),net_pnl FROM ranked
  UNION ALL SELECT 'quoted_spread',CASE WHEN spread_bps IS NULL THEN 'unknown' WHEN spread_bps<25 THEN 'a'
    WHEN spread_bps<50 THEN 'b' WHEN spread_bps<100 THEN 'c' ELSE 'd' END,net_pnl FROM ranked
  UNION ALL SELECT 'arrival_latency',CASE WHEN latency_ms IS NULL THEN 'unknown' WHEN latency_ms<=175 THEN 'a'
    WHEN latency_ms<=300 THEN 'b' WHEN latency_ms<=600 THEN 'c' ELSE 'd' END,net_pnl FROM ranked
), buckets AS (
  SELECT dimension,bucket,count(*) AS fills,sum(net_pnl) AS pnl FROM dimensions GROUP BY dimension,bucket
), concentration AS (
  SELECT dimension,count(*) AS buckets,
    coalesce(max(greatest(pnl,0))/nullif(sum(greatest(pnl,0)),0),1) AS maximum_positive_bucket_share
  FROM buckets GROUP BY dimension
)
SELECT count(*) AS dimensions_reported,
  coalesce(max(maximum_positive_bucket_share),1) AS maximum_bucket_share,
  (count(*)=4 AND bool_and(maximum_positive_bucket_share<=:'maximum_concentration_fraction'::numeric))
    AS gate_pass
FROM concentration \gset regime_

\echo '== Preregistered realtime-paper experiment classification =='
SELECT :'identity_experiment_id'::uuid AS experiment_id,
  :'identity_experiment_name' AS experiment_name,
  :'identity_experiment_status' AS experiment_status,
  :'identity_identity_gate_pass'::boolean AS identity_gate_pass,
  :'operational_expected_windows'::bigint AS expected_windows,
  :'operational_complete_windows'::bigint AS complete_windows,
  :'operational_coverage_pct'::numeric AS coverage_pct,
  :'operational_official_slo_pct'::numeric AS official_resolution_slo_pct,
  :'integrity_gate_pass'::boolean AS safety_execution_accounting_gate_pass,
  :'preview_telemetry_gate_pass'::boolean AS online_preview_telemetry_gate_pass,
  :'economics_fills'::bigint AS qualifying_filled_orders,
  :'economics_days'::bigint AS distinct_utc_days,
  :'economics_mean_net'::numeric AS normal_mean_net_pnl_per_filled_order,
  :'economics_fee_stress_mean'::numeric AS fee_missed_stress_mean_per_original_filled_order,
  :'economics_uncertainty_stress_mean'::numeric AS uncertainty_stress_mean_per_original_filled_order,
  :'preview_minimum_scenario_mean'::numeric AS minimum_online_preview_mean_per_primary_eligible_order,
  :'bootstrap_normal_lower'::numeric AS normal_day_block_lower_95_bound,
  :'bootstrap_fee_stress_lower'::numeric AS fee_missed_day_block_lower_95_bound,
  :'bootstrap_uncertainty_stress_lower'::numeric AS uncertainty_day_block_lower_95_bound,
  :'regime_maximum_bucket_share'::numeric AS maximum_positive_regime_bucket_share,
  CASE
    WHEN :'identity_experiment_status'='failed'
      OR NOT :'identity_identity_gate_pass'::boolean
      OR NOT :'integrity_gate_pass'::boolean
      OR NOT :'preview_telemetry_gate_pass'::boolean
      THEN 'OPERATIONAL_FAILURE'
    WHEN :'operational_expected_windows'::bigint>0
      AND (:'operational_coverage_pct'::numeric<:'minimum_coverage_pct'::numeric
        OR :'operational_official_slo_pct'::numeric<:'minimum_official_slo_pct'::numeric)
      THEN 'OPERATIONAL_FAILURE'
    WHEN :'identity_experiment_status'='running' THEN 'COLLECTING'
    WHEN :'operational_expected_windows'::bigint < :'required_complete_windows'::bigint
      AND (
        :'operational_expected_windows'::bigint=0
        OR (
          :'operational_coverage_pct'::numeric>=:'minimum_coverage_pct'::numeric
          AND :'operational_official_slo_pct'::numeric>=:'minimum_official_slo_pct'::numeric
        )
      ) THEN 'COLLECTING'
    WHEN :'operational_coverage_pct'::numeric<:'minimum_coverage_pct'::numeric
      OR :'operational_official_slo_pct'::numeric<:'minimum_official_slo_pct'::numeric
      THEN 'OPERATIONAL_FAILURE'
    WHEN :'operational_expected_windows'::bigint<:'required_complete_windows'::bigint
      OR :'economics_fills'::bigint<:'minimum_qualifying_fills'::bigint
      OR :'economics_days'::bigint<:'minimum_distinct_utc_days'::bigint
      THEN 'COLLECTING'
    WHEN :'economics_mean_net'::numeric<=0
      OR :'economics_fee_stress_mean'::numeric<=0
      OR :'economics_uncertainty_stress_mean'::numeric<=0
      OR NOT :'preview_economic_gate_pass'::boolean
      THEN 'NO_VIABLE_EDGE'
    WHEN NOT :'economics_gate_pass'::boolean
      OR NOT :'bootstrap_gate_pass'::boolean
      OR NOT :'regime_gate_pass'::boolean
      THEN 'PROMISING_INCONCLUSIVE'
    ELSE 'EDGE_SUPPORTED_FOR_LIVE_CANARY_PLANNING'
  END AS experiment_classification,
  CASE
    WHEN :'identity_experiment_status'='failed'
      OR NOT :'identity_identity_gate_pass'::boolean
      OR NOT :'integrity_gate_pass'::boolean
      OR NOT :'preview_telemetry_gate_pass'::boolean
      THEN 'Identity, safety, execution/accounting, or required non-mutating preview telemetry failed.'
    WHEN :'operational_expected_windows'::bigint>0
      AND (:'operational_coverage_pct'::numeric<:'minimum_coverage_pct'::numeric
        OR :'operational_official_slo_pct'::numeric<:'minimum_official_slo_pct'::numeric)
      THEN 'An eligible expected window has a persistent completeness or official-resolution SLO defect.'
    WHEN :'identity_experiment_status'='running'
      THEN 'The immutable cohort is still running; daily results are diagnostics, not a final conclusion.'
    WHEN :'operational_expected_windows'::bigint<:'required_complete_windows'::bigint
      THEN 'The cohort has not reached 2,000 eligible complete-window opportunities.'
    WHEN :'operational_coverage_pct'::numeric<:'minimum_coverage_pct'::numeric
      OR :'operational_official_slo_pct'::numeric<:'minimum_official_slo_pct'::numeric
      THEN 'Final expected-window or official-resolution SLO coverage failed.'
    WHEN :'economics_fills'::bigint<:'minimum_qualifying_fills'::bigint
      OR :'economics_days'::bigint<:'minimum_distinct_utc_days'::bigint
      THEN 'The economic cohort has not reached the preregistered fill/day sample.'
    WHEN :'economics_mean_net'::numeric<=0
      OR :'economics_fee_stress_mean'::numeric<=0
      OR :'economics_uncertainty_stress_mean'::numeric<=0
      OR NOT :'preview_economic_gate_pass'::boolean
      THEN 'At least one unstressed or preregistered stressed point expectancy is non-positive.'
    WHEN NOT :'economics_gate_pass'::boolean
      OR NOT :'bootstrap_gate_pass'::boolean
      OR NOT :'regime_gate_pass'::boolean
      THEN 'Point estimates are positive, but confidence, tail, drawdown, leave-one-day-out, or concentration evidence is incomplete.'
    ELSE 'All preregistered realtime-paper experiment gates pass; this supports planning a separately gated live canary and does not authorize live capital.'
  END AS classification_reason \gset final_

SELECT :'final_experiment_id'::uuid AS experiment_id,
  :'final_experiment_name' AS experiment_name,
  :'final_experiment_status' AS experiment_status,
  :'final_identity_gate_pass'::boolean AS identity_gate_pass,
  :'readiness_gate_pass'::boolean AS consolidated_readiness_gate_pass,
  :'capital_gate_pass'::boolean AS paper_capital_settlement_gate_pass,
  (:'uncertainty_input_violations'::bigint=0) AS uncertainty_input_gate_pass,
  :'final_expected_windows'::bigint AS expected_windows,
  :'final_qualifying_filled_orders'::bigint AS qualifying_filled_orders,
  :'final_experiment_classification' AS experiment_classification,
  :'final_classification_reason' AS classification_reason;
\echo EXPERIMENT_CLASSIFICATION= :final_experiment_classification

COMMIT;
