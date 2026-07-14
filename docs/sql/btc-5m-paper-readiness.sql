-- BTC 5-minute realtime paper experiment audit.
-- SELECT-only: safe to run against a live experiment with psql.

\pset pager off
\timing on

-- Pass -v experiment_id=<uuid> to pin the audit to a specific immutable cohort.
-- With no value, psql selects the newest experiment. Grace periods keep in-flight
-- writes and just-ended markets out of completeness denominators.
\if :{?experiment_id}
\else
SELECT coalesce((
  SELECT experiment_id::text
  FROM polymarket.btc_paper_experiments
  ORDER BY created_at DESC
  LIMIT 1
), '') AS experiment_id \gset
\endif
\if :{?coverage_grace_seconds}
\else
\set coverage_grace_seconds 5
\endif
\if :{?official_resolution_grace_seconds}
\else
\set official_resolution_grace_seconds 120
\endif
\if :{?resolution_watch_retention_seconds}
\else
\set resolution_watch_retention_seconds 3600
\endif
\if :{?boundary_max_delay_seconds}
\else
\set boundary_max_delay_seconds 5
\endif
\if :{?required_consecutive_intervals}
\else
\set required_consecutive_intervals 13
\endif
\if :{?minimum_snapshots_per_interval}
\else
\set minimum_snapshots_per_interval 285
\endif
\if :{?maximum_snapshot_gap_seconds}
\else
\set maximum_snapshot_gap_seconds 5
\endif

\echo 'audit experiment_id=' :'experiment_id'
\echo 'ML coverage grace seconds=' :'coverage_grace_seconds'
\echo 'official-resolution grace seconds=' :'official_resolution_grace_seconds'
\echo 'official-resolution watch retention seconds=' :'resolution_watch_retention_seconds'
\echo 'boundary maximum delay seconds=' :'boundary_max_delay_seconds'
\echo 'required consecutive complete intervals=' :'required_consecutive_intervals'
\echo 'minimum snapshots per complete interval=' :'minimum_snapshots_per_interval'
\echo 'maximum snapshot gap seconds=' :'maximum_snapshot_gap_seconds'

\echo '== migration and BTC tables =='
WITH expected_migrations(name) AS (
  VALUES
    ('AddBtcRealtimePaperAndMlShadow1777120000000'::text),
    ('AddBtcOfficialMarketResolution1777121000000'::text),
    ('AddBtcOfficialResolutionRecovery1777122000000'::text),
    ('AddBtcPhase6PaperCapital1777123000000'::text)
)
SELECT e.name, m.id, m.timestamp, (m.id IS NOT NULL) AS applied
FROM expected_migrations e
LEFT JOIN public.migrations m USING (name)
ORDER BY e.name;

WITH expected_columns(column_name, expected_data_type) AS (
  VALUES
    ('official_outcome'::text, 'text'::text),
    ('official_resolved_at', 'timestamp with time zone'),
    ('official_winning_token_id', 'text'),
    ('official_resolution_source', 'text'),
    ('official_resolution_received_at', 'timestamp with time zone'),
    ('official_resolution_payload', 'jsonb')
)
SELECT e.column_name, e.expected_data_type,
       c.data_type AS actual_data_type,
       (c.column_name IS NOT NULL) AS present,
       (c.data_type = e.expected_data_type) AS type_matches
FROM expected_columns e
LEFT JOIN information_schema.columns c
  ON c.table_schema = 'polymarket'
 AND c.table_name = 'btc_interval_markets'
 AND c.column_name = e.column_name
ORDER BY e.column_name;

SELECT count(*) AS btc_tables_present
FROM information_schema.tables
WHERE table_schema = 'polymarket'
  AND table_name IN (
    'btc_interval_markets', 'reference_price_ticks', 'market_feed_events',
    'orderbook_checkpoints', 'feed_sessions', 'btc_feature_snapshots',
    'btc_strategy_decisions', 'btc_paper_experiments', 'btc_market_labels',
    'ml_feature_vectors', 'ml_dataset_manifests', 'ml_model_versions',
    'ml_shadow_predictions', 'ml_evaluation_runs', 'ml_evaluation_metrics',
    'btc_official_resolution_watches', 'btc_paper_settlement_ledger'
  );

WITH expected_constraints(table_name, constraint_name) AS (
  VALUES
    ('btc_interval_markets'::text, 'chk_btc_official_resolution_all_or_none'::text),
    ('btc_interval_markets', 'chk_btc_official_resolution_winner'),
    ('btc_interval_markets', 'chk_btc_official_resolution_time'),
    ('btc_interval_markets', 'chk_btc_official_resolution_provenance'),
    ('btc_interval_markets', 'chk_btc_official_resolution_payload'),
    ('btc_interval_markets', 'uq_btc_interval_market_official_identity'),
    ('btc_official_resolution_watches', 'btc_official_resolution_watches_pkey'),
    ('btc_official_resolution_watches', 'btc_official_resolution_watches_market_id_fkey'),
    ('btc_official_resolution_watches', 'chk_btc_resolution_watch_status'),
    ('btc_official_resolution_watches', 'chk_btc_resolution_watch_deadline'),
    ('btc_official_resolution_watches', 'chk_btc_resolution_watch_subscription_count'),
    ('btc_official_resolution_watches', 'chk_btc_resolution_watch_source'),
    ('btc_official_resolution_watches', 'chk_btc_resolution_watch_state'),
    ('btc_paper_settlement_ledger', 'btc_paper_settlement_ledger_pkey'),
    ('btc_paper_settlement_ledger', 'btc_paper_settlement_ledger_experiment_id_fkey'),
    ('btc_paper_settlement_ledger', 'btc_paper_settlement_ledger_process_id_fkey'),
    ('btc_paper_settlement_ledger', 'btc_paper_settlement_ledger_order_id_fkey'),
    ('btc_paper_settlement_ledger', 'btc_paper_settlement_ledger_market_id_fkey'),
    ('btc_paper_settlement_ledger', 'uq_btc_paper_settlement_experiment_order'),
    ('btc_paper_settlement_ledger', 'fk_btc_paper_settlement_official_identity'),
    ('btc_paper_settlement_ledger', 'chk_btc_paper_settlement_outcome'),
    ('btc_paper_settlement_ledger', 'chk_btc_paper_settlement_fill_ids'),
    ('btc_paper_settlement_ledger', 'chk_btc_paper_settlement_source'),
    ('btc_paper_settlement_ledger', 'chk_btc_paper_settlement_amounts'),
    ('btc_paper_settlement_ledger', 'chk_btc_paper_settlement_credit_status'),
    ('btc_paper_settlement_ledger', 'chk_btc_paper_settlement_credit_state'),
    ('btc_paper_settlement_ledger', 'chk_btc_paper_settlement_evidence')
)
SELECT e.table_name, e.constraint_name, c.contype,
       pg_get_constraintdef(c.oid) AS definition,
       (c.oid IS NOT NULL) AS present
FROM expected_constraints e
LEFT JOIN pg_namespace n ON n.nspname = 'polymarket'
LEFT JOIN pg_class t ON t.relnamespace = n.oid AND t.relname = e.table_name
LEFT JOIN pg_constraint c
  ON c.conrelid = t.oid AND c.conname = e.constraint_name
ORDER BY e.table_name, e.constraint_name;

WITH expected_indexes(index_name) AS (
  VALUES
    ('idx_btc_resolution_watches_pending_deadline'::text),
    ('idx_btc_interval_markets_pending_official'),
    ('idx_btc_interval_markets_condition_id'),
    ('idx_btc_paper_settlement_pending'),
    ('idx_btc_paper_settlement_experiment_credited'),
    ('idx_btc_features_process_window_asof'),
    ('idx_btc_decisions_experiment_process_at'),
    ('idx_book_checkpoints_token_received_source')
)
SELECT e.index_name, i.indexdef, (i.indexname IS NOT NULL) AS present
FROM expected_indexes e
LEFT JOIN pg_indexes i
  ON i.schemaname = 'polymarket' AND i.indexname = e.index_name
ORDER BY e.index_name;

\echo '== Timescale compression and retention jobs =='
SELECT hypertable_schema, hypertable_name, proc_name, scheduled, config
FROM timescaledb_information.jobs
WHERE hypertable_schema = 'polymarket'
  AND hypertable_name IN (
    'market_feed_events', 'orderbook_checkpoints', 'reference_price_ticks',
    'btc_feature_snapshots', 'ml_feature_vectors', 'ml_shadow_predictions'
  )
ORDER BY hypertable_name, proc_name;

\echo '== BTC process and experiment state =='
SELECT process_id, name, process_type, process_scope, process_key, status, enabled,
       started_at, heartbeat_at, stopped_at, stop_reason,
       config #>> '{execution,mode}' AS execution_mode
FROM polymarket.trading_processes
WHERE process_type = 'btc_5m'
   OR process_key LIKE 'btc-5m-%'
ORDER BY started_at DESC;

SELECT experiment_id, name, status, process_id, strategy_version,
       feature_schema_version, config_hash, started_at, stopped_at, stop_reason,
       markets_observed, snapshots_recorded, decisions_recorded,
       trades_entered, trades_resolved, gross_pnl, fees_paid, net_pnl, updated_at
FROM polymarket.btc_paper_experiments
ORDER BY created_at DESC;

\echo '== selected immutable experiment/process cohort =='
SELECT experiment_id, name, status, process_id, strategy_version,
       feature_schema_version, config_hash, started_at, stopped_at, stop_reason,
       markets_observed, snapshots_recorded, decisions_recorded,
       trades_entered, trades_resolved, gross_pnl, fees_paid, net_pnl
FROM polymarket.btc_paper_experiments
WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid;

\echo '== selected cohort frozen v10 safety/config contract =='
WITH selected AS (
  SELECT e.experiment_id, e.process_id, e.config_hash AS experiment_config_hash,
         e.config AS experiment_config, p.config AS process_config,
         p.metadata AS process_metadata
  FROM polymarket.btc_paper_experiments e
  LEFT JOIN polymarket.trading_processes p ON p.process_id = e.process_id
  WHERE e.experiment_id = NULLIF(:'experiment_id', '')::uuid
)
SELECT experiment_id, process_id,
       process_config #>> '{raw,pipeline_version}' AS pipeline_version,
       (process_config #>> '{raw,runtime,official_resolution_audit_grace,secs}')::bigint
         AS official_resolution_audit_grace_seconds,
       (process_config #>> '{raw,runtime,official_resolution_watch_retention,secs}')::bigint
         AS official_resolution_watch_retention_seconds,
       (process_config #>> '{raw,paper,execution_enabled}')::boolean AS paper_execution_enabled,
       (experiment_config #>> '{execution_enabled}')::boolean
         AS experiment_execution_enabled,
       (process_config #>> '{raw,ml_shadow,execution_authority}')::boolean
         AS ml_execution_authority,
       process_config #>> '{execution,mode}' AS process_execution_mode,
       (process_config #>> '{execution,live_capital}')::boolean AS process_live_capital,
       experiment_config_hash,
       process_metadata->>'config_hash' AS process_config_hash,
       (
         process_config #>> '{raw,pipeline_version}' = 'btc_realtime_paper_pipeline_v10'
         AND (process_config #>> '{raw,runtime,official_resolution_audit_grace,secs}')::bigint
               = :'official_resolution_grace_seconds'::bigint
         AND (process_config #>> '{raw,runtime,official_resolution_watch_retention,secs}')::bigint
               = :'resolution_watch_retention_seconds'::bigint
         AND (process_config #>> '{raw,paper,execution_enabled}')::boolean
         AND (experiment_config #>> '{execution_enabled}')::boolean
         AND NOT (process_config #>> '{raw,ml_shadow,execution_authority}')::boolean
         AND process_config #>> '{execution,mode}' = 'paper'
         AND NOT (process_config #>> '{execution,live_capital}')::boolean
         AND experiment_config_hash = process_metadata->>'config_hash'
       ) AS frozen_v10_safety_config_gate_pass
FROM selected;

WITH selected_experiment AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
),
process_snapshots AS (
  SELECT s.snapshot_id, s.market_id
  FROM selected_experiment e
  JOIN polymarket.btc_feature_snapshots s
    ON (s.features->>'process_id') = e.process_id::text
   AND s.feature_as_of >= e.started_at
   AND s.feature_as_of <= coalesce(e.stopped_at, now())
  WHERE e.process_id IS NOT NULL
),
scoped_decisions AS (
  SELECT d.decision_id, d.snapshot_id, d.market_id, d.process_id, e.process_id AS expected_process_id
  FROM selected_experiment e
  JOIN polymarket.btc_strategy_decisions d ON d.experiment_id = e.experiment_id
)
SELECT (SELECT count(*) FROM scoped_decisions) AS scoped_decisions,
       (SELECT count(*) FROM scoped_decisions
        WHERE expected_process_id IS NOT NULL
          AND process_id IS DISTINCT FROM expected_process_id
       ) AS decisions_outside_experiment_process,
       (SELECT count(*) FROM scoped_decisions WHERE process_id IS NULL
       ) AS decisions_without_process,
       (SELECT count(*) FROM process_snapshots) AS process_owned_snapshots,
       (SELECT count(DISTINCT snapshot_id) FROM process_snapshots
       ) AS process_owned_snapshot_ids,
       (SELECT count(*) - count(DISTINCT snapshot_id) FROM process_snapshots
       ) AS excess_snapshot_rows_per_id,
       (SELECT count(*)
        FROM process_snapshots s
        LEFT JOIN scoped_decisions d USING (snapshot_id)
        WHERE d.decision_id IS NULL
       ) AS process_snapshots_without_decision,
       (SELECT count(*)
        FROM scoped_decisions d
        LEFT JOIN process_snapshots s USING (snapshot_id)
        WHERE s.snapshot_id IS NULL
       ) AS decisions_without_process_snapshot,
       (SELECT coalesce(sum(decisions - 1), 0)
        FROM (
          SELECT snapshot_id, count(*) AS decisions
          FROM scoped_decisions
          GROUP BY snapshot_id
          HAVING count(*) > 1
        ) duplicates
       ) AS excess_decisions_per_snapshot,
       (SELECT count(DISTINCT market_id) FROM process_snapshots) AS scoped_markets;

\echo '== selected cohort startup-partial interval (diagnostic only; excluded from streak gate) =='
WITH selected_experiment AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
),
startup_window AS (
  SELECT e.*,
         to_timestamp(
           floor(extract(epoch FROM e.started_at) / 300.0) * 300.0
         ) AS window_start
  FROM selected_experiment e
  WHERE e.started_at IS NOT NULL
),
startup_facts AS (
  SELECT w.*,
         w.window_start + interval '5 minutes' AS window_end,
         (w.started_at > w.window_start) AS is_startup_partial,
         (SELECT count(*)
          FROM polymarket.btc_interval_markets m
          WHERE w.started_at > w.window_start
            AND m.window_start = w.window_start
            AND m.window_end = w.window_start + interval '5 minutes'
            AND m.event_slug = 'btc-updown-5m-' ||
                floor(extract(epoch FROM w.window_start))::bigint
         ) AS exact_market_rows,
         (SELECT count(*)
          FROM polymarket.btc_feature_snapshots s
          WHERE w.started_at > w.window_start
            AND (s.features->>'process_id') = w.process_id::text
            AND s.window_start = w.window_start
            AND s.window_end = w.window_start + interval '5 minutes'
            AND s.feature_as_of >= w.started_at
            AND s.feature_as_of <= coalesce(w.stopped_at, now())
         ) AS process_snapshot_rows,
         (SELECT count(*)
          FROM polymarket.btc_strategy_decisions d
          WHERE w.started_at > w.window_start
            AND d.experiment_id = w.experiment_id
            AND d.process_id = w.process_id
            AND EXISTS (
              SELECT 1
              FROM polymarket.btc_feature_snapshots s
              WHERE s.snapshot_id = d.snapshot_id
                AND (s.features->>'process_id') = w.process_id::text
                AND s.window_start = w.window_start
                AND s.window_end = w.window_start + interval '5 minutes'
            )
         ) AS process_decision_rows,
         (SELECT count(*)
          FROM polymarket.btc_market_labels l
          WHERE w.started_at > w.window_start
            AND l.window_start = w.window_start
            AND l.window_end = w.window_start + interval '5 minutes'
         ) AS boundary_label_rows
  FROM startup_window w
)
SELECT window_start, window_end, started_at,
       is_startup_partial,
       CASE WHEN is_startup_partial THEN 1 ELSE 0 END AS startup_partial_intervals,
       exact_market_rows, process_snapshot_rows, process_decision_rows,
       boundary_label_rows,
       is_startup_partial AS excluded_from_consecutive_complete_gate
FROM startup_facts;

\echo '== selected cohort expected complete-window sequence and 13-interval gate =='
WITH selected_experiment AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
),
experiment_bounds AS (
  SELECT e.*,
         to_timestamp(
           ceil(extract(epoch FROM e.started_at) / 300.0) * 300.0
         ) AS first_complete_window_start,
         least(
           coalesce(e.stopped_at, now()),
           now() - make_interval(
             secs => :'coverage_grace_seconds'::double precision
           )
         ) AS complete_through
  FROM selected_experiment e
  WHERE e.started_at IS NOT NULL
),
expected_windows AS (
  SELECT b.experiment_id, b.process_id, b.started_at, b.stopped_at,
         g.window_start,
         g.window_start + interval '5 minutes' AS window_end,
         'btc-updown-5m-' ||
           floor(extract(epoch FROM g.window_start))::bigint AS expected_event_slug
  FROM experiment_bounds b
  CROSS JOIN LATERAL generate_series(
    b.first_complete_window_start,
    b.complete_through - interval '5 minutes',
    interval '5 minutes'
  ) AS g(window_start)
),
window_facts AS (
  SELECT w.*,
         (SELECT count(*)
          FROM polymarket.btc_interval_markets m
          WHERE m.window_start = w.window_start
            AND m.window_end = w.window_end
            AND m.event_slug = w.expected_event_slug
         ) AS exact_market_rows,
         (SELECT count(*)
          FROM polymarket.btc_interval_markets m
          WHERE m.window_start = w.window_start
            AND m.window_end = w.window_end
            AND m.event_slug = w.expected_event_slug
            AND m.validation_status = 'valid'
         ) AS valid_market_rows,
         (SELECT count(*)
          FROM polymarket.btc_feature_snapshots s
          WHERE (s.features->>'process_id') = w.process_id::text
            AND s.window_start = w.window_start
            AND s.window_end = w.window_end
            AND s.feature_as_of >= w.started_at
            AND s.feature_as_of <= coalesce(w.stopped_at, now())
         ) AS process_snapshot_rows,
         (SELECT min(s.feature_as_of)
          FROM polymarket.btc_feature_snapshots s
          WHERE (s.features->>'process_id') = w.process_id::text
            AND s.window_start = w.window_start
            AND s.window_end = w.window_end
            AND s.feature_as_of >= w.started_at
            AND s.feature_as_of <= coalesce(w.stopped_at, now())
         ) AS first_snapshot_at,
         (SELECT max(s.feature_as_of)
          FROM polymarket.btc_feature_snapshots s
          WHERE (s.features->>'process_id') = w.process_id::text
            AND s.window_start = w.window_start
            AND s.window_end = w.window_end
            AND s.feature_as_of >= w.started_at
            AND s.feature_as_of <= coalesce(w.stopped_at, now())
         ) AS last_snapshot_at,
         (SELECT coalesce(max(extract(epoch FROM (g.feature_as_of - g.previous_at))), 0)
          FROM (
            SELECT s.feature_as_of,
                   lag(s.feature_as_of) OVER (ORDER BY s.feature_as_of) AS previous_at
            FROM polymarket.btc_feature_snapshots s
            WHERE (s.features->>'process_id') = w.process_id::text
              AND s.window_start = w.window_start
              AND s.window_end = w.window_end
              AND s.feature_as_of >= w.started_at
              AND s.feature_as_of <= coalesce(w.stopped_at, now())
          ) g
         ) AS max_snapshot_gap_seconds,
         (SELECT count(*)
          FROM polymarket.btc_feature_snapshots s
          WHERE (s.features->>'process_id') = w.process_id::text
            AND s.window_start = w.window_start
            AND s.window_end = w.window_end
            AND s.feature_as_of >= w.started_at
            AND s.feature_as_of <= coalesce(w.stopped_at, now())
            AND s.readiness_status = 'ready'
         ) AS ready_snapshot_rows,
         (SELECT count(*)
          FROM polymarket.btc_feature_snapshots s
          WHERE (s.features->>'process_id') = w.process_id::text
            AND s.window_start = w.window_start
            AND s.window_end = w.window_end
            AND s.feature_as_of >= w.started_at
            AND s.feature_as_of <= coalesce(w.stopped_at, now())
            AND NOT EXISTS (
              SELECT 1
              FROM polymarket.btc_interval_markets m
              WHERE m.market_id = s.market_id
                AND m.window_start = w.window_start
                AND m.window_end = w.window_end
                AND m.event_slug = w.expected_event_slug
                AND m.validation_status = 'valid'
            )
         ) AS snapshot_market_mismatch_rows,
         (SELECT count(*)
          FROM polymarket.btc_strategy_decisions d
          WHERE d.experiment_id = w.experiment_id
            AND d.process_id = w.process_id
            AND EXISTS (
              SELECT 1
              FROM polymarket.btc_feature_snapshots s
              WHERE s.snapshot_id = d.snapshot_id
                AND (s.features->>'process_id') = w.process_id::text
                AND s.window_start = w.window_start
                AND s.window_end = w.window_end
                AND s.feature_as_of >= w.started_at
                AND s.feature_as_of <= coalesce(w.stopped_at, now())
            )
         ) AS process_decision_rows,
         (SELECT count(*)
          FROM polymarket.btc_market_labels l
          WHERE l.window_start = w.window_start
            AND l.window_end = w.window_end
         ) AS boundary_label_rows,
         (SELECT count(*)
          FROM polymarket.btc_market_labels l
          WHERE l.window_start = w.window_start
            AND l.window_end = w.window_end
            AND l.source_open_timestamp >= w.window_start
            AND l.source_open_timestamp <= w.window_start + make_interval(
              secs => :'boundary_max_delay_seconds'::double precision
            )
            AND l.source_close_timestamp >= w.window_end
            AND l.source_close_timestamp <= w.window_end + make_interval(
              secs => :'boundary_max_delay_seconds'::double precision
            )
            AND l.label_available_at >= l.source_close_timestamp
            AND EXISTS (
              SELECT 1
              FROM polymarket.btc_interval_markets m
              WHERE m.market_id = l.market_id
                AND m.window_start = w.window_start
                AND m.window_end = w.window_end
                AND m.event_slug = w.expected_event_slug
                AND m.validation_status = 'valid'
            )
         ) AS valid_boundary_label_rows
  FROM expected_windows w
),
classified AS (
  SELECT f.*,
         (
           exact_market_rows = 1
           AND valid_market_rows = 1
           AND process_snapshot_rows >= :'minimum_snapshots_per_interval'::bigint
           AND first_snapshot_at <= window_start + make_interval(
             secs => :'maximum_snapshot_gap_seconds'::double precision
           )
           AND last_snapshot_at >= window_end - make_interval(
             secs => :'maximum_snapshot_gap_seconds'::double precision
           )
           AND max_snapshot_gap_seconds <= :'maximum_snapshot_gap_seconds'::numeric
           AND ready_snapshot_rows > 0
           AND snapshot_market_mismatch_rows = 0
           AND process_decision_rows = process_snapshot_rows
           AND boundary_label_rows = 1
           AND valid_boundary_label_rows = 1
         ) AS complete_interval,
         concat_ws(',',
           CASE WHEN exact_market_rows <> 1 THEN 'exact_market_count' END,
           CASE WHEN valid_market_rows <> 1 THEN 'valid_market_count' END,
           CASE WHEN process_snapshot_rows < :'minimum_snapshots_per_interval'::bigint
             THEN 'insufficient_snapshot_coverage' END,
           CASE WHEN first_snapshot_at IS NULL
                  OR first_snapshot_at > window_start + make_interval(
                    secs => :'maximum_snapshot_gap_seconds'::double precision
                  ) THEN 'late_first_snapshot' END,
           CASE WHEN last_snapshot_at IS NULL
                  OR last_snapshot_at < window_end - make_interval(
                    secs => :'maximum_snapshot_gap_seconds'::double precision
                  ) THEN 'early_last_snapshot' END,
           CASE WHEN max_snapshot_gap_seconds > :'maximum_snapshot_gap_seconds'::numeric
             THEN 'snapshot_gap_exceeded' END,
           CASE WHEN ready_snapshot_rows = 0 THEN 'missing_ready_snapshot' END,
           CASE WHEN snapshot_market_mismatch_rows <> 0 THEN 'snapshot_market_mismatch' END,
           CASE WHEN process_decision_rows <> process_snapshot_rows THEN 'snapshot_decision_count_mismatch' END,
           CASE WHEN boundary_label_rows <> 1 THEN 'boundary_label_count' END,
           CASE WHEN valid_boundary_label_rows <> 1 THEN 'invalid_boundary_label' END
         ) AS completeness_issues
  FROM window_facts f
),
streak_groups AS (
  SELECT c.*,
         sum(CASE WHEN complete_interval THEN 0 ELSE 1 END)
           OVER (ORDER BY window_start) AS break_group
  FROM classified c
),
streaks AS (
  SELECT break_group, count(*) AS consecutive_complete_intervals
  FROM streak_groups
  WHERE complete_interval
  GROUP BY break_group
),
summary AS (
  SELECT count(*) AS expected_complete_intervals,
         count(*) FILTER (WHERE c.complete_interval) AS complete_intervals,
         count(*) FILTER (WHERE NOT c.complete_interval) AS incomplete_intervals,
         coalesce((SELECT max(consecutive_complete_intervals) FROM streaks), 0) AS longest_consecutive_complete_intervals,
         (SELECT count(*)
          FROM classified t
          WHERE t.complete_interval
            AND t.window_start > coalesce(
              (SELECT max(failed.window_start)
               FROM classified failed
               WHERE NOT failed.complete_interval),
              '-infinity'::timestamptz
            )
         ) AS latest_consecutive_complete_intervals,
         :'required_consecutive_intervals'::bigint AS required_consecutive_complete_intervals,
         count(*) >= :'required_consecutive_intervals'::bigint
           AND count(*) FILTER (WHERE NOT c.complete_interval) = 0
           AND (SELECT count(*)
                FROM classified t
                WHERE t.complete_interval
                  AND t.window_start > coalesce(
                    (SELECT max(failed.window_start)
                     FROM classified failed
                     WHERE NOT failed.complete_interval),
                    '-infinity'::timestamptz
                  )) >= :'required_consecutive_intervals'::bigint
           AS consecutive_complete_interval_gate_pass,
         min(c.window_start) AS first_expected_window_start,
         max(c.window_end) AS last_expected_window_end
  FROM classified c
),
report AS (
  SELECT 0 AS sort_group, 'summary'::text AS record_type,
         NULL::timestamptz AS window_start,
         NULL::timestamptz AS window_end,
         NULL::text AS expected_event_slug,
         NULL::bigint AS exact_market_rows,
         NULL::bigint AS valid_market_rows,
         NULL::bigint AS process_snapshot_rows,
         NULL::bigint AS ready_snapshot_rows,
         NULL::timestamptz AS first_snapshot_at,
         NULL::timestamptz AS last_snapshot_at,
         NULL::numeric AS max_snapshot_gap_seconds,
         NULL::bigint AS process_decision_rows,
         NULL::bigint AS boundary_label_rows,
         NULL::bigint AS valid_boundary_label_rows,
         NULL::boolean AS complete_interval,
         NULL::text AS completeness_issues,
         s.*
  FROM summary s
  UNION ALL
  SELECT 1, 'expected_window', c.window_start, c.window_end,
         c.expected_event_slug, c.exact_market_rows, c.valid_market_rows,
         c.process_snapshot_rows, c.ready_snapshot_rows,
         c.first_snapshot_at, c.last_snapshot_at, c.max_snapshot_gap_seconds,
         c.process_decision_rows, c.boundary_label_rows,
         c.valid_boundary_label_rows, c.complete_interval,
         c.completeness_issues, s.*
  FROM classified c
  CROSS JOIN summary s
)
SELECT record_type, window_start, window_end, expected_event_slug,
       exact_market_rows, valid_market_rows, process_snapshot_rows,
       ready_snapshot_rows, first_snapshot_at, last_snapshot_at,
       max_snapshot_gap_seconds, process_decision_rows, boundary_label_rows,
       valid_boundary_label_rows, complete_interval, completeness_issues,
       expected_complete_intervals, complete_intervals, incomplete_intervals,
       longest_consecutive_complete_intervals,
       latest_consecutive_complete_intervals,
       required_consecutive_complete_intervals,
       consecutive_complete_interval_gate_pass,
       first_expected_window_start, last_expected_window_end
FROM report
ORDER BY sort_group, window_start;

\echo '== exact-window market discovery =='
SELECT market_id, event_slug, window_start, window_end, validation_status,
       accepting_orders, active, closed, up_token_id, down_token_id,
       fee_rate, last_refreshed_at, validation_errors
FROM polymarket.btc_interval_markets
WHERE window_end >= now() - interval '15 minutes'
  AND window_start <= now() + interval '10 minutes'
ORDER BY window_start;

\echo '== reference feed freshness and integrity =='
SELECT source, symbol,
       count(*) FILTER (WHERE received_at >= now() - interval '10 minutes') AS ticks_10m,
       max(source_timestamp) AS latest_source_at,
       max(received_at) AS latest_received_at,
       round(extract(epoch FROM (now() - max(received_at)))::numeric, 3) AS receive_age_seconds,
       count(*) FILTER (
         WHERE received_at >= now() - interval '15 minutes'
           AND integrity_status NOT IN ('valid', 'ok', 'accepted')
       ) AS non_ok_15m
FROM polymarket.reference_price_ticks
GROUP BY source, symbol
ORDER BY source, symbol;

\echo '== current token book freshness =='
WITH current_tokens AS (
  SELECT market_id, 'up'::text AS outcome, up_token_id AS token_id
  FROM polymarket.btc_interval_markets
  WHERE window_start <= now() AND window_end > now()
  UNION ALL
  SELECT market_id, 'down'::text, down_token_id
  FROM polymarket.btc_interval_markets
  WHERE window_start <= now() AND window_end > now()
)
SELECT t.market_id, t.outcome, t.token_id, b.source_timestamp, b.received_at,
       round(extract(epoch FROM (now() - b.received_at))::numeric, 3) AS receive_age_seconds,
       b.best_bid, b.best_ask, b.spread, b.depth_bid, b.depth_ask,
       b.bootstrap_source, b.integrity_status
FROM current_tokens t
LEFT JOIN LATERAL (
  SELECT *
  FROM polymarket.orderbook_checkpoints ob
  WHERE ob.token_id = t.token_id
  ORDER BY ob.source_timestamp DESC, ob.received_at DESC
  LIMIT 1
) b ON true
ORDER BY t.market_id, t.outcome;

\echo '== recent feed-event integrity and session loss counters =='
WITH selected_bounds AS (
  SELECT started_at, coalesce(stopped_at, now()) AS ended_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
)
SELECT event_type, integrity_status, applied, count(*) AS events
FROM polymarket.market_feed_events f
CROSS JOIN selected_bounds b
WHERE f.received_at >= b.started_at AND f.received_at <= b.ended_at
GROUP BY event_type, integrity_status, applied
ORDER BY event_type, integrity_status, applied;

WITH selected_bounds AS (
  SELECT started_at, coalesce(stopped_at, now()) AS ended_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
)
SELECT f.feed_name, f.connection_id, f.reconnect_ordinal, f.started_at, f.connected_at,
       f.disconnected_at, f.messages_received, f.messages_persisted, f.decode_errors,
       f.integrity_gaps, f.dropped_messages, f.disconnect_reason
FROM polymarket.feed_sessions f
CROSS JOIN selected_bounds b
WHERE f.started_at <= b.ended_at
  AND coalesce(f.disconnected_at, b.ended_at) >= b.started_at
ORDER BY f.started_at DESC;

WITH selected_bounds AS (
  SELECT started_at, coalesce(stopped_at, now()) AS ended_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
), scoped_sessions AS (
  SELECT f.*
  FROM polymarket.feed_sessions f
  CROSS JOIN selected_bounds b
  WHERE f.started_at <= b.ended_at
    AND coalesce(f.disconnected_at, b.ended_at) >= b.started_at
)
SELECT count(*) AS cohort_feed_sessions,
       coalesce(sum(decode_errors), 0) AS cohort_decode_errors,
       coalesce(sum(integrity_gaps), 0) AS cohort_session_integrity_gaps,
       coalesce(sum(dropped_messages), 0) AS cohort_dropped_messages,
       count(*) FILTER (
         WHERE disconnect_reason IS NOT NULL
           AND disconnect_reason NOT IN ('shutdown','market_watch_changed')
       ) AS unexpected_disconnects
FROM scoped_sessions;

\echo '== boundary-label coverage, delay, and official agreement =='
WITH ended AS (
  SELECT market_id
  FROM polymarket.btc_interval_markets
  WHERE validation_status = 'valid'
    AND window_end >= now() - interval '24 hours'
    AND window_end < now() - interval '5 seconds'
)
SELECT count(*) AS ended_valid_markets_24h,
       count(l.market_id) AS labelled_markets_24h,
       count(*) - count(l.market_id) AS missing_labels_24h
FROM ended e
LEFT JOIN polymarket.btc_market_labels l USING (market_id);

SELECT l.market_id, l.window_start, l.window_end, l.outcome, l.label_version,
       l.source_open_timestamp, l.source_close_timestamp, l.label_available_at,
       round(extract(epoch FROM (l.source_open_timestamp - l.window_start))::numeric, 3) AS open_delay_seconds,
       round(extract(epoch FROM (l.source_close_timestamp - l.window_end))::numeric, 3) AS close_delay_seconds,
       m.official_outcome, m.official_resolved_at,
       m.official_winning_token_id,
       (m.official_outcome IS NOT NULL AND m.official_outcome <> l.outcome) AS official_disagreement
FROM polymarket.btc_market_labels l
LEFT JOIN polymarket.btc_interval_markets m USING (market_id)
WHERE l.window_end >= now() - interval '24 hours'
ORDER BY l.window_end DESC;

SELECT count(*) FILTER (WHERE l.source_open_timestamp < l.window_start) AS open_before_boundary,
       count(*) FILTER (WHERE l.source_close_timestamp < l.window_end) AS close_before_boundary,
       count(*) FILTER (WHERE l.label_available_at < l.source_close_timestamp) AS label_available_too_early,
       count(*) FILTER (
         WHERE m.official_outcome IS NOT NULL
           AND m.official_outcome <> l.outcome
       ) AS official_disagreements
FROM polymarket.btc_market_labels l
LEFT JOIN polymarket.btc_interval_markets m USING (market_id);

\echo '== selected cohort official-resolution coverage after grace =='
WITH selected_experiment AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
),
observed_markets AS (
  SELECT DISTINCT m.market_id, m.event_slug, m.window_start, m.window_end,
         m.official_outcome, m.official_resolved_at,
         m.official_winning_token_id, m.up_token_id, m.down_token_id,
         m.official_resolution_source, m.official_resolution_received_at,
         m.official_resolution_payload
  FROM selected_experiment e
  JOIN polymarket.btc_feature_snapshots s
    ON (s.features->>'process_id') = e.process_id::text
   AND s.feature_as_of >= e.started_at
   AND s.feature_as_of <= coalesce(e.stopped_at, now())
  JOIN polymarket.btc_interval_markets m ON m.market_id = s.market_id
  WHERE e.process_id IS NOT NULL
    AND m.window_start >= to_timestamp(
      ceil(extract(epoch FROM e.started_at) / 300.0) * 300.0
    )
    AND m.window_end <= to_timestamp(
      floor(extract(epoch FROM coalesce(e.stopped_at, now())) / 300.0) * 300.0
    )
    AND m.window_end <= now() - make_interval(
      secs => :'official_resolution_grace_seconds'::double precision
    )
)
SELECT count(*) AS ended_observed_markets,
       count(l.market_id) AS boundary_labels,
       count(o.official_outcome) AS official_outcomes,
       count(*) - count(l.market_id) AS missing_boundary_labels,
       count(*) - count(o.official_outcome) AS unverified_official_outcomes,
       count(*) FILTER (
         WHERE o.official_outcome IS NOT NULL AND o.official_outcome <> l.outcome
       ) AS official_disagreements,
       count(*) FILTER (
         WHERE o.official_outcome IS NOT NULL AND o.official_resolved_at IS NULL
       ) AS official_outcome_without_resolved_at,
       count(*) FILTER (
         WHERE o.official_resolved_at IS NOT NULL
           AND o.official_resolved_at < o.window_end
       ) AS official_resolution_before_window_end,
       count(*) FILTER (
         WHERE (o.official_outcome = 'up'
                  AND o.official_winning_token_id IS DISTINCT FROM o.up_token_id)
            OR (o.official_outcome = 'down'
                  AND o.official_winning_token_id IS DISTINCT FROM o.down_token_id)
       ) AS official_winning_token_mismatches,
       count(*) FILTER (
         WHERE o.official_outcome IS NOT NULL
           AND (
             o.official_resolution_source NOT IN ('clob_websocket','clob_rest_reconciliation')
             OR o.official_resolution_received_at IS NULL
             OR o.official_resolution_payload IS NULL
             OR jsonb_typeof(o.official_resolution_payload) <> 'object'
           )
       ) AS invalid_official_provenance,
       count(*) FILTER (
         WHERE o.official_resolution_received_at > o.window_end + make_interval(
           secs => :'official_resolution_grace_seconds'::double precision
         )
       ) AS official_received_after_slo,
       count(*) FILTER (WHERE w.market_id IS NULL) AS missing_resolution_watch,
       count(*) FILTER (WHERE w.status = 'pending') AS watch_pending_after_grace,
       count(*) FILTER (WHERE w.status IN ('expired','resolved_late')) AS failed_watch_status,
       count(*) FILTER (
         WHERE w.deadline_at IS DISTINCT FROM o.window_end + make_interval(
           secs => :'resolution_watch_retention_seconds'::double precision
         )
       ) AS watch_deadline_mismatches,
       count(*) FILTER (
         WHERE w.status IN ('resolved','resolved_late')
           AND (
             w.resolution_source IS DISTINCT FROM o.official_resolution_source
             OR w.resolution_received_at IS DISTINCT FROM o.official_resolution_received_at
           )
       ) AS watch_official_provenance_mismatches,
       coalesce(round((
         100.0 * count(o.official_outcome) / NULLIF(count(*), 0)
       )::numeric, 3), 0) AS official_coverage_pct,
       (
         count(*) > 0
         AND count(o.official_outcome) = count(*)
         AND count(l.market_id) = count(*)
         AND count(*) FILTER (
           WHERE o.official_resolved_at >= o.window_end
             AND o.official_outcome = l.outcome
             AND (
               (o.official_outcome = 'up' AND o.official_winning_token_id = o.up_token_id)
               OR
               (o.official_outcome = 'down' AND o.official_winning_token_id = o.down_token_id)
             )
             AND o.official_resolution_source IN ('clob_websocket','clob_rest_reconciliation')
             AND o.official_resolution_received_at >= o.window_end
             AND o.official_resolution_received_at <= o.window_end + make_interval(
               secs => :'official_resolution_grace_seconds'::double precision
             )
             AND jsonb_typeof(o.official_resolution_payload) = 'object'
             AND w.status = 'resolved'
             AND w.deadline_at = o.window_end + make_interval(
               secs => :'resolution_watch_retention_seconds'::double precision
             )
             AND w.resolution_source = o.official_resolution_source
             AND w.resolution_received_at = o.official_resolution_received_at
         ) = count(*)
       ) AS official_resolution_slo_gate_pass
FROM observed_markets o
LEFT JOIN polymarket.btc_market_labels l ON l.market_id = o.market_id
LEFT JOIN polymarket.btc_official_resolution_watches w ON w.market_id = o.market_id;

WITH selected_experiment AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
),
observed_markets AS (
  SELECT DISTINCT m.market_id, m.event_slug, m.window_start, m.window_end,
         m.official_outcome, m.official_resolved_at,
         m.official_winning_token_id, m.up_token_id, m.down_token_id,
         m.official_resolution_source, m.official_resolution_received_at,
         m.official_resolution_payload
  FROM selected_experiment e
  JOIN polymarket.btc_feature_snapshots s
    ON (s.features->>'process_id') = e.process_id::text
   AND s.feature_as_of >= e.started_at
   AND s.feature_as_of <= coalesce(e.stopped_at, now())
  JOIN polymarket.btc_interval_markets m ON m.market_id = s.market_id
  WHERE e.process_id IS NOT NULL
    AND m.window_start >= to_timestamp(
      ceil(extract(epoch FROM e.started_at) / 300.0) * 300.0
    )
    AND m.window_end <= to_timestamp(
      floor(extract(epoch FROM coalesce(e.stopped_at, now())) / 300.0) * 300.0
    )
    AND m.window_end <= now() - make_interval(
      secs => :'official_resolution_grace_seconds'::double precision
    )
)
SELECT o.market_id, o.event_slug, o.window_start, o.window_end,
       l.outcome AS boundary_outcome, o.official_outcome,
       o.official_resolved_at, o.official_winning_token_id,
       o.official_resolution_source, o.official_resolution_received_at,
       w.status AS watch_status, w.deadline_at, w.last_checked_at,
       w.last_subscribed_at, w.subscription_count,
       CASE
         WHEN l.market_id IS NULL THEN 'missing_boundary_label'
         WHEN o.official_outcome IS NULL THEN 'official_unverified'
         WHEN o.official_resolved_at IS NULL THEN 'official_timestamp_missing'
         WHEN o.official_resolved_at < o.window_end THEN 'official_timestamp_before_close'
         WHEN o.official_resolution_source NOT IN ('clob_websocket','clob_rest_reconciliation')
           THEN 'official_source_invalid'
         WHEN o.official_resolution_received_at IS NULL THEN 'official_received_at_missing'
         WHEN o.official_resolution_received_at < o.window_end
           THEN 'official_received_before_close'
         WHEN o.official_resolution_received_at > o.window_end + make_interval(
           secs => :'official_resolution_grace_seconds'::double precision
         ) THEN 'official_received_after_slo'
         WHEN jsonb_typeof(o.official_resolution_payload) <> 'object'
           THEN 'official_payload_invalid'
         WHEN o.official_outcome <> l.outcome THEN 'official_disagreement'
         WHEN o.official_winning_token_id IS NULL THEN 'official_winning_token_missing'
         WHEN (o.official_outcome = 'up'
                 AND o.official_winning_token_id IS DISTINCT FROM o.up_token_id)
           OR (o.official_outcome = 'down'
                 AND o.official_winning_token_id IS DISTINCT FROM o.down_token_id)
           THEN 'official_winning_token_mismatch'
         WHEN w.market_id IS NULL THEN 'resolution_watch_missing'
         WHEN w.status <> 'resolved' THEN 'resolution_watch_not_resolved_on_time'
         WHEN w.deadline_at IS DISTINCT FROM o.window_end + make_interval(
           secs => :'resolution_watch_retention_seconds'::double precision
         ) THEN 'resolution_watch_deadline_mismatch'
         WHEN w.resolution_source IS DISTINCT FROM o.official_resolution_source
           OR w.resolution_received_at IS DISTINCT FROM o.official_resolution_received_at
           THEN 'resolution_watch_provenance_mismatch'
       END AS integrity_issue
FROM observed_markets o
LEFT JOIN polymarket.btc_market_labels l ON l.market_id = o.market_id
LEFT JOIN polymarket.btc_official_resolution_watches w ON w.market_id = o.market_id
WHERE l.market_id IS NULL
   OR o.official_outcome IS NULL
   OR o.official_resolved_at IS NULL
   OR o.official_resolved_at < o.window_end
   OR o.official_resolution_source NOT IN ('clob_websocket','clob_rest_reconciliation')
   OR o.official_resolution_received_at IS NULL
   OR o.official_resolution_received_at < o.window_end
   OR o.official_resolution_received_at > o.window_end + make_interval(
     secs => :'official_resolution_grace_seconds'::double precision
   )
   OR jsonb_typeof(o.official_resolution_payload) <> 'object'
   OR o.official_outcome <> l.outcome
   OR o.official_winning_token_id IS NULL
   OR (o.official_outcome = 'up'
         AND o.official_winning_token_id IS DISTINCT FROM o.up_token_id)
   OR (o.official_outcome = 'down'
         AND o.official_winning_token_id IS DISTINCT FROM o.down_token_id)
   OR w.market_id IS NULL
   OR w.status <> 'resolved'
   OR w.deadline_at IS DISTINCT FROM o.window_end + make_interval(
     secs => :'resolution_watch_retention_seconds'::double precision
   )
   OR w.resolution_source IS DISTINCT FROM o.official_resolution_source
   OR w.resolution_received_at IS DISTINCT FROM o.official_resolution_received_at
ORDER BY o.window_end
LIMIT 200;

\echo '== deterministic snapshots and decisions =='
SELECT readiness_status, count(*) AS snapshots,
       min(feature_as_of) AS first_snapshot, max(feature_as_of) AS latest_snapshot,
       count(DISTINCT market_id) AS markets,
       count(DISTINCT feature_schema_version) AS schema_versions,
       count(DISTINCT feature_hash) AS feature_hashes
FROM polymarket.btc_feature_snapshots
WHERE feature_as_of >= now() - interval '24 hours'
GROUP BY readiness_status
ORDER BY readiness_status;

SELECT experiment_id, strategy_version, config_hash, action, status,
       coalesce(reject_reason, '<none>') AS reject_reason,
       count(*) AS decisions, count(DISTINCT market_id) AS markets,
       min(decision_at) AS first_decision, max(decision_at) AS latest_decision
FROM polymarket.btc_strategy_decisions
WHERE decision_at >= now() - interval '24 hours'
GROUP BY experiment_id, strategy_version, config_hash, action, status, reject_reason
ORDER BY experiment_id, action, status, reject_reason;

SELECT experiment_id, market_id, count(*) AS entered_decisions
FROM polymarket.btc_strategy_decisions
WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
  AND action = 'buy' AND status IN ('approved', 'submitted', 'filled')
GROUP BY experiment_id, market_id
HAVING count(*) > 1;

WITH selected_experiment AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
)
SELECT count(*) AS decisions_at_or_after_label,
       count(*) FILTER (
         WHERE d.action = 'buy' AND d.status IN ('approved', 'submitted', 'filled')
       ) AS entered_decisions_at_or_after_label
FROM selected_experiment e
JOIN polymarket.btc_strategy_decisions d
  ON d.experiment_id = e.experiment_id
 AND (e.process_id IS NULL OR d.process_id = e.process_id)
JOIN polymarket.btc_market_labels l USING (market_id)
WHERE d.decision_at >= l.label_available_at;

\echo '== selected cohort epoch and snapshot lineage integrity =='
WITH selected_experiment AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
),
observed_markets AS (
  SELECT DISTINCT m.*
  FROM selected_experiment e
  JOIN polymarket.btc_feature_snapshots s
    ON (s.features->>'process_id') = e.process_id::text
   AND s.feature_as_of >= e.started_at
   AND s.feature_as_of <= coalesce(e.stopped_at, now())
  JOIN polymarket.btc_interval_markets m ON m.market_id = s.market_id
  WHERE e.process_id IS NOT NULL
)
SELECT count(*) AS observed_markets,
       count(*) FILTER (
         WHERE event_slug <> 'btc-updown-5m-' || floor(extract(epoch FROM window_start))::bigint
       ) AS slug_epoch_mismatches,
       count(*) FILTER (
         WHERE mod(floor(extract(epoch FROM window_start))::bigint, 300) <> 0
       ) AS non_aligned_window_epochs,
       count(*) FILTER (
         WHERE window_end IS DISTINCT FROM window_start + interval '5 minutes'
       ) AS non_five_minute_windows
FROM observed_markets;

WITH selected_experiment AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
),
scoped_snapshots AS (
  SELECT DISTINCT s.*
  FROM selected_experiment e
  JOIN polymarket.btc_feature_snapshots s
    ON (s.features->>'process_id') = e.process_id::text
   AND s.feature_as_of >= e.started_at
   AND s.feature_as_of <= coalesce(e.stopped_at, now())
  WHERE e.process_id IS NOT NULL
)
SELECT count(*) AS scoped_snapshots,
       count(*) FILTER (
         WHERE (s.features->>'snapshot_id') IS DISTINCT FROM s.snapshot_id::text
       ) AS snapshot_id_json_mismatches,
       count(*) FILTER (
         WHERE (s.features->>'market_id') IS DISTINCT FROM s.market_id
       ) AS market_id_json_mismatches,
       count(*) FILTER (
         WHERE NULLIF(s.features->>'observed_at', '') IS NULL
            OR abs(extract(epoch FROM (
                 NULLIF(s.features->>'observed_at', '')::timestamptz -
                 s.feature_as_of
               ))) > 0.000001
       ) AS observed_at_json_mismatches,
       round(max(abs(extract(epoch FROM (
         NULLIF(s.features->>'observed_at', '')::timestamptz - s.feature_as_of
       ))) * 1000000)::numeric, 3) AS max_observed_at_delta_microseconds,
       count(*) FILTER (
         WHERE s.features->'lineage' IS DISTINCT FROM s.lineage
       ) AS embedded_lineage_mismatches,
       count(*) FILTER (
         WHERE s.readiness_status = 'ready'
           AND s.window_start >= (
             SELECT to_timestamp(
               ceil(extract(epoch FROM e.started_at) / 300.0) * 300.0
             )
             FROM selected_experiment e
           )
           AND (
             s.lineage->>'chainlink_open_tick_id' IS NULL
            OR s.lineage->>'chainlink_tick_id' IS NULL
            OR s.lineage->>'binance_tick_id' IS NULL
            OR s.lineage->>'up_book_checkpoint_id' IS NULL
            OR s.lineage->>'down_book_checkpoint_id' IS NULL
           )
       ) AS complete_ready_missing_core_lineage,
       count(*) FILTER (
         WHERE s.window_start < (
           SELECT to_timestamp(
             ceil(extract(epoch FROM e.started_at) / 300.0) * 300.0
           )
           FROM selected_experiment e
         )
           AND s.lineage->>'chainlink_open_tick_id' IS NULL
       ) AS startup_partial_missing_chainlink_open,
       count(*) FILTER (
         WHERE NULLIF(s.lineage->>'chainlink_open_source_timestamp', '')::timestamptz > s.feature_as_of
            OR NULLIF(s.lineage->>'chainlink_source_timestamp', '')::timestamptz > s.feature_as_of
            OR NULLIF(s.lineage->>'binance_source_timestamp', '')::timestamptz > s.feature_as_of
            OR NULLIF(s.lineage #>> '{chainlink_history,last_source_timestamp}', '')::timestamptz > s.feature_as_of
            OR NULLIF(s.lineage #>> '{binance_history,last_source_timestamp}', '')::timestamptz > s.feature_as_of
            OR NULLIF(s.features #>> '{up_book,source_timestamp}', '')::timestamptz > s.feature_as_of
            OR NULLIF(s.features #>> '{down_book,source_timestamp}', '')::timestamptz > s.feature_as_of
       ) AS future_source_event_snapshots,
       count(*) FILTER (
         WHERE NULLIF(s.lineage->>'chainlink_open_received_at', '')::timestamptz > s.received_at
            OR NULLIF(s.lineage->>'chainlink_received_at', '')::timestamptz > s.received_at
            OR NULLIF(s.lineage->>'binance_received_at', '')::timestamptz > s.received_at
            OR NULLIF(s.lineage #>> '{chainlink_history,max_received_at}', '')::timestamptz > s.received_at
            OR NULLIF(s.lineage #>> '{binance_history,max_received_at}', '')::timestamptz > s.received_at
            OR NULLIF(s.features #>> '{up_book,received_at}', '')::timestamptz > s.received_at
            OR NULLIF(s.features #>> '{down_book,received_at}', '')::timestamptz > s.received_at
       ) AS future_received_event_snapshots
FROM scoped_snapshots s;

WITH selected_experiment AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
),
scoped_snapshots AS (
  SELECT DISTINCT s.*
  FROM selected_experiment e
  JOIN polymarket.btc_feature_snapshots s
    ON (s.features->>'process_id') = e.process_id::text
   AND s.feature_as_of >= e.started_at
   AND s.feature_as_of <= coalesce(e.stopped_at, now())
  WHERE e.process_id IS NOT NULL
),
issues AS (
  SELECT s.snapshot_id, s.market_id, s.feature_as_of, s.window_start, s.window_end,
         m.event_slug,
         concat_ws(',',
           CASE WHEN m.market_id IS NULL THEN 'missing_market' END,
           CASE WHEN s.market_id IS DISTINCT FROM m.market_id THEN 'market_id_mismatch' END,
           CASE WHEN s.window_start IS DISTINCT FROM m.window_start THEN 'window_start_mismatch' END,
           CASE WHEN s.window_end IS DISTINCT FROM m.window_end THEN 'window_end_mismatch' END,
           CASE WHEN (s.features->>'event_slug') IS DISTINCT FROM m.event_slug THEN 'event_slug_mismatch' END,
           CASE WHEN (s.features->>'process_id') IS NULL THEN 'snapshot_process_missing' END,
           CASE WHEN s.features->'lineage' IS DISTINCT FROM s.lineage THEN 'embedded_lineage_mismatch' END
         ) AS integrity_issues
  FROM scoped_snapshots s
  LEFT JOIN polymarket.btc_interval_markets m ON m.market_id = s.market_id
)
SELECT *
FROM issues
WHERE integrity_issues <> ''
ORDER BY feature_as_of
LIMIT 200;

\echo '== paper order and fill parity =='
SELECT e.experiment_id, e.name,
       count(DISTINCT o.order_id) AS orders,
       count(DISTINCT f.fill_id) AS fills,
       coalesce(sum(f.size), 0) AS filled_shares,
       coalesce(sum(f.price * f.size), 0) AS filled_notional,
       coalesce(sum(f.fee), 0) AS fees,
       count(*) FILTER (WHERE f.source = 'live') AS live_fill_rows
FROM polymarket.btc_paper_experiments e
LEFT JOIN polymarket.orders o ON o.process_id = e.process_id
LEFT JOIN polymarket.fills f ON f.order_id = o.order_id
WHERE e.experiment_id = NULLIF(:'experiment_id', '')::uuid
GROUP BY e.experiment_id, e.name
ORDER BY e.name;

SELECT o.order_id, o.process_id, o.market_id, o.token_id, o.side, o.order_type,
       o.size AS order_size, coalesce(sum(f.size), 0) AS filled_size, o.state
FROM polymarket.orders o
JOIN polymarket.btc_paper_experiments e
  ON e.process_id = o.process_id
 AND e.experiment_id = NULLIF(:'experiment_id', '')::uuid
LEFT JOIN polymarket.fills f ON f.order_id = o.order_id
GROUP BY o.order_id, o.process_id, o.market_id, o.token_id, o.side, o.order_type, o.size, o.state
HAVING o.side <> 'buy'
    OR o.order_type <> 'fok'
    OR (coalesce(sum(f.size), 0) > 0 AND coalesce(sum(f.size), 0) <> o.size)
ORDER BY o.created_at DESC;

WITH selected_experiment AS (
  SELECT e.experiment_id, e.process_id, p.config AS process_config
  FROM polymarket.btc_paper_experiments e
  JOIN polymarket.trading_processes p ON p.process_id = e.process_id
  WHERE e.experiment_id = NULLIF(:'experiment_id', '')::uuid
), scoped_orders AS (
  SELECT o.*,
         count(f.fill_id) AS fill_rows,
         coalesce(sum(f.size), 0) AS filled_size,
         coalesce(sum(f.fee), 0) AS recorded_fee,
         coalesce(sum(round(
           f.size
           * (o.raw_payload #>> '{request,metadata,paper_execution,dynamic_fee_rate}')::numeric
           * f.price * (1::numeric - f.price),
           10
         )), 0) AS recomputed_fee,
         count(f.fill_id) FILTER (
           WHERE o.raw_payload #>> '{request,metadata,paper_execution,dynamic_fee_rate}' IS NULL
         ) AS fills_without_dynamic_fee_rate
  FROM selected_experiment e
  JOIN polymarket.orders o ON o.process_id = e.process_id
  LEFT JOIN polymarket.fills f ON f.order_id = o.order_id
  GROUP BY o.order_id
)
SELECT count(o.order_id) AS paper_order_plans,
       coalesce(sum(o.fill_rows), 0) AS paper_fill_rows,
       (count(o.order_id) > 0 AND coalesce(sum(o.fill_rows), 0) > 0)
         AS phase4_execution_non_vacuously_exercised,
       count(o.order_id) FILTER (
         WHERE o.state = 'filled' AND o.filled_size <> o.size
       ) AS filled_orders_without_exact_fok_size,
       count(o.order_id) FILTER (
         WHERE o.state <> 'filled' AND o.filled_size <> 0
       ) AS nonfilled_orders_with_fills,
       count(o.order_id) FILTER (
         WHERE o.side <> 'buy' OR o.order_type <> 'fok'
       ) AS non_buy_or_non_fok_orders,
       count(o.order_id) FILTER (
         WHERE d.decision_id IS NULL
       ) AS orders_without_linked_decision,
       count(o.order_id) FILTER (
         WHERE d.decision_id IS NOT NULL
           AND (
             d.order_plan_id IS NULL
             OR d.metadata #>> '{paper_order_plan,plan_id}' IS DISTINCT FROM d.order_plan_id::text
           )
       ) AS decision_order_plan_link_mismatches,
       count(o.order_id) FILTER (
         WHERE o.raw_payload #>> '{request,metadata,experiment_id}'
                 IS DISTINCT FROM e.experiment_id::text
            OR o.process_id IS DISTINCT FROM e.process_id
       ) AS order_experiment_process_mismatches,
       count(o.order_id) FILTER (
         WHERE o.state = 'filled'
           AND (
             o.raw_payload #>> '{request,metadata,paper_execution,arrival_checkpoint_id}' IS NULL
             OR o.raw_payload #>> '{request,metadata,paper_execution,arrival_at}' IS NULL
             OR o.raw_payload #>> '{request,metadata,paper_execution,arrival_checkpoint,received_at}' IS NULL
           )
       ) AS filled_orders_missing_arrival_causality,
       count(o.order_id) FILTER (
         WHERE o.state = 'filled'
           AND (
             (o.raw_payload #>> '{request,metadata,paper_execution,arrival_checkpoint,received_at}')::timestamptz
               > (o.raw_payload #>> '{request,metadata,paper_execution,arrival_at}')::timestamptz
             OR o.raw_payload #>> '{request,metadata,paper_execution,arrival_checkpoint_id}'
                  IS DISTINCT FROM
                o.raw_payload #>> '{request,metadata,paper_execution,arrival_checkpoint,checkpoint_id}'
           )
       ) AS arrival_checkpoint_causality_mismatches,
       count(o.order_id) FILTER (
         WHERE (o.raw_payload #>> '{request,metadata,paper_execution,configured_latency_ms}')::numeric
                 IS DISTINCT FROM (
                   (e.process_config #>> '{raw,paper,venue,arrival_latency,secs}')::numeric * 1000
                   + (e.process_config #>> '{raw,paper,venue,arrival_latency,nanos}')::numeric / 1000000
                 )
            OR (o.raw_payload #>> '{request,metadata,paper_execution,visible_depth_haircut}')::numeric
                 IS DISTINCT FROM
               (e.process_config #>> '{raw,paper,venue,visible_depth_haircut}')::numeric
       ) AS frozen_latency_or_haircut_mismatches,
       coalesce(sum(o.fills_without_dynamic_fee_rate), 0) AS fills_without_dynamic_fee_rate,
       count(o.order_id) FILTER (
         WHERE o.recorded_fee IS DISTINCT FROM o.recomputed_fee
       ) AS order_fee_recomputation_mismatches,
       coalesce(sum(o.recorded_fee), 0) AS recorded_dynamic_fees,
       coalesce(sum(o.recomputed_fee), 0) AS independently_recomputed_dynamic_fees
FROM selected_experiment e
LEFT JOIN scoped_orders o ON true
LEFT JOIN polymarket.btc_strategy_decisions d
  ON d.experiment_id = e.experiment_id
 AND d.decision_id::text = o.raw_payload #>> '{request,metadata,decision_id}'
GROUP BY e.experiment_id, e.process_id, e.process_config;

SELECT f.fill_id, f.order_id, f.process_id, f.source, f.timestamp_utc
FROM polymarket.fills f
LEFT JOIN polymarket.orders o ON o.order_id = f.order_id
WHERE o.order_id IS NULL
   OR (f.process_id = (
         SELECT process_id
         FROM polymarket.btc_paper_experiments
         WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
       )
       AND f.source <> 'paper')
ORDER BY f.timestamp_utc DESC;

\echo '== independently recomputed official-only resolved paper P&L =='
WITH cohort_fills AS (
  SELECT e.experiment_id, e.name, e.net_pnl AS recorded_net_pnl,
         f.fill_id, f.size, f.price, f.fee, f.token_id,
         m.window_end, m.up_token_id, m.down_token_id,
         m.official_outcome, m.official_resolved_at,
         coalesce((
           m.official_outcome IN ('up', 'down')
           AND m.official_resolved_at IS NOT NULL
           AND m.official_resolved_at >= m.window_end
           AND m.official_resolution_source IN ('clob_websocket','clob_rest_reconciliation')
           AND m.official_resolution_received_at >= m.window_end
           AND jsonb_typeof(m.official_resolution_payload) = 'object'
           AND w.status = 'resolved'
           AND w.resolution_source = m.official_resolution_source
           AND w.resolution_received_at = m.official_resolution_received_at
         ), false) AS officially_resolved
  FROM polymarket.btc_paper_experiments e
  JOIN polymarket.orders o ON o.process_id = e.process_id
  JOIN polymarket.fills f ON f.order_id = o.order_id AND f.source = 'paper'
  JOIN polymarket.btc_interval_markets m ON m.market_id = o.market_id
  LEFT JOIN polymarket.btc_official_resolution_watches w ON w.market_id = m.market_id
  WHERE e.experiment_id = NULLIF(:'experiment_id', '')::uuid
)
SELECT e.experiment_id, e.name,
       count(DISTINCT f.fill_id) AS paper_fills,
       count(DISTINCT f.fill_id) FILTER (
         WHERE f.officially_resolved
       ) AS officially_resolved_fills,
       count(DISTINCT f.fill_id) FILTER (
         WHERE NOT f.officially_resolved
       ) AS official_unverified_fills_excluded,
       coalesce(sum(
         CASE
           WHEN NOT f.officially_resolved THEN 0::numeric
           WHEN f.token_id = f.up_token_id AND f.official_outcome = 'up' THEN f.size
           WHEN f.token_id = f.down_token_id AND f.official_outcome = 'down' THEN f.size
           ELSE 0::numeric
         END
       ), 0) AS official_gross_payout,
       coalesce(sum(f.price * f.size) FILTER (
         WHERE f.officially_resolved
       ), 0) AS officially_resolved_entry_notional,
       coalesce(sum(f.fee) FILTER (
         WHERE f.officially_resolved
       ), 0) AS officially_resolved_fees,
       coalesce(sum(
         CASE
           WHEN NOT f.officially_resolved THEN 0::numeric
           WHEN f.token_id = f.up_token_id AND f.official_outcome = 'up'
             THEN f.size - f.price * f.size - f.fee
           WHEN f.token_id = f.down_token_id AND f.official_outcome = 'down'
             THEN f.size - f.price * f.size - f.fee
           ELSE 0::numeric - f.price * f.size - f.fee
         END
       ), 0) AS recomputed_official_net_pnl,
       e.net_pnl AS recorded_net_pnl,
       e.net_pnl - coalesce(sum(
         CASE
           WHEN NOT f.officially_resolved THEN 0::numeric
           WHEN f.token_id = f.up_token_id AND f.official_outcome = 'up'
             THEN f.size - f.price * f.size - f.fee
           WHEN f.token_id = f.down_token_id AND f.official_outcome = 'down'
             THEN f.size - f.price * f.size - f.fee
           ELSE 0::numeric - f.price * f.size - f.fee
         END
       ), 0) AS recorded_minus_recomputed_net_pnl
FROM polymarket.btc_paper_experiments e
LEFT JOIN cohort_fills f ON f.experiment_id = e.experiment_id
WHERE e.experiment_id = NULLIF(:'experiment_id', '')::uuid
GROUP BY e.experiment_id, e.name, e.net_pnl
ORDER BY name;

\echo '== ML-A/ML-B point-in-time shadow coverage =='
SELECT task, feature_schema_version, feature_schema_sha256,
       count(*) AS vectors, count(DISTINCT market_id) AS markets,
       min(feature_as_of) AS first_vector, max(feature_as_of) AS latest_vector
FROM polymarket.ml_feature_vectors
WHERE feature_as_of >= now() - interval '24 hours'
GROUP BY task, feature_schema_version, feature_schema_sha256
ORDER BY task, feature_schema_version;

SELECT task, model_version, status, artifact_sha256,
       count(*) AS predictions, count(DISTINCT market_id) AS markets,
       min(predicted_at) AS first_prediction, max(predicted_at) AS latest_prediction
FROM polymarket.ml_shadow_predictions
WHERE predicted_at >= now() - interval '24 hours'
GROUP BY task, model_version, status, artifact_sha256
ORDER BY task, model_version, status;

SELECT count(*) FILTER (
         WHERE p.predicted_at < s.feature_as_of
            OR p.predicted_at < v.feature_as_of
       ) AS shadow_before_snapshot,
       count(*) FILTER (WHERE p.feature_hash <> v.feature_hash) AS feature_hash_mismatches,
       count(*) FILTER (
         WHERE v.feature_as_of <> date_trunc('milliseconds', s.feature_as_of)
       ) AS vector_snapshot_time_mismatches
FROM polymarket.ml_shadow_predictions p
JOIN polymarket.btc_feature_snapshots s ON s.snapshot_id = p.snapshot_id
JOIN polymarket.ml_feature_vectors v
  ON v.snapshot_id = p.snapshot_id
 AND v.task = p.task
WHERE p.predicted_at >= now() - interval '24 hours'
  AND s.feature_as_of >= now() - interval '24 hours'
  AND v.feature_as_of >= now() - interval '24 hours';

SELECT v.task, v.feature_schema_version,
       count(*) AS vectors_without_prediction
FROM polymarket.ml_feature_vectors v
LEFT JOIN polymarket.ml_shadow_predictions p
  ON p.snapshot_id = v.snapshot_id AND p.task = v.task
WHERE v.feature_as_of >= now() - interval '24 hours'
  AND p.prediction_id IS NULL
GROUP BY v.task, v.feature_schema_version
ORDER BY v.task, v.feature_schema_version;

\echo '== selected cohort expected snapshot-to-task ML coverage after grace =='
WITH selected_experiment AS MATERIALIZED (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
),
cohort_snapshots AS MATERIALIZED (
  SELECT DISTINCT s.snapshot_id, s.market_id, s.feature_as_of,
         s.fair_up_probability
  FROM selected_experiment e
  JOIN polymarket.btc_feature_snapshots s
    ON (s.features->>'process_id') = e.process_id::text
   AND s.feature_as_of >= e.started_at
   AND s.feature_as_of <= coalesce(e.stopped_at, now())
  WHERE e.process_id IS NOT NULL
    AND s.feature_as_of <= now() - make_interval(
      secs => :'coverage_grace_seconds'::double precision
    )
),
task_specs(task, expected_schema, expected_schema_sha256,
           expected_model, expected_artifact_sha256, requires_fair_value) AS (
  VALUES
    ('fill_probability'::text, 'btc_5m_ml_b_fill_shadow_v2'::text,
     '40bb87bfa8ac2e65862bf95d2072a98c68568b94fbe5432b56f89021608b3ded'::text,
     'schema_canary_v2_fok_fill_probability'::text,
     '92492b87c0cfee7138859deb27397aaeb8f5cc10f909fa18ee6d1e7798d4617c'::text,
     false),
    ('toxicity', 'btc_5m_ml_b_toxicity_shadow_v2',
     'a9411cac6e821aa19d8e9599442e505a41b2a140653b08f2d4e2bda851286f1c',
     'schema_canary_v2_fok_post_fill_toxicity_probability',
     '02ad7ee0b191a3ad3932e58dc661b3d1e20315b1fd1d90c3c6a37b8eb2d3385c',
     false),
    ('fair_value_residual', 'btc_5m_ml_a_shadow_v2',
     '9fa7bd2fc404c0cd211cce37638cb008efd6cf4ec6a822d7294a4f974356185b',
     'schema_canary_v2_settlement_probability_residual',
     'd235ca4a8c4f3269a01e968c40150d3a78a9486cf6a1c47d0b9b0a2424b79949',
     true)
),
expected_tasks AS MATERIALIZED (
  SELECT s.snapshot_id, s.market_id, s.feature_as_of,
         t.task, t.expected_schema, t.expected_schema_sha256,
         t.expected_model, t.expected_artifact_sha256
  FROM cohort_snapshots s
  CROSS JOIN task_specs t
  WHERE NOT t.requires_fair_value OR s.fair_up_probability IS NOT NULL
),
key_rollup AS MATERIALIZED (
  SELECT e.snapshot_id, e.market_id, e.feature_as_of, e.task,
         v.vector_rows, v.contract_vector_rows,
         v.invalid_vector_identity_rows,
         p.prediction_rows, p.contract_prediction_rows
  FROM expected_tasks e
  CROSS JOIN LATERAL (
    SELECT count(*) AS vector_rows,
           count(*) FILTER (
             WHERE x.feature_schema_version = e.expected_schema
               AND x.feature_schema_sha256 = e.expected_schema_sha256
           ) AS contract_vector_rows,
           count(*) FILTER (
             WHERE x.feature_hash IS DISTINCT FROM x.vector->>'vector_sha256'
                OR x.snapshot_id::text IS DISTINCT FROM x.vector->>'snapshot_id'
                OR x.market_id IS DISTINCT FROM x.vector->>'market_id'
           ) AS invalid_vector_identity_rows
    FROM polymarket.ml_feature_vectors x
    WHERE x.snapshot_id = e.snapshot_id
      AND x.task = e.task
  ) v
  CROSS JOIN LATERAL (
    SELECT count(*) AS prediction_rows,
           count(*) FILTER (
             WHERE x.model_version = e.expected_model
               AND x.artifact_sha256 = e.expected_artifact_sha256
               AND EXISTS (
                 SELECT 1
                 FROM polymarket.ml_feature_vectors matching_vector
                 WHERE matching_vector.snapshot_id = e.snapshot_id
                   AND matching_vector.task = e.task
                   AND matching_vector.feature_hash = x.feature_hash
               )
           ) AS contract_prediction_rows
    FROM polymarket.ml_shadow_predictions x
    WHERE x.snapshot_id = e.snapshot_id
      AND x.task = e.task
  ) p
)
SELECT r.task,
       count(*) AS expected_snapshot_tasks,
       count(*) FILTER (WHERE r.vector_rows = 1) AS single_vector_keys,
       count(*) FILTER (WHERE r.prediction_rows = 1) AS single_prediction_keys,
       count(*) FILTER (WHERE r.vector_rows = 0) AS missing_vector_keys,
       count(*) FILTER (WHERE r.prediction_rows = 0) AS missing_prediction_keys,
       count(*) FILTER (WHERE r.vector_rows > 1) AS duplicate_vector_keys,
       coalesce(sum(greatest(r.vector_rows - 1, 0)), 0) AS excess_vector_rows,
       count(*) FILTER (WHERE r.prediction_rows > 1) AS duplicate_prediction_keys,
       coalesce(sum(greatest(r.prediction_rows - 1, 0)), 0) AS excess_prediction_rows,
       coalesce(sum(r.vector_rows - r.contract_vector_rows), 0)
         AS vector_contract_mismatch_rows,
       coalesce(sum(r.invalid_vector_identity_rows), 0) AS invalid_vector_identity_rows,
       coalesce(sum(r.prediction_rows - r.contract_prediction_rows), 0)
         AS prediction_contract_mismatch_rows,
       coalesce(round((
         100.0 * count(*) FILTER (WHERE r.vector_rows = 1) / NULLIF(count(*), 0)
       )::numeric, 3), 0) AS single_vector_coverage_pct,
       coalesce(round((
         100.0 * count(*) FILTER (WHERE r.prediction_rows = 1) / NULLIF(count(*), 0)
       )::numeric, 3), 0) AS single_prediction_coverage_pct
FROM key_rollup r
GROUP BY r.task
ORDER BY r.task;

\echo '== selected cohort missing, duplicate, or contract-mismatched ML keys =='
WITH selected_experiment AS MATERIALIZED (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
),
cohort_snapshots AS MATERIALIZED (
  SELECT DISTINCT s.snapshot_id, s.market_id, s.feature_as_of,
         s.fair_up_probability
  FROM selected_experiment e
  JOIN polymarket.btc_feature_snapshots s
    ON (s.features->>'process_id') = e.process_id::text
   AND s.feature_as_of >= e.started_at
   AND s.feature_as_of <= coalesce(e.stopped_at, now())
  WHERE e.process_id IS NOT NULL
    AND s.feature_as_of <= now() - make_interval(
      secs => :'coverage_grace_seconds'::double precision
    )
),
task_specs(task, expected_schema, expected_schema_sha256,
           expected_model, expected_artifact_sha256, requires_fair_value) AS (
  VALUES
    ('fill_probability'::text, 'btc_5m_ml_b_fill_shadow_v2'::text,
     '40bb87bfa8ac2e65862bf95d2072a98c68568b94fbe5432b56f89021608b3ded'::text,
     'schema_canary_v2_fok_fill_probability'::text,
     '92492b87c0cfee7138859deb27397aaeb8f5cc10f909fa18ee6d1e7798d4617c'::text,
     false),
    ('toxicity', 'btc_5m_ml_b_toxicity_shadow_v2',
     'a9411cac6e821aa19d8e9599442e505a41b2a140653b08f2d4e2bda851286f1c',
     'schema_canary_v2_fok_post_fill_toxicity_probability',
     '02ad7ee0b191a3ad3932e58dc661b3d1e20315b1fd1d90c3c6a37b8eb2d3385c',
     false),
    ('fair_value_residual', 'btc_5m_ml_a_shadow_v2',
     '9fa7bd2fc404c0cd211cce37638cb008efd6cf4ec6a822d7294a4f974356185b',
     'schema_canary_v2_settlement_probability_residual',
     'd235ca4a8c4f3269a01e968c40150d3a78a9486cf6a1c47d0b9b0a2424b79949',
     true)
),
expected_tasks AS MATERIALIZED (
  SELECT s.snapshot_id, s.market_id, s.feature_as_of,
         t.task, t.expected_schema, t.expected_schema_sha256,
         t.expected_model, t.expected_artifact_sha256
  FROM cohort_snapshots s
  CROSS JOIN task_specs t
  WHERE NOT t.requires_fair_value OR s.fair_up_probability IS NOT NULL
),
key_rollup AS MATERIALIZED (
  SELECT e.snapshot_id, e.market_id, e.feature_as_of, e.task,
         v.vector_rows, v.contract_vector_rows,
         v.invalid_vector_identity_rows,
         p.prediction_rows, p.contract_prediction_rows
  FROM expected_tasks e
  CROSS JOIN LATERAL (
    SELECT count(*) AS vector_rows,
           count(*) FILTER (
             WHERE x.feature_schema_version = e.expected_schema
               AND x.feature_schema_sha256 = e.expected_schema_sha256
           ) AS contract_vector_rows,
           count(*) FILTER (
             WHERE x.feature_hash IS DISTINCT FROM x.vector->>'vector_sha256'
                OR x.snapshot_id::text IS DISTINCT FROM x.vector->>'snapshot_id'
                OR x.market_id IS DISTINCT FROM x.vector->>'market_id'
           ) AS invalid_vector_identity_rows
    FROM polymarket.ml_feature_vectors x
    WHERE x.snapshot_id = e.snapshot_id
      AND x.task = e.task
  ) v
  CROSS JOIN LATERAL (
    SELECT count(*) AS prediction_rows,
           count(*) FILTER (
             WHERE x.model_version = e.expected_model
               AND x.artifact_sha256 = e.expected_artifact_sha256
               AND EXISTS (
                 SELECT 1
                 FROM polymarket.ml_feature_vectors matching_vector
                 WHERE matching_vector.snapshot_id = e.snapshot_id
                   AND matching_vector.task = e.task
                   AND matching_vector.feature_hash = x.feature_hash
               )
           ) AS contract_prediction_rows
    FROM polymarket.ml_shadow_predictions x
    WHERE x.snapshot_id = e.snapshot_id
      AND x.task = e.task
  ) p
)
SELECT snapshot_id, market_id, feature_as_of, task,
       vector_rows, prediction_rows, contract_vector_rows,
       invalid_vector_identity_rows, contract_prediction_rows,
       CASE
         WHEN vector_rows = 0 THEN 'missing_vector'
         WHEN vector_rows > 1 THEN 'duplicate_vector'
         WHEN contract_vector_rows <> 1 THEN 'vector_contract_mismatch'
         WHEN invalid_vector_identity_rows <> 0 THEN 'vector_identity_mismatch'
         WHEN prediction_rows = 0 THEN 'missing_prediction'
         WHEN prediction_rows > 1 THEN 'duplicate_prediction'
         WHEN contract_prediction_rows <> 1 THEN 'prediction_contract_mismatch'
       END AS coverage_issue
FROM key_rollup
WHERE vector_rows <> 1
   OR prediction_rows <> 1
   OR contract_vector_rows <> 1
   OR invalid_vector_identity_rows <> 0
   OR contract_prediction_rows <> 1
ORDER BY feature_as_of, task
LIMIT 200;

\echo '== selected cohort unexpected ML task keys =='
WITH selected_experiment AS MATERIALIZED (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
),
cohort_snapshots AS MATERIALIZED (
  SELECT DISTINCT s.snapshot_id, s.market_id, s.feature_as_of,
         s.fair_up_probability
  FROM selected_experiment e
  JOIN polymarket.btc_feature_snapshots s
    ON (s.features->>'process_id') = e.process_id::text
   AND s.feature_as_of >= e.started_at
   AND s.feature_as_of <= coalesce(e.stopped_at, now())
  WHERE e.process_id IS NOT NULL
    AND s.feature_as_of <= now() - make_interval(
      secs => :'coverage_grace_seconds'::double precision
    )
),
expected_tasks AS MATERIALIZED (
  SELECT snapshot_id, 'fill_probability'::text AS task FROM cohort_snapshots
  UNION ALL
  SELECT snapshot_id, 'toxicity' FROM cohort_snapshots
  UNION ALL
  SELECT snapshot_id, 'fair_value_residual'
  FROM cohort_snapshots
  WHERE fair_up_probability IS NOT NULL
),
actual_keys AS MATERIALIZED (
  SELECT 'vector'::text AS record_type, s.snapshot_id, s.market_id,
         s.feature_as_of, v.task, count(*) AS rows
  FROM cohort_snapshots s
  JOIN polymarket.ml_feature_vectors v ON v.snapshot_id = s.snapshot_id
  GROUP BY s.snapshot_id, s.market_id, s.feature_as_of, v.task
  UNION ALL
  SELECT 'prediction', s.snapshot_id, s.market_id,
         s.feature_as_of, p.task, count(*)
  FROM cohort_snapshots s
  JOIN polymarket.ml_shadow_predictions p ON p.snapshot_id = s.snapshot_id
  GROUP BY s.snapshot_id, s.market_id, s.feature_as_of, p.task
)
SELECT a.record_type, a.snapshot_id, a.market_id, a.feature_as_of, a.task, a.rows
FROM actual_keys a
LEFT JOIN expected_tasks e USING (snapshot_id, task)
WHERE e.snapshot_id IS NULL
ORDER BY a.feature_as_of, a.record_type, a.task
LIMIT 200;

\echo '== selected cohort ML vector point-in-time lineage =='
WITH selected_experiment AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
),
cohort_snapshots AS (
  SELECT DISTINCT s.snapshot_id, s.market_id, s.feature_as_of
  FROM selected_experiment e
  JOIN polymarket.btc_feature_snapshots s
    ON (s.features->>'process_id') = e.process_id::text
   AND s.feature_as_of >= e.started_at
   AND s.feature_as_of <= coalesce(e.stopped_at, now())
  WHERE e.process_id IS NOT NULL
),
scoped_vectors AS (
  SELECT v.*, s.feature_as_of AS snapshot_feature_as_of,
         s.market_id AS snapshot_market_id
  FROM cohort_snapshots s
  JOIN polymarket.ml_feature_vectors v ON v.snapshot_id = s.snapshot_id
)
SELECT count(*) AS vector_rows,
       count(*) FILTER (
         WHERE v.market_id IS DISTINCT FROM v.snapshot_market_id
            OR v.market_id IS DISTINCT FROM v.vector->>'market_id'
            OR v.snapshot_id::text IS DISTINCT FROM v.vector->>'snapshot_id'
       ) AS vector_identity_mismatches,
       count(*) FILTER (
         WHERE v.feature_as_of <> date_trunc('milliseconds', v.snapshot_feature_as_of)
            OR (v.vector->>'feature_as_of_ms')::bigint IS DISTINCT FROM
               floor(extract(epoch FROM v.snapshot_feature_as_of) * 1000)::bigint
       ) AS vector_snapshot_time_mismatches,
       count(*) FILTER (
         WHERE (v.vector->>'feature_received_at_ms')::bigint >
               (v.vector->>'feature_as_of_ms')::bigint
       ) AS receipt_cutoff_after_feature_cutoff,
       count(*) FILTER (
         WHERE v.feature_schema_version IS DISTINCT FROM v.vector->>'schema_version'
            OR v.feature_schema_sha256 IS DISTINCT FROM v.vector->>'schema_sha256'
            OR v.feature_hash IS DISTINCT FROM v.vector->>'vector_sha256'
       ) AS persisted_vector_contract_mismatches
FROM scoped_vectors v;

WITH selected_experiment AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
),
cohort_snapshots AS (
  SELECT DISTINCT s.snapshot_id, s.market_id, s.feature_as_of
  FROM selected_experiment e
  JOIN polymarket.btc_feature_snapshots s
    ON (s.features->>'process_id') = e.process_id::text
   AND s.feature_as_of >= e.started_at
   AND s.feature_as_of <= coalesce(e.stopped_at, now())
  WHERE e.process_id IS NOT NULL
),
scoped_predictions AS (
  SELECT p.*, s.feature_as_of AS snapshot_feature_as_of,
         s.market_id AS snapshot_market_id
  FROM cohort_snapshots s
  JOIN polymarket.ml_shadow_predictions p ON p.snapshot_id = s.snapshot_id
)
SELECT count(*) AS prediction_rows,
       count(*) FILTER (
         WHERE p.market_id IS DISTINCT FROM p.snapshot_market_id
       ) AS prediction_identity_mismatches,
       count(*) FILTER (
         WHERE p.predicted_at < p.snapshot_feature_as_of
       ) AS predictions_before_snapshot,
       count(*) FILTER (
         WHERE NOT EXISTS (
           SELECT 1
           FROM polymarket.ml_feature_vectors v
           WHERE v.snapshot_id = p.snapshot_id
             AND v.task = p.task
             AND v.feature_hash = p.feature_hash
         )
       ) AS predictions_without_exact_vector_hash,
       count(*) FILTER (
         WHERE (p.status = 'predicted' AND p.prediction IS NULL)
            OR (p.status <> 'predicted' AND p.reject_reason IS NULL)
       ) AS prediction_status_payload_mismatches
FROM scoped_predictions p;

WITH selected_experiment AS (
  SELECT experiment_id, process_id, started_at, stopped_at
  FROM polymarket.btc_paper_experiments
  WHERE experiment_id = NULLIF(:'experiment_id', '')::uuid
),
cohort_snapshots AS (
  SELECT DISTINCT s.snapshot_id
  FROM selected_experiment e
  JOIN polymarket.btc_feature_snapshots s
    ON (s.features->>'process_id') = e.process_id::text
   AND s.feature_as_of >= e.started_at
   AND s.feature_as_of <= coalesce(e.stopped_at, now())
  WHERE e.process_id IS NOT NULL
),
scoped_features AS (
  SELECT v.vector_id, v.feature_as_of, v.snapshot_id, v.market_id, v.task,
         v.vector, feature.value AS feature
  FROM cohort_snapshots s
  JOIN polymarket.ml_feature_vectors v ON v.snapshot_id = s.snapshot_id
  LEFT JOIN LATERAL jsonb_array_elements(
    coalesce(v.vector->'features', '[]'::jsonb)
  ) feature(value) ON true
)
SELECT count(DISTINCT (vector_id, feature_as_of)) AS vectors_with_feature_rows,
       count(*) FILTER (
         WHERE feature IS NULL
            OR feature->>'name' IS NULL
            OR feature->>'source_event_at_ms' IS NULL
            OR feature->>'source_received_at_ms' IS NULL
       ) AS missing_feature_lineage_rows,
       count(*) FILTER (
         WHERE (feature->>'source_event_at_ms')::bigint >
               (vector->>'feature_as_of_ms')::bigint
       ) AS future_source_feature_rows,
       count(DISTINCT (vector_id, feature_as_of)) FILTER (
         WHERE (feature->>'source_event_at_ms')::bigint >
               (vector->>'feature_as_of_ms')::bigint
       ) AS vectors_with_future_source_features,
       count(*) FILTER (
         WHERE (feature->>'source_received_at_ms')::bigint >
               (vector->>'feature_received_at_ms')::bigint
       ) AS future_received_feature_rows,
       count(DISTINCT (vector_id, feature_as_of)) FILTER (
         WHERE (feature->>'source_received_at_ms')::bigint >
               (vector->>'feature_received_at_ms')::bigint
       ) AS vectors_with_future_received_features
FROM scoped_features;

\echo '== selected cohort ML schema-canary authority, artifacts, and runtime counters =='
WITH expected(model_version, task, feature_schema_version, feature_schema_sha256, artifact_sha256) AS (
  VALUES
    ('schema_canary_v2_fok_fill_probability'::text, 'fill_probability'::text,
     'btc_5m_ml_b_fill_shadow_v2'::text,
     '40bb87bfa8ac2e65862bf95d2072a98c68568b94fbe5432b56f89021608b3ded'::text,
     '92492b87c0cfee7138859deb27397aaeb8f5cc10f909fa18ee6d1e7798d4617c'::text),
    ('schema_canary_v2_fok_post_fill_toxicity_probability', 'toxicity',
     'btc_5m_ml_b_toxicity_shadow_v2',
     'a9411cac6e821aa19d8e9599442e505a41b2a140653b08f2d4e2bda851286f1c',
     '02ad7ee0b191a3ad3932e58dc661b3d1e20315b1fd1d90c3c6a37b8eb2d3385c'),
    ('schema_canary_v2_settlement_probability_residual', 'fair_value_residual',
     'btc_5m_ml_a_shadow_v2',
     '9fa7bd2fc404c0cd211cce37638cb008efd6cf4ec6a822d7294a4f974356185b',
     'd235ca4a8c4f3269a01e968c40150d3a78a9486cf6a1c47d0b9b0a2424b79949')
)
SELECT count(*) AS expected_canary_models,
       count(m.model_version) AS present_canary_models,
       count(m.model_version) FILTER (
         WHERE m.state = 'shadow'
           AND m.feature_schema_version = e.feature_schema_version
           AND m.feature_schema_sha256 = e.feature_schema_sha256
           AND m.artifact_sha256 = e.artifact_sha256
           AND m.artifact->>'prior_kind' = 'provided_probability'
           AND m.artifact->>'trained_through_ms' IS NULL
           AND (m.artifact->>'intercept')::numeric = 0
           AND NOT EXISTS (
             SELECT 1
             FROM jsonb_array_elements(m.artifact->'features') feature
             WHERE (feature->>'coefficient')::numeric <> 0
           )
       ) AS valid_schema_canary_models,
       count(m.model_version) = count(*)
         AND count(m.model_version) FILTER (
           WHERE m.state = 'shadow'
             AND m.feature_schema_version = e.feature_schema_version
             AND m.feature_schema_sha256 = e.feature_schema_sha256
             AND m.artifact_sha256 = e.artifact_sha256
             AND m.artifact->>'prior_kind' = 'provided_probability'
             AND m.artifact->>'trained_through_ms' IS NULL
             AND (m.artifact->>'intercept')::numeric = 0
             AND NOT EXISTS (
               SELECT 1
               FROM jsonb_array_elements(m.artifact->'features') feature
               WHERE (feature->>'coefficient')::numeric <> 0
             )
         ) = count(*) AS canary_model_contract_gate_pass
FROM expected e
LEFT JOIN polymarket.ml_model_versions m
  ON m.model_version = e.model_version AND m.task = e.task;

WITH selected_experiment AS (
  SELECT e.experiment_id, e.process_id, e.status, e.summary,
         p.config AS process_config
  FROM polymarket.btc_paper_experiments e
  JOIN polymarket.trading_processes p ON p.process_id = e.process_id
  WHERE e.experiment_id = NULLIF(:'experiment_id', '')::uuid
), cohort_snapshots AS (
  SELECT s.snapshot_id
  FROM selected_experiment e
  JOIN polymarket.btc_feature_snapshots s
    ON (s.features->>'process_id') = e.process_id::text
   AND s.feature_as_of >= (
     SELECT started_at FROM polymarket.btc_paper_experiments
     WHERE experiment_id = e.experiment_id
   )
), scoped_predictions AS (
  SELECT p.*, v.vector, d.decision_at
  FROM cohort_snapshots s
  JOIN polymarket.ml_shadow_predictions p ON p.snapshot_id = s.snapshot_id
  JOIN polymarket.ml_feature_vectors v
    ON v.snapshot_id = p.snapshot_id
   AND v.task = p.task
   AND v.feature_hash = p.feature_hash
  LEFT JOIN polymarket.btc_strategy_decisions d ON d.snapshot_id = p.snapshot_id
)
SELECT (e.process_config #>> '{raw,ml_shadow,execution_authority}')::boolean
         AS ml_execution_authority,
       count(p.prediction_id) AS cohort_predictions,
       count(p.prediction_id) FILTER (
         WHERE p.metadata->>'schema_canary' IS DISTINCT FROM 'true'
            OR p.metadata->>'shadow_only' IS DISTINCT FROM 'true'
            OR p.metadata->>'influenced_order_plan' IS DISTINCT FROM 'false'
       ) AS canary_authority_metadata_mismatches,
       count(p.prediction_id) FILTER (
         WHERE p.status = 'predicted'
           AND p.prediction IS DISTINCT FROM p.prior
       ) AS canary_prediction_prior_mismatches,
       count(p.prediction_id) FILTER (
         WHERE p.prior IS DISTINCT FROM round(
           (p.vector->>'prior_probability')::numeric, 10
         )
       ) AS persisted_vector_prior_mismatches,
       count(p.prediction_id) FILTER (
         WHERE p.decision_at IS NULL OR p.predicted_at < p.decision_at
       ) AS predictions_before_deterministic_decision,
       (e.summary #>> '{ml_shadow_runtime,enqueued}')::bigint AS ml_work_enqueued,
       (e.summary #>> '{ml_shadow_runtime,completed}')::bigint AS ml_work_completed,
       (e.summary #>> '{ml_shadow_runtime,rejected_full}')::bigint AS ml_rejected_full,
       (e.summary #>> '{ml_shadow_runtime,rejected_closed}')::bigint AS ml_rejected_closed,
       (e.summary #>> '{ml_shadow_runtime,tasks_failed}')::bigint AS ml_tasks_failed,
       (e.summary #>> '{ml_shadow_runtime,queue_depth}')::bigint AS ml_queue_depth,
       (
         NOT (e.process_config #>> '{raw,ml_shadow,execution_authority}')::boolean
         AND count(p.prediction_id) > 0
         AND count(p.prediction_id) FILTER (
           WHERE p.metadata->>'schema_canary' IS DISTINCT FROM 'true'
              OR p.metadata->>'shadow_only' IS DISTINCT FROM 'true'
              OR p.metadata->>'influenced_order_plan' IS DISTINCT FROM 'false'
         ) = 0
         AND count(p.prediction_id) FILTER (
           WHERE p.status = 'predicted'
             AND p.prediction IS DISTINCT FROM p.prior
         ) = 0
         AND count(p.prediction_id) FILTER (
           WHERE p.prior IS DISTINCT FROM round(
             (p.vector->>'prior_probability')::numeric, 10
           )
         ) = 0
         AND count(p.prediction_id) FILTER (
           WHERE p.decision_at IS NULL OR p.predicted_at < p.decision_at
         ) = 0
         AND coalesce((e.summary #>> '{ml_shadow_runtime,rejected_full}')::bigint, 0) = 0
         AND coalesce((e.summary #>> '{ml_shadow_runtime,rejected_closed}')::bigint, 0) = 0
         AND coalesce((e.summary #>> '{ml_shadow_runtime,tasks_failed}')::bigint, 0) = 0
         AND (e.status = 'running'
              OR (
                coalesce((e.summary #>> '{ml_shadow_runtime,queue_depth}')::bigint, 0) = 0
                AND (e.summary #>> '{ml_shadow_runtime,completed}')::bigint
                      = (e.summary #>> '{ml_shadow_runtime,enqueued}')::bigint
              ))
       ) AS ml_schema_canary_runtime_gate_pass
FROM selected_experiment e
LEFT JOIN scoped_predictions p ON true
GROUP BY e.experiment_id, e.status, e.process_config, e.summary;
