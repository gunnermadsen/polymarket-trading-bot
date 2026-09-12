-- Current application schema baseline for fresh installations.
-- Generated from the accepted development schema; contains no operational data.
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
--
-- PostgreSQL database dump
--

-- Dumped from database version 14.11
-- Dumped by pg_dump version 14.11

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: ingester; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA ingester;


--
-- Name: market_data; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA market_data;


--
-- Name: polymarket; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA polymarket;


--
-- Name: notify_profile_change(); Type: FUNCTION; Schema: ingester; Owner: -
--

CREATE FUNCTION ingester.notify_profile_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      BEGIN
        IF TG_OP = 'INSERT' THEN
          PERFORM pg_notify('ingester_profile_changed', NEW.strategy_key);
        ELSIF (
          NEW.desired_state,
          NEW.desired_generation,
          NEW.config_schema_version,
          NEW.config
        ) IS DISTINCT FROM (
          OLD.desired_state,
          OLD.desired_generation,
          OLD.config_schema_version,
          OLD.config
        ) THEN
          PERFORM pg_notify('ingester_profile_changed', NEW.strategy_key);
        END IF;
        RETURN NEW;
      END;
      $$;


--
-- Name: reject_completed_backfill_artifact_change(); Type: FUNCTION; Schema: ingester; Owner: -
--

CREATE FUNCTION ingester.reject_completed_backfill_artifact_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      BEGIN
        IF TG_OP='DELETE' THEN
          IF OLD.status='completed' THEN
            RAISE EXCEPTION 'completed backfill artifact % is immutable', OLD.artifact_id
              USING ERRCODE='integrity_constraint_violation';
          END IF;
          RETURN OLD;
        END IF;
        IF OLD.status='completed' AND NEW IS DISTINCT FROM OLD THEN
          IF OLD.provider='pmxt_v2_capacity_execution_snapshots_v2'
            AND COALESCE((OLD.metadata->>'source_events_consumed')::bigint,0)=0
            AND NEW.record_count=0
            AND NEW.metadata->>'superseded_reason'='reconstructed_from_local_canonical_orderbooks'
            AND NEW.metadata->>'superseded_by_artifact_id' IS NOT NULL
            AND NEW.artifact_id=OLD.artifact_id
            AND NEW.job_id=OLD.job_id
            AND NEW.strategy_key=OLD.strategy_key
            AND NEW.logical_key=OLD.logical_key
            AND NEW.provider=OLD.provider
            AND NEW.source_uri=OLD.source_uri
            AND NEW.status=OLD.status
            AND NEW.checksum IS NOT DISTINCT FROM OLD.checksum
            AND NEW.minimum_source_timestamp IS NOT DISTINCT FROM OLD.minimum_source_timestamp
            AND NEW.maximum_source_timestamp IS NOT DISTINCT FROM OLD.maximum_source_timestamp
          THEN
            RETURN NEW;
          END IF;
          RAISE EXCEPTION 'completed backfill artifact % is immutable', OLD.artifact_id
            USING ERRCODE='integrity_constraint_violation';
        END IF;
        RETURN NEW;
      END;
      $$;


--
-- Name: reject_data_gap_identity_change(); Type: FUNCTION; Schema: ingester; Owner: -
--

CREATE FUNCTION ingester.reject_data_gap_identity_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      BEGIN
        IF TG_OP = 'INSERT' THEN
          IF NEW.status = 'repaired' AND NOT EXISTS (
            SELECT 1
            FROM ingester.capture_artifacts artifact
            WHERE artifact.strategy_key = NEW.strategy_key
              AND artifact.artifact_id = NEW.repair_artifact_id
              AND artifact.status = 'completed'
          ) THEN
            RAISE EXCEPTION 'repair artifact must be completed for data gap %', NEW.gap_id
              USING ERRCODE = 'integrity_constraint_violation';
          END IF;
          RETURN NEW;
        END IF;

        IF TG_OP = 'DELETE' THEN
          RAISE EXCEPTION 'data gap % cannot be deleted', OLD.gap_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF OLD.status IN ('repaired', 'unrecoverable')
          AND NEW IS DISTINCT FROM OLD THEN
          RAISE EXCEPTION 'terminal data gap % is immutable', OLD.gap_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF (
          NEW.gap_id,
          NEW.gap_fingerprint,
          NEW.strategy_key,
          NEW.detected_artifact_id,
          NEW.gap_kind,
          NEW.reason_code,
          NEW.reason_message,
          NEW.source_time_start,
          NEW.source_time_end,
          NEW.start_cursor,
          NEW.end_cursor,
          NEW.detected_at,
          NEW.created_at
        ) IS DISTINCT FROM (
          OLD.gap_id,
          OLD.gap_fingerprint,
          OLD.strategy_key,
          OLD.detected_artifact_id,
          OLD.gap_kind,
          OLD.reason_code,
          OLD.reason_message,
          OLD.source_time_start,
          OLD.source_time_end,
          OLD.start_cursor,
          OLD.end_cursor,
          OLD.detected_at,
          OLD.created_at
        ) THEN
          RAISE EXCEPTION 'data gap identity is immutable for %', OLD.gap_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF OLD.status = 'repairing' AND NEW.status = 'open' THEN
          RAISE EXCEPTION 'data gap % cannot return to open after repair begins', OLD.gap_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF NEW.repair_attempts < OLD.repair_attempts THEN
          RAISE EXCEPTION 'data gap repair attempts cannot regress for %', OLD.gap_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF NEW.status = 'repaired' AND NOT EXISTS (
          SELECT 1
          FROM ingester.capture_artifacts artifact
          WHERE artifact.strategy_key = NEW.strategy_key
            AND artifact.artifact_id = NEW.repair_artifact_id
            AND artifact.status = 'completed'
        ) THEN
          RAISE EXCEPTION 'repair artifact must be completed for data gap %', OLD.gap_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        RETURN NEW;
      END;
      $$;


--
-- Name: reject_terminal_artifact_change(); Type: FUNCTION; Schema: ingester; Owner: -
--

CREATE FUNCTION ingester.reject_terminal_artifact_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      BEGIN
        IF TG_OP = 'DELETE' THEN
          RAISE EXCEPTION 'capture artifact % cannot be deleted', OLD.artifact_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF TG_OP = 'UPDATE'
          AND OLD.status IN ('completed', 'failed')
          AND NEW IS DISTINCT FROM OLD THEN
          RAISE EXCEPTION 'terminal capture artifact % is immutable', OLD.artifact_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF TG_OP = 'UPDATE' AND (
          NEW.artifact_id,
          NEW.strategy_key,
          NEW.profile_generation,
          NEW.config_schema_version,
          NEW.config_sha256,
          NEW.config_snapshot,
          NEW.capture_window_start,
          NEW.capture_window_end,
          NEW.created_at
        ) IS DISTINCT FROM (
          OLD.artifact_id,
          OLD.strategy_key,
          OLD.profile_generation,
          OLD.config_schema_version,
          OLD.config_sha256,
          OLD.config_snapshot,
          OLD.capture_window_start,
          OLD.capture_window_end,
          OLD.created_at
        ) THEN
          RAISE EXCEPTION 'capture artifact identity is immutable for %', OLD.artifact_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF TG_OP = 'UPDATE' AND NEW.record_count < OLD.record_count THEN
          RAISE EXCEPTION 'capture artifact record count cannot regress for %', OLD.artifact_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        RETURN NEW;
      END;
      $$;


--
-- Name: remove_verified_binance_aggregate_trade_chunk(uuid, text); Type: FUNCTION; Schema: ingester; Owner: -
--

CREATE FUNCTION ingester.remove_verified_binance_aggregate_trade_chunk(requested_object_id uuid, expected_sha256 text) RETURNS bigint
    LANGUAGE plpgsql
    SET search_path TO 'pg_catalog', 'public', 'ingester', 'market_data'
    AS $$
      DECLARE object_record ingester.drain_objects%ROWTYPE; matching_chunks integer; dropped_chunks integer;
      BEGIN
        SELECT * INTO object_record FROM ingester.drain_objects WHERE object_id=requested_object_id FOR UPDATE;
        IF NOT FOUND OR object_record.status <> 'published'
          OR object_record.strategy_key <> 'binance_spot_btcusdt_aggregate_trades'
          OR object_record.source_relation <> 'market_data.binance_spot_btcusdt_aggregate_trades'
          OR object_record.sha256 <> expected_sha256 THEN
          RAISE EXCEPTION 'drain object is not a verified Binance aggregate-trade publication';
        END IF;
        SELECT count(*) INTO matching_chunks FROM timescaledb_information.chunks
        WHERE hypertable_schema='market_data' AND hypertable_name='binance_spot_btcusdt_aggregate_trades'
          AND chunk_schema=object_record.source_chunk_schema AND chunk_name=object_record.source_chunk_name
          AND range_start=object_record.source_start AND range_end=object_record.source_end;
        IF matching_chunks <> 1 THEN RAISE EXCEPTION 'source chunk identity or bounds changed before removal'; END IF;
        SELECT count(*) INTO dropped_chunks FROM drop_chunks(
          'market_data.binance_spot_btcusdt_aggregate_trades'::regclass,
          older_than=>object_record.source_end,newer_than=>object_record.source_start
        );
        IF dropped_chunks <> 1 THEN RAISE EXCEPTION 'expected one removed chunk, removed %',dropped_chunks; END IF;
        UPDATE ingester.drain_objects SET status='removed',removed_at=clock_timestamp(),updated_at=clock_timestamp()
        WHERE object_id=requested_object_id;
        RETURN object_record.row_count;
      END $$;


--
-- Name: remove_verified_drain_chunk(uuid, text); Type: FUNCTION; Schema: ingester; Owner: -
--

CREATE FUNCTION ingester.remove_verified_drain_chunk(requested_object_id uuid, expected_sha256 text) RETURNS bigint
    LANGUAGE plpgsql
    SET search_path TO 'pg_catalog', 'public', 'ingester', 'market_data', 'polymarket'
    AS $$
      DECLARE
        object_record ingester.drain_objects%ROWTYPE;
        job_cutoff timestamptz;
        target_relation regclass;
        target_schema text;
        target_name text;
        matching_chunks integer;
        dropped_chunks integer;
      BEGIN
        SELECT object.* INTO object_record
        FROM ingester.drain_objects object
        WHERE object.object_id = requested_object_id
        FOR UPDATE;

        IF NOT FOUND
          OR object_record.status <> 'published'
          OR object_record.sha256 <> expected_sha256 THEN
          RAISE EXCEPTION 'drain object is not a verified publication';
        END IF;

        SELECT job.cutoff INTO job_cutoff
        FROM ingester.drain_jobs job
        WHERE job.job_id = object_record.job_id
          AND job.strategy_key = object_record.strategy_key
          AND job.status = 'running';
        IF NOT FOUND OR object_record.source_end > job_cutoff THEN
          RAISE EXCEPTION 'source chunk is outside the drain cutoff';
        END IF;

        CASE
          WHEN object_record.strategy_key = 'polymarket_btc_five_minute_orderbooks'
            AND object_record.source_relation = 'polymarket.btc_five_minute_orderbook_snapshots'
          THEN target_relation := 'polymarket.btc_five_minute_orderbook_snapshots'::regclass;
            target_schema := 'polymarket'; target_name := 'btc_five_minute_orderbook_snapshots';
          WHEN object_record.strategy_key = 'binance_spot_btcusdt_one_second_ohlcv'
            AND object_record.source_relation = 'market_data.binance_spot_btcusdt_one_second_ohlcv'
          THEN target_relation := 'market_data.binance_spot_btcusdt_one_second_ohlcv'::regclass;
            target_schema := 'market_data'; target_name := 'binance_spot_btcusdt_one_second_ohlcv';
          WHEN object_record.strategy_key = 'pmdata_chainlink_btcusd_reference_price'
            AND object_record.source_relation = 'market_data.pmdata_chainlink_btcusd_reference_prices'
          THEN target_relation := 'market_data.pmdata_chainlink_btcusd_reference_prices'::regclass;
            target_schema := 'market_data'; target_name := 'pmdata_chainlink_btcusd_reference_prices';
          WHEN object_record.strategy_key = 'pmdata_chainlink_btcusd_twap'
            AND object_record.source_relation = 'market_data.pmdata_chainlink_btcusd_twap'
          THEN target_relation := 'market_data.pmdata_chainlink_btcusd_twap'::regclass;
            target_schema := 'market_data'; target_name := 'pmdata_chainlink_btcusd_twap';
          WHEN object_record.strategy_key = 'polymarket_chainlink_btcusd_twap'
            AND object_record.source_relation = 'market_data.polymarket_chainlink_btcusd_twap'
          THEN target_relation := 'market_data.polymarket_chainlink_btcusd_twap'::regclass;
            target_schema := 'market_data'; target_name := 'polymarket_chainlink_btcusd_twap';
          WHEN object_record.strategy_key = 'polymarket_reference_price_ticks'
            AND object_record.source_relation = 'polymarket.reference_price_ticks'
          THEN target_relation := 'polymarket.reference_price_ticks'::regclass;
            target_schema := 'polymarket'; target_name := 'reference_price_ticks';
          
          WHEN object_record.strategy_key = 'chainlink_btcusd_one_minute_candles'
            AND object_record.source_relation = 'market_data.chainlink_btcusd_one_minute_candles'
          THEN target_relation := 'market_data.chainlink_btcusd_one_minute_candles'::regclass;
            target_schema := 'market_data'; target_name := 'chainlink_btcusd_one_minute_candles';
          WHEN object_record.strategy_key = 'polymarket_btc_capacity_execution_snapshots'
            AND object_record.source_relation = 'polymarket.btc_market_capacity_execution_snapshots'
          THEN target_relation := 'polymarket.btc_market_capacity_execution_snapshots'::regclass;
            target_schema := 'polymarket'; target_name := 'btc_market_capacity_execution_snapshots';
          WHEN object_record.strategy_key = 'polymarket_btc_feature_snapshots'
            AND object_record.source_relation = 'polymarket.btc_feature_snapshots'
          THEN target_relation := 'polymarket.btc_feature_snapshots'::regclass;
            target_schema := 'polymarket'; target_name := 'btc_feature_snapshots';
          WHEN object_record.strategy_key = 'binance_spot_btcusdt_l2_snapshots'
            AND object_record.source_relation = 'market_data.binance_spot_btcusdt_l2_snapshots'
          THEN target_relation := 'market_data.binance_spot_btcusdt_l2_snapshots'::regclass;
            target_schema := 'market_data'; target_name := 'binance_spot_btcusdt_l2_snapshots';
          WHEN object_record.strategy_key = 'polygon_chainlink_btcusd_oracle_rounds'
            AND object_record.source_relation = 'market_data.polygon_chainlink_btcusd_oracle_rounds'
          THEN target_relation := 'market_data.polygon_chainlink_btcusd_oracle_rounds'::regclass;
            target_schema := 'market_data'; target_name := 'polygon_chainlink_btcusd_oracle_rounds';
          ELSE RAISE EXCEPTION 'dataset is not registered for verified drain removal';
        END CASE;

        SELECT count(*) INTO matching_chunks
        FROM timescaledb_information.chunks chunk
        WHERE chunk.hypertable_schema = target_schema
          AND chunk.hypertable_name = target_name
          AND chunk.chunk_schema = object_record.source_chunk_schema
          AND chunk.chunk_name = object_record.source_chunk_name
          AND chunk.range_start = object_record.source_start
          AND chunk.range_end = object_record.source_end;
        IF matching_chunks <> 1 THEN
          RAISE EXCEPTION 'source chunk identity or bounds changed before removal';
        END IF;

        SELECT count(*) INTO dropped_chunks
        FROM drop_chunks(target_relation, older_than => object_record.source_end,
          newer_than => object_record.source_start);
        IF dropped_chunks <> 1 THEN
          RAISE EXCEPTION 'expected one removed chunk, removed %', dropped_chunks;
        END IF;

        UPDATE ingester.drain_objects
        SET status = 'removed', removed_at = clock_timestamp(), updated_at = clock_timestamp()
        WHERE object_id = requested_object_id;
        RETURN object_record.row_count;
      END
      $$;


--
-- Name: validate_profile_change(); Type: FUNCTION; Schema: ingester; Owner: -
--

CREATE FUNCTION ingester.validate_profile_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      BEGIN
        IF NEW.strategy_key IS DISTINCT FROM OLD.strategy_key
          OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
          RAISE EXCEPTION 'ingester profile identity is immutable for %', OLD.strategy_key
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF NEW.desired_generation < OLD.desired_generation
          OR NEW.desired_generation::numeric > OLD.desired_generation::numeric + 1 THEN
          RAISE EXCEPTION 'invalid desired generation transition for ingester profile %', OLD.strategy_key
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF (
          NEW.desired_state,
          NEW.config_schema_version,
          NEW.config
        ) IS DISTINCT FROM (
          OLD.desired_state,
          OLD.config_schema_version,
          OLD.config
        ) AND NEW.desired_generation::numeric <> OLD.desired_generation::numeric + 1 THEN
          RAISE EXCEPTION 'profile control changes must advance desired generation for %', OLD.strategy_key
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF OLD.applied_generation IS NOT NULL AND (
          NEW.applied_generation IS NULL
          OR NEW.applied_generation < OLD.applied_generation
        ) THEN
          RAISE EXCEPTION 'applied generation cannot regress for ingester profile %', OLD.strategy_key
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF OLD.lease_token IS NOT NULL
          AND NEW.lease_token IS NOT NULL
          AND NEW.lease_token IS DISTINCT FROM OLD.lease_token
          AND OLD.lease_expires_at > clock_timestamp() THEN
          RAISE EXCEPTION 'active lease cannot be replaced for ingester profile %', OLD.strategy_key
            USING ERRCODE = 'lock_not_available';
        END IF;

        IF OLD.lease_token IS NOT NULL
          AND NEW.lease_token = OLD.lease_token
          AND (
            NEW.lease_owner IS DISTINCT FROM OLD.lease_owner
            OR NEW.heartbeat_at < OLD.heartbeat_at
            OR NEW.lease_expires_at < OLD.lease_expires_at
          ) THEN
          RAISE EXCEPTION 'lease ownership and timestamps cannot regress for ingester profile %', OLD.strategy_key
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        RETURN NEW;
      END;
      $$;


--
-- Name: is_valid_binance_spot_btcusdt_l2_book(jsonb, jsonb, integer); Type: FUNCTION; Schema: market_data; Owner: -
--

CREATE FUNCTION market_data.is_valid_binance_spot_btcusdt_l2_book(bids jsonb, asks jsonb, sample_depth integer) RETURNS boolean
    LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE
    SET search_path TO 'pg_catalog'
    AS $_$
      DECLARE
        level jsonb;
        raw_price text;
        raw_quantity text;
        price numeric;
        quantity numeric;
        previous_price numeric;
        best_bid numeric;
        best_ask numeric;
      BEGIN
        IF jsonb_typeof(bids) <> 'array'
          OR jsonb_typeof(asks) <> 'array'
          OR sample_depth NOT BETWEEN 1 AND 1000
          OR jsonb_array_length(bids) <> sample_depth
          OR jsonb_array_length(asks) <> sample_depth THEN
          RETURN FALSE;
        END IF;

        previous_price := NULL;
        FOR level IN SELECT value FROM jsonb_array_elements(bids) LOOP
          IF jsonb_typeof(level) <> 'array'
            OR jsonb_array_length(level) <> 2
            OR jsonb_typeof(level -> 0) <> 'string'
            OR jsonb_typeof(level -> 1) <> 'string' THEN
            RETURN FALSE;
          END IF;

          raw_price := level ->> 0;
          raw_quantity := level ->> 1;
          IF length(raw_price) NOT BETWEEN 1 AND 64
            OR length(raw_quantity) NOT BETWEEN 1 AND 64
            OR raw_price !~ '^(0|[1-9][0-9]*)(\.[0-9]+)?$'
            OR raw_quantity !~ '^(0|[1-9][0-9]*)(\.[0-9]+)?$' THEN
            RETURN FALSE;
          END IF;

          price := raw_price::numeric;
          quantity := raw_quantity::numeric;
          IF price <= 0
            OR quantity <= 0
            OR (previous_price IS NOT NULL AND price >= previous_price) THEN
            RETURN FALSE;
          END IF;
          IF best_bid IS NULL THEN
            best_bid := price;
          END IF;
          previous_price := price;
        END LOOP;

        previous_price := NULL;
        FOR level IN SELECT value FROM jsonb_array_elements(asks) LOOP
          IF jsonb_typeof(level) <> 'array'
            OR jsonb_array_length(level) <> 2
            OR jsonb_typeof(level -> 0) <> 'string'
            OR jsonb_typeof(level -> 1) <> 'string' THEN
            RETURN FALSE;
          END IF;

          raw_price := level ->> 0;
          raw_quantity := level ->> 1;
          IF length(raw_price) NOT BETWEEN 1 AND 64
            OR length(raw_quantity) NOT BETWEEN 1 AND 64
            OR raw_price !~ '^(0|[1-9][0-9]*)(\.[0-9]+)?$'
            OR raw_quantity !~ '^(0|[1-9][0-9]*)(\.[0-9]+)?$' THEN
            RETURN FALSE;
          END IF;

          price := raw_price::numeric;
          quantity := raw_quantity::numeric;
          IF price <= 0
            OR quantity <= 0
            OR (previous_price IS NOT NULL AND price <= previous_price) THEN
            RETURN FALSE;
          END IF;
          IF best_ask IS NULL THEN
            best_ask := price;
          END IF;
          previous_price := price;
        END LOOP;

        RETURN best_bid IS NOT NULL
          AND best_ask IS NOT NULL
          AND best_bid < best_ask;
      END;
      $_$;


--
-- Name: is_valid_polymarket_btc_five_minute_book(jsonb, jsonb, integer, integer, numeric, numeric, integer); Type: FUNCTION; Schema: market_data; Owner: -
--

CREATE FUNCTION market_data.is_valid_polymarket_btc_five_minute_book(bids jsonb, asks jsonb, bid_depth integer, ask_depth integer, best_bid numeric, best_ask numeric, maximum_depth integer) RETURNS boolean
    LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE
    SET search_path TO 'pg_catalog'
    AS $_$
      DECLARE
        level jsonb;
        raw_price text;
        raw_quantity text;
        price numeric;
        quantity numeric;
        previous_price numeric;
        derived_best_bid numeric;
        derived_best_ask numeric;
      BEGIN
        IF jsonb_typeof(bids) <> 'array'
          OR jsonb_typeof(asks) <> 'array'
          OR maximum_depth NOT BETWEEN 1 AND 1000
          OR bid_depth NOT BETWEEN 0 AND maximum_depth
          OR ask_depth NOT BETWEEN 0 AND maximum_depth
          OR jsonb_array_length(bids) <> bid_depth
          OR jsonb_array_length(asks) <> ask_depth THEN
          RETURN FALSE;
        END IF;

        previous_price := NULL;
        FOR level IN SELECT value FROM jsonb_array_elements(bids) LOOP
          IF jsonb_typeof(level) <> 'array'
            OR jsonb_array_length(level) <> 2
            OR jsonb_typeof(level -> 0) <> 'string'
            OR jsonb_typeof(level -> 1) <> 'string' THEN
            RETURN FALSE;
          END IF;

          raw_price := level ->> 0;
          raw_quantity := level ->> 1;
          IF length(raw_price) NOT BETWEEN 1 AND 64
            OR length(raw_quantity) NOT BETWEEN 1 AND 64
            OR raw_price !~ '^(0|[1-9][0-9]*)(\.[0-9]+)?$'
            OR raw_quantity !~ '^(0|[1-9][0-9]*)(\.[0-9]+)?$' THEN
            RETURN FALSE;
          END IF;

          price := raw_price::numeric;
          quantity := raw_quantity::numeric;
          IF price <= 0
            OR price >= 1
            OR quantity <= 0
            OR (previous_price IS NOT NULL AND price >= previous_price) THEN
            RETURN FALSE;
          END IF;
          IF derived_best_bid IS NULL THEN
            derived_best_bid := price;
          END IF;
          previous_price := price;
        END LOOP;

        previous_price := NULL;
        FOR level IN SELECT value FROM jsonb_array_elements(asks) LOOP
          IF jsonb_typeof(level) <> 'array'
            OR jsonb_array_length(level) <> 2
            OR jsonb_typeof(level -> 0) <> 'string'
            OR jsonb_typeof(level -> 1) <> 'string' THEN
            RETURN FALSE;
          END IF;

          raw_price := level ->> 0;
          raw_quantity := level ->> 1;
          IF length(raw_price) NOT BETWEEN 1 AND 64
            OR length(raw_quantity) NOT BETWEEN 1 AND 64
            OR raw_price !~ '^(0|[1-9][0-9]*)(\.[0-9]+)?$'
            OR raw_quantity !~ '^(0|[1-9][0-9]*)(\.[0-9]+)?$' THEN
            RETURN FALSE;
          END IF;

          price := raw_price::numeric;
          quantity := raw_quantity::numeric;
          IF price <= 0
            OR price >= 1
            OR quantity <= 0
            OR (previous_price IS NOT NULL AND price <= previous_price) THEN
            RETURN FALSE;
          END IF;
          IF derived_best_ask IS NULL THEN
            derived_best_ask := price;
          END IF;
          previous_price := price;
        END LOOP;

        RETURN (best_bid IS NOT DISTINCT FROM derived_best_bid)
          AND (best_ask IS NOT DISTINCT FROM derived_best_ask)
          AND (
            derived_best_bid IS NULL
            OR derived_best_ask IS NULL
            OR derived_best_bid < derived_best_ask
          );
      END;
      $_$;


--
-- Name: reject_source_fact_change(); Type: FUNCTION; Schema: market_data; Owner: -
--

CREATE FUNCTION market_data.reject_source_fact_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      BEGIN
        RAISE EXCEPTION 'market source facts are immutable'
          USING ERRCODE = 'integrity_constraint_violation';
      END;
      $$;


--
-- Name: reject_backfill_retention_event_change(); Type: FUNCTION; Schema: polymarket; Owner: -
--

CREATE FUNCTION polymarket.reject_backfill_retention_event_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      BEGIN
        RAISE EXCEPTION
          'backfill materialization retention evidence is immutable'
          USING ERRCODE = 'integrity_constraint_violation';
      END;
      $$;


--
-- Name: reject_binance_btcusdt_l2_feature_change(); Type: FUNCTION; Schema: polymarket; Owner: -
--

CREATE FUNCTION polymarket.reject_binance_btcusdt_l2_feature_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      BEGIN
        RAISE EXCEPTION
          'historical Binance BTCUSDT L2 feature row is immutable'
          USING ERRCODE = 'integrity_constraint_violation';
      END;
      $$;


--
-- Name: reject_binance_spot_btcusdt_l2_feature_change(); Type: FUNCTION; Schema: polymarket; Owner: -
--

CREATE FUNCTION polymarket.reject_binance_spot_btcusdt_l2_feature_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      BEGIN
        RAISE EXCEPTION
          'historical Binance spot BTCUSDT L2 feature row is immutable'
          USING ERRCODE = 'integrity_constraint_violation';
      END;
      $$;


--
-- Name: reject_btc_market_execution_snapshot_change(); Type: FUNCTION; Schema: polymarket; Owner: -
--

CREATE FUNCTION polymarket.reject_btc_market_execution_snapshot_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      BEGIN
        IF TG_OP = 'DELETE'
          AND EXISTS (
            SELECT 1
            FROM polymarket.backfill_artifacts artifact
            WHERE artifact.artifact_id = OLD.artifact_id
              AND artifact.provider = 'pmxt_v2_capacity_execution_snapshots_v2'
              AND COALESCE(
                (artifact.metadata ->> 'source_events_consumed')::bigint,
                0
              ) = 0
          )
        THEN
          RETURN OLD;
        END IF;

        RAISE EXCEPTION
          'historical BTC execution snapshot is immutable'
          USING ERRCODE = 'integrity_constraint_violation';
      END;
      $$;


--
-- Name: reject_completed_backfill_artifact_change(); Type: FUNCTION; Schema: polymarket; Owner: -
--

CREATE FUNCTION polymarket.reject_completed_backfill_artifact_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      BEGIN
        IF TG_OP = 'DELETE' THEN
          IF OLD.status = 'completed' THEN
            RAISE EXCEPTION
              'completed backfill artifact % is immutable', OLD.artifact_id
              USING ERRCODE = 'integrity_constraint_violation';
          END IF;
          RETURN OLD;
        END IF;

        IF OLD.status = 'completed' AND NEW IS DISTINCT FROM OLD THEN
          IF OLD.provider = 'pmxt_v2_capacity_execution_snapshots_v2'
            AND COALESCE(
              (OLD.metadata ->> 'source_events_consumed')::bigint,
              0
            ) = 0
            AND NEW.record_count = 0
            AND NEW.metadata ->> 'superseded_reason'
              = 'reconstructed_from_local_canonical_orderbooks'
            AND NEW.metadata ->> 'superseded_by_artifact_id' IS NOT NULL
            AND NEW.artifact_id = OLD.artifact_id
            AND NEW.job_id = OLD.job_id
            AND NEW.ingester_key = OLD.ingester_key
            AND NEW.logical_key = OLD.logical_key
            AND NEW.provider = OLD.provider
            AND NEW.source_uri = OLD.source_uri
            AND NEW.status = OLD.status
            AND NEW.actual_checksum IS NOT DISTINCT FROM OLD.actual_checksum
            AND NEW.minimum_source_timestamp IS NOT DISTINCT FROM OLD.minimum_source_timestamp
            AND NEW.maximum_source_timestamp IS NOT DISTINCT FROM OLD.maximum_source_timestamp
          THEN
            RETURN NEW;
          END IF;

          RAISE EXCEPTION
            'completed backfill artifact % is immutable', OLD.artifact_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;
        RETURN NEW;
      END;
      $$;


--
-- Name: reject_historical_market_event_change(); Type: FUNCTION; Schema: polymarket; Owner: -
--

CREATE FUNCTION polymarket.reject_historical_market_event_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      BEGIN
        RAISE EXCEPTION 'historical market source rows are immutable';
      END;
      $$;


--
-- Name: reject_immutable_btc_reference_fact_change(); Type: FUNCTION; Schema: polymarket; Owner: -
--

CREATE FUNCTION polymarket.reject_immutable_btc_reference_fact_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      BEGIN
        RAISE EXCEPTION
          'BTC market reference fact % is immutable', OLD.fact_id
          USING ERRCODE = 'integrity_constraint_violation';
      END;
      $$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: polymarket_chainlink_btcusd_twap; Type: TABLE; Schema: market_data; Owner: -
--

CREATE TABLE market_data.polymarket_chainlink_btcusd_twap (
    source_timestamp timestamp with time zone NOT NULL,
    published_at timestamp with time zone NOT NULL,
    received_at timestamp with time zone NOT NULL,
    source text DEFAULT 'polymarket_rtds_chainlink_twap'::text NOT NULL,
    symbol text NOT NULL,
    window_seconds smallint NOT NULL,
    twap_price numeric(38,18) NOT NULL,
    full_accuracy_value text NOT NULL,
    source_payload jsonb NOT NULL,
    payload_sha256 character(64) NOT NULL,
    strategy_key text DEFAULT 'polymarket_chainlink_btcusd_twap'::text NOT NULL,
    capture_artifact_id uuid NOT NULL,
    ingested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT chk_market_data_polymarket_chainlink_btcusd_twap_identity CHECK (((source = 'polymarket_rtds_chainlink_twap'::text) AND (strategy_key = 'polymarket_chainlink_btcusd_twap'::text) AND (symbol = 'btc/usd'::text) AND (window_seconds = ANY (ARRAY[30, 60])))),
    CONSTRAINT chk_market_data_polymarket_chainlink_btcusd_twap_payload CHECK (((jsonb_typeof(source_payload) = 'object'::text) AND (octet_length((source_payload)::text) <= 4096) AND (payload_sha256 ~ '^[0-9a-f]{64}$'::text))),
    CONSTRAINT chk_market_data_polymarket_chainlink_btcusd_twap_time CHECK (((source_timestamp <= published_at) AND (published_at <= (received_at + '00:00:05'::interval)))),
    CONSTRAINT chk_market_data_polymarket_chainlink_btcusd_twap_value CHECK (((full_accuracy_value ~ '^-?[0-9]{1,29}$'::text) AND ((twap_price * ('1000000000000000000'::bigint)::numeric) = (full_accuracy_value)::numeric) AND (twap_price > (0)::numeric)))
);


--
-- Name: pmdata_chainlink_btcusd_twap; Type: TABLE; Schema: market_data; Owner: -
--

CREATE TABLE market_data.pmdata_chainlink_btcusd_twap (
    source_timestamp timestamp with time zone NOT NULL,
    provider_received_at timestamp with time zone NOT NULL,
    valid_from_timestamp timestamp with time zone NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    symbol text DEFAULT 'BTCUSD'::text NOT NULL,
    window_seconds smallint NOT NULL,
    twap_price numeric(38,18) NOT NULL,
    full_accuracy_value text NOT NULL,
    report_version text NOT NULL,
    source_date date NOT NULL,
    archive_row_number bigint NOT NULL,
    artifact_id uuid NOT NULL,
    ingested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT chk_market_data_pmdata_chainlink_btcusd_twap_identity CHECK (((symbol = 'BTCUSD'::text) AND (window_seconds = ANY (ARRAY[30, 60])) AND (archive_row_number >= 0))),
    CONSTRAINT chk_market_data_pmdata_chainlink_btcusd_twap_time CHECK (((valid_from_timestamp <= source_timestamp) AND (expires_at > source_timestamp) AND (source_timestamp >= (source_date)::timestamp with time zone) AND (source_timestamp < ((source_date)::timestamp with time zone + '1 day'::interval)))),
    CONSTRAINT chk_market_data_pmdata_chainlink_btcusd_twap_value CHECK (((full_accuracy_value ~ '^[0-9]{1,29}$'::text) AND ((twap_price * ('1000000000000000000'::bigint)::numeric) = (full_accuracy_value)::numeric) AND (twap_price > (0)::numeric))),
    CONSTRAINT chk_market_data_pmdata_chainlink_btcusd_twap_version CHECK ((length(btrim(report_version)) > 0))
);


--
-- Name: pmdata_chainlink_btcusd_reference_prices; Type: TABLE; Schema: market_data; Owner: -
--

CREATE TABLE market_data.pmdata_chainlink_btcusd_reference_prices (
    source text NOT NULL,
    feed_id text NOT NULL,
    source_timestamp timestamp with time zone NOT NULL,
    valid_from_timestamp timestamp with time zone,
    provider_available_at timestamp with time zone,
    received_at timestamp with time zone NOT NULL,
    price numeric(38,18) NOT NULL,
    bid numeric(38,18),
    ask numeric(38,18),
    report_sha256 character(64) NOT NULL,
    payload_sha256 character(64) NOT NULL,
    strategy_key text NOT NULL,
    capture_artifact_id uuid,
    ingested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    expires_at timestamp with time zone,
    report_version text,
    source_date date,
    archive_row_number bigint,
    backfill_artifact_id uuid,
    report_hash_kind text NOT NULL,
    CONSTRAINT chk_market_data_pmdata_chainlink_btcusd_reference_prices_hashes CHECK (((report_sha256 ~ '^[0-9a-f]{64}$'::text) AND (payload_sha256 ~ '^[0-9a-f]{64}$'::text) AND ((report_version IS NULL) OR (length(btrim(report_version)) > 0)))),
    CONSTRAINT chk_market_data_pmdata_chainlink_btcusd_reference_prices_identi CHECK (((source = 'pmdata_chainlink_streams'::text) AND (feed_id = '0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8'::text) AND (strategy_key = 'pmdata_chainlink_btcusd_refprice_backfill'::text) AND (report_hash_kind = 'canonical_archive_row'::text) AND (capture_artifact_id IS NULL) AND (backfill_artifact_id IS NOT NULL) AND (source_date IS NOT NULL) AND (archive_row_number >= 0))),
    CONSTRAINT chk_market_data_pmdata_chainlink_btcusd_reference_prices_time CHECK ((((valid_from_timestamp IS NULL) OR (valid_from_timestamp <= source_timestamp)) AND ((expires_at IS NULL) OR (expires_at > source_timestamp)) AND ((source_date IS NULL) OR ((source_timestamp >= (source_date)::timestamp with time zone) AND (source_timestamp < ((source_date)::timestamp with time zone + '1 day'::interval)))))),
    CONSTRAINT chk_market_data_pmdata_chainlink_btcusd_reference_prices_values CHECK (((price > (0)::numeric) AND (((bid IS NULL) AND (ask IS NULL)) OR ((bid > (0)::numeric) AND (bid <= price) AND (price <= ask)))))
);


--
-- Name: binance_spot_btcusdt_l2_one_second_features; Type: TABLE; Schema: market_data; Owner: -
--

CREATE TABLE market_data.binance_spot_btcusdt_l2_one_second_features (
    symbol text NOT NULL,
    second_start timestamp with time zone NOT NULL,
    source_event_timestamp timestamp with time zone NOT NULL,
    provider_received_at timestamp with time zone NOT NULL,
    available_at timestamp with time zone NOT NULL,
    source_update_id bigint NOT NULL,
    feature_schema_version text NOT NULL,
    quality_status text NOT NULL,
    artifact_id uuid NOT NULL,
    midpoint numeric(30,10) NOT NULL,
    microprice numeric(30,10) NOT NULL,
    spread_bps numeric(20,10) NOT NULL,
    bid_depth_5 numeric(30,10) NOT NULL,
    ask_depth_5 numeric(30,10) NOT NULL,
    imbalance_5 numeric(20,10) NOT NULL,
    bid_depth_10 numeric(30,10) NOT NULL,
    ask_depth_10 numeric(30,10) NOT NULL,
    imbalance_10 numeric(20,10) NOT NULL,
    bid_depth_20 numeric(30,10) NOT NULL,
    ask_depth_20 numeric(30,10) NOT NULL,
    imbalance_20 numeric(20,10) NOT NULL,
    bid_depth_slope_20 numeric(20,10) NOT NULL,
    ask_depth_slope_20 numeric(20,10) NOT NULL,
    bid_depth_concentration_20 numeric(20,10) NOT NULL,
    ask_depth_concentration_20 numeric(20,10) NOT NULL,
    bid_quote_replenishment_1s numeric(30,10) NOT NULL,
    ask_quote_replenishment_1s numeric(30,10) NOT NULL,
    bid_quote_churn_1s numeric(30,10) NOT NULL,
    ask_quote_churn_1s numeric(30,10) NOT NULL,
    midpoint_change_bps_1s numeric(30,10) NOT NULL,
    spread_bps_delta_1s numeric(30,10) NOT NULL,
    depth_20_change_bps_1s numeric(30,10) NOT NULL,
    imbalance_20_delta_1s numeric(20,10) NOT NULL,
    midpoint_change_bps_5s numeric(30,10) NOT NULL,
    spread_bps_delta_5s numeric(30,10) NOT NULL,
    depth_20_change_bps_5s numeric(30,10) NOT NULL,
    imbalance_20_delta_5s numeric(20,10) NOT NULL,
    midpoint_change_bps_15s numeric(30,10) NOT NULL,
    spread_bps_delta_15s numeric(30,10) NOT NULL,
    depth_20_change_bps_15s numeric(30,10) NOT NULL,
    imbalance_20_delta_15s numeric(20,10) NOT NULL,
    midpoint_change_bps_30s numeric(30,10) NOT NULL,
    spread_bps_delta_30s numeric(30,10) NOT NULL,
    depth_20_change_bps_30s numeric(30,10) NOT NULL,
    imbalance_20_delta_30s numeric(20,10) NOT NULL,
    midpoint_change_bps_60s numeric(30,10) NOT NULL,
    spread_bps_delta_60s numeric(30,10) NOT NULL,
    depth_20_change_bps_60s numeric(30,10) NOT NULL,
    imbalance_20_delta_60s numeric(20,10) NOT NULL,
    ingested_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_binance_spot_btcusdt_l2_causality CHECK (((source_event_timestamp <= available_at) AND (provider_received_at <= available_at) AND (second_start <= available_at) AND (available_at < (second_start + '00:00:01'::interval)))),
    CONSTRAINT chk_binance_spot_btcusdt_l2_depth CHECK (((bid_depth_5 > (0)::numeric) AND (bid_depth_5 <= bid_depth_10) AND (bid_depth_10 <= bid_depth_20) AND (ask_depth_5 > (0)::numeric) AND (ask_depth_5 <= ask_depth_10) AND (ask_depth_10 <= ask_depth_20))),
    CONSTRAINT chk_binance_spot_btcusdt_l2_feature_schema CHECK ((feature_schema_version = 'binance-spot-btcusdt-l2-one-second-features-v1'::text)),
    CONSTRAINT chk_binance_spot_btcusdt_l2_flow CHECK (((bid_quote_replenishment_1s >= (0)::numeric) AND (ask_quote_replenishment_1s >= (0)::numeric) AND (bid_quote_churn_1s >= (0)::numeric) AND (ask_quote_churn_1s >= (0)::numeric))),
    CONSTRAINT chk_binance_spot_btcusdt_l2_imbalance CHECK ((((imbalance_5 >= ('-1'::integer)::numeric) AND (imbalance_5 <= (1)::numeric)) AND ((imbalance_10 >= ('-1'::integer)::numeric) AND (imbalance_10 <= (1)::numeric)) AND ((imbalance_20 >= ('-1'::integer)::numeric) AND (imbalance_20 <= (1)::numeric)) AND ((imbalance_20_delta_1s >= ('-2'::integer)::numeric) AND (imbalance_20_delta_1s <= (2)::numeric)) AND ((imbalance_20_delta_5s >= ('-2'::integer)::numeric) AND (imbalance_20_delta_5s <= (2)::numeric)) AND ((imbalance_20_delta_15s >= ('-2'::integer)::numeric) AND (imbalance_20_delta_15s <= (2)::numeric)) AND ((imbalance_20_delta_30s >= ('-2'::integer)::numeric) AND (imbalance_20_delta_30s <= (2)::numeric)) AND ((imbalance_20_delta_60s >= ('-2'::integer)::numeric) AND (imbalance_20_delta_60s <= (2)::numeric)))),
    CONSTRAINT chk_binance_spot_btcusdt_l2_prices CHECK (((midpoint > (0)::numeric) AND (microprice > (0)::numeric) AND (spread_bps >= (0)::numeric))),
    CONSTRAINT chk_binance_spot_btcusdt_l2_quality CHECK ((quality_status = 'qualified'::text)),
    CONSTRAINT chk_binance_spot_btcusdt_l2_second_alignment CHECK ((second_start = date_trunc('second'::text, second_start))),
    CONSTRAINT chk_binance_spot_btcusdt_l2_shape CHECK (((bid_depth_slope_20 >= (0)::numeric) AND (ask_depth_slope_20 >= (0)::numeric) AND ((bid_depth_concentration_20 >= (0)::numeric) AND (bid_depth_concentration_20 <= (1)::numeric)) AND ((ask_depth_concentration_20 >= (0)::numeric) AND (ask_depth_concentration_20 <= (1)::numeric)))),
    CONSTRAINT chk_binance_spot_btcusdt_l2_source_update_id CHECK ((source_update_id >= 0)),
    CONSTRAINT chk_binance_spot_btcusdt_l2_symbol CHECK ((symbol = 'BTCUSDT'::text))
);


--
-- Name: binance_futures_btcusdt_l2_one_second_features; Type: TABLE; Schema: market_data; Owner: -
--

CREATE TABLE market_data.binance_futures_btcusdt_l2_one_second_features (
    symbol text NOT NULL,
    second_start timestamp with time zone NOT NULL,
    source_event_timestamp timestamp with time zone NOT NULL,
    provider_received_at timestamp with time zone NOT NULL,
    available_at timestamp with time zone NOT NULL,
    source_update_id bigint NOT NULL,
    feature_schema_version text NOT NULL,
    quality_status text NOT NULL,
    artifact_id uuid NOT NULL,
    midpoint numeric(30,10) NOT NULL,
    microprice numeric(30,10) NOT NULL,
    spread_bps numeric(20,10) NOT NULL,
    bid_depth_5 numeric(30,10) NOT NULL,
    ask_depth_5 numeric(30,10) NOT NULL,
    imbalance_5 numeric(20,10) NOT NULL,
    bid_depth_10 numeric(30,10) NOT NULL,
    ask_depth_10 numeric(30,10) NOT NULL,
    imbalance_10 numeric(20,10) NOT NULL,
    bid_depth_20 numeric(30,10) NOT NULL,
    ask_depth_20 numeric(30,10) NOT NULL,
    imbalance_20 numeric(20,10) NOT NULL,
    bid_depth_slope_20 numeric(20,10) NOT NULL,
    ask_depth_slope_20 numeric(20,10) NOT NULL,
    bid_depth_concentration_20 numeric(20,10) NOT NULL,
    ask_depth_concentration_20 numeric(20,10) NOT NULL,
    bid_quote_replenishment_1s numeric(30,10) NOT NULL,
    ask_quote_replenishment_1s numeric(30,10) NOT NULL,
    bid_quote_churn_1s numeric(30,10) NOT NULL,
    ask_quote_churn_1s numeric(30,10) NOT NULL,
    midpoint_change_bps_1s numeric(30,10) NOT NULL,
    spread_bps_delta_1s numeric(30,10) NOT NULL,
    depth_20_change_bps_1s numeric(30,10) NOT NULL,
    imbalance_20_delta_1s numeric(20,10) NOT NULL,
    midpoint_change_bps_5s numeric(30,10) NOT NULL,
    spread_bps_delta_5s numeric(30,10) NOT NULL,
    depth_20_change_bps_5s numeric(30,10) NOT NULL,
    imbalance_20_delta_5s numeric(20,10) NOT NULL,
    midpoint_change_bps_15s numeric(30,10) NOT NULL,
    spread_bps_delta_15s numeric(30,10) NOT NULL,
    depth_20_change_bps_15s numeric(30,10) NOT NULL,
    imbalance_20_delta_15s numeric(20,10) NOT NULL,
    midpoint_change_bps_30s numeric(30,10) NOT NULL,
    spread_bps_delta_30s numeric(30,10) NOT NULL,
    depth_20_change_bps_30s numeric(30,10) NOT NULL,
    imbalance_20_delta_30s numeric(20,10) NOT NULL,
    midpoint_change_bps_60s numeric(30,10) NOT NULL,
    spread_bps_delta_60s numeric(30,10) NOT NULL,
    depth_20_change_bps_60s numeric(30,10) NOT NULL,
    imbalance_20_delta_60s numeric(20,10) NOT NULL,
    ingested_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_binance_btcusdt_l2_causality CHECK (((source_event_timestamp <= available_at) AND (provider_received_at <= available_at) AND (second_start <= available_at) AND (available_at < (second_start + '00:00:01'::interval)))),
    CONSTRAINT chk_binance_btcusdt_l2_depth CHECK (((bid_depth_5 > (0)::numeric) AND (bid_depth_5 <= bid_depth_10) AND (bid_depth_10 <= bid_depth_20) AND (ask_depth_5 > (0)::numeric) AND (ask_depth_5 <= ask_depth_10) AND (ask_depth_10 <= ask_depth_20))),
    CONSTRAINT chk_binance_btcusdt_l2_feature_schema CHECK ((feature_schema_version = 'binance-btcusdt-l2-one-second-features-v1'::text)),
    CONSTRAINT chk_binance_btcusdt_l2_flow CHECK (((bid_quote_replenishment_1s >= (0)::numeric) AND (ask_quote_replenishment_1s >= (0)::numeric) AND (bid_quote_churn_1s >= (0)::numeric) AND (ask_quote_churn_1s >= (0)::numeric))),
    CONSTRAINT chk_binance_btcusdt_l2_imbalance CHECK ((((imbalance_5 >= ('-1'::integer)::numeric) AND (imbalance_5 <= (1)::numeric)) AND ((imbalance_10 >= ('-1'::integer)::numeric) AND (imbalance_10 <= (1)::numeric)) AND ((imbalance_20 >= ('-1'::integer)::numeric) AND (imbalance_20 <= (1)::numeric)) AND ((imbalance_20_delta_1s >= ('-2'::integer)::numeric) AND (imbalance_20_delta_1s <= (2)::numeric)) AND ((imbalance_20_delta_5s >= ('-2'::integer)::numeric) AND (imbalance_20_delta_5s <= (2)::numeric)) AND ((imbalance_20_delta_15s >= ('-2'::integer)::numeric) AND (imbalance_20_delta_15s <= (2)::numeric)) AND ((imbalance_20_delta_30s >= ('-2'::integer)::numeric) AND (imbalance_20_delta_30s <= (2)::numeric)) AND ((imbalance_20_delta_60s >= ('-2'::integer)::numeric) AND (imbalance_20_delta_60s <= (2)::numeric)))),
    CONSTRAINT chk_binance_btcusdt_l2_prices CHECK (((midpoint > (0)::numeric) AND (microprice > (0)::numeric) AND (spread_bps >= (0)::numeric))),
    CONSTRAINT chk_binance_btcusdt_l2_quality CHECK ((quality_status = 'qualified'::text)),
    CONSTRAINT chk_binance_btcusdt_l2_second_alignment CHECK ((second_start = date_trunc('second'::text, second_start))),
    CONSTRAINT chk_binance_btcusdt_l2_shape CHECK (((bid_depth_slope_20 >= (0)::numeric) AND (ask_depth_slope_20 >= (0)::numeric) AND ((bid_depth_concentration_20 >= (0)::numeric) AND (bid_depth_concentration_20 <= (1)::numeric)) AND ((ask_depth_concentration_20 >= (0)::numeric) AND (ask_depth_concentration_20 <= (1)::numeric)))),
    CONSTRAINT chk_binance_btcusdt_l2_source_update_id CHECK ((source_update_id >= 0)),
    CONSTRAINT chk_binance_btcusdt_l2_symbol CHECK ((symbol = 'BTCUSDT'::text))
);


--
-- Name: fills; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.fills (
    fill_id uuid NOT NULL,
    order_id text NOT NULL,
    token_id text NOT NULL,
    timestamp_utc timestamp with time zone NOT NULL,
    price numeric(18,8) NOT NULL,
    size numeric(30,10) NOT NULL,
    fee numeric(30,10) DEFAULT 0 NOT NULL,
    source text NOT NULL,
    raw_payload jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    process_id uuid,
    CONSTRAINT chk_polymarket_fills_source CHECK ((source = ANY (ARRAY['sim'::text, 'paper'::text, 'live'::text])))
);


--
-- Name: risk_events; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.risk_events (
    event_id uuid DEFAULT gen_random_uuid() NOT NULL,
    timestamp_utc timestamp with time zone NOT NULL,
    event_type text NOT NULL,
    severity text NOT NULL,
    message text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: trading_process_events; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.trading_process_events (
    event_id uuid DEFAULT gen_random_uuid() NOT NULL,
    process_id uuid NOT NULL,
    timestamp_utc timestamp with time zone DEFAULT now() NOT NULL,
    level text NOT NULL,
    event_type text NOT NULL,
    message text,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_poly_trading_process_events_level CHECK ((level = ANY (ARRAY['debug'::text, 'info'::text, 'warn'::text, 'error'::text])))
);


--
-- Name: reference_price_ticks; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.reference_price_ticks (
    tick_id uuid NOT NULL,
    source_timestamp timestamp with time zone NOT NULL,
    received_at timestamp with time zone NOT NULL,
    persisted_at timestamp with time zone DEFAULT now() NOT NULL,
    source text NOT NULL,
    symbol text NOT NULL,
    price numeric(30,10) NOT NULL,
    envelope_timestamp timestamp with time zone,
    connection_id uuid NOT NULL,
    ingest_sequence bigint NOT NULL,
    source_event_id text,
    dedup_key text NOT NULL,
    clock_skew_ms bigint NOT NULL,
    integrity_status text NOT NULL,
    raw_payload jsonb DEFAULT '{}'::jsonb NOT NULL,
    CONSTRAINT chk_reference_price_positive CHECK ((price > (0)::numeric)),
    CONSTRAINT chk_reference_price_source CHECK ((source = ANY (ARRAY['direct_binance'::text, 'rtds_binance'::text, 'rtds_chainlink'::text])))
);


--
-- Name: btc_feature_snapshots; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.btc_feature_snapshots (
    snapshot_id uuid NOT NULL,
    feature_as_of timestamp with time zone NOT NULL,
    received_at timestamp with time zone NOT NULL,
    market_id text NOT NULL,
    window_start timestamp with time zone NOT NULL,
    window_end timestamp with time zone NOT NULL,
    feature_schema_version text NOT NULL,
    feature_hash text NOT NULL,
    chainlink_price numeric(30,10) NOT NULL,
    chainlink_open_price numeric(30,10) NOT NULL,
    binance_price numeric(30,10) NOT NULL,
    seconds_to_close numeric(18,6) NOT NULL,
    chainlink_gap_bps numeric(30,10) NOT NULL,
    binance_return_1s_bps numeric(30,10) NOT NULL,
    binance_return_5s_bps numeric(30,10) NOT NULL,
    binance_return_30s_bps numeric(30,10) NOT NULL,
    realized_vol_30s_bps numeric(30,10) NOT NULL,
    basis_bps numeric(30,10) NOT NULL,
    up_best_bid numeric(18,8),
    up_best_ask numeric(18,8),
    down_best_bid numeric(18,8),
    down_best_ask numeric(18,8),
    up_depth_ask numeric(30,10) DEFAULT 0 NOT NULL,
    down_depth_ask numeric(30,10) DEFAULT 0 NOT NULL,
    up_imbalance numeric(18,8),
    down_imbalance numeric(18,8),
    chainlink_age_ms bigint NOT NULL,
    binance_age_ms bigint NOT NULL,
    book_age_ms bigint NOT NULL,
    source_skew_ms bigint NOT NULL,
    fair_up_probability numeric(18,10),
    fair_up_lower numeric(18,10),
    fair_up_upper numeric(18,10),
    deterministic_logit numeric(30,10),
    readiness_status text NOT NULL,
    quality_flags jsonb DEFAULT '[]'::jsonb NOT NULL,
    features jsonb NOT NULL,
    lineage jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: binance_spot_btcusdt_aggregate_trades; Type: TABLE; Schema: market_data; Owner: -
--

CREATE TABLE market_data.binance_spot_btcusdt_aggregate_trades (
    source text DEFAULT 'binance_spot'::text NOT NULL,
    symbol text NOT NULL,
    aggregate_trade_id bigint NOT NULL,
    trade_timestamp timestamp with time zone NOT NULL,
    provider_available_at timestamp with time zone,
    received_at timestamp with time zone NOT NULL,
    price numeric(30,10) NOT NULL,
    quantity numeric(30,10) NOT NULL,
    first_trade_id bigint NOT NULL,
    last_trade_id bigint NOT NULL,
    buyer_maker boolean NOT NULL,
    best_match boolean NOT NULL,
    payload_sha256 character(64) NOT NULL,
    strategy_key text DEFAULT 'binance_spot_btcusdt_aggregate_trades'::text NOT NULL,
    capture_artifact_id uuid NOT NULL,
    ingested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT chk_market_data_binance_spot_aggregate_trade_ids CHECK (((aggregate_trade_id >= 0) AND (first_trade_id >= 0) AND (last_trade_id >= first_trade_id))),
    CONSTRAINT chk_market_data_binance_spot_aggregate_trade_payload CHECK ((payload_sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT chk_market_data_binance_spot_aggregate_trade_source CHECK (((source = 'binance_spot'::text) AND (symbol = 'BTCUSDT'::text) AND (strategy_key = 'binance_spot_btcusdt_aggregate_trades'::text))),
    CONSTRAINT chk_market_data_binance_spot_aggregate_trade_values CHECK (((price > (0)::numeric) AND (quantity > (0)::numeric)))
);


--
-- Name: binance_spot_btcusdt_one_second_ohlcv; Type: TABLE; Schema: market_data; Owner: -
--

CREATE TABLE market_data.binance_spot_btcusdt_one_second_ohlcv (
    source text DEFAULT 'binance_spot'::text NOT NULL,
    symbol text NOT NULL,
    open_timestamp timestamp with time zone NOT NULL,
    close_timestamp timestamp with time zone NOT NULL,
    provider_available_at timestamp with time zone,
    received_at timestamp with time zone NOT NULL,
    open_price numeric(30,10) NOT NULL,
    high_price numeric(30,10) NOT NULL,
    low_price numeric(30,10) NOT NULL,
    close_price numeric(30,10) NOT NULL,
    base_volume numeric(30,10) NOT NULL,
    quote_volume numeric(30,10) NOT NULL,
    trade_count bigint NOT NULL,
    taker_buy_base_volume numeric(30,10) NOT NULL,
    taker_buy_quote_volume numeric(30,10) NOT NULL,
    payload_sha256 character(64) NOT NULL,
    strategy_key text DEFAULT 'binance_spot_btcusdt_one_second_ohlcv'::text NOT NULL,
    capture_artifact_id uuid NOT NULL,
    ingested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT chk_market_data_binance_spot_one_second_ohlcv_payload CHECK ((payload_sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT chk_market_data_binance_spot_one_second_ohlcv_prices CHECK (((open_price > (0)::numeric) AND (high_price > (0)::numeric) AND (low_price > (0)::numeric) AND (close_price > (0)::numeric) AND (high_price >= open_price) AND (high_price >= close_price) AND (high_price >= low_price) AND (low_price <= open_price) AND (low_price <= close_price))),
    CONSTRAINT chk_market_data_binance_spot_one_second_ohlcv_source CHECK (((source = 'binance_spot'::text) AND (symbol = 'BTCUSDT'::text) AND (strategy_key = 'binance_spot_btcusdt_one_second_ohlcv'::text))),
    CONSTRAINT chk_market_data_binance_spot_one_second_ohlcv_volume CHECK (((base_volume >= (0)::numeric) AND (quote_volume >= (0)::numeric) AND (trade_count >= 0) AND (taker_buy_base_volume >= (0)::numeric) AND (taker_buy_quote_volume >= (0)::numeric) AND (taker_buy_base_volume <= base_volume) AND (taker_buy_quote_volume <= quote_volume))),
    CONSTRAINT chk_market_data_binance_spot_one_second_ohlcv_window CHECK (((date_trunc('second'::text, open_timestamp) = open_timestamp) AND (close_timestamp = (open_timestamp + '00:00:00.999'::interval))))
);


--
-- Name: binance_spot_btcusdt_l2_snapshots; Type: TABLE; Schema: market_data; Owner: -
--

CREATE TABLE market_data.binance_spot_btcusdt_l2_snapshots (
    source_timestamp timestamp with time zone NOT NULL,
    received_at timestamp with time zone NOT NULL,
    source text DEFAULT 'binance_spot'::text NOT NULL,
    symbol text NOT NULL,
    source_update_id bigint NOT NULL,
    connection_epoch uuid NOT NULL,
    sample_depth integer NOT NULL,
    bids jsonb NOT NULL,
    asks jsonb NOT NULL,
    book_sha256 text NOT NULL,
    sampling_policy jsonb NOT NULL,
    sampling_policy_sha256 text NOT NULL,
    payload_sha256 text NOT NULL,
    strategy_key text DEFAULT 'binance_spot_btcusdt_l2_snapshots'::text NOT NULL,
    capture_artifact_id uuid NOT NULL,
    ingested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT chk_market_data_binance_spot_l2_snapshot_book CHECK (((jsonb_typeof(bids) = 'array'::text) AND (jsonb_typeof(asks) = 'array'::text) AND (jsonb_array_length(bids) = sample_depth) AND (jsonb_array_length(asks) = sample_depth) AND (octet_length((bids)::text) <= 262144) AND (octet_length((asks)::text) <= 262144) AND market_data.is_valid_binance_spot_btcusdt_l2_book(bids, asks, sample_depth))),
    CONSTRAINT chk_market_data_binance_spot_l2_snapshot_hashes CHECK (((book_sha256 ~ '^[0-9a-f]{64}$'::text) AND (sampling_policy_sha256 ~ '^[0-9a-f]{64}$'::text) AND (payload_sha256 ~ '^[0-9a-f]{64}$'::text))),
    CONSTRAINT chk_market_data_binance_spot_l2_snapshot_identity CHECK (((source_update_id >= 0) AND ((sample_depth >= 1) AND (sample_depth <= 1000)))),
    CONSTRAINT chk_market_data_binance_spot_l2_snapshot_sampling CHECK (((jsonb_typeof(sampling_policy) = 'object'::text) AND (octet_length((sampling_policy)::text) <= 4096) AND ((sampling_policy ->> 'version'::text) = 'binance-spot-btcusdt-l2-top-n-v1'::text) AND ((sampling_policy ->> 'source'::text) = 'binance_spot_diff_depth'::text) AND ((sampling_policy ->> 'symbol'::text) = 'BTCUSDT'::text) AND (jsonb_typeof((sampling_policy -> 'sample_depth'::text)) = 'number'::text) AND (((sampling_policy ->> 'sample_depth'::text))::integer = sample_depth) AND ((sampling_policy ->> 'selection'::text) = 'latest_contiguous_update_at_or_after_interval'::text))),
    CONSTRAINT chk_market_data_binance_spot_l2_snapshot_source CHECK (((source = 'binance_spot'::text) AND (symbol = 'BTCUSDT'::text) AND (strategy_key = 'binance_spot_btcusdt_l2_snapshots'::text)))
);


--
-- Name: binance_futures_btcusdt_open_interest; Type: TABLE; Schema: market_data; Owner: -
--

CREATE TABLE market_data.binance_futures_btcusdt_open_interest (
    source text DEFAULT 'binance_usd_m_futures'::text NOT NULL,
    source_timestamp timestamp with time zone NOT NULL,
    symbol text NOT NULL,
    period_seconds integer NOT NULL,
    sum_open_interest numeric(38,18) NOT NULL,
    sum_open_interest_value numeric(38,18) NOT NULL,
    cmc_circulating_supply numeric(38,18),
    provider_available_at timestamp with time zone,
    received_at timestamp with time zone NOT NULL,
    source_payload jsonb NOT NULL,
    payload_sha256 character(64) NOT NULL,
    strategy_key text DEFAULT 'binance_futures_btcusdt_open_interest'::text NOT NULL,
    capture_artifact_id uuid NOT NULL,
    ingested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT chk_market_data_binance_futures_open_interest_payload CHECK (((jsonb_typeof(source_payload) = 'object'::text) AND (octet_length((source_payload)::text) <= 16384) AND (payload_sha256 ~ '^[0-9a-f]{64}$'::text))),
    CONSTRAINT chk_market_data_binance_futures_open_interest_period CHECK (((period_seconds = 300) AND (((EXTRACT(epoch FROM source_timestamp))::bigint % (300)::bigint) = 0))),
    CONSTRAINT chk_market_data_binance_futures_open_interest_source CHECK (((source = 'binance_usd_m_futures'::text) AND (symbol = 'BTCUSDT'::text) AND (strategy_key = 'binance_futures_btcusdt_open_interest'::text))),
    CONSTRAINT chk_market_data_binance_futures_open_interest_values CHECK (((sum_open_interest >= (0)::numeric) AND (sum_open_interest_value >= (0)::numeric) AND ((cmc_circulating_supply IS NULL) OR (cmc_circulating_supply >= (0)::numeric))))
);


--
-- Name: chainlink_btcusd_one_minute_candles; Type: TABLE; Schema: market_data; Owner: -
--

CREATE TABLE market_data.chainlink_btcusd_one_minute_candles (
    source text DEFAULT 'chainlink_candlestick'::text NOT NULL,
    symbol text NOT NULL,
    open_timestamp timestamp with time zone NOT NULL,
    close_timestamp timestamp with time zone NOT NULL,
    provider_available_at timestamp with time zone,
    received_at timestamp with time zone NOT NULL,
    open_price numeric(38,18) NOT NULL,
    high_price numeric(38,18) NOT NULL,
    low_price numeric(38,18) NOT NULL,
    close_price numeric(38,18) NOT NULL,
    volume numeric(38,18),
    volume_supported boolean DEFAULT false NOT NULL,
    payload_sha256 character(64) NOT NULL,
    strategy_key text DEFAULT 'chainlink_btcusd_one_minute_ohlc'::text NOT NULL,
    capture_artifact_id uuid NOT NULL,
    ingested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT chk_market_data_chainlink_btcusd_one_minute_candles_hash CHECK ((payload_sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT chk_market_data_chainlink_btcusd_one_minute_candles_identity CHECK (((source = 'chainlink_candlestick'::text) AND (symbol = 'BTCUSD'::text) AND (strategy_key = 'chainlink_btcusd_one_minute_ohlc'::text))),
    CONSTRAINT chk_market_data_chainlink_btcusd_one_minute_candles_prices CHECK (((open_price > (0)::numeric) AND (high_price > (0)::numeric) AND (low_price > (0)::numeric) AND (close_price > (0)::numeric) AND (high_price >= open_price) AND (high_price >= close_price) AND (high_price >= low_price) AND (low_price <= open_price) AND (low_price <= close_price))),
    CONSTRAINT chk_market_data_chainlink_btcusd_one_minute_candles_volume CHECK (((volume IS NULL) AND (volume_supported = false))),
    CONSTRAINT chk_market_data_chainlink_btcusd_one_minute_candles_window CHECK (((date_trunc('minute'::text, open_timestamp) = open_timestamp) AND (close_timestamp = (open_timestamp + '00:01:00'::interval))))
);


--
-- Name: polygon_chainlink_btcusd_oracle_rounds; Type: TABLE; Schema: market_data; Owner: -
--

CREATE TABLE market_data.polygon_chainlink_btcusd_oracle_rounds (
    source text DEFAULT 'chainlink_polygon_data_feed'::text NOT NULL,
    chain_id bigint NOT NULL,
    feed_proxy_address text NOT NULL,
    aggregator_address text NOT NULL,
    phase_id integer NOT NULL,
    aggregator_round_id bigint NOT NULL,
    source_timestamp timestamp with time zone NOT NULL,
    block_timestamp timestamp with time zone NOT NULL,
    answer_raw numeric(38,0) NOT NULL,
    price numeric(38,18) NOT NULL,
    decimals integer NOT NULL,
    block_number bigint NOT NULL,
    block_hash text NOT NULL,
    transaction_hash text NOT NULL,
    log_index integer NOT NULL,
    provider_available_at timestamp with time zone NOT NULL,
    received_at timestamp with time zone NOT NULL,
    source_payload jsonb NOT NULL,
    payload_sha256 character(64) NOT NULL,
    strategy_key text DEFAULT 'polygon_chainlink_btcusd_oracle'::text NOT NULL,
    capture_artifact_id uuid NOT NULL,
    ingested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT chk_market_data_polygon_chainlink_btcusd_oracle_identity CHECK (((source = 'chainlink_polygon_data_feed'::text) AND (chain_id = 137) AND (feed_proxy_address = '0xc907e116054ad103354f2d350fd2514433d57f6f'::text) AND (strategy_key = 'polygon_chainlink_btcusd_oracle'::text) AND (feed_proxy_address ~ '^0x[0-9a-f]{40}$'::text) AND (aggregator_address ~ '^0x[0-9a-f]{40}$'::text) AND (block_hash ~ '^0x[0-9a-f]{64}$'::text) AND (transaction_hash ~ '^0x[0-9a-f]{64}$'::text))),
    CONSTRAINT chk_market_data_polygon_chainlink_btcusd_oracle_payload CHECK (((jsonb_typeof(source_payload) = 'object'::text) AND (octet_length((source_payload)::text) <= 32768) AND (payload_sha256 ~ '^[0-9a-f]{64}$'::text))),
    CONSTRAINT chk_market_data_polygon_chainlink_btcusd_oracle_round CHECK (((phase_id > 0) AND (aggregator_round_id > 0) AND (block_number >= 0) AND (log_index >= 0) AND ((decimals >= 0) AND (decimals <= 18)) AND (answer_raw > (0)::numeric) AND (price > (0)::numeric))),
    CONSTRAINT chk_market_data_polygon_chainlink_btcusd_oracle_time CHECK (((source_timestamp <= block_timestamp) AND (provider_available_at = block_timestamp)))
);


--
-- Name: btc_five_minute_orderbook_snapshots; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.btc_five_minute_orderbook_snapshots (
    sampled_at timestamp with time zone NOT NULL,
    source_timestamp timestamp with time zone NOT NULL,
    provider_available_at timestamp with time zone NOT NULL,
    received_at timestamp with time zone NOT NULL,
    source text DEFAULT 'polymarket_clob_market'::text NOT NULL,
    market_id text NOT NULL,
    condition_id text NOT NULL,
    event_slug text NOT NULL,
    window_start timestamp with time zone NOT NULL,
    window_end timestamp with time zone NOT NULL,
    token_id text NOT NULL,
    outcome text NOT NULL,
    connection_epoch uuid NOT NULL,
    ingest_sequence bigint NOT NULL,
    tick_size numeric(18,8) NOT NULL,
    best_bid numeric(18,8),
    best_ask numeric(18,8),
    bid_depth integer NOT NULL,
    ask_depth integer NOT NULL,
    bids jsonb NOT NULL,
    asks jsonb NOT NULL,
    source_hash text,
    book_sha256 character(64) NOT NULL,
    sampling_policy jsonb NOT NULL,
    sampling_policy_sha256 character(64) NOT NULL,
    payload_sha256 character(64) NOT NULL,
    strategy_key text DEFAULT 'polymarket_btc_five_minute_orderbooks'::text NOT NULL,
    capture_artifact_id uuid NOT NULL,
    ingested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT chk_market_data_polymarket_btc_five_minute_orderbook_identity CHECK (((source = 'polymarket_clob_market'::text) AND (strategy_key = 'polymarket_btc_five_minute_orderbooks'::text) AND ((octet_length(market_id) >= 1) AND (octet_length(market_id) <= 256)) AND (condition_id ~ '^0x[0-9a-f]{64}$'::text) AND (token_id ~ '^[0-9]{1,100}$'::text) AND (outcome = ANY (ARRAY['up'::text, 'down'::text])) AND (ingest_sequence >= 0))),
    CONSTRAINT chk_market_data_polymarket_btc_five_minute_orderbook_payload CHECK ((((source_hash IS NULL) OR ((octet_length(source_hash) >= 1) AND (octet_length(source_hash) <= 256))) AND (book_sha256 ~ '^[0-9a-f]{64}$'::text) AND (sampling_policy_sha256 ~ '^[0-9a-f]{64}$'::text) AND (payload_sha256 ~ '^[0-9a-f]{64}$'::text))),
    CONSTRAINT chk_market_data_polymarket_btc_five_minute_orderbook_sampling CHECK (((jsonb_typeof(sampling_policy) = 'object'::text) AND (octet_length((sampling_policy)::text) <= 4096) AND ((sampling_policy ->> 'version'::text) = 'polymarket-clob-btc-5m-orderbook-top-n-v1'::text) AND ((sampling_policy ->> 'source'::text) = 'polymarket_clob_market'::text) AND ((sampling_policy ->> 'selection'::text) = 'latest_valid_subscribed_market_book_at_aligned_wall_clock_slot'::text) AND (jsonb_typeof((sampling_policy -> 'market_interval_seconds'::text)) = 'number'::text) AND (((sampling_policy ->> 'market_interval_seconds'::text))::integer = 300) AND (jsonb_typeof((sampling_policy -> 'top_n'::text)) = 'number'::text) AND ((((sampling_policy ->> 'top_n'::text))::integer >= 1) AND (((sampling_policy ->> 'top_n'::text))::integer <= 1000)) AND (jsonb_typeof((sampling_policy -> 'sample_interval_ms'::text)) = 'number'::text) AND ((((sampling_policy ->> 'sample_interval_ms'::text))::integer >= 100) AND (((sampling_policy ->> 'sample_interval_ms'::text))::integer <= 60000)))),
    CONSTRAINT chk_market_data_polymarket_btc_five_minute_orderbook_window CHECK (((window_end = (window_start + '00:05:00'::interval)) AND (mod((EXTRACT(epoch FROM window_start))::bigint, (300)::bigint) = 0) AND (event_slug = ('btc-updown-5m-'::text || ((EXTRACT(epoch FROM window_start))::bigint)::text))))
);


--
-- Name: backfill_artifacts; Type: TABLE; Schema: ingester; Owner: -
--

CREATE TABLE ingester.backfill_artifacts (
    artifact_id uuid DEFAULT gen_random_uuid() NOT NULL,
    job_id uuid NOT NULL,
    ingester_key text,
    logical_key text NOT NULL,
    provider text NOT NULL,
    source_uri text NOT NULL,
    source_date date,
    checksum_algorithm text DEFAULT 'sha256'::text NOT NULL,
    expected_checksum text,
    actual_checksum text,
    compressed_bytes bigint,
    record_count bigint,
    minimum_source_timestamp timestamp with time zone,
    maximum_source_timestamp timestamp with time zone,
    status text DEFAULT 'pending'::text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    completed_at timestamp with time zone,
    strategy_key text NOT NULL,
    checksum text,
    byte_size bigint,
    durable_target text,
    legacy_source text,
    legacy_artifact_id uuid,
    CONSTRAINT chk_ingester_backfill_artifact_checksum CHECK (((checksum_algorithm = 'sha256'::text) AND ((checksum IS NULL) OR (checksum ~ '^[0-9a-f]{64}$'::text)))),
    CONSTRAINT chk_ingester_backfill_artifact_counts CHECK ((((byte_size IS NULL) OR (byte_size >= 0)) AND ((record_count IS NULL) OR (record_count >= 0)))),
    CONSTRAINT chk_poly_backfill_artifacts_checksum_algorithm CHECK ((checksum_algorithm = 'sha256'::text)),
    CONSTRAINT chk_poly_backfill_artifacts_metadata CHECK ((jsonb_typeof(metadata) = 'object'::text)),
    CONSTRAINT chk_poly_backfill_artifacts_status CHECK ((status = ANY (ARRAY['pending'::text, 'downloading'::text, 'downloaded'::text, 'verified'::text, 'ingesting'::text, 'completed'::text, 'failed'::text])))
);


--
-- Name: backfill_job_events; Type: TABLE; Schema: ingester; Owner: -
--

CREATE TABLE ingester.backfill_job_events (
    event_id uuid DEFAULT gen_random_uuid() NOT NULL,
    job_id uuid NOT NULL,
    recorded_at timestamp with time zone DEFAULT now() NOT NULL,
    level text NOT NULL,
    event_code text NOT NULL,
    message text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    CONSTRAINT chk_ingester_backfill_event_level CHECK ((level = ANY (ARRAY['debug'::text, 'info'::text, 'warn'::text, 'error'::text]))),
    CONSTRAINT chk_ingester_backfill_event_metadata CHECK ((jsonb_typeof(metadata) = 'object'::text))
);


--
-- Name: backfill_jobs; Type: TABLE; Schema: ingester; Owner: -
--

CREATE TABLE ingester.backfill_jobs (
    job_id uuid DEFAULT gen_random_uuid() NOT NULL,
    parent_job_id uuid,
    job_kind text NOT NULL,
    strategy_key text NOT NULL,
    strategy_contract_version integer NOT NULL,
    request_schema_version integer NOT NULL,
    canonical_request jsonb NOT NULL,
    request_hash text NOT NULL,
    range_start timestamp with time zone NOT NULL,
    range_end timestamp with time zone NOT NULL,
    shard_key text,
    status text DEFAULT 'queued'::text NOT NULL,
    attempt integer DEFAULT 0 NOT NULL,
    max_attempts integer DEFAULT 3 NOT NULL,
    next_attempt_at timestamp with time zone DEFAULT now() NOT NULL,
    assigned_worker_id text,
    required_worker_id text,
    required_deployment text,
    lease_token uuid,
    lease_expires_at timestamp with time zone,
    heartbeat_at timestamp with time zone,
    progress jsonb DEFAULT '{}'::jsonb NOT NULL,
    checkpoint jsonb DEFAULT '{}'::jsonb NOT NULL,
    verified_coverage jsonb DEFAULT '{}'::jsonb NOT NULL,
    summary jsonb DEFAULT '{}'::jsonb NOT NULL,
    last_error_kind text,
    last_error_code text,
    last_error_message text,
    requested_at timestamp with time zone DEFAULT now() NOT NULL,
    started_at timestamp with time zone,
    completed_at timestamp with time zone,
    cancel_requested_at timestamp with time zone,
    assigned_worker_image_digest text,
    assigned_worker_source_revision text,
    legacy_source text,
    legacy_job_id uuid,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    allocation_units integer DEFAULT 2 NOT NULL,
    CONSTRAINT chk_ingester_backfill_allocation_units CHECK (((allocation_units >= 1) AND (allocation_units <= 32))),
    CONSTRAINT chk_ingester_backfill_assignment CHECK ((((job_kind = 'shard'::text) AND (status = ANY (ARRAY['running'::text, 'cancel_requested'::text])) AND (assigned_worker_id IS NOT NULL) AND (lease_token IS NOT NULL) AND (lease_expires_at IS NOT NULL) AND (heartbeat_at IS NOT NULL)) OR (job_kind = 'request'::text) OR (status <> ALL (ARRAY['running'::text, 'cancel_requested'::text])) OR (legacy_source IS NOT NULL))),
    CONSTRAINT chk_ingester_backfill_attempts CHECK (((attempt >= 0) AND (max_attempts > 0) AND (attempt <= max_attempts))),
    CONSTRAINT chk_ingester_backfill_documents CHECK (((jsonb_typeof(progress) = 'object'::text) AND (jsonb_typeof(checkpoint) = 'object'::text) AND (jsonb_typeof(verified_coverage) = 'object'::text) AND (jsonb_typeof(summary) = 'object'::text))),
    CONSTRAINT chk_ingester_backfill_error_kind CHECK (((last_error_kind IS NULL) OR (last_error_kind = ANY (ARRAY['transient_source'::text, 'transient_database'::text, 'rate_limited'::text, 'invalid_request'::text, 'integrity'::text, 'lease_lost'::text, 'cancelled'::text])))),
    CONSTRAINT chk_ingester_backfill_hash CHECK ((request_hash ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT chk_ingester_backfill_kind CHECK ((job_kind = ANY (ARRAY['request'::text, 'shard'::text]))),
    CONSTRAINT chk_ingester_backfill_parent CHECK ((((job_kind = 'request'::text) AND (parent_job_id IS NULL) AND (shard_key IS NULL)) OR ((job_kind = 'shard'::text) AND (parent_job_id IS NOT NULL) AND (shard_key IS NOT NULL)))),
    CONSTRAINT chk_ingester_backfill_range CHECK ((range_start < range_end)),
    CONSTRAINT chk_ingester_backfill_request CHECK ((jsonb_typeof(canonical_request) = 'object'::text)),
    CONSTRAINT chk_ingester_backfill_status CHECK ((status = ANY (ARRAY['queued'::text, 'running'::text, 'cancel_requested'::text, 'completed'::text, 'failed'::text, 'cancelled'::text]))),
    CONSTRAINT chk_ingester_backfill_strategy CHECK ((length(btrim(strategy_key)) > 0)),
    CONSTRAINT chk_ingester_backfill_versions CHECK (((strategy_contract_version > 0) AND (request_schema_version > 0)))
);


--
-- Name: capture_artifacts; Type: TABLE; Schema: ingester; Owner: -
--

CREATE TABLE ingester.capture_artifacts (
    artifact_id uuid DEFAULT gen_random_uuid() NOT NULL,
    strategy_key text NOT NULL,
    profile_generation bigint NOT NULL,
    config_schema_version integer NOT NULL,
    config_sha256 text NOT NULL,
    config_snapshot jsonb NOT NULL,
    capture_window_start timestamp with time zone NOT NULL,
    capture_window_end timestamp with time zone NOT NULL,
    minimum_source_timestamp timestamp with time zone,
    maximum_source_timestamp timestamp with time zone,
    minimum_received_at timestamp with time zone,
    maximum_received_at timestamp with time zone,
    start_cursor text,
    end_cursor text,
    record_count bigint DEFAULT 0 NOT NULL,
    content_sha256 text,
    status text DEFAULT 'open'::text NOT NULL,
    failure_code text,
    failure_message text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    completed_at timestamp with time zone,
    CONSTRAINT chk_ingester_capture_artifact_completion CHECK ((((status = 'open'::text) AND (content_sha256 IS NULL) AND (completed_at IS NULL) AND (failure_code IS NULL) AND (failure_message IS NULL)) OR ((status = 'completed'::text) AND (content_sha256 ~ '^[0-9a-f]{64}$'::text) AND (completed_at IS NOT NULL) AND (failure_code IS NULL) AND (failure_message IS NULL)) OR ((status = 'failed'::text) AND (completed_at IS NOT NULL) AND ((length(btrim(failure_code)) >= 1) AND (length(btrim(failure_code)) <= 128)) AND ((failure_message IS NULL) OR (octet_length(failure_message) <= 2048))))),
    CONSTRAINT chk_ingester_capture_artifact_config CHECK (((config_sha256 ~ '^[0-9a-f]{64}$'::text) AND (jsonb_typeof(config_snapshot) = 'object'::text) AND (octet_length((config_snapshot)::text) <= 16384))),
    CONSTRAINT chk_ingester_capture_artifact_content_hash CHECK (((content_sha256 IS NULL) OR (content_sha256 ~ '^[0-9a-f]{64}$'::text))),
    CONSTRAINT chk_ingester_capture_artifact_counts CHECK ((record_count >= 0)),
    CONSTRAINT chk_ingester_capture_artifact_cursors CHECK ((((start_cursor IS NULL) OR (octet_length(start_cursor) <= 2048)) AND ((end_cursor IS NULL) OR (octet_length(end_cursor) <= 2048)))),
    CONSTRAINT chk_ingester_capture_artifact_generation CHECK (((profile_generation > 0) AND (config_schema_version > 0))),
    CONSTRAINT chk_ingester_capture_artifact_receipt_range CHECK ((((minimum_received_at IS NULL) AND (maximum_received_at IS NULL)) OR ((minimum_received_at IS NOT NULL) AND (maximum_received_at IS NOT NULL) AND (maximum_received_at >= minimum_received_at)))),
    CONSTRAINT chk_ingester_capture_artifact_source_range CHECK ((((minimum_source_timestamp IS NULL) AND (maximum_source_timestamp IS NULL)) OR ((minimum_source_timestamp IS NOT NULL) AND (maximum_source_timestamp IS NOT NULL) AND (maximum_source_timestamp >= minimum_source_timestamp)))),
    CONSTRAINT chk_ingester_capture_artifact_status CHECK ((status = ANY (ARRAY['open'::text, 'completed'::text, 'failed'::text]))),
    CONSTRAINT chk_ingester_capture_artifact_times CHECK (((updated_at >= created_at) AND ((completed_at IS NULL) OR (completed_at >= created_at)))),
    CONSTRAINT chk_ingester_capture_artifact_window CHECK (((capture_window_end > capture_window_start) AND (capture_window_end <= (capture_window_start + '7 days'::interval))))
);


--
-- Name: data_gaps; Type: TABLE; Schema: ingester; Owner: -
--

CREATE TABLE ingester.data_gaps (
    gap_id uuid DEFAULT gen_random_uuid() NOT NULL,
    gap_fingerprint text NOT NULL,
    strategy_key text NOT NULL,
    detected_artifact_id uuid,
    repair_artifact_id uuid,
    gap_kind text NOT NULL,
    reason_code text NOT NULL,
    reason_message text,
    source_time_start timestamp with time zone,
    source_time_end timestamp with time zone,
    start_cursor text,
    end_cursor text,
    status text DEFAULT 'open'::text NOT NULL,
    repair_attempts integer DEFAULT 0 NOT NULL,
    detected_at timestamp with time zone DEFAULT now() NOT NULL,
    repair_started_at timestamp with time zone,
    resolved_at timestamp with time zone,
    resolution_code text,
    resolution_message text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_ingester_data_gap_classification CHECK ((((length(btrim(gap_kind)) >= 1) AND (length(btrim(gap_kind)) <= 64)) AND ((length(btrim(reason_code)) >= 1) AND (length(btrim(reason_code)) <= 128)) AND ((reason_message IS NULL) OR (octet_length(reason_message) <= 2048)))),
    CONSTRAINT chk_ingester_data_gap_cursors CHECK ((((start_cursor IS NULL) OR (octet_length(start_cursor) <= 2048)) AND ((end_cursor IS NULL) OR (octet_length(end_cursor) <= 2048)) AND ((source_time_start IS NOT NULL) OR (start_cursor IS NOT NULL) OR (end_cursor IS NOT NULL)))),
    CONSTRAINT chk_ingester_data_gap_fingerprint CHECK ((gap_fingerprint ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT chk_ingester_data_gap_resolution CHECK ((((status = 'open'::text) AND (repair_artifact_id IS NULL) AND (repair_started_at IS NULL) AND (resolved_at IS NULL) AND (resolution_code IS NULL) AND (resolution_message IS NULL)) OR ((status = 'repairing'::text) AND (repair_started_at IS NOT NULL) AND (repair_attempts > 0) AND (resolved_at IS NULL) AND (resolution_code IS NULL) AND (resolution_message IS NULL)) OR ((status = 'repaired'::text) AND (repair_artifact_id IS NOT NULL) AND (repair_attempts > 0) AND (resolved_at IS NOT NULL) AND ((length(btrim(resolution_code)) >= 1) AND (length(btrim(resolution_code)) <= 128)) AND ((resolution_message IS NULL) OR (octet_length(resolution_message) <= 2048))) OR ((status = 'unrecoverable'::text) AND (resolved_at IS NOT NULL) AND ((length(btrim(resolution_code)) >= 1) AND (length(btrim(resolution_code)) <= 128)) AND ((resolution_message IS NULL) OR (octet_length(resolution_message) <= 2048))))),
    CONSTRAINT chk_ingester_data_gap_source_time CHECK ((((source_time_start IS NULL) AND (source_time_end IS NULL)) OR ((source_time_start IS NOT NULL) AND (source_time_end IS NOT NULL) AND (source_time_end >= source_time_start)))),
    CONSTRAINT chk_ingester_data_gap_status CHECK (((status = ANY (ARRAY['open'::text, 'repairing'::text, 'repaired'::text, 'unrecoverable'::text])) AND (repair_attempts >= 0))),
    CONSTRAINT chk_ingester_data_gap_times CHECK (((updated_at >= created_at) AND ((repair_started_at IS NULL) OR (repair_started_at >= detected_at)) AND ((resolved_at IS NULL) OR (resolved_at >= COALESCE(repair_started_at, detected_at)))))
);


--
-- Name: drain_jobs; Type: TABLE; Schema: ingester; Owner: -
--

CREATE TABLE ingester.drain_jobs (
    job_id uuid DEFAULT gen_random_uuid() NOT NULL,
    strategy_key text NOT NULL,
    strategy_contract_version integer NOT NULL,
    cutoff timestamp with time zone NOT NULL,
    dry_run boolean DEFAULT false NOT NULL,
    status text DEFAULT 'queued'::text NOT NULL,
    required_worker_id text,
    required_deployment text,
    assigned_worker_id text,
    lease_token uuid,
    lease_expires_at timestamp with time zone,
    attempt integer DEFAULT 0 NOT NULL,
    max_attempts integer DEFAULT 5 NOT NULL,
    rows_exported bigint DEFAULT 0 NOT NULL,
    rows_removed bigint DEFAULT 0 NOT NULL,
    objects_published bigint DEFAULT 0 NOT NULL,
    bytes_written bigint DEFAULT 0 NOT NULL,
    summary jsonb DEFAULT '{}'::jsonb NOT NULL,
    last_error_code text,
    last_error_message text,
    requested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    started_at timestamp with time zone,
    completed_at timestamp with time zone,
    cancel_requested_at timestamp with time zone,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    mode text DEFAULT 'drain'::text NOT NULL,
    CONSTRAINT chk_ingester_drain_job_contract CHECK (((strategy_contract_version > 0) AND (attempt >= 0) AND (max_attempts > 0))),
    CONSTRAINT chk_ingester_drain_job_selector CHECK (((required_worker_id IS NULL) OR (required_deployment IS NULL))),
    CONSTRAINT chk_ingester_drain_job_status CHECK ((status = ANY (ARRAY['queued'::text, 'running'::text, 'completed'::text, 'failed'::text, 'cancelled'::text]))),
    CONSTRAINT ck_ingester_drain_jobs_mode CHECK ((mode = ANY (ARRAY['drain'::text, 'reconcile'::text])))
);


--
-- Name: drain_objects; Type: TABLE; Schema: ingester; Owner: -
--

CREATE TABLE ingester.drain_objects (
    object_id uuid DEFAULT gen_random_uuid() NOT NULL,
    job_id uuid NOT NULL,
    strategy_key text NOT NULL,
    source_relation text NOT NULL,
    source_chunk_schema text NOT NULL,
    source_chunk_name text NOT NULL,
    source_start timestamp with time zone NOT NULL,
    source_end timestamp with time zone NOT NULL,
    row_count bigint,
    minimum_aggregate_trade_id bigint,
    maximum_aggregate_trade_id bigint,
    relative_path text,
    sha256 character(64),
    byte_size bigint,
    status text DEFAULT 'staging'::text NOT NULL,
    published_at timestamp with time zone,
    removed_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT chk_ingester_drain_object_publication CHECK (((status = 'staging'::text) OR ((row_count >= 0) AND (relative_path IS NOT NULL) AND (sha256 ~ '^[0-9a-f]{64}$'::text) AND (byte_size > 0) AND (published_at IS NOT NULL)))),
    CONSTRAINT chk_ingester_drain_object_range CHECK ((source_end > source_start)),
    CONSTRAINT chk_ingester_drain_object_status CHECK ((status = ANY (ARRAY['staging'::text, 'published'::text, 'removed'::text])))
);


--
-- Name: profiles; Type: TABLE; Schema: ingester; Owner: -
--

CREATE TABLE ingester.profiles (
    strategy_key text NOT NULL,
    config_schema_version integer NOT NULL,
    config jsonb NOT NULL,
    desired_state text DEFAULT 'stopped'::text NOT NULL,
    desired_generation bigint DEFAULT 1 NOT NULL,
    observed_state text DEFAULT 'stopped'::text NOT NULL,
    health_status text DEFAULT 'unknown'::text NOT NULL,
    applied_generation bigint,
    checkpoint_schema_version integer DEFAULT 1 NOT NULL,
    checkpoint jsonb DEFAULT '{}'::jsonb NOT NULL,
    lease_owner text,
    lease_token uuid,
    lease_expires_at timestamp with time zone,
    heartbeat_at timestamp with time zone,
    started_at timestamp with time zone,
    stopped_at timestamp with time zone,
    last_source_event_at timestamp with time zone,
    last_provider_available_at timestamp with time zone,
    last_persisted_at timestamp with time zone,
    source_watermark timestamp with time zone,
    availability_watermark timestamp with time zone,
    consecutive_failures integer DEFAULT 0 NOT NULL,
    restart_count bigint DEFAULT 0 NOT NULL,
    last_error_code text,
    last_error_message text,
    last_error_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_ingester_profiles_checkpoint CHECK (((checkpoint_schema_version > 0) AND (jsonb_typeof(checkpoint) = 'object'::text) AND (octet_length((checkpoint)::text) <= 8192))),
    CONSTRAINT chk_ingester_profiles_config CHECK (((config_schema_version > 0) AND (jsonb_typeof(config) = 'object'::text) AND (octet_length((config)::text) <= 16384))),
    CONSTRAINT chk_ingester_profiles_counters CHECK (((consecutive_failures >= 0) AND (restart_count >= 0))),
    CONSTRAINT chk_ingester_profiles_desired_state CHECK ((desired_state = ANY (ARRAY['running'::text, 'stopped'::text]))),
    CONSTRAINT chk_ingester_profiles_error CHECK ((((last_error_code IS NULL) OR ((length(btrim(last_error_code)) >= 1) AND (length(btrim(last_error_code)) <= 128))) AND ((last_error_message IS NULL) OR (octet_length(last_error_message) <= 2048)) AND (((last_error_code IS NULL) AND (last_error_message IS NULL) AND (last_error_at IS NULL)) OR ((last_error_code IS NOT NULL) AND (last_error_at IS NOT NULL))))),
    CONSTRAINT chk_ingester_profiles_generations CHECK (((desired_generation > 0) AND ((applied_generation IS NULL) OR ((applied_generation > 0) AND (applied_generation <= desired_generation))))),
    CONSTRAINT chk_ingester_profiles_health CHECK ((health_status = ANY (ARRAY['unknown'::text, 'healthy'::text, 'degraded'::text, 'unhealthy'::text]))),
    CONSTRAINT chk_ingester_profiles_lease CHECK ((((lease_owner IS NULL) AND (lease_token IS NULL) AND (lease_expires_at IS NULL)) OR (((length(btrim(lease_owner)) >= 1) AND (length(btrim(lease_owner)) <= 128)) AND (lease_token IS NOT NULL) AND (lease_expires_at IS NOT NULL) AND (heartbeat_at IS NOT NULL) AND (lease_expires_at > heartbeat_at)))),
    CONSTRAINT chk_ingester_profiles_observed_state CHECK ((observed_state = ANY (ARRAY['starting'::text, 'running'::text, 'degraded'::text, 'restarting'::text, 'stopping'::text, 'stopped'::text, 'failed'::text, 'unsupported'::text]))),
    CONSTRAINT chk_ingester_profiles_strategy_key CHECK ((((length(btrim(strategy_key)) >= 1) AND (length(btrim(strategy_key)) <= 128)) AND (strategy_key ~ '^[a-z0-9_]+$'::text))),
    CONSTRAINT chk_ingester_profiles_times CHECK ((updated_at >= created_at))
);


--
-- Name: workers; Type: TABLE; Schema: ingester; Owner: -
--

CREATE TABLE ingester.workers (
    worker_id text NOT NULL,
    hostname text NOT NULL,
    worker_contract_version integer NOT NULL,
    supported_strategies jsonb NOT NULL,
    maximum_backfills integer NOT NULL,
    active_backfills integer DEFAULT 0 NOT NULL,
    realtime_strategies jsonb DEFAULT '[]'::jsonb NOT NULL,
    image_digest text NOT NULL,
    source_revision text NOT NULL,
    deployment_id text NOT NULL,
    lifecycle_state text DEFAULT 'active'::text NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    heartbeat_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    capacity_units integer DEFAULT 4 NOT NULL,
    realtime_slot_limit integer DEFAULT 1 NOT NULL,
    allocation_contract_version integer DEFAULT 1 NOT NULL,
    CONSTRAINT chk_ingester_worker_allocation_capacity CHECK (((capacity_units >= 1) AND (capacity_units <= 32))),
    CONSTRAINT chk_ingester_worker_allocation_contract CHECK ((allocation_contract_version > 0)),
    CONSTRAINT chk_ingester_worker_capabilities CHECK (((jsonb_typeof(supported_strategies) = 'object'::text) AND (jsonb_typeof(realtime_strategies) = 'array'::text))),
    CONSTRAINT chk_ingester_worker_capacity CHECK (((maximum_backfills > 0) AND (active_backfills >= 0) AND (active_backfills <= maximum_backfills))),
    CONSTRAINT chk_ingester_worker_contract CHECK ((worker_contract_version > 0)),
    CONSTRAINT chk_ingester_worker_identity CHECK (((length(btrim(worker_id)) > 0) AND (length(btrim(hostname)) > 0) AND (length(btrim(image_digest)) > 0) AND (length(btrim(source_revision)) > 0) AND (length(btrim(deployment_id)) > 0))),
    CONSTRAINT chk_ingester_worker_lifecycle CHECK ((lifecycle_state = ANY (ARRAY['active'::text, 'draining'::text]))),
    CONSTRAINT chk_ingester_worker_realtime_slots CHECK ((realtime_slot_limit = 1))
);


--
-- Name: chainlink_btcusd_reference_prices; Type: TABLE; Schema: market_data; Owner: -
--

CREATE TABLE market_data.chainlink_btcusd_reference_prices (
    source text NOT NULL,
    feed_id text NOT NULL,
    source_timestamp timestamp with time zone NOT NULL,
    valid_from_timestamp timestamp with time zone,
    provider_available_at timestamp with time zone,
    received_at timestamp with time zone NOT NULL,
    price numeric(38,18) NOT NULL,
    bid numeric(38,18),
    ask numeric(38,18),
    report_sha256 character(64) NOT NULL,
    payload_sha256 character(64) NOT NULL,
    strategy_key text NOT NULL,
    capture_artifact_id uuid,
    ingested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    expires_at timestamp with time zone,
    report_version text,
    source_date date,
    archive_row_number bigint,
    backfill_artifact_id uuid,
    report_hash_kind text NOT NULL,
    CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_hashes CHECK (((report_sha256 ~ '^[0-9a-f]{64}$'::text) AND (payload_sha256 ~ '^[0-9a-f]{64}$'::text) AND ((report_version IS NULL) OR (length(btrim(report_version)) > 0)))),
    CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_identity CHECK (((source = 'chainlink_data_streams'::text) AND (feed_id = '0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8'::text) AND (report_hash_kind = 'signed_report'::text) AND (((strategy_key = 'chainlink_btcusd_reference_price'::text) AND (capture_artifact_id IS NOT NULL) AND (backfill_artifact_id IS NULL)) OR ((strategy_key = 'chainlink_btcusd_reference_ticks_backfill'::text) AND (capture_artifact_id IS NULL) AND (backfill_artifact_id IS NOT NULL))))),
    CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_time CHECK ((((valid_from_timestamp IS NULL) OR (valid_from_timestamp <= source_timestamp)) AND ((expires_at IS NULL) OR (expires_at > source_timestamp)) AND ((source_date IS NULL) OR ((source_timestamp >= (source_date)::timestamp with time zone) AND (source_timestamp < ((source_date)::timestamp with time zone + '1 day'::interval)))))),
    CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_values CHECK (((price > (0)::numeric) AND (((bid IS NULL) AND (ask IS NULL)) OR ((bid > (0)::numeric) AND (bid <= price) AND (price <= ask)))))
);


--
-- Name: polymarket_btc_five_minute_contracts; Type: TABLE; Schema: market_data; Owner: -
--

CREATE TABLE market_data.polymarket_btc_five_minute_contracts (
    source text DEFAULT 'polymarket_gamma_rest'::text NOT NULL,
    event_id text NOT NULL,
    event_slug text NOT NULL,
    series_slug text NOT NULL,
    market_id text NOT NULL,
    condition_id text NOT NULL,
    window_start timestamp with time zone NOT NULL,
    window_end timestamp with time zone NOT NULL,
    up_token_id text NOT NULL,
    down_token_id text NOT NULL,
    tick_size numeric(18,8) NOT NULL,
    minimum_order_size numeric(38,18),
    resolution_source text NOT NULL,
    active boolean NOT NULL,
    closed boolean NOT NULL,
    accepting_orders boolean NOT NULL,
    fees_enabled boolean NOT NULL,
    fee_schedule jsonb NOT NULL,
    received_at timestamp with time zone NOT NULL,
    source_payload jsonb NOT NULL,
    revision_sha256 character(64) NOT NULL,
    payload_sha256 character(64) NOT NULL,
    strategy_key text DEFAULT 'polymarket_btc_five_minute_market_contracts'::text NOT NULL,
    capture_artifact_id uuid NOT NULL,
    ingested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT chk_market_data_polymarket_btc_five_minute_contract_fee CHECK (((jsonb_typeof(fee_schedule) = 'object'::text) AND (octet_length((fee_schedule)::text) <= 32768))),
    CONSTRAINT chk_market_data_polymarket_btc_five_minute_contract_identity CHECK (((source = 'polymarket_gamma_rest'::text) AND (strategy_key = 'polymarket_btc_five_minute_market_contracts'::text) AND (series_slug = 'btc-up-or-down-5m'::text) AND ((octet_length(event_id) >= 1) AND (octet_length(event_id) <= 256)) AND ((octet_length(market_id) >= 1) AND (octet_length(market_id) <= 256)) AND (condition_id ~ '^0x[0-9a-f]{64}$'::text) AND (up_token_id ~ '^[0-9]{1,100}$'::text) AND (down_token_id ~ '^[0-9]{1,100}$'::text) AND (up_token_id <> down_token_id))),
    CONSTRAINT chk_market_data_polymarket_btc_five_minute_contract_payload CHECK (((jsonb_typeof(source_payload) = 'object'::text) AND (octet_length((source_payload)::text) <= 65536) AND ((source_payload ->> 'version'::text) = 'polymarket-btc-5m-contract-v1'::text) AND (revision_sha256 ~ '^[0-9a-f]{64}$'::text) AND (payload_sha256 ~ '^[0-9a-f]{64}$'::text) AND (revision_sha256 = payload_sha256))),
    CONSTRAINT chk_market_data_polymarket_btc_five_minute_contract_terms CHECK (((tick_size > (0)::numeric) AND (tick_size < (1)::numeric) AND ((minimum_order_size IS NULL) OR (minimum_order_size > (0)::numeric)) AND ((octet_length(resolution_source) >= 1) AND (octet_length(resolution_source) <= 2048)) AND (lower(resolution_source) ~ '(chainlink|chain\.link)'::text) AND (lower(resolution_source) ~ 'btc'::text) AND (lower(resolution_source) ~ 'usd'::text))),
    CONSTRAINT chk_market_data_polymarket_btc_five_minute_contract_window CHECK (((window_end = (window_start + '00:05:00'::interval)) AND (mod((EXTRACT(epoch FROM window_start))::bigint, (300)::bigint) = 0) AND (event_slug = ('btc-updown-5m-'::text || ((EXTRACT(epoch FROM window_start))::bigint)::text))))
);


--
-- Name: polymarket_btc_five_minute_resolutions; Type: TABLE; Schema: market_data; Owner: -
--

CREATE TABLE market_data.polymarket_btc_five_minute_resolutions (
    source text NOT NULL,
    market_id text NOT NULL,
    condition_id text NOT NULL,
    event_slug text NOT NULL,
    window_start timestamp with time zone NOT NULL,
    window_end timestamp with time zone NOT NULL,
    up_token_id text NOT NULL,
    down_token_id text NOT NULL,
    winning_token_id text NOT NULL,
    winning_outcome text NOT NULL,
    source_timestamp timestamp with time zone,
    provider_available_at timestamp with time zone,
    received_at timestamp with time zone NOT NULL,
    source_payload jsonb NOT NULL,
    revision_sha256 character(64) NOT NULL,
    payload_sha256 character(64) NOT NULL,
    strategy_key text DEFAULT 'polymarket_btc_five_minute_resolutions'::text NOT NULL,
    capture_artifact_id uuid NOT NULL,
    ingested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT chk_market_data_polymarket_btc_five_minute_resolution_identity CHECK (((source = ANY (ARRAY['clob_websocket'::text, 'clob_rest_reconciliation'::text, 'gamma_rest_reconciliation'::text])) AND (strategy_key = 'polymarket_btc_five_minute_resolutions'::text) AND ((octet_length(market_id) >= 1) AND (octet_length(market_id) <= 256)) AND (condition_id ~ '^0x[0-9a-f]{64}$'::text) AND (up_token_id ~ '^[0-9]{1,100}$'::text) AND (down_token_id ~ '^[0-9]{1,100}$'::text) AND (up_token_id <> down_token_id))),
    CONSTRAINT chk_market_data_polymarket_btc_five_minute_resolution_payload CHECK (((jsonb_typeof(source_payload) = 'object'::text) AND (octet_length((source_payload)::text) <= 1048576) AND (revision_sha256 ~ '^[0-9a-f]{64}$'::text) AND (payload_sha256 ~ '^[0-9a-f]{64}$'::text))),
    CONSTRAINT chk_market_data_polymarket_btc_five_minute_resolution_time CHECK (((received_at >= window_end) AND ((source_timestamp IS NULL) OR (source_timestamp >= window_end)))),
    CONSTRAINT chk_market_data_polymarket_btc_five_minute_resolution_window CHECK (((window_end = (window_start + '00:05:00'::interval)) AND (mod((EXTRACT(epoch FROM window_start))::bigint, (300)::bigint) = 0) AND (event_slug = ('btc-updown-5m-'::text || ((EXTRACT(epoch FROM window_start))::bigint)::text)))),
    CONSTRAINT chk_market_data_polymarket_btc_five_minute_resolution_winner CHECK (((winning_outcome = ANY (ARRAY['up'::text, 'down'::text])) AND (((winning_outcome = 'up'::text) AND (winning_token_id = up_token_id)) OR ((winning_outcome = 'down'::text) AND (winning_token_id = down_token_id))))),
    CONSTRAINT chk_md_polymarket_btc_five_minute_resolution_source_time CHECK ((((source = 'clob_rest_reconciliation'::text) AND (source_timestamp IS NULL) AND (provider_available_at IS NULL)) OR ((source = ANY (ARRAY['clob_websocket'::text, 'gamma_rest_reconciliation'::text])) AND (source_timestamp IS NOT NULL) AND (provider_available_at = source_timestamp))))
);


--
-- Name: account_position_snapshots; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.account_position_snapshots (
    snapshot_id uuid DEFAULT gen_random_uuid() NOT NULL,
    account_address text NOT NULL,
    token_id text NOT NULL,
    market_id text,
    size numeric(30,10) DEFAULT 0 NOT NULL,
    avg_price numeric(18,8),
    current_price numeric(18,8),
    current_value numeric(30,10),
    cash_pnl numeric(30,10),
    percent_pnl numeric(30,10),
    snapshot_at timestamp with time zone DEFAULT now() NOT NULL,
    source text NOT NULL,
    raw_payload jsonb DEFAULT '{}'::jsonb NOT NULL,
    CONSTRAINT chk_poly_account_position_snapshots_amounts CHECK (((size >= (0)::numeric) AND ((avg_price IS NULL) OR ((avg_price >= (0)::numeric) AND (avg_price <= (1)::numeric))) AND ((current_price IS NULL) OR ((current_price >= (0)::numeric) AND (current_price <= (1)::numeric))))),
    CONSTRAINT chk_poly_account_position_snapshots_source CHECK ((source = ANY (ARRAY['data_api'::text, 'manual_backfill'::text, 'poll'::text])))
);


--
-- Name: account_reconciliation_runs; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.account_reconciliation_runs (
    run_id uuid DEFAULT gen_random_uuid() NOT NULL,
    account_address text NOT NULL,
    source text NOT NULL,
    dry_run boolean DEFAULT true NOT NULL,
    token_id text,
    lookback_hours integer NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    completed_at timestamp with time zone,
    status text NOT NULL,
    activities_fetched integer DEFAULT 0 NOT NULL,
    account_trades_inserted integer DEFAULT 0 NOT NULL,
    position_snapshots_inserted integer DEFAULT 0 NOT NULL,
    exits_detected integer DEFAULT 0 NOT NULL,
    exits_applied integer DEFAULT 0 NOT NULL,
    mismatches_found integer DEFAULT 0 NOT NULL,
    unmatched_trades integer DEFAULT 0 NOT NULL,
    raw_summary jsonb DEFAULT '{}'::jsonb NOT NULL,
    process_id uuid,
    CONSTRAINT chk_poly_account_reconciliation_runs_source CHECK ((source = ANY (ARRAY['user_ws'::text, 'poll'::text, 'manual_backfill'::text, 'admin'::text]))),
    CONSTRAINT chk_poly_account_reconciliation_runs_status CHECK ((status = ANY (ARRAY['completed'::text, 'dry_run'::text, 'error'::text])))
);


--
-- Name: account_trades; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.account_trades (
    account_trade_id uuid DEFAULT gen_random_uuid() NOT NULL,
    account_address text NOT NULL,
    token_id text NOT NULL,
    market_id text,
    side text NOT NULL,
    price numeric(18,8) NOT NULL,
    size numeric(30,10) NOT NULL,
    notional numeric(30,10) NOT NULL,
    timestamp_utc timestamp with time zone NOT NULL,
    transaction_hash text,
    venue_order_id text,
    venue_trade_id text,
    source text NOT NULL,
    linked_order_id text,
    raw_payload jsonb DEFAULT '{}'::jsonb NOT NULL,
    applied_exit_size numeric(30,10) DEFAULT 0 NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_poly_account_trades_amounts CHECK (((price >= (0)::numeric) AND (price <= (1)::numeric) AND (size > (0)::numeric) AND (notional >= (0)::numeric) AND (applied_exit_size >= (0)::numeric) AND (applied_exit_size <= size))),
    CONSTRAINT chk_poly_account_trades_side CHECK ((side = ANY (ARRAY['buy'::text, 'sell'::text]))),
    CONSTRAINT chk_poly_account_trades_source CHECK ((source = ANY (ARRAY['user_ws'::text, 'data_api'::text, 'manual_backfill'::text, 'poll'::text])))
);


--
-- Name: backfill_materialization_retention_events; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.backfill_materialization_retention_events (
    retention_event_id uuid DEFAULT gen_random_uuid() NOT NULL,
    source_artifact_id uuid NOT NULL,
    replacement_artifact_id uuid,
    materialization text NOT NULL,
    action text NOT NULL,
    source_record_count bigint NOT NULL,
    occurred_at timestamp with time zone DEFAULT now() NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    CONSTRAINT chk_backfill_materialization_replacement CHECK (((action <> 'replaced'::text) OR (replacement_artifact_id IS NOT NULL))),
    CONSTRAINT chk_backfill_materialization_retention_count CHECK ((source_record_count >= 0)),
    CONSTRAINT chk_backfill_materialization_retention_identity CHECK (((length(btrim(materialization)) > 0) AND (action = ANY (ARRAY['replaced'::text, 'pruned'::text])))),
    CONSTRAINT chk_backfill_materialization_retention_metadata CHECK ((jsonb_typeof(metadata) = 'object'::text))
);


--
-- Name: binance_btcusdt_l2_training_features; Type: VIEW; Schema: polymarket; Owner: -
--

CREATE VIEW polymarket.binance_btcusdt_l2_training_features AS
 SELECT feature.symbol,
    feature.second_start,
    feature.source_event_timestamp,
    feature.provider_received_at,
    feature.available_at,
    feature.source_update_id,
    feature.midpoint,
    feature.microprice,
    feature.spread_bps,
    feature.bid_depth_5,
    feature.ask_depth_5,
    feature.imbalance_5,
    feature.bid_depth_10,
    feature.ask_depth_10,
    feature.imbalance_10,
    feature.bid_depth_20,
    feature.ask_depth_20,
    feature.imbalance_20,
    feature.bid_depth_slope_20,
    feature.ask_depth_slope_20,
    feature.bid_depth_concentration_20,
    feature.ask_depth_concentration_20,
    feature.bid_quote_replenishment_1s,
    feature.ask_quote_replenishment_1s,
    feature.bid_quote_churn_1s,
    feature.ask_quote_churn_1s,
    feature.midpoint_change_bps_1s,
    feature.spread_bps_delta_1s,
    feature.depth_20_change_bps_1s,
    feature.imbalance_20_delta_1s,
    feature.midpoint_change_bps_5s,
    feature.spread_bps_delta_5s,
    feature.depth_20_change_bps_5s,
    feature.imbalance_20_delta_5s,
    feature.midpoint_change_bps_15s,
    feature.spread_bps_delta_15s,
    feature.depth_20_change_bps_15s,
    feature.imbalance_20_delta_15s,
    feature.midpoint_change_bps_30s,
    feature.spread_bps_delta_30s,
    feature.depth_20_change_bps_30s,
    feature.imbalance_20_delta_30s,
    feature.midpoint_change_bps_60s,
    feature.spread_bps_delta_60s,
    feature.depth_20_change_bps_60s,
    feature.imbalance_20_delta_60s
   FROM (market_data.binance_futures_btcusdt_l2_one_second_features feature
     JOIN ingester.backfill_artifacts artifact ON (((artifact.artifact_id = feature.artifact_id) AND (artifact.status = 'completed'::text))));


--
-- Name: binance_spot_btcusdt_l2_training_features; Type: VIEW; Schema: polymarket; Owner: -
--

CREATE VIEW polymarket.binance_spot_btcusdt_l2_training_features AS
 SELECT feature.symbol,
    feature.second_start,
    feature.source_event_timestamp,
    feature.provider_received_at,
    feature.available_at,
    feature.source_update_id,
    feature.midpoint,
    feature.microprice,
    feature.spread_bps,
    feature.bid_depth_5,
    feature.ask_depth_5,
    feature.imbalance_5,
    feature.bid_depth_10,
    feature.ask_depth_10,
    feature.imbalance_10,
    feature.bid_depth_20,
    feature.ask_depth_20,
    feature.imbalance_20,
    feature.bid_depth_slope_20,
    feature.ask_depth_slope_20,
    feature.bid_depth_concentration_20,
    feature.ask_depth_concentration_20,
    feature.bid_quote_replenishment_1s,
    feature.ask_quote_replenishment_1s,
    feature.bid_quote_churn_1s,
    feature.ask_quote_churn_1s,
    feature.midpoint_change_bps_1s,
    feature.spread_bps_delta_1s,
    feature.depth_20_change_bps_1s,
    feature.imbalance_20_delta_1s,
    feature.midpoint_change_bps_5s,
    feature.spread_bps_delta_5s,
    feature.depth_20_change_bps_5s,
    feature.imbalance_20_delta_5s,
    feature.midpoint_change_bps_15s,
    feature.spread_bps_delta_15s,
    feature.depth_20_change_bps_15s,
    feature.imbalance_20_delta_15s,
    feature.midpoint_change_bps_30s,
    feature.spread_bps_delta_30s,
    feature.depth_20_change_bps_30s,
    feature.imbalance_20_delta_30s,
    feature.midpoint_change_bps_60s,
    feature.spread_bps_delta_60s,
    feature.depth_20_change_bps_60s,
    feature.imbalance_20_delta_60s
   FROM (market_data.binance_spot_btcusdt_l2_one_second_features feature
     JOIN ingester.backfill_artifacts artifact ON (((artifact.artifact_id = feature.artifact_id) AND (artifact.status = 'completed'::text))));


--
-- Name: btc_interval_markets; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.btc_interval_markets (
    market_id text NOT NULL,
    event_id text NOT NULL,
    event_slug text NOT NULL,
    question text NOT NULL,
    series_slug text NOT NULL,
    window_start timestamp with time zone NOT NULL,
    window_end timestamp with time zone NOT NULL,
    condition_id text NOT NULL,
    up_token_id text NOT NULL,
    down_token_id text NOT NULL,
    resolution_source text NOT NULL,
    accepting_orders boolean DEFAULT false NOT NULL,
    active boolean DEFAULT false NOT NULL,
    closed boolean DEFAULT false NOT NULL,
    min_tick_size numeric(18,8) NOT NULL,
    min_order_size numeric(30,10) NOT NULL,
    fee_rate numeric(18,8),
    fee_exponent integer,
    fee_taker_only boolean,
    validation_status text NOT NULL,
    validation_errors jsonb DEFAULT '[]'::jsonb NOT NULL,
    reference_price numeric(30,10),
    reference_source_timestamp timestamp with time zone,
    resolution_price numeric(30,10),
    resolution_source_timestamp timestamp with time zone,
    resolved_outcome text,
    discovered_at timestamp with time zone NOT NULL,
    last_refreshed_at timestamp with time zone NOT NULL,
    raw_payload jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    official_outcome text,
    official_resolved_at timestamp with time zone,
    official_winning_token_id text,
    official_resolution_source text,
    official_resolution_received_at timestamp with time zone,
    official_resolution_payload jsonb,
    CONSTRAINT chk_btc_interval_official_outcome CHECK (((official_outcome IS NULL) OR (official_outcome = ANY (ARRAY['up'::text, 'down'::text])))),
    CONSTRAINT chk_btc_interval_outcome CHECK (((resolved_outcome IS NULL) OR (resolved_outcome = ANY (ARRAY['up'::text, 'down'::text])))),
    CONSTRAINT chk_btc_interval_tokens CHECK ((up_token_id <> down_token_id)),
    CONSTRAINT chk_btc_interval_validation CHECK ((validation_status = ANY (ARRAY['valid'::text, 'invalid'::text, 'ineligible'::text]))),
    CONSTRAINT chk_btc_interval_window CHECK ((window_end = (window_start + '00:05:00'::interval))),
    CONSTRAINT chk_btc_official_resolution_all_or_none CHECK ((((official_outcome IS NULL) AND (official_resolved_at IS NULL) AND (official_winning_token_id IS NULL) AND (official_resolution_source IS NULL) AND (official_resolution_received_at IS NULL) AND (official_resolution_payload IS NULL)) OR ((official_outcome IS NOT NULL) AND (official_resolved_at IS NOT NULL) AND (official_winning_token_id IS NOT NULL) AND (official_resolution_source IS NOT NULL) AND (official_resolution_received_at IS NOT NULL) AND (official_resolution_payload IS NOT NULL)))),
    CONSTRAINT chk_btc_official_resolution_payload CHECK (((official_resolution_payload IS NULL) OR (jsonb_typeof(official_resolution_payload) = 'object'::text))),
    CONSTRAINT chk_btc_official_resolution_provenance CHECK (((official_resolution_source IS NULL) OR (official_resolution_source = ANY (ARRAY['clob_websocket'::text, 'clob_rest_reconciliation'::text, 'clob_websocket_legacy'::text, 'gamma_rest_reconciliation'::text])))),
    CONSTRAINT chk_btc_official_resolution_time CHECK (((official_resolved_at IS NULL) OR ((official_resolved_at >= window_end) AND (official_resolution_received_at >= window_end)))),
    CONSTRAINT chk_btc_official_resolution_winner CHECK (((official_outcome IS NULL) OR ((official_outcome = 'up'::text) AND (official_winning_token_id = up_token_id)) OR ((official_outcome = 'down'::text) AND (official_winning_token_id = down_token_id))))
);


--
-- Name: btc_market_capacity_execution_snapshots; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.btc_market_capacity_execution_snapshots (
    market_id text NOT NULL,
    sampled_at timestamp with time zone NOT NULL,
    artifact_id uuid NOT NULL,
    schema_version text NOT NULL,
    up_source_row_number bigint,
    up_source_timestamp timestamp with time zone,
    up_provider_received_at timestamp with time zone,
    up_best_bid numeric(18,8),
    up_best_ask numeric(18,8),
    up_best_bid_size numeric(30,10),
    up_best_ask_size numeric(30,10),
    up_bid_depth numeric(30,10),
    up_ask_depth numeric(30,10),
    up_ask_vwap_1 numeric(18,8),
    up_ask_vwap_5 numeric(18,8),
    up_ask_vwap_10 numeric(18,8),
    up_imbalance numeric(18,8),
    down_source_row_number bigint,
    down_source_timestamp timestamp with time zone,
    down_provider_received_at timestamp with time zone,
    down_best_bid numeric(18,8),
    down_best_ask numeric(18,8),
    down_best_bid_size numeric(30,10),
    down_best_ask_size numeric(30,10),
    down_bid_depth numeric(30,10),
    down_ask_depth numeric(30,10),
    down_ask_vwap_1 numeric(18,8),
    down_ask_vwap_5 numeric(18,8),
    down_ask_vwap_10 numeric(18,8),
    down_imbalance numeric(18,8),
    quality_flags integer NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    up_ask_vwap_15 numeric(18,8),
    up_ask_vwap_20 numeric(18,8),
    up_ask_vwap_25 numeric(18,8),
    up_ask_vwap_30 numeric(18,8),
    up_ask_vwap_40 numeric(18,8),
    up_ask_vwap_50 numeric(18,8),
    up_ask_vwap_75 numeric(18,8),
    up_ask_vwap_100 numeric(18,8),
    up_ask_vwap_125 numeric(18,8),
    up_ask_vwap_150 numeric(18,8),
    up_ask_vwap_175 numeric(18,8),
    up_ask_vwap_200 numeric(18,8),
    down_ask_vwap_15 numeric(18,8),
    down_ask_vwap_20 numeric(18,8),
    down_ask_vwap_25 numeric(18,8),
    down_ask_vwap_30 numeric(18,8),
    down_ask_vwap_40 numeric(18,8),
    down_ask_vwap_50 numeric(18,8),
    down_ask_vwap_75 numeric(18,8),
    down_ask_vwap_100 numeric(18,8),
    down_ask_vwap_125 numeric(18,8),
    down_ask_vwap_150 numeric(18,8),
    down_ask_vwap_175 numeric(18,8),
    down_ask_vwap_200 numeric(18,8),
    CONSTRAINT chk_btc_market_capacity_execution_snapshot_expanded_prices CHECK ((((up_ask_vwap_25 IS NULL) OR ((up_ask_vwap_25 >= (0)::numeric) AND (up_ask_vwap_25 <= (1)::numeric))) AND ((up_ask_vwap_30 IS NULL) OR ((up_ask_vwap_30 >= (0)::numeric) AND (up_ask_vwap_30 <= (1)::numeric))) AND ((up_ask_vwap_40 IS NULL) OR ((up_ask_vwap_40 >= (0)::numeric) AND (up_ask_vwap_40 <= (1)::numeric))) AND ((up_ask_vwap_50 IS NULL) OR ((up_ask_vwap_50 >= (0)::numeric) AND (up_ask_vwap_50 <= (1)::numeric))) AND ((up_ask_vwap_75 IS NULL) OR ((up_ask_vwap_75 >= (0)::numeric) AND (up_ask_vwap_75 <= (1)::numeric))) AND ((up_ask_vwap_100 IS NULL) OR ((up_ask_vwap_100 >= (0)::numeric) AND (up_ask_vwap_100 <= (1)::numeric))) AND ((up_ask_vwap_125 IS NULL) OR ((up_ask_vwap_125 >= (0)::numeric) AND (up_ask_vwap_125 <= (1)::numeric))) AND ((up_ask_vwap_150 IS NULL) OR ((up_ask_vwap_150 >= (0)::numeric) AND (up_ask_vwap_150 <= (1)::numeric))) AND ((up_ask_vwap_175 IS NULL) OR ((up_ask_vwap_175 >= (0)::numeric) AND (up_ask_vwap_175 <= (1)::numeric))) AND ((up_ask_vwap_200 IS NULL) OR ((up_ask_vwap_200 >= (0)::numeric) AND (up_ask_vwap_200 <= (1)::numeric))) AND ((down_ask_vwap_25 IS NULL) OR ((down_ask_vwap_25 >= (0)::numeric) AND (down_ask_vwap_25 <= (1)::numeric))) AND ((down_ask_vwap_30 IS NULL) OR ((down_ask_vwap_30 >= (0)::numeric) AND (down_ask_vwap_30 <= (1)::numeric))) AND ((down_ask_vwap_40 IS NULL) OR ((down_ask_vwap_40 >= (0)::numeric) AND (down_ask_vwap_40 <= (1)::numeric))) AND ((down_ask_vwap_50 IS NULL) OR ((down_ask_vwap_50 >= (0)::numeric) AND (down_ask_vwap_50 <= (1)::numeric))) AND ((down_ask_vwap_75 IS NULL) OR ((down_ask_vwap_75 >= (0)::numeric) AND (down_ask_vwap_75 <= (1)::numeric))) AND ((down_ask_vwap_100 IS NULL) OR ((down_ask_vwap_100 >= (0)::numeric) AND (down_ask_vwap_100 <= (1)::numeric))) AND ((down_ask_vwap_125 IS NULL) OR ((down_ask_vwap_125 >= (0)::numeric) AND (down_ask_vwap_125 <= (1)::numeric))) AND ((down_ask_vwap_150 IS NULL) OR ((down_ask_vwap_150 >= (0)::numeric) AND (down_ask_vwap_150 <= (1)::numeric))) AND ((down_ask_vwap_175 IS NULL) OR ((down_ask_vwap_175 >= (0)::numeric) AND (down_ask_vwap_175 <= (1)::numeric))) AND ((down_ask_vwap_200 IS NULL) OR ((down_ask_vwap_200 >= (0)::numeric) AND (down_ask_vwap_200 <= (1)::numeric))))),
    CONSTRAINT chk_btc_market_capacity_execution_snapshot_expanded_vwap CHECK ((((up_ask_vwap_25 IS NULL) OR (up_ask_vwap_20 IS NULL) OR (up_ask_vwap_25 >= up_ask_vwap_20)) AND ((up_ask_vwap_30 IS NULL) OR (up_ask_vwap_25 IS NULL) OR (up_ask_vwap_30 >= up_ask_vwap_25)) AND ((up_ask_vwap_40 IS NULL) OR (up_ask_vwap_30 IS NULL) OR (up_ask_vwap_40 >= up_ask_vwap_30)) AND ((up_ask_vwap_50 IS NULL) OR (up_ask_vwap_40 IS NULL) OR (up_ask_vwap_50 >= up_ask_vwap_40)) AND ((up_ask_vwap_75 IS NULL) OR (up_ask_vwap_50 IS NULL) OR (up_ask_vwap_75 >= up_ask_vwap_50)) AND ((up_ask_vwap_100 IS NULL) OR (up_ask_vwap_75 IS NULL) OR (up_ask_vwap_100 >= up_ask_vwap_75)) AND ((up_ask_vwap_125 IS NULL) OR (up_ask_vwap_100 IS NULL) OR (up_ask_vwap_125 >= up_ask_vwap_100)) AND ((up_ask_vwap_150 IS NULL) OR (up_ask_vwap_125 IS NULL) OR (up_ask_vwap_150 >= up_ask_vwap_125)) AND ((up_ask_vwap_175 IS NULL) OR (up_ask_vwap_150 IS NULL) OR (up_ask_vwap_175 >= up_ask_vwap_150)) AND ((up_ask_vwap_200 IS NULL) OR (up_ask_vwap_175 IS NULL) OR (up_ask_vwap_200 >= up_ask_vwap_175)) AND ((down_ask_vwap_25 IS NULL) OR (down_ask_vwap_20 IS NULL) OR (down_ask_vwap_25 >= down_ask_vwap_20)) AND ((down_ask_vwap_30 IS NULL) OR (down_ask_vwap_25 IS NULL) OR (down_ask_vwap_30 >= down_ask_vwap_25)) AND ((down_ask_vwap_40 IS NULL) OR (down_ask_vwap_30 IS NULL) OR (down_ask_vwap_40 >= down_ask_vwap_30)) AND ((down_ask_vwap_50 IS NULL) OR (down_ask_vwap_40 IS NULL) OR (down_ask_vwap_50 >= down_ask_vwap_40)) AND ((down_ask_vwap_75 IS NULL) OR (down_ask_vwap_50 IS NULL) OR (down_ask_vwap_75 >= down_ask_vwap_50)) AND ((down_ask_vwap_100 IS NULL) OR (down_ask_vwap_75 IS NULL) OR (down_ask_vwap_100 >= down_ask_vwap_75)) AND ((down_ask_vwap_125 IS NULL) OR (down_ask_vwap_100 IS NULL) OR (down_ask_vwap_125 >= down_ask_vwap_100)) AND ((down_ask_vwap_150 IS NULL) OR (down_ask_vwap_125 IS NULL) OR (down_ask_vwap_150 >= down_ask_vwap_125)) AND ((down_ask_vwap_175 IS NULL) OR (down_ask_vwap_150 IS NULL) OR (down_ask_vwap_175 >= down_ask_vwap_150)) AND ((down_ask_vwap_200 IS NULL) OR (down_ask_vwap_175 IS NULL) OR (down_ask_vwap_200 >= down_ask_vwap_175)))),
    CONSTRAINT chk_btc_market_capacity_execution_snapshot_prices CHECK ((((up_ask_vwap_15 IS NULL) OR ((up_ask_vwap_15 >= (0)::numeric) AND (up_ask_vwap_15 <= (1)::numeric))) AND ((up_ask_vwap_20 IS NULL) OR ((up_ask_vwap_20 >= (0)::numeric) AND (up_ask_vwap_20 <= (1)::numeric))) AND ((down_ask_vwap_15 IS NULL) OR ((down_ask_vwap_15 >= (0)::numeric) AND (down_ask_vwap_15 <= (1)::numeric))) AND ((down_ask_vwap_20 IS NULL) OR ((down_ask_vwap_20 >= (0)::numeric) AND (down_ask_vwap_20 <= (1)::numeric))))),
    CONSTRAINT chk_btc_market_capacity_execution_snapshot_schema CHECK ((length(btrim(schema_version)) > 0)),
    CONSTRAINT chk_btc_market_capacity_execution_snapshot_vwap CHECK ((((up_ask_vwap_15 IS NULL) OR (up_ask_vwap_10 IS NULL) OR (up_ask_vwap_15 >= up_ask_vwap_10)) AND ((up_ask_vwap_20 IS NULL) OR (up_ask_vwap_15 IS NULL) OR (up_ask_vwap_20 >= up_ask_vwap_15)) AND ((down_ask_vwap_15 IS NULL) OR (down_ask_vwap_10 IS NULL) OR (down_ask_vwap_15 >= down_ask_vwap_10)) AND ((down_ask_vwap_20 IS NULL) OR (down_ask_vwap_15 IS NULL) OR (down_ask_vwap_20 >= down_ask_vwap_15)))),
    CONSTRAINT chk_btc_market_execution_snapshot_books CHECK ((((up_best_bid IS NULL) OR (up_best_ask IS NULL) OR (up_best_bid < up_best_ask)) AND ((down_best_bid IS NULL) OR (down_best_ask IS NULL) OR (down_best_bid < down_best_ask)))),
    CONSTRAINT chk_btc_market_execution_snapshot_causality CHECK ((((up_provider_received_at IS NULL) OR (up_provider_received_at <= sampled_at)) AND ((down_provider_received_at IS NULL) OR (down_provider_received_at <= sampled_at)))),
    CONSTRAINT chk_btc_market_execution_snapshot_imbalance CHECK ((((up_imbalance IS NULL) OR ((up_imbalance >= ('-1'::integer)::numeric) AND (up_imbalance <= (1)::numeric))) AND ((down_imbalance IS NULL) OR ((down_imbalance >= ('-1'::integer)::numeric) AND (down_imbalance <= (1)::numeric))))),
    CONSTRAINT chk_btc_market_execution_snapshot_prices CHECK ((((up_best_bid IS NULL) OR ((up_best_bid >= (0)::numeric) AND (up_best_bid <= (1)::numeric))) AND ((up_best_ask IS NULL) OR ((up_best_ask >= (0)::numeric) AND (up_best_ask <= (1)::numeric))) AND ((down_best_bid IS NULL) OR ((down_best_bid >= (0)::numeric) AND (down_best_bid <= (1)::numeric))) AND ((down_best_ask IS NULL) OR ((down_best_ask >= (0)::numeric) AND (down_best_ask <= (1)::numeric))) AND ((up_ask_vwap_1 IS NULL) OR ((up_ask_vwap_1 >= (0)::numeric) AND (up_ask_vwap_1 <= (1)::numeric))) AND ((up_ask_vwap_5 IS NULL) OR ((up_ask_vwap_5 >= (0)::numeric) AND (up_ask_vwap_5 <= (1)::numeric))) AND ((up_ask_vwap_10 IS NULL) OR ((up_ask_vwap_10 >= (0)::numeric) AND (up_ask_vwap_10 <= (1)::numeric))) AND ((down_ask_vwap_1 IS NULL) OR ((down_ask_vwap_1 >= (0)::numeric) AND (down_ask_vwap_1 <= (1)::numeric))) AND ((down_ask_vwap_5 IS NULL) OR ((down_ask_vwap_5 >= (0)::numeric) AND (down_ask_vwap_5 <= (1)::numeric))) AND ((down_ask_vwap_10 IS NULL) OR ((down_ask_vwap_10 >= (0)::numeric) AND (down_ask_vwap_10 <= (1)::numeric))))),
    CONSTRAINT chk_btc_market_execution_snapshot_quality CHECK ((quality_flags >= 0)),
    CONSTRAINT chk_btc_market_execution_snapshot_schema CHECK ((length(btrim(schema_version)) > 0)),
    CONSTRAINT chk_btc_market_execution_snapshot_sizes CHECK ((((up_best_bid_size IS NULL) OR (up_best_bid_size >= (0)::numeric)) AND ((up_best_ask_size IS NULL) OR (up_best_ask_size >= (0)::numeric)) AND ((up_bid_depth IS NULL) OR (up_bid_depth >= (0)::numeric)) AND ((up_ask_depth IS NULL) OR (up_ask_depth >= (0)::numeric)) AND ((down_best_bid_size IS NULL) OR (down_best_bid_size >= (0)::numeric)) AND ((down_best_ask_size IS NULL) OR (down_best_ask_size >= (0)::numeric)) AND ((down_bid_depth IS NULL) OR (down_bid_depth >= (0)::numeric)) AND ((down_ask_depth IS NULL) OR (down_ask_depth >= (0)::numeric)))),
    CONSTRAINT chk_btc_market_execution_snapshot_source_rows CHECK ((((up_source_row_number IS NULL) OR (up_source_row_number >= 0)) AND ((down_source_row_number IS NULL) OR (down_source_row_number >= 0)))),
    CONSTRAINT chk_btc_market_execution_snapshot_vwap CHECK ((((up_ask_vwap_1 IS NULL) OR (up_best_ask IS NULL) OR (up_ask_vwap_1 >= up_best_ask)) AND ((up_ask_vwap_5 IS NULL) OR (up_ask_vwap_1 IS NULL) OR (up_ask_vwap_5 >= up_ask_vwap_1)) AND ((up_ask_vwap_10 IS NULL) OR (up_ask_vwap_5 IS NULL) OR (up_ask_vwap_10 >= up_ask_vwap_5)) AND ((down_ask_vwap_1 IS NULL) OR (down_best_ask IS NULL) OR (down_ask_vwap_1 >= down_best_ask)) AND ((down_ask_vwap_5 IS NULL) OR (down_ask_vwap_1 IS NULL) OR (down_ask_vwap_5 >= down_ask_vwap_1)) AND ((down_ask_vwap_10 IS NULL) OR (down_ask_vwap_5 IS NULL) OR (down_ask_vwap_10 >= down_ask_vwap_5))))
);


--
-- Name: btc_market_execution_snapshots; Type: VIEW; Schema: polymarket; Owner: -
--

CREATE VIEW polymarket.btc_market_execution_snapshots AS
 SELECT snapshot.market_id,
    snapshot.sampled_at,
    snapshot.artifact_id,
    snapshot.schema_version,
    snapshot.up_source_row_number,
    snapshot.up_source_timestamp,
    snapshot.up_provider_received_at,
    snapshot.up_best_bid,
    snapshot.up_best_ask,
    snapshot.up_best_bid_size,
    snapshot.up_best_ask_size,
    snapshot.up_bid_depth,
    snapshot.up_ask_depth,
    snapshot.up_ask_vwap_1,
    snapshot.up_ask_vwap_5,
    snapshot.up_ask_vwap_10,
    snapshot.up_imbalance,
    snapshot.down_source_row_number,
    snapshot.down_source_timestamp,
    snapshot.down_provider_received_at,
    snapshot.down_best_bid,
    snapshot.down_best_ask,
    snapshot.down_best_bid_size,
    snapshot.down_best_ask_size,
    snapshot.down_bid_depth,
    snapshot.down_ask_depth,
    snapshot.down_ask_vwap_1,
    snapshot.down_ask_vwap_5,
    snapshot.down_ask_vwap_10,
    snapshot.down_imbalance,
    snapshot.quality_flags,
    snapshot.created_at
   FROM polymarket.btc_market_capacity_execution_snapshots snapshot
 OFFSET 0;


--
-- Name: btc_market_decision_execution_snapshots; Type: VIEW; Schema: polymarket; Owner: -
--

CREATE VIEW polymarket.btc_market_decision_execution_snapshots AS
 SELECT snapshot.market_id,
    snapshot.sampled_at,
    snapshot.artifact_id,
    snapshot.schema_version,
    snapshot.up_source_row_number,
    snapshot.up_source_timestamp,
    snapshot.up_provider_received_at,
    snapshot.up_best_bid,
    snapshot.up_best_ask,
    snapshot.up_best_bid_size,
    snapshot.up_best_ask_size,
    snapshot.up_bid_depth,
    snapshot.up_ask_depth,
    snapshot.up_ask_vwap_1,
    snapshot.up_ask_vwap_5,
    snapshot.up_ask_vwap_10,
    snapshot.up_imbalance,
    snapshot.down_source_row_number,
    snapshot.down_source_timestamp,
    snapshot.down_provider_received_at,
    snapshot.down_best_bid,
    snapshot.down_best_ask,
    snapshot.down_best_bid_size,
    snapshot.down_best_ask_size,
    snapshot.down_bid_depth,
    snapshot.down_ask_depth,
    snapshot.down_ask_vwap_1,
    snapshot.down_ask_vwap_5,
    snapshot.down_ask_vwap_10,
    snapshot.down_imbalance,
    snapshot.quality_flags,
    snapshot.created_at
   FROM (polymarket.btc_market_execution_snapshots snapshot
     JOIN polymarket.btc_interval_markets market USING (market_id))
  WHERE ((((EXTRACT(epoch FROM (snapshot.sampled_at - market.window_start)))::integer >= 90) AND ((EXTRACT(epoch FROM (snapshot.sampled_at - market.window_start)))::integer <= 140)) AND (mod((EXTRACT(epoch FROM (snapshot.sampled_at - market.window_start)))::integer, 5) = 0));


--
-- Name: btc_market_execution_snapshots_one_second; Type: VIEW; Schema: polymarket; Owner: -
--

CREATE VIEW polymarket.btc_market_execution_snapshots_one_second AS
 SELECT snapshot.market_id,
    snapshot.sampled_at,
    snapshot.artifact_id,
    snapshot.schema_version,
    snapshot.up_source_row_number,
    snapshot.up_source_timestamp,
    snapshot.up_provider_received_at,
    snapshot.up_best_bid,
    snapshot.up_best_ask,
    snapshot.up_best_bid_size,
    snapshot.up_best_ask_size,
    snapshot.up_bid_depth,
    snapshot.up_ask_depth,
    snapshot.up_ask_vwap_1,
    snapshot.up_ask_vwap_5,
    snapshot.up_ask_vwap_10,
    snapshot.up_imbalance,
    snapshot.down_source_row_number,
    snapshot.down_source_timestamp,
    snapshot.down_provider_received_at,
    snapshot.down_best_bid,
    snapshot.down_best_ask,
    snapshot.down_best_bid_size,
    snapshot.down_best_ask_size,
    snapshot.down_bid_depth,
    snapshot.down_ask_depth,
    snapshot.down_ask_vwap_1,
    snapshot.down_ask_vwap_5,
    snapshot.down_ask_vwap_10,
    snapshot.down_imbalance,
    snapshot.quality_flags,
    snapshot.created_at
   FROM polymarket.btc_market_execution_snapshots snapshot
  WHERE (((EXTRACT(milliseconds FROM snapshot.sampled_at))::integer % 1000) = 0);


--
-- Name: btc_market_labels; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.btc_market_labels (
    market_id text NOT NULL,
    window_start timestamp with time zone NOT NULL,
    window_end timestamp with time zone NOT NULL,
    open_price numeric(30,10) NOT NULL,
    close_price numeric(30,10) NOT NULL,
    outcome text NOT NULL,
    label_source text NOT NULL,
    label_version text NOT NULL,
    source_open_timestamp timestamp with time zone NOT NULL,
    source_close_timestamp timestamp with time zone NOT NULL,
    label_available_at timestamp with time zone NOT NULL,
    official_outcome text,
    official_resolved_at timestamp with time zone,
    evidence jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_btc_label_outcome CHECK ((outcome = ANY (ARRAY['up'::text, 'down'::text]))),
    CONSTRAINT chk_btc_official_outcome CHECK (((official_outcome IS NULL) OR (official_outcome = ANY (ARRAY['up'::text, 'down'::text]))))
);


--
-- Name: btc_market_reference_facts; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.btc_market_reference_facts (
    fact_id uuid DEFAULT gen_random_uuid() NOT NULL,
    market_id text NOT NULL,
    artifact_id uuid NOT NULL,
    fact_type text NOT NULL,
    value numeric(30,10) NOT NULL,
    provider text NOT NULL,
    source_effective_at timestamp with time zone NOT NULL,
    fetched_at timestamp with time zone NOT NULL,
    payload_sha256 text NOT NULL,
    evidence jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_btc_market_reference_fact_evidence CHECK ((jsonb_typeof(evidence) = 'object'::text)),
    CONSTRAINT chk_btc_market_reference_fact_payload_sha256 CHECK ((payload_sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT chk_btc_market_reference_fact_provider CHECK ((length(btrim(provider)) > 0)),
    CONSTRAINT chk_btc_market_reference_fact_times CHECK ((fetched_at >= source_effective_at)),
    CONSTRAINT chk_btc_market_reference_fact_type CHECK ((fact_type = ANY (ARRAY['opening_boundary'::text, 'final_price'::text]))),
    CONSTRAINT chk_btc_market_reference_fact_value CHECK ((value > (0)::numeric))
);


--
-- Name: btc_official_resolution_watches; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.btc_official_resolution_watches (
    market_id text NOT NULL,
    status text DEFAULT 'pending'::text NOT NULL,
    watch_started_at timestamp with time zone NOT NULL,
    deadline_at timestamp with time zone NOT NULL,
    last_checked_at timestamp with time zone,
    last_subscribed_at timestamp with time zone,
    last_subscription_connection_id uuid,
    subscription_count bigint DEFAULT 0 NOT NULL,
    resolution_received_at timestamp with time zone,
    resolution_source text,
    expired_at timestamp with time zone,
    last_error text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_btc_resolution_watch_deadline CHECK ((deadline_at > watch_started_at)),
    CONSTRAINT chk_btc_resolution_watch_source CHECK (((resolution_source IS NULL) OR (resolution_source = ANY (ARRAY['clob_websocket'::text, 'clob_rest_reconciliation'::text, 'gamma_rest_reconciliation'::text])))),
    CONSTRAINT chk_btc_resolution_watch_state CHECK ((((status = 'pending'::text) AND (resolution_received_at IS NULL) AND (resolution_source IS NULL) AND (expired_at IS NULL)) OR ((status = 'resolved'::text) AND (resolution_received_at IS NOT NULL) AND (resolution_source IS NOT NULL) AND (expired_at IS NULL)) OR ((status = 'expired'::text) AND (resolution_received_at IS NULL) AND (resolution_source IS NULL) AND (expired_at IS NOT NULL)) OR ((status = 'resolved_late'::text) AND (resolution_received_at IS NOT NULL) AND (resolution_source IS NOT NULL) AND (expired_at IS NOT NULL)))),
    CONSTRAINT chk_btc_resolution_watch_status CHECK ((status = ANY (ARRAY['pending'::text, 'resolved'::text, 'expired'::text, 'resolved_late'::text]))),
    CONSTRAINT chk_btc_resolution_watch_subscription_count CHECK ((subscription_count >= 0))
);


--
-- Name: btc_orderbook_archive_events; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.btc_orderbook_archive_events (
    artifact_id uuid NOT NULL,
    source_row_number bigint NOT NULL,
    provider_received_at timestamp with time zone NOT NULL,
    source_timestamp timestamp with time zone NOT NULL,
    condition_id text NOT NULL,
    asset_id text NOT NULL,
    event_type text NOT NULL,
    bids jsonb,
    asks jsonb,
    price numeric(18,8),
    size numeric(30,10),
    side text,
    best_bid numeric(18,8),
    best_ask numeric(18,8),
    fee_rate_bps integer,
    transaction_hash text,
    old_tick_size numeric(18,8),
    new_tick_size numeric(18,8),
    ingested_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_btc_orderbook_archive_book CHECK (((event_type <> 'book'::text) OR ((jsonb_typeof(bids) = 'array'::text) AND (jsonb_typeof(asks) = 'array'::text)))),
    CONSTRAINT chk_btc_orderbook_archive_event_type CHECK ((event_type = ANY (ARRAY['book'::text, 'price_change'::text, 'last_trade_price'::text, 'tick_size_change'::text]))),
    CONSTRAINT chk_btc_orderbook_archive_identity CHECK (((condition_id ~ '^0x[0-9a-fA-F]{64}$'::text) AND (asset_id ~ '^[0-9]+$'::text))),
    CONSTRAINT chk_btc_orderbook_archive_prices CHECK ((((price IS NULL) OR ((price >= (0)::numeric) AND (price <= (1)::numeric))) AND ((best_bid IS NULL) OR ((best_bid >= (0)::numeric) AND (best_bid <= (1)::numeric))) AND ((best_ask IS NULL) OR ((best_ask >= (0)::numeric) AND (best_ask <= (1)::numeric))) AND ((old_tick_size IS NULL) OR ((old_tick_size > (0)::numeric) AND (old_tick_size <= (1)::numeric))) AND ((new_tick_size IS NULL) OR ((new_tick_size > (0)::numeric) AND (new_tick_size <= (1)::numeric))) AND ((size IS NULL) OR (size >= (0)::numeric)) AND ((fee_rate_bps IS NULL) OR (fee_rate_bps >= 0)) AND ((side IS NULL) OR (side = ANY (ARRAY['buy'::text, 'sell'::text]))))),
    CONSTRAINT chk_btc_orderbook_archive_source_row CHECK ((source_row_number >= 0))
);


--
-- Name: btc_paper_settlement_ledger; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.btc_paper_settlement_ledger (
    settlement_id uuid DEFAULT gen_random_uuid() NOT NULL,
    run_id uuid NOT NULL,
    process_id uuid NOT NULL,
    order_id text NOT NULL,
    market_id text NOT NULL,
    token_id text NOT NULL,
    fill_ids jsonb NOT NULL,
    official_outcome text NOT NULL,
    official_winning_token_id text NOT NULL,
    official_resolution_received_at timestamp with time zone NOT NULL,
    official_resolution_source text NOT NULL,
    filled_size numeric(30,10) NOT NULL,
    entry_notional numeric(30,10) NOT NULL,
    entry_fees numeric(30,10) NOT NULL,
    payout numeric(30,10) NOT NULL,
    net_pnl numeric(30,10) NOT NULL,
    credit_status text DEFAULT 'pending'::text NOT NULL,
    credited_at timestamp with time zone,
    credit_attempts bigint DEFAULT 0 NOT NULL,
    credit_evidence jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    execution_mode text DEFAULT 'paper'::text NOT NULL,
    CONSTRAINT chk_btc_paper_settlement_amounts CHECK (((filled_size > (0)::numeric) AND (entry_notional >= (0)::numeric) AND (entry_fees >= (0)::numeric) AND (payout >= (0)::numeric) AND (payout =
CASE
    WHEN (token_id = official_winning_token_id) THEN filled_size
    ELSE (0)::numeric
END) AND (net_pnl = ((payout - entry_notional) - entry_fees)))),
    CONSTRAINT chk_btc_paper_settlement_credit_state CHECK ((((credit_status = 'pending'::text) AND (credited_at IS NULL) AND (credit_attempts = 0)) OR ((credit_status = 'credited'::text) AND (credited_at IS NOT NULL) AND (credit_attempts >= 1)))),
    CONSTRAINT chk_btc_paper_settlement_credit_status CHECK ((credit_status = ANY (ARRAY['pending'::text, 'credited'::text]))),
    CONSTRAINT chk_btc_paper_settlement_evidence CHECK ((jsonb_typeof(credit_evidence) = 'object'::text)),
    CONSTRAINT chk_btc_paper_settlement_fill_ids CHECK (((jsonb_typeof(fill_ids) = 'array'::text) AND (jsonb_array_length(fill_ids) > 0))),
    CONSTRAINT chk_btc_paper_settlement_outcome CHECK ((official_outcome = ANY (ARRAY['up'::text, 'down'::text]))),
    CONSTRAINT chk_btc_paper_settlement_source CHECK ((official_resolution_source = ANY (ARRAY['clob_websocket'::text, 'clob_rest_reconciliation'::text, 'gamma_rest_reconciliation'::text]))),
    CONSTRAINT chk_btc_settlement_execution_mode CHECK ((execution_mode = ANY (ARRAY['paper'::text, 'live'::text])))
);


--
-- Name: btc_strategy_decisions; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.btc_strategy_decisions (
    decision_id uuid NOT NULL,
    decision_at timestamp with time zone NOT NULL,
    run_id uuid NOT NULL,
    process_id uuid NOT NULL,
    market_id text NOT NULL,
    snapshot_id uuid NOT NULL,
    strategy_version text NOT NULL,
    config_hash text NOT NULL,
    action text NOT NULL,
    outcome text,
    token_id text,
    fair_probability numeric(18,10),
    executable_price numeric(18,8),
    gross_edge_per_share numeric(30,10),
    fee_per_share numeric(30,10),
    reserve_per_share numeric(30,10),
    net_edge_per_share numeric(30,10),
    size numeric(30,10) DEFAULT 0 NOT NULL,
    status text NOT NULL,
    reject_reason text,
    order_plan_id uuid,
    execution_mode text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_btc_decision_action CHECK ((action = ANY (ARRAY['buy'::text, 'no_trade'::text]))),
    CONSTRAINT chk_btc_decision_outcome CHECK (((outcome IS NULL) OR (outcome = ANY (ARRAY['up'::text, 'down'::text]))))
);


--
-- Name: cryptohft_request_budget; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.cryptohft_request_budget (
    provider text NOT NULL,
    next_request_at timestamp with time zone DEFAULT now() NOT NULL,
    spacing_milliseconds integer NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_cryptohft_request_budget_provider CHECK ((provider = 'cryptohftdata'::text)),
    CONSTRAINT chk_cryptohft_request_budget_spacing CHECK ((spacing_milliseconds = 1100))
);


--
-- Name: fill_identities; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.fill_identities (
    fill_id uuid NOT NULL,
    process_id uuid,
    timestamp_utc timestamp with time zone NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: live_reconciliation_runs; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.live_reconciliation_runs (
    run_id uuid DEFAULT gen_random_uuid() NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    completed_at timestamp with time zone,
    status text NOT NULL,
    open_orders_seen integer DEFAULT 0 NOT NULL,
    fills_seen integer DEFAULT 0 NOT NULL,
    balances_seen integer DEFAULT 0 NOT NULL,
    mismatches_found integer DEFAULT 0 NOT NULL,
    mismatches_repaired integer DEFAULT 0 NOT NULL,
    unresolved_count integer DEFAULT 0 NOT NULL,
    raw_summary jsonb DEFAULT '{}'::jsonb NOT NULL,
    process_id uuid,
    account_ref text,
    CONSTRAINT chk_poly_live_reconciliation_runs_account_ref CHECK (((account_ref IS NULL) OR ((length(btrim(account_ref)) >= 1) AND (length(btrim(account_ref)) <= 128)))),
    CONSTRAINT chk_poly_live_reconciliation_runs_scope_pair CHECK (((process_id IS NULL) = (account_ref IS NULL)))
);


--
-- Name: live_venue_events; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.live_venue_events (
    event_id uuid DEFAULT gen_random_uuid() NOT NULL,
    source text NOT NULL,
    event_type text NOT NULL,
    venue_event_id text,
    venue_order_id text,
    venue_trade_id text,
    client_order_id uuid,
    market_id text,
    token_id text,
    event_status text,
    event_timestamp timestamp with time zone,
    event_hash text NOT NULL,
    raw_payload jsonb DEFAULT '{}'::jsonb NOT NULL,
    applied boolean DEFAULT false NOT NULL,
    apply_error text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: orders; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.orders (
    order_id text NOT NULL,
    client_order_id uuid NOT NULL,
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    market_id text NOT NULL,
    token_id text NOT NULL,
    side text NOT NULL,
    order_type text NOT NULL,
    price numeric(18,8) NOT NULL,
    size numeric(30,10) NOT NULL,
    state text NOT NULL,
    raw_payload jsonb DEFAULT '{}'::jsonb NOT NULL,
    venue_order_id text,
    venue_status text,
    submitted_at timestamp with time zone,
    accepted_at timestamp with time zone,
    last_venue_update_at timestamp with time zone,
    reconciliation_status text,
    process_id uuid,
    CONSTRAINT chk_polymarket_orders_order_type CHECK ((order_type = ANY (ARRAY['fok'::text, 'gtc'::text, 'gtd'::text]))),
    CONSTRAINT chk_polymarket_orders_side CHECK ((side = ANY (ARRAY['buy'::text, 'sell'::text])))
);


--
-- Name: trading_processes; Type: TABLE; Schema: polymarket; Owner: -
--

CREATE TABLE polymarket.trading_processes (
    process_id uuid DEFAULT gen_random_uuid() NOT NULL,
    name text NOT NULL,
    process_type text NOT NULL,
    process_scope text DEFAULT 'default'::text NOT NULL,
    process_key text,
    status text DEFAULT 'created'::text NOT NULL,
    enabled boolean DEFAULT false NOT NULL,
    hostname text,
    pid integer,
    version text,
    started_at timestamp with time zone DEFAULT now(),
    heartbeat_at timestamp with time zone,
    stopped_at timestamp with time zone,
    stop_reason text,
    last_error text,
    config jsonb DEFAULT '{}'::jsonb NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_poly_trading_processes_live_account_ref CHECK ((((config #>> '{execution,mode}'::text[]) <> 'live'::text) OR (((config #>> '{execution,account_ref}'::text[]) IS NOT NULL) AND ((config #>> '{execution,account_ref}'::text[]) = btrim((config #>> '{execution,account_ref}'::text[]))) AND ((config #>> '{execution,account_ref}'::text[]) ~ '^[A-Za-z0-9._:-]{1,128}$'::text)))),
    CONSTRAINT chk_poly_trading_processes_name CHECK ((length(btrim(name)) > 0)),
    CONSTRAINT chk_poly_trading_processes_pid CHECK (((pid IS NULL) OR (pid > 0))),
    CONSTRAINT chk_poly_trading_processes_scope CHECK ((length(btrim(process_scope)) > 0)),
    CONSTRAINT chk_poly_trading_processes_status CHECK ((length(btrim(status)) > 0)),
    CONSTRAINT chk_poly_trading_processes_type CHECK ((length(btrim(process_type)) > 0))
);


--
-- Name: backfill_artifacts backfill_artifacts_pkey; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.backfill_artifacts
    ADD CONSTRAINT backfill_artifacts_pkey PRIMARY KEY (artifact_id);


--
-- Name: backfill_job_events backfill_job_events_pkey; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.backfill_job_events
    ADD CONSTRAINT backfill_job_events_pkey PRIMARY KEY (event_id);


--
-- Name: backfill_jobs backfill_jobs_pkey; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.backfill_jobs
    ADD CONSTRAINT backfill_jobs_pkey PRIMARY KEY (job_id);


--
-- Name: capture_artifacts capture_artifacts_pkey; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.capture_artifacts
    ADD CONSTRAINT capture_artifacts_pkey PRIMARY KEY (artifact_id);


--
-- Name: data_gaps data_gaps_gap_fingerprint_key; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.data_gaps
    ADD CONSTRAINT data_gaps_gap_fingerprint_key UNIQUE (gap_fingerprint);


--
-- Name: data_gaps data_gaps_pkey; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.data_gaps
    ADD CONSTRAINT data_gaps_pkey PRIMARY KEY (gap_id);


--
-- Name: drain_jobs drain_jobs_pkey; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.drain_jobs
    ADD CONSTRAINT drain_jobs_pkey PRIMARY KEY (job_id);


--
-- Name: drain_objects drain_objects_pkey; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.drain_objects
    ADD CONSTRAINT drain_objects_pkey PRIMARY KEY (object_id);


--
-- Name: profiles profiles_pkey; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.profiles
    ADD CONSTRAINT profiles_pkey PRIMARY KEY (strategy_key);


--
-- Name: backfill_artifacts uq_ingester_backfill_artifact_legacy; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.backfill_artifacts
    ADD CONSTRAINT uq_ingester_backfill_artifact_legacy UNIQUE (legacy_source, legacy_artifact_id);


--
-- Name: backfill_artifacts uq_ingester_backfill_artifact_logical; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.backfill_artifacts
    ADD CONSTRAINT uq_ingester_backfill_artifact_logical UNIQUE (strategy_key, logical_key);


--
-- Name: backfill_jobs uq_ingester_backfill_legacy; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.backfill_jobs
    ADD CONSTRAINT uq_ingester_backfill_legacy UNIQUE (legacy_source, legacy_job_id);


--
-- Name: capture_artifacts uq_ingester_capture_artifact_strategy_id; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.capture_artifacts
    ADD CONSTRAINT uq_ingester_capture_artifact_strategy_id UNIQUE (strategy_key, artifact_id);


--
-- Name: drain_objects uq_ingester_drain_object_chunk; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.drain_objects
    ADD CONSTRAINT uq_ingester_drain_object_chunk UNIQUE (strategy_key, source_chunk_schema, source_chunk_name);


--
-- Name: workers workers_pkey; Type: CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.workers
    ADD CONSTRAINT workers_pkey PRIMARY KEY (worker_id);


--
-- Name: binance_futures_btcusdt_l2_one_second_features pk_market_data_binance_futures_btcusdt_l2_one_second_features; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.binance_futures_btcusdt_l2_one_second_features
    ADD CONSTRAINT pk_market_data_binance_futures_btcusdt_l2_one_second_features PRIMARY KEY (symbol, second_start);


--
-- Name: binance_futures_btcusdt_open_interest pk_market_data_binance_futures_open_interest; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.binance_futures_btcusdt_open_interest
    ADD CONSTRAINT pk_market_data_binance_futures_open_interest PRIMARY KEY (source_timestamp, symbol, period_seconds);


--
-- Name: binance_spot_btcusdt_aggregate_trades pk_market_data_binance_spot_aggregate_trades; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.binance_spot_btcusdt_aggregate_trades
    ADD CONSTRAINT pk_market_data_binance_spot_aggregate_trades PRIMARY KEY (symbol, trade_timestamp, aggregate_trade_id);


--
-- Name: binance_spot_btcusdt_l2_one_second_features pk_market_data_binance_spot_btcusdt_l2_one_second_features; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.binance_spot_btcusdt_l2_one_second_features
    ADD CONSTRAINT pk_market_data_binance_spot_btcusdt_l2_one_second_features PRIMARY KEY (symbol, second_start);


--
-- Name: binance_spot_btcusdt_l2_snapshots pk_market_data_binance_spot_l2_snapshots; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.binance_spot_btcusdt_l2_snapshots
    ADD CONSTRAINT pk_market_data_binance_spot_l2_snapshots PRIMARY KEY (source_timestamp, symbol, source_update_id, sampling_policy_sha256);


--
-- Name: binance_spot_btcusdt_one_second_ohlcv pk_market_data_binance_spot_one_second_ohlcv; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.binance_spot_btcusdt_one_second_ohlcv
    ADD CONSTRAINT pk_market_data_binance_spot_one_second_ohlcv PRIMARY KEY (symbol, open_timestamp);


--
-- Name: chainlink_btcusd_one_minute_candles pk_market_data_chainlink_btcusd_one_minute_candles; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.chainlink_btcusd_one_minute_candles
    ADD CONSTRAINT pk_market_data_chainlink_btcusd_one_minute_candles PRIMARY KEY (symbol, open_timestamp);


--
-- Name: chainlink_btcusd_reference_prices pk_market_data_chainlink_btcusd_reference_prices; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.chainlink_btcusd_reference_prices
    ADD CONSTRAINT pk_market_data_chainlink_btcusd_reference_prices PRIMARY KEY (feed_id, source_timestamp, report_sha256);


--
-- Name: pmdata_chainlink_btcusd_reference_prices pk_market_data_pmdata_chainlink_btcusd_reference_prices; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.pmdata_chainlink_btcusd_reference_prices
    ADD CONSTRAINT pk_market_data_pmdata_chainlink_btcusd_reference_prices PRIMARY KEY (feed_id, source_timestamp, report_sha256);


--
-- Name: pmdata_chainlink_btcusd_twap pk_market_data_pmdata_chainlink_btcusd_twap; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.pmdata_chainlink_btcusd_twap
    ADD CONSTRAINT pk_market_data_pmdata_chainlink_btcusd_twap PRIMARY KEY (source_timestamp, window_seconds);


--
-- Name: polygon_chainlink_btcusd_oracle_rounds pk_market_data_polygon_chainlink_btcusd_oracle_rounds; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.polygon_chainlink_btcusd_oracle_rounds
    ADD CONSTRAINT pk_market_data_polygon_chainlink_btcusd_oracle_rounds PRIMARY KEY (source_timestamp, chain_id, feed_proxy_address, transaction_hash, log_index);


--
-- Name: polymarket_btc_five_minute_contracts pk_market_data_polymarket_btc_five_minute_contracts; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.polymarket_btc_five_minute_contracts
    ADD CONSTRAINT pk_market_data_polymarket_btc_five_minute_contracts PRIMARY KEY (market_id, revision_sha256);


--
-- Name: polymarket_btc_five_minute_resolutions pk_market_data_polymarket_btc_five_minute_resolutions; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.polymarket_btc_five_minute_resolutions
    ADD CONSTRAINT pk_market_data_polymarket_btc_five_minute_resolutions PRIMARY KEY (market_id, source, payload_sha256);


--
-- Name: polymarket_chainlink_btcusd_twap pk_market_data_polymarket_chainlink_btcusd_twap; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.polymarket_chainlink_btcusd_twap
    ADD CONSTRAINT pk_market_data_polymarket_chainlink_btcusd_twap PRIMARY KEY (source_timestamp, symbol, window_seconds);


--
-- Name: pmdata_chainlink_btcusd_twap uq_market_data_pmdata_chainlink_btcusd_twap_archive_row; Type: CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.pmdata_chainlink_btcusd_twap
    ADD CONSTRAINT uq_market_data_pmdata_chainlink_btcusd_twap_archive_row UNIQUE (artifact_id, source_timestamp, archive_row_number);


--
-- Name: account_position_snapshots account_position_snapshots_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.account_position_snapshots
    ADD CONSTRAINT account_position_snapshots_pkey PRIMARY KEY (snapshot_id);


--
-- Name: account_reconciliation_runs account_reconciliation_runs_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.account_reconciliation_runs
    ADD CONSTRAINT account_reconciliation_runs_pkey PRIMARY KEY (run_id);


--
-- Name: account_trades account_trades_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.account_trades
    ADD CONSTRAINT account_trades_pkey PRIMARY KEY (account_trade_id);


--
-- Name: backfill_materialization_retention_events backfill_materialization_retention_events_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.backfill_materialization_retention_events
    ADD CONSTRAINT backfill_materialization_retention_events_pkey PRIMARY KEY (retention_event_id);


--
-- Name: btc_interval_markets btc_interval_markets_event_slug_key; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_interval_markets
    ADD CONSTRAINT btc_interval_markets_event_slug_key UNIQUE (event_slug);


--
-- Name: btc_interval_markets btc_interval_markets_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_interval_markets
    ADD CONSTRAINT btc_interval_markets_pkey PRIMARY KEY (market_id);


--
-- Name: btc_interval_markets btc_interval_markets_window_start_key; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_interval_markets
    ADD CONSTRAINT btc_interval_markets_window_start_key UNIQUE (window_start);


--
-- Name: btc_market_labels btc_market_labels_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_market_labels
    ADD CONSTRAINT btc_market_labels_pkey PRIMARY KEY (market_id);


--
-- Name: btc_market_reference_facts btc_market_reference_facts_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_market_reference_facts
    ADD CONSTRAINT btc_market_reference_facts_pkey PRIMARY KEY (fact_id);


--
-- Name: btc_official_resolution_watches btc_official_resolution_watches_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_official_resolution_watches
    ADD CONSTRAINT btc_official_resolution_watches_pkey PRIMARY KEY (market_id);


--
-- Name: btc_paper_settlement_ledger btc_paper_settlement_ledger_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_paper_settlement_ledger
    ADD CONSTRAINT btc_paper_settlement_ledger_pkey PRIMARY KEY (settlement_id);


--
-- Name: btc_strategy_decisions chk_btc_decision_mode; Type: CHECK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE polymarket.btc_strategy_decisions
    ADD CONSTRAINT chk_btc_decision_mode CHECK ((execution_mode = ANY (ARRAY['sim'::text, 'paper'::text, 'live'::text]))) NOT VALID;


--
-- Name: btc_five_minute_orderbook_snapshots chk_market_data_polymarket_btc_five_minute_orderbook_book; Type: CHECK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE polymarket.btc_five_minute_orderbook_snapshots
    ADD CONSTRAINT chk_market_data_polymarket_btc_five_minute_orderbook_book CHECK (((octet_length((bids)::text) <= 262144) AND (octet_length((asks)::text) <= 262144) AND market_data.is_valid_polymarket_btc_five_minute_book(bids, asks, bid_depth, ask_depth, best_bid, best_ask, ((sampling_policy ->> 'top_n'::text))::integer))) NOT VALID;


--
-- Name: btc_five_minute_orderbook_snapshots chk_market_data_polymarket_btc_five_minute_orderbook_prices; Type: CHECK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE polymarket.btc_five_minute_orderbook_snapshots
    ADD CONSTRAINT chk_market_data_polymarket_btc_five_minute_orderbook_prices CHECK (((tick_size > (0)::numeric) AND (tick_size < (1)::numeric) AND ((best_bid IS NULL) OR ((best_bid > (0)::numeric) AND (best_bid < (1)::numeric))) AND ((best_ask IS NULL) OR ((best_ask > (0)::numeric) AND (best_ask < (1)::numeric))) AND ((best_bid IS NULL) OR (best_ask IS NULL) OR (best_bid < best_ask)))) NOT VALID;


--
-- Name: btc_five_minute_orderbook_snapshots chk_market_data_polymarket_btc_five_minute_orderbook_time; Type: CHECK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE polymarket.btc_five_minute_orderbook_snapshots
    ADD CONSTRAINT chk_market_data_polymarket_btc_five_minute_orderbook_time CHECK (((provider_available_at = source_timestamp) AND (received_at <= sampled_at))) NOT VALID;


--
-- Name: cryptohft_request_budget cryptohft_request_budget_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.cryptohft_request_budget
    ADD CONSTRAINT cryptohft_request_budget_pkey PRIMARY KEY (provider);


--
-- Name: fill_identities fill_identities_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.fill_identities
    ADD CONSTRAINT fill_identities_pkey PRIMARY KEY (fill_id);


--
-- Name: live_reconciliation_runs live_reconciliation_runs_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.live_reconciliation_runs
    ADD CONSTRAINT live_reconciliation_runs_pkey PRIMARY KEY (run_id);


--
-- Name: live_venue_events live_venue_events_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.live_venue_events
    ADD CONSTRAINT live_venue_events_pkey PRIMARY KEY (event_id);


--
-- Name: orders orders_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.orders
    ADD CONSTRAINT orders_pkey PRIMARY KEY (order_id);


--
-- Name: btc_feature_snapshots pk_btc_feature_snapshots; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_feature_snapshots
    ADD CONSTRAINT pk_btc_feature_snapshots PRIMARY KEY (snapshot_id, feature_as_of);


--
-- Name: btc_market_capacity_execution_snapshots pk_btc_market_capacity_execution_snapshots; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_market_capacity_execution_snapshots
    ADD CONSTRAINT pk_btc_market_capacity_execution_snapshots PRIMARY KEY (market_id, sampled_at);


--
-- Name: btc_orderbook_archive_events pk_btc_orderbook_archive_events; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_orderbook_archive_events
    ADD CONSTRAINT pk_btc_orderbook_archive_events PRIMARY KEY (artifact_id, source_row_number, provider_received_at);


--
-- Name: btc_strategy_decisions pk_btc_strategy_decisions; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_strategy_decisions
    ADD CONSTRAINT pk_btc_strategy_decisions PRIMARY KEY (decision_id, decision_at);


--
-- Name: btc_five_minute_orderbook_snapshots pk_market_data_polymarket_btc_five_minute_orderbook_snapshots; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_five_minute_orderbook_snapshots
    ADD CONSTRAINT pk_market_data_polymarket_btc_five_minute_orderbook_snapshots PRIMARY KEY (sampled_at, market_id, token_id, sampling_policy_sha256);


--
-- Name: trading_process_events pk_poly_trading_process_events; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.trading_process_events
    ADD CONSTRAINT pk_poly_trading_process_events PRIMARY KEY (event_id, timestamp_utc);


--
-- Name: fills pk_polymarket_fills; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.fills
    ADD CONSTRAINT pk_polymarket_fills PRIMARY KEY (fill_id, timestamp_utc);


--
-- Name: risk_events pk_polymarket_risk_events; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.risk_events
    ADD CONSTRAINT pk_polymarket_risk_events PRIMARY KEY (event_id, timestamp_utc);


--
-- Name: reference_price_ticks pk_reference_price_ticks; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.reference_price_ticks
    ADD CONSTRAINT pk_reference_price_ticks PRIMARY KEY (tick_id, source_timestamp);


--
-- Name: trading_processes trading_processes_pkey; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.trading_processes
    ADD CONSTRAINT trading_processes_pkey PRIMARY KEY (process_id);


--
-- Name: backfill_materialization_retention_events uq_backfill_materialization_retention_event; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.backfill_materialization_retention_events
    ADD CONSTRAINT uq_backfill_materialization_retention_event UNIQUE (source_artifact_id, materialization, action);


--
-- Name: btc_interval_markets uq_btc_interval_market_official_identity; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_interval_markets
    ADD CONSTRAINT uq_btc_interval_market_official_identity UNIQUE (market_id, official_outcome, official_winning_token_id, official_resolution_received_at, official_resolution_source);


--
-- Name: btc_market_reference_facts uq_btc_market_reference_fact; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_market_reference_facts
    ADD CONSTRAINT uq_btc_market_reference_fact UNIQUE (market_id, fact_type, provider);


--
-- Name: btc_paper_settlement_ledger uq_btc_paper_settlement_run_order; Type: CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_paper_settlement_ledger
    ADD CONSTRAINT uq_btc_paper_settlement_run_order UNIQUE (run_id, order_id);


--
-- Name: idx_ingester_backfill_artifacts_job; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_backfill_artifacts_job ON ingester.backfill_artifacts USING btree (job_id, created_at, artifact_id);


--
-- Name: idx_ingester_backfill_claim; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_backfill_claim ON ingester.backfill_jobs USING btree (strategy_key, next_attempt_at, requested_at, job_id) WHERE ((job_kind = 'shard'::text) AND (status = 'queued'::text));


--
-- Name: idx_ingester_backfill_events_job_time; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_backfill_events_job_time ON ingester.backfill_job_events USING btree (job_id, recorded_at DESC, event_id);


--
-- Name: idx_ingester_backfill_expired_lease; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_backfill_expired_lease ON ingester.backfill_jobs USING btree (lease_expires_at, job_id) WHERE ((job_kind = 'shard'::text) AND (status = ANY (ARRAY['running'::text, 'cancel_requested'::text])));


--
-- Name: idx_ingester_backfill_history; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_backfill_history ON ingester.backfill_jobs USING btree (strategy_key, status, requested_at DESC, job_id DESC);


--
-- Name: idx_ingester_backfill_parent; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_backfill_parent ON ingester.backfill_jobs USING btree (parent_job_id, status, job_id) WHERE (parent_job_id IS NOT NULL);


--
-- Name: idx_ingester_backfill_worker; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_backfill_worker ON ingester.backfill_jobs USING btree (assigned_worker_id, status, heartbeat_at) WHERE (assigned_worker_id IS NOT NULL);


--
-- Name: idx_ingester_capture_artifact_strategy_status; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_capture_artifact_strategy_status ON ingester.capture_artifacts USING btree (strategy_key, status, capture_window_start DESC, artifact_id);


--
-- Name: idx_ingester_data_gap_open; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_data_gap_open ON ingester.data_gaps USING btree (strategy_key, detected_at, gap_id) WHERE (status = ANY (ARRAY['open'::text, 'repairing'::text]));


--
-- Name: idx_ingester_data_gap_source_range; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_data_gap_source_range ON ingester.data_gaps USING btree (strategy_key, source_time_start, source_time_end, gap_id);


--
-- Name: idx_ingester_drain_jobs_claim; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_drain_jobs_claim ON ingester.drain_jobs USING btree (requested_at, job_id) WHERE (status = ANY (ARRAY['queued'::text, 'running'::text]));


--
-- Name: idx_ingester_drain_objects_job; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_drain_objects_job ON ingester.drain_objects USING btree (job_id, source_start, object_id);


--
-- Name: idx_ingester_gap_pm_btc5m_contract_retry; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_gap_pm_btc5m_contract_retry ON ingester.data_gaps USING btree (updated_at, detected_at, gap_id) WHERE ((strategy_key = 'polymarket_btc_five_minute_market_contracts'::text) AND (reason_code = 'polymarket_gamma_contract_window_unavailable'::text) AND (status = ANY (ARRAY['open'::text, 'repairing'::text])));


--
-- Name: idx_ingester_profiles_expired_lease; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_profiles_expired_lease ON ingester.profiles USING btree (lease_expires_at, strategy_key) WHERE (lease_token IS NOT NULL);


--
-- Name: idx_ingester_workers_available; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_ingester_workers_available ON ingester.workers USING btree (lifecycle_state, heartbeat_at DESC, worker_id);


--
-- Name: idx_poly_backfill_artifacts_job; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_poly_backfill_artifacts_job ON ingester.backfill_artifacts USING btree (job_id, created_at, artifact_id);


--
-- Name: idx_poly_backfill_artifacts_status; Type: INDEX; Schema: ingester; Owner: -
--

CREATE INDEX idx_poly_backfill_artifacts_status ON ingester.backfill_artifacts USING btree (status, updated_at, artifact_id);


--
-- Name: uq_ingester_backfill_request_hash; Type: INDEX; Schema: ingester; Owner: -
--

CREATE UNIQUE INDEX uq_ingester_backfill_request_hash ON ingester.backfill_jobs USING btree (request_hash) WHERE ((job_kind = 'request'::text) AND (legacy_source IS NULL));


--
-- Name: uq_ingester_backfill_shard; Type: INDEX; Schema: ingester; Owner: -
--

CREATE UNIQUE INDEX uq_ingester_backfill_shard ON ingester.backfill_jobs USING btree (parent_job_id, shard_key) WHERE (job_kind = 'shard'::text);


--
-- Name: uq_ingester_capture_artifact_open_strategy; Type: INDEX; Schema: ingester; Owner: -
--

CREATE UNIQUE INDEX uq_ingester_capture_artifact_open_strategy ON ingester.capture_artifacts USING btree (strategy_key) WHERE (status = 'open'::text);


--
-- Name: idx_market_data_binance_futures_open_interest_artifact; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_binance_futures_open_interest_artifact ON market_data.binance_futures_btcusdt_open_interest USING btree (capture_artifact_id, source_timestamp DESC);


--
-- Name: idx_market_data_binance_futures_open_interest_recovery; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_binance_futures_open_interest_recovery ON market_data.binance_futures_btcusdt_open_interest USING btree (symbol, period_seconds, source_timestamp DESC);


--
-- Name: idx_market_data_binance_spot_aggregate_trade_artifact; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_binance_spot_aggregate_trade_artifact ON market_data.binance_spot_btcusdt_aggregate_trades USING btree (capture_artifact_id, trade_timestamp DESC);


--
-- Name: idx_market_data_binance_spot_aggregate_trade_recovery; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_binance_spot_aggregate_trade_recovery ON market_data.binance_spot_btcusdt_aggregate_trades USING btree (symbol, aggregate_trade_id DESC, trade_timestamp DESC);


--
-- Name: idx_market_data_binance_spot_l2_snapshot_artifact; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_binance_spot_l2_snapshot_artifact ON market_data.binance_spot_btcusdt_l2_snapshots USING btree (capture_artifact_id, source_timestamp DESC);


--
-- Name: idx_market_data_binance_spot_l2_snapshot_recovery; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_binance_spot_l2_snapshot_recovery ON market_data.binance_spot_btcusdt_l2_snapshots USING btree (symbol, source_update_id DESC, sampling_policy_sha256, source_timestamp DESC);


--
-- Name: idx_market_data_binance_spot_one_second_ohlcv_artifact; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_binance_spot_one_second_ohlcv_artifact ON market_data.binance_spot_btcusdt_one_second_ohlcv USING btree (capture_artifact_id, open_timestamp DESC);


--
-- Name: idx_market_data_chainlink_btcusd_one_minute_candles_artifact; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_chainlink_btcusd_one_minute_candles_artifact ON market_data.chainlink_btcusd_one_minute_candles USING btree (capture_artifact_id, open_timestamp DESC);


--
-- Name: idx_market_data_chainlink_btcusd_reference_prices_backfill_arti; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_chainlink_btcusd_reference_prices_backfill_arti ON market_data.chainlink_btcusd_reference_prices USING btree (backfill_artifact_id, archive_row_number) WHERE (backfill_artifact_id IS NOT NULL);


--
-- Name: idx_market_data_chainlink_btcusd_reference_prices_capture_artif; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_chainlink_btcusd_reference_prices_capture_artif ON market_data.chainlink_btcusd_reference_prices USING btree (capture_artifact_id, source_timestamp DESC) WHERE (capture_artifact_id IS NOT NULL);


--
-- Name: idx_market_data_chainlink_btcusd_reference_prices_recovery; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_chainlink_btcusd_reference_prices_recovery ON market_data.chainlink_btcusd_reference_prices USING btree (feed_id, source_timestamp DESC);


--
-- Name: idx_market_data_pmdata_chainlink_btcusd_reference_prices_backfi; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_pmdata_chainlink_btcusd_reference_prices_backfi ON market_data.pmdata_chainlink_btcusd_reference_prices USING btree (backfill_artifact_id, archive_row_number) WHERE (backfill_artifact_id IS NOT NULL);


--
-- Name: idx_market_data_pmdata_chainlink_btcusd_reference_prices_captur; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_pmdata_chainlink_btcusd_reference_prices_captur ON market_data.pmdata_chainlink_btcusd_reference_prices USING btree (capture_artifact_id, source_timestamp DESC) WHERE (capture_artifact_id IS NOT NULL);


--
-- Name: idx_market_data_pmdata_chainlink_btcusd_reference_prices_recove; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_pmdata_chainlink_btcusd_reference_prices_recove ON market_data.pmdata_chainlink_btcusd_reference_prices USING btree (feed_id, source_timestamp DESC);


--
-- Name: idx_market_data_pmdata_chainlink_btcusd_twap_artifact; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_pmdata_chainlink_btcusd_twap_artifact ON market_data.pmdata_chainlink_btcusd_twap USING btree (artifact_id, archive_row_number);


--
-- Name: idx_market_data_pmdata_chainlink_btcusd_twap_latest; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_pmdata_chainlink_btcusd_twap_latest ON market_data.pmdata_chainlink_btcusd_twap USING btree (window_seconds, source_timestamp DESC);


--
-- Name: idx_market_data_polygon_chainlink_btcusd_oracle_artifact; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polygon_chainlink_btcusd_oracle_artifact ON market_data.polygon_chainlink_btcusd_oracle_rounds USING btree (capture_artifact_id, source_timestamp DESC);


--
-- Name: idx_market_data_polygon_chainlink_btcusd_oracle_identity; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polygon_chainlink_btcusd_oracle_identity ON market_data.polygon_chainlink_btcusd_oracle_rounds USING btree (chain_id, feed_proxy_address, transaction_hash, log_index, source_timestamp DESC);


--
-- Name: idx_market_data_polygon_chainlink_btcusd_oracle_phase_boundary; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polygon_chainlink_btcusd_oracle_phase_boundary ON market_data.polygon_chainlink_btcusd_oracle_rounds USING btree (chain_id, feed_proxy_address, phase_id, block_number DESC, log_index DESC, source_timestamp DESC) INCLUDE (aggregator_round_id);


--
-- Name: idx_market_data_polygon_chainlink_btcusd_oracle_recovery; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polygon_chainlink_btcusd_oracle_recovery ON market_data.polygon_chainlink_btcusd_oracle_rounds USING btree (chain_id, feed_proxy_address, block_number DESC);


--
-- Name: idx_market_data_polymarket_btc_five_minute_contract_artifact; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_contract_artifact ON market_data.polymarket_btc_five_minute_contracts USING btree (capture_artifact_id, window_start, market_id, revision_sha256);


--
-- Name: idx_market_data_polymarket_btc_five_minute_contract_condition; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_contract_condition ON market_data.polymarket_btc_five_minute_contracts USING btree (condition_id, received_at DESC);


--
-- Name: idx_market_data_polymarket_btc_five_minute_contract_down_token; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_contract_down_token ON market_data.polymarket_btc_five_minute_contracts USING btree (down_token_id, window_start DESC);


--
-- Name: idx_market_data_polymarket_btc_five_minute_contract_event_id; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_contract_event_id ON market_data.polymarket_btc_five_minute_contracts USING btree (event_id, received_at DESC, market_id);


--
-- Name: idx_market_data_polymarket_btc_five_minute_contract_event_slug; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_contract_event_slug ON market_data.polymarket_btc_five_minute_contracts USING btree (event_slug, received_at DESC, market_id);


--
-- Name: idx_market_data_polymarket_btc_five_minute_contract_up_token; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_contract_up_token ON market_data.polymarket_btc_five_minute_contracts USING btree (up_token_id, window_start DESC);


--
-- Name: idx_market_data_polymarket_btc_five_minute_contract_window; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_contract_window ON market_data.polymarket_btc_five_minute_contracts USING btree (window_start DESC, market_id, received_at DESC);


--
-- Name: idx_market_data_polymarket_btc_five_minute_resolution_artifact; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_resolution_artifact ON market_data.polymarket_btc_five_minute_resolutions USING btree (capture_artifact_id, received_at DESC);


--
-- Name: idx_market_data_polymarket_btc_five_minute_resolution_condition; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_resolution_condition ON market_data.polymarket_btc_five_minute_resolutions USING btree (condition_id, received_at DESC);


--
-- Name: idx_market_data_polymarket_btc_five_minute_resolution_revision; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_resolution_revision ON market_data.polymarket_btc_five_minute_resolutions USING btree (market_id, revision_sha256, received_at DESC);


--
-- Name: idx_market_data_polymarket_btc_five_minute_resolution_window; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_resolution_window ON market_data.polymarket_btc_five_minute_resolutions USING btree (window_end DESC, market_id, source, received_at DESC);


--
-- Name: idx_market_data_polymarket_chainlink_btcusd_twap_artifact; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polymarket_chainlink_btcusd_twap_artifact ON market_data.polymarket_chainlink_btcusd_twap USING btree (capture_artifact_id, source_timestamp DESC);


--
-- Name: idx_market_data_polymarket_chainlink_btcusd_twap_latest; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_market_data_polymarket_chainlink_btcusd_twap_latest ON market_data.polymarket_chainlink_btcusd_twap USING btree (symbol, window_seconds, source_timestamp DESC);


--
-- Name: idx_md_binance_futures_l2_artifact; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_md_binance_futures_l2_artifact ON market_data.binance_futures_btcusdt_l2_one_second_features USING btree (artifact_id, second_start);


--
-- Name: idx_md_binance_futures_l2_available; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_md_binance_futures_l2_available ON market_data.binance_futures_btcusdt_l2_one_second_features USING btree (symbol, available_at DESC, second_start DESC);


--
-- Name: idx_md_binance_spot_l2_artifact; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_md_binance_spot_l2_artifact ON market_data.binance_spot_btcusdt_l2_one_second_features USING btree (artifact_id, second_start);


--
-- Name: idx_md_binance_spot_l2_available; Type: INDEX; Schema: market_data; Owner: -
--

CREATE INDEX idx_md_binance_spot_l2_available ON market_data.binance_spot_btcusdt_l2_one_second_features USING btree (symbol, available_at DESC, second_start DESC);


--
-- Name: btc_feature_snapshots_feature_as_of_idx; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX btc_feature_snapshots_feature_as_of_idx ON polymarket.btc_feature_snapshots USING btree (feature_as_of DESC);


--
-- Name: btc_orderbook_archive_events_provider_received_at_idx; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX btc_orderbook_archive_events_provider_received_at_idx ON polymarket.btc_orderbook_archive_events USING btree (provider_received_at DESC);


--
-- Name: fills_timestamp_utc_idx; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX fills_timestamp_utc_idx ON polymarket.fills USING btree (timestamp_utc DESC);


--
-- Name: idx_backfill_materialization_retention_replacement; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_backfill_materialization_retention_replacement ON polymarket.backfill_materialization_retention_events USING btree (replacement_artifact_id, occurred_at) WHERE (replacement_artifact_id IS NOT NULL);


--
-- Name: idx_btc_decisions_market_ts; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_decisions_market_ts ON polymarket.btc_strategy_decisions USING btree (market_id, decision_at DESC);


--
-- Name: idx_btc_decisions_process_config_candidate; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_decisions_process_config_candidate ON polymarket.btc_strategy_decisions USING btree (process_id, config_hash, market_id, decision_at, decision_id) INCLUDE (outcome, fair_probability) WHERE ((process_id IS NOT NULL) AND (action = 'buy'::text));


--
-- Name: idx_btc_decisions_run_process_at; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_decisions_run_process_at ON polymarket.btc_strategy_decisions USING btree (run_id, process_id, decision_at DESC, snapshot_id);


--
-- Name: idx_btc_features_market_ts; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_features_market_ts ON polymarket.btc_feature_snapshots USING btree (market_id, feature_as_of DESC);


--
-- Name: idx_btc_features_process_window_asof; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_features_process_window_asof ON polymarket.btc_feature_snapshots USING btree (((features ->> 'process_id'::text)), window_start, feature_as_of DESC);


--
-- Name: idx_btc_interval_active_window; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_interval_active_window ON polymarket.btc_interval_markets USING btree (active, closed, window_start DESC);


--
-- Name: idx_btc_interval_markets_condition_id; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE UNIQUE INDEX idx_btc_interval_markets_condition_id ON polymarket.btc_interval_markets USING btree (condition_id);


--
-- Name: idx_btc_interval_markets_pending_official; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_interval_markets_pending_official ON polymarket.btc_interval_markets USING btree (window_end, window_start, market_id) WHERE (official_outcome IS NULL);


--
-- Name: idx_btc_market_capacity_execution_snapshots_artifact; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_market_capacity_execution_snapshots_artifact ON polymarket.btc_market_capacity_execution_snapshots USING btree (artifact_id, sampled_at);


--
-- Name: idx_btc_market_reference_facts_artifact; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_market_reference_facts_artifact ON polymarket.btc_market_reference_facts USING btree (artifact_id, market_id, fact_type);


--
-- Name: idx_btc_market_reference_facts_effective; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_market_reference_facts_effective ON polymarket.btc_market_reference_facts USING btree (fact_type, source_effective_at, market_id);


--
-- Name: idx_btc_orderbook_archive_artifact; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_orderbook_archive_artifact ON polymarket.btc_orderbook_archive_events USING btree (artifact_id, provider_received_at, source_row_number);


--
-- Name: idx_btc_orderbook_archive_market_time; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_orderbook_archive_market_time ON polymarket.btc_orderbook_archive_events USING btree (condition_id, asset_id, source_timestamp, provider_received_at);


--
-- Name: idx_btc_paper_settlement_pending; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_paper_settlement_pending ON polymarket.btc_paper_settlement_ledger USING btree (run_id, official_resolution_received_at, settlement_id) WHERE (credit_status = 'pending'::text);


--
-- Name: idx_btc_paper_settlement_run_credited; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_paper_settlement_run_credited ON polymarket.btc_paper_settlement_ledger USING btree (run_id, credited_at DESC, settlement_id) WHERE (credit_status = 'credited'::text);


--
-- Name: idx_btc_resolution_watches_pending_deadline; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_resolution_watches_pending_deadline ON polymarket.btc_official_resolution_watches USING btree (deadline_at, market_id) WHERE (status = 'pending'::text);


--
-- Name: idx_btc_settlement_process_mode_credited; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_settlement_process_mode_credited ON polymarket.btc_paper_settlement_ledger USING btree (process_id, execution_mode, credited_at DESC, order_id, settlement_id) WHERE (credit_status = 'credited'::text);


--
-- Name: idx_btc_settlement_process_mode_pending; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_btc_settlement_process_mode_pending ON polymarket.btc_paper_settlement_ledger USING btree (process_id, run_id, execution_mode, official_resolution_received_at, settlement_id) WHERE (credit_status = 'pending'::text);


--
-- Name: idx_market_data_polymarket_btc_five_minute_orderbook_artifact; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_orderbook_artifact ON polymarket.btc_five_minute_orderbook_snapshots USING btree (capture_artifact_id, sampled_at DESC);


--
-- Name: idx_market_data_polymarket_btc_five_minute_orderbook_condition; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_orderbook_condition ON polymarket.btc_five_minute_orderbook_snapshots USING btree (condition_id, sampled_at DESC);


--
-- Name: idx_market_data_polymarket_btc_five_minute_orderbook_recovery; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_orderbook_recovery ON polymarket.btc_five_minute_orderbook_snapshots USING btree (market_id, token_id, source_timestamp DESC, sampled_at DESC);


--
-- Name: idx_market_data_polymarket_btc_five_minute_orderbook_window; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_market_data_polymarket_btc_five_minute_orderbook_window ON polymarket.btc_five_minute_orderbook_snapshots USING btree (window_start, outcome, sampled_at DESC);


--
-- Name: idx_poly_account_position_snapshots_latest; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_account_position_snapshots_latest ON polymarket.account_position_snapshots USING btree (account_address, token_id, snapshot_at DESC);


--
-- Name: idx_poly_account_reconciliation_runs_process_started; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_account_reconciliation_runs_process_started ON polymarket.account_reconciliation_runs USING btree (process_id, started_at DESC) WHERE (process_id IS NOT NULL);


--
-- Name: idx_poly_account_reconciliation_runs_started; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_account_reconciliation_runs_started ON polymarket.account_reconciliation_runs USING btree (started_at DESC);


--
-- Name: idx_poly_account_trades_recognized_exit_order; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_account_trades_recognized_exit_order ON polymarket.account_trades USING btree (linked_order_id, token_id) INCLUDE (size, applied_exit_size, timestamp_utc) WHERE ((linked_order_id IS NOT NULL) AND (side = 'sell'::text) AND (applied_exit_size > (0)::numeric));


--
-- Name: idx_poly_account_trades_token_ts; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_account_trades_token_ts ON polymarket.account_trades USING btree (account_address, token_id, timestamp_utc DESC);


--
-- Name: idx_poly_account_trades_unapplied; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_account_trades_unapplied ON polymarket.account_trades USING btree (account_address, token_id, timestamp_utc) WHERE (applied_exit_size < size);


--
-- Name: idx_poly_fills_process_ts; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_fills_process_ts ON polymarket.fills USING btree (process_id, timestamp_utc DESC) WHERE (process_id IS NOT NULL);


--
-- Name: idx_poly_live_reconciliation_runs_account_started; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_live_reconciliation_runs_account_started ON polymarket.live_reconciliation_runs USING btree (account_ref, started_at DESC) WHERE (account_ref IS NOT NULL);


--
-- Name: idx_poly_live_reconciliation_runs_process_started; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_live_reconciliation_runs_process_started ON polymarket.live_reconciliation_runs USING btree (process_id, started_at DESC) WHERE (process_id IS NOT NULL);


--
-- Name: idx_poly_live_reconciliation_runs_started; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_live_reconciliation_runs_started ON polymarket.live_reconciliation_runs USING btree (started_at DESC);


--
-- Name: idx_poly_live_venue_events_order_created; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_live_venue_events_order_created ON polymarket.live_venue_events USING btree (venue_order_id, created_at DESC) WHERE (venue_order_id IS NOT NULL);


--
-- Name: idx_poly_live_venue_events_trade_created; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_live_venue_events_trade_created ON polymarket.live_venue_events USING btree (venue_trade_id, created_at DESC) WHERE (venue_trade_id IS NOT NULL);


--
-- Name: idx_poly_orders_process_created; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_orders_process_created ON polymarket.orders USING btree (process_id, created_at DESC) WHERE (process_id IS NOT NULL);


--
-- Name: idx_poly_orders_whale_exit_position_source_created; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_orders_whale_exit_position_source_created ON polymarket.orders USING btree (((raw_payload #>> '{request,metadata,position_id}'::text[])), ((raw_payload #>> '{request,metadata,exit_source_trade_id}'::text[])), created_at DESC) WHERE ((raw_payload #>> '{request,metadata,purpose}'::text[]) = 'whale_led_exit'::text);


--
-- Name: idx_poly_trading_process_events_process_ts; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_trading_process_events_process_ts ON polymarket.trading_process_events USING btree (process_id, timestamp_utc DESC);


--
-- Name: idx_poly_trading_process_events_type_ts; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_trading_process_events_type_ts ON polymarket.trading_process_events USING btree (event_type, timestamp_utc DESC);


--
-- Name: idx_poly_trading_processes_active_live_account_ref; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_trading_processes_active_live_account_ref ON polymarket.trading_processes USING btree (lower(btrim((config #>> '{execution,account_ref}'::text[]))), process_id) WHERE (enabled AND (status = ANY (ARRAY['starting'::text, 'running'::text, 'stopping'::text])) AND ((config #>> '{execution,mode}'::text[]) = 'live'::text));


--
-- Name: idx_poly_trading_processes_status_heartbeat; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_trading_processes_status_heartbeat ON polymarket.trading_processes USING btree (enabled, status, heartbeat_at DESC);


--
-- Name: idx_poly_trading_processes_type_scope_started; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_trading_processes_type_scope_started ON polymarket.trading_processes USING btree (process_type, process_scope, started_at DESC);


--
-- Name: idx_poly_trading_processes_updated; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_poly_trading_processes_updated ON polymarket.trading_processes USING btree (updated_at DESC);


--
-- Name: idx_polymarket_fills_order_ts; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_polymarket_fills_order_ts ON polymarket.fills USING btree (order_id, timestamp_utc DESC);


--
-- Name: idx_polymarket_orders_market_state; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_polymarket_orders_market_state ON polymarket.orders USING btree (market_id, state);


--
-- Name: idx_polymarket_risk_type_ts; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_polymarket_risk_type_ts ON polymarket.risk_events USING btree (event_type, timestamp_utc DESC);


--
-- Name: idx_reference_ticks_source_ts; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX idx_reference_ticks_source_ts ON polymarket.reference_price_ticks USING btree (source, symbol, source_timestamp DESC);


--
-- Name: reference_price_ticks_source_timestamp_idx; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX reference_price_ticks_source_timestamp_idx ON polymarket.reference_price_ticks USING btree (source_timestamp DESC);


--
-- Name: risk_events_timestamp_utc_idx; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX risk_events_timestamp_utc_idx ON polymarket.risk_events USING btree (timestamp_utc DESC);


--
-- Name: trading_process_events_timestamp_utc_idx; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE INDEX trading_process_events_timestamp_utc_idx ON polymarket.trading_process_events USING btree (timestamp_utc DESC);


--
-- Name: uq_btc_decision_entry_per_run_market; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE UNIQUE INDEX uq_btc_decision_entry_per_run_market ON polymarket.btc_strategy_decisions USING btree (run_id, market_id) WHERE ((action = 'buy'::text) AND (status = ANY (ARRAY['approved'::text, 'submitted'::text, 'filled'::text])));


--
-- Name: uq_btc_features_market_hash_ts; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE UNIQUE INDEX uq_btc_features_market_hash_ts ON polymarket.btc_feature_snapshots USING btree (market_id, feature_hash, feature_as_of);


--
-- Name: uq_btc_interval_condition_id; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE UNIQUE INDEX uq_btc_interval_condition_id ON polymarket.btc_interval_markets USING btree (condition_id);


--
-- Name: uq_poly_account_trades_identity; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE UNIQUE INDEX uq_poly_account_trades_identity ON polymarket.account_trades USING btree (account_address, token_id, side, price, size, timestamp_utc, COALESCE(transaction_hash, ''::text), COALESCE(venue_trade_id, ''::text));


--
-- Name: uq_poly_live_venue_events_hash; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE UNIQUE INDEX uq_poly_live_venue_events_hash ON polymarket.live_venue_events USING btree (event_hash);


--
-- Name: uq_poly_orders_client_order_id; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE UNIQUE INDEX uq_poly_orders_client_order_id ON polymarket.orders USING btree (client_order_id);


--
-- Name: uq_poly_orders_venue_order_id; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE UNIQUE INDEX uq_poly_orders_venue_order_id ON polymarket.orders USING btree (venue_order_id) WHERE (venue_order_id IS NOT NULL);


--
-- Name: uq_poly_trading_processes_active_key; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE UNIQUE INDEX uq_poly_trading_processes_active_key ON polymarket.trading_processes USING btree (process_type, process_scope, process_key) WHERE ((process_key IS NOT NULL) AND (enabled = true) AND (status <> ALL (ARRAY['stopped'::text, 'failed'::text, 'expired'::text])));


--
-- Name: uq_poly_trading_processes_key; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE UNIQUE INDEX uq_poly_trading_processes_key ON polymarket.trading_processes USING btree (process_type, process_scope, process_key) WHERE (process_key IS NOT NULL);


--
-- Name: uq_reference_ticks_dedup; Type: INDEX; Schema: polymarket; Owner: -
--

CREATE UNIQUE INDEX uq_reference_ticks_dedup ON polymarket.reference_price_ticks USING btree (source, dedup_key, source_timestamp);


--
-- Name: profiles trg_notify_ingester_profile_change; Type: TRIGGER; Schema: ingester; Owner: -
--

CREATE TRIGGER trg_notify_ingester_profile_change AFTER INSERT OR UPDATE OF desired_state, desired_generation, config_schema_version, config ON ingester.profiles FOR EACH ROW EXECUTE FUNCTION ingester.notify_profile_change();


--
-- Name: backfill_artifacts trg_reject_completed_backfill_artifact_change; Type: TRIGGER; Schema: ingester; Owner: -
--

CREATE TRIGGER trg_reject_completed_backfill_artifact_change BEFORE DELETE OR UPDATE ON ingester.backfill_artifacts FOR EACH ROW EXECUTE FUNCTION ingester.reject_completed_backfill_artifact_change();


--
-- Name: data_gaps trg_reject_data_gap_identity_change; Type: TRIGGER; Schema: ingester; Owner: -
--

CREATE TRIGGER trg_reject_data_gap_identity_change BEFORE INSERT OR DELETE OR UPDATE ON ingester.data_gaps FOR EACH ROW EXECUTE FUNCTION ingester.reject_data_gap_identity_change();


--
-- Name: capture_artifacts trg_reject_terminal_capture_artifact_change; Type: TRIGGER; Schema: ingester; Owner: -
--

CREATE TRIGGER trg_reject_terminal_capture_artifact_change BEFORE DELETE OR UPDATE ON ingester.capture_artifacts FOR EACH ROW EXECUTE FUNCTION ingester.reject_terminal_artifact_change();


--
-- Name: profiles trg_validate_ingester_profile_change; Type: TRIGGER; Schema: ingester; Owner: -
--

CREATE TRIGGER trg_validate_ingester_profile_change BEFORE UPDATE ON ingester.profiles FOR EACH ROW EXECUTE FUNCTION ingester.validate_profile_change();


--
-- Name: binance_futures_btcusdt_l2_one_second_features trg_reject_market_data_binance_futures_btcusdt_l2_one_second_fe; Type: TRIGGER; Schema: market_data; Owner: -
--

CREATE TRIGGER trg_reject_market_data_binance_futures_btcusdt_l2_one_second_fe BEFORE DELETE OR UPDATE ON market_data.binance_futures_btcusdt_l2_one_second_features FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: binance_futures_btcusdt_open_interest trg_reject_market_data_binance_futures_open_interest_change; Type: TRIGGER; Schema: market_data; Owner: -
--

CREATE TRIGGER trg_reject_market_data_binance_futures_open_interest_change BEFORE DELETE OR UPDATE ON market_data.binance_futures_btcusdt_open_interest FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: binance_spot_btcusdt_aggregate_trades trg_reject_market_data_binance_spot_aggregate_trade_change; Type: TRIGGER; Schema: market_data; Owner: -
--

CREATE TRIGGER trg_reject_market_data_binance_spot_aggregate_trade_change BEFORE DELETE OR UPDATE ON market_data.binance_spot_btcusdt_aggregate_trades FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: binance_spot_btcusdt_l2_one_second_features trg_reject_market_data_binance_spot_btcusdt_l2_one_second_featu; Type: TRIGGER; Schema: market_data; Owner: -
--

CREATE TRIGGER trg_reject_market_data_binance_spot_btcusdt_l2_one_second_featu BEFORE DELETE OR UPDATE ON market_data.binance_spot_btcusdt_l2_one_second_features FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: binance_spot_btcusdt_l2_snapshots trg_reject_market_data_binance_spot_l2_snapshot_change; Type: TRIGGER; Schema: market_data; Owner: -
--

CREATE TRIGGER trg_reject_market_data_binance_spot_l2_snapshot_change BEFORE DELETE OR UPDATE ON market_data.binance_spot_btcusdt_l2_snapshots FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: binance_spot_btcusdt_one_second_ohlcv trg_reject_market_data_binance_spot_one_second_ohlcv_change; Type: TRIGGER; Schema: market_data; Owner: -
--

CREATE TRIGGER trg_reject_market_data_binance_spot_one_second_ohlcv_change BEFORE DELETE OR UPDATE ON market_data.binance_spot_btcusdt_one_second_ohlcv FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: chainlink_btcusd_one_minute_candles trg_reject_market_data_chainlink_btcusd_one_minute_candles_chan; Type: TRIGGER; Schema: market_data; Owner: -
--

CREATE TRIGGER trg_reject_market_data_chainlink_btcusd_one_minute_candles_chan BEFORE DELETE OR UPDATE ON market_data.chainlink_btcusd_one_minute_candles FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: chainlink_btcusd_reference_prices trg_reject_market_data_chainlink_btcusd_reference_prices_change; Type: TRIGGER; Schema: market_data; Owner: -
--

CREATE TRIGGER trg_reject_market_data_chainlink_btcusd_reference_prices_change BEFORE DELETE OR UPDATE ON market_data.chainlink_btcusd_reference_prices FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: pmdata_chainlink_btcusd_reference_prices trg_reject_market_data_pmdata_chainlink_btcusd_reference_prices; Type: TRIGGER; Schema: market_data; Owner: -
--

CREATE TRIGGER trg_reject_market_data_pmdata_chainlink_btcusd_reference_prices BEFORE DELETE OR UPDATE ON market_data.pmdata_chainlink_btcusd_reference_prices FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: polygon_chainlink_btcusd_oracle_rounds trg_reject_market_data_polygon_chainlink_btcusd_oracle_change; Type: TRIGGER; Schema: market_data; Owner: -
--

CREATE TRIGGER trg_reject_market_data_polygon_chainlink_btcusd_oracle_change BEFORE DELETE OR UPDATE ON market_data.polygon_chainlink_btcusd_oracle_rounds FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: pmdata_chainlink_btcusd_twap trg_reject_md_pmdata_chainlink_btcusd_twap_change; Type: TRIGGER; Schema: market_data; Owner: -
--

CREATE TRIGGER trg_reject_md_pmdata_chainlink_btcusd_twap_change BEFORE DELETE OR UPDATE ON market_data.pmdata_chainlink_btcusd_twap FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: polymarket_btc_five_minute_contracts trg_reject_md_polymarket_btc_five_minute_contract_change; Type: TRIGGER; Schema: market_data; Owner: -
--

CREATE TRIGGER trg_reject_md_polymarket_btc_five_minute_contract_change BEFORE DELETE OR UPDATE ON market_data.polymarket_btc_five_minute_contracts FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: polymarket_btc_five_minute_resolutions trg_reject_md_polymarket_btc_five_minute_resolution_change; Type: TRIGGER; Schema: market_data; Owner: -
--

CREATE TRIGGER trg_reject_md_polymarket_btc_five_minute_resolution_change BEFORE DELETE OR UPDATE ON market_data.polymarket_btc_five_minute_resolutions FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: polymarket_chainlink_btcusd_twap trg_reject_md_polymarket_chainlink_btcusd_twap_change; Type: TRIGGER; Schema: market_data; Owner: -
--

CREATE TRIGGER trg_reject_md_polymarket_chainlink_btcusd_twap_change BEFORE DELETE OR UPDATE ON market_data.polymarket_chainlink_btcusd_twap FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: binance_futures_btcusdt_l2_one_second_features ts_insert_blocker; Type: TRIGGER; Schema: market_data; Owner: -
--



--
-- Name: binance_futures_btcusdt_open_interest ts_insert_blocker; Type: TRIGGER; Schema: market_data; Owner: -
--



--
-- Name: binance_spot_btcusdt_aggregate_trades ts_insert_blocker; Type: TRIGGER; Schema: market_data; Owner: -
--



--
-- Name: binance_spot_btcusdt_l2_one_second_features ts_insert_blocker; Type: TRIGGER; Schema: market_data; Owner: -
--



--
-- Name: binance_spot_btcusdt_l2_snapshots ts_insert_blocker; Type: TRIGGER; Schema: market_data; Owner: -
--



--
-- Name: binance_spot_btcusdt_one_second_ohlcv ts_insert_blocker; Type: TRIGGER; Schema: market_data; Owner: -
--



--
-- Name: chainlink_btcusd_one_minute_candles ts_insert_blocker; Type: TRIGGER; Schema: market_data; Owner: -
--



--
-- Name: chainlink_btcusd_reference_prices ts_insert_blocker; Type: TRIGGER; Schema: market_data; Owner: -
--



--
-- Name: pmdata_chainlink_btcusd_reference_prices ts_insert_blocker; Type: TRIGGER; Schema: market_data; Owner: -
--



--
-- Name: pmdata_chainlink_btcusd_twap ts_insert_blocker; Type: TRIGGER; Schema: market_data; Owner: -
--



--
-- Name: polygon_chainlink_btcusd_oracle_rounds ts_insert_blocker; Type: TRIGGER; Schema: market_data; Owner: -
--



--
-- Name: polymarket_chainlink_btcusd_twap ts_insert_blocker; Type: TRIGGER; Schema: market_data; Owner: -
--



--
-- Name: backfill_materialization_retention_events trg_reject_backfill_retention_event_change; Type: TRIGGER; Schema: polymarket; Owner: -
--

CREATE TRIGGER trg_reject_backfill_retention_event_change BEFORE DELETE OR UPDATE ON polymarket.backfill_materialization_retention_events FOR EACH ROW EXECUTE FUNCTION polymarket.reject_backfill_retention_event_change();


--
-- Name: btc_market_capacity_execution_snapshots trg_reject_btc_market_capacity_execution_snapshot_change; Type: TRIGGER; Schema: polymarket; Owner: -
--

CREATE TRIGGER trg_reject_btc_market_capacity_execution_snapshot_change BEFORE DELETE OR UPDATE ON polymarket.btc_market_capacity_execution_snapshots FOR EACH ROW EXECUTE FUNCTION polymarket.reject_btc_market_execution_snapshot_change();


--
-- Name: btc_orderbook_archive_events trg_reject_btc_orderbook_archive_event_change; Type: TRIGGER; Schema: polymarket; Owner: -
--

CREATE TRIGGER trg_reject_btc_orderbook_archive_event_change BEFORE DELETE OR UPDATE ON polymarket.btc_orderbook_archive_events FOR EACH ROW EXECUTE FUNCTION polymarket.reject_historical_market_event_change();


--
-- Name: btc_market_reference_facts trg_reject_immutable_btc_reference_fact_change; Type: TRIGGER; Schema: polymarket; Owner: -
--

CREATE TRIGGER trg_reject_immutable_btc_reference_fact_change BEFORE DELETE OR UPDATE ON polymarket.btc_market_reference_facts FOR EACH ROW EXECUTE FUNCTION polymarket.reject_immutable_btc_reference_fact_change();


--
-- Name: btc_five_minute_orderbook_snapshots trg_reject_md_polymarket_btc_five_minute_orderbook_change; Type: TRIGGER; Schema: polymarket; Owner: -
--

CREATE TRIGGER trg_reject_md_polymarket_btc_five_minute_orderbook_change BEFORE DELETE OR UPDATE ON polymarket.btc_five_minute_orderbook_snapshots FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();


--
-- Name: btc_feature_snapshots ts_insert_blocker; Type: TRIGGER; Schema: polymarket; Owner: -
--



--
-- Name: btc_five_minute_orderbook_snapshots ts_insert_blocker; Type: TRIGGER; Schema: polymarket; Owner: -
--



--
-- Name: btc_market_capacity_execution_snapshots ts_insert_blocker; Type: TRIGGER; Schema: polymarket; Owner: -
--



--
-- Name: btc_orderbook_archive_events ts_insert_blocker; Type: TRIGGER; Schema: polymarket; Owner: -
--



--
-- Name: fills ts_insert_blocker; Type: TRIGGER; Schema: polymarket; Owner: -
--



--
-- Name: reference_price_ticks ts_insert_blocker; Type: TRIGGER; Schema: polymarket; Owner: -
--



--
-- Name: risk_events ts_insert_blocker; Type: TRIGGER; Schema: polymarket; Owner: -
--



--
-- Name: trading_process_events ts_insert_blocker; Type: TRIGGER; Schema: polymarket; Owner: -
--



--
-- Name: backfill_artifacts backfill_artifacts_job_id_fkey; Type: FK CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.backfill_artifacts
    ADD CONSTRAINT backfill_artifacts_job_id_fkey FOREIGN KEY (job_id) REFERENCES ingester.backfill_jobs(job_id) ON DELETE RESTRICT;


--
-- Name: backfill_job_events backfill_job_events_job_id_fkey; Type: FK CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.backfill_job_events
    ADD CONSTRAINT backfill_job_events_job_id_fkey FOREIGN KEY (job_id) REFERENCES ingester.backfill_jobs(job_id) ON DELETE RESTRICT;


--
-- Name: backfill_jobs backfill_jobs_parent_job_id_fkey; Type: FK CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.backfill_jobs
    ADD CONSTRAINT backfill_jobs_parent_job_id_fkey FOREIGN KEY (parent_job_id) REFERENCES ingester.backfill_jobs(job_id) ON DELETE RESTRICT;


--
-- Name: capture_artifacts capture_artifacts_strategy_key_fkey; Type: FK CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.capture_artifacts
    ADD CONSTRAINT capture_artifacts_strategy_key_fkey FOREIGN KEY (strategy_key) REFERENCES ingester.profiles(strategy_key) ON DELETE RESTRICT;


--
-- Name: data_gaps data_gaps_strategy_key_fkey; Type: FK CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.data_gaps
    ADD CONSTRAINT data_gaps_strategy_key_fkey FOREIGN KEY (strategy_key) REFERENCES ingester.profiles(strategy_key) ON DELETE RESTRICT;


--
-- Name: drain_objects drain_objects_job_id_fkey; Type: FK CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.drain_objects
    ADD CONSTRAINT drain_objects_job_id_fkey FOREIGN KEY (job_id) REFERENCES ingester.drain_jobs(job_id) ON DELETE RESTRICT;


--
-- Name: data_gaps fk_ingester_data_gap_detected_artifact; Type: FK CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.data_gaps
    ADD CONSTRAINT fk_ingester_data_gap_detected_artifact FOREIGN KEY (strategy_key, detected_artifact_id) REFERENCES ingester.capture_artifacts(strategy_key, artifact_id) ON DELETE RESTRICT;


--
-- Name: data_gaps fk_ingester_data_gap_repair_artifact; Type: FK CONSTRAINT; Schema: ingester; Owner: -
--

ALTER TABLE ONLY ingester.data_gaps
    ADD CONSTRAINT fk_ingester_data_gap_repair_artifact FOREIGN KEY (strategy_key, repair_artifact_id) REFERENCES ingester.capture_artifacts(strategy_key, artifact_id) ON DELETE RESTRICT;


--
-- Name: binance_futures_btcusdt_l2_one_second_features fk_market_data_binance_futures_btcusdt_l2_one_second_features_a; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.binance_futures_btcusdt_l2_one_second_features
    ADD CONSTRAINT fk_market_data_binance_futures_btcusdt_l2_one_second_features_a FOREIGN KEY (artifact_id) REFERENCES ingester.backfill_artifacts(artifact_id) ON DELETE RESTRICT;


--
-- Name: binance_futures_btcusdt_open_interest fk_market_data_binance_futures_open_interest_artifact; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.binance_futures_btcusdt_open_interest
    ADD CONSTRAINT fk_market_data_binance_futures_open_interest_artifact FOREIGN KEY (strategy_key, capture_artifact_id) REFERENCES ingester.capture_artifacts(strategy_key, artifact_id) ON DELETE RESTRICT;


--
-- Name: binance_spot_btcusdt_aggregate_trades fk_market_data_binance_spot_aggregate_trade_artifact; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.binance_spot_btcusdt_aggregate_trades
    ADD CONSTRAINT fk_market_data_binance_spot_aggregate_trade_artifact FOREIGN KEY (strategy_key, capture_artifact_id) REFERENCES ingester.capture_artifacts(strategy_key, artifact_id) ON DELETE RESTRICT;


--
-- Name: binance_spot_btcusdt_l2_one_second_features fk_market_data_binance_spot_btcusdt_l2_one_second_features_arti; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.binance_spot_btcusdt_l2_one_second_features
    ADD CONSTRAINT fk_market_data_binance_spot_btcusdt_l2_one_second_features_arti FOREIGN KEY (artifact_id) REFERENCES ingester.backfill_artifacts(artifact_id) ON DELETE RESTRICT;


--
-- Name: binance_spot_btcusdt_l2_snapshots fk_market_data_binance_spot_l2_snapshot_artifact; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.binance_spot_btcusdt_l2_snapshots
    ADD CONSTRAINT fk_market_data_binance_spot_l2_snapshot_artifact FOREIGN KEY (strategy_key, capture_artifact_id) REFERENCES ingester.capture_artifacts(strategy_key, artifact_id) ON DELETE RESTRICT;


--
-- Name: binance_spot_btcusdt_one_second_ohlcv fk_market_data_binance_spot_one_second_ohlcv_artifact; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.binance_spot_btcusdt_one_second_ohlcv
    ADD CONSTRAINT fk_market_data_binance_spot_one_second_ohlcv_artifact FOREIGN KEY (strategy_key, capture_artifact_id) REFERENCES ingester.capture_artifacts(strategy_key, artifact_id) ON DELETE RESTRICT;


--
-- Name: chainlink_btcusd_one_minute_candles fk_market_data_chainlink_btcusd_one_minute_candles_artifact; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.chainlink_btcusd_one_minute_candles
    ADD CONSTRAINT fk_market_data_chainlink_btcusd_one_minute_candles_artifact FOREIGN KEY (strategy_key, capture_artifact_id) REFERENCES ingester.capture_artifacts(strategy_key, artifact_id) ON DELETE RESTRICT;


--
-- Name: chainlink_btcusd_reference_prices fk_market_data_chainlink_btcusd_reference_prices_backfill_artif; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.chainlink_btcusd_reference_prices
    ADD CONSTRAINT fk_market_data_chainlink_btcusd_reference_prices_backfill_artif FOREIGN KEY (backfill_artifact_id) REFERENCES ingester.backfill_artifacts(artifact_id) ON DELETE RESTRICT;


--
-- Name: chainlink_btcusd_reference_prices fk_market_data_chainlink_btcusd_reference_prices_capture_artifa; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.chainlink_btcusd_reference_prices
    ADD CONSTRAINT fk_market_data_chainlink_btcusd_reference_prices_capture_artifa FOREIGN KEY (strategy_key, capture_artifact_id) REFERENCES ingester.capture_artifacts(strategy_key, artifact_id) ON DELETE RESTRICT;


--
-- Name: pmdata_chainlink_btcusd_reference_prices fk_market_data_pmdata_chainlink_btcusd_reference_prices_backfil; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.pmdata_chainlink_btcusd_reference_prices
    ADD CONSTRAINT fk_market_data_pmdata_chainlink_btcusd_reference_prices_backfil FOREIGN KEY (backfill_artifact_id) REFERENCES ingester.backfill_artifacts(artifact_id) ON DELETE RESTRICT;


--
-- Name: pmdata_chainlink_btcusd_reference_prices fk_market_data_pmdata_chainlink_btcusd_reference_prices_capture; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.pmdata_chainlink_btcusd_reference_prices
    ADD CONSTRAINT fk_market_data_pmdata_chainlink_btcusd_reference_prices_capture FOREIGN KEY (strategy_key, capture_artifact_id) REFERENCES ingester.capture_artifacts(strategy_key, artifact_id) ON DELETE RESTRICT;


--
-- Name: pmdata_chainlink_btcusd_twap fk_market_data_pmdata_chainlink_btcusd_twap_artifact; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.pmdata_chainlink_btcusd_twap
    ADD CONSTRAINT fk_market_data_pmdata_chainlink_btcusd_twap_artifact FOREIGN KEY (artifact_id) REFERENCES ingester.backfill_artifacts(artifact_id) ON DELETE RESTRICT;


--
-- Name: polygon_chainlink_btcusd_oracle_rounds fk_market_data_polygon_chainlink_btcusd_oracle_rounds_artifact; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.polygon_chainlink_btcusd_oracle_rounds
    ADD CONSTRAINT fk_market_data_polygon_chainlink_btcusd_oracle_rounds_artifact FOREIGN KEY (strategy_key, capture_artifact_id) REFERENCES ingester.capture_artifacts(strategy_key, artifact_id) ON DELETE RESTRICT;


--
-- Name: polymarket_btc_five_minute_contracts fk_market_data_polymarket_btc_five_minute_contracts_artifact; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.polymarket_btc_five_minute_contracts
    ADD CONSTRAINT fk_market_data_polymarket_btc_five_minute_contracts_artifact FOREIGN KEY (strategy_key, capture_artifact_id) REFERENCES ingester.capture_artifacts(strategy_key, artifact_id) ON DELETE RESTRICT;


--
-- Name: polymarket_btc_five_minute_resolutions fk_market_data_polymarket_btc_five_minute_resolution_artifact; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.polymarket_btc_five_minute_resolutions
    ADD CONSTRAINT fk_market_data_polymarket_btc_five_minute_resolution_artifact FOREIGN KEY (strategy_key, capture_artifact_id) REFERENCES ingester.capture_artifacts(strategy_key, artifact_id) ON DELETE RESTRICT;


--
-- Name: polymarket_chainlink_btcusd_twap fk_market_data_polymarket_chainlink_btcusd_twap_artifact; Type: FK CONSTRAINT; Schema: market_data; Owner: -
--

ALTER TABLE ONLY market_data.polymarket_chainlink_btcusd_twap
    ADD CONSTRAINT fk_market_data_polymarket_chainlink_btcusd_twap_artifact FOREIGN KEY (strategy_key, capture_artifact_id) REFERENCES ingester.capture_artifacts(strategy_key, artifact_id) ON DELETE RESTRICT;


--
-- Name: account_trades account_trades_linked_order_id_fkey; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.account_trades
    ADD CONSTRAINT account_trades_linked_order_id_fkey FOREIGN KEY (linked_order_id) REFERENCES polymarket.orders(order_id) ON DELETE SET NULL;


--
-- Name: backfill_materialization_retention_events backfill_materialization_retention_even_source_artifact_id_fkey; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.backfill_materialization_retention_events
    ADD CONSTRAINT backfill_materialization_retention_even_source_artifact_id_fkey FOREIGN KEY (source_artifact_id) REFERENCES ingester.backfill_artifacts(artifact_id) ON DELETE RESTRICT;


--
-- Name: backfill_materialization_retention_events backfill_materialization_retention_replacement_artifact_id_fkey; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.backfill_materialization_retention_events
    ADD CONSTRAINT backfill_materialization_retention_replacement_artifact_id_fkey FOREIGN KEY (replacement_artifact_id) REFERENCES ingester.backfill_artifacts(artifact_id) ON DELETE RESTRICT;


--
-- Name: btc_market_reference_facts btc_market_reference_facts_artifact_id_fkey; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_market_reference_facts
    ADD CONSTRAINT btc_market_reference_facts_artifact_id_fkey FOREIGN KEY (artifact_id) REFERENCES ingester.backfill_artifacts(artifact_id) ON DELETE RESTRICT;


--
-- Name: btc_market_reference_facts btc_market_reference_facts_market_id_fkey; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_market_reference_facts
    ADD CONSTRAINT btc_market_reference_facts_market_id_fkey FOREIGN KEY (market_id) REFERENCES polymarket.btc_interval_markets(market_id) ON DELETE RESTRICT;


--
-- Name: btc_official_resolution_watches btc_official_resolution_watches_market_id_fkey; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_official_resolution_watches
    ADD CONSTRAINT btc_official_resolution_watches_market_id_fkey FOREIGN KEY (market_id) REFERENCES polymarket.btc_interval_markets(market_id) ON DELETE CASCADE;


--
-- Name: btc_orderbook_archive_events btc_orderbook_archive_events_artifact_id_fkey; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_orderbook_archive_events
    ADD CONSTRAINT btc_orderbook_archive_events_artifact_id_fkey FOREIGN KEY (artifact_id) REFERENCES ingester.backfill_artifacts(artifact_id) ON DELETE RESTRICT;


--
-- Name: btc_paper_settlement_ledger btc_paper_settlement_ledger_market_id_fkey; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_paper_settlement_ledger
    ADD CONSTRAINT btc_paper_settlement_ledger_market_id_fkey FOREIGN KEY (market_id) REFERENCES polymarket.btc_interval_markets(market_id) ON DELETE RESTRICT;


--
-- Name: btc_paper_settlement_ledger btc_paper_settlement_ledger_order_id_fkey; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_paper_settlement_ledger
    ADD CONSTRAINT btc_paper_settlement_ledger_order_id_fkey FOREIGN KEY (order_id) REFERENCES polymarket.orders(order_id) ON DELETE RESTRICT;


--
-- Name: btc_paper_settlement_ledger btc_paper_settlement_ledger_process_id_fkey; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_paper_settlement_ledger
    ADD CONSTRAINT btc_paper_settlement_ledger_process_id_fkey FOREIGN KEY (process_id) REFERENCES polymarket.trading_processes(process_id) ON DELETE RESTRICT;


--
-- Name: btc_market_capacity_execution_snapshots fk_btc_market_capacity_execution_snapshots_artifact; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_market_capacity_execution_snapshots
    ADD CONSTRAINT fk_btc_market_capacity_execution_snapshots_artifact FOREIGN KEY (artifact_id) REFERENCES ingester.backfill_artifacts(artifact_id) ON DELETE RESTRICT;


--
-- Name: btc_market_capacity_execution_snapshots fk_btc_market_capacity_execution_snapshots_market; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_market_capacity_execution_snapshots
    ADD CONSTRAINT fk_btc_market_capacity_execution_snapshots_market FOREIGN KEY (market_id) REFERENCES polymarket.btc_interval_markets(market_id) ON DELETE RESTRICT;


--
-- Name: btc_paper_settlement_ledger fk_btc_paper_settlement_official_identity; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_paper_settlement_ledger
    ADD CONSTRAINT fk_btc_paper_settlement_official_identity FOREIGN KEY (market_id, official_outcome, official_winning_token_id, official_resolution_received_at, official_resolution_source) REFERENCES polymarket.btc_interval_markets(market_id, official_outcome, official_winning_token_id, official_resolution_received_at, official_resolution_source) ON DELETE RESTRICT;


--
-- Name: btc_five_minute_orderbook_snapshots fk_market_data_polymarket_btc_five_minute_orderbook_artifact; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.btc_five_minute_orderbook_snapshots
    ADD CONSTRAINT fk_market_data_polymarket_btc_five_minute_orderbook_artifact FOREIGN KEY (strategy_key, capture_artifact_id) REFERENCES ingester.capture_artifacts(strategy_key, artifact_id) ON DELETE RESTRICT;


--
-- Name: account_reconciliation_runs fk_poly_account_reconciliation_runs_process; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.account_reconciliation_runs
    ADD CONSTRAINT fk_poly_account_reconciliation_runs_process FOREIGN KEY (process_id) REFERENCES polymarket.trading_processes(process_id) ON DELETE RESTRICT;


--
-- Name: fills fk_poly_fills_process; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.fills
    ADD CONSTRAINT fk_poly_fills_process FOREIGN KEY (process_id) REFERENCES polymarket.trading_processes(process_id) ON DELETE SET NULL NOT VALID;


--
-- Name: live_reconciliation_runs fk_poly_live_reconciliation_runs_process; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.live_reconciliation_runs
    ADD CONSTRAINT fk_poly_live_reconciliation_runs_process FOREIGN KEY (process_id) REFERENCES polymarket.trading_processes(process_id) ON DELETE RESTRICT;


--
-- Name: orders fk_poly_orders_process; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.orders
    ADD CONSTRAINT fk_poly_orders_process FOREIGN KEY (process_id) REFERENCES polymarket.trading_processes(process_id) ON DELETE SET NULL NOT VALID;


--
-- Name: trading_process_events trading_process_events_process_id_fkey; Type: FK CONSTRAINT; Schema: polymarket; Owner: -
--

ALTER TABLE ONLY polymarket.trading_process_events
    ADD CONSTRAINT trading_process_events_process_id_fkey FOREIGN KEY (process_id) REFERENCES polymarket.trading_processes(process_id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

-- Restore TimescaleDB semantics after the ordinary PostgreSQL objects exist.
SET search_path = public, pg_catalog;
SELECT create_hypertable('market_data.binance_futures_btcusdt_l2_one_second_features', 'second_start', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('market_data.binance_futures_btcusdt_open_interest', 'source_timestamp', chunk_time_interval => '7 days'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('market_data.binance_spot_btcusdt_aggregate_trades', 'trade_timestamp', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('market_data.binance_spot_btcusdt_l2_one_second_features', 'second_start', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('market_data.binance_spot_btcusdt_l2_snapshots', 'source_timestamp', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('market_data.binance_spot_btcusdt_one_second_ohlcv', 'open_timestamp', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('market_data.chainlink_btcusd_one_minute_candles', 'open_timestamp', chunk_time_interval => '7 days'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('market_data.chainlink_btcusd_reference_prices', 'source_timestamp', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('market_data.pmdata_chainlink_btcusd_reference_prices', 'source_timestamp', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('market_data.pmdata_chainlink_btcusd_twap', 'source_timestamp', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('market_data.polygon_chainlink_btcusd_oracle_rounds', 'source_timestamp', chunk_time_interval => '7 days'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('market_data.polymarket_chainlink_btcusd_twap', 'source_timestamp', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('polymarket.btc_feature_snapshots', 'feature_as_of', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('polymarket.btc_five_minute_orderbook_snapshots', 'sampled_at', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('polymarket.btc_market_capacity_execution_snapshots', 'sampled_at', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('polymarket.btc_orderbook_archive_events', 'provider_received_at', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('polymarket.fills', 'timestamp_utc', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('polymarket.reference_price_ticks', 'source_timestamp', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('polymarket.risk_events', 'timestamp_utc', chunk_time_interval => '1 day'::interval, migrate_data => false, create_default_indexes => false);
SELECT create_hypertable('polymarket.trading_process_events', 'timestamp_utc', chunk_time_interval => '7 days'::interval, migrate_data => false, create_default_indexes => false);
ALTER TABLE market_data.binance_futures_btcusdt_l2_one_second_features SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'symbol, artifact_id', timescaledb.compress_orderby = 'second_start ASC');
ALTER TABLE market_data.binance_futures_btcusdt_open_interest SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'symbol, source, period_seconds, strategy_key, capture_artifact_id', timescaledb.compress_orderby = 'source_timestamp ASC');
ALTER TABLE market_data.binance_spot_btcusdt_aggregate_trades SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'symbol, source, strategy_key, capture_artifact_id', timescaledb.compress_orderby = 'trade_timestamp ASC, aggregate_trade_id ASC');
ALTER TABLE market_data.binance_spot_btcusdt_l2_one_second_features SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'symbol, artifact_id', timescaledb.compress_orderby = 'second_start ASC');
ALTER TABLE market_data.binance_spot_btcusdt_l2_snapshots SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'symbol, source, sampling_policy_sha256, strategy_key, capture_artifact_id', timescaledb.compress_orderby = 'source_timestamp ASC, source_update_id ASC');
ALTER TABLE market_data.binance_spot_btcusdt_one_second_ohlcv SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'symbol, source, strategy_key, capture_artifact_id', timescaledb.compress_orderby = 'open_timestamp ASC');
ALTER TABLE market_data.chainlink_btcusd_one_minute_candles SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'symbol, source, strategy_key, capture_artifact_id', timescaledb.compress_orderby = 'open_timestamp ASC');
ALTER TABLE market_data.chainlink_btcusd_reference_prices SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'feed_id, source, strategy_key, capture_artifact_id, backfill_artifact_id', timescaledb.compress_orderby = 'source_timestamp ASC, report_sha256 ASC');
ALTER TABLE market_data.pmdata_chainlink_btcusd_reference_prices SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'feed_id, source, strategy_key, capture_artifact_id, backfill_artifact_id', timescaledb.compress_orderby = 'source_timestamp ASC, report_sha256 ASC');
ALTER TABLE market_data.polygon_chainlink_btcusd_oracle_rounds SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'chain_id, feed_proxy_address, strategy_key, capture_artifact_id', timescaledb.compress_orderby = 'source_timestamp ASC, block_number ASC, log_index ASC');
ALTER TABLE market_data.polymarket_chainlink_btcusd_twap SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'symbol, window_seconds, strategy_key, capture_artifact_id', timescaledb.compress_orderby = 'source_timestamp ASC, published_at ASC');
ALTER TABLE polymarket.btc_feature_snapshots SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'market_id, feature_schema_version', timescaledb.compress_orderby = 'feature_as_of DESC, snapshot_id ASC');
ALTER TABLE polymarket.btc_five_minute_orderbook_snapshots SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'market_id, token_id, sampling_policy_sha256, strategy_key, capture_artifact_id', timescaledb.compress_orderby = 'sampled_at ASC, source_timestamp ASC, ingest_sequence ASC');
ALTER TABLE polymarket.btc_market_capacity_execution_snapshots SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'market_id, artifact_id', timescaledb.compress_orderby = 'sampled_at ASC');
ALTER TABLE polymarket.btc_orderbook_archive_events SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'condition_id, asset_id, artifact_id', timescaledb.compress_orderby = 'provider_received_at ASC, source_row_number ASC');
ALTER TABLE polymarket.fills SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'token_id, source, process_id', timescaledb.compress_orderby = 'timestamp_utc DESC, fill_id ASC');
ALTER TABLE polymarket.reference_price_ticks SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'source, symbol', timescaledb.compress_orderby = 'source_timestamp DESC, tick_id ASC');
SELECT add_compression_policy('market_data.binance_futures_btcusdt_l2_one_second_features', compress_after => '7 days'::interval, if_not_exists => true);
SELECT add_compression_policy('market_data.binance_futures_btcusdt_open_interest', compress_after => '7 days'::interval, if_not_exists => true);
SELECT add_compression_policy('market_data.binance_spot_btcusdt_aggregate_trades', compress_after => '2 days'::interval, if_not_exists => true);
SELECT add_compression_policy('market_data.binance_spot_btcusdt_l2_one_second_features', compress_after => '7 days'::interval, if_not_exists => true);
SELECT add_compression_policy('market_data.binance_spot_btcusdt_l2_snapshots', compress_after => '1 day'::interval, if_not_exists => true);
SELECT add_compression_policy('market_data.binance_spot_btcusdt_one_second_ohlcv', compress_after => '7 days'::interval, if_not_exists => true);
SELECT add_compression_policy('market_data.chainlink_btcusd_one_minute_candles', compress_after => '7 days'::interval, if_not_exists => true);
SELECT add_compression_policy('market_data.chainlink_btcusd_reference_prices', compress_after => '2 days'::interval, if_not_exists => true);
SELECT add_compression_policy('market_data.pmdata_chainlink_btcusd_reference_prices', compress_after => '2 days'::interval, if_not_exists => true);
SELECT add_compression_policy('market_data.polygon_chainlink_btcusd_oracle_rounds', compress_after => '7 days'::interval, if_not_exists => true);
SELECT add_compression_policy('market_data.polymarket_chainlink_btcusd_twap', compress_after => '1 day'::interval, if_not_exists => true);
SELECT add_compression_policy('polymarket.btc_feature_snapshots', compress_after => '1 day'::interval, if_not_exists => true);
SELECT add_retention_policy('polymarket.btc_feature_snapshots', drop_after => '365 days'::interval, if_not_exists => true);
SELECT add_compression_policy('polymarket.btc_five_minute_orderbook_snapshots', compress_after => '1 day'::interval, if_not_exists => true);
SELECT add_compression_policy('polymarket.btc_market_capacity_execution_snapshots', compress_after => '7 days'::interval, if_not_exists => true);
SELECT add_compression_policy('polymarket.fills', compress_after => '14 days'::interval, if_not_exists => true);
SELECT add_retention_policy('polymarket.fills', drop_after => '180 days'::interval, if_not_exists => true);
SELECT add_compression_policy('polymarket.reference_price_ticks', compress_after => '1 day'::interval, if_not_exists => true);
SELECT add_retention_policy('polymarket.reference_price_ticks', drop_after => '180 days'::interval, if_not_exists => true);
SELECT add_retention_policy('polymarket.risk_events', drop_after => '180 days'::interval, if_not_exists => true);
-- Mark the historical incremental chain represented by this baseline as applied.
INSERT INTO migrations(timestamp,name) VALUES (1777102000000,'CreatePolymarketBotSchema1777102000000');
INSERT INTO migrations(timestamp,name) VALUES (1777103000000,'PolymarketBotScannerPersistenceDrift1777103000000');
INSERT INTO migrations(timestamp,name) VALUES (1777104000000,'CreatePolymarketWhaleBackfillSchema1777104000000');
INSERT INTO migrations(timestamp,name) VALUES (1777105000000,'AddPolymarketWhalePersistenceDrift1777105000000');
INSERT INTO migrations(timestamp,name) VALUES (1777106000000,'AddPolymarketWalletPerformance1777106000000');
INSERT INTO migrations(timestamp,name) VALUES (1777107000000,'AddPolymarketTradePnlLifecycle1777107000000');
INSERT INTO migrations(timestamp,name) VALUES (1777108000000,'OptimizePolymarketTradePnlReadPaths1777108000000');
INSERT INTO migrations(timestamp,name) VALUES (1777109000000,'AddPolymarketLiveReadiness1777109000000');
INSERT INTO migrations(timestamp,name) VALUES (1777110000000,'AddPolymarketTradingProcessScope1777110000000');
INSERT INTO migrations(timestamp,name) VALUES (1777111000000,'AddPolymarketTradingProcessKeyUniqueness1777111000000');
INSERT INTO migrations(timestamp,name) VALUES (1777112000000,'AddPolymarketMarkSourceFailures1777112000000');
INSERT INTO migrations(timestamp,name) VALUES (1777113000000,'AddPolymarketAccountReconciliation1777113000000');
INSERT INTO migrations(timestamp,name) VALUES (1777114000000,'DropCopyTradeSignalWalletScore1777114000000');
INSERT INTO migrations(timestamp,name) VALUES (1777115000000,'AddPolymarketWalletSegmentPerformance1777115000000');
INSERT INTO migrations(timestamp,name) VALUES (1777116000000,'AddPolymarketGammaTaxonomyCache1777116000000');
INSERT INTO migrations(timestamp,name) VALUES (1777117000000,'AddPolymarketBacktestReplayRuns1777117000000');
INSERT INTO migrations(timestamp,name) VALUES (1777118000000,'AddPolymarketExpectancyFlowCells1777118000000');
INSERT INTO migrations(timestamp,name) VALUES (1777119000000,'AddPolymarketWalletScoreRefreshQueue1777119000000');
INSERT INTO migrations(timestamp,name) VALUES (1777120000000,'AddBtcRealtimePaperAndMlShadow1777120000000');
INSERT INTO migrations(timestamp,name) VALUES (1777121000000,'AddBtcOfficialMarketResolution1777121000000');
INSERT INTO migrations(timestamp,name) VALUES (1777122000000,'AddBtcOfficialResolutionRecovery1777122000000');
INSERT INTO migrations(timestamp,name) VALUES (1777123000000,'AddBtcPhase6PaperCapital1777123000000');
INSERT INTO migrations(timestamp,name) VALUES (1777124000000,'AllowUnstartedTradingProcesses1777124000000');
INSERT INTO migrations(timestamp,name) VALUES (1777125000000,'AddMlTrainingBackfillIngestion1777125000000');
INSERT INTO migrations(timestamp,name) VALUES (1784222637000,'RetirePolymarketReplayAndBacktests1784222637000');
INSERT INTO migrations(timestamp,name) VALUES (1784224000000,'RetirePolymarketWhaleCopyTrading1784224000000');
INSERT INTO migrations(timestamp,name) VALUES (1784225000000,'RetirePolymarketLegacyMarketScanner1784225000000');
INSERT INTO migrations(timestamp,name) VALUES (1784226000000,'RetirePolymarketWalletAnalytics1784226000000');
INSERT INTO migrations(timestamp,name) VALUES (1784227000000,'RetireBtcMlShadow1784227000000');
INSERT INTO migrations(timestamp,name) VALUES (1784391000000,'AddBtcAdmissionCandidateHistoryIndex1784391000000');
INSERT INTO migrations(timestamp,name) VALUES (1784661463000,'AddCanonicalFillIdentity1784661463000');
INSERT INTO migrations(timestamp,name) VALUES (1784662000000,'RetirePolymarketLegacyMarketScaffold1784662000000');
INSERT INTO migrations(timestamp,name) VALUES (1784846080000,'RetireBtcExperimentLifecycle1784846080000');
INSERT INTO migrations(timestamp,name) VALUES (1784846090000,'RetirePolymarketNegativeRiskConversions1784846090000');
INSERT INTO migrations(timestamp,name) VALUES (1784846100000,'RetirePolymarketOrphanedPersistence1784846100000');
INSERT INTO migrations(timestamp,name) VALUES (1784846110000,'RetirePolymarketOrderSignalIdentity1784846110000');
INSERT INTO migrations(timestamp,name) VALUES (1784913000000,'AddBtcHistoricalReferenceAndOrderbook1784913000000');
INSERT INTO migrations(timestamp,name) VALUES (1784914000000,'AddBtcHistoricalExecutionSnapshots1784914000000');
INSERT INTO migrations(timestamp,name) VALUES (1784916000000,'PruneSupersededBtcOrderbookMaterialization1784916000000');
INSERT INTO migrations(timestamp,name) VALUES (1785157648000,'ReconcileTerminalPmxtExecutionArtifactFailures1785157648000');
INSERT INTO migrations(timestamp,name) VALUES (1785160067000,'FinalizeFailedPmxtExecutionArtifacts1785160067000');
INSERT INTO migrations(timestamp,name) VALUES (1785161573000,'RepairFailedPmxtExecutionArtifactState1785161573000');
INSERT INTO migrations(timestamp,name) VALUES (1785163021000,'RepairCorruptPmxtExecutionArtifactState1785163021000');
INSERT INTO migrations(timestamp,name) VALUES (1785163418000,'ReconcileMalformedPmxtExecutionArtifact1785163418000');
INSERT INTO migrations(timestamp,name) VALUES (1785164097000,'ReconcileInvalidPmxtCacheArtifact1785164097000');
INSERT INTO migrations(timestamp,name) VALUES (1785166500000,'EnableBtcDirectionalModelPaperValidation1785166500000');
INSERT INTO migrations(timestamp,name) VALUES (1785176400000,'AddBtcDirectionalModelCoveragePaperProcess1785176400000');
INSERT INTO migrations(timestamp,name) VALUES (1785248631000,'AllowGammaOfficialResolutionReconciliation1785248631000');
INSERT INTO migrations(timestamp,name) VALUES (1785357000000,'AddBtcDirectionalRecencyPaperProcess1785357000000');
INSERT INTO migrations(timestamp,name) VALUES (1785359400000,'AddPolygonChainlinkBtcusdOracleRounds1785359400000');
INSERT INTO migrations(timestamp,name) VALUES (1785443400000,'AllowDecisionWindowExecutionSnapshots1785443400000');
INSERT INTO migrations(timestamp,name) VALUES (1785443500000,'ClearFailedDecisionExecutionSnapshots1785443500000');
INSERT INTO migrations(timestamp,name) VALUES (1785531000000,'AddBtcLiveExecutionReconciliationScope1785531000000');
INSERT INTO migrations(timestamp,name) VALUES (1785531100000,'AddBtcBoundaryAlignmentLivePilot1785531100000');
INSERT INTO migrations(timestamp,name) VALUES (1785612000000,'AddChainlinkCandlesAndBinanceOpenInterest1785612000000');
INSERT INTO migrations(timestamp,name) VALUES (1785638500000,'AddBtcChainlinkOiPaperProcesses1785638500000');
INSERT INTO migrations(timestamp,name) VALUES (1785685200000,'AddBinanceBtcusdtL2OneSecondFeatures1785685200000');
INSERT INTO migrations(timestamp,name) VALUES (1785688800000,'QueueFailedBinanceBtcusdtL2Retries1785688800000');
INSERT INTO migrations(timestamp,name) VALUES (1785776400000,'AddBinanceSpotBtcusdtL2OneSecondFeatures1785776400000');
INSERT INTO migrations(timestamp,name) VALUES (1785780000000,'AllowCoinapiBinanceSpotL2Lineage1785780000000');
INSERT INTO migrations(timestamp,name) VALUES (1785866400000,'AllowHuggingFaceBinanceSpotL2Lineage1785866400000');
INSERT INTO migrations(timestamp,name) VALUES (1786046400000,'AddBtcAsymmetricValuePaperProcesses1786046400000');
INSERT INTO migrations(timestamp,name) VALUES (1786047300000,'CorrectBtcAsymmetricValuePaperRunIds1786047300000');
INSERT INTO migrations(timestamp,name) VALUES (1786381000000,'AddMarketDataIngesterControlPlane1786381000000');
INSERT INTO migrations(timestamp,name) VALUES (1786381050000,'AllowMultipleCaptureArtifactsPerWindow1786381050000');
INSERT INTO migrations(timestamp,name) VALUES (1786381100000,'AddBinanceSpotBtcusdtAggregateTrades1786381100000');
INSERT INTO migrations(timestamp,name) VALUES (1786381200000,'AddBinanceSpotBtcusdtOneSecondOhlcv1786381200000');
INSERT INTO migrations(timestamp,name) VALUES (1786381300000,'AddBinanceSpotBtcusdtL2Snapshots1786381300000');
INSERT INTO migrations(timestamp,name) VALUES (1786381400000,'AddBinanceFuturesBtcusdtOpenInterest1786381400000');
INSERT INTO migrations(timestamp,name) VALUES (1786381500000,'AddChainlinkBtcusdReferencePrices1786381500000');
INSERT INTO migrations(timestamp,name) VALUES (1786381600000,'AddChainlinkBtcusdOneMinuteCandles1786381600000');
INSERT INTO migrations(timestamp,name) VALUES (1786381700000,'AddPolygonChainlinkBtcusdOracleRounds1786381700000');
INSERT INTO migrations(timestamp,name) VALUES (1786381750000,'CorrectChainlinkReferencePricePageLimit1786381750000');
INSERT INTO migrations(timestamp,name) VALUES (1786381760000,'TerminalizeChainlinkReferencePriceCadenceGaps1786381760000');
INSERT INTO migrations(timestamp,name) VALUES (1786381800000,'AddPolymarketBtcFiveMinuteContracts1786381800000');
INSERT INTO migrations(timestamp,name) VALUES (1786381900000,'AddPolymarketBtcFiveMinuteOrderbookSnapshots1786381900000');
INSERT INTO migrations(timestamp,name) VALUES (1786382000000,'AddPolymarketBtcFiveMinuteResolutions1786382000000');
INSERT INTO migrations(timestamp,name) VALUES (1786382100000,'AllowSharedLiveExecutionAccounts1786382100000');
INSERT INTO migrations(timestamp,name) VALUES (1786737600000,'AddBtcCapacityExecutionSnapshots1786737600000');
INSERT INTO migrations(timestamp,name) VALUES (1786824000000,'ExpandBtcCapacityVwapTiers1786824000000');
INSERT INTO migrations(timestamp,name) VALUES (1786929929000,'CorrectLiveDynamicFees1786929929000');
INSERT INTO migrations(timestamp,name) VALUES (1786929930000,'IndexRecognizedManualLiveExits1786929930000');
INSERT INTO migrations(timestamp,name) VALUES (1787241600000,'AddPolymarketChainlinkBtcusdTwap1787241600000');
INSERT INTO migrations(timestamp,name) VALUES (1787241700000,'CorrectPolymarketChainlinkTwapExactValueCheck1787241700000');
INSERT INTO migrations(timestamp,name) VALUES (1787313600000,'RetireFeedAuditTables1787313600000');
INSERT INTO migrations(timestamp,name) VALUES (1787594400000,'AddPmdataChainlinkBtcusdTwap1787594400000');
INSERT INTO migrations(timestamp,name) VALUES (1787598000000,'ExtendChainlinkRefpriceForPmdata1787598000000');
INSERT INTO migrations(timestamp,name) VALUES (1787678525000,'AllowEmptyPmxtCapacityReplacement1787678525000');
INSERT INTO migrations(timestamp,name) VALUES (1787679000000,'CorrectEmptyPmxtCapacityReplacement1787679000000');
INSERT INTO migrations(timestamp,name) VALUES (1787679600000,'AllowEmptyPmxtArtifactSupersession1787679600000');
INSERT INTO migrations(timestamp,name) VALUES (1788295500000,'CreateUnifiedIngesterBackfills1788295500000');
INSERT INTO migrations(timestamp,name) VALUES (1788295600000,'ImportLegacyArchiveBackfills1788295600000');
INSERT INTO migrations(timestamp,name) VALUES (1788295700000,'CorrectUnifiedIngesterAssignmentConstraint1788295700000');
INSERT INTO migrations(timestamp,name) VALUES (1788295800000,'ConsolidateWeatherBackfills1788295800000');
INSERT INTO migrations(timestamp,name) VALUES (1788295900000,'CompleteBackfillLedgerConsolidation1788295900000');
INSERT INTO migrations(timestamp,name) VALUES (1788296200000,'RetryMigratedWeatherBackfills1788296200000');
INSERT INTO migrations(timestamp,name) VALUES (1788382800000,'ConsolidateBinanceAggregateTrades1788382800000');
INSERT INTO migrations(timestamp,name) VALUES (1788472800000,'ConsolidateBinanceFuturesOpenInterest1788472800000');
INSERT INTO migrations(timestamp,name) VALUES (1788559200000,'ConsolidateCanonicalMarketDataTables1788559200000');
INSERT INTO migrations(timestamp,name) VALUES (1788645900000,'RestorePreConsolidationOrderbooks1788645900000');
INSERT INTO migrations(timestamp,name) VALUES (1788646000000,'FinalizePolymarketOrderbookStorage1788646000000');
INSERT INTO migrations(timestamp,name) VALUES (1788646001000,'DropLegacyPolymarketOrderbookStorage1788646001000');
INSERT INTO migrations(timestamp,name) VALUES (1788647000000,'CanonicalizeChainlinkReferencePriceStorage1788647000000');
INSERT INTO migrations(timestamp,name) VALUES (1788648000000,'AddCanonicalBinanceL2FeatureStorage1788648000000');
INSERT INTO migrations(timestamp,name) VALUES (1788648100000,'DropLegacyBinanceL2FeatureStorage1788648100000');
INSERT INTO migrations(timestamp,name) VALUES (1788648200000,'ConsolidateBtcExecutionSnapshotStorage1788648200000');
INSERT INTO migrations(timestamp,name) VALUES (1788732000000,'AddIngesterDrainJobs1788732000000');
INSERT INTO migrations(timestamp,name) VALUES (1788818400000,'AddVerifiedDatasetDrainRemoval1788818400000');
INSERT INTO migrations(timestamp,name) VALUES (1788818500000,'RegisterVerifiedMarketDataDrains1788818500000');
INSERT INTO migrations(timestamp,name) VALUES (1788818600000,'RegisterTrainingMarketDataDrains1788818600000');
INSERT INTO migrations(timestamp,name) VALUES (1788818700000,'AddDrainReconciliationMode1788818700000');
INSERT INTO migrations(timestamp,name) VALUES (1788818800000,'AddIngesterWorkerAllocationPolicy1788818800000');
INSERT INTO migrations(timestamp,name) VALUES (1788818900000,'CorrectIngesterWorkerAllocationColumnTypes1788818900000');
