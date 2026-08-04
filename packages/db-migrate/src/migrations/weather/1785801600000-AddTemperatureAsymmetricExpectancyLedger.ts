import { MigrationInterface, QueryRunner } from 'typeorm';

const PROCESS_ID = 'd48e2b47-18df-4c8c-95b9-0f12e6ca7d41';

export class AddTemperatureAsymmetricExpectancyLedger1785801600000
  implements MigrationInterface
{
  name = 'AddTemperatureAsymmetricExpectancyLedger1785801600000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF NOT EXISTS (
          SELECT 1
          FROM polymarket.trading_processes
          WHERE process_id = '${PROCESS_ID}'
        ) THEN
          RAISE EXCEPTION 'missing canonical NYC temperature trading process';
        END IF;
      END $$;

      UPDATE polymarket.trading_processes
      SET config = (config - 'side') || '{
            "sides":["buy_yes","buy_no"],
            "objective":"positive_net_expectancy_at_low_executable_cost_without_accuracy_gate",
            "forecast_selection":{
              "eligible_ml_candidates":["linear_bias","histogram_residual"],
              "diagnostic_candidates":["raw_hrrr"],
              "metric":"calibration_loo_rounded_log_loss",
              "secondary_metric":"calibration_loo_ranked_probability_score",
              "point_diagnostic":"calibration_rmse_f",
              "rounded_temperature_support_min_f":-20,
              "rounded_temperature_support_max_f":130,
              "training_start":"2019-01-01",
              "training_end":"2024-12-31",
              "calibration_start":"2025-01-01",
              "calibration_end":"2025-12-31",
              "holdout_influence":false
            },
            "economic_evaluation":{
              "discovery_start":"2026-04-14",
              "discovery_end":"2026-06-30",
              "evaluation_start":"2026-07-01",
              "evaluation_end":"2026-07-30",
              "quantity":5,
              "modeled_slippage_per_share":0.01,
              "slippage_stress_per_share":[0,0.005,0.01,0.02],
              "maximum_positions_per_event_day":1,
              "price_cell_minimum_cluster_days":14
            },
            "policy_frontier":{
              "minimum_all_in_cost":0.04,
              "rank_by":"robust_expected_roi",
              "probability_lower_quantile":0.10,
              "residual_bootstrap_block_days":7,
              "probability_bootstrap_iterations":2000,
              "probability_bootstrap_seed":24051985,
              "probability_bootstrap_common_random_numbers":true,
              "economic_bootstrap_confidence":0.90,
              "economic_bootstrap_iterations":5000,
              "economic_bootstrap_block_days":7,
              "economic_bootstrap_seed":41819850,
              "decision_modes":["midnight_only","noon_only","midnight_then_noon"],
              "specifications":[
                {"name":"deep_asymmetry","maximum_all_in_cost":0.16,"minimum_robust_edge":0.03,"minimum_robust_roi":0.35},
                {"name":"balanced_asymmetry","maximum_all_in_cost":0.25,"minimum_robust_edge":0.04,"minimum_robust_roi":0.35},
                {"name":"strong_edge","maximum_all_in_cost":0.25,"minimum_robust_edge":0.06,"minimum_robust_roi":0.50}
              ]
            }
          }'::jsonb,
          metadata = metadata || '{
            "purpose":"offline bidirectional low-price asymmetric expectancy qualification"
          }'::jsonb,
          updated_at = now()
      WHERE process_id = '${PROCESS_ID}';
    `);

    await queryRunner.query(`
      ALTER TABLE weather.temperature_markets
        ADD COLUMN IF NOT EXISTS fees_enabled boolean NOT NULL DEFAULT false,
        ADD COLUMN IF NOT EXISTS fee_rate numeric(18,10) NOT NULL DEFAULT 0,
        ADD COLUMN IF NOT EXISTS fee_exponent numeric(18,10) NOT NULL DEFAULT 1,
        ADD COLUMN IF NOT EXISTS fee_taker_only boolean NOT NULL DEFAULT true;

      UPDATE weather.temperature_markets
      SET fees_enabled = CASE lower(COALESCE(raw_payload->>'feesEnabled', 'false'))
            WHEN 'true' THEN true ELSE false END,
          fee_rate = CASE
            WHEN COALESCE(raw_payload->'feeSchedule'->>'rate', '')
                   ~ '^[0-9]+([.][0-9]+)?$'
              THEN (raw_payload->'feeSchedule'->>'rate')::numeric
            WHEN fee_rate_bps IS NOT NULL THEN fee_rate_bps::numeric / 10000
            ELSE 0
          END,
          fee_exponent = CASE
            WHEN COALESCE(raw_payload->'feeSchedule'->>'exponent', '')
                   ~ '^[0-9]+([.][0-9]+)?$'
              THEN (raw_payload->'feeSchedule'->>'exponent')::numeric
            ELSE 1
          END,
          fee_taker_only = CASE lower(
            COALESCE(raw_payload->'feeSchedule'->>'takerOnly', 'true')
          ) WHEN 'false' THEN false ELSE true END;

      UPDATE weather.temperature_markets
      SET fee_rate_bps = round(fee_rate * 10000)::integer;

      ALTER TABLE weather.temperature_markets
        ADD CONSTRAINT chk_weather_market_fee_rate
          CHECK (fee_rate >= 0 AND fee_rate <= 1),
        ADD CONSTRAINT chk_weather_market_fee_exponent
          CHECK (fee_exponent > 0),
        ADD CONSTRAINT chk_weather_market_enabled_fee
          CHECK (NOT fees_enabled OR fee_rate > 0);
    `);

    await queryRunner.query(`
      CREATE TABLE weather.asymmetric_policy_runs (
        policy_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        process_id uuid NOT NULL
          REFERENCES polymarket.trading_processes (process_id) ON DELETE RESTRICT,
        discovery_start date NOT NULL,
        discovery_end date NOT NULL,
        evaluation_start date NOT NULL,
        evaluation_end date NOT NULL,
        quantity numeric(12,4) NOT NULL,
        decision_models jsonb NOT NULL,
        policy jsonb NOT NULL,
        discovery_metrics jsonb NOT NULL,
        evaluation_metrics jsonb NOT NULL,
        qualified boolean NOT NULL,
        report_uri text,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_weather_asymmetric_run_ranges CHECK (
          discovery_end >= discovery_start
          AND evaluation_start > discovery_end
          AND evaluation_end >= evaluation_start
        ),
        CONSTRAINT chk_weather_asymmetric_run_quantity CHECK (quantity > 0),
        CONSTRAINT chk_weather_asymmetric_run_documents CHECK (
          jsonb_typeof(decision_models) = 'object'
          AND jsonb_typeof(policy) = 'object'
          AND jsonb_typeof(discovery_metrics) = 'object'
          AND jsonb_typeof(evaluation_metrics) = 'object'
        )
      );

      ALTER TABLE weather.asymmetric_policy_runs
        ADD CONSTRAINT uq_weather_asymmetric_run_process
          UNIQUE (process_id, policy_run_id);

      ALTER TABLE weather.model_runs
        ADD CONSTRAINT uq_weather_model_run_process
          UNIQUE (process_id, model_run_id);

      CREATE INDEX idx_weather_asymmetric_run_process_created
        ON weather.asymmetric_policy_runs (process_id, created_at DESC);

      CREATE TABLE weather.asymmetric_candidate_ledger (
        process_id uuid NOT NULL
          REFERENCES polymarket.trading_processes (process_id) ON DELETE RESTRICT,
        policy_run_id uuid NOT NULL,
        model_run_id uuid NOT NULL,
        market_id text NOT NULL
          REFERENCES weather.temperature_markets (market_id) ON DELETE RESTRICT,
        split text NOT NULL,
        event_date date NOT NULL,
        decision_time timestamptz NOT NULL,
        decision_hour_local integer NOT NULL,
        side text NOT NULL,
        quantity numeric(12,4) NOT NULL,
        probability numeric(18,12) NOT NULL,
        probability_lower numeric(18,12) NOT NULL,
        market_probability_proxy numeric(18,12),
        ask_vwap numeric(18,8),
        fees_enabled boolean NOT NULL,
        fee_rate numeric(18,10) NOT NULL,
        fee_exponent numeric(18,10) NOT NULL,
        fee_taker_only boolean NOT NULL,
        fee_per_share numeric(18,12),
        modeled_slippage_per_share numeric(18,12),
        all_in_cost_per_share numeric(18,12),
        break_even_probability numeric(18,12),
        model_edge_per_share numeric(18,12),
        robust_edge_per_share numeric(18,12),
        expected_roi numeric(24,12),
        robust_expected_roi numeric(24,12),
        resolved_side boolean NOT NULL,
        executable boolean NOT NULL,
        eligible boolean NOT NULL DEFAULT false,
        selected boolean NOT NULL DEFAULT false,
        realized_net_per_share numeric(18,12),
        source_timestamp timestamptz,
        quote_age_seconds numeric(18,6),
        quality_flags jsonb NOT NULL DEFAULT '[]'::jsonb,
        rejection_reasons jsonb NOT NULL DEFAULT '[]'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (policy_run_id, model_run_id, market_id, decision_time, side),
        CONSTRAINT fk_weather_asymmetric_candidate_policy_process
          FOREIGN KEY (process_id, policy_run_id)
          REFERENCES weather.asymmetric_policy_runs (process_id, policy_run_id)
          ON DELETE RESTRICT,
        CONSTRAINT fk_weather_asymmetric_candidate_model_process
          FOREIGN KEY (process_id, model_run_id)
          REFERENCES weather.model_runs (process_id, model_run_id)
          ON DELETE RESTRICT,
        CONSTRAINT chk_weather_asymmetric_candidate_split CHECK (
          split IN ('discovery','evaluation')
        ),
        CONSTRAINT chk_weather_asymmetric_candidate_hour CHECK (
          decision_hour_local IN (0,12)
        ),
        CONSTRAINT chk_weather_asymmetric_candidate_side CHECK (side IN ('YES','NO')),
        CONSTRAINT chk_weather_asymmetric_candidate_quantity CHECK (quantity > 0),
        CONSTRAINT chk_weather_asymmetric_candidate_probability CHECK (
          probability BETWEEN 0 AND 1
          AND probability_lower BETWEEN 0 AND probability
          AND (market_probability_proxy IS NULL OR market_probability_proxy BETWEEN 0 AND 1)
        ),
        CONSTRAINT chk_weather_asymmetric_candidate_fee CHECK (
          fee_rate >= 0 AND fee_rate <= 1
          AND fee_exponent > 0
          AND (NOT fees_enabled OR fee_rate > 0)
        ),
        CONSTRAINT chk_weather_asymmetric_candidate_price CHECK (
          (ask_vwap IS NULL OR ask_vwap BETWEEN 0 AND 1)
          AND (fee_per_share IS NULL OR fee_per_share >= 0)
          AND (modeled_slippage_per_share IS NULL OR modeled_slippage_per_share >= 0)
          AND (all_in_cost_per_share IS NULL OR all_in_cost_per_share > 0)
          AND (break_even_probability IS NULL OR break_even_probability BETWEEN 0 AND 1)
          AND (quote_age_seconds IS NULL OR quote_age_seconds >= 0)
        ),
        CONSTRAINT chk_weather_asymmetric_candidate_execution CHECK (
          NOT executable OR (
            ask_vwap IS NOT NULL
            AND fee_per_share IS NOT NULL
            AND modeled_slippage_per_share IS NOT NULL
            AND all_in_cost_per_share IS NOT NULL
            AND break_even_probability IS NOT NULL
            AND model_edge_per_share IS NOT NULL
            AND robust_edge_per_share IS NOT NULL
            AND expected_roi IS NOT NULL
            AND robust_expected_roi IS NOT NULL
            AND realized_net_per_share IS NOT NULL
          )
        ),
        CONSTRAINT chk_weather_asymmetric_candidate_selection CHECK (
          NOT selected OR (eligible AND executable)
        ),
        CONSTRAINT chk_weather_asymmetric_candidate_flags CHECK (
          jsonb_typeof(quality_flags) = 'array'
          AND jsonb_typeof(rejection_reasons) = 'array'
        )
      );

      CREATE INDEX idx_weather_asymmetric_candidate_process_time
        ON weather.asymmetric_candidate_ledger (
          process_id, policy_run_id, split, event_date, decision_time
        );
      CREATE UNIQUE INDEX uq_weather_asymmetric_candidate_selected_day
        ON weather.asymmetric_candidate_ledger (process_id, policy_run_id, event_date)
        WHERE selected;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DROP TABLE IF EXISTS weather.asymmetric_candidate_ledger;
      DROP TABLE IF EXISTS weather.asymmetric_policy_runs;

      ALTER TABLE weather.temperature_markets
        DROP CONSTRAINT IF EXISTS chk_weather_market_enabled_fee,
        DROP CONSTRAINT IF EXISTS chk_weather_market_fee_exponent,
        DROP CONSTRAINT IF EXISTS chk_weather_market_fee_rate,
        DROP COLUMN IF EXISTS fee_taker_only,
        DROP COLUMN IF EXISTS fee_exponent,
        DROP COLUMN IF EXISTS fee_rate,
        DROP COLUMN IF EXISTS fees_enabled;

      ALTER TABLE weather.model_runs
        DROP CONSTRAINT IF EXISTS uq_weather_model_run_process;

      UPDATE polymarket.trading_processes
      SET config = (
            config
            - 'sides'
            - 'objective'
            - 'forecast_selection'
            - 'economic_evaluation'
            - 'policy_frontier'
          ) || '{"side":"buy_yes"}'::jsonb,
          metadata = metadata || '{
            "purpose":"offline data and economic qualification only"
          }'::jsonb,
          updated_at = now()
      WHERE process_id = '${PROCESS_ID}';
    `);
  }
}
