import { MigrationInterface, QueryRunner } from 'typeorm';

export class EnableBtcDirectionalModelPaperValidation1785166500000
  implements MigrationInterface
{
  name = 'EnableBtcDirectionalModelPaperValidation1785166500000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    // Process starts use an in-memory lifecycle mutex. Quiesce polymarket-bot after
    // stopping both target processes and before applying this definition cutover.
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);

    await queryRunner.query(`
      DO $$
      DECLARE
        candidate record;
        affected integer;
      BEGIN
        IF to_regclass('polymarket.trading_processes') IS NULL
          OR to_regclass('polymarket.trading_process_events') IS NULL
        THEN
          RAISE EXCEPTION
            'refusing to enable directional-model paper validation: required tables are missing';
        END IF;

        FOR candidate IN
          SELECT *
          FROM (
            VALUES
              (
                '88fc207a-39d0-407d-88a7-cc33ac3f795e'::uuid,
                'btc-5m-directional-model-paper-min-entry-030-strategy-only',
                'btc-5m-directional-model-20260727-strategy-only-v1',
                'btc-5m-directional-model-20260727-strategy-only-validation-v1',
                '70189b3d-f71d-5ebc-b11f-99fb9627886b'::uuid,
                '66f23a0daa9a7439f641006b979d88106abfbb911797941e8287906a0b311ac3',
                '77bfa9102751a1bd6c26891421bda1b9b1a26ff6c27aa6a307dc36de09b787c5',
                'strategy_only',
                'none',
                'validate_native_model_decision_order_fill_settlement_and_paper_pnl_without_entry_admission',
                '2026-07-27|btc5m-directional-model-histogram-enriched-20260421-20260620-v1|strategy-only-no-entry-admission|first-confidence-crossing-60-240-5|confidence-0.89|target-size-5|min-entry-0.30|paper-only|entry-policy-execute-directional-prediction'
              ),
              (
                '5763ff0f-f153-4af8-828c-c867f944aee3'::uuid,
                'btc-5m-directional-model-paper-min-entry-030-floor',
                'btc-5m-directional-model-20260727-floor-v1',
                'btc-5m-directional-model-20260727-floor-validation-v1',
                '280618d7-61bc-580e-93e3-b5b6d8bd3099'::uuid,
                'f60b4bbdb87249f2a03f3d8ab9afc89971efecf7eff9a146710a06b3f246268d',
                'ae09715aebb390451370792a6a57f3052da1e64f6283fdb36dc70003412ba783',
                'loss_regime_floor',
                'loss_regime_confidence_floor_v1',
                'validate_native_model_decision_order_fill_settlement_and_paper_pnl_with_loss_regime_entry_admission',
                '2026-07-27|btc5m-directional-model-histogram-enriched-20260421-20260620-v1|loss-regime-floor-v1|first-confidence-crossing-60-240-5|confidence-0.89|target-size-5|min-entry-0.30|paper-only|entry-policy-execute-directional-prediction'
              )
          ) AS definitions(
            process_id,
            process_key,
            old_run_key,
            new_run_key,
            new_run_id,
            old_preregistration_sha256,
            new_preregistration_sha256,
            comparison_arm,
            entry_admission_policy,
            evidence_objective,
            preregistration
          )
        LOOP
          IF EXISTS (
            SELECT 1
            FROM polymarket.trading_process_events event
            WHERE event.event_type = 'btc_run_manifest'
              AND (
                event.event_id = candidate.new_run_id
                OR event.metadata ->> 'run_id' = candidate.new_run_id::text
                OR event.metadata ->> 'run_key' = candidate.new_run_key
              )
          ) THEN
            RAISE EXCEPTION
              'refusing to reuse directional-model paper run key %',
              candidate.new_run_key;
          END IF;

          UPDATE polymarket.trading_processes process
          SET
            config = jsonb_set(
              jsonb_set(
                jsonb_set(
                  process.config,
                  '{raw,btc_realtime_paper,next_experiment_key}',
                  to_jsonb(candidate.new_run_key::text),
                  false
                ),
                '{raw,btc_realtime_paper,preregistration_sha256}',
                to_jsonb(candidate.new_preregistration_sha256::text),
                false
              ),
              '{raw,btc_realtime_paper,paper,directional_model_entry_policy}',
              '"execute_directional_prediction"'::jsonb,
              true
            ),
            metadata = jsonb_set(
              process.metadata || jsonb_build_object(
                'comparison_cohort',
                'btc5m-directional-model-paper-validation-v1-20260727',
                'comparison_arm',
                candidate.comparison_arm,
                'entry_admission_policy',
                candidate.entry_admission_policy,
                'directional_model_entry_policy',
                'execute_directional_prediction',
                'evidence_objective',
                candidate.evidence_objective,
                'preregistration',
                candidate.preregistration
              ),
              '{run_definition_metadata}',
              COALESCE(
                process.metadata -> 'run_definition_metadata',
                '{}'::jsonb
              )
                || jsonb_build_object(
                  candidate.old_run_key,
                  process.metadata - 'run_definition_metadata'
                )
                || jsonb_build_object(
                  candidate.new_run_key,
                  jsonb_build_object(
                    'comparison_cohort',
                    'btc5m-directional-model-paper-validation-v1-20260727',
                    'comparison_arm',
                    candidate.comparison_arm,
                    'entry_admission_policy',
                    candidate.entry_admission_policy,
                    'directional_model_entry_policy',
                    'execute_directional_prediction',
                    'evidence_objective',
                    candidate.evidence_objective,
                    'preregistration',
                    candidate.preregistration
                  )
                ),
              true
            ),
            updated_at = now()
          WHERE process.process_id = candidate.process_id
            AND process.process_key = candidate.process_key
            AND process.process_type = 'btc_5m'
            AND process.process_scope = 'realtime_paper'
            AND NOT process.enabled
            AND process.status IN ('stopped', 'failed', 'completed')
            AND process.stopped_at IS NOT NULL
            AND process.config #>> '{execution,mode}' = 'paper'
            AND process.config #>> '{execution,execute_signals}' = 'true'
            AND process.config #>> '{execution,live_capital}' = 'false'
            AND process.config #>> '{raw,btc_realtime_paper,schema_version}'
                  = 'btc_realtime_paper_process_v3'
            AND jsonb_typeof(
                  process.config #> '{raw,btc_realtime_paper,paper}'
                ) = 'object'
            AND jsonb_typeof(process.metadata) = 'object'
            AND (
              NOT process.metadata ? 'run_definition_metadata'
              OR jsonb_typeof(process.metadata -> 'run_definition_metadata') = 'object'
            )
            AND process.config #>> '{raw,btc_realtime_paper,next_experiment_key}'
                  = candidate.old_run_key
            AND process.config #>> '{raw,btc_realtime_paper,preregistration_sha256}'
                  = candidate.old_preregistration_sha256
            AND process.config #>> '{raw,btc_realtime_paper,strategy,decision_strategy,type}'
                  = 'btc_directional_model'
            AND process.config #>> '{raw,btc_realtime_paper,strategy,decision_strategy,model_key}'
                  = 'btc-5m-directional-histogram-enriched-20260421-20260620-v1'
            AND process.config #>> '{raw,btc_realtime_paper,strategy,decision_strategy,artifact_sha256}'
                  = 'e07160295cac3df02d9bec631e96ce34fa66a71a72d04e79e4bcf0fd0f8336b7'
            AND process.config #>> '{raw,btc_realtime_paper,strategy,decision_strategy,feature_schema_sha256}'
                  = '392aac87ecbc6704929b0ab91aed5de90dff26daf40edfa4e22d2c78a9d03dfc'
            AND COALESCE(
                  process.config #>> '{raw,btc_realtime_paper,paper,directional_model_entry_policy}',
                  'require_positive_direct_edge'
                ) = 'require_positive_direct_edge'
            AND (
              (
                candidate.comparison_arm = 'strategy_only'
                AND process.config #> '{raw,btc_realtime_paper,entry_admission}'
                      IS NULL
              )
              OR (
                candidate.comparison_arm = 'loss_regime_floor'
                AND process.config #>>
                      '{raw,btc_realtime_paper,entry_admission,loss_regime_confidence_floor,schema_version}'
                      = 'loss_regime_confidence_floor_v1'
                AND process.config #>>
                      '{raw,btc_realtime_paper,entry_admission,loss_regime_confidence_floor,activation_consecutive_candidate_losses}'
                      = '2'
                AND process.config #>>
                      '{raw,btc_realtime_paper,entry_admission,loss_regime_confidence_floor,min_conservative_probability}'
                      = '0.50'
                AND process.config #>>
                      '{raw,btc_realtime_paper,entry_admission,loss_regime_confidence_floor,release_consecutive_candidate_wins}'
                      = '1'
              )
            );

          GET DIAGNOSTICS affected = ROW_COUNT;
          IF affected <> 1 THEN
            RAISE EXCEPTION
              'refusing to update directional-model paper process %: expected one inactive exact match, updated %',
              candidate.process_id,
              affected;
          END IF;

          IF NOT EXISTS (
            SELECT 1
            FROM polymarket.trading_processes process
            WHERE process.process_id = candidate.process_id
              AND process.config #>>
                    '{raw,btc_realtime_paper,next_experiment_key}'
                    = candidate.new_run_key
              AND process.config #>>
                    '{raw,btc_realtime_paper,preregistration_sha256}'
                    = candidate.new_preregistration_sha256
              AND process.config #>>
                    '{raw,btc_realtime_paper,paper,directional_model_entry_policy}'
                    = 'execute_directional_prediction'
              AND process.metadata ->> 'directional_model_entry_policy'
                    = 'execute_directional_prediction'
              AND process.metadata -> 'run_definition_metadata'
                    ? candidate.old_run_key
              AND process.metadata -> 'run_definition_metadata'
                    ? candidate.new_run_key
          ) THEN
            RAISE EXCEPTION
              'directional-model paper process % failed post-update validation',
              candidate.process_id;
          END IF;
        END LOOP;
      END $$;
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    throw new Error(
      'EnableBtcDirectionalModelPaperValidation1785166500000 is intentionally irreversible because each validation definition owns a unique run key that becomes immutable once started',
    );
  }
}
