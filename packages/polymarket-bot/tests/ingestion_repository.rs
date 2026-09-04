use std::{env, sync::OnceLock, time::Duration as StdDuration};

use anyhow::{anyhow, ensure, Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, NaiveDate, Utc};
use polymarket_bot::ingestion::{
    job::{
        ArtifactCompletion, ArtifactDisposition, ArtifactSpec, BackfillArtifactStatus,
        BackfillJobStatus, BackfillJobSummary, BackfillRequest, BinanceL2OneSecondFeature,
        BinanceOneSecondKlineRecord, IngesterKey, ValidatedBackfillRequest, WorkerControl,
        BACKFILL_REQUEST_VERSION,
    },
    repository::IngestionRepository,
};
use rust_decimal_macros::dec;
use serde_json::json;
use sqlx::{postgres::PgPoolOptions, PgPool};
use tokio::sync::Mutex;
use uuid::Uuid;

const TEST_DATABASE_ENV: &str = "POLYMARKET_TEST_DATABASE_URL";
const TEST_PREFIX: &str = "ingestion-repository-test";
const ACTIVE_LEASE: StdDuration = StdDuration::from_secs(60);

static TEST_SERIALIZER: OnceLock<Mutex<()>> = OnceLock::new();

#[tokio::test]
async fn enqueue_is_idempotent_and_rejects_conflicting_reuse() -> Result<()> {
    let _guard = test_serializer().lock().await;
    let Some(pool) = connect_test_database().await? else {
        return Ok(());
    };
    let repository = IngestionRepository::from_pool(pool.clone());
    let tag = unique_tag();

    let outcome = async {
        let start = unique_five_minute_start();
        let initial_request = request(
            IngesterKey::BtcFiveMinuteMarkets,
            start,
            start + ChronoDuration::minutes(5),
            format!("{tag}enqueue"),
        )?;

        let first = repository.enqueue(&initial_request).await?;
        let repeated = repository.enqueue(&initial_request).await?;
        ensure!(
            first.job_id == repeated.job_id,
            "repeating an identical idempotent request created another job"
        );

        let job_count = sqlx::query_scalar::<_, i64>(
            r#"
            SELECT count(*)
            FROM polymarket.backfill_jobs
            WHERE ingester_key = $1 AND idempotency_key = $2
            "#,
        )
        .bind(IngesterKey::BtcFiveMinuteMarkets.as_str())
        .bind(&initial_request.idempotency_key)
        .fetch_one(&pool)
        .await?;
        ensure!(
            job_count == 1,
            "idempotent enqueue persisted {job_count} jobs"
        );

        let conflicting = request(
            IngesterKey::BtcFiveMinuteMarkets,
            start,
            start + ChronoDuration::minutes(10),
            initial_request.idempotency_key.clone(),
        )?;
        let error = repository
            .enqueue(&conflicting)
            .await
            .expect_err("conflicting idempotency-key reuse must fail");
        ensure!(
            error.to_string().contains("different"),
            "unexpected idempotency conflict error: {error:#}"
        );
        Ok(())
    }
    .await;

    finish_committed_test(pool, &tag, outcome).await
}

#[tokio::test]
async fn reclaimed_lease_fences_the_previous_worker() -> Result<()> {
    let _guard = test_serializer().lock().await;
    let Some(pool) = connect_test_database().await? else {
        return Ok(());
    };
    let repository = IngestionRepository::from_pool(pool.clone());
    let tag = unique_tag();

    let outcome = async {
        assert_no_runnable_jobs(&pool).await?;
        let start = unique_five_minute_start();
        let queued = repository
            .enqueue(&request(
                IngesterKey::BtcFiveMinuteMarkets,
                start,
                start + ChronoDuration::minutes(5),
                format!("{tag}lease"),
            )?)
            .await?;

        let first_claim = repository
            .claim_next(&format!("{tag}worker-a"), ACTIVE_LEASE)
            .await?
            .context("worker A did not claim the queued test job")?;
        ensure!(first_claim.job.job_id == queued.job_id);

        sqlx::query(
            r#"
            UPDATE polymarket.backfill_jobs
            SET lease_expires_at = now() - interval '1 second'
            WHERE job_id = $1
            "#,
        )
        .bind(queued.job_id)
        .execute(&pool)
        .await?;

        let second_claim = repository
            .claim_next(&format!("{tag}worker-b"), ACTIVE_LEASE)
            .await?
            .context("worker B did not reclaim the expired test lease")?;
        ensure!(second_claim.job.job_id == queued.job_id);
        ensure!(
            second_claim.lease_token != first_claim.lease_token,
            "lease reclamation reused its fencing token"
        );

        let stale_error = repository
            .heartbeat(&first_claim, ACTIVE_LEASE)
            .await
            .expect_err("the stale lease holder must be fenced");
        ensure!(
            stale_error.to_string().contains("lost lease fencing"),
            "unexpected stale-lease error: {stale_error:#}"
        );
        repository
            .heartbeat(&second_claim, ACTIVE_LEASE)
            .await
            .context("the current lease holder should retain write authority")?;
        Ok(())
    }
    .await;

    finish_committed_test(pool, &tag, outcome).await
}

#[tokio::test]
async fn running_job_cancellation_is_cooperative_and_terminal() -> Result<()> {
    let _guard = test_serializer().lock().await;
    let Some(pool) = connect_test_database().await? else {
        return Ok(());
    };
    let repository = IngestionRepository::from_pool(pool.clone());
    let tag = unique_tag();

    let outcome = async {
        assert_no_runnable_jobs(&pool).await?;
        let start = unique_five_minute_start();
        let queued = repository
            .enqueue(&request(
                IngesterKey::BtcFiveMinuteMarkets,
                start,
                start + ChronoDuration::minutes(5),
                format!("{tag}cancel"),
            )?)
            .await?;
        let claim = repository
            .claim_next(&format!("{tag}worker"), ACTIVE_LEASE)
            .await?
            .context("worker did not claim the cancellation test job")?;
        ensure!(claim.job.job_id == queued.job_id);

        let cancellation = repository
            .request_cancel(queued.job_id)
            .await?
            .context("cancellation target disappeared")?;
        ensure!(cancellation.status == BackfillJobStatus::CancelRequested);
        ensure!(
            repository.is_cancel_requested(&claim).await? == WorkerControl::CancelRequested,
            "the active worker did not observe cancellation"
        );

        let cancelled = repository
            .mark_cancelled(&claim, &BackfillJobSummary::default())
            .await?;
        ensure!(cancelled.status == BackfillJobStatus::Cancelled);
        ensure!(cancelled.completed_at.is_some());
        ensure!(cancelled.worker_id.is_none());
        ensure!(cancelled.lease_expires_at.is_none());
        Ok(())
    }
    .await;

    finish_committed_test(pool, &tag, outcome).await
}

#[tokio::test]
async fn database_batches_are_bounded_and_idempotent() -> Result<()> {
    let _guard = test_serializer().lock().await;
    let Some(pool) = connect_test_database().await? else {
        return Ok(());
    };
    let repository = IngestionRepository::from_pool(pool.clone());
    let tag = unique_tag();

    let outcome = async {
        assert_no_runnable_jobs(&pool).await?;
        let day_start = unique_day_start();

        let kline_job = repository
            .enqueue(&request(
                IngesterKey::BinanceBtcusdtOneSecondKlines,
                day_start,
                day_start + ChronoDuration::days(1),
                format!("{tag}kline-job"),
            )?)
            .await?;
        let kline_claim = repository
            .claim_next(&format!("{tag}kline-worker"), ACTIVE_LEASE)
            .await?
            .context("worker did not claim the one-second-kline test job")?;
        ensure!(kline_claim.job.job_id == kline_job.job_id);
        let kline_artifact = prepare_ingesting_artifact(
            &repository,
            &kline_claim,
            IngesterKey::BinanceBtcusdtOneSecondKlines,
            &tag,
            "kline",
            day_start.date_naive(),
        )
        .await?;
        let kline_records = vec![
            one_second_kline(day_start + ChronoDuration::seconds(3)),
            one_second_kline(day_start + ChronoDuration::seconds(4)),
        ];

        let first_kline_write = repository
            .insert_one_second_kline_batch(
                &kline_claim,
                kline_artifact.artifact.artifact_id,
                &kline_records,
            )
            .await?;
        ensure!(first_kline_write.input_records == 2);
        ensure!(first_kline_write.inserted_records == 2);
        ensure!(first_kline_write.duplicate_records == 0);

        let repeated_kline_write = repository
            .insert_one_second_kline_batch(
                &kline_claim,
                kline_artifact.artifact.artifact_id,
                &kline_records,
            )
            .await?;
        ensure!(repeated_kline_write.input_records == 2);
        ensure!(repeated_kline_write.inserted_records == 0);
        ensure!(repeated_kline_write.duplicate_records == 2);
        Ok(())
    }
    .await;

    finish_committed_test(pool, &tag, outcome).await
}

#[tokio::test]
async fn completed_artifacts_reject_updates_and_deletes() -> Result<()> {
    let _guard = test_serializer().lock().await;
    let Some(pool) = connect_test_database().await? else {
        return Ok(());
    };

    assert_completed_artifact_mutation_rejected(&pool, CompletedArtifactMutation::Update).await?;
    assert_completed_artifact_mutation_rejected(&pool, CompletedArtifactMutation::Delete).await?;
    pool.close().await;
    Ok(())
}

#[tokio::test]
async fn cryptohft_request_budget_is_global_across_independent_pools() -> Result<()> {
    let _guard = test_serializer().lock().await;
    let Some(first_pool) = connect_test_database().await? else {
        return Ok(());
    };
    let second_pool = connect_additional_test_database().await?;
    let first_repository = IngestionRepository::from_pool(first_pool.clone());
    let second_repository = IngestionRepository::from_pool(second_pool.clone());

    let (first_slot, second_slot) = tokio::try_join!(
        first_repository.reserve_cryptohft_request_slot(),
        second_repository.reserve_cryptohft_request_slot(),
    )?;
    let separation_milliseconds = (second_slot - first_slot).num_milliseconds().abs();
    ensure!(
        separation_milliseconds >= 1_100,
        "independent pools reserved CryptoHFT request slots only {separation_milliseconds}ms apart"
    );

    first_pool.close().await;
    second_pool.close().await;
    Ok(())
}

#[tokio::test]
async fn failed_l2_artifact_clears_staging_without_publishing() -> Result<()> {
    let _guard = test_serializer().lock().await;
    let Some(pool) = connect_test_database().await? else {
        return Ok(());
    };
    let repository = IngestionRepository::from_pool(pool.clone());
    let tag = unique_tag();

    let outcome = async {
        assert_no_runnable_jobs(&pool).await?;
        let day_start = l2_day_start();
        let queued = repository
            .enqueue(&request(
                IngesterKey::BinanceBtcusdtL2OneSecondFeatures,
                day_start,
                day_start + ChronoDuration::days(1),
                format!("{tag}l2-failure-job"),
            )?)
            .await?;
        let claim = repository
            .claim_next(&format!("{tag}l2-failure-worker"), ACTIVE_LEASE)
            .await?
            .context("worker did not claim the Binance L2 failure test job")?;
        ensure!(claim.job.job_id == queued.job_id);
        let prepared = prepare_ingesting_artifact(
            &repository,
            &claim,
            IngesterKey::BinanceBtcusdtL2OneSecondFeatures,
            &tag,
            "l2-failure",
            day_start.date_naive(),
        )
        .await?;
        let artifact_id = prepared.artifact.artifact_id;
        let feature = l2_feature(day_start + ChronoDuration::seconds(1));

        repository
            .stage_binance_l2_one_second_feature_batch(&claim, artifact_id, &[feature.clone()])
            .await?;
        ensure!(l2_staging_count(&pool, artifact_id).await? == 1);
        ensure!(l2_final_count(&pool, artifact_id).await? == 0);
        ensure!(l2_training_count(&pool, &feature).await? == 0);

        let failed = repository
            .fail_binance_l2_artifact(&claim, artifact_id, "expected test failure")
            .await?;
        ensure!(failed.status == BackfillArtifactStatus::Failed);
        ensure!(l2_staging_count(&pool, artifact_id).await? == 0);
        ensure!(l2_final_count(&pool, artifact_id).await? == 0);
        ensure!(l2_training_count(&pool, &feature).await? == 0);
        repository
            .complete(&claim, &BackfillJobSummary::default())
            .await?;
        Ok(())
    }
    .await;

    finish_committed_test(pool, &tag, outcome).await
}

#[tokio::test]
async fn l2_publication_is_atomic_training_visible_and_immutable() -> Result<()> {
    let _guard = test_serializer().lock().await;
    let Some(pool) = connect_test_database().await? else {
        return Ok(());
    };
    let repository = IngestionRepository::from_pool(pool.clone());
    let tag = unique_tag();

    assert_no_runnable_jobs(&pool).await?;
    let day_start = l2_day_start();
    let queued = repository
        .enqueue(&request(
            IngesterKey::BinanceBtcusdtL2OneSecondFeatures,
            day_start,
            day_start + ChronoDuration::days(1),
            format!("{tag}l2-publish-job"),
        )?)
        .await?;
    let claim = repository
        .claim_next(&format!("{tag}l2-publish-worker"), ACTIVE_LEASE)
        .await?
        .context("worker did not claim the Binance L2 publication test job")?;
    ensure!(claim.job.job_id == queued.job_id);
    let prepared = prepare_ingesting_artifact(
        &repository,
        &claim,
        IngesterKey::BinanceBtcusdtL2OneSecondFeatures,
        &tag,
        "l2-publish",
        day_start.date_naive(),
    )
    .await?;
    let artifact_id = prepared.artifact.artifact_id;
    let feature = l2_feature(day_start + ChronoDuration::seconds(2));

    repository
        .stage_binance_l2_one_second_feature_batch(&claim, artifact_id, &[feature.clone()])
        .await?;
    ensure!(l2_staging_count(&pool, artifact_id).await? == 1);
    ensure!(l2_final_count(&pool, artifact_id).await? == 0);
    ensure!(l2_training_count(&pool, &feature).await? == 0);

    let completed = repository
        .publish_binance_l2_one_second_features(
            &claim,
            artifact_id,
            &ArtifactCompletion {
                actual_checksum: "b".repeat(64),
                compressed_bytes: 1_024,
                record_count: 1,
                minimum_source_timestamp: Some(feature.source_event_timestamp),
                maximum_source_timestamp: Some(feature.source_event_timestamp),
                metadata: json!({
                    "materialization_contract":
                        "cryptohft-binance-futures-btcusdt-l2-features-v1"
                }),
            },
        )
        .await?;
    ensure!(completed.status == BackfillArtifactStatus::Completed);
    ensure!(l2_staging_count(&pool, artifact_id).await? == 0);
    ensure!(l2_final_count(&pool, artifact_id).await? == 1);
    ensure!(l2_training_count(&pool, &feature).await? == 1);
    repository
        .complete(&claim, &BackfillJobSummary::default())
        .await?;

    assert_l2_feature_mutation_rejected(&pool, artifact_id, L2FeatureMutation::Update).await?;
    assert_l2_feature_mutation_rejected(&pool, artifact_id, L2FeatureMutation::Delete).await?;

    pool.close().await;
    Ok(())
}

#[tokio::test]
async fn failed_spot_l2_artifact_is_market_isolated_and_clears_staging() -> Result<()> {
    let _guard = test_serializer().lock().await;
    let Some(pool) = connect_test_database().await? else {
        return Ok(());
    };
    let repository = IngestionRepository::from_pool(pool.clone());
    let tag = unique_tag();

    let outcome = async {
        assert_no_runnable_jobs(&pool).await?;
        let day_start = l2_day_start();
        let queued = repository
            .enqueue(&request(
                IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures,
                day_start,
                day_start + ChronoDuration::days(1),
                format!("{tag}spot-l2-failure-job"),
            )?)
            .await?;
        let claim = repository
            .claim_next(&format!("{tag}spot-l2-failure-worker"), ACTIVE_LEASE)
            .await?
            .context("worker did not claim the Binance spot L2 failure test job")?;
        ensure!(claim.job.job_id == queued.job_id);
        let prepared = prepare_ingesting_artifact(
            &repository,
            &claim,
            IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures,
            &tag,
            "spot-l2-failure",
            day_start.date_naive(),
        )
        .await?;
        let artifact_id = prepared.artifact.artifact_id;
        let feature = spot_l2_feature(day_start + ChronoDuration::seconds(3));

        let cross_market_error = repository
            .stage_binance_l2_one_second_feature_batch(&claim, artifact_id, &[feature.clone()])
            .await
            .expect_err("a spot claim must not write futures staging");
        ensure!(
            cross_market_error.to_string().contains("rejected"),
            "unexpected cross-market staging error: {cross_market_error:#}"
        );
        ensure!(l2_staging_count(&pool, artifact_id).await? == 0);

        repository
            .stage_binance_spot_l2_one_second_feature_batch(&claim, artifact_id, &[feature.clone()])
            .await?;
        ensure!(spot_l2_staging_count(&pool, artifact_id).await? == 1);
        ensure!(spot_l2_final_count(&pool, artifact_id).await? == 0);
        ensure!(l2_staging_count(&pool, artifact_id).await? == 0);
        ensure!(l2_final_count(&pool, artifact_id).await? == 0);

        repository
            .reset_binance_spot_l2_feature_staging(&claim, artifact_id)
            .await?;
        ensure!(spot_l2_staging_count(&pool, artifact_id).await? == 0);
        repository
            .stage_binance_spot_l2_one_second_feature_batch(&claim, artifact_id, &[feature.clone()])
            .await?;

        let cross_market_failure_error = repository
            .fail_binance_l2_artifact(&claim, artifact_id, "must remain spot")
            .await
            .expect_err("a spot claim must not invoke futures failure cleanup");
        ensure!(
            cross_market_failure_error.to_string().contains("rejected"),
            "unexpected cross-market failure error: {cross_market_failure_error:#}"
        );
        ensure!(spot_l2_staging_count(&pool, artifact_id).await? == 1);

        let failed = repository
            .fail_binance_spot_l2_artifact(&claim, artifact_id, "expected test failure")
            .await?;
        ensure!(failed.status == BackfillArtifactStatus::Failed);
        ensure!(spot_l2_staging_count(&pool, artifact_id).await? == 0);
        ensure!(spot_l2_final_count(&pool, artifact_id).await? == 0);
        ensure!(spot_l2_training_count(&pool, &feature).await? == 0);
        repository
            .complete(&claim, &BackfillJobSummary::default())
            .await?;
        Ok(())
    }
    .await;

    finish_committed_test(pool, &tag, outcome).await
}

#[tokio::test]
async fn spot_l2_publication_is_atomic_training_visible_and_immutable() -> Result<()> {
    let _guard = test_serializer().lock().await;
    let Some(pool) = connect_test_database().await? else {
        return Ok(());
    };
    let repository = IngestionRepository::from_pool(pool.clone());
    let tag = unique_tag();

    assert_no_runnable_jobs(&pool).await?;
    let day_start = l2_day_start();
    let queued = repository
        .enqueue(&request(
            IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures,
            day_start,
            day_start + ChronoDuration::days(1),
            format!("{tag}spot-l2-publish-job"),
        )?)
        .await?;
    let claim = repository
        .claim_next(&format!("{tag}spot-l2-publish-worker"), ACTIVE_LEASE)
        .await?
        .context("worker did not claim the Binance spot L2 publication test job")?;
    ensure!(claim.job.job_id == queued.job_id);
    let prepared =
        prepare_spot_l2_ingesting_artifact(&repository, &claim, &tag, day_start.date_naive())
            .await?;
    let artifact_id = prepared.artifact.artifact_id;
    let feature = spot_l2_feature(day_start + ChronoDuration::seconds(4));

    repository
        .stage_binance_spot_l2_one_second_feature_batch(&claim, artifact_id, &[feature.clone()])
        .await?;
    ensure!(spot_l2_staging_count(&pool, artifact_id).await? == 1);
    ensure!(spot_l2_final_count(&pool, artifact_id).await? == 0);
    ensure!(spot_l2_training_count(&pool, &feature).await? == 0);

    let wrong_contract = repository
        .publish_binance_spot_l2_one_second_features(
            &claim,
            artifact_id,
            &ArtifactCompletion {
                actual_checksum: "c".repeat(64),
                compressed_bytes: 2_048,
                record_count: 1,
                minimum_source_timestamp: Some(feature.source_event_timestamp),
                maximum_source_timestamp: Some(feature.source_event_timestamp),
                metadata: json!({
                    "materialization_contract":
                        "cryptohft-binance-futures-btcusdt-l2-features-v1"
                }),
            },
        )
        .await
        .expect_err("a futures contract must not publish spot rows");
    ensure!(
        wrong_contract.to_string().contains("contract"),
        "unexpected spot materialization-contract error: {wrong_contract:#}"
    );
    ensure!(spot_l2_staging_count(&pool, artifact_id).await? == 1);
    ensure!(spot_l2_final_count(&pool, artifact_id).await? == 0);

    let completed = repository
        .publish_binance_spot_l2_one_second_features(
            &claim,
            artifact_id,
            &ArtifactCompletion {
                actual_checksum: "c".repeat(64),
                compressed_bytes: 2_048,
                record_count: 1,
                minimum_source_timestamp: Some(feature.source_event_timestamp),
                maximum_source_timestamp: Some(feature.source_event_timestamp),
                metadata: json!({
                    "materialization_contract":
                        "cryptohft-binance-spot-btcusdt-l2-features-v1"
                }),
            },
        )
        .await?;
    ensure!(completed.status == BackfillArtifactStatus::Completed);
    ensure!(spot_l2_staging_count(&pool, artifact_id).await? == 0);
    ensure!(spot_l2_final_count(&pool, artifact_id).await? == 1);
    ensure!(spot_l2_training_count(&pool, &feature).await? == 1);
    ensure!(l2_final_count(&pool, artifact_id).await? == 0);
    ensure!(l2_training_count(&pool, &feature).await? == 0);
    repository
        .complete(&claim, &BackfillJobSummary::default())
        .await?;

    assert_spot_l2_feature_mutation_rejected(&pool, artifact_id, L2FeatureMutation::Update).await?;
    assert_spot_l2_feature_mutation_rejected(&pool, artifact_id, L2FeatureMutation::Delete).await?;

    pool.close().await;
    Ok(())
}

fn test_serializer() -> &'static Mutex<()> {
    TEST_SERIALIZER.get_or_init(|| Mutex::new(()))
}

async fn connect_test_database() -> Result<Option<PgPool>> {
    let database_url = match env::var(TEST_DATABASE_ENV) {
        Ok(value) if !value.trim().is_empty() => value,
        _ => {
            eprintln!("skipping database integration test: {TEST_DATABASE_ENV} is not set");
            return Ok(None);
        }
    };
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .acquire_timeout(StdDuration::from_secs(10))
        .connect(&database_url)
        .await
        .with_context(|| format!("failed to connect using {TEST_DATABASE_ENV}"))?;
    let schema_ready = sqlx::query_scalar::<_, bool>(
        r#"
        SELECT to_regclass('polymarket.backfill_jobs') IS NOT NULL
          AND to_regclass('polymarket.backfill_artifacts') IS NOT NULL
          AND to_regclass('polymarket.binance_aggregate_trades') IS NOT NULL
          AND to_regclass('market_data.binance_spot_btcusdt_one_second_ohlcv') IS NOT NULL
          AND to_regclass('polymarket.binance_btcusdt_l2_one_second_features') IS NOT NULL
          AND to_regclass('polymarket.binance_btcusdt_l2_one_second_features_staging') IS NOT NULL
          AND to_regclass('polymarket.binance_btcusdt_l2_training_features') IS NOT NULL
          AND to_regclass('polymarket.binance_spot_btcusdt_l2_one_second_features') IS NOT NULL
          AND to_regclass('polymarket.binance_spot_btcusdt_l2_one_second_features_staging') IS NOT NULL
          AND to_regclass('polymarket.binance_spot_btcusdt_l2_training_features') IS NOT NULL
          AND to_regclass('polymarket.cryptohft_request_budget') IS NOT NULL
          AND EXISTS (
            SELECT 1
            FROM polymarket.cryptohft_request_budget
            WHERE provider = 'cryptohftdata' AND spacing_milliseconds = 1100
          )
        "#,
    )
    .fetch_one(&pool)
    .await
    .context("failed to inspect the ingestion test schema")?;
    ensure!(
        schema_ready,
        "{TEST_DATABASE_ENV} must point to a migrated disposable test database"
    );
    Ok(Some(pool))
}

async fn connect_additional_test_database() -> Result<PgPool> {
    let database_url = env::var(TEST_DATABASE_ENV)
        .with_context(|| format!("{TEST_DATABASE_ENV} disappeared during the test"))?;
    PgPoolOptions::new()
        .max_connections(2)
        .acquire_timeout(StdDuration::from_secs(10))
        .connect(&database_url)
        .await
        .with_context(|| format!("failed to open a second pool using {TEST_DATABASE_ENV}"))
}

fn unique_tag() -> String {
    format!("{TEST_PREFIX}-{}-", Uuid::new_v4())
}

fn unique_day_start() -> DateTime<Utc> {
    let offset = i64::try_from(Uuid::new_v4().as_u128() % 5_000).expect("offset fits i64");
    let date = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid test date")
        + ChronoDuration::days(offset);
    date.and_hms_opt(0, 0, 0).expect("valid midnight").and_utc()
}

fn unique_five_minute_start() -> DateTime<Utc> {
    unique_day_start() + ChronoDuration::minutes(5)
}

fn l2_day_start() -> DateTime<Utc> {
    NaiveDate::from_ymd_opt(2026, 4, 14)
        .expect("valid Binance L2 test date")
        .and_hms_opt(0, 0, 0)
        .expect("valid UTC midnight")
        .and_utc()
}

fn unique_positive_i64() -> i64 {
    let value = Uuid::new_v4().as_u128() & 0x3fff_ffff_ffff_0000;
    i64::try_from(value).expect("masked UUID fits i64")
}

fn request(
    ingester: IngesterKey,
    range_start: DateTime<Utc>,
    range_end: DateTime<Utc>,
    idempotency_key: String,
) -> Result<ValidatedBackfillRequest> {
    Ok(BackfillRequest {
        ingester,
        request_version: BACKFILL_REQUEST_VERSION,
        range_start,
        range_end,
        parameters: json!({}),
        idempotency_key,
    }
    .validate()?)
}

async fn assert_no_runnable_jobs(pool: &PgPool) -> Result<()> {
    let supported = IngesterKey::ALL
        .iter()
        .map(|ingester| ingester.as_str().to_string())
        .collect::<Vec<_>>();
    let runnable = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)
        FROM polymarket.backfill_jobs
        WHERE ingester_key = ANY($1)
          AND (
            (status = 'queued' AND next_attempt_at <= now() AND attempt < max_attempts)
            OR
            (status IN ('running', 'cancel_requested')
              AND (lease_expires_at IS NULL OR lease_expires_at <= now()))
          )
        "#,
    )
    .bind(supported)
    .fetch_one(pool)
    .await?;
    ensure!(
        runnable == 0,
        "{TEST_DATABASE_ENV} is not isolated: {runnable} unrelated job(s) are claimable"
    );
    Ok(())
}

async fn prepare_ingesting_artifact(
    repository: &IngestionRepository,
    claim: &polymarket_bot::ingestion::job::ClaimedJob,
    ingester: IngesterKey,
    tag: &str,
    suffix: &str,
    source_date: NaiveDate,
) -> Result<polymarket_bot::ingestion::job::PreparedArtifact> {
    let prepared = repository
        .prepare_artifact(
            claim,
            &ArtifactSpec {
                job_id: claim.job.job_id,
                ingester,
                logical_key: format!("{tag}{suffix}-logical-key"),
                provider: format!("{tag}{suffix}-provider"),
                source_uri: format!("https://example.invalid/{tag}{suffix}.zip"),
                source_date: Some(source_date),
                expected_checksum: None,
                metadata: json!({"test": true}),
            },
        )
        .await?;
    ensure!(prepared.disposition == ArtifactDisposition::Process);
    repository
        .set_artifact_status(
            claim,
            prepared.artifact.artifact_id,
            BackfillArtifactStatus::Ingesting,
            json!({"test_status": "ingesting"}),
        )
        .await?;
    Ok(prepared)
}

async fn prepare_spot_l2_ingesting_artifact(
    repository: &IngestionRepository,
    claim: &polymarket_bot::ingestion::job::ClaimedJob,
    tag: &str,
    source_date: NaiveDate,
) -> Result<polymarket_bot::ingestion::job::PreparedArtifact> {
    let prepared = repository
        .prepare_artifact(
            claim,
            &ArtifactSpec {
                job_id: claim.job.job_id,
                ingester: IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures,
                logical_key: format!("{tag}spot-l2-publish-logical-key"),
                provider: "cryptohftdata".to_string(),
                source_uri: format!("https://example.invalid/{tag}spot-l2.parquet.zst"),
                source_date: Some(source_date),
                expected_checksum: None,
                metadata: json!({"test": true}),
            },
        )
        .await?;
    ensure!(prepared.disposition == ArtifactDisposition::Process);
    repository
        .set_artifact_status(
            claim,
            prepared.artifact.artifact_id,
            BackfillArtifactStatus::Ingesting,
            json!({"test_status": "ingesting"}),
        )
        .await?;
    Ok(prepared)
}

fn one_second_kline(open_timestamp: DateTime<Utc>) -> BinanceOneSecondKlineRecord {
    BinanceOneSecondKlineRecord {
        symbol: "BTCUSDT".to_string(),
        open_timestamp,
        close_timestamp: open_timestamp + ChronoDuration::milliseconds(999),
        open_price: dec!(50000.00),
        high_price: dec!(50002.00),
        low_price: dec!(49999.00),
        close_price: dec!(50001.00),
        base_volume: dec!(1.25),
        quote_volume: dec!(62500.00),
        trade_count: 4,
        taker_buy_base_volume: dec!(0.75),
        taker_buy_quote_volume: dec!(37500.00),
    }
}

fn l2_feature(second_start: DateTime<Utc>) -> BinanceL2OneSecondFeature {
    BinanceL2OneSecondFeature {
        symbol: "BTCUSDT".to_string(),
        second_start,
        source_event_timestamp: second_start + ChronoDuration::milliseconds(100),
        provider_received_at: second_start + ChronoDuration::milliseconds(150),
        available_at: second_start + ChronoDuration::milliseconds(250),
        source_update_id: unique_positive_i64(),
        feature_schema_version: "binance-btcusdt-l2-one-second-features-v1".to_string(),
        quality_status: "qualified".to_string(),
        midpoint: dec!(50000),
        microprice: dec!(50000.01),
        spread_bps: dec!(0.2),
        bid_depth_5: dec!(1),
        ask_depth_5: dec!(1),
        imbalance_5: dec!(0),
        bid_depth_10: dec!(2),
        ask_depth_10: dec!(2),
        imbalance_10: dec!(0),
        bid_depth_20: dec!(3),
        ask_depth_20: dec!(3),
        imbalance_20: dec!(0),
        bid_depth_slope_20: dec!(0.1),
        ask_depth_slope_20: dec!(0.1),
        bid_depth_concentration_20: dec!(0.5),
        ask_depth_concentration_20: dec!(0.5),
        bid_quote_replenishment_1s: dec!(0),
        ask_quote_replenishment_1s: dec!(0),
        bid_quote_churn_1s: dec!(0),
        ask_quote_churn_1s: dec!(0),
        midpoint_change_bps_1s: dec!(0),
        spread_bps_delta_1s: dec!(0),
        depth_20_change_bps_1s: dec!(0),
        imbalance_20_delta_1s: dec!(0),
        midpoint_change_bps_5s: dec!(0),
        spread_bps_delta_5s: dec!(0),
        depth_20_change_bps_5s: dec!(0),
        imbalance_20_delta_5s: dec!(0),
        midpoint_change_bps_15s: dec!(0),
        spread_bps_delta_15s: dec!(0),
        depth_20_change_bps_15s: dec!(0),
        imbalance_20_delta_15s: dec!(0),
        midpoint_change_bps_30s: dec!(0),
        spread_bps_delta_30s: dec!(0),
        depth_20_change_bps_30s: dec!(0),
        imbalance_20_delta_30s: dec!(0),
        midpoint_change_bps_60s: dec!(0),
        spread_bps_delta_60s: dec!(0),
        depth_20_change_bps_60s: dec!(0),
        imbalance_20_delta_60s: dec!(0),
    }
}

fn spot_l2_feature(second_start: DateTime<Utc>) -> BinanceL2OneSecondFeature {
    let mut feature = l2_feature(second_start);
    feature.feature_schema_version = "binance-spot-btcusdt-l2-one-second-features-v1".to_string();
    feature
}

async fn l2_staging_count(pool: &PgPool, artifact_id: Uuid) -> Result<i64> {
    Ok(sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)::bigint
        FROM polymarket.binance_btcusdt_l2_one_second_features_staging
        WHERE artifact_id = $1
        "#,
    )
    .bind(artifact_id)
    .fetch_one(pool)
    .await?)
}

async fn l2_final_count(pool: &PgPool, artifact_id: Uuid) -> Result<i64> {
    Ok(sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)::bigint
        FROM polymarket.binance_btcusdt_l2_one_second_features
        WHERE artifact_id = $1
        "#,
    )
    .bind(artifact_id)
    .fetch_one(pool)
    .await?)
}

async fn l2_training_count(pool: &PgPool, feature: &BinanceL2OneSecondFeature) -> Result<i64> {
    Ok(sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)::bigint
        FROM polymarket.binance_btcusdt_l2_training_features
        WHERE symbol = $1 AND second_start = $2 AND source_update_id = $3
        "#,
    )
    .bind(&feature.symbol)
    .bind(feature.second_start)
    .bind(feature.source_update_id)
    .fetch_one(pool)
    .await?)
}

async fn spot_l2_staging_count(pool: &PgPool, artifact_id: Uuid) -> Result<i64> {
    Ok(sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)::bigint
        FROM polymarket.binance_spot_btcusdt_l2_one_second_features_staging
        WHERE artifact_id = $1
        "#,
    )
    .bind(artifact_id)
    .fetch_one(pool)
    .await?)
}

async fn spot_l2_final_count(pool: &PgPool, artifact_id: Uuid) -> Result<i64> {
    Ok(sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)::bigint
        FROM polymarket.binance_spot_btcusdt_l2_one_second_features
        WHERE artifact_id = $1
        "#,
    )
    .bind(artifact_id)
    .fetch_one(pool)
    .await?)
}

async fn spot_l2_training_count(pool: &PgPool, feature: &BinanceL2OneSecondFeature) -> Result<i64> {
    Ok(sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)::bigint
        FROM polymarket.binance_spot_btcusdt_l2_training_features
        WHERE symbol = $1 AND second_start = $2 AND source_update_id = $3
        "#,
    )
    .bind(&feature.symbol)
    .bind(feature.second_start)
    .bind(feature.source_update_id)
    .fetch_one(pool)
    .await?)
}

async fn finish_committed_test(pool: PgPool, tag: &str, outcome: Result<()>) -> Result<()> {
    let cleanup = cleanup_tagged_rows(&pool, tag).await;
    pool.close().await;
    match (outcome, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(test_error), Ok(())) => Err(test_error),
        (Ok(()), Err(cleanup_error)) => Err(cleanup_error),
        (Err(test_error), Err(cleanup_error)) => Err(anyhow!(
            "test failed: {test_error:#}; cleanup also failed: {cleanup_error:#}"
        )),
    }
}

async fn cleanup_tagged_rows(pool: &PgPool, tag: &str) -> Result<()> {
    let pattern = format!("{tag}%");
    let mut transaction = pool.begin().await?;
    let immutable_facts = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)
        FROM polymarket.btc_market_reference_facts AS fact
        JOIN polymarket.backfill_artifacts AS artifact
          ON artifact.artifact_id = fact.artifact_id
        JOIN polymarket.backfill_jobs AS job ON job.job_id = artifact.job_id
        WHERE job.idempotency_key LIKE $1
        "#,
    )
    .bind(&pattern)
    .fetch_one(&mut *transaction)
    .await?;
    ensure!(
        immutable_facts == 0,
        "test cleanup cannot remove {immutable_facts} immutable reference fact(s)"
    );

    let immutable_l2_features = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)
        FROM polymarket.binance_btcusdt_l2_one_second_features AS feature
        JOIN polymarket.backfill_artifacts AS artifact
          ON artifact.artifact_id = feature.artifact_id
        JOIN polymarket.backfill_jobs AS job ON job.job_id = artifact.job_id
        WHERE job.idempotency_key LIKE $1
        "#,
    )
    .bind(&pattern)
    .fetch_one(&mut *transaction)
    .await?;
    ensure!(
        immutable_l2_features == 0,
        "test cleanup cannot remove {immutable_l2_features} immutable Binance L2 feature(s)"
    );

    let immutable_spot_l2_features = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)
        FROM polymarket.binance_spot_btcusdt_l2_one_second_features AS feature
        JOIN polymarket.backfill_artifacts AS artifact
          ON artifact.artifact_id = feature.artifact_id
        JOIN polymarket.backfill_jobs AS job ON job.job_id = artifact.job_id
        WHERE job.idempotency_key LIKE $1
        "#,
    )
    .bind(&pattern)
    .fetch_one(&mut *transaction)
    .await?;
    ensure!(
        immutable_spot_l2_features == 0,
        "test cleanup cannot remove {immutable_spot_l2_features} immutable Binance spot L2 feature(s)"
    );

    sqlx::query(
        r#"
        DELETE FROM polymarket.binance_btcusdt_l2_one_second_features_staging
        WHERE artifact_id IN (
          SELECT artifact.artifact_id
          FROM polymarket.backfill_artifacts AS artifact
          JOIN polymarket.backfill_jobs AS job ON job.job_id = artifact.job_id
          WHERE job.idempotency_key LIKE $1
        )
        "#,
    )
    .bind(&pattern)
    .execute(&mut *transaction)
    .await?;

    sqlx::query(
        r#"
        DELETE FROM polymarket.binance_spot_btcusdt_l2_one_second_features_staging
        WHERE artifact_id IN (
          SELECT artifact.artifact_id
          FROM polymarket.backfill_artifacts AS artifact
          JOIN polymarket.backfill_jobs AS job ON job.job_id = artifact.job_id
          WHERE job.idempotency_key LIKE $1
        )
        "#,
    )
    .bind(&pattern)
    .execute(&mut *transaction)
    .await?;

    sqlx::query(
        r#"
        DELETE FROM polymarket.binance_aggregate_trades
        WHERE artifact_id IN (
          SELECT artifact.artifact_id
          FROM polymarket.backfill_artifacts AS artifact
          JOIN polymarket.backfill_jobs AS job ON job.job_id = artifact.job_id
          WHERE job.idempotency_key LIKE $1
        )
        "#,
    )
    .bind(&pattern)
    .execute(&mut *transaction)
    .await?;
    sqlx::query(
        r#"
        DELETE FROM market_data.binance_spot_btcusdt_one_second_ohlcv
        WHERE artifact_id IN (
          SELECT artifact.artifact_id
          FROM polymarket.backfill_artifacts AS artifact
          JOIN polymarket.backfill_jobs AS job ON job.job_id = artifact.job_id
          WHERE job.idempotency_key LIKE $1
        )
        "#,
    )
    .bind(&pattern)
    .execute(&mut *transaction)
    .await?;
    sqlx::query(
        r#"
        DELETE FROM polymarket.backfill_artifacts AS artifact
        USING polymarket.backfill_jobs AS job
        WHERE artifact.job_id = job.job_id AND job.idempotency_key LIKE $1
        "#,
    )
    .bind(&pattern)
    .execute(&mut *transaction)
    .await?;
    sqlx::query("DELETE FROM polymarket.backfill_jobs WHERE idempotency_key LIKE $1")
        .bind(&pattern)
        .execute(&mut *transaction)
        .await?;
    transaction.commit().await?;

    let remaining = sqlx::query_scalar::<_, i64>(
        "SELECT count(*) FROM polymarket.backfill_jobs WHERE idempotency_key LIKE $1",
    )
    .bind(&pattern)
    .fetch_one(pool)
    .await?;
    ensure!(remaining == 0, "cleanup left {remaining} tagged job(s)");
    Ok(())
}

#[derive(Clone, Copy)]
enum CompletedArtifactMutation {
    Update,
    Delete,
}

#[derive(Clone, Copy)]
enum L2FeatureMutation {
    Update,
    Delete,
}

impl L2FeatureMutation {
    fn name(self) -> &'static str {
        match self {
            Self::Update => "update",
            Self::Delete => "delete",
        }
    }
}

async fn assert_l2_feature_mutation_rejected(
    pool: &PgPool,
    artifact_id: Uuid,
    mutation: L2FeatureMutation,
) -> Result<()> {
    let mutation_result = match mutation {
        L2FeatureMutation::Update => {
            sqlx::query(
                r#"
                UPDATE polymarket.binance_btcusdt_l2_one_second_features
                SET midpoint = midpoint + 1
                WHERE artifact_id = $1
                "#,
            )
            .bind(artifact_id)
            .execute(pool)
            .await
        }
        L2FeatureMutation::Delete => sqlx::query(
            "DELETE FROM polymarket.binance_btcusdt_l2_one_second_features WHERE artifact_id = $1",
        )
        .bind(artifact_id)
        .execute(pool)
        .await,
    };
    let sql_state = mutation_result
        .as_ref()
        .err()
        .and_then(sqlx::Error::as_database_error)
        .and_then(|error| error.code())
        .map(|code| code.into_owned());
    ensure!(
        mutation_result.is_err(),
        "immutable Binance L2 feature {} unexpectedly succeeded",
        mutation.name()
    );
    ensure!(
        sql_state.as_deref() == Some("23000"),
        "immutable Binance L2 feature {} returned SQLSTATE {:?}, expected 23000",
        mutation.name(),
        sql_state
    );
    ensure!(
        l2_final_count(pool, artifact_id).await? == 1,
        "immutable Binance L2 feature disappeared after rejected {}",
        mutation.name()
    );
    Ok(())
}

async fn assert_spot_l2_feature_mutation_rejected(
    pool: &PgPool,
    artifact_id: Uuid,
    mutation: L2FeatureMutation,
) -> Result<()> {
    let mutation_result = match mutation {
        L2FeatureMutation::Update => {
            sqlx::query(
                r#"
                UPDATE polymarket.binance_spot_btcusdt_l2_one_second_features
                SET midpoint = midpoint + 1
                WHERE artifact_id = $1
                "#,
            )
            .bind(artifact_id)
            .execute(pool)
            .await
        }
        L2FeatureMutation::Delete => sqlx::query(
            "DELETE FROM polymarket.binance_spot_btcusdt_l2_one_second_features WHERE artifact_id = $1",
        )
        .bind(artifact_id)
        .execute(pool)
        .await,
    };
    let sql_state = mutation_result
        .as_ref()
        .err()
        .and_then(sqlx::Error::as_database_error)
        .and_then(|error| error.code())
        .map(|code| code.into_owned());
    ensure!(
        mutation_result.is_err(),
        "immutable Binance spot L2 feature {} unexpectedly succeeded",
        mutation.name()
    );
    ensure!(
        sql_state.as_deref() == Some("23000"),
        "immutable Binance spot L2 feature {} returned SQLSTATE {:?}, expected 23000",
        mutation.name(),
        sql_state
    );
    ensure!(
        spot_l2_final_count(pool, artifact_id).await? == 1,
        "immutable Binance spot L2 feature disappeared after rejected {}",
        mutation.name()
    );
    Ok(())
}

impl CompletedArtifactMutation {
    fn name(self) -> &'static str {
        match self {
            Self::Update => "update",
            Self::Delete => "delete",
        }
    }
}

async fn assert_completed_artifact_mutation_rejected(
    pool: &PgPool,
    mutation: CompletedArtifactMutation,
) -> Result<()> {
    let job_id = Uuid::new_v4();
    let artifact_id = Uuid::new_v4();
    let identity = format!("{TEST_PREFIX}-rollback-{artifact_id}");
    let start = unique_five_minute_start();
    let mut transaction = pool.begin().await?;
    sqlx::query(
        r#"
        INSERT INTO polymarket.backfill_jobs (
          job_id, ingester_key, request_version, status, range_start, range_end,
          idempotency_key, request, progress, checkpoint, summary, attempt, max_attempts,
          next_attempt_at, requested_at, started_at, completed_at, updated_at
        )
        VALUES (
          $1, 'btc_five_minute_markets', 1, 'completed', $2, $3, $4,
          '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, 1, 3,
          now(), now(), now(), now(), now()
        )
        "#,
    )
    .bind(job_id)
    .bind(start)
    .bind(start + ChronoDuration::minutes(5))
    .bind(&identity)
    .execute(&mut *transaction)
    .await?;
    sqlx::query(
        r#"
        INSERT INTO polymarket.backfill_artifacts (
          artifact_id, job_id, ingester_key, logical_key, provider, source_uri,
          checksum_algorithm, actual_checksum, compressed_bytes, record_count,
          status, metadata, completed_at
        )
        VALUES (
          $1, $2, 'btc_five_minute_markets', $3, $4, $5,
          'sha256', $6, 0, 0, 'completed', '{}'::jsonb, now()
        )
        "#,
    )
    .bind(artifact_id)
    .bind(job_id)
    .bind(format!("{identity}-logical"))
    .bind(format!("{identity}-provider"))
    .bind(format!("https://example.invalid/{identity}"))
    .bind("a".repeat(64))
    .execute(&mut *transaction)
    .await?;

    let mutation_result = match mutation {
        CompletedArtifactMutation::Update => {
            sqlx::query(
                "UPDATE polymarket.backfill_artifacts SET metadata = $2 WHERE artifact_id = $1",
            )
            .bind(artifact_id)
            .bind(json!({"mutated": true}))
            .execute(&mut *transaction)
            .await
        }
        CompletedArtifactMutation::Delete => {
            sqlx::query("DELETE FROM polymarket.backfill_artifacts WHERE artifact_id = $1")
                .bind(artifact_id)
                .execute(&mut *transaction)
                .await
        }
    };
    let sql_state = mutation_result
        .as_ref()
        .err()
        .and_then(sqlx::Error::as_database_error)
        .and_then(|error| error.code())
        .map(|code| code.into_owned());
    transaction
        .rollback()
        .await
        .context("failed to roll back immutable-artifact fixture")?;

    ensure!(
        mutation_result.is_err(),
        "completed artifact {} unexpectedly succeeded",
        mutation.name()
    );
    ensure!(
        sql_state.as_deref() == Some("23000"),
        "completed artifact {} returned SQLSTATE {:?}, expected 23000",
        mutation.name(),
        sql_state
    );
    let persisted = sqlx::query_scalar::<_, bool>(
        "SELECT EXISTS (SELECT 1 FROM polymarket.backfill_jobs WHERE job_id = $1)",
    )
    .bind(job_id)
    .fetch_one(pool)
    .await?;
    ensure!(
        !persisted,
        "rolled-back immutable-artifact fixture was unexpectedly committed"
    );
    Ok(())
}
