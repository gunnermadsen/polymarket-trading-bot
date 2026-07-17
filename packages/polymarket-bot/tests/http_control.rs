use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use axum::{
    body::{to_bytes, Body},
    http::{header::AUTHORIZATION, Request, StatusCode},
};
use chrono::{DateTime, Utc};
use polymarket_bot::{
    execution::{
        LiveIdentityDiagnostics, LiveOrderDryRunDiagnostics, LiveOrderDryRunRequest,
        LivePoly1271FunderProbeCandidate, LivePoly1271FunderProbeRequest,
        LivePoly1271FunderProbeResponse, LiveVenueStatus, LiveWalletAddressDiagnostics,
        LiveWalletCandidateAddressDiagnostics, LiveWalletTokenBalances,
    },
    http::{self, ControlApi, HttpError, MetricsResponse},
    ingestion::job::{
        BackfillEventLevel as IngestionBackfillEventLevel, BackfillJob as IngestionBackfillJob,
        BackfillJobEvent as IngestionBackfillJobEvent,
        BackfillJobStatus as IngestionBackfillJobStatus,
        BackfillRequest as IngestionBackfillRequest, IngesterKey, TrainingReadiness,
    },
    models::{ProcessExecutionConfig, TradingProcess, TradingProcessConfig},
    store::TradingProcessResetReport,
};
use rust_decimal::Decimal;
use serde_json::Value;
use tower::ServiceExt;
use uuid::Uuid;

#[derive(Debug, Default)]
struct FakeControlApi;

#[async_trait]
impl ControlApi for FakeControlApi {
    async fn metrics(&self) -> Result<MetricsResponse, HttpError> {
        Ok(MetricsResponse {
            service: "polymarket-bot".to_string(),
            captured_at: Utc::now(),
            counters: serde_json::json!({"scans": 1}),
            gauges: serde_json::json!({}),
        })
    }

    async fn btc_realtime_status(&self) -> Result<Value, HttpError> {
        Ok(serde_json::json!({"enabled": true, "readiness": {"ready": true}}))
    }

    async fn btc_paper_experiment_status(&self) -> Result<Value, HttpError> {
        Ok(serde_json::json!({"configured": true, "experiment": {"status": "running"}}))
    }

    async fn enqueue_ingestion_backfill(
        &self,
        request: IngestionBackfillRequest,
    ) -> Result<http::IngestionBackfillEnqueueResponse, HttpError> {
        let request = request
            .validate()
            .map_err(|error| HttpError::bad_request(error.to_string()))?;
        let persisted_request = request.persisted_request();
        let mut job = test_ingestion_job(
            Uuid::new_v4(),
            IngestionBackfillJobStatus::Queued,
            request.ingester,
        );
        job.request_version = request.request_version;
        job.range_start = Some(request.range_start);
        job.range_end = Some(request.range_end);
        job.idempotency_key = Some(request.idempotency_key);
        job.request = persisted_request;
        job.progress = serde_json::json!({
            "expected_work_units": request.expected_work_units,
            "completed_work_units": 0,
            "records_read": 0,
            "records_committed": 0,
            "bytes_downloaded": 0
        });
        Ok(job.into())
    }

    async fn list_ingestion_backfills(
        &self,
        _request: http::ListIngestionBackfillsRequest,
    ) -> Result<http::IngestionBackfillJobsResponse, HttpError> {
        Ok(http::IngestionBackfillJobsResponse {
            jobs: vec![test_ingestion_job(
                Uuid::new_v4(),
                IngestionBackfillJobStatus::Completed,
                IngesterKey::BinanceBtcusdtAggTrades,
            )],
        })
    }

    async fn get_ingestion_backfill(
        &self,
        job_id: Uuid,
    ) -> Result<http::IngestionBackfillJobResponse, HttpError> {
        Ok(http::IngestionBackfillJobResponse {
            job: test_ingestion_job(
                job_id,
                IngestionBackfillJobStatus::Running,
                IngesterKey::BtcFiveMinuteMarkets,
            ),
        })
    }

    async fn list_ingestion_backfill_events(
        &self,
        job_id: Uuid,
        _request: http::ListIngestionBackfillEventsRequest,
    ) -> Result<http::IngestionBackfillEventsResponse, HttpError> {
        Ok(http::IngestionBackfillEventsResponse {
            job_id,
            events: vec![IngestionBackfillJobEvent {
                event_id: Uuid::new_v4(),
                job_id,
                timestamp_utc: Utc::now(),
                level: IngestionBackfillEventLevel::Info,
                message: "fixture checkpoint committed".to_string(),
                metadata: serde_json::json!({"committed_work_units": 1}),
            }],
        })
    }

    async fn cancel_ingestion_backfill(
        &self,
        job_id: Uuid,
    ) -> Result<http::IngestionBackfillCancelResponse, HttpError> {
        Ok(http::IngestionBackfillCancelResponse {
            job_id,
            status: IngestionBackfillJobStatus::CancelRequested,
            cancel_requested: true,
        })
    }

    async fn ingestion_training_readiness(
        &self,
        request: http::IngestionReadinessRequest,
    ) -> Result<TrainingReadiness, HttpError> {
        Ok(TrainingReadiness {
            range_start: request.range_start,
            range_end: request.range_end,
            expected_markets: 288,
            valid_market_identities: 287,
            opening_boundaries: 286,
            final_prices: 285,
            official_outcomes: 284,
            aggregate_trade_covered_markets: 283,
            one_second_kline_covered_markets: 282,
            usable_markets: 281,
            aggregate_trade_min_timestamp: Some(request.range_start),
            aggregate_trade_max_timestamp: Some(request.range_end),
            one_second_kline_min_timestamp: Some(request.range_start),
            one_second_kline_max_timestamp: Some(request.range_end),
            missing_by_reason: BTreeMap::from([("missing_final_price".to_string(), 3)]),
            artifact_status_counts: BTreeMap::from([("completed".to_string(), 2)]),
        })
    }

    async fn live_status(&self) -> Result<LiveVenueStatus, HttpError> {
        Ok(LiveVenueStatus {
            mode: "sim".to_string(),
            live_confirmed: false,
            order_submit_enabled: false,
            user_ws_enabled: false,
            user_ws_connected: false,
            last_user_ws_pong_age_secs: None,
            last_rest_reconcile_age_secs: None,
            idempotency_clean: true,
            unresolved_live_order_count: 0,
            max_order_notional_usd: Decimal::ZERO,
            max_open_notional_usd: Decimal::ZERO,
            entries_enabled: false,
            reason: Some("sim_mode".to_string()),
        })
    }

    async fn live_identity_diagnostics(&self) -> Result<LiveIdentityDiagnostics, HttpError> {
        Ok(LiveIdentityDiagnostics {
            mode: "live".to_string(),
            clob_api_base_url: "https://clob.polymarket.com".to_string(),
            signer_address: Some("0x0000000000000000000000000000000000000001".to_string()),
            configured_funder_address: Some(
                "0x0000000000000000000000000000000000000002".to_string(),
            ),
            configured_signature_type: Some("3".to_string()),
            resolved_signature_type: Some("Poly1271".to_string()),
            authenticated_client_address: Some(
                "0x0000000000000000000000000000000000000001".to_string(),
            ),
            credentials_present: true,
            api_keys_readable: true,
            api_keys_error: None,
            balance_allowance_readable: true,
            balance_allowance_error: None,
            collateral_balance: Some("20".to_string()),
            open_orders_readable: true,
            open_orders_error: None,
            open_orders_count: Some(0),
            checked_at: Utc::now(),
        })
    }

    async fn live_wallet_address_diagnostics(
        &self,
        candidate_addresses: Vec<String>,
    ) -> Result<LiveWalletAddressDiagnostics, HttpError> {
        Ok(LiveWalletAddressDiagnostics {
            mode: "live".to_string(),
            signer_address: Some("0x0000000000000000000000000000000000000001".to_string()),
            configured_funder_address: Some(
                "0x0000000000000000000000000000000000000002".to_string(),
            ),
            configured_signature_type: Some("3".to_string()),
            resolved_signature_type: Some("Poly1271".to_string()),
            authenticated_client_address: Some(
                "0x0000000000000000000000000000000000000001".to_string(),
            ),
            derived_proxy_wallet_address: Some(
                "0x0000000000000000000000000000000000000003".to_string(),
            ),
            derived_safe_wallet_address: Some(
                "0x0000000000000000000000000000000000000004".to_string(),
            ),
            expected_order_maker_address: Some(
                "0x0000000000000000000000000000000000000002".to_string(),
            ),
            expected_order_signer_field: Some(
                "0x0000000000000000000000000000000000000002".to_string(),
            ),
            configured_funder_matches_signer: Some(false),
            configured_funder_matches_proxy_wallet: Some(false),
            configured_funder_matches_safe_wallet: Some(false),
            configured_funder_deployed_as_deposit_wallet: Some(true),
            configured_funder_deployed_as_deposit_wallet_error: None,
            relayer_base_url: Some("https://relayer-v2.polymarket.com".to_string()),
            relayer_deployment_check_url: Some(
                "https://relayer-v2.polymarket.com/deployed?address=0x0000000000000000000000000000000000000002&type=WALLET"
                    .to_string(),
            ),
            signer_balances: Some(LiveWalletTokenBalances {
                address: "0x0000000000000000000000000000000000000001".to_string(),
                pol_wei: Some("1".to_string()),
                pusd: Some("20".to_string()),
                usdc_e: Some("0".to_string()),
                native_usdc: Some("0".to_string()),
                error: None,
            }),
            configured_funder_balances: Some(LiveWalletTokenBalances {
                address: "0x0000000000000000000000000000000000000002".to_string(),
                pol_wei: Some("0".to_string()),
                pusd: Some("30".to_string()),
                usdc_e: Some("0".to_string()),
                native_usdc: Some("0".to_string()),
                error: None,
            }),
            candidate_addresses: candidate_addresses
                .into_iter()
                .map(|address| LiveWalletCandidateAddressDiagnostics {
                    address: address.clone(),
                    matches_signer: Some(false),
                    matches_configured_funder: Some(false),
                    matches_authenticated_client: Some(false),
                    matches_proxy_wallet: Some(false),
                    matches_safe_wallet: Some(false),
                    deployed_as_deposit_wallet: Some(false),
                    deployed_as_deposit_wallet_error: None,
                    deposit_wallet_deployment_check_url: Some(
                        "https://relayer-v2.polymarket.com/deployed?address=0x0000000000000000000000000000000000000005&type=WALLET"
                            .to_string(),
                    ),
                    deployed_as_safe_wallet: Some(false),
                    deployed_as_safe_wallet_error: None,
                    safe_wallet_deployment_check_url: Some(
                        "https://relayer-v2.polymarket.com/deployed?address=0x0000000000000000000000000000000000000005&type=SAFE"
                            .to_string(),
                    ),
                    balances: None,
                    poly1271_authenticated_client_address: Some(address),
                    poly1271_api_keys_readable: true,
                    poly1271_api_keys_error: None,
                    poly1271_balance_allowance_readable: true,
                    poly1271_balance_allowance_error: None,
                    poly1271_collateral_balance: Some("0".to_string()),
                    poly1271_open_orders_readable: true,
                    poly1271_open_orders_error: None,
                    poly1271_open_orders_count: Some(0),
                })
                .collect(),
            verified_deposit_wallet_address: Some(
                "0x0000000000000000000000000000000000000002".to_string(),
            ),
            verified_deposit_wallet_candidates_count: 1,
            checked_at: Utc::now(),
        })
    }

    async fn live_order_dry_run(
        &self,
        _request: LiveOrderDryRunRequest,
    ) -> Result<LiveOrderDryRunDiagnostics, HttpError> {
        Ok(LiveOrderDryRunDiagnostics {
            mode: "live".to_string(),
            clob_api_base_url: "https://clob.polymarket.com".to_string(),
            signer_address: Some("0x0000000000000000000000000000000000000001".to_string()),
            configured_funder_address: Some(
                "0x0000000000000000000000000000000000000002".to_string(),
            ),
            configured_signature_type: Some("3".to_string()),
            resolved_signature_type: Some("Poly1271".to_string()),
            authenticated_client_address: Some(
                "0x0000000000000000000000000000000000000001".to_string(),
            ),
            order_signer: Some("0x0000000000000000000000000000000000000002".to_string()),
            order_maker: Some("0x0000000000000000000000000000000000000002".to_string()),
            order_signature_type: Some("3".to_string()),
            order_signer_matches_authenticated_client: Some(false),
            order_signer_matches_configured_funder: Some(true),
            order_maker_matches_configured_funder: Some(true),
            owner_redacted: true,
            signature_redacted: true,
            signed_order: serde_json::json!({
                "owner": "<redacted>",
                "order": {
                    "maker": "0x0000000000000000000000000000000000000002",
                    "signer": "0x0000000000000000000000000000000000000002",
                    "signatureType": "3",
                    "signature": "<redacted>"
                }
            }),
            checked_at: Utc::now(),
        })
    }

    async fn live_poly1271_funder_probe(
        &self,
        request: LivePoly1271FunderProbeRequest,
    ) -> Result<LivePoly1271FunderProbeResponse, HttpError> {
        let order_type = request
            .order_type
            .unwrap_or(polymarket_bot::models::OrderType::Fok);
        let verified = request.addresses.first().cloned();
        Ok(LivePoly1271FunderProbeResponse {
            mode: "live".to_string(),
            clob_api_base_url: "https://clob.polymarket.com".to_string(),
            signer_address: Some("0x0000000000000000000000000000000000000001".to_string()),
            token_id: request.token_id,
            side: request.side,
            order_type,
            price: request.price,
            size: request.size,
            candidates: request
                .addresses
                .into_iter()
                .map(|address| LivePoly1271FunderProbeCandidate {
                    address: address.clone(),
                    address_valid: true,
                    derive_credentials_ok: true,
                    derive_credentials_error: None,
                    authenticated_client_address: Some(
                        "0x0000000000000000000000000000000000000001".to_string(),
                    ),
                    api_keys_readable: true,
                    api_keys_error: None,
                    update_balance_allowance_ok: true,
                    update_balance_allowance_error: None,
                    balance_allowance_readable: true,
                    balance_allowance_error: None,
                    collateral_balance: Some("20".to_string()),
                    open_orders_readable: true,
                    open_orders_error: None,
                    open_orders_count: Some(0),
                    signed_order_build_ok: true,
                    signed_order_error: None,
                    signed_order_maker: Some(address.clone()),
                    signed_order_signer: Some(address.clone()),
                    signed_order_signature_type: Some("3".to_string()),
                    maker_matches_candidate: Some(true),
                    signer_matches_candidate: Some(true),
                    signature_type_is_poly1271: Some(true),
                    ready_for_live_canary: true,
                    signed_order: serde_json::json!({
                        "owner": "<redacted>",
                        "order": {
                            "maker": address,
                            "signer": address,
                            "signatureType": "3",
                            "signature": "<redacted>"
                        }
                    }),
                })
                .collect(),
            verified_funder_address: verified,
            verified_funder_candidates_count: 1,
            checked_at: Utc::now(),
        })
    }

    async fn live_halt(&self) -> Result<Value, HttpError> {
        Ok(serde_json::json!({"halted": true}))
    }

    async fn live_reconcile(&self) -> Result<Value, HttpError> {
        Ok(serde_json::json!({"open_orders": 0, "unresolved_count": 0}))
    }

    async fn live_set_entries_enabled(&self, enabled: bool) -> Result<LiveVenueStatus, HttpError> {
        Ok(LiveVenueStatus {
            mode: "live".to_string(),
            live_confirmed: true,
            order_submit_enabled: true,
            user_ws_enabled: true,
            user_ws_connected: true,
            last_user_ws_pong_age_secs: Some(1),
            last_rest_reconcile_age_secs: Some(1),
            idempotency_clean: true,
            unresolved_live_order_count: 0,
            max_order_notional_usd: Decimal::from(2),
            max_open_notional_usd: Decimal::from(30),
            entries_enabled: enabled,
            reason: (!enabled).then(|| "manual_disable".to_string()),
        })
    }

    async fn create_trading_process(
        &self,
        request: http::CreateTradingProcessRequest,
    ) -> Result<http::TradingProcessResponse, HttpError> {
        Ok(http::TradingProcessResponse {
            process: test_process(
                Uuid::new_v4(),
                request.name,
                request.process_type,
                request.process_scope,
                request.process_key,
                "created",
                request.enabled,
                request.config,
            ),
        })
    }

    async fn list_trading_processes(
        &self,
        _request: http::ListTradingProcessesRequest,
    ) -> Result<http::TradingProcessesResponse, HttpError> {
        Ok(http::TradingProcessesResponse {
            processes: vec![test_process(
                Uuid::new_v4(),
                "default-generic-process".to_string(),
                "generic".to_string(),
                "default".to_string(),
                Some("default-generic-process".to_string()),
                "running",
                true,
                TradingProcessConfig {
                    execution: Some(ProcessExecutionConfig {
                        mode: Some("sim".to_string()),
                        execute_signals: true,
                        live_capital: false,
                        taker_fee_rate: None,
                    }),
                    ..Default::default()
                },
            )],
        })
    }

    async fn get_trading_process(
        &self,
        process_id: Uuid,
    ) -> Result<http::TradingProcessResponse, HttpError> {
        Ok(http::TradingProcessResponse {
            process: test_process(
                process_id,
                "paper-canary".to_string(),
                "generic".to_string(),
                "default".to_string(),
                Some("paper-canary".to_string()),
                "running",
                true,
                TradingProcessConfig::default(),
            ),
        })
    }

    async fn update_trading_process(
        &self,
        process_id: Uuid,
        request: http::UpdateTradingProcessRequest,
    ) -> Result<http::TradingProcessResponse, HttpError> {
        Ok(http::TradingProcessResponse {
            process: test_process(
                process_id,
                request.name.unwrap_or_else(|| "paper-canary".to_string()),
                request
                    .process_type
                    .unwrap_or_else(|| "generic".to_string()),
                request
                    .process_scope
                    .unwrap_or_else(|| "default".to_string()),
                request
                    .process_key
                    .unwrap_or(Some("paper-canary".to_string())),
                request.status.as_deref().unwrap_or("created"),
                request.enabled.unwrap_or(true),
                request.config.unwrap_or_default(),
            ),
        })
    }

    async fn start_trading_process(
        &self,
        process_id: Uuid,
    ) -> Result<http::TradingProcessResponse, HttpError> {
        Ok(http::TradingProcessResponse {
            process: test_process(
                process_id,
                "paper-canary".to_string(),
                "generic".to_string(),
                "default".to_string(),
                Some("paper-canary".to_string()),
                "running",
                true,
                TradingProcessConfig::default(),
            ),
        })
    }

    async fn stop_trading_process(
        &self,
        process_id: Uuid,
    ) -> Result<http::TradingProcessResponse, HttpError> {
        Ok(http::TradingProcessResponse {
            process: test_process(
                process_id,
                "paper-canary".to_string(),
                "generic".to_string(),
                "default".to_string(),
                Some("paper-canary".to_string()),
                "stopped",
                false,
                TradingProcessConfig::default(),
            ),
        })
    }

    async fn upsert_trading_process_by_key(
        &self,
        process_key: String,
        request: http::UpsertTradingProcessByKeyRequest,
    ) -> Result<http::TradingProcessResponse, HttpError> {
        Ok(http::TradingProcessResponse {
            process: test_process(
                Uuid::new_v4(),
                request.name,
                request.process_type,
                request.process_scope,
                Some(process_key),
                request.status.as_str(),
                request.enabled,
                request.config,
            ),
        })
    }

    async fn get_trading_process_status(
        &self,
        process_id: Uuid,
    ) -> Result<http::TradingProcessStatusResponse, HttpError> {
        Ok(http::TradingProcessStatusResponse {
            process_id,
            status: serde_json::json!({
                "orders": {"total": 1},
                "fills": {"total": 1},
            }),
        })
    }

    async fn preview_trading_process_start(
        &self,
        process_id: Uuid,
    ) -> Result<http::TradingProcessStartPreviewResponse, HttpError> {
        let experiment_key = "btc-5m-paper-preview-test".to_string();
        let experiment_id = Uuid::new_v5(
            &Uuid::NAMESPACE_URL,
            format!("polymarket-bot/btc-paper/{experiment_key}").as_bytes(),
        );
        Ok(http::TradingProcessStartPreviewResponse {
            process_id,
            experiment_id,
            experiment_key,
            preregistration_sha256: "b".repeat(64),
            config_hash: "c".repeat(64),
            frozen_process_config: TradingProcessConfig {
                execution: Some(ProcessExecutionConfig {
                    mode: Some("paper".to_string()),
                    execute_signals: true,
                    live_capital: false,
                    taker_fee_rate: None,
                }),
                raw: serde_json::json!({
                    "pipeline_version": "btc_realtime_paper_pipeline_v11",
                    "process_schema_version": "btc_realtime_paper_process_v1",
                    "preregistration_sha256": "b".repeat(64),
                }),
                ..TradingProcessConfig::default()
            },
        })
    }

    async fn reset_trading_process_simulation(
        &self,
        process_id: Uuid,
    ) -> Result<http::TradingProcessResetResponse, HttpError> {
        Ok(http::TradingProcessResetResponse {
            report: TradingProcessResetReport {
                process_id,
                process_name: "paper-canary".to_string(),
                orders_deleted: 2,
                fills_deleted: 4,
                process_events_deleted: 1,
                backfill_job_events_deleted: 0,
                backfill_jobs_deleted: 0,
                process_stopped: true,
            },
        })
    }
}

#[tokio::test]
async fn health_is_public_and_admin_routes_require_bearer() {
    let app = http::router(Arc::new(FakeControlApi), "secret");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let process_id = Uuid::new_v4();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/admin/trading-processes/{process_id}/start-preview"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/admin/backfill/jobs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn generic_ingestion_admin_routes_require_bearer() {
    let app = http::router(Arc::new(FakeControlApi), "secret");
    let job_id = Uuid::new_v4();
    let cases = vec![
        ("GET", "/admin/backfill/ingesters".to_string(), ""),
        ("GET", "/admin/backfill/jobs".to_string(), ""),
        (
            "POST",
            "/admin/backfill/jobs".to_string(),
            r#"{"ingester":"btc_five_minute_markets"}"#,
        ),
        ("GET", format!("/admin/backfill/jobs/{job_id}"), ""),
        (
            "GET",
            format!("/admin/backfill/jobs/{job_id}/events"),
            "",
        ),
        (
            "POST",
            format!("/admin/backfill/jobs/{job_id}/cancel"),
            "",
        ),
        (
            "GET",
            "/admin/backfill/readiness/btc-five-minute-training?range_start=2026-01-01T00%3A00%3A00Z&range_end=2026-01-02T00%3A00%3A00Z"
                .to_string(),
            "",
        ),
    ];

    for (method, uri, body) in cases {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(&uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{uri}");
    }
}

#[tokio::test]
async fn authenticated_admin_can_list_generic_ingesters() {
    let app = http::router(Arc::new(FakeControlApi), "secret");
    let response = app
        .oneshot(
            Request::builder()
                .uri("/admin/backfill/ingesters")
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        json,
        serde_json::json!({
            "ingesters": [
                {
                    "key": "btc_five_minute_markets",
                    "request_version": 1,
                    "range_alignment_seconds": 300
                },
                {
                    "key": "btc_five_minute_resolutions",
                    "request_version": 1,
                    "range_alignment_seconds": 300
                },
                {
                    "key": "binance_btcusdt_agg_trades",
                    "request_version": 1,
                    "range_alignment_seconds": 86400
                },
                {
                    "key": "binance_btcusdt_one_second_klines",
                    "request_version": 1,
                    "range_alignment_seconds": 86400
                }
            ]
        })
    );
}

#[tokio::test]
async fn authenticated_admin_can_enqueue_one_generic_ingester_with_http_200() {
    let app = http::router(Arc::new(FakeControlApi), "secret");
    let request = serde_json::json!({
        "ingester": "btc_five_minute_markets",
        "request_version": 1,
        "range_start": "2026-01-01T00:00:00Z",
        "range_end": "2026-01-01T01:00:00Z",
        "parameters": {},
        "idempotency_key": "markets-2026-01-01-00"
    });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/backfill/jobs")
                .header(AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(Uuid::parse_str(json["job_id"].as_str().unwrap()).is_ok());
    assert_eq!(json["ingester"], "btc_five_minute_markets");
    assert_eq!(json["status"], "queued");
    assert!(json["requested_at"].is_string());
    assert!(json.get("job").is_none());

    let invalid_request = serde_json::json!({
        "ingester": "btc_five_minute_markets",
        "request_version": 1,
        "range_start": "2026-01-01T00:00:01Z",
        "range_end": "2026-01-01T01:00:00Z",
        "parameters": {},
        "idempotency_key": "unaligned-markets-range"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/backfill/jobs")
                .header(AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(invalid_request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["code"], "bad_request");
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("range_start must align to a 300-second UTC boundary"));
}

#[tokio::test]
async fn authenticated_admin_can_inspect_cancel_and_check_generic_backfills() {
    let app = http::router(Arc::new(FakeControlApi), "secret");
    let job_id = Uuid::new_v4();

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/admin/backfill/jobs?limit=10")
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["jobs"].as_array().unwrap().len(), 1);
    assert_eq!(json["jobs"][0]["ingester"], "binance_btcusdt_agg_trades");
    assert_eq!(json["jobs"][0]["status"], "completed");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/admin/backfill/jobs/{job_id}"))
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["job"]["job_id"], job_id.to_string());
    assert_eq!(json["job"]["ingester"], "btc_five_minute_markets");
    assert_eq!(json["job"]["status"], "running");
    assert!(json["job"].get("lease_token").is_none());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/admin/backfill/jobs/{job_id}/events?limit=25"))
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["job_id"], job_id.to_string());
    assert_eq!(json["events"].as_array().unwrap().len(), 1);
    assert_eq!(json["events"][0]["job_id"], job_id.to_string());
    assert_eq!(json["events"][0]["level"], "info");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/admin/backfill/jobs/{job_id}/cancel"))
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["job_id"], job_id.to_string());
    assert_eq!(json["status"], "cancel_requested");
    assert_eq!(json["cancel_requested"], true);

    let response = app
        .oneshot(
            Request::builder()
                .uri(
                    "/admin/backfill/readiness/btc-five-minute-training?range_start=2026-01-01T00%3A00%3A00Z&range_end=2026-01-02T00%3A00%3A00Z",
                )
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["range_start"], "2026-01-01T00:00:00Z");
    assert_eq!(json["range_end"], "2026-01-02T00:00:00Z");
    assert_eq!(json["expected_markets"], 288);
    assert_eq!(json["usable_markets"], 281);
    assert_eq!(json["missing_by_reason"]["missing_final_price"], 3);
    assert_eq!(json["artifact_status_counts"]["completed"], 2);
}

#[tokio::test]
async fn authenticated_admin_can_read_btc_experiment_status() {
    let app = http::router(Arc::new(FakeControlApi), "secret");
    for uri in [
        "/admin/strategy/btc-5m/readiness",
        "/admin/strategy/btc-5m/paper-experiment",
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(uri)
                    .header(AUTHORIZATION, "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{uri}");
    }
}

#[tokio::test]
async fn retired_whale_copy_routes_are_not_found() {
    let app = http::router(Arc::new(FakeControlApi), "secret");
    for uri in [
        "/admin/backfill/whales",
        "/admin/pnl/stats",
        "/admin/trades/pnl/summary",
        "/admin/trades/pnl/wallets",
        "/admin/trades/pnl/open-positions",
        "/admin/trades/pnl/mark-health",
        "/admin/trades/pnl/recent-exits",
        "/admin/copy-trade/expectancy-flow/cells",
        "/admin/copy-trade/expectancy-flow/wallet-cells",
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(uri)
                    .header(AUTHORIZATION, "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{uri}");
    }
}

#[tokio::test]
async fn retired_wallet_analytics_routes_are_not_found() {
    let app = http::router(Arc::new(FakeControlApi), "secret");
    for uri in [
        "/admin/wallets/mrs/recompute",
        "/admin/wallets/scoring/refresh/enqueue",
        "/admin/wallets/scoring/refresh/process",
        "/admin/wallets/scoring/refresh/jobs",
        "/admin/wallets/mrs/segments",
        "/admin/wallets/mrs/segments/recompute",
        "/admin/gamma/taxonomy/status",
        "/admin/gamma/taxonomy/backfill",
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(uri)
                    .header(AUTHORIZATION, "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{uri}");
    }
}

#[tokio::test]
async fn retired_replay_and_backtest_routes_are_not_found() {
    let app = http::router(Arc::new(FakeControlApi), "secret");
    for uri in [
        "/admin/copy-trade/backtest",
        "/admin/copy-trade/calibration",
        "/admin/copy-trade/replay-existing",
        "/admin/backtests/replay",
        "/admin/backtests",
        "/admin/backtests/00000000-0000-0000-0000-000000000000",
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(uri)
                    .header(AUTHORIZATION, "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{uri}");
    }
}

#[tokio::test]
async fn authenticated_admin_can_read_live_status_and_halt() {
    let app = http::router(Arc::new(FakeControlApi), "secret");
    let status_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/admin/live/status")
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(status_response.status(), StatusCode::OK);
    let status_body = to_bytes(status_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let status_json: Value = serde_json::from_slice(&status_body).unwrap();
    assert_eq!(status_json["mode"], "sim");
    assert_eq!(status_json["entries_enabled"], false);

    let reconcile_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/live/reconcile")
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reconcile_response.status(), StatusCode::OK);

    let diagnostics_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/admin/live/diagnostics")
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(diagnostics_response.status(), StatusCode::OK);
    let diagnostics_body = to_bytes(diagnostics_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let diagnostics_json: Value = serde_json::from_slice(&diagnostics_body).unwrap();
    assert_eq!(diagnostics_json["credentials_present"], true);
    assert_eq!(diagnostics_json["api_keys_readable"], true);

    let wallet_diagnostics_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/admin/live/wallet-diagnostics?address=0x0000000000000000000000000000000000000005")
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(wallet_diagnostics_response.status(), StatusCode::OK);
    let wallet_diagnostics_body = to_bytes(wallet_diagnostics_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let wallet_diagnostics_json: Value = serde_json::from_slice(&wallet_diagnostics_body).unwrap();
    assert_eq!(
        wallet_diagnostics_json["configured_funder_deployed_as_deposit_wallet"],
        true
    );
    assert_eq!(wallet_diagnostics_json["signer_balances"]["pusd"], "20");
    assert_eq!(
        wallet_diagnostics_json["candidate_addresses"][0]["address"],
        "0x0000000000000000000000000000000000000005"
    );

    let dry_run_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/live/order-dry-run")
                .header(AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "token_id": "123",
                        "side": "buy",
                        "order_type": "fok",
                        "price": "0.39",
                        "size": "5.12"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(dry_run_response.status(), StatusCode::OK);
    let dry_run_body = to_bytes(dry_run_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let dry_run_json: Value = serde_json::from_slice(&dry_run_body).unwrap();
    assert_eq!(
        dry_run_json["order_signer_matches_authenticated_client"],
        false
    );
    assert_eq!(dry_run_json["signature_redacted"], true);

    let funder_probe_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/live/poly1271-funder-probe")
                .header(AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "addresses": ["0x0000000000000000000000000000000000000002"],
                        "token_id": "123",
                        "side": "buy",
                        "order_type": "fok",
                        "price": "0.39",
                        "size": "5.12"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(funder_probe_response.status(), StatusCode::OK);
    let funder_probe_body = to_bytes(funder_probe_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let funder_probe_json: Value = serde_json::from_slice(&funder_probe_body).unwrap();
    assert_eq!(funder_probe_json["verified_funder_candidates_count"], 1);
    assert_eq!(
        funder_probe_json["candidates"][0]["maker_matches_candidate"],
        true
    );

    let enable_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/live/entries/enable")
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(enable_response.status(), StatusCode::OK);

    let halt_response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/live/halt")
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(halt_response.status(), StatusCode::OK);
}

#[tokio::test]
async fn authenticated_admin_can_manage_trading_processes() {
    let app = http::router(Arc::new(FakeControlApi), "secret");
    let create_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/trading-processes")
                .header(AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"paper-canary","process_type":"generic","enabled":true,"config":{"execution":{"mode":"paper","execute_signals":true,"live_capital":false}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_response.status(), StatusCode::OK);
    let create_body = to_bytes(create_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let create_json: Value = serde_json::from_slice(&create_body).unwrap();
    let process_id = create_json["process"]["process_id"].as_str().unwrap();
    assert_eq!(create_json["process"]["name"], "paper-canary");
    assert_eq!(
        create_json["process"]["config"]["execution"]["mode"],
        "paper"
    );

    let list_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/admin/trading-processes?limit=5")
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list_response.status(), StatusCode::OK);
    let list_body = to_bytes(list_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let list_json: Value = serde_json::from_slice(&list_body).unwrap();
    assert_eq!(list_json["processes"][0]["status"], "running");

    let upsert_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/admin/trading-processes/by-key/prod-sim-generic-canary")
                .header(AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"prod-sim-generic-canary","process_type":"generic","process_scope":"production","enabled":true,"status":"running","config":{"execution":{"mode":"sim","execute_signals":true,"live_capital":false}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(upsert_response.status(), StatusCode::OK);
    let upsert_body = to_bytes(upsert_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let upsert_json: Value = serde_json::from_slice(&upsert_body).unwrap();
    assert_eq!(
        upsert_json["process"]["process_key"],
        "prod-sim-generic-canary"
    );
    assert_eq!(upsert_json["process"]["process_scope"], "production");

    let status_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/admin/trading-processes/{process_id}/status"))
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(status_response.status(), StatusCode::OK);

    let update_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri(format!("/admin/trading-processes/{process_id}"))
                .header(AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"status":"created","enabled":false}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(update_response.status(), StatusCode::OK);

    let preview_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/admin/trading-processes/{process_id}/start-preview"
                ))
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(preview_response.status(), StatusCode::OK);
    let preview_body = to_bytes(preview_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let preview_json: Value = serde_json::from_slice(&preview_body).unwrap();
    assert_eq!(preview_json["process_id"], process_id);
    assert_eq!(preview_json["experiment_key"], "btc-5m-paper-preview-test");
    assert_eq!(
        preview_json["preregistration_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    assert_eq!(preview_json["config_hash"].as_str().unwrap().len(), 64);
    assert_eq!(
        preview_json["frozen_process_config"]["execution"]["mode"],
        "paper"
    );

    let start_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/admin/trading-processes/{process_id}/start"))
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(start_response.status(), StatusCode::OK);

    let reset_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!(
                    "/admin/trading-processes/{process_id}/reset-simulation"
                ))
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reset_response.status(), StatusCode::OK);
    let reset_body = to_bytes(reset_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let reset_json: Value = serde_json::from_slice(&reset_body).unwrap();
    assert_eq!(reset_json["report"]["process_id"], process_id);
    assert_eq!(reset_json["report"]["orders_deleted"], 2);
    assert_eq!(reset_json["report"]["process_stopped"], true);

    let stop_response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/admin/trading-processes/{process_id}/stop"))
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(stop_response.status(), StatusCode::OK);
}

fn test_ingestion_job(
    job_id: Uuid,
    status: IngestionBackfillJobStatus,
    ingester: IngesterKey,
) -> IngestionBackfillJob {
    let range_start = "2026-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
    let range_end = "2026-01-02T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
    let now = Utc::now();
    IngestionBackfillJob {
        job_id,
        ingester_key: ingester.to_string(),
        request_version: 1,
        status,
        range_start: Some(range_start),
        range_end: Some(range_end),
        idempotency_key: Some(format!("fixture-{ingester}")),
        request: serde_json::json!({
            "ingester": ingester,
            "request_version": 1,
            "range_start": range_start,
            "range_end": range_end,
            "parameters": {},
            "idempotency_key": format!("fixture-{ingester}")
        }),
        progress: serde_json::json!({
            "expected_work_units": 1,
            "completed_work_units": u8::from(status.is_terminal()),
            "records_read": 10,
            "records_committed": 10,
            "bytes_downloaded": 1024
        }),
        checkpoint: serde_json::json!({"committed_record_ordinal": 10}),
        summary: serde_json::json!({}),
        attempt: 1,
        max_attempts: 3,
        next_attempt_at: now,
        worker_id: None,
        lease_token: None,
        lease_expires_at: None,
        heartbeat_at: None,
        cancel_requested_at: None,
        requested_at: now,
        started_at: None,
        completed_at: status.is_terminal().then_some(now),
        error: None,
        updated_at: now,
        lookback_days: None,
        min_trade_usd: None,
    }
}

fn test_process(
    process_id: Uuid,
    name: String,
    process_type: String,
    process_scope: String,
    process_key: Option<String>,
    status: &str,
    enabled: bool,
    config: TradingProcessConfig,
) -> TradingProcess {
    TradingProcess {
        process_id,
        name,
        process_type,
        process_scope,
        process_key,
        status: status.to_string(),
        enabled,
        config,
        metadata: serde_json::json!({}),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        started_at: None,
        stopped_at: None,
        last_error: None,
    }
}
