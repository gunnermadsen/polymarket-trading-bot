use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::{to_bytes, Body},
    http::{header::AUTHORIZATION, Request, StatusCode},
};
use chrono::Utc;
use polymarket_bot::{
    execution::{
        LiveIdentityDiagnostics, LiveOrderDryRunDiagnostics, LiveOrderDryRunRequest,
        LivePoly1271FunderProbeCandidate, LivePoly1271FunderProbeRequest,
        LivePoly1271FunderProbeResponse, LiveVenueStatus, LiveWalletAddressDiagnostics,
        LiveWalletCandidateAddressDiagnostics, LiveWalletTokenBalances,
    },
    http::{
        self, BackfillJobResponse, BackfillJobsResponse, BackfillWhalesRequest,
        CancelBackfillJobResponse, ControlApi, CopyTradeBacktestRequest,
        CopyTradeCalibrationRequest, HttpError, MetricsResponse,
    },
    models::{
        BackfillJob, BackfillJobStatus, ProcessExecutionConfig, TradingProcess,
        TradingProcessConfig,
    },
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

    async fn start_whales_backfill(
        &self,
        request: BackfillWhalesRequest,
    ) -> Result<BackfillJobResponse, HttpError> {
        Ok(BackfillJobResponse {
            job: test_job(
                BackfillJobStatus::Queued,
                serde_json::json!({
                    "lookback_days": request.lookback_days,
                    "min_trade_usd": request.min_trade_usd,
                    "max_pages": request.max_pages,
                    "dry_run": request.dry_run
                }),
            ),
        })
    }

    async fn list_backfill_jobs(&self) -> Result<BackfillJobsResponse, HttpError> {
        Ok(BackfillJobsResponse {
            jobs: vec![test_job(
                BackfillJobStatus::Completed,
                serde_json::json!({}),
            )],
        })
    }

    async fn start_copy_trade_backtest(
        &self,
        request: CopyTradeBacktestRequest,
    ) -> Result<BackfillJobResponse, HttpError> {
        Ok(BackfillJobResponse {
            job: test_job(
                BackfillJobStatus::Queued,
                serde_json::json!({
                    "mode": "copy_trade_backtest",
                    "lookback_days": request.lookback_days,
                    "min_wallet_score": request.min_wallet_score,
                    "execute_signals": request.execute_signals,
                    "dry_run": request.dry_run
                }),
            ),
        })
    }

    async fn start_copy_trade_calibration(
        &self,
        request: CopyTradeCalibrationRequest,
    ) -> Result<BackfillJobResponse, HttpError> {
        Ok(BackfillJobResponse {
            job: test_job(
                BackfillJobStatus::Queued,
                serde_json::json!({
                    "mode": "copy_trade_calibration",
                    "lookback_days": request.lookback_days,
                    "dry_run": request.dry_run
                }),
            ),
        })
    }

    async fn replay_existing_copy_trades(
        &self,
        _request: http::CopyTradeReplayRequest,
    ) -> Result<Value, HttpError> {
        Ok(serde_json::json!({
            "trades_replayed": 1,
            "summary": {
                "trades_evaluated": 1,
                "signals_inserted": 1
            }
        }))
    }

    async fn get_backfill_job(&self, _job_id: Uuid) -> Result<BackfillJobResponse, HttpError> {
        Ok(BackfillJobResponse {
            job: test_job(BackfillJobStatus::Completed, serde_json::json!({})),
        })
    }

    async fn cancel_backfill_job(
        &self,
        job_id: Uuid,
    ) -> Result<CancelBackfillJobResponse, HttpError> {
        Ok(CancelBackfillJobResponse {
            job_id,
            status: BackfillJobStatus::CancelRequested,
            cancel_requested: true,
        })
    }

    async fn trade_pnl_summary(&self) -> Result<Value, HttpError> {
        Ok(serde_json::json!({
            "positions": 1,
            "total_pnl": "1.23",
            "reports_by_process_id": {
                "unbound": {
                    "positions": 1,
                    "total_pnl": "1.23"
                }
            }
        }))
    }

    async fn trade_pnl_wallets(
        &self,
        _request: http::TradePnlListRequest,
    ) -> Result<Value, HttpError> {
        Ok(serde_json::json!([]))
    }

    async fn trade_pnl_open_positions(
        &self,
        _request: http::TradePnlListRequest,
    ) -> Result<Value, HttpError> {
        Ok(serde_json::json!([]))
    }

    async fn trade_pnl_mark_health(
        &self,
        _request: http::TradePnlListRequest,
    ) -> Result<Value, HttpError> {
        Ok(serde_json::json!({
            "coverage": [],
            "failure_reasons": [],
            "unmarked_availability": []
        }))
    }

    async fn trade_pnl_recent_exits(
        &self,
        _request: http::TradePnlListRequest,
    ) -> Result<Value, HttpError> {
        Ok(serde_json::json!([]))
    }

    async fn trade_pnl_backfill(&self) -> Result<Value, HttpError> {
        Ok(serde_json::json!({"positions_backfilled": 1}))
    }

    async fn trade_pnl_mark_now(&self) -> Result<Value, HttpError> {
        Ok(serde_json::json!({"marks_written": 1}))
    }

    async fn recompute_mrs_scores(
        &self,
        _request: http::MrsRecomputeRequest,
    ) -> Result<Value, HttpError> {
        Ok(serde_json::json!({
            "score_version": "mrs_v1",
            "updated_wallets": 1,
            "top_scores": []
        }))
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
                "default-env-copy-trade".to_string(),
                "copy_trade".to_string(),
                "default".to_string(),
                Some("default-env-copy-trade".to_string()),
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
                "copy_trade".to_string(),
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
                    .unwrap_or_else(|| "copy_trade".to_string()),
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
                "copy_trade".to_string(),
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
                "copy_trade".to_string(),
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
                "signals": {"total": 1},
                "orders": {"total": 1},
                "fills": {"total": 1},
                "positions": {"total": 1, "total_pnl": "0"}
            }),
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
                signal_candidates_deleted: 1,
                copy_trade_signals_deleted: 1,
                trade_marks_deleted: 3,
                trade_exits_deleted: 2,
                trade_positions_deleted: 1,
                wallet_performance_deleted: 1,
                process_events_deleted: 1,
                copy_trade_backtest_results_deleted: 0,
                copy_trade_backtest_runs_deleted: 0,
                copy_trade_backtests_deleted: 0,
                backfill_job_events_deleted: 0,
                backfill_jobs_deleted: 0,
                whale_poll_checkpoints_deleted: 1,
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
async fn authenticated_admin_can_start_whale_backfill() {
    let app = http::router(Arc::new(FakeControlApi), "secret");
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/backfill/whales")
                .header(AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"lookback_days":14,"min_trade_usd":"2500","max_pages":25,"dry_run":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["job"]["status"], "queued");
    assert_eq!(json["job"]["request"]["lookback_days"], 14);
    assert_eq!(json["job"]["request"]["max_pages"], 25);
    assert_eq!(json["job"]["request"]["dry_run"], true);
}

#[tokio::test]
async fn authenticated_admin_can_start_copy_trade_backtest() {
    let app = http::router(Arc::new(FakeControlApi), "secret");
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/copy-trade/backtest")
                .header(AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"lookback_days":7,"min_wallet_score":"60","execute_signals":true,"dry_run":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["job"]["request"]["mode"], "copy_trade_backtest");
    assert_eq!(json["job"]["request"]["lookback_days"], 7);
    assert_eq!(json["job"]["request"]["execute_signals"], true);
}

#[tokio::test]
async fn authenticated_admin_can_start_copy_trade_calibration() {
    let app = http::router(Arc::new(FakeControlApi), "secret");
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/copy-trade/calibration")
                .header(AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"lookback_days":21,"dry_run":true}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["job"]["request"]["mode"], "copy_trade_calibration");
    assert_eq!(json["job"]["request"]["lookback_days"], 21);
}

#[tokio::test]
async fn authenticated_admin_can_read_pnl_stats() {
    let app = http::router(Arc::new(FakeControlApi), "secret");
    for uri in ["/admin/pnl/stats", "/admin/trades/pnl/summary"] {
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

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["positions"], 1);
        assert_eq!(json["total_pnl"], "1.23");
        assert_eq!(json["reports_by_process_id"]["unbound"]["positions"], 1);
        assert_eq!(
            json["reports_by_process_id"]["unbound"]["total_pnl"],
            "1.23"
        );
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
                    r#"{"name":"paper-canary","process_type":"copy_trade","enabled":true,"config":{"execution":{"mode":"paper","execute_signals":true,"live_capital":false}}}"#,
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
                .uri("/admin/trading-processes/by-key/prod-sim-copy-trade-canary")
                .header(AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"prod-sim-copy-trade-canary","process_type":"copy_trade","process_scope":"production","enabled":true,"status":"running","config":{"execution":{"mode":"sim","execute_signals":true,"live_capital":false}}}"#,
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
        "prod-sim-copy-trade-canary"
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

fn test_job(status: BackfillJobStatus, request: Value) -> BackfillJob {
    BackfillJob {
        job_id: Uuid::new_v4(),
        job_type: "whales".to_string(),
        status,
        requested_at: Utc::now(),
        started_at: None,
        completed_at: None,
        cancel_requested_at: None,
        lookback_days: 30,
        min_trade_usd: Decimal::from(1000),
        request,
        summary: serde_json::json!({}),
        error: None,
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
