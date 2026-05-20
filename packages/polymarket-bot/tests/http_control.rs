use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::{to_bytes, Body},
    http::{header::AUTHORIZATION, Request, StatusCode},
};
use chrono::Utc;
use polymarket_bot::{
    execution::LiveVenueStatus,
    http::{
        self, BackfillJobResponse, BackfillJobsResponse, BackfillWhalesRequest,
        CancelBackfillJobResponse, ControlApi, CopyTradeBacktestRequest,
        CopyTradeCalibrationRequest, HttpError, MetricsResponse,
    },
    models::{BackfillJob, BackfillJobStatus},
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
        Ok(serde_json::json!({"positions": 1, "total_pnl": "1.23"}))
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
