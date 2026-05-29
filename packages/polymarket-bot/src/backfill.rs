use std::{collections::BTreeSet, sync::Arc};

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use tracing::warn;
use uuid::Uuid;

use crate::{
    clob::ClobClient,
    copytrade::{
        evaluate_copy_trade_with_segment, run_copy_trade_backtest, CopyTradeConfig,
        CopyTradeMrsScore, CopyTradeSegmentScore, CopyTradeWalletPerformance, ObservedMarket,
        COPY_SCORE_VERSION,
    },
    data_api::{ClosedPositionsQuery, DataApiClient, TradesQuery},
    execution::{execute_order_plan, ExecutionVenue, OrderPlan, OrderPlanReport},
    models::{
        BackfillJobStatus, CopyTradeBacktestRun, DataApiClosedPosition, OrderRequest, OrderState,
        WhaleTrade,
    },
    orderbook::{BookSide, LocalOrderBook},
    segments::{
        classify_trade_segment, score_wallet_segments_from_samples, SegmentClassification,
        GAMMA_SEGMENT_CLASSIFIER_VERSION,
    },
    store::{OrderbookSnapshot, Store, TradeMarkSourceFailure},
    trade_pnl::{refresh_trade_pnl_with_config, TradePnlConfig},
    wallets::{
        score_closed_position_performance, score_closed_position_wallets, score_mrs, score_wallets,
        MrsScoreInput,
    },
};

const POLYMARKET_TRADES_OFFSET_LIMIT: usize = 4000;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackfillMode {
    TradesOnly,
    WalletScoresOnly,
    TradesAndWalletScores,
    CopyTradeBacktest,
    Full,
}

impl Default for BackfillMode {
    fn default() -> Self {
        Self::Full
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleBackfillRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process_id: Option<Uuid>,
    #[serde(default = "default_lookback_days")]
    pub lookback_days: u32,
    #[serde(default = "default_min_trade_usd")]
    pub min_trade_usd: Decimal,
    #[serde(default)]
    pub market_ids: Vec<String>,
    #[serde(default)]
    pub event_ids: Vec<i64>,
    #[serde(default)]
    pub wallets: Vec<String>,
    #[serde(default)]
    pub mode: BackfillMode,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default = "default_max_pages")]
    pub max_pages: usize,
    #[serde(default = "default_copy_min_wallet_score")]
    pub copy_min_wallet_score: Decimal,
    #[serde(default = "default_copy_size_fraction")]
    pub copy_size_fraction: Decimal,
    #[serde(default = "default_copy_max_size_usd")]
    pub copy_max_size_usd: Decimal,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copy_trade_config: Option<CopyTradeConfig>,
    #[serde(default)]
    pub execute_signals: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackfillSummary {
    pub pages_fetched: usize,
    pub api_trades_seen: usize,
    pub trades_persisted: usize,
    pub wallets_scored: usize,
    pub wallet_performances_scored: usize,
    pub copy_trade_signals: usize,
    pub copy_trade_orders: usize,
    pub copy_trade_fills: usize,
    pub copy_trade_rejections: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<serde_json::Value>,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeRunConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process_id: Option<Uuid>,
    pub copy_trade: CopyTradeConfig,
    pub execute_signals: bool,
    #[serde(default = "default_require_entry_markability")]
    pub require_entry_markability: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CopyTradeRunSummary {
    pub trades_evaluated: usize,
    pub signals_inserted: usize,
    pub signals_detected: usize,
    pub orders_inserted: usize,
    pub fills_inserted: usize,
    pub rejections: usize,
    pub dry_run: bool,
}

pub struct ResolvedCopyTradeSignal<'a> {
    pub trade: &'a WhaleTrade,
    pub performance: Option<&'a CopyTradeWalletPerformance>,
    pub mrs_score: Option<&'a CopyTradeMrsScore>,
    pub segment_score: Option<&'a CopyTradeSegmentScore>,
    pub segment_classification: SegmentClassification,
    pub observed: ObservedMarket,
    pub order_metadata: serde_json::Value,
}

impl WhaleBackfillRequest {
    pub fn with_defaults(
        mut self,
        lookback_days: u32,
        min_trade_usd: Decimal,
        max_pages: usize,
    ) -> Self {
        if self.lookback_days == 0 {
            self.lookback_days = lookback_days;
        }
        if self.min_trade_usd == Decimal::ZERO {
            self.min_trade_usd = min_trade_usd;
        }
        if self.max_pages == 0 {
            self.max_pages = max_pages;
        }
        if self.copy_min_wallet_score == Decimal::ZERO {
            self.copy_min_wallet_score = default_copy_min_wallet_score();
        }
        if self.copy_size_fraction == Decimal::ZERO {
            self.copy_size_fraction = default_copy_size_fraction();
        }
        if self.copy_max_size_usd == Decimal::ZERO {
            self.copy_max_size_usd = default_copy_max_size_usd();
        }
        self
    }
}

pub async fn enqueue_and_spawn(
    store: Store,
    data_api: DataApiClient,
    request: WhaleBackfillRequest,
) -> Result<Uuid> {
    enqueue_and_spawn_with_venue(store, data_api, None, request).await
}

pub async fn enqueue_and_spawn_with_venue(
    store: Store,
    data_api: DataApiClient,
    venue: Option<Arc<dyn ExecutionVenue>>,
    request: WhaleBackfillRequest,
) -> Result<Uuid> {
    let job_id = Uuid::new_v4();
    store
        .create_backfill_job(
            job_id,
            request.lookback_days as i32,
            request.min_trade_usd,
            serde_json::to_value(&request)?,
        )
        .await?;
    tokio::spawn(async move {
        if let Err(error) = run_job(store.clone(), data_api, venue, job_id, request).await {
            let _ = store
                .complete_backfill_job(
                    job_id,
                    BackfillJobStatus::Failed,
                    serde_json::json!({}),
                    Some(error.to_string()),
                )
                .await;
        }
    });
    Ok(job_id)
}

pub async fn run_job(
    store: Store,
    data_api: DataApiClient,
    venue: Option<Arc<dyn ExecutionVenue>>,
    job_id: Uuid,
    request: WhaleBackfillRequest,
) -> Result<BackfillSummary> {
    store.mark_backfill_job_running(job_id).await?;
    store
        .insert_backfill_event(
            job_id,
            "info",
            "whale backfill started",
            serde_json::to_value(&request)?,
        )
        .await
        .ok();

    let mut summary = BackfillSummary {
        dry_run: request.dry_run,
        ..BackfillSummary::default()
    };
    let cutoff = Utc::now() - Duration::days(request.lookback_days as i64);
    let mut all_trades = Vec::new();

    if !matches!(request.mode, BackfillMode::WalletScoresOnly) {
        for page in 0..request.max_pages {
            let offset = page * request.limit;
            if offset >= POLYMARKET_TRADES_OFFSET_LIMIT {
                store
                    .insert_backfill_event(
                        job_id,
                        "info",
                        "whale backfill stopped at Polymarket trades offset limit",
                        serde_json::json!({
                            "page": page,
                            "offset": offset,
                            "offset_limit": POLYMARKET_TRADES_OFFSET_LIMIT,
                            "api_trades_seen": summary.api_trades_seen,
                            "trades_persisted": summary.trades_persisted
                        }),
                    )
                    .await
                    .ok();
                break;
            }
            let trades =
                fetch_whale_trade_page(&data_api, request.limit, offset, request.min_trade_usd)
                    .await
                    .with_context(|| format!("failed to fetch whale trades page {page}"))?;
            if trades.is_empty() {
                break;
            }
            summary.pages_fetched += 1;
            summary.api_trades_seen += trades.len();

            for trade in trades {
                if trade.timestamp_utc < cutoff {
                    continue;
                }
                if request.dry_run {
                    all_trades.push(trade);
                    continue;
                }
                store.ensure_whale_wallet(&trade).await?;
                if store.upsert_whale_trade(&trade).await? {
                    store.record_wallet_observed_trade(&trade).await?;
                    if let Err(error) = store.apply_cached_taxonomy_to_trade(&trade).await {
                        tracing::warn!(
                            error = %error,
                            trade_id = %trade.trade_id,
                            "failed to apply cached Gamma taxonomy to backfilled wallet trade"
                        );
                    }
                    summary.trades_persisted += 1;
                }
                all_trades.push(trade);
            }

            store
                .insert_backfill_event(
                    job_id,
                    "info",
                    "whale backfill page completed",
                    serde_json::json!({
                        "page": page,
                        "offset": offset,
                        "api_trades_seen": summary.api_trades_seen,
                        "trades_persisted": summary.trades_persisted
                    }),
                )
                .await
                .ok();
        }
    }

    if matches!(
        request.mode,
        BackfillMode::WalletScoresOnly | BackfillMode::TradesAndWalletScores | BackfillMode::Full
    ) {
        let mut trades_for_scores = if request.dry_run {
            all_trades.clone()
        } else {
            store.fetch_recent_whale_trades(cutoff).await?
        };
        if trades_for_scores.is_empty() {
            trades_for_scores = all_trades.clone();
        }
        let wallets = distinct_wallets_for_scoring(&request.wallets, &trades_for_scores);
        let mut closed_positions = Vec::with_capacity(wallets.len());
        for (index, wallet) in wallets.iter().enumerate() {
            let positions = fetch_closed_positions_for_wallet(
                &data_api,
                wallet,
                request.limit,
                request.max_pages,
            )
            .await?;
            summary.wallet_performances_scored += 1;
            if !request.dry_run {
                let performance_score = score_closed_position_performance(wallet, &positions);
                let performance = performance_score
                    .clone()
                    .into_wallet_performance(Utc::now(), serde_json::to_value(&positions)?);
                store.upsert_wallet_performance(&performance).await?;
                let observed_stats = store
                    .fetch_wallet_observed_trade_stats(wallet, cutoff)
                    .await?;
                let mut mrs_input = MrsScoreInput::from(&performance_score);
                mrs_input.observed_trade_count = observed_stats.observed_trade_count;
                mrs_input.observed_volume_usd = observed_stats.observed_volume_usd;
                mrs_input.observed_market_count = observed_stats.observed_market_count;
                mrs_input.avg_trade_size = observed_stats.avg_trade_size;
                let mut mrs_score = score_mrs(mrs_input).into_wallet_score();
                mrs_score.metadata = merge_json(
                    mrs_score.metadata,
                    serde_json::json!({
                        "source": "historic_backfill",
                        "lookback_days": request.lookback_days,
                        "min_trade_usd": request.min_trade_usd,
                        "sample_start": observed_stats.sample_start,
                        "sample_end": observed_stats.sample_end
                    }),
                );
                store.upsert_wallet_score(&mrs_score).await?;
                let observed_trades = store.fetch_wallet_observed_trades(wallet, cutoff).await?;
                for mut segment_performance in
                    score_wallet_segments_from_samples(wallet, &positions, &observed_trades)
                {
                    segment_performance.metadata = merge_json(
                        segment_performance.metadata,
                        serde_json::json!({
                            "source": "historic_backfill",
                            "lookback_days": request.lookback_days,
                            "min_trade_usd": request.min_trade_usd
                        }),
                    );
                    store
                        .upsert_wallet_segment_performance(&segment_performance)
                        .await?;
                }
            }
            closed_positions.push((wallet.clone(), positions));

            if !request.dry_run && (index + 1) % 25 == 0 {
                store
                    .insert_backfill_event(
                        job_id,
                        "info",
                        "wallet performance backfill progress",
                        serde_json::json!({
                            "wallets_processed": index + 1,
                            "wallets_total": wallets.len(),
                            "wallet_performances_scored": summary.wallet_performances_scored
                        }),
                    )
                    .await
                    .ok();
            }
        }
        let scores = score_closed_position_wallets(&closed_positions);
        summary.wallets_scored = scores.len();
        if !request.dry_run {
            for score in &scores {
                store.upsert_wallet_score(score).await?;
            }
        }
    }

    if matches!(
        request.mode,
        BackfillMode::CopyTradeBacktest | BackfillMode::Full
    ) {
        let copy_trade_config =
            request
                .copy_trade_config
                .clone()
                .unwrap_or_else(|| CopyTradeConfig {
                    min_wallet_score: request.copy_min_wallet_score,
                    copy_size_fraction: request.copy_size_fraction,
                    max_copy_size_usd: request.copy_max_size_usd,
                    ..CopyTradeConfig::default()
                });
        let trades_for_copy = if !request.dry_run {
            store.fetch_recent_whale_trades(cutoff).await?
        } else if all_trades.is_empty() {
            store.fetch_recent_whale_trades(cutoff).await?
        } else {
            all_trades.clone()
        };
        let copy_summary = run_copy_trade_signal_engine(
            &store,
            venue.as_deref(),
            None,
            &trades_for_copy,
            &CopyTradeRunConfig {
                process_id: request.process_id,
                copy_trade: copy_trade_config.clone(),
                execute_signals: request.execute_signals,
                require_entry_markability: true,
            },
            request.dry_run,
        )
        .await?;
        summary.copy_trade_signals = copy_summary.signals_inserted;
        summary.copy_trade_orders = copy_summary.orders_inserted;
        summary.copy_trade_fills = copy_summary.fills_inserted;
        summary.copy_trade_rejections = copy_summary.rejections;
        let backtest_config = CopyTradeRunConfig {
            process_id: request.process_id,
            copy_trade: copy_trade_config.clone(),
            execute_signals: request.execute_signals,
            require_entry_markability: true,
        };
        let calibration = calibrate_copy_trade_thresholds(&trades_for_copy, &backtest_config);
        if !request.dry_run {
            let scores = score_wallets(&trades_for_copy);
            let (backtest_result, calibration_snapshot) =
                run_copy_trade_backtest(&trades_for_copy, &scores, &backtest_config.copy_trade);
            let run = CopyTradeBacktestRun {
                backtest_id: backtest_result.backtest_id,
                job_id: Some(job_id),
                status: "completed".to_string(),
                score_version: COPY_SCORE_VERSION.to_string(),
                strategy_name: "whale_follow_v1".to_string(),
                range_start: calibration_snapshot.sample_start,
                range_end: calibration_snapshot.sample_end,
                config: serde_json::json!({
                    "request": request,
                    "source": "backfill"
                }),
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
                error: None,
            };
            store.upsert_copy_trade_backtest_run(&run).await?;
            store
                .insert_copy_trade_backtest_result(&backtest_result)
                .await?;
            store
                .insert_wallet_score_calibration_snapshot(&calibration_snapshot)
                .await?;
        }
        summary.calibration = Some(calibration);
    }

    store
        .complete_backfill_job(
            job_id,
            BackfillJobStatus::Completed,
            serde_json::to_value(&summary)?,
            None,
        )
        .await?;
    store
        .insert_backfill_event(
            job_id,
            "info",
            "whale backfill completed",
            serde_json::to_value(&summary)?,
        )
        .await
        .ok();
    Ok(summary)
}

pub async fn fetch_whale_trade_page(
    data_api: &DataApiClient,
    limit: usize,
    offset: usize,
    min_trade_usd: Decimal,
) -> Result<Vec<WhaleTrade>> {
    let api_trades = data_api
        .fetch_trades(&TradesQuery::whale_page(limit, offset, min_trade_usd))
        .await?;
    Ok(api_trades
        .into_iter()
        .filter_map(|api_trade| api_trade.into_whale_trade(min_trade_usd))
        .collect())
}

pub fn distinct_wallets_for_scoring(
    request_wallets: &[String],
    trades: &[WhaleTrade],
) -> Vec<String> {
    let mut wallets = BTreeSet::new();
    for wallet in request_wallets {
        let wallet = wallet.trim();
        if !wallet.is_empty() {
            wallets.insert(wallet.to_ascii_lowercase());
        }
    }
    for trade in trades {
        let wallet = trade.proxy_wallet.trim();
        if !wallet.is_empty() {
            wallets.insert(wallet.to_ascii_lowercase());
        }
    }
    wallets.into_iter().collect()
}

pub async fn fetch_closed_positions_for_wallets(
    data_api: &DataApiClient,
    wallets: &[String],
    page_limit: usize,
    max_pages: usize,
) -> Result<Vec<(String, Vec<DataApiClosedPosition>)>> {
    let mut results = Vec::with_capacity(wallets.len());

    for wallet in wallets {
        let wallet_positions =
            fetch_closed_positions_for_wallet(data_api, wallet, page_limit, max_pages).await?;
        results.push((wallet.clone(), wallet_positions));
    }

    Ok(results)
}

pub async fn fetch_closed_positions_for_wallet(
    data_api: &DataApiClient,
    wallet: &str,
    page_limit: usize,
    max_pages: usize,
) -> Result<Vec<DataApiClosedPosition>> {
    let page_limit = page_limit.max(1);
    let max_pages = max_pages.max(1);
    let mut wallet_positions = Vec::new();
    for page in 0..max_pages {
        let mut query = ClosedPositionsQuery::for_user(wallet);
        query.limit = Some(page_limit);
        query.offset = Some(page * page_limit);
        let positions = match data_api.fetch_closed_positions(&query).await {
            Ok(positions) => positions,
            Err(error) => {
                warn!(
                    error = %error,
                    wallet = %wallet,
                    page,
                    "failed to fetch closed positions for wallet; continuing"
                );
                break;
            }
        };
        let fetched = positions.len();
        wallet_positions.extend(positions);
        if fetched < page_limit {
            break;
        }
    }
    Ok(wallet_positions)
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    use crate::models::{OrderRecord, OrderRequest, OrderSide, OrderState, OrderType};

    use super::*;

    fn trade(wallet: &str) -> WhaleTrade {
        WhaleTrade {
            trade_id: Uuid::nil(),
            proxy_wallet: wallet.to_string(),
            asset: "asset".to_string(),
            condition_id: Some("condition".to_string()),
            market_id: None,
            side: "BUY".to_string(),
            outcome: Some("Yes".to_string()),
            price: dec!(0.5),
            size: dec!(10),
            cash_value: dec!(5),
            timestamp_utc: Utc::now(),
            title: None,
            slug: None,
            event_slug: None,
            transaction_hash: None,
            raw_payload: serde_json::json!({}),
        }
    }

    #[test]
    fn distinct_wallets_are_deduped_lowercase_and_sorted() {
        let wallets = distinct_wallets_for_scoring(
            &["0xBBB".to_string(), " ".to_string(), "0xaaa".to_string()],
            &[trade("0xbbb"), trade("0xCCC")],
        );

        assert_eq!(wallets, vec!["0xaaa", "0xbbb", "0xccc"]);
    }

    #[test]
    fn entry_markability_requires_two_sided_book() {
        let base = OrderbookSnapshot {
            snapshot_id: Uuid::new_v4(),
            timestamp_utc: Utc::now(),
            market_id: Some("market".to_string()),
            token_id: "token".to_string(),
            best_bid: Some(dec!(0.49)),
            best_ask: Some(dec!(0.51)),
            tick_size: None,
            stale_level_count: 0,
            fresh_depth_bid: None,
            fresh_depth_ask: None,
            book: serde_json::json!({}),
        };

        assert_eq!(entry_mark_book_failure_reason(&base), None);

        let mut missing_bid = base.clone();
        missing_bid.best_bid = None;
        assert_eq!(
            entry_mark_book_failure_reason(&missing_bid),
            Some("missing_best_bid")
        );

        let mut missing_ask = base.clone();
        missing_ask.best_ask = None;
        assert_eq!(
            entry_mark_book_failure_reason(&missing_ask),
            Some("missing_best_ask")
        );

        let mut empty = base;
        empty.best_bid = None;
        empty.best_ask = None;
        assert_eq!(
            entry_mark_book_failure_reason(&empty),
            Some("empty_orderbook")
        );
    }

    #[test]
    fn entry_markability_classifies_clob_errors() {
        let unavailable = anyhow::anyhow!("CLOB book response was not successful");
        assert_eq!(
            entry_mark_error_reason(&unavailable),
            "clob_orderbook_unavailable"
        );

        let request_failed = anyhow::anyhow!("failed to request CLOB book for token abc");
        assert_eq!(
            entry_mark_error_reason(&request_failed),
            "clob_orderbook_request_failed"
        );
    }

    #[test]
    fn copy_signal_execution_status_does_not_mark_rejected_orders_filled() {
        let order = OrderRecord {
            order_id: "live-pending-order".to_string(),
            request: OrderRequest {
                client_order_id: Uuid::new_v4(),
                process_id: Some(Uuid::new_v4()),
                market_id: "market".to_string(),
                token_id: "token".to_string(),
                side: OrderSide::Buy,
                order_type: OrderType::Fok,
                price: dec!(0.50),
                size: dec!(4),
                signal_id: Some(Uuid::new_v4()),
                metadata: serde_json::json!({"purpose": "whale_follow_entry"}),
            },
            state: OrderState::Rejected,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let report = OrderPlanReport {
            plan_id: Uuid::new_v4(),
            orders: vec![order],
            fills: vec![],
            reconciliation: crate::execution::ReconciliationReport {
                open_orders: 0,
                balances_checked: true,
                mismatches_found: 0,
                unresolved_count: 0,
                checked_at: Utc::now(),
            },
        };

        let (status, metadata) = copy_signal_execution_status(&report);

        assert_eq!(status, "rejected");
        assert_eq!(metadata["orders"], 1);
        assert_eq!(metadata["fills"], 0);
    }

    fn safety_order(side: OrderSide, price: rust_decimal::Decimal) -> OrderRequest {
        OrderRequest {
            client_order_id: Uuid::new_v4(),
            process_id: Some(Uuid::new_v4()),
            market_id: "market".to_string(),
            token_id: "token".to_string(),
            side,
            order_type: OrderType::Fok,
            price,
            size: dec!(4),
            signal_id: Some(Uuid::new_v4()),
            metadata: serde_json::json!({"purpose": "whale_follow_entry"}),
        }
    }

    fn safety_config() -> CopyTradeConfig {
        let mut config = CopyTradeConfig::default();
        config.entry_safety.enabled = true;
        config.entry_safety.min_time_to_expiry_secs = 120;
        config.entry_safety.require_two_sided_book = true;
        config.entry_safety.max_spread_bps = dec!(2500);
        config.entry_safety.require_exit_depth = true;
        config.entry_safety.exit_depth_size_fraction = dec!(1.0);
        config.entry_safety.exit_depth_slippage_bps = dec!(150);
        config.entry_safety.min_entry_price = dec!(0.05);
        config.entry_safety.max_entry_price = dec!(0.95);
        config
    }

    fn safety_book(
        bid: rust_decimal::Decimal,
        bid_size: rust_decimal::Decimal,
        ask: rust_decimal::Decimal,
    ) -> LocalOrderBook {
        let now = Utc::now();
        let mut book = LocalOrderBook::default();
        book.upsert_level(BookSide::Bid, bid, bid_size, now);
        book.upsert_level(BookSide::Ask, ask, dec!(20), now);
        book
    }

    #[test]
    fn entry_safety_accepts_liquid_two_sided_book() {
        let now = Utc::now();
        let request = safety_order(OrderSide::Buy, dec!(0.51));
        let book = safety_book(dec!(0.505), dec!(10), dec!(0.51));

        let rejection = entry_safety_rejection(
            &request,
            &book,
            Some(now + Duration::seconds(300)),
            &safety_config(),
            now,
        );

        assert!(rejection.is_none());
    }

    #[test]
    fn entry_safety_rejects_missing_expiry_when_required() {
        let now = Utc::now();
        let request = safety_order(OrderSide::Buy, dec!(0.51));
        let book = safety_book(dec!(0.50), dec!(10), dec!(0.51));

        let rejection =
            entry_safety_rejection(&request, &book, None, &safety_config(), now).unwrap();

        assert_eq!(rejection["reject_reason"], "missing_market_expiry");
    }

    #[test]
    fn entry_safety_rejects_expiry_too_close() {
        let now = Utc::now();
        let request = safety_order(OrderSide::Buy, dec!(0.51));
        let book = safety_book(dec!(0.50), dec!(10), dec!(0.51));

        let rejection = entry_safety_rejection(
            &request,
            &book,
            Some(now + Duration::seconds(30)),
            &safety_config(),
            now,
        )
        .unwrap();

        assert_eq!(rejection["reject_reason"], "entry_expiry_too_close");
    }

    #[test]
    fn entry_safety_rejects_one_sided_book() {
        let now = Utc::now();
        let request = safety_order(OrderSide::Buy, dec!(0.51));
        let mut book = LocalOrderBook::default();
        book.upsert_level(BookSide::Ask, dec!(0.51), dec!(20), now);

        let rejection = entry_safety_rejection(
            &request,
            &book,
            Some(now + Duration::seconds(300)),
            &safety_config(),
            now,
        )
        .unwrap();

        assert_eq!(rejection["reject_reason"], "entry_book_not_two_sided");
    }

    #[test]
    fn entry_safety_rejects_wide_spread() {
        let now = Utc::now();
        let request = safety_order(OrderSide::Buy, dec!(0.70));
        let book = safety_book(dec!(0.30), dec!(10), dec!(0.70));

        let rejection = entry_safety_rejection(
            &request,
            &book,
            Some(now + Duration::seconds(300)),
            &safety_config(),
            now,
        )
        .unwrap();

        assert_eq!(rejection["reject_reason"], "entry_spread_too_wide");
    }

    #[test]
    fn entry_safety_rejects_insufficient_exit_depth() {
        let now = Utc::now();
        let request = safety_order(OrderSide::Buy, dec!(0.51));
        let book = safety_book(dec!(0.50), dec!(1), dec!(0.51));

        let rejection = entry_safety_rejection(
            &request,
            &book,
            Some(now + Duration::seconds(300)),
            &safety_config(),
            now,
        )
        .unwrap();

        assert_eq!(rejection["reject_reason"], "entry_exit_depth_insufficient");
    }

    #[test]
    fn entry_safety_rejects_price_out_of_bounds() {
        let now = Utc::now();
        let request = safety_order(OrderSide::Buy, dec!(0.04));
        let book = safety_book(dec!(0.039), dec!(10), dec!(0.04));

        let rejection = entry_safety_rejection(
            &request,
            &book,
            Some(now + Duration::seconds(300)),
            &safety_config(),
            now,
        )
        .unwrap();

        assert_eq!(rejection["reject_reason"], "entry_price_out_of_bounds");
    }
}
pub async fn run_copy_trade_signal_engine(
    store: &Store,
    venue: Option<&dyn ExecutionVenue>,
    clob: Option<&ClobClient>,
    trades: &[WhaleTrade],
    config: &CopyTradeRunConfig,
    dry_run: bool,
) -> Result<CopyTradeRunSummary> {
    let mut summary = CopyTradeRunSummary {
        dry_run,
        ..CopyTradeRunSummary::default()
    };
    let fallback_wallet_performances: Vec<CopyTradeWalletPerformance> = score_wallets(trades)
        .iter()
        .map(CopyTradeWalletPerformance::from)
        .collect();
    let fallback_performance_by_wallet = fallback_wallet_performances
        .iter()
        .map(|performance| (performance.proxy_wallet.as_str(), performance))
        .collect::<std::collections::HashMap<_, _>>();
    let mut persisted_performance_by_wallet =
        std::collections::HashMap::<String, Option<CopyTradeWalletPerformance>>::new();
    let mut persisted_mrs_by_wallet =
        std::collections::HashMap::<String, Option<CopyTradeMrsScore>>::new();
    let mut persisted_segment_by_wallet_segment =
        std::collections::HashMap::<(String, String), Option<CopyTradeSegmentScore>>::new();
    let copy_config = config.copy_trade.clone();
    let mut open_notional_with_in_run_orders = if !dry_run && config.execute_signals {
        match config.process_id {
            Some(process_id) => Some(store.process_open_notional(process_id).await?),
            None => None,
        }
    } else {
        None
    };

    for trade in trades {
        summary.trades_evaluated += 1;
        let observed_at = if dry_run {
            trade.timestamp_utc
        } else {
            Utc::now()
        };
        let performance = if dry_run {
            fallback_performance_by_wallet
                .get(trade.proxy_wallet.as_str())
                .copied()
        } else {
            if !persisted_performance_by_wallet.contains_key(&trade.proxy_wallet) {
                let persisted = store
                    .fetch_latest_wallet_performance(&trade.proxy_wallet)
                    .await?
                    .as_ref()
                    .map(CopyTradeWalletPerformance::from);
                persisted_performance_by_wallet.insert(trade.proxy_wallet.clone(), persisted);
            }
            persisted_performance_by_wallet
                .get(&trade.proxy_wallet)
                .and_then(|performance| performance.as_ref())
        };
        if !dry_run
            && copy_config.mrs_enabled
            && !persisted_mrs_by_wallet.contains_key(&trade.proxy_wallet)
        {
            let persisted = store
                .fetch_wallet_score_by_version(&trade.proxy_wallet, &copy_config.mrs_score_version)
                .await?
                .map(|score| CopyTradeMrsScore {
                    score: score.score,
                    score_version: score.score_version,
                    percentile: None,
                    metadata: score.metadata,
                });
            persisted_mrs_by_wallet.insert(trade.proxy_wallet.clone(), persisted);
        }
        let mrs_score = if dry_run || !copy_config.mrs_enabled {
            None
        } else {
            persisted_mrs_by_wallet
                .get(&trade.proxy_wallet)
                .and_then(|score| score.as_ref())
        };
        let segment_classification =
            if copy_config.segment_classifier_version == GAMMA_SEGMENT_CLASSIFIER_VERSION {
                store
                    .fetch_wallet_trade_gamma_segment_classification(trade.trade_id)
                    .await?
                    .unwrap_or_else(|| classify_trade_segment(trade))
            } else {
                classify_trade_segment(trade)
            };
        if !dry_run
            && copy_config.segment_scoring_enabled
            && !persisted_segment_by_wallet_segment.contains_key(&(
                trade.proxy_wallet.clone(),
                segment_classification.segment_key.clone(),
            ))
        {
            let persisted = store
                .fetch_wallet_segment_performance(
                    &trade.proxy_wallet,
                    &segment_classification.segment_key,
                    &copy_config.segment_score_version,
                )
                .await?
                .as_ref()
                .map(CopyTradeSegmentScore::from);
            persisted_segment_by_wallet_segment.insert(
                (
                    trade.proxy_wallet.clone(),
                    segment_classification.segment_key.clone(),
                ),
                persisted,
            );
        }
        let segment_score = if dry_run || !copy_config.segment_scoring_enabled {
            None
        } else {
            persisted_segment_by_wallet_segment
                .get(&(
                    trade.proxy_wallet.clone(),
                    segment_classification.segment_key.clone(),
                ))
                .and_then(|score| score.as_ref())
        };
        let trade_summary = run_resolved_copy_trade_signal(
            store,
            venue,
            clob,
            ResolvedCopyTradeSignal {
                trade,
                performance,
                mrs_score,
                segment_score,
                segment_classification,
                observed: ObservedMarket {
                    observed_price: trade.price,
                    available_depth_usd: trade.cash_value,
                    observed_at,
                },
                order_metadata: serde_json::Value::Null,
            },
            config,
            dry_run,
            open_notional_with_in_run_orders.as_mut(),
        )
        .await?;
        summary.signals_inserted += trade_summary.signals_inserted;
        summary.signals_detected += trade_summary.signals_detected;
        summary.orders_inserted += trade_summary.orders_inserted;
        summary.fills_inserted += trade_summary.fills_inserted;
        summary.rejections += trade_summary.rejections;
    }

    if !dry_run && summary.trades_evaluated > 0 {
        refresh_trade_pnl_with_config(store, venue, clob, &TradePnlConfig::default()).await?;
    }

    Ok(summary)
}

pub async fn run_resolved_copy_trade_signal(
    store: &Store,
    venue: Option<&dyn ExecutionVenue>,
    clob: Option<&ClobClient>,
    resolved: ResolvedCopyTradeSignal<'_>,
    config: &CopyTradeRunConfig,
    dry_run: bool,
    mut open_notional_with_in_run_orders: Option<&mut Decimal>,
) -> Result<CopyTradeRunSummary> {
    let mut summary = CopyTradeRunSummary {
        trades_evaluated: 1,
        dry_run,
        ..CopyTradeRunSummary::default()
    };
    let decision = evaluate_copy_trade_with_segment(
        resolved.trade,
        resolved.performance,
        resolved.mrs_score,
        resolved.segment_score,
        Some(resolved.segment_classification),
        resolved.observed,
        &config.copy_trade,
        config.process_id,
    );

    if dry_run {
        if decision.order_plan.is_some() {
            summary.signals_inserted += 1;
            summary.signals_detected += 1;
        } else {
            summary.rejections += 1;
        }
        return Ok(summary);
    }

    store.insert_signal(&decision.signal_candidate).await?;
    store
        .insert_copy_trade_signal(&decision.copy_signal)
        .await?;
    summary.signals_inserted += 1;
    if decision.signal_candidate.status == crate::models::SignalStatus::Detected {
        summary.signals_detected += 1;
    }
    if decision.order_plan.is_none() {
        summary.rejections += 1;
        return Ok(summary);
    }

    if !config.execute_signals {
        return Ok(summary);
    }
    let Some(venue) = venue else {
        return Ok(summary);
    };
    let Some(mut plan) = decision.order_plan else {
        return Ok(summary);
    };
    if resolved.order_metadata.is_object() {
        for order in &mut plan.orders {
            order.metadata = merge_order_metadata(order.metadata.clone(), &resolved.order_metadata);
        }
    }
    if let Some(current_open_notional) = open_notional_with_in_run_orders.as_deref_mut() {
        let order_notional: Decimal = plan
            .orders
            .iter()
            .map(|order| order.price * order.size)
            .sum();
        let max_open_notional = config.copy_trade.max_open_notional_usd;
        if max_open_notional > Decimal::ZERO
            && *current_open_notional + order_notional > max_open_notional
        {
            summary.rejections += 1;
            store
                .update_copy_trade_signal_status(
                    decision.copy_signal.signal_id,
                    decision.copy_signal.timestamp_utc,
                    "rejected",
                    serde_json::json!({
                        "reason": "copy_open_notional_cap_exceeded",
                        "current_open_notional_usd": *current_open_notional,
                        "order_notional_usd": order_notional,
                        "max_open_notional_usd": max_open_notional
                    }),
                )
                .await?;
            return Ok(summary);
        }
    }
    if config.require_entry_markability {
        if let Some(rejection) =
            ensure_order_plan_markable_at_entry(store, clob, &plan, &config.copy_trade).await?
        {
            summary.rejections += 1;
            store
                .update_copy_trade_signal_status(
                    decision.copy_signal.signal_id,
                    decision.copy_signal.timestamp_utc,
                    "rejected",
                    rejection,
                )
                .await?;
            return Ok(summary);
        }
    }
    let execution = execute_order_plan(venue, plan).await?;
    summary.orders_inserted += execution.orders.len();
    summary.fills_inserted += execution.fills.len();
    store.persist_order_plan_report(&execution).await?;
    let (status, metadata) = copy_signal_execution_status(&execution);
    if status == "rejected" {
        summary.rejections += 1;
    } else if let Some(current_open_notional) = open_notional_with_in_run_orders.as_deref_mut() {
        let filled_notional: Decimal = execution
            .fills
            .iter()
            .map(|fill| fill.price * fill.size)
            .sum();
        let order_notional: Decimal = execution
            .orders
            .iter()
            .map(|order| order.request.price * order.request.size)
            .sum();
        *current_open_notional += if filled_notional > Decimal::ZERO {
            filled_notional
        } else {
            order_notional
        };
    }
    store
        .update_copy_trade_signal_status(
            decision.copy_signal.signal_id,
            decision.copy_signal.timestamp_utc,
            status,
            metadata,
        )
        .await?;

    Ok(summary)
}

fn merge_order_metadata(
    mut left: serde_json::Value,
    right: &serde_json::Value,
) -> serde_json::Value {
    if let (Some(left), Some(right)) = (left.as_object_mut(), right.as_object()) {
        for (key, value) in right {
            left.insert(key.clone(), value.clone());
        }
    }
    left
}

fn copy_signal_execution_status(execution: &OrderPlanReport) -> (&'static str, serde_json::Value) {
    let has_fill = !execution.fills.is_empty()
        || execution.orders.iter().any(|order| {
            matches!(
                order.state,
                OrderState::Filled | OrderState::PartiallyFilled
            )
        });
    let all_terminal_rejected = !execution.orders.is_empty()
        && execution.orders.iter().all(|order| {
            matches!(
                order.state,
                OrderState::Rejected | OrderState::Cancelled | OrderState::Expired
            )
        });
    let status = if has_fill {
        "filled"
    } else if all_terminal_rejected {
        "rejected"
    } else {
        "submitted"
    };
    let order_states = execution
        .orders
        .iter()
        .map(|order| {
            serde_json::json!({
                "order_id": order.order_id,
                "state": order.state,
            })
        })
        .collect::<Vec<_>>();
    (
        status,
        serde_json::json!({
            "orders": execution.orders.len(),
            "fills": execution.fills.len(),
            "order_states": order_states,
        }),
    )
}

async fn ensure_order_plan_markable_at_entry(
    store: &Store,
    clob: Option<&ClobClient>,
    plan: &OrderPlan,
    config: &CopyTradeConfig,
) -> Result<Option<serde_json::Value>> {
    let Some(clob) = clob else {
        for request in &plan.orders {
            record_entry_mark_failure(
                store,
                request.process_id,
                &request.market_id,
                &request.token_id,
                "missing_clob_client",
                serde_json::json!({
                    "operation": "copy_trade_entry_markability",
                    "client_order_id": request.client_order_id,
                    "plan_id": plan.plan_id
                }),
            )
            .await?;
        }
        return Ok(Some(serde_json::json!({
            "status": "rejected",
            "reject_reason": "unmarkable_at_entry",
            "mark_failure_reason": "missing_clob_client",
            "plan_id": plan.plan_id
        })));
    };

    for request in &plan.orders {
        let fetched = clob.fetch_orderbook(&request.token_id).await;
        let book = match fetched {
            Ok(book) => book,
            Err(error) => {
                let reason = entry_mark_error_reason(&error);
                record_entry_mark_failure(
                    store,
                    request.process_id,
                    &request.market_id,
                    &request.token_id,
                    reason,
                    serde_json::json!({
                        "operation": "copy_trade_entry_markability",
                        "client_order_id": request.client_order_id,
                        "plan_id": plan.plan_id,
                        "error": error.to_string(),
                        "error_chain": format!("{error:#}")
                    }),
                )
                .await?;
                return Ok(Some(serde_json::json!({
                    "status": "rejected",
                    "reject_reason": "unmarkable_at_entry",
                    "mark_failure_reason": reason,
                    "client_order_id": request.client_order_id,
                    "token_id": request.token_id
                })));
            }
        };

        let snapshot = OrderbookSnapshot::from_local_book(
            Some(request.market_id.clone()),
            request.token_id.clone(),
            None,
            &book,
        )?;
        store.insert_orderbook_snapshot(&snapshot).await?;
        if let Some(reason) = entry_mark_book_failure_reason(&snapshot) {
            record_entry_mark_failure(
                store,
                request.process_id,
                &request.market_id,
                &request.token_id,
                reason,
                serde_json::json!({
                    "operation": "copy_trade_entry_markability",
                    "client_order_id": request.client_order_id,
                    "plan_id": plan.plan_id,
                    "best_bid": snapshot.best_bid,
                    "best_ask": snapshot.best_ask,
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_timestamp": snapshot.timestamp_utc
                }),
            )
            .await?;
            return Ok(Some(serde_json::json!({
                "status": "rejected",
                "reject_reason": "unmarkable_at_entry",
                "mark_failure_reason": reason,
                "client_order_id": request.client_order_id,
                "token_id": request.token_id,
                "best_bid": snapshot.best_bid,
                "best_ask": snapshot.best_ask
            })));
        }
        let market_end_date =
            if config.entry_safety.enabled && config.entry_safety.min_time_to_expiry_secs > 0 {
                store
                    .fetch_market_end_date_for_entry(&request.market_id, &request.token_id)
                    .await?
            } else {
                None
            };
        if let Some(rejection) =
            entry_safety_rejection(request, &book, market_end_date, config, Utc::now())
        {
            record_entry_mark_failure(
                store,
                request.process_id,
                &request.market_id,
                &request.token_id,
                rejection
                    .get("reject_reason")
                    .and_then(|value| value.as_str())
                    .unwrap_or("entry_safety_rejected"),
                serde_json::json!({
                    "operation": "copy_trade_entry_safety",
                    "client_order_id": request.client_order_id,
                    "plan_id": plan.plan_id,
                    "entry_safety": rejection
                }),
            )
            .await?;
            return Ok(Some(rejection));
        }
        store
            .resolve_trade_mark_source_failure(
                request.process_id,
                None,
                &request.token_id,
                "entry_orderbook_precheck",
            )
            .await?;
    }

    Ok(None)
}

fn entry_safety_rejection(
    request: &OrderRequest,
    book: &LocalOrderBook,
    market_end_date: Option<DateTime<Utc>>,
    config: &CopyTradeConfig,
    now: DateTime<Utc>,
) -> Option<serde_json::Value> {
    let safety = &config.entry_safety;
    if !safety.enabled {
        return None;
    }

    let mut metadata = serde_json::json!({
        "status": "rejected",
        "entry_safety": {
            "enabled": true,
            "decision": "accepted",
            "best_bid": book.best_bid(),
            "best_ask": book.best_ask(),
            "min_time_to_expiry_secs": safety.min_time_to_expiry_secs,
            "require_two_sided_book": safety.require_two_sided_book,
            "max_spread_bps": safety.max_spread_bps,
            "require_exit_depth": safety.require_exit_depth,
            "exit_depth_size_fraction": safety.exit_depth_size_fraction,
            "exit_depth_slippage_bps": safety.exit_depth_slippage_bps,
            "min_entry_price": safety.min_entry_price,
            "max_entry_price": safety.max_entry_price
        }
    });

    if safety.min_time_to_expiry_secs > 0 {
        let Some(end_date) = market_end_date else {
            return Some(entry_safety_reject(
                metadata,
                "missing_market_expiry",
                serde_json::json!({}),
            ));
        };
        let time_to_expiry_secs = (end_date - now).num_seconds();
        merge_object(
            &mut metadata,
            serde_json::json!({
                "entry_safety": {
                    "market_end_date": end_date,
                    "time_to_expiry_secs": time_to_expiry_secs
                }
            }),
        );
        if time_to_expiry_secs < safety.min_time_to_expiry_secs {
            return Some(entry_safety_reject(
                metadata,
                "entry_expiry_too_close",
                serde_json::json!({}),
            ));
        }
    }

    let best_bid = book.best_bid();
    let best_ask = book.best_ask();
    if safety.require_two_sided_book && (best_bid.is_none() || best_ask.is_none()) {
        return Some(entry_safety_reject(
            metadata,
            "entry_book_not_two_sided",
            serde_json::json!({}),
        ));
    }

    let entry_price = match request.side {
        crate::models::OrderSide::Buy => best_ask.unwrap_or(request.price),
        crate::models::OrderSide::Sell => best_bid.unwrap_or(request.price),
    };
    merge_object(
        &mut metadata,
        serde_json::json!({
            "entry_safety": {
                "entry_price": entry_price
            }
        }),
    );
    if entry_price < safety.min_entry_price || entry_price > safety.max_entry_price {
        return Some(entry_safety_reject(
            metadata,
            "entry_price_out_of_bounds",
            serde_json::json!({}),
        ));
    }

    if let (Some(best_bid), Some(best_ask)) = (best_bid, best_ask) {
        let mid = (best_bid + best_ask) / Decimal::from(2);
        let spread_bps = if mid > Decimal::ZERO {
            ((best_ask - best_bid).abs() / mid) * Decimal::from(10000)
        } else {
            Decimal::MAX
        };
        merge_object(
            &mut metadata,
            serde_json::json!({
                "entry_safety": {
                    "spread_bps": spread_bps
                }
            }),
        );
        if safety.max_spread_bps > Decimal::ZERO && spread_bps > safety.max_spread_bps {
            return Some(entry_safety_reject(
                metadata,
                "entry_spread_too_wide",
                serde_json::json!({}),
            ));
        }
    }

    if safety.require_exit_depth {
        let required_size = request.size
            * safety
                .exit_depth_size_fraction
                .clamp(Decimal::ZERO, dec!(1.0));
        let slippage = safety.exit_depth_slippage_bps.max(Decimal::ZERO) / Decimal::from(10000);
        let (exit_side, exit_limit_price) = match request.side {
            crate::models::OrderSide::Buy => {
                (BookSide::Bid, entry_price * (Decimal::ONE - slippage))
            }
            crate::models::OrderSide::Sell => {
                (BookSide::Ask, entry_price * (Decimal::ONE + slippage))
            }
        };
        let depth =
            book.limit_depth_summary(exit_side, exit_limit_price, Duration::seconds(10), now);
        merge_object(
            &mut metadata,
            serde_json::json!({
                "entry_safety": {
                    "exit_side": exit_side,
                    "exit_limit_price": exit_limit_price,
                    "exit_depth_required": required_size,
                    "exit_depth_available": depth.fillable_size,
                    "exit_depth_avg_price": depth.avg_price
                }
            }),
        );
        if depth.fillable_size < required_size {
            return Some(entry_safety_reject(
                metadata,
                "entry_exit_depth_insufficient",
                serde_json::json!({}),
            ));
        }
    }

    None
}

fn entry_safety_reject(
    mut metadata: serde_json::Value,
    reason: &'static str,
    extra: serde_json::Value,
) -> serde_json::Value {
    merge_object(
        &mut metadata,
        serde_json::json!({
            "reject_reason": reason,
            "entry_safety": {
                "decision": "rejected",
                "reject_reason": reason
            }
        }),
    );
    merge_object(&mut metadata, extra);
    metadata
}

fn merge_object(left: &mut serde_json::Value, right: serde_json::Value) {
    match (left, right) {
        (serde_json::Value::Object(left), serde_json::Value::Object(right)) => {
            for (key, value) in right {
                if let Some(existing) = left.get_mut(&key) {
                    merge_object(existing, value);
                } else {
                    left.insert(key, value);
                }
            }
        }
        (left, right) => *left = right,
    }
}

async fn record_entry_mark_failure(
    store: &Store,
    process_id: Option<Uuid>,
    market_id: &str,
    token_id: &str,
    reason: &str,
    mut metadata: serde_json::Value,
) -> Result<()> {
    if let Some(object) = metadata.as_object_mut() {
        object.insert("market_id".to_string(), serde_json::json!(market_id));
        object.insert("token_id".to_string(), serde_json::json!(token_id));
        object.insert("attempted_at".to_string(), serde_json::json!(Utc::now()));
    }
    store
        .upsert_trade_mark_source_failure(&TradeMarkSourceFailure {
            process_id,
            position_id: None,
            token_id: token_id.to_string(),
            market_id: Some(market_id.to_string()),
            failure_source: "entry_orderbook_precheck".to_string(),
            failure_reason: reason.to_string(),
            metadata,
        })
        .await
}

fn entry_mark_book_failure_reason(snapshot: &OrderbookSnapshot) -> Option<&'static str> {
    match (snapshot.best_bid, snapshot.best_ask) {
        (Some(_), Some(_)) => None,
        (None, Some(_)) => Some("missing_best_bid"),
        (Some(_), None) => Some("missing_best_ask"),
        (None, None) => Some("empty_orderbook"),
    }
}

fn entry_mark_error_reason(error: &anyhow::Error) -> &'static str {
    let chain = format!("{error:#}");
    if chain.contains("CLOB book response was not successful") {
        "clob_orderbook_unavailable"
    } else if chain.contains("failed to request CLOB book") {
        "clob_orderbook_request_failed"
    } else if chain.contains("failed to decode CLOB book") {
        "clob_orderbook_decode_failed"
    } else {
        "entry_orderbook_precheck_failed"
    }
}

pub fn calibrate_copy_trade_thresholds(
    trades: &[WhaleTrade],
    config: &CopyTradeRunConfig,
) -> serde_json::Value {
    let scores = score_wallets(trades);
    let (backtest, calibration) = run_copy_trade_backtest(trades, &scores, &config.copy_trade);
    serde_json::json!({
        "score_version": crate::wallets::SCORE_VERSION,
        "wallets_scored": scores.len(),
        "backtest": backtest,
        "calibration": calibration
    })
}

fn merge_json(mut left: serde_json::Value, right: serde_json::Value) -> serde_json::Value {
    if let (Some(left), Some(right)) = (left.as_object_mut(), right.as_object()) {
        for (key, value) in right {
            left.insert(key.clone(), value.clone());
        }
    }
    left
}

fn default_lookback_days() -> u32 {
    30
}

fn default_min_trade_usd() -> Decimal {
    Decimal::from(1000)
}

fn default_limit() -> usize {
    1000
}

fn default_max_pages() -> usize {
    10
}

fn default_require_entry_markability() -> bool {
    true
}

fn default_copy_min_wallet_score() -> Decimal {
    Decimal::from(70)
}

fn default_copy_size_fraction() -> Decimal {
    Decimal::new(10, 2)
}

fn default_copy_max_size_usd() -> Decimal {
    Decimal::from(500)
}
