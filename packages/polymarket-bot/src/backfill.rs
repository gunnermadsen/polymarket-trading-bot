use std::{collections::BTreeSet, sync::Arc};

use anyhow::{Context, Result};
use chrono::{Duration, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use tracing::warn;
use uuid::Uuid;

use crate::{
    copytrade::{
        evaluate_copy_trade, run_copy_trade_backtest, CopyTradeConfig, CopyTradeWalletPerformance,
        ObservedMarket, COPY_SCORE_VERSION,
    },
    data_api::{ClosedPositionsQuery, DataApiClient, TradesQuery},
    execution::{execute_order_plan, ExecutionVenue},
    models::{BackfillJobStatus, CopyTradeBacktestRun, DataApiClosedPosition, WhaleTrade},
    store::Store,
    trade_pnl::refresh_trade_pnl,
    wallets::{score_closed_position_performance, score_closed_position_wallets, score_wallets},
};

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
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CopyTradeRunSummary {
    pub trades_evaluated: usize,
    pub signals_inserted: usize,
    pub orders_inserted: usize,
    pub fills_inserted: usize,
    pub rejections: usize,
    pub dry_run: bool,
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
                let performance = score_closed_position_performance(wallet, &positions)
                    .into_wallet_performance(Utc::now(), serde_json::to_value(&positions)?);
                store.upsert_wallet_performance(&performance).await?;
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
            &trades_for_copy,
            &CopyTradeRunConfig {
                process_id: request.process_id,
                copy_trade: copy_trade_config.clone(),
                execute_signals: request.execute_signals,
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
}
pub async fn run_copy_trade_signal_engine(
    store: &Store,
    venue: Option<&dyn ExecutionVenue>,
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
    let copy_config = config.copy_trade.clone();

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
        let decision = evaluate_copy_trade(
            trade,
            performance,
            ObservedMarket {
                observed_price: trade.price,
                available_depth_usd: trade.cash_value,
                observed_at,
            },
            &copy_config,
            config.process_id,
        );

        if dry_run {
            if decision.order_plan.is_some() {
                summary.signals_inserted += 1;
            } else {
                summary.rejections += 1;
            }
            continue;
        }

        store.insert_signal(&decision.signal_candidate).await?;
        store
            .insert_copy_trade_signal(&decision.copy_signal)
            .await?;
        summary.signals_inserted += 1;
        if decision.order_plan.is_none() {
            summary.rejections += 1;
            continue;
        }

        if !config.execute_signals {
            continue;
        }
        let Some(venue) = venue else {
            continue;
        };
        let Some(plan) = decision.order_plan else {
            continue;
        };
        let execution = execute_order_plan(venue, plan).await?;
        summary.orders_inserted += execution.orders.len();
        summary.fills_inserted += execution.fills.len();
        store.persist_order_plan_report(&execution).await?;
        store
            .update_copy_trade_signal_status(
                decision.copy_signal.signal_id,
                decision.copy_signal.timestamp_utc,
                "filled",
                serde_json::json!({
                    "orders": execution.orders.len(),
                    "fills": execution.fills.len()
                }),
            )
            .await?;
    }

    if !dry_run && summary.trades_evaluated > 0 {
        refresh_trade_pnl(store, venue).await?;
    }

    Ok(summary)
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

fn default_copy_min_wallet_score() -> Decimal {
    Decimal::from(70)
}

fn default_copy_size_fraction() -> Decimal {
    Decimal::new(10, 2)
}

fn default_copy_max_size_usd() -> Decimal {
    Decimal::from(500)
}
