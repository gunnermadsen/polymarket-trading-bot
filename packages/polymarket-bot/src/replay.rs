use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    account_reconcile::{AccountReconcileReport, AccountReconcileRequest},
    copytrade::{
        evaluate_copy_trade_with_segment, CopyTradeConfig, CopyTradeMrsScore,
        CopyTradeSegmentScore, CopyTradeWalletPerformance, ObservedMarket,
    },
    execution::{
        execute_order_plan, ExecutionVenue, LiveIdentityDiagnostics, LiveOrderDryRunDiagnostics,
        LiveOrderDryRunRequest, LivePoly1271FunderProbeRequest, LivePoly1271FunderProbeResponse,
        LiveVenueStatus, LiveWalletAddressDiagnostics, ReconciliationReport,
    },
    models::{
        ConversionRequest, ConversionResult, FillRecord, FillSource, OrderRecord, OrderRequest,
        OrderState, SignalStatus, TradingProcessConfig, WhaleTrade,
    },
    segments::{
        classify_trade_segment, score_wallet_segment, SegmentClassification,
        WalletSegmentPerformanceInput, GAMMA_SEGMENT_CLASSIFIER_VERSION,
    },
    store::Store,
    wallets::{score_mrs, MrsScoreInput},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReplayRequest {
    pub process_ids: Vec<Uuid>,
    pub lookback_days: Option<i32>,
    pub warmup_days: Option<i32>,
    pub min_trade_usd: Option<Decimal>,
    pub max_trades: Option<i64>,
    #[serde(default = "default_execute_signals")]
    pub execute_signals: bool,
    #[serde(default)]
    pub config_overrides: serde_json::Value,
}

fn default_execute_signals() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReplayQueued {
    pub backtest_run_id: Uuid,
    pub status: String,
    pub backtest_process_ids: Vec<Uuid>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BacktestReplaySummary {
    pub trades_loaded: usize,
    pub warmup_trades: usize,
    pub replay_trades: usize,
    pub signals_inserted: usize,
    pub signals_detected: usize,
    pub rejections: usize,
    pub orders_inserted: usize,
    pub fills_inserted: usize,
    pub positions_created: u64,
    pub whale_exits_applied: u64,
    pub marks_inserted: u64,
    pub process_count: usize,
}

#[derive(Debug, Clone)]
pub struct BacktestReplayJob {
    pub backtest_run_id: Uuid,
    pub range_start: DateTime<Utc>,
    pub range_end: DateTime<Utc>,
    pub warmup_start: DateTime<Utc>,
    pub min_trade_usd: Decimal,
    pub max_trades: i64,
    pub processes: Vec<BacktestReplayProcess>,
}

#[derive(Debug, Clone)]
pub struct BacktestReplayProcess {
    pub source_process_id: Uuid,
    pub backtest_process_id: Uuid,
    pub config: CopyTradeConfig,
    pub process_config: TradingProcessConfig,
}

pub async fn run_backtest_replay(
    store: Store,
    job: BacktestReplayJob,
) -> Result<BacktestReplaySummary> {
    store
        .mark_backtest_run_status(
            job.backtest_run_id,
            "running",
            serde_json::json!({"started": true}),
            None,
        )
        .await?;
    for process in &job.processes {
        store
            .update_trading_process_status(process.backtest_process_id, "running", false, None)
            .await?;
    }

    let trades = store
        .fetch_whale_trades_for_replay(
            job.warmup_start,
            job.range_end,
            job.min_trade_usd,
            job.max_trades,
        )
        .await?;
    let mut summary = BacktestReplaySummary {
        trades_loaded: trades.len(),
        process_count: job.processes.len(),
        ..BacktestReplaySummary::default()
    };
    let mut ledger = PointInTimeScoreLedger::default();

    for trade in trades {
        let classification = classify_for_replay(&store, &trade, &job.processes).await?;
        if trade.timestamp_utc < job.range_start {
            summary.warmup_trades += 1;
            ledger.observe_trade(&trade, &classification);
            continue;
        }
        summary.replay_trades += 1;
        for process in &job.processes {
            let mut config = process.config.clone();
            config.mrs_enforce = config.mrs_enforce || config.segment_scoring_enabled;
            let performance = ledger.wallet_performance(&trade.proxy_wallet);
            let mrs_score = ledger.mrs_score(&trade.proxy_wallet, &config.mrs_score_version);
            let segment_score = ledger.segment_score(
                &trade.proxy_wallet,
                &classification.segment_key,
                &config.segment_score_version,
            );
            let decision = evaluate_copy_trade_with_segment(
                &trade,
                performance.as_ref(),
                mrs_score.as_ref(),
                segment_score.as_ref(),
                Some(classification.clone()),
                ObservedMarket {
                    observed_price: trade.price,
                    available_depth_usd: trade.cash_value,
                    observed_at: trade.timestamp_utc,
                },
                &config,
                Some(process.backtest_process_id),
            );

            store.insert_signal(&decision.signal_candidate).await?;
            store
                .insert_copy_trade_signal(&decision.copy_signal)
                .await?;
            summary.signals_inserted += 1;
            if decision.signal_candidate.status == SignalStatus::Detected {
                summary.signals_detected += 1;
            } else {
                summary.rejections += 1;
            }

            let Some(mut plan) = decision.order_plan else {
                continue;
            };
            for order in &mut plan.orders {
                order.metadata = merge_json(
                    order.metadata.clone(),
                    serde_json::json!({
                        "backtest": true,
                        "backtest_run_id": job.backtest_run_id,
                        "source_process_id": process.source_process_id,
                        "backtest_process_id": process.backtest_process_id,
                        "backtest_fill_timestamp": trade.timestamp_utc
                    }),
                );
            }
            let venue = HistoricalReplayVenue::default();
            let execution = execute_order_plan(&venue, plan).await?;
            summary.orders_inserted += execution.orders.len();
            summary.fills_inserted += execution.fills.len();
            store.persist_order_plan_report(&execution).await?;
            let (status, metadata) = copy_signal_execution_status(&execution);
            if status == "rejected" {
                summary.rejections += 1;
            }
            store
                .update_copy_trade_signal_status(
                    decision.copy_signal.signal_id,
                    decision.copy_signal.timestamp_utc,
                    status,
                    metadata,
                )
                .await?;
            summary.positions_created += store.backfill_trade_positions_from_copy_signals().await?;
            summary.whale_exits_applied += store
                .apply_backtest_whale_led_trade_exits_for_process_until(
                    process.backtest_process_id,
                    trade.timestamp_utc,
                )
                .await?;
        }
        ledger.observe_trade(&trade, &classification);
    }

    for process in &job.processes {
        summary.marks_inserted += store
            .mark_open_trade_positions_for_process_as_of(process.backtest_process_id, job.range_end)
            .await?;
        store
            .update_trading_process_status(process.backtest_process_id, "completed", false, None)
            .await?;
    }
    store
        .mark_backtest_run_status(
            job.backtest_run_id,
            "completed",
            serde_json::to_value(&summary)?,
            None,
        )
        .await?;
    Ok(summary)
}

async fn classify_for_replay(
    store: &Store,
    trade: &WhaleTrade,
    processes: &[BacktestReplayProcess],
) -> Result<SegmentClassification> {
    if processes.iter().any(|process| {
        process.config.segment_classifier_version == GAMMA_SEGMENT_CLASSIFIER_VERSION
    }) {
        if let Some(classification) = store
            .fetch_wallet_trade_gamma_segment_classification(trade.trade_id)
            .await?
        {
            return Ok(classification);
        }
    }
    Ok(classify_trade_segment(trade))
}

fn copy_signal_execution_status(
    execution: &crate::execution::OrderPlanReport,
) -> (&'static str, serde_json::Value) {
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
    (
        status,
        serde_json::json!({
            "orders": execution.orders.len(),
            "fills": execution.fills.len(),
            "backtest": true
        }),
    )
}

#[derive(Debug, Default)]
struct PointInTimeScoreLedger {
    wallet_stats: HashMap<String, ReplayStats>,
    segment_stats: HashMap<(String, String), ReplayStats>,
    positions: HashMap<(String, String), ReplayWalletPosition>,
}

impl PointInTimeScoreLedger {
    fn wallet_performance(&self, wallet: &str) -> Option<CopyTradeWalletPerformance> {
        let stats = self.wallet_stats.get(&wallet.to_ascii_lowercase())?;
        Some(stats.wallet_performance(wallet))
    }

    fn mrs_score(&self, wallet: &str, score_version: &str) -> Option<CopyTradeMrsScore> {
        let stats = self.wallet_stats.get(&wallet.to_ascii_lowercase())?;
        let input = stats.mrs_input(wallet);
        let score = score_mrs(input);
        Some(CopyTradeMrsScore {
            score: score.score,
            score_version: score_version.to_string(),
            percentile: None,
            metadata: serde_json::json!({
                "score_mode": "point_in_time_replay",
                "components": {
                    "roi_score": score.roi_score,
                    "pnl_score": score.pnl_score,
                    "win_rate_score": score.win_rate_score,
                    "sample_score": score.sample_score,
                    "activity_score": score.activity_score
                }
            }),
        })
    }

    fn segment_score(
        &self,
        wallet: &str,
        segment_key: &str,
        score_version: &str,
    ) -> Option<CopyTradeSegmentScore> {
        let key = (wallet.to_ascii_lowercase(), segment_key.to_string());
        let stats = self.segment_stats.get(&key)?;
        let input = stats.segment_input(wallet, segment_key);
        let score = score_wallet_segment(input);
        let percentile = self.segment_percentile(segment_key, score.score);
        Some(CopyTradeSegmentScore {
            segment_key: segment_key.to_string(),
            score: score.score,
            score_version: score_version.to_string(),
            classifier_version: score.input.classifier_version.clone(),
            confidence: score.confidence,
            closed_positions: score.input.closed_positions,
            winning_positions: score.input.winning_positions,
            win_rate: stats.win_rate(),
            realized_pnl_usd: stats.realized_pnl,
            total_bought_usd: stats.total_bought,
            roi: stats.roi(),
            observed_trade_count: stats.observed_trade_count,
            observed_volume_usd: stats.observed_volume,
            percentile,
            metadata: serde_json::json!({
                "score_mode": "point_in_time_replay",
                "segment_percentile": percentile.map(|value| value.to_string())
            }),
        })
    }

    fn segment_percentile(&self, segment_key: &str, score: Decimal) -> Option<Decimal> {
        let mut count = 0u64;
        let mut less_or_equal = 0u64;
        for ((_, segment), stats) in &self.segment_stats {
            if segment != segment_key {
                continue;
            }
            let candidate = score_wallet_segment(stats.segment_input("", segment)).score;
            count += 1;
            if candidate <= score {
                less_or_equal += 1;
            }
        }
        if count == 0 {
            None
        } else {
            Some((Decimal::from(less_or_equal) / Decimal::from(count)).round_dp(6))
        }
    }

    fn observe_trade(&mut self, trade: &WhaleTrade, classification: &SegmentClassification) {
        let wallet = trade.proxy_wallet.to_ascii_lowercase();
        let asset = trade.asset.clone();
        let segment_key = classification.segment_key.clone();
        let side = trade.side.to_ascii_uppercase();

        self.wallet_stats
            .entry(wallet.clone())
            .or_default()
            .observe_activity(trade);
        self.segment_stats
            .entry((wallet.clone(), segment_key.clone()))
            .or_insert_with(|| ReplayStats {
                classifier_version: classification.classifier_version.clone(),
                ..ReplayStats::default()
            })
            .observe_activity(trade);

        if side == "BUY" {
            self.positions.entry((wallet, asset)).or_default().buy(
                trade.size,
                trade.price,
                segment_key,
            );
            return;
        }
        if side != "SELL" {
            return;
        }
        let Some(position) = self.positions.get_mut(&(wallet.clone(), asset)) else {
            return;
        };
        let Some(close) = position.sell(trade.size, trade.price) else {
            return;
        };
        let close_segment_key = close.segment_key.clone();
        self.wallet_stats
            .entry(wallet.clone())
            .or_default()
            .observe_close(close.clone());
        self.segment_stats
            .entry((wallet, close_segment_key))
            .or_default()
            .observe_close(close);
    }
}

#[derive(Debug, Clone)]
struct ReplayClose {
    realized_pnl: Decimal,
    total_bought: Decimal,
    segment_key: String,
}

#[derive(Debug, Clone, Default)]
struct ReplayWalletPosition {
    size: Decimal,
    avg_price: Decimal,
    segment_key: String,
}

impl ReplayWalletPosition {
    fn buy(&mut self, size: Decimal, price: Decimal, segment_key: String) {
        if size <= Decimal::ZERO {
            return;
        }
        let notional = self.avg_price * self.size + price * size;
        self.size += size;
        if self.size > Decimal::ZERO {
            self.avg_price = notional / self.size;
        }
        self.segment_key = segment_key;
    }

    fn sell(&mut self, size: Decimal, price: Decimal) -> Option<ReplayClose> {
        if self.size <= Decimal::ZERO || size <= Decimal::ZERO {
            return None;
        }
        let close_size = self.size.min(size);
        self.size -= close_size;
        Some(ReplayClose {
            realized_pnl: (price - self.avg_price) * close_size,
            total_bought: self.avg_price * close_size,
            segment_key: self.segment_key.clone(),
        })
    }
}

#[derive(Debug, Clone, Default)]
struct ReplayStats {
    classifier_version: String,
    closed_positions: i32,
    winning_positions: i32,
    realized_pnl: Decimal,
    total_bought: Decimal,
    observed_trade_count: i32,
    observed_volume: Decimal,
    market_keys: HashSet<String>,
}

impl ReplayStats {
    fn observe_activity(&mut self, trade: &WhaleTrade) {
        self.observed_trade_count = self.observed_trade_count.saturating_add(1);
        self.observed_volume += trade.cash_value;
        if let Some(key) = trade
            .market_id
            .as_ref()
            .or(trade.condition_id.as_ref())
            .or(trade.slug.as_ref())
        {
            self.market_keys.insert(key.clone());
        }
    }

    fn observe_close(&mut self, close: ReplayClose) {
        self.closed_positions = self.closed_positions.saturating_add(1);
        if close.realized_pnl > Decimal::ZERO {
            self.winning_positions = self.winning_positions.saturating_add(1);
        }
        self.realized_pnl += close.realized_pnl;
        self.total_bought += close.total_bought;
    }

    fn roi(&self) -> Decimal {
        if self.total_bought > Decimal::ZERO {
            self.realized_pnl / self.total_bought
        } else {
            Decimal::ZERO
        }
    }

    fn win_rate(&self) -> Decimal {
        if self.closed_positions > 0 {
            Decimal::from(self.winning_positions) / Decimal::from(self.closed_positions)
        } else {
            Decimal::ZERO
        }
    }

    fn avg_trade_size(&self) -> Decimal {
        if self.observed_trade_count > 0 {
            self.observed_volume / Decimal::from(self.observed_trade_count)
        } else {
            Decimal::ZERO
        }
    }

    fn wallet_performance(&self, wallet: &str) -> CopyTradeWalletPerformance {
        CopyTradeWalletPerformance {
            proxy_wallet: wallet.to_ascii_lowercase(),
            realized_pnl_usd: self.realized_pnl,
            roi: self.roi(),
            closed_positions: self.closed_positions,
            wallet_score_diagnostic: None,
        }
    }

    fn mrs_input(&self, wallet: &str) -> MrsScoreInput {
        MrsScoreInput {
            proxy_wallet: wallet.to_ascii_lowercase(),
            realized_pnl_usd: self.realized_pnl,
            total_bought_usd: self.total_bought,
            roi: self.roi(),
            closed_positions: self.closed_positions,
            winning_positions: self.winning_positions,
            win_rate: self.win_rate(),
            observed_trade_count: self.observed_trade_count,
            observed_volume_usd: self.observed_volume,
            observed_market_count: self.market_keys.len() as i32,
            avg_trade_size: self.avg_trade_size(),
        }
    }

    fn segment_input(&self, wallet: &str, segment_key: &str) -> WalletSegmentPerformanceInput {
        WalletSegmentPerformanceInput {
            proxy_wallet: wallet.to_ascii_lowercase(),
            segment_key: segment_key.to_string(),
            classifier_version: if self.classifier_version.is_empty() {
                GAMMA_SEGMENT_CLASSIFIER_VERSION.to_string()
            } else {
                self.classifier_version.clone()
            },
            closed_positions: self.closed_positions,
            winning_positions: self.winning_positions,
            realized_pnl_usd: self.realized_pnl,
            total_bought_usd: self.total_bought,
            observed_trade_count: self.observed_trade_count,
            observed_volume_usd: self.observed_volume,
            sample_start: None,
            sample_end: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct HistoricalReplayVenue {
    state: Arc<Mutex<HistoricalReplayState>>,
}

#[derive(Debug, Clone, Default)]
struct HistoricalReplayState {
    orders: HashMap<String, OrderRecord>,
    fills_by_order: HashMap<String, Vec<FillRecord>>,
}

#[async_trait]
impl ExecutionVenue for HistoricalReplayVenue {
    async fn submit_order(&self, request: OrderRequest) -> Result<OrderRecord> {
        let timestamp = replay_timestamp(&request).unwrap_or_else(Utc::now);
        let order_id = format!("backtest-{}", request.client_order_id);
        let order = OrderRecord {
            order_id: order_id.clone(),
            request: request.clone(),
            state: if request.size > Decimal::ZERO {
                OrderState::Filled
            } else {
                OrderState::Rejected
            },
            created_at: timestamp,
            updated_at: timestamp,
        };
        let fills = if order.state == OrderState::Filled {
            vec![FillRecord {
                fill_id: Uuid::new_v5(
                    &Uuid::NAMESPACE_URL,
                    format!("polymarket-backtest-fill:{order_id}:0").as_bytes(),
                ),
                process_id: request.process_id,
                order_id: order_id.clone(),
                token_id: request.token_id.clone(),
                price: request.price,
                size: request.size,
                fee: request.price * request.size * dec!(0.03),
                source: FillSource::Sim,
                filled_at: timestamp,
            }]
        } else {
            Vec::new()
        };
        let mut state = self.state.lock().await;
        state.orders.insert(order_id.clone(), order.clone());
        state.fills_by_order.insert(order_id, fills);
        Ok(order)
    }

    async fn cancel_order(&self, order_id: &str) -> Result<OrderRecord> {
        let mut state = self.state.lock().await;
        if let Some(order) = state.orders.get_mut(order_id) {
            order.state = OrderState::Cancelled;
            return Ok(order.clone());
        }
        anyhow::bail!("backtest order not found: {order_id}")
    }

    async fn cancel_all(&self) -> Result<usize> {
        Ok(0)
    }

    async fn get_balances(&self) -> Result<Vec<(String, Decimal)>> {
        Ok(Vec::new())
    }

    async fn get_open_orders(&self) -> Result<Vec<OrderRecord>> {
        Ok(Vec::new())
    }

    async fn convert_negative_risk(&self, _request: ConversionRequest) -> Result<ConversionResult> {
        anyhow::bail!("backtest venue does not support conversions")
    }

    async fn split_ctf(&self, _market_id: &str, _size: Decimal) -> Result<ConversionResult> {
        anyhow::bail!("backtest venue does not support split_ctf")
    }

    async fn merge_ctf(&self, _market_id: &str, _size: Decimal) -> Result<ConversionResult> {
        anyhow::bail!("backtest venue does not support merge_ctf")
    }

    async fn reconcile(&self) -> Result<ReconciliationReport> {
        Ok(ReconciliationReport {
            open_orders: 0,
            balances_checked: true,
            mismatches_found: 0,
            unresolved_count: 0,
            checked_at: Utc::now(),
        })
    }

    async fn fills_for_order(&self, order_id: &str) -> Result<Vec<FillRecord>> {
        Ok(self
            .state
            .lock()
            .await
            .fills_by_order
            .get(order_id)
            .cloned()
            .unwrap_or_default())
    }

    async fn live_status(&self) -> Result<LiveVenueStatus> {
        anyhow::bail!("backtest venue does not expose live status")
    }

    async fn live_identity_diagnostics(&self) -> Result<LiveIdentityDiagnostics> {
        anyhow::bail!("backtest venue does not expose live diagnostics")
    }

    async fn live_wallet_address_diagnostics(
        &self,
        _candidate_addresses: Vec<String>,
    ) -> Result<LiveWalletAddressDiagnostics> {
        anyhow::bail!("backtest venue does not expose live wallet diagnostics")
    }

    async fn live_order_dry_run(
        &self,
        _request: LiveOrderDryRunRequest,
    ) -> Result<LiveOrderDryRunDiagnostics> {
        anyhow::bail!("backtest venue does not support live dry runs")
    }

    async fn live_poly1271_funder_probe(
        &self,
        _request: LivePoly1271FunderProbeRequest,
    ) -> Result<LivePoly1271FunderProbeResponse> {
        anyhow::bail!("backtest venue does not support live funder probes")
    }

    async fn live_account_reconcile(
        &self,
        _request: AccountReconcileRequest,
    ) -> Result<AccountReconcileReport> {
        anyhow::bail!("backtest venue does not support live account reconciliation")
    }

    async fn set_live_entries_enabled(
        &self,
        _enabled: bool,
        _reason: Option<String>,
    ) -> Result<LiveVenueStatus> {
        anyhow::bail!("backtest venue does not support live entries")
    }
}

fn replay_timestamp(request: &OrderRequest) -> Option<DateTime<Utc>> {
    request
        .metadata
        .get("backtest_fill_timestamp")
        .and_then(|value| value.as_str())
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        .map(|value| value.with_timezone(&Utc))
}

fn merge_json(mut left: serde_json::Value, right: serde_json::Value) -> serde_json::Value {
    if let (Some(left), Some(right)) = (left.as_object_mut(), right.as_object()) {
        for (key, value) in right {
            left.insert(key.clone(), value.clone());
        }
    }
    left
}
