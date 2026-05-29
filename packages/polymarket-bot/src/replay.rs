use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    backfill::{run_resolved_copy_trade_signal, CopyTradeRunConfig, ResolvedCopyTradeSignal},
    clob::ClobClient,
    copytrade::{
        CopyTradeConfig, CopyTradeMrsScore, CopyTradeSegmentScore, CopyTradeWalletPerformance,
        ObservedMarket,
    },
    execution::{execute_order_plan, ExecutionVenue, OrderPlan},
    models::{TradingProcessConfig, WhaleTrade},
    segments::{
        classify_trade_segment, score_wallet_segment, SegmentClassification,
        WalletSegmentPerformanceInput, GAMMA_SEGMENT_CLASSIFIER_VERSION,
    },
    store::Store,
    trade_pnl::risk_control_order_request,
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
    pub take_profit_exits_applied: u64,
    pub stop_loss_exits_applied: u64,
    pub risk_control_exit_orders_inserted: usize,
    pub risk_control_exit_fills_inserted: usize,
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
    venue: Arc<dyn ExecutionVenue>,
    clob: ClobClient,
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
    let mut open_notional_by_process = HashMap::<Uuid, Decimal>::new();
    for process in &job.processes {
        if process.process_config.effective_execution().execute_signals {
            open_notional_by_process.insert(
                process.backtest_process_id,
                store
                    .process_open_notional(process.backtest_process_id)
                    .await?,
            );
        }
    }

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
            let execution = process.process_config.effective_execution();
            let performance = ledger.wallet_performance(&trade.proxy_wallet);
            let mrs_score = ledger.mrs_score(&trade.proxy_wallet, &config.mrs_score_version);
            let segment_score = ledger.segment_score(
                &trade.proxy_wallet,
                &classification.segment_key,
                &config.segment_score_version,
            );
            let trade_summary = run_resolved_copy_trade_signal(
                &store,
                Some(venue.as_ref()),
                Some(&clob),
                ResolvedCopyTradeSignal {
                    trade: &trade,
                    performance: performance.as_ref(),
                    mrs_score: mrs_score.as_ref(),
                    segment_score: segment_score.as_ref(),
                    segment_classification: classification.clone(),
                    observed: ObservedMarket {
                        observed_price: trade.price,
                        available_depth_usd: trade.cash_value,
                        observed_at: trade.timestamp_utc,
                    },
                    order_metadata: serde_json::json!({
                        "backtest": true,
                        "backtest_run_id": job.backtest_run_id,
                        "source_process_id": process.source_process_id,
                        "backtest_process_id": process.backtest_process_id,
                        "backtest_fill_timestamp": trade.timestamp_utc
                    }),
                },
                &CopyTradeRunConfig {
                    process_id: Some(process.backtest_process_id),
                    copy_trade: config,
                    execute_signals: execution.execute_signals,
                    require_entry_markability: true,
                },
                false,
                open_notional_by_process.get_mut(&process.backtest_process_id),
            )
            .await?;
            summary.signals_inserted += trade_summary.signals_inserted;
            summary.signals_detected += trade_summary.signals_detected;
            summary.orders_inserted += trade_summary.orders_inserted;
            summary.fills_inserted += trade_summary.fills_inserted;
            summary.rejections += trade_summary.rejections;
            summary.positions_created += store.backfill_trade_positions_from_copy_signals().await?;
            summary.whale_exits_applied += store
                .apply_backtest_whale_led_trade_exits_for_process_until(
                    process.backtest_process_id,
                    trade.timestamp_utc,
                )
                .await?;
            let risk_control = execute_replay_risk_control_exits(
                &store,
                venue.as_ref(),
                process,
                job.backtest_run_id,
                trade.timestamp_utc,
            )
            .await?;
            summary.marks_inserted += risk_control.marks_inserted;
            summary.take_profit_exits_applied += risk_control.take_profit_exits_applied;
            summary.stop_loss_exits_applied += risk_control.stop_loss_exits_applied;
            summary.risk_control_exit_orders_inserted += risk_control.orders_inserted;
            summary.risk_control_exit_fills_inserted += risk_control.fills_inserted;
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

#[derive(Debug, Default)]
struct ReplayRiskControlReport {
    marks_inserted: u64,
    take_profit_exits_applied: u64,
    stop_loss_exits_applied: u64,
    orders_inserted: usize,
    fills_inserted: usize,
}

async fn execute_replay_risk_control_exits(
    store: &Store,
    venue: &dyn ExecutionVenue,
    process: &BacktestReplayProcess,
    backtest_run_id: Uuid,
    as_of: DateTime<Utc>,
) -> Result<ReplayRiskControlReport> {
    let exit_rules = process.process_config.effective_exit_rules();
    let take_profit = &exit_rules.take_profit;
    let stop_loss = &exit_rules.stop_loss;
    if !take_profit.take_profit_enabled && !stop_loss.stop_loss_enabled {
        return Ok(ReplayRiskControlReport::default());
    }

    let mut report = ReplayRiskControlReport {
        marks_inserted: store
            .mark_open_trade_positions_for_process_as_of(process.backtest_process_id, as_of)
            .await?,
        ..ReplayRiskControlReport::default()
    };

    if take_profit.take_profit_enabled {
        let candidates = store
            .fetch_take_profit_trade_exit_candidates_as_of(
                process.backtest_process_id,
                take_profit.take_profit_roi,
                take_profit.exit_size_fraction,
                Duration::seconds(take_profit.min_hold_secs.max(0)),
                Duration::seconds(take_profit.require_fresh_mark_secs.max(0)),
                take_profit.max_exit_slippage_bps,
                100,
                as_of,
            )
            .await?;
        let applied = execute_replay_risk_control_candidates(
            store,
            venue,
            process,
            backtest_run_id,
            candidates,
        )
        .await?;
        report.take_profit_exits_applied += applied.exits_applied;
        report.orders_inserted += applied.orders_inserted;
        report.fills_inserted += applied.fills_inserted;
    }

    if stop_loss.stop_loss_enabled && stop_loss.stop_loss_roi < Decimal::ZERO {
        let candidates = store
            .fetch_stop_loss_trade_exit_candidates_as_of(
                process.backtest_process_id,
                stop_loss.stop_loss_roi,
                stop_loss.exit_size_fraction,
                Duration::seconds(stop_loss.min_hold_secs.max(0)),
                Duration::seconds(stop_loss.require_fresh_mark_secs.max(0)),
                stop_loss.max_exit_slippage_bps,
                100,
                as_of,
            )
            .await?;
        let applied = execute_replay_risk_control_candidates(
            store,
            venue,
            process,
            backtest_run_id,
            candidates,
        )
        .await?;
        report.stop_loss_exits_applied += applied.exits_applied;
        report.orders_inserted += applied.orders_inserted;
        report.fills_inserted += applied.fills_inserted;
    }

    Ok(report)
}

#[derive(Debug, Default)]
struct ReplayRiskControlApplied {
    exits_applied: u64,
    orders_inserted: usize,
    fills_inserted: usize,
}

async fn execute_replay_risk_control_candidates(
    store: &Store,
    venue: &dyn ExecutionVenue,
    process: &BacktestReplayProcess,
    backtest_run_id: Uuid,
    candidates: Vec<crate::store::TakeProfitTradeExitCandidate>,
) -> Result<ReplayRiskControlApplied> {
    let mut report = ReplayRiskControlApplied::default();
    for candidate in candidates {
        let mut request = risk_control_order_request(&candidate);
        request.metadata = merge_json(
            request.metadata.clone(),
            serde_json::json!({
                "backtest": true,
                "backtest_run_id": backtest_run_id,
                "source_process_id": process.source_process_id,
                "backtest_process_id": process.backtest_process_id,
                "backtest_fill_timestamp": candidate.exit_timestamp
            }),
        );
        let execution = execute_order_plan(
            venue,
            OrderPlan {
                plan_id: Uuid::new_v4(),
                orders: vec![request],
            },
        )
        .await?;
        report.orders_inserted += execution.orders.len();
        report.fills_inserted += execution.fills.len();
        store.persist_order_plan_report(&execution).await?;
        report.exits_applied += store
            .apply_take_profit_trade_exit(&candidate, &execution)
            .await?;
    }
    Ok(report)
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

fn merge_json(mut left: serde_json::Value, right: serde_json::Value) -> serde_json::Value {
    if let (Some(left), Some(right)) = (left.as_object_mut(), right.as_object()) {
        for (key, value) in right {
            left.insert(key.clone(), value.clone());
        }
    }
    left
}
