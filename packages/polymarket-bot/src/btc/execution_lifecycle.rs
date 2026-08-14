use std::{sync::Arc, time::Duration};

use anyhow::{bail, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::warn;
use uuid::Uuid;

use crate::{
    execution::{ExecutionVenue, ReconciliationReport},
    models::OrderRequest,
};

use super::{
    paper::{PaperPreviewConfig, PaperPreviewResult, PaperVenue},
    repository::BtcRepository,
};

const PAPER_RECONCILE_INTERVAL: Duration = Duration::from_secs(5);
const LIVE_PENDING_REDEMPTION_GATE_REASON: &str = "live_settlement_redemption_unproven";
const LIVE_UNCLEAN_RECONCILIATION_GATE_REASON: &str = "live_reconciliation_unclean";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BtcExecutionMode {
    #[default]
    Paper,
    Live,
}

impl BtcExecutionMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Paper => "paper",
            Self::Live => "live",
        }
    }
}

/// Owns the venue-specific lifecycle around the shared BTC strategy and order pathway.
///
/// Order construction and submission remain on `ExecutionVenue`. These hooks are limited to
/// state that cannot be shared between a simulated balance and an exchange account: resume,
/// settlement/reconciliation, and optional non-mutating execution previews.
#[async_trait]
pub trait BtcExecutionLifecycle: Send + Sync {
    fn mode(&self) -> BtcExecutionMode;

    fn reconcile_interval(&self) -> Duration;

    async fn resume_run(
        &self,
        repository: &BtcRepository,
        process_id: Uuid,
        run_id: Uuid,
    ) -> Result<()>;

    async fn reconcile_run(
        &self,
        repository: &BtcRepository,
        process_id: Uuid,
        run_id: Uuid,
        config_hash: &str,
    ) -> Result<()>;

    async fn preview_order(
        &self,
        _request: &OrderRequest,
        _config: &PaperPreviewConfig,
    ) -> Result<Option<PaperPreviewResult>> {
        Ok(None)
    }
}

pub struct PaperExecutionLifecycle {
    venue: Arc<PaperVenue>,
}

impl PaperExecutionLifecycle {
    pub fn new(venue: Arc<PaperVenue>) -> Self {
        Self { venue }
    }
}

#[async_trait]
impl BtcExecutionLifecycle for PaperExecutionLifecycle {
    fn mode(&self) -> BtcExecutionMode {
        BtcExecutionMode::Paper
    }

    fn reconcile_interval(&self) -> Duration {
        PAPER_RECONCILE_INTERVAL
    }

    async fn resume_run(
        &self,
        repository: &BtcRepository,
        process_id: Uuid,
        run_id: Uuid,
    ) -> Result<()> {
        let state = repository
            .paper_venue_resume_state(process_id, run_id)
            .await?;
        self.venue
            .rehydrate_capital(
                state.entry_debits_usd,
                state.settlement_credits_usd,
                state.order_count,
                state.fill_count,
                state.credited_settlement_ids,
            )
            .await
    }

    async fn reconcile_run(
        &self,
        repository: &BtcRepository,
        process_id: Uuid,
        run_id: Uuid,
        config_hash: &str,
    ) -> Result<()> {
        let pending = repository
            .discover_pending_paper_settlements(process_id, run_id)
            .await?;
        for settlement in pending {
            let credit = self
                .venue
                .apply_settlement_credit(settlement.settlement_id, settlement.payout)
                .await?;
            let evidence = serde_json::json!({
                "evidence_version": "btc_paper_capital_credit_v1",
                "settlement_id": settlement.settlement_id,
                "process_id": settlement.process_id,
                "run_id": settlement.run_id,
                "order_id": settlement.order_id,
                "market_id": settlement.market_id,
                "token_id": settlement.token_id,
                "fill_ids": settlement.fill_ids,
                "official_outcome": settlement.official_outcome,
                "official_winning_token_id": settlement.official_winning_token_id,
                "official_resolution_received_at": settlement.official_resolution_received_at,
                "official_resolution_source": settlement.official_resolution_source,
                "filled_size": settlement.filled_size,
                "entry_notional": settlement.entry_notional,
                "entry_fees": settlement.entry_fees,
                "payout": settlement.payout,
                "net_pnl": settlement.net_pnl,
                "venue_credit": credit,
                "credited_by_config_hash": config_hash,
            });
            let marked = repository
                .mark_paper_settlement_credited(
                    process_id,
                    run_id,
                    settlement.settlement_id,
                    &evidence,
                )
                .await?;
            if !marked {
                warn!(
                    settlement_id = %settlement.settlement_id,
                    run_id = %run_id,
                    "BTC paper settlement was already credited by a concurrent reconciliation"
                );
            }
        }
        Ok(())
    }

    async fn preview_order(
        &self,
        request: &OrderRequest,
        config: &PaperPreviewConfig,
    ) -> Result<Option<PaperPreviewResult>> {
        self.venue.preview_order(request, config).await.map(Some)
    }
}

pub struct LiveExecutionLifecycle {
    venue: Arc<dyn ExecutionVenue>,
    reconcile_interval: Duration,
}

impl LiveExecutionLifecycle {
    pub fn new(venue: Arc<dyn ExecutionVenue>, reconcile_interval: Duration) -> Result<Self> {
        if reconcile_interval.is_zero() {
            bail!("live execution reconcile interval must be positive");
        }
        Ok(Self {
            venue,
            reconcile_interval,
        })
    }
}

#[async_trait]
impl BtcExecutionLifecycle for LiveExecutionLifecycle {
    fn mode(&self) -> BtcExecutionMode {
        BtcExecutionMode::Live
    }

    fn reconcile_interval(&self) -> Duration {
        self.reconcile_interval
    }

    async fn resume_run(
        &self,
        _repository: &BtcRepository,
        _process_id: Uuid,
        _run_id: Uuid,
    ) -> Result<()> {
        // The runner performs one mandatory reconciliation after resume hydration and admission
        // initialization. Live execution has no in-memory capital state to hydrate here.
        Ok(())
    }

    async fn reconcile_run(
        &self,
        repository: &BtcRepository,
        process_id: Uuid,
        run_id: Uuid,
        _config_hash: &str,
    ) -> Result<()> {
        let reconciliation = match self.venue.reconcile().await {
            Ok(reconciliation) => reconciliation,
            Err(error) => {
                warn!(
                    process_id = %process_id,
                    run_id = %run_id,
                    error = %error,
                    "BTC live HTTP reconciliation backup failed; preserving process authorization and retrying"
                );
                return Ok(());
            }
        };
        let pending = repository
            .discover_pending_settlements(process_id, run_id, BtcExecutionMode::Live)
            .await?;
        if let Some(reason) = live_reconciliation_gate_reason(&reconciliation, pending.len()) {
            if reason == LIVE_PENDING_REDEMPTION_GATE_REASON {
                let status = self
                    .venue
                    .set_live_entries_enabled(false, Some(reason.to_string()))
                    .await?;
                if status.entries_enabled {
                    bail!("BTC pending redemption did not close execution entries");
                }
                warn!(
                    process_id = %process_id,
                    run_id = %run_id,
                    pending_settlement_count = pending.len(),
                    reason,
                    "BTC live settlement redemption remains unproven; entries remain manually closed"
                );
                return Ok(());
            }
            warn!(
                process_id = %process_id,
                run_id = %run_id,
                pending_settlement_count = pending.len(),
                balances_checked = reconciliation.balances_checked,
                reconciliation_mismatches = reconciliation.mismatches_found,
                reconciliation_unresolved = reconciliation.unresolved_count,
                reason,
                "BTC live HTTP reconciliation backup reported an unsafe state; clean reconciliation restores readiness automatically"
            );
            return Ok(());
        }
        Ok(())
    }
}

fn live_reconciliation_gate_reason(
    report: &ReconciliationReport,
    pending_settlement_count: usize,
) -> Option<&'static str> {
    if pending_settlement_count != 0 {
        Some(LIVE_PENDING_REDEMPTION_GATE_REASON)
    } else if !report.balances_checked
        || report.mismatches_found != 0
        || report.unresolved_count != 0
    {
        Some(LIVE_UNCLEAN_RECONCILIATION_GATE_REASON)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;

    use crate::execution::ReconciliationReport;

    use super::{
        live_reconciliation_gate_reason, BtcExecutionMode, LIVE_PENDING_REDEMPTION_GATE_REASON,
        LIVE_UNCLEAN_RECONCILIATION_GATE_REASON,
    };

    #[test]
    fn execution_mode_names_are_stable() {
        assert_eq!(BtcExecutionMode::Paper.as_str(), "paper");
        assert_eq!(BtcExecutionMode::Live.as_str(), "live");
    }

    #[test]
    fn live_reconciliation_gate_is_nonfatal_and_fail_closed() {
        let clean = ReconciliationReport {
            open_orders: 0,
            balances_checked: true,
            mismatches_found: 0,
            unresolved_count: 0,
            checked_at: Utc::now(),
        };
        assert_eq!(live_reconciliation_gate_reason(&clean, 0), None);

        let mut unchecked = clean.clone();
        unchecked.balances_checked = false;
        assert_eq!(
            live_reconciliation_gate_reason(&unchecked, 0),
            Some(LIVE_UNCLEAN_RECONCILIATION_GATE_REASON)
        );

        let mut mismatched = clean.clone();
        mismatched.mismatches_found = 1;
        assert_eq!(
            live_reconciliation_gate_reason(&mismatched, 0),
            Some(LIVE_UNCLEAN_RECONCILIATION_GATE_REASON)
        );

        let mut unresolved = clean;
        unresolved.unresolved_count = 1;
        assert_eq!(
            live_reconciliation_gate_reason(&unresolved, 0),
            Some(LIVE_UNCLEAN_RECONCILIATION_GATE_REASON)
        );
        assert_eq!(
            live_reconciliation_gate_reason(&unresolved, 1),
            Some(LIVE_PENDING_REDEMPTION_GATE_REASON),
            "unredeemed settlement is the stronger bounded gate reason"
        );
    }

    #[test]
    fn pending_live_settlement_uses_a_bounded_redemption_proof_gate() {
        assert_eq!(
            LIVE_PENDING_REDEMPTION_GATE_REASON,
            "live_settlement_redemption_unproven"
        );
        assert!(LIVE_PENDING_REDEMPTION_GATE_REASON.len() <= 128);
    }
}
