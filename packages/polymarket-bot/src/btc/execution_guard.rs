use std::fmt;

use anyhow::{ensure, Context, Result};
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::{
    strategy::{
        ApprovedIntent, BtcDecision, BtcFeatureSnapshot,
        BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_LINEAGE_VERSION, BTC_FEATURE_LINEAGE_VERSION,
    },
    types::BtcOutcome,
};
use crate::models::{OrderRequest, OrderSide, OrderType};

pub const BTC_REFERENCE_EXECUTION_GUARD_METADATA_KEY: &str = "reference_execution_guard";
pub const BTC_REFERENCE_EXECUTION_GUARD_VERSION: &str = "btc_reference_execution_guard_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BtcReferenceTickEvidence {
    pub tick_id: Uuid,
    pub source_timestamp: DateTime<Utc>,
    pub received_at: DateTime<Utc>,
    pub ingest_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BtcReferenceExecutionGuard {
    pub guard_version: String,
    pub process_id: Uuid,
    pub intent_id: Uuid,
    pub decision_id: Uuid,
    pub decision_at: DateTime<Utc>,
    pub snapshot_id: Uuid,
    pub feature_as_of: DateTime<Utc>,
    pub market_id: String,
    pub token_id: String,
    pub outcome: BtcOutcome,
    pub strategy_version: String,
    pub feature_schema_version: String,
    pub lineage_version: String,
    pub feature_sha256: String,
    pub client_order_id: Uuid,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub limit_price: Decimal,
    pub size: Decimal,
    pub signal_id: Option<Uuid>,
    pub dynamic_fee_rate: Decimal,
    pub chainlink_open: BtcReferenceTickEvidence,
    pub chainlink: BtcReferenceTickEvidence,
    pub binance: BtcReferenceTickEvidence,
    pub max_reference_age_ms: i64,
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BtcReferenceExecutionAssessment {
    pub guard_version: String,
    pub evidence_sha256: String,
    pub validated_at: DateTime<Utc>,
    pub max_reference_age_ms: i64,
    pub chainlink_source_age_ms: i64,
    pub chainlink_receive_age_ms: i64,
    pub binance_source_age_ms: i64,
    pub binance_receive_age_ms: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BtcReferenceExecutionRejectReason {
    MissingGuard,
    InvalidGuard,
    UnsupportedVersion,
    IdentityMismatch,
    EvidenceHashMismatch,
    InvalidFreshnessBound,
    NoncausalEvidence,
    FutureEvidence,
    StaleEvidence,
}

impl BtcReferenceExecutionRejectReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MissingGuard => "missing_reference_execution_guard",
            Self::InvalidGuard => "invalid_reference_execution_guard",
            Self::UnsupportedVersion => "unsupported_reference_execution_guard_version",
            Self::IdentityMismatch => "reference_execution_identity_mismatch",
            Self::EvidenceHashMismatch => "reference_execution_evidence_hash_mismatch",
            Self::InvalidFreshnessBound => "invalid_reference_execution_freshness_bound",
            Self::NoncausalEvidence => "noncausal_reference_execution_evidence",
            Self::FutureEvidence => "future_reference_execution_evidence",
            Self::StaleEvidence => "stale_reference_execution_evidence",
        }
    }
}

impl fmt::Display for BtcReferenceExecutionRejectReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::error::Error for BtcReferenceExecutionRejectReason {}

#[derive(Serialize)]
struct GuardHashEvidence<'a> {
    guard_version: &'a str,
    process_id: Uuid,
    intent_id: Uuid,
    decision_id: Uuid,
    decision_at: DateTime<Utc>,
    snapshot_id: Uuid,
    feature_as_of: DateTime<Utc>,
    market_id: &'a str,
    token_id: &'a str,
    outcome: BtcOutcome,
    strategy_version: &'a str,
    feature_schema_version: &'a str,
    lineage_version: &'a str,
    feature_sha256: &'a str,
    client_order_id: Uuid,
    side: OrderSide,
    order_type: OrderType,
    limit_price: Decimal,
    size: Decimal,
    signal_id: Option<Uuid>,
    dynamic_fee_rate: Decimal,
    chainlink_open: &'a BtcReferenceTickEvidence,
    chainlink: &'a BtcReferenceTickEvidence,
    binance: &'a BtcReferenceTickEvidence,
    max_reference_age_ms: i64,
}

impl BtcReferenceExecutionGuard {
    pub fn from_snapshot(
        snapshot: &BtcFeatureSnapshot,
        decision: &BtcDecision,
        intent: &ApprovedIntent,
        request: &OrderRequest,
        feature_sha256: &str,
        dynamic_fee_rate: Decimal,
        max_reference_age_ms: i64,
    ) -> Result<Self> {
        ensure!(
            max_reference_age_ms > 0,
            "reference execution age bound must be positive"
        );
        ensure!(
            snapshot.process_id != Uuid::nil()
                && snapshot.process_id == decision.process_id
                && snapshot.process_id == intent.process_id,
            "reference execution evidence has inconsistent process ownership"
        );
        ensure!(
            snapshot.snapshot_id == decision.feature_snapshot_id
                && snapshot.snapshot_id == intent.feature_snapshot_id,
            "reference execution evidence has inconsistent snapshot identity"
        );
        ensure!(
            snapshot.market_id == intent.market_id,
            "reference execution evidence has inconsistent market identity"
        );
        ensure!(
            snapshot.feature_schema_version == intent.feature_schema_version,
            "reference execution evidence has inconsistent feature schema"
        );
        ensure!(
            decision.approved_intent.as_ref() == Some(intent),
            "reference execution evidence does not match the approved intent"
        );
        ensure!(
            supported_lineage_version(&snapshot.lineage.lineage_version),
            "reference execution evidence has an unsupported lineage version"
        );
        ensure!(
            request.process_id == Some(intent.process_id)
                && request.market_id == intent.market_id
                && request.token_id == intent.token_id
                && request.side == OrderSide::Buy
                && request.order_type == OrderType::Fok
                && request.price == intent.limit_price
                && request.size == intent.size,
            "reference execution request does not match the approved intent"
        );
        ensure!(
            is_sha256(feature_sha256),
            "reference execution feature hash is invalid"
        );
        ensure!(
            dynamic_fee_rate >= Decimal::ZERO,
            "reference execution fee rate is invalid"
        );

        let mut guard = Self {
            guard_version: BTC_REFERENCE_EXECUTION_GUARD_VERSION.to_string(),
            process_id: snapshot.process_id,
            intent_id: intent.intent_id,
            decision_id: decision.decision_id,
            decision_at: decision.evaluated_at,
            snapshot_id: snapshot.snapshot_id,
            feature_as_of: snapshot.observed_at,
            market_id: intent.market_id.clone(),
            token_id: intent.token_id.clone(),
            outcome: intent.outcome,
            strategy_version: intent.strategy_version.clone(),
            feature_schema_version: intent.feature_schema_version.clone(),
            lineage_version: snapshot.lineage.lineage_version.clone(),
            feature_sha256: feature_sha256.to_string(),
            client_order_id: request.client_order_id,
            side: request.side,
            order_type: request.order_type,
            limit_price: request.price,
            size: request.size,
            signal_id: request.signal_id,
            dynamic_fee_rate,
            chainlink_open: required_tick_evidence(
                "Chainlink open",
                snapshot.lineage.chainlink_open_tick_id,
                snapshot.lineage.chainlink_open_source_timestamp,
                snapshot.lineage.chainlink_open_received_at,
                snapshot.lineage.chainlink_open_ingest_sequence,
            )?,
            chainlink: required_tick_evidence(
                "Chainlink current",
                snapshot.lineage.chainlink_tick_id,
                snapshot.lineage.chainlink_source_timestamp,
                snapshot.lineage.chainlink_received_at,
                snapshot.lineage.chainlink_ingest_sequence,
            )?,
            binance: required_tick_evidence(
                "Binance current",
                snapshot.lineage.binance_tick_id,
                snapshot.lineage.binance_source_timestamp,
                snapshot.lineage.binance_received_at,
                snapshot.lineage.binance_ingest_sequence,
            )?,
            max_reference_age_ms,
            evidence_sha256: String::new(),
        };
        guard.validate_causality().map_err(anyhow::Error::new)?;
        guard.evidence_sha256 = guard.calculate_evidence_sha256()?;
        Ok(guard)
    }

    pub fn insert_into_metadata(&self, metadata: &mut serde_json::Value) -> Result<()> {
        let object = metadata
            .as_object_mut()
            .context("BTC order metadata must be an object")?;
        object.insert(
            BTC_REFERENCE_EXECUTION_GUARD_METADATA_KEY.to_string(),
            serde_json::to_value(self)?,
        );
        object.insert(
            "intent_id".to_string(),
            serde_json::to_value(self.intent_id)?,
        );
        object.insert(
            "decision_at".to_string(),
            serde_json::to_value(self.decision_at)?,
        );
        object.insert(
            "feature_as_of".to_string(),
            serde_json::to_value(self.feature_as_of)?,
        );
        object.insert(
            "lineage_version".to_string(),
            self.lineage_version.clone().into(),
        );
        object.insert(
            "feature_sha256".to_string(),
            self.feature_sha256.clone().into(),
        );
        Ok(())
    }

    pub fn validate_for_request(
        &self,
        request: &OrderRequest,
        checked_at: DateTime<Utc>,
        expected_process_id: Uuid,
        expected_max_reference_age: Duration,
    ) -> std::result::Result<BtcReferenceExecutionAssessment, BtcReferenceExecutionRejectReason>
    {
        if self.guard_version != BTC_REFERENCE_EXECUTION_GUARD_VERSION
            || !supported_lineage_version(&self.lineage_version)
        {
            return Err(BtcReferenceExecutionRejectReason::UnsupportedVersion);
        }
        if self.process_id == Uuid::nil()
            || self.intent_id == Uuid::nil()
            || self.decision_id == Uuid::nil()
            || self.snapshot_id == Uuid::nil()
            || self.process_id != expected_process_id
            || request.process_id != Some(self.process_id)
            || request.client_order_id != self.client_order_id
            || request.market_id != self.market_id
            || request.token_id != self.token_id
            || request.side != self.side
            || request.order_type != self.order_type
            || request.price != self.limit_price
            || request.size != self.size
            || request.signal_id != self.signal_id
            || metadata_uuid(&request.metadata, "process_id") != Some(self.process_id)
            || metadata_uuid(&request.metadata, "intent_id") != Some(self.intent_id)
            || metadata_uuid(&request.metadata, "decision_id") != Some(self.decision_id)
            || metadata_uuid(&request.metadata, "feature_snapshot_id") != Some(self.snapshot_id)
            || request
                .metadata
                .get("strategy_version")
                .and_then(|value| value.as_str())
                != Some(self.strategy_version.as_str())
            || request
                .metadata
                .get("feature_schema_version")
                .and_then(|value| value.as_str())
                != Some(self.feature_schema_version.as_str())
            || request
                .metadata
                .get("lineage_version")
                .and_then(serde_json::Value::as_str)
                != Some(self.lineage_version.as_str())
            || request
                .metadata
                .get("feature_sha256")
                .and_then(serde_json::Value::as_str)
                != Some(self.feature_sha256.as_str())
            || metadata_datetime(&request.metadata, "decision_at") != Some(self.decision_at)
            || metadata_datetime(&request.metadata, "feature_as_of") != Some(self.feature_as_of)
            || metadata_outcome(&request.metadata) != Some(self.outcome)
            || metadata_decimal(&request.metadata, "dynamic_fee_rate")
                != Some(self.dynamic_fee_rate)
        {
            return Err(BtcReferenceExecutionRejectReason::IdentityMismatch);
        }
        if !is_sha256(&self.feature_sha256)
            || self.side != OrderSide::Buy
            || self.order_type != OrderType::Fok
            || self.limit_price <= Decimal::ZERO
            || self.limit_price > Decimal::ONE
            || self.size <= Decimal::ZERO
            || self.dynamic_fee_rate < Decimal::ZERO
            || self.dynamic_fee_rate > Decimal::ONE
            || [&self.chainlink_open, &self.chainlink, &self.binance]
                .into_iter()
                .any(|evidence| evidence.tick_id == Uuid::nil() || evidence.ingest_sequence == 0)
        {
            return Err(BtcReferenceExecutionRejectReason::InvalidGuard);
        }
        self.validate_causality()?;
        if self.feature_as_of > checked_at || self.decision_at > checked_at {
            return Err(BtcReferenceExecutionRejectReason::FutureEvidence);
        }
        let calculated = self
            .calculate_evidence_sha256()
            .map_err(|_| BtcReferenceExecutionRejectReason::InvalidGuard)?;
        if self.evidence_sha256.len() != 64 || calculated != self.evidence_sha256 {
            return Err(BtcReferenceExecutionRejectReason::EvidenceHashMismatch);
        }
        let max_reference_age = Duration::milliseconds(self.max_reference_age_ms);
        if self.max_reference_age_ms <= 0
            || expected_max_reference_age <= Duration::zero()
            || max_reference_age != expected_max_reference_age
        {
            return Err(BtcReferenceExecutionRejectReason::InvalidFreshnessBound);
        }
        let (chainlink_source_age, chainlink_receive_age) =
            evidence_ages(&self.chainlink, checked_at, max_reference_age)?;
        let (binance_source_age, binance_receive_age) =
            evidence_ages(&self.binance, checked_at, max_reference_age)?;
        Ok(BtcReferenceExecutionAssessment {
            guard_version: self.guard_version.clone(),
            evidence_sha256: self.evidence_sha256.clone(),
            validated_at: checked_at,
            max_reference_age_ms: self.max_reference_age_ms,
            chainlink_source_age_ms: chainlink_source_age.num_milliseconds(),
            chainlink_receive_age_ms: chainlink_receive_age.num_milliseconds(),
            binance_source_age_ms: binance_source_age.num_milliseconds(),
            binance_receive_age_ms: binance_receive_age.num_milliseconds(),
        })
    }

    fn validate_causality(&self) -> std::result::Result<(), BtcReferenceExecutionRejectReason> {
        if self.decision_at < self.feature_as_of
            || [&self.chainlink_open, &self.chainlink, &self.binance]
                .into_iter()
                .any(|tick| {
                    tick.source_timestamp > self.feature_as_of
                        || tick.received_at > self.feature_as_of
                })
        {
            return Err(BtcReferenceExecutionRejectReason::NoncausalEvidence);
        }
        Ok(())
    }

    fn calculate_evidence_sha256(&self) -> Result<String> {
        let evidence = GuardHashEvidence {
            guard_version: &self.guard_version,
            process_id: self.process_id,
            intent_id: self.intent_id,
            decision_id: self.decision_id,
            decision_at: self.decision_at,
            snapshot_id: self.snapshot_id,
            feature_as_of: self.feature_as_of,
            market_id: &self.market_id,
            token_id: &self.token_id,
            outcome: self.outcome,
            strategy_version: &self.strategy_version,
            feature_schema_version: &self.feature_schema_version,
            lineage_version: &self.lineage_version,
            feature_sha256: &self.feature_sha256,
            client_order_id: self.client_order_id,
            side: self.side,
            order_type: self.order_type,
            limit_price: self.limit_price,
            size: self.size,
            signal_id: self.signal_id,
            dynamic_fee_rate: self.dynamic_fee_rate,
            chainlink_open: &self.chainlink_open,
            chainlink: &self.chainlink,
            binance: &self.binance,
            max_reference_age_ms: self.max_reference_age_ms,
        };
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&evidence)?)
        ))
    }

    #[cfg(test)]
    pub(crate) fn reseal_for_test(&mut self) {
        self.evidence_sha256 = self
            .calculate_evidence_sha256()
            .expect("test reference evidence must serialize");
    }
}

pub fn reference_execution_guard(
    request: &OrderRequest,
) -> std::result::Result<BtcReferenceExecutionGuard, BtcReferenceExecutionRejectReason> {
    let value = request
        .metadata
        .get(BTC_REFERENCE_EXECUTION_GUARD_METADATA_KEY)
        .cloned()
        .ok_or(BtcReferenceExecutionRejectReason::MissingGuard)?;
    serde_json::from_value(value).map_err(|_| BtcReferenceExecutionRejectReason::InvalidGuard)
}

fn required_tick_evidence(
    name: &str,
    tick_id: Option<Uuid>,
    source_timestamp: Option<DateTime<Utc>>,
    received_at: Option<DateTime<Utc>>,
    ingest_sequence: Option<u64>,
) -> Result<BtcReferenceTickEvidence> {
    let evidence = BtcReferenceTickEvidence {
        tick_id: tick_id.with_context(|| format!("{name} tick ID is missing"))?,
        source_timestamp: source_timestamp
            .with_context(|| format!("{name} source timestamp is missing"))?,
        received_at: received_at.with_context(|| format!("{name} receipt timestamp is missing"))?,
        ingest_sequence: ingest_sequence
            .with_context(|| format!("{name} ingest sequence is missing"))?,
    };
    ensure!(evidence.tick_id != Uuid::nil(), "{name} tick ID is nil");
    ensure!(
        evidence.ingest_sequence > 0,
        "{name} ingest sequence must be positive"
    );
    Ok(evidence)
}

fn metadata_uuid(metadata: &serde_json::Value, key: &str) -> Option<Uuid> {
    metadata
        .get(key)
        .and_then(serde_json::Value::as_str)
        .and_then(|value| Uuid::parse_str(value).ok())
}

fn metadata_datetime(metadata: &serde_json::Value, key: &str) -> Option<DateTime<Utc>> {
    serde_json::from_value(metadata.get(key)?.clone()).ok()
}

fn metadata_decimal(metadata: &serde_json::Value, key: &str) -> Option<Decimal> {
    serde_json::from_value(metadata.get(key)?.clone()).ok()
}

fn metadata_outcome(metadata: &serde_json::Value) -> Option<BtcOutcome> {
    serde_json::from_value(metadata.get("outcome")?.clone()).ok()
}

fn supported_lineage_version(value: &str) -> bool {
    value == BTC_FEATURE_LINEAGE_VERSION
        || value == BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_LINEAGE_VERSION
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn evidence_ages(
    evidence: &BtcReferenceTickEvidence,
    checked_at: DateTime<Utc>,
    max_age: Duration,
) -> std::result::Result<(Duration, Duration), BtcReferenceExecutionRejectReason> {
    if evidence.source_timestamp > checked_at || evidence.received_at > checked_at {
        return Err(BtcReferenceExecutionRejectReason::FutureEvidence);
    }
    let source_age = checked_at - evidence.source_timestamp;
    let receive_age = checked_at - evidence.received_at;
    if source_age > max_age || receive_age > max_age {
        return Err(BtcReferenceExecutionRejectReason::StaleEvidence);
    }
    Ok((source_age, receive_age))
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use rust_decimal_macros::dec;

    use super::*;
    use crate::models::OrderRequest;

    fn sealed_guard(checked_at: DateTime<Utc>) -> BtcReferenceExecutionGuard {
        let tick = |id, age_ms| BtcReferenceTickEvidence {
            tick_id: Uuid::from_u128(id),
            source_timestamp: checked_at - Duration::milliseconds(age_ms),
            received_at: checked_at - Duration::milliseconds(age_ms - 10),
            ingest_sequence: id as u64,
        };
        let mut guard = BtcReferenceExecutionGuard {
            guard_version: BTC_REFERENCE_EXECUTION_GUARD_VERSION.to_string(),
            process_id: Uuid::from_u128(1),
            intent_id: Uuid::from_u128(2),
            decision_id: Uuid::from_u128(3),
            decision_at: checked_at,
            snapshot_id: Uuid::from_u128(4),
            feature_as_of: checked_at,
            market_id: "market".to_string(),
            token_id: "up".to_string(),
            outcome: BtcOutcome::Up,
            strategy_version: "strategy-v1".to_string(),
            feature_schema_version: "features-v1".to_string(),
            lineage_version: BTC_FEATURE_LINEAGE_VERSION.to_string(),
            feature_sha256: "a".repeat(64),
            client_order_id: Uuid::from_u128(8),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            limit_price: dec!(0.40),
            size: dec!(2),
            signal_id: None,
            dynamic_fee_rate: dec!(0.25),
            chainlink_open: tick(5, 1_000),
            chainlink: tick(6, 100),
            binance: tick(7, 90),
            max_reference_age_ms: 2_000,
            evidence_sha256: String::new(),
        };
        guard.evidence_sha256 = guard.calculate_evidence_sha256().unwrap();
        guard
    }

    fn request(guard: &BtcReferenceExecutionGuard) -> OrderRequest {
        let mut metadata = serde_json::json!({
            "execution_intent": "entry",
            "process_id": guard.process_id,
            "intent_id": guard.intent_id,
            "decision_id": guard.decision_id,
            "feature_snapshot_id": guard.snapshot_id,
            "strategy_version": guard.strategy_version,
            "feature_schema_version": guard.feature_schema_version,
            "outcome": guard.outcome,
            "dynamic_fee_rate": guard.dynamic_fee_rate,
        });
        guard.insert_into_metadata(&mut metadata).unwrap();
        OrderRequest {
            client_order_id: guard.client_order_id,
            process_id: Some(guard.process_id),
            market_id: guard.market_id.clone(),
            token_id: guard.token_id.clone(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: guard.limit_price,
            size: guard.size,
            signal_id: guard.signal_id,
            metadata,
        }
    }

    #[test]
    fn guard_enforces_source_and_receipt_freshness_at_the_exact_boundary() {
        let checked_at = Utc.with_ymd_and_hms(2026, 7, 21, 12, 0, 0).unwrap();
        let mut guard = sealed_guard(checked_at);
        let guarded_request = request(&guard);
        assert!(guard
            .validate_for_request(
                &guarded_request,
                checked_at,
                guard.process_id,
                Duration::seconds(2),
            )
            .is_ok());

        guard.chainlink.source_timestamp = checked_at - Duration::seconds(2);
        guard.evidence_sha256 = guard.calculate_evidence_sha256().unwrap();
        let guarded_request = request(&guard);
        assert!(guard
            .validate_for_request(
                &guarded_request,
                checked_at,
                guard.process_id,
                Duration::seconds(2),
            )
            .is_ok());
        assert_eq!(
            guard
                .validate_for_request(
                    &guarded_request,
                    checked_at + Duration::milliseconds(1),
                    guard.process_id,
                    Duration::seconds(2),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::StaleEvidence
        );
    }

    #[test]
    fn guard_rejects_fresh_receipt_with_stale_source_and_future_evidence() {
        let checked_at = Utc.with_ymd_and_hms(2026, 7, 21, 12, 0, 0).unwrap();
        let mut guard = sealed_guard(checked_at);
        guard.chainlink.source_timestamp = checked_at - Duration::milliseconds(2_001);
        guard.chainlink.received_at = checked_at - Duration::milliseconds(1);
        guard.evidence_sha256 = guard.calculate_evidence_sha256().unwrap();
        let guarded_request = request(&guard);
        assert_eq!(
            guard
                .validate_for_request(
                    &guarded_request,
                    checked_at,
                    guard.process_id,
                    Duration::seconds(2),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::StaleEvidence
        );

        guard.chainlink.source_timestamp = checked_at + Duration::milliseconds(1);
        guard.feature_as_of = guard.chainlink.source_timestamp;
        guard.decision_at = guard.feature_as_of;
        guard.evidence_sha256 = guard.calculate_evidence_sha256().unwrap();
        let guarded_request = request(&guard);
        assert_eq!(
            guard
                .validate_for_request(
                    &guarded_request,
                    checked_at,
                    guard.process_id,
                    Duration::seconds(2),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::FutureEvidence
        );
    }

    #[test]
    fn guard_rejects_identity_hash_version_and_causality_mutation() {
        let checked_at = Utc.with_ymd_and_hms(2026, 7, 21, 12, 0, 0).unwrap();
        let guard = sealed_guard(checked_at);

        let mut wrong_process = request(&guard);
        wrong_process.process_id = Some(Uuid::from_u128(99));
        assert_eq!(
            guard
                .validate_for_request(
                    &wrong_process,
                    checked_at,
                    guard.process_id,
                    Duration::seconds(2),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::IdentityMismatch
        );

        let mut changed = guard.clone();
        changed.binance.tick_id = Uuid::from_u128(99);
        let changed_request = request(&changed);
        assert_eq!(
            changed
                .validate_for_request(
                    &changed_request,
                    checked_at,
                    changed.process_id,
                    Duration::seconds(2),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::EvidenceHashMismatch
        );

        let mut unsupported = guard.clone();
        unsupported.guard_version = "future".to_string();
        let unsupported_request = request(&unsupported);
        assert_eq!(
            unsupported
                .validate_for_request(
                    &unsupported_request,
                    checked_at,
                    unsupported.process_id,
                    Duration::seconds(2),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::UnsupportedVersion
        );

        let mut noncausal = guard.clone();
        noncausal.binance.received_at = checked_at + Duration::milliseconds(1);
        noncausal.evidence_sha256 = noncausal.calculate_evidence_sha256().unwrap();
        let noncausal_request = request(&noncausal);
        assert_eq!(
            noncausal
                .validate_for_request(
                    &noncausal_request,
                    checked_at,
                    noncausal.process_id,
                    Duration::seconds(2),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::NoncausalEvidence
        );
    }
}
