use std::fmt;

use anyhow::{ensure, Context, Result};
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::directional_model::BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION;
use super::strategy::{
    BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_FAMILY, BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_VERSION,
    BTC_DIRECTIONAL_MODEL_STRATEGY_FAMILY,
};
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
const LEGACY_BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION: &str =
    "btc_reference_execution_guard_directional_model_v1";
pub const BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION: &str =
    "btc_reference_execution_guard_directional_model_v2";
pub const BTC_ASYMMETRIC_VALUE_MODEL_EXECUTION_GUARD_VERSION: &str =
    "btc_reference_execution_guard_asymmetric_value_model_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BtcExecutionFreshnessBounds {
    pub max_reference_age_ms: i64,
    pub max_directional_feature_age_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BtcReferenceTickEvidence {
    pub tick_id: Uuid,
    pub source_timestamp: DateTime<Utc>,
    pub received_at: DateTime<Utc>,
    pub ingest_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BtcDirectionalModelExecutionEvidence {
    pub model_key: String,
    pub artifact_sha256: String,
    pub feature_schema_sha256: String,
    pub input_sha256: String,
    pub window_start: DateTime<Utc>,
    pub feature_as_of: DateTime<Utc>,
    pub seconds_elapsed: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BtcExecutionBookEvidence {
    pub market_id: String,
    pub token_id: String,
    pub checkpoint_id: Uuid,
    pub connection_id: Uuid,
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
    #[serde(default, rename = "signal_id")]
    legacy_signal_id: Option<Uuid>,
    pub dynamic_fee_rate: Decimal,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_open: Option<BtcReferenceTickEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink: Option<BtcReferenceTickEvidence>,
    pub binance: BtcReferenceTickEvidence,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directional_model: Option<BtcDirectionalModelExecutionEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_book: Option<BtcExecutionBookEvidence>,
    pub max_reference_age_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_directional_feature_age_ms: Option<i64>,
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BtcReferenceExecutionAssessment {
    pub guard_version: String,
    pub evidence_sha256: String,
    pub validated_at: DateTime<Utc>,
    pub max_reference_age_ms: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_directional_feature_age_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chainlink_source_age_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chainlink_receive_age_ms: Option<i64>,
    pub binance_source_age_ms: i64,
    pub binance_receive_age_ms: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub directional_model_feature_age_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_book_source_age_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_book_receive_age_ms: Option<i64>,
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
struct ReferenceGuardHashEvidence<'a> {
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

#[derive(Serialize)]
struct DirectionalModelGuardHashEvidence<'a> {
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
    directional_model: &'a BtcDirectionalModelExecutionEvidence,
    binance: &'a BtcReferenceTickEvidence,
    selected_book: &'a BtcExecutionBookEvidence,
    max_reference_age_ms: i64,
}

#[derive(Serialize)]
struct DirectionalModelGuardHashEvidenceV2<'a> {
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
    directional_model: &'a BtcDirectionalModelExecutionEvidence,
    binance: &'a BtcReferenceTickEvidence,
    selected_book: &'a BtcExecutionBookEvidence,
    max_reference_age_ms: i64,
    max_directional_feature_age_ms: i64,
}

impl BtcReferenceExecutionGuard {
    pub fn from_snapshot(
        snapshot: &BtcFeatureSnapshot,
        decision: &BtcDecision,
        intent: &ApprovedIntent,
        request: &OrderRequest,
        feature_sha256: &str,
        dynamic_fee_rate: Decimal,
        freshness_bounds: BtcExecutionFreshnessBounds,
    ) -> Result<Self> {
        let BtcExecutionFreshnessBounds {
            max_reference_age_ms,
            max_directional_feature_age_ms,
        } = freshness_bounds;
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

        let model_strategy = matches!(
            intent.strategy_version.as_str(),
            BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION | BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_VERSION
        );
        if model_strategy {
            ensure!(
                max_directional_feature_age_ms.is_some_and(|max_age| max_age > 0),
                "directional-model execution age bound must be positive"
            );
        } else {
            ensure!(
                max_directional_feature_age_ms.is_none(),
                "non-model execution cannot carry a directional feature age bound"
            );
        }
        let directional_model = if model_strategy {
            let features = snapshot
                .directional_model
                .as_ref()
                .context("directional-model execution evidence is missing")?;
            Some(BtcDirectionalModelExecutionEvidence {
                model_key: features.model_key.clone(),
                artifact_sha256: features.model_artifact_sha256.clone(),
                feature_schema_sha256: features.feature_schema_sha256.clone(),
                input_sha256: features.input_sha256.clone(),
                window_start: snapshot.window_start,
                feature_as_of: features.feature_as_of,
                seconds_elapsed: features.seconds_elapsed,
            })
        } else {
            ensure!(
                snapshot.directional_model.is_none(),
                "non-model intent includes directional-model execution evidence"
            );
            None
        };
        let (guard_version, chainlink_open, chainlink, selected_book) = if model_strategy {
            (
                if intent.strategy_version == BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION {
                    BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION.to_string()
                } else {
                    BTC_ASYMMETRIC_VALUE_MODEL_EXECUTION_GUARD_VERSION.to_string()
                },
                None,
                None,
                Some(required_book_evidence(snapshot, intent)?),
            )
        } else {
            (
                BTC_REFERENCE_EXECUTION_GUARD_VERSION.to_string(),
                Some(required_tick_evidence(
                    "Chainlink open",
                    snapshot.lineage.chainlink_open_tick_id,
                    snapshot.lineage.chainlink_open_source_timestamp,
                    snapshot.lineage.chainlink_open_received_at,
                    snapshot.lineage.chainlink_open_ingest_sequence,
                )?),
                Some(required_tick_evidence(
                    "Chainlink current",
                    snapshot.lineage.chainlink_tick_id,
                    snapshot.lineage.chainlink_source_timestamp,
                    snapshot.lineage.chainlink_received_at,
                    snapshot.lineage.chainlink_ingest_sequence,
                )?),
                None,
            )
        };
        let mut guard = Self {
            guard_version,
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
            legacy_signal_id: None,
            dynamic_fee_rate,
            chainlink_open,
            chainlink,
            binance: required_tick_evidence(
                "Binance current",
                snapshot.lineage.binance_tick_id,
                snapshot.lineage.binance_source_timestamp,
                snapshot.lineage.binance_received_at,
                snapshot.lineage.binance_ingest_sequence,
            )?,
            directional_model,
            selected_book,
            max_reference_age_ms,
            max_directional_feature_age_ms,
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
        expected_max_directional_feature_age: Option<Duration>,
    ) -> std::result::Result<BtcReferenceExecutionAssessment, BtcReferenceExecutionRejectReason>
    {
        if !matches!(
            self.guard_version.as_str(),
            BTC_REFERENCE_EXECUTION_GUARD_VERSION
                | LEGACY_BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION
                | BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION
                | BTC_ASYMMETRIC_VALUE_MODEL_EXECUTION_GUARD_VERSION
        ) || !supported_lineage_version(&self.lineage_version)
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
            || self.legacy_signal_id.is_some()
            || self.side != OrderSide::Buy
            || self.order_type != OrderType::Fok
            || self.limit_price <= Decimal::ZERO
            || self.limit_price > Decimal::ONE
            || self.size <= Decimal::ZERO
            || self.dynamic_fee_rate < Decimal::ZERO
            || self.dynamic_fee_rate > Decimal::ONE
            || self.binance.tick_id == Uuid::nil()
            || self.binance.ingest_sequence == 0
        {
            return Err(BtcReferenceExecutionRejectReason::InvalidGuard);
        }
        self.validate_version_contract(request)?;
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
        let max_reference_age = Duration::try_milliseconds(self.max_reference_age_ms)
            .filter(|duration| *duration > Duration::zero())
            .ok_or(BtcReferenceExecutionRejectReason::InvalidFreshnessBound)?;
        if expected_max_reference_age <= Duration::zero()
            || max_reference_age != expected_max_reference_age
        {
            return Err(BtcReferenceExecutionRejectReason::InvalidFreshnessBound);
        }
        let (binance_source_age, binance_receive_age) =
            evidence_ages(&self.binance, checked_at, max_reference_age)?;
        let (chainlink_source_age_ms, chainlink_receive_age_ms) =
            if let Some(chainlink) = self.chainlink.as_ref() {
                let (source_age, receive_age) =
                    evidence_ages(chainlink, checked_at, max_reference_age)?;
                (
                    Some(source_age.num_milliseconds()),
                    Some(receive_age.num_milliseconds()),
                )
            } else {
                (None, None)
            };
        let directional_model_feature_age_ms = if let Some(model) = self.directional_model.as_ref()
        {
            let max_directional_feature_age =
                if self.guard_version == LEGACY_BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION {
                    if self.max_directional_feature_age_ms.is_some() {
                        return Err(BtcReferenceExecutionRejectReason::InvalidGuard);
                    }
                    max_reference_age
                } else {
                    let max_age = self
                        .max_directional_feature_age_ms
                        .and_then(Duration::try_milliseconds)
                        .filter(|duration| *duration > Duration::zero())
                        .ok_or(BtcReferenceExecutionRejectReason::InvalidFreshnessBound)?;
                    if expected_max_directional_feature_age != Some(max_age) {
                        return Err(BtcReferenceExecutionRejectReason::InvalidFreshnessBound);
                    }
                    max_age
                };
            Some(
                bounded_timestamp_age(
                    model.feature_as_of,
                    checked_at,
                    max_directional_feature_age,
                )?
                .num_milliseconds(),
            )
        } else {
            if self.max_directional_feature_age_ms.is_some()
                || expected_max_directional_feature_age.is_some()
            {
                return Err(BtcReferenceExecutionRejectReason::InvalidFreshnessBound);
            }
            None
        };
        let (selected_book_source_age_ms, selected_book_receive_age_ms) =
            if let Some(book) = self.selected_book.as_ref() {
                let source_age =
                    bounded_timestamp_age(book.source_timestamp, checked_at, max_reference_age)?;
                let receive_age =
                    bounded_timestamp_age(book.received_at, checked_at, max_reference_age)?;
                (
                    Some(source_age.num_milliseconds()),
                    Some(receive_age.num_milliseconds()),
                )
            } else {
                (None, None)
            };
        Ok(BtcReferenceExecutionAssessment {
            guard_version: self.guard_version.clone(),
            evidence_sha256: self.evidence_sha256.clone(),
            validated_at: checked_at,
            max_reference_age_ms: self.max_reference_age_ms,
            max_directional_feature_age_ms: self.max_directional_feature_age_ms,
            chainlink_source_age_ms,
            chainlink_receive_age_ms,
            binance_source_age_ms: binance_source_age.num_milliseconds(),
            binance_receive_age_ms: binance_receive_age.num_milliseconds(),
            directional_model_feature_age_ms,
            selected_book_source_age_ms,
            selected_book_receive_age_ms,
        })
    }

    fn validate_version_contract(
        &self,
        request: &OrderRequest,
    ) -> std::result::Result<(), BtcReferenceExecutionRejectReason> {
        match self.guard_version.as_str() {
            BTC_REFERENCE_EXECUTION_GUARD_VERSION => {
                let chainlink_open = self
                    .chainlink_open
                    .as_ref()
                    .ok_or(BtcReferenceExecutionRejectReason::InvalidGuard)?;
                let chainlink = self
                    .chainlink
                    .as_ref()
                    .ok_or(BtcReferenceExecutionRejectReason::InvalidGuard)?;
                if matches!(
                    self.strategy_version.as_str(),
                    BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION
                        | BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_VERSION
                ) || self.directional_model.is_some()
                    || self.selected_book.is_some()
                    || self.max_directional_feature_age_ms.is_some()
                    || [chainlink_open, chainlink].into_iter().any(|evidence| {
                        evidence.tick_id == Uuid::nil() || evidence.ingest_sequence == 0
                    })
                {
                    return Err(BtcReferenceExecutionRejectReason::InvalidGuard);
                }
            }
            LEGACY_BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION
            | BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION
            | BTC_ASYMMETRIC_VALUE_MODEL_EXECUTION_GUARD_VERSION => {
                let model = self
                    .directional_model
                    .as_ref()
                    .ok_or(BtcReferenceExecutionRejectReason::InvalidGuard)?;
                let book = self
                    .selected_book
                    .as_ref()
                    .ok_or(BtcReferenceExecutionRejectReason::InvalidGuard)?;
                let elapsed = Duration::try_seconds(model.seconds_elapsed)
                    .and_then(|elapsed| model.window_start.checked_add_signed(elapsed));
                let asymmetric =
                    self.guard_version == BTC_ASYMMETRIC_VALUE_MODEL_EXECUTION_GUARD_VERSION;
                let expected_strategy_version = if asymmetric {
                    BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_VERSION
                } else {
                    BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION
                };
                let expected_strategy_family = if asymmetric {
                    BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_FAMILY
                } else {
                    BTC_DIRECTIONAL_MODEL_STRATEGY_FAMILY
                };
                if self.chainlink_open.is_some()
                    || self.chainlink.is_some()
                    || self.strategy_version != expected_strategy_version
                    || model.model_key.trim().is_empty()
                    || !is_sha256(&model.artifact_sha256)
                    || !is_sha256(&model.feature_schema_sha256)
                    || !is_sha256(&model.input_sha256)
                    || model.seconds_elapsed <= 0
                    || model.seconds_elapsed >= 300
                    || elapsed != Some(model.feature_as_of)
                    || book.market_id != self.market_id
                    || book.token_id != self.token_id
                    || book.checkpoint_id == Uuid::nil()
                    || book.connection_id == Uuid::nil()
                    || book.ingest_sequence == 0
                    || (self.guard_version == LEGACY_BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION
                        && self.max_directional_feature_age_ms.is_some())
                    || ((self.guard_version == BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION
                        || asymmetric)
                        && self
                            .max_directional_feature_age_ms
                            .is_none_or(|max_age| max_age <= 0))
                {
                    return Err(BtcReferenceExecutionRejectReason::InvalidGuard);
                }
                if request
                    .metadata
                    .get("strategy")
                    .and_then(serde_json::Value::as_str)
                    != Some(expected_strategy_family)
                    || request
                        .metadata
                        .get("profile_id")
                        .and_then(serde_json::Value::as_str)
                        != Some(model.model_key.as_str())
                    || request
                        .metadata
                        .get("profile_sha256")
                        .and_then(serde_json::Value::as_str)
                        != Some(model.artifact_sha256.as_str())
                    || (!asymmetric
                        && (request
                            .metadata
                            .pointer("/prediction/status")
                            .and_then(serde_json::Value::as_str)
                            != Some("directional_prediction")
                            || request
                                .metadata
                                .pointer("/prediction/outcome")
                                .and_then(serde_json::Value::as_str)
                                != Some(match self.outcome {
                                    BtcOutcome::Up => "up",
                                    BtcOutcome::Down => "down",
                                })))
                    || (asymmetric && request.metadata.get("prediction").is_some())
                {
                    return Err(BtcReferenceExecutionRejectReason::IdentityMismatch);
                }
            }
            _ => return Err(BtcReferenceExecutionRejectReason::UnsupportedVersion),
        }
        Ok(())
    }

    fn validate_causality(&self) -> std::result::Result<(), BtcReferenceExecutionRejectReason> {
        if self.decision_at < self.feature_as_of
            || self.binance.source_timestamp > self.feature_as_of
            || self.binance.received_at > self.feature_as_of
        {
            return Err(BtcReferenceExecutionRejectReason::NoncausalEvidence);
        }
        match self.guard_version.as_str() {
            BTC_REFERENCE_EXECUTION_GUARD_VERSION => {
                let chainlink_open = self
                    .chainlink_open
                    .as_ref()
                    .ok_or(BtcReferenceExecutionRejectReason::InvalidGuard)?;
                let chainlink = self
                    .chainlink
                    .as_ref()
                    .ok_or(BtcReferenceExecutionRejectReason::InvalidGuard)?;
                if [chainlink_open, chainlink].into_iter().any(|tick| {
                    tick.source_timestamp > self.feature_as_of
                        || tick.received_at > self.feature_as_of
                }) {
                    return Err(BtcReferenceExecutionRejectReason::NoncausalEvidence);
                }
            }
            LEGACY_BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION
            | BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION
            | BTC_ASYMMETRIC_VALUE_MODEL_EXECUTION_GUARD_VERSION => {
                let model = self
                    .directional_model
                    .as_ref()
                    .ok_or(BtcReferenceExecutionRejectReason::InvalidGuard)?;
                let book = self
                    .selected_book
                    .as_ref()
                    .ok_or(BtcReferenceExecutionRejectReason::InvalidGuard)?;
                if model.feature_as_of > self.feature_as_of
                    || book.source_timestamp > self.feature_as_of
                    || book.received_at > self.feature_as_of
                {
                    return Err(BtcReferenceExecutionRejectReason::NoncausalEvidence);
                }
            }
            _ => return Err(BtcReferenceExecutionRejectReason::UnsupportedVersion),
        }
        Ok(())
    }

    fn calculate_evidence_sha256(&self) -> Result<String> {
        let bytes = match self.guard_version.as_str() {
            BTC_REFERENCE_EXECUTION_GUARD_VERSION => {
                let evidence = ReferenceGuardHashEvidence {
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
                    signal_id: self.legacy_signal_id,
                    dynamic_fee_rate: self.dynamic_fee_rate,
                    chainlink_open: self
                        .chainlink_open
                        .as_ref()
                        .context("reference guard Chainlink open evidence is missing")?,
                    chainlink: self
                        .chainlink
                        .as_ref()
                        .context("reference guard Chainlink evidence is missing")?,
                    binance: &self.binance,
                    max_reference_age_ms: self.max_reference_age_ms,
                };
                serde_json::to_vec(&evidence)?
            }
            LEGACY_BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION => {
                let evidence = DirectionalModelGuardHashEvidence {
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
                    signal_id: self.legacy_signal_id,
                    dynamic_fee_rate: self.dynamic_fee_rate,
                    directional_model: self
                        .directional_model
                        .as_ref()
                        .context("directional-model guard evidence is missing")?,
                    binance: &self.binance,
                    selected_book: self
                        .selected_book
                        .as_ref()
                        .context("directional-model guard book evidence is missing")?,
                    max_reference_age_ms: self.max_reference_age_ms,
                };
                serde_json::to_vec(&evidence)?
            }
            BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION
            | BTC_ASYMMETRIC_VALUE_MODEL_EXECUTION_GUARD_VERSION => {
                let evidence = DirectionalModelGuardHashEvidenceV2 {
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
                    signal_id: self.legacy_signal_id,
                    dynamic_fee_rate: self.dynamic_fee_rate,
                    directional_model: self
                        .directional_model
                        .as_ref()
                        .context("directional-model guard evidence is missing")?,
                    binance: &self.binance,
                    selected_book: self
                        .selected_book
                        .as_ref()
                        .context("directional-model guard book evidence is missing")?,
                    max_reference_age_ms: self.max_reference_age_ms,
                    max_directional_feature_age_ms: self
                        .max_directional_feature_age_ms
                        .context("directional-model guard feature age bound is missing")?,
                };
                serde_json::to_vec(&evidence)?
            }
            version => anyhow::bail!("unsupported reference execution guard version {version}"),
        };
        Ok(format!("{:x}", Sha256::digest(bytes)))
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

fn required_book_evidence(
    snapshot: &BtcFeatureSnapshot,
    intent: &ApprovedIntent,
) -> Result<BtcExecutionBookEvidence> {
    let (book, checkpoint_id, connection_id, ingest_sequence) = match intent.outcome {
        BtcOutcome::Up => (
            &snapshot.up_book,
            snapshot.lineage.up_book_checkpoint_id,
            snapshot.lineage.up_book_connection_id,
            snapshot.lineage.up_book_ingest_sequence,
        ),
        BtcOutcome::Down => (
            &snapshot.down_book,
            snapshot.lineage.down_book_checkpoint_id,
            snapshot.lineage.down_book_connection_id,
            snapshot.lineage.down_book_ingest_sequence,
        ),
    };
    ensure!(
        book.token_id == intent.token_id,
        "selected book token does not match the approved intent"
    );
    let evidence = BtcExecutionBookEvidence {
        market_id: snapshot.market_id.clone(),
        token_id: book.token_id.clone(),
        checkpoint_id: checkpoint_id.context("selected book checkpoint ID is missing")?,
        connection_id: connection_id.context("selected book connection ID is missing")?,
        source_timestamp: book
            .source_timestamp
            .context("selected book source timestamp is missing")?,
        received_at: book
            .received_at
            .context("selected book receipt timestamp is missing")?,
        ingest_sequence: ingest_sequence.context("selected book ingest sequence is missing")?,
    };
    ensure!(
        evidence.checkpoint_id != Uuid::nil(),
        "selected book checkpoint ID is nil"
    );
    ensure!(
        evidence.connection_id != Uuid::nil(),
        "selected book connection ID is nil"
    );
    ensure!(
        evidence.ingest_sequence > 0,
        "selected book ingest sequence must be positive"
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

fn bounded_timestamp_age(
    timestamp: DateTime<Utc>,
    checked_at: DateTime<Utc>,
    max_age: Duration,
) -> std::result::Result<Duration, BtcReferenceExecutionRejectReason> {
    if timestamp > checked_at {
        return Err(BtcReferenceExecutionRejectReason::FutureEvidence);
    }
    let age = checked_at - timestamp;
    if age > max_age {
        return Err(BtcReferenceExecutionRejectReason::StaleEvidence);
    }
    Ok(age)
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
            legacy_signal_id: None,
            dynamic_fee_rate: dec!(0.25),
            chainlink_open: Some(tick(5, 1_000)),
            chainlink: Some(tick(6, 100)),
            binance: tick(7, 90),
            directional_model: None,
            selected_book: None,
            max_reference_age_ms: 2_000,
            max_directional_feature_age_ms: None,
            evidence_sha256: String::new(),
        };
        guard.evidence_sha256 = guard.calculate_evidence_sha256().unwrap();
        guard
    }

    fn sealed_directional_model_guard(checked_at: DateTime<Utc>) -> BtcReferenceExecutionGuard {
        let tick = BtcReferenceTickEvidence {
            tick_id: Uuid::from_u128(17),
            source_timestamp: checked_at - Duration::milliseconds(90),
            received_at: checked_at - Duration::milliseconds(80),
            ingest_sequence: 17,
        };
        let mut guard = BtcReferenceExecutionGuard {
            guard_version: BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION.to_string(),
            process_id: Uuid::from_u128(11),
            intent_id: Uuid::from_u128(12),
            decision_id: Uuid::from_u128(13),
            decision_at: checked_at,
            snapshot_id: Uuid::from_u128(14),
            feature_as_of: checked_at,
            market_id: "model-market".to_string(),
            token_id: "model-up".to_string(),
            outcome: BtcOutcome::Up,
            strategy_version: BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION.to_string(),
            feature_schema_version: "btc-5m-directional-core-features-v2".to_string(),
            lineage_version: BTC_FEATURE_LINEAGE_VERSION.to_string(),
            feature_sha256: "a".repeat(64),
            client_order_id: Uuid::from_u128(18),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            limit_price: dec!(0.40),
            size: dec!(2),
            legacy_signal_id: None,
            dynamic_fee_rate: dec!(0.25),
            chainlink_open: None,
            chainlink: None,
            binance: tick,
            directional_model: Some(BtcDirectionalModelExecutionEvidence {
                model_key: "model-v1".to_string(),
                artifact_sha256: "b".repeat(64),
                feature_schema_sha256: "c".repeat(64),
                input_sha256: "d".repeat(64),
                window_start: checked_at - Duration::seconds(180),
                feature_as_of: checked_at,
                seconds_elapsed: 180,
            }),
            selected_book: Some(BtcExecutionBookEvidence {
                market_id: "model-market".to_string(),
                token_id: "model-up".to_string(),
                checkpoint_id: Uuid::from_u128(19),
                connection_id: Uuid::from_u128(20),
                source_timestamp: checked_at - Duration::milliseconds(70),
                received_at: checked_at - Duration::milliseconds(60),
                ingest_sequence: 21,
            }),
            max_reference_age_ms: 2_000,
            max_directional_feature_age_ms: Some(5_000),
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
            metadata,
        }
    }

    fn directional_model_request(guard: &BtcReferenceExecutionGuard) -> OrderRequest {
        let model = guard.directional_model.as_ref().unwrap();
        let mut request = request(guard);
        request.metadata["strategy"] = BTC_DIRECTIONAL_MODEL_STRATEGY_FAMILY.into();
        request.metadata["profile_id"] = model.model_key.clone().into();
        request.metadata["profile_sha256"] = model.artifact_sha256.clone().into();
        request.metadata["prediction"] = serde_json::json!({
            "status": "directional_prediction",
            "outcome": "up",
        });
        guard.insert_into_metadata(&mut request.metadata).unwrap();
        request
    }

    fn asymmetric_value_model_request(guard: &BtcReferenceExecutionGuard) -> OrderRequest {
        let model = guard.directional_model.as_ref().unwrap();
        let mut request = request(guard);
        request.metadata["strategy"] = BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_FAMILY.into();
        request.metadata["profile_id"] = model.model_key.clone().into();
        request.metadata["profile_sha256"] = model.artifact_sha256.clone().into();
        guard.insert_into_metadata(&mut request.metadata).unwrap();
        request
    }

    #[test]
    fn legacy_signal_id_json_preserves_guard_v1_hash() {
        let checked_at = Utc.with_ymd_and_hms(2026, 7, 21, 12, 0, 0).unwrap();
        let guard = sealed_guard(checked_at);
        assert_eq!(
            guard.evidence_sha256,
            "1c4ff39f324097b0ab7f97b956bf40cb7a864c23a04422ed0143c0fa116143e8"
        );

        let mut legacy_guard_json = serde_json::to_value(&guard).unwrap();
        assert_eq!(
            legacy_guard_json.get("signal_id"),
            Some(&serde_json::Value::Null)
        );
        legacy_guard_json["signal_id"] = serde_json::Value::Null;
        let legacy_guard: BtcReferenceExecutionGuard =
            serde_json::from_value(legacy_guard_json).unwrap();
        assert_eq!(
            legacy_guard.calculate_evidence_sha256().unwrap(),
            guard.evidence_sha256
        );
        assert_eq!(
            serde_json::to_value(&legacy_guard)
                .unwrap()
                .get("signal_id"),
            Some(&serde_json::Value::Null)
        );
        assert!(serde_json::to_value(&legacy_guard)
            .unwrap()
            .get("directional_model")
            .is_none());
        assert!(serde_json::to_value(&legacy_guard)
            .unwrap()
            .get("selected_book")
            .is_none());

        let mut legacy_request_json = serde_json::to_value(request(&guard)).unwrap();
        legacy_request_json["signal_id"] = serde_json::Value::Null;
        legacy_request_json["metadata"][BTC_REFERENCE_EXECUTION_GUARD_METADATA_KEY]["signal_id"] =
            serde_json::Value::Null;
        let legacy_request: OrderRequest = serde_json::from_value(legacy_request_json).unwrap();
        assert_eq!(legacy_request.client_order_id, guard.client_order_id);
        assert_eq!(
            reference_execution_guard(&legacy_request).unwrap(),
            legacy_guard
        );
        assert!(legacy_guard
            .validate_for_request(
                &legacy_request,
                checked_at,
                guard.process_id,
                Duration::seconds(2),
                None,
            )
            .is_ok());
        assert!(serde_json::to_value(legacy_request)
            .unwrap()
            .get("signal_id")
            .is_none());

        let mut unsupported_signal_guard = guard.clone();
        unsupported_signal_guard.legacy_signal_id = Some(Uuid::from_u128(99));
        unsupported_signal_guard.reseal_for_test();
        let unsupported_signal_request = request(&unsupported_signal_guard);
        assert_eq!(
            unsupported_signal_guard
                .validate_for_request(
                    &unsupported_signal_request,
                    checked_at,
                    guard.process_id,
                    Duration::seconds(2),
                    None,
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::InvalidGuard
        );
    }

    #[test]
    fn directional_model_guard_validates_without_chainlink_and_seals_model_book_evidence() {
        let checked_at = Utc.with_ymd_and_hms(2026, 7, 27, 12, 0, 0).unwrap();
        let guard = sealed_directional_model_guard(checked_at);
        let guarded_request = directional_model_request(&guard);

        let assessment = guard
            .validate_for_request(
                &guarded_request,
                checked_at,
                guard.process_id,
                Duration::seconds(2),
                Some(Duration::seconds(5)),
            )
            .unwrap();

        let serialized = serde_json::to_value(&guard).unwrap();
        assert!(serialized.get("chainlink_open").is_none());
        assert!(serialized.get("chainlink").is_none());
        assert!(serialized.get("directional_model").is_some());
        assert!(serialized.get("selected_book").is_some());
        assert_eq!(assessment.chainlink_source_age_ms, None);
        assert_eq!(assessment.chainlink_receive_age_ms, None);
        assert_eq!(assessment.directional_model_feature_age_ms, Some(0));
        assert_eq!(assessment.max_directional_feature_age_ms, Some(5_000));
        assert_eq!(assessment.selected_book_source_age_ms, Some(70));
        assert_eq!(assessment.selected_book_receive_age_ms, Some(60));

        let mut wrong_identity = guarded_request.clone();
        wrong_identity.metadata["profile_id"] = "another-model".into();
        assert_eq!(
            guard
                .validate_for_request(
                    &wrong_identity,
                    checked_at,
                    guard.process_id,
                    Duration::seconds(2),
                    Some(Duration::seconds(5)),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::IdentityMismatch
        );

        let mut wrong_outcome = guarded_request.clone();
        wrong_outcome.metadata["prediction"]["outcome"] = "down".into();
        assert_eq!(
            guard
                .validate_for_request(
                    &wrong_outcome,
                    checked_at,
                    guard.process_id,
                    Duration::seconds(2),
                    Some(Duration::seconds(5)),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::IdentityMismatch
        );

        let mut tampered = guard.clone();
        tampered.directional_model.as_mut().unwrap().input_sha256 = "e".repeat(64);
        assert_eq!(
            tampered
                .validate_for_request(
                    &guarded_request,
                    checked_at,
                    guard.process_id,
                    Duration::seconds(2),
                    Some(Duration::seconds(5)),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::EvidenceHashMismatch
        );

        let mut tampered_bound = guard.clone();
        tampered_bound.max_directional_feature_age_ms = Some(6_000);
        assert_eq!(
            tampered_bound
                .validate_for_request(
                    &guarded_request,
                    checked_at,
                    guard.process_id,
                    Duration::seconds(2),
                    Some(Duration::seconds(5)),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::EvidenceHashMismatch
        );
    }

    #[test]
    fn asymmetric_value_model_guard_is_additive_and_requires_no_directional_prediction() {
        let checked_at = Utc.with_ymd_and_hms(2026, 8, 6, 12, 0, 0).unwrap();
        let mut guard = sealed_directional_model_guard(checked_at);
        guard.guard_version = BTC_ASYMMETRIC_VALUE_MODEL_EXECUTION_GUARD_VERSION.to_string();
        guard.strategy_version = BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_VERSION.to_string();
        guard.reseal_for_test();
        let request = asymmetric_value_model_request(&guard);

        guard
            .validate_for_request(
                &request,
                checked_at,
                guard.process_id,
                Duration::seconds(2),
                Some(Duration::seconds(5)),
            )
            .unwrap();

        let mut directional_metadata = request.clone();
        directional_metadata.metadata["prediction"] = serde_json::json!({
            "status": "directional_prediction",
            "outcome": "up",
        });
        assert_eq!(
            guard
                .validate_for_request(
                    &directional_metadata,
                    checked_at,
                    guard.process_id,
                    Duration::seconds(2),
                    Some(Duration::seconds(5)),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::IdentityMismatch
        );
    }

    #[test]
    fn directional_model_guard_uses_separate_model_and_reference_freshness_bounds() {
        let checked_at = Utc.with_ymd_and_hms(2026, 7, 27, 12, 0, 0).unwrap();
        let mut guard = sealed_directional_model_guard(checked_at);
        let model = guard.directional_model.as_mut().unwrap();
        model.feature_as_of = checked_at - Duration::milliseconds(2_900);
        model.window_start = model.feature_as_of - Duration::seconds(model.seconds_elapsed);
        guard.reseal_for_test();
        let guarded_request = directional_model_request(&guard);

        let assessment = guard
            .validate_for_request(
                &guarded_request,
                checked_at,
                guard.process_id,
                Duration::seconds(2),
                Some(Duration::seconds(5)),
            )
            .unwrap();
        assert_eq!(assessment.directional_model_feature_age_ms, Some(2_900));

        let mut stale_model = sealed_directional_model_guard(checked_at);
        let model = stale_model.directional_model.as_mut().unwrap();
        model.feature_as_of = checked_at - Duration::milliseconds(5_001);
        model.window_start = model.feature_as_of - Duration::seconds(model.seconds_elapsed);
        stale_model.reseal_for_test();
        let stale_model_request = directional_model_request(&stale_model);
        assert_eq!(
            stale_model
                .validate_for_request(
                    &stale_model_request,
                    checked_at,
                    stale_model.process_id,
                    Duration::seconds(2),
                    Some(Duration::seconds(5)),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::StaleEvidence
        );

        let mut stale_reference = sealed_directional_model_guard(checked_at);
        stale_reference.binance.source_timestamp = checked_at - Duration::milliseconds(2_001);
        stale_reference.reseal_for_test();
        let stale_reference_request = directional_model_request(&stale_reference);
        assert_eq!(
            stale_reference
                .validate_for_request(
                    &stale_reference_request,
                    checked_at,
                    stale_reference.process_id,
                    Duration::seconds(2),
                    Some(Duration::seconds(5)),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::StaleEvidence
        );

        let mut stale_book = sealed_directional_model_guard(checked_at);
        stale_book.selected_book.as_mut().unwrap().received_at =
            checked_at - Duration::milliseconds(2_001);
        stale_book.reseal_for_test();
        let stale_book_request = directional_model_request(&stale_book);
        assert_eq!(
            stale_book
                .validate_for_request(
                    &stale_book_request,
                    checked_at,
                    stale_book.process_id,
                    Duration::seconds(2),
                    Some(Duration::seconds(5)),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::StaleEvidence
        );
    }

    #[test]
    fn legacy_directional_guard_keeps_the_original_hash_and_freshness_contract() {
        let checked_at = Utc.with_ymd_and_hms(2026, 7, 27, 12, 0, 0).unwrap();
        let mut guard = sealed_directional_model_guard(checked_at);
        guard.guard_version = LEGACY_BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION.to_string();
        guard.max_directional_feature_age_ms = None;
        guard.reseal_for_test();
        let guarded_request = directional_model_request(&guard);

        assert!(guard
            .validate_for_request(
                &guarded_request,
                checked_at,
                guard.process_id,
                Duration::seconds(2),
                Some(Duration::seconds(5)),
            )
            .is_ok());

        let mut stale = guard;
        let model = stale.directional_model.as_mut().unwrap();
        model.feature_as_of = checked_at - Duration::milliseconds(2_001);
        model.window_start = model.feature_as_of - Duration::seconds(model.seconds_elapsed);
        stale.reseal_for_test();
        let stale_request = directional_model_request(&stale);
        assert_eq!(
            stale
                .validate_for_request(
                    &stale_request,
                    checked_at,
                    stale.process_id,
                    Duration::seconds(2),
                    Some(Duration::seconds(5)),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::StaleEvidence
        );
    }

    #[test]
    fn directional_model_strategy_cannot_use_the_legacy_guard_contract() {
        let checked_at = Utc.with_ymd_and_hms(2026, 7, 27, 12, 0, 0).unwrap();
        let mut guard = sealed_directional_model_guard(checked_at);
        guard.guard_version = BTC_REFERENCE_EXECUTION_GUARD_VERSION.to_string();
        guard.chainlink_open = Some(guard.binance.clone());
        guard.chainlink = Some(guard.binance.clone());
        guard.directional_model = None;
        guard.selected_book = None;
        guard.reseal_for_test();
        let guarded_request = request(&guard);

        assert_eq!(
            guard
                .validate_for_request(
                    &guarded_request,
                    checked_at,
                    guard.process_id,
                    Duration::seconds(2),
                    Some(Duration::seconds(5)),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::InvalidGuard
        );
    }

    #[test]
    fn directional_model_guard_rejects_unbounded_elapsed_seconds_without_panicking() {
        let checked_at = Utc.with_ymd_and_hms(2026, 7, 27, 12, 0, 0).unwrap();
        let mut guard = sealed_directional_model_guard(checked_at);
        guard.directional_model.as_mut().unwrap().seconds_elapsed = i64::MAX;
        guard.reseal_for_test();
        let guarded_request = directional_model_request(&guard);

        assert_eq!(
            guard
                .validate_for_request(
                    &guarded_request,
                    checked_at,
                    guard.process_id,
                    Duration::seconds(2),
                    Some(Duration::seconds(5)),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::InvalidGuard
        );

        let mut invalid_freshness = sealed_directional_model_guard(checked_at);
        invalid_freshness.max_reference_age_ms = i64::MIN;
        invalid_freshness.reseal_for_test();
        let invalid_freshness_request = directional_model_request(&invalid_freshness);
        assert_eq!(
            invalid_freshness
                .validate_for_request(
                    &invalid_freshness_request,
                    checked_at,
                    invalid_freshness.process_id,
                    Duration::seconds(2),
                    Some(Duration::seconds(5)),
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::InvalidFreshnessBound
        );
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
                None,
            )
            .is_ok());

        guard.chainlink.as_mut().unwrap().source_timestamp = checked_at - Duration::seconds(2);
        guard.evidence_sha256 = guard.calculate_evidence_sha256().unwrap();
        let guarded_request = request(&guard);
        assert!(guard
            .validate_for_request(
                &guarded_request,
                checked_at,
                guard.process_id,
                Duration::seconds(2),
                None,
            )
            .is_ok());
        assert_eq!(
            guard
                .validate_for_request(
                    &guarded_request,
                    checked_at + Duration::milliseconds(1),
                    guard.process_id,
                    Duration::seconds(2),
                    None,
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::StaleEvidence
        );
    }

    #[test]
    fn guard_rejects_fresh_receipt_with_stale_source_and_future_evidence() {
        let checked_at = Utc.with_ymd_and_hms(2026, 7, 21, 12, 0, 0).unwrap();
        let mut guard = sealed_guard(checked_at);
        guard.chainlink.as_mut().unwrap().source_timestamp =
            checked_at - Duration::milliseconds(2_001);
        guard.chainlink.as_mut().unwrap().received_at = checked_at - Duration::milliseconds(1);
        guard.evidence_sha256 = guard.calculate_evidence_sha256().unwrap();
        let guarded_request = request(&guard);
        assert_eq!(
            guard
                .validate_for_request(
                    &guarded_request,
                    checked_at,
                    guard.process_id,
                    Duration::seconds(2),
                    None,
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::StaleEvidence
        );

        guard.chainlink.as_mut().unwrap().source_timestamp = checked_at + Duration::milliseconds(1);
        guard.feature_as_of = guard.chainlink.as_ref().unwrap().source_timestamp;
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
                    None,
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
                    None,
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
                    None,
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
                    None,
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
                    None,
                )
                .unwrap_err(),
            BtcReferenceExecutionRejectReason::NoncausalEvidence
        );
    }
}
