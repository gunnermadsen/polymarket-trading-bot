pub mod admission;
pub mod execution_guard;
pub mod experiment;
pub mod feeds;
pub mod market;
pub mod paper;
pub mod predictive_regime_v2;
pub mod repository;
pub mod runtime;
pub mod strategy;
pub mod types;

pub use admission::{
    AdmissionDisposition, BtcEntryAdmissionConfig, DailyRealizedPnlCredit,
    DailyRealizedPnlHighWaterMarkConfig, DailyRealizedPnlHighWaterMarkEvaluation,
    DailyRealizedPnlHighWaterMarkState, LossRegimeCandidate, LossRegimeConfidenceFloorConfig,
    LossRegimeConfidenceFloorEvaluation, LossRegimeConfidenceFloorState,
    LossRegimeConfidenceFloorTransition, ProposedEntryExposure, ShadowPredictiveRegimeCandidate,
    ShadowPredictiveRegimeCircuitBreakerConfig, ShadowPredictiveRegimeEvaluation,
    ShadowPredictiveRegimeState, ShadowPredictiveRegimeTransition, UnsettledEntryExposure,
    DAILY_REALIZED_PNL_HIGH_WATER_MARK_SCHEMA_VERSION, LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION,
    SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE,
    SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION,
};
pub use execution_guard::{
    BtcReferenceExecutionAssessment, BtcReferenceExecutionGuard, BtcReferenceExecutionRejectReason,
    BTC_REFERENCE_EXECUTION_GUARD_METADATA_KEY, BTC_REFERENCE_EXECUTION_GUARD_VERSION,
};
pub use experiment::{BtcPaperExperimentConfig, BtcPaperExperimentRunner};
pub use feeds::{
    parse_binance_agg_trade, parse_clob_messages, parse_rtds_reference_tick, BookRegistry,
    BookUpdateSide, ClobMessage, PriceChange,
};
pub use market::{
    aligned_window_start, discovery_windows, parse_gamma_btc_interval_event, slug_for_window,
    window_start_from_slug,
};
pub use paper::{
    PaperPreviewConfig, PaperPreviewResult, PaperSettlementCreditResult, PaperVenue,
    PaperVenueConfig, PaperVenueStatus, PAPER_DYNAMIC_FEE_RATE_METADATA_KEY,
};
pub use predictive_regime_v2::{
    ShadowPredictiveRegimeCircuitBreakerConfigSelector,
    ShadowPredictiveRegimeCircuitBreakerV2Config, ShadowPredictiveRegimeV2Candidate,
    ShadowPredictiveRegimeV2CandidateSource, ShadowPredictiveRegimeV2Evaluation,
    ShadowPredictiveRegimeV2State, SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_V2_SCHEMA_VERSION,
};
pub use repository::{
    BtcMarketLabel, BtcOfficialResolutionWatch, BtcPaperSettlementLedgerSummary,
    BtcPaperSettlementRecord, BtcRepository, FeedSession, PersistedOfficialResolution,
};
pub use runtime::{
    runtime_status_from_inputs, BtcHeartbeatConfig, BtcPlaybookRuntimeHandle, BtcRuntime,
    BtcRuntimeConfig, BtcRuntimeHandle, BtcRuntimeMetrics, BtcRuntimeStatus, BtcStrategyRunner,
    NoopStrategyRunner, StrategyObservation,
};
pub use strategy::*;
pub use types::*;
