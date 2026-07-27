pub mod admission;
pub mod directional_features;
pub mod directional_model;
pub mod execution_guard;
pub mod feeds;
pub mod market;
pub mod paper;
pub mod predictive_regime_v2;
pub mod process_runner;
pub mod reliability_calibration;
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
pub use directional_features::{
    build_directional_features, DirectionalFeatureError, DirectionalFeatureTimingReason,
    DirectionalFeatureVector, BTC_DIRECTIONAL_CANDIDATE_CADENCE_SECONDS,
    BTC_DIRECTIONAL_FEATURE_COUNT, BTC_DIRECTIONAL_FEATURE_NAMES,
    BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION, BTC_DIRECTIONAL_FIRST_CANDIDATE_SECOND,
    BTC_DIRECTIONAL_LAST_CANDIDATE_SECOND,
};
pub use directional_model::{
    runtime_model, BtcDirectionalModelFeatureSnapshot, RuntimeDirectionalModel, RuntimeModelAction,
    RuntimeModelRegistry, RuntimeModelScore, RuntimeModelSelection, RuntimePredictionPolicy,
    BTC_DIRECTIONAL_MODEL_DIR_ENV, BTC_DIRECTIONAL_MODEL_FAMILY,
    BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION, BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION,
    BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256, BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256,
    BTC_DIRECTIONAL_MODEL_V1_KEY, DEFAULT_BTC_DIRECTIONAL_MODEL_DIR,
};
pub use execution_guard::{
    BtcReferenceExecutionAssessment, BtcReferenceExecutionGuard, BtcReferenceExecutionRejectReason,
    BTC_REFERENCE_EXECUTION_GUARD_METADATA_KEY, BTC_REFERENCE_EXECUTION_GUARD_VERSION,
};
pub use feeds::{
    parse_binance_agg_trade, parse_binance_agg_trade_with_details, parse_binance_aggregate_trade,
    parse_clob_messages, parse_rtds_reference_tick, BookRegistry, BookUpdateSide, ClobMessage,
    PriceChange,
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
pub use process_runner::{BtcPaperProcessConfig, BtcPaperProcessRunner};
pub use repository::{
    BtcMarketLabel, BtcOfficialResolutionWatch, BtcPaperSettlementRecord, BtcRepository,
    BtcRunManifest, FeedSession, PersistedOfficialResolution,
};
pub use runtime::{
    runtime_status_from_inputs, BtcHeartbeatConfig, BtcPlaybookRuntimeHandle, BtcRuntime,
    BtcRuntimeConfig, BtcRuntimeHandle, BtcRuntimeMetrics, BtcRuntimeStatus, BtcStrategyRunner,
    NoopStrategyRunner, StrategyObservation,
};
pub use strategy::*;
pub use types::*;
