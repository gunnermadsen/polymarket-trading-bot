pub mod admission;
mod asymmetric_value_features;
pub mod binance_spot_l2;
pub mod directional_external_runtime;
pub mod directional_features;
pub mod directional_model;
pub mod execution_freshness;
pub mod execution_guard;
pub mod execution_lifecycle;
pub mod feed_contract;
pub mod feeds;
pub mod live_execution;
pub mod market;
pub mod paper;
pub mod process_runner;
pub mod repository;
pub mod runtime;
pub mod strategy;
pub mod types;
pub mod unified_model_runtime;

pub use admission::{
    AdmissionDisposition, BtcEntryAdmissionConfig, DailyRealizedPnlCredit,
    DailyRealizedPnlHighWaterMarkConfig, DailyRealizedPnlHighWaterMarkEvaluation,
    DailyRealizedPnlHighWaterMarkState, LossRegimeCandidate, LossRegimeConfidenceFloorConfig,
    LossRegimeConfidenceFloorEvaluation, LossRegimeConfidenceFloorState,
    LossRegimeConfidenceFloorTransition, ProposedEntryExposure, UnsettledEntryExposure,
    DAILY_REALIZED_PNL_HIGH_WATER_MARK_SCHEMA_VERSION, LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION,
};
pub use binance_spot_l2::{
    parse_depth_snapshot, parse_depth_update, BinanceSpotL2ApplyOutcome,
    BinanceSpotL2DepthSnapshot, BinanceSpotL2DepthUpdate, BinanceSpotL2Engine,
    BinanceSpotL2FeatureWindow, BinanceSpotL2Level, BinanceSpotL2Side,
    BINANCE_SPOT_L2_FEATURE_SCHEMA_VERSION,
};
pub use directional_external_runtime::{
    BinanceOpenInterestPoint, ChainlinkMidPoint, ChainlinkRefPricePoint,
    DirectionalExternalSourceStatus, DirectionalExternalState, PolygonOraclePoint,
};
pub use directional_features::{
    build_directional_features, build_directional_features_for_schema,
    build_directional_features_for_schema_with_boundary, directional_feature_names,
    DirectionalFeatureError, DirectionalFeatureTimingReason, DirectionalFeatureVector,
    BTC_DIRECTIONAL_BOUNDARY_FEATURE_COUNT, BTC_DIRECTIONAL_BOUNDARY_FEATURE_NAMES,
    BTC_DIRECTIONAL_BOUNDARY_FEATURE_SCHEMA_VERSION, BTC_DIRECTIONAL_BOUNDARY_FEATURE_SUFFIX_NAMES,
    BTC_DIRECTIONAL_CANDIDATE_CADENCE_SECONDS, BTC_DIRECTIONAL_FEATURE_COUNT,
    BTC_DIRECTIONAL_FEATURE_NAMES, BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION,
    BTC_DIRECTIONAL_FIRST_CANDIDATE_SECOND, BTC_DIRECTIONAL_LAST_CANDIDATE_SECOND,
    BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_COUNT, BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_NAMES,
    BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
    BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SUFFIX_NAMES,
    BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_COUNT, BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_NAMES,
    BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SCHEMA_VERSION,
    BTC_DIRECTIONAL_PATH_PREWINDOW_FEATURE_SUFFIX_NAMES,
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
    BtcDirectionalModelExecutionEvidence, BtcExecutionBookEvidence,
    BtcReferenceExecutionAssessment, BtcReferenceExecutionGuard, BtcReferenceExecutionRejectReason,
    BTC_ASYMMETRIC_VALUE_MODEL_EXECUTION_GUARD_VERSION,
    BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION, BTC_REFERENCE_EXECUTION_GUARD_METADATA_KEY,
    BTC_REFERENCE_EXECUTION_GUARD_VERSION,
};
pub use execution_lifecycle::{
    BtcExecutionLifecycle, BtcExecutionMode, LiveExecutionLifecycle, PaperExecutionLifecycle,
};
pub use feed_contract::{
    validate_feed_requirements, BtcModelFeedId, BtcModelFeedRequirement,
    BTC_MODEL_FEED_CONTRACT_VERSION,
};
pub use feeds::BookRegistry;
pub use live_execution::BtcLiveExecutionAdapter;
pub use market::{
    aligned_window_start, discovery_windows, parse_gamma_btc_interval_event, slug_for_window,
    window_start_from_slug,
};
pub use paper::{
    PaperPreviewConfig, PaperPreviewResult, PaperSettlementCreditResult, PaperVenue,
    PaperVenueConfig, PaperVenueStatus, PAPER_DYNAMIC_FEE_RATE_METADATA_KEY,
};
pub use process_runner::{
    process_runtime_readiness, BtcPaperProcessConfig, BtcPaperProcessRunner, BtcProcessConfig,
    BtcProcessRunner,
};
pub use repository::{
    BtcMarketLabel, BtcOfficialResolutionWatch, BtcPaperSettlementRecord, BtcRepository,
    BtcRunManifest, PersistedOfficialResolution,
};
pub use runtime::{
    runtime_status_from_inputs, BtcPlaybookRuntimeHandle, BtcRuntime, BtcRuntimeConfig,
    BtcRuntimeHandle, BtcRuntimeMetrics, BtcRuntimeStatus, BtcStrategyRunner, NoopStrategyRunner,
    StrategyObservation,
};
pub use strategy::*;
pub use types::*;
