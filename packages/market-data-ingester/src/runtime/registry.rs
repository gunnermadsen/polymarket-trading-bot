use std::{collections::BTreeMap, sync::Arc};

use serde_json::Value;
use sqlx::PgPool;
use thiserror::Error;

use crate::domain::{
    BackfillWorkerStrategy, DrainWorkerStrategy, IngesterProfile, IngesterStrategyKey,
    RealtimeWorkerStrategy,
};

pub trait StrategyFactory: Send + Sync {
    fn key(&self) -> IngesterStrategyKey;
    fn config_schema_version(&self) -> i32;
    fn validate_config(&self, config: &Value) -> Result<(), StrategyFactoryError>;
    fn build(
        &self,
        profile: &IngesterProfile,
        pool: PgPool,
    ) -> Result<Box<dyn RealtimeWorkerStrategy>, StrategyFactoryError>;
}

#[derive(Clone, Default)]
pub struct StrategyRegistry {
    factories: Arc<BTreeMap<IngesterStrategyKey, Arc<dyn StrategyFactory>>>,
    backfills: Arc<BTreeMap<String, Arc<dyn BackfillWorkerStrategy>>>,
    drains: Arc<BTreeMap<String, Arc<dyn DrainWorkerStrategy>>>,
}

impl StrategyRegistry {
    pub fn from_factories(
        factories: impl IntoIterator<Item = Arc<dyn StrategyFactory>>,
    ) -> Result<Self, StrategyFactoryError> {
        let mut registered = BTreeMap::new();
        for factory in factories {
            let key = factory.key();
            if registered.insert(key, factory).is_some() {
                return Err(StrategyFactoryError::DuplicateRegistration(key));
            }
        }
        Ok(Self {
            factories: Arc::new(registered),
            backfills: Arc::new(BTreeMap::new()),
            drains: Arc::new(BTreeMap::new()),
        })
    }

    pub fn with_drains(
        mut self,
        strategies: impl IntoIterator<Item = Arc<dyn DrainWorkerStrategy>>,
    ) -> Result<Self, StrategyFactoryError> {
        let mut registered = BTreeMap::new();
        for strategy in strategies {
            let descriptor = strategy.descriptor();
            if descriptor.strategy_key.trim().is_empty() || descriptor.contract_version <= 0 {
                return Err(StrategyFactoryError::InvalidDrainDescriptor(
                    "drain strategy key must be non-empty and contract version positive".into(),
                ));
            }
            let key = descriptor.strategy_key.to_string();
            if registered.insert(key.clone(), strategy).is_some() {
                return Err(StrategyFactoryError::DuplicateDrainRegistration(key));
            }
        }
        self.drains = Arc::new(registered);
        Ok(self)
    }

    pub fn with_backfills(
        mut self,
        strategies: impl IntoIterator<Item = Arc<dyn BackfillWorkerStrategy>>,
    ) -> Result<Self, StrategyFactoryError> {
        let mut registered = BTreeMap::new();
        for strategy in strategies {
            strategy
                .descriptor()
                .validate()
                .map_err(|error| StrategyFactoryError::InvalidDescriptor(error.to_string()))?;
            let key = strategy.descriptor().strategy_key.to_string();
            if registered.insert(key.clone(), strategy).is_some() {
                return Err(StrategyFactoryError::DuplicateBackfillRegistration(key));
            }
        }
        self.backfills = Arc::new(registered);
        Ok(self)
    }

    pub fn factory(&self, key: IngesterStrategyKey) -> Option<&Arc<dyn StrategyFactory>> {
        self.factories.get(&key)
    }

    pub fn keys(&self) -> impl Iterator<Item = IngesterStrategyKey> + '_ {
        self.factories.keys().copied()
    }

    pub fn backfill(&self, key: &str) -> Option<&Arc<dyn BackfillWorkerStrategy>> {
        self.backfills.get(key)
    }

    pub fn backfills(&self) -> impl Iterator<Item = &Arc<dyn BackfillWorkerStrategy>> {
        self.backfills.values()
    }

    pub fn drain(&self, key: &str) -> Option<&Arc<dyn DrainWorkerStrategy>> {
        self.drains.get(key)
    }

    pub fn drains(&self) -> impl Iterator<Item = &Arc<dyn DrainWorkerStrategy>> {
        self.drains.values()
    }

    pub fn validate_profile(&self, profile: &IngesterProfile) -> Result<(), StrategyFactoryError> {
        let factory = self
            .factory(profile.strategy_key)
            .ok_or(StrategyFactoryError::Unsupported(profile.strategy_key))?;
        if factory.config_schema_version() != profile.config_schema_version {
            return Err(StrategyFactoryError::SchemaVersion {
                key: profile.strategy_key,
                expected: factory.config_schema_version(),
                actual: profile.config_schema_version,
            });
        }
        factory.validate_config(&profile.config)
    }
}

#[derive(Debug, Error)]
pub enum StrategyFactoryError {
    #[error("strategy {0} is registered more than once")]
    DuplicateRegistration(IngesterStrategyKey),
    #[error("backfill strategy {0} is registered more than once")]
    DuplicateBackfillRegistration(String),
    #[error("drain strategy {0} is registered more than once")]
    DuplicateDrainRegistration(String),
    #[error("invalid strategy descriptor: {0}")]
    InvalidDescriptor(String),
    #[error("invalid drain strategy descriptor: {0}")]
    InvalidDrainDescriptor(String),
    #[error("strategy {0} is not compiled into this service")]
    Unsupported(IngesterStrategyKey),
    #[error("strategy {key} config schema mismatch: expected {expected}, received {actual}")]
    SchemaVersion {
        key: IngesterStrategyKey,
        expected: i32,
        actual: i32,
    },
    #[error("invalid strategy configuration: {0}")]
    InvalidConfiguration(String),
    #[error("failed to construct strategy: {0}")]
    Construction(String),
}
