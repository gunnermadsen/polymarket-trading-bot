use std::{collections::BTreeMap, sync::Arc};

use serde_json::Value;
use sqlx::PgPool;
use thiserror::Error;

use crate::domain::{IngesterProfile, IngesterStrategy, IngesterStrategyKey};

pub trait StrategyFactory: Send + Sync {
    fn key(&self) -> IngesterStrategyKey;
    fn config_schema_version(&self) -> i32;
    fn validate_config(&self, config: &Value) -> Result<(), StrategyFactoryError>;
    fn build(
        &self,
        profile: &IngesterProfile,
        pool: PgPool,
    ) -> Result<Box<dyn IngesterStrategy>, StrategyFactoryError>;
}

#[derive(Clone, Default)]
pub struct StrategyRegistry {
    factories: Arc<BTreeMap<IngesterStrategyKey, Arc<dyn StrategyFactory>>>,
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
        })
    }

    pub fn factory(&self, key: IngesterStrategyKey) -> Option<&Arc<dyn StrategyFactory>> {
        self.factories.get(&key)
    }

    pub fn keys(&self) -> impl Iterator<Item = IngesterStrategyKey> + '_ {
        self.factories.keys().copied()
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
