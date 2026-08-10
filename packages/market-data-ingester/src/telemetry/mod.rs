use anyhow::{anyhow, Result};
use tracing_subscriber::EnvFilter;

pub fn initialize() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .json()
        .try_init()
        .map_err(|error| anyhow!(error.to_string()))?;
    Ok(())
}
