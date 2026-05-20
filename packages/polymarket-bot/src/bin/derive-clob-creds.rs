use std::{env, fs, path::Path, str::FromStr};

use anyhow::{bail, Context, Result};
use polymarket_client_sdk_v2::{
    auth::{ExposeSecret, LocalSigner, Signer},
    clob::{Client, Config},
    POLYGON,
};

#[tokio::main]
async fn main() -> Result<()> {
    load_env_file(".env")?;

    let private_key = required_env("POLYMARKET_PRIVATE_KEY")?;
    let host = env::var("POLYMARKET_LIVE_CLOB_BASE_URL")
        .or_else(|_| env::var("POLYMARKET_CLOB_BASE_URL"))
        .unwrap_or_else(|_| "https://clob-v2.polymarket.com".to_string());

    let signer = LocalSigner::from_str(&private_key)
        .context("failed to parse POLYMARKET_PRIVATE_KEY")?
        .with_chain_id(Some(POLYGON));
    let client = Client::new(&host, Config::default())
        .with_context(|| format!("failed to create CLOB client for {host}"))?;
    let creds = client
        .create_or_derive_api_key(&signer, None)
        .await
        .context("failed to create or derive CLOB API credentials")?;

    println!("POLYMARKET_CLOB_API_KEY={}", creds.key());
    println!("POLYMARKET_CLOB_SECRET={}", creds.secret().expose_secret());
    println!(
        "POLYMARKET_CLOB_PASSPHRASE={}",
        creds.passphrase().expose_secret()
    );
    Ok(())
}

fn load_env_file(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(());
    }
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() || env::var_os(key).is_some() {
            continue;
        }
        env::set_var(key, unquote(value.trim()));
    }
    Ok(())
}

fn required_env(key: &str) -> Result<String> {
    let value = env::var(key).with_context(|| format!("{key} is required"))?;
    if value.trim().is_empty() {
        bail!("{key} is empty");
    }
    Ok(value)
}

fn unquote(value: &str) -> String {
    if value.len() >= 2 {
        let bytes = value.as_bytes();
        if (bytes[0] == b'"' && bytes[value.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[value.len() - 1] == b'\'')
        {
            return value[1..value.len() - 1].to_string();
        }
    }
    value.to_string()
}
