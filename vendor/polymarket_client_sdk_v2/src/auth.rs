// Re-exported types for public API convenience
/// The [`Signer`] trait from alloy for signing operations.
/// Implement this trait or use provided signers like [`LocalSigner`] or AWS KMS signers.
pub use alloy::signers::Signer;
/// Local wallet signer for signing with a private key.
/// This is the most common signer implementation.
pub use alloy::signers::local::LocalSigner;
use async_trait::async_trait;
#[cfg(any(feature = "clob", test))]
use base64::Engine as _;
#[cfg(any(feature = "clob", test))]
use base64::engine::general_purpose::URL_SAFE;
#[cfg(any(feature = "clob", test))]
use hmac::{Hmac, Mac as _};
#[cfg(any(feature = "clob", test))]
use reqwest::Body;
use reqwest::Request;
use reqwest::header::HeaderMap;
/// Secret string types that redact values in debug output for security.
pub use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
#[cfg(any(feature = "clob", test))]
use sha2::Sha256;
/// UUID type used for API keys and identifiers.
pub use uuid::Uuid;

use crate::{Result, Timestamp};

/// Type alias for API keys, which are UUIDs.
pub type ApiKey = Uuid;

/// Generic set of credentials used to authenticate to the Polymarket API. These credentials are
/// returned when calling [`crate::clob::Client::create_or_derive_api_key`], [`crate::clob::Client::derive_api_key`], or
/// [`crate::clob::Client::create_api_key`]. They are used by the [`state::Authenticated`] client to
/// sign the [`Request`] when making calls to the API.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Credentials {
    #[serde(alias = "apiKey")]
    pub(crate) key: ApiKey,
    pub(crate) secret: SecretString,
    pub(crate) passphrase: SecretString,
}

impl Credentials {
    #[must_use]
    pub fn new(key: Uuid, secret: String, passphrase: String) -> Self {
        Self {
            key,
            secret: SecretString::from(secret),
            passphrase: SecretString::from(passphrase),
        }
    }

    /// Returns the API key.
    #[must_use]
    pub fn key(&self) -> ApiKey {
        self.key
    }

    /// Returns the secret.
    #[must_use]
    pub fn secret(&self) -> &SecretString {
        &self.secret
    }

    /// Returns the passphrase.
    #[must_use]
    pub fn passphrase(&self) -> &SecretString {
        &self.passphrase
    }
}

/// Each client can exist in one state at a time, i.e. [`state::Unauthenticated`] or
/// [`state::Authenticated`].
pub mod state {
    use crate::auth::{Credentials, Kind};
    use crate::types::Address;

    /// The initial state of the client
    #[non_exhaustive]
    #[derive(Clone, Debug)]
    pub struct Unauthenticated;

    /// The elevated state of the client. For example, calling [`crate::clob::Client::authentication_builder`]
    /// will return an [`crate::clob::client::AuthenticationBuilder`], which can be turned into
    /// an authenticated clob via [`crate::clob::client::AuthenticationBuilder::authenticate`].
    ///
    /// See `examples/authenticated.rs` for more context.
    #[non_exhaustive]
    #[derive(Clone, Debug)]
    #[cfg_attr(
        not(feature = "clob"),
        expect(dead_code, reason = "Fields used by clob module when feature enabled")
    )]
    pub struct Authenticated<K: Kind> {
        /// The signer's address that created the credentials
        pub(crate) address: Address,
        /// The [`Credentials`]'s `secret` is used to generate an [`crate::signer::hmac`] which is
        /// passed in the L2 headers ([`super::HeaderMap`]) `POLY_SIGNATURE` field.
        pub(crate) credentials: Credentials,
        /// The [`Kind`] that this [`Authenticated`] exhibits. Used to generate additional headers
        /// for different types of authentication, e.g. Builder.
        pub(crate) kind: K,
    }

    /// The clob state can only be [`Unauthenticated`] or [`Authenticated`].
    pub trait State: sealed::Sealed {}

    impl State for Unauthenticated {}
    impl sealed::Sealed for Unauthenticated {}

    impl<K: Kind> State for Authenticated<K> {}
    impl<K: Kind> sealed::Sealed for Authenticated<K> {}

    mod sealed {
        pub trait Sealed {}
    }
}

/// Asynchronous authentication enricher. Reserved for future L2-header extensions;
/// in V2 every authenticated client uses [`Normal`] and adds no headers beyond L2.
#[async_trait]
pub trait Kind: sealed::Sealed + Clone + Send + Sync + 'static {
    async fn extra_headers(&self, request: &Request, timestamp: Timestamp) -> Result<HeaderMap>;
}

/// Non-special, generic authentication. Sometimes referred to as L2 authentication.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Normal;

#[async_trait]
impl Kind for Normal {
    async fn extra_headers(&self, _request: &Request, _timestamp: Timestamp) -> Result<HeaderMap> {
        Ok(HeaderMap::new())
    }
}

impl sealed::Sealed for Normal {}

mod sealed {
    pub trait Sealed {}
}

#[cfg(feature = "clob")]
pub(crate) mod l1 {
    use std::borrow::Cow;

    use alloy::core::sol;
    use alloy::dyn_abi::Eip712Domain;
    use alloy::hex::ToHexExt as _;
    use alloy::primitives::{Address, B256, ChainId, Signature, U256, keccak256};
    use alloy::signers::Signer;
    use alloy::sol_types::SolStruct as _;
    use alloy::sol_types::SolValue as _;
    use reqwest::header::HeaderMap;

    use crate::{Result, Timestamp};

    pub(crate) const POLY_ADDRESS: &str = "POLY_ADDRESS";
    pub(crate) const POLY_NONCE: &str = "POLY_NONCE";
    pub(crate) const POLY_SIGNATURE: &str = "POLY_SIGNATURE";
    pub(crate) const POLY_TIMESTAMP: &str = "POLY_TIMESTAMP";
    const DEPOSIT_WALLET_NAME: &str = "DepositWallet";
    const DEPOSIT_WALLET_VERSION: &str = "1";
    const CLOB_AUTH_TYPE_STRING: &str =
        "ClobAuth(address address,string timestamp,uint256 nonce,string message)";
    const SOLADY_CLOB_AUTH_TYPE_STRING: &str = concat!(
        "TypedDataSign(ClobAuth contents,string name,string version,uint256 chainId,",
        "address verifyingContract,bytes32 salt)",
        "ClobAuth(address address,string timestamp,uint256 nonce,string message)"
    );

    sol! {
        #[non_exhaustive]
        struct ClobAuth {
            address address;
            string  timestamp;
            uint256 nonce;
            string  message;
        }
    }

    fn push_hex(out: &mut String, bytes: &[u8]) {
        const LUT: &[u8; 16] = b"0123456789abcdef";
        out.reserve(bytes.len() * 2);
        for byte in bytes {
            out.push(LUT[(byte >> 4) as usize] as char);
            out.push(LUT[(byte & 0x0f) as usize] as char);
        }
    }

    fn signature_hex_no_prefix(signature: &Signature) -> String {
        let signature = signature.to_string();
        signature
            .strip_prefix("0x")
            .unwrap_or(&signature)
            .to_owned()
    }

    async fn sign_poly1271_clob_auth<S: Signer>(
        signer: &S,
        auth: &ClobAuth,
        app_domain: &Eip712Domain,
        chain_id: ChainId,
        deposit_wallet: Address,
    ) -> Result<String> {
        let contents_hash = auth.eip712_hash_struct();
        let app_domain_separator = app_domain.hash_struct();
        let typed_data_sign_struct_hash = keccak256(
            (
                keccak256(SOLADY_CLOB_AUTH_TYPE_STRING.as_bytes()),
                contents_hash,
                keccak256(DEPOSIT_WALLET_NAME.as_bytes()),
                keccak256(DEPOSIT_WALLET_VERSION.as_bytes()),
                U256::from(chain_id),
                deposit_wallet,
                B256::ZERO,
            )
                .abi_encode(),
        );

        let mut digest_input = [0_u8; 66];
        digest_input[0] = 0x19;
        digest_input[1] = 0x01;
        digest_input[2..34].copy_from_slice(app_domain_separator.as_slice());
        digest_input[34..66].copy_from_slice(typed_data_sign_struct_hash.as_slice());
        let digest = keccak256(digest_input);

        let inner_signature = signer.sign_hash(&digest).await?;
        let mut wrapped =
            String::with_capacity(2 + 130 + 64 + 64 + (CLOB_AUTH_TYPE_STRING.len() * 2) + 4);
        wrapped.push_str("0x");
        wrapped.push_str(&signature_hex_no_prefix(&inner_signature));
        push_hex(&mut wrapped, app_domain_separator.as_slice());
        push_hex(&mut wrapped, contents_hash.as_slice());
        push_hex(&mut wrapped, CLOB_AUTH_TYPE_STRING.as_bytes());
        let contents_type_len = u16::try_from(CLOB_AUTH_TYPE_STRING.len())
            .expect("ClobAuth type string length fits in u16");
        push_hex(&mut wrapped, &contents_type_len.to_be_bytes());

        Ok(wrapped)
    }

    /// Returns the [`HeaderMap`] needed to obtain [`Credentials`] .
    pub(crate) async fn create_headers<S: Signer>(
        signer: &S,
        chain_id: ChainId,
        timestamp: Timestamp,
        nonce: Option<u32>,
        poly1271_funder: Option<Address>,
    ) -> Result<HeaderMap> {
        let naive_nonce = nonce.unwrap_or(0);
        let auth_address = poly1271_funder.unwrap_or_else(|| signer.address());

        let auth = ClobAuth {
            address: auth_address,
            timestamp: timestamp.to_string(),
            nonce: U256::from(naive_nonce),
            message: "This message attests that I control the given wallet".to_owned(),
        };

        let domain = Eip712Domain {
            name: Some(Cow::Borrowed("ClobAuthDomain")),
            version: Some(Cow::Borrowed("1")),
            chain_id: Some(U256::from(chain_id)),
            ..Eip712Domain::default()
        };

        let signature = if let Some(deposit_wallet) = poly1271_funder {
            sign_poly1271_clob_auth(signer, &auth, &domain, chain_id, deposit_wallet).await?
        } else {
            let hash = auth.eip712_signing_hash(&domain);
            signer.sign_hash(&hash).await?.to_string()
        };

        let mut map = HeaderMap::new();
        map.insert(POLY_ADDRESS, auth_address.encode_hex_with_prefix().parse()?);
        map.insert(POLY_NONCE, naive_nonce.to_string().parse()?);
        map.insert(POLY_SIGNATURE, signature.parse()?);
        map.insert(POLY_TIMESTAMP, timestamp.to_string().parse()?);

        Ok(map)
    }
}

#[cfg(feature = "clob")]
pub(crate) mod l2 {
    use reqwest::Request;
    use reqwest::header::HeaderMap;
    use secrecy::ExposeSecret as _;

    use crate::auth::state::Authenticated;
    use crate::auth::{Kind, hmac, to_message};
    use crate::{Result, Timestamp};

    pub(crate) const POLY_ADDRESS: &str = "POLY_ADDRESS";
    pub(crate) const POLY_API_KEY: &str = "POLY_API_KEY";
    pub(crate) const POLY_PASSPHRASE: &str = "POLY_PASSPHRASE";
    pub(crate) const POLY_SIGNATURE: &str = "POLY_SIGNATURE";
    pub(crate) const POLY_TIMESTAMP: &str = "POLY_TIMESTAMP";

    /// Returns the [`Headers`] needed to interact with any authenticated endpoints.
    pub(crate) async fn create_headers<K: Kind>(
        state: &Authenticated<K>,
        request: &Request,
        timestamp: Timestamp,
    ) -> Result<HeaderMap> {
        let credentials = &state.credentials;
        let signature = hmac(&credentials.secret, &to_message(request, timestamp))?;

        let mut map = HeaderMap::new();

        map.insert(POLY_ADDRESS, state.address.to_checksum(None).parse()?);
        map.insert(POLY_API_KEY, state.credentials.key.to_string().parse()?);
        map.insert(
            POLY_PASSPHRASE,
            state.credentials.passphrase.expose_secret().parse()?,
        );
        map.insert(POLY_SIGNATURE, signature.parse()?);
        map.insert(POLY_TIMESTAMP, timestamp.to_string().parse()?);

        let extra_headers = state.kind.extra_headers(request, timestamp).await?;

        map.extend(extra_headers);

        Ok(map)
    }
}

// These helpers back `l2::create_headers` and are covered by tests in this module.
// Gated so the `lib` target without `clob` doesn't flag them as dead code.
#[cfg(any(feature = "clob", test))]
#[must_use]
fn to_message(request: &Request, timestamp: Timestamp) -> String {
    let method = request.method();
    let body = request.body().and_then(body_to_string).unwrap_or_default();
    let path = request.url().path();

    format!("{timestamp}{method}{path}{body}")
}

#[cfg(any(feature = "clob", test))]
#[must_use]
fn body_to_string(body: &Body) -> Option<String> {
    body.as_bytes()
        .map(String::from_utf8_lossy)
        .map(|b| b.replace('\'', "\""))
}

#[cfg(any(feature = "clob", test))]
fn hmac(secret: &SecretString, message: &str) -> Result<String> {
    let decoded_secret = URL_SAFE.decode(secret.expose_secret())?;
    let mut mac = Hmac::<Sha256>::new_from_slice(&decoded_secret)?;
    mac.update(message.as_bytes());

    let result = mac.finalize().into_bytes();
    Ok(URL_SAFE.encode(result))
}

#[cfg(test)]
mod tests {
    use std::str::FromStr as _;

    #[cfg(feature = "clob")]
    use alloy::signers::local::LocalSigner;
    use reqwest::{Client, Method, RequestBuilder};
    use serde_json::json;
    use url::Url;
    use uuid::Uuid;

    use super::*;
    #[cfg(feature = "clob")]
    use crate::auth::state::Authenticated;
    #[cfg(feature = "clob")]
    use crate::types::address;
    #[cfg(feature = "clob")]
    use crate::{AMOY, Result};

    // publicly known private key
    #[cfg(feature = "clob")]
    const PRIVATE_KEY: &str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";

    #[cfg(feature = "clob")]
    #[tokio::test]
    async fn l1_headers_should_succeed() -> anyhow::Result<()> {
        let signer = LocalSigner::from_str(PRIVATE_KEY)?.with_chain_id(Some(AMOY));

        let headers = l1::create_headers(&signer, AMOY, 10_000_000, Some(23), None).await?;

        assert_eq!(
            signer.address(),
            address!("0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266")
        );
        assert_eq!(
            headers[l1::POLY_ADDRESS],
            "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266"
        );
        assert_eq!(headers[l1::POLY_NONCE], "23");
        assert_eq!(
            headers[l1::POLY_SIGNATURE],
            "0xf62319a987514da40e57e2f4d7529f7bac38f0355bd88bb5adbb3768d80de6c1682518e0af677d5260366425f4361e7b70c25ae232aff0ab2331e2b164a1aedc1b"
        );
        assert_eq!(headers[l1::POLY_TIMESTAMP], "10000000");

        Ok(())
    }

    #[cfg(feature = "clob")]
    #[tokio::test]
    async fn l2_headers_should_succeed() -> anyhow::Result<()> {
        let signer = LocalSigner::from_str(PRIVATE_KEY)?;

        let authenticated = Authenticated {
            address: signer.address(),
            credentials: Credentials {
                key: Uuid::nil(),
                passphrase: SecretString::from(
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
                ),
                secret: SecretString::from(
                    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=".to_owned(),
                ),
            },
            kind: Normal,
        };

        let request = Request::new(Method::GET, Url::parse("http://localhost/")?);
        let headers = l2::create_headers(&authenticated, &request, 1).await?;

        assert_eq!(
            headers[l2::POLY_ADDRESS],
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
        );
        assert_eq!(
            headers[l2::POLY_PASSPHRASE],
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
        assert_eq!(headers[l2::POLY_API_KEY], Uuid::nil().to_string());
        assert_eq!(
            headers[l2::POLY_SIGNATURE],
            "eHaylCwqRSOa2LFD77Nt_SaTpbsxzN8eTEI3LryhEj4="
        );
        assert_eq!(headers[l2::POLY_TIMESTAMP], "1");

        Ok(())
    }

    #[test]
    fn request_args_should_succeed() -> Result<()> {
        let request = Request::new(Method::POST, Url::parse("http://localhost/path")?);
        let request = RequestBuilder::from_parts(Client::new(), request)
            .json(&json!({"foo": "bar"}))
            .build()?;

        let timestamp = 1;

        assert_eq!(
            to_message(&request, timestamp),
            r#"1POST/path{"foo":"bar"}"#
        );

        Ok(())
    }

    #[test]
    fn hmac_succeeds() -> Result<()> {
        let json = json!({
            "hash": "0x123"
        });

        let method = Method::from_str("test-sign")
            .expect("To avoid needing an error variant just for one test");
        let request = Request::new(method, Url::parse("http://localhost/orders")?);
        let request = RequestBuilder::from_parts(Client::new(), request)
            .json(&json)
            .build()?;

        let message = to_message(&request, 1_000_000);
        let signature = hmac(
            &SecretString::from("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=".to_owned()),
            &message,
        )?;

        assert_eq!(message, r#"1000000test-sign/orders{"hash":"0x123"}"#);
        assert_eq!(signature, "4gJVbox-R6XlDK4nlaicig0_ANVL1qdcahiL8CXfXLM=");

        Ok(())
    }

    #[test]
    fn credentials_key_returns_api_key() {
        let key = Uuid::new_v4();
        let credentials = Credentials::new(key, "secret".to_owned(), "passphrase".to_owned());
        assert_eq!(credentials.key(), key);
    }

    #[test]
    fn debug_does_not_expose_secrets() {
        let secret_value = "my_super_secret_value_12345";
        let passphrase_value = "my_super_secret_passphrase_67890";
        let credentials = Credentials::new(
            Uuid::nil(),
            secret_value.to_owned(),
            passphrase_value.to_owned(),
        );

        let debug_output = format!("{credentials:?}");

        // Verify that the secret values are NOT present in the debug output
        assert!(
            !debug_output.contains(secret_value),
            "Debug output should NOT contain the secret value. Got: {debug_output}"
        );
        assert!(
            !debug_output.contains(passphrase_value),
            "Debug output should NOT contain the passphrase value. Got: {debug_output}"
        );
    }
}
