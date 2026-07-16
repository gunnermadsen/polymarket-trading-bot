use uuid::Uuid;

use crate::models::{OrderRequest, OrderSide};

const ORDER_NAMESPACE: Uuid = Uuid::from_u128(0x7b0d_5a3c_2b87_4d2f_94f2_6d7c52d5c001);

#[derive(Debug, Clone)]
pub struct ClientOrderIdSeed<'a> {
    pub strategy_version: &'a str,
    pub process_id: Option<Uuid>,
    pub source_id: Uuid,
    pub purpose: &'a str,
    pub market_id: &'a str,
    pub token_id: &'a str,
    pub side: OrderSide,
    pub notional_key: &'a str,
}

pub fn deterministic_client_order_id(seed: &ClientOrderIdSeed<'_>) -> Uuid {
    let raw = format!(
        "{}|{}|{}|{}|{}|{}|{}|{}",
        seed.strategy_version,
        seed.process_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "legacy".to_string()),
        seed.source_id,
        seed.purpose,
        seed.market_id,
        seed.token_id,
        serde_name(seed.side),
        seed.notional_key
    );
    Uuid::new_v5(&ORDER_NAMESPACE, raw.as_bytes())
}

pub fn event_hash(raw: &serde_json::Value) -> String {
    let canonical = serde_json::to_string(raw).unwrap_or_else(|_| raw.to_string());
    let digest = <sha2::Sha256 as sha2::Digest>::digest(canonical.as_bytes());
    format!("{digest:x}")
}

pub fn order_request_notional_key(request: &OrderRequest) -> String {
    let notional = request.price * request.size;
    notional.round_dp(4).normalize().to_string()
}

fn serde_name(side: OrderSide) -> &'static str {
    match side {
        OrderSide::Buy => "buy",
        OrderSide::Sell => "sell",
    }
}

#[cfg(test)]
mod tests {
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    use crate::models::{OrderRequest, OrderSide, OrderType};

    use super::*;

    #[test]
    fn deterministic_order_id_is_stable_for_same_seed() {
        let source_id = Uuid::parse_str("11111111-1111-4111-8111-111111111111").unwrap();
        let seed = ClientOrderIdSeed {
            strategy_version: "v1",
            process_id: None,
            source_id,
            purpose: "entry",
            market_id: "m1",
            token_id: "t1",
            side: OrderSide::Buy,
            notional_key: "5",
        };

        assert_eq!(
            deterministic_client_order_id(&seed),
            deterministic_client_order_id(&seed)
        );
    }

    #[test]
    fn order_notional_key_rounds_to_four_decimals() {
        let request = OrderRequest {
            client_order_id: Uuid::new_v4(),
            process_id: None,
            market_id: "m1".to_string(),
            token_id: "t1".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.333333),
            size: dec!(15),
            signal_id: None,
            metadata: serde_json::json!({}),
        };

        assert_eq!(order_request_notional_key(&request), "5");
    }

    #[test]
    fn event_hash_is_stable_for_same_payload() {
        let raw = serde_json::json!({"type":"TRADE","id":"abc","status":"CONFIRMED"});
        assert_eq!(event_hash(&raw), event_hash(&raw));
    }
}
