use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context, Result};
use chrono::{TimeZone, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    data_api::{ActivityQuery, DataApiClient, PositionsQuery},
    execution::{live::LiveVenueEvent, LIVE_EXTERNAL_EVENT_CLOCK_SKEW},
    idempotency::event_hash,
    models::{DataApiActivity, DataApiPosition},
    store::{
        AccountLiveFillEvidence, AccountPositionSnapshot, AccountTrade, LiveRedemptionEvidence,
        Store,
    },
};

const DATA_API_RECONCILIATION_PAGE_SIZE: usize = 500;
const MAX_DATA_API_RECONCILIATION_ROWS: usize = 4_000;
const MAX_DATA_API_RECONCILIATION_REQUESTS: usize =
    MAX_DATA_API_RECONCILIATION_ROWS / DATA_API_RECONCILIATION_PAGE_SIZE + 1;
fn data_api_position_size_tolerance() -> Decimal {
    Decimal::new(1, 6)
}

fn data_api_trade_price_tolerance() -> Decimal {
    Decimal::new(1, 8)
}

fn data_api_trade_size_tolerance() -> Decimal {
    Decimal::new(1, 6)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountReconcileRequest {
    pub account_address: Option<String>,
    pub lookback_hours: Option<i64>,
    #[serde(default)]
    pub process_id: Option<Uuid>,
    #[serde(default)]
    pub account_ref: Option<String>,
    #[serde(default)]
    pub credential_account_fingerprint_sha256: Option<String>,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub token_id: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountPositionMismatch {
    pub token_id: String,
    pub db_open_size: Decimal,
    pub account_size: Decimal,
    pub delta_size: Decimal,
    pub mismatch_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProcessAccountingProof {
    pub status: String,
    pub position_ownership: String,
    pub realized_pnl: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountReconcileReport {
    pub process_id: Option<Uuid>,
    pub account_ref: Option<String>,
    pub credential_account_fingerprint_sha256: Option<String>,
    pub process_accounting_proven: bool,
    pub process_accounting_status: String,
    pub process_accounting_proof: ProcessAccountingProof,
    pub account_address: String,
    pub source: String,
    pub dry_run: bool,
    pub token_id: Option<String>,
    pub lookback_hours: i64,
    pub activities_fetched: usize,
    pub account_trades_detected: usize,
    pub account_trades_inserted: u64,
    pub position_snapshots_detected: usize,
    pub position_snapshots_inserted: u64,
    pub exits_detected: u64,
    pub exits_applied: u64,
    pub exit_size_applied: Decimal,
    pub position_adjustments_detected: u64,
    pub position_adjustments_applied: u64,
    pub position_adjustment_size_applied: Decimal,
    pub mismatches: Vec<AccountPositionMismatch>,
    pub unmatched_trades: u64,
}

pub async fn reconcile_account_positions(
    store: &Store,
    data_api: &DataApiClient,
    request: AccountReconcileRequest,
) -> Result<AccountReconcileReport> {
    let process_id = request.process_id;
    if process_id.is_some_and(|process_id| process_id.is_nil()) {
        bail!("account reconciliation process_id must not be nil");
    }
    let account_ref = normalize_account_ref(request.account_ref.as_deref())?;
    if process_id.is_some() != account_ref.is_some() {
        bail!("process-scoped account reconciliation requires process_id and account_ref together");
    }
    if process_id.is_some() && request.token_id.is_some() {
        bail!("process-scoped account reconciliation cannot use a token filter");
    }
    let credential_account_fingerprint_sha256 = request
        .credential_account_fingerprint_sha256
        .as_deref()
        .map(str::trim)
        .map(str::to_string);
    if process_id.is_some() != credential_account_fingerprint_sha256.is_some() {
        bail!("process-scoped account reconciliation requires a credential/account fingerprint");
    }
    if credential_account_fingerprint_sha256
        .as_deref()
        .is_some_and(|fingerprint| !is_sha256_hex(fingerprint))
    {
        bail!("account reconciliation credential/account fingerprint must be SHA-256 hex");
    }

    let account_address = request
        .account_address
        .as_deref()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let lookback_hours = request.lookback_hours.unwrap_or(24).clamp(1, 24 * 30);
    let source = request
        .source
        .as_deref()
        .unwrap_or(if request.dry_run { "admin" } else { "poll" })
        .to_string();
    let trade_source = match source.as_str() {
        "manual_backfill" => "manual_backfill",
        "poll" => "poll",
        _ => "data_api",
    };
    // Freeze the activity window before starting offset pagination. Without an upper bound, a
    // newly inserted account event can shift every later page and make a clean reconciliation
    // silently skip or duplicate evidence.
    let activity_window_end = Utc::now();
    let activity_window_start = activity_window_end - chrono::Duration::hours(lookback_hours);
    let start = activity_window_start.timestamp();
    let mut activity_query = ActivityQuery::for_user(account_address.clone());
    activity_query.limit = Some(DATA_API_RECONCILIATION_PAGE_SIZE);
    activity_query.start = Some(start);
    activity_query.end = Some(activity_window_end.timestamp());
    activity_query.sort_by = Some("timestamp".to_string());
    activity_query.sort_direction = Some("desc".to_string());
    activity_query.activity_types = vec!["TRADE".to_string()];
    let activities = fetch_bounded_activity(data_api, activity_query).await?;
    let redemption_activities = if let Some(process_id) = process_id {
        let oldest_unrecognized_fill = store.oldest_unrecognized_live_fill_at(process_id).await?;
        if let Some(oldest_unrecognized_fill) = oldest_unrecognized_fill {
            let maximum_backfill_start = activity_window_end - chrono::Duration::days(30);
            if oldest_unrecognized_fill < maximum_backfill_start {
                bail!(
                    "live redemption discovery exceeds the bounded 30-day process evidence window"
                );
            }
            let redemption_start = (oldest_unrecognized_fill - chrono::Duration::minutes(5))
                .max(maximum_backfill_start)
                .timestamp();
            let mut redemption_query = ActivityQuery::for_user(account_address.clone());
            redemption_query.limit = Some(DATA_API_RECONCILIATION_PAGE_SIZE);
            redemption_query.start = Some(redemption_start);
            redemption_query.end = Some(activity_window_end.timestamp());
            redemption_query.sort_by = Some("timestamp".to_string());
            redemption_query.sort_direction = Some("desc".to_string());
            redemption_query.activity_types = vec!["REDEEM".to_string()];
            fetch_bounded_activity(data_api, redemption_query).await?
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    let external_trade_activities = activities
        .iter()
        .filter(|activity| is_trade_activity(activity))
        .count();
    let mut trades = activities
        .iter()
        .filter(|activity| is_trade_activity(activity))
        .filter_map(|activity| {
            account_trade_from_activity(&account_address, activity, trade_source)
        })
        .filter(|trade| {
            request
                .token_id
                .as_ref()
                .map(|token| token == &trade.token_id)
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    trades.sort_by_key(|trade| trade.timestamp_utc);

    let unmatched_trades = if process_id.is_some() {
        let mut venue_order_ids = trades
            .iter()
            .filter_map(|trade| trade.venue_order_id.clone())
            .collect::<Vec<_>>();
        venue_order_ids.sort_unstable();
        venue_order_ids.dedup();
        let owned_orders = account_owned_orders_for_reconciliation(
            store,
            account_ref
                .as_deref()
                .context("process-scoped reconciliation is missing account_ref")?,
            &venue_order_ids,
        )
        .await?;
        let mut transaction_hashes = trades
            .iter()
            .filter(|trade| trade.venue_order_id.is_none())
            .filter_map(|trade| {
                trade
                    .transaction_hash
                    .as_deref()
                    .and_then(normalize_transaction_hash)
            })
            .collect::<Vec<_>>();
        transaction_hashes.sort_unstable();
        transaction_hashes.dedup();
        let owned_fills = account_owned_fills_for_reconciliation(
            store,
            account_ref
                .as_deref()
                .context("process-scoped reconciliation is missing account_ref")?,
            &transaction_hashes,
            activity_window_start - LIVE_EXTERNAL_EVENT_CLOCK_SKEW,
            activity_window_end + LIVE_EXTERNAL_EVENT_CLOCK_SKEW,
        )
        .await?;
        let unlinked_normalized =
            link_process_owned_trades(&mut trades, &owned_orders, &owned_fills);
        let unmatched = unlinked_normalized
            .saturating_add(external_trade_activities.saturating_sub(trades.len()));
        unmatched as u64
    } else {
        // Account-wide administrative reconciliation predates process ownership. Preserve its
        // reporting contract instead of pretending wallet activity belongs to one process.
        0
    };

    let mut positions_query = PositionsQuery::for_user(account_address.clone());
    positions_query.limit = Some(DATA_API_RECONCILIATION_PAGE_SIZE);
    positions_query.size_threshold = Some(Decimal::ZERO);
    let positions = fetch_bounded_positions(data_api, positions_query).await?;
    let snapshots = positions
        .iter()
        .filter_map(|position| {
            account_position_snapshot_from_data_api(&account_address, position, trade_source)
        })
        .filter(|snapshot| {
            request
                .token_id
                .as_ref()
                .map(|token| token == &snapshot.token_id)
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    let redemption_evidence = redemption_activities
        .iter()
        .map(|activity| live_redemption_from_activity(&account_address, activity, trade_source))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let mut inserted_trades = 0;
    let mut inserted_snapshots = 0;
    if !request.dry_run {
        for trade in &trades {
            if store.upsert_account_trade(trade).await? {
                inserted_trades += 1;
            }
        }
        for snapshot in &snapshots {
            if store.insert_account_position_snapshot(snapshot).await? {
                inserted_snapshots += 1;
            }
        }
    }

    let mut exits_detected = 0u64;
    let mut exits_applied = 0u64;
    let mut exit_size_applied = Decimal::ZERO;
    if let Some(process_id) = process_id {
        let account_position_tokens = snapshots
            .iter()
            .filter(|snapshot| snapshot.size > Decimal::ZERO)
            .map(|snapshot| snapshot.token_id.as_str())
            .collect::<HashSet<_>>();
        for redemption in &redemption_evidence {
            if account_position_tokens.contains(redemption.token_id.as_str()) {
                continue;
            }
            let recognition = store
                .recognize_process_live_redemption(
                    process_id,
                    account_ref
                        .as_deref()
                        .context("process-scoped reconciliation is missing account_ref")?,
                    redemption,
                    !request.dry_run,
                )
                .await?;
            if recognition.matched {
                exits_detected = exits_detected
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("live redemption count overflow"))?;
            }
            if recognition.applied {
                exits_applied = exits_applied
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("applied live redemption count overflow"))?;
                exit_size_applied = exit_size_applied
                    .checked_add(recognition.redeemed_size)
                    .ok_or_else(|| anyhow::anyhow!("applied live redemption size overflow"))?;
            }
        }
    }

    let (process_accounting_proof, mismatches) = if let Some(process_id) = process_id {
        // Validate the bound sleeve's own immutable order/fill lineage independently, then prove
        // custody against the aggregate of every sleeve assigned to the account.
        let _process_positions = store.live_process_position_sizes(process_id).await?;
        let unsettled_resolved_positions = store
            .unsettled_resolved_live_position_sizes(process_id)
            .await?;
        let account_positions = store
            .live_account_position_sizes(
                account_ref
                    .as_deref()
                    .context("process-scoped reconciliation is missing account_ref")?,
            )
            .await?;
        process_accounting_proof(
            &account_positions,
            &snapshots,
            unmatched_trades,
            &unsettled_resolved_positions,
        )?
    } else {
        (
            ProcessAccountingProof {
                status: "legacy_unscoped".to_string(),
                position_ownership: "not_applicable".to_string(),
                realized_pnl: "not_applicable".to_string(),
                reason: "legacy_account_wide_admin_reconciliation".to_string(),
            },
            Vec::new(),
        )
    };
    let process_accounting_proven = process_accounting_proof.status == "proven";
    let process_accounting_status = process_accounting_proof.status.clone();
    let report = AccountReconcileReport {
        process_id,
        account_ref,
        credential_account_fingerprint_sha256,
        process_accounting_proven,
        process_accounting_status,
        process_accounting_proof,
        account_address,
        source: source.clone(),
        dry_run: request.dry_run,
        token_id: request.token_id,
        lookback_hours,
        activities_fetched: activities.len().saturating_add(redemption_activities.len()),
        account_trades_detected: trades.len(),
        account_trades_inserted: inserted_trades,
        position_snapshots_detected: snapshots.len(),
        position_snapshots_inserted: inserted_snapshots,
        exits_detected,
        exits_applied,
        exit_size_applied,
        position_adjustments_detected: 0,
        position_adjustments_applied: 0,
        position_adjustment_size_applied: Decimal::ZERO,
        mismatches,
        unmatched_trades,
    };
    store.insert_account_reconciliation_run(&report).await?;
    Ok(report)
}

fn process_accounting_proof(
    expected_account_positions: &HashMap<String, Decimal>,
    account_positions: &[AccountPositionSnapshot],
    unmatched_trades: u64,
    unsettled_resolved_positions: &[(String, Decimal)],
) -> Result<(ProcessAccountingProof, Vec<AccountPositionMismatch>)> {
    let mut account_sizes = HashMap::with_capacity(account_positions.len());
    for position in account_positions {
        if position.size <= Decimal::ZERO || is_settled_zero_payout_position(position) {
            continue;
        }
        if account_sizes
            .insert(position.token_id.clone(), position.size)
            .is_some()
        {
            bail!(
                "account position reconciliation contains duplicate token identity {}",
                position.token_id
            );
        }
    }

    let mut token_ids = expected_account_positions
        .keys()
        .chain(account_sizes.keys())
        .cloned()
        .collect::<Vec<_>>();
    token_ids.sort_unstable();
    token_ids.dedup();
    let mut mismatches = Vec::new();
    for token_id in token_ids {
        let process_size = expected_account_positions
            .get(&token_id)
            .copied()
            .unwrap_or(Decimal::ZERO);
        let account_size = account_sizes
            .get(&token_id)
            .copied()
            .unwrap_or(Decimal::ZERO);
        if decimal_difference(process_size, account_size) <= data_api_position_size_tolerance() {
            continue;
        }
        let mismatch_type = if process_size == Decimal::ZERO {
            "foreign_account_position"
        } else if account_size == Decimal::ZERO {
            "missing_account_position"
        } else {
            "position_size_mismatch"
        };
        mismatches.push(AccountPositionMismatch {
            token_id,
            db_open_size: process_size,
            account_size,
            delta_size: account_size - process_size,
            mismatch_type: mismatch_type.to_string(),
        });
    }

    for (token_id, unsettled_size) in unsettled_resolved_positions {
        mismatches.push(AccountPositionMismatch {
            token_id: token_id.clone(),
            db_open_size: *unsettled_size,
            account_size: Decimal::ZERO,
            delta_size: -*unsettled_size,
            mismatch_type: "resolved_fill_missing_credited_settlement".to_string(),
        });
    }

    let proof = if unmatched_trades > 0 {
        ProcessAccountingProof {
            status: "unproven".to_string(),
            position_ownership: "unproven".to_string(),
            realized_pnl: "unproven".to_string(),
            reason: "unmatched_account_trades".to_string(),
        }
    } else if !mismatches.is_empty() {
        ProcessAccountingProof {
            status: "unproven".to_string(),
            position_ownership: "mismatch".to_string(),
            realized_pnl: "unproven".to_string(),
            reason: "account_positions_do_not_match_aggregate_sleeve_live_fills".to_string(),
        }
    } else {
        ProcessAccountingProof {
            status: "proven".to_string(),
            position_ownership: "account_sleeve_fill_ledger_match".to_string(),
            realized_pnl: "process_owned_settlement_ledger".to_string(),
            reason: if expected_account_positions.is_empty() {
                "clean_account_baseline".to_string()
            } else {
                "account_positions_match_aggregate_sleeve_live_fills".to_string()
            },
        }
    };
    Ok((proof, mismatches))
}

fn is_settled_zero_payout_position(position: &AccountPositionSnapshot) -> bool {
    position.current_value == Some(Decimal::ZERO)
        && position
            .raw_payload
            .get("redeemable")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
}

async fn fetch_bounded_activity(
    data_api: &DataApiClient,
    mut query: ActivityQuery,
) -> Result<Vec<DataApiActivity>> {
    let mut rows = Vec::new();
    let mut seen_pages = HashSet::new();
    let mut offset = query.offset.unwrap_or(0);
    query.limit = Some(DATA_API_RECONCILIATION_PAGE_SIZE);
    for _ in 0..MAX_DATA_API_RECONCILIATION_REQUESTS {
        query.offset = Some(offset);
        let page = data_api.fetch_activity(&query).await?;
        let continue_paging =
            append_bounded_data_api_page(&mut rows, &mut seen_pages, page, "activity")?;
        if !continue_paging {
            return Ok(rows);
        }
        offset = offset
            .checked_add(DATA_API_RECONCILIATION_PAGE_SIZE)
            .ok_or_else(|| anyhow::anyhow!("Polymarket Data API activity offset overflow"))?;
    }
    bail!(
        "Polymarket Data API activity exceeds the bounded {}-row reconciliation window",
        MAX_DATA_API_RECONCILIATION_ROWS
    )
}

async fn fetch_bounded_positions(
    data_api: &DataApiClient,
    mut query: PositionsQuery,
) -> Result<Vec<DataApiPosition>> {
    let mut rows = Vec::new();
    let mut seen_pages = HashSet::new();
    let mut offset = query.offset.unwrap_or(0);
    query.limit = Some(DATA_API_RECONCILIATION_PAGE_SIZE);
    for _ in 0..MAX_DATA_API_RECONCILIATION_REQUESTS {
        query.offset = Some(offset);
        let page = data_api.fetch_positions(&query).await?;
        let continue_paging =
            append_bounded_data_api_page(&mut rows, &mut seen_pages, page, "positions")?;
        if !continue_paging {
            return Ok(rows);
        }
        offset = offset
            .checked_add(DATA_API_RECONCILIATION_PAGE_SIZE)
            .ok_or_else(|| anyhow::anyhow!("Polymarket Data API positions offset overflow"))?;
    }
    bail!(
        "Polymarket Data API positions exceed the bounded {}-row reconciliation window",
        MAX_DATA_API_RECONCILIATION_ROWS
    )
}

fn append_bounded_data_api_page<T: Serialize>(
    rows: &mut Vec<T>,
    seen_pages: &mut HashSet<String>,
    page: Vec<T>,
    resource: &str,
) -> Result<bool> {
    if page.len() > DATA_API_RECONCILIATION_PAGE_SIZE {
        bail!(
            "Polymarket Data API {} page exceeded the requested {}-row bound",
            resource,
            DATA_API_RECONCILIATION_PAGE_SIZE
        );
    }
    if page.is_empty() {
        return Ok(false);
    }
    let page_len = page.len();
    let page_sha256 = event_hash(&serde_json::to_value(&page)?);
    if !seen_pages.insert(page_sha256) {
        bail!("Polymarket Data API {resource} repeated a reconciliation page");
    }
    if rows.len().saturating_add(page_len) > MAX_DATA_API_RECONCILIATION_ROWS {
        bail!(
            "Polymarket Data API {} exceeds the bounded {}-row reconciliation window",
            resource,
            MAX_DATA_API_RECONCILIATION_ROWS
        );
    }
    rows.extend(page);
    Ok(page_len == DATA_API_RECONCILIATION_PAGE_SIZE)
}

pub fn account_trade_from_live_event(
    account_address: &str,
    event: &LiveVenueEvent,
) -> Option<AccountTrade> {
    if event.event_type != "trade" {
        return None;
    }
    let token_id = json_str(&event.raw_payload, "asset_id")
        .or_else(|| json_str(&event.raw_payload, "asset"))
        .or_else(|| maker_order_str(&event.raw_payload, "asset_id"))?
        .to_string();
    let side = normalize_side(
        json_str(&event.raw_payload, "side")
            .or_else(|| maker_order_str(&event.raw_payload, "side"))
            .unwrap_or(""),
    )?;
    let price = json_decimal(&event.raw_payload, "price")
        .or_else(|| maker_order_decimal(&event.raw_payload, "price"))?;
    let size = json_decimal(&event.raw_payload, "size")
        .or_else(|| json_decimal(&event.raw_payload, "matched_amount"))
        .or_else(|| maker_order_decimal(&event.raw_payload, "matched_amount"))?;
    let timestamp = json_i64(&event.raw_payload, "match_time")
        .or_else(|| json_i64(&event.raw_payload, "timestamp"))
        .and_then(timestamp_from_raw)
        .unwrap_or_else(Utc::now);
    let venue_trade_id = event
        .venue_trade_id
        .clone()
        .or_else(|| json_str(&event.raw_payload, "trade_id").map(str::to_string));
    Some(AccountTrade::new(
        account_address,
        &token_id,
        json_str(&event.raw_payload, "market").map(str::to_string),
        side,
        price,
        size,
        timestamp,
        json_str(&event.raw_payload, "transaction_hash").map(str::to_string),
        event.venue_order_id.clone(),
        venue_trade_id,
        "user_ws",
        event.raw_payload.clone(),
    ))
}

fn account_trade_from_activity(
    account_address: &str,
    activity: &DataApiActivity,
    source: &str,
) -> Option<AccountTrade> {
    let token_id = activity.asset.clone()?;
    let side = normalize_side(activity.side.as_deref()?)?;
    let price = activity.price?;
    let size = activity.size?;
    let timestamp = timestamp_from_raw(activity.timestamp?)?;
    let mut raw_payload = serde_json::to_value(activity).ok()?;
    if let serde_json::Value::Object(object) = &mut raw_payload {
        object.insert("normalizer_source".to_string(), serde_json::json!(source));
    }
    Some(AccountTrade::new(
        account_address,
        &token_id,
        activity.condition_id.clone(),
        side,
        price,
        size,
        timestamp,
        activity.transaction_hash.clone(),
        activity_external_id(
            activity,
            &[
                "venue_order_id",
                "venueOrderId",
                "order_id",
                "orderId",
                "taker_order_id",
                "takerOrderId",
            ],
        ),
        activity_external_id(
            activity,
            &["venue_trade_id", "venueTradeId", "trade_id", "tradeId"],
        ),
        source,
        raw_payload,
    ))
}

fn normalize_account_ref(account_ref: Option<&str>) -> Result<Option<String>> {
    let Some(account_ref) = account_ref else {
        return Ok(None);
    };
    let account_ref = account_ref.trim();
    if account_ref.is_empty() {
        bail!("account reconciliation account_ref must not be blank");
    }
    if account_ref.len() > 128 {
        bail!("account reconciliation account_ref must not exceed 128 bytes");
    }
    Ok(Some(account_ref.to_string()))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn live_redemption_from_activity(
    account_address: &str,
    activity: &DataApiActivity,
    source: &str,
) -> Result<Option<LiveRedemptionEvidence>> {
    if !activity
        .activity_type
        .as_deref()
        .is_some_and(|activity_type| activity_type.eq_ignore_ascii_case("redeem"))
    {
        return Ok(None);
    }
    let Some(redeemed_size) = activity.size.filter(|size| *size > Decimal::ZERO) else {
        return Ok(None);
    };
    let proxy_wallet = activity
        .proxy_wallet
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("positive REDEEM activity is missing proxyWallet"))?;
    if !proxy_wallet.eq_ignore_ascii_case(account_address) {
        bail!("REDEEM activity wallet does not match the reconciled account");
    }
    let condition_id = activity
        .condition_id
        .as_deref()
        .map(str::trim)
        .filter(|value| is_prefixed_hex(value, 32))
        .ok_or_else(|| anyhow::anyhow!("positive REDEEM activity has an invalid conditionId"))?
        .to_ascii_lowercase();
    let token_id = activity
        .asset
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
        .ok_or_else(|| anyhow::anyhow!("positive REDEEM activity has an invalid asset"))?
        .to_string();
    let payout_usd = activity
        .usdc_size
        .filter(|payout| *payout == Decimal::ZERO || *payout == redeemed_size)
        .ok_or_else(|| {
            anyhow::anyhow!("positive REDEEM activity payout is neither zero nor its token size")
        })?;
    let redeemed_at = activity
        .timestamp
        .and_then(timestamp_from_raw)
        .ok_or_else(|| anyhow::anyhow!("positive REDEEM activity has an invalid timestamp"))?;
    if redeemed_at > Utc::now() + LIVE_EXTERNAL_EVENT_CLOCK_SKEW {
        bail!("positive REDEEM activity timestamp is in the future");
    }
    let transaction_hash = activity
        .transaction_hash
        .as_deref()
        .map(str::trim)
        .filter(|value| is_prefixed_hex(value, 32))
        .ok_or_else(|| anyhow::anyhow!("positive REDEEM activity has an invalid transactionHash"))?
        .to_ascii_lowercase();
    let mut raw_payload = serde_json::to_value(activity)?;
    if let serde_json::Value::Object(object) = &mut raw_payload {
        object.insert("normalizer_source".to_string(), serde_json::json!(source));
    }
    Ok(Some(LiveRedemptionEvidence {
        account_address: account_address.to_ascii_lowercase(),
        condition_id,
        token_id,
        redeemed_size,
        payout_usd,
        redeemed_at,
        transaction_hash,
        raw_payload,
    }))
}

fn is_prefixed_hex(value: &str, bytes: usize) -> bool {
    value.len() == bytes.saturating_mul(2).saturating_add(2)
        && value.starts_with("0x")
        && value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_trade_activity(activity: &DataApiActivity) -> bool {
    activity
        .activity_type
        .as_deref()
        .map(|activity_type| activity_type.eq_ignore_ascii_case("trade"))
        .unwrap_or_else(|| {
            activity.side.is_some()
                && activity.price.is_some()
                && activity.size.is_some()
                && activity.asset.is_some()
        })
}

fn activity_external_id(activity: &DataApiActivity, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        activity
            .extra
            .get(*key)
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

fn normalize_transaction_hash(value: &str) -> Option<String> {
    let value = value.trim();
    is_prefixed_hex(value, 32).then(|| value.to_ascii_lowercase())
}

/// Attributes a normalized account trade through either its explicit venue order identity or one
/// unique persisted live fill carrying the same venue transaction identity. Fill economics only
/// disambiguate candidates inside that exact transaction; token or market similarity alone never
/// establishes ownership.
fn link_process_owned_trades(
    trades: &mut [AccountTrade],
    owned_orders: &HashMap<String, String>,
    owned_fills: &[AccountLiveFillEvidence],
) -> usize {
    let mut fills_by_transaction = HashMap::<&str, Vec<&AccountLiveFillEvidence>>::new();
    for evidence in owned_fills {
        fills_by_transaction
            .entry(evidence.transaction_hash.as_str())
            .or_default()
            .push(evidence);
    }
    let mut unmatched = 0usize;
    for trade in trades {
        let order_id = if let Some(venue_order_id) = trade.venue_order_id.as_ref() {
            owned_orders.get(venue_order_id).cloned()
        } else {
            trade
                .transaction_hash
                .as_deref()
                .and_then(normalize_transaction_hash)
                .and_then(|transaction_hash| fills_by_transaction.get(transaction_hash.as_str()))
                .and_then(|candidates| {
                    let identity_candidates = candidates
                        .iter()
                        .filter(|evidence| {
                            evidence.token_id == trade.token_id
                                && evidence.side.eq_ignore_ascii_case(&trade.side)
                        })
                        .copied()
                        .collect::<Vec<_>>();
                    unique_order_id(&identity_candidates).or_else(|| {
                        let economic_candidates = identity_candidates
                            .into_iter()
                            .filter(|evidence| {
                                decimal_difference(evidence.price, trade.price)
                                    <= data_api_trade_price_tolerance()
                                    && decimal_difference(evidence.size, trade.size)
                                        <= data_api_trade_size_tolerance()
                            })
                            .collect::<Vec<_>>();
                        unique_order_id(&economic_candidates)
                    })
                })
        };
        let Some(order_id) = order_id else {
            unmatched = unmatched.saturating_add(1);
            continue;
        };
        trade.linked_order_id = Some(order_id);
    }
    unmatched
}

fn unique_order_id(candidates: &[&AccountLiveFillEvidence]) -> Option<String> {
    let order_ids = candidates
        .iter()
        .map(|evidence| evidence.order_id.as_str())
        .collect::<HashSet<_>>();
    (order_ids.len() == 1).then(|| {
        order_ids
            .into_iter()
            .next()
            .expect("one fill order identity must exist")
            .to_string()
    })
}

fn decimal_difference(left: Decimal, right: Decimal) -> Decimal {
    if left >= right {
        left - right
    } else {
        right - left
    }
}

async fn account_owned_fills_for_reconciliation(
    store: &Store,
    account_ref: &str,
    transaction_hashes: &[String],
    window_start: chrono::DateTime<Utc>,
    window_end: chrono::DateTime<Utc>,
) -> Result<Vec<AccountLiveFillEvidence>> {
    if transaction_hashes.len() > MAX_DATA_API_RECONCILIATION_ROWS {
        bail!(
            "account reconciliation exceeds the bounded {}-transaction ownership window",
            MAX_DATA_API_RECONCILIATION_ROWS
        );
    }
    let mut owned_fills = Vec::new();
    for chunk in transaction_hashes.chunks(DATA_API_RECONCILIATION_PAGE_SIZE) {
        owned_fills.extend(
            store
                .account_live_fill_evidence_by_transaction_hashes(
                    account_ref,
                    chunk,
                    window_start,
                    window_end,
                )
                .await?,
        );
        if owned_fills.len() > MAX_DATA_API_RECONCILIATION_ROWS {
            bail!(
                "account reconciliation exceeds the bounded {}-fill ownership window",
                MAX_DATA_API_RECONCILIATION_ROWS
            );
        }
    }
    Ok(owned_fills)
}

async fn account_owned_orders_for_reconciliation(
    store: &Store,
    account_ref: &str,
    venue_order_ids: &[String],
) -> Result<HashMap<String, String>> {
    if venue_order_ids.len() > MAX_DATA_API_RECONCILIATION_ROWS {
        bail!(
            "account reconciliation exceeds the bounded {}-order ownership window",
            MAX_DATA_API_RECONCILIATION_ROWS
        );
    }
    let mut owned_orders = HashMap::new();
    for chunk in venue_order_ids.chunks(DATA_API_RECONCILIATION_PAGE_SIZE) {
        let chunk_owned = store
            .account_order_ids_by_venue_order_ids(account_ref, chunk)
            .await?;
        merge_owned_order_evidence(&mut owned_orders, chunk_owned)?;
    }
    Ok(owned_orders)
}

fn merge_owned_order_evidence(
    owned_orders: &mut HashMap<String, String>,
    chunk_owned: HashMap<String, String>,
) -> Result<()> {
    for (venue_order_id, local_order_id) in chunk_owned {
        if owned_orders
            .insert(venue_order_id.clone(), local_order_id.clone())
            .is_some_and(|existing| existing != local_order_id)
        {
            bail!(
                "venue order {} maps to conflicting local order identities during account reconciliation",
                venue_order_id
            );
        }
    }
    Ok(())
}

fn account_position_snapshot_from_data_api(
    account_address: &str,
    position: &DataApiPosition,
    source: &str,
) -> Option<AccountPositionSnapshot> {
    let token_id = position.asset.clone()?;
    let raw_payload = serde_json::to_value(position).ok()?;
    Some(AccountPositionSnapshot {
        snapshot_id: Uuid::new_v4(),
        account_address: account_address.to_ascii_lowercase(),
        token_id,
        market_id: position.condition_id.clone(),
        size: position.size.unwrap_or(Decimal::ZERO).max(Decimal::ZERO),
        avg_price: position.avg_price,
        current_price: position.cur_price,
        current_value: position.current_value,
        cash_pnl: position.cash_pnl,
        percent_pnl: position.percent_pnl,
        snapshot_at: Utc::now(),
        source: source.to_string(),
        raw_payload,
    })
}

fn normalize_side(value: &str) -> Option<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "buy" | "bought" => Some("buy"),
        "sell" | "sold" => Some("sell"),
        _ => None,
    }
}

fn timestamp_from_raw(raw: i64) -> Option<chrono::DateTime<Utc>> {
    let seconds = if raw > 10_000_000_000 {
        raw / 1000
    } else {
        raw
    };
    Utc.timestamp_opt(seconds, 0).single()
}

fn json_str<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(|value| value.as_str())
}

fn json_i64(value: &serde_json::Value, key: &str) -> Option<i64> {
    match value.get(key)? {
        serde_json::Value::Number(number) => number.as_i64(),
        serde_json::Value::String(raw) => raw.parse().ok(),
        _ => None,
    }
}

fn json_decimal(value: &serde_json::Value, key: &str) -> Option<Decimal> {
    decimal_from_json(value.get(key)?)
}

fn maker_order_str<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value
        .get("maker_orders")
        .and_then(|orders| orders.as_array())
        .and_then(|orders| orders.first())
        .and_then(|order| order.get(key))
        .and_then(|value| value.as_str())
}

fn maker_order_decimal(value: &serde_json::Value, key: &str) -> Option<Decimal> {
    value
        .get("maker_orders")
        .and_then(|orders| orders.as_array())
        .and_then(|orders| orders.first())
        .and_then(|order| order.get(key))
        .and_then(decimal_from_json)
}

fn decimal_from_json(value: &serde_json::Value) -> Option<Decimal> {
    match value {
        serde_json::Value::String(raw) => raw.parse().ok(),
        serde_json::Value::Number(number) => number.to_string().parse().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use rust_decimal_macros::dec;

    use super::*;

    #[test]
    fn websocket_trade_normalizes_to_account_trade() {
        let event = LiveVenueEvent {
            source: "user_ws".to_string(),
            event_type: "trade".to_string(),
            venue_event_id: Some("evt-1".to_string()),
            venue_order_id: Some("order-1".to_string()),
            venue_trade_id: Some("trade-1".to_string()),
            event_status: Some("MATCHED".to_string()),
            raw_payload: serde_json::json!({
                "asset_id": "token-1",
                "side": "SELL",
                "price": "0.42",
                "size": "3.5",
                "match_time": 1710000000,
                "transaction_hash": "0xabc"
            }),
        };

        let trade = account_trade_from_live_event("0xABC", &event).unwrap();

        assert_eq!(trade.account_address, "0xabc");
        assert_eq!(trade.token_id, "token-1");
        assert_eq!(trade.side, "sell");
        assert_eq!(trade.price, dec!(0.42));
        assert_eq!(trade.size, dec!(3.5));
        assert_eq!(trade.notional, dec!(1.470));
        assert_eq!(trade.source, "user_ws");
    }

    #[test]
    fn websocket_maker_order_fields_are_supported() {
        let event = LiveVenueEvent {
            source: "user_ws".to_string(),
            event_type: "trade".to_string(),
            venue_event_id: None,
            venue_order_id: None,
            venue_trade_id: Some("trade-2".to_string()),
            event_status: Some("MATCHED".to_string()),
            raw_payload: serde_json::json!({
                "maker_orders": [{
                    "asset_id": "token-2",
                    "side": "BUY",
                    "price": "0.55",
                    "matched_amount": "2"
                }],
                "timestamp": 1710000000000i64
            }),
        };

        let trade = account_trade_from_live_event("0xabc", &event).unwrap();

        assert_eq!(trade.token_id, "token-2");
        assert_eq!(trade.side, "buy");
        assert_eq!(trade.price, dec!(0.55));
        assert_eq!(trade.size, dec!(2));
    }

    #[test]
    fn positive_redeem_activity_normalizes_to_exact_live_redemption_evidence() {
        let activity: DataApiActivity = serde_json::from_value(serde_json::json!({
            "proxyWallet": "0x74d0da822ba46c7325bb78e74c915976e76159af",
            "type": "REDEEM",
            "size": "5",
            "usdcSize": "5",
            "timestamp": 1786677774,
            "transactionHash": "0x072aa67fafa8381f5bd25c3092f2146b3e6e1c6717ce71599bee3352a93e3da4",
            "asset": "64891112840096581114786599417318199598343837807127888883355968643722878718210",
            "conditionId": "0x21582805dbfc8aea9dc1cbbe7171f6a60c477726681588c98a97410d4f194fec"
        }))
        .unwrap();

        let evidence = live_redemption_from_activity(
            "0x74D0dA822ba46c7325bB78E74C915976e76159af",
            &activity,
            "poll",
        )
        .unwrap()
        .unwrap();

        assert_eq!(evidence.redeemed_size, dec!(5));
        assert_eq!(evidence.payout_usd, dec!(5));
        assert_eq!(
            evidence.transaction_hash,
            "0x072aa67fafa8381f5bd25c3092f2146b3e6e1c6717ce71599bee3352a93e3da4"
        );
        assert_eq!(evidence.raw_payload["normalizer_source"], "poll");
    }

    #[test]
    fn zero_redeem_is_ignored_and_positive_redeem_requires_exact_wallet_and_payout() {
        let zero: DataApiActivity = serde_json::from_value(serde_json::json!({
            "proxyWallet": "0xabc",
            "type": "REDEEM",
            "size": 0,
            "usdcSize": 0,
            "timestamp": 1786677975,
            "transactionHash": "0x05d2c9eda77f7e4aa2f395b7577c6745d705af9f374951a74f0b699692e3ac2d"
        }))
        .unwrap();
        assert!(live_redemption_from_activity("0xabc", &zero, "poll")
            .unwrap()
            .is_none());

        let mismatched: DataApiActivity = serde_json::from_value(serde_json::json!({
            "proxyWallet": "0x0000000000000000000000000000000000000000",
            "type": "REDEEM",
            "size": "5",
            "usdcSize": "4.99",
            "timestamp": 1786677774,
            "transactionHash": "0x072aa67fafa8381f5bd25c3092f2146b3e6e1c6717ce71599bee3352a93e3da4",
            "asset": "1",
            "conditionId": "0x21582805dbfc8aea9dc1cbbe7171f6a60c477726681588c98a97410d4f194fec"
        }))
        .unwrap();
        assert!(live_redemption_from_activity(
            "0x74d0da822ba46c7325bb78e74c915976e76159af",
            &mismatched,
            "poll"
        )
        .is_err());
    }

    #[test]
    fn legacy_account_reconcile_request_deserializes_without_scope() {
        let request: AccountReconcileRequest = serde_json::from_value(serde_json::json!({
            "account_address": "0xabc",
            "lookback_hours": 1,
            "dry_run": true
        }))
        .unwrap();

        assert_eq!(request.process_id, None);
        assert_eq!(request.account_ref, None);
        assert_eq!(request.credential_account_fingerprint_sha256, None);
    }

    #[test]
    fn process_trade_linkage_requires_explicit_owned_venue_order_identity() {
        let activity: DataApiActivity = serde_json::from_value(serde_json::json!({
            "type": "TRADE",
            "timestamp": 1710000000,
            "conditionId": "market-1",
            "asset": "token-1",
            "side": "BUY",
            "price": "0.40",
            "size": "2",
            "transactionHash": "0xtrade",
            "orderId": "venue-owned"
        }))
        .unwrap();
        let owned_trade = account_trade_from_activity("0xabc", &activity, "poll").unwrap();
        assert_eq!(owned_trade.venue_order_id.as_deref(), Some("venue-owned"));

        let mut same_token_without_order_identity = owned_trade.clone();
        same_token_without_order_identity.account_trade_id = Uuid::new_v4();
        same_token_without_order_identity.venue_order_id = None;
        let mut trades = vec![owned_trade, same_token_without_order_identity];
        let owned_orders =
            HashMap::from([("venue-owned".to_string(), "persisted-order".to_string())]);

        let unmatched = link_process_owned_trades(&mut trades, &owned_orders, &[]);

        assert_eq!(unmatched, 1);
        assert_eq!(
            trades[0].linked_order_id.as_deref(),
            Some("persisted-order")
        );
        assert_eq!(trades[1].linked_order_id, None);
    }

    #[test]
    fn process_trade_linkage_uses_unique_exact_account_fill_transaction_evidence() {
        let transaction_hash = format!("0x{}", "a".repeat(64));
        let activity: DataApiActivity = serde_json::from_value(serde_json::json!({
            "type": "TRADE",
            "timestamp": 1710000000,
            "conditionId": "market-1",
            "asset": "token-1",
            "side": "BUY",
            "price": "0.40",
            "size": "2",
            "transactionHash": transaction_hash
        }))
        .unwrap();
        let trade = account_trade_from_activity("0xabc", &activity, "poll").unwrap();
        assert_eq!(trade.venue_order_id, None);
        let exact = AccountLiveFillEvidence {
            transaction_hash: transaction_hash.clone(),
            order_id: "persisted-order".to_string(),
            token_id: "token-1".to_string(),
            side: "buy".to_string(),
            price: Decimal::new(40, 2),
            size: Decimal::new(2, 0),
        };

        let mut uniquely_owned = vec![trade.clone()];
        assert_eq!(
            link_process_owned_trades(&mut uniquely_owned, &HashMap::new(), &[exact.clone()]),
            0
        );
        assert_eq!(
            uniquely_owned[0].linked_order_id.as_deref(),
            Some("persisted-order")
        );

        let mut conflicting = exact;
        conflicting.order_id = "different-order".to_string();
        let mut ambiguous = vec![trade];
        assert_eq!(
            link_process_owned_trades(
                &mut ambiguous,
                &HashMap::new(),
                &[
                    AccountLiveFillEvidence {
                        order_id: "persisted-order".to_string(),
                        ..conflicting.clone()
                    },
                    conflicting,
                ],
            ),
            1
        );
        assert_eq!(ambiguous[0].linked_order_id, None);
    }

    #[test]
    fn process_trade_linkage_tolerates_data_api_precision_when_disambiguating() {
        let transaction_hash = format!("0x{}", "b".repeat(64));
        let activity: DataApiActivity = serde_json::from_value(serde_json::json!({
            "type": "TRADE",
            "timestamp": 1710000000,
            "conditionId": "market-1",
            "asset": "token-1",
            "side": "BUY",
            "price": "0.40000000",
            "size": "2.000000",
            "transactionHash": transaction_hash
        }))
        .unwrap();
        let mut trades = vec![account_trade_from_activity("0xabc", &activity, "poll").unwrap()];
        let candidates = vec![
            AccountLiveFillEvidence {
                transaction_hash: transaction_hash.clone(),
                order_id: "owned-near".to_string(),
                token_id: "token-1".to_string(),
                side: "buy".to_string(),
                price: dec!(0.400000009),
                size: dec!(2.0000009),
            },
            AccountLiveFillEvidence {
                transaction_hash,
                order_id: "owned-far".to_string(),
                token_id: "token-1".to_string(),
                side: "buy".to_string(),
                price: dec!(0.41),
                size: dec!(2.1),
            },
        ];

        assert_eq!(
            link_process_owned_trades(&mut trades, &HashMap::new(), &candidates),
            0
        );
        assert_eq!(trades[0].linked_order_id.as_deref(), Some("owned-near"));
    }

    #[test]
    fn zero_payout_losing_redemption_is_valid_normalized_evidence() {
        let account = "0x1111111111111111111111111111111111111111";
        let activity: DataApiActivity = serde_json::from_value(serde_json::json!({
            "type": "REDEEM",
            "timestamp": 1710000000,
            "conditionId": format!("0x{}", "c".repeat(64)),
            "asset": "12345",
            "size": "2.5",
            "usdcSize": "0",
            "proxyWallet": account,
            "transactionHash": format!("0x{}", "d".repeat(64))
        }))
        .unwrap();

        let evidence = live_redemption_from_activity(account, &activity, "poll")
            .unwrap()
            .unwrap();
        assert_eq!(evidence.redeemed_size, dec!(2.5));
        assert_eq!(evidence.payout_usd, Decimal::ZERO);
    }

    #[test]
    fn process_order_ownership_chunks_are_store_bounded_and_conflicts_fail_closed() {
        let venue_order_ids = (0..1_201)
            .map(|index| format!("venue-{index}"))
            .collect::<Vec<_>>();
        assert_eq!(
            venue_order_ids
                .chunks(DATA_API_RECONCILIATION_PAGE_SIZE)
                .map(|chunk| chunk.len())
                .collect::<Vec<_>>(),
            vec![500, 500, 201]
        );

        let mut merged = HashMap::new();
        merge_owned_order_evidence(
            &mut merged,
            HashMap::from([("venue-1".to_string(), "local-1".to_string())]),
        )
        .unwrap();
        merge_owned_order_evidence(
            &mut merged,
            HashMap::from([("venue-1".to_string(), "local-1".to_string())]),
        )
        .unwrap();
        assert!(merge_owned_order_evidence(
            &mut merged,
            HashMap::from([("venue-1".to_string(), "local-other".to_string())]),
        )
        .is_err());
    }

    #[test]
    fn account_reference_is_trimmed_bounded_and_nonblank() {
        assert_eq!(
            normalize_account_ref(Some("  polymarket-primary  ")).unwrap(),
            Some("polymarket-primary".to_string())
        );
        assert!(normalize_account_ref(Some("   ")).is_err());
        assert!(normalize_account_ref(Some(&"x".repeat(129))).is_err());
        assert!(is_sha256_hex(&"a".repeat(64)));
        assert!(!is_sha256_hex(&"z".repeat(64)));
    }

    fn account_position(token_id: &str, size: Decimal) -> AccountPositionSnapshot {
        AccountPositionSnapshot {
            snapshot_id: Uuid::new_v4(),
            account_address: "0xabc".to_string(),
            token_id: token_id.to_string(),
            market_id: Some("market-1".to_string()),
            size,
            avg_price: None,
            current_price: None,
            current_value: None,
            cash_pnl: None,
            percent_pnl: None,
            snapshot_at: Utc::now(),
            source: "poll".to_string(),
            raw_payload: serde_json::json!({}),
        }
    }

    #[test]
    fn clean_account_baseline_proves_process_accounting() {
        let (proof, mismatches) = process_accounting_proof(&HashMap::new(), &[], 0, &[]).unwrap();

        assert_eq!(proof.status, "proven");
        assert_eq!(proof.reason, "clean_account_baseline");
        assert!(mismatches.is_empty());
    }

    #[test]
    fn exact_process_fill_position_proves_process_accounting() {
        let process_positions = HashMap::from([("token-1".to_string(), dec!(2.5))]);
        let positions = vec![account_position("token-1", dec!(2.5))];

        let (proof, mismatches) =
            process_accounting_proof(&process_positions, &positions, 0, &[]).unwrap();

        assert_eq!(proof.status, "proven");
        assert_eq!(
            proof.reason,
            "account_positions_match_aggregate_sleeve_live_fills"
        );
        assert!(mismatches.is_empty());
    }

    #[test]
    fn rounded_data_api_position_within_resolution_proves_accounting() {
        let process_positions = HashMap::from([("token-1".to_string(), dec!(2.5000005))]);
        let positions = vec![account_position("token-1", dec!(2.5))];

        let (proof, mismatches) =
            process_accounting_proof(&process_positions, &positions, 0, &[]).unwrap();

        assert_eq!(proof.status, "proven");
        assert!(mismatches.is_empty());
    }

    #[test]
    fn redeemable_zero_value_losing_token_is_not_open_wallet_custody() {
        let mut losing_position = account_position("losing-token", dec!(5));
        losing_position.current_value = Some(Decimal::ZERO);
        losing_position.raw_payload = serde_json::json!({
            "size": "5",
            "currentValue": "0",
            "redeemable": true
        });

        let (proof, mismatches) =
            process_accounting_proof(&HashMap::new(), &[losing_position], 0, &[]).unwrap();

        assert_eq!(proof.status, "proven");
        assert!(mismatches.is_empty());
    }

    #[test]
    fn unresolved_fill_settlement_coverage_prevents_realized_pnl_proof() {
        let (proof, mismatches) = process_accounting_proof(
            &HashMap::new(),
            &[],
            0,
            &[("losing-token".to_string(), dec!(2.5))],
        )
        .unwrap();

        assert_eq!(proof.status, "unproven");
        assert_eq!(proof.realized_pnl, "unproven");
        assert_eq!(
            mismatches[0].mismatch_type,
            "resolved_fill_missing_credited_settlement"
        );
    }

    #[test]
    fn foreign_position_or_unmatched_trade_fails_process_accounting_closed() {
        let foreign = vec![account_position("foreign-token", dec!(1))];
        let (foreign_proof, mismatches) =
            process_accounting_proof(&HashMap::new(), &foreign, 0, &[]).unwrap();
        assert_eq!(foreign_proof.status, "unproven");
        assert_eq!(mismatches[0].mismatch_type, "foreign_account_position");

        let (trade_proof, _) = process_accounting_proof(&HashMap::new(), &[], 1, &[]).unwrap();
        assert_eq!(trade_proof.status, "unproven");
        assert_eq!(trade_proof.reason, "unmatched_account_trades");
    }

    #[test]
    fn process_position_size_mismatch_fails_closed() {
        let process_positions = HashMap::from([("token-1".to_string(), dec!(2.5))]);
        let positions = vec![account_position("token-1", dec!(2))];

        let (proof, mismatches) =
            process_accounting_proof(&process_positions, &positions, 0, &[]).unwrap();

        assert_eq!(proof.status, "unproven");
        assert_eq!(mismatches[0].mismatch_type, "position_size_mismatch");
        assert_eq!(mismatches[0].delta_size, dec!(-0.5));
    }

    #[test]
    fn bounded_data_api_pages_stop_on_partial_and_reject_repetition_or_overflow() {
        let mut rows = Vec::new();
        let mut seen = HashSet::new();
        let full_page = (0..DATA_API_RECONCILIATION_PAGE_SIZE)
            .map(|index| serde_json::json!({"row": index}))
            .collect::<Vec<_>>();
        assert!(
            append_bounded_data_api_page(&mut rows, &mut seen, full_page.clone(), "test").unwrap()
        );
        assert_eq!(rows.len(), DATA_API_RECONCILIATION_PAGE_SIZE);

        let mut repeated_rows = Vec::new();
        let mut repeated_seen = HashSet::new();
        append_bounded_data_api_page(
            &mut repeated_rows,
            &mut repeated_seen,
            full_page.clone(),
            "test",
        )
        .unwrap();
        assert!(append_bounded_data_api_page(
            &mut repeated_rows,
            &mut repeated_seen,
            full_page,
            "test"
        )
        .is_err());

        let mut partial_rows = Vec::new();
        let mut partial_seen = HashSet::new();
        assert!(!append_bounded_data_api_page(
            &mut partial_rows,
            &mut partial_seen,
            vec![serde_json::json!({"row": 1})],
            "test"
        )
        .unwrap());

        let mut full_rows = (0..MAX_DATA_API_RECONCILIATION_ROWS)
            .map(|index| serde_json::json!({"existing": index}))
            .collect::<Vec<_>>();
        let mut overflow_seen = HashSet::new();
        assert!(append_bounded_data_api_page(
            &mut full_rows,
            &mut overflow_seen,
            vec![serde_json::json!({"overflow": true})],
            "test"
        )
        .is_err());
    }
}
