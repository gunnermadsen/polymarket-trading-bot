use chrono::{DateTime, Utc};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct CausalRow {
    pub event_at: DateTime<Utc>,
    pub released_at: DateTime<Utc>,
    pub available_at: DateTime<Utc>,
    pub revision: String,
    pub payload: Value,
}

#[derive(Debug)]
pub struct FetchResult {
    pub source_url: String,
    pub source_bytes: Vec<u8>,
    pub rows: Vec<CausalRow>,
}

#[derive(Debug, Clone, Copy)]
pub enum EconomicDataset {
    FredEconomicSeries,
    NewYorkFedReferenceRates,
    NewYorkFedSomaHoldings,
    CftcLegacyFutures,
    CftcTradersFinancialFutures,
    TreasuryAuctions,
    TreasuryDebtToPenny,
    TreasuryDepositsWithdrawals,
    TreasuryOperatingCashBalance,
}

impl EconomicDataset {
    pub const fn provider(self) -> &'static str {
        match self {
            Self::FredEconomicSeries => "fred",
            Self::NewYorkFedReferenceRates | Self::NewYorkFedSomaHoldings => "new_york_fed",
            Self::CftcLegacyFutures | Self::CftcTradersFinancialFutures => "cftc",
            Self::TreasuryAuctions
            | Self::TreasuryDebtToPenny
            | Self::TreasuryDepositsWithdrawals
            | Self::TreasuryOperatingCashBalance => "us_treasury",
        }
    }

    pub const fn dataset(self) -> &'static str {
        match self {
            Self::FredEconomicSeries => "economic_series",
            Self::NewYorkFedReferenceRates => "reference_rates",
            Self::NewYorkFedSomaHoldings => "soma_holdings",
            Self::CftcLegacyFutures => "legacy_futures",
            Self::CftcTradersFinancialFutures => "traders_financial_futures",
            Self::TreasuryAuctions => "auctions",
            Self::TreasuryDebtToPenny => "debt_to_penny",
            Self::TreasuryDepositsWithdrawals => "deposits_withdrawals",
            Self::TreasuryOperatingCashBalance => "operating_cash_balance",
        }
    }
}
