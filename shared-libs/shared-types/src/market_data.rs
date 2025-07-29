use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: u64,
    pub adjusted_close: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataBatch {
    pub data: Vec<MarketData>,
    pub symbol: String,
    pub timeframe: String,
    pub ingestion_timestamp: DateTime<Utc>,
}