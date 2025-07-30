use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

// ================================================================================================
// COLLECTION MODELS
// ================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionResult {
    pub symbol: String,
    pub interval: Interval,
    pub collected_count: usize,
    pub processed_count: usize,
    pub cached_count: usize,
    pub collection_time: DateTime<Utc>,
    pub processing_duration_ms: u64,
    pub batch_id: String,
    pub source: DataSource,
    pub quality_score: Option<u8>,
}

impl CollectionResult {
    pub fn new(symbol: &str, interval: Interval, processed_count: usize) -> Self {
        Self {
            symbol: symbol.to_string(),
            interval,
            collected_count: processed_count,
            processed_count,
            cached_count: 0,
            collection_time: Utc::now(),
            processing_duration_ms: 0,
            batch_id: Uuid::new_v4().to_string(),
            source: DataSource::AlphaVantage,
            quality_score: None,
        }
    }
    
    pub fn empty(symbol: &str, interval: Interval) -> Self {
        Self {
            symbol: symbol.to_string(),
            interval,
            collected_count: 0,
            processed_count: 0,
            cached_count: 0,
            collection_time: Utc::now(),
            processing_duration_ms: 0,
            batch_id: Uuid::new_v4().to_string(),
            source: DataSource::None,
            quality_score: Some(0),
        }
    }
    
    pub fn cached_only(symbol: &str, interval: Interval, cached_count: usize) -> Self {
        Self {
            symbol: symbol.to_string(),
            interval,
            collected_count: 0,
            processed_count: 0,
            cached_count,
            collection_time: Utc::now(),
            processing_duration_ms: 0,
            batch_id: Uuid::new_v4().to_string(),
            source: DataSource::Cache,
            quality_score: Some(80), // Cached data has decent quality
        }
    }
    
    pub fn from_cache(symbol: &str, interval: Interval, cached_data: Vec<shared_types::MarketData>) -> Self {
        Self {
            symbol: symbol.to_string(),
            interval,
            collected_count: 0,
            processed_count: cached_data.len(),
            cached_count: cached_data.len(),
            collection_time: Utc::now(),
            processing_duration_ms: 0,
            batch_id: Uuid::new_v4().to_string(),
            source: DataSource::Cache,
            quality_score: Some(80),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataSource {
    AlphaVantage,
    Cache,
    None,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Interval {
    OneMin,
    FiveMin,
    FifteenMin,
    ThirtyMin,
    SixtyMin,
    Daily,
}

impl Interval {
    pub fn as_str(&self) -> &'static str {
        match self {
            Interval::OneMin => "1min",
            Interval::FiveMin => "5min",
            Interval::FifteenMin => "15min",
            Interval::ThirtyMin => "30min",
            Interval::SixtyMin => "60min",
            Interval::Daily => "daily",
        }
    }
    
    pub fn as_alpha_vantage_param(&self) -> &'static str {
        match self {
            Interval::OneMin => "1min",
            Interval::FiveMin => "5min",
            Interval::FifteenMin => "15min",
            Interval::ThirtyMin => "30min",
            Interval::SixtyMin => "60min",
            Interval::Daily => "daily",
        }
    }
}

// ================================================================================================
// RAW DATA MODELS (from Alpha Vantage)
// ================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawMarketData {
    pub timestamp: DateTime<Utc>,
    pub open: String,      // Alpha Vantage returns strings
    pub high: String,
    pub low: String,
    pub close: String,
    pub volume: String,
    pub adjusted_close: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaVantageResponse {
    #[serde(rename = "Meta Data")]
    pub meta_data: MetaData,
    #[serde(flatten)]
    pub time_series: std::collections::HashMap<String, TimeSeriesEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaData {
    #[serde(rename = "1. Information")]
    pub information: String,
    #[serde(rename = "2. Symbol")]
    pub symbol: String,
    #[serde(rename = "3. Last Refreshed")]
    pub last_refreshed: String,
    #[serde(rename = "4. Interval")]
    pub interval: Option<String>,
    #[serde(rename = "5. Output Size")]
    pub output_size: Option<String>,
    #[serde(rename = "6. Time Zone")]
    pub time_zone: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesEntry {
    #[serde(rename = "1. open")]
    pub open: String,
    #[serde(rename = "2. high")]
    pub high: String,
    #[serde(rename = "3. low")]
    pub low: String,
    #[serde(rename = "4. close")]
    pub close: String,
    #[serde(rename = "5. volume")]
    pub volume: String,
    #[serde(rename = "6. adjusted close")]
    pub adjusted_close: Option<String>,
}

impl TryFrom<(&str, TimeSeriesEntry)> for RawMarketData {
    type Error = core_traits::ServiceError;
    
    fn try_from((timestamp_str, entry): (&str, TimeSeriesEntry)) -> Result<Self, Self::Error> {
        let timestamp = chrono::DateTime::parse_from_str(timestamp_str, "%Y-%m-%d %H:%M:%S")
            .or_else(|_| chrono::DateTime::parse_from_str(timestamp_str, "%Y-%m-%d"))
            .map_err(|e| core_traits::ServiceError::System {
                message: format!("Failed to parse timestamp '{}': {}", timestamp_str, e),
            })?
            .with_timezone(&chrono::Utc);
        
        Ok(Self {
            timestamp,
            open: entry.open,
            high: entry.high,
            low: entry.low,
            close: entry.close,
            volume: entry.volume,
            adjusted_close: entry.adjusted_close,
        })
    }
} 