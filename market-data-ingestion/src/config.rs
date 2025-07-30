use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataIngestionConfig {
    pub service: ServiceConfig,
    pub alpha_vantage: AlphaVantageConfig,
    pub rate_limits: RateLimitsConfig,
    pub collection: CollectionConfig,
    pub data_quality: DataQualityConfig,
    pub storage: StorageConfig,
}

impl Default for MarketDataIngestionConfig {
    fn default() -> Self {
        Self {
            service: ServiceConfig::default(),
            alpha_vantage: AlphaVantageConfig::default(),
            rate_limits: RateLimitsConfig::default(),
            collection: CollectionConfig::default(),
            data_quality: DataQualityConfig::default(),
            storage: StorageConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    pub service_name: String,
    pub port: u16,
    pub worker_threads: usize,
    pub max_concurrent_collections: usize,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            service_name: "market-data-ingestion".to_string(),
            port: 8080,
            worker_threads: 4,
            max_concurrent_collections: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaVantageConfig {
    pub base_url: String,
    pub api_key: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
}

impl Default for AlphaVantageConfig {
    fn default() -> Self {
        Self {
            base_url: "https://www.alphavantage.co/query".to_string(),
            api_key: std::env::var("ALPHA_VANTAGE_API_KEY").unwrap_or_default(),
            timeout_seconds: 30,
            max_retries: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitsConfig {
    pub calls_per_minute: u32,
    pub calls_per_day: u32,
    pub premium_calls_per_minute: u32,
    pub premium_calls_per_day: u32,
    pub is_premium: bool,
}

impl Default for RateLimitsConfig {
    fn default() -> Self {
        Self {
            calls_per_minute: 5,
            calls_per_day: 500,
            premium_calls_per_minute: 75,
            premium_calls_per_day: 75000,
            is_premium: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub default_symbols: Vec<String>,
    pub priority_symbols: Vec<String>,
    pub collection_intervals: Vec<String>,
    pub max_batch_size: usize,
    pub parallel_collections: usize,
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            default_symbols: vec![
                "AAPL".to_string(),
                "GOOGL".to_string(),
                "MSFT".to_string(),
                "AMZN".to_string(),
                "TSLA".to_string(),
            ],
            priority_symbols: vec![
                "SPY".to_string(),
                "QQQ".to_string(),
                "IWM".to_string(),
            ],
            collection_intervals: vec![
                "1min".to_string(),
                "5min".to_string(),
                "15min".to_string(),
                "1hour".to_string(),
                "1day".to_string(),
            ],
            max_batch_size: 1000,
            parallel_collections: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityConfig {
    pub quality_threshold: u8,
    pub enable_deduplication: bool,
    pub max_price_deviation: f64,
    pub volume_threshold: u64,
}

impl Default for DataQualityConfig {
    fn default() -> Self {
        Self {
            quality_threshold: 70,
            enable_deduplication: true,
            max_price_deviation: 0.1, // 10% max deviation
            volume_threshold: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub batch_size: usize,
    pub flush_interval_seconds: u64,
    pub enable_compression: bool,
    pub retention_days: u32,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            flush_interval_seconds: 30,
            enable_compression: true,
            retention_days: 365,
        }
    }
} 