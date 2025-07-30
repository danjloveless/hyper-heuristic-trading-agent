use serde::{Deserialize, Serialize};

// ================================================================================================
// SERVICE CONFIGURATION STRUCTURES
// ================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaVantageConfig {
    pub base_url: String,
    pub api_key: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    pub default_output_size: String,
}

impl Default for AlphaVantageConfig {
    fn default() -> Self {
        Self {
            base_url: "https://www.alphavantage.co/query".to_string(),
            api_key: "YOUR_API_KEY".to_string(), // Placeholder for actual API key
            timeout_seconds: 30,
            max_retries: 3,
            default_output_size: "compact".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitsConfig {
    pub calls_per_minute: u32,
    pub calls_per_day: u32,
    pub is_premium: bool,
    pub burst_allowance: u32,
}

impl Default for RateLimitsConfig {
    fn default() -> Self {
        Self {
            calls_per_minute: 5,    // Free tier limit
            calls_per_day: 500,     // Free tier limit
            is_premium: false,
            burst_allowance: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub default_symbols: Vec<String>,
    pub priority_symbols: Vec<String>,
    pub batch_size: usize,
    pub concurrent_collections: usize,
    pub data_quality: DataQualityConfig,
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
            ],
            batch_size: 100,
            concurrent_collections: 5,
            data_quality: DataQualityConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]  
pub struct DataQualityConfig {
    pub min_quality_score: u8,
    pub enable_validation: bool,
    pub max_price_deviation_percent: f64,
    pub min_volume_threshold: u64,
    pub enable_deduplication: bool,
}

impl Default for DataQualityConfig {
    fn default() -> Self {
        Self {
            min_quality_score: 70,
            enable_validation: true,
            max_price_deviation_percent: 10.0,
            min_volume_threshold: 1000,
            enable_deduplication: true,
        }
    }
} 