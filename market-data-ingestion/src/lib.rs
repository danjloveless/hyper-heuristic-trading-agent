// market-data-ingestion/src/lib.rs
// QuantumTrade AI - Market Data Ingestion Module
// Complete implementation following the technical specification

// Module declarations
pub mod errors;
pub mod models;
pub mod config;
pub mod alpha_vantage;
pub mod rate_limiter;
pub mod data_quality;
pub mod batch_processor;
pub mod scheduler;
pub mod health;
pub mod metrics;
pub mod service;

// Re-export main types for convenience
pub use errors::{Result, IngestionError};
pub use models::{MarketData, MarketDataBatch};
pub use config::{
    MarketDataIngestionConfig, ServiceConfig, AlphaVantageConfig, 
    RateLimitsConfig, CollectionConfig, DataQualityConfig, StorageConfig
};
pub use alpha_vantage::{AlphaVantageClient, Interval, OutputSize};
pub use rate_limiter::{RateLimiter, RateLimitStatus};
pub use data_quality::DataQualityController;
pub use batch_processor::BatchProcessor;
pub use scheduler::IngestionScheduler;
pub use health::{HealthChecker, HealthStatus};
pub use metrics::{MetricsCollector, IngestionMetrics};
pub use service::MarketDataIngestionService;

// ================================================================================================
// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_data_validation() {
        let valid_data = MarketData {
            symbol: "AAPL".to_string(),
            timestamp: chrono::Utc::now(),
            open: rust_decimal::Decimal::from(100),
            high: rust_decimal::Decimal::from(105),
            low: rust_decimal::Decimal::from(95),
            close: rust_decimal::Decimal::from(102),
            volume: 1000000,
            adjusted_close: rust_decimal::Decimal::from(102),
            source: "Test".to_string(),
            quality_score: 100,
        };
        
        assert!(valid_data.validate().is_ok());
        
        let invalid_data = MarketData {
            symbol: "AAPL".to_string(),
            timestamp: chrono::Utc::now(),
            open: rust_decimal::Decimal::from(100),
            high: rust_decimal::Decimal::from(90), // High less than low - invalid
            low: rust_decimal::Decimal::from(95),
            close: rust_decimal::Decimal::from(102),
            volume: 1000000,
            adjusted_close: rust_decimal::Decimal::from(102),
            source: "Test".to_string(),
            quality_score: 100,
        };
        
        assert!(invalid_data.validate().is_err());
    }
    
    #[test]
    fn test_market_data_calculations() {
        let data = MarketData {
            symbol: "AAPL".to_string(),
            timestamp: chrono::Utc::now(),
            open: rust_decimal::Decimal::from(100),
            high: rust_decimal::Decimal::from(105),
            low: rust_decimal::Decimal::from(95),
            close: rust_decimal::Decimal::from(110),
            volume: 1000000,
            adjusted_close: rust_decimal::Decimal::from(110),
            source: "Test".to_string(),
            quality_score: 100,
        };
        
        let percentage_change = data.percentage_change();
        assert_eq!(percentage_change, rust_decimal::Decimal::from(10)); // 10% gain
        
        let true_range = data.true_range(Some(rust_decimal::Decimal::from(98)));
        assert_eq!(true_range, rust_decimal::Decimal::from(12)); // Max of 10, 7, 3
    }
    
    #[test]
    fn test_interval_display() {
        assert_eq!(Interval::OneMin.to_string(), "1min");
        assert_eq!(Interval::FiveMin.to_string(), "5min");
        assert_eq!(Interval::FifteenMin.to_string(), "15min");
        assert_eq!(Interval::ThirtyMin.to_string(), "30min");
        assert_eq!(Interval::SixtyMin.to_string(), "60min");
    }
    
    #[tokio::test]
    async fn test_rate_limiter() {
        let mut rate_limiter = RateLimiter::new(2, 10);
        
        // Should allow first requests
        assert!(rate_limiter.can_make_request().await);
        rate_limiter.record_request().await;
        
        assert!(rate_limiter.can_make_request().await);
        rate_limiter.record_request().await;
        
        // Should block third request within the minute
        assert!(!rate_limiter.can_make_request().await);
        
        // Test rate limit recording
        rate_limiter.record_rate_limit().await;
        assert!(rate_limiter.backoff_until.is_some());
    }
    
    #[test]
    fn test_alpha_vantage_endpoint_params() {
        let endpoint = alpha_vantage::AlphaVantageEndpoint::IntradayData {
            symbol: "AAPL".to_string(),
            interval: Interval::FiveMin,
            outputsize: OutputSize::Compact,
        };
        
        let params = endpoint.to_query_params();
        assert_eq!(params.get("function"), Some(&"TIME_SERIES_INTRADAY".to_string()));
        assert_eq!(params.get("symbol"), Some(&"AAPL".to_string()));
        assert_eq!(params.get("interval"), Some(&"5min".to_string()));
        assert_eq!(params.get("outputsize"), Some(&"compact".to_string()));
    }
}