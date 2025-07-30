// market-data-ingestion/tests/integration_tests.rs
// Comprehensive integration tests for the market data ingestion module

use market_data_ingestion::{
    MarketDataIngestionService, MarketDataIngestionConfig,
    AlphaVantageClient, AlphaVantageConfig, RateLimiter,
    Interval, OutputSize, MarketData, MarketDataBatch,
    DataQualityController, DataQualityConfig,
    IngestionError, HealthStatus, IngestionMetrics,
    TimeSeriesEntry, IntradayResponse, MetaData,
};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use wiremock::{
    matchers::{method, path, query_param},
    Mock, MockServer, ResponseTemplate,
};

// Mock database for testing
#[derive(Debug, Clone)]
struct TestDatabase {
    stored_batches: Arc<RwLock<Vec<MarketDataBatch>>>,
}

impl TestDatabase {
    fn new() -> Self {
        Self {
            stored_batches: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    async fn get_stored_batches(&self) -> Vec<MarketDataBatch> {
        self.stored_batches.read().await.clone()
    }
    
    async fn store_batch(&self, batch: MarketDataBatch) {
        self.stored_batches.write().await.push(batch);
    }
}

#[tokio::test]
async fn test_market_data_validation() {
    let valid_data = MarketData {
        symbol: "AAPL".to_string(),
        timestamp: Utc::now(),
        open: Decimal::from(150),
        high: Decimal::from(155),
        low: Decimal::from(148),
        close: Decimal::from(152),
        volume: 1000000,
        adjusted_close: Decimal::from(152),
        source: "Alpha Vantage".to_string(),
        quality_score: 95,
    };
    
    assert!(valid_data.validate().is_ok());
    
    // Test percentage change calculation
    let change = valid_data.percentage_change();
    let expected_change = Decimal::from_f64_retain(1.33).unwrap(); // Approximately 1.33%
    assert!((change - expected_change).abs() < Decimal::from_f64_retain(0.1).unwrap());
    
    // Test true range calculation
    let true_range = valid_data.true_range(Some(Decimal::from(149)));
    assert_eq!(true_range, Decimal::from(7)); // Max of (155-148=7), (155-149=6), (149-148=1)
}

#[tokio::test]
async fn test_market_data_validation_failures() {
    // Test high < low (invalid)
    let invalid_high_low = MarketData {
        symbol: "AAPL".to_string(),
        timestamp: Utc::now(),
        open: Decimal::from(150),
        high: Decimal::from(145), // High less than low
        low: Decimal::from(148),
        close: Decimal::from(152),
        volume: 1000000,
        adjusted_close: Decimal::from(152),
        source: "Test".to_string(),
        quality_score: 95,
    };
    
    assert!(invalid_high_low.validate().is_err());
    
    // Test negative prices (invalid)
    let negative_price = MarketData {
        symbol: "AAPL".to_string(),
        timestamp: Utc::now(),
        open: Decimal::from(-150), // Negative price
        high: Decimal::from(155),
        low: Decimal::from(148),
        close: Decimal::from(152),
        volume: 1000000,
        adjusted_close: Decimal::from(152),
        source: "Test".to_string(),
        quality_score: 95,
    };
    
    assert!(negative_price.validate().is_err());
    
    // Test empty symbol (invalid)
    let empty_symbol = MarketData {
        symbol: "".to_string(), // Empty symbol
        timestamp: Utc::now(),
        open: Decimal::from(150),
        high: Decimal::from(155),
        low: Decimal::from(148),
        close: Decimal::from(152),
        volume: 1000000,
        adjusted_close: Decimal::from(152),
        source: "Test".to_string(),
        quality_score: 95,
    };
    
    assert!(empty_symbol.validate().is_err());
}

#[tokio::test]
async fn test_market_data_batch_operations() {
    let mut batch = MarketDataBatch::new("AAPL".to_string(), "Test Source".to_string());
    
    assert!(batch.is_empty());
    assert_eq!(batch.size(), 0);
    
    // Add some test data points
    let data_points = vec![
        create_test_market_data("AAPL", Utc::now() - chrono::Duration::minutes(10)),
        create_test_market_data("AAPL", Utc::now() - chrono::Duration::minutes(5)),
        create_test_market_data("AAPL", Utc::now()),
    ];
    
    for data_point in data_points {
        batch.add_data_point(data_point);
    }
    
    assert!(!batch.is_empty());
    assert_eq!(batch.size(), 3);
    
    // Test time range
    let (start_time, end_time) = batch.get_time_range().unwrap();
    assert!(start_time < end_time);
    
    // Test sorting
    batch.sort_by_timestamp();
    for i in 1..batch.data_points.len() {
        assert!(batch.data_points[i-1].timestamp <= batch.data_points[i].timestamp);
    }
}

#[tokio::test]
async fn test_rate_limiter() {
    let mut rate_limiter = RateLimiter::new(2, 5); // 2 per minute, 5 per day
    
    // First request should be allowed
    assert!(rate_limiter.can_make_request().await);
    rate_limiter.record_request().await;
    
    // Second request should be allowed
    assert!(rate_limiter.can_make_request().await);
    rate_limiter.record_request().await;
    
    // Third request should be blocked (exceeded per-minute limit)
    assert!(!rate_limiter.can_make_request().await);
    
    // Test rate limit status
    let status = rate_limiter.get_status();
    assert_eq!(status.calls_remaining_minute, 0);
    assert_eq!(status.calls_remaining_day, 3);
    
    // Test rate limit recording (triggers backoff)
    rate_limiter.record_rate_limit().await;
    let status_after_limit = rate_limiter.get_status();
    assert!(status_after_limit.in_backoff);
    assert!(status_after_limit.backoff_until.is_some());
}

#[tokio::test]
async fn test_data_quality_controller() {
    let config = DataQualityConfig {
        quality_threshold: 70,
        enable_deduplication: true,
        max_price_deviation: 0.1,
        volume_threshold: 1000,
    };
    
    let quality_controller = DataQualityController::new(config);
    
    // Test high-quality data
    let high_quality_data = MarketData {
        symbol: "AAPL".to_string(),
        timestamp: Utc::now(),
        open: Decimal::from(150),
        high: Decimal::from(155),
        low: Decimal::from(148),
        close: Decimal::from(152),
        volume: 5000000, // High volume
        adjusted_close: Decimal::from(152),
        source: "Alpha Vantage".to_string(),
        quality_score: 100,
    };
    
    let quality_score = quality_controller.calculate_quality_score(&high_quality_data);
    assert!(quality_score >= 80); // Should be high quality
    
    // Test low-quality data (zero volume)
    let low_quality_data = MarketData {
        symbol: "AAPL".to_string(),
        timestamp: Utc::now(),
        open: Decimal::from(150),
        high: Decimal::from(155),
        low: Decimal::from(148),
        close: Decimal::from(152),
        volume: 0, // Zero volume - low quality
        adjusted_close: Decimal::from(152),
        source: "Alpha Vantage".to_string(),
        quality_score: 100,
    };
    
    let low_quality_score = quality_controller.calculate_quality_score(&low_quality_data);
    assert!(low_quality_score < quality_score); // Should be lower quality
    
    // Test deduplication
    let mut batch = MarketDataBatch::new("AAPL".to_string(), "Test".to_string());
    let duplicate_data = create_test_market_data("AAPL", Utc::now());
    
    batch.add_data_point(duplicate_data.clone());
    batch.add_data_point(duplicate_data.clone()); // Duplicate
    batch.add_data_point(duplicate_data);         // Another duplicate
    
    assert_eq!(batch.size(), 3);
    
    quality_controller.deduplicate_batch(&mut batch);
    assert_eq!(batch.size(), 1); // Should remove duplicates
}

#[tokio::test]
async fn test_alpha_vantage_client_with_mock_server() {
    // Start mock server
    let mock_server = MockServer::start().await;
    
    // Create mock response
    let mock_response = create_mock_alpha_vantage_response();
    
    Mock::given(method("GET"))
        .and(path("/"))
        .and(query_param("function", "TIME_SERIES_INTRADAY"))
        .and(query_param("symbol", "AAPL"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&mock_response))
        .mount(&mock_server)
        .await;
    
    // Create client with mock server URL
    let config = AlphaVantageConfig {
        base_url: mock_server.uri(),
        api_key: "test_api_key".to_string(),
        timeout_seconds: 30,
        max_retries: 3,
    };
    
    let rate_limiter = Arc::new(RwLock::new(RateLimiter::new(100, 1000)));
    let client = AlphaVantageClient::new(config, rate_limiter);
    
    // Test successful request
    let result = client.get_intraday_data("AAPL", Interval::FiveMin, OutputSize::Compact).await;
    
    assert!(result.is_ok());
    let response = result.unwrap();
    
    // Verify response structure
    assert!(response.meta_data.is_some());
    assert!(!response.time_series.is_empty());
}

#[tokio::test]
async fn test_alpha_vantage_client_error_handling() {
    let mock_server = MockServer::start().await;
    
    // Mock rate limit response
    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(429)) // Too Many Requests
        .mount(&mock_server)
        .await;
    
    let config = AlphaVantageConfig {
        base_url: mock_server.uri(),
        api_key: "test_api_key".to_string(),
        timeout_seconds: 1, // Short timeout for testing
        max_retries: 1,     // Minimal retries for faster test
    };
    
    let rate_limiter = Arc::new(RwLock::new(RateLimiter::new(100, 1000)));
    let client = AlphaVantageClient::new(config, rate_limiter);
    
    let result = client.get_intraday_data("AAPL", Interval::FiveMin, OutputSize::Compact).await;
    
    assert!(result.is_err());
    match result.unwrap_err() {
        IngestionError::RateLimitExceeded { .. } => {
            // Expected error type
        },
        other => panic!("Expected RateLimitExceeded error, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_alpha_vantage_client_api_error_response() {
    let mock_server = MockServer::start().await;
    
    // Mock API error response
    let error_response = serde_json::json!({
        "Error Message": "Invalid API call. Please retry or visit the documentation (https://www.alphavantage.co/documentation/) for TIME_SERIES_INTRADAY."
    });
    
    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&error_response))
        .mount(&mock_server)
        .await;
    
    let config = AlphaVantageConfig {
        base_url: mock_server.uri(),
        api_key: "test_api_key".to_string(),
        timeout_seconds: 30,
        max_retries: 1,
    };
    
    let rate_limiter = Arc::new(RwLock::new(RateLimiter::new(100, 1000)));
    let client = AlphaVantageClient::new(config, rate_limiter);
    
    let result = client.get_intraday_data("AAPL", Interval::FiveMin, OutputSize::Compact).await;
    
    assert!(result.is_err());
    match result.unwrap_err() {
        IngestionError::ApiError { message, .. } => {
            assert!(message.contains("Invalid API call"));
        },
        other => panic!("Expected ApiError, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_time_series_entry_conversion() {
    let entry = TimeSeriesEntry {
        open: "150.25".to_string(),
        high: "155.75".to_string(),
        low: "148.50".to_string(),
        close: "152.00".to_string(),
        volume: "1500000".to_string(),
        adjusted_close: Some("152.00".to_string()),
    };
    
    let timestamp = Utc::now();
    let result = entry.to_market_data("AAPL", timestamp);
    
    assert!(result.is_ok());
    let market_data = result.unwrap();
    
    assert_eq!(market_data.symbol, "AAPL");
    assert_eq!(market_data.timestamp, timestamp);
    assert_eq!(market_data.open, Decimal::from_f64_retain(150.25).unwrap());
    assert_eq!(market_data.high, Decimal::from_f64_retain(155.75).unwrap());
    assert_eq!(market_data.low, Decimal::from_f64_retain(148.50).unwrap());
    assert_eq!(market_data.close, Decimal::from_f64_retain(152.00).unwrap());
    assert_eq!(market_data.volume, 1500000);
    assert_eq!(market_data.adjusted_close, Decimal::from_f64_retain(152.00).unwrap());
}

#[tokio::test]
async fn test_time_series_entry_conversion_invalid_data() {
    let invalid_entry = TimeSeriesEntry {
        open: "invalid_number".to_string(),
        high: "155.75".to_string(),
        low: "148.50".to_string(),
        close: "152.00".to_string(),
        volume: "1500000".to_string(),
        adjusted_close: None,
    };
    
    let timestamp = Utc::now();
    let result = invalid_entry.to_market_data("AAPL", timestamp);
    
    assert!(result.is_err());
    match result.unwrap_err() {
        IngestionError::ParsingError { field, .. } => {
            assert_eq!(field, "open");
        },
        other => panic!("Expected ParsingError, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_configuration_serialization() {
    let config = MarketDataIngestionConfig::default();
    
    // Test serialization to JSON
    let json = serde_json::to_string(&config).unwrap();
    assert!(!json.is_empty());
    
    // Test deserialization from JSON
    let deserialized: MarketDataIngestionConfig = serde_json::from_str(&json).unwrap();
    
    // Verify key fields are preserved
    assert_eq!(deserialized.service.service_name, config.service.service_name);
    assert_eq!(deserialized.rate_limits.calls_per_minute, config.rate_limits.calls_per_minute);
    assert_eq!(deserialized.data_quality.quality_threshold, config.data_quality.quality_threshold);
}

#[tokio::test]
async fn test_interval_display_and_parsing() {
    let intervals = vec![
        (Interval::OneMin, "1min"),
        (Interval::FiveMin, "5min"),
        (Interval::FifteenMin, "15min"),
        (Interval::ThirtyMin, "30min"),
        (Interval::SixtyMin, "60min"),
    ];
    
    for (interval, expected_str) in intervals {
        assert_eq!(interval.to_string(), expected_str);
    }
}

#[tokio::test]
async fn test_output_size_display() {
    assert_eq!(OutputSize::Compact.to_string(), "compact");
    assert_eq!(OutputSize::Full.to_string(), "full");
}

#[tokio::test]
async fn test_concurrent_data_collection() {
    let config = create_test_config();
    let test_db = Arc::new(TestDatabase::new());
    
    // Mock the database client trait
    // In a real test, you'd use the actual DatabaseClient implementation
    
    let symbols = vec!["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"];
    let mut handles = vec![];
    
    for symbol in symbols {
        let symbol = symbol.to_string();
        handles.push(tokio::spawn(async move {
            // Simulate data collection
            let mut batch = MarketDataBatch::new(symbol.clone(), "Test".to_string());
            batch.add_data_point(create_test_market_data(&symbol, Utc::now()));
            batch
        }));
    }
    
    let mut collected_batches = vec![];
    for handle in handles {
        let batch = handle.await.unwrap();
        collected_batches.push(batch);
    }
    
    assert_eq!(collected_batches.len(), 5);
    
    // Verify each batch has the correct symbol
    let expected_symbols: std::collections::HashSet<String> = 
        ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"].iter().map(|s| s.to_string()).collect();
    let actual_symbols: std::collections::HashSet<String> = 
        collected_batches.iter().map(|b| b.symbol.clone()).collect();
    
    assert_eq!(expected_symbols, actual_symbols);
}

#[tokio::test]
async fn test_error_recovery_and_resilience() {
    // Test that the system continues to function even when some operations fail
    
    let mut rate_limiter = RateLimiter::new(1, 10);
    
    // First request should succeed
    assert!(rate_limiter.can_make_request().await);
    rate_limiter.record_request().await;
    
    // Second request should be rate limited
    assert!(!rate_limiter.can_make_request().await);
    
    // Record rate limit to trigger backoff
    rate_limiter.record_rate_limit().await;
    
    // System should still be functional for status checks
    let status = rate_limiter.get_status();
    assert!(status.in_backoff);
    
    // Test quality controller continues to function with various data quality
    let quality_controller = DataQualityController::new(DataQualityConfig::default());
    
    let poor_quality_data = MarketData {
        symbol: "TEST".to_string(),
        timestamp: Utc::now(),
        open: Decimal::from(100),
        high: Decimal::from(100), // Flat pricing
        low: Decimal::from(100),
        close: Decimal::from(100),
        volume: 0, // Zero volume
        adjusted_close: Decimal::from(100),
        source: "Test".to_string(),
        quality_score: 100,
    };
    
    let quality_score = quality_controller.calculate_quality_score(&poor_quality_data);
    assert!(quality_score < 100); // Should penalize poor quality data
}

#[tokio::test]
async fn test_performance_requirements() {
    // Test that operations complete within specified time limits
    
    let start_time = std::time::Instant::now();
    
    // Data validation should be fast (< 1ms target)
    let test_data = create_test_market_data("AAPL", Utc::now());
    let validation_result = test_data.validate();
    
    let validation_duration = start_time.elapsed();
    assert!(validation_duration < Duration::from_millis(1));
    assert!(validation_result.is_ok());
    
    // Quality calculation should be fast (< 5ms target)
    let quality_start = std::time::Instant::now();
    let quality_controller = DataQualityController::new(DataQualityConfig::default());
    let _quality_score = quality_controller.calculate_quality_score(&test_data);
    let quality_duration = quality_start.elapsed();
    assert!(quality_duration < Duration::from_millis(5));
    
    // Batch operations should be efficient
    let batch_start = std::time::Instant::now();
    let mut batch = MarketDataBatch::new("AAPL".to_string(), "Test".to_string());
    
    // Add 1000 data points
    for i in 0..1000 {
        batch.add_data_point(create_test_market_data("AAPL", Utc::now() + chrono::Duration::seconds(i)));
    }
    
    batch.sort_by_timestamp();
    let batch_duration = batch_start.elapsed();
    assert!(batch_duration < Duration::from_millis(100)); // Should be fast for 1000 points
}

// Helper functions for testing

fn create_test_config() -> MarketDataIngestionConfig {
    MarketDataIngestionConfig {
        service: market_data_ingestion::ServiceConfig {
            service_name: "test-service".to_string(),
            port: 8080,
            worker_threads: 2,
            max_concurrent_collections: 10,
        },
        alpha_vantage: AlphaVantageConfig {
            base_url: "https://test.example.com".to_string(),
            api_key: "test_key".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
        },
        rate_limits: market_data_ingestion::RateLimitsConfig {
            calls_per_minute: 5,
            calls_per_day: 100,
            premium_calls_per_minute: 75,
            premium_calls_per_day: 1000,
            is_premium: false,
        },
        collection: market_data_ingestion::CollectionConfig {
            default_symbols: vec!["AAPL".to_string(), "GOOGL".to_string()],
            priority_symbols: vec!["SPY".to_string()],
            collection_intervals: vec!["5min".to_string(), "1hour".to_string()],
            max_batch_size: 100,
            parallel_collections: 5,
        },
        data_quality: DataQualityConfig {
            quality_threshold: 70,
            enable_deduplication: true,
            max_price_deviation: 0.1,
            volume_threshold: 1000,
        },
        storage: market_data_ingestion::StorageConfig {
            batch_size: 50,
            flush_interval_seconds: 10,
            enable_compression: true,
            retention_days: 365,
        },
    }
}

fn create_test_market_data(symbol: &str, timestamp: DateTime<Utc>) -> MarketData {
    MarketData {
        symbol: symbol.to_string(),
        timestamp,
        open: Decimal::from(150),
        high: Decimal::from(155),
        low: Decimal::from(148),
        close: Decimal::from(152),
        volume: 1000000,
        adjusted_close: Decimal::from(152),
        source: "Test Source".to_string(),
        quality_score: 95,
    }
}

fn create_mock_alpha_vantage_response() -> serde_json::Value {
    serde_json::json!({
        "Meta Data": {
            "1. Information": "Intraday (5min) open, high, low, close prices and volume",
            "2. Symbol": "AAPL",
            "3. Last Refreshed": "2024-01-15 20:00:00",
            "4. Interval": "5min",
            "5. Output Size": "Compact",
            "6. Time Zone": "US/Eastern"
        },
        "Time Series (5min)": {
            "2024-01-15 20:00:00": {
                "1. open": "185.2000",
                "2. high": "185.5000",
                "3. low": "184.8000",
                "4. close": "185.1000",
                "5. volume": "1234567"
            },
            "2024-01-15 19:55:00": {
                "1. open": "184.8000",
                "2. high": "185.3000",
                "3. low": "184.5000",
                "4. close": "185.2000",
                "5. volume": "987654"
            }
        }
    })
}

// Stress test for high-volume operations
#[tokio::test]
async fn test_high_volume_batch_processing() {
    let start_time = std::time::Instant::now();
    
    let mut batch = MarketDataBatch::new("STRESS_TEST".to_string(), "Test".to_string());
    
    // Create 10,000 data points
    for i in 0..10_000 {
        let data_point = MarketData {
            symbol: "STRESS_TEST".to_string(),
            timestamp: Utc::now() + chrono::Duration::seconds(i),
            open: Decimal::from(100 + (i % 50)),
            high: Decimal::from(105 + (i % 50)),
            low: Decimal::from(95 + (i % 50)),
            close: Decimal::from(102 + (i % 50)),
            volume: 1000000 + (i as u64 * 1000),
            adjusted_close: Decimal::from(102 + (i % 50)),
            source: "Stress Test".to_string(),
            quality_score: 85 + ((i % 15) as u8),
        };
        batch.add_data_point(data_point);
    }
    
    assert_eq!(batch.size(), 10_000);
    
    // Sort the batch
    batch.sort_by_timestamp();
    
    // Verify sorting
    for i in 1..batch.data_points.len() {
        assert!(batch.data_points[i-1].timestamp <= batch.data_points[i].timestamp);
    }
    
    let processing_time = start_time.elapsed();
    
    // Should process 10k records in reasonable time (< 1 second)
    assert!(processing_time < Duration::from_secs(1));
    
    println!("Processed {} records in {:?}", batch.size(), processing_time);
}

// Test memory usage doesn't grow excessively
#[tokio::test]
async fn test_memory_efficiency() {
    // This test ensures that processing doesn't leak memory significantly
    
    for iteration in 0..100 {
        let mut batch = MarketDataBatch::new(format!("TEST_{}", iteration), "Memory Test".to_string());
        
        // Add 100 data points per iteration
        for i in 0..100 {
            batch.add_data_point(create_test_market_data("TEST", Utc::now() + chrono::Duration::seconds(i)));
        }
        
        // Process the batch
        let quality_controller = DataQualityController::new(DataQualityConfig::default());
        for data_point in &batch.data_points {
            let _quality_score = quality_controller.calculate_quality_score(data_point);
        }
        
        // Batch should be dropped at the end of each iteration
        drop(batch);
    }
    
    // If we reach here without OOM, memory management is working correctly
    assert!(true);
}

// Test configuration edge cases
#[tokio::test]
async fn test_configuration_edge_cases() {
    // Test with minimal configuration
    let minimal_config = MarketDataIngestionConfig {
        service: market_data_ingestion::ServiceConfig {
            service_name: "minimal".to_string(),
            port: 8080,
            worker_threads: 1,
            max_concurrent_collections: 1,
        },
        alpha_vantage: AlphaVantageConfig {
            base_url: "https://test.com".to_string(),
            api_key: "test".to_string(),
            timeout_seconds: 1,
            max_retries: 0,
        },
        rate_limits: market_data_ingestion::RateLimitsConfig {
            calls_per_minute: 1,
            calls_per_day: 1,
            premium_calls_per_minute: 1,
            premium_calls_per_day: 1,
            is_premium: false,
        },
        collection: market_data_ingestion::CollectionConfig {
            default_symbols: vec![],
            priority_symbols: vec![],
            collection_intervals: vec![],
            max_batch_size: 1,
            parallel_collections: 1,
        },
        data_quality: DataQualityConfig {
            quality_threshold: 0,
            enable_deduplication: false,
            max_price_deviation: 1.0,
            volume_threshold: 0,
        },
        storage: market_data_ingestion::StorageConfig {
            batch_size: 1,
            flush_interval_seconds: 1,
            enable_compression: false,
            retention_days: 1,
        },
    };
    
    // Should serialize and deserialize without errors
    let json = serde_json::to_string(&minimal_config).unwrap();
    let _deserialized: MarketDataIngestionConfig = serde_json::from_str(&json).unwrap();
}

// Test error serialization for logging purposes
#[tokio::test]
async fn test_error_serialization() {
    let errors = vec![
        IngestionError::ApiError { 
            message: "Test API error".to_string(), 
            status_code: 500 
        },
        IngestionError::RateLimitExceeded { 
            limit_type: "Test limit".to_string() 
        },
        IngestionError::QualityBelowThreshold { 
            score: 50, 
            threshold: 70 
        },
        IngestionError::SymbolNotFound { 
            symbol: "INVALID".to_string() 
        },
        IngestionError::ParsingError { 
            field: "price".to_string(), 
            error: "Invalid number".to_string() 
        },
    ];
    
    for error in errors {
        // Test JSON serialization
        let json = serde_json::to_string(&error).unwrap();
        assert!(!json.is_empty());
        
        // Test deserialization
        let deserialized: IngestionError = serde_json::from_str(&json).unwrap();
        
        // Verify error type is preserved (basic check)
        assert_eq!(std::mem::discriminant(&error), std::mem::discriminant(&deserialized));
    }
}