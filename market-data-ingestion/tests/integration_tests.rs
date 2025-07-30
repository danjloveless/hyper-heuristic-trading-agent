//! Integration tests for the market data ingestion service
//! These tests verify that all components work together correctly

use market_data_ingestion::*;
use core_traits::*;
use database_abstraction::{DatabaseManager, DatabaseClient};
use std::sync::Arc;
use wiremock::{MockServer, Mock, ResponseTemplate};
use wiremock::matchers::{method, path, query_param};

#[tokio::test]
async fn test_complete_service_integration() {
    // Setup mock Alpha Vantage server
    let mock_server = MockServer::start().await;
    
    Mock::given(method("GET"))
        .and(path("/query"))
        .and(query_param("function", "TIME_SERIES_INTRADAY"))
        .and(query_param("symbol", "AAPL"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_mock_alpha_vantage_response()))
        .mount(&mock_server)
        .await;
    
    // Create test infrastructure
    let config_provider = Arc::new(create_test_config_provider(&mock_server.uri()));
    let database_manager = Arc::new(create_test_database_manager().await);
    let error_handler = Arc::new(TestErrorHandler::new());
    let monitoring = Arc::new(TestMonitoringProvider::new());
    
    // Create service
    let service = market_data_ingestion::create_service(
        config_provider,
        database_manager.clone(),
        error_handler,
        monitoring,
    ).await.expect("Service creation should succeed");
    
    // Start service
    service.start().await.expect("Service start should succeed");
    
    // Test health check
    let health = service.health_check().await;
    assert_eq!(health.status, ServiceStatus::Healthy);
    assert!(health.checks.contains_key("clickhouse"));
    assert!(health.checks.contains_key("redis"));
    
    // Test data collection
    let result = service.collect_symbol_data("AAPL", Interval::FiveMin).await
        .expect("Data collection should succeed");
    
    assert!(result.processed_count > 0);
    assert!(result.quality_score.unwrap_or(0) > 0);
    
    // Verify data was stored in database
    let clickhouse = database_manager.clickhouse();
    let stored_data = clickhouse.get_market_data(
        "AAPL",
        chrono::Utc::now() - chrono::Duration::hours(1),
        chrono::Utc::now(),
    ).await.expect("Should retrieve stored data");
    
    assert!(!stored_data.is_empty());
}

#[tokio::test]
async fn test_error_handling_integration() {
    // Setup mock server that returns errors
    let mock_server = MockServer::start().await;
    
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(429)) // Rate limit error
        .mount(&mock_server)
        .await;
    
    let config_provider = Arc::new(create_test_config_provider(&mock_server.uri()));
    let database_manager = Arc::new(create_test_database_manager().await);
    let error_handler = Arc::new(TestErrorHandler::new());
    let monitoring = Arc::new(TestMonitoringProvider::new());
    
    let service = market_data_ingestion::create_service(
        config_provider,
        database_manager,
        error_handler,
        monitoring,
    ).await.expect("Service creation should succeed");
    
    // This should trigger error handling
    let result = service.collect_symbol_data("AAPL", Interval::FiveMin).await;
    
    // Should either retry and succeed, use cache, or fail gracefully
    match result {
        Ok(_) => {
            // Success after retry or from cache
            println!("Service recovered from error successfully");
        }
        Err(ServiceError::RateLimit { .. }) => {
            // Expected rate limit error
            println!("Rate limit error handled correctly");
        }
        Err(e) => {
            panic!("Unexpected error type: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_data_quality_validation() {
    let mock_server = MockServer::start().await;
    
    // Mock response with potentially problematic data
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_mock_quality_test_response()))
        .mount(&mock_server)
        .await;
    
    let config_provider = Arc::new(create_test_config_provider(&mock_server.uri()));
    let database_manager = Arc::new(create_test_database_manager().await);
    let error_handler = Arc::new(TestErrorHandler::new());
    let monitoring = Arc::new(TestMonitoringProvider::new());
    
    let service = market_data_ingestion::create_service(
        config_provider,
        database_manager,
        error_handler,
        monitoring,
    ).await.expect("Service creation should succeed");
    
    let result = service.collect_symbol_data("TEST", Interval::FiveMin).await
        .expect("Data collection should succeed");
    
    // Should have lower quality score due to data issues
    assert!(result.quality_score.unwrap_or(100) < 100);
}

// Helper functions for testing
fn create_mock_alpha_vantage_response() -> serde_json::Value {
    serde_json::json!({
        "Meta Data": {
            "1. Information": "Intraday (5min) open, high, low, close prices and volume",
            "2. Symbol": "AAPL",
            "3. Last Refreshed": "2024-01-15 16:00:00",
            "4. Interval": "5min",
            "5. Output Size": "Compact",
            "6. Time Zone": "US/Eastern"
        },
        "Time Series (5min)": {
            "2024-01-15 16:00:00": {
                "1. open": "150.0000",
                "2. high": "151.0000",
                "3. low": "149.5000",
                "4. close": "150.5000",
                "5. volume": "1000000"
            },
            "2024-01-15 15:55:00": {
                "1. open": "149.5000",
                "2. high": "150.2000",
                "3. low": "149.0000",
                "4. close": "150.0000",
                "5. volume": "950000"
            }
        }
    })
}

fn create_mock_quality_test_response() -> serde_json::Value {
    serde_json::json!({
        "Meta Data": {
            "1. Information": "Intraday (5min) open, high, low, close prices and volume",
            "2. Symbol": "TEST",
            "3. Last Refreshed": "2024-01-15 16:00:00",
            "4. Interval": "5min",
            "5. Output Size": "Compact",
            "6. Time Zone": "US/Eastern"
        },
        "Time Series (5min)": {
            "2024-01-15 16:00:00": {
                "1. open": "150.0000",
                "2. high": "149.0000", // High < Low (data quality issue)
                "3. low": "149.5000",
                "4. close": "150.5000",
                "5. volume": "500" // Low volume (below threshold)
            }
        }
    })
}

fn create_test_config_provider(base_url: &str) -> TestConfigurationProvider {
    TestConfigurationProvider::new(base_url)
}

async fn create_test_database_manager() -> database_abstraction::DatabaseManager {
    // Create in-memory or test database configuration
    let config = database_abstraction::DatabaseConfig {
        clickhouse: database_abstraction::ClickHouseConfig {
            url: "http://localhost:8123".to_string(),
            database: "quantumtrade".to_string(), // Use the actual database name
            username: Some("default".to_string()),
            password: None,
            connection_timeout: std::time::Duration::from_secs(5),
            query_timeout: std::time::Duration::from_secs(10),
            max_connections: 5,
            retry_attempts: 1,
        },
        redis: database_abstraction::RedisConfig {
            url: "redis://localhost:6379".to_string(),
            pool_size: 2,
            connection_timeout: std::time::Duration::from_secs(5),
            default_ttl: std::time::Duration::from_secs(300),
            max_connections: 5,
        },
    };
    
    database_abstraction::DatabaseManager::new(config).await
        .expect("Failed to create test database manager")
}

// Test implementations
struct TestConfigurationProvider {
    base_url: String,
}

impl TestConfigurationProvider {
    fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
        }
    }
}

#[async_trait::async_trait]
impl ConfigurationProvider for TestConfigurationProvider {
    async fn get_string(&self, key: &str) -> ServiceResult<String> {
        match key {
            "alpha_vantage.base_url" => Ok(self.base_url.clone()),
            _ => Err(ServiceError::Configuration {
                message: format!("Unknown config key: {}", key),
            }),
        }
    }
    
    async fn get_u32(&self, _key: &str) -> ServiceResult<u32> {
        Ok(30)
    }
    
    async fn get_u64(&self, _key: &str) -> ServiceResult<u64> {
        Ok(30)
    }
    
    async fn get_bool(&self, _key: &str) -> ServiceResult<bool> {
        Ok(true)
    }
    
    async fn get_secret(&self, key: &str) -> ServiceResult<String> {
        match key {
            "alpha_vantage.api_key" => Ok("test_key".to_string()),
            _ => Err(ServiceError::Configuration {
                message: format!("Unknown secret: {}", key),
            }),
        }
    }
    
    async fn get_alpha_vantage_config(&self) -> ServiceResult<serde_json::Value> {
        let config = AlphaVantageConfig {
            base_url: self.base_url.clone(),
            timeout_seconds: 30,
            max_retries: 3,
            default_output_size: "compact".to_string(),
        };
        Ok(serde_json::to_value(config).unwrap())
    }
    
    async fn get_rate_limits_config(&self) -> ServiceResult<serde_json::Value> {
        let config = RateLimitsConfig::default();
        Ok(serde_json::to_value(config).unwrap())
    }
    
    async fn get_collection_config(&self) -> ServiceResult<serde_json::Value> {
        let config = CollectionConfig::default();
        Ok(serde_json::to_value(config).unwrap())
    }
}

struct TestErrorHandler;

impl TestErrorHandler {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl ErrorHandler for TestErrorHandler {
    async fn handle_error(&self, error: &(dyn std::error::Error + Send + Sync), _context: &ErrorContext) -> ErrorDecision {
        let error_str = error.to_string();
        
        if error_str.contains("rate limit") {
            ErrorDecision::Retry { delay: std::time::Duration::from_secs(60), max_attempts: 3 }
        } else {
            ErrorDecision::Fail
        }
    }
    
    async fn classify_error(&self, error: &(dyn std::error::Error + Send + Sync)) -> ErrorClassification {
        let error_str = error.to_string();
        
        if error_str.contains("rate limit") {
            ErrorClassification {
                error_type: ErrorType::Transient,
                severity: ErrorSeverity::Medium,
                retryable: true,
                timeout_ms: Some(60000),
            }
        } else {
            ErrorClassification {
                error_type: ErrorType::System,
                severity: ErrorSeverity::High,
                retryable: false,
                timeout_ms: None,
            }
        }
    }
    
    async fn should_retry(&self, error: &(dyn std::error::Error + Send + Sync), attempt: u32) -> bool {
        let classification = self.classify_error(error).await;
        classification.retryable && attempt < 3
    }
    
    async fn report_error(&self, _error: &(dyn std::error::Error + Send + Sync), _context: &ErrorContext) {
        // Test implementation - just ignore
    }
}

struct TestMonitoringProvider;

impl TestMonitoringProvider {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl MonitoringProvider for TestMonitoringProvider {
    async fn record_metric(&self, _name: &str, _value: f64, _tags: &[(&str, &str)]) {
        // No-op for tests
    }
    
    async fn record_counter(&self, _name: &str, _tags: &[(&str, &str)]) {
        // No-op for tests
    }
    
    async fn record_timing(&self, _name: &str, _duration: std::time::Duration, _tags: &[(&str, &str)]) {
        // No-op for tests
    }
    
    async fn log_info(&self, _message: &str, _context: &std::collections::HashMap<String, String>) {
        // No-op for tests
    }
    
    async fn log_warn(&self, _message: &str, _context: &std::collections::HashMap<String, String>) {
        // No-op for tests
    }
    
    async fn log_error(&self, _message: &str, _context: &std::collections::HashMap<String, String>) {
        // No-op for tests
    }
} 