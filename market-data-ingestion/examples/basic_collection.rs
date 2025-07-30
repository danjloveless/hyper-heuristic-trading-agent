// market-data-ingestion/examples/basic_collection.rs
// Comprehensive example demonstrating market data collection and processing

use market_data_ingestion::{
    MarketDataIngestionService, MarketDataIngestionConfig,
    AlphaVantageClient, AlphaVantageConfig, RateLimiter,
    Interval, OutputSize, MarketDataBatch,
    DataQualityController, DataQualityConfig,
    HealthStatus, IngestionMetrics, IngestionError,
    ServiceConfig, RateLimitsConfig, CollectionConfig, StorageConfig,
};
use database_abstraction::DatabaseClient;
use shared_types;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use async_trait::async_trait;
use dotenv;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file
    dotenv::dotenv().ok();
    
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("market_data_ingestion=debug,basic_collection=info")
        .init();

    println!("🚀 QuantumTrade AI Market Data Ingestion - Basic Collection Example");
    println!("===================================================================");

    // Check for API key
    if std::env::var("ALPHA_VANTAGE_API_KEY").is_err() {
        eprintln!("❌ Error: ALPHA_VANTAGE_API_KEY environment variable not set");
        eprintln!("Please set your Alpha Vantage API key:");
        eprintln!("export ALPHA_VANTAGE_API_KEY=\"your_api_key_here\"");
        return Ok(());
    }

    // Create configuration
    let config = create_demo_config();
    info!("Configuration created successfully");

    // Create mock database (in real usage, you'd use the actual database)
    let mock_database = Arc::new(MockDatabase::new());
    
    // Initialize the ingestion service
    let service = MarketDataIngestionService::new(config.clone(), mock_database.clone()).await?;
    info!("Market Data Ingestion Service initialized");

    // Demonstrate different collection scenarios
    demonstrate_single_symbol_collection(&service).await?;
    demonstrate_multiple_symbols_collection(&service).await?;
    demonstrate_different_intervals(&service).await?;
    demonstrate_data_quality_validation(&service).await?;
    demonstrate_rate_limiting(&service).await?;
    demonstrate_health_monitoring(&service).await?;
    demonstrate_metrics_collection(&service).await?;
    demonstrate_error_scenarios(&service).await?;

    println!("\n✅ All demonstrations completed successfully!");
    Ok(())
}

fn create_demo_config() -> MarketDataIngestionConfig {
    MarketDataIngestionConfig {
        service: ServiceConfig {
            service_name: "market-data-demo".to_string(),
            port: 8080,
            worker_threads: 2,
            max_concurrent_collections: 10,
        },
        alpha_vantage: AlphaVantageConfig {
            base_url: "https://www.alphavantage.co/query".to_string(),
            api_key: std::env::var("ALPHA_VANTAGE_API_KEY").unwrap_or_default(),
            timeout_seconds: 30,
            max_retries: 3,
        },
        rate_limits: RateLimitsConfig {
            calls_per_minute: 5,
            calls_per_day: 500,
            premium_calls_per_minute: 75,
            premium_calls_per_day: 75000,
            is_premium: false,
        },
        collection: CollectionConfig {
            default_symbols: vec![
                "AAPL".to_string(),
                "GOOGL".to_string(),
                "MSFT".to_string(),
            ],
            priority_symbols: vec![
                "SPY".to_string(),
                "QQQ".to_string(),
            ],
            collection_intervals: vec![
                "5min".to_string(),
                "1hour".to_string(),
                "1day".to_string(),
            ],
            max_batch_size: 100,
            parallel_collections: 5,
        },
        data_quality: DataQualityConfig {
            quality_threshold: 70,
            enable_deduplication: true,
            max_price_deviation: 0.1,
            volume_threshold: 1000,
        },
        storage: StorageConfig {
            batch_size: 50,
            flush_interval_seconds: 10,
            enable_compression: true,
            retention_days: 365,
        },
    }
}

async fn demonstrate_single_symbol_collection(
    service: &MarketDataIngestionService
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Single Symbol Collection Demonstration");
    println!("------------------------------------------");

    let symbol = "AAPL";
    let interval = Interval::FiveMin;

    info!("Collecting {} data at {:?} interval", symbol, interval);

    match service.collect_symbol_data(symbol, interval).await {
        Ok(batch) => {
            info!("✅ Successfully collected {} data points for {}", batch.size(), symbol);
            
            if !batch.is_empty() {
                let (start_time, end_time) = batch.get_time_range().unwrap();
                info!("📅 Data range: {} to {}", start_time, end_time);
                
                // Show first data point as example
                if let Some(first_point) = batch.data_points.first() {
                    info!("📈 Sample data point:");
                    info!("   Open: ${}, High: ${}, Low: ${}, Close: ${}",
                          first_point.open, first_point.high, first_point.low, first_point.close);
                    info!("   Volume: {}, Quality Score: {}",
                          first_point.volume, first_point.quality_score);
                }
                
                // Process the batch
                service.process_batch(batch).await?;
                info!("✅ Batch processed and stored successfully");
            } else {
                warn!("⚠️ No data points collected for {}", symbol);
            }
        },
        Err(e) => {
            error!("❌ Failed to collect data for {}: {}", symbol, e);
            match e {
                IngestionError::RateLimitExceeded { .. } => {
                    info!("💡 Rate limit exceeded - this is normal for demo API keys");
                },
                IngestionError::ApiError { .. } => {
                    info!("💡 API error - check your API key and symbol validity");
                },
                _ => {}
            }
        }
    }

    Ok(())
}

async fn demonstrate_multiple_symbols_collection(
    service: &MarketDataIngestionService
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Multiple Symbols Collection Demonstration");
    println!("---------------------------------------------");

    let symbols = vec!["AAPL", "GOOGL", "MSFT"];
    let interval = Interval::FifteenMin;

    info!("Collecting data for {} symbols at {:?} interval", symbols.len(), interval);

    for symbol in symbols {
        info!("Collecting data for {}", symbol);
        
        match service.collect_symbol_data(symbol, interval).await {
            Ok(batch) => {
                info!("✅ Collected {} data points for {}", batch.size(), symbol);
                service.process_batch(batch).await?;
            },
            Err(e) => {
                warn!("⚠️ Failed to collect data for {}: {}", symbol, e);
                
                // Implement delay between requests to respect rate limits
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
        
        // Small delay between collections to respect rate limits
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    Ok(())
}

async fn demonstrate_different_intervals(
    service: &MarketDataIngestionService
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⏱️ Different Intervals Demonstration");
    println!("------------------------------------");

    let symbol = "SPY";
    let intervals = vec![
        Interval::OneMin,
        Interval::FiveMin,
        Interval::FifteenMin,
        Interval::SixtyMin,
    ];

    for interval in intervals {
        info!("Collecting {} data at {:?} interval", symbol, interval);
        
        match service.collect_symbol_data(symbol, interval).await {
            Ok(batch) => {
                info!("✅ {:?}: {} data points collected", interval, batch.size());
                
                if !batch.is_empty() {
                    let avg_volume: u64 = batch.data_points.iter()
                        .map(|dp| dp.volume)
                        .sum::<u64>() / batch.size() as u64;
                    
                    info!("   Average volume: {}", avg_volume);
                }
                
                service.process_batch(batch).await?;
            },
            Err(e) => {
                warn!("⚠️ {:?}: Collection failed - {}", interval, e);
            }
        }
        
        // Delay between requests
        tokio::time::sleep(Duration::from_millis(300)).await;
    }

    Ok(())
}

async fn demonstrate_data_quality_validation(
    service: &MarketDataIngestionService
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🛡️ Data Quality Validation Demonstration");
    println!("------------------------------------------");

    // Create a data quality controller for demonstration
    let quality_config = DataQualityConfig {
        quality_threshold: 80,
        enable_deduplication: true,
        max_price_deviation: 0.05, // 5% max deviation
        volume_threshold: 10000,
    };
    
    let quality_controller = DataQualityController::new(quality_config);

    info!("Collecting data with quality validation");

    match service.collect_symbol_data("AAPL", Interval::FiveMin).await {
        Ok(batch) => {
            info!("✅ Collected {} data points", batch.size());
            
            let mut high_quality_count = 0;
            let mut medium_quality_count = 0;
            let mut low_quality_count = 0;
            
            for data_point in &batch.data_points {
                let quality_score = quality_controller.calculate_quality_score(data_point);
                
                if quality_score >= 90 {
                    high_quality_count += 1;
                } else if quality_score >= 70 {
                    medium_quality_count += 1;
                } else {
                    low_quality_count += 1;
                }
            }
            
            info!("📊 Quality Distribution:");
            info!("   High Quality (90-100): {}", high_quality_count);
            info!("   Medium Quality (70-89): {}", medium_quality_count);
            info!("   Low Quality (<70): {}", low_quality_count);
            
            service.process_batch(batch).await?;
        },
        Err(e) => {
            warn!("⚠️ Data collection failed: {}", e);
        }
    }

    Ok(())
}

async fn demonstrate_rate_limiting(
    _service: &MarketDataIngestionService
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚦 Rate Limiting Demonstration");
    println!("-------------------------------");

    // Create a rate limiter for demonstration
    let mut rate_limiter = RateLimiter::new(3, 10); // 3 per minute, 10 per day

    info!("Demonstrating rate limiting with 3 requests per minute");

    for i in 1..=5 {
        if rate_limiter.can_make_request().await {
            info!("✅ Request {}: Allowed", i);
            rate_limiter.record_request().await;
        } else {
            warn!("🚫 Request {}: Rate limited", i);
        }
        
        let status = rate_limiter.get_status();
        info!("   Remaining this minute: {}", status.calls_remaining_minute);
        info!("   Remaining today: {}", status.calls_remaining_day);
        
        if status.in_backoff {
            if let Some(backoff_until) = status.backoff_until {
                info!("   In backoff until: {}", backoff_until);
            }
        }
        
        println!();
    }

    // Simulate rate limit hit
    rate_limiter.record_rate_limit().await;
    let status = rate_limiter.get_status();
    
    if status.in_backoff {
        info!("🔄 Rate limiter is now in backoff mode");
        if let Some(backoff_until) = status.backoff_until {
            info!("   Backoff until: {}", backoff_until);
        }
    }

    Ok(())
}

async fn demonstrate_health_monitoring(
    service: &MarketDataIngestionService
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🏥 Health Monitoring Demonstration");
    println!("-----------------------------------");

    let health_status = service.get_health().await;

    info!("Service Health Status:");
    info!("   Alpha Vantage Reachable: {}", 
          if health_status.alpha_vantage_reachable { "✅" } else { "❌" });
    info!("   Database Connected: {}", 
          if health_status.database_connected { "✅" } else { "❌" });
    info!("   Rate Limit Healthy: {}", 
          if health_status.rate_limit_healthy { "✅" } else { "❌" });
    info!("   Last Successful Collection: {}", health_status.last_successful_collection);
    info!("   Current Error Rate: {:.2}%", health_status.current_error_rate * 100.0);

    // Health check interpretation
    if health_status.alpha_vantage_reachable && 
       health_status.database_connected && 
       health_status.rate_limit_healthy &&
       health_status.current_error_rate < 0.1 {
        info!("🎉 All systems are healthy!");
    } else {
        warn!("⚠️ Some health checks are failing - check system status");
    }

    Ok(())
}

async fn demonstrate_metrics_collection(
    service: &MarketDataIngestionService
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Metrics Collection Demonstration");
    println!("------------------------------------");

    let metrics = service.get_metrics().await;

    info!("Service Metrics:");
    info!("   Collections Completed: {}", metrics.collections_completed);
    info!("   Collections Failed: {}", metrics.collections_failed);
    info!("   API Calls Made: {}", metrics.api_calls_made);
    info!("   API Calls Failed: {}", metrics.api_calls_failed);
    info!("   Rate Limit Hits: {}", metrics.rate_limit_hits);

    // Calculate success rate
    let total_collections = metrics.collections_completed + metrics.collections_failed;
    if total_collections > 0 {
        let success_rate = (metrics.collections_completed as f64 / total_collections as f64) * 100.0;
        info!("   Success Rate: {:.1}%", success_rate);
    }

    let total_api_calls = metrics.api_calls_made + metrics.api_calls_failed;
    if total_api_calls > 0 {
        let api_success_rate = (metrics.api_calls_made as f64 / total_api_calls as f64) * 100.0;
        info!("   API Success Rate: {:.1}%", api_success_rate);
    }

    Ok(())
}

async fn demonstrate_error_scenarios(
    service: &MarketDataIngestionService
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n❌ Error Scenarios Demonstration");
    println!("---------------------------------");

    // Test with invalid symbol
    info!("Testing with invalid symbol...");
    match service.collect_symbol_data("INVALID_SYMBOL_123", Interval::FiveMin).await {
        Ok(_) => info!("Unexpected success with invalid symbol"),
        Err(e) => {
            info!("✅ Expected error with invalid symbol: {}", e);
            match e {
                IngestionError::SymbolNotFound { symbol } => {
                    info!("   Symbol '{}' not found - this is expected", symbol);
                },
                IngestionError::ApiError { message, status_code } => {
                    info!("   API Error ({}): {} - this is expected", status_code, message);
                },
                _ => {
                    info!("   Other error type: {:?}", e);
                }
            }
        }
    }

    // Test with empty symbol
    info!("\nTesting with empty symbol...");
    match service.collect_symbol_data("", Interval::FiveMin).await {
        Ok(_) => info!("Unexpected success with empty symbol"),
        Err(e) => {
            info!("✅ Expected error with empty symbol: {}", e);
        }
    }

    info!("\n💡 Error handling demonstration completed");
    info!("The system gracefully handles various error scenarios without crashing");

    Ok(())
}

// Mock database for demonstration purposes
#[derive(Debug)]
struct MockDatabase;

impl MockDatabase {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatabaseClient for MockDatabase {
    async fn insert_market_data(&self, _data: &[shared_types::MarketData]) -> Result<(), database_abstraction::DatabaseError> {
        info!("📝 Mock Database: Inserting {} market data records", _data.len());
        Ok(())
    }
    
    async fn get_market_data(&self, _symbol: &str, _start: chrono::DateTime<chrono::Utc>, _end: chrono::DateTime<chrono::Utc>) -> Result<Vec<shared_types::MarketData>, database_abstraction::DatabaseError> {
        Ok(Vec::new())
    }
    
    async fn get_latest_market_data(&self, _symbol: &str) -> Result<Option<shared_types::MarketData>, database_abstraction::DatabaseError> {
        Ok(None)
    }
    
    async fn insert_sentiment_data(&self, _data: &[shared_types::SentimentData]) -> Result<(), database_abstraction::DatabaseError> {
        Ok(())
    }
    
    async fn get_sentiment_data(&self, _symbol: &str, _start: chrono::DateTime<chrono::Utc>, _end: chrono::DateTime<chrono::Utc>) -> Result<Vec<shared_types::SentimentData>, database_abstraction::DatabaseError> {
        Ok(Vec::new())
    }
    
    async fn get_aggregated_sentiment(&self, _symbol: &str, _timestamp: chrono::DateTime<chrono::Utc>) -> Result<Option<shared_types::AggregatedSentiment>, database_abstraction::DatabaseError> {
        Ok(None)
    }
    
    async fn insert_features(&self, _features: &[shared_types::FeatureSet]) -> Result<(), database_abstraction::DatabaseError> {
        Ok(())
    }
    
    async fn get_features(&self, _symbol: &str, _timestamp: chrono::DateTime<chrono::Utc>) -> Result<Option<shared_types::FeatureSet>, database_abstraction::DatabaseError> {
        Ok(None)
    }
    
    async fn get_latest_features(&self, _symbol: &str) -> Result<Option<shared_types::FeatureSet>, database_abstraction::DatabaseError> {
        Ok(None)
    }
    
    async fn insert_technical_indicators(&self, _indicators: &[shared_types::TechnicalIndicators]) -> Result<(), database_abstraction::DatabaseError> {
        Ok(())
    }
    
    async fn insert_predictions(&self, _predictions: &[shared_types::PredictionResult]) -> Result<(), database_abstraction::DatabaseError> {
        Ok(())
    }
    
    async fn get_predictions(&self, _symbol: &str, _start: chrono::DateTime<chrono::Utc>, _end: chrono::DateTime<chrono::Utc>) -> Result<Vec<shared_types::PredictionResult>, database_abstraction::DatabaseError> {
        Ok(Vec::new())
    }
    
    async fn insert_prediction_outcomes(&self, _outcomes: &[shared_types::PredictionOutcome]) -> Result<(), database_abstraction::DatabaseError> {
        Ok(())
    }
    
    async fn insert_strategy_performance(&self, _performance: &[shared_types::StrategyPerformance]) -> Result<(), database_abstraction::DatabaseError> {
        Ok(())
    }
    
    async fn get_strategy_performance(&self, _strategy_name: &str, _symbol: Option<&str>, _start: chrono::DateTime<chrono::Utc>, _end: chrono::DateTime<chrono::Utc>) -> Result<Vec<shared_types::StrategyPerformance>, database_abstraction::DatabaseError> {
        Ok(Vec::new())
    }
    
    async fn health_check(&self) -> Result<database_abstraction::HealthStatus, database_abstraction::DatabaseError> {
        Ok(database_abstraction::HealthStatus {
            service: "mock".to_string(),
            status: shared_types::ServiceStatus::Healthy,
            timestamp: chrono::Utc::now(),
            checks: Vec::new(),
        })
    }
}

// Legacy method for backward compatibility
impl MockDatabase {
    async fn store_market_data(&self, batch: &MarketDataBatch) -> Result<(), String> {
        info!("📝 Mock Database: Storing {} data points for symbol {}", 
              batch.size(), batch.symbol);
        
        // Simulate storage delay
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        Ok(())
    }
}

// Additional utility functions for demonstration

fn print_market_data_summary(batch: &MarketDataBatch) {
    if batch.is_empty() {
        info!("📊 Batch is empty");
        return;
    }

    let first = &batch.data_points[0];
    let last = &batch.data_points[batch.size() - 1];
    
    info!("📊 Market Data Summary for {}:", batch.symbol);
    info!("   Total Data Points: {}", batch.size());
    info!("   Time Range: {} to {}", first.timestamp, last.timestamp);
    info!("   Price Range: ${} - ${}", 
          batch.data_points.iter().map(|d| d.low).min().unwrap(),
          batch.data_points.iter().map(|d| d.high).max().unwrap());
    
    let total_volume: u64 = batch.data_points.iter().map(|d| d.volume).sum();
    info!("   Total Volume: {}", total_volume);
    
    let avg_quality: f64 = batch.data_points.iter()
        .map(|d| d.quality_score as f64)
        .sum::<f64>() / batch.size() as f64;
    info!("   Average Quality Score: {:.1}", avg_quality);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = create_demo_config();
        assert_eq!(config.service.service_name, "market-data-demo");
        assert!(config.data_quality.quality_threshold > 0);
        assert!(config.rate_limits.calls_per_minute > 0);
    }

    #[tokio::test]
    async fn test_mock_database() {
        let db = MockDatabase::new();
        let batch = MarketDataBatch::new("TEST".to_string(), "Test Source".to_string());
        
        let result = db.store_market_data(&batch).await;
        assert!(result.is_ok());
    }
}