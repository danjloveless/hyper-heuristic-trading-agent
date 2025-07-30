// market-data-ingestion/examples/rate_limit_demo.rs
// Demonstration of rate limiting functionality

use market_data_ingestion::{
    MarketDataIngestionService, MarketDataIngestionConfig, Interval
};
use database_abstraction::DatabaseClient;
use shared_types;
use std::sync::Arc;
use tracing::{info, warn, error};
use tracing_subscriber;
use async_trait::async_trait;
use dotenv;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file
    dotenv::dotenv().ok();
    
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("market_data_ingestion=info,rate_limit_demo=debug")
        .init();

    info!("🚀 Starting Rate Limit Demo");

    // Create configuration with low rate limits for demonstration
    let mut config = MarketDataIngestionConfig::default();
    config.rate_limits.calls_per_minute = 5; // Very low for demo
    config.rate_limits.calls_per_day = 100;
    config.rate_limits.is_premium = false;

    // Create mock database client
    let database = Arc::new(MockDatabaseClient);

    // Create ingestion service
    let service = MarketDataIngestionService::new(config, database).await?;
    info!("✅ Service initialized with rate limiting");

    // Demonstrate rate limiting
    info!("📊 Demonstrating rate limiting behavior...");

    let symbols = vec!["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"];
    let mut successful_collections = 0;
    let mut rate_limited_collections = 0;

    for (i, symbol) in symbols.iter().enumerate() {
        info!("🔄 Attempting collection {} for {}", i + 1, symbol);
        
        match service.collect_symbol_data(symbol, Interval::FiveMin).await {
            Ok(batch) => {
                successful_collections += 1;
                info!("✅ Successfully collected {} data points for {}", batch.size(), symbol);
            }
            Err(e) => {
                if e.to_string().contains("rate limit") {
                    rate_limited_collections += 1;
                    warn!("⏳ Rate limited for {}: {}", symbol, e);
                } else {
                    error!("❌ Error collecting {}: {}", symbol, e);
                }
            }
        }

        // Small delay between attempts
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    info!("📈 Rate Limit Demo Results:");
    info!("   Successful collections: {}", successful_collections);
    info!("   Rate limited collections: {}", rate_limited_collections);
    info!("   Total attempts: {}", symbols.len());

    // Note: Rate limit status would be available through the service's public interface
    // in a real implementation. For this demo, we're just showing the collection results.
    
    info!("📊 Rate Limit Demo Summary:");
    info!("   Rate limits configured: 5 calls per minute, 100 calls per day");
    info!("   Demo shows how the system handles rate limiting gracefully");

    info!("✅ Rate Limit Demo completed successfully");
    Ok(())
}

// Mock database client for demonstration
struct MockDatabaseClient;

#[async_trait]
impl DatabaseClient for MockDatabaseClient {
    async fn insert_market_data(&self, _data: &[shared_types::MarketData]) -> std::result::Result<(), database_abstraction::DatabaseError> {
        Ok(())
    }
    
    async fn get_market_data(&self, _symbol: &str, _start: chrono::DateTime<chrono::Utc>, _end: chrono::DateTime<chrono::Utc>) -> std::result::Result<Vec<shared_types::MarketData>, database_abstraction::DatabaseError> {
        Ok(Vec::new())
    }
    
    async fn get_latest_market_data(&self, _symbol: &str) -> std::result::Result<Option<shared_types::MarketData>, database_abstraction::DatabaseError> {
        Ok(None)
    }
    
    async fn insert_sentiment_data(&self, _data: &[shared_types::SentimentData]) -> std::result::Result<(), database_abstraction::DatabaseError> {
        Ok(())
    }
    
    async fn get_sentiment_data(&self, _symbol: &str, _start: chrono::DateTime<chrono::Utc>, _end: chrono::DateTime<chrono::Utc>) -> std::result::Result<Vec<shared_types::SentimentData>, database_abstraction::DatabaseError> {
        Ok(Vec::new())
    }
    
    async fn get_aggregated_sentiment(&self, _symbol: &str, _timestamp: chrono::DateTime<chrono::Utc>) -> std::result::Result<Option<shared_types::AggregatedSentiment>, database_abstraction::DatabaseError> {
        Ok(None)
    }
    
    async fn insert_features(&self, _features: &[shared_types::FeatureSet]) -> std::result::Result<(), database_abstraction::DatabaseError> {
        Ok(())
    }
    
    async fn get_features(&self, _symbol: &str, _timestamp: chrono::DateTime<chrono::Utc>) -> std::result::Result<Option<shared_types::FeatureSet>, database_abstraction::DatabaseError> {
        Ok(None)
    }
    
    async fn get_latest_features(&self, _symbol: &str) -> std::result::Result<Option<shared_types::FeatureSet>, database_abstraction::DatabaseError> {
        Ok(None)
    }
    
    async fn insert_technical_indicators(&self, _indicators: &[shared_types::TechnicalIndicators]) -> std::result::Result<(), database_abstraction::DatabaseError> {
        Ok(())
    }
    
    async fn insert_predictions(&self, _predictions: &[shared_types::PredictionResult]) -> std::result::Result<(), database_abstraction::DatabaseError> {
        Ok(())
    }
    
    async fn get_predictions(&self, _symbol: &str, _start: chrono::DateTime<chrono::Utc>, _end: chrono::DateTime<chrono::Utc>) -> std::result::Result<Vec<shared_types::PredictionResult>, database_abstraction::DatabaseError> {
        Ok(Vec::new())
    }
    
    async fn insert_prediction_outcomes(&self, _outcomes: &[shared_types::PredictionOutcome]) -> std::result::Result<(), database_abstraction::DatabaseError> {
        Ok(())
    }
    
    async fn insert_strategy_performance(&self, _performance: &[shared_types::StrategyPerformance]) -> std::result::Result<(), database_abstraction::DatabaseError> {
        Ok(())
    }
    
    async fn get_strategy_performance(&self, _strategy_name: &str, _symbol: Option<&str>, _start: chrono::DateTime<chrono::Utc>, _end: chrono::DateTime<chrono::Utc>) -> std::result::Result<Vec<shared_types::StrategyPerformance>, database_abstraction::DatabaseError> {
        Ok(Vec::new())
    }
    
    async fn health_check(&self) -> std::result::Result<database_abstraction::HealthStatus, database_abstraction::DatabaseError> {
        Ok(database_abstraction::HealthStatus {
            service: "mock".to_string(),
            status: shared_types::ServiceStatus::Healthy,
            timestamp: chrono::Utc::now(),
            checks: Vec::new(),
        })
    }
} 