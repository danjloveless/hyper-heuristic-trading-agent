//! Intelligent Data Collection Example
//! 
//! This example demonstrates the intelligent data collection functionality
//! that avoids redundant API calls by checking data freshness and only
//! collecting incremental data.

use market_data_ingestion::{
    MarketDataIngestionService, MarketDataIngestionConfig, Interval
};
use database_abstraction::DatabaseClient;
use shared_types;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{info, warn};
use async_trait::async_trait;
use dotenv;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file
    dotenv::dotenv().ok();
    
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("🚀 Starting Intelligent Data Collection Example");
    
    // Load configuration
    let config = MarketDataIngestionConfig::default();
    
    // Initialize database (using mock for this example)
    let database = Arc::new(MockDatabaseClient);
    
    // Create ingestion service
    let service = MarketDataIngestionService::new(config, database).await?;
    
    let symbol = "AAPL";
    let interval = Interval::FiveMin;
    
    info!("📊 Demonstrating intelligent collection for {} at {:?} intervals", symbol, interval);
    
    // First collection - should fetch data
    info!("\n1️⃣ First collection (should fetch data):");
    let batch1 = service.collect_symbol_data_intelligently(symbol, interval).await?;
    info!("   Collected {} data points", batch1.size());
    info!("   Batch ID: {}", batch1.batch_id);
    info!("   Skip reason: {:?}", batch1.metadata.get("skip_reason"));
    
    // Wait a short time
    sleep(Duration::from_secs(2)).await;
    
    // Second collection - should skip due to fresh data
    info!("\n2️⃣ Second collection (should skip due to fresh data):");
    let batch2 = service.collect_symbol_data_intelligently(symbol, interval).await?;
    info!("   Collected {} data points", batch2.size());
    info!("   Skip reason: {:?}", batch2.metadata.get("skip_reason"));
    info!("   Latest timestamp: {:?}", batch2.metadata.get("latest_timestamp"));
    
    // Check data freshness
    info!("\n3️⃣ Checking data freshness:");
    let latest_timestamp = service.get_latest_timestamp(symbol, interval).await?;
    if let Some(timestamp) = latest_timestamp {
        let is_fresh = service.is_data_fresh(timestamp, interval);
        info!("   Latest timestamp: {}", timestamp);
        info!("   Is fresh: {}", is_fresh);
    } else {
        info!("   No data found");
    }
    
    // Force collection
    info!("\n4️⃣ Force collection (bypasses freshness check):");
    let force_batch = service.force_collect_symbol_data(symbol, interval).await?;
    info!("   Collected {} data points", force_batch.size());
    info!("   Force collection: {:?}", force_batch.metadata.get("force_collection"));
    
    // Demonstrate incremental collection
    info!("\n5️⃣ Demonstrating incremental collection:");
    info!("   Note: Incremental data collection would be performed through the service's public interface");
    
    // Show collection statistics
    info!("\n📈 Collection Statistics:");
    let metrics = service.get_metrics().await;
    info!("   Collections completed: {}", metrics.collections_completed);
    info!("   Collections failed: {}", metrics.collections_failed);
    info!("   API calls made: {}", metrics.api_calls_made);
    info!("   API calls failed: {}", metrics.api_calls_failed);
    info!("   Rate limit hits: {}", metrics.rate_limit_hits);
    
    info!("✅ Intelligent collection example completed successfully!");
    
    Ok(())
}

// Mock database client for the example
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