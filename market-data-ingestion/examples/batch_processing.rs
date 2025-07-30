// market-data-ingestion/examples/batch_processing.rs
// Demonstration of batch processing functionality

use market_data_ingestion::{
    MarketDataIngestionService, MarketDataIngestionConfig, Interval, MarketDataBatch
};
use database_abstraction::DatabaseClient;
use shared_types;
use std::sync::Arc;
use tracing::{info, warn, error};
use tracing_subscriber;
use async_trait::async_trait;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("market_data_ingestion=info,batch_processing=debug")
        .init();

    info!("🚀 Starting Batch Processing Demo");

    // Create configuration optimized for batch processing
    let mut config = MarketDataIngestionConfig::default();
    config.collection.max_batch_size = 1000;
    config.collection.parallel_collections = 5;
    config.storage.batch_size = 500;
    config.storage.flush_interval_seconds = 30;

    // Create mock database client
    let database = Arc::new(MockDatabaseClient);

    // Create ingestion service
    let service = MarketDataIngestionService::new(config, database).await?;
    info!("✅ Service initialized for batch processing");

    // Demonstrate batch processing
    info!("📊 Demonstrating batch processing...");

    let symbols = vec!["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX"];
    let intervals = vec![Interval::FiveMin, Interval::FifteenMin];

    let mut total_batches = 0;
    let mut total_data_points = 0;
    let mut successful_batches = 0;
    let mut failed_batches = 0;

    // Process batches for different symbols and intervals
    for interval in intervals {
        info!("🔄 Processing batches for interval: {:?}", interval);
        
        for symbol in &symbols {
            info!("📈 Collecting data for {}", symbol);
            
            match service.collect_symbol_data(symbol, interval).await {
                Ok(batch) => {
                    total_batches += 1;
                    total_data_points += batch.size();
                    
                    if batch.size() > 0 {
                        successful_batches += 1;
                        info!("✅ Batch {}: {} data points for {}", 
                              batch.batch_id, batch.size(), symbol);
                        
                        // Process the batch
                        if let Err(e) = service.process_batch(batch).await {
                            warn!("⚠️ Failed to process batch: {}", e);
                            failed_batches += 1;
                        }
                    } else {
                        info!("⏭️ Skipped collection for {} (no new data)", symbol);
                    }
                }
                Err(e) => {
                    failed_batches += 1;
                    error!("❌ Failed to collect data for {}: {}", symbol, e);
                }
            }

            // Small delay between collections
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }
    }

    // Show batch processing results
    info!("📈 Batch Processing Results:");
    info!("   Total batches processed: {}", total_batches);
    info!("   Successful batches: {}", successful_batches);
    info!("   Failed batches: {}", failed_batches);
    info!("   Total data points collected: {}", total_data_points);
    info!("   Average data points per batch: {:.2}", 
          if total_batches > 0 { total_data_points as f64 / total_batches as f64 } else { 0.0 });

    // Show service metrics
    let metrics = service.get_metrics().await;
    info!("📊 Service Metrics:");
    info!("   Collections completed: {}", metrics.collections_completed);
    info!("   Collections failed: {}", metrics.collections_failed);
    info!("   API calls made: {}", metrics.api_calls_made);
    info!("   API calls failed: {}", metrics.api_calls_failed);
    info!("   Rate limit hits: {}", metrics.rate_limit_hits);

    info!("✅ Batch Processing Demo completed successfully");
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