use crate::alpha_vantage::{AlphaVantageClient, Interval, OutputSize};
use crate::batch_processor::BatchProcessor;
use crate::config::MarketDataIngestionConfig;
use crate::data_quality::DataQualityController;
use crate::errors::{Result, IngestionError};
use crate::health::HealthChecker;
use crate::metrics::MetricsCollector;
use crate::models::{MarketData, MarketDataBatch};
use crate::rate_limiter::RateLimiter;
use crate::scheduler::IngestionScheduler;
use chrono::{DateTime, Utc};
use std::sync::Arc;

use tokio::sync::{RwLock, Semaphore};
use tracing::{info, warn, debug};

/// Main market data ingestion service
pub struct MarketDataIngestionService {
    config: MarketDataIngestionConfig,
    alpha_vantage_client: AlphaVantageClient,
    data_quality_controller: DataQualityController,
    batch_processor: BatchProcessor,
    scheduler: IngestionScheduler,
    health_checker: HealthChecker,
    metrics_collector: MetricsCollector,
    collection_semaphore: Arc<Semaphore>,
}

impl MarketDataIngestionService {
    pub async fn new(
        config: MarketDataIngestionConfig,
        database: Arc<dyn database_abstraction::traits::DatabaseClient>,
    ) -> Result<Self> {
        let rate_limiter = if config.rate_limits.is_premium {
            Arc::new(RwLock::new(RateLimiter::new(
                config.rate_limits.premium_calls_per_minute,
                config.rate_limits.premium_calls_per_day,
            )))
        } else {
            Arc::new(RwLock::new(RateLimiter::new(
                config.rate_limits.calls_per_minute,
                config.rate_limits.calls_per_day,
            )))
        };
        
        let alpha_vantage_client = AlphaVantageClient::new(
            config.alpha_vantage.clone(),
            rate_limiter.clone(),
        );
        
        let data_quality_controller = DataQualityController::new(config.data_quality.clone());
        let batch_processor = BatchProcessor::new(config.storage.clone(), database.clone());
        let scheduler = IngestionScheduler::new(config.collection.clone());
        let health_checker = HealthChecker::new();
        let metrics_collector = MetricsCollector::new();
        let collection_semaphore = Arc::new(Semaphore::new(config.service.max_concurrent_collections));
        
        Ok(Self {
            config,
            alpha_vantage_client,
            data_quality_controller,
            batch_processor,
            scheduler,
            health_checker,
            metrics_collector,
            collection_semaphore,
        })
    }
    
    /// Start the ingestion service
    pub async fn start(&self) -> Result<()> {
        info!("Starting Market Data Ingestion Service");
        
        // Start background tasks
        let _scheduler_handle = self.start_scheduler().await?;
        let _health_check_handle = self.start_health_checks().await?;
        let _metrics_handle = self.start_metrics_collection().await?;
        
        info!("Market Data Ingestion Service started successfully");
        
        // Wait for shutdown signal (in a real implementation)
        // For now, we'll just return
        Ok(())
    }
    
    /// Collects market data intelligently, avoiding redundant API calls
    pub async fn collect_symbol_data_intelligently(&self, symbol: &str, interval: Interval) -> Result<MarketDataBatch> {
        let start_time = std::time::Instant::now();
        
        // 1. Check what data we already have
        let latest_timestamp = self.get_latest_timestamp(symbol, interval).await?;
        
        // 2. Only fetch if data is stale or missing
        if let Some(timestamp) = latest_timestamp {
            if self.is_data_fresh(timestamp, interval) {
                info!("Data for {} {} is fresh (latest: {}), skipping API call", 
                      symbol, interval, timestamp);
                
                // Return empty batch with metadata indicating skip
                let mut batch = MarketDataBatch::new(symbol.to_string(), "Alpha Vantage".to_string());
                batch.metadata.insert("skip_reason".to_string(), "data_fresh".to_string());
                batch.metadata.insert("latest_timestamp".to_string(), timestamp.to_rfc3339());
                batch.metadata.insert("collection_time_ms".to_string(), start_time.elapsed().as_millis().to_string());
                
                return Ok(batch);
            }
        }
        
        // 3. Fetch incremental data
        let new_data = self.fetch_incremental_data(symbol, interval, latest_timestamp).await?;
        
        // 4. Create batch with metadata
        let mut batch = MarketDataBatch::new(symbol.to_string(), "Alpha Vantage".to_string());
        for data_point in new_data {
            batch.add_data_point(data_point);
        }
        
        // Add metadata
        batch.metadata.insert("collection_time_ms".to_string(), start_time.elapsed().as_millis().to_string());
        if let Some(timestamp) = latest_timestamp {
            batch.metadata.insert("previous_latest_timestamp".to_string(), timestamp.to_rfc3339());
        }
        batch.metadata.insert("data_points_collected".to_string(), batch.size().to_string());
        
        info!("Intelligently collected {} data points for {} {} in {}ms", 
              batch.size(), symbol, interval, start_time.elapsed().as_millis());
        
        Ok(batch)
    }

    /// Gets the latest timestamp for a symbol and interval from the database
    pub async fn get_latest_timestamp(&self, symbol: &str, _interval: Interval) -> Result<Option<DateTime<Utc>>> {
        // Try to get the latest market data for this symbol
        match self.batch_processor.database.get_latest_market_data(symbol).await {
            Ok(Some(latest_data)) => {
                // For now, we'll use the latest timestamp regardless of interval
                // In a more sophisticated implementation, you might want to filter by interval
                Ok(Some(latest_data.timestamp))
            },
            Ok(None) => {
                debug!("No existing data found for symbol: {}", symbol);
                Ok(None)
            },
            Err(e) => {
                warn!("Failed to get latest timestamp for {}: {}", symbol, e);
                // Return None to allow collection to proceed
                Ok(None)
            }
        }
    }

    /// Determines if the data is fresh enough to skip collection
    pub fn is_data_fresh(&self, latest_timestamp: DateTime<Utc>, interval: Interval) -> bool {
        let now = Utc::now();
        let age = now - latest_timestamp;
        
        // Define freshness thresholds based on interval (in seconds)
        let freshness_threshold_seconds = match interval {
            Interval::OneMin => 120,      // 2 minutes for 1-min data
            Interval::FiveMin => 420,     // 7 minutes for 5-min data
            Interval::FifteenMin => 1200, // 20 minutes for 15-min data
            Interval::ThirtyMin => 2100,  // 35 minutes for 30-min data
            Interval::SixtyMin => 3900,   // 65 minutes for 1-hour data
        };
        
        age.num_seconds() < freshness_threshold_seconds
    }

    /// Fetches incremental data from the API
    async fn fetch_incremental_data(
        &self, 
        symbol: &str, 
        interval: Interval, 
        latest_timestamp: Option<DateTime<Utc>>
    ) -> Result<Vec<MarketData>> {
        // For now, we'll fetch the full dataset and let the deduplication handle it
        // In a more sophisticated implementation, you might use the latest_timestamp
        // to request only data after that point if the API supports it
        
        let response = self.alpha_vantage_client
            .get_intraday_data(symbol, interval, OutputSize::Compact)
            .await?;
        
        let mut market_data = Vec::new();
        
        // Parse the response and convert to MarketData
        for (series_name, entries) in response.time_series {
            for (timestamp_str, entry) in entries {
                let timestamp = self.parse_timestamp(&timestamp_str)?;
                
                // Skip data points we already have (if we have a latest timestamp)
                if let Some(latest) = latest_timestamp {
                    if timestamp <= latest {
                        continue;
                    }
                }
                
                let data_point = entry.to_market_data(symbol, timestamp)?;
                market_data.push(data_point);
            }
        }
        
        // Sort by timestamp to ensure chronological order
        market_data.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        
        Ok(market_data)
    }

    /// Parses timestamp string from Alpha Vantage response
    fn parse_timestamp(&self, timestamp_str: &str) -> Result<DateTime<Utc>> {
        // Alpha Vantage returns timestamps in format like "2024-01-15 16:00:00"
        // We need to parse this and convert to UTC
        
        // Try different timestamp formats
        let formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ];
        
        for format in &formats {
            if let Ok(naive_dt) = chrono::NaiveDateTime::parse_from_str(timestamp_str, format) {
                return Ok(DateTime::<Utc>::from_naive_utc_and_offset(naive_dt, Utc));
            }
        }
        
        Err(IngestionError::ParsingError {
            field: "timestamp".to_string(),
            error: format!("Unable to parse timestamp: {}", timestamp_str),
        })
    }

    /// Enhanced collection method that uses intelligent collection
    pub async fn collect_symbol_data(&self, symbol: &str, interval: Interval) -> Result<MarketDataBatch> {
        // Use intelligent collection by default
        self.collect_symbol_data_intelligently(symbol, interval).await
    }

    /// Force collection method that bypasses freshness checks
    pub async fn force_collect_symbol_data(&self, symbol: &str, interval: Interval) -> Result<MarketDataBatch> {
        info!("Force collecting data for symbol: {} at interval: {:?}", symbol, interval);
        
        let start_time = std::time::Instant::now();
        
        // Fetch full dataset without checking freshness
        let response = self.alpha_vantage_client
            .get_intraday_data(symbol, interval, OutputSize::Compact)
            .await?;
        
        let mut batch = MarketDataBatch::new(symbol.to_string(), "Alpha Vantage".to_string());
        
        // Parse the response and convert to MarketData
        for (_series_name, entries) in response.time_series {
            for (timestamp_str, entry) in entries {
                let timestamp = self.parse_timestamp(&timestamp_str)?;
                let data_point = entry.to_market_data(symbol, timestamp)?;
                batch.add_data_point(data_point);
            }
        }
        
        // Sort by timestamp
        batch.sort_by_timestamp();
        
        // Add metadata
        batch.metadata.insert("force_collection".to_string(), "true".to_string());
        batch.metadata.insert("collection_time_ms".to_string(), start_time.elapsed().as_millis().to_string());
        batch.metadata.insert("data_points_collected".to_string(), batch.size().to_string());
        
        info!("Force collected {} data points for {} {} in {}ms", 
              batch.size(), symbol, interval, start_time.elapsed().as_millis());
        
        Ok(batch)
    }
    
    /// Process and store a batch of market data
    pub async fn process_batch(&self, batch: MarketDataBatch) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        
        debug!("Processing batch {} with {} data points", batch.batch_id, batch.size());
        
        let start_time = std::time::Instant::now();
        
        // Process the batch
        self.batch_processor.process_batch(batch).await?;
        
        let processing_time = start_time.elapsed();
        self.metrics_collector.record_batch_processing_success(processing_time);
        
        debug!("Batch processed successfully in {:?}", processing_time);
        
        Ok(())
    }
    
    /// Get service health status
    pub async fn get_health(&self) -> crate::health::HealthStatus {
        self.health_checker.get_health_status().await
    }
    
    /// Get service metrics
    pub async fn get_metrics(&self) -> crate::metrics::IngestionMetrics {
        self.metrics_collector.get_metrics().await
    }
    
    async fn start_scheduler(&self) -> Result<tokio::task::JoinHandle<()>> {
        // Implementation would start the scheduler in the background
        Ok(tokio::spawn(async {}))
    }
    
    async fn start_health_checks(&self) -> Result<tokio::task::JoinHandle<()>> {
        // Implementation would start health checks in the background
        Ok(tokio::spawn(async {}))
    }
    
    async fn start_metrics_collection(&self) -> Result<tokio::task::JoinHandle<()>> {
        // Implementation would start metrics collection in the background
        Ok(tokio::spawn(async {}))
    }
} 