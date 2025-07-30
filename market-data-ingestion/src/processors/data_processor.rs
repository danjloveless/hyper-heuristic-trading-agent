use crate::models::*;
use configuration_management::models::DataQualityConfig;
use core_traits::*;
use database_abstraction::{DatabaseManager, CacheClient, DatabaseClient};
use shared_types::MarketData;
use std::sync::Arc;
use std::time::Instant;
use rust_decimal::{Decimal, prelude::FromPrimitive};
use tracing::{debug, info, warn};

pub struct DataProcessor {
    database_manager: Arc<DatabaseManager>,
    error_handler: Arc<dyn ErrorHandler>,
    monitoring: Arc<dyn MonitoringProvider>,
    quality_config: DataQualityConfig,
}

impl DataProcessor {
    pub fn new(
        database_manager: Arc<DatabaseManager>,
        error_handler: Arc<dyn ErrorHandler>,
        monitoring: Arc<dyn MonitoringProvider>,
        quality_config: DataQualityConfig,
    ) -> Self {
        Self {
            database_manager,
            error_handler,
            monitoring,
            quality_config,
        }
    }
    
    pub async fn process_market_data(
        &self,
        raw_data: Vec<RawMarketData>,
        symbol: &str,
        interval: Interval,
    ) -> ServiceResult<CollectionResult> {
        let start_time = Instant::now();
        let context = ErrorContext::new(
            "data_processor".to_string(),
            format!("process_{}_{}", symbol, interval.as_str()),
        );
        
        if raw_data.is_empty() {
            debug!("No raw data to process for {}", symbol);
            return Ok(CollectionResult::empty(symbol, interval));
        }
        
        // Step 1: Convert raw data to MarketData with validation
        let mut processed_data = Vec::new();
        let mut quality_scores = Vec::new();
        
        for raw_item in raw_data {
            match self.convert_and_validate_data_point(&raw_item, symbol).await {
                Ok((market_data, quality_score)) => {
                    if quality_score >= self.quality_config.min_quality_score {
                        processed_data.push(market_data);
                        quality_scores.push(quality_score);
                    } else {
                        warn!("Data point for {} failed quality check (score: {})", symbol, quality_score);
                        self.monitoring.record_counter("data_quality_failures", &[
                            ("symbol", symbol),
                            ("reason", "low_quality_score"),
                        ]).await;
                    }
                }
                Err(e) => {
                    warn!("Failed to convert data point for {}: {:?}", symbol, e);
                    self.monitoring.record_counter("data_conversion_errors", &[
                        ("symbol", symbol),
                        ("error_type", "conversion_failed"),
                    ]).await;
                }
            }
        }
        
        if processed_data.is_empty() {
            warn!("No valid data points after processing for {}", symbol);
            return Ok(CollectionResult::empty(symbol, interval));
        }
        
        // Step 2: Deduplication
        // Note: enable_deduplication is not in the core DataQualityConfig, so we'll default to true
        let enable_deduplication = true;
        if enable_deduplication {
            let original_count = processed_data.len();
            processed_data = self.deduplicate_data(processed_data).await;
            
            if processed_data.len() < original_count {
                let duplicates_removed = original_count - processed_data.len();
                debug!("Removed {} duplicate data points for {}", duplicates_removed, symbol);
                self.monitoring.record_metric("duplicates_removed", duplicates_removed as f64, &[
                    ("symbol", symbol),
                ]).await;
            }
        }
        
        // Step 3: Store in ClickHouse
        let storage_result = self.store_to_clickhouse(&processed_data).await;
        let mut result = match storage_result {
            Ok(_) => {
                info!("Successfully stored {} data points for {} to ClickHouse", 
                      processed_data.len(), symbol);
                
                self.monitoring.record_metric("data_points_stored", processed_data.len() as f64, &[
                    ("symbol", symbol),
                    ("storage", "clickhouse"),
                ]).await;
                
                CollectionResult::new(symbol, interval, processed_data.len())
            }
            Err(e) => {
                let decision = self.error_handler.handle_error(&e, &context).await;
                match decision {
                    ErrorDecision::UseCache => {
                        // Store in Redis as fallback
                        warn!("ClickHouse storage failed for {}, using Redis cache", symbol);
                        self.store_to_redis_cache(&processed_data, symbol, &interval).await?;
                        
                        let mut result = CollectionResult::cached_only(symbol, interval, processed_data.len());
                        result.processed_count = processed_data.len();
                        result
                    }
                    _ => return Err(e),
                }
            }
        };
        
        // Step 4: Cache recent data in Redis for fast access
        if result.processed_count > 0 {
            if let Err(e) = self.cache_recent_data(&processed_data, symbol).await {
                warn!("Failed to cache recent data for {}: {:?}", symbol, e);
                // Don't fail the entire operation for cache errors
            }
        }
        
        // Step 5: Calculate processing metrics
        let processing_duration = start_time.elapsed();
        result.processing_duration_ms = processing_duration.as_millis() as u64;
        
        // Calculate average quality score
        if !quality_scores.is_empty() {
            let avg_quality = quality_scores.iter().sum::<u8>() as f64 / quality_scores.len() as f64;
            result.quality_score = Some(avg_quality as u8);
            
            self.monitoring.record_metric("average_data_quality", avg_quality, &[
                ("symbol", symbol),
            ]).await;
        }
        
        self.monitoring.record_timing("data_processing_duration", processing_duration, &[
            ("symbol", symbol),
            ("interval", interval.as_str()),
        ]).await;
        
        debug!("Processed {} data points for {} in {:?}", 
               result.processed_count, symbol, processing_duration);
        
        Ok(result)
    }
    
    async fn convert_and_validate_data_point(
        &self,
        raw_data: &RawMarketData,
        symbol: &str,
    ) -> ServiceResult<(MarketData, u8)> {
        // Convert string values to Decimal
        let open = self.parse_decimal(&raw_data.open, "open")?;
        let high = self.parse_decimal(&raw_data.high, "high")?;
        let low = self.parse_decimal(&raw_data.low, "low")?;
        let close = self.parse_decimal(&raw_data.close, "close")?;
        let volume = raw_data.volume.parse::<u64>()
            .map_err(|e| ServiceError::System {
                message: format!("Failed to parse volume '{}': {}", raw_data.volume, e),
            })?;
        
        let adjusted_close = if let Some(adj_close_str) = &raw_data.adjusted_close {
            self.parse_decimal(adj_close_str, "adjusted_close")?
        } else {
            close // Use close price if adjusted_close not available
        };
        
        let market_data = MarketData {
            symbol: symbol.to_string(),
            timestamp: raw_data.timestamp,
            open,
            high,
            low,
            close,
            volume,
            adjusted_close,
        };
        
        // Validate data quality
        let quality_score = self.calculate_quality_score(&market_data).await;
        
        Ok((market_data, quality_score))
    }
    
    async fn calculate_quality_score(&self, data: &MarketData) -> u8 {
        let mut score = 100u8;
        
        // Check basic data integrity
        if data.high < data.low {
            score = score.saturating_sub(50); // Major error
        }
        
        if data.open > data.high || data.open < data.low {
            score = score.saturating_sub(20);
        }
        
        if data.close > data.high || data.close < data.low {
            score = score.saturating_sub(20);
        }
        
        // Check for reasonable price ranges (no negative prices)
        if data.open <= Decimal::ZERO || data.high <= Decimal::ZERO || 
           data.low <= Decimal::ZERO || data.close <= Decimal::ZERO {
            score = score.saturating_sub(50);
        }
        
        // Check volume threshold (using a reasonable default since it's not in core config)
        let min_volume_threshold = 1000u64; // Default value
        if data.volume < min_volume_threshold {
            score = score.saturating_sub(10);
        }
        
        // Check for extreme price movements (using a reasonable default)
        if data.open > Decimal::ZERO {
            let price_change = ((data.close - data.open) / data.open * Decimal::from(100)).abs();
            let max_deviation = Decimal::from_f64(10.0) // Default 10% deviation
                .unwrap_or(Decimal::from(10));
            
            if price_change > max_deviation {
                score = score.saturating_sub(15);
            }
        }
        
        score
    }
    
    fn parse_decimal(&self, value_str: &str, field_name: &str) -> ServiceResult<Decimal> {
        value_str.parse::<Decimal>()
            .map_err(|e| ServiceError::System {
                message: format!("Failed to parse {} '{}': {}", field_name, value_str, e),
            })
    }
    
    async fn deduplicate_data(&self, mut data: Vec<MarketData>) -> Vec<MarketData> {
        // Sort by timestamp and remove duplicates
        data.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        
        let mut deduplicated = Vec::new();
        let mut last_timestamp = None;
        
        for item in data {
            if last_timestamp != Some(item.timestamp) {
                deduplicated.push(item.clone());
                last_timestamp = Some(item.timestamp);
            }
        }
        
        deduplicated
    }
    
    async fn store_to_clickhouse(&self, data: &[MarketData]) -> ServiceResult<()> {
        let clickhouse = self.database_manager.clickhouse();
        
        clickhouse.insert_market_data(data).await
            .map_err(|e| ServiceError::Database {
                message: format!("ClickHouse insertion failed: {:?}", e),
                retryable: true,
            })
    }
    
    async fn store_to_redis_cache(
        &self,
        data: &[MarketData],
        symbol: &str,
        interval: &Interval,
    ) -> ServiceResult<()> {
        let redis = self.database_manager.redis();
        let cache_key = format!("fallback_data:{}:{}", symbol, interval.as_str());
        
        // Store with 1 hour TTL as fallback data
        redis.cache_set(&cache_key, &data.to_vec(), Some(chrono::Duration::hours(1))).await
            .map_err(|e| ServiceError::Database {
                message: format!("Redis cache storage failed: {:?}", e),
                retryable: true,
            })
    }
    
    async fn cache_recent_data(&self, data: &[MarketData], symbol: &str) -> ServiceResult<()> {
        let redis = self.database_manager.redis();
        
        // Cache the latest data point for quick access
        if let Some(latest) = data.last() {
            let cache_key = format!("market_data:{}:latest", symbol);
            redis.cache_set(&cache_key, latest, Some(chrono::Duration::minutes(5))).await // 5 minute TTL
                .map_err(|e| ServiceError::Database {
                    message: format!("Failed to cache latest data: {:?}", e),
                    retryable: true,
                })?;
        }
        
        // Cache the full dataset for recent access
        let recent_cache_key = format!("market_data:{}:recent", symbol);
        redis.cache_set(&recent_cache_key, &data.to_vec(), Some(chrono::Duration::minutes(15))).await // 15 minute TTL
            .map_err(|e| ServiceError::Database {
                message: format!("Failed to cache recent data: {:?}", e),
                retryable: true,
            })
    }
} 