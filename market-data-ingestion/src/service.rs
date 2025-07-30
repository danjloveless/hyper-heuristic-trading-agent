use crate::{collectors::AlphaVantageCollector, processors::DataProcessor, config::*, models::{Interval, CollectionResult}};
use core_traits::*;
use database_abstraction::{DatabaseManager, CacheClient};
use shared_types::MarketData;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Instant;
use chrono::Utc;
use tracing::{info, warn, error, debug};

pub struct MarketDataIngestionService {
    // Core infrastructure dependencies (injected)
    config_provider: Arc<dyn ConfigurationProvider>,
    database_manager: Arc<DatabaseManager>,
    error_handler: Arc<dyn ErrorHandler>,
    monitoring: Arc<dyn MonitoringProvider>,
    
    // Service-specific components
    collector: AlphaVantageCollector,
    processor: DataProcessor,
    
    // Service metadata
    start_time: Instant,
    service_name: String,
}

impl MarketDataIngestionService {
    pub async fn new(
        config_provider: Arc<dyn ConfigurationProvider>,
        database_manager: Arc<DatabaseManager>,
        error_handler: Arc<dyn ErrorHandler>,
        monitoring: Arc<dyn MonitoringProvider>,
    ) -> ServiceResult<Self> {
        let service_name = "market-data-ingestion".to_string();
        
        info!("Initializing Market Data Ingestion Service");
        
        // Load configuration with proper error handling
        let context = ErrorContext::new(service_name.clone(), "initialization".to_string());
        
        let alpha_vantage_config_value = match config_provider.get_alpha_vantage_config().await {
            Ok(config_value) => config_value,
            Err(e) => {
                error_handler.report_error(&e, &context).await;
                return Err(e);
            }
        };
        
        let alpha_vantage_config: AlphaVantageConfig = serde_json::from_value(alpha_vantage_config_value)
            .map_err(|e| ServiceError::Configuration {
                message: format!("Failed to deserialize alpha_vantage config: {}", e),
            })?;
        
        let rate_limits_config_value = config_provider
            .get_rate_limits_config()
            .await
            .unwrap_or_else(|_| serde_json::to_value(RateLimitsConfig::default()).unwrap());
        
        let rate_limits_config: RateLimitsConfig = serde_json::from_value(rate_limits_config_value)
            .unwrap_or_default();
        
        let collection_config_value = config_provider
            .get_collection_config()
            .await
            .unwrap_or_else(|_| serde_json::to_value(CollectionConfig::default()).unwrap());
        
        let collection_config: CollectionConfig = serde_json::from_value(collection_config_value)
            .unwrap_or_default();
        
        // Initialize collector with error handling integration
        let collector = AlphaVantageCollector::new(
            alpha_vantage_config,
            rate_limits_config,
            error_handler.clone(),
            monitoring.clone(),
        ).await?;
        
        // Initialize processor with database integration
        let processor = DataProcessor::new(
            database_manager.clone(),
            error_handler.clone(),
            monitoring.clone(),
            collection_config.data_quality,
        );
        
        let service = Self {
            config_provider,
            database_manager,
            error_handler,
            monitoring,
            collector,
            processor,
            start_time: Instant::now(),
            service_name,
        };
        
        info!("Market Data Ingestion Service initialized successfully");
        Ok(service)
    }
    
    /// Start the service and background tasks
    pub async fn start(&self) -> ServiceResult<()> {
        info!("Starting Market Data Ingestion Service background tasks");
        
        // Register health checks with monitoring
        self.monitoring.log_info("Service started", &std::collections::HashMap::new()).await;
        
        // Start collector background tasks (if any)
        self.collector.start_background_tasks().await?;
        
        info!("Market Data Ingestion Service started successfully");
        Ok(())
    }
    
    /// Collect market data for a specific symbol
    pub async fn collect_symbol_data(&self, symbol: &str, interval: Interval) -> ServiceResult<CollectionResult> {
        let context = ErrorContext::new(
            self.service_name.clone(),
            format!("collect_{}_{}", symbol, interval.as_str())
        );
        
        let start_time = Instant::now();
        
        // Record attempt metric
        self.monitoring.record_counter("data_collection_attempts", &[
            ("symbol", symbol),
            ("interval", interval.as_str()),
        ]).await;
        
        // Collect data with error handling
        let raw_data = match self.collector.collect_symbol_data(symbol, interval).await {
            Ok(data) => data,
            Err(e) => {
                let decision = self.error_handler.handle_error(&e, &context).await;
                match decision {
                    ErrorDecision::Retry { delay, max_attempts } => {
                        return self.retry_collection(symbol, interval, delay, max_attempts).await;
                    }
                    ErrorDecision::UseCache => {
                        return self.get_cached_data(symbol, interval).await;
                    }
                    ErrorDecision::UseDefault => {
                        warn!("Using empty result for {}", symbol);
                        return Ok(CollectionResult::empty(symbol, interval));
                    }
                    ErrorDecision::Fail => {
                        self.monitoring.record_counter("data_collection_failures", &[
                            ("symbol", symbol),
                            ("interval", interval.as_str()),
                            ("error_type", "permanent"),
                        ]).await;
                        return Err(e);
                    }
                }
            }
        };
        
        // Process the collected data
        let result = match self.processor.process_market_data(raw_data.clone(), symbol, interval).await {
            Ok(processed) => processed,
            Err(e) => {
                let decision = self.error_handler.handle_error(&e, &context).await;
                match decision {
                    ErrorDecision::UseCache => {
                        // Store raw data in cache and return partial success
                        warn!("Database storage failed, caching data for {}", symbol);
                        self.cache_raw_data(&raw_data, symbol, &interval).await?;
                        CollectionResult::cached_only(symbol, interval, raw_data.len())
                    }
                    _ => return Err(e),
                }
            }
        };
        
        // Record success metrics
        let duration = start_time.elapsed();
        self.monitoring.record_timing("data_collection_duration", duration, &[
            ("symbol", symbol),
            ("interval", interval.as_str()),
        ]).await;
        
        self.monitoring.record_metric("data_points_collected", result.processed_count as f64, &[
            ("symbol", symbol),
            ("interval", interval.as_str()),
        ]).await;
        
        debug!("Successfully collected {} data points for {} in {:?}", 
               result.processed_count, symbol, duration);
        
        Ok(result)
    }
    
    /// Retry collection with exponential backoff
    async fn retry_collection(
        &self,
        symbol: &str,
        interval: Interval,
        delay: std::time::Duration,
        max_attempts: u32,
    ) -> ServiceResult<CollectionResult> {
        for attempt in 1..=max_attempts {
            tokio::time::sleep(delay * attempt).await;
            
            match self.collector.collect_symbol_data(symbol, interval).await {
                Ok(data) => {
                    info!("Retry successful for {} on attempt {}", symbol, attempt);
                    return self.processor.process_market_data(data, symbol, interval).await;
                }
                Err(e) if attempt == max_attempts => {
                    error!("All retry attempts failed for {}: {:?}", symbol, e);
                    return Err(e);
                }
                Err(e) => {
                    warn!("Retry attempt {} failed for {}: {:?}", attempt, symbol, e);
                }
            }
        }
        
        unreachable!()
    }
    
    /// Get cached data as fallback
    async fn get_cached_data(&self, symbol: &str, interval: Interval) -> ServiceResult<CollectionResult> {
        let redis = self.database_manager.redis();
        let cache_key = format!("market_data:{}:{}", symbol, interval.as_str());
        
        match redis.cache_get::<Vec<MarketData>>(&cache_key).await {
            Ok(Some(cached_data)) => {
                info!("Retrieved {} cached data points for {}", cached_data.len(), symbol);
                Ok(CollectionResult::from_cache(symbol, interval, cached_data))
            }
            Ok(None) => {
                warn!("No cached data available for {}", symbol);
                Ok(CollectionResult::empty(symbol, interval))
            }
            Err(e) => {
                error!("Failed to retrieve cached data for {}: {:?}", symbol, e);
                Err(ServiceError::Database {
                    message: format!("Cache retrieval failed: {:?}", e),
                    retryable: true,
                })
            }
        }
    }
    
    /// Cache raw data as fallback
    async fn cache_raw_data(
        &self,
        raw_data: &[crate::models::RawMarketData],  
        symbol: &str,
        interval: &Interval,
    ) -> ServiceResult<()> {
        let redis = self.database_manager.redis();
        let cache_key = format!("raw_market_data:{}:{}", symbol, interval.as_str());
        
        redis.cache_set(&cache_key, &raw_data.to_vec(), Some(chrono::Duration::hours(1))).await // 1 hour TTL
            .map_err(|e| ServiceError::Database {
                message: format!("Failed to cache raw data: {:?}", e),
                retryable: true,
            })
    }
}

#[async_trait]
impl HealthCheckable for MarketDataIngestionService {
    async fn health_check(&self) -> HealthStatus {
        let mut health = HealthStatus::healthy(self.service_name.clone());
        health.uptime_seconds = self.start_time.elapsed().as_secs();
        
        // Check database health
        let db_health_result = self.database_manager.health_check().await;
        match db_health_result {
            Ok(db_health) => {
                health.add_component_check("clickhouse".to_string(), ComponentHealth {
                    name: "clickhouse".to_string(),
                    status: if db_health.clickhouse_healthy { ServiceStatus::Healthy } else { ServiceStatus::Unhealthy },
                    message: Some("ClickHouse database connection".to_string()),
                    last_check: Utc::now(),
                    response_time_ms: 0, // TODO: Add actual timing
                });
                
                health.add_component_check("redis".to_string(), ComponentHealth {
                    name: "redis".to_string(),
                    status: if db_health.redis_healthy { ServiceStatus::Healthy } else { ServiceStatus::Unhealthy },
                    message: Some("Redis cache connection".to_string()),
                    last_check: Utc::now(),
                    response_time_ms: 0,
                });
            }
            Err(_) => {
                health.add_component_check("database".to_string(), ComponentHealth {
                    name: "database".to_string(),
                    status: ServiceStatus::Unhealthy,
                    message: Some("Database health check failed".to_string()),
                    last_check: Utc::now(),
                    response_time_ms: 0,
                });
            }
        }
        
        // Check Alpha Vantage connectivity
        let api_health = self.collector.health_check().await;
        health.add_component_check("alpha_vantage".to_string(), ComponentHealth {
            name: "alpha_vantage".to_string(),
            status: if api_health { ServiceStatus::Healthy } else { ServiceStatus::Degraded },
            message: Some("Alpha Vantage API connectivity".to_string()),
            last_check: Utc::now(),
            response_time_ms: 0,
        });
        
        health
    }
    
    async fn ready_check(&self) -> ReadinessStatus {
        let health = self.health_check().await;
        
        // Service is ready if core components are healthy
        let critical_components = ["clickhouse", "alpha_vantage"];
        for component in &critical_components {
            if let Some(check) = health.checks.get(*component) {
                if check.status == ServiceStatus::Unhealthy {
                    return ReadinessStatus::NotReady {
                        reason: format!("Critical component {} is unhealthy", component),
                    };
                }
            }
        }
        
        ReadinessStatus::Ready
    }
    
    fn service_name(&self) -> &str {
        &self.service_name
    }
} 