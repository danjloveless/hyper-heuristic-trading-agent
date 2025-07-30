use async_trait::async_trait;
use configuration_management::{ConfigurationManager, config_manager::ConfigurationManagerImpl, Environment, sources::{ConfigSource, EnvironmentSource, DotEnvSource}};
use core_traits::{ConfigurationProvider, ServiceResult, ServiceError};
use serde_json::Value;
use std::sync::Arc;

pub struct CoreConfigurationProvider {
    config_manager: Arc<ConfigurationManagerImpl>,
}

impl CoreConfigurationProvider {
    pub async fn new() -> ServiceResult<Self> {
        // Load .env file first so environment variables are available
        dotenv::from_path("../.env").ok();
        
        // Create configuration sources
        let mut sources: Vec<Box<dyn ConfigSource>> = Vec::new();
        
        // Add environment source
        sources.push(Box::new(EnvironmentSource::new()));
        
        // Add .env file source
        sources.push(Box::new(DotEnvSource::new("../.env".to_string())));
        
        // Create configuration manager
        let config_manager = Arc::new(
            ConfigurationManagerImpl::new(Environment::Development, sources).await
                .map_err(|e| ServiceError::Configuration {
                    message: format!("Failed to create configuration manager: {}", e),
                })?
        );
        
        Ok(Self { config_manager })
    }
}

#[async_trait]
impl ConfigurationProvider for CoreConfigurationProvider {
    async fn get_string(&self, key: &str) -> ServiceResult<String> {
        self.config_manager.get_config::<String>(key).await
            .map_err(|e| ServiceError::Configuration {
                message: format!("Failed to get string config for {}: {}", key, e),
            })
    }
    
    async fn get_u32(&self, key: &str) -> ServiceResult<u32> {
        self.config_manager.get_config::<u32>(key).await
            .map_err(|e| ServiceError::Configuration {
                message: format!("Failed to get u32 config for {}: {}", key, e),
            })
    }
    
    async fn get_u64(&self, key: &str) -> ServiceResult<u64> {
        self.config_manager.get_config::<u64>(key).await
            .map_err(|e| ServiceError::Configuration {
                message: format!("Failed to get u64 config for {}: {}", key, e),
            })
    }
    
    async fn get_bool(&self, key: &str) -> ServiceResult<bool> {
        self.config_manager.get_config::<bool>(key).await
            .map_err(|e| ServiceError::Configuration {
                message: format!("Failed to get bool config for {}: {}", key, e),
            })
    }
    
    async fn get_secret(&self, key: &str) -> ServiceResult<String> {
        self.config_manager.get_secret(key).await
            .map_err(|e| ServiceError::Configuration {
                message: format!("Failed to get secret for {}: {}", key, e),
            })
    }
    
    async fn get_alpha_vantage_config(&self) -> ServiceResult<Value> {
        // Get the API key from environment variables
        let api_key = self.config_manager.get_secret("ALPHA_VANTAGE_API_KEY").await
            .map_err(|e| ServiceError::Configuration {
                message: format!("Failed to get API key: {}", e),
            })?;
        
        // Create a simple alpha vantage config with the API key using core infrastructure structure
        let alpha_vantage_config = configuration_management::models::AlphaVantageConfig {
            base_url: "https://www.alphavantage.co/query".to_string(),
            api_key,
            rate_limit: configuration_management::models::RateLimitConfig {
                requests_per_minute: 5,
                requests_per_day: 500,
                burst_size: 2,
            },
            timeout_ms: 30000,
        };
        
        Ok(serde_json::to_value(alpha_vantage_config)
            .map_err(|e| ServiceError::Configuration {
                message: format!("Failed to serialize alpha vantage config: {}", e),
            })?)
    }
    
    async fn get_rate_limits_config(&self) -> ServiceResult<Value> {
        // Create a simple rate limits config using core infrastructure structure
        let rate_limits_config = configuration_management::models::RateLimitConfig {
            requests_per_minute: 5,
            requests_per_day: 500,
            burst_size: 2,
        };
        
        Ok(serde_json::to_value(rate_limits_config)
            .map_err(|e| ServiceError::Configuration {
                message: format!("Failed to serialize rate limits config: {}", e),
            })?)
    }
    
    async fn get_collection_config(&self) -> ServiceResult<Value> {
        // Create a simple collection config
        let collection_config = serde_json::json!({
            "default_symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
            "priority_symbols": ["SPY", "QQQ"],
            "batch_size": 100,
            "concurrent_collections": 5,
            "data_quality": {
                "min_quality_score": 70,
                "enable_validation": true,
                "max_price_deviation_percent": 10.0,
                "min_volume_threshold": 1000,
                "enable_deduplication": true
            }
        });
        
        Ok(collection_config)
    }
} 