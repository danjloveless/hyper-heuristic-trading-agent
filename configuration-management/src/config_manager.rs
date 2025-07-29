use crate::{
    config_store::ConfigStore,
    config_validator::ConfigValidator,
    config_watcher::ConfigWatcher,
    errors::ConfigurationError,
    feature_flags::FeatureFlags,
    models::*,
    secret_manager::SecretManager,
    sources::ConfigSource,
};
use async_trait::async_trait;
use chrono::Utc;
use dashmap::DashMap;
use futures::stream::StreamExt;
use tokio::sync::Mutex;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
};
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

pub type Result<T> = std::result::Result<T, ConfigurationError>;

#[async_trait]
pub trait ConfigurationManager: Send + Sync {
    // Configuration retrieval
    async fn get_config<T: DeserializeOwned>(&self, key: &str) -> Result<T>;
    async fn get_config_with_default<T: DeserializeOwned + Send>(&self, key: &str, default: T) -> Result<T>;
    async fn get_all_configs(&self, prefix: &str) -> Result<HashMap<String, serde_json::Value>>;
    
    // Secret management
    async fn get_secret(&self, secret_name: &str) -> Result<String>;
    async fn set_secret(&self, secret_name: &str, value: &str) -> Result<()>;
    async fn rotate_secret(&self, secret_name: &str) -> Result<String>;
    
    // Feature flags
    async fn is_feature_enabled(&self, feature_name: &str) -> Result<bool>;
    async fn get_feature_config<T: DeserializeOwned>(&self, feature_name: &str) -> Result<Option<T>>;
    async fn enable_feature(&self, feature_name: &str, config: Option<serde_json::Value>) -> Result<()>;
    async fn disable_feature(&self, feature_name: &str) -> Result<()>;
    
    // Configuration updates
    async fn update_config(&self, key: &str, value: serde_json::Value) -> Result<()>;
    async fn reload_configuration(&self) -> Result<()>;
    async fn validate_configuration(&self) -> Result<ValidationResult>;
    
    // Watchers and notifications
    async fn watch_config(&self, key: &str) -> Result<ConfigWatcher>;
    async fn subscribe_to_changes(&self) -> Result<ConfigChangeStream>;
    
    // Service-specific configuration methods
    async fn get_service_config(&self) -> Result<ServiceConfig>;
    async fn get_database_config(&self) -> Result<DatabaseConfig>;
    async fn get_api_config(&self) -> Result<ApiConfig>;
    async fn get_logging_config(&self) -> Result<LoggingConfig>;
    async fn get_monitoring_config(&self) -> Result<MonitoringConfig>;
    async fn get_market_data_config(&self) -> Result<MarketDataIngestionConfig>;
    async fn get_technical_indicators_config(&self) -> Result<TechnicalIndicatorsConfig>;
    async fn get_prediction_service_config(&self) -> Result<PredictionServiceConfig>;
}

pub struct ConfigurationManagerImpl {
    config_store: Arc<ConfigStore>,
    secret_manager: Arc<Mutex<SecretManager>>,
    feature_flags: Arc<Mutex<FeatureFlags>>,
    config_validator: Arc<ConfigValidator>,
    config_watcher: Arc<ConfigWatcher>,
    change_sender: broadcast::Sender<ConfigurationChange>,
    cache: Arc<DashMap<String, serde_json::Value>>,
    environment: Environment,
}

impl ConfigurationManagerImpl {
    pub async fn new(
        environment: Environment,
        config_sources: Vec<Box<dyn ConfigSource>>,
    ) -> Result<Self> {
        let (change_sender, _) = broadcast::channel(1000);
        
        let config_store = Arc::new(ConfigStore::new(config_sources).await?);
        let secret_manager = Arc::new(Mutex::new(SecretManager::new().await?));
        let feature_flags = Arc::new(Mutex::new(FeatureFlags::new().await?));
        let config_validator = Arc::new(ConfigValidator::new().await?);
        let config_watcher = Arc::new(ConfigWatcher::new(change_sender.clone()).await?);
        
        let cache = Arc::new(DashMap::new());
        
        let manager = Self {
            config_store,
            secret_manager,
            feature_flags,
            config_validator,
            config_watcher,
            change_sender,
            cache,
            environment,
        };
        
        // Load initial configuration
        manager.load_initial_config().await?;
        
        // Start watching for configuration changes
        manager.start_config_watcher().await?;
        
        Ok(manager)
    }
    
    async fn load_initial_config(&self) -> Result<()> {
        info!("Loading initial configuration for environment: {}", self.environment);
        
        // Load base configuration with fallback to defaults
        match self.config_store.load_config("base").await {
            Ok(base_config) => {
                self.cache.insert("base".to_string(), base_config);
                info!("Loaded base configuration");
            }
            Err(ConfigurationError::ConfigNotFound { .. }) => {
                warn!("Base configuration not found, using defaults");
                let default_config = serde_json::to_value(ServiceConfig::default())?;
                self.cache.insert("base".to_string(), default_config);
            }
            Err(e) => {
                warn!("Failed to load base configuration: {}, using defaults", e);
                let default_config = serde_json::to_value(ServiceConfig::default())?;
                self.cache.insert("base".to_string(), default_config);
            }
        }
        
        // Load environment-specific configuration with fallback
        match self.config_store.load_config(&self.environment.to_string()).await {
            Ok(env_config) => {
                self.cache.insert(self.environment.to_string(), env_config);
                info!("Loaded environment-specific configuration");
            }
            Err(ConfigurationError::ConfigNotFound { .. }) => {
                warn!("Environment-specific configuration not found, using base config");
            }
            Err(e) => {
                warn!("Failed to load environment configuration: {}, using base config", e);
            }
        }
        
        // Load service-specific configurations
        self.load_service_configs().await?;
        
        // Validate configuration
        match self.validate_configuration().await {
            Ok(validation_result) => {
                if !validation_result.is_valid {
                    warn!("Configuration validation failed with {} errors, {} warnings", 
                          validation_result.errors.len(), validation_result.warnings.len());
                    for error in &validation_result.errors {
                        error!("Configuration error: {} - {}", error.field, error.message);
                    }
                    for warning in &validation_result.warnings {
                        warn!("Configuration warning: {} - {}", warning.field, warning.message);
                    }
                } else {
                    info!("Configuration validation passed");
                }
            }
            Err(e) => {
                warn!("Configuration validation failed: {}", e);
            }
        }
        
        info!("Initial configuration loaded successfully");
        Ok(())
    }
    
    async fn load_service_configs(&self) -> Result<()> {
        let service_configs: Vec<(&str, serde_json::Value)> = vec![
            ("market_data_ingestion", serde_json::to_value(MarketDataIngestionConfig::default())?),
            ("technical_indicators", serde_json::to_value(TechnicalIndicatorsConfig::default())?),
            ("prediction_service", serde_json::to_value(PredictionServiceConfig::default())?),
        ];
        
        for (key, default_config) in service_configs {
            match self.config_store.load_config(key).await {
                Ok(config) => {
                    self.cache.insert(key.to_string(), config);
                    debug!("Loaded service configuration: {}", key);
                }
                Err(ConfigurationError::ConfigNotFound { .. }) => {
                    self.cache.insert(key.to_string(), default_config);
                    debug!("Using default configuration for service: {}", key);
                }
                Err(e) => {
                    warn!("Failed to load service configuration {}: {}, using defaults", key, e);
                    self.cache.insert(key.to_string(), default_config);
                }
            }
        }
        
        Ok(())
    }
    
    async fn start_config_watcher(&self) -> Result<()> {
        let mut change_stream = self.config_watcher.watch_changes().await?;
        let change_sender = self.change_sender.clone();
        let cache = self.cache.clone();
        
        tokio::spawn(async move {
            while let Some(change) = change_stream.next().await {
                match change {
                    Ok(config_change) => {
                        // Update cache
                        cache.insert(config_change.key.clone(), config_change.new_value.clone());
                        
                        // Broadcast change
                        if let Err(e) = change_sender.send(config_change) {
                            warn!("Failed to broadcast configuration change: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Configuration watcher error: {}", e);
                    }
                }
            }
        });
        
        Ok(())
    }
    
    async fn get_cached_config<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        if let Some(cached_value) = self.cache.get(key) {
            match serde_json::from_value::<T>(cached_value.clone()) {
                Ok(config) => Ok(Some(config)),
                Err(e) => {
                    warn!("Failed to deserialize cached config for key {}: {}", key, e);
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }
    
    #[allow(dead_code)]
    async fn set_cached_config<T: Serialize>(&self, key: &str, config: &T) -> Result<()> {
        let value = serde_json::to_value(config)?;
        self.cache.insert(key.to_string(), value);
        Ok(())
    }
}

#[async_trait]
impl ConfigurationManager for ConfigurationManagerImpl {
    async fn get_config<T: DeserializeOwned>(&self, key: &str) -> Result<T> {
        debug!("Getting configuration for key: {}", key);
        
        // Try cache first
        if let Some(cached_config) = self.get_cached_config::<T>(key).await? {
            return Ok(cached_config);
        }
        
        // Load from store
        let config_value = self.config_store.get_config(key).await?;
        let config: T = serde_json::from_value(config_value.clone())?;
        
        // Cache the result
        self.cache.insert(key.to_string(), config_value);
        
        Ok(config)
    }
    
    async fn get_config_with_default<T: DeserializeOwned + Send>(&self, key: &str, default: T) -> Result<T> {
        match self.get_config::<T>(key).await {
            Ok(config) => Ok(config),
            Err(ConfigurationError::ConfigNotFound { .. }) => {
                debug!("Configuration not found for key: {}, using default", key);
                Ok(default)
            }
            Err(e) => Err(e),
        }
    }
    
    async fn get_all_configs(&self, prefix: &str) -> Result<HashMap<String, serde_json::Value>> {
        debug!("Getting all configurations with prefix: {}", prefix);
        
        let mut configs = HashMap::new();
        
        // Get from cache first
        for entry in self.cache.iter() {
            if entry.key().starts_with(prefix) {
                configs.insert(entry.key().clone(), entry.value().clone());
            }
        }
        
        // Get from store for any missing configs
        let store_configs = self.config_store.get_all_configs(prefix).await?;
        for (key, value) in store_configs {
            if !configs.contains_key(&key) {
                configs.insert(key, value);
            }
        }
        
        Ok(configs)
    }
    
    async fn get_secret(&self, secret_name: &str) -> Result<String> {
        debug!("Getting secret: {}", secret_name);
        let mut secret_manager = self.secret_manager.lock().await;
        secret_manager.get_secret(secret_name).await
    }
    
    async fn set_secret(&self, secret_name: &str, value: &str) -> Result<()> {
        debug!("Setting secret: {}", secret_name);
        let secret_manager = self.secret_manager.lock().await;
        secret_manager.set_secret(secret_name, value).await
    }
    
    async fn rotate_secret(&self, secret_name: &str) -> Result<String> {
        debug!("Rotating secret: {}", secret_name);
        let secret_manager = self.secret_manager.lock().await;
        secret_manager.rotate_secret(secret_name).await
    }
    
    async fn is_feature_enabled(&self, feature_name: &str) -> Result<bool> {
        debug!("Checking if feature is enabled: {}", feature_name);
        let feature_flags = self.feature_flags.lock().await;
        feature_flags.is_enabled(feature_name).await
    }
    
    async fn get_feature_config<T: DeserializeOwned>(&self, feature_name: &str) -> Result<Option<T>> {
        debug!("Getting feature config: {}", feature_name);
        let feature_flags = self.feature_flags.lock().await;
        feature_flags.get_config::<T>(feature_name).await
    }
    
    async fn enable_feature(&self, feature_name: &str, config: Option<serde_json::Value>) -> Result<()> {
        debug!("Enabling feature: {}", feature_name);
        let mut feature_flags = self.feature_flags.lock().await;
        feature_flags.enable_feature(feature_name, config).await
    }
    
    async fn disable_feature(&self, feature_name: &str) -> Result<()> {
        debug!("Disabling feature: {}", feature_name);
        let mut feature_flags = self.feature_flags.lock().await;
        feature_flags.disable_feature(feature_name).await
    }
    
    async fn update_config(&self, key: &str, value: serde_json::Value) -> Result<()> {
        debug!("Updating configuration: {}", key);
        
        // Validate the configuration
        self.config_validator.validate_config(key, &value).await?;
        
        // Update in store
        self.config_store.set_config(key, value.clone()).await?;
        
        // Update cache
        self.cache.insert(key.to_string(), value.clone());
        
        // Broadcast change
        let change = ConfigurationChange {
            key: key.to_string(),
            old_value: self.cache.get(key).map(|v| v.clone()),
            new_value: value,
            timestamp: Utc::now(),
            source: "configuration_manager".to_string(),
            user: None,
        };
        
        if let Err(e) = self.change_sender.send(change) {
            warn!("Failed to broadcast configuration change: {}", e);
        }
        
        Ok(())
    }
    
    async fn reload_configuration(&self) -> Result<()> {
        info!("Reloading configuration");
        
        // Clear cache
        self.cache.clear();
        
        // Reload configuration
        self.load_initial_config().await?;
        
        info!("Configuration reloaded successfully");
        Ok(())
    }
    
    async fn validate_configuration(&self) -> Result<ValidationResult> {
        debug!("Validating configuration");
        
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Validate all cached configurations
        for entry in self.cache.iter() {
            match self.config_validator.validate_config(entry.key(), entry.value()).await {
                Ok(validation_result) => {
                    errors.extend(validation_result.errors);
                    warnings.extend(validation_result.warnings);
                }
                Err(e) => {
                    errors.push(ValidationError {
                        field: entry.key().clone(),
                        message: format!("Validation failed: {}", e),
                        severity: ValidationSeverity::Error,
                    });
                }
            }
        }
        
        let is_valid = errors.is_empty();
        
        Ok(ValidationResult {
            is_valid,
            errors,
            warnings,
        })
    }
    
    async fn watch_config(&self, key: &str) -> Result<ConfigWatcher> {
        debug!("Setting up config watcher for key: {}", key);
        self.config_watcher.watch_config(key).await
    }
    
    async fn subscribe_to_changes(&self) -> Result<ConfigChangeStream> {
        debug!("Subscribing to configuration changes");
        Ok(self.change_sender.subscribe())
    }
    
    // Service-specific configuration methods
    async fn get_service_config(&self) -> Result<ServiceConfig> {
        self.get_config("service").await
    }
    
    async fn get_database_config(&self) -> Result<DatabaseConfig> {
        self.get_config("database").await
    }
    
    async fn get_api_config(&self) -> Result<ApiConfig> {
        self.get_config("apis").await
    }
    
    async fn get_logging_config(&self) -> Result<LoggingConfig> {
        self.get_config("logging").await
    }
    
    async fn get_monitoring_config(&self) -> Result<MonitoringConfig> {
        self.get_config("monitoring").await
    }
    
    async fn get_market_data_config(&self) -> Result<MarketDataIngestionConfig> {
        self.get_config("market_data_ingestion").await
    }
    
    async fn get_technical_indicators_config(&self) -> Result<TechnicalIndicatorsConfig> {
        self.get_config("technical_indicators").await
    }
    
    async fn get_prediction_service_config(&self) -> Result<PredictionServiceConfig> {
        self.get_config("prediction_service").await
    }
}

 