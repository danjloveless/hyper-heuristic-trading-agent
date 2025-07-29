use crate::{errors::ConfigurationError, sources::ConfigSource};
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, info};

pub type Result<T> = std::result::Result<T, ConfigurationError>;

pub struct ConfigStore {
    sources: Vec<Box<dyn ConfigSource>>,
}

impl ConfigStore {
    pub async fn new(sources: Vec<Box<dyn ConfigSource>>) -> Result<Self> {
        info!("Initializing configuration store with {} sources", sources.len());
        Ok(Self { sources })
    }
    
    pub async fn load_config(&self, key: &str) -> Result<Value> {
        debug!("Loading configuration for key: {}", key);
        
        for source in &self.sources {
            match source.load_config(key).await {
                Ok(config) => {
                    debug!("Successfully loaded config for key {} from source", key);
                    return Ok(config);
                }
                Err(e) => {
                    debug!("Failed to load config for key {} from source: {}", key, e);
                    continue;
                }
            }
        }
        
        Err(ConfigurationError::ConfigNotFound {
            key: key.to_string(),
        })
    }
    
    pub async fn get_config(&self, key: &str) -> Result<Value> {
        self.load_config(key).await
    }
    
    pub async fn get_all_configs(&self, prefix: &str) -> Result<HashMap<String, Value>> {
        debug!("Getting all configurations with prefix: {}", prefix);
        
        let mut configs = HashMap::new();
        
        for source in &self.sources {
            match source.get_all_configs(prefix).await {
                Ok(source_configs) => {
                    for (key, value) in source_configs {
                        if !configs.contains_key(&key) {
                            configs.insert(key, value);
                        }
                    }
                }
                Err(e) => {
                    debug!("Failed to get configs from source: {}", e);
                    continue;
                }
            }
        }
        
        Ok(configs)
    }
    
    pub async fn set_config(&self, key: &str, value: Value) -> Result<()> {
        debug!("Setting configuration for key: {}", key);
        
        let mut last_error = None;
        
        for source in &self.sources {
            if source.supports_writes() {
                match source.set_config(key, value.clone()).await {
                    Ok(_) => {
                        debug!("Successfully set config for key {} in source", key);
                        return Ok(());
                    }
                    Err(e) => {
                        last_error = Some(e);
                        continue;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| {
            ConfigurationError::Internal {
                message: "No writable configuration sources available".to_string(),
            }
        }))
    }
    
    pub async fn delete_config(&self, key: &str) -> Result<()> {
        debug!("Deleting configuration for key: {}", key);
        
        let mut last_error = None;
        
        for source in &self.sources {
            if source.supports_writes() {
                match source.delete_config(key).await {
                    Ok(_) => {
                        debug!("Successfully deleted config for key {} from source", key);
                        return Ok(());
                    }
                    Err(e) => {
                        last_error = Some(e);
                        continue;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| {
            ConfigurationError::Internal {
                message: "No writable configuration sources available".to_string(),
            }
        }))
    }
    
    pub async fn list_configs(&self, prefix: &str) -> Result<Vec<String>> {
        debug!("Listing configurations with prefix: {}", prefix);
        
        let mut all_keys = std::collections::HashSet::new();
        
        for source in &self.sources {
            match source.list_configs(prefix).await {
                Ok(keys) => {
                    all_keys.extend(keys);
                }
                Err(e) => {
                    debug!("Failed to list configs from source: {}", e);
                    continue;
                }
            }
        }
        
        Ok(all_keys.into_iter().collect())
    }
} 