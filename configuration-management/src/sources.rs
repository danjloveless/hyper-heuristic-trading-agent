use crate::errors::ConfigurationError;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, error};

pub type Result<T> = std::result::Result<T, ConfigurationError>;

#[async_trait]
pub trait ConfigSource: Send + Sync {
    async fn load_config(&self, key: &str) -> Result<Value>;
    async fn get_all_configs(&self, prefix: &str) -> Result<HashMap<String, Value>>;
    async fn set_config(&self, key: &str, value: Value) -> Result<()>;
    async fn delete_config(&self, key: &str) -> Result<()>;
    async fn list_configs(&self, prefix: &str) -> Result<Vec<String>>;
    fn supports_writes(&self) -> bool;
    fn get_source_name(&self) -> &str;
}

// Environment Variables Source
pub struct EnvironmentSource;

impl EnvironmentSource {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ConfigSource for EnvironmentSource {
    async fn load_config(&self, key: &str) -> Result<Value> {
        debug!("Loading config from environment: {}", key);
        
        if let Ok(value) = std::env::var(key) {
            // Try to parse as JSON first, fallback to string
            if let Ok(json_value) = serde_json::from_str::<Value>(&value) {
                Ok(json_value)
            } else {
                Ok(Value::String(value))
            }
        } else {
            Err(ConfigurationError::ConfigNotFound {
                key: key.to_string(),
            })
        }
    }
    
    async fn get_all_configs(&self, prefix: &str) -> Result<HashMap<String, Value>> {
        debug!("Getting all configs from environment with prefix: {}", prefix);
        
        let mut configs = HashMap::new();
        
        for (key, value) in std::env::vars() {
            if key.starts_with(prefix) {
                // Try to parse as JSON first, fallback to string
                if let Ok(json_value) = serde_json::from_str::<Value>(&value) {
                    configs.insert(key, json_value);
                } else {
                    configs.insert(key, Value::String(value));
                }
            }
        }
        
        Ok(configs)
    }
    
    async fn set_config(&self, _key: &str, _value: Value) -> Result<()> {
        Err(ConfigurationError::Internal {
            message: "Environment variables are read-only".to_string(),
        })
    }
    
    async fn delete_config(&self, _key: &str) -> Result<()> {
        Err(ConfigurationError::Internal {
            message: "Environment variables are read-only".to_string(),
        })
    }
    
    async fn list_configs(&self, prefix: &str) -> Result<Vec<String>> {
        debug!("Listing configs from environment with prefix: {}", prefix);
        
        let keys: Vec<String> = std::env::vars()
            .filter_map(|(key, _)| {
                if key.starts_with(prefix) {
                    Some(key)
                } else {
                    None
                }
            })
            .collect();
        
        Ok(keys)
    }
    
    fn supports_writes(&self) -> bool {
        false
    }
    
    fn get_source_name(&self) -> &str {
        "environment"
    }
}

// File Source
pub struct FileSource {
    base_path: String,
}

impl FileSource {
    pub fn new(base_path: String) -> Self {
        Self { base_path }
    }
}

#[async_trait]
impl ConfigSource for FileSource {
    async fn load_config(&self, key: &str) -> Result<Value> {
        debug!("Loading config from file: {}", key);
        
        let file_path = format!("{}/{}.json", self.base_path, key);
        
        match tokio::fs::read_to_string(&file_path).await {
            Ok(content) => {
                let value = serde_json::from_str::<Value>(&content)?;
                Ok(value)
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Err(ConfigurationError::ConfigNotFound {
                        key: key.to_string(),
                    })
                } else {
                    Err(ConfigurationError::Internal {
                        message: format!("Failed to read file {}: {}", file_path, e),
                    })
                }
            }
        }
    }
    
    async fn get_all_configs(&self, prefix: &str) -> Result<HashMap<String, Value>> {
        debug!("Getting all configs from files with prefix: {}", prefix);
        
        let mut configs = HashMap::new();
        
        match tokio::fs::read_dir(&self.base_path).await {
            Ok(mut entries) => {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    if let Ok(file_name) = entry.file_name().into_string() {
                        if file_name.starts_with(prefix) && file_name.ends_with(".json") {
                            let key = file_name.trim_end_matches(".json");
                            if let Ok(content) = tokio::fs::read_to_string(entry.path()).await {
                                if let Ok(value) = serde_json::from_str::<Value>(&content) {
                                    configs.insert(key.to_string(), value);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                return Err(ConfigurationError::Internal {
                    message: format!("Failed to read directory {}: {}", self.base_path, e),
                });
            }
        }
        
        Ok(configs)
    }
    
    async fn set_config(&self, key: &str, value: Value) -> Result<()> {
        debug!("Setting config in file: {}", key);
        
        let file_path = format!("{}/{}.json", self.base_path, key);
        
        // Ensure directory exists
        if let Some(parent) = std::path::Path::new(&file_path).parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                ConfigurationError::Internal {
                    message: format!("Failed to create directory: {}", e),
                }
            })?;
        }
        
        let content = serde_json::to_string_pretty(&value)?;
        tokio::fs::write(&file_path, content).await.map_err(|e| {
            ConfigurationError::Internal {
                message: format!("Failed to write file {}: {}", file_path, e),
            }
        })?;
        
        Ok(())
    }
    
    async fn delete_config(&self, key: &str) -> Result<()> {
        debug!("Deleting config file: {}", key);
        
        let file_path = format!("{}/{}.json", self.base_path, key);
        
        tokio::fs::remove_file(&file_path).await.map_err(|e| {
            ConfigurationError::Internal {
                message: format!("Failed to delete file {}: {}", file_path, e),
            }
        })?;
        
        Ok(())
    }
    
    async fn list_configs(&self, prefix: &str) -> Result<Vec<String>> {
        debug!("Listing config files with prefix: {}", prefix);
        
        let mut keys = Vec::new();
        
        match tokio::fs::read_dir(&self.base_path).await {
            Ok(mut entries) => {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    if let Ok(file_name) = entry.file_name().into_string() {
                        if file_name.starts_with(prefix) && file_name.ends_with(".json") {
                            let key = file_name.trim_end_matches(".json");
                            keys.push(key.to_string());
                        }
                    }
                }
            }
            Err(e) => {
                return Err(ConfigurationError::Internal {
                    message: format!("Failed to read directory {}: {}", self.base_path, e),
                });
            }
        }
        
        Ok(keys)
    }
    
    fn supports_writes(&self) -> bool {
        true
    }
    
    fn get_source_name(&self) -> &str {
        "file"
    }
}

// AWS Parameter Store Source
pub struct ParameterStoreSource {
    client: aws_sdk_ssm::Client,
    prefix: String,
}

impl ParameterStoreSource {
    pub async fn new(prefix: String) -> Result<Self> {
        debug!("Initializing AWS Parameter Store source with prefix: {}", prefix);
        
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let client = aws_sdk_ssm::Client::new(&config);
        
        Ok(Self { client, prefix })
    }
}

#[async_trait]
impl ConfigSource for ParameterStoreSource {
    async fn load_config(&self, key: &str) -> Result<Value> {
        debug!("Loading config from Parameter Store: {}", key);
        
        let param_name = format!("{}/{}", self.prefix, key);
        
        let result = self
            .client
            .get_parameter()
            .name(&param_name)
            .with_decryption(true)
            .send()
            .await;
        
        match result {
            Ok(response) => {
                if let Some(parameter) = response.parameter() {
                    let value = parameter.value().unwrap_or_default();
                    
                    // Try to parse as JSON first, fallback to string
                    if let Ok(json_value) = serde_json::from_str::<Value>(value) {
                        Ok(json_value)
                    } else {
                        Ok(Value::String(value.to_string()))
                    }
                } else {
                    Err(ConfigurationError::ConfigNotFound {
                        key: key.to_string(),
                    })
                }
            }
            Err(e) => {
                error!("Failed to get parameter {}: {}", param_name, e);
                Err(ConfigurationError::ParameterStoreError {
                    message: e.to_string(),
                })
            }
        }
    }
    
    async fn get_all_configs(&self, prefix: &str) -> Result<HashMap<String, Value>> {
        debug!("Getting all configs from Parameter Store with prefix: {}", prefix);
        
        let mut configs = HashMap::new();
        let search_prefix = format!("{}/{}", self.prefix, prefix);
        
        let result = self
            .client
            .get_parameters_by_path()
            .path(&search_prefix)
            .recursive(true)
            .with_decryption(true)
            .send()
            .await;
        
        match result {
            Ok(response) => {
                for parameter in response.parameters() {
                    if let (Some(name), Some(value)) = (parameter.name(), parameter.value()) {
                        // Extract the key from the full parameter name
                        if let Some(key) = name.strip_prefix(&format!("{}/", self.prefix)) {
                            // Try to parse as JSON first, fallback to string
                            if let Ok(json_value) = serde_json::from_str::<Value>(value) {
                                configs.insert(key.to_string(), json_value);
                            } else {
                                configs.insert(key.to_string(), Value::String(value.to_string()));
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to get parameters by path {}: {}", search_prefix, e);
                return Err(ConfigurationError::ParameterStoreError {
                    message: e.to_string(),
                });
            }
        }
        
        Ok(configs)
    }
    
    async fn set_config(&self, key: &str, value: Value) -> Result<()> {
        debug!("Setting config in Parameter Store: {}", key);
        
        let param_name = format!("{}/{}", self.prefix, key);
        let value_string = serde_json::to_string(&value)?;
        
        let result = self
            .client
            .put_parameter()
            .name(&param_name)
            .value(&value_string)
            .overwrite(true)
            .send()
            .await;
        
        match result {
            Ok(_) => Ok(()),
            Err(e) => {
                error!("Failed to put parameter {}: {}", param_name, e);
                Err(ConfigurationError::ParameterStoreError {
                    message: e.to_string(),
                })
            }
        }
    }
    
    async fn delete_config(&self, key: &str) -> Result<()> {
        debug!("Deleting config from Parameter Store: {}", key);
        
        let param_name = format!("{}/{}", self.prefix, key);
        
        let result = self
            .client
            .delete_parameter()
            .name(&param_name)
            .send()
            .await;
        
        match result {
            Ok(_) => Ok(()),
            Err(e) => {
                error!("Failed to delete parameter {}: {}", param_name, e);
                Err(ConfigurationError::ParameterStoreError {
                    message: e.to_string(),
                })
            }
        }
    }
    
    async fn list_configs(&self, prefix: &str) -> Result<Vec<String>> {
        debug!("Listing configs from Parameter Store with prefix: {}", prefix);
        
        let mut keys = Vec::new();
        let search_prefix = format!("{}/{}", self.prefix, prefix);
        
        let result = self
            .client
            .get_parameters_by_path()
            .path(&search_prefix)
            .recursive(true)
            .send()
            .await;
        
        match result {
            Ok(response) => {
                for parameter in response.parameters() {
                    if let Some(name) = parameter.name() {
                        // Extract the key from the full parameter name
                        if let Some(key) = name.strip_prefix(&format!("{}/", self.prefix)) {
                            keys.push(key.to_string());
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to list parameters by path {}: {}", search_prefix, e);
                return Err(ConfigurationError::ParameterStoreError {
                    message: e.to_string(),
                });
            }
        }
        
        Ok(keys)
    }
    
    fn supports_writes(&self) -> bool {
        true
    }
    
    fn get_source_name(&self) -> &str {
        "parameter_store"
    }
} 