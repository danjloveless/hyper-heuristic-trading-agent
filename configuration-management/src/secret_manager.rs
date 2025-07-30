use crate::errors::ConfigurationError;
use aws_config::BehaviorVersion;
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, error, info};

pub type Result<T> = std::result::Result<T, ConfigurationError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretMetadata {
    pub name: String,
    pub description: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_modified: chrono::DateTime<chrono::Utc>,
    pub version_id: String,
    pub tags: HashMap<String, String>,
}

#[derive(Clone)]
pub struct SecretManager {
    client: SecretsManagerClient,
    cache: std::collections::HashMap<String, (String, chrono::DateTime<chrono::Utc>)>,
    cache_ttl: std::time::Duration,
}

impl SecretManager {
    pub async fn new() -> Result<Self> {
        info!("Initializing AWS Secrets Manager client");
        
        let config = aws_config::defaults(BehaviorVersion::latest())
            .load()
            .await;
        
        let client = SecretsManagerClient::new(&config);
        
        Ok(Self {
            client,
            cache: HashMap::new(),
            cache_ttl: std::time::Duration::from_secs(300), // 5 minutes
        })
    }
    
    pub async fn get_secret(&mut self, secret_name: &str) -> Result<String> {
        debug!("Getting secret: {}", secret_name);
        
        // Check environment variables first (for development/local use)
        if let Ok(env_value) = std::env::var(secret_name) {
            debug!("Found secret in environment variable: {}", secret_name);
            return Ok(env_value);
        }
        
        // Check cache first
        if let Some((cached_value, cached_time)) = self.cache.get(secret_name) {
            let age = chrono::Utc::now() - *cached_time;
            if age < chrono::Duration::from_std(self.cache_ttl).unwrap() {
                debug!("Returning cached secret for: {}", secret_name);
                return Ok(cached_value.clone());
            }
        }
        
        // Fetch from AWS Secrets Manager
        let result = self
            .client
            .get_secret_value()
            .secret_id(secret_name)
            .send()
            .await;
        
        match result {
            Ok(response) => {
                let secret_string = response
                    .secret_string()
                    .ok_or_else(|| {
                        ConfigurationError::SecretNotFound {
                            secret_name: secret_name.to_string(),
                        }
                    })?
                    .to_string();
                
                // Cache the result
                self.cache.insert(
                    secret_name.to_string(),
                    (secret_string.clone(), chrono::Utc::now()),
                );
                
                debug!("Successfully retrieved secret: {}", secret_name);
                Ok(secret_string)
            }
            Err(e) => {
                error!("Failed to get secret {}: {}", secret_name, e);
                Err(ConfigurationError::SecretsManagerError {
                    message: e.to_string(),
                })
            }
        }
    }
    
    pub async fn set_secret(&self, secret_name: &str, value: &str) -> Result<()> {
        debug!("Setting secret: {}", secret_name);
        
        let result = self
            .client
            .put_secret_value()
            .secret_id(secret_name)
            .secret_string(value)
            .send()
            .await;
        
        match result {
            Ok(_) => {
                info!("Successfully set secret: {}", secret_name);
                Ok(())
            }
            Err(e) => {
                error!("Failed to set secret {}: {}", secret_name, e);
                Err(ConfigurationError::SecretsManagerError {
                    message: e.to_string(),
                })
            }
        }
    }
    
    pub async fn create_secret(&self, secret_name: &str, value: &str, description: Option<&str>) -> Result<()> {
        debug!("Creating secret: {}", secret_name);
        
        let mut request = self
            .client
            .create_secret()
            .name(secret_name)
            .secret_string(value);
        
        if let Some(desc) = description {
            request = request.description(desc);
        }
        
        let result = request.send().await;
        
        match result {
            Ok(_) => {
                info!("Successfully created secret: {}", secret_name);
                Ok(())
            }
            Err(e) => {
                error!("Failed to create secret {}: {}", secret_name, e);
                Err(ConfigurationError::SecretsManagerError {
                    message: e.to_string(),
                })
            }
        }
    }
    
    pub async fn rotate_secret(&self, secret_name: &str) -> Result<String> {
        debug!("Rotating secret: {}", secret_name);
        
        // Generate a new secret value
        let new_value = self.generate_secret_value().await?;
        
        // Update the secret
        self.set_secret(secret_name, &new_value).await?;
        
        info!("Successfully rotated secret: {}", secret_name);
        Ok(new_value)
    }
    
    pub async fn delete_secret(&self, secret_name: &str, force_delete: bool) -> Result<()> {
        debug!("Deleting secret: {}", secret_name);
        
        let mut request = self
            .client
            .delete_secret()
            .secret_id(secret_name);
        
        if force_delete {
            request = request.force_delete_without_recovery(true);
        }
        
        let result = request.send().await;
        
        match result {
            Ok(_) => {
                info!("Successfully deleted secret: {}", secret_name);
                Ok(())
            }
            Err(e) => {
                error!("Failed to delete secret {}: {}", secret_name, e);
                Err(ConfigurationError::SecretsManagerError {
                    message: e.to_string(),
                })
            }
        }
    }
    
    pub async fn list_secrets(&self, prefix: Option<&str>) -> Result<Vec<SecretMetadata>> {
        debug!("Listing secrets with prefix: {:?}", prefix);
        
        let mut request = self.client.list_secrets();
        
        if let Some(prefix) = prefix {
            let filter = aws_sdk_secretsmanager::types::Filter::builder()
                .key(aws_sdk_secretsmanager::types::FilterNameStringType::Name)
                .values(prefix)
                .build();
            request = request.filters(filter);
        }
        
        let result = request.send().await;
        
        match result {
            Ok(response) => {
                let mut secrets = Vec::new();
                
                for secret in response.secret_list() {
                    let metadata = SecretMetadata {
                        name: secret.name().unwrap_or_default().to_string(),
                        description: secret.description().map(|s| s.to_string()),
                        created_at: secret.created_date()
                            .map(|dt| chrono::DateTime::from_timestamp(dt.secs(), dt.subsec_nanos()))
                            .unwrap_or_default()
                            .unwrap_or_default(),
                        last_modified: secret.last_rotated_date()
                            .map(|dt| chrono::DateTime::from_timestamp(dt.secs(), dt.subsec_nanos()))
                            .unwrap_or_default()
                            .unwrap_or_default(),
                        version_id: "latest".to_string(),
                        tags: secret
                            .tags()
                            .iter()
                            .map(|tag| {
                                (
                                    tag.key().unwrap_or_default().to_string(),
                                    tag.value().unwrap_or_default().to_string(),
                                )
                            })
                            .collect(),
                    };
                    secrets.push(metadata);
                }
                
                debug!("Found {} secrets", secrets.len());
                Ok(secrets)
            }
            Err(e) => {
                error!("Failed to list secrets: {}", e);
                Err(ConfigurationError::SecretsManagerError {
                    message: e.to_string(),
                })
            }
        }
    }
    
    pub async fn get_secret_metadata(&self, secret_name: &str) -> Result<SecretMetadata> {
        debug!("Getting secret metadata: {}", secret_name);
        
        let result = self
            .client
            .describe_secret()
            .secret_id(secret_name)
            .send()
            .await;
        
        match result {
            Ok(response) => {
                let metadata = SecretMetadata {
                    name: response.name().unwrap_or_default().to_string(),
                    description: response.description().map(|s| s.to_string()),
                    created_at: response.created_date()
                        .map(|dt| chrono::DateTime::from_timestamp(dt.secs(), dt.subsec_nanos()))
                        .unwrap_or_default()
                        .unwrap_or_default(),
                    last_modified: response.last_rotated_date()
                        .map(|dt| chrono::DateTime::from_timestamp(dt.secs(), dt.subsec_nanos()))
                        .unwrap_or_default()
                        .unwrap_or_default(),
                    version_id: "latest".to_string(),
                    tags: response
                        .tags()
                        .iter()
                        .map(|tag| {
                            (
                                tag.key().unwrap_or_default().to_string(),
                                tag.value().unwrap_or_default().to_string(),
                            )
                        })
                        .collect(),
                };
                
                Ok(metadata)
            }
            Err(e) => {
                error!("Failed to get secret metadata {}: {}", secret_name, e);
                Err(ConfigurationError::SecretsManagerError {
                    message: e.to_string(),
                })
            }
        }
    }
    
    async fn generate_secret_value(&self) -> Result<String> {
        // Generate a cryptographically secure random string
        use rand::Rng;
        
        let secret: String = rand::rngs::OsRng
            .sample_iter(&rand::distributions::Alphanumeric)
            .take(32)
            .map(char::from)
            .collect();
        
        Ok(secret)
    }
    
    pub fn clear_cache(&mut self) {
        debug!("Clearing secret cache");
        self.cache.clear();
    }
    
    pub fn set_cache_ttl(&mut self, ttl: std::time::Duration) {
        self.cache_ttl = ttl;
    }
} 