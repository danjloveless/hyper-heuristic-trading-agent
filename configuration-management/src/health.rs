use crate::errors::ConfigurationError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{error, info};

pub type Result<T> = std::result::Result<T, ConfigurationError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: HealthState,
    pub timestamp: DateTime<Utc>,
    pub checks: HashMap<String, HealthCheck>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthState {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthState,
    pub message: Option<String>,
    pub last_check: DateTime<Utc>,
    pub response_time_ms: u64,
}

pub struct HealthChecker {
    checks: HashMap<String, Box<dyn HealthCheckable>>,
}

#[async_trait::async_trait]
pub trait HealthCheckable: Send + Sync {
    async fn check(&self) -> Result<HealthCheck>;
    fn name(&self) -> &str;
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            checks: HashMap::new(),
        }
    }
    
    pub fn add_check(&mut self, check: Box<dyn HealthCheckable>) {
        let name = check.name().to_string();
        self.checks.insert(name, check);
    }
    
    pub async fn run_health_check(&self) -> HealthStatus {
        let start_time = Utc::now();
        let mut checks = HashMap::new();
        let mut overall_status = HealthState::Healthy;
        let mut messages = Vec::new();
        
        for (name, check) in &self.checks {
            let check_start = Utc::now();
            match check.check().await {
                Ok(health_check) => {
                    let response_time = (Utc::now() - check_start).num_milliseconds() as u64;
                    let mut check_with_time = health_check;
                    check_with_time.response_time_ms = response_time;
                    check_with_time.last_check = Utc::now();
                    
                    checks.insert(name.clone(), check_with_time.clone());
                    
                    match check_with_time.status {
                        HealthState::Unhealthy => {
                            overall_status = HealthState::Unhealthy;
                            messages.push(format!("{}: {}", name, check_with_time.message.unwrap_or_default()));
                        }
                        HealthState::Degraded => {
                            if overall_status != HealthState::Unhealthy {
                                overall_status = HealthState::Degraded;
                            }
                            messages.push(format!("{}: {}", name, check_with_time.message.unwrap_or_default()));
                        }
                        HealthState::Healthy => {}
                    }
                }
                Err(e) => {
                    error!("Health check failed for {}: {}", name, e);
                    overall_status = HealthState::Unhealthy;
                    messages.push(format!("{}: Health check failed - {}", name, e));
                    
                    checks.insert(name.clone(), HealthCheck {
                        name: name.clone(),
                        status: HealthState::Unhealthy,
                        message: Some(format!("Health check failed: {}", e)),
                        last_check: Utc::now(),
                        response_time_ms: (Utc::now() - check_start).num_milliseconds() as u64,
                    });
                }
            }
        }
        
        let response_time = (Utc::now() - start_time).num_milliseconds() as u64;
        info!("Health check completed in {}ms with status: {:?}", response_time, overall_status);
        
        HealthStatus {
            status: overall_status,
            timestamp: Utc::now(),
            checks,
            message: if messages.is_empty() {
                None
            } else {
                Some(messages.join("; "))
            },
        }
    }
}

// Configuration-specific health checks
pub struct ConfigStoreHealthCheck {
    config_store: crate::config_store::ConfigStore,
}

impl ConfigStoreHealthCheck {
    pub fn new(config_store: crate::config_store::ConfigStore) -> Self {
        Self { config_store }
    }
}

#[async_trait::async_trait]
impl HealthCheckable for ConfigStoreHealthCheck {
    async fn check(&self) -> Result<HealthCheck> {
        let start_time = Utc::now();
        
        // Try to load a test configuration
        match self.config_store.load_config("health_check").await {
            Ok(_) => {
                let response_time = (Utc::now() - start_time).num_milliseconds() as u64;
                Ok(HealthCheck {
                    name: "config_store".to_string(),
                    status: HealthState::Healthy,
                    message: Some("Configuration store is accessible".to_string()),
                    last_check: Utc::now(),
                    response_time_ms: response_time,
                })
            }
            Err(ConfigurationError::ConfigNotFound { .. }) => {
                // Config not found is acceptable for health check
                let response_time = (Utc::now() - start_time).num_milliseconds() as u64;
                Ok(HealthCheck {
                    name: "config_store".to_string(),
                    status: HealthState::Healthy,
                    message: Some("Configuration store is accessible (test config not found)".to_string()),
                    last_check: Utc::now(),
                    response_time_ms: response_time,
                })
            }
            Err(e) => {
                let response_time = (Utc::now() - start_time).num_milliseconds() as u64;
                Ok(HealthCheck {
                    name: "config_store".to_string(),
                    status: HealthState::Unhealthy,
                    message: Some(format!("Configuration store error: {}", e)),
                    last_check: Utc::now(),
                    response_time_ms: response_time,
                })
            }
        }
    }
    
    fn name(&self) -> &str {
        "config_store"
    }
}

pub struct SecretManagerHealthCheck {
    secret_manager: crate::secret_manager::SecretManager,
}

impl SecretManagerHealthCheck {
    pub fn new(secret_manager: crate::secret_manager::SecretManager) -> Self {
        Self { secret_manager }
    }
}

#[async_trait::async_trait]
impl HealthCheckable for SecretManagerHealthCheck {
    async fn check(&self) -> Result<HealthCheck> {
        let start_time = Utc::now();
        
        // Try to access a test secret
        let mut secret_manager = self.secret_manager.clone();
        match secret_manager.get_secret("health_check_test").await {
            Ok(_) => {
                let response_time = (Utc::now() - start_time).num_milliseconds() as u64;
                Ok(HealthCheck {
                    name: "secret_manager".to_string(),
                    status: HealthState::Healthy,
                    message: Some("Secret manager is accessible".to_string()),
                    last_check: Utc::now(),
                    response_time_ms: response_time,
                })
            }
            Err(ConfigurationError::SecretNotFound { .. }) => {
                // Secret not found is acceptable for health check
                let response_time = (Utc::now() - start_time).num_milliseconds() as u64;
                Ok(HealthCheck {
                    name: "secret_manager".to_string(),
                    status: HealthState::Healthy,
                    message: Some("Secret manager is accessible (test secret not found)".to_string()),
                    last_check: Utc::now(),
                    response_time_ms: response_time,
                })
            }
            Err(e) => {
                let response_time = (Utc::now() - start_time).num_milliseconds() as u64;
                Ok(HealthCheck {
                    name: "secret_manager".to_string(),
                    status: HealthState::Unhealthy,
                    message: Some(format!("Secret manager error: {}", e)),
                    last_check: Utc::now(),
                    response_time_ms: response_time,
                })
            }
        }
    }
    
    fn name(&self) -> &str {
        "secret_manager"
    }
}

pub struct CacheHealthCheck {
    cache: dashmap::DashMap<String, serde_json::Value>,
}

impl CacheHealthCheck {
    pub fn new(cache: dashmap::DashMap<String, serde_json::Value>) -> Self {
        Self { cache }
    }
}

#[async_trait::async_trait]
impl HealthCheckable for CacheHealthCheck {
    async fn check(&self) -> Result<HealthCheck> {
        let start_time = Utc::now();
        
        // Test cache operations
        let test_key = "health_check_test";
        let test_value = serde_json::json!("test_value");
        
        // Insert test value
        self.cache.insert(test_key.to_string(), test_value.clone());
        
        // Retrieve test value
        match self.cache.get(test_key) {
            Some(value) => {
                if value.value() == &test_value {
                    // Clean up
                    self.cache.remove(test_key);
                    
                    let response_time = (Utc::now() - start_time).num_milliseconds() as u64;
                    Ok(HealthCheck {
                        name: "cache".to_string(),
                        status: HealthState::Healthy,
                        message: Some("Cache is operational".to_string()),
                        last_check: Utc::now(),
                        response_time_ms: response_time,
                    })
                } else {
                    let response_time = (Utc::now() - start_time).num_milliseconds() as u64;
                    Ok(HealthCheck {
                        name: "cache".to_string(),
                        status: HealthState::Degraded,
                        message: Some("Cache data integrity issue".to_string()),
                        last_check: Utc::now(),
                        response_time_ms: response_time,
                    })
                }
            }
            None => {
                let response_time = (Utc::now() - start_time).num_milliseconds() as u64;
                Ok(HealthCheck {
                    name: "cache".to_string(),
                    status: HealthState::Unhealthy,
                    message: Some("Cache read operation failed".to_string()),
                    last_check: Utc::now(),
                    response_time_ms: response_time,
                })
            }
        }
    }
    
    fn name(&self) -> &str {
        "cache"
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
} 