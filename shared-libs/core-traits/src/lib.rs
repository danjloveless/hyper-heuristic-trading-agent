use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ================================================================================================
// STANDARDIZED HEALTH CHECK INTERFACE
// ================================================================================================

#[async_trait]
pub trait HealthCheckable: Send + Sync {
    async fn health_check(&self) -> HealthStatus;
    async fn ready_check(&self) -> ReadinessStatus;
    fn service_name(&self) -> &str;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub service_name: String,
    pub status: ServiceStatus,
    pub timestamp: DateTime<Utc>,
    pub checks: HashMap<String, ComponentHealth>,
    pub uptime_seconds: u64,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: ServiceStatus,
    pub message: Option<String>,
    pub last_check: DateTime<Utc>,
    pub response_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReadinessStatus {
    Ready,
    NotReady { reason: String },
}

impl HealthStatus {
    pub fn healthy(service_name: String) -> Self {
        Self {
            service_name,
            status: ServiceStatus::Healthy,
            timestamp: Utc::now(),
            checks: HashMap::new(),
            uptime_seconds: 0,
            version: "1.0.0".to_string(),
        }
    }
    
    pub fn add_component_check(&mut self, name: String, health: ComponentHealth) {
        if health.status != ServiceStatus::Healthy {
            self.status = match (self.status.clone(), health.status.clone()) {
                (ServiceStatus::Healthy, ServiceStatus::Degraded) => ServiceStatus::Degraded,
                (_, ServiceStatus::Unhealthy) => ServiceStatus::Unhealthy,
                (current, _) => current,
            };
        }
        self.checks.insert(name, health);
    }
    
    pub fn is_healthy(&self) -> bool {
        self.status == ServiceStatus::Healthy
    }
}

// ================================================================================================
// STANDARDIZED ERROR HANDLING INTERFACE
// ================================================================================================

use std::time::Duration;

#[async_trait]
pub trait ErrorHandler: Send + Sync {
    async fn handle_error(&self, error: &(dyn std::error::Error + Send + Sync), context: &ErrorContext) -> ErrorDecision;
    async fn classify_error(&self, error: &(dyn std::error::Error + Send + Sync)) -> ErrorClassification;
    async fn should_retry(&self, error: &(dyn std::error::Error + Send + Sync), attempt: u32) -> bool;
    async fn report_error(&self, error: &(dyn std::error::Error + Send + Sync), context: &ErrorContext);
}

#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub service_name: String,
    pub operation: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(service_name: String, operation: String) -> Self {
        Self {
            service_name,
            operation,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

#[derive(Debug, Clone)]
pub enum ErrorDecision {
    Retry { delay: Duration, max_attempts: u32 },
    UseCache,
    UseDefault,
    Fail,
}

#[derive(Debug, Clone)]
pub struct ErrorClassification {
    pub error_type: ErrorType,
    pub severity: ErrorSeverity,
    pub retryable: bool,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorType {
    Transient,
    Permanent,
    System,
    Business,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

// ================================================================================================
// SERVICE RESULT TYPES
// ================================================================================================

pub type ServiceResult<T> = Result<T, ServiceError>;

#[derive(thiserror::Error, Debug, Clone, Serialize, Deserialize)]
pub enum ServiceError {
    #[error("Configuration error: {message}")]
    Configuration { message: String },
    
    #[error("Database error: {message}")]
    Database { message: String, retryable: bool },
    
    #[error("External API error: {api} - {message}")]
    ExternalApi { api: String, message: String, status_code: Option<u16> },
    
    #[error("Rate limit exceeded: {service}")]
    RateLimit { service: String, retry_after: Option<Duration> },
    
    #[error("Data quality error: {message}")]
    DataQuality { message: String, quality_score: u8 },
    
    #[error("System error: {message}")]
    System { message: String },
}

// ================================================================================================
// CONFIGURATION INTERFACE
// ================================================================================================

#[async_trait]
pub trait ConfigurationProvider: Send + Sync {
    async fn get_string(&self, key: &str) -> ServiceResult<String>;
    async fn get_u32(&self, key: &str) -> ServiceResult<u32>;
    async fn get_u64(&self, key: &str) -> ServiceResult<u64>;
    async fn get_bool(&self, key: &str) -> ServiceResult<bool>;
    async fn get_secret(&self, key: &str) -> ServiceResult<String>;
    
    // Remove generic method to make trait dyn compatible
    // Use specific methods instead
    async fn get_alpha_vantage_config(&self) -> ServiceResult<serde_json::Value>;
    async fn get_rate_limits_config(&self) -> ServiceResult<serde_json::Value>;
    async fn get_collection_config(&self) -> ServiceResult<serde_json::Value>;
}

// ================================================================================================
// MONITORING INTERFACE
// ================================================================================================

#[async_trait]
pub trait MonitoringProvider: Send + Sync {
    async fn record_metric(&self, name: &str, value: f64, tags: &[(&str, &str)]);
    async fn record_counter(&self, name: &str, tags: &[(&str, &str)]);
    async fn record_timing(&self, name: &str, duration: Duration, tags: &[(&str, &str)]);
    async fn log_info(&self, message: &str, context: &HashMap<String, String>);
    async fn log_warn(&self, message: &str, context: &HashMap<String, String>);
    async fn log_error(&self, message: &str, context: &HashMap<String, String>);
} 