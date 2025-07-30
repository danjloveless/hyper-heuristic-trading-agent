use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Health checker
#[derive(Debug)]
pub struct HealthChecker {
    last_successful_collection: Option<DateTime<Utc>>,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            last_successful_collection: None,
        }
    }
    
    pub async fn get_health_status(&self) -> HealthStatus {
        HealthStatus {
            alpha_vantage_reachable: true, // Would check actual connectivity
            database_connected: true,      // Would check database connection
            rate_limit_healthy: true,      // Would check rate limit status
            last_successful_collection: self.last_successful_collection.unwrap_or(Utc::now()),
            current_error_rate: 0.0,       // Would calculate actual error rate
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub alpha_vantage_reachable: bool,
    pub database_connected: bool,
    pub rate_limit_healthy: bool,
    pub last_successful_collection: DateTime<Utc>,
    pub current_error_rate: f32,
} 