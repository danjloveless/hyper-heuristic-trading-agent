use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use shared_types::ServiceStatus;

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthStatus {
    pub service: String,
    pub status: ServiceStatus,
    pub timestamp: DateTime<Utc>,
    pub checks: Vec<HealthCheck>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: ServiceStatus,
    pub latency_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatabaseHealthStatus {
    pub clickhouse_healthy: bool,
    pub redis_healthy: bool,
    pub schema_version: u32,
    pub connection_count: u32,
    pub last_successful_query: DateTime<Utc>,
}

impl HealthStatus {
    pub fn new(service: String) -> Self {
        Self {
            service,
            status: ServiceStatus::Healthy,
            timestamp: Utc::now(),
            checks: Vec::new(),
        }
    }
    
    pub fn add_check(&mut self, check: HealthCheck) {
        if matches!(check.status, ServiceStatus::Unhealthy) {
            self.status = ServiceStatus::Unhealthy;
        } else if matches!(check.status, ServiceStatus::Degraded) && matches!(self.status, ServiceStatus::Healthy) {
            self.status = ServiceStatus::Degraded;
        }
        self.checks.push(check);
    }
}